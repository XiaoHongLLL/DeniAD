#!/usr/bin/env python3
"""Run a deterministic allowlisted TT-Core workload and record Oracle calls."""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib import error, request

sys.path.insert(0, str(Path(__file__).resolve().parent))
from validate_core_closure import load_simple_yaml  # noqa: E402


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def http_call(base_url: str, op: dict, timeout: int) -> dict:
    url = base_url.rstrip("/") + str(op["path"])
    method = str(op.get("method") or "GET").upper()
    req = request.Request(url, method=method)
    started_at = utc_now()
    started = time.perf_counter()
    body = ""
    try:
        with request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read(4096)
            body = raw.decode("utf-8", errors="replace")
            status = resp.status
            transport_ok = True
            error_text = ""
    except error.HTTPError as exc:
        raw = exc.read(4096)
        body = raw.decode("utf-8", errors="replace")
        status = exc.code
        transport_ok = False
        error_text = str(exc)
    except Exception as exc:  # noqa: BLE001 - Oracle records external failures.
        status = None
        transport_ok = False
        error_text = str(exc)
    elapsed_ms = (time.perf_counter() - started) * 1000
    completed_at = utc_now()
    expected_status = int(op.get("expected_status", 200))
    expected_contains = str(op.get("expected_contains") or "")
    assertion_ok = status == expected_status and (not expected_contains or expected_contains in body)
    return {
        "timestamp": started_at,
        "timestamp_semantics": "request_start",
        "completed_at": completed_at,
        "operation_id": op.get("id"),
        "operation": op.get("operation"),
        "method": method,
        "url": url,
        "status": status,
        "elapsed_ms": elapsed_ms,
        "transport_ok": transport_ok,
        "assertion_ok": assertion_ok,
        "expected_status": expected_status,
        "expected_contains": expected_contains,
        "body_prefix": body[:240],
        "error": error_text,
        "latency_slo_usable": False,
        "latency_note": "Port-forward workload timings are functional diagnostics only.",
    }


def append_jsonl(path: Path, item: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def load_profile(path: Path, profile_id: str | None) -> dict | None:
    if not profile_id:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    for profile in payload.get("profiles", []):
        if profile.get("id") == profile_id:
            profile = dict(profile)
            profile["profile_file"] = str(path)
            profile["schema_version"] = payload.get("schema_version")
            return profile
    raise SystemExit(f"workload profile not found: {profile_id}")


def operation_plan(operations: list[dict], profile: dict | None) -> tuple[list[dict], dict[str, dict]]:
    by_id = {op.get("id"): op for op in operations}
    if not profile:
        return operations, by_id
    mode = profile.get("mode")
    if mode == "round_robin":
        selected = []
        for op_id in profile.get("operation_ids", []):
            if op_id not in by_id:
                raise SystemExit(f"profile references disabled or missing operation: {op_id}")
            selected.append(by_id[op_id])
        return selected, by_id
    if mode in {"weighted_round_robin", "weighted_random"}:
        selected = []
        for op_id, weight in (profile.get("operation_weights") or {}).items():
            if op_id not in by_id:
                raise SystemExit(f"profile references disabled or missing operation: {op_id}")
            selected.extend([by_id[op_id]] * max(0, int(weight)))
        if not selected:
            raise SystemExit(f"profile has no selected operations: {profile.get('id')}")
        return selected, by_id
    if mode == "markov":
        for state, data in (profile.get("states") or {}).items():
            op_id = data.get("operation_id")
            if op_id not in by_id:
                raise SystemExit(f"profile state {state} references disabled or missing operation: {op_id}")
        return operations, by_id
    raise SystemExit(f"unsupported workload profile mode: {mode}")


def pick_markov(profile: dict, by_id: dict[str, dict], state: str, rng: random.Random) -> tuple[dict, str]:
    states = profile.get("states") or {}
    if state not in states:
        state = profile.get("start_state") or next(iter(states))
    current = states[state]
    op = by_id[current["operation_id"]]
    transitions = current.get("transitions") or {}
    if transitions:
        names = list(transitions.keys())
        weights = [float(transitions[name]) for name in names]
        state = rng.choices(names, weights=weights, k=1)[0]
    return op, state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allowlist", default="configs/trainticket_collection/workload_allowlist.yaml")
    parser.add_argument("--profile-file", default="configs/trainticket_collection/workload_profiles.json")
    parser.add_argument("--profile-id", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--duration-seconds", type=int, default=120)
    parser.add_argument("--rate-per-second", type=float, default=1.0)
    parser.add_argument("--timeout-seconds", type=int, default=8)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", default=None)
    args = parser.parse_args()

    allowlist = load_simple_yaml(Path(args.allowlist))
    base_url = args.base_url or str(allowlist.get("api_base") or "http://localhost:18888")
    operations = [op for op in allowlist.get("operations", []) if op.get("enabled", True)]
    if not operations:
        raise SystemExit("No enabled operations in workload allowlist")
    profile = load_profile(Path(args.profile_file), args.profile_id)
    plan, by_id = operation_plan(operations, profile)
    rng = random.Random(args.seed)
    markov_state = profile.get("start_state") if profile and profile.get("mode") == "markov" else None

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    summary_output = Path(args.summary_output) if args.summary_output else output.parent / "workload_profile.json"

    interval = 1.0 / max(args.rate_per_second, 0.001)
    started = time.monotonic()
    next_at = started
    total = 0
    failed = 0
    index = 0
    while time.monotonic() - started < args.duration_seconds:
        now = time.monotonic()
        if now < next_at:
            time.sleep(min(0.25, next_at - now))
            continue
        if profile and profile.get("mode") == "weighted_random":
            op = rng.choice(plan)
        elif profile and profile.get("mode") == "markov":
            op, markov_state = pick_markov(profile, by_id, str(markov_state), rng)
        else:
            op = plan[index % len(plan)]
        result = http_call(base_url, op, args.timeout_seconds)
        result["sequence_index"] = index
        result["workload_profile_id"] = profile.get("id") if profile else "legacy_round_robin_all_enabled"
        result["workload_seed"] = args.seed
        append_jsonl(output, result)
        total += 1
        failed += 0 if result["assertion_ok"] else 1
        index += 1
        next_at += interval

    summary = {
        "output": str(output),
        "allowlist": str(Path(args.allowlist)),
        "profile_file": str(Path(args.profile_file)) if profile else None,
        "workload_profile_id": profile.get("id") if profile else "legacy_round_robin_all_enabled",
        "workload_profile": profile,
        "workload_seed": args.seed,
        "operations": len(plan),
        "selected_operation_ids": [op.get("id") for op in plan],
        "calls": total,
        "failed_assertions": failed,
        "duration_seconds": args.duration_seconds,
        "rate_per_second": args.rate_per_second,
        "base_url": base_url,
    }
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_output"] = str(summary_output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
