#!/usr/bin/env python3
"""Audit whether a run's workload naturally produces enough run-bounded logs."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

from validate_core_closure import load_simple_yaml


NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
UUID_RE = re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I)
HEX_RE = re.compile(r"\b0x[0-9a-f]+\b", re.I)
ISO_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(?:[.,]\d+)?(?:Z|[+-]\d{2}:?\d{2})?\b")
IP_RE = re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b")
SPACE_RE = re.compile(r"\s+")


def parse_rfc3339(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, tail = text.split(".", 1)
        offset = ""
        fraction = tail
        for marker in ("+", "-"):
            if marker in tail:
                fraction, offset_part = tail.split(marker, 1)
                offset = marker + offset_part
                break
        if len(fraction) > 6:
            fraction = fraction[:6]
        text = f"{head}.{fraction}{offset}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def rfc3339_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def call_started_at(call: dict) -> datetime | None:
    started = parse_rfc3339(call.get("request_started_at") or call.get("timestamp"))
    if not started:
        return None
    if call.get("timestamp_semantics") == "request_start" or call.get("completed_at"):
        return started
    elapsed_ms = float(call.get("elapsed_ms") or 0.0)
    return started - timedelta(milliseconds=elapsed_ms)


def call_completed_at(call: dict) -> datetime | None:
    completed = parse_rfc3339(call.get("completed_at"))
    if completed:
        return completed
    started = call_started_at(call)
    if not started:
        return None
    elapsed_ms = float(call.get("elapsed_ms") or 0.0)
    return started + timedelta(milliseconds=elapsed_ms)


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: Path) -> list[dict]:
    items: list[dict] = []
    if not path.exists():
        return items
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return items


def normalize_template(message: str) -> str:
    text = ISO_TS_RE.sub("<TS>", message)
    text = UUID_RE.sub("<UUID>", text)
    text = IP_RE.sub("<IP>", text)
    text = HEX_RE.sub("<HEX>", text)
    text = NUMBER_RE.sub("<NUM>", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text[:500]


def percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    frac = rank - low
    return ordered[low] * (1 - frac) + ordered[high] * frac


def run_bounds(
    run_dir: Path,
    calls: list[dict],
    logs: list[tuple[datetime, dict]],
    start_key: str | None = None,
    end_key: str | None = None,
) -> tuple[datetime | None, datetime | None]:
    timeline = read_json(run_dir / "phase_timeline.json")
    manifest = read_json(run_dir / "run_manifest.json")
    summary = read_json(run_dir / "collection_summary.json")
    if start_key:
        start = parse_rfc3339(timeline.get(start_key) or manifest.get(start_key) or summary.get(start_key))
    else:
        start = (
            parse_rfc3339(timeline.get("run_start_time"))
            or parse_rfc3339(manifest.get("run_start_time"))
            or parse_rfc3339(summary.get("run_start_time"))
        )
    if end_key:
        end = parse_rfc3339(timeline.get(end_key) or manifest.get(end_key) or summary.get(end_key))
    else:
        end = (
            parse_rfc3339(timeline.get("run_end_time"))
            or parse_rfc3339(timeline.get("collection_end_time"))
            or parse_rfc3339(manifest.get("run_end_time"))
            or parse_rfc3339(summary.get("run_end_time"))
        )
    if not start:
        call_times = [call_started_at(call) for call in calls]
        log_times = [ts for ts, _ in logs]
        times = [ts for ts in call_times if ts] + log_times
        start = min(times) if times else None
    if not end:
        call_ends = []
        for call in calls:
            ts = call_completed_at(call)
            if ts:
                call_ends.append(ts)
        log_times = [ts for ts, _ in logs]
        times = call_ends + log_times
        end = max(times) if times else None
    return start, end


def call_windows(calls: list[dict], margin_seconds: float) -> list[dict]:
    windows = []
    for call in calls:
        started = call_started_at(call)
        if not started:
            continue
        completed = call_completed_at(call) or started
        end = completed + timedelta(seconds=margin_seconds)
        windows.append({
            "operation_id": call.get("operation_id"),
            "start": started,
            "end": end,
        })
    windows.sort(key=lambda item: item["start"])
    return windows


def overlap_summary(windows: list[dict]) -> dict:
    if len(windows) < 2:
        return {
            "call_window_overlap_count": 0,
            "call_window_overlap_ratio": 0.0,
            "operation_attribution_reliable": True,
        }
    overlap_count = 0
    previous_end = windows[0]["end"]
    for window in windows[1:]:
        if window["start"] < previous_end:
            overlap_count += 1
        previous_end = max(previous_end, window["end"])
    return {
        "call_window_overlap_count": overlap_count,
        "call_window_overlap_ratio": overlap_count / max(1, len(windows) - 1),
        "operation_attribution_reliable": overlap_count == 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--allowlist", default="configs/trainticket_collection/workload_allowlist.yaml")
    parser.add_argument("--oracle", default=None)
    parser.add_argument("--logs", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--post_call_margin_seconds", type=float, default=2.0)
    parser.add_argument("--window_seconds", type=int, default=300)
    parser.add_argument("--window_start_key", default=None)
    parser.add_argument("--window_end_key", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    oracle_path = Path(args.oracle) if args.oracle else run_dir / "oracle_calls.jsonl"
    logs_path = Path(args.logs) if args.logs else run_dir / "logs.jsonl"
    output = Path(args.output) if args.output else run_dir / "log_observability_audit.json"
    if not oracle_path.exists():
        raise SystemExit(f"Missing oracle file: {oracle_path}")
    if not logs_path.exists():
        raise SystemExit(f"Missing logs file: {logs_path}")

    allowlist = load_simple_yaml(Path(args.allowlist))
    op_meta = {
        op.get("id"): op
        for op in allowlist.get("operations", [])
        if op.get("enabled", True)
    }
    core_services = set(allowlist.get("core_api_services") or [])

    calls = read_jsonl(oracle_path)
    logs: list[tuple[datetime, dict]] = []
    log_source_counts: Counter[str] = Counter()
    for event in read_jsonl(logs_path):
        ts = parse_rfc3339(event.get("timestamp_utc") or event.get("timestamp"))
        if ts:
            logs.append((ts, event))
            log_source_counts[str(event.get("log_source") or "unknown")] += 1
    logs.sort(key=lambda item: item[0])
    attribution = overlap_summary(call_windows(calls, args.post_call_margin_seconds))

    operation_summary: dict[str, dict] = {}
    all_services_with_logs: set[str] = set()
    all_templates: set[str] = set()

    for call in calls:
        op_id = call.get("operation_id")
        started = call_started_at(call)
        if not op_id or not started:
            continue
        completed = call_completed_at(call) or started
        end = completed + timedelta(seconds=args.post_call_margin_seconds)
        expected_services = set(op_meta.get(op_id, {}).get("expected_services") or [])

        services = defaultdict(int)
        templates = set()
        first_log_delay = None
        for ts, event in logs:
            if ts < started:
                continue
            if ts > end:
                break
            service = event.get("service") or "unknown"
            template = normalize_template(str(event.get("message") or ""))
            services[service] += 1
            templates.add(template)
            all_services_with_logs.add(service)
            all_templates.add(template)
            delay = (ts - started).total_seconds()
            first_log_delay = delay if first_log_delay is None else min(first_log_delay, delay)

        item = operation_summary.setdefault(op_id, {
            "operation_id": op_id,
            "operation": op_meta.get(op_id, {}).get("operation"),
            "expected_service_path": sorted(expected_services),
            "request_count": 0,
            "assertion_pass_count": 0,
            "assertion_failure_count": 0,
            "services_with_new_logs": set(),
            "new_log_count": 0,
            "unique_templates": set(),
            "zero_log_request_count": 0,
            "first_log_delays_seconds": [],
            "call_windows": [],
        })
        item["request_count"] += 1
        if call.get("assertion_ok"):
            item["assertion_pass_count"] += 1
        else:
            item["assertion_failure_count"] += 1
        new_log_count = sum(services.values())
        item["new_log_count"] += new_log_count
        if new_log_count == 0:
            item["zero_log_request_count"] += 1
        for service in services:
            item["services_with_new_logs"].add(service)
        item["unique_templates"].update(templates)
        if first_log_delay is not None:
            item["first_log_delays_seconds"].append(first_log_delay)
        item["call_windows"].append({
            "start": rfc3339_utc(started),
            "end": rfc3339_utc(end),
            "new_log_count": new_log_count,
            "services": dict(sorted(services.items())),
            "first_log_delay_seconds": first_log_delay,
        })

    serializable_ops = []
    for item in operation_summary.values():
        expected_services = set(item["expected_service_path"])
        services_with_logs = set(item["services_with_new_logs"])
        request_count = int(item["request_count"])
        delays = list(item["first_log_delays_seconds"])
        op = dict(item)
        op["services_with_new_logs"] = sorted(services_with_logs)
        op["expected_services_with_logs"] = sorted(expected_services & services_with_logs)
        op["missing_expected_services_in_logs"] = sorted(expected_services - services_with_logs)
        op["unexpected_services_with_logs"] = sorted(services_with_logs - expected_services)
        op["unique_template_count"] = len(op.pop("unique_templates"))
        op["zero_log_request_ratio"] = (op["zero_log_request_count"] / request_count) if request_count else None
        op["first_log_delay_seconds_min"] = min(delays) if delays else None
        op["first_log_delay_seconds_median"] = statistics.median(delays) if delays else None
        op["first_log_delay_seconds_p95"] = percentile(delays, 0.95)
        op.pop("first_log_delays_seconds", None)
        serializable_ops.append(op)
    serializable_ops.sort(key=lambda item: item["operation_id"])

    start, end = run_bounds(run_dir, calls, logs, args.window_start_key, args.window_end_key)
    windows = []
    zero_log_windows = 0
    complete_windows = 0
    if start and end and args.window_seconds > 0:
        cursor = start
        while cursor < end:
            window_end = min(cursor + timedelta(seconds=args.window_seconds), end)
            is_complete = (window_end - cursor).total_seconds() >= args.window_seconds
            count = sum(1 for ts, _ in logs if cursor <= ts < window_end)
            services = sorted({event.get("service") or "unknown" for ts, event in logs if cursor <= ts < window_end})
            if is_complete:
                complete_windows += 1
                if count == 0:
                    zero_log_windows += 1
            windows.append({
                "start": rfc3339_utc(cursor),
                "end": rfc3339_utc(window_end),
                "complete": is_complete,
                "log_count": count,
                "services_with_logs": services,
            })
            cursor = window_end

    sliding_windows = []
    sliding_complete = 0
    sliding_zero = 0
    if start and end and args.window_seconds > 0:
        stride = timedelta(seconds=60)
        cursor = start
        while cursor + timedelta(seconds=args.window_seconds) <= end:
            window_end = cursor + timedelta(seconds=args.window_seconds)
            count = sum(1 for ts, _ in logs if cursor <= ts < window_end)
            services = sorted({event.get("service") or "unknown" for ts, event in logs if cursor <= ts < window_end})
            sliding_complete += 1
            if count == 0:
                sliding_zero += 1
            sliding_windows.append({
                "start": rfc3339_utc(cursor),
                "end": rfc3339_utc(window_end),
                "complete": True,
                "stride_seconds": 60,
                "log_count": count,
                "services_with_logs": services,
            })
            cursor += stride

    operations_with_logs = [op["operation_id"] for op in serializable_ops if op["new_log_count"] > 0]
    core_services_with_logs = sorted(core_services & all_services_with_logs)
    payload = {
        "generated_at": rfc3339_utc(datetime.now(timezone.utc)),
        "run_dir": str(run_dir),
        "oracle": str(oracle_path),
        "logs": str(logs_path),
        "post_call_margin_seconds": args.post_call_margin_seconds,
        "window_seconds": args.window_seconds,
        "run_start_time": rfc3339_utc(start) if start else None,
        "run_end_time": rfc3339_utc(end) if end else None,
        "window_start_key": args.window_start_key,
        "window_end_key": args.window_end_key,
        "overall": {
            "request_count": len(calls),
            "assertion_pass_count": sum(1 for call in calls if call.get("assertion_ok")),
            "assertion_failure_count": sum(1 for call in calls if not call.get("assertion_ok")),
            "operation_count": len(serializable_ops),
            "operations_with_logs_count": len(operations_with_logs),
            "operations_with_logs": operations_with_logs,
            "new_log_count": len(logs),
            "log_source_counts": dict(sorted(log_source_counts.items())),
            "unique_template_count": len(all_templates),
            "services_with_logs": sorted(all_services_with_logs),
            "core_api_services": sorted(core_services),
            "core_api_services_with_logs": core_services_with_logs,
            "core_api_service_log_coverage_ratio": (
                len(core_services_with_logs) / len(core_services) if core_services else None
            ),
            "complete_log_windows": complete_windows,
            "complete_nonoverlap_5min_windows": complete_windows,
            "zero_log_windows": zero_log_windows,
            "zero_log_window_ratio": zero_log_windows / complete_windows if complete_windows else None,
            "complete_sliding_5min_windows_stride_1min": sliding_complete,
            "zero_sliding_5min_windows_stride_1min": sliding_zero,
            "zero_sliding_window_ratio": sliding_zero / sliding_complete if sliding_complete else None,
            **attribution,
            "operation_log_attribution_method": "timestamp_window_without_request_correlation",
        },
        "operations": serializable_ops,
        "windows": windows,
        "sliding_windows": sliding_windows,
    }

    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "request_count": payload["overall"]["request_count"],
        "new_log_count": payload["overall"]["new_log_count"],
        "operations_with_logs_count": payload["overall"]["operations_with_logs_count"],
        "core_api_service_log_coverage_ratio": payload["overall"]["core_api_service_log_coverage_ratio"],
        "complete_log_windows": complete_windows,
        "complete_sliding_5min_windows_stride_1min": sliding_complete,
        "zero_log_window_ratio": payload["overall"]["zero_log_window_ratio"],
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
