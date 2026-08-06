#!/usr/bin/env python3
"""Classify run semantics from external oracle and Kubernetes facts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--oracle", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--expected_semantic", choices=["expected", "unexpected", "indeterminate"], default=None)
    parser.add_argument("--unexpected_failure_ratio", type=float, default=0.05)
    parser.add_argument(
        "--indeterminate_failure_ratio_min",
        type=float,
        default=0.0,
        help=(
            "Optional lower failure-ratio bound for pre-registered boundary runs. "
            "Only used when --expected_semantic=indeterminate."
        ),
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    oracle_path = Path(args.oracle) if args.oracle else run_dir / "oracle_calls.jsonl"
    output = Path(args.output) if args.output else run_dir / "semantic_label_report.json"
    oracle = read_jsonl(oracle_path)
    stable = read_json(run_dir / "post_change_stable_report.json") or read_json(run_dir / "stable_state_report.json")
    collection = read_json(run_dir / "collection_summary.json")

    request_count = len(oracle)
    failure_count = sum(1 for item in oracle if not item.get("assertion_ok"))
    failure_ratio = failure_count / request_count if request_count else 1.0
    infra_stable = (
        stable.get("deployment_success") is True
        and stable.get("readiness_success") is True
        and stable.get("service_registered") is True
    )

    stable_state = str(stable.get("post_change_state") or "").lower()

    boundary_failure = (
        args.expected_semantic == "indeterminate"
        and failure_ratio >= args.indeterminate_failure_ratio_min
        and failure_ratio < args.unexpected_failure_ratio
        and failure_count > 0
    )

    if stable_state == "deployment_failure":
        semantic_label = "unexpected"
        post_change_state = "deployment_failure"
    elif not request_count:
        semantic_label = "indeterminate"
        post_change_state = "indeterminate"
    elif failure_ratio >= args.unexpected_failure_ratio:
        semantic_label = "unexpected"
        post_change_state = "runtime_failure"
    elif boundary_failure:
        semantic_label = "indeterminate"
        post_change_state = "stable_degradation"
    elif not infra_stable:
        semantic_label = "indeterminate"
        post_change_state = stable_state or "indeterminate"
    else:
        semantic_label = "expected"
        post_change_state = "stable_success"

    expectation_matched = args.expected_semantic is None or semantic_label == args.expected_semantic
    payload = {
        "run_dir": str(run_dir),
        "oracle": str(oracle_path),
        "semantic_label": semantic_label,
        "expected_semantic": args.expected_semantic,
        "expectation_matched": expectation_matched,
        "post_change_state": post_change_state,
        "request_count": request_count,
        "assertion_failure_count": failure_count,
        "assertion_failure_ratio": failure_ratio,
        "unexpected_failure_ratio_threshold": args.unexpected_failure_ratio,
        "indeterminate_failure_ratio_min": args.indeterminate_failure_ratio_min,
        "infra_stable": infra_stable,
        "collector_complete": collection.get("collector_complete"),
        "notes": "Semantic label is derived only from external oracle and Kubernetes facts.",
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "semantic_label": semantic_label,
        "post_change_state": post_change_state,
        "request_count": request_count,
        "assertion_failure_count": failure_count,
        "expectation_matched": expectation_matched,
        "output": str(output),
    }, ensure_ascii=False, indent=2))
    return 0 if expectation_matched else 2


if __name__ == "__main__":
    raise SystemExit(main())
