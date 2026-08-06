#!/usr/bin/env python3
"""Summarize which audited workload paths are usable for log-based runs."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def operation_bucket(op: dict) -> str:
    request_count = int(op.get("request_count") or 0)
    new_log_count = int(op.get("new_log_count") or 0)
    zero_ratio = op.get("zero_log_request_ratio")
    zero_ratio = float(zero_ratio) if zero_ratio is not None else 1.0
    if request_count == 0:
        return "not_exercised"
    if new_log_count > 0 and zero_ratio <= 0.2:
        return "log_observable"
    if new_log_count > 0:
        return "partial"
    return "silent"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--audit", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    audit_path = Path(args.audit) if args.audit else run_dir / "log_observability_audit.json"
    output = Path(args.output) if args.output else run_dir / "log_observability_gap_report.json"
    audit = read_json(audit_path)
    overall = audit.get("overall") or {}

    buckets: dict[str, list[dict]] = {
        "log_observable": [],
        "partial": [],
        "silent": [],
        "not_exercised": [],
    }
    for op in audit.get("operations") or []:
        item = {
            "operation_id": op.get("operation_id"),
            "request_count": op.get("request_count"),
            "new_log_count": op.get("new_log_count"),
            "unique_template_count": op.get("unique_template_count"),
            "zero_log_request_ratio": op.get("zero_log_request_ratio"),
            "expected_service_path": op.get("expected_service_path") or [],
            "services_with_new_logs": op.get("services_with_new_logs") or [],
            "missing_expected_services_in_logs": op.get("missing_expected_services_in_logs") or [],
            "unexpected_services_with_logs": op.get("unexpected_services_with_logs") or [],
        }
        buckets[operation_bucket(op)].append(item)

    core_services = set(overall.get("core_api_services") or [])
    services_with_logs = set(overall.get("services_with_logs") or [])
    missing_core_services = sorted(core_services - services_with_logs)

    recommendations = []
    if not overall.get("operation_attribution_reliable", False):
        recommendations.append(
            "rerun W0 with non-overlapping request windows before using operation-service closure claims"
        )
    if buckets["silent"]:
        recommendations.append(
            "exclude silent hello/welcome operations from formal log-drift runs or replace them with deeper business APIs"
        )
    if missing_core_services:
        recommendations.append(
            "add tokenized business workflow or component-specific probe paths for core services that produced no logs"
        )

    payload = {
        "generated_at": utc_now(),
        "run_dir": str(run_dir),
        "audit": str(audit_path),
        "operation_attribution_reliable": bool(overall.get("operation_attribution_reliable", False)),
        "call_window_overlap_count": overall.get("call_window_overlap_count"),
        "call_window_overlap_ratio": overall.get("call_window_overlap_ratio"),
        "core_api_service_log_coverage_ratio": overall.get("core_api_service_log_coverage_ratio"),
        "services_with_logs": sorted(services_with_logs),
        "missing_core_api_services": missing_core_services,
        "buckets": buckets,
        "recommended_formal_operation_ids": [item["operation_id"] for item in buckets["log_observable"]],
        "recommendations": recommendations,
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "operation_attribution_reliable": payload["operation_attribution_reliable"],
        "recommended_formal_operation_ids": payload["recommended_formal_operation_ids"],
        "missing_core_api_services": missing_core_services,
        "recommendations": recommendations,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
