#!/usr/bin/env python3
"""Quality gate for Tencent Cloud Expected/Unexpected formal runs."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--expected-cluster-type", default="linux_kubeadm_multinode")
    parser.add_argument("--cloud-freeze-id", required=True)
    parser.add_argument("--threshold-sha256", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    manifest = read_json(run_dir / "run_manifest.json")
    collection = read_json(run_dir / "collection_summary.json")
    semantic = read_json(run_dir / "semantic_label_report.json")
    before = read_json(run_dir / "cloud_state_before.json")
    after = read_json(run_dir / "cloud_state_after.json")
    cleanup = read_json(run_dir / "cloud_state_cleanup.json")
    pre_stable = read_json(run_dir / "pre_change_stable_report.json")

    expected_semantic = str(manifest.get("semantic_label") or "")
    actual_semantic = str(semantic.get("semantic_label") or "")
    assertion_failures = int(semantic.get("assertion_failure_count") or 0)
    request_count = int(semantic.get("request_count") or 0)
    log_events = int(collection.get("events") or 0)
    parse_errors = int(collection.get("parse_error_count") or 0)
    cluster_type = str(manifest.get("cloud_cluster_type") or manifest.get("cluster_type") or "")
    cloud_freeze_id = str(manifest.get("cloud_freeze_id") or "")
    threshold_sha = str(manifest.get("cloud_drift_gate_threshold_sha256") or "")

    expected_checks = {
        "semantic_label_expected": actual_semantic == "expected",
        "assertion_failure_count_zero": assertion_failures == 0,
        "cloud_state_after_pass": bool(after.get("pass")),
    }
    unexpected_checks = {
        "semantic_label_unexpected": actual_semantic == "unexpected",
        "assertion_failure_count_positive": assertion_failures > 0,
    }
    indeterminate_checks = {
        "semantic_label_indeterminate": actual_semantic == "indeterminate",
        "indeterminate_not_binary_main_label": actual_semantic not in {"expected", "unexpected"},
    }

    checks = {
        "cloud_state_before_pass": bool(before.get("pass")),
        "pre_change_stable_gate_pass": bool(pre_stable.get("deployment_success"))
        and bool(pre_stable.get("readiness_success"))
        and bool(pre_stable.get("service_registered")),
        "archive_complete": bool(collection.get("collector_complete")),
        "logs_nonempty": log_events > 0,
        "parse_errors_zero": parse_errors == 0,
        "semantic_expectation_matched": bool(semantic.get("expectation_matched")),
        "request_count_positive": request_count > 0,
        "cleanup_pass": bool(cleanup.get("pass")),
        "cloud_freeze_id_match": cloud_freeze_id == args.cloud_freeze_id,
        "threshold_sha256_match": threshold_sha == args.threshold_sha256,
        "cluster_type_match": cluster_type == args.expected_cluster_type,
        "nodes_ready_before": bool(before.get("all_nodes_ready")),
        "nodes_ready_after_cleanup": bool(cleanup.get("all_nodes_ready")),
        "pod_node_mapping_recorded": bool(before.get("pod_node_mapping_recorded"))
        and bool(cleanup.get("pod_node_mapping_recorded")),
    }
    if expected_semantic == "expected":
        checks.update(expected_checks)
    elif expected_semantic == "unexpected":
        checks.update(unexpected_checks)
    elif expected_semantic == "indeterminate":
        checks.update(indeterminate_checks)
    else:
        checks["declared_semantic_known"] = False

    payload = {
        "run_dir": str(run_dir),
        "checks": checks,
        "pass": all(checks.values()),
        "expected_semantic": expected_semantic,
        "actual_semantic": actual_semantic,
        "post_change_state": semantic.get("post_change_state"),
        "oracle_request_count": request_count,
        "assertion_failure_count": assertion_failures,
        "assertion_failure_ratio": semantic.get("assertion_failure_ratio"),
        "log_events": log_events,
        "post_change_log_lines": sum(1 for _ in (run_dir / "post_change_logs.jsonl").open("r", encoding="utf-8")) if (run_dir / "post_change_logs.jsonl").exists() else 0,
        "parse_errors": parse_errors,
        "cluster_type": cluster_type,
        "expected_cluster_type": args.expected_cluster_type,
        "cloud_freeze_id": cloud_freeze_id,
        "expected_cloud_freeze_id": args.cloud_freeze_id,
        "cloud_drift_gate_threshold_sha256": threshold_sha,
        "expected_threshold_sha256": args.threshold_sha256,
    }
    output = run_dir / "cloud_formal_quality_report.json"
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "pass": payload["pass"],
        "semantic_label": actual_semantic,
        "assertion_failure_count": assertion_failures,
        "log_events": log_events,
    }, ensure_ascii=False, indent=2))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
