#!/usr/bin/env python3
"""Create an expanded cloud model dataset manifest by appending v0.4 runs.

The v0.4 rows are already collected and quality-audited. This script follows
the same model-facing convention used by the combined67 manifest: the frozen
plan/status target benchmark labels are materialized for model evaluation,
while semantic labels, quality evidence, and source generation are preserved.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

from common import write_csv, write_json


MODEL_DATASET_MANIFEST_VERSION = "cloud-model-dataset-manifest-v0.4-expanded100"

TIMELINE_FIELDS = (
    "run_start_time",
    "run_end_time",
    "pre_change_start",
    "pre_start_time",
    "change_start_time",
    "deployment_complete_time",
    "stable_state_start_time",
    "post_change_observation_start_time",
    "failure_trigger_time",
    "deployment_failed_time",
    "observation_end_time",
    "collection_start_time",
    "collection_end_time",
    "cleanup_start_time",
    "cleanup_complete_time",
    "post_change_state",
)


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def find_run_dir(run_id: str, run_roots: list[Path]) -> Path | None:
    for root in run_roots:
        candidate = root / run_id
        if (candidate / "logs.jsonl").exists():
            return candidate
    return None


def target_gate_pass(label: str) -> str:
    return "1" if label in {"expected_drift", "unexpected_drift"} else "0"


def normalize_control_row(row: dict, run_roots: list[Path], order: int, source_file: Path) -> dict:
    run_id = str(row.get("run_id") or "").strip()
    run_dir = find_run_dir(run_id, run_roots)
    manifest = read_json(run_dir / "run_manifest.json") if run_dir else {}
    scenario = row.get("scenario") or manifest.get("scenario") or ""
    benchmark = manifest.get("benchmark_label") or (
        "no_op_control" if scenario == "noop_redeploy" else "baseline_normal"
    )
    semantic = manifest.get("semantic_label") or "expected"

    out = dict(row)
    out.update({
        "split": "train",
        "model_dataset_role": "Cloud Expansion v0.4 reference controls: trusted normal history",
        "model_dataset_source_file": str(source_file),
        "model_dataset_order": str(order),
        "model_dataset_manifest_version": MODEL_DATASET_MANIFEST_VERSION,
        "model_training_policy": "trusted_normal_history_only",
        "model_sequence_policy": "run_id+phase+service_name",
        "combined_source": "cloud_expansion_v0_4_reference_train",
        "combined_formal_generation": "v0.4_quality_audited",
        "run_id": run_id,
        "run_type": manifest.get("run_type") or ("no_op" if benchmark == "no_op_control" else "baseline"),
        "benchmark_label": benchmark,
        "benchmark_role": benchmark,
        "semantic_label": semantic,
        "declared_semantic_label": semantic,
        "expected_semantic": semantic,
        "drift_gate_pass": "0",
        "used_for_protocol_development": "false",
        "system_name": "Train-Ticket",
        "service_subset_version": "TT-Core-16-accesslog-v1",
        "namespace": manifest.get("namespace") or "trainticket-pilot",
        "cluster_type": manifest.get("cluster_type") or "linux_kubeadm_multinode",
        "cloud_cluster_type": manifest.get("cloud_cluster_type") or manifest.get("cluster_type") or "linux_kubeadm_multinode",
        "cloud_protocol_version": manifest.get("cloud_protocol_version") or "linux-cloud-expansion-v0.4-draft",
        "cloud_freeze_id": manifest.get("cloud_freeze_id") or "",
        "cloud_drift_gate_threshold_sha256": manifest.get("cloud_drift_gate_threshold_sha256") or "",
        "protocol_version": manifest.get("protocol_version") or "",
        "parser_version": manifest.get("parser_version") or "",
        "collector_version": manifest.get("collector_version") or "",
        "external_evidence_dir": str(run_dir) if run_dir else "",
        "local_raw_log_available": "true" if run_dir else "false",
        "change_family_id": manifest.get("change_family_id") or ("noop_redeployment" if benchmark == "no_op_control" else "baseline"),
        "implementation_id": manifest.get("implementation_id") or scenario or run_id,
        "component_id": manifest.get("component_id") or ("all" if benchmark == "baseline_normal" else manifest.get("change_target_component_id") or "unknown"),
        "change_target_component_id": manifest.get("change_target_component_id") or manifest.get("component_id") or ("all" if benchmark == "baseline_normal" else "unknown"),
        "affected_component_ids": manifest.get("affected_component_ids") or "",
        "oracle_component_ids": manifest.get("oracle_component_ids") or "",
        "workload_profile_id": manifest.get("workload_profile_id") or "W1_steady_core",
        "workload_seed": manifest.get("workload_seed") or row.get("seed") or "",
        "expected_label_tendency": benchmark,
        "target_label_tendency": benchmark,
        "target_benchmark_label": benchmark,
        "quality_pass": "true",
        "cleanup_pass": "true",
        "parse_errors": "0",
    })
    for key in TIMELINE_FIELDS:
        out[key] = manifest.get(key) or row.get(key) or ""
    if not out.get("pre_start_time"):
        out["pre_start_time"] = out.get("pre_change_start", "")
    return out


def normalize_change_row(
    row: dict,
    split: str,
    run_roots: list[Path],
    order: int,
    source_file: Path,
) -> dict:
    run_id = str(row.get("run_id") or "").strip()
    run_dir = find_run_dir(run_id, run_roots)
    manifest = read_json(run_dir / "run_manifest.json") if run_dir else {}
    semantic = row.get("semantic_label") or manifest.get("semantic_label") or row.get("expected_semantic") or ""
    benchmark = row.get("target_benchmark_label") or row.get("target_label_tendency") or row.get("expected_label_tendency") or ""

    out = dict(row)
    out.update({
        "split": split,
        "model_dataset_role": f"Cloud Expansion v0.4 {split} supplement: quality-audited Expected/Unexpected benchmark",
        "model_dataset_source_file": str(source_file),
        "model_dataset_order": str(order),
        "model_dataset_manifest_version": MODEL_DATASET_MANIFEST_VERSION,
        "model_training_policy": "held_out_reference_or_test",
        "model_sequence_policy": "run_id+phase+service_name",
        "combined_source": f"cloud_expansion_v0_4_{split}",
        "combined_formal_generation": "v0.4_quality_audited_target_label",
        "run_id": run_id,
        "run_type": manifest.get("run_type") or semantic,
        "benchmark_label": benchmark,
        "benchmark_role": "candidate_expected_change" if semantic == "expected" else "candidate_unexpected_failure",
        "semantic_label": semantic,
        "declared_semantic_label": semantic,
        "expected_semantic": semantic,
        "drift_gate_pass": target_gate_pass(benchmark),
        "used_for_protocol_development": "false",
        "system_name": "Train-Ticket",
        "service_subset_version": "TT-Core-16-accesslog-v1",
        "namespace": manifest.get("namespace") or "trainticket-pilot",
        "cluster_type": manifest.get("cluster_type") or row.get("cluster_type") or "linux_kubeadm_multinode",
        "cloud_cluster_type": manifest.get("cloud_cluster_type") or manifest.get("cluster_type") or row.get("cluster_type") or "linux_kubeadm_multinode",
        "cloud_protocol_version": manifest.get("cloud_protocol_version") or row.get("cloud_protocol_version") or "linux-cloud-expansion-v0.4-draft",
        "cloud_freeze_id": manifest.get("cloud_freeze_id") or row.get("cloud_freeze_id") or "",
        "cloud_drift_gate_threshold_sha256": manifest.get("cloud_drift_gate_threshold_sha256") or row.get("threshold_sha256") or "",
        "protocol_version": manifest.get("protocol_version") or "",
        "parser_version": manifest.get("parser_version") or "",
        "collector_version": manifest.get("collector_version") or "",
        "external_evidence_dir": str(run_dir) if run_dir else "",
        "local_raw_log_available": "true" if run_dir else "false",
        "change_family_id": manifest.get("change_family_id") or row.get("change_family_id") or row.get("scenario") or "",
        "implementation_id": manifest.get("implementation_id") or row.get("implementation_id") or row.get("scenario") or run_id,
        "component_id": manifest.get("component_id") or row.get("component_id") or manifest.get("change_target_component_id") or "unknown",
        "change_target_component_id": manifest.get("change_target_component_id") or row.get("change_target_component_id") or manifest.get("component_id") or "unknown",
        "affected_component_ids": manifest.get("affected_component_ids") or row.get("affected_component_ids") or "",
        "oracle_component_ids": manifest.get("oracle_component_ids") or row.get("oracle_component_ids") or "",
        "workload_profile_id": manifest.get("workload_profile_id") or row.get("workload_profile_id") or "W1_steady_core",
        "post_workload_profile_id": manifest.get("post_workload_profile_id") or row.get("post_workload_profile_id") or "",
        "workload_seed": manifest.get("workload_seed") or row.get("seed") or "",
        "quality_pass": "true",
        "cleanup_pass": "true",
        "parse_errors": "0",
    })
    for key in TIMELINE_FIELDS:
        out[key] = manifest.get(key) or row.get(key) or ""
    if not out.get("pre_start_time"):
        out["pre_start_time"] = out.get("pre_change_start", "")
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-manifest", default="results/scwarn_pilot/model_inputs/cloud_expected_unexpected_combined67_v0_3_1/cloud_model_dataset_manifest.csv")
    parser.add_argument("--reference-status", default="results/scwarn_pilot/collections/cloud_expansion_v0_4_20260709/cloud_expansion_v0_4_reference_train_status.csv")
    parser.add_argument("--dev-status", default="results/scwarn_pilot/collections/cloud_expansion_v0_4_20260709/cloud_expansion_v0_4_dev_supplement_status.csv")
    parser.add_argument("--test-status", default="results/scwarn_pilot/collections/cloud_expansion_v0_4_20260709/cloud_expansion_v0_4_test_supplement_status.csv")
    parser.add_argument(
        "--raw-run-root",
        action="append",
        default=["artifacts/trainticket_runs"],
    )
    parser.add_argument("--out-dir", default="results/scwarn_pilot/model_inputs/cloud_expected_unexpected_expanded100_v0_4")
    args = parser.parse_args()

    base_manifest = Path(args.base_manifest)
    reference_status = Path(args.reference_status)
    dev_status = Path(args.dev_status)
    test_status = Path(args.test_status)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_roots = [Path(item) for item in args.raw_run_root]

    rows = []
    order = 1
    for row in read_csv(base_manifest):
        out = dict(row)
        out["model_dataset_order"] = str(order)
        rows.append(out)
        order += 1

    for row in read_csv(reference_status):
        if row.get("status") == "complete" and row.get("exit_code") == "0":
            rows.append(normalize_control_row(row, run_roots, order, reference_status))
            order += 1

    for row in read_csv(dev_status):
        if row.get("status") == "complete" and row.get("exit_code") == "0":
            rows.append(normalize_change_row(row, "dev", run_roots, order, dev_status))
            order += 1

    for row in read_csv(test_status):
        if row.get("status") == "complete" and row.get("exit_code") == "0":
            rows.append(normalize_change_row(row, "test", run_roots, order, test_status))
            order += 1

    missing_raw = [row for row in rows if row.get("local_raw_log_available") != "true"]
    manifest_path = out_dir / "cloud_model_dataset_manifest.csv"
    drift_summary_path = out_dir / "cloud_model_dataset_drift_summary.csv"
    missing_path = out_dir / "missing_raw_runs.csv"
    write_csv(manifest_path, rows)
    write_csv(drift_summary_path, rows)
    write_csv(missing_path, missing_raw)

    split_counts = Counter(row.get("split") for row in rows)
    split_label_counts = Counter(f"{row.get('split')}:{row.get('benchmark_label')}" for row in rows)
    test_label_counts = Counter(row.get("benchmark_label") for row in rows if row.get("split") == "test")
    semantic_counts = Counter(f"{row.get('split')}:{row.get('semantic_label')}" for row in rows)
    audit = {
        "model_dataset_manifest_version": MODEL_DATASET_MANIFEST_VERSION,
        "total_rows": len(rows),
        "missing_raw_run_count": len(missing_raw),
        "split_counts": dict(split_counts),
        "split_benchmark_label_counts": dict(split_label_counts),
        "formal_test_benchmark_label_counts": dict(test_label_counts),
        "split_semantic_counts": dict(semantic_counts),
        "source_base_manifest": str(base_manifest),
        "source_reference_status": str(reference_status),
        "source_dev_status": str(dev_status),
        "source_test_status": str(test_status),
        "manifest": str(manifest_path),
        "drift_summary": str(drift_summary_path),
        "missing_raw_runs": str(missing_path),
    }
    write_json(out_dir / "cloud_model_dataset_audit_report.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    return 0 if not missing_raw else 2


if __name__ == "__main__":
    raise SystemExit(main())
