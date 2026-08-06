#!/usr/bin/env python3
"""Create an analysis manifest from a completed collection_status.csv."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


PREFERRED_FIELDS = [
    "run_id",
    "split",
    "benchmark_role",
    "benchmark_label",
    "semantic_label",
    "declared_semantic_label",
    "scenario_id",
    "run_type",
    "system_name",
    "service_subset_version",
    "namespace",
    "cluster_type",
    "change_family_id",
    "implementation_id",
    "component_id",
    "service",
    "change_target_component_id",
    "affected_component_ids",
    "oracle_component_ids",
    "batch_id",
    "pair_block_id",
    "batch_position",
    "preceding_scenario",
    "matched_workload_seed",
    "workload_profile_id",
    "workload_seed",
    "seed",
    "pre_start_time",
    "change_start_time",
    "deployment_complete_time",
    "stable_state_start_time",
    "post_change_observation_start_time",
    "failure_trigger_time",
    "deployment_failed_time",
    "observation_end_time",
    "post_change_state",
    "baseline_reference_start_time",
    "baseline_reference_end_time",
    "baseline_evaluation_start_time",
    "baseline_evaluation_end_time",
    "external_evidence_dir",
    "collection_summary_path",
    "semantic_label_report_path",
    "analysis_protocol_version",
    "used_for_protocol_development",
]


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def read_status(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    seen = set()
    for field in PREFERRED_FIELDS:
        if field not in seen:
            fields.append(field)
            seen.add(field)
    for row in rows:
        for field in row:
            if field not in seen:
                fields.append(field)
                seen.add(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def merge_nonempty(target: dict, source: dict) -> None:
    for key, value in source.items():
        if value is None:
            continue
        if isinstance(value, (dict, list)):
            continue
        text = str(value)
        if text:
            target[key] = text


def scenario_split(row: dict, mark_protocol_development: bool) -> str:
    if mark_protocol_development:
        return "dev"
    role = str(row.get("benchmark_role") or row.get("benchmark_label") or "").lower()
    if role in {"baseline_normal", "no_op_control"}:
        return "train"
    return "test"


def build_row(run_root: Path, status_row: dict, mark_protocol_development: bool) -> dict:
    run_id = status_row["run_id"]
    run_dir = run_root / run_id
    manifest = read_json(run_dir / "run_manifest.json")
    timeline = read_json(run_dir / "phase_timeline.json")
    semantic = read_json(run_dir / "semantic_label_report.json")
    stable = read_json(run_dir / "post_change_stable_report.json")
    if not stable:
        stable = read_json(run_dir / "stable_state_report.json")
    collection = read_json(run_dir / "collection_summary.json")

    row: dict[str, str] = {}
    merge_nonempty(row, status_row)
    merge_nonempty(row, manifest)
    merge_nonempty(row, timeline)
    merge_nonempty(row, stable)
    merge_nonempty(row, semantic)

    row["run_id"] = run_id
    row["benchmark_role"] = row.get("benchmark_label") or row.get("benchmark_role") or ""
    row["declared_semantic_label"] = row.get("semantic_label") or ""
    row["split"] = scenario_split(row, mark_protocol_development)
    row["pre_start_time"] = row.get("pre_start_time") or row.get("pre_change_start") or row.get("run_start_time") or ""
    row["external_evidence_dir"] = str(run_dir.resolve())
    row["collection_summary_path"] = str((run_dir / "collection_summary.json").resolve())
    row["semantic_label_report_path"] = str((run_dir / "semantic_label_report.json").resolve())
    row["analysis_protocol_version"] = "scwarn-pilot-analysis-v0.6"
    row["used_for_protocol_development"] = "true" if mark_protocol_development else "false"

    if not row.get("observation_end_time"):
        row["observation_end_time"] = row.get("collection_end_time") or row.get("run_end_time") or ""
    if not row.get("post_change_observation_start_time"):
        row["post_change_observation_start_time"] = (
            row.get("stable_state_start_time")
            or row.get("failure_trigger_time")
            or row.get("deployment_failed_time")
            or row.get("change_start_time")
            or ""
        )
    if row.get("benchmark_role") == "baseline_normal":
        row["baseline_reference_start_time"] = (
            row.get("baseline_reference_start_time") or row.get("pre_start_time") or ""
        )
        row["baseline_reference_end_time"] = (
            row.get("baseline_reference_end_time") or row.get("change_start_time") or ""
        )
        row["baseline_evaluation_start_time"] = (
            row.get("baseline_evaluation_start_time")
            or row.get("post_change_observation_start_time")
            or row.get("stable_state_start_time")
            or row.get("change_start_time")
            or ""
        )
        row["baseline_evaluation_end_time"] = (
            row.get("baseline_evaluation_end_time")
            or row.get("observation_end_time")
            or row.get("run_end_time")
            or ""
        )

    row["event_count"] = str(collection.get("events") or "")
    row["collector_complete"] = str(collection.get("collector_complete") or "")
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status_csv", required=True)
    parser.add_argument("--run_root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mark_protocol_development", action="store_true")
    args = parser.parse_args()

    rows = [
        build_row(Path(args.run_root), row, args.mark_protocol_development)
        for row in read_status(Path(args.status_csv))
        if str(row.get("status") or "").lower() == "complete"
    ]
    write_csv(Path(args.output), rows)
    print({
        "status_csv": args.status_csv,
        "output": args.output,
        "rows": len(rows),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
