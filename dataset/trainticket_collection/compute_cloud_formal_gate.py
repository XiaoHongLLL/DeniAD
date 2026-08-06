#!/usr/bin/env python3
"""Apply frozen Tencent Cloud drift thresholds to Expected/Unexpected formal runs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--status", required=True)
    parser.add_argument("--run-root", default="artifacts/trainticket_runs")
    parser.add_argument("--threshold-file", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    repo = Path.cwd()
    scripts = repo / "dataset/trainticket_collection"
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    import compute_drift_gate as gate
    from common import write_csv
    from create_analysis_manifest_from_status import build_row

    status_rows = [row for row in read_csv(Path(args.status)) if row.get("status") == "complete"]
    run_root = Path(args.run_root)
    thresholds_payload = json.loads(Path(args.threshold_file).read_text(encoding="utf-8-sig"))
    thresholds = thresholds_payload["thresholds"]
    strong_thresholds = thresholds_payload["strong_thresholds"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = [build_row(run_root, row, mark_protocol_development=False) for row in status_rows]
    write_csv(output_dir / "cloud_formal_analysis_manifest.csv", manifest_rows)

    all_window_rows = []
    summary_rows = []
    for row in manifest_rows:
        run_rows = gate.metric_rows_for_run(row, run_root, window_seconds=300.0, step_seconds=60.0)
        gate.annotate_gate_rows(run_rows, thresholds, strong_thresholds)
        for item in run_rows:
            full = dict(row)
            full.update(item)
            full["threshold_source"] = thresholds_payload.get("threshold_source")
            full["threshold_file_sha256"] = thresholds_payload.get("threshold_file_sha256", "")
            all_window_rows.append(full)

        gate_result = gate.gate_for_rows(
            run_rows,
            sustain_windows=int(thresholds_payload.get("sustain_windows") or 3),
            min_families=int(thresholds_payload.get("min_families_per_window") or 2),
            strong_sustain_windows=int(thresholds_payload.get("strong_sustain_windows") or 5),
            min_strong_families=int(thresholds_payload.get("min_strong_families_per_window") or 1),
        )
        pass_gate = gate_result["drift_gate_pass"]
        confirmed_at = gate.row_window_end(run_rows, gate_result["confirmed_pos"])
        start_time = gate.evaluation_start(row)
        delay_seconds = ""
        if confirmed_at is not None and start_time is not None:
            delay_seconds = max(0.0, confirmed_at - start_time)
        summary = dict(row)
        summary["drift_gate_pass"] = int(pass_gate)
        summary["drift_gate_version"] = thresholds_payload.get("gate_version") or gate.GATE_VERSION
        summary["threshold_source"] = thresholds_payload.get("threshold_source")
        summary["drift_gate_first_window"] = gate.row_window_index(run_rows, gate_result["sustained_start_pos"])
        summary["first_exceed_window"] = gate.row_window_index(run_rows, gate_result["first_exceed_pos"])
        summary["sustained_gate_start_window"] = gate.row_window_index(run_rows, gate_result["sustained_start_pos"])
        summary["gate_confirmed_window"] = gate.row_window_index(run_rows, gate_result["confirmed_pos"])
        summary["gate_mode"] = gate_result["gate_mode"]
        summary["gate_confirmed_at"] = "" if confirmed_at is None else confirmed_at
        summary["detection_delay_seconds"] = delay_seconds
        summary["drift_windows_evaluated"] = len(run_rows)
        summary["reference_evaluation_overlap_event_count"] = gate.baseline_overlap_event_count(row, run_root)
        summary["benchmark_label"] = gate.benchmark_label(row, pass_gate)
        summary_rows.append(summary)

    write_csv(output_dir / "cloud_formal_drift_gate_windows.csv", all_window_rows)
    write_csv(output_dir / "cloud_formal_drift_gate_summary.csv", summary_rows)
    label_counts = Counter(row.get("benchmark_label") for row in summary_rows)
    semantic_gate = Counter(f"{row.get('semantic_label')}|gate={row.get('drift_gate_pass')}" for row in summary_rows)
    report = {
        "status": str(args.status),
        "threshold_file": str(args.threshold_file),
        "threshold_source": thresholds_payload.get("threshold_source"),
        "threshold_cloud_freeze_id": thresholds_payload.get("cloud_freeze_id"),
        "threshold_cluster_type": thresholds_payload.get("cluster_type"),
        "runs": len(summary_rows),
        "window_rows": len(all_window_rows),
        "benchmark_label_counts": dict(label_counts),
        "semantic_gate_counts": dict(semantic_gate),
        "summary_csv": str(output_dir / "cloud_formal_drift_gate_summary.csv"),
        "windows_csv": str(output_dir / "cloud_formal_drift_gate_windows.csv"),
    }
    write_json(output_dir / "cloud_formal_gate_report.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
