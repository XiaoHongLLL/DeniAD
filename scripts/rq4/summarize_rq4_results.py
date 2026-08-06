#!/usr/bin/env python3
"""Summarize controlled RQ4 drift-diagnosis result CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def transition_fields(prefix):
    fields = []
    for truth in ("Expected", "Unexpected", "Reject"):
        base = f"{prefix}_{truth}"
        fields.append(f"{base}_True_Count")
        for pred in ("Normal", "Expected", "Unexpected", "Reject"):
            fields.append(f"{base}_to_{pred}_Rate")
    return fields


KEEP_FIELDS = [
    "Dataset",
    "Result_File",
    "RQ4_Candidate_Mode",
    "Component_Drift_Diagnosis",
    "RQ4_Window_Diagnosis",
    "RQ4_Expected_Precision",
    "RQ4_Expected_Recall",
    "RQ4_Expected_F1",
    "RQ4_Unexpected_Precision",
    "RQ4_Unexpected_Recall",
    "RQ4_Unexpected_F1",
    "RQ4_Reject_Precision",
    "RQ4_Reject_Recall",
    "RQ4_Reject_F1",
    "RQ4_EU_Avg_F1",
    "RQ4_Macro_F1",
    "RQ4_Selective_Coverage_EU",
    "RQ4_Selective_Risk_EU",
    "RQ4_Selective_Accuracy_EU",
    "RQ4_Unexpected_False_Acceptance_Rate",
    "RQ4_Expected_False_Alarm_Rate",
    "RQ4_Expected_Review_Burden_Rate",
    "RQ4_Unexpected_Reject_Rate",
    "RQ4_Unexpected_Safe_Rate",
    "RQ4_Risk_Coverage_AURC",
    "RQ4_Risk_Coverage_RiskAt25",
    "RQ4_Risk_Coverage_RiskAt50",
    "RQ4_Risk_Coverage_RiskAt75",
    "RQ4_Forced_Expected_F1",
    "RQ4_Forced_Unexpected_F1",
    "RQ4_Forced_EU_Macro_F1",
    "RQ4_Memory_Contamination",
    "RQ4_IV_Labeled_Events",
    "RQ4_IV_Expected_F1",
    "RQ4_IV_Unexpected_F1",
    "RQ4_IV_EU_Avg_F1",
    "RQ4_IV_Macro_F1",
    "RQ4_IV_Unexpected_False_Acceptance_Rate",
    "RQ4_OOV_Labeled_Events",
    "RQ4_OOV_Expected_F1",
    "RQ4_OOV_Unexpected_F1",
    "RQ4_OOV_EU_Avg_F1",
    "RQ4_OOV_Macro_F1",
    "RQ4_OOV_Unexpected_False_Acceptance_Rate",
    "RQ4_OOVRateBaseline_Expected_F1",
    "RQ4_OOVRateBaseline_Unexpected_F1",
    "RQ4_OOVRateBaseline_EU_Avg_F1",
    "RQ4_OOVRateBaseline_Macro_F1",
    "RQ4_Segment_Expected_Precision",
    "RQ4_Segment_Expected_Recall",
    "RQ4_Segment_Expected_F1",
    "RQ4_Segment_Unexpected_Precision",
    "RQ4_Segment_Unexpected_Recall",
    "RQ4_Segment_Unexpected_F1",
    "RQ4_Segment_Reject_Precision",
    "RQ4_Segment_Reject_Recall",
    "RQ4_Segment_Reject_F1",
    "RQ4_Segment_EU_Avg_F1",
    "RQ4_Segment_Macro_F1",
    "RQ4_Segment_Selective_Coverage_EU",
    "RQ4_Segment_Selective_Risk_EU",
    "RQ4_Segment_Unexpected_False_Acceptance_Rate",
    "RQ4_Segment_Forced_EU_Macro_F1",
    "RQ4_Segment_Memory_Contamination",
    "RQ4_Expected_True_Count",
    "RQ4_Unexpected_True_Count",
    "RQ4_Reject_True_Count",
    "RQ4_Unexpected_to_Normal_Rate",
    "RQ4_Unexpected_to_Expected_Rate",
    "RQ4_Unexpected_to_Unexpected_Rate",
    "RQ4_Unexpected_to_Reject_Rate",
    "RQ4_Segment_Expected_True_Count",
    "RQ4_Segment_Unexpected_True_Count",
    "RQ4_Segment_Reject_True_Count",
    "RQ4_Segment_Unexpected_to_Normal_Rate",
    "RQ4_Segment_Unexpected_to_Expected_Rate",
    "RQ4_Segment_Unexpected_to_Unexpected_Rate",
    "RQ4_Segment_Unexpected_to_Reject_Rate",
    "Traditional_F1",
    "UA_F1",
    "Ensemble_Candidates",
    "Ensemble_Component_Unexpected",
    "Ensemble_Component_Reject",
    "Ensemble_Component_Disagreement",
    "Ensemble_Disagreement_Threshold",
    "Ensemble_Correction_Rate",
    "Drift_Adapter_Refits",
]
KEEP_FIELDS += [
    field
    for field in transition_fields("RQ4") + transition_fields("RQ4_Segment")
    if field not in KEEP_FIELDS
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result_root", default="./results/rq4")
    parser.add_argument("--output", default=None)
    return parser.parse_args()


def infer_dataset(path: Path, result_root: Path) -> str:
    try:
        rel = path.relative_to(result_root)
        if len(rel.parts) > 1:
            return rel.parts[0]
    except ValueError:
        pass

    name = path.name.lower()
    marker = "rq4_controlled_"
    if marker in name:
        tail = name.split(marker, 1)[1]
        return tail.split("_", 1)[0]
    return "unknown"


def to_float(value, default=0.0):
    if value in (None, ""):
        return default
    try:
        return float(value)
    except ValueError:
        return default


def read_result(path: Path, result_root: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return None
    row = rows[0]
    out = {field: row.get(field, "") for field in KEEP_FIELDS}
    out["Dataset"] = infer_dataset(path, result_root)
    out["Result_File"] = str(path)
    return out


def main():
    args = parse_args()
    result_root = Path(args.result_root)
    paths = sorted(result_root.rglob("*_results.csv"))
    rows = []
    for path in paths:
        if "_rq4_events" in path.name:
            continue
        row = read_result(path, result_root)
        if row is None:
            continue
        if not row.get("RQ4_Macro_F1") and not row.get("RQ4_Segment_Macro_F1"):
            continue
        rows.append(row)

    rows.sort(
        key=lambda item: (
            to_float(item.get("RQ4_Segment_Macro_F1")),
            to_float(item.get("RQ4_Macro_F1")),
        ),
        reverse=True,
    )

    output = Path(args.output) if args.output else result_root / "summary_rq4.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=KEEP_FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {output}")
    for row in rows:
        print(
            f"{row['Dataset']:<12} "
            f"SegMacroF1={to_float(row.get('RQ4_Segment_Macro_F1')):.4f} "
            f"EventMacroF1={to_float(row.get('RQ4_Macro_F1')):.4f} "
            f"EUAvgF1={to_float(row.get('RQ4_EU_Avg_F1')):.4f} "
            f"Coverage={to_float(row.get('RQ4_Selective_Coverage_EU')):.4f} "
            f"AURC={to_float(row.get('RQ4_Risk_Coverage_AURC')):.4f} "
            f"RejectSegF1={to_float(row.get('RQ4_Segment_Reject_F1')):.4f} "
            f"MemoryContam={to_float(row.get('RQ4_Memory_Contamination')):.4f}"
        )


if __name__ == "__main__":
    main()
