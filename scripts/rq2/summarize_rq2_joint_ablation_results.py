#!/usr/bin/env python3
"""Summarize RQ2 joint type-time anomaly ablation results."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


VARIANT_INFO = {
    "transformer_only": ("0", "0", "none; direct Transformer hidden state"),
    "type_only": ("1", "0", "h + p(m|H)"),
    "time_only": ("0", "1", "h + p(tau|H)"),
    "independent_joint": ("1", "1", "h + p(m|H)+p(tau|H)"),
    "ours_joint": ("1", "1", "h + p(m|H)+p_CFM(tau|H,m)"),
}

FIELDS = [
    "Evaluation_Mode",
    "Classifier_Variant",
    "Classifier_Feature_Source",
    "Classifier_Feature_Dim",
    "Classifier_Head_Architecture",
    "Classifier_Probabilistic_Residual_Weight",
    "Classifier_Best_Dev_F1",
    "Classifier_Best_Dev_FNR",
    "Classifier_Best_Dev_FPR",
    "Classifier_Best_Dev_Balanced_Accuracy",
    "Classifier_Best_Dev_Objective",
    "Classifier_Class_Balance",
    "Classifier_Binary_Loss_Weight",
    "Classifier_Target_Dev_FPR",
    "Classifier_Threshold_Strategy",
    "Classifier_Dev_Max_FPR",
    "Classifier_Event_Threshold",
    "Classifier_Segment_Threshold",
    "Classifier_Segment_Score_Mode",
    "Classifier_Segment_TopK",
    "Classifier_Degenerate_High_Recall_Point",
    "Segment_Precision",
    "Segment_Recall",
    "Segment_F1",
    "Segment_FNR",
    "Segment_FPR",
    "Segment_Balanced_Accuracy",
    "Segment_Predicted_Positive_Rate",
    "Segment_AUPRC",
    "Segment_AUROC",
    "Event_F1",
    "Event_FNR",
    "Event_FPR",
    "Event_Balanced_Accuracy",
    "Event_Predicted_Positive_Rate",
    "Event_AUPRC",
    "Event_AUROC",
    "RQ2_TypeSegment_F1",
    "RQ2_TimeSegment_F1",
    "RQ2_JointSegment_F1",
    "RQ2_TypeSegment_FNR",
    "RQ2_TimeSegment_FNR",
    "RQ2_JointSegment_FNR",
    "RQ2_TypeSegment_FPR",
    "RQ2_TimeSegment_FPR",
    "RQ2_JointSegment_FPR",
    "Segment_Normal_Score_Mean",
    "Segment_Anomaly_Score_Mean",
    "Segment_Normal_Score_P95",
    "Segment_Anomaly_Score_P50",
    "Gamma_Segment_Anomaly",
    "Anomaly_Score_Mode",
    "Type_Score_Weight",
    "Time_Score_Weight",
    "Conditional_Gap_Weight",
]


def read_single_row(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def infer_dataset_and_variant(path: Path, result_root: Path):
    dataset = path.parent.name if path.parent != result_root else "unknown"
    stem = path.name.removesuffix("_results.csv")
    for variant in VARIANT_INFO:
        if stem.endswith(f"_{variant}"):
            return dataset, variant
    match = re.search(
        r"_(transformer_only|type_only|time_only|independent_joint|ours_joint)$",
        stem,
    )
    if match:
        return dataset, match.group(1)
    return dataset, "unknown"


def collect(result_root: Path):
    rows = []
    for csv_path in sorted(result_root.rglob("*_results.csv")):
        lower = csv_path.name.lower()
        if not any(lower.endswith(f"_{variant}_results.csv") for variant in VARIANT_INFO):
            continue
        source = read_single_row(csv_path)
        dataset, variant = infer_dataset_and_variant(csv_path, result_root)
        type_modeling, time_modeling, dependency = VARIANT_INFO.get(variant, ("", "", ""))
        row = {
            "Dataset": dataset,
            "Variant": variant,
            "Type_Modeling": type_modeling,
            "Time_Modeling": time_modeling,
            "Type_Time_Dependency": dependency,
            "ResultFile": str(csv_path),
        }
        for field in FIELDS:
            row[field] = source.get(field, "")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="./results/rq2_joint_ablation")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output = Path(args.output) if args.output else result_root / "summary.csv"
    rows = collect(result_root)
    if not rows:
        print(f"[Warn] No RQ2 ablation result CSV files found under {result_root}")
        return

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {output}")
    for row in rows:
        print(
            f"{row['Dataset']:14s} {row['Variant']:18s} "
            f"F1/FNR/FPR={row.get('Segment_F1', '')}/"
            f"{row.get('Segment_FNR', '')}/"
            f"{row.get('Segment_FPR', '')} "
            f"Type/Time/Joint={row.get('RQ2_TypeSegment_F1', '')}/"
            f"{row.get('RQ2_TimeSegment_F1', '')}/"
            f"{row.get('RQ2_JointSegment_F1', '')}"
        )


if __name__ == "__main__":
    main()
