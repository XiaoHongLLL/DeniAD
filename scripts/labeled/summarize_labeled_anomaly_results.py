#!/usr/bin/env python3
"""Summarize labeled anomaly benchmark CSV files."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ANOMALY_FIELDS = [
    "Calib_Size",
    "Gamma_Anomaly",
    "Gamma_Segment_Anomaly",
    "Anomaly_Score_Mode",
    "Anomaly_Quantile",
    "Type_Score_Weight",
    "Time_Score_Weight",
    "Profile_Score_Weight",
    "Profile_Unigram_Weight",
    "Profile_Bigram_Weight",
    "Profile_Hist_Signature_Weight",
    "Trace_Profile_Enabled",
    "Detected_Dataset",
    "Segment_Score_Mode",
    "Segment_TopK",
    "Event_TP",
    "Event_FP",
    "Event_FN",
    "Event_TN",
    "Event_Precision",
    "Event_Recall",
    "Event_F1",
    "Event_FPR",
    "Event_AUPRC",
    "Event_AUROC",
    "Segment_TP",
    "Segment_FP",
    "Segment_FN",
    "Segment_TN",
    "Segment_Precision",
    "Segment_Recall",
    "Segment_F1",
    "Segment_FPR",
    "Segment_AUPRC",
    "Segment_AUROC",
    "Point_Alert_Rate",
    "Point_Alerts",
    "Segment_Alerts",
    "Segment_BestF1_Oracle",
    "Segment_BestF1_Threshold",
    "Segment_BestF1_Precision",
    "Segment_BestF1_Recall",
    "Segment_BestF1_FPR",
    "Segment_BestF1_Alerts",
    "Segment_BestF1_Alert_Rate",
]

RELIABILITY_FIELDS = [
    "Anomaly_Score_Mode",
    "Anomaly_Quantile",
    "Type_Score_Weight",
    "Time_Score_Weight",
    "Profile_Score_Weight",
    "Profile_Unigram_Weight",
    "Profile_Bigram_Weight",
    "Profile_Hist_Signature_Weight",
    "Trace_Profile_Enabled",
    "Detected_Dataset",
    "Segment_Score_Mode",
    "Segment_TopK",
    "Traditional_TP",
    "Traditional_FP",
    "Traditional_FN",
    "Traditional_TN",
    "Traditional_Precision",
    "Traditional_Recall",
    "Traditional_F1",
    "Traditional_FPR",
    "UA_TP",
    "UA_FP",
    "UA_FN",
    "UA_TN",
    "UA_Precision",
    "UA_Recall",
    "UA_F1",
    "UA_FPR",
    "TraditionalSegment_TP",
    "TraditionalSegment_FP",
    "TraditionalSegment_FN",
    "TraditionalSegment_TN",
    "TraditionalSegment_Precision",
    "TraditionalSegment_Recall",
    "TraditionalSegment_F1",
    "TraditionalSegment_FPR",
    "UASegment_TP",
    "UASegment_FP",
    "UASegment_FN",
    "UASegment_TN",
    "UASegment_Precision",
    "UASegment_Recall",
    "UASegment_F1",
    "UASegment_FPR",
    "OOD_Candidate_Events",
    "OOD_Candidate_Segments",
    "OODAllAnomaly_TP",
    "OODAllAnomaly_FP",
    "OODAllAnomaly_FN",
    "OODAllAnomaly_TN",
    "OODAllAnomaly_Precision",
    "OODAllAnomaly_Recall",
    "OODAllAnomaly_F1",
    "OODAllAnomaly_FPR",
    "OODAllNormal_TP",
    "OODAllNormal_FP",
    "OODAllNormal_FN",
    "OODAllNormal_TN",
    "OODAllNormal_Precision",
    "OODAllNormal_Recall",
    "OODAllNormal_F1",
    "OODAllNormal_FPR",
    "OursOOD_TP",
    "OursOOD_FP",
    "OursOOD_FN",
    "OursOOD_TN",
    "OursOOD_Precision",
    "OursOOD_Recall",
    "OursOOD_F1",
    "OursOOD_FPR",
    "OODAllAnomalySegment_TP",
    "OODAllAnomalySegment_FP",
    "OODAllAnomalySegment_FN",
    "OODAllAnomalySegment_TN",
    "OODAllAnomalySegment_Precision",
    "OODAllAnomalySegment_Recall",
    "OODAllAnomalySegment_F1",
    "OODAllAnomalySegment_FPR",
    "OODAllNormalSegment_TP",
    "OODAllNormalSegment_FP",
    "OODAllNormalSegment_FN",
    "OODAllNormalSegment_TN",
    "OODAllNormalSegment_Precision",
    "OODAllNormalSegment_Recall",
    "OODAllNormalSegment_F1",
    "OODAllNormalSegment_FPR",
    "OursOODSegment_TP",
    "OursOODSegment_FP",
    "OursOODSegment_FN",
    "OursOODSegment_TN",
    "OursOODSegment_Precision",
    "OursOODSegment_Recall",
    "OursOODSegment_F1",
    "OursOODSegment_FPR",
    "Point_Alert_Reduction",
    "Expected_Drift_Events",
    "Rejected_Uncertain_Events",
    "Ensemble_Candidates",
    "Ensemble_Correction_Rate",
    "Drift_Adapter_Ready",
    "Drift_Adapter_Refits",
]


def read_single_row(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        return {}
    return rows[0]


def infer_dataset(csv_path: Path, result_root: Path):
    parent = csv_path.parent
    if parent != result_root and parent.name:
        return parent.name
    match = re.match(r"([^_]+)_labeled_", csv_path.name)
    if match:
        return match.group(1)
    return parent.name or "unknown"


def collect(result_root: Path):
    rows = []
    for csv_path in sorted(result_root.rglob("*_results.csv")):
        lower = csv_path.name.lower()
        if "_anomaly_results.csv" in lower:
            eval_type = "likelihood_detector"
            fields = ANOMALY_FIELDS
        elif "_reliability_results.csv" in lower:
            eval_type = "ua_reliability"
            fields = RELIABILITY_FIELDS
        else:
            continue

        dataset = infer_dataset(csv_path, result_root)
        source = read_single_row(csv_path)
        row = {
            "Dataset": dataset,
            "EvalType": eval_type,
            "ResultFile": str(csv_path),
            "Events": source.get("Events", ""),
            "Labeled_Events": source.get("Labeled_Events", ""),
            "Labeled_Segments": source.get("Labeled_Segments", ""),
        }
        for field in fields:
            row[field] = source.get(field, "")
        rows.append(row)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="./results/labeled_anomaly")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output = Path(args.output) if args.output else result_root / "summary.csv"
    rows = collect(result_root)
    if not rows:
        print(f"[Warn] No result CSV files found under {result_root}")
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
        dataset = row["Dataset"]
        eval_type = row["EvalType"]
        if eval_type == "likelihood_detector":
            print(
                f"{dataset:14s} likelihood "
                f"EventF1={row.get('Event_F1', '')} "
                f"SegmentF1={row.get('Segment_F1', '')} "
                f"FP={row.get('Segment_FP', '')} "
                f"FN={row.get('Segment_FN', '')}"
            )
        else:
            print(
                f"{dataset:14s} reliability "
                f"TradF1={row.get('Traditional_F1', '')} "
                f"UAF1={row.get('UA_F1', '')} "
                f"Reduction={row.get('Point_Alert_Reduction', '')}"
            )


if __name__ == "__main__":
    main()
