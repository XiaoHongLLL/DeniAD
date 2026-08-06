#!/usr/bin/env python3
"""Build the three-policy public-dataset FP/FN table from reliability results.

The comparison is binary and segment/session level:

OOD candidates are the samples rejected by the pre-revision selective detector
because their uncertainty exceeds the calibrated threshold. Predictions on
non-OOD samples remain fixed for all three policies.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


DATASET_ORDER = ["hdfs", "bgl", "thunderbird", "spirit", "liberty"]
COUNT_PREFIXES = (
    "OODAllAnomalySegment",
    "OODAllNormalSegment",
    "OursOODSegment",
)


def parse_count(row: dict[str, str], key: str) -> int:
    value = str(row.get(key, "")).strip()
    if not value:
        raise ValueError(
            f"missing {key}; rerun the updated summarizer or reliability evaluation"
        )
    return int(float(value))


def metrics(method: str, dataset: str, tp: int, fp: int, fn: int, tn: int, source: str):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    fnr = fn / (fn + tp) if fn + tp else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    balanced_accuracy = 0.5 * (recall + specificity)
    return {
        "Dataset": dataset,
        "Policy": method,
        "Granularity": "segment/session",
        "Actual_Anomalous": tp + fn,
        "Actual_Normal": fp + tn,
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "TN": tn,
        "Precision": precision,
        "Recall": recall,
        "F1": f1,
        "FPR": fpr,
        "FNR": fnr,
        "Balanced_Accuracy": balanced_accuracy,
        "Total_Errors": fp + fn,
        "Better_Than_Both_F1": "",
        "Source": source,
    }


def load_reliability_rows(path: Path, datasets):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    selected = {}
    for row in rows:
        if row.get("EvalType", "").strip() != "ua_reliability":
            continue
        dataset = row.get("Dataset", "").strip().lower()
        if dataset in datasets:
            selected[dataset] = row
    missing = [dataset for dataset in datasets if dataset not in selected]
    if missing:
        raise ValueError(f"missing ua_reliability rows for: {', '.join(missing)}")
    return selected


def build_rows(selected, datasets=None):
    datasets = datasets or DATASET_ORDER
    output = []
    for dataset in datasets:
        row = selected[dataset]
        counts = {
            prefix: {
                key: parse_count(row, f"{prefix}_{key}")
                for key in ("TP", "FP", "FN", "TN")
            }
            for prefix in COUNT_PREFIXES
        }
        truth_totals = {
            (
                values["TP"] + values["FN"],
                values["FP"] + values["TN"],
            )
            for values in counts.values()
        }
        if len(truth_totals) != 1:
            raise ValueError(f"{dataset}: policy truth totals differ")

        policies = (
            ("Baseline: OOD candidates -> anomaly", "OODAllAnomalySegment"),
            ("Baseline: OOD candidates -> non-anomaly", "OODAllNormalSegment"),
            ("Ours: score-and-memory OOD handling", "OursOODSegment"),
        )
        for label, prefix in policies:
            values = counts[prefix]
            output.append(metrics(
                label,
                dataset,
                tp=values["TP"],
                fp=values["FP"],
                fn=values["FN"],
                tn=values["TN"],
                source=prefix,
            ))
        dataset_rows = output[-3:]
        baseline_best_f1 = max(dataset_rows[0]["F1"], dataset_rows[1]["F1"])
        dataset_rows[2]["Better_Than_Both_F1"] = int(
            dataset_rows[2]["F1"] > baseline_best_f1
        )
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", required=True, help="summary.csv produced after RUN_RELIABILITY=1")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--datasets",
        default=" ".join(DATASET_ORDER),
        help="space-separated dataset names to include",
    )
    args = parser.parse_args()

    datasets = [item.strip().lower() for item in args.datasets.split() if item.strip()]
    unknown = [item for item in datasets if item not in DATASET_ORDER]
    if unknown:
        raise ValueError(f"unknown datasets: {', '.join(unknown)}")
    rows = build_rows(load_reliability_rows(Path(args.summary), datasets), datasets)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"[OK] Wrote {output}")
    print("Dataset\tPolicy\tFP\tFN\tFPR\tFNR\tF1\tErrors\tBetterThanBoth")
    for row in rows:
        print(
            f"{row['Dataset']}\t{row['Policy']}\t{row['FP']}\t{row['FN']}\t"
            f"{100 * row['FPR']:.4f}%\t{100 * row['FNR']:.4f}%\t"
            f"{row['F1']:.6f}\t{row['Total_Errors']}\t{row['Better_Than_Both_F1']}"
        )


if __name__ == "__main__":
    main()
