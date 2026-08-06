#!/usr/bin/env python3
"""Build the RQ3 core component ablation table from run-level summaries."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


DEFAULT_VARIANTS = [
    (
        "Global-score detector",
        "Frozen global anomaly score plus uncertainty; low-risk changed runs are Expected, high-score runs are Unexpected, and uncertain runs are Reject.",
        "core_ablation_base_raw_summary.json",
    ),
    (
        "+ Local Memory",
        "Add dev-calibrated profile-shift candidates, local normal memory correction, and historical context support.",
        "core_ablation_memory_correction_summary.json",
    ),
    (
        "+ Absence-aware revision",
        "Add log-derived service-coverage and service-silence evidence; strong absence can become positive Unexpected evidence.",
        "core_ablation_absence_aware_summary.json",
    ),
]


METRIC_KEYS = [
    "Expected_Precision",
    "Expected_Recall",
    "Expected_F1",
    "Unexpected_Precision",
    "Unexpected_Recall",
    "Unexpected_F1",
    "EU_Avg_F1",
    "Unexpected_False_Acceptance_Rate",
    "Unexpected_SafeRate",
    "Unexpected_Reject_Rate",
    "Run_Count",
]


def fmt(value):
    if isinstance(value, float):
        return f"{value:.4f}"
    return value


def load_summary(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def table_rows(summary_dir: Path, variants=None):
    variants = variants or DEFAULT_VARIANTS
    rows = []
    for variant, description, filename in variants:
        path = Path(filename)
        if not path.is_absolute():
            path = summary_dir / path
        if not path.exists():
            rows.append({
                "Variant": variant,
                "Description": description,
                "Status": f"missing:{filename}",
            })
            continue
        payload = load_summary(path)
        metrics = payload.get("metrics", {})
        confusion = payload.get("confusion", {})
        absence = payload.get("absence_veto", {})
        row = {
            "Variant": variant,
            "Description": description,
            "Status": "ok",
        }
        for key in METRIC_KEYS:
            row[key] = metrics.get(key, "")
        row["Expected->Expected"] = confusion.get("Expected->Expected", 0)
        row["Expected->Unexpected"] = confusion.get("Expected->Unexpected", 0)
        row["Expected->Reject"] = confusion.get("Expected->Reject", 0)
        row["Unexpected->Expected"] = confusion.get("Unexpected->Expected", 0)
        row["Unexpected->Unexpected"] = confusion.get("Unexpected->Unexpected", 0)
        row["Unexpected->Reject"] = confusion.get("Unexpected->Reject", 0)
        row["AbsenceRejectRuns"] = absence.get("applied_runs", 0)
        row["AbsenceUnexpectedRuns"] = absence.get("unexpected_applied_runs", 0)
        rows.append(row)
    return rows


def write_csv(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else ["Variant"]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(rows, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    headers = [
        "Variant",
        "Expected F1",
        "Unexpected F1",
        "EU Avg F1",
        "UFA",
        "SafeRate",
        "E->E",
        "U->E",
        "U->U",
        "U->R",
        "Absence U",
    ]
    lines = [
        "# Table 1: Core Component Ablation",
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = [
            row.get("Variant", ""),
            fmt(row.get("Expected_F1", "")),
            fmt(row.get("Unexpected_F1", "")),
            fmt(row.get("EU_Avg_F1", "")),
            fmt(row.get("Unexpected_False_Acceptance_Rate", "")),
            fmt(row.get("Unexpected_SafeRate", "")),
            row.get("Expected->Expected", ""),
            row.get("Unexpected->Expected", ""),
            row.get("Unexpected->Unexpected", ""),
            row.get("Unexpected->Reject", ""),
            row.get("AbsenceUnexpectedRuns", ""),
        ]
        lines.append("| " + " | ".join(str(v) for v in values) + " |")
    lines.extend([
        "",
        "Notes:",
        "- UFA denotes Unexpected False Acceptance Rate, i.e., the fraction of true Unexpected runs predicted as Expected.",
        "- SafeRate denotes the fraction of true Unexpected runs predicted as Unexpected or Reject.",
        "- Reject is a selective prediction action rather than a ground-truth class.",
        "- Global-score detector uses the same dev-calibrated score definition and run-level policy as the other variants, without memory, historical support, or absence evidence.",
        "- All variants share the same frozen anomaly-score configuration; only the named diagnosis mechanisms are added cumulatively.",
        "- All thresholds must be frozen on dev before these test summaries are generated.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary_dir", default="results/rq4/cloud_pilot")
    parser.add_argument("--base_summary", default="core_ablation_base_raw_summary.json")
    parser.add_argument(
        "--memory_summary",
        default="core_ablation_memory_correction_summary.json",
        help="Compatibility/diagnostic input for the memory-correction-only intermediate variant.",
    )
    parser.add_argument(
        "--historical_summary",
        default="core_ablation_historical_support_summary.json",
        help="Summary used for the Local Memory row; it includes memory correction and historical context support.",
    )
    parser.add_argument("--absence_summary", default="core_ablation_absence_aware_summary.json")
    parser.add_argument("--output_csv", default="results/rq4/cloud_pilot/core_component_ablation_table.csv")
    parser.add_argument("--output_md", default="results/rq4/cloud_pilot/core_component_ablation_table.md")
    args = parser.parse_args()

    variants = [
        (DEFAULT_VARIANTS[0][0], DEFAULT_VARIANTS[0][1], args.base_summary),
        # In the paper, Local Memory denotes the complete mechanism: local
        # normal-neighbour retrieval, memory correction, and historical
        # context support.  Therefore the table row must use the nested
        # historical-support variant, not the correction-only intermediate.
        (DEFAULT_VARIANTS[1][0], DEFAULT_VARIANTS[1][1], args.historical_summary),
        (DEFAULT_VARIANTS[2][0], DEFAULT_VARIANTS[2][1], args.absence_summary),
    ]
    rows = table_rows(Path(args.summary_dir), variants)
    write_csv(rows, Path(args.output_csv))
    write_markdown(rows, Path(args.output_md))
    print(f"[OK] Wrote {args.output_csv}")
    print(f"[OK] Wrote {args.output_md}")


if __name__ == "__main__":
    main()
