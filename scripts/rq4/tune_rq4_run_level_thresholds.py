#!/usr/bin/env python3
"""Freeze run-level RQ4 thresholds on dev without touching test labels."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections import Counter
from itertools import product
from pathlib import Path


def parse_grid(value: str):
    return [float(item.strip()) for item in value.split(",") if item.strip()]


def optional_grid(value: str, fallback: float):
    values = parse_grid(value)
    return values or [fallback]


def as_bool(value):
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def f1(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = (
        2.0 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    return precision, recall, score


def decide_from_evidence(row, cfg, use_absence):
    """Replay the run decision from threshold-independent dev evidence."""
    expected_frac = float(row.get("expected_frac", 0.0) or 0.0)
    unexpected_frac = float(row.get("unexpected_frac", 0.0) or 0.0)
    reject_frac = float(row.get("reject_frac", 0.0) or 0.0)
    absence_low_risk = not (
        as_bool(row.get("absence_conflict", 0))
        or as_bool(row.get("coverage_conflict", 0))
    )

    if as_bool(row.get("absence_unexpected_applied", 0)):
        vote_pred_id = 2
    elif (
        unexpected_frac >= cfg["strong_unexpected_min"]
        or (
            unexpected_frac >= cfg["unexpected_min"]
            and unexpected_frac - expected_frac >= cfg["unexpected_margin"]
        )
    ):
        vote_pred_id = 2
    elif (
        expected_frac >= cfg["conflict_min"]
        and unexpected_frac >= cfg["conflict_min"]
    ):
        vote_pred_id = 3
    elif (
        expected_frac >= cfg["expected_min"]
        and unexpected_frac <= cfg["expected_unexpected_max"]
        and absence_low_risk
    ):
        vote_pred_id = 1
    elif (
        reject_frac >= cfg["reject_min"]
        and reject_frac - expected_frac >= cfg["reject_margin"]
    ):
        vote_pred_id = 3
    elif unexpected_frac >= cfg["unexpected_min"]:
        vote_pred_id = 2
    elif expected_frac >= cfg["expected_min"] and absence_low_risk:
        vote_pred_id = 1
    else:
        vote_pred_id = 0

    pred_id = vote_pred_id
    if use_absence and pred_id in {0, 1} and not absence_low_risk:
        pred_id = 3
    if as_bool(row.get("state_veto_applied", 0)):
        pred_id = 2
    return pred_id


def score_config(evidence_rows, cfg, use_absence):
    confusion = Counter()
    for row in evidence_rows:
        true_id = int(float(row.get("true_id", -1)))
        pred_id = decide_from_evidence(row, cfg, use_absence)
        confusion[(true_id, pred_id)] += 1

    metrics = {}
    # Preserve the explicit four-way diagnostic scores, but do not use them as
    # the formal RQ4 selection target.  The paper table evaluates operational
    # screening: Normal+Expected are accepted as Expected, while
    # Unexpected+Reject are escalated as Unexpected*.
    for cls_id, name in ((1, "Expected"), (2, "Unexpected")):
        tp = confusion[(cls_id, cls_id)]
        fp = sum(
            count for (true_id, pred_id), count in confusion.items()
            if true_id != cls_id and pred_id == cls_id
        )
        fn = sum(
            count for (true_id, pred_id), count in confusion.items()
            if true_id == cls_id and pred_id != cls_id
        )
        _, _, score = f1(tp, fp, fn)
        metrics[f"Semantic_{name}_F1"] = score
        metrics[f"{name}_True_Count"] = sum(
            count for (true_id, _), count in confusion.items()
            if true_id == cls_id
        )

    expected_tp = confusion[(1, 0)] + confusion[(1, 1)]
    expected_fp = confusion[(2, 0)] + confusion[(2, 1)]
    expected_fn = confusion[(1, 2)] + confusion[(1, 3)]
    unexpected_tp = confusion[(2, 2)] + confusion[(2, 3)]
    unexpected_fp = confusion[(1, 2)] + confusion[(1, 3)]
    unexpected_fn = confusion[(2, 0)] + confusion[(2, 1)]
    _, _, metrics["Expected_F1"] = f1(expected_tp, expected_fp, expected_fn)
    _, _, metrics["Unexpected_F1"] = f1(
        unexpected_tp,
        unexpected_fp,
        unexpected_fn,
    )
    metrics["EU_Avg_F1"] = (
        metrics["Expected_F1"] + metrics["Unexpected_F1"]
    ) / 2.0
    unexpected_total = metrics["Unexpected_True_Count"]
    ufa = (
        (confusion[(2, 0)] + confusion[(2, 1)]) / unexpected_total
        if unexpected_total
        else 0.0
    )
    metrics["Unexpected_False_Acceptance_Rate"] = ufa
    metrics["Unexpected_FalseExpected_Rate"] = (
        confusion[(2, 1)] / unexpected_total if unexpected_total else 0.0
    )
    metrics["Unexpected_Normal_Rate"] = (
        confusion[(2, 0)] / unexpected_total if unexpected_total else 0.0
    )
    metrics["Unexpected_SafeRate"] = 1.0 - ufa if unexpected_total else 0.0
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--events_csv", required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--event_pred_column", default="pred_drift_id")
    parser.add_argument("--decision_policy", default="evidence_priority")
    parser.add_argument("--state_veto", choices=["auto", "off"], default="off")
    parser.add_argument(
        "--include_labels",
        default=(
            "expected_drift,successful_no_drift,"
            "unexpected_drift,unexpected_without_observable_log_drift"
        ),
    )
    parser.add_argument("--expected_grid", default="0.30,0.35,0.40,0.45,0.50")
    parser.add_argument("--unexpected_grid", default="0.15,0.20,0.25,0.30")
    parser.add_argument("--reject_min", type=float, default=0.20)
    parser.add_argument("--conflict_min", type=float, default=0.20)
    parser.add_argument("--reject_margin", type=float, default=0.15)
    parser.add_argument("--unexpected_margin", type=float, default=0.0)
    parser.add_argument("--strong_unexpected_min", type=float, default=0.50)
    parser.add_argument("--expected_unexpected_max", type=float, default=0.05)
    parser.add_argument("--reject_min_grid", default="")
    parser.add_argument("--conflict_min_grid", default="")
    parser.add_argument("--reject_margin_grid", default="")
    parser.add_argument("--unexpected_margin_grid", default="")
    parser.add_argument("--strong_unexpected_grid", default="")
    parser.add_argument("--expected_unexpected_max_grid", default="")
    parser.add_argument("--max_ufa", type=float, default=0.25)
    parser.add_argument(
        "--min_expected_f1",
        type=float,
        default=0.0,
        help="Optional dev gate preventing a zero-Expected degenerate policy.",
    )
    parser.add_argument("--min_unexpected_f1", type=float, default=0.0)
    parser.add_argument("--min_safe_rate", type=float, default=0.0)
    parser.add_argument(
        "--selection_objective",
        choices=["balanced", "expected_f1"],
        default="balanced",
        help=(
            "Select by dev EU average F1 (balanced) or Expected F1 first. "
            "All safety and Unexpected-F1 constraints are applied before ranking."
        ),
    )
    parser.add_argument("--absence", action="store_true")
    parser.add_argument("--absence_context_mode", default="context_memory")
    parser.add_argument("--absence_reference_path", default="")
    parser.add_argument(
        "--absence_metadata_fields",
        default="",
    )
    parser.add_argument(
        "--absence_exclude_services",
        default="system-observability,tsdb-mysql,nacosdb-mysql",
    )
    parser.add_argument("--absence_k", type=int, default=20)
    parser.add_argument("--absence_active_beta", type=float, default=0.70)
    parser.add_argument("--absence_min_expected_count", type=float, default=20.0)
    parser.add_argument("--absence_count_ratio_threshold", type=float, default=0.50)
    parser.add_argument("--absence_anomaly_threshold", type=float, default=2.0)
    parser.add_argument("--absence_persistence_threshold", type=float, default=0.50)
    parser.add_argument("--absence_min_context_similarity", type=float, default=0.20)
    parser.add_argument("--absence_min_query_exposure", type=float, default=50.0)
    parser.add_argument("--absence_coverage_threshold", type=float, default=0.50)
    parser.add_argument("--absence_strong_anomaly_threshold", type=float, default=3.0)
    parser.add_argument("--absence_strong_coverage_threshold", type=float, default=1.0)
    parser.add_argument("--keep_intermediate", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summarizer = Path(__file__).with_name("summarize_rq4_run_level.py")
    # Materialize threshold-independent run evidence once.  The old tuner
    # launched the full summarizer for every grid point, repeatedly parsing the
    # same event CSV and rebuilding the same absence memory.  A single pass is
    # sufficient because the searched parameters affect only run aggregation.
    evidence_prefix = output_dir / "dev_evidence"
    cmd = [
            sys.executable,
            str(summarizer),
            "--data_dir",
            args.data_dir,
            "--events_csv",
            args.events_csv,
            "--batch_size",
            str(args.batch_size),
            "--state_veto",
            args.state_veto,
            "--include_labels",
            args.include_labels,
            "--event_pred_column",
            args.event_pred_column,
            "--decision_policy",
            args.decision_policy,
            "--expected_min",
            str(parse_grid(args.expected_grid)[0]),
            "--unexpected_min",
            str(parse_grid(args.unexpected_grid)[0]),
            "--reject_min",
            str(args.reject_min),
            "--conflict_min",
            str(args.conflict_min),
            "--reject_margin",
            str(args.reject_margin),
            "--unexpected_margin",
            str(args.unexpected_margin),
            "--strong_unexpected_min",
            str(args.strong_unexpected_min),
            "--expected_unexpected_max",
            str(args.expected_unexpected_max),
            "--output_prefix",
            str(evidence_prefix),
    ]
    if args.absence:
        cmd.extend([
                "--absence_veto",
                "reject",
                "--absence_apply_to",
                "normal_expected",
                "--absence_context_mode",
                args.absence_context_mode,
                "--absence_reference_path",
                args.absence_reference_path,
                "--absence_metadata_fields",
                args.absence_metadata_fields,
                "--absence_exclude_services",
                args.absence_exclude_services,
                "--absence_k",
                str(args.absence_k),
                "--absence_active_beta",
                str(args.absence_active_beta),
                "--absence_min_expected_count",
                str(args.absence_min_expected_count),
                "--absence_count_ratio_threshold",
                str(args.absence_count_ratio_threshold),
                "--absence_anomaly_threshold",
                str(args.absence_anomaly_threshold),
                "--absence_persistence_threshold",
                str(args.absence_persistence_threshold),
                "--absence_min_context_similarity",
                str(args.absence_min_context_similarity),
                "--absence_min_query_exposure",
                str(args.absence_min_query_exposure),
                "--absence_coverage_threshold",
                str(args.absence_coverage_threshold),
                "--absence_unexpected_mode",
                "strong",
                "--absence_strong_anomaly_threshold",
                str(args.absence_strong_anomaly_threshold),
                "--absence_strong_coverage_threshold",
                str(args.absence_strong_coverage_threshold),
        ])
    completed = subprocess.run(cmd, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            "Dev evidence extraction failed:\n"
            f"{completed.stdout}\n{completed.stderr}"
        )

    evidence_path = evidence_prefix.with_name(evidence_prefix.name + "_predictions.csv")
    summary_path = evidence_prefix.with_name(evidence_prefix.name + "_summary.json")
    with evidence_path.open("r", encoding="utf-8-sig", newline="") as f:
        evidence_rows = list(csv.DictReader(f))
    if not evidence_rows:
        raise RuntimeError("Dev evidence extraction produced no run-level rows.")

    grids = product(
        parse_grid(args.expected_grid),
        parse_grid(args.unexpected_grid),
        optional_grid(args.reject_min_grid, args.reject_min),
        optional_grid(args.conflict_min_grid, args.conflict_min),
        optional_grid(args.reject_margin_grid, args.reject_margin),
        optional_grid(args.unexpected_margin_grid, args.unexpected_margin),
        optional_grid(args.strong_unexpected_grid, args.strong_unexpected_min),
        optional_grid(
            args.expected_unexpected_max_grid,
            args.expected_unexpected_max,
        ),
    )
    rows = []
    for (
        expected_min,
        unexpected_min,
        reject_min,
        conflict_min,
        reject_margin,
        unexpected_margin,
        strong_unexpected_min,
        expected_unexpected_max,
    ) in grids:
        cfg = {
            "expected_min": expected_min,
            "unexpected_min": unexpected_min,
            "reject_min": reject_min,
            "conflict_min": conflict_min,
            "reject_margin": reject_margin,
            "unexpected_margin": unexpected_margin,
            "strong_unexpected_min": strong_unexpected_min,
            "expected_unexpected_max": expected_unexpected_max,
        }
        metrics = score_config(evidence_rows, cfg, args.absence)
        row = {
            "expected_min": expected_min,
            "unexpected_min": unexpected_min,
            "reject_min": reject_min,
            "conflict_min": conflict_min,
            "reject_margin": reject_margin,
            "unexpected_margin": unexpected_margin,
            "strong_unexpected_min": strong_unexpected_min,
            "expected_unexpected_max": expected_unexpected_max,
            "Expected_F1": metrics.get("Expected_F1", 0.0),
            "Unexpected_F1": metrics.get("Unexpected_F1", 0.0),
            "EU_Avg_F1": metrics.get("EU_Avg_F1", 0.0),
            "Unexpected_False_Acceptance_Rate": metrics.get(
                "Unexpected_False_Acceptance_Rate", 1.0
            ),
            "Unexpected_FalseExpected_Rate": metrics.get(
                "Unexpected_FalseExpected_Rate", 1.0
            ),
            "Unexpected_Normal_Rate": metrics.get("Unexpected_Normal_Rate", 1.0),
            "Unexpected_SafeRate": metrics.get("Unexpected_SafeRate", 0.0),
            "Semantic_Expected_F1": metrics.get("Semantic_Expected_F1", 0.0),
            "Semantic_Unexpected_F1": metrics.get("Semantic_Unexpected_F1", 0.0),
        }
        rows.append(row)

    if not args.keep_intermediate:
        summary_path.unlink(missing_ok=True)
        evidence_path.unlink(missing_ok=True)

    eligible = [
        row for row in rows
        if (
            row["Unexpected_False_Acceptance_Rate"] <= args.max_ufa
            and row["Expected_F1"] >= args.min_expected_f1
            and row["Unexpected_F1"] >= args.min_unexpected_f1
            and row["Unexpected_SafeRate"] >= args.min_safe_rate
        )
    ]
    pool = eligible or rows
    # Copy the selected grid row before attaching provenance/configuration
    # fields.  Mutating the original row would add JSON-only fields to one CSV
    # record after the CSV header had already been defined from the grid schema.
    if args.selection_objective == "expected_f1":
        rank_key = lambda row: (
            row["Expected_F1"],
            row["EU_Avg_F1"],
            row["Unexpected_F1"],
            row["Unexpected_SafeRate"],
        )
    else:
        rank_key = lambda row: (
            row["EU_Avg_F1"],
            row["Expected_F1"],
            row["Unexpected_F1"],
            row["Unexpected_SafeRate"],
        )
    best = dict(max(pool, key=rank_key))
    best.update({
        "decision_policy": args.decision_policy,
        "event_pred_column": args.event_pred_column,
        "state_veto": args.state_veto,
        "include_labels": args.include_labels,
        "max_ufa_constraint": args.max_ufa,
        "min_expected_f1_constraint": args.min_expected_f1,
        "min_unexpected_f1_constraint": args.min_unexpected_f1,
        "min_safe_rate_constraint": args.min_safe_rate,
        "selection_objective": args.selection_objective,
        "constraint_satisfied": bool(eligible),
        "selected_on": "dev",
    })

    grid_path = output_dir / "dev_threshold_grid.csv"
    with grid_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    best_path = output_dir / "best_threshold.json"
    with best_path.open("w", encoding="utf-8") as f:
        json.dump(best, f, ensure_ascii=False, indent=2)

    print(json.dumps(best, ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {grid_path}")
    print(f"[OK] Wrote {best_path}")


if __name__ == "__main__":
    main()
