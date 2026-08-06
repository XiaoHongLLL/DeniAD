#!/usr/bin/env python3
"""Merge run-level Unexpected and Reject decisions into one escalation class."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


TRUE_CLASSES = ("Expected", "Unexpected")


def safe_div(num: int, den: int) -> float:
    return num / den if den else 0.0


def class_metrics(confusion: Counter, positive: str):
    negative = "Unexpected*" if positive == "Expected" else "Expected"
    tp = confusion[(positive, positive)]
    fp = confusion[(negative, positive)]
    fn = confusion[(positive, negative)]
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {"precision": precision, "recall": recall, "f1": f1}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions_csv", required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument("--method", default="DeniAD + Context-conditioned Absence Memory")
    parser.add_argument(
        "--normal_policy",
        choices=["accept", "error"],
        default="accept",
        help=(
            "Operational screening maps Normal and Expected to the accepted channel. "
            "Use error to fail if a Normal run-level decision is present."
        ),
    )
    args = parser.parse_args()

    source = Path(args.predictions_csv)
    with source.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))

    confusion = Counter()
    original = Counter()
    subtype = Counter()
    included = 0
    for row in rows:
        truth = row.get("true_label", "")
        pred = row.get("pred_label", "")
        if truth not in TRUE_CLASSES:
            continue
        if pred == "Normal":
            if args.normal_policy == "error":
                raise ValueError(
                    f"Normal prediction found for run {row.get('run_id', '')}; "
                    "choose --normal_policy accept or revise the run-level rule."
                )
            merged_pred = "Expected"
        elif pred == "Expected":
            merged_pred = "Expected"
        elif pred in {"Unexpected", "Reject"}:
            merged_pred = "Unexpected*"
        else:
            raise ValueError(
                f"Unsupported pred_label={pred!r} for run {row.get('run_id', '')}"
            )
        confusion[(truth if truth == "Expected" else "Unexpected*", merged_pred)] += 1
        original[(truth, pred)] += 1
        subtype[(truth, row.get("benchmark_label", ""), merged_pred)] += 1
        included += 1

    expected = class_metrics(confusion, "Expected")
    operational_unexpected = class_metrics(confusion, "Unexpected*")
    correct = confusion[("Expected", "Expected")] + confusion[("Unexpected*", "Unexpected*")]
    accuracy = safe_div(correct, included)
    macro_f1 = (expected["f1"] + operational_unexpected["f1"]) / 2.0
    balanced_accuracy = (expected["recall"] + operational_unexpected["recall"]) / 2.0
    true_expected = sum(confusion[("Expected", p)] for p in ("Expected", "Unexpected*"))
    true_unexpected = sum(confusion[("Unexpected*", p)] for p in ("Expected", "Unexpected*"))
    ufa = safe_div(confusion[("Unexpected*", "Expected")], true_unexpected)
    expected_far = safe_div(confusion[("Expected", "Unexpected*")], true_expected)

    payload = {
        "method": args.method,
        "source_predictions": str(source),
        "merge_rule": {
            "accepted_expected": ["Normal", "Expected"],
            "operational_unexpected": ["Unexpected", "Reject"],
            "display_note": (
                "Unexpected* is an operational escalation class, not explicit "
                "fault-attribution recall."
            ),
        },
        "run_count": included,
        "confusion": {
            "Expected->Expected": confusion[("Expected", "Expected")],
            "Expected->Unexpected*": confusion[("Expected", "Unexpected*")],
            "Unexpected->Expected": confusion[("Unexpected*", "Expected")],
            "Unexpected->Unexpected*": confusion[("Unexpected*", "Unexpected*")],
        },
        "original_decision_counts": {
            f"{truth}->{pred}": count
            for (truth, pred), count in sorted(original.items())
        },
        "metrics": {
            "Expected_Precision": expected["precision"],
            "Expected_Recall": expected["recall"],
            "Expected_F1": expected["f1"],
            "UnexpectedStar_Precision": operational_unexpected["precision"],
            "UnexpectedStar_Recall": operational_unexpected["recall"],
            "UnexpectedStar_F1": operational_unexpected["f1"],
            "Macro_F1": macro_f1,
            "Accuracy": accuracy,
            "Balanced_Accuracy": balanced_accuracy,
            "Unexpected_False_Acceptance_Rate": ufa,
            "Expected_False_Alarm_Rate": expected_far,
        },
        "subtype_confusion": [
            {
                "true_label": truth,
                "benchmark_label": benchmark,
                "pred_label": pred,
                "count": count,
            }
            for (truth, benchmark, pred), count in sorted(subtype.items())
        ],
    }

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    json_path = out_prefix.with_name(out_prefix.name + "_table.json")
    csv_path = out_prefix.with_name(out_prefix.name + "_table.csv")
    md_path = out_prefix.with_name(out_prefix.name + "_table.md")

    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    table_rows = [
        {
            "Method": args.method,
            "Expected_P": expected["precision"],
            "Expected_R": expected["recall"],
            "Expected_F1": expected["f1"],
            "UnexpectedStar_P": operational_unexpected["precision"],
            "UnexpectedStar_R": operational_unexpected["recall"],
            "UnexpectedStar_F1": operational_unexpected["f1"],
            "Macro_F1": macro_f1,
            "Accuracy": accuracy,
            "UFA": ufa,
            "Expected_FAR": expected_far,
            "Runs": included,
        }
    ]
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(table_rows[0].keys()))
        writer.writeheader()
        writer.writerows(table_rows)

    def pct(value: float) -> str:
        return f"{100.0 * value:.1f}%"

    lines = [
        "# RQ4 run-level operational screening",
        "",
        "Unexpected* = explicit Unexpected + Reject. For operational screening, "
        "Normal and Expected decisions enter the accepted channel.",
        "",
        "## Confusion matrix",
        "",
        "| True label | Pred. Expected | Pred. Unexpected* |",
        "|---|---:|---:|",
        (
            f"| Expected (n={true_expected}) | {confusion[('Expected', 'Expected')]} | "
            f"{confusion[('Expected', 'Unexpected*')]} |"
        ),
        (
            f"| Unexpected (n={true_unexpected}) | {confusion[('Unexpected*', 'Expected')]} | "
            f"{confusion[('Unexpected*', 'Unexpected*')]} |"
        ),
        "",
        "## Main table",
        "",
        "| Method | Expected P | Expected R | Expected F1 | Unexpected* P | Unexpected* R | Unexpected* F1 | Macro-F1 | Accuracy | UFA | E-FAR | Runs |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        (
            f"| {args.method} | {expected['precision']:.3f} | {expected['recall']:.3f} | "
            f"{expected['f1']:.3f} | {operational_unexpected['precision']:.3f} | "
            f"{operational_unexpected['recall']:.3f} | {operational_unexpected['f1']:.3f} | "
            f"{macro_f1:.3f} | {accuracy:.3f} | {ufa:.3f} | {expected_far:.3f} | {included} |"
        ),
        "",
        "Unexpected* recall is the unsafe-run interception rate. It must not be "
        "reported as explicit Unexpected diagnosis recall because Reject decisions are included.",
        "",
        f"- Unsafe-run interception rate: {pct(operational_unexpected['recall'])}",
        f"- Unexpected false acceptance rate (UFA): {pct(ufa)}",
        f"- Expected false alarm rate (E-FAR): {pct(expected_far)}",
    ]
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"[OK] Wrote {csv_path}")
    print(f"[OK] Wrote {md_path}")
    print(f"[OK] Wrote {json_path}")


if __name__ == "__main__":
    main()
