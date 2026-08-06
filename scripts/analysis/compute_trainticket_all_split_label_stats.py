#!/usr/bin/env python3
"""Audit Train-Ticket Expected/Unexpected labels across train/dev/test.

Run counts are taken from the one-row-per-run annotation.csv. Sequence and
event counts are taken from the materialized pickle files actually consumed by
the model, because annotation.csv records pre-windowing event totals.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter
from pathlib import Path


SPLITS = ("train", "dev", "test")
SEMANTIC_LABELS = ("expected", "unexpected")


def load_pickle_sequences(data_root: Path, split: str) -> list[list[dict]]:
    with (data_root / f"{split}.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    return payload[split]


def uniform_sequence_label(sequence: list[dict], field: str) -> str:
    values = {str(event.get(field, "")) for event in sequence}
    if len(values) != 1:
        raise ValueError(f"Sequence has non-uniform {field}: {sorted(values)}")
    return next(iter(values))


def compute(data_root: Path) -> dict[str, object]:
    with (data_root / "annotation.csv").open(encoding="utf-8-sig", newline="") as handle:
        annotations = list(csv.DictReader(handle))

    run_ids = [row["run_id"] for row in annotations]
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("annotation.csv contains duplicate run_id values")

    run_counts: Counter[tuple[str, str, str]] = Counter()
    for row in annotations:
        run_counts[(row["split"], row["semantic_label"], row["benchmark_label"])] += 1

    sequence_counts: Counter[tuple[str, str, str]] = Counter()
    event_counts: Counter[tuple[str, str, str]] = Counter()
    for split in SPLITS:
        for sequence in load_pickle_sequences(data_root, split):
            if not sequence:
                continue
            semantic = uniform_sequence_label(sequence, "semantic_label")
            benchmark = uniform_sequence_label(sequence, "benchmark_label")
            key = (split, semantic, benchmark)
            sequence_counts[key] += 1
            event_counts[key] += len(sequence)

    semantic_rows: list[dict[str, object]] = []
    for split in (*SPLITS, "all"):
        selected_splits = SPLITS if split == "all" else (split,)
        for semantic in SEMANTIC_LABELS:
            semantic_rows.append(
                {
                    "Split": split,
                    "Semantic_Label": semantic,
                    "Runs": sum(
                        count
                        for (part, label, _), count in run_counts.items()
                        if part in selected_splits and label == semantic
                    ),
                    "Sequences": sum(
                        count
                        for (part, label, _), count in sequence_counts.items()
                        if part in selected_splits and label == semantic
                    ),
                    "Events": sum(
                        count
                        for (part, label, _), count in event_counts.items()
                        if part in selected_splits and label == semantic
                    ),
                }
            )

    benchmark_labels = sorted({label for _, _, label in run_counts})
    benchmark_rows: list[dict[str, object]] = []
    for split in (*SPLITS, "all"):
        selected_splits = SPLITS if split == "all" else (split,)
        for benchmark in benchmark_labels:
            semantic_values = {
                semantic
                for (part, semantic, label), count in run_counts.items()
                if part in selected_splits and label == benchmark and count
            }
            if not semantic_values:
                continue
            if len(semantic_values) != 1:
                raise ValueError(f"Benchmark label maps to multiple semantic labels: {benchmark}")
            semantic = next(iter(semantic_values))
            benchmark_rows.append(
                {
                    "Split": split,
                    "Semantic_Label": semantic,
                    "Benchmark_Label": benchmark,
                    "Runs": sum(
                        count
                        for (part, _, label), count in run_counts.items()
                        if part in selected_splits and label == benchmark
                    ),
                    "Sequences": sum(
                        count
                        for (part, _, label), count in sequence_counts.items()
                        if part in selected_splits and label == benchmark
                    ),
                    "Events": sum(
                        count
                        for (part, _, label), count in event_counts.items()
                        if part in selected_splits and label == benchmark
                    ),
                }
            )

    return {
        "data_root": str(data_root),
        "counting_policy": {
            "runs": "unique run_id rows in annotation.csv",
            "sequences": "materialized sequences in train/dev/test.pkl",
            "events": "materialized events in train/dev/test.pkl",
        },
        "semantic_summary": semantic_rows,
        "benchmark_detail": benchmark_rows,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(__file__).resolve().parents[2]
        / "data_cloud_expected_unexpected_expanded100_v0_4",
    )
    parser.add_argument("--semantic-csv", type=Path)
    parser.add_argument("--benchmark-csv", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    result = compute(args.data_root.resolve())
    if args.semantic_csv:
        write_csv(args.semantic_csv, result["semantic_summary"])
    if args.benchmark_csv:
        write_csv(args.benchmark_csv, result["benchmark_detail"])
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
