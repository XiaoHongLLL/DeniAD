#!/usr/bin/env python3
"""Compute event-type distribution shift statistics for the paper datasets.

The primary comparison uses the model-input ``type_event`` IDs in train.pkl
and test.pkl. Jensen--Shannon divergence uses log base 2 and is therefore
bounded to [0, 1]. No additive smoothing is applied; JSD is well-defined on
the union support because its mixture distribution is non-zero wherever
either input distribution is non-zero.

OOV is defined relative to the training split:

    number of test events whose type_event is absent from train
    ------------------------------------------------------------
                     number of test events

For public datasets, ``type_event`` is the retained preprocessed log-key ID.
For datasets whose preprocessing collapses rare raw templates into an UNK
bucket, this is a conservative model-input OOV estimate rather than an exact
count of distinct raw message templates.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter
from pathlib import Path
from typing import Iterable, Sequence


DATASETS = (
    ("HDFS", Path("data/labeled_hdfs")),
    ("BGL", Path("data/labeled_bgl")),
    ("Thunderbird", Path("data/labeled_thunderbird")),
    ("Spirit", Path("data/labeled_spirit")),
    ("Liberty", Path("data/labeled_liberty")),
    ("Train-Ticket", Path("data_cloud_expected_unexpected_expanded100_v0_4")),
)


def load_split(root: Path, split: str) -> list[list[dict]]:
    with (root / f"{split}.pkl").open("rb") as handle:
        payload = pickle.load(handle)
    return payload[split]


def event_type_counts(sequences: Iterable[Sequence[dict]]) -> Counter[int]:
    return Counter(int(event["type_event"]) for seq in sequences for event in seq)


def js_divergence_bits(left: Counter[int], right: Counter[int]) -> float:
    """Return JSD(P || Q) using log base 2 on the union support."""
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        raise ValueError("JSD requires two non-empty event distributions")

    divergence = 0.0
    for key in left.keys() | right.keys():
        p = left[key] / left_total
        q = right[key] / right_total
        mixture = 0.5 * (p + q)
        if p:
            divergence += 0.5 * p * math.log2(p / mixture)
        if q:
            divergence += 0.5 * q * math.log2(q / mixture)
    return divergence


def benign_test_subset(dataset: str, test: list[list[dict]]) -> list[list[dict]]:
    if dataset == "Train-Ticket":
        return [
            seq
            for seq in test
            if seq and str(seq[0].get("semantic_label", "")).lower() == "expected"
        ]
    return [
        seq
        for seq in test
        if not any(int(event.get("label", 0)) != 0 for event in seq)
    ]


def comparison(label: str, train: Counter[int], test: Counter[int]) -> dict[str, object]:
    train_types = set(train)
    test_types = set(test)
    oov_types = test_types - train_types
    test_events = sum(test.values())
    oov_events = sum(test[event_type] for event_type in oov_types)
    return {
        f"{label}_Events": test_events,
        f"{label}_Types": len(test_types),
        f"JSD_Train_{label}_Bits": js_divergence_bits(train, test),
        f"OOV_Events_{label}": oov_events,
        f"OOV_Types_{label}": len(oov_types),
        f"OOV_Rate_{label}": oov_events / test_events,
    }


def compute(repo_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, relative_root in DATASETS:
        root = repo_root / relative_root
        train_sequences = load_split(root, "train")
        test_sequences = load_split(root, "test")
        benign_sequences = benign_test_subset(dataset, test_sequences)

        train_counts = event_type_counts(train_sequences)
        test_counts = event_type_counts(test_sequences)
        benign_counts = event_type_counts(benign_sequences)

        row: dict[str, object] = {
            "Dataset": dataset,
            "Train_Events": sum(train_counts.values()),
            "Train_Types": len(train_counts),
            "JSD_Log_Base": 2,
            "OOV_Basis": "type_event absent from train",
            "Benign_Test_Definition": (
                "semantic_label=expected"
                if dataset == "Train-Ticket"
                else "sequence contains no anomalous event"
            ),
        }
        row.update(comparison("AllTest", train_counts, test_counts))
        row.update(comparison("BenignTest", train_counts, benign_counts))
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    rows = compute(args.repo_root.resolve())
    if args.csv:
        args.csv.parent.mkdir(parents=True, exist_ok=True)
        with args.csv.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
