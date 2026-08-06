#!/usr/bin/env python3
"""Validate the frozen processed Train-Ticket RQ4 artifact."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path("data_cloud_expected_unexpected_expanded100_v0_4")
    required = {
        "train.pkl",
        "dev.pkl",
        "test.pkl",
        "annotation.csv",
        "metadata.json",
        "sequences.csv",
        "run_sampling_weights.csv",
        "vocab_templates.json",
    }
    missing = sorted(name for name in required if not (root / name).is_file())
    if missing:
        raise SystemExit(f"Missing dataset files: {missing}")

    with (root / "annotation.csv").open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    counts = Counter(row["split"] for row in rows)
    expected = {"train": 30, "dev": 38, "test": 100}
    if dict(counts) != expected:
        raise SystemExit(f"Unexpected run counts: {dict(counts)} != {expected}")

    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    if int(metadata.get("dim_process", -1)) != 699:
        raise SystemExit(f"Unexpected dim_process: {metadata.get('dim_process')}")
    print("[OK] Train-Ticket processed dataset")
    print("run counts:", dict(counts))
    print("sequence counts:", metadata.get("split_sequences"))
    print("dim_process:", metadata.get("dim_process"))


if __name__ == "__main__":
    main()

