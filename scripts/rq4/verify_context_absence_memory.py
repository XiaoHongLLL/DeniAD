#!/usr/bin/env python3
"""Fail-fast integrity and leakage checks for an absence-memory file."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from context_conditioned_absence_memory import load_memory_sequences, safe_text


def load_split(path: Path, split: str):
    with (path / f"{split}.pkl").open("rb") as handle:
        return pickle.load(handle, encoding="latin-1")[split]


def run_ids(data) -> set[str]:
    return {
        safe_text(sequence[0].get("run_id"))
        for sequence in data
        if sequence and safe_text(sequence[0].get("run_id"))
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory", required=True)
    parser.add_argument("--dataset_dir", required=True)
    parser.add_argument("--expected_count", type=int, required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    memory = load_memory_sequences(args.memory)
    memory_ids = run_ids(memory)
    if len(memory_ids) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} memory runs, found {len(memory_ids)}")
    if len(memory) != len(memory_ids):
        raise ValueError(f"Expected one compact sequence per run; sequences={len(memory)} runs={len(memory_ids)}")
    non_normal = []
    for sequence in memory:
        for event in sequence:
            if int(float(event.get("label", 0))) != 0 or str(event.get("drift_label", "normal")).lower() not in {"normal", "expected", "expected_drift"}:
                non_normal.append(safe_text(event.get("run_id")))
                break
    if non_normal:
        raise ValueError(f"Non-normal records in absence memory: {sorted(set(non_normal))}")

    dataset_dir = Path(args.dataset_dir)
    test_ids = run_ids(load_split(dataset_dir, "test"))
    dev_ids = run_ids(load_split(dataset_dir, "dev"))
    overlap_test = sorted(memory_ids & test_ids)
    overlap_dev = sorted(memory_ids & dev_ids)
    if overlap_test or overlap_dev:
        raise ValueError(f"Reference leakage: test_overlap={overlap_test}, dev_overlap={overlap_dev}")

    report = {
        "status": "PASS",
        "mechanism": "Context-conditioned Absence Memory",
        "memory_runs": len(memory_ids),
        "dev_overlap": 0,
        "test_overlap": 0,
        "trusted_normal_only": True,
        "memory_path": str(Path(args.memory)),
        "dataset_dir": str(dataset_dir),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()
