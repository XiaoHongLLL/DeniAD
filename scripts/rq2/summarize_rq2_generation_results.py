#!/usr/bin/env python3
"""Summarize RQ2 generation metric CSV files."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
from pathlib import Path


GENERATION_FIELDS = ["Acc", "NLL", "Time_NLL", "Type_NLL", "RMSE", "CS"]


def read_single_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return rows[0] if rows else {}


def infer_dataset(csv_path: Path, result_root: Path) -> str:
    parent = csv_path.parent
    if parent != result_root and parent.name:
        return parent.name
    name = csv_path.name.lower()
    for dataset in ("thunderbird", "spirit", "liberty"):
        if name.startswith(f"{dataset}_"):
            return dataset
    return parent.name or "unknown"


def collect(result_root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for csv_path in sorted(result_root.rglob("*_results.csv")):
        lower = csv_path.name.lower()
        if "_anomaly_results.csv" in lower or "_reliability_results.csv" in lower:
            continue

        source = read_single_row(csv_path)
        if not any(field in source for field in GENERATION_FIELDS):
            continue

        stat = csv_path.stat()
        row = {
            "Dataset": infer_dataset(csv_path, result_root),
            "RunName": csv_path.name.removesuffix("_results.csv"),
            "ResultFile": str(csv_path),
            "ModifiedTime": datetime.fromtimestamp(stat.st_mtime).isoformat(timespec="seconds"),
        }
        for field in GENERATION_FIELDS:
            row[field] = source.get(field, "")
        rows.append(row)
    return rows


def write_rows(rows: list[dict[str, str]], output: Path) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_root", default="./results/rq2_generation")
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset-name filter, for example: hdfs bgl openstack thunderbird spirit liberty.",
    )
    args = parser.parse_args()

    result_root = Path(args.result_root)
    output = Path(args.output) if args.output else result_root / "summary.csv"
    rows = collect(result_root)
    if args.datasets:
        wanted = {name.lower() for name in args.datasets}
        rows = [row for row in rows if row["Dataset"].lower() in wanted]
    if not rows:
        print(f"[Warn] No RQ2 generation result CSV files found under {result_root}")
        return

    write_rows(rows, output)
    print(f"[OK] Wrote {output}")
    for row in rows:
        print(
            f"{row['Dataset']:12s} "
            f"Acc={row.get('Acc', '')} "
            f"NLL={row.get('NLL', '')} "
            f"RMSE={row.get('RMSE', '')} "
            f"CS={row.get('CS', '')}"
        )


if __name__ == "__main__":
    main()
