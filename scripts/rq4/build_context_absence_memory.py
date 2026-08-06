#!/usr/bin/env python3
"""Build a compact standalone Context-conditioned Absence Memory pickle."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path


NORMAL_BENCHMARK_LABELS = {"baseline_normal", "no_op_control", "expected_reference_normal"}


def read_run_ids(path: Path) -> list[str]:
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        return [str(row.get("run_id") or "").strip() for row in rows if str(row.get("run_id") or "").strip()]
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip() and not line.lstrip().startswith("#")]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_sequence(run_dir: Path, run_id: str) -> tuple[list[dict], dict]:
    manifest_path = run_dir / "run_manifest.json"
    logs_path = run_dir / "logs.jsonl"
    if not manifest_path.is_file() or not logs_path.is_file():
        raise FileNotFoundError(f"{run_id}: run_manifest.json or logs.jsonl is missing")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if str(manifest.get("run_id") or "") != run_id:
        raise ValueError(f"{run_id}: run_manifest run_id mismatch")
    benchmark_label = str(manifest.get("benchmark_label") or "").strip()
    if benchmark_label not in NORMAL_BENCHMARK_LABELS:
        raise ValueError(f"{run_id}: non-normal benchmark_label={benchmark_label!r}")

    counts: Counter[str] = Counter()
    malformed = 0
    with logs_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                malformed += 1
                raise ValueError(f"{run_id}: malformed logs.jsonl line {line_number}: {exc}") from exc
            service = str(event.get("service") or "").strip()
            if service:
                counts[service] += 1
    if not counts:
        raise ValueError(f"{run_id}: no service-tagged log events")

    sequence = [
        {
            "run_id": run_id,
            "sequence_id": f"{run_id}__absence_memory",
            "service": service,
            "sequence_service": service,
            "absence_count": int(count),
            "label": 0,
            "drift_label": "normal",
            "benchmark_label": benchmark_label,
            "semantic_label": str(manifest.get("semantic_label") or "expected"),
            "workload_profile_id": str(manifest.get("workload_profile_id") or ""),
            "phase": "reference_observation_horizon",
        }
        for service, count in sorted(counts.items())
    ]
    return sequence, {
        "run_id": run_id,
        "benchmark_label": benchmark_label,
        "workload_profile_id": str(manifest.get("workload_profile_id") or ""),
        "services": len(counts),
        "events": int(sum(counts.values())),
        "malformed_lines": malformed,
        "logs_sha256": sha256(logs_path),
        "manifest_sha256": sha256(manifest_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_runs", required=True, help="Directory containing one subdirectory per run_id")
    parser.add_argument("--run_ids", required=True, help="Text/CSV file defining the exact trusted-normal run set")
    parser.add_argument("--output", required=True, help="Output absence_memory.pkl")
    parser.add_argument("--expected_count", type=int, required=True)
    args = parser.parse_args()

    raw_runs = Path(args.raw_runs)
    run_ids = read_run_ids(Path(args.run_ids))
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Duplicate run_id in trusted-normal run list")
    if len(run_ids) != args.expected_count:
        raise ValueError(f"Expected {args.expected_count} run ids, found {len(run_ids)}")
    missing = [run_id for run_id in run_ids if not (raw_runs / run_id).is_dir()]
    if missing:
        raise FileNotFoundError(
            f"Missing {len(missing)} trusted-normal run directories under {raw_runs}:\n"
            + "\n".join(missing)
        )

    sequences = []
    rows = []
    for run_id in run_ids:
        sequence, row = build_sequence(raw_runs / run_id, run_id)
        sequences.append(sequence)
        rows.append(row)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": "context-conditioned-absence-memory-v1",
        "absence_memory": sequences,
        "metadata": {
            "trusted_normal_only": True,
            "reference_runs": len(sequences),
            "run_ids": run_ids,
            "compact_service_counts": True,
        },
    }
    with output.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    report = {
        "status": "PASS",
        "output": str(output),
        "output_sha256": sha256(output),
        "reference_runs": len(sequences),
        "reference_services_union": len({event["service"] for seq in sequences for event in seq}),
        "total_log_events": sum(row["events"] for row in rows),
        "runs": rows,
    }
    report_path = output.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".sha256").write_text(
        f"{report['output_sha256']}  {output.name}\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in ("status", "output", "output_sha256", "reference_runs", "reference_services_union", "total_log_events")}, indent=2))


if __name__ == "__main__":
    main()
