#!/usr/bin/env python3
"""Convert real-change run logs into train/dev/test pkl files."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    chunk_event_windows,
    iter_logs,
    load_manifest,
    post_events,
    pre_events,
    row_run_dir,
    stable_oov_bucket,
    write_csv,
    write_json,
)


DATASET_BUILDER_VERSION = "dataset-builder-v0.7.1-service-phase-sequences-rq4-semantic-labels"

EXPECTED_RQ4_LABELS = {"expected_drift", "successful_no_drift"}
UNEXPECTED_RQ4_LABELS = {"unexpected_drift", "unexpected_without_observable_log_drift"}
NORMAL_REFERENCE_LABELS = {"baseline_normal", "no_op_control"}


def truthy(value) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "y"}


def load_summary(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def selected_phase_events(row: dict, raw_runs: Path) -> list[tuple[str, list[dict]]]:
    events = iter_logs(row_run_dir(row, raw_runs))
    split = str(row.get("split") or "").lower()
    label = str(row.get("benchmark_label") or "").lower()
    if split == "train" or label in {"baseline_normal", "no_op_control"}:
        return [
            ("pre_change", pre_events(row, events)),
            ("post_change", post_events(row, events)),
        ]
    return [("post_change", post_events(row, events))]


def selected_events(row: dict, raw_runs: Path) -> list[dict]:
    events = []
    for _, phase_events in selected_phase_events(row, raw_runs):
        events.extend(phase_events)
    return events


def known_template_vocab(rows: list[dict], raw_runs: Path) -> dict[str, int]:
    counts = Counter()
    for row in rows:
        if str(row.get("split") or "").lower() != "train":
            continue
        for event in selected_events(row, raw_runs):
            counts[event["_event_key"]] += 1
    return {template: idx for idx, (template, _) in enumerate(counts.most_common())}


def rq4_label(benchmark_label: str) -> int:
    if benchmark_label in EXPECTED_RQ4_LABELS:
        return 1
    if benchmark_label in UNEXPECTED_RQ4_LABELS:
        return 2
    return -1


def binary_label(benchmark_label: str):
    if benchmark_label in UNEXPECTED_RQ4_LABELS:
        return 1
    if benchmark_label in EXPECTED_RQ4_LABELS:
        return 0
    if benchmark_label in NORMAL_REFERENCE_LABELS:
        return 0
    return "unknown"


def drift_label(benchmark_label: str) -> str:
    if benchmark_label in EXPECTED_RQ4_LABELS:
        return "expected"
    if benchmark_label in UNEXPECTED_RQ4_LABELS:
        return "unexpected"
    if benchmark_label in NORMAL_REFERENCE_LABELS:
        return "normal"
    return "unlabeled"


def control_shift_label(benchmark_label: str, drift_gate_pass) -> str:
    gate_pass = truthy(drift_gate_pass)
    if benchmark_label == "baseline_normal":
        return "baseline_observable_shift" if gate_pass else "baseline_no_shift"
    if benchmark_label == "no_op_control":
        return "no_op_observable_shift" if gate_pass else "no_op_no_shift"
    return ""


def event_type_id(event: dict, vocab: dict[str, int], num_oov_buckets: int) -> tuple[int, int, int]:
    key = event["_event_key"]
    if key in vocab:
        return vocab[key], 0, -1
    bucket = stable_oov_bucket(event["_template"], num_oov_buckets)
    return len(vocab) + bucket, 1, bucket


def sequence_from_chunk(
    chunk: list[dict],
    row: dict,
    vocab: dict[str, int],
    num_oov_buckets: int,
    sequence_id: str,
    sequence_unit: str,
    phase: str,
    sequence_service: str,
) -> list[dict]:
    first_ts = chunk[0]["_timestamp"]
    previous_ts = first_ts
    benchmark = str(row.get("benchmark_label") or "unlabeled")
    seq = []
    for index, event in enumerate(chunk):
        type_id, is_oov, bucket = event_type_id(event, vocab, num_oov_buckets)
        ts = event["_timestamp"]
        seq.append({
            "time_since_start": max(0.0, ts - first_ts),
            "time_since_last_event": 0.0 if index == 0 else max(0.0, ts - previous_ts),
            "type_event": int(type_id),
            "label": binary_label(benchmark),
            "drift_label": drift_label(benchmark),
            "rq4_label": rq4_label(benchmark),
            "benchmark_label": benchmark,
            "semantic_label": row.get("semantic_label") or row.get("oracle_semantic_label") or row.get("declared_semantic_label") or "",
            "drift_gate_pass": int(str(row.get("drift_gate_pass") or "0") in {"1", "true", "True"}),
            "control_shift_label": control_shift_label(benchmark, row.get("drift_gate_pass")),
            "run_id": row.get("run_id"),
            "case_id": row.get("implementation_id") or row.get("change_family_id") or row.get("run_id"),
            "sequence_id": sequence_id,
            "sequence_unit": sequence_unit,
            "sequence_service": sequence_service,
            "change_family_id": row.get("change_family_id"),
            "implementation_id": row.get("implementation_id"),
            "component_id": row.get("component_id"),
            "change_target_component_id": row.get("change_target_component_id") or row.get("component_id"),
            "affected_component_ids": row.get("affected_component_ids"),
            "oracle_component_ids": row.get("oracle_component_ids"),
            "service": event.get("_service"),
            "event_template": event.get("_template"),
            "event_key": event.get("_event_key"),
            "is_oov_template": int(is_oov),
            "oov_hash_bucket": int(bucket),
            "phase": phase,
        })
        previous_ts = ts
    return seq


def service_event_groups(events: list[dict]) -> dict[str, list[dict]]:
    groups = defaultdict(list)
    for event in events:
        groups[event.get("_service") or "unknown"].append(event)
    for items in groups.values():
        items.sort(key=lambda item: (item["_timestamp"], item["_line_id"]))
    return dict(groups)


def write_split(out_dir: Path, split: str, dim_process: int, data: list[list[dict]]) -> None:
    with (out_dir / f"{split}.pkl").open("wb") as f:
        pickle.dump({"dim_process": dim_process, split: data}, f, protocol=pickle.HIGHEST_PROTOCOL)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--drift_summary", required=True)
    parser.add_argument("--raw_runs", default="raw_runs")
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--window_size", type=int, default=50)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--min_window_events", type=int, default=10)
    parser.add_argument("--oov_buckets", type=int, default=256)
    parser.add_argument(
        "--sequence_unit",
        choices=["service", "global"],
        default="service",
        help="service keeps each run_id+phase+service_name as a separate sequence; global keeps run_id+phase+global.",
    )
    parser.add_argument(
        "--allow_missing_component_id",
        action="store_true",
        help="Allow legacy protocol-development rows without component metadata.",
    )
    args = parser.parse_args()

    raw_runs = Path(args.raw_runs)
    manifest_rows = load_manifest(Path(args.manifest))
    summary = load_summary(Path(args.drift_summary))
    rows = []
    for row in manifest_rows:
        merged = dict(row)
        merged.update(summary.get(row["run_id"], {}))
        merged["split"] = row.get("split", merged.get("split", ""))
        if truthy(merged.get("used_for_protocol_development")) and str(merged.get("split") or "").lower() == "test":
            raise ValueError(f"Protocol-development run cannot be test split: {merged.get('run_id')}")
        component = merged.get("component_id") or merged.get("change_target_component_id")
        if not args.allow_missing_component_id and not str(component or "").strip():
            raise ValueError(f"Missing component_id/change_target_component_id: {merged.get('run_id')}")
        rows.append(merged)

    vocab = known_template_vocab(rows, raw_runs)
    dim_process = len(vocab) + args.oov_buckets
    splits = {"train": [], "dev": [], "test": []}
    annotation = []
    sequence_rows = []
    sequence_unit_name = "run_id+phase+service_name" if args.sequence_unit == "service" else "run_id+phase+global"

    for row in rows:
        split = str(row.get("split") or "").lower()
        if split not in splits:
            continue
        phase_groups = selected_phase_events(row, raw_runs)
        total_events = sum(len(events) for _, events in phase_groups)
        total_windows = 0
        if args.sequence_unit == "service":
            for phase, events in phase_groups:
                if not events:
                    continue
                for service, service_events in service_event_groups(events).items():
                    service_windows = chunk_event_windows(
                        service_events,
                        args.window_size,
                        args.step_size,
                        args.min_window_events,
                    )
                    for chunk in service_windows:
                        total_windows += 1
                        sequence_id = f"{row.get('run_id')}__{phase}__{service}__{total_windows:06d}"
                        seq = sequence_from_chunk(
                            chunk,
                            row,
                            vocab,
                            args.oov_buckets,
                            sequence_id=sequence_id,
                            sequence_unit=sequence_unit_name,
                            phase=phase,
                            sequence_service=service,
                        )
                        splits[split].append(seq)
                        sequence_rows.append({
                            "sequence_id": sequence_id,
                            "run_id": row.get("run_id"),
                            "split": split,
                            "benchmark_label": row.get("benchmark_label"),
                            "semantic_label": row.get("semantic_label") or row.get("oracle_semantic_label") or row.get("declared_semantic_label"),
                            "drift_gate_pass": row.get("drift_gate_pass"),
                            "control_shift_label": control_shift_label(str(row.get("benchmark_label") or ""), row.get("drift_gate_pass")),
                            "change_family_id": row.get("change_family_id"),
                            "implementation_id": row.get("implementation_id"),
                            "component_id": row.get("component_id"),
                            "change_target_component_id": row.get("change_target_component_id") or row.get("component_id"),
                            "affected_component_ids": row.get("affected_component_ids"),
                            "oracle_component_ids": row.get("oracle_component_ids"),
                            "sequence_unit": sequence_unit_name,
                            "service_name": service,
                            "phase": phase,
                            "event_count": len(chunk),
                            "first_timestamp": chunk[0]["_timestamp"],
                            "last_timestamp": chunk[-1]["_timestamp"],
                        })
        else:
            for phase, events in phase_groups:
                if not events:
                    continue
                service_windows = chunk_event_windows(
                    events,
                    args.window_size,
                    args.step_size,
                    args.min_window_events,
                )
                for chunk in service_windows:
                    total_windows += 1
                    sequence_id = f"{row.get('run_id')}__{phase}__global__{total_windows:06d}"
                    seq = sequence_from_chunk(
                        chunk,
                        row,
                        vocab,
                        args.oov_buckets,
                        sequence_id=sequence_id,
                        sequence_unit=sequence_unit_name,
                        phase=phase,
                        sequence_service="all",
                    )
                    splits[split].append(seq)
                    sequence_rows.append({
                        "sequence_id": sequence_id,
                        "run_id": row.get("run_id"),
                        "split": split,
                        "benchmark_label": row.get("benchmark_label"),
                        "semantic_label": row.get("semantic_label") or row.get("oracle_semantic_label") or row.get("declared_semantic_label"),
                        "drift_gate_pass": row.get("drift_gate_pass"),
                        "control_shift_label": control_shift_label(str(row.get("benchmark_label") or ""), row.get("drift_gate_pass")),
                        "change_family_id": row.get("change_family_id"),
                        "implementation_id": row.get("implementation_id"),
                        "component_id": row.get("component_id"),
                        "change_target_component_id": row.get("change_target_component_id") or row.get("component_id"),
                        "affected_component_ids": row.get("affected_component_ids"),
                        "oracle_component_ids": row.get("oracle_component_ids"),
                        "sequence_unit": sequence_unit_name,
                        "service_name": "all",
                        "phase": phase,
                        "event_count": len(chunk),
                        "first_timestamp": chunk[0]["_timestamp"],
                        "last_timestamp": chunk[-1]["_timestamp"],
                    })
        annotation.append({
            "run_id": row.get("run_id"),
            "split": split,
            "semantic_label": row.get("semantic_label") or row.get("oracle_semantic_label") or row.get("declared_semantic_label"),
            "benchmark_label": row.get("benchmark_label"),
            "drift_gate_pass": row.get("drift_gate_pass"),
            "control_shift_label": control_shift_label(str(row.get("benchmark_label") or ""), row.get("drift_gate_pass")),
            "change_family_id": row.get("change_family_id"),
            "implementation_id": row.get("implementation_id"),
            "component_id": row.get("component_id"),
            "change_target_component_id": row.get("change_target_component_id") or row.get("component_id"),
            "affected_component_ids": row.get("affected_component_ids"),
            "oracle_component_ids": row.get("oracle_component_ids"),
            "used_for_protocol_development": row.get("used_for_protocol_development"),
            "windows": total_windows,
            "events": total_events,
            "sequence_unit": sequence_unit_name,
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, data in splits.items():
        write_split(out_dir, split, dim_process, data)

    sequences_per_run = Counter(row["run_id"] for row in sequence_rows)
    for row in sequence_rows:
        count = sequences_per_run[row["run_id"]]
        row["sequences_in_run"] = count
        row["run_balanced_sequence_weight"] = (1.0 / count) if count else 0.0

    write_csv(out_dir / "annotation.csv", annotation)
    write_csv(out_dir / "sequences.csv", sequence_rows)
    write_csv(out_dir / "run_sampling_weights.csv", [
        {
            "run_id": run_id,
            "sequences": count,
            "run_weight": 1.0,
            "per_sequence_weight": (1.0 / count) if count else 0.0,
        }
        for run_id, count in sorted(sequences_per_run.items())
    ])
    write_json(out_dir / "vocab_templates.json", {
        "known_template_to_id": vocab,
        "oov_bucket_start": len(vocab),
        "oov_buckets": args.oov_buckets,
    })

    label_counts = defaultdict(int)
    event_counts = defaultdict(int)
    run_counts = defaultdict(set)
    oov_events = defaultdict(int)
    total_events = defaultdict(int)
    control_shift_counts = defaultdict(int)
    for split, data in splits.items():
        for seq in data:
            if not seq:
                continue
            label = seq[0].get("benchmark_label", "unknown")
            key = f"{split}:{label}"
            label_counts[key] += 1
            run_counts[key].add(seq[0].get("run_id"))
            for event in seq:
                event_counts[key] += 1
                total_events[key] += 1
                oov_events[key] += int(event.get("is_oov_template", 0))
            shift_label = seq[0].get("control_shift_label") or ""
            if shift_label:
                control_shift_counts[f"{split}:{shift_label}"] += 1

    train_run_ids = {
        row.get("run_id")
        for row in rows
        if str(row.get("split") or "").lower() == "train"
    }
    if not train_run_ids:
        vocabulary_status = "unavailable_no_training_split"
        oov_statistics_valid = False
        oov_statistics_note = (
            "All runs are outside train, so every template is mapped to an OOV bucket. "
            "The 100% OOV rate is a construction artifact of the protocol-development split, "
            "not a property of Train-Ticket logs."
        )
    elif not vocab:
        vocabulary_status = "empty_training_vocab"
        oov_statistics_valid = False
        oov_statistics_note = "Training split exists but yielded no known templates."
    else:
        vocabulary_status = "available_from_training_split"
        oov_statistics_valid = True
        oov_statistics_note = "OOV statistics are computed relative to the training split vocabulary."

    metadata = {
        "source": "SCWarn-style Train-Ticket Kubernetes real-change pilot",
        "dataset_builder_version": DATASET_BUILDER_VERSION,
        "dim_process": dim_process,
        "known_template_count": len(vocab),
        "vocabulary_status": vocabulary_status,
        "oov_statistics_valid": oov_statistics_valid,
        "oov_statistics_note": oov_statistics_note,
        "oov_buckets": args.oov_buckets,
        "window_size": args.window_size,
        "step_size": args.step_size,
        "min_window_events": args.min_window_events,
        "sequence_unit": sequence_unit_name,
        "sequence_unit_argument": args.sequence_unit,
        "time_sequence_policy": {
            "primary_unit": sequence_unit_name,
            "event_type": "service_name+normalized_template",
            "do_not_cross": ["run", "phase", "service"] if args.sequence_unit == "service" else ["run", "phase"],
            "timestamp_timezone": "UTC",
        },
        "sampling_policy": {
            "training_sampler": "run-balanced",
            "per_sequence_weight": "1 / number_of_sequences_in_same_run",
            "evaluation_unit": "run; aggregate sequence/window predictions before reporting Precision/Recall/F1",
        },
        "split_sequences": {split: len(data) for split, data in splits.items()},
        "split_label_windows": dict(label_counts),
        "split_label_events": dict(event_counts),
        "run_counts_by_split_label": {key: len(value) for key, value in run_counts.items()},
        "control_shift_windows": dict(control_shift_counts),
        "oov_by_split_label": {
            key: {
                "events": total_events[key],
                "oov_events": oov_events[key],
                "oov_rate": (oov_events[key] / total_events[key]) if total_events[key] else 0.0,
            }
            for key in sorted(total_events)
        },
        "label_layers": {
            "semantic_label": "expected/unexpected/indeterminate from external Oracle",
            "drift_gate_pass": "model-independent drift gate result",
            "control_shift_label": "derived control descriptor: baseline/no-op with observable shift or no shift; not a model target",
            "benchmark_label": "baseline_normal/no_op_control/successful_no_drift/expected_drift/unexpected_drift/unexpected_without_observable_log_drift/indeterminate_*",
            "model_decision": "expected/unexpected/reject, produced later by the model",
        },
    }
    write_json(out_dir / "metadata.json", metadata)
    print({
        "out_dir": args.out_dir,
        "dim_process": dim_process,
        "splits": metadata["split_sequences"],
        "labels": metadata["split_label_windows"],
    })


if __name__ == "__main__":
    main()
