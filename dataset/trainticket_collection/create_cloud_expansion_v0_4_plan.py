#!/usr/bin/env python3
"""Create a non-blind v0.4 cloud expansion plan.

The current combined67 dataset is already a usable controlled benchmark. This
plan targets the next reviewer-risk items in order:

1. increase normal/reference memory for training;
2. increase dev coverage without touching the formal test set;
3. supplement the formal test set to about 100 run-level test cases;
4. preregister boundary/reject probes as a separate holdout backlog.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import create_cloud_formal_v0_3_plan as v03  # noqa: E402
from create_cloud_control_plan import scenario_metadata as control_metadata  # noqa: E402


BATCH_ID = "cloud_expansion_v0_4_20260709"
CLOUD_PROTOCOL_VERSION = "linux-cloud-expansion-v0.4-draft"
SOURCE_DATASET_ID = "cloud_expected_unexpected_combined67_v0_3_1"
THRESHOLD_SHA256 = "99b2a27a4f6192b86ca4ea83a550be26e5605b8254ba0511d52ced3fa7584c16"
THRESHOLD_SOURCE = "cloud_controls_calibration_only"
CLOUD_FREEZE_ID = "1e09d8d1edc77982565a9b66066ee6289feca7d9d57b51911546f1cca9ea804f"
CLUSTER_TYPE = "linux_kubeadm_multinode"

CURRENT = {
    "train_reference_runs": 9,
    "dev_expected_drift": 3,
    "dev_successful_no_drift": 1,
    "dev_unexpected_drift": 3,
    "dev_unexpected_without_observable_log_drift": 1,
    "test_expected_drift": 17,
    "test_successful_no_drift": 18,
    "test_unexpected_drift": 17,
    "test_unexpected_without_observable_log_drift": 15,
}

TARGET = {
    "train_reference_runs": 30,
    "dev_expected_drift": 8,
    "dev_successful_no_drift": 8,
    "dev_unexpected_drift": 8,
    "dev_unexpected_without_observable_log_drift": 8,
    "test_expected_drift": 25,
    "test_successful_no_drift": 25,
    "test_unexpected_drift": 25,
    "test_unexpected_without_observable_log_drift": 25,
    "boundary_reject_holdout": 12,
}


LABEL_TO_STRATUM = {
    "expected_drift": "expected_log_visible",
    "successful_no_drift": "expected_log_subtle",
    "unexpected_drift": "unexpected_log_visible",
    "unexpected_without_observable_log_drift": "unexpected_log_subtle",
}


DEV_SCENARIOS = {
    "expected_drift": [
        "expected_workload_mix_route_heavy",
        "expected_workload_mix_train_station_heavy",
        "expected_workload_mix_price_heavy",
        "expected_scale_route_1_to_2",
        "expected_scale_station_1_to_2",
    ],
    "successful_no_drift": [
        "expected_low_impact_ui_annotation",
        "expected_low_impact_price_annotation",
        "expected_low_impact_route_annotation",
        "expected_low_impact_train_annotation",
        "expected_low_impact_station_annotation",
        "expected_compatible_config_price_pool_increase",
        "expected_compatible_config_route_timeout_valid",
    ],
    "unexpected_drift": [
        "unexpected_price_bad_db_port",
        "unexpected_route_bad_db_port",
        "unexpected_train_bad_db_port",
        "unexpected_price_scale_zero",
        "unexpected_resource_limit_price_too_small",
    ],
    "unexpected_without_observable_log_drift": [
        "unexpected_seat_scale_zero_weak",
        "unexpected_user_scale_zero_weak",
        "unexpected_basic_scale_zero_weak",
        "unexpected_config_scale_zero_weak",
        "unexpected_seat_scale_zero_weak",
        "unexpected_user_scale_zero_weak",
        "unexpected_basic_scale_zero_weak",
    ],
}


TEST_SCENARIOS = {
    "expected_drift": [
        "expected_workload_mix_route_heavy",
        "expected_workload_mix_train_station_heavy",
        "expected_workload_mix_price_heavy",
        "expected_scale_price_1_to_2",
        "expected_scale_route_1_to_2",
        "expected_scale_train_1_to_2",
        "expected_scale_station_1_to_2",
        "expected_pod_migration_price_replacement",
    ],
    "successful_no_drift": [
        "expected_low_impact_ui_annotation",
        "expected_low_impact_price_annotation",
        "expected_low_impact_route_annotation",
        "expected_low_impact_train_annotation",
        "expected_low_impact_station_annotation",
        "expected_compatible_config_price_pool_increase",
        "expected_compatible_config_route_timeout_valid",
    ],
    "unexpected_drift": [
        "unexpected_price_bad_db_port",
        "unexpected_route_bad_db_port",
        "unexpected_train_bad_db_port",
        "unexpected_price_scale_zero",
        "unexpected_route_scale_zero",
        "unexpected_train_scale_zero",
        "unexpected_resource_limit_price_too_small",
        "unexpected_connection_pool_exhaustion_price",
    ],
    "unexpected_without_observable_log_drift": [
        "unexpected_seat_scale_zero_weak",
        "unexpected_user_scale_zero_weak",
        "unexpected_basic_scale_zero_weak",
        "unexpected_config_scale_zero_weak",
        "unexpected_seat_scale_zero_weak",
        "unexpected_user_scale_zero_weak",
        "unexpected_basic_scale_zero_weak",
        "unexpected_config_scale_zero_weak",
        "unexpected_seat_scale_zero_weak",
        "unexpected_user_scale_zero_weak",
    ],
}


BOUNDARY_SCENARIOS = [
    "boundary_success_rate_slight_drop",
    "boundary_p99_latency_near_slo",
    "boundary_strong_logs_oracle_success",
    "boundary_weak_oracle_failure_weak_logs",
    "boundary_transition_not_stable",
    "boundary_resource_pressure_recovered_short",
    "boundary_success_rate_slight_drop",
    "boundary_p99_latency_near_slo",
    "boundary_strong_logs_oracle_success",
    "boundary_weak_oracle_failure_weak_logs",
    "boundary_transition_not_stable",
    "boundary_resource_pressure_recovered_short",
]


REFERENCE_PLAN = [
    ("R01", "baseline_no_change", "", 9901),
    ("R01", "noop_redeploy", "ts-price-service", 9901),
    ("R01", "baseline_no_change", "", 9901),
    ("R02", "baseline_no_change", "", 9902),
    ("R02", "noop_redeploy", "ts-route-service", 9902),
    ("R02", "baseline_no_change", "", 9902),
    ("R03", "baseline_no_change", "", 9903),
    ("R03", "noop_redeploy", "ts-train-service", 9903),
    ("R03", "baseline_no_change", "", 9903),
    ("R04", "baseline_no_change", "", 9904),
    ("R04", "noop_redeploy", "ts-station-service", 9904),
    ("R04", "baseline_no_change", "", 9904),
    ("R05", "baseline_no_change", "", 9905),
    ("R05", "noop_redeploy", "ts-user-service", 9905),
    ("R05", "baseline_no_change", "", 9905),
    ("R06", "baseline_no_change", "", 9906),
    ("R06", "noop_redeploy", "ts-seat-service", 9906),
    ("R06", "baseline_no_change", "", 9906),
    ("R07", "baseline_no_change", "", 9907),
    ("R07", "noop_redeploy", "ts-price-service", 9907),
    ("R07", "baseline_no_change", "", 9907),
]


FORMAL_FIELDS = list(v03.FIELDS) + [
    "source_dataset_id",
    "dataset_target_split",
    "current_run_count_for_label",
    "target_run_count_for_label",
    "v0_4_phase",
    "freeze_status",
]

CONTROL_FIELDS = [
    "ordinal",
    "batch_id",
    "cloud_protocol_version",
    "source_dataset_id",
    "dataset_target_split",
    "control_subset",
    "execution_block",
    "block_id",
    "batch_position",
    "preceding_scenario",
    "scenario",
    "noop_target_deployment",
    "matched_workload_seed",
    "seed",
    "workload_profile_id",
    "post_workload_profile_id",
    "run_id",
    "status",
    "run_type",
    "semantic_label",
    "benchmark_label",
    "scenario_id",
    "change_family_id",
    "implementation_id",
    "component_id",
    "change_target_component_id",
    "affected_component_ids",
    "oracle_component_ids",
    "target_train_reference_count",
    "current_train_reference_count",
    "threshold_sha256",
    "threshold_source",
    "cloud_freeze_id",
    "cluster_type",
    "planned_stop_rule",
    "detector_feedback_used",
    "notes",
]


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def meta_for(name: str) -> dict:
    if name in v03.SCENARIOS:
        return dict(v03.SCENARIOS[name])
    return dict(v03.BOUNDARY_SCENARIOS[name])


def block_name(kind: str, index: int) -> tuple[str, str]:
    if kind == "dev":
        block = "V4_dev_A" if index <= 12 else "V4_dev_B"
        return block, f"{BATCH_ID}_{block}"
    if kind == "test":
        if index <= 12:
            block = "V4_test_A"
        elif index <= 24:
            block = "V4_test_B"
        else:
            block = "V4_test_C"
        return block, f"{BATCH_ID}_{block}"
    block = "V4_boundary_holdout"
    return block, f"{BATCH_ID}_{block}"


def formal_row(
    *,
    ordinal: int,
    name: str,
    label: str,
    phase: str,
    phase_index: int,
    previous: str,
    seed: int,
) -> dict:
    meta = meta_for(name)
    stratum = meta["stratum"]
    execution_block, block_id = block_name(phase, phase_index)
    allow_formal = "false" if phase in {"dev", "boundary"} or stratum == "boundary_indeterminate" else "true"
    split_role = {
        "dev": "dev_tuning_candidate",
        "test": "formal_test_candidate",
        "boundary": "reject_boundary_holdout",
    }[phase]
    if phase == "boundary":
        target_label = "indeterminate_boundary"
        target_count = TARGET["boundary_reject_holdout"]
        current_count = 0
        dataset_target_split = "reject_holdout"
    else:
        target_label = label
        target_count = TARGET[f"{phase}_{label}"]
        current_count = CURRENT[f"{phase}_{label}"]
        dataset_target_split = phase
    run_id = f"{meta['slug']}_{BATCH_ID}_{phase}_r{phase_index:02d}"
    return {
        "ordinal": ordinal,
        "batch_id": BATCH_ID,
        "cloud_protocol_version": CLOUD_PROTOCOL_VERSION,
        "source_v0_2_batch_id": "cloud_expected_unexpected_formal_v0_2_20260704",
        "execution_block": execution_block,
        "block_id": block_id,
        "batch_position": phase_index,
        "preceding_scenario": previous,
        "scenario": name,
        "semantic_label": meta["semantic_label"],
        "target_label_tendency": meta.get("target_label_tendency", target_label),
        "target_benchmark_label": target_label,
        "v0_3_stratum": stratum,
        "v0_3_role": "v0_4_expansion",
        "existing_v0_2_valid_count_for_stratum": "",
        "target_cumulative_valid_total_for_stratum": "",
        "matched_workload_seed": seed,
        "seed": seed,
        "workload_profile_id": "W1_steady_core",
        "post_workload_profile_id": meta.get("post_workload_profile_id", "W1_steady_core"),
        "run_id": run_id,
        "status": "planned",
        "run_type": meta["run_type"],
        "scenario_id": meta["scenario_id"],
        "change_family_id": meta["change_family_id"],
        "implementation_id": meta["implementation_id"],
        "component_id": meta["component_id"],
        "change_target_component_id": meta["change_target_component_id"],
        "affected_component_ids": meta["affected_component_ids"],
        "oracle_component_ids": meta["oracle_component_ids"],
        "split_role": split_role,
        "allow_as_formal_test": allow_formal,
        "threshold_sha256": THRESHOLD_SHA256,
        "threshold_source": THRESHOLD_SOURCE,
        "cloud_freeze_id": CLOUD_FREEZE_ID,
        "cluster_type": CLUSTER_TYPE,
        "runner_support": meta["runner_support"],
        "execution_status": meta["execution_status"],
        "pre_run_requirement": meta["pre_run_requirement"],
        "oracle_extension_required": meta["oracle_extension_required"],
        "planned_stop_rule": (
            "collect_reference_first; then dev supplements; audit; then test supplements; "
            "boundary holdout only after runner/oracle freeze"
        ),
        "detector_feedback_used": "false",
        "notes": meta["notes"],
        "source_dataset_id": SOURCE_DATASET_ID,
        "dataset_target_split": dataset_target_split,
        "current_run_count_for_label": current_count,
        "target_run_count_for_label": target_count,
        "v0_4_phase": phase,
        "freeze_status": "draft_plan_not_collected",
    }


def control_rows() -> list[dict]:
    rows = []
    previous = "none"
    for ordinal, (block, scenario, target, seed) in enumerate(REFERENCE_PLAN, start=1):
        meta = control_metadata(scenario, target)
        target_part = target or "all"
        run_id = f"{scenario}_{target_part}_{BATCH_ID}_trainref_r{ordinal:02d}".replace("/", "_")
        rows.append({
            "ordinal": ordinal,
            "batch_id": f"{BATCH_ID}_reference",
            "cloud_protocol_version": CLOUD_PROTOCOL_VERSION,
            "source_dataset_id": SOURCE_DATASET_ID,
            "dataset_target_split": "train",
            "control_subset": "train_reference",
            "execution_block": "V4_reference",
            "block_id": f"{BATCH_ID}_{block}",
            "batch_position": ordinal,
            "preceding_scenario": previous,
            "scenario": scenario,
            "noop_target_deployment": target,
            "matched_workload_seed": seed,
            "seed": seed,
            "workload_profile_id": "W1_steady_core",
            "post_workload_profile_id": "W1_steady_core",
            "run_id": run_id,
            "status": "planned",
            **meta,
            "target_train_reference_count": TARGET["train_reference_runs"],
            "current_train_reference_count": CURRENT["train_reference_runs"],
            "threshold_sha256": THRESHOLD_SHA256,
            "threshold_source": THRESHOLD_SOURCE,
            "cloud_freeze_id": CLOUD_FREEZE_ID,
            "cluster_type": CLUSTER_TYPE,
            "planned_stop_rule": "run_reference_in_7_blocks; audit before adding to train memory",
            "detector_feedback_used": "false",
            "notes": "Additional normal/reference memory; not used to tune cloud drift gate thresholds.",
        })
        previous = scenario if not target else f"{scenario}:{target}"
    return rows


def formal_rows() -> tuple[list[dict], list[dict], list[dict], list[dict]]:
    rows = []
    dev_rows = []
    test_rows = []
    boundary_rows = []
    ordinal = 1
    previous = "none"
    seed = 10100
    phase_index = 1
    for label, names in DEV_SCENARIOS.items():
        for name in names:
            row = formal_row(
                ordinal=ordinal,
                name=name,
                label=label,
                phase="dev",
                phase_index=phase_index,
                previous=previous,
                seed=seed + ordinal,
            )
            rows.append(row)
            dev_rows.append(row)
            previous = name
            ordinal += 1
            phase_index += 1
    phase_index = 1
    for label, names in TEST_SCENARIOS.items():
        for name in names:
            row = formal_row(
                ordinal=ordinal,
                name=name,
                label=label,
                phase="test",
                phase_index=phase_index,
                previous=previous,
                seed=seed + ordinal,
            )
            rows.append(row)
            test_rows.append(row)
            previous = name
            ordinal += 1
            phase_index += 1
    phase_index = 1
    for name in BOUNDARY_SCENARIOS:
        row = formal_row(
            ordinal=ordinal,
            name=name,
            label="indeterminate_boundary",
            phase="boundary",
            phase_index=phase_index,
            previous=previous,
            seed=seed + ordinal,
        )
        rows.append(row)
        boundary_rows.append(row)
        previous = name
        ordinal += 1
        phase_index += 1
    return rows, dev_rows, test_rows, boundary_rows


def write_summary(path: Path, controls: list[dict], dev: list[dict], test: list[dict], boundary: list[dict]) -> None:
    dev_counts = Counter(row["target_benchmark_label"] for row in dev)
    test_counts = Counter(row["target_benchmark_label"] for row in test)
    boundary_status = Counter(row["execution_status"] for row in boundary)
    control_counts = Counter(row["benchmark_label"] for row in controls)
    test_after = {
        "expected_drift": CURRENT["test_expected_drift"] + test_counts["expected_drift"],
        "successful_no_drift": CURRENT["test_successful_no_drift"] + test_counts["successful_no_drift"],
        "unexpected_drift": CURRENT["test_unexpected_drift"] + test_counts["unexpected_drift"],
        "unexpected_without_observable_log_drift": (
            CURRENT["test_unexpected_without_observable_log_drift"]
            + test_counts["unexpected_without_observable_log_drift"]
        ),
    }
    text = f"""# Cloud Expansion v0.4 Plan

Generated at: {datetime.now(timezone.utc).isoformat()}

This plan expands the combined67 cloud benchmark without changing the frozen
drift gate threshold. It prioritizes reviewer-risk items in this order:
normal/reference memory, dev coverage, formal test size, and separate
boundary/reject probes.

## Current Baseline

| Dataset part | current run count |
|---|---:|
| Train normal/reference | {CURRENT['train_reference_runs']} |
| Dev labelled Expected/Unexpected | 8 |
| Dev controls | 6 |
| Formal test | 67 |

Formal test labels now: expected_drift=17, successful_no_drift=18,
unexpected_drift=17, unexpected_without_observable_log_drift=15.

## Planned Additions

| Addition | planned runs | Purpose |
|---|---:|---|
| Train reference Baseline/No-op | {len(controls)} | raise normal memory from 9 to 30 runs |
| Dev labelled supplements | {len(dev)} | raise each dev label stratum to 8 runs |
| Formal test supplements | {len(test)} | raise formal test to 100 runs, about 25 per label |
| Boundary/reject holdout | {len(boundary)} | evaluate Reject separately after runner/oracle extension |

Reference controls: baseline_normal={control_counts['baseline_normal']},
no_op_control={control_counts['no_op_control']}.

Dev supplement labels: {dict(dev_counts)}.

Test supplement labels: {dict(test_counts)}.

Projected formal test labels after v0.4 test supplement: {test_after}.

Boundary execution status: {dict(boundary_status)}.

## Execution Order

1. Run reference controls in 7 small blocks and audit them before using them as
   train memory.
2. Run dev supplement in two blocks and audit label/quality distribution.
3. Run test supplement in three blocks. Do not tune thresholds or model rules
   on these results.
4. Run boundary/reject holdout only after a separate runner/oracle freeze.

## Fixed Rules

- Threshold SHA256 remains `{THRESHOLD_SHA256}`.
- Cloud freeze id remains `{CLOUD_FREEZE_ID}` unless runner/oracle changes are
  made; boundary/reject rows require a new freeze before execution.
- Detector feedback is not used for row selection.
- Valid rows are retained even if the final drift gate outcome differs from the
  target tendency.
- Run-level metrics remain primary; window counts are auxiliary only.
"""
    path.write_text(text, encoding="utf-8")


def write_protocol(path: Path, controls: list[dict], formal: list[dict], output_files: dict[str, str]) -> None:
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": BATCH_ID,
        "cloud_protocol_version": CLOUD_PROTOCOL_VERSION,
        "source_dataset_id": SOURCE_DATASET_ID,
        "threshold_sha256": THRESHOLD_SHA256,
        "threshold_source": THRESHOLD_SOURCE,
        "cloud_freeze_id": CLOUD_FREEZE_ID,
        "cluster_type": CLUSTER_TYPE,
        "current_counts": CURRENT,
        "target_counts": TARGET,
        "planned_counts": {
            "reference_controls": len(controls),
            "formal_dev_test_boundary": len(formal),
            "dev_supplement": sum(1 for row in formal if row["v0_4_phase"] == "dev"),
            "test_supplement": sum(1 for row in formal if row["v0_4_phase"] == "test"),
            "boundary_holdout": sum(1 for row in formal if row["v0_4_phase"] == "boundary"),
        },
        "output_files": output_files,
        "file_sha256": {
            name: sha256_file(Path(file_path))
            for name, file_path in output_files.items()
            if Path(file_path).exists()
        },
        "excluded_from_plan": {
            "unexpected_service_kill_replacement_low_oov": (
                "Excluded from formal v0.4 because prior smoke evidence showed unstable semantic landing."
            )
        },
        "execution_policy": [
            "Do not run all v0.4 rows as one unattended batch.",
            "Collect reference controls first; audit before updating training memory.",
            "Collect dev supplements before test supplements.",
            "Boundary/reject holdout requires separate runner/oracle freeze.",
            "No detector-feedback-based row selection.",
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/scwarn_pilot/collections/cloud_expansion_v0_4_20260709",
    )
    args = parser.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    controls = control_rows()
    formal, dev, test, boundary = formal_rows()

    reference_path = out / "cloud_expansion_v0_4_reference_train_plan.csv"
    dev_path = out / "cloud_expansion_v0_4_dev_supplement_plan.csv"
    test_path = out / "cloud_expansion_v0_4_test_supplement_plan.csv"
    boundary_path = out / "cloud_expansion_v0_4_boundary_reject_plan.csv"
    formal_path = out / "cloud_expansion_v0_4_formal_dev_test_boundary_plan.csv"
    summary_path = out / "cloud_expansion_v0_4_summary.md"
    protocol_path = out / "cloud_expansion_v0_4_protocol.json"

    write_csv(reference_path, controls, CONTROL_FIELDS)
    write_csv(dev_path, dev, FORMAL_FIELDS)
    write_csv(test_path, test, FORMAL_FIELDS)
    write_csv(boundary_path, boundary, FORMAL_FIELDS)
    write_csv(formal_path, formal, FORMAL_FIELDS)
    write_summary(summary_path, controls, dev, test, boundary)
    output_files = {
        "reference_plan": str(reference_path),
        "dev_supplement_plan": str(dev_path),
        "test_supplement_plan": str(test_path),
        "boundary_reject_plan": str(boundary_path),
        "formal_combined_plan": str(formal_path),
        "summary": str(summary_path),
    }
    write_protocol(protocol_path, controls, formal, output_files)

    print(json.dumps({
        "output_dir": str(out),
        "reference_runs": len(controls),
        "dev_runs": len(dev),
        "test_runs": len(test),
        "boundary_runs": len(boundary),
        "protocol": str(protocol_path),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
