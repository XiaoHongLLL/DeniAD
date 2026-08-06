#!/usr/bin/env python3
"""Create a preregistered Tencent Cloud Baseline/No-op control plan."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


PLAN = [
    # block, scenario, target, subset, seed
    ("cloud_ctrl_b01", "baseline_no_change", "", "calibration", 7401),
    ("cloud_ctrl_b01", "noop_redeploy", "ts-price-service", "calibration", 7401),
    ("cloud_ctrl_b01", "baseline_no_change", "", "validation", 7401),
    ("cloud_ctrl_b02", "baseline_no_change", "", "calibration", 7402),
    ("cloud_ctrl_b02", "baseline_no_change", "", "validation", 7402),
    ("cloud_ctrl_b02", "noop_redeploy", "ts-route-service", "calibration", 7402),
    ("cloud_ctrl_b03", "noop_redeploy", "ts-train-service", "validation", 7403),
    ("cloud_ctrl_b03", "baseline_no_change", "", "calibration", 7403),
    ("cloud_ctrl_b03", "baseline_no_change", "", "calibration", 7403),
    ("cloud_ctrl_b04", "baseline_no_change", "", "validation", 7404),
    ("cloud_ctrl_b04", "noop_redeploy", "ts-price-service", "calibration", 7404),
    ("cloud_ctrl_b04", "baseline_no_change", "", "calibration", 7404),
    ("cloud_ctrl_b05", "baseline_no_change", "", "calibration", 7405),
    ("cloud_ctrl_b05", "noop_redeploy", "ts-station-service", "validation", 7405),
    ("cloud_ctrl_b05", "baseline_no_change", "", "validation", 7405),
]


def scenario_metadata(scenario: str, target: str) -> dict[str, str]:
    if scenario == "baseline_no_change":
        return {
            "run_type": "baseline",
            "semantic_label": "expected",
            "benchmark_label": "baseline_normal",
            "scenario_id": "cloud_baseline_w1_accesslog",
            "change_family_id": "baseline",
            "implementation_id": "cloud_baseline_no_change_v0_1",
            "component_id": "all",
            "change_target_component_id": "none",
            "affected_component_ids": "all-observed-services",
            "oracle_component_ids": "ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service",
        }
    safe_target = target.replace("ts-", "").replace("-service", "").replace("-", "_")
    return {
        "run_type": "no_op",
        "semantic_label": "expected",
        "benchmark_label": "no_op_control",
        "scenario_id": f"cloud_noop_rollout_{safe_target}",
        "change_family_id": "noop_redeployment",
        "implementation_id": f"{target}_rollout_restart_same_spec_cloud_v0_1",
        "component_id": target,
        "change_target_component_id": target,
        "affected_component_ids": f"{target},ts-gateway-service",
        "oracle_component_ids": f"ts-gateway-service,{target}",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    preceding = "none"
    for ordinal, (block_id, scenario, target, subset, seed) in enumerate(PLAN, start=1):
        meta = scenario_metadata(scenario, target)
        target_part = target or "all"
        run_id = f"{scenario}_{target_part}_{args.batch_id}_r{ordinal:02d}".replace("/", "_")
        rows.append({
            "ordinal": ordinal,
            "batch_id": args.batch_id,
            "cloud_protocol_version": "linux-multinode-port-v0.1",
            "control_subset": subset,
            "block_id": block_id,
            "batch_position": ordinal,
            "preceding_scenario": preceding,
            "scenario": scenario,
            "noop_target_deployment": target,
            "matched_workload_seed": seed,
            "seed": seed,
            "workload_profile_id": "W1_steady_core",
            "post_workload_profile_id": "W1_steady_core",
            "run_id": run_id,
            "status": "planned",
            **meta,
        })
        preceding = scenario if not target else f"{scenario}:{target}"

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"cloud control plan written: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
