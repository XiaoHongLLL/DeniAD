#!/usr/bin/env python3
"""Create a stratified v0.3 formal supplement plan for Tencent Cloud TT-Core-16.

v0.2 already produced 36 valid formal runs. This plan is a supplement, not a
replacement: it targets a cumulative run-level benchmark with about 72 main
Expected/Unexpected runs plus 12 boundary/indeterminate Reject probes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


BATCH_ID = "cloud_expected_unexpected_formal_v0_3_20260707"
CLOUD_PROTOCOL_VERSION = "linux-cloud-formal-v0.3-draft"
SOURCE_V0_2_BATCH_ID = "cloud_expected_unexpected_formal_v0_2_20260704"
THRESHOLD_SHA256 = "99b2a27a4f6192b86ca4ea83a550be26e5605b8254ba0511d52ced3fa7584c16"
THRESHOLD_SOURCE = "cloud_controls_calibration_only"
CLOUD_FREEZE_ID = "1e09d8d1edc77982565a9b66066ee6289feca7d9d57b51911546f1cca9ea804f"
CLUSTER_TYPE = "linux_kubeadm_multinode"

EXISTING_V0_2_COUNTS = {
    "expected_log_visible": 8,
    "expected_log_subtle": 11,
    "unexpected_log_visible": 9,
    "unexpected_log_subtle": 8,
    "boundary_indeterminate": 0,
}

TARGET_CUMULATIVE_COUNTS = {
    "expected_log_visible": 18,
    "expected_log_subtle": 18,
    "unexpected_log_visible": 18,
    "unexpected_log_subtle": 18,
    "boundary_indeterminate": 12,
}

FIELDS = [
    "ordinal",
    "batch_id",
    "cloud_protocol_version",
    "source_v0_2_batch_id",
    "execution_block",
    "block_id",
    "batch_position",
    "preceding_scenario",
    "scenario",
    "semantic_label",
    "target_label_tendency",
    "target_benchmark_label",
    "v0_3_stratum",
    "v0_3_role",
    "existing_v0_2_valid_count_for_stratum",
    "target_cumulative_valid_total_for_stratum",
    "matched_workload_seed",
    "seed",
    "workload_profile_id",
    "post_workload_profile_id",
    "run_id",
    "status",
    "run_type",
    "scenario_id",
    "change_family_id",
    "implementation_id",
    "component_id",
    "change_target_component_id",
    "affected_component_ids",
    "oracle_component_ids",
    "split_role",
    "allow_as_formal_test",
    "threshold_sha256",
    "threshold_source",
    "cloud_freeze_id",
    "cluster_type",
    "runner_support",
    "execution_status",
    "pre_run_requirement",
    "oracle_extension_required",
    "planned_stop_rule",
    "detector_feedback_used",
    "notes",
]


SCENARIOS = {
    # Expected log-visible. Some rows are existing runner scenarios; the new
    # multinode/version/config variants are intentionally marked as extension
    # work so the plan cannot be mistaken for immediately executable code.
    "expected_workload_mix_route_heavy": {
        "stratum": "expected_log_visible",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "expected_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_workload_mix_w1_to_route_heavy",
        "change_family_id": "expected_workload_mix",
        "implementation_id": "w1_to_route_query_heavy_cloud_v2",
        "component_id": "workload_profile",
        "change_target_component_id": "workload-generator",
        "affected_component_ids": "ts-gateway-service,ts-route-service,ts-station-service",
        "oracle_component_ids": "ts-gateway-service,ts-route-service,ts-station-service",
        "post_workload_profile_id": "W1_route_query_heavy",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "exp_route_heavy",
        "notes": "Operational Expected Drift; retained for continuity but not overused.",
    },
    "expected_workload_mix_train_station_heavy": {
        "stratum": "expected_log_visible",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "expected_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_workload_mix_w1_to_train_station_heavy",
        "change_family_id": "expected_workload_mix",
        "implementation_id": "w1_to_train_station_query_heavy_cloud_v2",
        "component_id": "workload_profile",
        "change_target_component_id": "workload-generator",
        "affected_component_ids": "ts-gateway-service,ts-train-service,ts-station-service",
        "oracle_component_ids": "ts-gateway-service,ts-train-service,ts-station-service",
        "post_workload_profile_id": "W1_train_station_query_heavy",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "exp_train_station_heavy",
        "notes": "Operational Expected Drift; retained for continuity but not overused.",
    },
    "expected_workload_mix_price_heavy": {
        "stratum": "expected_log_visible",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "expected_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_workload_mix_w1_to_price_heavy",
        "change_family_id": "expected_workload_mix",
        "implementation_id": "w1_to_price_query_heavy_cloud_v1",
        "component_id": "workload_profile",
        "change_target_component_id": "workload-generator",
        "affected_component_ids": "ts-gateway-service,ts-price-service,ts-route-service,ts-train-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service,ts-route-service,ts-train-service",
        "post_workload_profile_id": "W1_price_query_heavy",
        "runner_support": "formal_runner_v0.3",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none; workload profile already exists",
        "oracle_extension_required": "false",
        "slug": "exp_price_heavy",
        "notes": "Adds a third workload-mix direction without changing semantics.",
    },
    "expected_scale_price_1_to_2": {
        "stratum": "expected_log_visible",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "expected_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_price_scale_1_to_2",
        "change_family_id": "expected_resource_scale",
        "implementation_id": "price_replicas_1_to_2_cloud_v2",
        "component_id": "ts-price-service",
        "change_target_component_id": "ts-price-service",
        "affected_component_ids": "ts-price-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "exp_scale_price",
        "notes": "Expected replica scale-out.",
    },
    "expected_scale_route_1_to_2": {
        "stratum": "expected_log_visible",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "expected_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_route_scale_1_to_2",
        "change_family_id": "expected_resource_scale",
        "implementation_id": "route_replicas_1_to_2_cloud_v2",
        "component_id": "ts-route-service",
        "change_target_component_id": "ts-route-service",
        "affected_component_ids": "ts-route-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-route-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "exp_scale_route",
        "notes": "Expected replica scale-out.",
    },
    "expected_scale_train_1_to_2": {
        "stratum": "expected_log_visible",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "expected_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_train_scale_1_to_2",
        "change_family_id": "expected_resource_scale",
        "implementation_id": "train_replicas_1_to_2_cloud_v2",
        "component_id": "ts-train-service",
        "change_target_component_id": "ts-train-service",
        "affected_component_ids": "ts-train-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-train-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "exp_scale_train",
        "notes": "Expected replica scale-out.",
    },
    "expected_scale_station_1_to_2": {
        "stratum": "expected_log_visible",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "expected_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_station_scale_1_to_2",
        "change_family_id": "expected_resource_scale",
        "implementation_id": "station_replicas_1_to_2_cloud_v2",
        "component_id": "ts-station-service",
        "change_target_component_id": "ts-station-service",
        "affected_component_ids": "ts-station-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-station-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "exp_scale_station",
        "notes": "Expected replica scale-out.",
    },
    "expected_pod_migration_price_replacement": {
        "stratum": "expected_log_visible",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "expected_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_price_pod_replacement",
        "change_family_id": "expected_pod_migration",
        "implementation_id": "delete_one_price_pod_replicas_2_cloud_v1",
        "component_id": "ts-price-service",
        "change_target_component_id": "ts-price-service",
        "affected_component_ids": "ts-price-service,ts-gateway-service,node",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "formal_runner_v0.3_needs_smoke",
        "execution_status": "implemented_needs_cluster_smoke",
        "pre_run_requirement": "smoke-test pod delete selector; record old/new pod and node; require oracle success",
        "oracle_extension_required": "false",
        "slug": "exp_pod_migrate_price",
        "notes": "Expected Kubernetes replacement/migration, not just workload mix.",
    },
    "expected_node_drain_failover_price_redundant": {
        "stratum": "expected_log_visible",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "expected_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_node_drain_price_redundant",
        "change_family_id": "expected_node_drain_failover",
        "implementation_id": "drain_worker_with_price_redundancy_cloud_v1",
        "component_id": "worker-node",
        "change_target_component_id": "worker-node",
        "affected_component_ids": "ts-price-service,ts-gateway-service,node",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "needs_runner_extension",
        "execution_status": "needs_runner_extension",
        "pre_run_requirement": "cordon/drain/uncordon workflow with redundant target replicas",
        "oracle_extension_required": "false",
        "slug": "exp_node_drain_redundant",
        "notes": "Expected multinode failover case.",
    },
    "expected_compatible_config_price_pool_increase": {
        "stratum": "expected_log_subtle",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "successful_no_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_price_pool_increase",
        "change_family_id": "expected_compatible_config",
        "implementation_id": "price_hikari_pool_valid_increase_cloud_v1",
        "component_id": "ts-price-service",
        "change_target_component_id": "ts-price-service",
        "affected_component_ids": "ts-price-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "formal_runner_v0.3_needs_smoke",
        "execution_status": "implemented_needs_cluster_smoke",
        "pre_run_requirement": "smoke-test Spring/Hikari env mapping and confirm semantic success",
        "oracle_extension_required": "false",
        "slug": "exp_price_pool_valid",
        "notes": "Compatible connection-pool adjustment; target is semantic success with weak log shift.",
    },
    "expected_compatible_config_route_timeout_valid": {
        "stratum": "expected_log_subtle",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "successful_no_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_route_timeout_valid",
        "change_family_id": "expected_compatible_config",
        "implementation_id": "route_client_timeout_valid_cloud_v1",
        "component_id": "ts-route-service",
        "change_target_component_id": "ts-route-service",
        "affected_component_ids": "ts-route-service",
        "oracle_component_ids": "ts-gateway-service,ts-route-service",
        "runner_support": "formal_runner_v0.3_needs_smoke",
        "execution_status": "implemented_needs_cluster_smoke",
        "pre_run_requirement": "smoke-test compatible Hikari timeout env mapping and confirm semantic success",
        "oracle_extension_required": "false",
        "slug": "exp_route_timeout_valid",
        "notes": "Compatible config change distinct from annotation.",
    },
    "expected_low_impact_ui_annotation": {
        "stratum": "expected_log_subtle",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "successful_no_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_low_impact_ui_annotation",
        "change_family_id": "expected_low_impact_config",
        "implementation_id": "ui_dashboard_metadata_annotation_cloud_v2",
        "component_id": "ts-ui-dashboard",
        "change_target_component_id": "ts-ui-dashboard",
        "affected_component_ids": "ts-ui-dashboard",
        "oracle_component_ids": "ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "exp_li_ui",
        "notes": "Low-impact metadata/config control.",
    },
    "expected_low_impact_price_annotation": {
        "stratum": "expected_log_subtle",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "successful_no_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_low_impact_price_annotation",
        "change_family_id": "expected_low_impact_config",
        "implementation_id": "price_metadata_annotation_cloud_v2",
        "component_id": "ts-price-service",
        "change_target_component_id": "ts-price-service",
        "affected_component_ids": "ts-price-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "exp_li_price",
        "notes": "Low-impact metadata/config control.",
    },
    "expected_low_impact_route_annotation": {
        "stratum": "expected_log_subtle",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "successful_no_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_low_impact_route_annotation",
        "change_family_id": "expected_low_impact_config",
        "implementation_id": "route_metadata_annotation_cloud_v2",
        "component_id": "ts-route-service",
        "change_target_component_id": "ts-route-service",
        "affected_component_ids": "ts-route-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "exp_li_route",
        "notes": "Low-impact metadata/config control.",
    },
    "expected_low_impact_train_annotation": {
        "stratum": "expected_log_subtle",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "successful_no_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_low_impact_train_annotation",
        "change_family_id": "expected_low_impact_config",
        "implementation_id": "train_metadata_annotation_cloud_v2",
        "component_id": "ts-train-service",
        "change_target_component_id": "ts-train-service",
        "affected_component_ids": "ts-train-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "exp_li_train",
        "notes": "Low-impact metadata/config control.",
    },
    "expected_low_impact_station_annotation": {
        "stratum": "expected_log_subtle",
        "role": "primary",
        "semantic_label": "expected",
        "target_label_tendency": "successful_no_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_low_impact_station_annotation",
        "change_family_id": "expected_low_impact_config",
        "implementation_id": "station_metadata_annotation_cloud_v2",
        "component_id": "ts-station-service",
        "change_target_component_id": "ts-station-service",
        "affected_component_ids": "ts-station-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "exp_li_station",
        "notes": "Low-impact metadata/config control.",
    },
    "expected_compatible_image_price_patch": {
        "stratum": "expected_log_subtle",
        "role": "reserve",
        "semantic_label": "expected",
        "target_label_tendency": "successful_no_drift",
        "run_type": "expected",
        "scenario_id": "cloud_expected_price_compatible_image_patch",
        "change_family_id": "expected_compatible_version",
        "implementation_id": "price_compatible_image_digest_cloud_v1",
        "component_id": "ts-price-service",
        "change_target_component_id": "ts-price-service",
        "affected_component_ids": "ts-price-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "needs_image_digest_pair",
        "execution_status": "needs_preverified_image",
        "pre_run_requirement": "build or select compatible image digest; record before/after digests",
        "oracle_extension_required": "false",
        "slug": "exp_price_image_patch",
        "notes": "Compatible image/version update reserve scenario.",
    },
    # Unexpected log-visible.
    "unexpected_price_bad_db_port": {
        "stratum": "unexpected_log_visible",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_price_wrong_db_port",
        "change_family_id": "unexpected_bad_port",
        "implementation_id": "price_wrong_db_port_3399_cloud_v2",
        "component_id": "ts-price-service",
        "change_target_component_id": "ts-price-service",
        "affected_component_ids": "ts-price-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "unexp_price_bad_port",
        "notes": "Hard dependency-port fault retained as anchor.",
    },
    "unexpected_route_bad_db_port": {
        "stratum": "unexpected_log_visible",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_route_wrong_db_port",
        "change_family_id": "unexpected_bad_port",
        "implementation_id": "route_wrong_db_port_cloud_v1",
        "component_id": "ts-route-service",
        "change_target_component_id": "ts-route-service",
        "affected_component_ids": "ts-route-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-route-service",
        "runner_support": "formal_runner_v0.3",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none; ROUTE_MYSQL_PORT verified in local application.yml",
        "oracle_extension_required": "false",
        "slug": "unexp_route_bad_port",
        "notes": "Wrong DB/service port on a second component.",
    },
    "unexpected_train_bad_db_port": {
        "stratum": "unexpected_log_visible",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_train_wrong_db_port",
        "change_family_id": "unexpected_bad_port",
        "implementation_id": "train_wrong_db_port_cloud_v1",
        "component_id": "ts-train-service",
        "change_target_component_id": "ts-train-service",
        "affected_component_ids": "ts-train-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-train-service",
        "runner_support": "formal_runner_v0.3",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none; TRAIN_MYSQL_PORT verified in local application.yml",
        "oracle_extension_required": "false",
        "slug": "unexp_train_bad_port",
        "notes": "Wrong DB/service port on a third component.",
    },
    "unexpected_price_scale_zero": {
        "stratum": "unexpected_log_visible",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_price_scale_to_zero",
        "change_family_id": "unexpected_service_termination",
        "implementation_id": "price_replicas_1_to_0_cloud_v2",
        "component_id": "ts-price-service",
        "change_target_component_id": "ts-price-service",
        "affected_component_ids": "ts-price-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "unexp_price_zero",
        "notes": "Hard service termination.",
    },
    "unexpected_route_scale_zero": {
        "stratum": "unexpected_log_visible",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_route_scale_to_zero",
        "change_family_id": "unexpected_service_termination",
        "implementation_id": "route_replicas_1_to_0_cloud_v2",
        "component_id": "ts-route-service",
        "change_target_component_id": "ts-route-service",
        "affected_component_ids": "ts-route-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-route-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "unexp_route_zero",
        "notes": "Hard service termination.",
    },
    "unexpected_train_scale_zero": {
        "stratum": "unexpected_log_visible",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_train_scale_to_zero",
        "change_family_id": "unexpected_service_termination",
        "implementation_id": "train_replicas_1_to_0_cloud_v2",
        "component_id": "ts-train-service",
        "change_target_component_id": "ts-train-service",
        "affected_component_ids": "ts-train-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-train-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "unexp_train_zero",
        "notes": "Hard service termination.",
    },
    "unexpected_resource_limit_price_too_small": {
        "stratum": "unexpected_log_visible",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_price_resource_limit_small",
        "change_family_id": "unexpected_resource_limit",
        "implementation_id": "price_cpu_memory_too_small_cloud_v1",
        "component_id": "ts-price-service",
        "change_target_component_id": "ts-price-service",
        "affected_component_ids": "ts-price-service,ts-gateway-service,node",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "formal_runner_v0.3_needs_smoke",
        "execution_status": "implemented_needs_cluster_smoke",
        "pre_run_requirement": "smoke-test resource patch and record OOMKilled/throttling/readiness evidence",
        "oracle_extension_required": "metrics_or_k8s_state_recommended",
        "slug": "unexp_price_limit_small",
        "notes": "Resource limit fault, not just service scale-to-zero.",
    },
    "unexpected_connection_pool_exhaustion_price": {
        "stratum": "unexpected_log_visible",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_price_pool_exhaustion",
        "change_family_id": "unexpected_pool_exhaustion",
        "implementation_id": "price_pool_too_small_cloud_v1",
        "component_id": "ts-price-service",
        "change_target_component_id": "ts-price-service",
        "affected_component_ids": "ts-price-service,ts-gateway-service,tsdb-mysql",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "formal_runner_v0.3_needs_smoke",
        "execution_status": "implemented_needs_cluster_smoke",
        "pre_run_requirement": "smoke-test Hikari pool exhaustion env mapping and failure mode",
        "oracle_extension_required": "false",
        "slug": "unexp_price_pool_exhaust",
        "notes": "Connection pool exhaustion fault.",
    },
    "unexpected_slow_path_route_latency_slo": {
        "stratum": "unexpected_log_visible",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_route_latency_slo",
        "change_family_id": "unexpected_slow_path",
        "implementation_id": "route_program_delay_or_tc_cloud_v1",
        "component_id": "ts-route-service",
        "change_target_component_id": "ts-route-service",
        "affected_component_ids": "ts-route-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-route-service",
        "runner_support": "needs_latency_injector",
        "execution_status": "needs_runner_extension",
        "pre_run_requirement": "choose tc/Litmus/code-delay injector and direct SLO source",
        "oracle_extension_required": "direct_latency_slo_required",
        "slug": "unexp_route_slow",
        "notes": "Slow path / latency SLO violation.",
    },
    # Unexpected log-subtle.
    "unexpected_seat_scale_zero_weak": {
        "stratum": "unexpected_log_subtle",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_without_observable_log_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_seat_scale_to_zero_weak",
        "change_family_id": "unexpected_low_coverage_service_loss",
        "implementation_id": "seat_replicas_1_to_0_cloud_v2",
        "component_id": "ts-seat-service",
        "change_target_component_id": "ts-seat-service",
        "affected_component_ids": "ts-seat-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-seat-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "unexp_seat_zero_weak",
        "notes": "Low-coverage service loss with weak log shift.",
    },
    "unexpected_user_scale_zero_weak": {
        "stratum": "unexpected_log_subtle",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_without_observable_log_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_user_scale_to_zero_weak",
        "change_family_id": "unexpected_low_coverage_service_loss",
        "implementation_id": "user_replicas_1_to_0_cloud_v2",
        "component_id": "ts-user-service",
        "change_target_component_id": "ts-user-service",
        "affected_component_ids": "ts-user-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-user-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "unexp_user_zero_weak",
        "notes": "Low-coverage service loss with weak log shift.",
    },
    "unexpected_basic_scale_zero_weak": {
        "stratum": "unexpected_log_subtle",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_without_observable_log_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_basic_scale_to_zero_weak",
        "change_family_id": "unexpected_low_coverage_service_loss",
        "implementation_id": "basic_replicas_1_to_0_cloud_v2",
        "component_id": "ts-basic-service",
        "change_target_component_id": "ts-basic-service",
        "affected_component_ids": "ts-basic-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-basic-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "unexp_basic_zero_weak",
        "notes": "Low-coverage service loss with weak log shift.",
    },
    "unexpected_config_scale_zero_weak": {
        "stratum": "unexpected_log_subtle",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_without_observable_log_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_config_scale_to_zero_weak",
        "change_family_id": "unexpected_low_coverage_service_loss",
        "implementation_id": "config_replicas_1_to_0_cloud_v2",
        "component_id": "ts-config-service",
        "change_target_component_id": "ts-config-service",
        "affected_component_ids": "ts-config-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-config-service",
        "runner_support": "formal_runner_v0.1",
        "execution_status": "ready_existing_runner",
        "pre_run_requirement": "none",
        "oracle_extension_required": "false",
        "slug": "unexp_config_zero_weak",
        "notes": "Low-coverage service loss with weak log shift.",
    },
    "unexpected_service_kill_replacement_low_oov": {
        "stratum": "boundary_indeterminate",
        "role": "primary",
        "semantic_label": "indeterminate",
        "target_label_tendency": "boundary_for_reject",
        "target_benchmark_label": "indeterminate_boundary",
        "run_type": "boundary",
        "scenario_id": "cloud_boundary_price_pod_kill_replacement_low_oov",
        "change_family_id": "boundary_runtime_replacement",
        "implementation_id": "delete_price_pod_single_replica_boundary_cloud_v1",
        "component_id": "ts-price-service",
        "change_target_component_id": "ts-price-service",
        "affected_component_ids": "ts-price-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "needs_boundary_stabilization",
        "execution_status": "protocol_development_unstable",
        "pre_run_requirement": (
            "smoke showed seed-sensitive semantic landing "
            "(indeterminate in one run, unexpected in another); do not use in formal v0.3"
        ),
        "oracle_extension_required": "stabilization_required",
        "slug": "bd_pod_kill_low_oov",
        "notes": "Runtime replacement is useful protocol evidence but is not stable enough for formal labels yet.",
    },
    "unexpected_partial_dependency_low_coverage_seat": {
        "stratum": "unexpected_log_subtle",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_without_observable_log_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_seat_partial_dependency",
        "change_family_id": "unexpected_partial_dependency_failure",
        "implementation_id": "seat_dependency_partial_loss_cloud_v1",
        "component_id": "ts-seat-service",
        "change_target_component_id": "ts-seat-service",
        "affected_component_ids": "ts-seat-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-seat-service",
        "runner_support": "needs_fault_injector",
        "execution_status": "needs_runner_extension",
        "pre_run_requirement": "define partial dependency and external Oracle failure condition",
        "oracle_extension_required": "false",
        "slug": "unexp_seat_partial_dep",
        "notes": "Partial dependency failure, expected to be log-subtle.",
    },
    "unexpected_slow_path_low_rate_latency": {
        "stratum": "unexpected_log_subtle",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_without_observable_log_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_low_rate_latency",
        "change_family_id": "unexpected_slow_path",
        "implementation_id": "low_rate_latency_slo_cloud_v1",
        "component_id": "ts-route-service",
        "change_target_component_id": "ts-route-service",
        "affected_component_ids": "ts-route-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-route-service",
        "runner_support": "needs_latency_injector",
        "execution_status": "needs_runner_extension",
        "pre_run_requirement": "direct latency SLO source required; port-forward latency cannot be used",
        "oracle_extension_required": "direct_latency_slo_required",
        "slug": "unexp_low_rate_slow",
        "notes": "Latency failure with weak log-distribution shift.",
    },
    "unexpected_node_resource_contention_weak": {
        "stratum": "unexpected_log_subtle",
        "role": "primary",
        "semantic_label": "unexpected",
        "target_label_tendency": "unexpected_without_observable_log_drift",
        "run_type": "unexpected",
        "scenario_id": "cloud_unexpected_node_contention_weak",
        "change_family_id": "unexpected_node_resource_contention",
        "implementation_id": "cpu_stress_worker_low_rate_cloud_v1",
        "component_id": "worker-node",
        "change_target_component_id": "worker-node",
        "affected_component_ids": "ts-price-service,ts-gateway-service,node",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "needs_node_stress_injector",
        "execution_status": "needs_runner_extension",
        "pre_run_requirement": "stress-ng or Litmus injector; record metrics/k8s pressure evidence",
        "oracle_extension_required": "metrics_or_k8s_state_recommended",
        "slug": "unexp_node_contention_weak",
        "notes": "Node-level weak-log fault.",
    },
}


BOUNDARY_SCENARIOS = {
    "boundary_success_rate_slight_drop": {
        "stratum": "boundary_indeterminate",
        "role": "primary",
        "semantic_label": "indeterminate",
        "target_label_tendency": "boundary_for_reject",
        "target_benchmark_label": "indeterminate_boundary",
        "run_type": "boundary",
        "scenario_id": "cloud_boundary_success_rate_slight_drop",
        "change_family_id": "boundary_semantic_threshold",
        "implementation_id": "success_rate_near_failure_threshold_cloud_v1",
        "component_id": "workload_profile",
        "change_target_component_id": "workload-generator",
        "affected_component_ids": "ts-gateway-service,ts-user-service",
        "oracle_component_ids": "ts-gateway-service,ts-user-service",
        "runner_support": "needs_boundary_oracle",
        "execution_status": "needs_oracle_extension",
        "pre_run_requirement": "freeze indeterminate success-rate band before execution",
        "oracle_extension_required": "true",
        "slug": "bd_success_slight_drop",
        "notes": "API success rate degrades but remains near semantic boundary.",
    },
    "boundary_p99_latency_near_slo": {
        "stratum": "boundary_indeterminate",
        "role": "primary",
        "semantic_label": "indeterminate",
        "target_label_tendency": "boundary_for_reject",
        "target_benchmark_label": "indeterminate_boundary",
        "run_type": "boundary",
        "scenario_id": "cloud_boundary_p99_latency_near_slo",
        "change_family_id": "boundary_latency_slo",
        "implementation_id": "p99_near_slo_boundary_cloud_v1",
        "component_id": "ts-route-service",
        "change_target_component_id": "ts-route-service",
        "affected_component_ids": "ts-route-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-route-service",
        "runner_support": "needs_latency_injector",
        "execution_status": "needs_oracle_extension",
        "pre_run_requirement": "direct latency SLO source; port-forward latency excluded",
        "oracle_extension_required": "direct_latency_slo_required",
        "slug": "bd_p99_near_slo",
        "notes": "Near-SLO boundary case for Reject analysis.",
    },
    "boundary_strong_logs_oracle_success": {
        "stratum": "boundary_indeterminate",
        "role": "primary",
        "semantic_label": "indeterminate",
        "target_label_tendency": "boundary_for_reject",
        "target_benchmark_label": "indeterminate_boundary",
        "run_type": "boundary",
        "scenario_id": "cloud_boundary_strong_logs_oracle_success",
        "change_family_id": "boundary_log_oracle_conflict",
        "implementation_id": "debug_or_verbose_log_oracle_success_cloud_v1",
        "component_id": "ts-route-service",
        "change_target_component_id": "ts-route-service",
        "affected_component_ids": "ts-route-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-route-service",
        "runner_support": "needs_log_level_injector",
        "execution_status": "needs_runner_extension",
        "pre_run_requirement": "freeze log-level/config patch; oracle must pass",
        "oracle_extension_required": "true",
        "slug": "bd_strong_logs_success",
        "notes": "Strong log evidence but external Oracle succeeds.",
    },
    "boundary_weak_oracle_failure_weak_logs": {
        "stratum": "boundary_indeterminate",
        "role": "primary",
        "semantic_label": "indeterminate",
        "target_label_tendency": "boundary_for_reject",
        "target_benchmark_label": "indeterminate_boundary",
        "run_type": "boundary",
        "scenario_id": "cloud_boundary_weak_oracle_failure_weak_logs",
        "change_family_id": "boundary_weak_evidence",
        "implementation_id": "weak_oracle_weak_logs_cloud_v1",
        "component_id": "ts-seat-service",
        "change_target_component_id": "ts-seat-service",
        "affected_component_ids": "ts-seat-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-seat-service",
        "runner_support": "needs_boundary_oracle",
        "execution_status": "needs_oracle_extension",
        "pre_run_requirement": "freeze weak-failure band distinct from unexpected threshold",
        "oracle_extension_required": "true",
        "slug": "bd_weak_fail_weak_logs",
        "notes": "Weak semantic failure and weak log evidence.",
    },
    "boundary_transition_not_stable": {
        "stratum": "boundary_indeterminate",
        "role": "primary",
        "semantic_label": "indeterminate",
        "target_label_tendency": "boundary_for_reject",
        "target_benchmark_label": "indeterminate_boundary",
        "run_type": "boundary",
        "scenario_id": "cloud_boundary_rollout_transition_unstable",
        "change_family_id": "boundary_transition",
        "implementation_id": "rollout_transition_not_stable_cloud_v1",
        "component_id": "ts-price-service",
        "change_target_component_id": "ts-price-service",
        "affected_component_ids": "ts-price-service,ts-gateway-service",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "needs_transition_capture_mode",
        "execution_status": "needs_runner_extension",
        "pre_run_requirement": "new phase policy: boundary uses transition evidence, main test still excludes transition",
        "oracle_extension_required": "true",
        "slug": "bd_transition_unstable",
        "notes": "Rollout transition remains ambiguous and is not forced into binary labels.",
    },
    "boundary_resource_pressure_recovered_short": {
        "stratum": "boundary_indeterminate",
        "role": "primary",
        "semantic_label": "indeterminate",
        "target_label_tendency": "boundary_for_reject",
        "target_benchmark_label": "indeterminate_boundary",
        "run_type": "boundary",
        "scenario_id": "cloud_boundary_short_resource_pressure",
        "change_family_id": "boundary_resource_pressure",
        "implementation_id": "short_cpu_pressure_recovered_cloud_v1",
        "component_id": "worker-node",
        "change_target_component_id": "worker-node",
        "affected_component_ids": "ts-price-service,ts-gateway-service,node",
        "oracle_component_ids": "ts-gateway-service,ts-price-service",
        "runner_support": "needs_node_stress_injector",
        "execution_status": "needs_oracle_extension",
        "pre_run_requirement": "freeze duration below unexpected sustained-failure threshold",
        "oracle_extension_required": "metrics_or_k8s_state_recommended",
        "slug": "bd_resource_short",
        "notes": "Transient pressure recovered before sustained failure criteria.",
    },
}


PRIMARY_PLAN = {
    "expected_log_visible": [
        "expected_workload_mix_route_heavy",
        "expected_workload_mix_train_station_heavy",
        "expected_workload_mix_price_heavy",
        "expected_scale_price_1_to_2",
        "expected_scale_route_1_to_2",
        "expected_scale_train_1_to_2",
        "expected_scale_station_1_to_2",
        "expected_pod_migration_price_replacement",
        "expected_node_drain_failover_price_redundant",
        "expected_workload_mix_price_heavy",
    ],
    "expected_log_subtle": [
        "expected_low_impact_ui_annotation",
        "expected_low_impact_price_annotation",
        "expected_low_impact_route_annotation",
        "expected_low_impact_train_annotation",
        "expected_low_impact_station_annotation",
        "expected_compatible_config_price_pool_increase",
        "expected_compatible_config_route_timeout_valid",
    ],
    "unexpected_log_visible": [
        "unexpected_price_bad_db_port",
        "unexpected_route_bad_db_port",
        "unexpected_train_bad_db_port",
        "unexpected_price_scale_zero",
        "unexpected_route_scale_zero",
        "unexpected_train_scale_zero",
        "unexpected_resource_limit_price_too_small",
        "unexpected_connection_pool_exhaustion_price",
        "unexpected_slow_path_route_latency_slo",
    ],
    "unexpected_log_subtle": [
        "unexpected_seat_scale_zero_weak",
        "unexpected_user_scale_zero_weak",
        "unexpected_basic_scale_zero_weak",
        "unexpected_config_scale_zero_weak",
        "unexpected_basic_scale_zero_weak",
        "unexpected_partial_dependency_low_coverage_seat",
        "unexpected_slow_path_low_rate_latency",
        "unexpected_node_resource_contention_weak",
        "unexpected_seat_scale_zero_weak",
        "unexpected_user_scale_zero_weak",
    ],
    "boundary_indeterminate": [
        "boundary_success_rate_slight_drop",
        "boundary_p99_latency_near_slo",
        "boundary_strong_logs_oracle_success",
        "boundary_weak_oracle_failure_weak_logs",
        "boundary_transition_not_stable",
        "unexpected_service_kill_replacement_low_oov",
        "boundary_success_rate_slight_drop",
        "boundary_p99_latency_near_slo",
        "boundary_strong_logs_oracle_success",
        "boundary_weak_oracle_failure_weak_logs",
        "boundary_transition_not_stable",
        "boundary_resource_pressure_recovered_short",
    ],
}

RESERVE_PLAN = {
    "expected_log_visible": [
        "expected_scale_route_1_to_2",
        "expected_pod_migration_price_replacement",
    ],
    "expected_log_subtle": [
        "expected_low_impact_route_annotation",
        "expected_compatible_image_price_patch",
    ],
    "unexpected_log_visible": [
        "unexpected_price_bad_db_port",
        "unexpected_resource_limit_price_too_small",
    ],
    "unexpected_log_subtle": [
        "unexpected_basic_scale_zero_weak",
        "unexpected_node_resource_contention_weak",
        "unexpected_partial_dependency_low_coverage_seat",
    ],
    "boundary_indeterminate": [
        "boundary_success_rate_slight_drop",
        "boundary_p99_latency_near_slo",
        "boundary_transition_not_stable",
    ],
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def scenario_meta(name: str) -> dict:
    if name in SCENARIOS:
        return SCENARIOS[name]
    return BOUNDARY_SCENARIOS[name]


def block_for_index(index: int, role: str) -> str:
    if role == "reserve":
        return "D_reserve"
    if index <= 15:
        return "A_starter"
    if index <= 30:
        return "B_expand"
    return "C_complete"


def ordered_primary_names() -> list[str]:
    rows: list[str] = []
    for i in range(3):
        for stratum in TARGET_CUMULATIVE_COUNTS:
            rows.append(PRIMARY_PLAN[stratum][i])
    for i in range(3, 6):
        for stratum in TARGET_CUMULATIVE_COUNTS:
            rows.append(PRIMARY_PLAN[stratum][i])
    for stratum, names in PRIMARY_PLAN.items():
        rows.extend(names[6:])
    return rows


def ordered_reserve_names() -> list[str]:
    rows: list[str] = []
    for stratum in TARGET_CUMULATIVE_COUNTS:
        rows.extend(RESERVE_PLAN[stratum])
    return rows


def build_rows() -> list[dict]:
    rows: list[dict] = []
    previous = "none"
    seed_base = 9700
    all_names = [(name, "primary") for name in ordered_primary_names()]
    all_names.extend((name, "reserve") for name in ordered_reserve_names())
    for ordinal, (name, role) in enumerate(all_names, start=1):
        meta = dict(scenario_meta(name))
        stratum = meta["stratum"]
        meta["role"] = role
        block = block_for_index(ordinal, role)
        seed = seed_base + ordinal
        target_label = meta.get("target_benchmark_label", meta["target_label_tendency"])
        allow_formal = "false" if stratum == "boundary_indeterminate" else "true"
        split_role = "reject_boundary_holdout" if stratum == "boundary_indeterminate" else "formal_test_candidate"
        run_id = f"{meta['slug']}_cloud_formal_v0_3_20260707_r{ordinal:02d}"
        row = {
            "ordinal": ordinal,
            "batch_id": BATCH_ID,
            "cloud_protocol_version": CLOUD_PROTOCOL_VERSION,
            "source_v0_2_batch_id": SOURCE_V0_2_BATCH_ID,
            "execution_block": block,
            "block_id": f"cloud_formal_v0_3_{block}",
            "batch_position": ordinal,
            "preceding_scenario": previous,
            "scenario": name,
            "semantic_label": meta["semantic_label"],
            "target_label_tendency": meta["target_label_tendency"],
            "target_benchmark_label": target_label,
            "v0_3_stratum": stratum,
            "v0_3_role": role,
            "existing_v0_2_valid_count_for_stratum": EXISTING_V0_2_COUNTS[stratum],
            "target_cumulative_valid_total_for_stratum": TARGET_CUMULATIVE_COUNTS[stratum],
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
                "run_A_starter_then_audit; run_B_expand_then_audit; "
                "run_C_complete_to_reach_primary_targets; run_D_reserve_only_for_invalid_or_shortfall"
            ),
            "detector_feedback_used": "false",
            "notes": meta["notes"],
        }
        rows.append(row)
        previous = name
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def write_protocol_note(path: Path, rows: list[dict]) -> None:
    counts = Counter(row["v0_3_stratum"] for row in rows if row["v0_3_role"] == "primary")
    reserves = Counter(row["v0_3_stratum"] for row in rows if row["v0_3_role"] == "reserve")
    ready = sum(1 for row in rows if row["execution_status"] == "ready_existing_runner")
    smoke = sum(1 for row in rows if row["execution_status"] == "implemented_needs_cluster_smoke")
    needs = len(rows) - ready - smoke
    text = f"""# Cloud Formal v0.3 Stratified Supplement Plan

Generated at: {datetime.now(timezone.utc).isoformat()}

This is a draft supplement plan, not a replacement for v0.2. The 36 valid v0.2
formal runs remain the final benchmark evidence already collected. v0.3 extends
coverage by strata instead of repeating the same workload-mix and scale-zero
patterns.

## Cumulative Target

| Stratum | v0.2 valid | v0.3 primary additions | cumulative target |
|---|---:|---:|---:|
| Expected log-visible | {EXISTING_V0_2_COUNTS['expected_log_visible']} | {counts['expected_log_visible']} | {TARGET_CUMULATIVE_COUNTS['expected_log_visible']} |
| Expected log-subtle / successful-no-drift | {EXISTING_V0_2_COUNTS['expected_log_subtle']} | {counts['expected_log_subtle']} | {TARGET_CUMULATIVE_COUNTS['expected_log_subtle']} |
| Unexpected log-visible | {EXISTING_V0_2_COUNTS['unexpected_log_visible']} | {counts['unexpected_log_visible']} | {TARGET_CUMULATIVE_COUNTS['unexpected_log_visible']} |
| Unexpected log-subtle | {EXISTING_V0_2_COUNTS['unexpected_log_subtle']} | {counts['unexpected_log_subtle']} | {TARGET_CUMULATIVE_COUNTS['unexpected_log_subtle']} |
| Boundary / indeterminate Reject probes | {EXISTING_V0_2_COUNTS['boundary_indeterminate']} | {counts['boundary_indeterminate']} | {TARGET_CUMULATIVE_COUNTS['boundary_indeterminate']} |

Primary v0.3 additions: {sum(counts.values())} runs. Reserve rows: {sum(reserves.values())}
runs. Total planned supplement rows: {len(rows)}.

## Execution Discipline

- Threshold file remains frozen: `{THRESHOLD_SHA256}`.
- Cloud freeze id remains frozen unless runner/oracle changes require a new freeze:
  `{CLOUD_FREEZE_ID}`.
- Detector output must not be used to select or remove valid runs.
- Boundary/indeterminate rows are Reject holdout probes and are not mixed into
  the binary Expected/Unexpected main test set.
- All valid primary rows are retained even if their final drift-gate outcome is
  different from the target tendency.

## Current Executability

Rows ready under the local v0.3 runner code path: {ready}.
Rows implemented but requiring live-cluster smoke before collection: {smoke}.
Rows still requiring runner/oracle/injector extension before execution: {needs}.

Therefore v0.3 must not be launched as a normal batch until the extension backlog
is implemented, live-cluster smoke tests pass, and a new executable freeze
manifest is generated.

## Block Policy

1. Block A starter: first 15 primary rows, then quality and label audit.
2. Block B expand: next 15 primary rows, then audit.
3. Block C complete: remaining 18 primary rows to reach cumulative targets.
4. Block D reserve: use only for invalid replacement or stratum shortfall.

## Definitions

- Expected log-visible: semantic_label=expected and target tendency is
  expected_drift.
- Expected log-subtle: semantic_label=expected and target tendency is
  successful_no_drift.
- Unexpected log-visible: semantic_label=unexpected and target tendency is
  unexpected_drift.
- Unexpected log-subtle: semantic_label=unexpected and target tendency is
  unexpected_without_observable_log_drift.
- Boundary/indeterminate: semantic_label=indeterminate, retained for Reject
  analysis rather than binary main-test metrics.
"""
    path.write_text(text, encoding="utf-8")


def write_backlog(path: Path, rows: list[dict]) -> None:
    grouped: dict[str, list[dict]] = {}
    seen: set[str] = set()
    for row in rows:
        if row["execution_status"] == "ready_existing_runner":
            continue
        if row["scenario"] in seen:
            continue
        seen.add(row["scenario"])
        grouped.setdefault(row["runner_support"], []).append(row)
    lines = ["# Cloud Formal v0.3 Runner/Oracle/Smoke Backlog", ""]
    lines.append("The rows below are not collection-ready yet.")
    lines.append(
        "Some already have local runner code and only need live-cluster smoke; "
        "others still need runner, oracle, or injector implementation."
    )
    lines.append("")
    for support, items in sorted(grouped.items()):
        lines.append(f"## {support}")
        for item in items:
            lines.append(
                f"- `{item['scenario']}` ({item['v0_3_stratum']}): "
                f"{item['pre_run_requirement']}; oracle_extension_required={item['oracle_extension_required']}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_smoke_plan(path: Path, rows: list[dict]) -> None:
    smoke_rows = [
        row for row in rows
        if row["execution_status"] in {"ready_existing_runner", "implemented_needs_cluster_smoke"}
    ]
    # One representative per scenario is enough for the first online smoke.
    seen = set()
    unique_rows = []
    for row in smoke_rows:
        if row["scenario"] in seen:
            continue
        seen.add(row["scenario"])
        unique_rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = [
            "scenario",
            "semantic_label",
            "target_label_tendency",
            "v0_3_stratum",
            "execution_status",
            "runner_support",
            "run_id",
            "pre_run_requirement",
            "notes",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in unique_rows:
            writer.writerow({key: row[key] for key in fieldnames})


def write_new_runner_smoke_executable_plan(path: Path, rows: list[dict]) -> None:
    smoke_statuses = {"ready_existing_runner", "implemented_needs_cluster_smoke"}
    selected = []
    seen = set()
    for row in rows:
        if row["execution_status"] not in smoke_statuses:
            continue
        if row["runner_support"] == "formal_runner_v0.1":
            continue
        if row["scenario"] in seen:
            continue
        seen.add(row["scenario"])
        smoke_row = dict(row)
        smoke_row["batch_id"] = f"{BATCH_ID}_new_runner_smoke"
        smoke_row["execution_block"] = "smoke"
        smoke_row["block_id"] = "cloud_formal_v0_3_new_runner_smoke"
        smoke_row["split_role"] = "protocol_smoke_only"
        smoke_row["allow_as_formal_test"] = "false"
        smoke_row["planned_stop_rule"] = "smoke_only_not_formal_dataset"
        selected.append(smoke_row)
    for index, row in enumerate(selected, start=1):
        row["ordinal"] = index
        row["batch_position"] = index
        row["run_id"] = row["run_id"].replace("_cloud_formal_v0_3_20260707_", "_cloud_formal_v0_3_smoke_")
    write_csv(path, selected)


def write_manifest(path: Path, output_files: dict[str, Path], rows: list[dict]) -> None:
    counts = Counter(row["v0_3_stratum"] for row in rows if row["v0_3_role"] == "primary")
    reserves = Counter(row["v0_3_stratum"] for row in rows if row["v0_3_role"] == "reserve")
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "batch_id": BATCH_ID,
        "protocol_version": CLOUD_PROTOCOL_VERSION,
        "freeze_status": "draft_not_executable_until_new_injectors_frozen",
        "source_v0_2_batch_id": SOURCE_V0_2_BATCH_ID,
        "source_v0_2_counts": EXISTING_V0_2_COUNTS,
        "target_cumulative_counts": TARGET_CUMULATIVE_COUNTS,
        "v0_3_primary_additions": dict(counts),
        "v0_3_reserve_rows": dict(reserves),
        "threshold_sha256": THRESHOLD_SHA256,
        "threshold_source": THRESHOLD_SOURCE,
        "cloud_freeze_id": CLOUD_FREEZE_ID,
        "cluster_type": CLUSTER_TYPE,
        "detector_feedback_used": False,
        "runner_ready_rows": sum(1 for row in rows if row["execution_status"] == "ready_existing_runner"),
        "runner_implemented_needs_cluster_smoke_rows": sum(
            1 for row in rows if row["execution_status"] == "implemented_needs_cluster_smoke"
        ),
        "runner_extension_required_rows": sum(
            1
            for row in rows
            if row["execution_status"] not in {"ready_existing_runner", "implemented_needs_cluster_smoke"}
        ),
        "files": {name: str(p) for name, p in output_files.items()},
        "file_sha256": {name: sha256_file(p) for name, p in output_files.items() if p.exists()},
    }
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def write_scenario_catalog(path: Path, rows: list[dict]) -> None:
    seen = {}
    for row in rows:
        if row["scenario"] in seen:
            continue
        seen[row["scenario"]] = {
            "scenario": row["scenario"],
            "semantic_label": row["semantic_label"],
            "target_label_tendency": row["target_label_tendency"],
            "target_benchmark_label": row["target_benchmark_label"],
            "v0_3_stratum": row["v0_3_stratum"],
            "change_family_id": row["change_family_id"],
            "implementation_id": row["implementation_id"],
            "component_id": row["component_id"],
            "affected_component_ids": row["affected_component_ids"],
            "oracle_component_ids": row["oracle_component_ids"],
            "runner_support": row["runner_support"],
            "execution_status": row["execution_status"],
            "pre_run_requirement": row["pre_run_requirement"],
            "oracle_extension_required": row["oracle_extension_required"],
            "notes": row["notes"],
        }
    catalog = {
        "batch_id": BATCH_ID,
        "protocol_version": CLOUD_PROTOCOL_VERSION,
        "freeze_status": "draft_not_executable_until_new_injectors_frozen",
        "scenario_count": len(seen),
        "scenarios": list(seen.values()),
    }
    path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default="results/scwarn_pilot/collections/cloud_expected_unexpected_formal_v0_3_plan_20260707",
    )
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    rows = build_rows()
    primary = [row for row in rows if row["v0_3_role"] == "primary"]
    reserve = [row for row in rows if row["v0_3_role"] == "reserve"]
    ready = [row for row in rows if row["execution_status"] == "ready_existing_runner"]

    files = {
        "stratified_plan_csv": out / "cloud_formal_v0_3_stratified_plan.csv",
        "primary_plan_csv": out / "cloud_formal_v0_3_primary_plan.csv",
        "reserve_plan_csv": out / "cloud_formal_v0_3_reserve_plan.csv",
        "ready_existing_runner_plan_csv": out / "cloud_formal_v0_3_ready_existing_runner_rows.csv",
        "protocol_note_md": out / "cloud_formal_v0_3_protocol_note_20260707.md",
        "runner_extension_backlog_md": out / "cloud_formal_v0_3_runner_extension_backlog.md",
        "online_smoke_plan_csv": out / "cloud_formal_v0_3_online_smoke_plan.csv",
        "new_runner_smoke_executable_plan_csv": out / "cloud_formal_v0_3_new_runner_smoke_executable_plan.csv",
        "online_smoke_runbook_md": out / "cloud_formal_v0_3_online_smoke_runbook.md",
        "local_implementation_report_md": out / "cloud_formal_v0_3_local_implementation_report.md",
        "stratum_targets_json": out / "cloud_formal_v0_3_stratum_targets.json",
        "scenario_catalog_json": out / "cloud_formal_v0_3_scenario_catalog.json",
    }

    write_csv(files["stratified_plan_csv"], rows)
    write_csv(files["primary_plan_csv"], primary)
    write_csv(files["reserve_plan_csv"], reserve)
    write_csv(files["ready_existing_runner_plan_csv"], ready)
    write_protocol_note(files["protocol_note_md"], rows)
    write_backlog(files["runner_extension_backlog_md"], rows)
    write_smoke_plan(files["online_smoke_plan_csv"], rows)
    write_new_runner_smoke_executable_plan(files["new_runner_smoke_executable_plan_csv"], rows)
    write_scenario_catalog(files["scenario_catalog_json"], rows)
    files["stratum_targets_json"].write_text(
        json.dumps(
            {
                "source_v0_2_counts": EXISTING_V0_2_COUNTS,
                "target_cumulative_counts": TARGET_CUMULATIVE_COUNTS,
                "primary_additions": dict(Counter(row["v0_3_stratum"] for row in primary)),
                "reserve_rows": dict(Counter(row["v0_3_stratum"] for row in reserve)),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest = out / "cloud_formal_v0_3_freeze_manifest_draft_20260707.json"
    write_manifest(manifest, files, rows)

    print(
        json.dumps(
            {
                "output_dir": str(out),
                "rows": len(rows),
                "primary_rows": len(primary),
                "reserve_rows": len(reserve),
                "ready_existing_runner_rows": len(ready),
                "implemented_needs_cluster_smoke_rows": sum(
                    1 for row in rows if row["execution_status"] == "implemented_needs_cluster_smoke"
                ),
                "manifest": str(manifest),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
