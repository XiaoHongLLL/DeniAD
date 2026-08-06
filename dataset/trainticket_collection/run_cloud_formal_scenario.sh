#!/usr/bin/env bash
set -euo pipefail

SCENARIO=""
NAMESPACE="trainticket-pilot"
RUN_ID=""
BASE_URL="${SCWARN_BASE_URL:-http://127.0.0.1:30467}"
WORKLOAD_PROFILE_ID="W1_steady_core"
POST_WORKLOAD_PROFILE_ID=""
SEED="8501"
STABLE_SECONDS="300"
WARMUP_SECONDS="120"
PRE_CHANGE_SECONDS="600"
POST_CHANGE_SECONDS="900"
RATE_PER_SECOND="0.2"
WORKLOAD_TIMEOUT_SECONDS="20"
POST_CALL_MARGIN_SECONDS="2.0"
TRANSITION_BUDGET_SECONDS="300"
POST_STABLE_SECONDS="120"
POST_COLLECTION_SECONDS="20"
BATCH_ID=""
BLOCK_ID=""
BATCH_POSITION="0"
PRECEDING_SCENARIO="none"
MATCHED_WORKLOAD_SEED=""
HOURS_SINCE_CLUSTER_START="0"
CLOUD_FREEZE_ID=""
CLUSTER_TYPE="linux_kubeadm_multinode"
THRESHOLD_FILE=""
THRESHOLD_SHA256=""
UNEXPECTED_FAILURE_RATIO="0.01"
INDETERMINATE_FAILURE_RATIO_MIN="0.001"
CLOUD_PROTOCOL_VERSION="linux-cloud-formal-v0.1"
SCENARIO_PREFERENCE="formal_expected_unexpected_label_probe"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --scenario) SCENARIO="$2"; shift 2 ;;
    --namespace) NAMESPACE="$2"; shift 2 ;;
    --run-id) RUN_ID="$2"; shift 2 ;;
    --base-url) BASE_URL="$2"; shift 2 ;;
    --workload-profile-id) WORKLOAD_PROFILE_ID="$2"; shift 2 ;;
    --post-workload-profile-id) POST_WORKLOAD_PROFILE_ID="$2"; shift 2 ;;
    --seed) SEED="$2"; shift 2 ;;
    --stable-seconds) STABLE_SECONDS="$2"; shift 2 ;;
    --warmup-seconds) WARMUP_SECONDS="$2"; shift 2 ;;
    --pre-change-seconds) PRE_CHANGE_SECONDS="$2"; shift 2 ;;
    --post-change-seconds) POST_CHANGE_SECONDS="$2"; shift 2 ;;
    --rate-per-second) RATE_PER_SECOND="$2"; shift 2 ;;
    --workload-timeout-seconds) WORKLOAD_TIMEOUT_SECONDS="$2"; shift 2 ;;
    --transition-budget-seconds) TRANSITION_BUDGET_SECONDS="$2"; shift 2 ;;
    --post-stable-seconds) POST_STABLE_SECONDS="$2"; shift 2 ;;
    --post-collection-seconds) POST_COLLECTION_SECONDS="$2"; shift 2 ;;
    --batch-id) BATCH_ID="$2"; shift 2 ;;
    --block-id) BLOCK_ID="$2"; shift 2 ;;
    --batch-position) BATCH_POSITION="$2"; shift 2 ;;
    --preceding-scenario) PRECEDING_SCENARIO="$2"; shift 2 ;;
    --matched-workload-seed) MATCHED_WORKLOAD_SEED="$2"; shift 2 ;;
    --hours-since-cluster-start) HOURS_SINCE_CLUSTER_START="$2"; shift 2 ;;
    --cloud-freeze-id) CLOUD_FREEZE_ID="$2"; shift 2 ;;
    --cluster-type) CLUSTER_TYPE="$2"; shift 2 ;;
    --threshold-file) THRESHOLD_FILE="$2"; shift 2 ;;
    --threshold-sha256) THRESHOLD_SHA256="$2"; shift 2 ;;
    --unexpected-failure-ratio) UNEXPECTED_FAILURE_RATIO="$2"; shift 2 ;;
    --indeterminate-failure-ratio-min) INDETERMINATE_FAILURE_RATIO_MIN="$2"; shift 2 ;;
    --cloud-protocol-version) CLOUD_PROTOCOL_VERSION="$2"; shift 2 ;;
    --scenario-preference) SCENARIO_PREFERENCE="$2"; shift 2 ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ -z "$SCENARIO" || -z "$RUN_ID" ]]; then
  echo "--scenario and --run-id are required" >&2
  exit 2
fi
if [[ -z "$MATCHED_WORKLOAD_SEED" ]]; then
  MATCHED_WORKLOAD_SEED="$SEED"
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
KUBECTL="${KUBECTL:-kubectl}"
PYTHON="python3"
ALLOWLIST="$REPO_ROOT/configs/trainticket_collection/workload_allowlist.yaml"
PROFILES="$REPO_ROOT/configs/trainticket_collection/workload_profiles.json"
RUN_ROOT="${SCWARN_RUN_ROOT:-$REPO_ROOT/artifacts/trainticket_runs}"
RUN_DIR="$RUN_ROOT/$RUN_ID"
mkdir -p "$RUN_DIR"

WATCHER_PID=""
WATCHER_SECONDS=$((PRE_CHANGE_SECONDS + TRANSITION_BUDGET_SECONDS + POST_CHANGE_SECONDS + POST_COLLECTION_SECONDS))
TARGET_DEPLOYMENT=""
BAD_DB_PORT_ENV=""
SCENARIO_ROLLOUT_EXIT_CODE=0
SCENARIO_ROLLOUT_COMPLETED=true
SCENARIO_FAILURE_STATE="runtime_failure"
SCENARIO_DEPLOYMENT_FAILED_TIME=""

utc_now() {
  date -u +%Y-%m-%dT%H:%M:%SZ
}

update_json() {
  local path="$1"
  local payload="$2"
  "$PYTHON" - "$path" "$payload" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(sys.argv[2])
if path.exists():
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        data = {}
else:
    data = {}
data.update(payload)
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
PY
}

cleanup_watcher() {
  if [[ -n "$WATCHER_PID" ]] && kill -0 "$WATCHER_PID" 2>/dev/null; then
    echo "Stopping watcher after premature run exit: $WATCHER_PID"
    kill "$WATCHER_PID" 2>/dev/null || true
    sleep 2
    kill -9 "$WATCHER_PID" 2>/dev/null || true
  fi
}
trap cleanup_watcher EXIT

restore_baseline_state() {
  "$KUBECTL" set env deployment/ts-price-service -n "$NAMESPACE" PRICE_MYSQL_PORT- || true
  "$KUBECTL" set env deployment/ts-route-service -n "$NAMESPACE" ROUTE_MYSQL_PORT- || true
  "$KUBECTL" set env deployment/ts-train-service -n "$NAMESPACE" TRAIN_MYSQL_PORT- || true
  "$KUBECTL" set env deployment/ts-station-service -n "$NAMESPACE" JAVA_TOOL_OPTIONS- || true
  "$KUBECTL" set env deployment/ts-gateway-service -n "$NAMESPACE" JAVA_TOOL_OPTIONS- || true
  for deployment in ts-price-service ts-route-service; do
    "$KUBECTL" set env "deployment/$deployment" -n "$NAMESPACE" \
      SPRING_DATASOURCE_HIKARI_MAXIMUM_POOL_SIZE- \
      SPRING_DATASOURCE_HIKARI_CONNECTION_TIMEOUT- \
      SCWARN_COMPAT_CONFIG_TOKEN- || true
  done
  "$KUBECTL" set resources deployment/ts-price-service -n "$NAMESPACE" \
    --limits=cpu=0,memory=0 --requests=cpu=0,memory=0 || true
  for deployment in ts-station-service ts-gateway-service; do
    "$KUBECTL" set resources "deployment/$deployment" -n "$NAMESPACE" \
      --limits=cpu=500m,memory=768Mi --requests=cpu=25m,memory=64Mi || true
  done
  for deployment in \
    ts-price-service \
    ts-route-service \
    ts-train-service \
    ts-station-service \
    ts-seat-service \
    ts-user-service \
    ts-basic-service \
    ts-config-service \
    ts-ui-dashboard \
    ts-gateway-service
  do
    "$KUBECTL" scale "deployment/$deployment" -n "$NAMESPACE" --replicas=1
    "$KUBECTL" rollout status "deployment/$deployment" -n "$NAMESPACE" --timeout=600s
  done

  "$KUBECTL" annotate deployment/ts-ui-dashboard -n "$NAMESPACE" scwarn-mini-low-impact- --overwrite || true
  "$KUBECTL" annotate deployment/ts-ui-dashboard -n "$NAMESPACE" scwarn-formal-low-impact- --overwrite || true
  for deployment in ts-price-service ts-route-service ts-train-service ts-station-service ts-seat-service ts-user-service ts-basic-service ts-config-service ts-ui-dashboard ts-gateway-service; do
    "$KUBECTL" annotate "deployment/$deployment" -n "$NAMESPACE" scwarn-formal-low-impact- --overwrite || true
  done
}

case "$SCENARIO" in
  expected_workload_mix_route_heavy)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_workload_mix_w1_to_route_heavy"
    CHANGE_FAMILY_ID="expected_workload_mix"
    IMPLEMENTATION_ID="w1_to_route_query_heavy_cloud_v1"
    COMPONENT_ID="workload_profile"
    CHANGE_TARGET_COMPONENT_ID="workload-generator"
    AFFECTED_COMPONENT_IDS="ts-gateway-service,ts-route-service,ts-station-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-route-service,ts-station-service"
    POST_WORKLOAD_PROFILE_ID="${POST_WORKLOAD_PROFILE_ID:-W1_route_query_heavy}"
    ;;
  expected_workload_mix_train_station_heavy)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_workload_mix_w1_to_train_station_heavy"
    CHANGE_FAMILY_ID="expected_workload_mix"
    IMPLEMENTATION_ID="w1_to_train_station_query_heavy_cloud_v1"
    COMPONENT_ID="workload_profile"
    CHANGE_TARGET_COMPONENT_ID="workload-generator"
    AFFECTED_COMPONENT_IDS="ts-gateway-service,ts-train-service,ts-station-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-train-service,ts-station-service"
    POST_WORKLOAD_PROFILE_ID="${POST_WORKLOAD_PROFILE_ID:-W1_train_station_query_heavy}"
    ;;
  expected_workload_mix_price_heavy)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_workload_mix_w1_to_price_heavy"
    CHANGE_FAMILY_ID="expected_workload_mix"
    IMPLEMENTATION_ID="w1_to_price_query_heavy_cloud_v1"
    COMPONENT_ID="workload_profile"
    CHANGE_TARGET_COMPONENT_ID="workload-generator"
    AFFECTED_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service"
    POST_WORKLOAD_PROFILE_ID="${POST_WORKLOAD_PROFILE_ID:-W1_price_query_heavy}"
    ;;
  expected_scale_price_1_to_2)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_price_scale_1_to_2"
    CHANGE_FAMILY_ID="expected_resource_scale"
    IMPLEMENTATION_ID="price_replicas_1_to_2_cloud_v1"
    COMPONENT_ID="ts-price-service"
    TARGET_DEPLOYMENT="ts-price-service"
    CHANGE_TARGET_COMPONENT_ID="ts-price-service"
    AFFECTED_COMPONENT_IDS="ts-price-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service"
    ;;
  expected_scale_route_1_to_2)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_route_scale_1_to_2"
    CHANGE_FAMILY_ID="expected_resource_scale"
    IMPLEMENTATION_ID="route_replicas_1_to_2_cloud_v1"
    COMPONENT_ID="ts-route-service"
    TARGET_DEPLOYMENT="ts-route-service"
    CHANGE_TARGET_COMPONENT_ID="ts-route-service"
    AFFECTED_COMPONENT_IDS="ts-route-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-route-service"
    ;;
  expected_scale_train_1_to_2)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_train_scale_1_to_2"
    CHANGE_FAMILY_ID="expected_resource_scale"
    IMPLEMENTATION_ID="train_replicas_1_to_2_cloud_v1"
    COMPONENT_ID="ts-train-service"
    TARGET_DEPLOYMENT="ts-train-service"
    CHANGE_TARGET_COMPONENT_ID="ts-train-service"
    AFFECTED_COMPONENT_IDS="ts-train-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-train-service"
    ;;
  expected_scale_station_1_to_2)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_station_scale_1_to_2"
    CHANGE_FAMILY_ID="expected_resource_scale"
    IMPLEMENTATION_ID="station_replicas_1_to_2_cloud_v1"
    COMPONENT_ID="ts-station-service"
    TARGET_DEPLOYMENT="ts-station-service"
    CHANGE_TARGET_COMPONENT_ID="ts-station-service"
    AFFECTED_COMPONENT_IDS="ts-station-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-station-service"
    ;;
  expected_scale_seat_1_to_2)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_seat_scale_1_to_2"
    CHANGE_FAMILY_ID="expected_resource_scale"
    IMPLEMENTATION_ID="seat_replicas_1_to_2_cloud_v1"
    COMPONENT_ID="ts-seat-service"
    TARGET_DEPLOYMENT="ts-seat-service"
    CHANGE_TARGET_COMPONENT_ID="ts-seat-service"
    AFFECTED_COMPONENT_IDS="ts-seat-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-seat-service"
    ;;
  expected_scale_user_1_to_2)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_user_scale_1_to_2"
    CHANGE_FAMILY_ID="expected_resource_scale"
    IMPLEMENTATION_ID="user_replicas_1_to_2_cloud_v1"
    COMPONENT_ID="ts-user-service"
    TARGET_DEPLOYMENT="ts-user-service"
    CHANGE_TARGET_COMPONENT_ID="ts-user-service"
    AFFECTED_COMPONENT_IDS="ts-user-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-user-service"
    ;;
  expected_scale_basic_1_to_2)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_basic_scale_1_to_2"
    CHANGE_FAMILY_ID="expected_resource_scale"
    IMPLEMENTATION_ID="basic_replicas_1_to_2_cloud_v1"
    COMPONENT_ID="ts-basic-service"
    TARGET_DEPLOYMENT="ts-basic-service"
    CHANGE_TARGET_COMPONENT_ID="ts-basic-service"
    AFFECTED_COMPONENT_IDS="ts-basic-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-basic-service"
    ;;
  expected_scale_config_1_to_2)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_config_scale_1_to_2"
    CHANGE_FAMILY_ID="expected_resource_scale"
    IMPLEMENTATION_ID="config_replicas_1_to_2_cloud_v1"
    COMPONENT_ID="ts-config-service"
    TARGET_DEPLOYMENT="ts-config-service"
    CHANGE_TARGET_COMPONENT_ID="ts-config-service"
    AFFECTED_COMPONENT_IDS="ts-config-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-config-service"
    ;;
  expected_scale_ui_dashboard_1_to_2)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_ui_dashboard_scale_1_to_2"
    CHANGE_FAMILY_ID="expected_resource_scale"
    IMPLEMENTATION_ID="ui_dashboard_replicas_1_to_2_cloud_v1"
    COMPONENT_ID="ts-ui-dashboard"
    TARGET_DEPLOYMENT="ts-ui-dashboard"
    CHANGE_TARGET_COMPONENT_ID="ts-ui-dashboard"
    AFFECTED_COMPONENT_IDS="ts-ui-dashboard"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service"
    ;;
  expected_scale_gateway_1_to_2)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_gateway_scale_1_to_2"
    CHANGE_FAMILY_ID="expected_resource_scale"
    IMPLEMENTATION_ID="gateway_replicas_1_to_2_cloud_v1"
    COMPONENT_ID="ts-gateway-service"
    TARGET_DEPLOYMENT="ts-gateway-service"
    CHANGE_TARGET_COMPONENT_ID="ts-gateway-service"
    AFFECTED_COMPONENT_IDS="ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service"
    ;;
  expected_low_impact_ui_annotation)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_low_impact_ui_annotation"
    CHANGE_FAMILY_ID="expected_low_impact_config"
    IMPLEMENTATION_ID="ui_dashboard_metadata_annotation_cloud_v1"
    COMPONENT_ID="ts-ui-dashboard"
    TARGET_DEPLOYMENT="ts-ui-dashboard"
    CHANGE_TARGET_COMPONENT_ID="ts-ui-dashboard"
    AFFECTED_COMPONENT_IDS="ts-ui-dashboard"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service"
    ;;
  expected_low_impact_price_annotation)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_low_impact_price_annotation"
    CHANGE_FAMILY_ID="expected_low_impact_config"
    IMPLEMENTATION_ID="price_metadata_annotation_cloud_v1"
    COMPONENT_ID="ts-price-service"
    TARGET_DEPLOYMENT="ts-price-service"
    CHANGE_TARGET_COMPONENT_ID="ts-price-service"
    AFFECTED_COMPONENT_IDS="ts-price-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service"
    ;;
  expected_low_impact_route_annotation)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_low_impact_route_annotation"
    CHANGE_FAMILY_ID="expected_low_impact_config"
    IMPLEMENTATION_ID="route_metadata_annotation_cloud_v1"
    COMPONENT_ID="ts-route-service"
    TARGET_DEPLOYMENT="ts-route-service"
    CHANGE_TARGET_COMPONENT_ID="ts-route-service"
    AFFECTED_COMPONENT_IDS="ts-route-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service"
    ;;
  expected_low_impact_train_annotation)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_low_impact_train_annotation"
    CHANGE_FAMILY_ID="expected_low_impact_config"
    IMPLEMENTATION_ID="train_metadata_annotation_cloud_v1"
    COMPONENT_ID="ts-train-service"
    TARGET_DEPLOYMENT="ts-train-service"
    CHANGE_TARGET_COMPONENT_ID="ts-train-service"
    AFFECTED_COMPONENT_IDS="ts-train-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service"
    ;;
  expected_low_impact_station_annotation)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_low_impact_station_annotation"
    CHANGE_FAMILY_ID="expected_low_impact_config"
    IMPLEMENTATION_ID="station_metadata_annotation_cloud_v1"
    COMPONENT_ID="ts-station-service"
    TARGET_DEPLOYMENT="ts-station-service"
    CHANGE_TARGET_COMPONENT_ID="ts-station-service"
    AFFECTED_COMPONENT_IDS="ts-station-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service"
    ;;
  expected_low_impact_seat_annotation)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_low_impact_seat_annotation"
    CHANGE_FAMILY_ID="expected_low_impact_config"
    IMPLEMENTATION_ID="seat_metadata_annotation_cloud_v1"
    COMPONENT_ID="ts-seat-service"
    TARGET_DEPLOYMENT="ts-seat-service"
    CHANGE_TARGET_COMPONENT_ID="ts-seat-service"
    AFFECTED_COMPONENT_IDS="ts-seat-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service,ts-seat-service"
    ;;
  expected_low_impact_user_annotation)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_low_impact_user_annotation"
    CHANGE_FAMILY_ID="expected_low_impact_config"
    IMPLEMENTATION_ID="user_metadata_annotation_cloud_v1"
    COMPONENT_ID="ts-user-service"
    TARGET_DEPLOYMENT="ts-user-service"
    CHANGE_TARGET_COMPONENT_ID="ts-user-service"
    AFFECTED_COMPONENT_IDS="ts-user-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service,ts-user-service"
    ;;
  expected_low_impact_basic_annotation)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_low_impact_basic_annotation"
    CHANGE_FAMILY_ID="expected_low_impact_config"
    IMPLEMENTATION_ID="basic_metadata_annotation_cloud_v1"
    COMPONENT_ID="ts-basic-service"
    TARGET_DEPLOYMENT="ts-basic-service"
    CHANGE_TARGET_COMPONENT_ID="ts-basic-service"
    AFFECTED_COMPONENT_IDS="ts-basic-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service,ts-basic-service"
    ;;
  expected_low_impact_config_annotation)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_low_impact_config_annotation"
    CHANGE_FAMILY_ID="expected_low_impact_config"
    IMPLEMENTATION_ID="config_metadata_annotation_cloud_v1"
    COMPONENT_ID="ts-config-service"
    TARGET_DEPLOYMENT="ts-config-service"
    CHANGE_TARGET_COMPONENT_ID="ts-config-service"
    AFFECTED_COMPONENT_IDS="ts-config-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service,ts-config-service"
    ;;
  expected_low_impact_gateway_annotation)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_low_impact_gateway_annotation"
    CHANGE_FAMILY_ID="expected_low_impact_config"
    IMPLEMENTATION_ID="gateway_metadata_annotation_cloud_v1"
    COMPONENT_ID="ts-gateway-service"
    TARGET_DEPLOYMENT="ts-gateway-service"
    CHANGE_TARGET_COMPONENT_ID="ts-gateway-service"
    AFFECTED_COMPONENT_IDS="ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service"
    ;;
  expected_compatible_config_price_pool_increase)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_price_pool_increase"
    CHANGE_FAMILY_ID="expected_compatible_config"
    IMPLEMENTATION_ID="price_hikari_pool_valid_increase_cloud_v1"
    COMPONENT_ID="ts-price-service"
    TARGET_DEPLOYMENT="ts-price-service"
    CHANGE_TARGET_COMPONENT_ID="ts-price-service"
    AFFECTED_COMPONENT_IDS="ts-price-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service"
    ;;
  expected_compatible_config_route_timeout_valid)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_route_timeout_valid"
    CHANGE_FAMILY_ID="expected_compatible_config"
    IMPLEMENTATION_ID="route_hikari_connection_timeout_valid_cloud_v1"
    COMPONENT_ID="ts-route-service"
    TARGET_DEPLOYMENT="ts-route-service"
    CHANGE_TARGET_COMPONENT_ID="ts-route-service"
    AFFECTED_COMPONENT_IDS="ts-route-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-route-service"
    ;;
  unexpected_price_bad_db_port)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_price_wrong_db_port"
    CHANGE_FAMILY_ID="unexpected_bad_port"
    IMPLEMENTATION_ID="price_wrong_db_port_3399_cloud_v1"
    COMPONENT_ID="ts-price-service"
    TARGET_DEPLOYMENT="ts-price-service"
    BAD_DB_PORT_ENV="PRICE_MYSQL_PORT"
    CHANGE_TARGET_COMPONENT_ID="ts-price-service"
    AFFECTED_COMPONENT_IDS="ts-price-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service"
    ;;
  unexpected_route_bad_db_port)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_route_wrong_db_port"
    CHANGE_FAMILY_ID="unexpected_bad_port"
    IMPLEMENTATION_ID="route_wrong_db_port_cloud_v1"
    COMPONENT_ID="ts-route-service"
    TARGET_DEPLOYMENT="ts-route-service"
    BAD_DB_PORT_ENV="ROUTE_MYSQL_PORT"
    CHANGE_TARGET_COMPONENT_ID="ts-route-service"
    AFFECTED_COMPONENT_IDS="ts-route-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-route-service"
    ;;
  unexpected_train_bad_db_port)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_train_wrong_db_port"
    CHANGE_FAMILY_ID="unexpected_bad_port"
    IMPLEMENTATION_ID="train_wrong_db_port_cloud_v1"
    COMPONENT_ID="ts-train-service"
    TARGET_DEPLOYMENT="ts-train-service"
    BAD_DB_PORT_ENV="TRAIN_MYSQL_PORT"
    CHANGE_TARGET_COMPONENT_ID="ts-train-service"
    AFFECTED_COMPONENT_IDS="ts-train-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-train-service"
    ;;
  unexpected_price_scale_zero)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_price_scale_to_zero"
    CHANGE_FAMILY_ID="unexpected_service_termination"
    IMPLEMENTATION_ID="price_replicas_1_to_0_cloud_v1"
    COMPONENT_ID="ts-price-service"
    TARGET_DEPLOYMENT="ts-price-service"
    CHANGE_TARGET_COMPONENT_ID="ts-price-service"
    AFFECTED_COMPONENT_IDS="ts-price-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service"
    ;;
  unexpected_route_scale_zero)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_route_scale_to_zero"
    CHANGE_FAMILY_ID="unexpected_service_termination"
    IMPLEMENTATION_ID="route_replicas_1_to_0_cloud_v1"
    COMPONENT_ID="ts-route-service"
    TARGET_DEPLOYMENT="ts-route-service"
    CHANGE_TARGET_COMPONENT_ID="ts-route-service"
    AFFECTED_COMPONENT_IDS="ts-route-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-route-service"
    ;;
  unexpected_train_scale_zero)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_train_scale_to_zero"
    CHANGE_FAMILY_ID="unexpected_service_termination"
    IMPLEMENTATION_ID="train_replicas_1_to_0_cloud_v1"
    COMPONENT_ID="ts-train-service"
    TARGET_DEPLOYMENT="ts-train-service"
    CHANGE_TARGET_COMPONENT_ID="ts-train-service"
    AFFECTED_COMPONENT_IDS="ts-train-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-train-service"
    ;;
  unexpected_station_scale_zero)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_station_scale_to_zero"
    CHANGE_FAMILY_ID="unexpected_service_termination"
    IMPLEMENTATION_ID="station_replicas_1_to_0_cloud_v1"
    COMPONENT_ID="ts-station-service"
    TARGET_DEPLOYMENT="ts-station-service"
    CHANGE_TARGET_COMPONENT_ID="ts-station-service"
    AFFECTED_COMPONENT_IDS="ts-station-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-station-service"
    ;;
  unexpected_gateway_scale_zero)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_gateway_scale_to_zero"
    CHANGE_FAMILY_ID="unexpected_service_termination"
    IMPLEMENTATION_ID="gateway_replicas_1_to_0_cloud_v1"
    COMPONENT_ID="ts-gateway-service"
    TARGET_DEPLOYMENT="ts-gateway-service"
    CHANGE_TARGET_COMPONENT_ID="ts-gateway-service"
    AFFECTED_COMPONENT_IDS="ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service"
    ;;
  unexpected_seat_scale_zero_weak)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_seat_scale_to_zero_weak"
    CHANGE_FAMILY_ID="unexpected_low_coverage_service_loss"
    IMPLEMENTATION_ID="seat_replicas_1_to_0_cloud_v1"
    COMPONENT_ID="ts-seat-service"
    TARGET_DEPLOYMENT="ts-seat-service"
    CHANGE_TARGET_COMPONENT_ID="ts-seat-service"
    AFFECTED_COMPONENT_IDS="ts-seat-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-seat-service"
    ;;
  unexpected_user_scale_zero_weak)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_user_scale_to_zero_weak"
    CHANGE_FAMILY_ID="unexpected_low_coverage_service_loss"
    IMPLEMENTATION_ID="user_replicas_1_to_0_cloud_v1"
    COMPONENT_ID="ts-user-service"
    TARGET_DEPLOYMENT="ts-user-service"
    CHANGE_TARGET_COMPONENT_ID="ts-user-service"
    AFFECTED_COMPONENT_IDS="ts-user-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-user-service"
    ;;
  unexpected_basic_scale_zero_weak)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_basic_scale_to_zero_weak"
    CHANGE_FAMILY_ID="unexpected_low_coverage_service_loss"
    IMPLEMENTATION_ID="basic_replicas_1_to_0_cloud_v1"
    COMPONENT_ID="ts-basic-service"
    TARGET_DEPLOYMENT="ts-basic-service"
    CHANGE_TARGET_COMPONENT_ID="ts-basic-service"
    AFFECTED_COMPONENT_IDS="ts-basic-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-basic-service"
    ;;
  unexpected_config_scale_zero_weak)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_config_scale_to_zero_weak"
    CHANGE_FAMILY_ID="unexpected_low_coverage_service_loss"
    IMPLEMENTATION_ID="config_replicas_1_to_0_cloud_v1"
    COMPONENT_ID="ts-config-service"
    TARGET_DEPLOYMENT="ts-config-service"
    CHANGE_TARGET_COMPONENT_ID="ts-config-service"
    AFFECTED_COMPONENT_IDS="ts-config-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-config-service"
    ;;
  expected_pod_migration_price_replacement)
    RUN_TYPE="expected"
    SEMANTIC_LABEL="expected"
    BENCHMARK_LABEL="candidate_expected_change"
    SCENARIO_ID="cloud_expected_price_pod_replacement"
    CHANGE_FAMILY_ID="expected_pod_migration"
    IMPLEMENTATION_ID="delete_one_price_pod_replicas_2_cloud_v1"
    COMPONENT_ID="ts-price-service"
    TARGET_DEPLOYMENT="ts-price-service"
    CHANGE_TARGET_COMPONENT_ID="ts-price-service"
    AFFECTED_COMPONENT_IDS="ts-price-service,ts-gateway-service,node"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service"
    ;;
  unexpected_resource_limit_price_too_small)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_price_resource_limit_small"
    CHANGE_FAMILY_ID="unexpected_resource_limit"
    IMPLEMENTATION_ID="price_cpu_memory_too_small_cloud_v1"
    COMPONENT_ID="ts-price-service"
    TARGET_DEPLOYMENT="ts-price-service"
    CHANGE_TARGET_COMPONENT_ID="ts-price-service"
    AFFECTED_COMPONENT_IDS="ts-price-service,ts-gateway-service,node"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service"
    ;;
  unexpected_station_jvm_oom)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_station_jvm_oom"
    CHANGE_FAMILY_ID="unexpected_resource_limit"
    IMPLEMENTATION_ID="station_jvm_heap_exceeds_memory_limit_cloud_v1"
    COMPONENT_ID="ts-station-service"
    TARGET_DEPLOYMENT="ts-station-service"
    CHANGE_TARGET_COMPONENT_ID="ts-station-service"
    AFFECTED_COMPONENT_IDS="ts-station-service,ts-gateway-service,node"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-station-service"
    ;;
  unexpected_gateway_jvm_oom)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_gateway_jvm_oom"
    CHANGE_FAMILY_ID="unexpected_resource_limit"
    IMPLEMENTATION_ID="gateway_jvm_heap_exceeds_memory_limit_cloud_v1"
    COMPONENT_ID="ts-gateway-service"
    TARGET_DEPLOYMENT="ts-gateway-service"
    CHANGE_TARGET_COMPONENT_ID="ts-gateway-service"
    AFFECTED_COMPONENT_IDS="ts-gateway-service,node"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service,ts-route-service,ts-train-service,ts-station-service"
    ;;
  unexpected_connection_pool_exhaustion_price)
    RUN_TYPE="unexpected"
    SEMANTIC_LABEL="unexpected"
    BENCHMARK_LABEL="candidate_unexpected_change"
    SCENARIO_ID="cloud_unexpected_price_pool_exhaustion"
    CHANGE_FAMILY_ID="unexpected_pool_exhaustion"
    IMPLEMENTATION_ID="price_pool_too_small_cloud_v1"
    COMPONENT_ID="ts-price-service"
    TARGET_DEPLOYMENT="ts-price-service"
    CHANGE_TARGET_COMPONENT_ID="ts-price-service"
    AFFECTED_COMPONENT_IDS="ts-price-service,ts-gateway-service,tsdb-mysql"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service"
    ;;
  unexpected_service_kill_replacement_low_oov)
    RUN_TYPE="boundary"
    SEMANTIC_LABEL="indeterminate"
    BENCHMARK_LABEL="candidate_indeterminate_boundary"
    SCENARIO_ID="cloud_boundary_price_pod_kill_replacement_low_oov"
    CHANGE_FAMILY_ID="boundary_runtime_replacement"
    IMPLEMENTATION_ID="delete_price_pod_single_replica_boundary_cloud_v1"
    COMPONENT_ID="ts-price-service"
    TARGET_DEPLOYMENT="ts-price-service"
    CHANGE_TARGET_COMPONENT_ID="ts-price-service"
    AFFECTED_COMPONENT_IDS="ts-price-service,ts-gateway-service"
    ORACLE_COMPONENT_IDS="ts-gateway-service,ts-price-service"
    ;;
  *)
    echo "Unsupported scenario: $SCENARIO" >&2
    exit 2
    ;;
esac

POST_WORKLOAD_PROFILE_ID="${POST_WORKLOAD_PROFILE_ID:-$WORKLOAD_PROFILE_ID}"
RUN_WORKLOAD_PROFILE_ID="$WORKLOAD_PROFILE_ID"
if [[ "$POST_WORKLOAD_PROFILE_ID" != "$WORKLOAD_PROFILE_ID" ]]; then
  RUN_WORKLOAD_PROFILE_ID="$WORKLOAD_PROFILE_ID=>$POST_WORKLOAD_PROFILE_ID"
fi

echo "run_id=$RUN_ID"
echo "run_dir=$RUN_DIR"
echo "scenario=$SCENARIO semantic=$SEMANTIC_LABEL target=$TARGET_DEPLOYMENT"

echo
echo "== Restore baseline state before run =="
restore_baseline_state

echo
echo "== Cloud state before run =="
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/collect_cloud_run_state.py" \
  --namespace "$NAMESPACE" \
  --kubectl "$KUBECTL" \
  --run-dir "$RUN_DIR" \
  --phase before \
  --base-url "$BASE_URL" \
  --target-deployment "$TARGET_DEPLOYMENT" \
  --output "$RUN_DIR/cloud_state_before.json"

echo
echo "== Pre-change stable-state gate =="
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/wait_stable_state.py" \
  --namespace "$NAMESPACE" \
  --run-dir "$RUN_DIR" \
  --kubectl "$KUBECTL" \
  --python "$PYTHON" \
  --stable-seconds "$STABLE_SECONDS" \
  --api-timeout-seconds "$WORKLOAD_TIMEOUT_SECONDS" \
  --gateway-base-url "$BASE_URL" \
  --allowlist "$ALLOWLIST"
cp "$RUN_DIR/stable_state_report.json" "$RUN_DIR/pre_change_stable_report.json"

if [[ "$WARMUP_SECONDS" -gt 0 ]]; then
  echo
  echo "== Pre-collection workload warmup =="
  "$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/run_workload.py" \
    --allowlist "$ALLOWLIST" \
    --profile-file "$PROFILES" \
    --profile-id "$WORKLOAD_PROFILE_ID" \
    --seed "$SEED" \
    --base-url "$BASE_URL" \
    --duration-seconds "$WARMUP_SECONDS" \
    --rate-per-second "$RATE_PER_SECOND" \
    --timeout-seconds "$WORKLOAD_TIMEOUT_SECONDS" \
    --output "$RUN_DIR/warmup_oracle_calls.jsonl" \
    --summary-output "$RUN_DIR/warmup_workload_profile.json"
fi

echo
echo "== Start run-bounded log watcher =="
RUN_START_TIME="$(utc_now)"
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/log_watcher.py" \
  --namespace "$NAMESPACE" \
  --kubectl "$KUBECTL" \
  --run_dir "$RUN_DIR" \
  --duration_seconds "$WATCHER_SECONDS" \
  --run_start_time "$RUN_START_TIME" \
  --run_id "$RUN_ID" \
  --run_type "$RUN_TYPE" \
  --cluster_type "$CLUSTER_TYPE" \
  --scenario_id "$SCENARIO_ID" \
  --benchmark_label "$BENCHMARK_LABEL" \
  --semantic_label "$SEMANTIC_LABEL" \
  --change_family_id "$CHANGE_FAMILY_ID" \
  --implementation_id "$IMPLEMENTATION_ID" \
  --component_id "$COMPONENT_ID" \
  --change_target_component_id "$CHANGE_TARGET_COMPONENT_ID" \
  --affected_component_ids "$AFFECTED_COMPONENT_IDS" \
  --oracle_component_ids "$ORACLE_COMPONENT_IDS" \
  --batch_id "$BATCH_ID" \
  --pair_block_id "$BLOCK_ID" \
  --batch_position "$BATCH_POSITION" \
  --preceding_scenario "$PRECEDING_SCENARIO" \
  --matched_workload_seed "$MATCHED_WORKLOAD_SEED" \
  --hours_since_cluster_start "$HOURS_SINCE_CLUSTER_START" \
  --workload_profile_id "$RUN_WORKLOAD_PROFILE_ID" \
  --workload_seed "$SEED" \
  > "$RUN_DIR/log_watcher_stdout.log" 2> "$RUN_DIR/log_watcher_stderr.log" &
WATCHER_PID=$!
echo "Watcher PID: $WATCHER_PID"
for _ in $(seq 1 30); do
  [[ -f "$RUN_DIR/phase_timeline.json" ]] && break
  sleep 1
done
if [[ ! -f "$RUN_DIR/phase_timeline.json" ]]; then
  echo "Watcher did not create phase_timeline.json" >&2
  exit 1
fi
update_json "$RUN_DIR/phase_timeline.json" "{\"pre_change_start\":\"$RUN_START_TIME\"}"

echo
echo "== Pre-change workload profile $WORKLOAD_PROFILE_ID =="
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/run_workload.py" \
  --allowlist "$ALLOWLIST" \
  --profile-file "$PROFILES" \
  --profile-id "$WORKLOAD_PROFILE_ID" \
  --seed "$SEED" \
  --base-url "$BASE_URL" \
  --duration-seconds "$PRE_CHANGE_SECONDS" \
  --rate-per-second "$RATE_PER_SECOND" \
  --timeout-seconds "$WORKLOAD_TIMEOUT_SECONDS" \
  --output "$RUN_DIR/pre_change_oracle_calls.jsonl" \
  --summary-output "$RUN_DIR/pre_change_workload_profile.json"

echo
echo "== Apply scenario change $SCENARIO_ID =="
CHANGE_START="$(utc_now)"
update_json "$RUN_DIR/phase_timeline.json" "{\"change_start_time\":\"$CHANGE_START\"}"
ROLLOUT_TIMEOUT="${TRANSITION_BUDGET_SECONDS}s"
case "$SCENARIO" in
  expected_workload_mix_*)
    echo "No deployment mutation; post workload changes from $WORKLOAD_PROFILE_ID to $POST_WORKLOAD_PROFILE_ID."
    DEPLOYMENT_COMPLETE="$(utc_now)"
    ;;
  expected_compatible_config_price_pool_increase)
    "$KUBECTL" set env "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" \
      SPRING_DATASOURCE_HIKARI_MAXIMUM_POOL_SIZE=20 \
      SCWARN_COMPAT_CONFIG_TOKEN="$RUN_ID"
    "$KUBECTL" rollout status "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"
    DEPLOYMENT_COMPLETE="$(utc_now)"
    ;;
  expected_compatible_config_route_timeout_valid)
    "$KUBECTL" set env "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" \
      SPRING_DATASOURCE_HIKARI_CONNECTION_TIMEOUT=30000 \
      SCWARN_COMPAT_CONFIG_TOKEN="$RUN_ID"
    "$KUBECTL" rollout status "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"
    DEPLOYMENT_COMPLETE="$(utc_now)"
    ;;
  expected_scale_*_1_to_2)
    "$KUBECTL" scale "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --replicas=2
    "$KUBECTL" rollout status "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"
    DEPLOYMENT_COMPLETE="$(utc_now)"
    ;;
  expected_pod_migration_price_replacement)
    "$KUBECTL" scale "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --replicas=2
    "$KUBECTL" rollout status "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"
    POD_TO_DELETE="$("$KUBECTL" get pod -n "$NAMESPACE" -l "app=$TARGET_DEPLOYMENT" -o jsonpath='{.items[0].metadata.name}')"
    if [[ -z "$POD_TO_DELETE" ]]; then
      echo "No pod found for expected pod replacement scenario: $TARGET_DEPLOYMENT" >&2
      exit 2
    fi
    "$KUBECTL" delete pod "$POD_TO_DELETE" -n "$NAMESPACE" --wait=false
    "$KUBECTL" rollout status "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"
    DEPLOYMENT_COMPLETE="$(utc_now)"
    ;;
  expected_low_impact_*_annotation)
    "$KUBECTL" annotate "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" "scwarn-formal-low-impact=$RUN_ID" --overwrite
    DEPLOYMENT_COMPLETE="$(utc_now)"
    ;;
  unexpected_*_bad_db_port)
    if [[ -z "$BAD_DB_PORT_ENV" || -z "$TARGET_DEPLOYMENT" ]]; then
      echo "Bad-port scenario missing BAD_DB_PORT_ENV or TARGET_DEPLOYMENT" >&2
      exit 2
    fi
    "$KUBECTL" set env "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" "$BAD_DB_PORT_ENV=3399"
    set +e
    "$KUBECTL" rollout status "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"
    SCENARIO_ROLLOUT_EXIT_CODE=$?
    set -e
    DEPLOYMENT_COMPLETE="$(utc_now)"
    if [[ "$SCENARIO_ROLLOUT_EXIT_CODE" -ne 0 ]]; then
      SCENARIO_ROLLOUT_COMPLETED=false
      SCENARIO_FAILURE_STATE="deployment_failure"
      SCENARIO_DEPLOYMENT_FAILED_TIME="$DEPLOYMENT_COMPLETE"
    fi
    ;;
  unexpected_resource_limit_price_too_small)
    "$KUBECTL" set resources "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" \
      --limits=cpu=20m,memory=128Mi --requests=cpu=10m,memory=64Mi
    set +e
    "$KUBECTL" rollout status "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"
    SCENARIO_ROLLOUT_EXIT_CODE=$?
    set -e
    DEPLOYMENT_COMPLETE="$(utc_now)"
    if [[ "$SCENARIO_ROLLOUT_EXIT_CODE" -ne 0 ]]; then
      SCENARIO_ROLLOUT_COMPLETED=false
      SCENARIO_FAILURE_STATE="deployment_failure"
      SCENARIO_DEPLOYMENT_FAILED_TIME="$DEPLOYMENT_COMPLETE"
    fi
    ;;
  unexpected_station_jvm_oom|unexpected_gateway_jvm_oom)
    "$KUBECTL" set env "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" \
      'JAVA_TOOL_OPTIONS=-Xms256m -Xmx2g'
    "$KUBECTL" set resources "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" \
      --limits=cpu=100m,memory=128Mi --requests=cpu=25m,memory=64Mi
    "$KUBECTL" scale "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --replicas=0
    "$KUBECTL" rollout status "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"
    "$KUBECTL" scale "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --replicas=1
    set +e
    "$KUBECTL" rollout status "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"
    SCENARIO_ROLLOUT_EXIT_CODE=$?
    set -e
    DEPLOYMENT_COMPLETE="$(utc_now)"
    if [[ "$SCENARIO_ROLLOUT_EXIT_CODE" -ne 0 ]]; then
      SCENARIO_ROLLOUT_COMPLETED=false
      SCENARIO_FAILURE_STATE="deployment_failure"
      SCENARIO_DEPLOYMENT_FAILED_TIME="$DEPLOYMENT_COMPLETE"
    fi
    ;;
  unexpected_connection_pool_exhaustion_price)
    "$KUBECTL" set env "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" \
      SPRING_DATASOURCE_HIKARI_MAXIMUM_POOL_SIZE=1 \
      SPRING_DATASOURCE_HIKARI_CONNECTION_TIMEOUT=250
    set +e
    "$KUBECTL" rollout status "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --timeout="$ROLLOUT_TIMEOUT"
    SCENARIO_ROLLOUT_EXIT_CODE=$?
    set -e
    DEPLOYMENT_COMPLETE="$(utc_now)"
    if [[ "$SCENARIO_ROLLOUT_EXIT_CODE" -ne 0 ]]; then
      SCENARIO_ROLLOUT_COMPLETED=false
      SCENARIO_FAILURE_STATE="deployment_failure"
      SCENARIO_DEPLOYMENT_FAILED_TIME="$DEPLOYMENT_COMPLETE"
    fi
    ;;
  unexpected_service_kill_replacement_low_oov)
    POD_TO_DELETE="$("$KUBECTL" get pod -n "$NAMESPACE" -l "app=$TARGET_DEPLOYMENT" -o jsonpath='{.items[0].metadata.name}')"
    if [[ -z "$POD_TO_DELETE" ]]; then
      echo "No pod found for pod-kill scenario: $TARGET_DEPLOYMENT" >&2
      exit 2
    fi
    "$KUBECTL" delete pod "$POD_TO_DELETE" -n "$NAMESPACE" --wait=false
    SCENARIO_ROLLOUT_COMPLETED=false
    SCENARIO_FAILURE_STATE="runtime_failure"
    SCENARIO_DEPLOYMENT_FAILED_TIME="$(utc_now)"
    DEPLOYMENT_COMPLETE="$SCENARIO_DEPLOYMENT_FAILED_TIME"
    ;;
  unexpected_*_scale_zero|unexpected_*_scale_zero_weak)
    "$KUBECTL" scale "deployment/$TARGET_DEPLOYMENT" -n "$NAMESPACE" --replicas=0
    SCENARIO_ROLLOUT_COMPLETED=false
    SCENARIO_FAILURE_STATE="runtime_failure"
    SCENARIO_DEPLOYMENT_FAILED_TIME="$(utc_now)"
    DEPLOYMENT_COMPLETE="$SCENARIO_DEPLOYMENT_FAILED_TIME"
    ;;
esac
if [[ "$SCENARIO_ROLLOUT_COMPLETED" == "true" ]]; then
  update_json "$RUN_DIR/phase_timeline.json" "{\"deployment_complete_time\":\"$DEPLOYMENT_COMPLETE\",\"deployment_rollout_exit_code\":$SCENARIO_ROLLOUT_EXIT_CODE,\"deployment_rollout_completed\":true}"
else
  update_json "$RUN_DIR/phase_timeline.json" "{\"deployment_failed_time\":\"$SCENARIO_DEPLOYMENT_FAILED_TIME\",\"post_change_observation_start_time\":\"$SCENARIO_DEPLOYMENT_FAILED_TIME\",\"deployment_rollout_exit_code\":$SCENARIO_ROLLOUT_EXIT_CODE,\"deployment_rollout_completed\":false}"
fi
cat > "$RUN_DIR/change_report.json" <<JSON
{
  "generated_at": "$(utc_now)",
  "scenario": "$SCENARIO",
  "scenario_id": "$SCENARIO_ID",
  "change_family_id": "$CHANGE_FAMILY_ID",
  "implementation_id": "$IMPLEMENTATION_ID",
  "component_id": "$COMPONENT_ID",
  "change_target_component_id": "$CHANGE_TARGET_COMPONENT_ID",
  "affected_component_ids": "$AFFECTED_COMPONENT_IDS",
  "oracle_component_ids": "$ORACLE_COMPONENT_IDS",
  "batch_id": "$BATCH_ID",
  "block_id": "$BLOCK_ID",
  "batch_position": "$BATCH_POSITION",
  "preceding_scenario": "$PRECEDING_SCENARIO",
  "matched_workload_seed": "$MATCHED_WORKLOAD_SEED",
  "hours_since_cluster_start": "$HOURS_SINCE_CLUSTER_START",
  "pre_workload_profile_id": "$WORKLOAD_PROFILE_ID",
  "post_workload_profile_id": "$POST_WORKLOAD_PROFILE_ID",
  "workload_profile_changed": $([[ "$POST_WORKLOAD_PROFILE_ID" != "$WORKLOAD_PROFILE_ID" ]] && echo true || echo false),
  "change_start_time": "$CHANGE_START",
  "deployment_rollout_completed": $SCENARIO_ROLLOUT_COMPLETED,
  "deployment_rollout_exit_code": $SCENARIO_ROLLOUT_EXIT_CODE,
  "deployment_complete_time": "$DEPLOYMENT_COMPLETE",
  "deployment_failed_time": "$SCENARIO_DEPLOYMENT_FAILED_TIME",
  "unexpected_failure_ratio_threshold": $UNEXPECTED_FAILURE_RATIO
}
JSON

echo
echo "== Post-change stable/failure observation gate =="
if [[ "$SCENARIO_ROLLOUT_COMPLETED" == "true" ]]; then
  "$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/wait_stable_state.py" \
    --namespace "$NAMESPACE" \
    --run-dir "$RUN_DIR" \
    --kubectl "$KUBECTL" \
    --python "$PYTHON" \
    --stable-seconds "$POST_STABLE_SECONDS" \
    --timeout-seconds "$TRANSITION_BUDGET_SECONDS" \
    --api-timeout-seconds "$WORKLOAD_TIMEOUT_SECONDS" \
    --gateway-base-url "$BASE_URL" \
    --allowlist "$ALLOWLIST" || true
  cp "$RUN_DIR/stable_state_report.json" "$RUN_DIR/post_change_stable_report.json"
else
  cat > "$RUN_DIR/post_change_stable_report.json" <<JSON
{
  "generated_at": "$(utc_now)",
  "namespace": "$NAMESPACE",
  "deployment_success": false,
  "deployment_failed": $([[ "$SCENARIO_FAILURE_STATE" == "deployment_failure" ]] && echo true || echo false),
  "readiness_success": false,
  "service_registered": false,
  "post_change_state": "$SCENARIO_FAILURE_STATE",
  "deployment_failed_time": "$SCENARIO_DEPLOYMENT_FAILED_TIME",
  "stable_required_seconds": $POST_STABLE_SECONDS,
  "timeout_seconds": $TRANSITION_BUDGET_SECONDS,
  "reason": "pre_registered_cloud_formal_failure_observation"
}
JSON
  cp "$RUN_DIR/post_change_stable_report.json" "$RUN_DIR/stable_state_report.json"
fi
"$PYTHON" - "$RUN_DIR" <<'PY'
import json
import sys
from pathlib import Path
run_dir = Path(sys.argv[1])
timeline = json.loads((run_dir / "phase_timeline.json").read_text(encoding="utf-8-sig"))
stable = json.loads((run_dir / "post_change_stable_report.json").read_text(encoding="utf-8-sig"))
if stable.get("stable_state_start_time"):
    timeline["stable_state_start_time"] = stable["stable_state_start_time"]
    timeline["post_change_observation_start_time"] = stable["stable_state_start_time"]
elif stable.get("deployment_failed_time"):
    timeline["deployment_failed_time"] = stable["deployment_failed_time"]
    timeline["post_change_observation_start_time"] = stable["deployment_failed_time"]
if stable.get("post_change_state"):
    timeline["post_change_state"] = stable["post_change_state"]
(run_dir / "phase_timeline.json").write_text(json.dumps(timeline, ensure_ascii=False, indent=2), encoding="utf-8")
PY

echo
echo "== Post-change workload profile $POST_WORKLOAD_PROFILE_ID =="
if [[ "$SCENARIO_ROLLOUT_COMPLETED" == "true" ]]; then
  update_json "$RUN_DIR/phase_timeline.json" "{\"post_change_observation_start_time\":\"$(utc_now)\"}"
fi
set +e
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/run_workload.py" \
  --allowlist "$ALLOWLIST" \
  --profile-file "$PROFILES" \
  --profile-id "$POST_WORKLOAD_PROFILE_ID" \
  --seed "$SEED" \
  --base-url "$BASE_URL" \
  --duration-seconds "$POST_CHANGE_SECONDS" \
  --rate-per-second "$RATE_PER_SECOND" \
  --timeout-seconds "$WORKLOAD_TIMEOUT_SECONDS" \
  --output "$RUN_DIR/oracle_calls.jsonl" \
  --summary-output "$RUN_DIR/post_change_workload_profile.json"
POST_WORKLOAD_EXIT=$?
set -e
echo "Post workload exit code: $POST_WORKLOAD_EXIT"
update_json "$RUN_DIR/phase_timeline.json" "{\"observation_end_time\":\"$(utc_now)\"}"

echo
echo "== Wait for watcher completion =="
wait "$WATCHER_PID"
WATCHER_PID=""

echo
echo "== Materialize and audit logs =="
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/materialize_logs_jsonl.py" --run_dir "$RUN_DIR"
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/slice_logs_by_time.py" \
  --run_dir "$RUN_DIR" \
  --start_key post_change_observation_start_time \
  --output "$RUN_DIR/post_change_logs.jsonl"
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/audit_log_observability.py" \
  --run_dir "$RUN_DIR" \
  --allowlist "$ALLOWLIST" \
  --logs "$RUN_DIR/post_change_logs.jsonl" \
  --window_start_key post_change_observation_start_time \
  --window_end_key observation_end_time \
  --post_call_margin_seconds "$POST_CALL_MARGIN_SECONDS"
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/summarize_log_observability_gaps.py" --run_dir "$RUN_DIR"

echo
echo "== Classify semantic label =="
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/classify_semantic_label.py" \
  --run_dir "$RUN_DIR" \
  --expected_semantic "$SEMANTIC_LABEL" \
  --unexpected_failure_ratio "$UNEXPECTED_FAILURE_RATIO" \
  --indeterminate_failure_ratio_min "$INDETERMINATE_FAILURE_RATIO_MIN"

echo
echo "== Finalize metadata =="
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/finalize_run_metadata.py" \
  --run_dir "$RUN_DIR" \
  --namespace "$NAMESPACE" \
  --kubectl "$KUBECTL" \
  --run_type "$RUN_TYPE" \
  --scenario_id "$SCENARIO_ID" \
  --benchmark_label "$BENCHMARK_LABEL" \
  --semantic_label "$SEMANTIC_LABEL" \
  --change_family_id "$CHANGE_FAMILY_ID" \
  --implementation_id "$IMPLEMENTATION_ID" \
  --component_id "$COMPONENT_ID" \
  --change_target_component_id "$CHANGE_TARGET_COMPONENT_ID" \
  --affected_component_ids "$AFFECTED_COMPONENT_IDS" \
  --oracle_component_ids "$ORACLE_COMPONENT_IDS" \
  --batch_id "$BATCH_ID" \
  --pair_block_id "$BLOCK_ID" \
  --batch_position "$BATCH_POSITION" \
  --preceding_scenario "$PRECEDING_SCENARIO" \
  --matched_workload_seed "$MATCHED_WORKLOAD_SEED" \
  --hours_since_cluster_start "$HOURS_SINCE_CLUSTER_START"

update_json "$RUN_DIR/run_manifest.json" "{\"cloud_protocol_version\":\"$CLOUD_PROTOCOL_VERSION\",\"cloud_freeze_id\":\"$CLOUD_FREEZE_ID\",\"cloud_cluster_type\":\"$CLUSTER_TYPE\",\"cloud_drift_gate_threshold_file\":\"$THRESHOLD_FILE\",\"cloud_drift_gate_threshold_sha256\":\"$THRESHOLD_SHA256\",\"cloud_formal_unexpected_failure_ratio\":$UNEXPECTED_FAILURE_RATIO,\"cloud_formal_indeterminate_failure_ratio_min\":$INDETERMINATE_FAILURE_RATIO_MIN,\"post_workload_profile_id\":\"$POST_WORKLOAD_PROFILE_ID\",\"scenario_preference\":\"$SCENARIO_PREFERENCE\"}"

echo
echo "== Cloud state after run before cleanup =="
set +e
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/collect_cloud_run_state.py" \
  --namespace "$NAMESPACE" \
  --kubectl "$KUBECTL" \
  --run-dir "$RUN_DIR" \
  --phase after \
  --base-url "$BASE_URL" \
  --target-deployment "$TARGET_DEPLOYMENT" \
  --output "$RUN_DIR/cloud_state_after.json"
set -e

echo
echo "== Cleanup to baseline state =="
update_json "$RUN_DIR/phase_timeline.json" "{\"cleanup_start_time\":\"$(utc_now)\"}"
restore_baseline_state
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/wait_cloud_cleanup_convergence.py" \
  --namespace "$NAMESPACE" \
  --kubectl "$KUBECTL" \
  --timeout-seconds "$TRANSITION_BUDGET_SECONDS" \
  --poll-seconds 5 \
  --output "$RUN_DIR/cleanup_convergence_report.json"
update_json "$RUN_DIR/phase_timeline.json" "{\"cleanup_complete_time\":\"$(utc_now)\"}"

echo
echo "== Cloud state after cleanup =="
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/collect_cloud_run_state.py" \
  --namespace "$NAMESPACE" \
  --kubectl "$KUBECTL" \
  --run-dir "$RUN_DIR" \
  --phase cleanup \
  --base-url "$BASE_URL" \
  --target-deployment "$TARGET_DEPLOYMENT" \
  --output "$RUN_DIR/cloud_state_cleanup.json"

echo
echo "== Cloud formal quality gate =="
"$PYTHON" "$REPO_ROOT/dataset/trainticket_collection/check_cloud_formal_run_quality.py" \
  --run-dir "$RUN_DIR" \
  --expected-cluster-type "$CLUSTER_TYPE" \
  --cloud-freeze-id "$CLOUD_FREEZE_ID" \
  --threshold-sha256 "$THRESHOLD_SHA256"

echo
echo "Run directory: $RUN_DIR"
