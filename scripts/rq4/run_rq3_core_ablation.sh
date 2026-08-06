#!/usr/bin/env bash
set -euo pipefail

# Evaluate the four nested RQ3 component variants with one frozen checkpoint.
# This script never trains the backbone. Calibration statistics are fitted on
# dev by main.py; all reported run-level metrics are computed on test.

[[ -f main.py && -d scripts/rq4 ]] || {
  echo "ERROR: run this script from the DeniAD repository root." >&2
  exit 1
}

data_dir=${DATA_DIR:-./data_cloud_expected_unexpected_combined67_v0_3_1}
checkpoint=${CHECKPOINT:-./checkpoints/rq4/cloud_pilot/cloud_pilot_combined67_logonly_gmm_ep60.pth}
result_dir=${RESULT_DIR:-./results/rq3/core_ablation}
run_tag=${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}
threshold_json=${THRESHOLD_JSON:-}
absence_context_mode=${ABSENCE_CONTEXT_MODE:-hybrid}
absence_metadata_fields=${ABSENCE_METADATA_FIELDS:-affected_component_ids,change_target_component_id}

if [ ! -f "$checkpoint" ]; then
  echo "ERROR: checkpoint not found: $checkpoint" >&2
  exit 1
fi
for split in train dev test; do
  if [ ! -f "$data_dir/$split.pkl" ]; then
    echo "ERROR: missing $data_dir/$split.pkl" >&2
    exit 1
  fi
done
mkdir -p "$result_dir"

if [ -n "$threshold_json" ]; then
  if [ ! -f "$threshold_json" ]; then
    echo "ERROR: frozen threshold JSON not found: $threshold_json" >&2
    exit 1
  fi
  IFS=$'\t' read -r frozen_policy frozen_e frozen_u frozen_r frozen_c frozen_rm frozen_um frozen_su frozen_eum < <(
    python - "$threshold_json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    p = json.load(f)
values = [
    p.get("decision_policy", "evidence_priority"),
    p.get("expected_min", 0.35),
    p.get("unexpected_min", 0.20),
    p.get("reject_min", 0.20),
    p.get("conflict_min", 0.20),
    p.get("reject_margin", 0.15),
    p.get("unexpected_margin", 0.0),
    p.get("strong_unexpected_min", 0.50),
    p.get("expected_unexpected_max", 0.05),
]
print("\t".join(str(value) for value in values))
PY
  )
else
  frozen_policy=${RUN_LEVEL_DECISION_POLICY:-evidence_priority}
  frozen_e=${RUN_LEVEL_EXPECTED_MIN:-0.35}
  frozen_u=${RUN_LEVEL_UNEXPECTED_MIN:-0.20}
  frozen_r=${RUN_LEVEL_REJECT_MIN:-0.20}
  frozen_c=${RUN_LEVEL_CONFLICT_MIN:-0.20}
  frozen_rm=${RUN_LEVEL_REJECT_MARGIN:-0.15}
  frozen_um=${RUN_LEVEL_UNEXPECTED_MARGIN:-0.0}
  frozen_su=${RUN_LEVEL_STRONG_UNEXPECTED_MIN:-0.50}
  frozen_eum=${RUN_LEVEL_EXPECTED_UNEXPECTED_MAX:-0.05}
  echo "WARNING: THRESHOLD_JSON is not set; using environment/default thresholds." >&2
  echo "Do not report these as frozen test results unless the values were selected on dev." >&2
fi

export DATA_DIR="$data_dir"
export CHECKPOINT="$checkpoint"
export RESULT_DIR="$result_dir"
export RUN_TRAIN=0
export RQ4_CANDIDATE_MODE=${RQ4_CANDIDATE_MODE:-score}
export ABSENCE_CONTEXT_MODE="$absence_context_mode"
# Legacy Absence-aware revision fits its reference from the frozen train split.
# Do not inject the standalone CAM reference into this ablation.
export ABSENCE_REFERENCE_PATH=""
export ABSENCE_METADATA_FIELDS="$absence_metadata_fields"
export ABSENCE_K=${ABSENCE_K:-20}
export ABSENCE_PERSISTENCE_THRESHOLD=${ABSENCE_PERSISTENCE_THRESHOLD:-0.50}
export ABSENCE_MIN_CONTEXT_SIMILARITY=${ABSENCE_MIN_CONTEXT_SIMILARITY:-0.20}
export ABSENCE_MIN_QUERY_EXPOSURE=${ABSENCE_MIN_QUERY_EXPOSURE:-50}
export RUN_LEVEL_STATE_VETO=off
export USE_DRIFT_ADAPTER=0
export USE_COMPONENT_DRIFT_DIAGNOSIS=1
export USE_RQ4_WINDOW_DIAGNOSIS=1
export USE_TRACE_PROFILE=1
# Keep the ablation on the same score definition used for dev calibration and
# the formal RQ4 test.  Callers may still override both values explicitly.
export RQ4_ANOMALY_SCORE_MODE=${RQ4_ANOMALY_SCORE_MODE:-profile_zscore}
export RQ4_PROFILE_SCORE_WEIGHT=${RQ4_PROFILE_SCORE_WEIGHT:-1.0}
export DRIFT_CANDIDATE_PROFILE_QUANTILE=${DRIFT_CANDIDATE_PROFILE_QUANTILE:-0.95}
export RQ4_EVENT_DETAIL_INCLUDE_NORMAL=1

# These run-level thresholds must be frozen before test evaluation. Override
# them only with values selected on dev, never with values selected on test.
export RUN_LEVEL_DECISION_POLICY="$frozen_policy"
export RUN_LEVEL_EXPECTED_MIN="$frozen_e"
export RUN_LEVEL_UNEXPECTED_MIN="$frozen_u"
export RUN_LEVEL_REJECT_MIN="$frozen_r"
export RUN_LEVEL_CONFLICT_MIN="$frozen_c"
export RUN_LEVEL_REJECT_MARGIN="$frozen_rm"
export RUN_LEVEL_UNEXPECTED_MARGIN="$frozen_um"
export RUN_LEVEL_STRONG_UNEXPECTED_MIN="$frozen_su"
export RUN_LEVEL_EXPECTED_UNEXPECTED_MAX="$frozen_eum"

run_base() {
  echo "[RQ3] 1/4 Base selective type-time detector"
  RESULT_PREFIX="$result_dir/${run_tag}_base" \
  RUN_ID="${run_tag}_base" \
  USE_ENSEMBLE_CORRECTION=0 \
  USE_COUNTERFACTUAL_CONTEXT_SUPPORT=0 \
  USE_ABSENCE_AWARE_REVISION=0 \
  RUN_LEVEL_ABSENCE_VETO=0 \
  RUN_LEVEL_ABSENCE_UNEXPECTED_MODE=off \
  RUN_LEVEL_EVENT_PRED_COLUMN=score_selective \
    bash scripts/rq4/run_rq4_cloud_pilot.sh
}

run_memory() {
  echo "[RQ3] 2/4 + Memory correction"
  RESULT_PREFIX="$result_dir/${run_tag}_memory" \
  RUN_ID="${run_tag}_memory" \
  USE_ENSEMBLE_CORRECTION=1 \
  USE_COUNTERFACTUAL_CONTEXT_SUPPORT=0 \
  USE_ABSENCE_AWARE_REVISION=0 \
  RUN_LEVEL_ABSENCE_VETO=0 \
  RUN_LEVEL_ABSENCE_UNEXPECTED_MODE=off \
  RUN_LEVEL_EVENT_PRED_COLUMN=revised_selective \
    bash scripts/rq4/run_rq4_cloud_pilot.sh
}

run_historical() {
  echo "[RQ3] 3/4 + Counterfactual historical support"
  RESULT_PREFIX="$result_dir/${run_tag}_historical" \
  RUN_ID="${run_tag}_historical" \
  USE_ENSEMBLE_CORRECTION=1 \
  USE_COUNTERFACTUAL_CONTEXT_SUPPORT=1 \
  COUNTERFACTUAL_SUPPORT_TIME_MODE=${COUNTERFACTUAL_SUPPORT_TIME_MODE:-off} \
  USE_ABSENCE_AWARE_REVISION=0 \
  RUN_LEVEL_ABSENCE_VETO=0 \
  RUN_LEVEL_ABSENCE_UNEXPECTED_MODE=off \
  RUN_LEVEL_EVENT_PRED_COLUMN=revised_selective \
    bash scripts/rq4/run_rq4_cloud_pilot.sh
}

run_absence() {
  echo "[RQ3] 4/4 + Absence-aware revision"
  RESULT_PREFIX="$result_dir/${run_tag}_absence" \
  RUN_ID="${run_tag}_absence" \
  USE_ENSEMBLE_CORRECTION=1 \
  USE_COUNTERFACTUAL_CONTEXT_SUPPORT=1 \
  COUNTERFACTUAL_SUPPORT_TIME_MODE=${COUNTERFACTUAL_SUPPORT_TIME_MODE:-off} \
  USE_ABSENCE_AWARE_REVISION=1 \
  RUN_LEVEL_ABSENCE_VETO=1 \
  ABSENCE_APPLY_TO=normal_expected \
  RUN_LEVEL_ABSENCE_UNEXPECTED_MODE=strong \
  RUN_LEVEL_ABSENCE_STRONG_ANOMALY_THRESHOLD=${RUN_LEVEL_ABSENCE_STRONG_ANOMALY_THRESHOLD:-3.0} \
  RUN_LEVEL_ABSENCE_STRONG_COVERAGE_THRESHOLD=${RUN_LEVEL_ABSENCE_STRONG_COVERAGE_THRESHOLD:-1.0} \
  RUN_LEVEL_EVENT_PRED_COLUMN=revised_selective \
    bash scripts/rq4/run_rq4_cloud_pilot.sh
}

run_base
run_memory
run_historical
run_absence

python scripts/rq4/build_core_component_ablation.py \
  --summary_dir "$result_dir" \
  --base_summary "${run_tag}_base_run_level_summary.json" \
  --memory_summary "${run_tag}_memory_run_level_summary.json" \
  --historical_summary "${run_tag}_historical_run_level_summary.json" \
  --absence_summary "${run_tag}_absence_run_level_summary.json" \
  --output_csv "$result_dir/${run_tag}_core_component_ablation.csv" \
  --output_md "$result_dir/${run_tag}_core_component_ablation.md"

echo "[OK] RQ3 core ablation complete"
echo "Table: $result_dir/${run_tag}_core_component_ablation.md"
