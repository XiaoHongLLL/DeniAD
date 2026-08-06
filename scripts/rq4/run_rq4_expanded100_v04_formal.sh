#!/usr/bin/env bash
set -euo pipefail

# Formal expanded100 RQ4 pipeline with a selectable absence mechanism:
#   train(base train split) -> fit the configured normal reference
#   -> calibrate(38 dev runs) -> evaluate the untouched 100-run test split
#   -> merge Unexpected and Reject into the operational Unexpected* class.

[[ -f main.py && -d scripts/rq4 ]] || {
  echo "ERROR: run this script from the DeniAD repository root." >&2
  exit 1
}

python_bin=${PYTHON_BIN:-python}
data_dir=${DATA_DIR:-./data_cloud_expected_unexpected_expanded100_v0_4}
dev_view=${DEV_VIEW_DIR:-./data_cloud_expected_unexpected_expanded100_v0_4_dev_as_test}
checkpoint=${CHECKPOINT:-./checkpoints/rq4/cloud_pilot/cloud_pilot_expanded100_v04_logonly_gmm_ep60.pth}
result_root=${RESULT_ROOT:-./results/rq4/expanded100_v0_4}
absence_context_mode=${ABSENCE_CONTEXT_MODE:-context_memory}
if [[ "$absence_context_mode" == "context_memory" ]]; then
  absence_reference_path=${ABSENCE_REFERENCE_PATH:-./data_context_absence_memory_v0_5_control36/absence_memory.pkl}
else
  # Legacy hybrid/memory/metadata modes fit from the frozen train split unless
  # an explicit alternative reference is intentionally supplied.
  absence_reference_path=${ABSENCE_REFERENCE_PATH:-}
fi
absence_metadata_fields=${ABSENCE_METADATA_FIELDS:-}
if [[ "$absence_context_mode" == "context_memory" ]]; then
  method_label="DeniAD + Context-conditioned Absence Memory"
else
  method_label="DeniAD + Absence-aware revision"
fi
run_tag=${RUN_TAG:-expanded100_v04_cam_v05_$(date +%Y%m%d_%H%M%S)}
candidate_mode=${RQ4_CANDIDATE_MODE:-score_or_profile}
run_level_state_veto=${RUN_LEVEL_STATE_VETO:-off}
event_pred_column=${RUN_LEVEL_EVENT_PRED_COLUMN:-pred_drift_id}
rq4_include_labels=${RQ4_INCLUDE_LABELS:-expected_drift,successful_no_drift,unexpected_drift,unexpected_without_observable_log_drift}
expected_grid=${RQ4_EXPECTED_GRID:-0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50}
unexpected_grid=${RQ4_UNEXPECTED_GRID:-0.05,0.10,0.15,0.20,0.25,0.30}
unexpected_margin_grid=${RQ4_UNEXPECTED_MARGIN_GRID:-0.00,0.05,0.10}
strong_unexpected_grid=${RQ4_STRONG_UNEXPECTED_GRID:-0.40,0.50,0.60}
expected_unexpected_max_grid=${RQ4_EXPECTED_UNEXPECTED_MAX_GRID:-0.05,0.10,0.15}
selection_objective=${RQ4_SELECTION_OBJECTIVE:-balanced}
max_ufa=${RQ4_MAX_UFA:-0.25}
min_expected_f1=${RQ4_MIN_EXPECTED_F1:-0.05}
min_unexpected_f1=${RQ4_MIN_UNEXPECTED_F1:-0.00}
min_safe_rate=${RQ4_MIN_SAFE_RATE:-0.00}
run_train=${RUN_TRAIN:-1}
allow_degenerate_dev=${ALLOW_DEGENERATE_DEV:-0}
reuse_dev_events=${REUSE_DEV_EVENTS:-0}
calibration_only=${CALIBRATION_ONLY:-0}

if [[ "$candidate_mode" == "labeled" || "$candidate_mode" == "labeled_or_score" ]]; then
  echo "ERROR: $candidate_mode reads test drift labels and is not allowed for the formal main table." >&2
  echo "Use RQ4_CANDIDATE_MODE=score (default) or score_or_profile." >&2
  exit 1
fi

for name in train.pkl dev.pkl test.pkl annotation.csv metadata.json; do
  [[ -f "$data_dir/$name" ]] || { echo "ERROR: missing $data_dir/$name" >&2; exit 1; }
done
if [[ "$absence_context_mode" == "context_memory" ]]; then
  [[ -f "$absence_reference_path" ]] || { echo "ERROR: missing $absence_reference_path" >&2; exit 1; }
fi

if [[ "$absence_context_mode" == "context_memory" ]]; then
  echo "[0/5] Verifying the frozen 36-run trusted-normal memory and split isolation..."
  "$python_bin" scripts/rq4/verify_context_absence_memory.py \
    --memory "$absence_reference_path" \
    --dataset_dir "$data_dir" \
    --expected_count 36 \
    --output "$result_root/context_absence_memory_integrity.json"
else
  echo "[0/5] Legacy Absence-aware revision: fitting absence reference from the frozen train split."
fi

mkdir -p "$(dirname "$checkpoint")" "$result_root/dev" "$result_root/test" "$result_root/calibration"

if [[ "$run_train" == "1" || ! -f "$checkpoint" ]]; then
  echo "[1/5] Training expanded100 checkpoint from the 30-run reference split..."
  "$python_bin" main.py \
    -data "$data_dir/" \
    -normalize log \
    -d_model 128 -d_inner_hid 256 -n_head 4 -n_layers 4 -d_k 32 -d_v 32 \
    -dropout 0.1 -type_head gmm -lr 5e-5 \
    -loss_weighting fixed -fm_loss_weight 0.05 -loss_lambda 5.0 -fm_sigma 0.5 \
    -solver_method euler -solver_step_size 0.05 \
    -clamp_threshold 6.0 -flow_cond_clip 5.0 -min_max_len 0 -n_samples 100 \
    -checkpoint_metric valid_acc -checkpoint_min_delta 0.0001 \
    -context_mask_prob 0.1 -context_mask_min_history 1 \
    -seed 2023 -batch_size 64 -epoch 60 -eval_epoch 1000000 \
    -save_path "$checkpoint" \
    2>&1 | tee "$result_root/${run_tag}_train.log"
else
  echo "[1/5] Reusing checkpoint: $checkpoint"
fi

echo "[2/5] Creating a dev-as-test view for run-level threshold calibration..."
"$python_bin" scripts/rq4/make_dev_as_test_dataset.py \
  --source "$data_dir" \
  --output "$dev_view"

dev_prefix="$result_root/dev/${run_tag}_dev"
if [[ "$reuse_dev_events" == "1" && -f "${dev_prefix}_rq4_events.csv" ]]; then
  echo "[3/5] Reusing existing dev event details: ${dev_prefix}_rq4_events.csv"
else
  echo "[3/5] Evaluating the dev split without test-label candidate injection..."
  DATA_DIR="$dev_view" \
  CHECKPOINT="$checkpoint" \
  RESULT_DIR="$result_root/dev" \
  RESULT_PREFIX="$dev_prefix" \
  RUN_ID="${run_tag}_dev" \
  RUN_TRAIN=0 \
  RQ4_CANDIDATE_MODE="$candidate_mode" \
  USE_ABSENCE_AWARE_REVISION=1 \
  RUN_LEVEL_STATE_VETO="$run_level_state_veto" \
  RUN_LEVEL_ABSENCE_VETO=1 \
  RUN_LEVEL_ABSENCE_UNEXPECTED_MODE=strong \
  RUN_LEVEL_EVENT_PRED_COLUMN="$event_pred_column" \
  RUN_LEVEL_DECISION_POLICY=evidence_priority \
  RQ4_INCLUDE_LABELS="$rq4_include_labels" \
  ABSENCE_CONTEXT_MODE="$absence_context_mode" \
  ABSENCE_REFERENCE_PATH="$absence_reference_path" \
  ABSENCE_METADATA_FIELDS="$absence_metadata_fields" \
  ABSENCE_K="${ABSENCE_K:-20}" \
  ABSENCE_PERSISTENCE_THRESHOLD="${ABSENCE_PERSISTENCE_THRESHOLD:-0.50}" \
  ABSENCE_MIN_CONTEXT_SIMILARITY="${ABSENCE_MIN_CONTEXT_SIMILARITY:-0.20}" \
  ABSENCE_MIN_QUERY_EXPOSURE="${ABSENCE_MIN_QUERY_EXPOSURE:-50}" \
  RQ4_EVENT_DETAIL_MAX=400000 \
  bash scripts/rq4/run_rq4_cloud_pilot.sh
fi

calibration_dir="$result_root/calibration/$run_tag"
echo "[4/5] Freezing run-level thresholds on dev (UFA <= 0.25 when feasible)..."
"$python_bin" scripts/rq4/tune_rq4_run_level_thresholds.py \
  --data_dir "$dev_view" \
  --events_csv "${dev_prefix}_rq4_events.csv" \
  --batch_size 64 \
  --output_dir "$calibration_dir" \
  --event_pred_column "$event_pred_column" \
  --state_veto "$run_level_state_veto" \
  --include_labels "$rq4_include_labels" \
  --decision_policy evidence_priority \
  --expected_grid "$expected_grid" \
  --unexpected_grid "$unexpected_grid" \
  --reject_min 0.20 \
  --conflict_min 0.20 \
  --reject_margin 0.15 \
  --unexpected_margin 0.0 \
  --strong_unexpected_min 0.50 \
  --expected_unexpected_max 0.05 \
  --unexpected_margin_grid "$unexpected_margin_grid" \
  --strong_unexpected_grid "$strong_unexpected_grid" \
  --expected_unexpected_max_grid "$expected_unexpected_max_grid" \
  --selection_objective "$selection_objective" \
  --max_ufa "$max_ufa" \
  --min_expected_f1 "$min_expected_f1" \
  --min_unexpected_f1 "$min_unexpected_f1" \
  --min_safe_rate "$min_safe_rate" \
  --absence \
  --absence_context_mode "$absence_context_mode" \
  --absence_reference_path "$absence_reference_path" \
  --absence_metadata_fields "$absence_metadata_fields" \
  --absence_exclude_services system-observability,tsdb-mysql,nacosdb-mysql \
  --absence_k "${ABSENCE_K:-20}" \
  --absence_persistence_threshold "${ABSENCE_PERSISTENCE_THRESHOLD:-0.50}" \
  --absence_min_context_similarity "${ABSENCE_MIN_CONTEXT_SIMILARITY:-0.20}" \
  --absence_min_query_exposure "${ABSENCE_MIN_QUERY_EXPOSURE:-50}" \
  --absence_strong_anomaly_threshold 3.0 \
  --absence_strong_coverage_threshold 1.0

if [[ "$allow_degenerate_dev" != "1" ]]; then
  "$python_bin" - "$calibration_dir/best_threshold.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)

errors = []
if not cfg.get("constraint_satisfied", False):
    errors.append(
        "no dev threshold satisfies the operational UFA and Expected-F1 constraints"
    )
if float(cfg.get("Expected_F1", 0.0)) <= 0.0:
    errors.append("dev operational Expected_F1 is zero (degenerate acceptance path)")
if errors:
    raise SystemExit(
        "DEV GATE FAILED: " + "; ".join(errors)
        + ". Formal test was not started. Diagnose dev candidates first."
    )
print(
    "[OK] Dev gate passed: "
    f"Expected_F1={cfg.get('Expected_F1')} "
    f"Unexpected_F1={cfg.get('Unexpected_F1')} "
    f"Operational_UFA={cfg.get('Unexpected_False_Acceptance_Rate')}"
)
PY
fi

if [[ "$calibration_only" == "1" ]]; then
  echo "[OK] CALIBRATION_ONLY=1; stopping before the formal test split."
  echo "Frozen dev thresholds: $calibration_dir/best_threshold.json"
  echo "Dev threshold grid:     $calibration_dir/dev_threshold_grid.csv"
  exit 0
fi

read -r expected_min unexpected_min reject_min conflict_min reject_margin unexpected_margin strong_unexpected_min expected_unexpected_max < <(
  "$python_bin" - "$calibration_dir/best_threshold.json" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as f:
    cfg = json.load(f)
keys = (
    "expected_min",
    "unexpected_min",
    "reject_min",
    "conflict_min",
    "reject_margin",
    "unexpected_margin",
    "strong_unexpected_min",
    "expected_unexpected_max",
)
print(" ".join(str(cfg[key]) for key in keys))
PY
)

test_prefix="$result_root/test/${run_tag}_test"
echo "[5/5] Evaluating the untouched 100-run formal test split with frozen thresholds..."
DATA_DIR="$data_dir" \
CHECKPOINT="$checkpoint" \
RESULT_DIR="$result_root/test" \
RESULT_PREFIX="$test_prefix" \
RUN_ID="${run_tag}_test" \
RUN_TRAIN=0 \
RQ4_CANDIDATE_MODE="$candidate_mode" \
USE_ABSENCE_AWARE_REVISION=1 \
RUN_LEVEL_STATE_VETO="$run_level_state_veto" \
RUN_LEVEL_ABSENCE_VETO=1 \
RUN_LEVEL_ABSENCE_UNEXPECTED_MODE=strong \
RUN_LEVEL_EVENT_PRED_COLUMN="$event_pred_column" \
RUN_LEVEL_DECISION_POLICY=evidence_priority \
RQ4_INCLUDE_LABELS="$rq4_include_labels" \
RUN_LEVEL_EXPECTED_MIN="$expected_min" \
RUN_LEVEL_UNEXPECTED_MIN="$unexpected_min" \
RUN_LEVEL_REJECT_MIN="$reject_min" \
RUN_LEVEL_CONFLICT_MIN="$conflict_min" \
RUN_LEVEL_REJECT_MARGIN="$reject_margin" \
RUN_LEVEL_UNEXPECTED_MARGIN="$unexpected_margin" \
RUN_LEVEL_STRONG_UNEXPECTED_MIN="$strong_unexpected_min" \
RUN_LEVEL_EXPECTED_UNEXPECTED_MAX="$expected_unexpected_max" \
ABSENCE_CONTEXT_MODE="$absence_context_mode" \
ABSENCE_REFERENCE_PATH="$absence_reference_path" \
ABSENCE_METADATA_FIELDS="$absence_metadata_fields" \
ABSENCE_K="${ABSENCE_K:-20}" \
ABSENCE_PERSISTENCE_THRESHOLD="${ABSENCE_PERSISTENCE_THRESHOLD:-0.50}" \
ABSENCE_MIN_CONTEXT_SIMILARITY="${ABSENCE_MIN_CONTEXT_SIMILARITY:-0.20}" \
ABSENCE_MIN_QUERY_EXPOSURE="${ABSENCE_MIN_QUERY_EXPOSURE:-50}" \
RQ4_EVENT_DETAIL_MAX=400000 \
bash scripts/rq4/run_rq4_cloud_pilot.sh

"$python_bin" scripts/rq4/build_rq4_binary_escalation_table.py \
  --predictions_csv "${test_prefix}_run_level_predictions.csv" \
  --output_prefix "${test_prefix}_binary" \
  --method "$method_label"

echo "============================================================"
echo "Formal test run-level summary: ${test_prefix}_run_level_summary.json"
echo "Binary RQ4 table:             ${test_prefix}_binary_table.csv"
echo "Paper-readable Markdown:      ${test_prefix}_binary_table.md"
echo "Frozen dev thresholds:        $calibration_dir/best_threshold.json"
if [[ "$absence_context_mode" == "context_memory" ]]; then
  echo "Normal memory integrity:      $result_root/context_absence_memory_integrity.json"
else
  echo "Absence reference:             frozen train split ($absence_context_mode)"
fi
echo "============================================================"
