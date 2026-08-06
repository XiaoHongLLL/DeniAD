#!/usr/bin/env bash
set -euo pipefail

# RQ4/RQ3 real-change pilot on the Train-Ticket Expected/Unexpected dataset.
#
# Upload the cloud pilot dataset directory to the server, then run this script
# from the project directory. Use DATA_DIR to switch between the log-only and
# state-aware variants.
#
# Diagnostic RQ4:
#   RUN_TRAIN=1 RQ4_CANDIDATE_MODE=labeled_or_score bash scripts/rq4/run_rq4_cloud_pilot.sh
#
# End-to-end RQ3/RQ4:
#   RUN_TRAIN=0 RQ4_CANDIDATE_MODE=score bash scripts/rq4/run_rq4_cloud_pilot.sh

[[ -f main.py && -d scripts/rq4 ]] || {
  echo "ERROR: run this script from the DeniAD repository root." >&2
  exit 1
}

device=${DEVICE:-0}
seed=${SEED:-2023}
data_dir=${DATA_DIR:-./data_cloud_expected_unexpected_v0_2}
ckpt_dir=${CKPT_DIR:-./checkpoints/rq4/cloud_pilot}
result_dir=${RESULT_DIR:-./results/rq4/cloud_pilot}
run_id=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
run_train=${RUN_TRAIN:-1}
eval_epoch=${EVAL_EPOCH:-1000000}

d_model=${D_MODEL:-128}
d_inner_hid=${D_INNER_HID:-256}
n_head=${N_HEAD:-4}
n_layers=${N_LAYERS:-4}
d_k=${D_K:-32}
d_v=${D_V:-32}
dropout=${DROPOUT:-0.1}
type_head=${TYPE_HEAD:-gmm}
epoch=${EPOCH:-60}
batch_size=${BATCH_SIZE:-64}
eval_batch_size=${EVAL_BATCH_SIZE:-64}
lr=${LR:-5e-5}
loss_weighting=${LOSS_WEIGHTING:-fixed}
fm_loss_weight=${FM_LOSS_WEIGHT:-0.05}
loss_lambda=${LOSS_LAMBDA:-5.0}
checkpoint_metric=${CHECKPOINT_METRIC:-valid_acc}
checkpoint_min_delta=${CHECKPOINT_MIN_DELTA:-0.0001}
fm_sigma=${FM_SIGMA:-0.5}
train_step_size=${TRAIN_STEP_SIZE:-0.05}
eval_step_size=${EVAL_STEP_SIZE:-0.01}
clamp_threshold=${CLAMP_THRESHOLD:-6.0}
flow_cond_clip=${FLOW_COND_CLIP:-5.0}
min_max_len=${MIN_MAX_LEN:-0}
normalize=${NORMALIZE:-log}
context_mask_prob=${CONTEXT_MASK_PROB:-0.1}
context_mask_min_history=${CONTEXT_MASK_MIN_HISTORY:-1}
use_ensemble_correction=${USE_ENSEMBLE_CORRECTION:-1}
use_drift_adapter=${USE_DRIFT_ADAPTER:-1}
use_component_drift_diagnosis=${USE_COMPONENT_DRIFT_DIAGNOSIS:-1}
use_rq4_window_diagnosis=${USE_RQ4_WINDOW_DIAGNOSIS:-1}
use_trace_profile=${USE_TRACE_PROFILE:-1}

if [ "$use_drift_adapter" = "1" ] && [ "$use_ensemble_correction" != "1" ]; then
  echo "ERROR: USE_DRIFT_ADAPTER=1 requires USE_ENSEMBLE_CORRECTION=1." >&2
  exit 1
fi
if [ "${USE_COUNTERFACTUAL_CONTEXT_SUPPORT:-0}" = "1" ] && [ "$use_ensemble_correction" != "1" ]; then
  echo "ERROR: counterfactual support requires USE_ENSEMBLE_CORRECTION=1." >&2
  exit 1
fi
if [ "${USE_ABSENCE_AWARE_REVISION:-0}" = "1" ] && [ "$use_ensemble_correction" != "1" ]; then
  echo "ERROR: absence-aware revision requires USE_ENSEMBLE_CORRECTION=1." >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=$device
export PYTHONUNBUFFERED=1

mkdir -p "$ckpt_dir" "$result_dir"

if [ ! -f "$data_dir/train.pkl" ] || [ ! -f "$data_dir/dev.pkl" ] || [ ! -f "$data_dir/test.pkl" ]; then
  echo "ERROR: missing train/dev/test.pkl under $data_dir" >&2
  echo "Upload the cloud pilot dataset directory, or set DATA_DIR to its location." >&2
  exit 1
fi

checkpoint=${CHECKPOINT:-"$ckpt_dir/cloud_pilot_${type_head}_${checkpoint_metric}_ep${epoch}_dm${d_model}_bs${batch_size}_sigma${fm_sigma}_fmw${fm_loss_weight}_lambda${loss_lambda}.pth"}
result_prefix=${RESULT_PREFIX:-"$result_dir/rq4_cloud_pilot_${type_head}_${checkpoint_metric}_ep${epoch}_${run_id}_${RQ4_CANDIDATE_MODE:-labeled_or_score}"}

common_args=(
  -data "$data_dir/"
  -normalize "$normalize"
  -d_model "$d_model"
  -d_inner_hid "$d_inner_hid"
  -n_head "$n_head"
  -n_layers "$n_layers"
  -d_k "$d_k"
  -d_v "$d_v"
  -dropout "$dropout"
  -type_head "$type_head"
  -lr "$lr"
  -loss_weighting "$loss_weighting"
  -fm_loss_weight "$fm_loss_weight"
  -loss_lambda "$loss_lambda"
  -fm_sigma "$fm_sigma"
  -solver_method "${SOLVER_METHOD:-euler}"
  -clamp_threshold "$clamp_threshold"
  -flow_cond_clip "$flow_cond_clip"
  -min_max_len "$min_max_len"
  -n_samples "${N_SAMPLES:-100}"
  -checkpoint_metric "$checkpoint_metric"
  -checkpoint_min_delta "$checkpoint_min_delta"
  -context_mask_prob "$context_mask_prob"
  -context_mask_min_history "$context_mask_min_history"
  -seed "$seed"
)

counterfactual_args=()
if [ "${USE_COUNTERFACTUAL_CONTEXT_SUPPORT:-0}" = "1" ]; then
  counterfactual_args=(
    -use_counterfactual_context_support
    -counterfactual_support_k "${COUNTERFACTUAL_SUPPORT_K:-3}"
    -counterfactual_support_epsilon "${COUNTERFACTUAL_SUPPORT_EPSILON:-0.0}"
    -counterfactual_support_chunk_size "${COUNTERFACTUAL_SUPPORT_CHUNK_SIZE:-128}"
    -counterfactual_support_time_mode "${COUNTERFACTUAL_SUPPORT_TIME_MODE:-off}"
    -counterfactual_type_support_ratio "${COUNTERFACTUAL_TYPE_SUPPORT_RATIO:-0.34}"
    -counterfactual_time_support_ratio "${COUNTERFACTUAL_TIME_SUPPORT_RATIO:-0.34}"
    -counterfactual_type_support_strength "${COUNTERFACTUAL_TYPE_SUPPORT_STRENGTH:-0.0}"
    -counterfactual_time_support_strength "${COUNTERFACTUAL_TIME_SUPPORT_STRENGTH:-0.0}"
    -counterfactual_mem_gain_threshold "${COUNTERFACTUAL_MEM_GAIN_THRESHOLD:-0.0}"
  )
fi

ensemble_args=()
if [ "$use_ensemble_correction" = "1" ]; then
  ensemble_args=(-use_ensemble_correction)
fi

adapter_args=()
if [ "$use_drift_adapter" = "1" ]; then
  adapter_args=(-use_drift_adapter)
fi

component_diagnosis_args=()
if [ "$use_component_drift_diagnosis" = "1" ]; then
  component_diagnosis_args=(-use_component_drift_diagnosis)
fi

window_diagnosis_args=()
if [ "$use_rq4_window_diagnosis" = "1" ]; then
  window_diagnosis_args=(-use_rq4_window_diagnosis)
fi

trace_profile_args=()
if [ "$use_trace_profile" = "1" ]; then
  trace_profile_args=(-enable_trace_profile)
fi

absence_args=()
if [ "${USE_ABSENCE_AWARE_REVISION:-0}" = "1" ]; then
  absence_args=(
    -use_absence_aware_revision
    -absence_reference_split "${ABSENCE_REFERENCE_SPLIT:-train}"
    -absence_reference_path "${ABSENCE_REFERENCE_PATH:-}"
    -absence_context_mode "${ABSENCE_CONTEXT_MODE:-context_memory}"
    -absence_metadata_fields "${ABSENCE_METADATA_FIELDS:-}"
    -absence_exclude_services "${ABSENCE_EXCLUDE_SERVICES:-system-observability,tsdb-mysql,nacosdb-mysql}"
    -absence_service_prefixes "${ABSENCE_SERVICE_PREFIXES:-}"
    -absence_k "${ABSENCE_K:-20}"
    -absence_active_beta "${ABSENCE_ACTIVE_BETA:-0.70}"
    -absence_min_expected_count "${ABSENCE_MIN_EXPECTED_COUNT:-20}"
    -absence_count_ratio_threshold "${ABSENCE_COUNT_RATIO_THRESHOLD:-0.50}"
    -absence_anomaly_threshold "${ABSENCE_ANOMALY_THRESHOLD:-2.0}"
    -absence_persistence_threshold "${ABSENCE_PERSISTENCE_THRESHOLD:-0.50}"
    -absence_min_context_similarity "${ABSENCE_MIN_CONTEXT_SIMILARITY:-0.20}"
    -absence_min_query_exposure "${ABSENCE_MIN_QUERY_EXPOSURE:-50}"
    -absence_sigma_floor_ratio "${ABSENCE_SIGMA_FLOOR_RATIO:-0.25}"
    -absence_coverage_threshold "${ABSENCE_COVERAGE_THRESHOLD:-0.50}"
  )
fi

detail_args=()
if [ "${RQ4_EVENT_DETAIL_INCLUDE_NORMAL:-0}" = "1" ]; then
  detail_args=(-rq4_event_detail_include_normal)
fi

run_level_absence_args=()
if [ "${RUN_LEVEL_ABSENCE_VETO:-0}" = "1" ]; then
  run_level_absence_args=(
    --absence_veto reject
    --absence_apply_to "${ABSENCE_APPLY_TO:-normal_expected}"
    --absence_reference_path "${ABSENCE_REFERENCE_PATH:-}"
    --absence_context_mode "${ABSENCE_CONTEXT_MODE:-context_memory}"
    --absence_metadata_fields "${ABSENCE_METADATA_FIELDS:-}"
    --absence_exclude_services "${ABSENCE_EXCLUDE_SERVICES:-system-observability,tsdb-mysql,nacosdb-mysql}"
    --absence_service_prefixes "${ABSENCE_SERVICE_PREFIXES:-}"
    --absence_k "${ABSENCE_K:-20}"
    --absence_active_beta "${ABSENCE_ACTIVE_BETA:-0.70}"
    --absence_min_expected_count "${ABSENCE_MIN_EXPECTED_COUNT:-20}"
    --absence_count_ratio_threshold "${ABSENCE_COUNT_RATIO_THRESHOLD:-0.50}"
    --absence_anomaly_threshold "${ABSENCE_ANOMALY_THRESHOLD:-2.0}"
    --absence_persistence_threshold "${ABSENCE_PERSISTENCE_THRESHOLD:-0.50}"
    --absence_min_context_similarity "${ABSENCE_MIN_CONTEXT_SIMILARITY:-0.20}"
    --absence_min_query_exposure "${ABSENCE_MIN_QUERY_EXPOSURE:-50}"
    --absence_sigma_floor_ratio "${ABSENCE_SIGMA_FLOOR_RATIO:-0.25}"
    --absence_coverage_threshold "${ABSENCE_COVERAGE_THRESHOLD:-0.50}"
  )
fi

run_level_decision_args=(
  --event_pred_column "${RUN_LEVEL_EVENT_PRED_COLUMN:-pred_drift_id}"
  --expected_min "${RUN_LEVEL_EXPECTED_MIN:-0.50}"
  --unexpected_min "${RUN_LEVEL_UNEXPECTED_MIN:-0.25}"
  --reject_min "${RUN_LEVEL_REJECT_MIN:-0.20}"
  --conflict_min "${RUN_LEVEL_CONFLICT_MIN:-0.20}"
  --decision_policy "${RUN_LEVEL_DECISION_POLICY:-reject_first}"
  --reject_margin "${RUN_LEVEL_REJECT_MARGIN:-0.0}"
  --unexpected_margin "${RUN_LEVEL_UNEXPECTED_MARGIN:-0.0}"
  --strong_unexpected_min "${RUN_LEVEL_STRONG_UNEXPECTED_MIN:-0.50}"
  --expected_unexpected_max "${RUN_LEVEL_EXPECTED_UNEXPECTED_MAX:-0.05}"
  --absence_unexpected_mode "${RUN_LEVEL_ABSENCE_UNEXPECTED_MODE:-off}"
  --absence_strong_anomaly_threshold "${RUN_LEVEL_ABSENCE_STRONG_ANOMALY_THRESHOLD:-3.0}"
  --absence_strong_coverage_threshold "${RUN_LEVEL_ABSENCE_STRONG_COVERAGE_THRESHOLD:-1.0}"
)

echo "================================================================"
echo "[RQ4/RQ3 Dataset] $(basename "$data_dir")"
echo "Data:       $data_dir"
echo "Checkpoint: $checkpoint"
echo "Result:     $result_prefix"
echo "Candidate:  ${RQ4_CANDIDATE_MODE:-labeled_or_score}"
echo "Model:      head=$type_head layers=$n_layers d_model=$d_model batch=$batch_size/$eval_batch_size"
echo "Components: memory=$use_ensemble_correction adapter=$use_drift_adapter component=$use_component_drift_diagnosis window=$use_rq4_window_diagnosis profile=$use_trace_profile cf=${USE_COUNTERFACTUAL_CONTEXT_SUPPORT:-0} absence=${USE_ABSENCE_AWARE_REVISION:-0}"
echo "Run-level:  source=${RUN_LEVEL_EVENT_PRED_COLUMN:-pred_drift_id} policy=${RUN_LEVEL_DECISION_POLICY:-reject_first} state=${RUN_LEVEL_STATE_VETO:-auto}"
echo "Note:       main paper metrics should be run-level; use summarize_rq4_run_level.py after eval."
echo "================================================================"

if [ "$run_train" = "1" ] || [ ! -f "$checkpoint" ]; then
  if [ "$run_train" != "1" ]; then
    echo "ERROR: checkpoint not found at $checkpoint" >&2
    echo "Set RUN_TRAIN=1 to train, or CHECKPOINT=/path/to/model.pth to evaluate an existing model." >&2
    exit 1
  fi
  python main.py "${common_args[@]}" \
    -batch_size "$batch_size" \
    -epoch "$epoch" \
    -eval_epoch "$eval_epoch" \
    -solver_step_size "$train_step_size" \
    -save_path "$checkpoint" \
    2>&1 | tee "${result_prefix}_train.log"
else
  echo "[Info] Using existing checkpoint: $checkpoint"
fi

python main.py "${common_args[@]}" \
  -batch_size "$eval_batch_size" \
  -just_eval \
  -eval_reliability \
  "${ensemble_args[@]}" \
  "${adapter_args[@]}" \
  -load_path_name "$checkpoint" \
  -solver_step_size "$eval_step_size" \
  -rq4_candidate_mode "${RQ4_CANDIDATE_MODE:-labeled_or_score}" \
  "${component_diagnosis_args[@]}" \
  "${window_diagnosis_args[@]}" \
  "${trace_profile_args[@]}" \
  -anomaly_score_mode "${RQ4_ANOMALY_SCORE_MODE:-profile_zscore}" \
  -type_score_weight "${RQ4_TYPE_SCORE_WEIGHT:-1.0}" \
  -time_score_weight "${RQ4_TIME_SCORE_WEIGHT:-1.0}" \
  -profile_score_weight "${RQ4_PROFILE_SCORE_WEIGHT:-1.0}" \
  -profile_smoothing "${PROFILE_SMOOTHING:-0.01}" \
  -profile_bigram_weight "${PROFILE_BIGRAM_WEIGHT:-2.0}" \
  -profile_unigram_weight "${PROFILE_UNIGRAM_WEIGHT:-0.5}" \
  -drift_candidate_profile_quantile "${DRIFT_CANDIDATE_PROFILE_QUANTILE:-0.95}" \
  -drift_diagnosis_type_quantile "${DRIFT_DIAGNOSIS_TYPE_QUANTILE:-0.99}" \
  -drift_diagnosis_time_quantile "${DRIFT_DIAGNOSIS_TIME_QUANTILE:-0.995}" \
  -drift_diagnosis_profile_quantile "${DRIFT_DIAGNOSIS_PROFILE_QUANTILE:-0.99}" \
  -drift_diagnosis_strong_quantile "${DRIFT_DIAGNOSIS_STRONG_QUANTILE:-0.997}" \
  -drift_diagnosis_extreme_time_quantile "${DRIFT_DIAGNOSIS_EXTREME_TIME_QUANTILE:-0.9995}" \
  "${counterfactual_args[@]}" \
  "${absence_args[@]}" \
  -rq4_window_expected_min_frac "${RQ4_WINDOW_EXPECTED_MIN_FRAC:-0.50}" \
  -rq4_window_unexpected_min_frac "${RQ4_WINDOW_UNEXPECTED_MIN_FRAC:-0.25}" \
  -rq4_window_reject_min_frac "${RQ4_WINDOW_REJECT_MIN_FRAC:-0.20}" \
  -rq4_window_conflict_min_frac "${RQ4_WINDOW_CONFLICT_MIN_FRAC:-0.20}" \
  -uncertainty_mc "${UNCERTAINTY_MC:-8}" \
  -anomaly_quantile "${ANOMALY_QUANTILE:-0.95}" \
  -uncertainty_quantile "${UNCERTAINTY_QUANTILE:-0.99}" \
  -calibration_max_size "${CALIBRATION_MAX_SIZE:-200000}" \
  -ensemble_k "${ENSEMBLE_K:-20}" \
  -ensemble_samples "${ENSEMBLE_SAMPLES:-16}" \
  -ensemble_kernel "${ENSEMBLE_KERNEL:-0.2}" \
  -ensemble_noise_scale "${ENSEMBLE_NOISE_SCALE:-0.1}" \
  -ensemble_correction_weight "${ENSEMBLE_CORRECTION_WEIGHT:-0.7}" \
  -ensemble_disagreement_threshold "${ENSEMBLE_DISAGREEMENT_THRESHOLD:-2.0}" \
  -ensemble_support_quantile "${ENSEMBLE_SUPPORT_QUANTILE:-0.90}" \
  -ensemble_max_reference "${ENSEMBLE_MAX_REFERENCE:-4096}" \
  -ensemble_max_search "${ENSEMBLE_MAX_SEARCH:-50000}" \
  -drift_adapter_min_events "${DRIFT_ADAPTER_MIN_EVENTS:-8}" \
  -drift_adapter_fit_interval "${DRIFT_ADAPTER_FIT_INTERVAL:-8}" \
  -save_rq4_event_details \
  -rq4_event_detail_max "${RQ4_EVENT_DETAIL_MAX:-200000}" \
  "${detail_args[@]}" \
  -save_result "$result_prefix" \
  2>&1 | tee "${result_prefix}_reliability.log"

python scripts/rq4/summarize_rq4_results.py \
  --result_root "$result_dir" \
  --output "$result_dir/summary_${run_id}.csv"

python scripts/rq4/summarize_rq4_run_level.py \
  --data_dir "$data_dir" \
  --events_csv "${result_prefix}_rq4_events.csv" \
  --batch_size "$eval_batch_size" \
  --state_veto "${RUN_LEVEL_STATE_VETO:-auto}" \
  "${run_level_decision_args[@]}" \
  "${run_level_absence_args[@]}" \
  --output_prefix "${result_prefix}_run_level"
