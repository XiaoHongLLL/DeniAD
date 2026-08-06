#!/usr/bin/env bash
set -euo pipefail

# RQ2 generation benchmark runner for large labeled log datasets.
#
# This script recomputes the normal prediction metrics reported by main.py:
#   Acc, NLL, Time_NLL, Type_NLL, RMSE, CS
#
# Default usage:
#   DATASETS="thunderbird spirit liberty" bash scripts/rq2/run_rq2_labeled_log_benchmarks.sh
#
# If pkl files are missing and raw logs are available:
#   RAW_ROOT=/path/to/raw_logs AUTO_PREP=1 bash scripts/rq2/run_rq2_labeled_log_benchmarks.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

python_bin=${PYTHON:-python}
device=${DEVICE:-0}
seed=${SEED:-2023}
data_root=${DATA_ROOT:-./data}
raw_root=${RAW_ROOT:-./data/raw}
datasets=${DATASETS:-"thunderbird spirit liberty"}

run_train=${RUN_TRAIN:-1}
run_eval=${RUN_EVAL:-1}
auto_prep=${AUTO_PREP:-0}
force_prep=${FORCE_PREP:-0}

normalize=${NORMALIZE:-log}
solver_method=${SOLVER_METHOD:-euler}
time_norm_guard_threshold=${TIME_NORM_GUARD_THRESHOLD:-20.0}
eval_time_scale=${EVAL_TIME_SCALE:-legacy}
eval_epoch=${EVAL_EPOCH:-1000000}
train_n_samples=${TRAIN_N_SAMPLES:-10}
n_samples=${N_SAMPLES:-100}
fm_debug=${FM_DEBUG:-0}
fm_debug_threshold=${FM_DEBUG_THRESHOLD:-1000.0}

ckpt_root=${CKPT_ROOT:-./checkpoints/rq2_generation}
result_root=${RESULT_ROOT:-./results/rq2_generation}
run_id=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$ckpt_root" "$result_root"

dataset_dir() {
  case "$1" in
    thunderbird) echo "$data_root/labeled_thunderbird/" ;;
    spirit) echo "$data_root/labeled_spirit/" ;;
    liberty) echo "$data_root/labeled_liberty/" ;;
    *) echo "ERROR: unsupported RQ2 dataset '$1'" >&2; return 1 ;;
  esac
}

has_processed_split() {
  local dir="$1"
  [ -f "${dir}/train.pkl" ] && [ -f "${dir}/dev.pkl" ] && [ -f "${dir}/test.pkl" ]
}

maybe_prepare_dataset() {
  local dataset="$1"
  local data_dir="$2"
  if has_processed_split "$data_dir"; then
    return 0
  fi

  if [ "$auto_prep" != "1" ]; then
    echo "ERROR: missing train/dev/test pkl under $data_dir" >&2
    echo "Set AUTO_PREP=1 RAW_ROOT=/path/to/raw_logs to build the dataset automatically." >&2
    return 1
  fi

  echo "[Info] Missing pkl files for $dataset; running preprocessing first."
  RAW_ROOT="$raw_root" OUT_ROOT="$data_root" DATASETS="$dataset" FORCE_PREP="$force_prep" \
    bash scripts/rq2/prepare_rq2_log_datasets.sh

  if ! has_processed_split "$data_dir"; then
    echo "ERROR: preprocessing did not create train/dev/test pkl under $data_dir" >&2
    return 1
  fi
}

set_dataset_hparams() {
  local dataset="$1"
  case "$dataset" in
    thunderbird)
      d_model=${THUNDERBIRD_D_MODEL:-${D_MODEL:-128}}
      d_inner_hid=${THUNDERBIRD_D_INNER_HID:-${D_INNER_HID:-256}}
      n_head=${THUNDERBIRD_N_HEAD:-${N_HEAD:-4}}
      n_layers=${THUNDERBIRD_N_LAYERS:-${N_LAYERS:-4}}
      d_k=${THUNDERBIRD_D_K:-${D_K:-16}}
      d_v=${THUNDERBIRD_D_V:-${D_V:-16}}
      dropout=${THUNDERBIRD_DROPOUT:-${DROPOUT:-0.05}}
      type_head=${THUNDERBIRD_TYPE_HEAD:-${TYPE_HEAD:-hybrid_markov}}
      use_pos_enc=${THUNDERBIRD_USE_POS_ENC:-${USE_POS_ENC:-false}}
      disable_time_gap=${THUNDERBIRD_DISABLE_TIME_GAP:-${DISABLE_TIME_GAP:-false}}

      epoch=${THUNDERBIRD_EPOCH:-${EPOCH:-80}}
      lr=${THUNDERBIRD_LR:-${LR:-2e-4}}
      train_batch_size=${THUNDERBIRD_BATCH_SIZE:-${BATCH_SIZE:-128}}
      eval_batch_size=${THUNDERBIRD_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-128}}
      loss_weighting=${THUNDERBIRD_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-fixed}}
      fm_loss_weight=${THUNDERBIRD_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-0.05}}
      loss_lambda=${THUNDERBIRD_LOSS_LAMBDA:-${LOSS_LAMBDA:-5.0}}
      checkpoint_metric=${THUNDERBIRD_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_acc}}
      checkpoint_min_delta=${THUNDERBIRD_CHECKPOINT_MIN_DELTA:-${CHECKPOINT_MIN_DELTA:-0.0001}}

      fm_sigma=${THUNDERBIRD_FM_SIGMA:-${FM_SIGMA:-0.5}}
      train_step_size=${THUNDERBIRD_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${THUNDERBIRD_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.02}}
      clamp_threshold=${THUNDERBIRD_CLAMP_THRESHOLD:-${CLAMP_THRESHOLD:-6.0}}
      flow_cond_clip=${THUNDERBIRD_FLOW_COND_CLIP:-${FLOW_COND_CLIP:-5.0}}
      ;;
    spirit)
      d_model=${SPIRIT_D_MODEL:-${D_MODEL:-128}}
      d_inner_hid=${SPIRIT_D_INNER_HID:-${D_INNER_HID:-256}}
      n_head=${SPIRIT_N_HEAD:-${N_HEAD:-4}}
      n_layers=${SPIRIT_N_LAYERS:-${N_LAYERS:-4}}
      d_k=${SPIRIT_D_K:-${D_K:-32}}
      d_v=${SPIRIT_D_V:-${D_V:-32}}
      dropout=${SPIRIT_DROPOUT:-${DROPOUT:-0.1}}
      type_head=${SPIRIT_TYPE_HEAD:-${TYPE_HEAD:-gmm}}
      use_pos_enc=${SPIRIT_USE_POS_ENC:-${USE_POS_ENC:-false}}
      disable_time_gap=${SPIRIT_DISABLE_TIME_GAP:-${DISABLE_TIME_GAP:-false}}

      epoch=${SPIRIT_EPOCH:-${EPOCH:-60}}
      lr=${SPIRIT_LR:-${LR:-5e-5}}
      train_batch_size=${SPIRIT_BATCH_SIZE:-${BATCH_SIZE:-128}}
      eval_batch_size=${SPIRIT_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-64}}
      loss_weighting=${SPIRIT_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-adaptive}}
      fm_loss_weight=${SPIRIT_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-1.0}}
      loss_lambda=${SPIRIT_LOSS_LAMBDA:-${LOSS_LAMBDA:-1.0}}
      checkpoint_metric=${SPIRIT_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_acc}}
      checkpoint_min_delta=${SPIRIT_CHECKPOINT_MIN_DELTA:-${CHECKPOINT_MIN_DELTA:-0.0001}}

      fm_sigma=${SPIRIT_FM_SIGMA:-${FM_SIGMA:-0.01}}
      train_step_size=${SPIRIT_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${SPIRIT_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.01}}
      clamp_threshold=${SPIRIT_CLAMP_THRESHOLD:-${CLAMP_THRESHOLD:-6.0}}
      flow_cond_clip=${SPIRIT_FLOW_COND_CLIP:-${FLOW_COND_CLIP:-5.0}}
      ;;
    liberty)
      d_model=${LIBERTY_D_MODEL:-${D_MODEL:-128}}
      d_inner_hid=${LIBERTY_D_INNER_HID:-${D_INNER_HID:-256}}
      n_head=${LIBERTY_N_HEAD:-${N_HEAD:-4}}
      n_layers=${LIBERTY_N_LAYERS:-${N_LAYERS:-4}}
      d_k=${LIBERTY_D_K:-${D_K:-32}}
      d_v=${LIBERTY_D_V:-${D_V:-32}}
      dropout=${LIBERTY_DROPOUT:-${DROPOUT:-0.1}}
      type_head=${LIBERTY_TYPE_HEAD:-${TYPE_HEAD:-gmm}}
      use_pos_enc=${LIBERTY_USE_POS_ENC:-${USE_POS_ENC:-false}}
      disable_time_gap=${LIBERTY_DISABLE_TIME_GAP:-${DISABLE_TIME_GAP:-false}}

      epoch=${LIBERTY_EPOCH:-${EPOCH:-60}}
      lr=${LIBERTY_LR:-${LR:-5e-5}}
      train_batch_size=${LIBERTY_BATCH_SIZE:-${BATCH_SIZE:-64}}
      eval_batch_size=${LIBERTY_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-64}}
      loss_weighting=${LIBERTY_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-adaptive}}
      fm_loss_weight=${LIBERTY_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-1.0}}
      loss_lambda=${LIBERTY_LOSS_LAMBDA:-${LOSS_LAMBDA:-1.0}}
      checkpoint_metric=${LIBERTY_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_acc}}
      checkpoint_min_delta=${LIBERTY_CHECKPOINT_MIN_DELTA:-${CHECKPOINT_MIN_DELTA:-0.0001}}

      fm_sigma=${LIBERTY_FM_SIGMA:-${FM_SIGMA:-0.01}}
      train_step_size=${LIBERTY_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${LIBERTY_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.01}}
      clamp_threshold=${LIBERTY_CLAMP_THRESHOLD:-${CLAMP_THRESHOLD:-6.0}}
      flow_cond_clip=${LIBERTY_FLOW_COND_CLIP:-${FLOW_COND_CLIP:-5.0}}
      ;;
    *)
      echo "ERROR: no RQ2 hyperparameter defaults for dataset '$dataset'" >&2
      return 1
      ;;
  esac
}

build_model_flag_args() {
  model_flag_args=()
  if [ "$use_pos_enc" = "true" ]; then
    model_flag_args+=("-use_pos_enc")
  fi
  if [ "$disable_time_gap" = "true" ]; then
    model_flag_args+=("-disable_time_gap")
  fi
}

run_main_common() {
  local data_dir="$1"
  local batch_size="$2"
  local sample_count="$3"
  local step_size="$4"
  shift 4

  CUDA_VISIBLE_DEVICES="$device" PYTHONUNBUFFERED=1 "$python_bin" main.py \
    -data "$data_dir" \
    -normalize "$normalize" \
    -d_model "$d_model" \
    -d_inner_hid "$d_inner_hid" \
    -n_head "$n_head" \
    -n_layers "$n_layers" \
    -d_k "$d_k" \
    -d_v "$d_v" \
    -dropout "$dropout" \
    -type_head "$type_head" \
    "${model_flag_args[@]}" \
    -batch_size "$batch_size" \
    -lr "$lr" \
    -loss_weighting "$loss_weighting" \
    -fm_loss_weight "$fm_loss_weight" \
    -loss_lambda "$loss_lambda" \
    -flow_cond_clip "$flow_cond_clip" \
    -time_norm_guard_threshold "$time_norm_guard_threshold" \
    -fm_sigma "$fm_sigma" \
    -solver_method "$solver_method" \
    -solver_step_size "$step_size" \
    -clamp_threshold "$clamp_threshold" \
    -n_samples "$sample_count" \
    -checkpoint_metric "$checkpoint_metric" \
    -checkpoint_min_delta "$checkpoint_min_delta" \
    -eval_time_scale "$eval_time_scale" \
    -disable_dataset_adaptive_detection \
    -seed "$seed" \
    "$@"
}

for dataset in $datasets; do
  set_dataset_hparams "$dataset"
  data_dir=$(dataset_dir "$dataset")
  maybe_prepare_dataset "$dataset" "$data_dir"
  build_model_flag_args

  ds_ckpt_dir="$ckpt_root/$dataset"
  ds_result_dir="$result_root/$dataset"
  mkdir -p "$ds_ckpt_dir" "$ds_result_dir"

  save_name="${dataset}_rq2_${type_head}_${loss_weighting}_${checkpoint_metric}_ep${epoch}_dm${d_model}_bs${train_batch_size}_lr${lr}_sigma${fm_sigma}_fmw${fm_loss_weight}_lambda${loss_lambda}"
  save_path="${ds_ckpt_dir}/${save_name}.pth"
  train_log="${ds_result_dir}/${save_name}_train.log"
  eval_prefix="${ds_result_dir}/${save_name}_eval"
  eval_log="${eval_prefix}.log"

  echo "================================================================"
  echo "[RQ2 Dataset] $dataset"
  echo "Data:          $data_dir"
  echo "Checkpoint:    $save_path"
  echo "Result prefix: $eval_prefix"
  echo "Model:         head=$type_head layers=$n_layers d_k=$d_k d_v=$d_v dropout=$dropout"
  echo "Loss:          weighting=$loss_weighting fm_weight=$fm_loss_weight lambda=$loss_lambda metric=$checkpoint_metric"
  echo "Flow:          sigma=$fm_sigma train_step=$train_step_size eval_step=$eval_step_size clamp=$clamp_threshold"
  echo "Batch:         train=$train_batch_size eval=$eval_batch_size"
  echo "Eval scale:    $eval_time_scale"
  echo "Flags:         use_pos_enc=$use_pos_enc disable_time_gap=$disable_time_gap"
  echo "================================================================"

  if [ "$run_train" = "1" ]; then
    train_extra_args=()
    if [ "$fm_debug" = "1" ]; then
      train_extra_args+=("-fm_debug" "-fm_debug_threshold" "$fm_debug_threshold")
    fi

    run_main_common "$data_dir" "$train_batch_size" "$train_n_samples" "$train_step_size" \
      -epoch "$epoch" \
      -eval_epoch "$eval_epoch" \
      -save_path "$save_path" \
      "${train_extra_args[@]}" \
      2>&1 | tee "$train_log"
  else
    echo "[Info] RUN_TRAIN=0; skip training."
  fi

  if [ "$run_eval" = "1" ]; then
    if [ ! -f "$save_path" ]; then
      echo "ERROR: checkpoint not found at $save_path" >&2
      exit 1
    fi

    run_main_common "$data_dir" "$eval_batch_size" "$n_samples" "$eval_step_size" \
      -just_eval \
      -load_path_name "$save_path" \
      -save_result "$eval_prefix" \
      2>&1 | tee "$eval_log"

    echo "[OK] $dataset RQ2 CSV: ${eval_prefix}_results.csv"
  else
    echo "[Info] RUN_EVAL=0; skip final evaluation."
  fi
done

if [ "$run_eval" = "1" ]; then
  summary_path="$result_root/summary.csv"
  summary_snapshot="$result_root/summary_${run_id}.csv"
  "$python_bin" scripts/rq2/summarize_rq2_generation_results.py \
    --result_root "$result_root" \
    --output "$summary_path"
  if [ -f "$summary_path" ]; then
    cp "$summary_path" "$summary_snapshot"
    echo "[OK] Snapshot summary: $summary_snapshot"
  fi
fi

