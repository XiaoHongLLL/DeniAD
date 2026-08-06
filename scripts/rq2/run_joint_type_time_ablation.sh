#!/usr/bin/env bash
set -euo pipefail

# RQ2: Does joint event-type and inter-event-time modeling improve anomaly detection?
#
# This runner trains two models on the same controlled data:
#   independent: p(m|H) + p(tau|H), using -disable_mark_conditioned_flow
#   full:        p(m|H) + p_CFM(tau|H,m)
# and evaluates five classifier variants:
#   transformer_only: direct Transformer hidden state, without probability-model features
#   type_only, time_only, independent_joint, ours_joint

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

python_bin=${PYTHON:-python}
device=${DEVICE:-0}
seed=${SEED:-2023}
data_root=${DATA_ROOT:-./data}
datasets=${DATASETS:-"thunderbird spirit liberty"}

run_prepare=${RUN_PREPARE:-1}
run_train=${RUN_TRAIN:-1}
run_eval=${RUN_EVAL:-1}
force_prepare=${FORCE_PREPARE:-0}
rq2_eval_mode=${RQ2_EVAL_MODE:-classifier}
skip_existing_ckpt=${SKIP_EXISTING_CKPT:-0}
eval_variants=${EVAL_VARIANTS:-"transformer_only type_only time_only independent_joint ours_joint"}

num_sequences_per_class=${NUM_SEQUENCES_PER_CLASS:-1000}
head_train_sequences_per_class=${HEAD_TRAIN_SEQUENCES_PER_CLASS:-600}
head_dev_sequences_per_class=${HEAD_DEV_SEQUENCES_PER_CLASS:-200}
controlled_window_size=${CONTROLLED_WINDOW_SIZE:-12}
max_train_sequences=${MAX_TRAIN_SEQUENCES:-0}
max_dev_sequences=${MAX_DEV_SEQUENCES:-0}

normalize=${NORMALIZE:-log}
solver_method=${SOLVER_METHOD:-euler}
time_norm_guard_threshold=${TIME_NORM_GUARD_THRESHOLD:-20.0}
eval_time_scale=${EVAL_TIME_SCALE:-legacy}
eval_epoch=${EVAL_EPOCH:-1000000}
train_n_samples=${TRAIN_N_SAMPLES:-10}
n_samples=${N_SAMPLES:-100}
anomaly_quantile=${ANOMALY_QUANTILE:-0.99}
uncertainty_quantile=${UNCERTAINTY_QUANTILE:-0.95}
uncertainty_mc=${UNCERTAINTY_MC:-8}
calibration_max_size=${CALIBRATION_MAX_SIZE:-200000}
segment_score_mode=${SEGMENT_SCORE_MODE:-max}
segment_topk=${SEGMENT_TOPK:-3}
exact_time_nll=${EXACT_TIME_NLL:-1}
joint_score_mode=${JOINT_SCORE_MODE:-zscore_max}
independent_score_mode=${INDEPENDENT_SCORE_MODE:-$joint_score_mode}
ours_score_mode=${OURS_SCORE_MODE:-$joint_score_mode}
joint_mode=${JOINT_MODE:-conditional}
type_only_disable_time_gap=${TYPE_ONLY_DISABLE_TIME_GAP:-1}
independent_type_weight=${INDEPENDENT_TYPE_WEIGHT:-1.0}
independent_time_weight=${INDEPENDENT_TIME_WEIGHT:-1.0}
ours_type_weight=${OURS_TYPE_WEIGHT:-1.0}
ours_time_weight=${OURS_TIME_WEIGHT:-1.0}
conditional_gap_weight=${CONDITIONAL_GAP_WEIGHT:-1.0}
rq2_classifier_epochs=${RQ2_CLASSIFIER_EPOCHS:-30}
rq2_classifier_lr=${RQ2_CLASSIFIER_LR:-1e-3}
rq2_classifier_hidden=${RQ2_CLASSIFIER_HIDDEN:-128}
rq2_classifier_dropout=${RQ2_CLASSIFIER_DROPOUT:-0.1}
rq2_classifier_batch_size=${RQ2_CLASSIFIER_BATCH_SIZE:-8192}
rq2_classifier_binary_loss_weight=${RQ2_CLASSIFIER_BINARY_LOSS_WEIGHT:-0.0}
rq2_classifier_class_balance=${RQ2_CLASSIFIER_CLASS_BALANCE:-binary}
rq2_classifier_dev_fpr_target=${RQ2_CLASSIFIER_DEV_FPR_TARGET:-0.05}
rq2_classifier_threshold_strategy=${RQ2_CLASSIFIER_THRESHOLD_STRATEGY:-f1}
rq2_classifier_dev_max_fpr=${RQ2_CLASSIFIER_DEV_MAX_FPR:-0.10}
rq2_classifier_segment_score_mode=${RQ2_CLASSIFIER_SEGMENT_SCORE_MODE:-topk_mean}
rq2_classifier_segment_topk=${RQ2_CLASSIFIER_SEGMENT_TOPK:-3}
rq2_classifier_uncertainty_mc=${RQ2_CLASSIFIER_UNCERTAINTY_MC:-1}
rq2_classifier_use_profile_features=${RQ2_CLASSIFIER_USE_PROFILE_FEATURES:-0}

ckpt_root=${CKPT_ROOT:-./checkpoints/rq2_joint_ablation}
result_root=${RESULT_ROOT:-./results/rq2_joint_ablation}
run_id=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$ckpt_root" "$result_root"

check_python_deps() {
  "$python_bin" -c "import importlib.util, sys; missing=[m for m in ['torch','torchdiffeq'] if importlib.util.find_spec(m) is None]; print('Missing Python modules: ' + ', '.join(missing), file=sys.stderr) if missing else None; sys.exit(1 if missing else 0)" || {
    echo "ERROR: RQ2 joint ablation needs torch and torchdiffeq in the active Python environment." >&2
    echo "Try: conda activate tpp_flow" >&2
    echo "Or install in the current env: python -m pip install torchdiffeq" >&2
    exit 1
  }
}

check_rq2_protocol_sync() {
  local missing=0
  if ! grep -Fq "choices=['f1', 'balanced_accuracy', 'target_fpr']" main.py; then
    echo "ERROR: main.py does not contain the RQ2 development-F1 threshold protocol." >&2
    missing=1
  fi
  if ! grep -Fq "shared_context_plus_probabilistic_residual" main.py; then
    echo "ERROR: main.py does not contain the RQ2 probabilistic residual head." >&2
    missing=1
  fi
  if ! grep -Fq "Classifier_Probabilistic_Residual_Weight" \
      scripts/rq2/summarize_rq2_joint_ablation_results.py; then
    echo "ERROR: the RQ2 summarizer does not contain residual-head fields." >&2
    missing=1
  fi
  if [ "$missing" = "1" ]; then
    echo "Synchronize main.py and scripts/rq2 from the same code revision before running RQ2." >&2
    exit 1
  fi
}

check_python_deps
check_rq2_protocol_sync

source_dataset_dir() {
  case "$1" in
    hdfs) echo "$data_root/labeled_hdfs" ;;
    bgl) echo "$data_root/labeled_bgl" ;;
    openstack) echo "$data_root/labeled_openstack" ;;
    thunderbird) echo "$data_root/labeled_thunderbird" ;;
    spirit) echo "$data_root/labeled_spirit" ;;
    liberty) echo "$data_root/labeled_liberty" ;;
    *) echo "ERROR: unsupported RQ2 dataset '$1'" >&2; return 1 ;;
  esac
}

controlled_dataset_dir() {
  echo "$data_root/rq2_controlled_$1"
}

has_source_split() {
  local dir="$1"
  [ -f "$dir/train.pkl" ] && [ -f "$dir/dev.pkl" ] && [ -f "$dir/test.pkl" ]
}

has_controlled_split() {
  local dir="$1"
  has_source_split "$dir" || return 1
  if [ "$rq2_eval_mode" = "classifier" ]; then
    [ -f "$dir/rq2_head_train.pkl" ] && [ -f "$dir/rq2_head_dev.pkl" ]
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
      epoch=${THUNDERBIRD_EPOCH:-${EPOCH:-80}}
      lr=${THUNDERBIRD_LR:-${LR:-2e-4}}
      train_batch_size=${THUNDERBIRD_BATCH_SIZE:-${BATCH_SIZE:-128}}
      eval_batch_size=${THUNDERBIRD_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-128}}
      loss_weighting=${THUNDERBIRD_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-fixed}}
      fm_loss_weight=${THUNDERBIRD_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-0.05}}
      loss_lambda=${THUNDERBIRD_LOSS_LAMBDA:-${LOSS_LAMBDA:-5.0}}
      checkpoint_metric=${THUNDERBIRD_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_acc}}
      fm_sigma=${THUNDERBIRD_FM_SIGMA:-${FM_SIGMA:-0.5}}
      train_step_size=${THUNDERBIRD_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${THUNDERBIRD_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.02}}
      clamp_threshold=${THUNDERBIRD_CLAMP_THRESHOLD:-${CLAMP_THRESHOLD:-6.0}}
      flow_cond_clip=${THUNDERBIRD_FLOW_COND_CLIP:-${FLOW_COND_CLIP:-5.0}}
      ;;
    spirit|liberty|bgl)
      prefix=$(echo "$dataset" | tr '[:lower:]' '[:upper:]')
      d_model=${D_MODEL:-128}
      d_inner_hid=${D_INNER_HID:-256}
      n_head=${N_HEAD:-4}
      n_layers=${N_LAYERS:-4}
      d_k=${D_K:-32}
      d_v=${D_V:-32}
      dropout=${DROPOUT:-0.1}
      type_head=${TYPE_HEAD:-gmm}
      epoch=${EPOCH:-60}
      lr=${LR:-5e-5}
      train_batch_size=${BATCH_SIZE:-128}
      eval_batch_size=${EVAL_BATCH_SIZE:-64}
      if [ "$dataset" = "liberty" ]; then
        train_batch_size=${LIBERTY_BATCH_SIZE:-${BATCH_SIZE:-64}}
      fi
      loss_weighting=${LOSS_WEIGHTING:-adaptive}
      fm_loss_weight=${FM_LOSS_WEIGHT:-1.0}
      loss_lambda=${LOSS_LAMBDA:-1.0}
      checkpoint_metric=${CHECKPOINT_METRIC:-valid_acc}
      fm_sigma=${FM_SIGMA:-0.01}
      train_step_size=${TRAIN_STEP_SIZE:-0.05}
      eval_step_size=${EVAL_STEP_SIZE:-0.01}
      clamp_threshold=${CLAMP_THRESHOLD:-6.0}
      flow_cond_clip=${FLOW_COND_CLIP:-5.0}
      ;;
    hdfs)
      d_model=${D_MODEL:-128}
      d_inner_hid=${D_INNER_HID:-256}
      n_head=${N_HEAD:-4}
      n_layers=${N_LAYERS:-4}
      d_k=${D_K:-16}
      d_v=${D_V:-16}
      dropout=${DROPOUT:-0.1}
      type_head=${TYPE_HEAD:-hybrid}
      epoch=${EPOCH:-60}
      lr=${LR:-1e-4}
      train_batch_size=${BATCH_SIZE:-128}
      eval_batch_size=${EVAL_BATCH_SIZE:-64}
      loss_weighting=${LOSS_WEIGHTING:-fixed}
      fm_loss_weight=${FM_LOSS_WEIGHT:-0.05}
      loss_lambda=${LOSS_LAMBDA:-5.0}
      checkpoint_metric=${CHECKPOINT_METRIC:-valid_acc}
      fm_sigma=${FM_SIGMA:-1.0}
      train_step_size=${TRAIN_STEP_SIZE:-0.05}
      eval_step_size=${EVAL_STEP_SIZE:-0.01}
      clamp_threshold=${CLAMP_THRESHOLD:-2.5}
      flow_cond_clip=${FLOW_COND_CLIP:-5.0}
      ;;
    openstack)
      d_model=${D_MODEL:-128}
      d_inner_hid=${D_INNER_HID:-256}
      n_head=${N_HEAD:-4}
      n_layers=${N_LAYERS:-4}
      d_k=${D_K:-16}
      d_v=${D_V:-16}
      dropout=${DROPOUT:-0.1}
      type_head=${TYPE_HEAD:-gmm}
      epoch=${EPOCH:-60}
      lr=${LR:-1e-4}
      train_batch_size=${BATCH_SIZE:-32}
      eval_batch_size=${EVAL_BATCH_SIZE:-64}
      loss_weighting=${LOSS_WEIGHTING:-adaptive}
      fm_loss_weight=${FM_LOSS_WEIGHT:-1.0}
      loss_lambda=${LOSS_LAMBDA:-0.1}
      checkpoint_metric=${CHECKPOINT_METRIC:-valid_loss}
      fm_sigma=${FM_SIGMA:-1.0}
      train_step_size=${TRAIN_STEP_SIZE:-0.05}
      eval_step_size=${EVAL_STEP_SIZE:-0.01}
      clamp_threshold=${CLAMP_THRESHOLD:-2.5}
      flow_cond_clip=${FLOW_COND_CLIP:-5.0}
      ;;
    *)
      echo "ERROR: no defaults for dataset '$dataset'" >&2
      return 1
      ;;
  esac
  checkpoint_min_delta=${CHECKPOINT_MIN_DELTA:-0.0001}
}

common_main_args() {
  local data_dir="$1"
  local batch_size="$2"
  local sample_count="$3"
  local step_size="$4"
  shift 4
  CUDA_VISIBLE_DEVICES="$device" PYTHONUNBUFFERED=1 "$python_bin" main.py \
    -data "$data_dir/" \
    -normalize "$normalize" \
    -d_model "$d_model" \
    -d_inner_hid "$d_inner_hid" \
    -n_head "$n_head" \
    -n_layers "$n_layers" \
    -d_k "$d_k" \
    -d_v "$d_v" \
    -dropout "$dropout" \
    -type_head "$type_head" \
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

maybe_prepare_controlled() {
  local dataset="$1"
  local source_dir="$2"
  local controlled_dir="$3"
  if [ "$force_prepare" != "1" ] && has_controlled_split "$controlled_dir"; then
    echo "[Skip] Controlled RQ2 data exists: $controlled_dir"
    return
  fi
  if [ "$run_prepare" != "1" ]; then
    echo "ERROR: controlled data missing at $controlled_dir. Set RUN_PREPARE=1." >&2
    exit 1
  fi
  "$python_bin" scripts/rq2/prepare_controlled_joint_dataset.py \
    --source_data "$source_dir" \
    --out_dir "$controlled_dir" \
    --num_sequences_per_class "$num_sequences_per_class" \
    --head_train_sequences_per_class "$head_train_sequences_per_class" \
    --head_dev_sequences_per_class "$head_dev_sequences_per_class" \
    --window_size "$controlled_window_size" \
    --joint_mode "$joint_mode" \
    --seed "$seed" \
    --max_train_sequences "$max_train_sequences" \
    --max_dev_sequences "$max_dev_sequences"
}

validate_controlled_dataset() {
  local controlled_dir="$1"
  local mode="$2"
  "$python_bin" -c '
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
mode = sys.argv[2]
meta_path = root / "metadata.json"
if not meta_path.exists():
    raise SystemExit(f"ERROR: controlled metadata missing: {meta_path}")
meta = json.loads(meta_path.read_text(encoding="utf-8"))
scenarios = set((meta.get("stats") or {}).get("scenarios") or {})
if mode == "conditional" and not any(name.startswith("joint_same_type_") for name in scenarios):
    raise SystemExit(
        "ERROR: conditional RQ2 data is stale or was generated by the old "
        f"joint construction. scenarios={sorted(scenarios)}. "
        "Run with FORCE_PREPARE=1 after updating prepare_controlled_joint_dataset.py."
    )
print(f"[OK] Controlled metadata validated: {root} scenarios={sorted(scenarios)}")
' "$controlled_dir" "$mode"
}

train_variant() {
  local dataset="$1"
  local variant="$2"
  local data_dir="$3"
  local save_path="$4"
  local log_path="$5"
  shift 5
  local extra_args=("$@")

  if [ "$run_train" != "1" ]; then
    echo "[Info] RUN_TRAIN=0; skip training $dataset/$variant."
    return
  fi
  if [ "$skip_existing_ckpt" = "1" ] || [ "$skip_existing_ckpt" = "true" ]; then
    if [ -f "$save_path" ]; then
      echo "[Skip] Existing checkpoint for $dataset/$variant: $save_path"
      return
    fi
  fi

  echo "[Train] $dataset / $variant -> $save_path"
  common_main_args "$data_dir" "$train_batch_size" "$train_n_samples" "$train_step_size" \
    -epoch "$epoch" \
    -eval_epoch "$eval_epoch" \
    -save_path "$save_path" \
    "${extra_args[@]}" \
    2>&1 | tee "$log_path"
}

eval_variant() {
  local dataset="$1"
  local variant="$2"
  local data_dir="$3"
  local save_path="$4"
  local score_mode="$5"
  local type_weight="$6"
  local time_weight="$7"
  local result_prefix="$8"
  shift 8
  local extra_args=("$@")

  if [ "$run_eval" != "1" ]; then
    echo "[Info] RUN_EVAL=0; skip eval $dataset/$variant."
    return
  fi
  if [[ " $eval_variants " != *" $variant "* ]]; then
    echo "[Info] EVAL_VARIANTS excludes $dataset/$variant; skip evaluation."
    return
  fi
  if [ ! -f "$save_path" ]; then
    echo "ERROR: checkpoint not found: $save_path" >&2
    echo "Set RUN_TRAIN=1 to train the RQ2 backbone checkpoints, or set CKPT_ROOT to the directory that already contains them." >&2
    exit 1
  fi

  eval_extra=()
  if [ "$exact_time_nll" = "1" ]; then
    eval_extra+=("-reliability_exact_nll")
  fi
  if [ "$rq2_classifier_use_profile_features" = "1" ] || [ "$rq2_classifier_use_profile_features" = "true" ]; then
    eval_extra+=("-rq2_classifier_use_profile_features")
  fi

  echo "[Eval] $dataset / $variant"
  if [ "$rq2_eval_mode" = "classifier" ]; then
    common_main_args "$data_dir" "$eval_batch_size" "$n_samples" "$eval_step_size" \
      -just_eval \
      -eval_rq2_classifier \
      -load_path_name "$save_path" \
      -rq2_classifier_variant "$variant" \
      -rq2_classifier_epochs "$rq2_classifier_epochs" \
      -rq2_classifier_lr "$rq2_classifier_lr" \
      -rq2_classifier_hidden "$rq2_classifier_hidden" \
      -rq2_classifier_dropout "$rq2_classifier_dropout" \
      -rq2_classifier_batch_size "$rq2_classifier_batch_size" \
      -rq2_classifier_binary_loss_weight "$rq2_classifier_binary_loss_weight" \
      -rq2_classifier_class_balance "$rq2_classifier_class_balance" \
      -rq2_classifier_dev_fpr_target "$rq2_classifier_dev_fpr_target" \
      -rq2_classifier_threshold_strategy "$rq2_classifier_threshold_strategy" \
      -rq2_classifier_dev_max_fpr "$rq2_classifier_dev_max_fpr" \
      -rq2_classifier_segment_score_mode "$rq2_classifier_segment_score_mode" \
      -rq2_classifier_segment_topk "$rq2_classifier_segment_topk" \
      -rq2_classifier_uncertainty_mc "$rq2_classifier_uncertainty_mc" \
      -save_result "$result_prefix" \
      "${eval_extra[@]}" \
      "${extra_args[@]}" \
      2>&1 | tee "${result_prefix}.log"
  else
    common_main_args "$data_dir" "$eval_batch_size" "$n_samples" "$eval_step_size" \
      -just_eval \
      -eval_anomaly_detection \
      -eval_rq2_subsets \
      -load_path_name "$save_path" \
      -anomaly_score_mode "$score_mode" \
      -type_score_weight "$type_weight" \
      -time_score_weight "$time_weight" \
      -conditional_gap_weight "$conditional_gap_weight" \
      -profile_score_weight 0.0 \
      -segment_score_mode "$segment_score_mode" \
      -segment_topk "$segment_topk" \
      -anomaly_quantile "$anomaly_quantile" \
      -uncertainty_quantile "$uncertainty_quantile" \
      -uncertainty_mc "$uncertainty_mc" \
      -calibration_max_size "$calibration_max_size" \
      -save_result "$result_prefix" \
      "${eval_extra[@]}" \
      "${extra_args[@]}" \
      2>&1 | tee "${result_prefix}.log"
  fi
}

for dataset in $datasets; do
  set_dataset_hparams "$dataset"
  source_dir=$(source_dataset_dir "$dataset")
  controlled_dir=$(controlled_dataset_dir "$dataset")
  if ! has_source_split "$source_dir"; then
    echo "ERROR: source split missing under $source_dir" >&2
    exit 1
  fi
  maybe_prepare_controlled "$dataset" "$source_dir" "$controlled_dir"
  validate_controlled_dataset "$controlled_dir" "$joint_mode"

  ds_ckpt_dir="$ckpt_root/$dataset"
  ds_result_dir="$result_root/$dataset"
  mkdir -p "$ds_ckpt_dir" "$ds_result_dir"

  type_only_ckpt="$ds_ckpt_dir/${dataset}_type_only_no_time_gap.pth"
  independent_ckpt="$ds_ckpt_dir/${dataset}_independent_no_mark_flow.pth"
  full_ckpt="$ds_ckpt_dir/${dataset}_full_mark_conditioned_flow.pth"

  type_only_flags=("-disable_mark_conditioned_flow")
  if [ "$type_only_disable_time_gap" = "1" ] || [ "$type_only_disable_time_gap" = "true" ]; then
    type_only_flags+=("-disable_time_gap")
  fi

  echo "================================================================"
  echo "[RQ2] $dataset"
  echo "Controlled data: $controlled_dir"
  echo "Model: head=$type_head layers=$n_layers d_k=$d_k d_v=$d_v dropout=$dropout"
  echo "Loss: weighting=$loss_weighting fmw=$fm_loss_weight lambda=$loss_lambda"
  echo "Flow: sigma=$fm_sigma train_step=$train_step_size eval_step=$eval_step_size exact_nll=$exact_time_nll"
  echo "RQ2: eval_mode=$rq2_eval_mode joint_mode=$joint_mode independent_score_mode=$independent_score_mode ours_score_mode=$ours_score_mode type_only_disable_time_gap=$type_only_disable_time_gap"
  echo "RQ2 classifier: epochs=$rq2_classifier_epochs lr=$rq2_classifier_lr hidden=$rq2_classifier_hidden class_balance=$rq2_classifier_class_balance binary_loss=$rq2_classifier_binary_loss_weight"
  echo "RQ2 calibration: strategy=$rq2_classifier_threshold_strategy dev_fpr=$rq2_classifier_dev_fpr_target dev_max_fpr=$rq2_classifier_dev_max_fpr segment=$rq2_classifier_segment_score_mode topk=$rq2_classifier_segment_topk"
  echo "RQ2 head data: train/class=$head_train_sequences_per_class dev/class=$head_dev_sequences_per_class"
  echo "RQ2 eval variants: $eval_variants"
  echo "================================================================"

  train_variant "$dataset" "type_only" "$controlled_dir" "$type_only_ckpt" \
    "$ds_result_dir/${dataset}_type_only_train.log" \
    "${type_only_flags[@]}"
  train_variant "$dataset" "independent" "$controlled_dir" "$independent_ckpt" \
    "$ds_result_dir/${dataset}_independent_train.log" \
    -disable_mark_conditioned_flow
  train_variant "$dataset" "full" "$controlled_dir" "$full_ckpt" \
    "$ds_result_dir/${dataset}_full_train.log"

  eval_variant "$dataset" "transformer_only" "$controlled_dir" "$full_ckpt" \
    raw 0.0 0.0 "$ds_result_dir/${dataset}_transformer_only"
  eval_variant "$dataset" "type_only" "$controlled_dir" "$type_only_ckpt" \
    type_only 1.0 0.0 "$ds_result_dir/${dataset}_type_only" \
    "${type_only_flags[@]}"
  eval_variant "$dataset" "time_only" "$controlled_dir" "$independent_ckpt" \
    time_only 0.0 1.0 "$ds_result_dir/${dataset}_time_only" \
    -disable_mark_conditioned_flow
  eval_variant "$dataset" "independent_joint" "$controlled_dir" "$independent_ckpt" \
    "$independent_score_mode" "$independent_type_weight" "$independent_time_weight" "$ds_result_dir/${dataset}_independent_joint" \
    -disable_mark_conditioned_flow
  eval_variant "$dataset" "ours_joint" "$controlled_dir" "$full_ckpt" \
    "$ours_score_mode" "$ours_type_weight" "$ours_time_weight" "$ds_result_dir/${dataset}_ours_joint"
done

if [ "$run_eval" = "1" ]; then
  summary_path="$result_root/summary.csv"
  summary_snapshot="$result_root/summary_${run_id}.csv"
  "$python_bin" scripts/rq2/summarize_rq2_joint_ablation_results.py \
    --result_root "$result_root" \
    --output "$summary_path"
  if [ -f "$summary_path" ]; then
    cp "$summary_path" "$summary_snapshot"
    echo "[OK] Snapshot summary: $summary_snapshot"
  fi
fi
