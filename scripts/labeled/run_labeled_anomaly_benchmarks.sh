#!/usr/bin/env bash
set -euo pipefail

# Labeled anomaly-detection benchmark runner.
#
# Intended server usage:
#   cd /path/to/DeniAD
#   conda activate tpp_flow
#   DATASETS="hdfs bgl openstack thunderbird spirit liberty" bash scripts/labeled/run_labeled_anomaly_benchmarks.sh
#
# Quick smoke test:
#   DATASETS="openstack" EPOCH=1 RUN_RELIABILITY=0 bash scripts/labeled/run_labeled_anomaly_benchmarks.sh
#
# All outputs stay inside the current project directory:
#   checkpoints/labeled_anomaly/
#   results/labeled_anomaly/

[[ -f main.py && -d scripts/labeled ]] || {
  echo "ERROR: run this script from the DeniAD repository root." >&2
  exit 1
}

device=${DEVICE:-0}
seed=${SEED:-2023}
data_root=${DATA_ROOT:-./data}
datasets=${DATASETS:-"hdfs bgl openstack thunderbird"}
export CUDA_VISIBLE_DEVICES=$device
export PYTHONUNBUFFERED=1

normalize=${NORMALIZE:-log}
solver_method=${SOLVER_METHOD:-euler}
time_norm_guard_threshold=${TIME_NORM_GUARD_THRESHOLD:-20.0}

run_train=${RUN_TRAIN:-1}
run_anomaly=${RUN_ANOMALY:-1}
run_reliability=${RUN_RELIABILITY:-1}
eval_epoch=${EVAL_EPOCH:-1000000}
benchmark_efficiency=${BENCHMARK_EFFICIENCY:-0}
benchmark_warmup_batches=${BENCHMARK_WARMUP_BATCHES:-5}
benchmark_repeats=${BENCHMARK_REPEATS:-5}
save_unified_predictions=${SAVE_UNIFIED_PREDICTIONS:-0}

uncertainty_mc=${UNCERTAINTY_MC:-8}
anomaly_quantile=${ANOMALY_QUANTILE:-0.99}
uncertainty_quantile=${UNCERTAINTY_QUANTILE:-0.95}
calibration_max_size=${CALIBRATION_MAX_SIZE:-200000}
anomaly_score_mode=${ANOMALY_SCORE_MODE:-zscore}
type_score_weight=${TYPE_SCORE_WEIGHT:-1.0}
time_score_weight=${TIME_SCORE_WEIGHT:-0.5}
segment_score_mode=${SEGMENT_SCORE_MODE:-alert_fraction}
segment_topk=${SEGMENT_TOPK:-3}
dataset_adaptive_detection=${DATASET_ADAPTIVE_DETECTION:-1}
eval_segment_threshold_sweep=${EVAL_SEGMENT_THRESHOLD_SWEEP:-1}
threshold_sweep_steps=${THRESHOLD_SWEEP_STEPS:-200}
ensemble_k=${ENSEMBLE_K:-20}
ensemble_samples=${ENSEMBLE_SAMPLES:-16}
ensemble_kernel=${ENSEMBLE_KERNEL:-0.2}
ensemble_noise_scale=${ENSEMBLE_NOISE_SCALE:-0.1}
ensemble_correction_weight=${ENSEMBLE_CORRECTION_WEIGHT:-1.0}
ensemble_support_quantile=${ENSEMBLE_SUPPORT_QUANTILE:-0.95}
ensemble_max_reference=${ENSEMBLE_MAX_REFERENCE:-4096}
ensemble_max_search=${ENSEMBLE_MAX_SEARCH:-50000}

ckpt_root=${CKPT_ROOT:-./checkpoints/labeled_anomaly}
result_root=${RESULT_ROOT:-./results/labeled_anomaly}
run_id=${RUN_ID:-$(date +%Y%m%d_%H%M%S)}
mkdir -p "$ckpt_root" "$result_root"

dataset_dir() {
  case "$1" in
    hdfs) echo "$data_root/labeled_hdfs/" ;;
    bgl) echo "$data_root/labeled_bgl/" ;;
    hadoop) echo "$data_root/labeled_hadoop/" ;;
    openstack) echo "$data_root/labeled_openstack/" ;;
    thunderbird) echo "$data_root/labeled_thunderbird/" ;;
    spirit) echo "$data_root/labeled_spirit/" ;;
    liberty) echo "$data_root/labeled_liberty/" ;;
    adfa_java|adfa_hydra_ssh|adfa_hydra_ftp|adfa_meter|adfa_web|adfa_adduser)
      echo "$data_root/flexlog_real_ulad/adfa_u/${1#adfa_}/${FLEXLOG_PROTOCOL:-deniad_limited_normal}/"
      ;;
    logevol_hadoop)
      echo "$data_root/flexlog_real_ulad/logevol_u/hadoop2_to_3/${FLEXLOG_PROTOCOL:-deniad_limited_normal}/"
      ;;
    logevol_spark)
      echo "$data_root/flexlog_real_ulad/logevol_u/spark2_to_3/${FLEXLOG_PROTOCOL:-deniad_limited_normal}/"
      ;;
    *) echo "ERROR: unknown dataset $1" >&2; return 1 ;;
  esac
}

set_dataset_hparams() {
  local dataset="$1"

  # The script is a unified launcher, not a single shared hyperparameter
  # setting. Defaults below follow the strongest existing per-dataset scripts
  # and can still be overridden by global or dataset-specific env vars.
  case "$dataset" in
    hdfs)
      d_model=${HDFS_D_MODEL:-${D_MODEL:-128}}
      d_inner_hid=${HDFS_D_INNER_HID:-${D_INNER_HID:-256}}
      n_head=${HDFS_N_HEAD:-${N_HEAD:-4}}
      n_layers=${HDFS_N_LAYERS:-${N_LAYERS:-4}}
      d_k=${HDFS_D_K:-${D_K:-16}}
      d_v=${HDFS_D_V:-${D_V:-16}}
      dropout=${HDFS_DROPOUT:-${DROPOUT:-0.1}}
      type_head=${HDFS_TYPE_HEAD:-${TYPE_HEAD:-hybrid}}
      use_pos_enc=${HDFS_USE_POS_ENC:-${USE_POS_ENC:-false}}
      disable_time_gap=${HDFS_DISABLE_TIME_GAP:-${DISABLE_TIME_GAP:-false}}

      epoch=${HDFS_EPOCH:-${EPOCH:-60}}
      lr=${HDFS_LR:-${LR:-1e-4}}
      loss_weighting=${HDFS_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-fixed}}
      fm_loss_weight=${HDFS_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-0.05}}
      loss_lambda=${HDFS_LOSS_LAMBDA:-${LOSS_LAMBDA:-5.0}}
      checkpoint_metric=${HDFS_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_acc}}
      checkpoint_min_delta=${HDFS_CHECKPOINT_MIN_DELTA:-${CHECKPOINT_MIN_DELTA:-0.0001}}

      fm_sigma=${HDFS_FM_SIGMA:-${FM_SIGMA:-1.0}}
      train_step_size=${HDFS_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${HDFS_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.01}}
      n_samples=${HDFS_N_SAMPLES:-${N_SAMPLES:-100}}
      clamp_threshold=${HDFS_CLAMP_THRESHOLD:-${CLAMP_THRESHOLD:-2.5}}
      flow_cond_clip=${HDFS_FLOW_COND_CLIP:-${FLOW_COND_CLIP:-5.0}}
      ;;
    bgl)
      d_model=${BGL_D_MODEL:-${D_MODEL:-128}}
      d_inner_hid=${BGL_D_INNER_HID:-${D_INNER_HID:-256}}
      n_head=${BGL_N_HEAD:-${N_HEAD:-4}}
      n_layers=${BGL_N_LAYERS:-${N_LAYERS:-4}}
      d_k=${BGL_D_K:-${D_K:-32}}
      d_v=${BGL_D_V:-${D_V:-32}}
      dropout=${BGL_DROPOUT:-${DROPOUT:-0.1}}
      type_head=${BGL_TYPE_HEAD:-${TYPE_HEAD:-gmm}}
      use_pos_enc=${BGL_USE_POS_ENC:-${USE_POS_ENC:-false}}
      disable_time_gap=${BGL_DISABLE_TIME_GAP:-${DISABLE_TIME_GAP:-false}}

      epoch=${BGL_EPOCH:-${EPOCH:-60}}
      lr=${BGL_LR:-${LR:-5e-5}}
      loss_weighting=${BGL_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-adaptive}}
      fm_loss_weight=${BGL_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-1.0}}
      loss_lambda=${BGL_LOSS_LAMBDA:-${LOSS_LAMBDA:-1.0}}
      checkpoint_metric=${BGL_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_acc}}
      checkpoint_min_delta=${BGL_CHECKPOINT_MIN_DELTA:-${CHECKPOINT_MIN_DELTA:-0.0001}}

      fm_sigma=${BGL_FM_SIGMA:-${FM_SIGMA:-0.01}}
      train_step_size=${BGL_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${BGL_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.01}}
      n_samples=${BGL_N_SAMPLES:-${N_SAMPLES:-100}}
      clamp_threshold=${BGL_CLAMP_THRESHOLD:-${CLAMP_THRESHOLD:-6.0}}
      flow_cond_clip=${BGL_FLOW_COND_CLIP:-${FLOW_COND_CLIP:-5.0}}
      ;;
    openstack)
      d_model=${OPENSTACK_D_MODEL:-${D_MODEL:-128}}
      d_inner_hid=${OPENSTACK_D_INNER_HID:-${D_INNER_HID:-256}}
      n_head=${OPENSTACK_N_HEAD:-${N_HEAD:-4}}
      n_layers=${OPENSTACK_N_LAYERS:-${N_LAYERS:-4}}
      d_k=${OPENSTACK_D_K:-${D_K:-16}}
      d_v=${OPENSTACK_D_V:-${D_V:-16}}
      dropout=${OPENSTACK_DROPOUT:-${DROPOUT:-0.1}}
      type_head=${OPENSTACK_TYPE_HEAD:-${TYPE_HEAD:-gmm}}
      use_pos_enc=${OPENSTACK_USE_POS_ENC:-${USE_POS_ENC:-false}}
      disable_time_gap=${OPENSTACK_DISABLE_TIME_GAP:-${DISABLE_TIME_GAP:-false}}

      epoch=${OPENSTACK_EPOCH:-${EPOCH:-60}}
      lr=${OPENSTACK_LR:-${LR:-1e-4}}
      loss_weighting=${OPENSTACK_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-adaptive}}
      fm_loss_weight=${OPENSTACK_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-1.0}}
      loss_lambda=${OPENSTACK_LOSS_LAMBDA:-${LOSS_LAMBDA:-0.1}}
      checkpoint_metric=${OPENSTACK_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_loss}}
      checkpoint_min_delta=${OPENSTACK_CHECKPOINT_MIN_DELTA:-${CHECKPOINT_MIN_DELTA:-0.0001}}

      fm_sigma=${OPENSTACK_FM_SIGMA:-${FM_SIGMA:-1.0}}
      train_step_size=${OPENSTACK_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${OPENSTACK_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.01}}
      n_samples=${OPENSTACK_N_SAMPLES:-${N_SAMPLES:-100}}
      clamp_threshold=${OPENSTACK_CLAMP_THRESHOLD:-${CLAMP_THRESHOLD:-2.5}}
      flow_cond_clip=${OPENSTACK_FLOW_COND_CLIP:-${FLOW_COND_CLIP:-5.0}}
      ;;
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
      loss_weighting=${THUNDERBIRD_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-fixed}}
      fm_loss_weight=${THUNDERBIRD_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-0.05}}
      loss_lambda=${THUNDERBIRD_LOSS_LAMBDA:-${LOSS_LAMBDA:-5.0}}
      checkpoint_metric=${THUNDERBIRD_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_acc}}
      checkpoint_min_delta=${THUNDERBIRD_CHECKPOINT_MIN_DELTA:-${CHECKPOINT_MIN_DELTA:-0.0001}}

      fm_sigma=${THUNDERBIRD_FM_SIGMA:-${FM_SIGMA:-0.5}}
      train_step_size=${THUNDERBIRD_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${THUNDERBIRD_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.02}}
      n_samples=${THUNDERBIRD_N_SAMPLES:-${N_SAMPLES:-100}}
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
      loss_weighting=${SPIRIT_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-adaptive}}
      fm_loss_weight=${SPIRIT_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-1.0}}
      loss_lambda=${SPIRIT_LOSS_LAMBDA:-${LOSS_LAMBDA:-1.0}}
      checkpoint_metric=${SPIRIT_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_acc}}
      checkpoint_min_delta=${SPIRIT_CHECKPOINT_MIN_DELTA:-${CHECKPOINT_MIN_DELTA:-0.0001}}

      fm_sigma=${SPIRIT_FM_SIGMA:-${FM_SIGMA:-0.01}}
      train_step_size=${SPIRIT_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${SPIRIT_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.01}}
      n_samples=${SPIRIT_N_SAMPLES:-${N_SAMPLES:-100}}
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
      loss_weighting=${LIBERTY_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-adaptive}}
      fm_loss_weight=${LIBERTY_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-1.0}}
      loss_lambda=${LIBERTY_LOSS_LAMBDA:-${LOSS_LAMBDA:-1.0}}
      checkpoint_metric=${LIBERTY_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_acc}}
      checkpoint_min_delta=${LIBERTY_CHECKPOINT_MIN_DELTA:-${CHECKPOINT_MIN_DELTA:-0.0001}}

      fm_sigma=${LIBERTY_FM_SIGMA:-${FM_SIGMA:-0.01}}
      train_step_size=${LIBERTY_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${LIBERTY_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.01}}
      n_samples=${LIBERTY_N_SAMPLES:-${N_SAMPLES:-100}}
      clamp_threshold=${LIBERTY_CLAMP_THRESHOLD:-${CLAMP_THRESHOLD:-6.0}}
      flow_cond_clip=${LIBERTY_FLOW_COND_CLIP:-${FLOW_COND_CLIP:-5.0}}
      ;;
    hadoop)
      d_model=${HADOOP_D_MODEL:-${D_MODEL:-128}}
      d_inner_hid=${HADOOP_D_INNER_HID:-${D_INNER_HID:-256}}
      n_head=${HADOOP_N_HEAD:-${N_HEAD:-4}}
      n_layers=${HADOOP_N_LAYERS:-${N_LAYERS:-3}}
      d_k=${HADOOP_D_K:-${D_K:-32}}
      d_v=${HADOOP_D_V:-${D_V:-32}}
      dropout=${HADOOP_DROPOUT:-${DROPOUT:-0.1}}
      type_head=${HADOOP_TYPE_HEAD:-${TYPE_HEAD:-hybrid}}
      use_pos_enc=${HADOOP_USE_POS_ENC:-${USE_POS_ENC:-false}}
      disable_time_gap=${HADOOP_DISABLE_TIME_GAP:-${DISABLE_TIME_GAP:-false}}

      epoch=${HADOOP_EPOCH:-${EPOCH:-30}}
      lr=${HADOOP_LR:-${LR:-1e-4}}
      loss_weighting=${HADOOP_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-fixed}}
      fm_loss_weight=${HADOOP_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-0.05}}
      loss_lambda=${HADOOP_LOSS_LAMBDA:-${LOSS_LAMBDA:-5.0}}
      checkpoint_metric=${HADOOP_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_loss}}
      checkpoint_min_delta=${HADOOP_CHECKPOINT_MIN_DELTA:-${CHECKPOINT_MIN_DELTA:-0.0}}

      fm_sigma=${HADOOP_FM_SIGMA:-${FM_SIGMA:-0.5}}
      train_step_size=${HADOOP_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${HADOOP_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.02}}
      n_samples=${HADOOP_N_SAMPLES:-${N_SAMPLES:-100}}
      clamp_threshold=${HADOOP_CLAMP_THRESHOLD:-${CLAMP_THRESHOLD:-6.0}}
      flow_cond_clip=${HADOOP_FLOW_COND_CLIP:-${FLOW_COND_CLIP:-5.0}}
      ;;
    adfa_java|adfa_hydra_ssh|adfa_hydra_ftp|adfa_meter|adfa_web|adfa_adduser|logevol_hadoop|logevol_spark)
      # FlexLog's released ADFA-U/LOGEVOL-U pkl files contain ordered event
      # types but no timestamps. Keep one fixed, type-only DeniAD setting
      # across all eight configurations; do not tune per test set.
      d_model=${FLEXLOG_D_MODEL:-${D_MODEL:-128}}
      d_inner_hid=${FLEXLOG_D_INNER_HID:-${D_INNER_HID:-256}}
      n_head=${FLEXLOG_N_HEAD:-${N_HEAD:-4}}
      n_layers=${FLEXLOG_N_LAYERS:-${N_LAYERS:-4}}
      d_k=${FLEXLOG_D_K:-${D_K:-16}}
      d_v=${FLEXLOG_D_V:-${D_V:-16}}
      dropout=${FLEXLOG_DROPOUT:-${DROPOUT:-0.1}}
      type_head=${FLEXLOG_TYPE_HEAD:-${TYPE_HEAD:-hybrid_markov}}
      use_pos_enc=${FLEXLOG_USE_POS_ENC:-${USE_POS_ENC:-true}}
      disable_time_gap=true

      epoch=${FLEXLOG_EPOCH:-${EPOCH:-60}}
      lr=${FLEXLOG_LR:-${LR:-1e-4}}
      loss_weighting=${FLEXLOG_LOSS_WEIGHTING:-${LOSS_WEIGHTING:-fixed}}
      fm_loss_weight=${FLEXLOG_FM_LOSS_WEIGHT:-${FM_LOSS_WEIGHT:-0.05}}
      loss_lambda=${FLEXLOG_LOSS_LAMBDA:-${LOSS_LAMBDA:-5.0}}
      checkpoint_metric=${FLEXLOG_CHECKPOINT_METRIC:-${CHECKPOINT_METRIC:-valid_acc}}
      checkpoint_min_delta=${FLEXLOG_CHECKPOINT_MIN_DELTA:-${CHECKPOINT_MIN_DELTA:-0.0001}}

      fm_sigma=${FLEXLOG_FM_SIGMA:-${FM_SIGMA:-0.5}}
      train_step_size=${FLEXLOG_TRAIN_STEP_SIZE:-${TRAIN_STEP_SIZE:-0.05}}
      eval_step_size=${FLEXLOG_EVAL_STEP_SIZE:-${EVAL_STEP_SIZE:-0.02}}
      n_samples=${FLEXLOG_N_SAMPLES:-${N_SAMPLES:-100}}
      clamp_threshold=${FLEXLOG_CLAMP_THRESHOLD:-${CLAMP_THRESHOLD:-6.0}}
      flow_cond_clip=${FLEXLOG_FLOW_COND_CLIP:-${FLOW_COND_CLIP:-5.0}}
      ;;
    *)
      echo "ERROR: no hyperparameter defaults for dataset $dataset" >&2
      return 1
      ;;
  esac
}

train_batch_size() {
  case "$1" in
    hdfs) echo "${HDFS_BATCH_SIZE:-${BATCH_SIZE:-128}}" ;;
    bgl) echo "${BGL_BATCH_SIZE:-${BATCH_SIZE:-128}}" ;;
    openstack) echo "${OPENSTACK_BATCH_SIZE:-${BATCH_SIZE:-32}}" ;;
    thunderbird) echo "${THUNDERBIRD_BATCH_SIZE:-${BATCH_SIZE:-128}}" ;;
    spirit) echo "${SPIRIT_BATCH_SIZE:-${BATCH_SIZE:-128}}" ;;
    liberty) echo "${LIBERTY_BATCH_SIZE:-${BATCH_SIZE:-64}}" ;;
    # Hadoop is very long in the current preprocessing. Keep it tiny if enabled.
    hadoop) echo "${HADOOP_BATCH_SIZE:-${BATCH_SIZE:-1}}" ;;
    adfa_java|adfa_hydra_ssh|adfa_hydra_ftp|adfa_meter|adfa_web|adfa_adduser)
      echo "${FLEXLOG_ADFA_BATCH_SIZE:-${BATCH_SIZE:-2}}"
      ;;
    logevol_hadoop) echo "${FLEXLOG_HADOOP_BATCH_SIZE:-${BATCH_SIZE:-256}}" ;;
    logevol_spark) echo "${FLEXLOG_SPARK_BATCH_SIZE:-${BATCH_SIZE:-4}}" ;;
    *) echo "${BATCH_SIZE:-64}" ;;
  esac
}

eval_batch_size() {
  case "$1" in
    hdfs) echo "${HDFS_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-64}}" ;;
    bgl) echo "${BGL_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-64}}" ;;
    openstack) echo "${OPENSTACK_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-64}}" ;;
    thunderbird) echo "${THUNDERBIRD_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-128}}" ;;
    spirit) echo "${SPIRIT_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-64}}" ;;
    liberty) echo "${LIBERTY_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-64}}" ;;
    hadoop) echo "${HADOOP_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-1}}" ;;
    adfa_java|adfa_hydra_ssh|adfa_hydra_ftp|adfa_meter|adfa_web|adfa_adduser)
      echo "${FLEXLOG_ADFA_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-2}}"
      ;;
    logevol_hadoop) echo "${FLEXLOG_HADOOP_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-512}}" ;;
    logevol_spark) echo "${FLEXLOG_SPARK_EVAL_BATCH_SIZE:-${EVAL_BATCH_SIZE:-4}}" ;;
    *) echo "${EVAL_BATCH_SIZE:-64}" ;;
  esac
}

build_model_extra_args() {
  model_extra_args=()
  if [ "$use_pos_enc" = "true" ]; then
    model_extra_args+=("-use_pos_enc")
  fi
  if [ "$disable_time_gap" = "true" ]; then
    model_extra_args+=("-disable_time_gap")
  fi
}

build_eval_extra_args() {
  eval_extra_args=()
  if [ "$dataset_adaptive_detection" != "1" ]; then
    eval_extra_args+=("-disable_dataset_adaptive_detection")
  fi
  if [ "$eval_segment_threshold_sweep" = "1" ]; then
    eval_extra_args+=("-eval_segment_threshold_sweep")
  fi
}

common_model_args() {
  local data_dir="$1"
  local batch_size="$2"
  shift 2
  python main.py \
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
    -batch_size "$batch_size" \
    -lr "$lr" \
    -loss_weighting "$loss_weighting" \
    -fm_loss_weight "$fm_loss_weight" \
    -loss_lambda "$loss_lambda" \
    -flow_cond_clip "$flow_cond_clip" \
    -time_norm_guard_threshold "$time_norm_guard_threshold" \
    -fm_sigma "$fm_sigma" \
    -solver_method "$solver_method" \
    -solver_step_size "$train_step_size" \
    -clamp_threshold "$clamp_threshold" \
    -n_samples "$n_samples" \
    -checkpoint_metric "$checkpoint_metric" \
    -checkpoint_min_delta "$checkpoint_min_delta" \
    -seed "$seed" \
    "$@"
}

for dataset in $datasets; do
  if [ "$dataset" = "hadoop" ] && [ "${ALLOW_HADOOP:-0}" != "1" ]; then
    echo "[Skip] hadoop is skipped by default because current sequences are very long. Set ALLOW_HADOOP=1 to run it."
    continue
  fi

  set_dataset_hparams "$dataset"
  data_dir=$(dataset_dir "$dataset")
  if [ ! -f "${data_dir}/train.pkl" ] || [ ! -f "${data_dir}/dev.pkl" ] || [ ! -f "${data_dir}/test.pkl" ]; then
    echo "ERROR: missing train/dev/test pkl under ${data_dir}" >&2
    exit 1
  fi

  tb=$(train_batch_size "$dataset")
  eb=$(eval_batch_size "$dataset")
  build_model_extra_args
  build_eval_extra_args
  ds_result_dir="$result_root/$dataset"
  ds_ckpt_dir="$ckpt_root/$dataset"
  mkdir -p "$ds_result_dir" "$ds_ckpt_dir"

  save_name="${dataset}_labeled_${type_head}_${checkpoint_metric}_ep${epoch}_dm${d_model}_bs${tb}_sigma${fm_sigma}_fmw${fm_loss_weight}_lambda${loss_lambda}"
  save_path="${ds_ckpt_dir}/${save_name}.pth"
  train_log="${ds_result_dir}/${save_name}_train.log"
  anomaly_result="${ds_result_dir}/${save_name}_anomaly"
  reliability_result="${ds_result_dir}/${save_name}_reliability"
  unified_prediction_path="${ds_result_dir}/${save_name}_unified_predictions.csv"
  benchmark_args=()
  unified_prediction_args=()
  if [ "$benchmark_efficiency" = "1" ]; then
    benchmark_args+=(
      -benchmark_efficiency
      -benchmark_warmup_batches "$benchmark_warmup_batches"
      -benchmark_repeats "$benchmark_repeats"
    )
  fi
  if [ "$save_unified_predictions" = "1" ]; then
    unified_prediction_args+=(
      -save_unified_predictions
      -unified_method Ours
      -unified_dataset "$dataset"
      -unified_prediction_path "$unified_prediction_path"
    )
  fi

  echo "================================================================"
  echo "[Dataset] $dataset"
  echo "Data:       $data_dir"
  echo "Checkpoint: $save_path"
  echo "Train/Eval batch: $tb / $eb"
  echo "Model:      head=$type_head layers=$n_layers d_k=$d_k d_v=$d_v dropout=$dropout"
  echo "Loss:       weighting=$loss_weighting fm_weight=$fm_loss_weight lambda=$loss_lambda metric=$checkpoint_metric"
  echo "Flow:       sigma=$fm_sigma train_step=$train_step_size eval_step=$eval_step_size clamp=$clamp_threshold"
  echo "Detection:  score=$anomaly_score_mode type_w=$type_score_weight time_w=$time_score_weight segment=$segment_score_mode topk=$segment_topk adaptive=$dataset_adaptive_detection"
  echo "Flags:      use_pos_enc=$use_pos_enc disable_time_gap=$disable_time_gap"
  echo "================================================================"

  if [ "$run_train" = "1" ]; then
    common_model_args "$data_dir" "$tb" "${model_extra_args[@]}" \
      -epoch "$epoch" \
      -eval_epoch "$eval_epoch" \
      -save_path "$save_path" \
      2>&1 | tee "$train_log"
  else
    echo "[Info] RUN_TRAIN=0; skip training."
  fi

  if [ ! -f "$save_path" ]; then
    echo "ERROR: checkpoint not found at $save_path" >&2
    exit 1
  fi

  if [ "$run_anomaly" = "1" ]; then
    common_model_args "$data_dir" "$eb" "${model_extra_args[@]}" \
      -just_eval \
      -eval_anomaly_detection \
      -load_path_name "$save_path" \
      -solver_step_size "$eval_step_size" \
      -uncertainty_mc "$uncertainty_mc" \
      -anomaly_quantile "$anomaly_quantile" \
      -uncertainty_quantile "$uncertainty_quantile" \
      -calibration_max_size "$calibration_max_size" \
      -anomaly_score_mode "$anomaly_score_mode" \
      -type_score_weight "$type_score_weight" \
      -time_score_weight "$time_score_weight" \
      -segment_score_mode "$segment_score_mode" \
      -segment_topk "$segment_topk" \
      -threshold_sweep_steps "$threshold_sweep_steps" \
      -save_result "$anomaly_result" \
      "${eval_extra_args[@]}" \
      "${benchmark_args[@]}" \
      "${unified_prediction_args[@]}" \
      2>&1 | tee "${anomaly_result}.log"
  fi

  if [ "$run_reliability" = "1" ]; then
    common_model_args "$data_dir" "$eb" "${model_extra_args[@]}" \
      -just_eval \
      -eval_reliability \
      -use_ensemble_correction \
      -use_drift_adapter \
      -load_path_name "$save_path" \
      -solver_step_size "$eval_step_size" \
      -uncertainty_mc "$uncertainty_mc" \
      -anomaly_quantile "$anomaly_quantile" \
      -uncertainty_quantile "$uncertainty_quantile" \
      -calibration_max_size "$calibration_max_size" \
      -anomaly_score_mode "$anomaly_score_mode" \
      -type_score_weight "$type_score_weight" \
      -time_score_weight "$time_score_weight" \
      -segment_score_mode "$segment_score_mode" \
      -segment_topk "$segment_topk" \
      -ensemble_k "$ensemble_k" \
      -ensemble_samples "$ensemble_samples" \
      -ensemble_kernel "$ensemble_kernel" \
      -ensemble_noise_scale "$ensemble_noise_scale" \
      -ensemble_correction_weight "$ensemble_correction_weight" \
      -ensemble_support_quantile "$ensemble_support_quantile" \
      -ensemble_max_reference "$ensemble_max_reference" \
      -ensemble_max_search "$ensemble_max_search" \
      -save_result "$reliability_result" \
      "${eval_extra_args[@]}" \
      2>&1 | tee "${reliability_result}.log"
  fi
done

summary_path="$result_root/summary.csv"
summary_snapshot="$result_root/summary_${run_id}.csv"
python scripts/labeled/summarize_labeled_anomaly_results.py --result_root "$result_root" --output "$summary_path"
if [ -f "$summary_path" ]; then
  cp "$summary_path" "$summary_snapshot"
  echo "[OK] Snapshot summary: $summary_snapshot"
else
  echo "[Warn] Summary was not created; skip snapshot copy."
fi
