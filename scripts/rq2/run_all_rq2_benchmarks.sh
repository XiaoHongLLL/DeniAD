#!/usr/bin/env bash
set -euo pipefail

# Recompute the RQ2 generation table after model-code changes.
#
# It runs the existing AIOps generation benchmarks first:
#   HDFS, BGL, OpenStack
# and then runs the new large labeled log datasets:
#   Thunderbird, Spirit, Liberty
#
# Example:
#   bash scripts/rq2/run_all_rq2_benchmarks.sh
#
# To run only the new datasets:
#   RUN_AIOPS=0 bash scripts/rq2/run_all_rq2_benchmarks.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

python_bin=${PYTHON:-python}
run_aiops=${RUN_AIOPS:-1}
run_new_logs=${RUN_NEW_LOGS:-1}
new_datasets=${NEW_DATASETS:-"thunderbird spirit liberty"}
summary_out=${SUMMARY_OUT:-./results/rq2_generation_summary_all.csv}

run_existing_aiops() {
  if [ -f scripts/run_aiops_benchmarks.sh ]; then
    bash scripts/run_aiops_benchmarks.sh
    return
  fi

  if [ ! -f scripts/run_dataset_benchmark.sh ]; then
    echo "ERROR: neither scripts/run_aiops_benchmarks.sh nor scripts/run_dataset_benchmark.sh exists." >&2
    echo "Please sync the benchmark scripts, or run with RUN_AIOPS=0 to only recompute Thunderbird/Spirit/Liberty." >&2
    exit 1
  fi

  echo "[Info] scripts/run_aiops_benchmarks.sh not found; falling back to scripts/run_dataset_benchmark.sh."

  DATASET=hdfs \
  DATA_DIR=./data/data_hdfs/ \
  TYPE_HEAD=${HDFS_TYPE_HEAD:-hybrid} \
  FM_SIGMA=${HDFS_FM_SIGMA:-1.0} \
  LOSS_WEIGHTING=${HDFS_LOSS_WEIGHTING:-fixed} \
  LOSS_LAMBDA=${HDFS_LOSS_LAMBDA:-5.0} \
  FM_LOSS_WEIGHT=${HDFS_FM_LOSS_WEIGHT:-0.05} \
  TRAIN_BATCH_SIZE=${HDFS_TRAIN_BATCH_SIZE:-128} \
  EVAL_BATCH_SIZE=${HDFS_EVAL_BATCH_SIZE:-64} \
  CHECKPOINT_METRIC=${HDFS_CHECKPOINT_METRIC:-valid_acc} \
  bash scripts/run_dataset_benchmark.sh

  DATASET=bgl \
  DATA_DIR=./data/data_bgl/ \
  TYPE_HEAD=${BGL_TYPE_HEAD:-gmm} \
  LR=${BGL_LR:-5e-5} \
  DROPOUT=${BGL_DROPOUT:-0.1} \
  D_K=${BGL_D_K:-32} \
  D_V=${BGL_D_V:-32} \
  FM_SIGMA=${BGL_FM_SIGMA:-0.01} \
  LOSS_WEIGHTING=${BGL_LOSS_WEIGHTING:-adaptive} \
  LOSS_LAMBDA=${BGL_LOSS_LAMBDA:-1.0} \
  FM_LOSS_WEIGHT=${BGL_FM_LOSS_WEIGHT:-1.0} \
  CLAMP_THRESHOLD=${BGL_CLAMP_THRESHOLD:-6.0} \
  TRAIN_BATCH_SIZE=${BGL_TRAIN_BATCH_SIZE:-128} \
  EVAL_BATCH_SIZE=${BGL_EVAL_BATCH_SIZE:-128} \
  CHECKPOINT_METRIC=${BGL_CHECKPOINT_METRIC:-valid_acc} \
  bash scripts/run_dataset_benchmark.sh

  DATASET=openstack \
  DATA_DIR=./data/data_openstack/ \
  TYPE_HEAD=${OPENSTACK_TYPE_HEAD:-gmm} \
  FM_SIGMA=${OPENSTACK_FM_SIGMA:-1.0} \
  LOSS_WEIGHTING=${OPENSTACK_LOSS_WEIGHTING:-adaptive} \
  LOSS_LAMBDA=${OPENSTACK_LOSS_LAMBDA:-0.1} \
  FM_LOSS_WEIGHT=${OPENSTACK_FM_LOSS_WEIGHT:-1.0} \
  TRAIN_BATCH_SIZE=${OPENSTACK_TRAIN_BATCH_SIZE:-32} \
  EVAL_BATCH_SIZE=${OPENSTACK_EVAL_BATCH_SIZE:-64} \
  CHECKPOINT_METRIC=${OPENSTACK_CHECKPOINT_METRIC:-valid_loss} \
  bash scripts/run_dataset_benchmark.sh
}

if [ "$run_aiops" = "1" ]; then
  echo "================================================================"
  echo "[RQ2] Running existing AIOps datasets: HDFS, BGL, OpenStack"
  echo "================================================================"
  run_existing_aiops
else
  echo "[Info] RUN_AIOPS=0; skip HDFS/BGL/OpenStack."
fi

if [ "$run_new_logs" = "1" ]; then
  echo "================================================================"
  echo "[RQ2] Running new log datasets: $new_datasets"
  echo "================================================================"
  DATASETS="$new_datasets" bash scripts/rq2/run_rq2_labeled_log_benchmarks.sh
else
  echo "[Info] RUN_NEW_LOGS=0; skip Thunderbird/Spirit/Liberty."
fi

"$python_bin" scripts/rq2/summarize_rq2_generation_results.py \
  --result_root ./results \
  --output "$summary_out" \
  --datasets hdfs bgl openstack thunderbird spirit liberty

echo "[OK] RQ2 all-dataset summary: $summary_out"
