#!/usr/bin/env bash
set -euo pipefail

# Reuse the five trained RQ1 checkpoints and evaluate three binary OOD policies
# at segment/session level. This script performs inference only.

device=${DEVICE:-0}
seed=${SEED:-2023}
run_tag=${RUN_TAG:-rq1_public_ood_binary_seed${seed}}
ckpt_root=${CKPT_ROOT:-./checkpoints/rq1/rq1_modified_model_seed2023_20260713}
result_root=${RESULT_ROOT:-./results/rq1/${run_tag}}
datasets=${DATASETS:-"hdfs bgl thunderbird spirit liberty"}

mkdir -p "$result_root" logs

env \
  DEVICE="$device" \
  SEED="$seed" \
  DATASETS="$datasets" \
  RUN_TRAIN=0 \
  RUN_ANOMALY=0 \
  RUN_RELIABILITY=1 \
  DATASET_ADAPTIVE_DETECTION=1 \
  EVAL_SEGMENT_THRESHOLD_SWEEP=0 \
  CKPT_ROOT="$ckpt_root" \
  RESULT_ROOT="$result_root" \
  RUN_ID="$run_tag" \
  bash scripts/labeled/run_labeled_anomaly_benchmarks.sh

python scripts/labeled/build_public_ood_binary_fpfn_table.py \
  --summary "$result_root/summary.csv" \
  --output "$result_root/public_five_datasets_ood_binary_fpfn.csv" \
  --datasets "$datasets"

echo "[OK] Final table: $result_root/public_five_datasets_ood_binary_fpfn.csv"
