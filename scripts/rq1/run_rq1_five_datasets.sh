#!/usr/bin/env bash
set -euo pipefail

# Re-train and evaluate DeniAD on the five public
# labeled-log datasets used by RQ1. Each run is isolated so older checkpoints
# and result CSV files cannot be mistaken for the new experiment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

seed=${SEED:-2023}
run_tag=${RUN_TAG:-rq1_modified_model_seed${seed}_$(date +%Y%m%d_%H%M%S)}

export SEED="$seed"
export RUN_ID="$run_tag"
export DATASETS="${DATASETS:-bgl hdfs thunderbird spirit liberty}"
export RUN_TRAIN="${RUN_TRAIN:-1}"
export RUN_ANOMALY="${RUN_ANOMALY:-1}"
export RUN_RELIABILITY=0

# RQ1 must measure the modified type-time model. The legacy adaptive profiles
# switch several datasets to profile_only and would bypass the trained model.
export DATASET_ADAPTIVE_DETECTION="${DATASET_ADAPTIVE_DETECTION:-0}"

# The paper result uses the threshold fitted on normal dev data. A test-label
# best-F1 sweep is disabled because it is an oracle diagnostic, not a fair RQ1
# result.
export EVAL_SEGMENT_THRESHOLD_SWEEP="${EVAL_SEGMENT_THRESHOLD_SWEEP:-0}"

export CKPT_ROOT="${CKPT_ROOT:-./checkpoints/rq1/$run_tag}"
export RESULT_ROOT="${RESULT_ROOT:-./results/rq1/runs/$run_tag}"

mkdir -p "$CKPT_ROOT" "$RESULT_ROOT"

echo "================================================================"
echo "RQ1 five-dataset run"
echo "Run tag:       $run_tag"
echo "Datasets:      $DATASETS"
echo "Seed:          $SEED"
echo "Checkpoints:   $CKPT_ROOT"
echo "Results:       $RESULT_ROOT"
echo "Adaptive mode: $DATASET_ADAPTIVE_DETECTION (must be 0 for main RQ1)"
echo "================================================================"

bash scripts/labeled/run_labeled_anomaly_benchmarks.sh

echo "================================================================"
echo "RQ1 run completed."
echo "Main summary:  $RESULT_ROOT/summary.csv"
echo "Snapshot:      $RESULT_ROOT/summary_${run_tag}.csv"
echo "Use Segment_* columns for the segment/session-level RQ1 table."
echo "================================================================"
