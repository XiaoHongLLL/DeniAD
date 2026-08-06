#!/usr/bin/env bash
set -euo pipefail

# Prepare labeled log datasets used by RQ2 generation metrics.
#
# Example:
#   RAW_ROOT=/path/to/raw_logs DATASETS="thunderbird spirit liberty" \
#     bash scripts/rq2/prepare_rq2_log_datasets.sh
#
# The generated train/dev/test pkl files are compatible with the normal
# generation evaluation path in main.py. Event labels are kept in the pkl files
# but are ignored unless -eval_anomaly_detection or -eval_reliability is used.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

python_bin=${PYTHON:-python}
raw_root=${RAW_ROOT:-./data/raw}
out_root=${OUT_ROOT:-./data}
datasets=${DATASETS:-"thunderbird spirit liberty"}
force_prep=${FORCE_PREP:-0}

seed=${SEED:-2023}
train_ratio=${TRAIN_RATIO:-0.6}
dev_ratio=${DEV_RATIO:-0.2}
window_size=${WINDOW_SIZE:-50}
step_size=${STEP_SIZE:-50}
max_event_types=${MAX_EVENT_TYPES:-512}
min_count=${MIN_COUNT:-1}

max_lines_thunderbird=${MAX_LINES_THUNDERBIRD:-2000000}
max_lines_spirit=${MAX_LINES_SPIRIT:-2000000}
max_lines_liberty=${MAX_LINES_LIBERTY:-2000000}
skip_lines_spirit=${SKIP_LINES_SPIRIT:-0}
skip_lines_liberty=${SKIP_LINES_LIBERTY:-0}

dataset_dir() {
  case "$1" in
    thunderbird) echo "$out_root/labeled_thunderbird" ;;
    spirit) echo "$out_root/labeled_spirit" ;;
    liberty) echo "$out_root/labeled_liberty" ;;
    *) echo "ERROR: unsupported RQ2 dataset '$1'" >&2; return 1 ;;
  esac
}

has_processed_split() {
  local dir="$1"
  [ -f "$dir/train.pkl" ] && [ -f "$dir/dev.pkl" ] && [ -f "$dir/test.pkl" ]
}

prep_datasets=()
for dataset in $datasets; do
  data_dir=$(dataset_dir "$dataset")
  if [ "$force_prep" = "1" ] || ! has_processed_split "$data_dir"; then
    prep_datasets+=("$dataset")
  else
    echo "[Skip] $dataset already has train/dev/test pkl under $data_dir"
  fi
done

if [ "${#prep_datasets[@]}" -eq 0 ]; then
  echo "[OK] No preprocessing needed."
  exit 0
fi

echo "----------------------------------------------------------------"
echo "Preparing RQ2 log datasets"
echo "Raw root:       $raw_root"
echo "Output root:    $out_root"
echo "Datasets:       ${prep_datasets[*]}"
echo "Window/step:    $window_size / $step_size"
echo "Event types:    max=$max_event_types min_count=$min_count"
echo "Max lines:      thunderbird=$max_lines_thunderbird spirit=$max_lines_spirit liberty=$max_lines_liberty"
echo "----------------------------------------------------------------"

"$python_bin" scripts/prepare_labeled_log_datasets.py \
  --raw_root "$raw_root" \
  --out_root "$out_root" \
  --datasets "${prep_datasets[@]}" \
  --seed "$seed" \
  --train_ratio "$train_ratio" \
  --dev_ratio "$dev_ratio" \
  --window_size "$window_size" \
  --step_size "$step_size" \
  --max_event_types "$max_event_types" \
  --min_count "$min_count" \
  --max_lines_thunderbird "$max_lines_thunderbird" \
  --max_lines_spirit "$max_lines_spirit" \
  --max_lines_liberty "$max_lines_liberty" \
  --skip_lines_spirit "$skip_lines_spirit" \
  --skip_lines_liberty "$skip_lines_liberty"

for dataset in "${prep_datasets[@]}"; do
  data_dir=$(dataset_dir "$dataset")
  if ! has_processed_split "$data_dir"; then
    echo "ERROR: preprocessing finished but $data_dir is missing train/dev/test pkl." >&2
    exit 1
  fi
  echo "[OK] $dataset -> $data_dir"
done

