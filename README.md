# DeniAD

DeniAD is a log anomaly detection model for non-stationary software systems.
It learns normal log-sequence behavior by jointly modeling the next event type
and its inter-event time. For software-change runs, it further combines global
anomaly evidence, local normal-memory support, and service-absence evidence to
distinguish benign Expected drift from harmful Unexpected drift.

## Requirements

The reference environment is:

- Ubuntu 22.04
- Python 3.9
- PyTorch 2.5.1
- CUDA 12.1
- `torchdiffeq` 0.2.5
- NVIDIA GPU with CUDA support

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate deniad
```

Alternatively, install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Repository structure

```text
.
├── main.py                         # training and evaluation entry point
├── transformer/                    # event-history encoder and prediction heads
├── flow_matching/                  # conditional flow-matching components
├── preprocess/                     # sequence data loader
├── scripts/
│   ├── labeled/                    # public log benchmark runner
│   ├── rq1/                        # five-dataset launch script
│   ├── rq2/                        # type/time modeling variants
│   └── rq4/                        # run-level diagnosis and component ablation
├── dataset/
│   ├── prepare/                    # public log preprocessing
│   ├── public_logs/                # public dataset source information
│   ├── trainticket_collection/     # Train-Ticket data construction pipeline
│   └── trainticket_processed/      # processed Train-Ticket dataset archive
└── configs/trainticket_collection/ # workload and scenario configurations
```

## Public log datasets

The public benchmark uses HDFS, BGL, Thunderbird, Spirit, and Liberty.

- HDFS, BGL, and Thunderbird: <https://github.com/logpai/loghub>
- Spirit and Liberty: <https://www.usenix.org/cfdr-data>

After downloading the raw logs, preprocess all five datasets with:

```bash
python dataset/prepare/prepare_labeled_log_datasets.py \
  --raw_root /path/to/raw_logs \
  --out_root ./data \
  --datasets hdfs bgl thunderbird spirit liberty \
  --seed 2023
```

The generated files are stored as:

```text
data/
├── labeled_hdfs/{train,dev,test}.pkl
├── labeled_bgl/{train,dev,test}.pkl
├── labeled_thunderbird/{train,dev,test}.pkl
├── labeled_spirit/{train,dev,test}.pkl
└── labeled_liberty/{train,dev,test}.pkl
```

Train and evaluate DeniAD on the five datasets:

```bash
mkdir -p logs

RUN_TAG=public_logs_seed2023

nohup env \
  DEVICE=0 \
  SEED=2023 \
  DATASETS="bgl hdfs thunderbird spirit liberty" \
  RUN_TRAIN=1 \
  RUN_ANOMALY=1 \
  RUN_RELIABILITY=0 \
  DATASET_ADAPTIVE_DETECTION=0 \
  EVAL_SEGMENT_THRESHOLD_SWEEP=0 \
  CKPT_ROOT="./checkpoints/public_logs/$RUN_TAG" \
  RESULT_ROOT="./results/public_logs/$RUN_TAG" \
  RUN_TAG="$RUN_TAG" \
  bash scripts/rq1/run_rq1_five_datasets.sh \
  > "logs/${RUN_TAG}.log" 2>&1 &
```

Monitor the run:

```bash
tail -f "logs/${RUN_TAG}.log"
```

The aggregated metrics are written to:

```text
results/public_logs/<run-tag>/summary.csv
```

## Type and time modeling variants

The repository provides four configurations:

- `Type-only`: models the conditional event-type distribution;
- `Time-only`: models the inter-event-time distribution;
- `Independent`: models type and time independently;
- `OursJoin`: models the time distribution conditioned on event history and
  event type.

Run the complete comparison with:

```bash
bash scripts/rq2/run_all_rq2_benchmarks.sh
```

Dataset-specific launch scripts and result summarizers are available under
`scripts/rq2/`.

## Train-Ticket Expected/Unexpected drift data

Extract the processed dataset from the repository root:

```bash
unzip \
  dataset/trainticket_processed/data_cloud_expected_unexpected_expanded100_v0_4.zip \
  -d .
```

Validate the dataset:

```bash
python dataset/trainticket_processed/validate_dataset.py
```

The validation output should report:

```text
run counts: {'train': 30, 'dev': 38, 'test': 100}
sequence counts: {'train': 879, 'dev': 1206, 'test': 2934}
dim_process: 699
```

The run-level split contains 30 normal reference runs, 38 development runs,
and 100 test runs. The test split contains 50 Expected and 50 Unexpected runs.

Train the model, select run-level thresholds on the development split, and
evaluate the frozen configuration on the test split:

```bash
mkdir -p logs

RUN_TAG=trainticket_seed2023
RESULT_ROOT="./results/trainticket/$RUN_TAG"

nohup env \
  DEVICE=0 \
  SEED=2023 \
  RUN_TRAIN=1 \
  RQ4_CANDIDATE_MODE=score \
  RUN_LEVEL_STATE_VETO=off \
  ABSENCE_CONTEXT_MODE=hybrid \
  ABSENCE_REFERENCE_PATH="" \
  RESULT_ROOT="$RESULT_ROOT" \
  RUN_TAG="$RUN_TAG" \
  bash scripts/rq4/run_rq4_expanded100_v04_formal.sh \
  > "logs/${RUN_TAG}.log" 2>&1 &
```

Monitor the run:

```bash
tail -f "logs/${RUN_TAG}.log"
```

The main run-level output is stored under:

```text
results/trainticket/<run-tag>/test/
```

This directory contains event-level predictions, run-level predictions,
confusion matrices, and the binary Expected/Unexpected summary table.

## Component ablation

Run the component comparison with a trained checkpoint:

```bash
DATA_DIR=./data_cloud_expected_unexpected_expanded100_v0_4 \
CHECKPOINT=./checkpoints/rq4/cloud_pilot/cloud_pilot_expanded100_v04_logonly_gmm_ep60.pth \
RESULT_DIR=./results/trainticket/component_ablation \
ABSENCE_CONTEXT_MODE=hybrid \
ABSENCE_REFERENCE_PATH="" \
bash scripts/rq4/run_rq3_core_ablation.sh
```

## Dataset construction

The scripts under `dataset/trainticket_collection/` reproduce the data
collection and conversion pipeline for a deployed Train-Ticket Kubernetes
system. The main stages are:

1. generate and freeze a software-change scenario plan;
2. execute each scenario and collect logs and Kubernetes state;
3. assign semantic labels and run quality checks;
4. create the train/dev/test manifest;
5. convert valid runs into model-ready event sequences.

Collection requires `kubectl`, a running Train-Ticket deployment, and the
workload endpoint supplied through `SCWARN_BASE_URL`. Detailed commands are in
`dataset/trainticket_collection/README.md`.

## Data format

Each split is stored as a pickle dictionary:

```python
{
    "dim_process": number_of_event_types,
    "train": [sequence_1, sequence_2, ...]
}
```

Each event contains at least:

```python
{
    "time_since_start": float,
    "time_since_last_event": float,
    "type_event": int,
    "label": int
}
```

The development and test files use the corresponding `dev` and `test` keys.

## Citation

If you use DeniAD or the processed Train-Ticket dataset, please cite the
associated paper. The final BibTeX entry and archived software DOI will be
added after publication.

