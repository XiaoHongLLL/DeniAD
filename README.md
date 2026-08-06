# DeniAD research artifact (review snapshot)

This repository contains the minimal code and processed data needed to inspect
and reproduce the DeniAD experiments described in the accompanying manuscript.
DeniAD models the conditional distribution of log-event types and inter-event
times, and performs selective run-level diagnosis of Expected and Unexpected
software drift.

> This is a pre-publication review snapshot. Complete the items in
> `docs/PUBLICATION_BLOCKERS.md` before making the repository public.

## What is included

- Core DeniAD model and joint type--time likelihood implementation.
- RQ1--RQ4 training, evaluation, calibration, and aggregation scripts.
- Preprocessing code for HDFS, BGL, Thunderbird, Spirit, and Liberty.
- Train-Ticket workload, collection, labeling, quality-control, and dataset
  construction scripts.
- A compressed processed Train-Ticket artifact with 30 train/reference runs,
  38 development runs, and 100 formal test runs (50 Expected and 50
  Unexpected).

## What is intentionally not included

- Model checkpoints, caches, raw execution logs, failed runs, notebooks, local
  IDE files, and exploratory server patches.
- Post-paper experiments, supervised calibration prototypes, and FlexLog or
  Multi-CAD source trees.
- Raw third-party HDFS/BGL/Thunderbird/Spirit/Liberty logs. Obtain these from
  their original repositories and respect their terms.
- Kubernetes binaries, private cluster addresses, credentials, database
  passwords, or machine-specific paths.

These exclusions reduce the repository size without withholding implementation
details needed for the paper's central claims.

## Repository layout

```text
main.py                              DeniAD training and evaluation entry point
transformer/                         event-history encoder and prediction heads
flow_matching/                       conditional flow-matching components
preprocess/                          model data loader
scripts/rq1 ... scripts/rq4          paper experiment pipelines
dataset/prepare/                     public-log preprocessing
dataset/trainticket_collection/      Train-Ticket run collection and labeling
dataset/trainticket_processed/       compressed processed RQ4 dataset
configs/trainticket_collection/      workload and scenario definitions
docs/                                release, provenance, and availability notes
```

## Environment

The reported experiments used Python 3.9, PyTorch 2.5.1, CUDA 12.1, and one
NVIDIA A100 80-GB GPU. Create the environment with:

```bash
conda env create -f environment.yml
conda activate deniad
```

## Prepare the processed Train-Ticket data

From the repository root:

```bash
unzip dataset/trainticket_processed/data_cloud_expected_unexpected_expanded100_v0_4.zip
python scripts/rq4/make_dev_as_test_dataset.py \
  --source data_cloud_expected_unexpected_expanded100_v0_4 \
  --output data_cloud_expected_unexpected_expanded100_v0_4_dev_as_test
```

Validate the split metadata:

```bash
python dataset/trainticket_processed/validate_dataset.py
```

Expected run counts are 30 train, 38 dev, and 100 test.

## RQ1: public log anomaly detection

Download the raw datasets from the original sources, place them under one raw
root, and run:

```bash
python dataset/prepare/prepare_labeled_log_datasets.py \
  --raw_root /path/to/raw_logs \
  --out_root ./data \
  --datasets hdfs bgl thunderbird spirit liberty \
  --seed 2023

RUN_TAG=rq1_seed2023 \
DATASETS="bgl hdfs thunderbird spirit liberty" \
bash scripts/rq1/run_rq1_five_datasets.sh
```

The main RQ1 protocol disables dataset-adaptive profile-only detection and does
not select thresholds on test labels.

## RQ2: joint type--time modeling

```bash
bash scripts/rq2/run_all_rq2_benchmarks.sh
```

The RQ2 variants are Type-only, Time-only, Independent, and OursJoin. They are
probabilistic ablations of conditional joint modeling, not conventional
classification baselines.

## RQ3: core mechanism ablation

```bash
ABSENCE_CONTEXT_MODE=hybrid \
ABSENCE_REFERENCE_PATH="" \
bash scripts/rq4/run_rq3_core_ablation.sh
```

## RQ4: Expected/Unexpected run-level diagnosis

The legacy Absence-aware revision configuration corresponding to the current
main-table protocol can be launched with:

```bash
RUN_TAG=expanded100_legacy_seed2023 \
RESULT_ROOT=./results/rq4/expanded100_legacy_seed2023 \
DEVICE=0 \
SEED=2023 \
RUN_TRAIN=1 \
RQ4_CANDIDATE_MODE=score \
RUN_LEVEL_STATE_VETO=off \
ABSENCE_CONTEXT_MODE=hybrid \
ABSENCE_REFERENCE_PATH="" \
bash scripts/rq4/run_rq4_expanded100_v04_formal.sh
```

Thresholds are selected on dev and frozen before test evaluation. Do not use
test-label threshold sweeps for the reported result.

## Train-Ticket collection code

The processed data are sufficient to reproduce model evaluation. Recollecting
the dataset additionally requires a deployed Train-Ticket Kubernetes system,
`kubectl`, the workload profiles under `configs/trainticket_collection/`, and
the scripts under `dataset/trainticket_collection/`. Cluster endpoints must be
provided through `SCWARN_BASE_URL`; no private endpoint or credential is stored
in this artifact. See `dataset/trainticket_collection/README.md`.

## Public data sources

- HDFS, BGL, and Thunderbird: [Loghub](https://github.com/logpai/loghub).
- Spirit and Liberty: [USENIX Computer Failure Data Repository](https://www.usenix.org/cfdr-data).
- Train-Ticket system: use the upstream Train-Ticket repository identified in
  the manuscript and record the exact commit in the final release metadata.

## Citation

Complete `CITATION.cff.template`, rename it to `CITATION.cff`, and replace all
bracketed fields after the manuscript and repository identifiers are final.

## Licence and third-party code

Do not publish this snapshot until the licensing questions in
`THIRD_PARTY_NOTICES.md` and `docs/PUBLICATION_BLOCKERS.md` are resolved. In
particular, a top-level licence cannot override third-party terms.

