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
|-- main.py                         # training and evaluation entry point
|-- transformer/                    # event-history encoder and prediction heads
|-- flow_matching/                  # conditional flow-matching components
|-- preprocess/                     # sequence data loader
|-- scripts/
|   |-- labeled/                    # public log benchmark runner
|   |-- rq1/                        # five-dataset launch script
|   |-- rq2/                        # type/time modeling variants
|   `-- rq4/                        # run-level diagnosis and component ablation
|-- dataset/
|   |-- prepare/                    # public log preprocessing
|   |-- public_logs/                # public dataset source information
|   |-- trainticket_collection/     # Train-Ticket data construction pipeline
|   `-- trainticket_processed/      # processed Train-Ticket dataset archive
`-- configs/trainticket_collection/ # workload and scenario configurations
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
|-- labeled_hdfs/{train,dev,test}.pkl
|-- labeled_bgl/{train,dev,test}.pkl
|-- labeled_thunderbird/{train,dev,test}.pkl
|-- labeled_spirit/{train,dev,test}.pkl
`-- labeled_liberty/{train,dev,test}.pkl
```





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

## Train-Ticket dataset construction

The dataset was collected from the open-source Train-Ticket microservice system
deployed on Kubernetes. Each run contains a stable pre-change period, one
controlled software or operational change, and a post-change observation
period under a fixed workload profile. Logs and Kubernetes runtime states are
collected throughout the run.

Expected runs contain benign system evolution, such as compatible or
low-impact configuration changes, resource scaling, workload changes, pod
migration, and successful no-op redeployment. Unexpected runs contain harmful
changes, including service termination, invalid dependency ports, resource
limits, connection-pool exhaustion, and loss of weakly observable services.
Runs that fail collection, parsing, oracle validation, or quality checks are
excluded before the dataset split is frozen.

The released dataset contains 30 normal reference runs for training, 38 runs
for development (22 Expected and 16 Unexpected), and 100 runs for testing
(50 Expected and 50 Unexpected). Logs are grouped by run, service, and execution
phase, parsed into event types, and converted into sequences containing event
types and inter-event times. The collection and conversion scripts are under
`dataset/trainticket_collection/`, and the processed dataset is provided under
`dataset/trainticket_processed/`.

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
