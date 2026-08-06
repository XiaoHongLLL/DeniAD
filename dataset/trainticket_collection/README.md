# Train-Ticket data collection

These scripts generate the run-level Expected/Unexpected software-change data
used by RQ4. They assume an already deployed Train-Ticket Kubernetes system.
No credentials, cluster addresses, Kubernetes binaries, container images, or
raw runs are bundled.

## Required external components

- Linux, Bash, Python 3.9+, and `kubectl` on `PATH`.
- A reachable Train-Ticket deployment and its exact upstream commit.
- The workload endpoint supplied as `SCWARN_BASE_URL`.
- Workload and scenario definitions in `configs/trainticket_collection/`.

## Main stages

1. Create and freeze a scenario plan with
   `create_cloud_formal_v0_3_plan.py` and
   `create_cloud_expansion_v0_4_plan.py`.
2. Execute the frozen plan with `run_cloud_formal_batch.py`.
3. Collect logs and state, apply semantic labeling, and run quality checks.
4. Build a split manifest with
   `create_cloud_expanded100_model_dataset_manifest.py`.
5. Convert valid raw runs to model input with `build_log_dataset.py`.

Generated raw runs are written under `artifacts/trainticket_runs/` by default
and are ignored by Git. Keep protocol freezes, status tables, checksums, and
invalid-run records when producing the final archived dataset.

