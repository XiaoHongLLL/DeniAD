#!/usr/bin/env python3
"""Run a preregistered Tencent Cloud Expected/Unexpected formal batch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def hours_since_uptime() -> str:
    try:
        text = subprocess.check_output(["uptime", "-s"], text=True).strip()
        boot = datetime.fromisoformat(text).replace(tzinfo=None)
        now = datetime.now()
        return f"{((now - boot).total_seconds() / 3600.0):.6f}"
    except Exception:
        return "0"


def completed_run_ids(status_rows: list[dict]) -> set[str]:
    return {row.get("run_id", "") for row in status_rows if row.get("status") == "complete"}


def annotate_manifest(repo: Path, row: dict) -> None:
    manifest_path = repo / "artifacts/trainticket_runs" / row["run_id"] / "run_manifest.json"
    if not manifest_path.exists():
        return
    payload = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    canonical_row = json.dumps(
        row, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    payload.update(
        {
            "split": row.get("split") or row.get("protocol_role") or "",
            "protocol_role": row.get("protocol_role") or "",
            "planned_semantic_label": row.get("semantic_label") or "",
            "target_benchmark_label": row.get("target_benchmark_label") or "",
            "scenario_template_id": row.get("scenario_template_id") or "",
            "implementation_instance_id": row.get("implementation_instance_id") or "",
            "workload_profile_id": row.get("workload_profile_id") or "",
            "workload_seed": row.get("seed") or row.get("matched_workload_seed") or "",
            "formal_test_sealed": str(row.get("formal_test_sealed") or "").lower()
            == "true",
            "blind_run_token": row.get("blind_run_token") or "",
            "v0_6_plan_freeze_id": row.get("v0_6_plan_freeze_id") or "",
            "v0_6_plan_row_sha256": hashlib.sha256(
                canonical_row.encode("utf-8")
            ).hexdigest(),
        }
    )
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--namespace", default="trainticket-pilot")
    parser.add_argument("--base-url", default="http://127.0.0.1:30467")
    parser.add_argument("--cloud-freeze-id", required=True)
    parser.add_argument("--cluster-type", default="linux_kubeadm_multinode")
    parser.add_argument("--threshold-file", required=True)
    parser.add_argument("--threshold-sha256", required=True)
    parser.add_argument("--unexpected-failure-ratio", default="0.01")
    parser.add_argument("--max-runs", type=int, default=0, help="0 means run all remaining rows.")
    parser.add_argument("--stable-seconds", type=int, default=300)
    parser.add_argument("--warmup-seconds", type=int, default=120)
    parser.add_argument("--pre-change-seconds", type=int, default=600)
    parser.add_argument("--post-change-seconds", type=int, default=900)
    parser.add_argument("--rate-per-second", type=float, default=0.2)
    parser.add_argument("--transition-budget-seconds", type=int, default=300)
    parser.add_argument("--post-stable-seconds", type=int, default=120)
    args = parser.parse_args()

    repo = Path.cwd()
    runner = repo / "dataset/trainticket_collection/run_cloud_formal_scenario.sh"
    plan_rows = read_csv(Path(args.plan))
    status_path = Path(args.status)
    status_rows = read_csv(status_path)
    done = completed_run_ids(status_rows)
    ran = 0

    for row in plan_rows:
        if row["run_id"] in done:
            print(f"skip complete: {row['run_id']}", flush=True)
            continue
        if args.max_runs and ran >= args.max_runs:
            break
        started = utc_now()
        run_log = status_path.parent / f"{row['run_id']}.orchestrator.log"
        cmd = [
            "bash", str(runner),
            "--scenario", row["scenario"],
            "--namespace", args.namespace,
            "--run-id", row["run_id"],
            "--base-url", args.base_url,
            "--workload-profile-id", row.get("workload_profile_id") or "W1_steady_core",
            "--post-workload-profile-id", row.get("post_workload_profile_id") or row.get("workload_profile_id") or "W1_steady_core",
            "--seed", row.get("seed") or row.get("matched_workload_seed") or "8501",
            "--stable-seconds", str(args.stable_seconds),
            "--warmup-seconds", str(args.warmup_seconds),
            "--pre-change-seconds", str(args.pre_change_seconds),
            "--post-change-seconds", str(args.post_change_seconds),
            "--rate-per-second", str(args.rate_per_second),
            "--transition-budget-seconds", str(args.transition_budget_seconds),
            "--post-stable-seconds", str(args.post_stable_seconds),
            "--batch-id", args.batch_id,
            "--block-id", row.get("block_id") or "",
            "--batch-position", row.get("batch_position") or row.get("ordinal") or "",
            "--preceding-scenario", row.get("preceding_scenario") or "none",
            "--matched-workload-seed", row.get("matched_workload_seed") or row.get("seed") or "",
            "--hours-since-cluster-start", hours_since_uptime(),
            "--cloud-freeze-id", args.cloud_freeze_id,
            "--cluster-type", args.cluster_type,
            "--threshold-file", args.threshold_file,
            "--threshold-sha256", args.threshold_sha256,
            "--unexpected-failure-ratio", args.unexpected_failure_ratio,
            "--cloud-protocol-version", row.get("cloud_protocol_version") or "linux-cloud-formal-v0.1",
            "--scenario-preference", "formal_expected_unexpected_label_probe",
        ]
        print(f"\n== [{row.get('ordinal')}/{len(plan_rows)}] {row['run_id']} ==", flush=True)
        print(" ".join(cmd), flush=True)
        run_log.parent.mkdir(parents=True, exist_ok=True)
        with run_log.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            assert proc.stdout is not None
            for line in proc.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                log.flush()
            exit_code = proc.wait()
        if exit_code == 0:
            annotate_manifest(repo, row)
        finished = utc_now()
        status_rows.append({
            "ordinal": row.get("ordinal"),
            "batch_id": args.batch_id,
            "execution_block": row.get("execution_block"),
            "block_id": row.get("block_id"),
            "batch_position": row.get("batch_position"),
            "scenario": row.get("scenario"),
            "expected_semantic": row.get("semantic_label"),
            "target_label_tendency": row.get("target_label_tendency") or row.get("expected_label_tendency"),
            "expected_label_tendency": row.get("expected_label_tendency") or row.get("target_label_tendency"),
            "matched_workload_seed": row.get("matched_workload_seed"),
            "seed": row.get("seed"),
            "run_id": row.get("run_id"),
            "started_at": started,
            "finished_at": finished,
            "exit_code": exit_code,
            "status": "complete" if exit_code == 0 else "failed",
            "run_log": str(run_log),
        })
        write_csv(status_path, status_rows)
        ran += 1
        if exit_code != 0:
            raise SystemExit(f"Mini batch stopped after failed run: {row['run_id']}")

    print(f"formal batch runner finished; newly completed runs={ran}; status={status_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
