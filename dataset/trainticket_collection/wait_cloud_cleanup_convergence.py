#!/usr/bin/env python3
"""Wait until Kubernetes cleanup has converged to the declared workload replica set."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def kubectl_json(kubectl: str, args: list[str], timeout: int) -> dict:
    proc = subprocess.run(
        [kubectl, *args],
        text=True,
        capture_output=True,
        timeout=timeout,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return json.loads(proc.stdout)


def pod_ready(pod: dict) -> bool:
    if pod.get("metadata", {}).get("deletionTimestamp"):
        return False
    if pod.get("status", {}).get("phase") != "Running":
        return False
    return any(
        condition.get("type") == "Ready" and condition.get("status") == "True"
        for condition in pod.get("status", {}).get("conditions") or []
    )


def desired_replicas(items: list[dict]) -> int:
    return sum(int((item.get("spec") or {}).get("replicas") or 0) for item in items)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="trainticket-pilot")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--timeout-seconds", type=int, default=300)
    parser.add_argument("--poll-seconds", type=float, default=5.0)
    parser.add_argument("--command-timeout-seconds", type=int, default=30)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    started = time.monotonic()
    observations: list[dict] = []
    passed = False
    last_error = ""
    while time.monotonic() - started <= args.timeout_seconds:
        try:
            pods = kubectl_json(
                args.kubectl,
                ["get", "pods", "-n", args.namespace, "-o", "json"],
                args.command_timeout_seconds,
            ).get("items", [])
            deployments = kubectl_json(
                args.kubectl,
                ["get", "deployments", "-n", args.namespace, "-o", "json"],
                args.command_timeout_seconds,
            ).get("items", [])
            statefulsets = kubectl_json(
                args.kubectl,
                ["get", "statefulsets", "-n", args.namespace, "-o", "json"],
                args.command_timeout_seconds,
            ).get("items", [])
            expected = desired_replicas(deployments) + desired_replicas(statefulsets)
            deleting = [
                pod.get("metadata", {}).get("name", "")
                for pod in pods
                if pod.get("metadata", {}).get("deletionTimestamp")
            ]
            not_ready = [
                pod.get("metadata", {}).get("name", "")
                for pod in pods
                if not pod_ready(pod)
            ]
            restart_sum = sum(
                int(status.get("restartCount") or 0)
                for pod in pods
                for status in pod.get("status", {}).get("containerStatuses") or []
            )
            observation = {
                "observed_at": utc_now(),
                "expected_pod_count": expected,
                "observed_pod_count": len(pods),
                "deleting_pods": deleting,
                "not_ready_pods": not_ready,
                "restart_sum": restart_sum,
            }
            observations.append(observation)
            passed = (
                expected > 0
                and len(pods) == expected
                and not deleting
                and not not_ready
                and restart_sum == 0
            )
            print(
                "cleanup convergence: "
                f"pods={len(pods)}/{expected} "
                f"not_ready={len(not_ready)} deleting={len(deleting)} "
                f"restarts={restart_sum}",
                flush=True,
            )
            if passed:
                break
        except (RuntimeError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            last_error = str(exc)
            observations.append({"observed_at": utc_now(), "error": last_error})
        time.sleep(args.poll_seconds)

    payload = {
        "generated_at": utc_now(),
        "namespace": args.namespace,
        "timeout_seconds": args.timeout_seconds,
        "elapsed_seconds": time.monotonic() - started,
        "pass": passed,
        "last_error": last_error,
        "observations": observations,
        "final_observation": observations[-1] if observations else {},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "pass": passed,
                "elapsed_seconds": payload["elapsed_seconds"],
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
