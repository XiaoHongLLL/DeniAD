#!/usr/bin/env python3
"""Wait for a Kubernetes stable state using objective TT-Core pilot rules."""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def utc_text() -> str:
    return utc_now().isoformat()


def run_capture(command: list[str], timeout: int) -> dict:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        return {
            "command": " ".join(command),
            "exit_code": proc.returncode,
            "timed_out": False,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "command": " ".join(command),
            "exit_code": 124,
            "timed_out": True,
            "stdout": exc.stdout or "",
            "stderr": exc.stderr or "",
        }


def run_json(command: list[str], timeout: int) -> dict:
    cap = run_capture(command, timeout)
    if cap["exit_code"] != 0:
        return {"ok": False, "capture": cap, "items": []}
    try:
        payload = json.loads(cap["stdout"])
    except json.JSONDecodeError:
        return {"ok": False, "capture": cap, "items": []}
    return {"ok": True, "capture": cap, "payload": payload, "items": payload.get("items") or []}


def compact(capture: dict) -> dict:
    stdout = capture.get("stdout") or ""
    return {
        "command": capture.get("command", ""),
        "exit_code": capture.get("exit_code"),
        "timed_out": capture.get("timed_out", False),
        "stdout_bytes": len(stdout.encode("utf-8", errors="replace")),
        "stderr": capture.get("stderr") or "",
    }


def ready_status(pod: dict) -> str:
    for cond in pod.get("status", {}).get("conditions") or []:
        if cond.get("type") == "Ready":
            return str(cond.get("status") or "")
    return ""


def snapshot_state(kubectl: str, namespace: str, command_timeout: int) -> dict:
    deployments = run_json([kubectl, "get", "deployments", "-n", namespace, "-o", "json"], command_timeout)
    statefulsets = run_json([kubectl, "get", "statefulsets", "-n", namespace, "-o", "json"], command_timeout)
    pods = run_json([kubectl, "get", "pods", "-n", namespace, "-o", "json"], command_timeout)

    problems: list[str] = []
    restart_sum = 0
    pending_pods = 0
    if not deployments["ok"]:
        problems.append(f"deployments query failed: {deployments['capture'].get('stderr', '')}")
    if not statefulsets["ok"]:
        problems.append(f"statefulsets query failed: {statefulsets['capture'].get('stderr', '')}")
    if not pods["ok"]:
        problems.append(f"pods query failed: {pods['capture'].get('stderr', '')}")

    for item in deployments["items"]:
        name = item.get("metadata", {}).get("name", "")
        desired = int(item.get("spec", {}).get("replicas") or 1)
        status = item.get("status") or {}
        generation = item.get("metadata", {}).get("generation")
        observed = status.get("observedGeneration")
        updated = int(status.get("updatedReplicas") or 0)
        available = int(status.get("availableReplicas") or 0)
        unavailable = int(status.get("unavailableReplicas") or 0)
        if observed != generation:
            problems.append(f"{name}: observedGeneration lag")
        if updated != desired or available != desired or unavailable != 0:
            problems.append(
                f"{name}: replicas desired={desired} updated={updated} "
                f"available={available} unavailable={unavailable}"
            )

    for item in statefulsets["items"]:
        name = item.get("metadata", {}).get("name", "")
        desired = int(item.get("spec", {}).get("replicas") or 1)
        status = item.get("status") or {}
        generation = item.get("metadata", {}).get("generation")
        observed = status.get("observedGeneration")
        ready = int(status.get("readyReplicas") or 0)
        if observed != generation:
            problems.append(f"{name}: observedGeneration lag")
        if ready != desired:
            problems.append(f"{name}: ready={ready} desired={desired}")

    for pod in pods["items"]:
        name = pod.get("metadata", {}).get("name", "")
        phase = pod.get("status", {}).get("phase", "")
        ready = ready_status(pod)
        if phase == "Pending":
            pending_pods += 1
        if phase != "Running" or ready != "True":
            problems.append(f"{name}: phase={phase} ready={ready}")
        for cs in pod.get("status", {}).get("containerStatuses") or []:
            restart_sum += int(cs.get("restartCount") or 0)

    return {
        "checked_at": utc_text(),
        "ok": not problems,
        "problems": problems,
        "restart_sum": restart_sum,
        "pending_pods": pending_pods,
        "deployment_count": len(deployments["items"]),
        "statefulset_count": len(statefulsets["items"]),
        "pod_count": len(pods["items"]),
    }


def resource_snapshot(kubectl: str, namespace: str, command_timeout: int) -> dict:
    pods = run_json([kubectl, "get", "pods", "-n", namespace, "-o", "json"], command_timeout)
    pending = 0
    specs = []
    for pod in pods["items"]:
        if pod.get("status", {}).get("phase") == "Pending":
            pending += 1
        for container in pod.get("spec", {}).get("containers") or []:
            specs.append({
                "pod": pod.get("metadata", {}).get("name", ""),
                "container": container.get("name", ""),
                "requests": (container.get("resources") or {}).get("requests") or {},
                "limits": (container.get("resources") or {}).get("limits") or {},
            })
    return {
        "checked_at": utc_text(),
        "pending_pods": pending,
        "top_nodes": compact(run_capture([kubectl, "top", "nodes"], command_timeout)),
        "top_pods": compact(run_capture([kubectl, "top", "pods", "-n", namespace], command_timeout)),
        "pod_resource_specs": specs,
    }


def run_api_check(args: argparse.Namespace, output: Path) -> dict:
    if args.skip_api_check:
        return {"skipped": True, "passed": True, "reason": "SkipApiCheck"}
    script = Path(__file__).with_name("validate_core_closure.py")
    cmd = [
        args.python,
        str(script),
        "--allowlist", args.allowlist,
        "--namespace", args.namespace,
        "--kubectl", args.kubectl,
        "--base-url", args.gateway_base_url,
        "--repeats", "1",
        "--timeout-seconds", str(args.api_timeout_seconds),
        "--output", str(output),
    ]
    cap = run_capture(cmd, args.api_command_timeout_seconds)
    return {
        "skipped": False,
        "passed": cap["exit_code"] == 0,
        "output": str(output),
        "capture": compact(cap),
    }


def write_report(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def deployment_events_payload(
    kubectl: str,
    namespace: str,
    command_timeout: int,
    snapshots: list[dict],
    final_event: str,
) -> dict:
    k8s_events = run_json([kubectl, "get", "events", "-n", namespace, "-o", "json"], command_timeout)
    events = []
    for snap in snapshots:
        events.append({
            "timestamp": snap.get("checked_at"),
            "event": "deployment_status_poll",
            "ok": snap.get("ok"),
            "restart_sum": snap.get("restart_sum"),
            "pending_pods": snap.get("pending_pods"),
            "problems": snap.get("problems", []),
        })
    events.append({
        "timestamp": utc_text(),
        "event": final_event,
    })
    return {
        "generated_at": utc_text(),
        "namespace": namespace,
        "events": events,
        "kubernetes_events_capture": compact(k8s_events["capture"]),
        "kubernetes_events": k8s_events.get("payload", {}).get("items", []) if k8s_events["ok"] else [],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="trainticket-pilot")
    parser.add_argument("--run-dir", default="artifacts/trainticket_runs/latest_run")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--python", default="python")
    parser.add_argument("--initial-sleep-seconds", type=int, default=20)
    parser.add_argument("--poll-seconds", type=int, default=5)
    parser.add_argument("--stable-seconds", type=int, default=300)
    parser.add_argument("--timeout-seconds", type=int, default=900)
    parser.add_argument("--command-timeout-seconds", type=int, default=20)
    parser.add_argument("--api-timeout-seconds", type=int, default=20)
    parser.add_argument("--api-command-timeout-seconds", type=int, default=300)
    parser.add_argument("--skip-api-check", action="store_true")
    parser.add_argument("--allowlist", default="configs/trainticket_collection/workload_allowlist.yaml")
    parser.add_argument("--gateway-base-url", default="http://localhost:18888")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    report_path = run_dir / "stable_state_report.json"
    events_path = run_dir / "deployment_events.json"

    print(f"Initial sleep {args.initial_sleep_seconds}s before stable polling...")
    time.sleep(args.initial_sleep_seconds)

    started = time.monotonic()
    stable_since_monotonic = None
    stable_since_wall: datetime | None = None
    last_restart_sum = None
    snapshots = []

    while time.monotonic() - started < args.timeout_seconds:
        snap = snapshot_state(args.kubectl, args.namespace, args.command_timeout_seconds)
        snapshots.append(snap)
        restart_stable = last_restart_sum is None or snap["restart_sum"] == last_restart_sum
        if snap["ok"] and restart_stable:
            if stable_since_monotonic is None:
                stable_since_monotonic = time.monotonic()
                stable_since_wall = utc_now()
            stable_elapsed = time.monotonic() - stable_since_monotonic
            print(f"Stable candidate: {stable_elapsed:.1f}s/{args.stable_seconds}s, restart_sum={snap['restart_sum']}")
            if stable_elapsed >= args.stable_seconds:
                api = run_api_check(args, run_dir / "api_assertions.json")
                api_passed = bool(api["passed"])
                stable_text = stable_since_wall.isoformat() if stable_since_wall else utc_text()
                report = {
                    "generated_at": utc_text(),
                    "namespace": args.namespace,
                    "deployment_success": True,
                    "deployment_failed": False,
                    "post_change_state": "stable_success",
                    "readiness_success": True,
                    "service_registered": True,
                    "api_check_success": api_passed,
                    "stable_state_start_time": stable_text,
                    "deployment_complete_time": stable_text,
                    "stable_required_seconds": args.stable_seconds,
                    "timeout_seconds": args.timeout_seconds,
                    "api_check": api,
                    "resource_headroom": resource_snapshot(args.kubectl, args.namespace, args.command_timeout_seconds),
                    "snapshots": snapshots,
                }
                write_report(report_path, report)
                write_report(
                    events_path,
                    deployment_events_payload(
                        args.kubectl,
                        args.namespace,
                        args.command_timeout_seconds,
                        snapshots,
                        "rollout_completed" if api_passed else "rollout_stable_api_check_failed",
                    ),
                )
                if api_passed:
                    print(f"stable_state_start_time={stable_text}")
                else:
                    print("Stable infrastructure reached; API check failed and was recorded separately")
                return 0
        else:
            if not snap["ok"]:
                print("Not stable: " + "; ".join(snap["problems"]))
            elif not restart_stable:
                print(f"Not stable: restart sum changed from {last_restart_sum} to {snap['restart_sum']}")
            stable_since_monotonic = None
            stable_since_wall = None
        last_restart_sum = snap["restart_sum"]
        time.sleep(args.poll_seconds)

    report = {
        "generated_at": utc_text(),
        "namespace": args.namespace,
        "deployment_success": False,
        "deployment_failed": True,
        "readiness_success": False,
        "service_registered": False,
        "post_change_state": "deployment_failure",
        "timeout_seconds": args.timeout_seconds,
        "resource_headroom": resource_snapshot(args.kubectl, args.namespace, args.command_timeout_seconds),
        "snapshots": snapshots,
    }
    write_report(report_path, report)
    write_report(
        events_path,
        deployment_events_payload(
            args.kubectl,
            args.namespace,
            args.command_timeout_seconds,
            snapshots,
            "rollout_failed",
        ),
    )
    print(f"deployment_failure: stable state was not reached within {args.timeout_seconds}s")
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
