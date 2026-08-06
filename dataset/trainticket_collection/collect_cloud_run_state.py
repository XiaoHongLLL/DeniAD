#!/usr/bin/env python3
"""Capture cloud Kubernetes state for SCWarn control-run quality gates."""

from __future__ import annotations

import argparse
import json
import subprocess
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run_json(command: list[str], timeout: int) -> dict:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "items": []}
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr.strip(), "stdout": proc.stdout, "items": []}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "json parse failed", "stdout": proc.stdout[:1000], "items": []}
    return {"ok": True, "payload": payload, "items": payload.get("items", [])}


def run_text(command: list[str], timeout: int) -> dict:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
        return {
            "ok": proc.returncode == 0,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "exit_code": 124, "stdout": "", "stderr": "timeout"}


def node_ready(node: dict) -> bool:
    for cond in node.get("status", {}).get("conditions") or []:
        if cond.get("type") == "Ready":
            return cond.get("status") == "True"
    return False


def pod_ready(pod: dict) -> bool:
    if pod.get("status", {}).get("phase") != "Running":
        return False
    for cond in pod.get("status", {}).get("conditions") or []:
        if cond.get("type") == "Ready":
            return cond.get("status") == "True"
    return False


def owner_kind_name(pod: dict) -> str:
    owners = pod.get("metadata", {}).get("ownerReferences") or []
    if not owners:
        return ""
    owner = owners[0]
    return f"{owner.get('kind','')}/{owner.get('name','')}"


def deployment_ready(item: dict) -> bool:
    desired = int((item.get("spec") or {}).get("replicas") or 1)
    status = item.get("status") or {}
    generation = (item.get("metadata") or {}).get("generation")
    return (
        status.get("observedGeneration") == generation
        and int(status.get("updatedReplicas") or 0) == desired
        and int(status.get("availableReplicas") or 0) == desired
        and int(status.get("unavailableReplicas") or 0) == 0
    )


def statefulset_ready(item: dict) -> bool:
    desired = int((item.get("spec") or {}).get("replicas") or 1)
    status = item.get("status") or {}
    generation = (item.get("metadata") or {}).get("generation")
    return status.get("observedGeneration") == generation and int(status.get("readyReplicas") or 0) == desired


def http_probe(base_url: str, timeout: int) -> dict:
    url = base_url.rstrip("/") + "/api/v1/auth/hello"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            body = response.read(200).decode("utf-8", errors="replace")
            return {"ok": response.status == 200 and "hello" in body, "status": response.status, "body": body}
    except (urllib.error.URLError, TimeoutError) as exc:
        return {"ok": False, "error": str(exc), "url": url}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="trainticket-pilot")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--base-url", default="")
    parser.add_argument("--target-deployment", default="")
    parser.add_argument("--output", default=None)
    parser.add_argument("--timeout-seconds", type=int, default=25)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / f"cloud_state_{args.phase}.json"
    output.parent.mkdir(parents=True, exist_ok=True)

    nodes = run_json([args.kubectl, "get", "nodes", "-o", "json"], args.timeout_seconds)
    pods = run_json([args.kubectl, "get", "pods", "-n", args.namespace, "-o", "json"], args.timeout_seconds)
    deployments = run_json([args.kubectl, "get", "deployments", "-n", args.namespace, "-o", "json"], args.timeout_seconds)
    statefulsets = run_json([args.kubectl, "get", "statefulsets", "-n", args.namespace, "-o", "json"], args.timeout_seconds)
    endpoints = run_json([args.kubectl, "get", "endpoints", "-n", args.namespace, "-o", "json"], args.timeout_seconds)
    top_nodes = run_text([args.kubectl, "top", "nodes"], args.timeout_seconds)
    top_pods = run_text([args.kubectl, "top", "pods", "-n", args.namespace], args.timeout_seconds)

    node_rows = [
        {
            "name": item.get("metadata", {}).get("name", ""),
            "ready": node_ready(item),
            "internal_ip": next(
                (addr.get("address") for addr in item.get("status", {}).get("addresses") or [] if addr.get("type") == "InternalIP"),
                "",
            ),
            "kubelet_version": item.get("status", {}).get("nodeInfo", {}).get("kubeletVersion", ""),
            "container_runtime": item.get("status", {}).get("nodeInfo", {}).get("containerRuntimeVersion", ""),
        }
        for item in nodes.get("items", [])
    ]
    pod_rows = []
    restart_sum = 0
    containers_with_restarts = []
    for pod in pods.get("items", []):
        labels = pod.get("metadata", {}).get("labels") or {}
        service = labels.get("app") or labels.get("app.kubernetes.io/name") or pod.get("metadata", {}).get("name", "")
        restarts = 0
        images = []
        image_ids = []
        for cs in pod.get("status", {}).get("containerStatuses") or []:
            count = int(cs.get("restartCount") or 0)
            restarts += count
            restart_sum += count
            images.append(cs.get("image", ""))
            image_ids.append(cs.get("imageID", ""))
            if count > 0:
                containers_with_restarts.append({
                    "pod": pod.get("metadata", {}).get("name", ""),
                    "container": cs.get("name", ""),
                    "restart_count": count,
                })
        pod_rows.append({
            "name": pod.get("metadata", {}).get("name", ""),
            "uid": pod.get("metadata", {}).get("uid", ""),
            "service": service,
            "node": pod.get("spec", {}).get("nodeName", ""),
            "phase": pod.get("status", {}).get("phase", ""),
            "ready": pod_ready(pod),
            "owner": owner_kind_name(pod),
            "restart_count": restarts,
            "images": images,
            "image_ids": image_ids,
            "created_at": pod.get("metadata", {}).get("creationTimestamp", ""),
        })

    deployment_rows = [
        {
            "name": item.get("metadata", {}).get("name", ""),
            "ready": deployment_ready(item),
            "replicas": int((item.get("spec") or {}).get("replicas") or 1),
            "available": int((item.get("status") or {}).get("availableReplicas") or 0),
            "updated": int((item.get("status") or {}).get("updatedReplicas") or 0),
        }
        for item in deployments.get("items", [])
    ]
    statefulset_rows = [
        {
            "name": item.get("metadata", {}).get("name", ""),
            "ready": statefulset_ready(item),
            "replicas": int((item.get("spec") or {}).get("replicas") or 1),
            "ready_replicas": int((item.get("status") or {}).get("readyReplicas") or 0),
        }
        for item in statefulsets.get("items", [])
    ]

    target_pods = [
        item for item in pod_rows
        if args.target_deployment and item["service"] == args.target_deployment
    ]
    pass_gate = (
        nodes.get("ok")
        and pods.get("ok")
        and deployments.get("ok")
        and statefulsets.get("ok")
        and bool(node_rows)
        and all(item["ready"] for item in node_rows)
        and bool(pod_rows)
        and all(item["ready"] for item in pod_rows)
        and all(item["ready"] for item in deployment_rows)
        and all(item["ready"] for item in statefulset_rows)
    )

    payload = {
        "generated_at": utc_now(),
        "phase": args.phase,
        "namespace": args.namespace,
        "base_url": args.base_url,
        "target_deployment": args.target_deployment,
        "all_nodes_ready": all(item["ready"] for item in node_rows) if node_rows else False,
        "all_pods_running_ready": all(item["ready"] for item in pod_rows) if pod_rows else False,
        "deployments_available": all(item["ready"] for item in deployment_rows) if deployment_rows else False,
        "statefulsets_ready": all(item["ready"] for item in statefulset_rows) if statefulset_rows else False,
        "restart_sum": restart_sum,
        "containers_with_restarts": containers_with_restarts,
        "pod_node_mapping_recorded": bool(pod_rows) and all(item["node"] for item in pod_rows),
        "node_count": len(node_rows),
        "pod_count": len(pod_rows),
        "nodes": node_rows,
        "pods": pod_rows,
        "deployments": deployment_rows,
        "statefulsets": statefulset_rows,
        "target_pods": target_pods,
        "captures": {
            "nodes_ok": nodes.get("ok"),
            "pods_ok": pods.get("ok"),
            "deployments_ok": deployments.get("ok"),
            "statefulsets_ok": statefulsets.get("ok"),
            "endpoints_ok": endpoints.get("ok"),
            "top_nodes_ok": top_nodes.get("ok"),
            "top_pods_ok": top_pods.get("ok"),
        },
        "gateway_probe": http_probe(args.base_url, min(args.timeout_seconds, 15)) if args.base_url else {"skipped": True},
        "top_nodes": top_nodes,
        "top_pods": top_pods,
        "endpoints_item_count": len(endpoints.get("items", [])),
        "pass": bool(pass_gate),
    }
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "pass": payload["pass"],
        "node_count": payload["node_count"],
        "pod_count": payload["pod_count"],
        "restart_sum": payload["restart_sum"],
    }, ensure_ascii=False, indent=2))
    return 0 if payload["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
