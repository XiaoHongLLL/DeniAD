#!/usr/bin/env python3
"""Finalize run metadata, hashes, phase times, and image inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


ACCESS_LOG_CONFIG = {
    "SERVER_TOMCAT_ACCESSLOG_ENABLED": "true",
    "SERVER_TOMCAT_ACCESSLOG_DIRECTORY": "/dev",
    "SERVER_TOMCAT_ACCESSLOG_PREFIX": "stdout",
    "SERVER_TOMCAT_ACCESSLOG_SUFFIX": "",
    "SERVER_TOMCAT_ACCESSLOG_FILE_DATE_FORMAT": "",
    "SERVER_TOMCAT_ACCESSLOG_BUFFERED": "false",
    "SERVER_TOMCAT_ACCESSLOG_PATTERN": "SCWARN_ACCESS %t %a %m %U %s %D",
    "JAVA_TOOL_OPTIONS": "-Dreactor.netty.http.server.accessLogEnabled=true",
    "LOGGING_LEVEL_REACTOR_NETTY_HTTP_SERVER_ACCESSLOG": "INFO",
}

PROTOCOL_VERSION = "scwarn-pilot-protocol-v0.6"
PARSER_VERSION = "materialize-jsonl-v0.8-normalized-templates-collector-artifact-filter"
COLLECTOR_VERSION = "log-watcher-v0.6-component-fields"


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def first_oracle_call_time(run_dir: Path) -> str | None:
    path = run_dir / "oracle_calls.jsonl"
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            return item.get("request_started_at") or item.get("timestamp")
    return None


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def files_hash(root: Path, patterns: list[str]) -> dict[str, str]:
    hashes = {}
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            if path.is_file():
                digest = sha256_file(path)
                if digest:
                    hashes[str(path)] = digest
    return hashes


def run_json(command: list[str], timeout: int) -> dict:
    try:
        proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "items": []}
    if proc.returncode != 0:
        return {"ok": False, "error": proc.stderr, "items": []}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "error": "json parse failed", "items": []}
    return {"ok": True, "payload": payload, "items": payload.get("items", [])}


def image_inventory(kubectl: str, namespace: str) -> dict:
    payload = run_json([kubectl, "get", "pods", "-n", namespace, "-o", "json"], 30)
    services = {}
    for pod in payload.get("items", []):
        labels = pod.get("metadata", {}).get("labels", {}) or {}
        service = labels.get("app") or pod.get("metadata", {}).get("name")
        for status in pod.get("status", {}).get("containerStatuses") or []:
            services.setdefault(service, []).append({
                "pod": pod.get("metadata", {}).get("name"),
                "container": status.get("name"),
                "image": status.get("image"),
                "image_id": status.get("imageID"),
                "ready": status.get("ready"),
                "restart_count": status.get("restartCount"),
            })
    return {
        "capture_ok": payload.get("ok", False),
        "error": payload.get("error"),
        "services": dict(sorted(services.items())),
    }


def sync_phase_times(run_dir: Path, timeline: dict) -> dict:
    stable = read_json(run_dir / "stable_state_report.json")
    post_stable = read_json(run_dir / "post_change_stable_report.json")
    change = read_json(run_dir / "change_report.json")
    source = post_stable or stable
    if change.get("change_start_time"):
        timeline["change_start_time"] = change.get("change_start_time")
    if change.get("deployment_complete_time"):
        timeline["deployment_complete_time"] = change.get("deployment_complete_time")
    elif not timeline.get("deployment_complete_time") and source.get("deployment_complete_time"):
        timeline["deployment_complete_time"] = source.get("deployment_complete_time")
    if change.get("deployment_failed_time"):
        timeline["deployment_failed_time"] = change.get("deployment_failed_time")
    elif source.get("deployment_failed_time"):
        timeline["deployment_failed_time"] = source.get("deployment_failed_time")
    if source.get("post_change_state"):
        timeline["post_change_state"] = source.get("post_change_state")
    if source.get("stable_state_start_time"):
        timeline["stable_state_start_time"] = source.get("stable_state_start_time")
        if str(source.get("post_change_state") or "").lower() == "stable_success":
            timeline["post_change_observation_start_time"] = (
                first_oracle_call_time(run_dir) or source.get("stable_state_start_time")
            )
        else:
            timeline.setdefault("post_change_observation_start_time", source.get("stable_state_start_time"))
    elif timeline.get("deployment_failed_time"):
        timeline.setdefault("post_change_observation_start_time", timeline.get("deployment_failed_time"))
    if not timeline.get("observation_end_time"):
        collection = read_json(run_dir / "collection_summary.json")
        if collection.get("run_end_time"):
            timeline["observation_end_time"] = collection.get("run_end_time")
    return timeline


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--namespace", default="trainticket-pilot")
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--run_type", default=None)
    parser.add_argument("--scenario_id", default=None)
    parser.add_argument("--benchmark_label", default=None)
    parser.add_argument("--semantic_label", default=None)
    parser.add_argument("--change_family_id", default=None)
    parser.add_argument("--implementation_id", default=None)
    parser.add_argument("--component_id", default=None)
    parser.add_argument("--change_target_component_id", default=None)
    parser.add_argument("--affected_component_ids", default=None)
    parser.add_argument("--oracle_component_ids", default=None)
    parser.add_argument("--batch_id", default=None)
    parser.add_argument("--pair_block_id", default=None)
    parser.add_argument("--batch_position", default=None)
    parser.add_argument("--preceding_scenario", default=None)
    parser.add_argument("--matched_workload_seed", default=None)
    parser.add_argument("--hours_since_cluster_start", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    repo = Path.cwd()
    manifest_path = run_dir / "run_manifest.json"
    timeline_path = run_dir / "phase_timeline.json"
    manifest = read_json(manifest_path)
    timeline = sync_phase_times(run_dir, read_json(timeline_path))

    for key in (
        "run_type",
        "scenario_id",
        "benchmark_label",
        "semantic_label",
        "change_family_id",
        "implementation_id",
        "component_id",
        "change_target_component_id",
        "affected_component_ids",
        "oracle_component_ids",
        "batch_id",
        "pair_block_id",
        "batch_position",
        "preceding_scenario",
        "matched_workload_seed",
        "hours_since_cluster_start",
    ):
        value = getattr(args, key)
        if value is not None:
            manifest[key] = value

    tracked_files = {
        "workload_allowlist": repo / "configs/trainticket_collection/workload_allowlist.yaml",
        "workload_profiles": repo / "configs/trainticket_collection/workload_profiles.json",
        "scenario_catalog": repo / "configs/trainticket_collection/scenario_catalog.json",
        "deploy_yaml": repo / "external/train-ticket/deployment/kubernetes-manifests/quickstart-k8s/yamls/deploy.yaml",
        "svc_yaml": repo / "external/train-ticket/deployment/kubernetes-manifests/quickstart-k8s/yamls/svc.yaml",
    }
    artifact_hashes = {
        name: sha256_file(path)
        for name, path in tracked_files.items()
        if sha256_file(path)
    }
    artifact_hashes["access_log_config"] = sha256_text(json.dumps(ACCESS_LOG_CONFIG, sort_keys=True))
    artifact_hashes["scwarn_pilot_scripts"] = sha256_text(json.dumps(
        files_hash(repo, ["dataset/trainticket_collection/*.py"]),
        sort_keys=True,
    ))

    manifest["artifact_hashes"] = artifact_hashes
    manifest["access_log_config"] = ACCESS_LOG_CONFIG
    manifest["image_inventory"] = image_inventory(args.kubectl, args.namespace)
    manifest["service_subset_version"] = "TT-Core-16-accesslog-v1"
    manifest["protocol_version"] = PROTOCOL_VERSION
    manifest["parser_version"] = PARSER_VERSION
    manifest["collector_version"] = COLLECTOR_VERSION
    manifest["git_commit"] = manifest.get("git_commit") or ""
    if str(manifest.get("benchmark_label") or "").lower() == "baseline_normal":
        manifest["baseline_reference_start_time"] = timeline.get("pre_change_start") or timeline.get("run_start_time")
        manifest["baseline_reference_end_time"] = timeline.get("change_start_time")
        manifest["baseline_evaluation_start_time"] = (
            timeline.get("post_change_observation_start_time")
            or timeline.get("stable_state_start_time")
            or timeline.get("change_start_time")
        )
        manifest["baseline_evaluation_end_time"] = timeline.get("observation_end_time") or timeline.get("run_end_time")

    write_json(manifest_path, manifest)
    write_json(timeline_path, timeline)
    print(json.dumps({
        "run_manifest": str(manifest_path),
        "phase_timeline": str(timeline_path),
        "hash_count": len(artifact_hashes),
        "image_inventory_ok": manifest["image_inventory"].get("capture_ok"),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
