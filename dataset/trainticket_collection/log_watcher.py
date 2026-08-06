#!/usr/bin/env python3
"""Stream Kubernetes logs during a run and archive lifecycle evidence."""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


PROTOCOL_VERSION = "scwarn-pilot-protocol-v0.6"
PARSER_VERSION = "materialize-jsonl-v0.8-normalized-templates-collector-artifact-filter"
COLLECTOR_VERSION = "log-watcher-v0.6-component-fields"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_rfc3339(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def rfc3339_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def event_timestamp(item: dict) -> tuple[str | None, str]:
    for field in ("eventTime", "lastTimestamp", "firstTimestamp"):
        value = item.get(field)
        if value:
            return value, field
    created = item.get("metadata", {}).get("creationTimestamp")
    if created:
        return created, "metadata.creationTimestamp"
    return None, "unparseable"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def run_json(command: list[str], timeout: int = 20) -> dict:
    proc = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if proc.returncode != 0:
        return {"ok": False, "stdout": proc.stdout, "stderr": proc.stderr, "items": []}
    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError:
        return {"ok": False, "stdout": proc.stdout, "stderr": "json parse failed", "items": []}
    return {"ok": True, "payload": payload, "items": payload.get("items", [])}


def append_jsonl(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


class LogWatcher:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.run_dir = Path(args.run_dir)
        self.current_dir = self.run_dir / "raw_logs" / "current"
        self.previous_dir = self.run_dir / "raw_logs" / "previous"
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.previous_dir.mkdir(parents=True, exist_ok=True)
        self.lifecycle_path = self.run_dir / "pod_lifecycle.jsonl"
        self.events_path = self.run_dir / "kubernetes_events.jsonl"
        self.phase_timeline_path = self.run_dir / "phase_timeline.json"
        self.run_manifest_path = self.run_dir / "run_manifest.json"
        self.collection_state_path = self.run_dir / "collector_state.json"
        self.processes: dict[tuple[str, str], subprocess.Popen] = {}
        self.streams: dict[str, dict] = {}
        self.previous_captured: set[tuple[str, str, int]] = set()
        self.seen_events: set[str] = set()
        self.stop_requested = False
        self.stop_reason: str | None = None
        self.run_start_dt = parse_rfc3339(args.run_start_time) or datetime.now(timezone.utc)
        self.run_start_time = rfc3339_utc(self.run_start_dt)
        self.collection_start_time = utc_now()
        self.collection_end_time: str | None = None

    def ensure_run_metadata(self) -> None:
        run_id = self.args.run_id or self.run_dir.name
        if self.run_manifest_path.exists():
            try:
                manifest = json.loads(self.run_manifest_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                manifest = {}
        else:
            manifest = {}
        manifest.update({
            "run_id": run_id,
            "run_type": self.args.run_type,
            "scenario_id": self.args.scenario_id,
            "system_name": "Train-Ticket",
            "service_subset_version": "TT-Core-16-accesslog-v1",
            "namespace": self.args.namespace,
            "cluster_type": self.args.cluster_type,
            "benchmark_label": self.args.benchmark_label,
            "semantic_label": self.args.semantic_label,
            "change_family_id": self.args.change_family_id,
            "implementation_id": self.args.implementation_id,
            "component_id": self.args.component_id,
            "change_target_component_id": self.args.change_target_component_id or self.args.component_id,
            "affected_component_ids": self.args.affected_component_ids,
            "oracle_component_ids": self.args.oracle_component_ids,
            "batch_id": self.args.batch_id,
            "pair_block_id": self.args.pair_block_id,
            "batch_position": self.args.batch_position,
            "preceding_scenario": self.args.preceding_scenario,
            "matched_workload_seed": self.args.matched_workload_seed,
            "hours_since_cluster_start": self.args.hours_since_cluster_start,
            "workload_profile_id": self.args.workload_profile_id,
            "workload_seed": self.args.workload_seed,
            "run_start_time": self.run_start_time,
            "protocol_version": PROTOCOL_VERSION,
            "parser_version": PARSER_VERSION,
            "collector_version": COLLECTOR_VERSION,
        })
        manifest.setdefault("git_commit", "")
        manifest.setdefault("image_digests", {})
        manifest.setdefault("config_patch_id", None)
        write_json(self.run_manifest_path, manifest)

        if self.phase_timeline_path.exists():
            try:
                timeline = json.loads(self.phase_timeline_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                timeline = {}
        else:
            timeline = {}
        for key in (
            "pre_change_start",
            "change_start_time",
            "deployment_complete_time",
            "stable_state_start_time",
            "failure_trigger_time",
            "observation_end_time",
            "cleanup_start_time",
            "cleanup_complete_time",
        ):
            timeline.setdefault(key, None)
        timeline["run_start_time"] = self.run_start_time
        timeline["collection_start_time"] = self.collection_start_time
        timeline.setdefault("run_end_time", None)
        write_json(self.phase_timeline_path, timeline)

    def update_collection_end(self) -> None:
        self.collection_end_time = utc_now()
        if self.phase_timeline_path.exists():
            try:
                timeline = json.loads(self.phase_timeline_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                timeline = {}
        else:
            timeline = {}
        timeline["run_start_time"] = self.run_start_time
        timeline.setdefault("collection_start_time", self.collection_start_time)
        timeline["collection_end_time"] = self.collection_end_time
        timeline["run_end_time"] = self.collection_end_time
        timeline.setdefault("observation_end_time", self.collection_end_time)
        write_json(self.phase_timeline_path, timeline)
        if self.run_manifest_path.exists():
            try:
                manifest = json.loads(self.run_manifest_path.read_text(encoding="utf-8-sig"))
            except json.JSONDecodeError:
                manifest = {}
        else:
            manifest = {}
        manifest["run_end_time"] = self.collection_end_time
        write_json(self.run_manifest_path, manifest)

    def stream_since_time(self, pod_created_at: str | None) -> str:
        pod_created_dt = parse_rfc3339(pod_created_at)
        if pod_created_dt and pod_created_dt > self.run_start_dt:
            return rfc3339_utc(pod_created_dt)
        return self.run_start_time

    def stop(self, signum=None, *_args) -> None:
        self.stop_requested = True
        if self.stop_reason is None:
            self.stop_reason = f"signal_{signum}" if signum is not None else "signal"

    def start_log_stream(self, pod: str, container: str, pod_created_at: str | None) -> None:
        key = (pod, container)
        proc = self.processes.get(key)
        if proc and proc.poll() is None:
            return
        since_time = self.stream_since_time(pod_created_at)
        outfile = self.current_dir / f"{safe_name(pod)}__{safe_name(container)}.log"
        handle = outfile.open("ab")
        command = [
            self.args.kubectl,
            "logs",
            "-n",
            self.args.namespace,
            pod,
            "-c",
            container,
            "--timestamps",
            "--since-time",
            since_time,
            "--follow",
        ]
        proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT)
        proc._scwarn_handle = handle  # type: ignore[attr-defined]
        self.processes[key] = proc
        append_jsonl(self.lifecycle_path, {
            "timestamp": utc_now(),
            "event": "log_stream_started",
            "pod": pod,
            "container": container,
            "pod_created_at": pod_created_at,
            "since_time": since_time,
            "file": str(outfile),
        })
        self.streams[f"{pod}/{container}/current"] = {
            "pod": pod,
            "container": container,
            "log_category": "current",
            "stream_started_at": utc_now(),
            "pod_created_at": pod_created_at,
            "since_time": since_time,
            "file": str(outfile),
        }

    def capture_previous(self, pod: str, container: str, restart_count: int, pod_created_at: str | None) -> None:
        if restart_count <= 0:
            return
        key = (pod, container, restart_count)
        if key in self.previous_captured:
            return
        since_time = self.stream_since_time(pod_created_at)
        outfile = self.previous_dir / f"{safe_name(pod)}__{safe_name(container)}__restart_{restart_count}.log"
        command = [
            self.args.kubectl,
            "logs",
            "-n",
            self.args.namespace,
            pod,
            "-c",
            container,
            "--timestamps",
            "--since-time",
            since_time,
            "--previous",
        ]
        proc = subprocess.run(command, text=True, capture_output=True, timeout=self.args.command_timeout_seconds)
        outfile.write_text(proc.stdout + proc.stderr, encoding="utf-8", errors="replace")
        self.previous_captured.add(key)
        append_jsonl(self.lifecycle_path, {
            "timestamp": utc_now(),
            "event": "previous_log_captured",
            "pod": pod,
            "container": container,
            "restart_count": restart_count,
            "pod_created_at": pod_created_at,
            "since_time": since_time,
            "exit_code": proc.returncode,
            "file": str(outfile),
        })
        self.streams[f"{pod}/{container}/previous/{restart_count}"] = {
            "pod": pod,
            "container": container,
            "log_category": "previous",
            "stream_started_at": utc_now(),
            "stream_stopped_at": utc_now(),
            "pod_created_at": pod_created_at,
            "since_time": since_time,
            "restart_count": restart_count,
            "exit_code": proc.returncode,
            "file": str(outfile),
        }

    def poll_pods(self) -> None:
        payload = run_json([
            self.args.kubectl,
            "get",
            "pods",
            "-n",
            self.args.namespace,
            "-o",
            "json",
        ], timeout=self.args.command_timeout_seconds)
        append_jsonl(self.lifecycle_path, {
            "timestamp": utc_now(),
            "event": "pod_poll",
            "ok": payload["ok"],
            "stderr": payload.get("stderr", ""),
        })
        if not payload["ok"]:
            return
        for pod in payload["items"]:
            pod_name = pod.get("metadata", {}).get("name", "")
            pod_created_at = pod.get("metadata", {}).get("creationTimestamp")
            status = pod.get("status", {})
            append_jsonl(self.lifecycle_path, {
                "timestamp": utc_now(),
                "event": "pod_snapshot",
                "pod": pod_name,
                "pod_created_at": pod_created_at,
                "labels": pod.get("metadata", {}).get("labels", {}),
                "phase": status.get("phase"),
                "container_statuses": status.get("containerStatuses", []),
            })
            for container in pod.get("spec", {}).get("containers", []):
                name = container.get("name")
                if name:
                    self.start_log_stream(pod_name, name, pod_created_at)
            for cs in status.get("containerStatuses", []) or []:
                self.capture_previous(pod_name, cs.get("name", ""), int(cs.get("restartCount") or 0), pod_created_at)

    def poll_events(self) -> None:
        payload = run_json([
            self.args.kubectl,
            "get",
            "events",
            "-n",
            self.args.namespace,
            "-o",
            "json",
        ], timeout=self.args.command_timeout_seconds)
        if not payload["ok"]:
            append_jsonl(self.events_path, {
                "timestamp": utc_now(),
                "event": "event_poll_failed",
                "stderr": payload.get("stderr", ""),
            })
            return
        for item in payload["items"]:
            meta = item.get("metadata", {})
            uid = meta.get("uid") or f"{meta.get('name')}:{item.get('count')}"
            count = item.get("count")
            key = f"{uid}:{count}"
            if key in self.seen_events:
                continue
            self.seen_events.add(key)
            event_time_text, event_time_source = event_timestamp(item)
            event_dt = parse_rfc3339(event_time_text)
            if event_dt is None:
                time_class = "historical_snapshot"
                within_run_time = False
            elif event_dt < self.run_start_dt:
                time_class = "historical_snapshot"
                within_run_time = False
            else:
                time_class = "within_run_or_open_interval"
                within_run_time = True
            append_jsonl(self.events_path, {
                "timestamp": utc_now(),
                "event": "kubernetes_event",
                "event_time": rfc3339_utc(event_dt) if event_dt else None,
                "event_time_source": event_time_source,
                "event_time_class": time_class,
                "within_run_time": within_run_time,
                "raw": item,
            })

    def reap_processes(self) -> None:
        for key, proc in list(self.processes.items()):
            if proc.poll() is not None:
                handle = getattr(proc, "_scwarn_handle", None)
                if handle:
                    handle.close()
                append_jsonl(self.lifecycle_path, {
                    "timestamp": utc_now(),
                    "event": "log_stream_exited",
                    "pod": key[0],
                    "container": key[1],
                    "exit_code": proc.returncode,
                    "termination_reason": "process_exited_before_collector_stop",
                    "expected_termination": False,
                })
                stream_key = f"{key[0]}/{key[1]}/current"
                if stream_key in self.streams:
                    self.streams[stream_key]["stream_stopped_at"] = utc_now()
                    self.streams[stream_key]["exit_code"] = proc.returncode
                    self.streams[stream_key]["termination_reason"] = "process_exited_before_collector_stop"
                    self.streams[stream_key]["expected_termination"] = False
                del self.processes[key]

    def shutdown(self) -> None:
        for key, proc in list(self.processes.items()):
            was_running_at_shutdown = proc.poll() is None
            termination_method = "already_exited"
            if proc.poll() is None:
                proc.terminate()
                termination_method = "terminate"
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    termination_method = "kill"
                    proc.wait(timeout=5)
            handle = getattr(proc, "_scwarn_handle", None)
            if handle:
                handle.close()
            expected_termination = bool(was_running_at_shutdown and self.stop_requested)
            append_jsonl(self.lifecycle_path, {
                "timestamp": utc_now(),
                "event": "log_stream_stopped",
                "pod": key[0],
                "container": key[1],
                "exit_code": proc.poll(),
                "stop_reason": self.stop_reason,
                "termination_reason": "collector_shutdown" if was_running_at_shutdown else "process_already_exited",
                "termination_method": termination_method,
                "expected_termination": expected_termination,
            })
            stream_key = f"{key[0]}/{key[1]}/current"
            if stream_key in self.streams:
                self.streams[stream_key]["stream_stopped_at"] = utc_now()
                self.streams[stream_key]["exit_code"] = proc.poll()
                self.streams[stream_key]["stop_reason"] = self.stop_reason
                self.streams[stream_key]["termination_reason"] = "collector_shutdown" if was_running_at_shutdown else "process_already_exited"
                self.streams[stream_key]["termination_method"] = termination_method
                self.streams[stream_key]["expected_termination"] = expected_termination

    def loop(self) -> None:
        self.ensure_run_metadata()
        append_jsonl(self.lifecycle_path, {
            "timestamp": utc_now(),
            "event": "watcher_started",
            "pid": os.getpid(),
            "namespace": self.args.namespace,
            "run_start_time": self.run_start_time,
            "collection_start_time": self.collection_start_time,
        })
        signal.signal(signal.SIGINT, self.stop)
        signal.signal(signal.SIGTERM, self.stop)
        try:
            started = time.monotonic()
            while not self.stop_requested:
                if self.args.duration_seconds and (time.monotonic() - started) >= self.args.duration_seconds:
                    self.stop_requested = True
                    self.stop_reason = "duration_elapsed"
                    break
                self.poll_pods()
                self.poll_events()
                self.reap_processes()
                time.sleep(self.args.poll_seconds)
        finally:
            if self.stop_reason is None:
                self.stop_reason = "loop_exited"
            self.shutdown()
            self.update_collection_end()
            write_json(self.collection_state_path, {
                "run_start_time": self.run_start_time,
                "collection_start_time": self.collection_start_time,
                "collection_end_time": self.collection_end_time,
                "stop_reason": self.stop_reason,
                "collection_complete": True,
                "streams": list(self.streams.values()),
            })
            append_jsonl(self.lifecycle_path, {
                "timestamp": utc_now(),
                "event": "watcher_stopped",
                "pid": os.getpid(),
                "run_end_time": self.collection_end_time,
                "stop_reason": self.stop_reason,
                "collection_complete": True,
            })


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--namespace", default="trainticket-pilot")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--kubectl", default="kubectl")
    parser.add_argument("--poll_seconds", type=int, default=5)
    parser.add_argument("--command_timeout_seconds", type=int, default=20)
    parser.add_argument("--duration_seconds", type=int, default=0)
    parser.add_argument("--run_start_time", default=None)
    parser.add_argument("--run_id", default=None)
    parser.add_argument("--run_type", default="unspecified")
    parser.add_argument("--cluster_type", default="Docker Desktop Kubernetes")
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
    parser.add_argument("--workload_profile_id", default="tt_core_16_allowlist_v1")
    parser.add_argument("--workload_seed", default=None)
    args = parser.parse_args()
    watcher = LogWatcher(args)
    watcher.loop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
