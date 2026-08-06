#!/usr/bin/env python3
"""Convert streamed raw Kubernetes logs into service-level logs.jsonl."""

from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from common import normalize_template


TIMESTAMP_RE = re.compile(r"^(?P<timestamp>\S+)\s+(?P<message>.*)$")
EMBEDDED_TS_RE = re.compile(r"\b\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:[.,]\d{3})?\b")
REPLICA_HASH_RE = re.compile(r"^(?P<app>.+)-[0-9a-f]{8,10}-[a-z0-9]{5}$")
STATEFUL_RE = re.compile(r"^(?P<app>.+)-\d+$")
JAVA_EXCEPTION_RE = re.compile(
    r"\b(?P<type>(?:[A-Za-z_$][\w$]*\.)*[A-Za-z_$][\w$]*(?:Exception|Error|Throwable|SQLException))\b"
)
STACK_CONTINUATION_RE = re.compile(
    r"^\s*(?:at\s+|Caused by:|Suppressed:|\.\.\.\s+\d+\s+more|"
    r"common frames omitted|Wrapped by:|Nested exception is:)"
)
ACCESS_RE = re.compile(
    r"^SCWARN_ACCESS\s+\[[^\]]+\]\s+\S+\s+"
    r"(?P<method>\S+)\s+(?P<uri>\S+)\s+(?P<status>\d{3})(?:\s+\S+)?"
)
PROTOCOL_VERSION = "scwarn-pilot-protocol-v0.6"
PARSER_VERSION = "materialize-jsonl-v0.8-normalized-templates-collector-artifact-filter"

COLLECTOR_ARTIFACT_PREFIXES = (
    "unable to retrieve container logs for containerd://",
)
COLLECTOR_ARTIFACT_EXACT = {
    "context canceled",
}
COLLECTOR_ARTIFACT_RE = re.compile(
    r'^container "[^"]+" in pod "[^"]+" is waiting to start: .+$'
)


def append_jsonl(path: Path, payload: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def parse_rfc3339(value: str | None) -> datetime | None:
    if not value:
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "." in text:
        head, tail = text.split(".", 1)
        offset = ""
        fraction = tail
        for marker in ("+", "-"):
            if marker in tail:
                fraction, offset_part = tail.split(marker, 1)
                offset = marker + offset_part
                break
        if len(fraction) > 6:
            fraction = fraction[:6]
        text = f"{head}.{fraction}{offset}"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def rfc3339_utc(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def run_bounds(run_dir: Path, start_arg: str | None, end_arg: str | None) -> tuple[datetime, datetime]:
    timeline = load_json(run_dir / "phase_timeline.json")
    manifest = load_json(run_dir / "run_manifest.json")
    start_text = start_arg or timeline.get("run_start_time") or manifest.get("run_start_time")
    end_text = (
        end_arg
        or timeline.get("run_end_time")
        or timeline.get("collection_end_time")
        or timeline.get("observation_end_time")
        or manifest.get("run_end_time")
    )
    start = parse_rfc3339(start_text)
    end = parse_rfc3339(end_text)
    if not start or not end:
        raise SystemExit(
            "Run log materialization requires run_start_time and run_end_time. "
            "Pass --run_start_time/--run_end_time or provide phase_timeline.json."
        )
    if end < start:
        raise SystemExit("run_end_time is earlier than run_start_time")
    return start, end


def load_pod_labels(run_dir: Path) -> dict[str, dict]:
    labels = {}
    lifecycle = run_dir / "pod_lifecycle.jsonl"
    if not lifecycle.exists():
        return labels
    with lifecycle.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") == "pod_snapshot" and event.get("pod"):
                labels[event["pod"]] = event.get("labels") or {}
    return labels


def derive_service(pod: str, labels: dict[str, dict]) -> str:
    app = (labels.get(pod) or {}).get("app")
    if app:
        return app
    match = REPLICA_HASH_RE.match(pod)
    if match:
        return match.group("app")
    match = STATEFUL_RE.match(pod)
    if match:
        return match.group("app")
    return pod


def classify_log_source(service: str, message: str) -> str:
    if message.startswith("SCWARN_ACCESS "):
        return "access"
    if service.startswith("ts-"):
        return "application_native"
    return "infrastructure"


def exception_type(message: str) -> str | None:
    match = JAVA_EXCEPTION_RE.search(message or "")
    return match.group("type") if match else None


def is_stack_continuation(message: str) -> bool:
    return bool(STACK_CONTINUATION_RE.match(message or ""))


def should_attach_to_pending(pending: dict | None, message: str) -> bool:
    if pending is None:
        return False
    if pending.get("log_source") == "access":
        return False
    if is_stack_continuation(message):
        return True
    if pending.get("exception_type") and exception_type(message):
        return True
    if "ERROR" in str(pending.get("message") or "") and exception_type(message):
        return True
    return False


def build_template(message: str, log_source: str, exc_type: str | None, exc_header: str | None) -> str:
    if log_source == "access":
        match = ACCESS_RE.match(message or "")
        if match:
            uri = normalize_template(match.group("uri"))
            return f"SCWARN_ACCESS {match.group('method')} {uri} {match.group('status')}"
        return normalize_template(message)
    if exc_type:
        header = exc_header or message
        return f"{exc_type}: {normalize_template(header)}"
    return normalize_template(message)


def parse_filename(path: Path) -> tuple[str, str]:
    stem = path.stem
    if "__" in stem:
        pod, container = stem.split("__", 1)
    else:
        pod, container = stem, ""
    container = container.split("__restart_", 1)[0]
    return pod, container


def iter_log_files(run_dir: Path):
    for category in ("current", "previous"):
        root = run_dir / "raw_logs" / category
        if not root.exists():
            continue
        for path in sorted(root.glob("*.log")):
            yield category, path


def is_collector_artifact_line(line: str) -> bool:
    text = (line or "").strip()
    return text in COLLECTOR_ARTIFACT_EXACT or any(
        text.startswith(prefix) for prefix in COLLECTOR_ARTIFACT_PREFIXES
    ) or bool(
        COLLECTOR_ARTIFACT_RE.match(text)
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--run_start_time", default=None)
    parser.add_argument("--run_end_time", default=None)
    parser.add_argument("--summary_output", default=None)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output = Path(args.output) if args.output else run_dir / "logs.jsonl"
    summary_output = Path(args.summary_output) if args.summary_output else run_dir / "collection_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    start_bound, end_bound = run_bounds(run_dir, args.run_start_time, args.run_end_time)
    labels = load_pod_labels(run_dir)
    collector_state = load_json(run_dir / "collector_state.json")
    per_pod: dict[str, dict] = {}
    count = 0
    parse_error_count = 0
    dropped_before = 0
    dropped_after = 0
    raw_line_count_total = 0
    empty_message_count = 0
    empty_message_attached_count = 0
    stack_trace_line_count = 0
    multiline_merged_event_count = 0
    collector_artifact_line_count = 0
    first_log_time: datetime | None = None
    last_log_time: datetime | None = None
    for category, path in iter_log_files(run_dir):
        pod, container = parse_filename(path)
        service = derive_service(pod, labels)
        pod_summary = per_pod.setdefault(pod, {
            "pod": pod,
            "service": service,
            "current_log_count": 0,
            "previous_log_count": 0,
            "parse_error_count": 0,
            "collector_artifact_line_count": 0,
            "dropped_before_run_count": 0,
            "dropped_after_run_count": 0,
            "first_log_time": None,
            "last_log_time": None,
            "containers": {},
        })
        container_summary = pod_summary["containers"].setdefault(container, {
            "container": container,
            "current_log_count": 0,
            "previous_log_count": 0,
            "parse_error_count": 0,
            "collector_artifact_line_count": 0,
            "raw_line_count": 0,
            "empty_message_count": 0,
            "stack_trace_line_count": 0,
            "multiline_merged_event_count": 0,
            "first_log_time": None,
            "last_log_time": None,
        })

        pending: dict | None = None

        def emit_pending() -> None:
            nonlocal pending
            nonlocal count, first_log_time, last_log_time, multiline_merged_event_count
            if pending is None:
                return
            event = dict(pending)
            ts = event.pop("_dt")
            if event.get("raw_line_count", 1) > 1:
                multiline_merged_event_count += 1
                container_summary["multiline_merged_event_count"] += 1
            event["event_template"] = build_template(
                str(event.get("message") or ""),
                str(event.get("log_source") or ""),
                event.get("exception_type"),
                event.get("exception_header"),
            )
            if not event.get("stack_trace"):
                event.pop("stack_trace", None)
            append_jsonl(output, event)
            count += 1
            if category == "previous":
                pod_summary["previous_log_count"] += 1
                container_summary["previous_log_count"] += 1
            else:
                pod_summary["current_log_count"] += 1
                container_summary["current_log_count"] += 1
            first_log_time = ts if first_log_time is None else min(first_log_time, ts)
            last_log_time = ts if last_log_time is None else max(last_log_time, ts)
            pod_first = parse_rfc3339(pod_summary["first_log_time"])
            pod_last = parse_rfc3339(pod_summary["last_log_time"])
            pod_summary["first_log_time"] = rfc3339_utc(ts if pod_first is None else min(pod_first, ts))
            pod_summary["last_log_time"] = rfc3339_utc(ts if pod_last is None else max(pod_last, ts))
            c_first = parse_rfc3339(container_summary["first_log_time"])
            c_last = parse_rfc3339(container_summary["last_log_time"])
            container_summary["first_log_time"] = rfc3339_utc(ts if c_first is None else min(c_first, ts))
            container_summary["last_log_time"] = rfc3339_utc(ts if c_last is None else max(c_last, ts))
            pending = None

        def start_pending(ts: datetime, message: str, line_number: int) -> None:
            nonlocal pending
            ts_text = rfc3339_utc(ts)
            log_source = classify_log_source(service, message)
            embedded_match = EMBEDDED_TS_RE.search(message)
            exc_type = exception_type(message)
            pending = {
                "_dt": ts,
                "timestamp": ts_text,
                "timestamp_utc": ts_text,
                "embedded_timestamp": embedded_match.group(0) if embedded_match else None,
                "message": message,
                "log_source": log_source,
                "service": service,
                "pod": pod,
                "container": container,
                "log_category": category,
                "source_file": str(path),
                "source_line": line_number,
                "source_line_end": line_number,
                "raw_line_count": 1,
                "repeat_count": 1,
                "exception_type": exc_type,
                "exception_header": message if exc_type else None,
                "stack_trace": [],
                "logical_event_kind": "java_exception" if exc_type else "single_line",
                "protocol_version": PROTOCOL_VERSION,
                "parser_version": PARSER_VERSION,
            }

        def attach_to_pending(message: str, line_number: int) -> None:
            nonlocal pending, stack_trace_line_count
            if pending is None:
                return
            pending["raw_line_count"] = int(pending.get("raw_line_count") or 1) + 1
            pending["source_line_end"] = line_number
            pending["logical_event_kind"] = "java_exception"
            exc_type = exception_type(message)
            if exc_type and not pending.get("exception_type"):
                pending["exception_type"] = exc_type
            if exc_type and not pending.get("exception_header"):
                pending["exception_header"] = message
            pending.setdefault("stack_trace", []).append(message)
            stack_trace_line_count += 1
            container_summary["stack_trace_line_count"] += 1

        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line_number, line in enumerate(f, 1):
                raw_line_count_total += 1
                container_summary["raw_line_count"] += 1
                line = line.rstrip("\r\n")
                if not line:
                    empty_message_count += 1
                    container_summary["empty_message_count"] += 1
                    continue
                if is_collector_artifact_line(line):
                    collector_artifact_line_count += 1
                    pod_summary["collector_artifact_line_count"] += 1
                    container_summary["collector_artifact_line_count"] += 1
                    continue
                match = TIMESTAMP_RE.match(line)
                if not match:
                    if pending is not None and pending.get("logical_event_kind") == "java_exception":
                        attach_to_pending(line, line_number)
                        continue
                    parse_error_count += 1
                    pod_summary["parse_error_count"] += 1
                    container_summary["parse_error_count"] += 1
                    continue
                timestamp = match.group("timestamp")
                ts = parse_rfc3339(timestamp)
                if ts is None:
                    parse_error_count += 1
                    pod_summary["parse_error_count"] += 1
                    container_summary["parse_error_count"] += 1
                    continue
                if ts < start_bound:
                    emit_pending()
                    dropped_before += 1
                    pod_summary["dropped_before_run_count"] += 1
                    continue
                if ts > end_bound:
                    emit_pending()
                    dropped_after += 1
                    pod_summary["dropped_after_run_count"] += 1
                    continue
                message = match.group("message")
                if not message.strip():
                    empty_message_count += 1
                    container_summary["empty_message_count"] += 1
                    if pending is not None and pending.get("logical_event_kind") == "java_exception":
                        attach_to_pending("", line_number)
                        empty_message_attached_count += 1
                    continue
                if should_attach_to_pending(pending, message):
                    attach_to_pending(message, line_number)
                else:
                    emit_pending()
                    start_pending(ts, message, line_number)
        emit_pending()

    generated_at = rfc3339_utc(datetime.now(timezone.utc))
    summary = {
        "generated_at": generated_at,
        "run_dir": str(run_dir),
        "run_start_time": rfc3339_utc(start_bound),
        "run_end_time": rfc3339_utc(end_bound),
        "output": str(output),
        "events": count,
        "protocol_version": PROTOCOL_VERSION,
        "parser_version": PARSER_VERSION,
        "raw_line_count": raw_line_count_total,
        "empty_message_count": empty_message_count,
        "empty_message_attached_count": empty_message_attached_count,
        "stack_trace_line_count": stack_trace_line_count,
        "multiline_merged_event_count": multiline_merged_event_count,
        "collector_artifact_line_count": collector_artifact_line_count,
        "parse_error_count": parse_error_count,
        "dropped_before_run_count": dropped_before,
        "dropped_after_run_count": dropped_after,
        "first_log_time": rfc3339_utc(first_log_time) if first_log_time else None,
        "last_log_time": rfc3339_utc(last_log_time) if last_log_time else None,
        "collector_stop_reason": collector_state.get("stop_reason"),
        "collector_complete": collector_state.get("collection_complete"),
        "collector_streams": collector_state.get("streams", []),
        "pods": sorted(per_pod.values(), key=lambda item: item["pod"]),
    }
    summary_output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    append_jsonl(run_dir / "pod_lifecycle.jsonl", {
        "timestamp": generated_at,
        "event": "archive_complete",
        "output": str(output),
        "summary_output": str(summary_output),
        "events": count,
        "raw_line_count": raw_line_count_total,
        "empty_message_count": empty_message_count,
        "stack_trace_line_count": stack_trace_line_count,
        "multiline_merged_event_count": multiline_merged_event_count,
        "collector_artifact_line_count": collector_artifact_line_count,
        "parse_error_count": parse_error_count,
        "dropped_before_run_count": dropped_before,
        "dropped_after_run_count": dropped_after,
    })
    print(json.dumps({
        "output": str(output),
        "events": count,
        "raw_line_count": raw_line_count_total,
        "multiline_merged_event_count": multiline_merged_event_count,
        "collector_artifact_line_count": collector_artifact_line_count,
        "run_start_time": summary["run_start_time"],
        "run_end_time": summary["run_end_time"],
        "dropped_before_run_count": dropped_before,
        "dropped_after_run_count": dropped_after,
        "summary_output": str(summary_output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
