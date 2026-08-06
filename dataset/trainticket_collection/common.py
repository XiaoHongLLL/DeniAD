#!/usr/bin/env python3
"""Shared helpers for the SCWarn-style Train-Ticket pilot."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


TIME_FORMATS = (
    "%Y-%m-%d %H:%M:%S.%f",
    "%Y-%m-%d %H:%M:%S,%f",
    "%Y-%m-%d %H:%M:%S",
)

UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
HEX_RE = re.compile(r"\b[0-9a-fA-F]{12,}\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
SPACE_RE = re.compile(r"\s+")
ISO_FRACTION_RE = re.compile(r"^(?P<prefix>.+\.\d{6})\d+(?P<suffix>(?:[+-]\d{2}:\d{2})?)$")


def parse_time(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    iso = text.replace("Z", "+00:00")
    match = ISO_FRACTION_RE.match(iso)
    if match:
        iso = match.group("prefix") + match.group("suffix")
    try:
        dt = datetime.fromisoformat(iso)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    except ValueError:
        pass
    for fmt in TIME_FORMATS:
        try:
            return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    return None


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_manifest(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k: (v.strip() if isinstance(v, str) else v) for k, v in row.items()})
    return rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fields = []
        seen = set()
        for row in rows:
            for key in row:
                if key not in seen:
                    fields.append(key)
                    seen.add(key)
        fieldnames = fields
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_template(message: str) -> str:
    text = str(message or "").strip()
    text = UUID_RE.sub("<*>", text)
    text = IP_RE.sub("<*>", text)
    text = HEX_RE.sub("<*>", text)
    text = NUMBER_RE.sub("<*>", text)
    text = SPACE_RE.sub(" ", text)
    return text or "<EMPTY>"


def event_template(raw: dict) -> str:
    for key in ("event_template", "EventTemplate", "template", "Template"):
        value = raw.get(key)
        if value:
            return str(value).strip()
    for key in ("message", "content", "Content", "log", "raw"):
        value = raw.get(key)
        if value:
            return normalize_template(str(value))
    return "<EMPTY>"


def event_service(raw: dict) -> str:
    return str(raw.get("service") or raw.get("Service") or raw.get("component") or raw.get("pod") or "unknown")


def event_key(raw: dict) -> str:
    return f"{event_service(raw)}::{event_template(raw)}"


def iter_logs(run_dir: Path) -> list[dict]:
    path = run_dir / "logs.jsonl"
    if not path.exists():
        return []
    events = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line_id, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError:
                raw = {"message": line}
            timestamp = parse_time(raw.get("timestamp") or raw.get("@timestamp") or raw.get("time") or raw.get("DateTime"))
            if timestamp is None:
                continue
            raw["_timestamp"] = timestamp
            raw["_line_id"] = line_id
            raw["_template"] = event_template(raw)
            raw["_service"] = event_service(raw)
            raw["_event_key"] = f"{raw['_service']}::{raw['_template']}"
            events.append(raw)
    events.sort(key=lambda item: (item["_timestamp"], item["_line_id"]))
    return events


def row_run_dir(row: dict, raw_runs: Path) -> Path:
    evidence = row.get("external_evidence_dir") or row.get("external_evidence_path") or ""
    if evidence:
        path = Path(evidence)
        if path.is_absolute():
            return path
        if path.parts and path.parts[0] == "raw_runs":
            return raw_runs.parent / path
        return raw_runs / path
    return raw_runs / str(row["run_id"])


def benchmark_role(row: dict) -> str:
    return str(row.get("benchmark_role") or row.get("benchmark_label") or "").lower()


def time_bounds(row: dict, events: list[dict]) -> dict:
    first = events[0]["_timestamp"] if events else None
    last = events[-1]["_timestamp"] if events else None
    change_start = parse_time(row.get("change_start_time"))
    stable_start = parse_time(row.get("stable_state_start_time"))
    post_observation_start = parse_time(row.get("post_change_observation_start_time"))
    failure_start = (
        parse_time(row.get("failure_trigger_time"))
        or parse_time(row.get("deployment_failed_time"))
        or post_observation_start
    )
    observation_end = parse_time(row.get("observation_end_time")) or last
    baseline_reference_start = parse_time(row.get("baseline_reference_start_time")) or parse_time(row.get("pre_start_time")) or first
    baseline_reference_end = parse_time(row.get("baseline_reference_end_time"))
    baseline_evaluation_start = parse_time(row.get("baseline_evaluation_start_time"))
    baseline_evaluation_end = parse_time(row.get("baseline_evaluation_end_time")) or observation_end
    if change_start is None and first is not None and last is not None:
        change_start = first + (last - first) / 2.0
    if benchmark_role(row) == "baseline_normal":
        if baseline_reference_end is None:
            baseline_reference_end = change_start
        if baseline_evaluation_start is None:
            baseline_evaluation_start = stable_start or post_observation_start or change_start
    if stable_start is None:
        stable_start = post_observation_start or parse_time(row.get("deployment_complete_time")) or change_start
    if failure_start is None:
        failure_start = post_observation_start or stable_start or change_start
    return {
        "first": first,
        "last": last,
        "pre_start": parse_time(row.get("pre_start_time")) or first,
        "change_start": change_start,
        "stable_start": stable_start,
        "failure_start": failure_start,
        "post_observation_start": post_observation_start,
        "observation_end": observation_end,
        "baseline_reference_start": baseline_reference_start,
        "baseline_reference_end": baseline_reference_end,
        "baseline_evaluation_start": baseline_evaluation_start,
        "baseline_evaluation_end": baseline_evaluation_end,
    }


def filter_events(events: list[dict], start, end, include_end: bool = True) -> list[dict]:
    if start is None and end is None:
        return list(events)
    result = []
    for event in events:
        ts = event["_timestamp"]
        if start is not None and ts < start:
            continue
        if end is not None and (ts > end if include_end else ts >= end):
            continue
        result.append(event)
    return result


def pre_events(row: dict, events: list[dict]) -> list[dict]:
    bounds = time_bounds(row, events)
    if benchmark_role(row) == "baseline_normal" and bounds.get("baseline_reference_end") is not None:
        return filter_events(
            events,
            bounds["baseline_reference_start"],
            bounds["baseline_reference_end"],
            include_end=False,
        )
    return filter_events(events, bounds["pre_start"], bounds["change_start"], include_end=False)


def post_events(row: dict, events: list[dict]) -> list[dict]:
    bounds = time_bounds(row, events)
    if benchmark_role(row) == "baseline_normal" and bounds.get("baseline_evaluation_start") is not None:
        return filter_events(events, bounds["baseline_evaluation_start"], bounds["baseline_evaluation_end"])
    if bounds.get("post_observation_start") is not None:
        return filter_events(events, bounds["post_observation_start"], bounds["observation_end"])
    state = str(row.get("post_change_state") or "").lower()
    semantic = str(row.get("semantic_label") or row.get("oracle_semantic_label") or row.get("declared_semantic_label") or "").lower()
    if state in {"deployment_failure", "runtime_failure"} or semantic == "unexpected":
        start = bounds["failure_start"] or bounds["stable_start"] or bounds["change_start"]
    else:
        start = bounds["stable_start"] or bounds["change_start"]
    return filter_events(events, start, bounds["observation_end"])


def control_events(row: dict, events: list[dict]) -> list[dict]:
    bounds = time_bounds(row, events)
    if benchmark_role(row) == "baseline_normal" and bounds.get("baseline_evaluation_start") is not None:
        return pre_events(row, events) + post_events(row, events)
    if bounds["change_start"] is None:
        return list(events)
    left = filter_events(events, bounds["pre_start"], bounds["change_start"])
    right = filter_events(events, bounds["stable_start"] or bounds["change_start"], bounds["observation_end"])
    return left + right


def sliding_time_windows(events: list[dict], length_seconds: float, step_seconds: float) -> list[list[dict]]:
    if not events:
        return []
    start = events[0]["_timestamp"]
    end = events[-1]["_timestamp"]
    windows = []
    t = start
    while t + length_seconds <= end:
        chunk = [event for event in events if t <= event["_timestamp"] < t + length_seconds]
        if chunk:
            windows.append(chunk)
        t += step_seconds
    return windows


def chunk_event_windows(events: list[dict], window_size: int, step_size: int, min_events: int) -> list[list[dict]]:
    if len(events) < min_events:
        return []
    starts = list(range(0, max(1, len(events) - window_size + 1), step_size))
    if len(events) > window_size and starts[-1] != len(events) - window_size:
        starts.append(len(events) - window_size)
    chunks = []
    for start in starts:
        chunk = events[start:start + window_size]
        if len(chunk) >= min_events:
            chunks.append(chunk)
    return chunks


def counts(keys: list[str]) -> Counter:
    return Counter(keys)


def transitions(keys: list[str]) -> Counter:
    return Counter(zip(keys, keys[1:]))


def probs(counter: Counter) -> dict:
    total = sum(counter.values())
    if total <= 0:
        return {}
    return {key: value / total for key, value in counter.items()}


def js_divergence(a: Counter, b: Counter) -> float:
    pa = probs(a)
    pb = probs(b)
    keys = set(pa) | set(pb)
    if not keys:
        return 0.0
    div = 0.0
    for key in keys:
        p = pa.get(key, 0.0)
        q = pb.get(key, 0.0)
        m = 0.5 * (p + q)
        if p > 0:
            div += 0.5 * p * math.log(p / m)
        if q > 0:
            div += 0.5 * q * math.log(q / m)
    value = div / math.log(2.0)
    if value < 0.0 and value > -1e-12:
        return 0.0
    if value > 1.0 and value < 1.0 + 1e-12:
        return 1.0
    return value


def gaps(events: list[dict]) -> list[float]:
    return [
        max(0.0, events[i]["_timestamp"] - events[i - 1]["_timestamp"])
        for i in range(1, len(events))
    ]


def quantile(values: list[float], q: float) -> float:
    clean = sorted(float(v) for v in values if v is not None and math.isfinite(float(v)))
    if not clean:
        return float("nan")
    if len(clean) == 1:
        return clean[0]
    pos = (len(clean) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return clean[lo]
    frac = pos - lo
    return clean[lo] * (1 - frac) + clean[hi] * frac


def wasserstein_1d(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return float("nan")
    qs = [i / 100.0 for i in range(101)]
    return sum(abs(quantile(a, q) - quantile(b, q)) for q in qs) / len(qs)


def vector_from_counts(counter: Counter, vocab: list[str]) -> list[float]:
    total = sum(counter.values()) or 1
    return [counter.get(key, 0) / total for key in vocab]


def sqdist(a: list[float], b: list[float]) -> float:
    return sum((x - y) ** 2 for x, y in zip(a, b))


def rbf(x: list[float], y: list[float], sigma2: float) -> float:
    if sigma2 <= 0 or not math.isfinite(sigma2):
        sigma2 = 1.0
    return math.exp(-sqdist(x, y) / (2.0 * sigma2))


def mmd_single(pre_vectors: list[list[float]], y: list[float]) -> float:
    if not pre_vectors:
        return float("nan")
    distances = []
    for i, x in enumerate(pre_vectors):
        for z in pre_vectors[i + 1:]:
            distances.append(sqdist(x, z))
    sigma2 = quantile(distances, 0.5) if distances else 1.0
    xx = sum(rbf(x, z, sigma2) for x in pre_vectors for z in pre_vectors) / (len(pre_vectors) ** 2)
    xy = sum(rbf(x, y, sigma2) for x in pre_vectors) / len(pre_vectors)
    yy = 1.0
    return max(0.0, xx - 2.0 * xy + yy)


def stable_oov_bucket(template: str, num_buckets: int) -> int:
    digest = hashlib.sha256(normalize_template(template).encode("utf-8", errors="replace")).hexdigest()
    return int(digest[:16], 16) % num_buckets
