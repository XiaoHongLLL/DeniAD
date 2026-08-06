#!/usr/bin/env python3
"""Prepare labeled log datasets for FlowMatching-THP.

The output format matches preprocess/Dataset.py:

{
    "dim_process": <number of event types>,
    "train": [[{"time_since_start": ..., "time_since_last_event": ...,
                "type_event": ..., "label": ...}, ...], ...],
    "dev": [...],
    "test": [...]
}

Training and calibration/dev splits are built from normal-only sequences.
Anomalous sequences are kept for test so anomaly metrics can be computed.
"""

from __future__ import annotations

import argparse
import ast
import csv
import gzip
import json
import pickle
import random
import re
from collections import Counter, defaultdict, deque
from datetime import datetime
from pathlib import Path


NORMAL_LABELS = {"-", "0", "false", "normal", "benign", "ok", "", "none", "nan", "success"}

BGL_LINE_RE = re.compile(
    r"^(?P<label>\S+)\s+"
    r"(?P<timestamp>\d+)\s+"
    r"(?P<date>\S+)\s+"
    r"(?P<node>\S+)\s+"
    r"(?P<time>\S+)\s+"
    r"(?P<node_repeat>\S+)\s+"
    r"(?P<type>\S+)\s+"
    r"(?P<component>\S+)\s+"
    r"(?P<level>\S+)\s+"
    r"(?P<content>.*)$"
)

THUNDERBIRD_LINE_RE = re.compile(
    r"^(?P<label>\S+)\s+"
    r"(?P<timestamp>\d+)\s+"
    r"(?P<date>\S+)\s+"
    r"(?P<user>\S+)\s+"
    r"(?P<month>\S+)\s+"
    r"(?P<day>\d+)\s+"
    r"(?P<time>\S+)\s+"
    r"(?P<location>\S+)\s+"
    r"(?P<rest>.*)$"
)

HPC4_LINE_RE = re.compile(
    r"^(?P<label>\S+)\s+"
    r"(?P<timestamp>\d+)\s+"
    r"(?P<date>\S+)\s+"
    r"(?P<user>\S+)\s+"
    r"(?P<month>\S+)\s+"
    r"(?P<day>\d+)\s+"
    r"(?P<time>\S+)\s+"
    r"(?P<location>\S+)"
    r"(?:\s+(?P<rest>.*))?$"
)

OPENSTACK_RE = re.compile(
    r"^(?P<logfile>\S+)\s+"
    r"(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2}\.\d+)\s+"
    r"(?P<pid>\d+)\s+"
    r"(?P<level>\S+)\s+"
    r"(?P<logger>\S+)\s+"
    r"(?P<content>.*)$"
)

HADOOP_RE = re.compile(
    r"^(?P<date>\d{4}-\d{2}-\d{2})\s+"
    r"(?P<time>\d{2}:\d{2}:\d{2},\d+)\s+"
    r"(?P<level>\S+)\s+"
    r"\[(?P<thread>[^\]]*)\]\s+"
    r"(?P<logger>[^:]+):\s*"
    r"(?P<content>.*)$"
)

COMPONENT_RE = re.compile(r"^(?P<component>[^\s:\[]+)(?:\[(?P<pid>\d+)\])?:?\s*(?P<content>.*)$")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
HEX_RE = re.compile(r"\b0x[0-9a-fA-F]+\b")
IP_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
NUMBER_RE = re.compile(r"(?<![A-Za-z])[-+]?\d+(?:\.\d+)?")
APP_RE = re.compile(r"\b(?:application|appattempt|container)_\d+_\d+(?:_\d+)*(?:_\d+)?\b")
BLOCK_RE = re.compile(r"\bblk_-?\d+\b")
PATH_RE = re.compile(r"(?:/[^\s]+)+")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare labeled log datasets for DeniAD.")
    parser.add_argument("--raw_root", default="./raw_logs", help="Root directory containing raw datasets.")
    parser.add_argument("--out_root", default="./data", help="Directory where labeled dataset folders are written.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["hdfs", "bgl", "hadoop", "openstack"],
        choices=["hdfs", "hdfs_v3", "bgl", "hadoop", "openstack", "thunderbird", "spirit", "liberty", "all"],
        help="Datasets to prepare. 'all' expands to all supported datasets.",
    )
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--train_ratio", type=float, default=0.6, help="Fraction of normal sequences used for train.")
    parser.add_argument("--dev_ratio", type=float, default=0.2, help="Fraction of normal sequences used for dev.")
    parser.add_argument("--window_size", type=int, default=50, help="Window size for streaming log datasets.")
    parser.add_argument("--step_size", type=int, default=50, help="Step size for streaming log datasets.")
    parser.add_argument("--max_event_types", type=int, default=512, help="Top event types to keep; the rest map to E_OTHER.")
    parser.add_argument("--min_count", type=int, default=1, help="Minimum mark count before it can be kept.")
    parser.add_argument("--max_lines_bgl", type=int, default=0, help="Maximum BGL lines to read; <=0 means all.")
    parser.add_argument(
        "--max_lines_thunderbird",
        type=int,
        default=2_000_000,
        help="Maximum Thunderbird lines to read; <=0 means all. Full file is very large.",
    )
    parser.add_argument(
        "--max_lines_spirit",
        type=int,
        default=2_000_000,
        help="Maximum Spirit lines to read; <=0 means all. Full file is very large.",
    )
    parser.add_argument(
        "--max_lines_liberty",
        type=int,
        default=2_000_000,
        help="Maximum Liberty lines to read; <=0 means all. Full file is very large.",
    )
    parser.add_argument("--skip_lines_spirit", type=int, default=0, help="Spirit lines to skip before reading.")
    parser.add_argument("--skip_lines_liberty", type=int, default=0, help="Liberty lines to skip before reading.")
    parser.add_argument("--max_sequences_hdfs", type=int, default=120_000, help="Maximum HDFS traces to keep; <=0 means all.")
    parser.add_argument(
        "--max_sequences_hdfs_v3",
        type=int,
        default=120_000,
        help="Maximum TraceBench/HDFS_v3 traces to keep; <=0 means all.",
    )
    parser.add_argument(
        "--max_hdfs_v3_normal_files",
        type=int,
        default=0,
        help="Maximum HDFS_v3 NM_*.sql files to scan; <=0 means scan until the sequence cap is met.",
    )
    parser.add_argument(
        "--max_hdfs_v3_anomaly_files",
        type=int,
        default=0,
        help="Maximum HDFS_v3 AN_*.sql files to scan; <=0 means scan until the sequence cap is met.",
    )
    parser.add_argument(
        "--hdfs_v3_event_type_mode",
        choices=["agent_op", "agent_op_status"],
        default="agent_op",
        help="TraceBench/HDFS_v3 event abstraction.",
    )
    parser.add_argument("--max_sequences", type=int, default=0, help="Global cap per prepared dataset after split; <=0 disables.")
    return parser.parse_args()


def is_anomaly_label(label) -> bool:
    if label is None:
        return False
    return str(label).strip().lower() not in NORMAL_LABELS


def binary_label(label) -> int:
    return int(is_anomaly_label(label))


def normalize_message(text: str) -> str:
    text = UUID_RE.sub("<UUID>", text)
    text = APP_RE.sub("<APP>", text)
    text = BLOCK_RE.sub("<BLOCK>", text)
    text = IP_RE.sub("<IP>", text)
    text = HEX_RE.sub("<HEX>", text)
    text = PATH_RE.sub("<PATH>", text)
    text = NUMBER_RE.sub("<NUM>", text)
    return " ".join(text.strip().split())


def open_text(path: Path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="latin-1", errors="replace")
    return path.open("r", encoding="latin-1", errors="replace")


def make_event(timestamp: float, first_timestamp: float, previous_timestamp: float, type_event: int, label) -> dict:
    return {
        "time_since_start": max(0.0, float(timestamp - first_timestamp)),
        "time_since_last_event": max(0.0, float(timestamp - previous_timestamp)),
        "type_event": int(type_event),
        "label": int(binary_label(label)),
        "raw_label": str(label),
    }


def split_normal_train_dev_test(sequences, train_ratio: float, dev_ratio: float, seed: int, max_sequences: int = 0):
    normal = [seq for seq in sequences if not any(is_anomaly_label(ev.get("label")) for ev in seq)]
    anomalous = [seq for seq in sequences if any(is_anomaly_label(ev.get("label")) for ev in seq)]

    rng = random.Random(seed)
    rng.shuffle(normal)
    rng.shuffle(anomalous)

    n_train = int(len(normal) * train_ratio)
    n_dev = int(len(normal) * dev_ratio)
    if len(normal) >= 3:
        n_train = max(1, n_train)
        n_dev = max(1, n_dev)
    if n_train + n_dev > len(normal):
        n_dev = max(0, len(normal) - n_train)

    train = normal[:n_train]
    dev = normal[n_train:n_train + n_dev]
    test = normal[n_train + n_dev:] + anomalous
    rng.shuffle(test)

    if max_sequences and max_sequences > 0:
        train = train[:max_sequences]
        dev = dev[:max_sequences]
        test_anomalous = [seq for seq in test if any(is_anomaly_label(ev.get("label")) for ev in seq)]
        test_normal = [seq for seq in test if not any(is_anomaly_label(ev.get("label")) for ev in seq)]
        if len(test_anomalous) >= max_sequences:
            test = test_anomalous[:max_sequences]
        else:
            test = test_anomalous + test_normal[:max_sequences - len(test_anomalous)]
            rng.shuffle(test)
    return train, dev, test


def dataset_stats(splits):
    stats = {}
    for name, seqs in splits.items():
        events = sum(len(seq) for seq in seqs)
        anomaly_events = sum(binary_label(ev.get("label")) for seq in seqs for ev in seq)
        anomaly_sequences = sum(any(binary_label(ev.get("label")) for ev in seq) for seq in seqs)
        stats[name] = {
            "sequences": len(seqs),
            "events": events,
            "anomaly_events": anomaly_events,
            "anomaly_sequences": anomaly_sequences,
        }
    return stats


def write_dataset(out_dir: Path, dim_process: int, train, dev, test, metadata: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    payloads = {
        "train.pkl": {"dim_process": dim_process, "train": train},
        "dev.pkl": {"dim_process": dim_process, "dev": dev},
        "test.pkl": {"dim_process": dim_process, "test": test},
    }
    for name, payload in payloads.items():
        with (out_dir / name).open("wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = dict(metadata)
    metadata["dim_process"] = dim_process
    metadata["stats"] = dataset_stats({"train": train, "dev": dev, "test": test})
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote {out_dir}")
    print(json.dumps(metadata["stats"], indent=2, ensure_ascii=False))


def parse_feature_list(text: str):
    text = text.strip()
    if text.startswith("[") and text.endswith("]"):
        text = text[1:-1]
    return [item.strip().strip("'\"") for item in text.split(",") if item.strip()]


def parse_float_list(text: str):
    try:
        values = ast.literal_eval(text)
    except Exception:
        values = []
    return [float(v) for v in values]


def prepare_hdfs(raw_root: Path, out_root: Path, args):
    traces_path = raw_root / "HDFS_v1" / "preprocessed" / "Event_traces.csv"
    labels_path = raw_root / "HDFS_v1" / "preprocessed" / "anomaly_label.csv"
    if not traces_path.exists() or not labels_path.exists():
        raise FileNotFoundError("HDFS_v1 preprocessed Event_traces.csv/anomaly_label.csv not found.")

    labels = {}
    with labels_path.open(newline="", encoding="utf-8", errors="replace") as f:
        for row in csv.DictReader(f):
            labels[row["BlockId"]] = row["Label"]

    rows = []
    with traces_path.open(newline="", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        for row in reader:
            block_id = row.get("BlockId")
            label = labels.get(block_id, row.get("Label", "Normal"))
            rows.append((row, label))

    rng = random.Random(args.seed)
    rng.shuffle(rows)
    if args.max_sequences_hdfs and args.max_sequences_hdfs > 0:
        normal_rows = [x for x in rows if not is_anomaly_label(x[1])]
        anomaly_rows = [x for x in rows if is_anomaly_label(x[1])]
        max_anom = min(len(anomaly_rows), max(1, args.max_sequences_hdfs // 5))
        max_norm = max(0, args.max_sequences_hdfs - max_anom)
        rows = normal_rows[:max_norm] + anomaly_rows[:max_anom]
        rng.shuffle(rows)

    all_event_ids = sorted({eid for row, _ in rows for eid in parse_feature_list(row["Features"])})
    event_to_type = {eid: idx for idx, eid in enumerate(all_event_ids)}

    sequences = []
    for row, label in rows:
        features = parse_feature_list(row["Features"])
        intervals = parse_float_list(row.get("TimeInterval", "[]"))
        if not features:
            continue
        if len(intervals) < len(features):
            intervals = intervals + [0.0] * (len(features) - len(intervals))
        timestamp = 0.0
        previous = 0.0
        seq = []
        for idx, eid in enumerate(features):
            gap = max(0.0, float(intervals[idx]))
            timestamp = previous + gap
            seq.append(
                {
                    "time_since_start": timestamp,
                    "time_since_last_event": gap,
                    "type_event": event_to_type[eid],
                    "label": int(binary_label(label)),
                    "raw_label": str(label),
                    "event_id": eid,
                    "block_id": row.get("BlockId", ""),
                }
            )
            previous = timestamp
        if len(seq) >= 2:
            sequences.append(seq)

    train, dev, test = split_normal_train_dev_test(
        sequences, args.train_ratio, args.dev_ratio, args.seed, args.max_sequences
    )
    write_dataset(
        out_root / "labeled_hdfs",
        len(event_to_type),
        train,
        dev,
        test,
        {
            "dataset": "HDFS_v1",
            "source": str(traces_path),
            "label_granularity": "block_trace",
            "event_type_mode": "preprocessed_event_id",
            "num_raw_sequences": len(sequences),
        },
    )


def iter_sql_tuple_bodies(values_text: str):
    in_quote = False
    escaped = False
    depth = 0
    start = None
    for idx, ch in enumerate(values_text):
        if in_quote:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_quote = False
            continue
        if ch == "'":
            in_quote = True
        elif ch == "(":
            if depth == 0:
                start = idx + 1
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0 and start is not None:
                yield values_text[start:idx]
                start = None


def parse_sql_tuple_fields(tuple_body: str):
    fields = []
    buf = []
    in_quote = False
    escaped = False
    for ch in tuple_body:
        if in_quote:
            if escaped:
                buf.append(ch)
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_quote = False
            else:
                buf.append(ch)
            continue
        if ch == "'":
            in_quote = True
        elif ch == ",":
            fields.append("".join(buf).strip())
            buf = []
        else:
            buf.append(ch)
    fields.append("".join(buf).strip())
    return [None if field.upper() == "NULL" else field for field in fields]


def iter_tracebench_report_rows(path: Path):
    prefix = "INSERT INTO `Report` VALUES "
    with path.open("r", encoding="latin-1", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line.startswith(prefix):
                continue
            values_text = line[len(prefix):].rstrip(";")
            for tuple_body in iter_sql_tuple_bodies(values_text):
                fields = parse_sql_tuple_fields(tuple_body)
                if len(fields) >= 9:
                    yield {
                        "task_id": fields[0],
                        "tid": fields[1],
                        "op_name": fields[2],
                        "start_time": fields[3],
                        "end_time": fields[4],
                        "host_address": fields[5],
                        "host_name": fields[6],
                        "agent": fields[7],
                        "description": fields[8],
                    }


def find_hdfs_v3_root(raw_root: Path) -> Path:
    candidates = [
        raw_root / "HDFS_v3" / "TraceBench-master",
        raw_root / "HDFS_v3" / "TraceBench-main",
        raw_root / "TraceBench-master",
        raw_root / "TraceBench-main",
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(candidate.glob("*.sql")):
            return candidate
    candidate_text = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"HDFS_v3/TraceBench SQL directory not found. Tried:\n{candidate_text}")


def make_tracebench_mark(row: dict, mode: str) -> str:
    agent = row.get("agent") or "UNKNOWN"
    op_name = row.get("op_name") or "UNKNOWN"
    if mode == "agent_op_status":
        description = row.get("description") or ""
        status = description.split(":", 1)[0].strip() or "EMPTY"
        return "|".join([agent, op_name, status])
    return "|".join([agent, op_name])


def collect_tracebench_sequences(files, label: int, raw_label: str, args, max_sequences: int = 0):
    sequences = []
    counter = Counter()
    files_scanned = 0
    for file_idx, path in enumerate(files, start=1):
        if max_sequences and len(sequences) >= max_sequences:
            break
        print(f"  [HDFS_v3] {raw_label}: scanning {file_idx}/{len(files)} {path.name}", flush=True)
        files_scanned += 1
        by_task = defaultdict(list)
        for row in iter_tracebench_report_rows(path):
            try:
                timestamp = float(row["start_time"]) / 1_000_000.0
            except (TypeError, ValueError):
                continue
            mark = make_tracebench_mark(row, args.hdfs_v3_event_type_mode)
            by_task[row["task_id"]].append((timestamp, mark, row["tid"]))

        for task_id, events in by_task.items():
            if len(events) < 2:
                continue
            events.sort(key=lambda item: (item[0], item[2] or ""))
            first = events[0][0]
            previous = first
            seq = []
            for idx, (timestamp, mark, tid) in enumerate(events):
                gap = 0.0 if idx == 0 else max(0.0, timestamp - previous)
                seq.append(
                    {
                        "time_since_start": max(0.0, timestamp - first),
                        "time_since_last_event": gap,
                        "mark": mark,
                        "label": int(label),
                        "raw_label": raw_label,
                        "task_id": task_id,
                        "tid": tid,
                        "source_file": path.name,
                    }
                )
                counter[mark] += 1
                previous = timestamp
            sequences.append(seq)
            if max_sequences and len(sequences) >= max_sequences:
                break
    return sequences, counter, files_scanned


def prepare_hdfs_v3(raw_root: Path, out_root: Path, args):
    root = find_hdfs_v3_root(raw_root)
    normal_files = sorted(root.glob("NM_*.sql"))
    anomaly_files = sorted(root.glob("AN_*.sql"))
    if not normal_files or not anomaly_files:
        raise FileNotFoundError(f"HDFS_v3 requires NM_*.sql and AN_*.sql files under {root}")

    rng = random.Random(args.seed)
    rng.shuffle(normal_files)
    rng.shuffle(anomaly_files)
    if args.max_hdfs_v3_normal_files and args.max_hdfs_v3_normal_files > 0:
        normal_files = normal_files[:args.max_hdfs_v3_normal_files]
    if args.max_hdfs_v3_anomaly_files and args.max_hdfs_v3_anomaly_files > 0:
        anomaly_files = anomaly_files[:args.max_hdfs_v3_anomaly_files]

    max_total = args.max_sequences_hdfs_v3
    if max_total and max_total > 0:
        max_anomaly = max(1, max_total // 5)
        max_normal = max(1, max_total - max_anomaly)
    else:
        max_normal = 0
        max_anomaly = 0

    normal_sequences, normal_counter, normal_scanned = collect_tracebench_sequences(
        normal_files, 0, "Normal", args, max_normal
    )
    anomaly_sequences, anomaly_counter, anomaly_scanned = collect_tracebench_sequences(
        anomaly_files, 1, "Anomaly", args, max_anomaly
    )
    sequences = normal_sequences + anomaly_sequences
    counter = normal_counter + anomaly_counter
    if not sequences:
        raise RuntimeError(f"No HDFS_v3 sequences parsed from {root}")

    mapping = build_mapping_from_counter(counter, args.max_event_types, args.min_count)
    for seq in sequences:
        for event in seq:
            event["type_event"] = map_mark(event.pop("mark"), mapping)

    train, dev, test = split_normal_train_dev_test(
        sequences, args.train_ratio, args.dev_ratio, args.seed, args.max_sequences
    )
    write_dataset(
        out_root / "labeled_hdfs_v3",
        len(mapping),
        train,
        dev,
        test,
        {
            "dataset": "HDFS_v3/TraceBench",
            "source": str(root),
            "label_granularity": "trace_file_task",
            "event_type_mode": args.hdfs_v3_event_type_mode,
            "normal_files_available": len(list(root.glob("NM_*.sql"))),
            "anomaly_files_available": len(list(root.glob("AN_*.sql"))),
            "normal_files_scanned": normal_scanned,
            "anomaly_files_scanned": anomaly_scanned,
            "max_sequences_hdfs_v3": args.max_sequences_hdfs_v3,
            "num_raw_sequences": len(sequences),
        },
    )


def build_mapping_from_counter(counter: Counter, max_event_types: int, min_count: int):
    kept = [
        key
        for key, count in counter.most_common(max(1, max_event_types - 1))
        if count >= min_count
    ]
    if not kept:
        kept = [counter.most_common(1)[0][0]]
    mapping = {key: idx for idx, key in enumerate(kept)}
    other_idx = len(mapping)
    mapping["E_OTHER"] = other_idx
    return mapping


def map_mark(mark: str, mapping: dict):
    return mapping.get(mark, mapping["E_OTHER"])


def iter_bgl_rows(path: Path, max_lines: int = 0):
    with path.open("r", encoding="latin-1", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if max_lines and max_lines > 0 and line_no > max_lines:
                break
            match = BGL_LINE_RE.match(line.rstrip("\n"))
            if not match:
                continue
            row = match.groupdict()
            row["timestamp"] = float(row["timestamp"])
            row["mark"] = "|".join(
                [
                    row["type"],
                    row["component"],
                    row["level"],
                    normalize_message(row["content"]),
                ]
            )
            yield row


def iter_thunderbird_rows(path: Path, max_lines: int = 0):
    with open_text(path) as f:
        for line_no, line in enumerate(f, start=1):
            if max_lines and max_lines > 0 and line_no > max_lines:
                break
            match = THUNDERBIRD_LINE_RE.match(line.rstrip("\n"))
            if not match:
                continue
            row = match.groupdict()
            comp_match = COMPONENT_RE.match(row.pop("rest"))
            if comp_match:
                row.update(comp_match.groupdict())
            else:
                row.update({"component": "UNKNOWN", "pid": "", "content": ""})
            row["timestamp"] = float(row["timestamp"])
            row["mark"] = "|".join([row["component"] or "UNKNOWN", normalize_message(row.get("content") or "")])
            yield row


def find_first_existing_file(candidates):
    for path in candidates:
        if path.is_file():
            return path
    return None


def iter_hpc4_rows(path: Path, max_lines: int = 0, skip_lines: int = 0):
    with open_text(path) as f:
        kept_lines = 0
        for line_no, line in enumerate(f, start=1):
            if skip_lines and line_no <= skip_lines:
                continue
            if max_lines and max_lines > 0 and kept_lines >= max_lines:
                break
            kept_lines += 1
            match = HPC4_LINE_RE.match(line.rstrip("\n"))
            if not match:
                continue
            row = match.groupdict()
            row["rest"] = row.get("rest") or ""
            comp_match = COMPONENT_RE.match(row["rest"])
            if comp_match:
                row.update(comp_match.groupdict())
            else:
                row.update({"component": "UNKNOWN", "pid": "", "content": row["rest"]})
            row["timestamp"] = float(row["timestamp"])
            component = row.get("component") or "UNKNOWN"
            normalized = normalize_message(row.get("content") or "")
            if normalized:
                row["mark"] = "|".join([component, normalized])
            else:
                row["mark"] = "|".join(["EMPTY", row.get("location") or "UNKNOWN"])
            yield row


def windows_from_rows(row_iter, mapping: dict, window_size: int, step_size: int):
    sequences = []
    buffer = deque()
    for row in row_iter:
        event = {
            "timestamp": float(row["timestamp"]),
            "type_event": map_mark(row["mark"], mapping),
            "label": int(binary_label(row.get("label", "-"))),
            "raw_label": str(row.get("label", "-")),
        }
        buffer.append(event)
        if len(buffer) == window_size:
            seq_items = list(buffer)
            first = seq_items[0]["timestamp"]
            previous = first
            seq = []
            for idx, item in enumerate(seq_items):
                timestamp = item["timestamp"]
                gap = 0.0 if idx == 0 else max(0.0, timestamp - previous)
                seq.append(
                    {
                        "time_since_start": max(0.0, timestamp - first),
                        "time_since_last_event": gap,
                        "type_event": item["type_event"],
                        "label": item["label"],
                        "raw_label": item["raw_label"],
                    }
                )
                previous = timestamp
            sequences.append(seq)
            for _ in range(min(step_size, len(buffer))):
                buffer.popleft()
    return sequences


def prepare_bgl(raw_root: Path, out_root: Path, args):
    path = raw_root / "BGL" / "BGL.log"
    if not path.exists():
        raise FileNotFoundError(f"BGL.log not found: {path}")
    counter = Counter(row["mark"] for row in iter_bgl_rows(path, args.max_lines_bgl))
    mapping = build_mapping_from_counter(counter, args.max_event_types, args.min_count)
    sequences = windows_from_rows(
        iter_bgl_rows(path, args.max_lines_bgl),
        mapping,
        args.window_size,
        args.step_size,
    )
    train, dev, test = split_normal_train_dev_test(
        sequences, args.train_ratio, args.dev_ratio, args.seed, args.max_sequences
    )
    write_dataset(
        out_root / "labeled_bgl",
        len(mapping),
        train,
        dev,
        test,
        {
            "dataset": "BGL",
            "source": str(path),
            "label_granularity": "event_line",
            "event_type_mode": "type_component_level_normalized_content",
            "max_lines": args.max_lines_bgl,
            "window_size": args.window_size,
            "step_size": args.step_size,
            "num_raw_sequences": len(sequences),
        },
    )


def prepare_thunderbird(raw_root: Path, out_root: Path, args):
    candidates = [
        raw_root / "Thunderbird_full" / "tbird2.gz",
        raw_root / "log-thunderbird-master" / "tbird2.gz",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise FileNotFoundError("Full Thunderbird tbird2.gz not found. Expected E:\\鏁版嵁闆哱\Thunderbird_full\\tbird2.gz.")
    counter = Counter(row["mark"] for row in iter_thunderbird_rows(path, args.max_lines_thunderbird))
    mapping = build_mapping_from_counter(counter, args.max_event_types, args.min_count)
    sequences = windows_from_rows(
        iter_thunderbird_rows(path, args.max_lines_thunderbird),
        mapping,
        args.window_size,
        args.step_size,
    )
    train, dev, test = split_normal_train_dev_test(
        sequences, args.train_ratio, args.dev_ratio, args.seed, args.max_sequences
    )
    write_dataset(
        out_root / "labeled_thunderbird",
        len(mapping),
        train,
        dev,
        test,
        {
            "dataset": "Thunderbird",
            "source": str(path),
            "label_granularity": "event_line",
            "event_type_mode": "component_normalized_content",
            "max_lines": args.max_lines_thunderbird,
            "window_size": args.window_size,
            "step_size": args.step_size,
            "num_raw_sequences": len(sequences),
        },
    )


def prepare_hpc4_dataset(
    raw_root: Path,
    out_root: Path,
    args,
    dataset_name: str,
    folder_name: str,
    file_stem: str,
    max_lines: int,
    skip_lines: int = 0,
):
    candidates = [
        raw_root / folder_name / file_stem / file_stem,
        raw_root / folder_name / file_stem,
        raw_root / folder_name / f"{file_stem}.gz",
        raw_root / file_stem / file_stem,
        raw_root / file_stem,
        raw_root / f"{file_stem}.gz",
    ]
    path = find_first_existing_file(candidates)
    if path is None:
        candidate_text = "\n".join(str(p) for p in candidates)
        raise FileNotFoundError(f"{dataset_name} raw log not found. Tried:\n{candidate_text}")

    counter = Counter(row["mark"] for row in iter_hpc4_rows(path, max_lines, skip_lines))
    mapping = build_mapping_from_counter(counter, args.max_event_types, args.min_count)
    sequences = windows_from_rows(
        iter_hpc4_rows(path, max_lines, skip_lines),
        mapping,
        args.window_size,
        args.step_size,
    )
    train, dev, test = split_normal_train_dev_test(
        sequences, args.train_ratio, args.dev_ratio, args.seed, args.max_sequences
    )
    write_dataset(
        out_root / f"labeled_{dataset_name.lower()}",
        len(mapping),
        train,
        dev,
        test,
        {
            "dataset": dataset_name,
            "source": str(path),
            "label_granularity": "event_line",
            "event_type_mode": "component_normalized_content_or_empty_location",
            "skip_lines": skip_lines,
            "max_lines": max_lines,
            "window_size": args.window_size,
            "step_size": args.step_size,
            "num_raw_sequences": len(sequences),
        },
    )


def prepare_spirit(raw_root: Path, out_root: Path, args):
    prepare_hpc4_dataset(
        raw_root,
        out_root,
        args,
        "Spirit",
        "Spirit",
        "spirit2",
        args.max_lines_spirit,
        args.skip_lines_spirit,
    )


def prepare_liberty(raw_root: Path, out_root: Path, args):
    prepare_hpc4_dataset(
        raw_root,
        out_root,
        args,
        "Liberty",
        "Liberty",
        "liberty2",
        args.max_lines_liberty,
        args.skip_lines_liberty,
    )


def parse_openstack_time(date_text: str, time_text: str) -> float:
    dt = datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M:%S.%f")
    return dt.timestamp()


def iter_openstack_events(raw_root: Path, mapping: dict | None = None):
    root = raw_root / "OpenStack.tar" / "OpenStack"
    label_path = root / "anomaly_labels.txt"
    abnormal_ids = set()
    if label_path.exists():
        for line in label_path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if UUID_RE.fullmatch(line):
                abnormal_ids.add(line.lower())

    files = [
        (root / "openstack_normal1.log", False),
        (root / "openstack_normal2.log", False),
        (root / "openstack_abnormal.log", True),
    ]
    for path, abnormal_file in files:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                match = OPENSTACK_RE.match(line.rstrip("\n"))
                if not match:
                    continue
                row = match.groupdict()
                timestamp = parse_openstack_time(row["date"], row["time"])
                content = row["content"]
                label = int(abnormal_file and any(uuid in line.lower() for uuid in abnormal_ids))
                mark = "|".join(
                    [
                        row["logfile"].split(".")[0],
                        row["level"],
                        row["logger"],
                        normalize_message(content),
                    ]
                )
                yield {
                    "timestamp": timestamp,
                    "mark": mark,
                    "label": label,
                    "raw_label": "anomaly_vm" if label else "normal",
                    "source_file": path.name,
                }


def prepare_openstack(raw_root: Path, out_root: Path, args):
    counter = Counter(row["mark"] for row in iter_openstack_events(raw_root))
    mapping = build_mapping_from_counter(counter, args.max_event_types, args.min_count)
    rows_by_file = {}
    for row in iter_openstack_events(raw_root):
        source_file = row["source_file"]
        rows_by_file.setdefault(source_file, []).append(
            {
                "timestamp": row["timestamp"],
                "type_event": map_mark(row["mark"], mapping),
                "label": row["label"],
                "raw_label": row["raw_label"],
                "source_file": source_file,
            }
        )
    sequences = []
    for source_file, rows in rows_by_file.items():
        rows.sort(key=lambda item: item["timestamp"])
        buffer = deque()
        for item in rows:
            buffer.append(item)
            if len(buffer) == args.window_size:
                seq_items = list(buffer)
                first = seq_items[0]["timestamp"]
                previous = first
                seq = []
                for idx, ev in enumerate(seq_items):
                    timestamp = ev["timestamp"]
                    gap = 0.0 if idx == 0 else max(0.0, timestamp - previous)
                    seq.append(
                        {
                            "time_since_start": max(0.0, timestamp - first),
                            "time_since_last_event": gap,
                            "type_event": ev["type_event"],
                            "label": int(ev["label"]),
                            "raw_label": ev["raw_label"],
                            "source_file": source_file,
                        }
                    )
                    previous = timestamp
                sequences.append(seq)
                for _ in range(min(args.step_size, len(buffer))):
                    buffer.popleft()

    train, dev, test = split_normal_train_dev_test(
        sequences, args.train_ratio, args.dev_ratio, args.seed, args.max_sequences
    )
    write_dataset(
        out_root / "labeled_openstack",
        len(mapping),
        train,
        dev,
        test,
        {
            "dataset": "OpenStack",
            "source": str(raw_root / "OpenStack.tar" / "OpenStack"),
            "label_granularity": "event_line_contains_injected_vm_id",
            "event_type_mode": "logfile_level_logger_normalized_content",
            "window_size": args.window_size,
            "step_size": args.step_size,
            "num_raw_sequences": len(sequences),
        },
    )


def parse_hadoop_labels(label_file: Path):
    labels = {}
    current_label = None
    for line in label_file.read_text(encoding="utf-8", errors="replace").splitlines():
        text = line.strip()
        if not text:
            continue
        if text.endswith(":") and not text.startswith("+"):
            current_label = text[:-1].strip()
            continue
        if text.startswith("+") and current_label:
            app_id = text[1:].strip()
            labels[app_id] = current_label
    return labels


def parse_hadoop_time(date_text: str, time_text: str) -> float:
    dt = datetime.strptime(f"{date_text} {time_text}", "%Y-%m-%d %H:%M:%S,%f")
    return dt.timestamp()


def iter_hadoop_app_events(app_dir: Path):
    for log_file in sorted(app_dir.glob("*.log")):
        with log_file.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                match = HADOOP_RE.match(line.rstrip("\n"))
                if not match:
                    continue
                row = match.groupdict()
                timestamp = parse_hadoop_time(row["date"], row["time"])
                mark = "|".join(
                    [
                        row["level"],
                        normalize_message(row["logger"].strip()),
                        normalize_message(row["content"]),
                    ]
                )
                yield timestamp, mark


def prepare_hadoop(raw_root: Path, out_root: Path, args):
    root = raw_root / "Hadoop"
    labels = parse_hadoop_labels(root / "abnormal_label.txt")
    all_marks = Counter()
    app_events = {}
    for app_id, label in sorted(labels.items()):
        app_dir = root / app_id
        if not app_dir.exists():
            continue
        events = list(iter_hadoop_app_events(app_dir))
        if not events:
            continue
        events.sort(key=lambda item: item[0])
        app_events[app_id] = (label, events)
        all_marks.update(mark for _, mark in events)

    mapping = build_mapping_from_counter(all_marks, args.max_event_types, args.min_count)
    sequences = []
    for app_id, (label, events) in app_events.items():
        seq = []
        first = events[0][0]
        previous = first
        for idx, (timestamp, mark) in enumerate(events):
            gap = 0.0 if idx == 0 else max(0.0, timestamp - previous)
            seq.append(
                {
                    "time_since_start": max(0.0, timestamp - first),
                    "time_since_last_event": gap,
                    "type_event": map_mark(mark, mapping),
                    "label": int(is_anomaly_label(label)),
                    "raw_label": label,
                    "application_id": app_id,
                }
            )
            previous = timestamp
        if len(seq) >= 2:
            sequences.append(seq)

    train, dev, test = split_normal_train_dev_test(
        sequences, args.train_ratio, args.dev_ratio, args.seed, args.max_sequences
    )
    write_dataset(
        out_root / "labeled_hadoop",
        len(mapping),
        train,
        dev,
        test,
        {
            "dataset": "Hadoop",
            "source": str(root),
            "label_granularity": "application_job",
            "event_type_mode": "level_logger_normalized_content",
            "num_raw_sequences": len(sequences),
        },
    )


def main():
    args = parse_args()
    datasets = args.datasets
    if "all" in datasets:
        datasets = ["hdfs", "hdfs_v3", "bgl", "hadoop", "openstack", "thunderbird", "spirit", "liberty"]

    raw_root = Path(args.raw_root)
    out_root = Path(args.out_root)
    handlers = {
        "hdfs": prepare_hdfs,
        "hdfs_v3": prepare_hdfs_v3,
        "bgl": prepare_bgl,
        "hadoop": prepare_hadoop,
        "openstack": prepare_openstack,
        "thunderbird": prepare_thunderbird,
        "spirit": prepare_spirit,
        "liberty": prepare_liberty,
    }

    print(f"[Info] raw_root={raw_root}")
    print(f"[Info] out_root={out_root}")
    print(f"[Info] datasets={datasets}")
    for name in datasets:
        print(f"\n[Info] Preparing {name} ...")
        handlers[name](raw_root, out_root, args)


if __name__ == "__main__":
    main()
