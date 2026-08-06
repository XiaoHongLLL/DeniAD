#!/usr/bin/env python3
"""Slice logs.jsonl by phase-timeline timestamps."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


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


def read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--start_key", default="post_change_observation_start_time")
    parser.add_argument("--end_key", default="observation_end_time")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    input_path = Path(args.input) if args.input else run_dir / "logs.jsonl"
    output_path = Path(args.output) if args.output else run_dir / "post_change_logs.jsonl"
    timeline = read_json(run_dir / "phase_timeline.json")
    start = parse_rfc3339(timeline.get(args.start_key))
    if not start and args.start_key == "post_change_observation_start_time":
        start = (
            parse_rfc3339(timeline.get("stable_state_start_time"))
            or parse_rfc3339(timeline.get("deployment_failed_time"))
            or parse_rfc3339(timeline.get("failure_trigger_time"))
        )
    end = parse_rfc3339(timeline.get(args.end_key) or timeline.get("run_end_time"))
    if not start or not end:
        raise SystemExit(f"Missing phase bounds: {args.start_key}, {args.end_key}")

    kept = 0
    scanned = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with input_path.open("r", encoding="utf-8", errors="replace") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            scanned += 1
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = parse_rfc3339(item.get("timestamp_utc") or item.get("timestamp"))
            if ts and start <= ts <= end:
                dst.write(json.dumps(item, ensure_ascii=False) + "\n")
                kept += 1
    print(json.dumps({
        "input": str(input_path),
        "output": str(output_path),
        "start": timeline.get(args.start_key),
        "end": timeline.get(args.end_key) or timeline.get("run_end_time"),
        "scanned": scanned,
        "kept": kept,
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
