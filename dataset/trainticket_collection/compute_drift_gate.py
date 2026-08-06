#!/usr/bin/env python3
"""Compute frozen model-independent drift gates for pilot runs."""

from __future__ import annotations

import argparse
import math
from collections import Counter, defaultdict
from pathlib import Path

from common import (
    counts,
    event_key,
    filter_events,
    gaps,
    iter_logs,
    js_divergence,
    load_manifest,
    mmd_single,
    parse_time,
    post_events,
    pre_events,
    quantile,
    row_run_dir,
    sliding_time_windows,
    time_bounds,
    transitions,
    vector_from_counts,
    wasserstein_1d,
    write_csv,
    write_json,
)


GATE_VERSION = "drift-gate-v0.6-family-vote-strong-family-guard"

METRICS = [
    "template_js",
    "service_macro_template_js",
    "access_template_js",
    "native_template_js",
    "max_service_template_js",
    "transition_js",
    "time_wasserstein",
    "oov_rate_change",
    "template_mmd",
]

METRIC_FAMILIES = {
    "frequency": [
        "template_js",
        "service_macro_template_js",
        "access_template_js",
        "native_template_js",
        "max_service_template_js",
        "template_mmd",
    ],
    "transition": ["transition_js"],
    "timing": ["time_wasserstein"],
    "novelty": ["oov_rate_change"],
}

STRONG_FAMILY_MIN_METRICS = {
    "frequency": 2,
    "transition": 1,
    "timing": 1,
    "novelty": 1,
}


def template_only_counts(events: list[dict]) -> Counter:
    return counts([event.get("_template") or event_key(event) for event in events])


def service_template_js(pre: list[dict], window: list[dict]) -> tuple[float | None, float | None]:
    services = sorted({event.get("_service") or "unknown" for event in pre + window})
    values = []
    for service in services:
        pre_service = [event for event in pre if (event.get("_service") or "unknown") == service]
        window_service = [event for event in window if (event.get("_service") or "unknown") == service]
        if not pre_service and not window_service:
            continue
        values.append(js_divergence(template_only_counts(pre_service), template_only_counts(window_service)))
    if not values:
        return None, None
    return sum(values) / len(values), max(values)


def source_template_js(pre: list[dict], window: list[dict], source: str) -> float | None:
    pre_source = [event for event in pre if str(event.get("log_source") or "unknown") == source]
    window_source = [event for event in window if str(event.get("log_source") or "unknown") == source]
    if not pre_source and not window_source:
        return None
    return js_divergence(template_only_counts(pre_source), template_only_counts(window_source))


def metric_rows_for_run(row: dict, raw_runs: Path, window_seconds: float, step_seconds: float) -> list[dict]:
    run_dir = row_run_dir(row, raw_runs)
    events = iter_logs(run_dir)
    if len(events) < 2:
        return []

    pre = pre_events(row, events)
    post = post_events(row, events)
    if len(pre) < 2 or len(post) < 2:
        return []

    pre_keys = [event_key(event) for event in pre]
    pre_counts = counts(pre_keys)
    pre_transitions = transitions(pre_keys)
    pre_gaps = gaps(pre)
    pre_templates = set(pre_keys)

    pre_windows = sliding_time_windows(pre, window_seconds, step_seconds)
    post_windows = sliding_time_windows(post, window_seconds, step_seconds)
    vocab = sorted(set(pre_keys) | {event_key(event) for event in post})
    pre_vectors = [
        vector_from_counts(counts([event_key(event) for event in window]), vocab)
        for window in pre_windows
        if len(window) >= 2
    ]

    rows = []
    for index, window in enumerate(post_windows):
        if len(window) < 2:
            continue
        keys = [event_key(event) for event in window]
        c = counts(keys)
        t = transitions(keys)
        window_gaps = gaps(window)
        new_rate = sum(1 for key in keys if key not in pre_templates) / len(keys)
        vector = vector_from_counts(c, vocab)
        service_macro_js, max_service_js = service_template_js(pre, window)
        payload = {
            "run_id": row["run_id"],
            "window_index": index,
            "window_start_time": window[0]["_timestamp"],
            "window_end_time": window[-1]["_timestamp"],
            "template_js": js_divergence(pre_counts, c),
            "service_macro_template_js": service_macro_js,
            "access_template_js": source_template_js(pre, window, "access"),
            "native_template_js": source_template_js(pre, window, "application_native"),
            "max_service_template_js": max_service_js,
            "transition_js": js_divergence(pre_transitions, t),
            "time_wasserstein": wasserstein_1d(pre_gaps, window_gaps),
            "oov_rate_change": new_rate,
            "template_mmd": mmd_single(pre_vectors, vector),
        }
        rows.append(payload)
    return rows


def is_control(row: dict) -> bool:
    role = str(row.get("benchmark_role") or "").lower()
    return role in {"baseline_normal", "no_op_control"}


def finite_float(value) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def metric_exceeds(row: dict, metric: str, thresholds: dict) -> bool:
    value = finite_float(row.get(metric))
    threshold = finite_float(thresholds.get(metric))
    if value is None or threshold is None:
        return False
    return value > threshold


def family_exceedance(row: dict, thresholds: dict) -> dict[str, list[str]]:
    exceeded = {}
    for family, metrics in METRIC_FAMILIES.items():
        hits = [metric for metric in metrics if metric_exceeds(row, metric, thresholds)]
        if hits:
            exceeded[family] = hits
    return exceeded


def annotate_gate_rows(rows: list[dict], thresholds: dict, strong_thresholds: dict) -> None:
    for row in rows:
        ordinary = family_exceedance(row, thresholds)
        strong_raw = family_exceedance(row, strong_thresholds)
        strong = {
            family: hits
            for family, hits in strong_raw.items()
            if len(hits) >= STRONG_FAMILY_MIN_METRICS.get(family, 1)
        }
        for family in METRIC_FAMILIES:
            row[f"{family}_family_flag"] = int(family in ordinary)
            row[f"{family}_family_exceed"] = int(family in ordinary)
            row[f"{family}_family_metrics"] = ";".join(ordinary.get(family, []))
            row[f"strong_{family}_family_flag"] = int(family in strong)
            row[f"strong_{family}_family_exceed"] = int(family in strong)
            row[f"strong_{family}_family_metrics"] = ";".join(strong.get(family, []))
            row[f"strong_{family}_raw_metric_hits"] = len(strong_raw.get(family, []))
        row["ordinary_exceeded_families"] = ";".join(sorted(ordinary))
        row["strong_exceeded_families"] = ";".join(sorted(strong))
        row["family_vote_count"] = len(ordinary)
        row["strong_family_vote_count"] = len(strong)
        row["any_family_exceeded"] = int(bool(ordinary or strong))


def sustained_span(rows: list[dict], flag_key: str, min_votes: int, sustain_windows: int) -> tuple[int | None, int | None, int | None]:
    run = 0
    first_pos = None
    for pos, row in enumerate(rows):
        flag = int(row.get(flag_key) or 0) >= min_votes
        run = run + 1 if flag else 0
        if flag and first_pos is None:
            first_pos = pos
        if run >= sustain_windows:
            start_pos = pos - sustain_windows + 1
            return first_pos, start_pos, pos
    return first_pos, None, None


def evaluation_start(row: dict) -> float | None:
    return (
        parse_time(row.get("baseline_evaluation_start_time"))
        or parse_time(row.get("observable_impact_start_time"))
        or parse_time(row.get("post_change_observation_start_time"))
        or parse_time(row.get("failure_trigger_time"))
        or parse_time(row.get("deployment_failed_time"))
        or parse_time(row.get("stable_state_start_time"))
        or parse_time(row.get("change_start_time"))
    )


def baseline_overlap_event_count(row: dict, raw_runs: Path) -> int | str:
    if str(row.get("benchmark_role") or "").lower() != "baseline_normal":
        return ""
    events = iter_logs(row_run_dir(row, raw_runs))
    if not events:
        return 0
    reference_ids = {event.get("_line_id") for event in pre_events(row, events)}
    evaluation_ids = {event.get("_line_id") for event in post_events(row, events)}
    return len(reference_ids & evaluation_ids)


def gate_for_rows(
    rows: list[dict],
    sustain_windows: int,
    min_families: int,
    strong_sustain_windows: int,
    min_strong_families: int,
) -> dict:
    first_ordinary, ordinary_start, ordinary_confirmed = sustained_span(
        rows,
        "family_vote_count",
        min_families,
        sustain_windows,
    )
    first_strong, strong_start, strong_confirmed = sustained_span(
        rows,
        "strong_family_vote_count",
        min_strong_families,
        strong_sustain_windows,
    )
    first_positions = [pos for pos in (first_ordinary, first_strong) if pos is not None]
    first_exceed = min(first_positions) if first_positions else None

    candidates = []
    if ordinary_confirmed is not None:
        candidates.append(("ordinary_family_vote", ordinary_start, ordinary_confirmed))
    if strong_confirmed is not None:
        candidates.append(("strong_single_family", strong_start, strong_confirmed))
    if not candidates:
        return {
            "drift_gate_pass": False,
            "gate_mode": "",
            "first_exceed_pos": first_exceed,
            "sustained_start_pos": None,
            "confirmed_pos": None,
            "ordinary_start_pos": ordinary_start,
            "ordinary_confirmed_pos": ordinary_confirmed,
            "strong_start_pos": strong_start,
            "strong_confirmed_pos": strong_confirmed,
        }
    gate_mode, sustained_start, confirmed = min(candidates, key=lambda item: item[2])
    return {
        "drift_gate_pass": True,
        "gate_mode": gate_mode,
        "first_exceed_pos": first_exceed,
        "sustained_start_pos": sustained_start,
        "confirmed_pos": confirmed,
        "ordinary_start_pos": ordinary_start,
        "ordinary_confirmed_pos": ordinary_confirmed,
        "strong_start_pos": strong_start,
        "strong_confirmed_pos": strong_confirmed,
    }


def row_window_index(rows: list[dict], pos: int | None):
    if pos is None or pos < 0 or pos >= len(rows):
        return ""
    return rows[pos].get("window_index", "")


def row_window_end(rows: list[dict], pos: int | None) -> float | None:
    if pos is None or pos < 0 or pos >= len(rows):
        return None
    return finite_float(rows[pos].get("window_end_time"))


def benchmark_label(row: dict, pass_gate: bool) -> str:
    role = str(row.get("benchmark_role") or "").lower()
    semantic = str(row.get("semantic_label") or row.get("oracle_semantic_label") or row.get("declared_semantic_label") or "").lower()
    if role in {"baseline_normal", "no_op_control"}:
        return role
    if semantic == "expected":
        return "expected_drift" if pass_gate else "successful_no_drift"
    if semantic == "unexpected":
        return "unexpected_drift" if pass_gate else "unexpected_without_observable_log_drift"
    if semantic == "indeterminate":
        return "indeterminate_drift" if pass_gate else "indeterminate_no_drift"
    return "unlabeled"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw_runs", default="raw_runs")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--window_seconds", type=float, default=300.0)
    parser.add_argument("--step_seconds", type=float, default=60.0)
    parser.add_argument("--threshold_quantile", type=float, default=0.95)
    parser.add_argument("--strong_threshold_quantile", type=float, default=0.99)
    parser.add_argument("--sustain_windows", type=int, default=3)
    parser.add_argument("--min_families_per_window", type=int, default=2)
    parser.add_argument("--strong_sustain_windows", type=int, default=5)
    parser.add_argument("--min_strong_families_per_window", type=int, default=1)
    args = parser.parse_args()

    raw_runs = Path(args.raw_runs)
    rows = load_manifest(Path(args.manifest))
    per_run_windows = {}
    for row in rows:
        metrics = metric_rows_for_run(row, raw_runs, args.window_seconds, args.step_seconds)
        per_run_windows[row["run_id"]] = metrics

    control_values = defaultdict(list)
    for row in rows:
        if not is_control(row):
            continue
        for item in per_run_windows.get(row["run_id"], []):
            for metric in METRICS:
                value = finite_float(item.get(metric))
                if value is not None:
                    control_values[metric].append(value)

    thresholds = {
        metric: quantile(control_values[metric], args.threshold_quantile)
        for metric in METRICS
    }
    strong_thresholds = {
        metric: quantile(control_values[metric], args.strong_threshold_quantile)
        for metric in METRICS
    }

    all_window_rows = []
    for row in rows:
        annotate_gate_rows(per_run_windows.get(row["run_id"], []), thresholds, strong_thresholds)
        for item in per_run_windows.get(row["run_id"], []):
            full = dict(row)
            full.update(item)
            full["is_control"] = int(is_control(row))
            full["drift_gate_version"] = GATE_VERSION
            all_window_rows.append(full)

    summary = []
    for row in rows:
        run_rows = per_run_windows.get(row["run_id"], [])
        gate = gate_for_rows(
            run_rows,
            sustain_windows=args.sustain_windows,
            min_families=args.min_families_per_window,
            strong_sustain_windows=args.strong_sustain_windows,
            min_strong_families=args.min_strong_families_per_window,
        )
        pass_gate = gate["drift_gate_pass"]
        confirmed_at = row_window_end(run_rows, gate["confirmed_pos"])
        start_time = evaluation_start(row)
        delay_seconds = ""
        if confirmed_at is not None and start_time is not None:
            delay_seconds = max(0.0, confirmed_at - start_time)
        summary_row = dict(row)
        summary_row["drift_gate_pass"] = int(pass_gate)
        summary_row["drift_gate_version"] = GATE_VERSION
        summary_row["drift_gate_first_window"] = row_window_index(run_rows, gate["sustained_start_pos"])
        summary_row["first_exceed_window"] = row_window_index(run_rows, gate["first_exceed_pos"])
        summary_row["sustained_gate_start_window"] = row_window_index(run_rows, gate["sustained_start_pos"])
        summary_row["gate_confirmed_window"] = row_window_index(run_rows, gate["confirmed_pos"])
        summary_row["ordinary_sustained_start_window"] = row_window_index(run_rows, gate["ordinary_start_pos"])
        summary_row["ordinary_confirmed_window"] = row_window_index(run_rows, gate["ordinary_confirmed_pos"])
        summary_row["strong_sustained_start_window"] = row_window_index(run_rows, gate["strong_start_pos"])
        summary_row["strong_confirmed_window"] = row_window_index(run_rows, gate["strong_confirmed_pos"])
        summary_row["gate_mode"] = gate["gate_mode"]
        summary_row["gate_confirmed_at"] = "" if confirmed_at is None else confirmed_at
        summary_row["detection_delay_seconds"] = delay_seconds
        summary_row["drift_windows_evaluated"] = len(run_rows)
        summary_row["reference_evaluation_overlap_event_count"] = baseline_overlap_event_count(row, raw_runs)
        summary_row["benchmark_label"] = benchmark_label(row, pass_gate)
        summary.append(summary_row)

    output_dir = Path(args.output_dir)
    write_csv(output_dir / "drift_gate_windows.csv", all_window_rows)
    write_csv(output_dir / "drift_gate_summary.csv", summary)
    write_json(output_dir / "drift_gate_thresholds.json", {
        "gate_version": GATE_VERSION,
        "metrics": METRICS,
        "metric_families": METRIC_FAMILIES,
        "strong_family_min_metrics": STRONG_FAMILY_MIN_METRICS,
        "threshold_quantile": args.threshold_quantile,
        "strong_threshold_quantile": args.strong_threshold_quantile,
        "thresholds": thresholds,
        "strong_thresholds": strong_thresholds,
        "control_value_counts": {metric: len(values) for metric, values in control_values.items()},
        "window_seconds": args.window_seconds,
        "step_seconds": args.step_seconds,
        "sustain_windows": args.sustain_windows,
        "min_families_per_window": args.min_families_per_window,
        "strong_sustain_windows": args.strong_sustain_windows,
        "min_strong_families_per_window": args.min_strong_families_per_window,
        "calibration_status": "pilot_trial_thresholds; controls are not yet an independent validation set",
    })

    counts_by_label = Counter(row["benchmark_label"] for row in summary)
    print({
        "runs": len(summary),
        "window_rows": len(all_window_rows),
        "benchmark_counts": dict(counts_by_label),
        "output_dir": args.output_dir,
    })


if __name__ == "__main__":
    main()
