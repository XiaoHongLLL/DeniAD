#!/usr/bin/env python3
"""Build controlled RQ2 type/time/joint anomaly datasets.

The generated dataset keeps train/dev from the source data and replaces test.pkl
with a balanced controlled benchmark.  It also writes rq2_head_train.pkl and
rq2_head_dev.pkl for the supervised RQ2 classification-head protocol:

  rq2_label = 0: normal
  rq2_label = 1: type-only anomaly
  rq2_label = 2: time-only anomaly
  rq2_label = 3: joint type-time anomaly

Binary ``label`` remains 0 for normal events and 1 for injected anomalies, so
the usual anomaly-detection evaluator can still be used.
"""

from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
from collections import Counter
from pathlib import Path


NORMAL_LABELS = {"", "0", "-", "false", "normal", "benign", "ok", "none", "nan"}


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare controlled RQ2 joint type-time benchmark.")
    parser.add_argument("--source_data", default="./data/labeled_bgl", help="Folder with train/dev/test.pkl.")
    parser.add_argument("--out_dir", default="./data/rq2_controlled_bgl", help="Output dataset folder.")
    parser.add_argument("--num_sequences_per_class", type=int, default=1000)
    parser.add_argument(
        "--head_train_sequences_per_class",
        type=int,
        default=600,
        help="Balanced controlled sequences per class for rq2_head_train.pkl.",
    )
    parser.add_argument(
        "--head_dev_sequences_per_class",
        type=int,
        default=200,
        help="Balanced controlled sequences per class for rq2_head_dev.pkl.",
    )
    parser.add_argument("--window_size", type=int, default=12)
    parser.add_argument("--seed", type=int, default=2023)
    parser.add_argument("--max_train_sequences", type=int, default=0)
    parser.add_argument("--max_dev_sequences", type=int, default=0)
    parser.add_argument("--rare_type_pool", type=int, default=32)
    parser.add_argument("--common_type_pool", type=int, default=32)
    parser.add_argument(
        "--joint_mode",
        choices=["conditional", "type_swap", "rare_time"],
        default="conditional",
        help=(
            "conditional keeps the original/common event type and injects a "
            "globally plausible but type-incompatible gap; type_swap keeps the "
            "older common-type swap construction; rare_time keeps the older "
            "rare-type plus time-perturbation construction."
        ),
    )
    return parser.parse_args()


def load_split(root: Path, name: str):
    with (root / f"{name}.pkl").open("rb") as f:
        payload = pickle.load(f, encoding="latin-1")
    return payload["dim_process"], payload[name]


def write_split(out_dir: Path, name: str, dim_process: int, data):
    with (out_dir / f"{name}.pkl").open("wb") as f:
        pickle.dump({"dim_process": dim_process, name: data}, f, protocol=pickle.HIGHEST_PROTOCOL)


def is_anomaly_event(event: dict) -> bool:
    value = event.get("label", event.get("is_anomaly", 0))
    if isinstance(value, str):
        return value.strip().lower() not in NORMAL_LABELS
    return bool(value)


def normal_sequences(*splits):
    seqs = []
    for split in splits:
        for seq in split:
            if len(seq) >= 6 and not any(is_anomaly_event(event) for event in seq):
                seqs.append(seq)
    return seqs


def reset_labels(seq):
    for event in seq:
        event["label"] = 0
        event["raw_label"] = "Normal"
        event["rq2_label"] = 0
        event["anomaly_subtype"] = "normal"
        event["rq2_scenario"] = "normal"


def mark_event(event, class_id: int, class_name: str, scenario: str):
    event["label"] = 1
    event["raw_label"] = class_name
    event["rq2_label"] = class_id
    event["anomaly_subtype"] = class_name
    event["rq2_scenario"] = scenario


def recompute_time(seq):
    timestamp = 0.0
    for idx, event in enumerate(seq):
        if idx == 0:
            event["time_since_start"] = 0.0
            event["time_since_last_event"] = 0.0
            continue
        gap = max(0.0, float(event.get("time_since_last_event", 0.0)))
        timestamp += gap
        event["time_since_start"] = timestamp


def type_statistics(train, common_size: int, rare_size: int):
    counter = Counter(event["type_event"] for seq in train for event in seq)
    ordered = [event_type for event_type, _ in counter.most_common()]
    common = ordered[: max(1, common_size)]
    rare = ordered[-max(1, rare_size):]
    if not common:
        common = [0]
    if not rare:
        rare = common[:]
    return common, rare, counter


def gap_statistics(train):
    gaps = [
        float(event.get("time_since_last_event", 0.0))
        for seq in train
        for event in seq[1:]
        if float(event.get("time_since_last_event", 0.0)) > 0
    ]
    if not gaps:
        return {"median": 1.0, "p95": 1.0, "p99": 1.0}
    gaps_sorted = sorted(gaps)

    def quantile(q):
        idx = min(len(gaps_sorted) - 1, max(0, int(round(q * (len(gaps_sorted) - 1)))))
        return gaps_sorted[idx]

    return {
        "median": quantile(0.50),
        "p05": quantile(0.05),
        "p10": quantile(0.10),
        "p90": quantile(0.90),
        "p95": quantile(0.95),
        "p99": quantile(0.99),
    }


def type_gap_statistics(train, min_count: int = 20):
    per_type = {}
    for seq in train:
        for event in seq[1:]:
            gap = float(event.get("time_since_last_event", 0.0))
            if gap <= 0:
                continue
            per_type.setdefault(int(event["type_event"]), []).append(gap)

    stats = {}
    for event_type, gaps in per_type.items():
        if len(gaps) < min_count:
            continue
        gaps = sorted(gaps)

        def quantile(q):
            idx = min(len(gaps) - 1, max(0, int(round(q * (len(gaps) - 1)))))
            return gaps[idx]

        stats[event_type] = {
            "count": len(gaps),
            "p05": quantile(0.05),
            "p10": quantile(0.10),
            "median": quantile(0.50),
            "p90": quantile(0.90),
            "p95": quantile(0.95),
        }
    return stats


def build_joint_pair_bank(type_gap_stats, gap_stats, common_types):
    candidates = [
        (event_type, stat)
        for event_type, stat in type_gap_stats.items()
        if event_type in set(common_types) and stat["median"] > 0
    ]
    if len(candidates) < 2:
        return {
            "fast": common_types[:1],
            "slow": common_types[-1:],
            "target_types": [event_type for event_type, _ in candidates] or common_types[:],
            "stats": type_gap_stats,
        }

    candidates = sorted(candidates, key=lambda item: item[1]["median"])
    half = max(1, len(candidates) // 4)
    fast = [event_type for event_type, _ in candidates[:half]]
    slow = [event_type for event_type, _ in candidates[-half:]]
    if not fast:
        fast = [candidates[0][0]]
    if not slow:
        slow = [candidates[-1][0]]
    return {
        "fast": fast,
        "slow": slow,
        "target_types": [event_type for event_type, _ in candidates],
        "stats": type_gap_stats,
    }


def clamp_gap(value, gap_stats, lower_key="p05", upper_key="p95"):
    lower = max(float(gap_stats.get(lower_key, gap_stats.get("p05", 0.0))), 1e-9)
    upper = max(float(gap_stats.get(upper_key, gap_stats.get("p95", value))), lower)
    return min(max(float(value), lower), upper)


def clone_base_sequence(pool, rng):
    seq = copy.deepcopy(rng.choice(pool))
    reset_labels(seq)
    return seq


def make_type_anomaly(seq, start, end, rare_types, rng):
    scenario = "type_rare_substitution"
    for idx in range(start, end):
        seq[idx]["type_event"] = int(rng.choice(rare_types))
        mark_event(seq[idx], 1, "type", scenario)


def make_time_anomaly(seq, start, end, gap_stats, rng):
    scenario = rng.choice(["time_long_gap", "time_burst"])
    for idx in range(start, end):
        gap = float(seq[idx].get("time_since_last_event", 0.0))
        if scenario == "time_long_gap":
            seq[idx]["time_since_last_event"] = max(gap * rng.uniform(8.0, 20.0), gap_stats["p99"] * 2.0)
        else:
            seq[idx]["time_since_last_event"] = max(gap * rng.uniform(0.005, 0.03), 1e-9)
        mark_event(seq[idx], 2, "time", scenario)


def choose_gap_mismatch(event_type, gap_stats, joint_bank, rng):
    type_stats = joint_bank.get("stats", {})
    fast_types = [int(t) for t in (joint_bank.get("fast") or [])]
    slow_types = [int(t) for t in (joint_bank.get("slow") or [])]
    event_type = int(event_type)
    stat = type_stats.get(event_type, {})
    global_median = float(gap_stats.get("median", 1.0))
    target_median = float(stat.get("median", global_median))

    if target_median <= global_median:
        donor_pool = [t for t in slow_types if t != event_type] or slow_types
        fallback = max(float(gap_stats.get("p90", global_median)), float(stat.get("p95", target_median)))
        donor_quantile = "p90"
        scenario = "joint_same_type_slow_gap"
    else:
        donor_pool = [t for t in fast_types if t != event_type] or fast_types
        fallback = min(float(gap_stats.get("p10", global_median)), float(stat.get("p05", target_median)))
        donor_quantile = "p10"
        scenario = "joint_same_type_fast_gap"

    donor_stat = {}
    if donor_pool:
        donor_stat = type_stats.get(int(rng.choice(donor_pool)), {})
    donor_gap = float(donor_stat.get(donor_quantile, donor_stat.get("median", fallback)))
    new_gap = donor_gap * rng.uniform(0.95, 1.05)
    return clamp_gap(new_gap, gap_stats, lower_key="p05", upper_key="p95"), scenario


def make_joint_anomaly(seq, start, end, rare_types, gap_stats, joint_bank, rng, mode="conditional"):
    if mode == "rare_time":
        scenario = rng.choice(["joint_rare_type_long_gap", "joint_rare_type_burst"])
        for idx in range(start, end):
            seq[idx]["type_event"] = int(rng.choice(rare_types))
            gap = float(seq[idx].get("time_since_last_event", 0.0))
            if scenario == "joint_rare_type_long_gap":
                seq[idx]["time_since_last_event"] = max(gap * rng.uniform(8.0, 20.0), gap_stats["p99"] * 2.0)
            else:
                seq[idx]["time_since_last_event"] = max(gap * rng.uniform(0.005, 0.03), 1e-9)
            mark_event(seq[idx], 3, "joint", scenario)
        return

    if mode == "type_swap":
        scenario = rng.choice(["joint_fast_type_slow_gap", "joint_slow_type_fast_gap"])
        fast_types = joint_bank.get("fast") or rare_types
        slow_types = joint_bank.get("slow") or rare_types
        type_stats = joint_bank.get("stats", {})
        for idx in range(start, end):
            if scenario == "joint_fast_type_slow_gap":
                target_type = int(rng.choice(fast_types))
                donor_type = int(rng.choice(slow_types))
                donor_stat = type_stats.get(donor_type, {})
                new_gap = donor_stat.get("median", gap_stats["p90"]) * rng.uniform(0.8, 1.2)
            else:
                target_type = int(rng.choice(slow_types))
                donor_type = int(rng.choice(fast_types))
                donor_stat = type_stats.get(donor_type, {})
                new_gap = donor_stat.get("median", gap_stats["p10"]) * rng.uniform(0.8, 1.2)
            seq[idx]["type_event"] = target_type
            seq[idx]["time_since_last_event"] = clamp_gap(new_gap, gap_stats)
            mark_event(seq[idx], 3, "joint", scenario)
        return

    # True joint anomaly: keep the event type marginally normal, and only make
    # its gap incompatible with that type while still plausible globally.
    for idx in range(start, end):
        event_type = int(seq[idx]["type_event"])
        new_gap, scenario = choose_gap_mismatch(event_type, gap_stats, joint_bank, rng)
        seq[idx]["time_since_last_event"] = new_gap
        mark_event(seq[idx], 3, "joint", scenario)


def choose_window(seq, window_size: int, rng: random.Random, preferred_types=None, require_preferred=False):
    n = len(seq)
    length = min(max(2, window_size), n - 1)
    max_start = max(1, n - length)
    preferred = set(int(t) for t in preferred_types or [])
    if preferred:
        for _ in range(64):
            start = rng.randint(1, max_start)
            end = min(n, start + length)
            if all(int(event["type_event"]) in preferred for event in seq[start:end]):
                return start, end
        if require_preferred:
            return None

    start = rng.randint(1, max_start)
    end = min(n, start + length)
    return start, end


def build_controlled_test(pool, rare_types, gap_stats, joint_bank, args, split_seed=None, num_sequences_per_class=None):
    rng = random.Random(args.seed if split_seed is None else split_seed)
    n_per_class = args.num_sequences_per_class if num_sequences_per_class is None else int(num_sequences_per_class)
    if n_per_class <= 0:
        return []
    test = []

    for _ in range(n_per_class):
        seq = clone_base_sequence(pool, rng)
        test.append(seq)

    builders = [
        ("type", lambda seq, start, end: make_type_anomaly(seq, start, end, rare_types, rng)),
        ("time", lambda seq, start, end: make_time_anomaly(seq, start, end, gap_stats, rng)),
        (
            "joint",
            lambda seq, start, end: make_joint_anomaly(
                seq, start, end, rare_types, gap_stats, joint_bank, rng, args.joint_mode
            ),
        ),
    ]
    for name, builder in builders:
        for _ in range(n_per_class):
            preferred = joint_bank.get("target_types", []) if name == "joint" and args.joint_mode == "conditional" else None
            window = None
            for _ in range(64):
                seq = clone_base_sequence(pool, rng)
                window = choose_window(
                    seq,
                    args.window_size,
                    rng,
                    preferred_types=preferred,
                    require_preferred=bool(preferred),
                )
                if window is not None:
                    break
            if window is None:
                seq = clone_base_sequence(pool, rng)
                window = choose_window(seq, args.window_size, rng)
            start, end = window
            builder(seq, start, end)
            recompute_time(seq)
            test.append(seq)

    rng.shuffle(test)
    return test


def stats(test):
    rq2_counter = Counter()
    scenario_counter = Counter()
    segment_counter = Counter()
    for seq in test:
        labels = [int(event.get("rq2_label", -1)) for event in seq]
        if any(label == 3 for label in labels):
            segment_counter["joint"] += 1
        elif any(label == 2 for label in labels):
            segment_counter["time"] += 1
        elif any(label == 1 for label in labels):
            segment_counter["type"] += 1
        else:
            segment_counter["normal"] += 1
        for event in seq:
            rq2_counter[str(event.get("rq2_label", -1))] += 1
            scenario_counter[event.get("rq2_scenario", "none")] += 1
    return {
        "event_rq2_labels": dict(rq2_counter),
        "segment_labels": dict(segment_counter),
        "scenarios": dict(scenario_counter),
    }


def main():
    args = parse_args()
    source = Path(args.source_data)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dim_train, train = load_split(source, "train")
    dim_dev, dev = load_split(source, "dev")
    dim_test, test = load_split(source, "test")
    if not (dim_train == dim_dev == dim_test):
        raise ValueError("dim_process mismatch across source splits.")

    if args.max_train_sequences > 0:
        train = train[:args.max_train_sequences]
    if args.max_dev_sequences > 0:
        dev = dev[:args.max_dev_sequences]

    pool = normal_sequences(train, dev, test)
    if not pool:
        raise RuntimeError("No normal source sequences are available for controlled RQ2 construction.")

    common_types, rare_types, type_counter = type_statistics(train, args.common_type_pool, args.rare_type_pool)
    gap_stats = gap_statistics(train)
    type_gap_stats = type_gap_statistics(train)
    joint_bank = build_joint_pair_bank(type_gap_stats, gap_stats, common_types)
    controlled_test = build_controlled_test(pool, rare_types, gap_stats, joint_bank, args)
    head_train = build_controlled_test(
        pool,
        rare_types,
        gap_stats,
        joint_bank,
        args,
        split_seed=args.seed + 101,
        num_sequences_per_class=args.head_train_sequences_per_class,
    )
    head_dev = build_controlled_test(
        pool,
        rare_types,
        gap_stats,
        joint_bank,
        args,
        split_seed=args.seed + 202,
        num_sequences_per_class=args.head_dev_sequences_per_class,
    )

    write_split(out_dir, "train", dim_train, train)
    write_split(out_dir, "dev", dim_train, dev)
    write_split(out_dir, "test", dim_train, controlled_test)
    write_split(out_dir, "rq2_head_train", dim_train, head_train)
    write_split(out_dir, "rq2_head_dev", dim_train, head_dev)

    metadata = {
        "dataset": "RQ2 controlled type-time anomaly",
        "source_data": str(source),
        "dim_process": dim_train,
        "num_sequences_per_class": args.num_sequences_per_class,
        "window_size": args.window_size,
        "seed": args.seed,
        "train_sequences": len(train),
        "dev_sequences": len(dev),
        "test_sequences": len(controlled_test),
        "head_train_sequences": len(head_train),
        "head_dev_sequences": len(head_dev),
        "head_train_sequences_per_class": args.head_train_sequences_per_class,
        "head_dev_sequences_per_class": args.head_dev_sequences_per_class,
        "common_types": common_types,
        "rare_types": rare_types,
        "joint_mode": args.joint_mode,
        "joint_fast_types": joint_bank.get("fast", []),
        "joint_slow_types": joint_bank.get("slow", []),
        "gap_stats": gap_stats,
        "type_count_total": int(sum(type_counter.values())),
        "stats": stats(controlled_test),
        "head_train_stats": stats(head_train),
        "head_dev_stats": stats(head_dev),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[OK] Wrote {out_dir}")
    print(json.dumps(metadata["stats"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
