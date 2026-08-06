import argparse
import csv
import json
import math
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from context_conditioned_absence_memory import (
    build_context_conditioned_absence_evidence,
    load_memory_sequences,
)


PRED_NAMES = {
    0: "Normal",
    1: "Expected",
    2: "Unexpected",
    3: "Reject",
}

STATE_STRONG_UNEXPECTED_TOKENS = {
    "__STATE_DEPLOYMENTS_AVAILABLE::false",
    "__STATE_ANY_DEPLOYMENT_UNAVAILABLE::true",
    "__STATE_TARGET_READY::unready",
    "__STATE_TARGET_AVAILABLE::zero_available",
    "__STATE_TARGET_AVAILABLE::zero_replicas",
}


def load_split(data_dir, split):
    with (Path(data_dir) / f"{split}.pkl").open("rb") as f:
        obj = pickle.load(f, encoding="latin-1")
    return obj[split]


def load_annotation(data_dir):
    path = Path(data_dir) / "annotation.csv"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {row["run_id"]: row for row in csv.DictReader(f)}


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def safe_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_text(value):
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "<missing>", "unknown"}:
        return ""
    return text


def split_service_list(value):
    text = safe_text(value)
    if not text:
        return []
    out = []
    for chunk in text.replace(";", ",").replace("|", ",").split(","):
        item = chunk.strip()
        if item:
            out.append(item)
    return out


def service_allowed(service, args):
    service = safe_text(service)
    if not service:
        return False
    lowered = service.lower()
    if lowered.startswith("__state_") or lowered.startswith("__log_"):
        return False
    excluded = {x.strip() for x in args.absence_exclude_services.split(",") if x.strip()}
    if service in excluded:
        return False
    prefixes = [x.strip() for x in args.absence_service_prefixes.split(",") if x.strip()]
    if prefixes and not any(service.startswith(prefix) for prefix in prefixes):
        return False
    return True


def event_service(event):
    for key in ("service", "sequence_service", "component_id"):
        service = safe_text(event.get(key))
        if service:
            return service
    return ""


def is_normal_reference(event):
    label = event.get("label", event.get("is_anomaly", 0))
    try:
        if int(float(label)) != 0:
            return False
    except (TypeError, ValueError):
        if str(label).strip().lower() not in {"0", "false", "normal", "benign", "ok"}:
            return False
    drift = str(event.get("drift_label", "normal")).strip().lower().replace("-", "_")
    return drift not in {"unexpected", "unexpected_drift", "reject", "rejected"}


def aggregate_run_counts(data, args, normal_only=False):
    runs = defaultdict(lambda: {"counts": Counter(), "meta": {}, "events": 0, "normal": True})
    for seq in data:
        if not seq:
            continue
        first = seq[0]
        run_id = safe_text(first.get("run_id")) or safe_text(first.get("sequence_id"))
        if not run_id:
            continue
        rec = runs[run_id]
        if not rec["meta"]:
            rec["meta"] = dict(first)
        for event in seq:
            if normal_only and not is_normal_reference(event):
                rec["normal"] = False
            service = event_service(event)
            if not service_allowed(service, args):
                continue
            rec["counts"][service] += 1
            rec["events"] += 1
    if normal_only:
        return {
            run_id: rec for run_id, rec in runs.items()
            if rec["normal"] and rec["events"] > 0
        }
    return {
        run_id: rec for run_id, rec in runs.items()
        if rec["events"] > 0
    }


def metadata_expected_services(meta, args):
    ignored = {"none", "all", "all-observed-services", "workload-generator", "system-observability"}
    services = []
    seen = set()
    fields = [x.strip() for x in args.absence_metadata_fields.split(",") if x.strip()]
    for field in fields:
        for service in split_service_list(meta.get(field)):
            if service.lower() in ignored or not service_allowed(service, args):
                continue
            if service not in seen:
                seen.add(service)
                services.append(service)
    return services


def cosine_log_counts(counts, ref_counts, services):
    q = [math.log1p(counts.get(s, 0.0)) for s in services]
    r = [math.log1p(ref_counts.get(s, 0.0)) for s in services]
    qn = math.sqrt(sum(x * x for x in q))
    rn = math.sqrt(sum(x * x for x in r))
    if qn <= 1e-8 or rn <= 1e-8:
        return 0.0
    return sum(x * y for x, y in zip(q, r)) / (qn * rn)


def build_absence_evidence(reference_data, eval_data, args):
    if args.absence_context_mode == "context_memory":
        return build_context_conditioned_absence_evidence(
            reference_data,
            eval_data,
            args,
        )
    ref_runs = aggregate_run_counts(reference_data, args, normal_only=True)
    eval_runs = aggregate_run_counts(eval_data, args, normal_only=False)
    services = sorted({s for rec in ref_runs.values() for s in rec["counts"]})
    if not ref_runs or not services:
        return {}, {"reference_runs": 0, "reference_services": 0, "eval_runs": len(eval_runs)}

    ref_items = sorted(ref_runs.items())
    evidence = {}
    conflict_runs = 0
    absence_values = []
    for run_id, rec in eval_runs.items():
        sims = [
            (cosine_log_counts(rec["counts"], ref_rec["counts"], services), ref_id, ref_rec)
            for ref_id, ref_rec in ref_items
        ]
        sims.sort(reverse=True, key=lambda x: x[0])
        neighbors = sims[:max(1, min(args.absence_k, len(sims)))]
        nn_cosine = neighbors[0][0] if neighbors else 0.0

        metadata_services = metadata_expected_services(rec["meta"], args)
        metadata_known = [s for s in metadata_services if s in services]
        memory_services = []
        for service in services:
            values = [nbr[2]["counts"].get(service, 0.0) for nbr in neighbors]
            if not values:
                continue
            active_prob = sum(1 for v in values if v > 0) / len(values)
            mu = sum(values) / len(values)
            if active_prob >= args.absence_active_beta and mu >= args.absence_min_expected_count:
                memory_services.append(service)
        if args.absence_context_mode == "metadata":
            expected_services = metadata_known
        elif args.absence_context_mode == "memory":
            expected_services = memory_services
        else:
            expected_services = metadata_known if metadata_known else memory_services

        known = []
        silenced = []
        service_scores = {}
        max_absence = 0.0
        for service in expected_services:
            if service not in services:
                continue
            values = [nbr[2]["counts"].get(service, 0.0) for nbr in neighbors]
            if not values:
                continue
            mu = sum(values) / len(values)
            if mu < args.absence_min_expected_count:
                continue
            var = sum((v - mu) ** 2 for v in values) / len(values)
            sigma = math.sqrt(var)
            observed = rec["counts"].get(service, 0.0)
            floor = max(sigma, mu * args.absence_sigma_floor_ratio, 1.0)
            score = max(0.0, (mu - observed) / floor)
            ratio_low = observed <= mu * args.absence_count_ratio_threshold
            if not ratio_low:
                score = 0.0
            known.append(service)
            service_scores[service] = {
                "observed": observed,
                "expected": mu,
                "score": score,
                "ratio_low": ratio_low,
            }
            max_absence = max(max_absence, score)
            if score >= args.absence_anomaly_threshold:
                silenced.append(service)

        coverage_support = 1.0 - len(silenced) / max(len(known), 1) if known else max(0.0, min(1.0, nn_cosine))
        absence_conflict = bool(silenced)
        coverage_conflict = coverage_support < args.absence_coverage_threshold
        if absence_conflict or coverage_conflict:
            conflict_runs += 1
        absence_values.append(max_absence)
        evidence[run_id] = {
            "absence_anomaly": max_absence,
            "coverage_support": coverage_support,
            "coverage_nn_cosine": max(0.0, min(1.0, nn_cosine)),
            "absence_conflict": absence_conflict,
            "coverage_conflict": coverage_conflict,
            "expected_services": ",".join(expected_services),
            "known_expected_services": ",".join(known),
            "silenced_services": ",".join(silenced),
            "service_scores": service_scores,
        }
    summary = {
        "reference_runs": len(ref_runs),
        "reference_services": len(services),
        "eval_runs": len(eval_runs),
        "conflict_runs": conflict_runs,
        "mean_absence_anomaly": sum(absence_values) / len(absence_values) if absence_values else 0.0,
        "max_absence_anomaly": max(absence_values) if absence_values else 0.0,
        "context_mode": args.absence_context_mode,
    }
    return evidence, summary


def f1(tp, fp, fn):
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    score = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, score


def run_fractions(counter):
    total = sum(counter.values())
    if total <= 0:
        return {
            "total": 0,
            "normal": 0.0,
            "expected": 0.0,
            "unexpected": 0.0,
            "reject": 0.0,
        }
    return {
        "total": total,
        "normal": counter[0] / total,
        "expected": counter[1] / total,
        "unexpected": counter[2] / total,
        "reject": counter[3] / total,
    }


def absence_is_low_risk(absence_st):
    if not absence_st:
        return True
    return not bool(
        absence_st.get("absence_conflict", False)
        or absence_st.get("coverage_conflict", False)
    )


def strong_absence_unexpected(absence_st, args):
    if args.absence_unexpected_mode != "strong" or not absence_st:
        return False
    anomaly = float(absence_st.get("absence_anomaly", 0.0) or 0.0)
    coverage = float(absence_st.get("coverage_support", 1.0) or 1.0)
    silenced = safe_text(absence_st.get("silenced_services"))
    return (
        anomaly >= args.absence_strong_anomaly_threshold
        and bool(silenced)
        and coverage <= args.absence_strong_coverage_threshold
    )


def decide_run_reject_first(counter, expected_min, unexpected_min, reject_min, conflict_min):
    frac = run_fractions(counter)
    if frac["total"] <= 0:
        return 0, "empty"
    expected_frac = frac["expected"]
    unexpected_frac = frac["unexpected"]
    reject_frac = frac["reject"]
    if reject_frac >= reject_min:
        return 3, "reject_first"
    if expected_frac >= conflict_min and unexpected_frac >= conflict_min:
        return 3, "expected_unexpected_conflict"
    if unexpected_frac >= unexpected_min:
        return 2, "unexpected_fraction"
    if expected_frac >= expected_min:
        return 1, "expected_fraction"
    return 0, "insufficient_evidence"


def decide_run_evidence_priority(counter, args, absence_st=None):
    frac = run_fractions(counter)
    if frac["total"] <= 0:
        return 0, "empty"
    expected_frac = frac["expected"]
    unexpected_frac = frac["unexpected"]
    reject_frac = frac["reject"]
    reject_margin_value = reject_frac - expected_frac

    if strong_absence_unexpected(absence_st, args):
        return 2, "strong_absence_unexpected"

    strong_unexpected = (
        unexpected_frac >= args.strong_unexpected_min
        or (
            unexpected_frac >= args.unexpected_min
            and unexpected_frac - expected_frac >= args.unexpected_margin
        )
    )
    if strong_unexpected:
        return 2, "strong_unexpected_fraction"

    if expected_frac >= args.conflict_min and unexpected_frac >= args.conflict_min:
        return 3, "expected_unexpected_conflict"

    safe_expected = (
        expected_frac >= args.expected_min
        and unexpected_frac <= args.expected_unexpected_max
        and absence_is_low_risk(absence_st)
    )
    if safe_expected:
        return 1, "safe_expected_support"

    margin_reject = (
        reject_frac >= args.reject_min
        and reject_margin_value >= args.reject_margin
    )
    if margin_reject:
        return 3, "reject_margin"

    if unexpected_frac >= args.unexpected_min:
        return 2, "unexpected_fraction"
    if expected_frac >= args.expected_min and absence_is_low_risk(absence_st):
        return 1, "expected_fraction"
    return 0, "insufficient_evidence"


def decide_run(counter, args, absence_st=None):
    if args.decision_policy == "evidence_priority":
        return decide_run_evidence_priority(counter, args, absence_st)
    return decide_run_reject_first(
        counter,
        args.expected_min,
        args.unexpected_min,
        args.reject_min,
        args.conflict_min,
    )


def load_state_token_names(data_dir):
    path = Path(data_dir) / "state_aware_tokens.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        token_to_id = json.load(f)
    return {int(v): k for k, v in token_to_id.items()}


def event_token_name(event, id_to_state_token):
    template = str(event.get("event_template", ""))
    if template.startswith("__STATE_") or template.startswith("__LOG_"):
        return template
    return id_to_state_token.get(safe_int(event.get("type_event"), -1), "")


def build_state_evidence(data, data_dir):
    id_to_state_token = load_state_token_names(data_dir)
    evidence = defaultdict(lambda: {
        "state_strong_unexpected": False,
        "state_unexpected_tokens": [],
    })
    for seq in data:
        if not seq:
            continue
        first = seq[0]
        if first.get("sequence_service") != "system-observability":
            continue
        run_id = first.get("run_id", "")
        if not run_id:
            continue
        st = evidence[run_id]
        for event in seq:
            token = event_token_name(event, id_to_state_token)
            if not token:
                continue
            if token in STATE_STRONG_UNEXPECTED_TOKENS or token.startswith("__STATE_UNAVAILABLE_SERVICE::"):
                st["state_strong_unexpected"] = True
                st["state_unexpected_tokens"].append(token)
    return evidence


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate saved RQ4 event details into run-level metrics."
    )
    parser.add_argument("--data_dir", required=True, help="Directory containing test.pkl and annotation.csv.")
    parser.add_argument("--events_csv", required=True, help="*_rq4_events.csv produced by main.py.")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--output_prefix", required=True)
    parser.add_argument(
        "--event_pred_column",
        choices=[
            "pred_drift_id",
            "raw_pred_drift_id",
            "score_threshold",
            "final_score_threshold",
            "score_selective",
            "final_score_selective",
            "revised_selective",
        ],
        default="pred_drift_id",
        help=(
            "Event-level prediction source to aggregate. Use score_threshold "
            "for a score-only OOD base detector, raw_pred_drift_id for raw "
            "diagnosis, and pred_drift_id for revised diagnosis. The selective "
            "score modes map low-score/low-uncertainty events to Expected, "
            "high-score events to Unexpected, and high-uncertainty events to Reject. "
            "revised_selective preserves non-Normal revised decisions and applies "
            "that selective mapping only to unrevised Normal events."
        ),
    )
    parser.add_argument("--expected_min", type=float, default=0.50)
    parser.add_argument("--unexpected_min", type=float, default=0.25)
    parser.add_argument("--reject_min", type=float, default=0.20)
    parser.add_argument("--conflict_min", type=float, default=0.20)
    parser.add_argument(
        "--decision_policy",
        choices=["reject_first", "evidence_priority"],
        default="reject_first",
        help=(
            "Run-level aggregation policy. 'reject_first' preserves the original "
            "conservative rule. 'evidence_priority' first checks strong unexpected "
            "evidence, then safe Expected support, and rejects only under evidence "
            "conflict or sufficient reject margin."
        ),
    )
    parser.add_argument(
        "--reject_margin",
        type=float,
        default=0.0,
        help="For evidence_priority, require reject_frac - expected_frac to exceed this margin before Reject.",
    )
    parser.add_argument(
        "--unexpected_margin",
        type=float,
        default=0.0,
        help="For evidence_priority, require unexpected_frac - expected_frac for strong unexpected evidence.",
    )
    parser.add_argument(
        "--strong_unexpected_min",
        type=float,
        default=0.50,
        help="For evidence_priority, treat unexpected_frac above this value as strong Unexpected evidence.",
    )
    parser.add_argument(
        "--expected_unexpected_max",
        type=float,
        default=0.05,
        help="For evidence_priority, safe Expected acceptance requires unexpected_frac at or below this value.",
    )
    parser.add_argument(
        "--state_veto",
        choices=["auto", "off"],
        default="auto",
        help=(
            "Use state-aware system-observability sequences as a run-level Unexpected veto "
            "when objective deployment/target-unavailable evidence is present."
        ),
    )
    parser.add_argument(
        "--absence_veto",
        choices=["off", "reject"],
        default="off",
        help=(
            "Apply log-derived absence-aware rejection at run level when the "
            "model vote is Expected but expected service logs are silently missing."
        ),
    )
    parser.add_argument(
        "--absence_apply_to",
        choices=["expected", "normal_expected"],
        default="normal_expected",
        help=(
            "Votes that can be abstained by absence evidence. 'expected' only "
            "blocks unsafe Expected acceptance; 'normal_expected' also blocks "
            "silent-failure runs that otherwise aggregate to Normal."
        ),
    )
    parser.add_argument(
        "--absence_context_mode",
        choices=["context_memory", "hybrid", "metadata", "memory"],
        default="context_memory",
    )
    parser.add_argument(
        "--absence_reference_path",
        default="",
        help="Standalone absence_memory.pkl or its containing directory.",
    )
    parser.add_argument("--absence_metadata_fields", default="")
    parser.add_argument("--absence_exclude_services", default="system-observability,tsdb-mysql,nacosdb-mysql")
    parser.add_argument("--absence_service_prefixes", default="")
    parser.add_argument("--absence_k", type=int, default=20)
    parser.add_argument("--absence_active_beta", type=float, default=0.70)
    parser.add_argument("--absence_min_expected_count", type=float, default=20.0)
    parser.add_argument("--absence_count_ratio_threshold", type=float, default=0.50)
    parser.add_argument("--absence_anomaly_threshold", type=float, default=2.0)
    parser.add_argument("--absence_persistence_threshold", type=float, default=0.50)
    parser.add_argument("--absence_min_context_similarity", type=float, default=0.20)
    parser.add_argument("--absence_min_query_exposure", type=float, default=50.0)
    parser.add_argument("--absence_sigma_floor_ratio", type=float, default=0.25)
    parser.add_argument("--absence_coverage_threshold", type=float, default=0.50)
    parser.add_argument(
        "--absence_unexpected_mode",
        choices=["off", "strong"],
        default="off",
        help=(
            "When set to 'strong', high run-level service-silence evidence is "
            "treated as positive Unexpected evidence instead of only a Reject veto."
        ),
    )
    parser.add_argument("--absence_strong_anomaly_threshold", type=float, default=3.0)
    parser.add_argument("--absence_strong_coverage_threshold", type=float, default=1.0)
    parser.add_argument(
        "--include_labels",
        default=(
            "expected_drift,successful_no_drift,"
            "unexpected_drift,unexpected_without_observable_log_drift"
        ),
        help="Comma-separated benchmark_label values used for the main run-level RQ4 table.",
    )
    args = parser.parse_args()

    data = load_split(args.data_dir, args.split)
    if args.absence_veto == "off":
        train_data = []
    elif args.absence_reference_path:
        train_data = load_memory_sequences(args.absence_reference_path)
    else:
        train_data = load_split(args.data_dir, "train")
    annotations = load_annotation(args.data_dir)
    include_labels = {item.strip() for item in args.include_labels.split(",") if item.strip()}
    state_evidence = build_state_evidence(data, args.data_dir) if args.state_veto == "auto" else {}
    absence_evidence, absence_summary = (
        build_absence_evidence(train_data, data, args)
        if args.absence_veto != "off" else ({}, {})
    )

    by_run = defaultdict(lambda: {
        "event_pred": Counter(),
        "event_true": Counter(),
        "traditional": Counter(),
        "diagnosis": Counter(),
        "events_seen": 0,
    })

    with Path(args.events_csv).open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            batch_idx = safe_int(row.get("batch_index"))
            seq_idx = safe_int(row.get("sequence_index"))
            global_seq_idx = batch_idx * args.batch_size + seq_idx
            if global_seq_idx < 0 or global_seq_idx >= len(data):
                continue
            seq = data[global_seq_idx]
            if not seq:
                continue
            run_id = seq[0].get("run_id", "")
            if not run_id:
                continue
            ann = annotations.get(run_id, {})
            benchmark_label = ann.get("benchmark_label", seq[0].get("benchmark_label", ""))
            if include_labels and benchmark_label not in include_labels:
                continue
            true_id = safe_int(row.get("true_drift_id"), -1)
            if args.event_pred_column == "revised_selective":
                revised_id = safe_int(row.get("pred_drift_id"), 0)
                if revised_id in {1, 2, 3}:
                    pred_id = revised_id
                else:
                    score = safe_float(row.get("anomaly_score"), 0.0)
                    gamma = safe_float(row.get("gamma_anomaly"), float("inf"))
                    uncertainty = safe_float(row.get("uncertainty_score"), 0.0)
                    delta = safe_float(row.get("delta_uncertainty"), float("inf"))
                    if uncertainty > delta:
                        pred_id = 3
                    elif score > gamma:
                        pred_id = 2
                    else:
                        pred_id = 1
            elif args.event_pred_column in {"score_selective", "final_score_selective"}:
                score_key = (
                    "final_anomaly_score"
                    if args.event_pred_column == "final_score_selective"
                    else "anomaly_score"
                )
                score = safe_float(row.get(score_key), 0.0)
                gamma = safe_float(row.get("gamma_anomaly"), float("inf"))
                uncertainty = safe_float(row.get("uncertainty_score"), 0.0)
                delta = safe_float(row.get("delta_uncertainty"), float("inf"))
                if uncertainty > delta:
                    pred_id = 3
                elif score > gamma:
                    pred_id = 2
                else:
                    pred_id = 1
            elif args.event_pred_column == "score_threshold":
                score = safe_float(row.get("anomaly_score"), 0.0)
                gamma = safe_float(row.get("gamma_anomaly"), float("inf"))
                pred_id = 2 if score > gamma else 0
            elif args.event_pred_column == "final_score_threshold":
                score = safe_float(row.get("final_anomaly_score"), 0.0)
                gamma = safe_float(row.get("gamma_anomaly"), float("inf"))
                pred_id = 2 if score > gamma else 0
            else:
                pred_id = safe_int(row.get(args.event_pred_column), 0)
            pred_id = pred_id if pred_id in PRED_NAMES else 0
            st = by_run[run_id]
            st["event_pred"][pred_id] += 1
            st["event_true"][true_id] += 1
            st["traditional"][safe_int(row.get("traditional_candidate"), 0)] += 1
            st["diagnosis"][safe_int(row.get("diagnosis_candidate"), 0)] += 1
            st["events_seen"] += 1

    rows = []
    confusion = Counter()
    for run_id, st in sorted(by_run.items()):
        ann = annotations.get(run_id, {})
        true_id = st["event_true"].most_common(1)[0][0] if st["event_true"] else -1
        state_st = state_evidence.get(run_id, {})
        state_veto_applied = bool(state_st.get("state_strong_unexpected", False))
        absence_st = absence_evidence.get(run_id, {})
        vote_pred_id, decision_reason = decide_run(st["event_pred"], args, absence_st)
        absence_vote_scope = {1}
        if args.absence_apply_to == "normal_expected":
            absence_vote_scope = {0, 1}
        absence_unexpected_applied = strong_absence_unexpected(absence_st, args)
        absence_reject_applied = (
            args.absence_veto == "reject"
            and not absence_unexpected_applied
            and vote_pred_id in absence_vote_scope
            and bool(absence_st.get("absence_conflict", False) or absence_st.get("coverage_conflict", False))
        )
        pred_id = vote_pred_id
        if absence_unexpected_applied:
            pred_id = 2
        elif absence_reject_applied:
            pred_id = 3
        if state_veto_applied:
            pred_id = 2
        confusion[(true_id, pred_id)] += 1
        total = sum(st["event_pred"].values())
        frac = run_fractions(st["event_pred"])
        row = {
            "run_id": run_id,
            "semantic_label": ann.get("semantic_label", ""),
            "benchmark_label": ann.get("benchmark_label", ""),
            "change_family_id": ann.get("change_family_id", ""),
            "change_target_component_id": ann.get("change_target_component_id", ""),
            "true_id": true_id,
            "true_label": PRED_NAMES.get(true_id, "Other"),
            "vote_pred_id": vote_pred_id,
            "vote_pred_label": PRED_NAMES.get(vote_pred_id, "Other"),
            "decision_policy": args.decision_policy,
            "decision_reason": decision_reason,
            "state_veto_applied": int(state_veto_applied),
            "state_unexpected_tokens": "|".join(sorted(set(state_st.get("state_unexpected_tokens", [])))),
            "absence_unexpected_applied": int(absence_unexpected_applied),
            "absence_reject_applied": int(absence_reject_applied),
            "absence_anomaly": absence_st.get("absence_anomaly", ""),
            "coverage_support": absence_st.get("coverage_support", ""),
            "coverage_nn_cosine": absence_st.get("coverage_nn_cosine", ""),
            "absence_conflict": int(bool(absence_st.get("absence_conflict", False))) if absence_st else "",
            "coverage_conflict": int(bool(absence_st.get("coverage_conflict", False))) if absence_st else "",
            "absence_expected_services": absence_st.get("expected_services", ""),
            "absence_known_services": absence_st.get("known_expected_services", ""),
            "absence_silenced_services": absence_st.get("silenced_services", ""),
            "pred_id": pred_id,
            "pred_label": PRED_NAMES.get(pred_id, "Other"),
            "events_seen": total,
            "pred_normal_events": st["event_pred"][0],
            "pred_expected_events": st["event_pred"][1],
            "pred_unexpected_events": st["event_pred"][2],
            "pred_reject_events": st["event_pred"][3],
            "expected_frac": frac["expected"],
            "unexpected_frac": frac["unexpected"],
            "reject_frac": frac["reject"],
            "reject_margin_value": frac["reject"] - frac["expected"],
        }
        rows.append(row)

    metrics = {}
    for cls_id, name in [(1, "Expected"), (2, "Unexpected"), (3, "Reject")]:
        tp = confusion[(cls_id, cls_id)]
        fp = sum(v for (true_id, pred_id), v in confusion.items() if true_id != cls_id and pred_id == cls_id)
        fn = sum(v for (true_id, pred_id), v in confusion.items() if true_id == cls_id and pred_id != cls_id)
        precision, recall, score = f1(tp, fp, fn)
        metrics[f"{name}_Precision"] = precision
        metrics[f"{name}_Recall"] = recall
        metrics[f"{name}_F1"] = score
        metrics[f"{name}_True_Count"] = sum(v for (true_id, _), v in confusion.items() if true_id == cls_id)

    expected_f1 = metrics.get("Expected_F1", 0.0)
    unexpected_f1 = metrics.get("Unexpected_F1", 0.0)
    metrics["EU_Avg_F1"] = (expected_f1 + unexpected_f1) / 2.0
    metrics["Macro_F1_EUR"] = (
        metrics.get("Expected_F1", 0.0)
        + metrics.get("Unexpected_F1", 0.0)
        + metrics.get("Reject_F1", 0.0)
    ) / 3.0
    metrics["Run_Count"] = len(rows)
    unexpected_total = metrics.get("Unexpected_True_Count", 0)
    # Operational acceptance is Normal + Expected.  The previous definition
    # counted only Unexpected->Expected and therefore reported UFA=0 whenever
    # a degenerate detector mapped every unsafe run to Normal.  Keep the
    # narrower semantic error as a separate diagnostic, but use the operational
    # definition for the paper-facing UFA and threshold constraint.
    metrics["Unexpected_FalseExpected_Rate"] = (
        confusion[(2, 1)] / unexpected_total if unexpected_total else 0.0
    )
    metrics["Unexpected_False_Acceptance_Rate"] = (
        (confusion[(2, 0)] + confusion[(2, 1)]) / unexpected_total
        if unexpected_total else 0.0
    )
    metrics["Unexpected_Normal_Rate"] = (
        confusion[(2, 0)] / unexpected_total if unexpected_total else 0.0
    )
    metrics["Unexpected_Reject_Rate"] = (
        confusion[(2, 3)] / unexpected_total if unexpected_total else 0.0
    )
    metrics["Unexpected_SafeRate"] = (
        (confusion[(2, 2)] + confusion[(2, 3)]) / unexpected_total
        if unexpected_total else 0.0
    )

    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    pred_path = out_prefix.with_name(out_prefix.name + "_predictions.csv")
    with pred_path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else [
            "run_id", "true_id", "pred_id", "events_seen",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    summary_path = out_prefix.with_name(out_prefix.name + "_summary.json")
    confusion_json = {
        f"{PRED_NAMES.get(t, t)}->{PRED_NAMES.get(p, p)}": v
        for (t, p), v in sorted(confusion.items())
    }
    payload = {
        "metrics": metrics,
        "confusion": confusion_json,
        "include_labels": sorted(include_labels),
        "decision_thresholds": {
            "event_pred_column": args.event_pred_column,
            "expected_min": args.expected_min,
            "unexpected_min": args.unexpected_min,
            "reject_min": args.reject_min,
            "conflict_min": args.conflict_min,
            "decision_policy": args.decision_policy,
            "reject_margin": args.reject_margin,
            "unexpected_margin": args.unexpected_margin,
            "strong_unexpected_min": args.strong_unexpected_min,
            "expected_unexpected_max": args.expected_unexpected_max,
        },
        "state_veto": {
            "mode": args.state_veto,
            "strong_unexpected_tokens": sorted(STATE_STRONG_UNEXPECTED_TOKENS),
            "also_matches_prefix": "__STATE_UNAVAILABLE_SERVICE::",
            "applied_runs": sum(1 for row in rows if row.get("state_veto_applied") == 1),
        },
        "absence_veto": {
            "mode": args.absence_veto,
            "summary": absence_summary,
            "applied_runs": sum(1 for row in rows if row.get("absence_reject_applied") == 1),
            "unexpected_applied_runs": sum(1 for row in rows if row.get("absence_unexpected_applied") == 1),
            "apply_to": args.absence_apply_to,
            "context_mode": args.absence_context_mode,
            "metadata_fields": args.absence_metadata_fields,
            "count_ratio_threshold": args.absence_count_ratio_threshold,
            "anomaly_threshold": args.absence_anomaly_threshold,
            "unexpected_mode": args.absence_unexpected_mode,
            "strong_anomaly_threshold": args.absence_strong_anomaly_threshold,
            "strong_coverage_threshold": args.absence_strong_coverage_threshold,
        },
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"[OK] Wrote {pred_path}")
    print(f"[OK] Wrote {summary_path}")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
