"""Context-conditioned absence memory for service-level log diagnosis.

The detector learns service-activity expectations only from trusted normal
runs.  For every candidate service, nearest-reference retrieval excludes that
service from the query and reference context (leave-one-service-out), so the
fact being tested cannot make its own neighbours look similar.  Service
activity is compared as an exposure-normalised rate and the anomaly score is
the smoothed empirical lower-tail probability rather than a Gaussian z-score.

This module intentionally does not inspect change targets, affected-component
annotations, oracle labels, or test labels when constructing expectations.
"""

from __future__ import annotations

import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np


def safe_text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "<missing>", "unknown"}:
        return ""
    return text


def _csv_set(value: Any) -> set[str]:
    return {item.strip() for item in str(value or "").split(",") if item.strip()}


def _csv_list(value: Any) -> list[str]:
    return [item.strip() for item in str(value or "").split(",") if item.strip()]


def _option(options: Any, name: str, default: Any) -> Any:
    if isinstance(options, Mapping):
        return options.get(name, default)
    return getattr(options, name, default)


def service_allowed(service: Any, options: Any) -> bool:
    service = safe_text(service)
    if not service:
        return False
    lowered = service.lower()
    if lowered.startswith("__state_") or lowered.startswith("__log_"):
        return False
    if service in _csv_set(_option(options, "absence_exclude_services", "")):
        return False
    prefixes = _csv_list(_option(options, "absence_service_prefixes", ""))
    return not prefixes or any(service.startswith(prefix) for prefix in prefixes)


def event_service(event: Mapping[str, Any]) -> str:
    for key in ("service", "sequence_service", "component_id"):
        service = safe_text(event.get(key))
        if service:
            return service
    return ""


def is_normal_reference(event: Mapping[str, Any]) -> bool:
    label = event.get("label", event.get("is_anomaly", 0))
    try:
        if int(float(label)) != 0:
            return False
    except (TypeError, ValueError):
        if str(label).strip().lower() not in {"0", "false", "normal", "benign", "ok"}:
            return False
    drift = str(event.get("drift_label", "normal")).strip().lower().replace("-", "_")
    return drift not in {"unexpected", "unexpected_drift", "reject", "rejected"}


def aggregate_run_service_counts(
    raw_data: list[list[dict[str, Any]]] | None,
    options: Any,
    *,
    normal_only: bool,
) -> dict[str, dict[str, Any]]:
    """Aggregate non-overlapping service sequences into run-level activity.

    The observation horizon of a run is the persistence window: a service must
    remain under-active across the aggregated run, not merely in one event.
    """
    runs: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"counts": Counter(), "events": 0, "normal": True}
    )
    for sequence in raw_data or []:
        if not sequence:
            continue
        first = sequence[0]
        run_id = safe_text(first.get("run_id")) or safe_text(first.get("sequence_id"))
        if not run_id:
            continue
        record = runs[run_id]
        for event in sequence:
            if normal_only and not is_normal_reference(event):
                record["normal"] = False
            service = event_service(event)
            if not service_allowed(service, options):
                continue
            try:
                weight = max(1, int(event.get("absence_count", 1)))
            except (TypeError, ValueError):
                weight = 1
            record["counts"][service] += weight
            record["events"] += weight
    return {
        run_id: record
        for run_id, record in runs.items()
        if record["events"] > 0 and (not normal_only or record["normal"])
    }


def load_memory_sequences(path: str | Path) -> list[list[dict[str, Any]]]:
    """Load a standalone absence-memory pickle or a dataset directory."""
    source = Path(path)
    if source.is_dir():
        source = source / "absence_memory.pkl"
    if not source.is_file():
        raise FileNotFoundError(f"Absence memory not found: {source}")
    with source.open("rb") as handle:
        obj = pickle.load(handle, encoding="latin-1")
    if isinstance(obj, list):
        return obj
    if not isinstance(obj, dict):
        raise ValueError(f"Unsupported absence-memory pickle object: {type(obj).__name__}")
    for key in ("absence_memory", "reference", "train"):
        value = obj.get(key)
        if isinstance(value, list):
            return value
    list_keys = [key for key, value in obj.items() if isinstance(value, list)]
    if len(list_keys) == 1:
        return obj[list_keys[0]]
    raise ValueError(
        f"Cannot identify sequence list in {source}; expected absence_memory/reference/train key"
    )


def _matrix_from_runs(runs: dict[str, dict[str, Any]]) -> dict[str, Any] | None:
    services = sorted({service for record in runs.values() for service in record["counts"]})
    if not runs or not services:
        return None
    run_ids = sorted(runs)
    service_to_idx = {service: index for index, service in enumerate(services)}
    counts = np.zeros((len(run_ids), len(services)), dtype=np.float64)
    for row, run_id in enumerate(run_ids):
        for service, count in runs[run_id]["counts"].items():
            counts[row, service_to_idx[service]] = float(count)
    return {
        "run_ids": run_ids,
        "services": services,
        "service_to_idx": service_to_idx,
        "counts": counts,
        "totals": counts.sum(axis=1),
    }


def _loo_cosine_similarities(reference_counts: np.ndarray, query: np.ndarray, target: int) -> np.ndarray:
    """Cosine similarities after removing the service whose absence is tested."""
    ref_context = reference_counts.copy()
    query_context = query.copy()
    ref_context[:, target] = 0.0
    query_context[target] = 0.0
    ref_features = np.log1p(ref_context)
    query_features = np.log1p(query_context)
    ref_norms = np.linalg.norm(ref_features, axis=1)
    query_norm = float(np.linalg.norm(query_features))
    if query_norm <= 1e-12:
        return np.zeros(reference_counts.shape[0], dtype=np.float64)
    numerators = ref_features.dot(query_features / query_norm)
    return np.divide(
        numerators,
        ref_norms,
        out=np.zeros_like(numerators),
        where=ref_norms > 1e-12,
    )


def build_context_conditioned_absence_evidence(
    raw_reference_data: list[list[dict[str, Any]]] | None,
    raw_eval_data: list[list[dict[str, Any]]] | None,
    options: Any,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Fit the normal memory and score persistent run-level service absence."""
    reference_runs = aggregate_run_service_counts(
        raw_reference_data, options, normal_only=True
    )
    eval_runs = aggregate_run_service_counts(raw_eval_data, options, normal_only=False)
    reference = _matrix_from_runs(reference_runs)
    if reference is None:
        return {}, {
            "enabled": True,
            "mechanism": "context_conditioned_absence_memory",
            "reference_runs": 0,
            "reference_services": 0,
            "eval_runs": len(eval_runs),
            "conflict_runs": 0,
        }

    k = max(1, int(_option(options, "absence_k", 20)))
    beta = float(_option(options, "absence_active_beta", 0.70))
    min_expected = float(_option(options, "absence_min_expected_count", 20.0))
    ratio_threshold = float(_option(options, "absence_count_ratio_threshold", 0.50))
    anomaly_threshold = float(_option(options, "absence_anomaly_threshold", 2.0))
    persistence_threshold = float(_option(options, "absence_persistence_threshold", 0.50))
    coverage_threshold = float(_option(options, "absence_coverage_threshold", 0.50))
    min_context_similarity = float(_option(options, "absence_min_context_similarity", 0.20))
    min_query_exposure = float(_option(options, "absence_min_query_exposure", 50.0))

    counts = reference["counts"]
    services = reference["services"]
    service_to_idx = reference["service_to_idx"]
    evidence: dict[str, dict[str, Any]] = {}
    anomaly_values: list[float] = []
    conflict_runs = 0

    for run_id, record in eval_runs.items():
        query = np.zeros(len(services), dtype=np.float64)
        for service, count in record["counts"].items():
            index = service_to_idx.get(service)
            if index is not None:
                query[index] = float(count)

        expected_services: list[str] = []
        known_services: list[str] = []
        silenced_services: list[str] = []
        service_scores: dict[str, dict[str, Any]] = {}
        similarities_used: list[float] = []
        max_anomaly = 0.0

        for service, target in service_to_idx.items():
            similarities = _loo_cosine_similarities(counts, query, target)
            k_eff = min(k, similarities.size)
            neighbor_idx = np.argsort(-similarities, kind="stable")[:k_eff]
            if neighbor_idx.size == 0:
                continue
            neighbor_similarity = float(np.mean(similarities[neighbor_idx]))
            neighbor_counts = counts[neighbor_idx, target]
            active_probability = float(np.mean(neighbor_counts > 0.0))
            expected_reference_count = float(np.median(neighbor_counts))
            if active_probability < beta or expected_reference_count < min_expected:
                continue
            expected_services.append(service)

            neighbor_exposure = (
                counts[neighbor_idx].sum(axis=1) - neighbor_counts
            )
            neighbor_rates = np.divide(
                neighbor_counts,
                np.maximum(neighbor_exposure, 1.0),
            )
            query_exposure = float(max(query.sum() - query[target], 0.0))
            observed = float(query[target])
            observed_rate = observed / max(query_exposure, 1.0)
            expected_rate = float(np.median(neighbor_rates))
            expected_scaled_count = expected_rate * query_exposure
            context_supported = (
                query_exposure >= min_query_exposure
                and neighbor_similarity >= min_context_similarity
                and expected_rate > 0.0
            )
            if not context_supported:
                service_scores[service] = {
                    "observed": observed,
                    "observed_rate": observed_rate,
                    "expected_reference_count": expected_reference_count,
                    "expected_scaled_count": expected_scaled_count,
                    "expected_rate": expected_rate,
                    "active_probability": active_probability,
                    "context_similarity": neighbor_similarity,
                    "context_supported": False,
                    "lower_tail_probability": 1.0,
                    "score": 0.0,
                    "persistence": 0.0,
                    "ratio_low": False,
                }
                continue

            known_services.append(service)
            lower_or_equal = int(np.sum(neighbor_rates <= observed_rate + 1e-12))
            lower_tail_probability = (1.0 + lower_or_equal) / (neighbor_idx.size + 1.0)
            empirical_score = -math.log(max(lower_tail_probability, 1e-12))
            rate_ratio = observed_rate / max(expected_rate, 1e-12)
            persistence = max(0.0, min(1.0, 1.0 - rate_ratio))
            ratio_low = bool(observed_rate <= expected_rate * ratio_threshold)
            persistent = bool(persistence >= persistence_threshold)
            score = empirical_score if ratio_low and persistent else 0.0
            max_anomaly = max(max_anomaly, score)
            similarities_used.append(neighbor_similarity)
            if score >= anomaly_threshold:
                silenced_services.append(service)
            service_scores[service] = {
                "observed": observed,
                "observed_rate": observed_rate,
                "expected_reference_count": expected_reference_count,
                "expected_scaled_count": expected_scaled_count,
                "expected_rate": expected_rate,
                "active_probability": active_probability,
                "context_similarity": neighbor_similarity,
                "context_supported": True,
                "lower_tail_probability": lower_tail_probability,
                "score": score,
                "persistence": persistence,
                "ratio_low": ratio_low,
                "persistent": persistent,
            }

        coverage_support = (
            1.0 - len(silenced_services) / len(known_services)
            if known_services
            else 1.0
        )
        absence_conflict = bool(silenced_services)
        coverage_conflict = bool(
            known_services and coverage_support < coverage_threshold
        )
        if absence_conflict or coverage_conflict:
            conflict_runs += 1
        anomaly_values.append(max_anomaly)
        evidence[run_id] = {
            "absence_anomaly": float(max_anomaly),
            "coverage_support": float(coverage_support),
            "coverage_nn_cosine": float(np.mean(similarities_used)) if similarities_used else 0.0,
            "absence_conflict": absence_conflict,
            "coverage_conflict": coverage_conflict,
            "expected_services": ",".join(expected_services),
            "known_expected_services": ",".join(known_services),
            "silenced_services": ",".join(silenced_services),
            "service_scores": service_scores,
        }

    summary = {
        "enabled": True,
        "mechanism": "context_conditioned_absence_memory",
        "reference_runs": len(reference["run_ids"]),
        "reference_services": len(services),
        "eval_runs": len(eval_runs),
        "conflict_runs": conflict_runs,
        "mean_absence_anomaly": float(np.mean(anomaly_values)) if anomaly_values else 0.0,
        "max_absence_anomaly": float(np.max(anomaly_values)) if anomaly_values else 0.0,
        "context_mode": "context_memory",
        "leave_one_service_out": True,
        "exposure_normalized": True,
        "empirical_lower_tail": True,
        "persistence_scope": "aggregated_run_observation_horizon",
        "k": k,
        "active_beta": beta,
        "anomaly_threshold": anomaly_threshold,
        "persistence_threshold": persistence_threshold,
        "min_context_similarity": min_context_similarity,
        "min_query_exposure": min_query_exposure,
    }
    return evidence, summary
