#!/usr/bin/env python3
"""Validate run-level semantic labels using external runtime evidence."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from common import load_manifest, read_json, row_run_dir, write_csv


def as_bool(value):
    if isinstance(value, bool):
        return value
    if value is None:
        return None
    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "pass", "passed", "success"}:
        return True
    if text in {"0", "false", "no", "fail", "failed", "failure"}:
        return False
    return None


def number(payload, *keys):
    for key in keys:
        if key in payload and payload[key] not in {None, ""}:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                pass
    return None


def evidence_decision(row: dict, run_dir: Path, allow_declared_on_missing: bool) -> tuple[str, str, list[str]]:
    semantic_report = read_json(run_dir / "semantic_label_report.json")
    post_deploy = read_json(run_dir / "post_change_stable_report.json")
    api = read_json(run_dir / "api_assertions.json")
    deploy = post_deploy or read_json(run_dir / "deployment_events.json") or read_json(run_dir / "stable_state_report.json")
    pods = read_json(run_dir / "pod_status.json")
    slo = read_json(run_dir / "slo_summary.json")

    evidence_present = any([semantic_report, api, deploy, pods, slo])
    declared = (row.get("declared_semantic_label") or row.get("semantic_label") or "").strip().lower()
    if not evidence_present:
        if allow_declared_on_missing and declared:
            return declared, "missing_external_evidence_used_declared", ["external evidence files missing"]
        return "indeterminate", "missing_external_evidence", ["external evidence files missing"]

    report_label = str(semantic_report.get("semantic_label") or "").strip().lower()
    if report_label in {"expected", "unexpected", "indeterminate"}:
        reasons = ["semantic_label_report derived from external oracle and Kubernetes facts"]
        post_state = semantic_report.get("post_change_state")
        if post_state:
            reasons.append(f"post_change_state={post_state}")
        request_count = semantic_report.get("request_count")
        failure_count = semantic_report.get("assertion_failure_count")
        if request_count not in {None, ""}:
            reasons.append(f"request_count={request_count}")
        if failure_count not in {None, ""}:
            reasons.append(f"assertion_failure_count={failure_count}")
        return report_label, "semantic_label_report", reasons

    api_threshold = float(row.get("api_assertion_threshold") or 0.99)
    success_threshold = float(row.get("success_rate_threshold") or 0.99)
    p99_threshold = float(row.get("slo_p99_ms_threshold") or 1500)

    reasons = []
    failure = False
    success_checks = []

    deployment_success = as_bool(deploy.get("deployment_success"))
    deployment_failed = as_bool(deploy.get("deployment_failed"))
    readiness_success = as_bool(deploy.get("readiness_success"))
    service_registered = as_bool(deploy.get("service_registered"))
    if deployment_failed is True or deployment_success is False:
        failure = True
        reasons.append("deployment failure")
    if readiness_success is False:
        failure = True
        reasons.append("readiness failure")
    if service_registered is False:
        failure = True
        reasons.append("service registration failure")
    for value in (deployment_success, readiness_success, service_registered):
        if value is not None:
            success_checks.append(value)

    api_pass_rate = number(api, "pass_rate", "api_pass_rate", "assertion_pass_rate")
    api_success = as_bool(api.get("passed") if "passed" in api else api.get("success"))
    if api_pass_rate is not None:
        ok = api_pass_rate >= api_threshold
        success_checks.append(ok)
        if not ok:
            failure = True
            reasons.append(f"api pass rate {api_pass_rate:.4f} < {api_threshold:.4f}")
    elif api_success is not None:
        success_checks.append(api_success)
        if not api_success:
            failure = True
            reasons.append("api assertions failed")

    success_rate = number(slo, "success_rate", "request_success_rate")
    if success_rate is not None:
        ok = success_rate >= success_threshold
        success_checks.append(ok)
        if not ok:
            failure = True
            reasons.append(f"request success rate {success_rate:.4f} < {success_threshold:.4f}")

    p99 = number(slo, "p99_ms", "p99_latency_ms", "latency_p99_ms")
    slo_violated = as_bool(slo.get("slo_violated"))
    if p99 is not None:
        ok = p99 <= p99_threshold
        success_checks.append(ok)
        if not ok:
            failure = True
            reasons.append(f"p99 latency {p99:.3f} > {p99_threshold:.3f}")
    if slo_violated is True:
        failure = True
        reasons.append("slo violated")

    crash_loop = as_bool(pods.get("crash_loop") or pods.get("crashLoopBackOff"))
    oom_killed = as_bool(pods.get("oom_killed") or pods.get("OOMKilled"))
    pod_ready = as_bool(pods.get("ready") if "ready" in pods else pods.get("pod_ready"))
    if crash_loop:
        failure = True
        reasons.append("CrashLoopBackOff")
    if oom_killed:
        failure = True
        reasons.append("OOMKilled")
    if pod_ready is False:
        failure = True
        reasons.append("pod not ready")
    if pod_ready is not None:
        success_checks.append(pod_ready)

    if failure:
        if any(success_checks) and not all(success_checks):
            return "unexpected", "failure_evidence", reasons
        return "unexpected", "failure_evidence", reasons

    if success_checks and all(success_checks):
        return "expected", "all_external_checks_pass", reasons

    if allow_declared_on_missing and declared in {"expected", "unexpected", "indeterminate"}:
        return declared, "insufficient_evidence_used_declared", reasons or ["insufficient evidence"]
    return "indeterminate", "insufficient_evidence", reasons or ["insufficient evidence"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--raw_runs", default="raw_runs")
    parser.add_argument("--output", required=True)
    parser.add_argument("--strict", action="store_true", help="Do not fall back to declared labels when evidence is missing.")
    args = parser.parse_args()

    raw_runs = Path(args.raw_runs)
    rows = load_manifest(Path(args.manifest))
    out = []
    for row in rows:
        run_dir = row_run_dir(row, raw_runs)
        label, status, reasons = evidence_decision(row, run_dir, allow_declared_on_missing=not args.strict)
        payload = dict(row)
        payload["oracle_semantic_label"] = label
        payload["semantic_label"] = label
        payload["oracle_status"] = status
        payload["oracle_reasons"] = " | ".join(reasons)
        payload["resolved_run_dir"] = str(run_dir)
        out.append(payload)

    write_csv(Path(args.output), out)
    counts = {}
    for row in out:
        counts[row["oracle_semantic_label"]] = counts.get(row["oracle_semantic_label"], 0) + 1
    print({"rows": len(out), "semantic_counts": counts, "output": args.output})


if __name__ == "__main__":
    main()
