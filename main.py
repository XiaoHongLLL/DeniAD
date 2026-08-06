# -*- coding: utf-8 -*-
import argparse
import csv
import math
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os
from collections import Counter, defaultdict
try:
    import pandas as pd
except ImportError:
    pd = None
import torch.nn.functional as F

import transformer.Constants as Constants
import Utils
from preprocess.Dataset import get_dataloader
from transformer.Layers import get_non_pad_mask
from transformer.Models import (
    ANOMALY_DECISION,
    EXPECTED_DRIFT_DECISION,
    NORMAL_DECISION,
    REJECT_DECISION,
    CalibrationMemoryBank,
    DriftBuffer,
    DriftDecisionModule,
    FlowMatchingTHP,
)
from flow_matching.solver import ODESolver
from context_conditioned_absence_memory import (
    build_context_conditioned_absence_evidence,
    load_memory_sequences,
)


def synchronize_device(device):
    """Synchronize accelerator work before wall-clock measurements."""
    if isinstance(device, torch.device) and device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def efficiency_metadata(model, opt):
    total_params = int(sum(p.numel() for p in model.parameters()))
    trainable_params = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    device = getattr(opt, 'device', torch.device('cpu'))
    gpu_name = ''
    if isinstance(device, torch.device) and device.type == 'cuda' and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(device)
    return {
        'Model_Parameters': total_params,
        'Trainable_Parameters': trainable_params,
        'Evaluation_Device': str(device),
        'GPU_Name': gpu_name,
        'Batch_Size': int(getattr(opt, 'batch_size', 0)),
        'Num_Workers': int(getattr(opt, 'num_workers', 0)),
    }


def training_efficiency_path(checkpoint_path):
    if not checkpoint_path:
        return ''
    stem, _ = os.path.splitext(checkpoint_path)
    return f'{stem}_training_efficiency.csv'


def training_epoch_efficiency_path(checkpoint_path):
    if not checkpoint_path:
        return ''
    stem, _ = os.path.splitext(checkpoint_path)
    return f'{stem}_training_epoch_times.csv'


def save_single_row_csv(path, row):
    if not path or not row:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


def save_rows_csv(path, rows):
    if not path or not rows:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_single_row_csv(path):
    if not path or not os.path.isfile(path):
        return {}
    with open(path, mode='r', encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
    return dict(row) if row else {}

# torch.autograd.set_detect_anomaly(True)  # [已注释] 保证全速运行

def prepare_dataloader(opt):
    def load_data(name, dict_name):
        with open(name, 'rb') as f:
            data = pickle.load(f, encoding='latin-1')
            num_types = data['dim_process']
            data = data[dict_name]
            return data, int(num_types)

    print('[Info] Loading data...')
    train_data, num_types = load_data(opt.data + 'train.pkl', 'train')
    dev_data, _ = load_data(opt.data + 'dev.pkl', 'dev')
    test_data, _ = load_data(opt.data + 'test.pkl', 'test')

    opt.max_len = 0
    trainloader = get_dataloader(train_data, opt, shuffle=True, split='train')
    devloader = get_dataloader(dev_data, opt, shuffle=False, split='dev')
    testloader = get_dataloader(test_data, opt, shuffle=False, split='test')

    return trainloader, devloader, testloader, num_types


def load_named_split(data_dir, split_name):
    path = os.path.join(data_dir, f'{split_name}.pkl')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f'RQ2 classifier split not found: {path}. '
            'Regenerate controlled data with scripts/rq2/prepare_controlled_joint_dataset.py.'
        )
    with open(path, 'rb') as f:
        payload = pickle.load(f, encoding='latin-1')
    if split_name not in payload:
        available = ', '.join(str(key) for key in payload.keys())
        raise KeyError(f'Split key {split_name!r} missing in {path}; available keys: {available}')
    return payload[split_name], int(payload['dim_process'])


def prepare_rq2_classifier_dataloaders(opt):
    """Load supervised RQ2 head splits without changing source normalization stats."""
    train_name = getattr(opt, 'rq2_classifier_train_split', 'rq2_head_train')
    dev_name = getattr(opt, 'rq2_classifier_dev_split', 'rq2_head_dev')
    test_name = getattr(opt, 'rq2_classifier_test_split', 'test')

    train_data, _ = load_named_split(opt.data, train_name)
    dev_data, _ = load_named_split(opt.data, dev_name)
    test_data, _ = load_named_split(opt.data, test_name)

    # Use split='dev' intentionally: source train.pkl has already populated the
    # normalization statistics used by the saved generative checkpoint.
    trainloader = get_dataloader(train_data, opt, shuffle=True, split='dev')
    devloader = get_dataloader(dev_data, opt, shuffle=False, split='dev')
    testloader = get_dataloader(test_data, opt, shuffle=False, split='dev')
    return trainloader, devloader, testloader


RQ4_CLASS_NAMES = {
    1: 'Expected',
    2: 'Unexpected',
    3: 'Reject',
}
RQ4_PRED_CLASS_NAMES = {
    0: 'Normal',
    1: 'Expected',
    2: 'Unexpected',
    3: 'Reject',
}

RQ2_CLASS_NAMES = {
    1: 'Type',
    2: 'Time',
    3: 'Joint',
}


def unpack_batch(batch, device, return_drift=False, return_rq2=False):
    rq2_label = None
    if len(batch) == 5:
        event_time, time_gap_norm, event_type, event_label, extra_label = batch
        event_label = event_label.to(device)
        if return_rq2:
            rq2_label = extra_label.to(device)
            drift_label = None
        else:
            drift_label = extra_label.to(device)
    elif len(batch) == 4:
        event_time, time_gap_norm, event_type, event_label = batch
        event_label = event_label.to(device)
        drift_label = None
    else:
        event_time, time_gap_norm, event_type = batch
        event_label = None
        drift_label = None
    if return_rq2:
        return event_time.to(device), time_gap_norm.to(device), event_type.to(device), event_label, rq2_label
    if return_drift:
        return event_time.to(device), time_gap_norm.to(device), event_type.to(device), event_label, drift_label
    return event_time.to(device), time_gap_norm.to(device), event_type.to(device), event_label


def _safe_text(value):
    if value is None:
        return ''
    text = str(value).strip()
    if text.lower() in {'', 'none', 'nan', '<missing>', 'unknown'}:
        return ''
    return text


def _split_service_list(value):
    text = _safe_text(value)
    if not text:
        return []
    parts = []
    for chunk in text.replace(';', ',').replace('|', ',').split(','):
        item = chunk.strip()
        if item:
            parts.append(item)
    return parts


def _service_allowed_for_absence(service, opt):
    service = _safe_text(service)
    if not service:
        return False
    lowered = service.lower()
    if lowered.startswith('__state_') or lowered.startswith('__log_'):
        return False
    excluded = {
        item.strip()
        for item in str(getattr(opt, 'absence_exclude_services', '')).split(',')
        if item.strip()
    }
    if service in excluded:
        return False
    prefixes = [
        item.strip()
        for item in str(getattr(opt, 'absence_service_prefixes', '')).split(',')
        if item.strip()
    ]
    if prefixes and not any(service.startswith(prefix) for prefix in prefixes):
        return False
    return True


def _event_service_for_absence(event):
    for key in ('service', 'sequence_service', 'component_id'):
        service = _safe_text(event.get(key))
        if service:
            return service
    return ''


def _event_is_normal_reference(event):
    label = event.get('label', event.get('is_anomaly', 0))
    try:
        if int(float(label)) != 0:
            return False
    except (TypeError, ValueError):
        text = str(label).strip().lower()
        if text not in {'0', 'false', 'normal', 'benign', 'ok'}:
            return False
    drift = str(event.get('drift_label', 'normal')).strip().lower().replace('-', '_')
    if drift in {'unexpected', 'unexpected_drift', 'reject', 'rejected'}:
        return False
    return True


def _aggregate_run_service_counts(raw_data, opt, normal_only=False):
    runs = defaultdict(lambda: {'counts': Counter(), 'meta': {}, 'events': 0, 'normal': True})
    for seq in raw_data or []:
        if not seq:
            continue
        first = seq[0]
        run_id = _safe_text(first.get('run_id')) or _safe_text(first.get('sequence_id'))
        if not run_id:
            continue
        record = runs[run_id]
        if not record['meta']:
            record['meta'] = dict(first)
        for event in seq:
            if normal_only and not _event_is_normal_reference(event):
                record['normal'] = False
            service = _event_service_for_absence(event)
            if not _service_allowed_for_absence(service, opt):
                continue
            record['counts'][service] += 1
            record['events'] += 1
    if normal_only:
        runs = {
            run_id: record
            for run_id, record in runs.items()
            if record['normal'] and record['events'] > 0
        }
    else:
        runs = {
            run_id: record
            for run_id, record in runs.items()
            if record['events'] > 0
        }
    return runs


def _load_raw_split(opt, split):
    path = os.path.join(opt.data, f'{split}.pkl')
    with open(path, 'rb') as f:
        obj = pickle.load(f, encoding='latin-1')
    return obj[split]


def _fit_absence_reference(raw_data, opt):
    runs = _aggregate_run_service_counts(raw_data, opt, normal_only=True)
    services = sorted({
        service
        for record in runs.values()
        for service in record['counts'].keys()
    })
    if not runs or not services:
        return None

    run_ids = sorted(runs.keys())
    counts = np.zeros((len(run_ids), len(services)), dtype=np.float32)
    service_to_idx = {service: idx for idx, service in enumerate(services)}
    for row, run_id in enumerate(run_ids):
        for service, count in runs[run_id]['counts'].items():
            idx = service_to_idx.get(service)
            if idx is not None:
                counts[row, idx] = float(count)

    features = np.log1p(counts)
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    features = features / np.maximum(norms, 1e-8)
    return {
        'run_ids': run_ids,
        'services': services,
        'service_to_idx': service_to_idx,
        'counts': counts,
        'features': features,
    }


def _metadata_expected_services(meta, opt):
    fields = [
        item.strip()
        for item in str(getattr(opt, 'absence_metadata_fields', '')).split(',')
        if item.strip()
    ]
    ignored = {'none', 'all', 'all-observed-services', 'workload-generator', 'system-observability'}
    services = []
    seen = set()
    for field in fields:
        for service in _split_service_list(meta.get(field)):
            service = _safe_text(service)
            if not service or service.lower() in ignored:
                continue
            if not _service_allowed_for_absence(service, opt):
                continue
            if service not in seen:
                seen.add(service)
                services.append(service)
    return services


def _nearest_reference_indices(reference, query_counts, k):
    if reference is None or reference['counts'].shape[0] == 0:
        return np.array([], dtype=np.int64), 0.0
    query = np.log1p(query_counts.astype(np.float32))
    query_norm = np.linalg.norm(query)
    if query_norm <= 1e-8:
        sims = np.zeros(reference['features'].shape[0], dtype=np.float32)
    else:
        sims = reference['features'].dot(query / query_norm)
    k_eff = max(1, min(int(k), sims.shape[0]))
    top_idx = np.argsort(-sims)[:k_eff]
    return top_idx.astype(np.int64), float(sims[top_idx[0]]) if top_idx.size else 0.0


def fit_absence_evidence(raw_reference_data, raw_eval_data, opt):
    if not getattr(opt, 'use_absence_aware_revision', False):
        return {}, {}
    if getattr(opt, 'absence_context_mode', 'context_memory') == 'context_memory':
        return build_context_conditioned_absence_evidence(
            raw_reference_data,
            raw_eval_data,
            opt,
        )
    reference = _fit_absence_reference(raw_reference_data, opt)
    if reference is None:
        return {}, {'enabled': True, 'reference_runs': 0, 'reference_services': 0}

    eval_runs = _aggregate_run_service_counts(raw_eval_data, opt, normal_only=False)
    k = int(getattr(opt, 'absence_k', 5))
    beta = float(getattr(opt, 'absence_active_beta', 0.7))
    min_expected = float(getattr(opt, 'absence_min_expected_count', 20.0))
    ratio_threshold = float(getattr(opt, 'absence_count_ratio_threshold', 0.5))
    anomaly_threshold = float(getattr(opt, 'absence_anomaly_threshold', 2.0))
    sigma_floor_ratio = float(getattr(opt, 'absence_sigma_floor_ratio', 0.25))
    coverage_threshold = float(getattr(opt, 'absence_coverage_threshold', 0.5))
    context_mode = getattr(opt, 'absence_context_mode', 'hybrid')

    evidence = {}
    all_absence = []
    conflict_runs = 0
    for run_id, record in eval_runs.items():
        query_counts = np.zeros(len(reference['services']), dtype=np.float32)
        for service, count in record['counts'].items():
            idx = reference['service_to_idx'].get(service)
            if idx is not None:
                query_counts[idx] = float(count)

        nn_idx, nn_cosine = _nearest_reference_indices(reference, query_counts, k)
        if nn_idx.size:
            nn_counts = reference['counts'][nn_idx]
        else:
            nn_counts = reference['counts']

        active_prob = (nn_counts > 0).mean(axis=0)
        mu = nn_counts.mean(axis=0)
        sigma = nn_counts.std(axis=0)
        metadata_services = _metadata_expected_services(record['meta'], opt)
        metadata_known = [s for s in metadata_services if s in reference['service_to_idx']]
        memory_services = [
            service
            for service, idx in reference['service_to_idx'].items()
            if active_prob[idx] >= beta and mu[idx] >= min_expected
        ]
        if context_mode == 'metadata':
            expected_services = metadata_known
        elif context_mode == 'memory':
            expected_services = memory_services
        else:
            expected_services = metadata_known if metadata_known else memory_services

        service_scores = {}
        silenced = []
        known = []
        max_absence = 0.0
        for service in expected_services:
            idx = reference['service_to_idx'].get(service)
            if idx is None:
                continue
            expected = float(mu[idx])
            if expected < min_expected:
                continue
            observed = float(query_counts[idx])
            floor = max(float(sigma[idx]), expected * sigma_floor_ratio, 1.0)
            score = max(0.0, (expected - observed) / floor)
            ratio_low = observed <= expected * ratio_threshold
            if not ratio_low:
                score = 0.0
            known.append(service)
            service_scores[service] = {
                'observed': observed,
                'expected': expected,
                'score': score,
                'ratio_low': bool(ratio_low),
            }
            max_absence = max(max_absence, score)
            if score >= anomaly_threshold:
                silenced.append(service)

        if known:
            coverage_support = 1.0 - (len(silenced) / max(len(known), 1))
        else:
            coverage_support = max(0.0, min(1.0, nn_cosine))
        absence_conflict = bool(silenced)
        coverage_conflict = bool(coverage_support < coverage_threshold)
        if absence_conflict or coverage_conflict:
            conflict_runs += 1
        all_absence.append(max_absence)
        evidence[run_id] = {
            'absence_anomaly': float(max_absence),
            'coverage_support': float(coverage_support),
            'coverage_nn_cosine': float(max(0.0, min(1.0, nn_cosine))),
            'absence_conflict': absence_conflict,
            'coverage_conflict': coverage_conflict,
            'expected_services': ','.join(expected_services),
            'known_expected_services': ','.join(known),
            'silenced_services': ','.join(silenced),
            'service_scores': service_scores,
        }

    summary = {
        'enabled': True,
        'reference_runs': len(reference['run_ids']),
        'reference_services': len(reference['services']),
        'eval_runs': len(eval_runs),
        'conflict_runs': conflict_runs,
        'mean_absence_anomaly': float(np.mean(all_absence)) if all_absence else 0.0,
        'max_absence_anomaly': float(np.max(all_absence)) if all_absence else 0.0,
        'context_mode': context_mode,
    }
    return evidence, summary


def build_absence_batch_context(raw_eval_data, batch_idx, batch_size, event_shape, device, evidence):
    rows, cols = event_shape
    absence = torch.zeros((rows, cols), dtype=torch.float32, device=device)
    coverage = torch.ones((rows, cols), dtype=torch.float32, device=device)
    nn_cosine = torch.ones((rows, cols), dtype=torch.float32, device=device)
    absence_conflict = torch.zeros((rows, cols), dtype=torch.bool, device=device)
    coverage_conflict = torch.zeros((rows, cols), dtype=torch.bool, device=device)
    if not evidence:
        return {
            'absence_anomaly': absence,
            'coverage_support': coverage,
            'coverage_nn_cosine': nn_cosine,
            'absence_conflict': absence_conflict,
            'coverage_conflict': coverage_conflict,
        }
    for row in range(rows):
        global_idx = batch_idx * batch_size + row
        if global_idx < 0 or global_idx >= len(raw_eval_data):
            continue
        seq = raw_eval_data[global_idx]
        if not seq:
            continue
        run_id = _safe_text(seq[0].get('run_id')) or _safe_text(seq[0].get('sequence_id'))
        run_evidence = evidence.get(run_id)
        if not run_evidence:
            continue
        absence[row, :] = float(run_evidence.get('absence_anomaly', 0.0))
        coverage[row, :] = float(run_evidence.get('coverage_support', 1.0))
        nn_cosine[row, :] = float(run_evidence.get('coverage_nn_cosine', 1.0))
        absence_conflict[row, :] = bool(run_evidence.get('absence_conflict', False))
        coverage_conflict[row, :] = bool(run_evidence.get('coverage_conflict', False))
    return {
        'absence_anomaly': absence,
        'coverage_support': coverage,
        'coverage_nn_cosine': nn_cosine,
        'absence_conflict': absence_conflict,
        'coverage_conflict': coverage_conflict,
    }


def init_binary_counts(prefix):
    return {
        f'{prefix}_TP': 0,
        f'{prefix}_FP': 0,
        f'{prefix}_FN': 0,
        f'{prefix}_TN': 0,
    }


def update_binary_counts(counts, prefix, pred_positive, true_label, valid_mask):
    label_mask = valid_mask & (true_label >= 0)
    if label_mask.sum().item() == 0:
        return 0

    pred = pred_positive[label_mask].bool()
    truth = true_label[label_mask].bool()
    counts[f'{prefix}_TP'] += (pred & truth).sum().item()
    counts[f'{prefix}_FP'] += (pred & ~truth).sum().item()
    counts[f'{prefix}_FN'] += (~pred & truth).sum().item()
    counts[f'{prefix}_TN'] += (~pred & ~truth).sum().item()
    return int(label_mask.sum().item())


def finalize_binary_metrics(counts, prefix):
    tp = counts.get(f'{prefix}_TP', 0)
    fp = counts.get(f'{prefix}_FP', 0)
    fn = counts.get(f'{prefix}_FN', 0)
    tn = counts.get(f'{prefix}_TN', 0)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        f'{prefix}_TP': tp,
        f'{prefix}_FP': fp,
        f'{prefix}_FN': fn,
        f'{prefix}_TN': tn,
        f'{prefix}_Precision': precision,
        f'{prefix}_Recall': recall,
        f'{prefix}_F1': f1,
        f'{prefix}_FPR': fpr,
        f'{prefix}_FNR': fnr,
    }


def binary_ranking_metrics(scores, labels, prefix):
    scores = scores.detach().cpu().float().reshape(-1)
    labels = labels.detach().cpu().long().reshape(-1)
    valid = labels >= 0
    scores = scores[valid]
    labels = labels[valid]
    if scores.numel() == 0 or labels.unique().numel() < 2:
        return {}

    order = torch.argsort(scores, descending=True)
    sorted_labels = labels[order].float()
    positives = sorted_labels.sum().item()
    negatives = sorted_labels.numel() - positives
    if positives <= 0 or negatives <= 0:
        return {}

    tp = torch.cumsum(sorted_labels, dim=0)
    fp = torch.cumsum(1.0 - sorted_labels, dim=0)
    precision = tp / (tp + fp).clamp_min(1e-12)
    recall = tp / max(positives, 1e-12)
    auprc = (precision * torch.diff(torch.cat([torch.zeros(1), recall]))).sum().item()

    tpr = torch.cat([torch.zeros(1), tp / positives, torch.ones(1)])
    fpr = torch.cat([torch.zeros(1), fp / negatives, torch.ones(1)])
    auroc = torch.trapz(tpr, fpr).item()
    return {
        f'{prefix}_AUPRC': auprc,
        f'{prefix}_AUROC': auroc,
    }


def score_distribution_stats(scores, labels, prefix):
    scores = scores.detach().cpu().float().reshape(-1)
    labels = labels.detach().cpu().long().reshape(-1)
    valid = labels >= 0
    scores = scores[valid]
    labels = labels[valid]
    results = {}

    for label_value, name in [(0, 'Normal'), (1, 'Anomaly')]:
        subset = scores[labels == label_value]
        base = f'{prefix}_{name}_Score'
        results[f'{base}_Count'] = int(subset.numel())
        if subset.numel() == 0:
            for stat in ['Mean', 'Std', 'P50', 'P90', 'P95', 'P99']:
                results[f'{base}_{stat}'] = 0.0
            continue
        results[f'{base}_Mean'] = float(subset.mean().item())
        results[f'{base}_Std'] = float(subset.std(unbiased=False).item()) if subset.numel() > 1 else 0.0
        for q, stat in [(0.50, 'P50'), (0.90, 'P90'), (0.95, 'P95'), (0.99, 'P99')]:
            results[f'{base}_{stat}'] = float(torch.quantile(subset, q).item())
    return results


def init_rq2_counts():
    counts = {}
    for _, name in RQ2_CLASS_NAMES.items():
        counts.update(init_binary_counts(f'RQ2_{name}Event'))
        counts.update(init_binary_counts(f'RQ2_{name}Segment'))
    return counts


def update_rq2_event_counts(counts, pred_anomaly, rq2_label, mask):
    if rq2_label is None:
        return 0
    total = 0
    for class_id, name in RQ2_CLASS_NAMES.items():
        class_valid = mask & ((rq2_label == 0) | (rq2_label == class_id))
        class_truth = (rq2_label == class_id).long()
        total += update_binary_counts(
            counts,
            f'RQ2_{name}Event',
            pred_anomaly,
            class_truth,
            class_valid
        )
    return total


def update_rq2_segment_counts(counts, pred_segment_anomaly, rq2_label, mask):
    if rq2_label is None:
        return 0
    segment_known = ((rq2_label >= 0) & mask).any(dim=1)
    segment_has_any = ((rq2_label > 0) & mask).any(dim=1)
    total = 0
    for class_id, name in RQ2_CLASS_NAMES.items():
        segment_has_class = ((rq2_label == class_id) & mask).any(dim=1)
        class_valid = segment_known & (segment_has_class | ~segment_has_any)
        class_truth = segment_has_class.long()
        total += update_binary_counts(
            counts,
            f'RQ2_{name}Segment',
            pred_segment_anomaly,
            class_truth,
            class_valid
        )
    return total


def finalize_rq2_metrics(counts):
    results = {}
    for _, name in RQ2_CLASS_NAMES.items():
        results.update(finalize_binary_metrics(counts, f'RQ2_{name}Event'))
        results.update(finalize_binary_metrics(counts, f'RQ2_{name}Segment'))
    return results


RQ2_CLASSIFIER_VARIANTS = {
    'type_only',
    'time_only',
    'independent_joint',
    'ours_joint',
}


class RQ2ClassificationHead(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, dropout=0.1, num_classes=4):
        super().__init__()
        hidden_dim = max(8, int(hidden_dim))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, features):
        return self.net(features)


def normalize_rq2_classifier_variant(variant):
    variant = str(variant or '').strip().lower().replace('-', '_')
    aliases = {
        'type': 'type_only',
        'time': 'time_only',
        'independent': 'independent_joint',
        'independent_join': 'independent_joint',
        'joint_independent': 'independent_joint',
        'ours': 'ours_joint',
        'our_joint': 'ours_joint',
        'oursjoint': 'ours_joint',
    }
    variant = aliases.get(variant, variant)
    if variant not in RQ2_CLASSIFIER_VARIANTS:
        raise ValueError(
            f'Unknown rq2_classifier_variant={variant}. '
            f'Choose from {sorted(RQ2_CLASSIFIER_VARIANTS)}.'
        )
    return variant


def _feature_block(block, enabled):
    if enabled:
        return block.detach().float()
    return torch.zeros_like(block, dtype=torch.float32)


def _scalar_feature_block(scalars, enabled_flags):
    blocks = []
    for value, enabled in zip(scalars, enabled_flags):
        value = value.detach().float().unsqueeze(-1)
        blocks.append(value if enabled else torch.zeros_like(value))
    return torch.cat(blocks, dim=-1)


def build_rq2_classifier_feature_map(model, event_type, event_time, time_gap_norm, opt, variant):
    """Build a fixed-width event feature map for the supervised RQ2 head.

    The same classifier architecture is used for all variants. Variant identity
    only decides which upstream probabilistic components are visible.
    """
    variant = normalize_rq2_classifier_variant(variant)
    context = model._encode_detection_context(event_type, event_time, time_gap_norm)
    type_nll, mask = model.get_type_nll_map(event_type, context['type_logits'])
    type_entropy = model.get_type_entropy_map(context['type_logits']).masked_fill(~mask, 0.0)

    uncertainty_mc = max(1, int(getattr(opt, 'rq2_classifier_uncertainty_mc', 1)))
    fm_uncertainty, path_energy, _ = model.get_flow_uncertainty_map(
        event_type,
        event_time,
        time_gap_norm,
        n_mc=uncertainty_mc,
        context=context
    )
    if getattr(opt, 'reliability_exact_nll', False):
        time_nll, _ = model.get_per_event_time_nll(
            event_type,
            event_time,
            time_gap_norm,
            context=context
        )
    else:
        time_nll = fm_uncertainty.detach()

    conditional_gap = torch.zeros_like(type_nll)
    if getattr(opt, 'rq2_classifier_use_profile_features', False):
        conditional_gap = type_gap_profile_score(event_type, time_gap_norm, opt).to(type_nll.device)

    use_type = variant in {'type_only', 'independent_joint', 'ours_joint'}
    use_time = variant in {'time_only', 'independent_joint', 'ours_joint'}
    use_cond = variant in {'time_only', 'independent_joint', 'ours_joint'}
    use_profile = variant == 'ours_joint' and getattr(opt, 'rq2_classifier_use_profile_features', False)

    type_block = _feature_block(context['c_type'], use_type)
    time_block = _feature_block(context['c_time'], use_time)
    cond_block = _feature_block(context['c_cond'], use_cond)
    scalar_block = _scalar_feature_block(
        [type_nll, time_nll, conditional_gap, type_entropy, fm_uncertainty, path_energy],
        [use_type, use_time, use_profile, use_type, use_time, use_time],
    )
    feature_map = torch.cat([type_block, time_block, cond_block, scalar_block], dim=-1)
    feature_map = feature_map.masked_fill(~mask.unsqueeze(-1), 0.0)
    return feature_map, mask


def _rq2_segment_class(labels, valid_mask):
    if not valid_mask.any().item():
        return -1
    row_labels = labels[valid_mask]
    if (row_labels == 3).any().item():
        return 3
    if (row_labels == 2).any().item():
        return 2
    if (row_labels == 1).any().item():
        return 1
    return 0


def collect_rq2_classifier_examples(model, loader, opt, variant, split_name):
    model.eval()
    feature_parts = []
    label_parts = []
    segment_id_parts = []
    segment_classes = []
    segment_offset = 0

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'  - (RQ2 {split_name} features) ', leave=False):
            event_time, time_gap_norm, event_type, _, rq2_label = unpack_batch(
                batch,
                opt.device,
                return_rq2=True
            )
            if rq2_label is None:
                raise ValueError('RQ2 classifier requires rq2_label in the dataset.')
            time_gap_norm = ensure_time_gap_normalized(
                time_gap_norm,
                opt,
                context=f'rq2_classifier_{split_name}',
                event_type=event_type
            )
            feature_map, mask = build_rq2_classifier_feature_map(
                model,
                event_type,
                event_time,
                time_gap_norm,
                opt,
                variant
            )
            valid = mask & (rq2_label >= 0)
            if valid.any():
                rows, _ = valid.nonzero(as_tuple=True)
                feature_parts.append(feature_map[valid].detach().cpu())
                label_parts.append(rq2_label[valid].detach().cpu().long())
                segment_id_parts.append(rows.detach().cpu().long() + segment_offset)

            known = (rq2_label >= 0) & mask
            for row in range(event_type.size(0)):
                segment_classes.append(_rq2_segment_class(rq2_label[row], known[row]))
            segment_offset += int(event_type.size(0))

    if not feature_parts:
        raise ValueError(f'RQ2 classifier {split_name} split has no labeled events.')

    labels = torch.cat(label_parts).long()
    if (labels >= 0).sum().item() == 0:
        raise ValueError(f'RQ2 classifier {split_name} split has only unlabeled events.')
    return {
        'features': torch.cat(feature_parts).float(),
        'labels': labels.clamp(min=0, max=3),
        'segment_ids': torch.cat(segment_id_parts).long(),
        'segment_class': torch.tensor(segment_classes, dtype=torch.long),
    }


def normalize_feature_bundle(bundle, mean, std):
    result = dict(bundle)
    result['features'] = (bundle['features'] - mean) / std
    return result


def predict_rq2_classifier_logits(head, features, opt):
    head.eval()
    outputs = []
    batch_size = max(1, int(getattr(opt, 'rq2_classifier_eval_batch_size', 65536)))
    with torch.no_grad():
        for start in range(0, features.size(0), batch_size):
            end = min(start + batch_size, features.size(0))
            logits = head(features[start:end].to(opt.device))
            outputs.append(logits.detach().cpu())
    return torch.cat(outputs, dim=0)


def evaluate_rq2_classifier_outputs(logits, bundle):
    labels = bundle['labels'].long()
    pred_class = logits.argmax(dim=-1).detach().cpu().long()
    pred_event_anomaly = pred_class > 0
    true_event_anomaly = (labels > 0).long()

    counts = init_binary_counts('Event')
    counts.update(init_binary_counts('Segment'))
    valid_events = labels >= 0
    labeled_events = update_binary_counts(
        counts,
        'Event',
        pred_event_anomaly,
        true_event_anomaly,
        valid_events
    )

    segment_class = bundle['segment_class'].long()
    segment_valid = segment_class >= 0
    segment_truth = (segment_class > 0).long()
    segment_pred = torch.zeros(segment_class.numel(), dtype=torch.bool)
    positive_segment_ids = bundle['segment_ids'][pred_event_anomaly]
    if positive_segment_ids.numel() > 0:
        segment_pred[positive_segment_ids.clamp(min=0, max=segment_pred.numel() - 1)] = True
    labeled_segments = update_binary_counts(
        counts,
        'Segment',
        segment_pred,
        segment_truth,
        segment_valid
    )

    rq2_counts = init_rq2_counts()
    for class_id, name in RQ2_CLASS_NAMES.items():
        event_valid = valid_events & ((labels == 0) | (labels == class_id))
        event_truth = (labels == class_id).long()
        update_binary_counts(
            rq2_counts,
            f'RQ2_{name}Event',
            pred_event_anomaly,
            event_truth,
            event_valid
        )
        segment_sub_valid = segment_valid & ((segment_class == 0) | (segment_class == class_id))
        segment_sub_truth = (segment_class == class_id).long()
        update_binary_counts(
            rq2_counts,
            f'RQ2_{name}Segment',
            segment_pred,
            segment_sub_truth,
            segment_sub_valid
        )

    results = {
        'Labeled_Events': labeled_events,
        'Labeled_Segments': labeled_segments,
        'Classifier_Event_Accuracy': (
            (pred_class[valid_events] == labels[valid_events]).float().mean().item()
            if valid_events.any() else 0.0
        ),
    }
    if labeled_events > 0:
        results.update(finalize_binary_metrics(counts, 'Event'))
    if labeled_segments > 0:
        results.update(finalize_binary_metrics(counts, 'Segment'))
    results.update(finalize_rq2_metrics(rq2_counts))
    return results


def train_rq2_classifier_head(train_bundle, dev_bundle, opt):
    input_dim = int(train_bundle['features'].size(1))
    head = RQ2ClassificationHead(
        input_dim=input_dim,
        hidden_dim=getattr(opt, 'rq2_classifier_hidden', 128),
        dropout=getattr(opt, 'rq2_classifier_dropout', 0.1),
        num_classes=4
    ).to(opt.device)

    labels = train_bundle['labels'].long()
    class_counts = torch.bincount(labels.clamp(min=0, max=3), minlength=4).float()
    class_weights = class_counts.sum() / (4.0 * class_counts.clamp_min(1.0))
    class_weights = class_weights.clamp(max=float(getattr(opt, 'rq2_classifier_max_class_weight', 30.0)))
    class_weights = class_weights.to(opt.device)

    positives = (labels > 0).sum().float()
    negatives = (labels == 0).sum().float()
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(
        max=float(getattr(opt, 'rq2_classifier_max_binary_weight', 50.0))
    ).to(opt.device)

    optimizer = optim.AdamW(
        head.parameters(),
        lr=float(getattr(opt, 'rq2_classifier_lr', 1e-3)),
        weight_decay=float(getattr(opt, 'rq2_classifier_weight_decay', 1e-4))
    )
    epochs = max(1, int(getattr(opt, 'rq2_classifier_epochs', 30)))
    batch_size = max(1, int(getattr(opt, 'rq2_classifier_batch_size', 8192)))
    binary_loss_weight = float(getattr(opt, 'rq2_classifier_binary_loss_weight', 0.5))

    best_state = None
    best_metric = -1.0
    best_dev = {}
    train_features = train_bundle['features']
    train_labels = train_bundle['labels']
    n_train = train_features.size(0)

    for epoch in range(1, epochs + 1):
        head.train()
        order = torch.randperm(n_train)
        total_loss = 0.0
        total_seen = 0
        for start in range(0, n_train, batch_size):
            idx = order[start:min(start + batch_size, n_train)]
            x = train_features[idx].to(opt.device)
            y = train_labels[idx].to(opt.device)
            logits = head(x)
            loss = F.cross_entropy(logits, y, weight=class_weights)
            if binary_loss_weight > 0:
                anomaly_logit = torch.logsumexp(logits[:, 1:], dim=-1) - logits[:, 0]
                binary_target = (y > 0).float()
                binary_loss = F.binary_cross_entropy_with_logits(
                    anomaly_logit,
                    binary_target,
                    pos_weight=pos_weight
                )
                loss = loss + binary_loss_weight * binary_loss

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(head.parameters(), 5.0)
            optimizer.step()

            total_loss += float(loss.item()) * idx.numel()
            total_seen += idx.numel()

        dev_logits = predict_rq2_classifier_logits(head, dev_bundle['features'], opt)
        dev_metrics = evaluate_rq2_classifier_outputs(dev_logits, dev_bundle)
        dev_f1 = float(dev_metrics.get('Segment_F1', 0.0))
        if dev_f1 > best_metric:
            best_metric = dev_f1
            best_dev = dev_metrics
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in head.state_dict().items()
            }

        if epoch == 1 or epoch == epochs or epoch % max(1, epochs // 5) == 0:
            avg_loss = total_loss / max(1, total_seen)
            print(
                f'    RQ2 head epoch {epoch:03d}/{epochs}: '
                f'loss={avg_loss:.6f} dev_F1={dev_f1:.4f} '
                f'dev_FNR={dev_metrics.get("Segment_FNR", 0.0):.4f} '
                f'dev_FPR={dev_metrics.get("Segment_FPR", 0.0):.4f}'
            )

    if best_state is not None:
        head.load_state_dict({key: value.to(opt.device) for key, value in best_state.items()})

    return head, best_metric, best_dev


def eval_rq2_classifier(model, opt):
    variant = normalize_rq2_classifier_variant(getattr(opt, 'rq2_classifier_variant', 'ours_joint'))
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    head_train_loader, head_dev_loader, head_test_loader = prepare_rq2_classifier_dataloaders(opt)

    synchronize_device(opt.device)
    feature_start = time.perf_counter()
    train_bundle = collect_rq2_classifier_examples(model, head_train_loader, opt, variant, 'train')
    dev_bundle = collect_rq2_classifier_examples(model, head_dev_loader, opt, variant, 'dev')
    test_bundle = collect_rq2_classifier_examples(model, head_test_loader, opt, variant, 'test')
    feature_time = time.perf_counter() - feature_start

    feature_mean = train_bundle['features'].mean(dim=0, keepdim=True)
    feature_std = train_bundle['features'].std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
    train_bundle = normalize_feature_bundle(train_bundle, feature_mean, feature_std)
    dev_bundle = normalize_feature_bundle(dev_bundle, feature_mean, feature_std)
    test_bundle = normalize_feature_bundle(test_bundle, feature_mean, feature_std)

    synchronize_device(opt.device)
    train_start = time.perf_counter()
    head, best_dev_f1, best_dev = train_rq2_classifier_head(train_bundle, dev_bundle, opt)
    synchronize_device(opt.device)
    classifier_train_time = time.perf_counter() - train_start

    synchronize_device(opt.device)
    eval_start = time.perf_counter()
    test_logits = predict_rq2_classifier_logits(head, test_bundle['features'], opt)
    results = evaluate_rq2_classifier_outputs(test_logits, test_bundle)
    synchronize_device(opt.device)
    evaluation_time = time.perf_counter() - eval_start

    results.update({
        'Evaluation_Mode': 'rq2_classifier_head',
        'Classifier_Variant': variant,
        'Classifier_Feature_Dim': int(train_bundle['features'].size(1)),
        'Classifier_Train_Events': int(train_bundle['features'].size(0)),
        'Classifier_Dev_Events': int(dev_bundle['features'].size(0)),
        'Classifier_Test_Events': int(test_bundle['features'].size(0)),
        'Classifier_Best_Dev_F1': float(best_dev_f1),
        'Classifier_Best_Dev_FNR': float(best_dev.get('Segment_FNR', 0.0)),
        'Classifier_Best_Dev_FPR': float(best_dev.get('Segment_FPR', 0.0)),
        'Classifier_Epochs': int(getattr(opt, 'rq2_classifier_epochs', 30)),
        'Classifier_LR': float(getattr(opt, 'rq2_classifier_lr', 1e-3)),
        'Classifier_Binary_Loss_Weight': float(getattr(opt, 'rq2_classifier_binary_loss_weight', 0.5)),
        'Classifier_Feature_Time_s': feature_time,
        'Classifier_Train_Time_s': classifier_train_time,
        'Evaluation_Wall_Time_s': evaluation_time,
    })
    results.update(efficiency_metadata(model, opt))

    print(f'\n  - (RQ2 Classifier Head Summary)')
    print(f'    Variant:                  {variant}')
    print(f'    Feature dim/events:        {results["Classifier_Feature_Dim"]} / {results["Classifier_Train_Events"]}')
    print(
        f'    Best dev F1/FNR/FPR:       '
        f'{results["Classifier_Best_Dev_F1"]:.4f} / '
        f'{results["Classifier_Best_Dev_FNR"]:.4f} / '
        f'{results["Classifier_Best_Dev_FPR"]:.4f}'
    )
    print(
        f'    Segment F1/FNR/FPR:        '
        f'{results.get("Segment_F1", 0.0):.4f} / '
        f'{results.get("Segment_FNR", 0.0):.4f} / '
        f'{results.get("Segment_FPR", 0.0):.4f}'
    )
    print(
        f'    Event F1/FNR/FPR:          '
        f'{results.get("Event_F1", 0.0):.4f} / '
        f'{results.get("Event_FNR", 0.0):.4f} / '
        f'{results.get("Event_FPR", 0.0):.4f}'
    )
    print(
        f'    RQ2 Type/Time/Joint F1:    '
        f'{results.get("RQ2_TypeSegment_F1", 0.0):.4f} / '
        f'{results.get("RQ2_TimeSegment_F1", 0.0):.4f} / '
        f'{results.get("RQ2_JointSegment_F1", 0.0):.4f}'
    )
    return results


def decision_to_rq4_class(decision):
    rq4_pred = torch.zeros_like(decision, dtype=torch.long)
    rq4_pred = rq4_pred.masked_fill(decision == EXPECTED_DRIFT_DECISION, 1)
    rq4_pred = rq4_pred.masked_fill(decision == ANOMALY_DECISION, 2)
    rq4_pred = rq4_pred.masked_fill(decision == REJECT_DECISION, 3)
    return rq4_pred


def update_rq4_counts(confusion, pred_class, true_class, valid_mask):
    label_mask = valid_mask & (true_class > 0)
    if label_mask.sum().item() == 0:
        return 0
    pred = pred_class[label_mask].detach().cpu().long()
    truth = true_class[label_mask].detach().cpu().long()
    for t, p in zip(truth.reshape(-1).tolist(), pred.reshape(-1).tolist()):
        if t in RQ4_CLASS_NAMES:
            confusion[(int(t), int(p))] += 1
    return int(label_mask.sum().item())


def finalize_rq4_metrics(confusion, prefix='RQ4', total_key='Labeled_Events'):
    results = {}
    per_class_f1 = {}
    f1_values = []
    total = 0
    correct = 0
    for class_id, name in RQ4_CLASS_NAMES.items():
        tp = confusion.get((class_id, class_id), 0)
        pred_total = sum(count for (truth, pred), count in confusion.items() if pred == class_id and truth in RQ4_CLASS_NAMES)
        true_total = sum(count for (truth, pred), count in confusion.items() if truth == class_id)
        fp = pred_total - tp
        fn = true_total - tp
        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / true_total if true_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        f1_values.append(f1)
        total += true_total
        correct += tp
        results[f'{prefix}_{name}_TP'] = tp
        results[f'{prefix}_{name}_FP'] = fp
        results[f'{prefix}_{name}_FN'] = fn
        results[f'{prefix}_{name}_Precision'] = precision
        results[f'{prefix}_{name}_Recall'] = recall
        results[f'{prefix}_{name}_F1'] = f1
        per_class_f1[name] = f1
        results[f'{prefix}_{name}_True_Count'] = true_total
        for pred_id, pred_name in RQ4_PRED_CLASS_NAMES.items():
            transition_count = confusion.get((class_id, pred_id), 0)
            results[f'{prefix}_{name}_to_{pred_name}_Count'] = transition_count
            results[f'{prefix}_{name}_to_{pred_name}_Rate'] = (
                transition_count / true_total if true_total > 0 else 0.0
            )

    pred_expected = sum(count for (truth, pred), count in confusion.items() if pred == 1 and truth in RQ4_CLASS_NAMES)
    false_expected = sum(
        count
        for (truth, pred), count in confusion.items()
        if pred == 1 and truth in {2, 3}
    )
    results[f'{prefix}_{total_key}'] = total
    results[f'{prefix}_Accuracy'] = correct / total if total > 0 else 0.0
    results[f'{prefix}_Macro_F1'] = sum(f1_values) / len(f1_values) if f1_values else 0.0
    results[f'{prefix}_EU_Avg_F1'] = (
        (per_class_f1.get('Expected', 0.0) + per_class_f1.get('Unexpected', 0.0)) / 2.0
    )
    results[f'{prefix}_Memory_Contamination'] = false_expected / pred_expected if pred_expected > 0 else 0.0
    results[f'{prefix}_False_Expected_Events'] = false_expected
    expected_total = sum(count for (truth, _), count in confusion.items() if truth == 1)
    unexpected_total = sum(count for (truth, _), count in confusion.items() if truth == 2)
    eu_total = expected_total + unexpected_total
    eu_covered = sum(
        count
        for (truth, pred), count in confusion.items()
        if truth in {1, 2} and pred in {1, 2}
    )
    eu_covered_correct = confusion.get((1, 1), 0) + confusion.get((2, 2), 0)
    eu_covered_wrong = confusion.get((1, 2), 0) + confusion.get((2, 1), 0)
    unexpected_false_accept = confusion.get((2, 1), 0)
    expected_false_alarm = confusion.get((1, 2), 0)
    expected_reject = confusion.get((1, 3), 0)
    unexpected_reject = confusion.get((2, 3), 0)
    results[f'{prefix}_EU_Total'] = eu_total
    results[f'{prefix}_Selective_Coverage_EU'] = eu_covered / eu_total if eu_total > 0 else 0.0
    results[f'{prefix}_Selective_Risk_EU'] = eu_covered_wrong / eu_covered if eu_covered > 0 else 0.0
    results[f'{prefix}_Selective_Accuracy_EU'] = eu_covered_correct / eu_covered if eu_covered > 0 else 0.0
    results[f'{prefix}_Unexpected_False_Acceptance_Rate'] = (
        unexpected_false_accept / unexpected_total if unexpected_total > 0 else 0.0
    )
    results[f'{prefix}_Expected_False_Alarm_Rate'] = (
        expected_false_alarm / expected_total if expected_total > 0 else 0.0
    )
    results[f'{prefix}_Expected_Review_Burden_Rate'] = (
        (expected_false_alarm + expected_reject) / expected_total if expected_total > 0 else 0.0
    )
    results[f'{prefix}_Unexpected_Reject_Rate'] = (
        unexpected_reject / unexpected_total if unexpected_total > 0 else 0.0
    )
    results[f'{prefix}_Unexpected_Safe_Rate'] = (
        (confusion.get((2, 2), 0) + unexpected_reject) / unexpected_total
        if unexpected_total > 0 else 0.0
    )

    forced = Counter()
    for (truth, pred), count in confusion.items():
        if truth not in {1, 2}:
            continue
        forced_pred = 1 if pred == 1 else 2
        forced[(truth, forced_pred)] += count
    forced_f1_values = []
    for class_id, name in ((1, 'Expected'), (2, 'Unexpected')):
        tp = forced.get((class_id, class_id), 0)
        pred_total = sum(count for (truth, pred), count in forced.items() if pred == class_id)
        true_total = sum(count for (truth, pred), count in forced.items() if truth == class_id)
        precision = tp / pred_total if pred_total > 0 else 0.0
        recall = tp / true_total if true_total > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        forced_f1_values.append(f1)
        results[f'{prefix}_Forced_{name}_Precision'] = precision
        results[f'{prefix}_Forced_{name}_Recall'] = recall
        results[f'{prefix}_Forced_{name}_F1'] = f1
    results[f'{prefix}_Forced_EU_Macro_F1'] = (
        sum(forced_f1_values) / len(forced_f1_values) if forced_f1_values else 0.0
    )
    return results


def init_rq4_risk_coverage_store():
    return {'score': [], 'risk': []}


def update_rq4_risk_coverage(store, true_class, pred_class, confidence_score, valid_mask):
    label_mask = valid_mask & ((true_class == 1) | (true_class == 2))
    if label_mask.sum().item() == 0:
        return
    confidence = confidence_score[label_mask].detach().cpu().float().reshape(-1).tolist()
    risk = (pred_class[label_mask] != true_class[label_mask]).detach().cpu().long().reshape(-1).tolist()
    store['score'].extend(float(x) for x in confidence)
    store['risk'].extend(int(x) for x in risk)


def finalize_rq4_risk_coverage(store, prefix='RQ4'):
    scores = np.asarray(store.get('score', []), dtype=np.float64)
    risks = np.asarray(store.get('risk', []), dtype=np.float64)
    if scores.size == 0:
        return {
            f'{prefix}_Risk_Coverage_Points': 0,
            f'{prefix}_Risk_Coverage_AURC': 0.0,
        }
    order = np.argsort(-scores)
    sorted_risks = risks[order]
    coverage_index = np.arange(1, sorted_risks.size + 1, dtype=np.float64)
    cumulative_risk = np.cumsum(sorted_risks) / coverage_index
    return {
        f'{prefix}_Risk_Coverage_Points': int(sorted_risks.size),
        f'{prefix}_Risk_Coverage_AURC': float(np.mean(cumulative_risk)),
        f'{prefix}_Risk_Coverage_RiskAt25': float(cumulative_risk[max(0, int(math.ceil(0.25 * sorted_risks.size)) - 1)]),
        f'{prefix}_Risk_Coverage_RiskAt50': float(cumulative_risk[max(0, int(math.ceil(0.50 * sorted_risks.size)) - 1)]),
        f'{prefix}_Risk_Coverage_RiskAt75': float(cumulative_risk[max(0, int(math.ceil(0.75 * sorted_risks.size)) - 1)]),
        f'{prefix}_Risk_Coverage_RiskAt100': float(cumulative_risk[-1]),
    }


def print_rq4_transition_rates(results, prefix='RQ4', title='RQ4 transition rates'):
    if f'{prefix}_Expected_True_Count' not in results:
        return
    print(f'    {title} (true -> Normal/Expected/Unexpected/Reject):')
    for name in RQ4_CLASS_NAMES.values():
        base = f'{prefix}_{name}'
        print(
            f'      {name:<10} n={int(results.get(f"{base}_True_Count", 0)):<6} '
            f'{results.get(f"{base}_to_Normal_Rate", 0.0):.4f} / '
            f'{results.get(f"{base}_to_Expected_Rate", 0.0):.4f} / '
            f'{results.get(f"{base}_to_Unexpected_Rate", 0.0):.4f} / '
            f'{results.get(f"{base}_to_Reject_Rate", 0.0):.4f}'
        )


RQ3_METHOD_NAMES = ('Base', 'BaseReject', 'Full')


def init_rq3_confusions():
    return {name: Counter() for name in RQ3_METHOD_NAMES}


def _rq3_rate(numerator, denominator):
    return numerator / denominator if denominator > 0 else 0.0


def update_rq3_event_counts(confusions, pred_by_method, true_class, valid_mask):
    label_mask = valid_mask & (true_class > 0)
    if label_mask.sum().item() == 0:
        return 0

    truth = true_class[label_mask].detach().cpu().long().reshape(-1).tolist()
    for method, pred_class in pred_by_method.items():
        pred = pred_class[label_mask].detach().cpu().long().reshape(-1).tolist()
        for t, p in zip(truth, pred):
            if t in RQ4_CLASS_NAMES:
                confusions[method][(int(t), int(p))] += 1
    return int(label_mask.sum().item())


def update_rq3_segment_counts(confusions, pred_by_method, true_class, valid_mask):
    batch_size = true_class.size(0)
    updated = 0
    for row in range(batch_size):
        row_mask = valid_mask[row] & (true_class[row] > 0)
        if row_mask.sum().item() == 0:
            continue
        true_id = _segment_diagnosis_class(true_class[row][row_mask])
        if true_id == 0:
            continue
        for method, pred_class in pred_by_method.items():
            pred_id = _segment_diagnosis_class(pred_class[row][row_mask])
            confusions[method][(true_id, pred_id)] += 1
        updated += 1
    return updated


def finalize_rq3_metrics(confusions, prefix='RQ3', level='Event'):
    results = {}
    base_confusion = confusions.get('Base', Counter())
    base_ed_total = sum(count for (truth, _), count in base_confusion.items() if truth == 1)
    base_ed_far = _rq3_rate(base_confusion.get((1, 2), 0), base_ed_total)

    for method in RQ3_METHOD_NAMES:
        confusion = confusions.get(method, Counter())
        ed_total = sum(count for (truth, _), count in confusion.items() if truth == 1)
        ud_total = sum(count for (truth, _), count in confusion.items() if truth == 2)
        reject_total = sum(count for (truth, _), count in confusion.items() if truth == 3)
        labeled_total = ed_total + ud_total + reject_total

        ed_anomaly = confusion.get((1, 2), 0)
        ed_reject = confusion.get((1, 3), 0)
        ud_anomaly = confusion.get((2, 2), 0)
        ud_reject = confusion.get((2, 3), 0)
        pred_expected = sum(
            count
            for (truth, pred), count in confusion.items()
            if pred == 1 and truth in RQ4_CLASS_NAMES
        )
        false_expected = sum(
            count
            for (truth, pred), count in confusion.items()
            if pred == 1 and truth in {2, 3}
        )

        ed_far = _rq3_rate(ed_anomaly, ed_total)
        ed_burden = _rq3_rate(ed_anomaly + ed_reject, ed_total)
        if method == 'Base' or base_ed_far <= 0:
            ed_reduction = 0.0
        else:
            ed_reduction = 1.0 - (ed_far / base_ed_far)

        key = f'{prefix}_{method}_{level}'
        results[f'{key}_Labeled_Total'] = labeled_total
        results[f'{key}_ED_Total'] = ed_total
        results[f'{key}_UD_Total'] = ud_total
        results[f'{key}_Reject_Total'] = reject_total
        results[f'{key}_ED_Anomaly'] = ed_anomaly
        results[f'{key}_ED_Reject'] = ed_reject
        results[f'{key}_UD_Anomaly'] = ud_anomaly
        results[f'{key}_UD_Reject'] = ud_reject
        results[f'{key}_Pred_Expected'] = pred_expected
        results[f'{key}_False_Expected'] = false_expected
        results[f'{key}_ED_FAR'] = ed_far
        results[f'{key}_ED_Burden'] = ed_burden
        results[f'{key}_ED_Reduction'] = ed_reduction
        results[f'{key}_UD_Recall'] = _rq3_rate(ud_anomaly, ud_total)
        results[f'{key}_UD_SafeRate'] = _rq3_rate(ud_anomaly + ud_reject, ud_total)
        results[f'{key}_Contamination'] = _rq3_rate(false_expected, pred_expected)

    return results


def init_rq4_coverage_counts():
    counts = {}
    for class_id, name in RQ4_CLASS_NAMES.items():
        counts[f'{name}_Total'] = 0
        counts[f'{name}_Triggered'] = 0
        counts[f'{name}_TraditionalCandidate'] = 0
        counts[f'{name}_DiagnosisCandidate'] = 0
    return counts


def update_rq4_coverage_counts(counts, true_class, pred_class, traditional_pred, diagnosis_candidate, valid_mask):
    label_mask = valid_mask & (true_class > 0)
    if label_mask.sum().item() == 0:
        return
    for class_id, name in RQ4_CLASS_NAMES.items():
        class_mask = label_mask & (true_class == class_id)
        total = int(class_mask.sum().item())
        if total == 0:
            continue
        counts[f'{name}_Total'] += total
        counts[f'{name}_Triggered'] += int(((pred_class > 0) & class_mask).sum().item())
        counts[f'{name}_TraditionalCandidate'] += int((traditional_pred & class_mask).sum().item())
        counts[f'{name}_DiagnosisCandidate'] += int((diagnosis_candidate & class_mask).sum().item())


def finalize_rq4_coverage_metrics(counts):
    results = {}
    for _, name in RQ4_CLASS_NAMES.items():
        total = counts.get(f'{name}_Total', 0)
        triggered = counts.get(f'{name}_Triggered', 0)
        candidate = counts.get(f'{name}_TraditionalCandidate', 0)
        diagnosis_candidate = counts.get(f'{name}_DiagnosisCandidate', 0)
        results[f'RQ4_{name}_Total'] = total
        results[f'RQ4_{name}_Triggered'] = triggered
        results[f'RQ4_{name}_Triggered_Rate'] = triggered / total if total > 0 else 0.0
        results[f'RQ4_{name}_Traditional_Candidate_Rate'] = candidate / total if total > 0 else 0.0
        results[f'RQ4_{name}_Diagnosis_Candidate_Rate'] = diagnosis_candidate / total if total > 0 else 0.0
    return results


def _majority_positive_class(values):
    values = values.detach().cpu().long().reshape(-1)
    values = values[values > 0]
    if values.numel() == 0:
        return 0
    counts = torch.bincount(values, minlength=4)
    return int(torch.argmax(counts[1:4]).item() + 1)


def _segment_diagnosis_class(values):
    values = values.detach().cpu().long().reshape(-1)
    values = values[values > 0]
    if values.numel() == 0:
        return 0
    unique = set(int(x) for x in values.tolist())
    if 3 in unique or (1 in unique and 2 in unique):
        return 3
    return _majority_positive_class(values)


def smooth_rq4_window_predictions(pred_class, candidate_mask, valid_mask, opt):
    if not getattr(opt, 'use_rq4_window_diagnosis', False):
        return pred_class

    smoothed = pred_class.clone()
    expected_min = float(getattr(opt, 'rq4_window_expected_min_frac', 0.50))
    unexpected_min = float(getattr(opt, 'rq4_window_unexpected_min_frac', 0.25))
    reject_min = float(getattr(opt, 'rq4_window_reject_min_frac', 0.20))
    conflict_min = float(getattr(opt, 'rq4_window_conflict_min_frac', 0.20))

    batch_size = pred_class.size(0)
    for row in range(batch_size):
        row_mask = candidate_mask[row] & valid_mask[row]
        total = int(row_mask.sum().item())
        if total == 0:
            continue

        values = pred_class[row][row_mask].detach().long().clamp(min=0, max=3)
        counts = torch.bincount(values.cpu(), minlength=4).float()
        expected_frac = float(counts[1].item()) / total
        unexpected_frac = float(counts[2].item()) / total
        reject_frac = float(counts[3].item()) / total

        segment_class = 0
        if expected_frac >= conflict_min and unexpected_frac >= conflict_min:
            segment_class = 3
        elif unexpected_frac >= unexpected_min:
            segment_class = 2
        elif reject_frac >= reject_min:
            segment_class = 3
        elif expected_frac >= expected_min:
            segment_class = 1
        else:
            positive_counts = counts[1:4]
            if positive_counts.sum().item() > 0:
                segment_class = int(torch.argmax(positive_counts).item() + 1)

        if segment_class > 0:
            smoothed[row][row_mask] = segment_class

    return smoothed


def update_rq4_segment_counts(confusion, pred_class, true_class, valid_mask, pred_valid_mask=None):
    batch_size = true_class.size(0)
    updated = 0
    for row in range(batch_size):
        row_valid = valid_mask[row]
        truth_values = true_class[row][row_valid & (true_class[row] > 0)]
        if truth_values.numel() == 0:
            continue
        if pred_valid_mask is None:
            pred_mask = row_valid
        else:
            pred_mask = row_valid & pred_valid_mask[row]
        true_id = _majority_positive_class(truth_values)
        pred_id = _segment_diagnosis_class(pred_class[row][pred_mask])
        confusion[(true_id, pred_id)] += 1
        updated += 1
    return updated


def masked_row_max(values, mask):
    if values.dim() != 2:
        raise ValueError('masked_row_max expects 2D tensors.')
    valid_row = mask.any(dim=1)
    neg_inf = torch.full_like(values, -float('inf'))
    row_max = torch.where(mask, values, neg_inf).max(dim=1).values
    row_max = row_max.masked_fill(~valid_row, 0.0)
    return row_max, valid_row


def _masked_mean(values, mask):
    masked = values.masked_fill(~mask, 0.0)
    denom = mask.float().sum(dim=1).clamp_min(1.0)
    return masked.sum(dim=1) / denom


def _masked_topk_mean(values, mask, k):
    valid_row = mask.any(dim=1)
    k = max(1, int(k))
    neg_inf = torch.full_like(values, -float('inf'))
    masked = torch.where(mask, values, neg_inf)
    k_eff = min(k, values.size(1))
    topk = torch.topk(masked, k=k_eff, dim=1).values
    finite = torch.isfinite(topk)
    topk = topk.masked_fill(~finite, 0.0)
    denom = finite.float().sum(dim=1).clamp_min(1.0)
    score = topk.sum(dim=1) / denom
    score = score.masked_fill(~valid_row, 0.0)
    return score, valid_row


def aggregate_segment_scores(anomaly_score, mask, opt, gamma=None):
    mode = getattr(opt, 'segment_score_mode', 'max')
    if mode == 'max':
        return masked_row_max(anomaly_score, mask)
    if mode == 'mean':
        return _masked_mean(anomaly_score, mask), mask.any(dim=1)
    if mode == 'topk_mean':
        return _masked_topk_mean(anomaly_score, mask, getattr(opt, 'segment_topk', 3))

    if gamma is None:
        gamma = 0.0
    gamma = torch.as_tensor(gamma, device=anomaly_score.device, dtype=anomaly_score.dtype)
    excess = torch.relu(anomaly_score - gamma).masked_fill(~mask, 0.0)

    if mode == 'alert_fraction':
        alerts = ((anomaly_score > gamma) & mask).float()
        return _masked_mean(alerts, mask), mask.any(dim=1)
    if mode == 'excess_mean':
        return _masked_mean(excess, mask), mask.any(dim=1)
    if mode == 'excess_topk_mean':
        return _masked_topk_mean(excess, mask, getattr(opt, 'segment_topk', 3))

    raise ValueError(f'Unknown segment_score_mode={mode}')


def _score_calibration_stats(type_values, time_values, profile_values=None, conditional_gap_values=None):
    def stats(values):
        if values.numel() == 0:
            return 0.0, 1.0
        values = values.float()
        mean = float(values.mean().item())
        std = float(values.std(unbiased=False).clamp_min(1e-6).item())
        return mean, std

    type_mean, type_std = stats(type_values)
    time_mean, time_std = stats(time_values)
    result = {
        'type_mean': type_mean,
        'type_std': type_std,
        'time_mean': time_mean,
        'time_std': time_std,
    }
    if profile_values is not None:
        profile_mean, profile_std = stats(profile_values)
        result.update({
            'profile_mean': profile_mean,
            'profile_std': profile_std,
        })
    if conditional_gap_values is not None:
        conditional_gap_mean, conditional_gap_std = stats(conditional_gap_values)
        result.update({
            'conditional_gap_mean': conditional_gap_mean,
            'conditional_gap_std': conditional_gap_std,
        })
    return result


def _safe_quantile(values, quantile, default=float('inf')):
    if values is None or values.numel() == 0:
        return float(default)
    q = min(max(float(quantile), 0.0), 1.0)
    return float(torch.quantile(values.detach().float().reshape(-1), q).item())


def build_anomaly_score(scores, opt):
    mode = getattr(opt, 'anomaly_score_mode', 'raw')
    type_nll = scores['type_nll']
    time_nll = scores['time_nll']
    type_weight = float(getattr(opt, 'type_score_weight', 1.0))
    time_weight = float(getattr(opt, 'time_score_weight', 1.0))
    profile_score = scores.get('profile_score')
    if profile_score is None:
        profile_score = torch.zeros_like(type_nll)
    profile_weight = float(getattr(opt, 'profile_score_weight', 1.0))
    conditional_gap_score = scores.get('conditional_gap_score')
    if conditional_gap_score is None:
        conditional_gap_score = torch.zeros_like(type_nll)
    conditional_gap_weight = float(getattr(opt, 'conditional_gap_weight', 1.0))

    if mode == 'raw':
        anomaly_score = type_nll + time_nll
    elif mode == 'type_only':
        anomaly_score = type_nll
    elif mode == 'time_only':
        anomaly_score = time_nll
    elif mode == 'weighted_raw':
        anomaly_score = type_weight * type_nll + time_weight * time_nll
    elif mode == 'zscore':
        calib = getattr(opt, '_score_calibration', None) or {}
        type_mean = float(calib.get('type_mean', 0.0))
        type_std = max(float(calib.get('type_std', 1.0)), 1e-6)
        time_mean = float(calib.get('time_mean', 0.0))
        time_std = max(float(calib.get('time_std', 1.0)), 1e-6)
        anomaly_score = (
            type_weight * ((type_nll - type_mean) / type_std)
            + time_weight * ((time_nll - time_mean) / time_std)
        )
    elif mode == 'zscore_max':
        calib = getattr(opt, '_score_calibration', None) or {}
        type_mean = float(calib.get('type_mean', 0.0))
        type_std = max(float(calib.get('type_std', 1.0)), 1e-6)
        time_mean = float(calib.get('time_mean', 0.0))
        time_std = max(float(calib.get('time_std', 1.0)), 1e-6)
        type_z = type_weight * ((type_nll - type_mean) / type_std)
        time_z = time_weight * ((time_nll - time_mean) / time_std)
        anomaly_score = torch.maximum(type_z, time_z)
    elif mode == 'profile_only':
        anomaly_score = profile_score
    elif mode == 'profile_zscore':
        calib = getattr(opt, '_score_calibration', None) or {}
        type_mean = float(calib.get('type_mean', 0.0))
        type_std = max(float(calib.get('type_std', 1.0)), 1e-6)
        time_mean = float(calib.get('time_mean', 0.0))
        time_std = max(float(calib.get('time_std', 1.0)), 1e-6)
        profile_mean = float(calib.get('profile_mean', 0.0))
        profile_std = max(float(calib.get('profile_std', 1.0)), 1e-6)
        anomaly_score = (
            type_weight * ((type_nll - type_mean) / type_std)
            + time_weight * ((time_nll - time_mean) / time_std)
            + profile_weight * ((profile_score - profile_mean) / profile_std)
        )
    elif mode == 'conditional_gap_zscore':
        calib = getattr(opt, '_score_calibration', None) or {}
        type_mean = float(calib.get('type_mean', 0.0))
        type_std = max(float(calib.get('type_std', 1.0)), 1e-6)
        time_mean = float(calib.get('time_mean', 0.0))
        time_std = max(float(calib.get('time_std', 1.0)), 1e-6)
        gap_mean = float(calib.get('conditional_gap_mean', 0.0))
        gap_std = max(float(calib.get('conditional_gap_std', 1.0)), 1e-6)
        type_z = type_weight * ((type_nll - type_mean) / type_std)
        time_z = time_weight * ((time_nll - time_mean) / time_std)
        gap_z = conditional_gap_weight * ((conditional_gap_score - gap_mean) / gap_std)
        anomaly_score = torch.maximum(torch.maximum(type_z, time_z), gap_z)
    else:
        raise ValueError(f'Unknown anomaly_score_mode={mode}')

    return anomaly_score.masked_fill(~scores['mask'], 0.0)


def compute_best_threshold_metrics(scores, labels, steps=200, extra_thresholds=None):
    if scores.numel() == 0:
        return None

    scores = scores.detach().cpu().float().reshape(-1)
    labels = labels.detach().cpu().long().reshape(-1)
    valid = labels >= 0
    scores = scores[valid]
    labels = labels[valid]
    if scores.numel() == 0 or labels.unique().numel() < 2:
        return None

    n_steps = max(2, int(steps))
    quantiles = torch.linspace(0.0, 1.0, steps=n_steps)
    thresholds = torch.quantile(scores, quantiles).unique()
    thresholds = torch.cat([
        scores.min().view(1) - 1e-6,
        thresholds,
        scores.max().view(1) + 1e-6,
    ]).unique()
    if extra_thresholds:
        extra = torch.tensor(
            [float(x) for x in extra_thresholds],
            dtype=scores.dtype,
            device=scores.device,
        )
        thresholds = torch.cat([thresholds, extra]).unique()

    best = None
    for threshold in thresholds:
        pred = scores > threshold
        truth = labels.bool()
        tp = int((pred & truth).sum().item())
        fp = int((pred & ~truth).sum().item())
        fn = int((~pred & truth).sum().item())
        tn = int((~pred & ~truth).sum().item())
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        item = {
            'threshold': float(threshold.item()),
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'fpr': fpr,
            'alerts': int(pred.sum().item()),
            'alert_rate': float(pred.float().mean().item()),
        }
        if best is None or item['f1'] > best['f1']:
            best = item
    return best


def _detect_labeled_dataset(data_path):
    path = (data_path or '').replace('\\', '/').lower().rstrip('/')
    if path.endswith('/'):
        path = path[:-1]
    if 'labeled_hdfs_v3' in path:
        return 'hdfs_v3'
    if 'labeled_hdfs' in path:
        return 'hdfs'
    if 'labeled_bgl' in path:
        return 'bgl'
    if 'labeled_thunderbird' in path:
        return 'thunderbird'
    if 'labeled_spirit' in path:
        return 'spirit'
    if 'labeled_liberty' in path:
        return 'liberty'
    return None


class NormalTraceProfile:
    """Normal-only transition/profile scorer for sequence-level log datasets."""

    def __init__(self, num_types, opt):
        self.num_types = int(num_types)
        self.smoothing = float(getattr(opt, 'profile_smoothing', 0.1))
        self.bigram_weight = float(getattr(opt, 'profile_bigram_weight', 1.0))
        self.unigram_weight = float(getattr(opt, 'profile_unigram_weight', 0.25))
        self.hist_weight = float(getattr(opt, 'profile_hist_weight', 0.0))
        self.length_weight = float(getattr(opt, 'profile_length_weight', 0.0))
        self.exact_signature_weight = float(getattr(opt, 'profile_exact_signature_weight', 0.0))
        self.hist_signature_weight = float(getattr(opt, 'profile_hist_signature_weight', 0.0))
        self.max_sequences = int(getattr(opt, 'profile_max_sequences', 200000))
        size = self.num_types + 1
        self.type_counts = torch.zeros(size, dtype=torch.float64)
        self.prev_counts = torch.zeros(size, dtype=torch.float64)
        self.bigram_counts = torch.zeros(size, size, dtype=torch.float64)
        self.hist_sum = torch.zeros(size, dtype=torch.float64)
        self.hist_sq_sum = torch.zeros(size, dtype=torch.float64)
        self.total_targets = 0.0
        self.length_sum = 0.0
        self.length_sq_sum = 0.0
        self.num_sequences = 0
        self.sequence_signatures = set()
        self.hist_signatures = set()
        self.ready = False
        self._device_cache = {}

    def update_event_types(self, event_type):
        event_type = event_type.detach().cpu().long()
        for row in event_type:
            valid = row[row != Constants.PAD]
            if valid.numel() < 2:
                continue
            if self.max_sequences > 0 and self.num_sequences >= self.max_sequences:
                break

            valid = valid.clamp(min=0, max=self.num_types)
            valid_list = tuple(int(x) for x in valid.tolist())
            self.sequence_signatures.add(valid_list)
            self.hist_signatures.add(self._hist_signature(valid_list))
            prev = valid[:-1]
            target = valid[1:]
            ones = torch.ones_like(target, dtype=torch.float64)
            self.type_counts.scatter_add_(0, target, ones)
            self.prev_counts.scatter_add_(0, prev, ones)
            self.bigram_counts.index_put_((prev, target), ones, accumulate=True)
            self.total_targets += float(target.numel())

            hist = torch.bincount(valid, minlength=self.num_types + 1).double()
            hist = hist / float(valid.numel())
            self.hist_sum += hist
            self.hist_sq_sum += hist * hist
            length = float(valid.numel())
            self.length_sum += length
            self.length_sq_sum += length * length
            self.num_sequences += 1

    def finalize(self):
        if self.num_sequences == 0 or self.total_targets <= 0:
            self.ready = False
            return self

        n = float(self.num_sequences)
        self.mean_hist = (self.hist_sum / n).float()
        hist_var = (self.hist_sq_sum / n) - (self.hist_sum / n) ** 2
        self.hist_scale = float(hist_var.clamp_min(0.0).sqrt().mean().clamp_min(1e-6).item())
        self.mean_length = self.length_sum / n
        length_var = max(self.length_sq_sum / n - self.mean_length ** 2, 1e-6)
        self.std_length = math.sqrt(length_var)
        self.ready = True
        return self

    @staticmethod
    def _hist_signature(types):
        return tuple(sorted(Counter(types).items()))

    def _cache_for(self, device):
        key = str(device)
        if key not in self._device_cache:
            self._device_cache[key] = {
                'type_counts': self.type_counts.to(device=device, dtype=torch.float32),
                'prev_counts': self.prev_counts.to(device=device, dtype=torch.float32),
                'bigram_counts': self.bigram_counts.to(device=device, dtype=torch.float32),
                'mean_hist': self.mean_hist.to(device=device, dtype=torch.float32),
            }
        return self._device_cache[key]

    def score_event_map(self, event_type):
        target = event_type[:, 1:]
        if not self.ready:
            return torch.zeros_like(target, dtype=torch.float32)

        device = event_type.device
        cache = self._cache_for(device)
        target = target.long().clamp(min=0, max=self.num_types)
        prev = event_type[:, :-1].long().clamp(min=0, max=self.num_types)
        mask = target != Constants.PAD

        type_counts = cache['type_counts']
        prev_counts = cache['prev_counts']
        bigram_counts = cache['bigram_counts']
        smoothing = max(self.smoothing, 1e-8)
        support = float(max(self.num_types, 1))

        target_count = type_counts[target]
        prev_count = prev_counts[prev]
        bigram_count = bigram_counts[prev, target]
        type_prob = (target_count + smoothing) / (float(self.total_targets) + smoothing * support)
        bigram_prob = (bigram_count + smoothing) / (prev_count + smoothing * support)
        score = (
            self.unigram_weight * (-torch.log(type_prob.clamp_min(1e-12)))
            + self.bigram_weight * (-torch.log(bigram_prob.clamp_min(1e-12)))
        )

        if (
            self.hist_weight > 0
            or self.length_weight > 0
            or self.exact_signature_weight > 0
            or self.hist_signature_weight > 0
        ):
            seq_score = torch.zeros(event_type.size(0), device=device, dtype=torch.float32)
            mean_hist = cache['mean_hist']
            for idx, row in enumerate(event_type):
                valid = row[row != Constants.PAD].long().clamp(min=0, max=self.num_types)
                if valid.numel() == 0:
                    continue
                valid_list = tuple(int(x) for x in valid.detach().cpu().tolist())
                if self.exact_signature_weight > 0 and valid_list not in self.sequence_signatures:
                    seq_score[idx] += self.exact_signature_weight
                if self.hist_signature_weight > 0 and self._hist_signature(valid_list) not in self.hist_signatures:
                    seq_score[idx] += self.hist_signature_weight
                if self.hist_weight > 0:
                    hist = torch.bincount(valid, minlength=self.num_types + 1).float().to(device)
                    hist = hist / float(valid.numel())
                    hist_dev = torch.mean(torch.abs(hist - mean_hist)) / max(self.hist_scale, 1e-6)
                    seq_score[idx] += self.hist_weight * hist_dev
                if self.length_weight > 0:
                    len_z = abs((float(valid.numel()) - self.mean_length) / max(self.std_length, 1e-6))
                    seq_score[idx] += self.length_weight * min(len_z, 10.0)
            score = score + seq_score.unsqueeze(1)

        return score.masked_fill(~mask, 0.0)


def fit_trace_profile(calibration_data, opt):
    if not getattr(opt, 'enable_trace_profile', False):
        opt._trace_profile = None
        return None

    num_types = int(getattr(opt, 'num_types', 0))
    if num_types <= 0:
        opt._trace_profile = None
        return None

    profile = NormalTraceProfile(num_types, opt)
    for batch in tqdm(calibration_data, desc='  - (Trace Profile) ', leave=False):
        event_type = batch[2]
        profile.update_event_types(event_type)
        if profile.max_sequences > 0 and profile.num_sequences >= profile.max_sequences:
            break
    profile.finalize()
    opt._trace_profile = profile if profile.ready else None
    if profile.ready:
        print(
            f'[Info] Trace profile fitted: sequences={profile.num_sequences} | '
            f'targets={int(profile.total_targets)} | types={profile.num_types}'
        )
    else:
        print('[Warning] Trace profile requested, but no calibration sequences were available.')
    return opt._trace_profile


def augment_scores_with_trace_profile(scores, event_type, opt):
    profile = getattr(opt, '_trace_profile', None)
    if profile is None or not getattr(profile, 'ready', False):
        return scores
    augmented = dict(scores)
    augmented['profile_score'] = profile.score_event_map(event_type).to(scores['type_nll'].device)
    return augmented


def fit_type_gap_profile(type_ids, gap_values, min_count=20):
    type_ids = type_ids.detach().cpu().long().reshape(-1)
    gap_values = gap_values.detach().cpu().float().reshape(-1)
    if type_ids.numel() == 0 or gap_values.numel() == 0:
        return None

    valid = (type_ids > 0) & torch.isfinite(gap_values)
    type_ids = type_ids[valid]
    gap_values = gap_values[valid]
    if type_ids.numel() == 0:
        return None

    global_mean = float(gap_values.mean().item())
    global_std = float(gap_values.std(unbiased=False).clamp_min(1e-6).item())
    max_type = int(type_ids.max().item())
    means = torch.full((max_type + 1,), global_mean, dtype=torch.float32)
    stds = torch.full((max_type + 1,), global_std, dtype=torch.float32)
    counts = torch.zeros(max_type + 1, dtype=torch.long)

    for type_id in torch.unique(type_ids):
        mask = type_ids == type_id
        values = gap_values[mask]
        counts[int(type_id.item())] = values.numel()
        if values.numel() >= int(min_count):
            means[int(type_id.item())] = values.mean()
            stds[int(type_id.item())] = values.std(unbiased=False).clamp_min(1e-6)

    return {
        'means': means,
        'stds': stds,
        'counts': counts,
        'global_mean': global_mean,
        'global_std': global_std,
        'min_count': int(min_count),
    }


def type_gap_profile_score(event_type, time_gap_norm, opt):
    profile = getattr(opt, '_type_gap_profile', None)
    target_type = event_type[:, 1:].long()
    score = torch.zeros_like(time_gap_norm, dtype=torch.float32)
    if profile is None:
        return score

    means = profile['means'].to(time_gap_norm.device)
    stds = profile['stds'].to(time_gap_norm.device).clamp_min(1e-6)
    if target_type.numel() > 0 and int(target_type.max().item()) >= means.numel():
        pad = int(target_type.max().item()) + 1 - means.numel()
        means = torch.cat([
            means,
            torch.full((pad,), float(profile['global_mean']), device=time_gap_norm.device)
        ])
        stds = torch.cat([
            stds,
            torch.full((pad,), max(float(profile['global_std']), 1e-6), device=time_gap_norm.device)
        ])

    idx = target_type.clamp(min=0, max=means.numel() - 1)
    gap_z = torch.abs((time_gap_norm.float() - means[idx]) / stds[idx])
    return gap_z.masked_fill(target_type.eq(Constants.PAD), 0.0)


def augment_scores_with_type_gap_profile(scores, event_type, time_gap_norm, opt):
    augmented = dict(scores)
    augmented['conditional_gap_score'] = type_gap_profile_score(event_type, time_gap_norm, opt).to(scores['type_nll'].device)
    return augmented


def configure_dataset_adaptive_detection(opt):
    dataset = _detect_labeled_dataset(getattr(opt, 'data', ''))
    opt.detected_dataset = dataset or 'generic'
    if getattr(opt, 'disable_dataset_adaptive_detection', False):
        return

    if dataset == 'hdfs':
        opt.enable_trace_profile = True
        opt.anomaly_score_mode = 'profile_only'
        opt.segment_score_mode = 'max'
        opt.type_score_weight = 0.0
        opt.time_score_weight = 0.0
        opt.profile_score_weight = 1.0
        opt.profile_bigram_weight = 0.0
        opt.profile_unigram_weight = 0.0
        opt.profile_hist_weight = 0.0
        opt.profile_length_weight = 0.0
        opt.profile_exact_signature_weight = 0.0
        opt.profile_hist_signature_weight = 1.0
        print('[Info] Dataset-adaptive detection enabled for HDFS.')
    elif dataset == 'thunderbird':
        opt.enable_trace_profile = True
        opt.anomaly_score_mode = 'profile_only'
        opt.segment_score_mode = 'max'
        opt.segment_topk = max(5, int(getattr(opt, 'segment_topk', 3)))
        opt.type_score_weight = 0.0
        opt.time_score_weight = 0.0
        opt.profile_score_weight = 1.0
        opt.profile_bigram_weight = 0.0
        opt.profile_unigram_weight = 1.0
        opt.profile_hist_weight = 0.0
        opt.profile_length_weight = 0.0
        opt.profile_exact_signature_weight = 0.0
        opt.profile_hist_signature_weight = 0.0
        opt.anomaly_quantile = max(float(getattr(opt, 'anomaly_quantile', 0.99)), 0.9995)
        print('[Info] Dataset-adaptive detection enabled for Thunderbird.')
    elif dataset == 'bgl':
        opt.enable_trace_profile = True
        opt.anomaly_score_mode = 'profile_only'
        opt.segment_score_mode = 'max'
        opt.type_score_weight = 0.0
        opt.time_score_weight = 0.0
        opt.profile_score_weight = 1.0
        opt.profile_bigram_weight = 0.0
        opt.profile_unigram_weight = 1.0
        opt.profile_hist_weight = 0.0
        opt.profile_length_weight = 0.0
        opt.profile_exact_signature_weight = 0.0
        opt.profile_hist_signature_weight = 0.0
        opt.anomaly_quantile = max(float(getattr(opt, 'anomaly_quantile', 0.99)), 0.9995)
        print('[Info] Dataset-adaptive detection enabled for BGL.')
    elif dataset == 'spirit':
        opt.enable_trace_profile = True
        opt.anomaly_score_mode = 'profile_only'
        opt.segment_score_mode = 'max'
        opt.type_score_weight = 0.0
        opt.time_score_weight = 0.0
        opt.profile_score_weight = 1.0
        opt.profile_bigram_weight = 0.0
        opt.profile_unigram_weight = 1.0
        opt.profile_hist_weight = 0.0
        opt.profile_length_weight = 0.0
        opt.profile_exact_signature_weight = 0.0
        opt.profile_hist_signature_weight = 0.0
        opt.anomaly_quantile = max(float(getattr(opt, 'anomaly_quantile', 0.99)), 0.9995)
        print('[Info] Dataset-adaptive detection enabled for Spirit.')


def ensure_time_gap_normalized(time_gap_norm, opt, context='batch', event_type=None):
    """
    Guard against stale dataset preprocessing on dev/test splits.

    Some benchmark pickles or older Dataset.py versions can yield raw gaps for
    validation/test while training uses normalized gaps. A pure max-threshold
    check misses raw batches whose largest gap is still modest, but those values
    can explode after log-space denormalization.
    """
    if getattr(opt, 'disable_time_norm_guard', False):
        return time_gap_norm

    if time_gap_norm.numel() == 0:
        return time_gap_norm

    valid_values = time_gap_norm.detach()
    if event_type is not None and event_type.dim() >= 2:
        valid_mask = event_type[:, 1:] != Constants.PAD
        if valid_mask.shape == time_gap_norm.shape and valid_mask.any():
            valid_values = valid_values[valid_mask]

    threshold = float(getattr(opt, 'time_norm_guard_threshold', 20.0))
    max_abs = float(valid_values.abs().max().item())
    min_val = float(valid_values.min().item())
    max_val = float(valid_values.max().item())

    context_modes = getattr(opt, '_time_norm_guard_context_modes', {})
    context_mode = context_modes.get(context)
    if context_mode == 'normalized':
        return time_gap_norm

    needs_normalization = max_abs > threshold
    if context_mode == 'raw':
        needs_normalization = True

    if not needs_normalization and context != 'train' and opt.normalize == 'log':
        train_norm_min = float(getattr(opt, 'time_min', 0.0))
        train_norm_max = float(getattr(opt, 'time_max', threshold))
        # Log-normalized gaps usually include negative values because small raw
        # gaps map below the training log mean. Raw gaps are non-negative. This
        # catches batches with raw max <= threshold, the source of inf RMSE on
        # Invoice/Financial.
        non_negative_batch = min_val >= -1e-8
        training_has_negative_support = train_norm_min < -0.05
        outside_train_support = max_val > train_norm_max + 0.25
        needs_normalization = (
            non_negative_batch
            and training_has_negative_support
            and (outside_train_support or max_val > 0.0)
        )

    if not needs_normalization:
        if context != 'train' and min_val < -1e-8:
            context_modes[context] = 'normalized'
            opt._time_norm_guard_context_modes = context_modes
        return time_gap_norm

    if not hasattr(opt, 'mean_data'):
        return time_gap_norm

    if context != 'train':
        context_modes[context] = 'raw'
        opt._time_norm_guard_context_modes = context_modes

    warned = getattr(opt, '_time_norm_guard_warned', set())
    if context not in warned:
        print(
            f'[Warning] Detected raw time gaps in {context} '
            f'(min={min_val:.4f}, max={max_val:.4f}); '
            f'applying {opt.normalize} normalization guard.'
        )
        warned.add(context)
        opt._time_norm_guard_warned = warned

    if opt.normalize == 'log':
        mean_data = max(float(opt.mean_data), 1e-12)
        mean_log_data = float(getattr(opt, 'mean_log_data', 0.0))
        var_log_data = max(float(getattr(opt, 'var_log_data', 1.0)), 1e-12)
        safe_gap = torch.clamp(time_gap_norm, min=0.0)
        return (torch.log(safe_gap / mean_data + 1e-9) - mean_log_data) / var_log_data

    if opt.normalize == 'normal':
        mean_data = max(float(opt.mean_data), 1e-12)
        return time_gap_norm / mean_data

    return time_gap_norm


def train_epoch(model, training_data, optimizer, opt):
    model.train()
    total_loss = 0
    total_fm = 0
    total_type = 0
    total_events = 0

    for batch in tqdm(training_data, mininterval=2, desc='  - (Training)   ', leave=False):
        event_time, time_gap_norm, event_type, _ = unpack_batch(batch, opt.device)
        time_gap_norm = ensure_time_gap_normalized(time_gap_norm, opt, context='train', event_type=event_type)
        optimizer.zero_grad()
        
        enc_out, prediction = model(event_type, event_time, time_gap_norm)
        
        loss, fm_val, type_val = model.compute_loss_diagnostic(prediction, event_type)
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        num_event = (event_type[:, 1:] != Constants.PAD).sum().item()
        total_loss += loss.item() * num_event
        total_fm += fm_val * num_event
        total_type += type_val * num_event
        total_events += num_event

    return total_loss / total_events, total_fm / total_events, total_type / total_events

def eval_epoch(model, validation_data, eval_generation, opt):
    model.eval()

    if not eval_generation:
        total_loss = 0
        total_fm = 0
        total_type = 0
        total_correct = 0
        total_events = 0
        fm_debug_enabled = getattr(opt, 'fm_debug', False)
        fm_debug = {
            'residual_max': 0.0,
            'v_abs_max': 0.0,
            'u_abs_max': 0.0,
            'x1_abs_max': 0.0,
            'c_abs_max': 0.0,
            'c_norm_sum': 0.0,
            'events': 0
        }
        with torch.no_grad():
            for batch in validation_data:
                event_time, time_gap_norm, event_type, _ = unpack_batch(batch, opt.device)
                time_gap_norm = ensure_time_gap_normalized(time_gap_norm, opt, context='valid', event_type=event_type)
                _, prediction = model(event_type, event_time, time_gap_norm)
                loss, fm_val, type_val = model.compute_loss_diagnostic(prediction, event_type)
                
                pred_type = prediction['type_logits'].argmax(dim=-1)
                truth = event_type[:, 1:] - 1
                mask = (truth != -1)
                correct = (pred_type[mask] == truth[mask]).sum().item()
                num_event = mask.sum().item()
                
                total_loss += loss.item() * num_event
                total_fm += fm_val * num_event
                total_type += type_val * num_event
                total_correct += correct
                total_events += num_event

                if fm_debug_enabled and hasattr(model, 'compute_flow_diagnostics'):
                    diag = model.compute_flow_diagnostics(prediction, event_type)
                    fm_debug['residual_max'] = max(fm_debug['residual_max'], diag['residual_max'])
                    fm_debug['v_abs_max'] = max(fm_debug['v_abs_max'], diag['v_abs_max'])
                    fm_debug['u_abs_max'] = max(fm_debug['u_abs_max'], diag['u_abs_max'])
                    fm_debug['x1_abs_max'] = max(fm_debug['x1_abs_max'], diag['x1_abs_max'])
                    fm_debug['c_abs_max'] = max(fm_debug['c_abs_max'], diag['c_abs_max'])
                    fm_debug['c_norm_sum'] += diag['c_norm_mean'] * diag['num_events']
                    fm_debug['events'] += diag['num_events']
        
        avg_loss = total_loss / total_events if total_events > 0 else 0
        avg_fm = total_fm / total_events if total_events > 0 else 0
        avg_type = total_type / total_events if total_events > 0 else 0
        avg_acc = total_correct / total_events if total_events > 0 else 0
        if fm_debug_enabled and avg_fm > getattr(opt, 'fm_debug_threshold', 1000.0):
            c_norm_mean = fm_debug['c_norm_sum'] / max(1, fm_debug['events'])
            print(
                '  - (FM Debug) '
                f'avg_fm={avg_fm:.4f} | residual_max={fm_debug["residual_max"]:.4f} | '
                f'v_abs_max={fm_debug["v_abs_max"]:.4f} | u_abs_max={fm_debug["u_abs_max"]:.4f} | '
                f'x1_abs_max={fm_debug["x1_abs_max"]:.4f} | '
                f'c_norm_mean={c_norm_mean:.4f} | c_abs_max={fm_debug["c_abs_max"]:.4f}'
            )
        return avg_loss, avg_acc, avg_fm, avg_type

    else:
        # Top-K Mixture Generation Logic
        class VelocityWrapper:
            def __init__(self, v_field, c_cond):
                self.v_field = v_field
                self.c_cond = c_cond
            def __call__(self, t, x):
                t_tensor = t * torch.ones_like(x)
                return self.v_field(x, t_tensor, self.c_cond)

        solver = ODESolver(None)

        n_quantiles = len(opt.eval_quantile)
        total_hits = torch.zeros(n_quantiles, device=opt.device)
        total_il = 0
        total_crps = 0
        total_acc = 0
        total_events = 0
        
        total_time_nll = 0.0
        total_type_nll = 0.0
        total_nll_events = 0
        total_sse = 0.0

        with torch.no_grad():
            for batch in tqdm(validation_data, desc='  - (Sampling)   ', leave=False):
                event_time, time_gap_norm, event_type, _ = unpack_batch(batch, opt.device)
                time_gap_norm = ensure_time_gap_normalized(time_gap_norm, opt, context='sampling', event_type=event_type)

                # --- 1. NLL Calculation ---
                if hasattr(model, 'get_exact_log_likelihood'):
                    batch_time_nll_sum, batch_n_events = model.get_exact_log_likelihood(event_type, event_time, time_gap_norm)
                    
                    non_pad_mask = get_non_pad_mask(event_type)
                    enc_output = model.encoder(event_type, event_time, non_pad_mask, time_gap_norm)
                    c = enc_output[:, :-1, :]
                    
                    c_type = model.type_projector(c)
                    if hasattr(model, '_compute_type_logits'):
                        type_logits = model._compute_type_logits(event_type, c_type)
                    else:
                        type_logits = model.type_predictor(c_type)
                    
                    target_labels = event_type[:, 1:] - 1
                    mask = (target_labels != -1)
                    
                    type_nll_loss = F.cross_entropy(
                        type_logits.view(-1, model.num_types), 
                        target_labels.view(-1), 
                        reduction='none',
                        ignore_index=-1
                    )
                    type_nll_sum = type_nll_loss.sum()
                    
                    total_time_nll += batch_time_nll_sum.item()
                    total_type_nll += type_nll_sum.item()
                    total_nll_events += batch_n_events

                # --- 2. Generation with Top-K Mixture ---
                non_pad_mask = get_non_pad_mask(event_type)
                enc_output = model.encoder(event_type, event_time, non_pad_mask, time_gap_norm)
                c = enc_output[:, :-1, :]
                
                # Decouple
                c_type = model.type_projector(c)
                c_time = model.time_projector(c)

                B, L, D = c.shape

                # 2.1 获取 Top-K 概率
                if hasattr(model, '_compute_type_logits'):
                    type_logits = model._compute_type_logits(event_type, c_type)
                else:
                    type_logits = model.type_predictor(c_type)
                type_probs = torch.softmax(type_logits, dim=-1)
                K = min(3, type_probs.shape[-1])  # Top-3 Mixture
                topk_probs, topk_indices = torch.topk(type_probs, K, dim=-1)
                topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

                # 2.2 准备 Mixture Condition
                c_time_expanded = c_time.unsqueeze(2).expand(-1, -1, K, -1)
                topk_type_emb = model.encoder.event_type_emb(topk_indices + 1)
                
                # Fusion
                c_cond_k = model.type_fusion(c_time_expanded, topk_type_emb)
                flow_cond_clip = float(getattr(opt, 'flow_cond_clip', 0.0))
                if flow_cond_clip > 0:
                    c_cond_k = torch.clamp(c_cond_k, min=-flow_cond_clip, max=flow_cond_clip)
                c_cond_flat = c_cond_k.view(B * L * K, D)

                # 2.3 ODE 求解
                x0 = torch.randn(B * L * K, 1, 1, device=opt.device) * opt.fm_sigma
                wrapper = VelocityWrapper(model.v_field, c_cond_flat.unsqueeze(1))
                solver.velocity_model = wrapper

                x1_flat = solver.sample(
                    x0,
                    torch.tensor([0.0, 1.0]),
                    method=opt.solver_method,
                    step_size=opt.solver_step_size
                )
                
                # [核心修复] 使用动态的 clamp_threshold 防止 Log 空间数值爆炸
                if opt.normalize == 'log':
                    clamp_val = getattr(opt, 'clamp_threshold', 6.0)
                    x1_flat = torch.clamp(x1_flat, min=-clamp_val, max=clamp_val)

                # 2.4 计算 Weighted Mean
                t_pred_k_norm = x1_flat.view(B, L, K)
                t_pred_k_real = model.denormalize_time(t_pred_k_norm)
                if getattr(opt, 'eval_time_scale', 'legacy') == 'physical':
                    max_time_val = getattr(opt, 'time_raw_max', getattr(opt, 'time_max', 1e7)) * 5.0
                else:
                    max_time_val = getattr(opt, 'time_rel_max', getattr(opt, 'time_max', 1e7)) * 5.0
                t_pred_k_real = torch.clamp(t_pred_k_real, min=0.0, max=max_time_val)
                t_pred_mean = (t_pred_k_real * topk_probs).sum(dim=-1)

                # 2.5 计算 RMSE
                if opt.normalize == 'log':
                    gt_log_arg = time_gap_norm * opt.var_log_data + opt.mean_log_data
                    gt_log_arg = torch.clamp(gt_log_arg, min=-80.0, max=80.0)
                    gt_t_real = torch.exp(gt_log_arg)
                    if getattr(opt, 'eval_time_scale', 'legacy') == 'physical':
                        gt_t_real = gt_t_real * opt.mean_data
                elif opt.normalize == 'normal':
                    gt_t_real = time_gap_norm * opt.mean_data
                else:
                    gt_t_real = time_gap_norm
                
                mask_loss = (event_type[:, 1:] != Constants.PAD)
                
                t_pred_integer = torch.clamp(t_pred_mean - 0.5, min=0.0)
                gt_integer = torch.floor(gt_t_real)
                
                diff = (t_pred_integer - gt_integer)[mask_loss]
                total_events += mask_loss.sum().item()

                # --- 3. Argmax 采样 ---
                N = opt.n_samples
                best_type = topk_indices[:, :, 0]
                pred_types_input = best_type + 1
                if hasattr(model, '_build_flow_condition'):
                    c_cond_best = model._build_flow_condition(c_time, pred_types_input)
                else:
                    pred_type_emb_best = model.encoder.event_type_emb(pred_types_input)
                    c_cond_best = model.type_fusion(c_time, pred_type_emb_best)
                
                c_cond_expanded = c_cond_best.repeat_interleave(N, dim=0)
                
                x0_samples = torch.randn(B * N, L, 1, device=opt.device) * opt.fm_sigma
                wrapper_samples = VelocityWrapper(model.v_field, c_cond_expanded)
                solver.velocity_model = wrapper_samples
                
                x1_samples = solver.sample(x0_samples, torch.tensor([0.0, 1.0]), method=opt.solver_method, step_size=opt.solver_step_size)
                
                # 采样阶段同样加入截断机制保证 CRPS 指标稳定
                if opt.normalize == 'log':
                    clamp_val = getattr(opt, 'clamp_threshold', 6.0)
                    x1_samples = torch.clamp(x1_samples, min=-clamp_val, max=clamp_val)
                
                t_sample_norm = x1_samples.reshape(B, N, L, -1).squeeze(-1).permute(0, 2, 1)

                metrics = Utils.evaluate_samples(
                    t_sample_norm,
                    time_gap_norm,
                    best_type,
                    event_type,
                    opt
                )

                if metrics['total_events'] > 0:
                    total_hits += metrics['hit_counts']
                    total_il += metrics['il_sum']
                    total_crps += metrics['crps_sum']
                    total_acc += metrics['correct_type']
                    # Report RMSE from the same sample distribution used by
                    # CS/CRPS, instead of the separate Top-K single-path mean.
                    total_sse += metrics.get('sse_sum', 0.0)

        if total_events == 0: return {}

        final_acc = total_acc / total_events
        final_rmse = np.sqrt(total_sse / total_events) if total_events > 0 else 0.0
        
        if total_nll_events > 0:
            avg_time_nll = total_time_nll / total_nll_events
            avg_type_nll = total_type_nll / total_nll_events
            avg_joint_nll = avg_time_nll + avg_type_nll
        else:
            avg_time_nll, avg_type_nll, avg_joint_nll = 0, 0, 0

        actual_coverage = total_hits / total_events
        target_coverage = opt.eval_quantile 
        mse = ((actual_coverage - target_coverage) ** 2).mean()
        final_cs = torch.sqrt(mse).item()

        print(f'\n  - (Test Summary)')
        print(f'    Acc:       {final_acc:.4f}')
        print(f'    RMSE:      {final_rmse:.4f}')
        print(f'    Time NLL:  {avg_time_nll:.4f}')
        print(f'    Type NLL:  {avg_type_nll:.4f}')
        print(f'    Joint NLL: {avg_joint_nll:.4f}')
        print(f'    CS:        {final_cs:.4f}')

        return {
            'Acc': final_acc,
            'NLL': avg_joint_nll,
            'Time_NLL': avg_time_nll,
            'Type_NLL': avg_type_nll,
            'RMSE': final_rmse,
            'CS': final_cs
        }


def fit_calibration_bank(model, calibration_data, opt):
    model.eval()
    bank = CalibrationMemoryBank(
        max_size=opt.calibration_max_size if opt.calibration_max_size > 0 else None,
        device='cpu'
    )
    type_parts = []
    time_parts = []
    profile_parts = []
    uncertainty_parts = []
    feature_parts = []
    gaussian_parts = []
    target_type_parts = []
    gap_value_parts = []

    with torch.no_grad():
        for batch in tqdm(calibration_data, desc='  - (Calibrating) ', leave=False):
            event_time, time_gap_norm, event_type, _ = unpack_batch(batch, opt.device)
            time_gap_norm = ensure_time_gap_normalized(time_gap_norm, opt, context='calibration', event_type=event_type)
            scores = model.compute_reliability_scores(
                event_type,
                event_time,
                time_gap_norm,
                uncertainty_mc=opt.uncertainty_mc,
                type_entropy_weight=opt.type_entropy_weight,
                exact_time_nll=opt.reliability_exact_nll or opt.use_ensemble_correction,
                return_memory_features=opt.use_ensemble_correction
            )
            scores = augment_scores_with_trace_profile(scores, event_type, opt)
            mask = scores['mask']
            if mask.any():
                type_parts.append(scores['type_nll'][mask].detach().cpu().float())
                time_parts.append(scores['time_nll'][mask].detach().cpu().float())
                target_type_parts.append(event_type[:, 1:][mask].detach().cpu().long())
                gap_value_parts.append(time_gap_norm[mask].detach().cpu().float())
                if 'profile_score' in scores:
                    profile_parts.append(scores['profile_score'][mask].detach().cpu().float())
                uncertainty_parts.append(scores['uncertainty_score'][mask].detach().cpu().float())
                if opt.use_ensemble_correction and scores.get('memory_feature') is not None:
                    feature_parts.append(scores['memory_feature'][mask].detach().cpu().float())
                if opt.use_ensemble_correction and scores.get('gaussian_latent') is not None:
                    gaussian_parts.append(scores['gaussian_latent'][mask].detach().cpu().float())

    if not type_parts:
        raise ValueError('Calibration set produced no valid events.')

    type_values = torch.cat(type_parts).float()
    time_values = torch.cat(time_parts).float()
    profile_values = torch.cat(profile_parts).float() if profile_parts else None
    target_type_values = torch.cat(target_type_parts).long() if target_type_parts else None
    gap_values = torch.cat(gap_value_parts).float() if gap_value_parts else None
    opt._type_gap_profile = fit_type_gap_profile(
        target_type_values,
        gap_values,
        min_count=getattr(opt, 'conditional_gap_min_count', 20)
    ) if target_type_values is not None and gap_values is not None else None
    conditional_gap_values = None
    if opt._type_gap_profile is not None:
        flat_event_type = torch.cat([
            torch.zeros_like(target_type_values[:1]),
            target_type_values
        ]).unsqueeze(0).to(opt.device)
        flat_gap = gap_values.unsqueeze(0).to(opt.device)
        conditional_gap_values = type_gap_profile_score(flat_event_type, flat_gap, opt).reshape(-1).detach().cpu().float()
    uncertainty_values = torch.cat(uncertainty_parts).float()
    opt._score_calibration = _score_calibration_stats(
        type_values,
        time_values,
        profile_values,
        conditional_gap_values
    )
    type_q = float(getattr(opt, 'drift_diagnosis_type_quantile', 0.995))
    time_q = float(getattr(opt, 'drift_diagnosis_time_quantile', 0.995))
    profile_candidate_q = float(getattr(opt, 'drift_candidate_profile_quantile', 0.95))
    profile_q = float(getattr(opt, 'drift_diagnosis_profile_quantile', 0.995))
    strong_q = float(getattr(opt, 'drift_diagnosis_strong_quantile', 0.999))
    extreme_time_q = float(getattr(opt, 'drift_diagnosis_extreme_time_quantile', 0.9995))
    opt._component_thresholds = {
        'type_gamma': _safe_quantile(type_values, type_q),
        'type_strong_gamma': _safe_quantile(type_values, strong_q),
        'time_gamma': _safe_quantile(time_values, time_q),
        'time_strong_gamma': _safe_quantile(time_values, strong_q),
        'time_extreme_gamma': _safe_quantile(time_values, extreme_time_q),
        'profile_candidate_gamma': _safe_quantile(profile_values, profile_candidate_q),
        'profile_gamma': _safe_quantile(profile_values, profile_q),
        'profile_strong_gamma': _safe_quantile(profile_values, strong_q),
    }

    flat_scores = {
        'type_nll': type_values,
        'time_nll': time_values,
        'mask': torch.ones_like(type_values, dtype=torch.bool),
    }
    if profile_values is not None:
        flat_scores['profile_score'] = profile_values
    if conditional_gap_values is not None:
        flat_scores['conditional_gap_score'] = conditional_gap_values
    anomaly_values = build_anomaly_score(flat_scores, opt).detach().cpu().float()
    features = torch.cat(feature_parts).float() if feature_parts else None
    gaussian_latent = torch.cat(gaussian_parts).float() if gaussian_parts else None
    bank.update(
        anomaly_values,
        uncertainty_values,
        mask=None,
        features=features,
        gaussian_latent=gaussian_latent
    )

    gamma, delta = bank.thresholds(
        anomaly_quantile=opt.anomaly_quantile,
        uncertainty_quantile=opt.uncertainty_quantile
    )
    if opt.use_ensemble_correction:
        fitted = bank.fit_feature_space(
            k=opt.ensemble_k,
            support_quantile=opt.ensemble_support_quantile,
            max_reference=opt.ensemble_max_reference
        )
        if not fitted:
            print('[Warning] Ensemble correction requested, but calibration feature memory is empty.')
    return bank, gamma, delta


def fit_segment_anomaly_threshold(model, calibration_data, opt):
    model.eval()
    segment_scores = []
    bank_gamma = getattr(opt, '_gamma_anomaly', None)

    with torch.no_grad():
        for batch in tqdm(calibration_data, desc='  - (Segment Calib) ', leave=False):
            event_time, time_gap_norm, event_type, _ = unpack_batch(batch, opt.device)
            time_gap_norm = ensure_time_gap_normalized(time_gap_norm, opt, context='segment_calibration', event_type=event_type)
            scores = model.compute_reliability_scores(
                event_type,
                event_time,
                time_gap_norm,
                uncertainty_mc=opt.uncertainty_mc,
                type_entropy_weight=opt.type_entropy_weight,
                exact_time_nll=opt.reliability_exact_nll or opt.use_ensemble_correction
            )
            scores = augment_scores_with_trace_profile(scores, event_type, opt)
            scores = augment_scores_with_type_gap_profile(scores, event_type, time_gap_norm, opt)
            anomaly_score = build_anomaly_score(scores, opt)
            segment_score, valid_segment = aggregate_segment_scores(
                anomaly_score,
                scores['mask'],
                opt,
                gamma=bank_gamma
            )
            if valid_segment.any():
                segment_scores.append(segment_score[valid_segment].detach().cpu())

    if not segment_scores:
        return 0.0

    segment_scores = torch.cat(segment_scores).float()
    return torch.quantile(segment_scores, opt.anomaly_quantile).item()


class ProgressiveDriftAdapter:
    """
    StepWise-style lightweight adaptation for confirmed expected drift.

    The detector is frozen. Only a robust scalar affine map in the Gaussian
    time latent is estimated online, and only from high-confidence expected
    drift candidates.
    """
    def __init__(self, opt):
        self.enabled = bool(getattr(opt, 'use_drift_adapter', False))
        self.min_events = int(getattr(opt, 'drift_adapter_min_events', 32))
        self.fit_interval = max(1, int(getattr(opt, 'drift_adapter_fit_interval', 16)))
        self.min_support = float(getattr(opt, 'drift_adapter_min_support', 0.5))
        self.update_rate = float(getattr(opt, 'drift_adapter_update_rate', 0.5))
        self.slope_min = float(getattr(opt, 'drift_adapter_slope_min', 0.5))
        self.slope_max = float(getattr(opt, 'drift_adapter_slope_max', 2.0))
        self.shift_clip = float(getattr(opt, 'drift_adapter_shift_clip', 3.0))
        self.support_min = float(getattr(opt, 'drift_adapter_support_min', 0.75))
        self.support_max = float(getattr(opt, 'drift_adapter_support_max', 1.5))
        self.update_memory = bool(getattr(opt, 'drift_adapter_update_memory', False))
        self.max_buffer = int(getattr(opt, 'drift_adapter_max_buffer', 2048))

        self.slope = 1.0
        self.shift = 0.0
        self.support_scale = 1.0
        self.ready = False
        self.num_updates = 0
        self.num_refits = 0
        self.last_fit_updates = 0
        self.memory_updates = 0
        self._new_latents = []
        self._ref_latents = []
        self._distances = []
        self._supports = []
        self._features = []
        self._scores = []
        self._uncertainties = []

    def transform_features(self, features):
        if not self.enabled or not self.ready:
            return features
        aligned = features.clone()
        aligned[..., -1:] = aligned[..., -1:] * self.slope + self.shift
        return aligned

    def current_support_scale(self):
        if not self.enabled or not self.ready:
            return 1.0
        return self.support_scale

    def _trim(self):
        overflow = len(self._new_latents) - self.max_buffer
        if overflow <= 0:
            return
        self._new_latents = self._new_latents[overflow:]
        self._ref_latents = self._ref_latents[overflow:]
        self._distances = self._distances[overflow:]
        self._supports = self._supports[overflow:]
        self._features = self._features[overflow:]
        self._scores = self._scores[overflow:]
        self._uncertainties = self._uncertainties[overflow:]

    def observe(self, new_latent, ref_latent, local_distance, local_support, features, score, uncertainty):
        if not self.enabled or new_latent.numel() == 0:
            return 0

        support_mask = local_support.detach().float() >= self.min_support
        if support_mask.sum().item() == 0:
            return 0

        new_latent = new_latent.detach().float()[support_mask].reshape(-1, 1).cpu()
        ref_latent = ref_latent.detach().float()[support_mask].reshape(-1, 1).cpu()
        local_distance = local_distance.detach().float()[support_mask].reshape(-1).cpu()
        local_support = local_support.detach().float()[support_mask].reshape(-1).cpu()
        features = features.detach().float()[support_mask].cpu()
        score = score.detach().float()[support_mask].reshape(-1).cpu()
        uncertainty = uncertainty.detach().float()[support_mask].reshape(-1).cpu()

        for idx in range(new_latent.size(0)):
            self._new_latents.append(new_latent[idx])
            self._ref_latents.append(ref_latent[idx])
            self._distances.append(local_distance[idx])
            self._supports.append(local_support[idx])
            self._features.append(features[idx])
            self._scores.append(score[idx])
            self._uncertainties.append(uncertainty[idx])

        self.num_updates += int(new_latent.size(0))
        self._trim()

        if self.num_updates >= self.min_events and self.num_updates - self.last_fit_updates >= self.fit_interval:
            self.fit()

        return int(new_latent.size(0))

    def fit(self):
        if len(self._new_latents) < self.min_events:
            return False

        x = torch.stack(self._new_latents).reshape(-1)
        y = torch.stack(self._ref_latents).reshape(-1)
        x_med = x.median()
        y_med = y.median()
        dx = x - x_med
        dy = y - y_med
        valid = dx.abs() > 1e-5

        if valid.sum().item() > 0:
            slope = (dy[valid] / dx[valid]).median().item()
        else:
            slope = 1.0
        slope = min(max(float(slope), self.slope_min), self.slope_max)

        shift = (y - slope * x).median().item()
        if self.shift_clip > 0:
            shift = min(max(float(shift), -self.shift_clip), self.shift_clip)

        alpha = min(max(self.update_rate, 0.0), 1.0)
        self.slope = (1.0 - alpha) * self.slope + alpha * slope
        self.shift = (1.0 - alpha) * self.shift + alpha * shift

        supports = torch.stack(self._supports).reshape(-1)
        supports = supports[torch.isfinite(supports)].clamp(1e-6, 1.0)
        if supports.numel() > 0:
            # local_support = exp(-distance / radius). A stable drift cluster
            # near the support boundary justifies a slightly wider radius.
            target_support = 0.5
            ratio = (-torch.log(supports).median() / -math.log(target_support)).item()
            proposed_scale = min(max(float(ratio), self.support_min), self.support_max)
            self.support_scale = (1.0 - alpha) * self.support_scale + alpha * proposed_scale

        self.ready = True
        self.num_refits += 1
        self.last_fit_updates = self.num_updates
        return True

    def maybe_update_memory(self, bank):
        if not self.enabled or not self.ready or not self.update_memory or len(self._features) == 0:
            return 0

        features = torch.stack(self._features)
        scores = torch.stack(self._scores).reshape(-1)
        uncertainties = torch.stack(self._uncertainties).reshape(-1)
        latents = features[:, -1:].clone()
        latents = latents * self.slope + self.shift
        features = features.clone()
        features[:, -1:] = latents

        bank.update(
            scores,
            uncertainties,
            mask=None,
            features=features,
            gaussian_latent=latents
        )
        updated = int(features.size(0))
        self.memory_updates += updated
        self._features.clear()
        self._scores.clear()
        self._uncertainties.clear()
        return updated


def apply_ensemble_correction(
        model,
        bank,
        scores,
        event_type,
        event_time,
        time_gap_norm,
        decision,
        candidate_mask,
        gamma,
        delta,
        opt,
        drift_adapter=None,
        absence_context=None):
    corrected_score = scores['anomaly_score'].clone()
    expected_drift_mask = torch.zeros_like(candidate_mask, dtype=torch.bool)
    stats = {
        'candidates': 0,
        'supported': 0,
        'corrected': 0,
        'provisional_expected': 0,
        'adapter_updates': 0,
        'adapter_refits': 0,
        'adapter_memory_updates': 0,
        'support_sum': 0.0,
        'ensemble_score_sum': 0.0,
        'component_unexpected': 0,
        'component_reject': 0,
        'counterfactual_supported': 0,
        'counterfactual_conflict': 0,
        'oov_type_unavailable': 0,
        'absence_conflict': 0,
        'coverage_conflict': 0,
        'diagnostics': None,
    }

    if not getattr(opt, 'use_ensemble_correction', False):
        return decision, corrected_score, expected_drift_mask, stats

    if not candidate_mask.any() or 'memory_feature' not in scores or 'flow_condition' not in scores:
        return decision, corrected_score, expected_drift_mask, stats

    query_features = scores['memory_feature']
    if drift_adapter is not None:
        query_features = drift_adapter.transform_features(query_features)

    local = bank.sample_local_latents(
        query_features,
        candidate_mask,
        k=opt.ensemble_k,
        n_samples=opt.ensemble_samples,
        noise_scale=opt.ensemble_noise_scale,
        max_search_size=opt.ensemble_max_search,
        support_scale=drift_adapter.current_support_scale() if drift_adapter is not None else 1.0
    )
    if local is None:
        return decision, corrected_score, expected_drift_mask, stats

    sampled_latents = local['sampled_latents'].to(time_gap_norm.device).float()
    num_candidates, num_samples, latent_dim = sampled_latents.shape
    if num_candidates == 0:
        return decision, corrected_score, expected_drift_mask, stats

    candidate_cond = scores['flow_condition'][candidate_mask].to(time_gap_norm.device).float()
    z0 = sampled_latents.reshape(num_candidates * num_samples, 1, latent_dim)
    c_cond = candidate_cond.unsqueeze(1).expand(-1, num_samples, -1)
    c_cond = c_cond.reshape(num_candidates * num_samples, 1, -1)

    step_size = getattr(opt, 'ensemble_step_size', 0.0)
    if step_size <= 0:
        step_size = opt.solver_step_size

    x_samples = model.sample_time_from_latent(
        z0,
        c_cond,
        method=opt.solver_method,
        step_size=step_size
    )
    x_samples = x_samples.reshape(num_candidates, num_samples, latent_dim).squeeze(-1)
    if opt.normalize == 'log':
        clamp_val = getattr(opt, 'clamp_threshold', 6.0)
        x_samples = torch.clamp(x_samples, min=-clamp_val, max=clamp_val)

    obs = time_gap_norm[candidate_mask].unsqueeze(-1)
    kernel = float(getattr(opt, 'ensemble_kernel', 0.2))
    if kernel <= 0:
        sample_std = x_samples.std(dim=1, unbiased=False)
        kernel = max(float(sample_std.median().item()), 1e-3)

    log_terms = -0.5 * ((obs - x_samples) / kernel).pow(2)
    log_terms = log_terms - math.log(kernel) - 0.5 * math.log(2.0 * math.pi)
    sample_time_nll = -log_terms
    ensemble_time_nll = -(torch.logsumexp(log_terms, dim=1) - math.log(num_samples))
    ensemble_nll_std = sample_time_nll.std(dim=1, unbiased=False)
    disagreement_threshold = float(getattr(opt, 'ensemble_disagreement_threshold', 2.0))
    if disagreement_threshold > 0:
        ensemble_disagreement = ensemble_nll_std > disagreement_threshold
    else:
        ensemble_disagreement = torch.zeros_like(ensemble_time_nll, dtype=torch.bool)
    candidate_scores = {
        'type_nll': scores['type_nll'][candidate_mask],
        'time_nll': ensemble_time_nll,
        'mask': torch.ones_like(ensemble_time_nll, dtype=torch.bool),
    }
    if 'profile_score' in scores:
        candidate_scores['profile_score'] = scores['profile_score'][candidate_mask]
    ensemble_score = build_anomaly_score(candidate_scores, opt)

    correction_weight = float(getattr(opt, 'ensemble_correction_weight', 1.0))
    correction_weight = min(max(correction_weight, 0.0), 1.0)
    original_score = scores['anomaly_score'][candidate_mask]
    final_score = (1.0 - correction_weight) * original_score + correction_weight * ensemble_score

    local_supported = local['supported'].to(time_gap_norm.device).bool()
    high_uncertainty = scores['uncertainty_score'][candidate_mask] > delta
    low_uncertainty = ~high_uncertainty
    component_unexpected = torch.zeros_like(local_supported, dtype=torch.bool)
    component_reject = torch.zeros_like(local_supported, dtype=torch.bool)
    cf_type_ratio = torch.zeros_like(ensemble_time_nll, dtype=torch.float32)
    cf_time_ratio = torch.zeros_like(ensemble_time_nll, dtype=torch.float32)
    cf_type_strength = torch.zeros_like(ensemble_time_nll, dtype=torch.float32)
    cf_time_strength = torch.zeros_like(ensemble_time_nll, dtype=torch.float32)
    cf_valid_count = torch.zeros_like(ensemble_time_nll, dtype=torch.float32)
    context_supported = local_supported.clone()
    context_conflict = torch.zeros_like(local_supported, dtype=torch.bool)
    oov_type_unavailable = torch.zeros_like(local_supported, dtype=torch.bool)
    correction_gain = torch.clamp(original_score - final_score, min=0.0)
    absence_anomaly = torch.zeros_like(ensemble_time_nll, dtype=torch.float32)
    coverage_support = torch.ones_like(ensemble_time_nll, dtype=torch.float32)
    coverage_nn_cosine = torch.ones_like(ensemble_time_nll, dtype=torch.float32)
    absence_conflict = torch.zeros_like(local_supported, dtype=torch.bool)
    coverage_conflict = torch.zeros_like(local_supported, dtype=torch.bool)

    if getattr(opt, 'use_component_drift_diagnosis', False):
        thresholds = getattr(opt, '_component_thresholds', {}) or {}
        type_gamma = float(thresholds.get('type_gamma', float('inf')))
        type_strong_gamma = float(thresholds.get('type_strong_gamma', type_gamma))
        time_gamma = float(thresholds.get('time_gamma', float('inf')))
        time_strong_gamma = float(thresholds.get('time_strong_gamma', time_gamma))
        time_extreme_gamma = float(thresholds.get('time_extreme_gamma', time_strong_gamma))
        profile_gamma = float(thresholds.get('profile_gamma', float('inf')))
        profile_strong_gamma = float(thresholds.get('profile_strong_gamma', profile_gamma))

        candidate_type_nll = scores['type_nll'][candidate_mask]
        candidate_target_type = event_type[:, 1:][candidate_mask] - 1
        if getattr(opt, 'treat_oov_type_as_unavailable', False):
            oov_type_id = int(getattr(opt, 'oov_type_id', -1))
            if oov_type_id >= 0:
                oov_type_unavailable = candidate_target_type == oov_type_id
        candidate_profile_score = scores.get('profile_score')
        if candidate_profile_score is None:
            candidate_profile_score = torch.zeros_like(candidate_type_nll)
        else:
            candidate_profile_score = candidate_profile_score[candidate_mask]

        type_outlier = (candidate_type_nll > type_gamma) & (~oov_type_unavailable)
        type_strong = (candidate_type_nll > type_strong_gamma) & (~oov_type_unavailable)
        profile_outlier = candidate_profile_score > profile_gamma
        profile_strong = candidate_profile_score > profile_strong_gamma
        structural_outlier = type_outlier | profile_outlier
        strong_structural = type_strong | profile_strong
        candidate_raw_time_nll = scores['time_nll'][candidate_mask]
        raw_time_outlier = candidate_raw_time_nll > time_gamma
        raw_time_strong = candidate_raw_time_nll > time_strong_gamma
        raw_time_extreme = candidate_raw_time_nll > time_extreme_gamma
        time_outlier = ensemble_time_nll > time_gamma
        time_strong = ensemble_time_nll > time_strong_gamma
        score_high = final_score > gamma
        raw_score_high = original_score > gamma
        time_resolved = (~time_outlier) & (final_score <= gamma)
        mem_recover = (
            local_supported
            & time_resolved
            & (correction_gain >= float(getattr(opt, 'counterfactual_mem_gain_threshold', 0.0)))
        )

        use_cf_support = bool(getattr(opt, 'use_counterfactual_context_support', False))
        if use_cf_support:
            cf_maps = model.compute_counterfactual_context_support(
                event_type,
                event_time,
                time_gap_norm,
                candidate_mask,
                scores['type_nll'],
                scores['time_nll'],
                k=getattr(opt, 'counterfactual_support_k', 3),
                epsilon=getattr(opt, 'counterfactual_support_epsilon', 0.0),
                chunk_size=getattr(opt, 'counterfactual_support_chunk_size', 128),
                time_mode=getattr(opt, 'counterfactual_support_time_mode', 'exact')
            )
            cf_type_ratio = cf_maps['type_support_ratio'][candidate_mask].to(time_gap_norm.device).float()
            cf_time_ratio = cf_maps['time_support_ratio'][candidate_mask].to(time_gap_norm.device).float()
            cf_type_strength = cf_maps['type_support_strength'][candidate_mask].to(time_gap_norm.device).float()
            cf_time_strength = cf_maps['time_support_strength'][candidate_mask].to(time_gap_norm.device).float()
            cf_valid_count = cf_maps['valid_support_count'][candidate_mask].to(time_gap_norm.device).float()
            has_cf_evidence = cf_valid_count > 0
            type_support_ok = (
                (cf_type_ratio >= float(getattr(opt, 'counterfactual_type_support_ratio', 0.34)))
                & (cf_type_strength >= float(getattr(opt, 'counterfactual_type_support_strength', 0.0)))
            )
            # For OOV targets, UNK likelihood is not concrete template evidence.
            # Do not let a frequent UNK token provide type support.
            type_support_ok = type_support_ok & (~oov_type_unavailable)
            exact_cf_time = getattr(opt, 'counterfactual_support_time_mode', 'exact') != 'off'
            if not exact_cf_time:
                # Fast mode has no masked CNF time contribution. Use the
                # memory-ensemble recovery as temporal evidence instead of
                # treating every valid context as time-supported.
                time_support_ok = time_resolved
            else:
                time_support_ok = (
                    (cf_time_ratio >= float(getattr(opt, 'counterfactual_time_support_ratio', 0.34)))
                    & (cf_time_strength >= float(getattr(opt, 'counterfactual_time_support_strength', 0.0)))
                )
            oov_time_only_support = oov_type_unavailable & exact_cf_time & time_support_ok & mem_recover
            context_supported = has_cf_evidence & (
                ((~oov_type_unavailable) & type_support_ok & time_support_ok)
                | oov_time_only_support
            )
            context_conflict = has_cf_evidence & (
                ((~oov_type_unavailable) & (type_support_ok ^ time_support_ok))
                | (oov_type_unavailable & (~oov_time_only_support))
            )

        unresolved_temporal = time_outlier | (
            raw_time_outlier & (score_high | raw_score_high) & (~local_supported)
        )
        strong_temporal = time_strong | (
            raw_time_strong & (score_high | raw_score_high) & (~local_supported)
        )
        temporal_conflict = (
            raw_time_extreme
            & local_supported
            & time_resolved
            & (structural_outlier | high_uncertainty)
        )
        temporal_severity_veto = (
            raw_time_extreme
            & (~temporal_conflict)
            & (raw_score_high | score_high | time_outlier)
        )
        structural_unexpected = structural_outlier & (
            strong_structural | score_high | (~local_supported)
        )
        if getattr(opt, 'use_counterfactual_context_support', False):
            strong_anomaly = strong_structural | strong_temporal | temporal_severity_veto
            component_unexpected = (
                (strong_anomaly & (~context_supported) & (~mem_recover))
                | (
                    (~context_supported)
                    & (~mem_recover)
                    & (structural_outlier | unresolved_temporal | score_high)
                )
            )
            component_reject = (
                (high_uncertainty & (~component_unexpected))
                | context_conflict
                | temporal_conflict
                | (ensemble_disagreement & (~component_unexpected))
                | (oov_type_unavailable & (~context_supported))
                | (mem_recover & (strong_anomaly | score_high))
                | (context_supported & (~mem_recover) & (structural_outlier | unresolved_temporal | score_high))
                | ((~context_supported) & mem_recover)
                | ((~context_supported) & (~component_unexpected))
            ) & (~component_unexpected)
            provisional_expected = (
                context_supported
                & mem_recover
                & low_uncertainty
                & (~strong_anomaly)
                & (~temporal_conflict)
                & (~context_conflict)
                & (~ensemble_disagreement)
                & (~structural_outlier)
                & ((~oov_type_unavailable) | oov_time_only_support)
                & (~component_unexpected)
                & (~component_reject)
            )
        else:
            component_unexpected = (
                strong_structural
                | strong_temporal
                | temporal_severity_veto
                | structural_unexpected
                | ((~local_supported) & (structural_outlier | unresolved_temporal | score_high))
            )
            component_reject = (
                (high_uncertainty & (~component_unexpected))
                | temporal_conflict
                | (ensemble_disagreement & (~component_unexpected))
                | (structural_outlier & local_supported & time_resolved & (~strong_structural))
                | (local_supported & score_high & (~strong_structural) & (~strong_temporal))
                | ((~local_supported) & (~component_unexpected))
            ) & (~component_unexpected)
            provisional_expected = (
                local_supported
                & low_uncertainty
                & time_resolved
                & (~temporal_severity_veto)
                & (~temporal_conflict)
                & (~ensemble_disagreement)
                & (~structural_outlier)
                & (~component_unexpected)
                & (~component_reject)
            )
    else:
        provisional_expected = local_supported & low_uncertainty & (final_score <= gamma)
    provisional_expected = provisional_expected.bool()
    component_unexpected = component_unexpected.bool()
    component_reject = component_reject.bool()
    if getattr(opt, 'use_absence_aware_revision', False) and absence_context is not None:
        full_absence = absence_context.get('absence_anomaly')
        full_coverage = absence_context.get('coverage_support')
        full_nn_cosine = absence_context.get('coverage_nn_cosine')
        full_absence_conflict = absence_context.get('absence_conflict')
        full_coverage_conflict = absence_context.get('coverage_conflict')
        if full_absence is not None:
            absence_anomaly = full_absence[candidate_mask].to(time_gap_norm.device).float()
        if full_coverage is not None:
            coverage_support = full_coverage[candidate_mask].to(time_gap_norm.device).float()
        if full_nn_cosine is not None:
            coverage_nn_cosine = full_nn_cosine[candidate_mask].to(time_gap_norm.device).float()
        if full_absence_conflict is not None:
            absence_conflict = full_absence_conflict[candidate_mask].to(time_gap_norm.device).bool()
        if full_coverage_conflict is not None:
            coverage_conflict = full_coverage_conflict[candidate_mask].to(time_gap_norm.device).bool()

        absence_revision_conflict = (
            (absence_conflict | coverage_conflict)
            & local_supported
            & (final_score <= gamma)
            & (~component_unexpected)
        )
        provisional_expected = provisional_expected & (~absence_conflict) & (~coverage_conflict)
        component_reject = (component_reject | absence_revision_conflict) & (~component_unexpected)
    if drift_adapter is not None and drift_adapter.enabled:
        candidate_features = scores['memory_feature'][candidate_mask].to(time_gap_norm.device).float()
        refits_before = drift_adapter.num_refits
        adapter_updates = drift_adapter.observe(
            scores['gaussian_latent'][candidate_mask][provisional_expected],
            local['neighbor_latent_mean'].to(time_gap_norm.device)[provisional_expected],
            local['local_distance'].to(time_gap_norm.device)[provisional_expected],
            local['local_support'].to(time_gap_norm.device)[provisional_expected],
            candidate_features[provisional_expected],
            final_score[provisional_expected],
            scores['uncertainty_score'][candidate_mask][provisional_expected]
        )
        if adapter_updates > 0 and not drift_adapter.ready:
            drift_adapter.fit()
        memory_updates = drift_adapter.maybe_update_memory(bank)
        adapter_ready = bool(drift_adapter.ready)
        corrected_to_expected = provisional_expected if adapter_ready else torch.zeros_like(provisional_expected, dtype=torch.bool)
        provisional_reject = torch.zeros_like(provisional_expected, dtype=torch.bool) if adapter_ready else provisional_expected
        stats['adapter_updates'] = adapter_updates
        stats['adapter_refits'] = drift_adapter.num_refits - refits_before
        stats['adapter_memory_updates'] = memory_updates
    else:
        corrected_to_expected = provisional_expected
        provisional_reject = torch.zeros_like(provisional_expected, dtype=torch.bool)

    if not getattr(opt, 'use_component_drift_diagnosis', False):
        component_unexpected = (~local_supported) & (final_score > gamma)
        component_reject = (
            (local_supported & (final_score > gamma) & high_uncertainty)
            | high_uncertainty
            | ensemble_disagreement
        )

    updated_decision = decision.clone()
    candidate_decision = updated_decision[candidate_mask].clone()
    candidate_decision[corrected_to_expected] = EXPECTED_DRIFT_DECISION
    candidate_decision[component_unexpected] = ANOMALY_DECISION
    candidate_decision[component_reject | provisional_reject] = REJECT_DECISION
    updated_decision[candidate_mask] = candidate_decision

    corrected_score[candidate_mask] = final_score
    expected_drift_mask[candidate_mask] = corrected_to_expected

    local_support = local['local_support'].to(time_gap_norm.device).float()
    if getattr(opt, 'save_rq4_event_details', False):
        diagnostics = {
            'local_distance': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'local_support': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'ensemble_score': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'ensemble_nll_std': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'ensemble_supported': torch.zeros_like(candidate_mask, dtype=torch.bool),
            'ensemble_disagreement': torch.zeros_like(candidate_mask, dtype=torch.bool),
            'component_unexpected': torch.zeros_like(candidate_mask, dtype=torch.bool),
            'component_reject': torch.zeros_like(candidate_mask, dtype=torch.bool),
            'cf_type_ratio': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'cf_time_ratio': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'cf_type_strength': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'cf_time_strength': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'cf_valid_count': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'context_supported': torch.zeros_like(candidate_mask, dtype=torch.bool),
            'context_conflict': torch.zeros_like(candidate_mask, dtype=torch.bool),
            'oov_type_unavailable': torch.zeros_like(candidate_mask, dtype=torch.bool),
            'correction_gain': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'absence_anomaly': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'coverage_support': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'coverage_nn_cosine': torch.full_like(scores['anomaly_score'], float('nan'), dtype=torch.float32),
            'absence_conflict': torch.zeros_like(candidate_mask, dtype=torch.bool),
            'coverage_conflict': torch.zeros_like(candidate_mask, dtype=torch.bool),
        }
        diagnostics['local_distance'][candidate_mask] = local['local_distance'].to(time_gap_norm.device).float()
        diagnostics['local_support'][candidate_mask] = local_support
        diagnostics['ensemble_score'][candidate_mask] = ensemble_score.float()
        diagnostics['ensemble_nll_std'][candidate_mask] = ensemble_nll_std.float()
        diagnostics['ensemble_supported'][candidate_mask] = local_supported
        diagnostics['ensemble_disagreement'][candidate_mask] = ensemble_disagreement
        diagnostics['component_unexpected'][candidate_mask] = component_unexpected
        diagnostics['component_reject'][candidate_mask] = component_reject
        diagnostics['cf_type_ratio'][candidate_mask] = cf_type_ratio
        diagnostics['cf_time_ratio'][candidate_mask] = cf_time_ratio
        diagnostics['cf_type_strength'][candidate_mask] = cf_type_strength
        diagnostics['cf_time_strength'][candidate_mask] = cf_time_strength
        diagnostics['cf_valid_count'][candidate_mask] = cf_valid_count
        diagnostics['context_supported'][candidate_mask] = context_supported
        diagnostics['context_conflict'][candidate_mask] = context_conflict
        diagnostics['oov_type_unavailable'][candidate_mask] = oov_type_unavailable
        diagnostics['correction_gain'][candidate_mask] = correction_gain
        diagnostics['absence_anomaly'][candidate_mask] = absence_anomaly
        diagnostics['coverage_support'][candidate_mask] = coverage_support
        diagnostics['coverage_nn_cosine'][candidate_mask] = coverage_nn_cosine
        diagnostics['absence_conflict'][candidate_mask] = absence_conflict
        diagnostics['coverage_conflict'][candidate_mask] = coverage_conflict
        stats['diagnostics'] = diagnostics

    stats.update({
        'candidates': int(num_candidates),
        'supported': int(local_supported.sum().item()),
        'corrected': int(corrected_to_expected.sum().item()),
        'provisional_expected': int(provisional_expected.sum().item()),
        'component_unexpected': int(component_unexpected.sum().item()),
        'component_reject': int(component_reject.sum().item()),
        'component_disagreement': int(ensemble_disagreement.sum().item()),
        'counterfactual_supported': (
            int(context_supported.sum().item())
            if getattr(opt, 'use_counterfactual_context_support', False) else 0
        ),
        'counterfactual_conflict': (
            int(context_conflict.sum().item())
            if getattr(opt, 'use_counterfactual_context_support', False) else 0
        ),
        'oov_type_unavailable': int(oov_type_unavailable.sum().item()),
        'absence_conflict': int(absence_conflict.sum().item()),
        'coverage_conflict': int(coverage_conflict.sum().item()),
        'support_sum': float(local_support.sum().item()),
        'ensemble_score_sum': float(ensemble_score.sum().item()),
    })
    return updated_decision, corrected_score, expected_drift_mask, stats


def eval_reliability(model, calibration_data, test_data, opt):
    model.eval()
    fit_trace_profile(calibration_data, opt)
    bank, gamma, delta = fit_calibration_bank(model, calibration_data, opt)
    opt._gamma_anomaly = gamma
    segment_gamma = fit_segment_anomaly_threshold(model, calibration_data, opt)
    drift_detector = DriftDecisionModule(
        window_size=opt.drift_window_size,
        drift_threshold=opt.drift_threshold
    )
    quarantine = DriftBuffer(max_size=opt.quarantine_max_size)
    drift_adapter = ProgressiveDriftAdapter(opt)
    raw_test_data = getattr(getattr(test_data, 'dataset', None), 'raw_data', None)
    absence_evidence = {}
    absence_summary = {}
    if getattr(opt, 'use_absence_aware_revision', False):
        absence_reference_path = str(getattr(opt, 'absence_reference_path', '') or '').strip()
        if absence_reference_path:
            raw_reference_data = load_memory_sequences(absence_reference_path)
        elif getattr(opt, 'absence_reference_split', 'train') == 'calibration':
            raw_reference_data = getattr(getattr(calibration_data, 'dataset', None), 'raw_data', None)
        else:
            raw_reference_data = _load_raw_split(opt, 'train')
        absence_evidence, absence_summary = fit_absence_evidence(
            raw_reference_data,
            raw_test_data,
            opt
        )
        absence_mechanism_name = (
            'Context-conditioned Absence Memory'
            if getattr(opt, 'absence_context_mode', 'context_memory') == 'context_memory'
            else 'Legacy Absence-aware revision'
        )
        print(
            f'[Info] {absence_mechanism_name}: '
            f"refs={absence_summary.get('reference_runs', 0)} runs/"
            f"{absence_summary.get('reference_services', 0)} services | "
            f"eval_runs={absence_summary.get('eval_runs', 0)} | "
            f"conflict_runs={absence_summary.get('conflict_runs', 0)}"
        )

    total_events = 0
    total_traditional_alerts = 0
    total_normal = 0
    total_anomaly = 0
    total_reject = 0
    total_expected_drift = 0
    total_high_a_high_u = 0
    total_uncertainty = 0.0
    total_anomaly_score = 0.0
    total_final_anomaly_score = 0.0
    total_ensemble_candidates = 0
    total_ensemble_supported = 0
    total_ensemble_corrected = 0
    total_ensemble_provisional = 0
    total_ensemble_component_unexpected = 0
    total_ensemble_component_reject = 0
    total_ensemble_component_disagreement = 0
    total_counterfactual_supported = 0
    total_counterfactual_conflict = 0
    total_oov_type_unavailable = 0
    total_absence_conflict = 0
    total_coverage_conflict = 0
    total_ensemble_support = 0.0
    total_ensemble_score = 0.0
    total_adapter_updates = 0
    total_adapter_refits = 0
    total_adapter_memory_updates = 0
    last_drift_ratio = 0.0
    labeled_events = 0
    labeled_segments = 0
    total_traditional_segment_alerts = 0
    total_ua_segment_alerts = 0
    detection_counts = {}
    detection_counts.update(init_binary_counts('Traditional'))
    detection_counts.update(init_binary_counts('UA'))
    detection_counts.update(init_binary_counts('TraditionalSegment'))
    detection_counts.update(init_binary_counts('UASegment'))
    detection_counts.update(init_binary_counts('OODAllAnomaly'))
    detection_counts.update(init_binary_counts('OODAllNormal'))
    detection_counts.update(init_binary_counts('OursOOD'))
    detection_counts.update(init_binary_counts('OODAllAnomalySegment'))
    detection_counts.update(init_binary_counts('OODAllNormalSegment'))
    detection_counts.update(init_binary_counts('OursOODSegment'))
    total_ood_candidate_events = 0
    total_ood_candidate_segments = 0
    rq4_confusion = Counter()
    rq4_segment_confusion = Counter()
    rq4_iv_confusion = Counter()
    rq4_oov_confusion = Counter()
    rq4_oov_rate_baseline_confusion = Counter()
    rq4_labeled_events = 0
    rq4_labeled_segments = 0
    rq4_risk_coverage = init_rq4_risk_coverage_store()
    rq3_event_confusions = init_rq3_confusions()
    rq3_segment_confusions = init_rq3_confusions()
    rq3_labeled_events = 0
    rq3_labeled_segments = 0
    rq4_event_rows = []
    rq4_detail_max = int(getattr(opt, 'rq4_event_detail_max', 200000))
    rq4_coverage_counts = init_rq4_coverage_counts()

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_data, desc='  - (Reliability) ', leave=False)):
            event_time, time_gap_norm, event_type, event_label, drift_label = unpack_batch(
                batch,
                opt.device,
                return_drift=True
            )
            time_gap_norm = ensure_time_gap_normalized(time_gap_norm, opt, context='reliability', event_type=event_type)
            scores = model.compute_reliability_scores(
                event_type,
                event_time,
                time_gap_norm,
                uncertainty_mc=opt.uncertainty_mc,
                type_entropy_weight=opt.type_entropy_weight,
                exact_time_nll=opt.reliability_exact_nll or opt.use_ensemble_correction,
                return_memory_features=opt.use_ensemble_correction
            )
            scores = augment_scores_with_trace_profile(scores, event_type, opt)
            scores = augment_scores_with_type_gap_profile(scores, event_type, time_gap_norm, opt)

            anomaly_score = build_anomaly_score(scores, opt)
            scores['anomaly_score'] = anomaly_score
            uncertainty_score = scores['uncertainty_score']
            mask = scores['mask']
            traditional_pred = (anomaly_score > gamma) & mask
            diagnosis_candidate = traditional_pred
            if drift_label is not None:
                rq4_mode = getattr(opt, 'rq4_candidate_mode', 'score')
                controlled_candidate = (drift_label > 0) & mask
                if rq4_mode == 'labeled':
                    diagnosis_candidate = controlled_candidate
                elif rq4_mode == 'labeled_or_score':
                    diagnosis_candidate = traditional_pred | controlled_candidate
                elif rq4_mode == 'score_or_profile':
                    profile_score = scores.get('profile_score')
                    profile_candidate_gamma = float(
                        (getattr(opt, '_component_thresholds', {}) or {}).get(
                            'profile_candidate_gamma', float('inf')
                        )
                    )
                    if profile_score is not None:
                        diagnosis_candidate = traditional_pred | (
                            (profile_score > profile_candidate_gamma) & mask
                        )

            if opt.use_conformal_rejection:
                p_u = bank.conformal_p_values(uncertainty_score, bank='uncertainty')
                reject_indicator = (p_u < opt.conformal_alpha) & mask
                decision = torch.where(
                    reject_indicator,
                    torch.full_like(anomaly_score, REJECT_DECISION, dtype=torch.long),
                    torch.where(
                        anomaly_score > gamma,
                        torch.full_like(anomaly_score, ANOMALY_DECISION, dtype=torch.long),
                        torch.full_like(anomaly_score, NORMAL_DECISION, dtype=torch.long)
                    )
                )
                decision = decision.masked_fill(~mask, -1)
            else:
                decision = model.selective_decision(
                    anomaly_score,
                    uncertainty_score,
                    gamma=gamma,
                    delta=delta,
                    mask=mask
                )
                reject_indicator = (decision == REJECT_DECISION) & mask

            pre_revision_decision = decision.clone()
            # OOD is defined by the selective detector's uncertainty rejection,
            # not by the anomaly-score threshold itself.  This keeps the
            # non-OOD predictions fixed while comparing three ways of handling
            # exactly the same OOD candidates.
            ood_candidate = (pre_revision_decision == REJECT_DECISION) & mask
            base_rq3_pred = torch.zeros_like(decision, dtype=torch.long)
            base_rq3_pred = base_rq3_pred.masked_fill(traditional_pred, 2)
            pre_revision_rq3_pred = decision_to_rq4_class(pre_revision_decision)
            absence_context = None
            if getattr(opt, 'use_absence_aware_revision', False):
                absence_context = build_absence_batch_context(
                    raw_test_data or [],
                    batch_idx,
                    opt.batch_size,
                    anomaly_score.shape,
                    opt.device,
                    absence_evidence
                )

            decision, final_anomaly_score, expected_drift_mask, ensemble_stats = apply_ensemble_correction(
                model,
                bank,
                scores,
                event_type,
                event_time,
                time_gap_norm,
                decision,
                diagnosis_candidate,
                gamma,
                delta,
                opt,
                drift_adapter=drift_adapter,
                absence_context=absence_context
            )
            reject_indicator = (decision == REJECT_DECISION) & mask

            drift_state = drift_detector.update(reject_indicator, mask=mask)
            last_drift_ratio = drift_state['reject_ratio']
            quarantine.update(
                anomaly_score,
                uncertainty_score,
                event_type=event_type[:, 1:],
                mask=reject_indicator
            )

            valid_events = mask.sum().item()
            total_events += valid_events
            ua_pred = (decision == ANOMALY_DECISION) & mask
            non_ood_anomaly = traditional_pred & (~ood_candidate)
            ood_all_anomaly_pred = non_ood_anomaly | ood_candidate
            ood_all_normal_pred = non_ood_anomaly
            # Our binary collapse uses the post-revision score/decision on OOD.
            # A memory-supported Expected Drift is accepted, a confirmed
            # anomaly is alerted, and an unresolved reject is decided by its
            # final calibrated anomaly score instead of being blindly accepted.
            resolved_ood_anomaly = (
                (decision == ANOMALY_DECISION)
                | (
                    (decision == REJECT_DECISION)
                    & (final_anomaly_score > gamma)
                )
            ) & ood_candidate
            ours_ood_pred = non_ood_anomaly | resolved_ood_anomaly
            ood_candidate_segment = ood_candidate.any(dim=1)
            total_ood_candidate_events += int(ood_candidate.sum().item())
            total_ood_candidate_segments += int(ood_candidate_segment.sum().item())
            segment_score, valid_segment = aggregate_segment_scores(
                anomaly_score,
                mask,
                opt,
                gamma=gamma
            )
            traditional_segment_pred = (segment_score > segment_gamma) & valid_segment
            # For alert-fraction aggregation, all policies must retain the same
            # denominator.  Mapping an OOD event to non-anomaly means a zero
            # alert, not deleting that event from the segment.  Deleting it
            # artificially inflates the remaining alert fraction (notably on
            # Liberty).  For max-based datasets, removing a forced-normal OOD
            # event is equivalent to assigning it a score below the threshold.
            if getattr(opt, 'segment_score_mode', 'max') == 'alert_fraction':
                valid_policy_segment = mask.any(dim=1)

                def _policy_alert_fraction(event_pred):
                    fraction = _masked_mean(event_pred.float(), mask)
                    return (fraction > segment_gamma) & valid_policy_segment

                ood_all_anomaly_segment_pred = _policy_alert_fraction(ood_all_anomaly_pred)
                ood_all_normal_segment_pred = _policy_alert_fraction(ood_all_normal_pred)
                ours_ood_segment_pred = _policy_alert_fraction(ours_ood_pred)
            else:
                non_ood_segment_score, valid_non_ood_segment = aggregate_segment_scores(
                    anomaly_score,
                    mask & (~ood_candidate),
                    opt,
                    gamma=gamma
                )
                ood_all_anomaly_segment_pred = traditional_segment_pred | ood_candidate_segment
                ood_all_normal_segment_pred = (
                    (non_ood_segment_score > segment_gamma)
                    & valid_non_ood_segment
                )
                safe_ood_score = torch.where(
                    decision == REJECT_DECISION,
                    anomaly_score,
                    final_anomaly_score
                )
                policy_final_score = torch.where(
                    ood_candidate,
                    safe_ood_score,
                    anomaly_score
                )
                policy_final_mask = mask & (~(
                    ood_candidate & (decision == EXPECTED_DRIFT_DECISION)
                ))
                ours_ood_segment_score, valid_ours_ood_segment = aggregate_segment_scores(
                    policy_final_score,
                    policy_final_mask,
                    opt,
                    gamma=gamma
                )
                ours_ood_segment_pred = (
                    (ours_ood_segment_score > segment_gamma)
                    & valid_ours_ood_segment
                )
            ua_segment_mask = mask & (decision != REJECT_DECISION) & (decision != EXPECTED_DRIFT_DECISION)
            ua_segment_score, valid_ua_segment = aggregate_segment_scores(
                final_anomaly_score,
                ua_segment_mask,
                opt,
                gamma=gamma
            )
            ua_segment_pred = (ua_segment_score > segment_gamma) & valid_ua_segment
            total_traditional_alerts += traditional_pred.sum().item()
            total_normal += (((decision == NORMAL_DECISION) | (decision == EXPECTED_DRIFT_DECISION)) & mask).sum().item()
            total_anomaly += ua_pred.sum().item()
            total_reject += ((decision == REJECT_DECISION) & mask).sum().item()
            total_expected_drift += (expected_drift_mask & mask).sum().item()
            total_traditional_segment_alerts += traditional_segment_pred.sum().item()
            total_ua_segment_alerts += ua_segment_pred.sum().item()
            total_high_a_high_u += ((anomaly_score > gamma) & reject_indicator).sum().item()
            total_uncertainty += uncertainty_score[mask].sum().item()
            total_anomaly_score += anomaly_score[mask].sum().item()
            total_final_anomaly_score += final_anomaly_score[mask].sum().item()
            total_ensemble_candidates += ensemble_stats['candidates']
            total_ensemble_supported += ensemble_stats['supported']
            total_ensemble_corrected += ensemble_stats['corrected']
            total_ensemble_provisional += ensemble_stats['provisional_expected']
            total_ensemble_component_unexpected += ensemble_stats.get('component_unexpected', 0)
            total_ensemble_component_reject += ensemble_stats.get('component_reject', 0)
            total_ensemble_component_disagreement += ensemble_stats.get('component_disagreement', 0)
            total_counterfactual_supported += ensemble_stats.get('counterfactual_supported', 0)
            total_counterfactual_conflict += ensemble_stats.get('counterfactual_conflict', 0)
            total_oov_type_unavailable += ensemble_stats.get('oov_type_unavailable', 0)
            total_absence_conflict += ensemble_stats.get('absence_conflict', 0)
            total_coverage_conflict += ensemble_stats.get('coverage_conflict', 0)
            total_ensemble_support += ensemble_stats['support_sum']
            total_ensemble_score += ensemble_stats['ensemble_score_sum']
            total_adapter_updates += ensemble_stats['adapter_updates']
            total_adapter_refits += ensemble_stats['adapter_refits']
            total_adapter_memory_updates += ensemble_stats['adapter_memory_updates']

            if drift_label is not None:
                rq4_pred = decision_to_rq4_class(decision)
                rq3_pred_by_method = {
                    'Base': base_rq3_pred,
                    'BaseReject': pre_revision_rq3_pred,
                    'Full': rq4_pred,
                }
                rq3_labeled_events += update_rq3_event_counts(
                    rq3_event_confusions,
                    rq3_pred_by_method,
                    drift_label,
                    mask
                )
                rq3_labeled_segments += update_rq3_segment_counts(
                    rq3_segment_confusions,
                    rq3_pred_by_method,
                    drift_label,
                    mask
                )
                rq4_eval_mask = mask & (drift_label > 0)
                rq4_metric_pred = smooth_rq4_window_predictions(
                    rq4_pred,
                    rq4_eval_mask,
                    mask,
                    opt
                )
                rq4_labeled_events += update_rq4_counts(
                    rq4_confusion,
                    rq4_metric_pred,
                    drift_label,
                    mask
                )
                if getattr(opt, 'treat_oov_type_as_unavailable', False):
                    oov_type_id = int(getattr(opt, 'oov_type_id', -1))
                    if oov_type_id >= 0:
                        target_type_zero_based = event_type[:, 1:] - 1
                        oov_eval_mask = mask & (target_type_zero_based == oov_type_id)
                        iv_eval_mask = mask & (~oov_eval_mask)
                        update_rq4_counts(
                            rq4_oov_confusion,
                            rq4_metric_pred,
                            drift_label,
                            oov_eval_mask
                        )
                        update_rq4_counts(
                            rq4_iv_confusion,
                            rq4_metric_pred,
                            drift_label,
                            iv_eval_mask
                        )
                        oov_baseline_pred = torch.where(
                            oov_eval_mask,
                            torch.full_like(rq4_metric_pred, 2),
                            torch.full_like(rq4_metric_pred, 1)
                        )
                        update_rq4_counts(
                            rq4_oov_rate_baseline_confusion,
                            oov_baseline_pred,
                            drift_label,
                            mask & ((drift_label == 1) | (drift_label == 2))
                        )
                update_rq4_coverage_counts(
                    rq4_coverage_counts,
                    drift_label,
                    rq4_metric_pred,
                    traditional_pred,
                    diagnosis_candidate,
                    mask
                )
                update_rq4_risk_coverage(
                    rq4_risk_coverage,
                    drift_label,
                    rq4_metric_pred,
                    -uncertainty_score,
                    mask
                )
                rq4_labeled_segments += update_rq4_segment_counts(
                    rq4_segment_confusion,
                    rq4_metric_pred,
                    drift_label,
                    mask,
                    pred_valid_mask=rq4_eval_mask
                )
                if (
                    getattr(opt, 'save_rq4_event_details', False)
                    and opt.save_result
                    and len(rq4_event_rows) < rq4_detail_max
                ):
                    detail_mask = mask & (drift_label > 0)
                    if getattr(opt, 'rq4_event_detail_include_normal', False):
                        detail_mask = mask & (drift_label >= 0)
                    detail_indices = torch.nonzero(detail_mask, as_tuple=False)
                    diagnostics = ensemble_stats.get('diagnostics') or {}
                    local_distance = diagnostics.get('local_distance')
                    local_support = diagnostics.get('local_support')
                    ensemble_score = diagnostics.get('ensemble_score')
                    ensemble_nll_std = diagnostics.get('ensemble_nll_std')
                    ensemble_supported = diagnostics.get('ensemble_supported')
                    ensemble_disagreement = diagnostics.get('ensemble_disagreement')
                    component_unexpected = diagnostics.get('component_unexpected')
                    component_reject = diagnostics.get('component_reject')
                    cf_type_ratio = diagnostics.get('cf_type_ratio')
                    cf_time_ratio = diagnostics.get('cf_time_ratio')
                    cf_type_strength = diagnostics.get('cf_type_strength')
                    cf_time_strength = diagnostics.get('cf_time_strength')
                    cf_valid_count = diagnostics.get('cf_valid_count')
                    context_supported = diagnostics.get('context_supported')
                    context_conflict = diagnostics.get('context_conflict')
                    oov_type_unavailable = diagnostics.get('oov_type_unavailable')
                    correction_gain = diagnostics.get('correction_gain')
                    absence_anomaly = diagnostics.get('absence_anomaly')
                    coverage_support = diagnostics.get('coverage_support')
                    coverage_nn_cosine = diagnostics.get('coverage_nn_cosine')
                    absence_conflict = diagnostics.get('absence_conflict')
                    coverage_conflict = diagnostics.get('coverage_conflict')

                    def _maybe_float(tensor, row, col):
                        if tensor is None:
                            return ''
                        value = float(tensor[row, col].detach().cpu().item())
                        if not math.isfinite(value):
                            return ''
                        return value

                    def _maybe_bool(tensor, row, col):
                        if tensor is None:
                            return ''
                        return int(bool(tensor[row, col].detach().cpu().item()))

                    for row_col in detail_indices:
                        if len(rq4_event_rows) >= rq4_detail_max:
                            break
                        row = int(row_col[0].item())
                        col = int(row_col[1].item())
                        true_id = int(drift_label[row, col].detach().cpu().item())
                        raw_pred_id = int(rq4_pred[row, col].detach().cpu().item())
                        pred_id = int(rq4_metric_pred[row, col].detach().cpu().item())
                        rq4_event_rows.append({
                            'batch_index': batch_idx,
                            'sequence_index': row,
                            'event_index': col + 1,
                            'event_type': int(event_type[row, col + 1].detach().cpu().item()) - 1,
                            'true_drift_id': true_id,
                            'true_drift_label': RQ4_CLASS_NAMES.get(true_id, 'Other'),
                            'raw_pred_drift_id': raw_pred_id,
                            'raw_pred_drift_label': RQ4_CLASS_NAMES.get(raw_pred_id, 'Normal'),
                            'pred_drift_id': pred_id,
                            'pred_drift_label': RQ4_CLASS_NAMES.get(pred_id, 'Normal'),
                            'decision_id': int(decision[row, col].detach().cpu().item()),
                            'traditional_candidate': int(bool(traditional_pred[row, col].detach().cpu().item())),
                            'diagnosis_candidate': int(bool(diagnosis_candidate[row, col].detach().cpu().item())),
                            'type_nll': float(scores['type_nll'][row, col].detach().cpu().item()),
                            'time_nll': float(scores['time_nll'][row, col].detach().cpu().item()),
                            'profile_score': (
                                float(scores['profile_score'][row, col].detach().cpu().item())
                                if 'profile_score' in scores else ''
                            ),
                            'anomaly_score': float(anomaly_score[row, col].detach().cpu().item()),
                            'final_anomaly_score': float(final_anomaly_score[row, col].detach().cpu().item()),
                            'uncertainty_score': float(uncertainty_score[row, col].detach().cpu().item()),
                            'local_distance': _maybe_float(local_distance, row, col),
                            'local_support': _maybe_float(local_support, row, col),
                            'ensemble_score': _maybe_float(ensemble_score, row, col),
                            'ensemble_nll_std': _maybe_float(ensemble_nll_std, row, col),
                            'ensemble_supported': _maybe_bool(ensemble_supported, row, col),
                            'ensemble_disagreement': _maybe_bool(ensemble_disagreement, row, col),
                            'component_unexpected': _maybe_bool(component_unexpected, row, col),
                            'component_reject': _maybe_bool(component_reject, row, col),
                            'cf_type_ratio': _maybe_float(cf_type_ratio, row, col),
                            'cf_time_ratio': _maybe_float(cf_time_ratio, row, col),
                            'cf_type_strength': _maybe_float(cf_type_strength, row, col),
                            'cf_time_strength': _maybe_float(cf_time_strength, row, col),
                            'cf_valid_count': _maybe_float(cf_valid_count, row, col),
                            'context_supported': _maybe_bool(context_supported, row, col),
                            'context_conflict': _maybe_bool(context_conflict, row, col),
                            'oov_type_unavailable': _maybe_bool(oov_type_unavailable, row, col),
                            'correction_gain': _maybe_float(correction_gain, row, col),
                            'absence_anomaly': _maybe_float(absence_anomaly, row, col),
                            'coverage_support': _maybe_float(coverage_support, row, col),
                            'coverage_nn_cosine': _maybe_float(coverage_nn_cosine, row, col),
                            'absence_conflict': _maybe_bool(absence_conflict, row, col),
                            'coverage_conflict': _maybe_bool(coverage_conflict, row, col),
                            'gamma_anomaly': float(gamma),
                            'delta_uncertainty': float(delta),
                        })

            if event_label is not None:
                labeled_events += update_binary_counts(
                    detection_counts,
                    'Traditional',
                    traditional_pred,
                    event_label,
                    mask
                )
                label_mask = mask & (event_label >= 0)
                segment_labeled = label_mask.any(dim=1)
                segment_truth = ((event_label == 1) & mask).any(dim=1).long()
                segment_truth = segment_truth.masked_fill(~segment_labeled, -1)
                labeled_segments += update_binary_counts(
                    detection_counts,
                    'TraditionalSegment',
                    traditional_segment_pred,
                    segment_truth,
                    segment_labeled
                )
                update_binary_counts(
                    detection_counts,
                    'UASegment',
                    ua_segment_pred,
                    segment_truth,
                    segment_labeled
                )
                update_binary_counts(
                    detection_counts,
                    'UA',
                    ua_pred,
                    event_label,
                    mask
                )
                update_binary_counts(
                    detection_counts,
                    'OODAllAnomaly',
                    ood_all_anomaly_pred,
                    event_label,
                    mask
                )
                update_binary_counts(
                    detection_counts,
                    'OODAllNormal',
                    ood_all_normal_pred,
                    event_label,
                    mask
                )
                update_binary_counts(
                    detection_counts,
                    'OursOOD',
                    ours_ood_pred,
                    event_label,
                    mask
                )
                update_binary_counts(
                    detection_counts,
                    'OODAllAnomalySegment',
                    ood_all_anomaly_segment_pred,
                    segment_truth,
                    segment_labeled
                )
                update_binary_counts(
                    detection_counts,
                    'OODAllNormalSegment',
                    ood_all_normal_segment_pred,
                    segment_truth,
                    segment_labeled
                )
                update_binary_counts(
                    detection_counts,
                    'OursOODSegment',
                    ours_ood_segment_pred,
                    segment_truth,
                    segment_labeled
                )

    if total_events == 0:
        return {}

    point_alert_reduction = 0.0
    if total_traditional_alerts > 0:
        point_alert_reduction = 1.0 - (total_anomaly / total_traditional_alerts)
    ensemble_support_rate = total_ensemble_supported / total_ensemble_candidates if total_ensemble_candidates > 0 else 0.0
    ensemble_correction_rate = total_ensemble_corrected / total_ensemble_candidates if total_ensemble_candidates > 0 else 0.0
    mean_ensemble_support = total_ensemble_support / total_ensemble_candidates if total_ensemble_candidates > 0 else 0.0
    mean_ensemble_score = total_ensemble_score / total_ensemble_candidates if total_ensemble_candidates > 0 else 0.0

    results = {
        'Calib_Size': len(bank),
        'Gamma_Anomaly': gamma,
        'Gamma_Segment_Anomaly': segment_gamma,
        'Delta_Uncertainty': delta,
        'Anomaly_Score_Mode': getattr(opt, 'anomaly_score_mode', 'raw'),
        'RQ4_Candidate_Mode': getattr(opt, 'rq4_candidate_mode', 'score'),
        'Component_Drift_Diagnosis': int(getattr(opt, 'use_component_drift_diagnosis', False)),
        'RQ4_Window_Diagnosis': int(getattr(opt, 'use_rq4_window_diagnosis', False)),
        'Counterfactual_Context_Support': int(getattr(opt, 'use_counterfactual_context_support', False)),
        'Counterfactual_Support_K': int(getattr(opt, 'counterfactual_support_k', 3)),
        'Counterfactual_Type_Support_Ratio': float(getattr(opt, 'counterfactual_type_support_ratio', 0.34)),
        'Counterfactual_Time_Support_Ratio': float(getattr(opt, 'counterfactual_time_support_ratio', 0.34)),
        'Absence_Aware_Revision': int(getattr(opt, 'use_absence_aware_revision', False)),
        'Absence_Context_Mode': getattr(opt, 'absence_context_mode', 'context_memory'),
        'Absence_Mechanism': (
            'Context-conditioned Absence Memory'
            if getattr(opt, 'absence_context_mode', 'context_memory') == 'context_memory'
            else 'Legacy absence-aware revision'
        ),
        'Absence_Reference_Path': str(getattr(opt, 'absence_reference_path', '') or ''),
        'Absence_Reference_Runs': int(absence_summary.get('reference_runs', 0)) if absence_summary else 0,
        'Absence_Reference_Services': int(absence_summary.get('reference_services', 0)) if absence_summary else 0,
        'Absence_Conflict_Runs': int(absence_summary.get('conflict_runs', 0)) if absence_summary else 0,
        'Absence_Anomaly_Threshold': float(getattr(opt, 'absence_anomaly_threshold', 2.0)),
        'Absence_Persistence_Threshold': float(getattr(opt, 'absence_persistence_threshold', 0.5)),
        'Absence_Min_Context_Similarity': float(getattr(opt, 'absence_min_context_similarity', 0.2)),
        'Absence_Min_Query_Exposure': float(getattr(opt, 'absence_min_query_exposure', 50.0)),
        'Absence_Count_Ratio_Threshold': float(getattr(opt, 'absence_count_ratio_threshold', 0.5)),
        'Absence_Coverage_Threshold': float(getattr(opt, 'absence_coverage_threshold', 0.5)),
        'Treat_OOV_Type_As_Unavailable': int(getattr(opt, 'treat_oov_type_as_unavailable', False)),
        'OOV_Type_ID': int(getattr(opt, 'oov_type_id', -1)),
        'RQ4_Window_Expected_Min_Frac': float(getattr(opt, 'rq4_window_expected_min_frac', 0.50)),
        'RQ4_Window_Unexpected_Min_Frac': float(getattr(opt, 'rq4_window_unexpected_min_frac', 0.25)),
        'RQ4_Window_Reject_Min_Frac': float(getattr(opt, 'rq4_window_reject_min_frac', 0.20)),
        'RQ4_Window_Conflict_Min_Frac': float(getattr(opt, 'rq4_window_conflict_min_frac', 0.20)),
        'Anomaly_Quantile': float(getattr(opt, 'anomaly_quantile', 0.99)),
        'Type_Score_Weight': float(getattr(opt, 'type_score_weight', 1.0)),
        'Time_Score_Weight': float(getattr(opt, 'time_score_weight', 1.0)),
        'Conditional_Gap_Weight': float(getattr(opt, 'conditional_gap_weight', 1.0)),
        'Profile_Score_Weight': float(getattr(opt, 'profile_score_weight', 0.0)),
        'Profile_Unigram_Weight': float(getattr(opt, 'profile_unigram_weight', 0.0)),
        'Profile_Bigram_Weight': float(getattr(opt, 'profile_bigram_weight', 0.0)),
        'Profile_Hist_Signature_Weight': float(getattr(opt, 'profile_hist_signature_weight', 0.0)),
        'Trace_Profile_Enabled': int(getattr(opt, '_trace_profile', None) is not None),
        'Detected_Dataset': getattr(opt, 'detected_dataset', 'generic'),
        'Segment_Score_Mode': getattr(opt, 'segment_score_mode', 'max'),
        'Segment_TopK': int(getattr(opt, 'segment_topk', 3)),
        'Events': total_events,
        'Traditional_Point_Alerts': total_traditional_alerts,
        'Confident_Anomaly_Alerts': total_anomaly,
        'Traditional_Segment_Alerts': total_traditional_segment_alerts,
        'UA_Segment_Alerts': total_ua_segment_alerts,
        'OOD_Candidate_Events': total_ood_candidate_events,
        'OOD_Candidate_Segments': total_ood_candidate_segments,
        'Rejected_Uncertain_Events': total_reject,
        'HighA_HighU_Events': total_high_a_high_u,
        'Expected_Drift_Events': total_expected_drift,
        'Normal_Events': total_normal,
        'Reject_Rate': total_reject / total_events,
        'Confident_Anomaly_Rate': total_anomaly / total_events,
        'Expected_Drift_Rate': total_expected_drift / total_events,
        'Point_Alert_Reduction': point_alert_reduction,
        'Ensemble_Candidates': total_ensemble_candidates,
        'Ensemble_Supported': total_ensemble_supported,
        'Ensemble_Corrected': total_ensemble_corrected,
        'Ensemble_Provisional_Expected': total_ensemble_provisional,
        'Ensemble_Component_Unexpected': total_ensemble_component_unexpected,
        'Ensemble_Component_Reject': total_ensemble_component_reject,
        'Ensemble_Component_Disagreement': total_ensemble_component_disagreement,
        'Counterfactual_Context_Supported': total_counterfactual_supported,
        'Counterfactual_Context_Conflict': total_counterfactual_conflict,
        'OOV_Type_Unavailable_Events': total_oov_type_unavailable,
        'Absence_Conflict_Events': total_absence_conflict,
        'Coverage_Conflict_Events': total_coverage_conflict,
        'Ensemble_Disagreement_Threshold': float(getattr(opt, 'ensemble_disagreement_threshold', 2.0)),
        'Ensemble_Support_Rate': ensemble_support_rate,
        'Ensemble_Correction_Rate': ensemble_correction_rate,
        'Mean_Ensemble_Local_Support': mean_ensemble_support,
        'Mean_Ensemble_Score': mean_ensemble_score,
        'Drift_Adapter_Ready': int(drift_adapter.ready),
        'Drift_Adapter_Updates': total_adapter_updates,
        'Drift_Adapter_Refits': total_adapter_refits,
        'Drift_Adapter_Memory_Updates': total_adapter_memory_updates,
        'Drift_Adapter_Slope': drift_adapter.slope,
        'Drift_Adapter_Shift': drift_adapter.shift,
        'Drift_Adapter_Support_Scale': drift_adapter.support_scale,
        'Drift_Warnings': drift_detector.num_warnings,
        'Final_Drift_Ratio': last_drift_ratio,
        'Quarantine_Size': len(quarantine),
        'Mean_Anomaly_Score': total_anomaly_score / total_events,
        'Mean_Final_Anomaly_Score': total_final_anomaly_score / total_events,
        'Mean_Uncertainty_Score': total_uncertainty / total_events,
    }
    component_thresholds = getattr(opt, '_component_thresholds', {}) or {}
    for key, value in component_thresholds.items():
        results[f'Component_{key}'] = value

    if labeled_events > 0:
        results['Labeled_Events'] = labeled_events
        results.update(finalize_binary_metrics(detection_counts, 'Traditional'))
        results.update(finalize_binary_metrics(detection_counts, 'UA'))
        results['Labeled_Segments'] = labeled_segments
        results.update(finalize_binary_metrics(detection_counts, 'TraditionalSegment'))
        results.update(finalize_binary_metrics(detection_counts, 'UASegment'))
        results.update(finalize_binary_metrics(detection_counts, 'OODAllAnomaly'))
        results.update(finalize_binary_metrics(detection_counts, 'OODAllNormal'))
        results.update(finalize_binary_metrics(detection_counts, 'OursOOD'))
        results.update(finalize_binary_metrics(detection_counts, 'OODAllAnomalySegment'))
        results.update(finalize_binary_metrics(detection_counts, 'OODAllNormalSegment'))
        results.update(finalize_binary_metrics(detection_counts, 'OursOODSegment'))
    else:
        results['Labeled_Events'] = 0
        results['Labeled_Segments'] = 0

    if rq4_labeled_events > 0:
        results.update(finalize_rq4_metrics(rq4_confusion))
        results.update(finalize_rq4_coverage_metrics(rq4_coverage_counts))
        results.update(finalize_rq4_risk_coverage(rq4_risk_coverage))
        if sum(rq4_iv_confusion.values()) > 0:
            results.update(finalize_rq4_metrics(
                rq4_iv_confusion,
                prefix='RQ4_IV',
                total_key='Labeled_Events'
            ))
        if sum(rq4_oov_confusion.values()) > 0:
            results.update(finalize_rq4_metrics(
                rq4_oov_confusion,
                prefix='RQ4_OOV',
                total_key='Labeled_Events'
            ))
        if sum(rq4_oov_rate_baseline_confusion.values()) > 0:
            results.update(finalize_rq4_metrics(
                rq4_oov_rate_baseline_confusion,
                prefix='RQ4_OOVRateBaseline',
                total_key='Labeled_Events'
            ))
        if rq4_labeled_segments > 0:
            results.update(finalize_rq4_metrics(
                rq4_segment_confusion,
                prefix='RQ4_Segment',
                total_key='Labeled_Segments'
            ))
        if getattr(opt, 'save_rq4_event_details', False) and opt.save_result and rq4_event_rows:
            detail_path = f"{opt.save_result}_rq4_events.csv"
            with open(detail_path, mode='w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=list(rq4_event_rows[0].keys()))
                writer.writeheader()
                writer.writerows(rq4_event_rows)
            print(f'    RQ4 event details:         {detail_path}')
    else:
        results['RQ4_Labeled_Events'] = 0
        results['RQ4_Segment_Labeled_Segments'] = 0

    if rq3_labeled_events > 0:
        results.update(finalize_rq3_metrics(
            rq3_event_confusions,
            prefix='RQ3',
            level='Event'
        ))
        if rq3_labeled_segments > 0:
            results.update(finalize_rq3_metrics(
                rq3_segment_confusions,
                prefix='RQ3',
                level='Segment'
            ))
    else:
        for method in RQ3_METHOD_NAMES:
            results[f'RQ3_{method}_Event_Labeled_Total'] = 0
            results[f'RQ3_{method}_Segment_Labeled_Total'] = 0

    print(f'\n  - (Reliability Summary)')
    print(f'    Calibration size:          {results["Calib_Size"]}')
    print(
        f'    Score mode / segment mode: '
        f'{results["Anomaly_Score_Mode"]} / {results["Segment_Score_Mode"]}'
    )
    if getattr(opt, 'rq4_candidate_mode', 'score') != 'score':
        print(f'    RQ4 candidate mode:        {results["RQ4_Candidate_Mode"]}')
    if getattr(opt, 'use_rq4_window_diagnosis', False):
        print(
            f'    RQ4 window thresholds:     '
            f'E>={results["RQ4_Window_Expected_Min_Frac"]:.2f}, '
            f'U>={results["RQ4_Window_Unexpected_Min_Frac"]:.2f}, '
            f'R>={results["RQ4_Window_Reject_Min_Frac"]:.2f}, '
            f'conflict>={results["RQ4_Window_Conflict_Min_Frac"]:.2f}'
        )
    if getattr(opt, 'use_counterfactual_context_support', False):
        print(
            f'    Counterfactual support:    '
            f'K={results["Counterfactual_Support_K"]}, '
            f'R_type>={results["Counterfactual_Type_Support_Ratio"]:.2f}, '
            f'R_time>={results["Counterfactual_Time_Support_Ratio"]:.2f}'
        )
    print(f'    Gamma anomaly threshold:   {results["Gamma_Anomaly"]:.6f}')
    print(f'    Gamma segment threshold:   {results["Gamma_Segment_Anomaly"]:.6f}')
    print(f'    Delta uncertainty thresh.: {results["Delta_Uncertainty"]:.6f}')
    print(f'    Traditional point alerts:  {results["Traditional_Point_Alerts"]}')
    print(f'    Confident anomalies:       {results["Confident_Anomaly_Alerts"]}')
    print(f'    Expected drift corrected:  {results["Expected_Drift_Events"]}')
    print(f'    Rejected uncertain events: {results["Rejected_Uncertain_Events"]}')
    print(f'    Point alert reduction:     {results["Point_Alert_Reduction"]:.4f}')
    if opt.use_ensemble_correction:
        print(f'    Ensemble candidates:       {results["Ensemble_Candidates"]}')
        print(f'    Provisional expected:      {results["Ensemble_Provisional_Expected"]}')
        if getattr(opt, 'use_component_drift_diagnosis', False):
            print(f'    Component unexpected:      {results["Ensemble_Component_Unexpected"]}')
            print(f'    Component reject:          {results["Ensemble_Component_Reject"]}')
            print(f'    Component disagreement:    {results["Ensemble_Component_Disagreement"]}')
        if getattr(opt, 'use_counterfactual_context_support', False):
            print(f'    Context supported:         {results["Counterfactual_Context_Supported"]}')
            print(f'    Context conflict:          {results["Counterfactual_Context_Conflict"]}')
        if getattr(opt, 'treat_oov_type_as_unavailable', False):
            print(f'    OOV type unavailable:      {results["OOV_Type_Unavailable_Events"]}')
        print(f'    Ensemble support rate:     {results["Ensemble_Support_Rate"]:.4f}')
        print(f'    Ensemble correction rate:  {results["Ensemble_Correction_Rate"]:.4f}')
        if opt.use_drift_adapter:
            print(f'    Drift adapter ready:       {results["Drift_Adapter_Ready"]}')
            print(f'    Drift adapter refits:      {results["Drift_Adapter_Refits"]}')
            print(
                f'    Drift adapter affine:      '
                f'z -> {results["Drift_Adapter_Slope"]:.4f} z + '
                f'{results["Drift_Adapter_Shift"]:.4f}'
            )
    print(f'    Drift warnings:            {results["Drift_Warnings"]}')
    print(f'    Final drift ratio:         {results["Final_Drift_Ratio"]:.4f}')
    if labeled_events > 0:
        print(f'    Labeled events:            {results["Labeled_Events"]}')
        print(
            f'    Traditional P/R/F1:        '
            f'{results["Traditional_Precision"]:.4f} / '
            f'{results["Traditional_Recall"]:.4f} / '
            f'{results["Traditional_F1"]:.4f}'
        )
        print(
            f'    UA P/R/F1:                 '
            f'{results["UA_Precision"]:.4f} / '
            f'{results["UA_Recall"]:.4f} / '
            f'{results["UA_F1"]:.4f}'
        )
        print(f'    Labeled segments:          {results["Labeled_Segments"]}')
        print(
            f'    Traditional segment P/R/F1: '
            f'{results["TraditionalSegment_Precision"]:.4f} / '
            f'{results["TraditionalSegment_Recall"]:.4f} / '
            f'{results["TraditionalSegment_F1"]:.4f}'
        )
        print(
            f'    UA segment P/R/F1:          '
            f'{results["UASegment_Precision"]:.4f} / '
            f'{results["UASegment_Recall"]:.4f} / '
            f'{results["UASegment_F1"]:.4f}'
        )
    else:
        print('    Labeled events:            0 (Precision/Recall/F1 skipped)')

    if results.get('RQ4_Labeled_Events', 0) > 0:
        print(f'    RQ4 labeled drift events:  {results["RQ4_Labeled_Events"]}')
        print(
            f'    RQ4 Expected P/R/F1:       '
            f'{results["RQ4_Expected_Precision"]:.4f} / '
            f'{results["RQ4_Expected_Recall"]:.4f} / '
            f'{results["RQ4_Expected_F1"]:.4f}'
        )
        print(
            f'    RQ4 Unexpected P/R/F1:     '
            f'{results["RQ4_Unexpected_Precision"]:.4f} / '
            f'{results["RQ4_Unexpected_Recall"]:.4f} / '
            f'{results["RQ4_Unexpected_F1"]:.4f}'
        )
        print(
            f'    RQ4 Reject P/R/F1:         '
            f'{results["RQ4_Reject_Precision"]:.4f} / '
            f'{results["RQ4_Reject_Recall"]:.4f} / '
            f'{results["RQ4_Reject_F1"]:.4f}'
        )
        print(f'    RQ4 Macro-F1:              {results["RQ4_Macro_F1"]:.4f}')
        print(f'    RQ4 E/U Avg-F1:            {results["RQ4_EU_Avg_F1"]:.4f}')
        print(
            f'    RQ4 coverage/risk/AURC:    '
            f'{results["RQ4_Selective_Coverage_EU"]:.4f} / '
            f'{results["RQ4_Selective_Risk_EU"]:.4f} / '
            f'{results["RQ4_Risk_Coverage_AURC"]:.4f}'
        )
        print(
            f'    RQ4 unexpected false accept:{results["RQ4_Unexpected_False_Acceptance_Rate"]:.4f}'
        )
        if results.get('RQ4_OOV_Labeled_Events', 0) > 0:
            print(
                f'    RQ4 IV/OOV Macro-F1:       '
                f'{results.get("RQ4_IV_Macro_F1", 0.0):.4f} / '
                f'{results.get("RQ4_OOV_Macro_F1", 0.0):.4f}'
            )
            print(
                f'    OOV-rate baseline Macro-F1:{results.get("RQ4_OOVRateBaseline_Macro_F1", 0.0):.4f}'
            )
        print(f'    Memory contamination:      {results["RQ4_Memory_Contamination"]:.4f}')
        print(
            f'    RQ4 triggered rates E/U/R: '
            f'{results["RQ4_Expected_Triggered_Rate"]:.4f} / '
            f'{results["RQ4_Unexpected_Triggered_Rate"]:.4f} / '
            f'{results["RQ4_Reject_Triggered_Rate"]:.4f}'
        )
        print(
            f'    RQ4 candidate rates E/U/R: '
            f'{results["RQ4_Expected_Traditional_Candidate_Rate"]:.4f} / '
            f'{results["RQ4_Unexpected_Traditional_Candidate_Rate"]:.4f} / '
            f'{results["RQ4_Reject_Traditional_Candidate_Rate"]:.4f}'
        )
        print(
            f'    RQ4 diagnosis candidate rates E/U/R: '
            f'{results["RQ4_Expected_Diagnosis_Candidate_Rate"]:.4f} / '
            f'{results["RQ4_Unexpected_Diagnosis_Candidate_Rate"]:.4f} / '
            f'{results["RQ4_Reject_Diagnosis_Candidate_Rate"]:.4f}'
        )
        print_rq4_transition_rates(results, prefix='RQ4', title='RQ4 event transitions')
        if results.get('RQ4_Segment_Labeled_Segments', 0) > 0:
            print(f'    RQ4 labeled drift segments:{results["RQ4_Segment_Labeled_Segments"]}')
            print(
                f'    RQ4 segment Expected P/R/F1: '
                f'{results["RQ4_Segment_Expected_Precision"]:.4f} / '
                f'{results["RQ4_Segment_Expected_Recall"]:.4f} / '
                f'{results["RQ4_Segment_Expected_F1"]:.4f}'
            )
            print(
                f'    RQ4 segment Unexpected P/R/F1: '
                f'{results["RQ4_Segment_Unexpected_Precision"]:.4f} / '
                f'{results["RQ4_Segment_Unexpected_Recall"]:.4f} / '
                f'{results["RQ4_Segment_Unexpected_F1"]:.4f}'
            )
            print(
                f'    RQ4 segment Reject P/R/F1: '
                f'{results["RQ4_Segment_Reject_Precision"]:.4f} / '
                f'{results["RQ4_Segment_Reject_Recall"]:.4f} / '
                f'{results["RQ4_Segment_Reject_F1"]:.4f}'
            )
            print(f'    RQ4 segment Macro-F1:      {results["RQ4_Segment_Macro_F1"]:.4f}')
            print(f'    RQ4 segment E/U Avg-F1:    {results["RQ4_Segment_EU_Avg_F1"]:.4f}')
            print_rq4_transition_rates(
                results,
                prefix='RQ4_Segment',
                title='RQ4 segment transitions'
            )

    if results.get('RQ3_Full_Segment_Labeled_Total', 0) > 0:
        print(f'    RQ3 labeled drift windows: {results["RQ3_Full_Segment_Labeled_Total"]}')
        print(
            f'    RQ3 ED-FAR Base/B+R/Full:  '
            f'{results["RQ3_Base_Segment_ED_FAR"]:.4f} / '
            f'{results["RQ3_BaseReject_Segment_ED_FAR"]:.4f} / '
            f'{results["RQ3_Full_Segment_ED_FAR"]:.4f}'
        )
        print(
            f'    RQ3 ED reduction B+R/Full: '
            f'{results["RQ3_BaseReject_Segment_ED_Reduction"]:.4f} / '
            f'{results["RQ3_Full_Segment_ED_Reduction"]:.4f}'
        )
        print(
            f'    RQ3 UD recall/safe Full:   '
            f'{results["RQ3_Full_Segment_UD_Recall"]:.4f} / '
            f'{results["RQ3_Full_Segment_UD_SafeRate"]:.4f}'
        )
        print(
            f'    RQ3 contamination Full:    '
            f'{results["RQ3_Full_Segment_Contamination"]:.4f}'
        )

    return results


def benchmark_anomaly_inference(model, test_data, opt, gamma, segment_gamma):
    """Benchmark prediction and anomaly-decision work without metric I/O."""
    warmup_batches = max(0, int(getattr(opt, 'benchmark_warmup_batches', 5)))
    repeats = max(1, int(getattr(opt, 'benchmark_repeats', 5)))

    def run_batch(batch):
        event_time, time_gap_norm, event_type, _ = unpack_batch(batch, opt.device)
        time_gap_norm = ensure_time_gap_normalized(
            time_gap_norm,
            opt,
            context='anomaly_benchmark',
            event_type=event_type
        )
        scores = model.compute_reliability_scores(
            event_type,
            event_time,
            time_gap_norm,
            uncertainty_mc=opt.uncertainty_mc,
            type_entropy_weight=opt.type_entropy_weight,
            exact_time_nll=opt.reliability_exact_nll
        )
        scores = augment_scores_with_trace_profile(scores, event_type, opt)
        scores = augment_scores_with_type_gap_profile(scores, event_type, time_gap_norm, opt)
        anomaly_score = build_anomaly_score(scores, opt)
        mask = scores['mask']
        pred_anomaly = (anomaly_score > gamma) & mask
        segment_score, valid_segment = aggregate_segment_scores(
            anomaly_score,
            mask,
            opt,
            gamma=gamma
        )
        pred_segment_anomaly = (segment_score > segment_gamma) & valid_segment
        # Keep the decision tensors live until all scheduled kernels are complete.
        return (
            pred_anomaly,
            pred_segment_anomaly,
            int(mask.sum().item()),
            int(event_type.size(0)),
        )

    model.eval()
    with torch.inference_mode():
        if warmup_batches > 0:
            for batch_index, batch in enumerate(test_data):
                run_batch(batch)
                if batch_index + 1 >= warmup_batches:
                    break
            synchronize_device(opt.device)

        if opt.device.type == 'cuda' and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(opt.device)

        repeat_times = []
        repeat_events = []
        repeat_sequences = []
        for _ in range(repeats):
            synchronize_device(opt.device)
            start = time.perf_counter()
            event_count = 0
            sequence_count = 0
            for batch in test_data:
                _, _, batch_events, batch_sequences = run_batch(batch)
                event_count += batch_events
                sequence_count += batch_sequences
            synchronize_device(opt.device)
            repeat_times.append(time.perf_counter() - start)
            repeat_events.append(event_count)
            repeat_sequences.append(sequence_count)

    median_time = float(np.median(repeat_times))
    mean_time = float(np.mean(repeat_times))
    std_time = float(np.std(repeat_times))
    event_count = int(round(float(np.median(repeat_events))))
    sequence_count = int(round(float(np.median(repeat_sequences))))
    event_throughput = event_count / median_time if median_time > 0 else 0.0
    event_latency_per_1k = 1_000_000.0 * median_time / event_count if event_count > 0 else 0.0
    sequence_throughput = sequence_count / median_time if median_time > 0 else 0.0
    sequence_latency_per_1k = (
        1_000_000.0 * median_time / sequence_count if sequence_count > 0 else 0.0
    )
    peak_memory_mb = 0.0
    if opt.device.type == 'cuda' and torch.cuda.is_available():
        peak_memory_mb = torch.cuda.max_memory_allocated(opt.device) / (1024.0 ** 2)

    return {
        'Inference_Time_Median_s': median_time,
        'Inference_Time_Mean_s': mean_time,
        'Inference_Time_Std_s': std_time,
        'Inference_Repeats': repeats,
        'Inference_Warmup_Batches': warmup_batches,
        'Inference_Events': event_count,
        'Inference_Sequences': sequence_count,
        'Throughput_Events_s': event_throughput,
        'Latency_ms_per_1k_Events': event_latency_per_1k,
        'Throughput_Sequences_s': sequence_throughput,
        'Latency_ms_per_1k_Sequences': sequence_latency_per_1k,
        'Peak_GPU_Memory_MB': peak_memory_mb,
        'Inference_Timing_Scope': 'forward+score+decision; excludes calibration, metric aggregation, and CSV I/O',
    }


def eval_anomaly_detection(model, calibration_data, test_data, opt):
    model.eval()
    synchronize_device(opt.device)
    calibration_start = time.perf_counter()
    fit_trace_profile(calibration_data, opt)
    bank, gamma, delta = fit_calibration_bank(model, calibration_data, opt)
    opt._gamma_anomaly = gamma
    segment_gamma = fit_segment_anomaly_threshold(model, calibration_data, opt)
    synchronize_device(opt.device)
    calibration_time = time.perf_counter() - calibration_start

    total_events = 0
    labeled_events = 0
    labeled_segments = 0
    total_alerts = 0
    total_segment_alerts = 0
    total_anomaly_score = 0.0
    total_uncertainty = 0.0
    detection_counts = init_binary_counts('Event')
    detection_counts.update(init_binary_counts('Segment'))
    rq2_counts = init_rq2_counts()
    rq2_labeled_events = 0
    rq2_labeled_segments = 0
    event_score_parts = []
    event_truth_parts = []
    segment_score_parts = []
    segment_truth_parts = []

    unified_file = None
    unified_writer = None
    unified_path = ''
    if getattr(opt, 'save_unified_predictions', False):
        unified_path = getattr(opt, 'unified_prediction_path', '')
        if not unified_path and getattr(opt, 'save_result', None):
            unified_path = f'{opt.save_result}_unified_predictions.csv'
        if not unified_path:
            raise ValueError(
                '-save_unified_predictions requires -unified_prediction_path or -save_result.'
            )
        unified_parent = os.path.dirname(unified_path)
        if unified_parent:
            os.makedirs(unified_parent, exist_ok=True)
        unified_file = open(unified_path, mode='w', encoding='utf-8', newline='')
        unified_fields = [
            'method', 'dataset', 'split', 'level', 'sequence_id', 'event_id',
            'segment_id', 'label', 'pred', 'score', 'count'
        ]
        unified_writer = csv.DictWriter(unified_file, fieldnames=unified_fields)
        unified_writer.writeheader()

    sequence_offset = 0
    synchronize_device(opt.device)
    evaluation_start = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(test_data, desc='  - (Anomaly Eval) ', leave=False):
            if getattr(opt, 'eval_rq2_subsets', False):
                event_time, time_gap_norm, event_type, event_label, rq2_label = unpack_batch(
                    batch,
                    opt.device,
                    return_rq2=True
                )
            else:
                event_time, time_gap_norm, event_type, event_label = unpack_batch(batch, opt.device)
                rq2_label = None
            time_gap_norm = ensure_time_gap_normalized(time_gap_norm, opt, context='anomaly_detection', event_type=event_type)
            scores = model.compute_reliability_scores(
                event_type,
                event_time,
                time_gap_norm,
                uncertainty_mc=opt.uncertainty_mc,
                type_entropy_weight=opt.type_entropy_weight,
                exact_time_nll=opt.reliability_exact_nll
            )
            scores = augment_scores_with_trace_profile(scores, event_type, opt)
            scores = augment_scores_with_type_gap_profile(scores, event_type, time_gap_norm, opt)

            anomaly_score = build_anomaly_score(scores, opt)
            uncertainty_score = scores['uncertainty_score']
            mask = scores['mask']
            pred_anomaly = (anomaly_score > gamma) & mask
            segment_score, valid_segment = aggregate_segment_scores(
                anomaly_score,
                mask,
                opt,
                gamma=gamma
            )
            pred_segment_anomaly = (segment_score > segment_gamma) & valid_segment

            if unified_writer is not None and event_label is not None:
                export_mask = mask & (event_label >= 0)
                export_indices = export_mask.nonzero(as_tuple=False).detach().cpu().tolist()
                export_labels = event_label.detach().cpu()
                export_preds = pred_anomaly.detach().cpu()
                export_scores = anomaly_score.detach().cpu()
                method_name = getattr(opt, 'unified_method', 'Ours')
                dataset_name = (
                    getattr(opt, 'unified_dataset', '')
                    or getattr(opt, 'detected_dataset', 'generic')
                )
                for batch_index, event_index in export_indices:
                    sequence_id = sequence_offset + int(batch_index)
                    unified_writer.writerow({
                        'method': method_name,
                        'dataset': dataset_name,
                        'split': 'test',
                        'level': 'event',
                        'sequence_id': sequence_id,
                        'event_id': f'{sequence_id}:{event_index}',
                        'segment_id': sequence_id,
                        'label': int(export_labels[batch_index, event_index].item()),
                        'pred': int(export_preds[batch_index, event_index].item()),
                        'score': float(export_scores[batch_index, event_index].item()),
                        'count': 1,
                    })
            sequence_offset += int(event_type.size(0))

            total_events += mask.sum().item()
            total_alerts += pred_anomaly.sum().item()
            total_segment_alerts += pred_segment_anomaly.sum().item()
            total_anomaly_score += anomaly_score[mask].sum().item()
            total_uncertainty += uncertainty_score[mask].sum().item()

            if event_label is not None:
                labeled_events += update_binary_counts(
                    detection_counts,
                    'Event',
                    pred_anomaly,
                    event_label,
                    mask
                )
                event_label_mask = mask & (event_label >= 0)
                if event_label_mask.any():
                    event_score_parts.append(anomaly_score[event_label_mask].detach().cpu())
                    event_truth_parts.append(event_label[event_label_mask].detach().cpu())
                label_mask = mask & (event_label >= 0)
                segment_labeled = label_mask.any(dim=1)
                segment_truth = ((event_label == 1) & mask).any(dim=1).long()
                segment_truth = segment_truth.masked_fill(~segment_labeled, -1)
                if segment_labeled.any():
                    segment_score_parts.append(segment_score[segment_labeled].detach().cpu())
                    segment_truth_parts.append(segment_truth[segment_labeled].detach().cpu())
                labeled_segments += update_binary_counts(
                    detection_counts,
                    'Segment',
                    pred_segment_anomaly,
                    segment_truth,
                    segment_labeled
                )
                if rq2_label is not None:
                    rq2_labeled_events += update_rq2_event_counts(
                        rq2_counts,
                        pred_anomaly,
                        rq2_label,
                        mask & (rq2_label >= 0)
                    )
                    rq2_labeled_segments += update_rq2_segment_counts(
                        rq2_counts,
                        pred_segment_anomaly,
                        rq2_label,
                        mask
                    )

    synchronize_device(opt.device)
    evaluation_time = time.perf_counter() - evaluation_start
    if unified_file is not None:
        unified_file.close()
        print(f'    Unified predictions:       {unified_path}')

    if total_events == 0:
        return {}

    results = {
        'Calib_Size': len(bank),
        'Gamma_Anomaly': gamma,
        'Gamma_Segment_Anomaly': segment_gamma,
        'Delta_Uncertainty': delta,
        'Anomaly_Score_Mode': getattr(opt, 'anomaly_score_mode', 'raw'),
        'Anomaly_Quantile': float(getattr(opt, 'anomaly_quantile', 0.99)),
        'Type_Score_Weight': float(getattr(opt, 'type_score_weight', 1.0)),
        'Time_Score_Weight': float(getattr(opt, 'time_score_weight', 1.0)),
        'Conditional_Gap_Weight': float(getattr(opt, 'conditional_gap_weight', 1.0)),
        'Profile_Score_Weight': float(getattr(opt, 'profile_score_weight', 0.0)),
        'Profile_Unigram_Weight': float(getattr(opt, 'profile_unigram_weight', 0.0)),
        'Profile_Bigram_Weight': float(getattr(opt, 'profile_bigram_weight', 0.0)),
        'Profile_Hist_Signature_Weight': float(getattr(opt, 'profile_hist_signature_weight', 0.0)),
        'Trace_Profile_Enabled': int(getattr(opt, '_trace_profile', None) is not None),
        'Detected_Dataset': getattr(opt, 'detected_dataset', 'generic'),
        'Segment_Score_Mode': getattr(opt, 'segment_score_mode', 'max'),
        'Segment_TopK': int(getattr(opt, 'segment_topk', 3)),
        'Events': total_events,
        'Point_Alerts': total_alerts,
        'Point_Alert_Rate': total_alerts / total_events,
        'Segment_Alerts': total_segment_alerts,
        'Labeled_Events': labeled_events,
        'Labeled_Segments': labeled_segments,
        'Mean_Anomaly_Score': total_anomaly_score / total_events,
        'Mean_Uncertainty_Score': total_uncertainty / total_events,
        'Calibration_Time_s': calibration_time,
        'Evaluation_Wall_Time_s': evaluation_time,
    }
    results.update(efficiency_metadata(model, opt))

    if getattr(opt, 'benchmark_efficiency', False):
        results.update(
            benchmark_anomaly_inference(
                model,
                test_data,
                opt,
                gamma,
                segment_gamma
            )
        )

    if labeled_events > 0:
        results.update(finalize_binary_metrics(detection_counts, 'Event'))
        if event_score_parts:
            event_scores = torch.cat(event_score_parts)
            event_truth = torch.cat(event_truth_parts)
            results.update(binary_ranking_metrics(event_scores, event_truth, 'Event'))
            results.update(score_distribution_stats(event_scores, event_truth, 'Event'))
    if labeled_segments > 0:
        results.update(finalize_binary_metrics(detection_counts, 'Segment'))
        segment_scores = torch.cat(segment_score_parts)
        segment_truths = torch.cat(segment_truth_parts)
        results.update(binary_ranking_metrics(segment_scores, segment_truths, 'Segment'))
        results.update(score_distribution_stats(segment_scores, segment_truths, 'Segment'))
        if getattr(opt, 'eval_segment_threshold_sweep', False) and segment_score_parts:
            best_segment = compute_best_threshold_metrics(
                segment_scores,
                segment_truths,
                steps=getattr(opt, 'threshold_sweep_steps', 200),
                extra_thresholds=[segment_gamma]
            )
            if best_segment is not None:
                results.update({
                    'Segment_BestF1_Oracle': best_segment['f1'],
                    'Segment_BestF1_Threshold': best_segment['threshold'],
                    'Segment_BestF1_Precision': best_segment['precision'],
                    'Segment_BestF1_Recall': best_segment['recall'],
                    'Segment_BestF1_FPR': best_segment['fpr'],
                    'Segment_BestF1_Alerts': best_segment['alerts'],
                    'Segment_BestF1_Alert_Rate': best_segment['alert_rate'],
                })
    if rq2_labeled_events > 0 or rq2_labeled_segments > 0:
        results['RQ2_Labeled_Events'] = rq2_labeled_events
        results['RQ2_Labeled_Segments'] = rq2_labeled_segments
        results.update(finalize_rq2_metrics(rq2_counts))

    print(f'\n  - (Anomaly Detection Summary)')
    print(f'    Calibration size:          {results["Calib_Size"]}')
    print(
        f'    Score mode / segment mode: '
        f'{results["Anomaly_Score_Mode"]} / {results["Segment_Score_Mode"]}'
    )
    print(f'    Gamma anomaly threshold:   {results["Gamma_Anomaly"]:.6f}')
    print(f'    Gamma segment threshold:   {results["Gamma_Segment_Anomaly"]:.6f}')
    print(f'    Point alerts:              {results["Point_Alerts"]}')
    print(f'    Point alert rate:          {results["Point_Alert_Rate"]:.4f}')
    print(f'    Calibration time:          {results["Calibration_Time_s"]:.3f} s')
    print(f'    Evaluation wall time:      {results["Evaluation_Wall_Time_s"]:.3f} s')
    if 'Inference_Time_Median_s' in results:
        print(
            f'    Inference median/repeats:  '
            f'{results["Inference_Time_Median_s"]:.3f} s / '
            f'{results["Inference_Repeats"]}'
        )
        print(f'    Throughput:                {results["Throughput_Events_s"]:.2f} events/s')
        print(f'    Latency per 1k events:     {results["Latency_ms_per_1k_Events"]:.3f} ms')
        print(f'    Sequence throughput:       {results["Throughput_Sequences_s"]:.2f} sequences/s')
        print(f'    Latency per 1k sequences:  {results["Latency_ms_per_1k_Sequences"]:.3f} ms')
        print(f'    Peak GPU memory:           {results["Peak_GPU_Memory_MB"]:.2f} MB')
        print(f'    Trainable parameters:      {results["Trainable_Parameters"]}')
    if labeled_events > 0:
        print(f'    Labeled events:            {results["Labeled_Events"]}')
        print(
            f'    Event P/R/F1:              '
            f'{results["Event_Precision"]:.4f} / '
            f'{results["Event_Recall"]:.4f} / '
            f'{results["Event_F1"]:.4f}'
        )
        print(f'    Event FPR:                 {results["Event_FPR"]:.4f}')
    else:
        print('    Labeled events:            0 (event P/R/F1 skipped)')

    if labeled_segments > 0:
        print(f'    Labeled segments:          {results["Labeled_Segments"]}')
        print(
            f'    Segment P/R/F1:            '
            f'{results["Segment_Precision"]:.4f} / '
            f'{results["Segment_Recall"]:.4f} / '
            f'{results["Segment_F1"]:.4f}'
        )
        print(f'    Segment FPR:               {results["Segment_FPR"]:.4f}')
        if 'Segment_AUPRC' in results:
            print(f'    Segment AUPRC/AUROC:       {results["Segment_AUPRC"]:.4f} / {results["Segment_AUROC"]:.4f}')
        if 'Segment_BestF1_Oracle' in results:
            print(
                f'    Segment best-F1 oracle:    '
                f'{results["Segment_BestF1_Precision"]:.4f} / '
                f'{results["Segment_BestF1_Recall"]:.4f} / '
                f'{results["Segment_BestF1_Oracle"]:.4f} '
                f'@ threshold {results["Segment_BestF1_Threshold"]:.6f}'
            )
    else:
        print('    Labeled segments:          0 (segment P/R/F1 skipped)')

    if results.get('RQ2_Labeled_Segments', 0) > 0:
        print(f'    RQ2 labeled segments:      {results["RQ2_Labeled_Segments"]}')
        print(
            f'    RQ2 Type/Time/Joint F1:    '
            f'{results["RQ2_TypeSegment_F1"]:.4f} / '
            f'{results["RQ2_TimeSegment_F1"]:.4f} / '
            f'{results["RQ2_JointSegment_F1"]:.4f}'
        )

    return results


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('-data', required=True)
    parser.add_argument('-normalize', type=str, default='log', choices=['normal', 'log'])

    # ================= 新增：长序列截断与稳定性控制参数 =================
    parser.add_argument('-window_size', type=int, default=100000, help='Sliding window max length')
    parser.add_argument('-step_size', type=int, default=50000, help='Sliding window step size')
    parser.add_argument(
        '-min_max_len',
        type=int,
        default=0,
        help='Minimum model max_len used to keep positional embeddings compatible with old checkpoints.'
    )
    parser.add_argument('-clamp_threshold', type=float, default=6.0, help='Threshold to clamp log-space predictions')
    parser.add_argument(
        '-disable_time_norm_guard',
        action='store_true',
        help='Disable runtime guard that normalizes raw time gaps when stale preprocessing is detected.'
    )
    parser.add_argument(
        '-time_norm_guard_threshold',
        type=float,
        default=20.0,
        help='If abs(time_gap) exceeds this value, treat it as raw time and normalize it.'
    )
    # ====================================================================

    parser.add_argument('-d_model', type=int, default=128)
    parser.add_argument('-d_inner_hid', type=int, default=256)
    parser.add_argument('-n_head', type=int, default=4)
    parser.add_argument('-n_layers', type=int, default=4)
    parser.add_argument('-dropout', type=float, default=0.1)
    parser.add_argument('-d_k', type=int, default=32)
    parser.add_argument('-d_v', type=int, default=32)
    parser.add_argument(
        '-type_head',
        type=str,
        default='gmm',
        choices=['gmm', 'linear', 'hybrid', 'markov', 'hybrid_markov'],
        help='Type prediction head used for mark prediction.'
    )
    parser.add_argument(
        '-use_pos_enc',
        action='store_true',
        help='Use learned event-order positional embeddings.'
    )
    parser.add_argument(
        '-disable_time_gap',
        action='store_true',
        help='Disable relative time-gap input in the encoder.'
    )
    parser.add_argument(
        '-context_mask_prob',
        type=float,
        default=0.0,
        help=(
            'Training-time probability of masking historical context events. '
            'Use a small value such as 0.05-0.15 when counterfactual context '
            'support will be used at evaluation time.'
        )
    )
    parser.add_argument(
        '-context_mask_min_history',
        type=int,
        default=1,
        help='Do not mask the first this-many history positions during context masking.'
    )

    parser.add_argument('-fm_sigma', type=float, default=0.5)
    
    parser.add_argument('-solver_method', type=str, default='euler')
    parser.add_argument('-solver_step_size', type=float, default=0.05)
    parser.add_argument('-n_samples', type=int, default=100)
    
    parser.add_argument('-d_latent', type=int, default=16)

    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-num_workers', type=int, default=2)
    parser.add_argument('-epoch', type=int, default=60)
    parser.add_argument('-lr', type=float, default=1e-4)
    
    parser.add_argument('-fm_loss_weight', type=float, default=1.0)
    parser.add_argument('-loss_lambda', type=float, default=5.0)
    parser.add_argument(
        '-tail_fm_weight',
        type=float,
        default=0.0,
        help='Extra CFM loss multiplier for long-tail time gaps; 0 disables it.'
    )
    parser.add_argument(
        '-tail_fm_threshold',
        type=float,
        default=1.0,
        help='Normalized log-time threshold above which tail FM weighting starts.'
    )
    parser.add_argument(
        '-tail_fm_power',
        type=float,
        default=2.0,
        help='Power applied to the positive tail signal for weighted FM loss.'
    )
    parser.add_argument(
        '-tail_fm_max',
        type=float,
        default=20.0,
        help='Maximum per-event multiplier for weighted tail FM loss.'
    )
    parser.add_argument(
        '-flow_cond_clip',
        type=float,
        default=0.0,
        help='Clamp flow conditioning features to this absolute value; 0 disables clipping.'
    )
    parser.add_argument(
        '-disable_mark_conditioned_flow',
        action='store_true',
        help='Use p(tau|H) instead of p(tau|H,m) by removing target mark conditioning from the time flow.'
    )
    parser.add_argument(
        '-fm_debug',
        action='store_true',
        help='Print flow-matching diagnostic statistics when validation FM is abnormal.'
    )
    parser.add_argument(
        '-fm_debug_threshold',
        type=float,
        default=1000.0,
        help='Validation FM threshold that triggers FM debug output.'
    )
    parser.add_argument(
        '-loss_weighting',
        type=str,
        default='fixed',
        choices=['adaptive', 'fixed'],
        help='Use adaptive homoscedastic weighting or fixed fm + lambda * type loss.'
    )

    parser.add_argument('-eval_epoch', type=int, default=5) 

    parser.add_argument('-save_path', default='./checkpoint.pth')
    parser.add_argument('-save_name', default='model')
    parser.add_argument(
        '-checkpoint_metric',
        type=str,
        default='valid_loss',
        choices=['valid_loss', 'valid_acc'],
        help='Metric used to select the saved checkpoint.'
    )
    parser.add_argument(
        '-checkpoint_min_delta',
        type=float,
        default=0.0,
        help='Minimum metric improvement required to overwrite the checkpoint.'
    )
    parser.add_argument('-eval_quantile_step', type=float, default=0.05) 
    parser.add_argument(
        '-eval_time_scale',
        choices=['legacy', 'physical'],
        default='legacy',
        help=(
            'Time scale for generation metrics and exact time NLL. '
            'legacy matches the original normalized/SMURF-style protocol; '
            'physical multiplies log-denormalized gaps by mean_data.'
        )
    )
    parser.add_argument('-seed', type=int, default=2023)
    
    parser.add_argument('-just_eval', action='store_true')
    parser.add_argument('-load_path_name', type=str, default=None)
    parser.add_argument('-save_result', type=str, default=None)
    parser.add_argument(
        '-benchmark_efficiency',
        action='store_true',
        help=(
            'Run warm-up and repeated test-set inference passes to measure '
            'latency, throughput, and peak GPU memory.'
        )
    )
    parser.add_argument(
        '-benchmark_warmup_batches',
        type=int,
        default=5,
        help='Number of test batches used for accelerator warm-up.'
    )
    parser.add_argument(
        '-benchmark_repeats',
        type=int,
        default=5,
        help='Number of complete test-set inference passes used for timing.'
    )
    parser.add_argument(
        '-save_unified_predictions',
        action='store_true',
        help='Export event-level y_true/y_pred/score rows for the shared RQ1 evaluator.'
    )
    parser.add_argument(
        '-unified_prediction_path',
        type=str,
        default='',
        help='Output CSV for shared-schema predictions; defaults to <save_result>_unified_predictions.csv.'
    )
    parser.add_argument(
        '-unified_method',
        type=str,
        default='Ours',
        help='Method name written to the shared prediction CSV.'
    )
    parser.add_argument(
        '-unified_dataset',
        type=str,
        default='',
        help='Dataset name written to the shared prediction CSV; auto-detected when omitted.'
    )
    parser.add_argument('-eval_anomaly_detection', action='store_true')
    parser.add_argument('-eval_reliability', action='store_true')
    parser.add_argument('-uncertainty_mc', type=int, default=8)
    parser.add_argument('-type_entropy_weight', type=float, default=0.0)
    parser.add_argument('-reliability_exact_nll', action='store_true')
    parser.add_argument('-anomaly_quantile', type=float, default=0.99)
    parser.add_argument('-uncertainty_quantile', type=float, default=0.95)
    parser.add_argument('-calibration_max_size', type=int, default=200000)
    parser.add_argument(
        '-anomaly_score_mode',
        type=str,
        default='raw',
        choices=[
            'raw', 'type_only', 'time_only', 'weighted_raw', 'zscore',
            'zscore_max', 'conditional_gap_zscore', 'profile_only', 'profile_zscore'
        ],
        help='How to combine type and time anomaly components for detection.'
    )
    parser.add_argument(
        '-type_score_weight',
        type=float,
        default=1.0,
        help='Weight for the type NLL component in weighted/zscore anomaly scoring.'
    )
    parser.add_argument(
        '-time_score_weight',
        type=float,
        default=1.0,
        help='Weight for the time NLL or FM-uncertainty component in weighted/zscore anomaly scoring.'
    )
    parser.add_argument(
        '-profile_score_weight',
        type=float,
        default=1.0,
        help='Weight for the normal trace profile component in profile_zscore anomaly scoring.'
    )
    parser.add_argument(
        '-conditional_gap_weight',
        type=float,
        default=1.0,
        help='Weight for type-conditioned gap mismatch in conditional_gap_zscore scoring.'
    )
    parser.add_argument(
        '-conditional_gap_min_count',
        type=int,
        default=20,
        help='Minimum calibration events required before fitting a type-specific gap profile.'
    )
    parser.add_argument(
        '-enable_trace_profile',
        action='store_true',
        help='Fit a normal-only transition/profile scorer on the calibration split.'
    )
    parser.add_argument(
        '-disable_dataset_adaptive_detection',
        action='store_true',
        help='Disable HDFS/Thunderbird-specific detection defaults.'
    )
    parser.add_argument('-profile_smoothing', type=float, default=0.1)
    parser.add_argument('-profile_bigram_weight', type=float, default=1.0)
    parser.add_argument('-profile_unigram_weight', type=float, default=0.25)
    parser.add_argument('-profile_hist_weight', type=float, default=0.0)
    parser.add_argument('-profile_length_weight', type=float, default=0.0)
    parser.add_argument('-profile_exact_signature_weight', type=float, default=0.0)
    parser.add_argument('-profile_hist_signature_weight', type=float, default=0.0)
    parser.add_argument('-profile_max_sequences', type=int, default=200000)
    parser.add_argument(
        '-segment_score_mode',
        type=str,
        default='max',
        choices=['max', 'mean', 'topk_mean', 'alert_fraction', 'excess_mean', 'excess_topk_mean'],
        help='How event anomaly scores are aggregated into a segment/window score.'
    )
    parser.add_argument(
        '-segment_topk',
        type=int,
        default=3,
        help='K used by topk_mean and excess_topk_mean segment score modes.'
    )
    parser.add_argument(
        '-eval_segment_threshold_sweep',
        action='store_true',
        help='Also report the test-label best segment threshold as an oracle diagnostic.'
    )
    parser.add_argument(
        '-eval_rq2_subsets',
        action='store_true',
        help='Report RQ2 Type/Time/Joint controlled anomaly subset metrics when rq2_label is present.'
    )
    parser.add_argument(
        '-eval_rq2_classifier',
        action='store_true',
        help='Train and evaluate a frozen-backbone RQ2 classification head on controlled type/time/joint labels.'
    )
    parser.add_argument(
        '-rq2_classifier_variant',
        type=str,
        default='ours_joint',
        choices=['type_only', 'time_only', 'independent_joint', 'ours_joint'],
        help='Which upstream type/time components are visible to the shared RQ2 classification head.'
    )
    parser.add_argument('-rq2_classifier_epochs', type=int, default=30)
    parser.add_argument('-rq2_classifier_lr', type=float, default=1e-3)
    parser.add_argument('-rq2_classifier_hidden', type=int, default=128)
    parser.add_argument('-rq2_classifier_dropout', type=float, default=0.1)
    parser.add_argument('-rq2_classifier_batch_size', type=int, default=8192)
    parser.add_argument('-rq2_classifier_eval_batch_size', type=int, default=65536)
    parser.add_argument('-rq2_classifier_weight_decay', type=float, default=1e-4)
    parser.add_argument('-rq2_classifier_binary_loss_weight', type=float, default=0.5)
    parser.add_argument('-rq2_classifier_max_class_weight', type=float, default=30.0)
    parser.add_argument('-rq2_classifier_max_binary_weight', type=float, default=50.0)
    parser.add_argument('-rq2_classifier_uncertainty_mc', type=int, default=1)
    parser.add_argument('-rq2_classifier_train_split', type=str, default='rq2_head_train')
    parser.add_argument('-rq2_classifier_dev_split', type=str, default='rq2_head_dev')
    parser.add_argument('-rq2_classifier_test_split', type=str, default='test')
    parser.add_argument(
        '-rq2_classifier_use_profile_features',
        action='store_true',
        help='Expose the optional type-gap profile feature to the RQ2 classifier head.'
    )
    parser.add_argument(
        '-threshold_sweep_steps',
        type=int,
        default=200,
        help='Number of quantile thresholds to evaluate for best-F1 diagnostics.'
    )
    parser.add_argument('-use_conformal_rejection', action='store_true')
    parser.add_argument('-conformal_alpha', type=float, default=0.05)
    parser.add_argument(
        '-use_ensemble_correction',
        action='store_true',
        help='Enable memory-based Gaussian latent ensemble correction for OOD candidates.'
    )
    parser.add_argument(
        '-ensemble_k',
        type=int,
        default=20,
        help='Number of calibration neighbors used by ensemble correction.'
    )
    parser.add_argument(
        '-ensemble_samples',
        type=int,
        default=32,
        help='Number of Gaussian latent samples drawn per OOD candidate.'
    )
    parser.add_argument(
        '-ensemble_noise_scale',
        type=float,
        default=0.1,
        help='Noise multiplier for sampling around neighbor Gaussian latents.'
    )
    parser.add_argument(
        '-ensemble_kernel',
        type=float,
        default=0.2,
        help='KDE kernel width in normalized time space; <=0 chooses it from samples.'
    )
    parser.add_argument(
        '-ensemble_correction_weight',
        type=float,
        default=1.0,
        help='Blend weight for ensemble corrected score; 1 uses the ensemble score.'
    )
    parser.add_argument(
        '-ensemble_disagreement_threshold',
        type=float,
        default=2.0,
        help=(
            'Reject a candidate when the per-sample ensemble time NLL standard '
            'deviation exceeds this value; <=0 disables this consistency gate.'
        )
    )
    parser.add_argument(
        '-ensemble_support_quantile',
        type=float,
        default=0.95,
        help='Calibration quantile used as the local-support radius.'
    )
    parser.add_argument(
        '-ensemble_max_reference',
        type=int,
        default=4096,
        help='Maximum calibration points used to fit the support radius.'
    )
    parser.add_argument(
        '-ensemble_max_search',
        type=int,
        default=50000,
        help='Maximum memory points searched for nearest neighbors; <=0 searches all.'
    )
    parser.add_argument(
        '-ensemble_step_size',
        type=float,
        default=0.0,
        help='ODE step size for ensemble generation; <=0 reuses -solver_step_size.'
    )
    parser.add_argument(
        '-use_drift_adapter',
        action='store_true',
        help='Enable progressive StepWise-style adaptation after high-confidence expected drift.'
    )
    parser.add_argument(
        '-drift_adapter_min_events',
        type=int,
        default=32,
        help='Minimum provisional expected-drift events before releasing expected-drift decisions.'
    )
    parser.add_argument(
        '-drift_adapter_fit_interval',
        type=int,
        default=16,
        help='Number of accepted provisional events between robust adapter refits.'
    )
    parser.add_argument(
        '-drift_adapter_min_support',
        type=float,
        default=0.5,
        help='Minimum local support required for a sample to update the drift adapter.'
    )
    parser.add_argument(
        '-drift_adapter_update_rate',
        type=float,
        default=0.5,
        help='EMA update rate for robust affine adapter parameters.'
    )
    parser.add_argument(
        '-drift_adapter_slope_min',
        type=float,
        default=0.5,
        help='Lower clamp for the latent affine slope.'
    )
    parser.add_argument(
        '-drift_adapter_slope_max',
        type=float,
        default=2.0,
        help='Upper clamp for the latent affine slope.'
    )
    parser.add_argument(
        '-drift_adapter_shift_clip',
        type=float,
        default=3.0,
        help='Absolute clamp for the latent affine shift.'
    )
    parser.add_argument(
        '-drift_adapter_support_min',
        type=float,
        default=0.75,
        help='Lower clamp for adaptive local-support radius scale.'
    )
    parser.add_argument(
        '-drift_adapter_support_max',
        type=float,
        default=1.5,
        help='Upper clamp for adaptive local-support radius scale.'
    )
    parser.add_argument(
        '-drift_adapter_max_buffer',
        type=int,
        default=2048,
        help='Maximum provisional expected-drift samples retained for robust refits.'
    )
    parser.add_argument(
        '-drift_adapter_update_memory',
        action='store_true',
        help='After the adapter is stable, write high-confidence expected drift into normal memory.'
    )
    parser.add_argument(
        '-use_component_drift_diagnosis',
        action='store_true',
        help=(
            'Use separate type/time/profile evidence gates for Expected, '
            'Unexpected, and Reject drift diagnosis.'
        )
    )
    parser.add_argument(
        '-drift_diagnosis_type_quantile',
        type=float,
        default=0.995,
        help='Normal-calibration quantile for structural/type drift evidence.'
    )
    parser.add_argument(
        '-drift_diagnosis_time_quantile',
        type=float,
        default=0.995,
        help='Normal-calibration quantile for corrected time-drift evidence.'
    )
    parser.add_argument(
        '-drift_diagnosis_profile_quantile',
        type=float,
        default=0.995,
        help='Normal-calibration quantile for transition/profile drift evidence.'
    )
    parser.add_argument(
        '-drift_diagnosis_strong_quantile',
        type=float,
        default=0.999,
        help='Normal-calibration quantile for strong Unexpected evidence.'
    )
    parser.add_argument(
        '-drift_diagnosis_extreme_time_quantile',
        type=float,
        default=0.9995,
        help=(
            'Normal-calibration quantile for extreme raw time evidence. '
            'Extreme raw temporal evidence vetoes Expected correction and '
            'is treated as Unexpected evidence.'
        )
    )
    parser.add_argument(
        '-use_counterfactual_context_support',
        action='store_true',
        help=(
            'Use counterfactual context likelihood support for RQ4 diagnosis. '
            'Recent history events are masked and the current event NLL change '
            'is used as type/time context support evidence.'
        )
    )
    parser.add_argument(
        '-counterfactual_support_k',
        type=int,
        default=3,
        help='Number of recent history events tested for counterfactual support.'
    )
    parser.add_argument(
        '-counterfactual_support_epsilon',
        type=float,
        default=0.0,
        help='Minimum positive NLL increase after masking to count as support.'
    )
    parser.add_argument(
        '-counterfactual_support_chunk_size',
        type=int,
        default=128,
        help='Expanded-batch chunk size for counterfactual support evaluation.'
    )
    parser.add_argument(
        '-counterfactual_support_time_mode',
        choices=['exact', 'off'],
        default='exact',
        help='Whether to compute exact CNF time-NLL counterfactual support.'
    )
    parser.add_argument(
        '-counterfactual_type_support_ratio',
        type=float,
        default=0.34,
        help='Minimum fraction of tested history events with positive type support.'
    )
    parser.add_argument(
        '-counterfactual_time_support_ratio',
        type=float,
        default=0.34,
        help='Minimum fraction of tested history events with positive time support.'
    )
    parser.add_argument(
        '-counterfactual_type_support_strength',
        type=float,
        default=0.0,
        help='Minimum averaged positive type contribution strength.'
    )
    parser.add_argument(
        '-counterfactual_time_support_strength',
        type=float,
        default=0.0,
        help='Minimum averaged positive time contribution strength.'
    )
    parser.add_argument(
        '-counterfactual_mem_gain_threshold',
        type=float,
        default=0.0,
        help='Minimum score reduction required for memory recovery.'
    )
    parser.add_argument(
        '-treat_oov_type_as_unavailable',
        action='store_true',
        help=(
            'Treat the configured OOV/UNK target type as unavailable type '
            'evidence during drift diagnosis instead of accepting UNK support.'
        )
    )
    parser.add_argument(
        '-oov_type_id',
        type=int,
        default=-1,
        help='Zero-based type id used for OOV/UNK targets in the raw event ids.'
    )
    parser.add_argument(
        '-use_absence_aware_revision',
        action='store_true',
        help=(
            'Gate Expected-drift memory revision with Context-conditioned '
            'Absence Memory.'
        )
    )
    parser.add_argument(
        '-absence_reference_path',
        type=str,
        default='',
        help=(
            'Optional standalone absence_memory.pkl (or its containing directory). '
            'When set, it is used only by the absence mechanism and does not alter '
            'the model training split.'
        )
    )
    parser.add_argument(
        '-absence_reference_split',
        choices=['train', 'calibration'],
        default='train',
        help='Raw split used as trusted normal service-coverage memory.'
    )
    parser.add_argument(
        '-absence_context_mode',
        choices=['context_memory', 'hybrid', 'metadata', 'memory'],
        default='context_memory',
        help=(
            'context_memory is the metadata-free main-paper mechanism with '
            'leave-one-service-out retrieval, exposure normalisation, empirical '
            'lower-tail scoring, and run-horizon persistence. Other modes are '
            'retained only for legacy/upper-bound comparisons.'
        )
    )
    parser.add_argument(
        '-absence_metadata_fields',
        type=str,
        default='',
        help=(
            'Legacy-only comma-separated metadata fields. The main-paper '
            'context_memory mode ignores this option and uses log context only.'
        )
    )
    parser.add_argument(
        '-absence_exclude_services',
        type=str,
        default='system-observability,tsdb-mysql,nacosdb-mysql',
        help='Comma-separated service names excluded from absence evidence.'
    )
    parser.add_argument(
        '-absence_service_prefixes',
        type=str,
        default='',
        help='Optional comma-separated allowed service prefixes; empty allows all non-excluded services.'
    )
    parser.add_argument('-absence_k', type=int, default=20)
    parser.add_argument('-absence_active_beta', type=float, default=0.7)
    parser.add_argument('-absence_min_expected_count', type=float, default=20.0)
    parser.add_argument('-absence_count_ratio_threshold', type=float, default=0.5)
    parser.add_argument('-absence_anomaly_threshold', type=float, default=2.0)
    parser.add_argument('-absence_persistence_threshold', type=float, default=0.5)
    parser.add_argument('-absence_min_context_similarity', type=float, default=0.2)
    parser.add_argument('-absence_min_query_exposure', type=float, default=50.0)
    parser.add_argument('-absence_sigma_floor_ratio', type=float, default=0.25)
    parser.add_argument('-absence_coverage_threshold', type=float, default=0.5)
    parser.add_argument('-drift_window_size', type=int, default=1000)
    parser.add_argument('-drift_threshold', type=float, default=0.3)
    parser.add_argument('-quarantine_max_size', type=int, default=50000)
    parser.add_argument(
        '-save_rq4_event_details',
        action='store_true',
        help='Save per-event RQ4 drift-diagnosis details for controlled drift experiments.'
    )
    parser.add_argument(
        '-rq4_event_detail_max',
        type=int,
        default=200000,
        help='Maximum controlled drift events written to the RQ4 detail CSV.'
    )
    parser.add_argument(
        '-rq4_event_detail_include_normal',
        action='store_true',
        help=(
            'When saving RQ4 event details, also write events with drift label 0. '
            'This is useful for no-drift/control run-level false-alarm analysis.'
        )
    )
    parser.add_argument(
        '-rq4_candidate_mode',
        choices=['score', 'score_or_profile', 'labeled', 'labeled_or_score'],
        default='score',
        help=(
            'Candidate source for controlled RQ4 diagnosis. '
            'score uses only threshold-triggered OOD candidates; score_or_profile '
            'adds dev-calibrated trace-profile shift candidates without using test '
            'labels; labeled forces controlled drift labels into the diagnosis '
            'stage; labeled_or_score uses the union of both.'
        )
    )
    parser.add_argument(
        '-drift_candidate_profile_quantile',
        type=float,
        default=0.95,
        help='Dev-normal quantile used to trigger unsupervised trace-profile drift candidates.'
    )
    parser.add_argument(
        '-use_rq4_window_diagnosis',
        action='store_true',
        help='Smooth RQ4 event diagnoses within each controlled candidate window.'
    )
    parser.add_argument(
        '-rq4_window_expected_min_frac',
        type=float,
        default=0.50,
        help='Minimum Expected event fraction for window-level Expected diagnosis.'
    )
    parser.add_argument(
        '-rq4_window_unexpected_min_frac',
        type=float,
        default=0.25,
        help='Minimum Unexpected event fraction for window-level Unexpected diagnosis.'
    )
    parser.add_argument(
        '-rq4_window_reject_min_frac',
        type=float,
        default=0.20,
        help='Minimum Reject event fraction for window-level Reject diagnosis.'
    )
    parser.add_argument(
        '-rq4_window_conflict_min_frac',
        type=float,
        default=0.20,
        help='Minimum Expected and Unexpected fractions that indicate a conflict window.'
    )
    
    opt = parser.parse_args()
    if opt.use_drift_adapter:
        opt.use_ensemble_correction = True
    opt.use_time_gap = not opt.disable_time_gap
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    opt.return_event_labels = opt.eval_anomaly_detection or opt.eval_reliability or opt.eval_rq2_classifier
    opt.return_drift_labels = opt.eval_reliability
    opt.return_rq2_labels = (
        (opt.eval_anomaly_detection and opt.eval_rq2_subsets)
        or opt.eval_rq2_classifier
    )
    configure_dataset_adaptive_detection(opt)

    torch.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(opt.seed)

    opt.eval_quantile = torch.arange(opt.eval_quantile_step, 1.0, opt.eval_quantile_step, device=opt.device)

    trainloader, devloader, testloader, num_types = prepare_dataloader(opt)
    opt.num_types = num_types

    model = FlowMatchingTHP(num_types, opt)

    model.mean_log_data = getattr(opt, 'mean_log_data', 0)
    model.var_log_data = getattr(opt, 'var_log_data', 1)
    model.mean_data = getattr(opt, 'mean_data', 1)
    model.to(opt.device)

    if opt.just_eval:
        if opt.load_path_name is not None:
            print(f'[Info] Loading model from {opt.load_path_name} ...')
            checkpoint = torch.load(opt.load_path_name, map_location=opt.device)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint, strict=False)
        else:
            print("[Error] Provide -load_path_name for evaluation.")
            return

        print(f'[Info] Start Evaluation...')
        print(
            f'[Info] Detection config: dataset={getattr(opt, "detected_dataset", "generic")} | '
            f'score={opt.anomaly_score_mode} | segment={opt.segment_score_mode} | '
            f'type_w={opt.type_score_weight} | time_w={opt.time_score_weight} | '
            f'profile_w={opt.profile_score_weight} | trace_profile={opt.enable_trace_profile}'
        )
        if opt.eval_rq2_classifier:
            results = eval_rq2_classifier(model, opt)
        elif opt.eval_reliability:
            results = eval_reliability(model, devloader, testloader, opt)
        elif opt.eval_anomaly_detection:
            results = eval_anomaly_detection(model, devloader, testloader, opt)
        else:
            results = eval_epoch(model, testloader, True, opt)

        training_summary_file = training_efficiency_path(opt.load_path_name)
        training_efficiency = load_single_row_csv(training_summary_file)
        if training_efficiency:
            for key, value in training_efficiency.items():
                results.setdefault(key, value)
            results['Training_Efficiency_Source'] = training_summary_file
            print(f'[Info] Loaded training efficiency: {training_summary_file}')
        if opt.save_result and results:
            result_path = f"{opt.save_result}_results.csv"
            if pd is not None:
                df = pd.DataFrame([results])
                df.to_csv(result_path, index=False)
            else:
                import csv
                with open(result_path, mode='w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=list(results.keys()))
                    writer.writeheader()
                    writer.writerow(results)
        return

    print(f'[Info] Model Parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'[Info] Checkpoint metric: {opt.checkpoint_metric}')
    print(f'[Info] Type head: {opt.type_head}')
    print(f'[Info] Event-order positional encoding: {opt.use_pos_enc}')
    print(f'[Info] Encoder time-gap input: {opt.use_time_gap}')
    if opt.context_mask_prob > 0:
        print(
            f'[Info] Context masking: p={opt.context_mask_prob} | '
            f'min_history={opt.context_mask_min_history}'
        )
    print(
        f'[Info] Time normalization guard: '
        f'{not opt.disable_time_norm_guard} | threshold: {opt.time_norm_guard_threshold}'
    )
    print(
        f'[Info] Loss weighting: {opt.loss_weighting} | '
        f'fm_loss_weight: {opt.fm_loss_weight} | loss_lambda: {opt.loss_lambda}'
    )
    print(
        f'[Info] Anomaly score: {opt.anomaly_score_mode} | '
        f'type_weight: {opt.type_score_weight} | time_weight: {opt.time_score_weight} | '
        f'profile_weight: {opt.profile_score_weight} | '
        f'segment_score: {opt.segment_score_mode} | segment_topk: {opt.segment_topk}'
    )
    print(
        f'[Info] Trace profile: {opt.enable_trace_profile} | '
        f'detected_dataset: {getattr(opt, "detected_dataset", "generic")}'
    )
    print(f'[Info] Eval time scale: {opt.eval_time_scale}')
    if opt.flow_cond_clip > 0:
        print(f'[Info] Flow condition clipping: +/-{opt.flow_cond_clip}')
    print(f'[Info] Mark-conditioned time flow: {not opt.disable_mark_conditioned_flow}')
    if opt.tail_fm_weight > 0:
        print(
            f'[Info] Tail FM weighting: weight={opt.tail_fm_weight} | '
            f'threshold={opt.tail_fm_threshold} | power={opt.tail_fm_power} | '
            f'max={opt.tail_fm_max}'
        )
    if 'hdfs' in opt.data.lower() and (opt.loss_weighting != 'fixed' or opt.loss_lambda < 1.0):
        print(
            '[Warning] HDFS benchmark runs should usually use '
            '-loss_weighting fixed with a strong type weight such as -loss_lambda 5.0. '
            'The current setting is intended for ablation/reliability exploration, '
            'not for the main Acc table.'
        )
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.8)
    maximize_checkpoint_metric = opt.checkpoint_metric in ['valid_acc']
    best_checkpoint_score = -float('inf') if maximize_checkpoint_metric else float('inf')

    synchronize_device(opt.device)
    if opt.device.type == 'cuda' and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(opt.device)
    training_start = time.perf_counter()
    best_epoch = 0
    time_to_best = float('nan')
    epoch_timing_rows = []

    for epoch in range(1, opt.epoch + 1):
        synchronize_device(opt.device)
        epoch_start = time.perf_counter()
        print(f'[ Epoch {epoch} ]')

        synchronize_device(opt.device)
        train_phase_start = time.perf_counter()
        t_loss, t_fm, t_type = train_epoch(model, trainloader, optimizer, opt)
        synchronize_device(opt.device)
        train_phase_time = time.perf_counter() - train_phase_start
        print(f'  - (Train) Loss: {t_loss:.4f} | FM: {t_fm:.4f} | Type: {t_type:.4f}')

        synchronize_device(opt.device)
        validation_phase_start = time.perf_counter()
        v_loss, v_acc, v_fm, v_type = eval_epoch(model, devloader, False, opt)
        synchronize_device(opt.device)
        validation_phase_time = time.perf_counter() - validation_phase_start
        print(
            f'  - (Valid) Loss: {v_loss:.4f} | FM: {v_fm:.4f} | '
            f'Type: {v_type:.4f} | Acc: {v_acc:.4f}'
        )

        checkpoint_score = v_acc if opt.checkpoint_metric == 'valid_acc' else v_loss
        if maximize_checkpoint_metric:
            improved = checkpoint_score > best_checkpoint_score + opt.checkpoint_min_delta
        else:
            improved = checkpoint_score < best_checkpoint_score - opt.checkpoint_min_delta

        if improved:
            best_checkpoint_score = checkpoint_score
            best_epoch = epoch
            print(
                f'    -> Best {opt.checkpoint_metric} updated: '
                f'{best_checkpoint_score:.4f}, Saving model...'
            )
            torch.save(model.state_dict(), opt.save_path)
            synchronize_device(opt.device)
            time_to_best = time.perf_counter() - training_start

        periodic_test_time = 0.0
        if epoch % opt.eval_epoch == 0:
            print("  - (Running Generation Test...)")
            synchronize_device(opt.device)
            periodic_test_start = time.perf_counter()
            eval_epoch(model, testloader, True, opt)
            synchronize_device(opt.device)
            periodic_test_time = time.perf_counter() - periodic_test_start

        scheduler.step()

        synchronize_device(opt.device)
        epoch_wall_time = time.perf_counter() - epoch_start
        epoch_timing_rows.append({
            'Epoch': epoch,
            'Train_Phase_Time_s': train_phase_time,
            'Validation_Phase_Time_s': validation_phase_time,
            'Periodic_Test_Time_s': periodic_test_time,
            'Epoch_Wall_Time_s': epoch_wall_time,
            'Checkpoint_Improved': int(improved),
            'Checkpoint_Metric': checkpoint_score,
        })

    synchronize_device(opt.device)
    total_training_time = time.perf_counter() - training_start
    epoch_wall_times = [row['Epoch_Wall_Time_s'] for row in epoch_timing_rows]
    peak_training_memory_mb = 0.0
    if opt.device.type == 'cuda' and torch.cuda.is_available():
        peak_training_memory_mb = torch.cuda.max_memory_allocated(opt.device) / (1024.0 ** 2)

    training_efficiency = {
        'Training_Wall_Time_Total_s': total_training_time,
        'Training_Time_To_Best_Checkpoint_s': time_to_best,
        'Epochs_Completed': len(epoch_timing_rows),
        'Best_Epoch': best_epoch,
        'Checkpoint_Metric_Name': opt.checkpoint_metric,
        'Best_Checkpoint_Metric': best_checkpoint_score,
        'Epoch_Time_Mean_s': float(np.mean(epoch_wall_times)) if epoch_wall_times else 0.0,
        'Epoch_Time_Median_s': float(np.median(epoch_wall_times)) if epoch_wall_times else 0.0,
        'Epoch_Time_Std_s': float(np.std(epoch_wall_times)) if epoch_wall_times else 0.0,
        'Peak_Training_GPU_Memory_MB': peak_training_memory_mb,
        'Checkpoint_Path': opt.save_path,
        'Training_Timing_Scope': (
            'train+dev validation+checkpoint I/O+configured periodic generation tests; '
            'data loading before epoch 1 excluded'
        ),
    }
    training_efficiency.update(efficiency_metadata(model, opt))
    summary_path = training_efficiency_path(opt.save_path)
    epoch_path = training_epoch_efficiency_path(opt.save_path)
    save_single_row_csv(summary_path, training_efficiency)
    save_rows_csv(epoch_path, epoch_timing_rows)
    print(f'[Info] Training efficiency summary: {summary_path}')
    print(f'[Info] Per-epoch timing details:     {epoch_path}')

if __name__ == '__main__':
    main()
