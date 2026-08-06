# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint
from collections import deque

import transformer.Constants as Constants
from transformer.Layers import Encoder, get_non_pad_mask
from flow_matching.path import ConditionalFlowMatcher
from flow_matching.loss import FlowMatchingLoss


NORMAL_DECISION = 0
ANOMALY_DECISION = 1
REJECT_DECISION = 2
EXPECTED_DRIFT_DECISION = 3
PAD_DECISION = -1


# --- 1. GMM Classifier Head ---
class GMMClassifierHead(nn.Module):
    def __init__(self, d_model, num_types):
        super().__init__()
        self.d_model = d_model
        self.num_types = num_types
        self.centers = nn.Parameter(torch.randn(num_types, d_model))

    def forward(self, z):
        B, L, D = z.shape
        z_flat = z.reshape(-1, D)
        dists = torch.cdist(z_flat, self.centers, p=2).pow(2)
        logits = -0.5 * dists
        return logits.view(B, L, self.num_types)


class LinearClassifierHead(nn.Module):
    def __init__(self, d_model, num_types):
        super().__init__()
        self.linear = nn.Linear(d_model, num_types)

    def forward(self, z):
        return self.linear(z)


class HybridClassifierHead(nn.Module):
    """
    Prototype logits plus a linear correction. This keeps the stable prototype
    geometry while giving HDFS-style deterministic marks enough classifier capacity.
    """
    def __init__(self, d_model, num_types):
        super().__init__()
        self.prototype_head = GMMClassifierHead(d_model, num_types)
        self.linear_head = LinearClassifierHead(d_model, num_types)

    def forward(self, z):
        return self.prototype_head(z) + self.linear_head(z)


class CausalMarkContextHead(nn.Module):
    """
    Local-order mark predictor for log streams. The output at position i only
    uses event types up to i, so it predicts event i+1 without future leakage.
    """
    def __init__(self, d_model, num_types, kernel_sizes=(1, 2, 4, 8, 12), dropout=0.1):
        super().__init__()
        self.kernel_sizes = tuple(kernel_sizes)
        self.convs = nn.ModuleList([
            nn.Conv1d(d_model, d_model, kernel_size=k, padding=k - 1)
            for k in self.kernel_sizes
        ])
        self.proj = nn.Linear(d_model * len(self.kernel_sizes), d_model)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(d_model, num_types)

    def forward(self, event_emb):
        seq_len = event_emb.size(1)
        x = event_emb.transpose(1, 2)
        features = []
        for conv in self.convs:
            h = conv(x)[:, :, :seq_len]
            features.append(F.gelu(h.transpose(1, 2)))

        h = torch.cat(features, dim=-1)
        h = self.proj(h)
        h = self.norm(h + event_emb)
        h = self.dropout(h)
        return self.classifier(h[:, :-1, :])


# --- 2. Gated Fusion Layer ---
class GatedFusion(nn.Module):
    """
    Use a gating mechanism to fuse History Context (h) and Target Type Embedding (e).
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.linear_h = nn.Linear(d_model, d_model)
        self.linear_e = nn.Linear(d_model, d_model)
        
        self.gate_net = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, h, type_emb):
        concat = torch.cat([h, type_emb], dim=-1)
        g = self.gate_net(concat) 
        fused = g * self.linear_h(h) + (1 - g) * self.linear_e(type_emb)
        out = self.out_proj(fused)
        out = self.dropout(out)
        out = self.norm(out + h)
        return out


# --- 3. AdaIN Block ---
class AdaINBlock(nn.Module):
    def __init__(self, d_hid, d_cond):
        super().__init__()
        self.norm = nn.GroupNorm(8, d_hid)
        self.linear1 = nn.Linear(d_hid, d_hid)
        self.linear2 = nn.Linear(d_hid, d_hid)
        self.act = nn.GELU()
        self.scale_shift_gen = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_cond, 2 * d_hid)
        )

    def forward(self, x, c):
        residual = x
        h = self.norm(x)
        c_perm = c.permute(0, 2, 1)
        scale_shift = self.scale_shift_gen(c_perm.transpose(1, 2)).transpose(1, 2)
        scale, shift = torch.chunk(scale_shift, 2, dim=1)
        h = h * (1 + scale) + shift
        h = h.permute(0, 2, 1)
        h = self.act(self.linear1(h))
        h = self.linear2(h)
        h = h.permute(0, 2, 1)
        return residual + h


# --- 4. Optimized Vector Field ---
class VectorField_Optimized(nn.Module):
    def __init__(self, d_in, d_cond, d_hid, d_out, num_blocks=4):
        super().__init__()
        self.t_embed_layer = nn.Sequential(
            nn.Linear(1, d_hid),
            nn.GELU(),
            nn.Linear(d_hid, d_hid)
        )
        self.initial_mapping = nn.Linear(d_in + d_hid, d_hid)
        self.blocks = nn.ModuleList([
            AdaINBlock(d_hid, d_cond) for _ in range(num_blocks)
        ])
        self.final_output = nn.Linear(d_hid, d_out)

    def forward(self, x, t, c):
        t_embed = self.t_embed_layer(t)
        x_input = torch.cat([x, t_embed], dim=-1)
        h = self.initial_mapping(x_input)
        h = h.permute(0, 2, 1)
        for block in self.blocks:
            h = block(h, c)
        h = h.permute(0, 2, 1)
        out = self.final_output(h)
        return out


class CalibrationMemoryBank:
    """
    Stores validation scores and optional latent features from normal data.

    The feature memory is used by the OOD correction stage: candidate events are
    compared with calibration events, then local Gaussian latents are sampled
    from the nearest neighbors to estimate a corrected event likelihood.
    """
    def __init__(self, max_size=None, device='cpu'):
        self.max_size = max_size
        self.device = device
        self.anomaly_scores = torch.empty(0, device=device)
        self.uncertainty_scores = torch.empty(0, device=device)
        self.features = torch.empty(0, 0, device=device)
        self.gaussian_latents = torch.empty(0, 1, device=device)
        self.feature_mean = None
        self.feature_std = None
        self.support_radius = None

    def update(self, anomaly_score, uncertainty_score, mask=None, features=None, gaussian_latent=None):
        anomaly_score = anomaly_score.detach().to(self.device).float()
        uncertainty_score = uncertainty_score.detach().to(self.device).float()

        if mask is not None:
            mask = mask.detach().to(self.device).bool()
            anomaly_score = anomaly_score[mask]
            uncertainty_score = uncertainty_score[mask]
            if features is not None:
                features = features.detach().to(self.device).float()[mask]
            if gaussian_latent is not None:
                gaussian_latent = gaussian_latent.detach().to(self.device).float()[mask]
        else:
            anomaly_score = anomaly_score.reshape(-1)
            uncertainty_score = uncertainty_score.reshape(-1)
            if features is not None:
                features = features.detach().to(self.device).float()
                features = features.reshape(-1, features.size(-1))
            if gaussian_latent is not None:
                gaussian_latent = gaussian_latent.detach().to(self.device).float()
                gaussian_latent = gaussian_latent.reshape(-1, gaussian_latent.size(-1))

        if anomaly_score.numel() == 0:
            return

        self.anomaly_scores = torch.cat([self.anomaly_scores, anomaly_score.reshape(-1)])
        self.uncertainty_scores = torch.cat([self.uncertainty_scores, uncertainty_score.reshape(-1)])

        if features is not None and features.numel() > 0:
            features = features.reshape(-1, features.size(-1))
            if self.features.numel() == 0:
                self.features = features
            elif self.features.size(-1) == features.size(-1):
                self.features = torch.cat([self.features, features])
            else:
                raise ValueError('Calibration feature dimension changed.')

        if gaussian_latent is not None and gaussian_latent.numel() > 0:
            gaussian_latent = gaussian_latent.reshape(-1, gaussian_latent.size(-1))
            if self.gaussian_latents.numel() == 0:
                self.gaussian_latents = gaussian_latent
            else:
                self.gaussian_latents = torch.cat([self.gaussian_latents, gaussian_latent])

        if self.max_size is not None and self.anomaly_scores.numel() > self.max_size:
            self.anomaly_scores = self.anomaly_scores[-self.max_size:]
            self.uncertainty_scores = self.uncertainty_scores[-self.max_size:]
            if self.features.numel() > 0:
                self.features = self.features[-self.max_size:]
            if self.gaussian_latents.numel() > 0:
                self.gaussian_latents = self.gaussian_latents[-self.max_size:]

        self.feature_mean = None
        self.feature_std = None
        self.support_radius = None

    def __len__(self):
        return int(self.anomaly_scores.numel())

    @property
    def has_feature_memory(self):
        return (
            self.features.numel() > 0
            and self.gaussian_latents.numel() > 0
            and self.features.size(0) == self.anomaly_scores.numel()
            and self.gaussian_latents.size(0) == self.anomaly_scores.numel()
        )

    def thresholds(self, anomaly_quantile=0.99, uncertainty_quantile=0.95):
        if len(self) == 0:
            raise ValueError('CalibrationMemoryBank is empty.')

        gamma = torch.quantile(self.anomaly_scores, anomaly_quantile).item()
        delta = torch.quantile(self.uncertainty_scores, uncertainty_quantile).item()
        return gamma, delta

    def fit_feature_space(self, k=20, support_quantile=0.95, max_reference=4096):
        if not self.has_feature_memory:
            return False

        features = self.features.float()
        self.feature_mean = features.mean(dim=0, keepdim=True)
        self.feature_std = features.std(dim=0, unbiased=False, keepdim=True).clamp_min(1e-6)
        norm_features = (features - self.feature_mean) / self.feature_std

        n_items = norm_features.size(0)
        if n_items < 2:
            self.support_radius = float('inf')
            return True

        ref_size = min(int(max_reference), n_items)
        if ref_size < n_items:
            ref_idx = torch.linspace(0, n_items - 1, steps=ref_size, device=norm_features.device)
            ref_idx = ref_idx.round().long().unique()
        else:
            ref_idx = torch.arange(n_items, device=norm_features.device)

        ref_features = norm_features[ref_idx]
        dist = torch.cdist(ref_features, ref_features)
        if dist.size(0) > 1:
            dist.fill_diagonal_(float('inf'))
        k_eff = min(max(1, int(k)), max(1, dist.size(1) - 1))
        local_dist = torch.topk(dist, k=k_eff, largest=False, dim=1).values.mean(dim=1)
        local_dist = local_dist[torch.isfinite(local_dist)]
        if local_dist.numel() == 0:
            self.support_radius = 1.0
        else:
            radius = torch.quantile(local_dist, float(support_quantile)).item()
            self.support_radius = max(float(radius), 1e-6)
        return True

    def _normalized_memory(self, max_search_size=None):
        if self.feature_mean is None or self.feature_std is None:
            self.fit_feature_space()

        norm_features = (self.features - self.feature_mean) / self.feature_std
        n_items = norm_features.size(0)
        if max_search_size is not None and max_search_size > 0 and n_items > max_search_size:
            search_idx = torch.linspace(0, n_items - 1, steps=int(max_search_size), device=norm_features.device)
            search_idx = search_idx.round().long().unique()
            return norm_features[search_idx], search_idx

        search_idx = torch.arange(n_items, device=norm_features.device)
        return norm_features, search_idx

    def sample_local_latents(
            self,
            features,
            mask,
            k=20,
            n_samples=32,
            noise_scale=0.1,
            max_search_size=50000,
            support_scale=1.0):
        if not self.has_feature_memory:
            return None

        if self.feature_mean is None or self.feature_std is None or self.support_radius is None:
            self.fit_feature_space(k=k)

        mask_cpu = mask.detach().to(self.device).bool()
        query = features.detach().to(self.device).float()[mask_cpu]
        if query.numel() == 0:
            return None

        query = query.reshape(-1, query.size(-1))
        query_norm = (query - self.feature_mean) / self.feature_std
        memory_norm, search_idx = self._normalized_memory(max_search_size=max_search_size)
        if memory_norm.numel() == 0:
            return None

        dist = torch.cdist(query_norm, memory_norm)
        k_eff = min(max(1, int(k)), memory_norm.size(0))
        topk_dist, topk_pos = torch.topk(dist, k=k_eff, largest=False, dim=1)
        memory_idx = search_idx[topk_pos]

        neighbor_latents = self.gaussian_latents[memory_idx]
        neighbor_scores = self.anomaly_scores[memory_idx]
        radius = max(float(self.support_radius) * max(float(support_scale), 1e-6), 1e-6)
        weights = torch.softmax(-topk_dist / radius, dim=1)

        n_samples = max(1, int(n_samples))
        draw_idx = torch.multinomial(weights, num_samples=n_samples, replacement=True)
        latent_dim = neighbor_latents.size(-1)
        gather_idx = draw_idx.unsqueeze(-1).expand(-1, -1, latent_dim)
        sampled_latents = torch.gather(neighbor_latents, dim=1, index=gather_idx)

        local_std = neighbor_latents.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-4)
        if noise_scale > 0:
            sampled_latents = sampled_latents + torch.randn_like(sampled_latents) * float(noise_scale) * local_std

        local_distance = topk_dist.mean(dim=1)
        local_support = torch.exp(-local_distance / radius)
        supported = local_distance <= radius
        neighbor_score_mean = (neighbor_scores * weights).sum(dim=1)
        neighbor_latent_mean = (neighbor_latents * weights.unsqueeze(-1)).sum(dim=1)

        return {
            'sampled_latents': sampled_latents,
            'neighbor_latent_mean': neighbor_latent_mean,
            'local_distance': local_distance,
            'local_support': local_support,
            'supported': supported,
            'neighbor_score_mean': neighbor_score_mean,
            'support_radius': radius
        }

    def conformal_p_values(self, values, bank='uncertainty'):
        calib = self.uncertainty_scores if bank == 'uncertainty' else self.anomaly_scores
        if calib.numel() == 0:
            raise ValueError('CalibrationMemoryBank is empty.')

        original_shape = values.shape
        values_flat = values.detach().to(calib.device).float().reshape(-1)
        sorted_calib = torch.sort(calib).values
        idx = torch.searchsorted(sorted_calib, values_flat, right=False)
        num_ge = sorted_calib.numel() - idx
        p_values = (1.0 + num_ge.float()) / (1.0 + sorted_calib.numel())
        return p_values.reshape(original_shape).to(values.device)


class DriftBuffer:
    """
    Quarantine buffer for rejected high-uncertainty events. It intentionally stores
    scores instead of updating the model immediately.
    """
    def __init__(self, max_size=50000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def update(self, anomaly_score, uncertainty_score, event_type=None, mask=None):
        anomaly_score = anomaly_score.detach().cpu().float()
        uncertainty_score = uncertainty_score.detach().cpu().float()

        if event_type is not None:
            event_type = event_type.detach().cpu()
        if mask is None:
            mask = torch.ones_like(anomaly_score, dtype=torch.bool)
        else:
            mask = mask.detach().cpu().bool()

        flat_a = anomaly_score.reshape(-1)
        flat_u = uncertainty_score.reshape(-1)
        flat_m = mask.reshape(-1)
        flat_e = event_type.reshape(-1) if event_type is not None else None

        for idx in torch.nonzero(flat_m, as_tuple=False).flatten().tolist():
            item = {
                'anomaly_score': float(flat_a[idx]),
                'uncertainty_score': float(flat_u[idx])
            }
            if flat_e is not None:
                item['event_type'] = int(flat_e[idx])
            self.buffer.append(item)

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer.clear()


class DriftDecisionModule:
    """
    Sliding-window drift warning based on the rejected/uncertain event ratio.
    """
    def __init__(self, window_size=1000, drift_threshold=0.3):
        self.window_size = int(window_size)
        self.drift_threshold = float(drift_threshold)
        self.window = deque(maxlen=self.window_size)
        self.in_drift = False
        self.num_warnings = 0

    def update(self, reject_indicator, mask=None):
        reject_indicator = reject_indicator.detach().cpu().bool()
        if mask is not None:
            mask = mask.detach().cpu().bool()
            reject_indicator = reject_indicator[mask]
        else:
            reject_indicator = reject_indicator.reshape(-1)

        for value in reject_indicator.reshape(-1).tolist():
            self.window.append(int(value))

        ratio = self.ratio
        is_warning = len(self.window) > 0 and ratio > self.drift_threshold
        new_warning = is_warning and not self.in_drift
        if new_warning:
            self.num_warnings += 1
        self.in_drift = is_warning

        return {
            'reject_ratio': ratio,
            'is_warning': is_warning,
            'new_warning': new_warning,
            'num_warnings': self.num_warnings
        }

    @property
    def ratio(self):
        if len(self.window) == 0:
            return 0.0
        return float(sum(self.window) / len(self.window))


# --- 5. FlowMatchingTHP (With Adaptive Multi-task Weighting) ---
class FlowMatchingTHP(nn.Module):

    def __init__(self, num_types, config):
        super().__init__()
        self.config = config
        self.num_types = num_types
        self.normalize = config.normalize

        self.encoder = Encoder(
            num_types=num_types,
            d_model=config.d_model,
            d_inner=config.d_inner_hid,
            n_layers=config.n_layers,
            n_head=config.n_head,
            d_k=config.d_k,
            d_v=config.d_v,
            dropout=config.dropout,
            max_len=getattr(config, 'max_len', 5000),
            use_pos_enc=getattr(config, 'use_pos_enc', False),
            use_time_gap=getattr(config, 'use_time_gap', True),
        )

        # Feature Decoupling Projectors
        # 1. Type Projector: Mapping to GMM centers
        self.type_projector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )
        
        # 2. Time Projector: Mapping to Flow Condition
        self.time_projector = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, config.d_model)
        )

        type_head = getattr(config, 'type_head', 'gmm')
        self.type_head_name = type_head
        self.mark_context_head = None
        if type_head == 'linear':
            self.type_predictor = LinearClassifierHead(config.d_model, num_types)
        elif type_head == 'hybrid':
            self.type_predictor = HybridClassifierHead(config.d_model, num_types)
        elif type_head == 'markov':
            self.type_predictor = None
            self.mark_context_head = CausalMarkContextHead(
                config.d_model,
                num_types,
                dropout=config.dropout
            )
        elif type_head == 'hybrid_markov':
            self.type_predictor = HybridClassifierHead(config.d_model, num_types)
            self.mark_context_head = CausalMarkContextHead(
                config.d_model,
                num_types,
                dropout=config.dropout
            )
        else:
            self.type_predictor = GMMClassifierHead(config.d_model, num_types)
        self.type_loss_func = nn.CrossEntropyLoss(ignore_index=-1)

        self.type_fusion = GatedFusion(config.d_model, dropout=config.dropout)

        self.x_dim = 1
        self.v_field = VectorField_Optimized(
            d_in=self.x_dim,
            d_cond=config.d_model,
            d_hid=config.d_inner_hid,
            d_out=self.x_dim,
            num_blocks=4
        )

        self.flow_matcher = ConditionalFlowMatcher(sigma=config.fm_sigma)
        self.fm_loss_func = FlowMatchingLoss()

        self.mean_log_data = 0.0
        self.var_log_data = 1.0
        self.mean_data = 1.0

        # =====================================================================
        # Adaptive Homoscedastic Uncertainty Weighting Parameters
        # =====================================================================
        self.log_var_fm = nn.Parameter(torch.zeros(1))
        self.log_var_type = nn.Parameter(torch.zeros(1))

    def _compute_type_logits(self, event_type, c_type):
        type_logits = None
        if self.type_predictor is not None:
            type_logits = self.type_predictor(c_type)

        if self.mark_context_head is not None:
            event_emb = self.encoder.event_type_emb(event_type)
            mark_logits = self.mark_context_head(event_emb)
            type_logits = mark_logits if type_logits is None else type_logits + mark_logits

        return type_logits

    def _build_flow_condition(self, c_time, target_type_idx):
        if getattr(self.config, 'disable_mark_conditioned_flow', False):
            c_cond = c_time
            clip_value = float(getattr(self.config, 'flow_cond_clip', 0.0))
            if clip_value > 0:
                c_cond = torch.clamp(c_cond, min=-clip_value, max=clip_value)
            return c_cond

        target_type_emb = self.encoder.event_type_emb(target_type_idx)
        c_cond = self.type_fusion(c_time, target_type_emb)

        clip_value = float(getattr(self.config, 'flow_cond_clip', 0.0))
        if clip_value > 0:
            c_cond = torch.clamp(c_cond, min=-clip_value, max=clip_value)

        return c_cond

    def _apply_context_masking(self, event_type, event_time, time_gap_norm):
        mask_prob = float(getattr(self.config, 'context_mask_prob', 0.0))
        if (not self.training) or mask_prob <= 0.0:
            return event_type, event_time, time_gap_norm

        min_history = max(0, int(getattr(self.config, 'context_mask_min_history', 1)))
        max_positions = max(0, event_type.size(1) - 1)
        if max_positions <= min_history:
            return event_type, event_time, time_gap_norm

        valid = event_type.ne(Constants.PAD)
        position = torch.arange(event_type.size(1), device=event_type.device).unsqueeze(0)
        # The final event has no future target, so masking it does not regularize
        # next-event prediction. Keep it untouched and mask earlier history only.
        maskable = valid & (position >= min_history) & (position < event_type.size(1) - 1)
        random_mask = torch.rand_like(event_type.float()) < mask_prob
        mask = maskable & random_mask
        if not mask.any():
            return event_type, event_time, time_gap_norm

        context_type = event_type.clone()
        context_time = event_time.clone()
        context_gap = time_gap_norm.clone()
        context_type[mask] = Constants.PAD
        context_time[mask] = 0.0
        # time_gap_norm is one step shorter than event_type; position p in
        # event_type corresponds to gap p-1.
        gap_mask = mask[:, 1:]
        if gap_mask.numel() > 0:
            context_gap[gap_mask] = 0.0
        return context_type, context_time, context_gap

    def forward(self, event_type, event_time, time_gap_norm):
        context_type, context_time, context_gap = self._apply_context_masking(
            event_type,
            event_time,
            time_gap_norm
        )
        non_pad_mask = get_non_pad_mask(context_type)

        # 1. Shared Encoder History
        enc_output = self.encoder(context_type, context_time, non_pad_mask, context_gap)
        c = enc_output[:, :-1, :]

        # 2. Decouple Features
        c_type = self.type_projector(c)  
        c_time = self.time_projector(c)  

        # 3. Predict Type (using c_type)
        type_logits = self._compute_type_logits(context_type, c_type)

        # 4. Prepare Flow Condition (using c_time)
        target_type_idx = event_type[:, 1:]
        c_cond = self._build_flow_condition(c_time, target_type_idx)

        # 5. Flow Matching
        x_1 = time_gap_norm.unsqueeze(-1)
        x_0 = torch.randn_like(x_1) * self.config.fm_sigma 
        
        t = torch.rand(x_1.shape[0], x_1.shape[1], 1, device=x_1.device)
        x_t, u_t = self.flow_matcher.sample_conditional_path(x_0, x_1, t)

        v_pred = self.v_field(x_t, t, c_cond)

        prediction = {
            'v_pred': v_pred,
            'u_t': u_t,
            'type_logits': type_logits,
            'c_cond': c_cond,
            'x_1': x_1
        }

        return enc_output, prediction

    def compute_flow_diagnostics(self, prediction, event_type):
        v_pred = prediction['v_pred'].detach()
        u_t = prediction['u_t'].detach()
        c_cond = prediction.get('c_cond')
        x_1 = prediction.get('x_1')

        target_type = event_type[:, 1:]
        mask = (target_type != Constants.PAD)
        if mask.sum() == 0:
            return {
                'fm_mse': 0.0,
                'residual_max': 0.0,
                'v_abs_max': 0.0,
                'u_abs_max': 0.0,
                'x1_abs_max': 0.0,
                'c_norm_mean': 0.0,
                'c_abs_max': 0.0,
                'num_events': 0
            }

        residual = (v_pred - u_t).pow(2).mean(dim=-1)
        stats = {
            'fm_mse': float(residual[mask].mean().item()),
            'residual_max': float(residual[mask].max().item()),
            'v_abs_max': float(v_pred[mask].abs().max().item()),
            'u_abs_max': float(u_t[mask].abs().max().item()),
            'x1_abs_max': float(x_1.detach()[mask].abs().max().item()) if x_1 is not None else 0.0,
            'c_norm_mean': float(c_cond.detach()[mask].norm(dim=-1).mean().item()) if c_cond is not None else 0.0,
            'c_abs_max': float(c_cond.detach()[mask].abs().max().item()) if c_cond is not None else 0.0,
            'num_events': int(mask.sum().item())
        }
        return stats

    def compute_loss_diagnostic(self, prediction, event_type):
        v_pred = prediction['v_pred']
        u_t = prediction['u_t']
        type_logits = prediction['type_logits']

        target_type = event_type[:, 1:]
        mask = (target_type != Constants.PAD)

        mask_fm = mask.unsqueeze(-1).float()
        tail_weight = float(getattr(self.config, 'tail_fm_weight', 0.0))
        if tail_weight > 0:
            event_loss = (v_pred - u_t).pow(2).mean(dim=-1)
            tail_threshold = float(getattr(self.config, 'tail_fm_threshold', 1.0))
            tail_power = float(getattr(self.config, 'tail_fm_power', 2.0))
            tail_max = float(getattr(self.config, 'tail_fm_max', 20.0))
            x_1 = prediction.get('x_1')
            tail_signal = torch.relu(x_1.squeeze(-1).detach() - tail_threshold)
            tail_factor = 1.0 + tail_weight * tail_signal.pow(tail_power)
            tail_factor = torch.clamp(tail_factor, max=tail_max)
            fm_loss = (event_loss * tail_factor * mask.float()).sum() / mask.float().sum().clamp_min(1.0)
        else:
            fm_loss = self.fm_loss_func(v_pred, u_t, mask_fm)

        target_labels = target_type - 1
        target_labels[~mask] = -1

        type_loss = self.type_loss_func(
            type_logits.reshape(-1, self.num_types),
            target_labels.reshape(-1)
        )

        loss_weighting = getattr(self.config, 'loss_weighting', 'adaptive')
        if loss_weighting == 'fixed':
            lambda_type = getattr(self.config, 'loss_lambda', 1.0)
            fm_weight = getattr(self.config, 'fm_loss_weight', 1.0)
            total_loss = fm_weight * fm_loss + lambda_type * type_loss
        else:
            # Adaptive homoscedastic weighting is useful for reliability experiments,
            # but benchmark runs can switch to fixed weighting to preserve Acc tables.
            precision_fm = torch.exp(-self.log_var_fm)
            precision_type = torch.exp(-self.log_var_type)

            weighted_fm_loss = 0.5 * precision_fm * fm_loss + self.log_var_fm
            weighted_type_loss = precision_type * type_loss + self.log_var_type

            total_loss = weighted_fm_loss + weighted_type_loss

        return total_loss, fm_loss.item(), type_loss.item()

    def _encode_detection_context(self, event_type, event_time, time_gap_norm, target_type_idx=None):
        non_pad_mask = get_non_pad_mask(event_type)
        enc_output = self.encoder(event_type, event_time, non_pad_mask, time_gap_norm)
        c = enc_output[:, :-1, :]

        c_type = self.type_projector(c)
        c_time = self.time_projector(c)
        type_logits = self._compute_type_logits(event_type, c_type)

        if target_type_idx is None:
            target_type_idx = event_type[:, 1:]
        c_cond = self._build_flow_condition(c_time, target_type_idx)

        mask = (target_type_idx != Constants.PAD)
        return {
            'enc_output': enc_output,
            'c': c,
            'c_type': c_type,
            'c_time': c_time,
            'c_cond': c_cond,
            'type_logits': type_logits,
            'mask': mask
        }

    def get_type_nll_map(self, event_type, type_logits):
        target_type = event_type[:, 1:]
        mask = (target_type != Constants.PAD)
        target_labels = target_type - 1
        target_labels = target_labels.masked_fill(~mask, -1)

        type_nll = F.cross_entropy(
            type_logits.reshape(-1, self.num_types),
            target_labels.reshape(-1),
            reduction='none',
            ignore_index=-1
        ).view_as(target_labels).float()
        return type_nll, mask

    def get_type_entropy_map(self, type_logits):
        log_probs = F.log_softmax(type_logits, dim=-1)
        probs = log_probs.exp()
        return -(probs * log_probs).sum(dim=-1)

    def get_flow_uncertainty_map(self, event_type, event_time, time_gap_norm, n_mc=8, context=None):
        if context is None:
            context = self._encode_detection_context(event_type, event_time, time_gap_norm)

        x_1 = time_gap_norm.unsqueeze(-1)
        c_cond = context['c_cond']
        mask = context['mask']
        residuals = []
        energies = []

        for _ in range(max(1, int(n_mc))):
            x_0 = torch.randn_like(x_1) * self.config.fm_sigma
            s = torch.rand_like(x_1)
            x_s, u_s = self.flow_matcher.sample_conditional_path(x_0, x_1, s)
            v_pred = self.v_field(x_s, s, c_cond)
            residuals.append((v_pred - u_s).pow(2).mean(dim=-1))
            energies.append(u_s.pow(2).mean(dim=-1))

        fm_uncertainty = torch.stack(residuals, dim=0).mean(dim=0)
        path_energy = torch.stack(energies, dim=0).mean(dim=0)
        fm_uncertainty = fm_uncertainty.masked_fill(~mask, 0.0)
        path_energy = path_energy.masked_fill(~mask, 0.0)
        return fm_uncertainty, path_energy, mask

    def _time_nll_from_flat_condition(self, x1_flat, c_cond_flat):
        latent_flat = torch.zeros_like(x1_flat, dtype=torch.float32)
        if x1_flat.shape[0] == 0:
            return torch.zeros(x1_flat.shape[0], device=x1_flat.device), latent_flat

        def ode_func(t, states):
            x = states[0]
            with torch.enable_grad():
                x = x.requires_grad_(True)
                x_in = x.unsqueeze(1)
                t_in = t * torch.ones(x.shape[0], 1, 1, device=x.device)
                c_in = c_cond_flat.unsqueeze(1)

                v_out = self.v_field(x_in, t_in, c_in)
                v = v_out.squeeze(1)
                grad_v = torch.autograd.grad(v.sum(), x, create_graph=True)[0]
                divergence = grad_v.view(-1, 1)
            return v, divergence

        z_t0 = x1_flat
        delta_logp_t0 = torch.zeros(x1_flat.shape[0], 1, device=x1_flat.device)
        times = torch.tensor([1.0, 0.0], device=x1_flat.device)

        state_t = odeint(
            ode_func,
            (z_t0, delta_logp_t0),
            times,
            atol=1e-5,
            rtol=1e-5,
            method='dopri5'
        )

        z_0 = state_t[0][-1]
        delta_logp = state_t[1][-1]

        sigma_fm = self.config.fm_sigma
        log_p_z0 = -0.5 * math.log(2 * math.pi) - math.log(sigma_fm) - 0.5 * (z_0 / sigma_fm).pow(2)
        log_prob_norm = log_p_z0 + delta_logp
        nll_flat = -log_prob_norm.squeeze(-1)

        if self.config.normalize == 'log':
            data_mu = self.mean_log_data
            data_sigma = self.var_log_data
            log_t_val = (x1_flat * data_sigma + data_mu).squeeze(-1)
            if getattr(self.config, 'eval_time_scale', 'legacy') == 'physical':
                log_t_val = log_t_val + math.log(max(float(self.mean_data), 1e-12))
            nll_flat = nll_flat + log_t_val + math.log(data_sigma)
        elif self.config.normalize == 'normal':
            nll_flat = nll_flat + math.log(self.mean_data)

        return nll_flat.float(), z_0.detach().float()

    def get_per_event_time_nll(self, event_type, event_time, time_gap_norm, context=None, return_latent=False):
        if context is None:
            context = self._encode_detection_context(event_type, event_time, time_gap_norm)

        mask = context['mask']
        x1 = time_gap_norm.unsqueeze(-1)
        x1_flat = x1[mask]
        c_cond_flat = context['c_cond'][mask]

        nll_map = torch.zeros_like(time_gap_norm, dtype=torch.float32)
        latent_map = torch.zeros_like(x1, dtype=torch.float32)
        if x1_flat.shape[0] == 0:
            if return_latent:
                return nll_map, mask, latent_map
            return nll_map, mask

        nll_flat, z_0 = self._time_nll_from_flat_condition(x1_flat, c_cond_flat)
        nll_map[mask] = nll_flat.float()
        latent_map[mask] = z_0.detach().float()
        if return_latent:
            return nll_map, mask, latent_map
        return nll_map, mask

    def compute_counterfactual_context_support(
            self,
            event_type,
            event_time,
            time_gap_norm,
            candidate_mask,
            base_type_nll,
            base_time_nll,
            k=3,
            epsilon=0.0,
            chunk_size=128,
            time_mode='exact'):
        """
        Estimate whether recent history events increase the conditional likelihood
        of each candidate event. For a target event i and a selected history event
        j, the contribution is Delta_{j->i}=NLL(e_i|H^{-j})-NLL(e_i|H).

        The implementation masks one recent history position at a time with PAD,
        preserving sequence length. This is an inference-time diagnostic used by
        RQ4; it is not used for training.
        """
        target_mask = candidate_mask & (event_type[:, 1:] != Constants.PAD)
        shape = candidate_mask.shape
        device = event_type.device
        type_count = torch.zeros(shape, device=device, dtype=torch.float32)
        time_count = torch.zeros(shape, device=device, dtype=torch.float32)
        valid_count = torch.zeros(shape, device=device, dtype=torch.float32)
        type_strength = torch.zeros(shape, device=device, dtype=torch.float32)
        time_strength = torch.zeros(shape, device=device, dtype=torch.float32)

        indices = torch.nonzero(target_mask, as_tuple=False)
        if indices.numel() == 0:
            return {
                'type_support_ratio': type_count,
                'time_support_ratio': time_count,
                'type_support_strength': type_strength,
                'time_support_strength': time_strength,
                'valid_support_count': valid_count,
            }

        k = max(1, int(k))
        chunk_size = max(1, int(chunk_size))
        epsilon = float(epsilon)
        time_mode = str(time_mode)

        for offset in range(k):
            rows = indices[:, 0]
            target_cols = indices[:, 1]
            history_pos = target_cols - offset
            valid = history_pos >= 0
            if valid.any():
                valid = valid & (event_type[rows, history_pos.clamp_min(0)] != Constants.PAD)
            if not valid.any():
                continue

            rows = rows[valid]
            target_cols = target_cols[valid]
            history_pos = history_pos[valid]

            for start in range(0, rows.numel(), chunk_size):
                end = min(start + chunk_size, rows.numel())
                row_chunk = rows[start:end]
                col_chunk = target_cols[start:end]
                hist_chunk = history_pos[start:end]
                local_n = row_chunk.numel()
                arange = torch.arange(local_n, device=device)

                cf_event_type = event_type[row_chunk].clone()
                cf_event_time = event_time[row_chunk].clone()
                cf_time_gap = time_gap_norm[row_chunk].clone()
                cf_event_type[arange, hist_chunk] = Constants.PAD
                cf_event_time[arange, hist_chunk] = 0.0
                cf_time_gap[arange, hist_chunk] = 0.0

                context = self._encode_detection_context(cf_event_type, cf_event_time, cf_time_gap)
                target_type = event_type[row_chunk, col_chunk + 1]
                target_labels = (target_type - 1).long()
                logits = context['type_logits'][arange, col_chunk]
                cf_type_nll = F.cross_entropy(logits, target_labels, reduction='none')
                delta_type = cf_type_nll - base_type_nll[row_chunk, col_chunk]

                if time_mode == 'off':
                    delta_time = torch.zeros_like(delta_type)
                else:
                    x1 = time_gap_norm[row_chunk, col_chunk].unsqueeze(-1)
                    c_cond = context['c_cond'][arange, col_chunk]
                    cf_time_nll, _ = self._time_nll_from_flat_condition(x1, c_cond)
                    delta_time = cf_time_nll - base_time_nll[row_chunk, col_chunk]

                type_count[row_chunk, col_chunk] += (delta_type > epsilon).float()
                time_count[row_chunk, col_chunk] += (delta_time > epsilon).float()
                type_strength[row_chunk, col_chunk] += torch.clamp(delta_type, min=0.0)
                time_strength[row_chunk, col_chunk] += torch.clamp(delta_time, min=0.0)
                valid_count[row_chunk, col_chunk] += 1.0

        denom = valid_count.clamp_min(1.0)
        return {
            'type_support_ratio': type_count / denom,
            'time_support_ratio': time_count / denom,
            'type_support_strength': type_strength / denom,
            'time_support_strength': time_strength / denom,
            'valid_support_count': valid_count,
        }

    def sample_time_from_latent(self, z0, c_cond, method='euler', step_size=0.05):
        """
        Push base Gaussian latents through the learned conditional vector field.

        z0: (N, 1, 1)
        c_cond: (N, 1, D)
        """
        times = torch.tensor([0.0, 1.0], device=z0.device)

        def ode_func(t, x):
            t_in = t * torch.ones_like(x)
            return self.v_field(x, t_in, c_cond)

        ode_opts = {"step_size": step_size} if step_size is not None else {}
        sol = odeint(
            ode_func,
            z0,
            times,
            method=method,
            options=ode_opts,
            atol=1e-5,
            rtol=1e-5
        )
        return sol[-1]

    def _build_correction_feature(self, context, gaussian_latent):
        c_type = F.normalize(context['c_type'].detach(), dim=-1)
        c_time = F.normalize(context['c_time'].detach(), dim=-1)
        if gaussian_latent is None:
            gaussian_latent = torch.zeros(
                c_type.size(0),
                c_type.size(1),
                1,
                device=c_type.device,
                dtype=c_type.dtype
            )
        else:
            gaussian_latent = gaussian_latent.detach().float()
        return torch.cat([c_type, c_time, gaussian_latent], dim=-1)

    def compute_reliability_scores(
            self,
            event_type,
            event_time,
            time_gap_norm,
            uncertainty_mc=8,
            type_entropy_weight=0.0,
            exact_time_nll=True,
            return_memory_features=False):
        context = self._encode_detection_context(event_type, event_time, time_gap_norm)
        type_nll, mask = self.get_type_nll_map(event_type, context['type_logits'])
        type_entropy = self.get_type_entropy_map(context['type_logits']).masked_fill(~mask, 0.0)
        fm_uncertainty, path_energy, _ = self.get_flow_uncertainty_map(
            event_type,
            event_time,
            time_gap_norm,
            n_mc=uncertainty_mc,
            context=context
        )

        gaussian_latent = None
        if exact_time_nll:
            if return_memory_features:
                time_nll, _, gaussian_latent = self.get_per_event_time_nll(
                    event_type,
                    event_time,
                    time_gap_norm,
                    context=context,
                    return_latent=True
                )
            else:
                time_nll, _ = self.get_per_event_time_nll(
                    event_type,
                    event_time,
                    time_gap_norm,
                    context=context
                )
        else:
            time_nll = fm_uncertainty.detach()
            if return_memory_features:
                # Fallback proxy when exact CNF inversion is disabled.
                gaussian_latent = time_gap_norm.unsqueeze(-1).detach()

        anomaly_score = (type_nll + time_nll).masked_fill(~mask, 0.0)
        uncertainty_score = (fm_uncertainty + type_entropy_weight * type_entropy).masked_fill(~mask, 0.0)

        result = {
            'anomaly_score': anomaly_score,
            'uncertainty_score': uncertainty_score,
            'flow_uncertainty': fm_uncertainty,
            'type_entropy': type_entropy,
            'path_energy': path_energy,
            'time_nll': time_nll.masked_fill(~mask, 0.0),
            'type_nll': type_nll.masked_fill(~mask, 0.0),
            'mask': mask
        }

        if return_memory_features:
            result.update({
                'gaussian_latent': gaussian_latent.masked_fill(~mask.unsqueeze(-1), 0.0),
                'memory_feature': self._build_correction_feature(context, gaussian_latent),
                'flow_condition': context['c_cond'].detach()
            })

        return result

    @staticmethod
    def selective_decision(anomaly_score, uncertainty_score, gamma, delta, mask=None):
        gamma = torch.as_tensor(gamma, device=anomaly_score.device, dtype=anomaly_score.dtype)
        delta = torch.as_tensor(delta, device=uncertainty_score.device, dtype=uncertainty_score.dtype)

        decision = torch.where(
            uncertainty_score > delta,
            torch.full_like(anomaly_score, REJECT_DECISION, dtype=torch.long),
            torch.where(
                anomaly_score > gamma,
                torch.full_like(anomaly_score, ANOMALY_DECISION, dtype=torch.long),
                torch.full_like(anomaly_score, NORMAL_DECISION, dtype=torch.long)
            )
        )
        if mask is not None:
            decision = decision.masked_fill(~mask, PAD_DECISION)
        return decision

    def configure_lightweight_adaptation(self, train_vector_tail_layers=1, train_adain=True):
        """
        Freeze semantic history modeling and expose a low-risk parameter subset for
        online adaptation to timing drift.
        """
        for param in self.parameters():
            param.requires_grad = False

        trainable_modules = [self.time_projector, self.type_fusion]
        for module in trainable_modules:
            for param in module.parameters():
                param.requires_grad = True

        tail_layers = max(0, int(train_vector_tail_layers))
        if tail_layers > 0:
            for block in self.v_field.blocks[-tail_layers:]:
                if train_adain:
                    for param in block.scale_shift_gen.parameters():
                        param.requires_grad = True
                else:
                    for param in block.parameters():
                        param.requires_grad = True

        for param in self.v_field.final_output.parameters():
            param.requires_grad = True

        return [param for param in self.parameters() if param.requires_grad]

    def build_lightweight_optimizer(
            self,
            lr=1e-4,
            weight_decay=0.0,
            train_vector_tail_layers=1,
            train_adain=True):
        params = self.configure_lightweight_adaptation(
            train_vector_tail_layers=train_vector_tail_layers,
            train_adain=train_adain
        )
        return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    def compute_time_adaptation_loss(self, event_type, event_time, time_gap_norm, adapt_mask=None):
        """
        FM-only loss for confirmed drift/new-normal data. This is intended to be used
        after quarantine confirmation, with semantic parameters frozen.
        """
        context = self._encode_detection_context(event_type, event_time, time_gap_norm)
        x_1 = time_gap_norm.unsqueeze(-1)
        x_0 = torch.randn_like(x_1) * self.config.fm_sigma
        s = torch.rand_like(x_1)
        x_s, u_s = self.flow_matcher.sample_conditional_path(x_0, x_1, s)
        v_pred = self.v_field(x_s, s, context['c_cond'])

        event_loss = (v_pred - u_s).pow(2).mean(dim=-1)
        mask = context['mask']
        if adapt_mask is not None:
            mask = mask & adapt_mask.bool().to(mask.device)

        if mask.sum() == 0:
            return event_loss.sum() * 0.0
        return event_loss[mask].mean()
    
    def get_exact_log_likelihood(self, event_type, event_time, time_gap_norm):
        """
        Calculate Exact Log-Likelihood for Flow Matching.
        """
        nll_map, mask = self.get_per_event_time_nll(event_type, event_time, time_gap_norm)
        return nll_map[mask].sum(), int(mask.sum().item())

    def denormalize_time(self, time_norm):
        if self.normalize == 'log':
            log_arg = torch.clamp(
                time_norm * self.var_log_data + self.mean_log_data,
                min=-80.0,
                max=80.0
            )
            time_value = torch.exp(log_arg)
            if getattr(self.config, 'eval_time_scale', 'legacy') == 'physical':
                time_value = time_value * self.mean_data
            return time_value
        elif self.normalize == 'normal':
            return time_norm * self.mean_data
        else:
            return time_norm
