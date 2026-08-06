# -*- coding: utf-8 -*-
import math
import torch
import torch.nn as nn
from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

def get_attn_key_pad_mask(seq_k, seq_q):
    len_q = seq_q.size(1)
    padding_mask = seq_k.eq(0)
    padding_mask = padding_mask.unsqueeze(1).expand(-1, len_q, -1)
    return padding_mask

def get_non_pad_mask(seq):
    return seq.ne(0).unsqueeze(-1)

def get_subsequent_mask(seq):
    sz_b, len_s = seq.size()
    subsequent_mask = torch.triu(
        torch.ones((len_s, len_s), device=seq.device, dtype=torch.uint8), diagonal=1)
    subsequent_mask = subsequent_mask.unsqueeze(0).expand(sz_b, -1, -1)
    return subsequent_mask

class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_inner, n_head, d_k, d_v, dropout=0.1):
        super(EncoderLayer, self).__init__()
        self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, slf_attn_mask=None):
        enc_output, enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        enc_output = self.pos_ffn(enc_output)
        if non_pad_mask is not None:
            enc_output *= non_pad_mask
        return enc_output, enc_slf_attn

class Encoder(nn.Module):
    def __init__(
            self,
            num_types,
            d_model,
            n_layers,
            n_head,
            d_k,
            d_v,
            d_inner,
            dropout=0.1,
            max_len=5000,
            use_pos_enc=False,
            use_time_gap=True):
        super().__init__()
        self.d_model = d_model
        self.use_pos_enc = use_pos_enc
        self.use_time_gap = use_time_gap

        # 銆愭柟妗?淇敼銆戯細绉婚櫎缁濆鏃堕棿鐨?Positional encoding (Absolute Time) 
        # 浠庤€屼繚璇佹ā鍨嬬殑 Temporal Stationarity
        
        # Event type embedding
        self.event_type_emb = nn.Embedding(num_types + 1, d_model, padding_idx=0)
        self.position_emb = nn.Embedding(max_len + 1, d_model, padding_idx=0)
        
        # Explicit Time Gap Projection (Relative Time)
        self.time_gap_proj = nn.Linear(1, d_model)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, event_type, event_time, non_pad_mask, time_gap_norm=None):
        # Prepare masks
        slf_attn_mask_subseq = get_subsequent_mask(event_type)
        slf_attn_mask_keypad = get_attn_key_pad_mask(seq_k=event_type, seq_q=event_type)
        slf_attn_mask = (slf_attn_mask_keypad + slf_attn_mask_subseq).gt(0)

        # 1. 鍩虹 Embeddings (浠呬繚鐣?Type锛屾姏寮冪粷瀵规椂闂?
        type_enc = self.event_type_emb(event_type)
        enc_output = type_enc

        if self.use_pos_enc:
            positions = torch.arange(
                1,
                event_type.size(1) + 1,
                device=event_type.device
            ).unsqueeze(0).expand_as(event_type)
            positions = positions.masked_fill(event_type.eq(0), 0)
            positions = positions.clamp(max=self.position_emb.num_embeddings - 1)
            enc_output = enc_output + self.position_emb(positions)

        # 2. 铻嶅悎 Relative Time Gap (鏃堕棿骞崇ǔ鎬х殑鏍稿績鏉ユ簮)
        if self.use_time_gap and time_gap_norm is not None:
            # 鑷姩 Padding 澶勭悊 (瑙ｅ喅 L vs L-1 闂)
            if time_gap_norm.size(1) != event_type.size(1):
                diff = event_type.size(1) - time_gap_norm.size(1)
                if diff > 0:
                    zeros = torch.zeros(time_gap_norm.size(0), diff, device=time_gap_norm.device)
                    time_gap_norm = torch.cat([zeros, time_gap_norm], dim=1)
            
            gap_emb = self.time_gap_proj(time_gap_norm.unsqueeze(-1))
            enc_output = enc_output + gap_emb

        # 3. Transformer Layers
        for enc_layer in self.layer_stack:
            enc_output, _ = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)

        enc_output = self.layer_norm(enc_output)
        return enc_output

class Predictor(nn.Module):
    def __init__(self, d_model, num_types):
        super(Predictor, self).__init__()
        self.linear = nn.Linear(d_model, num_types, bias=False)
        nn.init.xavier_normal_(self.linear.weight)

    def forward(self, data, non_pad_mask):
        out = self.linear(data)
        out = out * non_pad_mask
        return out
