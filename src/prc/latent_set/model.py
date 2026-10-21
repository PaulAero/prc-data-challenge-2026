"""Joint window transformer.

Tokens = all movements of one airport window. Attention carries a learned additive bias per head
from three pair relations, so the encoder does not have to rediscover them from raw features:
- signed time difference between the two events (bucketed), per (type_i, type_j) pair;
- same runway, per pair type;
- same stand, with the signed stand-time difference (take-off for departures, in-block for
  arrivals), per pair type. This is how "an arrival blocked in at my stand before I took off"
  reaches the departure token.
Departure tokens get a bottleneck embedding z and heads: mean, q10/q50/q90, P(fill).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

DT_EDGES_MIN = [1, 2, 3, 5, 7, 10, 15, 20, 30, 45, 60, 90, 120]
N_DT = 2 * (len(DT_EDGES_MIN) + 1)
HEADS_OUT = ["mean", "q10", "q50", "q90", "fill_logit"]
QUANTILES = (0.1, 0.5, 0.9)


# Minutes are computed as seconds * float32(1/60), not seconds / 60. CUDA divides by a Python scalar
# through that reciprocal multiply, CPU divides exactly, and the two disagree on exact multiples of
# 60 s that sit on a bucket edge (common: several airports log minute timestamps). The models were
# trained on CUDA, so this form reproduces their training semantics bit-for-bit on every device.
_INV60 = torch.tensor(1.0, dtype=torch.float32) / torch.tensor(60.0, dtype=torch.float32)


def dt_bucket(dt_sec: torch.Tensor, edges: torch.Tensor) -> torch.Tensor:
    b = torch.bucketize(dt_sec.abs() * _INV60.to(dt_sec.device), edges)
    return b + (dt_sec < 0).long() * (len(edges) + 1)


class Block(nn.Module):
    def __init__(self, d: int, h: int, ff: int, drop: float):
        super().__init__()
        self.h = h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(drop), nn.Linear(ff, d))
        self.drop = nn.Dropout(drop)
        self.p = drop

    def forward(self, x: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=bias.to(q.dtype),
                                           dropout_p=self.p if self.training else 0.0)
        x = x + self.drop(self.out(a.transpose(1, 2).reshape(B, L, D)))
        return x + self.drop(self.ff(self.ln2(x)))


class WindowEncoder(nn.Module):
    def __init__(self, n_num: int, vocab_sizes: list[int], d: int = 128, heads: int = 8, layers: int = 4,
                 ff: int = 256, z_dim: int = 16, drop: float = 0.1, context: bool = True,
                 stand_bias: bool = True, runway_bias: bool = True, n_aux: int = 0):
        super().__init__()
        self.context = context
        self.stand_bias, self.runway_bias = stand_bias, runway_bias
        self.heads = heads
        self.cat_emb = nn.ModuleList([nn.Embedding(v + 1, d) for v in vocab_sizes])
        self.type_emb = nn.Embedding(2, d)
        self.num_proj = nn.Sequential(nn.Linear(n_num + 1, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, ff, drop) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.register_buffer("edges", torch.tensor(DT_EDGES_MIN, dtype=torch.float32), persistent=False)
        self.b_dt = nn.Embedding(4 * N_DT, heads)
        self.b_rw = nn.Embedding(8, heads)
        self.b_st = nn.Embedding(4 * N_DT + 1, heads)
        for e in (self.b_dt, self.b_rw, self.b_st):
            nn.init.zeros_(e.weight)
        self.to_z = nn.Linear(d, z_dim)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(z_dim, 64), nn.GELU(), nn.Linear(64, len(HEADS_OUT)))
        # EXP-047: training-only head predicting the surface state from the same bottleneck z
        self.aux_head = (nn.Sequential(nn.GELU(), nn.Linear(z_dim, 64), nn.GELU(), nn.Linear(64, n_aux))
                         if n_aux > 0 else None)

    def pair_bias(self, is_dep, t_ref, t_stand, rw, st, pad):
        B, L = is_dep.shape
        ptype = (is_dep.long()[:, :, None] * 2 + is_dep.long()[:, None, :])  # 0..3
        dt = (t_ref[:, None, :] - t_ref[:, :, None]).float()
        bias = self.b_dt(ptype * N_DT + dt_bucket(dt, self.edges))
        if self.runway_bias:
            same_rw = ((rw[:, :, None] == rw[:, None, :]) & (rw[:, :, None] > 0)).long()
            bias = bias + self.b_rw(ptype * 2 + same_rw)
        if self.stand_bias:
            same_st = (st[:, :, None] == st[:, None, :]) & (st[:, :, None] > 0)
            dst = (t_stand[:, None, :] - t_stand[:, :, None]).float()
            st_idx = torch.where(same_st, 1 + ptype * N_DT + dt_bucket(dst, self.edges), torch.zeros_like(ptype))
            bias = bias + self.b_st(st_idx)
        bias = bias.permute(0, 3, 1, 2)  # B,H,L,L
        neg = torch.finfo(torch.float32).min / 2
        bias = bias.masked_fill(pad[:, None, None, :], neg)
        if not self.context:
            eye = torch.eye(L, dtype=torch.bool, device=bias.device)
            bias = bias.masked_fill(~eye[None, None], neg)
        return bias

    def forward(self, num, cats, is_dep, t_rel, t_ref, t_stand, rw, st, pad):
        """num [B,L,n_num] float, cats [B,L,C] long, is_dep [B,L] bool, t_rel [B,L] float (hours from
        window centre), t_ref/t_stand [B,L] int64 epoch s (relative is fine), rw/st [B,L] int64 pair
        ids (0 = missing), pad [B,L] bool."""
        x = torch.cat([num.clamp(-8, 8), t_rel[..., None]], -1)
        h = self.num_proj(x) + self.type_emb(is_dep.long())
        for k, emb in enumerate(self.cat_emb):
            h = h + emb(cats[..., k])
        bias = self.pair_bias(is_dep, t_ref, t_stand, rw, st, pad)
        for blk in self.blocks:
            h = blk(h, bias)
        z = self.to_z(self.ln(h))
        return self.head(z), z


def loss_fn(out: torch.Tensor, y: torch.Tensor, fill: torch.Tensor, w_reg: torch.Tensor, w_fill: torch.Tensor,
            fill_weight: float = 0.2, huber: float = 0.0) -> torch.Tensor:
    """out [N,5], y scaled target, weights zero where a term does not apply. huber > 0 (scaled units):
    Huber loss for the mean head instead of squared error (EXP-045), x2 so both agree near zero."""
    if huber > 0:
        mse = 2.0 * F.huber_loss(out[:, 0], y, reduction="none", delta=huber)
    else:
        mse = (out[:, 0] - y) ** 2
    pin = 0.0
    for j, q in enumerate(QUANTILES, start=1):
        e = y - out[:, j]
        pin = pin + torch.maximum(q * e, (q - 1) * e)
    bce = F.binary_cross_entropy_with_logits(out[:, 4], fill, reduction="none")
    reg = ((mse + pin) * w_reg).sum() / w_reg.sum().clamp(min=1.0)
    clf = (bce * w_fill).sum() / w_fill.sum().clamp(min=1.0)
    return reg + fill_weight * clf
