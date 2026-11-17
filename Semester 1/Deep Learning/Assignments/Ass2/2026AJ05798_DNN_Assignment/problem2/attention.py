"""Scaled dot-product and multi-head attention (no nn.MultiheadAttention)."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """
    q,k,v: (B, H, T_q, D) / (B, H, T_k, D) / (B, H, T_k, D)
    mask: broadcastable; positions with True are masked (set to -inf before softmax).
  """
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask, float("-inf"))
    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    return out, weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        b, t, _ = x.shape
        q = self._split(self.w_q(x))
        k = self._split(self.w_k(x))
        v = self._split(self.w_v(x))
        out, w = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(1, 2).contiguous().view(b, t, self.d_model)
        return self.drop(self.w_o(out)), w

    def _split(self, x: Tensor) -> Tensor:
        b, t, _ = x.shape
        x = x.view(b, t, self.n_heads, self.d_head).transpose(1, 2)
        return x


def causal_mask(t: int, device: torch.device) -> Tensor:
    return torch.triu(torch.ones(t, t, device=device, dtype=torch.bool), diagonal=1)


def hand_check_attention_example() -> dict:
    """3-token instructor-style toy (d=2); document in notebook/report."""
    q = torch.tensor([[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]])
    k = q.clone()
    v = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
    q = q.unsqueeze(1)
    k = k.unsqueeze(1)
    v = v.unsqueeze(1)
    out, w = scaled_dot_product_attention(q, k, v, causal_mask(3, q.device))
    return {"out": out.squeeze(), "weights": w.squeeze()}