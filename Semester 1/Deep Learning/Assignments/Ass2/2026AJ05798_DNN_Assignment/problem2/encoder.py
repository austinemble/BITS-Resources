"""Transformer encoder block, positional encoding, HAR classifier."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from problem2.attention import MultiHeadAttention, causal_mask, scaled_dot_product_attention


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_len: int, d_model: int) -> None:
        super().__init__()
        self.emb = nn.Embedding(max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        idx = torch.arange(t, device=x.device).unsqueeze(0).expand(x.size(0), -1)
        return x + self.emb(idx)


class EncoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.mha = MultiHeadAttention(d_model, n_heads, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        h, w = self.mha(self.ln1(x), mask)
        self.last_attn = w
        x = x + h
        x = x + self.ff(self.ln2(x))
        return x


def encoder_block_param_count(d_model: int, d_ff: int) -> int:
    """Weight-only count 4*d^2 + 2*d*d_ff plus biases and LayerNorm."""
    mha = 4 * d_model * d_model + 4 * d_model  # Q,K,V,O biases
    ff = d_model * d_ff + d_ff + d_ff * d_model + d_model
    ln = 4 * d_model  # 2 LayerNorms × (weight+bias)
    return mha + ff + ln


class HAREncoderClassifier(nn.Module):
    def __init__(
        self,
        input_channels: int = 9,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        d_ff: int = 512,
        num_classes: int = 6,
        patch_size: int = 8,
        pos_type: str = "sinusoidal",
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        patch_dim = input_channels * patch_size
        self.proj = nn.Linear(patch_dim, d_model)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        self.pos_type = pos_type
        seq_len = 128 // patch_size + 1
        if pos_type == "sinusoidal":
            self.pos = SinusoidalPositionalEncoding(d_model)
        elif pos_type == "learned":
            self.pos = LearnedPositionalEncoding(seq_len, d_model)
        else:
            self.pos = None
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def _patchify(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        p = self.patch_size
        x = x[:, : (t // p) * p, :]
        x = x.view(b, -1, p * c)
        return self.proj(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tok = self._patchify(x)
        cls = self.cls.expand(x.size(0), -1, -1)
        x = torch.cat([cls, tok], dim=1)
        if self.pos is not None:
            x = self.pos(x)
        for blk in self.blocks:
            x = blk(x, mask=None)
        return self.head(self.ln(x[:, 0, :]))


class CausalTransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        n_heads: int,
        n_layers: int,
        d_ff: int,
        context: int,
        dropout: float = 0.1,
        use_causal_mask: bool = True,
    ) -> None:
        super().__init__()
        self.context = context
        self.use_causal_mask = use_causal_mask
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len=context + 1)
        self.blocks = nn.ModuleList(
            [EncoderBlock(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok_emb(idx)
        x = self.pos(x)
        mask = causal_mask(idx.size(1), idx.device) if self.use_causal_mask else None
        for blk in self.blocks:
            x = blk(x, mask)
        return self.head(self.ln(x))


# Re-export for attention tests vs F.scaled_dot_product_attention
__all__ = [
    "scaled_dot_product_attention",
    "MultiHeadAttention",
    "EncoderBlock",
    "HAREncoderClassifier",
    "CausalTransformerLM",
    "encoder_block_param_count",
]