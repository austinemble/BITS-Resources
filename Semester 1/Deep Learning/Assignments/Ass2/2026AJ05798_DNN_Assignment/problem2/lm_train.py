"""Train / evaluate character LM and sample text."""

from __future__ import annotations

import math
import time

import numpy as np
import torch
import torch.nn as nn

from common.config import P2_LM_BATCH, P2_LM_STEPS, P2_VAL_BATCHES
from common.results_log import log_result
from problem2.encoder import CausalTransformerLM
from problem2.text_data import sample_batch


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def eval_nats(
    model: CausalTransformerLM,
    data: np.ndarray,
    n_batches: int,
    batch_size: int,
    seed: int,
) -> float:
    model.eval()
    dev = _device()
    rng = np.random.default_rng(seed)
    total, count = 0.0, 0
    crit = nn.CrossEntropyLoss(reduction="sum")
    for _ in range(n_batches):
        xb, yb = sample_batch(data, model.context, batch_size, rng)
        xb, yb = xb.to(dev), yb.to(dev)
        logits = model(xb)
        loss = crit(logits.view(-1, logits.size(-1)), yb.view(-1))
        total += loss.item()
        count += yb.numel()
    return total / max(count, 1)


def train_lm(
    model: CausalTransformerLM,
    train_data: np.ndarray,
    val_data: np.ndarray,
    epochs: int,
    lr: float,
    seed: int,
    notes: str = "shakespeare",
    steps_per_epoch: int | None = None,
) -> dict:
    dev = _device()
    model = model.to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    steps = steps_per_epoch or P2_LM_STEPS
    rng = np.random.default_rng(seed)
    history = []
    print(f"LM start notes={notes} epochs={epochs} steps/epoch={steps} device={dev}", flush=True)

    for epoch in range(epochs):
        model.train()
        t0 = time.perf_counter()
        running = 0.0
        for _ in range(steps):
            xb, yb = sample_batch(train_data, model.context, P2_LM_BATCH, rng)
            xb, yb = xb.to(dev), yb.to(dev)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits.view(-1, logits.size(-1)), yb.view(-1))
            loss.backward()
            opt.step()
            running += loss.item()
        val_nats = eval_nats(model, val_data, P2_VAL_BATCHES, P2_LM_BATCH, seed + epoch)
        history.append(val_nats)
        print(
            f"  epoch {epoch + 1}/{epochs} train={running / steps:.3f} val={val_nats:.3f} "
            f"sec={time.perf_counter() - t0:.1f}",
            flush=True,
        )

    bits = val_nats / math.log(2)
    params = sum(p.numel() for p in model.parameters())
    log_result(
        {
            "problem": "P2",
            "model": "transformer_lm",
            "seed": seed,
            "seq_len": model.context,
            "test_acc": "",
            "macro_f1": "",
            "test_mse": "",
            "params": params,
            "sec_per_epoch": "",
            "val_loss_nats": f"{val_nats:.6f}",
            "bits_per_char": f"{bits:.6f}",
            "notes": notes,
        }
    )
    return {"val_nats": val_nats, "bits_per_char": bits, "history": history, "model": model}


@torch.no_grad()
def generate(
    model: CausalTransformerLM,
    itos: dict[int, str],
    prompt: str,
    stoi: dict[str, int],
    n_chars: int = 500,
    temperature: float = 1.0,
    top_k: int = 10,
) -> str:
    model.eval()
    dev = _device()
    idx = [stoi[c] for c in prompt if c in stoi]
    if not idx:
        idx = [0]
    for _ in range(n_chars):
        x = torch.tensor([idx[-model.context :]], device=dev)
        logits = model(x)[:, -1, :] / max(temperature, 1e-6)
        if top_k > 0:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, 1).item()
        idx.append(next_id)
    return "".join(itos[i] for i in idx)


def measure_context_cost(
    vocab_size: int,
    d_model: int,
    n_heads: int,
    n_layers: int,
    d_ff: int,
    contexts: list[int],
) -> list[dict]:
    dev = _device()
    rows = []
    for ctx in contexts:
        model = CausalTransformerLM(
            vocab_size, d_model, n_heads, n_layers, d_ff, ctx
        ).to(dev)
        x = torch.randint(0, vocab_size, (4, ctx), device=dev)
        if dev.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            _ = model(x)
            torch.cuda.synchronize()
            ms = (time.perf_counter() - t0) * 1000
            mem = torch.cuda.max_memory_allocated() / (1024**2)
        else:
            t0 = time.perf_counter()
            _ = model(x)
            ms = (time.perf_counter() - t0) * 1000
            # No CUDA here. The n^2 term is the score and weight matrices, one pair per layer.
            attn = model.blocks[-1].last_attn
            per_layer = attn.numel() * attn.element_size() * 2
            mem = per_layer * len(model.blocks) / (1024**2)
        rows.append({"context": ctx, "ms_forward": ms, "peak_mb": mem})
    return rows