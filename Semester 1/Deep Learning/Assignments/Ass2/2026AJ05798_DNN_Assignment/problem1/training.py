"""Training loops for Problem 1."""

from __future__ import annotations

import time
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

from common.results_log import log_result


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iter_batches(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    ds = TensorDataset(torch.from_numpy(x), torch.from_numpy(y))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=torch.Generator().manual_seed(seed),
    )
    for xb, yb in loader:
        yield xb, yb


@torch.no_grad()
def eval_classifier(
    model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int
) -> tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    dev = _device()
    preds: list[int] = []
    ys: list[int] = []
    for xb, yb in iter_batches(x, y, batch_size, shuffle=False, seed=0):
        xb = xb.to(dev)
        logits = model(xb)
        preds.extend(logits.argmax(-1).cpu().tolist())
        ys.extend(yb.long().tolist())
    acc = accuracy_score(ys, preds)
    f1 = f1_score(ys, preds, average="macro")
    return float(acc), float(f1), np.array(ys), np.array(preds)


def train_har_classifier(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    model_name: str,
    problem: str = "P1",
) -> dict[str, float]:
    dev = _device()
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    times: list[float] = []

    for _ in range(epochs):
        model.train()
        t0 = time.perf_counter()
        for xb, yb in iter_batches(x_tr, y_tr, batch_size, shuffle=True, seed=seed):
            xb, yb = xb.to(dev), yb.to(dev).long()
            opt.zero_grad()
            loss = crit(model(xb), yb)
            loss.backward()
            opt.step()
        times.append(time.perf_counter() - t0)

    test_acc, macro_f1, y_true, y_pred = eval_classifier(model, x_te, y_te, batch_size)
    params = sum(p.numel() for p in model.parameters())
    row = {
        "problem": problem,
        "model": model_name,
        "seed": seed,
        "seq_len": 128,
        "test_acc": f"{test_acc:.6f}",
        "macro_f1": f"{macro_f1:.6f}",
        "test_mse": "",
        "params": params,
        "sec_per_epoch": f"{float(np.mean(times)):.4f}",
        "val_loss_nats": "",
        "bits_per_char": "",
        "notes": "HAR",
    }
    log_result(row)
    return {
        "test_acc": test_acc,
        "macro_f1": macro_f1,
        "sec_per_epoch": float(np.mean(times)),
        "params": params,
        "model": model,
        "y_true": y_true,
        "y_pred": y_pred,
    }


@torch.no_grad()
def eval_mse(model: nn.Module, x: np.ndarray, y: np.ndarray, batch_size: int) -> float:
    model.eval()
    dev = _device()
    se = []
    for xb, yb in iter_batches(x, y, batch_size, shuffle=False, seed=0):
        xb = xb.to(dev)
        pred = model(xb)
        se.append(((pred.cpu() - yb) ** 2).numpy())
    return float(np.concatenate(se).mean())


def train_adding(
    model: nn.Module,
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    x_te: np.ndarray,
    y_te: np.ndarray,
    *,
    T: int,
    epochs: int,
    batch_size: int,
    lr: float,
    seed: int,
    model_name: str,
    grad_clip: float | None = None,
    history: bool = False,
) -> dict:
    dev = _device()
    model = model.to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()
    losses: list[float] = []

    for _ in range(epochs):
        model.train()
        epoch_loss = []
        for xb, yb in iter_batches(x_tr, y_tr, batch_size, shuffle=True, seed=seed + T):
            xb, yb = xb.to(dev), yb.to(dev).float()
            opt.zero_grad()
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
            epoch_loss.append(loss.item())
        losses.append(float(np.mean(epoch_loss)))

    test_mse = eval_mse(model, x_te, y_te, batch_size)
    params = sum(p.numel() for p in model.parameters())
    log_result(
        {
            "problem": "P1",
            "model": model_name,
            "seed": seed,
            "seq_len": T,
            "test_acc": "",
            "macro_f1": "",
            "test_mse": f"{test_mse:.6f}",
            "params": params,
            "sec_per_epoch": "",
            "val_loss_nats": "",
            "bits_per_char": "",
            "notes": "adding" if grad_clip is None else f"adding_clip={grad_clip}",
        }
    )
    out = {"test_mse": test_mse, "params": params, "losses": losses, "model": model}
    if not history:
        del out["losses"]
    return out


def gradient_norms_over_time(
    model: nn.Module,
    x_sample: np.ndarray,
    y_sample: np.ndarray,
    trained: bool,
    T: int,
) -> np.ndarray:
    """Norm of ∂L/∂h_t at each timestep (single sequence, last hidden)."""
    dev = _device()
    model = model.to(dev)
    if trained:
        model.train()
    else:
        model.eval()
    x = torch.from_numpy(x_sample[:1]).to(dev)  # (1, T, d)
    y = torch.tensor([y_sample[0]], device=dev, dtype=torch.float32)
    h_states: list[torch.Tensor] = []

    h_dim = model.rnn.hidden_size
    if isinstance(model.rnn, nn.LSTM):
        h = torch.zeros(1, 1, h_dim, device=dev)
        c = torch.zeros(1, 1, h_dim, device=dev)
        for t in range(T):
            xt = x[:, t : t + 1, :]
            out, (h, c) = model.rnn(xt, (h, c))
            h_states.append(out[:, -1, :].squeeze(0))
    elif isinstance(model.rnn, (nn.GRU, nn.RNN)):
        h = torch.zeros(1, 1, h_dim, device=dev)
        for t in range(T):
            xt = x[:, t : t + 1, :]
            out, h = model.rnn(xt, h)
            h_states.append(out[:, -1, :].squeeze(0))
    else:
        raise TypeError(type(model.rnn))

    norms = []
    for h_t in h_states:
        h_t = h_t.unsqueeze(0)
        h_t.retain_grad()
        pred = model.head(h_t).squeeze(-1)
        loss = nn.functional.mse_loss(pred, y)
        model.zero_grad()
        loss.backward(retain_graph=True)
        g = h_t.grad
        norms.append(float(g.norm().item()) if g is not None else 0.0)
    return np.array(norms, dtype=np.float64)