"""Tiny Shakespeare character-level data.

Training windows are sampled on the fly. Building every overlapping window
(~1e6 rows) in a Python loop never finishes in time and writes no outputs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def download_shakespeare(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return path.read_text(encoding="utf-8")
    import ssl
    import urllib.request

    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(certifi.where())
    except ImportError:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as resp:
        text = resp.read().decode("utf-8")
    path.write_text(text, encoding="utf-8")
    return text


def build_char_dataset(
    text: str, train_frac: float = 0.9
) -> tuple[dict[str, int], dict[int, str], np.ndarray, np.ndarray]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for ch, i in stoi.items()}
    data = np.array([stoi[c] for c in text], dtype=np.int64)
    n = int(len(data) * train_frac)
    return stoi, itos, data[:n], data[n:]


def load_shakespeare(
    data_dir: Path | str = "data", train_frac: float = 0.9
) -> tuple[dict[str, int], dict[int, str], np.ndarray, np.ndarray]:
    text = download_shakespeare(Path(data_dir) / "tiny_shakespeare.txt")
    return build_char_dataset(text, train_frac)


def sample_batch(
    data: np.ndarray,
    context: int,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Random next-character windows. x,y shapes: (B, context)."""
    hi = len(data) - context - 1
    ix = rng.integers(0, hi, size=batch_size)
    x = np.stack([data[i : i + context] for i in ix])
    y = np.stack([data[i + 1 : i + 1 + context] for i in ix])
    return torch.from_numpy(x), torch.from_numpy(y)
