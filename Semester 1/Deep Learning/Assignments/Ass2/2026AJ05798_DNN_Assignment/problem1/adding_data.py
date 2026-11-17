"""Synthetic adding problem (assignment Dataset 3)."""

from __future__ import annotations

import numpy as np


def make_adding(n: int, T: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.zeros((n, T, 2), dtype=np.float32)
    x[:, :, 0] = rng.random((n, T))
    i1 = rng.integers(0, T // 2, n)
    i2 = rng.integers(T // 2, T, n)
    x[np.arange(n), i1, 1] = 1
    x[np.arange(n), i2, 1] = 1
    y = x[np.arange(n), i1, 0] + x[np.arange(n), i2, 0]
    return x, y.astype(np.float32)


def validate_adding_batch(x: np.ndarray, y: np.ndarray, T: int) -> None:
    markers = x[:, :, 1]
    count = markers.sum(axis=1)
    assert np.all(count == 2), "each sequence must have exactly two markers"
    first = (markers[:, : T // 2] > 0).sum(axis=1)
    second = (markers[:, T // 2 :] > 0).sum(axis=1)
    assert np.all(first == 1) and np.all(second == 1), "one marker per half"
    n = x.shape[0]
    i1 = (markers[:, : T // 2] > 0).argmax(axis=1)
    i2 = (markers[:, T // 2 :] > 0).argmax(axis=1) + T // 2
    expected = x[np.arange(n), i1, 0] + x[np.arange(n), i2, 0]
    assert np.allclose(expected, y), "target must equal sum of marked values"


def load_adding_splits(T: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x_tr, y_tr = make_adding(10_000, T, 0)
    x_va, y_va = make_adding(1_000, T, 1)
    x_te, y_te = make_adding(1_000, T, 2)
    validate_adding_batch(x_tr, y_tr, T)
    validate_adding_batch(x_va, y_va, T)
    validate_adding_batch(x_te, y_te, T)
    return x_tr, y_tr, x_va, y_va, x_te, y_te