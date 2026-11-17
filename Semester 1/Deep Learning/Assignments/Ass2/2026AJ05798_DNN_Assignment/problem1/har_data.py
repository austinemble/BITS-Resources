"""UCI HAR dataset loading, subject-wise splits, and normalization."""

from __future__ import annotations

import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

CHANNEL_FILES = [
    "body_acc_x_train.txt",
    "body_acc_y_train.txt",
    "body_acc_z_train.txt",
    "body_gyro_x_train.txt",
    "body_gyro_y_train.txt",
    "body_gyro_z_train.txt",
    "total_acc_x_train.txt",
    "total_acc_y_train.txt",
    "total_acc_z_train.txt",
]
CHANNEL_FILES_TEST = [f.replace("_train", "_test") for f in CHANNEL_FILES]


@dataclass
class HARData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    mean: np.ndarray
    std: np.ndarray
    train_subjects: np.ndarray
    val_subjects: np.ndarray


def har_root(data_dir: Path) -> Path:
    inner = data_dir / "UCI HAR Dataset"
    if inner.exists():
        return inner
    nested = data_dir / "UCI HAR Dataset.zip"
    if nested.exists():
        with zipfile.ZipFile(nested) as zf:
            zf.extractall(data_dir)
    return inner


def download_har(data_dir: Path) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    root = har_root(data_dir)
    if (root / "train" / "Inertial Signals").exists():
        return root
    import ssl
    import urllib.request

    url = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
    zip_path = data_dir / "har.zip"
    ctx = ssl.create_default_context()
    try:
        import certifi

        ctx.load_verify_locations(certifi.where())
    except ImportError:
        ctx.check_hostname = False
        ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(url, context=ctx) as resp, zip_path.open("wb") as f:
        f.write(resp.read())
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(data_dir)
    inner_zip = data_dir / "UCI HAR Dataset.zip"
    if inner_zip.exists():
        with zipfile.ZipFile(inner_zip) as zf:
            zf.extractall(data_dir)
    return har_root(data_dir)


def _load_split(root: Path, split: str, files: list[str]) -> np.ndarray:
    signals = []
    base = root / split / "Inertial Signals"
    for fname in files:
        signals.append(np.loadtxt(base / fname))
    # Each file: (N, 128) -> stack channels -> (N, 128, 9)
    stacked = np.stack(signals, axis=-1)
    return stacked.astype(np.float32)


def load_har(
    data_dir: Path | str = "data/har",
    seed: int = 5798,
    n_val_subjects: int = 4,
) -> HARData:
    root = download_har(Path(data_dir))
    x_all = _load_split(root, "train", CHANNEL_FILES)
    y_all = np.loadtxt(root / "train" / "y_train.txt", dtype=np.int64).ravel() - 1
    subj = np.loadtxt(root / "train" / "subject_train.txt", dtype=np.int64).ravel()

    x_test = _load_split(root, "test", CHANNEL_FILES_TEST)
    y_test = np.loadtxt(root / "test" / "y_test.txt", dtype=np.int64).ravel() - 1

    train_subjects = np.unique(subj)
    rng = np.random.default_rng(seed)
    val_subjects = rng.choice(train_subjects, size=n_val_subjects, replace=False)
    val_mask = np.isin(subj, val_subjects)
    tr_mask = ~val_mask

    x_train, y_train = x_all[tr_mask], y_all[tr_mask]
    x_val, y_val = x_all[val_mask], y_all[val_mask]

    mean = x_train.mean(axis=(0, 1), keepdims=True)
    std = x_train.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-8, 1.0, std)

    def norm(x: np.ndarray) -> np.ndarray:
        return (x - mean) / std

    return HARData(
        x_train=norm(x_train),
        y_train=y_train,
        x_val=norm(x_val),
        y_val=y_val,
        x_test=norm(x_test),
        y_test=y_test,
        mean=mean,
        std=std,
        train_subjects=train_subjects,
        val_subjects=val_subjects,
    )


def class_balance_frame(y: np.ndarray, split_name: str) -> pd.DataFrame:
    labels = [
        "walking",
        "walking_upstairs",
        "walking_downstairs",
        "sitting",
        "standing",
        "laying",
    ]
    counts = pd.Series(np.bincount(y, minlength=6), index=labels, name="count")
    counts.index.name = "class"
    out = counts.reset_index()
    out["split"] = split_name
    return out