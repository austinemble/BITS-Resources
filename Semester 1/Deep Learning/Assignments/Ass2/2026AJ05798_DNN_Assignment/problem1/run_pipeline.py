"""Run Problem 1 experiments (HAR, adding, ablations)."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch

from common.config import (
    ADDING_BATCH,
    ADDING_LR,
    HAR_BATCH,
    HAR_LR,
    HIDDEN_SIZE,
    LAST_DIGIT,
    SEEDS,
    adding_epochs,
    har_epochs,
)
from common.results_log import ensure_results_file
from common.seed_utils import set_seed
from problem1.adding_data import load_adding_splits
from problem1.har_data import class_balance_frame, load_har
from problem1.rnn_models import AddingRegressor, SequenceClassifier
from sklearn.metrics import ConfusionMatrixDisplay

from problem1.training import gradient_norms_over_time, train_adding, train_har_classifier


def _fig_dir() -> Path:
    d = Path(__file__).parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def plot_class_balance(har) -> None:
    df = class_balance_frame(har.y_train, "train")
    ax = df.plot.bar(x="class", y="count", legend=False, figsize=(8, 4))
    ax.set_title("HAR class balance (training split)")
    plt.tight_layout()
    plt.savefig(_fig_dir() / "har_class_balance.png", dpi=150)
    plt.close()


def run_har() -> None:
    data_dir = _ROOT / "data" / "har"
    best: tuple[float, str, np.ndarray, np.ndarray] | None = None
    for seed in SEEDS:
        set_seed(seed)
        har = load_har(data_dir, seed=seed)
        if seed == SEEDS[0]:
            plot_class_balance(har)
        for model_type in ("rnn", "lstm", "gru"):
            model = SequenceClassifier(9, HIDDEN_SIZE, 6, model_type)
            out = train_har_classifier(
                model,
                har.x_train,
                har.y_train,
                har.x_val,
                har.y_val,
                har.x_test,
                har.y_test,
                epochs=har_epochs(),
                batch_size=HAR_BATCH,
                lr=HAR_LR,
                seed=seed,
                model_name=model_type,
            )
            score = out["test_acc"]
            if best is None or score > best[0]:
                best = (score, model_type, out["y_true"], out["y_pred"])
    if best:
        _, name, yt, yp = best
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_predictions(yt, yp, ax=ax)
        ax.set_title(f"Best HAR model on test ({name}, acc={best[0]:.3f})")
        plt.tight_layout()
        plt.savefig(_fig_dir() / "har_confusion_matrix.png", dpi=150)
        plt.close()


def run_adding() -> None:
    lengths = [50, 100, 200, 400]
    curves: dict[str, list[float]] = {m: [] for m in ("rnn", "lstm", "gru")}
    mse_table: dict[tuple[str, int], list[float]] = {}
    for T in lengths:
        x_tr, y_tr, _, _, x_te, y_te = load_adding_splits(T)
        for seed in SEEDS:
            for model_type in ("rnn", "lstm", "gru"):
                set_seed(seed)
                model = AddingRegressor(HIDDEN_SIZE, model_type)
                hist = T == 200 and seed == SEEDS[0]
                out = train_adding(
                    model,
                    x_tr,
                    y_tr,
                    x_te,
                    y_te,
                    T=T,
                    epochs=adding_epochs(),
                    batch_size=ADDING_BATCH,
                    lr=ADDING_LR,
                    seed=seed,
                    model_name=model_type,
                    history=hist,
                )
                if hist and "losses" in out:
                    curves[model_type] = out["losses"]
                mse_table.setdefault((model_type, T), []).append(out["test_mse"])
    if curves["rnn"]:
        plt.figure(figsize=(7, 4))
        for m, loss in curves.items():
            if loss:
                plt.plot(loss, label=m)
        plt.xlabel("epoch")
        plt.ylabel("train MSE")
        plt.title("Adding problem learning curves (T=200)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(_fig_dir() / "adding_curves_T200.png", dpi=150)
        plt.close()
    fig, ax = plt.subplots(figsize=(8, 4))
    for model_type in ("rnn", "lstm", "gru"):
        means = [np.mean(mse_table[(model_type, T)]) for T in lengths]
        ax.plot(lengths, means, marker="o", label=model_type)
    ax.axhline(0.167, color="gray", linestyle="--", label="constant 1.0 baseline")
    ax.set_xlabel("sequence length T")
    ax.set_ylabel("test MSE (mean over seeds)")
    ax.legend()
    ax.set_title("Adding problem: MSE vs T")
    plt.tight_layout()
    plt.savefig(_fig_dir() / "adding_mse_vs_T.png", dpi=150)
    plt.close()


def run_ablation_rnn_clip() -> None:
    T = 200
    x_tr, y_tr, _, _, x_te, y_te = load_adding_splits(T)
    scores: list[tuple[str, float]] = []
    for clip, label in [(None, "none"), (1.0, "clip1")]:
        set_seed(SEEDS[0])
        model = AddingRegressor(HIDDEN_SIZE, "rnn")
        out = train_adding(
            model,
            x_tr,
            y_tr,
            x_te,
            y_te,
            T=T,
            epochs=adding_epochs(),
            batch_size=ADDING_BATCH,
            lr=ADDING_LR,
            seed=SEEDS[0],
            model_name=f"rnn_{label}",
            grad_clip=clip,
        )
        scores.append((label, out["test_mse"]))
    labels, vals = zip(*scores)
    plt.figure(figsize=(5, 4))
    plt.bar(labels, vals)
    plt.ylabel("test MSE")
    plt.title("RNN T=200: gradient clipping ablation (ID last digit even)")
    plt.tight_layout()
    plt.savefig(_fig_dir() / "ablation_rnn_grad_clip.png", dpi=150)
    plt.close()


def run_hidden_size_sweep() -> None:
    har = load_har(_ROOT / "data" / "har", seed=SEEDS[0])
    sizes = [16, 32, 64, 128]
    fig, ax = plt.subplots(figsize=(7, 4))
    for model_type in ("lstm", "gru"):
        accs = []
        for h in sizes:
            set_seed(SEEDS[0])
            model = SequenceClassifier(9, h, 6, model_type)
            out = train_har_classifier(
                model,
                har.x_train,
                har.y_train,
                har.x_val,
                har.y_val,
                har.x_test,
                har.y_test,
                epochs=har_epochs(),
                batch_size=HAR_BATCH,
                lr=HAR_LR,
                seed=SEEDS[0],
                model_name=f"{model_type}_h{h}",
            )
            accs.append(out["test_acc"])
        ax.plot(sizes, accs, marker="o", label=model_type)
    ax.set_xlabel("hidden size")
    ax.set_ylabel("test accuracy")
    ax.set_title("HAR hidden-size sweep (LSTM vs GRU)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(_fig_dir() / "har_hidden_sweep.png", dpi=150)
    plt.close()


def run_gradient_plot() -> None:
    T = 100
    x_tr, y_tr, _, _, _, _ = load_adding_splits(T)
    fig, ax = plt.subplots(figsize=(8, 4))
    for model_type in ("rnn", "lstm", "gru"):
        set_seed(SEEDS[0])
        untrained = AddingRegressor(HIDDEN_SIZE, model_type)
        norms_pre = gradient_norms_over_time(untrained, x_tr, y_tr, False, T)
        model = AddingRegressor(HIDDEN_SIZE, model_type)
        train_adding(
            model,
            x_tr,
            y_tr,
            x_tr[:500],
            y_tr[:500],
            T=T,
            epochs=min(adding_epochs(), 15),
            batch_size=ADDING_BATCH,
            lr=ADDING_LR,
            seed=SEEDS[0],
            model_name=model_type,
        )
        norms_post = gradient_norms_over_time(model, x_tr, y_tr, True, T)
        t = np.arange(T)
        ax.plot(t, norms_pre, linestyle="--", label=f"{model_type} pre")
        ax.plot(t, norms_post, label=f"{model_type} post")
    ax.set_yscale("log")
    ax.set_xlabel("time step")
    ax.set_ylabel("||dL/dh_t||")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(_fig_dir() / "gradient_norms_T100.png", dpi=150)
    plt.close()


def main() -> None:
    ensure_results_file()
    print(f"Student variant: h={HIDDEN_SIZE}, seeds={SEEDS}, last_digit={LAST_DIGIT}")
    run_har()
    run_adding()
    if LAST_DIGIT % 2 == 0:
        run_ablation_rnn_clip()
    run_gradient_plot()
    run_hidden_size_sweep()
    print("Problem 1 pipeline complete. See results.csv and problem1/figures/")


if __name__ == "__main__":
    main()