"""Plots that the main run did not save: heads, no-mask, attention maps."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from common.config import P2_CONTEXT, P2_D_MODEL, P2_FF, P2_LR, P2_N_HEADS, P2_N_LAYERS, SEEDS
from common.seed_utils import set_seed
from problem2.encoder import CausalTransformerLM
from problem2.lm_train import generate, train_lm
from problem2.text_data import load_shakespeare

FIG = Path(__file__).parent / "figures"


def plot_ablation_bars() -> None:
    import csv

    rows = list(csv.DictReader((ROOT / "results.csv").open()))
    heads, losses = [], []
    nomask = None
    main = None
    for r in rows:
        if r["notes"] == "shakespeare":
            main = float(r["val_loss_nats"])
        elif r["notes"].startswith("heads="):
            heads.append(int(r["notes"].split("=")[1]))
            losses.append(float(r["val_loss_nats"]))
        elif r["notes"] == "no_causal_mask":
            nomask = float(r["val_loss_nats"])
    order = np.argsort(heads)
    heads = [heads[i] for i in order]
    losses = [losses[i] for i in order]
    plt.figure(figsize=(5.2, 3.4))
    plt.bar([str(h) for h in heads], losses, color="#4c72b0")
    plt.axhline(main, color="#c44e52", linestyle="--", label=f"main 4-head run ({main:.2f})")
    plt.ylabel("val loss (nats)")
    plt.xlabel("number of heads")
    plt.title("Head count, short runs (4 epochs x 80 steps)")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(FIG / "heads_ablation.png", dpi=140)
    plt.close()

    plt.figure(figsize=(4.6, 3.4))
    plt.bar(["causal\n(8x200)", "no mask\n(4x80)"], [main, nomask], color=["#4c72b0", "#dd8452"])
    plt.ylabel("final val loss (nats)")
    plt.title("Causal mask vs none")
    plt.tight_layout()
    plt.savefig(FIG / "no_mask_loss.png", dpi=140)
    plt.close()


def attention_and_nomask_sample() -> None:
    stoi, itos, train_data, val_data = load_shakespeare(ROOT / "data")
    vocab = len(stoi)
    set_seed(SEEDS[0])
    model = CausalTransformerLM(vocab, P2_D_MODEL, P2_N_HEADS, P2_N_LAYERS, P2_FF, P2_CONTEXT)
    train_lm(model, train_data, val_data, epochs=4, lr=P2_LR, seed=SEEDS[0], notes="attn_map_run", steps_per_epoch=120)
    sentence = "To be, or not to be"
    ids = [stoi[c] for c in sentence]
    model.eval()
    with torch.no_grad():
        model(torch.tensor([ids]))
    attn = model.blocks[-1].last_attn[0].cpu().numpy()  # (H, T, T)
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    labels = list(sentence)
    for ax, h in zip(axes, (0, 1)):
        im = ax.imshow(attn[h], cmap="Blues", vmin=0, vmax=max(attn[h].max(), 1e-6))
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("key")
        ax.set_ylabel("query")
        ax.set_title(f"last layer, head {h}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(FIG / "attention_heatmaps.png", dpi=140)
    plt.close()

    set_seed(SEEDS[0])
    bare = CausalTransformerLM(
        vocab, P2_D_MODEL, P2_N_HEADS, P2_N_LAYERS, P2_FF, P2_CONTEXT, use_causal_mask=False
    )
    train_lm(
        bare, train_data, val_data, epochs=3, lr=P2_LR, seed=SEEDS[0], notes="no_mask_sample", steps_per_epoch=80
    )
    text = generate(bare, itos, "ROMEO:", stoi, n_chars=280, temperature=0.8, top_k=10)
    (FIG.parent / "no_mask_sample.txt").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    plot_ablation_bars()
    attention_and_nomask_sample()
    print("extra figures done", flush=True)
