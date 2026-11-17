"""Run Problem 2 experiments."""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from common.config import (
    P2_CONTEXT,
    P2_D_MODEL,
    P2_FF,
    P2_LM_STEPS,
    P2_LR,
    P2_N_HEADS,
    P2_N_LAYERS,
    SEEDS,
    har_epochs,
    p2_lm_epochs,
)
from common.results_log import ensure_results_file
from common.seed_utils import set_seed
from problem1.har_data import load_har
from problem1.training import train_har_classifier
from problem2.attention import causal_mask, hand_check_attention_example, scaled_dot_product_attention
from problem2.encoder import CausalTransformerLM, EncoderBlock, HAREncoderClassifier, encoder_block_param_count
from problem2.lm_train import generate, measure_context_cost, train_lm
from problem2.text_data import load_shakespeare


def _fig_dir() -> Path:
    d = Path(__file__).parent / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def test_attention() -> None:
    b, h, t, d = 2, 4, 5, 16
    q = torch.randn(b, h, t, d)
    k = torch.randn(b, h, t, d)
    v = torch.randn(b, h, t, d)
    mask = causal_mask(t, q.device)
    out, w = scaled_dot_product_attention(q, k, v, mask)
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)
    # Future positions must get zero weight under the causal mask.
    assert (w.masked_select(mask.unsqueeze(0).unsqueeze(0)) < 1e-6).all()
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    assert (out - ref).abs().max() < 1e-5
    hand_check_attention_example()


def run_har_transformer() -> None:
    print("HAR transformer: 3 seeds, same 30 epochs as Problem 1", flush=True)
    for seed in SEEDS:
        har = load_har(_ROOT / "data" / "har", seed=seed)
        for pos in ("none", "sinusoidal", "learned"):
            set_seed(seed)
            model = HAREncoderClassifier(pos_type=pos)
            train_har_classifier(
                model,
                har.x_train,
                har.y_train,
                har.x_val,
                har.y_val,
                har.x_test,
                har.y_test,
                epochs=har_epochs(),
                batch_size=64,
                lr=1e-3,
                seed=seed,
                model_name=f"transformer_har_{pos}",
                problem="P2",
            )


def run_lm() -> None:
    print("Loading Tiny Shakespeare (on-the-fly batches, no full window matrix)...", flush=True)
    stoi, itos, train_data, val_data = load_shakespeare(_ROOT / "data")
    vocab = len(stoi)
    print(f"vocab={vocab} train_chars={len(train_data)} val_chars={len(val_data)}", flush=True)
    samples = []
    for seed in SEEDS[:1]:
        set_seed(seed)
        model = CausalTransformerLM(
            vocab, P2_D_MODEL, P2_N_HEADS, P2_N_LAYERS, P2_FF, P2_CONTEXT
        )
        out = train_lm(model, train_data, val_data, p2_lm_epochs(), P2_LR, seed)
        history = out["history"]
        plt.figure(figsize=(6, 4))
        plt.plot(history)
        plt.xlabel("epoch")
        plt.ylabel("val loss (nats)")
        plt.title("Tiny Shakespeare validation loss")
        plt.tight_layout()
        plt.savefig(_fig_dir() / "lm_val_loss.png", dpi=150)
        plt.close()
        prompt = "ROMEO:"
        for temp in (0.2, 0.8, 1.5):
            text = generate(model, itos, prompt, stoi, 500, temp, top_k=10)
            samples.append(f"=== temperature {temp} ===\n{text}\n")
        _save_attention_map(model, stoi)
    (_ROOT / "problem2" / "generated_samples.txt").write_text("\n".join(samples), encoding="utf-8")


def _save_attention_map(model: CausalTransformerLM, stoi: dict[str, int]) -> None:
    sentence = "To be, or not to be"
    ids = [stoi[c] for c in sentence]
    model.eval()
    with torch.no_grad():
        model(torch.tensor([ids], device=next(model.parameters()).device))
    attn = model.blocks[-1].last_attn[0].cpu().numpy()
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.6))
    labels = list(sentence)
    for ax, h in zip(axes, (0, 1)):
        im = ax.imshow(attn[h], cmap="Blues", vmin=0, vmax=max(float(attn[h].max()), 1e-6))
        ax.set_xticks(range(len(labels)))
        ax.set_yticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)
        ax.set_xlabel("key")
        ax.set_ylabel("query")
        ax.set_title(f"last layer, head {h}")
        fig.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(_fig_dir() / "attention_heatmaps.png", dpi=140)
    plt.close()


def run_ablations() -> None:
    print(f"Ablations at the same budget: {p2_lm_epochs()} epochs x {P2_LM_STEPS} steps", flush=True)
    stoi, itos, train_data, val_data = load_shakespeare(_ROOT / "data")
    vocab = len(stoi)
    head_hist: dict[int, list[float]] = {}
    for heads in (1, 4, 8):
        set_seed(SEEDS[0])
        model = CausalTransformerLM(vocab, P2_D_MODEL, heads, P2_N_LAYERS, P2_FF, P2_CONTEXT)
        out = train_lm(
            model,
            train_data,
            val_data,
            p2_lm_epochs(),
            P2_LR,
            SEEDS[0],
            notes=f"heads={heads}",
        )
        head_hist[heads] = out["history"]
    plt.figure(figsize=(6, 4))
    for heads, hist in head_hist.items():
        plt.plot(hist, label=f"{heads} heads")
    plt.xlabel("epoch")
    plt.ylabel("val loss (nats)")
    plt.title("Head count, same training budget")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_fig_dir() / "heads_ablation.png", dpi=140)
    plt.close()

    set_seed(SEEDS[0])
    bare = CausalTransformerLM(
        vocab, P2_D_MODEL, P2_N_HEADS, P2_N_LAYERS, P2_FF, P2_CONTEXT, use_causal_mask=False
    )
    nomask = train_lm(
        bare,
        train_data,
        val_data,
        p2_lm_epochs(),
        P2_LR,
        SEEDS[0],
        notes="no_causal_mask",
    )
    text = generate(bare, itos, "ROMEO:", stoi, 500, temperature=0.8, top_k=10)
    (_ROOT / "problem2" / "no_mask_sample.txt").write_text(text, encoding="utf-8")
    # Main run history is already plotted; overlay the no-mask curve from this return value.
    plt.figure(figsize=(6, 4))
    plt.plot(nomask["history"], label="no causal mask")
    plt.xlabel("epoch")
    plt.ylabel("val loss (nats)")
    plt.title("Training without the causal mask")
    plt.legend()
    plt.tight_layout()
    plt.savefig(_fig_dir() / "no_mask_loss.png", dpi=140)
    plt.close()

    costs = measure_context_cost(vocab, P2_D_MODEL, P2_N_HEADS, P2_N_LAYERS, P2_FF, [64, 128, 256, 512])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    ctx = [r["context"] for r in costs]
    axes[0].plot(ctx, [r["ms_forward"] for r in costs], marker="o")
    axes[0].set_xlabel("context length")
    axes[0].set_ylabel("forward time (ms)")
    axes[1].plot(ctx, [r["peak_mb"] for r in costs], marker="o", color="#c44e52")
    axes[1].set_xlabel("context length")
    axes[1].set_ylabel("attention memory (MB)")
    fig.suptitle("Time and attention memory vs context")
    plt.tight_layout()
    plt.savefig(_fig_dir() / "context_time.png", dpi=140)
    plt.close()


def main() -> None:
    ensure_results_file()
    test_attention()
    d = P2_D_MODEL
    derived = encoder_block_param_count(d, P2_FF)
    block = EncoderBlock(d, P2_N_HEADS, P2_FF)
    actual = sum(p.numel() for p in block.parameters())
    print(f"Encoder block params: derived≈{derived}, actual={actual}", flush=True)
    print("Stage: HAR transformer (3 positional variants)", flush=True)
    run_har_transformer()
    print("Stage: character LM", flush=True)
    run_lm()
    run_ablations()
    print("Problem 2 pipeline complete.")


if __name__ == "__main__":
    main()