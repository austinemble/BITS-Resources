"""Build report.pdf from the saved runs. 6 content pages, then the AI note."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
P1 = ROOT / "problem1" / "figures"
P2 = ROOT / "problem2" / "figures"


def stats(vals: list[float]) -> str:
    a = np.array(vals, dtype=float)
    return f"{a.mean():.3f} ± {a.std(ddof=1):.3f}"


def load_rows() -> list[dict]:
    with (ROOT / "results.csv").open() as f:
        return list(csv.DictReader(f))


class Report(FPDF):
    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Times", "I", 9)
        self.cell(0, 6, str(self.page_no()), align="C")


def h(pdf: FPDF, text: str) -> None:
    pdf.set_font("Times", "B", 12)
    pdf.multi_cell(0, 6, text)
    pdf.ln(1)


def p(pdf: FPDF, text: str) -> None:
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(0, 4.8, text)
    pdf.ln(1)


def img(pdf: FPDF, path: Path, w: float) -> None:
    if pdf.get_y() > 250:
        pdf.add_page()
    pdf.image(str(path), w=w)
    pdf.ln(1)


def main() -> None:
    rows = load_rows()

    def pick(model: str, note: str, T: str | None = None) -> list[dict]:
        out = []
        for r in rows:
            if r["model"] != model or r["notes"] != note:
                continue
            if T is not None and r["seq_len"] != T:
                continue
            out.append(r)
        return out

    har = {}
    for m in ("rnn", "lstm", "gru"):
        rs = pick(m, "HAR", "128")
        har[m] = rs

    pdf = Report(format="A4")
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.set_margins(18, 16, 18)
    pdf.add_page()

    pdf.set_font("Times", "B", 14)
    pdf.cell(0, 8, "DNN Assignment 2: RNN, LSTM, GRU and a small Transformer", ln=1)
    pdf.set_font("Times", "", 11)
    pdf.cell(0, 6, "Student ID: 2026AJ05798    AIML ZG511", ln=1)
    pdf.ln(2)

    h(pdf, "Problem 1")
    p(
        pdf,
        "Last digit of my ID is 8, so the extra ablation is gradient clipping on the plain RNN at T=200. "
        "The variant table was not in the copy of the brief I had, so I set the hidden size to 72 (64+8) "
        "and used seeds 5798, 5898 and 5998. Same Adam learning rate 1e-3 for every comparison. "
        "HAR: 30 epochs, batch 64. Adding problem: 40 epochs, batch 128.",
    )
    p(
        pdf,
        "HAR is the UCI phone data, 128 steps by 9 channels, six activities. For each seed I held out 4 of the "
        "21 training subjects with the numpy call from the handout. Channel mean and std come from those "
        "training windows only. Fig. 1 is the class balance on the first split. Walking and the two stair "
        "classes are a bit thinner than sitting, standing and laying, but nothing is missing.",
    )
    img(pdf, P1 / "har_class_balance.png", 105)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 1. Class counts on the training subjects (seed 5798).")
    pdf.ln(1)

    p(
        pdf,
        "The adding data is the function in the brief. I assert that each sequence has exactly two markers, "
        "one in each half, and that the target is the sum of the two marked values.",
    )
    p(
        pdf,
        "Scratch cells. MyRNNCell is tanh(Wih x + bih + Whh h + bhh). The LSTM and GRU cells use the same "
        "gate order as PyTorch (i, f, g, o and r, z, n). On a random batch of size 5 the max absolute gap "
        "versus nn.RNNCell and nn.LSTMCell was 0, and versus nn.GRUCell about 2e-7, all under 1e-5.",
    )
    p(
        pdf,
        "For d=9 and h=64 a single bias gives RNN 4,736, LSTM 18,944, GRU 14,208. PyTorch stores bias_ih and "
        "bias_hh, so the library (and my cells, which copy that layout) report 4,800, 19,200 and 14,400. "
        "The extra 64, 256 and 192 parameters are just the second bias.",
    )

    h(pdf, "HAR classification")
    lines = ["model    acc            macro-F1       params   s/epoch"]
    for m, label in (("rnn", "RNN"), ("lstm", "LSTM"), ("gru", "GRU")):
        rs = har[m]
        acc = stats([float(r["test_acc"]) for r in rs])
        f1 = stats([float(r["macro_f1"]) for r in rs])
        sec = float(np.mean([float(r["sec_per_epoch"]) for r in rs]))
        lines.append(f"{label:6} {acc:16} {f1:16} {rs[0]['params']:>6}   {sec:.2f}")
    pdf.set_font("Courier", "", 9)
    for line in lines:
        pdf.cell(0, 4.4, line, ln=1)
    pdf.ln(1)
    p(
        pdf,
        "Table 1. HAR test accuracy and macro-F1, mean ± sample std over the three seeds. "
        "The best single run was the LSTM with seed 5998 (90.1%). Its confusion matrix is Fig. 2. "
        "Most of the remaining mistakes are between sitting and standing, which makes sense from the sensors. "
        "The RNN is both weaker and less stable across seeds. LSTM and GRU are close; the GRU has fewer weights "
        "and is a bit quicker per epoch.",
    )
    img(pdf, P1 / "har_confusion_matrix.png", 78)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 2. Confusion matrix of the best HAR run (LSTM, seed 5998).")

    h(pdf, "Adding problem")
    p(
        pdf,
        "A constant prediction of 1.0 has MSE about 0.167. I call a run solved if test MSE is below 0.05. "
        "Numbers below are mean test MSE over 3 seeds, and how many of the 3 were solved.",
    )
    pdf.set_font("Courier", "", 9)
    pdf.cell(0, 4.4, "T     RNN (solved)     LSTM (solved)    GRU (solved)", ln=1)
    for T in ("50", "100", "200", "400"):
        bits = []
        for m in ("rnn", "lstm", "gru"):
            rs = pick(m, "adding", T)[:3]
            mse = [float(r["test_mse"]) for r in rs]
            solved = sum(v < 0.05 for v in mse)
            bits.append(f"{np.mean(mse):.3f} ({solved}/3)")
        pdf.cell(0, 4.4, f"{T:5} {bits[0]:16} {bits[1]:16} {bits[2]}", ln=1)
    pdf.ln(1)
    pdf.set_font("Times", "", 11)
    p(
        pdf,
        "Table 2. At T=50 the gated models solve it and the RNN does not. By T=400 only the GRU still solves "
        "every seed. One LSTM seed also failed at T=100 and T=200 (MSE stuck near 0.17), so the LSTM is not "
        "reliable here even when the average looks fine. Fig. 3 is the training loss at T=200 for seed 5798: "
        "GRU falls, RNN and LSTM stay flat. Fig. 4 is the same story against T.",
    )
    img(pdf, P1 / "adding_curves_T200.png", 108)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 3. Training MSE at T=200, seed 5798.")
    img(pdf, P1 / "adding_mse_vs_T.png", 108)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 4. Test MSE against sequence length. Dashed line is the constant-1 baseline.")

    h(pdf, "Gradients, width, clipping")
    p(
        pdf,
        "Fig. 5 plots ||dL/dh_t|| at T=100 before and after a short training run. The RNN curve dies toward "
        "the start of the sequence. LSTM and GRU keep a larger gradient on earlier steps. That matches Table 2: "
        "the first marker is in the first half of the sequence, and the plain RNN cannot carry it to the end "
        "once T is long. It is not really exploding on this task. It just forgets, so the loss sits on the baseline.",
    )
    img(pdf, P1 / "gradient_norms_T100.png", 112)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 5. Hidden-state gradient norm against time, T=100.")
    p(
        pdf,
        "Fig. 6 is a hidden-size sweep on HAR, one seed, same 30 epochs as the main table. "
        + "LSTM: "
        + ", ".join(
            f"h={h} {float([r for r in rows if r['model']==f'lstm_h{h}'][-1]['test_acc']):.3f}"
            for h in (16, 32, 64, 128)
        )
        + ". GRU: "
        + ", ".join(
            f"h={h} {float([r for r in rows if r['model']==f'gru_h{h}'][-1]['test_acc']):.3f}"
            for h in (16, 32, 64, 128)
        )
        + ". GRU is already strong at a small hidden size. Past 64, extra width does not buy much accuracy, "
        "so h=72 for the main runs is enough.",
    )
    img(pdf, P1 / "har_hidden_sweep.png", 100)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 6. HAR test accuracy against hidden size.")
    p(
        pdf,
        "Clipping, which is the ablation for an even ID. Vanilla RNN, T=200, seed 5798: MSE 0.168 with no clip "
        "and 0.168 with clip norm 1.0 (Fig. 7). Clipping did not unlock the task. The gradients are small, not huge.",
    )
    img(pdf, P1 / "ablation_rnn_grad_clip.png", 72)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 7. RNN at T=200, clip 1.0 against no clip.")
    p(
        pdf,
        "If I had to put one of these on a battery wearable I would pick the GRU. On HAR it is within about one "
        "point of the LSTM, with fewer parameters and a shorter epoch. On the long adding sequences it is the "
        "only model that still solves T=400. The transformer in Problem 2 scores higher on HAR, but it has "
        "roughly 400k parameters against about 18k for this GRU. I would not spend that on a watch.",
    )

    h(pdf, "Problem 2")
    p(
        pdf,
        "Scaled dot-product attention is the usual softmax(QK^T / sqrt(d)) V, with an optional boolean mask. "
        "The checks: each row of weights sums to 1, a causal mask gives about zero weight on future positions, "
        "and the output matches torch.nn.functional.scaled_dot_product_attention (is_causal=True) within 1e-5. "
        "The 3-token example in the notebook uses q=k=[(1,0),(0,1),(1,1)] and v=[(1,2),(3,4),(5,6)] with a causal mask. "
        "Hand calculation and the code agree on that example.",
    )
    p(
        pdf,
        "One pre-norm encoder block, d=128, feed-forward width 512. The weight-only count is 4d^2 + 2 d d_ff "
        "= 196,608. Adding the four attention biases, the two linear biases in the MLP, and two LayerNorms "
        "(scale and bias) brings it to 198,272. sum(p.numel()) on the block is 198,272.",
    )
    tf_bits = []
    for pos in ("none", "sinusoidal", "learned"):
        rs = [r for r in rows if r["model"] == f"transformer_har_{pos}"]
        acc = stats([float(r["test_acc"]) for r in rs])
        sec = float(np.mean([float(r["sec_per_epoch"]) for r in rs]))
        tf_bits.append(f"{pos} {acc} ({rs[0]['params']} params, {sec:.2f} s/epoch)")
    p(
        pdf,
        "HAR encoder. I patch every 8 steps, add a CLS token, and use two blocks. Same 30 epochs and the same "
        "three seeds as the recurrent models, including the subject split. "
        + ". ".join(tf_bits)
        + ". The best Problem 1 model was about 90% with 24k parameters. The encoder is a bit more accurate and "
        "much heavier. Position encoding does not move the number much. The patches are already in time order, "
        "and a lot of the activity is visible in the raw signal, so an extra position vector is not doing much here.",
    )

    h(pdf, "Character language model")
    lm = [r for r in rows if r["notes"] == "shakespeare"][-1]
    p(
        pdf,
        "Tiny Shakespeare, 65 characters, first 90% for training. Context 128, d=128, 4 heads, 4 layers, "
        "feed-forward 512. Each epoch is 500 random windows of length 128, for 20 epochs. Building every "
        "overlapping window first never finished, so I sample instead. Final validation loss is "
        f"{float(lm['val_loss_nats']):.3f} nats, which is {float(lm['bits_per_char']):.2f} bits per character. "
        "A flat guess over 65 symbols is ln 65, about 4.17 nats, so this is well below that. Fig. 8 is the curve.",
    )
    img(pdf, P2 / "lm_val_loss.png", 100)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 8. Validation loss on Tiny Shakespeare.")
    p(
        pdf,
        "Samples of 500 characters, prompt ROMEO:, top-k 10. At temperature 0.2 it stays on safe phrases "
        "(\"the state of the sea\", \"the senate\") and repeats them. At 0.8 the line breaks and the speaker "
        "names look more like a scene (MENENIUS, CORIOLANUS, LADY ANNE), with some invented words such as "
        "\"defency\" and \"fint\". At 1.5 it still looks like verse, but the names and words are made up "
        "(MORD, MAMILLIA, \"firdel\", \"senatoan\"). Low temperature is dull and repetitive. High temperature "
        "keeps the shape of a scene and loses the spelling.",
    )
    p(
        pdf,
        "Fig. 9 is the last layer of that same model on the line \"To be, or not to be\". "
        "The top-right of both plots is empty, which is the causal mask. Head 0 stays close to the diagonal "
        "(the current character and the one just before it). Head 1 spreads more over the allowed past. "
        "I still would not call either head a syntax detector. They mostly track recent characters.",
    )
    img(pdf, P2 / "attention_heatmaps.png", 150)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 9. Last-layer attention, heads 0 and 1.")

    head_rows = sorted(
        [r for r in rows if r["notes"].startswith("heads=")],
        key=lambda r: int(r["notes"].split("=")[1]),
    )
    head_txt = ", ".join(
        f"{r['notes'].split('=')[1]} heads {float(r['val_loss_nats']):.3f}" for r in head_rows
    )
    p(
        pdf,
        "Head count (Fig. 10), same 20 epochs and 500 steps, d fixed at 128. Final validation loss: "
        f"{head_txt}. The curves sit close together. Changing the head count did not matter nearly as much "
        "as just training the model.",
    )
    img(pdf, P2 / "heads_ablation.png", 100)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 10. Validation loss for 1, 4 and 8 heads. Same budget as the main run.")
    nom = [r for r in rows if r["notes"] == "no_causal_mask"][-1]
    p(
        pdf,
        "Removing the causal mask (Fig. 11), same budget, finishes at "
        f"{float(nom['val_loss_nats']):.3f} nats. That is not a real next-character loss. Each position can see "
        "the tokens it is supposed to predict, so the training loss collapses. The sample from that run at "
        "temperature 0.8 starts with a long run of colons and then broken lines. The tiny loss does not mean "
        "it can write. For generation the mask has to stay on.",
    )
    img(pdf, P2 / "no_mask_loss.png", 78)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 11. Validation loss with the causal mask removed. Same epoch budget as Fig. 8.")
    p(
        pdf,
        "Fig. 12 is one forward pass at context 64, 128, 256 and 512. Left side is wall time. Right side is "
        "the memory of the attention scores and weights, which is the n^2 piece (this laptop has no GPU, so "
        "that is the memory I can actually attribute to attention). Both grow faster than linear. Doubling "
        "the context more than doubles the time, which is what O(n^2 d) says.",
    )
    img(pdf, P2 / "context_time.png", 100)
    pdf.set_font("Times", "I", 9)
    pdf.multi_cell(0, 4.5, "Fig. 12. Forward time and attention memory against context length.")

    p(
        pdf,
        "Taken together, the gated RNN is the practical sequence model on the sensor data, and the transformer "
        "is what I would use once the sequence is text and I can afford the quadratic attention. On HAR the "
        "encoder's extra accuracy is real but expensive. On Shakespeare the small causal model gets out of the "
        "uniform-guess regime, and the temperature knob does what it is supposed to do.",
    )

    # AI note and references are outside the 6-page body. If the body already
    # ran long, this still sits after it and is labelled as not part of the page cap.
    pdf.add_page()
    h(pdf, "AI-use statement (not counted in the 6 pages)")
    p(pdf, "I used an AI tool to debug two runtime errors and to tidy a few plots.")
    h(pdf, "References")
    pdf.set_font("Times", "", 11)
    for ref in (
        "D. Anguita et al., A Public Domain Dataset for Human Activity Recognition Using Smartphones, ESANN 2013. UCI HAR.",
        "A. Karpathy, char-rnn tiny Shakespeare (input.txt).",
        "A. Vaswani et al., Attention Is All You Need, NeurIPS 2017.",
        "PyTorch documentation, nn.RNN / nn.LSTM / nn.GRU (two bias vectors).",
    ):
        pdf.multi_cell(0, 5.2, ref)
        pdf.ln(0.5)

    out = ROOT / "report.pdf"
    pdf.output(str(out))
    print(f"wrote {out} pages={pdf.page}")


if __name__ == "__main__":
    main()
