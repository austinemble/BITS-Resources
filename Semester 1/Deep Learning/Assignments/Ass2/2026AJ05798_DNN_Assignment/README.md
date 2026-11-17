# 2026AJ05798 — DNN Assignment 2

Student ID 2026AJ05798. Last digit 8, so the Problem 1 ablation is gradient clipping on the vanilla RNN at T = 200 (norm 1.0 vs none). Hidden size 72. Seeds 5798, 5898, 5998. The variant table was not in the brief I downloaded, so those three choices are written down here and in `common/config.py`.

## Environment

Python 3.10+ and PyTorch 2.x. Also numpy, pandas, matplotlib, scikit-learn. Tested on a CPU laptop (no CUDA). A Colab T4 is fine and faster for the language model.

```bash
cd 2026AJ05798_DNN_Assignment
pip install -r requirements.txt
```

## How to run

From this folder:

```bash
PYTHONPATH=. python problem1/run_pipeline.py
PYTHONPATH=. python problem2/run_pipeline.py
PYTHONPATH=. python problem2/extra_figures.py
```

Or open `problem1/problem1.ipynb` and `problem2/problem2.ipynb` and run all cells. The notebooks check the scratch cells and then show the tables and figures already saved from the runs above. Set `FAST_MODE = True` in `common/config.py` if you only want a smoke test.

Datasets download into `data/` on first use. Do not put `data/` in the zip.

On this CPU the Problem 1 runs took a few hours. The language model is 20 epochs of 500 random windows, and the head-count and no-mask runs use that same budget. That block was a bit over two hours.

## What is in the folder

- `report.pdf` — write-up
- `results.csv` — one row per run
- `problem1/figures/` — HAR, adding, gradients, sweep, clipping
- `problem2/figures/` — LM curve, attention, ablations, context time
- `problem2/generated_samples.txt` — 500 characters at temperatures 0.2, 0.8 and 1.5

## Assumption

Hidden size and seeds are as above because the individualisation table was missing from the handout.