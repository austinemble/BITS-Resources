"""Write the two notebooks. They check the code and display saved results."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def nb(cells: list[dict]) -> dict:
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def md(source: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "source": source.splitlines(keepends=True),
        "outputs": [],
        "execution_count": None,
    }


P1 = [
    md(
        "# Problem 1 — RNN, LSTM and GRU\n\n"
        "2026AJ05798. Last digit 8, so the extra run is gradient clipping on the plain RNN at T=200. "
        "Hidden size 72, seeds 5798 / 5898 / 5998.\n\n"
        "The long training was already run (`problem1/run_pipeline.py`). "
        "The cells below re-check the scratch cells and then show the saved figures and `results.csv`."
    ),
    code(
        """import sys
from pathlib import Path
ROOT = Path('..').resolve()
sys.path.insert(0, str(ROOT))

from common.config import HIDDEN_SIZE, LAST_DIGIT, SEEDS, STUDENT_ID
print(STUDENT_ID, 'h', HIDDEN_SIZE, 'seeds', SEEDS, 'last digit', LAST_DIGIT)"""
    ),
    md("## Cells\n\nThese have to match `nn.RNNCell`, `nn.LSTMCell` and `nn.GRUCell` within 1e-5. The parameter counts use both bias vectors, which is why they are 4800 / 19200 / 14400 rather than 4736 / 18944 / 14208."),
    code(
        """from problem1.cells import verify_cells_match_library, count_cell_params

diffs = verify_cells_match_library()
print(diffs)
assert diffs['rnn'] < 1e-5 and diffs['lstm'] < 1e-5 and diffs['gru'] < 1e-5
for name in ('rnn', 'lstm', 'gru'):
    print(name, count_cell_params(9, 64, name))"""
    ),
    md("## Adding-problem check\n\nEvery row should have two markers, one in each half, and y equal to the sum of the marked values."),
    code(
        """from problem1.adding_data import load_adding_splits
x, y, *_ = load_adding_splits(50)
print(x.shape, y.shape, float(y.mean()))"""
    ),
    md("## Saved HAR / adding results"),
    code(
        """import pandas as pd
from IPython.display import Image, display

df = pd.read_csv(ROOT / 'results.csv')
har = df[(df.notes == 'HAR') & (df.model.isin(['rnn', 'lstm', 'gru']))]
print(har.groupby('model')[['test_acc', 'macro_f1', 'sec_per_epoch']].agg(['mean', 'std']))
add = df[(df.notes == 'adding') & (df.model.isin(['rnn', 'lstm', 'gru']))]
# gradient-plot rerun appended a second T=100 block; keep the first three seeds
add = add.groupby(['model', 'seq_len', 'seed'], as_index=False).first()
print(add.groupby(['seq_len', 'model'])['test_mse'].mean())

for name in [
    'har_class_balance.png',
    'har_confusion_matrix.png',
    'adding_curves_T200.png',
    'adding_mse_vs_T.png',
    'gradient_norms_T100.png',
    'har_hidden_sweep.png',
    'ablation_rnn_grad_clip.png',
]:
    print(name)
    display(Image(filename=str(ROOT / 'problem1' / 'figures' / name)))"""
    ),
]

P2 = [
    md(
        "# Problem 2 — Transformer from scratch\n\n"
        "No `nn.MultiheadAttention` or `nn.Transformer` in the model code. "
        "Library attention is only used in the check below.\n\n"
        "Training outputs are already in `results.csv`, `problem2/figures/` and `generated_samples.txt`."
    ),
    code(
        """import sys
from pathlib import Path
ROOT = Path('..').resolve()
sys.path.insert(0, str(ROOT))"""
    ),
    md("## Attention check"),
    code(
        """from problem2.run_pipeline import test_attention
from problem2.attention import hand_check_attention_example
from problem2.encoder import EncoderBlock, encoder_block_param_count

test_attention()
ex = hand_check_attention_example()
print(ex['weights'])
print('row sums', ex['weights'].sum(-1))

block = EncoderBlock(128, 4, 512)
print('derived', encoder_block_param_count(128, 512), 'actual', sum(p.numel() for p in block.parameters()))"""
    ),
    md("## Saved language-model results"),
    code(
        """import pandas as pd
from IPython.display import Image, display

df = pd.read_csv(ROOT / 'results.csv')
print(df[df.problem == 'P2'][['model', 'val_loss_nats', 'bits_per_char', 'notes']])
print((ROOT / 'problem2' / 'generated_samples.txt').read_text()[:500])
for name in ['lm_val_loss.png', 'attention_heatmaps.png', 'heads_ablation.png', 'no_mask_loss.png', 'context_time.png']:
    print(name)
    display(Image(filename=str(ROOT / 'problem2' / 'figures' / name)))"""
    ),
]

for name, cells in [("problem1/problem1.ipynb", P1), ("problem2/problem2.ipynb", P2)]:
    path = ROOT / name
    path.write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("wrote", path)
