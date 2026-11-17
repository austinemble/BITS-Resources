"""Assignment-wide configuration for student 2026AJ05798."""

from __future__ import annotations

STUDENT_ID = "2026AJ05798"
LAST_DIGIT = 8  # 2026AJ05798

# Individualisation table was not in the distributed brief; see README for assumptions.
# Last digit 8 -> even -> P1.5(c): vanilla RNN gradient clipping (1.0) vs none at T=200.
HIDDEN_SIZE = 72
SEEDS = [5798, 5898, 5998]
DROPOUT = 0.1

HAR_EPOCHS = 30
HAR_BATCH = 64
HAR_LR = 1e-3
ADDING_EPOCHS = 40
ADDING_BATCH = 128
ADDING_LR = 1e-3

P2_D_MODEL = 128
P2_N_HEADS = 4
P2_N_LAYERS = 4
P2_CONTEXT = 128
P2_FF = 512
P2_LR = 3e-4
P2_LM_EPOCHS = 20
P2_LM_STEPS = 500
P2_LM_BATCH = 64
P2_VAL_BATCHES = 40

# Set True to shrink runs for smoke tests (not for final submission).
FAST_MODE = False

FAST_HAR_EPOCHS = 2
FAST_ADDING_EPOCHS = 3
FAST_P2_LM_EPOCHS = 2

RESULTS_COLUMNS = [
    "problem",
    "model",
    "seed",
    "seq_len",
    "test_acc",
    "macro_f1",
    "test_mse",
    "params",
    "sec_per_epoch",
    "val_loss_nats",
    "bits_per_char",
    "notes",
]


def har_epochs() -> int:
    return FAST_HAR_EPOCHS if FAST_MODE else HAR_EPOCHS


def adding_epochs() -> int:
    return FAST_ADDING_EPOCHS if FAST_MODE else ADDING_EPOCHS


def p2_lm_epochs() -> int:
    return FAST_P2_LM_EPOCHS if FAST_MODE else P2_LM_EPOCHS