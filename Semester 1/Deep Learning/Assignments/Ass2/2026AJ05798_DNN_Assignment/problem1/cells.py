"""RNN / LSTM / GRU cells implemented from equations (PyTorch-compatible layout)."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

Tensor = torch.Tensor


class MyRNNCell(nn.Module):
    """h' = tanh(W_ih x + b_ih + W_hh h + b_hh). Shapes: x (B, d), h (B, h)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(hidden_size))
        self.bias_hh = nn.Parameter(torch.empty(hidden_size))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        return torch.tanh(
            F.linear(x, self.weight_ih, self.bias_ih)
            + F.linear(h, self.weight_hh, self.bias_hh)
        )


class MyLSTMCell(nn.Module):
    """Four-gate LSTM matching nn.LSTMCell (input, hidden) layout."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        h4 = hidden_size * 4
        self.weight_ih = nn.Parameter(torch.empty(h4, input_size))
        self.weight_hh = nn.Parameter(torch.empty(h4, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(h4))
        self.bias_hh = nn.Parameter(torch.empty(h4))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        h, c = state
        gates = (
            F.linear(x, self.weight_ih, self.bias_ih)
            + F.linear(h, self.weight_hh, self.bias_hh)
        )
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_new = f * c + i * g
        h_new = o * torch.tanh(c_new)
        return h_new, c_new


class MyGRUCell(nn.Module):
    """GRU cell matching nn.GRUCell gate ordering (r, z, n)."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        h3 = hidden_size * 3
        self.weight_ih = nn.Parameter(torch.empty(h3, input_size))
        self.weight_hh = nn.Parameter(torch.empty(h3, hidden_size))
        self.bias_ih = nn.Parameter(torch.empty(h3))
        self.bias_hh = nn.Parameter(torch.empty(h3))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = 1.0 / math.sqrt(self.hidden_size)
        for p in self.parameters():
            nn.init.uniform_(p, -std, std)

    def forward(self, x: Tensor, h: Tensor) -> Tensor:
        gi = F.linear(x, self.weight_ih, self.bias_ih)
        gh = F.linear(h, self.weight_hh, self.bias_hh)
        i_r, i_z, i_n = gi.chunk(3, dim=1)
        h_r, h_z, h_n = gh.chunk(3, dim=1)
        resetgate = torch.sigmoid(i_r + h_r)
        updategate = torch.sigmoid(i_z + h_z)
        newgate = torch.tanh(i_n + resetgate * h_n)
        h_new = (1.0 - updategate) * newgate + updategate * h
        return h_new


def copy_rnn_cell_weights(src: nn.RNNCell, dst: MyRNNCell) -> None:
    with torch.no_grad():
        dst.weight_ih.copy_(src.weight_ih)
        dst.weight_hh.copy_(src.weight_hh)
        dst.bias_ih.copy_(src.bias_ih)
        dst.bias_hh.copy_(src.bias_hh)


def copy_lstm_cell_weights(src: nn.LSTMCell, dst: MyLSTMCell) -> None:
    with torch.no_grad():
        dst.weight_ih.copy_(src.weight_ih)
        dst.weight_hh.copy_(src.weight_hh)
        dst.bias_ih.copy_(src.bias_ih)
        dst.bias_hh.copy_(src.bias_hh)


def copy_gru_cell_weights(src: nn.GRUCell, dst: MyGRUCell) -> None:
    with torch.no_grad():
        dst.weight_ih.copy_(src.weight_ih)
        dst.weight_hh.copy_(src.weight_hh)
        dst.bias_ih.copy_(src.bias_ih)
        dst.bias_hh.copy_(src.bias_hh)


def count_cell_params(input_size: int, hidden_size: int, cell: str) -> int:
    if cell == "rnn":
        m = MyRNNCell(input_size, hidden_size)
    elif cell == "lstm":
        m = MyLSTMCell(input_size, hidden_size)
    elif cell == "gru":
        m = MyGRUCell(input_size, hidden_size)
    else:
        raise ValueError(cell)
    return sum(p.numel() for p in m.parameters())


def verify_cells_match_library(
    input_size: int = 9,
    hidden_size: int = 64,
    batch: int = 5,
    tol: float = 1e-5,
) -> dict[str, float]:
    """Return max abs diffs for RNN, LSTM, GRU vs nn.*Cell on one random step."""
    torch.manual_seed(0)
    x = torch.randn(batch, input_size)
    h = torch.randn(batch, hidden_size)
    c = torch.randn(batch, hidden_size)

    ref_rnn = nn.RNNCell(input_size, hidden_size)
    my_rnn = MyRNNCell(input_size, hidden_size)
    copy_rnn_cell_weights(ref_rnn, my_rnn)
    d_rnn = (my_rnn(x, h) - ref_rnn(x, h)).abs().max().item()

    ref_lstm = nn.LSTMCell(input_size, hidden_size)
    my_lstm = MyLSTMCell(input_size, hidden_size)
    copy_lstm_cell_weights(ref_lstm, my_lstm)
    h_ref, c_ref = ref_lstm(x, (h, c))
    h_my, c_my = my_lstm(x, (h, c))
    d_lstm = max((h_my - h_ref).abs().max().item(), (c_my - c_ref).abs().max().item())

    ref_gru = nn.GRUCell(input_size, hidden_size)
    my_gru = MyGRUCell(input_size, hidden_size)
    copy_gru_cell_weights(ref_gru, my_gru)
    d_gru = (my_gru(x, h) - ref_gru(x, h)).abs().max().item()

    return {"rnn": d_rnn, "lstm": d_lstm, "gru": d_gru, "tol": tol}