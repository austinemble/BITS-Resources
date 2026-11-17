"""Sequence models for HAR classification and adding regression."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

ModelType = Literal["rnn", "lstm", "gru"]


class SequenceClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        model_type: ModelType,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.model_type = model_type
        if model_type == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(model_type)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rnn(x)
        h = out[0][:, -1, :]
        return self.head(self.drop(h))


class AddingRegressor(nn.Module):
    def __init__(self, hidden_size: int, model_type: ModelType) -> None:
        super().__init__()
        input_size = 2
        if model_type == "rnn":
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif model_type == "lstm":
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        elif model_type == "gru":
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError(model_type)
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rnn(x)
        h = out[0][:, -1, :]
        return self.head(h).squeeze(-1)


def init_lstm_forget_bias(module: nn.Module, value: float) -> None:
    if isinstance(module, nn.LSTM):
        for name, param in module.named_parameters():
            if "bias_hh" in name:
                n = param.shape[0]
                with torch.no_grad():
                    param[n // 4 : n // 2].fill_(value)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())