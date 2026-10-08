"""OutputCard —— 模因 [n, d] → logits [n, V]。

与 S19/S14 的 `head = nn.Linear(d, V)` 逐字一致（输出卡是接口的一半，见 §7.9①）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import IO_D_TO_LOGITS, Card


class OutputCard(Card):
    io = IO_D_TO_LOGITS
    card_name = "output"

    def __init__(self, d: int = 128, vocab_size: int = 8192, maxlen: int = 512):
        super().__init__(d, maxlen)
        self.vocab_size = int(vocab_size)
        self.head = nn.Linear(d, vocab_size)

    def apply_card(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.head(x)

    def out_last_dim(self) -> int:
        return self.vocab_size
