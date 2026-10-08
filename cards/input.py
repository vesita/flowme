"""InputCard —— token ids → 模因 [n, d]（embedding + 编码层 + 位置编码）。

数值路径与 S19/S14 逐字一致（框架不许自己发明架构）：

    x = emb(ids) + sin_pe[:n]        # 位置编码是加性的固定正弦
    h = TransformerEncoderLayer(d, nhead, ff, dropout, gelu, norm_first=True)(x, src_mask=m)

mask 约定 = 因果上三角 + padding 行全 -inf（`causal_mask` 是唯一实现点）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import IO_IDS_TO_D, Card


def sin_pe(max_len: int, d: int) -> torch.Tensor:
    """固定正弦位置编码（与 S19 `sin_pe` 逐字相同）。"""
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def encoder_layer(d: int, ff: int, nhead: int, dropout: float = 0.1) -> nn.TransformerEncoderLayer:
    """框架里唯一的标准卡内层（S19/S22 验证过的那个，不许换）。"""
    return nn.TransformerEncoderLayer(
        d, nhead, ff, dropout=dropout, activation="gelu", batch_first=True, norm_first=True
    )


class InputCard(Card):
    io = IO_IDS_TO_D
    card_name = "input"

    def __init__(self, d: int = 128, vocab_size: int = 8192, ff: int = 512, nhead: int = 4,
                 maxlen: int = 512, pad_id: int = 0, dropout: float = 0.1):
        super().__init__(d, maxlen, pad_id)
        self.vocab_size = int(vocab_size)
        self.nhead = int(nhead)
        self.emb = nn.Embedding(vocab_size, d)
        self.enc = encoder_layer(d, ff, nhead, dropout)
        self.register_buffer("pe", sin_pe(maxlen, d), persistent=False)

    def apply_card(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        n = ids.size(1)
        pe = self.pe if n <= self.pe.size(0) else sin_pe(n, self.d).to(self.pe.device)
        assert n <= pe.size(0), f"[契约] input: PE 长度 {pe.size(0)} < 序列长 {n}"
        x = self.emb(ids) + pe[:n].unsqueeze(0)
        m = mask if mask is not None else self.causal_mask(ids)
        return self.enc(x, src_mask=m)

    def causal_mask(self, ids: torch.Tensor) -> torch.Tensor:
        """因果 + padding 掩码 (b*nhead, n, n)（与 S19 逐字相同）。"""
        n = ids.size(1)
        m = torch.full((n, n), float("-inf"), device=ids.device)
        m = torch.triu(m, diagonal=1)
        m = m.unsqueeze(0).expand(ids.size(0), -1, -1).clone()
        m.masked_fill_((ids == self.pad_id).unsqueeze(1).expand_as(m), float("-inf"))
        return m.repeat_interleave(self.nhead, dim=0)
