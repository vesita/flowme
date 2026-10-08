"""TargetCard —— 模因 [n, d] → 模因 [n, d]（形状保持）。

卡内结构（S19/S22 已验证，逐字复用，不许自己发明）：

    Think = TransformerEncoderLayer(d, nhead, ff, dropout=0.1, gelu, norm_first=True)
    h = h + Think(h)        ← ★外层残差；S19 实测去掉它 add_3d EM 37.25% → 2.50%

残差是纯加法、不新增参数；`residual=False` 只保留给对照臂。
"""
from __future__ import annotations

import torch

from .base import IO_D_TO_D, Card
from .input import encoder_layer


class TargetCard(Card):
    io = IO_D_TO_D
    card_name = "target"

    def __init__(self, d: int = 128, ff: int = 512, nhead: int = 4, maxlen: int = 512,
                 dropout: float = 0.1, residual: bool = True):
        super().__init__(d, maxlen)
        self.residual = bool(residual)
        self.think = encoder_layer(d, ff, nhead, dropout)

    def apply_card(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        t = self.think(x, src_mask=mask)
        return x + t if self.residual else t
