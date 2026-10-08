"""cards —— 卡片式模型的框架骨架：三类卡 + 黑板 + 版本化。

    from cards import CardPipeline, Card, InputCard, TargetCard, OutputCard, Blackboard
    from cards.version import snapshot, load_snapshot, mv_of

不依赖仓库其它任何包（尤其不 import stages/），只依赖 torch。
"""
from __future__ import annotations

from .base import (CONTRACT_VERSION, IO_D_TO_D, IO_D_TO_LOGITS, IO_IDS_TO_D, Card, Contract,
                   ShapeContractError, state_sha256)
from .blackboard import EXTERNAL, Blackboard, Read
from .input import InputCard, encoder_layer, sin_pe
from .output import OutputCard
from .pipeline import CardPipeline
from .target import TargetCard
from .version import (MV, W_VERSION, Snapshot, freeze, interface_sha256, load_snapshot,
                      make_mv, mv_of, snapshot, snapshot_is_readonly)

__all__ = [
    "Card", "Contract", "ShapeContractError", "CardPipeline",
    "InputCard", "TargetCard", "OutputCard", "Blackboard", "Read", "EXTERNAL",
    "sin_pe", "encoder_layer", "state_sha256",
    "MV", "Snapshot", "freeze", "snapshot", "load_snapshot", "mv_of", "make_mv",
    "interface_sha256", "snapshot_is_readonly", "W_VERSION",
    "CONTRACT_VERSION", "IO_IDS_TO_D", "IO_D_TO_D", "IO_D_TO_LOGITS",
]
