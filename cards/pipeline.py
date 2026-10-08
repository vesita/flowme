"""CardPipeline —— 组装三类卡 + 黑板 + 版本化（框架骨架的入口）。

    ids ──▶ InputCard ──meme[n,d]──▶ TargetCard ──meme[n,d]──▶ OutputCard ──▶ logits[n,V]
                ▲                        ▲                          ▲
                └────── Blackboard：谁读了谁都有日志 ────────────────┘

  · `forward(ids)` = 走一遍黑板路由（S19 的 `Cards.logits` 等价物，参数路径逐字一致）；
  · `freeze(["input", "output"])` = 软冻结接口卡，只训目标卡（`trainable_parameters()`）；
  · `freeze` 之后被冻结卡的 `max|Δθ| == 0`（**精确**），由 `assert_frozen_unchanged` 强制；
  · `snapshot()`（version.py）把接口落成只读文件 + sha256 ⇒ MV 稳定。
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn

from .base import Card, Contract
from .blackboard import EXTERNAL, Blackboard
from .input import InputCard
from .output import OutputCard
from .target import TargetCard
from .version import INTERFACE_CARDS, MV, Snapshot, interface_sha256, make_mv, snapshot


class CardPipeline(nn.Module):
    CARD_NAMES = ("input", "target", "output")
    INTERFACE = INTERFACE_CARDS          # ("input", "output")

    def __init__(self, d: int = 128, vocab_size: int = 8192, ff: int = 512, nhead: int = 4,
                 maxlen: int = 512, pad_id: int = 0, dropout: float = 0.1,
                 residual: bool = True):
        super().__init__()
        # ★构造顺序 = S19 `Cards`：emb → in_enc → thought → head
        #   （nn 的参数初始化吃全局 RNG，顺序一致才能与 S19 逐位同初值）
        self.input = InputCard(d, vocab_size, ff, nhead, maxlen, pad_id, dropout)
        self.target = TargetCard(d, ff, nhead, maxlen, dropout, residual)
        self.output = OutputCard(d, vocab_size, maxlen)
        self.blackboard = Blackboard()
        self.frozen_cards: tuple = ()
        self.route_history: list = []

    # -------------------------------------------------------------- 组装
    def cards(self) -> dict:
        return {"input": self.input, "target": self.target, "output": self.output}

    def card(self, name: str) -> Card:
        assert name in self.CARD_NAMES, f"[管线] 未知卡名 {name!r}，可选 {self.CARD_NAMES}"
        return self.cards()[name]

    @property
    def contract(self) -> Contract:
        cs = [c.contract for c in self.cards().values()]
        assert all(c == cs[0] for c in cs), "[契约] 三张卡的契约必须一致（否则不是同一个模因）"
        return cs[0]

    # -------------------------------------------------------------- 路由
    def route(self, ids: torch.Tensor, mask: torch.Tensor | None = None,
              record: bool = True) -> dict:
        """走一遍黑板：input → target → output，并记录「谁读了谁」。"""
        bb = self.blackboard
        bb.clear()
        ver = {n: c.contract_sha256[:16] for n, c in self.cards().items()}
        bb.write("ids", ids, writer=EXTERNAL, version="-")
        meme_in = self.input(bb.read("ids", reader="input"))
        bb.write("meme", meme_in, writer="input", version=ver["input"])
        m = mask if mask is not None else self.input.causal_mask(ids)
        bb.write("mask", m, writer="input", version=ver["input"])
        meme = self.target(bb.read("meme", reader="target"),
                           mask=bb.read("mask", reader="target"))
        bb.write("meme", meme, writer="target", version=ver["target"])
        logits = self.output(bb.read("meme", reader="output"))
        bb.write("logits", logits, writer="output", version=ver["output"])
        if record:
            self.route_history.append(bb.read_log.copy())
            del self.route_history[:-64]              # 只留最近 64 次路由
        return {"logits": logits, "meme_in": meme_in, "meme": meme}

    def forward(self, ids: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.route(ids, mask=mask)["logits"]

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        """S19 `Cards.logits(ids)` 的同名等价物。"""
        return self.forward(ids)

    def who_reads_whom(self) -> list:
        return self.blackboard.who_reads_whom()

    # -------------------------------------------------------------- 冻结
    def freeze(self, names) -> tuple:
        """软冻结接口卡：requires_grad=False 且 eval；返回冻结名单（按 CARD_NAMES 排序）。"""
        names = tuple(names)
        for n in names:
            assert n in self.CARD_NAMES, f"[冻结] 未知卡名 {n!r}，可选 {self.CARD_NAMES}"
        for n in names:
            self.cards()[n].freeze()
        self.frozen_cards = tuple(n for n in self.CARD_NAMES if n in names)
        return self.frozen_cards

    def thaw(self, names=None) -> None:
        names = self.CARD_NAMES if names is None else tuple(names)
        for n in names:
            self.cards()[n].thaw()
        self.frozen_cards = tuple(n for n in self.CARD_NAMES if n not in names)

    def train(self, mode: bool = True):                 # noqa: D102 - 覆写以保护冻结卡
        super().train(mode)
        for n in self.frozen_cards:
            self.cards()[n].eval()                      # 冻结卡不许被 train() 拉回 dropout
        return self

    def trainable_parameters(self) -> list:
        out = []
        for n in self.CARD_NAMES:
            if n in self.frozen_cards:
                continue
            out += [p for p in self.cards()[n].parameters() if p.requires_grad]
        return out

    # -------------------------------------------------- Δθ / 冻结可验证
    def theta(self) -> dict:
        return {n: c.theta() for n, c in self.cards().items()}

    def max_delta(self, before: dict) -> dict:
        return {n: c.max_delta(before[n]) for n, c in self.cards().items()}

    def assert_frozen_unchanged(self, before: dict) -> dict:
        """★被冻结卡 max|Δθ| == 0（精确），否则当场炸。"""
        delta = self.max_delta(before)
        for n in self.frozen_cards:
            assert delta[n] == 0.0, (
                f"[冻结] {n} 卡被冻结却动了：max|Δθ|={delta[n]!r}（要求精确 0.0）")
        return delta

    # ------------------------------------------------------------ 版本化
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def interface_sha256(self) -> str:
        """★接口快照 sha256 = hash(契约, 输入卡权重, 输出卡权重)。"""
        return interface_sha256(self.cards(), self.contract)

    def mv(self) -> MV:
        return make_mv(self.contract, self.interface_sha256())

    def snapshot(self, path: str | Path) -> Snapshot:
        return snapshot(self, path)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        torch.save({"format": "pipeline-v1", "contract": self.contract.to_dict(),
                    "frozen_cards": list(self.frozen_cards), "state": self.state_dict()}, p)
        return p

    def load(self, path: str | Path, map_location: str = "cpu") -> Contract:
        blob = torch.load(Path(path), map_location=map_location, weights_only=False)
        got = Contract(**blob["contract"])
        assert got.sha256() == self.contract.sha256(), "[契约] 管线快照契约不一致（假同版本）"
        self.load_state_dict(blob["state"])
        if blob.get("frozen_cards"):
            self.freeze(blob["frozen_cards"])
        return got

    def __repr__(self) -> str:                            # pragma: no cover
        return (f"<CardPipeline d={self.input.d} maxlen={self.input.maxlen} "
                f"params={self.n_params()} frozen={list(self.frozen_cards)}>")
