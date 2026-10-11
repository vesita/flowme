#!/usr/bin/env python3
"""L3-m2 中英模因锚定：**4 张卡**（中入 / 英入 / 中出 / 英出）+ **AaBb / AbBa 映射**。

    AaBb（自映射）= 中→中、英→英（重构）        AbBa（互映射）= 中→英、英→中（翻译）
    金标来自**平行语料**（Tatoeba）：互映射的目标就是配对句的另一半。

★ 为什么是**两个输出卡**而不是一个共享卡（`PREREG_ZH_EN.md` §2）：
  共享卡与"同内容两路模因应相似"**直接冲突** —— 若 `m_zh ≈ m_en`，同一个确定性解码器必须既出中文又出英文，
  唯一出路是让模因**另外再编码"语言"**，而那正是要它丢掉的东西（L3-m1 已见该逃逸：λ>0 时池化余弦全挤到 0.9996）。
  两个输出卡把"语言"交给**拓扑**承担 ⇒ 模因只承载**内容**。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model import TextInputCard, TextOutputCard


class ZHENModel(nn.Module):
    def __init__(self, shared_out: bool = False):
        super().__init__()
        self.shared_out = bool(shared_out)
        self.zh_in = TextInputCard()
        self.en_in = TextInputCard()
        self.zh_out = TextOutputCard()
        # `shared_out=True` 只用于对照（论文口径外的变体）：两路共用一个出卡
        self.en_out = self.zh_out if self.shared_out else TextOutputCard()

    def meme_zh(self, ids: torch.Tensor) -> torch.Tensor:
        return self.zh_in(ids)

    def meme_en(self, ids: torch.Tensor) -> torch.Tensor:
        return self.en_in(ids)

    def out_card(self, lang: str) -> TextOutputCard:
        assert lang in ("zh", "en"), lang
        return self.zh_out if lang == "zh" else self.en_out

    def logits(self, lang: str, meme: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.out_card(lang)(meme, tgt_in)

    def n_params(self) -> dict:
        d = {"zh_in": sum(p.numel() for p in self.zh_in.parameters()),
             "en_in": sum(p.numel() for p in self.en_in.parameters()),
             "zh_out": sum(p.numel() for p in self.zh_out.parameters())}
        d["en_out"] = 0 if self.shared_out else sum(p.numel() for p in self.en_out.parameters())
        return d


def build(seed: int, shared_out: bool = False) -> ZHENModel:
    """★ 先播种再构造（`L3_PLAN.md` §9.7：保证同 seed 的构造逐位可复现）。"""
    torch.manual_seed(int(seed))
    return ZHENModel(shared_out)
