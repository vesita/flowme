#!/usr/bin/env python3
"""L3-m2/L3-m3 中英模因：输入卡（中入/英入）+ 输出卡（中出/英出）+ 可选的 **parser 卡 t**（语言解析卡）。

    AaBb（自映射）= 中→中、英→英        AbBa（互映射）= 中→英、英→中
    ★ 插入 t 后就是用户 2026-10-11 的写法：**Ata Btb Atb Bta**（t 在输入模因与输出卡之间，两语言共用）。

★ 为什么是**两个输出卡**而不是一个共享卡（`PREREG_ZH_EN.md` §2）：
  共享卡与"同内容两路模因应相似"**直接冲突** —— 若 `m_zh ≈ m_en`，同一个确定性解码器必须既出中文又出英文，
  唯一出路是让模因**另外再编码"语言"**，而那正是要它丢掉的东西（L3-m1 已见该逃逸：λ>0 时池化余弦全挤到 0.9996）。
  两个输出卡把"语言"交给**拓扑**承担 ⇒ 模因只承载**内容**。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from model import D, DROPOUT, TextInputCard, TextOutputCard


class ParserCard(nn.Module):
    """**parser 卡 t**（用户 2026-10-11 命名）：模因 `[b,16,128]` → 模因 `[b,16,128]`（两种语言**共用同一张**）。

    - 卡内结构 = **自注意力 + 逐位置 FFN**（`MEME_FRAMEWORK.md §5.1` 的硬要求）；
    - 带**容量瓶颈** `d_mid < d`（`§13.2`：强制路由必须与瓶颈配对，否则 t 会**退化成恒等**）；
    - 输出 **残差 + LayerNorm**：t 是"改写模因"，不是"替换模因"。
    """

    def __init__(self, mid: int = 64, layers: int = 2, nhead: int = 4, ff: int = 256):
        super().__init__()
        assert mid < D, "t 必须有容量瓶颈（§13.2）"
        self.proj_in = nn.Linear(D, mid)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(mid, nhead, ff, DROPOUT, batch_first=True, norm_first=True)
            for _ in range(layers)])
        self.proj_out = nn.Linear(mid, D)
        self.norm = nn.LayerNorm(D)

    def forward(self, m: torch.Tensor) -> torch.Tensor:
        x = self.proj_in(m)
        for layer in self.layers:
            x = layer(x)
        return self.norm(m + self.proj_out(x))


class ZHENModel(nn.Module):
    def __init__(self, shared_out: bool = False, bridge: str = "none", bridge_mid: int = 64):
        super().__init__()
        assert bridge in ("none", "parser", "dual"), bridge
        self.shared_out = bool(shared_out)
        self.bridge_kind = bridge
        self.zh_in = TextInputCard()
        self.en_in = TextInputCard()
        self.zh_out = TextOutputCard()
        # `shared_out=True` 只用于对照（口径外的变体）：两路共用一个出卡
        self.en_out = self.zh_out if self.shared_out else TextOutputCard()
        # bridge="parser"：**一张共享**的解析卡（两语言都过它）
        # bridge="dual"  ：**两条不共享**的 adapter（中/英各一张，同单卡尺寸）—— 控制臂 V：
        #                  用来分离"共享"与"额外容量"（见 PREREG_PARSER_CTRL.md §1）
        self.bridge: ParserCard | None = ParserCard(bridge_mid) if bridge == "parser" else None
        self.bridge_zh: ParserCard | None = ParserCard(bridge_mid) if bridge == "dual" else None
        self.bridge_en: ParserCard | None = ParserCard(bridge_mid) if bridge == "dual" else None

    def meme_zh(self, ids: torch.Tensor) -> torch.Tensor:
        return self.zh_in(ids)

    def meme_en(self, ids: torch.Tensor) -> torch.Tensor:
        return self.en_in(ids)

    def apply_bridge(self, m: torch.Tensor, lang: str) -> torch.Tensor:
        """把输入模因送过解析卡；`bridge="none"` 时是恒等 ⇒ 与 L3-m2 的行为逐位一致。

        `lang` 只在 `bridge="dual"`（不共享）时用到。
        """
        if self.bridge is not None:                      # 共享
            return self.bridge(m)
        if self.bridge_zh is not None:                   # 不共享：按语言选卡
            return (self.bridge_zh if lang == "zh" else self.bridge_en)(m)
        return m

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
        d["bridge"] = 0
        for c in (self.bridge, self.bridge_zh, self.bridge_en):
            if c is not None:
                d["bridge"] += sum(p.numel() for p in c.parameters())
        return d


def build(seed: int, shared_out: bool = False, bridge: str = "none",
          bridge_mid: int = 64) -> ZHENModel:
    """★ 先播种再构造（`L3_PLAN.md` §9.7：保证同 seed 的构造逐位可复现）。"""
    torch.manual_seed(int(seed))
    return ZHENModel(shared_out, bridge, bridge_mid)
