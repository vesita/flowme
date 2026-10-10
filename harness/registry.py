"""harness.registry —— 卡片注册表：名字 →（构造器, 接口声明, 可判目标声明, 依赖张量, ckpt 加载器）。

★ **加一张卡只需要两件事**（H-d）：
    1) 继承 `harness.pipeline.HarnessCard`，声明 `spec_name / interface / deps / judged /
       ckpt_prefix`，实现 `forward_card(pkt)`；
    2) `REGISTRY.register(CardSpec(...))`（或对已有 spec 加参数）。
  **不需要**改 `pipeline.py`、`chat.py`、`probe.py`、`cli.py`。

`list-cards` 会把整张表打成人可读 + `--json` 机读两版。

依赖张量（`deps`）的写法（供冻结/微调与显存计算用，H-e）：
    "tokens"        读 Packet.tokens
    "h" / "mask"    读卡间接口隐态 / 掩码
    "ctx_len"       读回复区起点
    "aux:<名>"      读上游卡 emit 的张量（如 "aux:e_sel"）
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Sequence

import torch
import torch.nn as nn

from harness.pipeline import (D, DROPOUT, FF, MAXLEN, NHEAD, EncLayerHarnessCard, HarnessCard,
                              InputHarnessCard, Interface, JudgedTarget, OutputHarnessCard,
                              RealignHarnessCard, SelectHarnessCard, TargetHarnessCard,
                              encoder_layer)

ALL_KW = ("d", "ff", "nhead", "maxlen", "dropout", "vocab_size", "pad")


# ---------------------------------------------------------------------------
# 注册表条目
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CardSpec:
    """一张卡的登记项（构造器 + 全部声明 + ckpt 装载说明）。"""
    name: str
    build: Callable[..., HarnessCard]
    interface: Interface
    judged: JudgedTarget | None
    deps: tuple
    accepts: tuple = ALL_KW
    ckpt_prefix: str | None = None
    ckpt_root_keys: tuple = ()
    kind: str = "layer"          # interface | layer | head | example
    doc: str = ""
    source: str = "harness.registry"     # 构造器所在的实现文件（诚实边界：来自哪里）

    def filter_kwargs(self, kw: dict) -> dict:
        return {k: v for k, v in kw.items() if k in self.accepts}

    def row(self) -> dict:
        return {"name": self.name, "kind": self.kind,
                "params_style": f"d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN}",
                "interface": {"d_in": self.interface.d_in, "d_out": self.interface.d_out,
                              "injection": self.interface.injection,
                              "emits": list(self.interface.emits),
                              "carries_position": self.interface.carries_position},
                "deps": list(self.deps),
                "judged": None if self.judged is None else {
                    "name": self.judged.name, "metric": self.judged.metric,
                    "needs": list(self.judged.needs), "chance": self.judged.chance,
                    "doc": self.judged.doc},
                "ckpt_prefix": self.ckpt_prefix, "ckpt_root_keys": list(self.ckpt_root_keys),
                "source": self.source, "doc": self.doc}


class Registry:
    def __init__(self) -> None:
        self._specs: dict = {}

    def register(self, spec: CardSpec) -> CardSpec:
        assert spec.name not in self._specs, f"[注册表] 名字已存在：{spec.name!r}（不许静默覆盖）"
        self._specs[spec.name] = spec
        return spec

    def get(self, name: str) -> CardSpec:
        if name not in self._specs:
            raise KeyError(f"[注册表] 没有这张卡：{name!r}；可选 {sorted(self._specs)}")
        return self._specs[name]

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __getitem__(self, name: str) -> CardSpec:
        return self.get(name)

    def names(self) -> list:
        return sorted(self._specs)

    def rows(self) -> list:
        return [self._specs[n].row() for n in self.names()]

    def table(self) -> str:
        lines = [f"{'名字':<14}{'类':<10}{'接口':<12}{'注入':<6}{'可判目标':<46}{'ckpt 前缀':<12}"]
        for r in self.rows():
            j = "—" if r["judged"] is None else r["judged"]["metric"]
            if len(j) > 44:
                j = j[:43] + "…"
            lines.append(f"{r['name']:<14}{r['kind']:<10}"
                         f"{str(r['interface']['d_in']) + '→' + str(r['interface']['d_out']):<12}"
                         f"{r['interface']['injection']:<6}{j:<46}{str(r['ckpt_prefix']):<12}")
        return "\n".join(lines)


REGISTRY = Registry()

SPEC_INPUT = REGISTRY.register(CardSpec(
    name="input", build=InputHarnessCard, kind="interface",
    interface=InputHarnessCard.interface, judged=None, deps=("tokens",),
    accepts=("d", "ff", "nhead", "maxlen", "dropout", "vocab_size", "pad"),
    ckpt_prefix="input.",
    doc="token ids → 隐态；唯一施加固定正弦 PE 的地方（位置不进接口）"))

SPEC_SELECT = REGISTRY.register(CardSpec(
    name="select", build=SelectHarnessCard, kind="layer",
    interface=SelectHarnessCard.interface, judged=SelectHarnessCard.judged,
    deps=SelectHarnessCard.deps, accepts=("d", "ff", "nhead", "dropout"),
    ckpt_prefix="select.", ckpt_root_keys=("q",),
    doc="选择卡：h_j = 候选末位 token 隐态；s_j = q·h_j；emit e_sel"))

SPEC_REALIGN = REGISTRY.register(CardSpec(
    name="realign", build=RealignHarnessCard, kind="layer",
    interface=RealignHarnessCard.interface, judged=RealignHarnessCard.judged,
    deps=RealignHarnessCard.deps, accepts=("d", "ff", "nhead", "dropout"),
    ckpt_prefix="realign.",
    doc="重整卡：回复区加性注入 e_sel 后重整"))

SPEC_OUTPUT = REGISTRY.register(CardSpec(
    name="output", build=OutputHarnessCard, kind="head",
    interface=OutputHarnessCard.interface, judged=None, deps=("h",),
    accepts=("d", "maxlen", "vocab_size"), ckpt_prefix="output.",
    doc="输出卡：线性头 h → logits"))

SPEC_ENC = REGISTRY.register(CardSpec(
    name="enc_layer", build=EncLayerHarnessCard, kind="layer",
    interface=EncLayerHarnessCard.interface, judged=None, deps=("h", "mask"),
    accepts=("d", "ff", "nhead", "dropout"), ckpt_prefix=None,
    doc="通用裸 encoder 层卡（probe 的 N 张同构卡就是它）"))

class EncStackHarnessCard(HarnessCard):
    """把 N 层**裸 encoder 层**压成一个模块（probe 的「整体图」对照臂 C：等价计算，不是顺序执行）。"""
    spec_name = "enc_stack"
    interface = Interface(d_in=D, d_out=D, doc="N 层裸 encoder 层堆叠（整体图对照）")
    deps = ("h", "mask")
    ckpt_prefix = None

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, dropout: float = DROPOUT,
                 n_layers: int = 1):
        super().__init__()
        self.n_layers = int(n_layers)
        self.layers = nn.ModuleList([encoder_layer(d, ff, nhead, dropout)
                                     for _ in range(self.n_layers)])

    def forward_card(self, pkt):
        for L in self.layers:
            pkt.h = L(pkt.h, src_mask=pkt.mask)
        return pkt


SPEC_STACK = REGISTRY.register(CardSpec(
    name="enc_stack", build=EncStackHarnessCard, kind="layer",
    interface=EncStackHarnessCard.interface, judged=None, deps=("h", "mask"),
    accepts=("d", "ff", "nhead", "dropout", "n_layers"), ckpt_prefix=None,
    doc="N 层裸 encoder 层堆叠（probe 的整体图对照臂）"))


SPEC_TARGET = REGISTRY.register(CardSpec(
    name="target", build=TargetHarnessCard, kind="layer",
    interface=TargetHarnessCard.interface, judged=None, deps=("h", "mask"),
    accepts=("d", "ff", "nhead", "maxlen", "dropout"), ckpt_prefix="target.",
    source="cards.TargetCard(residual=False)",
    doc="★cards/ 正规路径：cards.TargetCard(residual=False) 原样接入"))


# ---------------------------------------------------------------------------
# A 臂的 2 层目标卡（79::DeepTargetCard 的同形实现；仅 ckpt 兼容需要）
# ---------------------------------------------------------------------------
class DeepTargetHarnessCard(HarnessCard):
    """A 臂的目标部分：把单卡的 1 层 think 堆到 2 层（`cards/target.py` 的 `residual=False` 口径）。"""
    spec_name = "target2"
    interface = Interface(d_in=D, d_out=D, doc="2 层裸 encoder 层（堆叠）")
    deps = ("h", "mask")
    ckpt_prefix = "target."

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT, n_layers: int = 2):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer(d, ff, nhead, dropout) for _ in range(n_layers)])

    def forward_card(self, pkt):
        for L in self.layers:
            pkt.h = L(pkt.h, src_mask=pkt.mask)
        return pkt


SPEC_TARGET2 = REGISTRY.register(CardSpec(
    name="target2", build=DeepTargetHarnessCard, kind="layer",
    interface=DeepTargetHarnessCard.interface, judged=None, deps=("h", "mask"),
    accepts=("d", "ff", "nhead", "maxlen", "dropout"), ckpt_prefix="target.",
    doc="A 臂的 2 层目标卡（与 79::DeepTargetCard 同形；只为复现 A 臂记录值）"))


# ---------------------------------------------------------------------------
# ★ 最小示例卡（README 引用它；H-d 的证据）
# ---------------------------------------------------------------------------
class ExampleGainHarnessCard(HarnessCard):
    """**最小示例卡**：`h → h + tanh(W h) ⊙ g`。

    这是"加一张卡"的最小完整形态：
      · 只实现 `forward_card(pkt)`（读 pkt.h、写 pkt.h）；
      · `g` 零初始化 ⇒ **未训练时逐位恒等**（输出与不加这张卡**逐字节相同**）；
      · 因此它同时是一个"诚实边界"的实证：**未训练的加卡 = 零增量 + 纯延迟**（§18.1）。

    训练它只需要 `g`（128 个参数）有梯度 —— 冻结其余卡即可（`deps` 就声明了它只读 `h`）。
    """
    spec_name = "example_gain"
    interface = Interface(d_in=D, d_out=D, injection="none",
                          doc="示例卡：h + tanh(W h) ⊙ g（g 零初始化 ⇒ 未训练时恒等）")
    deps = ("h",)
    judged = JudgedTarget(
        name="示例卡：**无**可判中间目标（故意留空 —— 这就是「只加卡、不加目标」的对照）",
        metric="无（对照读数：插进 B2 后输出应逐位不变）",
        needs=(), chance="", doc="要有可判目标，请照 README 的三步补 `judged` 字段。")
    ckpt_prefix = "example_gain."

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.g = nn.Parameter(torch.zeros(d))          # ★零初始化：未训练 = 恒等

    def forward_card(self, pkt):
        pkt.h = pkt.h + torch.tanh(self.proj(pkt.h)) * self.g
        return pkt


SPEC_EXAMPLE = REGISTRY.register(CardSpec(
    name="example_gain", build=ExampleGainHarnessCard, kind="example",
    interface=ExampleGainHarnessCard.interface, judged=ExampleGainHarnessCard.judged,
    deps=ExampleGainHarnessCard.deps,
    accepts=("d", "ff", "nhead", "maxlen", "dropout"), ckpt_prefix="example_gain.",
    source="harness.registry（示例）",
    doc="★最小示例卡：照它抄一遍 + register 就能加卡（README 有完整三步）"))


# ---------------------------------------------------------------------------
# 卡组（arch）→ [(实例名, 注册名), …]
# ---------------------------------------------------------------------------
CARD_SETS = {
    "A":       [("input", "input"), ("target", "target2"), ("output", "output")],
    "B1":      [("input", "input"), ("select", "select"), ("realign", "realign"), ("output", "output")],
    "B2":      [("input", "input"), ("select", "select"), ("realign", "realign"), ("output", "output")],
    "native3": [("input", "input"), ("target", "target"), ("output", "output")],
    "B2+ex":   [("input", "input"), ("select", "select"), ("example", "example_gain"),
                ("realign", "realign"), ("output", "output")],
}


def default_layers(arch: str) -> list:
    if arch not in CARD_SETS:
        raise KeyError(f"[注册表] 未知卡组 {arch!r}；可选 {sorted(CARD_SETS)}")
    return list(CARD_SETS[arch])


def parse_cards(spec: str) -> list:
    """解析 `--cards input:input,select:select,realign:realign,output:output`。"""
    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            iid, name = part.split(":", 1)
        else:
            iid = name = part
        REGISTRY.get(name)                       # fail-closed：名字不在表里当场炸
        layers.append((iid.strip(), name.strip()))
    assert layers, "--cards 解析后为空"
    return layers


def card_set_rows() -> list:
    return [{"arch": k, "layers": [{"instance": i, "spec": s} for i, s in v]}
            for k, v in CARD_SETS.items()]
