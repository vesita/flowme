"""显式调度层：把「该调哪张卡、怎么收尾」写成**可枚举的有限状态机**（规则策略 v0）。

dev-notes/16 §2.0–§2.1 的形式化在这里落地，分工严格照抄那张表：

    MDP = (S, A, P, R, γ)，本层只手写 S / A / π；P 只给抽象关系，不给概率。
      S = `Signals` = (调用方声明的 type, 4 位三态 info, 3 档 conf, steps∈{0,1,2,3})
          ⇒ |S| ≤ 5 × 3⁴ × 3 × 4 = 4860，`enumerate_state_space()` 可全枚举；
      A = `CallCard`（调一张卡）+ `Terminate`（终止三选一：指针/生成/拒答）；
          **可行动作 = `admissible(s)`，不是全局 A**（§2.2 D2：规格要求，见 `_authorized`）；
      π = `decide()`：dev-notes/16 §2.1 那张规则表的 if-else 链，逐行照抄；
          必需位按 type 声明（§2.2 D5，见 `REQUIRED_BITS`），行序按 §2.2 D3（缺位先拒答）；
      P = `possible_outcomes()`：**抽象、非确定、不接真卡** —— 只描述"调一张卡后
          它服务的信息位可能变有/无、conf 换档、steps 减一"（conf 转移见 §2.2 D4）。
          真实转移概率留待用数据估（手写概率等于把猜测固化）；这一层只够做可达性 / 死状态检查。

为什么必须显式（实测，dev-notes/7、13、14）：元决策交给模型会崩 ——
路由 98.83%→9.19%、输入类型 91.67%→29.78%、拒答分布外 ~96% 误开火。
所以 type 由调用方声明，四项信号全部"算出来"，策略是手写规则表：
**显式、有限、可枚举、可审**。

两条 fail-closed 硬要求（dev-notes/16 §2.1），破坏任意一条都是 bug：
  1. 信息位对应的卡不在 `plugin.all_tasks()` 里 ⇒ 该状态**拒答**，
     不许静默跳过这个信息位；
  2. 任何"读不到 / 判不动"（type 不认识、info 位数不对、位种类未声明/取值未知、
     conf/steps 越界、注册表读不出来、source_span 结构非法）都通向**拒答**，
     不许落到默认值上 —— 抛异常也不行，那只是把 fail-closed 换成 fail-crash。

§3.1 规格订正（位种类 `bit_kinds`）：`required` = 「必须被**判定**过」，不是"必须为有"；
`无` 对抽取类是缺信息（拒答）、对检测类是**判否**（有效结论 ⇒ 继续）；
「调用失败」不进状态机，在对话层的调用出口 fail-closed 拒答。

用法::

    reg = {"slot": card, "intent": card}      # 测试/审计注入；传 None 用生产注册表
    s = Signals("plain", (UNCHECKED,) * 4, Conf.LOW, MAX_STEPS)
    p = plan(s, registry=reg, source_span=(0, 12))   # Plan{action, reason, steps}
"""
from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from itertools import product

__all__ = [
    "BIT_KINDS",
    "CONF_LABELS",
    "CONF_RANK",
    "GENERATE",
    "INFO_BITS",
    "INFO_LABELS",
    "INPUT_TYPES",
    "KIND_DETECT",
    "KIND_EXTRACT",
    "MAX_STEPS",
    "NUM_INFO_BITS",
    "POINTER",
    "REGISTRY_ERROR_KEY",
    "REJECT",
    "REQUIRED_BITS",
    "STATE_SPACE_SIZE",
    "STEPS_VALUES",
    "TERMINAL_KINDS",
    "UNKNOWN_TYPE",
    "Action",
    "CallCard",
    "Conf",
    "InfoBit",
    "InfoState",
    "Plan",
    "Registry",
    "Signals",
    "Step",
    "Terminate",
    "action_key",
    "action_kind",
    "admissible_actions",
    "closure_problems",
    "dead_states",
    "decide",
    "dispatch_problems",
    "enumerate_state_space",
    "initial_states",
    "load_default_registry",
    "plan",
    "possible_outcomes",
    "reachable_states",
    "registry_problems",
    "required_bit_indices",
    "resolve_registry",
    "signals_problems",
    "terminal_kinds_reachable",
]

# ---- 信号的取值域（S 的四个分量） ------------------------------------------

#: 调用方**声明**的输入类型（§11.1）—— 模型判不动（对抗集 29.78%），所以不判。
INPUT_TYPES: tuple[str, ...] = ("plain", "candidates", "cloze", "multi_turn", "unknown")
UNKNOWN_TYPE = "unknown"

#: 剩余可调用次数的取值域（§2.1 + §2.2 D1：steps ∈ {0,1,2,3}）。
#: 上限必须容得下 `INFO_BITS` 映射声明的卡各调一次 —— 否则调用没走完就被
#: `steps==0` 兜底成拒答，`generate` / `pointer` 永不可达。
STEPS_VALUES: tuple[int, ...] = (0, 1, 2, 3)
MAX_STEPS = max(STEPS_VALUES)


class InfoState(Enum):
    """一个信息位的三态。value 是稳定 id（可序列化），中文标签见 `INFO_LABELS`。"""

    UNCHECKED = "unchecked"   # 未查：这张卡还没调过
    HAVE = "have"             # 有：调过、产出了切片、最高置信度 ≥ 该卡阈值
    ABSENT = "absent"         # 无：调过但没产出 —— 含义按位种类（BIT_KINDS）分岔：
                              #   抽取类 = 缺信息 ⇒ 拒答；检测类 = 判否（有效结论）⇒ 继续


INFO_LABELS = {InfoState.UNCHECKED: "未查", InfoState.HAVE: "有", InfoState.ABSENT: "无"}


class Conf(Enum):
    """置信度的 3 档（最近一次调用的置信度**硬阈值**化，阈值取该卡在自己 val 上的分位）。"""

    LOW = "low"
    MID = "mid"
    HIGH = "high"


CONF_RANK = {Conf.LOW: 0, Conf.MID: 1, Conf.HIGH: 2}
CONF_LABELS = {Conf.LOW: "低", Conf.MID: "中", Conf.HIGH: "高"}

TERMINAL_KINDS: tuple[str, ...] = ("pointer", "generate", "reject")
POINTER = "pointer"
GENERATE = "generate"
REJECT = "reject"


@dataclass(frozen=True)
class InfoBit:
    """一个必需信息位，以及**声明式**给出的「它由哪张卡供给」（整张表在 `INFO_BITS`）。"""

    key: str
    """稳定 id，如 `time`。也是 `CallCard.bit` 里记的值。"""

    label: str
    """中文展示名，供打印审阅。"""

    card: str
    """供给这个位的卡名 —— 必须与 `plugin.all_tasks()` 里的注册名**逐字**相同；
    对不上就是"卡未注册"，全线拒答（fail-closed，不许静默跳过该位）。"""


#: 4 位必需信息位 → 卡的声明式映射（dev-notes/16 §2.1 的例子：时间/地点/对象/意图；
#: 供给它们的两张卡是 §2 第二层"要补的卡"：槽位抽取、意图/言语行为）。
#:
#: **映射允许非单射（§2.2 D1）**：一张卡可以服务多个信息位 —— `slot` 一张卡供
#: time / place / object 三位，`intent` 供 intent 一位，位 → 卡共 2 张不同的卡。
#: 位的个数（4）与卡的张数（2）是两回事：调用单位是**卡**，`steps` 按卡扣。
#:
#: 这张表是补卡时**唯一**的改动点：`slot` / `intent` 就是待补卡的注册名契约（§2.2 D7），
#: 新卡必须**逐字同名**注册进 `plugin.all_tasks()`；在册与否由 `registry_problems()` 判。
INFO_BITS: tuple[InfoBit, ...] = (
    InfoBit("time", "时间", "slot"),
    InfoBit("place", "地点", "slot"),
    InfoBit("object", "对象", "slot"),
    InfoBit("intent", "意图", "intent"),
)
NUM_INFO_BITS = len(INFO_BITS)  # 必须是 4（§2.1）

BIT_INDEX: Mapping[str, int] = {bit.key: i for i, bit in enumerate(INFO_BITS)}

#: 每类 type 的**必需信息位**（§2.2 D5：不再对 5 类一刀切）—— 只有必需位驱动
#: 「有未查位 ⇒ 调卡」「全有 ⇒ 输出」两行规则；`unknown` 没有必需位，直接拒答。
REQUIRED_BITS: Mapping[str, tuple[str, ...]] = {
    "plain": ("intent",),
    "cloze": ("object",),
    "candidates": ("intent", "object"),
    "multi_turn": ("time", "place", "object", "intent"),
    UNKNOWN_TYPE: (),
}

#: 位的**种类**（dev-notes/16 §3.1 规格订正）—— `无` 这个值对两类含义完全不同：
#:
#: | 种类 | `无` 的含义 | 规则 |
#: |---|---|---|
#: | 抽取类（时间/地点/对象/人物） | 该信息确实不存在 | `无` ⇒ 拒答（缺必需信息） |
#: | 检测类（否定/情绪/安全） | **判否**，是有效结论 | `无` ⇒ 继续（记「否」并推进） |
#:
#: 取值域就是这两个常量；未声明 / 取值未知 ⇒ `signals_problems()` 判不动 ⇒ 拒答（fail-closed）。
#: 默认表给 `INFO_BITS` 全部标 `extract`（time/place/object/intent 本来就全是抽取类）
#: ⇒ 默认规则行为一字不变；对话层注入自己的那张（`dialogue.DIALOGUE_BIT_KINDS`）。
KIND_EXTRACT = "extract"
KIND_DETECT = "detect"
BIT_KINDS: Mapping[str, str] = {bit.key: KIND_EXTRACT for bit in INFO_BITS}

#: 状态空间上界：5 类 type × 3⁴ 位 × 3 档 conf × 4 档 steps = 4860。
#: 位仍是**三态**：§3.1 的第 4 态「调用失败」不进状态机 —— 它在规则表里没有出边
#: （唯一出口就是拒答），所以在对话层的**调用出口**实现（fail-closed），观测等价且 |S| 不变。
STATE_SPACE_SIZE = (
    len(INPUT_TYPES) * (len(InfoState) ** NUM_INFO_BITS) * len(Conf) * len(STEPS_VALUES)
)


def required_bit_indices(
    type_: str,
    *,
    info_bits: tuple[InfoBit, ...] = INFO_BITS,
    required_bits: Mapping[str, tuple[str, ...]] = REQUIRED_BITS,
) -> tuple[int, ...]:
    """该 type 的必需位在 `info_bits` 里的下标，**按位序**（升序）；声明表里没有的 type ⇒ 空元组。

    声明表与 `INPUT_TYPES`/`info_bits` 对不齐由 `signals_problems()` 判成"判不动"
    ⇒ 拒答，不会静默漏掉一个必需位。按位序保证同一状态下授权集的行序固定（确定性）。

    `info_bits` / `required_bits` 默认即模块级 `INFO_BITS` / `REQUIRED_BITS`（默认行为
    一字不变）；对话层注入自己那张映射到现有卡的位表（dev-notes/16 §3 四段之第 2 段）。
    """
    keys = required_bits.get(type_, ())
    index = {bit.key: i for i, bit in enumerate(info_bits)}
    return tuple(sorted(index[key] for key in keys if key in index))


#: 注册表**读取失败**时写进表里的哨兵键：让"读不到"在拒答原因里露头，
#: 而不是悄悄退化成"卡未注册"（两者都拒答，但排查入口完全不同）。
REGISTRY_ERROR_KEY = "__dtseek_registry_error__"


# ---- 状态 S ----------------------------------------------------------------

@dataclass(frozen=True)
class Signals:
    """状态 s = (声明的 type, 4 位 info, conf, steps)。四项全部"算出来"，无一项靠模型判。

    **故意不在构造时校验**：读不到/判不动的信号要走到 `decide()` 变成拒答，
    而不是在这里抛异常。所以 `signals_problems()` 才是校验入口。
    """

    type: str
    info: tuple[InfoState, ...]
    conf: Conf
    steps: int

    def describe(self) -> str:
        """一行中文状态描述 —— 每个可达状态都能打印出来人工审（§2.1 的要求）。"""
        info = ",".join(INFO_LABELS.get(v, repr(v)) for v in self.info)
        conf = CONF_LABELS.get(self.conf, repr(self.conf))
        return f"type={self.type} info=[{info}] conf={conf} steps={self.steps}"


def initial_states(*, info_bits: tuple[InfoBit, ...] = INFO_BITS) -> tuple[Signals, ...]:
    """每个可声明 type 一个起点：各位全"未查"、steps=MAX_STEPS、conf=LOW。

    conf 初值 = `LOW`（§2.2 D4）：初态没有"最近一次调用"可读，取最保守档；
    它只会把终点从生成降级到指针，绝不把拒答变成放行。

    `info_bits` 决定位的个数（默认 `INFO_BITS` ⇒ 4 位，`NUM_INFO_BITS`）——
    注入别的位表时起点随它变长；默认行为一字不变。
    """
    info = (InfoState.UNCHECKED,) * len(info_bits)
    return tuple(Signals(t, info, Conf.LOW, MAX_STEPS) for t in INPUT_TYPES)


def enumerate_state_space() -> Iterator[Signals]:
    """全状态空间的笛卡尔积：恰好 `STATE_SPACE_SIZE` = 4860 个，**可完全枚举**。

    枚举全空间是为了给"真实可达集是它的子集"做交叉校验（`reachable_states` 的测试）。
    """
    for type_ in INPUT_TYPES:
        for info in product(InfoState, repeat=NUM_INFO_BITS):
            for conf in Conf:
                for steps in STEPS_VALUES:
                    yield Signals(type_, tuple(info), conf, steps)


# ---- 动作 A ----------------------------------------------------------------

@dataclass(frozen=True)
class CallCard:
    """调用一张卡。`card` 是注册表里的卡名，`bit` 是触发这次调用的信息位 key（审计用）。"""

    card: str
    bit: str

    def __post_init__(self) -> None:
        if not self.card or not self.bit:
            raise ValueError("CallCard 的 card / bit 都不能为空")


@dataclass(frozen=True)
class Terminate:
    """终止三选一：`pointer` 降级指针输出 / `generate` 生成输出 / `reject` 拒答。

    `reason` 记录触发的规则行 —— 计划要可审，就必须说得出"为什么是这个动作"。
    """

    kind: str
    reason: str

    def __post_init__(self) -> None:
        if self.kind not in TERMINAL_KINDS:
            raise ValueError(f"终止动作 {self.kind!r} 不在 {TERMINAL_KINDS}")


Action = CallCard | Terminate


def action_kind(action: Action) -> str:
    """动作的可比标签：`call` / `pointer` / `generate` / `reject`。"""
    if isinstance(action, CallCard):
        return "call"
    return action.kind


def action_key(action: Action) -> tuple:
    """动作的逐字可比表示（确定性 / diff 用）。"""
    if isinstance(action, CallCard):
        return ("call", action.card, action.bit)
    return ("terminate", action.kind, action.reason)


# ---- 计划 Plan --------------------------------------------------------------

@dataclass(frozen=True)
class Step:
    """计划里的一步：调 `module_id` 这个模块，`input_span` 落在原文或上游产出上。

    `input_span` **不在构造时校验** —— 闭合性校验器 `closure_problems()` 得能拿到
    坏样本才测得住自己；构造期直接抛错会让校验器永远无从失败。
    """

    module_id: str
    input_span: tuple[int, int] | None = None
    params: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if not self.module_id:
            raise ValueError("Step.module_id 不能为空")


@dataclass(frozen=True)
class Plan:
    """一次决策的完整产物：状态 + 选中动作 + 触发的规则行 + 要执行的 Step 序列。"""

    signals: Signals
    action: Action
    reason: str
    steps: tuple[Step, ...]

    def as_tuple(self) -> tuple:
        """逐字可比的纯元组 —— 确定性验证就是比这个（`repr` 也逐字相同）。"""
        s = self.signals
        return (
            (s.type, tuple(v.value for v in s.info), s.conf.value, s.steps),
            action_key(self.action),
            self.reason,
            tuple((st.module_id, st.input_span, st.params) for st in self.steps),
        )

    def describe(self) -> str:
        """一行中文计划描述：状态 ⇒ 动作（原因）+ Step 序列。"""
        seq = " → ".join(f"{st.module_id}@{st.input_span}" for st in self.steps) or "(无 Step)"
        return f"[{self.signals.describe()}] ⇒ {action_kind(self.action)}：{self.reason} | {seq}"


def closure_problems(steps: Iterable[Step]) -> list[str]:
    """闭合性的**结构**校验：每个 `Step.input_span` 必须是 `None` 或 `(s, e)` 且 `0 <= s <= e`。

    本层只做结构校验 —— "区间真的落在原文/上游产出上"要等真文本进来才能判。
    读不懂的 span 一律报问题，绝不放行（fail-closed：判不动 ⇒ 不通过）。
    """
    bad: list[str] = []
    for i, st in enumerate(steps):
        span = st.input_span
        if span is None:
            continue
        if not isinstance(span, (tuple, list)) or len(span) != 2:
            bad.append(f"steps[{i}].input_span={span!r} 不是 (start, end)")
            continue
        start, end = span
        if isinstance(start, bool) or not isinstance(start, int):
            bad.append(f"steps[{i}].input_span={span!r} 起点不是整数")
            continue
        if isinstance(end, bool) or not isinstance(end, int):
            bad.append(f"steps[{i}].input_span={span!r} 终点不是整数")
            continue
        if not (0 <= start <= end):
            bad.append(f"steps[{i}].input_span={span!r} 必须满足 0 <= start <= end")
    return bad


# ---- 注册表（fail-closed 门禁的数据来源） -----------------------------------

Registry = Mapping[str, object]
"""卡名 → 卡对象。本层只用**成员关系**判断"这张卡在不在册"，不调用卡本身。"""


def load_default_registry() -> dict[str, object]:
    """生产注册表 = `plugin.all_tasks()`；读不到就返回只含哨兵键的表。

    读失败时绝不能退回某个默认卡集继续跑：哨兵键会让
    `registry_problems()` 把"注册表读取失败"写进拒答原因，排查入口不会丢。
    """
    try:
        from dtseek.tasks.plugin import all_tasks

        return dict(all_tasks())
    except Exception as exc:  # noqa: BLE001 —— 任何加载失败都通向拒答，不许抛给上层
        return {REGISTRY_ERROR_KEY: f"{type(exc).__name__}: {exc}"}


def resolve_registry(registry: Registry | None = None) -> Registry:
    """`None` = 用生产注册表；显式传入则原样用（测试 / 审计注入）。"""
    return load_default_registry() if registry is None else registry


def registry_problems(
    registry: Registry, *, info_bits: tuple[InfoBit, ...] = INFO_BITS
) -> list[str]:
    """注册表侧的判不动项：读取失败哨兵 + 每个信息位的卡是否在册。空列表 = 都在册。

    查哪些位由 `info_bits` 决定（默认 `INFO_BITS`，一字不变）；对话层注入的位表
    映射到**现有卡**，这里因此查的是那张表声明的卡。
    """
    bad: list[str] = []
    if REGISTRY_ERROR_KEY in registry:
        bad.append(f"注册表读取失败：{registry[REGISTRY_ERROR_KEY]}")
    for bit in info_bits:
        if bit.card not in registry:
            bad.append(f"信息位 {bit.key}({bit.label}) 的卡 {bit.card!r} 未注册")
    return bad


# ---- 信号校验（fail-closed 的入口） ------------------------------------------

def signals_problems(
    signals: Signals,
    *,
    info_bits: tuple[InfoBit, ...] = INFO_BITS,
    required_bits: Mapping[str, tuple[str, ...]] = REQUIRED_BITS,
    bit_kinds: Mapping[str, str] = BIT_KINDS,
) -> list[str]:
    """信号结构的判不动项（纯结构，不看注册表）。空列表 = 读得到、判得动。

    `info_bits` / `required_bits` / `bit_kinds` 默认即模块级表（默认行为一字不变）；对齐检查
    （必需位声明 vs 位表、位种类声明 vs 位表、info 位数）都按**传入的那张表**判 ——
    表对不齐 ⇒ 判不动 ⇒ 拒答。§3.1：位种类（抽取/检测）决定 `无` 的语义，
    没声明种类的位等于"判不动它是什么"，必须 fail-closed 而不是默认当抽取类。
    """
    if not isinstance(signals, Signals):
        return [f"signals 不是 Signals：{type(signals).__name__}"]
    bad: list[str] = []
    if signals.type not in INPUT_TYPES:
        bad.append(f"type={signals.type!r} 不在 {INPUT_TYPES}")
    else:
        declared = required_bits.get(signals.type, ())
        checked = required_bit_indices(
            signals.type, info_bits=info_bits, required_bits=required_bits
        )
        if len(checked) != len(declared):
            bad.append(f"REQUIRED_BITS[{signals.type!r}] 声明了不存在的信息位：{declared}")
    for bit in info_bits:
        kind = bit_kinds.get(bit.key)
        if kind is None:
            bad.append(f"信息位 {bit.key}({bit.label}) 没有声明种类：判不动")
        elif kind not in (KIND_EXTRACT, KIND_DETECT):
            bad.append(
                f"信息位 {bit.key}({bit.label}) 的种类 {kind!r} 不在 "
                f"{{{KIND_EXTRACT!r}, {KIND_DETECT!r}}}：判不动"
            )
    if not isinstance(signals.info, tuple):
        bad.append(f"info={signals.info!r} 不是 tuple（不可哈希 ⇒ 进不了可达集）")
    elif len(signals.info) != len(info_bits):
        bad.append(f"info 位数={len(signals.info)}，必须 {len(info_bits)}")
    else:
        bad += [
            f"info[{i}]={v!r} 不是 InfoState"
            for i, v in enumerate(signals.info)
            if not isinstance(v, InfoState)
        ]
    if not isinstance(signals.conf, Conf):
        bad.append(f"conf={signals.conf!r} 不是 Conf 三档")
    if (
        isinstance(signals.steps, bool)
        or not isinstance(signals.steps, int)
        or signals.steps not in STEPS_VALUES
    ):
        bad.append(f"steps={signals.steps!r} 不在 {STEPS_VALUES}")
    return bad


def dispatch_problems(
    signals: Signals,
    registry: Registry,
    *,
    info_bits: tuple[InfoBit, ...] = INFO_BITS,
    required_bits: Mapping[str, tuple[str, ...]] = REQUIRED_BITS,
    bit_kinds: Mapping[str, str] = BIT_KINDS,
) -> list[str]:
    """一次决策的全部判不动项：信号结构 + 注册表。任一非空 ⇒ 拒答。"""
    return signals_problems(
        signals, info_bits=info_bits, required_bits=required_bits, bit_kinds=bit_kinds
    ) + registry_problems(registry, info_bits=info_bits)


# ---- 规则策略 π（dev-notes/16 §2.1 那张表，逐行照抄） ------------------------

def _authorized(
    signals: Signals,
    registry: Registry,
    *,
    info_bits: tuple[InfoBit, ...] = INFO_BITS,
    required_bits: Mapping[str, tuple[str, ...]] = REQUIRED_BITS,
    bit_kinds: Mapping[str, str] = BIT_KINDS,
) -> tuple[Action, ...]:
    """规则表 if-else 链：返回**第一个命中行**授权的动作集合（永不为空）。

    **规格要求（dev-notes/16 §2.1 + §2.2 D2），不是实现细节**：
    可行动作 = `admissible(s)`，规则表按状态**收窄**动作集，只授权调「未查」**必需位**
    对应的卡；可行动作不等于全局动作空间 A。理由（§2.2 D2，实测）：把整个动作空间
    当可行动作，缺位态可以经「重调 slot → 调 intent」把必需位洗成全有，
    验证点 3（拒答受控）当场不成立。

    行序（§2.2 D3：**缺位先拒答** —— 缺位意味着必需信息已确定拿不到，
    继续调别的卡无意义，拒答要早）：

      R0  信号读不到/判不动（含位种类未声明） ⇒ 拒答
      R1  type == unknown                  ⇒ 拒答（类型不确定不许猜）
      Rg  映射声明的卡未注册                ⇒ 拒答（fail-closed，与注册表总数无关）
      R2  存在**抽取类**"无"位               ⇒ 拒答（缺必需信息）——先于 R3
      R3  存在"未查"必需位且 steps > 0       ⇒ 调该位对应的卡
      R4  必需位**全部判定过**且 conf ≥ 中    ⇒ 生成输出
      R5  必需位**全部判定过**且 conf == 低   ⇒ 降级为指针输出
      R6  steps == 0 且未完成               ⇒ 拒答（不许无限调用）

    §3.1 的两条订正（位种类表 `bit_kinds`，默认全 `extract` ⇒ 行为一字不变）：

    1. **`required` 的语义 = 「必须被判定过」**，不是"必须为 `有`"：`未查` ⇒ R3 去调卡；
       抽取类位的 `无` 已在 R2 拒答（信息确实不存在），所以走到 R4/R5 时抽取位只可能是
       `有`；**检测类位的 `无` = 判否，是有效结论** ⇒ 算"判定过"，参与 R4/R5 而不是拒答。
    2. **「读不到 / 调用失败」与上面两种分开**：位种类没声明属于"判不动" ⇒ R0 拒答；
       真实调用失败不进状态机（它在规则表里唯一出口就是拒答），由对话层在**调用出口**
       fail-closed 拒答 —— 与"判否"走的是完全不同的两条路。

    必需位按 type 声明（§2.2 D5，表在 `REQUIRED_BITS`）：plain=intent、
    cloze=object、candidates=intent+object、multi_turn=四位全要、unknown=无（走 R1）。
    非必需位不驱动规则；R2 看全部位，与"只看必需位"在可达集上等价 ——
    非必需位与必需位共卡（D6），没有调用就永远停在"未查"，不会单独变"无"。

    R6 是兜底行。推导：走到 R3 之后已无"抽取类无"位（R2 挡过）且 type 已知、卡都在册；
    若必需位无"未查"则必需位全部判定过 ⇒ R4/R5；否则 R3 没命中只能是 steps==0 ⇒ R6。
    所以规则表自身完备 —— 没有哪个状态会落进空分支。

    `info_bits` / `required_bits` / `bit_kinds` 默认即模块级表（默认行为一字不变）；
    对话层注入映射到现有卡的位表时，**行序与门禁一字不改**，只换数据。
    """
    problems = signals_problems(
        signals, info_bits=info_bits, required_bits=required_bits, bit_kinds=bit_kinds
    )
    if problems:
        return (Terminate(REJECT, "信号读不到/判不动：" + "；".join(problems)),)
    if signals.type == UNKNOWN_TYPE:
        return (Terminate(REJECT, "type=unknown：类型不确定不许猜"),)
    reg_problems = registry_problems(registry, info_bits=info_bits)
    if reg_problems:
        return (Terminate(REJECT, "fail-closed（不许静默跳过信息位）：" + "；".join(reg_problems)),)

    # R2（§2.2 D3：缺位先拒答，先于"调卡"；§3.1：只有**抽取类**的"无"才叫缺信息 ——
    # 检测类的"无"是判否，是有效结论，放行走 R4/R5）
    if any(
        state is InfoState.ABSENT
        and bit_kinds[info_bits[i].key] == KIND_EXTRACT
        for i, state in enumerate(signals.info)
    ):
        return (Terminate(REJECT, "存在 info 位=无：缺必需信息"),)

    required = required_bit_indices(signals.type, info_bits=info_bits, required_bits=required_bits)
    unchecked = [i for i in required if signals.info[i] is InfoState.UNCHECKED]
    if unchecked and signals.steps > 0:
        actions: list[Action] = []
        seen: set[str] = set()
        for i in unchecked:  # 按位序，去重到"卡"：调用的单位是卡，不是位
            card = info_bits[i].card
            if card not in seen:
                seen.add(card)
                actions.append(CallCard(card, info_bits[i].key))
        return tuple(actions)
    judged = [
        state is InfoState.HAVE
        or (
            state is InfoState.ABSENT
            and bit_kinds[info_bits[i].key] == KIND_DETECT  # 检测位判否 = 判定过
        )
        for i, state in enumerate(signals.info)
        if i in required
    ]
    if all(judged):  # required 为空时与旧口径一致（all(空) = True ⇒ 照样出输出）
        # reason 的两种说法只在"含检测位判否"时分岔；默认（全抽取类）永远走前一种，
        # 文案与 §2.2 原表逐字相同。
        judged_note = (
            "必需位全部判定过（检测位判否=有效结论）"
            if any(signals.info[i] is InfoState.ABSENT for i in required)
            else "必需位全=有"
        )
        if CONF_RANK[signals.conf] >= CONF_RANK[Conf.MID]:
            return (Terminate(GENERATE, f"{judged_note} 且 conf ≥ 中 ⇒ 生成输出"),)
        return (Terminate(POINTER, f"{judged_note} 但 conf == 低 ⇒ 降级为指针输出"),)
    return (Terminate(REJECT, "steps=0 且未完成：不许无限调用"),)


def admissible_actions(
    signals: Signals,
    *,
    registry: Registry | None = None,
    info_bits: tuple[InfoBit, ...] = INFO_BITS,
    required_bits: Mapping[str, tuple[str, ...]] = REQUIRED_BITS,
    bit_kinds: Mapping[str, str] = BIT_KINDS,
) -> tuple[Action, ...]:
    """该状态下规则策略**授权**的动作集合 —— §2.2 D2 把它写成规格：可行动作 = `admissible(s)`。

    与"行动空间"的区别：行动空间 = 位表**声明**的卡（与注册表里的卡总数
    无关，§2.2 D9）+ 终止三选一；**授权**只有第一个命中行那些 —— 比如 type=unknown
    时授权集只有拒答。把整个行动空间当可行动作，"缺位只能到拒答"立刻不成立
    （重调那张卡可以把"无"救回"有"），验证点 3 就废了。
    """
    return _authorized(
        signals,
        resolve_registry(registry),
        info_bits=info_bits,
        required_bits=required_bits,
        bit_kinds=bit_kinds,
    )


def decide(
    signals: Signals,
    *,
    registry: Registry | None = None,
    info_bits: tuple[InfoBit, ...] = INFO_BITS,
    required_bits: Mapping[str, tuple[str, ...]] = REQUIRED_BITS,
    bit_kinds: Mapping[str, str] = BIT_KINDS,
) -> Action:
    """规则策略 π：每个状态**唯一**动作（授权集的第一个元素）。

    确定性来自"纯函数 + 固定行序 + 固定取位序"，不来自任何随机或模型判断 ⇒
    同一 (type, info, conf, steps) 两次 `plan()` 逐字相同。
    """
    return _authorized(
        signals,
        resolve_registry(registry),
        info_bits=info_bits,
        required_bits=required_bits,
        bit_kinds=bit_kinds,
    )[0]


# ---- 抽象转移 P（非确定，不接真卡） ------------------------------------------

def possible_outcomes(
    signals: Signals,
    action: Action,
    *,
    registry: Registry | None = None,
    info_bits: tuple[InfoBit, ...] = INFO_BITS,
    bit_kinds: Mapping[str, str] = BIT_KINDS,
) -> set[Signals]:
    """抽象（非确定）转移：在该状态执行 `action` 后**可能**到达的状态集合。不接真卡。

    规格（§2.1 + §2.2 D4）：调一张卡后，它服务的信息位可能变"有"或"无"，
    conf 换档，steps 减一。

    - 位变**有** ⇒ 该卡服务的位全部离开"未查"，conf 取这次调用的硬阈值档
      （低/中/高三种后继都可能 —— 这就是"非确定"，概率留待用数据估；
      即"变有 ⇒ 可能升到 mid/high"）；
    - 位变**无** ⇒ 该位组变缺位；这次没产出 ⇒ 无置信度可读 ⇒ conf 取最低档
      （fail-closed：不许拿旧的高置信度给一个失败的调用撑腰）；
    - 终止动作 / 信号判不动 / 卡未注册 / 不对应任何信息位 / steps 已为 0
      ⇒ **空集**：没有后继，而不是"当没事发生"继续走。

    conf 转移是**规格必写项**（§2.2 D4）：只写位与 steps 的转移、不写 conf 的话
    conf 恒为初值 LOW ⇒ R4（生成输出）永不可达，生成输出变成死路。

    TODO(补卡时)：`InfoState.HAVE`（"有"）目前**按卡判定** —— 卡被调过且产出了切片，
    它服务的**全部**位一起离开"未查"（共卡的位同生共死，名义 4 位、独立自由度只有 2）。
    这是临时简化；补卡时改成**按位**，前提是卡能声明"我覆盖了哪一位"
    （`INFO_BITS` 加字段，"有" 只推被声明覆盖的那位）。

    重复调同一张卡会把它服务的位**重新判定**；规则授权只调"未查"必需位的卡，
    所以这种后继进不了真实可达集 —— 这里仍然如实给出（抽象模型不替策略兜底）。

    位的种类（`bit_kinds`）**不改变后继集合本身**：抽取类与检测类的"无"都是
    「调过、没产出切片」这同一个可观测事实 —— 分岔发生在**规则层**（R2 拒答还是
    放行），不在转移层。§3.1 的第 4 态"调用失败"也不在这里：失败在调用出口就拒答了，
    抽象转移只描述"调用成功返回"的世界。
    """
    reg = resolve_registry(registry)
    if not isinstance(action, CallCard):
        return set()
    if signals_problems(signals, info_bits=info_bits, bit_kinds=bit_kinds):
        return set()
    if action.card not in reg:
        return set()
    served = tuple(i for i, bit in enumerate(info_bits) if bit.card == action.card)
    if not served:
        return set()
    if signals.steps == 0:
        return set()

    rest = signals.steps - 1

    def with_bits(state: InfoState, conf: Conf) -> Signals:
        info = tuple(state if i in served else v for i, v in enumerate(signals.info))
        return replace(signals, info=info, conf=conf, steps=rest)

    out = {with_bits(InfoState.HAVE, conf) for conf in Conf}
    out.add(with_bits(InfoState.ABSENT, Conf.LOW))
    return out


# ---- 可达性 / 死状态 / 拒答受控（可判代理） ----------------------------------

def reachable_states(
    *,
    registry: Registry | None = None,
    info_bits: tuple[InfoBit, ...] = INFO_BITS,
    required_bits: Mapping[str, tuple[str, ...]] = REQUIRED_BITS,
    bit_kinds: Mapping[str, str] = BIT_KINDS,
) -> set[Signals]:
    """**真实可达**状态集：从 `initial_states()` 出发，沿"授权动作 × 抽象转移"BFS。

    用授权集（而不是只跟 `decide()` 的单一轨迹）算是更保守的上界：
    连这个集合都 ≤ 4860，策略轨迹自然也 ≤。
    """
    reg = resolve_registry(registry)
    seen: set[Signals] = set(initial_states(info_bits=info_bits))
    queue: deque[Signals] = deque(seen)
    while queue:
        state = queue.popleft()
        for action in _authorized(
            state, reg, info_bits=info_bits, required_bits=required_bits, bit_kinds=bit_kinds
        ):
            for nxt in possible_outcomes(
                state, action, registry=reg, info_bits=info_bits, bit_kinds=bit_kinds
            ):
                if nxt not in seen:
                    seen.add(nxt)
                    queue.append(nxt)
    return seen


def terminal_kinds_reachable(
    signals: Signals,
    *,
    registry: Registry | None = None,
    info_bits: tuple[InfoBit, ...] = INFO_BITS,
    required_bits: Mapping[str, tuple[str, ...]] = REQUIRED_BITS,
    bit_kinds: Mapping[str, str] = BIT_KINDS,
) -> frozenset[str]:
    """从该状态出发、沿授权动作可达的**全部终止动作种类**（拒答受控的判据）。"""
    reg = resolve_registry(registry)
    seen: set[Signals] = set()
    stack = [signals]
    found: set[str] = set()
    while stack:
        state = stack.pop()
        if state in seen:
            continue
        seen.add(state)
        for action in _authorized(
            state, reg, info_bits=info_bits, required_bits=required_bits, bit_kinds=bit_kinds
        ):
            if isinstance(action, Terminate):
                found.add(action.kind)
            else:
                stack.extend(
                    possible_outcomes(
                        state, action, registry=reg, info_bits=info_bits, bit_kinds=bit_kinds
                    )
                )
    return frozenset(found)


def dead_states(
    *,
    registry: Registry | None = None,
    info_bits: tuple[InfoBit, ...] = INFO_BITS,
    required_bits: Mapping[str, tuple[str, ...]] = REQUIRED_BITS,
    bit_kinds: Mapping[str, str] = BIT_KINDS,
) -> set[Signals]:
    """**死状态**：可达、但沿授权动作怎么走都到不了任何终止动作的状态。必须为空。"""
    reg = resolve_registry(registry)
    kwargs = {"info_bits": info_bits, "required_bits": required_bits, "bit_kinds": bit_kinds}
    return {
        state
        for state in reachable_states(registry=reg, **kwargs)
        if not terminal_kinds_reachable(state, registry=reg, **kwargs)
    }


# ---- 计划生成 ---------------------------------------------------------------

def plan(
    signals: Signals,
    *,
    registry: Registry | None = None,
    source_span: tuple[int, int] | None = None,
    info_bits: tuple[InfoBit, ...] = INFO_BITS,
    required_bits: Mapping[str, tuple[str, ...]] = REQUIRED_BITS,
    bit_kinds: Mapping[str, str] = BIT_KINDS,
) -> Plan:
    """把规则策略落成可执行计划：`Plan = [Step{module_id, params, input_span}]`（§1）。

    一次只产出**当前这一步**：转移是非确定的（一个动作有多个后继），
    排到终点的"整条计划"根本不存在，排了就是把猜测写死。

    `source_span` 是调用方给的原文区间（没有就传 None）。结构非法 ⇒ 整个计划
    降级为拒答 —— 坏 span 绝不流到执行层。

    `info_bits` / `required_bits` / `bit_kinds` 默认即模块级表（默认行为一字不变）；
    对话层传入映射到现有卡的位表 ⇒ Step.module_id 是那张表声明的**现有卡**。
    """
    reg = resolve_registry(registry)
    action = decide(
        signals,
        registry=reg,
        info_bits=info_bits,
        required_bits=required_bits,
        bit_kinds=bit_kinds,
    )
    if (
        action_kind(action) != REJECT
        and source_span is not None
        and closure_problems((Step("probe", source_span),))
    ):
        action = Terminate(REJECT, f"source_span 结构非法：{source_span!r}")

    if isinstance(action, CallCard):
        serves = "|".join(bit.key for bit in info_bits if bit.card == action.card)
        steps = (
            Step(action.card, source_span, (("trigger_bit", action.bit), ("serves", serves))),
        )
        reason = f"必需位 info[{action.bit}]=未查 且 steps={signals.steps}>0 ⇒ 调用 {action.card}"
    else:
        # 拒答不引用任何区间：它既不消费原文，也不消费上游产出。
        span = None if action.kind == REJECT else source_span
        steps = (Step(f"output.{action.kind}", span, (("reason", action.reason),)),)
        reason = action.reason
    return Plan(signals, action, reason, steps)
