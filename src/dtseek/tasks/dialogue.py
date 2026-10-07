"""单轮对话落地：dev-notes/16 §3 的四段 —— **一句话进，一句话出**。

四段与实现一一对应（每段一个函数，顺序即执行顺序）：

  1. 声明 type：`resolve_type()` —— 调用方给（CLI `--type`）；**不给/不认识 ⇒ `unknown`**，
     unknown 由 dispatch 的 R1 拒答。**不许猜**（对抗集上类型判别只有 29.78%，§11.1）。
  2. 出计划：`turn_plan()` —— dispatch 的规则策略，注入 `DIALOGUE_BITS` 这张
     **映射到现有卡**的位表（情绪=sentiment、否定=negation、人称=pronoun、人物=person）
     和 `DIALOGUE_BIT_KINDS` 位种类表（§3.1：情绪/否定/人称=检测类，人物=抽取类）；
     必需位表 `DIALOGUE_REQUIRED_BITS`：plain = 情绪 + 否定。计划逐字可复现（纯函数）。
  3. 调卡：`run_cards()` —— `engine.predict(text, tasks=计划里的卡)`，**只跑计划里的卡**。
  4. 拼回复：`compose_reply()` —— 手写模板表 + 输入里的锚点片段 ⇒
     `{kind:"text", text, evidence:[{card, span, class_name}]}`。

**零幻觉的来源**：现有卡全是指针型，只发射**区间**；回复的内容词因此只能是
输入的逐字子串（`compose_reply` 越界即拒答），句子骨架来自有限、手写的 `CARD_TEMPLATES`
—— 代码，不是模型输出 ⇒ 可审、可穷举、可 diff。

本轮最小配置：**只支持 `type=plain`**，必需位 = 情绪 + 否定（都是**检测类**：
判否 = 有效结论 ⇒ 继续，不因"没有否定"被拒 —— dev-notes/16 §3.1）。
三类出口互不混淆：抽取类必需位 `无` ⇒ 缺必需信息 ⇒ 拒答；
检测类判否 ⇒ 继续；调用失败（抛错/报错/未挂载）⇒ 拒答（fail-closed）。

用法::

    rec = respond(engine, "他不是好人", type_="plain")
    rec["text"]     # 一句话回复；kind=="reject" 时是拒答
    rec["evidence"] # [{card, span, class_name}]，span 落在输入上
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from dtseek.tasks.dispatch import (
    INPUT_TYPES,
    KIND_DETECT,
    KIND_EXTRACT,
    MAX_STEPS,
    REJECT,
    UNKNOWN_TYPE,
    CallCard,
    Conf,
    InfoBit,
    InfoState,
    Plan,
    Signals,
    Terminate,
    admissible_actions,
)
from dtseek.tasks.dispatch import (
    plan as dispatch_plan,
)

__all__ = [
    "CARD_TEMPLATES",
    "CONF_HIGH",
    "CONF_MID",
    "DEFAULT_ATTACH",
    "DIALOGUE_BITS",
    "DIALOGUE_BIT_KINDS",
    "DIALOGUE_REQUIRED_BITS",
    "SUPPORTED_TYPES",
    "compose_reply",
    "observe",
    "resolve_type",
    "respond",
    "run_cards",
    "turn_plan",
]

# ---- 位表（dev-notes/16 §3 第 2 段：映射到**现有**卡） ----------------------

DIALOGUE_BITS: tuple[InfoBit, ...] = (
    InfoBit("mood", "情绪", "sentiment"),
    InfoBit("neg", "否定", "negation"),
    InfoBit("pron", "人称", "pronoun"),
    InfoBit("person", "人物", "person"),
)

#: 本轮最小配置的必需位：plain = 情绪 + 否定；unknown 无必需位（走 R1 拒答）。
#: 位表与必需位表一起注入 dispatch ⇒ 规则表行序与门禁一字不改，只换数据。
DIALOGUE_REQUIRED_BITS: dict[str, tuple[str, ...]] = {
    "plain": ("mood", "neg"),
    UNKNOWN_TYPE: (),
}

#: 位的**种类**（dev-notes/16 §3.1）：`无` 的语义按种类分岔 ——
#: 检测类（情绪/否定/人称）`无` = **判否**，是有效结论 ⇒ 继续；
#: 抽取类（人物）`无` = 该信息确实不存在 ⇒ 缺必需信息 ⇒ 拒答。
#: 「调用失败 / 读不到」既不是 `无` 也不是「未查」：它不进状态，
#: 由 `respond()` 在调用出口 fail-closed 拒答（§3.1 四态的等价实现，|S| 不变）。
DIALOGUE_BIT_KINDS: dict[str, str] = {
    "mood": KIND_DETECT,
    "neg": KIND_DETECT,
    "pron": KIND_DETECT,
    "person": KIND_EXTRACT,
}

#: 本轮只支持的 type；别的合法 type（candidates/cloze/multi_turn）也在对话层拒答 ——
#: 它们在这张位表上没有必需位声明，放行等于"没有任何必需信息也要开口"。
SUPPORTED_TYPES: tuple[str, ...] = ("plain",)

#: negation 不在 `checkpoints/cards/`（默认只挂 4 张），对话必需位要用它 ⇒ 由示例/测试补挂。
DEFAULT_ATTACH: tuple[str, ...] = ("experiments/fix_negation/cards/negation_r2_s42.pt",)

# ---- 模板表（手写、有限、可穷举） -------------------------------------------

#: 卡 → 句子骨架。`{span}` 是输入的逐字子串，`{label}` 是锚点的**规范类名**（class_name，
#: 不是带副标题的 display —— dev-notes/16 §5 的类别名统一）。骨架本身是代码，不是模型输出。
CARD_TEMPLATES: dict[str, str] = {
    "sentiment": "你这句话里的『{span}』表达了{label}",
    "negation": "并且有否定成分『{span}』",
}

REPLY_JOIN = "；"
REPLY_END = "。"

#: 该卡最近一次调用的置信度硬阈值化成 conf 三档（§2.2 D4）。
#: 阈值写死在这里：本层不读 val 分位（那要额外产物），所以宁可保守 ——
#: LOW 只会把终点降级成 pointer，绝不把拒答变成放行。
CONF_MID = 0.5
CONF_HIGH = 0.8


# ---- 第一段 · 声明 type ------------------------------------------------------

def resolve_type(declared: str | None) -> str:
    """调用方声明的 type；空/不认识 ⇒ `unknown`（**不猜**，由 dispatch R1 拒答）。"""
    if declared is None:
        return UNKNOWN_TYPE
    text = declared.strip()
    if not text:
        return UNKNOWN_TYPE
    return text if text in INPUT_TYPES else UNKNOWN_TYPE


# ---- 第二段 · 出计划 ---------------------------------------------------------

def turn_plan(
    signals: Signals,
    *,
    registry: Any = None,
    source_span: tuple[int, int] | None = None,
    info_bits: tuple[InfoBit, ...] = DIALOGUE_BITS,
    required_bits: dict[str, tuple[str, ...]] = DIALOGUE_REQUIRED_BITS,
    bit_kinds: dict[str, str] = DIALOGUE_BIT_KINDS,
) -> Plan:
    """dispatch 规则策略出**当前这一步**的计划，位表/必需位表/位种类表换成对话层那张。"""
    return dispatch_plan(
        signals,
        registry=registry,
        source_span=source_span,
        info_bits=info_bits,
        required_bits=required_bits,
        bit_kinds=bit_kinds,
    )


def conf_of(anchors: list[dict]) -> Conf:
    """一次调用的 conf 硬阈值档：无切片 ⇒ LOW（读不到正面证据就取最保守档）。

    §3.1 口径：这里的"无切片"是**调用成功后的判定结果**（含检测类判否），
    不是"失败的调用" —— 失败根本不进这个函数，`respond()` 在调用出口就拒答了。
    所以判否拿 LOW 不是给失败撑腰，是"没有正面证据 ⇒ 不升 conf"。
    """
    if not anchors:
        return Conf.LOW
    top = max(float(a.get("confidence", 0.0)) for a in anchors)
    if top >= CONF_HIGH:
        return Conf.HIGH
    if top >= CONF_MID:
        return Conf.MID
    return Conf.LOW


def observe(
    signals: Signals,
    card: str,
    anchors: list[dict],
    *,
    info_bits: tuple[InfoBit, ...] = DIALOGUE_BITS,
) -> Signals:
    """把一张卡的真实输出折回状态：该卡服务的位整体变 有/无，conf 取本次档，steps 减一。

    「按卡判有」与 `dispatch.possible_outcomes()` 的抽象转移同一口径（共卡的位同生共死）。
    `无` 落在哪个语义里由位种类（`DIALOGUE_BIT_KINDS`）决定：检测类 = 判否（有效结论，
    规则层放行），抽取类 = 缺信息（规则层拒答）—— `observe` 只如实记录"调过、没产出"。
    """
    served = [i for i, bit in enumerate(info_bits) if bit.card == card]
    state = InfoState.HAVE if anchors else InfoState.ABSENT
    info = tuple(state if i in served else v for i, v in enumerate(signals.info))
    return replace(
        signals,
        info=info,
        conf=conf_of(anchors) if anchors else Conf.LOW,
        steps=max(0, signals.steps - 1),
    )


# ---- 第三段 · 调卡 -----------------------------------------------------------

def run_cards(engine: Any, text: str, cards: list[str]) -> dict:
    """`engine.predict(text, tasks=计划里的卡)` —— **只跑计划里的卡**，绝不全跑。"""
    return engine.predict(text, tasks=list(cards))


# ---- 第四段 · 拼回复 ---------------------------------------------------------

def compose_reply(text: str, picks: list[tuple[str, dict]]) -> tuple[str, list[dict]]:
    """模板表 + 锚点片段 ⇒ (回复文本, evidence)。**片段必须是输入的逐字子串**。

    越界区间 / 没有模板的卡 / 没有任何可引用切片 ⇒ 抛 `ValueError` ⇒ 调用方拒答
    （fail-closed：判不动就不出口，不落到"随便编一句"）。
    """
    parts: list[str] = []
    evidence: list[dict] = []
    for card, anchor in picks:
        start, end = anchor["s0"], anchor["e0"]
        if not (0 <= start <= end < len(text)):
            raise ValueError(f"锚点区间不合法：{card} ({start},{end})")
        span = text[start : end + 1]
        if not span or span not in text:
            raise ValueError(f"锚点片段不是输入的逐字子串：{card} {span!r}")
        template = CARD_TEMPLATES.get(card)
        if template is None:
            raise ValueError(f"卡 {card} 没有模板：不编句子")
        parts.append(template.format(span=span, label=anchor["class_name"]))
        evidence.append({"card": card, "span": (start, end), "class_name": anchor["class_name"]})
    if not parts:
        raise ValueError("没有任何可引用的切片：不编句子")
    return REPLY_JOIN.join(parts) + REPLY_END, evidence


# ---- 驱动：四段串起来 --------------------------------------------------------

def _record(
    kind: str,
    text: str,
    evidence: list[dict],
    type_: str,
    cards_run: list[str],
    plan_lines: list[str],
    terminal: str,
    reason: str,
) -> dict:
    """统一记录（dev-notes/16 §1）：kind + text + evidence + 可溯源的计划与原因。

    **整条记录都是确定性的**（不含耗时）⇒ 同一输入两次跑 `==` 相等，就是 P3。
    """
    return {
        "kind": kind,
        "text": text,
        "evidence": evidence,
        "type": type_,
        "cards_run": cards_run,
        "plan": plan_lines,
        "terminal": terminal,
        "reason": reason,
    }


def _reject(
    type_: str, cards_run: list[str], plan_lines: list[str], reason: str
) -> dict:
    """拒答出口：`kind="reject"`、evidence 为空、不引用任何区间（也不消费原文）。"""
    return _record("reject", f"（拒答）{reason}", [], type_, cards_run, plan_lines, REJECT, reason)


def respond(
    engine: Any,
    text: str,
    *,
    type_: str | None = None,
    registry: Any = None,
    info_bits: tuple[InfoBit, ...] = DIALOGUE_BITS,
    required_bits: dict[str, tuple[str, ...]] = DIALOGUE_REQUIRED_BITS,
    bit_kinds: dict[str, str] = DIALOGUE_BIT_KINDS,
) -> dict:
    """跑一轮对话：声明 type → 出计划 → 只调计划里的卡 → 拼回复（或拒答）。

    拒答的入口（P4）：type 未声明/不认识（=unknown）、type 不在本轮支持集、
    必需卡不在注册表/未挂载、**调用失败（predict 抛错或返回 error）**、
    抽取类必需位 `无`（缺必需信息）、锚点越界。任何一条都**不给内容输出**。

    §3.1 的三类出口互不混淆：

    - **检测类判否**（卡成功返回、没有切片）⇒ 位记「无」= 判定过 ⇒ **继续**；
    - **抽取类 `无`** ⇒ 规则层 R2 拒答（缺必需信息）；
    - **调用失败**（抛错/error/未挂载）⇒ 本函数在**调用出口**拒答（fail-closed）。
      这就是 §3.1 说的第 4 态「失败」的等价实现：它在规则表里没有别的出边
      （唯一出口就是拒答），所以不进 `Signals.info` —— 三态保留、|S| 仍 = 4860，
      但与"判否"走的是完全不同的两条路，行为上可区分（见测试）。
    """
    declared = resolve_type(type_)
    cards_run: list[str] = []
    plan_lines: list[str] = []

    if declared != UNKNOWN_TYPE and declared not in SUPPORTED_TYPES:
        return _reject(
            declared, cards_run, plan_lines,
            f"本轮只支持 type={'/'.join(SUPPORTED_TYPES)}，声明了 {declared!r}：不猜",
        )

    source = (text or "").strip()
    if not source:
        return _reject(declared, cards_run, plan_lines, "输入为空")

    signals = Signals(
        declared,
        (InfoState.UNCHECKED,) * len(info_bits),
        Conf.LOW,
        MAX_STEPS,
    )
    anchors_by_card: dict[str, list[dict]] = {}

    while True:
        current = turn_plan(
            signals,
            registry=registry,
            source_span=(0, len(source)),
            info_bits=info_bits,
            required_bits=required_bits,
            bit_kinds=bit_kinds,
        )
        plan_lines.append(current.describe())
        if isinstance(current.action, Terminate):
            break

        # 授权集 = 这一步计划要调的卡（R3 只授权"未查必需位"的卡）⇒ 只跑这些
        authorized = admissible_actions(
            signals,
            registry=registry,
            info_bits=info_bits,
            required_bits=required_bits,
            bit_kinds=bit_kinds,
        )
        todo = [
            a.card
            for a in authorized
            if isinstance(a, CallCard) and a.card not in anchors_by_card
        ]
        if not todo:
            return _reject(
                declared, cards_run, plan_lines, "计划要求重复调用同一张卡：不许"
            )
        decoders = getattr(engine, "decoders", {})
        missing = [card for card in todo if card not in decoders]
        if missing:
            return _reject(
                declared, cards_run, plan_lines,
                f"必需卡未挂载（引擎里没有）：{missing}：不许静默跳过",
            )

        try:
            result = run_cards(engine, source, todo)
        except Exception as exc:  # noqa: BLE001 —— §3.1「调用失败」：抛错也是失败，
            # fail-closed 拒答（抛给上层 = fail-crash，违反 §2.1 硬要求 2）
            return _reject(
                declared, cards_run, plan_lines,
                f"调卡失败（调用抛错）：{type(exc).__name__}: {exc}",
            )
        if not isinstance(result, dict):
            return _reject(
                declared, cards_run, plan_lines,
                f"调卡失败：predict 返回 {type(result).__name__}，不是 dict",
            )
        if "error" in result:
            return _reject(declared, cards_run, plan_lines, f"调卡失败：{result['error']}")
        for card in todo:
            anchors = result.get("tasks", {}).get(card, [])
            anchors_by_card[card] = anchors
            cards_run.append(card)
            signals = observe(signals, card, anchors, info_bits=info_bits)

    if not isinstance(current.action, Terminate):  # 不可能：循环只在 Terminate 时 break
        return _reject(declared, cards_run, plan_lines, "计划停在非终止动作：判不动")
    action = current.action
    if action.kind == REJECT:
        return _reject(declared, cards_run, plan_lines, action.reason)

    # 每张跑过的卡取**发射序第一个**切片（模板单槽；发射序确定 ⇒ 逐字可复现）
    picks: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for bit in info_bits:  # 按位序：情绪 → 否定 → …
        anchors = anchors_by_card.get(bit.card)
        if anchors and bit.card not in seen:
            seen.add(bit.card)
            picks.append((bit.card, anchors[0]))

    try:
        reply_text, evidence = compose_reply(source, picks)
    except ValueError as exc:
        return _reject(declared, cards_run, plan_lines, str(exc))

    return _record(
        "text", reply_text, evidence, declared, cards_run, plan_lines, action.kind, action.reason
    )
