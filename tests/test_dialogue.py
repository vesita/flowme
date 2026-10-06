"""对话路径的五条判据（dev-notes/16 §3「跑前写死」，确定性路径 ⇒ 不用 seed，改用逐字可复现）+ §3.1 的四条规格订正。

| # | 判据 | 门槛 | 测试 |
|---|---|---|---|
| P1 | 端到端可跑 | 一句话 → 一句回复，无异常 | `test_p1_end_to_end_one_turn` |
| P2 | 零幻觉（自动校验） | 回复里的引用片段必须是输入的逐字子串 | `test_p2_zero_hallucination_substrings` |
| P3 | 计划与回复可复现 | 同一输入两次跑，逐字相同 | `test_p3_reproducible_plan_and_reply` |
| P4 | 拒答受控 | type=unknown 或必需卡缺失 ⇒ 拒答，不定输出 | `test_p4_refusal_controlled` |
| P5 | 性能 | 单轮 < 100ms 且只跑计划里的卡 | `test_p5_latency_under_100ms_and_plan_only` |
| S1 | §3.1 干净对照 | **无否定的正常句子必须回复**（不再看检测器脸色） | `test_s1_clean_input_without_negation_replies` |
| S2 | §3.1 抽取位缺仍拒答 | 同一份卡输出，只换位种类表：抽取类 ⇒ reject、检测类 ⇒ 继续 | `test_s2_extraction_absent_refuses_detector_negation_continues` |
| S3 | §3.1 判否 ≠ 失败 | 同一输入两变体：判否继续、调用失败拒答（fail-closed） | `test_s3_judge_no_vs_call_failure_differ` |
| S4 | §3.1 规则行跟种类走 | R2/R4/R5 按 `bit_kinds` 分岔；种类未声明 ⇒ 判不动 | `test_s4_rule_rows_follow_bit_kinds` |

P2 是命门：它把"零幻觉"降成一次字符串包含检查（evidence 片段 / 回复里的『…』
逐个拿去 `in` 输入），**可进 CI，不靠人看**。
"""
from __future__ import annotations

import re
import statistics
import time

import pytest

from dtseek.tasks.dialogue import (
    DEFAULT_ATTACH,
    DIALOGUE_BIT_KINDS,
    DIALOGUE_BITS,
    DIALOGUE_REQUIRED_BITS,
    SUPPORTED_TYPES,
    resolve_type,
    respond,
)
from dtseek.tasks.dispatch import (
    KIND_EXTRACT,
    MAX_STEPS,
    CallCard,
    Conf,
    InfoState,
    Signals,
    action_kind,
    decide,
)
from dtseek.tasks.engine import MultiTaskEngine

#: 两张必需卡都会出切片的输入（情绪 + 否定都命中 ⇒ 有回复）
REPLY_INPUTS = [
    "他不是好人",
    "今天天气不错",
    "她不开心我也不开心",
    "不好看，我不满意",
]

#: §3.1 干净对照：**全句没有「不/没」** ⇒ 否定位必然判否。
#: 修复前它被拒答（"缺必需信息"）——修复后必须回复（S1 的直接靶子）。
CLEAN_INPUT = "今天天气很好。"

#: 情绪有切片、否定没有 ⇒ 否定**判否**（检测类有效结论，§3.1）⇒ 照常回复。
#: （修复前这条被当成"必需位有缺 ⇒ 拒答"的样例 —— 那是规格错，S1/S3 钉住新语义。）
ABSENT_INPUT = "我不认为这是个好主意"

#: 计划外的卡：对话层永远不该跑它们（本轮必需位只有 sentiment + negation）
PLANNED_CARDS = ["sentiment", "negation"]

#: 注册表门禁查位表里**全部**卡（裁决 2 的收窄还没落地）⇒ 测试注入要给齐 4 张。
FULL_DIALOGUE_REGISTRY: dict[str, object] = {b.card: object() for b in DIALOGUE_BITS}


def _anchor(s0: int = 0, e0: int = 1, cls: str = "否定", conf: float = 0.9) -> dict:
    """一条合规切片（`compose_reply` 只要 s0/e0/class_name，`conf_of` 要 confidence）。"""
    return {"s0": s0, "e0": e0, "class_name": cls, "confidence": conf}


class _FakeEngine:
    """只造输出的假引擎：decoders 齐全，predict 按卡返回给定切片。

    三种失败形态（§3.1「调用失败」）都从这里出：`raise_exc` 抛错、`error` 返回
    error 字典、`decoders` 少一张 = 未挂载。
    """

    def __init__(
        self,
        *,
        sentiment: list[dict] | None = None,
        negation: list[dict] | None = None,
        raise_exc: Exception | None = None,
        error: str | None = None,
        decoders: dict | None = None,
    ) -> None:
        self.decoders = (
            {"sentiment": object(), "negation": object()}
            if decoders is None
            else decoders
        )
        self._outputs = {
            "sentiment": list(sentiment or []),
            "negation": list(negation or []),
        }
        self._raise = raise_exc
        self._error = error

    def predict(self, text: str, tasks: list[str] | None = None) -> dict:
        if self._raise is not None:
            raise self._raise
        if self._error is not None:
            return {"error": self._error}
        return {"text": text, "tasks": {c: self._outputs.get(c, []) for c in tasks}}


@pytest.fixture(scope="module")
def engine() -> MultiTaskEngine:
    """真基座 + 4 张默认卡 + negation（对话必需位要用，`DEFAULT_ATTACH` 里声明）。"""
    eng = MultiTaskEngine()
    for path in DEFAULT_ATTACH:
        eng.attach(path)
    return eng


# ---- P1 ----------------------------------------------------------------------

def test_p1_end_to_end_one_turn(engine: MultiTaskEngine):
    """一句话进 ⇒ 一句话出，无异常；回复是模板拼出来的一整句。"""
    for text in REPLY_INPUTS:
        rec = respond(engine, text, type_="plain")
        assert rec["kind"] == "text", (text, rec)
        assert rec["text"].endswith("。")
        assert len(rec["text"]) > 10
        assert rec["evidence"], (text, rec)
        assert rec["cards_run"] == PLANNED_CARDS, rec["cards_run"]
        assert rec["terminal"] in ("pointer", "generate")


# ---- P2（命门） ---------------------------------------------------------------

def test_p2_zero_hallucination_substrings(engine: MultiTaskEngine):
    """**自动断言**：evidence 片段与回复里每个『…』都必须是输入的逐字子串。"""
    for text in REPLY_INPUTS:
        rec = respond(engine, text, type_="plain")
        assert rec["kind"] == "text"
        for ev in rec["evidence"]:
            start, end = ev["span"]
            snippet = text[start:end + 1]
            assert snippet, ev
            assert snippet in text, f"片段不是输入的逐字子串：{snippet!r}"
            assert snippet in rec["text"], f"回复没引用该片段：{snippet!r}"
            assert ev["card"] in rec["cards_run"]
            assert ev["class_name"], ev
        quoted = re.findall(r"『(.*?)』", rec["text"])
        assert quoted, rec["text"]
        for frag in quoted:
            assert frag in text, f"回复引用了输入里没有的内容：{frag!r}"
        # evidence 与回复里的『…』一一对应（模板单槽：一张卡一个片段）
        assert len(quoted) == len(rec["evidence"])

    # 拒答不引用任何片段 ⇒ 零幻觉在拒答路径上同样成立（用 unknown-type 拒答验：
    # §3.1 之后"否定判否"不再拒答，拒答样例得换一条真实存在的入口）
    rec = respond(engine, REPLY_INPUTS[0])
    assert rec["kind"] == "reject"
    assert rec["evidence"] == []
    assert not re.findall(r"『(.*?)』", rec["text"])


# ---- P3 ----------------------------------------------------------------------

def test_p3_reproducible_plan_and_reply(engine: MultiTaskEngine):
    """同一输入两次跑：整条记录（回复文本 + 计划 + evidence + 卡序）逐字相同。"""
    for text in REPLY_INPUTS + [ABSENT_INPUT, "不声明 type 的输入"]:
        first = respond(engine, text, type_="plain")
        second = respond(engine, text, type_="plain")
        assert first == second, text
        assert first["plan"] == second["plan"]
        assert first["text"] == second["text"]

    # 计划本身也逐字可复现：状态行是纯函数的产物
    a = respond(engine, REPLY_INPUTS[0], type_="plain")
    b = respond(engine, REPLY_INPUTS[0], type_="plain")
    assert repr(a["plan"]) == repr(b["plan"])


# ---- P4 ----------------------------------------------------------------------

def test_p4_refusal_controlled(engine: MultiTaskEngine):
    """type 未声明 / 不认识 / 不支持 / 必需卡缺失 ⇒ **拒答**，绝不定输出。"""
    # 1) 不声明 type ⇒ unknown ⇒ 拒答（不许猜）
    rec = respond(engine, REPLY_INPUTS[0])
    assert rec["kind"] == "reject"
    assert "type=unknown" in rec["text"]
    assert rec["evidence"] == [] and rec["cards_run"] == []

    # 2) 显式声明 unknown ⇒ 同一条拒答出口
    assert respond(engine, REPLY_INPUTS[0], type_="unknown")["kind"] == "reject"

    # 3) 认识但本轮不支持的 type ⇒ 拒答（不是硬着头皮输出）
    rec = respond(engine, REPLY_INPUTS[0], type_="candidates")
    assert rec["kind"] == "reject"
    assert "本轮只支持" in rec["text"]

    # 4) 乱写的 type ⇒ 归一成 unknown ⇒ 拒答
    assert resolve_type("nonsense") == "unknown"
    assert resolve_type(None) == "unknown" and resolve_type("   ") == "unknown"
    assert resolve_type("plain") == "plain" == SUPPORTED_TYPES[0]

    # 5) 必需卡缺失（注册表里没有 negation）⇒ fail-closed 拒答，一张卡都不调
    registry = {"sentiment": object()}
    rec = respond(engine, REPLY_INPUTS[0], type_="plain", registry=registry)
    assert rec["kind"] == "reject"
    assert "未注册" in rec["text"]
    assert rec["cards_run"] == [] and rec["evidence"] == []

    # 6) 卡在册但引擎没挂载 ⇒ 也是拒答（不许静默跳过必需位）
    class _NoCards:
        def __init__(self) -> None:
            self.decoders: dict = {}

        def predict(self, *args, **kwargs):  # 不该被调到
            raise AssertionError("缺卡时不该调用 predict")

    rec = respond(_NoCards(), REPLY_INPUTS[0], type_="plain")
    assert rec["kind"] == "reject"
    assert "未挂载" in rec["text"]

    # 7) 两张必需卡都判否 ⇒ 没有任何可引用切片 ⇒ 拒答（不编句子；不是"缺必需信息"）
    rec = respond(
        _FakeEngine(), ABSENT_INPUT, type_="plain", registry=FULL_DIALOGUE_REGISTRY
    )
    assert rec["kind"] == "reject"
    assert rec["evidence"] == []
    assert "没有任何可引用的切片" in rec["text"]
    # §3.1 语义对照：真引擎上同一个"否定判否"的输入 ⇒ **回复**，不再被当成缺信息
    assert respond(engine, ABSENT_INPUT, type_="plain")["kind"] == "text"

    # 8) 空输入 ⇒ 拒答
    assert respond(engine, "   ", type_="plain")["kind"] == "reject"


# ---- P5 ----------------------------------------------------------------------

def test_p5_latency_under_100ms_and_plan_only(engine: MultiTaskEngine):
    """单轮 < 100ms（稳态），且**只跑计划里的卡**（绝不全跑已挂的 5 张）。"""
    seen: set[str] = set()
    for card in PLANNED_CARDS:
        seen.add(card)

    respond(engine, REPLY_INPUTS[1], type_="plain")   # 预热（CUDA 首轮编译不计入）

    durations: list[float] = []
    for text in REPLY_INPUTS:
        t0 = time.perf_counter()
        rec = respond(engine, text, type_="plain")
        durations.append((time.perf_counter() - t0) * 1000)
        assert rec["cards_run"] == PLANNED_CARDS, rec["cards_run"]
        assert not (set(rec["cards_run"]) - seen)
        for run in rec["cards_run"]:
            assert run not in ("person", "pronoun", "relation"), rec["cards_run"]

    median = statistics.median(durations)
    assert median < 100, f"单轮中位 {median:.1f}ms ≥ 100ms：{durations}"

    # 位表/必需位表/位种类表的形状钉死：本轮配置改了就得改判据
    assert [b.card for b in DIALOGUE_BITS] == [
        "sentiment", "negation", "pronoun", "person",
    ]
    assert DIALOGUE_REQUIRED_BITS["plain"] == ("mood", "neg")
    assert DIALOGUE_BIT_KINDS == {
        "mood": "detect", "neg": "detect", "pron": "detect", "person": "extract",
    }


# ---- §3.1 规格订正（S1–S4） ---------------------------------------------------

def test_s1_clean_input_without_negation_replies(engine: MultiTaskEngine):
    """S1 干净对照（§3.1 的直接靶子）：全句无「不/没」的正常句子**必须回复**。

    修复前：`今天天气很好。` ⇒ 拒答「存在 info 位=无：缺必需信息」，
    而 `今天天气不错。` 靠「不错」的"不"误触发否定卡才开口 ——
    开不开口取决于检测器有没有恰好误触发。修复后：判否 = 有效结论 ⇒ 照常回复。
    """
    assert not re.search(r"[不没]", CLEAN_INPUT), "对照输入必须真的没有否定词"
    rec = respond(engine, CLEAN_INPUT, type_="plain")
    assert rec["kind"] == "text", rec
    assert rec["terminal"] in ("pointer", "generate")
    assert rec["evidence"] and rec["cards_run"] == PLANNED_CARDS
    for frag in re.findall(r"『(.*?)』", rec["text"]):
        assert frag in CLEAN_INPUT, frag
    # 计划里能看到判否走的是"判定过"而不是"缺信息"（可审）
    assert any("判否" in line for line in rec["plan"]), rec["plan"]
    assert not any("缺必需信息" in line for line in rec["plan"]), rec["plan"]
    # 旧拒答样例（否定判否）同样必须回复
    old = respond(engine, ABSENT_INPUT, type_="plain")
    assert old["kind"] == "text", old


def test_s2_extraction_absent_refuses_detector_negation_continues():
    """S2 真·缺信息仍拒答：同一份卡输出，只换**位种类表** ⇒ 一个拒答、一个继续。

    假引擎：情绪卡成功返回但**没有切片**、否定卡有切片。
    - 把 mood 声明成**抽取类** ⇒ `无` = 该信息确实不存在 ⇒ 缺必需信息 ⇒ reject；
    - 默认（mood=**检测类**） ⇒ `无` = 判否 = 有效结论 ⇒ 继续 ⇒ text。
    分岔只由 `bit_kinds` 一张表决定 —— 这就是 §3.1 的修正点。
    """
    eng = _FakeEngine(sentiment=[], negation=[_anchor(0, 1, "否定")])
    refused = respond(
        eng, CLEAN_INPUT, type_="plain", registry=FULL_DIALOGUE_REGISTRY,
        bit_kinds={**DIALOGUE_BIT_KINDS, "mood": KIND_EXTRACT},
    )
    assert refused["kind"] == "reject", refused
    assert "缺必需信息" in refused["text"]
    assert refused["evidence"] == [] and not re.findall(r"『(.*?)』", refused["text"])

    continued = respond(eng, CLEAN_INPUT, type_="plain", registry=FULL_DIALOGUE_REGISTRY)
    assert continued["kind"] == "text", continued
    assert "『今天』" in continued["text"]
    # 同一个引擎、同一个输入：两张表的行为必须**不同**
    assert refused["kind"] != continued["kind"]


def test_s3_judge_no_vs_call_failure_differ(engine: MultiTaskEngine):
    """S3 判否 ≠ 失败：同一输入的两个变体 —— 判否**继续**、调用失败**拒答**（fail-closed）。

    变体 A（判否）：真引擎跑同一句话，negation 卡成功返回、没有切片 ⇒ 回复。
    变体 B（失败）：同一句话，卡分别以 抛错 / error 字典 / 未挂载 三种形态失败
    ⇒ 三种全部拒答，且与 A 的行为**不同**（§3.1：失败不许与判否混成一个「无」）。
    """
    judged_no = respond(engine, CLEAN_INPUT, type_="plain")
    assert judged_no["kind"] == "text", judged_no

    boom = respond(
        _FakeEngine(raise_exc=RuntimeError("显存炸了")),
        CLEAN_INPUT, type_="plain", registry=FULL_DIALOGUE_REGISTRY,
    )
    assert boom["kind"] == "reject", boom
    assert "调卡失败" in boom["text"] and "显存炸了" in boom["text"]

    errored = respond(
        _FakeEngine(error="decoder 爆了"),
        CLEAN_INPUT, type_="plain", registry=FULL_DIALOGUE_REGISTRY,
    )
    assert errored["kind"] == "reject", errored
    assert "调卡失败" in errored["text"] and "decoder 爆了" in errored["text"]

    unmounted = respond(
        _FakeEngine(decoders={"sentiment": object()}),
        CLEAN_INPUT, type_="plain", registry=FULL_DIALOGUE_REGISTRY,
    )
    assert unmounted["kind"] == "reject", unmounted
    assert "未挂载" in unmounted["text"]

    # 行为必须可区分：判否出 text（跑满计划卡），失败出 reject（一张卡都没跑成）
    assert judged_no["kind"] == "text" and judged_no["cards_run"] == PLANNED_CARDS
    for failed in (boom, errored, unmounted):
        assert failed["kind"] == "reject" and failed["evidence"] == []
        assert failed["kind"] != judged_no["kind"]


def test_s4_rule_rows_follow_bit_kinds():
    """S4 规则行按位种类分岔（dispatch 规则层，注入对话层的表 ⇒ 默认行为不受影响）。"""
    kw = {
        "registry": FULL_DIALOGUE_REGISTRY,
        "info_bits": DIALOGUE_BITS,
        "required_bits": DIALOGUE_REQUIRED_BITS,
    }
    #: mood=有、neg=判否（无）、其余未查 —— §3.1 讨论的那个状态
    judged_no = Signals(
        "plain",
        (InfoState.HAVE, InfoState.ABSENT, InfoState.UNCHECKED, InfoState.UNCHECKED),
        Conf.MID, MAX_STEPS,
    )
    # 检测位判否 ⇒ 不拒答：conf ≥ 中 ⇒ 生成，reason 点名"判否"
    a = decide(judged_no, bit_kinds=DIALOGUE_BIT_KINDS, **kw)
    assert action_kind(a) == "generate" and "判否" in a.reason, a
    # conf == 低 ⇒ 降级指针，同样不是拒答
    a = decide(Signals(judged_no.type, judged_no.info, Conf.LOW, MAX_STEPS),
               bit_kinds=DIALOGUE_BIT_KINDS, **kw)
    assert action_kind(a) == "pointer" and "判否" in a.reason, a
    # 同一状态、只把判否的 neg 换成抽取类 ⇒ `无` 变成缺必需信息 ⇒ 拒答
    a = decide(judged_no, bit_kinds={**DIALOGUE_BIT_KINDS, "neg": KIND_EXTRACT}, **kw)
    assert action_kind(a) == "reject" and "缺必需信息" in a.reason, a
    # required = 「必须被判定过」：未查 ⇒ 去调卡，不是"必须为有"
    start = Signals("plain", (InfoState.UNCHECKED,) * 4, Conf.LOW, MAX_STEPS)
    assert decide(start, bit_kinds=DIALOGUE_BIT_KINDS, **kw) == CallCard("sentiment", "mood")
    # 种类没声明 / 取值未知 ⇒ 判不动 ⇒ 拒答（fail-closed，不许默认当抽取类）
    a = decide(judged_no, bit_kinds={}, **kw)
    assert action_kind(a) == "reject" and "没有声明种类" in a.reason, a
    a = decide(judged_no, bit_kinds={**DIALOGUE_BIT_KINDS, "mood": "kind??"}, **kw)
    assert action_kind(a) == "reject" and "判不动" in a.reason, a
    # 默认表（全抽取类）分毫不动：抽取位=无 ⇒ 仍是"缺必需信息"拒答
    a = decide(
        Signals("plain", (InfoState.UNCHECKED, InfoState.ABSENT,
                          InfoState.UNCHECKED, InfoState.UNCHECKED), Conf.MID, MAX_STEPS),
        registry={"slot": object(), "intent": object()},
    )
    assert action_kind(a) == "reject" and "缺必需信息" in a.reason, a
