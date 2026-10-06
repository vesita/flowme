"""dev-notes/16 §2.1 + §2.2 规格订正（D1–D7、D9）的 5 条验证点 + 两条 fail-closed 门禁。

注册表的事实（本文件所有测试都建立在它上面）：生产注册表 `plugin.all_tasks()` 里
**没有** `slot` / `intent` 两张卡 —— 它们是 dev-notes/16 §2 第二层"要补的卡"，
也是 `INFO_BITS` 声明的注册名契约（§2.2 D7）。所以验证点用注入的完整注册表
`FULL_REGISTRY` 跑状态机本体；fail-closed 用生产注册表 / 残缺注册表 / 读失败哨兵跑，
补卡之前必须全线拒答（fail-closed 的**正确**行为，不是缺陷）。
"""
from __future__ import annotations

from collections import deque
from dataclasses import replace

import pytest

from dtseek.tasks.dispatch import (
    GENERATE,
    INPUT_TYPES,
    MAX_STEPS,
    POINTER,
    REGISTRY_ERROR_KEY,
    REQUIRED_BITS,
    STATE_SPACE_SIZE,
    STEPS_VALUES,
    UNKNOWN_TYPE,
    CallCard,
    Conf,
    InfoState,
    Signals,
    Step,
    Terminate,
    action_kind,
    admissible_actions,
    closure_problems,
    dead_states,
    decide,
    enumerate_state_space,
    initial_states,
    plan,
    possible_outcomes,
    reachable_states,
    registry_problems,
    required_bit_indices,
    signals_problems,
    terminal_kinds_reachable,
)

#: 测试用"完整"注册表：`INFO_BITS` 声明的两张卡都在册。
FULL_REGISTRY: dict[str, object] = {"slot": object(), "intent": object()}

UNCHECKED, HAVE, ABSENT = InfoState.UNCHECKED, InfoState.HAVE, InfoState.ABSENT


def _policy_reachable(registry) -> set[Signals]:
    """只跟 `decide()` 单条轨迹的可达集（`reachable_states` 是它的保守上界）。"""
    seen = set(initial_states())
    queue = deque(seen)
    while queue:
        state = queue.popleft()
        nxt = possible_outcomes(state, decide(state, registry=registry), registry=registry)
        for child in nxt:
            if child not in seen:
                seen.add(child)
                queue.append(child)
    return seen


def _shortest_path_to(target: str, registry, start: Signals) -> list[Signals]:
    """策略 `decide()` 单轨迹上、`start` → 终止动作 `target` 的最短状态序列（BFS）。"""
    parent: dict[Signals, Signals | None] = {start: None}
    queue: deque[Signals] = deque([start])
    while queue:
        state = queue.popleft()
        action = decide(state, registry=registry)
        if isinstance(action, Terminate) and action.kind == target:
            path: list[Signals] = []
            cur: Signals | None = state
            while cur is not None:
                path.append(cur)
                cur = parent[cur]
            return list(reversed(path))
        for nxt in possible_outcomes(state, action, registry=registry):
            if nxt not in parent:
                parent[nxt] = state
                queue.append(nxt)
    return []


# ---- 验证点 1：状态数 -------------------------------------------------------

def test_state_space_is_fully_enumerable():
    """§2.1 + D1 的硬指标：|S| = 5 × 3⁴ × 3 × 4 = 4860，可完全枚举、无重复。"""
    assert STEPS_VALUES == (0, 1, 2, 3), "D1：steps 取值域就是 {0,1,2,3}"
    space = list(enumerate_state_space())
    assert len(space) == STATE_SPACE_SIZE == 4860 == 5 * 3**4 * 3 * 4
    assert len(set(space)) == len(space), "枚举不许出现重复状态"
    assert set(reachable_states(registry=FULL_REGISTRY)) <= set(space)


def test_v1_reachable_state_count_within_bound():
    """验证点 1：枚举**真实可达**状态数并打印，必须 ≤ 4860。"""
    space = set(enumerate_state_space())
    reach = reachable_states(registry=FULL_REGISTRY)
    policy = _policy_reachable(FULL_REGISTRY)
    print(
        f"[v1] 全状态空间={len(space)} 真实可达={len(reach)}(授权集)/{len(policy)}(策略轨迹)"
        f" 上限={STATE_SPACE_SIZE}"
    )
    assert reach <= space, "可达集必须是全状态空间的子集"
    assert len(reach) <= STATE_SPACE_SIZE
    # 钉住实测数：改位→卡映射、改必需位表(D5)或改规则行序(D3)都会改它，改了必须说明理由。
    assert len(reach) == 39, f"可达状态数变了（当前 {len(reach)}）：映射/必需位表/行序动过？"
    assert len(policy) == 29, f"策略轨迹状态数变了（当前 {len(policy)}）"
    assert all(signals_problems(s) == [] for s in reach), "可达状态必须全都读得动、判得动"


# ---- 验证点 2：无死状态 -----------------------------------------------------

def test_v2_no_dead_states():
    """验证点 2：每个可达状态都存在动作序列能到某个终止动作（拒答/指针/生成）。"""
    reach = reachable_states(registry=FULL_REGISTRY)
    dead = dead_states(registry=FULL_REGISTRY)
    kinds: set[str] = set()
    for state in reach:
        kinds |= set(terminal_kinds_reachable(state, registry=FULL_REGISTRY))
    print(f"[v2] 可达={len(reach)} 死状态={len(dead)} 可达终止种类={sorted(kinds)}")

    assert dead == set(), f"死状态：{[s.describe() for s in dead]}"
    # 每个可达状态至少有一个授权动作（规则表永不空转），且策略不越出授权集。
    for state in reach:
        allowed = admissible_actions(state, registry=FULL_REGISTRY)
        assert allowed, f"空授权集：{state.describe()}"
        assert decide(state, registry=FULL_REGISTRY) in allowed
    # 三种终止都真可达 —— 这台机器不是"只会拒答"。
    assert kinds == {"reject", "pointer", "generate"}


# ---- D1：generate / pointer 可达（steps ∈ {0,1,2,3} 的目的） ----------------

def test_d1_generate_and_pointer_reachable():
    """D1 的目的：steps ∈ {0,1,2,3} + 位→卡映射允许非单射（一张卡供多个位）
    ⇒ 每类 type 都能真正走到 generate / pointer，而不是被 steps 兜底成拒答。

    最短成功路径 = 1 次调卡 + 1 次终止（plain 只必需 intent 一位）；
    四位全必需的 multi_turn 走满 2 次调卡（slot → intent）也仍在 MAX_STEPS 之内。
    """
    reg = FULL_REGISTRY
    steps_needed: dict[str, int] = {}
    for start in initial_states():
        if start.type == UNKNOWN_TYPE:
            continue
        for kind in (GENERATE, POINTER):
            path = _shortest_path_to(kind, reg, start)
            assert path, f"{start.type} 走不到 {kind}：D1 白改了"
            calls = len(path) - 1
            steps_needed[start.type] = max(steps_needed.get(start.type, 0), calls)
            assert 1 <= calls <= MAX_STEPS, f"{start.type}→{kind} 调卡 {calls} 次"
            assert isinstance(decide(path[0], registry=reg), CallCard)
            assert decide(path[-1], registry=reg).kind == kind
    print(
        "[d1] 各 type 到成功终止所需调卡次数=" + str(steps_needed)
        + f"（MAX_STEPS={MAX_STEPS}）\n"
        + "  最短成功路径(generate): "
        + " → ".join(s.describe() for s in _shortest_path_to(GENERATE, reg, initial_states()[0]))
        + "\n  最短成功路径(pointer): "
        + " → ".join(s.describe() for s in _shortest_path_to(POINTER, reg, initial_states()[0]))
    )
    # 最短成功路径的形状：起点 → 一次调卡 → 生成/指针（步数少于上限，说明上限不是刚好卡死）
    shortest = _shortest_path_to(GENERATE, reg, initial_states()[0])
    assert len(shortest) == 2 and shortest[0].steps == MAX_STEPS


# ---- 验证点 3：拒答受控 -----------------------------------------------------

def test_v3_refusal_controlled():
    """验证点 3：type=unknown 与 info 缺位（无）的可达状态，终止动作**只有**拒答。"""
    reg = FULL_REGISTRY
    reach = reachable_states(registry=reg)
    unknown = [s for s in reach if s.type == UNKNOWN_TYPE]
    absent = [s for s in reach if ABSENT in s.info]
    print(f"[v3] type=unknown 可达={len(unknown)} 含缺位(无)可达={len(absent)}")

    # 这两类状态必须真实存在于可达集里，否则下面的断言是空集恒真。
    assert len(unknown) == 1, f"type=unknown 可达数={len(unknown)}，应为 1"
    assert len(absent) == 10, f"含缺位的可达状态数={len(absent)}，应为 10"

    for state in reach:
        kinds = terminal_kinds_reachable(state, registry=reg)
        if state.type == UNKNOWN_TYPE:
            # 类型不确定：连"调卡"都不许，授权集只有拒答。
            assert {action_kind(a) for a in admissible_actions(state, registry=reg)} == {"reject"}
        if state.type == UNKNOWN_TYPE or ABSENT in state.info:
            assert kinds == frozenset({"reject"}), f"{state.describe()} → {sorted(kinds)}"
        if kinds & {"generate", "pointer"}:
            assert state.type != UNKNOWN_TYPE and ABSENT not in state.info


# ---- 验证点 4：确定性 -------------------------------------------------------

def test_v4_plan_deterministic():
    """验证点 4：同一 (type, info, conf, steps) 两次 plan() 产出**逐字相同**。"""
    reach = reachable_states(registry=FULL_REGISTRY)
    for state in reach:
        first = plan(state, registry=FULL_REGISTRY)
        # 第二次用独立构造的注册表实例，排除"共享可变对象"造成的假确定性。
        second = plan(state, registry=dict(FULL_REGISTRY))
        assert first.as_tuple() == second.as_tuple(), state.describe()
        assert repr(first) == repr(second), state.describe()
        assert first == second
        assert first.signals == state, "plan 不许改动状态"
    print(f"[v4] 确定性：{len(reach)} 个可达状态两次 plan() 的 as_tuple()/repr 逐字相同")


# ---- 验证点 5：闭合（结构校验） ---------------------------------------------

def test_v5_plan_step_spans_closed():
    """验证点 5：每个 Step.input_span 是 None，或 (s,e) 且 0 <= s <= e。"""
    reach = reachable_states(registry=FULL_REGISTRY)
    checked = 0
    for state in reach:
        for span in (None, (0, 4), (4, 4)):
            p = plan(state, registry=FULL_REGISTRY, source_span=span)
            assert closure_problems(p.steps) == [], (state.describe(), p.steps)
            checked += 1

    # 校验器必须抓得住坏样本 —— 否则上面的断言恒真（而且 Step 故意不在构造期校验）。
    assert closure_problems((Step("slot", (5, 2)),))          # start > end
    assert closure_problems((Step("slot", (-1, 2)),))         # 负区间
    assert closure_problems((Step("slot", (0,)),))            # 不是二元组
    assert closure_problems((Step("slot", "0:4"),))           # 读不懂的 span
    assert closure_problems((Step("slot", True),))            # bool 冒充 int
    assert closure_problems((Step("slot", (1, 1)),)) == []    # 单点区间合法
    assert closure_problems((Step("slot", None),)) == []

    # 坏 source_span ⇒ fail-closed 整体拒答，且拒答计划自身闭合。
    p = plan(initial_states()[1], registry=FULL_REGISTRY, source_span=(5, 2))
    assert action_kind(p.action) == "reject" and "source_span" in p.reason
    assert closure_problems(p.steps) == []
    print(f"[v5] 闭合：{len(reach)} 个可达状态 × 3 种 span，共 {checked} 份计划结构全合法")


# ---- 规则表逐行（§2.1 + §2.2 D3/D5） ----------------------------------------

def test_rule_table_rows():
    """规则表每一行的行为都被钉住；行序按 D3（缺位先于调卡）、必需位按 D5 分 type。"""
    all_free = (UNCHECKED,) * 4
    base = Signals("multi_turn", all_free, Conf.LOW, MAX_STEPS)

    # R1 type=unknown ⇒ 拒答
    a = decide(Signals(UNKNOWN_TYPE, all_free, Conf.LOW, MAX_STEPS), registry=FULL_REGISTRY)
    assert action_kind(a) == "reject" and "unknown" in a.reason

    # R3 有未查必需位且 steps>0 ⇒ 调**第一个**未查位对应的卡（multi_turn 全必需 ⇒ time → slot）
    a = decide(base, registry=FULL_REGISTRY)
    assert a == CallCard("slot", "time")
    # D5：plain 只必需 intent ⇒ 授权集里只有 intent 卡，不许顺手调 slot
    plain_start = Signals("plain", all_free, Conf.LOW, MAX_STEPS)
    assert decide(plain_start, registry=FULL_REGISTRY) == CallCard("intent", "intent")
    assert {x.card for x in admissible_actions(plain_start, registry=FULL_REGISTRY)} == {"intent"}
    # 前三位已查 ⇒ 未查必需位只剩 intent
    assert decide(Signals("multi_turn", (HAVE, HAVE, HAVE, UNCHECKED), Conf.MID, 1),
                  registry=FULL_REGISTRY) == CallCard("intent", "intent")
    # steps=0 时 R3 不再命中 ⇒ 落到 R6
    assert action_kind(decide(Signals("multi_turn", all_free, Conf.LOW, 0),
                              registry=FULL_REGISTRY)) == "reject"

    # R2 存在"无"位 ⇒ 拒答（缺必需信息）
    a = decide(Signals("multi_turn", (ABSENT, HAVE, HAVE, HAVE), Conf.MID, MAX_STEPS),
               registry=FULL_REGISTRY)
    assert action_kind(a) == "reject" and "缺必需信息" in a.reason

    # R4 / R5 看**必需**位（D5）：plain 只看 intent，其余三位仍"未查"也照样出输出
    partial = (UNCHECKED, UNCHECKED, UNCHECKED, HAVE)
    for conf in (Conf.MID, Conf.HIGH):
        a = decide(Signals("plain", partial, conf, 2), registry=FULL_REGISTRY)
        assert action_kind(a) == "generate", conf
    a = decide(Signals("plain", partial, Conf.LOW, 2), registry=FULL_REGISTRY)
    assert action_kind(a) == "pointer" and "降级" in a.reason
    # cloze 只必需 object；slot 一次调用把三位一起判"有" ⇒ 直接到输出
    a = decide(Signals("cloze", (HAVE, HAVE, HAVE, UNCHECKED), Conf.HIGH, 2),
               registry=FULL_REGISTRY)
    assert action_kind(a) == "generate"

    # R6 steps=0 且未完成 ⇒ 拒答（不许无限调用）
    a = decide(Signals("multi_turn", (HAVE, HAVE, HAVE, UNCHECKED), Conf.MID, 0),
               registry=FULL_REGISTRY)
    assert action_kind(a) == "reject" and "steps=0" in a.reason

    # D3 行序：有缺位 ⇒ **先拒答**，即便还有未查位且 steps>0；授权集里不许有调卡。
    # 缺位意味着必需信息已确定拿不到，再调别的卡无意义（拒答要早）。
    state = Signals("multi_turn", (ABSENT, UNCHECKED, UNCHECKED, UNCHECKED), Conf.LOW, MAX_STEPS)
    assert [action_kind(x) for x in admissible_actions(state, registry=FULL_REGISTRY)] == ["reject"]
    assert terminal_kinds_reachable(state, registry=FULL_REGISTRY) == frozenset({"reject"})


# ---- D2 / D5：可行动作与必需位（规格要求） -----------------------------------

def test_d2_admissible_is_narrowed_by_rule_table():
    """D2（**规格要求**，不是实现细节）：可行动作 = `admissible(s)`，由规则表按状态收窄。

    起点只授权"调未查必需位的卡"，终止动作一个都不授权；缺位态只授权拒答 ——
    否则「重调 slot → 调 intent」能把"无"洗成"有"，验证点 3 当场不成立。
    """
    reg = FULL_REGISTRY
    free = (UNCHECKED,) * 4
    start = Signals("multi_turn", free, Conf.LOW, MAX_STEPS)
    allowed = admissible_actions(start, registry=reg)
    assert {action_kind(a) for a in allowed} == {"call"}, allowed
    assert {a.card for a in allowed if isinstance(a, CallCard)} == {"slot", "intent"}

    absent = Signals("multi_turn", (ABSENT,) + free[1:], Conf.LOW, MAX_STEPS)
    assert {action_kind(a) for a in admissible_actions(absent, registry=reg)} == {"reject"}
    unknown = Signals(UNKNOWN_TYPE, free, Conf.LOW, MAX_STEPS)
    assert {action_kind(a) for a in admissible_actions(unknown, registry=reg)} == {"reject"}


def test_d5_required_bits_declared_per_type(monkeypatch):
    """D5：必需位按 type 声明（表钉死，不再对 5 类一刀切）；unknown 直接拒答。

    表声明了 `INFO_BITS` 里没有的位 key ⇒ 判不动 ⇒ 拒答（不许静默漏掉一个必需位）。
    """
    assert set(REQUIRED_BITS) == set(INPUT_TYPES)
    assert dict(REQUIRED_BITS) == {
        "plain": ("intent",),
        "cloze": ("object",),
        "candidates": ("intent", "object"),
        "multi_turn": ("time", "place", "object", "intent"),
        UNKNOWN_TYPE: (),
    }
    assert required_bit_indices("plain") == (3,)            # intent
    assert required_bit_indices("cloze") == (2,)            # object
    assert required_bit_indices("candidates") == (2, 3)     # 按位序：object, intent
    assert required_bit_indices("multi_turn") == (0, 1, 2, 3)
    assert required_bit_indices(UNKNOWN_TYPE) == ()

    monkeypatch.setitem(REQUIRED_BITS, "plain", ("nope",))
    s = Signals("plain", (UNCHECKED,) * 4, Conf.LOW, MAX_STEPS)
    assert signals_problems(s), "表与 INFO_BITS 对不齐必须判不动"
    assert action_kind(decide(s, registry=FULL_REGISTRY)) == "reject"


# ---- 抽象转移 P -------------------------------------------------------------

def test_possible_outcomes_abstract_transition():
    """转移层是抽象模型：非确定、steps 减一、判不动就给空集（不接真卡）。"""
    start = initial_states()[0]
    # D4：conf 初值 = LOW —— 初态没有"最近一次调用"可读，取最保守档。
    assert all(s.conf is Conf.LOW for s in initial_states())
    outs = possible_outcomes(start, CallCard("slot", "time"), registry=FULL_REGISTRY)
    assert len(outs) == 4, f"slot 一次调用应有 4 个后继（有×3 档 + 无），实际 {len(outs)}"
    for o in outs:
        assert o.steps == start.steps - 1, "steps 必须减一"
        assert o.type == start.type and o.info[3] is UNCHECKED, "type/未涉及的位不动"
    assert {o.info[:3] for o in outs} == {(HAVE, HAVE, HAVE), (ABSENT, ABSENT, ABSENT)}
    # D4 转移：位变"有"⇒ conf 可能升到 mid/high；变"无"⇒ 读不到置信度 ⇒ LOW。
    # 少写这条 conf 转移 ⇒ conf 恒 LOW ⇒ R4 生成输出永不可达（test_d1 会红）。
    assert {o.conf for o in outs if o.info[0] is HAVE} == {Conf.LOW, Conf.MID, Conf.HIGH}
    assert {o.conf for o in outs if o.info[0] is ABSENT} == {Conf.LOW}

    # 终止动作没有后继
    assert possible_outcomes(start, Terminate("reject", "x"), registry=FULL_REGISTRY) == set()
    # 卡未注册 / 不对应任何信息位 / steps 已耗尽 ⇒ 空集（不许静默继续）
    assert possible_outcomes(start, CallCard("slot", "time"), registry={}) == set()
    assert possible_outcomes(start, CallCard("sentiment", "time"),
                             registry={**FULL_REGISTRY, "sentiment": object()}) == set()
    assert possible_outcomes(replace(start, steps=0), CallCard("slot", "time"),
                             registry=FULL_REGISTRY) == set()
    # 信号判不动 ⇒ 空集
    assert possible_outcomes(replace(start, steps=9), CallCard("slot", "time"),
                             registry=FULL_REGISTRY) == set()


# ---- fail-closed 门禁 -------------------------------------------------------

def test_fail_closed_missing_card_refuses():
    """信息位对应的卡未注册 ⇒ 该状态拒答，且没有任何转移被静默跳过。"""
    only_slot = {"slot": object()}  # intent 没注册
    assert registry_problems(only_slot) == ["信息位 intent(意图) 的卡 'intent' 未注册"]

    # type=unknown 走的是 R1（更早的行），拒答原因点名 unknown 而不是缺卡 —— 两者都拒答。
    for state in [s for s in initial_states() if s.type != UNKNOWN_TYPE]:
        a = decide(state, registry=only_slot)
        assert action_kind(a) == "reject" and "未注册" in a.reason and "intent" in a.reason
    # 缺卡 ⇒ 可达集塌缩到起点本身：一个调用都不许悄悄发生
    assert reachable_states(registry=only_slot) == set(initial_states())
    # 半数卡缺失时，计划也必须是拒答（不是"跳过缺的那张继续跑"）
    p = plan(initial_states()[0], registry=only_slot, source_span=(0, 3))
    assert action_kind(p.action) == "reject" and "fail-closed" in p.reason


def test_production_registry_refuses_all_types_until_cards_added():
    """补卡**之前**的正确行为（fail-closed，§2.2 D7）：生产注册表没有 slot/intent
    ⇒ 全部 type 逐个拒答、一个调用都不发；这不是缺陷。

    `slot` / `intent` 是待补卡的**注册名契约**：新卡必须逐字同名注册进 `all_tasks()`，
    改位→卡映射只动 `INFO_BITS` 这一处。两张卡补进去之后，本测试改成走成功路径
    （届时这里的断言会红）。
    """
    from dtseek.tasks.plugin import all_tasks

    reg = all_tasks()
    assert "slot" not in reg and "intent" not in reg, "卡已注册？那就更新本测试走成功路径"
    for state in [s for s in initial_states() if s.type != UNKNOWN_TYPE]:
        a = decide(state)  # registry=None ⇒ 生产注册表
        assert action_kind(a) == "reject" and "未注册" in a.reason


def test_fail_closed_registry_unreadable_refuses():
    """注册表读不到 ⇒ 拒答原因里要露头"读取失败"，不能悄悄退化成"卡未注册"。"""
    broken = {REGISTRY_ERROR_KEY: "RuntimeError: entry point 爆了"}
    a = decide(initial_states()[0], registry=broken)
    assert action_kind(a) == "reject"
    assert "注册表读取失败" in a.reason and "entry point 爆了" in a.reason
    assert reachable_states(registry=broken) == set(initial_states())


@pytest.mark.parametrize("bad", [
    replace(Signals("plain", (UNCHECKED,) * 4, Conf.LOW, 2), type="gibberish"),
    replace(Signals("plain", (UNCHECKED,) * 4, Conf.LOW, 2), info=(UNCHECKED,) * 3),
    replace(Signals("plain", (UNCHECKED,) * 4, Conf.LOW, 2), info=(UNCHECKED,)*3 + ("maybe",)),
    replace(Signals("plain", (UNCHECKED,) * 4, Conf.LOW, 2), info=[UNCHECKED] * 4),
    replace(Signals("plain", (UNCHECKED,) * 4, Conf.LOW, 2), conf="medium"),
    replace(Signals("plain", (UNCHECKED,) * 4, Conf.LOW, 2), steps=4),
    replace(Signals("plain", (UNCHECKED,) * 4, Conf.LOW, 2), steps=-1),
    replace(Signals("plain", (UNCHECKED,) * 4, Conf.LOW, 2), steps="2"),
    "not a Signals",
])
def test_fail_closed_malformed_signals_refuse(bad):
    """任何"读不到/判不动"的信号都通向拒答，不许落到默认值上、也不许抛异常。"""
    a = decide(bad, registry=FULL_REGISTRY)
    assert action_kind(a) == "reject"
    assert "读不到/判不动" in a.reason
    # 结构校验函数本身也要能说清问题在哪
    if isinstance(bad, Signals):
        assert signals_problems(bad)
