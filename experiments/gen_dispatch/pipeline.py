"""单条输入的端到端执行：**声明 type → 出计划（含生成卡这一步）→ 通道开关 → 统一记录**。

四段（与 `dialogue.respond` 同构，但终点多了"生成卡"这条路）：

  1. 声明 type：`dialogue.resolve_type`（不猜；不认识 ⇒ `unknown` ⇒ dispatch R1 拒答）。
  2. 出计划：`dialogue.turn_plan` 注入对话层的位表/必需位表/位种类表；
     计划里的**生成卡**这一步由接入层补 `bag_cards` 参数（手写声明表 `GEN_BAG_CARDS_BY_TYPE`）。
  3. 调卡：**只用计划授权的卡推进状态**；生成卡要的袋按 `bag_cards` 声明取（逐字、可回溯）。
  4. 通道开关：`rules.switch_channel`（R-CH0–R-CH5，手写）→ `rules.seal` 出统一记录。

读得到 / 判不动 / 调不通一律走拒答（fail-closed），拒答记录按契约**不带 text**。
"""
from __future__ import annotations

import itertools
from dataclasses import asdict, replace
from typing import Any

from dtseek.tasks import render as R
from dtseek.tasks.dispatch import (
    MAX_STEPS,
    UNKNOWN_TYPE,
    CallCard,
    Conf,
    InfoState,
    Plan,
    Signals,
    Step,
    Terminate,
    admissible_actions,
)
from dtseek.tasks.dispatch import plan as dispatch_plan
from dtseek.tasks import dialogue as D

from rules import contract_problems, seal, switch_channel

__all__ = [
    "BAG_CARDS_ALL",
    "GEN_BAG_CARDS_BY_TYPE",
    "COMBO_CAP",
    "annotate_plan",
    "build_bag",
    "execute",
    "fake_propose_anchors",
    "propose_with_anchors",
    "search_renderable",
    "verify_record",
]

#: 生成卡可声明的袋卡全集（**手写声明**；只有在册且已挂载的才真跑）。
BAG_CARDS_ALL: tuple[str, ...] = (
    "person", "pronoun", "relation", "sentiment", "negation", "idiom",
)

#: `type` → 生成卡这一步声明的袋卡（跑前写死；本轮只支持 `plain`）。
#: 「动」槽只能由成语卡供给 —— 真实卡库存里没有动词卡（`card_flow/lexicon.py` 注释）。
GEN_BAG_CARDS_BY_TYPE: dict[str, tuple[str, ...]] = {
    "plain": BAG_CARDS_ALL,
}

#: 一个输入上的指派搜索上限（超了记 `combo_cap`，仍确定性）。
COMBO_CAP = 3000

#: 类型判定顺序（**跑前写死的词典谓词**，无学习量）：
#:   ① `render.CONTENT_LEXICON`（官方实义词表，名/动/形）按 surface 精确命中；
#:   ② `card_flow.lexicon.IDIOM_TYPE`（成语词典）；
#:   ③ `card_flow.lexicon.type_of(card, class_name, surface)` 卡默认类型表；
#: 三者都查不到 ⇒ `None` ⇒ **不进袋**（fail-closed，不是默认值）。
def type_pos(cand: dict) -> str | None:
    """候选 → 官方槽位类型（名/动/形）；查不到返回 `None`（不进袋）。"""
    surf = str(cand.get("surface", "")).strip()
    if not surf:
        return None
    for pos in R.POS_TYPES:
        if surf in R.CONTENT_LEXICON[pos]:
            return pos
    from lexicon import IDIOM_TYPE, type_of  # card_flow 词典（只读 import）

    t = IDIOM_TYPE.get(surf)
    if t in R.POS_TYPES:
        return t
    t = type_of(cand.get("card", ""), cand.get("class_name", ""), surf)
    return t if t in R.POS_TYPES else None


# ---- ① 候选区（真卡：与 `card_flow.proposers.real_propose` 同构，多留 anchors） -----

def propose_with_anchors(
    text: str,
    engine: Any,
    cards: list[str],
    *,
    also: list[tuple[Any, list[str]]] | None = None,
) -> tuple[list[dict], dict[str, tuple[str, ...]], dict[str, list[dict] | None], dict[str, str]]:
    """逐卡 `engine.predict`：返回 (候选, 类别体系, 每卡 anchors, 失败表)。

    `also` 是**别的引擎**上的卡（成语卡自带一副基座 ⇒ 必须用它自己的引擎跑）。
    锚点是原文上的区间，与编码器无关 ⇒ 所有候选合并后再统一编号。

    候选的 cid 编号与 `card_flow.proposers._finish` 同一条规则（按 `(s0,e0,card)` 排序后编号）
    —— `selfcheck` 里与 `real_propose` 逐字对账。调用失败的卡进失败表（fail-closed，
    由 `execute` 变成拒答），不静默跳过。
    """
    from lexicon import type_of

    jobs: list[tuple[Any, str]] = [(engine, c) for c in cards]
    jobs += [(e, c) for e, cs in (also or []) for c in cs]
    seen_cards = {c for _e, c in jobs}
    if len(seen_cards) != len(jobs):
        raise SystemExit(f"卡被安排了两次：{sorted(seen_cards)}（fail-closed）")

    raw: list[dict] = []
    class_index: dict[str, tuple[str, ...]] = {}
    anchors_by_card: dict[str, list[dict] | None] = {}
    errors: dict[str, str] = {}
    for engine, card in jobs:
        try:
            res = engine.predict(text, tasks=[card])
        except Exception as exc:  # noqa: BLE001 —— 调用失败 ⇒ 拒答，不抛给上层
            errors[card] = f"{type(exc).__name__}: {exc}"
            anchors_by_card[card] = None
            continue
        if not isinstance(res, dict):
            errors[card] = f"predict 返回 {type(res).__name__}，不是 dict"
            anchors_by_card[card] = None
            continue
        if "error" in res:
            errors[card] = f"predict error: {res['error']}"
            anchors_by_card[card] = None
            continue
        anchors = res.get("tasks", {}).get(card, [])
        anchors_by_card[card] = list(anchors)
        spec = engine.specs[card]
        class_index[card] = tuple(c.name for c in spec.classes)
        for a in anchors:
            # 引擎的 s0/e0 是 0-based **闭区间**（`compose.py` / `dialogue.py:222` /
            # `runtime.render_highlight` 三处同口径；实测 pronoun(0,0)='我'、person(4,5)='李四'）。
            # 候选与袋一律转成**半开区间**（`render` 的 span 口径），否则 surface 少一个字。
            s0, e0 = a["s0"], a["e0"]
            ok = isinstance(s0, int) and isinstance(e0, int)
            e1 = e0 + 1 if ok else e0
            surf = text[s0:e1] if ok and s0 >= 0 else ""
            raw.append({"card": card, "s0": s0, "e0": e1, "surface": surf,
                        "class_id": a["class_id"], "class_name": a["class_name"],
                        "theme": None})
    raw.sort(key=lambda c: (c["s0"], c["e0"], c["card"]))
    cands = [{**c, "cid": f"c{i}", "type": type_of(c["card"], c["class_name"], c["surface"])}
             for i, c in enumerate(raw)]
    return cands, class_index, anchors_by_card, errors


def fake_propose_anchors(text: str) -> tuple[list[dict], dict[str, tuple[str, ...]],
                                             dict[str, list[dict] | None], dict[str, str]]:
    """假卡自检用：`card_flow.fake_propose` 的词典候选 + 由候选反推的 anchors。

    假卡没有置信度 ⇒ anchors 的 `confidence` 一律 **1.0（构造规定，仅自检用）** ⇒ conf 档 = 高。
    """
    from proposers import fake_propose

    cands, class_index = fake_propose(text)     # 候选已是半开区间（find_all 给 i, i+len(w)）
    anchors: dict[str, list[dict] | None] = {card: [] for card in BAG_CARDS_ALL}
    for c in cands:
        anchors.setdefault(c["card"], []).append(
            {"s0": c["s0"], "e0": c["e0"] - 1, "class_id": c["class_id"],   # 转闭区间（引擎口径）
             "class_name": c["class_name"], "confidence": 1.0})
    return cands, class_index, anchors, {}


# ---- ② 袋 -------------------------------------------------------------------

def build_bag(text: str, candidates: list[dict], class_index: dict[str, tuple[str, ...]]
              ) -> tuple[list[R.BagItem], list[dict], dict]:
    """筛（5 条谓词）→ 定型（`type_pos` 词典谓词）→ 官方袋（逐字 span + 零信息新增可回溯）。

    同一 span 只留**排序后的第一条**（确定性去重：不同卡撞同一区间不许进两个槽）。
    """
    from predicates import filter_candidates

    filtered = filter_candidates(text, candidates, class_index)
    valid = [c for c in filtered if c["valid"]]
    bag: list[R.BagItem] = []
    seen: set[tuple[int, int]] = set()
    skipped = {"type_none": 0, "dup_span": 0}
    for c in valid:
        pos = type_pos(c)
        if pos is None:
            skipped["type_none"] += 1
            continue
        span = (c["s0"], c["e0"])
        if span in seen:
            skipped["dup_span"] += 1
            continue
        seen.add(span)
        bag.append(R.BagItem(ref=len(bag), text=c["surface"], span=span, pos=pos,
                             theta=c.get("theme"), candidate_id=c["cid"], screened=True))
    stats = {"n_candidates": len(filtered), "n_valid": len(valid), "n_bag": len(bag),
             "skipped": skipped,
             "pos_count": {p: sum(1 for b in bag if b.pos == p) for p in R.POS_TYPES}}
    return bag, valid, stats


# ---- ③ 生成可行 G：穷举官方骨架表 + 两道检查 ----------------------------------

def search_renderable(text: str, bag: list[R.BagItem], plan_step_id: str,
                      *, cap: int = COMBO_CAP) -> dict:
    """官方骨架表穷举：找到**第一组**通过两道检查的 (骨架, 指派) 即返回。

    确定性来源：骨架按官方表顺序、指派按 `itertools.product` 的池序（池内按 ref 升序）。
    返回 `{ok, instruction, record, stats}`；`stats` 记为什么没成（类型库存 / 词面守卫 / 指派搜不出）。
    """
    stats: dict[str, Any] = {"combos": 0, "cap": False, "no_type": {}, "fail_first": {},
                             "skeletons_tried": 0}
    if not bag:
        stats["why"] = "bag_empty"
        return {"ok": False, "instruction": None, "record": None, "stats": stats}
    for sk in R._SKELETON_LIST:
        pools = [[it for it in bag if it.pos == slot.pos] for slot in sk.slots]
        empty = [slot.pos for slot, pool in zip(sk.slots, pools) if not pool]
        if empty:
            for p in empty:
                stats["no_type"][p] = stats["no_type"].get(p, 0) + 1
            continue
        stats["skeletons_tried"] += 1
        hit_break = False
        for combo in itertools.product(*pools):
            if stats["combos"] >= cap:
                stats["cap"] = True
                hit_break = True
                break
            stats["combos"] += 1
            refs = tuple(it.ref for it in combo)
            if len(set(refs)) != len(refs):
                continue
            instr = R.Instruction(sk.sid, refs)
            probs = R.check_structure(instr, bag, text)
            if probs:
                key = probs[0].split("：")[0]
                stats["fail_first"][key] = stats["fail_first"].get(key, 0) + 1
                continue
            rec = R.render(instr, bag, text, plan_step_id=plan_step_id)
            if rec.get("kind") != "text":
                key = (rec.get("reason") or "").split("；")[0].split("：")[0]
                stats["fail_first"][key] = stats["fail_first"].get(key, 0) + 1
                continue
            probs2 = R.check_deref(rec, bag, text)
            if probs2:
                stats["fail_first"]["deref:" + probs2[0].split("：")[0]] = \
                    stats["fail_first"].get("deref:" + probs2[0].split("：")[0], 0) + 1
                continue
            return {"ok": True, "instruction": instr, "record": rec, "stats": stats}
        if hit_break:
            break
    stats.setdefault("why", "no_legal_render")
    return {"ok": False, "instruction": None, "record": None, "stats": stats}


# ---- 计划里的生成卡 -----------------------------------------------------------

def annotate_plan(p: Plan, type_: str) -> Plan:
    """给计划里**生成卡**那一步补 `bag_cards`（接入层的手写声明；其余字段原样来自 dispatch）。

    指针 / 拒答的 Step 不动。`Plan.as_tuple()` 含 params ⇒ H1 的逐字比对覆盖这次补参。
    """
    if not isinstance(p.action, Terminate) or p.action.kind != "generate":
        return p
    bag_cards = GEN_BAG_CARDS_BY_TYPE.get(type_, ())
    if not bag_cards:
        return p
    step = p.steps[0]
    new_step = Step(step.module_id, step.input_span,
                    step.params + (("bag_cards", "|".join(bag_cards)),
                                   ("skeleton_table", "render.SKELETONS(16)")))
    return replace(p, steps=(new_step,) + p.steps[1:])


# ---- 主执行 ------------------------------------------------------------------

def _reject_record(reason: str, plan_trace: list[str], n_steps: int, type_: str,
                   cards_run: list[str], terminal: str = "reject") -> dict:
    """拒答出口：`plan_step_id` 指到最后一个已出的 Step；一个 Plan 都没有则记 `none`。"""
    psid = "none" if n_steps == 0 else f"step:{n_steps - 1}"
    return seal(kind="reject", channel="reject", reason=reason, plan_step_id=psid,
                plan=plan_trace, extra={"type": type_, "cards_run": cards_run,
                                        "terminal": terminal})


def execute(
    text: str,
    *,
    type_: str | None = "plain",
    propose: Any = None,
    attached: set[str] | None = None,
    need_g: bool = True,
) -> tuple[dict, dict]:
    """跑一条输入，返回 `(统一记录, debug)`。

    `propose(text) -> (候选, 类别体系, 每卡 anchors, 失败表)`；`None` 时要求调用方给引擎不成立
    （真卡与假卡都由 `run_e2e` 注入）—— 本函数不自己决定"跑哪张卡"，只按计划推进。
    `attached` 给在册已挂载卡；给了就做「计划要的卡在不在」的 fail-closed 检查。
    `need_g=False` 时不算 G（省时间；max_naive 需要 G 时必须 True）。
    """
    declared = D.resolve_type(type_)
    plan_trace: list[str] = []
    turn_steps: list[Step] = []
    cards_run: list[str] = []
    seen_cards: set[str] = set()
    debug: dict[str, Any] = {"declared": declared, "n_turn_steps": 0}
    debug["turn_steps"] = turn_steps   # 同一个 list，边跑边长；报告要展示 Plan=[Step{...}]

    if declared != UNKNOWN_TYPE and declared not in D.SUPPORTED_TYPES:
        return _reject_record(
            f"本轮只支持 type={'/'.join(D.SUPPORTED_TYPES)}，声明了 {declared!r}：不猜",
            plan_trace, len(turn_steps), declared, cards_run), debug

    source = (text or "").strip()
    debug["source"] = source
    if not source:
        return _reject_record("输入为空", plan_trace, len(turn_steps), declared,
                              cards_run), debug
    if propose is None:
        raise ValueError("必须注入 propose（真卡 / 假卡由调用方决定）")

    cands, class_index, anchors_by_card, errors = propose(source)
    debug["propose_errors"] = dict(errors)

    signals = Signals(declared, (InfoState.UNCHECKED,) * len(D.DIALOGUE_BITS),
                      Conf.LOW, MAX_STEPS)
    while True:
        p = annotate_plan(
            D.turn_plan(signals, source_span=(0, len(source))), declared)
        plan_trace.append(p.describe())
        turn_steps.extend(p.steps)
        debug["n_turn_steps"] = len(turn_steps)
        if isinstance(p.action, Terminate):
            break
        authorized = admissible_actions(
            signals, info_bits=D.DIALOGUE_BITS,
            required_bits=D.DIALOGUE_REQUIRED_BITS, bit_kinds=D.DIALOGUE_BIT_KINDS)
        todo = [a.card for a in authorized
                if isinstance(a, CallCard) and a.card not in seen_cards]
        if not todo:
            return _reject_record("计划要求重复调用同一张卡：不许", plan_trace,
                                  len(turn_steps), declared, cards_run), debug
        if attached is not None:
            missing = [c for c in todo if c not in attached]
            if missing:
                return _reject_record(
                    f"必需卡未挂载（引擎里没有）：{missing}：不许静默跳过",
                    plan_trace, len(turn_steps), declared, cards_run), debug
        broken = [c for c in todo if anchors_by_card.get(c) is None]
        if broken:
            return _reject_record(
                f"调卡失败：{broken[0]}：{errors.get(broken[0], '未知错误')}",
                plan_trace, len(turn_steps), declared, cards_run), debug
        for card in todo:  # 按位序推进；只跑计划授权的卡
            anchors = anchors_by_card[card]
            seen_cards.add(card)
            cards_run.append(card)
            signals = D.observe(signals, card, anchors)

    terminal = p.action.kind if isinstance(p.action, Terminate) else "reject"
    action_reason = p.action.reason if isinstance(p.action, Terminate) else "计划停在非终止动作"
    debug.update(terminal=terminal, action_reason=action_reason, plan_trace=plan_trace,
                 n_turn_steps=len(turn_steps), cards_run=list(cards_run))
    psid = "none" if not turn_steps else f"step:{len(turn_steps) - 1}"

    # ---- P：指针有据（只看**计划授权跑过**的卡）----
    picks: list[tuple[str, dict]] = []
    for bit in D.DIALOGUE_BITS:
        anchors = anchors_by_card.get(bit.card)
        if anchors and bit.card in seen_cards and bit.card not in {c for c, _ in picks}:
            picks.append((bit.card, anchors[0]))
    ptr_text: str | None = None
    ptr_evidence: list[dict] = []
    ptr_err = ""
    if picks:
        try:
            ptr_text, raw_ev = D.compose_reply(source, picks)
            # `compose_reply` 的 span 是**闭区间**（引擎口径）；统一记录出口一律转成
            # **半开区间**（与 render 的 evidence 同口径），否则同一字段两种语义。
            ptr_evidence = [{**ev, "span": (ev["span"][0], ev["span"][1] + 1)}
                            for ev in raw_ev]
        except ValueError as exc:
            ptr_err = str(exc)
    else:
        ptr_err = "计划授权调用的卡没有发射任何切片"
    ptr_ok = ptr_text is not None
    debug.update(P=ptr_ok, ptr_err=ptr_err, picks=picks)

    # ---- G：可合法渲染（生成卡声明的袋）----
    bag_cards = GEN_BAG_CARDS_BY_TYPE.get(declared, ())
    unavailable = [c for c in bag_cards if (attached is not None and c not in attached)
                   or c in errors]
    bag: list[R.BagItem] = []
    bag_stats: dict = {}
    g_stats: dict = {}
    g_ok = False
    g_reason = ""
    if terminal == "generate" or need_g:
        if unavailable:
            g_reason = f"生成卡声明的袋卡不可用：{unavailable}（不许少一张卡硬搜）"
        else:
            bag, _valid, bag_stats = build_bag(source, cands, class_index)
            hit = search_renderable(source, bag, psid)
            g_stats = hit["stats"]
            g_ok = bool(hit["ok"])
            if g_ok:
                debug["g_record"] = hit["record"]
                debug["g_instruction"] = hit["instruction"]
            else:
                g_reason = (
                    "袋卡不可用" if unavailable else
                    f"官方骨架表无可渲染指派（{hit['stats'].get('why')}）")
    debug.update(G=g_ok, g_reason=g_reason, g_stats=g_stats, bag_stats=bag_stats,
                 bag=[asdict(b) for b in bag],
                 bag_cards=list(bag_cards), unavailable_bag=unavailable)

    # ---- 开关（R-CH0–R-CH5）----
    channel, row = switch_channel(terminal, g_ok, ptr_ok)
    debug["rule_row"] = row

    if channel == "reject":
        if terminal == "reject":
            reason = action_reason
        else:
            bits = [row]
            if not g_ok and g_reason:
                bits.append("G=" + g_reason)
            if not ptr_ok:
                bits.append("P=" + ptr_err)
            reason = "；".join(bits)
        return _reject_record(reason, plan_trace, len(turn_steps), declared,
                              cards_run, terminal), debug

    if channel == "generate":
        rec = debug["g_record"]
        record = seal(
            kind="text", channel="generate", text=rec["text"], evidence=rec["evidence"],
            reason=row, plan_step_id=rec.get("plan_step_id") or psid, plan=plan_trace,
            extra={"type": declared, "cards_run": cards_run, "terminal": terminal,
                   "instruction": rec.get("instruction"), "ref_map": rec.get("ref_map", []),
                   "bag_cards": list(bag_cards)})
    else:
        note = row if terminal == "pointer" else row
        record = seal(
            kind="text", channel="pointer", text=ptr_text, evidence=ptr_evidence,
            reason=note, plan_step_id=psid, plan=plan_trace,
            extra={"type": declared, "cards_run": cards_run, "terminal": terminal})
    return record, debug


# ---- 复核（H2/H3 的独立重跑） -------------------------------------------------

def verify_record(record: dict, debug: dict) -> list[str]:
    """**独立重跑**一条记录的合法性（不复用执行时的中间结论）。空 = 合法。

    - 记录契约 C1–C4；`n_turn_steps` 用 debug 里的轮次 Step 数核 `plan_step_id` 对齐；
    - 生成型：按记录里的 `instruction` + debug 里的袋**重跑** `check_structure` / `check_deref`；
    - 指针型：按 debug 里的 picks **重跑** `compose_reply`，逐字比对 text，并核每条 span。
    """
    problems = contract_problems(record, n_turn_steps=debug.get("n_turn_steps"))
    source = debug.get("source", "")
    if record.get("kind") != "text":
        return problems
    if record.get("channel") == "generate":
        spec = record.get("instruction") or {}
        instr = R.Instruction(spec.get("skeleton_id", ""),
                              tuple(spec.get("assignment") or ()))
        bag = [R.BagItem(**{**b, "span": tuple(b["span"])})
               for b in debug.get("bag", [])]
        problems += [f"结构侧重跑：{x}" for x in R.check_structure(instr, bag, source)]
        problems += [f"解引用侧重跑：{x}" for x in R.check_deref(record, bag, source)]
        for ev in record.get("evidence", []):
            if ev.get("source") == "span":
                s, e = ev["span"]
                if not (0 <= s < e <= len(source)) or source[s:e] != ev["text"]:
                    problems.append(f"evidence 逐字失败：{ev}")
    else:
        picks = debug.get("picks", [])
        try:
            text2, ev2 = D.compose_reply(source, picks)
        except ValueError as exc:
            problems.append(f"指针通道重跑失败：{exc}")
            return problems
        # `compose_reply` 给闭区间，统一记录里是半开区间 ⇒ 复核前先按同一规则归一
        ev2_norm = [{**ev, "span": (ev["span"][0], ev["span"][1] + 1)} for ev in ev2]
        if text2 != record.get("text"):
            problems.append("指针 text 与重跑结果不逐字一致")
        if ev2_norm != record.get("evidence"):
            problems.append("指针 evidence 与重跑结果不一致")
        for ev in record.get("evidence", []):
            s, e = ev["span"]
            if not (0 <= s < e <= len(source)):
                problems.append(f"evidence span 越界：{ev}")
            elif source[s:e] not in record.get("text", ""):
                problems.append(f"evidence 片段不在回复文本里：{ev}")
    return problems
