"""P11 指针通道的结构化记录 + 可判解释（实现层）。

纯 CPU、只读既有产物：`gen_dispatch/results_all.json` 的 12 条指针记录 + `explain_card` 的谓词套件 +
`src/dtseek/tasks/render.py`（只读 import）。**本模块不实现任何谓词。**

三层：
  1. **适配器**：既有指针记录（`input/text/evidence/terminal/plan_step_id`）+ **CPU 重放**
     （`pipeline.propose` 拿候选与 confidence、`pipeline.execute` 拿 `cards_run`）⇒ 结构化记录
     `{kind:"pointer", chosen[], candidates[], scores_kind, ref_map, evidence, unavailable[]}`；
  2. **L1 机械层**：候选集 + 选择序（卡位序→发射序）+ 分数序（score 降序）+ 入选者 + 分数，
     由**独立重算**（纯函数重排）核对；
  3. **L2 语义层**：四个候选理由句 ⇒ 逐条过 P9 谓词套 + 判别性（C2）+ 指名性（C3），
     任一不过 ⇒ fail-closed「无法给出结构性理由」。

骨架表 `PTR_SKELETONS` 在 PREREG §4 写死；`x0_ptr` 与 `ec.x0_problems` 同构，唯一差别是表参数。
"""
from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in ("src", "experiments/card_flow", "experiments/gen_dispatch", "experiments/explain_card"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

from dtseek.tasks import render as R  # 只读 import（硬约束）
import explain_card as ec  # 只读 import：P9 的谓词套件，不另立一套  # noqa: E402

GD_DIR = ROOT / "experiments" / "gen_dispatch"
INJECT_WORD = "方案"          # PREREG §6 R3-③：必须不在任何受测输入里
DIFF_LIT = ("比", "先", "第", "高", "低", "唯一", "最")   # PREREG §3 C2 封闭差异字面

# ---- 指针解释骨架表（PREREG §4 写死；门禁 = rule_a + slot_schema） ---------------

PTR_SKELETONS: dict[str, R.Skeleton] = {
    "P01": R.Skeleton("P01", "「[1]」。", (R.SlotSpec("形"),), direction_safe=True),
    "P02": R.Skeleton("P02", "「[1]」；「[2]」。",
                      (R.SlotSpec("形"), R.SlotSpec("形")), direction_safe=True),
    "P03": R.Skeleton("P03", "「[1]」。", (R.SlotSpec("名"),), direction_safe=True),
}
PTR_SIG: dict[tuple[str, ...], str] = {("形",): "P01", ("形", "形"): "P02", ("名",): "P03"}


def ptr_gate() -> list[str]:
    """骨架表门禁：复用 P9 的 rule_a / slot_schema 包装（带病不进表）。"""
    probs: list[str] = []
    for sk in PTR_SKELETONS.values():
        probs += [f"{sk.sid}: {p}" for p in ec._rule_a(sk)]
        probs += [f"{sk.sid}: {p}" for p in ec._slot_schema(sk)]
    return probs


# ---- 记录形状 ------------------------------------------------------------------


@dataclass
class PtrSrc:
    """一条指针记录（归一 + 重放后）。"""

    rid: str
    input: str
    text: str                       # 既有记录逐字
    evidence: list                  # 既有记录逐字（不改一个字）
    terminal: str
    plan_step_id: str
    cards_run: list[str] = field(default_factory=list)
    cards_run_provenance: str = "replay"
    candidates: list[dict] = field(default_factory=list)
    chosen: list[dict] = field(default_factory=list)
    replay_match: bool = False
    replay_note: str = ""
    bag: list = field(default_factory=list)     # list[R.BagItem]
    ref_map: list = field(default_factory=list)
    unavailable: list[str] = field(default_factory=list)


_BIT_CARDS: tuple[str, ...] = ()
_ENG: dict = {}


def _bit_cards() -> tuple[str, ...]:
    global _BIT_CARDS
    if not _BIT_CARDS:
        from dtseek.tasks.dialogue import DIALOGUE_BITS
        _BIT_CARDS = tuple(b.card for b in DIALOGUE_BITS)
    return _BIT_CARDS


def engines():
    """CPU 引擎（主引擎 + 成语卡独立引擎）。必须在 CUDA_VISIBLE_DEVICES="" 下调用。"""
    if not _ENG:
        import torch
        if torch.cuda.is_available():
            raise SystemExit("必须纯 CPU：CUDA 可见（未设 CUDA_VISIBLE_DEVICES=\"\"）")
        from dtseek.tasks.engine import MultiTaskEngine
        import idiom_card as ic
        main = MultiTaskEngine(device="cpu")
        main.attach(ROOT / "checkpoints" / "negation_accept_card.pt")
        idi, idi_name, _info = ic.make_idiom_engine()
        _ENG["main"] = (main, sorted(main.decoders))
        _ENG["idiom"] = (idi, [idi_name])
        _ENG["attached"] = set(main.decoders) | {idi_name}
    return _ENG


def make_propose():
    from pipeline import propose_with_anchors
    eng = engines()
    cache: dict[str, tuple] = {}

    def propose(text: str):
        if text not in cache:
            cache[text] = propose_with_anchors(text, eng["main"][0], eng["main"][1],
                                               also=[(eng["idiom"][0], eng["idiom"][1])])
        return cache[text]
    return propose


def _norm_ev(ev: list) -> list[dict]:
    return [{"card": e["card"], "span": [e["span"][0], e["span"][1]],
             "class_name": e["class_name"]} for e in (ev or [])]


def _pos_of(cand: dict) -> str | None:
    from pipeline import type_pos
    return type_pos({"surface": cand["span_text"], "card": cand["card"],
                     "class_name": cand["class_name"]})


# ---- 适配器：既有指针记录 + 重放 ⇒ 结构化记录 ----------------------------------


def load_pointer_records() -> list[PtrSrc]:
    from pipeline import execute
    eng = engines()
    propose = make_propose()
    data = json.loads((GD_DIR / "results_all.json").read_text())
    out: list[PtrSrc] = []
    for tag in ("random", "enriched"):
        for j, g in enumerate(data[tag]["pointer_examples"]):
            rid = f"P[{tag}:{j}]"
            inp = g["input"]
            rec, dbg = execute(inp, propose=propose, attached=eng["attached"], need_g=True)
            match = (rec.get("channel") == "pointer" and rec.get("text") == g["text"]
                     and _norm_ev(rec.get("evidence")) == _norm_ev(g["evidence"])
                     and rec.get("terminal") == g["terminal"]
                     and rec.get("plan_step_id") == g["plan_step_id"])
            note = "" if match else (f"重放不一致: ch={rec.get('channel')} "
                                     f"ev={_norm_ev(rec.get('evidence') or [])}")
            cards_run = list(dbg.get("cards_run") or [])
            cands, _ci, anchors_by_card, errors = propose(inp)
            bit_cards = _bit_cards()
            pool: list[dict] = []
            for card in sorted(anchors_by_card):
                anchors = anchors_by_card.get(card) or []
                for k, a in enumerate(anchors):
                    s0, e0 = int(a["s0"]), int(a["e0"])          # 引擎闭区间
                    span = (s0, e0 + 1)                          # 统一半开
                    c = {
                        "cid": f"{card}:{k}",
                        "card": card,
                        "span": [span[0], span[1]],
                        "span_text": inp[span[0]:span[1]],
                        "score": float(a.get("confidence", 0.0)),
                        "score_provenance": "replay",
                        "emission_index": k,
                        "class_name": a.get("class_name", ""),
                        "eligible": card in cards_run and card in bit_cards,
                        "ineligible_reason": ("" if (card in cards_run and card in bit_cards)
                                              else ("不在本轮 cards_run（计划未授权调用）"
                                                    if card not in cards_run
                                                    else "不在对话位表（无对应位）")),
                    }
                    c["pos"] = _pos_of(c)
                    pool.append(c)
            elig = [c for c in pool if c["eligible"]]
            sel_order = sorted(elig, key=lambda c: (bit_cards.index(c["card"]),
                                                    c["emission_index"]))
            for i, c in enumerate(sel_order):
                c["rank"] = i
            score_order = sorted(pool, key=lambda c: (-c["score"], c["card"], c["emission_index"]))
            for i, c in enumerate(score_order):
                c["score_rank"] = i
            for c in pool:
                c.setdefault("rank", None)
            chosen = []
            for card in bit_cards:
                same = [c for c in elig if c["card"] == card]
                if same and card in cards_run:
                    chosen.append(min(same, key=lambda c: c["emission_index"]))
            unavailable: list[str] = []
            if not match:
                unavailable.append("重放与既有证据不一致 ⇒ 整条标不可得")
            if errors:
                unavailable.append(f"调卡失败表 {errors}")
            for c in chosen:
                if c["pos"] is None:
                    unavailable.append(
                        f"chosen[{c['card']}] 的 pos 不可得（type_pos→None，card_flow.type_of 给"
                        f"「否」∉ POS_TYPES）⇒ 不入受约束槽")
            unavailable.append("chosen.score_original：原跑未持久化 confidence（provenance=replay）")
            unavailable.append("cards_run/plan：原记录未持久化（provenance=replay）")
            ptr = PtrSrc(rid=rid, input=inp, text=str(g.get("text", "")),
                         evidence=list(g.get("evidence") or []),
                         terminal=str(g.get("terminal", "")),
                         plan_step_id=str(g.get("plan_step_id", "")),
                         cards_run=cards_run, candidates=pool, chosen=chosen,
                         replay_match=bool(match), replay_note=note, unavailable=unavailable)
            ptr.bag = [R.BagItem(ref=i, text=c["span_text"],
                                 span=(c["span"][0], c["span"][1]),
                                 pos=c["pos"], theta=None,
                                 candidate_id=f"ptr:{rid}:{c['cid']}", screened=True)
                       for i, c in enumerate(pool)]
            ptr.ref_map = [{"unit_id": f"ptr:{rid}:{c['cid']}", "cls": "content",
                            "text": c["span_text"], "ref": i,
                            "span": (c["span"][0], c["span"][1]),
                            "out": (0, len(c["span_text"]))}
                           for i, c in enumerate(pool)]
            out.append(ptr)
    return out


def struct_record(ptr: PtrSrc) -> dict:
    """结构化记录（PREREG §2 的字段表）。"""
    return {
        "kind": "pointer",
        "rid": ptr.rid,
        "input": ptr.input,
        "chosen": [{"cid": c["cid"], "card": c["card"], "span": c["span"],
                    "span_text": c["span_text"],
                    "score": c["score"], "score_provenance": c["score_provenance"],
                    "rank": c["rank"], "score_rank": c["score_rank"],
                    "eligible": c["eligible"]} for c in ptr.chosen],
        "candidates": [{k: c[k] for k in ("cid", "card", "span", "span_text", "score",
                                          "score_provenance", "rank", "score_rank",
                                          "eligible", "ineligible_reason", "emission_index",
                                          "class_name", "pos")}
                       for c in ptr.candidates],
        "scores_kind": "engine_confidence（原跑未持久化；重放取回。跨卡不可比 ⇒ 只作记录内一致性核对）",
        "ref_map": ptr.ref_map,
        "evidence": ptr.evidence,
        "unavailable": ptr.unavailable,
        "replay_match": ptr.replay_match,
        "cards_run": ptr.cards_run,
        "cards_run_provenance": ptr.cards_run_provenance,
        "text": ptr.text,
    }


def to_ec_src(ptr: PtrSrc) -> "ec.Src":
    """给 `ec.x0_free` 用的源视图（只用 input/bag/ref_map 三个字段）。"""
    return ec.Src(rid=ptr.rid, set="P", kind="pointer", input=ptr.input,
                  text=ptr.text, ref_map=ptr.ref_map, bag=ptr.bag)


# ---- L1：机械层重算（纯函数，不碰引擎） ----------------------------------------


def recompute_selection(candidates: list[dict]) -> list[dict]:
    """独立实现的入选规则：位表序 → 卡内发射序（`anchors[0]`）。"""
    bit_cards = _bit_cards()
    elig = [c for c in candidates if c["eligible"]]
    out = []
    for card in bit_cards:
        same = [c for c in elig if c["card"] == card]
        if same:
            out.append(min(same, key=lambda c: c["emission_index"]))
    return out


def recompute_score_rank(candidates: list[dict]) -> dict[str, int]:
    order = sorted(candidates, key=lambda c: (-c["score"], c["card"], c["emission_index"]))
    return {c["cid"]: i for i, c in enumerate(order)}


def recompute_rank(candidates: list[dict]) -> dict[str, int]:
    bit_cards = _bit_cards()
    elig = sorted([c for c in candidates if c["eligible"]],
                  key=lambda c: (bit_cards.index(c["card"]), c["emission_index"]))
    return {c["cid"]: i for i, c in enumerate(elig)}


def r2_problems(rec: dict) -> list[str]:
    """R2：独立重算的两条序 + 入选者 == 记录里的值。"""
    probs: list[str] = []
    cands = rec["candidates"]
    ch = recompute_selection(cands)
    got = [(c["card"], tuple(c["span"])) for c in rec["chosen"]]
    want = [(c["card"], tuple(c["span"])) for c in ch]
    if got != want:
        probs.append(f"R2 入选者不一致：记录 {got} != 重算 {want}")
    # 【PREREG 外加测，只收紧不放松】入选者必须与**既有记录的 evidence**（真实输出）逐条一致
    got_ev = [(e["card"], tuple(e["span"])) for e in rec.get("evidence", [])]
    want_ev = [(c["card"], tuple(c["span"])) for c in rec["chosen"]]
    if got_ev != want_ev:
        probs.append(f"R2 入选者与既有 evidence 不一致：既有 {got_ev} != 结构记录 {want_ev}")
    rk, srk = recompute_rank(cands), recompute_score_rank(cands)
    for c in cands:
        if c["rank"] != rk.get(c["cid"]):
            probs.append(f"R2 选择序不一致：{c['cid']} 记录 rank={c['rank']} "
                         f"重算={rk.get(c['cid'])}")
        if c["score_rank"] != srk.get(c["cid"]):
            probs.append(f"R2 分数序不一致：{c['cid']} 记录 score_rank={c['score_rank']} "
                         f"重算={srk.get(c['cid'])}（score={c['score']}）")
    return probs


def r1_problems(card: dict, rec: dict, ptr: PtrSrc) -> list[str]:
    """R1：零信息新增 = 谓词套（x0_ptr）+ 逐字断言 + L1 内容 ⊆ chosen。"""
    probs = x0_ptr(card, ptr)
    inp = ptr.input
    for c in rec["candidates"] + rec["chosen"]:
        s, e = c["span"]
        if not (0 <= s < e <= len(inp)) or inp[s:e] != c["span_text"]:
            probs.append(f"R1 逐字断言：{c['cid']} span={c['span']} 给出的 {c['span_text']!r} "
                         f"≠ 输入逐字 {inp[s:e]!r}")
    chosen_txt = {(tuple(c["span"]), c["span_text"]) for c in rec["chosen"]}
    for e in card.get("ref_map", []):
        if e.get("cls") == "content":
            if (tuple(e.get("span") or ()), e.get("text")) not in chosen_txt:
                probs.append(f"R1 断言：解释内容单元 {e.get('text')!r} span={e.get('span')} "
                             f"不在 chosen 里（解释只许复述入选者）")
    return probs


def x0_ptr(card: dict, ptr: PtrSrc) -> list[str]:
    """与 `ec.x0_problems` 同构，唯一差别 = 骨架表换成 PTR_SKELETONS；谓词全走 ec 的包装。"""
    if card.get("kind") != "text" or not card.get("instruction"):
        return []
    table = PTR_SKELETONS
    probs: list[str] = []
    sid = card["instruction"]["skeleton_id"]
    sk = table.get(sid)
    if sk is None:
        return [f"X0: 解释骨架 {sid!r} 不在封闭指针表"]
    probs += [f"rule_a: {p}" for p in ec._rule_a(sk)]
    probs += [f"slot_schema: {p}" for p in ec._slot_schema(sk)]
    probs += [f"word_face: {p}" for p in ec._word_face(ptr.input, sk)]
    items = {b.ref: b for b in ptr.bag}
    src_content = {(e["text"], tuple(e["span"])) for e in ptr.ref_map if e["cls"] == "content"}
    for e in card.get("ref_map", []):
        if e.get("cls") != "content":
            continue
        it = items.get(e.get("ref"))
        if it is None:
            probs.append(f"回溯: 内容条目 ref={e.get('ref')!r} 不在袋里")
            continue
        probs += [f"item: {p}" for p in ec._item(it, ptr.input)]
        if (e.get("text"), tuple(e.get("span") or ())) not in src_content:
            probs.append(f"回溯断言: 内容单元 {e.get('text')!r} span={e.get('span')} "
                         f"不在源记录 ref_map 的 content 条目里（解释引入了新内容）")
    instr = R.Instruction(sid, tuple(card["instruction"]["assignment"]))
    probs += [f"structure: {p}" for p in ec._struct(instr, ptr.bag, ptr.input, table)]
    probs += [f"deref: {p}" for p in ec._deref(card, ptr.bag, ptr.input, table)]
    return probs


def l1_card(ptr: PtrSrc) -> dict:
    """L1 解释卡：用封闭指针表渲染「入选者复述句」+ 挂机械层字段。"""
    card: dict = {"rid": ptr.rid, "kind": "pointer", "explainable": False, "why": ""}
    renderable = [c for c in ptr.chosen if c["pos"] in R.POS_TYPES]
    skipped = [c for c in ptr.chosen if c["pos"] not in R.POS_TYPES]
    card["unrenderable_chosen"] = [
        {"card": c["card"], "span": c["span"], "span_text": c["span_text"],
         "why": "pos 不可得/不在 POS_TYPES ⇒ 不入受约束槽"} for c in skipped]
    sig = tuple(c["pos"] for c in renderable)
    if not sig:
        card["why"] = "未覆盖：没有任何入选切片能定型（pos 全为 None）"
        return card
    eid = PTR_SIG.get(sig)
    if eid is None:
        card["why"] = f"未覆盖：签名 {sig} 不在封闭指针解释表"
        return card
    idx = {c["cid"]: i for i, c in enumerate(ptr.candidates)}
    assignment = tuple(idx[c["cid"]] for c in renderable)
    instr = R.Instruction(eid, assignment)
    rec = R.render(instr, ptr.bag, ptr.input, plan_step_id=f"PC:{ptr.rid}",
                   skeletons=PTR_SKELETONS)
    if rec.get("kind") != "text":
        card["why"] = f"未覆盖：解释渲染拒答 —— {rec.get('reason', '')}"
        return card
    rec.pop("plan_step_id", None)
    rec["source"] = {"rid": ptr.rid, "kind": "pointer", "skeleton_id": eid,
                     "assignment": list(assignment)}
    card.update(rec)
    card["explainable"] = True
    return card


# ---- L2：语义层四个尝试 ---------------------------------------------------------


def synth_record(text: str, ptr: PtrSrc) -> tuple[dict, dict]:
    """文本 → 合成骨架（与 `ec.free_record` 同法的分词器，**不是谓词**）。"""
    items = {b.ref: b for b in ptr.bag}
    # 同一文本可能来自多张卡 ⇒ 去重时优先「在位表且能定型」的那个（确定性 tie-break）
    by_text: dict[str, int] = {}
    for e in ptr.ref_map:
        c = ptr.candidates[e["ref"]]
        cur = by_text.get(e["text"])
        if cur is None:
            by_text[e["text"]] = e["ref"]
        else:
            pc = ptr.candidates[cur]
            k_new = (0 if c["eligible"] else 1, 0 if c["pos"] else 1, c["cid"])
            k_cur = (0 if pc["eligible"] else 1, 0 if pc["pos"] else 1, pc["cid"])
            if k_new < k_cur:
                by_text[e["text"]] = e["ref"]
    contents = sorted(((w, r) for w, r in by_text.items()),
                      key=lambda t: (-len(t[0]), t[0]))
    matches: list[tuple[int, int, int]] = []
    i = 0
    while i < len(text):
        hit = None
        for w, ref in contents:
            if w and text.startswith(w, i):
                hit = (i, i + len(w), ref)
                break
        if hit:
            matches.append(hit)
            i = hit[1]
        else:
            i += 1
    empty = ({"kind": "text", "text": text, "ref_map": [], "instruction": None,
              "evidence": [], "reason": ""},
             {"problems": ["synth: 文本里没有任何可回溯内容单元"]})
    if not matches:
        return empty
    pat: list[str] = []
    prev = 0
    for k, (s, e, _r) in enumerate(matches, 1):
        gap = text[prev:s]
        if "[" in gap or "]" in gap:
            return empty[:1] + ({"problems": [f"synth: 字面含方括号 {gap!r}"]},)
        pat += [gap, f"[{k}]"]
        prev = e
    tail = text[prev:]
    if "[" in tail or "]" in tail:
        return (empty[0], {"problems": [f"synth: 字面含方括号 {tail!r}"]})
    pat.append(tail)
    assignment = tuple(r for _s, _e, r in matches)
    slots = tuple(R.SlotSpec(items[r].pos, None) for _s, _e, r in matches)
    # id 必须是 XFREE：`ec.x0_free`（P9 自由臂检查器）内部硬编码 Instruction("XFREE")
    synth = R.Skeleton("XFREE", "".join(pat), slots, direction_safe=True)
    full, ref_map = ec._replay(synth, assignment, items)
    if full != text:
        return (empty[0], {"problems": [f"synth: 回放不等 {full!r}!={text!r}"]})
    record = {"kind": "text", "text": text, "ref_map": ref_map,
              "evidence": ec._evidence_for(ref_map, items),
              "instruction": {"skeleton_id": "XFREE", "assignment": list(assignment)},
              "reason": ""}
    return record, {"skeleton": synth, "table": {"XFREE": synth}}


def l2_attempts(ptr: PtrSrc) -> list[dict]:
    """四个候选理由句：C1 过谓词套（ec.x0_free）、C2 判别性、C3 指名性。"""
    src = to_ec_src(ptr)
    alts = [c for c in ptr.candidates if c["eligible"] and c not in ptr.chosen]
    # 同卡对照候选（至少一个同卡、不同 span 的入选池内候选）
    same_card_alt = None
    for c in ptr.chosen:
        for a in ptr.candidates:
            if a["card"] == c["card"] and a["cid"] != c["cid"] and a["eligible"]:
                same_card_alt = (c, a)
                break
        if same_card_alt:
            break
    spans = [c["span_text"] for c in ptr.chosen]
    labels = [c["class_name"] for c in ptr.chosen]
    x = spans[0] if spans else ""
    # A2 的对照必须与被判定的差异事实同一对（同一张卡的入选者 vs 同卡另一候选）
    xp, yp = ((same_card_alt[0]["span_text"], same_card_alt[1]["span_text"])
              if same_card_alt else ("", ""))
    texts = {
        "A1 类名理由": f"「{x}」{labels[0]}。" if x else "",
        "A2 分数/序差异理由": (f"「{xp}」比「{yp}」高。" if yp else ""),
        "A3 因果理由": f"因为「{x}」。" if x else "",
        "A4 纯复述（对照）": f"「{x}」。" if x else "",
    }
    out: list[dict] = []
    for name, text in texts.items():
        att: dict = {"name": name, "text": text}
        if not text:
            att.update(ok=False, na=True,
                       fail="N/A：无同卡对照候选（结构上不可得差异事实）"
                       if name.startswith("A2") else "N/A：无入选 span",
                       problems=[])
            out.append(att)
            continue
        record, meta = synth_record(text, ptr)
        probs = ec.x0_free(record, src, meta) if "problems" not in meta else meta["problems"]
        # C1
        c1 = not probs
        # C2 判别性：必须含封闭差异字面，且差异在记录内重算为真
        has_diff = [w for w in DIFF_LIT if w in text]
        diff_true = False
        if has_diff and same_card_alt:
            ch, alt = same_card_alt
            if "高" in has_diff:
                diff_true = ch["score"] > alt["score"] and ch["score_rank"] < alt["score_rank"]
            elif "低" in has_diff:
                diff_true = ch["score"] < alt["score"]
            elif has_diff[0] in ("第", "先", "最", "唯一"):
                diff_true = ch["rank"] == 0
        c2 = bool(has_diff) and bool(diff_true)
        # C3 指名性：内容单元 ∈ ref_map 且逐字 ∈ 输入
        c3 = all((tuple(e.get("span") or (0, 0)) and
                  ptr.input[e["span"][0]:e["span"][1]] == e["text"])
                 for e in record.get("ref_map", []) if e.get("cls") == "content")
        att.update(ok=bool(c1 and c2 and c3), c1=bool(c1), c2=bool(c2), c3=bool(c3),
                   diff_lit=has_diff, diff_true=bool(diff_true),
                   problems=probs[:6], n_problems=len(probs),
                   fail=("" if (c1 and c2 and c3) else
                         "; ".join([f"C1 谓词套 {len(probs)} 个问题" if not c1 else "",
                                    "C2 判别性不成立（无差异字面 / 差异重算不为真）" if not c2 else "",
                                    "C3 指名性不成立" if not c3 else ""]).strip("; ")))
        out.append(att)
    return out


def l2_verdict(attempts: list[dict]) -> dict:
    ok = [a for a in attempts if a.get("ok")]
    if ok:
        return {"given": True, "reason": ok[0]["text"], "attempts": attempts}
    first = next((a for a in attempts if not a.get("na")), None)
    return {"given": False,
            "reason": "无法给出结构性理由",
            "detail": (first or {}).get("fail", ""),
            "first_problem": ((first or {}).get("problems") or [""])[0],
            "attempts": attempts}


# ---- 对照臂 G-free + 平凡基线 ---------------------------------------------------


def g_free_strict(ptr: PtrSrc) -> list[str]:
    src = ec.Src(rid=ptr.rid, set="P", kind="pointer", input=ptr.input,
                 text=ptr.text, ref_map=[], bag=[])
    _rec, meta = ec.free_record(src)
    return list(meta.get("problems") or ec.x0_free(_rec, src, meta))


def g_free_generous(ptr: PtrSrc) -> list[str]:
    """无结构臂（宽松版）：只从输出文本里词法抽取『…』当内容单元，不给候选/序/分数。"""
    segs = re.findall(r"『(.*?)』", ptr.text)
    ref_map, bag = [], []
    for k, s in enumerate(segs):
        st = ptr.input.find(s)
        if st < 0:
            st = 0
        ref_map.append({"unit_id": f"free:{k}", "cls": "content", "text": s,
                        "ref": k, "span": (st, st + len(s)), "out": (0, 0)})
        bag.append(R.BagItem(ref=k, text=s, span=(st, st + len(s)), pos=None, theta=None,
                             candidate_id=f"free:{k}", screened=True))
    src = ec.Src(rid=ptr.rid, set="P", kind="pointer", input=ptr.input,
                 text=ptr.text, ref_map=ref_map, bag=bag)
    rec, meta = ec.free_record(src)
    if "problems" in meta:
        return list(meta["problems"])
    return ec.x0_free(rec, src, meta)


def b_echo(ptr: PtrSrc) -> list[str]:
    """平凡基线：解释 = 输出文本原样复读；同一套合成 + 谓词判它。"""
    src = to_ec_src(ptr)
    rec, meta = synth_record(ptr.text, ptr)
    if "problems" in meta:
        return list(meta["problems"])
    return ec.x0_free(rec, src, meta)


# ---- R3 反例注入（突变后跑**同一套**检查） ---------------------------------------


def inject_span(card: dict, rec: dict, ptr: PtrSrc) -> tuple[dict, dict, PtrSrc, list[str]]:
    """① 改 chosen span（span_text 不动 ⇒ 必然不逐字）。"""
    r2 = copy.deepcopy(rec)
    r2["chosen"][0]["span"] = [r2["chosen"][0]["span"][0], r2["chosen"][0]["span"][1] + 1]
    c2 = copy.deepcopy(card)
    return c2, r2, ptr, r1_problems(c2, r2, ptr)


def inject_score(card: dict, rec: dict, ptr: PtrSrc) -> tuple[dict, dict, PtrSrc, list[str]]:
    """② 改分数使分数序矛盾（记录里存的 score_rank 不动 ⇒ 重算必不一致）。"""
    r2 = copy.deepcopy(rec)
    order = sorted(r2["candidates"], key=lambda c: c["score_rank"])
    if len(order) < 2:
        return copy.deepcopy(card), r2, ptr, ["注入不可用：候选池 < 2"]
    worst = order[-1]
    worst["score"] = max(c["score"] for c in r2["candidates"]) + 1.0
    worst["score_provenance"] = "injected"
    return copy.deepcopy(card), r2, ptr, r2_problems(r2)


def inject_word(card: dict, rec: dict, ptr: PtrSrc) -> tuple[dict, dict, PtrSrc, list[str]]:
    """③ 加一个输入没有的内容词。"""
    assert INJECT_WORD not in ptr.input, "注入词必须不在输入里"
    c2 = copy.deepcopy(card)
    rm = c2["ref_map"]
    last_end = rm[-1]["out"][1] if rm else 0
    real_ref = c2["instruction"]["assignment"][0]
    rm.append({"unit_id": f"ref:{real_ref}", "cls": "content", "text": INJECT_WORD,
               "ref": real_ref, "span": (0, min(2, len(ptr.input))),
               "out": (last_end, last_end + len(INJECT_WORD))})
    c2["text"] = c2.get("text", "") + INJECT_WORD
    return c2, rec, ptr, r1_problems(c2, rec, ptr)
