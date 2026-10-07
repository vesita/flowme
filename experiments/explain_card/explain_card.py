"""P9 受约束的解释卡 —— 实现（纯 CPU，只读既有产物，不训练）。

三层：
  1. **适配器**：把既有记录（card_flow 自有形状 / gen_dispatch 官方形状）归一成
     `src/dtseek/tasks/render.py` 的记录形状（`instruction/ref_map/evidence/kind/reason`）；
  2. **解释卡渲染**：拒绝记录 ⇒ `reason` 逐字携带、**不建 `text` 键**；
     成功记录 ⇒ 走 `render.render(..., skeletons=EXPL_SKELETONS)`（复用官方渲染器本身，
     它内部就是 `check_structure` → 渲染 → `check_deref`）；
  3. **谓词套件** `x0_problems` / `x2_problems`：全部调用 `render.py` 的既有谓词，不另立一套。

对照臂 `explain_free`：同一份记录用固定中文模板自由拼接，再用**同一套**谓词去咬它。
反例注入 `inject_*`：三类突变，逐类看被抓的报错原文。

用法：`run_explain.py`（驱动）。本模块无副作用、无 I/O 副作用（除 import）。
"""
from __future__ import annotations

import copy
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "experiments" / "card_flow") not in sys.path:
    sys.path.insert(0, str(ROOT / "experiments" / "card_flow"))

from dtseek.tasks import render as R  # 只读 import（硬约束）

CF_DIR = ROOT / "experiments" / "card_flow"
GD_DIR = ROOT / "experiments" / "gen_dispatch"

# ---- 谓词调用计数（证据强度：哪些谓词真被行使、几次返回非空） ------------------

PRED_CALLS: dict[str, int] = {}
PRED_NONEMPTY: dict[str, int] = {}


def _P(name: str, probs: list[str]) -> list[str]:
    PRED_CALLS[name] = PRED_CALLS.get(name, 0) + 1
    if probs:
        PRED_NONEMPTY[name] = PRED_NONEMPTY.get(name, 0) + 1
    return probs


def _rule_a(sk): return _P("rule_a_problems", R.rule_a_problems(sk))


def _slot_schema(sk): return _P("slot_schema_problems", R.slot_schema_problems(sk))


def _word_face(inp, sk): return _P("word_face_problems", R.word_face_problems(inp, sk))


def _item(it, inp): return _P("item_problems", R.item_problems(it, inp))


def _struct(instr, bag, inp, table): return _P("check_structure", R.check_structure(instr, bag, inp, skeletons=table))


def _deref(rec, bag, inp, table): return _P("check_deref", R.check_deref(rec, bag, inp, skeletons=table))


# ---- 封闭解释骨架表（PREREG §2 写死，跑中不改） --------------------------------

EXPL_SKELETONS: dict[str, R.Skeleton] = {
    "E01": R.Skeleton("E01", "「[1]」。", (R.SlotSpec("名"),), direction_safe=True),
    "E02": R.Skeleton("E02", "「[1]」；「[2]」。", (R.SlotSpec("名"), R.SlotSpec("形")), direction_safe=True),
    "E03": R.Skeleton("E03", "「[1]」；「[2]」。", (R.SlotSpec("名"), R.SlotSpec("名")), direction_safe=True),
    "E04": R.Skeleton("E04", "「[1]」；「[2]」。", (R.SlotSpec("名"), R.SlotSpec("动")), direction_safe=True),
    "E05": R.Skeleton(
        "E05", "「[1]」；「[2]」；「[3]」。",
        (R.SlotSpec("名"), R.SlotSpec("动"), R.SlotSpec("形")), direction_safe=True),
}
EXPL_SIG: dict[tuple[str, ...], str] = {
    ("名",): "E01",
    ("名", "形"): "E02",
    ("名", "名"): "E03",
    ("名", "动"): "E04",
    ("名", "动", "形"): "E05",
}


def expl_gate() -> list[str]:
    """解释表门禁：复用 `rule_a_problems` + `slot_schema_problems`（带病不进表）。"""
    probs: list[str] = []
    for sk in EXPL_SKELETONS.values():
        probs += [f"{sk.sid}: {p}" for p in _rule_a(sk)]
        probs += [f"{sk.sid}: {p}" for p in _slot_schema(sk)]
    return probs


# ---- card_flow 骨架表 → 官方 Skeleton 形状（适配器，只换 id 空间不改字面） ---------

_CF_SLOT_RE = re.compile(r"\[([a-z]\d+)\]")
_cf_cache: dict | None = None


def cf_tables():
    """返回 (card_flow 原表 BY_ID, 官方形状适配表 CF_SKELETONS)。"""
    global _cf_cache
    if _cf_cache is None:
        import skeletons as cf_sk  # card_flow 自有表（只读 import）

        orig = {s.id: s for s in cf_sk.SKELETONS}
        table: dict[str, R.Skeleton] = {}
        for sid, s in orig.items():
            names = _CF_SLOT_RE.findall(s.template)
            pat = s.template
            for k, n in enumerate(names, 1):
                pat = pat.replace(f"[{n}]", f"[{k}]")
            slots = tuple(R.SlotSpec(sl.type, None) for sl in s.slots)
            # 真卡不产题元 ⇒ 官方层同口径走「只复述、不换方向」分支（direction_safe）
            table[sid] = R.Skeleton(sid, pat, slots, direction_safe=True)
        _cf_cache = (orig, table)
    return _cf_cache


# ---- 记录 → 官方形状（中间表示） ----------------------------------------------


@dataclass
class Src:
    """一条被解释的记录（归一后）。"""

    rid: str
    set: str
    kind: str                      # text / reject / pointer
    input: str
    text: str = ""
    reason: str = ""
    instruction: dict | None = None
    ref_map: list = field(default_factory=list)
    evidence: list = field(default_factory=list)
    bag: list = field(default_factory=list)      # list[R.BagItem]
    table: dict | None = None                    # 源骨架表（复验用）
    source_skeleton: str = ""
    note: str = ""


def _replay(skeleton: R.Skeleton, assignment: tuple[int, ...],
            items: dict[int, R.BagItem]) -> tuple[str, list[dict]]:
    """按 `render()` 的读表逻辑重放 text + ref_map（纯构造，不判定）。"""
    parts: list[str] = []
    ref_map: list[dict] = []
    out = 0
    lit = 0
    for tok in R.pattern_tokens(skeleton.pattern):
        if tok[0] == "slot":
            it = items[assignment[int(tok[1]) - 1]]
            ref_map.append({
                "unit_id": f"ref:{it.ref}", "cls": "content", "text": it.text,
                "ref": it.ref, "span": (it.span[0], it.span[1]),
                "out": (out, out + len(it.text)),
            })
            parts.append(it.text)
            out += len(it.text)
        else:
            _, kind, word = tok  # type: ignore[misc]
            cls = kind if kind in ("logic", "align") else "formal"
            ref_map.append({
                "unit_id": f"{skeleton.sid}#{lit}", "cls": cls, "text": word,
                "ref": skeleton.sid, "span": None, "out": (out, out + len(word)),
            })
            parts.append(word)
            out += len(word)
            lit += 1
    return "".join(parts), ref_map


def _evidence_for(ref_map: list[dict], items: dict[int, R.BagItem]) -> list[dict]:
    ev: list[dict] = []
    for e in ref_map:
        if e["cls"] == "content":
            it = items[e["ref"]]
            ev.append({"source": "span", "ref": it.ref, "text": it.text,
                       "span": (it.span[0], it.span[1]), "candidate_id": it.candidate_id})
        else:
            ev.append({"source": "table", "unit_id": e["unit_id"],
                       "word": e["text"], "cls": e["cls"]})
    return ev


# ---- A 集：card_flow/results_real.json ---------------------------------------


def load_set_a() -> list[Src]:
    data = json.loads((CF_DIR / "results_real.json").read_text())
    recs = data["batch_real"]["records"]
    _, cf_table = cf_tables()
    out: list[Src] = []
    for i, r in enumerate(recs):
        rid = f"A[{i}]"
        inp = r["input"]
        if r["status"] != "ok":
            out.append(Src(rid, "A", "reject", inp,
                           reason=str(r.get("reason", "")),
                           note=f"stage={r.get('stage')}"))
            continue
        cands = r["candidates"]
        cid2idx = {c["cid"]: j for j, c in enumerate(cands)}
        bag = [R.BagItem(ref=j, text=c["surface"], span=(c["s0"], c["e0"]),
                         pos=c["type"], theta=c["theme"],
                         candidate_id=c["cid"], screened=bool(c["valid"]))
               for j, c in enumerate(cands)]
        items = {b.ref: b for b in bag}
        sid = r["skeleton"]
        sk = cf_table[sid]
        assignment = tuple(cid2idx[m["ref"]] for m in r["mapping"])
        text, ref_map = _replay(sk, assignment, items)
        if text != r["text"]:
            out.append(Src(rid, "A", "reject", inp,
                           reason=f"适配失败：重放文本 {text!r} != 记录 {r['text']!r}",
                           note="adapter_mismatch"))
            continue
        out.append(Src(
            rid, "A", "text", inp, text=text,
            instruction={"skeleton_id": sid, "assignment": list(assignment)},
            ref_map=ref_map, evidence=_evidence_for(ref_map, items),
            bag=bag, table=cf_table, source_skeleton=sid))
    return out


# ---- B / B′ 集：gen_dispatch -------------------------------------------------


def load_set_b() -> list[Src]:
    data = json.loads((GD_DIR / "results_all.json").read_text())
    out: list[Src] = []
    for tag in ("random", "enriched"):
        blk = data[tag]
        for j, g in enumerate(blk.get("generate_samples", [])):
            out.append(_from_generate(f"B[{tag}:g{j}]", g))
        for j, g in enumerate(blk.get("reject_examples", [])):
            out.append(Src(f"B[{tag}:r{j}]", "B", "reject", g["input"],
                           reason=str(g.get("reason", "")), note="dispatch-reject"))
        for j, g in enumerate(blk.get("pointer_examples", [])):
            out.append(Src(f"B[{tag}:p{j}]", "B", "pointer", g["input"],
                           text=str(g.get("text", "")),
                           note="指针通道记录：无 kind/instruction/ref_map ⇒ 非 render 层输出"))
    # B′ 对抗组：递归抽出 kind∈{text,reject} 的记录
    adv = json.loads((GD_DIR / "results_adv.json").read_text())
    found: list[dict] = []

    def walk(o):
        if isinstance(o, dict):
            if o.get("kind") in ("text", "reject") and "reason" in o:
                found.append(o)
                return
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk(adv["adv"])
    for j, rec in enumerate(found):
        has_ins = "instruction" in rec and "evidence" in rec
        note = ("官方 render() 原始拒答（占位串在记录 text 里，本解释卡不携带）"
                if has_ins else "dispatch 层拒答（无 instruction/ref_map）")
        if rec["kind"] == "reject":
            out.append(Src(f"B'[adv:{j}]", "B'", "reject", _adv_input(adv, rec),
                           reason=str(rec.get("reason", "")), note=note))
        elif has_ins:
            out.append(Src(f"B'[adv:{j}]", "B'", "text", _adv_input(adv, rec),
                           note="adv 正例"))
        else:
            out.append(Src(f"B'[adv:{j}]", "B'", "pointer", _adv_input(adv, rec),
                           text=str(rec.get("text", "")), note="text 记录无 instruction/ref_map"))
    return out


def _adv_input(adv_blob: dict, rec: dict) -> str:
    """对抗记录没带 input 时，从 case 里尽力找回（找不到给空串并由后续断言暴露）。"""
    for case in adv_blob["adv"].get("cases", []):
        det = case.get("detail", {})
        for v in det.values():
            if v is rec:
                return str(case.get("input", ""))
    return ""


def _from_generate(rid: str, g: dict) -> Src:
    """gen_dispatch 的 generate 样例：只有 ref_map_head ⇒ 由 instruction+evidence 重放全量 ref_map。"""
    ins = g["instruction"]
    sk = R.SKELETONS[ins["skeleton_id"]]
    assignment = tuple(ins["assignment"])
    # BagItem：type 由记录自带的 slots[k].pos 推出（PREREG §1.5：推断，非独立证据）
    slot_pos = {assignment[k]: sk.slots[k].pos for k in range(len(assignment))}
    bag = []
    for e in g["evidence"]:
        if e.get("source") != "span":
            continue
        bag.append(R.BagItem(ref=int(e["ref"]), text=e["text"],
                             span=(e["span"][0], e["span"][1]),
                             pos=slot_pos.get(int(e["ref"])), theta=None,
                             candidate_id=e.get("candidate_id", ""), screened=True))
    items = {b.ref: b for b in bag}
    text, ref_map = _replay(sk, assignment, items)
    head = g.get("ref_map_head", [])
    head_ok = [({**e, "span": tuple(e["span"]) if e.get("span") else None,
                 "out": tuple(e["out"])}) for e in head] == ref_map[:len(head)]
    note = "ref_map 由 instruction+evidence 重放；head 核对=" + ("一致" if head_ok else "不一致")
    if text != g["text"]:
        note += f"；重放文本不等({text!r}!={g['text']!r})"
    return Src(rid, "B", "text", g["input"], text=g["text"],
               instruction={"skeleton_id": ins["skeleton_id"], "assignment": list(assignment)},
               ref_map=ref_map, evidence=list(g["evidence"]), bag=bag,
               table=R.SKELETONS, source_skeleton=ins["skeleton_id"], note=note)


def load_all() -> list[Src]:
    return load_set_a() + load_set_b()


# ---- 源记录复验（官方谓词） ---------------------------------------------------


def source_problems(src: Src) -> list[str]:
    """用官方谓词复验被解释的记录本身（结构侧 + 解引用侧 + 规则 A + 词面守卫）。"""
    if src.kind != "text":
        return []
    probs: list[str] = []
    sk = (src.table or {}).get(src.instruction["skeleton_id"])  # type: ignore[index]
    if sk is None:
        return [f"源骨架 {src.instruction['skeleton_id']!r} 不在表里"]
    probs += [f"rule_a: {p}" for p in _rule_a(sk)]
    probs += [f"slot_schema: {p}" for p in _slot_schema(sk)]
    probs += [f"word_face: {p}" for p in _word_face(src.input, sk)]
    instr = R.Instruction(sk.sid, tuple(src.instruction["assignment"]))  # type: ignore[index]
    probs += [f"structure: {p}" for p in _struct(instr, src.bag, src.input, src.table)]
    record = {"kind": "text", "text": src.text, "evidence": src.evidence,
              "instruction": src.instruction, "ref_map": src.ref_map}
    probs += [f"deref: {p}" for p in _deref(record, src.bag, src.input, src.table)]
    return probs


# ---- 解释卡渲染 --------------------------------------------------------------


def explain(src: Src) -> dict:
    """返回解释卡（含 `explainable` / `why` 元信息，元信息不参与谓词）。"""
    card: dict = {"rid": src.rid, "set": src.set, "kind": src.kind,
                  "explainable": False, "why": ""}
    if src.kind == "reject":
        card.update(kind="reject", reason=src.reason)   # 不建 text 键（X2）
        card["source"] = {"set": src.set, "rid": src.rid, "kind": "reject",
                          "reason": src.reason, "note": src.note}
        card["explainable"] = True
        return card
    if src.kind == "pointer":
        card["why"] = "未覆盖：记录无 kind/instruction/ref_map（指针通道，非 render 层输出）"
        card["source"] = {"set": src.set, "rid": src.rid, "kind": "pointer", "note": src.note}
        return card

    sk = (src.table or {}).get(src.instruction["skeleton_id"])  # type: ignore[index]
    if sk is None:
        card["why"] = f"未覆盖：源骨架 {src.instruction['skeleton_id']!r} 不在表里"  # type: ignore[index]
        return card
    if len(src.instruction["assignment"]) != len(sk.slots):  # type: ignore[index]
        card["why"] = "未覆盖：源记录指派长度与骨架槽数不等"
        return card
    sig = tuple(sl.pos for sl in sk.slots)
    eid = EXPL_SIG.get(sig)
    if eid is None:
        card["why"] = f"未覆盖：签名 {sig} 不在封闭解释表（含 否/数 槽 ⇒ 官方 POS_TYPES 外）"
        card["source"] = {"set": src.set, "rid": src.rid, "kind": "text",
                          "skeleton_id": sk.sid, "signature": list(sig), "note": src.note}
        return card
    instr = R.Instruction(eid, tuple(src.instruction["assignment"]))  # type: ignore[index]
    rec = R.render(instr, src.bag, src.input, plan_step_id=f"EC:{src.rid}",
                   skeletons=EXPL_SKELETONS)
    if rec.get("kind") != "text":
        card["why"] = f"未覆盖：解释渲染拒答 —— {rec.get('reason', '')}"
        card["source"] = {"set": src.set, "rid": src.rid, "kind": "text",
                          "skeleton_id": sk.sid, "note": src.note}
        return card
    rec.pop("plan_step_id", None)
    rec["source"] = {"set": src.set, "rid": src.rid, "kind": "text",
                     "skeleton_id": sk.sid,
                     "assignment": list(src.instruction["assignment"]),  # type: ignore[index]
                     "text": src.text, "note": src.note}
    card.update(rec)
    card["explainable"] = True
    return card


# ---- 谓词套件（X0 / X2） ------------------------------------------------------


def x0_problems(card: dict, src: Src) -> list[str]:
    """X0（零信息新增，PREREG §3）：解释文本的每个内容单元 100% 回溯。"""
    if card.get("kind") != "text" or not card.get("instruction"):
        return []                      # 不可解释卡由 X5 计，不进 X0 分母
    table = EXPL_SKELETONS
    probs: list[str] = []
    sid = card["instruction"]["skeleton_id"]
    sk = table.get(sid)
    if sk is None:
        return [f"X0: 解释骨架 {sid!r} 不在封闭表"]
    probs += [f"rule_a: {p}" for p in _rule_a(sk)]
    probs += [f"slot_schema: {p}" for p in _slot_schema(sk)]
    probs += [f"word_face: {p}" for p in _word_face(src.input, sk)]
    items = {b.ref: b for b in src.bag}
    src_content = {(e["text"], tuple(e["span"])) for e in src.ref_map if e["cls"] == "content"}
    for e in card.get("ref_map", []):
        if e.get("cls") != "content":
            continue
        it = items.get(e.get("ref"))
        if it is None:
            probs.append(f"回溯: 内容条目 ref={e.get('ref')!r} 不在袋里")
            continue
        probs += [f"item: {p}" for p in _item(it, src.input)]
        if (e.get("text"), tuple(e.get("span") or ())) not in src_content:
            probs.append(
                f"回溯断言: 内容单元 {e.get('text')!r} span={e.get('span')} "
                f"不在源记录 ref_map 的 content 条目里（解释引入了新内容）")
    instr = R.Instruction(sid, tuple(card["instruction"]["assignment"]))
    probs += [f"structure: {p}" for p in _struct(instr, src.bag, src.input, table)]
    probs += [f"deref: {p}" for p in _deref(card, src.bag, src.input, table)]
    return probs


def x2_problems(card: dict) -> list[str]:
    """X2：拒答的解释必须含非空 reason，且 `text` 键缺省。"""
    if card.get("kind") != "reject":
        return []
    probs: list[str] = []
    if "text" in card:
        probs.append("X2 违规：kind=reject 的解释卡带了 text 键（拒答不许有文本）")
    if not str(card.get("reason", "")).strip():
        probs.append("X2 违规：kind=reject 的解释卡缺非空 reason")
    return probs


# ---- 对照臂：E-free 自由生成解释 ---------------------------------------------

_FREE_ROLE = {"名": "主语", "动": "谓语", "形": "谓语", "否": "否定成分", "数": "数量"}


def explain_free(src: Src) -> dict:
    """自由生成解释（无约束）：固定中文模板拼接，不查任何表。"""
    units = [e for e in src.ref_map if e["cls"] == "content"]
    items = {b.ref: b for b in src.bag}
    seg = [f"这句「{src.text}」其实是说{src.input}，"]
    for k, e in enumerate(units, 1):
        pos = items[e["ref"]].pos
        seg.append(f"其中「{e['text']}」是第{k}个成分、作{_FREE_ROLE.get(pos, '成分')}，")
    seg.append("整体读起来是一句完整通顺的话。")
    return {"kind": "text", "text": "".join(seg), "evidence": [], "instruction": None,
            "ref_map": [], "reason": ""}


def free_record(src: Src) -> tuple[dict, dict]:
    """把自由解释文本切词合成一条「记录 + 合成骨架」，好让同一套谓词去咬它。

    分词：贪心最长匹配 源 ref_map 的 content 文本 ∪ 逻辑词 ∪ 形式词 ∪ 标点 ∪ 数字；
    匹配不上的 CJK/ASCII 一律落进骨架字面 ⇒ 由 `rule_a_problems` 判（规则 A 的第二道谓词）。
    """
    text = explain_free(src)["text"]
    items = {b.ref: b for b in src.bag}
    contents: list[tuple[str, int]] = []
    for e in src.ref_map:
        if e["cls"] == "content":
            contents.append((e["text"], e["ref"]))
    contents.sort(key=lambda t: (-len(t[0]), t[0]))
    allow = sorted(set(R.FORMAL_WORDS + R.LOGIC_WORDS), key=lambda w: (-len(w), w))

    matches: list[tuple[int, int, int]] = []   # (start, end, ref)
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
            continue
        i += 1

    if not matches:
        return ({"kind": "text", "text": text, "ref_map": [], "instruction": None,
                 "evidence": [], "reason": ""},
                {"problems": ["free: 文本里没有任何可回溯内容单元"]})

    # 合成 pattern：content 处放 [k]，其余字面原样
    pat: list[str] = []
    prev = 0
    for k, (s, e, _r) in enumerate(matches, 1):
        gap = text[prev:s]
        if "[" in gap or "]" in gap:
            return ({"kind": "text", "text": text, "ref_map": [], "instruction": None,
                     "evidence": [], "reason": ""},
                    {"problems": [f"free: 字面含方括号，无法合成骨架 {gap!r}"]})
        pat.append(gap)
        pat.append(f"[{k}]")
        prev = e
    tail = text[prev:]
    if "[" in tail or "]" in tail:
        return ({"kind": "text", "text": text, "ref_map": [], "instruction": None,
                 "evidence": [], "reason": ""},
                {"problems": [f"free: 字面含方括号，无法合成骨架 {tail!r}"]})
    pat.append(tail)
    pattern = "".join(pat)

    assignment = tuple(r for _s, _e, r in matches)
    slots = tuple(R.SlotSpec(items[r].pos, None) for _s, _e, r in matches)
    synth = R.Skeleton("XFREE", pattern, slots, direction_safe=True)
    full_text, ref_map = _replay(synth, assignment, items)
    assert full_text == text, f"合成回放不等：{full_text!r} != {text!r}"
    record = {"kind": "text", "text": text, "ref_map": ref_map,
              "evidence": _evidence_for(ref_map, items),
              "instruction": {"skeleton_id": "XFREE", "assignment": list(assignment)},
              "reason": ""}
    return record, {"skeleton": synth, "table": {"XFREE": synth}}


def x0_free(record: dict, src: Src, meta: dict) -> list[str]:
    """与 `x0_problems` 同一套谓词，只是骨架表换成自由文本合成的那条。"""
    if "problems" in meta:
        return list(meta["problems"])
    sk = meta["skeleton"]
    probs: list[str] = []
    probs += [f"rule_a: {p}" for p in _rule_a(sk)]
    probs += [f"slot_schema: {p}" for p in _slot_schema(sk)]
    probs += [f"word_face: {p}" for p in _word_face(src.input, sk)]
    items = {b.ref: b for b in src.bag}
    src_content = {(e["text"], tuple(e["span"])) for e in src.ref_map if e["cls"] == "content"}
    for e in record.get("ref_map", []):
        if e.get("cls") != "content":
            continue
        it = items.get(e.get("ref"))
        if it is None:
            probs.append(f"回溯: 内容条目 ref={e.get('ref')!r} 不在袋里")
            continue
        probs += [f"item: {p}" for p in _item(it, src.input)]
        if (e.get("text"), tuple(e.get("span") or ())) not in src_content:
            probs.append(f"回溯断言: 内容单元 {e.get('text')!r} 不在源 ref_map 里")
    instr = R.Instruction("XFREE", tuple(record["instruction"]["assignment"]))
    probs += [f"structure: {p}" for p in _struct(instr, src.bag, src.input, meta["table"])]
    probs += [f"deref: {p}" for p in _deref(record, src.bag, src.input, meta["table"])]
    return probs


# ---- 反例注入（X1 三类） ------------------------------------------------------

_INJECT_WORD = "方案"


def inject_content_word(card: dict, src: Src) -> tuple[dict, list[str]]:
    """① 解释里加一个输入没有的内容词。"""
    assert _INJECT_WORD not in src.input, "注入词必须不在输入里"
    bad = copy.deepcopy(card)
    text = bad["text"]
    ref_map = bad["ref_map"]
    last_end = ref_map[-1]["out"][1] if ref_map else 0
    real_ref = bad["instruction"]["assignment"][0]
    span = (0, min(2, len(src.input)))
    if src.input[span[0]:span[1]] == _INJECT_WORD:      # 撞车就换个区间
        span = (len(src.input) - 1, len(src.input))
    ref_map.append({"unit_id": f"ref:{real_ref}", "cls": "content", "text": _INJECT_WORD,
                    "ref": real_ref, "span": span,
                    "out": (last_end, last_end + len(_INJECT_WORD))})
    bad["text"] = text + _INJECT_WORD
    return bad, x0_problems(bad, src)


def inject_content_word_bag(card: dict, src: Src) -> tuple[dict, list[str]]:
    """①′ **PREREG 外的加测**（只加强、不放松判据）：
    在 ① 的基础上再**伪造一个袋块**给这个输入外的"内容词"背书 ——
    让 `item_problems`（零信息新增的字面谓词）直接面对它。"""
    assert _INJECT_WORD not in src.input, "注入词必须不在输入里"
    bad = copy.deepcopy(card)
    text = bad["text"]
    ref_map = bad["ref_map"]
    last_end = ref_map[-1]["out"][1] if ref_map else 0
    span = (0, min(2, len(src.input)))
    if src.input[span[0]:span[1]] == _INJECT_WORD:
        span = (len(src.input) - 1, len(src.input))
    ref_map.append({"unit_id": "ref:9999", "cls": "content", "text": _INJECT_WORD,
                    "ref": 9999, "span": span,
                    "out": (last_end, last_end + len(_INJECT_WORD))})
    bad["text"] = text + _INJECT_WORD
    bad_src = copy.deepcopy(src)
    bad_src.bag.append(R.BagItem(ref=9999, text=_INJECT_WORD, span=span,
                                 pos="名", theta=None,
                                 candidate_id="c_injected", screened=True))
    return bad, x0_problems(bad, bad_src)


def inject_bad_ref(card: dict, src: Src) -> tuple[dict, list[str]]:
    """②a 引用指向不存在的 ref。"""
    bad = copy.deepcopy(card)
    for e in bad["ref_map"]:
        if e["cls"] == "content":
            e["ref"] = 999
            e["unit_id"] = "ref:999"
            break
    return bad, x0_problems(bad, src)


def inject_bad_span(card: dict, src: Src) -> tuple[dict, list[str]]:
    """②b 引用指向不存在的 span（越界）。"""
    bad = copy.deepcopy(card)
    for e in bad["ref_map"]:
        if e["cls"] == "content":
            e["span"] = (0, 999)
            break
    return bad, x0_problems(bad, src)


def inject_reject_text(card: dict, src: Src) -> tuple[dict, list[str]]:
    """③ 拒答的解释里偷偷带上 text。"""
    bad = copy.deepcopy(card)
    bad["text"] = f"（拒答）{bad.get('reason', '')}"
    return bad, x2_problems(bad)
