"""V4 / V5：**统一的机械层解释接口** + 反例注入。

- 解释接口只做**路由**：按记录的 `kind/channel` 转到既有谓词套件 ——
  `explain_card`（E 表受约束解释 + `x0/x2` 谓词）、`pointer_explain`（P 表 L1 + `x0_ptr`），
  **两者都只读 import，本文件不实现任何判定谓词**；谓词本体仍是 `render.py` 的
  `rule_a / slot_schema / word_face / item / check_structure / check_deref`。
- 指针记录在归一器里补上**结构字段**（`chosen` 入选者 / `candidates` 候选集 / `scores_kind` 分数口径），
  于是它从「无结构可解释」变成能过 `x0_ptr` 的可判解释（基线 0/12）。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in ("src", "experiments/card_flow", "experiments/gen_dispatch",
           "experiments/explain_card", "experiments/pointer_explain"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

from dtseek.tasks import card_contract as CC  # noqa: E402
from dtseek.tasks import render as R  # noqa: E402

import explain_card as ec  # noqa: E402  只读 import
import ptr_explain as PX  # noqa: E402  只读 import

import migration as M  # noqa: E402
import replay as RP  # noqa: E402

GD = ROOT / "experiments" / "gen_dispatch"
PE = ROOT / "experiments" / "pointer_explain"


# ---- 统一接口 -----------------------------------------------------------------

def explain(rec: dict, *, input_text: str, rid: str, table: dict | None = None,
            bag: list | None = None, ptr_src=None) -> dict:
    """**任何卡的记录 ⇒ 机械层可判解释**（返回 `{explainable, problems, card, why}`）。

    路由（构造规定，逐条可核对）：

    - `kind="reject"` ⇒ `ec.explain` 的拒答解释卡 + `ec.x2_problems`（不建 `text` 键）；
    - `kind="text"` 且带 `instruction` ⇒ `ec.explain`（E 表）+ `ec.x0_problems`；
    - `kind="pointer"` ⇒ `PX.l1_card`（P 表）+ `PX.x0_ptr`。
    """
    kind = rec.get("kind")
    if kind == "reject":
        src = ec.Src(rid, "X", "reject", input_text, reason=str(rec.get("reason") or ""))
        card = ec.explain(src)
        return {"rid": rid, "kind": kind, "channel": rec.get("channel"),
                "explainable": bool(card.get("explainable")),
                "problems": ec.x2_problems(card), "card": card,
                "why": card.get("why", ""), "suite": "explain_card.x2"}
    if kind == "text" and rec.get("instruction"):
        src = ec.Src(rid, "X", "text", input_text, text=str(rec.get("text") or ""),
                     instruction=rec.get("instruction"),
                     ref_map=list(rec.get("ref_map") or []),
                     evidence=list(rec.get("evidence") or []),
                     bag=list(bag or []), table=table or R.SKELETONS,
                     source_skeleton=(rec.get("instruction") or {}).get("skeleton_id", ""))
        card = ec.explain(src)
        probs = ec.x0_problems(card, src) if card.get("explainable") else []
        return {"rid": rid, "kind": kind, "channel": rec.get("channel"),
                "explainable": bool(card.get("explainable")),
                "problems": probs, "card": card, "why": card.get("why", ""),
                "suite": "explain_card.x0"}
    if kind in ("pointer", "text"):
        psrc = ptr_src if ptr_src is not None else _ptr_from_record(rec, input_text, rid)
        if psrc is None:
            return {"rid": rid, "kind": kind, "channel": rec.get("channel"),
                    "explainable": False, "problems": [],
                    "card": {}, "why": "指针记录缺 candidates ⇒ 无结构可解释",
                    "suite": "pointer_explain.x0_ptr"}
        card = PX.l1_card(psrc)
        probs = PX.x0_ptr(card, psrc) if card.get("explainable") else []
        return {"rid": rid, "kind": kind, "channel": rec.get("channel"),
                "explainable": bool(card.get("explainable")),
                "problems": probs, "card": card, "why": card.get("why", ""),
                "suite": "pointer_explain.x0_ptr"}
    return {"rid": rid, "kind": kind, "explainable": False, "problems": [],
            "card": {}, "why": f"kind={kind!r} 不可解释", "suite": "-"}


def _ptr_from_record(rec: dict, input_text: str, rid: str):
    """指针记录 → `PtrSrc`（结构字段来自契约记录本身；bag 由 candidates 重建）。"""
    cands = list(rec.get("candidates") or [])
    if not cands:
        return None
    pos_by_cid = {c.get("cid"): c.get("pos") for c in cands}
    chosen = [dict(c, pos=pos_by_cid.get(c.get("cid"))) for c in (rec.get("chosen") or [])]
    bag = [R.BagItem(ref=i, text=str(c.get("span_text") or ""),
                     span=(int(c["span"][0]), int(c["span"][1])),
                     pos=c.get("pos"), theta=None,
                     candidate_id=f"ptr:{rid}:{c.get('cid')}", screened=True)
           for i, c in enumerate(cands)]
    return PX.PtrSrc(rid=rid, input=input_text, text=str(rec.get("text") or ""),
                     evidence=list(rec.get("evidence") or []),
                     terminal=str(rec.get("terminal") or ""),
                     plan_step_id=str(rec.get("plan_step_id") or "none"),
                     cards_run=list(rec.get("cards_run") or []),
                     candidates=cands, chosen=chosen, replay_match=True,
                     bag=bag, ref_map=list(rec.get("ref_map") or []),
                     unavailable=[])


# ---- 记录来源 -----------------------------------------------------------------

def class_render() -> list[dict]:
    """类 1：render 层记录（A 集 35 条，迁移后在唯一家重放出来的完整记录）。"""
    out = []
    for r in RP.load_a():
        new = M.old_to_new(r["skeleton"])
        if new is None:
            continue
        full = _a_full(r, new)
        if full is None:
            continue
        rec, bag = full
        out.append({"rid": r["rid"], "rec": CC.to_dict(CC.to_contract(rec)),
                    "input": r["input"], "bag": bag, "table": R.SKELETONS})
    return out


def _a_full(r: dict, new: str):
    cands = r["candidates"]
    cid2idx = {c["cid"]: j for j, c in enumerate(cands)}
    bag = [R.BagItem(ref=j, text=c["surface"], span=(c["s0"], c["e0"]),
                     pos=c["type"], theta=c["theme"],
                     candidate_id=c["cid"], screened=bool(c["valid"]))
           for j, c in enumerate(cands)]
    assignment = tuple(cid2idx[m["ref"]] for m in r["mapping"])
    rec = R.render(R.Instruction(new, assignment), bag, r["input"], plan_step_id="A:1")
    if rec.get("kind") != "text":
        return None
    return rec, bag


def class_dispatch() -> list[dict]:
    """类 2：`gen_dispatch` 三通道记录（generate / pointer / reject）⇒ 契约归一。"""
    data = json.loads((GD / "results_all.json").read_text())
    struct = {s["rid"]: s for s in
              json.loads((PE / "results.json").read_text())["struct_records"]}
    out: list[dict] = []
    for tag in ("random", "enriched"):
        blk = data[tag]
        for j, g in enumerate(blk.get("generate_samples", [])):
            raw = {"kind": "text", "text": g["text"], "evidence": g["evidence"],
                   "plan_step_id": g.get("plan_step_id", ""), "instruction": g["instruction"],
                   "ref_map": g.get("ref_map_head") or [], "reason": "",
                   "plan": g.get("plan") or [], "cards_run": g.get("cards_run") or [],
                   "terminal": g.get("terminal", "")}
            out.append({"rid": f"D[{tag}:g{j}]", "rec": CC.to_dict(CC.to_contract(raw)),
                        "input": g["input"], "bag": _bag_of_g(g), "table": R.SKELETONS})
        for j, g in enumerate(blk.get("reject_examples", [])):
            as_record = {"kind": "reject", "evidence": [], "plan_step_id": "none",
                         "reason": str(g.get("reason") or ""), "plan": [],
                         "channel": "reject", "terminal": g.get("terminal"),
                         "type": None}
            out.append({"rid": f"D[{tag}:r{j}]",
                        "rec": CC.to_dict(CC.to_contract(as_record)),
                        "input": g.get("input", ""), "bag": [], "table": None})
        for j, g in enumerate(blk.get("pointer_examples", [])):
            s = struct.get(f"P[{tag}:{j}]")
            structure = None
            if s is not None:
                pos_by_cid = {c.get("cid"): c.get("pos") for c in s["candidates"]}
                structure = {
                    "chosen": [dict(c, pos=pos_by_cid.get(c.get("cid")))
                               for c in s["chosen"]],
                    "candidates": s["candidates"], "scores_kind": s["scores_kind"]}
            raw = {"kind": "pointer", "text": g.get("text", ""),
                   "evidence": g.get("evidence") or [],
                   "plan_step_id": g.get("plan_step_id", ""),
                   "reason": "", "terminal": g.get("terminal", ""),
                   "ref_map": (s or {}).get("ref_map") or [],
                   "cards_run": (s or {}).get("cards_run") or []}
            out.append({"rid": f"D[{tag}:p{j}]",
                        "rec": CC.to_dict(CC.to_contract(raw, structure=structure)),
                        "input": g["input"], "bag": [], "table": None})
    return out


def _bag_of_g(g: dict) -> list:
    sk = R.SKELETONS[g["instruction"]["skeleton_id"]]
    assignment = tuple(g["instruction"]["assignment"])
    slot_pos = {assignment[k]: sk.slots[k].pos for k in range(len(assignment))}
    return [R.BagItem(ref=int(e["ref"]), text=e["text"],
                      span=(e["span"][0], e["span"][1]),
                      pos=slot_pos.get(int(e["ref"])), theta=None,
                      candidate_id=str(e.get("candidate_id", "")), screened=True)
            for e in g["evidence"] if e.get("source") == "span"]


def class_pointer() -> list[dict]:
    """类 3：`pointer_explain` 的 12 条指针记录（结构字段已在既有产物里）。"""
    data = json.loads((PE / "results.json").read_text())
    out = []
    for s in data["struct_records"]:
        raw = {"kind": "pointer", "text": s["text"], "evidence": s["evidence"],
               "plan_step_id": "none", "reason": "", "terminal": "",
               "ref_map": s["ref_map"], "cards_run": s["cards_run"]}
        pos_by_cid = {c.get("cid"): c.get("pos") for c in s["candidates"]}
        structure = {"chosen": [dict(c, pos=pos_by_cid.get(c.get("cid")))
                                for c in s["chosen"]],
                     "candidates": s["candidates"], "scores_kind": s["scores_kind"]}
        rec = CC.to_dict(CC.to_contract(raw, structure=structure))
        out.append({"rid": s["rid"], "rec": rec, "input": s["input"],
                    "bag": [], "table": None})
    return out


# ---- V4 -----------------------------------------------------------------------

def _run(items: list[dict]) -> list[dict]:
    out = []
    for it in items:
        bag = it.get("bag")
        if not bag and it["rec"].get("candidates"):
            bag = None
        out.append(explain(it["rec"], input_text=it["input"], rid=it["rid"],
                           table=it.get("table"), bag=bag))
    return out


def v4() -> dict:
    """三类记录都产出可判解释；指针覆盖率 > 0（基线 0/12）。"""
    c1, c2, c3 = class_render(), class_dispatch(), class_pointer()

    # —— 基线：P9 的 X5 口径（把指针记录当 kind=pointer 直接喂 explain_card）——
    base = []
    for it in c3:
        src = ec.Src(it["rid"], "X", "pointer", it["input"],
                     text=str(it["rec"].get("text") or ""))
        card = ec.explain(src)
        base.append(bool(card.get("explainable")))

    r1, r2, r3 = _run(c1), _run(c2), _run(c3)
    pointer_covered = sum(1 for x in r3 if x["explainable"])
    classes = {"render": r1, "gen_dispatch": r2, "pointer": r3}
    summary = {}
    for name, rows in classes.items():
        summary[name] = {
            "n": len(rows),
            "explainable": sum(1 for x in rows if x["explainable"]),
            "problems_total": sum(len(x["problems"]) for x in rows),
            "with_problems": sum(1 for x in rows if x["problems"]),
            "why_not": sorted({x["why"] for x in rows if not x["explainable"]}),
        }
    # 指针记录的契约违例（补结构字段前 vs 后）
    ptr_raw_violate = 0
    ptr_full_violate = 0
    for it in c3:
        stripped = {k: v for k, v in it["rec"].items()
                    if k not in ("chosen", "candidates", "scores_kind")}
        if CC.contract_problems(stripped, input_text=it["input"]):
            ptr_raw_violate += 1
        if CC.contract_problems(it["rec"], input_text=it["input"]):
            ptr_full_violate += 1
    return {
        "classes": summary,
        "explainable_all": all(v["explainable"] > 0 for v in summary.values()),
        "pointer_covered": pointer_covered,
        "pointer_n": len(r3),
        "pointer_baseline_covered": sum(1 for b in base if b),
        "pointer_contract_violate_without_structure": ptr_raw_violate,
        "pointer_contract_violate_with_structure": ptr_full_violate,
        "samples": [{"rid": x["rid"], "suite": x["suite"],
                     "text": x["card"].get("text"),
                     "why": x["why"]} for x in (r1 + r2 + r3)[:6]],
        "pointer_samples": [{"rid": x["rid"], "text": x["card"].get("text"),
                             "problems": x["problems"][:2]} for x in r3[:4]],
    }


# ---- V5：三类反例注入 ---------------------------------------------------------

def v5() -> dict:
    c1, c2, c3 = class_render(), class_dispatch(), class_pointer()
    text_items = [it for it in c1 + c2 if it["rec"]["kind"] == "text"
                  and it["rec"].get("instruction")]
    rej_items = [it for it in c1 + c2 if it["rec"]["kind"] == "reject"]
    rej_items += [{"rid": r["src"], "input": "", "table": None, "bag": [],
                   "rec": CC.to_dict(CC.to_contract(r.get("as_record", r["rec"])))}
                  for r in RP.raw_rejects()]

    # ① 篡改引用（ref 指到不存在的结构单元 / span 越界）
    inj1 = caught1 = 0
    first1 = ""
    for it in text_items:
        out = explain(it["rec"], input_text=it["input"], rid=it["rid"],
                      table=it["table"], bag=it["bag"])
        if not out["explainable"]:
            continue
        for fn in (ec.inject_bad_ref, ec.inject_bad_span):
            inj1 += 1
            _bad, probs = fn(out["card"],
                             ec.Src(it["rid"], "X", "text", it["input"],
                                    text=it["rec"]["text"],
                                    instruction=it["rec"]["instruction"],
                                    ref_map=it["rec"]["ref_map"],
                                    evidence=it["rec"]["evidence"],
                                    bag=it["bag"], table=it["table"]))
            if probs:
                caught1 += 1
                first1 = first1 or str(probs[0])

    # ② 拒答带 text（解释卡侧 + 契约谓词侧，两边都必须抓）
    inj2 = caught2 = 0
    first2 = ""
    for it in rej_items:
        out = explain(it["rec"], input_text=it["input"], rid=it["rid"],
                      table=it["table"], bag=it["bag"])
        if not out["explainable"]:
            continue
        inj2 += 1
        bad, probs = ec.inject_reject_text(
            out["card"], ec.Src(it["rid"], "X", "reject", it["input"],
                                reason=it["rec"].get("reason", "")))
        contract_hit = [p for p in CC.contract_problems(dict(it["rec"], text=bad["text"]),
                                                        input_text=it["input"])
                        if p.startswith("C2")]
        allp = probs + contract_hit
        if allp:
            caught2 += 1
            first2 = first2 or str(allp[0])
    # 契约侧：对全部归一后的拒答记录注入 text
    inj2b = caught2b = 0
    for r in RP.raw_rejects() + RP.stored_rejects():
        rec = CC.to_dict(CC.to_contract(r.get("as_record", r["rec"])))
        inj2b += 1
        probs = CC.contract_problems(dict(rec, text="（拒答）注入"), input_text=None)
        if any(p.startswith("C2") for p in probs):
            caught2b += 1

    # ③ 骨架字面含实义词（规则 A + 表门禁 + 渲染出口，三处都必须抓）
    doctored = R.Skeleton("EVIL", "用户说[1]。", (R.SlotSpec("动"),))
    pa = R.rule_a_problems(doctored)
    gate = R.validate_skeleton_table({**R.SKELETONS, "EVIL": doctored})
    render_out = R.render(R.Instruction("EVIL", (2,)),
                          [R.BagItem(2, "说", (4, 5), "动", None, "c1", True)],
                          "用户说加载。", plan_step_id="v5",
                          skeletons={**R.SKELETONS, "EVIL": doctored})
    inj3 = 3
    caught3 = sum(1 for x in (bool(pa), bool(gate),
                              render_out.get("kind") == "reject"
                              and "规则 A" in str(render_out.get("reason", ""))) if x)
    return {
        "inj1_tamper_ref": {"injected": inj1, "caught": caught1,
                            "missed": inj1 - caught1, "sample": first1},
        "inj2_reject_with_text": {"injected": inj2, "caught": caught2,
                                  "missed": inj2 - caught2, "sample": first2},
        "inj2b_contract_reject_text": {"injected": inj2b, "caught": caught2b,
                                       "missed": inj2b - caught2b},
        "inj3_skeleton_content_word": {"injected": inj3, "caught": caught3,
                                       "missed": inj3 - caught3,
                                       "sample": pa[0] if pa else "",
                                       "gate_sample": next(
                                           (g for g in gate if "规则 A" in g),
                                           gate[0] if gate else ""),
                                       "render_reason": render_out.get("reason", "")},
        "missed": (inj1 - caught1) + (inj2 - caught2) + (inj2b - caught2b)
                  + (inj3 - caught3),
    }
