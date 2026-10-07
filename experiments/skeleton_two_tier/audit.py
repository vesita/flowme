"""G2/G3/G4 的**独立事后复核** —— 不复用流内检查的结论，从记录重新推一遍。

- G2 零信息新增：每个内容片段必须逐字等于原文 span，且 cid ∈ 该句「被筛为有效」的候选；
- G3 引用映射完整：`ref_map` 各区间 + 功能词字面**恰好铺满**输出（无洞无尾）、
  映射数 == 槽位数、链 片段 ↦ ref ↦ span 全通；Tier-2 另加「**独立解析器复原 == ref_map**」；
- G4 逐字可复现：两条记录 JSON 全等。
"""
from __future__ import annotations

import json

from lexicon import type_of

from t2core import (EMPTY_GAP, GAP_ALL, TERMINAL_SET, _names, t2_parse, t2_render)


# ---- 重渲染（按 tier 分派）----------------------------------------------------

def rerender(rec: dict) -> str | None:
    """结构 → 文本（纯函数，独立调用）。"""
    if rec.get("tier") == 2:
        return t2_render(rec["structure"])[0]
    from render import render_structure
    from skeletons import BY_ID
    return render_structure(rec["structure"], BY_ID[rec["skeleton"]])


# ---- 期望的单元序列（按 tier 分派）-------------------------------------------

def expected_units(rec: dict) -> list[dict]:
    """从**结构**（不是从 ref_map）推出应有序列：[content|func] × pos。"""
    units: list[dict] = []
    pos = 0
    if rec.get("tier") == 2:
        struct = rec["structure"]
        names = _names(struct)
        k = 0
        for i, name in enumerate(names):
            cid = struct["assign"][name]
            ref = struct["refs"][cid]
            units.append({"cls": "content", "text": ref["surface"], "ref": cid,
                          "span": [ref["s0"], ref["e0"]], "pos": pos})
            pos += len(ref["surface"])
            f = struct["funcs"][i]
            if f["text"]:
                units.append({"cls": f["cls"], "text": f["text"], "ref": None,
                              "span": None, "pos": pos, "unit_id": f"t2#{k}"})
                pos += len(f["text"])
                k += 1
        return units

    from skeletons import BY_ID
    skel = BY_ID[rec["skeleton"]]
    struct = rec["structure"]
    for kind, tok in skel.lit_tokens():
        if kind == "lit":
            units.append({"cls": "formal", "text": tok, "ref": None, "span": None,
                          "pos": pos})
            pos += len(tok)
        else:
            ref = struct["refs"][struct["assign"][tok]]
            units.append({"cls": "content", "text": ref["surface"], "ref": ref["cid"],
                          "span": [ref["s0"], ref["e0"]], "pos": pos, "slot": tok})
            pos += len(ref["surface"])
    return units


# ---- G2 / G3 -----------------------------------------------------------------

def audit_record(rec: dict, src_text: str) -> dict:
    """对一条 ok 记录独立复核 G2 / G3。返回 {"g2": [...], "g3": [...]}（空 = 过）。"""
    g2: list[str] = []
    g3: list[str] = []
    if rec.get("status") != "ok":
        return {"g2": g2, "g3": g3}

    # --- G3-a：结构重渲染必须逐字相同（纯函数可复现）---
    try:
        if rerender(rec) != rec["text"]:
            g3.append("rerender_mismatch")
    except Exception as exc:                      # noqa: BLE001
        g3.append(f"rerender_error:{type(exc).__name__}")

    # --- G3-b：由**结构**推出的单元序列必须恰好铺满输出 ---
    units = expected_units(rec)
    ptr = 0
    for u in units:
        if u["pos"] != ptr:
            g3.append("coverage_gap")
            break
        ptr += len(u["text"])
    else:
        if ptr != len(rec["text"]):
            g3.append("tail_uncovered")
        if rec["text"][:ptr] != "".join(u["text"] for u in units):
            g3.append("concat_mismatch")

    # --- 映射条数 == 内容槽位数（Tier-2 的 ref_map 还含功能词条目，只数内容）---
    n_content = sum(1 for u in units if u["cls"] == "content")
    mapping_all = rec.get("mapping") or []
    is_t2 = rec.get("tier") == 2
    content_map = [m for m in mapping_all if not is_t2 or m.get("cls") == "content"]
    if len(content_map) != n_content:
        g3.append("mapping_count_mismatch")
    if is_t2 and len(mapping_all) - len(content_map) != \
            sum(1 for u in units if u["cls"] != "content"):
        g3.append("func_mapping_count_mismatch")

    # --- G3-c：链 片段 ↦ ref ↦ span + G2 逐字回溯到「被筛为有效」的候选 ---
    valid_map = {c["cid"]: c for c in rec.get("candidates", []) if c.get("valid")}
    for m in content_map:
        if is_t2:
            cid, frag, pos, span = m["ref"], m["text"], m["pos"], m["span"]
        else:
            cid, frag, pos, span = m["ref"], m["frag"], m["pos"], m["span"]
        ref = rec["structure"]["refs"].get(cid)
        if ref is None or [ref["s0"], ref["e0"]] != list(span) or ref["surface"] != frag:
            g3.append("chain_broken")
        if rec["text"][pos:pos + len(frag)] != frag:
            g3.append("frag_not_at_pos")
        s0, e0 = span
        if not (0 <= s0 < e0 <= len(src_text)) or src_text[s0:e0] != frag:
            g2.append("not_verbatim")
        c = valid_map.get(cid)
        if c is None or (c["s0"], c["e0"], c["surface"]) != (s0, e0, frag):
            g2.append("not_filtered_valid")
        if c is None or c.get("failed"):
            g2.append("candidate_was_invalid")
        if c is not None and type_of(c["card"], c["class_name"], c["surface"]) != c.get("type"):
            g2.append("type_recompute_mismatch")

    # --- Tier-2 附加：**独立解析器**必须复原出同一个 ref_map（解析不出 ⇒ 不计分）---
    if is_t2:
        units_p, reason = t2_parse(rec["text"], src_text, valid_map)
        if reason:
            g3.append(f"parse_{reason}")
        elif units_p != mapping_all:
            g3.append("parse_ref_map_mismatch")
        # 功能词门禁（解析侧独立复核）
        for m in mapping_all:
            if m["cls"] == "content":
                continue
            if m["text"] not in TERMINAL_SET and m["text"] not in GAP_ALL \
                    and m["text"] != EMPTY_GAP:
                g3.append("func_not_in_table")
            if m["cls"] == "logic" and m["text"] not in src_text:
                g2.append("logic_no_evidence")
    return {"g2": g2, "g3": g3}


# ---- G4 ----------------------------------------------------------------------

def same_record(a: dict, b: dict) -> bool:
    return json.dumps(a, sort_keys=True, ensure_ascii=False) == \
        json.dumps(b, sort_keys=True, ensure_ascii=False)


def diff_record(a: dict, b: dict) -> list[str]:
    """G4 不一致时给出最浅的差异键（只用于报日志）。"""
    ka, kb = set(a), set(b)
    out = [f"key_only_in_A:{sorted(ka - kb)}", f"key_only_in_B:{sorted(kb - ka)}"]
    for k in sorted(ka & kb):
        if a[k] != b[k]:
            out.append(k)
    return out[:8]
