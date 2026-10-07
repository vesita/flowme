"""Tier-2「自由句式」—— 可判的降级版（自由的句法 + 受锁的词）。

对应 `PREREG.md` §3。三级锁，缺一即 fail-closed：

1. **内容词受锁**：每条内容片段必须是 ② 筛选通过、且 `lexicon.type_of` 能定型的候选，
   span 逐字落在输入上（G2 零信息新增）；
2. **功能词受锁**：间隙连接 / 句末只能取本文件的**封闭表**（形式词 ⊕ 逻辑词），
   逻辑词必须**输入逐字有据**（§7.7 两级规则）；
3. **输出受锁**：`t2_parse` 从**输出文本本身**（不读结构指令）独立复原
   「槽序 + 功能词序列」，**0 解析 ⇒ `parse_no_way`、≥2 解析 ⇒ `parse_ambiguous`
   ⇒ 该产出不计分**（struct_supervision A 臂口径）；复原结果必须与渲染器 `ref_map` 逐条相等。

自由的部分 = **句式实例化**：槽的类型序列与长度由输入句的类型库存当场给出（不查 25 条闭合表）、
间隙连接与句式（陈述/疑问/感叹，由输入末标点定）按规则选。**词序不自由** ——
真卡无题元 ⇒ fail-closed 到「只复述、不换方向」，恒取原文 span 升序（互换注入必拒）。
"""
from __future__ import annotations

import itertools
import re

from lexicon import FORMAL_PUNCT, FORMAL_WORDS, LOGIC_WORDS, SLOT_TYPES, type_of

# ---- 封闭功能词表（import 时审计，带病不进表）--------------------------------

#: 间隙连接 · 形式类（零命题贡献 ⇒ 可自由插，§7.7 第 1 条）。
GAP_FORMAL: tuple[str, ...] = ("，", "、", "；", "：", "和", "与")

#: 间隙连接 · 逻辑类（**有真值贡献** ⇒ 只许输入逐字有据，§7.7 第 2 条）。
GAP_LOGIC: tuple[str, ...] = LOGIC_WORDS          # 因为 所以 但 要 能 可以

#: 句末（形式类；由输入末标点定或默认 `。`）。
TERMINAL: tuple[str, ...] = ("。", "！", "？", "!", "?", "…")

GAP_ALL: tuple[str, ...] = GAP_FORMAL + GAP_LOGIC
EMPTY_GAP = ""
TERMINAL_SET = frozenset(TERMINAL)
GAP_LOGIC_SET = frozenset(GAP_LOGIC)
GAP_FORMAL_SET = frozenset(GAP_FORMAL)

SKELETON_ID = "T2"
MAX_PARSE_WAYS = 2                      # 计数到 2 即停：≥2 = 歧义 = 不计分
_SLOT_RE = re.compile(r"^p\d+$")


def audit_table() -> dict:
    """规则 A 的词典谓词（Tier-2 版）：功能词表字面只许形式词 ∪ 已登记逻辑词。"""
    bad: list[str] = []
    for w in GAP_ALL + TERMINAL:
        rest = w
        for x in LOGIC_WORDS + FORMAL_WORDS:
            rest = rest.replace(x, "")
        for ch in FORMAL_PUNCT + " \t":
            rest = rest.replace(ch, "")
        if rest:
            bad.append(w)
    if bad:
        raise ValueError(f"Tier-2 功能词表混入非形式/非逻辑字面（规则 A 违例）: {bad}")
    overlap = set(GAP_ALL) & set(TERMINAL)
    if overlap:
        raise ValueError(f"间隙表与句末表必须不相交（解析歧义源）: {sorted(overlap)}")
    if len(set(GAP_ALL + TERMINAL)) != len(GAP_ALL + TERMINAL):
        raise ValueError("Tier-2 功能词表有重复项")
    return {"gap_formal": list(GAP_FORMAL), "gap_logic": list(GAP_LOGIC),
            "terminal": list(TERMINAL), "n_entries": len(GAP_ALL) + len(TERMINAL)}


TABLE_AUDIT = audit_table()


# ---- 结构指令 ----------------------------------------------------------------

def _names(struct: dict) -> list[str]:
    """槽名按自然序（p1 < p2 < … < p10，不是字典序）。"""
    return sorted(struct.get("assign") or {},
                  key=lambda k: int(k[1:]) if _SLOT_RE.match(k) else -1)


def _content_unit(cid: str, ref: dict, pos: int) -> dict:
    return {"unit_id": f"ref:{cid}", "cls": "content", "text": ref["surface"],
            "ref": cid, "span": [ref["s0"], ref["e0"]], "pos": pos}


def _func_unit(text: str, cls: str, pos: int, k: int) -> dict:
    return {"unit_id": f"t2#{k}", "cls": cls, "text": text,
            "ref": SKELETON_ID, "span": None, "pos": pos}


def select_fragments(valid: list[dict]) -> list[dict]:
    """片段选择（PREREG §3.2-1）：已定型的有效候选 → 同 span 去重 → 贪心
    leftmost-longest 取互不重叠的升序序列。构造即保证方向安全。"""
    pool = [c for c in valid if c.get("valid") and c.get("type") in SLOT_TYPES]
    seen: set[tuple[int, int]] = set()
    uniq: list[dict] = []
    for c in sorted(pool, key=lambda x: (x["s0"], -(x["e0"] - x["s0"]),
                                         x["card"], x["cid"])):
        key = (c["s0"], c["e0"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    out: list[dict] = []
    last = -1
    for c in uniq:
        if c["s0"] >= last:
            out.append(c)
            last = c["e0"]
    return out


def _gap_connector(inp: str, a: int, b: int) -> dict:
    """PREREG §3.2-2：① 有据逻辑词（最长优先→表序）② gap 内从左第一个形式标点 ③ 不插。"""
    gap = inp[a:b]
    for w in sorted(GAP_LOGIC, key=lambda x: (-len(x), GAP_LOGIC.index(x))):
        if w in gap:
            return {"cls": "logic", "text": w}
    for ch in gap:
        if ch in GAP_FORMAL_SET:
            return {"cls": "formal", "text": ch}
    return {"cls": "formal", "text": EMPTY_GAP}


def _terminal(inp: str) -> dict:
    """PREREG §3.2-3：输入末字符是句末标点 ⇒ 逐字复用；否则 `。`。"""
    last = inp[-1] if inp else ""
    return {"cls": "formal", "text": last if last in TERMINAL_SET else "。"}


def build_struct(inp: str, frags: list[dict]) -> dict:
    """片段序列 → Tier-2 结构指令（纯函数）。"""
    assign: dict[str, str] = {}
    slots: dict[str, str] = {}
    refs: dict[str, dict] = {}
    for i, c in enumerate(frags, 1):
        assign[f"p{i}"] = c["cid"]
        slots[f"p{i}"] = c["type"]
        refs[c["cid"]] = {"cid": c["cid"], "card": c["card"], "class_id": c["class_id"],
                          "class_name": c["class_name"], "s0": c["s0"], "e0": c["e0"],
                          "surface": c["surface"], "type": c["type"],
                          "theme": c.get("theme")}
    funcs = [_gap_connector(inp, frags[i]["e0"], frags[i + 1]["s0"])
             for i in range(len(frags) - 1)]
    funcs.append(_terminal(inp))
    return {"tier": 2, "skeleton": SKELETON_ID, "slots": slots, "assign": assign,
            "refs": refs, "evidence": list(assign.values()), "funcs": funcs,
            "order": "span_asc"}


# ---- ⑤ 结构侧检查 -------------------------------------------------------------

def t2_structure_problems(inp: str, struct: dict,
                          valid_index: dict[str, dict]) -> list[str]:
    """schema + evidence + 方向 + 功能词门禁。返回问题列表（非空 = 拒答）。"""
    probs: list[str] = []
    if struct.get("tier") != 2 or struct.get("skeleton") != SKELETON_ID:
        return ["tier2_schema_bad"]
    slots: dict = struct.get("slots") or {}
    assign: dict = struct.get("assign") or {}
    refs: dict = struct.get("refs") or {}
    funcs: list = struct.get("funcs") or []
    names = _names(struct)
    n = len(names)

    # --- schema ---
    if any(not _SLOT_RE.match(k) for k in names):
        probs.append("tier2_schema_extra_slot")
    elif names != [f"p{i}" for i in range(1, n + 1)]:
        probs.append("tier2_schema_missing_slot")        # 编号有洞 = 缺块
    if set(assign) - set(slots):
        probs.append("tier2_schema_missing_slot")        # 槽没声明类型
    elif set(slots) - set(assign):
        probs.append("tier2_schema_extra_slot")          # 多声明了槽
    if n == 0:
        probs.append("tier2_schema_missing_slot")        # 零内容 = 无句
    if len(funcs) != n:
        probs.append("tier2_func_count_mismatch")
    if set(refs) != set(assign.values()):
        probs.append("tier2_ref_set_mismatch")
    if len(set(assign.values())) != len(assign):
        probs.append("tier2_duplicate_ref")
    if probs:
        return probs

    # --- 逐槽位：逐字 / 有效集 / 类型 ---
    filled: list[dict] = []
    for i, k in enumerate(names):
        cid = assign[k]
        ref = refs.get(cid)
        if ref is None:
            probs.append("tier2_schema_missing_slot")
            continue
        s0, e0, surf = ref.get("s0"), ref.get("e0"), ref.get("surface", "")
        if not (isinstance(s0, int) and isinstance(e0, int)
                and 0 <= s0 < e0 <= len(inp) and inp[s0:e0] == surf):
            probs.append("tier2_span_not_verbatim")     # 越界 / 改字 / 丢字
            continue
        good = valid_index.get(cid)
        if good is None or (good["s0"], good["e0"], good["surface"], good["card"],
                            good["class_name"]) != (s0, e0, surf, ref.get("card"),
                                                    ref.get("class_name")):
            probs.append("tier2_not_in_valid_set")      # G2 硬门
            continue
        recomputed = type_of(ref.get("card", ""), ref.get("class_name", ""), surf)
        if recomputed is None or recomputed not in SLOT_TYPES \
                or ref.get("type") != recomputed or slots.get(k) != recomputed:
            probs.append("tier2_type_mismatch")         # 未知类型 fail-closed
            continue
        filled.append(ref)

    # --- 方向：恒 span 升序 + 互不重叠（只复述、不换方向）---
    if not probs:
        for a, b in itertools.pairwise(filled):
            if b["s0"] <= a["s0"]:
                probs.append("tier2_reorder_unjustified")
                break
            if b["s0"] < a["e0"]:
                probs.append("tier2_span_overlap")
                break

    # --- 功能词：封闭表 + 类别 + 逻辑词有据 ---
    if not probs:
        for i, f in enumerate(funcs):
            w = (f or {}).get("text")
            cls = (f or {}).get("cls")
            if i < n - 1:                                # 间隙
                if w == EMPTY_GAP:
                    if cls != "formal":
                        probs.append("tier2_func_not_in_table")
                elif w in GAP_LOGIC:
                    if cls != "logic":
                        probs.append("tier2_func_not_in_table")
                    elif w not in inp:
                        probs.append("tier2_logic_no_evidence")
                elif w in GAP_FORMAL:
                    if cls != "formal":
                        probs.append("tier2_func_not_in_table")
                else:
                    probs.append("tier2_func_not_in_table")
            else:                                        # 句末
                if w not in TERMINAL_SET or cls != "formal":
                    probs.append("tier2_func_not_in_table")
            if probs:
                break
    return probs


# ---- ④ 渲染（确定性纯函数）---------------------------------------------------

def t2_render(struct: dict) -> tuple[str, list[dict]]:
    """结构指令 → (文本, ref_map)。纯函数：同输入逐字相同。"""
    names = _names(struct)
    parts: list[str] = []
    ref_map: list[dict] = []
    pos = 0
    k = 0
    for i, name in enumerate(names):
        cid = struct["assign"][name]
        ref = struct["refs"][cid]
        surf = ref["surface"]
        parts.append(surf)
        ref_map.append(_content_unit(cid, ref, pos))
        pos += len(surf)
        f = struct["funcs"][i]
        w = f["text"]
        if w:
            parts.append(w)
            ref_map.append(_func_unit(w, f["cls"], pos, k))
            pos += len(w)
            k += 1
    return "".join(parts), ref_map


# ---- ⑤ 解引用侧（独立复核）---------------------------------------------------

def t2_deref_verify(text_out: str, struct: dict, inp: str,
                    valid_index: dict[str, dict]) -> tuple[list[str], list[dict]]:
    """重走指令核对输出：丢块/改面 ⇒ `ref_render_mismatch`；多吐 ⇒ `unmapped_output`。"""
    probs: list[str] = []
    _, expected = t2_render(struct)
    pos = 0
    for unit in expected:
        if text_out.startswith(unit["text"], pos):
            pos += len(unit["text"])
            continue
        probs.append("tier2_ref_render_mismatch")
        break
    else:
        if pos != len(text_out):
            probs.append("tier2_unmapped_output")
    if probs:
        return probs, expected

    assign = struct.get("assign") or {}
    if {u["ref"] for u in expected if u["cls"] == "content"} != set(assign.values()):
        probs.append("tier2_ref_set_mismatch")
    for u in expected:
        if u["cls"] != "content":
            continue
        s0, e0 = u["span"]
        if not (0 <= s0 < e0 <= len(inp)) or inp[s0:e0] != u["text"]:
            probs.append("tier2_span_not_verbatim")
            break
        if valid_index.get(u["ref"]) is None:
            probs.append("tier2_not_in_valid_set")
            break
    return probs, expected


# ---- 解析器（不读结构指令，独立实现）------------------------------------------

def t2_parse(text_out: str, inp: str,
             valid_index: dict[str, dict]) -> tuple[list[dict] | None, str | None]:
    """从输出文本独立复原「槽序 + 功能词序列」。

    文法：`内容 (间隙连接 内容)* 句末` —— 必须以内容开头、以句末结尾、内容 ≥1；
    内容 token = 有效且已定型候选的 surface，并**强制 span 升序 + 互不重叠**；
    逻辑词类间隙须输入逐字有据；同 span 的多卡视作同一内容占位（cid 取规范序第一条）。

    返回 `(units, None)`（唯一解析）/ `(None, "parse_no_way")` / `(None, "parse_ambiguous")`。
    **解析不出 ⇒ 调用方不许计分**（A 臂口径）。
    """
    by_surf: dict[str, dict[tuple[int, int], dict]] = {}
    for c in valid_index.values():
        if not c.get("valid") or c.get("type") not in SLOT_TYPES:
            continue
        surf = c["surface"]
        bucket = by_surf.setdefault(surf, {})
        key = (c["s0"], c["e0"])
        cur = bucket.get(key)
        if cur is None or (c["card"], c["cid"]) < (cur["card"], cur["cid"]):
            bucket[key] = c
    if not by_surf:
        return None, "parse_no_way"

    surfs = sorted(by_surf, key=lambda s: (-len(s), s))
    n = len(text_out)
    ways: list[list[dict]] = []
    fail: set[tuple[int, int]] = set()
    capped = False

    def dfs(pos: int, last_end: int, acc: list[dict]) -> None:
        nonlocal capped
        if len(ways) >= MAX_PARSE_WAYS:
            capped = True
            return
        state = (pos, last_end)
        if state in fail:
            return
        w0 = len(ways)
        for surf in surfs:
            if not text_out.startswith(surf, pos):
                continue
            np = pos + len(surf)
            for c in sorted(by_surf[surf].values(), key=lambda x: (x["s0"], x["e0"])):
                if len(ways) >= MAX_PARSE_WAYS:
                    capped = True
                    return
                if c["s0"] < last_end:               # 升序 + 互不重叠
                    continue
                unit_c = _content_unit(c["cid"], {"surface": surf, "s0": c["s0"],
                                                  "e0": c["e0"]}, pos)
                acc2 = acc + [unit_c]
                rest = text_out[np:]
                if rest in TERMINAL_SET:             # 句末整段收尾
                    ways.append(acc2 + [_func_unit(rest, "formal", np, _k(acc2))])
                    continue
                for g in (EMPTY_GAP,) + GAP_ALL:
                    if g:
                        if not rest.startswith(g):
                            continue
                        if g in GAP_LOGIC_SET and g not in inp:
                            continue
                        nx = np + len(g)
                    else:
                        nx = np
                    if nx >= n:
                        continue
                    acc3 = acc2 + ([_func_unit(g, "logic" if g in GAP_LOGIC_SET
                                               else "formal", np, _k(acc2))]
                                   if g else [])
                    dfs(nx, c["e0"], acc3)
        if len(ways) == w0 and not capped and len(ways) < MAX_PARSE_WAYS:
            fail.add(state)

    def _k(units: list[dict]) -> int:
        return sum(1 for u in units if u["cls"] != "content")

    dfs(0, 0, [])
    if len(ways) == 1:
        return ways[0], None
    if not ways:
        return None, "parse_no_way"
    return None, "parse_ambiguous"


# ---- 主流程：Tier-2 降级链 ----------------------------------------------------

def _reject(reason: str, **extra) -> dict:
    out: dict = {"status": "reject", "stage": "tier2", "reason": reason}
    out.update(extra)
    assert "text" not in out, "拒答记录里不许出现 text"
    return out


def run_tier2(inp: str, candidates: list[dict]) -> dict:
    """Tier-2 兜底：候选（② 筛选后，含 valid 标记）→ 降级链 → 结构/解析全过才出文本。

    PREREG §3.4：R0 = 全片段；依次去尾；直到最左单片段；全败 ⇒ `tier2_all_attempts_failed`。
    """
    base: dict = {"n_candidates": len(candidates),
                  "n_valid": sum(1 for c in candidates if c.get("valid"))}
    if not candidates:
        return _reject("tier2_no_valid_candidate", **base)
    valid = [c for c in candidates if c.get("valid")]
    valid_index = {c["cid"]: c for c in valid}
    if not valid:
        return _reject("tier2_no_valid_candidate", **base)
    if not any(c.get("type") in SLOT_TYPES for c in valid):
        return _reject("tier2_type_inventory",
                       have=sorted({str(c.get("type")) for c in valid}), **base)

    frags = select_fragments(valid)
    if not frags:
        return _reject("tier2_type_inventory", **base)

    attempts: list[dict] = []
    r0: dict | None = None
    for rung in range(len(frags)):
        use = frags[:len(frags) - rung]
        struct = build_struct(inp, use)
        probs = t2_structure_problems(inp, struct, valid_index)
        if probs:
            attempts.append({"rung": rung, "n_frag": len(use),
                             "structure": probs[0], "parse": None, "ref": None})
        else:
            out_text, ref_map = t2_render(struct)
            dprobs, _ = t2_deref_verify(out_text, struct, inp, valid_index)
            if dprobs:
                attempts.append({"rung": rung, "n_frag": len(use),
                                 "structure": "ok", "parse": None, "ref": dprobs[0]})
            else:
                units, reason = t2_parse(out_text, inp, valid_index)
                if reason:
                    attempts.append({"rung": rung, "n_frag": len(use),
                                     "structure": "ok", "parse": "tier2_" + reason,
                                     "ref": None})
                elif units != ref_map:
                    attempts.append({"rung": rung, "n_frag": len(use),
                                     "structure": "ok", "parse": "ok",
                                     "ref": "tier2_ref_render_mismatch"})
                else:
                    if rung == 0:
                        r0 = {"structure": "ok", "parse": "ok", "ref": "ok"}
                    return {
                        "status": "ok", "tier": 2, "text": out_text,
                        "skeleton": SKELETON_ID, "structure": struct,
                        "mapping": ref_map, "parse": units, "r0": r0 or {
                            "structure": "ok", "parse": "ok", "ref": "ok"},
                        "rung": rung, "n_frag": len(use), "attempts": attempts,
                        "candidates": candidates, **base,
                    }
        if rung == 0:
            r0 = dict(attempts[-1])
    return _reject("tier2_all_attempts_failed", attempts=attempts, r0=r0, **base)


def run_tier2_struct(inp: str, struct: dict, candidates: list[dict],
                     text_out: str | None = None) -> dict:
    """**注入入口（只给对抗组用）**：给定（可能被篡改的）结构指令与（可能被污染的）文本，
    走与正式流完全相同的四道门：结构侧 → 解引用侧 → 独立解析 → 解析/ref_map 一致。"""
    valid = [c for c in candidates if c.get("valid")]
    valid_index = {c["cid"]: c for c in valid}
    probs = t2_structure_problems(inp, struct, valid_index)
    if probs:
        return _reject(probs[0], n_candidates=len(candidates), n_valid=len(valid))
    rendered, ref_map = t2_render(struct)
    out_text = rendered if text_out is None else text_out
    dprobs, _ = t2_deref_verify(out_text, struct, inp, valid_index)
    if dprobs:
        return _reject(dprobs[0], n_candidates=len(candidates), n_valid=len(valid))
    units, reason = t2_parse(out_text, inp, valid_index)
    if reason:
        return _reject("tier2_" + reason, n_candidates=len(candidates),
                       n_valid=len(valid))
    if units != ref_map:
        return _reject("tier2_ref_render_mismatch", n_candidates=len(candidates),
                       n_valid=len(valid))
    return {"status": "ok", "tier": 2, "text": out_text, "skeleton": SKELETON_ID,
            "structure": struct, "mapping": ref_map, "parse": units,
            "r0": {"structure": "ok", "parse": "ok", "ref": "ok"}, "rung": 0,
            "n_frag": len(struct.get("assign") or {}), "attempts": [],
            "candidates": candidates, "n_candidates": len(candidates),
            "n_valid": len(valid)}


# ---- 免费规则（C5 并列用）-----------------------------------------------------

def naive_n1(inp: str, candidates: list[dict]) -> dict:
    """免费规则 N1 = **最左单片段 + 同一句末规则**（= Tier-2 降级链的末档）。

    走完全一样的结构/解析/解引用检查；不过 ⇒ reject（不计分）。
    """
    valid = [c for c in candidates if c.get("valid")]
    valid_index = {c["cid"]: c for c in valid}
    frags = select_fragments(valid)
    if not frags:
        return _reject("tier2_type_inventory")
    struct = build_struct(inp, [frags[0]])
    if t2_structure_problems(inp, struct, valid_index):
        return _reject("tier2_naive_structure")
    out_text, ref_map = t2_render(struct)
    dprobs, _ = t2_deref_verify(out_text, struct, inp, valid_index)
    if dprobs:
        return _reject("tier2_naive_deref", detail=dprobs[0])
    units, reason = t2_parse(out_text, inp, valid_index)
    if reason:
        return _reject(f"tier2_{reason}")
    if units != ref_map:
        return _reject("tier2_ref_render_mismatch")
    return {"status": "ok", "text": out_text, "structure": struct,
            "mapping": ref_map, "candidates": candidates,
            "n_candidates": len(candidates), "n_valid": len(valid)}


def naive_n0(inp: str, candidates: list[dict]) -> dict:
    """免费规则 N0 = **直接复述原文**（把整句当一个内容片段）。"""
    valid_index = {c["cid"]: c for c in candidates if c.get("valid")}
    hit = next((c for c in valid_index.values()
                if c["s0"] == 0 and c["e0"] == len(inp) and c["type"] in SLOT_TYPES),
               None)
    if hit is None:
        return _reject("tier2_not_in_valid_set", note="整句不是被筛为有效的已定型候选")
    fake = dict(hit)
    struct = build_struct(inp, [fake])
    if t2_structure_problems(inp, struct, valid_index):
        return _reject("tier2_naive_structure")
    out_text, ref_map = t2_render(struct)
    units, reason = t2_parse(out_text, inp, valid_index)
    if reason or units != ref_map:
        return _reject("tier2_parse_mismatch")
    return {"status": "ok", "text": out_text, "structure": struct,
            "mapping": ref_map, "candidates": candidates,
            "n_candidates": len(candidates), "n_valid": len(valid_index)}
