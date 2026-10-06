"""可核对谓词 —— ②筛选区 与 ⑤结构侧（schema / evidence）。

设计原则（dev-notes/16 §7.2）：**能用可核对谓词判的，就不许交给学出来的分数**。
每条谓词都是纯函数、可审计、跑前写死（见 PREREG §6），跑中不许改。

  - `filter_candidates`   ② 筛选区：5 条谓词，全过才算 valid；
  - `structure_problems`  ⑤ 结构侧：schema + evidence（含类型 / 题元 / 顺序 / 逻辑词门禁）。

返回值约定：**问题列表非空 = 拒答**；`problems[0]` 是首要原因（对抗组按它对账）。
"""
from __future__ import annotations

from lexicon import FORMAL_PUNCT, LOGIC_WORDS, type_of
from skeletons import Skeleton

#: 背景类的类别 id（所有内置卡都约定 index 0 是背景）。
BACKGROUND_ID = 0

#: 背景类的类别名（不发射切片的那一类）。
BACKGROUND_NAMES = frozenset({"背景", "无关系", "无代词", "中性", "无归属", "非成语", "无人物"})


# ---- ② 筛选区 ---------------------------------------------------------------

def filter_candidates(text: str, candidates: list[dict],
                      class_index: dict[str, tuple[str, ...]]) -> list[dict]:
    """给每条候选打 `valid` 与 `failed`（5 条谓词逐条报）。

    F1 in_bounds / F2 non_empty / F3 canonical_class / F4 no_punct / F5 verbatim
    """
    n = len(text)
    out: list[dict] = []
    for c in candidates:
        s0, e0 = c.get("s0"), c.get("e0")
        surface = c.get("surface", "")
        failed: list[str] = []

        # F1 区间在原文内（且 s0 < e0）
        if not (isinstance(s0, int) and isinstance(e0, int) and 0 <= s0 < e0 <= n):
            failed.append("F1_out_of_bounds")
        # F2 非空
        if not surface or (isinstance(s0, int) and isinstance(e0, int) and e0 - s0 <= 0):
            failed.append("F2_empty")
        # F3 类别是规范名（卡的类别体系里有它，且不是背景类）
        names = class_index.get(c.get("card", ""), None)
        cid_, cname = c.get("class_id"), c.get("class_name", "")
        if names is None or not isinstance(cid_, int) or cid_ <= 0 or cname not in names \
                or cname in BACKGROUND_NAMES:
            failed.append("F3_bad_class")
        # F4 不跨标点
        if any(ch in FORMAL_PUNCT or ch.isspace() for ch in surface):
            failed.append("F4_has_punct")
        # F5 可回溯：逐字等于原文区间
        if "F1_out_of_bounds" not in failed and text[s0:e0] != surface:
            failed.append("F5_not_verbatim")

        out.append({**c, "valid": not failed, "failed": failed})
    return out


# ---- ⑤ 结构侧：schema / evidence --------------------------------------------

def _order_ok(refs_in_order: list[dict]) -> str | None:
    """无题元标签时的**只复述、不换方向**（§7.7 fail-closed 分支）。"""
    for prev, cur in zip(refs_in_order, refs_in_order[1:]):
        if cur["s0"] <= prev["s0"]:
            return "reorder_unjustified"
        if cur["s0"] < prev["e0"]:
            return "span_overlap"
    return None


def structure_problems(text: str, struct: dict, skel: Skeleton,
                       valid_index: dict[str, dict],
                       bag_themed: bool) -> list[str]:
    """结构指令的 schema + evidence 谓词。返回问题列表（非空 = 拒答）。"""
    probs: list[str] = []
    assign: dict = struct.get("assign") or {}
    refs: dict = struct.get("refs") or {}
    slot_names = [s.name for s in skel.slots]

    # --- schema：块必须齐 ---
    missing = [n for n in slot_names if n not in assign]
    if missing:
        probs.append("schema_missing_slot")
    extra = [n for n in assign if n not in slot_names]
    if extra:
        probs.append("schema_extra_slot")
    if set(refs) != set(assign.values()):
        probs.append("ref_set_mismatch")
    if len(set(assign.values())) != len(assign):
        probs.append("duplicate_ref")
    if probs:
        return probs

    # --- evidence：逻辑词必须输入有据（否则就是编的因果）---
    for w in skel.logic_words:
        if w not in text:
            probs.append("logic_no_evidence")
            break

    # --- 必出引用 ---
    for r in struct.get("evidence", []) or []:
        if r not in assign.values():
            probs.append("evidence_not_assigned")
            break

    # --- 逐槽位：逐字 / 属于有效集 / 类型 / 题元 ---
    filled: list[dict] = []
    for slot in skel.slots:
        ref = refs[assign[slot.name]]
        s0, e0, surface = ref.get("s0"), ref.get("e0"), ref.get("surface", "")
        bounds_ok = isinstance(s0, int) and isinstance(e0, int) and 0 <= s0 < e0 <= len(text)
        if not bounds_ok or text[s0:e0] != surface:
            probs.append("span_not_verbatim")     # 越界 / 改动 都落在这里
            continue
        good = valid_index.get(ref.get("cid"))
        if good is None or (good["s0"], good["e0"], good["surface"], good["card"],
                            good["class_name"]) != (s0, e0, surface, ref.get("card"),
                                                    ref.get("class_name")):
            probs.append("not_in_valid_set")       # G2 的硬门：必须是被筛为有效的候选
            continue
        recomputed = type_of(ref.get("card", ""), ref.get("class_name", ""), surface)
        if recomputed is None or ref.get("type") != recomputed:
            probs.append("type_mismatch")
            continue
        if recomputed != slot.type:
            probs.append("type_mismatch")
            continue
        if bag_themed:
            if slot.theme is None:
                if ref.get("theme") is not None:
                    probs.append("theme_conflict")
                    continue
            elif ref.get("theme") != slot.theme:
                probs.append("theme_conflict" if ref.get("theme") is not None
                             else "theme_missing")
                continue
        filled.append(ref)

    # --- 顺序：无题元标签 ⇒ 只复述、不换方向 ---
    if not probs and not bag_themed:
        why = _order_ok(filled)
        if why:
            probs.append(why)

    return probs
