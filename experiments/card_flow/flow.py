"""五段流：①候选 → ②筛选 → ③重整（骨架+槽位 ⇒ 结构指令）→ ④渲染 → ⑤解引用。

每段的职责与落点：
  ① 候选区  `proposers.py::real_propose / fake_propose` —— 指针卡在真句上抽切片（类别+span），
            内容词**天生可回溯**（dev-notes/16 §7.3 风险最低的一档）。
  ② 筛选区  `predicates.filter_candidates` —— 5 条可核对谓词，全过才算 valid。
  ③ 重整区  `flow.compose` —— 手写骨架表 + 槽位指派 ⇒ **结构指令**（不是文本）。
  ④ 渲染    `render.render_structure` —— 确定性纯函数。
  ⑤ 解引用  `predicates.structure_problems`（结构侧）+ `render.deref_verify`（解引用侧），
            两道检查都过才产出文本，并留存引用映射。

**fail-closed**：任何一段出问题都返回 `{"status":"reject", ...}`，绝不返回半句。
"""
from __future__ import annotations

import itertools

from predicates import filter_candidates, structure_problems
from render import deref_verify, render_structure
from skeletons import BY_ID, SKELETONS

#: 一个骨架的指派搜索上限（超了就跳过该骨架，记 `combo_cap`）。
COMBO_CAP = 20000


# ---- ③ 重整区 ---------------------------------------------------------------

def _pools(valid: list[dict], skel, bag_themed: bool) -> dict[str, list[dict]]:
    """每个槽位的候选池（按类型；带题元时再按题元收；按 span 去重，保持确定性顺序）。"""
    pools: dict[str, list[dict]] = {}
    for slot in skel.slots:
        pool = [c for c in valid if c.get("type") == slot.type]
        if bag_themed:
            if slot.theme is None:
                pool = [c for c in pool if c.get("theme") is None]
            else:
                pool = [c for c in pool if c.get("theme") == slot.theme]
        seen: set[tuple[int, int]] = set()
        uniq: list[dict] = []
        for c in sorted(pool, key=lambda x: (x["s0"], x["e0"], x["cid"])):
            key = (c["s0"], c["e0"])
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        pools[slot.name] = uniq
    return pools


def _combo_ok(combo: list[dict], bag_themed: bool) -> str | None:
    """指派合法性：**不换方向**（无题元）/ 题元一致（有题元）+ 不重叠 + 不重复。"""
    if len({c["cid"] for c in combo}) != len(combo):
        return "distinct"
    for a, b in itertools.pairwise(combo):
        if a["s0"] < b["e0"] and b["s0"] < a["e0"]:
            return "span_overlap"
    if not bag_themed:
        for a, b in itertools.pairwise(combo):
            if b["s0"] <= a["s0"]:
                return "reorder_unjustified"
    return None


def compose(text: str, valid: list[dict], bag_themed: bool) -> tuple[dict | None, object | None, dict]:
    """穷举骨架表 → 指派搜索。产出**结构指令**（骨架 id + 槽位指派），不含文本。

    返回 `(struct, skeleton, info)`；失败时 `struct is None`，`info` 记拒绝原因与逐骨架明细。
    """
    records: list[dict] = []
    near_miss: list[str] = []      # 类型齐 + 门禁过，但指派搜不出来
    logic_blocked: list[str] = []  # 类型齐但被逻辑词门禁挡住
    missing_types: list[str] = []  # 类型库存缺口

    for skel in SKELETONS:
        pools = _pools(valid, skel, bag_themed)
        empty = [s.name for s in skel.slots if not pools[s.name]]
        if empty:
            rec = {"skel": skel.id, "why": "type", "empty_slots": empty,
                   "need": [s.type for s in skel.slots if s.name in empty]}
            records.append(rec)
            missing_types.extend(rec["need"])
            continue
        gate_missing = [w for w in skel.logic_words if w not in text]
        if gate_missing:
            records.append({"skel": skel.id, "why": "logic", "missing": gate_missing})
            logic_blocked.append(skel.id)
            continue

        names = [s.name for s in skel.slots]
        fails: dict[str, int] = {}
        cap = False
        for combo in itertools.product(*(pools[n] for n in names)):
            why = _combo_ok(list(combo), bag_themed)
            if why is None:
                struct = {
                    "skeleton": skel.id,
                    "assign": {s.name: c["cid"] for s, c in zip(skel.slots, combo)},
                    "refs": {c["cid"]: {
                        "cid": c["cid"], "card": c["card"],
                        "class_id": c["class_id"], "class_name": c["class_name"],
                        "s0": c["s0"], "e0": c["e0"], "surface": c["surface"],
                        "type": c["type"], "theme": c.get("theme"),
                    } for c in combo},
                    "evidence": [c["cid"] for c in combo],
                }
                return struct, skel, {"records": records}
            fails[why] = fails.get(why, 0) + 1
            if sum(fails.values()) > COMBO_CAP:
                cap = True
                break
        records.append({"skel": skel.id, "why": "no_assignment", "fails": fails,
                        "cap": cap})
        near_miss.append(skel.id)

    if near_miss:
        tally: dict[str, int] = {}
        for r in records:
            if r["why"] == "no_assignment":
                for k, v in r["fails"].items():
                    tally[k] = tally.get(k, 0) + v
        top = max(tally, key=lambda k: tally[k]) if tally else "distinct"
        if top in ("reorder_unjustified", "span_overlap"):
            reason, detail = "order_theme", {"fail_tally": tally}
        elif top == "distinct":
            reason, detail = "distinct", {"fail_tally": tally}
        else:
            reason, detail = "combo_cap", {"fail_tally": tally}
    elif logic_blocked:
        reason, detail = "logic_gate", {"blocked": logic_blocked}
    elif missing_types:
        reason, detail = "type_inventory", {"missing": sorted(set(missing_types))}
    else:
        reason, detail = "no_valid_candidate", {}
    return None, None, {"reason": reason, "detail": detail, "records": records}


# ---- 主流程 -----------------------------------------------------------------

def _reject(stage: str, reason: str, **extra) -> dict:
    out: dict = {"status": "reject", "stage": stage, "reason": reason}
    out.update(extra)
    assert "text" not in out, "拒答记录里不许出现 text"
    return out


def run_flow(text: str, candidates: list[dict],
             class_index: dict[str, tuple[str, ...]],
             *, struct_override: dict | None = None,
             poison: dict | None = None) -> dict:
    """跑完整五段流。`struct_override` 用于注入外部（或被篡改）的结构指令做对抗测试。"""
    # ① 候选区
    if not candidates:
        return _reject("propose", "no_candidates", n_candidates=0, n_valid=0)

    # ② 筛选区
    filtered = filter_candidates(text, candidates, class_index)
    valid = [c for c in filtered if c["valid"]]
    valid_index = {c["cid"]: c for c in valid}
    bag_themed = any(c.get("theme") is not None for c in valid)
    base = {"n_candidates": len(filtered), "n_valid": len(valid),
            "candidates": filtered}
    if not valid:
        return _reject("filter", "no_valid_candidate", **base)

    # ③ 重整区
    if struct_override is None:
        struct, skel, info = compose(text, valid, bag_themed)
        if struct is None:
            return _reject("compose", info["reason"], detail=info.get("detail"),
                           records=info.get("records"), **base)
    else:
        struct = struct_override
        skel = BY_ID.get(struct.get("skeleton", ""))
        if skel is None:
            return _reject("compose", "unknown_skeleton", **base)

    # ⑤-结构侧（渲染前先过 I1：schema + evidence 谓词）
    probs = structure_problems(text, struct, skel, valid_index, bag_themed)
    if probs:
        return _reject("deref", probs[0], check="structure", problems=probs,
                       structure=struct, skeleton=skel.id, **base)

    # ④ 渲染（确定性纯函数；毒渲染仅用于对抗测试）
    try:
        out_text = render_structure(struct, skel)
        if poison:
            if "drop" in poison:
                drop_slot = poison["drop"]
                parts: list[str] = []
                for kind, token in skel.lit_tokens():
                    if kind == "lit":
                        parts.append(token)
                    elif token != drop_slot:
                        parts.append(struct["refs"][struct["assign"][token]]["surface"])
                out_text = "".join(parts)
            if "inject" in poison:
                out_text = out_text + poison["inject"]
    except Exception as exc:                     # noqa: BLE001 —— fail-closed
        return _reject("render", "render_error", error=type(exc).__name__, **base)

    # ⑤-解引用侧（I3：文本每个内容单元必须映射到结构单元）
    probs, mapping = deref_verify(out_text, struct, skel, text, valid_index)
    if probs:
        return _reject("deref", probs[0], check="text", problems=probs,
                       structure=struct, skeleton=skel.id, **base)

    return {
        "status": "ok",
        "text": out_text,
        "skeleton": skel.id,
        "structure": struct,
        "mapping": mapping,
        "n_candidates": len(filtered),
        "n_valid": len(valid),
        "candidates": filtered,
    }
