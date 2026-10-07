"""官方层交叉核对（**只读 import** `src/dtseek/tasks/render.py` 与 `dialogue.py`）。

两件事：
1. **G1 统一契约**：官方 `render()` 的 reject 与 `dialogue._reject()` 原始都带占位
   `text`（`（拒答）…`）⇒ `contract.normalize()` 归一后再判 G1；**非占位 text 不许被归一掉**。
2. **Tier-2 渲染的独立实现交叉核对**：把 Tier-2 结构指令翻译成官方 `Skeleton/BagItem/Instruction`
   （类型限定官方域 `{名,动,形}`，`direction_safe=True`），跑官方 `render()`：
   官方两道检查（`check_structure` + `check_deref`）必须通过、**文本必须与本实现逐字相同**。
   类型超出官方域（`否/数`）的记录 ⇒ **不翻译、如实计入 skip**（不改官方表）。
"""
from __future__ import annotations

from contract import is_reject, normalize

from t2core import _names


def g1_contract_demo() -> dict:
    """C3/G1 的「官方层带占位 text ⇒ 归一」演示（两条原始记录 + 归一结果）。"""
    from dtseek.tasks import dialogue
    from dtseek.tasks import render as o

    bad = o.Instruction("NOPE", (1,))
    bag = [o.BagItem(ref=1, text="我", span=(0, 1), pos="名", theta=None,
                     candidate_id="c1", screened=True)]
    official = o.render(bad, bag, "我打你。", plan_step_id="demo")
    dlg = dialogue._reject("plain", ["pronoun"], ["call:pronoun"], "demo 原因")

    ours = {"status": "reject", "stage": "tier2", "reason": "demo"}
    return {
        "raw": {
            "official_render": {k: official[k] for k in ("kind", "text", "reason")},
            "dialogue_reject": {k: dlg[k] for k in ("kind", "text", "reason")},
            "ours": dict(ours),
        },
        "normalized_has_text": {
            "official_render": "text" in normalize(official),
            "dialogue_reject": "text" in normalize(dlg),
            "ours": "text" in normalize(ours),
        },
        "placeholder_prefix_ok": {
            "official_render": official["text"].startswith("（拒答）"),
            "dialogue_reject": dlg["text"].startswith("（拒答）"),
        },
        "is_reject": {"official_render": is_reject(official),
                      "dialogue_reject": is_reject(dlg)},
    }


def crosscheck(records: list[dict]) -> dict:
    """Tier-2 ok 记录 → 官方层重渲染。报命中数、跳过数（类型超域）与失败原因。"""
    from dtseek.tasks import render as o

    out = {"n_tier2_ok": 0, "n_translatable": 0, "n_match": 0, "n_skipped_type": 0,
           "skipped_types": {}, "problems": [], "n_official_reject_with_text_raw": 0,
           "n_official_reject_with_text_norm": 0}
    for rec in records:
        if rec.get("status") != "ok" or rec.get("tier") != 2:
            continue
        out["n_tier2_ok"] += 1
        struct = rec["structure"]
        names = _names(struct)
        types = [struct["slots"][k] for k in names]
        bad_types = [t for t in types if t not in o.POS_TYPES]
        if bad_types:
            out["n_skipped_type"] += 1
            for t in bad_types:
                out["skipped_types"][t] = out["skipped_types"].get(t, 0) + 1
            continue
        out["n_translatable"] += 1

        parts: list[str] = []
        slots = []
        for i, name in enumerate(names, 1):
            parts.append(f"[{i}]")
            f = struct["funcs"][i - 1]
            if f["text"]:
                parts.append(f["text"])
            slots.append(o.SlotSpec(pos=struct["slots"][name], theta=None))
        sk = o.Skeleton("T2X", "".join(parts), tuple(slots), direction_safe=True)

        bag, assign = [], []
        for i, name in enumerate(names, 1):
            cid = struct["assign"][name]
            ref = struct["refs"][cid]
            bag.append(o.BagItem(ref=i, text=ref["surface"],
                                 span=(ref["s0"], ref["e0"]), pos=ref["type"],
                                 theta=None, candidate_id=cid, screened=True))
            assign.append(i)

        orec = o.render(o.Instruction("T2X", tuple(assign)), bag, rec["input"],
                        plan_step_id=f"tier2:{rec.get('input', '')[:8]}",
                        skeletons={"T2X": sk})
        norm = normalize(orec)
        if is_reject(norm):
            out["n_official_reject_with_text_raw"] += int("text" in orec)
            out["n_official_reject_with_text_norm"] += int("text" in norm)
            out["problems"].append({"input": rec.get("input"),
                                    "reason": norm.get("reason")})
            continue
        if norm.get("text") != rec["text"]:
            out["problems"].append({"input": rec.get("input"),
                                    "why": "text_mismatch",
                                    "official": norm.get("text"),
                                    "ours": rec["text"]})
            continue
        ours = [(u["text"], tuple(u["span"])) for u in rec["mapping"]
                if u["cls"] == "content"]
        theirs = [(u["text"], tuple(u["span"])) for u in norm["ref_map"]
                  if u["cls"] == "content"]
        if ours != theirs:
            out["problems"].append({"input": rec.get("input"),
                                    "why": "ref_map_mismatch", "ours": ours,
                                    "official": theirs})
            continue
        out["n_match"] += 1
    return out
