#!/usr/bin/env python3
"""**只读 import** 官方 `src/dtseek/tasks/render.py`（本实验开始时该文件尚不存在，
01:18 由另一代理建成；我的最小版已在 `experiments/card_flow/render.py` 里跑完判据）。

本脚本不改 src，只回答四件事：

  X1 官方骨架表（16 条，全含动槽）在**真卡候选袋**上的覆盖率；
  X2 我的 36 条真实输出，在官方表里找不找得到类型签名相同的骨架；
  X3 题元方向对抗在**官方层**复现（我打你 / 你打我 不许互换）；
  X4 逻辑词（因为 / 要）在官方词面守卫下的拒答。

用到的官方函数：`BagItem / Instruction / Declared / SKELETONS / render /
check_structure / check_deref / word_face_problems / rule_a_problems /
slot_schema_problems / validate_skeleton_table`。
"""
from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from dtseek.tasks import render as R  # noqa: E402  ← 官方渲染层（只读）

# 我的槽位类型 → 官方 POS_TYPES（否/数 是我加的扩展，官方域里没有）
MY2OFFICIAL = {"名": "名", "动": "动", "形": "形"}
CARD_QUALITY_HINT = HERE / "results_real.json"


def official_bag(cands: list[dict]) -> list[R.BagItem]:
    """真卡候选 → 官方袋块（类型不在官方域的块直接不进袋，等价于官方层拒收）。"""
    bag = []
    for i, c in enumerate(cands):
        if not c.get("valid"):
            continue
        pos = MY2OFFICIAL.get(c.get("type"))
        if pos is None:
            continue
        bag.append(R.BagItem(ref=i, text=c["surface"], span=(c["s0"], c["e0"]),
                             pos=pos, theta=c.get("theme"),
                             candidate_id=c["cid"], screened=True))
    return bag


def x1_coverage(records: list[dict]) -> dict:
    """官方骨架表 × 真卡候选袋：能不能渲染出文本。"""
    covered = 0
    miss_type = 0
    per_skel: dict[str, int] = {}
    no_bag = 0
    for rec in records:
        bag = official_bag(rec.get("candidates", []))
        if not bag:
            no_bag += 1
            continue
        hit = None
        for sk in R._SKELETON_LIST:
            pools = [[it for it in bag if it.pos == slot.pos] for slot in sk.slots]
            if any(not p for p in pools):
                continue
            for combo in itertools.product(*pools):
                if len({it.ref for it in combo}) != len(combo):
                    continue
                instr = R.Instruction(sk.sid, tuple(it.ref for it in combo))
                out = R.render(instr, bag, rec["input"])
                if out["kind"] == "text":
                    hit = sk.sid
                    break
            if hit:
                break
        if hit:
            covered += 1
            per_skel[hit] = per_skel.get(hit, 0) + 1
        else:
            # 只统计"类型库存填不满"的情形（动槽缺货是已知瓶颈）
            miss_type += 1
    n = len(records)
    return {"n": n, "official_covered": covered,
            "coverage": round(covered / n, 4) if n else 0.0,
            "uncovered": miss_type, "no_bag_at_all": no_bag,
            "per_skeleton": per_skel,
            "official_skeletons": len(R._SKELETON_LIST),
            "all_need_verb": all(any(s.pos == "动" for s in sk.slots)
                                 for sk in R._SKELETON_LIST)}


def x2_signature_match(records: list[dict]) -> dict:
    """我的输出骨架类型签名，在官方表里找不找得到同签名骨架。"""
    same_id_diff_sig = 0
    no_match = 0
    matched = 0
    examples = []
    off_sigs = {}
    for sk in R._SKELETON_LIST:
        off_sigs[sk.sid] = tuple(s.pos for s in sk.slots)
    for rec in records:
        if rec.get("status") != "ok":
            continue
        st = rec["structure"]
        sig = tuple(st["refs"][st["assign"][s.name]]["type"]
                    for s in _my_slots(rec["skeleton"]))
        cands = [sid for sid, s in off_sigs.items() if s == sig]
        if cands:
            matched += 1
        else:
            no_match += 1
            if len(examples) < 5:
                examples.append({"my": rec["skeleton"], "sig": sig,
                                 "text": rec["text"]})
        if rec["skeleton"] in off_sigs and off_sigs[rec["skeleton"]] != sig:
            same_id_diff_sig += 1
    return {"n_ok": matched + no_match, "signature_matched": matched,
            "signature_unmatched": no_match,
            "same_id_but_different_signature": same_id_diff_sig,
            "examples": examples}


def _my_slots(sid: str):
    from skeletons import BY_ID
    return BY_ID[sid].slots


def x3_direction() -> dict:
    """官方层的题元方向对抗：我打你 / 你打我，互换必拒。"""
    cases = []
    for text, a, b in (("我打你。", "我", "你"), ("你打我。", "你", "我")):
        bag = [R.BagItem(0, a, (text.index(a), text.index(a) + 1), "名", "施事", "f0", True),
               R.BagItem(1, "打", (text.index("打"), text.index("打") + 1), "动", None, "f1", True),
               R.BagItem(2, b, (text.index(b), text.index(b) + 1), "名", "受事", "f2", True)]
        ok = R.render(R.Instruction("S01", (0, 1, 2)), bag, text)
        bad = R.render(R.Instruction("S01", (2, 1, 0)), bag, text)
        cases.append({
            "input": text,
            "correct": {"kind": ok["kind"], "text": ok.get("text", ""),
                        "reason": ok.get("reason", "")},
            "swapped": {"kind": bad["kind"], "text": bad.get("text", ""),
                        "reason": bad.get("reason", "")[:120]},
        })
    return {"cases": cases,
            "correct_ok": all(c["correct"]["kind"] == "text" for c in cases),
            "swap_rejected": all(c["swapped"]["kind"] == "reject" for c in cases),
            "swap_has_text_field": any("text" in c["swapped"] for c in cases)}


def x4_logic_guard() -> dict:
    """官方词面守卫：逻辑词（因为/要）无据必拒、有据不误拒。"""
    sk = R.SKELETONS["S13"]                      # 因为[…]，[…]要[…]。
    no_ev = R.word_face_problems("我难过，你高兴。", sk)
    with_ev = R.word_face_problems("因为下雨，我们明天要修。", sk)
    return {"skeleton": sk.sid, "pattern": sk.pattern,
            "no_evidence": no_ev, "with_evidence": with_ev,
            "rejects_without": bool(no_ev), "accepts_with": not with_ev}


def main() -> int:
    data = json.loads(CARD_QUALITY_HINT.read_text(encoding="utf-8"))
    records = data["batch_real"]["records"]
    out = {
        "official_module": "src/dtseek/tasks/render.py",
        "official_table_limit": R.SKELETON_TABLE_LIMIT,
        "official_validate": R.validate_skeleton_table(),
        "X1_official_coverage_on_real_cards": x1_coverage(records),
        "X2_skeleton_signature_crosswalk": x2_signature_match(records),
        "X3_direction_on_official_layer": x3_direction(),
        "X4_logic_word_guard_on_official_layer": x4_logic_guard(),
    }
    path = HERE / "results_crosscheck_official.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"[写出] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
