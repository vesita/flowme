#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P12b 结构化指令 → 契约记录（PREREG §2.1；**不做 E5 解释层**，只产记录 + 机械谓词）。

- 判定 = 结构指令 `Instruction(skeleton_id, assignment)`，`assignment = (ref_m1, ref_verb, ref_m2)`：
  SAME → `S01`（内置骨架原样复用）；DIFF → `N01`（本单元新增，pattern `[1]不[2][3]。`）。
- **N01 口径判断（如实报出）**：PREREG §2.1 写的 pattern `[1]不[3][2]。` 与其**渲染例**
  `张伟不是张伟。`、槽位表（名施事/动/名受事）、全局 `assignment=(m1, verb, m2)` 三者不自洽
  （按字面渲出 `张伟不张伟是。`）；表内其余骨架的 pattern 槽位引用均单调递增 ⇒ 判为下标笔误，
  实现取 `[1]不[2][3]。`（渲染例与槽位表保持逐字一致）。判据不受影响（N01 只承载 DIFF 标签）。
- N01 进表前跑与 `validate_skeleton_table` 同两条谓词：`rule_a_problems` + `slot_schema_problems`；
  `不` 是逻辑词 ⇒ 用 `Declared(logic=("不",))` 显式声明（意图卡，跑前写死）。
- 每条记录经 `render.render()`（check_structure + check_deref 都过才出 text）→ `to_contract()`
  → `contract_problems(rec, input_text=...)` 空 = 通过；拒答记录 `to_dict()` 无 `text` 键。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

from dtseek.tasks import render as R  # noqa: E402
from dtseek.tasks.card_contract import contract_problems, to_contract  # noqa: E402

import train_arm as T  # noqa: E402

RES = HERE / "results"
DECL = R.Declared(logic=("不",))
N01 = R.Skeleton("N01", "[1]不[2][3]。",
                 (R.SlotSpec("名", "施事"), R.SlotSpec("动"), R.SlotSpec("名", "受事")))
TABLE = dict(R.SKELETONS)
TABLE["N01"] = N01

ARMS = {"x": "test.jsonl", "x-ir": "xir_test.jsonl", "x-cross": "xcross_test.jsonl"}


def load(p):
    return [json.loads(x) for x in (HERE / "data" / p).read_text(encoding="utf-8").splitlines()
            if x.strip()]


def bag_of(it):
    m1, m2, vb = tuple(it["m1_span"]), tuple(it["m2_span"]), tuple(it["verb_span"])
    return [
        R.BagItem(ref=0, text=it["s1"], span=m1, pos="名", theta="施事",
                  candidate_id=f"{it['id']}#c0", screened=True),
        R.BagItem(ref=1, text=it["text"][vb[0]:vb[1]], span=vb, pos="动", theta=None,
                  candidate_id=f"{it['id']}#c1", screened=True),
        R.BagItem(ref=2, text=it["s2"], span=m2, pos="名", theta="受事",
                  candidate_id=f"{it['id']}#c2", screened=True),
    ], m1, m2, vb


def build(it, label):
    bag, m1, m2, vb = bag_of(it)
    instr = R.Instruction("S01" if label == "SAME" else "N01", (0, 1, 2))
    rec = R.render(instr, bag, it["text"], plan_step_id="step:0", declared=DECL,
                   skeletons=TABLE)
    card = to_contract(rec)
    probs = contract_problems(card, input_text=it["text"])
    return rec, card, probs, (m1, m2, vb)


def head_predict(arm, seed=42):
    feats = {}
    for sp, fn in ((0, T.ARM_FILES[arm][0]), (1, T.ARM_FILES[arm][1])):
        z = np.load(HERE / "features" / f"{arm}_{['train', 'test'][sp]}.npz")
        feats[(arm, sp)] = (z["X"], z["y"])
    Xtr, ytr = feats[(arm, 0)]
    Xte, _ = feats[(arm, 1)]
    head, meta = T.train_head(Xtr, ytr, seed, T.STEPS, randlabel=False)
    _, A, _, _ = T.standardize(Xtr, Xte)
    import torch
    with torch.no_grad():
        pred = head(torch.as_tensor(A, dtype=torch.float32, device=T.DEV)).argmax(1)
    return pred.cpu().numpy(), meta


def main() -> int:
    out: dict = {"arms": {}, "gates": {}}

    # ---- N01 进表门禁（同两条谓词）+ 官方表自检 ----
    g1 = R.rule_a_problems(N01)
    g2 = R.slot_schema_problems(N01)
    g3 = R.validate_skeleton_table()
    out["gates"] = {"N01_rule_a": g1, "N01_slot_schema": g2,
                    "official_table_validate": g3, "table_size": len(TABLE),
                    "declared": {"logic": list(DECL.logic)}}
    print(f"[gate] N01 rule_a_problems={g1} | slot_schema_problems={g2}")
    print(f"[gate] 官方骨架表 validate_skeleton_table={g3}；本单元表规模={len(TABLE)}")
    assert not g1 and not g2 and not g3, "N01 / 官方表门禁未过"

    for arm, fn in ARMS.items():
        rows = load(fn)
        pred, meta = head_predict(arm, seed=42)
        labels = ["DIFF" if p == 1 else "SAME" for p in pred]
        ok = 0
        examples = []
        bad = []
        kinds = {"text": 0, "reject": 0}
        for it, lab in zip(rows, labels):
            rec, card, probs, _ = build(it, lab)
            kinds[card.kind] = kinds.get(card.kind, 0) + 1
            if not probs:
                ok += 1
            elif len(bad) < 3:
                bad.append({"id": it["id"], "probs": probs[:2]})
            if len(examples) < 2:
                examples.append({
                    "id": it["id"], "gold": it["label"], "pred": lab,
                    "spans": {"m1": it["m1_span"], "m2": it["m2_span"],
                              "verb": it["verb_span"]},
                    "instruction": card.instruction,
                    "rendered_text": card.text,
                    "contract_ok": not probs,
                    "to_dict_keys": sorted(card.to_dict().keys()),
                    "evidence_n": len(card.evidence), "ref_map_n": len(card.ref_map)})
        out["arms"][arm] = {"n": len(rows), "parse_ok": ok, "parse_ok_rate": ok / len(rows),
                            "kinds": kinds, "head_params": meta["head_params"],
                            "head_seed": 42, "examples": examples, "first_bad": bad}
        print(f"[cards] {arm:8s} n={len(rows)} contract 通过={ok} "
              f"({ok / len(rows):.4%}) kinds={kinds}")

    # ---- 拒答演示：不声明「不」 ⇒ 词面守卫拒 ⇒ text 缺省 ----
    it = load(ARMS["x"])[0]
    bag, *_ = bag_of(it)
    bad_instr = R.Instruction("N01", (0, 1, 2))
    rec = R.render(bad_instr, bag, it["text"], plan_step_id="step:0",
                   declared=R.DECLARED_NONE, skeletons=TABLE)
    card = to_contract(rec)
    probs = contract_problems(card, input_text=it["text"])
    d = card.to_dict()
    print(f"[reject] 不声明「不」 ⇒ kind={card.kind} 无 text 键={'text' not in d} "
          f"reason 非空={bool(card.reason)} evidence 空={not card.evidence} "
          f"contract 问题={len(probs)}")
    out["reject_demo"] = {"kind": card.kind, "no_text_key": "text" not in d,
                          "reason_nonempty": bool(card.reason),
                          "evidence_empty": not card.evidence,
                          "contract_problems": probs,
                          "reason_head": card.reason[:80]}
    # ---- 拒答演示 2：袋 span 越界 ⇒ item_problems 抓 ----
    it2 = load(ARMS["x"])[1]
    bag2, *_ = bag_of(it2)
    bag2[0] = R.BagItem(ref=0, text=it2["s1"], span=(0, 1), pos="名", theta="施事",
                        candidate_id="x", screened=True)
    rec2 = R.render(R.Instruction("S01", (0, 1, 2)), bag2, it2["text"],
                    plan_step_id="step:0", declared=DECL, skeletons=TABLE)
    card2 = to_contract(rec2)
    out["reject_demo_2"] = {"kind": card2.kind, "no_text_key": "text" not in card2.to_dict(),
                            "reason_head": card2.reason[:80]}
    print(f"[reject] 篡改 span ⇒ kind={card2.kind} 无 text 键={'text' not in card2.to_dict()} "
          f"reason={card2.reason[:60]}")

    RES.mkdir(exist_ok=True)
    (RES / "cards.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                    encoding="utf-8")
    all_ok = all(v["parse_ok_rate"] == 1.0 for v in out["arms"].values())
    print(f"OK cards（contract 全通过={all_ok}）→ {RES / 'cards.json'}")
    return 0 if all_ok and out["reject_demo"]["kind"] == "reject" else 2


if __name__ == "__main__":
    raise SystemExit(main())
