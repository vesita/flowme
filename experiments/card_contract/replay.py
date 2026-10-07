"""既有记录的**重放**（V0 / V3 的口径实现）：读既有只读产物 ⇒ 用当前 `src` 层重新渲染 ⇒ 与存档逐字比对。

- **G 集**：`gen_dispatch/results_all.json` 的 `generate_samples`（官方骨架）；
- **A 集**：`card_flow/results_real.json` 的 `status=ok` 记录（card_flow 骨架 ⇒ 走迁移映射）；
- **raw 拒答**：`render.render()` / `dialogue._reject()` 的原始拒答记录（V2 的输入）。

`status` 取值：`match`（逐字相同）/ `diff`（**不一致**，V0 判失败）/
`deprecated`（骨架已废弃 ⇒ 按 PREREG §1.3 单列，不算不一致）/
`no_skeleton`（id 不在表里）/ `reject`（重放被拒，带 reason）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in ("src", "experiments/card_flow", "experiments/gen_dispatch"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

from dtseek.tasks import render as R  # noqa: E402

from migration import MIGRATION, old_to_new  # noqa: E402

GD = ROOT / "experiments" / "gen_dispatch"
CF = ROOT / "experiments" / "card_flow"


# ---- G 集：官方骨架的既有记录 -------------------------------------------------

def load_g() -> list[dict]:
    data = json.loads((GD / "results_all.json").read_text())
    out: list[dict] = []
    for tag in ("random", "enriched"):
        for j, g in enumerate(data[tag].get("generate_samples", [])):
            out.append({"rid": f"G[{tag}:{j}]", **g})
    return out


def _bag_for(g: dict) -> list[R.BagItem]:
    """袋 = 记录的 span 类 evidence；**pos 由槽位规格回填**（G 集的 `evidence` 不带 pos）。

    构造规定（PREREG §1 注）：`render()` 要求 `item.pos == slot.pos`，原跑也正是靠这条才出的
    `kind="text"` ⇒ **回填值 == 原跑值**。这是**推断级**证据，不是独立实测：
    `bag_pos` 只是原跑整袋的 pos 直方图（不按 ref 对齐，且整袋 ⊃ 已指派），只能做
    ⊆ 不等式佐证（见 `run_contract.pos_check`）。A 集不同：pos 来自记录自带的
    `type` 字段，**不是回填**。
    """
    sk = R.SKELETONS[g["instruction"]["skeleton_id"]]
    assignment = tuple(g["instruction"]["assignment"])
    slot_pos = {assignment[k]: sk.slots[k].pos for k in range(len(assignment))}
    bag = []
    for e in g["evidence"]:
        if e.get("source") != "span":
            continue
        bag.append(R.BagItem(ref=int(e["ref"]), text=e["text"],
                             span=(e["span"][0], e["span"][1]),
                             pos=slot_pos.get(int(e["ref"])), theta=None,
                             candidate_id=str(e.get("candidate_id", "")),
                             screened=True))
    return bag


def replay_g(g: dict) -> dict:
    if g["instruction"]["skeleton_id"] not in R.SKELETONS:
        return {"rid": g["rid"], "status": "no_skeleton",
                "stored": g["text"], "replay": None}
    bag = _bag_for(g)
    rec = R.render(R.Instruction(g["instruction"]["skeleton_id"],
                                 tuple(g["instruction"]["assignment"])),
                   bag, g["input"], plan_step_id=g.get("plan_step_id", ""))
    if rec.get("kind") != "text":
        return {"rid": g["rid"], "status": "reject", "stored": g["text"],
                "replay": None, "reason": rec.get("reason", "")}
    return {"rid": g["rid"], "status": "match" if rec["text"] == g["text"] else "diff",
            "stored": g["text"], "replay": rec["text"]}


# ---- A 集：card_flow 的既有记录（走迁移映射） ---------------------------------

def load_a() -> list[dict]:
    data = json.loads((CF / "results_real.json").read_text())
    out = []
    for i, r in enumerate(data["batch_real"]["records"]):
        if r["status"] == "ok":
            out.append({"rid": f"A[{i}]", **r})
    return out


def replay_a(r: dict) -> dict:
    old_id = r["skeleton"]
    kind, new_id, _p, reason = MIGRATION[old_id]
    if new_id is None:
        return {"rid": r["rid"], "status": "deprecated", "old": old_id,
                "stored": r["text"], "replay": None, "reason": reason}
    if new_id not in R.SKELETONS:
        return {"rid": r["rid"], "status": "no_skeleton", "old": old_id,
                "stored": r["text"], "replay": None}
    cands = r["candidates"]
    cid2idx = {c["cid"]: j for j, c in enumerate(cands)}
    bag = [R.BagItem(ref=j, text=c["surface"], span=(c["s0"], c["e0"]),
                     pos=c["type"], theta=c["theme"],
                     candidate_id=c["cid"], screened=bool(c["valid"]))
           for j, c in enumerate(cands)]
    assignment = tuple(cid2idx[m["ref"]] for m in r["mapping"])
    rec = R.render(R.Instruction(new_id, assignment), bag, r["input"],
                   plan_step_id="A:1")
    if rec.get("kind") != "text":
        return {"rid": r["rid"], "status": "reject", "old": old_id, "new": new_id,
                "stored": r["text"], "replay": None, "reason": rec.get("reason", "")}
    return {"rid": r["rid"], "status": "match" if rec["text"] == r["text"] else "diff",
            "old": old_id, "new": new_id, "stored": r["text"], "replay": rec["text"]}


# ---- 原始拒答记录（V2 输入） --------------------------------------------------

def _theta_reject() -> dict:
    text = "你说喜欢我。"
    bag = [R.BagItem(1, "你", (0, 1), "名", "施事", "c1", True),
           R.BagItem(3, "喜欢", (1, 3), "动", None, "c2", True),
           R.BagItem(4, "我", (3, 4), "名", "受事", "c3", True)]
    return R.render(R.Instruction("S01", (4, 3, 1)), bag, text, plan_step_id="step:0")


def _rule_a_reject() -> dict:
    doctored = R.Skeleton("BAD", "用户说[1]。", (R.SlotSpec("动"),))
    return R.render(R.Instruction("BAD", (2,)),
                    [R.BagItem(2, "说", (4, 5), "动", None, "c1", True)],
                    "用户说加载。", plan_step_id="step:0",
                    skeletons={**R.SKELETONS, "BAD": doctored})


def _word_face_reject() -> dict:
    text = "我们修车。"
    bag = [R.BagItem(0, "我们", (0, 2), "名", "施事", "c1", True),
           R.BagItem(1, "修车", (2, 4), "动", None, "c2", True),
           R.BagItem(3, "车", (3, 4), "名", "时", "c3", True)]
    return R.render(R.Instruction("S13", (1, 1, 0, 3, 1)), bag, text,
                    plan_step_id="step:1")


def _missing_slot_reject() -> dict:
    text = "我打你。"
    bag = [R.BagItem(0, "我", (0, 1), "名", "施事", "c1", True),
           R.BagItem(1, "打", (1, 2), "动", None, "c2", True),
           R.BagItem(2, "你", (2, 3), "名", "受事", "c3", True)]
    return R.render(R.Instruction("S01", (0, 1)), bag, text, plan_step_id="step:0")


def _type_mismatch_reject() -> dict:
    text = "我打你。"
    bag = [R.BagItem(0, "我", (0, 1), "名", "施事", "c1", True),
           R.BagItem(1, "打", (1, 2), "动", None, "c2", True),
           R.BagItem(2, "你", (2, 3), "名", "受事", "c3", True)]
    return R.render(R.Instruction("S01", (1, 0, 2)), bag, text, plan_step_id="step:0")


def _zero_info_reject() -> dict:
    text = "你说喜欢我。"
    bag = [R.BagItem(1, "你", (0, 1), "名", "施事", "c1", False),
           R.BagItem(3, "喜欢", (1, 3), "动", None, "c2", True),
           R.BagItem(4, "我", (3, 4), "名", "受事", "c3", True)]
    return R.render(R.Instruction("S01", (1, 3, 4)), bag, text, plan_step_id="step:2")


def _span_reject() -> dict:
    text = "我们修3台机器。"
    bag = [R.BagItem(0, "2", (3, 4), "名", None, "f0", True),
           R.BagItem(1, "修", (2, 3), "动", None, "f1", True)]
    return R.render(R.Instruction("R02", (0, 1)), bag, text, plan_step_id="step:1")


def _no_skeleton_reject() -> dict:
    text = "我打你。"
    bag = [R.BagItem(0, "我", (0, 1), "名", "施事", "c1", True),
           R.BagItem(1, "打", (1, 2), "动", None, "c2", True)]
    return R.render(R.Instruction("ZZ99", (0, 1)), bag, text, plan_step_id="step:0")


def _no_theta_reject() -> dict:
    text = "用户说加载。"
    bag = [R.BagItem(1, "用户", (0, 2), "名", None, "c1", True),
           R.BagItem(2, "说", (2, 3), "动", None, "c2", True),
           R.BagItem(3, "加载", (3, 5), "动", None, "c3", True)]
    return R.render(R.Instruction("S10", (1, 2, 3, 3)), bag, text, plan_step_id="step:0")


_RENDER_REJECTS = {
    "theta_conflict": _theta_reject,
    "rule_a": _rule_a_reject,
    "word_face": _word_face_reject,
    "missing_slot": _missing_slot_reject,
    "type_mismatch": _type_mismatch_reject,
    "zero_info": _zero_info_reject,
    "span_not_verbatim": _span_reject,
    "unknown_skeleton": _no_skeleton_reject,
    "no_theta_fail_closed": _no_theta_reject,
}


def raw_rejects() -> list[dict]:
    """`render.render()` 与 `dialogue._reject()` 的**原始**拒答记录（未归一）。"""
    from dtseek.tasks import dialogue as D

    out: list[dict] = []
    for name, fn in _RENDER_REJECTS.items():
        rec = fn()
        if rec.get("kind") != "reject":
            raise AssertionError(f"{name} 没有产生 reject：{rec.get('kind')}")
        out.append({"src": f"render:{name}", "rec": rec})
    for name, (type_, cards, plan, reason) in {
        "unknown_type": (None, [], [], "输入为空"),
        "unsupported_type": ("cloze", [], [], "本轮只支持 type=plain，声明了 'cloze'：不猜"),
        "empty_input": ("plain", [], [], "输入为空"),
        "card_unmounted": ("plain", ["sentiment"], ["计划：调 sentiment"],
                           "必需卡未挂载（引擎里没有）：['negation']：不许静默跳过"),
        "call_failed": ("plain", ["sentiment"], ["计划：调 sentiment"],
                        "调卡失败（调用抛错）：RuntimeError: boom"),
        "missing_required": ("plain", ["sentiment"], ["计划：调 sentiment"],
                             "缺必需信息（抽取位 人物=无）⇒ 拒答"),
        "repeat_call": ("plain", ["sentiment"], ["计划：重复调用"],
                        "计划要求重复调用同一张卡：不许"),
        "no_slice": ("plain", ["sentiment"], ["计划：调 sentiment"],
                     "没有任何可引用的切片：不编句子"),
        "non_terminal": ("plain", ["sentiment"], ["计划：停在非终止动作"],
                         "计划停在非终止动作：判不动"),
    }.items():
        rec = D._reject(type_, cards, plan, reason)
        out.append({"src": f"dialogue:{name}", "rec": rec})
    return out


def stored_rejects() -> list[dict]:
    """既有产物里存档的**dispatch 拒答片段**（`reject_examples`）。

    它们是 `{input, reason, terminal, rule}` 四元组，**不是完整记录**（缺 kind /
    evidence / plan_step_id）—— 这本身就是 V2 归一前的违例之一。`as_record` 是按
    「它在 `reject_examples` 名下 ⇒ R-CH 的通道是 reject」做的显式适配（构造规定）。
    """
    data = json.loads((GD / "results_all.json").read_text())
    out: list[dict] = []
    for tag in ("random", "enriched"):
        for j, g in enumerate(data[tag].get("reject_examples", [])):
            as_record = {"kind": "reject", "evidence": [], "plan_step_id": "none",
                         "reason": str(g.get("reason") or ""), "plan": [],
                         "channel": "reject", "terminal": g.get("terminal"),
                         "type": None, "input": g.get("input", "")}
            out.append({"src": f"stored:{tag}:{j}", "rec": g, "as_record": as_record})
    return out
