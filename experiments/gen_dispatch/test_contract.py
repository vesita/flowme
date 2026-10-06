#!/usr/bin/env python3
"""拒答契约与开关规则的**检查**（`pytest experiments/gen_dispatch/test_contract.py` 亦可）。

第 3 条是本任务要的「**两边一致**的检查」：
同一个 `contract_problems()` 同时判 `render.render()` 的原始 reject、
`dialogue._reject()` 的原始 reject、以及我方唯一出口 `rules.seal()` 归一后的记录 ——
前两边**必违例**（这是冲突的实测证据），归一后**必须 0 违例**。

    uv run python experiments/gen_dispatch/test_contract.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "card_flow"))
sys.path.insert(0, str(HERE))

from dtseek.tasks import dialogue as D  # noqa: E402
from dtseek.tasks import render as R  # noqa: E402

from pipeline import execute, fake_propose_anchors, verify_record  # noqa: E402
from rules import contract_problems, rule_audit, seal, switch_channel  # noqa: E402


# ---- 素材 --------------------------------------------------------------------

def _render_reject() -> dict:
    """官方层的一个真 reject（题元互换）。"""
    text = "我打你。"
    bag = [R.BagItem(0, "我", (0, 1), "名", "施事", "f0", True),
           R.BagItem(1, "打", (1, 2), "动", None, "f1", True),
           R.BagItem(2, "你", (2, 3), "名", "受事", "f2", True)]
    return R.render(R.Instruction("S01", (2, 1, 0)), bag, text, plan_step_id="step:0")


def _dialogue_reject() -> dict:
    return D._reject("plain", [], [], "输入为空")


# ---- 1–3 两边一致 -------------------------------------------------------------

def test_render_reject_raw_violates_contract() -> None:
    probs = contract_problems(_render_reject(), n_turn_steps=1)
    assert any(p.startswith("C2") for p in probs), probs
    assert any("拒答记录带 text" in p for p in probs), probs


def test_dialogue_reject_raw_violates_contract() -> None:
    probs = contract_problems(_dialogue_reject())
    assert any(p.startswith("C2") for p in probs), probs


def test_both_sides_consistent_after_single_exit() -> None:
    raws = [_render_reject(), _dialogue_reject()]
    sealed = [seal(kind="reject", channel="reject", text=r.get("text"),
                   reason=r.get("reason", ""), plan_step_id="step:0") for r in raws]
    for r, s in zip(raws, sealed):
        assert contract_problems(r, n_turn_steps=1), "原始记录必须违例（冲突证据）"
        assert contract_problems(s, n_turn_steps=1) == [], contract_problems(s)
        assert "text" not in s and s["reason"].strip()
    # 反例：归一后又把 text 塞回去 ⇒ 同一个谓词立刻报
    bad = dict(sealed[0], text="（拒答）…")
    assert any("C2" in p for p in contract_problems(bad))


# ---- 契约反例 -----------------------------------------------------------------

def test_reject_without_reason_is_rejected() -> None:
    rec = {"kind": "reject", "evidence": [], "plan_step_id": "none", "plan": []}
    probs = contract_problems(rec)
    assert any("reason" in p for p in probs), probs


def test_text_without_evidence_is_rejected() -> None:
    rec = {"kind": "text", "text": "随便说说", "evidence": [],
           "plan_step_id": "none", "channel": "pointer"}
    probs = contract_problems(rec)
    assert any("evidence" in p for p in probs), probs


def test_missing_core_key_is_rejected() -> None:
    probs = contract_problems({"kind": "reject", "reason": "x", "evidence": []})
    assert any("plan_step_id" in p for p in probs), probs


def test_plan_step_id_must_align_with_plan() -> None:
    good = {"kind": "reject", "reason": "x", "evidence": [],
            "plan_step_id": "step:0", "plan": ["..."]}
    assert contract_problems(good, n_turn_steps=1) == []
    bad = dict(good, plan_step_id="step:7")
    assert any(p.startswith("C4") for p in contract_problems(bad, n_turn_steps=1))


# ---- 端到端（假卡） ------------------------------------------------------------

def test_generate_record_passes_contract_and_both_checks() -> None:
    rec, dbg = execute("我不喜欢这个方案，太慢了。", propose=fake_propose_anchors,
                       attached=set("person pronoun relation sentiment negation idiom"
                                    .split()), need_g=True)
    if rec["channel"] != "generate":          # 词典卡的产出由本测试自己兜底判
        return
    assert contract_problems(rec, n_turn_steps=dbg["n_turn_steps"]) == []
    assert verify_record(rec, dbg) == []
    assert rec["evidence"] and rec["instruction"] and rec["ref_map"]
    assert rec["plan_step_id"] == f"step:{dbg['n_turn_steps'] - 1}"


def test_pointer_evidence_is_half_open_and_verbatim() -> None:
    src = "我不喜欢这个方案，太慢了。"
    rec, dbg = execute(src, propose=fake_propose_anchors,
                       attached=set("person pronoun relation sentiment negation idiom"
                                    .split()), need_g=False)
    if rec["channel"] != "pointer":
        return
    assert verify_record(rec, dbg) == []
    for ev in rec["evidence"]:
        s, e = ev["span"]
        assert 0 <= s < e <= len(src)          # 统一记录里一律半开区间
        assert src[s:e] and src[s:e] in rec["text"]


def test_every_fake_record_satisfies_contract() -> None:
    bad = []
    for s in ["你说的对，我们明天试试。", "我不喜欢这个方案，太慢了。",
              "他不是坏人，你别生气。", "我们跑一次看看效果。",
              "这个太难了，我做不了。", "张三喜欢李四。", "", "   "]:
        rec, dbg = execute(s, propose=fake_propose_anchors,
                           attached=set("person pronoun relation sentiment negation idiom"
                                        .split()), need_g=True)
        p = contract_problems(rec, n_turn_steps=dbg.get("n_turn_steps"))
        if p:
            bad.append((s, p))
        if rec["kind"] == "text" and verify_record(rec, dbg):
            bad.append((s, "verify"))
    assert bad == [], bad


# ---- 开关规则表 ---------------------------------------------------------------

def test_switch_rule_table_is_exhaustive() -> None:
    expected = {
        ("reject", True, True): "reject",
        ("reject", True, False): "reject",
        ("reject", False, True): "reject",
        ("reject", False, False): "reject",
        ("generate", True, True): "generate",
        ("generate", True, False): "generate",
        ("generate", False, True): "pointer",
        ("generate", False, False): "reject",
        ("pointer", True, True): "pointer",
        ("pointer", False, True): "pointer",
        ("pointer", True, False): "reject",
        ("pointer", False, False): "reject",
    }
    for (t, g, p), want in expected.items():
        got, row = switch_channel(t, g, p)
        assert got == want, (t, g, p, got, want, row)
        assert row.startswith("R-CH"), row
    # 判不动 ⇒ fail-closed
    assert switch_channel("garbage", True, True)[0] == "reject"


def test_rule_audit_finds_no_learned_quantity() -> None:
    audit = rule_audit()
    assert audit["ok"], audit["hits"]
    assert "weight" not in audit["source"].lower()


def main() -> int:
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except AssertionError as exc:
                fails += 1
                print(f"[FAIL] {name}: {exc}")
    print(f"\n{'全部通过' if not fails else f'{fails} 条失败'}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
