#!/usr/bin/env python3
"""P7 结构出口自检（正例 + **故意注入的反例**）—— 没有反例的检查不算检查。

对 `mode_render.make_record` / `check_mode_structure` / `check_mode_deref`：
  · 正例：真实行上的合法指令 ⇒ `kind="text"` 且解引用过；
  · 反例：越界 span / 实义词当触发词 / 表外触发词 / 表外模式 id /
    `ref_map` 不铺满 / 内容 span 非逐字 —— 逐条必须被拒（fail-closed）。

用法：uv run python experiments/sentence_mode/selfcheck.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import labels as L  # noqa: E402
import mode_render as MR  # noqa: E402

ok = fail = 0


def check(name: str, cond: bool, detail: str = "") -> None:
    global ok, fail
    if cond:
        ok += 1
        print(f"  PASS  {name} {detail}")
    else:
        fail += 1
        print(f"  FAIL  {name} {detail}")


def main() -> int:
    s = "你吃了吗"
    sid = MR.sid_of(L.MODE_ID["是非疑问"])
    print("[正例]")
    rec = MR.make_record(L.MODE_ID["是非疑问"], (3, 4), s, plan_step_id="t:1")
    check("合法指令 kind=text", rec["kind"] == "text", f"text={rec['text']!r}")
    check("deref 通过", not MR.check_mode_deref(rec, s))
    check("evidence 逐字", rec["evidence"][0]["text"] == s[3:4])

    rec0 = MR.make_record(L.MODE_ID["陈述"], None, "今天天气不错", plan_step_id="t:2")
    check("∅ 触发词合法（陈述）", rec0["kind"] == "text" and rec0["text"] == "")

    print("[反例 · 必须被拒]")
    r = MR.make_record(L.MODE_ID["是非疑问"], (3, 9), s)
    check("span 越界 ⇒ reject", r["kind"] == "reject", r["reason"][:60])

    r = MR.make_record(L.MODE_ID["是非疑问"], (0, 2), s)   # 「你吃」是实义词
    check("实义词当触发词 ⇒ reject", r["kind"] == "reject", r["reason"][:70])

    r = MR.make_record(L.MODE_ID["是非疑问"], (0, 1), "嗯嗯嗯")
    check("表外触发词 ⇒ reject", r["kind"] == "reject", r["reason"][:70])

    r = MR.make_record(99, None, s)
    check("表外模式 id ⇒ reject", r["kind"] == "reject", r["reason"][:50])

    # ref_map 不铺满：手工把 out 区间缩短 1 位
    rec = MR.make_record(L.MODE_ID["是非疑问"], (3, 4), s)
    bad = json.loads(json.dumps(rec))
    bad["ref_map"][0]["out"] = [0, 0]
    check("ref_map 不铺满 ⇒ deref 拒", bool(MR.check_mode_deref(bad, s)))

    # 内容 span 非逐字（映射表里的 span 指到别处）
    bad = json.loads(json.dumps(rec))
    bad["ref_map"][0]["span"] = [0, 1]
    check("span 非逐字 ⇒ deref 拒", bool(MR.check_mode_deref(bad, s)))

    # kind=reject 的记录不得进解引用
    check("reject 记录不得 deref", bool(MR.check_mode_deref(
        {"kind": "reject", "text": "x", "ref_map": []}, s)))

    print("[反例 · 遮蔽断言]")
    try:
        L.assert_masked("你吃了吗？", "unit")
        check("未遮蔽输入应抛断言", False)
    except AssertionError as e:
        check("未遮蔽输入应抛断言", True, str(e)[:50])
    try:
        L.assert_masked(L.exit_mask("你吃了吗？"), "unit")
        check("遮蔽后通过", True)
    except AssertionError as e:
        check("遮蔽后通过", False, str(e)[:80])

    print(f"\n[selfcheck] PASS={ok} FAIL={fail}")
    return 1 if fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
