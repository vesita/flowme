"""§7.7 重整层 + §7.9 解引用层的测试（dev-notes/16）——**每条都有反例，证明非空**。

| # | 判据 | 测试 |
|---|---|---|
| T1 | 题元方向：`你说喜欢我` → `我喜欢你` 字符串全过、命题相反 ⇒ 拒 | `test_theta_direction_opposite_rejected` |
| T2 | 规则 A：骨架字面掺实义词 ⇒ 拒（词典谓词） | `test_rule_a_skeleton_literal_with_content_word_rejected` |
| T3 | 规则 B：槽位类型/题元不匹配 ⇒ 拒 | `test_rule_b_type_mismatch_rejected` / `test_rule_b_slot_without_theta_rejected` |
| T4 | 逻辑词：输入无据且未声明却出现 `因为` ⇒ 拒（编的因果） | `test_logic_word_undeclared_and_unbased_rejected` |
| T5 | 纯函数：同输入两次渲染逐字相同 | `test_render_is_pure_function` |
| T6 | 映射覆盖：渲染时多吐一个词 ⇒ 必须被抓（I3 靶子） | `test_deref_catches_extra_word` |
| T7 | 反例：把映射表改成错的键 ⇒ 断言必须失败（非橡皮图章） | `test_counterexample_wrong_map_key_fails` |
| T8 | 规则 0：指派独立于袋序 | `test_rule0_assignment_independent_of_bag_order` |
| T9 | evidence 两类 + I3 回溯链 | `test_evidence_two_kinds_and_ref_chain` |
| T10 | 零信息新增：内容词必须回溯到被筛为有效的候选 | `test_zero_info_content_rejected` |
| T11 | 无角色标签 fail-closed 到「只复述不换方向」子集 | `test_unlabeled_theta_fails_closed_to_recite_subset` |
| T12 | 骨架表封闭可枚举（≤40）且全表过规则 A/schema | `test_skeleton_table_bounded_and_clean` |

T7 的「断言必须失败」用**直接调谓词**证明：错键的记录喂给 `check_deref` 必须产出非空问题
（同时同一记录改回正确键必须产出空 —— 两头都断言，空测试会当场失败）。
"""
from __future__ import annotations

from dtseek.tasks import render as R

# ---- 场景 1：dev-notes/16 §7.7 的例子 ---------------------------------------

IN1 = "这次项目复盘，用户说加载太慢，我们下周要修。"


def bag1() -> list[R.BagItem]:
    """§7.7 的袋：`1用户(名·施事) 2加载 3太慢 4我们 5下周 6修 7说` —— 这里补全 span/题元。"""
    return [
        R.BagItem(1, "用户", (7, 9), "名", "施事", "cand-1", True),
        R.BagItem(2, "说", (9, 10), "动", None, "cand-2", True),
        R.BagItem(3, "加载", (10, 12), "动", None, "cand-3", True),
        R.BagItem(4, "太慢", (12, 14), "形", None, "cand-4", True),
        R.BagItem(5, "我们", (15, 17), "名", "施事", "cand-5", True),
        R.BagItem(6, "下周", (17, 19), "名", "时", "cand-6", True),
        R.BagItem(7, "修", (20, 21), "动", None, "cand-7", True),
    ]


IN11 = R.Instruction("S11", (1, 2, 3, 4, 5, 6, 7))


# ---- 场景 2：题元方向的靶子 --------------------------------------------------

IN2 = "你说喜欢我"


def bag2() -> list[R.BagItem]:
    return [
        R.BagItem(1, "你", (0, 1), "名", "施事", "cand-a", True),
        R.BagItem(2, "说", (1, 2), "动", None, "cand-b", True),
        R.BagItem(3, "喜欢", (2, 4), "动", None, "cand-c", True),
        R.BagItem(4, "我", (4, 5), "名", "受事", "cand-d", True),
    ]


def _ok(rec: dict) -> bool:
    return rec["kind"] == "text"


# ---- T1 题元方向 -------------------------------------------------------------

def test_theta_direction_opposite_rejected() -> None:
    """`你说喜欢我` → `我喜欢你`：字符串全过、命题相反 ⇒ 题元谓词拒答。"""
    bag = bag2()
    reversed_instr = R.Instruction("S01", (4, 3, 1))  # 我 / 喜欢 / 你
    bad = R.render(reversed_instr, bag, IN2, plan_step_id="t1")
    assert bad["kind"] == "reject"
    assert "题元方向冲突" in bad["reason"], bad["reason"]
    assert bad["evidence"] == [] and bad["ref_map"] == []
    # 字符串层面全过 —— 这正是「P2 对题元全盲」的靶子，证明拒答靠的是题元谓词
    assert all(w in IN2 for w in ("我", "喜欢", "你"))

    good = R.render(R.Instruction("S01", (1, 3, 4)), bag, IN2, plan_step_id="t1")
    assert _ok(good) and good["text"] == "你喜欢我。"
    # 同骨架、只换指派：一头过一头拒 ⇒ 检查不是看字符串，是看题元
    assert R.check_structure(reversed_instr, bag, IN2) != []
    assert R.check_structure(R.Instruction("S01", (1, 3, 4)), bag, IN2) == []


# ---- T2 规则 A ---------------------------------------------------------------

def test_rule_a_skeleton_literal_with_content_word_rejected() -> None:
    """骨架字面掺实义词 ⇒ 拒；反证：同一批谓词对内置骨架全过（非空测试）。"""
    doctored = R.Skeleton("BAD", "用户说[1]。", (R.SlotSpec("动"),))
    problems = R.rule_a_problems(doctored)
    assert problems and any("规则 A" in p and "实义词" in p for p in problems), problems

    rec = R.render(
        R.Instruction("BAD", (2,)), bag1(), IN1, plan_step_id="t2",
        skeletons={**R.SKELETONS, "BAD": doctored},
    )
    assert rec["kind"] == "reject"
    assert "规则 A" in rec["reason"] and "用户" in rec["reason"], rec["reason"]

    # 非空证明：内置骨架全部通过同一套谓词（若谓词恒拒，下面这行会失败）
    for sk in R.SKELETONS.values():
        assert R.rule_a_problems(sk) == [], (sk.sid, R.rule_a_problems(sk))


# ---- T3 规则 B ---------------------------------------------------------------

def test_rule_b_type_mismatch_rejected() -> None:
    """槽位要名、指派了动 ⇒ 拒（类型匹配是谓词，不是模型判断）。"""
    instr = R.Instruction("S01", (2, 3, 1))  # 说(动) 进 名·施事 槽
    rec = R.render(instr, bag1(), IN1, plan_step_id="t3")
    assert rec["kind"] == "reject"
    assert "类型不匹配" in rec["reason"], rec["reason"]
    assert R.check_structure(instr, bag1(), IN1) != []


def test_rule_b_slot_without_theta_rejected() -> None:
    """名槽在非「只复述」骨架上不标题元 ⇒ schema 不合法 ⇒ 拒（I1 谓词）。"""
    doctored = R.Skeleton("NOTHETA", "[1][2]。", (R.SlotSpec("名"), R.SlotSpec("动")))
    instr = R.Instruction("NOTHETA", (1, 2))
    rec = R.render(
        instr, bag1(), IN1, plan_step_id="t3b",
        skeletons={**R.SKELETONS, "NOTHETA": doctored},
    )
    assert rec["kind"] == "reject"
    assert "名槽必须标题元" in rec["reason"], rec["reason"]


# ---- T4 逻辑词（编的因果） ---------------------------------------------------

def test_logic_word_undeclared_and_unbased_rejected() -> None:
    """输入没有 `因为`、意图卡也没声明 ⇒ S13 的因为是编的 ⇒ 拒；两条合法出口各验一次。"""
    instr = R.Instruction("S13", (3, 4, 5, 6, 7))
    rec = R.render(instr, bag1(), IN1, plan_step_id="t4")
    assert rec["kind"] == "reject"
    assert "词面守卫" in rec["reason"] and "因为" in rec["reason"], rec["reason"]

    # 出口 A：意图卡显式声明
    ok_decl = R.render(instr, bag1(), IN1, plan_step_id="t4", declared=R.Declared(logic=("因为",)))
    assert _ok(ok_decl) and ok_decl["text"] == "因为加载太慢，我们下周要修。"

    # 出口 B：输入逐字有据
    in2 = "因为加载太慢，我们下周要修。"
    bag = [
        R.BagItem(1, "加载", (2, 4), "动", None, "c1", True),
        R.BagItem(2, "太慢", (4, 6), "形", None, "c2", True),
        R.BagItem(3, "我们", (7, 9), "名", "施事", "c3", True),
        R.BagItem(4, "下周", (9, 11), "名", "时", "c4", True),
        R.BagItem(5, "修", (12, 13), "动", None, "c5", True),
    ]
    ok_basis = R.render(R.Instruction("S13", (1, 2, 3, 4, 5)), bag, in2, plan_step_id="t4")
    assert _ok(ok_basis) and ok_basis["text"] == in2


# ---- T5 纯函数 ---------------------------------------------------------------

def test_render_is_pure_function() -> None:
    """同输入两次渲染：整条记录 `==` 且 text 逐字相同（P2/P3 原样恢复）。"""
    a = R.render(IN11, bag1(), IN1, plan_step_id="t5")
    b = R.render(IN11, bag1(), IN1, plan_step_id="t5")
    assert a == b
    assert a["text"] == b["text"] == "用户说加载太慢，我们下周要修。"
    # 拒答路径同样确定
    ra = R.render(R.Instruction("S13", (3, 4, 5, 6, 7)), bag1(), IN1, plan_step_id="t5")
    rb = R.render(R.Instruction("S13", (3, 4, 5, 6, 7)), bag1(), IN1, plan_step_id="t5")
    assert ra == rb and ra["kind"] == "reject"


# ---- T6 映射覆盖（I3 靶子） --------------------------------------------------

def test_deref_catches_extra_word() -> None:
    """伪造一次「渲染时多吐一个词」⇒ 解引用侧谓词必须抓到（回溯链断）。"""
    rec = R.render(IN11, bag1(), IN1, plan_step_id="t6")
    assert rec["kind"] == "text"
    assert R.check_deref(rec, bag1(), IN1) == []  # 好记录必须过 —— 检查非恒拒

    def buggy_render() -> dict:
        """模拟有 bug 的渲染层：在文本尾巴上多吐一个词（映射表没跟着变）。"""
        out = dict(rec)
        out["text"] = rec["text"] + "哈哈"
        return out

    tampered = buggy_render()
    problems = R.check_deref(tampered, bag1(), IN1)
    assert problems, "多吐一个词必须被解引用侧检查抓到"
    assert "多吐" in problems[0] or "覆盖" in problems[0], problems
    # 甚至把映射表也一起伪造（给多吐的词补一条映射）⇒ 表项对不上照样抓
    forged = dict(tampered)
    forged["ref_map"] = list(rec["ref_map"]) + [
        {**rec["ref_map"][0], "unit_id": "ref:99", "ref": 99, "text": "哈哈",
         "span": (0, 2), "out": (len(rec["text"]), len(rec["text"]) + 2)}
    ]
    problems2 = R.check_deref(forged, bag1(), IN1)
    assert problems2, "伪造映射表也必须被抓（ref 不在指派里）"


# ---- T7 反例：错键的映射表 ---------------------------------------------------

def test_counterexample_wrong_map_key_fails() -> None:
    """故意把映射表改成错的键 ⇒ `check_deref` 必须产出问题；改回来必须为空。"""
    import copy

    rec = R.render(IN11, bag1(), IN1, plan_step_id="t7")
    assert R.check_deref(rec, bag1(), IN1) == []  # 基线：好记录过

    bad_ref = copy.deepcopy(rec)
    bad_ref["ref_map"][0]["ref"] = 99  # 内容条目的结构引用 id 改错
    assert R.check_deref(bad_ref, bag1(), IN1), "错 ref 键必须被抓"

    bad_unit = copy.deepcopy(rec)
    # 表项 id 改错（功能词走表项 id，不是 span —— 键错同样抓）
    lit_idx = next(i for i, e in enumerate(rec["ref_map"]) if e["cls"] != "content")
    bad_unit["ref_map"][lit_idx]["unit_id"] = "XXX#0"
    assert R.check_deref(bad_unit, bag1(), IN1), "错 unit_id 键必须被抓"

    bad_span = copy.deepcopy(rec)
    bad_span["ref_map"][0]["span"] = (0, 2)  # span 指到输入里别的位置
    assert R.check_deref(bad_span, bag1(), IN1), "span 键错必须被抓"

    # 反向断言：把键改回去，问题清零 —— 证明上面三条不是「永远非空」的假检查
    fixed = copy.deepcopy(bad_ref)
    fixed["ref_map"][0]["ref"] = rec["ref_map"][0]["ref"]
    assert R.check_deref(fixed, bag1(), IN1) == []


# ---- T8 规则 0：指派独立于袋序 ------------------------------------------------

def test_rule0_assignment_independent_of_bag_order() -> None:
    """袋序打乱（含乱序 ref）⇒ 同一指派渲染结果逐字不变（指派是重整层唯一输出）。"""
    base = R.render(IN11, bag1(), IN1, plan_step_id="t8")
    shuffled = [
        bag1()[6], bag1()[0], bag1()[3], bag1()[1], bag1()[5], bag1()[2], bag1()[4],
    ]
    got = R.render(IN11, shuffled, IN1, plan_step_id="t8")
    assert got == base
    assert got["text"] == base["text"] == "用户说加载太慢，我们下周要修。"


# ---- T9 evidence 两类 + I3 回溯链 --------------------------------------------

def test_evidence_two_kinds_and_ref_chain() -> None:
    """内容词 ⇒ 指针 span 指回输入；功能词 ⇒ 表项 id（不是 span）；记录里显式区分。"""
    rec = R.render(IN11, bag1(), IN1, plan_step_id="t9")
    assert rec["kind"] == "text"
    content = [e for e in rec["evidence"] if e["source"] == "span"]
    table = [e for e in rec["evidence"] if e["source"] == "table"]
    assert content and table, "两类 evidence 都必须出现"
    assert len(content) + len(table) == len(rec["evidence"])

    for e in content:  # 内容词：span 必须逐字指回输入
        start, end = e["span"]
        assert IN1[start:end] == e["text"]
        assert "unit_id" not in e
    for e in table:  # 功能词：表项 id，**不是** span
        assert "span" not in e and e["unit_id"].startswith("S11#")
        assert e["word"] in ("，", "要", "。")

    # I3 回溯链：文本片段 ↦ 结构引用 id ↦ 原始 span，链上每一跳都通
    for entry in rec["ref_map"]:
        assert set(entry) >= {"unit_id", "cls", "text", "ref", "span", "out"}
        if entry["cls"] == "content":
            item = {b.ref: b for b in bag1()}[entry["ref"]]
            start, end = entry["span"]
            assert IN1[start:end] == item.text == entry["text"]
            assert entry["unit_id"] == f"ref:{item.ref}"
            assert item.candidate_id  # 链尾还能接到候选（零信息新增）
        else:
            assert entry["span"] is None and entry["unit_id"].startswith("S11#")


# ---- T10 零信息新增 ----------------------------------------------------------

def test_zero_info_content_rejected() -> None:
    """内容词回溯不到「被筛为有效」的候选 ⇒ 拒（不让无出处的内容进文本）。"""
    bag = bag2()
    bag[0] = R.BagItem(1, "你", (0, 1), "名", "施事", "cand-a", False)  # 筛选器没盖章
    rec = R.render(R.Instruction("S01", (1, 3, 4)), bag, IN2, plan_step_id="t10")
    assert rec["kind"] == "reject"
    assert "零信息新增" in rec["reason"], rec["reason"]

    no_id = bag2()
    no_id[2] = R.BagItem(3, "喜欢", (2, 4), "动", None, "", True)  # 没有候选出处
    rec2 = R.render(R.Instruction("S01", (1, 3, 4)), no_id, IN2, plan_step_id="t10")
    assert rec2["kind"] == "reject" and "零信息新增" in rec2["reason"]


# ---- T11 无角色标签 fail-closed ----------------------------------------------

def test_unlabeled_theta_fails_closed_to_recite_subset() -> None:
    """名块没题元 ⇒ 不许进换方向的骨架，只许进「只复述、不换方向」子集（不猜）。"""
    bag = [
        R.BagItem(1, "用户", (7, 9), "名", None, "c1", True),   # 题元未标
        R.BagItem(2, "说", (9, 10), "动", None, "c2", True),
        R.BagItem(3, "加载", (10, 12), "动", None, "c3", True),
        R.BagItem(4, "太慢", (12, 14), "形", None, "c4", True),
    ]
    # 换方向骨架 S10（名槽必须带题元标签）⇒ fail-closed 拒答
    rec_fail = R.render(R.Instruction("S10", (1, 2, 3, 4)), bag, IN1, plan_step_id="t11")
    assert rec_fail["kind"] == "reject" and "fail-closed" in rec_fail["reason"], rec_fail["reason"]

    # 同一批块进只复述子集 R03：指派随 span 升序 ⇒ 放行
    ok = R.render(R.Instruction("R03", (1, 2, 3, 4)), bag, IN1, plan_step_id="t11")
    assert ok["kind"] == "text" and ok["text"] == "用户说加载太慢。"

    # 反向：只复述子集里把两块对调（类型仍匹配）⇒ 只因换了方向被拒
    flipped = R.render(R.Instruction("R03", (1, 3, 2, 4)), bag, IN1, plan_step_id="t11")
    assert flipped["kind"] == "reject" and "span 升序" in flipped["reason"], flipped["reason"]


# ---- T12 骨架表门禁 ----------------------------------------------------------

def test_skeleton_table_bounded_and_clean() -> None:
    """骨架表封闭、手写、可枚举（≤40）；全表过规则 A + 槽位 schema 门禁。"""
    assert len(R.SKELETONS) <= R.SKELETON_TABLE_LIMIT
    assert len(R.SKELETONS) >= 10  # 太小就没有表达力了
    assert R.validate_skeleton_table() == []
    # 反例：往表里塞一个坏骨架，门禁必须报（否则上面的 assert 是橡皮图章）
    broken = R.Skeleton("EVIL", "[1][2]试试。", (R.SlotSpec("动"), R.SlotSpec("动")))
    assert R.validate_skeleton_table({**R.SKELETONS, "EVIL": broken}) != []


if __name__ == "__main__":  # pragma: no cover —— 便于单测直跑
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))
