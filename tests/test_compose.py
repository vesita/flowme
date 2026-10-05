"""组合算子单测：无重叠 / 部分重叠 / 多对多 / 空输入。

    uv run python -m pytest tests/test_compose.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dtseek.tasks.compose import (
    NEGATED_CLASS,
    decide,
    filter,
    flip_rule,
    gap,
    pair,
    same_clause,
    to_items,
)


def it(label: str, start: int, end: int, score: float = 0.9) -> dict:
    return {"label": label, "start": start, "end": end, "score": score}


# ---------------------------------------------------------------------------
# 基础
# ---------------------------------------------------------------------------
def test_to_items_and_decide_empty():
    assert to_items([]) == []
    assert decide([]) == "中性"
    assert filter([], []) == []
    assert pair([], [], flip_rule()) == []


def test_decide_takes_top_score():
    assert decide([it("积极", 0, 1, 0.4), it("悲伤", 5, 6, 0.9)]) == "悲伤"
    # 同分时取更靠前的
    assert decide([it("积极", 0, 1, 0.9), it("悲伤", 5, 6, 0.9)]) == "积极"


def test_gap_and_clause():
    x, y = it("a", 0, 1), it("b", 5, 6)
    assert gap(x, y) == 3          # 相隔 2,3,4 三个字符
    assert gap(it("a", 0, 4), it("b", 4, 9)) == 0      # 部分重叠
    assert gap(it("a", 0, 1), it("b", 2, 3)) == 0      # 相接
    text = "这个方案不难，我很喜欢。"
    assert same_clause(text, it("a", 4, 4), it("b", 9, 10)) is False   # 跨逗号
    assert same_clause(text, it("a", 4, 4), it("b", 1, 1)) is True     # 同侧
    assert same_clause(text, it("a", 4, 4), it("b", 5, 5)) is True     # 紧邻无间隔


# ---------------------------------------------------------------------------
# filter：无重叠 / 部分重叠 / 多对多 / 空输入
# ---------------------------------------------------------------------------
def test_filter_empty_inputs():
    a, b = [it("否定", 0, 0)], [it("积极", 4, 5)]
    assert filter([], b) == []
    assert filter(a, []) == []


def test_filter_no_overlap_is_empty():
    a, b = [it("否定", 0, 0)], [it("积极", 20, 21)]
    assert filter(a, b, mode="overlap") == []
    assert filter(a, b, mode="adjacent", window=2) == []
    assert filter(a, b, mode="adjacent", window=40) == [b[0]]   # 拉大窗口就邻接上了


def test_filter_partial_overlap_modes():
    a, b = [it("neg", 3, 5)], [it("p", 5, 9)]     # 端点相接（部分重叠）
    assert filter(a, b, mode="overlap") == [b[0]]
    assert filter(a, b, mode="adjacent", window=0) == [b[0]]
    inside = [it("p", 4, 5)]
    assert filter(a, inside, mode="inside") == inside
    assert filter(a, [it("p", 4, 6)], mode="inside") == []      # 越界不算 inside


def test_filter_many_to_many_keeps_any_match():
    a = [it("neg", 0, 0), it("neg", 30, 30)]
    b = [it("p", 1, 2), it("p", 9, 9), it("p", 31, 32)]
    kept = filter(a, b, mode="adjacent", window=2)
    assert [x["start"] for x in kept] == [1, 31]     # 9 离两个参照都远 ⇒ 被丢


def test_filter_clause_guard_needs_text():
    text = "这个方案不难，我很喜欢。"
    a, b = [it("neg", 4, 4)], [it("p", 9, 10)]
    assert filter(a, b, mode="adjacent", window=8) == [b[0]]
    assert filter(a, b, mode="adjacent", window=8, clause=True, text=text) == []
    # 同分句内不受影响
    assert filter(a, [it("p", 6, 7)], mode="adjacent", window=8,
                  clause=True, text=text) != []


def test_filter_require_on_b():
    text = "这家书不错"
    a, b = [it("neg", 0, 2)], [it("p", 6, 7)]
    assert filter(a, b, mode="adjacent", window=8, text=text) == [b[0]]
    # b 这侧的参照不命中触发闭集 ⇒ 整条不参与配对
    assert filter(a, b, mode="adjacent", window=8, text=text, require=("不", "没")) == []
    # require 也可以校验 a 这侧
    assert filter(a, b, mode="adjacent", window=8, text=text,
                  require=("不", "没"), require_side="a") == []


# ---------------------------------------------------------------------------
# pair：配不上 / 翻转 / 分句阻断 / 多对多
# ---------------------------------------------------------------------------
def test_pair_empty_b_passes_through():
    a = [it("积极", 0, 1)]
    out = pair(a, [], flip_rule())
    assert out == a and out is not a
    assert pair(a, [], flip_rule())[0]["label"] == "积极"


def test_pair_flips_when_marker_adjacent_same_clause():
    text = "没觉得痛快。"
    sent = [it("积极", 3, 4)]
    neg = [it("否定", 0, 0)]
    out = pair(sent, neg, flip_rule(), mode="adjacent", window=8,
               clause=True, text=text)
    assert out[0]["label"] == "悲伤"           # 痛快 被否定 ⇒ 悲伤（词典既有约定）
    assert out[0]["start"] == 3 and out[0]["end"] == 4   # 位置不改


def test_pair_blocked_by_clause_and_by_missing_table_entry():
    text = "这个方案不难，我很喜欢。"
    sent = [it("积极", 9, 10)]
    neg = [it("否定", 4, 4)]
    out = pair(sent, neg, flip_rule(), mode="adjacent", window=8,
               clause=True, text=text)
    assert out[0]["label"] == "积极"           # 跨逗号 ⇒ 不翻

    # 位置规则过了、但锚点圈的不是情绪词 ⇒ 查表不中 ⇒ 不翻（default=None）
    text2 = "他说不难答案真棒"          # (4,5)='答案'，(2,2)='不'，同分句相邻
    out2 = pair([it("积极", 4, 5)], [it("否定", 2, 2)], flip_rule(),
                mode="adjacent", window=8, clause=True, text=text2)
    assert out2[0]["label"] == "积极"


def test_pair_only_flips_declared_classes():
    # apply_to 只含「积极」：否定一个悲伤词不产生翻转
    out = pair([it("悲伤", 2, 3)], [it("否定", 1, 1)], flip_rule(),
               mode="adjacent", window=8, text="并不难过")
    assert out[0]["label"] == "悲伤"


def test_pair_many_to_many_nearest_marker_wins():
    # 两个情绪锚点、两个否定标记：各自配**最近且同分句**的那个，长度不变
    text = "没高兴，也没满意。"
    sent = [it("积极", 1, 2), it("积极", 6, 7)]
    neg = [it("否定", 0, 0), it("否定", 5, 5)]
    out = pair(sent, neg, flip_rule(), mode="adjacent", window=8,
               clause=True, text=text)
    assert [x["label"] for x in out] == ["悲伤", "愤怒"]
    assert len(out) == 2


def test_pair_output_length_and_keys_stable():
    a = [it("积极", 0, 1), it("愤怒", 5, 6), it("悲伤", 9, 10)]
    out = pair(a, [it("否定", 1, 1)], flip_rule(), mode="adjacent", window=8)
    assert len(out) == 3
    assert all(set(x) == {"label", "start", "end", "score"} for x in out)
    assert out[2]["label"] == "悲伤"           # 没配上的原样透传


def test_negated_class_table_covers_both_buckets():
    assert NEGATED_CLASS["高兴"] == "悲伤" and NEGATED_CLASS["满意"] == "愤怒"
    assert all(v in ("悲伤", "愤怒") for v in NEGATED_CLASS.values())
