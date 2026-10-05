"""固定组合算子（纯后处理，零训练、不碰解码器）。

消费两张任务卡的锚点输出，输出同结构的锚点列表：

    item = {"label": str, "start": int, "end": int, "score": float}
    # start/end 是 0-based **闭区间**，与 MultiTaskEngine 的 s0/e0 同口径

两个算子：

    filter(A, B, ...)  保留 B 中落在 A 内 / 与 A 邻接的部分（B 被筛选，A 是参照）
    pair(A, B, rule)   按位置给 A 的每一项配一个 B 项，套**声明式规则**产出新列表
                       （A 项长度不变：配不上或规则不适用则原样透传）

规则是数据不是代码，形如：

    {"kind": "flip", "apply_to": ["积极"], "table": {...}, "default": None}

`kind` 目前只实现 `flip`（按查表把 label 换掉）；`default` 为 None 表示查不到就
不翻转 —— 这条"查不到不动"是刻意的：词典外的组合表达里，情绪词本身仍在词典内，
所以查表永远命中；查不中说明锚点圈的不是情绪词，这时候翻转只会制造错误。

本模块不 import 任何任务卡/引擎，纯函数，可单测。
"""
from __future__ import annotations

import re

__all__ = [
    "CLAUSE_BREAK",
    "NEGATED_CLASS",
    "TRIGGERS",
    "decide",
    "filter",
    "flip_rule",
    "gap",
    "pair",
    "same_clause",
    "to_items",
]

#: 分句标点：跨过它就认为否定作用域不覆盖另一侧（"这个方案不难，我很喜欢"）
CLAUSE_BREAK = "，。！？；、,.!?;…"

#: 否定触发闭集 —— 与 negation 卡的 NEG_MARKERS 同源（不/没/别/未/莫/勿/难道/何必）。
#: 位置由 negation 卡给出，这里只做**存在性校验**：卡吐出来的跨度有时会漂到
#: 完全不含标记字的区域（实测有 "这家书" / "我对" 这类假阳），不校验会把
#: 正向对照句一起翻掉。
TRIGGERS = ("不要", "不用", "没有", "别", "没", "不", "未", "莫", "勿", "难道", "何必")

#: 情绪词 → 被否定时的类别。**不是新知识**：就是 sentiment 卡自己显式否定式的约定
#: （dataset.py v3：不高兴/不开心 → 类别3 悲伤；不满意/不喜欢/不认可/不赞同 → 类别2 愤怒），
#: 这里把同一条映射从"词条"扩展到"词典外的否定组合式"。
NEGATED_CLASS: dict[str, str] = {
    # 否定后落 悲伤/焦虑（与词典里 不高兴/不开心/不快乐/不愉快/不舒服/不痛快 同族）
    "高兴": "悲伤", "开心": "悲伤", "快乐": "悲伤", "愉快": "悲伤",
    "喜悦": "悲伤", "欢喜": "悲伤", "兴奋": "悲伤", "激动": "悲伤",
    "幸福": "悲伤", "舒心": "悲伤", "畅快": "悲伤", "痛快": "悲伤",
    "惬意": "悲伤", "舒服": "悲伤", "舒适": "悲伤", "轻松": "悲伤",
    "享受": "悲伤",
    # 否定后落 愤怒/不满（与词典里 不满意/不喜欢/不认可/不赞同 同族）
    "满意": "愤怒", "喜欢": "愤怒", "喜爱": "愤怒", "认可": "愤怒",
    "赞同": "愤怒", "赞成": "愤怒", "欣赏": "愤怒", "佩服": "愤怒",
}


# ---------------------------------------------------------------------------
# 基础：结构转换 / 距离 / 分句
# ---------------------------------------------------------------------------
def to_items(anchors: list[dict], *, label_key: str = "category",
             start_key: str = "s0", end_key: str = "e0",
             score_key: str = "confidence") -> list[dict]:
    """引擎锚点（或任何同名结构）→ 算子用的 item 列表。空输入 → 空列表。"""
    return [
        {"label": str(a[label_key]), "start": int(a[start_key]), "end": int(a[end_key]),
         "score": float(a.get(score_key, 0.0) or 0.0)}
        for a in anchors
    ]


def gap(x: dict, y: dict) -> int:
    """两区间之间**严格相隔**的字符数；相交或相接都是 0。"""
    return max(0, max(x["start"], y["start"]) - min(x["end"], y["end"]) - 1)


def same_clause(text: str, x: dict, y: dict) -> bool:
    """两区间**之间**的那段原文里没有分句标点 ⇒ 认为在同一分句。"""
    a, b = (x, y) if x["start"] <= y["start"] else (y, x)
    lo = a["end"] + 1
    hi = b["start"]
    if hi <= lo:
        return True
    return not any(ch in CLAUSE_BREAK for ch in text[lo:hi])


def _slice(text: str, item: dict) -> str:
    return text[item["start"]:item["end"] + 1]


def _require_ok(text: str, item: dict, require: str | tuple[str, ...]) -> bool:
    """require 是字面闭集（任一命中即可）或正则串。"""
    s = _slice(text, item)
    if isinstance(require, (tuple, list)):
        return any(t in s for t in require)
    return re.search(require, s) is not None


def _adjacent(x: dict, y: dict, *, window: int, text: str | None,
              clause: bool) -> bool:
    if gap(x, y) > window:
        return False
    if not clause or text is None:
        return True
    return same_clause(text, x, y)


def _matches(a: dict, b: dict, *, mode: str, window: int,
             text: str | None, clause: bool) -> bool:
    if mode == "inside":
        return a["start"] <= b["start"] and b["end"] <= a["end"]
    if mode == "overlap":
        return a["start"] <= b["end"] and b["start"] <= a["end"]
    if mode == "adjacent":
        return _adjacent(a, b, window=window, text=text, clause=clause)
    raise ValueError(f"未知 mode={mode!r}（可选 inside / overlap / adjacent）")


# ---------------------------------------------------------------------------
# 算子 1：filter
# ---------------------------------------------------------------------------
def filter(a: list[dict], b: list[dict], *, mode: str = "adjacent",
           window: int = 8, clause: bool = False, text: str | None = None,
           require: str | tuple[str, ...] | None = None,
           require_side: str = "b") -> list[dict]:
    """保留 `b` 中与某个 `a` 满足位置规则的项；`a` 或 `b` 为空 → 空列表。

    mode:
      - "inside"   b 完整落在某个 a 内
      - "overlap"  与某个 a 相交
      - "adjacent" 与某个 a 的间隔 ≤ window（默认 8 字符）
    clause:        附加"两者之间无分句标点"（需给 text）
    require:       附加文本前置校验（字面闭集或正则），require_side 指定校验哪一侧
    """
    if not a or not b:
        return []
    refs = a
    if require is not None and require_side == "a":
        # 校验参照侧：不合格的参照不参与配对
        refs = [r for r in a if _require_ok(text or "", r, require)]
    out: list[dict] = []
    for item in b:
        if require is not None and require_side == "b" and not _require_ok(text or "", item, require):
            continue
        if any(_matches(ref, item, mode=mode, window=window, text=text, clause=clause)
               for ref in refs):
            out.append(item)
    return out


# ---------------------------------------------------------------------------
# 算子 2：pair
# ---------------------------------------------------------------------------
def flip_rule(table: dict[str, str] | None = None, *,
              apply_to: tuple[str, ...] = ("积极",), default: str | None = None) -> dict:
    """声明一条"情绪锚点被否定 ⇒ 换类别"的规则（纯数据）。"""
    return {"kind": "flip", "apply_to": tuple(apply_to),
            "table": dict(NEGATED_CLASS if table is None else table),
            "default": default}


def _lookup(rule: dict, text: str, item: dict) -> str | None:
    """在锚点跨度里找**最长**的词典命中；查不到返回 rule["default"]。"""
    s = _slice(text, item)
    hit = None
    for word in sorted(rule["table"], key=len, reverse=True):
        if word in s:
            hit = word
            break
    if hit is None:
        return rule.get("default")
    return rule["table"][hit]


def pair(a: list[dict], b: list[dict], rule: dict, *, mode: str = "adjacent",
         window: int = 8, clause: bool = False, text: str | None = None,
         require: str | tuple[str, ...] | None = None,
         require_side: str = "b") -> list[dict]:
    """对 `a` 的每一项在 `b` 里配一个最近的项并套 `rule`。

    - 配不上（`b` 空 / 无邻接项 / 分句隔断 / require 不过）⇒ 原样透传；
    - `rule["kind"] == "flip"` 且 `a.label` 在 `rule["apply_to"]` ⇒ 查表换 label；
    - 输出长度恒等于 len(a)，不增不删；`a` 为空 → 空列表。
    """
    if not a:
        return []
    pool = list(b)
    if require is not None and require_side == "b":
        pool = [x for x in pool if _require_ok(text or "", x, require)]
    out: list[dict] = []
    for item in a:
        if require is not None and require_side == "a" and not _require_ok(text or "", item, require):
            out.append({**item})
            continue
        cand = [x for x in pool
                if _matches(item, x, mode=mode, window=window, text=text, clause=clause)]
        best = min(cand, key=lambda x: (gap(item, x), -x["score"])) if cand else None
        new_label = item["label"]
        if (best is not None and rule.get("kind") == "flip"
                and item["label"] in rule.get("apply_to", ())):
            target = _lookup(rule, text or "", item)
            if target is not None:
                new_label = target
        out.append({**item, "label": new_label})
    return out


# ---------------------------------------------------------------------------
# 句级判定（基线与组合共用同一把尺子）
# ---------------------------------------------------------------------------
def decide(items: list[dict], default: str = "中性") -> str:
    """句级结论 = score 最高的锚点的 label；没有锚点 ⇒ default。"""
    if not items:
        return default
    return max(items, key=lambda x: (x["score"], -x["start"]))["label"]
