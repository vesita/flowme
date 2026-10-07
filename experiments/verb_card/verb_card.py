"""动词卡（P25）—— **抽取式**（词法/词表规则，不训模型），产出可作「动」槽填充的候选。

# 定义原文（跑前写死，报告须原样引用）

> **动词卡**：从输入文本 `text` 抽出**动词候选**集合。一条候选 = 三元组
> `(span, text, pos)`，其中
>   - `span = [s, e)`（0-based 半开区间）= `text` 中与**手写闭集动词表 `VERB_LEXICON`**
>     中某个词 `w` **逐字相等**的一次出现：`text[s:e] == w`；
>   - 候选的显示面 `surface = text[s:e]`（**逐字 = 输入切片**，由构造保证，非事后校验）；
>   - `pos = "动"`，`candidate_id = "verb:s:e"`，`screened = True`；
>   - 重叠处理：同一区间只保留**最长**的匹配（`(-len, s)` 贪心去重叠），保证确定性。
>
> **动词表 `VERB_LEXICON` = 三个既有手写闭集的并集**（本单元不新造词表）：
>   1. `builtin/cloze_fill.dataset.POS_LEXICON["动词"]`（完形填空卡的手写动词表）；
>   2. `tasks/render.CONTENT_LEXICON["动"]`（官方渲染层的实义词表动类）；
>   3. `builtin/person.dataset._SPEECH_VERBS`（说话动词表）。
> 并集去重后 N 条（N 由 `len(VERB_LEXICON)` 实测给出）。
>
> **可回溯性怎么保证**：候选的 `span` 是在输入上做 `str.find` 得到的下标，
> `surface` 直接取 `text[s:e]` ⇒ `item_problems` 的「span 逐字」与
> 「零信息新增（candidate_id + screened）」两条**由构造成立**，不是靠放宽谓词。

# 顺带的两个抽取器（W2 的 `否`/`数` 槽要用；同样只抽取、不训模型）

- `extract_number`：输入里最长的连续数字串（`\\d+`），`pos = "数"`；
- `extract_lexicon`：官方 `CONTENT_LEXICON` 的 名/形 词表匹配（供名/形槽补货，
  只为演示动槽骨架而复用**既有**闭集）。
"""
from __future__ import annotations

import re
from typing import Iterable

#: 抽取器产出的卡名 → 规范类别名（供 `card_flow.predicates.filter_candidates` 的 F3 核对）。
CARD_CLASS: dict[str, str] = {
    "verb": "动词",
    "number": "数量",
    "lexnoun": "名词",
    "lexadj": "形容词",
}

#: 卡名 → 类型（= 本卡产出的槽位类型）。
CARD_TYPE: dict[str, str] = {
    "verb": "动",
    "number": "数",
    "lexnoun": "名",
    "lexadj": "形",
}


def _load_verb_lexicon() -> tuple[str, ...]:
    """动词表 = 三个**既有**手写闭集的并集（去重、最长优先排序）。"""
    from dtseek.tasks.builtin.cloze_fill.dataset import POS_LEXICON
    from dtseek.tasks.builtin.person.dataset import _SPEECH_VERBS
    from dtseek.tasks.render import CONTENT_LEXICON

    words: set[str] = set()
    words.update(POS_LEXICON["动词"])          # 1) 完形填空卡的手写动词表
    words.update(CONTENT_LEXICON["动"])        # 2) 官方渲染层实义词表·动类
    words.update(_SPEECH_VERBS)                # 3) person 卡的说话动词表
    words.discard("")
    return tuple(sorted(words, key=lambda w: (-len(w), w)))


#: 手写闭集动词表（见上）。
VERB_LEXICON: tuple[str, ...] = _load_verb_lexicon()

_DIGIT_RE = re.compile(r"\d+")


def _unique_overlapping(matches: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """`(-len, s)` 贪心去重叠：最长优先，已取区间不再接受与之相交者。"""
    chosen: list[tuple[int, int]] = []
    for s, e in sorted(matches, key=lambda m: (-(m[1] - m[0]), m[0])):
        if any(not (e <= cs or s >= ce) for cs, ce in chosen):
            continue
        chosen.append((s, e))
    return sorted(chosen)


def _find_all(text: str, words: Iterable[str]) -> list[tuple[int, int]]:
    hits: list[tuple[int, int]] = []
    for w in words:
        start = 0
        while True:
            i = text.find(w, start)
            if i < 0:
                break
            hits.append((i, i + len(w)))
            start = i + 1
    return hits


def _mk(card: str, s0: int, e0: int, text: str) -> dict:
    return {
        "card": card,
        "s0": s0,
        "e0": e0,
        "surface": text[s0:e0],
        "class_id": 1,
        "class_name": CARD_CLASS[card],
        "theme": None,
        "type": CARD_TYPE[card],
        "cid": f"{card}:{s0}:{e0}",
    }


def extract_verb(text: str) -> list[dict]:
    """动词卡主出口：输入 → 动词候选（定义见模块 docstring，逐字可回溯）。"""
    spans = _unique_overlapping(_find_all(text, VERB_LEXICON))
    return [_mk("verb", s, e, text) for s, e in spans]


def extract_number(text: str) -> list[dict]:
    """数槽候选：输入里最长的连续数字串（逐字切片）。"""
    return [_mk("number", m.start(), m.end(), text) for m in _DIGIT_RE.finditer(text)]


def extract_lexicon(text: str) -> list[dict]:
    """名/形槽补货：官方 `CONTENT_LEXICON` 的名/形词逐字匹配（既有闭集，非本单元新造）。"""
    from dtseek.tasks.render import CONTENT_LEXICON

    out: list[dict] = []
    for card, pos_words in (("lexnoun", CONTENT_LEXICON["名"]), ("lexadj", CONTENT_LEXICON["形"])):
        for s, e in _unique_overlapping(_find_all(text, pos_words)):
            out.append(_mk(card, s, e, text))
    return out


def propose(text: str, *, with_verb: bool = True) -> list[dict]:
    """本单元的抽取出口（不含既有假卡）：动词卡 + 数字抽取 + 官方词典补货。"""
    out: list[dict] = []
    if with_verb:
        out += extract_verb(text)
    out += extract_number(text)
    out += extract_lexicon(text)
    return out
