"""① 候选区的两个提案者（都只产出「类别 + span」，内容词天生可回溯）。

  - `fake_propose`：**假卡**（词典匹配的模拟指针卡）—— 先用它把管线跑通、自检，
    不碰 GPU。用的词典全部 import 自 `src/dtseek/tasks/builtin/`（只读）。
  - `real_propose`：**真卡**（`MultiTaskEngine.predict`），逐卡计时供前向时间对账。

两者产出**同一形状**的候选，因此 ②③④⑤ 段完全共用 —— 假卡与真卡只差在提案质量。
"""
from __future__ import annotations

import time

#: 假卡用到的类别体系（与注册表里的 TaskSpec 一致；F3 会按它核对规范名）。
FAKE_CLASS_INDEX: dict[str, tuple[str, ...]] = {
    "person": tuple(f"人物{i}" for i in range(1, 8)),
    "pronoun": ("第一人称", "第二人称", "第三人称"),
    "sentiment": ("中性", "积极", "愤怒", "悲伤"),
    "negation": ("背景", "否定"),
    "idiom": ("非成语", "成语"),
}

_PRONOUN_CLASS = {
    "我": "第一人称", "我们": "第一人称", "咱们": "第一人称",
    "你": "第二人称", "您": "第二人称", "你们": "第二人称",
    "他": "第三人称", "她": "第三人称", "它": "第三人称",
    "他们": "第三人称", "她们": "第三人称",
}

_PRONOUNS = tuple(sorted(_PRONOUN_CLASS, key=len, reverse=True))


def _finish(raw: list[dict]) -> list[dict]:
    """统一 cid 与类型：按 (s0, e0, card) 排序后编号，保证确定性（同一输入必得同一 cid）。

    类型一律由 `lexicon.type_of` 现算（词典谓词），提案者不许自造类型。
    """
    from lexicon import type_of

    raw.sort(key=lambda c: (c["s0"], c["e0"], c["card"]))
    return [{**c, "cid": f"c{i}", "type": type_of(c["card"], c["class_name"], c["surface"])}
            for i, c in enumerate(raw)]


def fake_propose(text: str) -> tuple[list[dict], dict[str, tuple[str, ...]]]:
    """假卡：词典匹配出 人物 / 代词 / 情绪 / 否定 / 成语 五类切片。"""
    from dtseek.tasks.builtin.idiom.lexicon import IDIOMS
    from dtseek.tasks.builtin.negation.dataset import NEG_MARKERS
    from dtseek.tasks.builtin.person.dataset import FEMALE_NAMES, MALE_NAMES
    from dtseek.tasks.builtin.sentiment.dataset import LEXICON_BY_CAT

    CAT_NAME = {1: "积极", 2: "愤怒", 3: "悲伤"}
    raw: list[dict] = []

    def add(card: str, s0: int, e0: int, cls_id: int, cls_name: str) -> None:
        raw.append({"card": card, "s0": s0, "e0": e0, "surface": text[s0:e0],
                    "class_id": cls_id, "class_name": cls_name, "theme": None})

    def find_all(words, fn) -> None:
        for w in sorted(set(words), key=len, reverse=True):
            if not w:
                continue
            start = 0
            while True:
                i = text.find(w, start)
                if i < 0:
                    break
                fn(i, i + len(w))
                start = i + 1

    find_all(MALE_NAMES + FEMALE_NAMES,
             lambda a, b: add("person", a, b, 1, "人物1"))

    def _pron(a: int, b: int) -> None:
        name = _PRONOUN_CLASS[text[a:b]]
        cls_id = FAKE_CLASS_INDEX["pronoun"].index(name) + 1
        add("pronoun", a, b, cls_id, name)

    find_all(_PRONOUNS, _pron)
    for cat, words in LEXICON_BY_CAT.items():
        find_all(words, lambda a, b, c=cat: add("sentiment", a, b, c, CAT_NAME[c]))
    find_all(NEG_MARKERS, lambda a, b: add("negation", a, b, 1, "否定"))
    find_all(IDIOMS, lambda a, b: add("idiom", a, b, 1, "成语"))

    return _finish(raw), FAKE_CLASS_INDEX


def real_propose(text: str, engine, cards: list[str],
                 timers: dict[str, list[float]] | None = None,
                 steps: dict[str, list[int]] | None = None,
                 segs: dict[str, list[int]] | None = None
                 ) -> tuple[list[dict], dict[str, tuple[str, ...]]]:
    """真卡：逐卡 `engine.predict(text, tasks=[card])`，顺便记每卡前向时间 / 发射步 / 段数。"""
    raw: list[dict] = []
    class_index: dict[str, tuple[str, ...]] = {}
    for card in cards:
        t0 = time.perf_counter()
        res = engine.predict(text, tasks=[card])
        dt = (time.perf_counter() - t0) * 1000.0
        if timers is not None:
            timers.setdefault(card, []).append(dt)
        anchors = res.get("tasks", {}).get(card, [])
        if steps is not None:
            # 发射步数 = 每段实际吐出的 step 数（predict 已把 step 编进 anchors）
            steps.setdefault(card, []).append(len(anchors))
        if segs is not None:
            segs.setdefault(card, []).append(int(res.get("num_segments", 1)))
        spec = engine.specs[card]
        class_index[card] = tuple(c.name for c in spec.classes)
        for a in anchors:
            s0, e0 = a["s0"], a["e0"]
            if not (0 <= s0 < e0 <= len(text)):
                # 指针吐出越界区间：原样交给 ② 筛选区判（不在这儿私自修）
                surf = ""
            else:
                surf = text[s0:e0]
            raw.append({"card": card, "s0": s0, "e0": e0, "surface": surf,
                        "class_id": a["class_id"], "class_name": a["class_name"],
                        "theme": None})
    return _finish(raw), class_index
