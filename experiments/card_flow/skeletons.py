"""手写骨架表（≤40 条）+ 槽位签名 —— dev-notes/16 §7.7 的「结构指令」载体。

三条硬规则的落点：
  - 规则 0：指派独立于袋序 —— 本表只给**槽位**，指派由 `flow.compose` 独立搜索；
  - 规则 A：骨架字面只许纯形式词（标点 / 们 / 吗 / 和 / 与）+ 词序；
            **逻辑词**（因为/所以/但/要…）是字面里的**例外**，但必须"输入有据"才许渲染；
  - 规则 B：槽位带类型 + 题元，指派必须类型匹配（谓词，不是模型）。

`audit()` 在 import 时执行：任何一条骨架字面漏进实义词 ⇒ 直接抛错（fail-closed）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from lexicon import FORMAL_PUNCT, FORMAL_WORDS, LOGIC_WORDS, SLOT_TYPES, THEMES

_SLOT_RE = re.compile(r"\[([a-z]\d+)\]")
_TYPE_OF_PREFIX = {"n": "名", "v": "动", "a": "形", "g": "否", "d": "数"}


@dataclass(frozen=True)
class Slot:
    name: str
    type: str
    theme: str | None = None


@dataclass(frozen=True)
class Skeleton:
    id: str
    template: str
    slots: tuple[Slot, ...] = field(default=(), repr=False)

    def lit_tokens(self) -> list[tuple[str, str]]:
        """按顺序返回 [('lit', 文本) | ('slot', 槽位名)]。"""
        out: list[tuple[str, str]] = []
        pos = 0
        for m in _SLOT_RE.finditer(self.template):
            if m.start() > pos:
                out.append(("lit", self.template[pos:m.start()]))
            out.append(("slot", m.group(1)))
            pos = m.end()
        if pos < len(self.template):
            out.append(("lit", self.template[pos:]))
        return out

    @property
    def literal_text(self) -> str:
        return "".join(t for k, t in self.lit_tokens() if k == "lit")

    @property
    def logic_words(self) -> tuple[str, ...]:
        """骨架字面里出现的逻辑词（逐个要求"输入有据"）。"""
        lit = self.literal_text
        return tuple(w for w in LOGIC_WORDS if w in lit)

    @property
    def is_formal_only(self) -> bool:
        """规则 A：字面去掉白名单标点/形式词/逻辑词后必须为空。"""
        rest = self.literal_text
        for w in LOGIC_WORDS:
            rest = rest.replace(w, "")
        for w in FORMAL_WORDS:
            rest = rest.replace(w, "")
        for ch in FORMAL_PUNCT:
            rest = rest.replace(ch, "")
        for sp in " \t":
            rest = rest.replace(sp, "")
        return rest == ""


def _mk(sid: str, template: str, themes: dict[str, str] | None = None) -> Skeleton:
    themes = themes or {}
    slots: list[Slot] = []
    for name in _SLOT_RE.findall(template):
        p = name[0]
        if p not in _TYPE_OF_PREFIX:
            raise ValueError(f"{sid}: 槽位名 {name} 前缀不认识")
        typ = _TYPE_OF_PREFIX[p]
        if typ not in SLOT_TYPES:
            raise ValueError(f"{sid}: 槽位类型 {typ} 不在取值域")
        th = themes.get(name)
        if th is not None and th not in THEMES:
            raise ValueError(f"{sid}: 题元 {th} 不在取值域")
        slots.append(Slot(name=name, type=typ, theme=th))
    if len({s.name for s in slots}) != len(slots):
        raise ValueError(f"{sid}: 槽位重名")
    if not slots:
        raise ValueError(f"{sid}: 无槽位的骨架不允许（字面必须全是形式词，纯字面句没有内容）")
    return Skeleton(id=sid, template=template, slots=tuple(slots))


#: 骨架表：**跑前写死，跑中不得增删**（要改必须新旧覆盖率都报 + 理由时间写进 PREREG）。
#: 顺序 = compose 的优先顺序（信息量大的在前）。
SKELETONS: tuple[Skeleton, ...] = (
    _mk("S09", "[n1][a1]，[n2][a2]。"),
    _mk("S10", "[n1][g1][a1]，[n2][a2]。"),
    _mk("S18", "[n1][a1]，但[n2][a2]。"),
    _mk("S19", "[n1][a1]，所以[n2][a2]。"),
    _mk("S20", "因为[n1][a1]，[n2][a2]。"),
    _mk("S13", "[n1][v1][n2]。", themes={"n1": "施事", "n2": "受事"}),
    _mk("S16", "[n1][g1][v1][n2]。", themes={"n1": "施事", "n2": "受事"}),
    _mk("S14", "[n1][v1][n2]吗？", themes={"n1": "施事", "n2": "受事"}),
    _mk("S15", "[n1][g1][v1]。", themes={"n1": "施事"}),
    _mk("S12", "[n1][v1]。", themes={"n1": "施事"}),
    _mk("S21", "[n1]要[v1]。", themes={"n1": "施事"}),
    _mk("S22", "[n1][g1][v1]吗？", themes={"n1": "施事"}),
    _mk("S23", "[n1][d1][a1]。"),
    _mk("S24", "[n1][d1][a1]吗？"),
    _mk("S04", "[n1][g1][a1]。", themes={"n1": "施事"}),
    _mk("S05", "[n1][g1][a1]吗？", themes={"n1": "施事"}),
    _mk("S02", "[n1][a1]。", themes={"n1": "施事"}),
    _mk("S03", "[n1][a1]吗？", themes={"n1": "施事"}),
    _mk("S25", "[n1][a1]！", themes={"n1": "施事"}),
    _mk("S26", "[n1][g1][a1]！", themes={"n1": "施事"}),
    _mk("S06", "[n1]和[n2]。"),
    _mk("S07", "[n1]与[n2]。"),
    _mk("S08", "[n1]和[n2]和[n3]。"),
    _mk("S11", "[n1]吗？"),
    _mk("S01", "[n1]。"),
)

BY_ID: dict[str, Skeleton] = {s.id: s for s in SKELETONS}
assert len(BY_ID) == len(SKELETONS), "骨架 id 重复"

#: 复述子集（无题元标签时只许用这些 = **不换方向**）。
#: 本表里所有骨架的槽位顺序都与"模板里从左到右"一致，方向由指派的单调性保证
#: （见 predicates.order_ok），所以**全部**骨架都在复述子集内；
#: 这里仍然显式列一份"会换方向"的黑名单位（未来加倒装骨架时必须登记）。
REORDER_SKELETONS: frozenset[str] = frozenset()


def audit() -> dict:
    """规则 A 的词典谓词：骨架字面只许纯形式词 + 已登记的逻辑词。坏一条就抛。"""
    bad = []
    for s in SKELETONS:
        if not s.is_formal_only:
            bad.append((s.id, s.literal_text))
    if bad:
        raise ValueError(f"骨架字面混入实义词（规则 A 违例）: {bad}")
    logic = {s.id: s.logic_words for s in SKELETONS if s.logic_words}
    return {"n_skeletons": len(SKELETONS), "logic_gated": logic}


audit()
