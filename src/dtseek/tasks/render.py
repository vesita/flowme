"""重整层（dev-notes/16 §7.7 结构化指令 + 确定性渲染）与解引用层（§7.9 引用留存）。

输出链三段（§7.9）::

    上游给：袋（逐字 span + 词类 + 题元）+ 结构指令（骨架 id + 槽位指派）
       ↓ 本层：I1 结构侧谓词校验 → 渲染（纯函数）→ 解引用侧谓词校验
    用户看：text（自然语言）+ evidence（两类）+ ref_map（引用映射，存进记录）

**重整层的唯一输出是结构指令**（`Instruction` = 骨架 id + 槽位指派），
不是文本；`render()` 只是把已过检的指令做**确定性纯函数**渲染（同输入逐字相同）。

三条硬规则：

- **规则 0（指派独立于袋序）**：指派是 `slot → 袋引用` 的映射，按引用查表，
  与袋在列表里的顺序无关（`test_rule0` 实测）。
- **规则 A（骨架字面只许纯形式词）**：`rule_a_problems()` —— 实义词 ∩ 骨架字面 = ∅
  （词典谓词 `is_content_word`），且字面必须能整体分解进 形式词（们/吗/和/与 + 标点）
  ∪ 逻辑词表；词典外的实义词 / 专名也过不了分解。
- **规则 B（槽位带类型 + 题元）**：`slot_schema_problems()` + `check_structure()` ——
  槽位类型 ∈ {名,动,形,否,数}（门禁查 `POS_TYPES_ALL`；对外取值域 `POS_TYPES` 仍是
  {名,动,形}，见下）、名槽题元 ∈ {施事,受事,时}，指派必须类型匹配、题元一致。

词面守卫（§7.7 未解决的两条）：数字 / 否定 / 情态 / 专名 / 因果-转折连词
必须**逐字（输入有据）或显式（意图卡声明）**，否则拒答 —— `word_face_problems()`。

两道检查（§7.9）：

1. **结构侧** `check_structure()`：I1 schema 谓词（类型 / 题元 / evidence 齐备 / 零信息新增），
   不满足 ⇒ `kind="reject"`，**不交给模型判**；
2. **解引用侧** `check_deref()`：文本的每个字符必须被 `ref_map` 覆盖且每条映射
   都能解析回结构单元（内容词 ↦ 袋引用 ↦ 原始 span；功能词 ↦ 表项 id），
   挡住「渲染时偷偷加内容」。

用法::

    rec = render(instr, bag, text, plan_step_id="step:1")
    rec["kind"]      # "text" 或 "reject"
    rec["ref_map"]   # I3 引用映射：文本片段 ↦ 结构引用 id ↦ 原始 span
"""
from __future__ import annotations

import re
from dataclasses import dataclass

__all__ = [
    "CONTENT_LEXICON",
    "CONTENT_WORDS",
    "FORMAL_WORDS",
    "LOGIC_WORDS",
    "POS_TYPES",
    "POS_TYPES_ALL",
    "SKELETONS",
    "SKELETON_CF",
    "SKELETON_OFFICIAL",
    "SKELETON_TABLE_LIMIT",
    "THETAS",
    "BagItem",
    "Declared",
    "Instruction",
    "Skeleton",
    "SlotSpec",
    "check_deref",
    "check_structure",
    "is_content_word",
    "render",
    "rule_a_problems",
    "validate_skeleton_table",
    "word_face_problems",
]

# ---- 词表（手写闭集，可枚举） ------------------------------------------------

#: 槽位 / 袋块的**类型**取值域（§7.7 规则 B：名 / 动 / 形）。**对外取值域，保持 P13 取值不动**：
#: 既有读方拿它当「有哪些类型」去迭代并索引 `CONTENT_LEXICON[pos]`
#: （`experiments/gen_dispatch/pipeline.py::type_pos`），或当白名单筛候选
#: （`experiments/pointer_explain/ptr_explain.py`）⇒ 直接扩它会改它们的默认行为
#: （实测：`POS_TYPES` 加 否/数 ⇒ pytest 3 failed，`KeyError: '否'`；证据见
#: `experiments/verb_card/report.md`）。
POS_TYPES: tuple[str, ...] = ("名", "动", "形")

#: **门禁域**（P25）：`slot_schema_problems()` / `item_problems()` 查这张表
#: = `POS_TYPES` ∪ {否, 数}。`否`（否定标记）/ `数`（数字串）的候选已有抽取式产出
#: （`experiments/verb_card/`）⇒ card_flow 表里那 9 条含 `否`/`数` 槽的骨架可过门禁，
#: 而 `POS_TYPES` 不动 ⇒ 既有读方逐字不变（等价性实测：官方 16 条渲染逐字不变 +
#: 全库 pytest 258 passed，见 `experiments/verb_card/report.md`）。
POS_TYPES_ALL: tuple[str, ...] = POS_TYPES + ("否", "数")

#: **题元**取值域（§7.7 规则 B：施事 / 受事 / 时）。`None` = 未标（fail-closed，见下）。
THETAS: tuple[str, ...] = ("施事", "受事", "时")

#: 实义词词典（手写闭集）。规则 A 的谓词就查这张表：实义词 ∩ 骨架字面必须 = ∅。
CONTENT_LEXICON: dict[str, tuple[str, ...]] = {
    "名": (
        "用户", "我们", "项目", "复盘", "下周", "今天", "明天", "时间",
        "速度", "方案", "问题", "你", "我",
    ),
    "动": (
        "说", "加载", "修", "喜欢", "觉得", "讨论", "解决", "开始", "试试",
        "跑", "看", "做",
    ),
    "形": ("太慢", "快", "慢", "好", "重要", "清楚", "难"),
}

#: 全部实义词，按长度降序（报错信息稳定、最长匹配优先）。
CONTENT_WORDS: tuple[str, ...] = tuple(
    sorted({w for words in CONTENT_LEXICON.values() for w in words}, key=lambda w: (-len(w), w))
)
_CONTENT_SET: frozenset[str] = frozenset(CONTENT_WORDS)

#: 纯形式功能词（§7.7 规则 A）：零命题贡献，可自由插；标点同样算纯形式。
FORMAL_WORDS: tuple[str, ...] = ("们", "吗", "和", "与")

#: 逻辑词（§7.7「连词/情态分两级」）：因果-转折连词 + 情态 + 否定。
#: **有真值贡献** ⇒ 只能「输入逐字有据」或「意图卡显式声明」，否则拒（编的因果）。
LOGIC_WORDS: tuple[str, ...] = ("因为", "所以", "可以", "但", "要", "能", "不", "没")

#: 骨架字面里允许出现的标点（纯形式）。
PUNCT: frozenset[str] = frozenset("，。？！、；：…—“”‘’「」『』（）()《》〈〉·,?!;:\"' \t")

#: 骨架表规模上限（§7.7：表达力上限 = 骨架表规模，表必须封闭可枚举）。
SKELETON_TABLE_LIMIT = 40

_ALLOW_WORDS: tuple[str, ...] = tuple(
    sorted(FORMAL_WORDS + LOGIC_WORDS, key=lambda w: (-len(w), w))
)
_LOGIC_SET: frozenset[str] = frozenset(LOGIC_WORDS)
_SLOT_RE = re.compile(r"\[(\d+)\]")


def is_content_word(word: str) -> bool:
    """词典谓词：该词是否实义词（规则 A 就是拿这张表查骨架字面）。"""
    return word in _CONTENT_SET


# ---- 数据模型 ----------------------------------------------------------------

@dataclass(frozen=True)
class BagItem:
    """袋里的一个块：**逐字 + span + 类型 + 题元**，外加零信息新增要求的候选出处。

    `span` 是 0-based 半开区间，落在**输入原文**上；`text` 必须等于 `input[span]`。
    `pos` ∈ `POS_TYPES`；`theta` ∈ `THETAS` 或 `None`（未标 ⇒ fail-closed 到只复述子集）。
    `candidate_id` / `screened`：内容词必须能回溯到「被筛为有效」的候选（零信息新增）。
    """

    ref: int
    text: str
    span: tuple[int, int]
    pos: str | None = None
    theta: str | None = None
    candidate_id: str = ""
    screened: bool = False


@dataclass(frozen=True)
class SlotSpec:
    """骨架槽位的类型 + 题元声明。名槽在非复述骨架上**必须**标题元（规则 B）。"""

    pos: str
    theta: str | None = None


@dataclass(frozen=True)
class Skeleton:
    """一条骨架：`pattern` 里的 `[k]` 指向 `slots[k-1]`，其余字面只许纯形式词/逻辑词。

    `direction_safe=True` 是 §7.7 的「只复述、不换方向」子集：名槽不标题元，
    且指派必须随原文 span 升序（无角色标签的块只许进这个子集，不猜方向）。
    """

    sid: str
    pattern: str
    slots: tuple[SlotSpec, ...]
    direction_safe: bool = False


@dataclass(frozen=True)
class Instruction:
    """**结构指令 = 重整层唯一的输出**：骨架 id + 槽位指派（规则 0：与袋序无关）。"""

    skeleton_id: str
    assignment: tuple[int, ...]


@dataclass(frozen=True)
class Declared:
    """意图卡的**显式声明**：逻辑词与数字/专名对齐都要「输入有据」或在这里声明。"""

    logic: tuple[str, ...] = ()
    align: tuple[str, ...] = ()


DECLARED_NONE = Declared()

# ---- 骨架表（手写、封闭、可枚举） --------------------------------------------

#: 骨架表（**官方族**，P13 之前就存在的 16 条，一字未改）。
#: 顺序 = 穷举式调用方的优先序（`_SKELETON_LIST` 保持官方族 ⇒ 这些调用方看到的序列不变）。
_SKELETON_LIST: tuple[Skeleton, ...] = (
    Skeleton("S01", "[1][2][3]。", (SlotSpec("名", "施事"), SlotSpec("动"), SlotSpec("名", "受事"))),
    Skeleton("S02", "[1][2][3]吗？", (SlotSpec("名", "施事"), SlotSpec("动"), SlotSpec("名", "受事"))),
    Skeleton("S03", "[1][2][3]！", (SlotSpec("名", "施事"), SlotSpec("动"), SlotSpec("名", "受事"))),
    Skeleton("S04", "[1][2]。", (SlotSpec("名", "施事"), SlotSpec("动"))),
    Skeleton("S05", "[1][2]，[3][4]。", (
        SlotSpec("名", "施事"), SlotSpec("动"), SlotSpec("名", "施事"), SlotSpec("动"))),
    Skeleton("S06", "[1]们[2]。", (SlotSpec("名", "施事"), SlotSpec("动"))),
    Skeleton("S07", "[1]和[2][3]。", (SlotSpec("名", "施事"), SlotSpec("名", "受事"), SlotSpec("动"))),
    Skeleton("S08", "[1]与[2][3]。", (SlotSpec("名", "施事"), SlotSpec("名", "受事"), SlotSpec("动"))),
    Skeleton("S09", "[1][2]吗？", (SlotSpec("名", "施事"), SlotSpec("动"))),
    Skeleton("S10", "[1][2][3][4]。", (
        SlotSpec("名", "施事"), SlotSpec("动"), SlotSpec("动"), SlotSpec("形"))),
    # §7.7 的例子：字面里的「要」是情态逻辑词 ⇒ 走词面守卫（输入有据 / 意图卡声明）
    Skeleton("S11", "[1][2][3][4]，[5][6]要[7]。", (
        SlotSpec("名", "施事"), SlotSpec("动"), SlotSpec("动"), SlotSpec("形"),
        SlotSpec("名", "施事"), SlotSpec("名", "时"), SlotSpec("动"))),
    Skeleton("S12", "[1][2]，[3][4]要[5]。", (
        SlotSpec("动"), SlotSpec("形"), SlotSpec("名", "施事"), SlotSpec("名", "时"), SlotSpec("动"))),
    Skeleton("S13", "因为[1][2]，[3][4]要[5]。", (
        SlotSpec("动"), SlotSpec("形"), SlotSpec("名", "施事"), SlotSpec("名", "时"), SlotSpec("动"))),
    # —— 「只复述、不换方向」子集（无角色标签的块只能进这里）——
    Skeleton("R01", "[1][2][3]。", (SlotSpec("名"), SlotSpec("动"), SlotSpec("形")),
             direction_safe=True),
    Skeleton("R02", "[1][2]。", (SlotSpec("名"), SlotSpec("动")), direction_safe=True),
    Skeleton("R03", "[1][2][3][4]。",
             (SlotSpec("名"), SlotSpec("动"), SlotSpec("动"), SlotSpec("形")),
             direction_safe=True),
)

#: 迁移族（**P13 卡片契约统一**）：`experiments/card_flow/skeletons.py` 的 25 条里
#: 语义归并进官方 3 条（S13→S01 / S14→S02 / S12→S04）、废弃 9 条（含 `否`/`数` 槽，
#: P13 时过不了 `slot_schema_problems` 门禁；P25 起门禁域 `POS_TYPES_ALL` 已含这两类
#: ⇒ 可再过门禁，见 `experiments/verb_card/`）、其余 **13 条**落在这里，
#: id 改到 **`CF##` 命名空间**（`##` = 原 card_flow 编号）⇒ 与官方族 id 不相交，
#: 从结构上杜绝「同 id 不同签名」。逐 id 的 `旧 → 新` 见 `experiments/card_contract/migration.py`。
#:
#: 构造规定：`[n1]→[1]` 顺序重编号、**题元一律丢弃** ⇒ `direction_safe=True`
#: （实测 card_flow A 集 48/48 个候选 `theme=None`，真卡不产题元；保留题元会让这些记录
#: 全被「无角色标签 ⇒ fail-closed」拒掉）。
_CF_SKELETON_LIST: tuple[Skeleton, ...] = (
    Skeleton("CF01", "[1]。", (SlotSpec("名"),), direction_safe=True),
    Skeleton("CF02", "[1][2]。", (SlotSpec("名"), SlotSpec("形")), direction_safe=True),
    Skeleton("CF03", "[1][2]吗？", (SlotSpec("名"), SlotSpec("形")), direction_safe=True),
    Skeleton("CF06", "[1]和[2]。", (SlotSpec("名"), SlotSpec("名")), direction_safe=True),
    Skeleton("CF07", "[1]与[2]。", (SlotSpec("名"), SlotSpec("名")), direction_safe=True),
    Skeleton("CF08", "[1]和[2]和[3]。",
             (SlotSpec("名"), SlotSpec("名"), SlotSpec("名")), direction_safe=True),
    Skeleton("CF09", "[1][2]，[3][4]。",
             (SlotSpec("名"), SlotSpec("形"), SlotSpec("名"), SlotSpec("形")),
             direction_safe=True),
    Skeleton("CF11", "[1]吗？", (SlotSpec("名"),), direction_safe=True),
    Skeleton("CF18", "[1][2]，但[3][4]。",
             (SlotSpec("名"), SlotSpec("形"), SlotSpec("名"), SlotSpec("形")),
             direction_safe=True),
    Skeleton("CF19", "[1][2]，所以[3][4]。",
             (SlotSpec("名"), SlotSpec("形"), SlotSpec("名"), SlotSpec("形")),
             direction_safe=True),
    Skeleton("CF20", "因为[1][2]，[3][4]。",
             (SlotSpec("名"), SlotSpec("形"), SlotSpec("名"), SlotSpec("形")),
             direction_safe=True),
    Skeleton("CF21", "[1]要[2]。", (SlotSpec("名"), SlotSpec("动")), direction_safe=True),
    Skeleton("CF25", "[1][2]！", (SlotSpec("名"), SlotSpec("形")), direction_safe=True),
)

#: 手写表全集（唯一一家的全部手写条目；`_SKELETON_LIST` 仍是官方族的封闭序列）。
_ALL_SKELETON_LIST: tuple[Skeleton, ...] = _SKELETON_LIST + _CF_SKELETON_LIST

#: **骨架表（唯一一家）**：官方 16 条 + 迁移族 13 条 = 29 条，封闭、手写、可枚举。
#: import 时即跑 `validate_skeleton_table`（带病不进表）。
SKELETONS: dict[str, Skeleton] = {sk.sid: sk for sk in _ALL_SKELETON_LIST}

#: 官方族视图（id 前缀 `S`/`R`）与迁移族视图（id 前缀 `CF`）—— 两族 id 不相交（V1 的隔离证明）。
SKELETON_OFFICIAL: dict[str, Skeleton] = {sk.sid: sk for sk in _SKELETON_LIST}
SKELETON_CF: dict[str, Skeleton] = {sk.sid: sk for sk in _CF_SKELETON_LIST}


# ---- 读表：骨架 pattern → token 序列（无损分词） -----------------------------

def _literal_tokens(piece: str) -> list[tuple[str, str]]:
    """把骨架字面无损切成 token：(kind, text)，kind ∈ {formal, logic, align, raw}。

    标点/空白 ⇒ formal；数字串 ⇒ align；其余 CJK 用 形式词∪逻辑词 表**贪心最长匹配**；
    匹配不上 ⇒ raw（规则 A 的第二道谓词会拒，见 `rule_a_problems`）。
    """
    tokens: list[tuple[str, str]] = []
    i, n = 0, len(piece)
    while i < n:
        ch = piece[i]
        if ch in PUNCT:
            j = i
            while j < n and piece[j] in PUNCT:
                j += 1
            tokens.append(("formal", piece[i:j]))
            i = j
        elif ch.isdigit():
            j = i
            while j < n and piece[j].isdigit():
                j += 1
            tokens.append(("align", piece[i:j]))
            i = j
        else:
            for w in _ALLOW_WORDS:
                if piece.startswith(w, i):
                    tokens.append(("logic" if w in _LOGIC_SET else "formal", w))
                    i += len(w)
                    break
            else:
                tokens.append(("raw", piece[i]))
                i += 1
    return tokens


def pattern_tokens(pattern: str) -> list[tuple[str, object]]:
    """整条 pattern 的 token 序列：`("slot", 槽序号)` / `("lit", kind, 字面)`。

    顺序即**表面顺序**；渲染与解引用检查都从这张序列出发（读表层，非渲染层）。
    """
    out: list[tuple[str, object]] = []
    pos = 0
    for m in _SLOT_RE.finditer(pattern):
        out.extend(("lit", k, v) for k, v in _literal_tokens(pattern[pos:m.start()]))
        out.append(("slot", int(m.group(1))))
        pos = m.end()
    out.extend(("lit", k, v) for k, v in _literal_tokens(pattern[pos:]))
    return out


def _literal_seq(pattern: str) -> list[tuple[str, str]]:
    """骨架字面 token（不含槽位）：解引用侧用它核对表项 id 与字面逐字一致。"""
    return [(v[1], v[2]) for v in pattern_tokens(pattern) if v[0] == "lit"]  # type: ignore[index]


def _cls(kind: str) -> str:
    """字面 token 的记录分类：逻辑词/数字 ⇒ 自己的类，其余（含标点）⇒ 纯形式 formal。"""
    return kind if kind in ("logic", "align") else "formal"


# ---- 规则 A ------------------------------------------------------------------

def rule_a_problems(skeleton: Skeleton) -> list[str]:
    """规则 A：骨架字面只许纯形式词（+ 逻辑词走词面守卫），实义词一律进槽。

    两道谓词：① 词典谓词 —— 实义词 ∩ 骨架字面 = ∅；
    ② 无损分词 —— 字面必须整体分解进 形式词∪逻辑词，挡住词典外的实义词与专名。
    """
    probs: list[str] = []
    literal = _SLOT_RE.sub("", skeleton.pattern)
    for w in CONTENT_WORDS:
        if w in literal:
            probs.append(f"规则 A 违规：骨架 {skeleton.sid} 字面含实义词 {w!r}（实义词必须进槽）")
    for kind, text in _literal_tokens(literal):
        if kind == "raw":
            probs.append(
                f"规则 A 违规：骨架 {skeleton.sid} 字面 {text!r} 既非形式词也非逻辑词（疑似实义词/专名）")
    return probs


def word_face_problems(
    input_text: str, skeleton: Skeleton, declared: Declared = DECLARED_NONE
) -> list[str]:
    """词面守卫：逻辑词（因果-转折/情态/否定）与数字必须**逐字或显式**对齐。

    专名不在字面里出现由规则 A 保证（词典外字面直接拒）；槽位里的专名/数字
    由 `item_problems` 的逐字 span 检查保证 —— 守卫清单：数字/否定/情态/专名/因果连词。
    """
    probs: list[str] = []
    for kind, text in _literal_seq(skeleton.pattern):
        if kind == "logic":
            if text in declared.logic or text in input_text:
                continue
            probs.append(
                f"词面守卫：逻辑词 {text!r} 输入无据、意图卡未声明 ⇒ 拒（编的因果/情态/否定）")
        elif kind == "align":
            if text in declared.align or text in input_text:
                continue
            probs.append(f"词面守卫：骨架 {skeleton.sid} 的数字 {text!r} 未逐字对齐、也未显式声明")
    return probs


# ---- 槽位 schema（规则 B 的表侧） --------------------------------------------

def slot_schema_problems(skeleton: Skeleton) -> list[str]:
    """槽位 schema：类型域、题元域、pattern↔slots 一一对应（I1 的 schema 谓词之一）。"""
    probs: list[str] = []
    refs = [int(m.group(1)) for m in _SLOT_RE.finditer(skeleton.pattern)]
    if sorted(refs) != list(range(1, len(skeleton.slots) + 1)):
        probs.append(
            f"规则 B 违规：骨架 {skeleton.sid} 的 pattern 槽位引用 {refs} "
            f"与 {len(skeleton.slots)} 个槽位不是一一对应")
    for k, slot in enumerate(skeleton.slots, 1):
        if slot.pos not in POS_TYPES_ALL:
            probs.append(f"规则 B 违规：骨架 {skeleton.sid} 槽{k} 类型 {slot.pos!r} 不在 {POS_TYPES_ALL}")
            continue
        if slot.pos == "名":
            if skeleton.direction_safe and slot.theta is not None:
                probs.append(
                    f"规则 B 违规：骨架 {skeleton.sid} 槽{k} 是只复述子集，名槽不许标题元")
            elif not skeleton.direction_safe and slot.theta not in THETAS:
                probs.append(
                    f"规则 B 违规：骨架 {skeleton.sid} 槽{k} 名槽必须标题元 ∈ {THETAS}，"
                    f"当前 {slot.theta!r}")
        elif slot.theta is not None:
            probs.append(f"规则 B 违规：骨架 {skeleton.sid} 槽{k} 谓词/属性槽不标题元")
    return probs


# ---- I1 · 结构侧检查 ---------------------------------------------------------

def item_problems(item: BagItem, input_text: str) -> list[str]:
    """单个袋块：evidence 齐备（span 逐字 + 类型/题元在域内）+ 零信息新增可回溯。"""
    probs: list[str] = []
    start, end = item.span
    if not (0 <= start < end <= len(input_text)):
        probs.append(f"evidence 不齐备：袋 {item.ref} 区间 {item.span} 越界")
    elif input_text[start:end] != item.text:
        probs.append(
            f"evidence 不齐备：袋 {item.ref} {item.text!r} 不是输入逐字子串"
            f"（span 给出的是 {input_text[start:end]!r}）")
    if item.pos is not None and item.pos not in POS_TYPES_ALL:
        probs.append(f"evidence 不齐备：袋 {item.ref} 类型 {item.pos!r} 不在 {POS_TYPES_ALL}")
    if item.theta is not None and item.theta not in THETAS:
        probs.append(f"evidence 不齐备：袋 {item.ref} 题元 {item.theta!r} 不在 {THETAS}")
    if not item.screened or not item.candidate_id:
        probs.append(
            f"零信息新增违规：袋 {item.ref} {item.text!r} 未回溯到「被筛为有效」的候选"
            f"（candidate_id={item.candidate_id!r}, screened={item.screened}）")
    return probs


def check_structure(
    instruction: Instruction,
    bag: list[BagItem],
    input_text: str,
    *,
    declared: Declared = DECLARED_NONE,
    skeletons: dict[str, Skeleton] | None = None,
) -> list[str]:
    """**结构侧检查（I1）**：schema + 规则 0/A/B + 词面守卫 + 零信息新增。

    返回问题列表；非空 ⇒ 调用方**拒答**（fail-closed，不交给模型判）。
    """
    table = SKELETONS if skeletons is None else skeletons
    skeleton = table.get(instruction.skeleton_id)
    if skeleton is None:
        return [f"骨架 id 不存在：{instruction.skeleton_id!r}（封闭表，不猜）"]
    probs: list[str] = []
    probs += rule_a_problems(skeleton)
    probs += slot_schema_problems(skeleton)
    probs += word_face_problems(input_text, skeleton, declared)
    if len(instruction.assignment) != len(skeleton.slots):
        probs.append(
            f"规则 B 违规：骨架 {skeleton.sid} 有 {len(skeleton.slots)} 个槽，"
            f"指派了 {len(instruction.assignment)} 项（缺块/多块 ⇒ 拒）")
        return probs

    items = {b.ref: b for b in bag}
    seen: set[int] = set()
    assigned: list[BagItem] = []
    for k, (slot, ref) in enumerate(zip(skeleton.slots, instruction.assignment), 1):
        item = items.get(ref)
        if item is None:
            probs.append(f"槽{k} 指派了袋里不存在的引用 {ref}（指派必须指向袋块）")
            continue
        if ref in seen:
            probs.append(f"规则 0/B 违规：袋 {ref} 被指派到多个槽（内容词只回溯一次）")
            continue
        seen.add(ref)
        assigned.append(item)
        probs += item_problems(item, input_text)
        if item.pos is None:
            probs.append(f"规则 B 违规：槽{k} 指派的 {item.text!r} 没有类型（fail-closed）")
            continue
        if item.pos != slot.pos:
            probs.append(
                f"规则 B 违规（类型不匹配）：槽{k} 要 {slot.pos}，指派了 "
                f"{item.text!r}（{item.pos}）")
        if slot.pos == "名" and item.pos == "名" and not skeleton.direction_safe:
            if item.theta is None:
                probs.append(
                    f"无角色标签 ⇒ fail-closed：{item.text!r} 只能进「只复述、不换方向」"
                    f"的骨架子集（不猜方向）")
            elif item.theta != slot.theta:
                probs.append(
                    f"题元方向冲突（规则 B）：槽{k} 要 {slot.theta}，指派了 "
                    f"{item.text!r}（{item.theta}）⇒ 拒（字符串全过、命题相反的靶子）")
    if skeleton.direction_safe and len(assigned) == len(skeleton.slots):
        starts = [it.span[0] for it in assigned]
        if starts != sorted(starts):
            probs.append(f"只复述子集违规：骨架 {skeleton.sid} 的指派不随原文 span 升序（换了方向）")
    return probs


# ---- 渲染（确定性纯函数） ----------------------------------------------------

def _record(
    kind: str,
    text: str,
    evidence: list[dict],
    plan_step_id: str,
    *,
    instruction: Instruction | None = None,
    ref_map: list[dict] | None = None,
    reason: str = "",
) -> dict:
    """统一记录（dev-notes/16 §1）：{kind, text, evidence, plan_step_id} + 引用映射。"""
    return {
        "kind": kind,
        "text": text,
        "evidence": evidence,
        "plan_step_id": plan_step_id,
        "instruction": (
            {"skeleton_id": instruction.skeleton_id,
             "assignment": list(instruction.assignment)} if instruction else None),
        "ref_map": ref_map if ref_map is not None else [],
        "reason": reason,
    }


def render(
    instruction: Instruction,
    bag: list[BagItem],
    input_text: str,
    *,
    plan_step_id: str = "",
    declared: Declared = DECLARED_NONE,
    skeletons: dict[str, Skeleton] | None = None,
) -> dict:
    """结构指令 → 用户文本（§7.7 + §7.9 的唯一出口）。**确定性纯函数**。

    两道检查都过才出 `kind="text"`；任一不过 ⇒ `kind="reject"`（fail-closed），
    evidence 为空、不引用任何区间。
    """
    table = SKELETONS if skeletons is None else skeletons
    problems = check_structure(
        instruction, bag, input_text, declared=declared, skeletons=table)
    if problems:
        return _record(
            "reject", f"（拒答）{'；'.join(problems)}", [], plan_step_id,
            instruction=instruction, reason="；".join(problems))

    skeleton = table[instruction.skeleton_id]
    items = {b.ref: b for b in bag}
    parts: list[str] = []
    ref_map: list[dict] = []
    out = 0
    lit_seq = 0
    for tok in pattern_tokens(skeleton.pattern):
        if tok[0] == "slot":
            item = items[instruction.assignment[int(tok[1]) - 1]]
            ref_map.append({
                "unit_id": f"ref:{item.ref}",
                "cls": "content",
                "text": item.text,
                "ref": item.ref,
                "span": tuple(item.span),
                "out": (out, out + len(item.text)),
            })
            parts.append(item.text)
            out += len(item.text)
        else:
            _, kind, word = tok  # type: ignore[misc]
            ref_map.append({
                "unit_id": f"{skeleton.sid}#{lit_seq}",
                "cls": _cls(kind),
                "text": word,
                "ref": skeleton.sid,
                "span": None,
                "out": (out, out + len(word)),
            })
            parts.append(word)
            out += len(word)
            lit_seq += 1
    text = "".join(parts)

    evidence: list[dict] = []
    for entry in ref_map:
        if entry["cls"] == "content":
            item = items[entry["ref"]]
            evidence.append({
                "source": "span",
                "ref": item.ref,
                "text": item.text,
                "span": tuple(item.span),
                "candidate_id": item.candidate_id,
            })
        else:
            evidence.append({
                "source": "table",
                "unit_id": entry["unit_id"],
                "word": entry["text"],
                "cls": entry["cls"],
            })

    record = _record(
        "text", text, evidence, plan_step_id,
        instruction=instruction, ref_map=ref_map)
    problems = check_deref(record, bag, input_text, skeletons=table)
    if problems:
        return _record(
            "reject", f"（拒答）{'；'.join(problems)}", [], plan_step_id,
            instruction=instruction, reason="；".join(problems))
    return record


# ---- I3 · 解引用侧检查 -------------------------------------------------------

def check_deref(
    record: dict,
    bag: list[BagItem],
    input_text: str,
    *,
    skeletons: dict[str, Skeleton] | None = None,
) -> list[str]:
    """**解引用侧检查（I3）**：文本每个字符必须被引用映射覆盖，且每条映射都能解析。

    - **分区覆盖**：`ref_map` 各条的 `out` 区间必须恰好铺满 `[0, len(text))`
      ⇒ 渲染时多吐/漏吐一个字符都会被抓；
    - **内容词**：`text ↦ ref（结构引用 id）↦ span`，且 `input[span] == text`、
      ref 必须在指令指派里（回溯链不许断、不许编）；
    - **功能词**：`text ↦ 表项 id`（不是 span），与重新读骨架表得到的字面序列逐字一致。
    """
    table = SKELETONS if skeletons is None else skeletons
    probs: list[str] = []
    if record.get("kind") != "text":
        return [f"解引用检查只适用于 kind=text 的记录，当前 {record.get('kind')!r}"]
    text = record.get("text")
    ref_map = record.get("ref_map")
    instruction = record.get("instruction") or {}
    if not isinstance(text, str) or not isinstance(ref_map, list):
        return ["I3 违规：记录缺 text / ref_map（引用映射必须留存）"]
    sid = instruction.get("skeleton_id")
    assignment = instruction.get("assignment")
    skeleton = table.get(sid) if isinstance(sid, str) else None
    if skeleton is None or not isinstance(assignment, list):
        return [f"I3 违规：记录里的结构引用不可解析（skeleton_id={sid!r}）"]
    if len(assignment) != len(skeleton.slots):
        probs.append(
            f"I3 违规：记录的指派 {len(assignment)} 项与骨架 {sid} 的 "
            f"{len(skeleton.slots)} 个槽不等")
        return probs

    items = {b.ref: b for b in bag}
    # 1) 分区覆盖 + 字面自洽
    cursor = 0
    for i, entry in enumerate(ref_map):
        try:
            start, end = entry["out"]
        except (KeyError, TypeError, ValueError):
            probs.append(f"I3 违规：ref_map[{i}] 缺合法 out 区间")
            return probs
        if not (0 <= start <= end <= len(text)) or start != cursor:
            probs.append(
                f"I3 违规：ref_map[{i}] 区间 {entry['out']} 不覆盖文本位置 {cursor}"
                f"（多吐/漏吐内容 ⇒ 回溯链断）")
            return probs
        if text[start:end] != entry.get("text"):
            probs.append(f"I3 违规：ref_map[{i}] 文本 {entry.get('text')!r} 与实际 {text[start:end]!r} 不符")
        cursor = end
    if cursor != len(text):
        probs.append(f"I3 违规：ref_map 只覆盖到 {cursor}，文本长 {len(text)}（渲染时多吐了内容）")

    # 2) 逐条解析回结构单元
    expected_lits = [
        (f"{skeleton.sid}#{i}", _cls(kind), text)
        for i, (kind, text) in enumerate(_literal_seq(skeleton.pattern))
    ]
    lit_i = 0
    content_refs: list[object] = []
    for i, entry in enumerate(ref_map):
        if entry.get("cls") == "content":
            ref = entry.get("ref")
            content_refs.append(ref)
            if not isinstance(ref, int):
                probs.append(f"I3 违规：ref_map[{i}] 内容条目缺结构引用 id")
                continue
            if ref not in assignment:
                probs.append(f"I3 违规：ref_map[{i}] 内容条目 ref={ref} 不在指令指派里（编的内容词）")
                continue
            if entry.get("unit_id") != f"ref:{ref}":
                probs.append(
                    f"I3 违规：ref_map[{i}] 内容条目 unit_id={entry.get('unit_id')!r} "
                    f"与结构引用 ref={ref} 不一致（映射表键错了）")
                continue
            item = items.get(ref)
            if item is None:
                probs.append(f"I3 违规：ref_map[{i}] ref={ref} 不在袋里（回溯链断）")
                continue
            span = entry.get("span")
            if not isinstance(span, tuple) or len(span) != 2:
                probs.append(f"I3 违规：ref_map[{i}] 内容条目缺指针 span")
                continue
            start, end = span
            if not (0 <= start < end <= len(input_text)) or input_text[start:end] != item.text:
                probs.append(f"I3 违规：ref_map[{i}] span {span} 不是输入逐字子串")
            elif item.text != entry.get("text"):
                probs.append(f"I3 违规：ref_map[{i}] 文本 {entry.get('text')!r} 与袋块 {item.text!r} 不符")
        else:
            if lit_i >= len(expected_lits):
                probs.append(f"I3 违规：ref_map[{i}] 表项 {entry.get('unit_id')!r} 超出骨架字面表项数")
                continue
            want_id, want_cls, want_text = expected_lits[lit_i]
            lit_i += 1
            if (entry.get("unit_id"), entry.get("cls"), entry.get("text")) != (
                want_id, want_cls, want_text
            ):
                probs.append(
                    f"I3 违规：ref_map[{i}] 表项对不上（期望 {want_id}/{want_cls}={want_text!r}，"
                    f"实际 {entry.get('unit_id')}/{entry.get('cls')}={entry.get('text')!r}）")
            if entry.get("span") is not None:
                probs.append(f"I3 违规：功能词 {entry.get('unit_id')} 必须走表项 id，不是 span")
    if lit_i != len(expected_lits):
        probs.append(f"I3 违规：骨架字面表项 {len(expected_lits)} 条，记录里只出现 {lit_i} 条")
    if len(content_refs) != len(set(content_refs)):
        probs.append("I3 违规：同一袋块被映射多次（回溯链歧义）")
    if sorted(content_refs, key=str) != sorted(assignment, key=str):
        probs.append(
            f"I3 违规：内容单元与槽位指派不是一一对应"
            f"（文本内容 {sorted(content_refs, key=str)} vs 指派 {sorted(assignment, key=str)}）")
    return probs


def validate_skeleton_table(table: dict[str, Skeleton] | None = None) -> list[str]:
    """骨架表门禁（§7.7 表达力出口 = 人写新骨架，**过门禁**才进表）。

    查：id 唯一、规模上限、规则 A、槽位 schema。返回问题列表（空 = 通过）；
    builtin 表在 import 时就跑一遍 —— 带病的骨架进不了表。
    """
    table = SKELETONS if table is None else table
    probs: list[str] = []
    if len(table) > SKELETON_TABLE_LIMIT:
        probs.append(f"骨架表规模 {len(table)} 超过上限 {SKELETON_TABLE_LIMIT}")
    sids = [sk.sid for sk in _ALL_SKELETON_LIST]
    if len(set(sids)) != len(sids):
        probs.append(f"骨架 id 重复：{sids}")
    if set(table) != set(sids):
        probs.append("骨架表与手写表 _ALL_SKELETON_LIST（官方族 + CF 迁移族）不一致（表被改动过）")
    if set(SKELETON_OFFICIAL) & set(SKELETON_CF):
        probs.append(f"命名空间相交（V1 违例）：{sorted(set(SKELETON_OFFICIAL) & set(SKELETON_CF))}")
    for sk in table.values():
        probs += rule_a_problems(sk)
        probs += slot_schema_problems(sk)
        # 逻辑词/数字的对齐是**渲染时**按输入与意图卡判的（词面守卫），表内合法、不在此查。
    return probs


_validate_skeleton_table_problems = validate_skeleton_table()
if _validate_skeleton_table_problems:
    raise ValueError(
        "骨架表不合法（fail-closed，带病不进表）：" + "；".join(_validate_skeleton_table_problems))
