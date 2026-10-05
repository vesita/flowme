"""候选选择卡族（reply_pick / cloze_fill）共用的输入框架与符号纪律。

族形状（dev-notes/13 §1）：

    {头部分}|1）候选1|2）候选2|3）候选3|4）候选4

- **类别头**回答「选哪个候选」：`classes` 下标 = 候选编号，`classes[0]` = 无合适候选；
- **指针头**给出被选中候选在原文上的 0-based 半开区间 —— 下游 `chain/filter`
  组合算子消费的就是这个锚点（dev-notes/13 §7），所以锚点落在**候选正文**上，
  不含 `N）` 标号、也不含空位符号。

三条 fail-closed 不变量（dev-notes/13 §3、dev-notes/09 §2 的 `：`+`"` 事故）：

  1. **标记符号必须是词表内的确定 id**：逐字编码恰好 1 个 token、不是 UNK、
     不是特殊 token、标记之间 id 互不相同。实测本项目分词器里
     `｜`(U+FF5C)、`＿`(U+FF3F)、`１`(全角数字) 全部塌到 UNK=129 ——
     所以设计稿里的 `｜` / `＿＿` 换成 `|`(124) / `__`(95×2)，
     换形只改渲染不改协议；
  2. **内容纯净**：问句 / 句子 / 候选正文里不许出现标记字符（`|` `）` `_`），
     否则候选结构在字面上存在第二种读法，锚点也会指向标记而不是候选；
  3. **长度门禁**：整段文本 ≤ `max_len − 8` —— 引擎 `window` 策略的整段阈值就是
     `max_len - 8`（见 `engine.MultiTaskEngine.predict`），超了会在 `？` 处被切开，
     候选结构在句边界丢失（dev-notes/13 §2 的原题：推理怎么切必须和训练怎么喂一致）。
"""
from __future__ import annotations

import re
from functools import cache, lru_cache

#: 候选列表分隔符。设计稿写的是全角 `｜`，实测它是 UNK，故用 ASCII `|`（id 124）。
SEPARATOR = "|"
#: 标号收尾括号：`1）`。`）`= id 156（系统词区）。
SLOT_CLOSE = "）"
#: 候选编号（K=4）。
SLOTS: tuple[str, ...] = ("1", "2", "3", "4")
#: 完形填空的空位符号。设计稿写的是 `＿＿`，实测是 UNK，故用 ASCII `__`（id 95）。
BLANK = "__"
#: 择优回复的头部分。
QUESTION_HEAD = "问："

#: 标记字符闭集：这些字符**只允许**出现在框架标记里，不得出现在内容中。
MARKER_CHARS: tuple[str, ...] = ("|", "）", "_")
#: 框架里所有承载语义的标记字符（词表 id 校验的对象）。
FRAME_TOKENS: tuple[str, ...] = (QUESTION_HEAD[0], QUESTION_HEAD[1], SEPARATOR,
                                 SLOT_CLOSE, BLANK[0], *SLOTS)

#: 单个标号段（`|1）`）的字符数 = 1 + 1 + 1。
SLOT_OVERHEAD = len(SEPARATOR) + 1 + len(SLOT_CLOSE)
#: 整个候选列表的固定字符数（4 个标号段）。
LIST_OVERHEAD = SLOT_OVERHEAD * len(SLOTS)

_CJK_RE = re.compile(r"[一-龥]")
_CODE_SYMBOLS = frozenset("{}[]<>;$#\\`|~^=+_*/")
_ENGLISH_WORDS_RE = re.compile(r"[a-zA-Z]{2,}")


def style_problems(s: str) -> list[str]:
    """语体门禁：中文汉字 ≥70%、ASCII ≤15%、无代码符号、无英文单词。

    与 negation 卡同口径 —— 正例与负例必须过**同一道**门禁，否则「正例是通顺中文、
    负例是英文代码」会变成语体捷径（dev-notes/13 §6.3，本项目已踩过两次）。
    返回违规描述列表，空列表 = 通过。
    """
    problems: list[str] = []
    length = len(s)
    if length == 0:
        return ["内容为空"]
    cjk_ratio = len(_CJK_RE.findall(s)) / length
    if cjk_ratio < 0.70:
        problems.append(f"中文汉字占比过低 ({cjk_ratio * 100:.1f}% < 70%)")
    ascii_ratio = sum(1 for c in s if c.isascii()) / length
    if ascii_ratio > 0.15:
        problems.append(f"ASCII 字符占比过高 ({ascii_ratio * 100:.1f}% > 15%)")
    code_hits = sorted({c for c in s if c in _CODE_SYMBOLS})
    if code_hits:
        problems.append(f"含代码符号: {''.join(code_hits)}")
    en_matches = sorted(set(_ENGLISH_WORDS_RE.findall(s)))
    if en_matches:
        problems.append(f"含英文单词: {en_matches}")
    return problems


@lru_cache(maxsize=1)
def _tokenizer():
    from nano_char_tokenizer import NanoCharTokenizer

    return NanoCharTokenizer()


@cache
def char_in_vocab(ch: str) -> bool:
    """单字符是否落在词表里。

    词表外字符逐字编码会塌成同一个 UNK(129)：位置几何仍对（1 字 = 1 token），
    但内容里的字符彼此不可分辨，按字符对齐的形态判断会退化。
    所以内容里出现词表外字符按**内容不纯净**处理，构造期直接丢样本。
    """
    tok = _tokenizer()
    tid = tok.token_to_id(ch)
    return tid is not None and tid != tok.unk_token_id


def content_problems(*texts: str) -> list[str]:
    """内容纯净检查：标记字符污染或词表外字符（会塌成 UNK）都返回问题描述。"""
    bad: list[str] = []
    for t in texts:
        hits = sorted({c for c in t if c in MARKER_CHARS})
        if hits:
            bad.append(f"内容 {t!r} 含标记字符 {''.join(hits)}")
        unk = sorted({c for c in t if not char_in_vocab(c)})
        if unk:
            bad.append(f"内容 {t!r} 含词表外字符（塌成 UNK）: {''.join(unk)}")
    return bad


def frame_token_problems() -> list[str]:
    """标记符号的词表检查：单 token、非 UNK/特殊 token、标记之间 id 不同。

    这是 dev-notes/13 §3 第 1 条的唯一实现点（测试只调用它，不另抄一份逻辑）。
    """
    from nano_char_tokenizer import NanoCharTokenizer

    tok = NanoCharTokenizer()
    specials = {"unk": tok.unk_token_id, "pad": tok.pad_token_id, "bos": tok.bos_token_id,
                "eos": tok.eos_token_id, "sep": tok.sep_token_id}
    problems: list[str] = []
    seen: dict[int, str] = {}
    for ch in FRAME_TOKENS:
        ids = tok.encode(ch, max_length=8, padding=False)["input_ids"]
        if len(ids) != 1:
            problems.append(f"标记字符 {ch!r} 编码成 {len(ids)} 个 token（期望 1 个）：{ids}")
            continue
        tid = ids[0]
        for name, sid in specials.items():
            if tid == sid:
                problems.append(f"标记字符 {ch!r} 落在 {name} token（id={tid}）上")
        if tid in seen and seen[tid] != ch:
            problems.append(f"标记字符 {ch!r} 与 {seen[tid]!r} 发生 id 碰撞（id={tid}）")
        seen[tid] = ch
    return problems


def validate_markers() -> None:
    """标记纪律 fail-closed：任一条不满足就响亮抛错，绝不带病出数据。"""
    problems = frame_token_problems()
    if problems:
        raise ValueError("候选选择卡标记符号校验失败：\n  - " + "\n  - ".join(problems))


def render_choice_list(candidates: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """把 4 个候选渲染成列表段，返回 `(段文本, 各候选的半开区间列表)`。

    区间相对于**段文本**起点；调用方把它加上头部长度才是全文坐标。
    fail-closed：候选数量、重复候选、标记字符污染都在这里拦。
    """
    if len(candidates) != len(SLOTS):
        raise ValueError(f"候选数量必须是 {len(SLOTS)}，收到 {len(candidates)}")
    if len(set(candidates)) != len(candidates):
        raise ValueError(f"候选有重复：{candidates}")
    problems = content_problems(*candidates)
    if problems:
        raise ValueError("候选列表校验失败：\n  - " + "\n  - ".join(problems))
    parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for slot, cand in zip(SLOTS, candidates):
        prefix = f"{SEPARATOR}{slot}{SLOT_CLOSE}"
        parts.append(prefix)
        cursor += len(prefix)
        parts.append(cand)
        spans.append((cursor, cursor + len(cand)))
        cursor += len(cand)
    return "".join(parts), spans


def check_text_length(text: str, limit: int) -> None:
    """长度门禁：整段 ≤ `max_len - 8`（引擎 window 分段阈值），超了响亮失败。"""
    if len(text) > limit:
        raise ValueError(
            f"样本长度 {len(text)} 超过窗口阈值 {limit}（= max_len - 8）：{text!r}\n"
            "  超限会被引擎按标点切段，候选结构在句边界丢失 —— "
            "应当在数据构造侧收紧问句/候选长度，而不是截断。")
