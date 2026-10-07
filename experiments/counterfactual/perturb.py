"""P10 扰动族 + 随机对照（表与协议全部在 PREREG §3/§4 写死，此处只是实现）。

- 扰动大小 `lev(X, X')` = 字符级 Levenshtein；族内按 (lev, 首个不同位置, X') 升序去重。
- 随机对照用拒绝采样打到**完全相同的 lev**，打不到记「不可达」。
"""
from __future__ import annotations

import hashlib
import random
import re

PUNCT = set("，。！？!?；;、,")
SEG_SPLIT = re.compile(r"[，。！？!?；;、,]")

# ---- F1 换功能词（PREREG §3 逐字；`[修订1]` 一节记录了扩充前后的表） ----------
FW = [
    ("然而", "之前"), ("但是", "之后"), ("虽然", "如果"), ("如果", "虽然"),
    ("而且", "然后"), ("不过", "后来"), ("于是", "比如"), ("同时", "但是"),
    ("但是", "而且"), ("尽管", "比如"), ("并且", "之后"), ("却", "就"),
    ("否则", "比如"), ("要么", "比如"),
    # [修订1] 覆盖率扩充（纯枚举自检决定，未使用任何模型输出）
    ("也", "就"), ("就", "也"), ("还", "再"), ("再", "还"), ("但", "而"),
    ("而", "但"), ("才", "就"), ("又", "再"), ("更", "还"), ("却", "倒"),
    ("并且", "而且"), ("而且", "并且"), ("不过", "只是"), ("如果", "只有"),
    ("虽然", "即使"), ("除非", "如果"),
]
# ---- F3 改否定 ---------------------------------------------------------------
NEG_MARKERS = ["没有", "不是", "不会", "不能", "不要", "不", "没"]
NEG_INSERT = ["不", "没"]
# ---- F4 改数量/量词 -----------------------------------------------------------
QU = [
    ("全部", "有些"), ("所有", "有些"), ("一切", "有些"), ("都", "有些"),
    ("全", "有些"), ("总是", "有时"), ("一直", "偶尔"), ("一定", "也许"),
    ("很多", "有点"), ("太", "有点"), ("最", "比较"),
    ("一", "两"), ("两", "一"), ("几", "一些"),
    # [修订1] 覆盖率扩充（同上）
    ("很", "比较"), ("比较", "很"), ("真", "很"), ("挺", "很"), ("半", "一"),
    ("多", "少"), ("少", "多"), ("稍微", "有点"), ("有点", "稍微"),
]
# ---- F5 换连接词（方向反转） ---------------------------------------------------
SW = [
    ("因为", "所以"), ("所以", "因为"), ("由于", "因此"),
    ("因此", "由于"), ("因为", "因此"), ("之所以", "所以"),
    # [修订1] 覆盖率扩充（时间方向反转，同为连接词方向反转）
    ("之前", "之后"), ("之后", "之前"), ("以前", "以后"), ("以后", "以前"),
    ("之前", "以后"), ("以后", "之前"),
]
FAMILIES = ("F1", "F2", "F3", "F4", "F5")
FAMILY_NAME = {
    "F1": "换功能词", "F2": "换槽序", "F3": "改否定",
    "F4": "改数量/量词", "F5": "换连接词(方向反转)",
}
FAMILY_TABLE = {"F1": "FW", "F2": "(片段两两交换)", "F3": "NEG_MARKERS+NEG_INSERT",
                "F4": "QU", "F5": "SW"}

# ---- 随机对照的字/词库（PREREG §4；固定不随运行变） -----------------------------
CHAR_BANK = (
    "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得里后自以会家可下而过天去能对小多然于心学么之都好看起发当没成只如事把还用第样道想作种开美总从无情己面最女但现前些所同日手又行意动方期它头经长儿回位分爱老因很给名法间斯知世什两次使身者被高已亲其进此话常与活正感"
)

WORD_BANK = [
    "我", "你", "他", "她", "它", "我们", "你们", "他们", "这", "那",
    "这个", "那个", "是", "在", "有", "和", "与", "但", "很", "都",
    "也", "就", "还", "把", "被", "给", "让", "使", "对", "从",
    "到", "时候", "现在", "今天", "明天", "已经", "正在", "可以", "应该", "必须",
    "一个", "一些", "很多", "东西", "事情", "地方", "时间", "朋友", "大家", "自己",
    "怎么", "什么", "为什么", "其实", "可能", "大概", "真是", "有点", "稍微", "完全",
    "根本", "一直", "总是", "忽然", "仍然", "反正", "本来", "原来", "后来", "同时",
    "当然", "比如", "例如", "不过", "而且", "然后", "之前", "现在", "最近", "每天",
]


# ---- Levenshtein -------------------------------------------------------------

def lev(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _first_diff(a: str, b: str) -> int:
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


# ---- 分段（片段 / 标点 交替，标点原位不动） -------------------------------------

def segments(text: str) -> list[tuple[str, str]]:
    """返回 [(kind, piece)]，kind ∈ {'s' 片段, 'p' 标点}，拼回 == text。"""
    out: list[tuple[str, str]] = []
    buf: list[str] = []
    for ch in text:
        if ch in PUNCT:
            if buf:
                out.append(("s", "".join(buf)))
                buf = []
            out.append(("p", ch))
        else:
            buf.append(ch)
    if buf:
        out.append(("s", "".join(buf)))
    return out


def _uniq_sort(text: str, cands: set[str]) -> list[str]:
    good = {c for c in cands if c != text}
    return sorted(good, key=lambda c: (lev(text, c), _first_diff(text, c), c))


def _replace_one(text: str, table: list[tuple[str, str]]) -> set[str]:
    out: set[str] = set()
    for src, dst in table:
        start = 0
        while True:
            i = text.find(src, start)
            if i < 0:
                break
            out.add(text[:i] + dst + text[i + len(src):])
            start = i + 1
    return out


# ---- 五个族的候选枚举 -----------------------------------------------------------

def family_candidates(text: str, family: str) -> list[str]:
    segs = segments(text)
    s_idx = [k for k, (kind, _) in enumerate(segs) if kind == "s"]
    cands: set[str] = set()

    if family == "F1":
        cands = _replace_one(text, FW)

    elif family == "F2":
        for a in range(len(s_idx)):
            for b in range(a + 1, len(s_idx)):
                ia, ib = s_idx[a], s_idx[b]
                if segs[ia][1] == segs[ib][1]:
                    continue                      # 交换后 X' == X，断言层还会再挡一次
                new = list(segs)
                new[ia], new[ib] = new[ib], new[ia]
                cands.add("".join(p for _, p in new))

    elif family == "F3":
        for m in NEG_MARKERS:                     # 删一处
            start = 0
            while True:
                i = text.find(m, start)
                if i < 0:
                    break
                cands.add(text[:i] + text[i + len(m):])
                start = i + 1
        for k in s_idx:                           # 片段起始插入
            off = sum(len(p) for _, p in segs[:k])
            for w in NEG_INSERT:
                cands.add(text[:off] + w + text[off:])

    elif family == "F4":
        cands = _replace_one(text, QU)

    elif family == "F5":
        cands = _replace_one(text, SW)

    else:
        raise ValueError(family)
    return _uniq_sort(text, cands)


# ---- 随机对照（PREREG §4） ------------------------------------------------------

def _rng(seed0: int, idx: int, family: str, mode: str, draw: int) -> random.Random:
    key = f"{seed0}|{idx}|{family}|{mode}|{draw}".encode()
    h = hashlib.sha1(key).hexdigest()
    return random.Random(int(h[:16], 16))


def _rand_char_exact(text: str, size: int, rng: random.Random, tries: int = 400) -> str | None:
    """随机换字符/插字符，命中 lev == size 且 != text；打不到返回 None。"""
    n = len(text)
    if n == 0:
        return None
    for _ in range(tries):
        chars = list(text)
        n_sub = min(size, n)
        pos = rng.sample(range(n), n_sub)
        for p in pos:
            choices = [c for c in CHAR_BANK if c != chars[p]]
            if not choices:
                choices = [chr(0x4E00 + rng.randrange(20992))]
            chars[p] = rng.choice(choices)
        cur = "".join(chars)
        rest = size - n_sub
        if rest > 0:                              # 补插字符，凑够 size
            for _k in range(rest):
                p = rng.randrange(len(cur) + 1)
                cur = cur[:p] + rng.choice(CHAR_BANK) + cur[p:]
        if cur != text and lev(text, cur) == size:
            return cur
    return None


def _rand_word_exact(text: str, size: int, rng: random.Random, tries: int = 400) -> str | None:
    """随机换词（1..3 字区域替换成词库词，允许变长），命中 lev == size。"""
    n = len(text)
    if n == 0:
        return None
    for _ in range(tries):
        cur = text
        touched: list[tuple[int, int]] = []
        for _step in range(12):
            if lev(text, cur) == size and cur != text:
                return cur
            if lev(text, cur) > size:
                break
            L = rng.choice((1, 2, 3))
            for _retry in range(20):
                i = rng.randrange(n)
                j = min(len(cur), i + L)
                if j <= i:
                    continue
                if any(not (j <= a or i >= b) for a, b in touched):
                    continue                      # 区域不重叠
                w = rng.choice(WORD_BANK)
                if cur[i:j] == w:
                    continue
                touched.append((i, j))
                cur = cur[:i] + w + cur[j:]
                break
            else:
                break
        if cur != text and lev(text, cur) == size:
            return cur
    return None


def random_perturb(text: str, size: int, seed0: int, idx: int, family: str,
                   mode: str, draw: int) -> str | None:
    rng = _rng(seed0, idx, family, mode, draw)
    if mode == "R1":
        return _rand_char_exact(text, size, rng)
    return _rand_word_exact(text, size, rng)


__all__ = [
    "FAMILIES", "FAMILY_NAME", "FAMILY_TABLE", "FW", "NEG_MARKERS", "NEG_INSERT",
    "QU", "SW", "CHAR_BANK", "WORD_BANK", "lev", "segments", "family_candidates",
    "random_perturb",
]
