#!/usr/bin/env python3
"""P7 标签与出口构造（口径 = PREREG §2/§3，跑前写死，不事后增补）。

三个函数：
  · `label_full(s)`      —— **原文**上的标签定义（有句末标点才走到标点支）；
  · `label_nopunct(s)`   —— **去标点**版定义（RHO/QSET/YN/IMP/EXC/陈述 六支顺序），
                            既是 `max_naive` 的 R6，也是口语出口的标签源；
  · `find_trigger(s)`    —— 封闭触发词表里第一次命中的 span（在**该出口的输入**上算）。

出口（PREREG §3）：`keep` / `maskfinal` / `mask` 同一行三种输入；`colloq` 独立池。
遮蔽集 = `render.PUNCT ∪ FINAL_PUNCT`（复用 `src/dtseek/tasks/render.py` 的 `PUNCT`）。
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks import render  # noqa: E402  （只读 import，不改 src/）

# ---- 模式表（封闭） ---------------------------------------------------------
MODES: tuple[str, ...] = ("陈述", "是非疑问", "特指疑问", "祈使", "感叹", "反问")
MODE_ID: dict[str, int] = {m: i for i, m in enumerate(MODES)}
N_MODE = len(MODES)
MAJ_MODE = MODE_ID["陈述"]          # 训练侧的"全局多数"占位（实际以 train 计数为准）

# ---- 标点 ------------------------------------------------------------------
FINAL_PUNCT = frozenset("。．.？?！!")
#: 遮蔽集：`render.PUNCT`（标点是纯形式，render.py 的口径）+ 它没收的句末符号
MASK_SET: frozenset[str] = frozenset(render.PUNCT) | FINAL_PUNCT

# ---- 词表（跑前写死） -------------------------------------------------------
RHO: tuple[str, ...] = ("难道说", "难道", "谁说", "不是吗", "何尝", "岂")
QSET: tuple[str, ...] = (
    "什么", "啥", "谁", "哪儿", "哪里", "哪个", "哪些", "哪种", "怎么", "咋",
    "怎样", "如何", "为什么", "为啥", "为何", "多少", "几点", "几个", "几时",
    "多久", "几号", "何",
)
YN: tuple[str, ...] = (
    "是不是", "能不能", "会不会", "有没有", "对不对", "要不要", "行不行",
    "好不好", "可不可以", "好吗", "吗",
)
IMP: tuple[str, ...] = (
    "不要", "不准", "不许", "马上", "赶紧", "快点", "给我", "闭嘴", "停下",
    "等一下", "记得", "必须", "注意", "小心", "住手", "站起来", "走开", "加油",
    "请", "别", "滚", "试试",
)
EXC: tuple[str, ...] = (
    "极了", "绝了", "恭喜", "无语", "崩了", "服了", "疯了", "厉害", "太", "真",
    "好", "棒", "哇", "唉", "哎", "咦", "啊", "呀", "啦",
)

#: 触发词封闭表（PREREG §2：按此**固定顺序**取第一次命中；长词在前）
TRIGGER_ORDER: tuple[str, ...] = (
    tuple(sorted(RHO, key=lambda w: (-len(w), w)))
    + tuple(sorted(QSET, key=lambda w: (-len(w), w)))
    + tuple(sorted(YN, key=lambda w: (-len(w), w)))
    + tuple(sorted(IMP, key=lambda w: (-len(w), w)))
)
TRIGGER_SET: frozenset[str] = frozenset(TRIGGER_ORDER)


def _hit(s: str, table: tuple[str, ...]) -> str | None:
    for w in table:
        if w in s:
            return w
    return None


def _rho_hit(s: str) -> str | None:
    w = _hit(s, RHO)
    if w:
        return w
    if "不是" in s and "吗" in s:      # 「不是…吗」
        return "不是…吗"
    return None


def final_char(s: str) -> str:
    t = s.rstrip()
    return t[-1] if t else ""


def label_full(s: str) -> str | None:
    """原文上的标签定义（PREREG §2 优先级表）。返回 None = 丢弃。"""
    f = final_char(s)
    if _rho_hit(s):
        return "反问"
    if f in ("？", "?"):
        return "特指疑问" if _hit(s, QSET) else "是非疑问"
    if f in ("！", "!"):
        return "祈使" if _hit(s, IMP) else "感叹"
    if f in ("。", "．", "."):
        return "陈述"
    return None


def label_nopunct(s: str) -> str:
    """去标点版定义：RHO → QSET → YN → IMP → EXC → 陈述（六支，PREREG §2/§3）。"""
    if _rho_hit(s):
        return "反问"
    if _hit(s, QSET):
        return "特指疑问"
    if _hit(s, YN):
        return "是非疑问"
    if _hit(s, IMP):
        return "祈使"
    if _hit(s, EXC):
        return "感叹"
    return "陈述"


def label_defn_on(exit_input: str) -> str:
    """R6：标签定义函数在**该出口输入**上能走哪支就走哪支（零拟合）。"""
    return label_full(exit_input) if final_char(exit_input) in FINAL_PUNCT else label_nopunct(exit_input)


def find_trigger(s: str) -> tuple[int, int] | None:
    """封闭触发词表里按 `TRIGGER_ORDER` **第一个命中的词**的 span；无命中 = None。

    表序即优先级（反问 > 疑问 > 是非 > 祈使），命中即返回，不比较位置先后。
    """
    for w in TRIGGER_ORDER:
        i = s.find(w)
        if i >= 0:
            return (i, i + len(w))
    return None


# ---- 出口 ------------------------------------------------------------------
def exit_keep(text: str) -> str:
    return text


def exit_maskfinal(text: str) -> str:
    t = text.rstrip()
    if t and t[-1] in FINAL_PUNCT:
        return t[:-1] + text[len(t):]
    return text


def exit_mask(text: str) -> str:
    return "".join(ch for ch in text if ch not in MASK_SET)


EXITS = ("keep", "maskfinal", "mask")
EXIT_FN = {"keep": exit_keep, "maskfinal": exit_maskfinal, "mask": exit_mask,
           "colloq": lambda s: s, "colloq_mask": exit_mask}


def make_exits(text: str) -> dict[str, str]:
    return {k: EXIT_FN[k](text) for k in ("keep", "maskfinal", "mask")}


def assert_masked(s: str, where: str) -> None:
    bad = sorted({c for c in s if c in MASK_SET})
    if bad:
        raise AssertionError(f"[sentence_mode] {where} 遮蔽未生效，残留标点 {bad!r}：{s!r}")


if __name__ == "__main__":
    for t in ("今天天气真好。", "你吃了吗？", "这是什么东西？", "快给我出去！", "太好了！", "难道这不对吗？",
              "你吃了吗", "这是什么"):
        print(f"{t!r:30s} full={label_full(t)} nopunct={label_nopunct(t)} "
              f"trig={find_trigger(t)} mask={exit_mask(t)!r}")
