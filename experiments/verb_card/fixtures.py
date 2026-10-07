"""P25 夹具：**手写固定用例**（不抽样）—— 每条动槽/否数槽骨架配一句输入 + 人工判定的
gold 动词 + 名槽题元角色块。

三个字段都是**跑前写死**（PREREG.md 里原样抄了这张表）：

- `text`        输入原文；
- `gold`        **人工判定的动词切片**（中文说话人视角）—— 用于算动词卡与
                平凡地板的 gold 一致率（D4 构念效度），与门禁无关；
- `roles`       名槽题元**角色块**：`{候选面 → 施事/受事/时}`，
                与 `card_flow` 的 A1/A3/A14 同一手法（**题元不是本单元产出的**，
                出处写在报告里；不给角色块的骨架 = 只复述子集）。

⚠️ 题元只来自这里 ⇒ 报告必须把「无角色块 / 有角色块」两层分开报。
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Fixture:
    fid: str
    text: str
    gold: tuple[str, ...] = ()
    roles: dict[str, str] | None = None

    @property
    def role_map(self) -> dict[str, str]:
        return dict(self.roles or {})


def _f(fid: str, text: str, gold: tuple[str, ...] = (),
       roles: dict[str, str] | None = None) -> Fixture:
    return Fixture(fid=fid, text=text, gold=gold, roles=roles)


#: 夹具表（跑前写死；顺序即报告顺序）。
FIXTURES: tuple[Fixture, ...] = (
    _f("F01", "我们讨论你们", ("讨论",), {"我们": "施事", "你们": "受事"}),
    _f("F02", "我学习", ("学习",), {"我": "施事"}),
    _f("F03", "我学习，你讨论", ("学习", "讨论"), {"我": "施事", "你": "施事"}),
    _f("F04", "我们学习", ("学习",), {"我": "施事"}),
    _f("F05", "我们和你们讨论", ("讨论",), {"我们": "施事", "你们": "受事"}),
    _f("F06", "我们与你们讨论", ("讨论",), {"我们": "施事", "你们": "受事"}),
    _f("F07", "我们讨论研究开心", ("讨论", "研究"), {"我们": "施事"}),
    _f("F08", "我们讨论研究开心，我下周要学习",
       ("讨论", "研究", "学习"), {"我们": "施事", "我": "施事", "下周": "时"}),
    _f("F09", "讨论研究开心，我下周要学习",
       ("讨论", "研究", "学习"), {"我": "施事", "下周": "时"}),
    _f("F10", "因为讨论研究开心，我下周要学习",
       ("讨论", "研究", "学习"), {"我": "施事", "下周": "时"}),
    # —— 只复述子集（**无角色块**：名槽不标题元）——
    _f("F11", "我学习开心", ("学习",)),
    _f("F12", "我学习", ("学习",)),
    _f("F13", "我们讨论研究开心", ("讨论", "研究")),
    _f("F14", "我们要学习", ("学习",), {"我们": "施事"}),   # 给 CFS21（该表声明了题元）
    _f("F15", "我们要学习", ("学习",)),                     # 官方 CF21：只复述子集，无角色块
    # —— 否 / 数 槽（W2 的 9 条）——
    _f("F20", "我不开心", (), {"我": "施事"}),
    _f("F21", "我不开心，你难过", ()),
    _f("F22", "我不学习", ("学习",), {"我": "施事"}),
    _f("F23", "我不讨论你", ("讨论",), {"我": "施事", "你": "受事"}),
    _f("F24", "我18开心", ()),
)

BY_FID: dict[str, Fixture] = {f.fid: f for f in FIXTURES}
assert len(BY_FID) == len(FIXTURES), "夹具 fid 重复"

#: 骨架 → 夹具。id 命名空间：`S##`/`R##`/`CF##` = 官方生成链表（`tasks/render.py`）；
#: `CFS##` = `experiments/card_flow/skeletons.py` 的同号骨架（前缀隔离，避免同 id 异签名）。
SKELETON_FIXTURE: dict[str, str] = {
    # —— 官方族 16 条（全部含动槽）——
    "S01": "F01", "S02": "F01", "S03": "F01", "S04": "F02", "S05": "F03",
    "S06": "F04", "S07": "F05", "S08": "F06", "S09": "F02", "S10": "F07",
    "S11": "F08", "S12": "F09", "S13": "F10",
    "R01": "F11", "R02": "F12", "R03": "F13",
    # —— 迁移族里唯一含动槽的 1 条 ——
    "CF21": "F15",
    # —— card_flow 表 25 条里的 7 条动槽骨架（前提口径「16 条需动槽」的动槽部分）——
    "CFS12": "F02", "CFS13": "F01", "CFS14": "F01", "CFS15": "F22",
    "CFS16": "F23", "CFS21": "F14", "CFS22": "F22",
    # —— card_flow 表里的 9 条 否/数 槽骨架（W2）——
    "CFS04": "F20", "CFS05": "F20", "CFS10": "F21", "CFS26": "F20",
    "CFS23": "F24", "CFS24": "F24",
}

#: W2 的 9 条（`否`/`数` 槽）—— 报告按这个顺序列。
W2_SKELETONS: tuple[str, ...] = (
    "CFS04", "CFS05", "CFS10", "CFS15", "CFS16", "CFS22", "CFS26", "CFS23", "CFS24")

#: 前提口径的 7 条动槽骨架（card_flow 表）。
CF_VERB_SKELETONS: tuple[str, ...] = (
    "CFS12", "CFS13", "CFS14", "CFS15", "CFS16", "CFS21", "CFS22")


def gold_spans(fix: Fixture) -> set[tuple[int, int]]:
    """gold 动词切片：`fix.gold` 里每个词在 `fix.text` 中的**全部**出现区间。"""
    out: set[tuple[int, int]] = set()
    for w in fix.gold:
        start = 0
        while True:
            i = fix.text.find(w, start)
            if i < 0:
                break
            out.add((i, i + len(w)))
            start = i + 1
    return out
