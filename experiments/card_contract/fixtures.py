"""P13 夹具：官方 16 条骨架 + 迁移族 13 条的**参考渲染**（纯构造，不判定）。

夹具袋覆盖官方表全部槽位类型（名·施事 / 名·受事 / 名·时 / 动 / 形），
并让**每一条**官方骨架都能渲染出 `kind="text"`（非空测试：若某条恒拒，这里会立刻暴露）。
迁移族 13 条用同一袋（`direction_safe`，真卡不产题元 ⇒ 名槽不比题元）。
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in ("src",):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

from dtseek.tasks import render as R  # noqa: E402

#: 夹具输入：字面里出现 因为 / 要 / 但 / 所以 四个逻辑词（词面守卫要求逐字有据）。
FIX_INPUT = "因为用户说加载太慢，但我们下周要修，所以方案太慢。"

# 因0 为1 用2 户3 说4 加5 载6 太7 慢8 ，9 但10 我11 们12 下13 周14 要15 修16 ，17
# 所18 以19 方20 案21 太22 慢23 。24
BAG: tuple[R.BagItem, ...] = (
    R.BagItem(1, "用户", (2, 4), "名", "施事", "f1", True),
    R.BagItem(2, "说", (4, 5), "动", None, "f2", True),
    R.BagItem(3, "加载", (5, 7), "动", None, "f3", True),
    R.BagItem(4, "太慢", (7, 9), "形", None, "f4", True),
    R.BagItem(5, "我们", (11, 13), "名", "施事", "f5", True),
    R.BagItem(6, "下周", (13, 15), "名", "时", "f6", True),
    R.BagItem(7, "修", (16, 17), "动", None, "f7", True),
    R.BagItem(8, "要", (15, 16), "动", None, "f8", True),
    R.BagItem(9, "用户", (2, 4), "名", "受事", "f9", True),
    R.BagItem(10, "方案", (20, 22), "名", "施事", "f10", True),
    R.BagItem(11, "太慢", (22, 24), "形", None, "f11", True),
)

#: 官方 16 条的参考指派（跑前写死；每个都必须渲染成 `kind="text"`）。
OFFICIAL_ASSIGN: dict[str, tuple[int, ...]] = {
    "S01": (1, 2, 9),
    "S02": (1, 2, 9),
    "S03": (1, 2, 9),
    "S04": (1, 2),
    "S05": (1, 2, 5, 3),
    "S06": (1, 2),
    "S07": (1, 9, 2),
    "S08": (1, 9, 2),
    "S09": (1, 2),
    "S10": (1, 2, 3, 4),
    "S11": (1, 2, 3, 4, 5, 6, 7),
    "S12": (3, 4, 5, 6, 7),
    "S13": (3, 4, 5, 6, 7),
    "R01": (1, 2, 4),
    "R02": (1, 2),
    "R03": (1, 2, 3, 4),
}

#: 迁移族 13 条的参考指派（同袋；`direction_safe` ⇒ 指派随原文 span 升序）。
CF_ASSIGN: dict[str, tuple[int, ...]] = {
    "CF01": (1,),
    "CF02": (1, 4),
    "CF03": (1, 4),
    "CF06": (1, 5),
    "CF07": (1, 5),
    "CF08": (1, 5, 10),
    "CF09": (1, 4, 5, 11),
    "CF11": (1,),
    "CF18": (1, 4, 5, 11),
    "CF19": (1, 4, 5, 11),
    "CF20": (1, 4, 5, 11),
    "CF21": (5, 7),
    "CF25": (1, 4),
}


def render_fix(sids: dict[str, tuple[int, ...]],
               table: dict[str, R.Skeleton] | None = None) -> dict[str, dict]:
    """按参考指派渲染一批骨架，返回 `{sid: {"kind","text"|"reason", ...}}`。"""
    out: dict[str, dict] = {}
    for sid, assign in sids.items():
        rec = R.render(R.Instruction(sid, assign), list(BAG), FIX_INPUT,
                       plan_step_id="fix:1", skeletons=table)
        out[sid] = {
            "kind": rec.get("kind"),
            "text": rec.get("text"),
            "reason": rec.get("reason"),
            "assignment": list(assign),
            "pattern": (table or R.SKELETONS)[sid].pattern if sid in (table or R.SKELETONS) else None,
            "signature": [ [s.pos, s.theta] for s in (table or R.SKELETONS)[sid].slots ]
                         if sid in (table or R.SKELETONS) else None,
        }
    return out


def official_fix() -> dict[str, dict]:
    return render_fix(OFFICIAL_ASSIGN)


def cf_fix() -> dict[str, dict]:
    return render_fix(CF_ASSIGN)
