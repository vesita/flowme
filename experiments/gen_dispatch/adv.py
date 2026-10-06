"""对抗组 A1–A14（PREREG §6 写死的预期，逐条实测）。

A1–A12 直接打**官方重整/解引用层**（`dtseek.tasks.render`）—— 我的生成通道就是靠它守门，
所以对抗必须在这一层过；A13–A14 打**通道开关的入口**（type / 空输入）。
每条同时报「官方层怎么答」与「经 `rules.seal` 归一后的记录是否符合拒答契约」。
"""
from __future__ import annotations

from typing import Any

from dtseek.tasks import render as R

from rules import contract_problems

__all__ = ["CASES", "run_cases"]


def _bag(text: str, spans: list[tuple[str, int, int, str, str | None]]) -> list[R.BagItem]:
    """按 (surface, span, pos, theta) 造袋；`screened` 一律 True（零信息新增的门）。"""
    bag = []
    for i, (surf, s, e, pos, theta) in enumerate(spans):
        bag.append(R.BagItem(ref=i, text=surf, span=(s, e), pos=pos, theta=theta,
                             candidate_id=f"f{i}", screened=True))
    return bag


def _sealed(render_out: dict, psid: str = "step:0") -> dict:
    """官方层的 reject/text 记录 → 我方唯一出口的统一记录（拒答丢 text）。"""
    from rules import seal

    if render_out.get("kind") == "text":
        return seal(kind="text", channel="generate", text=render_out["text"],
                    evidence=render_out["evidence"], reason="", plan_step_id=psid,
                    extra={"instruction": render_out.get("instruction"),
                           "ref_map": render_out.get("ref_map", [])})
    return seal(kind="reject", channel="reject", text=render_out.get("text"),
                reason=render_out.get("reason", ""), plan_step_id=psid)


# ---- A1 题元方向 --------------------------------------------------------------

def _a1() -> dict:
    text = "我打你。"
    bag = _bag(text, [("我", 0, 1, "名", "施事"), ("打", 1, 2, "动", None),
                      ("你", 2, 3, "名", "受事")])
    good = R.render(R.Instruction("S01", (0, 1, 2)), bag, text)
    swap = R.render(R.Instruction("S01", (2, 1, 0)), bag, text)
    return {"ok_expected": good["kind"], "swap": swap,
            "pass": good["kind"] == "text" and swap["kind"] == "reject"
                    and "题元方向冲突" in (swap.get("reason") or "")}


# ---- A2 / A3 逻辑词词面守卫 ---------------------------------------------------

def _s13_fixture(text: str, bag_spans: list[tuple[str, int, int, str, str | None]]):
    bag = _bag(text, bag_spans)
    assign = tuple(range(len(bag)))
    return R.Instruction("S13", assign), bag


def _a2() -> dict:
    # 输入里既没有「因为」也没有「要」⇒ 词面守卫必须拒（编的因果/情态）；
    # 袋里每个块都逐字、类型/题元都对齐 ⇒ **只剩**逻辑词这一个问题可报。
    text = "我们修车。"
    instr, bag = _s13_fixture(
        text, [("修", 2, 3, "动", None), ("车", 3, 4, "形", None),
               ("我们", 0, 2, "名", "施事"), ("车", 3, 4, "名", "时"),
               ("我们", 0, 2, "动", None)])
    probs = R.check_structure(instr, bag, text)
    out = R.render(instr, bag, text)
    return {"problems": probs, "render": out,
            "pass": len(probs) == 2 and all("词面守卫" in p for p in probs)
                    and out["kind"] == "reject"}


def _a3() -> dict:
    text = "因为下雨，我们明天要修。"
    probs = R.word_face_problems(text, R.SKELETONS["S13"])
    return {"problems": probs, "pass": probs == []}


# ---- A4–A12 结构侧逐条 --------------------------------------------------------

def _a4() -> dict:  # 缺块
    text = "我打你。"
    bag = _bag(text, [("我", 0, 1, "名", "施事"), ("打", 1, 2, "动", None),
                      ("你", 2, 3, "名", "受事")])
    out = R.render(R.Instruction("S01", (0, 1)), bag, text)
    return {"render": out, "pass": out["kind"] == "reject"
            and "缺块" in (out.get("reason") or "")}


def _a5() -> dict:  # 类型不符（动块进名槽）
    text = "我打你。"
    bag = _bag(text, [("我", 0, 1, "名", "施事"), ("打", 1, 2, "动", None),
                      ("你", 2, 3, "名", "受事")])
    out = R.render(R.Instruction("S01", (1, 0, 2)), bag, text)
    return {"render": out, "pass": out["kind"] == "reject"
            and "类型不匹配" in (out.get("reason") or "")}


def _a6() -> dict:  # 数字改动（袋块的 surface 与 span 对不上）
    text = "我们修3台机器。"
    bag = [R.BagItem(ref=0, text="2", span=(3, 4), pos="名", theta=None,
                     candidate_id="f0", screened=True),
           R.BagItem(ref=1, text="修", span=(2, 3), pos="动", theta=None,
                     candidate_id="f1", screened=True)]
    instr = R.Instruction("R02", (0, 1))
    out = R.render(instr, bag, text)
    probs = R.check_structure(instr, bag, text)
    return {"problems": probs, "render": out,
            "pass": any("不是输入逐字子串" in p for p in probs)
                    and out["kind"] == "reject"}


def _a7() -> dict:  # 否定改动（span 指到别的字）
    text = "我不去。"
    bag = [R.BagItem(ref=0, text="不", span=(2, 3), pos="名", theta=None,
                     candidate_id="f0", screened=True),
           R.BagItem(ref=1, text="去", span=(2, 3), pos="动", theta=None,
                     candidate_id="f1", screened=True)]
    instr = R.Instruction("R02", (0, 1))
    out = R.render(instr, bag, text)
    probs = R.check_structure(instr, bag, text)
    return {"problems": probs, "render": out,
            "pass": out["kind"] == "reject"
                    and any("逐字" in p for p in probs)}


def _a8() -> dict:  # 专名改动
    text = "张三修车。"
    bag = [R.BagItem(ref=0, text="李四", span=(0, 2), pos="名", theta=None,
                     candidate_id="f0", screened=True),
           R.BagItem(ref=1, text="修", span=(2, 3), pos="动", theta=None,
                     candidate_id="f1", screened=True)]
    instr = R.Instruction("R02", (0, 1))
    out = R.render(instr, bag, text)
    probs = R.check_structure(instr, bag, text)
    return {"problems": probs, "render": out,
            "pass": out["kind"] == "reject"
                    and any("逐字" in p for p in probs)}


def _a9() -> dict:  # 越界 span
    text = "我打你。"
    bag = [R.BagItem(ref=0, text="我", span=(0, 99), pos="名", theta="施事",
                     candidate_id="f0", screened=True),
           R.BagItem(ref=1, text="打", span=(1, 2), pos="动", theta=None,
                     candidate_id="f1", screened=True),
           R.BagItem(ref=2, text="你", span=(2, 3), pos="名", theta="受事",
                     candidate_id="f2", screened=True)]
    out = R.render(R.Instruction("S01", (0, 1, 2)), bag, text)
    probs = R.check_structure(R.Instruction("S01", (0, 1, 2)), bag, text)
    return {"problems": probs, "render": out,
            "pass": out["kind"] == "reject" and any("越界" in p for p in probs)}


def _a10() -> dict:  # 骨架 id 不存在
    text = "我打你。"
    bag = _bag(text, [("我", 0, 1, "名", "施事"), ("打", 1, 2, "动", None),
                      ("你", 2, 3, "名", "受事")])
    out = R.render(R.Instruction("S99", (0, 1, 2)), bag, text)
    return {"render": out, "pass": out["kind"] == "reject"
            and "骨架 id 不存在" in (out.get("reason") or "")}


def _a11() -> dict:  # 同一块指派到两个槽
    text = "我打你。"
    bag = _bag(text, [("我", 0, 1, "名", "施事"), ("打", 1, 2, "动", None),
                      ("你", 2, 3, "名", "受事")])
    out = R.render(R.Instruction("S07", (0, 0, 1)), bag, text)
    probs = R.check_structure(R.Instruction("S07", (0, 0, 1)), bag, text)
    return {"problems": probs, "render": out,
            "pass": out["kind"] == "reject"
                    and any("被指派到多个槽" in p for p in probs)}


def _a12() -> dict:  # 只复述子集换方向（指派不随原文 span 升序）
    text = "打我。"
    bag = [R.BagItem(ref=0, text="我", span=(1, 2), pos="名", theta=None,
                     candidate_id="f0", screened=True),
           R.BagItem(ref=1, text="打", span=(0, 1), pos="动", theta=None,
                     candidate_id="f1", screened=True)]
    out = R.render(R.Instruction("R02", (0, 1)), bag, text)
    probs = R.check_structure(R.Instruction("R02", (0, 1)), bag, text)
    return {"problems": probs, "render": out,
            "pass": out["kind"] == "reject"
                    and any("不随原文 span 升序" in p for p in probs)}


# ---- A13 / A14 开关入口 --------------------------------------------------------

def _a13() -> dict:  # type 不声明 / 不认识
    from pipeline import execute, fake_propose_anchors

    outs = {}
    for label, decl in (("none", None), ("bogus", "grid-search"),
                        ("known_unsup", "candidates")):
        rec, _dbg = execute("你说得对", type_=decl, propose=fake_propose_anchors,
                            need_g=False)
        outs[label] = rec
    ok = all(r["kind"] == "reject" and "text" not in r
             and (r.get("reason") or "").strip() for r in outs.values())
    return {"records": outs, "pass": ok}


def _a14() -> dict:  # 空输入
    from pipeline import execute, fake_propose_anchors

    rec, _dbg = execute("   ", type_="plain", propose=fake_propose_anchors,
                        need_g=False)
    return {"record": rec, "pass": rec["kind"] == "reject" and "text" not in rec
            and bool((rec.get("reason") or "").strip())}


#: 逐条预期（与 PREREG §6 的表一一对应；`expect` 是人读的一句话）。
CASES: tuple[tuple[str, str, Any], ...] = (
    ("A1", "题元方向不许互换（互换 ⇒ 拒答）", _a1),
    ("A2", "无据逻辑词必须拒", _a2),
    ("A3", "有据逻辑词不误拒（正对照）", _a3),
    ("A4", "缺块必须拒", _a4),
    ("A5", "类型不符必须拒", _a5),
    ("A6", "数字改动必须拒", _a6),
    ("A7", "否定片段改动必须拒", _a7),
    ("A8", "专名改动必须拒", _a8),
    ("A9", "越界 span 必须拒", _a9),
    ("A10", "骨架 id 不存在必须拒", _a10),
    ("A11", "同一块指派到两槽必须拒", _a11),
    ("A12", "只复述子集换方向必须拒", _a12),
    ("A13", "type 未声明/不认识必须拒", _a13),
    ("A14", "空输入必须拒", _a14),
)


def _shrink(obj: Any) -> Any:
    """把 render 的 reject 记录缩成可读摘要（含 text 占位串，供 C6 对账）。"""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in ("ref_map", "evidence") and isinstance(v, list) and len(v) > 3:
                out[k] = v[:3] + [f"...共 {len(v)} 条"]
            else:
                out[k] = _shrink(v)
        return out
    if isinstance(obj, list):
        return [_shrink(x) for x in obj[:6]]
    if isinstance(obj, tuple):
        return [_shrink(x) for x in obj]
    return obj


def run_cases() -> dict:
    """跑完 A1–A14：逐条给 `预期 / 实测 / pass`，外加 C6（两边一致的契约检查）。"""
    rows = []
    c6_raw_render: list[dict] = []
    all_pass = True
    for cid, expect, fn in CASES:
        try:
            res = fn()
        except Exception as exc:  # noqa: BLE001 —— 对抗条目自身炸了就是不过
            res = {"pass": False, "error": f"{type(exc).__name__}: {exc}"}
        passed = bool(res.get("pass"))
        all_pass = all_pass and passed
        row = {"id": cid, "expect": expect, "pass": passed,
               "detail": _shrink({k: v for k, v in res.items() if k != "pass"})}
        # C6：把官方层的 reject 原始记录记下来，与归一后的记录一起过同一个谓词
        for key in ("render", "swap"):
            out = res.get(key)
            if isinstance(out, dict) and out.get("kind") is not None:
                c6_raw_render.append(out)
        for key in ("records",):
            for r in (res.get(key) or {}).values() if isinstance(res.get(key), dict) else []:
                row.setdefault("contract", []).append(contract_problems(r))
        if isinstance(res.get("record"), dict):
            row.setdefault("contract", []).append(contract_problems(res["record"]))
        rows.append(row)

    # C6 的「两边一致」：同一个谓词
    #   ① render 的原始 reject 记录（带「（拒答）…」占位串）→ 必须报 C2 违例
    #   ② 经 seal 归一后的同一批记录 → 必须 0 违例
    raw_reject = [r for r in c6_raw_render if r.get("kind") == "reject"]
    raw_viol = [contract_problems(r) for r in raw_reject]
    raw_viol_n = sum(1 for p in raw_viol if p)
    sealed = [_sealed(r) for r in c6_raw_render]
    sealed_viol = [contract_problems(r) for r in sealed]
    sealed_viol_n = sum(1 for p in sealed_viol if p)
    c6 = {
        "n_official_records": len(c6_raw_render),
        "n_official_reject": len(raw_reject),
        "raw_reject_violating": raw_viol_n,
        "raw_reject_examples": [contract_problems(r) for r in raw_reject[:3]],
        "sealed_violating": sealed_viol_n,
        "verdict": "两侧一致（归一后同一谓词 0 违例；原始违例如实报）"
                   if sealed_viol_n == 0 and raw_viol_n == len(raw_reject)
                   else "不一致",
    }
    return {"cases": rows, "all_pass": all_pass, "c6_both_sides": c6}
