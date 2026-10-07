#!/usr/bin/env python3
"""识别卡的**结构化指令出口**（口径对齐 `src/dtseek/tasks/render.py`）。

复用点（只读 import，不改 `src/`）：
  · **遮蔽字符集** = `render.PUNCT`（「标点是纯形式」的同一口径）；
  · **evidence 齐备谓词** = `render.item_problems`（span 越界 / 逐字子串 /
    `screened`+`candidate_id` 零信息新增），不满足 ⇒ `kind="reject"`（fail-closed）；
  · **词性谓词** = `render.is_content_word`（触发词必须是**形式/功能词**，
    实义词不得充当触发词 —— 对应规则 A「实义词不进字面」的同一立场）；
  · 记录形状 `{kind, text, evidence, plan_step_id, instruction, ref_map, reason}`
    与 `render._record` 同构；`check_mode_structure` / `check_mode_deref`
    逐条对齐 `check_structure` / `check_deref` 的语义（schema / 逐字 / 分区覆盖 / 可回溯）。

指令形状：
    {"mode_id": "M03", "assignment": [0]}     # assignment 指向袋里的触发词引用（0 个或 1 个）
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dtseek.tasks import render  # noqa: E402

import labels as L  # noqa: E402

#: 模式表（封闭）：`M01..M06`，与 `labels.MODES` 一一对应；`need_trigger` 只是**提示**，
#: 无触发词不是错误（陈述/感叹天然为 ∅），fail-closed 只看 id 是否在表内。
MODE_TABLE: dict[str, dict] = {
    f"M{i + 1:02d}": {"name": name, "idx": i}
    for i, name in enumerate(L.MODES)
}
IDX_TO_SID = {v["idx"]: k for k, v in MODE_TABLE.items()}
SID_TO_IDX = {v["idx"]: k for k, v in MODE_TABLE.items()}  # type: ignore[misc]


def sid_of(mode_idx: int) -> str:
    return IDX_TO_SID[mode_idx]


def item_for_trigger(span: tuple[int, int] | None, text: str, sid: str):
    """把触发词包成 `render.BagItem`（零信息新增：回溯到「被筛为有效」的候选）。"""
    if span is None:
        return None
    s, e = span
    return render.BagItem(ref=0, text=text[s:e], span=(s, e),
                          pos=None, theta=None,
                          candidate_id=f"mode:{sid}", screened=True)


def check_mode_structure(mode_idx: int, span: tuple[int, int] | None,
                         text: str) -> list[str]:
    """**结构侧（I1 同构）**：模式 id 在封闭表内 + 触发词是形式词 + evidence 齐备。"""
    probs: list[str] = []
    sid = SID_TO_IDX.get(mode_idx)
    if sid is None or sid not in MODE_TABLE:
        return [f"模式 id 不存在：{mode_idx}（封闭表，不猜）"]
    if span is None:
        return probs                                   # ∅ 触发词合法（陈述/感叹）
    s, e = span
    if not (0 <= s < e <= len(text)):
        probs.append(f"evidence 不齐备：触发词区间 {span} 越界")
        return probs
    word = text[s:e]
    if render.is_content_word(word):
        probs.append(f"规则 A 同构违规：触发词 {word!r} 是实义词（触发词只许形式/功能词）")
    if word not in L.TRIGGER_SET:
        probs.append(f"触发词不在封闭表：{word!r}")
    probs += render.item_problems(item_for_trigger(span, text, sid), text)
    return probs


def check_mode_deref(record: dict, text: str) -> list[str]:
    """**解引用侧（I3 同构）**：`ref_map` 的 `out` 区间恰好铺满 `text`（= 触发词），
    内容条目的 `span` 必须能解析回输入原文（逐字子串）。"""
    probs: list[str] = []
    if record.get("kind") != "text":
        return [f"解引用检查只适用于 kind=text，当前 {record.get('kind')!r}"]
    t = record.get("text")
    ref_map = record.get("ref_map")
    if not isinstance(t, str) or not isinstance(ref_map, list):
        return ["I3 违规：记录缺 text / ref_map"]
    cursor = 0
    for i, entry in enumerate(ref_map):
        try:
            a, b = entry["out"]
        except (KeyError, TypeError, ValueError):
            return [f"I3 违规：ref_map[{i}] 缺合法 out 区间"]
        if not (0 <= a <= b <= len(t)) or a != cursor:
            return [f"I3 违规：ref_map[{i}] 区间 {entry['out']} 不覆盖位置 {cursor}"]
        if t[a:b] != entry.get("text"):
            return [f"I3 违规：ref_map[{i}] 文本与实际不符"]
        cursor = b
    if cursor != len(t):
        return [f"I3 违规：ref_map 只覆盖 {cursor}/{len(t)}"]
    for i, entry in enumerate(ref_map):
        if entry.get("cls") == "content":
            sp = entry.get("span")
            if not isinstance(sp, (tuple, list)) or len(sp) != 2:
                probs.append(f"I3 违规：ref_map[{i}] 内容条目缺 span")
                continue
            a, b = sp
            if not (0 <= a < b <= len(text)) or text[a:b] != entry.get("text"):
                probs.append(f"I3 违规：ref_map[{i}] span {sp} 不是输入逐字子串")
    return probs


def make_record(mode_idx: int, span: tuple[int, int] | None, text: str,
                *, plan_step_id: str = "", input_text: str = "") -> dict:
    """结构指令 → 记录（与 `render._record` 同形状）。任一检查不过 ⇒ `kind="reject"`。"""
    src = input_text or text
    if mode_idx not in SID_TO_IDX:
        return {"kind": "reject", "text": "（拒答）模式 id 不在封闭表", "evidence": [],
                "plan_step_id": plan_step_id,
                "instruction": {"mode_id": str(mode_idx), "assignment": []},
                "ref_map": [], "reason": "模式 id 不在封闭表"}
    sid = sid_of(mode_idx)
    probs = check_mode_structure(mode_idx, span, src)
    if probs:
        return {"kind": "reject", "text": f"（拒答）{'；'.join(probs)}", "evidence": [],
                "plan_step_id": plan_step_id,
                "instruction": {"mode_id": sid, "assignment": []},
                "ref_map": [], "reason": "；".join(probs)}

    if span is None:
        rec = {"kind": "text", "text": "", "ref_map": [],
               "evidence": [{"source": "table", "unit_id": f"{sid}#0",
                             "word": "∅", "cls": "formal"}],
               "instruction": {"mode_id": sid, "assignment": []}}
    else:
        s, e = span
        word = src[s:e]
        rec = {"kind": "text", "text": word,
               "ref_map": [{"unit_id": "ref:0", "cls": "content", "text": word,
                            "ref": 0, "span": (s, e), "out": (0, len(word))}],
               "evidence": [{"source": "span", "ref": 0, "text": word, "span": (s, e),
                             "candidate_id": f"mode:{sid}"}],
               "instruction": {"mode_id": sid, "assignment": [0]}}
    rec["plan_step_id"] = plan_step_id
    rec["reason"] = ""
    probs = check_mode_deref(rec, src)
    if probs:
        return {"kind": "reject", "text": f"（拒答）{'；'.join(probs)}", "evidence": [],
                "plan_step_id": plan_step_id,
                "instruction": {"mode_id": sid, "assignment": []},
                "ref_map": [], "reason": "；".join(probs)}
    return rec
