"""卡片契约（P13）：**一套**记录 schema + 拒答语义 + 归一器 + 校验谓词。

四套东西在这里合成一套：

1. **记录格式**：`CardRecord` —— 字段
   `{kind, text, evidence, plan_step_id, instruction, ref_map, reason, channel}`
   ＋通道可选字段（`plan/cards_run/type/terminal/chosen/candidates/scores_kind`）。
2. **拒答语义**：`kind == "reject"` ⇒ **`text` 缺省**（`to_dict()` 不落地这个键）、
   `reason` 必填非空、`evidence` 必须为空；上游 `render.render()` 与 `dialogue._reject()`
   的 `（拒答）…` 占位串在 `to_contract()` 里并进 `reason` 并丢弃 `text`。
3. **唯一出口**：任何上游记录先过 `to_contract()` 再进系统；`contract_problems()` 是判它过不过的谓词。
4. **内容单元逐字回溯**：复用 `render.item_problems`（span 逐字 + 零信息新增），
   不另立一套谓词。

用法::

    rec = to_contract(render.render(...))          # CardRecord
    contract_problems(rec, input_text=src)         # [] = 通过
    rec.to_dict()                                  # 拒答记录里没有 "text" 键

本模块是**新增**文件：没有任何既有模块 import 它 ⇒ 默认行为逐字不变（等价性由
`experiments/card_contract/` 的重放 + 全库 pytest 实测证明）。
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from typing import Any

from dtseek.tasks import render as R

__all__ = [
    "CHANNELS",
    "CardRecord",
    "KINDS",
    "contract_problems",
    "content_problems",
    "to_contract",
    "to_dict",
]

#: 记录种类（`pointer` 单列：它有 `text` 但没有 `instruction`）。
KINDS: tuple[str, ...] = ("text", "reject", "pointer")

#: 通道取值域（与 `gen_dispatch.rules.CHANNELS` 同名同序，便于对账）。
CHANNELS: tuple[str, ...] = ("pointer", "generate", "reject")

_PLACEHOLDER = "（拒答）"


# ---- 1. 记录 schema ------------------------------------------------------------

@dataclass
class CardRecord:
    """一条卡片记录的**唯一形状**（字段顺序即文档顺序）。"""

    kind: str                                   # text / reject / pointer
    evidence: list[dict] = field(default_factory=list)
    plan_step_id: str = "none"
    reason: str = ""
    channel: str = ""
    text: str | None = None                     # 拒答时必须为 None（不落地）
    instruction: dict | None = None             # {skeleton_id, assignment}
    ref_map: list[dict] = field(default_factory=list)
    plan: list[str] = field(default_factory=list)
    cards_run: list[str] = field(default_factory=list)
    type: str | None = None
    terminal: str | None = None
    # —— 指针通道的结构字段（候选 / 序 / 分数）——
    chosen: list[dict] = field(default_factory=list)
    candidates: list[dict] = field(default_factory=list)
    scores_kind: str = ""

    def to_dict(self) -> dict:
        """落成普通 dict：**值为 `None` 的字段不落地**（所以拒答记录没有 `text` 键）。"""
        return to_dict(self)


def to_dict(rec: CardRecord | dict) -> dict:
    if isinstance(rec, dict):
        return dict(rec)
    out: dict[str, Any] = {}
    for f in fields(rec):
        v = getattr(rec, f.name)
        if v is None:
            continue
        out[f.name] = v
    return out


# ---- 2. 归一（唯一出口） -------------------------------------------------------

def _channel_of(kind: str, instruction: dict | None) -> str:
    if kind == "reject":
        return "reject"
    if kind == "pointer":
        return "pointer"
    return "generate" if instruction else "pointer"


def to_contract(raw: dict, *, channel: str | None = None,
                structure: dict | None = None) -> CardRecord:
    """上游记录 → 契约记录（**唯一出口**）。

    - `kind="reject"`：丢弃 `（拒答）…` 占位 `text`，其内容并进 `reason`（若 `reason` 已有则保留）；
      `evidence` 清空；
    - `plan_step_id` 缺失 ⇒ 由 `plan` 推：`plan` 非空 → `step:{len(plan)-1}`，否则 `none`
      （构造规定：记录里没有可指的计划步就指 `none`）；
    - 指针记录带 `structure={chosen, candidates, scores_kind}` 时补齐结构字段。
    """
    if not isinstance(raw, dict):
        raise TypeError(f"上游记录不是 dict：{type(raw).__name__}")
    kind = str(raw.get("kind") or "")
    if kind not in KINDS:
        raise ValueError(f"kind={kind!r} 不在 {KINDS}")
    plan = list(raw.get("plan") or [])
    psid = raw.get("plan_step_id")
    if not (isinstance(psid, str) and psid):
        psid = f"step:{len(plan) - 1}" if plan else "none"

    text = raw.get("text")
    reason = str(raw.get("reason") or "")
    evidence = list(raw.get("evidence") or [])
    instruction = raw.get("instruction")

    if kind == "reject":
        bare = str(text or "").removeprefix(_PLACEHOLDER).strip() if text else ""
        if not reason.strip() and bare:
            reason = bare
        elif bare and bare not in reason:
            reason = f"{reason}（{bare}）" if reason.strip() else bare
        text = None
        evidence = []

    rec = CardRecord(
        kind=kind,
        evidence=evidence,
        plan_step_id=psid,
        reason=reason.strip() if isinstance(reason, str) else reason,
        channel=channel or _channel_of(kind, instruction),
        text=text,
        instruction=instruction,
        ref_map=list(raw.get("ref_map") or []),
        plan=plan,
        cards_run=list(raw.get("cards_run") or []),
        type=raw.get("type"),
        terminal=raw.get("terminal"),
    )
    if structure:
        rec.chosen = list(structure.get("chosen") or [])
        rec.candidates = list(structure.get("candidates") or [])
        rec.scores_kind = str(structure.get("scores_kind") or "")
    return rec


# ---- 3. 校验谓词 ---------------------------------------------------------------

def _as_dict(rec: CardRecord | dict) -> dict:
    return rec if isinstance(rec, dict) else to_dict(rec)


def content_problems(rec: CardRecord | dict, input_text: str | None) -> list[str]:
    """**内容单元逐字回溯**：复用 `render.item_problems`（span 逐字 + 零信息新增）。

    `ref_map` 的每个内容条目都被当成一个袋块送去检；`candidates`（指针通道）同样检。
    缺 `input_text` ⇒ 返回空（调用方负责传，fail-closed 由 `contract_problems` 另判）。
    """
    if not input_text:
        return []
    d = _as_dict(rec)
    probs: list[str] = []
    seen: set[tuple] = set()

    def _check(ref: int, text: str, span: tuple[int, int], cid: str) -> None:
        nonlocal probs
        key = (ref, text, tuple(span))
        if key in seen:
            return
        seen.add(key)
        item = R.BagItem(ref=ref, text=text, span=tuple(span), pos=None,
                         theta=None, candidate_id=cid, screened=bool(cid))
        probs += [f"C5 {p}" for p in R.item_problems(item, input_text)]

    for i, e in enumerate(d.get("ref_map") or []):
        if e.get("cls") != "content":
            continue
        span = e.get("span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            probs.append(f"C5 ref_map[{i}] 内容条目缺 span：{e!r}")
            continue
        cid = ""
        cands = d.get("candidates") or []
        if cands and 0 <= int(e["ref"]) < len(cands):
            # 指针记录：`ref` 就是候选集下标（候选集本身就是逐字回溯的来源）
            cid = f"ptr:{cands[int(e['ref'])].get('cid', e['ref'])}"
        else:
            for ev in d.get("evidence") or []:
                if ev.get("source") == "span" and ev.get("ref") == e.get("ref") \
                        and ev.get("text") == e.get("text"):
                    cid = str(ev.get("candidate_id") or "")
                    break
        _check(int(e["ref"]), str(e.get("text") or ""), (int(span[0]), int(span[1])), cid)

    for j, c in enumerate(d.get("candidates") or []):
        span = c.get("span")
        if not isinstance(span, (list, tuple)) or len(span) != 2:
            probs.append(f"C5 candidates[{j}] 缺 span：{c!r}")
            continue
        _check(j, str(c.get("span_text") or ""),
               (int(span[0]), int(span[1])), f"ptr:{c.get('cid', j)}")
    return probs


def contract_problems(rec: CardRecord | dict, *, input_text: str | None = None,
                      n_turn_steps: int | None = None) -> list[str]:
    """契约谓词 C1–C7（**空 = 通过**）。

    - **C1** 核心键与取值域：`kind ∈ KINDS`、`evidence`、`plan_step_id`、`channel ∈ CHANNELS`；
    - **C2 拒答语义**：`text` 缺省、`reason` 非空、`evidence` 为空；
    - **C3 文本**：`text` 非空、`evidence` 非空；生成型必须带 `instruction` + `ref_map`；
    - **C4** `plan_step_id` 形制与计划对齐（`none` 不许挂非空 plan）；
    - **C5 内容逐字回溯**（给 `input_text` 才查，复用 `render.item_problems`）；
    - **C6** 指针记录必须带结构字段 `chosen`/`candidates`/`scores_kind`（候选/序/分数）；
    - **C7** 生成型 `ref_map` 必须非空且能铺满 `text`（分区覆盖，复用 `render.check_deref` 的
      覆盖判据）。
    """
    d = _as_dict(rec)
    p: list[str] = []
    kind = d.get("kind")
    if kind not in KINDS:
        p.append(f"C1 kind={kind!r} 不在 {KINDS}")
    if "evidence" not in d:
        p.append("C1 缺核心键 'evidence'")
    psid = d.get("plan_step_id")
    if not isinstance(psid, str) or not psid:
        p.append(f"C1 plan_step_id={psid!r} 不是非空 str")
    chan = d.get("channel")
    if chan not in CHANNELS:
        p.append(f"C1 channel={chan!r} 不在 {CHANNELS}")

    if isinstance(psid, str) and psid and psid != "none":
        m = re.fullmatch(r"step:(\d+)", psid)
        if m is None:
            p.append(f"C4 plan_step_id={psid!r} 既不是 'none' 也不是 'step:<i>'")
        elif n_turn_steps is not None and not (0 <= int(m.group(1)) < n_turn_steps):
            p.append(f"C4 plan_step_id={psid!r} 越出轮次 Step 数 {n_turn_steps}")

    if kind == "reject":
        if d.get("text") is not None:
            p.append(f"C2 拒答记录带 text（= {d.get('text')!r}）：按契约必须缺省")
        if not str(d.get("reason") or "").strip():
            p.append(f"C2 拒答记录的 reason 缺失或空白：{d.get('reason')!r}")
        if d.get("evidence"):
            p.append("C2 拒答记录带 evidence：必须为空")
        if psid == "none" and d.get("plan"):
            p.append("C4 plan_step_id='none' 但 plan 非空")
    elif kind == "text":
        text = d.get("text")
        if not isinstance(text, str) or not text:
            p.append(f"C3 kind=text 但 text 空/非 str：{text!r}")
        if not d.get("evidence"):
            p.append("C3 kind=text 但 evidence 为空")
        if chan == "generate":
            if not d.get("instruction"):
                p.append("C3 生成型记录缺 instruction（结构指令）")
            if not d.get("ref_map"):
                p.append("C3 生成型记录缺 ref_map（引用映射）")
            else:
                p += _cover_problems(text, d.get("ref_map") or [])
    elif kind == "pointer":
        if not d.get("evidence"):
            p.append("C3 kind=pointer 但 evidence 为空")
        if not d.get("candidates"):
            p.append("C6 指针记录缺结构字段 candidates（候选集）")
        if d.get("chosen") is None:
            p.append("C6 指针记录缺结构字段 chosen（入选者）")
        if not str(d.get("scores_kind") or "").strip():
            p.append("C6 指针记录缺 scores_kind（分数口径）")

    p += content_problems(d, input_text)
    return p


def _cover_problems(text: str, ref_map: list[dict]) -> list[str]:
    """C7：`ref_map` 各条 `out` 区间必须恰好铺满 `[0, len(text))` 且文本逐字相符。"""
    probs: list[str] = []
    cursor = 0
    for i, e in enumerate(ref_map):
        out = e.get("out")
        if not isinstance(out, (list, tuple)) or len(out) != 2:
            probs.append(f"C7 ref_map[{i}] 缺合法 out 区间")
            return probs
        start, end = int(out[0]), int(out[1])
        if not (0 <= start <= end <= len(text)) or start != cursor:
            probs.append(f"C7 ref_map[{i}] 区间 {out} 不覆盖文本位置 {cursor}")
            return probs
        if text[start:end] != e.get("text"):
            probs.append(
                f"C7 ref_map[{i}] 文本 {e.get('text')!r} 与实际 {text[start:end]!r} 不符")
        cursor = end
    if cursor != len(text):
        probs.append(f"C7 ref_map 只覆盖到 {cursor}，文本长 {len(text)}")
    return probs
