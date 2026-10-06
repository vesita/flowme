"""手写开关规则 R-CH + 统一记录契约（**无任何学出来的量**）。

三件事：

1. `switch_channel()` —— 指针 / 生成 / 拒答 的通道开关，if-else 链逐行照抄 `RULE_TEXT`；
   输入只有三个可核对量：计划终点 T（`dispatch.decide()` 手写规则表的输出）、
   生成可行 G（穷举手写骨架表 + 官方两道谓词）、指针有据 P（能不能用模板+切片拼出回复）。
2. `seal()` —— **唯一出口**：把三条通道的产物归一成统一记录
   `{kind, text, evidence, plan_step_id}`；拒答在这里**丢掉 text 占位串**。
3. `contract_problems()` —— 拒答契约 C1–C6 的谓词（空 = 通过）。
   同一个谓词跑「本层出口」与「render / dialogue 的原始记录」，两边一致（C6）。

`rule_audit()` 用 `inspect.getsource` 读 R-CH 源码，断言不含任何学习量字样。
"""
from __future__ import annotations

import inspect
import re
from typing import Any

__all__ = [
    "CHANNELS",
    "RULE_TEXT",
    "contract_problems",
    "rule_audit",
    "seal",
    "switch_channel",
]

#: 通道取值域（与 `dispatch.TERMINAL_KINDS` 同名同序，便于对账）。
CHANNELS: tuple[str, ...] = ("pointer", "generate", "reject")

#: 开关规则**原文**（跑前写死；报告必须原样贴出）。
RULE_TEXT = """\
规则 R-CH（指针 / 生成 / 拒答 的通道开关；手写 if-else，逐行照抄，行序固定）
  输入：
    T = dispatch 规则表 decide()（R0–R6 那张 if-else 链）在本状态给出的终止动作
        ∈ {generate, pointer, reject}；T 的输入 = (声明的 type, 4 位三态, conf, steps)
    G = 「可合法渲染」谓词：官方骨架表里存在 (骨架, 指派) 使
        check_structure() == [] 且 render() 出 kind="text" 且 check_deref() == []
    P = 「指针有据」谓词：计划授权调用过的卡能用「手写模板 + 逐字切片」拼出回复
        （compose_reply 成功；越界 / 无切片 / 无模板 ⇒ 假）
  行：
    R-CH0  T == reject                    ⇒ reject   （text 缺省，reason 必填）
    R-CH1  T == generate 且 G             ⇒ generate （两道检查 0 问题才出 text）
    R-CH2  T == generate 且 ¬G 且 P       ⇒ pointer  （降级；reason 记「生成不可行」）
    R-CH3  T == generate 且 ¬G 且 ¬P      ⇒ reject
    R-CH4  T == pointer  且 P             ⇒ pointer
    R-CH5  T == pointer  且 ¬P            ⇒ reject   （指针无据不许编；G 也不许越权升级）
  行序 = 上表顺序；同一输入两次执行命中同一行。
"""

#: `rule_audit()` 认定为「学出来的量」的字样（源码里出现任意一个即判失败）。
FORBIDDEN: tuple[str, ...] = (
    "state_dict", "torch", "nn.", "logit", "softmax", "sigmoid", "embedding",
    "checkpoint", "weights", ".pt", "learn", "trained", "probability",
)


def switch_channel(terminal: str, gen_ok: bool, ptr_ok: bool) -> tuple[str, str]:
    """R-CH0–R-CH5：给定计划终点与两条可核对谓词，返回 `(channel, 规则行)`。

    `terminal` 是 `dispatch` 规则表的终止动作种类（`generate` / `pointer` / `reject`）；
    未知的 `terminal` 值 ⇒ fail-closed 走拒答（判不动不许猜）。
    """
    if terminal == "reject":
        return "reject", "R-CH0 计划终点=reject"
    if terminal == "generate":
        if gen_ok:
            return "generate", "R-CH1 终点=generate 且可合法渲染"
        if ptr_ok:
            return "pointer", "R-CH2 终点=generate 但不可合法渲染 ⇒ 降级指针"
        return "reject", "R-CH3 终点=generate 但既不可渲染也无指针据"
    if terminal == "pointer":
        if ptr_ok:
            return "pointer", "R-CH4 终点=pointer 且指针有据"
        return "reject", "R-CH5 终点=pointer 但指针无据：不许编，也不许越权改走生成"
    return "reject", "R-CH-  计划终点读不懂（判不动）⇒ fail-closed"


def rule_audit() -> dict:
    """H4 的断言：R-CH 源码里不含任何学习量字样。返回 `{ok, hits, source}`。"""
    source = inspect.getsource(switch_channel)
    hits = [w for w in FORBIDDEN if w.lower() in source.lower()]
    return {"ok": not hits, "hits": hits, "source": source, "forbidden": list(FORBIDDEN)}


# ---- 统一记录 + 拒答契约 -----------------------------------------------------

#: 核心四键（`dev-notes/16 §1` 统一记录）。
CORE_KEYS: tuple[str, ...] = ("kind", "text", "evidence", "plan_step_id")


def seal(
    *,
    kind: str,
    channel: str,
    reason: str,
    plan_step_id: str,
    text: str | None = None,
    evidence: list[dict] | None = None,
    plan: list[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict:
    """**唯一出口**：把一条产出归一成统一记录（拒答在这里丢掉 `text` 占位串）。

    - `kind == "reject"`：`text` 键**不出现**，`evidence` 必须为空，`reason` 必须非空；
      非空 `text`（例如 render 的 `（拒答）…` 占位串）会被丢弃，其内容并入 `reason`。
    - `kind == "text"`：`text` 必须非空、`evidence` 必须非空（生成型**必须带 evidence**）。
    """
    if kind not in ("text", "reject"):
        raise ValueError(f"kind={kind!r} 不在 {{'text','reject'}}")
    if channel not in CHANNELS:
        raise ValueError(f"channel={channel!r} 不在 {CHANNELS}")
    ev = list(evidence or [])
    why = (reason or "").strip()
    if kind == "reject":
        # render/dialogue 的 `（拒答）…` 占位串 ⇒ 去掉前缀后并进 reason，**text 键不落地**
        bare = (text or "").removeprefix("（拒答）").strip()
        if not why and bare:
            why = bare
        elif bare and bare not in why:
            why = f"{why}（{bare}）"
        if not why:
            raise ValueError("拒答记录必须带非空 reason")
        if ev:
            raise ValueError("拒答记录不许带 evidence")
        out = {"kind": "reject", "evidence": [], "plan_step_id": plan_step_id,
               "channel": "reject", "reason": why, "plan": list(plan or [])}
    else:
        if not text:
            raise ValueError("kind=text 记录必须带非空 text")
        if not ev:
            raise ValueError("kind=text 记录必须带非空 evidence")
        out = {"kind": "text", "text": text, "evidence": ev, "plan_step_id": plan_step_id,
               "channel": channel, "reason": why, "plan": list(plan or [])}
    if extra:
        out.update(extra)
    return out


def contract_problems(rec: dict, *, n_turn_steps: int | None = None) -> list[str]:
    """拒答契约 C1–C6 的谓词。返回问题列表（**空 = 通过**）。

    `n_turn_steps` 给出这次轮次累计的 Step 数：给了就核 `plan_step_id` 与计划对齐（C4）。
    同一谓词对 `render.render()` / `dialogue.respond()` 的**原始记录**照样能判 ——
    这就是「两边一致」的那个检查（C6）。
    """
    p: list[str] = []
    if not isinstance(rec, dict):
        return [f"记录不是 dict：{type(rec).__name__}"]
    kind = rec.get("kind")
    if kind not in ("text", "reject"):
        p.append(f"C1 kind={kind!r} 不在 ('text','reject')")
    for key in ("kind", "evidence", "plan_step_id"):
        if key not in rec:
            p.append(f"C1 缺核心键 {key!r}")
    psid = rec.get("plan_step_id")
    if not isinstance(psid, str) or not psid:
        p.append(f"C1 plan_step_id={psid!r} 不是非空 str")
    elif psid != "none":
        m = re.fullmatch(r"step:(\d+)", psid)
        if m is None:
            p.append(f"C4 plan_step_id={psid!r} 既不是 'none' 也不是 'step:<i>'")
        elif n_turn_steps is not None and not (0 <= int(m.group(1)) < n_turn_steps):
            p.append(f"C4 plan_step_id={psid!r} 越出轮次 Step 数 {n_turn_steps}")

    if kind == "reject":
        if "text" in rec and rec["text"] is not None:
            p.append(f"C2 拒答记录带 text（= {rec['text']!r}）：按契约必须缺省")
        reason = rec.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            p.append(f"C2 拒答记录的 reason 缺失或空白：{reason!r}")
        if rec.get("evidence"):
            p.append("C2 拒答记录带 evidence：必须为空")
        if psid == "none" and rec.get("plan"):
            p.append("C4 plan_step_id='none' 但 plan 非空")
    elif kind == "text":
        text = rec.get("text")
        if not isinstance(text, str) or not text:
            p.append(f"C3 kind=text 但 text 空/非 str：{text!r}")
        if not rec.get("evidence"):
            p.append("C3 kind=text 但 evidence 为空（生成型/指针型都必须带 evidence）")
        if rec.get("channel") == "generate":
            if not rec.get("instruction"):
                p.append("C3 生成型记录缺 instruction（结构指令）")
            if not rec.get("ref_map"):
                p.append("C3 生成型记录缺 ref_map（引用映射）")
    return p
