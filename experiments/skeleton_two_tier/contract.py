"""统一契约（C3/G1）：**拒答记录不带 `text` 字段**。

官方 `src/dtseek/tasks/render.py` 的 reject 记录与 `dialogue.py` 的 `_reject()`
**原始都带占位 `text`**（形如 `（拒答）…`），而 `card_flow` 的 G1 判据字面是
「reject 不许有 `text` 字段」。两者不冲突，但必须**在同一张契约表下判**：

- `normalize()`：把**占位** text 归一掉（`kind/status == reject` 且 text 为空或以
  `（拒答）` 开头 ⇒ 删字段）；**非占位的拒答 text 不许删**（那是真违例，要留着被抓）；
- `is_reject()` / `is_text()`：两个命名空间（`status` vs `kind`）的统一判据；
- 判据因此写成：**归一后**拒答记录无 `text`，正例记录 `text` 非空。
"""
from __future__ import annotations

PLACEHOLDER_PREFIX = "（拒答）"


def is_reject(rec: dict) -> bool:
    return rec.get("status") == "reject" or rec.get("kind") == "reject"


def is_text(rec: dict) -> bool:
    return rec.get("status") == "ok" or rec.get("kind") == "text"


def normalize(rec: dict) -> dict:
    """归一：拒答记录的**占位** text ⇒ 删字段；非占位 ⇒ 保留（留给 G1 判违例）。"""
    if not is_reject(rec):
        return dict(rec)
    out = dict(rec)
    t = out.get("text")
    if t is None or t == "" or (isinstance(t, str) and t.startswith(PLACEHOLDER_PREFIX)):
        out.pop("text", None)
    return out


def g1_problems(records: list[dict]) -> list[dict]:
    """G1：归一后，每条拒答都**没有 text**；每条正例都有**非空 text**。"""
    bad: list[dict] = []
    for i, raw in enumerate(records):
        rec = normalize(raw)
        if is_reject(rec):
            if "text" in rec:
                bad.append({"i": i, "why": "reject_has_text", "text": rec.get("text")})
        elif is_text(rec):
            if not rec.get("text"):
                bad.append({"i": i, "why": "text_empty"})
    return bad
