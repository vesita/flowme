"""P25 共用管线：抽取 → 筛选 → 装袋 → 填槽 → 官方渲染 → 四门 + 回溯复核。

**只读复用既有零件**（不改它们）：

- `experiments/card_flow/proposers.fake_propose`  既有假卡（人称/人物/情绪/否定/成语）
- `experiments/card_flow/predicates.filter_candidates`  既有 5 条筛选谓词 F1–F5
- `src/dtseek/tasks/render.py`  官方骨架表 / 渲染 / 结构侧与解引用侧检查
- `src/dtseek/tasks/card_contract.py`  P13 契约 `to_contract` / `contract_problems`

本单元**新增**的只有 `verb_card.propose`（动词卡 + 数字 + 官方词典补货）。
"""
from __future__ import annotations

import itertools
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in (str(ROOT / "src"), str(HERE), str(ROOT / "experiments" / "card_flow")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dtseek.tasks import card_contract as CC  # noqa: E402
from dtseek.tasks import render as R  # noqa: E402

import verb_card as VC  # noqa: E402
from fixtures import CF_VERB_SKELETONS as CF_VERB_IDS  # noqa: E402
from fixtures import Fixture, W2_SKELETONS as W2_IDS  # noqa: E402
from fixtures import gold_spans  # noqa: E402

#: 我们这层新增的卡 → 类别体系（F3 按它核对规范名）。
OUR_CLASS_INDEX: dict[str, tuple[str, ...]] = {
    "verb": ("动词",),
    "number": ("数量",),
    "lexnoun": ("名词",),
    "lexadj": ("形容词",),
}

#: 去重优先级（同一 `(s0, e0, type)` 只留一条）：**既有卡优先**，本单元的抽取只补缺口。
_PRIORITY = {"person": 0, "pronoun": 1, "sentiment": 2, "negation": 3, "idiom": 4,
             "verb": 5, "number": 6, "lexnoun": 7, "lexadj": 8}


# ---- 1. 抽取 + 筛选 + 装袋 -----------------------------------------------------

def collect(text: str, *, with_verb: bool = True,
            floor_verb: list[dict] | None = None) -> tuple[list[dict], list[dict], dict]:
    """既有假卡 + 本单元抽取（动词卡可关 ⇒ 基线/地板），按 `(s0,e0,type)` 去重后过 F1–F5。

    返回 `(筛选后带 valid 的候选, 去重后全量候选, 类别体系)`。
    """
    from predicates import filter_candidates
    from proposers import fake_propose

    fake, fake_index = fake_propose(text)
    ours = VC.propose(text, with_verb=with_verb) + list(floor_verb or [])
    index = dict(fake_index)
    index.update(OUR_CLASS_INDEX)

    best: dict[tuple, dict] = {}
    for c in sorted(fake + ours, key=lambda c: _PRIORITY.get(c["card"], 99)):
        key = (c["s0"], c["e0"], c["type"])
        if key not in best:
            best[key] = c
    cands = sorted(best.values(), key=lambda c: (c["s0"], c["e0"], c["card"]))
    return filter_candidates(text, cands, index), cands, index


def make_bag(text: str, roles: dict[str, str] | None = None, *,
             with_verb: bool = True, floor_verb: list[dict] | None = None
             ) -> tuple[list[R.BagItem], list[dict]]:
    """筛选后的有效候选 → 官方 `BagItem`（题元只来自夹具角色块；`type` 即 `pos`）。"""
    valid, _all, _index = collect(text, with_verb=with_verb, floor_verb=floor_verb)
    roles = roles or {}
    bag: list[R.BagItem] = []
    for c in valid:
        if not c["valid"]:
            continue
        bag.append(R.BagItem(
            ref=len(bag), text=c["surface"], span=(c["s0"], c["e0"]),
            pos=c.get("type"), theta=roles.get(c["surface"]),
            candidate_id=c["cid"], screened=True))
    return bag, valid


# ---- 2. 卡片表：card_flow 25 条 → 官方 `Skeleton` 形制 --------------------------

_PREFIX_TYPE = {"n": "名", "v": "动", "a": "形", "g": "否", "d": "数"}


def cf_skeletons(ids: tuple[str, ...] | None = None) -> dict[str, R.Skeleton]:
    """`experiments/card_flow/skeletons.py` 的骨架 → 官方 `Skeleton` 形制。

    构造规则：`[n1]` 顺序重编号为 `[1]`；**题元原样保留** ⇒ 有题元的骨架
    `direction_safe=False`、没题元的为 `True`（测的就是 card_flow 声明的那批骨架本身）。
    """
    import re as _re

    from skeletons import BY_ID as CF_BY_ID

    out: dict[str, R.Skeleton] = {}
    for sid in ids or (CF_VERB_IDS + W2_IDS):
        src = CF_BY_ID[sid[2:]]                      # "CFS13" → "S13"
        parts: list[str] = []
        slots: list[R.SlotSpec] = []
        pos = 0
        idx = 0
        for m in _re.finditer(r"\[([a-z]\d+)\]", src.template):
            parts.append(src.template[pos:m.start()])
            idx += 1
            parts.append(f"[{idx}]")
            slot = next(s for s in src.slots if s.name == m.group(1))
            slots.append(R.SlotSpec(_PREFIX_TYPE[slot.name[0]], slot.theme))
            pos = m.end()
        parts.append(src.template[pos:])
        has_theme = any(s.theta is not None for s in slots)
        out[sid] = R.Skeleton(sid, "".join(parts), tuple(slots),
                              direction_safe=not has_theme)
    return out


# ---- 3. 填槽 + 渲染 -------------------------------------------------------------

def _literal_pieces(sk: R.Skeleton) -> list[str]:
    """骨架字面按「槽位边界」切成 n+1 段（段0=句首字面，段i=第 i-1 与第 i 槽之间，末段=句尾）。"""
    pieces: list[str] = [""]
    for tok in R.pattern_tokens(sk.pattern):
        if tok[0] == "lit":
            pieces[-1] += tok[2]
        else:
            pieces.append("")
    return pieces


def _aligned(sk: R.Skeleton, text: str, spans: list[tuple[int, int]]) -> bool:
    """**切分对齐**：输入里「槽与槽之间 / 句首」未被指派的文字，逐字等于骨架对应字面。

    - 句首字面**严格**比（把「讨论开心」这类**句首丢内容**的指派排到后面）；
    - 槽间字面**严格**比（于是不会把「我们」拆成「我」+ 骨架自己的「们」）；
    - 句尾**宽松**（夹具输入不带句末标点 ⇒ 空串也算过）；
    - **只是指派优先级，不是门禁** —— 对齐失败会退到下一轮扫描，渲染门禁一条没改。
    """
    pieces = _literal_pieces(sk)
    if text[:spans[0][0]] != pieces[0]:
        return False
    for k in range(len(spans) - 1):
        _s, e = spans[k]
        ns, _ne = spans[k + 1]
        if ns < e or text[e:ns] != pieces[k + 1]:
            return False
    tail = text[spans[-1][1]:]
    return tail == pieces[-1] or tail == ""


def fill(text: str, bag: list[R.BagItem], sk: R.Skeleton,
         table: dict[str, R.Skeleton], *, combo_cap: int = 60000) -> dict:
    """穷举指派：返回第一条 `kind="text"` 的渲染记录；否则给失败原因。

    指派优先级（**确定性**，四轮扫描同一乘积）：
    ① span 升序 + 切分对齐 → ② 切分对齐 → ③ span 升序 → ④ 任意。
    """
    pools = [[it for it in bag if it.pos == slot.pos] for slot in sk.slots]
    if any(not p for p in pools):
        miss = [sk.slots[i].pos for i, p in enumerate(pools) if not p]
        return {"kind": "no_candidate", "missing_pos": miss}
    tried = 0
    last: dict | None = None
    for want in ("asc+aligned", "aligned", "asc", "any"):
        for combo in itertools.product(*pools):
            tried += 1
            if tried > combo_cap:
                return {"kind": "combo_cap", "tried": tried}
            if len({it.ref for it in combo}) != len(combo):
                continue
            spans = [it.span for it in combo]
            starts = [s for s, _ in spans]
            if want.startswith("asc") and starts != sorted(starts):
                continue
            if want.endswith("aligned") and not _aligned(sk, text, spans):
                continue
            out = R.render(R.Instruction(sk.sid, tuple(it.ref for it in combo)),
                           bag, text, skeletons=table)
            if out["kind"] == "text":
                out["tried"] = tried
                return out
            last = out
    if last is None:
        return {"kind": "no_candidate", "missing_pos": []}
    return {"kind": "reject", "reason": last.get("reason", ""), "tried": tried}


def verb_spans(out: dict, bag: list[R.BagItem],
               table: dict[str, R.Skeleton]) -> set[tuple[int, int]]:
    """一条成功记录里**被填进 `动` 槽**的 span 集合（W0/W1 的 gold 比对用）。"""
    if out.get("kind") != "text":
        return set()
    sk = table[out["instruction"]["skeleton_id"]]
    items = {b.ref: b for b in bag}
    return {tuple(items[r].span) for r, s in
            zip(out["instruction"]["assignment"], sk.slots) if s.pos == "动"}


# ---- 4. 四门（W3）与回溯复核（W4） ----------------------------------------------

def gates(out: dict, bag: list[R.BagItem], text: str, table: dict[str, R.Skeleton]
          ) -> dict[str, int]:
    """W3：`rule_a` / `slot_schema` / `item` / `contract` 四门违例条数（外加结构侧/解引用侧）。"""
    if out.get("kind") != "text":
        return {"rule_a": -1, "slot_schema": -1, "item": -1, "contract": -1,
                "structure": -1, "deref": -1}
    sk = table[out["instruction"]["skeleton_id"]]
    used = [b for b in bag if b.ref in out["instruction"]["assignment"]]
    rec = CC.to_contract(dict(out))
    instr = R.Instruction(sk.sid, tuple(out["instruction"]["assignment"]))
    return {
        "rule_a": len(R.rule_a_problems(sk)),
        "slot_schema": len(R.slot_schema_problems(sk)),
        "item": sum(len(R.item_problems(b, text)) for b in used),
        "contract": len(CC.contract_problems(rec, input_text=text)),
        "structure": len(R.check_structure(instr, bag, text, skeletons=table)),
        "deref": len(R.check_deref(dict(out), bag, text, skeletons=table)),
    }


def trace_audit(out: dict, bag: list[R.BagItem], text: str,
                table: dict[str, R.Skeleton]) -> list[str]:
    """W4：**独立复核**（不复用渲染层结论）—— 内容单元逐字回溯 + 字面单元只走表项 id。"""
    probs: list[str] = []
    if out.get("kind") != "text":
        return ["not_text"]
    sk = table[out["instruction"]["skeleton_id"]]
    items = {b.ref: b for b in bag}
    lit_seq = [t[2] for t in R.pattern_tokens(sk.pattern) if t[0] == "lit"]
    lit_i = 0
    cursor = 0
    for i, e in enumerate(out.get("ref_map") or []):
        s, t = e["out"]
        if s != cursor or t < s or t > len(out["text"]):
            probs.append(f"ref_map[{i}] 区间不连续")
        if out["text"][s:t] != e["text"]:
            probs.append(f"ref_map[{i}] 面与文本不符")
        cursor = t
        if e["cls"] == "content":
            it = items.get(e["ref"])
            if it is None:
                probs.append(f"ref_map[{i}] ref 不在袋里")
                continue
            if text[it.span[0]:it.span[1]] != it.text or it.text != e["text"]:
                probs.append(f"ref_map[{i}] 内容单元非逐字")
            if tuple(e.get("span") or ()) != tuple(it.span):
                probs.append(f"ref_map[{i}] 内容单元 span 与袋块不符")
            ev = [x for x in out["evidence"]
                  if x.get("source") == "span" and x.get("ref") == e["ref"]]
            if not ev or tuple(ev[0]["span"]) != tuple(it.span) or ev[0]["text"] != it.text:
                probs.append(f"ref_map[{i}] 内容单元证据缺逐字 span")
            elif not ev[0].get("candidate_id") or not it.screened:
                probs.append(f"ref_map[{i}] 内容单元未回溯到「被筛为有效」的候选")
        else:
            if lit_i >= len(lit_seq) or e["text"] != lit_seq[lit_i]:
                probs.append(f"ref_map[{i}] 表项与骨架字面不符")
                continue
            lit_i += 1
            ev = [x for x in out["evidence"]
                  if x.get("unit_id") == e.get("unit_id")]
            if not ev or ev[0].get("source") != "table":
                probs.append(f"ref_map[{i}] 字面单元证据不是 table")
    if cursor != len(out["text"]):
        probs.append("ref_map 未铺满输出")
    if lit_i != len(lit_seq):
        probs.append("字面表项数不符")
    return probs


def gold_of(fix: Fixture) -> set[tuple[int, int]]:
    """夹具里人工判定的 gold 动词 span。"""
    return gold_spans(fix)
