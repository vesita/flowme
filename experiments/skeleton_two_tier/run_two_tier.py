#!/usr/bin/env python3
"""两级骨架机制驱动 —— 执行 `PREREG.md` 里跑前写死的判据。

    uv run python experiments/skeleton_two_tier/run_two_tier.py --mode selfcheck  # 假卡 + 对抗组 + 审计（无 GPU）
    uv run python experiments/skeleton_two_tier/run_two_tier.py --mode adv        # 对抗组（Tier-2 14 条 + Tier-1 复跑）
    uv run python experiments/skeleton_two_tier/run_two_tier.py --mode real       # 真卡 120 句 × 2 遍（G4 含重推理）
    uv run python experiments/skeleton_two_tier/run_two_tier.py --mode official   # 官方层交叉核对（G1 归一 + 渲染一致性）

产出：`results_<mode>.json` + 控制台摘要。
只写 `experiments/skeleton_two_tier/` 与 `logs/`；`src/`、`experiments/card_flow/` 只读 import。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "card_flow"))

from dtseek.tasks.corpus import resolve_corpus_files      # noqa: E402
from flow import run_flow                                 # noqa: E402  Tier-1 五段流
from predicates import filter_candidates                  # noqa: E402
from proposers import fake_propose, real_propose          # noqa: E402
from t2core import (TABLE_AUDIT, build_struct, naive_n0, naive_n1,  # noqa: E402
                    run_tier2, run_tier2_struct, select_fragments, t2_render)
from audit import audit_record, diff_record, same_record   # noqa: E402
from contract import g1_problems, normalize                # noqa: E402

SENT_SPLIT = re.compile(r"[。！？!?；]")
HANZI = re.compile(r"[一-鿿]")

#: PREREG §1：**显式按文件名取**（dev-notes/19 ✅ 档），不依赖 fs 顺序、不写 `fs[:N]`。
PICK_FILES = ("lccc_dialogue.txt", "dailychat_dialogue.txt")
#: dev-notes/19 的 ⛔ 档（汉字占比 <50%）—— 命中即 fail-closed。
FORBIDDEN_FILES = ("gsm8k_cot_dialogue.txt", "code_alpaca_dialogue.txt")


# ---- ① 样本 ------------------------------------------------------------------

def _files_by_name() -> dict[str, str]:
    files = resolve_corpus_files()
    by_name = {Path(f).name: f for f in files}
    missing = [n for n in PICK_FILES if n not in by_name]
    if missing:
        raise SystemExit(f"[fail-closed] 缺语料文件 {missing}（PREREG §1：不走退化分支）")
    hit = [n for n in FORBIDDEN_FILES if n in by_name]
    if hit:
        print(f"[语料] 明确不用（dev-notes/19 ⛔ 档）: {hit}")
    return {n: by_name[n] for n in PICK_FILES}


def load_samples(per_file: int = 60, scan_cap: int = 50000) -> list[str]:
    """与 `card_flow` 逐字同规则（同 seed）⇒ 同一批 120 句。"""
    by_name = _files_by_name()
    out: list[str] = []
    for name in PICK_FILES:
        pool: list[str] = []
        with open(by_name[name], encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                for pre in ("用户：", "模型：", "user:", "assistant:"):
                    if line.startswith(pre):
                        line = line[len(pre):]
                for piece in SENT_SPLIT.split(line):
                    s = piece.strip()
                    if 6 <= len(s) <= 40 and len(HANZI.findall(s)) >= 6:
                        pool.append(s)
                        if len(pool) >= scan_cap:
                            break
                if len(pool) >= scan_cap:
                    break
        out.extend(random.Random(42).sample(pool, min(per_file, len(pool))))
    return out


def file_stats() -> list[dict]:
    """PREREG §1：报出所选文件的汉字占比（实测，全文件读取）。"""
    out = []
    for name, path in _files_by_name().items():
        tot = han = 0
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.rstrip("\n")
                tot += len(line)
                han += len(HANZI.findall(line))
        out.append({"file": name, "chars": tot, "han_ratio": round(han / max(tot, 1), 4),
                    "dev_notes_19_ratio": {"lccc_dialogue.txt": 0.854,
                                           "dailychat_dialogue.txt": 0.829}[name]})
    return out


def assert_same_batch(sentences: list[str]) -> dict:
    """PREREG §1：与 `card_flow/results_real.json` 的 120 句逐句比对（不同 ⇒ 退出）。"""
    p = ROOT / "experiments" / "card_flow" / "results_real.json"
    if not p.exists():
        return {"checked": False, "why": "card_flow results_real.json 不存在"}
    ref = json.loads(p.read_text(encoding="utf-8"))["batch_real"]["records"]
    ref_inputs = [r["input"] for r in ref]
    same = ref_inputs == sentences
    if not same:
        raise SystemExit("[fail-closed] 与 card_flow 的 120 句不一致 ⇒ 不许比覆盖率")
    return {"checked": True, "n": len(sentences), "identical": True,
            "baseline_c1": 0.30, "baseline_ok": 36}


# ---- 跑一遍 ------------------------------------------------------------------

def _propose(text: str, mode: str, engine):
    if mode == "real":
        return real_propose(text, engine, sorted(engine.attached), None, None, None)
    return fake_propose(text)


def one_pass(sentences: list[str], mode: str, engine) -> dict:
    recs: list[dict] = []
    n0: list[dict] = []
    n1: list[dict] = []
    t0 = time.perf_counter()
    for text in sentences:
        cands, ci = _propose(text, mode, engine)
        filtered = filter_candidates(text, cands, ci)
        t1 = run_flow(text, cands, ci)
        if t1["status"] == "ok":
            rec = {**t1, "tier": 1, "input": text}
        else:
            t2 = run_tier2(text, filtered)
            rec = {**t2, "input": text, "tier1_stage": t1.get("stage"),
                   "tier1_reason": t1.get("reason")}
        recs.append(rec)
        n0.append({"input": text, **naive_n0(text, filtered)})
        n1.append({"input": text, **naive_n1(text, filtered)})
    return {"records": recs, "naive_n0": n0, "naive_n1": n1,
            "wall_sec": round(time.perf_counter() - t0, 3)}


# ---- 汇总（C1/C2/C5/C6）------------------------------------------------------

def summarize(pass_a: dict, pass_b: dict | None, sentences: list[str]) -> dict:
    recs = pass_a["records"]
    n = len(recs)
    ok = [r for r in recs if r["status"] == "ok"]
    t1_ok = [r for r in ok if r.get("tier") == 1]
    t2_ok = [r for r in ok if r.get("tier") == 2]
    rej = [r for r in recs if r["status"] == "reject"]

    # --- G2/G3 独立审计 + G1 ---
    g2 = g3 = 0
    g2_ex: list[dict] = []
    g3_ex: list[dict] = []
    for r in ok:
        a = audit_record(r, r["input"])
        g2 += len(a["g2"])
        g3 += len(a["g3"])
        if a["g2"] and len(g2_ex) < 5:
            g2_ex.append({"input": r["input"], "viol": a["g2"]})
        if a["g3"] and len(g3_ex) < 5:
            g3_ex.append({"input": r["input"], "viol": a["g3"]})
    g1 = g1_problems(recs)

    # --- G4 ---
    g4 = {"ran": pass_b is not None, "mismatch": None, "diff_example": None}
    if pass_b is not None:
        bad = [i for i, (a, b) in enumerate(zip(recs, pass_b["records"]))
               if not same_record(a, b)]
        g4["mismatch"] = len(bad)
        if bad:
            g4["diff_example"] = diff_record(recs[bad[0]], pass_b["records"][bad[0]])

    # --- C2：Tier-2 首选句式 R0 的可解析率 ---
    t2_reached = [r for r in recs if r.get("tier") == 2 or
                  (r["status"] == "reject" and r.get("stage") == "tier2")]
    constructed = [r for r in t2_reached if r.get("r0") is not None]
    r0_unique = [r for r in constructed if r["r0"].get("parse") == "ok"]
    r0_full = [r for r in constructed
               if r["r0"].get("parse") == "ok" and r["r0"].get("ref") == "ok"
               and r["r0"].get("structure") == "ok"]
    r0_parse_why = Counter(str(r["r0"].get("parse")) for r in constructed)

    # --- 降级链 / 句式统计 ---
    rung_dist = Counter(r.get("rung") for r in t2_ok)
    nfrag_dist = Counter(r.get("n_frag") for r in t2_ok)
    conn = Counter()
    for r in t2_ok:
        for i, f in enumerate(r["structure"]["funcs"]):
            if i == len(r["structure"]["funcs"]) - 1:
                conn["terminal:" + f["text"]] += 1
            elif f["text"] == "":
                conn["gap:∅"] += 1
            else:
                conn["gap:" + ("logic:" if f["cls"] == "logic" else "") + f["text"]] += 1

    # --- 拒答原因分布（C6）---
    rej_reason = Counter(f'{r.get("stage")}:{r.get("reason")}' for r in rej)
    t1_reason_of_t2 = Counter(f'via_tier1:{r.get("tier1_stage")}:{r.get("tier1_reason")}'
                              for r in t2_reached)
    # Tier-1 单独跑的拒因分布（同批、只看 tier==1 成功 + tier1 侧拒绝原因）
    t1_rej = [r for r in recs if r.get("tier") != 1 and r.get("tier1_reason") is not None]

    # --- C5 免费规则 ---
    def _nv(key: str) -> dict:
        xs = pass_a[key]
        oks = [x for x in xs if x["status"] == "ok"]
        return {"n_ok": len(oks), "coverage": round(len(oks) / n, 4) if n else 0.0,
                "reject_reasons": dict(Counter(x.get("reason") for x in xs
                                               if x["status"] == "reject"))}

    return {
        "n_sentences": n,
        "coverage": {
            "tier1_only_ok": len(t1_ok), "tier1_only_c1": round(len(t1_ok) / n, 4),
            "two_tier_ok": len(ok), "two_tier_c1": round(len(ok) / n, 4),
            "delta_pp": round((len(ok) - len(t1_ok)) * 100.0 / n, 2),
            "reject_rate_tier1_only": round((n - len(t1_ok)) / n, 4),
            "reject_rate_two_tier": round(len(rej) / n, 4),
            "n_with_valid_candidate": sum(1 for r in recs if r.get("n_valid", 0) > 0),
        },
        "c2_parse": {
            "n_tier2_reached": len(t2_reached), "n_r0_constructed": len(constructed),
            "n_r0_unique_parse": len(r0_unique), "n_r0_full_pass": len(r0_full),
            "c2a_unique_over_constructed": round(len(r0_unique) / len(constructed), 4)
            if constructed else None,
            "c2b_unique_over_reached": round(len(r0_unique) / len(t2_reached), 4)
            if t2_reached else None,
            "r0_outcome": dict(r0_parse_why),
            "final_tier2_ok": len(t2_ok),
            "final_parse_rate": round(len(t2_ok) / len(constructed), 4)
            if constructed else None,
        },
        "c6_share": {
            "tier2_ok": len(t2_ok), "tier2_share_of_120": round(len(t2_ok) / n, 4),
            "tier2_share_of_ok": round(len(t2_ok) / len(ok), 4) if ok else 0.0,
            "reject_reasons": dict(rej_reason),
            "tier1_reason_of_tier2_records": dict(t1_reason_of_t2),
            "n_tier1_side_reject": len(t1_rej),
        },
        "sentence_shape": {
            "rung_dist": {str(k): v for k, v in sorted(rung_dist.items())},
            "n_frag_dist": {str(k): v for k, v in sorted(nfrag_dist.items())},
            "connectors": dict(conn.most_common()),
            "n_frag_ge2": sum(v for k, v in nfrag_dist.items() if k and k >= 2),
        },
        "guarantees": {"g1_problems": g1, "g2_violations": g2, "g3_violations": g3,
                       "g4": g4, "g2_examples": g2_ex, "g3_examples": g3_ex},
        "naive_c5": {"N0_直接复述": _nv("naive_n0"), "N1_单片段": _nv("naive_n1")},
        "wall_sec": pass_a["wall_sec"],
    }


# ---- 对抗组（PREREG §5：Tier-2 14 条）----------------------------------------

CLASS_INDEX = {
    "pronoun": ("第一人称", "第二人称", "第三人称"),
    "person": tuple(f"人物{i}" for i in range(1, 8)),
    "verb": ("动词",), "number": ("数量",),
    "sentiment": ("中性", "积极", "愤怒", "悲伤"),
    "negation": ("背景", "否定"),
    "idiom": ("非成语", "成语"), "relation": ("反义词", "近义词", "成语"),
}


def _fx(text: str, spans: list[tuple]) -> list[dict]:
    """夹具候选（card_flow `_fixture` 同款）：(card, class_name, s0, e0, theme)。
    走**真实 ② 筛选**（`filter_candidates`），不做任何优待。"""
    from lexicon import type_of as _t
    cands = []
    for i, (card, cname, s0, e0, theme) in enumerate(spans):
        surf = text[s0:e0] if 0 <= s0 < e0 <= len(text) else ""
        cands.append({"cid": f"f{i}", "card": card, "class_id": 1, "class_name": cname,
                      "s0": s0, "e0": e0, "surface": surf, "theme": theme,
                      "type": _t(card, cname, surf)})
    return cands


def _mk(text: str, spans: list[tuple]) -> tuple[list[dict], list[dict]]:
    cands = _fx(text, spans)
    return cands, filter_candidates(text, cands, CLASS_INDEX)


def build_adv_cases() -> list[dict]:
    cases: list[dict] = []

    # T01 题元方向互换（有题元夹具，指派倒序）
    txt = "我打你。"
    cands, filt = _mk(txt, [("pronoun", "第一人称", 0, 1, "施事"),
                            ("verb", "动词", 1, 2, None),
                            ("pronoun", "第二人称", 2, 3, "受事")])
    st = build_struct(txt, select_fragments(filt))
    swap = json.loads(json.dumps(st))
    swap["assign"]["p1"], swap["assign"]["p3"] = st["assign"]["p3"], st["assign"]["p1"]
    cases.append({"id": "T01-题元方向互换", "text": txt, "cands": filt,
                  "struct": swap, "text_out": None,
                  "expect": {"status": "reject", "reason": "tier2_reorder_unjustified"}})

    # T02 无据逻辑词（输入没有 因为）
    st2 = json.loads(json.dumps(st))
    st2["funcs"][0] = {"cls": "logic", "text": "因为"}
    cases.append({"id": "T02-无据逻辑词", "text": txt, "cands": filt, "struct": st2,
                  "text_out": None,
                  "expect": {"status": "reject", "reason": "tier2_logic_no_evidence"}})

    # T03 数字改动
    txt3 = "我3点难过。"
    c3, f3 = _mk(txt3, [("pronoun", "第一人称", 0, 1, None),
                        ("number", "数量", 1, 2, None),
                        ("sentiment", "悲伤", 3, 5, None)])
    st3 = build_struct(txt3, select_fragments(f3))
    bad3 = json.loads(json.dumps(st3))
    d1 = bad3["assign"]["p2"]
    bad3["refs"][d1]["surface"] = "4"
    cases.append({"id": "T03-数字改动", "text": txt3, "cands": f3, "struct": bad3,
                  "text_out": None,
                  "expect": {"status": "reject", "reason": "tier2_span_not_verbatim"}})

    # T04 专名改动
    txt4 = "张伟和王强。"
    c4, f4 = _mk(txt4, [("person", "人物1", 0, 2, None), ("person", "人物2", 3, 5, None)])
    st4 = build_struct(txt4, select_fragments(f4))
    bad4 = json.loads(json.dumps(st4))
    bad4["refs"][bad4["assign"]["p1"]]["surface"] = "李军"
    cases.append({"id": "T04-专名改动", "text": txt4, "cands": f4, "struct": bad4,
                  "text_out": None,
                  "expect": {"status": "reject", "reason": "tier2_span_not_verbatim"}})

    # T05 否定块丢失（毒渲染）
    txt5 = "我不难过。"
    c5, f5 = _mk(txt5, [("pronoun", "第一人称", 0, 1, None),
                        ("negation", "否定", 1, 2, None),
                        ("sentiment", "悲伤", 2, 4, None)])
    st5 = build_struct(txt5, select_fragments(f5))
    full5, _ = t2_render(st5)
    drop5 = full5.replace("不", "", 1)
    cases.append({"id": "T05-否定块丢失", "text": txt5, "cands": f5, "struct": st5,
                  "text_out": drop5,
                  "expect": {"status": "reject", "reason": "tier2_ref_render_mismatch"}})

    # T06 缺块（assign 少一个槽）
    st6 = json.loads(json.dumps(st5))
    st6["assign"].pop("p2")
    st6["evidence"] = list(st6["assign"].values())
    cases.append({"id": "T06-缺块", "text": txt5, "cands": f5, "struct": st6,
                  "text_out": None,
                  "expect": {"status": "reject", "reason": "tier2_schema_missing_slot"}})

    # T07 类型不符（名槽声明成 形）
    st7 = json.loads(json.dumps(st5))
    st7["slots"]["p1"] = "形"
    cases.append({"id": "T07-类型不符", "text": txt5, "cands": f5, "struct": st7,
                  "text_out": None,
                  "expect": {"status": "reject", "reason": "tier2_type_mismatch"}})

    # T08 越界 span
    st8 = json.loads(json.dumps(st5))
    st8["refs"][st8["assign"]["p1"]]["e0"] = 99
    cases.append({"id": "T08-越界", "text": txt5, "cands": f5, "struct": st8,
                  "text_out": None,
                  "expect": {"status": "reject", "reason": "tier2_span_not_verbatim"}})

    # T09 表外功能词
    st9 = json.loads(json.dumps(st5))
    st9["funcs"][0] = {"cls": "formal", "text": "然后"}
    cases.append({"id": "T09-表外功能词", "text": txt5, "cands": f5, "struct": st9,
                  "text_out": None,
                  "expect": {"status": "reject", "reason": "tier2_func_not_in_table"}})

    # T10 用未筛为有效的候选
    st10 = json.loads(json.dumps(st5))
    inv = {"cid": "fx", "card": "person", "class_id": 1, "class_name": "人物1",
           "s0": 0, "e0": 1, "surface": "我", "theme": None, "type": "名",
           "valid": False, "failed": ["F4_has_punct"]}
    st10["assign"]["p1"] = "fx"
    st10["refs"] = {"fx": {k: inv[k] for k in ("cid", "card", "class_id", "class_name",
                                               "s0", "e0", "surface", "type", "theme")}}
    for name in ("p2", "p3"):
        cid = st10["assign"][name]
        st10["refs"][cid] = {k: st5["refs"][cid][k] for k in
                             ("cid", "card", "class_id", "class_name", "s0", "e0",
                              "surface", "type", "theme")}
    st10["evidence"] = list(st10["assign"].values())
    cases.append({"id": "T10-未筛有效候选", "text": txt5, "cands": f5 + [inv],
                  "struct": st10, "text_out": None,
                  "expect": {"status": "reject", "reason": "tier2_not_in_valid_set"}})

    # T11 未知类型进槽（成语不在类型表）
    txt11 = "这事儿莫名其妙。"
    _s = txt11.index("莫名其妙")
    c11, f11 = _mk(txt11, [("idiom", "成语", _s, _s + 4, None)])
    st11 = build_struct(txt11, select_fragments(f11))
    st11["assign"]["p1"] = f11[0]["cid"]
    st11["slots"]["p1"] = "动"
    st11["refs"] = {f11[0]["cid"]: {k: f11[0][k] for k in
                                    ("cid", "card", "class_id", "class_name", "s0",
                                     "e0", "surface", "type", "theme")}}
    st11["evidence"] = [f11[0]["cid"]]
    cases.append({"id": "T11-未知类型进槽", "text": txt11, "cands": f11, "struct": st11,
                  "text_out": None,
                  "expect": {"status": "reject", "reason": "tier2_type_mismatch"}})

    # T12 渲染注入（尾部多吐）
    full12, _ = t2_render(st5)
    cases.append({"id": "T12-渲染注入", "text": txt5, "cands": f5, "struct": st5,
                  "text_out": full12 + "因为",
                  "expect": {"status": "reject", "reason": "tier2_unmapped_output"}})

    # T13 解析歧义（同一文本两种切法 ⇒ 不计分）
    txt13 = "我不难过。"
    c13, f13 = _mk(txt13, [("pronoun", "第一人称", 0, 2, None),
                           ("pronoun", "第一人称", 0, 1, None),
                           ("negation", "否定", 1, 2, None),
                           ("sentiment", "悲伤", 2, 4, None)])
    st13 = build_struct(txt13, select_fragments(f13))
    st13["assign"] = {"p1": "f0", "p2": "f3"}
    st13["slots"] = {"p1": "名", "p2": "形"}
    st13["refs"] = {k: st13["refs"][k] for k in ("f0", "f3")}
    st13["evidence"] = ["f0", "f3"]
    cases.append({"id": "T13-解析歧义", "text": txt13, "cands": f13, "struct": st13,
                  "text_out": None,
                  "expect": {"status": "reject", "reason": "tier2_parse_ambiguous"}})

    # T14 正例（自动路径）
    cases.append({"id": "T14-正例", "text": txt, "cands": filt, "struct": None,
                  "text_out": None, "expect": {"status": "ok", "text": "我打你。"}})
    return cases


def run_adv() -> dict:
    out, wrong = [], []
    for case in build_adv_cases():
        if case["struct"] is None:
            rec = run_tier2(case["text"], case["cands"])
        else:
            rec = run_tier2_struct(case["text"], case["struct"], case["cands"],
                                   case["text_out"])
        exp = case["expect"]
        got = {"id": case["id"], "expect": exp, "status": rec.get("status"),
               "reason": rec.get("reason"), "text": rec.get("text"),
               "has_text_field": "text" in rec, "input": case["text"]}
        if exp["status"] == "reject":
            if rec.get("status") != "reject" or "text" in rec or \
                    rec.get("reason") != exp["reason"]:
                wrong.append({"id": case["id"], "want": exp["reason"],
                              "got": f'{rec.get("status")}:{rec.get("reason")}'
                                     f'{":HAS_TEXT" if "text" in rec else ""}'})
        else:
            if rec.get("status") != "ok" or rec.get("text") != exp["text"] \
                    or "text" not in rec:
                wrong.append({"id": case["id"], "want": f'ok:{exp["text"]}',
                              "got": f'{rec.get("status")}:{rec.get("text")}'})
        out.append(got)

    # Tier-1 复跑（card_flow 的 14 id / 17 次执行）
    import run_card_flow as cf
    t1 = cf.run_adversarial()

    return {
        "tier2": {"n_cases": len(out), "n_pass": len(out) - len(wrong),
                  "mismatch": wrong, "cases": out},
        "tier1_rerun": {k: t1[k] for k in ("n_cases", "must_reject", "reject_hit",
                                           "must_ok", "ok_hit", "g1_reject_rate",
                                           "g1_false_reject", "mismatch")},
        "g1_adv_problems": g1_problems(out),
    }


# ---- 主入口 ------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="selfcheck",
                    choices=["selfcheck", "adv", "real", "official", "all"])
    ap.add_argument("--n", type=int, default=60, help="每文件抽多少句（60 ⇒ 120）")
    ap.add_argument("--out", default=None)
    ap.add_argument("--in", dest="infile", default=None, help="official 模式读哪份结果")
    args = ap.parse_args()

    prereg_mtime = (HERE / "PREREG.md").stat().st_mtime
    t0 = time.time()
    assert prereg_mtime < t0, "PREREG 必须早于首次运行"
    print(f"[prereg] mtime={time.strftime('%F %T', time.localtime(prereg_mtime))} "
          f"< run_start={time.strftime('%F %T', time.localtime(t0))}")
    print(f"[tier2 表审计] {TABLE_AUDIT}")

    modes = (["selfcheck", "adv", "real", "official"] if args.mode == "all"
             else [args.mode])
    result: dict = {"prereg_mtime": prereg_mtime, "run_start": t0,
                    "tier2_table_audit": TABLE_AUDIT, "modes": modes}

    if any(m in modes for m in ("selfcheck", "real", "adv")):
        sentences = load_samples(per_file=args.n)
        print(f"[样本] {len(sentences)} 句；文件 {list(PICK_FILES)}（PREREG §1 显式取名）")
        result["n_sentences"] = len(sentences)
        result["sample_preview"] = sentences[:5]
        result["corpus_files"] = _files_by_name()
        if "real" in modes:
            result["batch_identity"] = assert_same_batch(sentences)
            print(f"[批次一致性] {result['batch_identity']}")
        else:
            result["corpus_stats"] = file_stats()

    if "selfcheck" in modes:
        pa = one_pass(sentences, "fake", None)
        pb = one_pass(sentences, "fake", None)
        summ = summarize(pa, pb, sentences)
        adv = run_adv()
        result["batch_fake"] = summ
        result["adversarial"] = adv
        print(f"[selfcheck/假卡] C1 仅Tier1={summ['coverage']['tier1_only_c1']} → "
              f"两级={summ['coverage']['two_tier_c1']} "
              f"({summ['coverage']['two_tier_ok']}/{summ['n_sentences']})；"
              f"C2a={summ['c2_parse']['c2a_unique_over_constructed']}；"
              f"G1={len(summ['guarantees']['g1_problems'])} G2="
              f"{summ['guarantees']['g2_violations']} G3="
              f"{summ['guarantees']['g3_violations']} G4="
              f"{summ['guarantees']['g4']['mismatch']}")
        print(f"[对抗] Tier-2 {adv['tier2']['n_pass']}/{adv['tier2']['n_cases']}；"
              f"Tier-1 复跑 {adv['tier1_rerun']['reject_hit']}/"
              f"{adv['tier1_rerun']['must_reject']} 必拒、正例 "
              f"{adv['tier1_rerun']['ok_hit']}/{adv['tier1_rerun']['must_ok']}；"
              f"mismatch={adv['tier2']['mismatch'] + adv['tier1_rerun']['mismatch']}")

    if "adv" in modes and "selfcheck" not in modes:
        adv = run_adv()
        result["adversarial"] = adv
        print(f"[对抗] Tier-2 {adv['tier2']['n_pass']}/{adv['tier2']['n_cases']}；"
              f"Tier-1 复跑 {adv['tier1_rerun']['reject_hit']}/"
              f"{adv['tier1_rerun']['must_reject']}；mismatch="
              f"{adv['tier2']['mismatch'] + adv['tier1_rerun']['mismatch']}")

    if "real" in modes:
        from dtseek.tasks.engine import MultiTaskEngine
        print("[真卡] 装载引擎…")
        engine = MultiTaskEngine()
        try:
            engine.attach(ROOT / "checkpoints" / "negation_accept_card.pt")
        except Exception as exc:                       # noqa: BLE001
            print(f"[warn] 否定卡挂载失败，本轮 4 张卡：{type(exc).__name__}: {exc}")
        print(f"[真卡] 已挂载 {sorted(engine.attached)}；跑第 1 遍…")
        pa = one_pass(sentences, "real", engine)
        print(f"[真卡] 第 1 遍 {pa['wall_sec']}s；跑第 2 遍（G4 含重推理）…")
        pb = one_pass(sentences, "real", engine)
        print(f"[真卡] 第 2 遍 {pb['wall_sec']}s")
        summ = summarize(pa, pb, sentences)
        result["batch_real"] = summ
        result["records"] = pa["records"]
        result["records_pass2"] = pb["records"]
        result["naive_n0"] = pa["naive_n0"]
        result["naive_n1"] = pa["naive_n1"]
        cv = summ["coverage"]
        print(f"[real] 仅Tier-1 C1={cv['tier1_only_c1']} ({cv['tier1_only_ok']}/"
              f"{summ['n_sentences']}) 拒答率={cv['reject_rate_tier1_only']} → "
              f"两级 C1={cv['two_tier_c1']} ({cv['two_tier_ok']}/"
              f"{summ['n_sentences']}) 拒答率={cv['reject_rate_two_tier']}")
        print(f"       C2a={summ['c2_parse']['c2a_unique_over_constructed']} "
              f"C2b={summ['c2_parse']['c2b_unique_over_reached']} "
              f"R0 结局={summ['c2_parse']['r0_outcome']}")
        print(f"       G1={len(summ['guarantees']['g1_problems'])} "
              f"G2={summ['guarantees']['g2_violations']} "
              f"G3={summ['guarantees']['g3_violations']} "
              f"G4={summ['guarantees']['g4']}")
        print(f"       C6 拒因={summ['c6_share']['reject_reasons']}")
        print(f"       C5 N0={summ['naive_c5']['N0_直接复述']} "
              f"N1={summ['naive_c5']['N1_单片段']}")
        print(f"       句式={summ['sentence_shape']}")

    if "official" in modes:
        import official as off
        recs = result.get("records")
        if recs is None:
            src = Path(args.infile) if args.infile else HERE / "results_real.json"
            if not src.exists():
                raise SystemExit(f"[fail-closed] official 模式需要真卡记录：{src} 不存在")
            data = json.loads(src.read_text(encoding="utf-8"))
            recs = data.get("records") or data.get("batch_real", {}).get("records", [])
            result["official_src"] = str(src)
        result["g1_contract_demo"] = off.g1_contract_demo()
        result["official_crosscheck"] = off.crosscheck(recs)
        result["g1_on_real_records"] = g1_problems(recs)
        print(f"[G1 归一] 原始带 text: {result['g1_contract_demo']['raw']} → "
              f"归一后含 text: {result['g1_contract_demo']['normalized_has_text']}")
        print(f"[官方交叉] {result['official_crosscheck']}")
        print(f"[G1 实记录] {result['g1_on_real_records']}")

    out = Path(args.out) if args.out else HERE / f"results_{args.mode}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[写出] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
