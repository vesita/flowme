#!/usr/bin/env python3
"""端到端卡片流测试驱动 —— 执行 `PREREG.md` 里跑前写死的判据。

    uv run python experiments/card_flow/run_card_flow.py --mode fake   # 假卡跑通管线 + 自检
    uv run python experiments/card_flow/run_card_flow.py --mode real   # 真卡推理（小规模）
    uv run python experiments/card_flow/run_card_flow.py --mode adv    # 对抗组 14 条
    uv run python experiments/card_flow/run_card_flow.py --mode all    # 三者都跑

产出：`experiments/card_flow/results_<mode>.json` + 控制台摘要。
**只写 experiments/card_flow/ 与 /tmp**（logs/ 被他方会话独占，日志走 --log 落本目录）。
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

from dtseek.tasks.corpus import resolve_corpus_files  # noqa: E402

from flow import run_flow  # noqa: E402
from lexicon import type_of  # noqa: E402
from proposers import fake_propose  # noqa: E402
from render import render_structure  # noqa: E402
from skeletons import BY_ID, audit as skel_audit  # noqa: E402

SENT_SPLIT = re.compile(r"[。！？!?；]")
HANZI = re.compile(r"[一-鿿]")
PICK_FILES = ("lccc_dialogue.txt", "dailychat_dialogue.txt")


# ---- ① 样本（PREREG §1 的抽样规则，跑前写死）--------------------------------

def load_samples(per_file: int = 60, scan_cap: int = 50000) -> list[str]:
    files = resolve_corpus_files()
    by_name = {Path(f).name: f for f in files}
    chosen = [by_name[n] for n in PICK_FILES if n in by_name]
    if len(chosen) < len(PICK_FILES):                      # 同规则退化：取排序前两个
        chosen = sorted(f for f in files if "dialogue" in Path(f).name)[:2]
    out: list[str] = []
    for path in chosen:
        pool: list[str] = []
        with open(path, encoding="utf-8") as fh:
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


# ---- G2 / G3 的**独立事后复核**（不复用流内检查的结论）------------------------

def audit_record(rec: dict, src_text: str) -> dict:
    """对一条 ok 记录独立复核 G2（零信息新增）与 G3（引用映射完整）。"""
    g2: list[str] = []
    g3: list[str] = []
    if rec.get("status") != "ok":
        return {"g2": g2, "g3": g3}
    skel = BY_ID[rec["skeleton"]]
    struct = rec["structure"]
    out_text = rec["text"]
    mapping = rec["mapping"]

    # G3-a：重渲染必须逐字相同（结构 → 文本唯一确定）
    if render_structure(struct, skel) != out_text:
        g3.append("rerender_mismatch")

    # G3-b：映射区间 + 骨架字面必须**恰好铺满**输出（有洞 = 有内容没引用）
    covered: list[tuple[int, int]] = []
    ptr = 0
    seen_slots = []
    for kind, tok in skel.lit_tokens():
        if kind == "lit":
            covered.append((ptr, ptr + len(tok)))
            ptr += len(tok)
        else:
            me = [m for m in mapping if m["slot"] == tok]
            if len(me) != 1:
                g3.append("slot_mapping_missing")
                continue
            seen_slots.append(tok)
            if me[0]["pos"] != ptr:
                g3.append("pos_mismatch")
            covered.append((ptr, ptr + len(me[0]["frag"])))
            ptr += len(me[0]["frag"])
    if ptr != len(out_text):
        g3.append("tail_uncovered")
    for a, b in zip(covered, covered[1:]):
        if a[1] != b[0]:
            g3.append("coverage_gap")
            break
    if len(seen_slots) != len(skel.slots) or len(mapping) != len(skel.slots):
        g3.append("mapping_count_mismatch")

    valid_map = {c["cid"]: c for c in rec["candidates"] if c["valid"]}
    for m in mapping:
        # G3-c：链 —— 片段 ↦ 结构 ref ↦ 原始 span
        ref = struct["refs"].get(m["ref"])
        if ref is None or [ref["s0"], ref["e0"]] != m["span"] or ref["surface"] != m["frag"]:
            g3.append("chain_broken")
        if out_text[m["pos"]:m["pos"] + len(m["frag"])] != m["frag"]:
            g3.append("frag_not_at_pos")
        # G2：逐字回溯 + 必须是"被筛为有效"的候选
        s0, e0 = m["span"]
        if not (0 <= s0 < e0 <= len(src_text)) or src_text[s0:e0] != m["frag"]:
            g2.append("not_verbatim")
        c = valid_map.get(m["ref"])
        if c is None or (c["s0"], c["e0"], c["surface"]) != (s0, e0, m["frag"]):
            g2.append("not_filtered_valid")
        if not c or c["failed"]:
            g2.append("candidate_was_invalid")
    return {"g2": g2, "g3": g3}


# ---- 批量跑 -----------------------------------------------------------------

def run_batch(sentences: list[str], mode: str) -> dict:
    timers: dict[str, list[float]] = {}
    steps: dict[str, list[int]] = {}
    segs: dict[str, list[int]] = {}
    specs_info: dict[str, dict] = {}
    engine = None
    if mode == "real":
        from dtseek.tasks.engine import MultiTaskEngine
        engine = MultiTaskEngine()
        try:
            engine.attach(ROOT / "checkpoints" / "negation_accept_card.pt")
        except Exception as exc:                            # noqa: BLE001
            print(f"[warn] 否定卡挂载失败，本轮用 4 张卡：{type(exc).__name__}: {exc}")
        specs_info = {c: {"max_steps": engine.specs[c].max_steps,
                          "max_len": engine.specs[c].max_len,
                          "segment_policy": engine.specs[c].segment_policy,
                          "emission": engine.specs[c].emission}
                      for c in sorted(engine.attached)}

    records: list[dict] = []
    g2_viol = g3_viol = g4_mismatch = 0
    g2_ex: list[dict] = []
    g3_ex: list[dict] = []
    t_start = time.perf_counter()

    for text in sentences:
        if mode == "real":
            cands, class_index = _real(text, engine, timers, steps, segs)
        else:
            cands, class_index = fake_propose(text)
        rec = run_flow(text, cands, class_index)
        rec["input"] = text
        if rec["status"] == "ok":
            a = audit_record(rec, text)
            g2_viol += len(a["g2"])
            g3_viol += len(a["g3"])
            if a["g2"] and len(g2_ex) < 5:
                g2_ex.append({"input": text, "viol": a["g2"], "mapping": rec["mapping"]})
            if a["g3"] and len(g3_ex) < 5:
                g3_ex.append({"input": text, "viol": a["g3"]})
            # G4：同输入再跑一次，逐字比较（第二遍不计入前向计时）
            cands2, ci2 = (_real(text, engine, None, None, None) if mode == "real"
                           else fake_propose(text))
            rec2 = run_flow(text, cands2, ci2)
            if rec2.get("status") != "ok" or rec2["text"].encode() != rec["text"].encode():
                g4_mismatch += 1
        records.append(rec)

    wall = time.perf_counter() - t_start
    n = len(records)
    n_ok = sum(1 for r in records if r["status"] == "ok")
    n_valid_nonzero = sum(1 for r in records if r.get("n_valid", 0) > 0)
    n_compose_ok = sum(1 for r in records
                       if r["status"] == "ok" or r.get("stage") in ("render", "deref"))
    by_stage: dict[str, Counter] = {}
    for r in records:
        if r["status"] == "reject":
            by_stage.setdefault(r["stage"], Counter())[r["reason"]] += 1
    reason_all: Counter = Counter()
    for r in records:
        if r["status"] == "reject":
            if r["stage"] == "compose":
                reason_all[r["reason"]] += 1
                have = sorted({c["type"] for c in r.get("candidates", [])
                               if c.get("valid") and c.get("type")})
                reason_all["have:" + (",".join(have) or "-")] += 1
                if r["reason"] == "type_inventory":
                    for t in r.get("detail", {}).get("missing", []):
                        reason_all[f"type:{t}"] += 1
            else:
                reason_all[f"{r['stage']}:{r['reason']}"] += 1

    timers_out = {}
    for card, xs in timers.items():
        xs_sorted = sorted(xs)
        info = specs_info.get(card, {})
        seg = sum(segs.get(card, [1])) / max(1, len(xs))
        budget = seg * info.get("max_steps", 1)             # 每句的解码步预算（段数 × max_steps）
        timers_out[card] = {
            "n": len(xs),
            "mean_ms": round(sum(xs) / len(xs), 3),
            "p50_ms": round(xs_sorted[len(xs_sorted) // 2], 3),
            "max_ms": round(max(xs), 3),
            "mean_segments": round(seg, 2),
            "mean_emitted_anchors": round(sum(steps[card]) / len(steps[card]), 2),
            "decode_step_budget": round(budget, 1),
            "ms_per_decode_step": round(sum(xs) / len(xs) / max(1.0, budget), 4),
            "spec": info,
        }
    return {
        "mode": mode,
        "n_sentences": n,
        "n_ok": n_ok,
        "n_with_valid": n_valid_nonzero,
        "n_compose_ok": n_compose_ok,
        "reject_rate": round((n - n_ok) / n, 4) if n else 0.0,
        "coverage_c1": round(n_compose_ok / n, 4) if n else 0.0,
        "coverage_c2": round(n_compose_ok / n_valid_nonzero, 4) if n_valid_nonzero else 0.0,
        "reject_by_stage": {k: dict(v) for k, v in by_stage.items()},
        "reject_reasons": dict(reason_all),
        "g2_violations": g2_viol,
        "g3_violations": g3_viol,
        "g4_mismatches": g4_mismatch,
        "g2_examples": g2_ex,
        "g3_examples": g3_ex,
        "forward_ms_per_sentence": {k: v for k, v in timers_out.items()},
        "card_specs": specs_info,
        "wall_sec": round(wall, 2),
        "records": records,
    }


def _real(text, engine, timers, steps, segs):
    from proposers import real_propose
    return real_propose(text, engine, sorted(engine.attached), timers, steps, segs)


# ---- 对抗组（PREREG §3 的 14 条，预期跑前写死）-------------------------------

def _ref(c: dict) -> dict:
    return {"cid": c["cid"], "card": c["card"], "class_id": c["class_id"],
            "class_name": c["class_name"], "s0": c["s0"], "e0": c["e0"],
            "surface": c["surface"], "type": c["type"], "theme": c.get("theme")}


def _struct(skel_id: str, assign: dict[str, str], cands: dict[str, dict]) -> dict:
    return {"skeleton": skel_id, "assign": dict(assign),
            "refs": {cid: _ref(cands[cid]) for cid in assign.values()},
            "evidence": list(assign.values())}


def _fixture(text: str, spans: list[tuple[str, str, int, int, str, str | None]],
             extra_class_index: dict[str, tuple[str, ...]] | None = None
             ) -> tuple[list[dict], dict[str, tuple[str, ...]]]:
    """手工夹具候选：(card, class_name, s0, e0, surface, theme)。"""
    from lexicon import type_of as _t

    cands = []
    for i, (card, cname, s0, e0, surf, theme) in enumerate(spans):
        cands.append({"cid": f"f{i}", "card": card, "class_id": 1, "class_name": cname,
                      "s0": s0, "e0": e0, "surface": surf, "theme": theme,
                      "type": _t(card, cname, surf)})
    idx = {"verb": ("动词",), "number": ("数量",),
           "pronoun": ("第一人称", "第二人称", "第三人称"),
           "person": tuple(f"人物{i}" for i in range(1, 8)),
           "sentiment": ("中性", "积极", "愤怒", "悲伤"),
           "negation": ("背景", "否定"),
           "idiom": ("非成语", "成语")}
    idx.update(extra_class_index or {})
    return cands, idx


def build_cases() -> list[dict]:
    """每条：id / 输入 / 候选 / 是否 override / poison / 预期。"""
    cases: list[dict] = []

    # --- 题元方向 A1/A2/A3/A14 ---
    for text, me, you in (("我打你。", "我", "你"), ("你打我。", "你", "我")):
        i_me, i_you = text.index(me), text.index(you)
        i_v = text.index("打")
        base = [("pronoun", "第一人称", i_me, i_me + 1, me, None),
                ("verb", "动词", i_v, i_v + 1, "打", None),
                ("pronoun", "第二人称", i_you, i_you + 1, you, None)]
        themed = [(c[0], c[1], c[2], c[3], c[4],
                   ("施事" if c[4] == me else "受事" if c[4] == you else None))
                  for c in base]
        for tag, spans in (("themed", themed), ("unthemed", base)):
            cands, idx = _fixture(text, spans)
            by = {c["cid"]: c for c in cands}
            ok_struct = _struct("S13", {"n1": "f0", "v1": "f1", "n2": "f2"}, by)
            swap = dict(ok_struct)
            swap = {"skeleton": "S13", "assign": {"n1": "f2", "v1": "f1", "n2": "f0"},
                    "refs": ok_struct["refs"], "evidence": ok_struct["evidence"]}
            cid = "A1" if text.startswith("我") and tag == "themed" else \
                  "A2" if text.startswith("我") else "A3"
            if tag == "themed":
                cases.append({"id": f"{cid}-{tag}-swap", "text": text, "cands": cands,
                              "idx": idx, "override": swap, "poison": None,
                              "expect": {"status": "reject", "reason": "theme_conflict"}})
                cases.append({"id": f"{cid}-{tag}-ok", "text": text, "cands": cands,
                              "idx": idx, "override": ok_struct, "poison": None,
                              "expect": {"status": "ok", "text": text}})
            else:
                cases.append({"id": f"{cid}-{tag}-swap", "text": text, "cands": cands,
                              "idx": idx, "override": swap, "poison": None,
                              "expect": {"status": "reject", "reason": "reorder_unjustified"}})

    # --- 逻辑词 A4/A5/A9 ---
    cands, idx = fake_propose("我难过，你高兴。")
    by = {c["cid"]: c for c in cands}
    r = run_flow("我难过，你高兴。", cands, idx)
    assert r["status"] == "ok", f"对抗组夹具自检失败: A4 基线 {r}"
    st = dict(r["structure"]); st["skeleton"] = "S20"      # 因为…（输入无因果依据）
    cases.append({"id": "A4-因为无据", "text": "我难过，你高兴。", "cands": cands, "idx": idx,
                  "override": st, "poison": None,
                  "expect": {"status": "reject", "reason": "logic_no_evidence"}})

    cands5, idx5 = fake_propose("我难过，但你高兴。")
    r5 = run_flow("我难过，但你高兴。", cands5, idx5)
    assert r5["status"] == "ok", f"对抗组夹具自检失败: A5 基线 {r5}"
    st5 = dict(r5["structure"]); st5["skeleton"] = "S18"    # 但：输入有据
    cases.append({"id": "A5-但有据", "text": "我难过，但你高兴。", "cands": cands5, "idx": idx5,
                  "override": st5, "poison": None,
                  "expect": {"status": "ok", "text": "我难过，但你高兴。"}})

    cands9, idx9 = _fixture("我打你。", [
        ("pronoun", "第一人称", 0, 1, "我", None),
        ("verb", "动词", 1, 2, "打", None),
        ("pronoun", "第二人称", 2, 3, "你", None)])
    by9 = {c["cid"]: c for c in cands9}
    cases.append({"id": "A9-要无据", "text": "我打你。", "cands": cands9, "idx": idx9,
                  "override": _struct("S21", {"n1": "f0", "v1": "f1"}, by9), "poison": None,
                  "expect": {"status": "reject", "reason": "logic_no_evidence"}})

    # --- 数字 / 专名 A6/A7 ---
    cands6, idx6 = fake_propose("我3点难过。")
    cands6 = list(cands6) + [{"cid": "f9", "card": "number", "class_id": 1,
                              "class_name": "数量", "s0": 1, "e0": 2, "surface": "3",
                              "theme": None, "type": type_of("number", "数量", "3")}]
    r6 = run_flow("我3点难过。", cands6, {**idx6, "number": ("数量",)})
    assert r6["status"] == "ok", f"对抗组夹具自检失败: A6 基线 {r6}"
    st6 = json.loads(json.dumps(r6["structure"]))
    d1 = st6["assign"]["d1"]
    st6["refs"][d1]["surface"] = "4"                        # 数字被改动
    cases.append({"id": "A6-数字改动", "text": "我3点难过。", "cands": cands6,
                  "idx": {**idx6, "number": ("数量",)}, "override": st6, "poison": None,
                  "expect": {"status": "reject", "reason": "span_not_verbatim"}})

    cands7, idx7 = fake_propose("张伟和王强。")
    r7 = run_flow("张伟和王强。", cands7, idx7)
    assert r7["status"] == "ok", f"对抗组夹具自检失败: A7 基线 {r7}"
    st7 = json.loads(json.dumps(r7["structure"]))
    st7["refs"][st7["assign"]["n1"]]["surface"] = "李军"     # 专名被改动
    cases.append({"id": "A7-专名改动", "text": "张伟和王强。", "cands": cands7, "idx": idx7,
                  "override": st7, "poison": None,
                  "expect": {"status": "reject", "reason": "span_not_verbatim"}})

    # --- 否定丢失 A8 ---
    cands8, idx8 = fake_propose("我不难过。")
    r8 = run_flow("我不难过。", cands8, idx8)
    assert r8["status"] == "ok", f"对抗组夹具自检失败: A8 基线 {r8}"
    cases.append({"id": "A8-否定丢失", "text": "我不难过。", "cands": cands8, "idx": idx8,
                  "override": r8["structure"], "poison": {"drop": "g1"},
                  "expect": {"status": "reject", "reason": "ref_render_mismatch"}})

    # --- 缺块 / 类型不符 / 越界 / 注入 A10/A11/A12/A13 ---
    cands10, idx10 = fake_propose("我难过。")
    r10 = run_flow("我难过。", cands10, idx10)
    assert r10["status"] == "ok", f"对抗组夹具自检失败: A10 基线 {r10}"
    st10 = json.loads(json.dumps(r10["structure"]))
    st10["assign"].pop("a1")                                 # 缺块
    cases.append({"id": "A10-缺块", "text": "我难过。", "cands": cands10, "idx": idx10,
                  "override": st10, "poison": None,
                  "expect": {"status": "reject", "reason": "schema_missing_slot"}})

    st11 = json.loads(json.dumps(r10["structure"]))
    a, b = st11["assign"]["n1"], st11["assign"]["a1"]
    st11["assign"]["n1"], st11["assign"]["a1"] = b, a         # 形候选进名槽
    cases.append({"id": "A11-类型不符", "text": "我难过。", "cands": cands10, "idx": idx10,
                  "override": st11, "poison": None,
                  "expect": {"status": "reject", "reason": "type_mismatch"}})

    cands12 = list(cands10) + [{"cid": "fX", "card": "pronoun", "class_id": 1,
                                "class_name": "第一人称", "s0": 0, "e0": 99,
                                "surface": "我难过。", "theme": None,
                                "type": type_of("pronoun", "第一人称", "我难过。")}]
    st12 = _struct(r10["skeleton"],
                   {**r10["structure"]["assign"], "n1": "fX"},
                   {**{c["cid"]: c for c in cands10}, "fX": cands12[-1]})
    cases.append({"id": "A12-越界", "text": "我难过。", "cands": cands12, "idx": idx10,
                  "override": st12, "poison": None,
                  "expect": {"status": "reject", "reason": "span_not_verbatim"}})

    cases.append({"id": "A13-注入", "text": "我难过。", "cands": cands10, "idx": idx10,
                  "override": r10["structure"], "poison": {"inject": "因为"},
                  "expect": {"status": "reject", "reason": "unmapped_output"}})

    # --- 正例 A14 ---
    cands14, idx14 = fake_propose("我难过。")
    cases.append({"id": "A14-正例", "text": "我难过。", "cands": cands14, "idx": idx14,
                  "override": None, "poison": None,
                  "expect": {"status": "ok", "text": "我难过。"}})
    return cases


def run_adversarial() -> dict:
    out = []
    must_reject = must_ok = 0
    reject_hit = ok_hit = 0
    wrong_reason: list[dict] = []
    for case in build_cases():
        rec = run_flow(case["text"], case["cands"], case["idx"],
                       struct_override=case["override"], poison=case["poison"])
        exp = case["expect"]
        got = {"id": case["id"], "expect": exp, "status": rec.get("status"),
               "reason": rec.get("reason"), "stage": rec.get("stage"),
               "text": rec.get("text"), "input": case["text"],
               "has_text_field": "text" in rec}
        if exp["status"] == "reject":
            must_reject += 1
            if rec.get("status") == "reject" and "text" not in rec:
                if rec.get("reason") == exp["reason"]:
                    reject_hit += 1
                else:
                    wrong_reason.append({"id": case["id"], "want": exp["reason"],
                                         "got": rec.get("reason")})
            else:
                wrong_reason.append({"id": case["id"], "want": exp["reason"],
                                     "got": rec.get("status")})
        else:
            must_ok += 1
            if rec.get("status") == "ok" and rec.get("text") == exp["text"] \
                    and "text" in rec:
                ok_hit += 1
            else:
                wrong_reason.append({"id": case["id"], "want": f"ok:{exp['text']}",
                                     "got": f"{rec.get('status')}:{rec.get('text')}"})
        out.append(got)
    return {
        "n_cases": len(out),
        "must_reject": must_reject, "reject_hit": reject_hit,
        "must_ok": must_ok, "ok_hit": ok_hit,
        "g1_reject_rate": round(reject_hit / must_reject, 4) if must_reject else None,
        "g1_false_reject": must_ok - ok_hit,
        "mismatch": wrong_reason,
        "cases": out,
    }


# ---- 主入口 -----------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="fake", choices=["fake", "real", "adv", "all"])
    ap.add_argument("--n", type=int, default=60, help="每个语料文件抽多少句（默认 60 ⇒ 120）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    prereg_mtime = (HERE / "PREREG.md").stat().st_mtime
    t0 = time.time()
    assert prereg_mtime < t0, "PREREG 必须早于首次运行"
    print(f"[prereg] mtime={time.strftime('%F %T', time.localtime(prereg_mtime))} "
          f"< run_start={time.strftime('%F %T', time.localtime(t0))}")

    sk = skel_audit()
    print(f"[骨架] {sk['n_skeletons']} 条；逻辑词门禁骨架: {list(sk['logic_gated'])}")

    modes = ["fake", "real", "adv"] if args.mode == "all" else [args.mode]
    result: dict = {"prereg_mtime": prereg_mtime, "run_start": t0, "skeletons": sk}

    if "adv" in modes:
        adv = run_adversarial()
        result["adversarial"] = adv
        print(f"[对抗] {adv['reject_hit']}/{adv['must_reject']} 必拒命中；"
              f"正例 {adv['ok_hit']}/{adv['must_ok']}；错因 {adv['mismatch']}")

    batch_modes = [m for m in modes if m in ("fake", "real")]
    if batch_modes:
        sentences = load_samples(per_file=args.n)
        result["n_sentences_sampled"] = len(sentences)
        result["sample_preview"] = sentences[:5]
        print(f"[样本] {len(sentences)} 句（规则见 PREREG §1）")
        for m in batch_modes:
            res = run_batch(sentences, m)
            key = f"batch_{m}"
            result[key] = res
            print(f"[{m}] ok={res['n_ok']}/{res['n_sentences']} "
                  f"C1={res['coverage_c1']} C2={res['coverage_c2']} "
                  f"拒答率={res['reject_rate']} G2={res['g2_violations']} "
                  f"G3={res['g3_violations']} G4={res['g4_mismatches']} "
                  f"wall={res['wall_sec']}s")
            print(f"    拒因: {res['reject_reasons']}")
            if res["forward_ms_per_sentence"]:
                print(f"    前向: {res['forward_ms_per_sentence']}")

    out = Path(args.out) if args.out else HERE / f"results_{args.mode}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[写出] {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
