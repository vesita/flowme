"""P11 驱动：跑指针结构化记录、L1/L2 解释、反例注入、对照臂、确定性、覆盖率。纯 CPU，不训练。

    CUDA_VISIBLE_DEVICES="" uv run python experiments/pointer_explain/run_ptr.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import ptr_explain as pe  # noqa: E402
import explain_card as ec  # noqa: E402


def jd(o) -> str:
    return json.dumps(o, sort_keys=True, ensure_ascii=False)


def build() -> tuple[dict, list]:
    """跑一遍完整流水线，返回 (可比较的 payload, PtrSrc 列表)。"""
    srcs = pe.load_pointer_records()
    p: dict = {"records": [], "l1": [], "r1": {}, "r2": {}, "l2": {},
               "arms": {}, "counts": {}}
    for ptr in srcs:
        rec = pe.struct_record(ptr)
        card = pe.l1_card(ptr)
        r1 = (pe.r1_problems(card, rec, ptr) if card.get("explainable")
              else [f"未覆盖：{card.get('why', '')}"])
        r2 = pe.r2_problems(rec)
        l2 = pe.l2_verdict(pe.l2_attempts(ptr))
        p["records"].append(rec)
        p["l1"].append({k: v for k, v in card.items() if k != "input"})
        p["r1"][ptr.rid] = r1
        p["r2"][ptr.rid] = r2
        p["l2"][ptr.rid] = {"given": l2["given"], "reason": l2["reason"],
                            "detail": l2.get("detail", ""),
                            "first_problem": l2.get("first_problem", ""),
                            "attempts": [{"name": a["name"], "text": a["text"],
                                          "ok": a.get("ok"), "na": a.get("na", False),
                                          "c1": a.get("c1"), "c2": a.get("c2"),
                                          "c3": a.get("c3"),
                                          "n_problems": a.get("n_problems", 0),
                                          "problems": a.get("problems", []),
                                          "fail": a.get("fail", "")}
                                         for a in l2["attempts"]]}
        p["arms"][ptr.rid] = {
            "G-free_strict": pe.g_free_strict(ptr),
            "G-free_generous": pe.g_free_generous(ptr),
            "B-echo": pe.b_echo(ptr),
        }
    p["counts"] = {"n_records": len(srcs),
                   "kinds": dict(Counter(r["kind"] for r in p["records"]))}
    return p, srcs


def main() -> int:
    prereg = HERE / "PREREG.md"
    now = time.time()
    assert prereg.exists(), "PREREG.md 必须先落"
    assert prereg.stat().st_mtime < now, "PREREG mtime 必须早于首跑"
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "必须纯 CPU（CUDA_VISIBLE_DEVICES=\"\"）"
    log_lines: list[str] = []

    def log(m: str) -> None:
        print(m, flush=True)
        log_lines.append(m)

    log(f"[prereg] mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(prereg.stat().st_mtime))}"
        f" < run={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(now))}")
    gate = pe.ptr_gate()
    log(f"[gate] 指针解释骨架表 rule_a+slot_schema 违例 = {len(gate)} {gate}")

    # ---- 1. 第一遍
    t0 = time.time()
    p1, srcs1 = build()
    pred1 = dict(ec.PRED_CALLS), dict(ec.PRED_NONEMPTY)
    log(f"[pass1] {p1['counts']} wall={time.time() - t0:.2f}s")

    # ---- 2. 第二遍（确定性）
    ec.PRED_CALLS.clear()
    ec.PRED_NONEMPTY.clear()
    p2, _srcs2 = build()
    same = jd(p1) == jd(p2)
    log(f"[R5-确定性] 两遍 payload 全等 = {same}（长度 {len(jd(p1))} vs {len(jd(p2))}）")
    if not same:
        a, b = jd(p1), jd(p2)
        k = next(i for i in range(min(len(a), len(b))) if a[i] != b[i])
        log(f"       首个差异位置 {k}: {a[max(0, k - 120):k + 120]!r} || {b[max(0, k - 120):k + 120]!r}")

    recs = p1["records"]
    n = len(recs)
    covered, cov_why = [], {}
    for i, rec in enumerate(recs):
        rid = rec["rid"]
        ok = (rec["replay_match"] and p1["l1"][i].get("explainable")
              and not p1["r1"][rid] and not p1["r2"][rid])
        if ok:
            covered.append(rid)
        else:
            cov_why[rid] = {"replay": rec["replay_match"],
                            "l1": p1["l1"][i].get("why") or p1["l1"][i].get("explainable"),
                            "r1": p1["r1"][rid][:2], "r2": p1["r2"][rid][:2]}

    # ---- 3. R1 / R2
    r1_bad = {k: v for k, v in p1["r1"].items() if v}
    r2_bad = {k: v for k, v in p1["r2"].items() if v}
    log(f"[R0] 覆盖 {len(covered)}/{n}；未覆盖原因 {json.dumps(cov_why, ensure_ascii=False)}")
    log(f"[R1] 违例记录 {len(r1_bad)}，问题总数 {sum(len(v) for v in r1_bad.values())}")
    for k, v in list(r1_bad.items())[:3]:
        log(f"     {k}: {v[:2]}")
    log(f"[R2] 不一致记录 {len(r2_bad)}，问题总数 {sum(len(v) for v in r2_bad.values())}")
    for k, v in list(r2_bad.items())[:3]:
        log(f"     {k}: {v[:2]}")

    # ---- 4. R3 反例注入
    inj: dict[str, dict] = {}
    rid_idx = {r["rid"]: i for i, r in enumerate(recs)}
    good = [r for r in recs if r["rid"] in covered]

    def run_inj(name, fn):
        caught = 0
        reps: dict[str, str] = {}
        cats: Counter = Counter()
        for rec in recs:
            if rec["rid"] not in covered:
                continue
            card = p1["l1"][rid_idx[rec["rid"]]]
            _c, _r, _p, probs = fn(card, rec, srcs1[rid_idx[rec["rid"]]])
            if probs:
                caught += 1
            for pb in probs:
                c = pb.split("：")[0].split(":")[0]
                cats[c] += 1
                reps.setdefault(c, pb)
        tot = len(good)
        inj[name] = {"injected": tot, "caught": caught, "missed": tot - caught,
                     "by_category": dict(cats), "representative": reps}
        log(f"[R3-{name}] 注入 {tot}，被抓 {caught}，漏 {tot - caught}；类别 {dict(cats)}")
        for c, m in reps.items():
            log(f"     - {c}: {m[:160]}")

    run_inj("①改 chosen span", pe.inject_span)
    run_inj("②改分数使分数序矛盾", pe.inject_score)
    run_inj("③加输入没有的内容词", pe.inject_word)
    r3_miss = sum(v["missed"] for v in inj.values())

    # ---- 5. R4 对照臂 + R5 平凡基线
    arms: dict[str, dict] = {}
    for arm in ("G-free_strict", "G-free_generous", "B-echo"):
        viol = {k: v[arm] for k, v in p1["arms"].items() if v[arm]}
        cats = Counter(pb.split("：")[0].split(":")[0]
                       for v in p1["arms"].values() for pb in v[arm])
        arms[arm] = {"n_records": n, "n_violation_records": len(viol),
                     "n_problems": sum(len(v) for v in p1["arms"].values()
                                       for v in [v[arm]]),
                     "by_category": dict(cats),
                     "example": (next(iter(viol.values()))[:3] if viol else [])}
        log(f"[R4/R5-{arm}] 违例记录 {len(viol)}/{n}，问题总数 "
            f"{arms[arm]['n_problems']}，类别 {dict(cats)}")
        if viol:
            log(f"     首条: {arms[arm]['example'][0][:170]}")

    # ---- 6. 不可得字段
    FIELD_TABLE = ["kind", "chosen.card", "chosen.span", "chosen.span_text", "chosen.rank",
                   "chosen.score_replay", "chosen.score_original", "candidates", "scores_kind",
                   "ref_map", "evidence", "L1解释文本", "L2理由", "差异事实"]
    unavail: list[dict] = []
    for i, rec in enumerate(recs):
        rid = rec["rid"]
        l2 = p1["l2"][rid]
        alt = any(c["eligible"] and
                  any(o["card"] == c["card"] and o["cid"] != c["cid"] for o in rec["candidates"])
                  for c in rec["chosen"])
        miss = ["chosen.score_original", "L2理由"] + ([] if alt else ["差异事实"])
        if not p1["l1"][i].get("explainable"):
            miss.append("L1解释文本")
        for f in miss:
            unavail.append({"rid": rid, "field": f})
    un_cnt = Counter(x["field"] for x in unavail)
    ratio = len(unavail) / (len(FIELD_TABLE) * n)
    log(f"[R0-不可得] 字段表 {len(FIELD_TABLE)}×{n}={len(FIELD_TABLE) * n}，"
        f"不可得 {len(unavail)}（{ratio:.4f}）；分布 {dict(un_cnt)}")

    # ---- 7. R6
    l2_ok = [k for k, v in p1["l2"].items() if v["given"]]
    r6 = 1 - len(l2_ok) / n
    log(f"[R6] 给不出结构性理由 = {n - len(l2_ok)}/{n} = {r6:.4f}；给出的 = {l2_ok}")
    for rid, v in p1["l2"].items():
        at = v["attempts"][0]
        log(f"     {rid}: L2={v['reason']} | detail={v['detail'][:80]} | "
            f"A1问题数={at['n_problems']} 首条={(at['problems'] or ['-'])[0][:90]}")

    # ---- 8. 判定
    r1n = sum(len(v) for v in r1_bad.values())
    r2n = sum(len(v) for v in r2_bad.values())
    r4ok = all(arms[a]["n_violation_records"] > 0 for a in ("G-free_strict", "G-free_generous"))
    base = (len(covered) >= 1 and r1n == 0 and r2n == 0 and r3_miss == 0 and r4ok)
    if base and r6 <= 1 / 3:
        verdict = "A 指针输出可获得可判的结构性解释"
    elif base and r6 >= 2 / 3:
        verdict = "B 只能给机械层、给不出理由"
    else:
        verdict = "C 证据不足"
    r4txt = " ".join(f"{a}={arms[a]['n_violation_records']}/{n}"
                     for a in ("G-free_strict", "G-free_generous"))
    log(f"[判定] R0={len(covered)}/{n} R1={r1n} R2={r2n} R3漏={r3_miss} "
        f"R4违例[{r4txt}] R5一致={same} R6={r6:.4f} ⇒ {verdict}")

    # ---- 9. 样例（原文照抄用）
    samples = []
    for i, rec in enumerate(recs[:8]):
        rid = rec["rid"]
        samples.append({
            "rid": rid, "input": rec["input"], "既有输出_text": rec["text"],
            "既有 evidence": rec["evidence"],
            "candidates": [{k: c[k] for k in ("card", "span", "span_text", "score",
                                              "rank", "score_rank", "eligible",
                                              "ineligible_reason")}
                           for c in rec["candidates"]],
            "chosen": rec["chosen"], "scores_kind": rec["scores_kind"],
            "L1 解释文本": p1["l1"][i].get("text"),
            "L1 ref_map": p1["l1"][i].get("ref_map"),
            "L1 未入槽的 chosen": p1["l1"][i].get("unrenderable_chosen"),
            "L2": p1["l2"][rid], "R1": p1["r1"][rid], "R2": p1["r2"][rid],
        })
    for s in samples:
        log(f"[sample] {s['rid']} input={s['input']}")
        log(f"         既有输出={s['既有输出_text']}")
        log(f"         candidates={json.dumps(s['candidates'], ensure_ascii=False)}")
        log(f"         chosen={json.dumps(s['chosen'], ensure_ascii=False)}")
        log(f"         L1={s['L1 解释文本']!r} 未入槽={json.dumps(s['L1 未入槽的 chosen'], ensure_ascii=False)}")
        log(f"         L2={s['L2']['reason']} detail={s['L2']['detail']}")
        for a in s["L2"]["attempts"]:
            log(f"            {a['name']}: text={a['text']!r} ok={a['ok']} na={a['na']} "
                f"C1={a['c1']} C2={a['c2']} C3={a['c3']} nprob={a['n_problems']} fail={a['fail']}")
            if a["problems"]:
                log(f"               首条问题: {a['problems'][0]}")

    out = {
        "prereg_mtime": prereg.stat().st_mtime, "run_start": now,
        "gate": gate, "n_records": n,
        "R0": {"covered": len(covered), "n": n, "uncovered": cov_why,
               "field_table": FIELD_TABLE, "n_unavailable": len(unavail),
               "unavailable_ratio": round(ratio, 4),
               "unavailable_by_field": dict(un_cnt),
               "unavailable_records": unavail},
        "R1": {"n_violation_records": len(r1_bad), "n_violations": r1n,
               "examples": {k: v[:3] for k, v in list(r1_bad.items())[:5]}},
        "R2": {"n_mismatch_records": len(r2_bad), "n_mismatches": r2n,
               "examples": {k: v[:3] for k, v in list(r2_bad.items())[:5]}},
        "R3": inj,
        "R4": {a: arms[a] for a in ("G-free_strict", "G-free_generous")},
        "R5": {"deterministic": same, "baseline": arms["B-echo"],
               "n_passes": 2},
        "R6": {"n_no_reason": n - len(l2_ok), "n": n, "ratio": round(r6, 4),
               "given": l2_ok,
               "detail": {k: {"reason": v["reason"], "detail": v["detail"],
                              "first_problem": v["first_problem"],
                              "attempts": v["attempts"]} for k, v in p1["l2"].items()}},
        "struct_records": recs,
        "l1_cards": p1["l1"],
        "samples": samples,
        "arm_records": {k: {a: v[:3] for a, v in vv.items()}
                        for k, vv in p1["arms"].items()},
        "predicate_calls_pass1": pred1[0],
        "predicate_nonempty_pass1": pred1[1],
        "verdict": verdict,
    }
    (HERE / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "run.log").write_text("\n".join(log_lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
