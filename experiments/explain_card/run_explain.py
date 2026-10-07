"""P9 驱动：跑解释卡、反例注入、对照臂、确定性。纯 CPU，不训练。

    uv run python experiments/explain_card/run_explain.py
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import explain_card as ec  # noqa: E402


def jdump(o) -> str:
    return json.dumps(o, sort_keys=True, ensure_ascii=False)


def main() -> int:
    prereg = HERE / "PREREG.md"
    now = time.time()
    assert prereg.exists(), "PREREG.md 必须先落"
    assert prereg.stat().st_mtime < now, "PREREG mtime 必须早于首跑"
    log_lines: list[str] = []

    def log(msg: str) -> None:
        print(msg, flush=True)
        log_lines.append(msg)

    log(f"[prereg] mtime={time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(prereg.stat().st_mtime))}"
        f" < run={time.strftime('%Y-%m-%d %H:%M:%S')}")

    # ---- 0. 解释表门禁
    gate = ec.expl_gate()
    log(f"[gate] EXPL_SKELETONS rule_a+slot_schema 违例 = {len(gate)}")
    for p in gate:
        log(f"       {p}")

    # ---- 1. 装载（跑两遍用于确定性）
    srcs = ec.load_all()
    src_by_rid = {s.rid: s for s in srcs}
    srcs2 = ec.load_all()
    by_rid2 = {s.rid: s for s in srcs2}
    log(f"[load] 记录 {len(srcs)} 条："
        + str(dict(Counter(s.kind for s in srcs)))
        + " 分集 " + str(dict(Counter(s.set for s in srcs))))

    # ---- 2. 源记录复验
    src_probs = {s.rid: ec.source_problems(s) for s in srcs}
    n_text = sum(1 for s in srcs if s.kind == "text")
    src_bad = {k: v for k, v in src_probs.items() if v}
    log(f"[source] text 记录 {n_text} 条，官方谓词复验违例记录数 = {len(src_bad)}")
    for k, v in list(src_bad.items())[:5]:
        log(f"         {k}: {v[:3]}")

    # ---- 3. 解释卡 + X0/X2 + 确定性
    cards: dict[str, dict] = {}
    x0: dict[str, list[str]] = {}
    x2: dict[str, list[str]] = {}
    x3_mismatch: list[str] = []
    for s in srcs:
        c = ec.explain(s)
        cards[s.rid] = c
        x0[s.rid] = ec.x0_problems(c, s)
        x2[s.rid] = ec.x2_problems(c)
        c2 = ec.explain(by_rid2[s.rid])
        if jdump(c) != jdump(c2):
            x3_mismatch.append(s.rid)

    expl_text = [r for r, c in cards.items()
                 if c.get("explainable") and c["kind"] == "text"]
    expl_rej = [r for r, c in cards.items()
                if c.get("explainable") and c["kind"] == "reject"]
    x0_viol = {r: x0[r] for r in expl_text if x0[r]}
    x2_viol = {r: x2[r] for r in expl_rej if x2[r]}
    log(f"[X0] 可解释 text 卡 {len(expl_text)} 条，违例卡 {len(x0_viol)}，问题总数 "
        f"{sum(len(v) for v in x0_viol.values())}")
    for r, v in list(x0_viol.items())[:5]:
        log(f"     {r}: {v[:2]}")
    log(f"[X2] 可解释 reject 卡 {len(expl_rej)} 条，违例卡 {len(x2_viol)}，"
        f"问题总数 {sum(len(v) for v in x2_viol.values())}")
    log(f"[X3] 两次渲染不一致 = {len(x3_mismatch)} {x3_mismatch[:5]}")

    # ---- 4. X1 反例注入
    good_text = [r for r in expl_text if not x0[r]]
    good_rej = [r for r in expl_rej if not x2[r]]
    inj: dict[str, dict] = {}

    def run_inj(name, rid_list, fn):
        caught, total = 0, 0
        cats: Counter = Counter()
        rep: dict[str, str] = {}
        for rid in rid_list:
            total += 1
            card, probs = fn(cards[rid], src_by_rid[rid])
            if probs:
                caught += 1
            for pb in probs:
                c = pb.split(":")[0]
                cats[c] += 1
                rep.setdefault(c, pb)
        inj[name] = {"injected": total, "caught": caught, "missed": total - caught,
                     "by_category": dict(cats), "representative": rep}
        log(f"[X1-{name}] 注入 {total} 条，被抓 {caught}，漏 {total - caught}；"
            f"类别 {dict(cats)}")
        for c, m in rep.items():
            log(f"     - {c}: {m[:150]}")

    run_inj("①加输入没有的内容词", good_text, ec.inject_content_word)
    run_inj("①′加内容词+伪袋块(PREREG外加测)", good_text, ec.inject_content_word_bag)
    run_inj("②a引用指向不存在的 ref", good_text, ec.inject_bad_ref)
    run_inj("②b引用指向不存在的 span", good_text, ec.inject_bad_span)
    run_inj("③拒答解释带上 text", good_rej, ec.inject_reject_text)

    # ---- 5. X4 对照臂 E-free
    free_viol: dict[str, list[str]] = {}
    free_kind = Counter()
    for rid in good_text:
        s = src_by_rid[rid]
        rec, meta = ec.free_record(s)
        probs = ec.x0_free(rec, s, meta)
        free_kind[rec.get("kind", "?")] += 1
        if probs:
            free_viol[rid] = probs
    free_total = sum(len(v) for v in free_viol.values())
    free_cats = Counter(p.split(":")[0] for v in free_viol.values() for p in v)
    free_sample = None
    if good_text:
        s0 = src_by_rid[good_text[0]]
        rec0, meta0 = ec.free_record(s0)
        free_sample = {"rid": s0.rid, "input": s0.input,
                       "constrained_text": cards[s0.rid].get("text"),
                       "free_text": rec0.get("text"),
                       "free_problems_head": ec.x0_free(rec0, s0, meta0)[:4]}
    log(f"[X4] E-free 在 {len(good_text)} 条上违例卡 = {len(free_viol)}，问题总数 {free_total}")
    log(f"     违例类别分布 = {dict(free_cats)}")
    for r, v in list(free_viol.items())[:3]:
        log(f"     {r}: {v[:2]}")

    # ---- 6. X5 覆盖率
    def bucket(pred):
        allids = [s.rid for s in srcs if pred(s)]
        ok = [r for r in allids if cards[r].get("explainable")]
        why = Counter(cards[r].get("why", "") for r in allids if r not in ok)
        return {"n": len(allids), "explained": len(ok),
                "ratio": round(len(ok) / len(allids), 4) if allids else None,
                "uncovered_reasons": dict(why)}

    cov = {
        "成功_text": bucket(lambda s: s.kind == "text"),
        "指针_pointer": bucket(lambda s: s.kind == "pointer"),
        "拒答_reject": bucket(lambda s: s.kind == "reject"),
        "全体": bucket(lambda s: True),
        "按集": {k: bucket(lambda s, k=k: s.set == k) for k in ("A", "B", "B'")},
    }
    log("[X5] 覆盖率 " + json.dumps(cov, ensure_ascii=False))

    # ---- 7. 样例
    samples = []
    quota = {("text", "A"): 3, ("text", "B"): 2, ("reject", "A"): 2, ("reject", "B"): 1}
    shown: Counter = Counter()
    for s in srcs:
        c = cards[s.rid]
        key = (s.kind, s.set)
        if not s.input or not c.get("explainable") or shown[key] >= quota.get(key, 0):
            continue
        if s.kind == "text":
            samples.append({"rid": s.rid, "input": s.input, "record": {
                "kind": s.kind, "skeleton": s.source_skeleton,
                "instruction": s.instruction, "text": s.text,
                "ref_map": [{k: e[k] for k in ("unit_id", "cls", "text", "ref", "span", "out")}
                            for e in s.ref_map]},
                "explain_card": {k: c[k] for k in ("kind", "text", "ref_map", "instruction") if k in c},
                "x0_problems": x0[s.rid]})
            shown[key] += 1
        elif s.kind == "reject":
            samples.append({"rid": s.rid, "input": s.input, "record": {
                "kind": "reject", "reason": s.reason, "note": s.note},
                "explain_card": {k: c[k] for k in ("kind", "reason") if k in c},
                "has_text_key": "text" in c, "x2_problems": x2[s.rid]})
            shown[key] += 1

    # ---- 8. 判定
    x0n = sum(len(v) for v in x0_viol.values())
    x1ok = all(v["missed"] == 0 and v["injected"] > 0 for v in inj.values())
    x2n = sum(len(v) for v in x2_viol.values())
    x3n = len(x3_mismatch)
    x4n = free_total
    n_ok_explained = cov["成功_text"]["explained"]
    verdict = ("T1 解释可被约束为不可编" if (x0n == 0 and x1ok and x2n == 0 and x3n == 0 and x4n > 0)
               else "T3 证据不足" if n_ok_explained < 10
               else "T2 约束无效")
    log(f"[pred] 显式调用次数 = {ec.PRED_CALLS}")
    log(f"[pred] 返回非空次数 = {ec.PRED_NONEMPTY}")
    log(f"[判定] X0={x0n} X1={'全抓' if x1ok else '有漏'} X2={x2n} X3={x3n} "
        f"X4={x4n} 覆盖(成功)={n_ok_explained}/{cov['成功_text']['n']} ⇒ {verdict}")

    out = {
        "prereg_mtime": prereg.stat().st_mtime,
        "run_start": now,
        "gate_problems": gate,
        "n_records": len(srcs),
        "kinds": dict(Counter(s.kind for s in srcs)),
        "source_recheck": {"n_text": n_text, "n_violation_records": len(src_bad),
                           "examples": {k: v[:3] for k, v in list(src_bad.items())[:10]}},
        "X0": {"n_explainable_text": len(expl_text), "n_violation_cards": len(x0_viol),
               "n_violations": x0n, "examples": {k: v[:3] for k, v in list(x0_viol.items())[:10]}},
        "X1": inj,
        "X2": {"n_reject_cards": len(expl_rej), "n_violation_cards": len(x2_viol),
               "n_violations": x2n, "examples": {k: v for k, v in list(x2_viol.items())[:5]}},
        "X3": {"n_cards": len(cards), "n_mismatch": x3n, "mismatch": x3_mismatch[:10]},
        "X4": {"free_sample": free_sample,
               "n_tested": len(good_text), "n_violation_cards": len(free_viol),
               "n_violations": x4n, "by_category": dict(free_cats),
               "render_kind": dict(free_kind),
               "examples": {k: v[:3] for k, v in list(free_viol.items())[:5]}},
        "X5": cov,
        "predicate_calls": ec.PRED_CALLS,
        "predicate_nonempty": ec.PRED_NONEMPTY,
        "samples": samples,
        "verdict": verdict,
    }
    (HERE / "results.json").write_text(json.dumps(out, ensure_ascii=False, indent=1))
    (HERE / "logs").mkdir(exist_ok=True)
    (HERE / "logs" / "run.log").write_text("\n".join(log_lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
