#!/usr/bin/env python3
"""汇总：主表 + δ + P1–P5 判定（判据全部来自 PREREG.md §4，此处只做机械计算）。

    uv run python experiments/anchored_select/summarize.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
LOG = HERE.parent.parent / "logs" / "anchored_stageA.log"
SEEDS = (42, 43)
M = 50.0                     # 多数类 / 盲猜（三集 1:1，跑前算死）
BANDS_PT = {"pronoun": 2.83, "sentiment": 0.41, "relation": 1.39, "person": 0.33}


def rd(p: Path) -> dict:
    return json.loads(p.read_text(encoding="utf-8"))


def main() -> int:
    main_r = {s: rd(RES / f"main_s{s}.json") for s in SEEDS}
    shuf_r = {s: rd(RES / f"shuf_s{s}.json") for s in SEEDS}
    joint_r = {s: rd(RES / f"joint_train_s{s}.json") for s in SEEDS}
    stats = rd(HERE / "data" / "stats.json")

    def acc(rep: dict, split: str) -> float:
        return rep["splits"][split]["bin_acc"] * 100

    out: dict = {"majority_blind_pct": M, "splits": {}}
    print("=== 主表（二分类准确率，%） ===")
    print("| 集合 | n | 多数类/盲猜 | s42 | s43 | 极差 | δ | 门槛 | P |")
    print("|---|---|---|---|---|---|---|---|---|")
    verdicts = {}
    for split, key in (("train", "P-"), ("test", "P1"), ("adversarial", "P2")):
        a42, a43 = acc(main_r[42], split), acc(main_r[43], split)
        rng = abs(a42 - a43)
        delta = max(2.0, rng)
        thr = M + delta
        ok = a42 >= thr and a43 >= thr
        n = main_r[42]["splits"][split]["n"]
        verdicts[key] = ok
        out["splits"][split] = {"n": n, "s42": round(a42, 2), "s43": round(a43, 2),
                                "range_pt": round(rng, 2), "delta_pt": round(delta, 2),
                                "threshold": round(thr, 2), "pass": ok}
        print(f"| {split} | {n} | {M:.1f} | {a42:.2f} | {a43:.2f} | {rng:.2f}pt | "
              f"{delta:.2f}pt | ≥{thr:.2f} | {'PASS' if ok else 'FAIL'} {key} |")

    print("\n=== 附报：exact（判对且锚点命中）与锚点命中 ===")
    for split in ("train", "test", "adversarial"):
        e42 = main_r[42]["splits"][split]["exact"] * 100
        e43 = main_r[43]["splits"][split]["exact"] * 100
        a42 = main_r[42]["splits"][split]["pos_anchor_hit"] * 100
        a43 = main_r[43]["splits"][split]["pos_anchor_hit"] * 100
        print(f"  {split:13s} exact {e42:.2f}/{e43:.2f}%   正例锚点命中 {a42:.2f}/{a43:.2f}%")

    print("\n=== P3 对照 ===")
    shuf = {}
    for s in SEEDS:
        v = acc(shuf_r[s], "test")
        shuf[s] = v
        print(f"  随机标签对照 s{s} test={v:.2f}%  （门槛 ≤ {M+3:.1f}）"
              f" {'PASS' if v <= M + 3 else 'FAIL'}")
    p3 = all(shuf[s] <= M + 3 for s in SEEDS)
    p3 &= all(acc(main_r[s], "test") > shuf[s] for s in SEEDS)
    verdicts["P3"] = p3
    print(f"  盲猜/多数类 = {M:.1f}%（三集 1:1，构建期断言）")
    print(f"  表层子串规则：train {stats['train']['lex_substring_baseline']*100:.1f}% / "
          f"test {stats['test']['lex_substring_baseline']*100:.1f}% / "
          f"adv {stats['adversarial']['lex_substring_baseline']*100:.1f}%")
    print(f"  P3 {'PASS' if p3 else 'FAIL'}（主臂 > 对照 且 对照 ≤ M+3）")

    print("\n=== P4 联合臂（老卡 Δ，pt） ===")
    p4 = {}
    for s in SEEDS:
        p4s = joint_r[s]["p4"]
        for n, v in p4s.items():
            p4.setdefault(n, []).append(v["delta_pt"])
        print(f"  s{s}: " + "  ".join(f"{n}={v['delta_pt']:+.2f}(带−{v['band_pt']:.2f})"
                                      for n, v in p4s.items()))
    p4_pass = all(joint_r[s]["p4_pass"] for s in SEEDS)
    verdicts["P4"] = p4_pass
    print(f"  P4 {'PASS' if p4_pass else 'FAIL'}（两 seed 都需 ≥ −各自噪声带）")
    out["p4"] = {n: [round(x, 3) for x in v] for n, v in p4.items()}
    out["p4_pass"] = p4_pass

    print("\n=== P5 代价 / 样本量 ===")
    log = LOG.read_text(encoding="utf-8")
    ab = [json.loads(m) for m in re.findall(r"AB_METRICS (\{.*\})", log)]
    for r in ab:
        print(f"  主臂 s{r['seed']}{'(shuf)' if r['seed'] and 'shuf' in str(r.get('tag','')) else ''}"
              f": steps={r['n_steps']} sec/step={r['sec_per_step']:.4f} "
              f"train={r['train_sec']:.1f}s peak={r['peak_mem_mb']:.0f}MB "
              f"head_params={r['n_head_params']}")
    for s in SEEDS:
        t = joint_r[s]["timing"]
        print(f"  联合臂 s{s}: steps={t['n_steps']} sec/step={t['sec_per_step']:.4f} "
              f"train={t['train_sec']:.1f}s peak={t['peak_mem_mb']:.0f}MB "
              f"可训参数={joint_r[s]['params']['trainable_total']} "
              f"(核 {joint_r[s]['params']['core_trainable']} + 头 "
              f"{joint_r[s]['params']['heads']['anchored_sel']['trainable']}) "
              f"核漂移={joint_r[s]['drift']:.4f}")
    out["train_ab"] = ab
    out["joint_timing"] = {str(s): joint_r[s]["timing"] for s in SEEDS}
    out["params"] = {"core_total": 1688460, "core_trainable_frozen_arm": 0,
                     "head": 629764,
                     "joint_trainable_total": {str(s): joint_r[s]["params"]["trainable_total"]
                                               for s in SEEDS}}
    out["verdicts"] = {k: bool(v) for k, v in verdicts.items()}
    p1p2p3 = all(verdicts[k] for k in ("P1", "P2", "P3"))
    out["verdict"] = ("① 能学" if p1p2p3 else
                      "② 不能学" if (not verdicts["P1"] and not verdicts["P2"]) else
                      "③ 证据不足")
    print(f"\n=== 判定：{out['verdict']}（P1={verdicts['P1']} P2={verdicts['P2']} "
          f"P3={verdicts['P3']} P4={verdicts['P4']}）===")

    (RES / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"[save] {RES / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
