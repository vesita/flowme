#!/usr/bin/env python3
"""L0–L6 门禁判定表（全部数字由 results/eval.json 直接读出）。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SE = math.sqrt(0.25 / 1200)
TWO = 2 * SE
TL = {"chain": "有效·直链", "variant": "有效·变体",
      "fallacy": "谬误", "distractor": "干扰"}


def pc(x):
    return f"{x * 100:.2f}"


def main() -> int:
    E = json.loads((HERE / "results" / "eval.json").read_text(encoding="utf-8"))["tags"]

    def S(tag, ex):
        return E[tag]["sets"][f"test/{ex}"]

    print(f"2SE = {TWO * 100:.2f} pt (n=1200)")
    print("\n### L0 主门（臂 T 在改写出口，含 REJECT 判别）")
    print("| seed | 卡 | max_naive(规则) | 阈值=naive+2SE | Δ−2SE | 判定 |")
    print("|---|---|---|---|---|---|")
    for s in (42, 43):
        r = S(f"T_s{s}", "para")
        th = r["max_naive"]["acc"] + TWO
        print(f"| {s} | {pc(r['acc'])} | {pc(r['max_naive']['acc'])} ({r['max_naive']['rule']}) | "
              f"{pc(th)} | {pc(r['card_minus_naive_2se'])} | "
              f"{'过' if r['acc'] > th else '**不过**'} |")
    print("\n  同表列出其余臂（改写出口）：")
    for tag in sorted(E):
        if E[tag]["meta"]["randlabel"]:
            continue
        r = S(tag, "para")
        print(f"  - {tag}: {pc(r['acc'])}% vs naive {pc(r['max_naive']['acc'])}% "
              f"⇒ {pc(r['card_minus_naive'])} pt")

    print("\n### L1 表内 vs 改写（臂 T）")
    print("| seed | 模板出口 | 改写出口 | Δ=模板−改写 | 改写−多数类 | 判定 |")
    print("|---|---|---|---|---|---|")
    for s in (42, 43):
        a, b = S(f"T_s{s}", "tmpl"), S(f"T_s{s}", "para")
        print(f"| {s} | {pc(a['acc'])} | {pc(b['acc'])} | {pc(a['acc'] - b['acc'])} pt | "
              f"{pc(b['acc'] - b['rules']['majority']['acc'])} pt | "
              f"{'改写崩' if a['acc'] - b['acc'] > TWO * 10 else '未崩'} |")

    print("\n### L2 谬误/干扰必须拒答（臂 T）")
    print("| seed | 出口 | 型 | n | 卡拒答率 | gold拒答 | max_naive(该型拒答率) | "
          "卡−该型拒答naive | 该型 max_naive(准确率) | 卡acc | 卡−该型naive |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for s in (42, 43):
        for ex in ("tmpl", "para"):
            r = S(f"T_s{s}", ex)
            for ty in ("fallacy", "distractor"):
                p = r["per_type"][ty]
                print(f"| {s} | {ex} | {TL[ty]} | {p['n']} | {pc(p['reject_pred'])} | "
                      f"{pc(p['reject_gold'])} | {pc(p['max_naive_reject'])} | "
                      f"{pc(p['reject_pred'] - p['max_naive_reject'])} | "
                      f"{pc(p['max_naive_acc'])} | {pc(p['acc'])} | "
                      f"{pc(p['card_minus_naive'])} |")
    print("  正例被误拒答条数（臂 T，两出口四型合计）：")
    for s in (42, 43):
        for ex in ("tmpl", "para"):
            r = S(f"T_s{s}", ex)
            n = sum(p["false_reject"] for p in r["per_type"].values())
            print(f"  - T/{s}/{ex}: {n}")

    print("\n### L3 引用映射")
    print("| seed | 出口 | 结构违例(必须0) | 归因 eid 匹配(有结论且判对) | 部署态归因全对/有结论 |")
    print("|---|---|---|---|---|")
    for s in (42, 43):
        for ex in ("tmpl", "para"):
            l3 = S(f"T_s{s}", ex)["l3"]
            print(f"| {s} | {ex} | {l3['violations']} | {pc(l3['attr_eid_match'])} | "
                  f"{l3['attr_exact_deployed']}/{l3['n_concluded']} |")

    print("\n### L4 随机标签对照（掉回多数类+2SE）")
    print("| 臂 | seed | 出口 | 卡acc | 多数类 | 阈值 | 判定 |")
    print("|---|---|---|---|---|---|---|")
    for tag in sorted(E):
        if not E[tag]["meta"]["randlabel"]:
            continue
        for ex in ("tmpl", "para"):
            r = S(tag, ex)
            th = r["rules"]["majority"]["acc"] + TWO
            print(f"| {E[tag]['meta']['arm']} | {E[tag]['meta']['seed']} | {ex} | "
                  f"{pc(r['acc'])} | {pc(r['rules']['majority']['acc'])} | {pc(th)} | "
                  f"{'掉回' if r['acc'] <= th else '**没掉回**'} |")

    print("\n### L6 每出口：卡 − 规则（R-lexdir 披露项不计入 max_naive）")
    print("| 臂 | seed | 出口 | 卡 | R-chain | R-keyword | R-lexdir | n_slots | 卡−R-lexdir |")
    print("|---|---|---|---|---|---|---|---|---|")
    for tag in sorted(E):
        if E[tag]["meta"]["randlabel"]:
            continue
        for ex in ("tmpl", "para"):
            r = S(tag, ex)
            g = r["rules"]
            print(f"| {E[tag]['meta']['arm']} | {E[tag]['meta']['seed']} | {ex} | "
                  f"{pc(r['acc'])} | {pc(g['R-chain']['acc'])} | {pc(g['R-keyword']['acc'])} | "
                  f"{pc(g['R-lexdir']['acc'])} | {pc(g['n_slots']['acc'])} | "
                  f"{pc(r['acc'] - g['R-lexdir']['acc'])} |")

    print("\n### 逐型（臂 T，test）")
    print("| seed | 出口 | 型 | n | 卡acc | 卡拒答 | gold拒答 | R-chain(单条全局最优) | 逐型最好naive |")
    print("|---|---|---|---|---|---|---|---|---|")
    for s in (42, 43):
        for ex in ("tmpl", "para"):
            r = S(f"T_s{s}", ex)
            for ty in TL:
                p = r["per_type"][ty]
                rc = r["rules"]["R-chain"]["per_type_acc"].get(ty)
                print(f"| {s} | {ex} | {TL[ty]} | {p['n']} | {pc(p['acc'])} | "
                      f"{pc(p['reject_pred'])} | {pc(p['reject_gold'])} | "
                      f"{pc(rc) if rc is not None else '—'} | "
                      f"{pc(p['max_naive_acc'])} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
