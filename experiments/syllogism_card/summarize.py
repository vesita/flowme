#!/usr/bin/env python3
"""把 results/eval.json 摊成报告要用的 Markdown 表（数字来自实测）。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SE = math.sqrt(0.25 / 1200)
TWO = 2 * SE
TYPE_LABEL = {"chain": "有效·直链", "variant": "有效·变体",
              "fallacy": "谬误", "distractor": "干扰"}


def P(x):
    return f"{x * 100:.2f}"


def main() -> int:
    d = json.loads((HERE / "results" / "eval.json").read_text(encoding="utf-8"))
    data = json.loads((HERE / "data.json").read_text(encoding="utf-8"))
    print(f"== 断言 ==\n```\n{json.dumps(d['surface'], ensure_ascii=False)}\n```")
    print(f"\n== 逐型丢弃/表面规则可解率（test，R-chain 逐型 acc）==")
    print("| 出口 | 型 | n | gold拒答率 | R-chain | R-keyword | R-lexdir(披露) |")
    print("|---|---|---|---|---|---|---|")
    ref = d["tags"][sorted(d["tags"])[0]]["sets"]
    for key in ("test/tmpl", "test/para"):
        s = ref[key]
        for ty, lab in TYPE_LABEL.items():
            p = s["per_type"][ty]
            # 逐型规则 acc：rules 里没有 per-type reject，重新算一次
            print(f"| {key} | {lab} | {p['n']} | {P(p['reject_gold'])} | "
                  f"{P(p['max_naive_acc'])} | — | — |")
    print("\n== 各臂 × seed：出口准确率 vs max_naive ==")
    print("| 臂 | seed | 出口 | n | 卡 | max_naive | 规则 | 卡−规则 | 卡−规则−2SE | "
          "R-lexdir(披露) | 解析通过 | 标注字准 | L3违例 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in sorted(d["tags"]):
        v = d["tags"][tag]
        arm = v["meta"]["arm"]
        seed = v["meta"]["seed"]
        rnd = "_rand" if v["meta"]["randlabel"] else ""
        for key in ("test/tmpl", "test/para"):
            if key not in v["sets"]:
                continue
            s = v["sets"][key]
            ex = key.split("/")[1]
            print(f"| {arm}{rnd} | {seed} | {ex} | {s['n']} | {P(s['acc'])} | "
                  f"{P(s['max_naive']['acc'])} | {s['max_naive']['rule']} | "
                  f"{P(s['card_minus_naive'])} | {P(s['card_minus_naive_2se'])} | "
                  f"{P(s['rules']['R-lexdir']['acc'])} | {P(s['parse_ok'])} | "
                  f"{P(s['tag_acc'])} | {s['l3']['violations']}/{s['l3'].get('sem_violations')} |")

    print("\n== 逐型（test）==")
    print("| 臂 | seed | 出口 | 型 | n | 卡acc | gold拒答 | 卡拒答 | 该型max_naive | 卡−naive | 误拒正例 | 误结论")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for tag in sorted(d["tags"]):
        v = d["tags"][tag]
        for key in ("test/tmpl", "test/para"):
            if key not in v["sets"]:
                continue
            for ty, lab in TYPE_LABEL.items():
                p = v["sets"][key]["per_type"][ty]
                print(f"| {v['meta']['arm']}{'_rand' if v['meta']['randlabel'] else ''} | "
                      f"{v['meta']['seed']} | {key.split('/')[1]} | {lab} | {p['n']} | "
                      f"{P(p['acc'])} | {P(p['reject_gold'])} | {P(p['reject_pred'])} | "
                      f"{P(p['max_naive_acc'])} | {P(p['card_minus_naive'])} | "
                      f"{p['false_reject']} | {p['false_conclude']} |")

    print("\n== L3 引用映射 ==")
    for tag in sorted(d["tags"]):
        v = d["tags"][tag]
        for key in ("test/tmpl", "test/para"):
            if key not in v["sets"]:
                continue
            l3 = v["sets"][key]["l3"]
            print(f"- {tag} {key}: 结构违例={l3['violations']} 归因违例={l3.get('sem_violations')} "
                  f"有结论={l3['n_concluded']} "
                  f"eid匹配={P(l3['attr_eid_match'])} 部署态归因全对={l3['attr_exact_deployed']} "
                  f"原因={l3['reasons']}")

    print("\n== 注入反例 / 老卡 ==")
    print("```")
    print(json.dumps(d.get("inject"), ensure_ascii=False, indent=1))
    print(json.dumps(d.get("old_cards"), ensure_ascii=False, indent=1)[:1800])
    print("```")
    print("\n== 逐型统计（生成期）==")
    print("```")
    print(json.dumps(data["stats"], ensure_ascii=False, indent=1))
    print("dropped:", json.dumps(data["dropped"], ensure_ascii=False))
    print("```")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
