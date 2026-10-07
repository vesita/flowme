#!/usr/bin/env python3
"""把 results/probe.json + results/reconcile.json 渲染成 REPORT 用的 Markdown 表（贴数不贴码）。"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
LAYERS = ("emb", "b0", "b1", "b2", "pool")
TASKS = ("emo", "skel", "logic", "idr")
CN = {"emo": "情绪", "skel": "骨架id", "logic": "逻辑词", "idr": "身份-回忆"}
PAIRS = (("emo", "skel"), ("emo", "idr"), ("skel", "idr"),
         ("emo", "logic"), ("skel", "logic"), ("logic", "idr"))


def f4(x):
    return f"{x:.4f}" if isinstance(x, (int, float)) else str(x)


def main(tag: str = "") -> int:
    res = json.loads((HERE / "results" / f"probe{tag}.json").read_text())
    out = []
    p = out.append

    p("### A. 线上口径对账")
    try:
        rec = json.loads((HERE / "results" / "reconcile.json").read_text())
        p(f"- 复现 sentiment 卡 val `cls_acc` = **{rec['reproduced']['cls_acc']:.4f}**"
          f"（期望 {rec['expected']['cls_acc']:.4f}，绝对差 {rec['abs_diff']:+.4f}，"
          f"相对差 {rec['rel_diff']:.4%}，门槛 <{rec['rel_tol']:.0%}）"
          f" ⇒ {'PASS' if rec['pass'] else 'FAIL'}；hook vs 直接前向 max|Δ|="
          f"{rec['hook_vs_direct_maxabs']}")
        p(f"- n_cls={rec['reproduced']['n_cls']} n_bg={rec['reproduced']['n_bg']}，"
          f"wall={rec['wall_sec']}s，设备={rec['device']}")
    except FileNotFoundError:
        p("- reconcile.json 缺失")

    lab = res["labels"]
    p("\n### B. 三类标签（P5：逐类 max_naive 并列）")
    p("| 任务 | 类 | 数据源 | n_tr / n_te | 类别分布(train) | max_naive@majority | max_naive@rule |")
    p("|---|---|---|---|---|---|---|")
    for t in TASKS:
        s = lab[t]
        dist = ", ".join(f"{k}:{v}" for k, v in list(s["train_dist"].items())[:6])
        if len(s["train_dist"]) > 6:
            dist += f" …(共 {len(s['train_dist'])} 类)"
        p(f"| {t} ({CN[t]}) | {s['n_classes']} | {s['source'][:60]} | "
          f"{s['n_train']} / {s['n_test']} | {dist} | {f4(s['max_naive@majority'])} | "
          f"{f4(s['max_naive@rule'])} |")
        p(f"  ↳ 规则口径：{s.get('rule_desc','')}")

    p("\n### C. P0 双向对照")
    p("| 对照 | 层 | acc (s42/s43) | 门槛 | 判 |")
    p("|---|---|---|---|---|")
    for t in TASKS:
        for ln in LAYERS:
            v = res["p0"]["random_label"][t][ln]
            p(f"| ① 随机标签 {t} | {ln} | {'/'.join(str(a) for a in v['acc'])} | "
              f"≤{f4(v['threshold'])} | {'PASS' if v['pass'] else 'FAIL'} |")
    for ln in LAYERS:
        v = res["p0"]["marker_in_input"][ln]
        p(f"| ②a 输入内标记 | {ln} | {'/'.join(str(a) for a in v['acc'])} | ≥0.99 | "
          f"{'PASS' if v['pass'] else 'FAIL'} |")
    s1 = res["p0"]["synthetic_linear"]
    s2 = res["p0"]["synthetic_linear_scaled"]
    p(f"| ②b 合成张量 n={s1['n']}（PREREG 字面） | — | {s1['42']}/{s1['43']} | ≥0.99 | "
      f"{'PASS' if s1['pass'] else 'FAIL'} |")
    p(f"| ②b′ 合成张量 n={s2['n']}（功效修正） | — | {s2['42']}/{s2['43']} | ≥0.99 | "
      f"{'PASS' if s2['pass'] else 'FAIL'} |")
    p(f"- P0 字面口径 pass={res['p0']['pass']}；有效口径 pass_effective="
      f"{res['p0']['pass_effective']}；继续条件 = {res['p0']['gate_used_for_continue']}")

    p("\n### D. P1/P3 分层 × 四任务主表（acc ± SE，2 seed；max_naive 并列）")
    p("| 任务 | " + " | ".join(LAYERS) + " | 最佳层 | max_naive@maj | max_naive@rule | P1 |")
    p("|---|" + "---|" * (len(LAYERS) + 4))
    for t in TASKS:
        row = res["p1"][t]
        cells = []
        for ln in LAYERS:
            v = row[ln]
            star = "*" if ln == row["_best_layer"] else ""
            cells.append(f"{v['mean']*100:.2f}±{v['se']*100:.2f}{star} "
                         f"({','.join(f'{a*100:.1f}' for a in v['acc'])})")
        p(f"| {t} ({CN[t]}) | " + " | ".join(cells) + f" | {row['_best_layer']} | "
          f"{f4(lab[t]['max_naive@majority'])} | {f4(lab[t]['max_naive@rule'])} | "
          f"{'PASS' if row['_pass'] else 'FAIL'} |")
    p(f"- 可读任务（过 P1）：{res['p1']['readable']}（n={res['p1']['n_readable']}）")

    p("\n### E. P2-1 联合 probe（共享瓶颈 k=8 vs 单任务同瓶颈）@ 各层 L*")
    p("| 任务对 | L* | 联合Δ (s42/s43) | paired SE | 显著下降? |")
    p("|---|---|---|---|---|")
    for k, v in res["p2"]["pairs"].items():
        w = v["worse_task"]
        jt = v["joint"][w]
        p(f"| {k} | {v['L_star']} | {'/'.join(f'{d:+.4f}' for d in jt['delta'])} | "
          f"{'/'.join(f'{s:.5f}' for s in jt['se'])} | {'是' if v['joint_sig_drop'] else '否'}"
          f"（取更差任务 {w}）|")
    zs = res["p2"]["zero_shared_sanity"]
    p("- 零共享 sanity（|Δ| ≤ 2SE ⇒ 构造等价）："
      + "；".join(f"{ln}: " + "/".join(f"{t}{zs[ln][t]['delta']:+.4f}"
                                       f"{'✓' if zs[ln][t]['ok'] else '✗'}" for t in TASKS)
                 for ln in ("pool",)))

    p("\n### F. P2-2 方向余弦分布（|cos|，各层）")
    p("| 任务对 | " + " | ".join(LAYERS) + " |")
    p("|---|" + "---|" * len(LAYERS))
    for a, b in PAIRS:
        v = res["p2"]["cosine"]["between"][f"{a}|{b}"]
        p(f"| {a}|{b} | " + " | ".join(
            f"p50={v[ln]['p50']} [p25={v[ln]['p25']},p75={v[ln]['p75']}]"
            if ln == res["p2"]["pairs"][f"{a}|{b}"]["L_star"]
            else f"{v[ln]['p50']}" for ln in LAYERS) + " |")
    p("- within（同任务跨 init，方向稳定性基线，pool 层）："
      + "；".join(f"{t}={res['p2']['cosine']['within'][t]['pool']['p50']}" for t in TASKS))

    p("\n### G. P2-3 交叉干扰（一维最近质心，X = self − cross，正值=方向不可互换）")
    p("| 方向 | " + " | ".join(LAYERS) + " |")
    p("|---|" + "---|" * len(LAYERS))
    for a, b in PAIRS:
        v = res["p2"]["cross_interference"][f"{a}->{b}"]
        p(f"| {a}->{b} | " + " | ".join(
            f"{v[ln]['42']['X_drop']:+.3f}" +
            (f"(43:{v[ln]['43']['X_drop']:+.3f}, se={v[ln]['42']['se_paired']:.4f})"
             if ln == res["p2"]["pairs"][f"{a}|{b}"]["L_star"] else "")
            for ln in LAYERS) + " |")

    p("\n### H. 判定（PREREG §6）")
    v = res["verdict"]
    p(f"- **判定 = {v['verdict']}**；gate={v['gate']}（P0字面={v.get('p0_pass_prereg_literal')} "
      f"有效={v.get('p0_pass_effective')}；可读数={v['n_readable']}；"
      f"within 稳定门={v['within_gate']}）")
    p(f"- 可读对 = {v['readable_pairs']}；满足「可分离」全部条件 = {v['separable_ok']}；"
      f"满足「缠绕」三条件的对 = {v['entangled_pairs']}")
    p("- 逐对判据（L* 处）：")
    for k, x in res["p2"]["pairs"].items():
        p(f"  - {k}@{x['L_star']}: |cos|p50={x['cos_p50']:.3f}（within_ok={x['within_ok']}）、"
          f"jointΔ={x['joint'][x['worse_task']]['delta']}（sig={x['joint_sig_drop']}）、"
          f"X_max={x['cross_X_max']:+.3f}（sig={x['cross_sig']}）")

    text = "\n".join(out)
    print(text)
    (HERE / "results" / f"tables{tag}.md").write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
