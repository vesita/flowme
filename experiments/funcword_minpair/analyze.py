#!/usr/bin/env python3
"""P4 汇总：P0 对账 / 主表 / 配对 Δ / L 分列 / 机制探针 / 完备地板 / 随机标签 / 判定。

用法：uv run python experiments/funcword_minpair/analyze.py
输出：stdout（Markdown 表）+ results/tables.md + results/verdict.json
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
ROOT = HERE.parents[1]
SL = ROOT / "experiments" / "skeleton_leak" / "results" / "eval.json"
SS = ROOT / "experiments" / "struct_supervision" / "results"

ARMS = ("M", "MP", "MP+")
SEEDS = (42, 43)
# P0 参照（skeleton_leak / bag_modules 公布值，实测）
P0_REF = {"test": {42: 0.6396, 43: 0.6424}, "a_bal": {42: 0.0635, 43: 0.0663}}
P0_SE = {"test": 0.0100, "a_bal": 0.0155}
MAJ = {"test": 0.4168, "a_bal": 0.0385, "a_lit": 0.0333, "b_pairs": 0.1071,
       "c_pairs": 0.0769}
THRESH = {"test": 0.4368, "a_bal": 0.0695}      # 多数类 + 2SE（leak_stats Q0_threshold）


def load(name: str) -> dict:
    p = RESULTS / f"eval_{name}.json"
    if not p.exists():
        raise SystemExit(f"缺 {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def paired(a: list, b: list) -> dict:
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    mu = sum(d) / n
    var = sum((x - mu) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    s = math.sqrt(var / n)
    return {"delta": round(mu, 6), "se": round(s, 6),
            "t": round(mu / s, 3) if s > 0 else None, "n": n,
            "pos": mu > 0}


def main() -> None:
    ev = {f"{a}_s{s}": load(f"{a}_s{s}") for a in ARMS for s in SEEDS}
    rnd = {f"{a}_s{s}": load(f"{a}_s{s}_rand") for a in ARMS for s in SEEDS}
    L: list[str] = []
    add = L.append
    floor = ev["M_s42"]["floor"]

    # ---------------- P0 对账 ----------------
    add("## P0 复现对账（臂 M vs `struct_supervision` B / `bag_modules` A）\n")
    add("| 集 | seed | M 实测 | 参照 | Δ | SE | 门槛 2SE | P0 |")
    add("|---|---|---|---|---|---|---|---|")
    p0_ok = True
    for sp in ("test", "a_bal"):
        for s in SEEDS:
            got = ev[f"M_s{s}"]["splits"][sp]["skel"]["acc"]
            ref = P0_REF[sp][s]
            se = P0_SE[sp]
            d = got - ref
            ok = abs(d) <= 2 * se
            p0_ok &= ok
            add(f"| {sp} | {s} | {got:.4f} | {ref:.4f} | {d:+.4f} | {se:.4f} | "
                f"{2 * se:.4f} | {'过' if ok else '不过'} |")
    add("")

    # ---------------- 主表 ----------------
    add("## 主表（骨架 acc±行级 SE；pair = held-out 最小对）\n")
    add("| 臂 seed | B1 pair_success (n=520) | 门槛 .5+2SE | B1 2AFC | B1 per-side | "
        "B2 pair_success (n=40) | a_bal 骨架 | a_bal 卡−地板 | test 骨架 | test 卡−地板 |")
    add("|---|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        for s in SEEDS:
            e = ev[f"{a}_s{s}"]
            p1 = e["pair"]["b1_word"]
            p2 = e["pair"]["b2_order"]
            ab = e["splits"]["a_bal"]["skel"]
            te = e["splits"]["test"]["skel"]
            add(f"| {a} s{s} | **{p1['pair_success']:.4f}±{p1['se_pair']:.4f}** | "
                f"{p1['gate_0.5_2se']:.4f} | {p1['pair_2afc']:.4f} | "
                f"{p1['per_side_acc']:.4f} | **{p2['pair_success']:.4f}±{p2['se_pair']:.4f}** | "
                f"{ab['acc']:.4f}±{ab['se']:.4f} | {ab['acc'] - floor['a_bal']['max_naive_complete']:+.4f} | "
                f"{te['acc']:.4f}±{te['se']:.4f} | "
                f"{te['acc'] - floor['test']['max_naive_complete']:+.4f} |")
    add("")

    # ---------------- 配对 Δ ----------------
    add("## 配对 Δ（逐行 0/1 差的样本 SE）\n")
    add("| 臂−M | 集/指标 | Δ | SE | t | 2 seed 同号 |")
    add("|---|---|---|---|---|---|")
    pair_del: dict = {}
    for arm in ("MP", "MP+"):
        for key, getter in (
                ("B1 pair_success",
                 lambda e: e["pair"]["b1_word"]["both_list"]),
                ("B2 pair_success",
                 lambda e: e["pair"]["b2_order"]["both_list"]),
                ("a_bal skel", lambda e: e["correct"]["a_bal"]),
                ("test skel", lambda e: e["correct"]["test"])):
            ds = []
            for s in SEEDS:
                d = paired(getter(ev[f"M_s{s}"]), getter(ev[f"{arm}_s{s}"]))
                ds.append(d)
                add(f"| {arm}−M s{s} | {key} | {d['delta']:+.4f} | {d['se']:.4f} | "
                    f"{d['t']} | {'是' if ds[0]['pos'] == d['pos'] else '否'} |")
            pair_del.setdefault(f"{arm}|{key}", ds)
    add("")

    # ---------------- P3 L 分列 ----------------
    add("## P3：L 组（{#35,#0,#1,#2}）vs 非 L 组\n")
    add("| 臂 seed | test L | test 非L | a_lit L | a_lit 非L | b_pairs L 对 | b_pairs 非L 对 |")
    add("|---|---|---|---|---|---|---|")
    for a in ARMS:
        for s in SEEDS:
            e = ev[f"{a}_s{s}"]["L_split"]
            f = lambda x: "—" if x is None else f"{x['acc']:.4f}(n={x['n']})"
            g = lambda x: "—" if x is None else (
                f"ps={x['pair_success']:.4f}/side={x['per_side_acc']:.4f}"
                f"(n={x['n_pairs']})")
            add(f"| {a} s{s} | {f(e['test']['L'])} | {f(e['test']['nonL'])} | "
                f"{f(e['a_lit']['L'])} | {f(e['a_lit']['nonL'])} | "
                f"{g(e['b_pairs']['L'])} | {g(e['b_pairs']['nonL'])} |")
    add("")

    # ---------------- P4 机制探针 ----------------
    add("## P4 机制探针（每臂每 seed）\n")
    add("| 臂 seed | test 真→只打乱内容 | a_bal 真→只打乱内容 | b_pairs 真→只打乱内容 | "
        "test shuffle_all | a_bal shuffle_all | B1 pair_success 真→换内容 | "
        "B1 pair_2afc 真→换内容 | aux_only |")
    add("|---|---|---|---|---|---|---|---|---|")
    for a in ARMS:
        for s in SEEDS:
            e = ev[f"{a}_s{s}"]
            def cell(sp):
                r = e["probe"][sp]
                return (f"{r['true']['acc']:.4f}→{r['shuffled']['acc']:.4f}"
                        f"(Δ{r['drop']:+.4f},n={r['n_survivor']})")
            pc = e["probe"]["pair_content"]
            if pc.get("true_pair"):
                pcs = (f"{pc['true_pair']['pair_success']:.4f}→"
                       f"{pc['shuffled_pair']['pair_success']:.4f}")
                p2 = (f"{pc['true_pair']['pair_2afc']:.4f}→"
                      f"{pc['shuffled_pair']['pair_2afc']:.4f}")
            else:
                pcs = p2 = "—"
            aux = e["aux_only"]["replacement_n_slots_rule"]["test"]
            add(f"| {a} s{s} | {cell('test')} | {cell('a_bal')} | {cell('b_pairs')} | "
                f"{e['probe']['test']['shuffle_all']['acc']:.4f} | "
                f"{e['probe']['a_bal']['shuffle_all']['acc']:.4f} | {pcs} | {p2} | "
                f"{aux:.4f} |")
    add("")
    add(f"- `shuffle_all` 落回判定（门槛 = 多数类+2SE，test {THRESH['test']:.4f} / "
        f"a_bal {THRESH['a_bal']:.4f}）："
        + "；".join(
            f"{a} s{s} test {ev[f'{a}_s{s}']['probe']['test']['shuffle_all']['acc']:.4f}"
            f"{'≤' if ev[f'{a}_s{s}']['probe']['test']['shuffle_all']['acc'] <= THRESH['test'] else '>'}阈,"
            f"a_bal {ev[f'{a}_s{s}']['probe']['a_bal']['shuffle_all']['acc']:.4f}"
            f"{'≤' if ev[f'{a}_s{s}']['probe']['a_bal']['shuffle_all']['acc'] <= THRESH['a_bal'] else '>'}阈"
            for a in ARMS for s in SEEDS))
    add(f"- `pair_2afc` 机会水平 = 0.25（独立随机猜，PREREG §3.1 写死）；"
        f"`pair_shuffle_content` 两侧同用另一对的槽文本 ⇒ 功能词差异保留、内容换掉。")
    add(f"- `aux_only` 说明：{ev['M_s42']['aux_only']['reason']}")
    add(f"- 替代口径（模型无关常数）= `n_slots→train多数` 免费规则："
        + ", ".join(f"{k} {v:.4f}" for k, v in
                    ev["M_s42"]["aux_only"]["replacement_n_slots_rule"].items()))
    add("")

    # ---------------- P5 地板 ----------------
    add("## P5 完备免费地图（fit=train，含 n_slots）与 pair 地板\n")
    add("| 集 | n | 完备 max_naive | 命中规则 | skeleton_leak 参照 | pair 地板 | 命中规则 |")
    add("|---|---|---|---|---|---|---|")
    for sp in ("test", "a_bal", "a_lit", "b_pairs", "c_pairs"):
        f = floor[sp]
        ref = f.get("leak_stats_ref", {})
        add(f"| {sp} | {ref.get('n')} | {f['max_naive_complete']:.4f} | "
            f"{f['max_naive_complete_rule']} | {ref.get('max_naive_complete')} | "
            f"{f.get('max_naive_pair_success', '—')} | {f.get('max_naive_pair_rule', '—')} |")
    add("")

    # ---------------- P6 随机标签 ----------------
    add("## P6 随机标签负对照（门禁：≤ 多数类+2SE）\n")
    add("| 臂 seed | test | 阈 .4368 | 判 | a_bal | 阈 .0695 | 判 | B1 pair_success |")
    add("|---|---|---|---|---|---|---|---|")
    p6_ok = True
    for a in ARMS:
        for s in SEEDS:
            e = rnd[f"{a}_s{s}"]
            t = e["splits"]["test"]["skel"]["acc"]
            b = e["splits"]["a_bal"]["skel"]["acc"]
            p = e["pair"]["b1_word"]["pair_success"]
            ok_t, ok_b = t <= THRESH["test"], b <= THRESH["a_bal"]
            p6_ok &= (ok_t and ok_b)
            add(f"| {a} s{s} | {t:.4f} | {THRESH['test']} | {'过' if ok_t else '不过'} | "
                f"{b:.4f} | {THRESH['a_bal']} | {'过' if ok_b else '不过'} | {p:.4f} |")
    add("")

    # ---------------- 2 seed 同号 + 判定 ----------------
    ps = {f"{a}_s{s}": ev[f"{a}_s{s}"]["pair"]["b1_word"] for a in ARMS for s in SEEDS}
    gate = ps["M_s42"]["gate_0.5_2se"]
    add("## P1 主判（held-out 最小对 B1 `pair_success`，机会 0.5）\n")
    add(f"- 门槛 = 0.5 + 2×SE = **{gate:.4f}**（SE = sqrt(0.25/520) = "
        f"{ps['M_s42']['se_pair']:.4f}）\n")
    add("| 臂 | s42 | s43 | 两 seed 过门槛 | 两 seed 同号(相对 .5) |")
    add("|---|---|---|---|---|")
    verdict_arm = {}
    for a in ARMS:
        a42, a43 = ps[f"{a}_s42"]["pair_success"], ps[f"{a}_s43"]["pair_success"]
        both = a42 > gate and a43 > gate
        same = (a42 > 0.5) == (a43 > 0.5)
        verdict_arm[a] = {"s42": a42, "s43": a43, "both_over": both, "same_sign": same}
        add(f"| {a} | {a42:.4f} | {a43:.4f} | {'是' if both else '否'} | "
            f"{'是' if same else '否'} |")
    add("")
    add("### per-side acc 伴随口径（= skeleton_leak 成对机会 0.5 比较对象）\n")
    add("| 臂 | s42 | s43 | 门槛 0.5+2SE_side | 两 seed 过 |")
    add("|---|---|---|---|---|")
    side_gate = 0.5 + 2 * ps["M_s42"]["per_side_se"]
    for a in ARMS:
        x42 = ps[f"{a}_s42"]["per_side_acc"]
        x43 = ps[f"{a}_s43"]["per_side_acc"]
        add(f"| {a} | {x42:.4f} | {x43:.4f} | {side_gate:.4f} | "
            f"{'是' if (x42 > side_gate and x43 > side_gate) else '否'} |")
    add("")

    # 判定
    mp = verdict_arm["MP"]
    over = mp["both_over"] and mp["same_sign"]
    side_over = (ps["MP_s42"]["per_side_acc"] > side_gate
                 and ps["MP_s43"]["per_side_acc"] > side_gate)
    if p0_ok and p6_ok and over:
        verdict = "最小对判别有效（功能词已被表示）"
        why = "P0 过 ∧ P1 主判两 seed 均 > 0.5+2SE 且同号 ∧ P6 过"
    elif p0_ok and p6_ok and (not over) and (not side_over) and mp["same_sign"]:
        verdict = "无效（仍不读功能词）"
        why = ("P0 过 ∧ P6 过 ∧ P1 主判两 seed 均未过 ∧ per-side acc 亦 ≤ 0.5+2SE "
               "∧ 两 seed 同判")
    else:
        verdict = "证据不足"
        bits = []
        if not p0_ok:
            bits.append("P0 未过")
        if not p6_ok:
            bits.append("P6 未过")
        if not mp["same_sign"]:
            bits.append("两 seed 不同号")
        if (not over) and side_over:
            bits.append("P1 未过但 per-side > 0.5+2SE（有部分判别信号）")
        if over and not (p0_ok and p6_ok):
            bits.append("P1 过但门禁未过")
        why = "；".join(bits) or "其余情形"
    add("## 判定（三选一）\n")
    add(f"**{verdict}** —— 依据：{why}")
    add("")
    add(f"- P0 = {'过' if p0_ok else '不过'}；P6 = {'过' if p6_ok else '不过'}")
    add(f"- 随机标签 B1 pair_success（对照）："
        + ", ".join(f"{a}: {rnd[f'{a}_s42']['pair']['b1_word']['pair_success']:.4f}"
                    for a in ARMS))

    txt = "\n".join(L)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "tables.md").write_text(txt, encoding="utf-8")
    (RESULTS / "verdict.json").write_text(json.dumps(
        {"verdict": verdict, "why": why, "p0_ok": p0_ok, "p6_ok": p6_ok,
         "p1": verdict_arm, "gate": gate, "side_gate": side_gate,
         "paired": pair_del}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(txt)
    print(f"\n[done] → results/tables.md, results/verdict.json")


if __name__ == "__main__":
    main()
