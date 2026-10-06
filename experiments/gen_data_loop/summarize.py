#!/usr/bin/env python3
"""汇总 Phase 2：主表 + D4(a)/D4(b)/D5/V-bonus + D6/D6′ + 结论三分（PREREG_PHASE2.md §6）。

    uv run python experiments/gen_data_loop/summarize.py

只读 results/eval_*.json 与 data/stats.json，产出 results/summary.md + results/summary.json。
**不改任何门槛**；所有门槛常量与 PREREG_PHASE2.md 逐字对应。
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
MAIN_SETS = ("natural", "adversarial", "ctrl_adversarial")

M = 0.5
N_NAT = 200
SE_M = math.sqrt(M * (1 - M) / N_NAT)          # 0.035355 → 3.54pt
THRESH_A = M + 2 * SE_M                          # 0.5707 → 57.1%
THRESH_SHUF_TEST = 0.53                          # V-bonus：自己 test ≤ 50% + 3pt


def se(p: float, n: int) -> float:
    return math.sqrt(max(p, 1e-12) * max(1 - p, 1e-12) / n)


def load(tag: str) -> dict:
    p = RES / f"eval_{tag}.json"
    if not p.exists():
        raise SystemExit(f"缺 {p} —— 先跑 run_all.sh")
    return json.loads(p.read_text(encoding="utf-8"))


def cell(rep: dict, split: str) -> dict:
    return rep["splits"][split]


def main() -> int:
    tags = {t: load(t) for t in
            ("P_s42", "P_s43", "C_s42", "C_s43",
             "P_shuf_s42", "P_shuf_s43", "C_shuf_s42", "C_shuf_s43")}
    stats = json.loads((HERE / "data" / "stats.json").read_text(encoding="utf-8"))

    lines: list[str] = []
    js: dict = {"thresholds": {"majority": M, "n_natural": N_NAT,
                               "se_majority": round(SE_M, 6),
                               "D4a_threshold": round(THRESH_A, 4)},
                "main": {}, "d4a": {}, "d4b": {}, "v_bonus": {}, "d6": {}}

    # ---------- 主表 ----------
    lines.append("## 主表（acc；每格并列 多数类 / SE / 余量 / 余量-SE / max_naive / 卡−max_naive）\n")
    lines.append("| 臂 | seed | 集 | n | acc | 多数类 | SE | 余量(acc−M) | 余量/SE | max_naive | 卡−max_naive | 备注 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in ("P", "C"):
        for seed in (42, 43):
            rep = tags[f"{arm}_s{seed}"]
            for split in MAIN_SETS:
                c = cell(rep, split)
                n = c["n"]
                acc = c["acc"]
                m = c["majority_baseline"]
                s = se(acc, n)
                margin = acc - m
                ms = margin / s
                mn = c["max_naive"]["best_dir"]
                d = acc - mn
                note = "" if d > 0 else "**未超过免费规则**"
                if split == "natural":
                    if acc >= THRESH_A:
                        note = (note + " " if note else "") + "D4(a)过"
                    else:
                        note = (note + " " if note else "") + "**D4(a)不过**"
                lines.append(f"| {arm} | {seed} | {split} | {n} | {acc:.4f} | {m:.4f} | "
                             f"{s:.4f} | {margin:+.4f} | {ms:+.2f} | {mn:.4f}"
                             f"({c['max_naive']['rule']}) | {d:+.4f} | {note} |")
                js["main"].setdefault(f"{arm}_{seed}", {})[split] = {
                    "acc": acc, "majority": m, "se": round(s, 4),
                    "margin": round(margin, 4), "margin_over_se": round(ms, 3),
                    "max_naive": mn, "max_naive_rule": c["max_naive"]["rule"],
                    "card_minus_max_naive": round(d, 4), "note": note}
    lines.append("")

    # ---------- 随机标签对照（V-bonus） ----------
    lines.append("## 随机标签对照（shuf，V-bonus）\n")
    lines.append("| 臂 | seed | natural acc | ≤0.5707? | 自己 test | 自己 test 集 | ≤0.53? |")
    lines.append("|---|---|---|---|---|---|---|")
    v_ok = True
    for arm in ("P", "C"):
        own = "test" if arm == "P" else "ctrl_test"
        for seed in (42, 43):
            rep = tags[f"{arm}_shuf_s{seed}"]
            a = cell(rep, "natural")["acc"]
            t = cell(rep, own)["acc"]
            ok1 = a <= THRESH_A
            ok2 = t <= THRESH_SHUF_TEST
            v_ok &= (ok1 and ok2)
            lines.append(f"| {arm} | {seed} | {a:.4f} | {'✅' if ok1 else '❌'} | "
                         f"{t:.4f} | {own} | {'✅' if ok2 else '❌'} |")
            js["v_bonus"][f"{arm}_shuf_s{seed}"] = {"natural_acc": a, "own_test_acc": t,
                                                    "own_test_split": own,
                                                    "ok_natural": ok1, "ok_test": ok2}
    lines.append("")
    js["v_bonus"]["pass"] = bool(v_ok)

    # ---------- D4(a) ----------
    a_ok = True
    lines.append("## D4(a)：提案臂在 natural 上 ≥ 多数类 + 2×SE = 0.5707（2 seed 各自满足）\n")
    lines.append("| seed | acc | 门槛 | 余量 | 余量/SE | 过? |")
    lines.append("|---|---|---|---|---|---|")
    for seed in (42, 43):
        c = cell(tags[f"P_s{seed}"], "natural")
        acc = c["acc"]
        s = se(acc, N_NAT)
        marg = acc - THRESH_A
        ok = acc >= THRESH_A
        a_ok &= ok
        lines.append(f"| {seed} | {acc:.4f} | {THRESH_A:.4f} | {marg:+.4f} | "
                     f"{marg/s:+.2f} | {'✅' if ok else '❌'} |")
        js["d4a"][f"s{seed}"] = {"acc": acc, "threshold": round(THRESH_A, 4),
                                 "margin": round(marg, 4),
                                 "margin_over_se": round(marg / s, 3), "pass": ok}
    lines.append("")
    js["d4a"]["pass"] = bool(a_ok)

    # ---------- D4(b) ----------
    b_ok = True
    lines.append("## D4(b)：提案臂 − 注入臂（natural，同 seed 配对）> 2×SE_diff\n")
    lines.append("| seed | acc_P | acc_C | 差 | SE_diff | 2×SE_diff | 差/SE_diff | 过? |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for seed in (42, 43):
        p = cell(tags[f"P_s{seed}"], "natural")["acc"]
        cc = cell(tags[f"C_s{seed}"], "natural")["acc"]
        sd = math.sqrt(p * (1 - p) / N_NAT + cc * (1 - cc) / N_NAT)
        d = p - cc
        ok = d > 2 * sd and d > 0
        b_ok &= ok
        lines.append(f"| {seed} | {p:.4f} | {cc:.4f} | {d:+.4f} | {sd:.4f} | {2*sd:.4f} | "
                     f"{d/sd:+.2f} | {'✅' if ok else '❌'} |")
        js["d4b"][f"s{seed}"] = {"acc_P": p, "acc_C": cc, "diff": round(d, 4),
                                 "se_diff": round(sd, 4), "two_se_diff": round(2 * sd, 4),
                                 "diff_over_se": round(d / sd, 3), "pass": bool(ok)}
    lines.append("")
    js["d4b"]["pass"] = bool(b_ok)

    # ---------- D6 / D6′ ----------
    prop = stats["proposal"]["train"]
    ctrl = stats["control"]["ctrl_train"]
    pool = stats["pipeline_meta"]["train"]["pool_distinct_candidate_rate"]
    d6_old = prop["diversity"]["distinct_candidate_rate"] >= 0.9 * ctrl["diversity"]["distinct_candidate_rate"]
    d6p1 = prop["diversity"]["distinct_candidate_rate"] >= pool
    d6p2 = prop["diversity"]["category_entropy_nats"] >= ctrl["diversity"]["category_entropy_nats"]
    d6p3 = prop["diversity"]["distinct_input_rate"] >= ctrl["diversity"]["distinct_input_rate"]
    lines.append("## D6 / D6′（**两个数字都报**，PREREG_PHASE2 §5）\n")
    lines.append("| 判据 | 口径 | 实测 | 过? |")
    lines.append("|---|---|---|---|")
    lines.append(f"| D6（原，未订正） | distinct 候选率 ≥ 0.9×注入臂 | "
                 f"{prop['diversity']['distinct_candidate_rate']:.4f} vs 0.9×"
                 f"{ctrl['diversity']['distinct_candidate_rate']:.4f}="
                 f"{0.9*ctrl['diversity']['distinct_candidate_rate']:.4f} | "
                 f"{'✅' if d6_old else '❌ 不过'} |")
    lines.append(f"| D6′-1 | 采样后 distinct 候选率 ≥ 原始提议池 | {pool:.4f} → "
                 f"{prop['diversity']['distinct_candidate_rate']:.4f} | {'✅' if d6p1 else '❌'} |")
    lines.append(f"| D6′-2 | 类别熵 ≥ 注入臂 | {prop['diversity']['category_entropy_nats']:.4f} vs "
                 f"{ctrl['diversity']['category_entropy_nats']:.4f} | {'✅' if d6p2 else '❌'} |")
    lines.append(f"| D6′-3 | distinct 输入率 ≥ 注入臂 | {prop['diversity']['distinct_input_rate']:.4f} vs "
                 f"{ctrl['diversity']['distinct_input_rate']:.4f} | {'✅' if d6p3 else '❌'} |")
    lines.append("")
    js["d6"] = {"original_pass": bool(d6_old), "d6p1": bool(d6p1), "d6p2": bool(d6p2),
                "d6p3": bool(d6p3), "d6p_pass": bool(d6p1 and d6p2 and d6p3)}

    # ---------- 引用地基 ----------
    naive_train = stats["proposal"]["train"]["naive"]["max_naive"]
    js["foundations"] = {
        "D1": "标签 100% 由 P2–P5 谓词判出（引用 PREREG §4.1）",
        "D2": [stats["pipeline_meta"][k]["accept_rate"] for k in ("train", "test", "adversarial")],
        "D3_train_max_naive": naive_train,
        # text/input/full 两两不相交必须全 0（candidate 属词表共享，单列不算泄漏）
        "overlap_text_input_full_nonzero": {
            kk: {k2: v for k2, v in d.items() if k2 in ("text", "input", "full") and v != 0}
            for kk, d in stats["overlap_check"].items()},
    }

    # ---------- 结论三分 ----------
    d1 = d2 = d3 = True   # 数据侧已核验，引用
    d3_pass = naive_train["best_dir"] < 0.90
    if d1 and d2 and d3_pass and a_ok and b_ok and d6p1 and d6p2 and d6p3 and v_ok:
        verdict, why = "①", "可用且优于注入式"
    elif d1 and d2 and d3_pass and a_ok and d6p1 and d6p2 and d6p3 and v_ok:
        verdict, why = "②", "可用但不优于注入式（D4(b) 不过）"
    else:
        verdict, why = "③", "产不出可用数据（D4(a) 或 D3/D6′/V-bonus 不过）"
    js["verdict"] = {"code": verdict, "why": why,
                     "D3_pass": bool(d3_pass), "D5": "不适用（本轮不走联合训练）"}
    lines.append("## 判定\n")
    lines.append(f"- **{verdict} {why}**")
    lines.append(f"- D1/D2/D3 = 引用地基（D3 train max_naive = {naive_train['best_dir']}"
                 f"({naive_train['rule']}) < 0.90 ⇒ {'过' if d3_pass else '不过'}）")
    lines.append(f"- D4(a) {'过' if a_ok else '不过'} / D4(b) {'过' if b_ok else '不过'} / "
                 f"D6′ {'过' if (d6p1 and d6p2 and d6p3) else '不过'} / "
                 f"V-bonus {'过' if v_ok else '不过'}")
    lines.append("- **D5 = 不适用**：本轮只训头、不走联合训练（PREREG_PHASE2 §0）")
    lines.append("- **D6（原判据）**：" + ("过" if d6_old else "**不过**（0.238 < 0.9×1.0）") +
                 "；**D6′（订正后）**：" + ("全过" if (d6p1 and d6p2 and d6p3) else "**不过**"))
    lines.append("")

    md = "\n".join(lines)
    (RES / "summary.md").write_text(md + "\n", encoding="utf-8")
    (RES / "summary.json").write_text(json.dumps(js, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(md)
    print(f"\n[save] {RES/'summary.md'}  {RES/'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
