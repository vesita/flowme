#!/usr/bin/env python3
"""W1–W5 汇总 + 判定（PREREG §4）。只读结果文件，不改任何实验。

口径：
  · 干扰量 Δ = 单通道臂 − 共享臂（C），**配对 SE**：同一批 test 样本逐条 0/1 之差
    `sd(d)/sqrt(n)`（d ∈ {−1,0,1}）；报 Δ 与 Δ/SE，两个 seed 各报一次。
  · 骨架 / 指派 / 联合**分开报**，不合成一个数。
  · 判定三选一（PREREG §4 末），不许硬选。

用法：uv run python experiments/two_channel_head/analyze.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
SEEDS = (42, 43)


def load(name: str) -> dict:
    p = RES / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"缺结果：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def paired(d: list[float]) -> dict:
    """d_i = correct_single(i) − correct_shared(i)；返回 Δ、配对 SE、Δ/SE。"""
    n = len(d)
    m = sum(d) / n
    if n > 1:
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
        se = sd / math.sqrt(n)
    else:
        se = 0.0
    return {"n": n, "delta": round(m, 6), "se_paired": round(se, 6),
            "delta_over_se": round(m / se, 2) if se > 0 else float("inf"),
            "same_sign_rule": "Δ ≤ 2SE（可接受）" if abs(m) <= 2 * se else "Δ > 2SE（干扰显著）"}


def acc(xs: list) -> float:
    return sum(xs) / len(xs)


def main() -> None:
    out: dict = {}

    # ---------- 载入 ----------
    R = {n: load(n) for n in
         [f"{a}_s{s}{r}" for a in "ABCD" for s in SEEDS for r in ("",)]
         + ["A_s42_rand", "B_s42_rand"]}
    stats = json.loads((HERE / "data" / "stats.json").read_text(encoding="utf-8"))
    timing = json.loads((RES / "timing_compare.json").read_text(encoding="utf-8"))
    try:
        w5 = json.loads((RES / "old_cards_delta.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        w5 = None
    sc = json.loads((HERE / "logs" / "selfcheck.json").read_text(encoding="utf-8"))

    # ---------- 主表 ----------
    print("## 主表 · 指针通道（test n=2500，SE=1.00pt；adv 同 n）\n")
    print("| 臂 | seed | test acc | ±SE | 余量/SE | adv acc | ±SE | shifted_pos | heldout_pair |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a in "ABCD":
        for s in SEEDS:
            m = R[f"{a}_s{s}"]["meta"]["eval"]
            print(f"| {a} | {s} | {m['ptr_test']['acc']:.4f} | {m['ptr_test']['se']:.4f} "
                  f"| {m['ptr_test']['margin_over_se']:+.1f} | {m['ptr_adv']['acc']:.4f} "
                  f"| {m['ptr_adv']['se']:.4f} | {m['ptr_adv_shifted_pos']['acc']:.4f} "
                  f"| {m['ptr_adv_heldout_pair']['acc']:.4f} |")
    mr = R["A_s42_rand"]["meta"]["eval"]
    print(f"| A-randlabel | 42 | {mr['ptr_test']['acc']:.4f} | {mr['ptr_test']['se']:.4f} "
          f"| {mr['ptr_test']['margin_over_se']:+.1f} | {mr['ptr_adv']['acc']:.4f} | "
          f"{mr['ptr_adv']['se']:.4f} | {mr['ptr_adv_shifted_pos']['acc']:.4f} | "
          f"{mr['ptr_adv_heldout_pair']['acc']:.4f} |")

    print("\n## 主表 · 生成通道（test n=2500；骨架/指派/联合**分开**）\n")
    print("| 臂 | seed | 骨架 acc | ±SE | 余量/SE | 指派 acc | ±SE | 联合 acc | ±SE |")
    print("|---|---|---|---|---|---|---|---|---|")
    for a in "BCD":
        for s in SEEDS:
            m = R[f"{a}_s{s}"]["meta"]["eval"]["gen_test"]
            print(f"| {a} | {s} | {m['skel']['acc']:.4f} | {m['skel']['se']:.4f} "
                  f"| {m['skel']['margin_over_se']:+.1f} | {m['slot']['acc']:.4f} "
                  f"| {m['slot']['se']:.4f} | {m['joint']['acc']:.4f} | {m['joint']['se']:.4f} |")
    mr = R["B_s42_rand"]["meta"]["eval"]["gen_test"]
    print(f"| B-randlabel | 42 | {mr['skel']['acc']:.4f} | {mr['skel']['se']:.4f} "
          f"| {mr['skel']['margin_over_se']:+.1f} | {mr['slot']['acc']:.4f} "
          f"| {mr['slot']['se']:.4f} | {mr['joint']['acc']:.4f} | {mr['joint']['se']:.4f} |")

    # ---------- W1 干扰（配对 SE） ----------
    print("\n## W1 · 两通道互相干扰（Δ = 单通道臂 − C，配对 SE）\n")
    print("| 通道 | 指标 | seed | A/B acc | C acc | Δ | 配对 SE | Δ/SE | 判 |")
    print("|---|---|---|---|---|---|---|---|---|")
    w1 = {}
    for ch, single, key, field in [
            ("指针", "A", "ptr_test", None),
            ("指针", "A", "ptr_adv", None),
            ("指针", "A", "ptr_adv_shifted_pos", None),
            ("指针", "A", "ptr_adv_heldout_pair", None)]:
        w1.setdefault(ch, [])
        for s in SEEDS:
            c = R[f"{single}_s{s}"]["correct"][key]
            h = R[f"C_s{s}"]["correct"][key]
            st = paired([x - y for x, y in zip(c, h)])
            st.update({"metric": key, "seed": s,
                       "single_acc": round(acc(c), 4), "C_acc": round(acc(h), 4)})
            w1[ch].append(st)
            print(f"| {ch} | {key} | {s} | {acc(c):.4f} | {acc(h):.4f} | "
                  f"{st['delta']:+.4f} | {st['se_paired']:.4f} | "
                  f"{st['delta_over_se']:+.2f} | {st['same_sign_rule']} |")
    for ch, single, sub in [("生成", "B", "skel"), ("生成", "B", "slot_frac"),
                            ("生成", "B", "joint")]:
        for s in SEEDS:
            c = R[f"{single}_s{s}"]["correct"]["gen_test"][sub]
            h = R[f"C_s{s}"]["correct"]["gen_test"][sub]
            st = paired([x - y for x, y in zip(c, h)])
            st.update({"metric": f"gen_{sub}", "seed": s,
                       "single_acc": round(acc(c), 4), "C_acc": round(acc(h), 4)})
            w1.setdefault(ch, []).append(st)
            print(f"| {ch} | gen_{sub} | {s} | {acc(c):.4f} | {acc(h):.4f} | "
                  f"{st['delta']:+.4f} | {st['se_paired']:.4f} | "
                  f"{st['delta_over_se']:+.2f} | {st['same_sign_rule']} |")
    out["w1"] = w1

    # ---------- W2 参数与前向时间 ----------
    print("\n## W2 · 共用主干省了多少（参数逐项 + 前向时间）\n")
    print("| 臂 | trunk_ptr | trunk_gen | ptr_out | gen_skel | gen_assign | 头可训合计 |")
    print("|---|---|---|---|---|---|---|")
    p4 = {a: R[f"{a}_s42"]["meta"]["params"] for a in "ABCD"}
    for a in "ABCD":
        p = p4[a]
        print(f"| {a} | {p['trunk_ptr']} | {p['trunk_gen']} | {p['ptr_out']} "
              f"| {p['gen_skel']} | {p['gen_assign']} | **{p['head_trainable']}** |")
    d, c = p4["D"]["head_trainable"], p4["C"]["head_trainable"]
    print(f"\nC vs D：{c} vs {d} → **省 {d - c} 参数（{(d - c) / d * 100:.2f}%）**"
          f"（= 一个主干 {p4['D']['trunk_ptr']}）；D = A + B = "
          f"{p4['A']['head_trainable']} + {p4['B']['head_trainable']} = {d}；"
          f"A 与 select_rerank 打分头逐项相同（{p4['A']['head_trainable']}）。")
    print("\n前向时间（轮转交错、n=60、p50 为主 + 噪声地板）：\n")
    print("| 量 | A | C | D | C/D | D/A |")
    print("|---|---|---|---|---|---|")
    fp = {k: v["p50_ms"] for k, v in timing["fwd_ptr"].items()}
    fg = {k: v["p50_ms"] for k, v in timing["fwd_gen"].items()}
    tot = {"A": fp["A"], "C": fp["C"] + fg["C"], "D": fp["D"] + fg["D"]}
    st = {k: v["p50_ms"] for k, v in timing["step"].items()}
    sth = {k: v["p50_ms"] for k, v in timing["step_h2d"].items()}
    print(f"| 指针路径 fwd (ms) | {fp['A']:.4f} | {fp['C']:.4f} | {fp['D']:.4f} "
          f"| {fp['C'] / fp['D']:.3f} | {fp['D'] / fp['A']:.3f} |")
    print(f"| 生成路径 fwd (ms) | — | {fg['C']:.4f} | {fg['D']:.4f} | "
          f"{fg['C'] / fg['D']:.3f} | — |")
    print(f"| 每卡 fwd 合计 (ms) | {tot['A']:.4f} | {tot['C']:.4f} | {tot['D']:.4f} "
          f"| **{tot['C'] / tot['D']:.3f}** | **{tot['D'] / tot['A']:.3f}** |")
    print(f"| 整步 fwd+bwd+opt (ms) | {st['A']:.4f} | {st['C']:.4f} | {st['D']:.4f} "
          f"| {st['C'] / st['D']:.3f} | {st['D'] / st['A']:.3f} |")
    print(f"| 整步含 H2D（同训练）(ms) | {sth['A']:.4f} | {sth['C']:.4f} | "
          f"{sth['D']:.4f} | {sth['C'] / sth['D']:.3f} | {sth['D'] / sth['A']:.3f} |")
    print(f"\n噪声地板（A vs A2 同模型同输入）= {timing['noise_floor_ms']} ms；"
          f"同步开销 p50 = {timing['sync_overhead_ms']} ms；对照 = {timing['control']}")
    out["w2"] = {"params": p4, "saved": d - c, "saved_pct": round((d - c) / d * 100, 2),
                 "timing": timing}

    # 训练逐步墙钟（逐步对账）
    print("\n训练内逐步墙钟（warm-skip 后，p50/mean）：\n")
    print("| 臂 | seed | step p50 (ms) | step mean (ms) | fwd 指针 p50 | fwd 生成 p50 |")
    print("|---|---|---|---|---|---|")
    for a in "ABCD":
        for s in SEEDS:
            t = R[f"{a}_s{s}"]["meta"]["timing"]
            sp = t.get("train_step_ms", {})
            fp50 = t.get("train_fwd_ms", {}).get("ptr", {}).get("p50")
            fg50 = t.get("train_fwd_ms", {}).get("gen", {}).get("p50")
            print(f"| {a} | {s} | {sp.get('p50')} | {sp.get('mean')} | "
                  f"{fp50 if fp50 is not None else '—'} | "
                  f"{fg50 if fg50 is not None else '—'} |")

    # ---------- W3 随机标签 ----------
    print("\n## W3 · 随机标签对照（两通道各一，真执行）\n")
    ra = R["A_s42_rand"]["meta"]["eval"]
    rb = R["B_s42_rand"]["meta"]["eval"]["gen_test"]
    na = stats["naive_assign"]
    w3 = {
        "ptr_rand": {"acc": ra["ptr_test"]["acc"], "blind": 0.5,
                     "pass": abs(ra["ptr_test"]["acc"] - 0.5) <= 2 * ra["ptr_test"]["se"]},
        "gen_skel_rand": {"acc": rb["skel"]["acc"],
                          "majority": stats["sets"]["train"]["majority_baseline"],
                          "blind": stats["sets"]["train"]["blind_guess"],
                          "pass": abs(rb["skel"]["acc"]
                                      - stats["sets"]["train"]["majority_baseline"])
                          <= 2 * rb["skel"]["se"]},
        "gen_slot_rand": {"acc": rb["slot"]["acc"], "blind_uniform": na["blind_uniform"],
                          "identity_rule": na["identity_rule"],
                          "pass": abs(rb["slot"]["acc"] - na["blind_uniform"])
                          <= 3 * rb["slot"]["se"]},
        "randlabel_executed": True,
    }
    print(f"- 指针 randlabel：acc={w3['ptr_rand']['acc']:.4f} vs 盲猜 0.5 "
          f"(2SE={2 * ra['ptr_test']['se']:.4f}) → {'过' if w3['ptr_rand']['pass'] else '不过'}")
    print(f"- 生成骨架 randlabel：acc={w3['gen_skel_rand']['acc']:.4f} vs 多数类 "
          f"{w3['gen_skel_rand']['majority']:.4f} / 盲猜 "
          f"{w3['gen_skel_rand']['blind']:.4f} → "
          f"{'过' if w3['gen_skel_rand']['pass'] else '不过'}")
    print(f"- 生成指派 randlabel：acc={w3['gen_slot_rand']['acc']:.4f} vs 盲猜 "
          f"{w3['gen_slot_rand']['blind_uniform']:.4f} / identity "
          f"{w3['gen_slot_rand']['identity_rule']:.4f} → "
          f"{'过' if w3['gen_slot_rand']['pass'] else '不过'}")
    out["w3"] = w3

    # ---------- W4 朴素规则 ----------
    w4 = {"max_naive_train": stats["max_naive_train"],
          "threshold": 0.90, "pass": stats["w4_pass"],
          "naive_train": stats["naive_skeleton_train"],
          "naive_test": stats["naive_skeleton_test"],
          "naive_assign": stats["naive_assign"],
          "annotation_inverse": stats["annotation_inverse"]}
    print(f"\n## W4 · 朴素规则（train）max_naive = {w4['max_naive_train']:.4f} < 0.90 "
          f"→ {'过' if w4['pass'] else '不过（数据集无效，停）'}（单列："
          f"annotation_inverse = {stats['naive_assign']['annotation_inverse_pos_sort']}，"
          f"不计入 max_naive）")
    out["w4"] = w4

    # ---------- W5 老卡 ----------
    if w5:
        print("\n## W5 · 四张老卡 Δ（核冻结 ⇒ 预期 0）\n")
        print("| 卡 | before | after | Δ | 噪声带 | 带内 | 指标逐位同 |")
        print("|---|---|---|---|---|---|---|")
        for k, v in w5.items():
            if k == "files_sha256_identical":
                continue
            print(f"| {k} | {v['before']:.6f} | {v['after']:.6f} | {v['delta']:+.6f} "
                  f"| {v['noise_band']:.4f} | {abs(v['delta']) <= v['noise_band']} "
                  f"| {v['metric_dict_identical']} |")
        print(f"\n文件 sha256 训前=训后：**{w5['files_sha256_identical']}**")
        w5_pass = all(v["delta"] == 0 for k, v in w5.items()
                      if k != "files_sha256_identical")
        out["w5"] = {"per_card": w5, "pass": w5_pass}
    else:
        print("\nW5：old_cards_delta.json 还没生成（训后跑还没做）")
        w5_pass = None

    # ---------- G1 / G2 证据强度门 ----------
    b_skel = R["B_s42"]["meta"]["eval"]["gen_test"]["skel"]
    b43 = R["B_s43"]["meta"]["eval"]["gen_test"]["skel"]
    g1 = {"B_acc_42": b_skel["acc"], "B_acc_43": b43["acc"],
          "max_naive": stats["max_naive_train"],
          "bar": stats["max_naive_train"] + 2 * b_skel["se"],
          "pass_42": b_skel["acc"] > stats["max_naive_train"] + 2 * b_skel["se"],
          "pass_43": b43["acc"] > stats["max_naive_train"] + 2 * b43["se"]}
    g2_rows = []
    for s in SEEDS:
        c = R[f"C_s{s}"]["meta"]["eval"]
        g2_rows.append({"seed": s,
                        "ptr_test": c["ptr_test"]["acc"],
                        "gen_skel": c["gen_test"]["skel"]["acc"],
                        "gen_slot": c["gen_test"]["slot"]["acc"],
                        "ptr_ok": c["ptr_test"]["acc"] > 0.5 + 2 * c["ptr_test"]["se"],
                        "skel_ok": c["gen_test"]["skel"]["acc"] > 0.025
                        + 2 * c["gen_test"]["skel"]["se"],
                        "slot_ok": c["gen_test"]["slot"]["acc"] > 0.4231
                        + 2 * c["gen_test"]["slot"]["se"]})
    print(f"\n## G1 生成通道是否超过免费规则：B 骨架 {g1['B_acc_42']:.4f}/{g1['B_acc_43']:.4f} "
          f"vs max_naive {g1['max_naive']:.4f} + 2SE = {g1['bar']:.4f} → "
          f"{'过' if g1['pass_42'] and g1['pass_43'] else '不过'}")
    print(f"## G2 C 是否掉到地板：{g2_rows}")
    out["g1"] = g1
    out["g2"] = g2_rows

    # ---------- 判定 ----------
    w1_ptr_ok = all(x["se_paired"] >= 0 and abs(x["delta"]) <= 2 * x["se_paired"]
                    for x in w1["指针"])
    w1_gen_skel = [x for x in w1["生成"] if x["metric"] == "gen_skel"]
    w1_gen_ok = all(abs(x["delta"]) <= 2 * x["se_paired"] for x in w1_gen_skel)
    w1_gen_slot = [x for x in w1["生成"] if x["metric"] == "gen_slot_frac"]
    w1_slot_ok = all(abs(x["delta"]) <= 2 * x["se_paired"] for x in w1_gen_slot)
    w3_ok = w3["ptr_rand"]["pass"] and w3["gen_skel_rand"]["pass"] \
        and w3["gen_slot_rand"]["pass"]
    signs_ptr = [1 if x["delta"] > 0 else -1 for x in w1["指针"]]
    signs_skel = [1 if x["delta"] > 0 else -1 for x in w1_gen_skel]

    verdict = ""
    reason = []
    if not (w3_ok and w4["pass"]):
        verdict = "证据不足"
        reason.append("W3/W4 未过")
    elif not (g1["pass_42"] and g1["pass_43"]):
        verdict = "证据不足"
        reason.append("G1：生成通道未超过免费规则")
    elif any(not r["ptr_ok"] or not r["skel_ok"] or not r["slot_ok"] for r in g2_rows):
        verdict = "证据不足"
        reason.append("G2：C 某通道掉到地板")
    elif w1_ptr_ok and w1_gen_ok and w1_slot_ok:
        verdict = "同核双通道可行"
        reason.append("W1 两通道 Δ ≤ 2SE（两 seed）")
    elif (all(s > 0 for s in signs_ptr) and not w1_ptr_ok) or \
            (all(s > 0 for s in signs_skel) and not w1_gen_ok):
        verdict = "必须分头（或加私有适配层）"
        reason.append("W1 干扰显著且两 seed 同号")
    else:
        verdict = "证据不足"
        reason.append("两 seed 不同号 / Δ 与 SE 同量级")
    if w5_pass is False:
        verdict = "证据不足"
        reason.append("W5 老卡 Δ ≠ 0（核被动过）")
    print(f"\n# 判定：**{verdict}**（{'；'.join(reason)}）")
    print(f"  W1指针过={w1_ptr_ok} W1骨架过={w1_gen_ok} W1指派过={w1_slot_ok} "
          f"W3过={w3_ok} W4过={w4['pass']} W5过={w5_pass} "
          f"指针Δ符号={signs_ptr} 骨架Δ符号={signs_skel}")
    out["verdict"] = verdict
    out["verdict_reason"] = reason
    out["w1_flags"] = {"ptr": w1_ptr_ok, "gen_skel": w1_gen_ok, "gen_slot": w1_slot_ok,
                       "signs_ptr": signs_ptr, "signs_skel": signs_skel}
    out["freeze"] = sc.get("freeze")
    out["init_identical"] = sc.get("init_identical")
    out["share_evidence"] = sc.get("share_evidence")
    (RES / "analyze.json").write_text(json.dumps(out, ensure_ascii=False, indent=2,
                                                  default=str), encoding="utf-8")
    print(f"\n[done] results/analyze.json")


if __name__ == "__main__":
    sys.exit(main())
