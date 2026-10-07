#!/usr/bin/env python3
"""P7 汇总：M0–M6 逐条实测 + 主表 + 真实样例（口径 = PREREG §7）。

用法：uv run python experiments/sentence_mode/analyze.py
产物：`results/summary.json`
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
ROOT = HERE.parents[1]

import labels as L  # noqa: E402


def L_MODES(i: int) -> str:
    return L.MODES[i]
import mode_render as MR  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
ARMS = ("KEEP", "MASK", "MIX")
SEEDS = (42, 43)
EXITS = ("keep", "maskfinal", "mask", "colloq", "colloq_mask")
EXIT_KEY = {"keep": "keep", "maskfinal": "maskfinal", "mask": "mask",
            "colloq": "keep", "colloq_mask": "mask"}


def se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def loadj(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def load_rows(name: str) -> list[dict]:
    with open(DATA / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def paired(model_corr, rule_corr):
    d = [a - b for a, b in zip(model_corr, rule_corr)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / max(1, n - 1)
    return m, math.sqrt(var / max(1, n))


def main() -> int:
    bat = loadj(RESULTS / "battery.json")
    stats = loadj(DATA / "stats.json")
    runs = {}
    for a in ARMS:
        for s in SEEDS:
            for suf in ("", "_rand"):
                p = RESULTS / f"{a}_s{s}{suf}.json"
                if p.exists():
                    runs[f"{a}_s{s}{suf}"] = loadj(p)
    # PREREG rev.B 收敛档（steps=8000，敏感性，单列，不进主判据）
    ext = {}
    for p in sorted(RESULTS.glob("*_steps8000.json")):
        d = loadj(p)
        if "arm" in d:
            ext[d["name"]] = d

    out: dict = {"runs": sorted(runs), "ext_runs": sorted(ext),
                 "battery": {}, "table": [], "M": {}}

    # ---------------- M6 / M0：电池 ----------------
    print("== M6 · max_naive 电池（fit=train，逐出口）==")
    print(f"{'exit':12s} {'n':>5s} {'maj':>7s} {'R1s':>7s} {'R1f/R6':>7s} {'R2p':>7s} "
          f"{'R3c':>7s} {'R4l':>7s} {'R5f':>7s} {'max':>7s} {'best':>16s} {'1-2SE':>7s}")
    for e in EXITS:
        r = bat["exits"][e]["pool"]
        out["battery"][e] = {k: r[k] for k in (
            "n", "R0_majority", "R1_punct_simple", "R1_punct_full", "R2_particle",
            "R3_final_char", "R4_len_bucket", "R5_first_char", "R6_labeldef",
            "max_naive", "max_naive_rule", "se_at_max", "one_minus_2se",
            "M3_rule_lt_1m2se", "R_trig_majority_single", "trig_none_pct_test")}
        print(f"{e:12s} {r['n']:5d} {r['R0_majority']:7.4f} {r['R1_punct_simple']:7.4f} "
              f"{r['R1_punct_full']:7.4f} {r['R2_particle']:7.4f} "
              f"{r['R3_final_char']:7.4f} {r['R4_len_bucket']:7.4f} "
              f"{r['R5_first_char']:7.4f} {r['max_naive']:7.4f} "
              f"{r['max_naive_rule']:>16s} {r['one_minus_2se']:7.4f}")
    m0 = {
        "R1_simple_keep": bat["exits"]["keep"]["pool"]["R1_punct_simple"],
        "R1_simple_mask": bat["exits"]["mask"]["pool"]["R1_punct_simple"],
        "majority_mask": bat["exits"]["mask"]["pool"]["R0_majority"],
        "R1_full_keep": bat["exits"]["keep"]["pool"]["R1_punct_full"],
        "R1_full_mask": bat["exits"]["mask"]["pool"]["R1_punct_full"],
    }
    m0["assert_equiv"] = abs(m0["R1_simple_mask"] - m0["majority_mask"]) <= 1e-9
    m0["drop_simple"] = round(m0["R1_simple_keep"] - m0["R1_simple_mask"], 4)
    m0["drop_full"] = round(m0["R1_full_keep"] - m0["R1_full_mask"], 4)
    m0["pass"] = bool(m0["assert_equiv"] and m0["R1_simple_keep"] > m0["R1_simple_mask"])
    out["M"]["M0"] = m0
    print(f"\n[M0] R-punct(simple) keep={m0['R1_simple_keep']:.4f} → "
          f"mask={m0['R1_simple_mask']:.4f}（≡ majority {m0['majority_mask']:.4f}，"
          f"断言 {m0['assert_equiv']}）落差 {m0['drop_simple']:+.4f}；"
          f"R-punct(full) 落差 {m0['drop_full']:+.4f} ⇒ M0 {m0['pass']}")

    # ---------------- 主表 ----------------
    print("\n== 主表 · 臂 × 出口 × seed（mode acc ± SE，Δ=卡−max_naive）==")
    print(f"{'run':16s} {'exit':11s} {'n':>5s} {'acc':>7s} {'SE':>6s} {'maxN':>7s} "
          f"{'Δ':>8s} {'Δ−2SE':>8s} {'配对Δ±SE':>14s} {'−R2p':>7s} {'−R3c':>7s}")
    for name, run in runs.items():
        for e in EXITS:
            r = run["eval"][e]
            corr = r["mode_correct"]
            rule = bat["exits"][e]["pred"][bat["exits"][e]["pool"]["max_naive_rule"]]
            pm, pse = paired(corr, rule)
            row = {"run": name, "exit": e, "n": r["n"], "acc": r["mode_acc"],
                   "se": r["mode_se"], "max_naive": r["max_naive"],
                   "over": r["over_max_naive"], "over_2se": r["over_max_naive_2se"],
                   "paired_mean": round(pm, 4), "paired_se": round(pse, 5),
                   "over_R2": round(r["mode_acc"] - r["rule_R2_particle"], 4),
                   "over_R3": round(r["mode_acc"] - r["rule_R3_final_char"], 4),
                   "trig_acc": r["trig_acc"], "rec_text_rate": r["record_text_rate"]}
            out["table"].append(row)
            print(f"{name:16s} {e:11s} {r['n']:5d} {r['mode_acc']:7.4f} "
                  f"{r['mode_se']:6.4f} {r['max_naive']:7.4f} {r['over_max_naive']:+8.4f} "
                  f"{r['over_max_naive_2se']:+8.4f} {pm:+.4f}±{pse:.4f}   "
                  f"{row['over_R2']:+7.4f} {row['over_R3']:+7.4f}")

    # ---------------- 收敛档（rev.B 单列）----------------
    if ext:
        print("\n== 收敛档 steps=8000（PREREG rev.B 敏感性，单列，不进主判据）==")
        out["ext"] = {}
        for name, run in ext.items():
            rr = run["eval"]["mask"]
            rk = run["eval"]["keep"]
            out["ext"][name] = {
                "train_acc_last100": run["train_mode_acc_last100"],
                "loss_last": run["loss_last"],
                "mask_acc": rr["mode_acc"], "mask_over": rr["over_max_naive"],
                "mask_over_2se": rr["over_max_naive_2se"],
                "keep_acc": rk["mode_acc"], "keep_over": rk["over_max_naive"],
                "trig_acc": rr["trig_acc"], "rec_text_rate": rr["record_text_rate"]}
            print(f"  {name:22s} train_acc={run['train_mode_acc_last100']:.4f} "
                  f"mask={rr['mode_acc']:.4f}(Δ={rr['over_max_naive']:+.4f}) "
                  f"keep={rk['mode_acc']:.4f}(Δ={rk['over_max_naive']:+.4f}) "
                  f"trig={rr['trig_acc']:.4f} rec_text={rr['record_text_rate']:.4f}")

    # ---------------- M1：遮蔽出口主判 ----------------
    m1 = {}
    for name, run in runs.items():
        if name.endswith("_rand"):
            continue
        r = run["eval"]["mask"]
        corr = r["mode_correct"]
        rule = bat["exits"]["mask"]["pred"][bat["exits"]["mask"]["pool"]["max_naive_rule"]]
        pm, pse = paired(corr, rule)
        m1[name] = {"acc": r["mode_acc"], "se": r["mode_se"], "max_naive": r["max_naive"],
                    "over": r["over_max_naive"], "over_2se": r["over_max_naive_2se"],
                    "paired_mean": round(pm, 4), "paired_se": round(pse, 5),
                    "pass_2se": bool(r["over_max_naive_2se"] > 0),
                    "pass_paired": bool(pm - 2 * pse > 0)}
    # 主判据 = **MASK 臂**（设计主臂）两 seed；MIX 作同族支持；KEEP 臂的地板按定义是 1.0
    #（= M2 的对照，不进 M1 判据 —— PREREG §7 M1 只写"E_mask 出口"，M2 单列 KEEP）。
    _ov = [m1[f"MASK_s{s}"]["over"] for s in SEEDS if f"MASK_s{s}" in m1]
    m1["two_seed_same_sign_MASK"] = len(_ov) == 2 and (_ov[0] > 0) == (_ov[1] > 0)
    m1["two_seed_sign"] = "同为正" if all(x > 0 for x in _ov) else (
        "同为负" if all(x < 0 for x in _ov) else "异号")
    m1["verdict_M1"] = all(m1[f"MASK_s{s}"]["pass_2se"] for s in SEEDS
                           if f"MASK_s{s}" in m1) and m1["two_seed_same_sign_MASK"]
    m1["MIX_support"] = {k: v["pass_2se"] for k, v in m1.items() if k.startswith("MIX_")}
    m1["KEEP_excluded_reason"] = "KEEP 臂地板按定义 = 1.0（R6=label_full），归 M2 对照"
    out["M"]["M1"] = m1
    print(f"\n[M1] 遮蔽出口：全部臂×seed 的 卡−max_naive−2SE>0 = {m1['verdict_M1']}；"
          f"MASK 两 seed 同号 = {m1['two_seed_same_sign_MASK']}")

    # ---------------- 逐类拆解（MASK 臂 × mask 出口）----------------
    percls: dict = {}
    for s in SEEDS:
        k = f"MASK_s{s}"
        if k not in runs:
            continue
        r = runs[k]["eval"]["mask"]
        g, pm = r["gold_mode"], r["pred_mode"]
        rulep = bat["exits"]["mask"]["pred"]["R6_labeldef"]
        cls = {}
        for i in range(6):
            idx = [j for j, x in enumerate(g) if x == i]
            if not idx:
                continue
            cls[L_MODES(i)] = {
                "n": len(idx),
                "card_rec": round(sum(1 for j in idx if pm[j] == i) / len(idx), 4),
                "R6_rec": round(sum(1 for j in idx if rulep[j] == i) / len(idx), 4),
            }
        percls[k] = cls
        print(f"\n[逐类 recall · {k} @ mask] " + " ".join(
            f"{n}({v['n']}) card={v['card_rec']:.3f}/R6={v['R6_rec']:.3f}"
            for n, v in cls.items()))
    out["per_class"] = percls

    # ---------------- M2 ----------------
    keep_max = bat["exits"]["keep"]["pool"]["max_naive"]
    mask_max = bat["exits"]["mask"]["pool"]["max_naive"]
    m2 = {"max_naive_keep": keep_max, "max_naive_mask": mask_max,
          "punct_gives": round(keep_max - mask_max, 4),
          "card_keep": {f"MASK_s{s}": runs[f"MASK_s{s}"]["eval"]["keep"]["mode_acc"]
                        for s in SEEDS if f"MASK_s{s}" in runs},
          "card_mask": {f"MASK_s{s}": runs[f"MASK_s{s}"]["eval"]["mask"]["mode_acc"]
                        for s in SEEDS if f"MASK_s{s}" in runs}}
    # 卡自己的"遮蔽后崩"要看**用标点训出来的卡**（KEEP 臂）：keep 出口 → mask 出口
    m2["card_drop_KEEP_arm"] = {
        f"KEEP_s{s}": round(runs[f"KEEP_s{s}"]["eval"]["keep"]["mode_acc"]
                            - runs[f"KEEP_s{s}"]["eval"]["mask"]["mode_acc"], 4)
        for s in SEEDS if f"KEEP_s{s}" in runs}
    m2["card_keep_vs_rule_KEEP_arm"] = {
        f"KEEP_s{s}": round(runs[f"KEEP_s{s}"]["eval"]["keep"]["mode_acc"] - keep_max, 4)
        for s in SEEDS if f"KEEP_s{s}" in runs}
    out["M"]["M2"] = m2
    print(f"[M2] 标点给多少：max_naive keep {keep_max:.4f} → mask {mask_max:.4f} "
          f"（地板 {m2['punct_gives']:+.4f}）；KEEP 臂卡 keep→mask 掉 "
          f"{m2['card_drop_KEEP_arm']}；KEEP 臂卡−地板(keep)={m2['card_keep_vs_rule_KEEP_arm']}")

    # ---------------- M3 ----------------
    r = bat["exits"]["mask"]["pool"]
    m3 = {"max_naive_mask": r["max_naive"], "one_minus_2se": r["one_minus_2se"],
          "pass": bool(r["max_naive"] < r["one_minus_2se"]),
          "source_independent": {
              "陈述/是非/感叹(顶部分支)": "标点定义 ⇒ 与遮蔽输入上的特征规则不同源",
              "特指疑问(细分)": "QSET 词表 ⇒ 与 R-particle **同源**",
              "祈使(细分)": "IMP 词表 ⇒ 不同源",
              "反问": "RHO 词表 ⇒ 与 R-particle **同源**"},
          "caveat": "更正 C：整体 rule<1−2SE 只覆盖标点分界；词表细分处来源不独立"}
    out["M"]["M3"] = m3
    print(f"[M3] max_naive(mask)={r['max_naive']:.4f} < 1−2SE={r['one_minus_2se']:.4f} "
          f"⇒ {m3['pass']}（来源独立性逐类见 summary）")

    # ---------------- M4 ----------------
    m4 = {}
    for s in SEEDS:
        k = f"MASK_s{s}_rand"
        if k in runs:
            rr = runs[k]["eval"]["mask"]
            maj = bat["exits"]["mask"]["pool"]["R0_majority"]
            m4[k] = {"acc": rr["mode_acc"], "se": rr["mode_se"], "majority": maj,
                     "pass": bool(rr["mode_acc"] <= maj + 2 * rr["mode_se"])}
    m4["verdict"] = all(v["pass"] for v in m4.values()) and bool(m4)
    out["M"]["M4"] = m4
    print(f"[M4] 随机标签对照：{ {k: v['acc'] for k, v in m4.items() if k != 'verdict'} } "
          f"vs majority={bat['exits']['mask']['pool']['R0_majority']:.4f} ⇒ "
          f"{m4['verdict']}")

    # ---------------- M5 ----------------
    dpath = RESULTS / "old_cards_delta.json"
    if dpath.exists():
        d = loadj(dpath)
        m5 = {n: {"delta": d[n]["delta"], "band": d[n]["noise_band"],
                  "pass": bool(d[n]["delta"] >= -d[n]["noise_band"]),
                  "identical": d[n]["metric_dict_identical"]}
              for n in ("pronoun", "sentiment", "relation", "person")}
        m5["hash_identical"] = d["files_sha256_identical"]
        m5["verdict"] = all(v["pass"] for v in m5.values() if isinstance(v, dict)) \
            and d["files_sha256_identical"]
        out["M"]["M5"] = m5
        print("[M5] " + " ".join(f"{n} Δ={m5[n]['delta']:+.6f}/带{m5[n]['band']}"
                                 for n in m5 if isinstance(m5[n], dict))
              + f" ⇒ {m5['verdict']}")
    else:
        out["M"]["M5"] = {"verdict": None, "note": "old_cards_delta.json 缺"}
        print("[M5] 缺 old_cards_delta.json")

    # ---------------- 判定 ----------------
    v1 = bool(m1["verdict_M1"] and m1["two_seed_same_sign_MASK"])
    v3 = bool(m3["pass"])
    v0 = bool(m0["pass"])
    v4 = bool(m4.get("verdict"))
    v5 = bool(out["M"].get("M5", {}).get("verdict"))
    big_punct = (keep_max - mask_max) >= 0.30
    if v0 and v1 and v3:
        verdict = "输入侧句式可读（且有构念效度）"
    elif v0 and not v1 and big_punct:
        verdict = "只靠标点（遮蔽后崩）"
    else:
        verdict = "证据不足"
    out["verdict"] = verdict
    out["gate"] = {"M0": v0, "M1": v1, "M3": v3, "M4": v4, "M5": v5,
                   "punct_drop_ge_0.30": big_punct}
    print(f"\n== 判定：{verdict}  （M0={v0} M1={v1} M3={v3} M4={v4} M5={v5} "
          f"标点落差≥.30={big_punct}）==")

    # ---------------- 真实样例 ----------------
    rows = load_rows("test")
    run = runs.get("MASK_s42") or next(iter(runs.values()))
    samples = []
    picked: dict[int, list[int]] = {k: [] for k in range(6)}
    for i, r in enumerate(rows):                      # 每类取前 1 条 + 前 1 条预测错的
        if len(picked[r["mode"]]) >= 2:
            continue
        wrong = run["eval"]["mask"]["pred_mode"][i] != r["mode"]
        tag = 1 if wrong else 0
        if tag not in picked[r["mode"]]:
            picked[r["mode"]].append(i)
        if all(len(v) >= 2 for v in picked.values()):
            break
    # 先保证六类各有一条（含反问），再补预测错的，总数 ≤10
    head = [(g, picked[g][0]) for g in sorted(picked) if picked[g]]
    tail = [(g, i) for g in sorted(picked) for i in picked[g][1:]]
    sel = (head + tail)[:10]
    for g, i in sel:
        r = rows[i]
        mi = run["eval"]["mask"]["pred_mode"][i]
        sp = r["trig"]["mask"]
        rec = MR.make_record(mi, sp, r["exit"]["mask"], input_text=r["exit"]["mask"])
        samples.append({
            "原文": r["text"], "输入(遮蔽)": r["exit"]["mask"],
            "gold": r["mode_name"], "pred": L.MODES[mi],
            "对错": "对" if mi == r["mode"] else "错",
            "触发词span": list(sp) if sp else None,
            "触发词": r["exit"]["mask"][sp[0]:sp[1]] if sp else "∅",
            "kind": rec["kind"], "instruction": rec["instruction"],
            "依据": f"遮蔽输入含触发词「{r['exit']['mask'][sp[0]:sp[1]]}」（封闭表）；"
                    f"构造性规则 R6 在此样本上判 {L.MODES[L.MODE_ID[L.label_defn_on(r['exit']['mask'])]]}"
                    if sp else f"无封闭表触发词；R6 判 "
                               f"{L.MODES[L.MODE_ID[L.label_defn_on(r['exit']['mask'])]]}",
        })
    out["samples"] = samples
    print("\n== 真实样例（MASK 臂 s42，遮蔽出口）==")
    for s_ in samples:
        print(f"  [{s_['对错']}] 原文={s_['原文']!r}\n    遮蔽输入={s_['输入(遮蔽)']!r} "
              f"gold={s_['gold']} pred={s_['pred']} 触发词={s_['触发词']!r} "
              f"span={s_['触发词span']} kind={s_['kind']}\n    依据={s_['依据']}")

    out["stats_head"] = {"rows": stats["rows"], "sizes": stats["sizes"],
                         "dist_test": stats["dist_test"]}
    (RESULTS / "summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] results/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
