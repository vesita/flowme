#!/usr/bin/env python3
"""struct_supervision 分析：主表 + S1/S2/S3 + 门禁 + max_naive + 判定（PREREG §4 原文执行）。

用法：uv run python experiments/struct_supervision/analyze.py
输出：stdout markdown 表 + results/analyze.json
"""
from __future__ import annotations

import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
DATA = HERE / "data"
TCH_STATS = HERE.parent / "two_channel_head" / "data" / "stats.json"

SPLITS = ("test", "adv1", "adv2", "adv")
SEEDS = (42, 43)
KEY = {"A": "decoded", "B": "head", "C": "head"}     # 结构指标取自哪个出口


def load(arm: str, seed: int, rand: bool = False) -> dict | None:
    p = RESULTS / f"{arm}_s{seed}{'_rand' if rand else ''}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def pool_split(res: dict, split: str) -> dict:
    """合并 adv1+adv2 的逐行 0/1（adv = 两子集拼接）。"""
    key = KEY[res["meta"]["arm"]]
    if split != "adv":
        return res["correct"][f"{split}_{key}"]
    a = res["correct"]["adv1_head" if key == "head" else "adv1_decoded"]
    b = res["correct"]["adv2_head" if key == "head" else "adv2_decoded"]
    return {k: a[k] + b[k] for k in a}


def acc_se(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    mu = sum(vals) / n
    if n < 2:
        return mu, math.sqrt(0.25 / n)
    var = sum((v - mu) ** 2 for v in vals) / (n - 1)
    return mu, math.sqrt(var / n)


def cell(res: dict, split: str, metric: str) -> dict:
    c = pool_split(res, split)
    vals = c[metric] if metric != "slot" else c["slot_frac"]
    mu, se = acc_se(vals)
    return {"acc": round(mu, 6), "se": round(se, 6), "n": len(vals)}


def paired(res_x: dict, res_y: dict, split: str, metric: str) -> dict:
    cx, cy = pool_split(res_x, split), pool_split(res_y, split)
    vx = cx[metric] if metric != "slot" else cx["slot_frac"]
    vy = cy[metric] if metric != "slot" else cy["slot_frac"]
    assert len(vx) == len(vy), (len(vx), len(vy))
    d = [a - b for a, b in zip(vx, vy)]
    n = len(d)
    mu = sum(d) / n
    se = math.sqrt(max(1e-12, sum((x - mu) ** 2 for x in d)) / (n - 1)) / math.sqrt(n)
    return {"delta": round(mu, 6), "se": round(se, 6),
            "t": round(mu / se, 3) if se > 0 else None, "n": n}


def tstat(acc: float, base: float, se: float) -> float:
    """(acc−基线)/SE；se=0（逐行全同）时用带符号哨兵，避免除零。"""
    if se > 0:
        return (acc - base) / se
    return 1e6 if acc > base else (-1e6 if acc < base else 0.0)


def fmt(c: dict) -> str:
    return f"{c['acc']:.4f}±{c['se']:.4f}"


def main() -> None:
    stats_tch = json.loads(TCH_STATS.read_text(encoding="utf-8"))
    stats_adv = json.loads((DATA / "stats_adv.json").read_text(encoding="utf-8"))
    MAXNAIVE = {"test": stats_tch["max_naive_train"],
                "test_eval": stats_tch["naive_skeleton_test"]["max_naive"],
                "adv1": stats_adv["naive"]["adv1_skel"]["max_naive"],
                "adv2": stats_adv["naive"]["adv2_skel"]["max_naive"]}
    # adv 合计的 max_naive：逐规则按 n 加权再取最大
    w = {k: stats_adv[k]["n"] for k in ("adv1", "adv2")}
    tot = sum(w.values())
    rules = [k for k in stats_adv["naive"]["adv1_skel"] if k != "max_naive"]
    pooled = {r: (stats_adv["naive"]["adv1_skel"][r] * w["adv1"]
                  + stats_adv["naive"]["adv2_skel"][r] * w["adv2"]) / tot
              for r in rules}
    MAXNAIVE["adv"] = round(max(pooled.values()), 4)
    BLIND_SKEL = 0.025
    MAJORITY = stats_tch["sets"]["train"]["majority_baseline"]
    BLIND_SLOT = {"test": stats_tch["naive_assign"]["blind_uniform"],
                  "adv1": stats_adv["naive"]["adv1_assign"]["blind_uniform"],
                  "adv2": stats_adv["naive"]["adv2_assign"]["blind_uniform"]}
    BLIND_SLOT["adv"] = round((BLIND_SLOT["adv1"] * w["adv1"]
                               + BLIND_SLOT["adv2"] * w["adv2"]) / tot, 4)

    runs: dict[str, dict] = {}
    for arm in "ABC":
        for s in SEEDS:
            r = load(arm, s)
            if r:
                runs[f"{arm}_s{s}"] = r
    for arm in "AB":
        for s in SEEDS:
            r = load(arm, s, rand=True)
            if r:
                runs[f"{arm}_s{s}_rand"] = r
    missing = [k for k in ("A_s42", "A_s43", "B_s42", "B_s43", "C_s42", "C_s43")
               if k not in runs]
    print(f"[runs] {sorted(runs)}；缺 {missing}")

    # ---- 主表 ----
    print("\n### 主表（acc±SE；骨架/槽位/联合分列）\n")
    print("| 臂 | seed | split | 骨架 | 槽位 | 联合 | 槽位盲猜 | 槽位余量/SE | 骨架−max_naive |")
    print("|---|---|---|---|---|---|---|---|---|")
    table = {}
    for name, res in sorted(runs.items()):
        meta = res["meta"]
        table[name] = {}
        for split in SPLITS:
            try:
                sk, sl, jn = (cell(res, split, "skel"), cell(res, split, "slot"),
                              cell(res, split, "joint"))
            except KeyError:
                continue
            blind = BLIND_SLOT[split]
            mo = tstat(sl["acc"], blind, sl["se"])
            mn = MAXNAIVE["test"] if split == "test" else MAXNAIVE[split]
            table[name][split] = {"skel": sk, "slot": sl, "joint": jn,
                                  "slot_blind": blind,
                                  "slot_margin_over_se": round(mo, 3),
                                  "max_naive": mn,
                                  "skel_minus_maxnaive": round(sk["acc"] - mn, 4)}
            if split == "test" or split == "adv":
                print(f"| {meta['arm']}{'-rand' if meta['randlabel'] else ''} | "
                      f"{meta['seed']} | {split} | {fmt(sk)} | {fmt(sl)} | {fmt(jn)} | "
                      f"{blind:.4f} | {tstat(sl['acc'], blind, sl['se']):+.2f} | "
                      f"{sk['acc']-mn:+.4f} |")

    # A/C 的整句诊断
    print("\n### 整句通道诊断（自由解码 / TF 字准确率）\n")
    print("| 跑 | split | TF 字准确率 | 逐字复现 | 可解析 |")
    print("|---|---|---|---|---|")
    diag = {}
    for name, res in sorted(runs.items()):
        meta = res["meta"]
        if meta["arm"] not in ("A", "C"):
            continue
        diag[name] = {}
        for split in ("test", "adv1", "adv2"):
            m = res["meta"]["eval"].get(f"{split}_decoded")
            if not m:
                continue
            d = m["sent_diag"]
            diag[name][split] = d
            if split == "test":
                print(f"| {meta['arm']}{'-rand' if meta['randlabel'] else ''} "
                      f"s{meta['seed']} | {split} | {d['tf_char_acc']:.4f} | "
                      f"{d['sent_exact']:.4f} | {d['parseable']:.4f} |")

    # ---- S1 ----
    print("\n### S1：槽位 / 骨架余量（两 seed，门槛 > 2SE）\n")
    s1 = {}
    for arm in ("B", "C"):
        for split in SPLITS:
            row = {}
            for s in SEEDS:
                r = runs.get(f"{arm}_s{s}")
                if not r:
                    continue
                sl = cell(r, split, "slot")
                sk = cell(r, split, "skel")
                blind = BLIND_SLOT[split]
                mn = MAXNAIVE["test"] if split == "test" else MAXNAIVE[split]
                row[f"s{s}"] = {
                    "slot": sl, "slot_blind": blind,
                    "slot_t": round(tstat(sl["acc"], blind, sl["se"]), 2),
                    "skel": sk, "max_naive": mn,
                    "skel_t": round(tstat(sk["acc"], mn, sk["se"]), 2)}
            s1[f"{arm}/{split}"] = row
            if row:
                print(f"{arm} {split:5s} 槽位 " + " ".join(
                    f"s{s}: {row[f's{s}']['slot']['acc']:.4f}(盲猜{row[f's{s}']['slot_blind']:.4f}, "
                    f"t={row[f's{s}']['slot_t']:+.2f})" for s in SEEDS if f"s{s}" in row)
                    + " | 骨架 " + " ".join(
                        f"s{s}: {row[f's{s}']['skel']['acc']:.4f}(max_naive"
                        f"{row[f's{s}']['max_naive']:.4f}, t={row[f's{s}']['skel_t']:+.2f})"
                        for s in SEEDS if f"s{s}" in row))
    s1_pass = all(v[f"s{s}"]["slot_t"] > 2 for k, v in s1.items()
                  for s in SEEDS if f"s{s}" in v)

    # ---- S2 / S3 配对差 ----
    print("\n### S2（test 骨架，主指标）/ S3（adv 槽位）配对差（Δ = B−A / C−A，配对 SE）\n")
    s2: dict = {}
    s3: dict = {}
    for metric, split, tag in (("skel", "test", "S2"), ("slot", "test", "S2-slot"),
                               ("joint", "test", "S2-joint"),
                               ("slot", "adv", "S3"), ("skel", "adv2", "S3-skel2"),
                               ("slot", "adv1", "S3-adv1"), ("slot", "adv2", "S3-adv2")):
        for other in ("B", "C"):
            key = f"{other}−A/{tag}/{split}"
            per = {}
            for s in SEEDS:
                ra, ro = runs.get(f"A_s{s}"), runs.get(f"{other}_s{s}")
                if not (ra and ro):
                    continue
                per[f"s{s}"] = paired(ro, ra, split, metric)
            if not per:
                continue
            (s2 if tag.startswith("S2") else s3)[key] = per
            print(f"{key}: " + "  ".join(
                f"s{s} Δ={per[f's{s}']['delta']:+.4f} SE={per[f's{s}']['se']:.4f} "
                f"Δ/SE={per[f's{s}']['t']:+.2f}" for s in SEEDS if f"s{s}" in per))

    def pass_pair(per: dict) -> str:
        """两 seed 同号且 |Δ|>2SE → '过'；两 seed 均 ≤2SE → '无差异'；否则 '不过/不同号'。"""
        if len(per) < 2:
            return "缺 seed"
        ts = [per[f"s{s}"]["t"] for s in SEEDS if f"s{s}" in per]
        ds = [per[f"s{s}"]["delta"] for s in SEEDS if f"s{s}" in per]
        if all(abs(t) <= 2 for t in ts):
            return "无差异"
        if all(abs(t) > 2 for t in ts) and all(d > 0 for d in ds):
            return "过"
        if all(abs(t) > 2 for t in ts) and all(d < 0 for d in ds):
            return "A 更好"
        return "不同号/未过"

    verdict_bits = {"S2_skel": pass_pair(s2.get("B−A/S2/test", {})),
                    "S2_slot": pass_pair(s2.get("B−A/S2-slot/test", {})),
                    "S2_joint": pass_pair(s2.get("B−A/S2-joint/test", {})),
                    "S2C_skel": pass_pair(s2.get("C−A/S2/test", {})),
                    "S3_slot": pass_pair(s3.get("B−A/S3/adv", {})),
                    "S3C_slot": pass_pair(s3.get("C−A/S3/adv", {})),
                    "S3_skel_adv2": pass_pair(s3.get("B−A/S3-skel2/adv2", {}))}

    # S3 附加：B/C 的 adv 槽位是否 > 盲猜 + 2SE（否则「背句式」）
    transfer = {}
    for arm in ("B", "C"):
        for s in SEEDS:
            r = runs.get(f"{arm}_s{s}")
            if not r:
                continue
            for split in ("adv1", "adv2"):
                sl = cell(r, split, "slot")
                t = tstat(sl["acc"], BLIND_SLOT[split], sl["se"])
                transfer[f"{arm}_s{s}/{split}"] = {"acc": sl["acc"],
                                                   "blind": BLIND_SLOT[split],
                                                   "t": round(t, 2)}
    print("\nS3 迁移（B/C 对抗集槽位 vs 盲猜）：" +
          json.dumps(transfer, ensure_ascii=False))
    transfer_pass = all(v["t"] > 2 for v in transfer.values())

    # ---- 门禁 ----
    print("\n### 门禁：随机标签对照\n")
    gate: dict = {}
    for arm in ("A", "B"):
        for s in SEEDS:
            r = runs.get(f"{arm}_s{s}_rand")
            if not r:
                continue
            split = "test"
            mkey = f"{split}_{'decoded' if arm == 'A' else 'head'}"
            m = r["meta"]["eval"][mkey]
            thr_skel = (MAJORITY if arm == "B" else BLIND_SKEL) + 2 * m["skel"]["se"]
            thr_slot = BLIND_SLOT["test"] + 2 * m["slot"]["se"]
            gate[f"{arm}_s{s}_rand"] = {
                "skel": m["skel"]["acc"], "skel_thr": round(thr_skel, 4),
                "slot": m["slot"]["acc"], "slot_thr": round(thr_slot, 4),
                "pass": bool(m["skel"]["acc"] <= thr_skel
                             and m["slot"]["acc"] <= thr_slot)}
            print(f"{arm}-rand s{s}: 骨架 {m['skel']['acc']:.4f} ≤ {thr_skel:.4f}？"
                  f"{m['skel']['acc'] <= thr_skel} | 槽位 {m['slot']['acc']:.4f} "
                  f"≤ {thr_slot:.4f}？{m['slot']['acc'] <= thr_slot}")
    gate_pass = bool(gate) and all(v["pass"] for v in gate.values())

    # ---- 判定 ----
    s2_main = verdict_bits["S2_skel"]
    if s2_main == "过" and s1_pass and verdict_bits["S3_slot"] == "过" and gate_pass \
            and transfer_pass:
        verdict = "结构化监督更好"
    elif s2_main == "无差异" and gate_pass and s1_pass:
        verdict = "与整句 loss 无差异"
    else:
        verdict = "证据不足"
    out = {"runs": sorted(runs), "missing": missing, "table": table, "diag": diag,
           "s1": s1, "s1_pass": s1_pass, "s2": s2, "s3": s3,
           "verdict_bits": verdict_bits, "transfer": transfer,
           "transfer_pass": transfer_pass, "gate": gate, "gate_pass": gate_pass,
           "max_naive": MAXNAIVE, "blind_slot": BLIND_SLOT,
           "majority": MAJORITY, "verdict": verdict}
    (RESULTS / "analyze.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print("\n### 判定要素\n")
    print(json.dumps({"S1_pass": s1_pass, **verdict_bits,
                      "transfer_pass": transfer_pass, "gate_pass": gate_pass,
                      "max_naive": MAXNAIVE}, ensure_ascii=False, indent=2))
    print(f"\n>>> 判定：{verdict}")


if __name__ == "__main__":
    main()
