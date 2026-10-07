#!/usr/bin/env python3
"""bag_modules 分析：主表 + 配对 SE + B1–B5 逐条 + 机制证据（PREREG §4）。"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import model as M  # noqa: E402
import train as T  # noqa: E402

RESULTS = HERE / "results"
SPLITS = ("test", "adv1", "adv2")
ARMS = ("A", "U", "P", "UP")
SEEDS = (42, 43)
MAX_NAIVE = T.MAX_NAIVE
MAJORITY = T.MAJORITY


def load(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"缺结果：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def acc(res: dict, split: str, k: str) -> float:
    return res["meta"]["eval"][split][k]["acc"]


def se(res: dict, split: str, k: str) -> float:
    return res["meta"]["eval"][split][k]["se"]


def paired(res_a: dict, res_b: dict, split: str, key: str = "skel") -> dict:
    """Δ = b − a 的配对 SE（逐行 0/1 差）。"""
    a = res_a["correct"][split][key]
    b = res_b["correct"][split][key]
    assert len(a) == len(b)
    d = [x - y for x, y in zip(b, a)]
    n = len(d)
    mu = sum(d) / n
    var = sum((x - mu) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    s = math.sqrt(var / n)
    return {"delta": round(mu, 6), "se": round(s, 6),
            "t": round(mu / s, 3) if s > 0 else None, "n": n}


def load_all() -> dict:
    out = {}
    for a in ARMS:
        for s in SEEDS:
            out[f"{a}_s{s}"] = load(f"{a}_s{s}")
    for s in SEEDS:
        out[f"U_s{s}_rand"] = load(f"U_s{s}_rand")
        for m in ("type", "role", "cls"):
            out[f"U_s{s}_m{m}"] = load(f"U_s{s}_m{m}")
        p = RESULTS / f"U_s{s}_mtype-cls.json"          # 补跑：B5 判定臂（M1+M3）
        if p.exists():
            out[f"Umr_s{s}"] = json.loads(p.read_text(encoding="utf-8"))
    return out


def _keys(allr: dict) -> list[str]:
    ks = [f"{a}_s{s}" for a in ARMS for s in SEEDS] + \
        [f"U_s{s}_rand" for s in SEEDS] + \
        [f"U_s{s}_m{m}" for s in SEEDS for m in ("type", "role", "cls")] + \
        [f"Umr_s{s}" for s in SEEDS]
    return [k for k in ks if k in allr]


# ---------------------------------------------------------------------------
# 机制：标签序列 → 骨架 的**免费规则**（train 拟合多数，eval 查表）
# ---------------------------------------------------------------------------
def label_keys(rows: list[dict], split: str, mods=("type", "role", "cls")) -> list:
    obj = json.loads((HERE / "data" / f"labels_{split}.json").read_text(encoding="utf-8"))
    keys = []
    for r, lab in zip(rows, obj["labels"]):
        order = sorted(range(r["n_slots"]), key=lambda k: r["bag_span"][k][0])
        ks = []
        for k in order:
            e = []
            if "type" in mods:
                e.append(lab[k]["t"])
            if "role" in mods:
                e.append(lab[k]["r"])
            if "cls" in mods:
                e.append(lab[k]["c"])
            ks.append(tuple(e))
        keys.append(tuple(ks))
    return keys


def rule_baseline() -> dict:
    rows_tr = T.get_rows("train")
    gold_tr = [r["skel_id"] for r in rows_tr]
    from collections import Counter, defaultdict
    glob = Counter(gold_tr).most_common(1)[0][0]
    out = {}
    for mods in (("type", "role", "cls"), ("type",), ("role",), ("cls",)):
        kk = label_keys(rows_tr, "train", mods)
        tab: dict = defaultdict(Counter)
        for k, y in zip(kk, gold_tr):
            tab[k][y] += 1
        maj = {k: c.most_common(1)[0][0] for k, c in tab.items()}
        res = {"mods": list(mods), "train_cover": round(len(tab) / len(kk), 4),
               "seen_rate": {}}
        for split in SPLITS:
            rows = T.get_rows(split)
            gold = [r["skel_id"] for r in rows]
            ks = label_keys(rows, split, mods)
            ok = sum(1 for k, y in zip(ks, gold)
                     if maj.get(k, glob) == y)
            seen = sum(1 for k in ks if k in maj)
            res["seen_rate"][split] = round(seen / len(ks), 4)
            res[split] = {"acc": round(ok / len(ks), 4),
                          "over_max_naive": round(ok / len(ks) - MAX_NAIVE[split], 4),
                          "majority": MAJORITY[split]}
        out[",".join(mods)] = res

    # 机制⑤：**袋项个数 n_slots → 骨架多数**（B3 失效后追查承载者；U-rand 保留了 mask）
    from collections import Counter, defaultdict
    tab: dict = defaultdict(Counter)
    for r, y in zip(rows_tr, gold_tr):
        tab[r["n_slots"]][y] += 1
    maj = {k: c.most_common(1)[0][0] for k, c in tab.items()}
    res = {"mods": ["n_slots"], "seen_rate": {}}
    for split in SPLITS:
        rows = T.get_rows(split)
        gold = [r["skel_id"] for r in rows]
        ok = sum(1 for r, y in zip(rows, gold) if maj.get(r["n_slots"], glob) == y)
        res[split] = {"acc": round(ok / len(rows), 4),
                      "over_max_naive": round(ok / len(rows) - MAX_NAIVE[split], 4),
                      "majority": MAJORITY[split]}
        res["seen_rate"][split] = 1.0
    out["n_slots"] = res
    return out


# ---------------------------------------------------------------------------
# 机制：对已训 U 做**评测期标签打乱**（模型对标签通道的依赖度）
# ---------------------------------------------------------------------------
def shuffle_probe(seed: int, device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = M.Spec()
    m = M.BagModModel("U", seed, spec).to(device)
    sd = torch.load(HERE / "weights" / f"U_s{seed}.pt", map_location="cpu",
                    weights_only=True)
    miss = [k for k in m.state_dict() if k not in sd]
    assert all(k.startswith("encoder.") for k in miss), miss[:5]   # 只许缺冻结核
    m.load_state_dict(sd, strict=False)
    m.eval()
    out = {}
    g = torch.Generator().manual_seed(seed * 1000 + 99)
    for split in SPLITS:
        rows = T.get_rows(split)
        blob = T.get_blob(rows, spec, device, split)
        labs = T.label_tensors(rows, split)
        perm = torch.randperm(len(rows), generator=g)
        labs_sh = {k: (v[perm] if v.shape[0] == len(rows) else v)
                   for k, v in labs.items()}
        r_true, _, _ = T.evaluate(m, blob, rows, labs, None, device)
        r_shuf, _, _ = T.evaluate(m, blob, rows, labs_sh, None, device)
        out[split] = {"true_skel": r_true["skel"]["acc"],
                      "shuffled_skel": r_shuf["skel"]["acc"],
                      "drop": round(r_true["skel"]["acc"] - r_shuf["skel"]["acc"], 4),
                      "shuffled_slot": r_shuf["slot"]["acc"]}
    return out


def main() -> None:
    allr = load_all()
    rep: dict = {"prereg": "PREREG.md"}
    print("=" * 78)
    print("主表（骨架/槽位/联合，acc±SE；max_naive = 免费规则）")
    for split in SPLITS:
        print(f"\n--- {split}（n={allr['A_s42']['meta']['eval'][split]['skel']['n']}，"
              f"max_naive={MAX_NAIVE[split]}，多数类={MAJORITY[split]}）")
        print(f"{'臂':10s} {'seed':5s} {'骨架':>16s} {'槽位':>16s} {'联合':>16s} "
              f"{'骨架−max_naive':>14s}")
        for key in _keys(allr):
            r = allr[key]
            m = r["meta"]["eval"][split]
            print(f"{key:16s} {m['skel']['acc']:.4f}±{m['skel']['se']:.4f}  "
                  f"{m['slot']['acc']:.4f}±{m['slot']['se']:.4f}  "
                  f"{m['joint']['acc']:.4f}±{m['joint']['se']:.4f}  "
                  f"{m['skel']['acc'] - MAX_NAIVE[split]:+.4f}")

    # ---- B1 / B2（配对）----
    b1, b2 = {}, {}
    for a in ("U", "P", "UP", "Umr"):
        b1[a], b2[a] = {}, {}
        for s in SEEDS:
            if f"{a}_s{s}" not in allr:
                continue
            b1[a][f"s{s}"] = paired(allr[f"A_s{s}"], allr[f"{a}_s{s}"], "adv2", "skel")
            b2[a][f"s{s}"] = paired(allr[f"A_s{s}"], allr[f"{a}_s{s}"], "test", "skel")
    rep["B1_adv2_paired"], rep["B2_test_paired"] = b1, b2
    print("\n" + "=" * 78)
    print("B1 主判据（adv2 骨架 Δ=臂−A，配对 SE） / B2（test 骨架 Δ）")
    for a in ("U", "P", "UP", "Umr"):
        for s in SEEDS:
            if f"s{s}" not in b1.get(a, {}):
                continue
            x, y = b1[a][f"s{s}"], b2[a][f"s{s}"]
            print(f"  {a:2s} s{s}: adv2 Δ={x['delta']:+.4f} SE={x['se']:.4f} "
                  f"t={x['t']}   | test Δ={y['delta']:+.4f} SE={y['se']:.4f} t={y['t']}")

    # ---- B3 门禁 ----
    b3 = {}
    for s in SEEDS:
        d_adv = paired(allr[f"A_s{s}"], allr[f"U_s{s}_rand"], "adv2", "skel")
        d_te = paired(allr[f"A_s{s}"], allr[f"U_s{s}_rand"], "test", "skel")
        aux = allr[f"U_s{s}_rand"]["meta"]["eval"]
        se_te = math.sqrt(0.25 / aux["test"]["skel"]["n"])
        se_adv = math.sqrt(0.25 / aux["adv2"]["skel"]["n"])
        b3[f"s{s}"] = {
            "rand_adv2_delta_vs_A": d_adv, "rand_test_delta_vs_A": d_te,
            "rand_aux_only_test": aux["test"]["diag"]["aux_only"],
            "rand_aux_only_adv2": aux["adv2"]["diag"]["aux_only"],
            "thr_test": round(MAJORITY["test"] + 2 * se_te, 4),
            "thr_adv2": round(max(MAJORITY["adv2"], MAX_NAIVE["adv2"]) + 2 * se_adv, 4),
            "b3a_pass": abs(d_adv["delta"]) <= 2 * d_adv["se"] and abs(d_te["delta"]) <= 2 * d_te["se"],
            "b3b_pass": aux["test"]["diag"]["aux_only"] <= MAJORITY["test"] + 2 * se_te
                        and aux["adv2"]["diag"]["aux_only"] <= max(MAJORITY["adv2"], MAX_NAIVE["adv2"]) + 2 * se_adv,
        }
    rep["B3"] = b3
    print("\nB3 门禁（随机标签）")
    for k, v in b3.items():
        print(f"  {k}: Δ_adv2={v['rand_adv2_delta_vs_A']['delta']:+.4f}"
              f"(t={v['rand_adv2_delta_vs_A']['t']}) "
              f"Δ_test={v['rand_test_delta_vs_A']['delta']:+.4f}"
              f"(t={v['rand_test_delta_vs_A']['t']}) | "
              f"aux_only test={v['rand_aux_only_test']}(≤{v['thr_test']}) "
              f"adv2={v['rand_aux_only_adv2']}(≤{v['thr_adv2']}) "
              f"=> B3a {v['b3a_pass']} B3b {v['b3b_pass']}")

    # ---- B5 模块可靠性 ----
    rel = json.loads((HERE / "data" / "reliability.json").read_text(encoding="utf-8"))
    b5 = {}
    for split in ("train", "test", "adv2"):
        t, r, c = rel[split]["M1_type"], rel[split]["M2_role"], rel[split]["M3_cls"]
        b5[split] = {
            "M1_acc": t["acc_vs_lexicon_gold"], "M1_gold_cov": t["lexicon_gold_coverage"],
            "M1_cover": t["cover_r1"],
            "M2_acc": r["acc_vs_struct_gold"], "M2_cover": r["cover_r1_non_other"],
            "M2_gold_cov": r["struct_gold_coverage"],
            "M3_agree": c["agreement_with_adv2_rule"], "M3_cover": c["cover_non_O"],
        }
    rep["B5"] = b5
    print("\nB5 模块可靠性（门槛：准确率≥0.70 ∧ 覆盖率≥0.10）")
    for split, v in b5.items():
        print(f"  {split}: M1 acc={v['M1_acc']} cov={v['M1_cover']} | "
              f"M2 acc={v['M2_acc']} cov={v['M2_cover']} | "
              f"M3 agree={v['M3_agree']} cov={v['M3_cover']}")

    # ---- 机制 ----
    mech = {"label_rule_baseline": rule_baseline(), "aux_only": {}, "head_only": {}}
    for a in ("U", "UP"):
        mech["aux_only"][a] = {f"s{s}": {sp: allr[f"{a}_s{s}"]["meta"]["eval"][sp]["diag"]["aux_only"]
                                          for sp in SPLITS} for s in SEEDS}
        mech["head_only"][a] = {f"s{s}": {sp: allr[f"{a}_s{s}"]["meta"]["eval"][sp]["diag"]["head_only"]
                                           for sp in SPLITS} for s in SEEDS}
    print("\n机制① 标签-only 通道（aux_only，骨架 acc）")
    print(json.dumps(mech["aux_only"], ensure_ascii=False))
    print("机制② 标签序列 → 骨架 的免费规则（train 拟合多数查表）")
    print(json.dumps(mech["label_rule_baseline"], ensure_ascii=False, indent=1))
    try:
        mech["shuffle_probe"] = {f"s{s}": shuffle_probe(s) for s in SEEDS}
        print("机制④ 评测期标签打乱（U）")
        print(json.dumps(mech["shuffle_probe"], ensure_ascii=False, indent=1))
    except Exception as e:  # 诊断失败不阻断主分析
        mech["shuffle_probe_error"] = repr(e)
        print("[probe] 失败", repr(e), flush=True)
    rep["mechanism"] = mech

    (HERE / "results" / "analyze.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[done] → results/analyze.json")


if __name__ == "__main__":
    main()
