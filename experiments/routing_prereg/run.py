#!/usr/bin/env python3
"""P24 主评测：四臂（F / C-mode=① / C-qc=④ / ①∩④）× 三出口 × 2 seed，唯一变量 = 解码掩码。

判据 V0–V5 见 `PREREG.md`（跑前写死）。**只读 import** P23 `run_eval.py`（①④的定义实现）
与 P8 `common.py`/`modes.py`；缓存写本目录（不碰既有实验）。

用法：uv run python experiments/routing_prereg/run.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P23 = ROOT / "experiments" / "orthogonal_tighten"
sys.path.insert(0, str(P23))

import torch  # noqa: E402
import run_eval as R  # noqa: E402  只读复用（P23：①④定义、encode、掩码解码）

C, MM = R.C, R.MM
C.CACHE = HERE / "cache"           # 覆盖 P23 目录：不写他人目录
RESULTS = HERE / "results"
L_GROUP = (35, 0, 1, 2)
SPLITS = ("a_bal", "test", "adv2")
ARMS = ("F", "Cmode", "Cqc", "IX")
#: V0 复现目标（P23 REPORT §3，a_bal·UP·2seed）
V0_REF = {"Cmode": (0.2606, 0.2673), "Cqc": (0.2500, 0.2510)}


def se(n: int) -> float:
    return math.sqrt(0.25 / n)


def acc_of(pred, gold) -> float:
    return sum(1 for a, b in zip(pred, gold) if a == b) / len(gold)


def paired(a: list[int], b: list[int]) -> dict:
    """b − a 的配对 Δ（逐行 0/1 差）。"""
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    s = math.sqrt(var / n)
    return {"delta": round(m, 6), "se": round(s, 6),
            "t": round(m / s, 2) if s > 0 else None}


def intersect(a: list[list[int]], b: list[list[int]]) -> list[list[int]]:
    return [[x for x in ka if x in set(kb)] for ka, kb in zip(a, b)]


def lookup(fit_keys, fit_y, eval_keys):
    """PREREG §2 口径（= free_rule_floor/rules.py::lookup）：fit=train，未见键回退 train 全局多数。"""
    glob = Counter(fit_y).most_common(1)[0][0]
    tab: dict = defaultdict(Counter)
    for k, y in zip(fit_keys, fit_y):
        tab[k][y] += 1
    maj = {k: c.most_common(1)[0][0] for k, c in tab.items()}
    return [maj.get(k, glob) for k in eval_keys]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    spec = C.Spec()
    torch.manual_seed(0)
    tab = MM.skeleton_table()
    sA = MM.subsets_A(tab)
    rows = {s: C.load_rows(R.SPLIT_PATH[s]) for s in SPLITS}
    if a.smoke:
        for s in rows:
            rows[s] = rows[s][:100]
    train = C.load_rows(C.TCH / "data" / "train.jsonl")
    st = R.build_stats(train)
    print(f"[sig] train 覆盖 {len(st['covered'])}/40 未见={st['unseen']}", flush=True)

    # ---- 四臂 keep（与 seed 无关；①④ 定义 = P23 run_eval 原文，未改）----
    keeps: dict[str, dict[str, list]] = {}
    struct: dict[str, dict] = {}
    for s in SPLITS:
        sig = R.make_keeps(st, sA, rows[s])
        ix = intersect(sig[0], sig[3])
        keeps[s] = {"F": None, "Cmode": sig[0], "Cqc": sig[3], "IX": ix}
        stq, st1, stx = R.size_stats(sig[3]), R.size_stats(sig[0]), R.size_stats(ix)
        struct[s] = {
            "Cqc": stq, "Cmode": st1, "IX": stx,
            "eq_1_4": sum(1 for x, y in zip(sig[0], sig[3]) if set(x) == set(y)) / len(ix),
            "qc_sub_1": sum(1 for x, y in zip(sig[0], sig[3]) if set(y) <= set(x)) / len(ix),
            "one_sub_qc": sum(1 for x, y in zip(sig[0], sig[3]) if set(x) <= set(y)) / len(ix),
            "jac_med": sorted(len(set(x) & set(y)) / max(1, len(set(x) | set(y)))
                              for x, y in zip(sig[0], sig[3]))[len(ix) // 2],
            "n_diff": sum(1 for x, y in zip(sig[0], sig[3]) if set(x) != set(y)),
        }
        print(f"[keep] {s} Cmode med={st1['med']} Cqc med={stq['med']}/e{stq['empty']} "
              f"IX med={stx['med']}/{stx['p25']}-{stx['p75']}/e{stx['empty']}", flush=True)

    blob = {s: C.encode_rows(rows[s], spec, "cpu") for s in SPLITS}
    labs = {s: C.label_tensors(rows[s]) for s in SPLITS}

    out: dict = {"prereg": str(HERE / "PREREG.md"), "arms": list(ARMS),
                 "n": {s: len(rows[s]) for s in SPLITS},
                 "se": {s: round(se(len(rows[s])), 4) for s in SPLITS},
                 "struct": struct, "seeds": {}}

    for seed in C.SEEDS:
        model = C.load_arm("UP", seed, spec, "cpu")
        for s in SPLITS:
            with torch.no_grad():
                vb = C.v_bag_of(blob[s]["v_items"], blob[s]["item_mask"])
                lab_in = None
                if model.lab is not None:
                    lab_in = {k: labs[s][k] for k in ("type_t", "role_t", "cls_t", "pos_b")}
                    lab_in["mask"] = labs[s]["mask"]
                h_tok = tmask = None
                if model.pool is not None:
                    h_tok, tmask = blob[s]["h"], blob[s]["hmask"]
                sk, _, _ = model.forward(blob[s]["v_sent"], vb, blob[s]["v_items"],
                                         blob[s]["item_mask"], lab_in, h_tok, tmask)
            logits = sk.cpu()
            gold = blob[s]["skel"].tolist()
            n = len(gold)
            # V4 随机置换对照（同子集大小多重集，seed*1000+99，P23/P8 同口径）
            g = torch.Generator().manual_seed(seed * 1000 + 99)
            p = torch.randperm(n, generator=g).tolist()
            arms = dict(keeps[s])
            arms["IXr"] = [arms["IX"][j] for j in p]

            rec: dict = {"acc": {}, "correct": {}, "cov": {}, "empty": {}, "size": {}}
            for name, ks in arms.items():
                pred = R.mask_argmax(logits, ks)
                corr = [int(x == y) for x, y in zip(pred, gold)]
                rec["acc"][name] = round(acc_of(pred, gold), 6)
                rec["correct"][name] = corr
                if ks is not None:
                    rec["cov"][name] = round(
                        sum(1 for i, k in enumerate(ks) if gold[i] in k) / n, 6)
                    rec["empty"][name] = sum(1 for k in ks if not k)
                    rec["size"][name] = R.size_stats(ks)
                else:
                    rec["size"][name] = {"n_rows": n, "med": 40, "empty": 0}
            # L / 非 L 分列（V4）
            grp = {}
            for gn, keep in (("L", [i for i in range(n) if gold[i] in L_GROUP]),
                             ("nonL", [i for i in range(n) if gold[i] not in L_GROUP])):
                grp[gn] = {"n": len(keep)}
                if keep:
                    grp[gn]["acc"] = {k: round(sum(rec["correct"][k][i] for i in keep)
                                               / len(keep), 6) for k in ARMS}
            rec["groups"] = grp
            # 配对 Δ（V1/V5/V4）
            rec["delta"] = {
                "IX-Cmode": paired(rec["correct"]["Cmode"], rec["correct"]["IX"]),
                "IX-F": paired(rec["correct"]["F"], rec["correct"]["IX"]),
                "Cqc-Cmode": paired(rec["correct"]["Cmode"], rec["correct"]["Cqc"]),
                "Cqc-F": paired(rec["correct"]["F"], rec["correct"]["Cqc"]),
                "IX-Cqc": paired(rec["correct"]["Cqc"], rec["correct"]["IX"]),
                "IX-IXr": paired(rec["correct"]["IXr"], rec["correct"]["IX"]),
            }
            # V5 消融：① 与 ④ 分歧行上的三臂 acc
            diff = [i for i in range(n)
                    if set(keeps[s]["Cmode"][i]) != set(keeps[s]["Cqc"][i])]
            rec["v5"] = {"n_diff": len(diff),
                         "acc_on_diff": {k: (round(sum(rec["correct"][k][i] for i in diff)
                                                   / len(diff), 6) if diff else None)
                                         for k in ("Cmode", "Cqc", "IX")}}
            out["seeds"].setdefault(f"s{seed}", {})[s] = rec
            print(f"[eval] s{seed} {s:6s} " + " ".join(
                f"{k}={rec['acc'][k]:.4f}" for k in ARMS + ("IXr",)) +
                f" covCqc={rec['cov'].get('Cqc')} covIX={rec['cov'].get('IX')} "
                f"eIX={rec['empty'].get('IX')} d(IX-Cmode)={rec['delta']['IX-Cmode']}",
                flush=True)
        del model

    # ---- V2：同出口最强免费查表（PREREG §2 12 条 + P8 rules.json 并集）----
    ytr = [r["skel_id"] for r in train]
    gr = [MM.MASK_ALIAS[MM.mode_of_sent(r["sent"])] for r in train]
    k4 = [R.sig4(r) for r in train]
    nt = [r["n_slots"] for r in train]
    keyset = {
        "n_slots": nt,
        "mode": gr,
        "sig4": k4,
        "mode+n_slots": list(zip(gr, nt)),
        "mode+sig4": list(zip(gr, k4)),
        "n_slots+last_char": [(r["n_slots"], r["sent"][-1]) for r in train],
        "mode+last_char": list(zip(gr, [r["sent"][-1] for r in train])),
        "mode+n_slots+last_char": list(zip(gr, nt, [r["sent"][-1] for r in train])),
        "n_slots+first_char": [(r["n_slots"], r["sent"][0]) for r in train],
        "n_slots+len_bucket": [(r["n_slots"], len(r["sent"]) // 6) for r in train],
        "mode+len_bucket": list(zip(gr, [len(r["sent"]) // 6 for r in train])),
    }
    p8 = json.loads((ROOT / "experiments" / "mode_conditioned_skel" / "results"
                     / "rules.json").read_text(encoding="utf-8"))
    best: dict = {}
    for s in SPLITS:
        rr = rows[s]
        gr_e = [MM.MASK_ALIAS[MM.mode_of_sent(r["sent"])] for r in rr]
        nt_e = [r["n_slots"] for r in rr]
        k4_e = [R.sig4(r) for r in rr]
        eval_keys = {"n_slots": nt_e, "mode": gr_e, "sig4": k4_e,
                     "mode+n_slots": list(zip(gr_e, nt_e)),
                     "mode+sig4": list(zip(gr_e, k4_e)),
                     "n_slots+last_char": [(r["n_slots"], r["sent"][-1]) for r in rr],
                     "mode+last_char": list(zip(gr_e, [r["sent"][-1] for r in rr])),
                     "mode+n_slots+last_char": list(
                         zip(gr_e, nt_e, [r["sent"][-1] for r in rr])),
                     "n_slots+first_char": [(r["n_slots"], r["sent"][0]) for r in rr],
                     "n_slots+len_bucket": [(r["n_slots"], len(r["sent"]) // 6) for r in rr],
                     "mode+len_bucket": list(zip(gr_e, [len(r["sent"]) // 6 for r in rr]))}
        y = [r["skel_id"] for r in rr]
        bat = {nm: round(acc_of(lookup(keyset[nm], ytr, eval_keys[nm]), y), 4)
               for nm in keyset}
        p8r = p8["splits"][s]
        bat["old8max"] = p8r["max_naive_old8"]
        ref = {k: p8r[k] for k in ("max_naive_complete", "max_naive_plus",
                                   "max_naive_disclosure")}
        best[s] = {"battery": bat, "p8_ref": ref,
                   "best": round(max([*bat.values(), *ref.values()]), 4),
                   "winner": max({**bat, **ref}, key=lambda k: max(
                       bat.get(k, 0), ref.get(k, 0)))}
        print(f"[best] {s} = {best[s]['best']} ({best[s]['winner']}) "
              f"battery={bat}", flush=True)
    out["lookup"] = best

    # ---- 汇总（V0/V1/V2 逐出口）----
    summ: dict = {}
    for s in SPLITS:
        e: dict = {"acc": {}, "card_minus_best": {}}
        for k in ARMS + ("IXr",):
            e["acc"][k] = [out["seeds"][f"s{sd}"][s]["acc"][k] for sd in C.SEEDS]
            e["card_minus_best"][k] = [round(v - best[s]["best"], 4)
                                       for v in e["acc"][k]]
        e["delta"] = {k: [out["seeds"][f"s{sd}"][s]["delta"][k] for sd in C.SEEDS]
                      for k in out["seeds"]["s42"][s]["delta"]}
        e["cov"] = {k: [out["seeds"][f"s{sd}"][s]["cov"].get(k) for sd in C.SEEDS]
                    for k in ("Cmode", "Cqc", "IX")}
        e["empty"] = {k: [out["seeds"][f"s{sd}"][s]["empty"].get(k) for sd in C.SEEDS]
                      for k in ("Cqc", "IX")}
        summ[s] = e
    out["summary"] = summ
    out["v0"] = {k: [summ["a_bal"]["acc"][k][i] for i in range(2)] for k in V0_REF}
    out["v0_pass"] = {k: [abs(out["v0"][k][i] - V0_REF[k][i]) for i in range(2)]
                      for k in V0_REF}

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print("[V0] " + json.dumps(out["v0_pass"], ensure_ascii=False), flush=True)
    for s in SPLITS:
        print(f"[sum] {s} best={best[s]['best']} " +
              " ".join(f"{k}={'/'.join(f'{v:.4f}' for v in summ[s]['acc'][k])}"
                       for k in ARMS + ("IXr",)), flush=True)
    print(f"[done] → {RESULTS / 'eval.json'}", flush=True)


if __name__ == "__main__":
    main()
