#!/usr/bin/env python3
"""free_rule_floor 分析：F0 复现对账 + F1 主判据 + F2 完备 max_naive + F3 旁路探针。

判据全部来自 PREREG §2（跑前写死），本文件不引入新阈值。

用法：uv run python experiments/free_rule_floor/analyze.py
"""
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

import narm as NA  # noqa: E402
import train as T  # noqa: E402

RESULTS = HERE / "results"
SPLITS = ("test", "adv1", "adv2")
ARMS = ("A", "N", "U", "UP")
SEEDS = (42, 43)

# F0 参照：bag_modules/REPORT.md §3 报告值（只读引用，不改其文件）
REF = {
    "A": {42: {"test": 0.6396, "adv2": 0.2310, "slot_test": 0.9658, "slot_adv1": 0.9776},
          43: {"test": 0.6424, "adv2": 0.2390, "slot_test": 0.9659, "slot_adv1": 0.9755}},
    "U": {42: {"test": 0.8328, "adv2": 0.3490}, 43: {"test": 0.8340, "adv2": 0.3540}},
    "UP": {42: {"test": 0.8416, "adv2": 0.3800}, 43: {"test": 0.8440, "adv2": 0.3760}},
}
SE_REF = {"test": 0.0100, "adv2": 0.0158}
MAX_NAIVE = T.MAX_NAIVE
MAX_NAIVE_OLD = T.MAX_NAIVE_OLD
MAX_NAIVE_OLD8 = T.MAX_NAIVE_OLD8


def load(name: str) -> dict:
    p = RESULTS / f"{name}.json"
    if not p.exists():
        raise SystemExit(f"缺结果：{p}")
    return json.loads(p.read_text(encoding="utf-8"))


def acc(res: dict, split: str, k: str = "skel") -> float:
    return res["meta"]["eval"][split][k]["acc"]


def se(res: dict, split: str, k: str = "skel") -> float:
    return res["meta"]["eval"][split][k]["se"]


def paired(a: dict, b: dict, split: str, key: str = "skel") -> dict:
    """Δ = b − a 的配对 SE（逐行 0/1 差）。"""
    x = a["correct"][split][key]
    y = b["correct"][split][key]
    assert len(x) == len(y)
    d = [u - v for u, v in zip(y, x)]
    n = len(d)
    mu = sum(d) / n
    var = sum((v - mu) ** 2 for v in d) / (n - 1) if n > 1 else 0.0
    s = math.sqrt(var / n)
    return {"delta": round(mu, 6), "se": round(s, 6),
            "t": round(mu / s, 3) if s > 0 else None, "n": n}


# ---------------------------------------------------------------------------
# F0：与 bag_modules 报告值对账
# ---------------------------------------------------------------------------
def f0(allr: dict) -> dict:
    rep, ok = {}, True
    for arm in ("A", "U", "UP"):
        rep[arm] = {}
        for s in SEEDS:
            r = allr[f"{arm}_s{s}"]["meta"]["eval"]
            rec = {}
            for split in ("test", "adv2"):
                mine, ref = r[split]["skel"]["acc"], REF[arm][s][split]
                d = round(mine - ref, 6)
                lim = 2 * SE_REF[split]
                rec[split] = {"mine": mine, "ref_bag_modules": ref, "delta": d,
                              "se": r[split]["skel"]["se"], "lim_2se": round(lim, 4),
                              "pass": bool(abs(d) <= lim)}
                ok = ok and abs(d) <= lim
            rec["slot_test"] = {"mine": r["test"]["slot"]["acc"],
                                "ref": REF[arm][s].get("slot_test")}
            if "slot_test" in REF[arm][s]:
                rec["slot_test"]["delta"] = round(
                    r["test"]["slot"]["acc"] - REF[arm][s]["slot_test"], 6)
                ok = ok and abs(rec["slot_test"]["delta"]) <= 2 * 0.0022
            rep[arm][f"s{s}"] = rec
    rep["pass"] = bool(ok)
    return rep


# ---------------------------------------------------------------------------
# F3：评测期旁路打乱
# ---------------------------------------------------------------------------
def paired_vec(a: list, b: list) -> dict:
    """Δ = b − a 的配对 SE（逐行 0/1）；a=true、b=shuffled ⇒ Δ 为负。"""
    d = [u - v for u, v in zip(b, a)]
    n = len(d)
    mu = sum(d) / n
    var = sum((v - mu) ** 2 for v in d) / (n - 1) if n > 1 else 0.0
    s = math.sqrt(var / n)
    return {"delta": round(mu, 6), "se": round(s, 6),
            "t": round(mu / s, 3) if s > 0 else None, "n": n}


def shuffle_probe(allr: dict, arm: str, seed: int, device: str) -> dict:
    spec = NA.Spec()
    m = NA.build_model(arm, seed, spec).to(device)
    sd = torch.load(HERE / "weights" / f"{arm}_s{seed}.pt", map_location="cpu",
                    weights_only=True)
    miss = [k for k in m.state_dict() if k not in sd]
    assert all(k.startswith("encoder.") for k in miss), miss[:5]   # 只许缺冻结核
    m.load_state_dict(sd, strict=False)
    m.eval()
    g = torch.Generator().manual_seed(seed * 1000 + 99)
    gc = torch.Generator().manual_seed(seed * 1000 + 199)
    out = {}
    for split in SPLITS:
        rows = T.get_rows(split)
        blob = T.get_blob(rows, spec, device, split)
        labs = T.label_tensors(split, rows)
        tok = T.encode_tokens(m, rows, spec, device) if m.pool is not None else None
        perm = torch.randperm(len(rows), generator=g)          # 全打乱（含 mask）
        perm_c = torch.randperm(len(rows), generator=gc)       # 只打内容、留 mask
        labs_all = {k: (v[perm] if v.shape[0] == len(rows) else v)
                    for k, v in labs.items()}
        labs_ct = dict(labs)
        for k in ("type_t", "role_t", "cls_t", "pos_b"):
            labs_ct[k] = labs[k][perm_c]
        r_true, c_true, _ = T.evaluate(m, blob, rows, labs, tok, device)
        rec = {"true": r_true["skel"]["acc"], "slot_true": r_true["slot"]["acc"]}
        if arm == "A":
            rec["note"] = "无标签通道，不适用"
        else:
            r_all, c_all, _ = T.evaluate(m, blob, rows, labs_all, tok, device)
            r_ct, c_ct, _ = T.evaluate(m, blob, rows, labs_ct, tok, device)
            rec.update({
                "shuffled_all": r_all["skel"]["acc"],
                "drop_all": round(rec["true"] - r_all["skel"]["acc"], 4),
                "paired_all": paired_vec(c_true["skel"], c_all["skel"]),
                "slot_shuffled_all": r_all["slot"]["acc"],
                "shuffled_content_only": r_ct["skel"]["acc"],
                "drop_content_only": round(rec["true"] - r_ct["skel"]["acc"], 4),
                "paired_content": paired_vec(c_true["skel"], c_ct["skel"])})
        rec["max_naive"] = MAX_NAIVE[split]
        out[split] = rec
    return out


# ---------------------------------------------------------------------------
def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    allr = {}
    for a in ARMS:
        for s in SEEDS:
            allr[f"{a}_s{s}"] = load(f"{a}_s{s}")
    for s in SEEDS:
        allr[f"U_s{s}_rand"] = load(f"U_s{s}_rand")
    bat = json.loads((RESULTS / "battery.json").read_text(encoding="utf-8"))

    rep: dict = {"prereg": "PREREG.md", "battery": "results/battery.json",
                 "max_naive": {"new": MAX_NAIVE, "old_reported": MAX_NAIVE_OLD,
                               "old8_recomputed": MAX_NAIVE_OLD8}}

    # ---- F0 ----
    rep["F0_reconciliation"] = f0(allr)
    print("=" * 84)
    print("F0 复现对账（我的 vs bag_modules/REPORT.md；门槛 |Δ| ≤ 2×SE）")
    for arm in ("A", "U", "UP"):
        for s in SEEDS:
            r = rep["F0_reconciliation"][arm][f"s{s}"]
            print(f"  {arm:2s} s{s}: test {r['test']['mine']:.4f} vs "
                  f"{r['test']['ref_bag_modules']:.4f} (Δ={r['test']['delta']:+.4f}) | "
                  f"adv2 {r['adv2']['mine']:.4f} vs {r['adv2']['ref_bag_modules']:.4f} "
                  f"(Δ={r['adv2']['delta']:+.4f}) | slot_test Δ="
                  f"{r['slot_test'].get('delta')}")
    print(f"  => F0 {'通过' if rep['F0_reconciliation']['pass'] else '未通过（先修探针）'}")

    # ---- 主表 ----
    print("=" * 84)
    print("主表（骨架 acc ± 行级 SE；卡−max_naive 新/旧两个口径）")
    for split in SPLITS:
        print(f"\n--- {split}  max_naive 新={MAX_NAIVE[split]} "
              f"旧={MAX_NAIVE_OLD[split]} 旧8重算={MAX_NAIVE_OLD8[split]} "
              f"多数类={bat['splits'][split]['majority']}")
        print(f"{'臂':6s} {'seed':4s} {'骨架':>6s} {'±SE':>6s} {'槽位':>6s} "
              f"{'卡−新地板':>9s} {'卡−旧地板':>9s} {'aux_only':>8s} {'head_only':>9s}")
        for a in ARMS:
            for s in SEEDS:
                m = allr[f"{a}_s{s}"]["meta"]["eval"][split]
                d = m["diag"]
                print(f"{a:6s} {s:<4d} {m['skel']['acc']:.4f} {m['skel']['se']:.4f} "
                      f"{m['slot']['acc']:.4f} "
                      f"{m['skel']['acc'] - MAX_NAIVE[split]:+.4f} "
                      f"{m['skel']['acc'] - MAX_NAIVE_OLD[split]:+.4f} "
                      f"{str(d['aux_only']):>8s} {str(d['head_only']):>9s}")

    # ---- F1：N vs U（配对）----
    f1 = {}
    for split in SPLITS:
        f1[split] = {}
        for s in SEEDS:
            f1[split][f"s{s}"] = paired(allr[f"N_s{s}"], allr[f"U_s{s}"], split, "skel")
    # 补充配对：U−A、N−A、UP−U、U−N 的两 split
    extra = {}
    for name, (b, a_) in {"U-A": ("U", "A"), "N-A": ("N", "A"), "UP-U": ("UP", "U"),
                          "U-rand-A": ("U_rand", "A")}.items():
        extra[name] = {}
        for split in ("test", "adv2"):
            extra[name][split] = {}
            for s in SEEDS:
                ka = f"{a_}_s{s}" if a_ != "U_rand" else f"U_s{s}_rand"
                kb = f"{b}_s{s}" if b != "U_rand" else f"U_s{s}_rand"
                extra[name][split][f"s{s}"] = paired(allr[ka], allr[kb], split, "skel")
    rep["F1_N_vs_U"] = f1
    rep["paired_extra"] = extra
    print("=" * 84)
    print("F1 主判据：Δ = U − N（配对 SE，逐行 0/1）")
    for split in ("test", "adv2"):
        for s in SEEDS:
            x = f1[split][f"s{s}"]
            print(f"  {split} s{s}: Δ={x['delta']:+.4f} SE={x['se']:.4f} t={x['t']} "
                  f"n={x['n']}  {'|Δ|>2SE' if abs(x['delta']) > 2 * x['se'] else '|Δ|≤2SE'}")
    print("  配对补充（Δ，两 seed）：")
    for k, v in extra.items():
        line = []
        for split in ("test", "adv2"):
            for s in SEEDS:
                x = v[split][f"s{s}"]
                line.append(f"{split}s{s} {x['delta']:+.4f}(t={x['t']})")
        print(f"    {k:10s} " + "  ".join(line))

    # ---- F1 判定（PREREG §2 写死；test 与 adv2 **两个出口都要成立**才判正/反向，
    #      任一出口定不了号 ⇒ 证据不足 —— 取最严读法）----
    det, over_flags, sign_all = [], [], True
    for split in ("test", "adv2"):
        ds = [f1[split][f"s{s}"] for s in SEEDS]
        over = [abs(x["delta"]) > 2 * x["se"] for x in ds]
        sgn = all((x["delta"] >= 0) == (ds[0]["delta"] >= 0) for x in ds)
        det.append(f"{split}: Δ={[x['delta'] for x in ds]} "
                   f"SE={[x['se'] for x in ds]} t={[x['t'] for x in ds]} "
                   f"|Δ|>2SE={over} 同号={sgn}")
        over_flags.append(over)
        sign_all = sign_all and sgn
    sig_all = all(all(o) for o in over_flags)
    none_over = not any(any(o) for o in over_flags)
    if none_over and sign_all:
        verdict = "增益由 n_slots 承载（模块内容无用）"
    elif sig_all and sign_all:
        verdict = "模块内容有独立贡献"
    else:
        verdict = "证据不足"
    rep["F1_verdict"] = {"verdict": verdict, "detail": det}

    # ---- 负对照 ----
    neg = {"pass": True, "detail": {}}
    for s in SEEDS:
        x = extra["U-rand-A"]["test"][f"s{s}"]
        neg["detail"][f"s{s}"] = x
        neg["pass"] = neg["pass"] and x["delta"] > 2 * x["se"] and x["delta"] > 0
    rep["negative_control"] = neg

    # ---- F3 旁路探针 ----
    print("=" * 84)
    print("F3 旁路探针（评测期打乱；true → shuffled，骨架 acc）")
    shuf = {}
    for a in ("N", "U", "UP"):
        shuf[a] = {}
        for s in SEEDS:
            shuf[a][f"s{s}"] = shuffle_probe(allr, a, s, device)
            for split in SPLITS:
                r = shuf[a][f"s{s}"][split]
                print(f"  {a} s{s} {split:5s} true={r['true']:.4f} "
                      f"全打乱={r.get('shuffled_all', 'n/a')} "
                      f"drop={r.get('drop_all', 'n/a')}"
                      f"(t={r.get('paired_all', {}).get('t')}) | 只打内容={r.get('shuffled_content_only', 'n/a')} "
                      f"drop={r.get('drop_content_only', 'n/a')}"
                      f"(t={r.get('paired_content', {}).get('t')}) | "
                      f"槽位 true={r['slot_true']:.4f}")
    shuf["A"] = {"note": "无标签通道，不适用"}
    rep["F3_shuffle"] = shuf

    # ---- F2 汇总 ----
    f2 = {}
    for a in ARMS:
        for s in SEEDS:
            m = allr[f"{a}_s{s}"]["meta"]["eval"]
            f2[f"{a}_s{s}"] = {sp: {"acc": m[sp]["skel"]["acc"],
                                    "minus_max_naive_new": round(
                                        m[sp]["skel"]["acc"] - MAX_NAIVE[sp], 4),
                                    "minus_max_naive_old": round(
                                        m[sp]["skel"]["acc"] - MAX_NAIVE_OLD[sp], 4)}
                               for sp in SPLITS}
    rep["F2_over_floor"] = f2

    # ---- 负对照 print ----
    print("=" * 84)
    print("负对照（U-rand 必须复现「增益不消失」）")
    for s in SEEDS:
        x = neg["detail"][f"s{s}"]
        print(f"  s{s}: Δ_test(U-rand − A)={x['delta']:+.4f} SE={x['se']:.4f} "
              f"t={x['t']} => {'过' if x['delta'] > 2 * x['se'] else '不过'}")
    print(f"  负对照通过 = {neg['pass']}")
    print(f"\n判定（三选一）：{rep['F1_verdict']['verdict']}")
    for d in rep["F1_verdict"]["detail"]:
        print("  " + d)

    (RESULTS / "analyze.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n[done] → results/analyze.json", flush=True)


if __name__ == "__main__":
    main()
