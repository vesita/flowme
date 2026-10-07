#!/usr/bin/env python3
"""P22 主评测：同一模型、同一 logits，只换「路由信号」的解码掩码。PREREG §0–§1。

四臂 F / C-mode / C-rarity / C-both + 随机标签对照（seed*1000+99，P8 同口径）。
**只读 import** P8 的 `common.py` / `modes.py`；缓存写入目录被改写到本目录（不碰既有实验）。

用法：uv run python experiments/rarity_routing/run_eval.py [--smoke] [--arm UP]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

import torch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P8 = ROOT / "experiments" / "mode_conditioned_skel"
sys.path.insert(0, str(P8))

import common as C  # noqa: E402  只读复用（P8）
import modes as MM  # noqa: E402  只读复用（P8）

C.CACHE = HERE / "cache"          # 关键：缓存只写本目录，绝不写 P8 目录
RESULTS = HERE / "results"
SPLIT_PATH = {
    "test": C.TCH / "data" / "test.jsonl",
    "adv2": C.SS / "data" / "adv2.jsonl",
    "a_bal": C.SLEAK / "data" / "a_bal.jsonl",
}
L_GROUP = (35, 0, 1, 2)
DECODES = ("F", "C_mode", "C_rarity", "C_both",
           "C_mode_rand", "C_rarity_rand", "C_both_rand")


def se(n: int) -> float:
    return math.sqrt(0.25 / n)


def mask_argmax(logits: torch.Tensor, keep: list[list[int]] | None) -> list[int]:
    if keep is None:
        return logits.argmax(-1).tolist()
    out = []
    for i, ks in enumerate(keep):
        if not ks:                       # 空子集 ⇒ 回退 F（计数另行报出）
            out.append(int(logits[i].argmax()))
            continue
        m = torch.full((logits.shape[1],), float("-inf"), dtype=logits.dtype)
        m[ks] = 0.0
        out.append(int((logits[i] + m).argmax()))
    return out


def acc_of(pred: list[int], gold: list[int]) -> float:
    return sum(1 for a, b in zip(pred, gold) if a == b) / len(gold)


def ns_buckets(train: list[dict]) -> tuple[dict[int, int], dict[int, list[int]]]:
    """桶函数 ns(sid)（train 确定性，train 未见 ⇒ 不入桶）+ 每个 n_slots 的骨架集。"""
    seen: dict[int, set] = defaultdict(set)
    for r in train:
        seen[r["skel_id"]].add(r["n_slots"])
    ns_of: dict[int, int] = {}
    for sid, v in seen.items():
        C.check(len(v) == 1, f"骨架 {sid} 的 n_slots 非确定性：{sorted(v)}")
        ns_of[sid] = next(iter(v))
    buck: dict[int, list[int]] = defaultdict(list)
    for sid, ns in sorted(ns_of.items()):
        buck[ns].append(sid)
    return ns_of, dict(buck)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--arm", default="UP")
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = C.Spec()
    torch.manual_seed(0)
    tab = MM.skeleton_table()
    print(f"[device] {device}", flush=True)

    sA = MM.subsets_A(tab)
    rows = {s: C.load_rows(p) for s, p in SPLIT_PATH.items()}
    if a.smoke:
        for s in rows:
            rows[s] = rows[s][:100]
    train = C.load_rows(C.TCH / "data" / "train.jsonl")
    ns_of, buck = ns_buckets(train)
    unseen = sorted(set(range(40)) - set(ns_of))
    print(f"[rarity] train 覆盖骨架 {len(ns_of)}/40；未见(常量排除) = {unseen}", flush=True)
    print(f"[rarity] 桶 = { {k: v for k, v in sorted(buck.items())} }", flush=True)

    blob = {s: C.encode_rows(rows[s], spec, device) for s in SPLIT_PATH}
    labs = {s: C.label_tensors(rows[s]) for s in SPLIT_PATH}

    rule_mode = {s: [MM.MASK_ALIAS[MM.mode_of_sent(r["sent"])] for r in rows[s]]
                 for s in rows}
    # 行级信号 = 输入侧 n_slots（**不用 gold**）；覆盖对账单独报
    ns_lab = {s: [r["n_slots"] for r in rows[s]] for s in rows}
    cov = {}
    for s in rows:
        hit = sum(1 for r in rows[s]
                  if r["skel_id"] in set(buck.get(r["n_slots"], [])))
        cov[s] = round(hit / len(rows[s]), 6)
    print(f"[cov] C-rarity gold 覆盖率（结构） = {cov}", flush=True)

    struct = {"ns_of": {str(k): v for k, v in sorted(ns_of.items())},
              "buckets": {str(k): v for k, v in sorted(buck.items())},
              "train_unseen_constant_excluded": unseen,
              "gold_coverage": cov,
              "size_A": {m: len(sA[m]) for m in MM.MODES},
              "per_split": {}}
    for s in rows:
        cm = Counter(rule_mode[s])
        cn = Counter(ns_lab[s])
        struct["per_split"][s] = {
            "n": len(rows[s]),
            "mode_rows": dict(cm),
            "mode_subset_size": {m: len(sA[m]) for m in cm},
            "ns_rows": {str(k): v for k, v in sorted(cn.items())},
            "ns_subset_size": {str(k): len(buck.get(k, [])) for k in sorted(cn)},
            "fallback": {"mode": sum(1 for m in rule_mode[s] if not sA[m]),
                         "ns": sum(1 for k in ns_lab[s] if not buck.get(k))},
        }
        print(f"[struct] {s} {struct['per_split'][s]}", flush=True)

    arms = (a.arm,)
    out: dict = {"device": device, "arm": a.arm, "n": {s: len(rows[s]) for s in SPLIT_PATH},
                 "struct": struct, "rows": {}}
    fallback: dict = {}

    for arm in arms:
        for seed in C.SEEDS:
            model = C.load_arm(arm, seed, spec, device)
            key = f"{arm}_s{seed}"
            out["rows"][key] = {}
            for s in SPLIT_PATH:
                t0 = time.time()
                with torch.no_grad():
                    vs = blob[s]["v_sent"].to(device)
                    vi = blob[s]["v_items"].to(device)
                    im = blob[s]["item_mask"].to(device)
                    vb = C.v_bag_of(vi, im)
                    lab_in = None
                    if model.lab is not None:
                        lab_in = {k: labs[s][k].to(device) for k in
                                  ("type_t", "role_t", "cls_t", "pos_b")}
                        lab_in["mask"] = labs[s]["mask"].to(device)
                    h_tok = tmask = None
                    if model.pool is not None:
                        h_tok = blob[s]["h"].to(device)
                        tmask = blob[s]["hmask"].to(device)
                    sk_logits, _, _ = model.forward(vs, vb, vi, im, lab_in, h_tok, tmask)
                logits = sk_logits.cpu()
                gold = blob[s]["skel"].tolist()
                n = len(gold)

                keep_mode = [sA[m] for m in rule_mode[s]]
                keep_ns = [buck.get(k, []) for k in ns_lab[s]]
                keep_both = [[x for x in a_ if x in set(b_)]
                             for a_, b_ in zip(keep_mode, keep_ns)]
                # 随机标签对照（seed*1000+99，P8/skeleton_leak 同口径）
                g = torch.Generator().manual_seed(seed * 1000 + 99)
                p = torch.randperm(n, generator=g)
                rm = [rule_mode[s][int(i)] for i in p]
                rn = [ns_lab[s][int(i)] for i in p]
                keep_mr = [sA[m] for m in rm]
                keep_nr = [buck.get(k, []) for k in rn]
                keep_br = [[x for x in a_ if x in set(b_)]
                           for a_, b_ in zip(keep_mr, keep_nr)]

                keeps = {"F": None, "C_mode": keep_mode, "C_rarity": keep_ns,
                         "C_both": keep_both, "C_mode_rand": keep_mr,
                         "C_rarity_rand": keep_nr, "C_both_rand": keep_br}
                rec: dict = {"gold": gold, "ns_lab": ns_lab[s], "rule_mode": rule_mode[s],
                             "L": [1 if x in L_GROUP else 0 for x in gold],
                             "decode": {}}
                fb = {}
                for d, ks in keeps.items():
                    pred = mask_argmax(logits, ks)
                    corr = [1 if x == y else 0 for x, y in zip(pred, gold)]
                    rec["decode"][d] = {"acc": round(acc_of(pred, gold), 6),
                                        "se": round(se(n), 6), "correct": corr,
                                        "pred": pred}
                    if ks is not None:
                        fb[d] = sum(1 for kk in ks if not kk)
                rec["fallback_empty_subset"] = fb
                fallback[f"{key}|{s}"] = fb
                out["rows"][key][s] = rec
                print(f"[eval] {key} {s:6s} " +
                      " ".join(f"{d}={rec['decode'][d]['acc']:.4f}" for d in DECODES) +
                      f" fb={fb} ({time.time() - t0:.1f}s)", flush=True)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    out["fallback_empty_subset"] = fallback
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[done] → {RESULTS / 'eval.json'}", flush=True)


if __name__ == "__main__":
    main()
