#!/usr/bin/env python3
"""P23 主评测：同一模型、同一 logits，只换「多信号候选集」的解码掩码。PREREG §0–§2。

臂 = F + S1..S6（六信号单独）+ A1..A6（累积求交）+ A1r..A6r（随机置换对照，同子集大小）。
**只读 import** P8 的 `common.py` / `modes.py`；缓存写本目录（不碰既有实验）。

用法：uv run python experiments/orthogonal_tighten/run_eval.py [--smoke]
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

C.CACHE = HERE / "cache"
RESULTS = HERE / "results"
SPLIT_PATH = {
    "test": C.TCH / "data" / "test.jsonl",
    "adv2": C.SS / "data" / "adv2.jsonl",
    "a_bal": C.SLEAK / "data" / "a_bal.jsonl",
}
L_GROUP = (35, 0, 1, 2)
TAIL = MM._TAIL
FUNC_LAST = "吗么呢吧"
SIG_NAMES = ("S1_句式", "S2_nslots", "S3_稀有度b", "S4_合格候选", "S5_首末字", "S6_类型库存")
#: 累积顺序 = PREREG §2（A1=② → ⑤ → ⑥ → ④ → ① → ③），索引 0-based [1,4,5,3,0,2]
CUM_ORDER = (1, 4, 5, 3, 0, 2)
DECODES = tuple(list(SIG_NAMES) +
                [f"A{i + 1}" for i in range(6)] +
                [f"A{i + 1}r" for i in range(6)])


def se(n: int) -> float:
    return math.sqrt(0.25 / n)


# ---------------------------------------------------------------------------
# 行侧键（只读 sent / bag_span / n_slots；无 gold）
# ---------------------------------------------------------------------------
def k_mode(row: dict) -> str:
    return MM.MASK_ALIAS[MM.mode_of_sent(row["sent"])]


def _split_tail(s: str) -> tuple[str, str]:
    t = s.rstrip()
    while t and t[-1] in TAIL:
        t = t[:-1]
    return (t[-1] if t else ""), (s.rstrip()[-1] if s.rstrip() else "")


def k_last(row: dict) -> tuple[str, str]:
    last, raw = _split_tail(row["sent"])
    cls = last if last in FUNC_LAST else "C"
    if raw in "！!?":
        p = "！"
    elif raw in "。．.…～~）)」』\"'”’":
        p = "。"
    elif raw in "，,；;：:·、":
        p = "，"
    else:
        p = "∅"
    return (cls, p)


def k_firstlast(row: dict) -> tuple[str, tuple[str, str]]:
    s = row["sent"].strip()
    return (s[0] if s else "", k_last(row))


def k_type_seq(row: dict) -> tuple:
    return tuple(l["t"] for l in C.label_row(row))


def sig4(row: dict) -> tuple:
    return (row["n_slots"], k_last(row))


# ---------------------------------------------------------------------------
# 骨架侧（train 统计）+ 候选集
# ---------------------------------------------------------------------------
def build_stats(train: list[dict]) -> dict:
    covered = sorted({r["skel_id"] for r in train})
    ns_of: dict[int, set] = defaultdict(set)
    cnt: Counter = Counter()
    s_last: dict[int, set] = defaultdict(set)
    s_fl: dict[int, set] = defaultdict(set)
    s_seq: dict[int, set] = defaultdict(set)
    s_s4: dict[int, set] = defaultdict(set)
    for r in train:
        sid = r["skel_id"]
        ns_of[sid].add(r["n_slots"])
        cnt[sid] += 1
        s_last[sid].add(k_last(r))
        s_fl[sid].add(k_firstlast(r))
        s_seq[sid].add(k_type_seq(r))
        s_s4[sid].add(sig4(r))
    for sid, v in ns_of.items():
        C.check(len(v) == 1, f"骨架 {sid} 的 n_slots 非确定性：{sorted(v)}")
    ns = {sid: next(iter(v)) for sid, v in ns_of.items()}
    order = sorted(covered, key=lambda s: (cnt[s], s))
    bq = {sid: min(2, i * 3 // len(order)) for i, sid in enumerate(order)}
    # ③ 条件众数：P(bq(gold)=b | train 行键=(句式,n_slots))
    by_key: dict[tuple, Counter] = defaultdict(Counter)
    for r in train:
        by_key[(k_mode(r), r["n_slots"])][bq[r["skel_id"]]] += 1
    modal = {k: min(c, key=lambda b: (-c[b], b)) for k, c in by_key.items()}
    return {"covered": covered, "ns": ns, "cnt": dict(cnt), "bq": bq,
            "modal": modal, "s_last": dict(s_last), "s_fl": dict(s_fl),
            "s_seq": dict(s_seq), "s_s4": dict(s_s4),
            "unseen": sorted(set(range(40)) - set(covered))}


def make_keeps(st: dict, sA: dict, rows: list[dict]) -> list[list[int]]:
    """返回 6 个信号的逐行 keep（与 seed 无关）。"""
    covered = st["covered"]
    cov_set = set(covered)
    sig: list[list[int]] = [[] for _ in range(6)]
    for r in rows:
        m, ns = k_mode(r), r["n_slots"]
        sig[0].append([s for s in sA[m]])
        sig[1].append([s for s in covered if st["ns"][s] == ns])
        b = st["modal"].get((m, ns))
        sig[2].append([s for s in covered if st["bq"][s] == b] if b is not None
                      else list(covered))
        k4 = sig4(r)
        sig[3].append([s for s in covered if k4 in st["s_s4"][s]])
        k5 = k_firstlast(r)
        sig[4].append([s for s in covered if k5 in st["s_fl"][s]])
        k6 = k_type_seq(r)
        sig[5].append([s for s in covered if k6 in st["s_seq"][s]])
    return sig


def cumulate(sig: list[list[list[int]]]) -> list[list[list[int]]]:
    """累积求交：A_k = 前 k 个信号（按 CUM_ORDER）的交；空集保留为空（回退 F）。"""
    out, cur = [], None
    for si in CUM_ORDER:
        nxt = []
        for i in range(len(sig[si])):
            prev = cur[i] if cur is not None else None
            ks = sig[si][i] if prev is None else [x for x in prev if x in set(sig[si][i])]
            nxt.append(ks)
        cur = nxt
        out.append(list(nxt))
    return out


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


def size_stats(keep: list[list[int]] | None) -> dict:
    if keep is None:
        return {"n_rows": 0}
    s = sorted(len(k) for k in keep)
    n = len(s)
    return {"n_rows": n, "med": s[n // 2], "p25": s[n // 4], "p75": s[3 * n // 4],
            "min": s[0], "max": s[-1], "empty": sum(1 for x in s if x == 0)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--arm", default="UP")
    a = ap.parse_args()
    device = a.device
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
    st = build_stats(train)
    print(f"[sig] train 覆盖 {len(st['covered'])}/40；未见(常量排除) = {st['unseen']}", flush=True)
    print(f"[sig] cnt 等频 3 分位 = "
          f"{ {b: sorted(s for s in st['covered'] if st['bq'][s] == b) for b in (0, 1, 2)} }",
          flush=True)
    print(f"[sig] 累积顺序 A1..A6 = {[SIG_NAMES[i] for i in CUM_ORDER]}", flush=True)

    blob = {s: C.encode_rows(rows[s], spec, device) for s in SPLIT_PATH}
    labs = {s: C.label_tensors(rows[s]) for s in SPLIT_PATH}

    keeps_by_split: dict[str, dict] = {}
    for s in rows:
        sig = make_keeps(st, sA, rows[s])
        cum = cumulate(sig)
        arms = {SIG_NAMES[i]: sig[i] for i in range(6)}
        arms.update({f"A{i + 1}": cum[i] for i in range(6)})
        keeps_by_split[s] = arms
        print(f"[keep] {s} " + " ".join(
            f"{k}={size_stats(v)['med']}/{size_stats(v)['p25']}-{size_stats(v)['p75']}"
            f"/e{size_stats(v)['empty']}" for k, v in arms.items()), flush=True)

    out: dict = {"device": device, "arm": a.arm, "n": {s: len(rows[s]) for s in SPLIT_PATH},
                 "cum_order": [SIG_NAMES[i] for i in CUM_ORDER],
                 "struct": {"unseen_constant_excluded": st["unseen"],
                            "cnt_tercile": {str(b): sorted(s for s in st["covered"]
                                                           if st["bq"][s] == b)
                                            for b in (0, 1, 2)},
                            "size_A": {m: len(sA[m]) for m in MM.MODES}},
                 "sizes": {}, "rows": {}}
    for s in rows:
        out["sizes"][s] = {k: size_stats(v) for k, v in keeps_by_split[s].items()}
        nrow = len(rows[s])
        out["sizes"][s]["F"] = {"n_rows": nrow, "med": 40, "p25": 40, "p75": 40,
                                "min": 40, "max": 40, "empty": 0}

    for seed in C.SEEDS:
        model = C.load_arm(a.arm, seed, spec, device)
        key = f"{a.arm}_s{seed}"
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

            # 随机置换对照（同子集大小多重集；seed*1000+99，P8 同口径）
            g = torch.Generator().manual_seed(seed * 1000 + 99)
            p = torch.randperm(n, generator=g).tolist()
            keeps = {"F": None}
            keeps.update(keeps_by_split[s])
            for i in range(6):
                base = keeps[f"A{i + 1}"]
                keeps[f"A{i + 1}r"] = [base[j] for j in p]

            rec: dict = {"gold": gold,
                         "L": [1 if x in L_GROUP else 0 for x in gold],
                         "decode": {}, "coverage": {}, "empty": {}}
            for d in DECODES + ("F",):
                ks = keeps[d]
                pred = mask_argmax(logits, ks)
                corr = [1 if x == y else 0 for x, y in zip(pred, gold)]
                rec["decode"][d] = {"acc": round(acc_of(pred, gold), 6),
                                    "se": round(se(n), 6), "correct": corr}
                if ks is not None:
                    rec["coverage"][d] = round(
                        sum(1 for i, k in enumerate(ks) if gold[i] in k) / n, 6)
                    rec["empty"][d] = sum(1 for k in ks if not k)
            out["rows"][key][s] = rec
            print(f"[eval] {key} {s:6s} " +
                  " ".join(f"{d}={rec['decode'][d]['acc']:.4f}"
                           for d in ("F",) + SIG_NAMES + tuple(f"A{i + 1}" for i in range(6))) +
                  f" emptyA6={rec['empty']['A6']} covA6={rec['coverage']['A6']} "
                  f"({time.time() - t0:.1f}s)", flush=True)
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[done] → {RESULTS / 'eval.json'}", flush=True)


if __name__ == "__main__":
    main()
