#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""P30 极小版: stage5 —— 可达性任务复杂度 D1/D2/D3 × {A0,B} × seed{42,43}.

A0 = 输出卡(输入卡(x))（无思维卡）；B = 输出卡(思维卡(输入卡(x)))。
目标: 从起点 s=0 可达的节点集合（n 维二值, 逐节点 sigmoid + BCE）。
"""
import json
import math
import os
import random
import time

import torch
import torch.nn as nn

# ---- 内联常数 ----
D, NH, FF = 128, 4, 512          # FF=4d（控制 CPU 墙钟）
V, SOS, SEP = 34, 32, 33          # 节点 id 0-31 + 分隔符 + 起点标记
LR, BATCH = 1e-3, 64
N_TRAIN, N_TEST = 8000, 2000
SEEDS = [42, 43]
DIFFS = [("D1", 8, [4, 8, 16]), ("D2", 16, [8, 16, 32]), ("D3", 32, [16, 32, 64])]
TRAIN_BUDGET_S = 45.0            # 每个 run 的训练墙钟预算
CALIB_STEPS, WARMUP = 100, 5
MAX_STEPS, MIN_STEPS = 2000, 50
MAX_WALL_S = 870.0               # 全程硬上限
MAX_LEN = 1 + 3 * 64

os.makedirs("logs", exist_ok=True)


def log(*a):
    print(*a, flush=True)


# ---- 数据 ----
def gen_graph(n, k, rng):
    edges = set()
    while True:                                   # 保证一条长度>=3 的链
        nodes = rng.sample(range(n), 4)
        cand = [(nodes[0], nodes[1]), (nodes[1], nodes[2]), (nodes[2], nodes[3])]
        if all(e not in edges for e in cand):
            edges.update(cand)
            break
    if not any(u == 0 for u, v in edges):          # 保证起点有出边
        ch = [v for v in range(1, n) if (0, v) not in edges]
        edges.add((0, rng.choice(ch)))
    while len(edges) < k:
        u, v = rng.randrange(n), rng.randrange(n)
        if u != v:
            edges.add((u, v))
    return sorted(edges)


def reach(n, edges):
    adj = {}
    for u, v in edges:
        adj.setdefault(u, []).append(v)
    seen = {0}
    st = [0]
    while st:
        u = st.pop()
        for v in adj.get(u, ()):
            if v not in seen:
                seen.add(v)
                st.append(v)
    return seen


def split_counts(total, Ks):
    base, rem = divmod(total, len(Ks))
    return [base + (1 if i < rem else 0) for i in range(len(Ks))]


def build_groups(n, Ks, counts, seed):
    rng = random.Random(seed)
    groups = {}
    for k, c in zip(Ks, counts):
        L = 1 + 3 * k
        X = torch.empty(c, L, dtype=torch.long)
        Y = torch.zeros(c, n)
        for i in range(c):
            e = gen_graph(n, k, rng)
            r = reach(n, e)
            ee = e[:]
            rng.shuffle(ee)
            toks = [SOS]
            for u, v in ee:
                toks += [u, v, SEP]
            X[i] = torch.tensor(toks, dtype=torch.long)
            Y[i] = torch.tensor([1.0 if j in r else 0.0 for j in range(n)])
        groups[k] = (X, Y)
        log(f"    data n={n} k={k}: {c} 样本, len={L}")
    return groups


# ---- 模型 ----
def sinusoidal(max_len, d):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def enc_layer():
    return nn.TransformerEncoderLayer(
        d_model=D, nhead=NH, dim_feedforward=FF, dropout=0.1, batch_first=True
    )


class Net(nn.Module):
    def __init__(self, n, with_thought):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.register_buffer("pe", sinusoidal(MAX_LEN, D))
        self.inp = nn.TransformerEncoder(enc_layer(), 1)          # 输入卡
        self.th = nn.TransformerEncoder(enc_layer(), 1) if with_thought else None  # 思维卡
        self.head = nn.Linear(D, n)                               # 输出卡

    def forward(self, x):
        h = self.emb(x) + self.pe[: x.size(1)].unsqueeze(0)
        h = self.inp(h)
        if self.th is not None:
            h = self.th(h)
        return self.head(h[:, 0])


# ---- 指标 ----
def metrics(pred, tgt):
    pred = pred.float()
    tgt = tgt.float()
    acc = (pred == tgt).float().mean().item()
    tp = (pred * tgt).sum(0)
    fp = (pred * (1 - tgt)).sum(0)
    fn = ((1 - pred) * tgt).sum(0)
    den = 2 * tp + fp + fn
    f1 = torch.where(den > 0, 2 * tp / den.clamp(min=1e-9), torch.zeros_like(den))
    return round(acc, 4), round(f1.mean().item(), 4)


def floors(n, groups, Ks):
    tgt = torch.cat([groups[k][1] for k in Ks])
    ks = torch.tensor([k for k in Ks for _ in range(groups[k][1].shape[0])])
    thr = n // 2
    out = {}
    p0 = torch.zeros_like(tgt)
    out["全部不可达"] = dict(zip(("acc", "macroF1"), metrics(p0, tgt)))
    p1 = torch.ones_like(tgt)
    out["全部可达"] = dict(zip(("acc", "macroF1"), metrics(p1, tgt)))
    pr = torch.zeros_like(tgt)
    pr[:, 0] = 1.0
    pr[ks > thr] = 1.0
    out[f"规则k<={thr}仅起点否则全可达"] = dict(zip(("acc", "macroF1"), metrics(pr, tgt)))
    return out


# ---- 训练 / 评估 ----
def make_opt(m):
    return torch.optim.AdamW(m.parameters(), lr=LR)


def train_steps(m, opt, groups, Ks, steps, start_i=0):
    lossf = nn.BCEWithLogitsLoss()
    ks = list(Ks)
    for i in range(start_i, start_i + steps):
        k = ks[i % len(ks)]
        X, Y = groups[k]
        idx = torch.randint(0, X.shape[0], (BATCH,))
        opt.zero_grad(set_to_none=True)
        loss = lossf(m(X[idx]), Y[idx])
        loss.backward()
        opt.step()


def calibrate(n, groups, Ks):
    torch.manual_seed(42)
    m = Net(n, False)
    opt = make_opt(m)
    m.train()
    train_steps(m, opt, groups, Ks, WARMUP)
    t0 = time.time()
    train_steps(m, opt, groups, Ks, CALIB_STEPS, start_i=WARMUP)
    return (time.time() - t0) / CALIB_STEPS


def evaluate(m, groups, Ks):
    m.eval()
    P, T = [], []
    with torch.no_grad():
        for k in Ks:
            X, Y = groups[k]
            for i in range(0, X.shape[0], BATCH):
                P.append(torch.sigmoid(m(X[i : i + BATCH])))
                T.append(Y[i : i + BATCH])
    return torch.cat(P) >= 0.5, torch.cat(T)


def run_one(n, tr, te, Ks, arm, seed, steps):
    torch.manual_seed(seed)
    random.seed(seed)
    m = Net(n, arm == "B")
    opt = make_opt(m)
    m.train()
    t0 = time.time()
    train_steps(m, opt, tr, Ks, steps)
    train_s = time.time() - t0
    P, T = evaluate(m, te, Ks)
    acc, f1 = metrics(P, T)
    return {
        "arm": arm, "seed": seed, "acc": acc, "macroF1": f1,
        "steps": steps, "train_s": round(train_s, 1),
    }


def main():
    t_start = time.time()
    results = {"floors": {}, "calib": {}, "runs": [], "skipped": []}
    for name, n, Ks in DIFFS:
        if time.time() - t_start > MAX_WALL_S:
            results["skipped"].append(name)
            log(f"!! 全局超时, 跳过 {name}")
            break
        log(f"== {name}: n={n}, k={Ks} ==")
        tr = build_groups(n, Ks, split_counts(N_TRAIN, Ks), seed=1000 + n)
        te = build_groups(n, Ks, split_counts(N_TEST, Ks), seed=2000 + n)
        fl = floors(n, te, Ks)
        results["floors"][name] = fl
        log(f"  地板: {json.dumps(fl, ensure_ascii=False)}")
        t_step = calibrate(n, tr, Ks)
        steps = int(TRAIN_BUDGET_S / t_step)
        steps = max(MIN_STEPS, min(MAX_STEPS, steps))
        results["calib"][name] = {"s_per_step": round(t_step, 4), "steps": steps}
        log(f"  校准: {t_step:.4f} s/step -> steps={steps}")
        for arm in ("A0", "B"):
            for seed in SEEDS:
                est = (steps + 15) * t_step + 5
                if time.time() + est - t_start > MAX_WALL_S:
                    results["skipped"].append(f"{name}/{arm}/{seed}")
                    log(f"  !! 时间不足, 跳过 {name}/{arm}/{seed}")
                    continue
                r = run_one(n, tr, te, Ks, arm, seed, steps)
                r["diff"] = name
                results["runs"].append(r)
                log(f"  run {name} {arm} seed={seed}: acc={r['acc']} "
                    f"macroF1={r['macroF1']} steps={steps} train={r['train_s']}s")
    results["total_wall_s"] = round(time.time() - t_start, 1)
    # 派生: 每档 A0/B 均值 + 配对 Δ
    summ = {}
    for name, _, _ in DIFFS:
        row = {}
        for arm in ("A0", "B"):
            xs = [r for r in results["runs"] if r["diff"] == name and r["arm"] == arm]
            if xs:
                row[arm] = {
                    "acc": round(sum(x["acc"] for x in xs) / len(xs), 4),
                    "macroF1": round(sum(x["macroF1"] for x in xs) / len(xs), 4),
                }
        if "A0" in row and "B" in row:
            ds = []
            for seed in SEEDS:
                a = [r for r in results["runs"] if r["diff"] == name
                     and r["arm"] == "A0" and r["seed"] == seed]
                b = [r for r in results["runs"] if r["diff"] == name
                     and r["arm"] == "B" and r["seed"] == seed]
                if a and b:
                    ds.append({"seed": seed, "d_acc": round(b[0]["acc"] - a[0]["acc"], 4),
                               "d_f1": round(b[0]["macroF1"] - a[0]["macroF1"], 4)})
            row["paired"] = ds
            if len(ds) == 2:
                for key in ("d_acc", "d_f1"):
                    v = [d[key] for d in ds]
                    mean = sum(v) / 2
                    se = (abs(v[0] - v[1]) / math.sqrt(2)) if len(v) == 2 else 0.0
                    row["delta_" + key] = {"mean": round(mean, 4), "se": round(se, 4)}
        summ[name] = row
    results["summary"] = summ
    log("=== SUMMARY JSON ===")
    log(json.dumps(results, ensure_ascii=False, indent=1))
    log("=== END ===")


if __name__ == "__main__":
    main()
