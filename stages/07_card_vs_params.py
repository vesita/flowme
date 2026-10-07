"""P32: "卡" vs "等量参数" —— 三臂参数对齐对照（A0 / A0-wide / B）。

只回答一个问题：P31 里 B-A0 的 +1pp 增益，是"思维卡这种结构"带来的，
还是"多了那约 20 万参数"带来的。A0-wide 用加宽的输出卡把总参数量对齐到 B。
"""
import json
import math
import multiprocessing as mp
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 固定设置（照抄 P31/P32 规格） ----------------
D = 128
V = 10          # 节点 0-7 + 分隔符(8) + 起点标记(9)
N_NODES = 8
SEP = 8
START = 9
HEADS = 4
FF = 512        # 4d
N_EDGES_SET = (4, 8, 16)
STEPS = 4000
LR = 1e-3
BATCH = 64
N_TRAIN = 4000
N_TEST = 1000
SEEDS = (42, 43)
MAX_LEN = 2 + 3 * max(N_EDGES_SET)   # 50
ARMS = ("A0", "A0-wide", "B")


# ---------------- 数据 ----------------
def gen_instance(rng, k):
    """随机有向图：含至少一条长度>=3 的链（3 条边、4 个不同节点）。"""
    chain = rng.choice(N_NODES, size=4, replace=False)
    edges = set()
    for i in range(3):
        edges.add((int(chain[i]), int(chain[i + 1])))
    while len(edges) < k:
        u = int(rng.integers(N_NODES))
        v = int(rng.integers(N_NODES))
        if u != v:
            edges.add((u, v))
    edges = list(edges)
    rng.shuffle(edges)
    s = int(rng.integers(N_NODES))
    return edges, s


def reach_set(edges, s):
    adj = [[] for _ in range(N_NODES)]
    for u, v in edges:
        adj[u].append(v)
    seen = {s}
    stack = [s]
    while stack:
        u = stack.pop()
        for w in adj[u]:
            if w not in seen:
                seen.add(w)
                stack.append(w)
    y = np.zeros(N_NODES, dtype=np.float32)
    for v in seen:
        y[v] = 1.0
    return y


def enc_tokens(edges, s):
    toks = [START, s]
    for u, v in edges:
        toks += [SEP, u, v]
    assert len(toks) <= MAX_LEN
    pad = MAX_LEN - len(toks)
    mask = [False] * len(toks) + [True] * pad      # True = pad
    toks = toks + [SEP] * pad
    return toks, mask


def gen_dataset(n, seed):
    rng = np.random.default_rng(seed)
    xs, ms, ys, ks = [], [], [], []
    for _ in range(n):
        k = int(rng.choice(N_EDGES_SET))
        edges, s = gen_instance(rng, k)
        toks, mask = enc_tokens(edges, s)
        xs.append(toks)
        ms.append(mask)
        ys.append(reach_set(edges, s))
        ks.append(k)
    return (torch.tensor(xs, dtype=torch.long),
            torch.tensor(ms, dtype=torch.bool),
            torch.tensor(np.stack(ys), dtype=torch.float32),
            np.array(ks))


# ---------------- 模型 ----------------
def sinusoid_pe(max_len, d):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


class EncLayer(nn.Module):
    def __init__(self, d=D, heads=HEADS, ff=FF):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.proj = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, ff)
        self.fc2 = nn.Linear(ff, d)
        self.heads = heads

    def forward(self, x, key_mask):
        h = self.ln1(x)
        b, n, _ = h.shape
        q, k, v = self.qkv(h).chunk(3, dim=-1)
        hd = D // self.heads
        q = q.view(b, n, self.heads, hd).transpose(1, 2)
        k = k.view(b, n, self.heads, hd).transpose(1, 2)
        v = v.view(b, n, self.heads, hd).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(hd)
        if key_mask is not None:
            att = att.masked_fill(key_mask[:, None, None, :], float("-inf"))
        att = att.softmax(dim=-1)
        att = torch.nan_to_num(att, nan=0.0)
        o = (att @ v).transpose(1, 2).reshape(b, n, D)
        x = x + self.proj(o)
        h2 = self.ln2(x)
        x = x + self.fc2(F.gelu(self.fc1(h2)))
        return x


class Model(nn.Module):
    def __init__(self, arm, wide_w):
        super().__init__()
        assert arm in ARMS
        self.arm = arm
        self.emb = nn.Embedding(V, D)
        self.register_buffer("pe", sinusoid_pe(MAX_LEN, D), persistent=False)
        self.in_card = EncLayer()
        if arm == "B":
            self.think_card = EncLayer()
        if arm == "A0-wide":
            self.out_card = nn.Sequential(
                nn.Linear(D, wide_w), nn.GELU(), nn.Linear(wide_w, N_NODES))
        else:
            self.out_card = nn.Linear(D, N_NODES)

    def forward(self, toks, padmask):
        x = self.emb(toks) + self.pe[:toks.size(1)]
        x = self.in_card(x, padmask)
        if self.arm == "B":
            x = self.think_card(x, padmask)
        pooled = x[:, 0]          # 起点标记位 readout
        return self.out_card(pooled)


def count_params(m):
    return sum(p.numel() for p in m.parameters())


def solve_wide_w():
    """解 137W+8 = (B 总参 - A0 公共部分) => A0-wide 总参 == B 总参。"""
    ref_b = count_params(Model("B", 0))
    ref_a0 = count_params(Model("A0", 0))
    # A0 的输出卡 = Linear(128,8)；A0-wide 输出卡 = 137W+8
    a0_out = D * N_NODES + N_NODES
    base = ref_a0 - a0_out
    want = ref_b - base
    w = int(round((want - N_NODES) / (D + 1 + N_NODES)))
    return w, ref_a0, ref_b


# ---------------- 训练 / 评估 ----------------
def evaluate(model, xs, ms, ys):
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, xs.size(0), 512):
            logits = model(xs[i:i + 512], ms[i:i + 512])
            preds.append((torch.sigmoid(logits) >= 0.5).float())
    p = torch.cat(preds).cpu().numpy()
    t = ys.cpu().numpy()
    acc = float((p == t).mean())
    f1s = []
    for j in range(N_NODES):
        yt, yp = t[:, j], p[:, j]
        tp = float((yt * yp).sum())
        fp = float(((1 - yt) * yp).sum())
        fn = float((yt * (1 - yp)).sum())
        if tp == 0:
            f1s.append(1.0 if (fp == 0 and fn == 0) else 0.0)
        else:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return acc, float(np.mean(f1s))


def floor_pred(kind, ks, s_arr):
    n = len(ks)
    p = np.zeros((n, N_NODES), dtype=np.float32)
    if kind == "all_zero":
        return p
    if kind == "all_one":
        return p + 1.0
    for i, k in enumerate(ks):
        if k <= N_NODES // 2:
            p[i, s_arr[i]] = 1.0
        else:
            p[i, :] = 1.0
    return p


def floor_metrics(pred, t):
    acc = float((pred == t).mean())
    f1s = []
    for j in range(N_NODES):
        yt, yp = t[:, j], pred[:, j]
        tp = float((yt * yp).sum())
        fp = float(((1 - yt) * yp).sum())
        fn = float((yt * (1 - yp)).sum())
        if tp == 0:
            f1s.append(1.0 if (fp == 0 and fn == 0) else 0.0)
        else:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    return acc, float(np.mean(f1s))


def run_one(arm, seed, wide_w, q):
    t0 = time.time()
    torch.set_num_threads(max(1, (os.cpu_count() or 6) // 6))
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    xs, ms, ys, ks = gen_dataset(N_TRAIN, seed + 100000)
    xt, mt, yt, kt = gen_dataset(N_TEST, seed + 200000)
    xs, ms, ys = xs.to(dev), ms.to(dev), ys.to(dev)
    xt, mt, yt = xt.to(dev), mt.to(dev), yt.to(dev)
    model = Model(arm, wide_w).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed)
    model.train()
    step = 0
    while step < STEPS:
        idx = torch.randperm(N_TRAIN, generator=g)[:BATCH].to(dev)
        logits = model(xs[idx], ms[idx])
        loss = F.binary_cross_entropy_with_logits(logits, ys[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        step += 1
    if dev == "cuda":
        torch.cuda.synchronize()
    acc, f1 = evaluate(model, xt, mt, yt)
    n_par = count_params(model)
    if dev == "cuda":
        del xs, ms, ys, xt, mt, yt, model
        torch.cuda.empty_cache()
    return {"arm": arm, "seed": seed, "acc": acc, "f1": f1,
            "params": n_par, "dev": dev, "wall": time.time() - t0,
            "steps": step}


def main():
    t_start = time.time()
    w, p_a0, p_b = solve_wide_w()
    p_wide = count_params(Model("A0-wide", w))
    gap = abs(p_wide - p_b) / p_b * 100.0
    print("P0 params: A0=%d A0-wide(W=%d)=%d B=%d | wide-vs-B gap=%.3f%%"
          % (p_a0, w, p_wide, p_b, gap), flush=True)

    # 地板
    _, _, y_test, k_test = gen_dataset(N_TEST, 42 + 200000)
    xs_f, ms_f, ys_f, _ = gen_dataset(N_TEST, 42 + 200000)
    s_arr = xs_f[:, 1].numpy()
    t_np = ys_f.numpy()
    floors = {}
    for kind in ("all_zero", "all_one", "rule"):
        p = floor_pred(kind, k_test, s_arr)
        a, f = floor_metrics(p, t_np)
        floors[kind] = (a, f)
        print("FLOOR %-9s acc=%.4f macroF1=%.4f" % (kind, a, f), flush=True)

    ctx = mp.get_context("spawn")
    res = []
    if torch.cuda.is_available():
        # 单 GPU：串行跑 6 个 run，避开 spawn+ROCm 的多进程初始化开销
        for arm in ARMS:
            for seed in SEEDS:
                res.append(run_one(arm, seed, w, None))
    else:
        q = ctx.Queue()
        procs = []
        for arm in ARMS:
            for seed in SEEDS:
                p = ctx.Process(target=run_one, args=(arm, seed, w, q))
                p.start()
                procs.append(p)
        for _ in procs:
            res.append(q.get())
        for p in procs:
            p.join()
    res.sort(key=lambda r: (ARMS.index(r["arm"]), r["seed"]))

    print("\n== 三臂 x 2 seed ==")
    for r in res:
        print("%-8s seed=%d acc=%.4f macroF1=%.4f params=%d steps=%d dev=%s wall=%.1fs"
              % (r["arm"], r["seed"], r["acc"], r["f1"], r["params"],
                 r["steps"], r["dev"], r["wall"]), flush=True)

    def get(arm, seed):
        return next(r for r in res if r["arm"] == arm and r["seed"] == seed)

    print("\n== P1 主判据：A0-wide - B（配对，按 seed）==")
    ds = []
    for seed in SEEDS:
        d_acc = get("A0-wide", seed)["acc"] - get("B", seed)["acc"]
        d_f1 = get("A0-wide", seed)["f1"] - get("B", seed)["f1"]
        ds.append((d_acc, d_f1))
        print("seed=%d d_acc=%+.4f d_macroF1=%+.4f" % (seed, d_acc, d_f1))
    d_acc = np.array([x[0] for x in ds])
    d_f1 = np.array([x[1] for x in ds])
    se_acc = float(d_acc.std(ddof=1) / math.sqrt(len(ds)))
    se_f1 = float(d_f1.std(ddof=1) / math.sqrt(len(ds)))
    print("d_acc mean=%+.4f SE=%.4f same_sign=%s | d_macroF1 mean=%+.4f SE=%.4f same_sign=%s"
          % (d_acc.mean(), se_acc, bool(np.all(d_acc > 0) or np.all(d_acc < 0)),
             d_f1.mean(), se_f1, bool(np.all(d_f1 > 0) or np.all(d_f1 < 0))))
    m = float(d_acc.mean())
    if abs(m) < 2 * se_acc:
        verdict = '|d|=%.4f < 2SE=%.4f => "卡"只是参数（未测出结构差异）' % (abs(m), 2 * se_acc)
    elif m < 0:
        verdict = 'A0-wide 显著低于 B => 卡有结构价值'
    else:
        verdict = 'A0-wide 显著高于 B => 卡反而不如等量参数'
    print("VERDICT: " + verdict)

    print("\n== P2：增益拆解 ==")
    for seed in SEEDS:
        b_a0 = get("B", seed)["acc"] - get("A0", seed)["acc"]
        w_a0 = get("A0-wide", seed)["acc"] - get("A0", seed)["acc"]
        print("seed=%d B-A0=%+.4f  A0-wide-A0=%+.4f" % (seed, b_a0, w_a0))
    ba = np.array([get("B", s)["acc"] - get("A0", s)["acc"] for s in SEEDS])
    wa = np.array([get("A0-wide", s)["acc"] - get("A0", s)["acc"] for s in SEEDS])
    print("B-A0 mean=%+.4f | A0-wide-A0 mean=%+.4f (纯加参数买到的)" % (ba.mean(), wa.mean()))

    total = time.time() - t_start
    print("\nwall_total=%.1fs (%.2f min) arms=%s seeds=%s steps=%d dev=%s"
          % (total, total / 60.0, ARMS, SEEDS, STEPS, res[0]["dev"]), flush=True)

    out = {"params": {"A0": p_a0, "A0-wide": p_wide, "B": p_b, "wide_w": w,
                      "gap_pct": gap},
           "floors": floors, "runs": res,
           "p1": {"d_acc": d_acc.tolist(), "se_acc": se_acc,
                  "d_f1": d_f1.tolist(), "se_f1": se_f1},
           "p2": {"B-A0": ba.tolist(), "A0-wide-A0": wa.tolist()},
           "wall": total, "steps": STEPS}
    os.makedirs("logs", exist_ok=True)
    with open("logs/stage7.json", "w") as f:
        json.dump(out, f, indent=1, default=float)
    print("log: logs/stage7.json", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
