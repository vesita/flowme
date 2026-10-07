#!/usr/bin/env python3
"""stage6 —— 8 节点有向图可达性，唯一变量 = 步数，看 B-A0 是否随步数单调上升。

两臂:
  A0 = out( in(x) )            （思维卡不存在）
  B  = out( think( in(x) ) )
扫步数: 500 / 1000 / 2000 / 4000（每 (arm,seed) 训到最大步数，途中在各点评测）。
其余全固定: d=128, V=10, FF=512(4d, 两处 Enc 一致), AdamW lr=1e-3, batch=64,
4000 train / 1000 test, seed {42,43}, CPU。
用法:  python meme/stage6.py        （全量）
       python meme/stage6.py probe  （先跑 100 步测速）
"""
import math
import random
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ---- 全固定超参 ----
D = 128
V = 10                      # 节点 0-7 + 分隔符(8) + 起点标记(9)
SEP, START = 8, 9
NH = 4
FF = 512                    # 4d，两处 Enc 统一
NOUT = 8
MAXLEN = 2 + 3 * 16         # k=16 最长: [START,start] + k*[SEP,src,dst]
LR = 1e-3
BATCH = 64
NTRAIN = 4000
NTEST = 1000
KS = (4, 8, 16)
STEPS = (500, 1000, 2000, 4000)
SEEDS = (42, 43)


def sinusoidal(max_len, d):
    pe = torch.zeros(max_len, d)
    pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def gen_split(seed, n):
    """随机 8 节点有向图，k∈{4,8,16}，保证至少一条长度>=3 的链（4 个不同点的 3 条边）。"""
    rng = random.Random(seed)
    toks, ys, starts, ks, lens = [], [], [], [], []
    for _ in range(n):
        k = rng.choice(KS)
        chain = rng.sample(range(8), 4)
        edges = [(chain[0], chain[1]), (chain[1], chain[2]), (chain[2], chain[3])]
        seen = set(edges)
        while len(edges) < k:
            s, d = rng.randrange(8), rng.randrange(8)
            if s == d:
                continue
            e = (s, d)
            if e in seen:
                continue
            seen.add(e)
            edges.append(e)
        rng.shuffle(edges)
        start = rng.randrange(8)
        adj = [[] for _ in range(8)]
        for s, d in edges:
            adj[s].append(d)
        reach = [False] * 8
        reach[start] = True
        stack = [start]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if not reach[v]:
                    reach[v] = True
                    stack.append(v)
        seq = [START, start]
        for s, d in edges:
            seq += [SEP, s, d]
        seq += [SEP] * (MAXLEN - len(seq))
        toks.append(seq)
        ys.append([1.0 if r else 0.0 for r in reach])
        starts.append(start)
        ks.append(k)
        lens.append(2 + 3 * k)
    pad = torch.ones(len(toks), MAXLEN, dtype=torch.bool)
    for i, L in enumerate(lens):
        pad[i, :L] = False
    return (torch.tensor(toks, dtype=torch.long), torch.tensor(ys, dtype=torch.float32),
            pad, starts, ks)


class Card(nn.Module):
    def __init__(self, with_think):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.register_buffer("pe", sinusoidal(MAXLEN, D))

        def layer():
            return nn.TransformerEncoderLayer(
                d_model=D, nhead=NH, dim_feedforward=FF,
                dropout=0.1, activation="relu", batch_first=True)

        self.enc = nn.TransformerEncoder(layer(), num_layers=1)
        self.think = nn.TransformerEncoder(layer(), num_layers=1) if with_think else None
        self.head = nn.Linear(D, NOUT)

    def forward(self, tok, padmask):
        h = self.emb(tok) + self.pe.unsqueeze(0)
        h = self.enc(h, src_key_padding_mask=padmask)
        if self.think is not None:
            h = self.think(h, src_key_padding_mask=padmask)
        return self.head(h[:, 0])          # 起点标记位


def build(seed, arm):
    """两臂共享参数同 seed 同初始化（B 复用 A0 的全部共享权重）。"""
    torch.manual_seed(seed)
    base_model = Card(with_think=False)
    base = {k: v.clone() for k, v in base_model.state_dict().items()}
    torch.manual_seed(seed)
    model = Card(with_think=(arm == "B"))
    sd = model.state_dict()
    for k, v in base.items():
        sd[k] = v
    model.load_state_dict(sd)
    return model


def metrics(logits, y):
    p = (logits.sigmoid() >= 0.5)
    yb = y >= 0.5
    acc = (p == yb).float().mean().item()
    f1s = []
    for c in range(NOUT):
        yt, yp = yb[:, c], p[:, c]
        tp = (yt & yp).sum().item()
        fp = (~yt & yp).sum().item()
        fn = (yt & ~yp).sum().item()
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return acc, sum(f1s) / NOUT


def eval_model(model, x, pad, y):
    model.eval()
    with torch.no_grad():
        logits = model(x, pad)
    return metrics(logits, y)


def floors(y, starts, ks):
    n = y.shape[0]
    yb = y >= 0.5
    out = {}
    pred0 = torch.zeros_like(yb)
    pred1 = torch.ones_like(yb)
    predh = torch.zeros_like(yb)
    for i in range(n):
        if ks[i] > 4:                       # k > n/2 -> 全可达
            predh[i] = True
        else:                               # k <= n/2 -> 仅起点
            predh[i, starts[i]] = True

    def score(pred):
        acc = (pred == yb).float().mean().item()
        f1s = []
        for c in range(NOUT):
            yt, yp = yb[:, c], pred[:, c]
            tp = (yt & yp).sum().item()
            fp = (~yt & yp).sum().item()
            fn = (yt & ~yp).sum().item()
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
        return acc, sum(f1s) / NOUT

    out["全不可达"] = score(pred0)
    out["全可达"] = score(pred1)
    out["k<=n/2:仅起点,否则全可达"] = score(predh)
    pos = yb.float().mean().item()
    out["_正例占比"] = (pos, pos)
    return out


def run_one(arm, seed, data, eval_steps, max_step, tag=""):
    xtr, ytr, padtr, xte, padte, yte = data
    model = build(seed, arm)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(xtr, ytr, padtr), batch_size=BATCH, shuffle=True,
                        generator=g, num_workers=0)
    it = iter(loader)
    torch.manual_seed(seed)                 # dropout 流也按 seed 配对
    results = {}
    t0 = time.time()
    step = 0
    while step < max_step:
        try:
            xb, yb, pb = next(it)
        except StopIteration:
            it = iter(loader)
            xb, yb, pb = next(it)
        model.train()
        opt.zero_grad(set_to_none=True)
        loss = lossf(model(xb, pb), yb)
        loss.backward()
        opt.step()
        step += 1
        if step in eval_steps:
            acc, f1 = eval_model(model, xte, padte, yte)
            results[step] = (acc, f1, time.time() - t0)
            print(f"  {tag} arm={arm} seed={seed} step={step:5d} "
                  f"acc={acc*100:.2f} f1={f1*100:.2f} "
                  f"({time.time()-t0:.1f}s)", flush=True)
    return results


def worker(job):
    """一个 (arm, seed) = 一个独立进程；数据/初始化都在进程内按 seed 复现。"""
    arm, seed = job
    torch.set_num_threads(3)
    xtr, ytr, padtr, _, _ = gen_split(seed, NTRAIN)
    xte, yte, padte, _, _ = gen_split(12345, NTEST)
    return (arm, seed), run_one(arm, seed, (xtr, ytr, padtr, xte, padte, yte),
                                STEPS, STEPS[-1])


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"
    eval_steps = STEPS
    max_step = STEPS[-1]
    if mode == "probe":
        eval_steps = (100,)
        max_step = 100

    t_start = time.time()
    xte, yte, padte, starts_te, ks_te = gen_split(12345, NTEST)   # test 固定,跨 seed/臂可比

    print("== 地板（test 1000，8 类 macro-F1）==", flush=True)
    fl = floors(yte, starts_te, ks_te)
    for name, (a, f) in fl.items():
        if name.startswith("_"):
            continue
        print(f"  {name:28s} acc={a*100:.2f}  macroF1={f*100:.2f}", flush=True)
    print(f"  (正例占比 {fl['_正例占比'][0]*100:.1f}%)", flush=True)

    if mode == "probe":
        xtr, ytr, padtr, _, _ = gen_split(42, NTRAIN)
        tt = time.time()
        run_one("B", 42, (xtr, ytr, padtr, xte, padte, yte), (100,), 100, "probe")
        dt = time.time() - tt
        proj = dt / 100 * max_step * len(SEEDS) * 2
        print(f"probe: 100 步 {dt:.1f}s -> 全量 4run*{max_step}步 预计 {proj/60:.1f} 分钟",
              flush=True)
        return

    all_res = {}
    jobs = [(arm, seed) for seed in SEEDS for arm in ("A0", "B")]
    import concurrent.futures
    import multiprocessing as mp
    # spawn: fork + torch 线程会在 exec 前挂死（实测 4 子进程 0% CPU 全卡）
    with concurrent.futures.ProcessPoolExecutor(
            max_workers=len(jobs), mp_context=mp.get_context("spawn")) as ex:
        for key, res in ex.map(worker, jobs):
            all_res[key] = res

    print("\n== 扫步数主表 (acc% / macroF1%) ==", flush=True)
    print(f"{'步数':>6} {'arm':>4} {'seed':>4} {'acc':>7} {'F1':>7}", flush=True)
    for s in STEPS:
        for seed in SEEDS:
            for arm in ("A0", "B"):
                a, f, _ = all_res[(arm, seed)][s]
                print(f"{s:>6} {arm:>4} {seed:>4} {a*100:>7.2f} {f*100:>7.2f}",
                      flush=True)

    print("\n== ★ B - A0 (配对, 2 seed) ==", flush=True)
    print(f"{'步数':>6} {'dAcc/seed':>18} {'mean+-SE':>14} {'dF1/seed':>18} {'mean+-SE':>14} 同号",
          flush=True)
    trend = []
    for s in STEPS:
        da, df = [], []
        for seed in SEEDS:
            a0, f0, _ = all_res[("A0", seed)][s]
            b, fb, _ = all_res[("B", seed)][s]
            da.append((b - a0) * 100)
            df.append((fb - f0) * 100)
        def ms(v):
            m = sum(v) / len(v)
            if len(v) > 1:
                var = sum((x - m) ** 2 for x in v) / (len(v) - 1)
                se = math.sqrt(var / len(v))
            else:
                se = float("nan")
            return m, se
        ma, sea = ms(da)
        mf, sef = ms(df)
        same = "Y" if (da[0] * da[1] > 0 and df[0] * df[1] > 0) else "N"
        trend.append(ma)
        print(f"{s:>6} [{da[0]:+6.2f},{da[1]:+6.2f}] {ma:+6.2f}+-{sea:4.2f} "
              f"[{df[0]:+6.2f},{df[1]:+6.2f}] {mf:+6.2f}+-{sef:4.2f}  {same}",
              flush=True)
    mono = all(trend[i] <= trend[i + 1] + 1e-9 for i in range(len(trend) - 1))
    print(f"\n趋势: dAcc(B-A0) = {['%+.2f' % v for v in trend]}  "
          f"随步数单调上升 = {'是' if mono else '否'}", flush=True)

    wall = time.time() - t_start
    print(f"\n墙钟: {wall/60:.1f} 分钟（每 run 实际步数 {max_step}，共 "
          f"{len(SEEDS)*2} run）", flush=True)


if __name__ == "__main__":
    main()
