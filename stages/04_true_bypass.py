# -*- coding: utf-8 -*-
"""P29 极小版：真旁路臂 A0 + A1/B/C，只做复杂任务（有向图可达性）。

四臂唯一变量 = 输出卡吃谁：
  A0: out = head(pool(m))                     思维卡完全不存在（真旁路基线）
  A1: out = head(pool(m + alpha * thought(m))) alpha 可学习（并行双路，复现上一轮 A）
  B : out = head(pool(thought(m)))             强制串联
  C : 同 B，思维卡内部瓶颈 Linear(128,32)->GELU->Linear(32,128)
P4: 对训练好的 B/C，推理时跳过思维卡（= 输出卡吃 m）看 acc/F1 掉落。
常数全部内联；只写本文件与 logs/。
"""
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

# ---------------- 内联常数 ----------------
D = 128
V = 10                      # 节点 0-7 + 分隔符(8) + 起点标记(9)
HEADS = 4
FF = 512                    # TransformerEncoder FFN = 4*d
NLAYERS = 1
NN = 8                      # 节点数
MAXLEN = 2 + 3 * 16         # [9,s] + 每条边 (u,v,sep)

STEPS = 1500
BS = 64
N_TRAIN = 8000
N_TEST = 2000
LR = 1e-3
SEEDS = [42, 43]
ARMS = ["A0", "A1", "B", "C"]
DATA_SEED = 20261007
BUDGET_SEC = 330.0          # 单跑预算（6 分钟）
PROBE_N = 60
EVAL_SEC_EST = 40.0

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(os.path.dirname(HERE), "logs")
RESULT_PATH = os.path.join(LOG_DIR, "stage4_results.json")


# ---------------- 数据：随机有向图可达性（8 节点，k∈{4,8,16}） ----------------
def gen_data(n, rng):
    X = np.full((n, MAXLEN), 8, dtype=np.int64)
    M = np.ones((n, MAXLEN), dtype=bool)          # True = padding
    Y = np.zeros((n, NN), dtype=np.float32)
    S = np.zeros(n, dtype=np.int64)
    K = np.zeros(n, dtype=np.int64)
    for i in range(n):
        k = int(rng.choice([4, 8, 16]))
        pairs = set()
        while len(pairs) < k:
            u = int(rng.integers(0, NN))
            v = int(rng.integers(0, NN))
            if u != v:
                pairs.add((u, v))
        s = int(rng.integers(0, NN))
        adj = [[] for _ in range(NN)]
        for u, v in pairs:
            adj[u].append(v)
        seen = {s}
        stack = [s]
        while stack:
            u = stack.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    stack.append(v)
        for v in seen:
            Y[i, v] = 1.0
        toks = [9, s]
        for u, v in pairs:
            toks += [u, v, 8]
        L = len(toks)
        X[i, :L] = toks
        M[i, :L] = False
        S[i] = s
        K[i] = k
    return (torch.from_numpy(X), torch.from_numpy(M), torch.from_numpy(Y),
            torch.from_numpy(S), torch.from_numpy(K))


# ---------------- 模型 ----------------
class Net(nn.Module):
    def __init__(self, arm, seed):
        super().__init__()
        self.arm = arm
        torch.manual_seed(seed + 1)
        self.emb = nn.Embedding(V, D)
        pe = torch.zeros(1, MAXLEN, D)
        pos = torch.arange(MAXLEN, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

        def layer():
            return nn.TransformerEncoderLayer(
                d_model=D, nhead=HEADS, dim_feedforward=FF,
                dropout=0.1, batch_first=True)

        torch.manual_seed(seed + 2)
        self.inp = nn.TransformerEncoder(layer(), num_layers=NLAYERS)
        if arm != "A0":
            torch.manual_seed(seed + 3)
            self.tho = nn.TransformerEncoder(layer(), num_layers=NLAYERS)
        if arm == "A1":
            self.alpha = nn.Parameter(torch.tensor(1.0))
        if arm == "C":
            torch.manual_seed(seed + 5)
            self.bn = nn.Sequential(nn.Linear(D, 32), nn.GELU(), nn.Linear(32, D))
        torch.manual_seed(seed + 4)
        self.head = nn.Linear(D, 8)

    @staticmethod
    def _pool(x, mask):
        keep = (~mask).float().unsqueeze(-1)
        return (x * keep).sum(1) / keep.sum(1).clamp(min=1.0)

    def forward(self, tok, mask, mode="use"):
        x = self.emb(tok) + self.pe
        m = self.inp(x, src_key_padding_mask=mask)
        if self.arm == "A0" or mode == "skip":
            rep = self._pool(m, mask)
        else:
            t = self.tho(m, src_key_padding_mask=mask)
            if self.arm == "A1":
                u = m + self.alpha * t
            elif self.arm == "B":
                u = t
            else:
                u = self.bn(t)
            rep = self._pool(u, mask)
        return self.head(rep)


# ---------------- 训练 / 评估 ----------------
def train_arm(arm, seed, steps, train):
    X, M, Y, S, K = train
    model = Net(arm, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    gen = torch.Generator().manual_seed(seed * 10007 + 11)
    n = X.shape[0]
    perm = torch.randperm(n, generator=gen)
    pos = 0
    step = 0
    t0 = time.time()
    model.train()
    while step < steps:
        if pos + BS > n:
            perm = torch.randperm(n, generator=gen)
            pos = 0
        b = perm[pos:pos + BS]
        pos += BS
        loss = lossf(model(X[b], M[b]), Y[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 300 == 0:
            print(f"  [{arm} seed{seed}] step {step}/{steps} loss {loss.item():.4f} "
                  f"elapsed {time.time() - t0:.0f}s", flush=True)
    return model


def node_f1(P, Y):
    out = []
    for j in range(NN):
        pj, yj = P[:, j] == 1, Y[:, j] == 1
        tp = int((pj & yj).sum())
        fp = int((pj & ~yj).sum())
        fn = int((~pj & yj).sum())
        den = 2 * tp + fp + fn
        out.append(2.0 * tp / den if den > 0 else 1.0)
    return out


@torch.no_grad()
def evaluate(model, data, mode="use"):
    X, M, Y, S, K = data
    model.eval()
    logits = model(X, M, mode=mode)
    P = (torch.sigmoid(logits) >= 0.5).numpy()
    acc = float((P == Y.numpy()).mean())
    return acc, node_f1(P, Y.numpy())


def fmt_f1(f1):
    return "[" + ",".join(f"{v:.2f}" for v in f1) + "]"


def probe_seconds_per_step(train):
    model = Net("A0", 42)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    lossf = nn.BCEWithLogitsLoss()
    X, M, Y, S, K = train
    gen = torch.Generator().manual_seed(0)
    perm = torch.randperm(X.shape[0], generator=gen)
    t0 = time.time()
    model.train()
    for i in range(PROBE_N):
        b = perm[i * BS:(i + 1) * BS]
        loss = lossf(model(X[b], M[b]), Y[b])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    return (time.time() - t0) / PROBE_N


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    print("=== P29 stage4: 有向图可达性（复杂任务） 四臂 × 2 seed ===", flush=True)
    rng = np.random.default_rng(DATA_SEED)
    train = gen_data(N_TRAIN, rng)
    test = gen_data(N_TEST, rng)
    Xtr, Mtr, Ytr, Str, Ktr = train
    Xte, Mte, Yte, Ste, Kte = test
    print(f"data: train {N_TRAIN} test {N_TEST} maxlen {MAXLEN}", flush=True)

    # ---- 地板 ----
    floors = {}
    p_node = Ytr.numpy().mean(axis=0)
    maj_pred = np.tile((p_node >= 0.5).astype(np.float32), (N_TEST, 1))
    floors["majority"] = (float((maj_pred == Yte.numpy()).mean()),
                          node_f1(maj_pred, Yte.numpy()), p_node.tolist())
    cnt_pred = np.zeros((N_TEST, NN), dtype=np.float32)
    for i in range(N_TEST):
        if Kte[i] <= 4:
            cnt_pred[i, int(Ste[i])] = 1.0
        else:
            cnt_pred[i, :] = 1.0
    floors["count_rule"] = (float((cnt_pred == Yte.numpy()).mean()),
                            node_f1(cnt_pred, Yte.numpy()), None)
    print("=== FLOORS ===", flush=True)
    for name, (acc, f1, extra) in floors.items():
        print(f"{name}: acc {acc:.4f} f1 {fmt_f1(f1)} macroF1 {np.mean(f1):.4f}"
              + (f" train_p_node {[round(v, 3) for v in extra]}" if extra else ""),
              flush=True)

    # ---- 步数自适应（单跑 > 6 分钟 ⇒ 减步数并写明）----
    tps = probe_seconds_per_step(train)
    n_runs = len(ARMS) * len(SEEDS)
    est = tps * STEPS * n_runs + EVAL_SEC_EST
    steps = STEPS
    reduced_note = None
    if est > BUDGET_SEC:
        new_steps = int((BUDGET_SEC - EVAL_SEC_EST) / (tps * n_runs))
        new_steps = max(200, new_steps)
        reduced_note = (f"[STEP REDUCED] {STEPS} -> {new_steps} 步：实测 {tps*1000:.0f} "
                        f"ms/step，原配置预计 {est:.0f}s > {BUDGET_SEC:.0f}s 单跑预算")
        steps = new_steps
        print(reduced_note, flush=True)
    else:
        print(f"[steps kept] {STEPS} 步，实测 {tps*1000:.0f} ms/step，"
              f"预计全程 {est:.0f}s", flush=True)

    # ---- 四臂 × 2 seed ----
    results = {}
    trained = {}
    print("=== ARMS ===", flush=True)
    for arm in ARMS:
        for seed in SEEDS:
            t0 = time.time()
            model = train_arm(arm, seed, steps, train)
            acc, f1 = evaluate(model, test, mode="use")
            alpha = float(model.alpha.item()) if arm == "A1" else None
            results[(arm, seed)] = {"acc": acc, "f1": f1, "alpha": alpha}
            trained[(arm, seed)] = model
            print(f"{arm} seed{seed}: acc {acc:.4f} f1 {fmt_f1(f1)} "
                  f"macroF1 {np.mean(f1):.4f}"
                  + (f" alpha {alpha:.4f}" if alpha is not None else "")
                  + f"  ({time.time()-t0:.0f}s)", flush=True)

    # ---- P4：对 B/C 推理时跳过思维卡 ----
    p4 = {}
    for arm in ["B", "C"]:
        for seed in SEEDS:
            n_acc, n_f1 = results[(arm, seed)]["acc"], results[(arm, seed)]["f1"]
            s_acc, s_f1 = evaluate(trained[(arm, seed)], test, mode="skip")
            p4[(arm, seed)] = (s_acc, s_f1)
            print(f"P4 skip {arm} seed{seed}: normal acc {n_acc:.4f} f1 {np.mean(n_f1):.4f} "
                  f"-> skip acc {s_acc:.4f} f1 {np.mean(s_f1):.4f} "
                  f"delta {s_acc-n_acc:+.4f} (floor majority acc {floors['majority'][0]:.4f})",
                  flush=True)

    # ---- 配对判据 ----
    def paired(x, y, tag):
        print(f"=== {tag}: {x} - {y} ===", flush=True)
        rows = []
        for metric in ["acc", "macro_f1"]:
            ds = []
            for seed in SEEDS:
                if metric == "acc":
                    a, b = results[(x, seed)]["acc"], results[(y, seed)]["acc"]
                else:
                    a = float(np.mean(results[(x, seed)]["f1"]))
                    b = float(np.mean(results[(y, seed)]["f1"]))
                ds.append(a - b)
            se = float(np.std(ds, ddof=1) / math.sqrt(len(ds))) if len(ds) > 1 else 0.0
            same = (ds[0] * ds[1] > 0)
            print(f"{metric}: d {np.mean(ds):+.4f} +- {se:.4f} "
                  f"(seed42 {ds[0]:+.4f}, seed43 {ds[1]:+.4f}) same_sign={same}",
                  flush=True)
            rows.append({"metric": metric, "mean": float(np.mean(ds)), "se": se,
                         "per_seed": [float(v) for v in ds], "same_sign": bool(same)})
        return rows

    p1 = {a: paired(a, "A0", f"P1 {a} vs A0(真旁路)") for a in ["A1", "B", "C"]}
    p2 = paired("B", "A1", "P2 串联 B vs 并行 A1")
    p3 = paired("C", "B", "P3 瓶颈 C vs B")

    # ---- 落盘 ----
    dump = {
        "steps": steps, "steps_original": STEPS, "reduced_note": reduced_note,
        "ms_per_step": tps * 1000.0,
        "floors": {k: {"acc": v[0], "f1": v[1]} for k, v in floors.items()},
        "arms": {f"{a}_{s}": {"acc": v["acc"], "f1": v["f1"], "alpha": v["alpha"]}
                 for (a, s), v in results.items()},
        "p4_skip": {f"{a}_{s}": {"acc": v[0], "f1": v[1]} for (a, s), v in p4.items()},
        "p1": {k: v for k, v in p1.items()}, "p2": p2, "p3": p3,
    }
    with open(RESULT_PATH, "w") as f:
        json.dump(dump, f, indent=1, ensure_ascii=False)
    print(f"JSON -> {RESULT_PATH}", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
