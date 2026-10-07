# -*- coding: utf-8 -*-
"""P28 极小版：四臂（强制路由 + 瓶颈） × 两任务（简单排序 / 复杂可达性）。
常数全部内联；只此一个文件；CPU。"""
import math, time, sys
import torch
import torch.nn as nn
import torch.nn.functional as Fn
# ---------------- 常数 ----------------
D, HEADS, FFN, DMID = 128, 4, 256, 32
LR, BS, N_TRAIN, N_TEST, STEPS = 1e-3, 64, 8000, 2000, 3000
V_S, LEN_S = 64, 16                       # 简单任务：词表 64、长度 16，目标 = 原序(升序)
V_C, NODES, KS = 10, 8, (4, 8, 16)        # 复杂任务：节点 8、边数 {4,8,16}、词表 10(0-7+SEP=8+START=9)
MAXLEN = 2 + 3 * max(KS)                  # 50
ARMS = ("A", "B", "C", "D"); torch.set_num_threads(6)
# ---------------- 数据 ----------------
def gen(task, n, seed):
    g = torch.Generator().manual_seed(seed)
    if task == "S":
        orig = torch.rand(n, V_S, generator=g).argsort(1)[:, :LEN_S]   # 升序 = 原序
        perm = torch.rand(n, LEN_S, generator=g).argsort(1)            # 打乱后作输入
        return orig.gather(1, perm), orig, torch.full((n,), LEN_S, dtype=torch.long)
    ksel = torch.tensor(KS)[torch.randint(0, 3, (n,), generator=g)]
    e = torch.rand(n, 56, generator=g).argsort(1)[:, :max(KS)]         # 56 条非自环有向边
    s = torch.randint(0, NODES, (n,), generator=g)
    x = torch.full((n, MAXLEN), 8, dtype=torch.long)                   # 8 = SEP，序列外即 pad
    y = torch.zeros(n, NODES)
    ln = torch.zeros(n, dtype=torch.long)
    for i in range(n):
        ki, seq, adj = int(ksel[i]), [9, int(s[i])], [[] for _ in range(NODES)]
        for j in range(ki):
            a = int(e[i, j]); u = a // 7; r = a % 7; v = r if r < u else r + 1
            adj[u].append(v); seq += [8, u, v]                         # [SEP, u, v]
        seen, st = {int(s[i])}, [int(s[i])]                             # BFS 传递闭包(可判)
        while st:
            u = st.pop()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v); st.append(v)
        y[i, list(seen)] = 1.0
        x[i, :len(seq)] = torch.tensor(seq); ln[i] = len(seq)
    return x, y, ln
# ---------------- 模型：四臂唯一差异 = 训练路径与中介 ----------------
class Net(nn.Module):
    def __init__(self, arm, task):
        super().__init__()
        self.arm, self.task = arm, task
        self.emb = nn.Embedding(V_S if task == "S" else V_C, D)        # 输入卡: Emb+Pos+1层
        self.pos = nn.Embedding(MAXLEN, D)
        self.enc = nn.TransformerEncoderLayer(D, HEADS, FFN, batch_first=True)
        self.mind = nn.TransformerEncoderLayer(D, HEADS, FFN, batch_first=True)  # 思维卡
        if arm in ("C", "D"):                                          # 瓶颈 128->32->128
            self.dn, self.up = nn.Linear(D, DMID), nn.Linear(DMID, D)
        if arm == "A":
            self.alpha = nn.Parameter(torch.tensor(1.0))               # 可学习旁路系数
        self.head = nn.Linear(D, V_S if task == "S" else NODES)        # 输出卡
    def forward(self, x, ln, skip=False):
        n = x.shape[1]
        pad = torch.arange(n, device=x.device)[None, :] >= ln[:, None]
        p = self.pos(torch.arange(n, device=x.device))[None, :, :]
        m = self.enc(self.emb(x) + p, src_key_padding_mask=pad)        # m = 输入卡
        if self.arm == "A":
            h = m + self.alpha * self.mind(m)                          # A: 允许旁路(alpha->0 即旁路)
        elif skip:                                                     # D: 运行时跳过思维卡
            h = m
        else:
            h = self.mind(m)                                           # B/C/D: 强制经过
            if self.arm in ("C", "D"):
                h = self.up(Fn.gelu(self.dn(h)))                       # 瓶颈块内无残差(残差留在注意力层) => 不能退化成恒等
        if self.task == "S":
            return self.head(h)                                        # 逐位置 64 类
        keep = (torch.arange(n, device=x.device)[None, :] < ln[:, None]).float()
        return self.head((h * keep[..., None]).sum(1) / keep.sum(1)[:, None])   # 8 维二值 logit
# ---------------- 训练 / 评估 ----------------
def train_one(net, opt, x, y, ln, task):
    net.train(); opt.zero_grad()
    out = net(x, ln)
    loss = (Fn.cross_entropy(out.reshape(-1, out.shape[-1]), y.reshape(-1)) if task == "S"
            else Fn.binary_cross_entropy_with_logits(out, y))
    loss.backward(); opt.step()
    return loss.item()
def train_run(task, arm, seed, steps, tr):
    torch.manual_seed(seed * 977 + ARMS.index(arm) * 131 + (0 if task == "S" else 7))
    net, opt = Net(arm, task), None
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    xtr, ytr, lntr = tr
    g = torch.Generator().manual_seed(seed * 31 + (0 if task == "S" else 1))
    order = torch.randperm(len(xtr), generator=g); ptr = 0; tl = 0.0
    for _ in range(steps):
        if ptr + BS > len(order):
            order = torch.randperm(len(xtr), generator=g); ptr = 0
        b = order[ptr:ptr + BS]; ptr += BS
        tl = train_one(net, opt, xtr[b], ytr[b], lntr[b], task)
    return net, tl
@torch.no_grad()
def logits_of(net, te, task, skip):
    x, ln = te[0], te[2]
    net.eval()
    shp = (len(x), LEN_S, V_S) if task == "S" else (len(x), NODES)
    res = torch.zeros(shp)
    for flag in (False, True):
        idx = torch.nonzero(skip == flag).squeeze(1)
        for i in range(0, len(idx), 256):
            b = idx[i:i + 256]
            res[b] = net(x[b], ln[b], skip=bool(flag))
    return res
def f1(p, y):
    tp = ((p == 1) & (y == 1)).sum().item(); fp = ((p == 1) & (y == 0)).sum().item(); fn = ((p == 0) & (y == 1)).sum().item()
    return 2 * tp / max(1.0, 2 * tp + fp + fn)
def metrics(lg, y, task):
    if task == "S":
        p = lg.argmax(-1)
        return dict(acc=(p == y).float().mean().item(),
                    ce=Fn.cross_entropy(lg.reshape(-1, V_S), y.reshape(-1)).item(),
                    per=(p == y).float().mean(1))
    p = (lg > 0).float()
    return dict(acc=(p == y).float().mean().item(), f1=f1(p, y),
                per=(p == y).float().mean(1))
def paired(a, b):                                     # b - a 的配对差
    d = (b - a).double()
    return d.mean().item(), (d.std(unbiased=True) / math.sqrt(len(d))).item()
def flags(task, ln):                                  # 运行时判据: n = 规模(简单=长度, 复杂=边数 k)
    return (ln if task == "S" else (ln - 2) // 3) <= 8
# ---------------- 地板 ----------------
def floors(task, tr, te):
    xtr, ytr, lntr, xte, yte, lnte = tr[0], tr[1], tr[2], te[0], te[1], te[2]
    if task == "S":
        cnt = torch.stack([torch.bincount(ytr[:, j], minlength=V_S) for j in range(LEN_S)]).float()
        prob = cnt / cnt.sum(1, keepdim=True)                          # 逐位置众数 = majority
        acc = (yte == prob.argmax(1)[None, :]).float().mean().item()
        ce = -prob.gather(1, yte.T).log().mean().item()
        rule = (torch.arange(LEN_S) * (V_S - 1) / (LEN_S - 1)).round().long()   # 按长度均匀猜第 i 小
        racc = (yte == rule[None, :]).float().mean().item()
        return (f"majority(逐位置众数): acc={acc:.4f} ce={ce:.3f}nats | "
                f"长度规则 out[i]=round(i*(V-1)/(L-1)): acc={racc:.4f}")
    maj = (ytr.mean(0) > 0.5).float()[None, :].expand_as(yte)          # 逐节点训练众数 = majority
    k = (lnte - 2) // 3
    cnt = torch.zeros_like(yte)
    for i in range(len(yte)):                                          # 计数规则(只看边数 k)
        if int(k[i]) <= 4:
            cnt[i, int(xte[i, 1])] = 1.0                               # k<=4: 只猜起点
        else:
            cnt[i] = 1.0                                               # 否则: 猜全可达
    return (f"majority(逐节点训练众数): acc={(maj == yte).float().mean():.4f} f1={f1(maj, yte):.4f} | "
            f"计数规则(k<=4→仅起点,否则全可达): acc={(cnt == yte).float().mean():.4f} f1={f1(cnt, yte):.4f}")
# ---------------- 主流程 ----------------
def main():
    t0 = time.time()
    pr = {}
    for task in ("S", "C"):                                            # 探针: 测每步耗时(与真实训练同构)
        x, y, ln = gen(task, 512, 999)
        net, opt = Net("C", task), None
        opt = torch.optim.AdamW(net.parameters(), lr=LR)
        for _ in range(4):
            train_one(net, opt, x[:BS], y[:BS], ln[:BS], task)
        t1 = time.time()
        for _ in range(8):
            train_one(net, opt, x[:BS], y[:BS], ln[:BS], task)
        pr[task] = (time.time() - t1) / 8
    unit = 3 * (pr["S"] + pr["C"])                                     # A/B/C 三个真训练臂
    over = 45.0
    cost = lambda st, ns: ns * st * unit + over
    seeds = [42] if len(sys.argv) < 2 else [int(sys.argv[1])]          # 可传 seed 覆盖(默认 42)
    if seeds == [42] and cost(STEPS, 2) <= 520:
        seeds = [42, 43]
    steps = STEPS if len(sys.argv) < 3 else int(sys.argv[2])           # 可传步数覆盖(与 seed42 同口径)
    if steps == STEPS and cost(steps, 1) > 520:
        steps = max(800, int((520 - over) / unit))
        print(f"[预算] 投影 {cost(STEPS, 1):.0f}s > 540s => 缩步数 STEPS {STEPS} -> {steps}"
              f" (探针 S={pr['S']*1000:.0f}ms/步, C={pr['C']*1000:.0f}ms/步)", flush=True)
    if len(seeds) > 1 and cost(steps, 2) > 520:
        seeds = [42]
        print("[预算] 加 seed43 会超 10min => 只跑 seed=42", flush=True)
    print(f"[预算] steps={steps} seeds={seeds} 单次投影={cost(steps, len(seeds)):.0f}s", flush=True)
    st = {}
    for seed in seeds:
        st[seed] = {}
        for task in ("S", "C"):
            tr, te = gen(task, N_TRAIN, seed), gen(task, N_TEST, seed + 5000)
            print(f"[FLOOR {task} s{seed}] {floors(task, tr, te)}", flush=True)
            res, alp, cnet = {}, float("nan"), None
            for arm in ("A", "B", "C"):                                # D = 训练同 C + 运行时判据
                net, tl = train_run(task, arm, seed, steps, tr)
                res[arm] = metrics(logits_of(net, te, task, torch.zeros(len(te[0]), dtype=torch.bool)), te[1], task)
                res[arm]["tl"] = tl
                if arm == "A":
                    alp = net.alpha.item()
                if arm == "C":
                    cnet = net
            res["D"] = metrics(logits_of(cnet, te, task, flags(task, te[2])), te[1], task)
            fmt = (lambda a: f"{a}: ce={res[a]['ce']:.4f} acc={res[a]['acc']:.4f}" if task == "S"
                   else f"{a}: acc={res[a]['acc']:.4f} f1={res[a]['f1']:.4f}")
            print(f"[MAIN {task} s{seed}] " + " | ".join(fmt(a) for a in ARMS), flush=True)
            print(f"[ALPHA {task} s{seed}] alpha={alp:.4f} final_train_loss A/B/C="
                  f"{res['A']['tl']:.4f}/{res['B']['tl']:.4f}/{res['C']['tl']:.4f}", flush=True)
            st[seed][task] = {}
            for a, b in (("B", "A"), ("C", "A"), ("C", "B"), ("D", "C")):
                d, se = paired(res[b]["per"], res[a]["per"])
                st[seed][task][(a, b)] = (d, se)
                print(f"[PAIR {task} {a}-{b} s{seed}] Δ={d:+.4f} SE={se:.4f} "
                      f">2SE={'YES' if abs(d) > 2 * se else 'NO'}", flush=True)
            if task == "C":
                k = (te[2] - 2) // 3
                for arm in ("C", "D"):
                    print(f"[PERK {arm} s{seed}] " + " ".join(
                        f"k={kk}:acc={res[arm]['per'][k == kk].mean():.4f}" for kk in KS), flush=True)
                sub = flags("C", te[2])
                d, se = paired(res["C"]["per"][sub], res["D"]["per"][sub])
                print(f"[PAIR C D-C s{seed} k<=8子集 n={int(sub.sum())}] Δ={d:+.4f} SE={se:.4f} "
                      f">2SE={'YES' if abs(d) > 2 * se else 'NO'}", flush=True)
    for seed in seeds:                                                 # P4: 复杂 vs 简单
        for p in (("B", "A"), ("C", "A")):
            ds, ess = st[seed]["S"][p]; dc, esc = st[seed]["C"][p]
            se = math.sqrt(ess ** 2 + esc ** 2)
            print(f"[P4 {p[0]}-A s{seed}] simple Δ={ds:+.4f}(SE {ess:.4f}) complex Δ={dc:+.4f}(SE {esc:.4f}) "
                  f"complex-simple={dc-ds:+.4f} SE={se:.4f} >2SE={'YES' if abs(dc-ds) > 2*se else 'NO'}", flush=True)
    print(f"[TIME] total={time.time()-t0:.0f}s steps={steps} seeds={seeds}", flush=True)
if __name__ == "__main__":
    main()
