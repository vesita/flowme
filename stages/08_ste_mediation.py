# -*- coding: utf-8 -*-
"""S08：STE 离散中介 vs 连续直传 —— 序列顺序重建任务（打乱输入 -> 原序输出）。

纪律 E16：离散中介必须用 STE 保证梯度回传；先探针（grad is not None）后训练。
两臂唯一变量 = 卡间表示：
  M: [16,128] 连续张量直传
  T: softmax(Linear(128,V_mid)/tau) -> STE(前向 argmax / 反向 softmax) -> Emb_mid(V_mid,128)
容量扫描 V_mid ∈ {8,32,128,1024}；E15 旁路 = 思考卡 Identity。
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import math
import multiprocessing as mp
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------- 内联常数 ----------------
VOCAB = 64
SEQ = 16
D = 128
FF = 512
HEADS = 4
N_LAYERS = 1
N_TRAIN = 8000
N_TEST = 2000
BATCH = 64
STEPS = 4000
LR = 1e-3
TAU0 = 1.0
TAU1 = 0.5
SEEDS = [42, 43]
V_MIDS = [8, 32, 128, 1024]
PROBE_STEPS = 20
PROBE_V = 128
EPS = 1e-6
IDENTITY_INVALID_CE = 1.0
MAX_WORKERS = 12


def log(msg=""):
    print(msg, flush=True)


def sinusoidal(seq, d):
    pos = torch.arange(seq, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d, 2, dtype=torch.float32) * (-math.log(10000.0) / d))
    pe = torch.zeros(seq, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def make_data(n, seed):
    """每行 = 16 个互不相同的 token；inputs = 打乱，targets = 排序（原序）。"""
    g = torch.Generator().manual_seed(seed)
    base = torch.rand(n, VOCAB, generator=g).topk(SEQ, dim=-1).indices
    perm = torch.rand(n, SEQ, generator=g).argsort(dim=-1)
    inputs = base.gather(1, perm)
    targets = base.sort(dim=-1).values
    return inputs, targets


class InputCard(nn.Module):
    """输入卡：Emb(64,128) + 正弦 PE + 1 层 TransformerEncoder(128,4)。"""

    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(D, HEADS, FF, dropout=0.0, batch_first=True)
        self.emb = nn.Embedding(VOCAB, D)
        self.enc = nn.TransformerEncoder(layer, N_LAYERS)
        self.register_buffer("pe", sinusoidal(SEQ, D), persistent=False)

    def forward(self, x):
        return self.enc(self.emb(x) + self.pe)


class ThinkCard(nn.Module):
    """思维卡：1 层 TransformerEncoder(128,4)。"""

    def __init__(self):
        super().__init__()
        layer = nn.TransformerEncoderLayer(D, HEADS, FF, dropout=0.0, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, N_LAYERS)

    def forward(self, h):
        return self.enc(h)


class Net(nn.Module):
    """arm: 'M' 连续直传 / 'T' STE 离散中介 / 'ID' 思考卡=Identity（E15 旁路）。"""

    def __init__(self, arm, v_mid=0):
        super().__init__()
        self.arm = arm
        self.v_mid = v_mid
        self.incard = InputCard()
        self.think = nn.Identity() if arm == "ID" else ThinkCard()
        if arm == "T":
            self.med_lin = nn.Linear(D, v_mid)          # 中介线性层
            self.emb_mid = nn.Embedding(v_mid, D)       # 中介嵌入
        self.out = nn.Linear(D, VOCAB)

    def forward(self, x, tau):
        h = self.think(self.incard(x))
        if self.arm == "T":
            p = F.softmax(self.med_lin(h) / tau, dim=-1)
            hard = F.one_hot(p.argmax(dim=-1), self.v_mid).to(dtype=p.dtype)
            ste = (hard - p).detach() + p               # 前向 hard，反向 soft
            h = ste @ self.emb_mid.weight
        return self.out(h)


def group_grads(net):
    """按卡分组统计 (grad 非 None 参数数, 参数总数)。"""
    groups = {"输入卡": [0, 0], "思维卡": [0, 0], "中介线性层": [0, 0], "中介嵌入": [0, 0], "输出卡": [0, 0]}
    for name, p in net.named_parameters():
        if name.startswith("incard."):
            key = "输入卡"
        elif name.startswith("think."):
            key = "思维卡"
        elif name.startswith("med_lin"):
            key = "中介线性层"
        elif name.startswith("emb_mid"):
            key = "中介嵌入"
        else:
            key = "输出卡"
        groups[key][1] += 1
        if p.grad is not None:
            groups[key][0] += 1
    return groups


def train_steps(net, opt, data, steps, seed, probe=False):
    x_all, y_all = data
    g = torch.Generator().manual_seed(seed + 7)
    order = torch.randperm(len(x_all), generator=g)
    ptr = 0
    for step in range(steps):
        sel = order[ptr:ptr + BATCH]
        if len(sel) < BATCH:
            sel = torch.cat([sel, order[: BATCH - len(sel)]])
        ptr = (ptr + BATCH) % len(order)
        tau = TAU0 + (TAU1 - TAU0) * (step / max(1, steps - 1))
        loss = F.cross_entropy(net(x_all[sel], tau).reshape(-1, VOCAB), y_all[sel].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if probe and step + 1 == PROBE_STEPS:
            break


def evaluate(net, x, y, tau=TAU1):
    net.eval()
    losses, correct, total = [], 0, 0
    with torch.no_grad():
        for i in range(0, len(x), 256):
            logits = net(x[i:i + 256], tau)
            b = y[i:i + 256].shape[0]
            per_ex = F.cross_entropy(logits.reshape(-1, VOCAB), y[i:i + 256].reshape(-1),
                                     reduction="none").reshape(b, SEQ).mean(dim=1)
            losses.append(per_ex)
            correct += (logits.argmax(-1) == y[i:i + 256]).sum().item()
            total += y[i:i + 256].numel()
    net.train()
    all_loss = torch.cat(losses)
    return all_loss.mean().item(), all_loss.std(unbiased=True).item() / math.sqrt(len(all_loss)), correct / total


def probe(arm, v_mid, seed):
    """E16 前门：训练 PROBE_STEPS 步后看各卡参数 grad is not None 的数量。"""
    torch.manual_seed(seed)
    net = Net(arm, v_mid)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    train_steps(net, opt, make_data(N_TRAIN, seed), PROBE_STEPS, seed, probe=True)
    return group_grads(net)


def run_one(cfg):
    torch.set_num_threads(1)
    arm, v_mid, seed, steps = cfg["arm"], cfg["v_mid"], cfg["seed"], cfg["steps"]
    torch.manual_seed(seed)
    train_x, train_y = make_data(N_TRAIN, seed)
    test_x, test_y = make_data(N_TEST, seed + 100000)
    net = Net(arm, v_mid)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    t0 = time.time()
    train_steps(net, opt, (train_x, train_y), steps, seed)
    train_ce, _, _ = evaluate(net, train_x, train_y)
    test_ce, test_se, acc = evaluate(net, test_x, test_y)
    return {
        "arm": arm, "v_mid": v_mid, "seed": seed, "steps": steps,
        "train_ce": train_ce, "test_ce": test_ce, "test_se": test_se,
        "acc": acc, "sec": time.time() - t0,
    }


def counting_baselines(seed):
    """地板（计数类）：复制输入 / 按长度猜 / majority。全是 test CE（越低越好）。"""
    train_x, train_y = make_data(N_TRAIN, seed)
    test_x, test_y = make_data(N_TEST, seed + 100000)
    counts = torch.bincount(train_y.flatten(), minlength=VOCAB).float()
    uni = (counts + EPS) / (counts.sum() + EPS * VOCAB)          # 长度恒为 16 => 用 unigram
    len_ce = -uni.log()[test_y].mean().item()
    maj = int(counts.argmax())

    def onehot_ce(pred, tgt):
        match = pred == tgt
        v = torch.where(match, -math.log(1.0 - EPS), -math.log(EPS / (VOCAB - 1)))
        return v.mean().item(), match.float().mean().item()

    copy_ce, copy_hit = onehot_ce(test_x, test_y)
    maj_ce, _ = onehot_ce(torch.full_like(test_y, maj), test_y)
    return {"copy_ce": copy_ce, "copy_hit": copy_hit, "len_ce": len_ce, "maj_ce": maj_ce}


def fmt_groups(g):
    return " ".join(f"{k}={v[0]}/{v[1]}" for k, v in g.items() if v[1] > 0)


def main():
    torch.set_num_threads(1)
    t_start = time.time()
    cfg_base = dict(steps=STEPS)
    log(f"S08 STE mediation | torch {torch.__version__} | steps={STEPS} batch={BATCH} "
        f"train={N_TRAIN} test={N_TEST} seeds={SEEDS} tau={TAU0}->{TAU1}")
    log(f"命令: CUDA_VISIBLE_DEVICES='' uv run --no-sync python stages/08_ste_mediation.py")

    # ---------- A. 前门 E16 ----------
    log("\n=== [A] 前门 E16 探针（20 步后 grad is not None 参数数）===")
    pm = probe("M", 0, SEEDS[0])
    pt = probe("T", PROBE_V, SEEDS[0])
    log(f"  M(直传)      {fmt_groups(pm)}")
    log(f"  T(STE,V=128) {fmt_groups(pt)}")
    m_ok = pm["输入卡"][0] > 0 and pm["思维卡"][0] > 0
    t_ok = (pt["输入卡"][0] > 0 and pt["思维卡"][0] > 0
            and pt["中介线性层"][0] > 0 and pt["中介嵌入"][0] > 0)
    if not (m_ok and t_ok):
        log("[FAIL] 前门未过：上游 grad 全为 None => STE 未生效 / 梯度断链，停止。")
        log(f"退出码 1，总墙钟 {time.time() - t_start:.1f}s")
        return 1
    log("[PASS] 前门通过：M 与 T 上游 grad 均非 None。")

    ctx = mp.get_context("spawn")
    workers = min(MAX_WORKERS, os.cpu_count() or 1, 12)

    # ---------- B. E15 旁路 + 计数地板 ----------
    log("\n=== [B] E15 旁路检查 + 计数类地板 ===")
    bl = counting_baselines(SEEDS[0])
    log(f"  复制输入 CE={bl['copy_ce']:.3f}（位置命中 {bl['copy_hit']:.3f}） | 按长度猜 CE={bl['len_ce']:.3f} "
        f"| majority CE={bl['maj_ce']:.3f} | 输入完全决定目标(完美模型) CE->0")
    id_cfgs = [dict(cfg_base, arm="ID", v_mid=0, seed=s) for s in SEEDS]
    with ctx.Pool(processes=min(workers, len(id_cfgs))) as pool:
        id_res = pool.map(run_one, id_cfgs)
    for r in id_res:
        log(f"  identity(思考卡=Identity) seed={r['seed']} test CE={r['test_ce']:.4f} "
            f"(se_example={r['test_se']:.4f}) train CE={r['train_ce']:.4f} acc={r['acc']:.3f} {r['sec']:.0f}s")
    id_min = min(r["test_ce"] for r in id_res)
    if id_min < IDENTITY_INVALID_CE:
        log(f"[FAIL] 恒等中介 test CE={id_min:.4f} < {IDENTITY_INVALID_CE} => 任务可被旁路解掉，任务无效，停止。")
        log(f"退出码 2，总墙钟 {time.time() - t_start:.1f}s")
        return 2
    log(f"[PASS] 旁路未解题（identity min CE={id_min:.4f} >= {IDENTITY_INVALID_CE}），任务有效。")

    # ---------- C. 容量扫描 ----------
    log("\n=== [C] 主表：M 直传 1 点 + T 离散 4 容量 × 2 seed ===")
    jobs = [dict(cfg_base, arm="M", v_mid=0, seed=s) for s in SEEDS]
    jobs += [dict(cfg_base, arm="T", v_mid=v, seed=s) for v in V_MIDS for s in SEEDS]
    t_c = time.time()
    with ctx.Pool(processes=min(workers, len(jobs))) as pool:
        res = pool.map(run_one, jobs)
    for r in sorted(res, key=lambda r: (r["arm"] != "M", r["v_mid"], r["seed"])):
        log(f"  arm={r['arm']} V_mid={r['v_mid']:>4} seed={r['seed']} test CE={r['test_ce']:.4f} "
            f"se_example={r['test_se']:.4f} train CE={r['train_ce']:.4f} acc={r['acc']:.3f} {r['sec']:.0f}s")

    def ce(arm, v, seed):
        return next(r["test_ce"] for r in res if r["arm"] == arm and r["v_mid"] == v and r["seed"] == seed)

    m_mean = sum(ce("M", 0, s) for s in SEEDS) / len(SEEDS)
    log(f"\n  M(连续直传) mean test CE = {m_mean:.4f}  [ "
        + "  ".join(f"seed{s}={ce('M', 0, s):.4f}" for s in SEEDS) + " ]")
    log("  P2 追平判定（Δ = CE_T - CE_M，按 seed 配对；SE = 配对差的 std/√2；追平 ⇔ Δ < 2SE）:")
    for v in V_MIDS:
        diffs = [ce("T", v, s) - ce("M", 0, s) for s in SEEDS]
        dm = sum(diffs) / len(diffs)
        se = math.sqrt(sum((d - dm) ** 2 for d in diffs) / (len(diffs) - 1)) / math.sqrt(len(diffs))
        mark = "追平" if dm < 2 * se else ("T优于M" if dm < 0 else "差于M")
        log(f"    V_mid={v:>4}: CE_T={sum(ce('T', v, s) for s in SEEDS)/len(SEEDS):.4f}  "
            f"Δ={dm:+.4f}  2SE={2*se:.4f}  => {mark}")

    wall = time.time() - t_start
    log(f"\n=== 墙钟 === 主扫描 {time.time() - t_c:.1f}s | 总 {wall:.1f}s ({wall/60:.1f} min) | 每 run {STEPS} 步 | workers={workers} spawn, torch threads=1")
    log("[DONE] exit 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
