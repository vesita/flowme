#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P26 极小版：卡间表示 M=张量直传 vs T=固定词表 argmax 离散中介，报重构交叉熵。CPU 单文件。"""
import os, math, json, time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- 内联常数（照用） ----
D, NMAX, V_IN, V_MID = 128, 64, 64, 64
BATCH, STEPS, N_TRAIN, N_TEST, LR = 64, 2000, 4000, 1000, 1e-3
SEEDS = (42, 43)
HEADS = 4

torch.set_num_threads(min(8, os.cpu_count() or 4))
DEV = torch.device("cpu")


def new_enc():
    """思维卡/输入卡共用的 1 层 TransformerEncoder(d=128, heads=4)。"""
    layer = nn.TransformerEncoderLayer(D, HEADS, dropout=0.1, batch_first=True)
    return nn.TransformerEncoder(layer, 1).to(DEV)


class Net(nn.Module):
    """输入卡 Emb+PE+Enc → 思维卡(Enc 或恒等) → [T: argmax 离散中介] → 输出卡 Linear。"""

    def __init__(self, arm, identity_think, seed):
        super().__init__()
        torch.manual_seed(seed)  # 共享模块两臂初值一致（中介在共享模块之后初始化）
        self.is_T = arm == "T"
        self.emb = nn.Embedding(V_IN, D).to(DEV)
        pe = torch.zeros(NMAX, D)
        pos = torch.arange(NMAX).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, D, 2).float() * (-math.log(10000.0) / D))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
        self.enc_in = new_enc()
        self.think = nn.Identity() if identity_think else new_enc()
        if self.is_T:
            self.med_lin = nn.Linear(D, V_MID).to(DEV)   # h -> logits
            self.med_emb = nn.Embedding(V_MID, D).to(DEV)  # argmax -> 再编码
        self.head = nn.Linear(D, V_IN).to(DEV)

    def forward(self, x, mask):
        h = self.emb(x) * math.sqrt(D) + self.pe.unsqueeze(0)
        h = self.enc_in(h, src_key_padding_mask=~mask)
        h = self.think(h)
        if self.is_T:
            h = self.med_emb(self.med_lin(h).argmax(-1))  # 离散化 + embedding 再编码
        return self.head(h)


def gen(seed, n):
    """合成数据：长度 16-32，词表 64，pad 到 64；mask=True 为真实位置。"""
    g = torch.Generator().manual_seed(seed)
    lens = torch.randint(16, 33, (n,), generator=g)
    xs = torch.randint(V_IN, (n, NMAX), generator=g)
    mask = torch.arange(NMAX).unsqueeze(0) < lens.unsqueeze(1)
    return xs.to(DEV), mask.to(DEV)


def masked_ce(model, x, mask):
    return F.cross_entropy(model(x, mask)[mask], x[mask])


def data_pair(seed):
    xtr, mtr = gen(seed, N_TRAIN)            # 同 seed 两臂数据完全一致
    xte, mte = gen(seed + 100000, N_TEST)
    return xtr, mtr, xte, mte


def train_eval(arm, seed, identity_think):
    xtr, mtr, xte, mte = data_pair(seed)
    torch.manual_seed(seed)
    model = Net(arm, identity_think, seed)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_med = sum(p.numel() for n, p in model.named_parameters()
                if p.requires_grad and n.startswith("med_"))
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    g = torch.Generator().manual_seed(seed + 7)
    t0 = time.time()
    for _ in range(STEPS):
        idx = torch.randint(N_TRAIN, (BATCH,), generator=g)
        loss = masked_ce(model, xtr[idx], mtr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        te = masked_ce(model, xte, mte).item()
        tr = masked_ce(model, xtr, mtr).item()
    return {"arm": arm, "seed": seed, "think": "ident" if identity_think else "enc",
            "params": n_params, "trunk": n_params - n_med, "med": n_med,
            "test_ce": round(te, 4), "train_ce": round(tr, 4),
            "sec": round(time.time() - t0, 1)}


def baselines(seed):
    xtr, mtr, xte, mte = data_pair(seed)
    copy_ce = 0.0  # copy 基线：输出恒等于输入 => 每位置预测其真值 => CE=0（任务有恒等解）
    lens_tr = mtr.sum(1)
    dist = {}
    for n in range(16, 33):
        sel = lens_tr == n
        cnt = torch.ones(V_IN)  # Laplace 平滑
        cnt += torch.bincount(xtr[sel][mtr[sel]].reshape(-1), minlength=V_IN)
        dist[n] = (cnt / cnt.sum()).to(DEV)
    tot, cnt = 0.0, 0
    for i in range(xte.shape[0]):
        n = int(mte[i].sum())
        tot += -torch.log(dist[n][xte[i][:n]]).sum().item()
        cnt += n
    return {"copy_ce": copy_ce, "len_ce": round(tot / cnt, 4)}


def main():
    print("CMD: CUDA_VISIBLE_DEVICES='' uv run --project DTSeek --no-sync python meme/stage1.py", flush=True)
    bl = baselines(42)
    print("BASELINE " + json.dumps(bl), flush=True)
    rows = []
    for ident in (False, True):  # False=正常；True=P5 思维卡=恒等
        for seed in SEEDS:
            for arm in ("M", "T"):
                r = train_eval(arm, seed, ident)
                rows.append(r)
                print("RUN " + json.dumps(r), flush=True)
    print("=== SUMMARY ===")
    print(f"params  M={rows[0]['params']}  T={rows[1]['params']}  "
          f"(共享干 trunk={rows[0]['trunk']}，T 多中介 {rows[1]['med']} 参数)")
    for tag, ident in (("NORMAL", False), ("P5_ident", True)):
        sub = [r for r in rows if r["think"] == ("ident" if ident else "enc")]
        for seed in SEEDS:
            m = next(r for r in sub if r["arm"] == "M" and r["seed"] == seed)
            t = next(r for r in sub if r["arm"] == "T" and r["seed"] == seed)
            print(f"{tag} seed={seed}  CE_M={m['test_ce']}  CE_T={t['test_ce']}  "
                  f"delta(T-M)={round(t['test_ce'] - m['test_ce'], 4)}")
    print(f"BASELINE copy={bl['copy_ce']}  length={bl['len_ce']}  (uniform={round(math.log(V_IN), 4)})")
    print("=== END ===", flush=True)


if __name__ == "__main__":
    main()
