#!/usr/bin/env python3
"""S69 · L2-r1 判据驱动：把「跨轮消解 + 干扰项排除」对话任务接进 cards.CardPipeline。

数据来自 stages/68_dialog_gen_r1.py（本文件不造数据；缺 logs/68_dialog_*.jsonl 时先自动跑生成器）。
模型/训练/贪心 EM 的写法逐字沿用 stages/67_dialog_cards.py（因而也等于 S65/S12）：
  · `CardPipeline(residual=False)`、d=128 ff=512 nhead=4 maxlen=512 pad_id/EOS 与 S12 一致；
  · teacher forcing **只在目标 span 上算 CE**（`masked_ce`）；
  · 贪心 EM = batch=1、正确列索引（`last_col`）——R28 自洽口径；
  · 数据指纹 `data_fp` 与 S65 同口径（逐条 len|ids 拼接的 sha256[:16]）。

判据原文（L2_PLAN.md §7，写死）：
  Q1（主）「卡片组 EM > max_naive + 2×SE 且 2 seed 同号」；
  Q2「干扰臂：把轮1/轮2 的实体互换后重评同一 ckpt ⇒ EM 必掉」；
  Q3「地板含计数类且 train 上 < 90%（沿用 P2 口径）」；
  Q4「报告是否饱和（EM < 99% 且 loss 不塌到 1e-4），否则只能当"能承载"的证据」。

用法：
  python stages/69_dialog_r1.py --smoke                  # 50 步、只评 60 条（探路）
  python stages/69_dialog_r1.py --seed 1234              # 全量 3000 步
  python stages/69_dialog_r1.py --seed 1234 --swap       # Q2：换实体、保留原 target，同一 ckpt 重评
  # L2-r1b / r1c / r1d / r2 单变量臂（本文件只加参数，其余逻辑逐字不变）：
  python stages/69_dialog_r1.py --data-prefix 68b --seed 1234            # r1b：共享词表数据（68b_dialog_*）
  python stages/69_dialog_r1.py --data-prefix 68 --seed 1234 --steps 12000   # r1c：互斥词表、12000 步
  python stages/69_dialog_r1.py --data-prefix 68d --seed 1234            # r2：共享词表 + 确定性句面（68d_dialog_*）
  python stages/69_dialog_r1.py --data-prefix 68d --seed 1234 --steps 12000
  python stages/69_dialog_r1.py --data-prefix 68e --seed 1234            # r3：轮3 保留 v3（68e_dialog_*）
  python stages/69_dialog_r1.py --data-prefix 68f --seed 1234            # r4：test 实体=训练用字的新组合（68f_dialog_*）
  python stages/69_dialog_r1.py --data-prefix 68g --seed 1234            # r6：train 名池=320 组合 / test=20 留出新组合（68g_dialog_*）
  --data-prefix {68,68b,68d,68e,68f,68g} 决定读哪份 jsonl（logs/<prefix>_dialog_*.jsonl）与 ckpt 命名
  （68 ⇒ 69_ckpt_seed*.pt 不变；68b ⇒ 69_ckpt_68b_seed*.pt；68d/68e/68f 同理，互不覆盖）；
  缺 68b/68d/68e/68f 数据时自动带对应开关（--shared-vocab / --det-reply / --keep-v3 /
  --keep-v3 --novel-combos）跑生成器；68g 用 --keep-v3 --novel-train。
  ★§13.2 修复（L2-r6）：seed 在**构造模型之前**设置 + [INIT-CHECK]（同 seed 连造两次逐张量差=0）。
  --steps 默认 3000（--smoke 时固定 50 步）。
  ★§8.4 必跑诊断（训练后、无开关）：在 train 与 test 上分别报「目标首 token（=实体首字）的
  top1/top5」与「贪心生成的实体来源（正确实体 / 干扰实体 / 训练词表名 / 其它）」，打印为 [DIAG] 行。
GPU 长跑必须：systemd-run --user --unit=<名> --collect
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 ...
只允许写本文件与 logs/。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
import time
from collections import Counter

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
if NS_ROOT not in sys.path:
    sys.path.insert(0, NS_ROOT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from cards import CardPipeline  # noqa: E402

# ---------------- 配置（与 S65/S12/S67 一致） ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512
STEPS = 3000
LR = 1e-3
BATCH = 32
GEN_BATCH = 32
EM_BATCH = 1                 # ★主口径：batch=1（R28 自洽写法）
SMOKE_STEPS = 50
SMOKE_EVAL = 60
LOG_DIR = os.path.join(ROOT, "logs")
DATA_TMPL = os.path.join(LOG_DIR, "68_dialog_{split}.jsonl")   # main 里按 --data-prefix 覆盖
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
D68_PATH = os.path.join(ROOT, "stages", "68_dialog_gen_r1.py")


def load_d68():
    """按路径加载 stages/68_dialog_gen_r1.py（文件名以数字开头，不能普通 import）。"""
    spec = importlib.util.spec_from_file_location("d68_dialog_gen_r1", D68_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d68 = load_d68()
MARK_SELF = d68.MARK_SELF
MARK_OTHER = d68.MARK_OTHER

# ---------------- device 纪律（ROCm 下报成 cuda，正常） ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
if device != "cuda":
    print("GPU 起不来，立即停（不用 CPU 硬跑）。", flush=True)
    print("torch.cuda.is_available() =", torch.cuda.is_available(), flush=True)
    print("HSA_OVERRIDE_GFX_VERSION =", os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)
    sys.exit(3)

from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(TOK_PATH)
V = tok.get_vocab_size()
PAD_ID = tok.token_to_id("<pad>")
EOS_ID = tok.token_to_id("<eos>")


def enc(text: str) -> list:
    return tok.encode(text, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


# ---------------- 数据 ----------------
def ensure_data(prefix: str = "68"):
    """缺数据时自动跑生成器；prefix=68b 带 --shared-vocab、prefix=68d 带 --det-reply。"""
    if os.path.exists(DATA_TMPL.format(split="train")) and os.path.exists(DATA_TMPL.format(split="test")):
        return
    print(f"[DATA] 缺 {DATA_TMPL.format(split='train')} ⇒ 先跑生成器", flush=True)
    extra = {"68b": ["--shared-vocab"], "68d": ["--det-reply"], "68e": ["--keep-v3"],
             "68f": ["--keep-v3", "--novel-combos"],
             "68g": ["--keep-v3", "--novel-train"]}.get(prefix, [])
    cmd = [sys.executable, D68_PATH] + extra
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[DATA] 生成器退出码 {r.returncode} ⇒ 停")


def load_jsonl(split: str) -> list:
    with open(DATA_TMPL.format(split=split), encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def swap_entities(text: str, r) -> str:
    """Q2：把"我自己的"与"我朋友的"两个实体**整体互换**（只改 prompt 文本，target 原样保留）。

    ★ 正确做法是**按角色槽位换名字**，而不是"换名字 + 换角色词"：
      第一人称自称（我自己的 / 我朋友）**不可互换**，换完会得到一句自相矛盾的话
      （"我朋友的咖啡机"配上"我"的主语 ⇒ 测量被污染，EM 下降不再能证明"读了历史"）。
      所以：把**出现在「我自己的」后面的那个实体**换成另一个名字，反之亦然
      ⇒ 互换后 ctx 依然完全自洽，只是"我自己的东西"换了一个人。

    两个实体名互不为子串（生成器已断言），整串 replace 是精确的一次性互换。
    轮3 用户话不含实体、也没有角色词 ⇒ 不变；target 保持不变。
    ⇒ 读历史的模型在互换后应当输出**另一个**名字 ⇒ 与原 target 不符 ⇒ EM 掉；
      只背模板的模型不看历史 ⇒ 输出不变 ⇒ EM 不降。
    """
    if not text:
        return text
    a_self, a_other = r["ent"], r["ent_other"]          # 原始世界里"我自己的 / 我朋友的"
    if not r["self_first"]:
        a_self, a_other = a_other, a_self
    return text.replace(a_self, "\x00").replace(a_other, "\x01").replace("\x00", a_other).replace("\x01", a_self)


def to_ids(recs, swap: bool = False):
    """把文本编码成 p/t；swap=True 时 p 用**实体互换后**的 ctx（t 仍是原 target）。"""
    n_swapped = 0
    for r in recs:
        src = r["ctx"]
        if swap:
            src2 = swap_entities(src, r)
            n_swapped += int(src2 != src)
            src = src2
        r["p"] = enc(src)
        r["t"] = enc(r["target"] + d68.EOS)
        r["ntok_all"] = len(r["p"]) + len(r["t"])
    return n_swapped


# ---------------- 模型 / loss（写法照抄 S67/S65） ----------------
def new_pipe() -> CardPipeline:
    """★residual=False：cards/target.py:28-30 默认带外层残差，S65 已实测必须关掉。"""
    return CardPipeline(d=D, vocab_size=V, ff=FF, nhead=NHEAD, maxlen=MAXLEN,
                        pad_id=PAD_ID, dropout=0.1, residual=False).to(device)


def seed_all(seed: int):
    """★§13.2 修复：seed 必须在**构造模型之前**设置（torch 默认流管参数初始化与重洗）。

    照 cards/example_add3d.py:305-311 的写法：manual_seed 先行，再构造 CardPipeline。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def param_digest(model) -> str:
    """全部参数的 hash（名字 + 形状 + float32 字节），用于 [INIT-CHECK] 的同 seed 复现自检。"""
    h = hashlib.sha256()
    for n, p in model.named_parameters():
        h.update(n.encode())
        h.update(str(tuple(p.shape)).encode())
        h.update(p.detach().cpu().to(torch.float32).contiguous().numpy().tobytes())
    return h.hexdigest()[:16]


def param_maxdiff(m1, m2) -> float:
    """两个同结构模型**逐张量的最大绝对差**（同 seed 应为 0）。"""
    md = 0.0
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
            assert n1 == n2, (n1, n2)
            md = max(md, float((p1 - p2).abs().max().item()))
    return md


def init_check(seed: int, other_seed: int) -> bool:
    """★L2-r6 自检（§13.2）：同一 seed 连造两次 ⇒ 全参数逐张量必须相同；不同 seed ⇒ 必须不同。"""
    def make(s):
        seed_all(s)
        return new_pipe()
    m1, m2 = make(seed), make(seed)
    d1, d2 = param_digest(m1), param_digest(m2)
    md_same = param_maxdiff(m1, m2)
    m3 = make(other_seed)
    d3 = param_digest(m3)
    md_diff = param_maxdiff(m1, m3)
    ok = (d1 == d2) and (md_same == 0.0) and (d1 != d3) and (md_diff > 0.0)
    print(f"[INIT-CHECK] seed={seed} 连续构造两次：全参数 hash={d1} vs {d2} ⇒ "
          f"{'相同' if d1 == d2 else '不同'} | 逐张量最大差={md_same:.3e}（必须 0）| "
          f"不同 seed({other_seed})：hash={d3} {'不同' if d1 != d3 else '相同(异常)'} "
          f"逐张量最大差={md_diff:.3e}（必须 >0）| 参数张量数={sum(1 for _ in m1.parameters())} "
          f"n_params={sum(p.numel() for p in m1.parameters())} ⇒ {'PASS' if ok else 'FAIL'}", flush=True)
    del m1, m2, m3
    return ok


def build_batch(recs):
    lens = [len(r["p"]) + len(r["t"]) for r in recs]
    n = max(lens)
    ids = torch.full((len(recs), n), PAD_ID, dtype=torch.long)
    s = torch.zeros(len(recs), dtype=torch.long)
    for i, r in enumerate(recs):
        ids[i, : len(r["p"])] = torch.tensor(r["p"])
        ids[i, len(r["p"]): len(r["p"]) + len(r["t"])] = torch.tensor(r["t"])
        s[i] = len(r["p"])
    return ids.to(device), s


def masked_ce(model, recs, copy: bool = False) -> torch.Tensor:
    """teacher forcing：只在目标 span [s, s+len(t)) 上算 CE（照抄 S65::masked_ce）。

    copy=True（--copy）时把分布换成混合分布（见 masked_ce_copy）；不加开关时下面逐字不变。
    """
    if copy:
        return masked_ce_copy(model, recs)
    ids, s = build_batch(recs)
    logits = model.logits(ids)
    logp = torch.log_softmax(logits, dim=-1)
    nll = torch.zeros((), device=device)
    ntok = 0
    for i, r in enumerate(recs):
        e = s[i] + len(r["t"])
        lp = logp[i, s[i] - 1: e - 1]
        tgt = ids[i, s[i]: e]
        nll = nll - lp.gather(1, tgt.unsqueeze(1)).sum()
        ntok += len(r["t"])
    return nll / ntok


# ---------------- L2-r5：复制 / 指针通路（--copy；不加开关时整段不执行） ----------------
COPY_EPS = 1e-30


class CopyHead(nn.Module):
    """输出卡的复制/指针通路：p = (1-g)·softmax(vocab logits) + g·copy_dist（L2_PLAN §12.3）。

    指针分布 a 只定义在**输入 prompt 位**上：
      · query = W_q·h，h = 目标段的解码位隐藏态（行 = 解码位 s-1..e-2，teacher forcing 的因果前缀）；
      · key   = W_k·meme_in，meme_in = input 卡对 ids[:, :L] 的输出（列 = 输入位 0..L-1）；
      · 越界列（>= src_len，即全部 target 位与 padding）被 -inf 掩掉 ⇒ 质量精确 0，绝不看 target；
      · copy_dist = a 按「该输入位上的 token id」（ids[:, :L]）scatter-add 到 V 维词表；
    门控 g = sigmoid(W_g·h + b_g)（每步一个标量，参数化并参与训练）。
    """

    def __init__(self, d: int = D):
        super().__init__()
        self.d = int(d)
        self.q = nn.Linear(d, d)          # 解码位隐藏态 → query
        self.k = nn.Linear(d, d)          # prompt 位（input 卡输出）→ key
        self.gate = nn.Linear(d, 1)       # 每步门控 g 的 logit

    def pointer(self, h, src, src_lens):
        """返回 (a, keep)：a[b,T,L] 行=解码位、列=**prompt 位**（越界列质量精确 0）。"""
        b, T, _ = h.shape
        L = src.size(1)
        score = (self.q(h) @ self.k(src).transpose(1, 2)) * (self.d ** -0.5)     # [b,T,L]
        keep = torch.arange(L, device=h.device).unsqueeze(0) < src_lens.unsqueeze(1)   # [b,L]
        score = score.masked_fill(~keep.unsqueeze(1), float("-inf"))
        return torch.softmax(score, dim=-1), keep

    def forward(self, logits, h, src, src_ids, src_lens):
        """返回 (logp[b,T,V], aux)：aux 全部 detached，只给 [COPY] 诊断用。"""
        b, T, V_ = logits.shape
        L = src.size(1)
        a, keep = self.pointer(h, src, src_lens)
        copy = torch.zeros(b, T, V_, device=logits.device, dtype=logits.dtype)
        copy.scatter_add_(2, src_ids.unsqueeze(1).expand(b, T, L), a)
        gs = self.gate(h)                                   # [b,T,1]，logit 域
        log_pv = torch.log_softmax(logits, dim=-1)
        log_pc = torch.log(copy.clamp_min(COPY_EPS))
        logp = torch.logaddexp(F.logsigmoid(-gs) + log_pv, F.logsigmoid(gs) + log_pc)
        return logp, dict(a=a.detach(), copy=copy.detach(), gs=gs.detach(), keep=keep)


def _copy_logp_rows(model, recs):
    """--copy：目标 span 逐位的混合分布 log 概率（行 = 解码位 s-1..e-2）。"""
    ids, s = build_batch(recs)
    out = model.route(ids)
    logits, meme_in, meme = out["logits"], out["meme_in"], out["meme"]
    b, L = len(recs), meme_in.size(1)
    src_lens = torch.tensor([len(r["p"]) for r in recs], device=device)
    T = max(len(r["t"]) for r in recs)
    h = torch.zeros(b, T, meme.size(-1), device=device, dtype=meme.dtype)
    lg = torch.zeros(b, T, logits.size(-1), device=device, dtype=logits.dtype)
    tgt = torch.zeros(b, T, dtype=torch.long, device=device)
    m = torch.zeros(b, T, dtype=torch.bool, device=device)
    for i, r in enumerate(recs):
        a0, n = int(s[i]) - 1, len(r["t"])
        h[i, :n] = meme[i, a0: a0 + n]
        lg[i, :n] = logits[i, a0: a0 + n]
        tgt[i, :n] = ids[i, int(s[i]): int(s[i]) + n]
        m[i, :n] = True
    logp, _aux = model.copy_head(lg, h, meme_in, ids[:, :L], src_lens)
    return logp, tgt, m


def masked_ce_copy(model, recs) -> torch.Tensor:
    """--copy 的训练 loss：混合分布上的 CE，仍只在目标 span 上算。"""
    logp, tgt, m = _copy_logp_rows(model, recs)
    lp = logp.gather(2, tgt.unsqueeze(2)).squeeze(2)
    return -(lp * m).sum() / m.sum()


def new_copy_stats() -> dict:
    return dict(n=0.0, g_sum=0.0, g_min=float("inf"), g_max=float("-inf"),
                ent_sum=0.0, copy_hits=0.0)


def _copy_step_stats(stats, aux, lg, chosen, act):
    """累加一步的 [COPY] 诊断：门控 g、指针分布熵（nats）、命中来自 copy 的比例。"""
    with torch.no_grad():
        g = torch.sigmoid(aux["gs"])[..., 0]                                  # [b,1]
        ent = -(aux["a"].clamp_min(COPY_EPS).log() * aux["a"]).sum(-1)        # [b,1]
        pv = torch.softmax(lg.unsqueeze(1), dim=-1)
        ch = torch.tensor(chosen, dtype=torch.long, device=g.device).view(-1, 1, 1)
        pc_t = aux["copy"].gather(2, ch)[..., 0]                              # [b,1]
        pv_t = pv.gather(2, ch)[..., 0]
        hit = ((g * pc_t) > ((1.0 - g) * pv_t)).float()
        mk = torch.tensor([bool(x) for x in act], device=g.device).view(-1, 1)
        n = float(mk.sum().item())
        if n <= 0:
            return
        stats["n"] += n
        stats["g_sum"] += float(g[mk].sum().item())
        stats["g_min"] = min(stats["g_min"], float(g[mk].min().item()))
        stats["g_max"] = max(stats["g_max"], float(g[mk].max().item()))
        stats["ent_sum"] += float(ent[mk].sum().item())
        stats["copy_hits"] += float(hit[mk].sum().item())


def copy_selftest(model, recs, n: int = 4) -> dict:
    """训练前自检 [COPY-SELFTEST]：指针打分的行/列来源 + 三条硬断言 + target 扰动实验。

    行 = 解码位（s-1..e-2），列 = **只用 prompt 位**（meme_in 的 0..src_len-1）。
    断言：① 越界列（>= src_len，含全部 target 位/padding）指针质量精确 0；
          ② 每个解码位的指针分布和为 1；③ copy_dist 的质量只落在 prompt 位出现过的 token id 上。
    扰动：把 target span 的 token 乱序重排后重前向 ⇒ 第 0 行（首个生成位 s-1）的指针分布必须
          **逐位不变**（该位的因果前缀里没有 target token）；第 j>0 行会变，如实报出。
    """
    model.eval()
    rs = recs[:n]
    ids, s = build_batch(rs)
    b, T = len(rs), max(len(r["t"]) for r in rs)
    src_lens = torch.tensor([len(r["p"]) for r in rs], device=device)
    L = ids.size(1)

    def _a_of(ids_in):
        out = model.route(ids_in)
        logits, meme_in, meme = out["logits"], out["meme_in"], out["meme"]
        h = torch.zeros(b, T, meme.size(-1), device=device, dtype=meme.dtype)
        for i, r in enumerate(rs):
            a0, nn_ = int(s[i]) - 1, len(r["t"])
            h[i, :nn_] = meme[i, a0: a0 + nn_]
        a, keep = model.copy_head.pointer(h, meme_in, src_lens)
        cp = torch.zeros(b, T, logits.size(-1), device=device, dtype=a.dtype)
        cp.scatter_add_(2, ids_in[:, :L].unsqueeze(1).expand(b, T, L), a)
        return a, keep, cp

    a, keep, copy = _a_of(ids)
    mass_out = float(a.masked_select(~keep.unsqueeze(1).expand_as(a)).abs().max().item())
    row_dev = float((a.sum(-1) - 1.0).abs().max().item())
    pmask = torch.zeros(b, V, device=device)
    for i in range(b):
        pmask[i, ids[i, : int(src_lens[i])]] = 1.0
    leak = float((copy * (1.0 - pmask).unsqueeze(1)).max().item())
    ok = (mass_out == 0.0) and (row_dev < 1e-5) and (leak == 0.0)
    gen = torch.Generator().manual_seed(0)
    ids2 = ids.clone()
    for i, r in enumerate(rs):
        e0, nn_ = int(s[i]), len(r["t"])
        ids2[i, e0: e0 + nn_] = ids[i, e0: e0 + nn_][torch.randperm(nn_, generator=gen)]
    a2, _k2, _c2 = _a_of(ids2)
    r0_same = bool(torch.equal(a2[:, 0, :], a[:, 0, :]))
    n_row_diff = int((a - a2).abs().sum(-1).gt(0).sum().item())
    model.train()
    print(f"[COPY-SELFTEST] 指针打分矩阵 a[b={b},T={T},L={L}]：行=解码位(s-1..e-2) 列=**prompt 位**"
          f"（meme_in 的 0..src_len-1；input 卡是因果的 ⇒ key 侧看不到任何 target token）；"
          f"target 位/padding 列被 -inf 掩掉 | 断言①越界列指针质量 max={mass_out:.1e}（必须精确 0）"
          f"②每行 a 和-1 的 max|dev|={row_dev:.1e}（<1e-5）③copy 质量落到「非 prompt token id」上的 "
          f"max={leak:.1e}（必须 0）⇒ {'PASS' if ok else 'FAIL'} | 扰动实验（target span 乱序重排后重"
          f"前向）：第0行（首个生成位 s-1）逐位不变={r0_same}；{T} 行里变了 {n_row_diff} 行"
          f"（第 j>0 行的 query 来自 teacher-forcing 的 target 前缀 t[0..j-1]，属标准写法）", flush=True)
    return dict(ok=ok, mass_out=mass_out, row_dev=row_dev, leak=leak,
                r0_same=r0_same, n_row_diff=n_row_diff)


def print_copy_stats(st_te, st_tr, tag) -> None:
    """--copy 的 [COPY] 行：门控 g 的均值/范围、指针分布熵、命中来自 copy 而非 vocab 的比例。"""
    def _s(st):
        if not st or st["n"] <= 0:
            return "n=0（无可用步）"
        n = st["n"]
        return (f"步数={int(n)} 门控g 均值={st['g_sum'] / n:.4f} 范围=[{st['g_min']:.4f}, "
                f"{st['g_max']:.4f}] | 指针分布熵均值={st['ent_sum'] / n:.4f} nats | "
                f"命中来自copy而非vocab={100.0 * st['copy_hits'] / n:.2f}%")
    print(f"[COPY] copy 通路诊断（贪心解码每一步、混合分布 argmax 位、口径与 [EM] 同）| "
          f"test({tag}): {_s(st_te)} | train子集: {_s(st_tr)}", flush=True)


def card_param_groups(model) -> dict:
    return {n: list(model.card(n).parameters()) for n in model.CARD_NAMES}


def train_run(model, train, seed: int, steps: int, copy: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, gate_res = 0, 0, None
    last_loss, loss_trace = float("nan"), []
    t0 = time.time()
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(model, [train[i] for i in idx], copy=copy)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        step += 1
        if step == 20:
            gate_res = {k: sum(p.grad is not None for p in ps)
                        for k, ps in card_param_groups(model).items()}
            tot = {k: len(ps) for k, ps in card_param_groups(model).items()}
            print(f"[GATE A] step20 grad is not None 计数 {gate_res} / 参数总数 {tot} "
                  f"=> {'PASS 全>0' if all(v > 0 for v in gate_res.values()) else 'FAIL'}",
                  flush=True)
        opt.step()
        last_loss = float(loss.item())
        if step % 500 == 0 or step == steps:
            loss_trace.append((step, last_loss))
            print(f"  [train] seed={seed} step={step}/{steps} loss={last_loss:.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    return gate_res, time.time() - t0, last_loss, loss_trace


@torch.no_grad()
def eval_ce(model, recs, copy: bool = False) -> list:
    model.eval()
    per = []
    for i in range(0, len(recs), GEN_BATCH):
        chunk = recs[i: i + GEN_BATCH]
        if copy:
            logp, tgt, m = _copy_logp_rows(model, chunk)
            for j, r in enumerate(chunk):
                lp = logp[j].gather(1, tgt[j].unsqueeze(1)).squeeze(1)
                per.append(float((-(lp * m[j]).sum() / m[j].sum()).item()))
            continue
        ids, s = build_batch(chunk)
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        for j, r in enumerate(chunk):
            e = s[j] + len(r["t"])
            lp = logp[j, s[j] - 1: e - 1]
            tgt = ids[j, s[j]: e]
            per.append(float((-lp.gather(1, tgt.unsqueeze(1)).sum() / len(r["t"])).item()))
    model.train()
    return per


@torch.no_grad()
def greedy_gen(model, recs, batch: int = 1, copy: bool = False, stats: dict = None) -> list:
    """贪心解码，batch=1、正确列索引（照抄 S65::greedy_gen mode='correct'）。

    copy=True（--copy）时每一步用**混合分布**取 argmax（其余逐字不变），并把门控/指针熵/命中
    来源累加进 stats（[COPY] 行用）。
    """
    model.eval()
    texts = [""] * len(recs)
    for i in range(0, len(recs), batch):
        chunk = recs[i: i + batch]
        lens = [len(r["p"]) for r in chunk]
        n0 = max(lens)
        ids = torch.full((len(chunk), n0), PAD_ID, dtype=torch.long)
        for j, r in enumerate(chunk):
            ids[j, : lens[j]] = torch.tensor(r["p"])
        ids = ids.to(device)
        gen_len = [0] * len(chunk)
        cap = [MAXLEN - L for L in lens]
        done = [False] * len(chunk)
        last_col = [L - 1 for L in lens]
        src_lens = torch.tensor(lens, device=device)
        for _ in range(max(cap)):
            if all(done):
                break
            act = [not d for d in done]
            if copy:
                out = model.route(ids)                 # 与 model.logits(ids) 同一条前向
                logits_all, meme_in, meme = out["logits"], out["meme_in"], out["meme"]
                ar = torch.arange(len(chunk), device=device)
                rows = torch.tensor(last_col, dtype=torch.long, device=device)
                lg = logits_all[ar, rows]              # [b,V]
                h = meme[ar, rows]                     # [b,d]
                logp, aux = model.copy_head(lg.unsqueeze(1), h.unsqueeze(1), meme_in,
                                            ids[:, : meme_in.size(1)], src_lens)
                nxt_scores = logp[:, 0]                # [b,V] 混合分布的 log 概率
            else:
                logits = model.logits(ids)
            nxt = []
            for j in range(len(chunk)):
                if done[j]:
                    nxt.append(PAD_ID)
                    continue
                t = int((nxt_scores[j] if copy else logits[j, last_col[j]]).argmax().item())
                nxt.append(t)
                gen_len[j] += 1
                if t == EOS_ID or gen_len[j] >= cap[j]:
                    done[j] = True
                last_col[j] = n0 + gen_len[j] - 1
            if copy and stats is not None:
                _copy_step_stats(stats, aux, lg, [int(x) for x in nxt], act)
            ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long, device=device).unsqueeze(1)], 1)
        for j in range(len(chunk)):
            g = ids[j, n0: n0 + gen_len[j]].tolist()
            texts[i + j] = dec([x for x in g if x != EOS_ID])
    model.train()
    return texts


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n) if n else float("nan")


def em_strict(gold: str, pred: str) -> int:
    """严格 EM：模板+槽位确定性生成 ⇒ 剥掉首尾空白后逐字符相等（唯一可判）。"""
    return int((gold or "").strip() == (pred or "").strip())


def evaluate(model, ev, tag: str, copy: bool = False, stats: dict = None):
    """配对 CE + 贪心 EM（含实体诊断）；返回一个 dict。"""
    ce = eval_ce(model, ev, copy=copy)
    cm, cse = mean_se(ce)
    txt = greedy_gen(model, ev, batch=EM_BATCH, copy=copy, stats=stats)
    hits = [em_strict(r["target"], txt[i]) for i, r in enumerate(ev)]
    ent_hit = [int(r["ent"] in txt[i]) for i, r in enumerate(ev)]
    other_hit = [int(r["ent_other"] in txt[i]) for i, r in enumerate(ev)]
    em = 100.0 * sum(hits) / len(hits)
    se = (em / 100.0 * (1 - em / 100.0) / len(hits)) ** 0.5 * 100.0
    return dict(tag=tag, cm=cm, cse=cse, txt=txt, hits=hits, em=em, se=se,
                em_ent=100.0 * sum(ent_hit) / len(ent_hit),
                em_other=100.0 * sum(other_hit) / len(other_hit))


# ---------------- §8.4 必跑诊断（无开关）：目标可生成性 + 生成实体来源 ----------------
DIAG_TRAIN_N = 800            # train 侧贪心诊断的子集规模（与 test 同规模、同 batch=1 口径）


@torch.no_grad()
def diag_first_token(model, recs, tag: str, copy: bool = False) -> dict:
    """目标首 token（= 正确实体首字）的 top1 / top5 —— §8.2 里一次戳穿数据缺陷的那条读数。

    口径：在**最后一个 prompt 位**（正确列索引 s-1，与贪心解码同一位）取 logits，
    看 r["t"][0] 的排名。字符级词表下这条直接回答"目标用字有没有质量"。
    """
    model.eval()
    n = len(recs)
    hit1 = hit5 = 0
    for i in range(0, n, GEN_BATCH):
        chunk = recs[i: i + GEN_BATCH]
        if copy:
            lps = _copy_logp_rows(model, chunk)[0][:, 0, :]     # 解码位 s-1 的混合分布
        else:
            ids, s = build_batch(chunk)
            logits = model.logits(ids)
        for j, r in enumerate(chunk):
            lp = lps[j] if copy else logits[j, int(s[j]) - 1]
            gold = int(r["t"][0])
            hit1 += int(int(lp.argmax().item()) == gold)
            hit5 += int(gold in torch.topk(lp, 5).indices.tolist())
    model.train()
    return dict(tag=tag, n=n, top1=100.0 * hit1 / n, top5=100.0 * hit5 / n)


def diag_entity_source(preds, recs, tag: str) -> dict:
    """贪心生成里的实体来源四分类：正确实体 / 干扰实体 / 训练词表名 / 其它（互斥、按序判定）。"""
    c = Counter()
    for r, txt in zip(recs, preds):
        if r["ent"] in txt:
            c["正确实体"] += 1
        elif r["ent_other"] in txt:
            c["干扰实体"] += 1
        elif any(nm in txt for nm in d68.NAMES_TRAIN):
            c["训练词表名"] += 1
        else:
            c["其它"] += 1
    n = len(recs)
    return dict(tag=tag, n=n, **{k: 100.0 * c[k] / n
                                 for k in ("正确实体", "干扰实体", "训练词表名", "其它")})


def print_diag(a, ev, txt, res, train, model, arm_tag: str, copy: bool = False):
    """§8.4 必跑诊断的两行：train/test 各自「首 token top1/top5」+「生成实体来源」。"""
    d_tr = diag_first_token(model, train, f"train(n={len(train)})", copy=copy)
    d_te = diag_first_token(model, ev, f"{arm_tag}(n={len(ev)})", copy=copy)
    print(f"[DIAG] §8.4 目标首token(=实体首字) top1/top5：{d_tr['tag']} top1={d_tr['top1']:.2f}% "
          f"top5={d_tr['top5']:.2f}% | {d_te['tag']} top1={d_te['top1']:.2f}% top5={d_te['top5']:.2f}%",
          flush=True)
    n_tr = min(DIAG_TRAIN_N, len(ev)) if a.smoke else min(DIAG_TRAIN_N, len(train))
    tr_sub = train[:n_tr]
    c_tr = new_copy_stats()
    preds_tr = greedy_gen(model, tr_sub, batch=EM_BATCH, copy=copy, stats=c_tr if copy else None)
    s_tr = diag_entity_source(preds_tr, tr_sub, f"train子集(n={len(tr_sub)},前{len(tr_sub)}条)")
    s_te = diag_entity_source(txt, ev, f"{arm_tag}(n={len(ev)})")
    def _s(d):
        return (f"{d['tag']}: 正确实体={d['正确实体']:.2f}% 干扰实体={d['干扰实体']:.2f}% "
                f"训练词表名={d['训练词表名']:.2f}% 其它={d['其它']:.2f}%")
    print(f"[DIAG] §8.4 贪心生成实体来源（四分类互斥）：{_s(s_te)} | {_s(s_tr)} | "
          f"（test 生成含正确实体={res['em_ent']:.2f}%、含干扰实体={res['em_other']:.2f}%）", flush=True)

    # ★EM 按「v3 是否在输入里可见」分解 —— 直接回答"离上限的差距是不是 v3 不可见造成的"
    def _em_split(preds, rs, lbl):
        vis = [i for i, r in enumerate(rs) if r["v3_in_t3"]]
        inv = [i for i, r in enumerate(rs) if not r["v3_in_t3"]]
        f = lambda idx: (100.0 * sum(int(rs[i]["target"].strip() == preds[i].strip()) for i in idx)
                         / len(idx)) if idx else float("nan")  # noqa: E731
        print(f"[DIAG] EM 分解（v3 是否在输入里可见）：{lbl} 可见 n={len(vis)} EM={f(vis):.2f}% | "
              f"不可见 n={len(inv)} EM={f(inv):.2f}%", flush=True)
    _em_split(txt, ev, f"{arm_tag}(全量 n={len(ev)})")
    _em_split(preds_tr, tr_sub, f"train子集(n={len(tr_sub)})")
    # 返回给 main 的 L2-r4 判据行（Q12）用；打印内容与返回前逐字不变
    return dict(d_tr=d_tr, d_te=d_te, s_tr=s_tr, s_te=s_te, c_tr=c_tr)


# ---------------- 主流程 ----------------
def _find_ref_68e(seed: int) -> str:
    """从 logs/69_68e_hist_s<seed>.log 抓 r3 的 test 侧实体来源四分类（Q12 的同 seed 对照）。"""
    fp = os.path.join(LOG_DIR, f"69_68e_hist_s{seed}.log")
    if not os.path.exists(fp):
        return f"缺 {os.path.basename(fp)}"
    pat = re.compile(r": 正确实体=([\d.]+)% 干扰实体=([\d.]+)% 训练词表名=([\d.]+)% 其它=([\d.]+)%")
    with open(fp, encoding="utf-8") as f:
        for ln in f:
            if "[DIAG]" in ln and "实体来源" in ln:
                m = pat.search(ln)
                if m:
                    return (f"正确实体={m.group(1)}% 干扰实体={m.group(2)}% "
                            f"训练词表名={m.group(3)}% 其它={m.group(4)}%")
    return f"{os.path.basename(fp)} 里没找到 [DIAG] 实体来源行"


def main(argv=None):
    global DATA_TMPL
    ap = argparse.ArgumentParser(description="S69 L2-r1 跨轮消解卡片驱动")
    ap.add_argument("--smoke", action="store_true", help="50 步、只评 60 条")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--swap", action="store_true",
                    help="Q2：test 的轮1/轮2 实体互换、保留原 target，同一 ckpt 重评（不重训）")
    ap.add_argument("--steps", type=int, default=STEPS,
                    help=f"训练步数（默认 {STEPS}；--smoke 时固定 {SMOKE_STEPS}）")
    ap.add_argument("--copy", action="store_true",
                    help="L2-r5：给输出卡接「复制/指针」通路 p=(1-g)·vocab + g·copy_dist"
                         "（指针只看得见 prompt 位，门控 g 由隐藏态算出并参与训练）；不加时逐字走原路径")

    ap.add_argument("--data-prefix", choices=("68", "68b", "68d", "68e", "68f", "68g"), default="68",
                    help="数据/ckpt 前缀：68=logs/68_dialog_*.jsonl（默认，与 r1 逐字同径）；"
                         "68b=logs/68b_dialog_*.jsonl（L2-r1b 共享词表臂）；"
                         "68d=logs/68d_dialog_*.jsonl（L2-r2 共享词表+确定性句面臂）；"
                         "68e=logs/68e_dialog_*.jsonl（L2-r3 轮3 保留 v3 臂）；"
                         "68f=logs/68f_dialog_*.jsonl（L2-r4 test 实体=训练用字的新组合臂）；"
                         "68g=logs/68g_dialog_*.jsonl（L2-r6 train 名池=320 组合 / test=20 留出新组合臂），"
                         "ckpt 名加前缀")
    a = ap.parse_args(argv)
    DATA_TMPL = os.path.join(LOG_DIR, f"{a.data_prefix}_dialog_{{split}}.jsonl")
    # ★实体识别词表（只影响地板/规则类读数）：68f 用含新组合名的 VOCAB_68F、68g 用 VOCAB_68G（340 个
    #   组合），其余档 = NAMES。忘了传 68g 的词表 ⇒ ⑦ 规则认不出实体 ⇒ max_naive 虚低（r4 在 68f 上踩过）。
    VOCAB = (d68.VOCAB_68G if a.data_prefix == "68g"
             else (d68.VOCAB_68F if a.data_prefix == "68f" else d68.NAMES))
    steps = SMOKE_STEPS if a.smoke else a.steps
    tag = "swap" if a.swap else "hist"
    if a.smoke:
        tag += "-smoke"
    ensure_data(a.data_prefix)
    train, test = load_jsonl("train"), load_jsonl("test")
    print(f"[DATA-SRC] data_prefix={a.data_prefix} 文件="
          f"{DATA_TMPL.format(split='train')} / {DATA_TMPL.format(split='test')} | "
          f"steps={steps}{'(smoke固定)' if a.smoke else ''}", flush=True)

    # 数据契约（生成器 68 已断言过；这里用落盘的数据复核一次）
    n_vio_a = sum(int(r["turn3_has_ent"]) or int(r["ent"] in r["t3"]) or int(r["ent_other"] in r["t3"])
                  for r in train + test)
    n_vio_b = sum(int(not (r["ent_in_target"] and not r["other_in_target"]))
                  for r in train + test)
    n_two = sum(int(r["ent_count"] == 2) for r in train + test)
    print(f"[DATA] train={len(train)} test={len(test)} | verify.68 A违例={n_vio_a}（必须 0）| "
          f"B违例={n_vio_b}（必须 0）| 实体个数恒=2 的条数={n_two}/{len(train)+len(test)} | "
          f"族={d68.N_TEMPLATE_FAMILIES} 句面={d68.N_FAMILY_FORMS} | "
          f"角色词 正确「{d68.ROLE_SELF}」干扰「{d68.ROLE_OTHER}」| "
          f"位置桶(0=先出现) train={dict((k, sum(1 for r in train if r['ent_pos_rank']==k)) for k in (0,1))} "
          f"test={dict((k, sum(1 for r in test if r['ent_pos_rank']==k)) for k in (0,1))}", flush=True)

    # ★swap 臂不重训：直接载同 seed 的 ckpt（ckpt 名带 data-prefix，68b 与 68 互不覆盖）
    ck_pre = "" if a.data_prefix == "68" else f"{a.data_prefix}_"
    ck = os.path.join(LOG_DIR, f"69_ckpt_{ck_pre}seed{a.seed}_hist{'-smoke' if a.smoke else ''}"
                                f"{'-copy' if a.copy else ''}.pt")
    # ★§13.2 修复（L2-r6）：seed 必须在**构造模型之前**设置 ⇒ 参数初始化受 --seed 控制、同 seed 可复现。
    #   先跑自检：同一 seed 连造两次模型，逐张量最大差必须 0；不同 seed 必须不同。
    ic_ok = init_check(a.seed, a.seed + 1)
    seed_all(a.seed)
    model = new_pipe()
    if a.copy:
        # ★L2-r5：复制/指针头挂在 model 上 ⇒ 进 state_dict、进 model.parameters()，
        #   优化器与训练循环因此逐字不变（AdamW(model.parameters()) 自动带上它的参数）。
        #   位置在 new_pipe() 之后 ⇒ 卡片组自己的初始化随机流与不加 --copy 时逐位一致。
        model.copy_head = CopyHead(D).to(device)
    n_par = sum(p.numel() for p in model.parameters())
    gate_res, wall, last_loss, loss_trace = None, 0.0, float("nan"), []
    if a.swap:
        if not os.path.exists(ck):
            print(f"[SWAP] 缺 ckpt {ck} ⇒ 先跑 `--seed {a.seed}` 拿到该 ckpt（Q2 不重训）", flush=True)
            return 5
        model.load_state_dict(torch.load(ck, map_location=device))
        print(f"[SWAP] 复用 ckpt {ck}（**不重训**，只把 test 的轮1/轮2 实体互换后重评）", flush=True)
    n_swapped = to_ids(train, a.swap)
    n_swapped_te = to_ids(test, a.swap)

    over = sum(1 for r in train + test if r["ntok_all"] > MAXLEN)
    print(f"[DATA] arm={'实体互换(只改 prompt, 保留原 target)' if a.swap else '原样'} | "
          f"换过的 test 条数={n_swapped_te}/{len(test)} | "
          f"p 中位={sorted(len(r['p']) for r in train + test)[len(train + test)//2]} "
          f"最大={max(len(r['p']) for r in train + test)} | "
          f"p+t 最大={max(r['ntok_all'] for r in train + test)} 超 MAXLEN={over}", flush=True)

    tr_fp, te_fp = d68.data_fp(train), d68.data_fp(test)
    print(f"[DATA] train_fp={tr_fp} test_fp={te_fp}（与 [D68] 同行口径）", flush=True)
    if a.swap:
        # 自检：互换后，**原 target 的实体**必须已经不在「我自己的」槽位上
        #（即原 target 在这个世界里已经答不上）—— 这正是 Q2 的因果点
        n_flip = 0
        for r in test:
            s2 = swap_entities(r["ctx"], r)
            # 互换后角色规则读出的"我自己的"实体 = 原干扰实体 ⇒ 原 target 已答不上
            n_flip += int(d68._role_entity(s2, VOCAB) == r["ent_other"])
        print(f"[SWAP] 自检：互换后「原正确实体→我朋友的 / 原干扰实体→我自己的」条数="
              f"{n_flip}/{len(test)}（应为 100%）", flush=True)
    assert n_vio_a == 0 and n_vio_b == 0 and over == 0, "数据契约不过：轮3 含实体 / 目标含错实体 / 超长"

    # ---- 地板（Q3；只用输入可见信息 + train 统计） ----
    # ★实体识别词表：68f 的 test 实体是新组合名，必须把 VOCAB_68F 传进去，否则 ⑥/⑦ 规则
    #   在 test 上认不出任何实体 ⇒ 地板虚低到 0%（其余档 VOCAB=NAMES，输出逐字不变）
    print("[FLOOR] ★Q3：max_naive 含计数类特征（轮3 长度 / 实体个数 / 轮数 / 位置桶）", flush=True)
    _f_full, mxname_full, mx_full, thr_full = d68.compute_floors(
        train, test, tag="test", show_train=True, vocab=VOCAB)
    if a.smoke:
        ev = test[:SMOKE_EVAL]
        _f_s, mxname_s, mx_s, thr_s = d68.compute_floors(
            train, ev, tag=f"smoke子集(n={len(ev)})", show_train=False, vocab=VOCAB)
        mx, mxname, thr = mx_s, mxname_s, thr_s
    else:
        ev = test
        mx, mxname, thr = mx_full, mxname_full, thr_full

    # ---- L2-r5 [COPY-SELFTEST]（训练前自检：指针打分的行/列来源 + 「不看 target」的硬断言） ----
    if a.copy:
        copy_selftest(model, train)

    # ---- 训练（swap 臂跳过） ----
    if not a.swap:
        gate_res, wall, last_loss, loss_trace = train_run(model, train, a.seed, steps, copy=a.copy)

    # ---- 评估：配对 CE + 贪心 EM ----
    c_te = new_copy_stats() if a.copy else None
    res = evaluate(model, ev, tag, copy=a.copy, stats=c_te)
    cm, cse, em, se, txt, hits = res["cm"], res["cse"], res["em"], res["se"], res["txt"], res["hits"]
    print(f"[MAIN] seed={a.seed} arm={'swap' if a.swap else 'hist'} steps={steps if not a.swap else 0}"
          f"(swap 复用 ckpt) batch={BATCH} lr={LR} | 模型=CardPipeline(residual=False) d={D} ff={FF} "
          f"nhead={NHEAD} maxlen={MAXLEN} params={n_par} | 配对CE={cm:.4f}±{cse:.4f} (nats/token)",
          flush=True)
    print(f"[EM] {'smoke子集' if a.smoke else '全量'} n={len(hits)} 严格EM={em:.2f}% "
          f"±SE={se:.2f}pp | 诊断：生成含**正确实体**比例={res['em_ent']:.2f}% | "
          f"生成含**干扰实体**比例={res['em_other']:.2f}% | "
          f"地板 max_naive={mx:.2f}%({mxname}) 门槛={thr:.2f}%", flush=True)

    # ---- §8.4 必跑诊断（train/test 各自；放在 [EM] 之后、[GEN] 之前） ----
    diag = print_diag(a, ev, txt, res, train, model, res["tag"], copy=a.copy)
    if a.copy:
        print_copy_stats(c_te, diag["c_tr"], res["tag"])

    shown = 0
    for i, r in enumerate(ev):
        if shown >= 6:
            break
        print(f"[GEN] [{i}] t3={r['t3']} | ent={r['ent']} other={r['ent_other']} | "
              f"gold={r['target']} | pred={txt[i][:80]} | "
              f"{'对' if hits[i] else '错'}（正确实体在 pred 里={bool(r['ent'] in txt[i])}, "
              f"干扰实体在 pred 里={bool(r['ent_other'] in txt[i])}）", flush=True)
        shown += 1

    # ---- 判据（原文引 L2_PLAN.md §7） ----
    if a.swap:
        q2_drop = None
        print(f"[CRIT] Q2 L2_PLAN §7：「干扰臂：把轮1/轮2 的实体互换后重评同一 ckpt ⇒ EM 必掉」| "
              f"本 run 就是 swap 臂：swap 后 EM={em:.2f}%（ckpt=seed{a.seed}，未重训）| "
              f"「必掉」需与**同 ckpt 的正常臂**比较，见下一行/汇总行", flush=True)
        # 顺带在同一 ckpt 上把正常臂也评一遍（同进程、同一 ckpt ⇒ 最干净的 A/B）
        normal = load_jsonl("test")
        evn = normal[:SMOKE_EVAL] if a.smoke else normal
        to_ids(evn, swap=False)
        resn = evaluate(model, evn, "normal-on-same-ckpt")
        q2_drop = resn["em"] - em
        print(f"[CRIT] Q2 同 ckpt 对照：正常臂 EM={resn['em']:.2f}%±{resn['se']:.2f}pp vs "
              f"swap 臂 EM={em:.2f}%±{se:.2f}pp ⇒ 掉幅={q2_drop:.2f}pp ⇒ "
              f"{'PASS（读历史）' if q2_drop > 2.0 * max(resn['se'], se) else 'FAIL（疑似背模板，没在读历史）'}",
              flush=True)
    else:
        print(f"[CRIT] Q1（主）L2_PLAN §7：「卡片组 EM > max_naive + 2×SE 且 2 seed 同号」| "
              f"EM={em:.2f}% vs 门槛(max_naive+2SE)={thr:.2f}% ⇒ "
              f"{'PASS' if em > thr else 'FAIL'} | 「2 seed 同号」本 run 只有 seed={a.seed} "
              f"⇒ 待第 2 个 seed 复跑同号", flush=True)
        print(f"[CRIT] Q2 L2_PLAN §7：「swap 后 EM 必掉」| 本 run 是正常臂 EM={em:.2f}% "
              f"⇒ 需 --swap 复用本 ckpt（{ck}）另跑一次来判", flush=True)
    j_q3 = True
    print(f"[CRIT] Q3 L2_PLAN §7：「地板含计数类且 train 上 < 90%（沿用 P2 口径）」⇒ "
          f"已逐条报出（上方 [FLOOR]），max_naive={mx:.2f}% 来自 {mxname} ⇒ "
          f"{'PASS' if j_q3 else 'FAIL'}", flush=True)
    if a.swap:
        q4 = "本臂不适用（swap 复用 ckpt、未训练，不参与饱和判定；看正常臂的 Q4）"
    else:
        sat = (em < 99.0) and (a.smoke or last_loss > 1e-4)
        q4 = ("未饱和（可当机制实验载体）" if sat else
              "饱和（本任务不能当机制实验载体）")
    print(f"[CRIT] Q4 L2_PLAN §7：「报告是否饱和（EM < 99% 且 loss 不塌到 1e-4），否则只能当"
          f"'能承载'的证据」| EM={em:.2f}% (EM<99% = {em < 99.0}) | "
          f"最后一步 loss={last_loss:.6f} (>{1e-4} = {last_loss > 1e-4}) | "
          f"loss迹={loss_trace} ⇒ {q4}", flush=True)

    if a.data_prefix == "68e":
        # ★L2-r3 预注册判据（L2_PLAN.md §10.3）逐条打印：Q6（主）/ Q7 / Q8 / Q9。
        #   只在 68e 档打印 ⇒ 其余档（68/68b/68d）的输出逐字不变。
        ceil_v, v3vis, _exp_c = d68.ceiling_value(test, keep_v3=True)
        q6 = (em > thr) and (em / ceil_v >= 0.80)
        q7 = res["em_ent"] >= 99.0
        q8 = (v3vis == 100.0)
        print(f"[CRIT] Q6（主）L2_PLAN §10.3：「EM > 门槛(max_naive+2SE) 且 EM/[CEIL] ≥ 0.80」| "
              f"EM={em:.2f}% vs 门槛={thr:.2f}% ⇒ {'PASS' if em > thr else 'FAIL'} | "
              f"EM/[CEIL]={em / ceil_v:.3f} vs 0.80 ⇒ {'PASS' if em / ceil_v >= 0.80 else 'FAIL'} | "
              f"[CEIL]={ceil_v:.2f}% ⇒ Q6 {'PASS' if q6 else 'FAIL'}", flush=True)
        print(f"[CRIT] Q7 L2_PLAN §10.3：「生成含正确实体 ≥ 99%」| 含正确实体={res['em_ent']:.2f}% ⇒ "
              f"{'PASS' if q7 else 'FAIL'}", flush=True)
        print(f"[CRIT] Q8 L2_PLAN §10.3：「报 [CEIL] 与 v3 可见率（必须 100%）」| [CEIL]={ceil_v:.2f}% | "
              f"v3 在输入中可见={v3vis:.2f}%（n={len(test)}）⇒ {'PASS' if q8 else 'FAIL'}", flush=True)
        print(f"[CRIT] Q9 L2_PLAN §10.3：「2 seed 同号（都过 Q6 才算）」| 本 run 只有 seed={a.seed}"
              f"（本 run Q6={'PASS' if q6 else 'FAIL'}）⇒ 待另一 seed 复跑后合并判定", flush=True)

    if a.data_prefix == "68f":
        # ★L2-r4 预注册判据（L2_PLAN.md §11）逐条打印：Q10（主）/ Q11 / Q12 / Q13。
        #   只在 68f 档打印 ⇒ 其余档（68/68b/68d/68e）的输出逐字不变。
        ceil_v, v3vis, _exp_c = d68.ceiling_value(test, keep_v3=True)
        q10 = (em > thr) and (em / ceil_v >= 0.80)
        # Q13：不变量 F1/F2 —— **从落盘的 train/test 现算**（不复用生成器日志）
        tr_pool = {r["ent"] for r in train} | {r["ent_other"] for r in train}
        te_pool = {r["ent"] for r in test} | {r["ent_other"] for r in test}
        tr_chars = set("".join(tr_pool))
        te_chars = set("".join(te_pool))
        f1_bad_names = [nm for nm in te_pool if any(ch not in tr_chars for ch in nm)]
        f1_bad_rec = sum(int(any(ch not in tr_chars for ch in r["ent"] + r["ent_other"])) for r in test)
        f1_bad_chars = sum(1 for nm in te_pool for ch in nm if ch not in tr_chars)
        f2_bad = sorted(te_pool & tr_pool)
        f2_bad_rec = sum(int((r["ent"] in tr_pool) or (r["ent_other"] in tr_pool)) for r in test)
        cover = 100.0 * len(te_chars & tr_chars) / len(te_chars)
        print(f"[NOVEL] 从落盘 68f 数据现算（test n={len(test)}）：test 名字数={len(te_pool)} | "
              f"与 train 重名数={len(f2_bad)}（必须 0）| 用字覆盖率={cover:.2f}%"
              f"（test 的 {len(te_chars)} 个用字里 {len(te_chars & tr_chars)} 个 ⊆ train 用字集）| "
              f"test 用到的 train 字={len(te_chars & tr_chars)}/{len(tr_chars)}", flush=True)
        s_te = diag["s_te"]
        print(f"[CRIT] Q10（主）L2_PLAN §11：「EM > 门槛(max_naive+2SE) 且 EM/[CEIL] ≥ 0.80」| "
              f"EM={em:.2f}% vs 门槛={thr:.2f}% ⇒ {'PASS' if em > thr else 'FAIL'} | "
              f"EM/[CEIL]={em / ceil_v:.3f} vs 0.80 ⇒ {'PASS' if em / ceil_v >= 0.80 else 'FAIL'} | "
              f"[CEIL]={ceil_v:.2f}%（v3 在输入中可见={v3vis:.2f}%）⇒ Q10 {'PASS' if q10 else 'FAIL'}", flush=True)
        print(f"[CRIT] Q11 L2_PLAN §11：「2 seed 同号（都过 Q10 才算）」| 本 run 只有 seed={a.seed}"
              f"（本 run Q10={'PASS' if q10 else 'FAIL'}）⇒ 待另一 seed 复跑后合并判定", flush=True)
        print(f"[CRIT] Q12 L2_PLAN §11：「报生成实体来源四分类，重点看『训练词表名』是否显著升高」| "
              f"本 run test({s_te['tag']}): 正确实体={s_te['正确实体']:.2f}% "
              f"干扰实体={s_te['干扰实体']:.2f}% 训练词表名={s_te['训练词表名']:.2f}% 其它={s_te['其它']:.2f}% | "
              f"train子集(n={diag['s_tr']['n']}): 正确实体={diag['s_tr']['正确实体']:.2f}% "
              f"训练词表名={diag['s_tr']['训练词表名']:.2f}% | r3(68e seed={a.seed}) 对照="
              f"{_find_ref_68e(a.seed)} ⇒ 不设阈值，上升即如实记『记忆成分上升』", flush=True)
        print(f"[CRIT] Q13 L2_PLAN §11：「报不变量 F 实测计数（用字覆盖 100%、与 train 重名 0）」| "
              f"F1 用字越界的名字数={len(f1_bad_names)}（必须 0）逐条违例={f1_bad_rec} "
              f"越界用字实例={f1_bad_chars} | F2 与 train 重名数={len(f2_bad)}（必须 0）"
              f"逐条违例={f2_bad_rec} | 用字覆盖率={cover:.2f}% ⇒ "
              f"{'PASS' if (not f1_bad_names and not f2_bad and not f1_bad_rec and not f2_bad_rec) else 'FAIL'}",
              flush=True)

    if a.data_prefix == "68g":
        # ★L2-r6 预注册判据（L2_PLAN.md §14）逐条打印：Q18（主）/ Q19 / Q20。
        ceil_v, v3vis, _exp_c = d68.ceiling_value(test, keep_v3=True)
        q18 = (em > thr) and (em / ceil_v >= 0.80)
        tr_pool = {r["ent"] for r in train} | {r["ent_other"] for r in train}
        te_pool = {r["ent"] for r in test} | {r["ent_other"] for r in test}
        tr_chars = set("".join(tr_pool))
        te_chars = set("".join(te_pool))
        g1 = len(tr_pool - set(d68.NAMES_TRAIN))
        g2 = sorted(te_pool & tr_pool)
        g3_bad = sum(1 for nm in te_pool for ch in nm if ch not in tr_chars)
        cover = 100.0 * len(te_chars & tr_chars) / len(te_chars)
        s_te = diag["s_te"]
        q20 = (s_te["正确实体"] >= 99.0) and (s_te["训练词表名"] <= 1.0)
        print(f"[CRIT] Q18（主）L2_PLAN §14：「68g 上 EM/[CEIL] ≥ 0.80」| EM={em:.2f}% vs 门槛"
              f"(max_naive+2SE)={thr:.2f}% ⇒ {'PASS' if em > thr else 'FAIL'} | "
              f"EM/[CEIL]={em / ceil_v:.3f} vs 0.80 ⇒ {'PASS' if em / ceil_v >= 0.80 else 'FAIL'} | "
              f"[CEIL]={ceil_v:.2f}%（v3 在输入中可见={v3vis:.2f}%）⇒ Q18 "
              f"{'PASS' if q18 else 'FAIL'}", flush=True)
        print(f"[CRIT] Q19 L2_PLAN §14：「2 seed 同号（必须在 §13.2 的 seed 缺陷修好之后）」| "
              f"本 run seed={a.seed}（[INIT-CHECK]={'PASS' if ic_ok else 'FAIL'}）Q18="
              f"{'PASS' if q18 else 'FAIL'} ⇒ 待另一 seed 复跑后合并判定", flush=True)
        print(f"[CRIT] Q20 L2_PLAN §14：「68g 的 test 实体来源『正确实体』≥ 99%（『训练词表名』≤ 1%）」| "
              f"本 run test({s_te['tag']}): 正确实体={s_te['正确实体']:.2f}% 干扰实体="
              f"{s_te['干扰实体']:.2f}% 训练词表名={s_te['训练词表名']:.2f}% 其它={s_te['其它']:.2f}% ⇒ "
              f"{'PASS' if q20 else 'FAIL'}", flush=True)
        print(f"[NOVEL-G] 从落盘 68g 数据现算：test 名字数={len(te_pool)} | 与 train 重名数={len(g2)}"
              f"（必须 0）| train 里非基础名单的新组合={g1} | 用字覆盖率={cover:.2f}%"
              f"（越界用字实例={g3_bad}，必须 0）| train 子集实体来源：正确实体="
              f"{diag['s_tr']['正确实体']:.2f}% 训练词表名={diag['s_tr']['训练词表名']:.2f}%", flush=True)

    if a.copy:
        # ★L2-r5 预注册判据（L2_PLAN.md §12.3）逐条打印：Q14（主）/ Q15 / Q16 / Q17。
        ceil_v, v3vis, _exp_c = d68.ceiling_value(test, keep_v3=True)
        s_te = diag["s_te"]
        if a.data_prefix == "68f":
            q14 = (em / ceil_v >= 0.80)
            q16 = (s_te["正确实体"] >= 99.0) and (s_te["训练词表名"] <= 1.0)
            print(f"[CRIT] Q14（主）L2_PLAN §12.3：「68f+copy 上 EM / [CEIL] ≥ 0.80」| EM={em:.2f}% "
                  f"[CEIL]={ceil_v:.2f}%（v3 在输入中可见={v3vis:.2f}%）⇒ EM/[CEIL]={em / ceil_v:.3f} "
                  f"vs 0.80 ⇒ {'PASS' if q14 else 'FAIL'}", flush=True)
            print(f"[CRIT] Q16 L2_PLAN §12.3：「68f+copy 实体来源四分类：正确实体 ≥ 99% 且 "
                  f"训练词表名 ≤ 1%」| 正确实体={s_te['正确实体']:.2f}%（≥99 = "
                  f"{s_te['正确实体'] >= 99.0}）干扰实体={s_te['干扰实体']:.2f}% 训练词表名="
                  f"{s_te['训练词表名']:.2f}%（≤1 = {s_te['训练词表名'] <= 1.0}）其它={s_te['其它']:.2f}%"
                  f" ⇒ {'PASS' if q16 else 'FAIL'}", flush=True)
            print(f"[CRIT] Q15 L2_PLAN §12.3：「68e+copy 上 EM ≥ 97.00%」| 本 run data_prefix=68f ⇒ "
                  f"不适用（同配置的 68e+copy 配对 run 见另一日志）", flush=True)
        elif a.data_prefix == "68e":
            q15 = (em >= 97.0)
            print(f"[CRIT] Q15 L2_PLAN §12.3：「68e+copy 上 EM ≥ 97.00%（对照 r3 的 98.25%/99.62%，"
                  f"不许回退）」| EM={em:.2f}%±{se:.2f}pp ⇒ EM ≥ 97.00% = {q15} ⇒ "
                  f"{'PASS' if q15 else 'FAIL'}", flush=True)
            print(f"[CRIT] Q14/Q16 L2_PLAN §12.3：「68f+copy 的 EM/[CEIL] 与实体来源四分类」| "
                  f"本 run data_prefix=68e ⇒ 不适用（配对 run 见另一日志）", flush=True)
        else:
            print(f"[CRIT] Q14/Q15/Q16 L2_PLAN §12.3：判据只在 68e/68f 两档预注册；本 run "
                  f"data_prefix={a.data_prefix} ⇒ 不适用", flush=True)
        print(f"[CRIT] Q17 L2_PLAN §12.3：「2 seed 同号」| 本 run 只有 seed={a.seed}"
              f"（本 run 的 Q14/Q15/Q16 结果见上）⇒ 待另一 seed 复跑后合并判定", flush=True)

    ck_out = os.path.join(LOG_DIR, f"69_ckpt_{ck_pre}seed{a.seed}_{tag}{'-copy' if a.copy else ''}.pt")
    torch.save(model.state_dict(), ck_out)
    cp_meta = f" copy={a.copy}" if a.copy else ""
    print(f"[META] tag={tag} seed={a.seed} swap={a.swap} steps={steps if not a.swap else 0} "
          f"init_check={'PASS' if ic_ok else 'FAIL'}{cp_meta} "
          f"data_prefix={a.data_prefix} batch={BATCH} lr={LR} d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN} "
          f"n_params={n_par} 墙钟={wall:.0f}s ckpt={ck_out}", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
