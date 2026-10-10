#!/usr/bin/env python3
"""S75 · L3-r3b「选择卡」三臂对照（A / B1 / B2）+ L4 反事实 + 配对判据。

数据（只读，不造）：`logs/74_dialog_{train,test}.jsonl`（k=4，7000/800）/ `logs/78_*`（k=8）。
  字段 = §9 全部字段 + cands / gold_idx / n_cands；本文件只用
  ctx / target / cands / gold_idx / n_cands / family / ent / cond / t3 / t3_idx / tpl_id。

★三臂实现方式（L3_PLAN.md §11.3；参数对齐方式写在这里，实测数字见 [PARAMS]）
  共用：d=128 ff=512 nhead=4 maxlen=512 dropout=0.1、外层残差关闭（cards/target.py:28-30 的实测结论）；
        输入卡（emb + 1 层 encoder）与输出卡（head）三臂逐字相同；batch=32 lr=1e-3 步数相同。
  A  （单卡直出）      ：input(1 层) → target(2 层「think」) → output；只有回复损失。
  B1 （纯架构拆分，负对照）：input(1 层) → 选择卡(1 层) → 重整卡(1 层) → output；
                        选择卡**没有自己的损失**，梯度只能经重整卡回流。
  B2 （带可判中间目标）  ：与 B1 结构逐字相同，额外加 λ·CE(选择打分, gold_idx)（λ=1.0）。
  ⇒ ★参数对齐做法 = 「把单卡 A 的 2 层目标层切成两半，分给 B 的选择卡与重整卡」：
      A = input(1)+target(2) = 3 个 encoder 层；B = input(1)+select(1)+realign(1) = 3 个 encoder 层。
      三臂构造顺序一致（input → 第 1 层 → 第 2 层 → output）；选择打分向量 q 用**零初始化**
      （不消耗 RNG）⇒ 同 seed 下三臂的共享子模块**逐位相同**（[INIT-CHECK] 实测判定），
      配对检验所需的「同初始化」前提由此成立。参数差只有 q 的 128 个（约 +0.005%），
      步数 = 3000×batch32 = 96000 样本，三臂完全相同（[PARAMS] 并列报实际数字）。
  选择卡怎么出「选中哪一对」：取候选行里每个候选**末位 token** 的隐藏态 h_j，打分 s_j = q·h_j（q 为 d 维），
      softmax 得 p ∈ R^k（k 之外的 logit 置 -inf）；选择向量 e_sel = Σ_j p_j h_j（soft 选择、可微）；
      重整卡把 e_sel **加性注入回复位置**（位置 >= ctx 长，即「重整卡看到 ctx + 选中的候选」）。
      生成/评测一律用模型自己的 p（**不是**金标）；选择准确率 = argmax p 命中的候选对 == 历史的 (ent, cond)。
      B1 的梯度只能经重整卡回流到选择卡（[GATE A] 实测选择卡 max|grad| > 0）；B2 额外有选择 CE。
  ★修 §9.7 的 init 缺陷：manual_seed **移到构造之前** + [INIT-CHECK]（同 seed 两次构造 hash 相同、
      逐张量差 0；异 seed hash 不同作阴性对照）。修好之前不跑三臂、不报 R3-b。

判据（L3_PLAN.md §11.3 / §11.4 / §11.6，逐条打印 [CRIT]，不为好看而改口径）
  R3-b（主）：B2 的 EM > A，且**配对** Δ >= 2SE_Δ（d_i = 1[臂X 对 i] − 1[臂Y 对 i]，SE_Δ = std(d)/sqrt(n)；
      **不用** sqrt(SE_X²+SE_Y²) 那个独立样本公式，见 §11.4），2 seed 同号；
      同时打印 McNemar 精确检验（不一致对 b/c 与双侧 p）。
  R3-c（L4 反事实，三臂都做）：①打乱候选顺序 ⇒ 回复应当不变（报一致率 + 配对 Δ；显著下降 ⇒ 位置泄漏、结论作废）；
      ②-a 替换掉正确候选（换成历史不一致的对，k 不变）；②-b 删除正确候选（k→k-1）⇒ 回复必变、
      且 EM 落到地板附近。
  R3-d（必报）：B1/B2 选择卡的**选择准确率**（选对候选对，不看句面）+ 三臂可训参数量 / 总步数 / 峰值显存并列。
  门槛（§11.6）：k=4 = max_chance 27.12% + 2SE；k=8 = 13.12% + 2SE（**不是** 0.81%）；捷径级 max_naive=0.38%。
  归因规则（§11.3）：B2>A 且 B1≈A ⇒ 增益来自**可判中间目标**（不是「分卡」）；B1>A ⇒ 与项目②
      「卡边界零增量」冲突，先排实现差异，**不得**直接宣称「分卡有增益」；B2≈A ⇒ 如实写「按 L3 口径无增益」。

用法（本机只有 .venv/bin/python 装了 torch/tokenizers）：
  .venv/bin/python stages/75_cand_cards.py --param-table                        # 参数对齐表（CPU，不占 GPU）
  .venv/bin/python stages/75_cand_cards.py --arm A  --seed 1234 --smoke --cf     # 探路
  .venv/bin/python stages/75_cand_cards.py --arm B2 --seed 1234 --cf --pair-hits logs/75_hits_A_s1234.json
  .venv/bin/python stages/75_cand_cards.py --report-pairs                       # 读数齐后出配对判词（CPU）
GPU 长跑必须：systemd-run --user --unit=<名> --collect
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 ... .venv/bin/python ...
只允许写本文件与 logs/。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

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

from cards import Card, InputCard, OutputCard, encoder_layer  # noqa: E402
from cards.base import IO_D_TO_D, state_sha256  # noqa: E402

# ---------------- 配置（与 S65/S67/S71/S73 一致） ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512
STEPS = 3000
LR = 1e-3
BATCH = 32
SMOKE_STEPS = 50
SMOKE_EVAL = 60
DROPOUT = 0.1
SEL_LAMBDA = 1.0                       # B2 选择 CE 的权重（写死）
LOG_DIR = os.path.join(ROOT, "logs")
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
D74_PATH = os.path.join(ROOT, "stages", "74_nl_dialog_cand_gen.py")

PREFIX_BY_K = {4: "74", 8: "78"}
# §11.6 的引文（机会级门槛用；原始行在 L3_PLAN.md §11.6）
MAX_CHANCE_QUOTED = {4: 27.12, 8: 13.12}
MAX_NAIVE_QUOTED = 0.38
# §11.6 引文里的 k=4 指纹（仅供参考性对齐，不作判据）
FP_QUOTED = {"74": ("213a203507787631", "25ce4966054660d0")}

CAND_PREFIX = "候选："
CAND_SEP = "；"
CAND_LETTERS = "ABCDEFGH"


def _load_path(name: str, path: str):
    """按路径加载（文件名以数字开头，不能普通 import）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d74 = _load_path("d74_nl_dialog_cand_gen", D74_PATH)   # EOS / compute_floors / data_fp / file_sha256

tok = None
V = 0
PAD_ID = EOS_ID = UNK_ID = RESP_ID = 0


def load_tok():
    """载入分词器并设全局 V/PAD/EOS（CPU 也要，参数表与配对报表不碰 GPU）。"""
    global tok, V, PAD_ID, EOS_ID, UNK_ID, RESP_ID
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    V = tok.get_vocab_size()
    PAD_ID = tok.token_to_id("<pad>")
    EOS_ID = tok.token_to_id("<eos>")
    UNK_ID = tok.token_to_id("<unk>")
    RESP_ID = tok.token_to_id("<resp>")


def enc(text: str) -> list:
    return tok.encode(text, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


# ---------------- 数据 ----------------
def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


POS_BAD = {"n": 0}


def prep_record(r: dict, with_target: bool = True) -> dict:
    """把文本 → token：p=enc(ctx)、t=enc(target+<eos>)、cend=每个候选**末位 token** 的位置。

    cend 的算法：候选文本在 ctx 里的字符区间 → 该前缀的 token 数 − 1；同时核验「前缀编码稳定」
    （enc(前缀) == enc(ctx)[:len]），违例计数进 POS_BAD（实测，不是推断）。
    """
    r["p"] = enc(r["ctx"])
    if with_target and "t" not in r:
        r["t"] = enc(r["target"] + d74.EOS)
    base = r["ctx"].index(CAND_PREFIX)
    ends = []
    for c in r["cands"]:
        pos = r["ctx"].index(c["text"], base)
        pref = enc(r["ctx"][: pos + len(c["text"])])
        if r["p"][: len(pref)] != pref:
            POS_BAD["n"] += 1
        ends.append(len(pref) - 1)
    r["cend"] = ends
    r["n_cands"] = len(r["cands"])
    return r


def pack(recs, with_target: bool = True):
    """(ids, ctx_len, end_idx, n_cands)：ctx 在左、回复在右，右 padding（不加任何位置偏移）。"""
    K = max(r["n_cands"] for r in recs)
    lens = [len(r["p"]) + (len(r["t"]) if with_target else 0) for r in recs]
    n = max(lens)
    ids = torch.full((len(recs), n), PAD_ID, dtype=torch.long)
    s = torch.zeros(len(recs), dtype=torch.long)
    eidx = torch.zeros((len(recs), K), dtype=torch.long)
    nc = torch.zeros(len(recs), dtype=torch.long)
    for i, r in enumerate(recs):
        ids[i, : len(r["p"])] = torch.tensor(r["p"])
        if with_target:
            ids[i, len(r["p"]): len(r["p"]) + len(r["t"])] = torch.tensor(r["t"])
        s[i] = len(r["p"])
        for j, e in enumerate(r["cend"]):
            eidx[i, j] = e
        nc[i] = r["n_cands"]
    return ids.to(device), s.to(device), eidx.to(device), nc.to(device)


# ---------------- 三臂模型 ----------------
class DeepTargetCard(Card):
    """A 臂的目标部分：把单卡的 1 层 think 堆到 n_layers 层（照 cards/target.py 的 residual=False 口径）。"""

    io = IO_D_TO_D
    card_name = "target_deep"

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT, n_layers: int = 2):
        super().__init__(d, maxlen)
        self.layers = nn.ModuleList([encoder_layer(d, ff, nhead, dropout) for _ in range(n_layers)])

    def apply_card(self, x, mask=None):
        for L in self.layers:
            x = L(x, src_mask=mask)          # residual=False：裸层，不加外层残差
        return x


class SelCard(Card):
    io = IO_D_TO_D
    card_name = "select"

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT):
        super().__init__(d, maxlen)
        self.enc = encoder_layer(d, ff, nhead, dropout)

    def apply_card(self, x, mask=None):
        return self.enc(x, src_mask=mask)


class RealignCard(Card):
    io = IO_D_TO_D
    card_name = "realign"

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT):
        super().__init__(d, maxlen)
        self.enc = encoder_layer(d, ff, nhead, dropout)

    def apply_card(self, x, mask=None):
        return self.enc(x, src_mask=mask)


class ArmA(nn.Module):
    """单卡直出：input(1 层) → target(2 层) → output。"""

    name = "A"
    use_sel_loss = False

    def __init__(self):
        super().__init__()
        self.input = InputCard(D, V, FF, NHEAD, MAXLEN, PAD_ID, DROPOUT)
        self.target = DeepTargetCard()
        self.output = OutputCard(D, V, MAXLEN)

    def forward_all(self, ids, ctx_len=None, end_idx=None, n_cands=None):
        m = self.input.causal_mask(ids)
        h = self.input(ids, mask=m)
        h = self.target(h, mask=m)
        return self.output(h), None

    def groups(self):
        return {"input": self.input, "target2": self.target, "output": self.output}


class ArmB(nn.Module):
    """B1/B2：input(1 层) → 选择卡(1 层) → 重整卡(1 层) → output；B2 额外有选择 CE。"""

    def __init__(self, arm: str = "B1"):
        super().__init__()
        self.name = arm
        self.use_sel_loss = (arm == "B2")
        self.input = InputCard(D, V, FF, NHEAD, MAXLEN, PAD_ID, DROPOUT)
        self.select = SelCard()
        self.realign = RealignCard()
        self.output = OutputCard(D, V, MAXLEN)
        # ★零初始化：不消耗 RNG ⇒ 与 A 臂的 output/head 初值逐位相同（[INIT-CHECK] 实测）
        self.q = nn.Parameter(torch.zeros(D))

    def forward_all(self, ids, ctx_len=None, end_idx=None, n_cands=None):
        assert ctx_len is not None and end_idx is not None and n_cands is not None, \
            "[B 臂] 必须给 ctx_len / end_idx / n_cands"
        n = ids.size(1)
        m = self.input.causal_mask(ids)
        hi = self.input(ids, mask=m)
        hs = self.select(hi, mask=m)
        g = torch.gather(hs, 1, end_idx.unsqueeze(-1).expand(-1, -1, hs.size(-1)))   # (b,k,d)
        sc = (g * self.q).sum(-1)                                                  # (b,k)
        k = sc.size(1)
        valid = torch.arange(k, device=sc.device)[None, :] < n_cands[:, None]
        sc = sc.masked_fill(~valid, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        e_sel = torch.einsum("bk,bkd->bd", p, g)                                   # (b,d) soft 选择
        pos = torch.arange(n, device=ids.device)[None, :]
        apply = (pos >= ctx_len[:, None]).unsqueeze(-1).to(hi.dtype)                # 只注入回复位置
        h = self.realign(hi + apply * e_sel.unsqueeze(1), mask=m)
        return self.output(h), sc

    def groups(self):
        return {"input": self.input, "select": self.select, "realign": self.realign, "output": self.output}

    def sel_pred(self, sc, recs):
        """argmax 选择 + 与「历史 (ent,cond) 一致」的命中判定（不看句面）。"""
        pred = [int(x) for x in sc.argmax(dim=-1).tolist()]
        hits = []
        for i, r in enumerate(recs):
            c = r["cands"][pred[i]] if pred[i] < r["n_cands"] else None
            hits.append(int(c is not None and c["ent"] == r["ent"] and c["cond"] == r["cond"]))
        return pred, hits


def build_arm(arm: str) -> nn.Module:
    """构造顺序三臂一致：input → 第 1 层 → 第 2 层 → output（B 的 q 零初始化不消耗 RNG）。"""
    if arm == "A":
        return ArmA()
    assert arm in ("B1", "B2"), f"未知臂 {arm!r}"
    return ArmB(arm)


def param_groups_named(model) -> dict:
    g = {k: sum(p.numel() for p in mod.parameters()) for k, mod in model.groups().items()}
    if isinstance(model, ArmB):
        g["q"] = model.q.numel()
    g["合计"] = sum(p.numel() for p in model.parameters())
    return g


# ---------------- 训练 / 评测骨架（写法照抄 S73） ----------------
def reply_ce(logits, ids, recs, s) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    nll = torch.zeros((), device=logits.device)
    ntok = 0
    for i, r in enumerate(recs):
        a, e = int(s[i]), int(s[i]) + len(r["t"])
        lp = logp[i, a - 1: e - 1]
        tgt = ids[i, a: e]
        nll = nll - lp.gather(1, tgt.unsqueeze(1)).sum()
        ntok += len(r["t"])
    return nll / ntok


def train_run(model, train, seed: int, steps: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train), generator=gen)
    pos, step, gate = 0, 0, None
    t0 = time.time()
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train), generator=gen)
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        batch = [train[i] for i in idx]
        ids, s, eidx, nc = pack(batch)
        logits, sc = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)
        lce = reply_ce(logits, ids, batch, s)
        loss = lce
        lsel = None
        if model.use_sel_loss:
            gold = torch.tensor([r["gold_idx"] for r in batch], device=device)
            lsel = F.cross_entropy(sc, gold)
            loss = lce + SEL_LAMBDA * lsel
        opt.zero_grad(set_to_none=True)
        loss.backward()
        step += 1
        if step == 20:
            parts = []
            for name, mod in model.groups().items():
                ps = list(mod.parameters())
                got = sum(p.grad is not None for p in ps)
                gm = max((float(p.grad.abs().max()) for p in ps if p.grad is not None), default=0.0)
                parts.append(f"{name}: {got}/{len(ps)} 张量有梯度 max|g|={gm:.2e}")
            if isinstance(model, ArmB):
                qg = model.q.grad
                parts.append(f"q: {'有' if qg is not None else '无'}梯度"
                             + (f" max|g|={float(qg.abs().max()):.2e}" if qg is not None else ""))
                if model.use_sel_loss:
                    pred, hits = model.sel_pred(sc, batch)
                    parts.append(f"batch 选择准确率={100.0 * sum(hits) / len(hits):.2f}%")
                else:
                    selg = max((float(p.grad.abs().max()) for p in model.select.parameters()
                                if p.grad is not None), default=0.0)
                    parts.append(f"选择卡梯度来源=仅回复损失（无中间监督）max|g|={selg:.2e}"
                                 f" ⇒ {'通路活' if selg > 0 else '★通路死'}")
            gate = " | ".join(parts)
            ok = all("0/" not in x for x in parts) and ("通路死" not in gate)
            print(f"[GATE A] step20 {gate} ⇒ {'PASS' if ok else '★CHECK'}", flush=True)
        opt.step()
        if step % 500 == 0 or step == steps:
            el = time.time() - t0
            extra = f" sel={float(lsel):.4f}" if lsel is not None else ""
            print(f"  [train] seed={seed} step={step}/{steps} loss={loss.item():.4f}{extra} "
                  f"ce={lce.item():.4f} elapsed={el:.0f}s eta={el / step * (steps - step):.0f}s", flush=True)
    return gate, time.time() - t0


@torch.no_grad()
def eval_ce(model, recs) -> list:
    model.eval()
    per = []
    for i in range(0, len(recs), BATCH):
        chunk = recs[i: i + BATCH]
        ids, s, eidx, nc = pack(chunk)
        logits = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)[0]
        logp = torch.log_softmax(logits, dim=-1)
        for j, r in enumerate(chunk):
            a, e = int(s[j]), int(s[j]) + len(r["t"])
            lp = logp[j, a - 1: e - 1]
            tgt = ids[j, a: e]
            per.append(float((-lp.gather(1, tgt.unsqueeze(1)).sum() / len(r["t"])).item()))
    model.train()
    return per


@torch.no_grad()
def eval_sel(model, recs) -> list:
    """选择准确率（只喂 ctx；与训练同口径的 p，argmax 命中历史 (ent,cond)）。A 臂返回 None。"""
    if not isinstance(model, ArmB):
        return None
    model.eval()
    hits = []
    for i in range(0, len(recs), BATCH):
        chunk = recs[i: i + BATCH]
        ids, s, eidx, nc = pack(chunk, with_target=False)
        sc = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)[1]
        _pred, h = model.sel_pred(sc, chunk)
        hits += h
    model.train()
    return hits


@torch.no_grad()
def greedy_gen(model, recs) -> list:
    """贪心解码 batch=1（照 S73/S71 的 EM_BATCH=1 自洽口径）；生成时用模型自己的选择（非金标）。"""
    model.eval()
    texts = []
    for r in recs:
        ids = torch.tensor([r["p"]], dtype=torch.long, device=device)
        s = torch.tensor([len(r["p"])], dtype=torch.long, device=device)
        eidx = torch.tensor([r["cend"]], dtype=torch.long, device=device)
        nc = torch.tensor([r["n_cands"]], dtype=torch.long, device=device)
        cap = MAXLEN - len(r["p"])
        gen = []
        for _ in range(cap):
            logits = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)[0]
            t = int(logits[0, -1].argmax().item())
            if t == EOS_ID:
                break
            gen.append(t)
            ids = torch.cat([ids, torch.tensor([[t]], dtype=torch.long, device=device)], dim=1)
        texts.append(dec(gen))
    model.train()
    return texts


# ---------------- 统计口径 ----------------
def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, (math.sqrt(var / n) if n else float("nan"))


def em_strict(gold: str, pred: str) -> int:
    return int((gold or "").strip() == (pred or "").strip())


def em_with_thr(p: float, n: int):
    se = (p / 100.0 * (1 - p / 100.0) / n) ** 0.5 * 100.0 if n else float("nan")
    return se, p + 2.0 * se


def mcnemar_exact(n01: int, n10: int) -> float:
    """McNemar 精确双侧 p（只看不一致对；n01=臂Y对而臂X错，n10=臂X对而臂Y错）。"""
    nd = n01 + n10
    if nd == 0:
        return 1.0
    m = min(n01, n10)
    tail = sum(math.comb(nd, i) for i in range(m + 1)) * (0.5 ** nd)
    return min(1.0, 2.0 * tail)


def paired_stats(hx, hy):
    """§11.4 的配对口径：d_i = 1[臂X 对 i] − 1[臂Y 对 i]；SE_Δ = std(d)/sqrt(n)（不是独立样本公式）。"""
    n = len(hx)
    d = [a - b for a, b in zip(hx, hy)]
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    n01 = sum(1 for a, b in zip(hx, hy) if a == 0 and b == 1)     # X 错、Y 对
    n10 = sum(1 for a, b in zip(hx, hy) if a == 1 and b == 0)     # X 对、Y 错
    return {"delta": 100.0 * m, "se": 100.0 * se, "n01": n01, "n10": n10,
            "p": mcnemar_exact(n01, n10), "n": n}


def paired_line(tag: str, hx, hy, name_x: str, name_y: str):
    st = paired_stats(hx, hy)
    print(f"[PAIR] {tag} {name_x} vs {name_y}：Δ={st['delta']:+.2f}pp SE_Δ(配对)={st['se']:.2f}pp "
          f"2SE_Δ={2 * st['se']:.2f}pp | 不一致对 b({name_x}错{name_y}对)={st['n01']} "
          f"c({name_x}对{name_y}错)={st['n10']} | McNemar 双侧精确 p={st['p']:.3g} (n={st['n']})", flush=True)
    return st


# ---------------- 反事实（R3-c） ----------------
CF_MODE_ID = {"shuffle": 1, "replace": 2, "delete": 3}
def cf_record(r: dict, mode: str, rng: random.Random) -> dict:
    """按模式重建候选行与 ctx；返回新记录（cend/n_cands 由 prep_record 重算）。"""
    cands = [dict(c) for c in r["cands"]]
    gold = r["gold_idx"]
    new, gi = cands, gold
    if mode == "shuffle":
        perm = list(range(len(cands)))
        rng.shuffle(perm)
        new = [cands[j] for j in perm]
        gi = perm.index(gold)
    elif mode == "replace":
        # 正确候选换成「历史不一致」的对：优先选两个槽位都对不上的组合
        pool = []
        for c in cands:
            for c2 in cands:
                pair = (c["ent"], c2["cond"])
                if pair != (r["ent"], r["cond"]):
                    pool.append(pair)
        strong = [p for p in pool if p[0] != r["ent"] and p[1] != r["cond"]]
        pick = rng.choice(strong if strong else pool)
        new = [dict(c) for c in cands]
        new[gold]["ent"], new[gold]["cond"] = pick
        gi = gold
    elif mode == "delete":
        new = [c for j, c in enumerate(cands) if j != gold]
        gi = -1
    else:                                                       # pragma: no cover
        raise AssertionError(f"未知反事实模式 {mode!r}")
    for pos, c in enumerate(new):
        c["letter"] = CAND_LETTERS[pos]
        c["text"] = f"{c['letter']}. {c['ent']} / {c['cond']}"
    line = CAND_PREFIX + CAND_SEP.join(c["text"] for c in new)
    out = dict(r)
    out["ctx"] = r["ctx"].replace(r["cand_line"], line)
    out["cands"] = new
    out["cand_line"] = line
    out["gold_idx"] = gi
    out["n_cands"] = len(new)
    out.pop("t", None)                                          # 回复目标不变，但强制重算 token（ctx 变了）
    return prep_record(out)


def run_cf(model, test, base_hits, base_txt, mode: str, seed: int, tag: str, model_name: str):
    """对同一 ckpt 做一次反事实评测并打印原始行；返回 (hits, txt, sel_hits, 位置移动数)。"""
    rng = random.Random(0xC0FFEE ^ seed ^ CF_MODE_ID[mode])   # ★跨臂确定：同 (mode,seed) 三臂见同一扰动输入
    recs = [cf_record(r, mode, rng) for r in test]
    moved = sum(1 for r0, r1 in zip(test, recs) if r0["gold_idx"] != r1["gold_idx"])
    txt = greedy_gen(model, recs)
    hits = [em_strict(r["target"], txt[i]) for i, r in enumerate(recs)]
    changed = sum(1 for i in range(len(recs)) if txt[i].strip() != base_txt[i].strip())
    em = 100.0 * sum(hits) / len(hits)
    sel = eval_sel(model, recs)
    # 「无正确候选」的违例计数（replace/delete 应当全部无正确候选；shuffle 应当仍有且唯一）
    no_right = sum(1 for r in recs
                   if not any(c["ent"] == r["ent"] and c["cond"] == r["cond"] for c in r["cands"]))
    print(f"[CF] 模式={mode} ({tag}) n={len(recs)} | 正确项位置发生变化={moved} | "
          f"候选集里没有正确对的条数={no_right} | EM={em:.2f}% | 回复文本变化率="
          f"{100.0 * changed / len(recs):.2f}% | 选择准确率(选对候选对)="
          + ("不适用(A 无选择卡)" if sel is None else f"{100.0 * sum(sel) / len(sel):.2f}%"), flush=True)
    return hits, txt, sel, moved, em, changed


# ---------------- 参数表 / 配对报表（CPU） ----------------
def shared_init_check(seed: int) -> dict:
    """三臂同 seed 构造，逐张量核对共享子模块初值是否逐位相同（A 的层 i ↔ B 的 select/realign）。"""
    built = {}
    for arm in ("A", "B1", "B2"):
        torch.manual_seed(seed)
        built[arm] = build_arm(arm)
    sa, sb = built["A"].state_dict(), built["B1"].state_dict()
    pairs = [("input", "input"), ("output", "output")]
    out = {}
    for k, v in sa.items():
        if k.startswith("input."):
            out[f"A.{k} ↔ B.{k}"] = float((v - sb[k]).abs().max().item())
        elif k.startswith("target.layers.0."):
            out[f"A.{k} ↔ B.select.enc.{k[len('target.layers.0.'):]}"] = float(
                (v - sb["select.enc." + k[len("target.layers.0."):]]).abs().max().item())
        elif k.startswith("target.layers.1."):
            out[f"A.{k} ↔ B.realign.enc.{k[len('target.layers.1.'):]}"] = float(
                (v - sb["realign.enc." + k[len("target.layers.1."):]]).abs().max().item())
        elif k.startswith("output."):
            out[f"A.{k} ↔ B.{k}"] = float((v - sb[k]).abs().max().item())
    worst = max(out.values()) if out else float("nan")
    print(f"[INIT-CHECK] 三臂同 seed={seed} 共享子模块逐张量 max|Δ|={worst:.3e}（检查 {len(out)} 个张量）⇒ "
          f"{'PASS 逐位同初值' if worst == 0.0 else '★FAIL 共享子模块初值不同'}", flush=True)
    # 同 seed 两次构造（同臂）hash 相同、逐张量差 0；异 seed hash 不同（§9.7 要求）
    for arm in ("A", "B1", "B2"):
        torch.manual_seed(seed)
        m1 = build_arm(arm)
        h1 = state_sha256(m1.state_dict())
        torch.manual_seed(seed)
        m2 = build_arm(arm)
        h2 = state_sha256(m2.state_dict())
        dmax = max((float((a - b).abs().max().item()) for a, b in
                    zip(m1.state_dict().values(), m2.state_dict().values())), default=0.0)
        torch.manual_seed(seed ^ 0x5EED)
        m3 = build_arm(arm)
        h3 = state_sha256(m3.state_dict())
        print(f"[INIT-CHECK] arm={arm} 同 seed 两次构造 sha={h1[:16]}/{h2[:16]} ⇒ "
              f"{'相同' if h1 == h2 else '★不同'} max|Δ|={dmax:.1e} | 异 seed sha={h3[:16]} ⇒ "
              f"{'不同（阴性对照通过）' if h3 != h1 else '★相同（seed 没生效）'}", flush=True)
    return out


def param_table() -> int:
    print(f"[PARAMS] 参数对齐表（同 seed=1234 构造；实现 = 把 A 的 2 层目标层切成 B 的选择卡/重整卡）", flush=True)
    rows = {}
    for arm in ("A", "B1", "B2"):
        torch.manual_seed(1234)
        m = build_arm(arm)
        rows[arm] = param_groups_named(m)
    keys = ["input", "target2", "select", "realign", "output", "q", "合计"]
    print("[PARAMS] 子模块 | " + " | ".join(f"{a}" for a in ("A", "B1", "B2")), flush=True)
    for k in keys:
        vals = [rows[a].get(k) for a in ("A", "B1", "B2")]
        if all(v is None for v in vals):
            continue
        print("[PARAMS] " + k + " | " + " | ".join("-" if v is None else f"{v}" for v in vals), flush=True)
    base = rows["A"]["合计"]
    for arm in ("B1", "B2"):
        d = rows[arm]["合计"] - base
        print(f"[PARAMS] {arm} vs A：可训参数 Δ={d:+d}（{100.0 * d / base:+.4f}%）| "
              f"步数 三臂同为 {STEPS}×batch{BATCH}={STEPS * BATCH} 样本", flush=True)
    shared_init_check(1234)
    return 0


def _hits_path(arm: str, seed: int, smoke: bool = False) -> str:
    return os.path.join(LOG_DIR, f"75_hits_{arm}_s{seed}{'_smoke' if smoke else ''}.json")


def report_pairs() -> int:
    seeds = (1234, 5678)
    data = {}
    for arm in ("A", "B1", "B2"):
        for s in seeds:
            p = _hits_path(arm, s)
            if os.path.exists(p):
                with open(p, encoding="utf-8") as f:
                    data[(arm, s)] = json.load(f)
    print(f"[PAIR] 读到 hits 文件：{sorted(f'{a}_s{s}' for (a, s) in data)}", flush=True)
    summary = {}
    for s in seeds:
        if ("A", s) not in data:
            print(f"[PAIR] seed={s}：缺 A 臂 hits，跳过", flush=True)
            continue
        ha = data[("A", s)]["hits"]
        ema = data[("A", s)]["em"]
        for arm in ("B1", "B2"):
            if (arm, s) not in data:
                print(f"[PAIR] seed={s}：缺 {arm} 臂 hits —— 如实报「未跑」", flush=True)
                continue
            hx = data[(arm, s)]["hits"]
            emx = data[(arm, s)]["em"]
            st = paired_line(f"seed={s}", hx, ha, arm, "A")
            print(f"[PAIR] seed={s} {arm} vs A 绝对口径：EM={emx:.2f}% vs {ema:.2f}%", flush=True)
            if arm == "B2":
                summary.setdefault(s, {})["B2"] = (st["delta"], st["se"], st["p"], emx, ema)
            else:
                summary.setdefault(s, {})["B1"] = (st["delta"], st["se"], st["p"], emx, ema)
    print("[CRIT] R3-b（主，L3_PLAN §11.3/§11.4）：B2 的 EM > A 且**配对** Δ >= 2SE_Δ（SE_Δ = std(d)/sqrt(n)，"
          "不是 sqrt(SE_X²+SE_Y²)），2 seed 同号", flush=True)
    signs = []
    for s in seeds:
        v = summary.get(s, {}).get("B2")
        if v is None:
            print(f"[CRIT] R3-b seed={s}：B2/A 读数不齐 ⇒ 未判", flush=True)
            continue
        delta, se, p, emx, ema = v
        ok = (delta > 0) and (delta >= 2 * se)
        signs.append(ok)
        print(f"[CRIT] R3-b seed={s}：Δ={delta:+.2f}pp 2SE_Δ={2 * se:.2f}pp McNemar p={p:.3g} "
              f"EM(B2)={emx:.2f}% EM(A)={ema:.2f}% ⇒ {'PASS' if ok else 'FAIL'}", flush=True)
    two_same = len(signs) == 2 and all(signs)
    print(f"[CRIT] R3-b「2 seed 同号」：{'PASS（两个 seed 都 Δ>=2SE_Δ 且同号）' if two_same else 'FAIL/未判'} "
          f"⇒ R3-b 总判：{'PASS' if two_same else 'FAIL（按 L3 口径：本读数下 B2 未显著优于 A）'}", flush=True)
    b1 = [summary.get(s, {}).get("B1") for s in seeds]
    b1_ok = [v is not None and v[0] > 0 and v[0] >= 2 * v[1] for v in b1]
    print(f"[ATTR] 归因规则：B1 vs A 显著={[bool(x) for x in b1_ok]}；B2 vs A 显著={[bool(x) for x in signs]}", flush=True)
    if two_same and not any(b1_ok):
        print("[ATTR] 判决：B2>A 且 B1≈A ⇒ 增益来自**可判中间目标**（选择卡的监督信号），不是「分卡」。", flush=True)
    elif any(b1_ok):
        print("[ATTR] 判决：B1>A 已出现 ⇒ 与项目②「卡边界零增量」冲突；必须先排实现差异"
              "（同参数/同步数/同数据/同初值），**不得**直接宣称「分卡有增益」。", flush=True)
    else:
        print("[ATTR] 判决：B2≈A ⇒ 如实写「按 L3 口径无增益」（有价值的负面结果）。", flush=True)
    return 0


# ---------------- 主流程 ----------------
device = "cpu"


def main(argv=None):
    global device
    ap = argparse.ArgumentParser(description="S75 L3-r3b 选择卡三臂（A/B1/B2）+ L4 反事实")
    ap.add_argument("--arm", choices=("A", "B1", "B2"), default="A")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--k", type=int, choices=(4, 8), default=4)
    ap.add_argument("--smoke", action="store_true", help="50 步、只评前 60 条（探路，不作判据）")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--cf", action="store_true", help="训练后对同一 ckpt 做 R3-c 反事实评测")
    ap.add_argument("--pair-hits", default=None, help="同 seed 另一臂的 75_hits_*.json（出配对原始行）")
    ap.add_argument("--load", default=None, help="载入 ckpt（跳过训练；配合 --cf 另记反事实日志）")
    ap.add_argument("--param-table", action="store_true", help="只打三臂参数对齐表（CPU）")
    ap.add_argument("--report-pairs", action="store_true", help="只读 hits 文件出配对判词（CPU）")
    a = ap.parse_args(argv)

    load_tok()
    if a.param_table:
        return param_table()
    if a.report_pairs:
        return report_pairs()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
    if device != "cuda":
        print("GPU 起不来，立即停（不用 CPU 硬跑）。", flush=True)
        print("HSA_OVERRIDE_GFX_VERSION =", os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)
        sys.exit(3)

    prefix = PREFIX_BY_K[a.k]
    steps = SMOKE_STEPS if a.smoke else a.steps
    smoke_note = "（--smoke 探路，不作判据）" if a.smoke else ""
    train = load_jsonl(os.path.join(LOG_DIR, f"{prefix}_dialog_train.jsonl"))
    test = load_jsonl(os.path.join(LOG_DIR, f"{prefix}_dialog_test.jsonl"))
    for r in train:
        prep_record(r)
    for r in test:
        prep_record(r)
    ev = test[:SMOKE_EVAL] if a.smoke else test
    keys = sorted(train[0].keys())
    sha_tr = d74.d72.file_sha256(os.path.join(LOG_DIR, f"{prefix}_dialog_train.jsonl"))
    sha_te = d74.d72.file_sha256(os.path.join(LOG_DIR, f"{prefix}_dialog_test.jsonl"))
    fp_tr = d74.d72.data_fp(train)
    fp_te = d74.d72.data_fp(test)
    print(f"[DATA] k={a.k} prefix={prefix} arm={a.arm} seed={a.seed} train={len(train)} test={len(test)} "
          f"评测 n={len(ev)} | 候选数={train[0]['n_cands']} | ctx tok 最大="
          f"{max(len(r['p']) for r in train + test)} p+t 最大="
          f"{max(len(r['p']) + len(r['t']) for r in train + test)} 超 MAXLEN="
          f"{sum(1 for r in train + test if len(r['p']) + len(r['t']) > MAXLEN)} | 候选末位 token "
          f"前缀稳定性违例={POS_BAD['n']} | 字段数={len(keys)}", flush=True)
    q = FP_QUOTED.get(prefix)
    print(f"[DATA] jsonl sha256 前 8 位 train={sha_tr[:8]} test={sha_te[:8]} | data_fp train={fp_tr} test={fp_te}"
          + (f" | §11.6 引文 train={q[0]} test={q[1]} ⇒ "
             f"{'表面一致' if (fp_tr == q[0] and fp_te == q[1]) else '★与引文不同（如实报，指纹口径见 §11.6）'}"
             f"（仅供参考性对齐，不作判据）" if q else ""), flush=True)

    # ---- 地板复算（§11.5/§11.6：机会级与捷径级分开） ----
    rows, ms_name, mx, ref_em, _rows_train, chance, scan = d74.compute_floors(
        train, ev, f"{prefix}-k{a.k}-{a.arm}{'-smoke' if a.smoke else ''}")
    mc_name = max(chance, key=chance.get)
    mc = chance[mc_name]
    se_mc, thr = em_with_thr(mc, len(ev))
    print(f"[FLOOR] 复算 机会级 max_chance={mc:.2f}%（来自 {mc_name}；§11.6 引文 {MAX_CHANCE_QUOTED[a.k]:.2f}%）"
          f" 1SE={se_mc:.3f}pp 2SE={2 * se_mc:.2f}pp ⇒ 门槛(机会级)=max_chance+2SE={thr:.2f}% | "
          f"捷径级 max_naive={mx:.2f}%（来自 {ms_name}；§11.6 引文 {MAX_NAIVE_QUOTED:.2f}%）"
          f" | 参考上限(完美选择+DET)={ref_em:.2f}%"
          f"{' | ★smoke 子集口径，不作判据' if a.smoke else ''}", flush=True)

    # ---- [INIT-CHECK]（§9.7：必须先用同 seed 构造两次核验，再构造真模型） ----
    print("[INIT-CHECK] §9.7 init 缺陷修复：manual_seed 先于构造", flush=True)
    shared_init_check(a.seed)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    model = build_arm(a.arm).to(device)
    named = param_groups_named(model)
    n_par = named["合计"]
    print(f"[PARAMS] arm={a.arm} 可训参数={n_par} | 分项=" +
          " ".join(f"{k}={v}" for k, v in named.items() if k != "合计") +
          f" | 步数={steps}×batch{BATCH}={steps * BATCH} 样本 lr={LR} SEL_LAMBDA={SEL_LAMBDA}", flush=True)

    torch.cuda.reset_peak_memory_stats()
    if a.load:
        model.load_state_dict(torch.load(a.load, map_location=device, weights_only=False))
        gate, wall = None, 0.0
        print(f"[TRAIN] --load={a.load} ⇒ 跳过训练", flush=True)
    else:
        gate, wall = train_run(model, train, a.seed, steps)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    ck = os.path.join(LOG_DIR, f"75_ckpt_{a.arm}_k{a.k}_s{a.seed}{'_smoke' if a.smoke else ''}.pt")
    torch.save(model.state_dict(), ck)

    # ---- 主评测 ----
    t_eval = time.time()
    ce = eval_ce(model, ev)
    cm, cse = mean_se(ce)
    txt = greedy_gen(model, ev)
    hits = [em_strict(r["target"], txt[i]) for i, r in enumerate(ev)]
    em = 100.0 * sum(hits) / len(hits)
    se, thr_run = em_with_thr(em, len(hits))
    sel_hits = eval_sel(model, ev)
    sel = None if sel_hits is None else 100.0 * sum(sel_hits) / len(sel_hits)
    eval_sec = time.time() - t_eval

    print(f"[MAIN] arm={a.arm} k={a.k} seed={a.seed} steps={steps} batch={BATCH} lr={LR} | "
          f"模型={a.arm} d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN} params={n_par} | "
          f"配对CE={cm:.4f}±{cse:.4f} (nats/token) | 评估墙钟={eval_sec:.0f}s", flush=True)
    print(f"[EM] {'smoke子集' if a.smoke else '全量'} n={len(hits)} 严格EM={em:.2f}% ±SE={se:.2f}pp | "
          f"主口径=strip 后逐字相等（照 L2 stages/67:268 em_strict）| 机会级门槛={thr:.2f}%"
          f"(=max_chance {mc:.2f}%+2SE {2 * se_mc:.2f}pp) | 捷径级地板={mx:.2f}%", flush=True)
    print(f"[SEL] {'选择准确率=' + format(sel, '.2f') + '%（argmax 命中的候选对 == 历史 (ent,cond)，不看句面）' if sel is not None else '不适用：A 臂没有选择卡'}"
          f" | 参考:完美选择+DET 句面 EM={ref_em:.2f}% | 位置扫描 min={min(scan):.2f}% max={max(scan):.2f}%", flush=True)
    for i in range(min(3, len(ev))):
        r = ev[i]
        print(f"[GEN] [{i}] gold_idx={r['gold_idx']} gold={r['target']!r} | pred={txt[i]!r} | "
              f"EM{'对' if hits[i] else '错'}"
              + (f" | 选中对{'对' if sel_hits[i] else '错'}" if sel_hits is not None else ""), flush=True)

    # ---- R3-d：选择准确率 + 参数量/步数/显存 ----
    print(f"[CRIT] R3-d（必报）L3_PLAN §11.3「三臂都报参数量、总步数、选择准确率（B1/B2 专属）」| "
          f"arm={a.arm} 可训参数={n_par}（分项 " +
          " ".join(f"{k}={v}" for k, v in named.items() if k != "合计") +
          f"）总步数={steps}×batch{BATCH}={steps * BATCH} 样本 峰值显存={peak:.0f}MiB | "
          f"选择准确率=" + (f"{sel:.2f}%（n={len(sel_hits)}）" if sel is not None else "不适用(A 无选择卡)")
          + f"{smoke_note}", flush=True)

    # ---- R3-c 反事实 ----
    cf = {}
    if a.cf:
        for mode, tag in (("shuffle", "①打乱候选顺序"), ("replace", "②-a 替换正确候选"),
                          ("delete", "②-b 删除正确候选")):
            chits, ctxt, csel, moved, cem, cchanged = run_cf(model, ev, hits, txt, mode, a.seed, tag, a.arm)
            cf[mode] = {"hits": chits, "txt": ctxt, "sel": csel, "em": cem}
            st = paired_stats(chits, hits)
            print(f"[PAIR] R3-c({mode}) 配对：Δ(反事实 − 原始)={st['delta']:+.2f}pp SE_Δ(配对)={st['se']:.2f}pp "
                  f"2SE_Δ={2 * st['se']:.2f}pp | 不一致对 b(反事实错原对)={st['n01']} "
                  f"c(反事实对原错)={st['n10']} | McNemar 双侧精确 p={st['p']:.3g}", flush=True)
            if mode == "shuffle":
                agree = 100.0 * sum(1 for x, y in zip(chits, hits) if x == y) / len(hits)
                ok = (st["delta"] >= -2 * st["se"]) and (st["p"] >= 0.05 or st["n01"] == st["n10"] == 0)
                print(f"[CRIT] R3-c① L3_PLAN §11.2「打乱候选顺序 ⇒ 回复应当不变（位置不变性）」| "
                      f"一致率={agree:.2f}%（逐条 EM 命中一致）| EM 原始={em:.2f}% → 打乱={cem:.2f}% "
                      f"Δ={st['delta']:+.2f}pp 2SE_Δ={2 * st['se']:.2f}pp McNemar p={st['p']:.3g} | "
                      f"正确项位置移动={moved}/{len(ev)} ⇒ {'PASS 无位置泄漏' if ok else '★FAIL 位置泄漏 ⇒ 结论作废'}"
                      f"{smoke_note}", flush=True)
            else:
                se_cf = (cem / 100.0 * (1 - cem / 100.0) / len(ev)) ** 0.5 * 100.0
                near_floor = cem <= max(mx, thr) + 2.0 * se_cf
                change_rate = 100.0 * cchanged / len(ev)
                ok = (change_rate >= 99.0) and near_floor
                sub = "②-a" if mode == "replace" else "②-b"
                print(f"[CRIT] R3-c({sub}) L3_PLAN §11.2「{tag} ⇒ 回复必变、且 EM 落到地板附近」| "
                      f"回复变化率={change_rate:.2f}%（{cchanged}/{len(ev)}）| "
                      f"EM={cem:.2f}% vs 机会级门槛={thr:.2f}% / 捷径级地板={mx:.2f}% ⇒ "
                      f"{'落到地板附近' if near_floor else '★未落到地板附近（如实报）'} | "
                      f"选择准确率=" + ("不适用(A 无选择卡)" if csel is None
                                        else f"{100.0 * sum(csel) / len(csel):.2f}%")
                      + f" ⇒ {'PASS' if ok else '★CHECK（变化率或落点不达标，如实报）'}{smoke_note}", flush=True)

    # ---- R3-b：本 run 的配对原始行（给了 --pair-hits） ----
    if a.pair_hits and os.path.exists(a.pair_hits):
        with open(a.pair_hits, encoding="utf-8") as f:
            other = json.load(f)
        hx, hy = (hits, other["hits"]) if a.arm != "A" else (other["hits"], hits)
        nx, ny = (a.arm, other["arm"]) if a.arm != "A" else (other["arm"], "A")
        st = paired_line(f"seed={a.seed}", hx, hy, nx, ny)
        print(f"[CRIT] R3-b（主）本 run 配对：{nx} vs {ny} Δ={st['delta']:+.2f}pp SE_Δ(配对)={st['se']:.2f}pp "
              f"2SE_Δ={2 * st['se']:.2f}pp 不一致对 b={st['n01']} c={st['n10']} McNemar p={st['p']:.3g} ⇒ "
              f"{'PASS（Δ>0 且 Δ>=2SE_Δ）' if (st['delta'] > 0 and st['delta'] >= 2 * st['se']) else 'FAIL'}"
              f"（「2 seed 同号」要另一 seed 的 hits 文件，跑完六 run 用 --report-pairs 出总判）{smoke_note}",
              flush=True)
    else:
        print(f"[CRIT] R3-b（主）本 run 未给 --pair-hits ⇒ 只出单臂读数；配对判定（Δ / SE_Δ(配对) / b,c / "
              f"McNemar p）在 --report-pairs 里出{smoke_note}", flush=True)

    print(f"[CRIT] R3-b(绝对) 本 run EM={em:.2f}% vs 机会级门槛= max_chance({mc:.2f}%)+2SE({2 * se_mc:.2f}pp)"
          f"={thr:.2f}% ⇒ {'PASS 击败机会水平' if em > thr else '★FAIL（未击败机会水平，如实报）'}"
          f" | 捷径级地板=max_naive {mx:.2f}%（EM 远高于它）{smoke_note}", flush=True)

    hp = _hits_path(a.arm, a.seed, a.smoke)
    with open(hp, "w", encoding="utf-8") as f:
        json.dump({"arm": a.arm, "seed": a.seed, "k": a.k, "smoke": a.smoke, "n": len(hits),
                   "em": em, "se": se, "sel": sel, "params": n_par, "steps": steps,
                   "peak_mib": peak, "wall_s": wall, "thr": thr, "max_chance": mc, "max_naive": mx,
                   "hits": hits, "sel_hits": sel_hits,
                   "cf": {m: {"em": v["em"], "hits": v["hits"], "sel_hits": v["sel"]}
                          for m, v in cf.items()}}, f)
    print(f"[META] arm={a.arm} k={a.k} seed={a.seed} steps={steps} batch={BATCH} lr={LR} d={D} ff={FF} "
          f"nhead={NHEAD} maxlen={MAXLEN} n_params={n_par} 可训={sum(p.numel() for p in model.parameters() if p.requires_grad)} "
          f"peak_mib={peak:.0f} EM={em:.2f}% SE={se:.2f}pp sel={sel if sel is None else round(sel, 2)} "
          f"配对CE={cm:.4f}±{cse:.4f} 机会级门槛={thr:.3f}% max_naive={mx:.3f}% smoke={a.smoke} "
          f"cf={a.cf} pair_hits={a.pair_hits} ckpt={ck} hits={hp} 训练墙钟={wall:.0f}s "
          f"评估墙钟={eval_sec:.0f}s 总墙钟={wall + eval_sec:.0f}s", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
