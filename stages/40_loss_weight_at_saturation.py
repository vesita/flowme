#!/usr/bin/env python3
"""S39/S40 · ★在【真饱和点 15000 步】重做 S37 的 loss 加权判决（只训 w3/w10）。

为什么必须做：
  S38（logs/s38_sat.log）：真饱和点 ≈9750–12000 步；w1 上限 = 91.39%(1234)/86.80%(5678)，
    但末段 14250→15000 seed5678 仍 +2.25pp ⇒ S38 自判「仍未饱和」；@6000→@15000 仍涨
    +5.63/+12.87pp ⇒ S37 的 @6000 判决确实欠训。
  S37（logs/s37_loss.log）：在 @6000 上判 w3/w10「未过判据且方向随 seed 翻转」，并自判
    「未饱和 ⇒ 结论不可下（R35）」。
  ⇒ 本单元唯一目的：在 15000 步重做同一判决。

设计（★只训 w3 与 w10；w1@15000 已有 = logs/38_ckpt_w1_s15000_seed*.pt，复用）：
  w1  = 全部 token 权重 1（复用 S38 的 @9000/@12000/@15000 ckpt，逐位核 sha16）；
  w3  = 数字/算子 token（classify()==0）权重 3，其余 1；
  w10 = 同上权重 10。
  加权口径 = sum(w·CE)/sum(w)（= torch CrossEntropyLoss(weight, reduction='mean')），
  与 S37 weighted_ce 逐字相同；其余（d=128 ff=512 NHEAD=4 lr=1e-3 batch=32 MAXLEN=512、
  思维卡带残差、add_3d train4000/test800、不切 dev）与 S38 逐字相同。
  4 run（w3/w10 × seed 1234/5678）× 15000 步，★每 750 步取快照 ⇒ 曲线 + 收敛速度对比。
  ckpt（S38 已证「快照 == 独立跑 N 步」等价）→ logs/40_ckpt_w<w>_s<steps>_seed<s>.pt。

★ 判据（写死，分开报；Δ 逐样本配对 ±SE）：
  Δ = 该臂 − w1@同一步（@9000 / @12000 / @15000 三点都报）：
    Δ ≥ 2SE 且 2 seed 同号 ⇒ 加权有效（报幅度）；|Δ| ≤ 1SE ⇒ 无效；
    Δ ≤ −2SE 且 2 seed 同号 ⇒ 有害；其余 ⇒ 未过写死判据，如实报。
  ★数字每步正确率同判据（主指标）。
  ★饱和检查（R35）：报 w3/w10 在 12000→15000 的增量 ⇒ 若 ≥1pp ⇒ 「未饱和、结论不可下」。
  ★额外：加权是否只是改变收敛速度？报 w3/w10 与 w1 的曲线形状（@3000 是否落后、
    @15000 是否追平）⇒ 追平即「收敛速度效应」。

口径（缺一即无效）：
  1 EM 一律 batch=1（R28）+ batch=1 vs 16 逐字一致性自检；
  2 ★锚点复核：复用 w1@15000 ckpt 必须复现 S38 的 91.25%(1234)/87.25%(5678)，Δ>1pp 点名；
  3 SEEN/UNSEEN 分层（train 对 test 的 (a,b) 泄漏 16/800）；
  4 地板用本桶：单答案 0.12% / top5 0.25%（不用 1.89%）；
  5 R29：干预 Δ 非零比例；TEST sha256[:16] 必须 == b34e7ea515203227；
  6 数字每步正确率 = 样本级 exp(−数字/算子 token 平均 CE)，@9000/@12000/@15000。

复用：exec stages/19_residual_scan.py 到 "# ③ 主循环" 之前（同 tokenizer / 同 add_3d
train4000+test800 / 同 greedy_gen(batch=1) / classify / parse_ans / r28_selfcheck / mean_se）。
只允许写：本文件、logs/s40_*.log、logs/40_*、/tmp；既有 stages/*.py 只读。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

ROOT = "/home/vesita/coding/my/flowme"
SRC19 = f"{ROOT}/stages/19_residual_scan.py"

SMOKE40 = os.environ.get("S40_SMOKE") == "1"
if SMOKE40:
    os.environ["S19_SMOKE"] = "1"          # 让 S19 头只造小数据（冒烟）

# ============================================================================
# ① 逐字复用 S19 前半段（⇒ 同 tokenizer、同 add_3d train4000/test800、同评估口径）
# ============================================================================
_src = open(SRC19, encoding="utf-8").read()
_cut = _src.index("# ③ 主循环")
NS = {"__name__": "s19_head", "__file__": SRC19}
exec(compile(_src[:_cut], SRC19, "exec"), NS)          # noqa: S102
for _k, _v in NS.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
del NS

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402

T0 = time.time()
BUCKET = "add_3d"
SEEDS40 = (1234, 5678)
ARMS40 = ("w3", "w10")                                   # ★只训这两臂；w1 复用 S38 ckpt
W40 = {"w1": 1.0, "w3": 3.0, "w10": 10.0}
STEPS40 = 60 if SMOKE40 else 15000
GRID = (30, 60) if SMOKE40 else tuple(range(750, 15001, 750))
SAVE_ATS = (60,) if SMOKE40 else (9000, 12000, 15000)
BASE_STEPS = (30, 60) if SMOKE40 else (9000, 12000, 15000)  # 配对基准的三个时间点
N_EVAL = 60 if SMOKE40 else N_TEST                       # 800
CKPT_TMPL = f"{ROOT}/logs/40_ckpt_{{arm}}_s{{steps}}_seed{{seed}}.pt"
BASE_TMPL = (f"/tmp/40_smoke_w1_s{{steps}}_seed{{seed}}.pt" if SMOKE40
             else f"{ROOT}/logs/38_ckpt_w1_s{{steps}}_seed{{seed}}.pt")
JSONL = "/tmp/40_results_smoke.jsonl" if SMOKE40 else f"{ROOT}/logs/40_results.jsonl"
S38_LOG = f"{ROOT}/logs/s38_sat.log"
TEST_SHA_EXPECT = "b34e7ea515203227"
LEAK_EXPECT = 16
# S38 入库值：w1@15000 = 锚点（必须复现）；@9000/@12000 用于交叉核对曲线
ANCHOR_W1_S38 = {1234: {9000: 0.8800, 12000: 0.9188, 15000: 0.9125},
                 5678: {9000: 0.8362, 12000: 0.8712, 15000: 0.8725}}
BASE_SHA_S38 = {(9000, 1234): "b535b30b9981b805", (9000, 5678): "317eec5686dffc1e",
                (12000, 1234): "639360d078db2493", (12000, 5678): "ee99239df44334e6",
                (15000, 1234): "a855df724c8f1004", (15000, 5678): "539adcd14c846fa8"}

print(f"\n[S40] ★在真饱和点重做 loss 加权判决：桶={BUCKET} 步数={STEPS40} 曲线格="
      f"{GRID[0]}..{GRID[-1]}/{GRID[1]-GRID[0]}步 seeds={SEEDS40} 新训臂={list(ARMS40)} "
      f"(w1 复用 S38 ckpt @{BASE_STEPS}) | ★不切 dev、固定末点 | d={D} ff={FF} NHEAD={NHEAD} "
      f"lr={LR} batch={BATCH} MAXLEN={MAXLEN} 思维卡残差 | EM主口径=batch{EM_BATCH} "
      f"自检=batch{CHK_BATCH} | smoke={SMOKE40}", flush=True)

# ============================================================================
# ② 数据核对（口径 3/4/5 + TEST 指纹 + 加权作用面）
# ============================================================================
TRAIN = DATA[BUCKET]["train"]
TEST = DATA[BUCKET]["test"]
TEST_EVAL = TEST[:N_EVAL]

_h = hashlib.sha256()
for r in TEST:
    _h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
TEST_SHA = _h.hexdigest()[:16]
_tp_tr = {(r["a"], r["b"]) for r in TRAIN}
for r in TEST:
    r["seen"] = (r["a"], r["b"]) in _tp_tr
LEAK = sum(1 for r in TEST if r["seen"])
N_SEEN, N_UNSEEN = LEAK, len(TEST) - LEAK

_c_tr = Counter(r["gold"] for r in TRAIN)
_top1, _top1n = _c_tr.most_common(1)[0]
_top5 = [v for v, _ in _c_tr.most_common(5)]
FLOOR_1 = sum(1 for r in TEST if r["gold"] == _top1) / len(TEST)
FLOOR_5 = sum(1 for r in TEST if r["gold"] in _top5) / len(TEST)

_n_dig = sum(1 for r in TEST for t in r["t"] if classify(t) == 0)
_n_tok = sum(len(r["t"]) for r in TEST)
W_SHARE = _n_dig / _n_tok
_n_dig_tr = sum(1 for r in TRAIN for t in r["t"] if classify(t) == 0)
_n_tok_tr = sum(len(r["t"]) for r in TRAIN)

sha_ok = TEST_SHA == TEST_SHA_EXPECT
leak_ok = LEAK == LEAK_EXPECT
print(f"[S40-DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)}(eval {len(TEST_EVAL)}) "
      f"答案域={min(answer_values(BUCKET))}..{max(answer_values(BUCKET))}"
      f"({len(answer_values(BUCKET))}个) | ★不切 dev ⇒ 训练集={len(TRAIN)}", flush=True)
print(f"[S40-CHK5] TEST 指纹 sha256[:16]={TEST_SHA} vs 入库 {TEST_SHA_EXPECT} ⇒ "
      f"{'逐位一致 ✓' if sha_ok else '★不一致'} | (a,b) 泄漏对 train={LEAK}/{len(TEST)} vs "
      f"入库 {LEAK_EXPECT} ⇒ {'一致 ✓' if leak_ok else '★不一致'}", flush=True)
print(f"[S40-CHK3] SEEN={N_SEEN} / UNSEEN={N_UNSEEN}（train 对 test 的 (a,b) 泄漏 "
      f"{100.0*LEAK/len(TEST):.2f}%，三点分层报 EM）", flush=True)
print(f"[S40-CHK4] add_3d 本桶地板：单答案 '{_top1}'(train {_top1n}/4000) 覆盖 test "
      f"{FLOOR_1*100:.2f}% | top5={_top5} 覆盖 test {FLOOR_5*100:.2f}% ⇒ ★本地板="
      f"{FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}%（不借用 1.89%）", flush=True)
print(f"[S40-CHK-INT] 加权作用面：数字/算子 token 占 target token {100.0*W_SHARE:.1f}%(test) / "
      f"{100.0*_n_dig_tr/_n_tok_tr:.1f}%(train) ⇒ w1/w3/w10 非 1 权重覆盖率相同，"
      f"唯一变量是权重值 | 口径=sum(w·CE)/sum(w)", flush=True)
if not SMOKE40:
    assert sha_ok and leak_ok, "数据指纹不符 ⇒ 与 S19/S36/S37/S38 不是同一数据集，本单元无效"


# ============================================================================
# ③ 模型与加权 loss：与 S37 Cards37 / weighted_ce 逐字相同（构造顺序同 ⇒ 同 seed 同初始化）
# ============================================================================
class Cards40(nn.Module):
    """h = h + Think(h)（k=1）；训练 loss 由 weighted_ce 决定（w1/w3/w10）。"""

    def __init__(self, d, ff):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        self.thought = enc_layer(d, ff)
        self.head = nn.Linear(d, V)
        self.register_buffer("pe", sin_pe(MAXLEN, d), persistent=False)

    def logits(self, ids):
        n = ids.size(1)
        m = torch.full((n, n), float("-inf"), device=ids.device)
        m = torch.triu(m, diagonal=1)
        m = m.unsqueeze(0).expand(ids.size(0), -1, -1).clone()
        m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
        m = m.repeat_interleave(NHEAD, dim=0)
        pe = self.pe if n <= self.pe.size(0) else sin_pe(n, self.d).to(self.pe.device)
        assert n <= pe.size(0), f"PE 长度 {pe.size(0)} < 序列长 {n}"
        x = self.emb(ids) + pe[:n].unsqueeze(0)
        h = self.in_enc(x, src_mask=m)
        h = h + self.thought(h, src_mask=m)
        return self.head(h)


def n_params_40():
    mdl = Cards40(D, FF)
    n = sum(p.numel() for p in mdl.parameters())
    del mdl
    return n


def weighted_ce(model, recs, w) -> torch.Tensor:
    """★本单元唯一改动（= S37 逐字）：数字/算子 token 的 CE 乘 w，其余乘 1；分母 = 权重之和。"""
    ids, s = build_batch(recs)
    logits = model.logits(ids)
    logp = torch.log_softmax(logits, dim=-1)
    nll = torch.zeros((), device=device)
    den = 0.0
    for i, r in enumerate(recs):
        e = s[i] + len(r["t"])
        lp = logp[i, s[i] - 1: e - 1]
        tgt = ids[i, s[i]: e]
        if w == 1.0:
            nll = nll - lp.gather(1, tgt.unsqueeze(1)).sum()
            den += len(r["t"])
        else:
            wt = torch.tensor([w if classify(int(t)) == 0 else 1.0 for t in tgt],
                              device=device, dtype=lp.dtype)
            nll = nll - (wt.unsqueeze(1) * lp.gather(1, tgt.unsqueeze(1))).sum()
            den += float(wt.sum())
    return nll / den


def train40(arm, seed, train, steps, grid, save_ats):
    """★与 S38 train38 唯一差别：loss 换成 weighted_ce(w)。快照在 step 边界取 ⇒
    @N 快照 == 独立跑 N 步（S38 已实测该等价性）。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards40(D, FF).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0 = 0, 0, time.time()
    last, snaps, ckpts = 0.0, {}, {}
    gset = set(grid)
    w = W40[arm]
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = weighted_ce(model, [train[i] for i in idx], w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step in gset:
            snaps[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step in save_ats:
                ck = CKPT_TMPL.format(arm=arm, steps=step, seed=seed)
                torch.save(snaps[step], ck)
                ckpts[step] = ck
        if step % 1500 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train:{arm}:w={w:g}] step={step}/{steps} loss={loss.item():.4f} "
                  f"elapsed={el:.0f}s s/step={(el - last) / (step % 1500 or 1500):.3f}", flush=True)
            last = el
    assert set(grid) <= set(snaps), f"未取到全部曲线格点快照: 缺 {sorted(set(grid)-set(snaps))}"
    return model, time.time() - t0, snaps, ckpts


@torch.no_grad()
def em_at(model, recs):
    txt = greedy_gen(model, recs, batch=EM_BATCH)          # ★R28 主口径 batch=1
    return [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, recs)], txt


@torch.no_grad()
def per_sample_ce(model, recs):
    """逐样本 (整体CE, 数字/算子 token 的 CE)；口径 = S19 ce_categories 的 samp_all / dig_samp。
    与 S37/S38 逐字相同。"""
    model.eval()
    all_ce, dig_ce, dig_tok = [], [], []
    for i in range(0, len(recs), GEN_BATCH):
        chunk = recs[i: i + GEN_BATCH]
        ids, s = build_batch(chunk)
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        for j, r in enumerate(chunk):
            e = s[j] + len(r["t"])
            lp = logp[j, s[j] - 1: e - 1]
            tgt = ids[j, s[j]: e]
            ces = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1).tolist()
            all_ce.append(sum(ces) / len(ces))
            ds = [v for p, v in enumerate(ces) if classify(int(tgt[p])) == 0]
            dig_tok.extend(ds)
            dig_ce.append(sum(ds) / len(ds) if ds else float("nan"))
    model.train()
    return all_ce, dig_ce, dig_tok


def dig_acc(dig_ce):
    """样本级：a_i = exp(−CE_i)，返回 (acc, acc_se, CE均值, CE的SE, n)。"""
    ys = [x for x in dig_ce if x == x]
    m, se = mean_se(ys)
    acc = math.exp(-m)
    return acc, acc * se, m, se, len(ys)


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def add_jsonl(obj):
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def frac(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def paired_stats(xa, xb):
    d = [a - b for a, b in zip(xa, xb)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return dict(n=n, d=m, se=se, nz=sum(1 for x in d if x != 0) / n,
                ratio=(m / se if se > 0 else float("inf")))


def paired_stats_skipnan(xa, xb):
    pairs = [(a, b) for a, b in zip(xa, xb) if a == a and b == b]
    if not pairs:
        return dict(n=0, d=float("nan"), se=float("nan"), nz=float("nan"), ratio=float("nan"))
    d = [a - b for a, b in pairs]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return dict(n=n, d=m, se=se, nz=sum(1 for x in d if x != 0) / n,
                ratio=(m / se if se > 0 else float("inf")))


def is_catch(d):
    return (d["se"] == 0 and d["d"] == 0) or (d["se"] > 0 and abs(d["d"]) <= d["se"])


def is_neg2(d):
    return d["se"] > 0 and d["d"] < 0 and d["d"] <= -2 * d["se"]


def is_pos2(d):
    return d["se"] > 0 and d["d"] > 0 and d["d"] >= 2 * d["se"]


def delta_tag(d):
    if d["n"] == 0:
        return "无有效样本"
    if d["se"] == 0:
        return "Δ≡0（逐样本完全相同，SE=0）" if d["d"] == 0 else \
               f"SE=0 且 Δ={d['d']*100:+.2f}pp（全样本同向）"
    if is_neg2(d):
        return "★≤−2SE（显著更低）"
    if is_pos2(d):
        return "★≥+2SE（显著更高）"
    if is_catch(d):
        return "|Δ|≤1SE（追平）"
    return "1SE<|Δ|<2SE（未过门槛）"


def verdict(per_seed: dict) -> str:
    if len(per_seed) < 2:
        return "★seed 不全 ⇒ 不可判"
    ds = list(per_seed.values())
    if all(is_pos2(d) for d in ds) and all(d["d"] > 0 for d in ds):
        return "★加权有效（≥+2SE 且 2 seed 同号）"
    if all(is_neg2(d) for d in ds) and all(d["d"] < 0 for d in ds):
        return "★加权有害（≤−2SE 且 2 seed 同号）"
    if all(is_catch(d) for d in ds):
        return "★无效（|Δ|≤1SE，如实报）"
    return "★未过写死判据（既非 2SE 显著，也非 |Δ|≤1SE）⇒ 如实报"


def eval_ckpt(path, tag):
    """加载一份 state_dict 并按主口径评估一点：EM(batch1) + 数字每步 + SEEN/UNSEEN。"""
    sd = torch.load(path, map_location="cpu")
    ev = Cards40(D, FF).to(device)
    ev.load_state_dict({k: v.to(device) for k, v in sd.items()})
    hits, _ = em_at(ev, TEST_EVAL)
    all_ce, dig_ce, _dt = per_sample_ce(ev, TEST_EVAL)
    acc, acc_se, dce, _s, _n = dig_acc(dig_ce)
    out = dict(em=frac(hits), hits=hits, acc=acc, acc_se=acc_se, dig_ce=dce,
               ce=frac(all_ce),
               acc_samp=[math.exp(-x) if x == x else float("nan") for x in dig_ce],
               em_seen=frac([hits[i] for i, r in enumerate(TEST_EVAL) if r["seen"]]),
               em_unseen=frac([hits[i] for i, r in enumerate(TEST_EVAL) if not r["seen"]]))
    print(f"  [S40-BASE] {tag} EM={out['em']*100:.2f}% | 数字每步={acc*100:.2f}% | "
          f"整体CE={out['ce']:.4f}", flush=True)
    del ev, sd
    torch.cuda.empty_cache()
    return out


def parse_s38_curve(path):
    """从 S38 日志读 w1 的每 750 步 EM 曲线（同一个评估管线、同一 ckpt 网格）。"""
    cur = {1234: {}, 5678: {}}
    if not os.path.exists(path):
        return cur
    for line in open(path, encoding="utf-8"):
        if not line.startswith("[S38-CURVE] "):
            continue
        parts = [p.strip() for p in line.split("]", 1)[1].split("|")]
        if parts[0] == "step":
            continue
        try:
            st = int(parts[0])
            cur[1234][st] = float(parts[1].rstrip("%")) / 100.0
            cur[5678][st] = float(parts[2].rstrip("%")) / 100.0
        except (ValueError, IndexError):
            continue
    return cur


# ============================================================================
# ④ 基准 w1：复用 S38 ckpt（先核 sha16），三点逐点评估 ⇒ 与 w3/w10 同管线配对
# ============================================================================
print(f"\n[S40-PARAMS] Cards40(res)={n_params_40()/1e6:.3f}M（w1/w3/w10 参数量完全相同，"
      f"唯一变量是 loss 权重）", flush=True)
RES: dict = {}
BASE: dict = {}
open(JSONL, "w").close()

if SMOKE40:
    for seed in SEEDS40:
        m1, _wt, snaps1, ck1 = train40("w1", seed, TRAIN, STEPS40, GRID, BASE_STEPS)
        for st in BASE_STEPS:
            torch.save(snaps1[st], BASE_TMPL.format(steps=st, seed=seed))
        del m1, snaps1
        torch.cuda.empty_cache()

print(f"\n[S40] ===== ★基准 w1（复用 S38 的 @{BASE_STEPS} ckpt，逐位核 sha16 后按同一 "
      f"batch={EM_BATCH} 管线重评）=====", flush=True)
for seed in SEEDS40:
    for st in BASE_STEPS:
        p = BASE_TMPL.format(steps=st, seed=seed)
        if not os.path.exists(p):
            print(f"[S40-BASE-MISS] seed={seed} @{st}: ★缺 {p} ⇒ 该点无法配对", flush=True)
            continue
        got = sha16(p)
        exp = BASE_SHA_S38.get((st, seed))
        flag = "✓" if exp is None or got == exp else f"★≠S38 入库({exp})"
        print(f"[S40-BASE-SHA] seed={seed} @{st}: {os.path.basename(p)} sha256[:16]={got} {flag}",
              flush=True)
        BASE[(seed, st)] = eval_ckpt(p, f"w1 seed={seed} @{st}")

# 锚点复核（口径 2）：w1@15000 必须复现 S38 的 91.25/87.25
print("\n[S40] ===== ★口径自检 2：锚点复核（w1@15000 必须复现 S38 入库值，Δ>1pp 点名）=====",
      flush=True)
ANCHOR_OK = True
for seed in SEEDS40:
    b = BASE.get((seed, 15000))
    if b is None:
        ANCHOR_OK = False
        print(f"[S40-ANCHOR] seed={seed}: 基准缺失 ⇒ 无法复核", flush=True)
        continue
    for st in BASE_STEPS:
        ref = ANCHOR_W1_S38[seed].get(st)
        bb = BASE.get((seed, st))
        if bb is None or ref is None:
            continue
        dd = (bb["em"] - ref) * 100
        tag = "（锚点，写死 ≤1pp）" if st == 15000 else "（交叉核对 S38 曲线）"
        ok = abs(dd) <= 1.0
        if st == 15000:
            ANCHOR_OK &= ok
        print(f"[S40-ANCHOR] w1 seed={seed} @{st}: 本次={bb['em']*100:.2f}% vs S38="
              f"{ref*100:.2f}% ⇒ Δ={dd:+.2f}pp {'✓' if ok else '★>1pp，点名'} {tag}", flush=True)
BASE_CURVE = parse_s38_curve(S38_LOG)
if BASE_CURVE[1234]:
    print(f"[S40-BASE-CURVE] 已从 {os.path.basename(S38_LOG)} 读入 w1 曲线 "
          f"{len(BASE_CURVE[1234])} 格（每 750 步，用于曲线形状对比）", flush=True)

if os.environ.get("S40_BASE_ONLY") == "1":        # 开发用：只核基准是否逐位复现 S38
    print(f"[S40-BASE-ONLY] 只做基准/锚点复核 ⇒ 退出（锚点复现="
          f"{'是' if ANCHOR_OK else '否(点名)'}）", flush=True)
    print("[DONE] exit=0", flush=True)
    sys.exit(0)

# ============================================================================
# ⑤ 主循环：w3 / w10 × 2 seed，各 15000 步，每 750 步评估
# ============================================================================
for arm in ARMS40:
    for seed in SEEDS40:
        t0 = time.time()
        print(f"\n[S40] ===== arm={arm}(w={W40[arm]:g}) seed={seed} steps={STEPS40} | "
              f"train={len(TRAIN)} test={len(TEST_EVAL)} | 数字/算子 token 权重 "
              f"{W40[arm]:g}，其余 1，口径 sum(w·CE)/sum(w) =====", flush=True)
        model, wall_train, snaps, ckpts = train40(arm, seed, TRAIN, STEPS40, GRID, SAVE_ATS)
        print(f"[S40] {arm} seed={seed} 训练结束 墙钟={wall_train/60:.2f}min | ckpt 落盘 "
              f"{sorted(ckpts)} ⇒ {[os.path.basename(v) for v in ckpts.values()]}", flush=True)

        ev = Cards40(D, FF).to(device)
        curve = {}
        for st in GRID:
            ev.load_state_dict({k: v.to(device) for k, v in snaps[st].items()})
            hits, _ = em_at(ev, TEST_EVAL)
            all_ce, dig_ce, _dt = per_sample_ce(ev, TEST_EVAL)
            acc, acc_se, dce, _s, _n = dig_acc(dig_ce)
            curve[st] = dict(em=frac(hits), hits=hits, acc=acc, acc_se=acc_se, dig_ce=dce,
                             ce=frac(all_ce),
                             acc_samp=[math.exp(-x) if x == x else float("nan")
                                       for x in dig_ce])
            if st in BASE_STEPS or st in (3000, 6000):
                print(f"  [S40-PT] {arm} seed={seed} step={st} EM={curve[st]['em']*100:.2f}% | "
                      f"数字每步={acc*100:.2f}% | 数字CE={dce:.4f} | 整体CE={curve[st]['ce']:.4f}",
                      flush=True)

        fin = GRID[-1]
        hits = curve[fin]["hits"]
        _r28 = r28_selfcheck(ev, TEST_EVAL, tag=f"[{arm} s{fin} seed{seed}] ",
                             k=min(CHK_BATCH, len(TEST_EVAL)))
        seen = [hits[i] for i, r in enumerate(TEST_EVAL) if r["seen"]]
        unseen = [hits[i] for i, r in enumerate(TEST_EVAL) if not r["seen"]]
        del snaps, model, ev
        torch.cuda.empty_cache()

        RES[(arm, seed)] = dict(
            arm=arm, w=W40[arm], seed=seed, steps=STEPS40, wall_train=wall_train,
            wall_total=time.time() - t0, n_eval=len(TEST_EVAL), curve=curve,
            em_seen=frac(seen), em_unseen=frac(unseen), n_seen=len(seen), n_unseen=len(unseen),
            r28_same=_r28["same"], r28_k=_r28["k"],
            ckpt={st: dict(path=ckpts[st], sha=sha16(ckpts[st])) for st in ckpts})
        add_jsonl(dict(arm=arm, w=W40[arm], seed=seed, steps=STEPS40, wall_train=wall_train,
                       curve={str(st): {k: v for k, v in curve[st].items()
                                        if k not in ("hits", "acc_samp")} for st in GRID},
                       em_seen=frac(seen), em_unseen=frac(unseen), n_seen=len(seen),
                       n_unseen=len(unseen), r28_same=_r28["same"], r28_k=_r28["k"],
                       ckpt={str(st): dict(path=ckpts[st], sha=sha16(ckpts[st]))
                             for st in ckpts}))
        print(f"[S40-RUN] {arm} seed={seed}: EM@{fin}={curve[fin]['em']*100:.2f}% "
              f"SEEN={frac(seen)*100:.2f}%({len(seen)}) UNSEEN={frac(unseen)*100:.2f}%"
              f"({len(unseen)}) | 数字每步@{fin}={curve[fin]['acc']*100:.2f}% | "
              f"R28={_r28['same']}/{_r28['k']} | 合计墙钟={(time.time()-t0)/60:.2f}min", flush=True)

# ============================================================================
# ⑥ 三臂 × 2 seed 表（@9000/@12000/@15000）
# ============================================================================
print("\n[S40] ===== ★三臂 × 2 seed 表（EM 主口径 batch=1；w1 复用 S38 ckpt）=====", flush=True)
print("[S40] arm seed step | EM | 数字每步 | SEEN | UNSEEN | 整体CE", flush=True)
for a in ("w1", *ARMS40):
    for s in SEEDS40:
        for st in BASE_STEPS:
            if a == "w1":
                r = BASE.get((s, st))
                if r is None:
                    continue
                print(f"[S40-T] w1 {s} {st} | {r['em']*100:.2f}% | {r['acc']*100:.2f}%"
                      f"±{r['acc_se']*100:.2f}pp | {r['em_seen']*100:.2f}%({N_SEEN}) | "
                      f"{r['em_unseen']*100:.2f}%({N_UNSEEN}) | {r['ce']:.4f}", flush=True)
            else:
                rr = RES.get((a, s))
                if rr is None:
                    continue
                c = rr["curve"][st]
                print(f"[S40-T] {a} {s} {st} | {c['em']*100:.2f}% | {c['acc']*100:.2f}%"
                      f"±{c['acc_se']*100:.2f}pp | {rr['em_seen']*100:.2f}%({N_SEEN})"
                      f"@{GRID[-1]} | {rr['em_unseen']*100:.2f}%({N_UNSEEN})@{GRID[-1]} | "
                      f"{c['ce']:.4f}", flush=True)

# ============================================================================
# ⑦ 判决：Δ = 臂 − w1@同一步，逐样本配对 ±SE（三点 × 2 seed），EM 与数字每步各判一次
# ============================================================================
for metric, key in (("EM", "hits"), ("数字每步正确率", "acc_samp")):
    print(f"\n[S40] ===== ★判决（{metric}）：Δ = {metric}(w) − {metric}(w1@同步数)，"
          f"逐样本配对 =====", flush=True)
    for a in ARMS40:
        for st in BASE_STEPS:
            per_seed = {}
            for s in SEEDS40:
                rr = RES.get((a, s))
                bb = BASE.get((s, st))
                if rr is None or bb is None or st not in rr["curve"]:
                    continue
                if key == "hits":
                    tot = len(TEST_EVAL)
                    a_hits = rr["curve"][st]["hits"]
                    b_hits = bb["hits"]
                    if len(a_hits) != len(b_hits):
                        # 不同长度时截到公共前缀（同评估集本应等长）
                        n = min(len(a_hits), len(b_hits))
                        a_hits, b_hits = a_hits[:n], b_hits[:n]
                        tot = n
                    d = paired_stats(a_hits, b_hits)
                else:
                    d = paired_stats_skipnan(rr["curve"][st]["acc_samp"], bb["acc_samp"])
                    tot = d["n"]
                per_seed[s] = d
                unit = "pp" if key == "hits" else "pp(acc)"
                print(f"[S40-D{key.upper()[:3]}] {a}−w1 @{st} seed={s}: "
                      f"Δ={d['d']*100:+.2f}±{d['se']*100:.2f}{unit} Δ/SE={d['ratio']:.2f} | "
                      f"R29 Δ非零比={d['nz']*100:.1f}%(n={tot}) | "
                      f"w1={((frac(BASE[(s,st)]['hits']) if key=='hits' else BASE[(s,st)]['acc'])*100):.2f}%"
                      f" → {a}="
                      f"{((frac(RES[(a,s)]['curve'][st]['hits']) if key=='hits' else RES[(a,s)]['curve'][st]['acc'])*100):.2f}%"
                      f" | {delta_tag(d)}", flush=True)
            if per_seed:
                print(f"[S40-VERDICT-{key.upper()[:3]}] {a} @{st}: {verdict(per_seed)}", flush=True)

# ============================================================================
# ⑧ 饱和检查（R35）：w3/w10 的 12000→15000 增量
# ============================================================================
print(f"\n[S40] ===== ★R35 饱和检查：w3/w10 末段（12000→15000）EM 增量，≥1pp ⇒ 未饱和、"
      f"结论不可下 =====", flush=True)
UNSAT = []
for a in ARMS40:
    for s in SEEDS40:
        rr = RES.get((a, s))
        if rr is None:
            continue
        c = rr["curve"]
        if not all(st in c for st in (9000, 12000, 15000)):
            continue
        g1 = (c[12000]["em"] - c[9000]["em"]) * 100
        g2 = (c[15000]["em"] - c[12000]["em"]) * 100
        note = "★仍未饱和（≥1pp）" if g2 >= 1.0 else "已平（<1pp）✓"
        if g2 >= 1.0:
            UNSAT.append(f"{a}/s{s}")
        print(f"[S40-SAT] {a} seed={s}: EM 9000={c[9000]['em']*100:.2f}% → 12000="
              f"{c[12000]['em']*100:.2f}% → 15000={c[15000]['em']*100:.2f}% "
              f"(Δ9000→12000={g1:+.2f}pp / Δ12000→15000={g2:+.2f}pp) ⇒ {note}", flush=True)
for s in SEEDS40:
    if not all((s, st) in BASE for st in (9000, 12000, 15000)):
        continue
    w1g = (BASE[(s, 15000)]["em"] - BASE[(s, 12000)]["em"]) * 100
    print(f"[S40-SAT] w1 seed={s}: EM 9000={BASE[(s,9000)]['em']*100:.2f}% → 12000="
          f"{BASE[(s,12000)]['em']*100:.2f}% → 15000={BASE[(s,15000)]['em']*100:.2f}% "
          f"(Δ12000→15000={w1g:+.2f}pp) [复用 S38 ckpt]", flush=True)
if UNSAT:
    print(f"[S40-SAT-VERDICT] ★未饱和（末段 12000→15000 ≥1pp 的臂={UNSAT}）⇒ "
          f"「加权有效/无效/有害」的结论不可下（R35）—— 需要更大预算", flush=True)
else:
    print("[S40-SAT-VERDICT] ★w3/w10 末段（12000→15000）增量均 <1pp ⇒ 已达饱和，判决可下",
          flush=True)

# ============================================================================
# ⑨ ★额外问题：加权是否只是改变收敛速度？（曲线形状 w1 vs w3 vs w10，每 750 步）
# ============================================================================
print(f"\n[S40] ===== ★额外问题：加权是否只是改变收敛速度？（w1 曲线来自 S38 日志，"
      f"每 {GRID[1]-GRID[0]} 步）=====", flush=True)
print("[S40-CURVE-CMP] step | w1(s1234) w3(s1234) w10(s1234) | w1(s5678) w3(s5678) w10(s5678)",
      flush=True)
for st in GRID:
    if st not in BASE_CURVE[1234]:
        continue
    cells = []
    for s in SEEDS40:
        w1v = BASE_CURVE[s].get(st, float("nan"))
        cells.append(f"{w1v*100:.2f}")
        for a in ARMS40:
            rr = RES.get((a, s))
            cells.append(f"{rr['curve'][st]['em']*100:.2f}" if rr and st in rr["curve"] else "-")
    print(f"[S40-CURVE-CMP] {st} | {' '.join(cells[:3])} | {' '.join(cells[3:])}", flush=True)

print(f"\n[S40] ===== ★收敛速度对比：ΔEM(w − w1) 随步数（pp；w1 取 S38 日志曲线，"
      f"@9000/12000/15000 与复用 ckpt 复核）=====", flush=True)
for a in ARMS40:
    for s in SEEDS40:
        rr = RES.get((a, s))
        if rr is None:
            continue
        pts = []
        for st in (3000, 6000, 9000, 12000, 15000):
            if st not in rr["curve"]:
                continue
            w1v = BASE_CURVE[s].get(st)
            pa = f"@{st}={((rr['curve'][st]['em']-w1v)*100):+.2f}" if w1v is not None else ""
            pb = ""
            if (s, st) in BASE:
                pb = f"(ckpt:{(rr['curve'][st]['em']-BASE[(s,st)]['em'])*100:+.2f})"
            pts.append(f"{pa}{pb}")
        print(f"[S40-SHAPE] {a} seed={s}: " + " ".join(pts), flush=True)
CATCH = {}
for a in ARMS40:
    CATCH[a] = {}
    for s in SEEDS40:
        rr = RES.get((a, s))
        bb = BASE.get((s, 15000))
        if rr is None or bb is None:
            continue
        d = paired_stats(rr["curve"][15000]["hits"], bb["hits"])
        CATCH[a][s] = d
if len(CATCH.get("w3", {})) == 2 and len(CATCH.get("w10", {})) == 2:
    allcatch = all(is_catch(d) for a in ARMS40 for d in CATCH[a].values())
    gap3000 = []
    for a in ARMS40:
        for s in SEEDS40:
            rr = RES.get((a, s))
            if rr and 3000 in rr["curve"] and 3000 in BASE_CURVE[s]:
                gap3000.append((rr["curve"][3000]["em"] - BASE_CURVE[s][3000]) * 100)
    if allcatch:
        print(f"[S40-SHAPE-VERDICT] ★w3/w10 在 @15000 与 w1 的 EM 差全部 |Δ|≤1SE ⇒ "
              f"「追平」；而在 @3000 落后（Δ w−w1 = "
              f"{min(gap3000):+.2f}..{max(gap3000):+.2f}pp）⇒ ★加权 = 收敛速度效应"
              f"（与 E6 的残差同型：只改变到达速度，不改变极限）", flush=True)
    else:
        print(f"[S40-SHAPE-VERDICT] ★@15000 未全部追平（存在 |Δ|>1SE 的臂/seed）⇒ "
              f"不能归为单纯收敛速度效应；@3000 落后 Δ={min(gap3000):+.2f}.."
              f"{max(gap3000):+.2f}pp，@15000 见上方 Δ±SE", flush=True)

# ============================================================================
# ⑩ 成本与 ckpt
# ============================================================================
print("\n[S40] ===== 成本与 ckpt =====", flush=True)
for a in ARMS40:
    for s in SEEDS40:
        r = RES.get((a, s))
        if r is None:
            continue
        for st in sorted(r["ckpt"]):
            print(f"[META] {a} seed={s}: steps={r['steps']} ckpt="
                  f"{os.path.basename(r['ckpt'][st]['path'])} "
                  f"sha256[:16]={r['ckpt'][st]['sha']}", flush=True)
        print(f"[META] {a} seed={s}: 训练墙钟={r['wall_train']/60:.2f}min "
              f"本run={r['wall_total']/60:.2f}min", flush=True)
_r28s = [(f"{a}/s{s}", RES[(a, s)]["r28_same"], RES[(a, s)]["r28_k"])
         for a in ARMS40 for s in SEEDS40 if (a, s) in RES]
print(f"[S40-R28] batch=1 vs batch={CHK_BATCH} 逐字一致：" +
      " | ".join(f"{t}={v}/{k}" for t, v, k in _r28s), flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"D={D} FF={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} | EM口径=batch{EM_BATCH} "
      f"自检=batch{CHK_BATCH} | TEST sha16={TEST_SHA} 泄漏={LEAK}/{len(TEST)} | 地板(单/top5)="
      f"{FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% | 加权口径=sum(w·CE)/sum(w) | 锚点复现="
      f"{'是' if ANCHOR_OK else '否(点名)'} | 总墙钟={(time.time()-T0)/60:.1f}min | run 数="
      f"{len(RES)}/{len(SEEDS40)*len(ARMS40)}（+w1 复用 {len(BASE)} 点）", flush=True)
print("[DONE] exit=0", flush=True)
sys.exit(0)
