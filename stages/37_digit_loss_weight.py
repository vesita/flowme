#!/usr/bin/env python3
"""S37 · 新主线第一枪：数字/算子 token 的 loss 加权 —— 在【饱和点 6000 步】检验。

背景（两条独立线索 + 一条实测）：
  · S13 自己提过「数字 token loss 加权」（logs/13_cap.log:252，方向 (a)）；
  · 独立视角 FRESH_VIEW 方案 4 同指；
  · S10 实测：数字/算子 token 的 CE 显著高于模板 token（0.9694/0.9738 vs 0.0802/0.0939）。
E1/E5/E6 已定：add_3d 在 3000 步是欠训区，6000 步才饱和 ⇒ 旧结论要在饱和点上重做。
本单元只回答一个问题：★在 6000 步（饱和点）上，把数字/算子 token 的 loss 加权，
能否提升 EM 与「数字每步正确率」？

设计（★只改 loss，不改结构/数据/超参）：
  w1  = 全部 token 权重 1（= E6/S36 的 res 臂，逐字相同）；
  w3  = 数字/算子 token（classify()==0，即 token 全由 [0-9+-*/=%$] 组成）权重 3，其余 1；
  w10 = 同上权重 10。
  加权口径 = 加权平均（分子 sum(w·CE)，分母 sum(w)，同 torch CrossEntropyLoss(weight, reduction='mean')）。
  3 臂 × 2 seed（1234/5678）= 6 run × 6000 步；
  d=128 ff=512 NHEAD=4 lr=1e-3 batch=32 MAXLEN=512 思维卡带残差（h = h + Think(h)）；
  数据只用 add_3d（train4000/test800，★不切 dev，固定末点，未用 dev 选步）。

★ 口径（缺一即无效）：
  1 EM 一律 batch=1（R28），并报 batch=1 vs 16 逐字一致性自检；
  2 ★饱和点（R35）：每臂报 @3000 / @4500 / @6000 的 EM ⇒ 声明是否已达饱和；
    未达饱和不得下「必要/有效」结论；
  3 ★数字每步正确率 = exp(−数字/算子 token 的 CE)（样本级为主口径，另报 token 加权）；
  4 ★SEEN/UNSEEN 分层 EM（train4000 对 test 的 (a,b) 泄漏 16/800 = 2.0%）；
  5 ★地板用本桶：单答案 0.12% / top5 0.25%（不用 1.89%）；
  6 R29：干预（加权）的 Δ 非零比例；TEST sha256[:16] 必须 == b34e7ea515203227；
  7 锚点：w1 @6000 必须复现 E6 的 85.62%(1234) / 74.38%(5678)，Δ>1pp 要点名。

判据（写死，EM 与数字每步正确率各判一次，配对 ±SE，逐 test 样本配对）：
  Δ = 该臂 − w1 @6000：
    Δ ≥ 2SE 且 2 seed 同号 ⇒ 加权有效（报幅度）；|Δ| ≤ 1SE ⇒ 无效（如实报）；
    Δ ≤ −2SE 且 2 seed 同号 ⇒ 有害。
  ★若两臂在 6000 步仍在上升（@4500→@6000 仍有可观增益）⇒ 明确写「未饱和，结论不可下」。

复用：exec stages/19_residual_scan.py 到 "# ③ 主循环" 之前（⇒ 同 tokenizer、同 add_3d
train4000/test800 数据生成器、同 enc_layer/greedy_gen/classify/parse_ans/r28_selfcheck 口径）。
模型：与 S36 的 res 臂逐字同构（emb → in_enc → thought(残差) → head，构造顺序相同 ⇒ 同 seed 同初始化）。
ckpt：logs/37_ckpt_w<w>_s6000_seed<s>.pt（报 sha256 前16）。结果落 logs/37_results.jsonl。
只允许写：本文件、logs/s37_loss.log、logs/37_*、/tmp；既有 stages/*.py 只读。
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

SMOKE37 = os.environ.get("S37_SMOKE") == "1"
if SMOKE37:
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
STEPS37 = 30 if SMOKE37 else 6000
SNAP_ATS = (15, 22) if SMOKE37 else (3000, 4500)
SEEDS37 = (1234, 5678)
ARMS37 = ("w1", "w3", "w10")
W37 = {"w1": 1.0, "w3": 3.0, "w10": 10.0}

# ★临时核对开关（只用于「w1 是否与 S36 res 逐位同源」的短跑核对，正式跑不设）：
#   S37_STEPS=500 → 只跑 w1/seed1234，跑满 500 步后把 step=500 的 loss 与 S36 res 的
#   0.8401 对比（同数据同种子同结构 ⇒ 应逐位相同）。结果落 /tmp/37_verify.jsonl。
VSTEPS = int(os.environ.get("S37_STEPS", "0") or 0)
if VSTEPS:
    STEPS37 = VSTEPS
    SNAP_ATS = (max(1, VSTEPS // 2), max(2, VSTEPS * 3 // 4))
    ARMS37, SEEDS37 = ("w1",), (1234,)
    JSONL = "/tmp/37_verify.jsonl"
    print(f"[S37-VERIFY] 临时核对模式：steps={STEPS37} arms={ARMS37} seeds={SEEDS37} "
          f"（w1 step500 loss 应 == S36 res 0.8401，逐位同源核对）", flush=True)
SNAP_EM = SNAP_ATS[0]                                   # @3000：EM + 数字每步 + SEEN/UNSEEN
ARM_DESC = {
    "w1": "基线：全部 token 权重 1（= E6/S36 res 臂，逐字相同）",
    "w3": "数字/算子 token（classify==0）权重 3，其余 1",
    "w10": "数字/算子 token（classify()==0）权重 10，其余 1",
}
N_EVAL = 60 if SMOKE37 else N_TEST                      # 800
CKPT_TMPL = f"{ROOT}/logs/37_ckpt_{{arm}}_s{{steps}}_seed{{seed}}.pt"
JSONL = "/tmp/37_results_smoke.jsonl" if SMOKE37 else f"{ROOT}/logs/37_results.jsonl"
TEST_SHA_EXPECT = "b34e7ea515203227"
LEAK_EXPECT = 16
ANCHOR_E6 = {1234: 0.8562, 5678: 0.7438}                # E6/S36 res @6000 入库值

print(f"\n[S37] ★数字/算子 token loss 加权 @饱和点：桶={BUCKET} 步数={STEPS37} "
      f"快照@{SNAP_ATS} seeds={SEEDS37} arms={list(ARMS37)} W={W37} | 6 run | "
      f"d={D} ff={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} 思维卡残差 | "
      f"EM主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | ★不切 dev、固定末点 | smoke={SMOKE37}",
      flush=True)

# ============================================================================
# ② 数据核对（口径 2/3/4/5/6 + TEST 指纹）
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

# 加权覆盖率（干预的实际作用面）
_n_dig = sum(1 for r in TEST for t in r["t"] if classify(t) == 0)
_n_tok = sum(len(r["t"]) for r in TEST)
W_SHARE = _n_dig / _n_tok
_n_dig_tr = sum(1 for r in TRAIN for t in r["t"] if classify(t) == 0)
_n_tok_tr = sum(len(r["t"]) for r in TRAIN)

sha_ok = TEST_SHA == TEST_SHA_EXPECT
leak_ok = LEAK == LEAK_EXPECT
print(f"[S37-DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)}(eval {len(TEST_EVAL)}) "
      f"文本零重叠 | 答案域={min(answer_values(BUCKET))}..{max(answer_values(BUCKET))}"
      f"({len(answer_values(BUCKET))}个) | ★不切 dev ⇒ 训练集={len(TRAIN)}", flush=True)
print(f"[S37-CHK6] TEST 指纹 sha256[:16]={TEST_SHA} vs 入库 {TEST_SHA_EXPECT} ⇒ "
      f"{'逐位一致 ✓' if sha_ok else '★不一致'} | (a,b) 泄漏对 train={LEAK}/{len(TEST)} vs "
      f"入库 {LEAK_EXPECT} ⇒ {'一致 ✓' if leak_ok else '★不一致'}", flush=True)
print(f"[S37-CHK4] SEEN={N_SEEN} / UNSEEN={N_UNSEEN}（train 对 test 的 (a,b) 泄漏 "
      f"{100.0*LEAK/len(TEST):.2f}%，分层报 EM）", flush=True)
print(f"[S37-CHK5] add_3d 本桶地板：单答案 '{_top1}'(train {_top1n}/4000) 覆盖 test "
      f"{FLOOR_1*100:.2f}% | top5={_top5} 覆盖 test {FLOOR_5*100:.2f}% ⇒ ★本地板="
      f"{FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}%（不借用 1.89%）", flush=True)
print(f"[S37-CHK-INT] 加权作用面：数字/算子 token 占 target token {100.0*W_SHARE:.1f}%"
      f"(test {_n_dig}/{_n_tok}) / {100.0*_n_dig_tr/_n_tok_tr:.1f}%(train) ⇒ "
      f"w1/w3/w10 的非 1 权重覆盖率相同，唯一变量是权重值", flush=True)
if not SMOKE37:
    assert sha_ok and leak_ok, "数据指纹不符 ⇒ 与 S19/S36 不是同一数据集，本单元无效"


# ============================================================================
# ③ 模型：与 S36 的 res 臂逐字同构（构造顺序 emb → in_enc → thought → head）
# ============================================================================
class Cards37(nn.Module):
    """h = h + Think(h)（k=1），与 S36 Cards36("res"/"A") 同一计算图、同一初始化顺序。"""

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


def n_params_37():
    mdl = Cards37(D, FF)
    n = sum(p.numel() for p in mdl.parameters())
    del mdl
    return n


def weighted_ce(model, recs, w) -> torch.Tensor:
    """★本单元唯一改动：数字/算子 token 的 CE 乘 w，其余乘 1；分母 = 权重之和。
    w==1 时走与 S19 masked_ce 逐字相同的算子序列（保证 w1 与 S36 res 逐位同源）。"""
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


def train37(arm, seed, train, steps, snap_ats):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards37(D, FF).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0 = 0, 0, time.time()
    last, snaps = 0.0, {}
    w = W37[arm]
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
        if step in snap_ats:
            snaps[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step % 500 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train:{arm}:w={w:g}] step={step}/{steps} loss={loss.item():.4f} "
                  f"elapsed={el:.0f}s s/step={(el - last) / (step % 500 or 500):.3f}", flush=True)
            last = el
    assert set(snap_ats) <= set(snaps), "未取到全部快照"
    return model, time.time() - t0, snaps


@torch.no_grad()
def em_at(model, recs):
    txt = greedy_gen(model, recs, batch=EM_BATCH)          # ★R28 主口径 batch=1
    return [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, recs)], txt


@torch.no_grad()
def per_sample_ce(model, recs):
    """逐样本 (整体CE, 数字/算子 token 的 CE)；口径 = S19 ce_categories 的 samp_all / dig_samp。"""
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
    """样本级：a_i = exp(−CE_i)，主口径返回 (均值, SE, CE均值, CE的SE, n)。"""
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


# ============================================================================
# ④ 主循环：seed 外层 ⇒ 同 seed 内 3 臂紧挨着跑，配对最干净
# ============================================================================
print(f"[S37-PARAMS] Cards37(res)={n_params_37()/1e6:.3f}M（3 臂参数量完全相同，只差 loss 权重）",
      flush=True)
RESULTS: dict = {}
open(JSONL, "w").close()

for seed in SEEDS37:
    for arm in ARMS37:
        t0 = time.time()
        print(f"\n[S37] ===== arm={arm}(w={W37[arm]:g}) seed={seed} steps={STEPS37} | "
              f"train={len(TRAIN)} test={len(TEST_EVAL)} | {ARM_DESC[arm]} =====", flush=True)
        model, wall_train, snaps = train37(arm, seed, TRAIN, STEPS37, SNAP_ATS)
        ck, dig = "-", "-"
        if not SMOKE37 and not VSTEPS:
            ck = CKPT_TMPL.format(arm=arm, steps=STEPS37, seed=seed)
            torch.save(model.state_dict(), ck)
            dig = sha16(ck)
            print(f"[CKPT] saved {ck} sha256[:16]={dig}", flush=True)

        # ---- @6000（末点，主口径 batch=1）----
        s6, txt6 = em_at(model, TEST_EVAL)
        em6 = sum(s6) / len(TEST_EVAL)
        em6_se = math.sqrt(em6 * (1 - em6) / len(TEST_EVAL))
        seen6 = [s6[i] for i, r in enumerate(TEST_EVAL) if r["seen"]]
        unseen6 = [s6[i] for i, r in enumerate(TEST_EVAL) if not r["seen"]]
        all_ce6, dig_ce6, dig_tok6 = per_sample_ce(model, TEST_EVAL)
        acc6, acc6_se, dce6, dce6_se, dn6 = dig_acc(dig_ce6)
        acc6_tok = math.exp(-(sum(dig_tok6) / len(dig_tok6)))
        _r28 = r28_selfcheck(model, TEST_EVAL, tag=f"[{arm} s{STEPS37} seed{seed}] ",
                             k=min(CHK_BATCH, len(TEST_EVAL)))
        r28s, r28k = _r28["same"], _r28["k"]
        acc_samp6 = [math.exp(-x) if x == x else float("nan") for x in dig_ce6]

        # ---- @3000（同一条 6000 步轨迹的快照；★不是 dev 选步，固定末点）----
        model.load_state_dict({k: v.to(device) for k, v in snaps[SNAP_EM].items()})
        s3, _ = em_at(model, TEST_EVAL)
        em3 = sum(s3) / len(TEST_EVAL)
        em3_se = math.sqrt(em3 * (1 - em3) / len(TEST_EVAL))
        seen3 = [s3[i] for i, r in enumerate(TEST_EVAL) if r["seen"]]
        unseen3 = [s3[i] for i, r in enumerate(TEST_EVAL) if not r["seen"]]
        _, dig_ce3, dig_tok3 = per_sample_ce(model, TEST_EVAL)
        acc3, acc3_se, dce3, _, _ = dig_acc(dig_ce3)
        acc_samp3 = [math.exp(-x) if x == x else float("nan") for x in dig_ce3]

        # ---- @4500（只取 EM，用来看饱和曲线的形状，不影响训练）----
        model.load_state_dict({k: v.to(device) for k, v in snaps[SNAP_ATS[1]].items()})
        s45, _ = em_at(model, TEST_EVAL)
        em45 = sum(s45) / len(TEST_EVAL)
        del snaps, model
        torch.cuda.empty_cache()

        r = dict(arm=arm, w=W37[arm], seed=seed, steps=STEPS37, snap_ats=list(SNAP_ATS),
                 train_n=len(TRAIN), wall_train=wall_train, wall_total=time.time() - t0,
                 em=em6, em_se=em6_se, em3000=em3, em3000_se=em3_se, em4500=em45,
                 em_seen=frac(seen6), em_unseen=frac(unseen6),
                 em_seen3000=frac(seen3), em_unseen3000=frac(unseen3),
                 n_seen=len(seen6), n_unseen=len(unseen6),
                 acc=acc6, acc_se=acc6_se, acc_tok=acc6_tok, dig_ce=dce6, dig_ce_se=dce6_se,
                 dig_n=dn6, acc3000=acc3, acc3000_se=acc3_se, dig_ce3000=dce3,
                 ce6=sum(all_ce6) / len(all_ce6), r28_same=r28s, r28_k=r28k,
                 ckpt=ck, sha=dig, strict6=s6, strict3=s3, acc_samp6=acc_samp6,
                 acc_samp3=acc_samp3, pred=[parse_ans(t) for t in txt6])
        RESULTS[(arm, seed)] = r
        add_jsonl({k: v for k, v in r.items()
                   if k not in ("strict6", "strict3", "acc_samp6", "acc_samp3", "pred")})
        print(f"[MAIN-TABLE] {arm} s{STEPS37} seed={seed} | EM6000={em6*100:.2f}%±{em6_se*100:.2f}"
              f"(n={len(TEST_EVAL)}) SEEN={frac(seen6)*100:.2f}%({len(seen6)}) "
              f"UNSEEN={frac(unseen6)*100:.2f}%({len(unseen6)}) | EM3000={em3*100:.2f}% "
              f"EM4500={em45*100:.2f}% | 数字每步 3000={acc3*100:.2f}% → 6000={acc6*100:.2f}%"
              f"±{acc6_se*100:.2f}pp(token加权={acc6_tok*100:.2f}%) | 数字CE6000={dce6:.4f} | "
              f"R28={r28s}/{r28k} | 训练={wall_train/60:.1f}min 合计={r['wall_total']/60:.1f}min | "
              f"sha16={dig}", flush=True)

# ============================================================================
# ⑤ 汇总：三臂表 + Δ±SE 判决 + 饱和检查 + 锚点
# ============================================================================
print("\n[S37] ===== ★三臂 × 2 seed 表（主口径 EM=batch1）=====", flush=True)
print("[S37] arm seed | EM@6000 | 数字每步@6000 | SEEN | UNSEEN | EM@3000 | 数字每步@3000 | "
      "EM@4500 | 整体CE", flush=True)
for a in ARMS37:
    for s in SEEDS37:
        r = RESULTS.get((a, s))
        if r is None:
            continue
        print(f"[S37-T] {a} {s} | {r['em']*100:.2f}%±{r['em_se']*100:.2f} | "
              f"{r['acc']*100:.2f}%±{r['acc_se']*100:.2f}pp | {r['em_seen']*100:.2f}%"
              f"({r['n_seen']}) | {r['em_unseen']*100:.2f}%({r['n_unseen']}) | "
              f"{r['em3000']*100:.2f}% | {r['acc3000']*100:.2f}% | {r['em4500']*100:.2f}% | "
              f"{r['ce6']:.4f}", flush=True)

print("\n[S37] ===== ★锚点：w1 @6000 复现 E6/S36 res =====", flush=True)
ANCHOR_OK = {}
for s in SEEDS37:
    r = RESULTS.get(("w1", s))
    if r is None:
        continue
    dd = (r["em"] - ANCHOR_E6[s]) * 100
    ANCHOR_OK[s] = abs(dd) <= 1.0
    print(f"[S37-ANCHOR] w1 seed={s}: 本次={r['em']*100:.2f}% vs E6={ANCHOR_E6[s]*100:.2f}% ⇒ "
          f"Δ={dd:+.2f}pp {'✓ 复现' if ANCHOR_OK[s] else '★>1pp，点名'}", flush=True)

print("\n[S37] ===== ★判决①（主指标）：Δ = 数字每步正确率(w) − (w1) @6000，逐样本配对 =====",
      flush=True)
DACC = {}
for a in ("w3", "w10"):
    DACC[a] = {}
    for s in SEEDS37:
        if (a, s) not in RESULTS or ("w1", s) not in RESULTS:
            continue
        d = paired_stats_skipnan(RESULTS[(a, s)]["acc_samp6"], RESULTS[("w1", s)]["acc_samp6"])
        DACC[a][s] = d
        print(f"[S37-DACC] {a}−w1 seed={s}: Δacc={d['d']*100:+.2f}±{d['se']*100:.2f}pp "
              f"Δ/SE={d['ratio']:.2f} | R29 Δ非零比={d['nz']*100:.1f}% | "
              f"w1={RESULTS[('w1', s)]['acc']*100:.2f}% → {a}="
              f"{RESULTS[(a, s)]['acc']*100:.2f}% | {delta_tag(d)}", flush=True)
for a in ("w3", "w10"):
    ds = DACC[a]
    if len(ds) == 2 and all(is_pos2(d) for d in ds.values()) and all(d["d"] > 0 for d in ds.values()):
        v = "★加权有效（数字每步正确率 ≥+2SE 且 2 seed 同号）"
    elif len(ds) == 2 and all(is_neg2(d) for d in ds.values()) and all(d["d"] < 0 for d in ds.values()):
        v = "★加权有害（数字每步正确率 ≤−2SE 且 2 seed 同号）"
    elif len(ds) == 2 and all(is_catch(d) for d in ds.values()):
        v = "★无效（|Δ|≤1SE，如实报）"
    else:
        v = "★未过写死判据（既非 2SE 显著，也非 |Δ|≤1SE）⇒ 如实报"
    print(f"[S37-VERDICT-ACC] {a}: {v}", flush=True)

print("\n[S37] ===== ★判决②：Δ = EM(w) − EM(w1) @6000，逐样本配对（二值 d∈{−1,0,1}）=====",
      flush=True)
DEM = {}
for a in ("w3", "w10"):
    DEM[a] = {}
    for s in SEEDS37:
        if (a, s) not in RESULTS or ("w1", s) not in RESULTS:
            continue
        d = paired_stats(RESULTS[(a, s)]["strict6"], RESULTS[("w1", s)]["strict6"])
        DEM[a][s] = d
        print(f"[S37-DEM] {a}−w1 seed={s}: ΔEM={d['d']*100:+.2f}±{d['se']*100:.2f}pp "
              f"Δ/SE={d['ratio']:.2f} | R29 Δ非零比={d['nz']*100:.1f}% | "
              f"w1={RESULTS[('w1', s)]['em']*100:.2f}% → {a}="
              f"{RESULTS[(a, s)]['em']*100:.2f}% | {delta_tag(d)}", flush=True)
for a in ("w3", "w10"):
    ds = DEM[a]
    if len(ds) == 2 and all(is_pos2(d) for d in ds.values()) and all(d["d"] > 0 for d in ds.values()):
        v = "★加权有效（EM ≥+2SE 且 2 seed 同号）"
    elif len(ds) == 2 and all(is_neg2(d) for d in ds.values()) and all(d["d"] < 0 for d in ds.values()):
        v = "★加权有害（EM ≤−2SE 且 2 seed 同号）"
    elif len(ds) == 2 and all(is_catch(d) for d in ds.values()):
        v = "★无效（|Δ|≤1SE，如实报）"
    else:
        v = "★未过写死判据（既非 2SE 显著，也非 |Δ|≤1SE）⇒ 如实报"
    print(f"[S37-VERDICT-EM] {a}: {v}", flush=True)

print("\n[S37] ===== ★R35 饱和检查：每臂 EM 3000 → 4500 → 6000，末段仍在升？ =====", flush=True)
SAT_RISING = []
for a in ARMS37:
    for s in SEEDS37:
        r = RESULTS.get((a, s))
        if r is None:
            continue
        g1 = (r["em4500"] - r["em3000"]) * 100
        g2 = (r["em"] - r["em4500"]) * 100
        note = "仍在大幅上升" if max(g1, g2) > 10 else ("仍在升" if max(g1, g2) > 1 else "已平")
        if g2 > 1.0:
            SAT_RISING.append(f"{a}/s{s}")
        print(f"[S37-SAT] {a} seed={s}: {r['em3000']*100:.2f}→{r['em4500']*100:.2f}→"
              f"{r['em']*100:.2f}% (Δ+{g1:.2f}pp / Δ+{g2:.2f}pp) ⇒ {note}", flush=True)
if SAT_RISING:
    SAT_V = (f"★末段（4500→6000）仍在上升的臂={SAT_RISING} ⇒ 未饱和，"
             f"「权重有效/无效」的结论不可下（R35）")
else:
    SAT_V = "★全部臂在 4500→6000 基本持平（≤1pp）⇒ 视为已达饱和，判决可下"
print(f"[S37-SAT-VERDICT] {SAT_V}", flush=True)
for s in SEEDS37:
    r = RESULTS.get(("w1", s))
    if r is None:
        continue
    _g = {a: RESULTS.get((a, s)) for a in ("w1", "w3", "w10")}
    if any(v is None for v in _g.values()):
        continue
    _d = lambda a, key: (_g[a][key] - r[key]) * 100          # noqa: E731
    print(f"[S37-SAT-CMP] seed={s}: w3/w10 与 w1 的差距在 3000/4500/6000 三点分别是 "
          f"{_d('w3','em3000'):+.2f}/{_d('w3','em4500'):+.2f}/{_d('w3','em'):+.2f}pp 与 "
          f"{_d('w10','em3000'):+.2f}/{_d('w10','em4500'):+.2f}/{_d('w10','em'):+.2f}pp",
          flush=True)

print("\n[S37] ===== 成本与 ckpt =====", flush=True)
for a in ARMS37:
    for s in SEEDS37:
        r = RESULTS.get((a, s))
        if r is None:
            continue
        print(f"[META] {a} seed={s}: steps={STEPS37} 训练墙钟={r['wall_train']/60:.2f}min "
              f"本run={r['wall_total']/60:.2f}min ckpt={os.path.basename(r['ckpt'])} "
              f"sha256[:16]={r['sha']}", flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"D={D} FF={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} | EM口径=batch{EM_BATCH} "
      f"自检=batch{CHK_BATCH} | TEST sha16={TEST_SHA} 泄漏={LEAK}/{len(TEST)} | 地板(单/top5)="
      f"{FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% | 加权口径=sum(w·CE)/sum(w) | "
      f"总墙钟={(time.time()-T0)/60:.1f}min | run 数={len(RESULTS)}/{len(SEEDS37)*len(ARMS37)}",
      flush=True)
print(f"[S37-SUMMARY] 饱和={SAT_V.replace('★', '')} | 逐条判定见 [S37-VERDICT-ACC]/[S37-VERDICT-EM]",
      flush=True)
print("[DONE] exit=0", flush=True)
sys.exit(0)
