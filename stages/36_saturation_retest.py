#!/usr/bin/env python3
"""S36 · E6 ★连锁影响修正：在饱和点（6000 步）重测两条最硬的结论。

E1（stages/35_mono_vs_cards.py）实测：add_3d 在 3000 步是欠训区（cards 33.88/13.88），
6000 步饱和（90.75/88.12）；dev 曲线 0.0 → 29.2 → 79.5 → 87.5 → 90.2。
⇒ 此前大量结论都在 3000 步测的，本单元在 6000 步重测两条最硬的：
  ① 思维卡必须带残差（S19：nores 2.50/9.62 vs res 37.25/31.13 @3000）
  ② 卡内结构 = 标准 Transformer block（S22：A 37.25/31.13 vs B/C/D/E ≤1.25% @3000）

设计（两条都做，各自对照）：① {nores,res}×6000×2seed = 4 run；
② {A,B,C,D,E}×6000×2seed = 10 run；共 14 run。
其余与 S19-res / S22 逐字相同（d=128 ff=512 NHEAD=4 lr=1e-3 batch=32 MAXLEN=512 思维卡带残差）。
数据只用 add_3d（train4000/test800）；★不切 dev；★固定 6000 步末点，未用 dev 选步。

★ 口径（缺一即无效）：
  1 EM 一律 batch=1（R28：batch=N 差 8 倍），并报 batch=1 vs 16 逐字一致性自检；
  2 SEEN/UNSEEN 分层 EM（train4000 对 test 的 (a,b) 泄漏 16/800 = 2.0%）；
  3 本桶自己的地板（add_3d：单答案 0.25% / top5 0.62%，不用 1.89%）；
  4 与 3000 步旧值并列（同一条 6000 步轨迹的 @3000 快照 + S19/S22 入库旧值）；
  5 饱和检查：每臂 @3000/6000 的 EM；
  6 R29：任何干预的 Δ 非零比例；TEST sha256[:16] 必须 == b34e7ea515203227。

判据（写死）：
  ① Δ = EM(res) − EM(nores) @6000 配对 ±SE：Δ≥2SE 且 2 seed 同号 ⇒ 残差仍必要；
     |Δ|≤1SE ⇒ 残差只是加速收敛、饱和后消失（结论要改）。
  ② 每臂 vs A 的配对 Δ±SE：B/C/D/E 全部 ≤−2SE 且 2 seed 同号 ⇒ 不可替代仍成立；
     有任一臂 |Δ|≤1SE ⇒ 那条「不可替代」是欠训假象（如实报）。

复用：exec stages/19_residual_scan.py 到 "# ③ 主循环" 之前（⇒ 同 tokenizer、
同 add_3d train4000/test800 数据生成器、同 enc_layer/greedy_gen/hits_from/classify 口径）。
ckpt：logs/36_ckpt_<臂>_s6000_seed<s>.pt（报 sha256 前16）。结果落 logs/36_results.jsonl。
只允许写：本文件、logs/e6_sat.log、logs/36_*、/tmp；既有 stages/*.py 只读。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter

ROOT = "/home/vesita/coding/my/flowme"
SRC19 = f"{ROOT}/stages/19_residual_scan.py"

SMOKE36 = os.environ.get("S36_SMOKE") == "1"
if SMOKE36:
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
FP = 0.1
BUCKET = "add_3d"
STEPS36 = 30 if SMOKE36 else 6000
SNAP_AT = 15 if SMOKE36 else 3000
SEEDS36 = (1234, 5678)
GRP1 = ("nores", "res")
GRP2 = ("A", "B", "C", "D", "E")
ARMS36 = GRP1 + GRP2
N_EVAL = 60 if SMOKE36 else N_TEST          # 800
CKPT_TMPL = f"{ROOT}/logs/36_ckpt_{{arm}}_s{{steps}}_seed{{seed}}.pt"
JSONL = "/tmp/36_results_smoke.jsonl" if SMOKE36 else f"{ROOT}/logs/36_results.jsonl"
TEST_SHA_EXPECT = "b34e7ea515203227"
LEAK_EXPECT = 16
S19_OLD = {("nores", 1234): 0.0250, ("nores", 5678): 0.0962,     # S19 @3000 旧值
           ("res", 1234): 0.3725, ("res", 5678): 0.3113}
S22_OLD = {("A", 1234): 0.3725, ("A", 5678): 0.3113,             # S22 @3000 旧值
           ("B", 1234): 0.0000, ("B", 5678): 0.0038,
           ("C", 1234): 0.0012, ("C", 5678): 0.0100,
           ("D", 1234): 0.0050, ("D", 5678): 0.0075,
           ("E", 1234): 0.0062, ("E", 5678): 0.0125}
ARM_DESC = {
    "nores": "S19 现状：h = Think(h)（无残差）",
    "res": "S19 res：h = h + Think(h)（TransformerEncoderLayer）",
    "A": "现状=S19-res：h + TransformerEncoderLayer(MHA4+FFN512/gelu+2×LN)",
    "B": "纯逐位置 MLP：h + MLP(LN(h))（零跨位置混合）",
    "C": "因果卷积 k=3：h + Linear(GELU(DepthwiseConv1d(128,k=3,左填充)))",
    "D": "门控 GRU 风格：g=sigmoid(Wg·LN(h)), c=tanh(MLP(LN(h))); g⊙c+(1−g)⊙h",
    "E": "仅注意力（去 FFN）：h + Dropout(MHA(LN(h)))",
}

print(f"\n[S36] ★E6 饱和点重测：桶={BUCKET} 步数={STEPS36} snap@{SNAP_AT} seeds={SEEDS36} "
      f"arms={list(ARMS36)} | 14 run = ①4(nores/res) + ②10(A..E) | d={D} ff={FF} NHEAD={NHEAD} "
      f"lr={LR} batch={BATCH} MAXLEN={MAXLEN} | EM主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | "
      f"★不切 dev、固定 {STEPS36} 步末点（未用 dev 选步）| smoke={SMOKE36}", flush=True)

# ============================================================================
# ② 数据核对（口径 2/3 + TEST 指纹）
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

sha_ok = TEST_SHA == TEST_SHA_EXPECT
leak_ok = LEAK == LEAK_EXPECT
print(f"[S36-DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)}(eval {len(TEST_EVAL)}) 文本零重叠 | "
      f"答案域={min(answer_values(BUCKET))}..{max(answer_values(BUCKET))}"
      f"({len(answer_values(BUCKET))}个) | ★不切 dev ⇒ 训练集=4000（与 S19/S22 逐位同源）", flush=True)
print(f"[S36-CHK6] TEST 指纹 sha256[:16]={TEST_SHA} vs 入库 {TEST_SHA_EXPECT} ⇒ "
      f"{'逐位一致 ✓' if sha_ok else '★不一致'} | (a,b) 泄漏对 train4000={LEAK}/{len(TEST)} vs "
      f"入库 {LEAK_EXPECT} ⇒ {'一致 ✓' if leak_ok else '★不一致'}", flush=True)
print(f"[S36-CHK2] SEEN={N_SEEN} / UNSEEN={N_UNSEEN}（train4000 对 test 的 (a,b) 泄漏 "
      f"{100.0*LEAK/len(TEST):.2f}%，分层报 EM）", flush=True)
print(f"[S36-CHK3] add_3d 本桶地板：单答案 '{_top1}'(train4000 {_top1n}/4000) 覆盖 test "
      f"{FLOOR_1*100:.2f}% | top5={_top5} 覆盖 test {FLOOR_5*100:.2f}% ⇒ ★本地板="
      f"{FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}%（不借用 1.89%）", flush=True)
if not SMOKE36:
    assert sha_ok and leak_ok, "数据指纹不符 ⇒ 与 S19/S22/S35 不是同一数据集，本单元无效"


# ============================================================================
# ③ 五臂卡实现（逐字复制 S22）+ 统一容器 Cards36
# ============================================================================
class CardA(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.layer = enc_layer(d, ff)

    def forward(self, h, m):
        return h + self.layer(h, src_mask=m)


class CardB(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(FP),
                                 nn.Linear(ff, d), nn.Dropout(FP))

    def forward(self, h, m):
        return h + self.mlp(self.norm(h))


class CardC(nn.Module):
    def __init__(self, d, k=3):
        super().__init__()
        self.k = k
        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, k, groups=d, padding=k - 1)
        self.pw = nn.Linear(d, d)
        self.act = nn.GELU()
        self.do = nn.Dropout(FP)

    def forward(self, h, m):
        z = self.norm(h).transpose(1, 2)
        z = self.dw(z)[..., : h.size(1)].transpose(1, 2)
        return h + self.do(self.pw(self.act(z)))


class CardD(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Linear(d, d)
        self.cand = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, h, m):
        z = self.norm(h)
        g = torch.sigmoid(self.gate(z))
        c = torch.tanh(self.cand(z))
        return g * c + (1.0 - g) * h


class CardE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, NHEAD, dropout=FP, batch_first=True)
        self.do = nn.Dropout(FP)

    def forward(self, h, m):
        z = self.norm(h)
        o = self.att(z, z, z, attn_mask=m, need_weights=False)[0]
        return h + self.do(o)


def build_card(arm, d, ff):
    if arm == "C":
        return CardC(d)
    if arm == "E":
        return CardE(d)
    return {"A": CardA, "B": CardB, "D": CardD}[arm](d, ff)


HOOK_IN = {"fn": None}          # 思维卡【输入】侧干预（R29 用）


class Cards36(nn.Module):
    """与 S19 Cards / S22 Cards22 逐字同构：emb → in_enc → card(或 thought) → head。
    nores/res 走 S19 的 thought 分支（保证与 S19 逐位同源）；A..E 走 S22 的 card 分支。"""

    def __init__(self, arm, d, ff):
        super().__init__()
        self.d = d
        self.arm = arm
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        if arm in ("nores", "res"):
            self.thought = enc_layer(d, ff)
        else:
            self.card = build_card(arm, d, ff)
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
        if HOOK_IN["fn"] is not None:
            h = HOOK_IN["fn"](h)
        if self.arm in ("nores", "res"):
            t = self.thought(h, src_mask=m)
            h = h + t if self.arm == "res" else t
        else:
            h = self.card(h, m)
        return self.head(h)


def n_params_arm(arm):
    mdl = Cards36(arm, D, FF)
    n = sum(p.numel() for p in mdl.parameters())
    del mdl
    return n


# ---- ★一致性自检：A 与 res 是同一计算图（构造顺序相同 ⇒ 同 seed 初始化应逐位相同）----
torch.manual_seed(999)
_a = Cards36("A", D, FF)
torch.manual_seed(999)
_r = Cards36("res", D, FF)
_sda, _sdr = _a.state_dict(), _r.state_dict()
_map = {k: k.replace("card.layer.", "thought.") for k in _sda}
INIT_DIFF = max(float((_sda[k] - _sdr[_map[k]]).abs().max()) for k in _sda)
del _a, _r
print(f"[S36-INIT] 臂 A 与臂 res 同 seed(999) 初始化逐参数最大差={INIT_DIFF:.3e} "
      f"⇒ {'✓ 同一计算图（A≡res，互为确定性复现）' if INIT_DIFF == 0.0 else '★不同起点'}", flush=True)


def train36(arm, seed, train, steps, snap_at):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards36(arm, D, FF).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0 = 0, 0, time.time()
    last, snap = 0.0, None
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(model, [train[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step == snap_at:
            snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step % 500 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train:{arm}] step={step}/{steps} loss={loss.item():.4f} elapsed={el:.0f}s "
                  f"s/step={(el - last) / (step % 500 or 500):.3f}", flush=True)
            last = el
    return model, time.time() - t0, snap


@torch.no_grad()
def em_at(model, recs):
    txt = greedy_gen(model, recs, batch=EM_BATCH)          # ★R28 主口径 batch=1
    return [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, recs)], txt


@torch.no_grad()
def per_sample_ce(model, recs, hook=None):
    """逐样本 (整体CE, 数字/算子 token 的 CE)；口径 = S19 report_ce 的 samp_all / dig_samp。"""
    model.eval()
    HOOK_IN["fn"] = hook
    all_ce, dig_ce, dig_tok = [], [], []
    try:
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
    finally:
        HOOK_IN["fn"] = None
    model.train()
    return all_ce, dig_ce, dig_tok


def dig_acc(dig_ce):
    ys = [x for x in dig_ce if x == x]
    m, se = mean_se(ys)
    acc = math.exp(-m)
    return acc, acc * se, m, se, len(ys)


def iv_zero(x, *rest):
    return torch.zeros_like(x)


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


def is_catch(d):
    """追平：Δ≡0（SE=0）或 |Δ|≤1SE。"""
    return (d["se"] == 0 and d["d"] == 0) or (d["se"] > 0 and abs(d["d"]) <= d["se"])


def is_neg2(d):
    return d["se"] > 0 and d["d"] < 0 and d["d"] <= -2 * d["se"]


def is_pos2(d):
    return d["se"] > 0 and d["d"] > 0 and d["d"] >= 2 * d["se"]


def delta_tag(d):
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


def paired_stats(xa, xb):
    d = [a - b for a, b in zip(xa, xb)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return dict(n=n, d=m, se=se, nz=sum(1 for x in d if x != 0) / n,
                ratio=(m / se if se > 0 else float("inf")))


# ============================================================================
# ④ 主循环：seed 外层 ⇒ 同 seed 内 7 臂紧挨着跑，配对最干净
# ============================================================================
PARAMS = {a: n_params_arm(a) for a in ARMS36}
print(f"[S36-PARAMS] " + " | ".join(f"{a}={PARAMS[a]/1e6:.3f}M" for a in ARMS36), flush=True)
RESULTS: dict = {}
open(JSONL, "w").close()

for seed in SEEDS36:
    for arm in ARMS36:
        t0 = time.time()
        print(f"\n[S36] ===== arm={arm} seed={seed} steps={STEPS36} | train={len(TRAIN)} "
              f"test={len(TEST_EVAL)} | {ARM_DESC[arm]} =====", flush=True)
        model, wall_train, snap = train36(arm, seed, TRAIN, STEPS36, SNAP_AT)
        assert snap is not None, "未取到 @3000 快照"
        ck = CKPT_TMPL.format(arm=arm, steps=STEPS36, seed=seed)
        dig = "-"
        if not SMOKE36:
            torch.save(model.state_dict(), ck)
            dig = sha16(ck)
            print(f"[CKPT] saved {ck} sha256[:16]={dig}", flush=True)

        # ---- @6000（末点，主口径 batch=1）----
        s6, txt6 = em_at(model, TEST_EVAL)
        em6 = sum(s6) / len(TEST_EVAL)
        em6_se = math.sqrt(em6 * (1 - em6) / len(TEST_EVAL))
        seen6 = [s6[i] for i, r in enumerate(TEST_EVAL) if r["seen"]]
        unseen6 = [s6[i] for i, r in enumerate(TEST_EVAL) if not r["seen"]]
        all_ce6, dig_ce6, _ = per_sample_ce(model, TEST_EVAL)
        acc6, acc6_se, dce6, _, dn6 = dig_acc(dig_ce6)
        _r28 = r28_selfcheck(model, TEST_EVAL, tag=f"[{arm} s{STEPS36} seed{seed}] ",
                             k=min(CHK_BATCH, len(TEST_EVAL)))
        r28s, r28k = _r28["same"], _r28["k"]      # ★S19 的 r28_selfcheck 返回 dict（不是元组）
        _, dig_iv6, _ = per_sample_ce(model, TEST_EVAL, hook=iv_zero)
        a_iv = [math.exp(-x) if x == x else float("nan") for x in dig_iv6]
        a_b = [math.exp(-x) for x in dig_ce6]
        r29_iv = sum(1 for x, y in zip(a_iv, a_b)
                     if x == x and abs(x - y) > 1e-3) / len(a_b)

        # ---- @3000（同一条 6000 步轨迹的快照；★不是 dev 选步，固定末点）----
        model.load_state_dict({k: v.to(device) for k, v in snap.items()})
        s3, _ = em_at(model, TEST_EVAL)
        em3 = sum(s3) / len(TEST_EVAL)
        em3_se = math.sqrt(em3 * (1 - em3) / len(TEST_EVAL))
        seen3 = [s3[i] for i, r in enumerate(TEST_EVAL) if r["seen"]]
        unseen3 = [s3[i] for i, r in enumerate(TEST_EVAL) if not r["seen"]]
        _, dig_ce3, _ = per_sample_ce(model, TEST_EVAL)
        acc3, acc3_se, dce3, _, _ = dig_acc(dig_ce3)
        del snap, model
        torch.cuda.empty_cache()

        r = dict(arm=arm, seed=seed, steps=STEPS36, snap_at=SNAP_AT, train_n=len(TRAIN),
                 params=PARAMS[arm], wall_train=wall_train, wall_total=time.time() - t0,
                 em=em6, em_se=em6_se, em3000=em3, em3000_se=em3_se,
                 em_seen=frac(seen6), em_unseen=frac(unseen6),
                 em_seen3000=frac(seen3), em_unseen3000=frac(unseen3),
                 n_seen=len(seen6), n_unseen=len(unseen6),
                 acc=acc6, acc_se=acc6_se, dig_ce=dce6, dig_n=dn6,
                 acc3000=acc3, acc3000_se=acc3_se, dig_ce3000=dce3,
                 ce6=sum(all_ce6) / len(all_ce6), r29_iv_nz=r29_iv,
                 r28_same=r28s, r28_k=r28k, ckpt=ck, sha=dig,
                 strict=s6, strict3000=s3, pred=[parse_ans(t) for t in txt6])
        RESULTS[(arm, seed)] = r
        add_jsonl({k: v for k, v in r.items() if not k.startswith("strict") and k != "pred"})
        print(f"[MAIN-TABLE] {arm} s{STEPS36} seed={seed} | EM6000={em6*100:.2f}%±{em6_se*100:.2f}"
              f"(n={len(TEST_EVAL)}) SEEN={frac(seen6)*100:.2f}%({len(seen6)}) "
              f"UNSEEN={frac(unseen6)*100:.2f}%({len(unseen6)}) | "
              f"EM3000={em3*100:.2f}%±{em3_se*100:.2f} | 数字每步 3000={acc3*100:.2f}% → "
              f"6000={acc6*100:.2f}%±{acc6_se*100:.2f}pp | R29干预Δ非零比={r29_iv*100:.1f}% | "
              f"R28={r28s}/{r28k} | 训练={wall_train/60:.1f}min 合计={r['wall_total']/60:.1f}min | "
              f"sha16={dig}", flush=True)

# ============================================================================
# ⑤ 汇总：两栏对照 + Δ±SE + 逐条判定 + 饱和检查
# ============================================================================
print("\n[S36] ===== ★两栏对照表：3000 步（旧值/本次同轨迹快照） vs 6000 步新值 =====", flush=True)
print("[S36] 臂 | 3000旧值(S19/S22) seed1234/5678 | 3000本次快照 1234/5678 | "
      "6000新值 1234/5678 | 6000均值", flush=True)
for a in ARMS36:
    r1, r2 = RESULTS.get((a, 1234)), RESULTS.get((a, 5678))
    old = S19_OLD if a in GRP1 else S22_OLD
    o1, o2 = old.get((a, 1234)), old.get((a, 5678))
    new3 = f"{r1['em3000']*100:.2f}/{r2['em3000']*100:.2f}" if r1 and r2 else "-"
    new6 = f"{r1['em']*100:.2f}/{r2['em']*100:.2f}" if r1 and r2 else "-"
    m6 = f"{(r1['em']+r2['em'])/2*100:.2f}" if r1 and r2 else "-"
    print(f"[S36] {a} | {o1*100:.2f}/{o2*100:.2f} | {new3} | {new6} | {m6}", flush=True)

print("\n[S36] ===== ★锚点核对：本次 @3000 快照 vs S19/S22 入库 @3000（应逐位复现）=====",
      flush=True)
for a in ARMS36:
    old = S19_OLD if a in GRP1 else S22_OLD
    for s in SEEDS36:
        rr = RESULTS.get((a, s))
        if rr is None:
            continue
        print(f"[S36-ANCHOR] {a} seed={s}: 入库={old[(a, s)]*100:.2f}% vs 本次@3000="
              f"{rr['em3000']*100:.2f}% ⇒ Δ={(rr['em3000']-old[(a, s)])*100:+.2f}pp", flush=True)

print("\n[S36] ===== ★① 残差判决：Δ = EM(res) − EM(nores) @6000（配对 ±SE）=====", flush=True)
D1 = {}
for s in SEEDS36:
    if ("res", s) not in RESULTS or ("nores", s) not in RESULTS:
        continue
    d = paired_stats(RESULTS[("res", s)]["strict"], RESULTS[("nores", s)]["strict"])
    D1[s] = d
    print(f"[S36-D1] seed={s}: Δ={d['d']*100:+.2f}±{d['se']*100:.2f}pp Δ/SE={d['ratio']:.2f} | "
          f"R29 Δ非零比={d['nz']*100:.1f}% | nores={RESULTS[('nores', s)]['em']*100:.2f}% → "
          f"res={RESULTS[('res', s)]['em']*100:.2f}% | {delta_tag(d)}", flush=True)
if len(D1) == 2:
    same = all(d["d"] > 0 for d in D1.values())
    all2 = all(is_pos2(d) for d in D1.values())
    all1 = all(is_catch(d) for d in D1.values())
    all0 = all(d["se"] == 0 and d["d"] == 0 for d in D1.values())
    if all2 and same:
        V1 = "★残差仍必要（饱和区成立）"
    elif all0:
        V1 = "★两臂 EM 逐样本完全相同（Δ≡0）⇒ 残差在饱和区无差异（结论要改）"
    elif all1:
        V1 = "★残差只是加速收敛、饱和后消失 ⇒ 结论要改"
    else:
        V1 = "★未过写死判据（既非 2SE 显著正，也非 |Δ|≤1SE）⇒ 如实报"
else:
    V1 = "★run 不全 ⇒ 不判"
print(f"[S36-VERDICT1] {V1}", flush=True)

print("\n[S36] ===== ★② 卡内结构判决：每臂 vs A 的配对 Δ @6000（判据 2SE / 1SE）=====",
      flush=True)
D2 = {}
for a in ("B", "C", "D", "E"):
    D2[a] = {}
    for s in SEEDS36:
        if (a, s) not in RESULTS or ("A", s) not in RESULTS:
            continue
        d = paired_stats(RESULTS[(a, s)]["strict"], RESULTS[("A", s)]["strict"])
        D2[a][s] = d
        print(f"[S36-D2] {a}−A seed={s}: Δ={d['d']*100:+.2f}±{d['se']*100:.2f}pp "
              f"Δ/SE={d['ratio']:.2f} | R29 Δ非零比={d['nz']*100:.1f}% | A="
              f"{RESULTS[('A', s)]['em']*100:.2f}% → {a}={RESULTS[(a, s)]['em']*100:.2f}% | "
              f"{delta_tag(d)}", flush=True)
_catch = [a for a in ("B", "C", "D", "E")
          if len(D2[a]) == 2 and all(is_catch(d) for d in D2[a].values())]
_allneg = [a for a in ("B", "C", "D", "E")
           if len(D2[a]) == 2 and all(is_neg2(d) for d in D2[a].values())]
if _catch:
    V2 = f"★{_catch} 在饱和区追平 A（|Δ|≤1SE）⇒ 那条「不可替代」是欠训假象（如实报）"
elif len(_allneg) == 4:
    V2 = "★B/C/D/E 全部 ≤−2SE 且 2 seed 同号 ⇒「不可替代」在饱和区仍成立"
else:
    V2 = f"★部分成立：≤−2SE 的臂={_allneg}；未过门槛/追平的臂={[a for a in ('B','C','D','E') if a not in _allneg]}"
print(f"[S36-VERDICT2] {V2}", flush=True)

print("\n[S36] ===== ★饱和检查：每臂 @3000/6000 的 EM + 末段仍升？ =====", flush=True)
for a in ARMS36:
    r1, r2 = RESULTS.get((a, 1234)), RESULTS.get((a, 5678))
    if not r1 or not r2:
        continue
    d1 = (r1["em"] - r1["em3000"]) * 100
    d2 = (r2["em"] - r2["em3000"]) * 100
    note = "仍在大幅上升" if max(d1, d2) > 10 else ("仍在升" if max(d1, d2) > 1 else "已平")
    print(f"[S36-SAT] {a}: EM 3000→6000 = {r1['em3000']*100:.2f}→{r1['em']*100:.2f}% "
          f"(Δ{d1:+.2f}pp) / {r2['em3000']*100:.2f}→{r2['em']*100:.2f}% (Δ{d2:+.2f}pp) ⇒ {note}",
          flush=True)
print("[S36-SAT] 口径5：6000 为固定末点（未用 dev 选步）；若某臂 3000→6000 仍大幅上升 ⇒ "
      "该臂在 6000 步仍未饱和（明确写「可能仍欠训」）", flush=True)

print("\n[S36] ===== 汇总一句 =====", flush=True)
_res_ok = V1.startswith("★残差仍必要")
_res_gone = V1.startswith("★残差只是") or V1.startswith("★两臂 EM")
_struct_ok = V2.startswith("★B/C/D/E")
if _res_ok and _struct_ok:
    SUMMARY = "两条结论在饱和区成立"
elif _res_gone and _catch:
    SUMMARY = "两条都只是欠训现象"
else:
    SUMMARY = "部分成立（逐条见上，如实报）"
print(f"[S36-SUMMARY] ① {V1.replace('★', '')} ；② {V2.replace('★', '')} ⇒ 综合 = {SUMMARY}",
      flush=True)

print("\n[S36] ===== 成本与 ckpt =====", flush=True)
for a in ARMS36:
    for s in SEEDS36:
        rr = RESULTS.get((a, s))
        if rr is None:
            continue
        print(f"[META] {a} seed={s}: steps={STEPS36} 训练墙钟={rr['wall_train']/60:.2f}min "
              f"本run={rr['wall_total']/60:.2f}min ckpt={os.path.basename(rr['ckpt'])} "
              f"sha256[:16]={rr['sha']}", flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"D={D} FF={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} | EM口径=batch{EM_BATCH} "
      f"自检=batch{CHK_BATCH} | TEST sha16={TEST_SHA} 泄漏={LEAK}/800 | 地板(单/top5)="
      f"{FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% | 总墙钟={(time.time()-T0)/60:.1f}min | "
      f"run 数={len(RESULTS)}/{len(SEEDS36)*len(ARMS36)}", flush=True)
print("[DONE] exit=0", flush=True)
