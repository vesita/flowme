#!/usr/bin/env python3
"""S22 · 卡片设计基型扫描（第一批）：思维卡【内部】用什么跨位置混合机制最合适？★GPU。

唯一变量 = 思维卡内部那一层结构；其余与 S19 的 res 臂逐字相同（数据生成器/超参/评估口径全部复用）。

现状思维卡（stages/19_residual_scan.py:372-407，逐层）：
  Cards.thought = nn.TransformerEncoderLayer(d=128, nhead=4, ff=512, dropout=0.1,
                                             activation="gelu", batch_first=True, norm_first=True)
  外层：h = h + thought(h, src_mask=m)   ← S19 的 res 臂（残差是必需项，S19 已钉死）
  即：自注意力(4 头，因果 + PAD 掩码) + 逐位置 FFN(512/gelu) + 2×LayerNorm(pre-norm) + 内部残差
  ⇒ 现状【已有 FFN】⇒ 原表 E 臂「注意力+FFN」与 A 重复，改用其互补消融：E = 仅注意力（去 FFN）。

五臂（都保留外层残差；只换 Card 内部）：
  A 现状 (S19-res 逐字复现)      B 纯逐位置 MLP（无跨位置混合）    C 因果卷积 k=3 depthwise
  D 门控 GRU 风格（逐位置）        E 仅注意力（去 FFN，互补消融）

复用：exec 19_residual_scan.py 到 "# ③ 主循环" 之前（⇒ tokenizer / 数据生成器 / 模型 /
build_batch / greedy_gen / hits_from / classify / r28_selfcheck / mean_se 全部现成，
且 add_3d 的 train/test 与 S14/S19 逐字同源）。S19 训练主循环在切片之外。

指标（五维报告卡）：能力(EM batch=1 + 数字每步正确率) / 因果(思维卡换 Identity 的配对 ΔEM) /
集中度(前 20% 单元的因果占比，S21 口径：|Δ数字每步正确率| 降序) / 成本(总参数 + 训练墙钟) /
自检(R28 batch=1 vs 16；R29 非零比例 + 互换两行判无效轴)。

只允许写：本文件、logs/、/tmp。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import time

ROOT = "/home/vesita/coding/my/flowme"
SRC19 = f"{ROOT}/stages/19_residual_scan.py"

# ============================================================================
# ① 逐字复用 S19 的前半段（⇒ 同 tokenizer、同 add_3d train4000/test800、同 enc_layer 口径）
# ============================================================================
_s19 = open(SRC19, encoding="utf-8").read()
_c19 = _s19.index("# ③ 主循环")
NS = {"__name__": "s19_head", "__file__": SRC19}
exec(compile(_s19[:_c19], SRC19, "exec"), NS)          # noqa: S102
for _k, _v in NS.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
del NS

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

T0 = time.time()
SMOKE22 = os.environ.get("S22_SMOKE") == "1"
RUN_STEPS = 20 if SMOKE22 else STEPS                    # S19 的 STEPS = 3000
N_EVAL = 32 if SMOKE22 else N_TEST                      # 800
N_CC = 32 if SMOKE22 else 300                           # 集中度冒烟评估子集
SEEDS22 = (1234, 5678)
ARMS = ("A", "B", "C", "D", "E")
BUCKET = "add_3d"
CKPT_TMPL = f"{ROOT}/logs/22_ckpt_{BUCKET}_{{arm}}_seed{{seed}}.pt"
JSONL = f"{ROOT}/logs/22_results.jsonl"
FP = 0.1

ARM_DESC = {
    "A": "现状=S19-res：h+TransformerEncoderLayer(MHA4+FFN512/gelu+2×LN pre-norm, dropout0.1)",
    "B": "纯逐位置 MLP：h+MLP(LN(h))，Linear(128,512)-GELU-Dropout-Linear(512,128)-Dropout（零跨位置混合）",
    "C": "因果卷积 k=3：h+Linear(GELU(DepthwiseConv1d(128,k=3,groups=128) 左填充))，局部跨位置混合",
    "D": "门控 GRU 风格：g=sigmoid(Wg·LN(h))，c=tanh(MLP(LN(h)))；h'=g⊙c+(1−g)⊙h（逐位置，自带残差）",
    "E": "仅注意力（去 FFN 的互补消融）：h+Dropout(MHA(LN(h)))，无逐位置 FFN",
}
assert "TransformerEncoderLayer" in _s19[: _c19] and "self.thought" in _s19[: _c19]

# ============================================================================
# ② 五臂的 Card 实现（签名统一 card(h, m) -> h_out；所有臂都保留残差通道）
# ============================================================================
class CardA(nn.Module):
    """现状：完整 TransformerEncoderLayer（自注意力 + FFN + 2×LN），外层加残差。"""

    def __init__(self, d, ff):
        super().__init__()
        self.layer = enc_layer(d, ff)                   # 与 S19 thought 逐字同构造

    def forward(self, h, m):
        return h + self.layer(h, src_mask=m)            # ★S19 res 臂那一行


class CardB(nn.Module):
    """纯逐位置 MLP：完全没有跨位置混合。"""

    def __init__(self, d, ff):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(FP),
                                 nn.Linear(ff, d), nn.Dropout(FP))

    def forward(self, h, m):
        return h + self.mlp(self.norm(h))


class CardC(nn.Module):
    """因果卷积：depthwise k=3（左填充 ⇒ 只看自己与左侧）+ pointwise 混合。"""

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
        z = self.dw(z)[..., : h.size(1)].transpose(1, 2)   # 砍掉右侧 ⇒ 严格因果
        return h + self.do(self.pw(self.act(z)))


class CardD(nn.Module):
    """门控（GRU 风格）：逐位置，自带 (1−g) 残差通道。"""

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
    """仅注意力：去掉逐位置 FFN（现状的互补消融）。"""

    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, NHEAD, dropout=FP, batch_first=True)
        self.do = nn.Dropout(FP)

    def forward(self, h, m):
        z = self.norm(h)
        o = self.att(z, z, z, attn_mask=m, need_weights=False)[0]
        return h + self.do(o)


def build_card(arm: str, d: int, ff: int) -> nn.Module:
    if arm == "A":
        return CardA(d, ff)
    if arm == "B":
        return CardB(d, ff)
    if arm == "C":
        return CardC(d)
    if arm == "D":
        return CardD(d, ff)
    if arm == "E":
        return CardE(d)
    raise ValueError(arm)


HOOK_IN: dict = {"fn": None}       # 思维卡【输入】侧干预（= S21 的 in 口径）
BYPASS: dict = {"on": False}       # 思维卡换 Identity（因果中介门 B）


class Cards22(nn.Module):
    """与 S19 Cards 逐字同构（emb → in_enc → card → head），card 可换、输入可干预、整卡可旁路。"""

    def __init__(self, arm: str, d: int, ff: int):
        super().__init__()
        self.d = d
        self.arm = arm
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        self.card = build_card(arm, d, ff)
        self.head = nn.Linear(d, V)
        self.register_buffer("pe", sin_pe(MAXLEN, d), persistent=False)

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
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
        if not BYPASS["on"]:
            if HOOK_IN["fn"] is not None:
                h = HOOK_IN["fn"](h)
            h = self.card(h, m)
        return self.head(h)


def n_params_arm(arm: str) -> int:
    mdl = Cards22(arm, D, FF)
    n = sum(p.numel() for p in mdl.parameters())
    del mdl
    return n


# ============================================================================
# ③ 训练（与 S19 train_run 逐字相同，只换成 Cards22）
# ============================================================================
def train22(arm: str, seed: int, train, steps: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards22(arm, D, FF).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0 = 0, 0, time.time()
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
        if step % 500 == 0 or step == steps:
            print(f"  [train:{arm}] step={step}/{steps} loss={loss.item():.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    return model, time.time() - t0


# ============================================================================
# ④ 评估口径（逐字沿用 S19；新增「带干预的逐样本 CE」与集中度）
# ============================================================================
def nmean(xs):
    ys = [float(x) for x in xs if x == x]
    return sum(ys) / len(ys) if ys else float("nan")


@torch.no_grad()
def dig_ce_samples(model, recs, hook=None):
    """逐样本 (整体 CE, 数字/算子 token 的 CE)；口径 = S19 report_ce 的 samp_all / dig_samp。"""
    model.eval()
    HOOK_IN["fn"] = hook
    all_ce, dce = [], []
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
                if ds:
                    dce.append(sum(ds) / len(ds))
    finally:
        HOOK_IN["fn"] = None
    model.train()
    return all_ce, dce


def acc_se(ce_pers):
    m, se = mean_se(ce_pers)
    acc = math.exp(-m)
    return acc, acc * se, m, se


def dacc_pp(base_ce_pers, iv_ce_pers):
    return (math.exp(-nmean(iv_ce_pers)) - math.exp(-nmean(base_ce_pers))) * 100.0


def iv_blk(lo: int, hi: int):
    def f(x, *rest):
        x = x.clone()
        x[:, :, lo:hi] = 0.0
        return x
    return f


def iv_all():
    def f(x, *rest):
        return torch.zeros_like(x)
    return f


def iv_swp(i: int):
    """互换两行（@in 非字面版）：R29 已知无效轴，必须实测剔除。"""
    def f(x, *rest):
        if i + 1 >= x.size(1):
            return x
        x = x.clone()
        a = x[:, i, :].clone()
        x[:, i, :] = x[:, i + 1, :]
        x[:, i + 1, :] = a
        return x
    return f


@torch.no_grad()
def em_of(model, recs, batch: int = EM_BATCH):
    model.eval()
    txt = greedy_gen(model, recs, batch=batch)
    hits = hits_from(txt, recs)
    model.train()
    return hits


def em_stats(hits):
    n = len(hits)
    s = sum(h["strict"] for h in hits) / n
    return s, math.sqrt(s * (1 - s) / n), n


def paired_d(a_hits, b_hits):
    """逐 test 样本配对 Δ = a − b（a/b 必须同序）。"""
    xs = [x["strict"] - y["strict"] for x, y in zip(a_hits, b_hits)]
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return m, se, n


def shares_top(ces, p):
    """S21 口径：按因果效应降序，前 max(1, round(p*N)) 个单元的因果占比。"""
    cs = sorted(ces, reverse=True)
    n = len(cs)
    k = max(1, int(round(p * n)))
    tot = sum(cs)
    return (sum(cs[:k]) / tot if tot > 0 else float("nan")), k


def concentration(model, recs, swap_pos):
    """思维卡输入 h 的分块单元单点清零 ⇒ |Δ数字每步正确率| = 因果效应；S21 的曲线口径。"""
    _, base_dig = dig_ce_samples(model, recs)
    conc = {}
    for ud in (16, 8):
        units = [(b * ud, (b + 1) * ud) for b in range(D // ud)]
        if SMOKE22:
            units = units[:4]
        ces = []
        for lo, hi in units:
            _, iv = dig_ce_samples(model, recs, iv_blk(lo, hi))
            ces.append(abs(dacc_pp(base_dig, iv)))
        s20, k20 = shares_top(ces, 0.2)
        conc[ud] = dict(unit_dim=ud, n_units=len(units), shares20=s20, k20=k20,
                        shares10=shares_top(ces, 0.1)[0], shares50=shares_top(ces, 0.5)[0],
                        ces=ces)
    _, c_all = dig_ce_samples(model, recs, iv_all())
    conc["pos_ctrl"] = dacc_pp(base_dig, c_all)
    swp = []
    for i in swap_pos:
        _, iv = dig_ce_samples(model, recs, iv_swp(i))
        swp.append(abs(dacc_pp(base_dig, iv)))
    nz = [c > 1e-3 for c in conc[8]["ces"]]
    srt = sorted(conc[8]["ces"])
    med = srt[len(srt) // 2] if srt else 0.0
    swp_nz = sum(c > 1e-3 for c in swp) / len(swp) if swp else 0.0
    srt_s = sorted(swp)
    swp_med = srt_s[len(srt_s) // 2] if srt_s else 0.0
    r29 = dict(pos_nz=sum(nz) / len(nz), pos_med=med, pos_max=max(conc[8]["ces"]),
               swp_nz=swp_nz, swp_med=swp_med, swap_pos=swap_pos,
               swp_valid=(swp_nz >= 0.20 and swp_med >= 0.01), n_cc=len(recs))
    return conc, r29


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def append_jsonl(obj: dict) -> None:
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================================
# ⑤ 主循环：2 seed × 5 臂（seed 外层 ⇒ 同 seed 内 A 与各臂紧挨着，配对最干净）
# ============================================================================
PARAMS = {a: n_params_arm(a) for a in ARMS}
train_all, test_all = DATA[BUCKET]["train"], DATA[BUCKET]["test"]
test = test_all[:N_EVAL]
cc_recs = test_all[:min(N_CC, N_EVAL)]
SWAP_POS = [i for i in (0, 3, 7, 11) if i + 1 < 15]

print(f"\n[S22] ===== ★卡片设计基型扫描 | device={device} | bucket={BUCKET} "
      f"train={len(train_all)} test={len(test_all)}(eval {len(test)}) | steps={RUN_STEPS} | "
      f"seeds={SEEDS22} | arms={ARMS} | smoke={SMOKE22} =====", flush=True)
for a in ARMS:
    print(f"[S22] 臂 {a}: 总参数={PARAMS[a]/1e6:.3f}M (ΔvsA={PARAMS[a]-PARAMS['A']:+d}) | "
          f"{ARM_DESC[a]}", flush=True)

RESULTS: dict = {}
open(JSONL, "w").close()

for seed in SEEDS22:
    for arm in ARMS:
        t0 = time.time()
        print(f"\n[S22] ===== 臂={arm} seed={seed} steps={RUN_STEPS} =====", flush=True)
        model, wall = train22(arm, seed, train_all, RUN_STEPS)

        # ---- 1) 能力：EM（主口径 batch=1）+ 数字每步正确率 ----
        hits = em_of(model, test, batch=EM_BATCH)
        em, em_se, n_eval = em_stats(hits)
        all_pers, b_dig_pers = dig_ce_samples(model, test)
        acc, acc_se_, dig_m, _ = acc_se(b_dig_pers)
        print(f"[S22] {arm} seed={seed} 能力：EM(batch1,n={n_eval})={em*100:.2f}%±{em_se*100:.2f} | "
              f"数字每步正确率={acc*100:.2f}%±{acc_se_*100:.2f}pp (数字CE={dig_m:.4f}) | "
              f"整体CE={nmean(all_pers):.4f}", flush=True)

        # ---- 2) 因果：思维卡换 Identity（门 B 恒等中介）----
        BYPASS["on"] = True
        try:
            hits_id = em_of(model, test, batch=EM_BATCH)
            _, id_dig_pers = dig_ce_samples(model, test)
        finally:
            BYPASS["on"] = False
        em_id, _, _ = em_stats(hits_id)
        d_med, d_med_se, _ = paired_d(hits, hits_id)        # Δ = 有卡 − 无卡
        d_med_dig = dacc_pp(b_dig_pers, id_dig_pers)
        print(f"[S22] {arm} seed={seed} 因果(卡→Identity)：EM {em*100:.2f}% → {em_id*100:.2f}% | "
              f"ΔEM={d_med*100:+.2f}±{d_med_se*100:.2f}pp (Δ/SE="
              f"{(d_med/d_med_se if d_med_se > 0 else float('inf')):.1f}) | "
              f"Δ数字={d_med_dig:+.2f}pp", flush=True)

        # ---- 3) 集中度 + R29 ----
        conc, r29 = concentration(model, cc_recs, SWAP_POS)
        print(f"[S22] {arm} seed={seed} 集中度(n={r29['n_cc']})：16维块(8单元) 前20%="
              f"{conc[16]['shares20']*100:.1f}%(k={conc[16]['k20']}) | 8维块(16单元) 前20%="
              f"{conc[8]['shares20']*100:.1f}%(k={conc[8]['k20']}) | 前50%(8维)="
              f"{conc[8]['shares50']*100:.1f}% | 整体清零Δ数字={conc['pos_ctrl']:+.2f}pp | "
              f"R29 单点非零比={r29['pos_nz']*100:.1f}% 中位|Δ|={r29['pos_med']:.3f}pp 最大="
              f"{r29['pos_max']:.3f}pp | 互换两行非零比={r29['swp_nz']*100:.1f}% 中位="
              f"{r29['swp_med']:.4f}pp ⇒ "
              f"{'★无效轴（作废剔除，不当因果效应=0 用）' if not r29['swp_valid'] else '有效轴'}",
              flush=True)

        # ---- 5) 自检 R28 ----
        r28 = r28_selfcheck(model, test, tag=f"[{arm} seed={seed}] ", k=min(CHK_BATCH, len(test)))

        ck = CKPT_TMPL.format(arm=arm, seed=seed)
        if not SMOKE22:
            torch.save(model.state_dict(), ck)
            dig = sha16(ck)
            print(f"[CKPT] {ck} sha256[:16]={dig}", flush=True)
        else:
            dig = "-"
        rec = dict(arm=arm, seed=seed, bucket=BUCKET, steps=RUN_STEPS, params=PARAMS[arm],
                   wall_train=wall, wall_total=time.time() - t0,
                   em=em, em_se=em_se, em_n=n_eval, acc=acc, acc_se=acc_se_,
                   dig_ce=dig_m, ce=nmean(all_pers),
                   em_id=em_id, d_med=d_med, d_med_se=d_med_se, d_med_dig=d_med_dig,
                   conc16=conc[16], conc8=conc[8], pos_ctrl=conc["pos_ctrl"], r29=r29,
                   r28_same=r28["same"], r28_k=r28["k"], ckpt=ck, sha=dig,
                   strict=[h["strict"] for h in hits],
                   strict_id=[h["strict"] for h in hits_id])
        RESULTS[(arm, seed)] = rec
        append_jsonl({k: v for k, v in rec.items() if not k.startswith("strict")})
        del model
        torch.cuda.empty_cache()
        print(f"[S22-MAIN] {arm} seed={seed} | EM={em*100:.2f}%±{em_se*100:.2f} | "
              f"数字={acc*100:.2f}% | Δ中介EM={d_med*100:+.2f}pp | "
              f"前20%(8维16单元)={conc[8]['shares20']*100:.1f}% | 参数={PARAMS[arm]/1e6:.3f}M | "
              f"训练={wall/60:.2f}min 本run={rec['wall_total']/60:.2f}min | "
              f"R28={r28['same']}/{r28['k']} | sha16={dig}", flush=True)

# ============================================================================
# ⑥ 汇总 + ★判定
# ============================================================================
print("\n[S22] ===== ★五维报告卡（每臂一行） =====", flush=True)
S: dict = {}
for a in ARMS:
    rs = [RESULTS.get((a, s)) for s in SEEDS22]
    if any(r is None for r in rs):
        print(f"[S22] {a} | ★未跑全", flush=True)
        continue
    ems = [r["em"] for r in rs]
    S[a] = dict(ems=ems, mean=sum(ems) / 2, params=rs[0]["params"],
                acc=[r["acc"] for r in rs],
                dmed=[r["d_med"] for r in rs], dmed_se=[r["d_med_se"] for r in rs],
                c16=sum(r["conc16"]["shares20"] for r in rs) / 2,
                c8=sum(r["conc8"]["shares20"] for r in rs) / 2,
                wall=sum(r["wall_train"] for r in rs) / 2)
    print(f"[S22] {a} | {S[a]['params']/1e6:.3f}M | {ems[0]*100:.2f}%/{ems[1]*100:.2f}%/"
          f"{S[a]['mean']*100:.2f}% | {S[a]['acc'][0]*100:.2f}%/{S[a]['acc'][1]*100:.2f}% | "
          f"{S[a]['dmed'][0]*100:+.2f}±{S[a]['dmed_se'][0]*100:.2f}/"
          f"{S[a]['dmed'][1]*100:+.2f}±{S[a]['dmed_se'][1]*100:.2f}pp | "
          f"{S[a]['c16']*100:.1f}%/{S[a]['c8']*100:.1f}% | {S[a]['wall']/60:.2f}min | "
          f"{rs[0]['r28_same']}/{rs[0]['r28_k']},{rs[1]['r28_same']}/{rs[1]['r28_k']} | "
          f"{ARM_DESC[a]}", flush=True)

print("\n[S22] ===== ★配对 Δ = EM(臂) − EM(A)，逐 test 样本配对，门槛 2×SE =====", flush=True)
DELTA: dict = {}
for a in ARMS:
    DELTA[a] = []
    for sd in SEEDS22:
        ra, rb = RESULTS.get((a, sd)), RESULTS.get(("A", sd))
        if ra is None or rb is None:
            continue
        xs = [x - y for x, y in zip(ra["strict"], rb["strict"])]
        n = len(xs)
        m = sum(xs) / n
        var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
        se = math.sqrt(var / n)
        DELTA[a].append(dict(seed=sd, d=m, se=se, ratio=(m / se if se > 0 else float("inf")),
                             sig=(m > 0 and m >= 2 * se), nsig=(m < 0 and m <= -2 * se)))
        print(f"[S22] Δ({a}−A) seed={sd}: {m*100:+.2f}±{se*100:.2f}pp Δ/SE="
              f"{DELTA[a][-1]['ratio']:.2f} ⇒ "
              f"{'★显著更高' if DELTA[a][-1]['sig'] else ('★显著更低' if DELTA[a][-1]['nsig'] else '不显著')}"
              f" | A={rb['em']*100:.2f}% → {a}={ra['em']*100:.2f}%", flush=True)

print("\n[S22] ===== ★基线复现核对（A 必须 = S19-res add_3d 37.25% / 31.13%） =====", flush=True)
ANCHOR_REF = {1234: 0.3725, 5678: 0.3113}
ANCHOR_OK = True
for sd in SEEDS22:
    r = RESULTS.get(("A", sd))
    if r is None:
        ANCHOR_OK = False
        print(f"[S22] 锚 A seed={sd}: ★无 run", flush=True)
        continue
    dev = (r["em"] - ANCHOR_REF[sd]) * 100
    ok = abs(dev) <= 0.5
    ANCHOR_OK &= ok
    print(f"[S22] 锚 A seed={sd}: S19-res={ANCHOR_REF[sd]*100:.2f}% vs A={r['em']*100:.2f}% "
          f"Δ={dev:+.2f}pp | 数字 {r['acc']*100:.2f}%（S19 52.59/49.63） | "
          f"{'一致(|Δ|≤0.5pp ⇒ 逐位复现)' if ok else '★偏离 >0.5pp（点名）'}", flush=True)

print("\n[VERDICT-S22] ===== ★判定（判据写死） =====", flush=True)
if len(RESULTS) < 2 * len(ARMS):
    print("[VERDICT-S22] ★未跑满 5 臂 × 2 seed ⇒ 无法判决"
          "（不许空报告：看日志尾部与退出码）", flush=True)
else:
    best = max(ARMS, key=lambda a: S[a]["mean"])
    print(f"[VERDICT-S22] 均值最高臂 = {best}（{S[best]['mean']*100:.2f}% vs A "
          f"{S['A']['mean']*100:.2f}%）", flush=True)
    for a in ARMS:
        if a == "A":
            continue
        ds = DELTA[a]
        if len(ds) < 2:
            print(f"[VERDICT-S22] {a}: run 不全 ⇒ 不判", flush=True)
            continue
        dtxt = (f"{ds[0]['d']*100:+.2f}±{ds[0]['se']*100:.2f} / "
                f"{ds[1]['d']*100:+.2f}±{ds[1]['se']*100:.2f}pp")
        if all(d["d"] > 0 for d in ds) and all(d["sig"] for d in ds):
            print(f"[VERDICT-S22] {a}: ★比现状好（2 seed 同号为正且各自 Δ≥2SE）：{dtxt}", flush=True)
        elif all(d["d"] < 0 for d in ds) and all(d["nsig"] for d in ds):
            print(f"[VERDICT-S22] {a}: ★显著更低 ⇒ 现状的跨位置混合不可替代（{a} 少了它）：{dtxt}",
                  flush=True)
        else:
            print(f"[VERDICT-S22] {a}: 不显著（Δ={dtxt}）⇒ 结构选择对它无显著影响", flush=True)
    base_c = S["A"]["c8"]
    for a in ARMS:
        if a == "A":
            continue
        dd = [RESULTS[(a, s)]["conc8"]["shares20"] - RESULTS[("A", s)]["conc8"]["shares20"]
              for s in SEEDS22]
        em_d = [d["d"] * 100 for d in DELTA[a]] if DELTA[a] else [0.0, 0.0]
        em_se = [d["se"] * 100 for d in DELTA[a]] if DELTA[a] else [1.0, 1.0]
        same_dir = len({x > 0 for x in dd}) == 1
        bigger = all(abs(x) > 0.05 for x in dd)
        em_not_lower = all(x >= -2 * e for x, e in zip(em_d, em_se))
        ok = same_dir and bigger and all(x < 0 for x in dd) and em_not_lower
        print(f"[VERDICT-S22] {a}: 集中度(前20%, 8维16单元) {base_c*100:.1f}% → "
              f"{S[a]['c8']*100:.1f}%（Δ={sum(dd)/2*100:+.1f}pp，2 seed "
              f"{'同向' if same_dir else '不同向'}）⇒ "
              f"{'★更多单元参与 ✓' if ok else '不满足「显著降 + EM 不降」'}", flush=True)

print("\n[S22] ===== 成本与 ckpt =====", flush=True)
for a in ARMS:
    for sd in SEEDS22:
        r = RESULTS.get((a, sd))
        if r is None:
            continue
        print(f"[META] {a} seed={sd}: 总参数={r['params']} 训练墙钟={r['wall_train']/60:.2f}min "
              f"本run={r['wall_total']/60:.2f}min ckpt={os.path.basename(r['ckpt'])} sha16={r['sha']}",
              flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | D={D} FF={FF} NHEAD={NHEAD} "
      f"lr={LR} batch={BATCH} steps={RUN_STEPS} MAXLEN={MAXLEN} | 生成主口径=batch{EM_BATCH} "
      f"自检=batch{CHK_BATCH} | 集中度评估子集 n={len(cc_recs)} 单元=8×16维+16×8维 | "
      f"smoke={SMOKE22} | 总墙钟={(time.time()-T0)/60:.1f}min", flush=True)
print(f"[META] 基线复现核对 = {'通过（A 逐位复现 S19-res）' if ANCHOR_OK else '★偏离（见上）'}",
      flush=True)
print("[DONE] exit=0", flush=True)
