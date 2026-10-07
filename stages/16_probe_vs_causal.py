#!/usr/bin/env python3
"""S16 · P-3 判决：模因「读出」≠「会用」—— 线性探针准度 vs 干预因果效应的相关系数。

★零训练：只读两个 S14 ckpt（add_1d EM96.6% / add_3d EM2.5%），不训练任何模型（探针除外，它就是被测量）。
★全程 CPU：文件一开头就 `CUDA_VISIBLE_DEVICES=""`，device 恒为 cpu（不碰 S15 的 GPU 任务）。

复用方式（不重写模型）：把 `stages/14_synth_arith.py` 的**前半段源码**（到注释 `# ③ 主循环` 之前 =
tokenizer + 数据生成器 + Cards 模型 + build_batch/eval_ce/greedy_gen/hits_from/em_report 等评估函数，
**不含任何训练循环**）exec 进本命名空间，逐字复用。
  · 14 号脚本 device!=cuda 时会 `sys.exit(3)` ⇒ exec 期间把 sys.exit 换成 no-op（只在内存里）；
  · 那行「立即停」文案在内存里替换成 S16 说明；**源文件一个字节都不改**。

只回答一个问题：模因张量里能读出多少答案信息 vs 模因对答案的因果贡献，是不是两回事。
  ① 探针（读出）：冻结模型，模因张量 [n,d]（prompt 段，防答案泄漏）→ 线性/小 MLP
     预测答案**位数**（分类：train/test 准度）与答案**值**（回归：train/test R²）；
  ② 因果（会用）：推理期 patching（不改权重）：清零 / 加噪σ=0.1,1.0 / 乱序 / 换样本模因
     ⇒ ΔEM（batch=1，R28 主口径）与 Δ数字每步正确率（=exp(−数字token CE)），配对 ±SE；
     顺带恒等中介（思维卡=Identity，h←in_enc 输出）⇒ ΔEM，与 S12 A臂 +0.0119 对照；
  ③ ★判决：Spearman ρ(探针test准度, |Δ|)，10 点（2 ckpt × 5 干预）；
     判据写死：ρ<0.5 且 n≥6 ⇒ 弱相关 ⇒ 确证 P-3；ρ≥0.8 ⇒ 强相关 ⇒ 推翻 P-3；中间 ⇒ 不确定。

只允许写：本文件、logs/、/tmp。
"""
from __future__ import annotations

import math
import os
import sys
import time
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # ★先于 torch 导入：强制关 GPU
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import Adam  # noqa: E402

torch.set_num_threads(int(os.environ.get("S16_THREADS", "8")))
DEV = "cpu"
T0 = time.time()
print(f"device: {DEV} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
      f"| torch.cuda.is_available()={torch.cuda.is_available()} "
      f"| threads={torch.get_num_threads()}", flush=True)
assert DEV == "cpu"

SMOKE = os.environ.get("S16_SMOKE") == "1"
N_EVAL = 16 if SMOKE else int(os.environ.get("S16_N", "400"))
ROOT = "/home/vesita/coding/my/flowme"
SRC = f"{ROOT}/stages/14_synth_arith.py"

# ============================================================================
# ① 复用 stage 14 的定义（数据生成器 + Cards + 评估函数），切在训练主循环之前
# ============================================================================
_src = open(SRC, encoding="utf-8").read()
_cut = _src.index("# ③ 主循环")
_head = _src[:_cut].replace("GPU 起不来，立即停（不用 CPU 硬跑）。",
                            "S16: GPU 不可用 ⇒ 本单元按任务要求强制 CPU（零训练，只复用定义）")
_nlines = len(_head.splitlines())
_real_exit = sys.exit
sys.exit = lambda *a, **k: None          # 14 号脚本的 device 门禁：本单元是故意用 CPU
NS = {"__name__": "s14_reuse", "__file__": SRC}
exec(compile(_head, SRC, "exec"), NS)
sys.exit = _real_exit
print(f"[S16] 复用 {os.path.basename(SRC)} 前 {_nlines} 行（生成器+Cards+评估函数，"
      f"训练主循环在第 {_src[:_cut].count(chr(10))} 行之后，未执行）", flush=True)

Cards, D, FF, NHEAD = NS["Cards"], NS["D"], NS["FF"], NS["NHEAD"]
MAXLEN, MAX_GEN = NS["MAXLEN"], NS["MAX_GEN"]
PAD_ID, EOS_ID = NS["PAD_ID"], NS["EOS_ID"]
build_batch, greedy_gen, hits_from = NS["build_batch"], NS["greedy_gen"], NS["hits_from"]
classify, mean_se, enc = NS["classify"], NS["mean_se"], NS["enc"]
sin_pe, DATA, tok = NS["sin_pe"], NS["DATA"], NS["tok"]
sha16 = NS["sha16"] if "sha16" in NS else None

CKPT = {"add_1d": f"{ROOT}/logs/14_ckpt_add_1d_seed1234.pt",
        "add_3d": f"{ROOT}/logs/14_ckpt_add_3d_seed1234.pt"}
# S14 锚点（比对用）：add_1d EM 96.6% / add_3d EM 2.5%
ANCHOR = {"add_1d": 0.966, "add_3d": 0.025}

# ============================================================================
# ② 干预钩子：替换 Cards.logits 里「思维卡输出」这一份模因张量（不改权重）
# ============================================================================
HOOK = {"fn": None, "cur": 0, "donor": None, "n_call": 0}


def _hooked_logits(self, ids: torch.Tensor) -> torch.Tensor:
    """与 Cards.logits 逐行相同，只多一步：模因 h 可被 HOOK["fn"] 替换。"""
    n = ids.size(1)
    m = torch.full((n, n), float("-inf"), device=ids.device)
    m = torch.triu(m, diagonal=1)
    m = m.unsqueeze(0).expand(ids.size(0), -1, -1).clone()
    m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
    m = m.repeat_interleave(NHEAD, dim=0)
    pe = self.pe if n <= self.pe.size(0) else sin_pe(n, self.d).to(self.pe.device)
    assert n <= pe.size(0)
    x = self.emb(ids) + pe[:n].unsqueeze(0)
    h = self.in_enc(x, src_mask=m)          # 输入卡输出 = 思维卡的输入
    h_in = h
    if self.thought is not None:
        h = self.thought(h, src_mask=m)     # ★思维卡输出 = 模因张量 [B,n,d]
    HOOK["n_call"] += 1
    if HOOK["fn"] is not None:
        h = HOOK["fn"](h, h_in, ids)
    return self.head(h)


def build_model(bucket: str):
    m = Cards(D, FF)
    sd = torch.load(CKPT[bucket], map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    m.logits = types.MethodType(_hooked_logits, m)
    return m


# ---------------- 干预定义 ----------------
def iv_zero(h, h_in, ids):
    return torch.zeros_like(h)


def iv_noise(sig: float):
    def f(h, h_in, ids):
        return h + sig * torch.randn_like(h)
    return f


def iv_shuffle(h, h_in, ids):
    B, n, _ = h.shape
    return torch.stack([h[j][torch.randperm(n)] for j in range(B)])


def iv_swap(h, h_in, ids):
    """把每个样本的模因换成**另一个**样本（eval[i+1]）的模因（来自其整条 prompt+target 的前向）。
    +1 必不可少：否则 donor[i] 就是自己 ⇒ 换了个寂寞（S16 首测踩过，Δ 恒为 0）。"""
    B, n, _ = h.shape
    rows = []
    for j in range(B):
        dm = HOOK["donor"][(HOOK["cur"] + 1 + j) % len(HOOK["donor"])]   # [Ld, d]
        if dm.size(0) >= n:
            rows.append(dm[:n])
        else:
            rows.append(torch.cat([dm, dm[-1:].expand(n - dm.size(0), -1)], 0))
    return torch.stack(rows, 0)


def iv_identity(h, h_in, ids):
    """恒等中介：思维卡 = Identity（h ← 输入卡输出），即 S12 GATE B 的口径。"""
    return h_in


IVS = [("清零", iv_zero), ("加噪0.1", iv_noise(0.1)), ("加噪1.0", iv_noise(1.0)),
       ("乱序", iv_shuffle), ("换样本模因", iv_swap)]
IVS_SEMI = IVS + [("恒等中介", iv_identity)]

CAP: dict = {}


def cap_fn(h, h_in, ids):
    CAP["h"] = h
    return h


# ============================================================================
# ③ 模因特征（prompt 段，防答案泄漏）与探针
# ============================================================================
@torch.no_grad()
def meme_rows(model, recs, span: str = "prompt", bs: int = 64):
    """返回每条样本的模因行张量 [n,d]。span=prompt 只取 prompt 段（答案还没生成，无泄漏）；
    span=full 取 prompt+target 全段（含答案 token，用于泄漏对照）。"""
    model.eval()
    HOOK["fn"], HOOK["donor"] = cap_fn, None
    out = []
    for i in range(0, len(recs), bs):
        ch = recs[i:i + bs]
        if span == "prompt":
            lens = [len(r["p"]) for r in ch]
            n = max(lens)
            ids = torch.full((len(ch), n), PAD_ID, dtype=torch.long)
            for j, r in enumerate(ch):
                ids[j, : len(r["p"])] = torch.tensor(r["p"])
            model.logits(ids)
            h = CAP["h"]
            out += [h[j, : lens[j]].clone() for j in range(len(ch))]
        else:
            ids, s = build_batch(ch)
            model.logits(ids)
            h = CAP["h"]
            out += [h[j, : int(s[j]) + len(r["t"])].clone() for j, r in enumerate(ch)]
    HOOK["fn"] = None
    return out


def featurize(rows):
    return torch.stack([torch.cat([r[-1], r.mean(0)]) for r in rows])   # [N, 2d]


def _probe(Xtr, Ytr, Xte, Yte, kind: str, mlp: bool, seed: int = 0):
    """kind='cls' 准度 / 'reg' R²。标准化只用 train 统计量；Adam 若干 epoch。"""
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    if kind == "cls":
        classes = sorted(set(Ytr.tolist()))
        c2i = {c: i for i, c in enumerate(classes)}
        Ytr_i = torch.tensor([c2i[v] for v in Ytr.tolist()])
        Yte_i = torch.tensor([c2i.get(v, -1) for v in Yte.tolist()])
        head_out = len(classes)
    else:
        ym, ys = Ytr.mean(), Ytr.std() + 1e-6
        Ytr_i = ((Ytr - ym) / ys).unsqueeze(1)
        Yte_i = ((Yte - ym) / ys).unsqueeze(1)
        head_out = 1
    F = Xtr.shape[1]
    net = (nn.Sequential(nn.Linear(F, 128), nn.ReLU(), nn.Linear(128, head_out)) if mlp
           else nn.Linear(F, head_out))
    opt = Adam(net.parameters(), lr=0.05)
    n = Xtr.shape[0]
    g = torch.Generator().manual_seed(seed)
    for _ in range(400):
        idx = torch.randint(0, n, (512,), generator=g)
        out = net(Xtr[idx])
        loss = (nn.functional.cross_entropy(out, Ytr_i[idx]) if kind == "cls"
                else nn.functional.mse_loss(out, Ytr_i[idx]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        if kind == "cls":
            a_tr = float((net(Xtr).argmax(1) == Ytr_i).float().mean())
            m = Yte_i >= 0
            a_te = float((net(Xte[m]).argmax(1) == Yte_i[m]).float().mean())
            return a_tr, a_te
        ptr, pte = net(Xtr).squeeze(1), net(Xte).squeeze(1)
        r2 = lambda p, y: float(1 - ((p - y) ** 2).sum() / ((y - y.mean()) ** 2).sum())
        return r2(ptr, Ytr_i.squeeze(1)), r2(pte, Yte_i.squeeze(1))


# ============================================================================
# ④ 因果评估：配对 Δ（EM batch=1 主口径 + 数字每步正确率）
# ============================================================================
@torch.no_grad()
def eval_ce(model, recs, hook):
    HOOK["fn"] = hook
    model.eval()
    all_ce, dig_ce = [], []
    for i, r in enumerate(recs):
        HOOK["cur"] = i
        ids, s = build_batch([r])
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        e = int(s[0]) + len(r["t"])
        lp = logp[0, int(s[0]) - 1: e - 1]
        tgt = ids[0, int(s[0]): e]
        ce = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1)
        all_ce.append(float(ce.mean()))
        dk = [k for k in range(ce.numel()) if classify(int(tgt[k])) == 0]
        dig_ce.append(float(ce[dk].mean()) if dk else float("nan"))
    HOOK["fn"] = None
    return all_ce, dig_ce


def eval_em(model, recs, hook):
    HOOK["fn"] = hook
    model.eval()
    texts = []
    for i, r in enumerate(recs):
        HOOK["cur"] = i
        texts.append(greedy_gen(model, [r], batch=1)[0])
    HOOK["fn"] = None
    return hits_from(texts, recs)


def se(xs):
    return mean_se(xs)


def cond_report(tag, base, iv, dig_b, dig_i):
    db = [h["strict"] for h in base]
    di = [h["strict"] for h in iv]
    d_em = [b - a for a, b in zip(db, di)]           # iv - base
    em_m, em_se_ = se(d_em)
    mb, mi = sum(dig_b) / len(dig_b), sum(dig_i) / len(dig_i)
    dce = [b - a for a, b in zip(dig_b, dig_i)]      # iv - base 的配对 CE 差
    dce_m, dce_se = se(dce)
    acc_b, acc_i = math.exp(-mb), math.exp(-mi)
    d_acc = acc_i - acc_b
    d_acc_se = math.exp(-(mb + mi) / 2) * dce_se
    print(f"[IV] {tag} | ΔEM={em_m*100:+.2f}pp±{em_se_*100:.2f} "
          f"(EM {acc_of(base)*100:.2f}%→{acc_of(iv)*100:.2f}%) "
          f"| Δ数字每步={d_acc*100:+.2f}pp±{d_acc_se*100:.2f} "
          f"(acc {acc_b*100:.2f}%→{acc_i*100:.2f}%)", flush=True)
    return dict(dem=em_m, dem_se=em_se_, dacc=d_acc, dacc_se=d_acc_se,
                em_b=acc_of(base), em_i=acc_of(iv), acc_b=acc_b, acc_i=acc_i)


def acc_of(hits):
    return sum(h["strict"] for h in hits) / len(hits)


def em_se_of(hits):
    s = acc_of(hits)
    return math.sqrt(s * (1 - s) / len(hits))


def spearman(x, y):
    from scipy.stats import spearmanr
    r = spearmanr(x, y)
    return float(r.statistic), float(r.pvalue)


# ============================================================================
# ⑤ 主流程：两个 ckpt × (探针 + 6 种干预)
# ============================================================================
PROBE: dict = {}
CAUSAL: dict = {}
CKS = ["add_1d", "add_3d"]

for bucket in CKS:
    t0 = time.time()
    print(f"\n[S16] ===== ckpt={bucket} sha16={sha16(CKPT[bucket])} eval N={N_EVAL} =====", flush=True)
    model = build_model(bucket)
    tr, te = DATA[bucket]["train"], DATA[bucket]["test"]
    eval_recs = te[:N_EVAL]

    # ---- ① 探针（读出）----
    fe_tr = featurize(meme_rows(model, tr, "prompt"))
    fe_te = featurize(meme_rows(model, te, "prompt"))
    y_tr = torch.tensor([float(r["y"]) for r in tr])
    y_te = torch.tensor([float(r["y"]) for r in te])
    d_tr = torch.tensor([len(str(int(r["y"]))) for r in tr])
    d_te = torch.tensor([len(str(int(r["y"]))) for r in te])
    maj_cnt = max(d_te.tolist().count(c) for c in set(d_te.tolist()))
    maj = maj_cnt / len(te)
    # 位数分类（线性 + 小 MLP）
    for mlp in (False, True):
        a_tr, a_te = _probe(fe_tr, d_tr, fe_te, d_te, "cls", mlp)
        r2_tr, r2_te = _probe(fe_tr, y_tr, fe_te, y_te, "reg", mlp)
        key = "mlp" if mlp else "linear"
        PROBE[(bucket, key)] = dict(cls_tr=a_tr, cls_te=a_te, r2_tr=r2_tr, r2_te=r2_te)
        print(f"[PROBE] {bucket} {key}(prompt段,2d={fe_tr.shape[1]}) | 位数分类 "
              f"train={a_tr*100:.1f}% test={a_te*100:.1f}% (多数类={maj*100:.1f}%) | "
              f"答案值回归 R² train={r2_tr:.3f} test={r2_te:.3f}", flush=True)
    # 对照：打乱标签的探针应 ≈ 机会水平（测量自检）
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(len(d_tr), generator=g)
    sh_a_tr, sh_a_te = _probe(fe_tr, d_tr[perm], fe_te, d_te, "cls", False)
    print(f"[CTRL] {bucket} 位数标签打乱后的线性探针 train={sh_a_tr*100:.1f}% "
          f"test={sh_a_te*100:.1f}%（应≈多数类水平，否则探针口径有问题）", flush=True)
    # 泄漏对照：整段（含答案 token）模因的读出上限
    le_tr = featurize(meme_rows(model, tr[:800], "full"))
    le_te = featurize(meme_rows(model, te, "full"))
    la_tr, la_te = _probe(le_tr, d_tr[:800], le_te, d_te, "cls", False)
    lr_tr, lr_te = _probe(le_tr, y_tr[:800], le_te, y_te, "reg", False)
    PROBE[(bucket, "full")] = dict(cls_tr=la_tr, cls_te=la_te, r2_tr=lr_tr, r2_te=lr_te)
    print(f"[PROBE] {bucket} linear(**整段含答案**,对照/泄漏) | 位数分类 train={la_tr*100:.1f}% "
          f"test={la_te*100:.1f}% | 回归 R² train={lr_tr:.3f} test={lr_te:.3f}", flush=True)

    # 模因量纲（给 σ 对照）
    with torch.no_grad():
        hs = meme_rows(model, eval_recs[:64], "prompt")
    rms = float(torch.sqrt((torch.stack([r.pow(2).mean() for r in hs])).mean()))
    print(f"[META] {bucket} 模因 h 每维 RMS={rms:.3f}（σ=0.1/1.0 相对它）", flush=True)

    # ---- 换样本模因的供体：每条 eval 样本整段前向的模因 ----
    HOOK["donor"] = [r for r in meme_rows(model, eval_recs, "full")]
    HOOK["donor"] = [r.clone() for r in HOOK["donor"]]

    # ---- 干预自检：每个干预确实改变了 logits（防空测试）----
    with torch.no_grad():
        ids0, _ = build_batch([eval_recs[0]])
        HOOK["fn"] = None
        base_l = model.logits(ids0)
        for nm, f in IVS_SEMI:
            # cur=3 ⇒ 「换样本」的供体是 eval[4]（≠本样本），避免自换导致假 0
            HOOK["fn"], HOOK["cur"] = f, 3
            dl = float((model.logits(ids0) - base_l).abs().max())
            print(f"[CHK] {bucket} 干预「{nm}」max|Δlogits|={dl:.4f}（应>0，否则=空测试）", flush=True)
    HOOK["fn"] = None

    if os.environ.get("S16_DBG2") == "1" and bucket == "add_1d":
        r0 = eval_recs[0]
        ids0t, _s0 = build_batch([r0])
        HOOK["fn"], HOOK["cur"] = None, 0
        l0 = model.logits(ids0t)
        HOOK["fn"], HOOK["cur"] = iv_swap, 0        # donor = eval[1]（+1 ⇒ 不是自己）
        l1 = model.logits(ids0t)
        print(f"[DBG2] teacher-forced ids={tuple(ids0t.shape)} donor={len(HOOK['donor'])} "
              f"donor[1]={tuple(HOOK['donor'][1].shape)} max|Δlogits|="
              f"{float((l0 - l1).abs().max()):.4f}", flush=True)
        HOOK["fn"] = None

    b_ce, b_dig = eval_ce(model, eval_recs, None)
    b_hits = eval_em(model, eval_recs, None)
    print(f"[BASE] {bucket} n={len(eval_recs)} EM={acc_of(b_hits)*100:.2f}%±{em_se_of(b_hits)*100:.2f}"
          f"（S14 锚点 {ANCHOR[bucket]*100:.1f}%） | 数字每步={math.exp(-sum(b_dig)/len(b_dig))*100:.2f}%"
          f" | 整体CE={sum(b_ce)/len(b_ce):.4f}", flush=True)

    for nm, f in IVS_SEMI:
        t1 = time.time()
        iv_ce, iv_dig = eval_ce(model, eval_recs, f)
        iv_hits = eval_em(model, eval_recs, f)
        if nm == "换样本模因" and os.environ.get("S16_DBG") == "1":
            print(f"[DBG] donor={None if HOOK['donor'] is None else len(HOOK['donor'])} "
                  f"fn={HOOK['fn']} | base dig[:3]={[round(v, 4) for v in b_dig[:3]]} "
                  f"iv dig[:3]={[round(v, 4) for v in iv_dig[:3]]}", flush=True)
            for k in range(3):
                print(f"[DBG] {k}: base={b_hits[k]['gen']!r} iv={iv_hits[k]['gen']!r} "
                      f"gold={b_hits[k]['gold']!r}", flush=True)
        CAUSAL[(bucket, nm)] = cond_report(f"{bucket} {nm}", b_hits, iv_hits, b_dig, iv_dig)
        CAUSAL[(bucket, nm)]["wall"] = time.time() - t1
    HOOK["donor"] = None

    print(f"[TIME] {bucket} 本 ckpt 合计 { (time.time()-t0)/60 :.1f} min", flush=True)
    del model

# ============================================================================
# ⑥ ★判决：Spearman ρ(探针准度, |Δ|)
# ============================================================================
IV5 = [n for n, _ in IVS]
xs_acc = [PROBE[(k, "linear")]["cls_te"] for k in CKS for n in IV5]
xs_r2 = [PROBE[(k, "linear")]["r2_te"] for k in CKS for n in IV5]
y_em = [abs(CAUSAL[(k, n)]["dem"]) for k in CKS for n in IV5]
y_acc = [abs(CAUSAL[(k, n)]["dacc"]) for k in CKS for n in IV5]
y_em_pp = [v * 100 for v in y_em]
y_acc_pp = [v * 100 for v in y_acc]

print("\n[SP] 点表 (ckpt, 干预, 探针test准度, 探针test R², |ΔEM|pp, |Δ数字|pp)", flush=True)
for i, (k, n) in enumerate([(k, n) for k in CKS for n in IV5]):
    print(f"[SP] {k:7s} {n:6s} x_acc={xs_acc[i]*100:5.1f}% x_r2={xs_r2[i]:6.3f} "
          f"|y_dem|={y_em_pp[i]:6.2f}pp |y_dacc|={y_acc_pp[i]:6.2f}pp", flush=True)

rho_em, p_em = spearman(xs_acc, y_em_pp)
rho_acc, p_acc = spearman(xs_acc, y_acc_pp)
rho_r2, p_r2 = spearman(xs_r2, y_em_pp)
n_pts = len(xs_acc)
print(f"\n[RHO] ★主判据 ρ(探针test分类准度, |ΔEM|) = {rho_em:.3f} (n={n_pts}, p={p_em:.3f})", flush=True)
print(f"[RHO] ρ(探针test分类准度, |Δ数字每步|) = {rho_acc:.3f} (n={n_pts}, p={p_acc:.3f})", flush=True)
print(f"[RHO] ρ(探针test R², |ΔEM|)            = {rho_r2:.3f} (n={n_pts}, p={p_r2:.3f})", flush=True)
print(f"[RHO] x 取值只有 {len(set(round(v,6) for v in xs_acc))} 个不同水平（2 ckpt）⇒ "
      f"ρ 本质是两组 |Δ| 的秩均值差，功效有限（限制如实记）", flush=True)

if n_pts < 6:
    verdict = "样本点 n<6 ⇒ 按判据无法判决"
elif rho_em < 0.5:
    verdict = (f"ρ={rho_em:.3f}<0.5 且 n={n_pts} ⇒ 弱相关 ⇒ **确证 P-3（读出≠会用）**")
elif rho_em >= 0.8:
    verdict = f"ρ={rho_em:.3f}≥0.8 ⇒ 强相关 ⇒ **推翻 P-3**"
else:
    verdict = f"ρ={rho_em:.3f} 落在 [0.5,0.8) 中间区间 ⇒ **不确定**（不硬选）"
print(f"[VERDICT-P3] {verdict}", flush=True)

# ---- 恒等中介对照（S12 GATE B：ΔCE(恒等-完整)=+0.0119）----
print("\n[IDENT] 恒等中介（思维卡=Identity）对照：", flush=True)
for k in CKS:
    r = CAUSAL[(k, "恒等中介")]
    print(f"[IDENT] {k}: ΔEM={r['dem']*100:+.2f}pp±{r['dem_se']*100:.2f} "
          f"(EM {r['em_b']*100:.2f}%→{r['em_i']*100:.2f}%) | "
          f"Δ数字每步={r['dacc']*100:+.2f}pp±{r['dacc_se']*100:.2f}", flush=True)
print("[IDENT] 对照：S12 A臂 GATE B 恒等中介 ΔCE(恒等−完整)=+0.0119±0.0046（EM 口径不同，只比方向）",
      flush=True)

print(f"\n[META] device={DEV} N_EVAL={N_EVAL} smoke={SMOKE} "
      f"总墙钟={(time.time()-T0)/60:.1f}min", flush=True)
print("[DONE] exit=0", flush=True)
