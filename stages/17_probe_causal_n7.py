#!/usr/bin/env python3
"""S17 · P-3 证据升级：把「读出 vs 会用」相关系数的 x 水平数从 2 提到 7（★零训练 · ★全程 CPU）。

为什么做（直接用 S16 的自我限定）：
  S16 已确证 P-3「读出 ≠ 会用」：ρ(探针准度, |ΔEM|)=0.283（n=10, p=0.43）。
  但它自己把强度限定为「弱-中」——**x 只有 2 个水平（仅 2 个 ckpt）**、add_3d 的 ΔEM 有 2.5% 地板效应。
  ⇒ 本单元**唯一目的：只提高 x 的水平数（2 → 7）**，用现成 ckpt，**不训练**。

现成资产（全部零训练，7 个 S14 ckpt）：
  5 桶 seed1234：add_1d / add_2d / add_3d / sub_2d / mul_2d
  第二 seed：add_1d@5678 / add_3d@5678（把 n 再加 2）

口径（与 S16 逐字一致，只换审计对象）：
  · 探针：模因取 **prompt 段**（防泄漏）、特征 [末位; 均值] → 256 维、train4000 训 / test800 评、
    **保留打乱标签对照**（+ 整段含答案的泄漏对照）；
  · patching：**清零 / 乱序 / 换样本** 三种（各自配对 Δ），EM 用 **batch=1**，并**同时报 Δ数字每步正确率**；
  · ★add_3d 的 ΔEM 有地板效应（基线仅 2.50%）⇒ **主判决量 = Δ数字每步正确率**（无地板问题），
    ΔEM **并列报告**；
  · **ρ 报两个版本**：ρ(探针, |ΔEM|) 与 ρ(探针, |Δ数字每步|)，都写 n 与 p；
  · 顺带补一条 **S16 完全同口径**的 ρ（5 种干预 × 7 ckpt = 35 点），只为和 S16 的 0.283(n=10) 对齐，
    **不参与主判决**（主判决只用上面三种）。

判据（写死）：ρ<0.5 且 n≥6 ⇒ P-3 确证（中-强）；ρ≥0.8 ⇒ 推翻；中间 ⇒ 如实报「不确定」。

复用方式（不重写模型）：把 `stages/14_synth_arith.py` 的**前半段源码**（到 `# ③ 主循环` 之前 =
tokenizer + 数据生成器 + Cards 模型 + build_batch/greedy_gen/hits_from/classify 等评估函数，
**不含任何训练循环**）exec 进本命名空间，逐字复用：
  · 14 号脚本 device!=cuda 时 `sys.exit(3)` ⇒ exec 期间把 sys.exit 换成 no-op（只在内存里）；
  · 那行「立即停」文案在内存里替换成 S17 说明；**源文件一个字节都不改**。

只允许写：本文件、logs/、/tmp。
"""
from __future__ import annotations

import math
import os
import sys
import time
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # ★先于 torch 导入：强制关 GPU（S15 占着，绝不碰）
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import Adam  # noqa: E402

torch.set_num_threads(int(os.environ.get("S17_THREADS", "8")))
DEV = "cpu"
T0 = time.time()
print(f"device: {DEV} | CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
      f"| torch.cuda.is_available()={torch.cuda.is_available()} "
      f"| threads={torch.get_num_threads()}", flush=True)
assert DEV == "cpu"
assert torch.cuda.is_available() is False, "必须全程 CPU"

SMOKE = os.environ.get("S17_SMOKE") == "1"
N_EVAL = 16 if SMOKE else int(os.environ.get("S17_N", "800"))
ROOT = "/home/vesita/coding/my/flowme"
SRC = f"{ROOT}/stages/14_synth_arith.py"

# ============================================================================
# ① 复用 stage 14 的定义（数据生成器 + Cards + 评估函数），切在训练主循环之前
# ============================================================================
_src = open(SRC, encoding="utf-8").read()
_cut = _src.index("# ③ 主循环")
_head = _src[:_cut].replace("GPU 起不来，立即停（不用 CPU 硬跑）。",
                            "S17: GPU 不可用 ⇒ 本单元按任务要求强制 CPU（零训练，只复用定义）")
_nlines = len(_head.splitlines())
_real_exit = sys.exit
sys.exit = lambda *a, **k: None          # 14 号脚本的 device 门禁：本单元是故意用 CPU
NS = {"__name__": "s14_reuse", "__file__": SRC}
exec(compile(_head, SRC, "exec"), NS)
sys.exit = _real_exit
print(f"[S17] 复用 {os.path.basename(SRC)} 前 {_nlines} 行（生成器+Cards+评估函数，"
      f"训练主循环在第 {_src[:_cut].count(chr(10))} 行之后，未执行 ⇒ 零训练）", flush=True)

Cards, D, FF, NHEAD = NS["Cards"], NS["D"], NS["FF"], NS["NHEAD"]
MAXLEN, MAX_GEN = NS["MAXLEN"], NS["MAX_GEN"]
PAD_ID, EOS_ID = NS["PAD_ID"], NS["EOS_ID"]
build_batch, greedy_gen, hits_from = NS["build_batch"], NS["greedy_gen"], NS["hits_from"]
classify, mean_se, enc = NS["classify"], NS["mean_se"], NS["enc"]
sin_pe, DATA, tok = NS["sin_pe"], NS["DATA"], NS["tok"]
sha16 = NS["sha16"] if "sha16" in NS else None

# ============================================================================
# ② ★唯一要改的一处：审计对象从 2 个 ckpt 扩到 7 个
# ============================================================================
CKS = ["add_1d@1234", "add_2d@1234", "add_3d@1234", "sub_2d@1234", "mul_2d@1234",
       "add_1d@5678", "add_3d@5678"]


def ck_path(key: str) -> str:
    b, s = key.split("@")
    return f"{ROOT}/logs/14_ckpt_{b}_seed{s}.pt"


CKPT = {k: ck_path(k) for k in CKS}
# S14/S16 锚点（比对用）：EM %
ANCHOR = {"add_1d@1234": 96.62, "add_2d@1234": 71.88, "add_3d@1234": 2.50,
          "sub_2d@1234": 61.38, "mul_2d@1234": 82.62,
          "add_1d@5678": 98.25, "add_3d@5678": 9.62}

# ============================================================================
# ③ 干预钩子：替换 Cards.logits 里「思维卡输出」这一份模因张量（不改权重）
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


def build_model(key: str):
    m = Cards(D, FF)
    sd = torch.load(CKPT[key], map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    m.logits = types.MethodType(_hooked_logits, m)
    return m


# ---------------- 干预定义（★主判决集 = 三种：清零 / 乱序 / 换样本）----------------
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


IVS_ALL = [("清零", iv_zero), ("乱序", iv_shuffle), ("换样本", iv_swap),
           ("加噪0.1", iv_noise(0.1)), ("加噪1.0", iv_noise(1.0))]
IVS = IVS_ALL[:3]        # ★主判决集（七行表与主 ρ 只用这三种）
IVS_S16 = IVS_ALL        # ★S16 同口径补算（5 干预 × 7 ckpt = 35 点），只用于与 S16 的 0.283 对齐

CAP: dict = {}


def cap_fn(h, h_in, ids):
    CAP["h"] = h
    return h


# ============================================================================
# ④ 模因特征（prompt 段，防答案泄漏）与探针 —— 口径与 S16 逐字一致
# ============================================================================
@torch.no_grad()
def meme_rows(model, recs, span: str = "prompt", bs: int = 64):
    """返回每条样本的模因行张量 [n,d]。span=prompt 只取 prompt 段（答案还没生成，无泄漏）。"""
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
    """kind='cls' 准度 / 'reg' R²。标准化只用 train 统计量；Adam 若干 epoch（探针=被测量物本身）。"""
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
# ⑤ 因果评估：配对 Δ（EM batch=1 主口径 + 数字每步正确率）
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
# ⑥ 主流程：7 个 ckpt × (探针 + 3 种干预)
# ============================================================================
PROBE: dict = {}
CAUSAL: dict = {}
BASE: dict = {}

for key in CKS:
    t0 = time.time()
    print(f"\n[S17] ===== ckpt={key} {os.path.basename(CKPT[key])} "
          f"sha16={sha16(CKPT[key])} eval N={N_EVAL} =====", flush=True)
    model = build_model(key)
    bucket = key.split("@")[0]
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
        k2 = "mlp" if mlp else "linear"
        PROBE[(key, k2)] = dict(cls_tr=a_tr, cls_te=a_te, r2_tr=r2_tr, r2_te=r2_te)
        print(f"[PROBE] {key} {k2}(prompt段,2d={fe_tr.shape[1]}) | 位数分类 "
              f"train={a_tr*100:.1f}% test={a_te*100:.1f}% (多数类={maj*100:.1f}%) | "
              f"答案值回归 R² train={r2_tr:.3f} test={r2_te:.3f}", flush=True)
    # 对照：打乱标签的探针应 ≈ 机会水平（测量自检）
    g = torch.Generator().manual_seed(7)
    perm = torch.randperm(len(d_tr), generator=g)
    sh_a_tr, sh_a_te = _probe(fe_tr, d_tr[perm], fe_te, d_te, "cls", False)
    print(f"[CTRL] {key} 位数标签打乱后的线性探针 train={sh_a_tr*100:.1f}% "
          f"test={sh_a_te*100:.1f}%（应≈多数类水平，否则探针口径有问题）", flush=True)
    # 泄漏对照：整段（含答案 token）模因的读出上限
    le_tr = featurize(meme_rows(model, tr[:800], "full"))
    le_te = featurize(meme_rows(model, te, "full"))
    la_tr, la_te = _probe(le_tr, d_tr[:800], le_te, d_te, "cls", False)
    lr_tr, lr_te = _probe(le_tr, y_tr[:800], le_te, y_te, "reg", False)
    PROBE[(key, "full")] = dict(cls_tr=la_tr, cls_te=la_te, r2_tr=lr_tr, r2_te=lr_te)
    print(f"[PROBE] {key} linear(**整段含答案**,对照/泄漏) | 位数分类 train={la_tr*100:.1f}% "
          f"test={la_te*100:.1f}% | 回归 R² train={lr_tr:.3f} test={lr_te:.3f}", flush=True)

    # ---- 换样本模因的供体：每条 eval 样本整段前向的模因 ----
    HOOK["donor"] = [r for r in meme_rows(model, eval_recs, "full")]
    HOOK["donor"] = [r.clone() for r in HOOK["donor"]]

    # ---- 干预自检：每个干预确实改变了 logits（防空测试）----
    with torch.no_grad():
        ids0, _ = build_batch([eval_recs[0]])
        HOOK["fn"] = None
        base_l = model.logits(ids0)
        for nm, f in IVS_ALL:
            # cur=3 ⇒ 「换样本」的供体是 eval[4]（≠本样本），避免自换导致假 0
            HOOK["fn"], HOOK["cur"] = f, 3
            dl = float((model.logits(ids0) - base_l).abs().max())
            print(f"[CHK] {key} 干预「{nm}」max|Δlogits|={dl:.4f}（应>0，否则=空测试）", flush=True)
    HOOK["fn"] = None

    b_ce, b_dig = eval_ce(model, eval_recs, None)
    b_hits = eval_em(model, eval_recs, None)
    em_b = acc_of(b_hits)
    d_anchor = (em_b - ANCHOR[key] / 100.0) * 100
    flag = "OK" if abs(d_anchor) <= 1.0 else "⚠差>1pp"
    BASE[key] = dict(em=em_b, em_se=em_se_of(b_hits), anchor=ANCHOR[key], diff_pp=d_anchor,
                     dig=math.exp(-sum(b_dig) / len(b_dig)), ce=sum(b_ce) / len(b_ce))
    print(f"[BASE] {key} n={len(eval_recs)} EM={em_b*100:.2f}%±{em_se_of(b_hits)*100:.2f}"
          f"（S14/S16 锚点 {ANCHOR[key]:.2f}% 差={d_anchor:+.2f}pp {flag}） | "
          f"数字每步={BASE[key]['dig']*100:.2f}% | 整体CE={BASE[key]['ce']:.4f}", flush=True)

    for nm, f in IVS_ALL:
        t1 = time.time()
        iv_ce, iv_dig = eval_ce(model, eval_recs, f)
        iv_hits = eval_em(model, eval_recs, f)
        CAUSAL[(key, nm)] = cond_report(f"{key} {nm}", b_hits, iv_hits, b_dig, iv_dig)
        CAUSAL[(key, nm)]["wall"] = time.time() - t1
    HOOK["donor"] = None

    print(f"[TIME] {key} 本 ckpt 合计 {(time.time()-t0)/60:.1f} min", flush=True)
    del model

# ============================================================================
# ⑦ ★七行表：每 ckpt 的探针 / ΔEM（三种）/ Δ数字每步（三种）
# ============================================================================
IV3 = [n for n, _ in IVS]
print("\n[TABLE] 七行表（探针 = 线性探针 prompt 段位数分类 test 准度 / 答案值回归 test R²；"
      "Δ 为 iv−base，n=800，EM batch=1）", flush=True)
print("[TABLE] " + "-" * 132, flush=True)
print("[TABLE] {ckpt:<13} | {acc:>7} {r2:>6} | {emb:>7} {anch:>7} | "
      "{e1:>9} {e2:>9} {e3:>9} | {a1:>9} {a2:>9} {a3:>9}".format(
          ckpt="ckpt", acc="探针acc", r2="R²", emb="基线EM", anch="(锚)",
          e1="ΔEM清零", e2="ΔEM乱序", e3="ΔEM换样本",
          a1="Δ数字清零", a2="Δ数字乱序", a3="Δ数字换样本"), flush=True)
print("[TABLE] " + "-" * 132, flush=True)
for key in CKS:
    p, b = PROBE[(key, "linear")], BASE[key]
    e = [CAUSAL[(key, n)] for n in IV3]
    print("[TABLE] {ckpt:<13} | {acc:>6.1f}% {r2:>6.3f} | {emb:>6.2f}% {anch:>6.2f}% | "
          "{e1:>+8.2f} {e2:>+8.2f} {e3:>+8.2f} | {a1:>+8.2f} {a2:>+8.2f} {a3:>+8.2f}".format(
              ckpt=key, acc=p["cls_te"] * 100, r2=p["r2_te"], emb=b["em"] * 100,
              anch=b["anchor"], e1=e[0]["dem"] * 100, e2=e[1]["dem"] * 100, e3=e[2]["dem"] * 100,
              a1=e[0]["dacc"] * 100, a2=e[1]["dacc"] * 100, a3=e[2]["dacc"] * 100), flush=True)
print("[TABLE] " + "-" * 132, flush=True)

# ============================================================================
# ⑧ 基线复测核对（7 ckpt 的 EM 必须与 S14/S16 锚一致，差 >1pp 要指出）
# ============================================================================
print("\n[ANCHOR] 基线 EM 复测核对（与 S14/S16 锚点，差 >1pp 标 ⚠）：", flush=True)
bad = []
for key in CKS:
    b = BASE[key]
    mark = "OK" if abs(b["diff_pp"]) <= 1.0 else "⚠差>1pp"
    if mark != "OK":
        bad.append(key)
    print(f"[ANCHOR] {key:<13} 实测={b['em']*100:6.2f}%±{b['em_se']*100:.2f} "
          f"锚={b['anchor']:6.2f}% 差={b['diff_pp']:+.2f}pp {mark} | "
          f"数字每步={b['dig']*100:.2f}% | 整体CE={b['ce']:.4f}", flush=True)
print(f"[ANCHOR] 汇总：7 个 ckpt 中 {len(bad)} 个差>1pp"
      f"{('：' + ', '.join(bad)) if bad else '（全部与锚一致）'}", flush=True)

# ============================================================================
# ⑨ ★判决：Spearman ρ 的两个版本（主判决量 = Δ数字每步，因 add_3d 的 ΔEM 有地板效应）
# ============================================================================
pts = [(k, n) for k in CKS for n in IV3]
xs_acc = [PROBE[(k, "linear")]["cls_te"] for k, n in pts]
xs_r2 = [PROBE[(k, "linear")]["r2_te"] for k, n in pts]
y_em = [abs(CAUSAL[(k, n)]["dem"]) * 100 for k, n in pts]
y_acc = [abs(CAUSAL[(k, n)]["dacc"]) * 100 for k, n in pts]

print("\n[SP] 点表 (ckpt, 干预, 探针test准度, 探针test R², |ΔEM|pp, |Δ数字|pp)", flush=True)
for i, (k, n) in enumerate(pts):
    print(f"[SP] {k:13s} {n:5s} x_acc={xs_acc[i]*100:5.1f}% x_r2={xs_r2[i]:6.3f} "
          f"|y_dem|={y_em[i]:6.2f}pp |y_dacc|={y_acc[i]:6.2f}pp", flush=True)

n21 = len(pts)
rho_em, p_em = spearman(xs_acc, y_em)
rho_acc, p_acc = spearman(xs_acc, y_acc)
rho_r2_em, p_r2_em = spearman(xs_r2, y_em)
rho_r2_acc, p_r2_acc = spearman(xs_r2, y_acc)
n_levels = len(set(round(v, 6) for v in xs_acc))
print(f"\n[RHO] ★版本1 ρ(探针test准度, |Δ数字每步|) = {rho_acc:.3f} (n={n21}=7ckpt×3干预, p={p_acc:.4f})",
      flush=True)
print(f"[RHO] ★版本2 ρ(探针test准度, |ΔEM|)       = {rho_em:.3f} (n={n21}=7ckpt×3干预, p={p_em:.4f})",
      flush=True)
print(f"[RHO] 辅助 ρ(探针test R², |Δ数字每步|)     = {rho_r2_acc:.3f} (n={n21}, p={p_r2_acc:.4f})",
      flush=True)
print(f"[RHO] 辅助 ρ(探针test R², |ΔEM|)           = {rho_r2_em:.3f} (n={n21}, p={p_r2_em:.4f})",
      flush=True)

# ★与 S16 完全同口径的补算（5 种干预 × 7 ckpt = 35 点，S16 是 5 × 2 = 10 点 ρ=0.283）
pts5 = [(k, n) for k in CKS for n in [n for n, _ in IVS_S16]]
xs5 = [PROBE[(k, "linear")]["cls_te"] for k, n in pts5]
ye5 = [abs(CAUSAL[(k, n)]["dem"]) * 100 for k, n in pts5]
ya5 = [abs(CAUSAL[(k, n)]["dacc"]) * 100 for k, n in pts5]
rho5_em, p5_em = spearman(xs5, ye5)
rho5_acc, p5_acc = spearman(xs5, ya5)
print(f"[RHO] S16同口径(5干预×7ckpt) ρ(准度, |ΔEM|)={rho5_em:.3f} (n={len(pts5)}, p={p5_em:.4f}) | "
      f"ρ(准度, |Δ数字每步|)={rho5_acc:.3f} (n={len(pts5)}, p={p5_acc:.4f}) "
      f"（S16 原值 0.283/0.313，n=10）", flush=True)

# 每 ckpt 先对 3 种干预取均值 ⇒ n=7 的独立版本（x 每个水平只出现一次）
xs7 = [PROBE[(k, "linear")]["cls_te"] for k in CKS]
ye7 = [sum(abs(CAUSAL[(k, n)]["dem"]) for n in IV3) / 3 * 100 for k in CKS]
ya7 = [sum(abs(CAUSAL[(k, n)]["dacc"]) for n in IV3) / 3 * 100 for k in CKS]
rho_em7, p_em7 = spearman(xs7, ye7)
rho_acc7, p_acc7 = spearman(xs7, ya7)
print(f"[RHO] 每ckpt取均值后 n=7 版本：ρ(准度, |Δ数字每步|)={rho_acc7:.3f} (n=7, p={p_acc7:.4f}) | "
      f"ρ(准度, |ΔEM|)={rho_em7:.3f} (n=7, p={p_em7:.4f})", flush=True)
print(f"[RHO] ★x 水平数 = {n_levels} 个不同取值（S16 只有 2 个）；"
      f"x 范围 {min(xs_acc)*100:.1f}%..{max(xs_acc)*100:.1f}% "
      f"（21 个点由 7 个水平 × 3 干预构成，ρ 的 p 值按 n=21 算 ⇒ 同水平内为并列秩）", flush=True)


def verdict_of(rho, n, tag):
    if n < 6:
        return f"{tag}: n={n}<6 ⇒ 按判据无法判决"
    if rho < 0.5:
        return f"{tag}: ρ={rho:.3f}<0.5 且 n={n} ⇒ 弱相关 ⇒ **P-3 确证（读出≠会用，中-强）**"
    if rho >= 0.8:
        return f"{tag}: ρ={rho:.3f}≥0.8 ⇒ 强相关 ⇒ **P-3 被推翻（读出会用同向）**"
    return f"{tag}: ρ={rho:.3f} 落在 [0.5,0.8) 中间区间 ⇒ **不确定**（不硬选）"


v_main = verdict_of(rho_acc, n21, "[VERDICT-P3·主判据Δ数字每步]")
v_em = verdict_of(rho_em, n21, "[VERDICT-P3·并列ΔEM]")
print(f"\n{v_main}", flush=True)
print(f"{v_em}", flush=True)
print(f"[VERDICT-P3] n=7 独立版本：{verdict_of(rho_acc7, 7, 'Δ数字每步')} | "
      f"{verdict_of(rho_em7, 7, 'ΔEM')}", flush=True)

# ============================================================================
# ⑩ 回填：读出最强 vs 最弱的卡，因果差多少倍
# ============================================================================
r2s = {k: PROBE[(k, "linear")]["r2_te"] for k in CKS}
best = max(r2s, key=r2s.get)
worst = min(r2s, key=r2s.get)
mean_em = {k: sum(abs(CAUSAL[(k, n)]["dem"]) for n in IV3) / 3 * 100 for k in CKS}
mean_ac = {k: sum(abs(CAUSAL[(k, n)]["dacc"]) for n in IV3) / 3 * 100 for k in CKS}
rat_em = mean_em[best] / mean_em[worst] if mean_em[worst] else float("inf")
rat_ac = mean_ac[best] / mean_ac[worst] if mean_ac[worst] else float("inf")
print(f"\n[RATIO] 读出最强 = {best}（R²={r2s[best]:.3f}，准度={PROBE[(best,'linear')]['cls_te']*100:.1f}%）"
      f" vs 读出最弱 = {worst}（R²={r2s[worst]:.3f}，"
      f"准度={PROBE[(worst,'linear')]['cls_te']*100:.1f}%）", flush=True)
print(f"[RATIO] 三种干预平均 |ΔEM|：{best}={mean_em[best]:.2f}pp vs {worst}={mean_em[worst]:.2f}pp "
      f"⇒ **{rat_em:.1f}×**；三种干预平均 |Δ数字每步|：{best}={mean_ac[best]:.2f}pp vs "
      f"{worst}={mean_ac[worst]:.2f}pp ⇒ **{rat_ac:.1f}×**", flush=True)
print(f"[RATIO] 读出差距只有 R² {r2s[worst]:.3f}→{r2s[best]:.3f}（{(r2s[best]/r2s[worst]):.2f}×）"
      f"却对应因果 {rat_ac:.1f}×（Δ数字口径）；注意 ΔEM 口径含 {worst} 的地板效应"
      f"（基线 EM 仅 {BASE[worst]['em']*100:.2f}%），故以 Δ数字每步为准", flush=True)
# 任务点名的那一对（add_1d vs add_3d）单独再算一次，避免与「实测最弱」混淆
A, B = "add_1d@1234", "add_3d@1234"
print(f"[RATIO] 点名对 add_1d@1234（R²={r2s[A]:.3f}）vs add_3d@1234（R²={r2s[B]:.3f}）："
      f"平均|ΔEM| {mean_em[A]:.2f}pp vs {mean_em[B]:.2f}pp ⇒ {mean_em[A]/mean_em[B]:.1f}×；"
      f"平均|Δ数字每步| {mean_ac[A]:.2f}pp vs {mean_ac[B]:.2f}pp ⇒ "
      f"**{mean_ac[A]/mean_ac[B]:.1f}×**（ΔEM 有 {BASE[B]['em']*100:.2f}% 地板，以 Δ数字为准）",
      flush=True)

print(f"\n[META] device={DEV} N_EVAL={N_EVAL} smoke={SMOKE} ckpt数={len(CKS)} "
      f"x水平数={n_levels} 总墙钟={(time.time()-T0)/60:.1f}min", flush=True)
print("[DONE] exit=0", flush=True)
