#!/usr/bin/env python3
"""S30 · 验证推测 8：同一模型上「集中度」(前20%单元的因果占比) 与「有效维度 PR」是否强负相关？
（★全程 CPU · ★零训练 · 只读现成 ckpt）

判据（写死）：
  主判据 Spearman ρ(PR, 集中度)：ρ<−0.6 且 p<0.01 ⇒ 推测8成立（可统一为"有效自由度"）；
  |ρ|<0.3 ⇒ 推测8被证伪（两者独立）；中间 ⇒ 不确定。
  分层：EM≥20% 子集算一次；EM<5% 子集算一次。
  方向性 sanity：ρ(PR, EM) 预期正、ρ(集中度, EM) 预期负。

口径复用（零改动、零训练）：
  · 集中度/因果面：exec stages/21_info_causal_curve.py 的「# ⑥ 主流程」之前那一段
    ⇒ 拿到 curve_stats / perm_test / dacc_pp / nmean / iv_pos / iv_blk / NBLK / set_hooks
    （其内部又 exec 17_probe_causal_n7.py 前半段 → 14_synth_arith.py 前半段）
    ⇒ 三个训练主循环都在被 exec 的切片之外。
  · PR：stages/26_meme_anchoring.py:257-261 的 pr_eff 逐字复用
    （去均值后 svdvals²，PR=(Σλ)²/Σλ²，作用于 mean-pool 后的 [K,d]）。
  · 干预 = S21 的非破坏性单点清零 side=in；CE = |Δ数字每步正确率|，同 dacc_pp。

★与 S21 的唯一差异（记录在案）：S21 的 eval_ce 是 batch=1 逐样本前向；本单元为省墙钟改成
  等价批式（teacher-forced，逐样本量完全相同；批次只影响速度不影响数值）⇒ 首个 ckpt 做互检。

只允许写：本文件、logs/（本单元因 S28 独占 logs/ 改写 /tmp/s30/）、/tmp。
"""
from __future__ import annotations

import json
import math
import os
import time
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ★先于 torch 导入：强制关 GPU
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = "/home/vesita/coding/my/flowme"
SRC21 = f"{ROOT}/stages/21_info_causal_curve.py"
OUT = os.environ.get("S30_OUT", "/tmp/s30")
os.makedirs(OUT, exist_ok=True)

# ============================================================================
# ① 逐字复用 S21 前半段（含其内部对 S17/S14 前半段的 exec）
# ============================================================================
_s21 = open(SRC21, encoding="utf-8").read()
_c21 = _s21.rindex("# ⑥ 主流程")        # 注意：S21 的 docstring 里也出现过该字面量 ⇒ 必须取最后一次
NS21 = {"__name__": "s21_head", "__file__": SRC21}
exec(compile(_s21[:_c21], SRC21, "exec"), NS21)          # noqa: S102
for _k, _v in NS21.items():
    if not _k.startswith("__"):
        globals()[_k] = _v

# 拿到 S17 真正的命名空间 ⇒ 把 CAP/cap_fn 换成能同时抓 h 与 h_in 的版本
# ★注意：meme_rows / eval_ce 带 @torch.no_grad() 装饰器，`meme_rows.__globals__` 是装饰器包装层的
#   globals（不含 HOOK/CAP）⇒ 必须用未被包装的 cap_fn 反查命名空间（实测 71 个名字、含 HOOK）。
_G17 = globals()["cap_fn"].__globals__
assert "HOOK" in _G17 and "CAP" in _G17 and "meme_rows" in _G17, "命名空间反查失败"
CAP30: dict = {}


def cap30(h, h_in, ids):
    CAP30["h"], CAP30["h_in"] = h, h_in
    return h


_G17["CAP"], _G17["cap_fn"] = CAP30, cap30
HOOK = globals()["HOOK"]
assert HOOK is _G17["HOOK"], "HOOK 必须是同一个 dict 对象"
# ============================================================================
# ② S30 配置
# ============================================================================
T0 = time.time()
DEV = "cpu"
assert not torch.cuda.is_available(), "必须全程 CPU"
N_EVAL = int(os.environ.get("S30_N", "800"))
BS_CE = int(os.environ.get("S30_BS", "64"))
torch.set_num_threads(int(os.environ.get("S30_THREADS", "12")))
NBLK, BLKD = 8, 128 // 8                                  # 与 S21 同：8 个 16 维块
CKPT_DIR = f"{ROOT}/logs"

print(f"[S30] ===== 集中度 vs 有效维度 PR | device={DEV} | "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} | "
      f"torch.cuda.is_available()={torch.cuda.is_available()} | threads={torch.get_num_threads()} | "
      f"N_EVAL={N_EVAL} BS_CE={BS_CE} OUT={OUT} =====", flush=True)
print("[S30] 口径：集中度=curve_stats 前 round(0.2N) 单元 |Δ数字每步| 累积占比；"
      "PR=26_meme_anchoring.py:257 pr_eff（去均值 svdvals²，mean-pool [K,d]）；干预=S21 iv_pos/iv_blk side=in",
      flush=True)


# ============================================================================
# ③ ckpt 清单（覆盖 5 桶 × 结构臂(res/nores/无残差) × 能力水平）
# ============================================================================
def ck(name: str) -> str:
    return f"{CKPT_DIR}/{name}"


def s20_remap(k: str) -> str:
    """S20 的 Cards20 把 think 卡包了一层 ⇒ thought.inner.* → thought.*（S20 自带 s19_key_map 的逆）。"""
    return k.replace("thought.inner.", "thought.") if k.startswith("thought.inner.") else k


ALL_CKPTS = [
    # --- S14 五桶（无残差臂）---
    dict(tag="S14/add_1d/1234",      path=ck("14_ckpt_add_1d_seed1234.pt"),      bucket="add_1d", residual=False, remap=None),
    dict(tag="S14/add_1d/5678",      path=ck("14_ckpt_add_1d_seed5678.pt"),      bucket="add_1d", residual=False, remap=None),
    dict(tag="S14/add_2d/1234",      path=ck("14_ckpt_add_2d_seed1234.pt"),      bucket="add_2d", residual=False, remap=None),
    dict(tag="S14/add_3d/1234",      path=ck("14_ckpt_add_3d_seed1234.pt"),      bucket="add_3d", residual=False, remap=None),
    dict(tag="S14/add_3d/5678",      path=ck("14_ckpt_add_3d_seed5678.pt"),      bucket="add_3d", residual=False, remap=None),
    dict(tag="S14/sub_2d/1234",      path=ck("14_ckpt_sub_2d_seed1234.pt"),      bucket="sub_2d", residual=False, remap=None),
    dict(tag="S14/mul_2d/1234",      path=ck("14_ckpt_mul_2d_seed1234.pt"),      bucket="mul_2d", residual=False, remap=None),
    # --- S19 五桶 × res/nores 两臂 ---
    dict(tag="S19/add_1d/res/1234",  path=ck("19_ckpt_add_1d_res_seed1234.pt"),  bucket="add_1d", residual=True,  remap=None),
    dict(tag="S19/add_2d/res/1234",  path=ck("19_ckpt_add_2d_res_seed1234.pt"),  bucket="add_2d", residual=True,  remap=None),
    dict(tag="S19/add_3d/res/1234",  path=ck("19_ckpt_add_3d_res_seed1234.pt"),  bucket="add_3d", residual=True,  remap=None),
    dict(tag="S19/sub_2d/res/5678",  path=ck("19_ckpt_sub_2d_res_seed5678.pt"),  bucket="sub_2d", residual=True,  remap=None),
    dict(tag="S19/mul_2d/res/5678",  path=ck("19_ckpt_mul_2d_res_seed5678.pt"),  bucket="mul_2d", residual=True,  remap=None),
    dict(tag="S19/add_1d/nores/1234", path=ck("19_ckpt_add_1d_nores_seed1234.pt"), bucket="add_1d", residual=False, remap=None),
    dict(tag="S19/add_3d/nores/1234", path=ck("19_ckpt_add_3d_nores_seed1234.pt"), bucket="add_3d", residual=False, remap=None),
    dict(tag="S19/add_3d/nores/5678", path=ck("19_ckpt_add_3d_nores_seed5678.pt"), bucket="add_3d", residual=False, remap=None),
    # --- S20 位置身份两臂（残差臂 + Cards20 包裹）---
    dict(tag="S20/add3d/pos/1234",   path=ck("20_ckpt_add3d_pos_seed1234.pt"),   bucket="add_3d", residual=True,  remap=s20_remap),
    dict(tag="S20/add3d/nopos/1234", path=ck("20_ckpt_add3d_nopos_seed1234.pt"), bucket="add_3d", residual=True,  remap=s20_remap),
    dict(tag="S20/add3d/pos/5678",   path=ck("20_ckpt_add3d_pos_seed5678.pt"),   bucket="add_3d", residual=True,  remap=s20_remap),
    # --- S25 单卡五臂（残差臂，能力跨度最大）---
    dict(tag="S25/e2e/1234",         path=ck("25_ckpt_e2e_seed1234.pt"),         bucket="add_3d", residual=True,  remap=None),
    dict(tag="S25/anchor/1234",      path=ck("25_ckpt_anchor_seed1234.pt"),      bucket="add_3d", residual=True,  remap=None),
    dict(tag="S25/alt/5678",         path=ck("25_ckpt_alt_seed5678.pt"),         bucket="add_3d", residual=True,  remap=None),
    dict(tag="S25/randtgt/1234",     path=ck("25_ckpt_randtgt_seed1234.pt"),     bucket="add_3d", residual=True,  remap=None),
    dict(tag="S25/randall/1234",     path=ck("25_ckpt_randall_seed1234.pt"),     bucket="add_3d", residual=True,  remap=None),
]
ONLY = os.environ.get("S30_ONLY", "")
if ONLY:
    keep = {x for x in ONLY.split(",")}
    ALL_CKPTS = [j for j in ALL_CKPTS if j["tag"] in keep]
JOBS = [j for j in ALL_CKPTS if os.path.exists(j["path"])]
MISSING = [j["tag"] for j in ALL_CKPTS if not os.path.exists(j["path"])]
print(f"[S30] ckpt 计划 {len(ALL_CKPTS)} 个 ⇒ 实到 {len(JOBS)} 个" +
      (f" | ★缺 {MISSING}" if MISSING else " | 无缺"), flush=True)


# ============================================================================
# ④ 残差版 hooked logits（S21 _hooked_logits21 逐行相同，只多抓 h_in）+ 输入侧干预钩子
# ============================================================================
def _hooked_logits30(self, ids: torch.Tensor) -> torch.Tensor:
    n = ids.size(1)
    m = torch.full((n, n), float("-inf"), device=ids.device)
    m = torch.triu(m, diagonal=1)
    m = m.unsqueeze(0).expand(ids.size(0), -1, -1).clone()
    m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
    m = m.repeat_interleave(NHEAD, dim=0)
    pe = self.pe if n <= self.pe.size(0) else sin_pe(n, self.d).to(self.pe.device)
    assert n <= pe.size(0)
    x = self.emb(ids) + pe[:n].unsqueeze(0)
    h = self.in_enc(x, src_mask=m)
    h_in = h
    if self.thought is not None:
        if getattr(self, "residual", False):
            h = h + self.thought(h, src_mask=m)          # ★res 臂
        else:
            h = self.thought(h, src_mask=m)              # S14 现状：无残差
    HOOK["n_call"] += 1
    if HOOK["fn"] is not None:
        h = HOOK["fn"](h, h_in, ids)
    return self.head(h)


def build_model30(job):
    m = Cards(D, FF)
    sd = torch.load(job["path"], map_location="cpu", weights_only=True)
    if job["remap"] is not None:
        sd = {job["remap"](k): v for k, v in sd.items()}
    m.load_state_dict(sd, strict=True)
    m.eval()
    m.residual = job["residual"]
    m.logits = types.MethodType(_hooked_logits30, m)
    m.thought = _InPatch(m.thought)
    return m


@torch.no_grad()
def meme_rows2(model, recs, bs: int = 64):
    """S17 meme_rows(span='prompt') 同构，只多返回 h_in（输入卡输出）。"""
    model.eval()
    HOOK["fn"], HOOK["donor"] = cap30, None
    H, HI = [], []
    try:
        for i in range(0, len(recs), bs):
            ch = recs[i:i + bs]
            lens = [len(r["p"]) for r in ch]
            n = max(lens)
            ids = torch.full((len(ch), n), PAD_ID, dtype=torch.long)
            for j, r in enumerate(ch):
                ids[j, : lens[j]] = torch.tensor(r["p"])
            model.logits(ids)
            H += [CAP30["h"][j, : lens[j]].clone() for j in range(len(ch))]
            HI += [CAP30["h_in"][j, : lens[j]].clone() for j in range(len(ch))]
    finally:
        HOOK["fn"] = None
    return H, HI


@torch.no_grad()
def dig_ce_batched(model, recs, hook, bs: int = BS_CE):
    """S17 eval_ce 的第二返回值 dig_ce（逐样本数字位 CE）的等价批式版。"""
    model.eval()
    set_hooks(hook, None)
    out: list[float] = []
    try:
        for i in range(0, len(recs), bs):
            ch = recs[i:i + bs]
            ids, s = build_batch(ch)
            logp = torch.log_softmax(model.logits(ids), dim=-1)
            for j, r in enumerate(ch):
                e = int(s[j]) + len(r["t"])
                lp = logp[j, int(s[j]) - 1: e - 1]
                tgt = ids[j, int(s[j]): e]
                ce = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1)
                dk = [k for k in range(ce.numel()) if classify(int(tgt[k])) == 0]
                out.append(float(ce[dk].mean()) if dk else float("nan"))
    finally:
        set_hooks(None, None)
    return out


def pr_eff(X: torch.Tensor) -> float:
    """S26 stages/26_meme_anchoring.py:257-261 逐字复用：PR=(Σλ)²/Σλ²（λ=去均值后奇异值）。"""
    Xc = X - X.mean(0, keepdim=True)
    s2 = torch.linalg.svdvals(Xc) ** 2
    return float(s2.sum() ** 2 / (s2 ** 2).sum())


def pool_mean(rows):
    return torch.stack([r.mean(0) for r in rows])


def pr_posavg(rows, n_pos):
    """逐位置 PR = 对每个位置算 PR 再平均（防池化掩盖结构）。"""
    vals = []
    for p in range(n_pos):
        sel = [r[p] for r in rows if r.size(0) > p]
        if len(sel) >= 8:
            vals.append(pr_eff(torch.stack(sel)))
    return float(np.mean(vals)) if vals else float("nan")


# ============================================================================
# ⑤ 主流程：每个 ckpt ⇒ EM · PR(pooled/逐位置, h/h_in) · 集中度(前20%) · R29
# ============================================================================
PTS_PATH = f"{OUT}/points.jsonl"
open(PTS_PATH, "w").close()
SUM: list[dict] = []

for ji, job in enumerate(JOBS):
    t0 = time.time()
    tag, bucket = job["tag"], job["bucket"]
    tr, te = DATA[bucket]["train"], DATA[bucket]["test"]
    eval_recs = te[:N_EVAL]
    n_pos = min(min(len(r["p"]) for r in tr), min(len(r["p"]) for r in te))   # 与 S21 同口径
    model = build_model30(job)

    rows_h, rows_hin = meme_rows2(model, eval_recs)
    PR_h, PR_hin = pr_eff(pool_mean(rows_h)), pr_eff(pool_mean(rows_hin))
    PRp_h, PRp_hin = pr_posavg(rows_h, n_pos), pr_posavg(rows_hin, n_pos)

    # 口径互检（仅首个 ckpt）：批式 dig_ce vs S17 的 batch=1 eval_ce
    chk = "-"
    if ji == 0:
        sub = eval_recs[:16]
        a = np.array(dig_ce_batched(model, sub, None), float)
        b = np.array(eval_ce(model, sub, None)[1], float)
        hk = iv_pos(0)
        a2 = np.array(dig_ce_batched(model, sub, hk), float)
        b2 = np.array(run_ce(model, sub, hk, "in")[1], float)
        ok0 = bool(np.allclose(a, b, equal_nan=True, atol=1e-6))
        ok1 = bool(np.allclose(a2, b2, equal_nan=True, atol=1e-6))
        chk = f"{'OK' if ok0 and ok1 else '★不一致'} (零干预 {ok0}, pos0清零 {ok1}, max|Δ|={np.nanmax(np.abs(a-b)):.2e})"
    print(f"\n[S30] ===== [{ji+1}/{len(JOBS)}] {tag} | {os.path.basename(job['path'])} | bucket={bucket} "
          f"| residual={job['residual']} | n_pos={n_pos} | 单元={n_pos}+{NBLK}={n_pos+NBLK} | 互检 {chk} =====",
          flush=True)

    # 能力（EM，batch=1，n=N_EVAL）
    t_em = time.time()
    b_hits = run_em(model, eval_recs, None, "in")
    em = acc_of(b_hits)
    t_em = time.time() - t_em

    # 集中度：主轴逐单元 |Δ数字每步|
    t_ce = time.time()
    b_dig = dig_ce_batched(model, eval_recs, None)
    units = [("pos", i) for i in range(n_pos)] + [("blk", b) for b in range(NBLK)]
    ces, ivs = [], []
    for kind, idx in units:
        hk = iv_pos(idx) if kind == "pos" else iv_blk(idx)
        iv_dig = dig_ce_batched(model, eval_recs, hk)
        d = dacc_pp(b_dig, iv_dig)
        ces.append(abs(d))
        ivs.append(d)
        with open(PTS_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(tag=tag, kind=kind, idx=idx, dacc=d, ce=abs(d))) + "\n")
    t_ce = time.time() - t_ce
    _q, shares, _st = curve_stats(ces, [1.0] * len(ces))     # ★S21 的 curve_stats 原函数
    conc = float(shares[0.2])
    nz = sum(1 for x in ces if x > 1e-3) / len(ces)          # R29：Δ 非零比例
    med = float(np.median(ces))

    SUM.append(dict(tag=tag, bucket=bucket, residual=job["residual"], n_pos=n_pos, n_units=len(units),
                    em=em, PR=PR_h, PR_hin=PR_hin, PR_pos=PRp_h, PR_pos_hin=PRp_hin,
                    conc=conc, nz=nz, med_ce=med, wall=time.time() - t0))
    print(f"[RES] {tag:<22} EM={em*100:6.2f}% | PR_pool(h)={PR_h:6.2f} PR_pool(h_in)={PR_hin:6.2f} | "
          f"PR_pos(h)={PRp_h:6.2f} | 集中度(前20%)={conc*100:5.1f}% | R29 非零={nz*100:5.1f}% "
          f"中位|Δ|={med:.3f}pp | EM={t_em:.1f}s CE={t_ce:.1f}s 墙钟={time.time()-t0:.1f}s", flush=True)
    del model

# ============================================================================
# ⑥ 判决
# ============================================================================
S = SUM
pr = np.array([x["PR"] for x in S], float)
po = np.array([x["PR_pos"] for x in S], float)
pi = np.array([x["PR_hin"] for x in S], float)
cc = np.array([x["conc"] for x in S], float)
em = np.array([x["em"] for x in S], float)


def rho(x, y, name):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return dict(name=name, n=len(x), rho=float("nan"), p=float("nan"))
    r = spearmanr(x, y)
    return dict(name=name, n=len(x), rho=float(r.statistic), p=float(r.pvalue))


MAIN = rho(pr, cc, "ρ(PR_pool(h), 集中度) 全池")
R_pos = rho(po, cc, "ρ(PR_pos(h), 集中度) 全池")
R_hin = rho(pi, cc, "ρ(PR_pool(h_in), 集中度) 全池")
m20 = em >= 0.20
m5 = em < 0.05
S20 = rho(pr[m20], cc[m20], "ρ(PR, 集中度) EM≥20%")
S05 = rho(pr[m5], cc[m5], "ρ(PR, 集中度) EM<5%")
SAN1 = rho(pr, em, "ρ(PR, EM) 方向性")
SAN2 = rho(cc, em, "ρ(集中度, EM) 方向性")

print("\n" + "=" * 100, flush=True)
for r in (MAIN, R_pos, R_hin, S20, S05, SAN1, SAN2):
    print(f"[RHO] {r['name']:<32} n={r['n']:>2} ρ={r['rho']:+.3f} p={r['p']:.4f}", flush=True)
print(f"[STRAT] EM≥20% 子集 {int(m20.sum())} 个：{sorted(x['tag'] for x, m in zip(S, m20) if m)}", flush=True)
print(f"[STRAT] EM<5%  子集 {int(m5.sum())} 个：{sorted(x['tag'] for x, m in zip(S, m5) if m)}", flush=True)
print(f"[R29] 全池 Δ 非零(>1e-3pp)比例：中位={np.median([x['nz'] for x in S])*100:.1f}% "
      f"最小={min(x['nz'] for x in S)*100:.1f}% | |Δ| 中位（跨 ckpt 再取中位）="
      f"{np.median([x['med_ce'] for x in S]):.3f}pp ⇒ 度量非常数", flush=True)
print(f"[META] device={DEV} cuda_avail={torch.cuda.is_available()} n_ckpt={len(S)} N_EVAL={N_EVAL} "
      f"墙钟={(time.time()-T0)/60:.1f}min", flush=True)

with open(f"{OUT}/summary.json", "w", encoding="utf-8") as f:
    json.dump(dict(rows=S, rhos=[MAIN, R_pos, R_hin, S20, S05, SAN1, SAN2],
                   n_eval=N_EVAL, wall_min=(time.time() - T0) / 60), f, ensure_ascii=False, indent=1)
print("[DONE] exit=0", flush=True)
