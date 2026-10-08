#!/usr/bin/env python3
"""S21 · 模因信息-因果曲线：把模因 [n,d] 的信息按「因果可用性」分层（★全程 CPU · ★零训练 · 现成 ckpt）。

唯一问题：累积因果效应 随 累积信息量（可读性）如何增长？
  （类比洛伦兹曲线：若「前 20% 单元贡献 ≥50% 因果效应」⇒ 信息高度集中、大量信息因果不可用。）

三条主口径（每条都报）：
  1) 可读性 R_i：只保留第 i 个位置（或第 b 个 16 维块）的模因、其余置零 ⇒ 同一个
     featurize([末位;均值])（256 维）+ 同一个线性探针 ⇒ test 位数分类准度 与 答案值 R²（S17 口径逐字复用）。
  2) 因果效应 CE_i：**非破坏性单点干预**（清零第 i 个位置的模因、其余不动；S18 的 side=in 口径）
     ⇒ 配对 Δ数字每步正确率 与 ΔEM（batch=1，R28）；同一 ckpt 内的 clean 基线 ⇒ 天然控制基线。
  3) ★曲线：按 CE_i 降序排单元 ⇒ 累积因果占比 vs 累积可读性占比（5 个分位点）+ 前 k% 贡献占比（k=10/20/50）。

为什么不用「先测探针准度、再测破坏性干预 Δ、再算相关」：S17 实测该路让 ρ 从 0.28 跳到 0.74 ——
  破坏性干预下 |Δ| ≈ 基线成绩 ⇒ 那个 ρ 测的是「模型好坏 vs 探针准度」（Canby & Davies 2408.15510 等点名）。
  本单元只走【同一干预轴 · 同一 clean 基线 · 非破坏性单点 · 成对记 (可读性, 因果效应)】。
R29：每种干预先报 Δ 非零比例；非零比例过低 ⇒ 该轴作废并剔除
  （已知：字面版「互换两行模因」的读列 Δ 恒 0/浮点噪声 ⇒ 不能当「因果效应=0」用）。
R28：自回归生成只用 batch=1。

复用方式（既有源文件零改动、零训练）：
  exec stages/17_probe_causal_n7.py 的「# ⑥ 主流程」之前那一段
  （其内部又 exec stages/14_synth_arith.py 的「# ③ 主循环」之前那一段 = tokenizer + 数据生成器 + Cards
   + build_batch/greedy_gen/hits_from/classify/eval_ce/eval_em/meme_rows/_probe/cond_report/spearman）
  ⇒ 两个训练主循环都在被 exec 的切片之外。S21 只新增：残差版 hooked logits、输入侧干预钩子、
  闭式局部特征、曲线/置换检验/自检。

只允许写：本文件、logs/、/tmp。
"""
from __future__ import annotations

import json
import math
import os
import time
import types

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ★先于 torch 导入：强制关 GPU
import numpy as np  # noqa: E402

ROOT = "/home/vesita/coding/my/flowme"
SRC17 = f"{ROOT}/stages/17_probe_causal_n7.py"

# ============================================================================
# ① 逐字复用 S17 前半段（其内部已 exec S14 前半段 ⇒ 模型/数据/评估函数全部现成）
# ============================================================================
_s17 = open(SRC17, encoding="utf-8").read()
_c17 = _s17.index("# ⑥ 主流程")
NS17 = {"__name__": "s17_head", "__file__": SRC17}
exec(compile(_s17[:_c17], SRC17, "exec"), NS17)          # noqa: S102
for _k, _v in NS17.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
r28_selfcheck = NS["r28_selfcheck"]                      # 在 S14 命名空间里
CHK_BATCH = NS["CHK_BATCH"]                              # S14 的 batch 自检口径（=16）

# ============================================================================
# ② S21 配置（在名字回填之后设置，避免被 S17/S14 同名变量盖掉）
# ============================================================================
T0 = time.time()
SMOKE = os.environ.get("S21_SMOKE") == "1"
N_EVAL = int(os.environ.get("S21_N", "800"))
N_R29 = int(os.environ.get("S21_R29N", "64"))            # R29 冒烟子集大小
N_ID = int(os.environ.get("S21_IDN", "64"))              # 零干预恒等自检子集
EM_BATCH = int(os.environ.get("S21_EMB", "16"))          # 仅用于 R28 自检
NPERM = int(os.environ.get("S21_NPERM", "1000"))
torch.set_num_threads(int(os.environ.get("S21_THREADS", "12")))
if SMOKE:
    N_EVAL = min(N_EVAL, 32)
DEV = "cpu"
assert DEV == "cpu"
assert torch.cuda.is_available() is False, "必须全程 CPU"

NBLK, BLKD = 8, 128 // 8                                  # 8 个 16 维块

JOBS_ALL = [
    dict(bucket="add_3d", seed=1234, em=37.25, dig=52.59),   # S19 res 臂锚点
    dict(bucket="add_3d", seed=5678, em=31.13, dig=49.63),
    dict(bucket="add_1d", seed=1234, em=96.88, dig=93.51),   # 可选对照：「信息已经很够用」
]
NCK = int(os.environ.get("S21_NCK", "2"))
JOBS = JOBS_ALL[:NCK]


def ckpt_path(job) -> str:
    return f"{ROOT}/logs/19_ckpt_{job['bucket']}_res_seed{job['seed']}.pt"


def job_key(job) -> str:
    return f"{job['bucket']}@res@{job['seed']}"


print(f"[S21] ===== 信息-因果曲线 | device={DEV} | "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} | "
      f"torch.cuda.is_available()={torch.cuda.is_available()} | threads={torch.get_num_threads()} | "
      f"N_EVAL={N_EVAL} N_R29={N_R29} NCK={NCK} smoke={SMOKE} =====", flush=True)
print(f"[S21] ckpt（全部现成、零训练）："
      + "；".join(f"{job_key(j)}={os.path.basename(ckpt_path(j))}" for j in JOBS), flush=True)
_s14 = open(f"{ROOT}/stages/14_synth_arith.py", encoding="utf-8").read()
print(f"[S21] 复用链：exec(17_probe_causal_n7.py 前 {_s17[:_c17].count(chr(10))} 行 → 其内 exec("
      f"14_synth_arith.py 前 {_s14[:_s14.index('# ③ 主循环')].count(chr(10))} 行) ⇒ 两个训练主循环都在切片外",
      flush=True)

# ============================================================================
# ③ 残差版 hooked logits（S14 Cards.logits + h=h+Think(h)）+ 输入侧干预钩子
# ============================================================================
HOOK_IN: dict = {"fn": None}


class _InPatch(nn.Module):
    """把干预打在思维卡**输入**侧：x ← fn(x) 后再进 inner。只包一层，不改任何权重。"""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x, src_mask=None, **kw):
        fn = HOOK_IN["fn"]
        if fn is not None:
            x = fn(x)
        return self.inner(x, src_mask=src_mask, **kw)


def _hooked_logits21(self, ids: torch.Tensor) -> torch.Tensor:
    """与 S19 res 臂 Cards.logits 逐行相同，只多一步：模因 h 可被 HOOK["fn"] 替换。"""
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
            h = self.thought(h, src_mask=m)
    HOOK["n_call"] += 1
    if HOOK["fn"] is not None:
        h = HOOK["fn"](h, h_in, ids)
    return self.head(h)


def build_model21(job):
    m = Cards(D, FF)
    sd = torch.load(ckpt_path(job), map_location="cpu", weights_only=True)
    m.load_state_dict(sd)
    m.eval()
    m.residual = True
    m.logits = types.MethodType(_hooked_logits21, m)
    m.thought = _InPatch(m.thought)
    return m


# ---------------- 干预定义 ----------------
def iv_end():
    """恒等（零干预自检用）。"""
    def f(x, *rest):
        return x
    return f


def iv_pos(i: int):
    """单点清零：只把第 i 个位置的模因向量清零（输入侧）。"""
    def f(x, *rest):
        if i >= x.size(1):
            return x
        x = x.clone()
        x[:, i, :] = 0.0
        return x
    return f


def iv_blk(b: int):
    """16 维块清零：一次清零第 b 块（输入侧，128 维里的 16 维）。"""
    lo, hi = b * BLKD, (b + 1) * BLKD

    def f(x, *rest):
        x = x.clone()
        x[:, :, lo:hi] = 0.0
        return x
    return f


def iv_all():
    """整体清零（阳性对照）：整个模因置零。"""
    def f(x, *rest):
        return torch.zeros_like(x)
    return f


def iv_swp_in(i: int):
    """把第 i 与 i+1 个位置对调（输入侧，对照口径）。"""
    def f(x, *rest):
        if i + 1 >= x.size(1):
            return x
        x = x.clone()
        a = x[:, i, :].clone()
        x[:, i, :] = x[:, i + 1, :]
        x[:, i + 1, :] = a
        return x
    return f


def iv_swp_out(i: int):
    """★字面版「互换两行模因」：在模因张量 h 上把第 i 与 i+1 行对调（输出侧）。"""
    def f(h, h_in, ids):
        if i + 1 >= h.size(1):
            return h
        h = h.clone()
        a = h[:, i, :].clone()
        h[:, i, :] = h[:, i + 1, :]
        h[:, i + 1, :] = a
        return h
    return f


def set_hooks(in_fn, out_fn):
    HOOK_IN["fn"], HOOK["fn"] = in_fn, out_fn


def run_ce(model, recs, hook, side: str):
    """CE（S17 eval_ce 逐字复用；S21 只决定 hook 挂哪一侧）。

    ★注意：S17 的 eval_ce 内部会 `HOOK["fn"] = hook` ⇒ 输出侧必须把 hook 作为参数传进去，
    不能只靠 set_hooks（会被它覆盖成 None）。
    """
    if side == "in":
        set_hooks(hook, None)
        try:
            return eval_ce(model, recs, None)
        finally:
            set_hooks(None, None)
    set_hooks(None, None)
    return eval_ce(model, recs, hook)


def run_em(model, recs, hook, side: str, batch: int = 1):
    """EM（贪心，主口径 batch=1，R28）。"""
    set_hooks(hook if side == "in" else None, hook if side == "out" else None)
    try:
        model.eval()
        texts = greedy_gen(model, recs, batch=batch)
    finally:
        set_hooks(None, None)
    return hits_from(texts, recs)


def nmean(xs):
    ys = [float(x) for x in xs if x == x]
    return sum(ys) / len(ys) if ys else float("nan")


def dacc_pp(base_dig, iv_dig) -> float:
    """Δ数字每步正确率（pp）= exp(−iv 数字CE) − exp(−base 数字CE)。"""
    return (math.exp(-nmean(iv_dig)) - math.exp(-nmean(base_dig))) * 100.0


def dem_pp(base_hits, iv_hits) -> float:
    """ΔEM（pp）= iv − base（配对）。"""
    db = [h["strict"] for h in base_hits]
    di = [h["strict"] for h in iv_hits]
    return (sum(di) - sum(db)) / len(db) * 100.0


# ============================================================================
# ④ 闭式局部特征（只保留第 i 位/第 b 块，其余置零 ⇒ featurize([末位;均值])）
# ============================================================================
def pad_rows(rows):
    N = len(rows)
    Lm = max(len(r) for r in rows)
    d = rows[0].size(1)
    P = torch.zeros(N, Lm, d)
    lens = torch.zeros(N, dtype=torch.long)
    for j, r in enumerate(rows):
        P[j, : len(r)] = r
        lens[j] = len(r)
    return P, lens


def _last(P, lens):
    return P[torch.arange(P.size(0)), lens - 1]


def feat_pos(P, lens, i):
    last = _last(P, lens) * ((lens - 1) == i).float().unsqueeze(1)
    return torch.cat([last, P[:, i, :] / lens.unsqueeze(1).float()], 1)


def feat_blk(P, lens, b):
    m = torch.zeros(P.size(2))
    m[b * BLKD:(b + 1) * BLKD] = 1.0
    last = _last(P, lens) * m
    mean = (P.sum(1) / lens.unsqueeze(1).float()) * m
    return torch.cat([last, mean], 1)


def feats(kind: str, idx: int, P, lens):
    return feat_pos(P, lens, idx) if kind == "pos" else feat_blk(P, lens, idx)


def ref_feat(rows, keep=None, blk=None):
    """参考实现：先把模因逐样本掩蔽，再逐字走 S17 featurize ⇒ 与闭式公式互检。"""
    out = []
    for r in rows:
        z = r.clone() if keep is None else torch.zeros_like(r)
        if keep is not None:
            for j in keep:
                if j < r.size(0):
                    z[j] = r[j]
        if blk is not None:
            m = torch.zeros(r.size(1))
            m[blk * BLKD:(blk + 1) * BLKD] = 1.0
            z = z * m
        out.append(torch.cat([z[-1], z.mean(0)]))
    return torch.stack(out)


def readability(kind, idx, P_tr, l_tr, P_te, l_te, d_tr, d_te, y_tr, y_te):
    """R_i：(位数分类准度 test, 答案值 R² test) —— 同一 [末位;均值] 线性探针。"""
    Xtr, Xte = feats(kind, idx, P_tr, l_tr), feats(kind, idx, P_te, l_te)
    a_tr, a_te = _probe(Xtr, d_tr, Xte, d_te, "cls", False, seed=0)
    r_tr, r_te = _probe(Xtr, y_tr, Xte, y_te, "reg", False, seed=0)
    return a_tr, a_te, r_tr, r_te


# ============================================================================
# ⑤ 曲线 / 判决 / 置换检验
# ============================================================================
def curve_stats(ce, R, qs=(0.2, 0.4, 0.6, 0.8, 1.0), tops=(0.1, 0.2, 0.5)):
    ce, R = np.asarray(ce, float), np.asarray(R, float)
    N = len(ce)
    o = np.argsort(-ce, kind="stable")
    cs, rs = ce[o], R[o]
    tc, tr = cs.sum(), rs.sum()
    cc = np.cumsum(cs) / tc if tc > 0 else np.zeros(N)
    cr = np.cumsum(rs) / tr if tr > 0 else np.zeros(N)
    qtab = []
    for q in qs:
        k = max(1, int(math.ceil(q * N)))
        qtab.append(dict(q=q, k=k, x=float(cr[k - 1]), y=float(cc[k - 1])))
    shares = {p: float(cc[max(1, int(round(p * N))) - 1]) for p in tops}
    dev = cc - cr
    return qtab, shares, dict(N=N, shares=shares, maxdev=float(np.max(np.abs(dev))),
                              meandev=float(np.mean(np.abs(dev))))


def verdict(st):
    s20 = st["shares"][0.2]
    if s20 >= 0.5:
        return "★信息高度集中（前20%%单元贡献 %.1f%% ≥50%% ⇒ 大量信息因果不可用）" % (s20 * 100)
    if st["maxdev"] < 0.10:
        return "★信息普遍可用（累积因果≈累积可读性，最大偏离 %.3f <0.10）" % st["maxdev"]
    return ("两者都不明显：前20%%单元贡献 %.1f%%(<50%%)，累积因果与累积可读性最大偏离 %.3f(≥0.10)"
            % (s20 * 100, st["maxdev"]))


def perm_test(R, CE, nperm=NPERM, seed=7):
    """轴内置换检验：把 CE 在单元间随机置换 nperm 次，看真实 ρ 的分位。"""
    from scipy.stats import spearmanr
    R, CE = np.asarray(R, float), np.asarray(CE, float)
    _r = spearmanr(R, CE)
    rho, p = float(_r.statistic), float(_r.pvalue)
    rng = np.random.default_rng(seed)
    rhos = np.empty(nperm)
    for i in range(nperm):
        rhos[i] = float(spearmanr(R, rng.permutation(CE)).statistic)
    pct = float((rhos < rho).mean())
    p2 = float((np.abs(rhos) >= abs(rho)).mean())
    return rho, p, pct, p2, float(np.percentile(rhos, 2.5)), float(np.percentile(rhos, 97.5))


# ============================================================================
# ⑥ 主流程：每个 ckpt × (n 位置 + 8 块) 个轴向单元
# ============================================================================
POINTS: list[dict] = []
CKSUM: dict[str, dict] = {}
JSONL = f"{ROOT}/logs/21_points.jsonl"
open(JSONL, "w").close()

for job in JOBS:
    key = job_key(job)
    t0 = time.time()
    model = build_model21(job)
    bucket = job["bucket"]
    tr, te = DATA[bucket]["train"], DATA[bucket]["test"]
    eval_recs = te[:N_EVAL]
    print(f"\n[S21] ===== ckpt={key} {os.path.basename(ckpt_path(job))} | bucket={bucket} "
          f"| eval N={len(eval_recs)} =====", flush=True)

    # ---- 模因素材（prompt 段，防答案泄漏）+ 位置数 n ----
    tr_rows = meme_rows(model, tr, "prompt")
    te_rows = meme_rows(model, te, "prompt")
    n_pos = min(min(len(r) for r in tr_rows), min(len(r) for r in te_rows))
    P_tr, len_tr = pad_rows(tr_rows)
    P_te, len_te = pad_rows(te_rows)
    d_tr = torch.tensor([len(str(int(r["y"]))) for r in tr])
    d_te = torch.tensor([len(str(int(r["y"]))) for r in te])
    y_tr = torch.tensor([float(r["y"]) for r in tr])
    y_te = torch.tensor([float(r["y"]) for r in te])
    maj = max(d_te.tolist().count(c) for c in set(d_te.tolist())) / len(te)
    print(f"[S21] {key} prompt 位置数 n={n_pos}（prompt 长 {min(len(r) for r in tr_rows)}.."
          f"{max(len(r) for r in tr_rows)}）| 单元数 = n + {NBLK} 块 = {n_pos + NBLK} | "
          f"多数类={maj*100:.1f}%", flush=True)

    # ---- 闭式特征互检 ----
    chk = []
    for kind, idx in (("pos", 0), ("pos", n_pos - 1), ("blk", 3)):
        keep = [idx] if kind == "pos" else None
        blk = idx if kind == "blk" else None
        a = feats(kind, idx, P_tr[:8], len_tr[:8])
        b = ref_feat(tr_rows[:8], keep, blk)
        chk.append(bool(torch.allclose(a, b, atol=1e-5)))
    print(f"[CHK] {key} 闭式局部特征 vs 掩蔽+featurize 8 样本互检 "
          f"{'3/3 一致 ⇒ 通过' if all(chk) else '★不一致 ' + str(chk)}", flush=True)

    # ---- 自检4：基线复测（全 test、batch=1，与 S19 res 臂同口径）----
    b_ce, b_dig = eval_ce(model, eval_recs, None)
    b_hits = run_em(model, eval_recs, None, "in")
    em_base, dig_base = acc_of(b_hits), math.exp(-nmean(b_dig))
    d_anchor = (em_base - job["em"] / 100.0) * 100
    full = len(eval_recs) == len(te)
    print(f"[SELF4] {key} 基线复测 n={len(eval_recs)}"
          f"{'（=全 test）' if full else '（子集，不对锚）'}：EM={em_base*100:.2f}% "
          f"（S19 res 锚 {job['em']:.2f}% 差={d_anchor:+.2f}pp "
          f"{'OK' if abs(d_anchor) <= 1.0 else '★差>1pp'}）| 数字每步={dig_base*100:.2f}% "
          f"（S19 锚 {job['dig']:.2f}%）", flush=True)

    # ---- 自检5：R28 batch=1 vs batch=16 ----
    r28 = r28_selfcheck(model, eval_recs[:16], f"{key} 基线", k=min(16, len(eval_recs)))
    print(f"[SELF5] {key} R28 batch=1 vs batch={CHK_BATCH} 逐字一致 "
          f"{r28['same']}/{r28['k']} ⇒ "
          f"{'一致' if r28['same'] == r28['k'] else '★不一致 ⇒ 仅 batch=1 口径可用（本单元全程 batch=1）'}",
          flush=True)

    # ---- 自检2：零干预恒等（Δ 必须精确为 0）----
    id_recs = eval_recs[:min(N_ID, len(eval_recs))]
    _, id_b_dig = eval_ce(model, id_recs, None)
    id_b_hits = run_em(model, id_recs, None, "in")
    _, id_i_dig = run_ce(model, id_recs, iv_end(), "in")
    id_i_hits = run_em(model, id_recs, iv_end(), "in")
    d_id_acc = dacc_pp(id_b_dig, id_i_dig)
    d_id_em = dem_pp(id_b_hits, id_i_hits)
    ok_id = (d_id_acc == 0.0) and (d_id_em == 0.0)
    print(f"[SELF2] {key} 零干预恒等（n={len(id_recs)}）：Δ数字每步={d_id_acc:.3e}pp "
          f"ΔEM={d_id_em:.3e}pp ⇒ {'精确为 0 ⇒ 通过' if ok_id else '★非 0 ⇒ 口径有误'}", flush=True)

    # ---- R29 非零性冒烟：各干预轴的 Δ 非零比例（本科目决定轴的去留）----
    sm_recs = eval_recs[:min(N_R29, len(eval_recs))]
    _, sm_b_dig = eval_ce(model, sm_recs, None)
    axes = [
        ("单点清零@in（主轴）", "in", [("pos", i) for i in range(n_pos)]),
        ("16维块清零@in（主轴）", "in", [("blk", b) for b in range(NBLK)]),
        ("整体清零@in（阳性对照）", "in", [("all", 0)]),
        ("互换两行@in（非字面）", "in", [("swp", i) for i in range(n_pos - 1)]),
        ("互换两行@out（字面版）", "out", [("swp", i) for i in range(n_pos - 1)]),
    ]
    R29 = []
    with torch.no_grad():
        ids0, s0 = build_batch([eval_recs[0]])
        lt0 = len(eval_recs[0]["t"])
        set_hooks(None, None)
        base_lg = model.logits(ids0)[0, int(s0[0]) - 1: int(s0[0]) - 1 + lt0].clone()

    def hook_of(kind, idx, side):
        if kind == "pos":
            return iv_pos(idx)
        if kind == "blk":
            return iv_blk(idx)
        if kind == "all":
            return iv_all()
        return iv_swp_in(idx) if side == "in" else iv_swp_out(idx)

    for nm, side, units in axes:
        dls, daccs = [], []
        for kind, idx in units:
            hk = hook_of(kind, idx, side)
            set_hooks(hk if side == "in" else None, hk if side == "out" else None)
            try:
                with torch.no_grad():
                    lg = model.logits(ids0)[0, int(s0[0]) - 1: int(s0[0]) - 1 + lt0]
            finally:
                set_hooks(None, None)
            dls.append(float((lg - base_lg).abs().max()))
            _, iv_dig = run_ce(model, sm_recs, hk, side)
            daccs.append(dacc_pp(sm_b_dig, iv_dig))
        if not daccs:
            continue
        nz = [abs(x) > 1e-3 for x in daccs]
        rec = dict(axis=nm, side=side, m=len(units), nz_ratio=sum(nz) / len(nz),
                   med=float(np.median(np.abs(daccs))), mx=float(np.max(np.abs(daccs))),
                   dls_med=float(np.median(dls)), dls_mx=float(np.max(dls)))
        rec["valid"] = (rec["nz_ratio"] >= 0.20) and (rec["med"] >= 0.01)
        R29.append(rec)
        print(f"[R29] {key} {nm:<22} 单元={rec['m']:>2} |Δ数字|中位={rec['med']:8.4f}pp "
              f"最大={rec['mx']:8.4f}pp | 非零比(>1e-3pp, n={len(sm_recs)})={rec['nz_ratio']*100:5.1f}% "
              f"| 读列 max|Δlogits| 中位={rec['dls_med']:.3e} 最大={rec['dls_mx']:.3e} ⇒ "
              f"{'有效轴' if rec['valid'] else '★无效轴 ⇒ 作废并剔除（不能当因果效应=0 用）'}", flush=True)

    # ---- 自检3：阳性对照（随机选一个位置 vs 整体清零）----
    rng = np.random.default_rng(20261008)
    rnd_i = int(rng.integers(0, n_pos))
    _, r_dig = run_ce(model, eval_recs, iv_pos(rnd_i), "in")
    r_hits = run_em(model, eval_recs, iv_pos(rnd_i), "in")
    _, a_dig = run_ce(model, eval_recs, iv_all(), "in")
    a_hits = run_em(model, eval_recs, iv_all(), "in")
    p_rnd, p_all = dacc_pp(b_dig, r_dig), dacc_pp(b_dig, a_dig)
    e_rnd, e_all = dem_pp(b_hits, r_hits), dem_pp(b_hits, a_hits)
    print(f"[SELF3] {key} 阳性对照：随机位置 pos{rnd_i} Δ数字={p_rnd:+.2f}pp ΔEM={e_rnd:+.2f}pp "
          f"(EM {em_base*100:.1f}%→{acc_of(r_hits)*100:.1f}%) | 整体清零 Δ数字={p_all:+.2f}pp "
          f"ΔEM={e_all:+.2f}pp (EM {em_base*100:.1f}%→{acc_of(a_hits)*100:.1f}%) ⇒ "
          f"{'整体掉到底 ⇒ 干预链有效' if acc_of(a_hits) < 0.5 * em_base else '★整体未掉到底'}",
          flush=True)

    # ---- 主轴：逐单元（可读性 + 因果效应）----
    units = [("pos", i) for i in range(n_pos)] + [("blk", b) for b in range(NBLK)]
    print(f"[UNIT] {key} 主轴单元 = pos(0..{n_pos-1}) + blk(0..{NBLK-1}) = {len(units)} 个；"
          f"每单元 = 1 个局部探针(位数+回归) + 干预[Δ数字/ΔEM](batch=1, n={len(eval_recs)})", flush=True)
    for u_i, (kind, idx) in enumerate(units):
        un = f"pos{idx}" if kind == "pos" else f"blk{idx}[{idx*BLKD}:{(idx+1)*BLKD}]"
        a_tr, a_te, r_tr, r_te = readability(kind, idx, P_tr, len_tr, P_te, len_te,
                                             d_tr, d_te, y_tr, y_te)
        hk = hook_of(kind, idx, "in")
        _, iv_dig = run_ce(model, eval_recs, hk, "in")
        iv_hits = run_em(model, eval_recs, hk, "in")
        dacc = dacc_pp(b_dig, iv_dig)
        dem = dem_pp(b_hits, iv_hits)
        pt = dict(ckpt=key, kind=kind, idx=idx, unit=un, n_pos=n_pos,
                  r_cls_tr=a_tr, r_cls=a_te, r2_tr=r_tr, r2=r_te,
                  dacc=dacc, dem=dem, ce=abs(dacc), ce_em=abs(dem))
        POINTS.append(pt)
        with open(JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(pt) + "\n")
        print(f"[PTS] {key} {un:<18} R_cls={a_te*100:5.1f}%(train {a_tr*100:.1f}) R²={r_te:+.3f} "
              f"| Δ数字={dacc:+8.3f}pp ΔEM={dem:+7.2f}pp |CE|={abs(dacc):7.3f}pp", flush=True)
        if (u_i + 1) % 10 == 0:
            print(f"[PROG] {key} {u_i+1}/{len(units)} 用时 {(time.time()-t0)/60:.1f} min", flush=True)

    pts = [p for p in POINTS if p["ckpt"] == key]
    nz_u = sum(1 for p in pts if abs(p["dacc"]) > 1e-3)
    print(f"[CKPT] {key} 完成：单元 {len(pts)} | Δ数字非零(>1e-3pp) {nz_u}/{len(pts)} | "
          f"R_cls min/中位/max={min(p['r_cls'] for p in pts)*100:.1f}/"
          f"{sorted(p['r_cls'] for p in pts)[len(pts)//2]*100:.1f}/"
          f"{max(p['r_cls'] for p in pts)*100:.1f}% | 墙钟 {(time.time()-t0)/60:.1f} min", flush=True)

    # ---- 曲线（本 ckpt）----
    ce = [p["ce"] for p in pts]
    R = [p["r_cls"] for p in pts]
    R2 = [p["r2"] for p in pts]
    qtab, shares, st = curve_stats(ce, R)
    qtab2, shares2, st2 = curve_stats(ce, R2)
    rho, pv, pct, p2, lo, hi = perm_test(R, ce)
    rho2, pv2, pct2, p22, lo2, hi2 = perm_test(R2, ce)
    mdev = st["maxdev"]
    CKSUM[key] = dict(job=job, n_pos=n_pos, em=em_base, dig=dig_base, d_anchor=d_anchor,
                      r28=r28["same"], id_ok=ok_id, R29=R29, pts=pts,
                      qtab=qtab, shares=shares, st=st, qtab2=qtab2, shares2=shares2, st2=st2,
                      rho=rho, p=pv, pct=pct, p2=p2, lo=lo, hi=hi,
                      rho2=rho2, p2v=pv2, pct2=pct2, p22=p22,
                      pos_rnd=rnd_i, p_rnd=p_rnd, p_all=p_all, e_rnd=e_rnd, e_all=e_all,
                      em_all=acc_of(a_hits), wall=time.time() - t0)
    print(f"\n[CURVE] {key} 按 CE 降序（N={st['N']} 单元）累积可读性 → 累积因果（Δ数字口径）：", flush=True)
    for q in qtab:
        print(f"[CURVE] {key} 前 {q['q']*100:3.0f}%（{q['k']:>2} 个单元）：累积可读性="
              f"{q['x']*100:5.1f}% → 累积因果={q['y']*100:5.1f}%", flush=True)
    print(f"[CURVE] {key} 前 10/20/50% 单元贡献因果 = {shares[0.1]*100:.1f}% / {shares[0.2]*100:.1f}% / "
          f"{shares[0.5]*100:.1f}% | 最大偏离对角线 {mdev:.3f}（均值 {st['meandev']:.3f}）", flush=True)
    print(f"[CURVE] {key} 同一曲线换 x=答案值R²：前 10/20/50% = {shares2[0.1]*100:.1f}% / "
          f"{shares2[0.2]*100:.1f}% / {shares2[0.5]*100:.1f}% | 最大偏离 {st2['maxdev']:.3f}", flush=True)
    print(f"[VERDICT-S21] {key} {verdict(st)}", flush=True)
    print(f"[RHO] {key} ρ(R_cls, |Δ数字|)={rho:.3f} (n={len(pts)}, p={pv:.4f}) | 轴内置换 1000 次："
          f"真实 ρ 分位={pct*100:.1f}%，双侧 p_perm={p2:.4f}，置换 95% 区间=[{lo:.3f}, {hi:.3f}]", flush=True)
    print(f"[RHO] {key} ρ(R², |Δ数字|)={rho2:.3f} (n={len(pts)}, p={pv2:.4f}) | 置换分位="
          f"{pct2*100:.1f}% 双侧 p_perm={p22:.4f}", flush=True)
    del model

# ============================================================================
# ⑦ 合并曲线（把各 ckpt 的单元汇成一池）+ 总判决
# ============================================================================
pool_ce = [p["ce"] for p in POINTS]
pool_R = [p["r_cls"] for p in POINTS]
pool_R2 = [p["r2"] for p in POINTS]
pq, psh, pst = curve_stats(pool_ce, pool_R)
pq2, psh2, pst2 = curve_stats(pool_ce, pool_R2)
prho, ppv, ppct, pp2, plo, phi = perm_test(pool_R, pool_ce)
prho2, ppv2, ppct2, pp22, _, _ = perm_test(pool_R2, pool_ce)
print(f"\n[CURVE-POOL] 合并 {pst['N']} 个单元（{len(CKSUM)} ckpt）按 CE 降序：", flush=True)
for q in pq:
    print(f"[CURVE-POOL] 前 {q['q']*100:3.0f}%（{q['k']:>2} 个）：累积可读性={q['x']*100:5.1f}% → "
          f"累积因果={q['y']*100:5.1f}%", flush=True)
print(f"[CURVE-POOL] 前 10/20/50% 单元贡献因果 = {psh[0.1]*100:.1f}% / {psh[0.2]*100:.1f}% / "
      f"{psh[0.5]*100:.1f}% | 最大偏离对角线 {pst['maxdev']:.3f}（均值 {pst['meandev']:.3f}）", flush=True)
print(f"[VERDICT-S21-POOL] {verdict(pst)}", flush=True)
print(f"[RHO-POOL] ρ(R_cls, |Δ数字|)={prho:.3f} (n={pst['N']}, p={ppv:.4f}) | 置换分位={ppct*100:.1f}% "
      f"双侧 p_perm={pp2:.4f} 95%区间=[{plo:.3f},{phi:.3f}] | ρ(R², |Δ数字|)={prho2:.3f} "
      f"(p={ppv2:.4f}, 分位={ppct2*100:.1f}%, p_perm={pp22:.4f})", flush=True)

# ---- 直观三例（逐 ckpt 各给一组，报告里取第一个 ckpt）----
for key, S in CKSUM.items():
    pts = S["pts"]
    hi_ce = max(pts, key=lambda p: p["ce"])
    lo_ce = min(pts, key=lambda p: p["ce"])
    order = sorted(pts, key=lambda p: p["ce"])
    lo_third = order[: max(1, len(order) // 3)]
    hr_lc = max(lo_third, key=lambda p: p["r_cls"])
    print(f"[EX] {key} 最高CE={hi_ce['unit']}(R_cls={hi_ce['r_cls']*100:.1f}%, CE={hi_ce['ce']:.3f}pp, "
          f"ΔEM={hi_ce['dem']:+.2f}pp) | 最低CE={lo_ce['unit']}(R_cls={lo_ce['r_cls']*100:.1f}%, "
          f"CE={lo_ce['ce']:.3f}pp) | 高可读低因果={hr_lc['unit']}"
          f"(R_cls={hr_lc['r_cls']*100:.1f}%, CE={hr_lc['ce']:.3f}pp, ΔEM={hr_lc['dem']:+.2f}pp)", flush=True)

# ---- 符号/量纲元信息 ----
sign = sum(p["dacc"] for p in POINTS)
print(f"\n[META] CE 用 |Δ|；Δ数字=iv−base（>0 ⇒ 清零反而变好）。ΣΔ数字={sign:+.1f}pp "
      f"(n={len(POINTS)} 单元) | |Δ数字| 中位={np.median([p['ce'] for p in POINTS]):.3f}pp "
      f"最大={max(p['ce'] for p in POINTS):.3f}pp | |ΔEM| 中位="
      f"{np.median([p['ce_em'] for p in POINTS]):.3f}pp 最大={max(p['ce_em'] for p in POINTS):.3f}pp",
      flush=True)
print(f"[META] device={DEV} CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} "
      f"cuda_is_available={torch.cuda.is_available()} threads={torch.get_num_threads()} "
      f"N_EVAL={N_EVAL} smoke={SMOKE} NCK={NCK} 墙钟={(time.time()-T0)/60:.1f}min", flush=True)
print("[DONE] exit=0", flush=True)
