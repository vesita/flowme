#!/usr/bin/env python3
"""S18 · 用【非破坏性干预】重测 P-3（★全程 CPU · ★零训练 · 不碰任何源文件）。

只回答一个问题：
  在【同一基线水平内】（= 同一个 ckpt 内部）做细粒度、非破坏性干预时，
  「探针能读出多少（局部可读性）」与「干预造成的因果损失 |Δ数字每步正确率|」是否弱相关？

为什么要重做（直接用 S16/S17 的实测）：
  S16 用破坏性干预（模因清零/乱序/换样本）得 ρ=0.283(n=10) ⇒ 判「确证 P-3」；
  S17 扩到 7 ckpt(n=21) 后 ρ=0.743 ⇒ 判「不确定」，并给出机械原因：
  破坏性干预下 |Δ| ≈ 基线成绩（差 ≤0.01pp）⇒ 那个 ρ 测的是「探针准度 vs 模型好坏」，
  不是「读出 vs 会用」。⇒ 本单元只改模因的一小部分，让它掉一点但仍可测。

★本单元踩到并实测确认的一条架构事实（决定了干预打在哪一侧）：
  模因 = 思维卡（thought 层）的**输出**，其后只有**逐位置线性 head**，没有任何跨位置混合
  ⇒ 在输出侧把第 i 列清零，**只改第 i 列的 logits**；而 CE/生成只读「被读列」
  （= 最后一个 prompt 位 s-1 及之后的 target 位）⇒ 只要 i 不是被读列，**Δ 恒为 0（bit 级）**。
  冒烟实测（7 ckpt × 39 点 = 269 点）：Δ数字非零 54/269、ΔEM 非零 10/269 —— 几乎全来自
  「维度块」（它一次改全部 128 列 ⇒ 被读列也被改）；逐位置/互换几乎恒 0 ⇒ 字面版是空实验。
  ⇒ 主臂把清零点移到**思维卡的输入侧**（`S18_SIDE=in`，默认）：仍「只动一处、非破坏性、同基线」，
  但改动会经思维卡的因果注意力传播到被读列 ⇒ Δ 非零且随位置分层；
  字面版（输出侧）作为**对照臂**用 `S18_SIDE=out` 单独跑，用来把上面这条机械事实钉死。

局部干预（同一 ckpt 内做 ⇒ 天然控制住基线；每个 ckpt 共 n + 8 + (n-1) 个点）：
  ① 逐位置清零：只把第 i 个位置的向量清零（其余不动），i = 0..n-1；
  ② 逐维度块清零：d=128 切 8 个 16 维块，一次清零一块 ⇒ 8 个点；
  ③ 互换：把第 i 与 i+1 个位置的向量对调，i = 0..n-2 ⇒ n-1 个点；
  其中 n = 模因的位置数 = train/test prompt 段的最短长度（保证该位置在每条样本里都真实存在）。
探针 x（读出）：对同一个 ckpt，用「只保留被干预的那个位置/块的模因（其余置零）」走**同一个**
  featurize([末位; 均值] → 256 维) → **同一个**线性探针（4000 训 / 800 测，位数分类 test 准度）
  ⇒ 得到「该局部的可读性」，这才是与「局部因果重要性」配对的 x。
判决量：ρ(局部可读性, |局部 Δ数字每步正确率|)（全部 ckpt 合并，n = Σ(2n_k+7)）；
  并列口径：ρ(局部可读性, |局部 ΔEM|)。
判据（写死）：ρ < 0.5 且 n ≥ 30 ⇒ 确证 P-3；ρ ≥ 0.8 ⇒ 推翻；中间 ⇒ 不确定。

复用方式（源文件一个字节都不改）：
  把 `stages/17_probe_causal_n7.py` 的前半段（到 `# ⑥ 主流程` 之前 = device 门禁 + ckpt 锚点
  + 干预 hook + 线性探针 _probe + patching 评估 eval_ce/eval_em/cond_report + spearman）
  exec 进本命名空间逐字复用；S17 的那一段内部又把 `stages/14_synth_arith.py` 的前半段
  （到 `# ③ 主循环` 之前 = tokenizer + 数据生成器 + Cards 模型 + build_batch/greedy_gen/hits_from）
  exec 进来 ⇒ **不重写模型、零训练**（两个训练主循环永远不在被 exec 的切片里）。
  S18 自己新增的只有：干预钩子、局部特征、以及 3 个 eval 包装（决定 hook 挂哪一侧）。

只允许写：本文件、logs/、/tmp。
"""
from __future__ import annotations

import math
import os
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""      # ★先于 torch 导入：强制关 GPU（S15 占着，绝不碰）
if os.environ.get("S18_SMOKE") == "1":        # 冒烟：S14 的数据切分缩到 train300/test60（必须在 exec 前设）
    os.environ.setdefault("S14_SMOKE", "1")
    os.environ.setdefault("S17_SMOKE", "1")

ROOT = "/home/vesita/coding/my/flowme"
SRC17 = f"{ROOT}/stages/17_probe_causal_n7.py"

# ============================================================================
# ① 逐字复用 S17 前半段（其内部已 exec S14 前半段 ⇒ 模型/数据/探针/patching 全部现成）
# ============================================================================
_s17 = open(SRC17, encoding="utf-8").read()
_c17 = _s17.index("# ⑥ 主流程")
NS17 = {"__name__": "s17_head", "__file__": SRC17}
exec(compile(_s17[:_c17], SRC17, "exec"), NS17)
for _k, _v in NS17.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
r28_selfcheck = NS["r28_selfcheck"]        # S17 没把它提出来，它在 S14 命名空间里（R28 批内一致性自检）

# ============================================================================
# ② S18 自己的配置（在 NS17 名字回填**之后**设置，避免被 S17 的同名变量覆盖）
# ============================================================================
SMOKE = os.environ.get("S18_SMOKE") == "1"
N_EVAL = int(os.environ.get("S18_N", "800"))
EM_BATCH = int(os.environ.get("S18_EMB", "16"))     # 贪心批（R28 自检不通过就退回 1）
MAX_CK = int(os.environ.get("S18_MAXCK", "7"))
N_UNIT_LIM = int(os.environ.get("S18_NUNIT", "0"))  # >0 ⇒ 每 ckpt 只跑前 N 个点（计时/对照臂用）
SIDE = os.environ.get("S18_SIDE", "in")             # in=思维卡输入侧（主臂）/ out=模因输出侧（字面对照臂）
assert SIDE in ("in", "out"), SIDE
if SMOKE:
    N_EVAL = min(N_EVAL, 16)
T0 = time.time()

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

DEV = globals().get("DEV", "cpu")
assert DEV == "cpu", DEV
assert torch.cuda.is_available() is False, "必须全程 CPU"
CKS = CKS[:MAX_CK]
NBLK, BLKD = 8, 128 // 8          # 8 个 16 维块
print(f"\n[S18] ===== 非破坏性干预重测 P-3 | device={DEV} | "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} | "
      f"torch.cuda.is_available()={torch.cuda.is_available()} | "
      f"threads={torch.get_num_threads()} | N_EVAL={N_EVAL} EM_batch={EM_BATCH} "
      f"side={SIDE} smoke={SMOKE} ckpt数={len(CKS)} =====", flush=True)
_s14 = open(f"{ROOT}/stages/14_synth_arith.py", encoding="utf-8").read()
_n14 = _s14[:_s14.index("# ③ 主循环")].count("\n")
print(f"[S18] 复用链：exec(stages/17_probe_causal_n7.py 前 {_s17[:_c17].count(chr(10))} 行，"
      f"到「# ⑥ 主流程」之前）→ 其内部 exec(stages/14_synth_arith.py 前 {_n14} 行，到「# ③ 主循环」之前) "
      f"⇒ 两个训练主循环都在切片之外 = 零训练、源文件零改动", flush=True)
print(f"[S18] 干预侧别 side={SIDE}："
      + ("思维卡**输入**侧（单点清零后经思维卡注意力传播到被读列 ⇒ Δ 非零；主臂）"
         if SIDE == "in" else
         "模因**输出**侧（字面版；head 逐位置 ⇒ 只改被读列的 Δ，其余恒 0 ⇒ 对照臂）"), flush=True)

# ============================================================================
# ③ 干预钩子：只动模因的一小部分（一个位置 / 一个维度块 / 一对相邻位置）
#    同一组函数两种挂法：SIDE=in 挂在思维卡输入上（x ← f(x) 再进思维卡）；
#    SIDE=out 挂在模因输出上（由 S17 的 _hooked_logits 调用 f(h, h_in, ids)）。
#    故 f 写成 f(x, *rest) 两种调用都吃得下。
# ============================================================================
HOOK_IN: dict = {"fn": None}


class _InPatch(nn.Module):
    """把干预打在思维卡（模因层）的**输入**侧：x ← fn(x) 后再进 inner。只包一层，不改任何权重。"""

    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x, src_mask=None, **kw):
        fn = HOOK_IN["fn"]
        if fn is not None:
            x = fn(x)
        return self.inner(x, src_mask=src_mask, **kw)


def build_model18(key: str):
    m = build_model(key)                     # S17 逐字复用：load_state_dict + eval + 挂 _hooked_logits
    if SIDE == "in":
        m.thought = _InPatch(m.thought)
    return m


def iv_pos(i: int):
    def f(x, *rest):
        if i >= x.size(1):
            return x
        x = x.clone()
        x[:, i, :] = 0.0
        return x
    return f


def iv_blk(b: int):
    lo, hi = b * BLKD, (b + 1) * BLKD

    def f(x, *rest):
        x = x.clone()
        x[:, :, lo:hi] = 0.0
        return x
    return f


def iv_swp(i: int):
    def f(x, *rest):
        if i + 1 >= x.size(1):
            return x
        x = x.clone()
        a = x[:, i, :].clone()
        x[:, i, :] = x[:, i + 1, :]
        x[:, i + 1, :] = a
        return x
    return f


def hook_of(kind: str, idx: int):
    return iv_pos(idx) if kind == "pos" else (iv_blk(idx) if kind == "blk" else iv_swp(idx))


def unit_name(kind: str, idx: int) -> str:
    if kind == "pos":
        return f"pos{idx}"
    if kind == "blk":
        return f"blk{idx}[{idx*BLKD}:{(idx+1)*BLKD}]"
    return f"swp{idx}-{idx+1}"


def _arm(fn):
    """把 fn 挂到当前侧别，并返回一个复位函数。"""
    if SIDE == "in":
        HOOK_IN["fn"], HOOK["fn"] = fn, None
    else:
        HOOK_IN["fn"], HOOK["fn"] = None, fn
    HOOK["cur"] = 0

    def off():
        HOOK_IN["fn"] = None
        HOOK["fn"] = None
    return off


def run_ce(model, recs, hook):
    """CE（S17 eval_ce 逐字复用；S18 只决定 hook 挂哪一侧）。"""
    if SIDE == "in":
        HOOK_IN["fn"] = hook
        try:
            return eval_ce(model, recs, None)
        finally:
            HOOK_IN["fn"] = None
    return eval_ce(model, recs, hook)


def run_em(model, recs, hook, batch=None):
    """EM（贪心仍走 greedy_gen / S17 口径，只换批与挂点）。"""
    batch = EM_BATCH if batch is None else batch
    off = _arm(hook)
    try:
        model.eval()
        texts = greedy_gen(model, recs, batch=batch)
    finally:
        off()
    return hits_from(texts, recs)


# ============================================================================
# ④ 局部可读性：对「只保留该位置/块的模因（其余置零）」走同一个 featurize + 同一个线性探针
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


def feat_swp(P, lens, i):
    keep = ((lens - 1) == i) | ((lens - 1) == i + 1)
    last = _last(P, lens) * keep.float().unsqueeze(1)
    mean = (P[:, i, :] + P[:, i + 1, :]) / lens.unsqueeze(1).float()
    return torch.cat([last, mean], 1)


def feats(kind: str, idx: int, P, lens):
    if kind == "pos":
        return feat_pos(P, lens, idx)
    if kind == "blk":
        return feat_blk(P, lens, idx)
    return feat_swp(P, lens, idx)


def ref_feat(rows, keep=None, blk=None):
    """参考实现：先把模因张量逐样本掩蔽（keep=None 表示位置全保留），再**逐字**走 S17 的
    featurize ⇒ 与上面的闭式公式互检（两者必须逐位相等）。"""
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


def spearman_r(x, y):
    r = spearman(x, y)
    return float(r[0]), float(r[1])


def verdict_of(rho, n, tag):
    if n < 30:
        return f"{tag}: n={n}<30 ⇒ 按判据无法判决"
    if rho < 0.5:
        return f"{tag}: ρ={rho:.3f}<0.5 且 n={n} ⇒ 弱相关 ⇒ **确证 P-3（读出≠会用）**"
    if rho >= 0.8:
        return f"{tag}: ρ={rho:.3f}≥0.8 ⇒ 强相关 ⇒ **推翻 P-3**"
    return f"{tag}: ρ={rho:.3f} 落在 [0.5,0.8) 中间区间 ⇒ **不确定**"


# ============================================================================
# ⑤ 主流程：7 个 ckpt × (n 位置 + 8 维度块 + (n-1) 互换) 个局部点
# ============================================================================
POINTS: list[dict] = []
EM_FALLBACK = False

for key in CKS:
    t0 = time.time()
    model = build_model18(key)
    bucket = key.split("@")[0]
    tr, te = DATA[bucket]["train"], DATA[bucket]["test"]
    eval_recs = te[:N_EVAL]

    # ---- 探针素材：prompt 段模因（防答案泄漏）----
    tr_rows = meme_rows(model, tr, "prompt")
    te_rows = meme_rows(model, te, "prompt")
    n_pos = min(min(len(r) for r in tr_rows), min(len(r) for r in te_rows))
    lm = min(min(len(r) for r in tr_rows), min(len(r) for r in te_rows))
    lx = max(max(len(r) for r in tr_rows), max(len(r) for r in te_rows))
    P_tr, len_tr = pad_rows(tr_rows)
    P_te, len_te = pad_rows(te_rows)
    d_tr = torch.tensor([len(str(int(r["y"]))) for r in tr])
    d_te = torch.tensor([len(str(int(r["y"]))) for r in te])
    maj = max(d_te.tolist().count(c) for c in set(d_te.tolist())) / len(te)
    print(f"\n[S18] ===== ckpt={key} {os.path.basename(CKPT[key])} | bucket={bucket} "
          f"prompt位置数 n={n_pos}（prompt长 {lm}..{lx}）| eval N={len(eval_recs)} =====", flush=True)

    # ---- 自检①：闭式局部特征 == 「掩蔽模因 → 逐字 featurize」----
    chk = []
    for kind, idx in (("pos", 0), ("pos", n_pos - 1), ("blk", 3), ("swp", 0)):
        keep = [idx] if kind == "pos" else ([idx, idx + 1] if kind == "swp" else None)
        blk = idx if kind == "blk" else None
        a = feats(kind, idx, P_tr[:8], len_tr[:8])
        b = ref_feat(tr_rows[:8], keep, blk)
        chk.append(bool(torch.allclose(a, b, atol=1e-5)))
    print(f"[CHK] {key} 局部特征闭式实现 vs 掩蔽+featurize 参考实现 8 样本互检 "
          f"{'4/4 一致 ⇒ 通过' if all(chk) else '★不一致 ' + str(chk)}", flush=True)

    # ---- R28：干预下 batch=1 vs batch={EM_BATCH} 贪心逐字一致（不一致 ⇒ EM 退回 batch=1）----
    if EM_BATCH != 1 and not EM_FALLBACK:
        off = _arm(None)
        r28_selfcheck(model, eval_recs[:16], f"{key} 基线", k=min(16, len(eval_recs)))
        off()
        off = _arm(iv_pos(0))
        r28_selfcheck(model, eval_recs[:16], f"{key} pos0干预", k=min(16, len(eval_recs)))
        off()

    # ---- 基线复测：① 全 test、batch=1（与 S14/S16/S17 锚完全同口径）
    b_hits_full = eval_em(model, te, None)
    em_full = acc_of(b_hits_full)
    d_anchor = (em_full - ANCHOR[key] / 100.0) * 100
    # ---- ② 与干预配对的子集基线（同批同侧）----
    b_ce, b_dig = eval_ce(model, eval_recs, None)
    b_hits = run_em(model, eval_recs, None)
    em_sub = acc_of(b_hits)
    slice1 = [h["strict"] for h in b_hits_full if h["idx"] < len(eval_recs)]
    em_sub_b1 = sum(slice1) / len(slice1)
    batch_ok = abs(em_sub - em_sub_b1) < 1e-9
    print(f"[BASE] {key} 基线EM(全test n={len(te)}, batch=1)={em_full*100:.2f}%"
          f"（S14/S16/S17 锚 {ANCHOR[key]:.2f}% 差={d_anchor:+.2f}pp "
          f"{'OK' if abs(d_anchor) <= 1.0 else '⚠差>1pp'}） | 子集基线(同配对) EM="
          f"{em_sub*100:.2f}% 数字每步={math.exp(-sum(b_dig)/len(b_dig))*100:.2f}%", flush=True)
    print(f"[CHK] {key} 子集基线 EM：batch={EM_BATCH}={em_sub*100:.2f}% vs batch=1={em_sub_b1*100:.2f}% "
          f"⇒ {'一致（R28 成立）' if batch_ok else '★不一致 ⇒ EM 全部退回 batch=1'}", flush=True)
    if not batch_ok and EM_BATCH != 1:
        EM_FALLBACK = True
        EM_BATCH = 1
        b_hits = run_em(model, eval_recs, None)
        em_sub = acc_of(b_hits)
        print(f"[CHK] ★已回退 EM_BATCH=1，子集基线 EM={em_sub*100:.2f}%", flush=True)

    # ---- 点列表 ----
    units = [("pos", i) for i in range(n_pos)] + \
            [("blk", b) for b in range(NBLK)] + \
            [("swp", i) for i in range(n_pos - 1)]
    n_all = len(units)
    if N_UNIT_LIM > 0:
        units = units[:N_UNIT_LIM]
    print(f"[UNIT] {key} 局部点数 = n({n_pos}) + 8 + (n-1)({n_pos-1}) = {n_all}，"
          f"本次跑 {len(units)} 个", flush=True)

    # ---- 自检②：每个干预确实改变 logits（防空测试）----
    with torch.no_grad():
        ids0, _ = build_batch([eval_recs[0]])
        HOOK_IN["fn"], HOOK["fn"] = None, None
        base_l = model.logits(ids0)
        dl = {}
        for kind, idx in units:
            off = _arm(hook_of(kind, idx))
            dl[(kind, idx)] = float((model.logits(ids0) - base_l).abs().max())
            off()
    zero = [unit_name(k, i) for k, i in units if dl[(k, i)] <= 0.0]
    print(f"[CHK] {key} 干预自检 max|Δlogits|：min={min(dl.values()):.5f} "
          f"max={max(dl.values()):.5f} | 零效应点 {len(zero)}/{len(units)}"
          f"{('：' + ','.join(zero)) if zero else '（无，全部非空测试）'}", flush=True)

    # ---- 逐点：局部探针（x） + 局部因果（y）----
    h1 = max(1, len(b_dig) // 2)          # 拆半：前 h1 条 / 后 h2 条（不相交），查 Δ 量表自身可靠性
    h2 = len(b_dig) - h1
    for u_i, (kind, idx) in enumerate(units):
        un = unit_name(kind, idx)
        Xtr = feats(kind, idx, P_tr, len_tr)
        Xte = feats(kind, idx, P_te, len_te)
        a_tr, a_te = _probe(Xtr, d_tr, Xte, d_te, "cls", False)
        iv = hook_of(kind, idx)
        iv_ce, iv_dig = run_ce(model, eval_recs, iv)
        iv_hits = run_em(model, eval_recs, iv)
        rep = cond_report(f"{key} {un}", b_hits, iv_hits, b_dig, iv_dig)
        d_h1 = (math.exp(-sum(iv_dig[:h1]) / h1) - math.exp(-sum(b_dig[:h1]) / h1)) * 100
        d_h2 = (math.exp(-sum(iv_dig[h1:]) / h2) - math.exp(-sum(b_dig[h1:]) / h2)) * 100
        POINTS.append(dict(ckpt=key, kind=kind, idx=idx, unit=un, acc_tr=a_tr, acc_te=a_te,
                           dem=rep["dem"] * 100, dacc=rep["dacc"] * 100,
                           d_h1=d_h1, d_h2=d_h2, dl=dl[(kind, idx)],
                           em_b=rep["em_b"], em_i=rep["em_i"]))
        print(f"[PTS] {key} {un:<18} 可读性 train={a_tr*100:5.1f}% test={a_te*100:5.1f}% "
              f"|Δ数字|={abs(rep['dacc'])*100:6.3f}pp |ΔEM|={abs(rep['dem'])*100:6.2f}pp "
              f"max|Δlogits|={dl[(kind, idx)]:.4f}", flush=True)
        if (u_i + 1) % 20 == 0:
            print(f"[PROG] {key} {u_i+1}/{len(units)} 用时 {(time.time()-t0)/60:.1f} min", flush=True)

    accs = [p["acc_te"] for p in POINTS if p["ckpt"] == key]
    nz = sum(1 for p in POINTS if p["ckpt"] == key and abs(p["dacc"]) > 1e-9)
    print(f"[CKPT] {key} 完成：局部点 {len(units)} 个 | Δ数字非零 {nz}/{len(units)} | 局部可读性 test "
          f"min={min(accs)*100:.1f}% 中位={sorted(accs)[len(accs)//2]*100:.1f}% "
          f"max={max(accs)*100:.1f}%（多数类={maj*100:.1f}%）| 墙钟 {(time.time()-t0)/60:.1f} min",
          flush=True)
    del model
    if N_UNIT_LIM > 0:
        break

# ============================================================================
# ⑥ ★主判决：ρ(局部可读性, |局部 Δ数字每步|) —— 全部 ckpt 合并
# ============================================================================
n_all = len(POINTS)
xs = [p["acc_te"] for p in POINTS]
y_d = [abs(p["dacc"]) for p in POINTS]        # 单位已经是 pp
y_e = [abs(p["dem"]) for p in POINTS]         # 单位已经是 pp
rho_d, p_d = spearman_r(xs, y_d)
rho_e, p_e = spearman_r(xs, y_e)
per_ck = sorted({p["ckpt"] for p in POINTS})
rho_ck = []
for k in per_ck:
    sub = [p for p in POINTS if p["ckpt"] == k]
    r, pv = spearman_r([q["acc_te"] for q in sub], [abs(q["dacc"]) for q in sub])
    rho_ck.append(r)
    print(f"[RHO-CK] {k:<13} ρ(局部可读性,|Δ数字|)={r:.3f} (n={len(sub)}, p={pv:.4f})", flush=True)
mean_ck = sum(rho_ck) / len(rho_ck)
# y 可靠性（拆半）：同一 Δ 量表在不相交两半 eval 样本上的秩相关
rel, rel_p = spearman_r([abs(p["d_h1"]) for p in POINTS], [abs(p["d_h2"]) for p in POINTS])
x_lv = len(set(round(v, 6) for v in xs))
nz_d = sum(1 for v in y_d if v > 1e-9)

print(f"\n[RHO] ★主判决 ρ(局部可读性, |局部Δ数字每步|) = {rho_d:.3f} "
      f"(n={n_all} = Σ(2n+7) 个局部点, p={p_d:.4f})", flush=True)
print(f"[RHO] 并列口径 ρ(局部可读性, |局部ΔEM|)     = {rho_e:.3f} (n={n_all}, p={p_e:.4f})", flush=True)
print(f"[RHO] 辅助：每 ckpt 内单独算 ρ 再取均值 = {mean_ck:.3f}（{len(rho_ck)} 个值 "
      f"min={min(rho_ck):.3f} max={max(rho_ck):.3f}）| y 可靠性 ρ拆半(|Δ数字|前半,|Δ数字|后半) = "
      f"{rel:.3f} (p={rel_p:.4f})", flush=True)
print(f"[RHO] x 水平数 = {x_lv} 个不同取值 | x 范围 {min(xs)*100:.1f}%..{max(xs)*100:.1f}% | "
      f"|Δ数字| 非零 {nz_d}/{n_all}，中位={sorted(y_d)[len(y_d)//2]:.3f}pp "
      f"最大={max(y_d):.3f}pp | |ΔEM| 中位={sorted(y_e)[len(y_e)//2]:.2f}pp "
      f"最大={max(y_e):.2f}pp", flush=True)

print(f"\n{verdict_of(rho_d, n_all, '[VERDICT-P3·主判据 |Δ数字每步|]')}", flush=True)
print(f"{verdict_of(rho_e, n_all, '[VERDICT-P3·并列 |ΔEM|]')}", flush=True)

# ============================================================================
# ⑦ 直观两例：高可读小因果 / 低可读大因果
# ============================================================================
srt = sorted(xs)
q_hi = srt[int(len(srt) * 2 / 3)]
q_lo = srt[int(len(srt) / 3)]
hi = [p for p in POINTS if p["acc_te"] >= q_hi]
lo = [p for p in POINTS if p["acc_te"] <= q_lo]
ex1 = min(hi, key=lambda p: abs(p["dacc"])) if hi else None
ex2 = max(lo, key=lambda p: abs(p["dacc"])) if lo else None
if ex1:
    print(f"\n[EX1] 高可读·低因果：{ex1['ckpt']} {ex1['unit']} 局部可读性={ex1['acc_te']*100:.1f}% "
          f"|Δ数字|={abs(ex1['dacc']):.3f}pp |ΔEM|={abs(ex1['dem']):.2f}pp"
          f"（基线EM={ex1['em_b']*100:.2f}%→{ex1['em_i']*100:.2f}%）", flush=True)
if ex2:
    print(f"[EX2] 低可读·高因果：{ex2['ckpt']} {ex2['unit']} 局部可读性={ex2['acc_te']*100:.1f}% "
          f"|Δ数字|={abs(ex2['dacc']):.3f}pp |ΔEM|={abs(ex2['dem']):.2f}pp"
          f"（基线EM={ex2['em_b']*100:.2f}%→{ex2['em_i']*100:.2f}%）", flush=True)

print(f"\n[META] device={DEV} N_EVAL={N_EVAL} EM_batch={EM_BATCH} side={SIDE} smoke={SMOKE} "
      f"ckpt数={len(per_ck)} 局部点总数n={n_all} Δ数字非零={nz_d} per-ckpt ρ均值={mean_ck:.3f} "
      f"EM回退={EM_FALLBACK} 总墙钟={(time.time()-T0)/60:.1f}min", flush=True)
print("[DONE] exit=0", flush=True)
