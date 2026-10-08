#!/usr/bin/env python3
"""example_attribution —— 卡片化的**第一个真卖点：可归因**（整块可替换 / 可消融 ⇒ 可归因）。

背景（E1/S35）：卡边界【不产生能力】—— 三卡权重装进单体容器 ⇒ logits 逐位相同。
⇒ 卡片化的价值不在能力，而在 ①可归因 ②可版本化；本文件演示 ①。

用**已训好的 ckpt**（默认 `logs/36_ckpt_res_s6000_seed1234.pt` = E6/S36 res 臂饱和点）
做三件事：

  ① 门 B（恒等中介）★**R16 口径 = retrain-without，不是推理期破坏**：
     把目标卡整块替换成 `IdentityCard`（io=d->d，零参数，h→h），**重训同一步数**（默认 6000），
     再报 ΔEM = EM(恒等中介重训) − EM(接口基线)。
     接口卡（输入/输出）的初始化与基线同 seed、同构造顺序 ⇒ 逐位同初值（被丢弃的只是
     target 那份参数），所以唯一变量 =「目标卡这一块在不在」。
  ② 局部因果（★**非破坏性单点**，S21/S18 的 side=in 口径）：
     只清零被干预的**那一个单元**（逐位置 / 逐 16 维块），其余位置/维度原样不动
     ⇒ 报每单元 Δ数字每步正确率（与 ΔEM），按 |Δ| 降序算**前 20% 因果占比**。
  ③ R29 非零性冒烟（各干预轴的 Δ 非零比例；比例过低 ⇒ 该轴作废）
     + R28（batch=1 vs batch=16 逐字一致）自检 + 零干预恒等自检（Δ 必须精确 0）。

⇒ 这就是「卡边界」的用途：**整块可替换 / 可消融 ⇒ 可归因** ✓（不是能力来源，E1 已证）。

env：CARDS_ATTR_CKPT / CARDS_ATTR_STEPS（默认 6000 = ex.STEPS）/ CARDS_ATTR_BASE_EM
     （由 example_add3d 同设备基线传入 ⇒ 最干净的 R16 对照）
     CARDS_ATTR_N_EM（默认 200，EM 子集）/ CARDS_ATTR_N_CE（默认 800）/ CARDS_ATTR_N_SMOKE（64）
     CARDS_SEED / CARDS_THREADS
运行：`.venv/bin/python cards/example_attribution.py`（**CPU**）
只写 /tmp；不写 logs/、不 import stages/。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                                  # noqa: E402
from torch.optim import AdamW                 # noqa: E402

from cards import CardPipeline                # noqa: E402
from cards.base import IO_D_TO_D, Card        # noqa: E402


class IdentityCard(Card):
    """恒等中介卡（门 B）：io = d->d，形状保持，**零参数**，h → h。

    门 B 的评测必须在**训练期**替换 + 重训（R16：消融必须 retrain-without，
    不是 post-hoc removal —— 推理期破坏测的是「分布外」，不是「这块在不在」）。
    """

    io = IO_D_TO_D
    card_name = "target-identity"

    def __init__(self, d: int = 128, maxlen: int = 512, pad_id: int | None = None):
        super().__init__(d, maxlen, pad_id)

    def apply_card(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        return x


# ---------------------------------------------------------------- 干预定义（side=in）
def iv_ident():
    """零干预（恒等自检用）。"""
    def f(x):
        return x
    return f


def iv_pos(i: int):
    """单点清零：只把第 i 个位置的模因向量清零（该样本长度不够则不动）。"""
    def f(x):
        if i >= x.size(1):
            return x
        x = x.clone()
        x[:, i, :] = 0.0
        return x
    return f


def iv_blk(b: int, d: int, nblk: int):
    """16 维块清零：一次清零第 b 块（d 维里的 [b*16:(b+1)*16]）。"""
    bd = d // nblk
    lo, hi = b * bd, (b + 1) * bd

    def f(x):
        x = x.clone()
        x[:, :, lo:hi] = 0.0
        return x
    return f


def iv_all():
    """整体清零（阳性对照）。"""
    def f(x):
        return torch.zeros_like(x)
    return f


def with_intervention(pipe: CardPipeline, fn, thunk):
    """把干预挂在**目标卡的 forward 前**（= 模因进入 Think 的入口 = S21/S18 的 side=in）。

    用 forward_pre_hook 而不是换卡：模块身份不变 ⇒ 契约/MV/参数集合全程不动。
    干预只改「进目标卡的那份张量」，不改任何权重。
    """
    h = None
    if fn is not None:
        def _pre(_mod, args):
            return (fn(args[0]),) + tuple(args[1:])
        h = pipe.target.register_forward_pre_hook(_pre)
    try:
        return thunk()
    finally:
        if h is not None:
            h.remove()


def top_share(vals, p: float) -> float:
    """前 p 比例单元的 |Δ| 之和 / 全部 |Δ| 之和（S21 curve_stats 的 shares[p]）。"""
    vs = sorted((abs(float(v)) for v in vals), reverse=True)
    tot = sum(vs)
    if not vs or tot <= 0:
        return float("nan")
    k = max(1, math.ceil(p * len(vs)))
    return sum(vs[:k]) / tot


def sha16_of(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ============================================================================
# 主流程
# ============================================================================
def main() -> int:
    t0 = time.time()
    import cards.example_add3d as ex          # ★复用数据/训练/评测口径（不 import stages/）

    ckpt = Path(os.environ.get("CARDS_ATTR_CKPT",
                               str(ROOT / "logs" / "36_ckpt_res_s6000_seed1234.pt")))
    seed = int(os.environ.get("CARDS_SEED", str(ex.SEED)))
    steps = int(os.environ.get("CARDS_ATTR_STEPS", str(ex.STEPS)))
    n_em = int(os.environ.get("CARDS_ATTR_N_EM", "200"))
    n_ce = int(os.environ.get("CARDS_ATTR_N_CE", "800"))
    n_smoke = int(os.environ.get("CARDS_ATTR_N_SMOKE", "64"))
    base_em_env = os.environ.get("CARDS_ATTR_BASE_EM")
    NBLK = ex.D // 16          # 8 个 16 维块（d=128）
    BD = ex.D // NBLK          # 每块维数 = 16

    print(f"[ATTR] ===== 卡片化卖点①：可归因 | device={ex.DEVICE} threads="
          f"{torch.get_num_threads()} | ckpt={ckpt.name} sha16={sha16_of(ckpt)} | "
          f"seed={seed} 重训步数={steps} | N_EM={n_em} N_CE={n_ce} N_smoke={n_smoke} =====",
          flush=True)
    print(f"[ATTR] ckpt 存在={ckpt.exists()} | 依据：R16（消融=retrain-without）/ "
          f"R29（干预先非零冒烟）/ R28（batch=1）/ S21（非破坏性单点 + 前 20% 因果占比）",
          flush=True)

    # ---- 数据（与 example_add3d / S19 同源）----
    train, test = ex.build_data()
    n_pos = min(len(r["p"]) for r in test)
    units = [("pos", i) for i in range(n_pos)] + [("blk", b) for b in range(NBLK)]
    print(f"[DATA] {ex.BUCKET}: train={len(train)} test={len(test)} | prompt 长度 "
          f"{min(len(r['p']) for r in test)}..{max(len(r['p']) for r in test)} ⇒ n_pos={n_pos} | "
          f"主轴单元 = pos(0..{n_pos-1}) + blk(0..{NBLK-1},{BD}维/块) = {len(units)} 个", flush=True)

    # ---- 基线：把 E6 res 臂 ckpt 灌进 CardPipeline（接口 + 目标卡都是已训好的）----
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)
    base = ex.load_into_pipeline(sd)
    base.eval()
    em_recs, ce_recs, sm_recs = test[:n_em], test[:n_ce], test[:n_smoke]
    r_base = ex.em_score(base, em_recs)
    d_base = ex.digit_report(base, ce_recs)
    print(f"[BASE] ckpt 原位复测（本管线·同一评测代码）：EM(batch=1,n={len(em_recs)})="
          f"{r_base['em']*100:.2f}%±{r_base['se']*100:.2f} | 数字每步(n={len(ce_recs)})="
          f"{d_base['acc']*100:.2f}%±{d_base['acc_se']*100:.2f} | MV="
          f"{base.mv().short()} 接口sha={base.interface_sha256()[:16]}", flush=True)

    # ========================================================================
    # ① 门 B（恒等中介）：替换目标卡 + 重训同一步数（R16）
    # ========================================================================
    print(f"\n[GATE-B] ===== ① 门 B（恒等中介）★R16：目标卡→IdentityCard，重训 {steps} 步 =====",
          flush=True)
    torch.manual_seed(seed)
    ex.DROP_GEN.manual_seed(seed)                    # ★R34：两条随机流结构与基线一致
    idpipe = CardPipeline(d=ex.D, vocab_size=ex.V, ff=ex.FF, nhead=ex.NHEAD, maxlen=ex.MAXLEN,
                          pad_id=ex.PAD_ID, dropout=0.1, residual=True).to(ex.DEVICE)
    idpipe.target = IdentityCard(ex.D, ex.MAXLEN)    # ★目标卡整块替换（零参数）
    assert idpipe.contract == idpipe.input.contract, "替换后契约必须仍然一致"
    opt = AdamW(idpipe.parameters(), lr=ex.LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t1 = 0, 0, time.time()
    while step < steps:
        idx = []
        for _ in range(ex.BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = ex.masked_ce(idpipe, [train[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step % 1000 == 0 or step == steps:
            print(f"  [gate-b train] step={step}/{steps} loss={loss.item():.4f} "
                  f"elapsed={time.time()-t1:.0f}s", flush=True)
    r_id = ex.em_score(idpipe, em_recs)
    d_id = ex.digit_report(idpipe, ce_recs)
    dem_ckpt = (r_id["em"] - r_base["em"]) * 100
    print(f"[GATE-B] 恒等中介重训完成：EM={r_id['em']*100:.2f}%±{r_id['se']*100:.2f} | "
          f"数字每步={d_id['acc']*100:.2f}%±{d_id['acc_se']*100:.2f} | 参数量="
          f"{idpipe.n_params()/1e6:.3f}M（ckpt 基线 {base.n_params()/1e6:.3f}M）| "
          f"墙钟={(time.time()-t1)/60:.1f}min", flush=True)
    print(f"[GATE-B] ΔEM(R16) = 恒等中介 − ckpt原位基线 = "
          f"{r_base['em']*100:.2f}% → {r_id['em']*100:.2f}% = {dem_ckpt:+.2f}pp | "
          f"Δ数字每步 = {(d_id['acc']-d_base['acc'])*100:+.2f}pp", flush=True)
    if base_em_env:
        be = float(base_em_env)
        print(f"[GATE-B] ★同设备（CPU）基线 CARDS_ATTR_BASE_EM={be*100:.2f}% ⇒ "
              f"ΔEM(R16 主口径) = {r_id['em']*100:.2f}% − {be*100:.2f}% = "
              f"{(r_id['em']-be)*100:+.2f}pp（两侧都在 CPU 训练、同 seed/同流结构）", flush=True)
    else:
        print("[GATE-B] 未传 CARDS_ATTR_BASE_EM ⇒ 只报 ΔEM vs ckpt 原位基线"
              "（注意 ckpt 是 GPU 训的，Δ 里含设备项；传 example_add3d 的 CPU EM 可消掉）",
              flush=True)

    # ========================================================================
    # ② 局部因果（非破坏性单点，S21/S18 side=in）：逐位置 + 逐 16 维块清零
    # ========================================================================
    print(f"\n[LOCAL] ===== ② 局部因果（非破坏性单点 @side=in）：{len(units)} 个单元 =====",
          flush=True)
    pts = []
    for u_i, (kind, idx) in enumerate(units):
        fn = iv_pos(idx) if kind == "pos" else iv_blk(idx, ex.D, NBLK)
        un = f"pos{idx}" if kind == "pos" else f"blk{idx}[{idx*BD}:{(idx+1)*BD}]"
        d_iv = with_intervention(base, fn, lambda: ex.digit_report(base, ce_recs))
        r_iv = with_intervention(base, fn, lambda: ex.em_score(base, em_recs))
        dacc = (d_iv["acc"] - d_base["acc"]) * 100          # >0 ⇒ 清零反而变好
        dem = (r_iv["em"] - r_base["em"]) * 100
        pts.append(dict(kind=kind, idx=idx, unit=un, dacc=dacc, dem=dem,
                        ce=abs(dacc), acc=d_iv["acc"]))
        print(f"[PTS] {un:<18} Δ数字={dacc:+8.3f}pp ΔEM={dem:+7.2f}pp |CE|={abs(dacc):7.3f}pp",
              flush=True)
        if (u_i + 1) % 10 == 0:
            print(f"[PROG] {u_i+1}/{len(units)} 用时 {(time.time()-t0)/60:.1f} min", flush=True)

    shares = {p: top_share([q["ce"] for q in pts], p) for p in (0.1, 0.2, 0.5)}
    top = sorted(pts, key=lambda q: -q["ce"])
    s20 = shares[0.2]
    verdict = (f"★因果高度集中（前20%单元贡献 {s20*100:.1f}% ≥50% ⇒ 少数单元承担大部分因果效应）"
               if s20 >= 0.5 else
               f"★因果分布较均匀（前20%单元贡献 {s20*100:.1f}% <50%）")
    print(f"[LOCAL] ★前 10/20/50% 单元因果占比 = {shares[0.1]*100:.1f}% / {shares[0.2]*100:.1f}% / "
          f"{shares[0.5]*100:.1f}% | Σ|Δ数字|={sum(q['ce'] for q in pts):.1f}pp "
          f"(n={len(pts)} 单元) | 最大单元={top[0]['unit']}(|Δ|={top[0]['ce']:.2f}pp)", flush=True)
    print("[LOCAL] 前 5 因果单元：" + " | ".join(
        f"{q['unit']} Δ数字={q['dacc']:+.2f}pp ΔEM={q['dem']:+.2f}pp" for q in top[:5]), flush=True)
    print(f"[VERDICT-ATTR] 可归因性：整块可消融且效应可定位 —— {verdict}", flush=True)

    # ========================================================================
    # ③ R29 非零冒烟 + R28 (batch=1) 自检 + 零干预恒等自检
    # ========================================================================
    print(f"\n[SELF] ===== ③ R29 非零冒烟（n={len(sm_recs)}）+ R28 (batch=1) + 恒等自检 =====",
          flush=True)
    sm_base = ex.digit_report(base, sm_recs)
    axes = [("单点清零@in（主轴）", [iv_pos(i) for i in range(n_pos)]),
            ("16维块清零@in（主轴）", [iv_blk(b, ex.D, NBLK) for b in range(NBLK)]),
            ("整体清零@in（阳性对照）", [iv_all()])]
    for nm, fns in axes:
        ds = []
        for fn in fns:
            d_iv = with_intervention(base, fn, lambda: ex.digit_report(base, sm_recs))
            ds.append((d_iv["acc"] - sm_base["acc"]) * 100)
        nz = [abs(x) > 1e-3 for x in ds]
        nzr = sum(nz) / len(nz)
        med = sorted(abs(x) for x in ds)[len(ds) // 2]
        valid = (nzr >= 0.20) and (med >= 0.01)
        print(f"[R29] {nm:<22} 单元={len(fns):>2} |Δ数字|中位={med:8.4f}pp 最大="
              f"{max(abs(x) for x in ds):8.4f}pp | 非零比(>1e-3pp)={nzr*100:5.1f}% ⇒ "
              f"{'有效轴 ✓' if valid else '★无效轴 ⇒ 作废并剔除'}", flush=True)

    d_id_ce = with_intervention(base, iv_ident(), lambda: ex.digit_report(base, sm_recs))
    r_id_em = with_intervention(base, iv_ident(), lambda: ex.em_score(base, em_recs[:16]))
    r_b_em = ex.em_score(base, em_recs[:16])
    d_ident = (d_id_ce["acc"] - sm_base["acc"]) * 100
    e_ident = (r_id_em["em"] - r_b_em["em"]) * 100
    print(f"[SELF2] 零干预恒等（n={len(sm_recs)}/16）：Δ数字={d_ident:.3e}pp ΔEM={e_ident:.3e}pp "
          f"⇒ {'精确为 0 ✓（干预链无副作用）' if d_ident == 0.0 and e_ident == 0.0 else '★非 0 ⇒ 口径有误'}",
          flush=True)

    r28_recs = test[:16]
    t1_ = ex.greedy_gen(base, r28_recs, batch=1)
    tN_ = ex.greedy_gen(base, r28_recs, batch=16)
    same = sum(a == b for a, b in zip(t1_, tN_))
    print(f"[R28] 批内一致性 K={len(r28_recs)}: batch=1 vs batch=16 逐字一致 {same}/{len(r28_recs)}"
          f" ⇒ {'一致 ✓（batch=1 口径自洽）' if same == len(r28_recs) else '★不一致 ⇒ 仅 batch=1 口径可用'}",
          flush=True)

    out = Path("/tmp") / "cards_attribution_result.json"
    out.write_text(json.dumps(dict(
        ckpt=ckpt.name, ckpt_sha16=sha16_of(ckpt), seed=seed, steps=steps,
        base=dict(em=r_base["em"], dig=d_base["acc"], n_em=len(em_recs)),
        gate_b=dict(em=r_id["em"], dig=d_id["acc"], dem_vs_ckpt=dem_ckpt,
                    base_em_env=base_em_env),
        shares={str(k): v for k, v in shares.items()},
        units=[{k: v for k, v in q.items() if k != "acc"} for q in pts],
        r28=dict(same=same, k=len(r28_recs)),
        ident=dict(dacc=d_ident, dem=e_ident),
        wall_min=(time.time() - t0) / 60), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[ATTR] 结果落 {out} | 总墙钟={(time.time()-t0)/60:.1f}min exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
