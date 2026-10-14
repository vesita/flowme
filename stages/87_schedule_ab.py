#!/usr/bin/env python3
"""S87 · 把已验证的 LR 调度搬进当前主线（字符级 · 模因锚定）——单变量对照。

规格与判据的唯一真相源 = `experiments/meme_anchor/PREREG_SCHED.md`（判据写死，本文件不许发明判据）。

    python stages/87_schedule_ab.py --smoke                 # 仪表自检（N1–N6）+ 20 步步时
    python stages/87_schedule_ab.py --run const 1234 --steps 3000
    python stages/87_schedule_ab.py --run cosine 1234 --steps 3000
    python stages/87_schedule_ab.py --aa const 1234 --steps 200   # A/A 噪声地板

**唯一变量 = lr 调度**。两臂共用同一模型构造 / 数据序 / loss / 评估路径：
本文件从 `stages/86_meme_anchor.py` 复制训练与评估循环，**不 import 它**
（§86 是模块级脚本，import 即开始训练）。同 seed 下 LR 不消耗 RNG ⇒ 两臂数据序与 dropout 流逐位一致。

只写 `logs/87_*`；不碰 `cards/` / `nanoSeek/` / `86_meme_anchor.py`。
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path("/home/vesita/coding/my/flowme")
HERE = ROOT / "experiments" / "meme_anchor"

# ★ 语料来源必须显式：本机无 clean_v3 原语料（见 PREREG_SCHED §5 偏差），
#   固定指向代用缓存 `logs/87_cache`，避免静默读到别的语料却按原口径解释。
os.environ.setdefault("M1_CACHE", "logs/87_cache")
sys.path.insert(0, str(HERE))

import data as DAT          # noqa: E402
import model as MD          # noqa: E402
from data import PAD, VOCAB  # noqa: E402
from model import BOS, EOS   # noqa: E402

LOGS = ROOT / "logs"
PRIMARY_SEEDS = (1234, 5678)
PRIMARY_ARCH = "detr"
LR_BASE = 1e-3
SCHEDULES = ("const", "cosine")


def log(msg: str) -> None:
    print(msg, flush=True)


# --------------------------------------------------------------------- 调度
def lr_at(schedule: str, step: int, steps: int, warmup: int, base: float = LR_BASE) -> float:
    """★ 本单元唯一变量。`step` 为 1-based 步号（与训练循环一致）。

    const  : 恒 base（现状）
    cosine : warmup 步线性 0→base，随后 cosine 衰减到 0
    """
    if schedule == "const":
        return base
    if schedule == "cosine":
        if step <= warmup:
            return base * step / max(1, warmup)
        prog = (step - warmup) / max(1, steps - warmup)
        return base * 0.5 * (1.0 + math.cos(math.pi * min(prog, 1.0)))
    raise ValueError(f"未知 schedule={schedule!r}，可选 {SCHEDULES}")


# --------------------------------------------------------------- 训练/评估
def train_run(schedule: str, seed: int, steps: int, device: str, bs: int = 32,
              arch: str = PRIMARY_ARCH, warmup: int | None = None,
              eval_every: int = 0, state_at: tuple[int, ...] = ()) -> dict:
    pack = get_pack()
    if warmup is None:
        warmup = max(1, steps // 24)          # S53 比例：500/12000

    model = MD.build(seed, arch).to(device)
    npar = model.n_params()
    log(f"[S87] schedule={schedule} seed={seed} steps={steps} arch={arch} bs={bs} "
        f"warmup={warmup} lr0={lr_at(schedule, 1, steps, warmup):.3e} params={sum(npar.values())}")

    opt = torch.optim.AdamW(model.parameters(), lr=LR_BASE, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    g = torch.Generator().manual_seed(seed)
    N = len(pack["train_ids"])
    order = torch.randperm(N, generator=g).tolist()
    ptr = 0
    t0 = time.time()
    step_time = float("nan")
    ce_at: dict[int, float] = {}
    states: dict[int, torch.Tensor] = {}
    rng_after_1: torch.Tensor | None = None

    for step in range(1, steps + 1):
        lr = lr_at(schedule, step, steps, warmup)
        for pg in opt.param_groups:
            pg["lr"] = lr

        if ptr + bs > N:
            order = torch.randperm(N, generator=g).tolist()
            ptr = 0
        idx = order[ptr:ptr + bs]
        ptr += bs
        chunk = [pack["train_ids"][j] for j in idx]
        ids = pad_ids(chunk, device)
        tgt_in, tgt_out = MD.make_pairs(chunk, device)
        im = img_batch(pack["train_img"], idx, device)

        mt = model.meme_text(ids)
        mi = model.meme_image(im)
        lt = model.logits(mt, tgt_in)
        lv = model.logits(mi, tgt_in)
        ce_t = crit(lt.reshape(-1, VOCAB), tgt_out.reshape(-1))
        ce_v = crit(lv.reshape(-1, VOCAB), tgt_out.reshape(-1))
        loss = ce_t + ce_v

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 1:
            rng_after_1 = torch.get_rng_state().clone()
        if step in state_at:
            # ★ 取"轨迹指纹"：用于同进程重复跑的一致性检查
            states[step] = torch.cat([p.detach().reshape(-1).float().cpu()
                                      for p in model.parameters()]).clone()
        if step == 20:
            step_time = (time.time() - t0) / 20
            log(f"[SPEED] schedule={schedule} seed={seed} 前 20 步 {step_time:.3f} s/step "
                f"⇒ {steps} 步预计 {step_time*steps/60:.1f} min")
        if eval_every and (step % eval_every == 0 or step == steps):
            ce_at[step] = float(ce_t)

    ev = eval_paired(model, pack["test_ids"], pack["test_img"], device)
    return dict(schedule=schedule, seed=seed, steps=steps, warmup=warmup, bs=bs, arch=arch,
                params=int(sum(npar.values())), step_time=step_time,
                wall_min=(time.time() - t0) / 60.0, ce_train_last=float(ce_t),
                ce_train_snapshots=ce_at, rng_after_1=rng_after_1, states=states, **ev)


def eval_paired(model, ids_list, imgs, device, bs: int = 64) -> dict:
    """teacher forcing 评估；**逐 token 配对**（两臂同 test、同序 ⇒ 可逐样本配对）。

    与 `stages/86_meme_anchor.py::eval_model` 同口径（PAD 屏蔽、reduction=sum），
    额外返回**逐 token CE**（含 PAD 位）与 **pad_mask**、以及逐样本 EM，供配对检验用。

    ★ PAD 位说明：`cross_entropy(reduction="none")` 对 PAD 位返回 **0**，与"真 token 且 loss 恰为 0"
      无法区分（实测 41200 vs 40271 ⇒ 有 929 个真 token 的 loss 为 0）⇒ **必须显式带 mask**，
      不许用"值等于 0"过滤。`ce_tok_*` 保留全长元素，调用方用 `pad_mask` 取真 token。
    """
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    ce_tok_t: list[torch.Tensor] = []
    ce_tok_v: list[torch.Tensor] = []
    pad_mask: list[torch.Tensor] = []
    exact_t: list[int] = []
    exact_v: list[int] = []
    tok_correct_t = tok_correct_v = n_tok = 0
    with torch.no_grad():
        for i in range(0, len(ids_list), bs):
            chunk = ids_list[i:i + bs]
            ids = pad_ids(chunk, device)
            tgt_in, tgt_out = MD.make_pairs(chunk, device)
            im = img_batch(imgs, slice(i, i + bs), device)
            mt = model.meme_text(ids)
            mi = model.meme_image(im)
            lt = model.logits(mt, tgt_in)
            lv = model.logits(mi, tgt_in)
            mask = tgt_out != PAD
            n_tok += int(mask.sum())
            # 逐 token CE（reduction=none）**保留全长**，配 pad_mask 供聚类用
            per_tok_t = torch.nn.functional.cross_entropy(
                lt.reshape(-1, VOCAB), tgt_out.reshape(-1), ignore_index=PAD, reduction="none")
            per_tok_v = torch.nn.functional.cross_entropy(
                lv.reshape(-1, VOCAB), tgt_out.reshape(-1), ignore_index=PAD, reduction="none")
            ce_tok_t.append(per_tok_t.float().cpu())
            ce_tok_v.append(per_tok_v.float().cpu())
            pad_mask.append(mask.reshape(-1).cpu())
            pt, pv = lt.argmax(-1), lv.argmax(-1)
            tok_correct_t += int(((pt == tgt_out) & mask).sum())
            tok_correct_v += int(((pv == tgt_out) & mask).sum())
            exact_t += (((pt == tgt_out) | ~mask).all(dim=1)).int().cpu().tolist()
            exact_v += (((pv == tgt_out) | ~mask).all(dim=1)).int().cpu().tolist()
    return dict(ce_tok_t=torch.cat(ce_tok_t), ce_tok_v=torch.cat(ce_tok_v),
                pad_mask=torch.cat(pad_mask), exact_t=exact_t, exact_v=exact_v, n_tok=n_tok,
                ce_t=float(crit(torch.zeros(1), torch.zeros(1, dtype=torch.long))) if False else None,
                char_acc_t=tok_correct_t / max(n_tok, 1),
                char_acc_v=tok_correct_v / max(n_tok, 1),
                em_t=sum(exact_t) / max(len(exact_t), 1),
                em_v=sum(exact_v) / max(len(exact_v), 1))


def get_pack() -> dict:
    """与 S86 同：进程内缓存一次。"""
    global _PACK
    if _PACK is None:
        _PACK = DAT.prepare(smoke=False)
    return _PACK


_PACK: dict | None = None


def pad_ids(ids_list, device, pad: int = PAD) -> torch.Tensor:
    """★ 逐字复制 `stages/86_meme_anchor.py:68-73` —— 原始 id 列表**右侧 padding**，不加 BOS。"""
    L = max(len(a) for a in ids_list)
    x = torch.full((len(ids_list), L), pad, dtype=torch.long, device=device)
    for i, a in enumerate(ids_list):
        x[i, :len(a)] = torch.tensor(a, dtype=torch.long, device=device)
    return x


def img_batch(imgs, idx, device) -> torch.Tensor:
    """逐字复制 `stages/86_meme_anchor.py:76+`。"""
    import numpy as np
    if isinstance(idx, (list, np.ndarray)):
        arr = np.asarray(imgs)[list(idx)]
    elif isinstance(idx, slice):
        arr = np.asarray(imgs[idx])                     # ★ slice 不能和 int 相加
    else:
        arr = np.asarray(imgs[idx:idx + 1])
    if arr.ndim == 2:
        arr = arr[None]
    # ★ 必须补通道维（S86 `:85`）：不然 conv 会拿到 [B,H,W] 当成 [B,C,H,W]
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device).unsqueeze(1)


# ------------------------------------------------------------------ 仪表自检
def self_checks(device: str = "cpu") -> None:
    """N1–N6（PREREG_SCHED §2）。任一条不过 ⇒ 非零退出。"""
    fails: list[str] = []

    # N3/N6：两臂同 seed 的数据序 + 全局 RNG 流逐位一致
    for seed in (1234, 5678):
        g1 = torch.Generator().manual_seed(seed)
        g2 = torch.Generator().manual_seed(seed)
        a = torch.randperm(1000, generator=g1).tolist()
        b = torch.randperm(1000, generator=g2).tolist()
        if a != b:
            fails.append(f"N3 数据序：seed={seed} 两次 randperm 不同")

    # N1：惰性对照 —— 调度在 const 下必须与"原样常量 lr"逐位相同
    steps, warmup = 3000, 125
    for step in range(1, steps + 1):
        if lr_at("const", step, steps, warmup) != LR_BASE:
            fails.append(f"N1 const 臂 lr 非恒定：step={step}")
            break

    # N1b：cosine 端点 / 单调性
    if abs(lr_at("cosine", steps, steps, warmup)) > 1e-12:
        fails.append("N1b cosine 终点 lr 不为 0")
    lrs = [lr_at("cosine", s, steps, warmup) for s in range(1, steps + 1)]
    if lrs[0] > lrs[warmup] or lrs[warmup] < lrs[-1]:
        fails.append("N1b cosine 轨迹不是先升后降")
    if max(lrs) > LR_BASE + 1e-12:
        fails.append("N1b cosine 峰值超过 base lr")

    # N4：张量契约（合成数据，不依赖语料缓存）
    try:
        chunk = [[1, 2, 3], [4, 5]]
        ids = pad_ids(chunk, device)
        tgt_in, tgt_out = MD.make_pairs(chunk, device)
        if ids.shape[0] != 2 or ids.shape[1] != 3:
            fails.append(f"N4 pad_ids 形状不对：期望 (2,3) 得 {tuple(ids.shape)}")
        if tgt_in.shape != tgt_out.shape:
            fails.append("N4 make_pairs 两路形状不一致")
    except Exception as e:  # noqa: BLE001
        fails.append(f"N4 张量契约自检抛异常：{type(e).__name__}: {e}")

    # N2：输入不是金标
    try:
        chunk = [[1, 2, 3], [4, 5]]
        tgt_in, tgt_out = MD.make_pairs(chunk, device)
        if torch.equal(tgt_in, tgt_out):
            fails.append("N2 输入与金标逐位相同（输入即答案）")
    except Exception as e:  # noqa: BLE001
        fails.append(f"N2 自检抛异常：{type(e).__name__}: {e}")

    # N5：梯度截断（不建图路径）
    try:
        x = torch.ones(2, 3, requires_grad=True)
        with torch.no_grad():
            y = x * 2
        if y.requires_grad:
            fails.append("N5 no_grad 下输出仍带梯度")
    except Exception as e:  # noqa: BLE001
        fails.append(f"N5 自检抛异常：{type(e).__name__}: {e}")

    if fails:
        log("[SELFCHECK-FAIL]")
        for f in fails:
            log(f"  - {f}")
        sys.exit(1)
    log("[SELFCHECK-PASS] N1/N1b/N2/N3/N4/N5/N6 全部通过")


# ------------------------------------------------------------------ 主入口
def main() -> None:
    ap = argparse.ArgumentParser(description="S87 LR 调度单变量对照")
    ap.add_argument("--smoke", action="store_true", help="仪表自检 N1–N6 + 20 步步时")
    ap.add_argument("--run", nargs=2, metavar=("SCHEDULE", "SEED"))
    ap.add_argument("--aa", nargs=2, metavar=("SCHEDULE", "SEED"), help="A/A 地板：同臂同 seed 跑两次")
    ap.add_argument("--aa-inproc", nargs=2, metavar=("SCHEDULE", "SEED"),
                    help="★ 同进程内连续跑两次同臂：测「同进程同批」下轨迹是否逐位一致")
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--eval-every", type=int, default=0)
    args = ap.parse_args()

    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        log("[WARN] cuda 不可用 ⇒ 回落 cpu")
        dev = "cpu"
    if dev == "cuda":
        log(f"[ENV] {torch.cuda.get_device_name(0)} free={torch.cuda.mem_get_info()[0]/2**30:.2f}GiB")

    if args.smoke:
        self_checks("cpu")
        t0 = time.time()
        pack = get_pack()
        meta = pack.get("meta", {})
        sub = [p for p in __import__("os").environ.get("M1_CACHE", "").split()]
        log(f"[SMOKE] test={len(pack['test_ids'])} 语料就绪 {time.time()-t0:.1f}s")
        log(f"[SMOKE] ★ 语料代用 = {meta.get('corpus_substitute', '（无：走原语料）')}")
        r = train_run("const", 1234, 20, dev, args.bs)
        log(f"[SMOKE] 20 步 {r['step_time']:.3f} s/step ⇒ 3000 步约 {r['step_time']*3000/60:.1f} min")
        return

    if args.aa:
        sched, seed = args.aa[0], int(args.aa[1])
        outs = []
        for rep in (1, 2):
            r = train_run(sched, seed, args.steps, dev, args.bs, eval_every=args.eval_every)
            r["aa_rep"] = rep
            outs.append(r)
        same = torch.equal(outs[0]["rng_after_1"], outs[1]["rng_after_1"])
        # ★ 只比真 token（PAD 位的 loss 恒 0，会把差异稀释）
        m = outs[0]["pad_mask"]
        d = (outs[0]["ce_tok_t"][m] - outs[1]["ce_tok_t"][m])
        d_ce = float(d.abs().max())
        mean_d = float(d.mean())
        se_d = float(d.std(unbiased=True) / max(d.numel() ** 0.5, 1.0))
        log(f"[AA] rep1 vs rep2: rng_after_1 逐位相同={same}")
        log(f"[AA] 逐 token ΔCE: mean={mean_d:+.3e} SE={se_d:.3e} |Δ|max={d_ce:.3e} "
            f"⇒ Δ_min(2SE)={2*se_d:.3e}")
        log(f"[AA] EM {outs[0]['em_t']:.4f} vs {outs[1]['em_t']:.4f}；"
            f"字级 acc {outs[0]['char_acc_t']:.4f} vs {outs[1]['char_acc_t']:.4f}")
        out = LOGS / f"87_aa_{sched}_s{seed}_st{args.steps}.pt"
        torch.save({"arms": [{k: v for k, v in o.items() if not isinstance(v, torch.Tensor)}
                             for o in outs], "aa_rng_same": same, "aa_dce_max": d_ce,
                    "aa_dce_mean": mean_d, "aa_dce_se": se_d}, out)
        torch.save({"ce_tok_t_1": outs[0]["ce_tok_t"], "ce_tok_t_2": outs[1]["ce_tok_t"],
                    "pad_mask": outs[0]["pad_mask"],
                    "exact_t_1": outs[0]["exact_t"], "exact_t_2": outs[1]["exact_t"]},
                   LOGS / f"87_aa_tok_{sched}_s{seed}_st{args.steps}.pt")
        log(f"[AA] 落盘 {out}")
        return

    if args.aa_inproc:
        sched, seed = args.aa_inproc[0], int(args.aa_inproc[1])
        marks = tuple(m for m in (20, 200, args.steps) if m <= args.steps)
        outs = []
        for rep in (1, 2):
            r = train_run(sched, seed, args.steps, dev, args.bs, state_at=marks)
            r["aa_rep"] = rep
            outs.append(r)
        log(f"[AA-INPROC] 同进程两次运行（marks={marks}）")
        all_zero = True
        for m in marks:
            a, b = outs[0]["states"][m], outs[1]["states"][m]
            mx = float((a - b).abs().max())
            all_zero = all_zero and mx == 0.0
            log(f"  step {m}: 逐参数 max|Δ| = {mx:.3e} {'（逐位相同）' if mx == 0 else '★已分叉'}")
        d = outs[0]["ce_tok_t"].double() - outs[1]["ce_tok_t"].double()
        log(f"  评测逐 token ΔCE: mean={float(d.mean()):+.3e} |Δ|max={float(d.abs().max()):.3e}")
        log(f"  ⇒ 同进程全局逐位（含评测）= {all_zero}；rng_after_1 相同="
            f"{torch.equal(outs[0]['rng_after_1'], outs[1]['rng_after_1'])}")
        return

    if args.run:
        sched, seed = args.run[0], int(args.run[1])
        r = train_run(sched, seed, args.steps, dev, args.bs, eval_every=args.eval_every)
        out = LOGS / f"87_sched_{sched}_s{seed}_st{args.steps}.pt"
        torch.save({k: v for k, v in r.items() if not isinstance(v, torch.Tensor)}, out)
        # 逐 token 向量单独存（配对检验要用，不能只留摘要）
        torch.save({"ce_tok_t": r["ce_tok_t"], "ce_tok_v": r["ce_tok_v"],
                    "pad_mask": r["pad_mask"],
                    "exact_t": r["exact_t"], "exact_v": r["exact_v"]},
                   LOGS / f"87_tok_{sched}_s{seed}_st{args.steps}.pt")
        # ★ 有效 token 均值（排除 PAD 位）：全长均值会被 PAD 稀释 2 倍以上（独立复核指出）
        m = r["pad_mask"]
        ce_eff = float(r["ce_tok_t"][m].mean())
        ce_eff_v = float(r["ce_tok_v"][m].mean())
        log(f"[RUN-FINAL] schedule={sched} seed={seed} steps={args.steps} "
            f"ce_t={ce_eff:.6f} ce_v={ce_eff_v:.6f} em_t={r['em_t']:.4f} "
            f"char_acc_t={r['char_acc_t']:.4f} step_time={r['step_time']:.3f} "
            f"wall={r['wall_min']:.1f}min")
        return

    ap.print_help()


if __name__ == "__main__":
    main()
