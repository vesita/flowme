#!/usr/bin/env python3
"""S86c · 度量仪器的**已知答案自检**（M1 模因锚定的仪器验证）。

为什么单独一个文件：6 run 正在跑，**不许改动** `measure.py` / `86_meme_anchor.py`（后面的臂会重新
import，改了就造成"前几臂旧代码、后几臂新代码"的不可比）。本文件只**读**它们，用已知答案的合成
张量验证度量实现，不产生任何实验结论。

    python stages/86c_instrument_check.py
"""
from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path

import torch

ROOT = Path("/home/vesita/coding/my/flowme")
HERE = ROOT / "experiments" / "meme_anchor"
sys.path.insert(0, str(HERE))
import measure as MS   # noqa: E402

# 驱动模块名以数字开头 ⇒ 用 spec 加载（不执行 main）
_spec = importlib.util.spec_from_file_location("m1driver", ROOT / "stages" / "86_meme_anchor.py")
DRV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DRV)

OK, BAD = [], []


def check(name: str, cond: bool, detail: str) -> None:
    (OK if cond else BAD).append(name)
    print(f"[{'PASS' if cond else '★FAIL'}] {name} | {detail}", flush=True)


def main() -> None:
    g = torch.Generator().manual_seed(20261011)

    # T1 自相似 ⇒ same=1、Δ>0
    A = torch.randn(64, 16, 128, generator=g)
    st = MS.pair_stats(MS.cosmat(MS.pooled(A), MS.pooled(A)))
    check("T1 自相似矩阵", abs(st["same"] - 1.0) < 1e-5 and st["delta"] > 0,
          f"same={st['same']:.6f} rand={st['rand']:.4f} Δ={st['delta']:+.4f}±{st['delta_se']:.4f}")

    # T2 两路独立随机 ⇒ Δ 落在噪声内（不显著）
    B = torch.randn(64, 16, 128, generator=g)
    st2 = MS.pair_stats(MS.cosmat(MS.pooled(A), MS.pooled(B)))
    check("T2 独立随机 ⇒ 不显著", abs(st2["delta"]) <= 3 * st2["delta_se"],
          f"Δ={st2['delta']:+.4f}±{st2['delta_se']:.4f}（{abs(st2['delta'])/max(st2['delta_se'],1e-12):.2f}SE）")

    # T3 随机行置换 ⇒ Δ ≈ 0（这是"随机配对"对照的定义）
    P = torch.randperm(64, generator=g)
    st3 = MS.pair_stats(MS.cosmat(MS.pooled(A), MS.pooled(A)[P]))
    check("T3 随机置换 ⇒ Δ≈0", abs(st3["delta"]) < 0.02,
          f"Δ={st3['delta']:+.4f}±{st3['delta_se']:.4f}")

    # T4 AUROC 已知答案
    pos = torch.randn(800, generator=g) + 2.0
    neg = torch.randn(800, generator=g)
    a_sep = MS.auroc_rank(pos, neg)
    a_same = MS.auroc_rank(neg, neg.clone())
    check("T4 AUROC 可分", a_sep > 0.9, f"分离分布 AUROC={a_sep:.4f}")
    check("T4 AUROC 同分布", abs(a_same - 0.5) < 0.03, f"同分布 AUROC={a_same:.4f}")

    # T5 Youden 阈值已知答案
    th = MS.youden_threshold(pos, neg)
    check("T5 最优阈值", th["tpr"] > 0.8 and th["fpr"] < 0.2,
          f"thr={th['thr']:+.3f} TPR={th['tpr']:.3f} FPR={th['fpr']:.3f} J={th['j']:.3f}")

    # T6 slot_matrix 与手算一致
    a = torch.randn(4, 3, 8, generator=g)
    b = torch.randn(5, 3, 8, generator=g)
    got = DRV.slot_matrix(a, b)
    an = a / a.norm(dim=2, keepdim=True)
    bn = b / b.norm(dim=2, keepdim=True)
    man = torch.stack([torch.stack([(an[i] * bn[j]).sum(-1).mean() for j in range(5)])
                       for i in range(4)])
    check("T6 逐槽矩阵=手算", torch.allclose(got, man, atol=1e-6),
          f"max|Δ|={float((got-man).abs().max()):.2e}")

    # T7 shuffle_slots 逐样本独立且可复现（M3 不能是空测试）
    M = torch.randn(8, 16, 128, generator=g)
    M1, M2 = MS.shuffle_slots(M, 7), MS.shuffle_slots(M, 7)
    same_rows = sum(bool(torch.allclose(M1[i], M[i])) for i in range(8))
    order_changed = sum(int((M1[i].argmax(0) != M[i].argmax(0)).sum()) for i in range(8))
    check("T7 shuffle 可复现", torch.equal(M1, M2), "两次调用逐位相同")
    check("T7 shuffle 非恒等", order_changed > 0 and same_rows == 0,
          f"8 行里有 {same_rows} 行未变；槽序改变计数={order_changed}")

    # T8 退化矩阵（全常数，Δ≡0）⇒ 不许判过（订正 3 的守护；SE 可能只是浮点噪声而非精确 0）
    const = torch.full((32, 32), 0.9)
    st8 = MS.pair_stats(const)
    check("T8 常数矩阵不许判过", st8["pass2se"] is False and st8["pass_final"] is False,
          f"Δ={st8['delta']:+.4f} SE={st8['delta_se']:.3e} pass2se={st8['pass2se']}")

    # T9 绝对下限生效：Δ=0.005 但 SE 极小 ⇒ 统计过、下限不过
    N = 400
    S9 = 0.9 + 0.001 * torch.randn(N, N, generator=g)
    diag = 0.905 + 0.001 * torch.randn(N, generator=g)
    S9[torch.arange(N), torch.arange(N)] = diag
    st9 = MS.pair_stats(S9)
    check("T9 绝对下限生效", st9["pass2se"] and not st9["pass_floor"] and not st9["pass_final"],
          f"Δ={st9['delta']:+.4f}±{st9['delta_se']:.5f} pass2se={st9['pass2se']} "
          f"pass_floor={st9['pass_floor']} pass_final={st9['pass_final']}")

    # T10 规模：N=800 的 score_report 能在可接受时间内完成、且不爆内存
    N2 = 800
    T10 = torch.randn(N2, 16, 128, generator=g)
    I10 = T10 + 0.3 * torch.randn(N2, 16, 128, generator=g)
    t0 = time.time()
    rep = MS.score_report(MS.cosmat(MS.pooled(T10), MS.pooled(I10)), "scale")
    dt = time.time() - t0
    check("T10 规模可用", rep["delta"] > 0 and dt < 60,
          f"N={N2} Δ={rep['delta']:+.4f} AUROC中心={rep['auroc_centered']:.4f} 用时 {dt:.1f}s")

    print(f"\n[INSTRUMENT] 通过 {len(OK)} / 失败 {len(BAD)}"
          f"{' ⇒ 仪器可用' if not BAD else ' ⇒ ★仪器有问题：' + ', '.join(BAD)}", flush=True)


if __name__ == "__main__":
    main()
