#!/usr/bin/env python3
"""S86n · H7 规模轴探针：**在这台 8G 卡上能训多大的模型**（预注册 `experiments/scale_axis/PREREG.md` 的 Q2）。

    .venv/bin/python stages/86n_scale_probe.py --d 256 --bs 8 --accum 4 --steps 300 --seed 1234

与主线驱动的差别只有两点：
  1. **可改尺寸**（`--d/--ff/--nhead`）：`model.py` 的维度是模块常量，这里在构造前改写它们 ⇒ 卡/解码器按新维度建；
  2. **不载入 v0 卡、全部随机初始化**：规模点上没有同尺寸的 v0 卡，只有同起点才可比。

规模一律同起点、同步数、同数据序。每条 8B / micro-batch 计的峰值都按对口径测（训练逐 micro-batch 取 max）。
**OOM 不算错**：捕获后照样写一行结果（`oom=true`），"装不下"本身就是这条轴的读数。
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

import importlib.util
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments" / "meme_anchor"))
sys.path.insert(0, str(ROOT / "stages"))


def load_driver():
    spec = importlib.util.spec_from_file_location("drv86g", ROOT / "stages" / "86g_zh_en.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def set_dims(d: int, ff: int, nhead: int) -> None:
    """把 `model.py` 的维度常量改写掉（`model_zh_en` 自己 import 了一份，也要同步）。"""
    import model as MD
    import model_zh_en as MZ
    MD.D, MD.FF, MD.NHEAD = int(d), int(ff), int(nhead)
    MZ.D = int(d)                      # ParserCard 用的是 import 进来的常量
    assert MD.D == MZ.D


def peak_mib(device) -> float:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / 2 ** 20
    return float("nan")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--ff", type=int, default=512)
    ap.add_argument("--nhead", type=int, default=4)
    ap.add_argument("--bs", type=int, default=8, help="micro-batch（峰值激活按它算）")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--bridge-mid", type=int, default=64)
    ap.add_argument("--eval-chunk", type=int, default=16)
    ap.add_argument("--gen-n", type=int, default=200)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    set_dims(a.d, a.ff, a.nhead)
    D = load_driver()
    pack = D.get_pack()
    dev = a.device
    rec: dict = dict(d=a.d, ff=a.ff, nhead=a.nhead, bs=a.bs, accum=a.accum, steps=a.steps, seed=a.seed,
                     eff_batch=a.bs * a.accum)

    model = D.MZ.build(a.seed, False, "parser", a.bridge_mid).to(dev)
    rec["params"] = sum(p.numel() for p in model.parameters())
    rec["params_mib"] = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20
    crit = nn.CrossEntropyLoss(ignore_index=D.PAD)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    if str(dev).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    g = torch.Generator().manual_seed(a.seed)
    order = torch.randperm(len(pack["train_ids"]), generator=g).tolist()
    ptr, t0 = 0, time.time()
    oom = False
    try:
        for _step in range(1, a.steps + 1):
            opt.zero_grad(set_to_none=True)
            for _k in range(a.accum):
                if ptr + a.bs > len(order):
                    order = torch.randperm(len(pack["train_ids"]), generator=g).tolist(); ptr = 0
                idx = order[ptr:ptr + a.bs]; ptr += a.bs
                zl = [pack["train_ids"][j][0] for j in idx]
                el = [pack["train_ids"][j][1] for j in idx]
                t, mz, me, bz, be, _ = D._terms(model, crit, zl, el, ("AaBb", "AbBa"), dev, obj="tf")
                D._combine_loss(t, None, mz, me, bz, be, "infonce", 0.3, 1.0,
                                "tf", 0.5, 0.07, 1.0, 0.2, "bridge").div(a.accum).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    except RuntimeError as e:                      # OOM：如实记一行，不当作失败
        oom = "out of memory" in str(e).lower() or "memory" in str(e).lower()
        rec["err"] = str(e)[:200]
        if not oom:
            raise
    rec["oom"] = bool(oom)
    rec["peak_train_mib"] = peak_mib(dev)
    rec["wall_min"] = (time.time() - t0) / 60
    rec["s_per_step"] = (time.time() - t0) / max(a.steps, 1)

    if not oom:
        if str(dev).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        ev = D.eval_model(model, pack, dev, chunk=a.eval_chunk)
        rec["peak_eval_mib"] = peak_mib(dev)
        for k in ("char_zz", "char_ee", "char_ze", "char_ez", "ce_zz", "ce_ee"):
            rec[k] = float(ev[k])
        if str(dev).startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        ge = D.gen_eval(model, pack, dev, n=a.gen_n)
        rec["peak_gen_mib"] = peak_mib(dev)
        rec["gen_em_zz"] = float(ge["gen_em_zz"]); rec["gen_em_ee"] = float(ge["gen_em_ee"])
    print("[SCALE] " + json.dumps(rec, ensure_ascii=False))
    if a.out:
        Path(a.out).write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
