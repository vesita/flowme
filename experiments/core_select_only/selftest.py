#!/usr/bin/env python3
"""core_select_only —— 训练前自检（PREREG §4 的空测试，**全部实测**）。

1. 两臂可训参数量与**冻结实况**（逐参数打印 `requires_grad`；核 eval() 模式）；
2. **selonly 未挂载老卡头**的实测证据（跑真实训练路径 100 步，读回 `mount_evidence`）；
3. 多数类基线（split 0.5000 / 两 ctype 0.5016）；
4. 随机标签对照**真执行**（randperm 后 1:1、不同条数 > 0）；
5. **核对臂代码同源**：本目录 joint 100 步 vs 只读 import 的 `select_semantic_joint/train_sem.py`
   joint 100 步（内存中把其输出路径重指到本目录，**不写他人目录**）⇒ loss 与权重对账；
6. T4 基线（`r4_baseline_s{42,43}.json`）就位。

用法：uv run python experiments/core_select_only/selftest.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SSJ = ROOT / "experiments" / "select_semantic_joint"
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(SSJ))

import torch  # noqa: E402

from common import (  # noqa: E402
    BAND, ENC_PARAMS, EXPECTED_N, HEAD_PARAMS, OLD_CARDS, SEEDS, SPLITS,
    SemModel, SemSpec, freeze_report_old, load_blob, load_old_heads, load_rows)
from train_core import RESULTS, train_one  # noqa: E402

FID_STEPS = 100          # ≥100，否则 train_sem 的 loss 前后 50 窗口重叠会误触发断言


def build(seed: int, device: str) -> SemModel:
    random.seed(seed)
    torch.manual_seed(seed)
    return SemModel(SemSpec.from_build_spec(), freeze_core=False).to(device)


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = 42
    out: dict = {"device": device, "seed": seed, "checks": {}}

    # ---- 1. 参数量 / 冻结实况（实测打印 requires_grad） ----
    m = build(seed, device)
    rep = m.freeze_report()
    assert rep["encoder_trainable"] == ENC_PARAMS, rep
    assert rep["trainable_params"] == HEAD_PARAMS + ENC_PARAMS, rep
    flags_enc = sorted({bool(p.requires_grad) for p in m.encoder.parameters()})
    flags_head = sorted({bool(p.requires_grad) for p in m.head.parameters()})
    assert flags_enc == [True] and flags_head == [True]
    heads = load_old_heads(seed, device)
    old_rep = freeze_report_old(heads)
    out["checks"]["params"] = {"model": rep, "encoder_requires_grad_flags": flags_enc,
                               "head_requires_grad_flags": flags_head,
                               "old_heads": old_rep}
    print(f"[1] 两臂可训 = {rep['trainable_params']}（核 {rep['encoder_trainable']} + "
          f"头 {rep['head_params']}）；encoder requires_grad={flags_enc}、"
          f"head requires_grad={flags_head}；核 training={rep['encoder_training']}", flush=True)
    print(f"[1] 老卡头冻结实测 = {json.dumps(old_rep, ensure_ascii=False)}", flush=True)
    del heads, m
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- 2. 多数类基线 ----
    rows = load_rows()
    blob = load_blob()
    maj = {}
    for i, s in enumerate(SPLITS):
        sel = blob["splits"] == i
        n = int(sel.sum())
        assert n == EXPECTED_N[s], f"{s} n={n}"
        cnt = torch.bincount(blob["labels"][sel], minlength=2)
        mm = float(cnt.max()) / n
        maj[s] = {"n": n, "counts": cnt.tolist(), "majority": round(mm, 6)}
        assert abs(mm - 0.5) < 1e-9, f"{s} 多数类 {mm} ≠ 0.5"
    for code, name in ((1, "heldout_pair"), (2, "shifted_pos")):
        sel = (blob["splits"] == 2) & (blob["ctype"] == code)
        n = int(sel.sum())
        assert n == 1250, f"{name} n={n}"
        cnt = torch.bincount(blob["labels"][sel], minlength=2).tolist()
        mm = float(max(cnt)) / n
        maj[name] = {"n": n, "counts": cnt, "majority": round(mm, 6)}
        assert abs(mm - 0.5) < 0.01, f"{name} 多数类 {mm} 偏离 0.5 太远：{cnt}"
    out["checks"]["majority"] = maj
    print(f"[2] 多数类基线：{json.dumps(maj, ensure_ascii=False)}", flush=True)

    # ---- 3. 随机标签真执行 ----
    n_train = EXPECTED_N["train"]
    g = torch.Generator().manual_seed(seed * 1000 + 7)
    idx = torch.arange(n_train)
    rl = blob["labels"].clone()
    rl[idx] = rl[idx][torch.randperm(n_train, generator=g)]
    cnt = torch.bincount(rl[:n_train], minlength=2).tolist()
    n_diff = int((rl[:n_train] != blob["labels"][:n_train]).sum())
    assert cnt == [n_train // 2, n_train - n_train // 2] and n_diff > 0
    out["checks"]["randlabel"] = {"counts": cnt, "n_diff": n_diff, "executed": True}
    print(f"[3] 随机标签真执行：计数 {cnt}，与真标签不同 {n_diff} 条", flush=True)

    # ---- 4. T4 基线就位 ----
    r4 = {}
    for s in SEEDS:
        p = SSJ / "results" / f"r4_baseline_s{s}.json"
        assert p.exists(), f"缺 step-0 参照：{p}"
        r4[str(s)] = json.loads(p.read_text(encoding="utf-8"))["combined_sha256"]
        assert r4[str(s)], "step-0 参照 sha256 为空"
    out["checks"]["step0_baseline"] = r4
    print(f"[4] step-0 老卡基线就位：{json.dumps(r4, ensure_ascii=False)}", flush=True)

    # ---- 5. 核对臂代码同源：本目录 joint vs 只读 import 的 train_sem joint（100 步） ----
    import train_sem  # noqa: E402  (select_semantic_joint/train_sem.py —— 只读 import)
    ref_results, ref_weights = HERE / "results", HERE / "weights"   # 内存中重指，不写他人目录
    train_sem.RESULTS = ref_results
    train_sem.WEIGHTS = ref_weights
    r_ref = train_sem.train_one("joint", seed, False, FID_STEPS, device, tag="_fidref")
    r_mine = train_one("joint", seed, False, FID_STEPS, device, tag="_fidmine")
    w_ref = torch.load(ref_weights / f"joint_fidref_s{seed}.pt", map_location="cpu",
                       weights_only=True)
    w_mine = torch.load(ref_weights / f"joint_fidmine_s{seed}.pt", map_location="cpu",
                        weights_only=True)
    diffs = {k: float((w_ref[k].float() - w_mine[k].float()).abs().max()) for k in w_ref}
    d_loss_first = abs(r_ref["loss_first"] - r_mine["loss_first"])
    d_loss50 = abs(r_ref["loss_select_first50"] - r_mine["loss_select_first50"])
    fid = {"steps": FID_STEPS, "loss_first_ref": r_ref["loss_first"],
           "loss_first_mine": r_mine["loss_first"], "d_loss_first": d_loss_first,
           "d_loss_first50": d_loss50,
           "max_abs_diff_per_tensor": diffs,
           "max_abs_diff": max(diffs.values()),
           "eval_acc_ref": {s: r_ref["eval"][s]["acc"] for s in SPLITS},
           "eval_acc_mine": {s: r_mine["eval"][s]["acc"] for s in SPLITS}}
    out["checks"]["fidelity"] = fid
    print(f"[5] 核对臂同源对账（{FID_STEPS} 步）：d_loss_first={d_loss_first:.3e} "
          f"d_loss_first50={d_loss50:.3e} 权重 max|Δ|={fid['max_abs_diff']:.3e}", flush=True)
    assert d_loss_first <= 1e-5, f"首步 loss 不同源：{d_loss_first}"
    del w_ref, w_mine
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- 6. selonly 真实训练路径的挂载证据（跑 100 步读回 json） ----
    r_sel = train_one("selonly", seed, False, FID_STEPS, device, tag="_selftest")
    me = r_sel["mount_evidence"]
    assert me["old_heads_mounted_during_training"] is False
    assert me["heads_keys_during_training"] == []
    assert me["old_loaders_during_training"] == []
    assert me["old_forward_count"] == 0
    assert r_sel["recipe"]["old_tasks_in_batch"] == []
    assert r_sel["old_heads_freeze"] in (None, {})
    assert not any(k in me["model_top_level_modules"] for k in OLD_CARDS)
    out["checks"]["selonly_mount"] = {
        "mount_evidence": me,
        "recipe_old_tasks_in_batch": r_sel["recipe"]["old_tasks_in_batch"],
        "model_top_level_modules": me["model_top_level_modules"]}
    print(f"[6] selonly 挂载证据 = {json.dumps(me, ensure_ascii=False)}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "selftest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("[save] results/selftest.json")
    print("SELFTEST_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
