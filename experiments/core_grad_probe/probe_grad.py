#!/usr/bin/env python3
"""core_grad_probe —— 在与 `select_semantic_joint/train_sem.py --arm joint` **逐行同构**的
一步训练循环上，拆出 select CE 与 4 个老任务 loss 对**核 θ_core** 的梯度量级与方向。

回答（跑前写死，见 PREREG.md）：
  g_select = ‖∂L_select/∂θ_core‖₂
  g_old_i  = ‖∂L_old_i/∂θ_core‖₂（i = pronoun/sentiment/relation/person）
  g_old    = ‖∂ΣL_old_i/∂θ_core‖₂        ← 合并求（与分别求因方向不同而不同，两个都报）
  cos(g_select, g_old)、分层拆解、‖θ_core‖、‖g_select‖/‖θ_core‖

同构怎么保证：
  * 构造段、数据、批量、精度、优化器、LR（T_max=1800）、反传路径全部复用主臂
    `common.py`（只读 import）与 `train_sem.IndexDataset`，**不另写前向**；
  * 探针只在 `total` 组装完之后、`opt.zero_grad()` **之前**插入 `torch.autograd.grad`
    （不写 `.grad`、不消耗 RNG、不动 optimizer/scheduler）⇒ 训练步本身与主臂逐位一致；
  * 对账：R4 step-0 sha256、loss_first / first50 均值 vs 主臂 results/joint_s42.json。

用法（由 run_probe.sh 经 systemd-run 启动）：
  uv run python experiments/core_grad_probe/probe_grad.py --seed 42 --steps 200
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
SEM = HERE.parent / "select_semantic_joint"          # 只读
sys.path.insert(0, str(SEM))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import (  # noqa: E402  (select_semantic_joint/common.py，只读)
    BATCH, CLIP, ENC_PARAMS, FP_EXPECTED, HEAD_PARAMS, LR_CORE, LR_HEAD,
    OLD_CARDS, STEPS, WD, SemModel, SemSpec, core_drift, data_fingerprint,
    eval_old_cards, freeze_report_old, inputs_for, load_blob, load_old_heads,
    load_rows, old_loaders, old_loss,
)
import train_sem  # noqa: E402  (只读 import，取它的 IndexDataset 保证取数同构)

OUT = HERE / "results"
FINAL_WEIGHTS = SEM / "weights" / "joint_s42.pt"     # 主臂 1800 步终态（只读）
R4_BASE = SEM / "results" / "r4_baseline_s42.json"
JOINT_JSON = SEM / "results" / "joint_s42.json"

# ---- 跑前写死（PREREG §1）----
SNAP_POINTS = (1, 50, 200)
GROUPS = ("embedding", "blocks.0", "blocks.1", "blocks.2", "norm")


# ================= 测量函数自检（已知答案，PREREG §3.3） =================
def unit_checks() -> dict:
    g = torch.Generator().manual_seed(0)
    v = torch.randn(1000, dtype=torch.float64, generator=g)
    z = torch.zeros(1000, dtype=torch.float64)
    z[7] = 5.0
    e1 = torch.zeros(8, dtype=torch.float64)
    e2 = torch.zeros(8, dtype=torch.float64)
    e1[0] = 1.0
    e2[1] = 1.0
    out = {
        "cos_v_v": float(_cos(v, v)),                  # 已知答案 +1
        "cos_v_negv": float(_cos(v, -v)),              # 已知答案 −1
        "cos_e1_e2": float(_cos(e1, e2)),              # 已知答案 0
        "norm_known": float(_norm(z)),                 # 已知答案 5.0
    }
    assert abs(out["cos_v_v"] - 1.0) < 1e-12, out
    assert abs(out["cos_v_negv"] + 1.0) < 1e-12, out
    assert abs(out["cos_e1_e2"]) < 1e-12, out
    assert abs(out["norm_known"] - 5.0) < 1e-12, out
    return out


def _norm(v: torch.Tensor) -> float:
    return float(v.norm())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float((a * b).sum() / (na * nb))


def layer_of(name: str) -> str:
    for g in GROUPS:
        if name == g or name.startswith(g + "."):
            return g
    raise AssertionError(f"未分组的核参数：{name}")


def flat(grads, params, ids: list[int]) -> torch.Tensor:
    parts = []
    for i in ids:
        g = grads[i]
        parts.append(g.detach().reshape(-1).double() if g is not None
                     else torch.zeros(params[i].numel(), dtype=torch.float64))
    return torch.cat(parts)


# ================= 单点测量 =================
def measure(tag: str, step_no: int, loss: torch.Tensor, old_terms: list[torch.Tensor],
            old_sum: torch.Tensor, total: torch.Tensor, core_named: list,
            groups_idx: dict[str, list[int]]) -> dict:
    """对同一条计算图分别求 6 组梯度（retain_graph，不写 .grad）。"""
    core_params = [p for _, p in core_named]
    t0 = time.perf_counter()
    gsel = torch.autograd.grad(loss, core_params, retain_graph=True, allow_unused=True)
    gis = [torch.autograd.grad(t, core_params, retain_graph=True, allow_unused=True)
           for t in old_terms]
    gold = torch.autograd.grad(old_sum, core_params, retain_graph=True, allow_unused=True)
    gtot = torch.autograd.grad(total, core_params, retain_graph=True, allow_unused=True)
    sec = time.perf_counter() - t0

    # θ_core 每次现算（终态快照载入了新权重，不能复用 step-0 的 θ）
    theta_g: dict[str, float] = {}
    for g, ids in groups_idx.items():
        s = 0.0
        for i in ids:
            a = core_params[i].detach().reshape(-1).double()
            s += float((a * a).sum())
        theta_g[g] = s ** 0.5

    per_layer, vec_sel_all, vec_old_all, vec_tot_all = {}, [], [], []
    for grp, ids in groups_idx.items():
        vs, vo, vt = flat(gsel, core_params, ids), flat(gold, core_params, ids), flat(gtot, core_params, ids)
        n_sel, n_old, n_tot = _norm(vs), _norm(vo), _norm(vt)
        per_layer[grp] = {
            "theta": theta_g[grp],
            "g_select": n_sel,
            "g_old": n_old,
            "ratio_sel_over_old": (n_sel / n_old) if n_old > 0 else float("nan"),
            "cos": _cos(vs, vo),
            "g_total": n_tot,
            "g_old_i": {OLD_CARDS[i]: _norm(flat(g, core_params, ids)) for i, g in enumerate(gis)},
            "g_select_over_theta": n_sel / theta_g[grp] if theta_g[grp] > 0 else float("nan"),
        }
        vec_sel_all.append(vs)
        vec_old_all.append(vo)
        vec_tot_all.append(vt)
    vsel, vold, vtot = torch.cat(vec_sel_all), torch.cat(vec_old_all), torch.cat(vec_tot_all)

    norms_i = {OLD_CARDS[i]: _norm(flat(g, core_params, list(range(len(core_params)))))
               for i, g in enumerate(gis)}
    res = {
        "tag": tag, "step": step_no,
        "loss_select": float(loss.detach()),
        "loss_old_i": {OLD_CARDS[i]: float(t.detach()) for i, t in enumerate(old_terms)},
        "loss_old_sum": float(old_sum.detach()),
        "loss_total": float(total.detach()),
        "g_select": _norm(vsel),
        "g_old": _norm(vold),
        "g_old_i": norms_i,
        "sum_norm_g_old_i": float(sum(norms_i.values())),
        "cancel_factor_g_old_over_sum": _norm(vold) / max(float(sum(norms_i.values())), 1e-30),
        "g_total": _norm(vtot),
        "cos_sel_old": _cos(vsel, vold),
        "theta_core": float(sum(t * t for t in theta_g.values()) ** 0.5),
        "rel_resid_total_vs_sel_plus_old": float((vtot - (vsel + vold)).norm() / max(vtot.norm(), 1e-30)),
        "cos_total_vs_sel_plus_old": _cos(vtot, vsel + vold),
        "layers": per_layer,
        "grad_sec": round(sec, 3),
    }
    res["g_select_over_theta"] = res["g_select"] / res["theta_core"]
    res["ratio_sel_over_old"] = res["g_select"] / res["g_old"]
    res["additivity_ok"] = res["rel_resid_total_vs_sel_plus_old"] <= 1e-4
    if not res["additivity_ok"]:
        print(f"[control] ❌ {tag}: 可加性残差 "
              f"{res['rel_resid_total_vs_sel_plus_old']:.3e} > 1e-4（g_total ≠ g_sel+g_old）",
              flush=True)
    del gsel, gis, gold, gtot, vsel, vold, vtot
    return res


def compact_line(r: dict) -> str:
    return (f"[snap] {r['tag']:10s} step={r['step']:<4d} "
            f"L_sel={r['loss_select']:.6f} L_old_sum={r['loss_old_sum']:.4f} "
            f"|g_sel|={r['g_select']:.6e} |g_old|={r['g_old']:.6e} "
            f"ratio={r['ratio_sel_over_old']:.4f} cos={r['cos_sel_old']:+.4f} "
            f"sum_i={r['sum_norm_g_old_i']:.6e} cancel={r['cancel_factor_g_old_over_sum']:.4f} "
            f"|theta|={r['theta_core']:.4f} |g_sel|/|theta|={r['g_select_over_theta']:.3e} "
            f"resid={r['rel_resid_total_vs_sel_plus_old']:.2e}")


# ================= 主流程 =================
def run(seed: int, steps: int, device: str) -> dict:
    t_start = time.time()
    checks = unit_checks()                       # 测量函数自检（已知答案，先过再测）
    print(f"[unit_checks] {json.dumps(checks, ensure_ascii=False)}", flush=True)
    random.seed(seed)
    torch.manual_seed(seed)

    # ---- 构造段：与 train_sem.train_one(arm='joint') 逐行同序（RNG 流同源） ----
    spec = SemSpec.from_build_spec()
    model = SemModel(spec, freeze_core=False).to(device)
    freeze = model.freeze_report()
    print(f"[freeze] {json.dumps(freeze, ensure_ascii=False)}", flush=True)

    fp = data_fingerprint()
    assert fp == FP_EXPECTED, f"数据指纹漂移：{fp}"
    rows = load_rows()
    blob = load_blob()
    assert len(rows) == blob["n"], f"行数 {len(rows)} ≠ 缓存 {blob['n']}"

    labels = blob["labels"].clone()
    train_idx = torch.nonzero(blob["splits"] == 0, as_tuple=False).squeeze(-1)
    ds = train_sem.IndexDataset(train_idx)
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=gen, drop_last=False)

    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    assert n_params == HEAD_PARAMS + ENC_PARAMS, f"可训参数 {n_params} 不符"
    core_named = [(n, p) for n, p in model.encoder.named_parameters()]
    core_params = [p for _, p in core_named]
    assert sum(p.numel() for p in core_params) == ENC_PARAMS
    head_params = list(model.head.parameters())
    opt = torch.optim.AdamW([{"params": core_params, "lr": LR_CORE},
                             {"params": head_params, "lr": LR_HEAD}], weight_decay=WD)
    clip_targets = core_params + head_params
    # ★ T_max 必须是主臂的 1800（不是本探针的 steps）⇒ 前 N 步 LR 曲线与主臂完全一致
    assert STEPS == 1800, f"主臂 STEPS 漂移：{STEPS}"
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    for p in params:
        p.requires_grad_(True)

    heads = load_old_heads(seed, device)
    old_rep = freeze_report_old(heads)
    print(f"[freeze] 老卡头 {json.dumps(old_rep, ensure_ascii=False)}", flush=True)
    old_dl = old_loaders(seed)

    # ---- R4 step-0 对账（构造段是否与主臂逐位一致） ----
    r4 = eval_old_cards(model.encoder, heads, seed, device, with_eval_task=False)
    r4_base = json.loads(R4_BASE.read_text(encoding="utf-8"))
    r4_ok = r4["combined_sha256"] == r4_base["combined_sha256"]
    print(f"R4_STEP0 combined={r4['combined_sha256']} 基线={r4_base['combined_sha256']} "
          f"{'相等 ✅' if r4_ok else '不等 ❌'}", flush=True)
    assert r4_ok, "R4 step-0 与主臂基线不逐位相同 ⇒ 构造段不同构，测量作废"

    groups_idx: dict[str, list[int]] = {g: [] for g in GROUPS}
    for i, (n, _) in enumerate(core_named):
        groups_idx[layer_of(n)].append(i)
    assert sum(len(v) for v in groups_idx.values()) == len(core_named), "层分组漏参数"
    gparams = {g: sum(core_named[i][1].numel() for i in ids) for g, ids in groups_idx.items()}
    assert sum(gparams.values()) == ENC_PARAMS, f"分组参数量 {sum(gparams.values())} ≠ {ENC_PARAMS}"
    print(f"[groups] {json.dumps(gparams, ensure_ascii=False)}", flush=True)

    # ---- 训练循环：与 train_sem 主臂同构（探针只插在 zero_grad 之前） ----
    iters = {n: iter(old_dl[n]) for n in OLD_CARDS}
    step = 0
    losses: list[float] = []
    totals: list[float] = []
    snapshots: list[dict] = []
    shapes_logged = False
    assert max(SNAP_POINTS) <= steps, f"steps={steps} 小于最大取值点 {max(SNAP_POINTS)}"
    # 迭代器创建时机 = 主臂 `for j in dl`（每个 epoch 一次 iter(dl) ⇒ 抽样序列同源）
    dl_iter = iter(dl)
    while step < steps:
        try:
            j = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            j = next(dl_iter)
        j = j[0] if isinstance(j, (list, tuple)) else j
        c, mc, d, md = inputs_for(model, blob, rows, j, True, device)
        y = labels[j].to(device)
        logits = model.forward_tokens(c, mc, d, md)
        loss = F.cross_entropy(logits, y)
        total = loss
        old_terms: list[torch.Tensor] = []
        for n in OLD_CARDS:
            try:
                b = next(iters[n])
            except StopIteration:
                iters[n] = iter(old_dl[n])
                b = next(iters[n])
            t = old_loss(heads, model.encoder, b, n, device)
            old_terms.append(t)
            total = total + t
        old_sum = old_terms[0]
        for t in old_terms[1:]:
            old_sum = old_sum + t

        if not shapes_logged:
            shapes_logged = True
            shape_info = {
                "select_ctx": list(c.shape), "select_ctx_mask": list(mc.shape),
                "select_cand": list(d.shape), "select_cand_mask": list(md.shape),
                "select_y": list(y.shape),
                "old_input_ids": list(b["input_ids"].shape),
                "old_attention_mask": list(b["attention_mask"].shape),
                "param_dtype": str(next(model.parameters()).dtype),
                "logit_dtype": str(logits.dtype),
                "batch": BATCH, "device": device,
                "lr_core_step1": opt.param_groups[0]["lr"],
                "lr_head_step1": opt.param_groups[1]["lr"],
            }
            print(f"[shapes] {json.dumps(shape_info, ensure_ascii=False)}", flush=True)

        if (step + 1) in SNAP_POINTS:
            r = measure(f"step{step + 1}", step + 1, loss, old_terms, old_sum, total,
                        core_named, groups_idx)
            snapshots.append(r)
            print(compact_line(r), flush=True)

        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(clip_targets, CLIP)
        opt.step()
        sched.step()
        losses.append(float(loss))
        totals.append(float(total))
        step += 1

    # ---- 同构对账：loss vs 主臂 joint_s42.json ----
    main = json.loads(JOINT_JSON.read_text(encoding="utf-8"))
    w = min(50, len(losses))
    acct = {
        "window": w,
        "loss_first": losses[0],
        "main_loss_first": main["loss_first"],
        "d_loss_first": losses[0] - main["loss_first"],
        "loss_select_first50": sum(losses[:w]) / w,
        "main_loss_select_first50": main["loss_select_first50"],
        "d_select_first50": (sum(losses[:w]) / w - main["loss_select_first50"]) if w == 50 else None,
        "loss_total_first50": sum(totals[:w]) / w,
        "main_loss_total_first50": main["loss_total_first50"],
        "d_total_first50": (sum(totals[:w]) / w - main["loss_total_first50"]) if w == 50 else None,
        "r4_step0_sha256": r4["combined_sha256"],
        "r4_match_baseline": r4_ok,
    }
    acct["ok"] = bool(abs(acct["d_loss_first"]) <= 1e-4 and all(
        acct[k] is None or abs(acct[k]) <= 1e-4
        for k in ("d_select_first50", "d_total_first50")))
    if not acct["ok"]:
        print("[account] ❌ 与主臂 loss 对账超出 1e-4 ⇒ 同构性存疑，结果不可用", flush=True)
    print("[account] " + json.dumps(acct, ensure_ascii=False), flush=True)
    print(f"[train] {step} 步 done，select loss 首 {losses[0]:.4f} / "
          f"前50均 {sum(losses[:w]) / w:.4f} | total 前50均 {sum(totals[:w]) / w:.4f}",
          flush=True)

    # ---- 终态快照（非轨迹点）：主臂 1800 步权重 + 本循环的下一批 ----
    final_rec = None
    if FINAL_WEIGHTS.exists():
        sd = torch.load(FINAL_WEIGHTS, map_location=device, weights_only=True)
        model.load_state_dict(sd, strict=True)
        drift = core_drift(model.encoder)
        main_drift = main["core_drift"]
        print(f"[final] 载入 {FINAL_WEIGHTS.name} core_drift={drift:.6f} "
              f"主臂={main_drift:.6f} Δ={drift - main_drift:+.2e}", flush=True)
        assert abs(drift - main_drift) < 1e-6, "终态权重与主臂 core_drift 对不上（载错权重）"

        try:
            j = next(dl_iter)
        except StopIteration:
            dl_iter = iter(dl)
            j = next(dl_iter)
        j = j[0] if isinstance(j, (list, tuple)) else j
        c, mc, d, md = inputs_for(model, blob, rows, j, True, device)
        y = labels[j].to(device)
        logits = model.forward_tokens(c, mc, d, md)
        loss = F.cross_entropy(logits, y)
        total = loss
        old_terms = []
        for n in OLD_CARDS:
            try:
                b = next(iters[n])
            except StopIteration:
                iters[n] = iter(old_dl[n])
                b = next(iters[n])
            t = old_loss(heads, model.encoder, b, n, device)
            old_terms.append(t)
            total = total + t
        old_sum = old_terms[0]
        for t in old_terms[1:]:
            old_sum = old_sum + t
        final_rec = measure("final", 1801, loss, old_terms, old_sum, total,
                            core_named, groups_idx)
        final_rec["note"] = ("非轨迹点：主臂 weights/joint_s42.pt（1800 步终态）+ 本 dataloader 的下一批"
                             "（= 第 201 步会用的那批）；此点不跑 optimizer.step")
        print(compact_line(final_rec), flush=True)
    else:
        print(f"[final] 缺 {FINAL_WEIGHTS}，跳过终态快照", flush=True)

    peak = (torch.cuda.max_memory_allocated() / 2 ** 20) if device.startswith("cuda") else 0.0
    return {
        "meta": {
            "seed": seed, "steps": steps, "snapshot_points": list(SNAP_POINTS),
            "final_weights": str(FINAL_WEIGHTS) if final_rec else None,
            "device": device, "batch": BATCH, "lr_core": LR_CORE, "lr_head": LR_HEAD,
            "clip": CLIP, "wd": WD, "sched_T_max": STEPS,
            "freeze": freeze, "old_heads_freeze": old_rep,
            "data_fingerprint": fp,
            "shapes": shape_info,
            "peak_mem_mb": round(peak, 1),
            "wall_sec": round(time.time() - t_start, 1),
            "group_params": {g: sum(core_named[i][1].numel() for i in ids)
                             for g, ids in groups_idx.items()},
        },
        "unit_checks": checks,
        "account": acct,
        "snapshots": snapshots,
        "final": final_rec,
        "loss_first50_mean": sum(losses[:w]) / w,
        "loss_total_first50_mean": sum(totals[:w]) / w,
    }


def main(argv: list[str]) -> None:
    global SNAP_POINTS
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="冒烟：只测 step 1 + 终态，2 步就停（产物另存，不污染正式结果）")
    a = ap.parse_args(argv)
    if a.smoke:
        SNAP_POINTS = (1,)
        a.steps = 2
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = run(a.seed, a.steps, device)
    OUT.mkdir(exist_ok=True)
    path = OUT / (f"grad_probe_smoke_s{a.seed}.json" if a.smoke else f"grad_probe_s{a.seed}.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] → {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
