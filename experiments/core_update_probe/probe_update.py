#!/usr/bin/env python3
"""core_update_probe —— 在与 `select_semantic_joint/train_sem.py --arm joint` **逐行同构**的
训练循环上，量 AdamW 的**实际更新口径**（补 `core_grad_probe` 遗留第 3 项）。

回答（跑前写死，见 PREREG.md）：
  在 AdamW 的实际更新下，核 θ_core 的一步位移由 select 还是老任务决定？
  (a) 真实优化器状态下，单步只喂 select / 只喂 old ⇒ 读 exp_avg/exp_avg_sq ⇒ ‖m/√v‖（分层）；
  (b) 同一 (θ, 状态) 三臂各跑一步 ⇒ ‖Δθ_sel‖ / ‖Δθ_old‖ / ‖Δθ_both‖、cos、可加性残差、净改变量。

同构怎么保证（与 core_grad_probe 同款）：
  * 构造段、数据、批量、精度、优化器、LR（T_max=1800）全部复用主臂 common.py / train_sem（只读 import）；
  * 探针只在 `total` 组装完、`opt.zero_grad()` 之前插入 autograd.grad + **克隆**优化器单步，
    单步跑在 θ0 的**叶子副本**上 ⇒ 真实参数/真实优化器/.grad/计算图一次都不碰，主臂训练步逐位不变；
  * 对账：R4 step-0 sha256、loss_first / first50 vs 主臂、‖g_select‖/‖g_old‖ vs core_grad_probe、
    both 臂 Δθ vs 主臂真实 opt.step() 的 Δθ。

用法（由 run_probe.sh 经 systemd-run 启动）：
  uv run python experiments/core_update_probe/probe_update.py --seed 42 --steps 200
"""
from __future__ import annotations

import argparse
import copy
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
    OLD_CARDS, STEPS, WD, SemModel, SemSpec, data_fingerprint,
    eval_old_cards, freeze_report_old, inputs_for, load_blob, load_old_heads,
    load_rows, old_loaders, old_loss,
)
import train_sem  # noqa: E402  (只读 import，取它的 IndexDataset 保证取数同构)

OUT = HERE / "results"
R4_BASE = SEM / "results" / "r4_baseline_s42.json"
JOINT_JSON = SEM / "results" / "joint_s42.json"
GRAD_JSON = HERE.parent / "core_grad_probe" / "results" / "grad_probe_s42.json"

# ---- 跑前写死（PREREG §1）----
SNAP_POINTS = (1, 50, 200)
GROUPS = ("embedding", "blocks.0", "blocks.1", "blocks.2", "norm")
ARMS = ("sel", "old", "both", "old_noclip", "sel_reset", "old_reset", "both_reset")


# ================= 测量函数自检（已知答案） =================
def _norm(v: torch.Tensor) -> float:
    return float(v.norm())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = float(a.norm()), float(b.norm())
    if na == 0.0 or nb == 0.0:
        return float("nan")
    return float((a * b).sum() / (na * nb))


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """‖a − b‖ / ‖b‖（b 为参照）"""
    nb = _norm(b)
    return _norm(a - b) / nb if nb > 0 else float("nan")


def unit_checks() -> dict:
    g = torch.Generator().manual_seed(0)
    v = torch.randn(1000, dtype=torch.float64, generator=g)
    z = torch.zeros(1000, dtype=torch.float64)
    z[7] = 5.0
    e1 = torch.zeros(8, dtype=torch.float64)
    e2 = torch.zeros(8, dtype=torch.float64)
    e1[0] = 1.0
    e2[1] = 1.0
    out = {"cos_v_v": _cos(v, v), "cos_v_negv": _cos(v, -v),
           "cos_e1_e2": _cos(e1, e2), "norm_known": _norm(z)}
    assert abs(out["cos_v_v"] - 1.0) < 1e-12, out
    assert abs(out["cos_v_negv"] + 1.0) < 1e-12, out
    assert abs(out["cos_e1_e2"]) < 1e-12, out
    assert abs(out["norm_known"] - 5.0) < 1e-12, out
    return out


def layer_of(name: str) -> str:
    for g in GROUPS:
        if name == g or name.startswith(g + "."):
            return g
    raise AssertionError(f"未分组的核参数：{name}")


def compact_line(r: dict) -> str:
    p, dd = r["precond"], r["dtheta"]
    return (f"[snap] {r['tag']:8s} step={r['step']:<4d} "
            f"rg={r['ratio_g_sel_over_old']:.4f} |theta|={r['theta_core']:.3f} || "
            f"(a) r_pc={p['ratio_sel_over_old']:.4f} r_pc_rst={p['ratio_sel_reset_over_old_reset']:.4f} "
            f"||(b) r_d={dd['ratio_sel_over_old']:.4f} r_d_rst={dd['ratio_sel_reset_over_old_reset']:.4f} "
            f"cos(s,o)={dd['cos_sel_old']:+.3f} cos(b,o)={dd['cos_both_old']:+.3f} "
            f"cos(b,s)={dd['cos_both_sel']:+.3f} resid={dd['additivity_resid']:.3e} "
            f"rel(b,o)={dd['rel_both_vs_old']:.3e} rel(b,s)={dd['rel_both_vs_sel']:.3e}")


# ================= 主流程 =================
def run(seed: int, steps: int, device: str) -> dict:
    t_start = time.time()
    checks = unit_checks()
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

    core_named = [(n, p) for n, p in model.encoder.named_parameters()]
    core_params = [p for _, p in core_named]
    head_params = list(model.head.parameters())
    assert sum(p.numel() for p in core_params) == ENC_PARAMS, "核参数量不符"
    assert sum(p.numel() for p in head_params) == HEAD_PARAMS, "头参数量不符"
    params = core_params + head_params            # 与 clip_targets 同序
    n_params = sum(p.numel() for p in params)
    assert n_params == HEAD_PARAMS + ENC_PARAMS, f"可训参数 {n_params} 不符"
    n_core = len(core_params)
    opt = torch.optim.AdamW([{"params": core_params, "lr": LR_CORE},
                             {"params": head_params, "lr": LR_HEAD}], weight_decay=WD)
    clip_targets = params
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
    assert sum(gparams.values()) == ENC_PARAMS, "分组参数量不符"
    offs: dict[int, tuple[int, int]] = {}
    _s = 0
    for i, (_, p) in enumerate(core_named):
        offs[i] = (_s, p.numel())
        _s += p.numel()
    assert _s == ENC_PARAMS
    print(f"[groups] {json.dumps(gparams, ensure_ascii=False)}", flush=True)

    # ================= 探针（闭包在 run 内 ⇒ 不污染模块全局） =================
    def make_opt(kind: str, plist: list):
        core_c, head_c = plist[:n_core], plist[n_core:]
        if kind == "adamw":
            return torch.optim.AdamW([{"params": core_c, "lr": LR_CORE},
                                      {"params": head_c, "lr": LR_HEAD}], weight_decay=WD)
        assert kind == "sgd"
        return torch.optim.SGD([{"params": core_c, "lr": LR_CORE},
                                {"params": head_c, "lr": LR_HEAD}],
                               weight_decay=0.0, momentum=0.0)

    def flat_core(grad_like) -> torch.Tensor:
        parts = []
        for g, p in zip(grad_like, core_params):
            parts.append(g.reshape(-1).double() if g is not None
                         else torch.zeros(p.numel(), dtype=torch.float64))
        return torch.cat(parts)

    def slice_groups(vec: torch.Tensor) -> dict[str, torch.Tensor]:
        return {g: torch.cat([vec[offs[i][0]:offs[i][0] + offs[i][1]] for i in ids])
                for g, ids in groups_idx.items()}

    def run_arm(grads, state0: dict | None, theta0: list[torch.Tensor],
                use_clip: bool = True, kind: str = "adamw") -> dict:
        """从 (θ0, state0) 出发跑**一步**。为不破坏保留的计算图（autograd 版本号校验），
        整步跑在 θ0 的**叶子副本**上：AdamW 更新逐参数独立 ⇒ 副本上的 Δθ 与在真实参数上跑逐位等价，
        真实参数、真实优化器、.grad 一次都不碰。"""
        pc = [t.detach().clone().requires_grad_(True) for t in theta0]   # θ0 副本
        for q, g in zip(pc, grads):
            q.grad = None if g is None else g.clone()
        clip_norm, scale = 0.0, 1.0
        if use_clip:
            gn = float(torch.nn.utils.clip_grad_norm_(pc, CLIP))
            clip_norm, scale = gn, min(1.0, CLIP / max(gn, 1e-30))
        o = make_opt(kind, pc)
        if state0:
            o.load_state_dict(copy.deepcopy(state0))
        for cg, src in zip(o.param_groups, opt.param_groups):
            cg["lr"] = src["lr"]                     # 载入该步真实（已衰减）LR
        o.step()

        d_parts, pc_parts, eff_parts = [], [], []
        for i, q in enumerate(pc[:n_core]):
            t0 = theta0[i]
            d_parts.append((q.data - t0).reshape(-1).double())
            st = o.state[q] if q in o.state else None
            if st is None or kind != "adamw":
                z = torch.zeros(q.numel(), dtype=torch.float64, device=q.device)
                pc_parts.append(z)
                eff_parts.append(z)
                continue
            m = st["exp_avg"].double()
            v = st["exp_avg_sq"].double()
            pc_parts.append((m / v.clamp_min(1e-30).sqrt()).reshape(-1))     # ‖m/√v‖ 口径
            bc1 = 1.0 - 0.9 ** float(st["step"])
            bc2 = 1.0 - 0.999 ** float(st["step"])
            eff_parts.append(((m / bc1) / ((v / bc2).sqrt() + 1e-8)).reshape(-1))  # 有效更新
        del pc, o
        return {"d": torch.cat(d_parts), "pc": torch.cat(pc_parts),
                "eff": torch.cat(eff_parts), "clip_norm": clip_norm, "clip_scale": scale}

    def measure(tag: str, step_no: int, loss, old_terms, old_sum, total) -> dict:
        t0 = time.perf_counter()
        gsel = torch.autograd.grad(loss, clip_targets, retain_graph=True, allow_unused=True)
        gold = torch.autograd.grad(old_sum, clip_targets, retain_graph=True, allow_unused=True)
        gtot = torch.autograd.grad(total, clip_targets, retain_graph=True, allow_unused=True)
        grad_sec = time.perf_counter() - t0

        gs_c, go_c, gt_c = (flat_core(gsel), flat_core(gold), flat_core(gtot))
        state0 = copy.deepcopy(opt.state_dict())
        theta0 = [p.detach().clone() for p in params]

        arms: dict[str, dict] = {}
        arms["sel"] = run_arm(gsel, state0, theta0, use_clip=True)
        arms["old"] = run_arm(gold, state0, theta0, use_clip=True)
        arms["both"] = run_arm(gtot, state0, theta0, use_clip=True)
        arms["old_noclip"] = run_arm(gold, state0, theta0, use_clip=False)
        arms["sel_reset"] = run_arm(gsel, None, theta0, use_clip=True)
        arms["old_reset"] = run_arm(gold, None, theta0, use_clip=True)
        arms["both_reset"] = run_arm(gtot, None, theta0, use_clip=True)

        # ---- 控制（已知答案，PREREG §1） ----
        ctrl: dict = {}
        theta_core_est = _norm(torch.cat([t.reshape(-1).double() for t in theta0[:n_core]]))
        zero = run_arm([None] * len(params), state0, theta0, use_clip=False)
        ctrl["zero_grad_dtheta_core_norm"] = _norm(zero["d"])          # 已知答案：恰 0
        assert ctrl["zero_grad_dtheta_core_norm"] == 0.0, ctrl
        sgd = {k: run_arm(g, None, theta0, use_clip=False, kind="sgd")
               for k, g in (("sel", gsel), ("old", gold), ("both", gtot))}
        sgd_resid = _rel(sgd["sel"]["d"] + sgd["old"]["d"], sgd["both"]["d"])
        ctrl["sgd_linear_additivity_resid"] = sgd_resid
        # fp32 参数空间量化下界：p − lr·g 的舍入 ‖round‖ ≈ 2^-24·‖θ‖ ⇒ resid ≈ 2^-24·‖θ‖/‖Δθ_sgd‖
        ctrl["sgd_resid_fp32_floor_est"] = (2.0 ** -24 * theta_core_est) / max(
            _norm(sgd["both"]["d"]), 1e-30)
        ctrl["grad_additivity_resid_f64"] = _norm(gt_c - (gs_c + go_c)) / max(_norm(gt_c), 1e-30)
        # 已知答案 1：梯度层可加性（float64）≈0；已知答案 2：SGD 步层残差 ≈ fp32 量化下界
        assert ctrl["grad_additivity_resid_f64"] < 1e-5, ctrl
        assert sgd_resid < 1e-3, ctrl
        nnz_sel, nnz_old = int((gs_c != 0).sum()), int((go_c != 0).sum())
        ctrl["nnz_g_select"] = nnz_sel
        ctrl["nnz_g_old"] = nnz_old
        ctrl["reset_precond_ratio_pred_sqrt_nnz"] = (nnz_sel / max(nnz_old, 1)) ** 0.5
        ctrl["reset_precond_ratio_meas"] = _norm(arms["sel_reset"]["pc"]) / max(
            _norm(arms["old_reset"]["pc"]), 1e-30)
        ctrl["old_clip_vs_noclip_reldiff"] = _rel(arms["old"]["d"], arms["old_noclip"]["d"])

        d = {k: v["d"] for k, v in arms.items()}
        pc = {k: v["pc"] for k, v in arms.items()}
        eff = {k: v["eff"] for k, v in arms.items()}
        dG = {k: slice_groups(v) for k, v in d.items()}
        pG = {k: slice_groups(v) for k, v in pc.items()}
        theta_core = theta_core_est
        lr_core_now = float(opt.param_groups[0]["lr"])

        def rat(a: torch.Tensor, b: torch.Tensor) -> float:
            nb = _norm(b)
            return _norm(a) / nb if nb > 0 else float("nan")

        layers = {}
        for gname, ids in groups_idx.items():
            layers[gname] = {
                "theta": _norm(torch.cat([core_params[i].detach().reshape(-1).double()
                                          for i in ids])),
                "ratio_dtheta_sel_over_old": rat(dG["sel"][gname], dG["old"][gname]),
                "ratio_dtheta_sel_reset_over_old_reset":
                    rat(dG["sel_reset"][gname], dG["old_reset"][gname]),
                "ratio_precond_sel_over_old": rat(pG["sel"][gname], pG["old"][gname]),
                "ratio_precond_sel_reset_over_old_reset":
                    rat(pG["sel_reset"][gname], pG["old_reset"][gname]),
                "cos_dtheta_sel_old": _cos(dG["sel"][gname], dG["old"][gname]),
                "cos_dtheta_both_old": _cos(dG["both"][gname], dG["old"][gname]),
            }

        rec = {
            "tag": tag, "step": step_no,
            "lr_core": lr_core_now, "lr_head": float(opt.param_groups[1]["lr"]),
            "loss_select": float(loss.detach()),
            "loss_old_i": {OLD_CARDS[i]: float(t.detach()) for i, t in enumerate(old_terms)},
            "loss_old_sum": float(old_sum.detach()),
            "loss_total": float(total.detach()),
            "grad_sec": round(grad_sec, 3),
            "theta_core": theta_core,
            # 原始梯度口径（与 core_grad_probe 对账）
            "g_select": _norm(gs_c), "g_old": _norm(go_c), "g_total": _norm(gt_c),
            "ratio_g_sel_over_old": rat(gs_c, go_c),
            "cos_g_sel_old": _cos(gs_c, go_c),
            # (a) 预条件范数
            "precond": {
                "norm_m_over_sqrt_v": {k: _norm(pc[k]) for k in ARMS},
                "norm_eff_update": {k: _norm(eff[k]) for k in ARMS},
                "ratio_sel_over_old": rat(pc["sel"], pc["old"]),
                "ratio_sel_reset_over_old_reset": rat(pc["sel_reset"], pc["old_reset"]),
                "eff_ratio_sel_over_old": rat(eff["sel"], eff["old"]),
                "eff_ratio_sel_reset_over_old_reset": rat(eff["sel_reset"], eff["old_reset"]),
                "cos_sel_old": _cos(pc["sel"], pc["old"]),
            },
            # (b) 真实位移
            "dtheta": {
                "norm": {k: _norm(d[k]) for k in ARMS},
                "ratio_sel_over_old": rat(d["sel"], d["old"]),
                "ratio_sel_reset_over_old_reset": rat(d["sel_reset"], d["old_reset"]),
                "cos_sel_old": _cos(d["sel"], d["old"]),
                "cos_both_old": _cos(d["both"], d["old"]),
                "cos_both_sel": _cos(d["both"], d["sel"]),
                "additivity_resid": _rel(d["sel"] + d["old"], d["both"]),
                "rel_both_vs_old": _rel(d["old"], d["both"]),
                "rel_both_vs_sel": _rel(d["sel"], d["both"]),
                "additivity_resid_reset": _rel(d["sel_reset"] + d["old_reset"],
                                               d["both_reset"]),
                "rel_both_reset_vs_old_reset": _rel(d["old_reset"], d["both_reset"]),
                "rel_both_reset_vs_sel_reset": _rel(d["sel_reset"], d["both_reset"]),
                "wd_share_both": (lr_core_now * WD * theta_core) / max(_norm(d["both"]), 1e-30),
            },
            "clip_scale": {k: arms[k]["clip_scale"] for k in ARMS},
            "clip_norm": {k: arms[k]["clip_norm"] for k in ARMS},
            "layers": layers,
            "controls": ctrl,
        }
        print(compact_line(rec), flush=True)
        print(f"[ctrl] {tag} zero={ctrl['zero_grad_dtheta_core_norm']:.1e} "
              f"sgd_resid={sgd_resid:.2e} (floor {ctrl['sgd_resid_fp32_floor_est']:.2e}) "
              f"grad_resid_f64={ctrl['grad_additivity_resid_f64']:.2e} nnz(sel,old)=({nnz_sel},{nnz_old}) "
              f"r_pred={ctrl['reset_precond_ratio_pred_sqrt_nnz']:.5f} "
              f"r_meas={ctrl['reset_precond_ratio_meas']:.5f} "
              f"clip_old_vs_noclip={ctrl['old_clip_vs_noclip_reldiff']:.3e} "
              f"clip_scale(sel,old,both)=({rec['clip_scale']['sel']:.4f},"
              f"{rec['clip_scale']['old']:.4f},{rec['clip_scale']['both']:.4f})", flush=True)
        return rec, theta0[:n_core], arms["both"]["d"]

    # ---- 训练循环：与 train_sem 主臂同构（探针只插在 zero_grad 之前） ----
    iters = {n: iter(old_dl[n]) for n in OLD_CARDS}
    step = 0
    losses: list[float] = []
    totals: list[float] = []
    snapshots: list[dict] = []
    shapes_logged = False
    assert max(SNAP_POINTS) <= steps, f"steps={steps} 小于最大取值点 {max(SNAP_POINTS)}"
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
                "select_ctx": list(c.shape), "select_cand": list(d.shape),
                "old_input_ids": list(b["input_ids"].shape),
                "param_dtype": str(next(model.parameters()).dtype),
                "logit_dtype": str(logits.dtype),
                "batch": BATCH, "device": device,
                "lr_core_step1": opt.param_groups[0]["lr"],
                "lr_head_step1": opt.param_groups[1]["lr"],
                "adamw_betas": list(opt.defaults["betas"]),
                "adamw_eps": opt.defaults["eps"],
            }
            print(f"[shapes] {json.dumps(shape_info, ensure_ascii=False)}", flush=True)

        pend = None
        if (step + 1) in SNAP_POINTS:
            rec, theta0_core, d_both = measure(f"step{step + 1}", step + 1,
                                               loss, old_terms, old_sum, total)
            snapshots.append(rec)
            pend = (rec, theta0_core, d_both)

        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(clip_targets, CLIP)
        opt.step()
        sched.step()
        if pend is not None:
            rec, theta0_core, d_both = pend
            dmain = torch.cat([(p.data - t).reshape(-1).double()
                               for p, t in zip(core_params, theta0_core)])
            rel_main = _rel(dmain, d_both)
            rec["controls"]["both_vs_main_step_reldiff"] = rel_main
            print(f"[ctrl] {rec['tag']} both臂 vs 主臂真实步 reldiff={rel_main:.3e}", flush=True)
            assert rel_main < 1e-4, f"both 臂与主臂真实步不一致：{rel_main}"
            del dmain, theta0_core, d_both, pend
        losses.append(float(loss))
        totals.append(float(total))
        step += 1

    # ---- 同构对账 1：loss vs 主臂 joint_s42.json ----
    main = json.loads(JOINT_JSON.read_text(encoding="utf-8"))
    w = min(50, len(losses))
    acct = {
        "window": w,
        "loss_first": losses[0], "main_loss_first": main["loss_first"],
        "d_loss_first": losses[0] - main["loss_first"],
        "loss_select_first50": sum(losses[:w]) / w,
        "main_loss_select_first50": main["loss_select_first50"],
        "d_select_first50": (sum(losses[:w]) / w - main["loss_select_first50"]) if w == 50 else None,
        "loss_total_first50": sum(totals[:w]) / w,
        "main_loss_total_first50": main["loss_total_first50"],
        "d_total_first50": (sum(totals[:w]) / w - main["loss_total_first50"]) if w == 50 else None,
        "r4_step0_sha256": r4["combined_sha256"], "r4_match_baseline": r4_ok,
    }
    acct["ok"] = bool(abs(acct["d_loss_first"]) <= 1e-4 and all(
        acct[k] is None or abs(acct[k]) <= 1e-4
        for k in ("d_select_first50", "d_total_first50")))
    print("[account_loss] " + json.dumps(acct, ensure_ascii=False), flush=True)
    assert acct["ok"], "与主臂 loss 对账超 1e-4 ⇒ 同构性存疑，结果作废"

    # ---- 同构对账 2：原始梯度比值 vs core_grad_probe ----
    gp = json.loads(GRAD_JSON.read_text(encoding="utf-8"))
    gp_by_step = {s["step"]: s for s in gp["snapshots"]}
    gp_cmp = []
    for r in snapshots:
        g = gp_by_step.get(r["step"])
        if g is None:
            continue
        item = {
            "step": r["step"],
            "g_select": r["g_select"], "gp_g_select": g["g_select"],
            "d_g_select": r["g_select"] - g["g_select"],
            "g_old": r["g_old"], "gp_g_old": g["g_old"], "d_g_old": r["g_old"] - g["g_old"],
            "ratio": r["ratio_g_sel_over_old"], "gp_ratio": g["ratio_sel_over_old"],
            "rel_d_ratio": (r["ratio_g_sel_over_old"] - g["ratio_sel_over_old"])
            / g["ratio_sel_over_old"],
            "cos_sel_old": r["cos_g_sel_old"], "gp_cos": g["cos_sel_old"],
        }
        gp_cmp.append(item)
    gp_ok = all(abs(i["rel_d_ratio"]) <= 1e-3 for i in gp_cmp) and len(gp_cmp) == len(SNAP_POINTS)
    print("[account_grad] " + json.dumps(gp_cmp, ensure_ascii=False), flush=True)
    print(f"[account_grad] ok={gp_ok}（复现 core_grad_probe 的 ‖g_sel‖/‖g_old‖，相对误差 ≤1e-3）",
          flush=True)
    assert gp_ok, "与 core_grad_probe 梯度比值对不上 ⇒ 口径不一致，结果作废"

    peak = (torch.cuda.max_memory_allocated() / 2 ** 20) if device.startswith("cuda") else 0.0
    return {
        "meta": {
            "seed": seed, "steps": steps, "snapshot_points": list(SNAP_POINTS),
            "device": device, "batch": BATCH, "lr_core": LR_CORE, "lr_head": LR_HEAD,
            "clip": CLIP, "wd": WD, "sched_T_max": STEPS,
            "freeze": freeze, "old_heads_freeze": old_rep, "data_fingerprint": fp,
            "shapes": shape_info, "group_params": gparams,
            "peak_mem_mb": round(peak, 1), "wall_sec": round(time.time() - t_start, 1),
            "no_final_point_reason": ("主臂 weights/joint_s42.pt 只存权重不存优化器状态 ⇒ "
                                      "1800 步的 (m,v) 不可复原，终态点需伪造状态 ⇒ 本测不取"),
        },
        "unit_checks": checks,
        "account_loss": acct,
        "account_grad_vs_core_grad_probe": gp_cmp,
        "account_grad_ok": gp_ok,
        "snapshots": snapshots,
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
                    help="冒烟：只测 step 1，2 步就停（产物另存，不污染正式结果）")
    a = ap.parse_args(argv)
    if a.smoke:
        SNAP_POINTS = (1,)
        a.steps = 2
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = run(a.seed, a.steps, device)
    OUT.mkdir(exist_ok=True)
    path = OUT / (f"update_probe_smoke_s{a.seed}.json" if a.smoke else f"update_probe_s{a.seed}.json")
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] → {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
