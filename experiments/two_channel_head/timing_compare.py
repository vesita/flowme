#!/usr/bin/env python3
"""四臂**前向时间**对照（W2）：同进程、同 batch、同设备、**轮转交错**（消掉时钟/温度漂移）。

PREREG §3：计时必须在**真实前向**内量（同一条 forward、同一批真实 batch、同一设备），
并与 (a) 训练逐步墙钟、(b) 独立单跑三处对账。

对照（skill §2.2：新测量先跑已知答案）：
  · **A 与 D 的指针前向在初始权重下逐位相同**（同初始化 + 同代码路径）→ 先断言这个"已知答案"，
    不成立说明测量或构造有问题；
  · **A(第 1 轮) vs A(第 2 轮)** = 测量噪声地板 → A vs C / A vs D 的差必须大于噪声地板才算数。

输出：experiments/two_channel_head/results/timing_compare.json
用法：uv run python experiments/two_channel_head/timing_compare.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "src"))

from model import Spec, TwoChannelModel, encode_gen, load_ptr_blob  # noqa: E402

BATCH = 64
ROUNDS = 60
SEED = 42


def sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    spec = Spec()
    ptr = load_ptr_blob()
    rows = [json.loads(x) for x in
            (HERE / "data" / "test.jsonl").read_text(encoding="utf-8").splitlines()]
    base = TwoChannelModel("A", SEED, spec).to(dev)
    gen_blob = encode_gen(base, rows, spec, dev)          # 缓存，之后三臂共用

    pc = ptr["v_ctx"][:BATCH].to(dev)
    pd = ptr["v_cand"][:BATCH].to(dev)
    gs = gen_blob["v_sent"][:BATCH].to(dev)
    gi = gen_blob["v_items"][:BATCH].to(dev)
    gm = gen_blob["item_mask"][:BATCH].to(dev)
    gb = (gi * gm.unsqueeze(-1)).sum(1) / gm.sum(1, keepdim=True).clamp(min=1)

    arms = {a: TwoChannelModel(a, SEED, spec).to(dev) for a in ("A", "C", "D")}
    arms["A2"] = TwoChannelModel("A", SEED, spec).to(dev)

    # ---- 已知答案对照：A 与 D 的指针前向在初始权重下逐位相同 ----
    with torch.no_grad():
        la = arms["A"].forward_ptr(pc, pd)
        ld = arms["D"].forward_ptr(pc, pd)
        lc = arms["C"].forward_ptr(pc, pd)
    control = {
        "A_vs_D_ptr_bitwise": bool(torch.equal(la, ld)),
        "A_vs_C_ptr_bitwise_at_init": bool(torch.equal(la, lc)),
        "A_vs_A2_ptr_bitwise": bool(torch.equal(la, arms["A2"].forward_ptr(pc, pd))),
        "max_abs_A_vs_D": float((la - ld).abs().max()),
    }
    assert control["A_vs_D_ptr_bitwise"], control
    assert control["A_vs_A2_ptr_bitwise"], control
    print(f"[control] {json.dumps(control, ensure_ascii=False)}", flush=True)

    params = {k: [p for p in m.parameters() if p.requires_grad]
              for k, m in arms.items()}

    def fwd_ptr(m):
        return m.forward_ptr(pc, pd)

    def fwd_gen(m):
        return m.forward_gen(gs, gb, gi, gm)

    # 训练里每个 step 都把 batch 从 CPU 搬到 GPU（DataLoader 出来的是 CPU 张量），
    # 探针必须复现这条数据路径，否则「探针 ≠ 训练逐步」（skill §2）。
    pc_c, pd_c = pc.cpu(), pd.cpu()
    gs_c, gb_c, gi_c, gm_c = gs.cpu(), gb.cpu(), gi.cpu(), gm.cpu()

    def step(m, want_ptr: bool, want_gen: bool, h2d: bool = False):
        """与训练一步同构：loss = L_ptr(+L_gen)，逐头 clip，AdamW 各 1 步。
        h2d=True 时先做训练里那条 CPU→GPU 搬运（探针与逐步墙钟对账用）。"""
        if h2d:
            pc_l, pd_l = pc_c.to(dev), pd_c.to(dev)
            gs_l, gb_l = (gs_c.to(dev), gb_c.to(dev)) if want_gen else (None, None)
            gi_l, gm_l = (gi_c.to(dev), gm_c.to(dev)) if want_gen else (None, None)
        else:
            pc_l, pd_l, gs_l, gb_l, gi_l, gm_l = pc, pd, gs, gb, gi, gm
        loss = None
        if want_ptr:
            y = torch.randint(0, 2, (BATCH,), device=dev)
            loss = m.ptr_loss(pc_l, pd_l, y)
            _ = float(loss.detach())              # 训练里每步都有这个 .item() 同步
        if want_gen:
            sk = torch.randint(0, spec.n_skel, (BATCH,), device=dev)
            asg = torch.randint(0, spec.max_slots, (BATCH, spec.max_slots), device=dev)
            gl = m.gen_loss(gs_l, gb_l, gi_l, gm_l, sk, asg)[0]
            _ = float(gl.detach())
            loss = gl if loss is None else loss + gl
        loss.backward()
        groups = [[p for n, p in m.named_parameters()
                   if p.requires_grad and ("trunk_ptr" in n or n.startswith("ptr_out"))],
                  [p for n, p in m.named_parameters()
                   if p.requires_grad and ("trunk_gen" in n or n.startswith("gen"))]]
        if m.arm == "C":
            torch.nn.utils.clip_grad_norm_(params[m.arm if m.arm != "A2" else "A"], 1.0)
        else:
            for g in groups:
                if g:
                    torch.nn.utils.clip_grad_norm_(g, 1.0)
        for p in params[m.arm if m.arm != "A2" else "A"]:
            if p.grad is not None:
                p.grad = None

    # A / A2 只有指针分支（PREREG §1：臂内分支数 = 它该有的分支数，不多算）
    plan = [("A", "ptr", fwd_ptr),
            ("C", "ptr", fwd_ptr), ("C", "gen", fwd_gen),
            ("D", "ptr", fwd_ptr), ("D", "gen", fwd_gen),
            ("A2", "ptr", fwd_ptr)]
    # 三个臂都要跑整步（与逐步墙钟对账）
    step_plan = [("A", True, False), ("C", True, True), ("D", True, True)]
    step_h2d_plan = [("A", True, False), ("C", True, True), ("D", True, True)]
    for name, wp, wg in step_h2d_plan:
        for _ in range(10):
            step(arms[name], wp, wg, h2d=True)
    sync(dev)

    for name, _, fn in plan:
        for _ in range(20):
            fn(arms[name])
    for name, wp, wg in step_plan:
        for _ in range(10):
            step(arms[name], wp, wg)
    sync(dev)

    acc = {"ptr": {k: [] for k in arms}, "gen": {k: [] for k in arms},
           "step": {k: [] for k in ("A", "C", "D")},
           "step_h2d": {k: [] for k in ("A", "C", "D")}}
    sync_overhead = []
    for _ in range(50):                       # 同步本身的开销（测量系统信度）
        sync(dev); t0 = time.perf_counter(); sync(dev)
        sync_overhead.append((time.perf_counter() - t0) * 1000)
    sync_overhead.sort()
    acc["sync_overhead_ms"] = round(sync_overhead[len(sync_overhead) // 2], 4)
    for r in range(ROUNDS):
        rot = r % len(plan)                    # 轮转起点：消掉"总在第一个测"的位置偏差
        for j in range(len(plan)):
            name, kind, fn = plan[(j + rot) % len(plan)]
            sync(dev)
            t0 = time.perf_counter()
            out = fn(arms[name])
            del out
            sync(dev)
            acc[kind][name].append((time.perf_counter() - t0) * 1000)
        for j in range(len(step_plan)):
            name, wp, wg = step_plan[(j + r) % len(step_plan)]
            sync(dev); t0 = time.perf_counter()
            step(arms[name], wp, wg)
            sync(dev)
            acc["step"][name].append((time.perf_counter() - t0) * 1000)
            name, wp, wg = step_h2d_plan[(j + r) % len(step_h2d_plan)]
            sync(dev); t0 = time.perf_counter()
            step(arms[name], wp, wg, h2d=True)
            sync(dev)
            acc["step_h2d"][name].append((time.perf_counter() - t0) * 1000)

    def st(v):
        v = sorted(v)
        return {"n": len(v), "mean_ms": round(sum(v) / len(v), 4),
                "p50_ms": round(v[len(v) // 2], 4),
                "p95_ms": round(v[int(len(v) * .95)], 4),
                "min_ms": round(v[0], 4)}

    out = {"device": torch.cuda.get_device_name(0) if dev == "cuda" else dev,
           "batch": BATCH, "rounds": ROUNDS, "seed": SEED,
           "control": control,
           "fwd_ptr": {k: st(v) for k, v in acc["ptr"].items() if v},
           "fwd_gen": {k: st(v) for k, v in acc["gen"].items() if v},
           "step": {k: st(v) for k, v in acc["step"].items() if v},
           "step_h2d": {k: st(v) for k, v in acc["step_h2d"].items() if v},
           "sync_overhead_ms": acc["sync_overhead_ms"]}
    # 每臂前向 = 各通道前向之和（实测求和，用于与整步、逐步墙钟对账）
    out["fwd_total_est_ms"] = {
        "A": round(out["fwd_ptr"]["A"]["mean_ms"], 4),
        "C": round(out["fwd_ptr"]["C"]["mean_ms"] + out["fwd_gen"]["C"]["mean_ms"], 4),
        "D": round(out["fwd_ptr"]["D"]["mean_ms"] + out["fwd_gen"]["D"]["mean_ms"], 4),
    }
    out["noise_floor_ms"] = round(abs(out["fwd_ptr"]["A"]["mean_ms"]
                                      - out["fwd_ptr"]["A2"]["mean_ms"]), 4)
    out["ratios"] = {
        "C/D fwd": round(out["fwd_total_est_ms"]["C"] / out["fwd_total_est_ms"]["D"], 4),
        "D/A  fwd": round(out["fwd_total_est_ms"]["D"] / out["fwd_total_est_ms"]["A"], 4),
        "gen/ptr (D)": round(out["fwd_gen"]["D"]["mean_ms"]
                             / out["fwd_ptr"]["D"]["mean_ms"], 4),
        "step_D/step_A": round(out["step"]["D"]["mean_ms"]
                               / out["step"]["A"]["mean_ms"], 4),
    }
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / "timing_compare.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
