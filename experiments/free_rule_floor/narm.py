#!/usr/bin/env python3
"""N 臂：**只给 `n_slots`（袋项个数）** 的骨架旁路（PREREG §1，跑前写死）。

`skel_logits = skel_out(h) + aux(z)`，其中 `z = f(n_slots)`：
GRU 每一步输入都是**同一个可学常向量**（零输入 + 零隐态会让 GRU 恒为 0，自检会拦），
再按 `mask` 做掩码均值 ⇒ 输出只依赖 `mask` 的前缀长度
（= 袋项个数 `n_slots`，`max_slots = 4`）。**不含 type/role/cls 标签内容，也不含位置桶。**

不变式（`--selfcheck`）：
  · trunk/gen 初值与 A/U/UP **逐位相同**（构造顺序同 bag_modules：trunk→占位→gen→lab）
  · aux 零初始化 ⇒ 第 0 步 N 的骨架 logits == A
  · **改标签内容不改 N 的 logits**（只读 mask 的直接证据）
  · **改 n_slots 会改 N 的 logits**（旁路非退化）
  · n_slots ∈ {1..4} 的 z 两两不同（查表可分辨）
  · 梯度可到 aux

只读复用 `experiments/bag_modules/model.py`（importlib 按路径加载，不改源码）。

用法：uv run python experiments/free_rule_floor/narm.py --selfcheck
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent
BAG_DIR = ROOT / "experiments" / "bag_modules"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bm = _load(BAG_DIR / "model.py", "frf_bagmod_ro")     # 只读
Spec = bm.Spec
BagModModel = bm.BagModModel
MODS = bm.MODS
ARMS = ("A", "N", "U", "UP")


class SlotCountBypass(nn.Module):
    """N 臂旁路：入参只有 `mask`（袋项存在位），内容入参显式丢弃。

    GRU 的每一步输入都是**同一个可学常向量** `seed_vec`（不是 0 —— 零输入 + 零隐态
    会让 GRU 恒等于 0，见 selfcheck 的退化断言），因此 `out[t]` 只依赖步数 t；
    再按 `mask` 掩码均值 ⇒ `z = f(n_slots)`（mask 恒为前 n 为真）。"""

    def __init__(self, d: int = 8, hidden: int = 32, n_skel: int = 40):
        super().__init__()
        self.d, self.hidden = d, hidden
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.aux = nn.Linear(hidden, n_skel, bias=False)
        nn.init.zeros_(self.aux.weight)                # 零初始化 ⇒ 第 0 步 == 臂 A
        self.seed_vec = nn.Parameter(torch.randn(d) / (d ** 0.5))

    def forward(self, mask: torch.Tensor) -> torch.Tensor:
        """[B,M] bool → z [B,hidden]：每步输入相同 ⇒ 只有 mask 的前缀长度起作用。"""
        e = self.seed_vec.view(1, 1, -1).expand(*mask.shape, self.d).to(
            self.aux.weight.dtype)
        out, _ = self.gru(e * mask.unsqueeze(-1).to(e.dtype))
        m = mask.unsqueeze(-1).to(out.dtype)
        return (out * m).sum(1) / m.sum(1).clamp(min=1.0)

    def logits(self, mask: torch.Tensor, type_t=None, role_t=None, cls_t=None,
               pos_b=None) -> torch.Tensor:
        """**只读 mask**；type/role/cls/pos_b 显式收下但不使用。"""
        return self.aux(self.forward(mask))


def build_model(arm: str, seed: int, spec: Spec | None = None,
                mods: tuple[str, ...] = MODS) -> BagModModel:
    """A/U/UP 直接用 bag_modules 的原模型（只读 import，代码逐字同源）；
    N = 先按 U 的构造顺序建（保证 trunk/gen 初值逐位一致），再把标签旁路换成
    只吃 mask 的 SlotCountBypass。"""
    assert arm in ARMS, arm
    if arm != "N":
        return BagModModel(arm, seed, spec, mods)
    m = BagModModel("U", seed, spec, mods)
    m.arm = "N"
    m.lab = SlotCountBypass(n_skel=m.spec.n_skel)
    return m


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def selfcheck(device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = Spec()
    rep: dict = {"device": device}
    models = {a: build_model(a, seed=42, spec=spec).to(device) for a in ARMS}

    # 0) 共享子系统自检（bag_modules 原样：冻结/逐位初值/臂A==structB/U==A/P…）
    rep["bag_modules_selfcheck"] = bm.selfcheck(device)

    # 1) trunk/gen 初值跨臂逐位相同
    def same(x, y):
        return all(torch.equal(x[k], y[k]) for k in x)
    t0, g0 = models["A"].trunk.state_dict(), models["A"].gen.state_dict()
    rep["init_identical_across_arms"] = all(
        same(t0, models[a].trunk.state_dict()) and same(g0, models[a].gen.state_dict())
        for a in ARMS)
    assert rep["init_identical_across_arms"], rep

    # 2) 参数量（PREREG §1 写死的数字）
    rep["params"] = {a: models[a].param_report() for a in ARMS}
    A, N, U = (rep["params"][a]["head_trainable"] for a in ("A", "N", "U"))
    rep["delta_vs_A"] = {"N": N - A, "U": U - A, "U_minus_N": U - N}
    assert N - A == 5320, rep["delta_vs_A"]     # GRU 4032 + aux 1280 + seed_vec 8
    assert U - A == 5456, rep["delta_vs_A"]

    # 3) N 零初始化 ⇒ 第 0 步 logits == A
    B, M, D = 32, spec.max_slots, spec.hidden
    v_sent = torch.randn(B, D, device=device)
    v_items = torch.randn(B, M, D, device=device)
    im = torch.ones(B, M, dtype=torch.bool, device=device)
    im[:, -1] = False
    lab = {"type_t": torch.randint(0, 4, (B, M), device=device),
           "role_t": torch.randint(0, 4, (B, M), device=device),
           "cls_t": torch.randint(0, 5, (B, M), device=device),
           "pos_b": torch.randint(0, 4, (B, M), device=device),
           "mask": im}
    vb = bm.v_bag_of(v_items, im)
    with torch.no_grad():
        s_a, _, _ = models["A"].forward(v_sent, vb, v_items, im)
        s_n, _, _ = models["N"].forward(v_sent, vb, v_items, im, lab)
    rep["N_step0_equals_A"] = bool(torch.allclose(s_n, s_a, atol=0))
    assert rep["N_step0_equals_A"], "N 的零初始化旁路没对齐 A"

    # 4) 只读 mask：**改标签内容不改 N 的 logits**
    #    aux 零初始化时任何比较都恒等 ⇒ 先把 aux 置成随机非零，让检验非平凡，
    #    测完再复位成 0（零初始化性质由上面第 3 项单独保证）。
    w0 = models["N"].lab.aux.weight.detach().clone()
    with torch.no_grad():
        models["N"].lab.aux.weight.copy_(torch.randn_like(w0) * 0.1)
        s_rnd, _, _ = models["N"].forward(v_sent, vb, v_items, im, lab)
        lab2 = dict(lab)
        for k in ("type_t", "role_t", "cls_t", "pos_b"):
            lab2[k] = torch.randint(0, 4, (B, M), device=device)
        s_n2, _, _ = models["N"].forward(v_sent, vb, v_items, im, lab2)
    rep["N_aux_nonzero_actually_outputs"] = bool((s_rnd - s_a).abs().max().item() > 0)
    rep["N_ignores_label_content"] = bool(torch.equal(s_n2, s_rnd))
    assert rep["N_ignores_label_content"] and rep["N_aux_nonzero_actually_outputs"], rep

    # 5a) 旁路非退化：n_slots ∈ {1..4} 的 z 两两不同
    zs = []
    with torch.no_grad():
        for n in (1, 2, 3, 4):
            mk = (torch.arange(M, device=device).view(1, -1) < n).expand(B, M)
            zs.append(models["N"].lab.forward(mk))
    rep["z_distinct_per_n"] = len({tuple(round(v, 6) for v in z[0].tolist())
                                   for z in zs}) == 4
    assert rep["z_distinct_per_n"], "n_slots 的 z 不可分辨（查表退化）"

    # 5) 旁路非退化：改 n_slots（mask 前缀长度）会改 logits（aux 非零下测）
    im2 = torch.ones(B, M, dtype=torch.bool, device=device)
    im2[:, 2:] = False                                  # n_slots 4 → 3
    with torch.no_grad():
        s_n3, _, _ = models["N"].forward(v_sent, vb, v_items, im2, {**lab, "mask": im2})
        models["N"].lab.aux.weight.copy_(w0)            # 复位零初始化
    d = (s_n3 - s_rnd).abs().max().item()
    rep["N_sensitive_to_n_slots"] = bool(d > 0)
    rep["N_sensitivity_maxabs"] = round(d, 6)
    assert rep["N_sensitive_to_n_slots"], "N 旁路对 n_slots 无反应（退化）"
    with torch.no_grad():                                # 复位后再验一次第 0 步性质
        s_n0, _, _ = models["N"].forward(v_sent, vb, v_items, im, lab)
    assert bool(torch.allclose(s_n0, s_a, atol=0)), "aux 复位后 N≠A"

    # 6) 梯度可到 aux
    models["N"].zero_grad(set_to_none=True)
    models["N"].loss(v_sent, vb, v_items, im,
                     torch.randint(0, spec.n_skel, (B,), device=device),
                     torch.randint(0, M, (B, M), device=device), lab)[0].backward()
    g = models["N"].lab.aux.weight.grad
    rep["aux_gets_grad"] = bool(g is not None and g.abs().sum() > 0)
    assert rep["aux_gets_grad"], rep

    # 7) n_slots 分布（数据事实）由 rules.py / analyze.py 报
    rep["ok"] = True
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.selfcheck:
        print(json.dumps(selfcheck(a.device), ensure_ascii=False, indent=2, default=str))
    else:
        ap.error("用 --selfcheck")
