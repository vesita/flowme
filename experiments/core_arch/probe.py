#!/usr/bin/env python3
"""P20 H4 表示诊断：核冻结，取臂的池化表示 → 线性 probe（简化 `core_probe` 口径）。

probe 目标（PREREG §5 H4 写死）：
  skel     —— `skel_id`（two_channel_head 拆 8000/2500），**结构信息**
  trig_pos —— Task S `trig` 触发词起点分桶（9 类，**该标签从未参与训练**），**位置信息**
  ent      —— Task T SAME/DIFF（4000/1200），**有真值信息**
  b_pairs  —— 用 skel probe 预测 → `pair_success`（560 对）+ c_pairs per-side acc

用法：uv run python experiments/core_arch/probe.py --arm A --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))

import data as D  # noqa: E402
from model_core import Core  # noqa: E402

RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
CS = (0.01, 0.1, 1.0, 10.0)


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


@torch.no_grad()
def feats(model: Core, ids: torch.Tensor, device: str, bs: int = 1024) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        out.append(model.encode(ids[i:i + bs].to(device)).float().cpu())
    return torch.cat(out)


def zscore(x: torch.Tensor, m: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
    return (x - m) / s.clamp(min=1e-6)


def fit_probe(xtr: torch.Tensor, ytr: torch.Tensor, xev: torch.Tensor, yev: torch.Tensor,
              k: int) -> dict:
    """L2 正则多项式逻辑回归；C 按 probe-train 内部 80/20 选，再全量重拟合。"""
    n = len(ytr)
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(n, generator=g)
    nfit = int(n * 0.8)
    fi, vi = perm[:nfit], perm[nfit:]
    mu, sd = xtr.mean(0), xtr.std(0)

    def run(xa, ya, xb, yb, C):
        lam = 1.0 / (C * len(ya))
        w = torch.zeros(xa.shape[1], int(ya.max()) + 1, requires_grad=True)
        b = torch.zeros(w.shape[1], requires_grad=True)
        opt = torch.optim.LBFGS([w, b], lr=1.0, max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = torch.nn.functional.cross_entropy(zscore(xa, mu, sd) @ w + b, ya)
            loss = loss + 0.5 * lam * (w * w).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            acc = float((zscore(xb, mu, sd) @ w + b).argmax(1).eq(yb).float().mean())
        return w.detach(), b.detach(), acc

    best_C, best_a = CS[0], -1.0
    for C in CS:
        _, _, a = run(xtr[fi], ytr[fi], xtr[vi], ytr[vi], C)
        if a > best_a + 1e-12:
            best_C, best_a = C, a
    w, b, _ = run(xtr, ytr, xev, yev, best_C)
    with torch.no_grad():
        pred = (zscore(xev, mu, sd) @ w + b).argmax(1)
    acc = float(pred.eq(yev).float().mean())
    return {"acc": round(acc, 6), "se": round(se(acc, len(yev)), 6),
            "n_eval": len(yev), "C": best_C, "inner_val_acc": round(best_a, 6),
            "n_fit": n, "n_class": int(ytr.max()) + 1,
            "_pred": pred, "_fit": (w, b, mu, sd)}


def apply_probe(fit, x: torch.Tensor) -> torch.Tensor:
    w, b, mu, sd = fit if isinstance(fit, tuple) else fit["_fit"]
    with torch.no_grad():
        return (zscore(x, mu, sd) @ w + b).argmax(1)


def trig_label(rows: list[dict]) -> list[int]:
    out = []
    for r in rows:
        t = r["trig"].get("mask")
        out.append(0 if not t else 1 + min(7, t[0] // 8))
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43])
    a = ap.parse_args(argv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ds = D.build_dataset()
    vocab = ds["vocab"]
    fw = {vocab[c] for c in D.FUNCWORD_SET if c in vocab}
    model = Core(a.arm, len(vocab), fw, base_seed=a.seed)
    name = f"{a.arm}_s{a.seed}"
    sd = torch.load(WEIGHTS / f"{name}.pt", map_location="cpu")
    model.load_state_dict(sd)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)

    def T(key: str) -> torch.Tensor:
        return torch.tensor(ds[key]["ids"], dtype=torch.long)

    def Y(key: str) -> torch.Tensor:
        return torch.tensor(ds[key]["y"], dtype=torch.long)

    rep: dict = {"arm": a.arm, "seed": a.seed, "probes": {}}

    # ---- 1) skel（结构信息）----
    xtr = feats(model, T("P_skel_train"), device)
    xte = feats(model, T("P_skel_test"), device)
    ytr, yte = Y("P_skel_train"), Y("P_skel_test")
    r = fit_probe(xtr, ytr, xte, yte, 40)
    skel_fit = r["_fit"]
    r.pop("_pred")
    r["floor_majority"] = round(max(Counter(ytr.tolist()).values()) / len(ytr), 6)
    r["eval_majority"] = round(max(Counter(yte.tolist()).values()) / len(yte), 6)
    r["max_naive_free_rule"] = 0.8316      # free_rule_floor：n_slots→train多数（test 实测）
    r["note"] = "skel_id 是字面结构标签；地板取 free_rule_floor 完备电池 .8316"
    rep["probes"]["skel"] = r

    # ---- 2) trig_pos（位置信息，标签从未参与训练）----
    sm = D.load_sentence_mode()
    ytr_t = torch.tensor(trig_label(sm["train"]))
    yte_t = torch.tensor(trig_label(sm["test"]))
    xtr_t = feats(model, T("S_train"), device)
    xte_t = feats(model, T("S_test"), device)
    r = fit_probe(xtr_t, ytr_t, xte_t, yte_t, 9)
    r.pop("_pred")
    r["floor_majority_train"] = round(max(Counter(ytr_t.tolist()).values()) / len(ytr_t), 6)
    r["eval_majority"] = round(max(Counter(yte_t.tolist()).values()) / len(yte_t), 6)
    r["note"] = "触发词起点位置；对无位置编码的 A 置换不可学 ⇒ 应落多数类"
    rep["probes"]["trig_pos"] = r

    # ---- 3) ent（有真值信息）----
    xtr_e = feats(model, T("T_train"), device)
    xte_e = feats(model, T("T_test"), device)
    r = fit_probe(xtr_e, Y("T_train"), xte_e, Y("T_test"), 2)
    r.pop("_pred")
    r["max_naive_free_rule"] = 0.875       # rules_floor.json：R_edit_dist（P12a 四条同值）
    r["floor_majority"] = 0.5
    rep["probes"]["ent"] = r

    # ---- 4) b_pairs pair_success（结构敏感的表示级读出）----
    xb = feats(model, T("P_b_pairs"), device)
    pred = apply_probe(skel_fit, xb)          # 同一个 skel probe（fit = P_skel_train）
    yb = Y("P_b_pairs")
    assert len(pred) % 2 == 0 and len(pred) == 1120, len(pred)
    pairs = [(pred[2 * i].item(), pred[2 * i + 1].item(),
              yb[2 * i].item(), yb[2 * i + 1].item()) for i in range(len(pred) // 2)]
    ps = sum(1 for p, q, _, _ in pairs if p != q) / len(pairs)
    both = sum(1 for p, q, a_, b_ in pairs if p == a_ and q == b_) / len(pairs)
    per_side = float(pred.eq(yb).float().mean())
    xc = feats(model, T("P_c_pairs"), device)
    rc = fit_probe(xtr, ytr, xc, Y("P_c_pairs"), 40)
    rc.pop("_pred")
    rep["probes"]["b_pairs"] = {
        "pair_success": round(ps, 6), "se_pair": round(math.sqrt(ps * (1 - ps) / len(pairs)), 6),
        "n_pairs": len(pairs), "pair_chance": 0.5,
        "both_correct": round(both, 6), "per_side_acc": round(per_side, 6),
        "ref_perm_invariant_rule": 0.0, "ref_literal_skeleton_rule": 1.0,
        "per_side_floor_free_rule": 0.1404, "per_side_note": "a_bal 完备电池 .1404（skeleton_leak 实测）"}
    rep["probes"]["c_pairs"] = {"per_side_acc": rc["acc"], "se": rc["se"],
                                "n_eval": rc["n_eval"], "floor": 0.1788,
                                "floor_note": "skeleton_leak c 完备 max_naive .1788"}

    out = {k: ({kk: vv for kk, vv in v.items()
                if not kk.startswith("_")} if isinstance(v, dict) else v)
           for k, v in rep["probes"].items()}
    rep["probes"] = out
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"probe_{name}.json").write_text(
        json.dumps(rep, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in out.items():
        bits = " ".join(f"{kk}={vv}" for kk, vv in v.items()
                        if isinstance(vv, (int, float)) and kk in
                        ("acc", "se", "pair_success", "per_side_acc", "eval_majority",
                         "max_naive_free_rule", "floor_majority", "C"))
        print(f"[probe] {name} {k:9s} {bits}", flush=True)
    print(f"[done] probe_{name}.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
