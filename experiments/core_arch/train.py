#!/usr/bin/env python3
"""P20 训练 + 判据实测（三臂 × 2 seed；同数据、同步数、同优化器；全部从头训）。

用法：
  uv run python experiments/core_arch/train.py --arm A --seed 42
  uv run python experiments/core_arch/train.py --arm A --seed 42 --steps 40 --tag smoke
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import data as D  # noqa: E402
from model_core import ARM_DESC, ARMS, assert_arm_isolation, build_all, count_params  # noqa: E402

RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
STEPS = 1200
BATCH = 32          # 每任务每步 32 行 ⇒ 总 batch 64
LR = 1e-3
WD = 1e-4
CLIP = 1.0
SEEDS = (42, 43)


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


@torch.no_grad()
def predict(model, ids: torch.Tensor, task: str, device: str, bs: int = 1024) -> torch.Tensor:
    model.eval()
    out = []
    for i in range(0, len(ids), bs):
        out.append(model(ids[i:i + bs].to(device), task).argmax(-1).cpu())
    return torch.cat(out)


def acc_of(pred: torch.Tensor, y: torch.Tensor) -> float:
    return float((pred == y).float().mean())


def permute_rows(ids_list: list[list[int]], seed: int) -> list[list[int]]:
    """逐行随机置换**非 pad** 前缀（保证字符多重集不变 ⇒ A 必须预测不变）。"""
    g = random.Random(seed * 1000 + 11)
    out = []
    for row in ids_list:
        n = row.index(D.PAD) if D.PAD in row else len(row)
        pre = row[:n]
        g.shuffle(pre)
        out.append(pre + row[n:])
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True, choices=list(SEEDS))
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--tag", default="")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)

    t0 = time.time()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(a.seed)
    random.seed(a.seed)

    ds = D.build_dataset()
    vocab = ds["vocab"]
    fw_ids = {vocab[c] for c in D.FUNCWORD_SET if c in vocab}
    print(f"[data] vocab={len(vocab)} fw_ids={len(fw_ids)} meta={json.dumps(ds['meta'], ensure_ascii=False)}",
          flush=True)

    # ---- 构造性自检：三臂差异只在声明部位 ----
    models = build_all(len(vocab), fw_ids, base_seed=a.seed)
    selfcheck = assert_arm_isolation(models)
    print(f"[selfcheck] {json.dumps(selfcheck, ensure_ascii=False)}", flush=True)
    model = models[a.arm].to(device)
    pc = count_params(model)
    print(f"[params] {a.arm} total={pc['total']} {pc['per_group']}", flush=True)

    def T(key: str) -> torch.Tensor:
        return torch.tensor(ds[key]["ids"], dtype=torch.long)

    def Y(key: str) -> torch.Tensor:
        return torch.tensor(ds[key]["y"], dtype=torch.long)

    S_tr, S_tr_y = T("S_train"), Y("S_train")
    T_tr, T_tr_y = T("T_train"), Y("T_train")
    S_te, S_te_y = T("S_test"), Y("S_test")
    T_te, T_te_y = T("T_test"), Y("T_test")

    # ---- 训练 ----
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    gs, gt = torch.Generator().manual_seed(a.seed), torch.Generator().manual_seed(a.seed + 1)
    s_order = torch.randperm(len(S_tr), generator=gs)
    t_order = torch.randperm(len(T_tr), generator=gt)
    losses: list[float] = []
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    t_train = time.time()
    model.train()
    for step in range(a.steps):
        si = s_order[(step * BATCH) % len(S_tr):(step * BATCH) % len(S_tr) + BATCH]
        ti = t_order[(step * BATCH) % len(T_tr):(step * BATCH) % len(T_tr) + BATCH]
        if len(si) < BATCH:                      # 回卷
            si = torch.cat([si, s_order[:BATCH - len(si)]])
        if len(ti) < BATCH:
            ti = torch.cat([ti, t_order[:BATCH - len(ti)]])
        ls = torch.nn.functional.cross_entropy(model(S_tr[si].to(device), "S"), S_tr_y[si].to(device))
        lt = torch.nn.functional.cross_entropy(model(T_tr[ti].to(device), "T"), T_tr_y[ti].to(device))
        loss = ls + lt
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        if step == 0:
            model.eval()
            with torch.no_grad():
                f0 = float(loss.detach())
            model.train()
    train_wall = time.time() - t_train
    assert all(math.isfinite(x) for x in losses), "loss 非有限值"
    k = min(50, len(losses) // 2)
    if a.steps >= 100:                     # 冒烟档不设收敛断言（PREREG §3：冒烟 steps≤40）
        assert sum(losses[-k:]) / k < sum(losses[:k]) / k, \
            f"loss 未降：前 {k} {sum(losses[:k])/k:.4f} 后 {k} {sum(losses[-k:])/k:.4f}"

    # ---- 评测 ----
    detail: dict = {}
    ev: dict = {}
    # Task S：三个出口
    for ek, key in (("mask", "S_test"), ("keep", "S_test"), ("maskfinal", "S_test")):
        ids = torch.tensor(ds[key][f"ids_{ek}"] if ek != "mask" else ds[key]["ids"],
                           dtype=torch.long)
        pred = predict(model, ids, "S", device)
        correct = (pred == S_te_y).int().tolist()
        p = acc_of(pred, S_te_y)
        ev[f"S_test_{ek}"] = {"n": len(correct), "acc": round(p, 6), "se": round(se(p, len(correct)), 6)}
        detail[f"S_test_{ek}_correct"] = correct
    # Task S train（表内 H5）
    pred = predict(model, S_tr, "S", device)
    p = acc_of(pred, S_tr_y)
    ev["S_train_mask"] = {"n": len(S_tr_y), "acc": round(p, 6), "se": round(se(p, len(S_tr_y)), 6)}
    # Task T
    pred = predict(model, T_te, "T", device)
    correct = (pred == T_te_y).int().tolist()
    p = acc_of(pred, T_te_y)
    ev["T_test"] = {"n": len(correct), "acc": round(p, 6), "se": round(se(p, len(correct)), 6)}
    detail["T_test_correct"] = correct
    pred = predict(model, T_tr, "T", device)
    p = acc_of(pred, T_tr_y)
    ev["T_train"] = {"n": len(T_tr_y), "acc": round(p, 6), "se": round(se(p, len(T_tr_y)), 6)}
    # Task T 按 variant / type 拆（H5 辅助：分布内子类型不塌）
    for field in ("variant", "type"):
        buckets: dict[str, list[bool]] = {}
        for v, c in zip(ds["T_test"][field], detail["T_test_correct"]):
            buckets.setdefault(str(v), []).append(bool(c))
        ev[f"T_test_by_{field}"] = {k: {"n": len(v), "acc": round(sum(v) / len(v), 6)}
                                    for k, v in sorted(buckets.items())}

    # ---- M 机制自检：逐行置换非 pad 前缀后重评 Task S mask ----
    shuf_ids = permute_rows(ds["S_test"]["ids"], a.seed)
    pred_shuf = predict(model, torch.tensor(shuf_ids, dtype=torch.long), "S", device)
    orig_pred = predict(model, S_te, "S", device)
    agree = float((pred_shuf == orig_pred).float().mean())
    p_shuf = acc_of(pred_shuf, S_te_y)
    ev["M_shuffle"] = {"n": len(S_te_y), "acc": round(p_shuf, 6),
                       "se": round(se(p_shuf, len(S_te_y)), 6),
                       "acc_delta": round(p_shuf - ev["S_test_mask"]["acc"], 6),
                       "pred_agree": round(agree, 6)}
    if a.arm == "A":
        assert agree >= 0.999 and abs(ev["M_shuffle"]["acc_delta"]) <= 0.002, \
            f"A 置换后预测改变：agree={agree} Δ={ev['M_shuffle']['acc_delta']}"

    peak = torch.cuda.max_memory_allocated() if device.startswith("cuda") else 0
    out = {
        "arm": a.arm, "seed": a.seed, "tag": a.tag, "device": device,
        "arm_desc": ARM_DESC[a.arm], "steps": a.steps, "batch_per_task": BATCH,
        "batch_total": BATCH * 2, "lr": LR, "wd": WD, "clip": CLIP,
        "recipe": {"steps": a.steps, "batch": BATCH * 2, "lr": LR, "weight_decay": WD,
                   "cosine": True, "grad_clip": CLIP, "loss": "CE_S + CE_T",
                   "optimizer": "AdamW", "seeds": list(SEEDS)},
        "params": pc, "selfcheck": selfcheck,
        "data_meta": ds["meta"], "vocab_meta": ds["vocab_meta"],
        "fw_ids": len(fw_ids),
        "eval": ev,
        "loss_first": round(losses[0], 6), "loss_last": round(losses[-1], 6),
        "loss_first50": round(sum(losses[:50]) / 50, 6),
        "loss_last50": round(sum(losses[-50:]) / 50, 6),
        "train_wall_sec": round(train_wall, 2),
        "step_ms": round(train_wall * 1000 / max(1, a.steps), 3),
        "wall_sec": round(time.time() - t0, 1),
        "peak_mem_bytes": int(peak),
    }
    RESULTS.mkdir(exist_ok=True)
    WEIGHTS.mkdir(exist_ok=True)
    name = f"{a.arm}_s{a.seed}{('_' + a.tag) if a.tag else ''}"
    (RESULTS / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    (RESULTS / f"detail_{name}.json").write_text(
        json.dumps({"detail": detail}, ensure_ascii=False), encoding="utf-8")
    torch.save(model.state_dict(), WEIGHTS / f"{name}.pt")
    print(f"[eval] {name} " + " ".join(
        f"{k}={v['acc']:.4f}" for k, v in ev.items() if isinstance(v, dict) and "acc" in v),
        flush=True)
    print(f"[done] {name} params={pc['total']} step_ms={out['step_ms']} "
          f"wall={out['wall_sec']}s peak={peak/1e9:.2f}GB → results/{name}.json", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
