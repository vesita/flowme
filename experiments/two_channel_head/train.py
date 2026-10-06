#!/usr/bin/env python3
"""two_channel_head 四臂训练 + 分通道评测（核全程冻结，只训头）。

配方 = PREREG §3（跑前写死）：
  **1800 步**、batch 64、AdamW lr 1e-3 / wd 1e-4、cosine、grad clip 1.0、seed 42/43。
  A 吃 1800 个指针 batch；B 吃 1800 个生成 batch；
  C/D 每步各 1 个指针 batch + 1 个生成 batch，`loss = L_ptr + L_gen`（λ=1）。
  两条 DataLoader 各自用 seed 播种 ⇒ A 与 C 的指针 batch 序列、B 与 C 的生成 batch 序列**逐位相同**。

随机标签对照（W3）：`--randlabel` 只打乱 **train** 标签（指针 = label randperm；生成 = 骨架 randperm +
每行指派值行内 randperm），test 仍用真标签评测；跑前打印 randperm 前后计数。

计时（PREREG §3）：**真实前向内量** —— 训练中逐步包住 forward（`cuda.synchronize` + `perf_counter`），
训练后再用**同一批真实 batch、同一 forward** 跑 N=200 次探针，并与训练逐步墙钟对账；
`--timing-only` 可在**独立单跑**里只跑探针（单跑对账）。

用法：
  uv run python experiments/two_channel_head/train.py --arm C --seed 42
  uv run python experiments/two_channel_head/train.py --arm A --seed 42 --randlabel
  uv run python experiments/two_channel_head/train.py --arm D --seed 43 --timing-only
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
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from model import (Spec, TwoChannelModel, encode_gen, load_ptr_blob,  # noqa: E402
                   ptr_row_meta)

DATA = HERE / "data"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"

STEPS = 1800
BATCH = 64
LR = 1e-3
WD = 1e-4
CLIP = 1.0
PROBE_N = 200
PROBE_WARMUP = 20
NEEDS_PTR = {"A", "C", "D"}
NEEDS_GEN = {"B", "C", "D"}


def sync(dev: str) -> None:
    if dev.startswith("cuda"):
        torch.cuda.synchronize()


def load_gen_rows(split: str) -> list[dict]:
    p = DATA / f"{split}.jsonl"
    if not p.exists():
        raise SystemExit(f"缺生成分支数据：{p}（先跑 build_gen_data.py）")
    with open(p, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def gen_tensors(rows: list[dict], blob: dict) -> tuple:
    v_items, mask = blob["v_items"], blob["item_mask"]
    v_bag = (v_items * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
    return (blob["v_sent"], v_bag, v_items, mask, blob["skel"], blob["assign"])


def randlabel_ptr(labels: torch.Tensor, n_train: int, seed: int) -> torch.Tensor:
    labels = labels.clone()
    g = torch.Generator().manual_seed(seed * 1000 + 7)
    idx = torch.arange(n_train)
    labels[idx] = labels[idx][torch.randperm(n_train, generator=g)]
    return labels


def randlabel_gen(skel: torch.Tensor, assign: torch.Tensor, mask: torch.Tensor,
                  n_train: int, seed: int) -> tuple:
    skel, assign = skel.clone(), assign.clone()
    g = torch.Generator().manual_seed(seed * 1000 + 13)
    idx = torch.arange(n_train)
    skel[idx] = skel[idx][torch.randperm(n_train, generator=g)]
    n = mask.sum(1).long()
    for i in range(n_train):
        k = int(n[i])
        perm = torch.randperm(k, generator=g)
        assign[i, :k] = assign[i, :k][perm]
    return skel, assign


def probe(fn, tensors: list, dev: str, n: int = PROBE_N,
          warmup: int = PROBE_WARMUP) -> dict:
    """真实前向探针：与训练完全相同的 forward / 相同 batch / 相同设备。"""
    for _ in range(warmup):
        out = fn(*tensors)
        del out
    sync(dev)
    ts = []
    for _ in range(n):
        sync(dev)
        t0 = time.perf_counter()
        out = fn(*tensors)
        del out
        sync(dev)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"n": n, "mean_ms": round(sum(ts) / len(ts), 4),
            "p50_ms": round(ts[len(ts) // 2], 4), "p95_ms": round(ts[int(len(ts) * 0.95)], 4)}


def evaluate_ptr(model, blob, mask, dev, bs: int = 4096) -> tuple[dict, list[int]]:
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    v_ctx, v_cand, y = blob["v_ctx"][idx], blob["v_cand"][idx], blob["labels"][idx]
    correct: list[int] = []
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            c = v_ctx[i:i + bs].to(dev)
            d = v_cand[i:i + bs].to(dev)
            t = y[i:i + bs].to(dev)
            logits = model.forward_ptr(c, d)
            loss += torch.nn.functional.cross_entropy(logits, t, reduction="sum").item()
            correct += (logits.argmax(-1) == t).cpu().tolist()
    n = len(idx)
    acc = sum(correct) / n
    se = math.sqrt(0.25 / n)
    return {"n": n, "acc": round(acc, 6), "ce": round(loss / n, 6),
            "se": round(se, 6), "margin_over_se": round((acc - 0.5) / se, 3),
            "blind_guess": 0.5}, correct


def evaluate_gen(model, blob, dev, bs: int = 4096) -> tuple[dict, dict, list]:
    """返回 (指标, 逐样本正确性, 逐样本明细)。骨架 / 指派 / 联合**分开报**。"""
    n_rows = blob["n"]
    skel_ok, slot_frac, joint_ok = [], [], []
    loss = 0.0
    model.eval()
    with torch.no_grad():
        for i in range(0, n_rows, bs):
            j = min(i + bs, n_rows)
            v_sent = blob["v_sent"][i:j].to(dev)
            v_items = blob["v_items"][i:j].to(dev)
            im = blob["item_mask"][i:j].to(dev)
            v_bag = (v_items * im.unsqueeze(-1)).sum(1) / im.sum(1, keepdim=True).clamp(min=1)
            skel_y = blob["skel"][i:j].to(dev)
            asg_y = blob["assign"][i:j].to(dev)
            skel_logits, a_logits = model.forward_gen(v_sent, v_bag, v_items, im)
            loss += torch.nn.functional.cross_entropy(skel_logits, skel_y,
                                                      reduction="sum").item()
            pred_skel = skel_logits.argmax(-1).cpu().tolist()
            sy = skel_y.cpu().tolist()
            ay = asg_y.cpu().tolist()
            am = im.cpu()
            apred = a_logits.argmax(-1).cpu()
            for r in range(j - i):
                k = int(am[r].sum())
                right = sum(int(apred[r, s]) == ay[r][s] for s in range(k))
                slot_frac.append(right / k)
                skel_ok.append(int(pred_skel[r] == sy[r]))
                joint_ok.append(int(pred_skel[r] == sy[r] and right == k))
    n = len(skel_ok)
    se = math.sqrt(0.25 / n)
    se_slot = math.sqrt(max(1e-12, sum((x - sum(slot_frac) / n) ** 2 for x in slot_frac)) / (n - 1)) / math.sqrt(n)
    out = {
        "skel": {"n": n, "acc": round(sum(skel_ok) / n, 6), "se": round(se, 6),
                 "margin_over_se": round((sum(skel_ok) / n - 1 / 40) / se, 3),
                 "blind_guess": 0.025},
        "slot": {"n": n, "acc": round(sum(slot_frac) / n, 6), "se": round(se_slot, 6)},
        "joint": {"n": n, "acc": round(sum(joint_ok) / n, 6), "se": round(se, 6)},
        "ce": round(loss / n, 6),
    }
    return out, {"skel": skel_ok, "slot_frac": [round(x, 6) for x in slot_frac],
                 "joint": joint_ok}, None


def train_one(arm: str, seed: int, randlabel: bool = False, device: str | None = None,
              timing_only: bool = False, steps: int = STEPS) -> dict:
    t_start = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)
    spec = Spec()
    model = TwoChannelModel(arm, seed, spec).to(device)
    freeze = model.freeze_report()
    params = model.param_report()
    print(f"[freeze] {json.dumps(freeze, ensure_ascii=False)}", flush=True)
    print(f"[params] {json.dumps(params, ensure_ascii=False)}", flush=True)
    print(f"[route] 手写开关（PREREG §5 原文）：有候选列表(len≥2) ⇒ 指针通道；"
          f"无候选但 match_sentence(text) 命中 ⇒ 生成通道；都不满足 ⇒ fail-closed 拒答",
          flush=True)

    ptr_blob = gen_blob = None
    ptr_rows = gen_rows = None
    n_ptr_train = n_gen_train = 0
    if arm in NEEDS_PTR:
        ptr_blob = load_ptr_blob()
        n_ptr_train = int((ptr_blob["splits"] == 0).sum())
        if randlabel:
            before = torch.bincount(ptr_blob["labels"][:n_ptr_train].cpu(), minlength=2).tolist()
            ptr_blob["labels"] = randlabel_ptr(ptr_blob["labels"], n_ptr_train, seed)
            after = torch.bincount(ptr_blob["labels"][:n_ptr_train].cpu(), minlength=2).tolist()
            print(f"[randlabel] 指针 train 标签 randperm：{before} → {after}（应保持 1:1）",
                  flush=True)
    if arm in NEEDS_GEN:
        gen_rows = load_gen_rows("train")
        n_gen_train = len(gen_rows)
        gen_blob = encode_gen(model, gen_rows, spec, device)
        if randlabel:
            before = torch.bincount(gen_blob["skel"]).tolist()
            gen_blob["skel"], gen_blob["assign"] = randlabel_gen(
                gen_blob["skel"], gen_blob["assign"], gen_blob["item_mask"],
                n_gen_train, seed)
            after = torch.bincount(gen_blob["skel"]).tolist()
            print(f"[randlabel] 生成 train 骨架 randperm：前 6 类 {before[:6]} → "
                  f"{after[:6]}（分布不变）；每行指派值行内 randperm", flush=True)

    if timing_only:
        rep = run_timing_probe(model, ptr_blob, gen_blob, device, n_ptr_train, n_gen_train)
        rep.update({"name": f"{arm}_s{seed}{'_rand' if randlabel else ''}_timing",
                    "arm": arm, "seed": seed, "randlabel": randlabel,
                    "timing_only": True, "freeze": freeze, "params": params})
        print(json.dumps(rep, ensure_ascii=False, indent=2), flush=True)
        return rep

    # ---- loaders（两条各自用 seed 播种 ⇒ A/C 指针序列一致、B/C 生成序列一致）----
    loaders = {}
    if arm in NEEDS_PTR:
        tr = ptr_blob["splits"] == 0
        ds = TensorDataset(ptr_blob["v_ctx"][tr], ptr_blob["v_cand"][tr],
                           ptr_blob["labels"][tr])
        loaders["ptr"] = DataLoader(ds, batch_size=BATCH, shuffle=True,
                                    generator=torch.Generator().manual_seed(seed))
    if arm in NEEDS_GEN:
        vs, vb, vi, im, sk, asg = gen_tensors(gen_rows, gen_blob)
        ds = TensorDataset(vs, vb, vi, im, sk, asg)
        loaders["gen"] = DataLoader(ds, batch_size=BATCH, shuffle=True,
                                    generator=torch.Generator().manual_seed(seed))

    head_params = [p for p in model.parameters() if p.requires_grad]
    ptr_group = [p for n, p in model.named_parameters()
                 if p.requires_grad and ("trunk_ptr" in n or n.startswith("ptr_out"))]
    gen_group = [p for n, p in model.named_parameters()
                 if p.requires_grad and ("trunk_gen" in n or n.startswith("gen"))]
    clip_groups = [g for g in (ptr_group, gen_group) if g]   # C 用联合分支，不走这里
    opt = torch.optim.AdamW(head_params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    it = {k: iter(v) for k, v in loaders.items()}

    def next_batch(k):
        try:
            return next(it[k])
        except StopIteration:
            it[k] = iter(loaders[k])
            return next(it[k])

    fwd_t = {"ptr": [], "gen": []}
    step_t = []
    warm_skip = min(50, max(1, steps // 4))     # ROCm 首步 JIT/自动调参会污染均值
    losses = []
    ch_loss = {"ptr": [], "gen": []}            # 分通道 loss（分开报，不合成）
    model.train()
    for step in range(steps):
        sync(device)
        t_step = time.perf_counter()
        lp = lg = None
        if arm in NEEDS_PTR:
            c, d, y = next_batch("ptr")
            c, d, y = c.to(device), d.to(device), y.to(device)
            sync(device)
            t0 = time.perf_counter()
            lp = model.ptr_loss(c, d, y)
            ch_loss["ptr"].append(float(lp.detach()))
            sync(device)
            fwd_t["ptr"].append((time.perf_counter() - t0) * 1000)
        if arm in NEEDS_GEN:
            vs, vb, vi, im, sk, asg = next_batch("gen")
            vs, vb, vi, im, sk, asg = (x.to(device) for x in (vs, vb, vi, im, sk, asg))
            sync(device)
            t0 = time.perf_counter()
            lg_all = model.gen_loss(vs, vb, vi, im, sk, asg)
            lg = lg_all[0]
            ch_loss["gen"].append(float(lg.detach()))
            sync(device)
            fwd_t["gen"].append((time.perf_counter() - t0) * 1000)
        loss = (lp if lp is not None else 0) + (lg if lg is not None else 0)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if arm == "C":
            torch.nn.utils.clip_grad_norm_(head_params, CLIP)   # 共享主干 ⇒ 全头联合
        else:
            for grp in clip_groups:                              # 逐头独立（D 无交叉耦合）
                torch.nn.utils.clip_grad_norm_(grp, CLIP)
        opt.step()
        sched.step()
        sync(device)
        step_t.append((time.perf_counter() - t_step) * 1000)
        losses.append(float(loss.detach()))
    print(f"[train] {steps} 步 done，loss 首/末 {losses[0]:.4f}/{losses[-1]:.4f}", flush=True)

    # ---- 计时探针（真实前向，同 batch、同设备）----
    timing = run_timing_probe(model, ptr_blob, gen_blob, device, n_ptr_train, n_gen_train)
    def stats(v: list) -> dict:
        v = sorted(v)
        return {"n": len(v), "mean": round(sum(v) / len(v), 4),
                "p50": round(v[len(v) // 2], 4),
                "p95": round(v[int(len(v) * 0.95)], 4)}
    timing["train_warm_skip"] = warm_skip
    timing["train_fwd_ms"] = {k: stats(v[warm_skip:]) for k, v in fwd_t.items() if v}
    timing["train_step_ms"] = stats(step_t[warm_skip:])
    timing["loss_by_channel"] = {
        k: {"n": len(v), "first": round(v[0], 6), "last": round(v[-1], 6),
            "mean_over_all": round(sum(v) / len(v), 6)}
        for k, v in ch_loss.items() if v}
    timing["train_step_all_ms"] = stats(step_t)

    # ---- 分通道评测（指标分开，不合成一个数）----
    eval_out: dict = {}
    correct: dict = {}
    if arm in NEEDS_PTR:
        splits = {s: ptr_blob["splits"] == i for i, s in enumerate(("train", "test", "adv"))}
        for s in ("train", "test", "adv"):
            m, c = evaluate_ptr(model, ptr_blob, splits[s], device)
            eval_out[f"ptr_{s}"] = m
            correct[f"ptr_{s}"] = c
            print(f"[eval:ptr] {arm}{'_rand' if randlabel else ''} s{seed} {s:5s} "
                  f"n={m['n']} acc={m['acc']:.4f} (SE={m['se']:.4f} "
                  f"余量/SE={m['margin_over_se']:+.2f})", flush=True)
        meta = ptr_row_meta()
        adv_pos = [i for i, r in enumerate(meta) if r["split"] == "adv"]   # adv 内的位次
        for ct in ("shifted_pos", "heldout_pair"):
            sel = [k for k, i in enumerate(adv_pos) if meta[i]["ctype"] == ct]
            c = [correct["ptr_adv"][k] for k in sel]
            acc = sum(c) / len(c)
            se = math.sqrt(0.25 / len(c))
            eval_out[f"ptr_adv_{ct}"] = {"n": len(c), "acc": round(acc, 6),
                                         "se": round(se, 6),
                                         "margin_over_se": round((acc - 0.5) / se, 3)}
            correct[f"ptr_adv_{ct}"] = c
            print(f"[eval:ptr]   adv/{ct} n={len(c)} acc={acc:.4f} "
                  f"(SE={se:.4f} 余量/SE={(acc - 0.5) / se:+.2f})", flush=True)
    if arm in NEEDS_GEN:
        test_blob = encode_gen(model, load_gen_rows("test"), spec, device)
        m, c, _ = evaluate_gen(model, test_blob, device)
        eval_out["gen_test"] = m
        correct["gen_test"] = c
        print(f"[eval:gen] {arm}{'_rand' if randlabel else ''} s{seed} 骨架 "
              f"acc={m['skel']['acc']:.4f} (SE={m['skel']['se']:.4f} 余量/SE="
              f"{m['skel']['margin_over_se']:+.2f}) | 指派 {m['slot']['acc']:.4f} | "
              f"联合 {m['joint']['acc']:.4f}", flush=True)

    name = f"{arm}_s{seed}{'_rand' if randlabel else ''}"
    if steps != STEPS:
        name += f"_steps{steps}"          # 冒烟跑不与正式跑同名
    out = {
        "name": name, "arm": arm, "seed": seed, "randlabel": randlabel,
        "device": device, "freeze": freeze, "params": params,
        "recipe": {"steps": steps, "batch": BATCH, "lr": LR, "weight_decay": WD,
                   "cosine": True, "grad_clip": CLIP,
                   "n_ptr_train": n_ptr_train, "n_gen_train": n_gen_train,
                   "loss": "L_ptr + L_gen (λ=1)" if arm in ("C", "D") else
                   ("L_ptr" if arm == "A" else "L_gen")},
        "eval": eval_out,
        "timing": timing,
        "loss_first": round(losses[0], 6), "loss_last": round(losses[-1], 6),
        "loss_by_channel": timing["loss_by_channel"],
        "wall_sec": round(time.time() - t_start, 1),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(
        json.dumps({"meta": out, "correct": correct}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (RESULTS / f"{name}.summary.json").write_text(json.dumps(out, ensure_ascii=False,
                                                             indent=2), encoding="utf-8")
    WEIGHTS.mkdir(exist_ok=True)
    torch.save({k: v for k, v in model.state_dict().items()
                if not k.startswith("encoder")}, WEIGHTS / f"{name}.pt")
    print(f"[done] {name} → results/{name}.json ({out['wall_sec']}s)", flush=True)
    return out


def run_timing_probe(model, ptr_blob, gen_blob, device, n_ptr_train, n_gen_train) -> dict:
    """与训练同构的真实前向探针：同一 forward、同一批真实 batch、同一设备。

    同时跑「前向+反向+优化器」的整步探针，供与训练逐步墙钟对账（PREREG §3）。
    """
    rep: dict = {}
    ptr_args = gen_args = None
    if model.arm in NEEDS_PTR and ptr_blob is not None:
        ptr_args = (ptr_blob["v_ctx"][:BATCH].to(device),
                    ptr_blob["v_cand"][:BATCH].to(device))
        rep["probe_fwd_ptr_ms"] = probe(model.forward_ptr, list(ptr_args), device)
    if model.arm in NEEDS_GEN and gen_blob is not None:
        vi = gen_blob["v_items"][:BATCH].to(device)
        im = gen_blob["item_mask"][:BATCH].to(device)
        vb = (vi * im.unsqueeze(-1)).sum(1) / im.sum(1, keepdim=True).clamp(min=1)
        gen_args = (gen_blob["v_sent"][:BATCH].to(device), vb, vi, im)
        rep["probe_fwd_gen_ms"] = probe(model.forward_gen, list(gen_args), device)

    if ptr_args is None and gen_args is None:
        return rep

    params = [p for p in model.parameters() if p.requires_grad]

    def full_step():
        loss = None
        if ptr_args is not None:
            y = torch.randint(0, 2, (BATCH,), device=device)
            loss = model.ptr_loss(ptr_args[0], ptr_args[1], y)
        if gen_args is not None:
            sk = torch.randint(0, model.spec.n_skel, (BATCH,), device=device)
            asg = torch.randint(0, model.spec.max_slots,
                                (BATCH, model.spec.max_slots), device=device)
            gl = model.gen_loss(*gen_args, sk, asg)[0]
            loss = gl if loss is None else loss + gl
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, CLIP)
        for p in params:
            p.grad = None

    rep["probe_step_fwd_bwd_ms"] = _probe_plain(full_step, device, n=50, warmup=10)
    rep["probe_n"] = PROBE_N
    return rep


def spec_n_skel(model) -> int:
    return model.spec.n_skel


def spec_max_slots(model) -> int:
    return model.spec.max_slots


def _probe_plain(fn, dev, n=50, warmup=10) -> dict:
    for _ in range(warmup):
        fn()
    sync(dev)
    ts = []
    for _ in range(n):
        sync(dev)
        t0 = time.perf_counter()
        fn()
        sync(dev)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return {"n": n, "mean_ms": round(sum(ts) / len(ts), 4),
            "p50_ms": round(ts[len(ts) // 2], 4), "p95_ms": round(ts[int(len(ts) * .95)], 4)}


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list("ABCD"))
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43])
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--timing-only", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS,
                    help="仅冒烟测试用；非默认值会在结果名后缀 _stepsN")
    a = ap.parse_args(argv)
    WEIGHTS.mkdir(exist_ok=True)
    RESULTS.mkdir(exist_ok=True)
    train_one(a.arm, a.seed, a.randlabel, a.device, a.timing_only, a.steps)


if __name__ == "__main__":
    main(sys.argv[1:])
