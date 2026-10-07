#!/usr/bin/env python3
"""P4 三臂训练（核全程冻结，只训头）。

配方 = PREREG §2.3（跑前写死）：1800 步、batch 64、AdamW lr 1e-3 / wd 1e-4、cosine、
grad clip 1.0、seed 42/43；DataLoader `generator=manual_seed(seed)` ⇒ 三臂 batch 序逐位相同。

  M   = CE_skel(row) + CE_assign(row)          （= struct_supervision B 臂，P0 复现对象）
  MP  = 最小对判别式 + CE_assign(row)
  MP+ = M + 1.0 × MP_pair                       （λ=1 写死）

用法：
  uv run python experiments/funcword_minpair/train.py --arm M --seed 42
  uv run python experiments/funcword_minpair/train.py --arm MP --seed 42 --randlabel
  uv run python experiments/funcword_minpair/train.py --arm MP+ --seed 42 --steps 60   # 冒烟
"""
from __future__ import annotations

import argparse
import json
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

from model import (ARMS, StructSupModel, Spec, arm_loss, check, get_blob,  # noqa: E402
                   v_bag_of)

TCH_DATA = ROOT / "experiments" / "two_channel_head" / "data"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
DATA = HERE / "data"

STEPS = 1800
BATCH = 64
LR = 1e-3
WD = 1e-4
CLIP = 1.0
SEEDS = (42, 43)


def load_rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def zero_pack(spec: Spec) -> dict:
    return {"v_sent": torch.zeros(1, spec.max_len_sent and 0 or 0)}   # 占位（不用）


def pack(rows: list[dict], blob: dict, spec: Spec) -> dict:
    """把一批行的编码堆成 [N,...] 张量（用于伙伴变体）。"""
    return {"v_sent": blob["v_sent"], "v_items": blob["v_items"],
            "item_mask": blob["item_mask"], "skel": blob["skel"]}


def build_variants(model, spec, device, seed: int | None = None):
    """读 train_pairs / train_order_pairs，编码伙伴变体（本目录缓存）。"""
    word_slot = [json.loads(x)["variant"] for x in
                 open(DATA / "train_pairs.jsonl", encoding="utf-8")]
    order_map = {}
    for x in open(DATA / "train_order_pairs.jsonl", encoding="utf-8"):
        rec = json.loads(x)
        order_map[rec["idx"]] = rec["variant"]
    n = len(word_slot)
    widx = [i for i, v in enumerate(word_slot) if v]
    oidx = sorted(order_map)
    wrows = [word_slot[i] for i in widx]
    orows = [order_map[i] for i in oidx]
    wb = get_blob(model, wrows, spec, device, "train_var_word") if wrows else None
    ob = get_blob(model, orows, spec, device, "train_var_order") if orows else None

    Zs = torch.zeros(n, spec.hidden)
    Zi = torch.zeros(n, spec.max_slots, spec.hidden)
    Zm = torch.zeros(n, spec.max_slots, dtype=torch.bool)
    Zy = torch.zeros(n, dtype=torch.long)
    vsv, viv, imv, yvw = Zs.clone(), Zi.clone(), Zm.clone(), Zy.clone()
    vso, vio, imo, yvo = Zs.clone(), Zi.clone(), Zm.clone(), Zy.clone()
    hw = torch.zeros(n, dtype=torch.bool)
    ho = torch.zeros(n, dtype=torch.bool)
    if wb is not None:
        for k, i in enumerate(widx):
            vsv[i] = wb["v_sent"][k]
            viv[i] = wb["v_items"][k]
            imv[i] = wb["item_mask"][k]
            yvw[i] = wb["skel"][k]
            hw[i] = True
    if ob is not None:
        for k, i in enumerate(oidx):
            vso[i] = ob["v_sent"][k]
            vio[i] = ob["v_items"][k]
            imo[i] = ob["item_mask"][k]
            yvo[i] = ob["skel"][k]
            ho[i] = True
    # 断言：伙伴 gold ≠ 行 gold
    return {"vsv": vsv, "vbv": v_bag_of(viv, imv), "viv": viv, "imv": imv,
            "yv_word": yvw, "has_word": hw,
            "vso": vso, "vbo": v_bag_of(vio, imo), "vio": vio, "imo": imo,
            "yv_order": yvo, "has_order": ho,
            "n_word": int(hw.sum()), "n_order": int(ho.sum())}


def randlabel(y: torch.Tensor, vw: torch.Tensor, vo: torch.Tensor,
              has_w: torch.Tensor, has_o: torch.Tensor, assign: torch.Tensor,
              mask: torch.Tensor, seed: int) -> tuple:
    """类级双射重标（30 个 train 骨架 id 上 randperm）+ 每行指派值行内 randperm。"""
    trained = sorted(set(y.tolist()))
    g = torch.Generator().manual_seed(seed * 1000 + 13)
    perm = torch.randperm(len(trained), generator=g)
    lut = torch.zeros(max(trained) + 1, dtype=torch.long)
    for src, dst in zip(trained, [trained[int(k)] for k in perm]):
        lut[src] = dst
    before = torch.bincount(y, minlength=40).tolist()
    y2 = lut[y]
    vw2 = torch.where(has_w, lut[vw], vw)
    vo2 = torch.where(has_o, lut[vo], vo)
    after = torch.bincount(y2, minlength=40).tolist()
    n = mask.sum(1).long()
    a2 = assign.clone()
    for i in range(len(a2)):
        k = int(n[i])
        if k <= 1:
            continue
        p = torch.randperm(k, generator=g)
        a2[i, :k] = a2[i, :k][p]
    return y2, vw2, vo2, a2, before, after


def train_one(arm: str, seed: int, rand: bool = False, device: str | None = None,
              steps: int = STEPS) -> dict:
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)
    spec = Spec()
    model = StructSupModel("B", seed, spec, vocab=None).to(device)   # 三臂同构造
    freeze = model.freeze_report()
    check(freeze["encoder_trainable"] == 0, "核没冻住")
    print(f"[freeze] {json.dumps(freeze, ensure_ascii=False)}", flush=True)

    rows_tr = load_rows(TCH_DATA / "train.jsonl")
    blob_tr = get_blob(model, rows_tr, spec, device, "train")
    var = build_variants(model, spec, device)
    print(f"[var] word 伙伴 {var['n_word']}/{len(rows_tr)}，order 伙伴 {var['n_order']}",
          flush=True)
    check(var["n_word"] == 4205, f"word 伙伴数漂移 {var['n_word']}")
    check(var["n_order"] == 83, f"order 伙伴数漂移 {var['n_order']}")

    vs = blob_tr["v_sent"]
    vi = blob_tr["v_items"]
    im = blob_tr["item_mask"]
    vb = v_bag_of(vi, im)
    y = blob_tr["skel"].clone()
    asg = blob_tr["assign"].clone()

    if rand:
        y, var["yv_word"], var["yv_order"], asg, before, after = randlabel(
            y, var["yv_word"], var["yv_order"], var["has_word"], var["has_order"],
            asg, im, seed)
        print(f"[randlabel] 骨架类级 randperm：前 6 类 {before[:6]} → {after[:6]}"
              f"；分布不变={sorted(before) == sorted(after)}；每行指派值行内 randperm",
              flush=True)
        check(sorted(before) == sorted(after), "randlabel 改变了标签分布")

    ds = TensorDataset(vs, vb, vi, im, y, asg,
                       var["vsv"], var["vbv"], var["viv"], var["imv"], var["yv_word"],
                       var["has_word"],
                       var["vso"], var["vbo"], var["vio"], var["imo"], var["yv_order"],
                       var["has_order"])
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))
    KEYS = ("vs", "vb", "vi", "im", "y", "asg", "vsv", "vbv", "viv", "imv",
            "yv_word", "has_word", "vso", "vbo", "vio", "imo", "yv_order", "has_order")

    head_params = [p for p in model.parameters() if p.requires_grad]
    n_head = sum(p.numel() for p in head_params)
    print(f"[params] 可训参数 {n_head}（trunk+gen），核 1,688,460 冻结", flush=True)
    opt = torch.optim.AdamW(head_params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    it = iter(loader)

    def nxt():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    losses, part_hist = [], {}
    model.train()
    for step in range(steps):
        b = [x.to(device) for x in nxt()]
        bd = dict(zip(KEYS, b))
        loss, parts = arm_loss(model, arm, bd)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, CLIP)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        for k, v in parts.items():
            part_hist.setdefault(k, []).append(float(v))
        if step == 0:
            print(f"[step0] {arm} s{seed} loss={losses[0]:.4f} "
                  f"parts={{{', '.join(f'{k}:{float(v):.4f}' for k, v in parts.items())}}}",
                  flush=True)
    print(f"[train] {steps} 步 done，loss 首/末 {losses[0]:.4f}/{losses[-1]:.4f}",
          flush=True)
    model.eval()

    name = f"{arm}_s{seed}{'_rand' if rand else ''}"
    if steps != STEPS:
        name += f"_steps{steps}"
    RESULTS.mkdir(exist_ok=True)
    WEIGHTS.mkdir(exist_ok=True)
    torch.save({k: v for k, v in model.state_dict().items()
                if not k.startswith("encoder")}, WEIGHTS / f"{name}.pt")
    out = {"name": name, "arm": arm, "seed": seed, "randlabel": rand,
           "device": device, "freeze": freeze,
           "recipe": {"steps": steps, "batch": BATCH, "lr": LR, "weight_decay": WD,
                      "cosine": True, "grad_clip": CLIP, "n_train": len(rows_tr),
                      "head_trainable": n_head,
                      "loss": {"M": "CE_skel + CE_assign",
                               "MP": "pair_CE({s,t}) + CE_assign",
                               "MP+": "CE_skel + CE_assign + 1.0*pair_CE"}[arm],
                      "lam": 1.0},
           "var": {"word": var["n_word"], "order": var["n_order"],
                   "cover": round(var["n_word"] / len(rows_tr), 6)},
           "loss_first": round(losses[0], 6), "loss_last": round(losses[-1], 6),
           "loss_parts": {k: {"first": round(v[0], 6), "last": round(v[-1], 6),
                              "mean": round(sum(v) / len(v), 6)}
                          for k, v in part_hist.items()},
           "wall_sec": round(time.time() - t0, 1)}
    (RESULTS / f"{name}.train.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {name} → weights/{name}.pt（{out['wall_sec']}s）", flush=True)
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True, choices=list(SEEDS))
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args(argv)
    train_one(a.arm, a.seed, a.randlabel, a.device, a.steps)


if __name__ == "__main__":
    main(sys.argv[1:])
