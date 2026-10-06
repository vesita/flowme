#!/usr/bin/env python3
"""select_rerank 打分头训练 + 全量评测（核全程冻结）。

配方 = PREREG §2（跑前写死）：
  **1800 步**（= anchored_select 的 12 ep × 150 步；本数据 8000/64 = 125 步/轮 ⇒ 14.4 轮）、
  batch 64、lr_head 1e-3、AdamW(wd 1e-4)、cosine、grad clip 1.0、seed 42 / 43。
  随机标签对照（P3）：train 标签 `randperm`（保持 1:1）后**同配方**重训，test/adv 仍用真标签评测。

核冻结 ⇒ 编码结果与现算逐位相同，故按 (数据, SPEC) 缓存 pooled 向量（一次编码，多 seed 复用）。

用法：
  uv run python experiments/select_rerank/train_rerank.py --arm clean --seed 42
  uv run python experiments/select_rerank/train_rerank.py --arm clean --seed 42 --randlabel
  uv run python experiments/select_rerank/train_rerank.py --arm shortcut --seed 43
"""
from __future__ import annotations

import argparse
import hashlib
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

from model import RerankModel, RerankSpec  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
CACHE = HERE / "cache"

STEPS = 1800          # PREREG §2：1800 步（12 ep × 150 步的等价总量）
BATCH = 64
LR = 1e-3
WD = 1e-4
CLIP = 1.0
SPLITS = ("train", "test", "adv")


def load_rows(arm: str) -> list[dict]:
    rows: list[dict] = []
    for s in SPLITS:
        p = DATA / arm / f"{s}.jsonl"
        if not p.exists():
            raise SystemExit(f"缺数据文件：{p}（先跑 build_data.py）")
        with open(p, encoding="utf-8") as fp:
            for line in fp:
                r = json.loads(line)
                r["split_name"] = s
                rows.append(r)
    return rows


def data_fingerprint(arm: str) -> str:
    h = hashlib.md5()
    for s in SPLITS:
        h.update((DATA / arm / f"{s}.jsonl").read_bytes())
    return h.hexdigest()


def get_vectors(model: RerankModel, rows: list[dict], arm: str, spec: RerankSpec,
                device: str) -> dict:
    """核冻结 ⇒ pooled 向量可缓存；键 = 数据 md5 + SPEC。"""
    CACHE.mkdir(exist_ok=True)
    fp = data_fingerprint(arm)
    key = f"{arm}_{fp[:12]}_ctx{spec.max_len_ctx}_cand{spec.max_len_cand}.pt"
    path = CACHE / key
    if path.exists():
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob["n"] == len(rows):
            print(f"[cache] 命中 {path.name}（n={blob['n']}）", flush=True)
            return blob
    print(f"[encode] 编码 {len(rows)} 条（核冻结，缓存 {key}）...", flush=True)
    t0 = time.time()
    v_ctx = model.encode_texts([r["context"] for r in rows], spec.max_len_ctx, device)
    flat = [c for r in rows for c in r["candidates"]]          # [2N, 32] → [N,2,D]
    v_cand = model.encode_texts(flat, spec.max_len_cand, device).view(len(rows), spec.k, -1)
    print(f"[encode] {time.time()-t0:.1f}s，v_ctx {tuple(v_ctx.shape)} "
          f"v_cand {tuple(v_cand.shape)}", flush=True)
    blob = {"v_ctx": v_ctx, "v_cand": v_cand,
            "labels": torch.tensor([r["label"] for r in rows], dtype=torch.long),
            "splits": torch.tensor([SPLITS.index(r["split_name"]) for r in rows]),
            "n": len(rows)}
    torch.save(blob, path)
    return blob


def evaluate(head, blob: dict, mask: torch.Tensor, device: str, bs: int = 2048) -> dict:
    head.eval()
    idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
    v_ctx, v_cand, y = blob["v_ctx"][idx], blob["v_cand"][idx], blob["labels"][idx]
    correct = 0
    loss_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            c = v_ctx[i:i + bs].to(device)
            d = v_cand[i:i + bs].to(device)
            t = y[i:i + bs].to(device)
            logits = head(c, d)
            loss_sum += torch.nn.functional.cross_entropy(logits, t, reduction="sum").item()
            correct += (logits.argmax(-1) == t).sum().item()
    n = len(idx)
    acc = correct / n
    se = math.sqrt(0.25 / n)
    return {"n": n, "acc": round(acc, 6), "ce": round(loss_sum / n, 6),
            "se": round(se, 6), "margin": round(acc - 0.5, 6),
            "margin_over_se": round((acc - 0.5) / se, 3),
            "passes_2se": bool(acc - 0.5 > 2 * se),
            "blind_guess": 0.5}


def train_one(arm: str, seed: int, randlabel: bool, device: str | None = None) -> dict:
    t_start = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    spec = RerankSpec.from_build_spec()
    model = RerankModel(spec).to(device)
    freeze = model.freeze_report()
    print(f"[freeze] {json.dumps(freeze, ensure_ascii=False)}", flush=True)

    rows = load_rows(arm)
    blob = get_vectors(model, rows, arm, spec, device)
    n_train = sum(1 for r in rows if r["split_name"] == "train")
    labels = blob["labels"].clone()
    if randlabel:                       # P3：只打乱 train 标签，保持 1:1；test/adv 保持真标签
        g = torch.Generator().manual_seed(seed * 1000 + 7)
        idx = torch.arange(n_train)
        labels[idx] = labels[idx][torch.randperm(n_train, generator=g)]
        cnt = torch.bincount(labels[:n_train], minlength=2).tolist()
        print(f"[randlabel] train 标签已 randperm，计数 {cnt}（应为 "
              f"[{n_train // 2}, {n_train - n_train // 2}]）", flush=True)

    train_mask = blob["splits"] == 0
    ds = TensorDataset(blob["v_ctx"][train_mask], blob["v_cand"][train_mask],
                       labels[train_mask])
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=gen, drop_last=False)
    head = model.head.to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    head.train()
    step = 0
    losses: list[float] = []
    while step < STEPS:
        for c, d, y in dl:
            c, d, y = c.to(device), d.to(device), y.to(device)
            logits = head(c, d)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step >= STEPS:
                break
    print(f"[train] {step} 步 done，loss 首/末 {losses[0]:.4f}/{losses[-1]:.4f}", flush=True)

    masks = {s: blob["splits"] == i for i, s in enumerate(SPLITS)}
    evals = {s: evaluate(head, blob, masks[s], device) for s in SPLITS}
    for s in SPLITS:
        e = evals[s]
        print(f"[eval] {arm}{'_rand' if randlabel else ''} seed{seed} {s:5s} "
              f"n={e['n']} acc={e['acc']:.4f} (SE={e['se']:.4f} 余量/SE={e['margin_over_se']:+.2f} "
              f"过2SE={e['passes_2se']})", flush=True)

    name = f"{arm}_s{seed}{'_rand' if randlabel else ''}"
    out = {
        "name": name, "arm": arm, "seed": seed, "randlabel": randlabel,
        "device": device, "freeze": freeze,
        "spec": {"max_len_ctx": spec.max_len_ctx, "max_len_cand": spec.max_len_cand,
                 "head_hidden": spec.head_hidden, "hidden": spec.hidden},
        "recipe": {"steps": STEPS, "batch": BATCH, "lr": LR, "weight_decay": WD,
                   "cosine": True, "grad_clip": CLIP, "n_train": n_train},
        "eval": evals,
        "loss_first": round(losses[0], 6), "loss_last": round(losses[-1], 6),
        "wall_sec": round(time.time() - t_start, 1),
        "data_fingerprint": data_fingerprint(arm),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    WEIGHTS.mkdir(exist_ok=True)
    torch.save({k: v for k, v in head.state_dict().items()},
               WEIGHTS / f"{name}.pt")
    print(f"[done] {name} → results/{name}.json  ({out['wall_sec']}s)", flush=True)
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["clean", "shortcut"])
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43])
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    train_one(a.arm, a.seed, a.randlabel, a.device)


if __name__ == "__main__":
    main(sys.argv[1:])
