#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P12b 训练主臂：核全程冻结、只训核外私有头（PREREG §2 / §2.1）。

口径（跑前写死，照抄 PREREG §2）：
  · 核 = checkpoints/base_encoder.pt::doc_encoder（hidden 128，eval() + requires_grad=False，逐参数实测打印）；
  · 取层 = blocks[2] 输出（hook 捕获）；enc() 返回值 = norm 输出，**同一次前向对账**（torch.equal）；
  · 特征 = [v1, v2, v1⊙v2, |v1−v2|, w1, w2, g]，v = 提及 span 内 masked mean，
    w = 提及两侧各 6 字窗口（不含提及本身）masked mean，g = 全文 masked mean ⇒ 7×128 = 896 维；
  · 头（核外私有）= Linear(896,256) → GELU → Linear(256,2)，2 类 SAME/DIFF，参数量打印进 meta；
  · 训练 = AdamW lr 1e-3、wd 1e-4、cosine、clip 1.0、batch 64、1000 步、seed 42/43；
    --randlabel 只对训练标签 randperm（保类分布）⇒ E3 对照；
  · 特征标准化 = train 统计（mu/sd，sd<1e-6 视为 1；与 core_probe 同口径）。

评测：三臂 test 全报（主表 3×3），逐型 / 逐子型 acc、SE。
产物：features/*.npz（特征缓存）、results/train_results.json、logs/train*.log。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402

from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402

BASE = ROOT / "checkpoints" / "base_encoder.pt"
FEAT = HERE / "features"
RES = HERE / "results"
DATA = HERE / "data"

ARM_FILES = {"x": ("train.jsonl", "test.jsonl"),
             "x-ir": ("xir_train.jsonl", "xir_test.jsonl"),
             "x-cross": ("xcross_train.jsonl", "xcross_test.jsonl")}
SEEDS = (42, 43)
STEPS = 1000
BATCH = 64
LR = 1e-3
WD = 1e-4
CLIP = 1.0
MAX_LEN = 64
WIN = 6
HID = 128
HEAD_HIDDEN = 256
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log(m: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_rows(name: str, split: int) -> list[dict]:
    p = DATA / ARM_FILES[name][split]
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ---------- 冻结实况（逐参数实测打印） ----------
def freeze_report(enc) -> dict:
    rows: dict[str, dict] = {}
    trainable = []
    for n, p in enc.named_parameters():
        top = n.split(".")[0] if "." in n else n
        r = rows.setdefault(top, {"params": 0, "requires_grad_true": 0,
                                  "requires_grad_false": 0})
        r["params"] += p.numel()
        if p.requires_grad:
            r["requires_grad_true"] += 1
            trainable.append(n)
        else:
            r["requires_grad_false"] += 1
    total = sum(v["params"] for v in rows.values())
    log(f"[freeze] enc.training={enc.training} 总参数={total:,} "
        f"requires_grad=True 的参数={len(trainable)}")
    for k, v in rows.items():
        log(f"[freeze]   {k:12s} params={v['params']:,} "
            f"T={v['requires_grad_true']} F={v['requires_grad_false']}")
    sample = [n for n, p in enc.named_parameters()][:6]
    log(f"[freeze]   逐参数样例（名称/req_grad）: " +
        ", ".join(f"{n}={p.requires_grad}" for n, p in list(enc.named_parameters())[:6]))
    log(f"[freeze]   模块状态: embedding={enc.embedding.training} "
        f"blocks[2]={enc.blocks[2].training} norm={enc.norm.training}")
    if trainable:
        log(f"[freeze] !! 仍有可训练参数: {trainable}")
    return {"total_params": total, "trainable_params": len(trainable),
            "per_module": rows, "training_mode": enc.training,
            "sample": {n: p.requires_grad for n, p in list(enc.named_parameters())[:6]}}


# ---------- 特征抽取 ----------
class Head(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(7 * HID, HEAD_HIDDEN), nn.GELU(), nn.Linear(HEAD_HIDDEN, 2))

    def forward(self, x):
        return self.net(x)


def extract(enc, tok, rows: list[dict]) -> tuple[np.ndarray, dict]:
    n = len(rows)
    ids, masks = [], []
    over = 0
    for r in rows:
        t = r["text"]
        if len(t) > MAX_LEN:
            over += 1
        e = tok.encode(t, max_length=MAX_LEN, padding=True)
        ids.append(e["input_ids"])
        masks.append(e["attention_mask"])
    ids_t = torch.tensor(np.array(ids), dtype=torch.long, device=DEV)
    mask_t = torch.tensor(np.array(masks), dtype=torch.bool, device=DEV)

    X = np.zeros((n, 7 * HID), dtype=np.float32)
    cap: dict = {}
    h1 = enc.blocks[2].register_forward_hook(lambda m, i, o: cap.__setitem__("b2", o))
    h2 = enc.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("norm", o))
    dropped = 0
    with torch.no_grad():
        for i0 in range(0, n, 256):
            sl = slice(i0, min(i0 + 256, n))
            doc = enc(ids_t[sl], attention_mask=mask_t[sl])
            if i0 == 0:
                # 构造性对账：hook 捕获的 norm 输出必须与同一次前向返回值逐位相同
                assert torch.equal(cap["norm"], doc), "hook 的 norm 输出 != enc() 返回值"
            b2 = cap["b2"]
            m = mask_t[sl]
            valid = m.sum(1).cpu().numpy()
            m3 = m.unsqueeze(-1).float()
            g = (b2 * m3).sum(1) / m3.sum(1)                      # 全文 masked mean
            for k, r in enumerate(rows[sl.start:sl.stop]):
                a, b = r["m1_span"][0], r["m1_span"][1]
                c, d = r["m2_span"][0], r["m2_span"][1]
                vl = int(valid[k])

                def span_mean(s, e, _k=k):
                    s2, e2 = max(0, s), min(e, vl)
                    if e2 - s2 < 1:
                        return None
                    return b2[_k, s2:e2].mean(0)

                def win_mean(s, e, _k=k):
                    wa = (max(0, s - WIN), min(s, vl))
                    wb = (max(e, 0), min(e + WIN, vl))
                    segs = [b2[_k, x:y] for x, y in (wa, wb) if y - x >= 1]
                    if not segs:
                        return None
                    return torch.cat(segs, 0).mean(0)

                v1, v2 = span_mean(a, b), span_mean(c, d)
                w1, w2 = win_mean(a, b), win_mean(c, d)
                if v1 is None or v2 is None:
                    dropped += 1
                    continue
                w1 = torch.zeros(HID, device=DEV) if w1 is None else w1
                w2 = torch.zeros(HID, device=DEV) if w2 is None else w2
                gg = g[k]
                X[i0 + k] = torch.cat([v1, v2, v1 * v2, (v1 - v2).abs(),
                                       w1, w2, gg]).detach().cpu().numpy().astype(np.float32)
    h1.remove()
    h2.remove()
    meta = {"n": n, "truncated": over, "span_dropped": dropped,
            "reconcile_norm_equal": True, "feat_dim": int(X.shape[1])}
    return X, meta


def feats_path(arm: str, split: int) -> Path:
    return FEAT / f"{arm}_{['train', 'test'][split]}.npz"


def get_features(enc, tok, arm: str, split: int, refresh: bool = False) -> tuple:
    p = feats_path(arm, split)
    if p.exists() and not refresh:
        z = np.load(p, allow_pickle=False)
        return z["X"], z["y"], json.loads(str(z["meta"]))
    rows = load_rows(arm, split)
    X, meta = extract(enc, tok, rows)
    y = np.array([1 if r["label"] == "DIFF" else 0 for r in rows], dtype=np.int64)
    FEAT.mkdir(exist_ok=True)
    np.savez_compressed(p, X=X, y=y, meta=json.dumps(meta, ensure_ascii=False))
    meta.update({"n_rows": len(rows)})
    return X, y, meta


# ---------- 训练 / 评测 ----------
def standardize(Xtr, X):
    mu = Xtr.mean(0)
    sd = Xtr.std(0)
    sd = np.where(sd < 1e-6, 1.0, sd).astype(np.float32)
    return (Xtr - mu) / sd, (X - mu) / sd, mu, sd


def train_head(Xtr, ytr, seed, steps, randlabel=False):
    yt = ytr.copy()
    if randlabel:
        np.random.default_rng(seed).shuffle(yt)        # randperm，保类分布
    A, _, mu, sd = standardize(Xtr, Xtr)
    torch.manual_seed(seed)
    head = Head().to(DEV)
    n_par = sum(p.numel() for p in head.parameters())
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    At = torch.as_tensor(A, dtype=torch.float32, device=DEV)
    yt_t = torch.as_tensor(yt, dtype=torch.long, device=DEV)
    g = torch.Generator().manual_seed(seed)
    losses = []
    for step in range(steps):
        idx = torch.randint(0, At.shape[0], (BATCH,), generator=g).to(DEV)
        opt.zero_grad()
        loss = nn.functional.cross_entropy(head(At[idx]), yt_t[idx])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), CLIP)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == steps - 1:
            losses.append(round(float(loss.detach().cpu()), 4))
    return head, {"head_params": n_par, "loss_trace": losses, "steps": steps,
                  "randlabel": randlabel}


def evaluate(head, Xtr, ytr, X, y) -> dict:
    """按 train 统计标准化后前向（与训练同口径）。"""
    _, Ate, _, _ = standardize(Xtr, X)
    with torch.no_grad():
        pred = head(torch.as_tensor(Ate, dtype=torch.float32, device=DEV)).argmax(1)
    p = pred.cpu().numpy()
    return {"acc": float((p == y).mean()), "n": int(len(y)),
            "se": math.sqrt(0.25 / max(len(y), 1)), "pred": p}


def per_group(y, p, groups) -> dict:
    out = {}
    for g in sorted(set(groups)):
        idx = [i for i, gg in enumerate(groups) if gg == g]
        out[g] = {"acc": float((p[idx] == y[idx]).mean()), "n": len(idx),
                  "se": math.sqrt(0.25 / max(len(idx), 1))}
    return out


def run(arm, seed, steps, randlabel, feats: dict) -> dict:
    """训练一个头（只训头），并在**三臂 test 全集**上评测（主表 3×3）。"""
    Xtr, ytr = feats[(arm, 0)]
    Xte, yte = feats[(arm, 1)]
    head, meta = train_head(Xtr, ytr, seed, steps, randlabel)
    own = evaluate(head, Xtr, ytr, Xte, yte)
    res = {
        "arm": arm, "seed": seed, "steps": steps, "randlabel": randlabel,
        "device": str(DEV),
        "head": meta,
        "own_test": {"acc": own["acc"], "n": own["n"], "se": own["se"],
                     "per_type": per_group(yte, own["pred"],
                                           [r["type"] for r in load_rows(arm, 1)]),
                     "per_subtype": per_group(yte, own["pred"],
                                              [r.get("subtype", r["type"]) for r in load_rows(arm, 1)])},
        "train_acc": float((evaluate(head, Xtr, ytr, Xtr, ytr)["pred"] == ytr).mean()),
        "cross_eval": {},
    }
    test_arms = sorted({k[0] for k in feats if k[1] == 1})
    for other in test_arms:
        Xo, yo = feats[(other, 1)]
        ev = evaluate(head, Xtr, ytr, Xo, yo)
        res["cross_eval"][other] = {
            "acc": ev["acc"], "n": ev["n"], "se": ev["se"],
            "per_type": per_group(yo, ev["pred"],
                                  [r["type"] for r in load_rows(other, 1)]),
            "per_subtype": per_group(yo, ev["pred"],
                                     [r.get("subtype", r["type"]) for r in load_rows(other, 1)])}
    return res, head


def load_enc():
    enc, _ = load_base_encoder(str(BASE), device=str(DEV))
    return enc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="短步数冒烟（管线通即可）")
    ap.add_argument("--refresh", action="store_true", help="忽略特征缓存重抽")
    a = ap.parse_args()
    RES.mkdir(exist_ok=True)
    log(f"device={DEV} torch={torch.__version__}")
    steps = 20 if a.smoke else STEPS
    out_path = RES / ("smoke_results.json" if a.smoke else "train_results.json")

    enc = load_enc()
    fr = freeze_report(enc)
    if fr["trainable_params"]:
        log("!! 冻结失败：核内存在 requires_grad=True 参数")
        return 2
    tok = NanoCharTokenizer()
    arms = ["x"] if a.smoke else list(ARM_FILES)
    feats, feat_meta = {}, {}
    for arm in arms:
        for sp in (0, 1):
            X, y, meta = get_features(enc, tok, arm, sp, a.refresh)
            feats[(arm, sp)] = (X, y)
            feat_meta[f"{arm}/{'train' if sp == 0 else 'test'}"] = meta
            log(f"[feat] {arm} {['train', 'test'][sp]}: X={X.shape} "
                f"mean={float(X.mean()):.4f} std={float(X.std()):.4f} meta={meta}")
    del enc
    torch.cuda.empty_cache()

    results = []
    runs = [(arm, seed, False) for arm in arms for seed in SEEDS]
    if not a.smoke:
        runs += [("x", seed, True) for seed in SEEDS]          # E3 随机标签对照
    for arm, seed, rl in runs:
        t = time.perf_counter()
        res, _ = run(arm, seed, steps, rl, feats)
        res["freeze"] = {"total_params": fr["total_params"],
                         "trainable_params": fr["trainable_params"],
                         "per_module": {k: {"params": v["params"],
                                            "T": v["requires_grad_true"],
                                            "F": v["requires_grad_false"]}
                                        for k, v in fr["per_module"].items()},
                         "training_mode": fr["training_mode"]}
        results.append(res)
        out_path.write_text(json.dumps(
            {"smoke": a.smoke, "freeze": results[-1]["freeze"],
             "feat_meta": feat_meta, "runs": results},
            ensure_ascii=False, indent=2), encoding="utf-8")
        log(f"[run] arm={arm} seed={seed} randlabel={rl} "
            f"own_test={res['own_test']['acc']:.4f} (n={res['own_test']['n']}) "
            f"train={res['train_acc']:.4f} head_params={res['head']['head_params']} "
            f"{time.perf_counter() - t:.1f}s → {out_path.name}")
    log(f"OK train_arm ({len(results)} runs) → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
