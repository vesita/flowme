#!/usr/bin/env python3
"""select_pool 训练 + 全量评测（核全程冻结；唯一变量 = 聚合方式）。

配方 = PREREG §3（跑前写死，与 select_rerank 逐项相同）：
  1800 步、batch 64、lr 1e-3、AdamW(wd 1e-4)、cosine、grad clip 1.0、seed 42/43。
  随机标签对照（Q3）：train 标签 randperm（保持 1:1，generator seed*1000+7）后同配方重训。

数据**只读** `experiments/select_rerank/data/`（不写该目录）；核冻结 ⇒ token 级隐藏可缓存，
按 (数据 md5 + SPEC) 键控存本目录 `cache/`；建缓存时做 PREREG Q4a 口径同一性校验
（标签逐元素相等 + mean-pool 向量与 B 族缓存逐元素差 ≤ 1e-6）。

用法：
  uv run python experiments/select_pool/train_pool.py --arm A --seed 42
  uv run python experiments/select_pool/train_pool.py --arm D --seed 42 --randlabel
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
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_pool import ARMS, ARM_DESC, PoolModel, PoolSpec, masked_mean  # noqa: E402

DATA = HERE.parent / "select_rerank" / "data"        # 只读复用
DATA_ARM = "clean"        # 数据臂（N）：B 族引的 49.52/51.04 与 92.56/92.48 就是它的 adv
POOL_ARMS_DESC = "本实验的 --arm 是**聚合臂**（A/B/C/D/Aplus），数据臂固定为 clean"
SR_CACHE = HERE.parent / "select_rerank" / "cache"   # 只读（Q4a 对照）
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
CACHE = HERE / "cache"

STEPS = 1800
BATCH = 64
LR = 1e-3
WD = 1e-4
CLIP = 1.0
SPLITS = ("train", "test", "adv")
CTYPE = {0: "-", 1: "heldout_pair", 2: "shifted_pos"}
EXPECTED_N = {"train": 8000, "test": 2500, "adv": 2500}


# ---------------------------------------------------------------------------
# 数据（只读）
# ---------------------------------------------------------------------------
def load_rows(arm: str) -> list[dict]:
    """行序逐行复刻 select_rerank/train_rerank.py::load_rows（train→test→adv）。"""
    rows: list[dict] = []
    for s in SPLITS:
        p = DATA / arm / f"{s}.jsonl"
        if not p.exists():
            raise SystemExit(f"缺数据文件：{p}（数据在 select_rerank，只读复用；"
                             f"数据臂固定 {DATA_ARM}，--arm 是聚合臂）")
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


def get_blob(model: PoolModel, rows: list[dict], arm: str, spec: PoolSpec,
             device: str) -> dict:
    """token 级缓存（核冻结 ⇒ 一次编码、多 seed/多臂复用）；含 Q4a 口径校验。"""
    CACHE.mkdir(exist_ok=True)
    fp = data_fingerprint(arm)
    key = f"{arm}_{fp[:12]}_tok_ctx{spec.max_len_ctx}_cand{spec.max_len_cand}.pt"
    path = CACHE / key
    parity: dict
    if path.exists():
        blob = torch.load(path, map_location="cpu")
        if blob["n"] == len(rows) and blob["fingerprint"] == fp:
            print(f"[cache] 命中 {path.name}（n={blob['n']}）", flush=True)
            parity = blob.get("parity", {"checked": False, "reason": "缓存已存在"})
            return blob
    print(f"[encode] 编码 {len(rows)} 条（核冻结，token 级缓存 {key}）...", flush=True)
    t0 = time.time()
    h_ctx, m_ctx = model.encode_tokens([r["context"] for r in rows],
                                       spec.max_len_ctx, device)
    flat = [c for r in rows for c in r["candidates"]]           # [2N] → [N,2,·]
    h_c, m_c = model.encode_tokens(flat, spec.max_len_cand, device)
    n = len(rows)
    h_cand = h_c.view(n, spec.k, spec.max_len_cand, -1)
    m_cand = m_c.view(n, spec.k, spec.max_len_cand)
    labels = torch.tensor([r["label"] for r in rows], dtype=torch.long)
    splits = torch.tensor([SPLITS.index(r["split_name"]) for r in rows])
    ct_map = {"heldout_pair": 1, "shifted_pos": 2}
    codes = torch.tensor([ct_map.get(r.get("ctype"), 0) for r in rows])
    n_adv = int((splits == 2).sum())
    assert int((codes == 1).sum()) + int((codes == 2).sum()) == n_adv, "adv 的 ctype 只能是两个已知值"
    print(f"[encode] {time.time()-t0:.1f}s h_ctx {tuple(h_ctx.shape)} "
          f"h_cand {tuple(h_cand.shape)}", flush=True)

    # ---- Q4a：与 B 族缓存逐元素比对 -------------------------------------
    parity = {"checked": False}
    cands = sorted(SR_CACHE.glob(f"{arm}_*_ctx*_cand*.pt"))
    if cands:
        sr = torch.load(cands[-1], map_location="cpu", weights_only=True)
        lab_eq = bool(torch.equal(sr["labels"], labels))
        pv_ctx = masked_mean(h_ctx.to(device), m_ctx.to(device)).cpu()
        pv_cand = masked_mean(h_cand.to(device), m_cand.to(device)).cpu()
        d_ctx = float((pv_ctx - sr["v_ctx"]).abs().max())
        d_cand = float((pv_cand - sr["v_cand"]).abs().max())
        parity = {"checked": True, "sr_cache": cands[-1].name, "labels_equal": lab_eq,
                  "max_abs_diff_ctx": d_ctx, "max_abs_diff_cand": d_cand,
                  "tol": 1e-6, "ok": bool(lab_eq and d_ctx <= 1e-6 and d_cand <= 1e-6)}
        print(f"[Q4a] 标签逐元素相等={lab_eq} max|Δv_ctx|={d_ctx:.3e} "
              f"max|Δv_cand|={d_cand:.3e} tol=1e-6 → ok={parity['ok']}", flush=True)
        assert parity["ok"], f"Q4a 口径不一致，先查口径：{parity}"
    blob = {"h_ctx": h_ctx, "m_ctx": m_ctx, "h_cand": h_cand, "m_cand": m_cand,
            "labels": labels, "splits": splits, "ctype": codes, "n": n,
            "fingerprint": fp, "parity": parity}
    torch.save(blob, path)
    print(f"[cache] 已写 {path.name}（{path.stat().st_size/1e9:.2f} GB）", flush=True)
    return blob


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def evaluate(model: PoolModel, blob: dict, idx: torch.Tensor, device: str,
             bs: int = 1024) -> dict:
    model.eval()
    y_all = blob["labels"][idx]
    correct: list[torch.Tensor] = []
    logit_var: list[float] = []
    loss_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            j = idx[i:i + bs]
            c = blob["h_ctx"][j].to(device)
            mc = blob["m_ctx"][j].to(device)
            d = blob["h_cand"][j].to(device)
            md = blob["m_cand"][j].to(device)
            t = y_all[i:i + bs].to(device)
            logits = model.forward_tokens(c, mc, d, md)
            assert torch.isfinite(logits).all(), "logits 出现非有限值"
            loss_sum += torch.nn.functional.cross_entropy(logits, t, reduction="sum").item()
            correct.append((logits.argmax(-1) == t).cpu())
            logit_var.append(float(logits.float().var()))
    ok = torch.cat(correct)
    n = len(idx)
    assert n > 0, "空评测集"
    assert min(logit_var) > 0, "logits 是常数（空跑/塌缩）"
    acc = float(ok.float().mean())
    se = math.sqrt(0.25 / n)
    lab = y_all
    maj = float(torch.bincount(lab, minlength=2).float().max() / n)
    return {"n": n, "acc": round(acc, 6), "ce": round(loss_sum / n, 6),
            "se": round(se, 6), "margin": round(acc - 0.5, 6),
            "margin_over_se": round((acc - 0.5) / se, 3),
            "passes_2se": bool(acc - 0.5 > 2 * se),
            "majority_baseline": round(maj, 6),
            "logit_var_min": round(min(logit_var), 8),
            "_correct": ok}


def eval_splits(model: PoolModel, blob: dict, device: str) -> dict:
    out = {}
    for i, s in enumerate(SPLITS):
        idx = torch.nonzero(blob["splits"] == i, as_tuple=False).squeeze(-1)
        assert len(idx) == EXPECTED_N[s], f"{s} n={len(idx)} ≠ {EXPECTED_N[s]}"
        e = evaluate(model, blob, idx, device)
        assert abs(e["majority_baseline"] - 0.5) < 1e-9, \
            f"{s} 多数类基线不是 0.5000：{e['majority_baseline']}（数据 1:1 被破坏）"
        out[s] = e
    # adv 按 ctype 拆（Q1/Q2 的决定性口径）
    adv_idx = torch.nonzero(blob["splits"] == 2, as_tuple=False).squeeze(-1)
    adv = out["adv"]
    correct = adv.pop("_correct")
    for code, name in ((1, "heldout_pair"), (2, "shifted_pos")):
        sel = blob["ctype"][adv_idx] == code
        sub = adv_idx[sel]
        assert len(sub) == 1250, f"{name} n={len(sub)} ≠ 1250"
        c = correct[sel]
        acc = float(c.float().mean())
        se = math.sqrt(0.25 / len(sub))
        out.setdefault("adv_by_ctype", {})[name] = {
            "n": len(sub), "acc": round(acc, 6), "se": round(se, 6),
            "margin": round(acc - 0.5, 6),
            "margin_over_se": round((acc - 0.5) / se, 3),
            "passes_2se": bool(acc - 0.5 > 2 * se)}
    for s in ("train", "test"):
        out[s].pop("_correct")
    return out


class IndexDataset(torch.utils.data.Dataset):
    """只按下标取数（避免把 0.8GB 训练张量整块复制一份）。"""

    def __init__(self, idx: torch.Tensor):
        self.idx = idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        return self.idx[i]


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_one(arm: str, seed: int, randlabel: bool, device: str | None = None) -> dict:
    t_start = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    spec = PoolSpec.from_build_spec()
    model = PoolModel(arm, spec).to(device)
    freeze = model.freeze_report()
    print(f"[freeze] {json.dumps(freeze, ensure_ascii=False)}", flush=True)

    rows = load_rows(DATA_ARM)
    blob = get_blob(model, rows, DATA_ARM, spec, device)
    n_train = EXPECTED_N["train"]
    labels = blob["labels"].clone()
    if randlabel:                       # Q3：只打乱 train 标签，保持 1:1
        g = torch.Generator().manual_seed(seed * 1000 + 7)
        idx = torch.arange(n_train)
        labels[idx] = labels[idx][torch.randperm(n_train, generator=g)]
        cnt = torch.bincount(labels[:n_train], minlength=2).tolist()
        n_diff = int((labels[:n_train] != blob["labels"][:n_train]).sum())
        print(f"[randlabel] train 标签已 randperm，计数 {cnt}，与真标签不同 {n_diff} 条", flush=True)
        assert cnt == [n_train // 2, n_train - n_train // 2], f"随机标签计数不是 1:1：{cnt}"
        assert n_diff > 0, "随机标签没有改变任何标签（空对照）"

    train_idx = torch.nonzero(blob["splits"] == 0, as_tuple=False).squeeze(-1)
    ds = IndexDataset(train_idx)                  # 只存下标，按 batch 现取（不复制 0.8GB）
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=gen, drop_last=False)
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    assert n_params == freeze["trainable_params"], "可训参数量与冻结报告不一致"
    assert all(not p.requires_grad for p in model.encoder.parameters()), "核参数可训"
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)

    for p in params:
        p.requires_grad_(True)
    step = 0
    losses: list[float] = []
    while step < STEPS:
        for j in dl:
            j = j[0] if isinstance(j, (list, tuple)) else j
            c = blob["h_ctx"][j].to(device)
            mc = blob["m_ctx"][j].to(device)
            d = blob["h_cand"][j].to(device)
            md = blob["m_cand"][j].to(device)
            y = labels[j].to(device)
            logits = model.forward_tokens(c, mc, d, md)
            loss = torch.nn.functional.cross_entropy(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, CLIP)
            opt.step()
            sched.step()
            losses.append(loss.item())
            step += 1
            if step >= STEPS:
                break
    assert step == STEPS, f"只跑了 {step} 步"
    m_first, m_last = sum(losses[:50]) / 50, sum(losses[-50:]) / 50
    assert all(math.isfinite(x) for x in losses), "loss 出现非有限值"
    if not randlabel:
        assert m_last < m_first, f"loss 没降（前50 {m_first:.4f} / 后50 {m_last:.4f}）"
    print(f"[train] {step} 步 done，loss 首/末 {losses[0]:.4f}/{losses[-1]:.4f} "
          f"前50均 {m_first:.4f} / 后50均 {m_last:.4f}"
          f"{'（randlabel：不设降 loss 断言）' if randlabel else ''}", flush=True)

    evals = eval_splits(model, blob, device)
    for s in SPLITS:
        e = evals[s]
        print(f"[eval] {arm}{'_rand' if randlabel else ''} seed{seed} {s:5s} "
              f"n={e['n']} acc={e['acc']:.4f} (SE={e['se']:.4f} "
              f"余量/SE={e['margin_over_se']:+.2f} 过2SE={e['passes_2se']} "
              f"多数类={e['majority_baseline']:.4f})", flush=True)
    for name, e in evals.get("adv_by_ctype", {}).items():
        print(f"[eval] {arm}{'_rand' if randlabel else ''} seed{seed} adv/{name} "
              f"n={e['n']} acc={e['acc']:.4f} (SE={e['se']:.4f} "
              f"余量/SE={e['margin_over_se']:+.2f} 过2SE={e['passes_2se']})", flush=True)

    name = f"{arm}_s{seed}{'_rand' if randlabel else ''}"
    out = {
        "name": name, "arm": arm, "arm_desc": ARM_DESC[arm], "seed": seed,
        "data_arm": DATA_ARM,
        "randlabel": randlabel, "device": device, "freeze": freeze,
        "spec": {"max_len_ctx": spec.max_len_ctx, "max_len_cand": spec.max_len_cand,
                 "head_hidden": spec.head_hidden, "hidden": spec.hidden},
        "recipe": {"steps": STEPS, "batch": BATCH, "lr": LR, "weight_decay": WD,
                   "cosine": True, "grad_clip": CLIP, "n_train": n_train},
        "eval": evals,
        "parity": blob.get("parity"),
        "loss_first": round(losses[0], 6), "loss_last": round(losses[-1], 6),
        "wall_sec": round(time.time() - t_start, 1),
        "data_fingerprint": blob.get("fingerprint"),
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    WEIGHTS.mkdir(exist_ok=True)
    torch.save({k: v for k, v in model.state_dict().items()
                if k.startswith(("agg.", "head."))}, WEIGHTS / f"{name}.pt")
    print(f"[done] {name} → results/{name}.json  ({out['wall_sec']}s)", flush=True)
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43])
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    train_one(a.arm, a.seed, a.randlabel, a.device)


if __name__ == "__main__":
    main(sys.argv[1:])
