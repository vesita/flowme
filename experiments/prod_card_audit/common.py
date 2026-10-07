"""A0/A2/A4 共用：口径与逐样本评测（与 `capability_map` / `evaluate_task` 逐字对齐）。"""
from __future__ import annotations

import copy
import pickle
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CAPMAP = ROOT / "experiments" / "capability_map"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CAPMAP))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.tasks.runtime import (  # noqa: E402
    GenericTaskDataset,
    _rollout,
    _truth_of,
)

CACHE = CAPMAP / "cache"
SEEDS = (42, 43)
#: 与 checkpoints/arm_neg5_seed{42,43}.pt 的 train_args.task_samples 逐项一致
TASK_SAMPLES = {"pronoun": 6000, "sentiment": 32000, "relation": 6000,
                "person": 6000, "negation": 6000}
DEFAULT_SEED = 20240927
EVAL_FILES = [f"{c}_ordered_s{s}.pkl" for c in TASK_SAMPLES for s in SEEDS]


def split_of(cap: str, seed: int) -> tuple[list[dict], list[dict]]:
    """与 `capability_map/probe.py::split_of` 逐字相同（读同一份 pkl）。"""
    ordered = pickle.loads((CACHE / f"{cap}_ordered_s{seed}.pkl").read_bytes())
    data = copy.deepcopy(ordered)
    random.Random(seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    return data[:n_val], data[n_val:]


def make_loader(samples: list[dict], spec, bs: int = 64) -> DataLoader:
    return DataLoader(GenericTaskDataset(samples, NanoCharTokenizer(), spec),
                      batch_size=bs, shuffle=False, drop_last=False)


@torch.no_grad()
def eval_persample(enc, dec, loader, spec, device) -> dict:
    """`runtime.evaluate_task` 的逐样本版：口径逐字相同，额外吐 0/1 向量。

    返回 report（与 evaluate_task 同键的子集）+ per-sample 位：
      cls_ok[i] / cls_real[i] / exact_ok[i] / span_ok[i]
    """
    was = (enc.training, dec.training)
    enc.eval()
    dec.eval()
    n = len(loader.dataset)
    cls_ok = [0] * n
    cls_real = [0] * n
    exact_ok = [0] * n
    span_ok = [0] * n
    cls_tot = span_ok_n = bg_fired = bg_tot = 0
    exact_tot = 0
    tp = fp = fn = 0
    i = 0
    for batch in loader:
        inp = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        mem = enc(inp, attention_mask=mask)
        B = inp.shape[0]
        q0 = dec.bos_query.expand(B, 1, -1)
        out = dec.forward_step(q0, mem, doc_mask=mask)
        pred_cls = out["cls_logits"].argmax(-1)
        pred_s = out["start_logits"].argmax(-1)
        pred_e = out["end_logits"].argmax(-1)

        t_labels = batch["labels"][:, 0].to(device)
        t_s = batch["starts"][:, 0].to(device)
        t_e = batch["ends"][:, 0].to(device)
        is_bg = batch["is_bg"].to(device)
        real = t_labels > 0

        bg_fired += ((pred_cls > 0) & (is_bg > 0.5)).sum().item()
        bg_tot += (is_bg > 0.5).sum().item()

        preds = _rollout(dec, mem, mask, B, spec, ndb=None, input_ids=inp)
        for b in range(B):
            truth = _truth_of(batch, b, spec)
            got = preds[b]
            exact_tot += 1
            eq = int(sorted(got) == sorted(truth))
            exact_ok[i + b] = eq
            ts, gs = set(truth), set(got)
            tp += len(ts & gs)
            fp += len(gs - ts)
            fn += len(ts - gs)
            r = bool(real[b].item())
            cls_real[i + b] = int(r)
            if r:
                c = int((pred_cls[b] == t_labels[b]).item())
                cls_ok[i + b] = c
                cls_tot += 1
                sh = int(((pred_s[b] == t_s[b]) & (pred_e[b] == t_e[b])).item())
                span_ok[i + b] = sh
                span_ok_n += sh
        i += B
    if was[0]:
        enc.train()
    if was[1]:
        dec.train()
    return {
        "cls_acc": ratio(cls_ok, cls_tot),
        "span_hit": span_ok_n / max(1, cls_tot),
        "bg_fp": bg_fired / max(1, bg_tot),
        "n_cls": cls_tot,
        "n_bg": bg_tot,
        "exact_match": ratio(exact_ok, exact_tot),
        "slice_precision": tp / max(1, tp + fp),
        "slice_recall": tp / max(1, tp + fn),
        "n": n,
        "cls_ok": cls_ok,
        "cls_real": cls_real,
        "exact_ok": exact_ok,
        "span_ok": span_ok,
    }


def ratio(vec: list[int], tot: int) -> float:
    return sum(vec) / max(1, tot)


def paired_delta(a_ok: list[int], a_real: list[int], b_ok: list[int], b_real: list[int],
                 mask: list[int] | None = None) -> dict:
    """配对 Δ = mean(b − a)，SE = std(b−a, ddof=1)/√n（与 free_rule_floor F1 同一套）。"""
    ds = []
    for i in range(len(a_ok)):
        if mask is not None and not mask[i]:
            continue
        if a_real[i] != b_real[i]:
            raise AssertionError(f"real 掩码不对齐 @{i}")
        ds.append(b_ok[i] - a_ok[i])
    n = len(ds)
    if n == 0:
        return {"n": 0, "delta": float("nan"), "se": float("nan")}
    m = sum(ds) / n
    if n > 1:
        var = sum((d - m) ** 2 for d in ds) / (n - 1)
        se = (var ** 0.5) / (n ** 0.5)
    else:
        se = 0.0
    return {"n": n, "delta": m, "se": se,
            "z": (m / se) if se > 0 else (0.0 if m == 0 else float("inf")),
            "significant": bool(abs(m) > 2 * se),
            "same_sign_two_seed": None}


def lookup(fit_keys, fit_y, eval_keys):
    """逐字复用 `experiments/free_rule_floor/rules.py::lookup` 的口径。"""
    glob = Counter(fit_y).most_common(1)[0][0]
    tab: dict = {}
    for k, y in zip(fit_keys, fit_y):
        tab.setdefault(k, Counter())[y] += 1
    maj = {k: c.most_common(1)[0][0] for k, c in tab.items()}
    seen = sum(1 for k in eval_keys if k in maj)
    return [maj.get(k, glob) for k in eval_keys], seen / max(1, len(eval_keys))
