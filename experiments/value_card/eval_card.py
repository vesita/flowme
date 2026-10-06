#!/usr/bin/env python3
"""价值观卡评测：train / test / adversarial / gold × 主指标 + 附报 + 基线 + 对账。

    uv run python experiments/value_card/eval_card.py \
        --ckpt experiments/value_card/cards/value_s42.pt \
        --out experiments/value_card/results/main_s42.json --tag main_s42

口径（PREREG §3.1）：
  acc   = 逐样本**类**判对的比例（含背景）= 第 0 步 argmax 类别 == 真值类。
          max_steps=1 时 rollout 至多发射一条切片，故它与「发射非空判得体/冒犯、
          不发射判背景」逐样本等价；V1/V2/V3 都用它。
  exact = 切片集合与真值逐位一致（类 + 区间）—— 附报，不过线。
  同时用 `evaluate_task` 复算 cls_acc / bg_fp / exact_match 做**对账**，不一致直接退出。
  gold 另算 Spearman：模型分 s = P(冒犯) − P(得体)（第 0 步 softmax），金标准序 v = ±1/0。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

import build_data as bd
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
from dtseek.tasks.artifacts import build_card_decoder, load_base_encoder, read_card
from dtseek.tasks.runtime import GenericTaskDataset, _rollout, evaluate_task
from nano_char_tokenizer import NanoCharTokenizer

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SPLITS = ("train", "test", "adversarial", "gold")
INV = {0: "背景", 1: "得体", 2: "冒犯"}


def load_split(name: str) -> list[dict]:
    return [json.loads(l) for l in open(DATA / f"{name}.jsonl", encoding="utf-8")]


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def spearman(x: list[float], y: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _ranks(x), _ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = math.sqrt(sum((a - mx) ** 2 for a in rx))
    dy = math.sqrt(sum((b - my) ** 2 for b in ry))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--task", default="value_judge")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, bck = load_base_encoder(args.base, device)
    card_ck = read_card(args.ckpt)
    dec, spec = build_card_decoder(card_ck, device)
    dec.eval()
    core_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    n_head = sum(p.numel() for p in dec.parameters())
    print(f"[eval] ckpt={args.ckpt} card={spec.name} head={n_head:,} "
          f"core_trainable={core_train} device={device}", flush=True)
    if core_train != 0:
        raise SystemExit("核可训 ⇒ 评测口径不符（阶段 1 必须冻结核）")

    stats = json.loads((DATA / "stats.json").read_text(encoding="utf-8"))
    report: dict = {"ckpt": args.ckpt, "tag": args.tag, "task": spec.name,
                    "n_head_params": n_head, "core": "frozen(核冻结)",
                    "splits": {}}
    tok = NanoCharTokenizer()

    for split in args.splits.split(","):
        rows = load_split(split)
        items = [{"text": r["text"], "spans": r["spans"]} for r in rows]
        loader = DataLoader(GenericTaskDataset(items, tok, spec),
                            batch_size=128, shuffle=False)
        n = len(rows)
        dist = Counter(r["label"] for r in rows)
        majority = max(dist.values()) / n

        acc_ok = exact_ok = 0
        score: list[float] = []
        gold_v: list[int] = []
        pred_dist: Counter = Counter()
        per_rule_tot: Counter = Counter()
        per_rule_ok: Counter = Counter()
        per_cls_tot: Counter = Counter()
        per_cls_ok: Counter = Counter()
        bi = 0
        with torch.no_grad():
            for batch in loader:
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                mem = enc(inp, attention_mask=mask)
                B = inp.shape[0]
                q0 = dec.bos_query.expand(B, 1, -1)
                out0 = dec.forward_step(q0, mem, doc_mask=mask)
                probs = torch.softmax(out0["cls_logits"], dim=-1)      # [B,3]
                pred_cls = probs.argmax(-1)
                preds = _rollout(dec, mem, mask, B, spec, ndb=None, input_ids=inp)
                for b in range(B):
                    r = rows[bi + b]
                    lab = int(r["label"])
                    pc = int(pred_cls[b])
                    pred_dist[INV[pc]] += 1
                    ok = pc == lab
                    acc_ok += ok
                    per_rule_tot[r["rule"]] += 1
                    per_rule_ok[r["rule"]] += int(ok)
                    per_cls_tot[INV[lab]] += 1
                    per_cls_ok[INV[lab]] += int(ok)
                    truth = sorted((s["label"], s["start"], s["end"] - 1) for s in r["spans"])
                    got = sorted(preds[b])
                    exact_ok += int(got == truth)
                    if split == "gold":
                        score.append(float(probs[b][2] - probs[b][1]))
                        gold_v.append({0: 0.0, 1: -1.0, 2: 1.0}[lab])
                bi += B
        if bi != n:
            raise SystemExit(f"对账失败 {split}: 只评了 {bi}/{n} 条")

        m = evaluate_task(enc, dec, loader, device, spec)
        n_pos = sum(v for k, v in dist.items() if k > 0)
        n_bg = n - n_pos
        recomposed = (m["cls_acc"] * n_pos + (1 - m["bg_fp"]) * n_bg) / n
        acc = acc_ok / n
        res = {
            "n": n,
            "class_dist": {INV[k]: dist.get(k, 0) for k in (0, 1, 2)},
            "acc": round(acc, 4),
            "exact": round(exact_ok / n, 4),
            "majority_baseline": round(majority, 4),
            "majority_class": INV[max(dist, key=lambda k: dist[k])],
            "naive_rule_cls": round(bd.naive_cls_acc(rows), 4),
            "naive_rule_exact": round(bd.naive_acc(rows), 4),
            "pred_dist": dict(pred_dist),
            "per_rule_acc": {k: round(per_rule_ok[k] / per_rule_tot[k], 4)
                             for k in sorted(per_rule_tot)},
            "per_rule_n": dict(sorted(per_rule_tot.items())),
            "per_class_recall": {k: round(per_cls_ok[k] / per_cls_tot[k], 4)
                                 for k in sorted(per_cls_tot)},
            "evaluate_task": {k: round(m[k], 4) for k in
                              ("cls_acc", "bg_fp", "exact_match", "span_hit",
                               "slice_precision", "slice_recall")},
            "recomposed_acc": round(recomposed, 4),
            "reconcile_ok": abs(recomposed - acc) < 1e-6,
        }
        if split == "gold":
            rho = spearman(score, gold_v)
            res["spearman_rho"] = round(rho, 4)
            res["spearman_n"] = n
            res["score_mean_by_class"] = {
                INV[k]: round(sum(s for s, v in zip(score, gold_v)
                                  if v == {0: 0.0, 1: -1.0, 2: 1.0}[k]) /
                             max(1, sum(1 for v in gold_v
                                        if v == {0: 0.0, 1: -1.0, 2: 1.0}[k])), 4)
                for k in (0, 1, 2)}
        if not res["reconcile_ok"]:
            raise SystemExit(f"对账失败 {split}: 自算 {acc:.6f} vs evaluate_task 复算 {recomposed:.6f}")
        report["splits"][split] = res
        print(f"  [{split:13s}] n={n:5d} acc={acc:.4f} exact={res['exact']:.4f} "
              f"多数类={res['majority_baseline']:.4f} 朴素={res['naive_rule_cls']:.4f} "
              f"pred={res['pred_dist']} recall={res['per_class_recall']} "
              f"rule={res['per_rule_acc']}"
              + (f" rho={res['spearman_rho']:.4f}" if split == "gold" else "")
              + " 对账=OK", flush=True)

    out = Path(args.out) if args.out else HERE / "results" / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
