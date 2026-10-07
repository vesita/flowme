#!/usr/bin/env python3
"""P8 句式探针（本单元唯一训练：核冻结、只训头）。PREREG §3。

输入 = 冻结核 `v_sent`；结构 = `Linear(128→5)` 无隐藏层；目标 = `A(gold 骨架)` 5 类；
普通 CE（不加类权重，写死）；Adam lr=1e-2、bs=256、≤60 epoch、train 内 10% val、
早停 patience=10；init seed 42/43 两个都训都报。

用法：uv run python experiments/mode_conditioned_skel/probe.py [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import modes as MM  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
EVAL_SPLITS = ("test", "adv2", "a_bal")
SPLIT_PATH = {
    "train": C.TCH / "data" / "train.jsonl",
    "test": C.TCH / "data" / "test.jsonl",
    "adv2": C.SS / "data" / "adv2.jsonl",
    "a_bal": C.SLEAK / "data" / "a_bal.jsonl",
}


def macro_f1(y, p, k: int) -> tuple[float, list[float]]:
    f1s = []
    for c in range(k):
        tp = sum(1 for a, b in zip(y, p) if a == c and b == c)
        fp = sum(1 for a, b in zip(y, p) if a != c and b == c)
        fn = sum(1 for a, b in zip(y, p) if a == c and b != c)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if prec + rec else 0.0)
    return sum(f1s) / k, f1s


def report(y, p, k: int) -> dict:
    acc = sum(1 for a, b in zip(y, p) if a == b) / len(y)
    mf, f1s = macro_f1(y, p, k)
    if k > 3:                      # 另报「3 个 train 中真实存在的类」的 macro-F1
        mf3, f1_3 = macro_f1(y, p, 3)
    else:
        mf3, f1_3 = mf, f1s
    conf = [[0] * k for _ in range(k)]
    for a, b in zip(y, p):
        conf[a][b] += 1
    rec = [conf[c][c] / max(1, sum(conf[c])) for c in range(k)]
    return {"n": len(y), "acc": round(acc, 4), "macro_f1": round(mf, 4),
            "macro_f1_3cls": round(mf3, 4),
            "recall": [round(x, 4) for x in rec],
            "f1": [round(x, 4) for x in f1s],
            "f1_3cls": [round(x, 4) for x in f1_3], "confusion": conf}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = C.Spec()
    tab = MM.skeleton_table()
    k = len(MM.MODES)
    idx = {m: i for i, m in enumerate(MM.MODES)}
    print(f"[device] {device}", flush=True)

    rows = {s: C.load_rows(p) for s, p in SPLIT_PATH.items()}
    blob = {s: C.encode_rows(rows[s], spec, device) for s in SPLIT_PATH}
    # 目标 = A(gold 骨架) 的 5 类序号（陈述0 疑问1 祈使2 感叹3 反问4）
    y = {s: [idx[MM.mode_of_skel(tab[r["skel_id"]])] for r in rows[s]] for s in rows}
    print("[target] train 分布 =", dict(Counter(MM.MODES[i] for i in y["train"])), flush=True)

    n = blob["train"]["n"]
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(n, generator=g)
    n_val = max(1, n // 10)
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    if a.smoke:
        tr_idx, val_idx = tr_idx[:200], val_idx[:50]

    X = blob["train"]["v_sent"]
    ytr = torch.tensor(y["train"], dtype=torch.long)
    out: dict = {"device": device, "target": "A(gold skeleton)", "modes": list(MM.MODES),
                 "n_train": int(len(tr_idx)), "n_val": int(len(val_idx)), "seeds": {}}

    WEIGHTS.mkdir(exist_ok=True)
    for seed in (42, 43):
        t0 = time.time()
        torch.manual_seed(seed)
        head = nn.Linear(spec.hidden, k).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=1e-2)
        best, best_ep, bad = -1.0, -1, 0
        best_sd = None
        for ep in range(60):
            head.train()
            p = torch.randperm(len(tr_idx), generator=torch.Generator().manual_seed(seed * 1000 + ep))
            order = tr_idx[p]
            for i in range(0, len(order), 256):
                b = order[i:i + 256]
                logits = head(X[b].to(device))
                loss = nn.functional.cross_entropy(logits, ytr[b].to(device))
                opt.zero_grad()
                loss.backward()
                opt.step()
            head.eval()
            with torch.no_grad():
                vl = head(X[val_idx].to(device)).argmax(-1).cpu().tolist()
            vacc = sum(1 for t, q in zip(vl, ytr[val_idx].tolist()) if t == q) / len(vl)
            if vacc > best + 1e-6:
                best, best_ep, bad = vacc, ep, 0
                best_sd = {k2: v.detach().clone() for k2, v in head.state_dict().items()}
            else:
                bad += 1
                if bad >= 10:
                    break
        head.load_state_dict(best_sd)
        head.eval()
        torch.save(best_sd, WEIGHTS / f"probe_s{seed}.pt")
        rec = {"val_acc": round(best, 4), "best_epoch": best_ep,
               "epochs_run": ep + 1, "secs": round(time.time() - t0, 1),
               "splits": {}, "pred": {}}
        with torch.no_grad():
            for s in rows:
                pr = head(blob[s]["v_sent"].to(device)).softmax(-1).cpu()
                ap_ = pr.argmax(-1).tolist()
                rec["splits"][s] = report(y[s], ap_, k)
                rec["pred"][s] = {"argmax": ap_,
                                  "probs": [[round(float(x), 6) for x in row] for row in pr]}
        out["seeds"][str(seed)] = rec
        print(f"[probe s{seed}] val={rec['val_acc']} ep={best_ep} "
              + " ".join(f"{s}:{rec['splits'][s]['acc']}/{rec['splits'][s]['macro_f1']}"
                         for s in rows), flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "probe.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    slim = {"seeds": {s: {"pred": r["pred"]} for s, r in out["seeds"].items()}}
    (RESULTS / "probe_pred.json").write_text(json.dumps(slim, ensure_ascii=False),
                                             encoding="utf-8")
    print(f"[done] → {RESULTS / 'probe.json'}", flush=True)


if __name__ == "__main__":
    main()
