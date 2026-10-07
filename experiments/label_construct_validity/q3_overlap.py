#!/usr/bin/env python3
"""P5-Q3：emo 跨 split 文本重复复核 + 重复/不重复子集上的 probe acc 分解。

只读：复用 experiments/core_probe 的 build_sets / extract / fit_probe（不改其文件）。
产物只落 experiments/label_construct_validity/results/。
设备写死 cpu（核冻结线性 probe，CPU 与 GPU 只差浮点末位；避免占用 GPU）。
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = HERE / "results"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "experiments/core_probe"))

import numpy as np  # noqa: E402

import run_probe as rp  # noqa: E402


def overlap(sets: dict) -> dict:
    """逐任务：test 行（或提及行）的文本是否在 train 文本集合里出现过。"""
    res = {}
    for t, d in sets.items():
        tr = set(d["tr_text"])
        te = d["te_text"]
        hit = sum(1 for x in te if x in tr)
        res[t] = {"n_train_rows": len(d["tr_text"]), "n_test_rows": len(te),
                  "n_test_rows_text_in_train": hit,
                  "frac": round(hit / max(len(te), 1), 6),
                  "n_unique_train_text": len(tr),
                  "frac_unique_test_text_in_train": round(
                      len({x for x in te if x in tr}) / max(len({x for x in te}), 1), 6)}
    return res


def se(p, n):
    return math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def main() -> int:
    sets = rp.build_sets()
    ov = overlap(sets)
    print("== Q3 跨 split 文本重复（行级，test 文本 ∈ train 文本集合）==")
    for t, v in ov.items():
        print(f"  {t:6s} {v['n_test_rows_text_in_train']}/{v['n_test_rows']} = "
              f"{v['frac']*100:.2f}%   (唯一文本口径 {v['frac_unique_test_text_in_train']*100:.2f}%)")

    # ── emo probe：同 core_probe P1 口径（5 层 × 2 seed），另按重复与否拆分 ──
    import torch  # noqa: E402
    from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
    from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402

    device = torch.device("cpu")
    tok = NanoCharTokenizer()
    enc, _ = load_base_encoder(str(rp.BASE), device)
    enc.eval()

    emo = sets["emo"]
    Xtr, meta_tr = rp.extract_texts(enc, tok, emo["tr_text"], rp.MAX_LEN["emo"], device)
    Xte, meta_te = rp.extract_texts(enc, tok, emo["te_text"], rp.MAX_LEN["emo"], device)

    tr_set = set(emo["tr_text"])
    dup = np.array([x in tr_set for x in emo["te_text"]])

    rows = []
    for seed in rp.SEEDS:
        for li, layer in enumerate(rp.LAYERS):
            acc, pred, _ = rp.fit_probe(Xtr[:, li, :], emo["tr_y"],
                                        Xte[:, li, :], emo["te_y"],
                                        emo["classes"], seed)
            yte = np.asarray(emo["te_y"])
            rows.append({"seed": seed, "layer": layer, "acc": acc,
                         "acc_dup": float((pred[dup] == yte[dup]).mean()) if dup.any() else None,
                         "acc_nodup": float((pred[~dup] == yte[~dup]).mean()) if (~dup).any() else None,
                         "n_dup": int(dup.sum()), "n_nodup": int((~dup).sum())})

    best = max(rows, key=lambda r: r["acc"])
    print("\n== emo 分层 probe（核冻结，2 seed，cpu）==")
    for r in rows:
        print(f"  seed{r['seed']} {r['layer']:5s} acc={r['acc']*100:.2f} "
              f"(dup {r['acc_dup']*100:.2f} n={r['n_dup']} | "
              f"nodup {r['acc_nodup']*100:.2f} n={r['n_nodup']})")

    # 多数类 / 规则在两个子集上的值（Q2 并列口径）
    yte = np.array(emo["te_y"])
    maj = max(set(emo["tr_y"]), key=emo["tr_y"].count)
    sub = {"n_dup": int(dup.sum()), "n_nodup": int((~dup).sum()),
           "majority_class": int(maj),
           "maj_dup": float((yte[dup] == maj).mean()),
           "maj_nodup": float((yte[~dup] == maj).mean())}

    out = {"overlap": ov, "probe_rows": rows, "best": best, "subset_baselines": sub,
           "extract_meta": {"train": meta_tr, "test": meta_te},
           "device": "cpu", "note": "复用 experiments/core_probe 的 build_sets/extract/fit_probe，未改其文件"}
    (OUT / "q3_overlap.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\n写入 {OUT / 'q3_overlap.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
