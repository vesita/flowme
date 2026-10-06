#!/usr/bin/env python3
"""跑后**描述性**拆解（不参与判据，PREREG 判据跑前已写死）：臂 A vs 臂 D 的 adv 逐 ctype / 逐词。

只读本目录 `cache/` + `weights/` + `results/`，不训练、不改判据。

用法：uv run python experiments/select_pool/breakdown.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_pool import PoolModel, PoolSpec  # noqa: E402
from train_pool import CACHE, DATA_ARM, RESULTS, WEIGHTS, load_rows  # noqa: E402

ARMS = ("A", "D")
SEEDS = (42, 43)


def load_blob() -> dict:
    paths = sorted(CACHE.glob(f"{DATA_ARM}_*_tok_ctx*_cand*.pt"))
    assert paths, "先跑 train_pool.py 建 token 缓存"
    return torch.load(paths[-1], map_location="cpu")


def per_sample(model: PoolModel, blob: dict, device: str, bs: int = 1024) -> torch.Tensor:
    model.eval()
    ok: list[torch.Tensor] = []
    with torch.no_grad():
        for i in range(0, blob["n"], bs):
            j = torch.arange(i, min(i + bs, blob["n"]))
            logits = model.forward_tokens(blob["h_ctx"][j].to(device),
                                          blob["m_ctx"][j].to(device),
                                          blob["h_cand"][j].to(device),
                                          blob["m_cand"][j].to(device))
            ok.append((logits.argmax(-1) == blob["labels"][j].to(device)).cpu())
    return torch.cat(ok)


def bucket(rows: list[dict], ok: torch.Tensor, key, mask: torch.Tensor) -> dict:
    agg: dict[str, list[bool]] = defaultdict(list)
    for r, o, m in zip(rows, ok.tolist(), mask.tolist()):
        if m:
            agg[str(key(r))].append(o)
    out = {}
    for k, v in sorted(agg.items(), key=lambda kv: -len(kv[1])):
        n = len(v)
        acc = sum(v) / n
        se = math.sqrt(0.25 / n) if n > 1 else float("nan")
        out[k] = {"n": n, "acc": round(acc, 4), "se": round(se, 4),
                  "margin_over_se": round((acc - 0.5) / se, 3) if n > 1 else None}
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    spec = PoolSpec.from_build_spec()
    blob = load_blob()
    rows = load_rows(DATA_ARM)
    assert len(rows) == blob["n"]
    adv = blob["splits"] == 2
    held = adv & (blob["ctype"] == 1)
    shif = adv & (blob["ctype"] == 2)
    report: dict = {"note": "跑后描述性拆解，不参与 PREREG 判据", "arms": {}}

    for arm in ARMS:
        for seed in SEEDS:
            wp = WEIGHTS / f"{arm}_s{seed}.pt"
            if not wp.exists():
                continue
            model = PoolModel(arm, spec).to(device)
            sd = torch.load(wp, map_location="cpu", weights_only=True)
            model.agg.load_state_dict({k[4:]: v for k, v in sd.items() if k.startswith("agg.")},
                                      strict=False) if any(
                k.startswith("agg.") for k in sd) else None
            model.head.load_state_dict({k[5:]: v for k, v in sd.items() if k.startswith("head.")})
            ok = per_sample(model, blob, device)
            key = f"{arm}_s{seed}"
            report["arms"][key] = {
                "heldout_by_word": bucket(rows, ok, lambda r: r["sub_word"], held),
                "shifted_by_word": bucket(rows, ok, lambda r: r["sub_word"], shif),
            }
            del model
            torch.cuda.empty_cache()
            print(f"[done] {key}", flush=True)

    p = RESULTS / "breakdown.json"
    p.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印 heldout 逐词对照（n ≥ 15）
    print("\n| sub_word | n | A s42 | A s43 | D s42 | D s43 |")
    print("|---|---|---|---|---|---|")
    words = {}
    for key, v in report["arms"].items():
        for w, e in v["heldout_by_word"].items():
            if e["n"] >= 15:
                words.setdefault(w, {"n": e["n"]})[key] = e["acc"]
    for w, d in words.items():
        cells = " | ".join(f"{d.get(f'{a}_s{s}', float('nan')):.2f}"
                           for a in ("A", "D") for s in (42, 43))
        print(f"| {w} | {d['n']} | {cells} |")
    print(f"→ {p}")


if __name__ == "__main__":
    main()
