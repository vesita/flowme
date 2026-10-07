#!/usr/bin/env python3
"""P8 老卡零损伤检查（N5 下半）：四张老卡只读评测，**结果写本目录**（不写 two_channel_head/）。

口径逐字复用 `two_channel_head/eval_old_cards.py`（OLD_SAMPLES / val = shuffle(seed)
后取前 max(200, n//10) / MultiTaskEngine + evaluate_task），另存
`checkpoints/base_encoder.pt` 与 `checkpoints/cards/*` 的 sha256。

用法：uv run python experiments/mode_conditioned_skel/old_cards.py --tag before|after
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.tasks.engine import MultiTaskEngine  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
OLD_SAMPLES = {"pronoun": 6000, "sentiment": 32000, "relation": 6000, "person": 6000}
NOISE_BAND = {"pronoun": 0.0283, "relation": 0.0139, "sentiment": 0.0041, "person": 0.0033}


def val_split(card, samples: int, seed: int):
    data = card.build_dataset(samples)
    random.Random(seed).shuffle(data)
    return data[: max(200, len(data) // 10)]


def sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["before", "after"])
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = {str(p.relative_to(ROOT)): sha256(p)
             for p in [ROOT / "checkpoints" / "base_encoder.pt"]}
    files |= {str(p.relative_to(ROOT)): sha256(p)
              for p in sorted((ROOT / "checkpoints" / "cards").glob("*.pt"))}

    cards = resolve_tasks(list(OLD_SAMPLES))
    vals = {n: val_split(cards[n], s, a.seed) for n, s in OLD_SAMPLES.items()}
    print(f"[data] seed={a.seed} " + " ".join(f"{k}={len(v)}" for k, v in vals.items()),
          flush=True)

    tok = NanoCharTokenizer()
    loaders = {n: DataLoader(GenericTaskDataset(vals[n], tok, cards[n].spec),
                             batch_size=64, shuffle=False) for n in OLD_SAMPLES}
    eng = MultiTaskEngine("checkpoints/base_encoder.pt", cards_dir="checkpoints/cards",
                          device=str(device))
    print(f"[engine] attached={eng.attached}", flush=True)

    out = {"tag": a.tag, "seed": a.seed, "device": str(device), "files_sha256": files,
           "val_n": {k: len(v) for k, v in vals.items()}, "cards": {}}
    for n in OLD_SAMPLES:
        t = time.perf_counter()
        m = evaluate_task(eng.doc_encoder, eng.decoders[n], loaders[n], device, eng.specs[n])
        out["cards"][n] = {k: (float(v) if isinstance(v, (int, float)) else str(v))
                           for k, v in m.items()}
        print(f"[old-card] {a.tag} {n:9s} em={m['exact_match']:.4f} "
              f"({time.perf_counter() - t:.1f}s)", flush=True)

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"old_cards_{a.tag}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {path}", flush=True)

    if a.tag == "after":
        pre = json.loads((RESULTS / "old_cards_before.json").read_text(encoding="utf-8"))
        print("\n[N5 老卡] Δ = after − before（对各自噪声带）", flush=True)
        for n in OLD_SAMPLES:
            b, af = pre["cards"][n]["exact_match"], out["cards"][n]["exact_match"]
            d = af - b
            band = NOISE_BAND[n]
            print(f"  {n:9s} before={b:.6f} after={af:.6f} Δ={d:+.6f} 带={band:.4f} "
                  f"带内={abs(d) <= band} 指标逐位同={pre['cards'][n] == out['cards'][n]} "
                  f"哈希同={pre['files_sha256'] == out['files_sha256']}", flush=True)
        json_out = {n: {"before": pre["cards"][n]["exact_match"],
                        "after": out["cards"][n]["exact_match"],
                        "delta": out["cards"][n]["exact_match"]
                                 - pre["cards"][n]["exact_match"],
                        "noise_band": NOISE_BAND[n],
                        "metric_identical": pre["cards"][n] == out["cards"][n]}
                    for n in OLD_SAMPLES}
        json_out["files_sha256_identical"] = pre["files_sha256"] == out["files_sha256"]
        (RESULTS / "old_cards_delta.json").write_text(
            json.dumps(json_out, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
