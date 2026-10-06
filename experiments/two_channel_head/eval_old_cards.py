#!/usr/bin/env python3
"""W5 老卡零损伤检查：核冻结 ⇒ 四张老卡的 Δ 应当 **= 0**（PREREG §4-W5）。

训前 / 训后各跑一次（`--tag before|after`），报：
  · 四卡 exact_match 与整份指标 dict；
  · **Δ = after − before**，与**各自**噪声带（pronoun 2.83 / sentiment 0.41 /
    relation 1.39 / person 0.33 pt）并列；
  · `checkpoints/base_encoder.pt` 与四张卡文件的 **sha256**（训前 = 训后 ⇒ 核与卡没被动过）。

口径照抄 `experiments/additivity/eval_old_cards.py`：
  OLD_SAMPLES、val = shuffle(seed) 后前 max(200, n//10)、MultiTaskEngine + evaluate_task。

用法：
  uv run python experiments/two_channel_head/eval_old_cards.py --tag before
  uv run python experiments/two_channel_head/eval_old_cards.py --tag after
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
OLD_SAMPLES = {"pronoun": 6000, "sentiment": 32000, "relation": 6000, "person": 6000}
NOISE_BAND = {"pronoun": 0.0283, "relation": 0.0139, "sentiment": 0.0041, "person": 0.0033}


def val_split(card, samples: int, seed: int):
    data = card.build_dataset(samples)
    random.Random(seed).shuffle(data)
    return data[: max(200, len(data) // 10)]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    h.update(p.read_bytes())
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, choices=["before", "after"])
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--cards-dir", default="checkpoints/cards")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    files = {}
    base_p = ROOT / a.base
    files[str(base_p.relative_to(ROOT))] = sha256(base_p)
    cards_dir = ROOT / a.cards_dir
    for f in sorted(cards_dir.glob("*.pt")) + sorted(cards_dir.glob("*.yaml")) + sorted(cards_dir.glob("*.yml")):
        files[str(f.relative_to(ROOT))] = sha256(f)

    cards = resolve_tasks(list(OLD_SAMPLES))
    vals = {n: val_split(cards[n], s, a.seed) for n, s in OLD_SAMPLES.items()}
    print(f"[data] seed={a.seed} " + " ".join(f"{k}={len(v)}" for k, v in vals.items()),
          flush=True)

    tok = NanoCharTokenizer()
    loaders = {n: DataLoader(GenericTaskDataset(vals[n], tok, cards[n].spec),
                             batch_size=64, shuffle=False) for n in OLD_SAMPLES}
    eng = MultiTaskEngine(a.base, cards_dir=a.cards_dir, device=str(device))
    print(f"[engine] attached={eng.attached}", flush=True)

    out = {"tag": a.tag, "seed": a.seed, "device": str(device), "files_sha256": files,
           "val_n": {k: len(v) for k, v in vals.items()}, "cards": {}}
    for n in OLD_SAMPLES:
        t = time.perf_counter()
        m = evaluate_task(eng.doc_encoder, eng.decoders[n], loaders[n], device,
                          eng.specs[n])
        out["cards"][n] = {k: (float(v) if isinstance(v, (int, float)) else str(v))
                           for k, v in m.items()}
        print(f"[old-card] {a.tag} {n:9s} em={m['exact_match']:.4f} "
              f"({time.perf_counter() - t:.1f}s)", flush=True)

    path = HERE / "results" / f"old_cards_{a.tag}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[done] {path}", flush=True)

    if a.tag == "after":
        pre = json.loads((HERE / "results" / "old_cards_before.json").read_text(
            encoding="utf-8"))
        print("\n[W5] Δ = after − before（对各自噪声带）", flush=True)
        for n in OLD_SAMPLES:
            b, af = pre["cards"][n]["exact_match"], out["cards"][n]["exact_match"]
            d = af - b
            band = NOISE_BAND[n]
            same_dict = pre["cards"][n] == out["cards"][n]
            files_same = pre["files_sha256"] == out["files_sha256"]
            print(f"  {n:9s} before={b:.6f} after={af:.6f} Δ={d:+.6f} "
                  f"噪声带={band:.4f} 带内={abs(d) <= band} 指标逐位同={same_dict} "
                  f"文件哈希逐位同={files_same}", flush=True)
        (HERE / "results" / "old_cards_delta.json").write_text(json.dumps(
            {n: {"before": pre["cards"][n]["exact_match"],
                 "after": out["cards"][n]["exact_match"],
                 "delta": out["cards"][n]["exact_match"] - pre["cards"][n]["exact_match"],
                 "noise_band": NOISE_BAND[n],
                 "metric_dict_identical": pre["cards"][n] == out["cards"][n]}
             for n in OLD_SAMPLES} |
            {"files_sha256_identical": pre["files_sha256"] == out["files_sha256"]},
            ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
