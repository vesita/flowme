"""core_ndb 阶段 2 · P3 推理代价：同一 16 条文本 × 4 卡，三种形态各跑 ≥3 遍取中位。

三种形态：
  base —— 冻结核 + 冻结卡，无任何记忆（参照）
  C    —— 核级记忆（`CoreNDBEngine`，训练过的门控）
  L    —— 卡级记忆（person 卡 `MentionNDB`，训练过的门控）
另报表显存（batch=1 与训练 batch=64 两个口径）。
判据见 `PREREG_PHASE2.md` §2/§3（P3 不设门槛，只报数值）。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in ("src", "experiments/additivity", "experiments/capability_map",
          "experiments/core_keep", "experiments/core_ndb"):
    sys.path.insert(0, str(ROOT / p))

import torch  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.tasks.engine import MultiTaskEngine  # noqa: E402
from dtseek.tasks.artifacts import read_card  # noqa: E402

from train_core_keep import FROZEN_CARD, OLD_CARDS  # noqa: E402
from pipeline import CoreNDBEngine  # noqa: E402
from eval_phase2 import BASE, build_engine  # noqa: E402

TEXTS = None  # 延迟：与阶段 1 同一批 16 条


def load_texts() -> list[str]:
    import pickle
    data = pickle.loads(Path("experiments/core_keep/cache/person_6000.pkl")
                        .read_bytes())
    out = []
    for s in data:
        t = s["text"].strip()
        if t and t not in out:
            out.append(t)
        if len(out) == 16:
            break
    return out


def time_engine(eng, texts: list[str], repeats: int = 3) -> dict:
    per = []
    for r in range(repeats):
        t0 = time.perf_counter()
        n = 0
        for t in texts:
            for c in OLD_CARDS:
                eng.predict(t, tasks=[c])
                n += 1
        torch.cuda.synchronize()
        per.append(1000 * (time.perf_counter() - t0) / n)
    return {"ms_per_predict": statistics.median(per),
            "runs_ms": [round(x, 2) for x in per]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "results_phase2.json"))
    args = ap.parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    texts = load_texts()
    print(f"[P3] {len(texts)} 条 × {len(OLD_CARDS)} 卡 × 3 遍", flush=True)

    res = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}

    # ---- base：冻结核 + 冻结卡，无记忆 ----
    base = MultiTaskEngine(base_path=BASE, auto_attach=False, device=str(device))
    for n in OLD_CARDS:
        base.attach(FROZEN_CARD.format(n, 42))
    out = {"base": time_engine(base, texts)}
    print("  base", json.dumps(out["base"]), flush=True)

    for arm in ("C", "L"):
        for seed in (42, 43):
            eng = build_engine(str(HERE / "cards" / f"p2_{arm}_s{seed}.pt"),
                               arm, seed, device)
            key = f"{arm}/s{seed}"
            out[key] = time_engine(eng, texts)
            print(f"  {key} {json.dumps(out[key])}", flush=True)
            if arm == "C":
                out[key]["table_gb_batch1"] = eng.core.table_gb(1)
                out[key]["table_gb_batch64"] = eng.core.table_gb(64)
                out[key]["scales"] = {"write": float(eng.core.write_scale),
                                      "read": float(eng.core.read_scale)}
            else:
                ndb = eng.ndbs["person"]
                out[key]["table_gb_batch1"] = ndb.table_gb(1)
                out[key]["table_gb_batch64"] = ndb.table_gb(64)
                out[key]["gate_bias"] = {"write": float(ndb.write_gate.bias),
                                         "read": float(ndb.read_gate.bias)}
            del eng
            torch.cuda.empty_cache()

    res["p3_infer"] = out
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                         default=float), encoding="utf-8")
    print("P3_INFER " + json.dumps(out, ensure_ascii=False, default=float), flush=True)
    print("P3_TIMING_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
