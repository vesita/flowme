"""core_generalize 数据准备：建/复用 5 份核任务数据 + 2 份 hold-out 数据（带缓存），
并算出每个 hold-out 的「盲猜地板」= val 里无切片样本占比（永远预测空切片集的 exact_match）。

只写 experiments/core_generalize/{cache,*.json}；capability_map 的 cache 只读复用
（那 5 份已验过，含 sentiment 的 4 分钟构建）。
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "capability_map"))

from dtseek.tasks.plugin import DEFAULT_SEED, resolve_tasks  # noqa: E402

#: 与 experiments/capability_map/prepare.py 的 TASK_SAMPLES 逐项一致（核臂口径）
CORE_SAMPLES = {"pronoun": 6000, "sentiment": 32000, "relation": 6000,
                "person": 6000, "negation": 6000}
#: hold-out 候选（先验；实际入选由可学性预检决定）
HOLDOUT_SAMPLES = {"idiom": 6000, "ownership": 6000}
SEEDS = (42, 43)

HERE = Path(__file__).parent
CACHE = HERE / "cache"
CAPMAP_CACHE = ROOT / "experiments" / "capability_map" / "cache"


def get_raw(name: str, n: int, card) -> list[dict]:
    CACHE.mkdir(parents=True, exist_ok=True)
    mine = CACHE / f"{name}_{n}.pkl"
    if mine.exists():
        return pickle.loads(mine.read_bytes())
    if name in CORE_SAMPLES:
        # capability_map 已建好并验过（含 4 分钟的 sentiment）：只读复用同一份数据
        src = CAPMAP_CACHE / f"{name}_{n}_{DEFAULT_SEED}.pkl"
        if src.exists():
            shutil.copyfile(src, mine)
            print(f"[reuse] {name} ← {src.name} (n={n})", flush=True)
            return pickle.loads(mine.read_bytes())
    t = time.perf_counter()
    data = card.build_dataset(n, seed=DEFAULT_SEED)
    print(f"[build] {name}: {len(data)} 条 ({time.perf_counter()-t:.1f}s)", flush=True)
    mine.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    return data


def split_of(data: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """与 train_multitask / capability_map 同口径：shuffle(seed) 后前 max(200, n//10) 是 val。"""
    work = list(data)
    random.Random(seed).shuffle(work)
    n_val = max(200, len(work) // 10)
    return work[:n_val], work[n_val:]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floors-only", action="store_true")
    args = ap.parse_args(argv)

    names = list(CORE_SAMPLES) + list(HOLDOUT_SAMPLES)
    cards = resolve_tasks(names)
    info: dict = {"core_samples": CORE_SAMPLES, "holdout_samples": HOLDOUT_SAMPLES,
                  "default_seed": DEFAULT_SEED, "seeds": list(SEEDS), "tasks": {}}

    for name, n in list(CORE_SAMPLES.items()) + list(HOLDOUT_SAMPLES.items()):
        card = cards[name]
        raw = get_raw(name, n, card)
        t: dict = {"n_total": len(raw), "num_classes": card.spec.num_classes,
                   "class_names": list(card.spec.class_names),
                   "max_steps": card.spec.max_steps, "max_len": card.spec.max_len}
        if name in HOLDOUT_SAMPLES:
            for S in SEEDS:
                ev, tr = split_of(raw, S)
                n_bg = sum(1 for s in ev if len(s["spans"]) == 0)
                t.setdefault("splits", {})[str(S)] = {
                    "n_eval": len(ev), "n_train": len(tr), "n_bg": n_bg,
                    "floor_exact_empty_predictor": n_bg / len(ev),
                    "n_gold_spans_mean": sum(len(s["spans"]) for s in ev) / len(ev),
                }
                print(f"[holdout] {name} s{S}: eval={len(ev)} train={len(tr)} "
                      f"n_bg={n_bg} floor={n_bg/len(ev):.4f}", flush=True)
        info["tasks"][name] = t
        print(f"[data] {name}: n={len(raw)} C={t['num_classes']} max_steps={t['max_steps']}",
              flush=True)

    out = HERE / "data_info.json"
    out.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}")
    print("PREPARE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
