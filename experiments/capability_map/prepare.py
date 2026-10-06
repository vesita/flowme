"""能力可加性筛查：构建 5 份数据集（带缓存）并导出 §3 定义的 held-out 划分。

不改 src/ training/ tests/ dev-notes/ experiments/additivity/ 下任何文件；
只写本目录与 logs/、/tmp。
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "additivity"))

from dtseek.tasks.plugin import DEFAULT_SEED, resolve_tasks  # noqa: E402

CAPS = ["pronoun", "sentiment", "relation", "person", "negation"]
#: 与 checkpoints/arm_neg5_seed{42,43}.pt 的 train_args.task_samples 逐项一致
TASK_SAMPLES = {"pronoun": 6000, "sentiment": 32000, "relation": 6000,
                "person": 6000, "negation": 6000}
SEEDS = (42, 43)
CACHE = Path(__file__).parent / "cache"


def get_raw(card, name: str) -> list[dict]:
    """构建（或从缓存读）某能力的原始数据集。缓存按 (能力, 样本量, DEFAULT_SEED) 键控。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    n = TASK_SAMPLES[name]
    path = CACHE / f"{name}_{n}_{DEFAULT_SEED}.pkl"
    if path.exists():
        t = time.perf_counter()
        data = pickle.loads(path.read_bytes())
        print(f"[cache] {name}: {len(data)} 条 ← {path.name} ({time.perf_counter()-t:.1f}s)", flush=True)
        return data
    t = time.perf_counter()
    data = card.build_dataset(n, seed=DEFAULT_SEED)
    print(f"[build] {name}: {len(data)} 条 ({time.perf_counter()-t:.1f}s)", flush=True)
    path.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    return data


def split_of(raw: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """§3 的定义：eval = shuffle(seed) 后前 max(200, n//10)，train = 其余。"""
    data = copy.deepcopy(raw)
    random.Random(seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    return data[:n_val], data[n_val:]


def ordered_for(raw: list[dict], seed: int, ev: list[dict], tr: list[dict]) -> list[dict]:
    """返回一个排列 P，使得 `random.Random(seed).shuffle(P)` 之后 `P[:n_val] == ev`。

    `train_bypass_card.py` 内部就是 `random.Random(args.seed).shuffle(data)` 然后取前 10% 当 val。
    我们把原始列表预先按**逆置换**排好，于是它那次 shuffle 之后 val 恰好是我们 §3 定义的
    eval_S —— 于是训练用的样本与评测用的样本逐条对齐，且 eval_S 不进训练集。
    """
    n = len(ev) + len(tr)
    assert n == len(raw)
    idx = list(range(n))
    random.Random(seed).shuffle(idx)          # shuffle 后位置 i 的元素来自原下标 idx[i]
    desired = ev + tr                          # 我们想要的输出顺序
    out = [None] * n
    for pos, src in enumerate(idx):            # out[idx[pos]] = desired[pos] ⇒ out[src] = desired[pos(src)]
        out[src] = desired[pos]
    # 自检：重放一次必须还原
    chk = list(out)
    random.Random(seed).shuffle(chk)
    n_val = max(200, n // 10)
    assert [id(x) for x in chk[:n_val]] == [id(x) for x in ev], "逆置换自检失败"
    assert len(chk[n_val:]) == len(tr)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).parent / "splits.json"))
    args = ap.parse_args(argv)

    from nano_char_tokenizer import NanoCharTokenizer
    from dtseek.tasks.runtime import GenericTaskDataset

    cards = resolve_tasks(CAPS)
    info: dict = {"caps": CAPS, "task_samples": TASK_SAMPLES,
                  "default_seed": DEFAULT_SEED, "seeds": list(SEEDS), "eval": {}}
    for name in CAPS:
        raw = get_raw(cards[name], name)
        card = cards[name]
        for S in SEEDS:
            ev, tr = split_of(raw, S)
            # 用 GenericTaskDataset 同一套 clamped 目标复算 n_cls / n_bg，供与 arm_neg5 对账
            ds = GenericTaskDataset(ev, NanoCharTokenizer(), card.spec)
            n_bg = sum(1 for i in range(len(ds)) if float(ds[i]["is_bg"]) > 0.5)
            n_cls = sum(1 for i in range(len(ds)) if int(ds[i]["labels"][0]) > 0)
            info["eval"].setdefault(name, {})[str(S)] = {
                "n_eval": len(ev), "n_train": len(tr), "n_cls": n_cls, "n_bg": n_bg,
                "num_classes": card.spec.num_classes,
                "class_names": list(card.spec.class_names),
            }
            print(f"[split] {name} s{S}: eval={len(ev)} (cls={n_cls} bg={n_bg}) train={len(tr)} "
                  f"C={card.spec.num_classes}", flush=True)
            # 供 train_cards.py 使用的逆置换排列
            ordered = ordered_for(raw, S, ev, tr)
            p = CACHE / f"{name}_ordered_s{S}.pkl"
            p.write_bytes(pickle.dumps(ordered, protocol=pickle.HIGHEST_PROTOCOL))
            print(f"[ordered] {p.name} n={len(ordered)}", flush=True)

    Path(args.out).write_text(json.dumps(info, ensure_ascii=False, indent=2))
    print(f"[save] {args.out}")
    print("PREPARE_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
