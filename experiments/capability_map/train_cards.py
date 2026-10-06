"""在冻结核上训练「旁路档 / 冻结档」任务卡 —— **薄封装，不改 train_bypass_card.py**。

它只做一件事：把任务卡单例的 `build_dataset` 换成「§3 定义的逆置换排列」，
于是 `train_bypass_card.py` 内部那次 `random.Random(seed).shuffle` 之后：
  - `val` 恰好 = `eval_S`（探针、联合档都评这一份，逐样本对齐）；
  - `train` = 其余（`eval_S` 不进训练集）。

用法与 train_bypass_card.py 完全一致，只是入口换成这个：
    uv run python experiments/capability_map/train_cards.py \
        --mode bypass --card pronoun --seed 42 \
        --out experiments/capability_map/cards/pronoun_bypass_s42.pt
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "additivity"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "training"))

from dtseek.tasks.plugin import DEFAULT_SEED, resolve_tasks  # noqa: E402

from prepare import CACHE, TASK_SAMPLES  # noqa: E402


def patch(card_name: str, seed: int) -> None:
    """把该卡单例的 build_dataset 换成预排好的排列（含自检）。"""
    card = resolve_tasks([card_name])[card_name]
    path = CACHE / f"{card_name}_ordered_s{seed}.pkl"
    if not path.exists():
        raise SystemExit(f"缺少 {path} —— 先跑 prepare.py")
    ordered = pickle.loads(path.read_bytes())

    def fake(target_samples: int = 0, seed: int = DEFAULT_SEED):  # noqa: A002
        if target_samples and target_samples != TASK_SAMPLES[card_name]:
            raise SystemExit(
                f"--samples={target_samples} 与预注册 task_samples="
                f"{TASK_SAMPLES[card_name]} 不一致（会导致 eval 划分漂移）")
        return list(ordered)

    card.build_dataset = fake   # type: ignore[method-assign]
    print(f"[patch] {card_name} s{seed}: build_dataset → {path.name} (n={len(ordered)})", flush=True)


def main(argv=None) -> int:
    import argparse
    args = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--card", default=None)
    ap.add_argument("--seed", type=int, default=42)
    known, _ = ap.parse_known_args(args)   # 只读，不改 args：原样透传给 train_bypass_card
    if not known.card:
        raise SystemExit("必须给 --card")
    if known.seed not in (42, 43):
        raise SystemExit(f"预注册只有 seed 42/43，收到 {known.seed}")
    patch(known.card, known.seed)

    import train_bypass_card as T  # 延迟 import：先完成单例补丁
    return T.main(args)


if __name__ == "__main__":
    sys.exit(main())
