"""价值观卡的单卡训练入口 —— 先打自检，再**原样委托** training/train_task_card.py。

    uv run python experiments/value_card/run_card.py \
        --out experiments/value_card/cards/value_s42.pt --seed 42

    随机标签对照（V-bonus，PREREG §4）：
    uv run python experiments/value_card/run_card.py --shuf --shuf-seed 42007 \
        --out experiments/value_card/cards/shuf_s42.pt --seed 42

自检（每次训练必打印，PREREG §3）：
  SELFTEST_1   真实可训参数量 + 冻结实况（核 requires_grad 实测，不看注释）
  SELFTEST_DATA 训练集类别比 / 多数类基线 / 朴素规则基线 / shuf 状态
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

import build_data as bd  # noqa: E402  朴素规则基线与词表的唯一来源
import value_card  # noqa: E402  import 即注册（本进程内）
from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402

TRAINER = ROOT / "training" / "train_task_card.py"


def selftest(base: str, shuf: bool, shuf_seed: int | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, ck = load_base_encoder(base, device)
    core_total = sum(p.numel() for p in enc.parameters())
    core_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    dec = RobustARSliceDecoder(hidden_dim=ck["hidden_dim"],
                               num_classes=value_card.SPEC.num_classes,
                               num_heads=4, num_layers=2)
    head = sum(p.numel() for p in dec.parameters())

    rows = [json.loads(l) for l in open(value_card.DATA / "train.jsonl", encoding="utf-8")]
    labels = [int(r["label"]) for r in rows]
    if shuf:
        import random
        rng = random.Random(shuf_seed or 0)
        labels = labels[:]
        rng.shuffle(labels)
    dist = Counter(labels)
    n = len(labels)
    naive_train = bd.naive_cls_acc(rows)     # 朴素规则在**原始**训练集上的类准确率
    naive_test = bd.naive_cls_acc(
        [json.loads(l) for l in open(value_card.DATA / "test.jsonl", encoding="utf-8")])
    naive_adv = bd.naive_cls_acc(
        [json.loads(l) for l in open(value_card.DATA / "adversarial.jsonl", encoding="utf-8")])
    max_len = max(len(r["text"]) for r in rows)

    print("SELFTEST_1 " + json.dumps({
        "card": "value_judge", "base": base, "device": str(device),
        "core_total": core_total, "core_trainable": core_train,
        "core_frozen": core_train == 0,
        "head_total": head, "head_trainable": head,
        "trainable_total": core_train + head,
        "n_train_samples": n, "max_len": value_card.SPEC.max_len,
        "max_text_len": max_len,
        "cls_weight_bg": value_card.SPEC.cls_weight_bg,
        "max_steps": value_card.SPEC.max_steps,
    }, ensure_ascii=False), flush=True)
    print("SELFTEST_DATA " + json.dumps({
        "split": "train", "n": n, "class_dist": {str(k): dist[k] for k in sorted(dist)},
        "majority_baseline": round(max(dist.values()) / n, 4),
        "blind_guess_3cls": 0.3333,
        "naive_rule_train": round(naive_train, 4),
        "naive_rule_test": round(naive_test, 4),
        "naive_rule_adversarial": round(naive_adv, 4),
        "shuf": shuf, "shuf_seed": shuf_seed,
    }, ensure_ascii=False), flush=True)
    if core_train != 0:
        raise SystemExit("SELFTEST_1 失败：核不是冻结的 ⇒ 本步作废")
    if value_card.SPEC.max_len < max_len:
        raise SystemExit(f"SELFTEST_1 失败：max_len={value_card.SPEC.max_len} < 实测最长 {max_len}")
    del enc, dec


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--shuf", action="store_true", help="随机标签对照臂")
    ap.add_argument("--shuf-seed", type=int, default=None)
    known, rest = ap.parse_known_args()

    if known.shuf:
        os.environ["DTSEEK_VALUE_SHUF"] = "1"
        os.environ["DTSEEK_VALUE_SHUF_SEED"] = str(known.shuf_seed or 0)
    selftest(known.base, known.shuf, known.shuf_seed)

    sys.argv = [str(TRAINER), "--card", "value_judge",
                "--base", known.base, "--out", known.out, *rest]
    runpy.run_path(str(TRAINER), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
