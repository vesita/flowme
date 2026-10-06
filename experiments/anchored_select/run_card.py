"""锚定候选打标卡的单卡训练入口 —— 先打自检，再**原样委托** training/train_task_card.py。

    uv run python experiments/anchored_select/run_card.py \
        --out experiments/anchored_select/cards/s42.pt --seed 42

    随机标签对照（P3）：
    uv run python experiments/anchored_select/run_card.py --shuf --shuf-seed 42 \
        --out experiments/anchored_select/cards/shuf_s42.pt --seed 42

自检（每次训练必打印，PREREG §3）：
  SELFTEST_1  真实可训参数量 + 冻结实况（核 requires_grad 实测，不看注释）
  SELFTEST_DATA 训练集正负比与多数类基线
"""
from __future__ import annotations

import argparse
import json
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import anchored_card  # noqa: E402  import 即注册（本进程内）
from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402

TRAINER = ROOT / "training" / "train_task_card.py"


def selftest(base: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, ck = load_base_encoder(base, device)
    core_total = sum(p.numel() for p in enc.parameters())
    core_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    dec = RobustARSliceDecoder(hidden_dim=ck["hidden_dim"],
                               num_classes=anchored_card.SPEC.num_classes,
                               num_heads=4, num_layers=2)
    head = sum(p.numel() for p in dec.parameters())
    n = sum(1 for _ in open(anchored_card.DATA / "train.jsonl", encoding="utf-8"))
    print("SELFTEST_1 " + json.dumps({
        "card": "anchored_sel", "base": base, "device": str(device),
        "core_total": core_total, "core_trainable": core_train,
        "core_frozen": core_train == 0,
        "head_total": head, "head_trainable": head,
        "trainable_total": core_train + head,
        "n_train_samples": n, "max_len": anchored_card.SPEC.max_len,
        "max_text_len": 70,
    }, ensure_ascii=False), flush=True)
    if core_train != 0:
        raise SystemExit("SELFTEST_1 失败：核不是冻结的 ⇒ 本步作废")
    del enc, dec


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--shuf", action="store_true", help="随机标签对照臂")
    ap.add_argument("--shuf-seed", type=int, default=None)
    known, rest = ap.parse_known_args()

    if known.shuf:
        import os
        os.environ["DTSEEK_ANCHORED_SHUF"] = "1"
        os.environ["DTSEEK_ANCHORED_SHUF_SEED"] = str(known.shuf_seed or 0)
    selftest(known.base)

    sys.argv = [str(TRAINER), "--card", "anchored_sel",
                "--base", known.base, "--out", known.out, *rest]
    runpy.run_path(str(TRAINER), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
