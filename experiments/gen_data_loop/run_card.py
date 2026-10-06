"""候选有效性卡的单跑入口 —— 先打自检，再**原样委托** training/train_task_card.py。

    提案臂 P：uv run python experiments/gen_data_loop/run_card.py \
        --arm P --out experiments/gen_data_loop/cards/cand_P_s42.pt --seed 42
    注入臂 C：... --arm C --out .../cand_C_s42.pt
    随机标签对照：... --shuf --out .../cand_P_shuf_s42.pt   （shuf-seed = seed×1000+7）

自检（每次训练必打印，PREREG_PHASE2 §2.1）：
  SELFTEST_1     真实可训参数量 + 冻结实况（核 requires_grad 实测，不看注释）
  SELFTEST_DATA  训练集正负比与多数类基线（+ shuf 标志）
  SELFTEST_TRUNC 静默截断守卫（超长条数必须 0）
"""
from __future__ import annotations

import argparse
import json
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402

import cand_card  # noqa: E402  import 即注册（本进程内）
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402

TRAINER = ROOT / "training" / "train_task_card.py"


def selftest(base: str) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, ck = load_base_encoder(base, device)
    core_total = sum(p.numel() for p in enc.parameters())
    core_train = sum(p.numel() for p in enc.parameters() if p.requires_grad)
    dec = RobustARSliceDecoder(hidden_dim=ck["hidden_dim"],
                               num_classes=cand_card.SPEC.num_classes,
                               num_heads=4, num_layers=2)
    head = sum(p.numel() for p in dec.parameters())
    info = {
        "card": "cand_validity", "base": base, "device": str(device),
        "core_total": core_total, "core_trainable": core_train,
        "core_frozen": core_train == 0,
        "head_total": head, "head_trainable": head,
        "trainable_total": core_train + head,
        "max_len": cand_card.SPEC.max_len, "max_steps": cand_card.SPEC.max_steps,
        "num_classes": cand_card.SPEC.num_classes,
    }
    print("SELFTEST_1 " + json.dumps(info, ensure_ascii=False), flush=True)
    if core_train != 0:
        raise SystemExit("SELFTEST_1 失败：核不是冻结的 ⇒ 本跑作废")
    # 静默截断守卫：训练集 + 三个评测集（PREREG_PHASE2 §4）
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    cand_card.check_trunc(cand_card.load_jsonl(cand_card.ARM_FILES[os.environ.get(
        "DTSEEK_CAND_ARM", "P")]), "train", tok)
    for name in ("natural", "adversarial", "ctrl_adversarial"):
        cand_card.check_trunc(cand_card.load_jsonl(cand_card.DATA / f"{name}.jsonl"), name, tok)
    del enc, dec


def main() -> int:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--arm", choices=("P", "C"), default="P")
    ap.add_argument("--shuf", action="store_true", help="随机标签对照臂")
    ap.add_argument("--shuf-seed", type=int, default=None)
    known, rest = ap.parse_known_args()

    os.environ["DTSEEK_CAND_ARM"] = known.arm
    if known.shuf:
        if known.shuf_seed is None:
            raise SystemExit("--shuf 必须给 --shuf-seed（PREREG：seed×1000+7）")
        os.environ["DTSEEK_CAND_SHUF"] = "1"
        os.environ["DTSEEK_CAND_SHUF_SEED"] = str(known.shuf_seed)
    selftest(known.base)

    sys.argv = [str(TRAINER), "--card", "cand_validity",
                "--base", known.base, "--out", known.out, *rest]
    runpy.run_path(str(TRAINER), run_name="__main__")
    return 0


if __name__ == "__main__":
    sys.exit(main())
