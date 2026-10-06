#!/usr/bin/env python3
"""候选有效性卡的评测：natural / adversarial / ctrl_adversarial (+ 自己 test) × 逐格基线。

    uv run python experiments/gen_data_loop/eval_card.py \
        --ckpt experiments/gen_data_loop/cards/cand_P_s42.pt --tag P_s42 \
        --splits natural,adversarial,ctrl_adversarial,test \
        --out experiments/gen_data_loop/results/eval_P_s42.json

口径（PREREG_PHASE2 §3 / §6.1）：
  acc   = 逐样本「有效/无效」判对的比例（第 0 步 argmax 类别 == 真值类）—— D4 用它
  exact = 类 + 区间逐位一致（无效要求不发射）—— 附报，不过线
  每格并列 majority / SE / **max_naive** / **卡 − max_naive**（低于免费规则必须如实标出）
  与 `evaluate_task` 的 cls_acc / bg_fp 逐数对账，不一致直接退出；
  max_naive 与 `build_data.naive_battery` 独立复算对账，不一致直接退出。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402

import cand_card  # noqa: E402  import 即注册
from dtseek.tasks.artifacts import build_card_decoder, load_base_encoder, read_card  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, _rollout, evaluate_task  # noqa: E402
from build_data import naive_battery  # noqa: E402  独立复算朴素规则（与 stats.json 对账）

STATS_KEY = {
    "natural": ("natural",),
    "adversarial": ("proposal", "adversarial"),
    "ctrl_adversarial": ("control", "ctrl_adversarial"),
    "test": ("proposal", "test"),
    "ctrl_test": ("control", "ctrl_test"),
    "train": ("proposal", "train"),
    "ctrl_train": ("control", "ctrl_train"),
}


def se(p: float, n: int) -> float:
    return math.sqrt(max(p, 1e-12) * max(1 - p, 1e-12) / n)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--splits", default="natural,adversarial,ctrl_adversarial")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    stats = json.loads((HERE / "data" / "stats.json").read_text(encoding="utf-8"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, _bck = load_base_encoder(args.base, device)
    card_ck = read_card(args.ckpt)
    dec, spec = build_card_decoder(card_ck, device)
    dec.eval()
    n_head = sum(p.numel() for p in dec.parameters())
    print(f"[eval] ckpt={args.ckpt} tag={args.tag} card={spec.name} head={n_head:,} "
          f"core=frozen(核冻结) device={device}", flush=True)

    tok = NanoCharTokenizer()
    report = {"ckpt": args.ckpt, "tag": args.tag, "task": spec.name,
              "n_head_params": n_head, "core": "frozen(核冻结)",
              "train_args": card_ck.get("train_args", {}), "splits": {}}

    for split in args.splits.split(","):
        rows = cand_card.load_jsonl(cand_card.DATA / f"{split}.jsonl")
        cand_card.check_trunc(rows, split, tok)
        items = [{"text": r["text"], "spans": r["spans"]} for r in rows]
        loader = DataLoader(GenericTaskDataset(items, tok, spec), batch_size=128,
                            shuffle=False)
        n = len(rows)
        n_pos = sum(1 for r in rows if r["label"])
        n_neg = n - n_pos
        bin_ok = exact_ok = 0
        pred_dist: Counter = Counter()
        with torch.no_grad():
            for bi, batch in enumerate(loader):
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                mem = enc(inp, attention_mask=mask)
                B = inp.shape[0]
                preds = _rollout(dec, mem, mask, B, spec, ndb=None, input_ids=inp)
                for b in range(B):
                    row = rows[bi * 128 + b]
                    truth = [(1, s["start"], s["end"] - 1) for s in row["spans"]]
                    got = preds[b]
                    fired = len(got) > 0
                    pred_dist["有效" if fired else "无效"] += 1
                    bin_ok += int(fired == bool(row["label"]))
                    exact_ok += int(sorted(got) == sorted(truth))
        m = evaluate_task(enc, dec, loader, device, spec)
        acc = bin_ok / n

        # ---- max_naive：stats.json 读数 + build_data.naive_battery 独立复算 ----
        key = STATS_KEY[split]
        node = stats
        for k in key:
            node = node[k]
        naive_stats = node["naive"]["max_naive"]
        naive_re = naive_battery(rows)["max_naive"]
        if naive_stats != naive_re:
            raise SystemExit(f"max_naive 对账失败 {split}: stats={naive_stats} 复算={naive_re}")
        mn = float(naive_re["best_dir"])

        recomposed = (m["cls_acc"] * n_pos + (1 - m["bg_fp"]) * n_neg) / n
        if abs(recomposed - acc) > 1e-6:
            raise SystemExit(f"对账失败 {split}: 自算 {acc:.6f} vs evaluate_task {recomposed:.6f}")

        res = {
            "n": n, "n_pos": n_pos, "n_neg": n_neg,
            "acc": round(acc, 4),
            "exact": round(exact_ok / n, 4),
            "majority_baseline": round(max(n_pos, n_neg) / n, 4),
            "blind_guess": 0.5,
            "se": round(se(acc, n), 4),
            "se_majority": round(se(max(n_pos, n_neg) / n, n), 4),
            "threshold_M_plus_2se_n200": (round(0.5 + 2 * se(0.5, n), 4) if n == 200 else None),
            "max_naive": naive_stats,
            "card_minus_max_naive": round(acc - mn, 4),
            "beats_max_naive": bool(acc > mn),
            "pred_dist": dict(pred_dist),
            "recomposed_bin_acc": round(recomposed, 4),
            "reconcile_ok": True,
            "evaluate_task": {k: round(m[k], 4) for k in
                              ("cls_acc", "bg_fp", "exact_match", "span_hit")},
        }
        report["splits"][split] = res
        flag = "" if res["beats_max_naive"] else "  <== 未超过免费规则"
        print(f"  [{split:16s}] n={n:5d} acc={acc:.4f} exact={res['exact']:.4f} "
              f"多数类={res['majority_baseline']:.4f} SE={res['se']:.4f} "
              f"max_naive={mn:.4f}({naive_re['rule']}) 卡-naive={res['card_minus_max_naive']:+.4f}"
              f" pred={res['pred_dist']} 对账=OK{flag}", flush=True)

    out = Path(args.out) if args.out else HERE / "results" / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
