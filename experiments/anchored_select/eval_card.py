#!/usr/bin/env python3
"""锚定候选打标卡的评测：train / test / adversarial × 逐规则，附基线与对账。

    uv run python experiments/anchored_select/eval_card.py \
        --ckpt experiments/anchored_select/cards/s42.pt \
        --out experiments/anchored_select/results/s42.json

    联合臂产物（cumulative_add 格式）：
    ... --ckpt experiments/anchored_select/cards/joint_s42.pt --joint --task anchored_sel

口径（PREREG §4）：
  bin_acc  = 逐样本「支持/不支持」判对的比例（发射非空 ⇒ 判支持）—— P1/P2 用它
  exact    = 切片集合与真值逐位一致（判对且锚点命中）—— 更严，附报
  同时用 `evaluate_task` 复算 cls_acc / bg_fp / exact_match 做**对账**，不一致直接退出。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
from dtseek.tasks.artifacts import build_card_decoder, load_base_encoder, read_card
from dtseek.tasks.plugin import TaskSpec
from dtseek.tasks.runtime import GenericTaskDataset, _rollout, evaluate_task

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SPLITS = ("train", "test", "adversarial")


def load_split(name: str) -> list[dict]:
    return [json.loads(l) for l in open(DATA / f"{name}.jsonl", encoding="utf-8")]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--splits", default=",".join(SPLITS))
    ap.add_argument("--joint", action="store_true", help="产物是联合臂格式")
    ap.add_argument("--task", default="anchored_sel")
    ap.add_argument("--out", default=None)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, bck = load_base_encoder(args.base, device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.joint:
        enc.load_state_dict(ck["doc_encoder"], strict=True)
        spec = TaskSpec.from_snapshot(ck["task_specs"][args.task])
        dec = RobustARSliceDecoder(hidden_dim=ck["hidden_dim"], num_classes=spec.num_classes,
                                   **ck["decoder_kwargs"]).to(device)
        dec.load_state_dict(ck["decoders"][args.task], strict=True)
        core_trainable = "joint(核已训)"
    else:
        card_ck = read_card(args.ckpt)
        dec, spec = build_card_decoder(card_ck, device)
        core_trainable = "frozen(核冻结)"
    dec.eval()

    n_head = sum(p.numel() for p in dec.parameters())
    print(f"[eval] ckpt={args.ckpt} card={spec.name} head={n_head:,} core={core_trainable} "
          f"device={device}", flush=True)

    report: dict = {"ckpt": args.ckpt, "tag": args.tag, "task": spec.name,
                    "n_head_params": n_head, "core": core_trainable, "splits": {}}

    for split in args.splits.split(","):
        rows = load_split(split)
        items = [{"text": r["text"], "spans": r["spans"]} for r in rows]
        loader = DataLoader(GenericTaskDataset(items, __import__("nano_char_tokenizer").NanoCharTokenizer(),
                                               spec), batch_size=128, shuffle=False)
        n = len(rows)
        n_pos = sum(1 for r in rows if r["label"])
        n_neg = n - n_pos
        bin_ok = exact_ok = pos_span_ok = 0
        per_rule_tot: Counter = Counter()
        per_rule_ok: Counter = Counter()
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
                    support_pred = len(got) > 0
                    pred_dist["支持" if support_pred else "不支持"] += 1
                    ok = support_pred == bool(row["label"])
                    bin_ok += ok
                    per_rule_tot[row["rule"]] += 1
                    per_rule_ok[row["rule"]] += ok
                    same = sorted(got) == sorted(truth)
                    exact_ok += same
                    if row["label"] and got and got[0] == truth[0]:
                        pos_span_ok += 1
        m = evaluate_task(enc, dec, loader, device, spec)
        lex = sum(1 for r in rows if (r["candidate"] in r["context"]) == bool(r["label"])) / n
        acc = bin_ok / n
        res = {
            "n": n, "n_pos": n_pos, "n_neg": n_neg,
            "bin_acc": round(acc, 4),
            "exact": round(exact_ok / n, 4),
            "pos_anchor_hit": round(pos_span_ok / max(1, n_pos), 4),
            "majority_baseline": round(max(n_pos, n_neg) / n, 4),
            "blind_guess": 0.5,
            "lex_substring_rule": round(lex, 4),
            "pred_dist": dict(pred_dist),
            "per_rule": {k: round(per_rule_ok[k] / per_rule_tot[k], 4)
                         for k in sorted(per_rule_tot)},
            "per_rule_n": dict(sorted(per_rule_tot.items())),
            "evaluate_task": {k: round(m[k], 4) for k in
                              ("cls_acc", "bg_fp", "exact_match", "span_hit")},
        }
        # 对账：自己的 bin_acc 与 evaluate_task 的 cls_acc/bg_fp 必须自洽
        recomposed = (m["cls_acc"] * n_pos + (1 - m["bg_fp"]) * n_neg) / n
        res["recomposed_bin_acc"] = round(recomposed, 4)
        res["reconcile_ok"] = abs(recomposed - acc) < 1e-6
        if not res["reconcile_ok"]:
            raise SystemExit(f"对账失败 {split}: 自算 {acc:.6f} vs evaluate_task 复算 {recomposed:.6f}")
        report["splits"][split] = res
        print(f"  [{split:13s}] n={n:5d} bin_acc={acc:.4f} exact={res['exact']:.4f} "
              f"anchor={res['pos_anchor_hit']:.4f} 多数类={res['majority_baseline']:.4f} "
              f"子串规则={lex:.4f} pred={res['pred_dist']} "
              f"rule={res['per_rule']} 对账=OK", flush=True)

    out = Path(args.out) if args.out else HERE / "results" / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
