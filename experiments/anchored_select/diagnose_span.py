#!/usr/bin/env python3
"""指针（锚点）为什么没学会 —— 逐样本看首步 start/end 预测分布（诊断，不参与判据）。

    uv run python experiments/anchored_select/diagnose_span.py --ckpt .../anchored_s42.pt

只评**正例**（有真值锚点），输出：
  真值 start/end 分布 vs 预测 start/end 分布；
  start 命中率 / end 命中率 / 端点差的中位数；
  预测区间落在候选字段内的比例（"至少指对了地方"）。
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import torch
from nano_char_tokenizer import NanoCharTokenizer
from torch.utils.data import DataLoader

from dtseek.tasks.artifacts import build_card_decoder, load_base_encoder, read_card
from dtseek.tasks.runtime import GenericTaskDataset

HERE = Path(__file__).resolve().parent


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--split", default="test")
    ap.add_argument("--joint", action="store_true")
    ap.add_argument("--task", default="anchored_sel")
    args = ap.parse_args()
    from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
    from dtseek.tasks.plugin import TaskSpec
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, _ = load_base_encoder(args.base, device)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if args.joint:
        enc.load_state_dict(ck["doc_encoder"], strict=True)
        spec = TaskSpec.from_snapshot(ck["task_specs"][args.task])
        dec = RobustARSliceDecoder(hidden_dim=ck["hidden_dim"], num_classes=spec.num_classes,
                                   **ck["decoder_kwargs"]).to(device)
        dec.load_state_dict(ck["decoders"][args.task], strict=True)
    else:
        dec, spec = build_card_decoder(read_card(args.ckpt), device)
    dec.eval()

    rows = [json.loads(l) for l in open(HERE / "data" / f"{args.split}.jsonl", encoding="utf-8")]
    pos = [r for r in rows if r["label"] == 1]
    items = [{"text": r["text"], "spans": r["spans"]} for r in pos]
    loader = DataLoader(GenericTaskDataset(items, NanoCharTokenizer(), spec),
                        batch_size=128, shuffle=False)

    start_hit = end_hit = both = inside = 0
    pred_s_dist: Counter = Counter()
    start_diffs: list[int] = []
    n = 0
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mem = enc(inp, attention_mask=mask)
            B = inp.shape[0]
            q0 = dec.bos_query.expand(B, 1, -1)
            out = dec.forward_step(q0, mem, doc_mask=mask)
            ps = out["start_logits"].argmax(-1).tolist()
            pe = out["end_logits"].argmax(-1).tolist()
            ts = batch["starts"][:, 0].tolist()
            te = batch["ends"][:, 0].tolist()
            for i in range(B):
                n += 1
                row = pos[n - 1]
                c_start = row["spans"][0]["start"]
                c_end = row["spans"][0]["end"] - 1
                pred_s_dist[ps[i]] += 1
                start_hit += ps[i] == ts[i]
                end_hit += pe[i] == te[i]
                both += (ps[i] == ts[i] and pe[i] == te[i])
                inside += c_start <= ps[i] <= c_end
                start_diffs.append(ps[i] - ts[i])
    start_diffs.sort()
    med = start_diffs[len(start_diffs) // 2] if start_diffs else 0
    top = pred_s_dist.most_common(8)
    true_starts = Counter(r["spans"][0]["start"] for r in pos)
    print(f"[{args.ckpt}] split={args.split} 正例 n={n}")
    print(f"  start 命中={start_hit/n:.4f}  end 命中={end_hit/n:.4f}  双命中={both/n:.4f}")
    print(f"  预测起点落在候选区间内={inside/n:.4f}  起点差中位数={med}")
    print(f"  预测 start top8={top}")
    print(f"  真值 start 分布范围={min(true_starts)}..{max(true_starts)} "
          f"top8={true_starts.most_common(8)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
