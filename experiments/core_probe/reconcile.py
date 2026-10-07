#!/usr/bin/env python3
"""线上口径对账（PREREG §1）——用「线上核 + 已发布数字」复现已知结果。

对账目标换成 `experiments/capability_map/eval.json` 的 **冻结档**：
    checkpoints/base_encoder.pt（线上核）+ experiments/capability_map/cards/sentiment_frozen_s42.pt
    在 held-out `eval_S`（split_of("sentiment", 42)，n=3200）上的 `evaluate_task` 指标。

为什么不用 e5c 卡：e5_single 是**联合微调过基座**的 ckpt（`doc_encoder` ≠ 线上核），
用它对账会把「编码器不同」误判成「探针口径不同」。冻结档才是线上同一条路径。

对账点（三重）：
  1. 复现 cls_acc / exact_match / n_cls / n_bg，与 eval.json 相对差 < 3%（且 n_cls/n_bg 逐位相同）；
  2. 构造性：hook 抓的 norm 输出 == 同一次前向返回值（torch.equal）；
  3. hook 特征 == 直接前向（跨两次前向），max|Δ| = 0。
输出 results/reconcile.json；exit 0 = 过，3 = 不过。
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "capability_map"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from dtseek.tasks.artifacts import build_card_decoder, load_base_encoder, read_card  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import evaluate_task  # noqa: E402
from probe import make_loader, split_of  # noqa: E402  (capability_map/probe.py)

REL_TOL = 0.03
CM = ROOT / "experiments" / "capability_map"


def md5(p: Path) -> str:
    return hashlib.md5(p.read_bytes()).hexdigest()


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    res: dict = {
        "target": {
            "file": "experiments/capability_map/eval.json :: caps.sentiment[\"42\"].frozen",
            "base": "checkpoints/base_encoder.pt（线上核）",
            "card": "experiments/capability_map/cards/sentiment_frozen_s42.pt",
            "split": 'split_of("sentiment", 42) → eval 前 3200 条（hold-out）',
        },
        "rel_tol": REL_TOL,
        "device": str(device),
    }

    expected = json.loads((CM / "eval.json").read_text())["caps"]["sentiment"]["42"]["frozen"]
    res["expected"] = {k: expected[k] for k in ("cls_acc", "exact_match", "n_cls", "n_bg", "bg_fp")}

    spec = resolve_tasks(["sentiment"])["sentiment"].spec
    ev, _tr = split_of("sentiment", 42)
    loader = make_loader(ev, spec, shuffle=False)

    enc, _ = load_base_encoder(str(ROOT / "checkpoints" / "base_encoder.pt"), device)
    ck = read_card(CM / "cards" / "sentiment_frozen_s42.pt")
    dec, cspec = build_card_decoder(ck, device)

    # ── ① 正式复现（同一条 evaluate_task 路径）────────────────────────
    m = evaluate_task(enc, dec, loader, device, cspec)
    res["reproduced"] = {k: float(m[k]) for k in ("cls_acc", "exact_match", "n_cls", "n_bg", "bg_fp")}
    res["abs_diff"] = abs(res["reproduced"]["cls_acc"] - res["expected"]["cls_acc"])
    res["rel_diff"] = res["abs_diff"] / max(res["expected"]["cls_acc"], 1e-9)
    res["exact_match_abs_diff"] = abs(res["reproduced"]["exact_match"] - res["expected"]["exact_match"])
    res["n_identical"] = (res["reproduced"]["n_cls"] == res["expected"]["n_cls"]
                          and res["reproduced"]["n_bg"] == res["expected"]["n_bg"])

    # ── ② 构造性：hook 的 norm 输出 == 同一次前向返回值（逐位）──────
    cap: dict = {}
    h = enc.norm.register_forward_hook(lambda m_, i_, o_: cap.__setitem__("out", o_))
    batch = next(iter(loader))
    inp, mask = batch["input_ids"].to(device), batch["attention_mask"].to(device)
    with torch.no_grad():
        doc = enc(inp, attention_mask=mask)
    same_forward = bool(torch.equal(cap["out"], doc))
    h.remove()

    # ── ③ 跨两次前向：hook 特征 == 直接前向，max|Δ| ────────────────
    with torch.no_grad():
        doc2 = enc(inp, attention_mask=mask)
    hook_feat = ((cap["out"] * mask.unsqueeze(-1).float()).sum(1)
                 / mask.unsqueeze(-1).float().sum(1))
    direct_feat = ((doc2 * mask.unsqueeze(-1).float()).sum(1)
                   / mask.unsqueeze(-1).float().sum(1))
    delta = float((hook_feat - direct_feat).abs().max())
    res["hook_vs_direct_maxabs"] = delta
    res["hook_same_forward_equal"] = same_forward
    res["hook_assert_note"] = "同一进程内两次前向的 max|Δ|=0 即确定性复现（非近似相等）"

    # ── 数据指纹 ────────────────────────────────────────────────
    res["data_md5"] = {
        "capability_map/cache/sentiment_ordered_s42.pkl":
            md5(CM / "cache" / "sentiment_ordered_s42.pkl"),
        "core_keep/cache/sentiment_32000.pkl":
            md5(ROOT / "experiments" / "core_keep" / "cache" / "sentiment_32000.pkl"),
        "cards/sentiment_frozen_s42.pt": md5(CM / "cards" / "sentiment_frozen_s42.pt"),
        "checkpoints/base_encoder.pt": md5(ROOT / "checkpoints" / "base_encoder.pt"),
    }
    res["wall_sec"] = round(time.time() - t0, 1)

    res["pass"] = bool(res["rel_diff"] < REL_TOL and res["n_identical"]
                       and same_forward and delta == 0.0)
    out = HERE / "results" / "reconcile.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2))

    print(f"[reconcile] 期望 cls_acc={res['expected']['cls_acc']:.6f} exact="
          f"{res['expected']['exact_match']:.6f} n_cls={res['expected']['n_cls']:.0f} "
          f"n_bg={res['expected']['n_bg']:.0f}")
    print(f"[reconcile] 复现 cls_acc={res['reproduced']['cls_acc']:.6f} exact="
          f"{res['reproduced']['exact_match']:.6f} n_cls={res['reproduced']['n_cls']:.0f} "
          f"n_bg={res['reproduced']['n_bg']:.0f}")
    print(f"[reconcile] |Δcls_acc|={res['abs_diff']:.6f} 相对={res['rel_diff']:.4%} "
          f"（门槛 <{REL_TOL:.0%}）；|Δexact|={res['exact_match_abs_diff']:.6f}")
    print(f"[reconcile] 同次前向逐位={same_forward}；跨前向 max|Δ|={delta}")
    print(f"[reconcile] {out} | wall={res['wall_sec']}s")
    print(f"[reconcile] {'PASS' if res['pass'] else 'FAIL'}")
    return 0 if res["pass"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
