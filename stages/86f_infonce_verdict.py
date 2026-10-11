#!/usr/bin/env python3
"""S86f · 全量重训的 **Q1–Q4 判据表**（口径见 `experiments/meme_anchor/PREREG_INFONCE.md` §3）。

- 臂 A = `infonce` λ_nce=0.1 × seed{1234,5678}；臂 B = 基线 `none`(λ0) × seed{1234,5678}（复用主档 run）。
- Δ/字准/spread 直接读各 run 的结果 JSON（同一 `measure.py` 口径）；
- ★ **Q4（自由生成）对 4 个 ckpt 用同一份 `gen_eval` 重算**（主档 run 当时还没有该口径，必须统一重算）。

    HSA_OVERRIDE_GFX_VERSION=10.3.0 .venv/bin/python stages/86f_infonce_verdict.py
"""
from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import torch

ROOT = Path("/home/vesita/coding/my/flowme")
sys.path.insert(0, str(ROOT / "experiments" / "meme_anchor"))
_spec = importlib.util.spec_from_file_location("m1driver", ROOT / "stages" / "86_meme_anchor.py")
DRV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DRV)
LOGS = ROOT / "logs"

ARMS = {"B 基线 λ=0": [("none", 0.0, 1234), ("none", 0.0, 5678)],
        "A 候选 infonce λ_nce=0.1": [("infonce", 0.1, 1234), ("infonce", 0.1, 5678)]}
Q2_MARGIN, Q4_MARGIN, Q3_RATIO, Q1_SE = 0.05, 0.05, 0.8, 2.0


def _resolve(prefix: str, kind: str, lam: float, seed: int, ext: str) -> Path:
    """先找带损失标记的文件，再退回不带标记的。

    ★ 基线（λ=0）那两 run 是**主档期**跑的，当时还没有 `--loss` ⇒ 文件名是 `86_result_detr_l0.0_s1234.json`
      （不带 `_none`）；候选（infonce）则带 `_infonce`。两种命名都要能解析到。
    """
    cands = [f"_{kind}", ""] if kind != "align" else [""]
    for tag in cands:
        p = LOGS / f"{prefix}{tag}_l{lam}_s{seed}.{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"找不到 {prefix}_l{lam}_s{seed}.{ext}（试过 tag={cands}）")


def load(kind: str, lam: float, seed: int, device: str) -> dict:
    rj = json.loads(_resolve("86_result_detr", kind, lam, seed, "json").read_text(encoding="utf-8"))
    m, s = rj["metrics"], rj["snaps"][-1]
    blob = torch.load(_resolve("86_ckpt_detr", kind, lam, seed, "pt"), map_location="cpu",
                      weights_only=False)
    model = DRV.MD.build(blob["seed"], blob["vision_arch"])
    model.load_state_dict(blob["state"])
    model.to(device).eval()
    gen = DRV.gen_eval(model, DRV.get_pack()["test_ids"], DRV.get_pack()["test_img"], device)
    del model
    return dict(seed=seed, loss=kind, lam=lam,
                d_pool=m["pooled"]["delta"], d_m8=m["m8_length_matched"]["delta"],
                slot=m["slot"]["delta"], cka=m["cka_delta"], lin=m["lin_mapped"]["delta"],
                auroc=m["pooled"]["auroc_centered"], thr=m["pooled"]["thr_centered"]["thr"],
                acc_t=rj["final"]["char_acc_t"], acc_v=rj["final"]["char_acc_v"],
                spread=s["spread_v"], gen_em_t=gen["gen_em_text"], gen_em_v=gen["gen_em_image"],
                gen_acc_t=gen["gen_char_acc_text"], gen_acc_v=gen["gen_char_acc_image"])


def mean(xs, k):
    return sum(x[k] for x in xs) / len(xs)


def se(xs, k):
    n = len(xs)
    mu = mean(xs, k)
    return math.sqrt(sum((x[k] - mu) ** 2 for x in xs) / max(n - 1, 1)) / math.sqrt(n)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = {name: [load(*cfg, device) for cfg in cfgs] for name, cfgs in ARMS.items()}
    for name, rs in data.items():
        for x in rs:
            print(f"[ARM] {name} seed={x['seed']} Δ池化={x['d_pool']:+.4f} Δ长度匹配={x['d_m8']:+.4f} "
                  f"字准T={x['acc_t']*100:.1f}% 字准V={x['acc_v']*100:.1f}% spread={x['spread']:.3f} "
                  f"生成EM T={x['gen_em_t']*100:.1f}% V={x['gen_em_v']*100:.1f}%", flush=True)

    keys = ["d_pool", "d_m8", "slot", "cka", "lin", "auroc", "thr", "acc_t", "acc_v", "spread",
            "gen_em_t", "gen_em_v", "gen_acc_t", "gen_acc_v"]
    bn, an = list(ARMS)[0], list(ARMS)[1]
    print("\n| 指标 | 基线 B（2 seed 均值） | 候选 A（2 seed 均值） | A−B |")
    print("|---|---|---|---|")
    for k in keys:
        b, a = mean(data[bn], k), mean(data[an], k)
        print(f"| {k} | {b:.4f} | {a:.4f} | {a-b:+.4f} |")

    B, A = data[bn], data[an]
    q1 = []
    for k in ("d_pool", "d_m8"):
        thr = mean(B, k) + Q1_SE * se(A, k)
        q1.append(mean(A, k) >= thr and all(x[k] > mean(B, k) for x in A))
    q2 = [mean(A, k) >= mean(B, k) - Q2_MARGIN for k in ("acc_t", "acc_v")]
    q3 = [mean(A, "spread") >= mean(B, "spread") * Q3_RATIO]
    q4 = [mean(A, k) >= mean(B, k) - Q4_MARGIN for k in ("gen_em_t", "gen_em_v")]
    rows = [("Q1 对齐提升（Δ池化 且 Δ长度匹配，各 ≥B+2SE 且 2 seed 同号）", q1),
            ("Q2 还原不掉（两条路字准）", q2),
            ("Q3 分化度不掉（≥B×0.8）", q3),
            ("Q4 自由生成不掉（贪心 EM 两条路）", q4)]
    print("\n| 判据 | 过？ | 明细 |")
    print("|---|---|---|")
    detail = {
        0: f"Δ池化 A={mean(A,'d_pool'):+.4f} vs 门槛 {mean(B,'d_pool')+Q1_SE*se(A,'d_pool'):+.4f}；"
           f"Δ长度匹配 A={mean(A,'d_m8'):+.4f} vs {mean(B,'d_m8')+Q1_SE*se(A,'d_m8'):+.4f}",
        1: f"字准T A={mean(A,'acc_t')*100:.1f}% B={mean(B,'acc_t')*100:.1f}%；"
           f"字准V A={mean(A,'acc_v')*100:.1f}% B={mean(B,'acc_v')*100:.1f}%",
        2: f"spread A={mean(A,'spread'):.3f} B={mean(B,'spread'):.3f}（门槛 {mean(B,'spread')*Q3_RATIO:.3f}）",
        3: f"生成EM T A={mean(A,'gen_em_t')*100:.1f}% B={mean(B,'gen_em_t')*100:.1f}%；"
           f"V A={mean(A,'gen_em_v')*100:.1f}% B={mean(B,'gen_em_v')*100:.1f}%"}
    for i, (name, ok_list) in enumerate(rows):
        print(f"| {name} | {'✅' if all(ok_list) else '❌'} | {detail[i]} |")
    allok = all(all(o) for _, o in rows)
    print(f"\n[INFONCE-VERDICT] {'✅ Q1∧Q2∧Q3∧Q4 全过 ⇒ 该配置可作为「模因 v0」的 IO 卡候选（下一步才谈锚定）' if allok else '❌ 未全过 ⇒ 不下结论，按上表指出卡在哪一条'}")
    (LOGS / "86_infonce_verdict.json").write_text(json.dumps(
        dict(arms={k: v for k, v in data.items()}, q={n: all(o) for n, o in rows}), ensure_ascii=False,
        indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
