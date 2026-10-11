#!/usr/bin/env python3
"""S86e · 损失形状探针的**判据表**（口径见 `experiments/meme_anchor/PREREG_SHAPE.md` §2/§4/§5）。

只读探针结果 JSON（`86_result_detr*_s4321.json`），按写死的判据对**基线（loss=none）**逐条判；
不训练、不改判据。★ 判据已按 PREREG_SHAPE §4 的**补丁**修正：P3′ 查**两条路**的字准。

    CUDA_VISIBLE_DEVICES="" .venv/bin/python stages/86e_shape_report.py
"""
from __future__ import annotations

import json
from pathlib import Path

LOGS = Path("/home/vesita/coding/my/flowme/logs")
SEED = 4321
P1_MARGIN, P2_RATIO, P3_MARGIN = 0.02, 0.8, 0.05     # PREREG_SHAPE §2 + §4 补丁


def main() -> None:
    files = sorted(LOGS.glob(f"86_result_detr*_s{SEED}.json"))
    rows = []
    for f in files:
        r = json.loads(f.read_text(encoding="utf-8"))
        m, s = r["metrics"], r["snaps"][-1]
        rows.append(dict(kind=r.get("loss_kind", "align"), lam=r["lam"], steps=r["steps"],
                         auroc=m["pooled"]["auroc_centered"], delta=m["pooled"]["delta"],
                         m8=m["m8_length_matched"]["delta"], slot=m["slot"]["delta"],
                         cka=m["cka_delta"], lin=m["lin_mapped"]["delta"], thr=m["pooled"]["thr_centered"]["thr"],
                         spread_v=s["spread_v"], spread_t=s["spread_t"],
                         acc_v=r["final"]["char_acc_v"], acc_t=r["final"]["char_acc_t"]))
    if not rows:
        raise SystemExit(f"没有探针结果（logs/86_result_detr*_s{SEED}.json）")
    rows.sort(key=lambda x: (x["kind"] != "none", x["lam"]))

    print(f"| 配置 | λ | 步 | 行中心化AUROC | Δ池化 | Δ长度匹配 | 逐槽Δ | CKAΔ | 线性映射Δ | 分化度V | 字准V | 字准T | 阈值(中心) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for x in rows:
        print(f"| {x['kind']} | {x['lam']} | {x['steps']} | {x['auroc']:.4f} | {x['delta']:+.4f} | {x['m8']:+.4f} | "
              f"{x['slot']:+.4f} | {x['cka']:+.3f} | {x['lin']:+.3f} | {x['spread_v']:.3f} | "
              f"{x['acc_v']*100:.1f}% | {x['acc_t']*100:.1f}% | {x['thr']:+.4f} |")

    base = next((x for x in rows if x["kind"] == "none"), None)
    if base is None:
        raise SystemExit("缺基线（loss=none）")
    print(f"\n**基线（none/λ0）**：AUROC={base['auroc']:.4f} 分化度V={base['spread_v']:.3f} "
          f"字准V={base['acc_v']*100:.1f}% 字准T={base['acc_t']*100:.1f}%")
    print("\n| 配置 | P1 AUROC≥基线−0.02 | P2 分化度≥基线×0.8 | **P3′ 两条路字准都不掉** | ⇒ 三全过？ |")
    print("|---|---|---|---|---|")
    for x in rows:
        p1 = x["auroc"] >= base["auroc"] - P1_MARGIN
        p2 = x["spread_v"] >= base["spread_v"] * P2_RATIO
        p3v = x["acc_v"] >= base["acc_v"] - P3_MARGIN
        p3t = x["acc_t"] >= base["acc_t"] - P3_MARGIN
        p3 = p3v and p3t
        print(f"| {x['kind']} λ={x['lam']} | {'✅' if p1 else '❌'} ({x['auroc']:.4f}) | "
              f"{'✅' if p2 else '❌'} ({x['spread_v']:.3f}) | "
              f"{'✅' if p3 else '❌'} (V {x['acc_v']*100:.1f}% / T {x['acc_t']*100:.1f}%) | "
              f"**{'—（基线）' if x['kind']=='none' else ('✅ 值得全量重训' if (p1 and p2 and p3) else '❌')}** |")
    print("\n★ 单 seed / 3000 步 / 权重一次拍定 ⇒ **只判方向**（PREREG_SHAPE §3/§5），不得写成『某配置更强』。")


if __name__ == "__main__":
    main()
