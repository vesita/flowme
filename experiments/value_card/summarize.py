#!/usr/bin/env python3
"""阶段 1 汇总：主表 + δ + V1/V2/V3/V-bonus 判定（判据全部来自 PREREG §4，跑前写死）。

    uv run python experiments/value_card/summarize.py
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE / "results"

M_TEST, M_ADV, M_GOLD = 40.00, 33.33, 33.33     # PREREG §2（跑前算死）
SHUF_TEST_MAX, SHUF_ADV_MAX = M_TEST + 3.0, M_ADV + 3.0
RHO_MIN = 0.40
PT = 100.0


def load(tag: str) -> dict:
    p = RES / f"{tag}.json"
    if not p.exists():
        raise SystemExit(f"缺结果文件 {p}（训练/评测未跑完）")
    return json.loads(p.read_text(encoding="utf-8"))


def acc(r: dict, split: str) -> float:
    return r["splits"][split]["acc"] * PT


def delta(vals: list[float]) -> float:
    """δ = max(2.0pt, 两 seed 极差)（PREREG §4，逐集合各算）"""
    return max(2.0, abs(vals[0] - vals[1]))


def main() -> None:
    main_s = [load(f"main_s{s}") for s in (42, 43)]
    shuf_s = [load(f"shuf_s{s}") for s in (42, 43)]

    t = [acc(r, "test") for r in main_s]
    a = [acc(r, "adversarial") for r in main_s]
    tr = [acc(r, "train") for r in main_s]
    g = [acc(r, "gold") for r in main_s]
    rho = [r["splits"]["gold"]["spearman_rho"] for r in main_s]
    st = [acc(r, "test") for r in shuf_s]
    sa = [acc(r, "adversarial") for r in shuf_s]

    d_t, d_a, d_g = delta(t), delta(a), delta(g)

    v1 = [x >= M_TEST + d_t for x in t]
    v2 = [x >= M_ADV + d_a for x in a]
    v3 = [g[i] >= M_GOLD + d_g and rho[i] >= RHO_MIN for i in range(2)]
    vb = [st[i] <= SHUF_TEST_MAX and sa[i] <= SHUF_ADV_MAX for i in range(2)]

    if (not v1[0] and not v1[1]) or (not v2[0] and not v2[1]) or (not vb[0] and not vb[1]):
        verdict = "② 不可靠（价值观卡在当前核上不可靠）"
    elif all(v1) and all(v2) and all(v3) and all(vb):
        verdict = "① 可靠"
    else:
        verdict = "③ 证据不足"

    out = {
        "train_acc": [round(x, 2) for x in tr],
        "test_acc": [round(x, 2) for x in t],
        "adv_acc": [round(x, 2) for x in a],
        "gold_acc": [round(x, 2) for x in g],
        "gold_spearman": rho,
        "shuf_test_acc": [round(x, 2) for x in st],
        "shuf_adv_acc": [round(x, 2) for x in sa],
        "majority": {"test": M_TEST, "adversarial": M_ADV, "gold": M_GOLD},
        "delta": {"test": round(d_t, 2), "adversarial": round(d_a, 2),
                  "gold": round(d_g, 2),
                  "raw_range": {"test": round(abs(t[0] - t[1]), 2),
                                "adversarial": round(abs(a[0] - a[1]), 2),
                                "gold": round(abs(g[0] - g[1]), 2)}},
        "thresholds": {"V1": round(M_TEST + d_t, 2), "V2": round(M_ADV + d_a, 2),
                       "V3": {"gold_acc": round(M_GOLD + d_g, 2), "rho": RHO_MIN},
                       "V-bonus": {"test_max": SHUF_TEST_MAX, "adv_max": SHUF_ADV_MAX}},
        "V1_pass": v1, "V2_pass": v2, "V3_pass": v3, "V_bonus_pass": vb,
        "verdict": verdict,
        "recommend_stage2": verdict.startswith("①"),
    }
    (RES / "summary.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if verdict.startswith("②"):
        print("\n**因此阶段 2 的学习率控制不应开做。**")


if __name__ == "__main__":
    main()
