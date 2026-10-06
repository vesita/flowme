"""bands.py —— 由**对照跑**算出新卡的噪声带，写 `bands.json`（PREREG §4.1）。

只读对照 metrics（`ctrl_*`）与 core_keep 的 J3（negation 对照），**不读链上任何数据**；
`run_all.sh` 在对照全部结束之后、链上开跑之前调用它。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CARDS = HERE / "cards"
SEEDS = (42, 43)

#: PREREG §4.1 写死：4 老卡的噪声带（dev-notes/12 §12.5，与 core_keep 相同）
LEGACY_BAND = {"pronoun": 0.0283, "sentiment": 0.0041,
               "relation": 0.0139, "person": 0.0033}


def exact_of(path: Path, seed: int) -> dict:
    d = json.loads(path.read_text(encoding="utf-8"))
    assert d["seed"] == seed, f"{path} seed 不符"
    return {k: v["exact_match"] for k, v in d["end"].items()}


def main() -> int:
    out: dict = {"generated_at": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                 "method": "2 seed 极差 |exact(s42) − exact(s43)|，来自对照跑（PREREG §4.1）",
                 "bands": dict(LEGACY_BAND), "sources": {}}

    # negation：对照 = core_keep 的 J3（同构单加卡），只读
    j3 = {}
    for s in SEEDS:
        p = ROOT / "experiments" / "core_keep" / "cards" / f"J3_s{s}_metrics.json"
        d = json.loads(p.read_text(encoding="utf-8"))
        j3[s] = d["metrics"]["negation"]["exact_match"]
    out["bands"]["negation"] = abs(j3[42] - j3[43])
    out["sources"]["negation"] = {"ctrl": "core_keep J3（单加卡，未重训）",
                                  "s42": j3[42], "s43": j3[43]}

    # idiom / ownership：本实验对照
    for card in ("idiom", "ownership"):
        ex = {}
        for s in SEEDS:
            p = CARDS / f"ctrl_{card}_s{s}.json"
            if not p.exists():
                raise SystemExit(f"[bands] 缺对照 {p} ⇒ 无法定带")
            ex[s] = exact_of(p, s)[card]
        out["bands"][card] = abs(ex[42] - ex[43])
        out["sources"][card] = {"ctrl": f"control {card}（4 老卡 + {card}，1344 步）",
                                "s42": ex[42], "s43": ex[43]}

    out["bands_pt"] = {k: round(v * 100, 4) for k, v in out["bands"].items()}
    dst = HERE / "bands.json"
    dst.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[bands] 写 {dst}")
    print(json.dumps(out["bands_pt"], ensure_ascii=False) + "  (单位 pt)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
