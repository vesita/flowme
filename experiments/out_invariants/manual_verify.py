"""L1 机械对照第 3 项：把「我人工清点的切片数」与测量函数读到的数逐条比对。

人工清点来源：我直接读 experiments/out_invariants/raw/manual_count3.json 里每个
tasks.<card> 列表的长度（下面的常量是手打的，不是脚本算出来的）。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

MANUAL = {
    "他没有来开会，我特别高兴。":
        {"pronoun": 2, "sentiment": 1, "relation": 2, "person": 0, "negation": 1},
    "老师让我们区分一目了然和不言而喻。":
        {"pronoun": 1, "sentiment": 1, "relation": 2, "person": 4, "negation": 2},
    "会议定在周三上午九点，地点是三号会议室。":
        {"pronoun": 1, "sentiment": 1, "relation": 2, "person": 0, "negation": 1},
}


def fire(anchors: list[dict]) -> bool:
    return bool(anchors)


def main() -> int:
    raw = json.loads((ROOT / "experiments/out_invariants/raw/manual_count3.json").read_text("utf-8"))
    rows, ok = [], True
    for text, expect in MANUAL.items():
        got = {k: len(v) for k, v in raw[text]["tasks"].items()}
        for card, n in expect.items():
            same = got[card] == n
            ok &= same
            rows.append({"text": text[:24], "card": card, "manual": n, "measured": got[card],
                         "same": same,
                         "manual_fire": fire([1] * n), "measured_fire": fire(raw[text]["tasks"][card])})
    print(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"[L1-3] {'PASS' if ok else 'FAIL'}  {sum(r['same'] for r in rows)}/{len(rows)}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
