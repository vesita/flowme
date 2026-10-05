"""汇总 task-11 方案 A（候选槽位嵌入）8 跑结果：两档 × 两臂 × 两 seed。

    uv run python experiments/selection_cards/summarize_cand_embed.py

输出：P1–P3 判定表、逐 seed 差值、均值，直接贴进报告。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "selection_cards"
BLIND = 0.20                      # 盲猜基线（5 类，含「无合适候选」）
P1, P2, P3 = 0.60, 0.10, 0.95     # PREREG_CAND_EMBED §4
SEEDS = (42, 43)
MODES = ("frozen", "joint")


def load(mode: str, cand: str, seed: int) -> dict:
    tag = f"{mode}_{'cand' if cand == 'on' else 'nocand'}_s{seed}"
    p = RES / f"results_cand_embed_{tag}.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def cell(m: dict) -> str:
    if not m:
        return "—"
    e = m["eval_metrics"]
    return (f"{e['exact_match']:.3f} / {e['bg_fp']:.3f} / {e['span_hit']:.3f} "
            f"| cls {e['cls_acc']:.3f} | anchor {e['anchor_exact']:.3f}")


def verdict(e: dict) -> str:
    return ("P1 " + ("√" if e["exact_match"] >= P1 else "×")
            + " P2 " + ("√" if e["bg_fp"] <= P2 else "×")
            + " P3 " + ("√" if e["span_hit"] >= P3 else "×"))


def main() -> int:
    print(f"# 方案 A 汇总（判据 P1 exact≥{P1} / P2 bg_fp≤{P2} / P3 span_hit≥{P3}，"
          f"盲猜 cls={BLIND}）\n")
    deltas: dict[str, dict[str, float]] = {}
    for mode in MODES:
        print(f"## {mode}\n")
        print("| seed | 臂 | exact | bg_fp | span_hit | cls | anchor(条件) | 判定 |"
              " train_exact | 评测折 Δ(train−eval) | s/step | peak MB |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for seed in SEEDS:
            for arm in ("on", "off"):
                arm_name = "cand" if arm == "on" else "nocand"
                r = load(mode, arm, seed)
                if not r:
                    print(f"| {seed} | {arm_name} | 缺 | | | | | | | | | |")
                    continue
                e, t = r["eval_metrics"], r["train_metrics"]
                print(f"| {seed} | {arm_name} "
                      f"| {e['exact_match']:.3f} | {e['bg_fp']:.3f} | {e['span_hit']:.3f} "
                      f"| {e['cls_acc']:.3f} | {e['anchor_exact']:.3f} "
                      f"| {verdict(e)} | {t['exact_match']:.3f} "
                      f"| {t['exact_match'] - e['exact_match']:+.3f} "
                      f"| {r['cost']['sec_per_step']} | {r['cost']['peak_mem_mb']} |")
        print()
        ds = []
        for seed in SEEDS:
            a, b = load(mode, "on", seed), load(mode, "off", seed)
            if a and b:
                d = a["eval_metrics"]["exact_match"] - b["eval_metrics"]["exact_match"]
                ds.append(d)
                deltas.setdefault(str(seed), {})[mode] = d
                print(f"- seed {seed} Δexact(cand−nocand) = {d:+.4f}；"
                      f"Δbg_fp = {a['eval_metrics']['bg_fp'] - b['eval_metrics']['bg_fp']:+.4f}；"
                      f"Δspan_hit = {a['eval_metrics']['span_hit'] - b['eval_metrics']['span_hit']:+.4f}；"
                      f"Δcls = {a['eval_metrics']['cls_acc'] - b['eval_metrics']['cls_acc']:+.4f}")
        if ds:
            print(f"- **均值 Δexact = {sum(ds) / len(ds):+.4f}**（原值 {', '.join(f'{x:+.4f}' for x in ds)}）\n")

    print("## 总判定\n")
    ok_all, same_sign = True, True
    vals = []
    for mode in MODES:
        for seed in SEEDS:
            a = load(mode, "on", seed)
            if not a:
                ok_all = False
                continue
            e = a["eval_metrics"]
            if not (e["exact_match"] >= P1 and e["bg_fp"] <= P2 and e["span_hit"] >= P3):
                ok_all = False
            d = deltas.get(str(seed), {}).get(mode)
            if d is not None:
                vals.append(d)
                if d <= 0:
                    same_sign = False
    mean = sum(vals) / len(vals) if vals else float("nan")
    print(f"- P1–P3 全过（cand 臂 4/4）：{'是' if ok_all else '否'}")
    print(f"- Δexact 两档两 seed 同号且均值 ≥ +0.10：{'是' if (same_sign and mean >= 0.10) else '否'}"
          f"（均值 {mean:+.4f}，原值 {', '.join(f'{x:+.4f}' for x in vals)}）")
    print(f"- ⇒ 假设成立：{'是' if (ok_all and same_sign and mean >= 0.10) else '否'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
