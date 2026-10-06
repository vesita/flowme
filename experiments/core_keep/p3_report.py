"""core_keep 汇总：把 P3（参数量/步时/显存/蒸馏占比）与三项自检的原始输出拼成一张表。

读 `cards/*_metrics.json`（训练脚本落盘）与 `logs/corekeep_*.log`（自检原始行），
输出 `summary.md`。只读，不改任何训练产物。
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CARDS = HERE / "cards"
LOGS = ROOT / "logs"
ARMS = ["F", "J0", "J1", "J2", "J3", "J1c", "J1l4", "J1l16"]
SEEDS = (42, 43)


def log_of(name: str) -> str:
    p = LOGS / f"corekeep_{name}.log"
    return p.read_text(encoding="utf-8", errors="replace") if p.exists() else ""


def grep1(text: str, pat: str) -> str | None:
    m = re.search(pat, text)
    return m.group(0) if m else None


def main() -> int:
    rows = []
    for arm in ARMS:
        for S in SEEDS:
            f = CARDS / f"{arm}_s{S}_metrics.json"
            if not f.exists():
                continue
            d = json.loads(f.read_text(encoding="utf-8"))
            ex = {k: v["exact_match"] for k, v in d["metrics"].items()}
            e = d["extra"]
            rows.append({
                "arm": arm, "seed": S, "exact": ex,
                "trainable": e["params"]["trainable_total"],
                "core_trainable": e["params"]["core_trainable"],
                "sec_step": e["timing"]["sec_per_step_steady"],
                "peak_mb": e["timing"]["peak_mem_mb"],
                "kd_first": (e["kd_stats"] or {}).get("first"),
                "kd_mean": (e["kd_stats"] or {}).get("mean"),
                "kd_share": (e["kd_stats"] or {}).get("share_all"),
                "kd_share_last": (e["kd_stats"] or {}).get("share_last_epoch"),
                "n_steps": e["timing"]["n_steps"],
                "warm": e["warm"], "kd": e["kd"],
            })

    out = ["# core_keep · P3 与自检汇总\n"]
    out.append("| arm | seed | 可训参数 | 核可训 | s/step | 峰值MB | kd_first | kd_mean | kd_share | kd_share(末epoch) | 步数 |")
    out.append("|" + "---|" * 11)
    for r in rows:
        out.append("| {arm} | {seed} | {trainable:,} | {core_trainable:,} | {sec_step:.4f} "
                   "| {peak_mb:.0f} | {kd_first} | {kd_mean} | {kd_share} | {kd_share_last} "
                   "| {n_steps} |".format(
                       kd_first="-" if r["kd_first"] is None else f"{r['kd_first']:.4f}",
                       kd_mean="-" if r["kd_mean"] is None else f"{r['kd_mean']:.4f}",
                       kd_share="-" if r["kd_share"] is None else f"{r['kd_share']*100:.2f}%",
                       kd_share_last="-" if r["kd_share_last"] is None else f"{r['kd_share_last']*100:.2f}%",
                       **{k: r[k] for k in ("arm", "seed", "trainable", "core_trainable",
                                            "sec_step", "peak_mb", "n_steps")}))

    out.append("\n## 三项自检的原始输出（逐档）\n")
    for arm in ARMS:
        for S in SEEDS:
            name = f"{arm}_s{S}"
            if not (CARDS / f"{name}_metrics.json").exists():
                continue
            t = log_of(name)
            out.append(f"### {name}")
            for pat in (r"SELFTEST_1 \{.*", r"SELFTEST_2 [^\n]*", r"SELFTEST_3[^\n]*",
                        r"ALIGN_CHECK [^\n]*"):
                for line in t.splitlines():
                    if re.match(pat, line):
                        out.append(f"    {line}")
            out.append("")
    (HERE / "summary.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out[:40]))
    print(f"[save] {HERE / 'summary.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
