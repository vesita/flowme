"""**探索性（跑后追加，不参与 P1 判定）**：分母塌缩后，看看探针还能不能排序别的量。

PREREG §7 的 P1 判据是「探针 vs 达成率」，本脚本算的四个量**都不是**它 ——
只作诊断，报告里必须标成 post-hoc。
"""
import json
from pathlib import Path
from scipy.stats import spearmanr

HERE = Path(__file__).parent
s = json.loads((HERE / "summary.json").read_text(encoding="utf-8"))
CAPS = ["pronoun", "sentiment", "relation", "person", "negation"]
SEEDS = (42, 43)

def rho(xs, ys):
    if len(xs) < 2:
        return None
    v = spearmanr(xs, ys).statistic
    return None if v != v else float(v)

targets = {
    "bypass_cls": lambda r: r["bypass"],
    "frozen_cls": lambda r: r["frozen"],
    "delta_cls(bypass-frozen)": lambda r: (r["bypass"] - r["frozen"]) if (r["bypass"] is not None and r["frozen"] is not None) else None,
    "joint_cls": lambda r: r["joint"],
    "probe_exact": lambda r: r["probe_exact"],
    "frozen_exact": lambda r: r["frozen_exact"],
}
out = {}
print("| seed | 探针对比目标 | n | Spearman rho |")
print("|---|---|---|---|")
for S in SEEDS:
    for name, fn in targets.items():
        xs, ys, kept = [], [], []
        for c in CAPS:
            r = s["rows"][c][str(S)]
            p = r["probe"] if "exact" not in name else r["probe_exact"]
            y = fn(r)
            if p is None or y is None:
                continue
            xs.append(p); ys.append(y); kept.append(c)
        v = rho(xs, ys)
        out[f"s{S}/{name}"] = {"rho": v, "n": len(xs), "kept": kept}
        print(f"| {S} | {name} | {len(xs)} | {v} |")

# 分母塌缩的直接证据
print("\n分母（joint - frozen）:")
for S in SEEDS:
    for c in CAPS:
        r = s["rows"][c][str(S)]
        print(f"  {c:10s} s{S}: denom={r['denom']:+.4f} probe={r['probe']:.4f} "
              f"frozen={r['frozen']:.4f} bypass={r['bypass']:.4f} joint={r['joint']:.4f} "
              f"delta={r['bypass']-r['frozen']:+.4f}")
(HERE / "exploratory.json").write_text(json.dumps(out, ensure_ascii=False, indent=2))
print(f"\n[save] {HERE/'exploratory.json'}")
