#!/usr/bin/env python3
"""skeleton_leak 泄露量化（Q0/Q3/Q4 的数据侧）：不评模型，只算免费规则与分布统计。

对 {a_bal, a_lit, b_pairs, c_pairs, test, adv2} 报：
  · `n_slots→train多数骨架` 免费规则 acc + SE 门槛（多数类 + 2SE）
  · 每桶骨架分布熵 H(skel|n)（nats / 归一化）
  · MI(n_slots; skeleton)（nats）+ NMI（几何/算术）
  · 完备朴素电池（fit=train）：majority / len_bucket / first_char / last_char /
    punct_pattern / fw_decision_list / tree_depth2 / tree_depth4 / **n_slots** → max_naive
  · 规模与语言构成（n、文件清单、汉字占比口径）

用法：uv run python experiments/skeleton_leak/analyze_leak.py
"""
from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "two_channel_head"))

from build_gen_data import naive_skeleton, naive_assign  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TCH_DATA = ROOT / "experiments" / "two_channel_head" / "data"
SS_DATA = ROOT / "experiments" / "struct_supervision" / "data"


def load(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def H(c: Counter) -> float:
    n = sum(c.values())
    return -sum((v / n) * math.log(v / n) for v in c.values() if v)


def leak_stats(rows: list[dict], maj_tab: dict, glob: int) -> dict:
    n = len(rows)
    mk = Counter(r["skel_id"] for r in rows)
    ms = Counter(r["n_slots"] for r in rows)
    by = defaultdict(Counter)
    for r in rows:
        by[r["n_slots"]][r["skel_id"]] += 1
    joint = Counter((r["n_slots"], r["skel_id"]) for r in rows)
    mi = 0.0
    for (s, k), v in joint.items():
        p = v / n
        mi += p * math.log(p / ((ms[s] / n) * (mk[k] / n)))
    hs, hk = H(ms), H(mk)
    hg = sum(ms[s] / n * H(c) for s, c in by.items())
    ok = sum(1 for r in rows if maj_tab.get(r["n_slots"], glob) == r["skel_id"])
    se = math.sqrt(0.25 / n)
    maj_rate = mk.most_common(1)[0][1] / n
    return {
        "n": n,
        "n_slots_rule": round(ok / n, 4),
        "n_slots_rule_se": round(se, 4),
        "majority_own": round(maj_rate, 4),
        "majority_own_class": int(mk.most_common(1)[0][0]),
        "majority_train_class_rate": round(mk[glob] / n, 4),
        "blind_1_40": 0.025,
        "Q0_threshold": round(maj_rate + 2 * se, 4),
        "Q0_pass": bool(ok / n <= maj_rate + 2 * se),
        "H_n_slots": round(hs, 4), "H_skel": round(hk, 4),
        "H_skel_given_nslots": round(hg, 4),
        "MI_nats": round(mi, 4),
        "NMI_geom": round(mi / math.sqrt(hs * hk), 4) if hs and hk else 0.0,
        "NMI_arith": round(2 * mi / (hs + hk), 4) if hs + hk else 0.0,
        "per_bucket": {str(s): {"n": sum(c.values()), "k": len(c),
                                "H_nats": round(H(c), 4),
                                "H_norm": round(H(c) / math.log(len(c)), 4)
                                if len(c) > 1 else 1.0,
                                "top": [(int(a), b) for a, b in c.most_common(4)]}
                       for s, c in sorted(by.items())},
        "label_dist": dict(sorted(mk.items())),
    }


def main() -> None:
    train = load(TCH_DATA / "train.jsonl")
    tab: dict[int, Counter] = defaultdict(Counter)
    for r in train:
        tab[r["n_slots"]][r["skel_id"]] += 1
    maj_tab = {k: c.most_common(1)[0][0] for k, c in tab.items()}
    glob = Counter(r["skel_id"] for r in train).most_common(1)[0][0]
    print(f"[fit=train] n_slots→多数骨架 表 = {maj_tab}；全局多数 = #{glob}", flush=True)

    splits = {"test": load(TCH_DATA / "test.jsonl"),
              "adv2": load(SS_DATA / "adv2.jsonl"),
              "a_bal": load(DATA / "a_bal.jsonl"),
              "a_lit": load(DATA / "a_lit.jsonl"),
              "c_pairs": load(DATA / "c_pairs.jsonl"),
              "b_pairs": load(DATA / "b_pairs.jsonl")}

    out: dict = {"fit": {"n_slots_rule_table": {str(k): v for k, v in maj_tab.items()},
                         "train_majority": glob}}
    for name, rows in splits.items():
        ls = leak_stats(rows, maj_tab, glob)
        nb = naive_skeleton(train, rows)
        na = naive_assign(train, rows)
        nb = {k: round(v, 4) for k, v in nb.items()}
        na = {k: round(v, 4) for k, v in na.items()}
        complete = dict(nb)
        complete["n_slots_rule"] = ls["n_slots_rule"]
        complete["majority_own"] = ls["majority_own"]
        mx = max(complete.values())
        ls.update({"naive_battery": nb, "naive_assign": na,
                   "max_naive_complete": round(mx, 4),
                   "max_naive_complete_rule": max(complete, key=complete.get),
                   "naive_battery_max": nb["max_naive"]})
        out[name] = ls
        print(f"\n== {name} n={ls['n']} ==", flush=True)
        print(f"  n_slots规则 acc={ls['n_slots_rule']:.4f}  多数类(本集)={ls['majority_own']:.4f}"
              f"(#{ls['majority_own_class']})  阈=多数类+2SE={ls['Q0_threshold']:.4f}"
              f"  Q0={'过' if ls['Q0_pass'] else '不过'}", flush=True)
        print(f"  H(n_slots)={ls['H_n_slots']} H(skel)={ls['H_skel']} "
              f"H(skel|n)={ls['H_skel_given_nslots']} MI={ls['MI_nats']} "
              f"NMIg={ls['NMI_geom']} NMIa={ls['NMI_arith']}", flush=True)
        print(f"  每桶: " + " | ".join(
            f"n={s}:n={v['n']} k={v['k']} H={v['H_nats']} Hn={v['H_norm']} top={v['top']}"
            for s, v in ls["per_bucket"].items()), flush=True)
        print(f"  完备电池={complete} → max_naive={mx:.4f} ({ls['max_naive_complete_rule']})",
              flush=True)
        print(f"  指派电池={na}", flush=True)

    # 语言构成（新集）
    b = json.loads((DATA / "stats_build.json").read_text(encoding="utf-8"))
    out["lang"] = {"file_cjk": b["scan"]["file_cjk"], "files_kept": b["scan"]["files_kept"],
                   "sets_files": {k: v["files"] for k, v in b["sets"].items()}}
    (HERE / "results" / "leak_stats.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n[done] → {HERE / 'results' / 'leak_stats.json'}", flush=True)


if __name__ == "__main__":
    main()
