#!/usr/bin/env python3
"""P0 完备免费规则电池（free_rule_floor）。

口径（PREREG §3，跑前写死）：
· 旧 8 条**逐字复用** `two_channel_head/build_gen_data.py::naive_skeleton`
  （majority / len_bucket / first_char / last_char / punct_pattern /
   fw_decision_list / tree_depth2 / tree_depth4），fit=train、eval=split；
  必须逐位复现 stats.json(.5329/.5208) 与 stats_adv.json(0.0/.182)，否则口径未对齐。
· 新 3 条（计入 max_naive）：n_slots 查表 / (n_slots,标签序列) 查表 / 袋项类别多重集查表。
  查表口径对齐 `bag_modules/analyze.py::rule_baseline`：**无 min_support**，
  未见键回退 train 全局多数；报 seen_rate。
· 单列披露不计入 max_naive：袋项**字面文本**多重集（"标注函数的逆"一类，跑前写死）。

用法：uv run python experiments/free_rule_floor/rules.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent
TCH_DIR = ROOT / "experiments" / "two_channel_head"
SS_DIR = ROOT / "experiments" / "struct_supervision"
RESULTS = HERE / "results"
SPLITS = ("test", "adv1", "adv2")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


bgd = _load(TCH_DIR / "build_gen_data.py", "frf_bgd_ro")   # 只读：naive_skeleton 等

OLD8 = ("majority", "len_bucket", "first_char", "last_char", "punct_pattern",
        "fw_decision_list", "tree_depth2", "tree_depth4")
NEW3 = ("n_slots", "n_slots_labels", "bag_label_multiset")
DISCLOSED = ("bag_text_multiset",)


def load_rows(split: str) -> list[dict]:
    if split in ("train", "test"):
        p = TCH_DIR / "data" / f"{split}.jsonl"
    else:
        p = SS_DIR / "data" / f"{split}.jsonl"
    if not p.exists():
        raise SystemExit(f"缺数据：{p}")
    with open(p, encoding="utf-8") as fp:
        return [json.loads(x) for x in fp]


def label_list(split: str, rows: list[dict]) -> list[list[dict]]:
    obj = json.loads((HERE / "data" / f"labels_{split}.json").read_text(encoding="utf-8"))
    assert obj["fp"] == [fp12(r) for r in rows], f"{split} 标签与数据行不对齐"
    assert obj["skel_id"] == [r["skel_id"] for r in rows], f"{split} skel_id 不对齐"
    return obj["labels"]


def fp12(r: dict) -> str:
    import hashlib
    return hashlib.md5((r["sent"] + str(r["bag_span"])).encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 新查表规则（口径 = bag_modules/analyze.py::rule_baseline：无 min_support）
# ---------------------------------------------------------------------------
def label_seq(rows: list[dict], labs: list[list[dict]]) -> list[tuple]:
    """(n_slots, 按 span 句序的 (t,r,c) 序列) 键。"""
    out = []
    for r, lab in zip(rows, labs):
        order = sorted(range(r["n_slots"]), key=lambda k: r["bag_span"][k][0])
        out.append((r["n_slots"],
                    tuple((lab[k]["t"], lab[k]["r"], lab[k]["c"]) for k in order)))
    return out


def bag_label_multi(rows: list[dict], labs: list[list[dict]]) -> list[tuple]:
    """袋项**类别**多重集：每袋项 (t,r,c) 三元组，排序去序。"""
    return [tuple(sorted((lab[k]["t"], lab[k]["r"], lab[k]["c"])
                         for k in range(r["n_slots"])))
            for r, lab in zip(rows, labs)]


def bag_text_multi(rows: list[dict]) -> list[tuple]:
    """袋项**字面**多重集（单列披露，不计入 max_naive）。"""
    return [tuple(sorted(r["bag"])) for r in rows]


def lookup(fit_keys, fit_y, eval_keys) -> tuple[list[int], float]:
    """按键取 train 多数类；未见键回退 train 全局多数。返回 (预测, seen_rate)。"""
    glob = Counter(fit_y).most_common(1)[0][0]
    tab: dict = defaultdict(Counter)
    for k, y in zip(fit_keys, fit_y):
        tab[k][y] += 1
    maj = {k: c.most_common(1)[0][0] for k, c in tab.items()}
    seen = sum(1 for k in eval_keys if k in maj)
    return [maj.get(k, glob) for k in eval_keys], seen / max(1, len(eval_keys))


def battery() -> dict:
    rows_tr = load_rows("train")
    y_tr = [r["skel_id"] for r in rows_tr]
    glob = Counter(y_tr).most_common(1)[0][0]
    labs_tr = label_list("train", rows_tr)

    fit_keys = {
        "n_slots": [r["n_slots"] for r in rows_tr],
        "n_slots_labels": label_seq(rows_tr, labs_tr),
        "bag_label_multiset": bag_label_multi(rows_tr, labs_tr),
        "bag_text_multiset": bag_text_multi(rows_tr),
    }
    out: dict = {"fit": "train", "n_fit": len(rows_tr), "splits": {}}
    for split in SPLITS:
        rows = load_rows(split)
        y = [r["skel_id"] for r in rows]
        labs = label_list(split, rows)
        rec: dict = {}

        # ---- 旧 8 条（逐字复用 tch 口径）----
        old = bgd.naive_skeleton(rows_tr, rows)
        for k in OLD8:
            rec[k] = round(old[k], 4)
        rec["max_naive_old8"] = round(old["max_naive"], 4)

        # ---- 新 3 条 + 披露条 ----
        seen = {}
        for name, feat in (("n_slots", lambda r: r["n_slots"]),
                           ("n_slots_labels", None),
                           ("bag_label_multiset", None),
                           ("bag_text_multiset", None)):
            if name == "n_slots":
                keys_e = [feat(r) for r in rows]
            elif name == "n_slots_labels":
                keys_e = label_seq(rows, labs)
            elif name == "bag_label_multiset":
                keys_e = bag_label_multi(rows, labs)
            else:
                keys_e = bag_text_multi(rows)
            pred, sr = lookup(fit_keys[name], y_tr, keys_e)
            acc = round(sum(1 for a, b in zip(pred, y) if a == b) / len(y), 4)
            seen[name] = round(sr, 4)
            rec[name] = acc
        rec["seen_rate"] = seen

        rec["max_naive_new"] = round(max(rec[k] for k in OLD8 + NEW3), 4)
        rec["delta_new_vs_old8"] = round(rec["max_naive_new"] - rec["max_naive_old8"], 4)
        rec["n"] = len(y)
        rec["majority"] = round(sum(1 for v in y if v == glob) / len(y), 4)
        out["splits"][split] = rec
    return out


# ---------------------------------------------------------------------------
# 口径对齐（F5 前置门）：旧 8 条必须逐位复现既有报告值
# ---------------------------------------------------------------------------
def align_check(bat: dict) -> dict:
    stats_tch = json.loads((TCH_DIR / "data" / "stats.json").read_text(encoding="utf-8"))
    stats_adv = json.loads((SS_DIR / "data" / "stats_adv.json").read_text(encoding="utf-8"))
    ref = {
        "test": stats_tch["naive_skeleton_test"],
        "adv1": stats_adv["naive"]["adv1_skel"],
        "adv2": stats_adv["naive"]["adv2_skel"],
    }
    rep, ok = {}, True
    for split, r in ref.items():
        mine = bat["splits"][split]
        d = {k: {"mine": mine[k], "ref": r[k]} for k in OLD8
             if abs(mine[k] - r[k]) > 1e-9}
        # train 口径（旧 test 地板 .5329 的来源）
        rep[split] = {"n_diff": len(d), "diff": d, "max_naive_old8": mine["max_naive_old8"],
                      "ref_max_naive": r["max_naive"]}
        ok = ok and not d
    rows_tr = load_rows("train")
    tr = bgd.naive_skeleton(rows_tr, rows_tr)
    rep["train"] = {"mine": round(tr["max_naive"], 4),
                    "ref": stats_tch["max_naive_train"],
                    "n_diff": 0 if round(tr["max_naive"], 4)
                              == stats_tch["max_naive_train"] else 1}
    ok = ok and rep["train"]["n_diff"] == 0
    rep["aligned"] = bool(ok)
    return rep


def main() -> None:
    bat = battery()
    al = align_check(bat)
    bat["align_old_battery"] = al
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "battery.json").write_text(
        json.dumps(bat, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(bat, ensure_ascii=False, indent=2))
    print(f"\n[align] 旧 8 条逐位复现既有报告值 = {al['aligned']}", flush=True)
    if not al["aligned"]:
        raise SystemExit("口径未对齐：先修探针（PREREG §3）")
    print("[done] → results/battery.json", flush=True)


if __name__ == "__main__":
    main()
