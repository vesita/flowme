#!/usr/bin/env python3
"""P8 免费规则电池：**既有完备电池 + 句式条件化查表**（N2/N6 关键对照）。PREREG §4。

fit 一律 = train；查表口径 = `free_rule_floor/rules.py::lookup`（无 min_support、
未见键回退 train 全局多数、报 seen_rate）。

  R1 = `(g(sent), n_slots) → train 多数骨架`   ← N2 主对照（完全免费：输入侧可观测）
  R2 = `g(sent) → train 多数骨架`
  R3 = `(A(gold), n_slots)`（oracle，单列、不计入 max_naive）
  R4 = `S^B_{g}` 内 train 多数（映射 B 版，单列）
  max_naive_complete = max(旧8 ∪ {n_slots, majority_own})（= skeleton_leak 同口径）
  max_naive_plus = max(max_naive_complete ∪ {R1, R2})

旧 8 条**逐条复用** `build_gen_data` 的私有谓词（`_fit_predict/_tree_depth/feats/...`）
重取**逐行预测**，并断言其总体 acc 与 `naive_skeleton()` **逐位相等**（口径 fail-closed）。

用法：uv run python experiments/mode_conditioned_skel/rules_mode.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

import build_gen_data as bgd  # noqa: E402  只读复用
import modes as MM  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
TCH = ROOT / "experiments" / "two_channel_head" / "data"
SS = ROOT / "experiments" / "struct_supervision" / "data"
SLEAK = ROOT / "experiments" / "skeleton_leak" / "data"
SPLITS = {"test": TCH / "test.jsonl", "adv2": SS / "adv2.jsonl",
          "a_bal": SLEAK / "a_bal.jsonl"}
L_GROUP = (35, 0, 1, 2)
OLD8 = ("majority", "len_bucket", "first_char", "last_char", "punct_pattern",
        "fw_decision_list", "tree_depth2", "tree_depth4")


def load(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def lookup(fit_keys, fit_y, eval_keys):
    """按键取 train 多数类；未见键回退 train 全局多数（= free_rule_floor 口径）。"""
    glob = Counter(fit_y).most_common(1)[0][0]
    tab: dict = defaultdict(Counter)
    for k, y in zip(fit_keys, fit_y):
        tab[k][y] += 1
    maj = {k: c.most_common(1)[0][0] for k, c in tab.items()}
    seen = sum(1 for k in eval_keys if k in maj)
    return [maj.get(k, glob) for k in eval_keys], seen / max(1, len(eval_keys))


def old8_preds(fit_rows: list[dict], eval_rows: list[dict]) -> dict[str, list[int]]:
    """旧 8 条的**逐行**预测（谓词逐字复用 build_gen_data）。"""
    glob = bgd._majority([r["skel_id"] for r in fit_rows])
    out: dict[str, list[int]] = {}
    out["majority"] = [glob] * len(eval_rows)
    out["len_bucket"] = bgd._fit_predict(lambda r: len(r["sent"]) // 6, fit_rows, eval_rows)
    out["first_char"] = bgd._fit_predict(lambda r: r["sent"][0], fit_rows, eval_rows)
    out["last_char"] = bgd._fit_predict(lambda r: r["sent"][-1], fit_rows, eval_rows)
    out["punct_pattern"] = bgd._fit_predict(
        lambda r: "".join(sorted(set(r["sent"]) & set(bgd.PUNCTS))), fit_rows, eval_rows)
    cnt = Counter(w for r in fit_rows for w in bgd.FUNCTION_WORDS if w in r["sent"])
    order_w = sorted(bgd.FUNCTION_WORDS, key=lambda w: -cnt.get(w, 0))
    best: dict = {}
    for w in order_w:
        sub = [r["skel_id"] for r in fit_rows if w in r["sent"]]
        if len(sub) >= 30:
            best[w] = bgd._majority(sub)
    preds = []
    for r in eval_rows:
        p = glob
        for w in order_w:
            if w in r["sent"] and w in best:
                p = best[w]
                break
        preds.append(p)
    out["fw_decision_list"] = preds
    Xf = [bgd.feats(r) for r in fit_rows]
    yf = [r["skel_id"] for r in fit_rows]
    out["tree_depth2"] = bgd._tree_depth(Xf, yf, [bgd.feats(r) for r in eval_rows], 2)
    out["tree_depth4"] = bgd._tree_depth(Xf, yf, [bgd.feats(r) for r in eval_rows], 4)
    return out


def acc(pred, y) -> float:
    return sum(1 for x, z in zip(pred, y) if x == z) / len(y)


def main() -> None:
    train = load(TCH / "train.jsonl")
    ytr = [r["skel_id"] for r in train]
    tab = MM.skeleton_table()
    mapj = json.loads((RESULTS / "mapping.json").read_text(encoding="utf-8"))
    sB = mapj["subsets_B"]
    glob = Counter(ytr).most_common(1)[0][0]

    gr_tr = [MM.MASK_ALIAS[MM.mode_of_sent(r["sent"])] for r in train]
    gg_tr = [MM.MASK_ALIAS[MM.mode_of_skel(tab[r["skel_id"]])] for r in train]
    k_r1 = list(zip(gr_tr, [r["n_slots"] for r in train]))
    k_r3 = list(zip(gg_tr, [r["n_slots"] for r in train]))
    cntB: dict = defaultdict(Counter)
    for m, sid in zip(gr_tr, ytr):
        if sid in sB.get(m, []):
            cntB[m][sid] += 1
    r4_tab = {m: (c.most_common(1)[0][0] if c else glob) for m, c in cntB.items()}
    print(f"[fit] train n={len(train)} 全局多数=#{glob} R4 表={r4_tab}", flush=True)

    out: dict = {"fit": "train", "n_fit": len(train), "train_majority": glob,
                 "fit_tables": {"R4": {k: int(v) for k, v in r4_tab.items()}},
                 "splits": {}}
    for name, p in SPLITS.items():
        rows = load(p)
        y = [r["skel_id"] for r in rows]
        gr = [MM.MASK_ALIAS[MM.mode_of_sent(r["sent"])] for r in rows]
        gg = [MM.MASK_ALIAS[MM.mode_of_skel(tab[r["skel_id"]])] for r in rows]
        nb_acc = {k: round(v, 4) for k, v in bgd.naive_skeleton(train, rows).items()}
        preds = old8_preds(train, rows)
        # ---- 口径 fail-closed：逐行预测的 acc 必须等于 naive_skeleton() ----
        for k in OLD8:
            mine = round(acc(preds[k], y), 4)
            assert abs(mine - nb_acc[k]) < 1e-9, f"{name}/{k}: {mine} != {nb_acc[k]}"

        ns_keys_tr = [r["n_slots"] for r in train]
        ns_keys = [r["n_slots"] for r in rows]
        n_slots_pred, sr_ns = lookup(ns_keys_tr, ytr, ns_keys)
        r1, sr1 = lookup(k_r1, ytr, list(zip(gr, ns_keys)))
        r2, sr2 = lookup(gr_tr, ytr, gr)
        r3, sr3 = lookup(k_r3, ytr, list(zip(gg, ns_keys)))
        r4 = [r4_tab.get(m, glob) for m in gr]
        preds.update({"n_slots": n_slots_pred, "R1_mode_nslots": r1, "R2_mode": r2,
                      "R3_oracle_mode_nslots": r3, "R4_mode_Bsubset": r4})

        rec = {k: round(acc(preds[k], y), 4) for k in OLD8}
        rec["max_naive_old8"] = round(max(rec[k] for k in OLD8), 4)
        rec["n_slots"] = round(acc(n_slots_pred, y), 4)
        rec["majority_own"] = round(max(Counter(y).values()) / len(y), 4)
        rec["R1_mode_nslots"] = round(acc(r1, y), 4)
        rec["R2_mode"] = round(acc(r2, y), 4)
        rec["R3_oracle_mode_nslots"] = round(acc(r3, y), 4)
        rec["R4_mode_Bsubset"] = round(acc(r4, y), 4)
        rec["seen_rate"] = {"n_slots": round(sr_ns, 4), "R1": round(sr1, 4),
                            "R2": round(sr2, 4), "R3": round(sr3, 4)}
        rec["max_naive_complete"] = round(max(list(rec[k] for k in OLD8)
                                              + [rec["n_slots"], rec["majority_own"]]), 4)
        rec["max_naive_plus"] = round(max(rec["max_naive_complete"],
                                          rec["R1_mode_nslots"], rec["R2_mode"]), 4)
        rec["max_naive_plus4"] = round(max(rec["max_naive_plus"], rec["R4_mode_Bsubset"]), 4)
        rec["n"] = len(y)
        rec["SE"] = round(math.sqrt(0.25 / len(y)), 4)

        # ---- 单列披露（跑后追加，**不改** PREREG 判据）：更强的免费组合查表 ----
        disc = {}
        keys_d = {
            "n_slots+last_char": [(r["n_slots"], r["sent"][-1]) for r in rows],
            "mode+last_char": [(m, r["sent"][-1]) for m, r in zip(gr, rows)],
            "mode+n_slots+last_char": [(m, r["n_slots"], r["sent"][-1])
                                       for m, r in zip(gr, rows)],
            "mode+len_bucket": [(m, len(r["sent"]) // 6) for m, r in zip(gr, rows)],
            "n_slots+first_char": [(r["n_slots"], r["sent"][0]) for r in rows],
            "n_slots+len_bucket": [(r["n_slots"], len(r["sent"]) // 6) for r in rows],
        }
        keys_d_tr = {
            "n_slots+last_char": [(r["n_slots"], r["sent"][-1]) for r in train],
            "mode+last_char": [(m, r["sent"][-1]) for m, r in zip(gr_tr, train)],
            "mode+n_slots+last_char": [(m, r["n_slots"], r["sent"][-1])
                                       for m, r in zip(gr_tr, train)],
            "mode+len_bucket": [(m, len(r["sent"]) // 6) for m, r in zip(gr_tr, train)],
            "n_slots+first_char": [(r["n_slots"], r["sent"][0]) for r in train],
            "n_slots+len_bucket": [(r["n_slots"], len(r["sent"]) // 6) for r in train],
        }
        for nm in keys_d:
            pr_, sr_ = lookup(keys_d_tr[nm], ytr, keys_d[nm])
            preds[f"DISC_{nm}"] = pr_
            disc[nm] = round(acc(pr_, y), 4)
        rec["disclosure_combo_lookup"] = disc
        rec["max_naive_disclosure"] = round(max(rec["max_naive_plus"], *disc.values()), 4)
        print(f"  披露·组合查表 = {disc} → max(含披露)={rec['max_naive_disclosure']}",
              flush=True)

        grp = {}
        for gname, keep in (("L", [i for i, r in enumerate(rows) if r["skel_id"] in L_GROUP]),
                            ("nonL", [i for i, r in enumerate(rows)
                                      if r["skel_id"] not in L_GROUP])):
            if not keep:
                grp[gname] = {"n": 0}
                continue
            yy = [y[i] for i in keep]
            sub = {k: round(acc([v[i] for i in keep], yy), 4) for k, v in preds.items()}
            sub["majority_own"] = round(max(Counter(yy).values()) / len(yy), 4)
            sub["max_naive_complete"] = round(max(list(sub[k] for k in OLD8)
                                                  + [sub["n_slots"], sub["majority_own"]]), 4)
            sub["max_naive_plus"] = round(max(sub["max_naive_complete"],
                                              sub["R1_mode_nslots"], sub["R2_mode"]), 4)
            sub["max_naive_disclosure"] = round(
                max([sub["max_naive_plus"]]
                    + [v for k, v in sub.items() if k.startswith("DISC_")]), 4)
            sub["disclosure_combo_lookup"] = {k[5:]: v for k, v in sub.items()
                                              if k.startswith("DISC_")}
            sub["n"] = len(keep)
            sub["SE"] = round(math.sqrt(0.25 / len(keep)), 4)
            grp[gname] = sub
        rec["groups"] = grp
        out["splits"][name] = rec
        print(f"\n== {name} n={rec['n']} SE={rec['SE']} ==", flush=True)
        print(f"  完备电池8={ {k: rec[k] for k in OLD8} }", flush=True)
        print(f"  n_slots={rec['n_slots']} majority_own={rec['majority_own']} | "
              f"R1={rec['R1_mode_nslots']} R2={rec['R2_mode']} | "
              f"R3(oracle)={rec['R3_oracle_mode_nslots']} R4(B)={rec['R4_mode_Bsubset']}",
              flush=True)
        print(f"  max_naive_complete={rec['max_naive_complete']} "
              f"max_naive_plus={rec['max_naive_plus']} max_naive_plus4={rec['max_naive_plus4']}",
              flush=True)
        for g in ("L", "nonL"):
            gg_ = grp[g]
            if gg_["n"] == 0:
                print(f"  [{g}] n=0", flush=True)
            else:
                print(f"  [{g}] n={gg_['n']} R1={gg_['R1_mode_nslots']} "
                      f"n_slots={gg_['n_slots']} max_plus={gg_['max_naive_plus']} "
                      f"max_complete={gg_['max_naive_complete']}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "rules.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                        encoding="utf-8")
    print(f"\n[done] → {RESULTS / 'rules.json'}", flush=True)


if __name__ == "__main__":
    main()
