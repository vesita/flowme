#!/usr/bin/env python3
"""S4 —— F2 免费规则地板（P15 同电池 8 条 + max_naive，含计数类 `n_cue`，纪律 D2）。

用法：PYTHONDONTWRITEBYTECODE=1 uv run python experiments/fix_negation/s4_floor.py [--after results/s3_after.json]

口径（PREREG §1 F2 + §2 步骤 4）：
  * `fit = train_S`、`eval = eval_S`（与 prod_card_audit/a2_floor.py 同公式）；
  * 查表逐字复用 `prod_card_audit/common.py::lookup`（无 min_support、未见键回退 train 多数）；
  * **口径 A（cls）**：目标 = `labels[:,0]` 且只取 `t_labels>0` ⇒ 与 `cls_acc` / 端到端 `M_cls` 同分母；
  * **口径 B（detect）**：目标 = 全 eval 样本「有无否定证据」⇒ 与端到端 `M_detect` 同分母；
  * 卡侧 = 端到端 `respond` 的 M_cls / M_detect（读 s3_before/s3_after 产物）。
词表 = `dtseek.tasks.builtin.negation.dataset.NEG_MARKERS`（闭集，不新造）。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True

from common import (  # noqa: E402
    HERE, NEG, SEEDS, lookup, split_of,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.tasks.builtin.negation.dataset import NEG_EXCEPTIONS, NEG_MARKERS  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset  # noqa: E402

PUNCTS = "，、：；！？…—"
RULES = ("majority", "len_bucket", "first_char", "last_char", "punct_pattern",
         "lex_rule", "lex_first_key", "n_cue")
_RX = re.compile("|".join(sorted((re.escape(w) for w in NEG_MARKERS), key=len, reverse=True)))
_RAW: dict[str, float] = {}
#: 披露项（不进 max_naive）：数据集自己的句子级过滤（同形/虚化/A-not-A）
from dtseek.tasks.builtin.negation.dataset import reject_reason  # noqa: E402


def acc(pred, y) -> float:
    return sum(1 for a, b in zip(pred, y) if a == b) / max(1, len(y))


def feats_of(texts: list[str]) -> dict[str, list]:
    f: dict[str, list] = defaultdict(list)
    for t in texts:
        f["first_char"].append(t[0] if t else "")
        f["last_char"].append(t[-1] if t else "")
        f["len_bucket"].append(len(t) // 6)
        f["punct_pattern"].append("".join(sorted(set(t) & set(PUNCTS))))
        m = _RX.search(t)
        f["lex_rule"].append(1 if m else None)          # 词表字面：命中 ⇒ 否定类
        f["lex_lit"].append(int(m is not None))        # 披露：纯字面（不回退、不拟合）
        f["lex_first_key"].append(m.group(0) if m else None)
        f["n_cue"].append(len(_RX.findall(t)))
        # 披露项：数据集自己的判否逻辑（有标记且句子未被丢弃 ⇒ 认为有否定）
        f["lex_dataset"].append(int(m is not None and reject_reason(t) is None))
    return f


def y_cls_of(samples, spec) -> list[int]:
    ds = GenericTaskDataset(samples, NanoCharTokenizer(), spec)
    return [int(ds[i]["labels"][0]) for i in range(len(ds))]


def battery(y_tr: list[int], y_ev: list[int], f_tr: dict, f_ev: dict) -> dict:
    """8 条规则 + max_naive；fit=train、eval=eval（与 a2_floor 同结构）。"""
    glob = Counter(y_tr).most_common(1)[0][0]
    rec: dict = {"n_train": len(y_tr), "n_eval": len(y_ev),
                 "label_dist_train": dict(Counter(y_tr)),
                 "label_dist_eval": dict(Counter(y_ev)),
                 "train_majority": glob, "seen_rate": {}, "raw": {}}
    _RAW.clear()
    _RAW["majority"] = acc([glob] * len(y_ev), y_ev)
    for k in ("len_bucket", "first_char", "last_char", "punct_pattern",
              "lex_first_key", "n_cue"):
        pred, sr = lookup(f_tr[k], y_tr, f_ev[k])
        rec["seen_rate"][k] = round(sr, 4)
        _RAW[k] = acc(pred, y_ev)
    # lex_rule：纯字面、无拟合；未命中回退 train 多数（a2_floor 同）
    _RAW["lex_rule"] = acc([glob if v is None else v for v in f_ev["lex_rule"]], y_ev)
    # 披露项：数据集句子级过滤版
    rec["disclosed_lex_dataset"] = round(acc(f_ev["lex_dataset"], y_ev), 4)
    rec["disclosed_lex_lit"] = round(acc(f_ev["lex_lit"], y_ev), 4)
    for k in RULES:
        rec[k] = round(_RAW[k], 4)
    rec["max_naive_raw"] = max(_RAW[k] for k in RULES)
    rec["max_naive"] = round(rec["max_naive_raw"], 6)
    rec["winner"] = max(RULES, key=lambda k: _RAW[k])
    return rec


def load_card_rates(path: Path) -> dict | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text(encoding="utf-8"))
    neg = d.get("neg") or {}
    return {S: {m: neg[str(S)]["summary"][m]["rate"] for m in ("detect", "cls", "span")}
            for S in SEEDS} if neg else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--after", default=str(HERE / "results" / "s3_after.json"))
    args = ap.parse_args(argv)
    spec = resolve_tasks([NEG])[NEG].spec

    cards = {
        "现默认卡（改前）": load_card_rates(HERE / "results" / "s3_before.json"),
        "改后默认卡": load_card_rates(Path(args.after)),
    }
    out: dict = {"prereg": "experiments/fix_negation/PREREG.md", "rules": list(RULES),
                 "cue_source": "builtin/negation/dataset.py::NEG_MARKERS", "seeds": {}, "cards": {}}
    for k, v in cards.items():
        if v:
            out["cards"][k] = v
            print(f"[卡] {k}: " + "  ".join(
                f"s{S} M_cls={v[S]['cls']:.4f} M_detect={v[S]['detect']:.4f}" for S in SEEDS))

    for S in SEEDS:
        ev, tr = split_of(NEG, S)
        # ---- 口径 A：cls（real positives，labels[:,0]）----
        ytr_all, yev_all = y_cls_of(tr, spec), y_cls_of(ev, spec)
        rtr_txt = [tr[i]["text"] for i, v in enumerate(ytr_all) if v > 0]
        rev_txt = [ev[i]["text"] for i, v in enumerate(yev_all) if v > 0]
        ytr = [v for v in ytr_all if v > 0]
        yev = [v for v in yev_all if v > 0]
        ftr, fev = feats_of(rtr_txt), feats_of(rev_txt)
        cls_rec = battery(ytr, yev, ftr, fev)

        # ---- 口径 B：detect（全样本，有无否定证据）----
        dtr = [1 if v > 0 else 0 for v in ytr_all]
        dev = [1 if v > 0 else 0 for v in yev_all]
        gtr_txt = [tr[i]["text"] for i in range(len(tr))]
        gev_txt = [ev[i]["text"] for i in range(len(ev))]
        gftr, gfev = feats_of(gtr_txt), feats_of(gev_txt)
        det_rec = battery(dtr, dev, gftr, gfev)

        out["seeds"][str(S)] = {"cls": cls_rec, "detect": det_rec}
        print(f"\n[s{S} 口径A cls] n_tr={cls_rec['n_train']} n_ev={cls_rec['n_eval']} dist={cls_rec['label_dist_eval']}")
        print("   " + "  ".join(f"{k}={cls_rec[k]:.4f}" for k in RULES)
              + f"  max_naive={cls_rec['max_naive']:.4f} (winner={cls_rec['winner']})")
        print(f"[s{S} 口径B detect] n_tr={det_rec['n_train']} n_ev={det_rec['n_eval']} "
              f"dist={det_rec['label_dist_eval']}")
        print("   " + "  ".join(f"{k}={det_rec[k]:.4f}" for k in RULES)
              + f"  max_naive={det_rec['max_naive']:.4f} (winner={det_rec['winner']})")

        # ---- 卡 − 地板 ----
        for name, rates in out["cards"].items():
            if not rates:
                continue
            d_cls = rates[S]["cls"] - cls_rec["max_naive_raw"]
            d_det = rates[S]["detect"] - det_rec["max_naive_raw"]
            out.setdefault("minus_floor", {}).setdefault(name, {})[str(S)] = {
                "cls": round(d_cls, 6), "detect": round(d_det, 6),
                "below_or_tie_cls": bool(d_cls <= 1e-12),
                "below_or_tie_detect": bool(d_det <= 1e-12)}
            print(f"   [卡−地板] {name} s{S}: cls {d_cls:+.4f} / detect {d_det:+.4f}"
                  f"{'  <=0 未超过免费规则' if d_cls <= 1e-12 or d_det <= 1e-12 else ''}")

    res = HERE / "results" / "s4_floor.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[save] {res}\nS4_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
