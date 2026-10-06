#!/usr/bin/env python3
"""按 ctype / 词头拆解每跑的逐样本正确率 —— 判据的决定性拆分。

为什么必须拆：adv = `heldout_pair`（**留出替换对**，输出串训练从未见过 ⇒ 背词表给不出答案）
+ `shifted_pos`（**训练词表**、只是换了位置 ⇒ 背词表 + 位置不变式仍可能全对）。
只报 adv 总分 71% 时，「位置不变的表内记忆」与「对新说法的真泛化」是**混在一起**的，
必须分开报，否则结论会看错。

用法：uv run python experiments/select_rerank/analyze.py
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import RerankModel, RerankSpec  # noqa: E402

DATA = HERE / "data"
SPLITS = ("train", "test", "adv")


def load_rows(arm: str) -> list[dict]:
    rows = []
    for s in SPLITS:
        with open(DATA / arm / f"{s}.jsonl", encoding="utf-8") as fp:
            for line in fp:
                r = json.loads(line)
                r["split_name"] = s
                rows.append(r)
    return rows


def per_correct(name: str, arm: str) -> list[bool]:
    spec = RerankSpec.from_build_spec()
    model = RerankModel(spec)
    rows = load_rows(arm)
    fp = None
    for p in (HERE / "cache").glob(f"{arm}_*_ctx{spec.max_len_ctx}_cand{spec.max_len_cand}.pt"):
        fp = p
    assert fp, f"找不到 {arm} 的向量缓存"
    blob = torch.load(fp, map_location="cpu", weights_only=True)
    assert blob["n"] == len(rows), "缓存与数据不一致"
    head = model.head
    head.load_state_dict(torch.load(HERE / "weights" / f"{name}.pt", map_location="cpu"))
    head.eval()
    with torch.no_grad():
        logits = head(blob["v_ctx"], blob["v_cand"])
    return (logits.argmax(-1) == blob["labels"]).tolist()


def bucket(rows: list[dict], ok: list[bool], key) -> dict[str, dict]:
    agg: dict[str, list[bool]] = defaultdict(list)
    for r, o in zip(rows, ok):
        agg[key(r)].append(o)
    out = {}
    for k, v in sorted(agg.items()):
        n = len(v)
        acc = sum(v) / n
        se = (0.25 / n) ** 0.5
        out[k] = {"n": n, "acc": round(acc, 4), "se": round(se, 4),
                  "margin_over_se": round((acc - 0.5) / se, 2) if n else None,
                  "passes_2se": bool(acc - 0.5 > 2 * se)}
    return out


def main() -> None:
    table = {}
    for arm in ("clean", "shortcut"):
        rows = load_rows(arm)
        for seed in (42, 43):
            for rand in (False, True):
                name = f"{arm}_s{seed}{'_rand' if rand else ''}"
                ok = per_correct(name, arm)
                assert len(ok) == len(rows)
                adv_rows = [r for r in rows if r["split_name"] == "adv"]
                adv_ok = [o for r, o in zip(rows, ok) if r["split_name"] == "adv"]
                by_split = bucket(rows, ok, lambda r: r["split_name"])
                by_ctype = bucket(adv_rows, adv_ok, lambda r: r["ctype"])
                entry = {"by_split": by_split, "adv_by_ctype": by_ctype}
                if not rand:               # 逐词（只对真标签跑；随机标签跑没有解读价值）
                    entry["adv_by_word"] = {
                        t: bucket([r for r in adv_rows if r["ctype"] == t],
                                  [o for r, o in zip(adv_rows, adv_ok) if r["ctype"] == t],
                                  lambda r: r["sub_word"])
                        for t in ("heldout_pair", "shifted_pos")}
                table[name] = entry
                print(f"\n=== {name}")
                for s, v in by_split.items():
                    print(f"  {s:5s} n={v['n']:5d} acc={v['acc']:.4f} 余/SE={v['margin_over_se']:+6.2f}"
                          f" 过2SE={v['passes_2se']}")
                for c, v in by_ctype.items():
                    print(f"  adv/{c:13s} n={v['n']:5d} acc={v['acc']:.4f} 余/SE={v['margin_over_se']:+6.2f}"
                          f" 过2SE={v['passes_2se']}")
                if not rand and arm == "clean":
                    for t in ("heldout_pair", "shifted_pos"):
                        pw = entry["adv_by_word"][t]
                        top = sorted(pw.items(), key=lambda kv: -kv[1]["n"])[:8]
                        s = " ".join(f"{w}:{v['n']}×{v['acc']:.2f}" for w, v in top)
                        print(f"  逐词 {t}: {s}")
    (HERE / "results" / "breakdown.json").write_text(
        json.dumps(table, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n→ results/breakdown.json")


if __name__ == "__main__":
    main()
