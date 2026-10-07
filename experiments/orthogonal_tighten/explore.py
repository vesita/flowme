#!/usr/bin/env python3
"""P23 **事后探索（非判据）**：只交「在 a_bal 上覆盖率仍高」的信号，看候选集能收到多小、acc 多高。

**纪律**：本文件不参与 T0–T5 判定（判定只看 PREREG 预注册的 A1–A6）；臂与组合是**看到
T2/T4 的覆盖率之后**挑的 ⇒ 全部标「事后/非判据」，不得用于声称 T1 通过。
用法：uv run python experiments/orthogonal_tighten/explore.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import run_eval as R  # noqa: E402  只读复用（同一目录）

RESULTS = HERE / "results"
#: 事后组合：索引 = 信号编号 0..5（0=①句式 1=②n_slots 2=③稀有度b 3=④合格候选 4=⑤首末字 5=⑥类型库存）
ARMS = {"E1_①∩②": (0, 1), "E2_①∩④": (0, 3), "E3_①∩②∩④": (0, 1, 3),
        "E4_①∩②∩④∩⑥": (0, 1, 3, 5)}


def main() -> None:
    spec = R.C.Spec()
    torch.manual_seed(0)
    tab = R.MM.skeleton_table()
    sA = R.MM.subsets_A(tab)
    rows = R.C.load_rows(R.SPLIT_PATH["a_bal"])
    train = R.C.load_rows(R.C.TCH / "data" / "train.jsonl")
    st = R.build_stats(train)
    sig = R.make_keeps(st, sA, rows)
    blob = R.C.encode_rows(rows, spec, "cpu")
    labs = R.C.label_tensors(rows)

    keeps: dict[str, list] = {}
    for name, idxs in ARMS.items():
        cur = None
        for i in idxs:
            if cur is None:
                cur = [list(k) for k in sig[i]]
            else:
                cur = [[x for x in cur[j] if x in set(sig[i][j])]
                       for j in range(len(cur))]
        keeps[name] = cur
    keeps = {"S1_①句式": [list(k) for k in sig[0]]} | keeps   # 参照臂（=C-mode）
    out: dict = {"note": "事后探索，非判据（组合由 T2/T4 覆盖率挑出）",
                 "n": len(rows), "arms": {}}
    for name, ks in keeps.items():
        s = R.size_stats(ks)
        out["arms"][name] = {"size": s, "cov_gold": round(
            sum(1 for i, k in enumerate(ks) if blob["skel"][i].item() in k) / len(rows), 6)}
        print(f"[keep] {name} size med={s['med']} p25={s['p25']} p75={s['p75']} "
              f"empty={s['empty']} cov={out['arms'][name]['cov_gold']}")

    for seed in R.C.SEEDS:
        model = R.C.load_arm("UP", seed, spec, "cpu")
        with torch.no_grad():
            vb = R.C.v_bag_of(blob["v_items"], blob["item_mask"])
            lab_in = {k: labs[k] for k in ("type_t", "role_t", "cls_t", "pos_b")}
            lab_in["mask"] = labs["mask"]
            sk, _, _ = model.forward(blob["v_sent"], vb, blob["v_items"],
                                     blob["item_mask"], lab_in, blob["h"], blob["hmask"])
        logits = sk.cpu()
        gold = blob["skel"].tolist()
        f = R.mask_argmax(logits, None)
        fc = [int(a == b) for a, b in zip(f, gold)]
        best = 0.1615
        s1p = R.mask_argmax(logits, keeps["S1_①句式"])
        s1c = [int(a == b) for a, b in zip(s1p, gold)]
        row = {"F": round(R.acc_of(f, gold), 4),
               "S1_①句式": {"acc": round(R.acc_of(s1p, gold), 4),
                            "card_minus_best": round(R.acc_of(s1p, gold) - best, 4)}}
        for name in ARMS:
            ks = keeps[name]
            p = R.mask_argmax(logits, ks)
            c = [int(a == b) for a, b in zip(p, gold)]
            row[name] = {"acc": round(R.acc_of(p, gold), 4),
                         "dF": _paired(fc, c),
                         "dS1_①句式": _paired(s1c, c),
                         "card_minus_best": round(R.acc_of(p, gold) - best, 4)}
        out.setdefault("seeds", {})[f"s{seed}"] = row
        print(f"[eval] s{seed} {json.dumps(row, ensure_ascii=False)}")
        del model

    (RESULTS / "exploratory.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                              encoding="utf-8")
    print(f"[done] → {RESULTS / 'exploratory.json'}（非判据）")


def _paired(a: list[int], b: list[int]) -> dict:
    import math
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    se = math.sqrt(var / n)
    return {"delta": round(m, 6), "se": round(se, 6),
            "t": round(m / se, 2) if se > 0 else None}


if __name__ == "__main__":
    main()
