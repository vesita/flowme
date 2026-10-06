#!/usr/bin/env python3
"""输入类型分类探针（E-A / P-类型），复用 experiments/adversarial_routing/probe_e4_style.py 口径。

- 冻结基座 checkpoints/base_encoder.pt -> NanoDocEncoder，有效 token doc_memory mask-mean-pool，D=128
- 零初始化单层逻辑回归，AdamW lr=0.05 wd=1e-4 200 epoch batch 64，seed 42/43
- 训练只用 id_train.jsonl（5 类 × 400），评测 ① id_test ② adversarial ③ known_answers
- 对照：① 打乱标签 ② 随机高斯特征（正交阵随机投影明确弃用：精确保持余弦，不构成独立证据）
- 判据在 README §5 预注册，本脚本只照抄执行
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(ROOT / "src"))

from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.artifacts import read_base  # noqa: E402
from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = ["plain", "candidates", "cloze", "multi_turn", "unknown"]
P2_BAR = 0.70          # README §5：unknown 召回下限
P1_MARGIN = 0.10       # README §5：对抗集相对分布内的最大允许跌幅


def load(name: str) -> list[dict]:
    with open(HERE / name, encoding="utf-8") as fp:
        return [json.loads(l) for l in fp if l.strip()]


def train_probe(X, y, num_classes, seed, lr=0.05, epochs=200, l2=1e-4):
    torch.manual_seed(seed)
    linear = nn.Linear(X.shape[1], num_classes)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    opt = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=l2)
    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)
    for _ in range(epochs):
        linear.train()
        for bx, by in loader:
            opt.zero_grad()
            F.cross_entropy(linear(bx), by).backward()
            opt.step()
    linear.eval()
    return linear


def predict(linear, X):
    with torch.no_grad():
        return linear(X).argmax(dim=-1)


def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    n = len(y_true)
    per = {}
    for c, name in enumerate(LABELS):
        idx = [i for i, t in enumerate(y_true) if t == c]
        per[name] = (sum(y_pred[i] == c for i in idx) / len(idx)) if idx else None
    return {
        "acc": sum(t == p for t, p in zip(y_true, y_pred)) / max(1, n),
        "n": n,
        "per_class_recall": per,
        "pred_dist": {LABELS[c]: y_pred.count(c) / n for c in range(len(LABELS))},
        "confusion": {LABELS[t]: {LABELS[p]: sum(1 for tt, pp in zip(y_true, y_pred)
                                                 if tt == t and pp == p)
                                  for p in range(len(LABELS)) if p != t}
                      for t in sorted(set(y_true))
                      if any(tt == t and pp != t for tt, pp in zip(y_true, y_pred))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="checkpoints/base_encoder.pt")
    ap.add_argument("--seeds", default="42,43")
    ap.add_argument("--out", default=str(HERE / "results.json"))
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device} seeds={seeds}", flush=True)

    base_info = read_base(args.base_ckpt)
    tokenizer = NanoCharTokenizer()
    enc = NanoDocEncoder(vocab_size=base_info["vocab_size"],
                         hidden_dim=base_info["hidden_dim"],
                         **base_info["encoder_kwargs"]).to(device)
    enc.load_state_dict(base_info["doc_encoder"])
    enc.eval()

    def extract(texts, batch_size=64):
        feats = []
        for i in range(0, len(texts), batch_size):
            es = [tokenizer.encode(t, max_length=128, padding=True)
                  for t in texts[i:i + batch_size]]
            max_l = max(len(e["input_ids"]) for e in es)
            inps, masks = [], []
            for e in es:
                pad = max_l - len(e["input_ids"])
                inps.append(e["input_ids"] + [0] * pad)
                masks.append(e["attention_mask"] + [0] * pad)
            ti = torch.tensor(inps, dtype=torch.long, device=device)
            tm = torch.tensor(masks, dtype=torch.bool, device=device)
            with torch.no_grad():
                mem = enc(ti, attention_mask=tm)
                m = tm.unsqueeze(-1).float()
                feats.append(((mem * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu())
        return torch.cat(feats, dim=0)

    train_rows, test_rows = load("id_train.jsonl"), load("id_test.jsonl")
    adv_rows, ka_rows = load("adversarial.jsonl"), load("known_answers.jsonl")
    y = lambda rows: torch.tensor([LABELS.index(r["true_label"]) for r in rows]).long()
    y_tr, y_te, y_adv, y_ka = y(train_rows), y(test_rows), y(adv_rows), y(ka_rows)

    print("[probe] 提取特征（分布内训练/测试 + 对抗 + 已知答案）...", flush=True)
    X_tr = extract([r["text"] for r in train_rows])
    X_te = extract([r["text"] for r in test_rows])
    X_adv = extract([r["text"] for r in adv_rows])
    X_ka = extract([r["text"] for r in ka_rows])
    print(f"[probe] train={tuple(X_tr.shape)} test={tuple(X_te.shape)} "
          f"adv={tuple(X_adv.shape)} ka={tuple(X_ka.shape)}", flush=True)

    stats = json.load(open(HERE / "stats.json", encoding="utf-8"))
    M_TEST = stats["test_majority_baseline"]
    M_ADV = stats["adv_majority_baseline"]

    results = {"config": {
        "base_ckpt": args.base_ckpt, "seeds": seeds, "labels": LABELS,
        "probe": "零初始化单层逻辑回归 AdamW lr=0.05 wd=1e-4 200ep bs64",
        "feat": "冻结基座 doc_memory 有效 token mean-pool D=128",
        "n_train": len(train_rows), "n_test": len(test_rows),
        "n_adv": len(adv_rows), "n_ka": len(ka_rows),
        "majority_baseline": {"id_test": M_TEST, "adv": M_ADV},
        "P1_margin": P1_MARGIN, "P2_bar": P2_BAR,
        "train_data": "id_train.jsonl（与测试/对抗/已知答案文本零重合，见 stats.json）",
    }, "seeds": {}}

    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        out: dict = {}

        lin = train_probe(X_tr, y_tr, len(LABELS), seed)
        out["real"] = {
            "id_test": metrics(y_te.tolist(), predict(lin, X_te).tolist()),
            "adv": metrics(y_adv.tolist(), predict(lin, X_adv).tolist()),
            "known_answer": metrics(y_ka.tolist(), predict(lin, X_ka).tolist()),
        }
        out["real"]["known_answer"]["per_item"] = [
            {"id": ka_rows[i]["id"], "true": LABELS[y_ka[i]],
             "pred": LABELS[p], "ok": bool(p == y_ka[i])}
            for i, p in enumerate(predict(lin, X_ka).tolist())]

        # 对抗集分规则明细
        out["adv_per_rule"] = {}
        for rule in sorted({r["rule"] for r in adv_rows}):
            idx = [i for i, r in enumerate(adv_rows) if r["rule"] == rule]
            p = predict(lin, X_adv).tolist()
            sub_t = [int(y_adv[i]) for i in idx]
            sub_p = [p[i] for i in idx]
            m = metrics(sub_t, sub_p)
            m["pred_as_surface_naive"] = sum(
                1 for i in idx
                if LABELS[p[i]] == adv_rows[i]["naive_label"]) / len(idx)
            out["adv_per_rule"][rule] = m

        # 对照 1：打乱标签
        perm = torch.randperm(y_tr.size(0))
        lin_sh = train_probe(X_tr, y_tr[perm], len(LABELS), seed + 1000)
        out["control_shuffled_label"] = {
            "id_test_acc": metrics(y_te.tolist(), predict(lin_sh, X_te).tolist())["acc"],
            "adv_acc": metrics(y_adv.tolist(), predict(lin_sh, X_adv).tolist())["acc"]}

        # 对照 2：随机高斯特征（剥离全部语言结构）
        torch.manual_seed(seed + 2000)
        lin_rf = train_probe(torch.randn_like(X_tr), y_tr, len(LABELS), seed + 2000)
        out["control_random_feature"] = {
            "id_test_acc": metrics(y_te.tolist(), predict(lin_rf, torch.randn_like(X_te)).tolist())["acc"],
            "adv_acc": metrics(y_adv.tolist(), predict(lin_rf, torch.randn_like(X_adv)).tolist())["acc"]}

        # ---- 探索性（未预注册，仅供 §7 讨论）：置信度门限扫描 ----
        # 不变量的另一半是「置信度低于阈值 ⇒ 判 unknown」，这里看是否存在一个阈值
        # 能同时满足 P1/P2/P3（阈值扫描不参与 §5 的判定）。
        with torch.no_grad():
            logits_adv, logits_te = lin(X_adv), lin(X_te)
            p_adv = F.softmax(logits_adv, dim=-1)
            p_te = F.softmax(logits_te, dim=-1)
        base_adv = predict(lin, X_adv).tolist()
        base_te = predict(lin, X_te).tolist()
        sweep = []
        for tau in [round(0.05 * i, 2) for i in range(21)]:
            pr = [(unknown_i if mx < tau else arg)
                  for arg, mx, unknown_i in
                  ((int(p.argmax()), float(p.max()), LABELS.index("unknown"))
                   for p in p_adv)]
            pr_te = [LABELS.index("unknown") if float(p.max()) < tau else arg
                     for arg, p in zip(base_te, p_te)]
            m = metrics(y_adv.tolist(), pr)
            unk_r = m["per_class_recall"]["unknown"]
            m_te = metrics(y_te.tolist(), pr_te)
            ok = (m["acc"] >= m_te["acc"] - P1_MARGIN
                  and (unk_r or 0) >= P2_BAR and m["acc"] > M_ADV)
            sweep.append({"tau": tau, "adv_acc": m["acc"],
                          "id_acc": m_te["acc"], "unknown_recall": unk_r,
                          "P1_P2_P3_all_pass": bool(ok)})
        out["exploratory_threshold_sweep"] = sweep

        a = out["real"]["adv"]
        unknown_recall = a["per_class_recall"]["unknown"]
        p1 = a["acc"] >= out["real"]["id_test"]["acc"] - P1_MARGIN
        p2 = unknown_recall is not None and unknown_recall >= P2_BAR
        p3 = a["acc"] > M_ADV
        out["criteria"] = {"P1_pass": bool(p1), "P2_pass": bool(p2), "P3_pass": bool(p3),
                           "unknown_recall_adv": unknown_recall,
                           "drop_pt": (out["real"]["id_test"]["acc"] - a["acc"]) * 100}
        results["seeds"][str(seed)] = out
        print(f"[seed {seed}] id={out['real']['id_test']['acc']*100:.2f}% "
              f"adv={a['acc']*100:.2f}% unk_recall={unknown_recall:.3f} "
              f"KA={out['real']['known_answer']['acc']*100:.1f}% "
              f"| shuffle {out['control_shuffled_label']['id_test_acc']*100:.1f}%/"
              f"{out['control_shuffled_label']['adv_acc']*100:.1f}% "
              f"| randfeat {out['control_random_feature']['id_test_acc']*100:.1f}%/"
              f"{out['control_random_feature']['adv_acc']*100:.1f}% "
              f"| P1={p1} P2={p2} P3={p3}", flush=True)

    # ---- 聚合与预注册判定（README §5） ----
    def agg(get):
        vals = [get(results["seeds"][str(s)]) for s in seeds]
        return {"mean": float(np.mean(vals)), "values": [float(v) for v in vals],
                "range": float(max(vals) - min(vals))}

    results["aggregate"] = {
        "id_test_acc": agg(lambda o: o["real"]["id_test"]["acc"]),
        "adv_acc": agg(lambda o: o["real"]["adv"]["acc"]),
        "unknown_recall_adv": agg(lambda o: o["criteria"]["unknown_recall_adv"]),
        "known_answer_acc": agg(lambda o: o["real"]["known_answer"]["acc"]),
        "ctrl_shuffle_id": agg(lambda o: o["control_shuffled_label"]["id_test_acc"]),
        "ctrl_shuffle_adv": agg(lambda o: o["control_shuffled_label"]["adv_acc"]),
        "ctrl_randfeat_id": agg(lambda o: o["control_random_feature"]["id_test_acc"]),
        "ctrl_randfeat_adv": agg(lambda o: o["control_random_feature"]["adv_acc"]),
    }
    crit = {k: [results["seeds"][str(s)]["criteria"][k] for s in seeds]
            for k in ("P1_pass", "P2_pass", "P3_pass")}
    all_pass = all(all(v) for v in crit.values())
    any_fail_both = any((not v[0]) and (not v[1]) for v in crit.values())
    verdict = ("不变量成立" if all_pass else
               ("不变量不成立（诊断：模型判不了类型）" if any_fail_both else "证据不足"))
    results["verdict"] = {
        "judged": verdict, "per_seed_criteria": crit,
        "rule": "README §5：P1/P2/P3 全部在两 seed 通过 ⇒ 成立；任一判据在两 seed 都不通过 ⇒ 不成立；其余 ⇒ 证据不足",
        "majority_baseline": {"id_test": M_TEST, "adv": M_ADV},
    }
    results["verdict"]["exploratory_threshold_note"] = (
        "未预注册的探索：是否存在置信度阈值使 P1∧P2∧P3 同时成立 —— "
        + "; ".join(f"seed {s}: " + (
            "none" if not any(x["P1_P2_P3_all_pass"]
                              for x in results["seeds"][s]["exploratory_threshold_sweep"])
            else "tau=" + ",".join(str(x["tau"]) for x in
                                   results["seeds"][s]["exploratory_threshold_sweep"]
                                   if x["P1_P2_P3_all_pass"]))
                    for s in map(str, seeds)))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n[probe] id={results['aggregate']['id_test_acc']['mean']*100:.2f}% "
          f"adv={results['aggregate']['adv_acc']['mean']*100:.2f}% "
          f"unk_recall={results['aggregate']['unknown_recall_adv']['mean']:.3f} "
          f"M_adv={M_ADV} ⇒ {verdict}", flush=True)
    print(f"[probe] 结果已存 {args.out}", flush=True)


if __name__ == "__main__":
    main()
