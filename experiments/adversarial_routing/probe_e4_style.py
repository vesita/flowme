#!/usr/bin/env python3
"""复刻 E4 线性探针，在原始分布与 adversarial.jsonl 上出对照数字。

口径严格对齐 experiments/rank_budget/measure_e4.py 与本目录 README §2–§5：
- 冻结基座 checkpoints/base_encoder.pt -> NanoDocEncoder，有效 token mask mean-pool，D=128
- 单层逻辑回归（零初始化，AdamW lr=0.05 wd=1e-4，200 epoch，batch 64）
- 训练集只用原始分布（4 卡 build_dataset 各 600 训 + 中性池 600 训），绝不接触对抗集
- 三个 seed：42 / 43 / 44
- 评测两处：① 原始分布测试集（4 卡各 200 + 中性 200）② 对抗集 225 条
- 对照：① 打乱标签 ② 随机高斯特征（正交阵投影按 README §5 明确弃用）
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.artifacts import read_base  # noqa: E402
from dtseek.tasks.corpus import CORPUS_GLOB, resolve_corpus_files  # noqa: E402
from dtseek.tasks.plugin import all_tasks  # noqa: E402
from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402

LABELS = ["pronoun", "sentiment", "relation", "person", "neutral"]
CARDS = LABELS[:4]
M0 = 0.60  # 对抗集经验多数类基线（README §1，跑前算死）


# ---------- 探针（与 measure_e4.train_logistic_regression 同构，改为返回模型） ----------
def train_probe(X_train, y_train, num_classes, seed, lr=0.05, epochs=200, l2=1e-4):
    torch.manual_seed(seed)
    D = X_train.shape[1]
    linear = nn.Linear(D, num_classes)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    opt = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=l2)
    loader = DataLoader(TensorDataset(X_train, y_train), batch_size=64, shuffle=True)
    for _ in range(epochs):
        linear.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = F.cross_entropy(linear(bx), by)
            loss.backward()
            opt.step()
    linear.eval()
    return linear


def predict(linear, X):
    with torch.no_grad():
        return linear(X).argmax(dim=-1)


# ---------- 指标（README §3 口径） ----------
def metrics(y_true: list[int], y_pred: list[int]) -> dict:
    n = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(1, n)
    # route_bg_fp：true=neutral 的样本被分到 4 张卡的比例
    neu_idx = [i for i, t in enumerate(y_true) if t == LABELS.index("neutral")]
    bg_fp = (sum(y_pred[i] != LABELS.index("neutral") for i in neu_idx) / len(neu_idx)) if neu_idx else None
    # route_macro：按 y_true 中实际覆盖到的类做召回宏平均
    present = sorted(set(y_true))
    recalls = []
    for c in present:
        idx = [i for i, t in enumerate(y_true) if t == c]
        recalls.append(sum(y_pred[i] == c for i in idx) / len(idx))
    return {
        "route_acc": acc,
        "route_bg_fp": bg_fp,
        "route_macro": float(np.mean(recalls)) if recalls else None,
        "n": n,
        "classes_present": [LABELS[c] for c in present],
        "per_class_recall": {LABELS[c]: r for c, r in zip(present, recalls)},
        "pred_dist": {LABELS[c]: y_pred.count(c) / n for c in range(len(LABELS))},
        "confusion_true_by_pred": {
            LABELS[t]: {LABELS[p]: sum(1 for tt, pp in zip(y_true, y_pred) if tt == t and pp == p)
                        for p in range(len(LABELS)) if any(tt == t and pp == p for tt, pp in zip(y_true, y_pred))}
            for t in present
        },
    }


# ---------- 中性池（照抄 measure_e4.mine_neutral_texts） ----------
def mine_neutral_texts(n, max_len=64):
    import re
    ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
    SENT_SPLIT = re.compile(r"[。！？\n；;]+")
    pool = []
    for p in resolve_corpus_files(CORPUS_GLOB):
        with open(p, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                t = ROLE_PREFIX.sub("", line.strip())
                for s in SENT_SPLIT.split(t):
                    s = s.strip()
                    if 10 <= len(s) <= max_len:
                        pool.append(s)
                        if len(pool) >= n:
                            return pool
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="checkpoints/base_encoder.pt")
    ap.add_argument("--n-train-per-class", type=int, default=600)
    ap.add_argument("--n-test-per-class", type=int, default=200)
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--adv", default="experiments/adversarial_routing/adversarial.jsonl")
    ap.add_argument("--out", default="experiments/adversarial_routing/probe_results.json")
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device} seeds={seeds}", flush=True)

    base_info = read_base(args.base_ckpt)
    tokenizer = NanoCharTokenizer()
    enc = NanoDocEncoder(
        vocab_size=base_info["vocab_size"],
        hidden_dim=base_info["hidden_dim"],
        **base_info["encoder_kwargs"],
    ).to(device)
    enc.load_state_dict(base_info["doc_encoder"])
    enc.eval()

    def extract(texts, batch_size=64):
        feats = []
        for i in range(0, len(texts), batch_size):
            chunk = texts[i:i + batch_size]
            es = [tokenizer.encode(t, max_length=128, padding=True) for t in chunk]
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

    # 对抗集（固定文本，特征与 seed 无关，只提一次）
    adv_rows = [json.loads(l) for l in open(args.adv, encoding="utf-8")]
    assert len(adv_rows) == 225, len(adv_rows)
    adv_texts = [r["text"] for r in adv_rows]
    X_adv = extract(adv_texts)
    print(f"[probe] 对抗集特征: {tuple(X_adv.shape)}", flush=True)
    known_idx = [i for i, r in enumerate(adv_rows) if r.get("known_answer")]
    assert len(known_idx) == 15, len(known_idx)

    registered = all_tasks()
    total_need = args.n_train_per_class + args.n_test_per_class

    results = {"config": {
        "base_ckpt": args.base_ckpt, "seeds": seeds,
        "n_train_per_class": args.n_train_per_class, "n_test_per_class": args.n_test_per_class,
        "labels": LABELS, "M0": M0, "probe": "零初始化单层逻辑回归 AdamW lr=0.05 wd=1e-4 200ep bs64",
        "train_data": "原始分布（4 卡 build_dataset + 中性池），未接触对抗集",
        "adv_path": args.adv, "n_adv": len(adv_rows), "n_known_answer": len(known_idx),
    }, "seeds": {}}

    for seed in seeds:
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        # ---- 原始分布数据（与 measure_e4 逐项一致） ----
        data = {}
        for name in CARDS:
            raw = registered[name].build_dataset(total_need * 2)
            pos = [d["text"] for d in raw if len(d.get("spans", [])) > 0]
            if len(pos) < total_need:
                pos = [d["text"] for d in raw]
            random.Random(seed).shuffle(pos)
            data[name] = pos[:total_need]
        neu = mine_neutral_texts(total_need * 2)
        random.Random(seed).shuffle(neu)
        data["neutral"] = neu[:total_need]

        Xtr_l, ytr_l, Xte_l, yte_l = [], [], [], []
        for ci, name in enumerate(LABELS):
            t = data[name]
            ftr = extract(t[:args.n_train_per_class])
            fte = extract(t[args.n_train_per_class:total_need])
            Xtr_l.append(ftr); ytr_l.append(torch.full((len(t[:args.n_train_per_class]),), ci))
            Xte_l.append(fte); yte_l.append(torch.full((len(t[args.n_train_per_class:total_need]),), ci))
        X_train = torch.cat(Xtr_l); y_train = torch.cat(ytr_l).long()
        X_test = torch.cat(Xte_l); y_test = torch.cat(yte_l).long()
        print(f"[seed {seed}] train={tuple(X_train.shape)} test={tuple(X_test.shape)}", flush=True)

        y_adv = torch.tensor([LABELS.index(r["true_label"]) for r in adv_rows])
        y_true_adv = y_adv.tolist()
        y_true_test = y_test.tolist()

        def evaluate(linear):
            p_test = predict(linear, X_test).tolist()
            p_adv = predict(linear, X_adv).tolist()
            return metrics(y_true_test, p_test), metrics(y_true_adv, p_adv), p_test, p_adv

        out_seed = {}

        # ---- ① 真实探针 ----
        lin = train_probe(X_train, y_train, 5, seed)
        m_test, m_adv, p_test, p_adv = evaluate(lin)
        out_seed["real"] = {"orig_test": m_test, "adv": m_adv}

        # 对抗集分项：每类 + 「预测成表面卡 A」比例
        per_cat, a_stats = {}, {"n_valid_a": 0, "n_pred_a": 0}
        for cat in ["c1", "c2", "c3", "c4", "c5"]:
            idx = [i for i, r in enumerate(adv_rows) if r["category"] == cat]
            r0 = adv_rows[idx[0]]
            surf = r0["surface_card"]
            cc = metrics([y_true_adv[i] for i in idx], [p_adv[i] for i in idx])
            entry = {"n": len(idx), "surface_card": surf, "true_label": r0["true_label"],
                     "route_acc": cc["route_acc"], "pred_dist": cc["pred_dist"]}
            if surf in LABELS:
                hits = sum(p_adv[i] == LABELS.index(surf) for i in idx)
                entry["pred_as_surface_A"] = hits / len(idx)
                a_stats["n_valid_a"] += len(idx); a_stats["n_pred_a"] += hits
            else:
                entry["pred_as_surface_A"] = None
                entry["note"] = f"surface_card={surf} 不在 E4 五类标签空间，无法「预测成 A」"
            per_cat[cat] = entry
        out_seed["adv_per_category"] = per_cat
        out_seed["adv_surface_A_rate"] = (a_stats["n_pred_a"] / a_stats["n_valid_a"]) if a_stats["n_valid_a"] else None
        out_seed["adv_surface_A_counts"] = a_stats

        # 已知答案 15 条
        ka_pred = [p_adv[i] for i in known_idx]
        ka_true = [y_true_adv[i] for i in known_idx]
        out_seed["known_answer"] = {
            "metrics": metrics(ka_true, ka_pred),
            "per_item": [{"id": adv_rows[i]["id"], "category": adv_rows[i]["category"],
                          "true_label": adv_rows[i]["true_label"],
                          "pred": LABELS[p_adv[i]], "ok": p_adv[i] == y_true_adv[i]} for i in known_idx],
        }

        # ---- 对照 1：打乱标签 ----
        perm = torch.randperm(y_train.size(0))
        lin_sh = train_probe(X_train, y_train[perm], 5, seed + 1000)
        s_test, s_adv, _, _ = evaluate(lin_sh)
        out_seed["control_shuffled_label"] = {"orig_test": s_test, "adv": s_adv}

        # ---- 对照 2：随机高斯特征 ----
        torch.manual_seed(seed + 2000)
        Xr_tr = torch.randn_like(X_train); Xr_te = torch.randn_like(X_test); Xr_adv = torch.randn_like(X_adv)
        lin_rf = train_probe(Xr_tr, y_train, 5, seed + 2000)
        rf_test = metrics(y_true_test, predict(lin_rf, Xr_te).tolist())
        rf_adv = metrics(y_true_adv, predict(lin_rf, Xr_adv).tolist())
        out_seed["control_random_feature"] = {"orig_test": rf_test, "adv": rf_adv}

        results["seeds"][str(seed)] = out_seed
        print(f"[seed {seed}] real: orig={m_test['route_acc']*100:.2f}% adv={m_adv['route_acc']*100:.2f}% "
              f"| shuffled: {s_test['route_acc']*100:.2f}%/{s_adv['route_acc']*100:.2f}% "
              f"| randfeat: {rf_test['route_acc']*100:.2f}%/{rf_adv['route_acc']*100:.2f}% "
              f"| A-rate={out_seed['adv_surface_A_rate']} KA={out_seed['known_answer']['metrics']['route_acc']*100:.1f}%",
              flush=True)

    # ---- 聚合：均值与极差 ----
    def agg(path):
        vals = []
        for s in seeds:
            d = results["seeds"][str(s)]
            for k in path:
                d = d[k]
            vals.append(d)
        vals = [v for v in vals if v is not None]
        if not vals:
            return None
        return {"mean": float(np.mean(vals)), "range": float(np.max(vals) - np.min(vals)),
                "values": [float(v) for v in vals]}

    agg_paths = {
        "P0_orig_acc": ["real", "orig_test", "route_acc"],
        "adv_acc": ["real", "adv", "route_acc"],
        "orig_bg_fp": ["real", "orig_test", "route_bg_fp"],
        "adv_bg_fp": ["real", "adv", "route_bg_fp"],
        "orig_macro": ["real", "orig_test", "route_macro"],
        "adv_macro": ["real", "adv", "route_macro"],
        "adv_surface_A_rate": ["adv_surface_A_rate"],
        "known_answer_acc": ["known_answer", "metrics", "route_acc"],
        "ctrl_shuffle_orig_acc": ["control_shuffled_label", "orig_test", "route_acc"],
        "ctrl_shuffle_adv_acc": ["control_shuffled_label", "adv", "route_acc"],
        "ctrl_randfeat_orig_acc": ["control_random_feature", "orig_test", "route_acc"],
        "ctrl_randfeat_adv_acc": ["control_random_feature", "adv", "route_acc"],
    }
    results["aggregate"] = {k: agg(p) for k, p in agg_paths.items()}
    P0 = results["aggregate"]["P0_orig_acc"]["mean"]
    adv = results["aggregate"]["adv_acc"]["mean"]
    drop = (P0 - adv) * 100
    within_m0 = abs(adv - M0) * 100 <= 5
    if drop > 20 or within_m0:
        verdict = "路由主要靠语体捷径"
    elif adv >= P0 - 0.10:
        verdict = "路由有真本事"
    else:
        verdict = "证据不足、需扩样本"
    results["verdict"] = {
        "P0": P0, "adv": adv, "M0": M0, "drop_pt": drop,
        "within_M0_5pt": within_m0, "judged": verdict,
        "rule": "README §4：adv<P0-20pt 或 |adv-M0|<=5pt ⇒ 捷径；adv>=P0-10pt ⇒ 有真本事；否则证据不足",
    }
    # 已知答案跨 seed 一致性
    ka_items = {}
    for s in seeds:
        for it in results["seeds"][str(s)]["known_answer"]["per_item"]:
            ka_items.setdefault(it["id"], []).append(it["pred"])
    results["known_answer_cross_seed"] = {
        k: {"preds": v, "stable": len(set(v)) == 1} for k, v in sorted(ka_items.items())
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)
    print(f"\n[probe] P0={P0*100:.2f}% adv={adv*100:.2f}% drop={drop:.2f}pt withinM0={within_m0} => {verdict}")
    print(f"[probe] 结果已存 {args.out}")


if __name__ == "__main__":
    main()
