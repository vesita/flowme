#!/usr/bin/env python3
"""对抗集路由验证：复刻 E4 口径的线性探针，分别在 ①原始分布测试集 ②对抗集 上评测。

E4 口径（experiments/rank_budget/measure_e4.py，逐项对齐）：
  冻结基座 NanoDocEncoder → 有效 token doc_memory mask-mean-pool（D=128）
  单层逻辑回归（零初始化，AdamW lr=0.05，weight_decay=1e-4，200 epoch，batch 64）
  4 卡各 600 训 + 200 测 + 中性池 600/200（5 类），seed 42/43/44
  同一个探针训练一次，分别在原始测试集与对抗集上评测；每次 fit 前 reseed，
  保证真实臂与对照臂的初始化/洗牌逐项一致（dev-notes/11 测量纪律）。

对照（全部跑在同一对抗集上）：
  打乱标签 / 随机高斯特征 —— dev-notes/11 教训：正交阵随机投影保余弦、无效，本脚本不用。
指标逐项报：route_acc / route_bg_fp / route_macro / 对抗集准确率 / 已知答案对照。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
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

TASKS = ["pronoun", "sentiment", "relation", "person"]
LABELS_5 = TASKS + ["neutral"]
CATS = ("c1", "c2", "c3", "c4", "c5")
HERE = Path(__file__).parent


# ── 与 E4 逐字对齐的部件 ──────────────────────────────────────────────────
def compute_metrics(y_true, y_pred, num_classes):
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    total = len(y_true)
    correct = sum(cm[i][i] for i in range(num_classes))
    f1s, recalls = [], []
    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes) if r != c)
        fn = sum(cm[c][p] for p in range(num_classes) if p != c)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1s.append((2 * prec * rec / max(1e-9, prec + rec)) if (prec + rec) > 0 else 0.0)
        recalls.append(cm[c][c] / max(1, sum(cm[c])))
    return {
        "accuracy": correct / max(1, total),
        "macro_f1": sum(f1s) / len(f1s),
        "macro_recall": sum(recalls) / len(recalls),
        "confusion_matrix": cm,
    }


def fit_probe(X_train, y_train, num_classes, seed, lr=0.05, epochs=200, l2=1e-4):
    """E4 的 train_logistic_regression（训练行为逐项一致），返回训好的线性层。"""
    torch.manual_seed(seed)  # 每臂 fit 前 reseed ⇒ 初始化/洗牌逐项一致
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


@torch.no_grad()
def eval_probe(linear, X):
    return linear(X).argmax(dim=-1)


def mine_neutral_texts(n: int, max_len: int = 64) -> list[str]:
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


# ── 语体（表面特征）统计：解释「捷径」用，纯字符串操作 ────────────────────
def surface_flags(texts):
    from dtseek.tasks.builtin.idiom.lexicon import IDIOMS
    from dtseek.tasks.builtin.person.dataset import FEMALE_NAMES, MALE_NAMES
    from dtseek.tasks.builtin.pronoun.dataset import PRONOUN_MAP
    from dtseek.tasks.builtin.relation.dataset import VOCAB
    from dtseek.tasks.builtin.sentiment.dataset import EMOTION_KEYWORDS

    prons = sorted({p for _, lst in PRONOUN_MAP for p in lst}, key=len, reverse=True)
    emo = [w for _, ws in EMOTION_KEYWORDS for w in ws]
    names = list(MALE_NAMES) + list(FEMALE_NAMES)
    neg = set("不没未非")  # 不含「别」：特别/别的是 会假阳性
    idiom_pool = sorted(set(IDIOMS) - VOCAB)

    def pcount(t):
        occ, n = [False] * len(t), 0
        for p in prons:
            s = 0
            while True:
                i = t.find(p, s)
                if i < 0:
                    break
                if not any(occ[i:i + len(p)]):
                    for k in range(i, i + len(p)):
                        occ[k] = True
                    n += 1
                s = i + 1
        return n

    if not texts:
        return {}
    pc = em = ng = nm = idn = 0
    for t in texts:
        pc += pcount(t)
        em += any(w in t for w in emo)
        ng += any(ch in t for ch in neg)
        nm += any(x in t for x in names)
        idn += sum(1 for i in idiom_pool if i in t)
    n = len(texts)
    return {"n": n, "pronoun_mean": round(pc / n, 2),
            "frac_emotion": round(em / n, 3), "frac_negation": round(ng / n, 3),
            "frac_name": round(nm / n, 3), "idiom_mean": round(idn / n, 3)}


def route_metrics(y_true, y_pred, present_classes):
    """route_acc / route_bg_fp / route_macro（present_classes 上的宏召回）。"""
    total = len(y_true)
    acc = sum(t == p for t, p in zip(y_true, y_pred)) / max(1, total)
    bg = LABELS_5.index("neutral")
    bg_rows = [i for i, t in enumerate(y_true) if t == bg]
    bg_fp = (sum(1 for i in bg_rows if y_pred[i] != bg) / len(bg_rows)) if bg_rows else None
    recalls = {}
    for c in present_classes:
        rows = [i for i, t in enumerate(y_true) if t == c]
        recalls[LABELS_5[c]] = sum(1 for i in rows if y_pred[i] == c) / len(rows)
    return {
        "route_acc": acc,
        "route_bg_fp": bg_fp,
        "route_macro": sum(recalls.values()) / len(recalls),
        "macro_classes": list(recalls),
        "per_class_recall": recalls,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-ckpt", default="checkpoints/base_encoder.pt")
    ap.add_argument("--seeds", default="42,43,44")
    ap.add_argument("--adv", default=str(HERE / "adversarial.jsonl"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--n-train-per-class", type=int, default=600)
    ap.add_argument("--n-test-per-class", type=int, default=200)
    args = ap.parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    t_start = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] 设备={device} seeds={seeds}", flush=True)

    # ── 对抗集 ────────────────────────────────────────────────────────────
    adv_rows = [json.loads(l) for l in open(args.adv, encoding="utf-8")]
    adv_texts = [r["text"] for r in adv_rows]
    adv_y_t = torch.tensor([LABELS_5.index(r["true_label"]) for r in adv_rows])
    label_dist = Counter(r["true_label"] for r in adv_rows)
    majority_adv = max(label_dist.values()) / len(adv_rows)
    print(f"[probe] 对抗集 {len(adv_rows)} 条 | 标签 {dict(label_dist)} | "
          f"多数类基线 {majority_adv:.3f}")

    # ── 基座与特征 ────────────────────────────────────────────────────────
    base_info = read_base(args.base_ckpt)
    tokenizer = NanoCharTokenizer()
    doc_encoder = NanoDocEncoder(
        vocab_size=base_info["vocab_size"],
        hidden_dim=base_info["hidden_dim"],
        **base_info["encoder_kwargs"],
    ).to(device)
    doc_encoder.load_state_dict(base_info["doc_encoder"])
    doc_encoder.eval()

    def extract_features(texts):
        feats = []
        for i in range(0, len(texts), 64):
            chunk = texts[i:i + 64]
            encs = [tokenizer.encode(t, max_length=128, padding=True) for t in chunk]
            max_l = max(len(e["input_ids"]) for e in encs)
            inps, masks = [], []
            for e in encs:
                pad = max_l - len(e["input_ids"])
                inps.append(e["input_ids"] + [0] * pad)
                masks.append(e["attention_mask"] + [0] * pad)
            tinps = torch.tensor(inps, dtype=torch.long, device=device)
            tmasks = torch.tensor(masks, dtype=torch.bool, device=device)
            with torch.no_grad():
                mem = doc_encoder(tinps, attention_mask=tmasks)
                m = tmasks.unsqueeze(-1).float()
                feats.append(((mem * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu())
        return torch.cat(feats, dim=0)

    t0 = time.time()
    X_adv = extract_features(adv_texts)
    print(f"[probe] 对抗集特征 {tuple(X_adv.shape)} {time.time()-t0:.1f}s", flush=True)

    registered = all_tasks()
    n_train, n_test = args.n_train_per_class, args.n_test_per_class
    total_need = n_train + n_test
    per_seed, train_texts_all = [], {}
    t_data = 0.0

    for seed in seeds:
        t_seed = time.time()
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        data_by_class = {}
        for name in TASKS:
            raw = registered[name].build_dataset(total_need * 2)
            pos = [d["text"] for d in raw if len(d.get("spans", [])) > 0]
            if len(pos) < total_need:
                pos = [d["text"] for d in raw]
            random.Random(seed).shuffle(pos)
            data_by_class[name] = pos[:total_need]
        neutral = mine_neutral_texts(total_need * 2)
        random.Random(seed).shuffle(neutral)
        data_by_class["neutral"] = neutral[:total_need]
        train_texts_all[seed] = {k: list(v) for k, v in data_by_class.items()}

        feats = {n: (extract_features(data_by_class[n][:n_train]),
                     extract_features(data_by_class[n][n_train:total_need]))
                 for n in LABELS_5}
        Xtr4 = torch.cat([feats[n][0] for n in TASKS])
        ytr4 = torch.cat([torch.full((n_train,), i) for i in range(4)])
        Xte4 = torch.cat([feats[n][1] for n in TASKS])
        yte4 = torch.cat([torch.full((n_test,), i) for i in range(4)])
        Xtr5 = torch.cat([Xtr4, feats["neutral"][0]])
        ytr5 = torch.cat([ytr4, torch.full((n_train,), 4)])
        Xte5 = torch.cat([Xte4, feats["neutral"][1]])
        yte5 = torch.cat([yte4, torch.full((n_test,), 4)])

        r = {"seed": seed}

        # ── 4 类复刻（对照 E4 的 98.50%）+ 在对抗集 sentiment 子集上的次要评测
        lin4 = fit_probe(Xtr4, ytr4, 4, seed)
        m4 = compute_metrics(yte4.tolist(), eval_probe(lin4, Xte4).tolist(), 4)
        r["acc4"] = m4["accuracy"]
        sent_idx = [i for i, v in enumerate(adv_y_t.tolist()) if v == 1]
        p4_adv = eval_probe(lin4, X_adv[torch.tensor(sent_idx)]).tolist()
        r["acc4_on_adv_sentiment"] = sum(1 for p in p4_adv if p == 1) / len(p4_adv)

        # ── 5 类真实臂：训练一次，两处评测
        lin5 = fit_probe(Xtr5, ytr5, 5, seed)
        p5_test = eval_probe(lin5, Xte5).tolist()
        p5_adv = eval_probe(lin5, X_adv).tolist()
        r["acc5"] = compute_metrics(yte5.tolist(), p5_test, 5)["accuracy"]
        r["orig5"] = route_metrics(yte5.tolist(), p5_test, list(range(5)))
        r["adv"] = route_metrics(adv_y_t.tolist(), p5_adv, [1, 4])
        r["adv_pred_dist"] = {LABELS_5[k]: v
                              for k, v in sorted(Counter(p5_adv).items())}

        r["adv_per_cat"] = {}
        for cat in CATS:
            idxs = [i for i, x in enumerate(adv_rows) if x["category"] == cat]
            yt = [adv_y_t[i].item() for i in idxs]
            yp = [p5_adv[i] for i in idxs]
            surface = adv_rows[idxs[0]]["surface_card"]
            r["adv_per_cat"][cat] = {
                "acc": sum(t == p for t, p in zip(yt, yp)) / len(idxs),
                "pred_dist": {LABELS_5[k]: v for k, v in sorted(Counter(yp).items())},
                "trap_rate": (sum(1 for p in yp if LABELS_5[p] == surface) / len(idxs)
                              if surface in LABELS_5 else None),
            }

        # ── 已知答案对照（15 条）
        ka_idx = [i for i, x in enumerate(adv_rows) if x["known_answer"]]
        ka_true = [adv_y_t[i].item() for i in ka_idx]
        ka_pred = [p5_adv[i] for i in ka_idx]
        r["known_answer_acc"] = sum(t == p for t, p in zip(ka_true, ka_pred)) / len(ka_idx)
        r["known_answer_detail"] = [
            {"id": adv_rows[i]["id"], "text": adv_rows[i]["text"],
             "true": LABELS_5[adv_y_t[i].item()], "pred": LABELS_5[p5_adv[i]]}
            for i in ka_idx
        ]

        # ── 对照 1：打乱标签（5 类，同一对抗集）
        torch.manual_seed(seed)
        perm = torch.randperm(ytr5.size(0))
        lin_s = fit_probe(Xtr5, ytr5[perm], 5, seed)
        r["ctrl_shuffled"] = {
            "orig": compute_metrics(yte5.tolist(),
                                    eval_probe(lin_s, Xte5).tolist(), 5)["accuracy"],
            "adv": sum(t == p for t, p in zip(adv_y_t.tolist(),
                                              eval_probe(lin_s, X_adv).tolist()))
                   / len(adv_y_t),
        }

        # ── 对照 2：随机高斯特征（5 类，同一标签）
        Xtr_r = torch.randn_like(Xtr5)
        Xte_r = torch.randn_like(Xte5)
        Xadv_r = torch.randn_like(X_adv)
        lin_r = fit_probe(Xtr_r, ytr5, 5, seed)
        r["ctrl_random"] = {
            "orig": compute_metrics(yte5.tolist(),
                                    eval_probe(lin_r, Xte_r).tolist(), 5)["accuracy"],
            "adv": sum(t == p for t, p in zip(adv_y_t.tolist(),
                                              eval_probe(lin_r, Xadv_r).tolist()))
                   / len(adv_y_t),
        }

        # ── 语体统计（原始分布训练数据按类；对抗集按类）
        r["surface_train"] = {n: surface_flags(data_by_class[n][:200])
                              for n in LABELS_5}
        r["surface_adv"] = {
            cat: surface_flags([x["text"] for x in adv_rows
                                if x["category"] == cat]) for cat in CATS
        }

        per_seed.append(r)
        t_data += time.time() - t_seed
        print(f"[seed {seed}] acc4={r['acc4']:.4f} acc5={r['acc5']:.4f} "
              f"adv={r['adv']['route_acc']:.4f} KA={r['known_answer_acc']:.4f} "
              f"bg_fp={r['adv']['route_bg_fp']:.4f} "
              f"shuf=({r['ctrl_shuffled']['orig']:.3f}/{r['ctrl_shuffled']['adv']:.3f}) "
              f"rand=({r['ctrl_random']['orig']:.3f}/{r['ctrl_random']['adv']:.3f}) "
              f"[{time.time()-t_seed:.0f}s]", flush=True)

    # ── 重复样本检查：对抗集文本是否与 E4 各卡数据完全重复 ────────────────
    all_e4 = set()
    for seed_texts in train_texts_all.values():
        for v in seed_texts.values():
            all_e4.update(v)
    dup = [t for t in adv_texts if t in all_e4]

    # ── 分类别指标跨 seed 聚合 ────────────────────────────────────────────
    per_cat = {}
    for cat in CATS:
        accs = [r["adv_per_cat"][cat]["acc"] for r in per_seed]
        traps = [r["adv_per_cat"][cat]["trap_rate"] for r in per_seed]
        dist = Counter()
        for r in per_seed:
            dist.update(r["adv_per_cat"][cat]["pred_dist"])
        per_cat[cat] = {
            "acc": {"per_seed": [round(a, 6) for a in accs],
                    "mean": float(np.mean(accs)), "std": float(np.std(accs))},
            "trap_rate": ({"per_seed": [round(t, 6) for t in traps],
                           "mean": float(np.mean(traps)),
                           "std": float(np.std(traps))}
                          if all(t is not None for t in traps) else None),
            "pred_dist_3seed_sum": dict(sorted(dist.items())),
        }

    def agg(key, sub=None):
        vals = [r[key] if sub is None else r[key][sub] for r in per_seed]
        return {"per_seed": [round(v, 6) for v in vals],
                "mean": float(np.mean(vals)), "std": float(np.std(vals))}

    P0 = agg("acc5")["mean"]
    adv_acc = agg("adv", "route_acc")["mean"]
    ka_acc = agg("known_answer_acc")["mean"]
    m0 = majority_adv

    # 判据（README §4 预注册，跑前写死，此处只照抄执行）
    if adv_acc < P0 - 0.20 or abs(adv_acc - m0) <= 0.05:
        decision = "路由主要靠语体捷径"
    elif adv_acc >= P0 - 0.10:
        decision = "路由有真本事"
    else:
        decision = "证据不足、需扩样本"

    results = {
        "meta": {
            "seeds": seeds, "device": str(device), "n_adv": len(adv_rows),
            "label_dist": dict(label_dist), "majority_adv": m0,
            "majority_orig5": 0.20, "dup_with_e4_data": dup,
            "wall_seconds": round(time.time() - t_start, 1),
            "data_and_probe_seconds": round(t_data, 1),
        },
        "replication_e4": {
            "acc4": agg("acc4"), "acc5": agg("acc5"),
            "e4_reference_acc4_mean": 0.9850, "e4_reference_acc5_mean": 0.9873,
        },
        "known_answer": {
            "acc": ka_acc, "per_seed": [r["known_answer_acc"] for r in per_seed],
            "detail": per_seed[0]["known_answer_detail"],
        },
        "original_test": {
            "route_acc": agg("acc5"),
            "route_bg_fp": agg("orig5", "route_bg_fp"),
            "route_macro": agg("orig5", "route_macro"),
        },
        "adversarial": {
            "route_acc": agg("adv", "route_acc"),
            "route_bg_fp": agg("adv", "route_bg_fp"),
            "route_macro": agg("adv", "route_macro"),
            "acc4_on_sentiment_subset": agg("acc4_on_adv_sentiment"),
            "per_category": per_cat,
            "pred_dist_seed42": per_seed[0]["adv_pred_dist"],
        },
        "controls": {
            "shuffled_label": {"orig": agg("ctrl_shuffled", "orig"),
                               "adv": agg("ctrl_shuffled", "adv")},
            "random_feature": {"orig": agg("ctrl_random", "orig"),
                               "adv": agg("ctrl_random", "adv")},
        },
        "surface_stats": {
            "train_by_class_seed42": per_seed[0]["surface_train"],
            "adv_by_category": per_seed[0]["surface_adv"],
        },
        "verdict": {
            "P0_original_acc": round(P0, 6), "adv_acc": round(adv_acc, 6),
            "majority_adv": m0, "known_answer_acc": round(ka_acc, 6),
            "drop_pt": round((P0 - adv_acc) * 100, 2), "decision": decision,
        },
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fp:
        json.dump(results, fp, indent=2, ensure_ascii=False)

    # 打印顺序：先已知答案对照，再结论（README 同序）
    print("\n== 已知答案对照（15 条，先于结论核验） ==")
    for d in results["known_answer"]["detail"]:
        mark = "OK " if d["true"] == d["pred"] else "ERR"
        print(f"  [{mark}] {d['id']} true={d['true']:9s} pred={d['pred']:9s} {d['text']}")
    print(f"  known_answer_acc = {ka_acc:.4f}（per-seed "
          f"{[round(r['known_answer_acc'], 4) for r in per_seed]}）")

    print("\n== 结果摘要 ==")
    print(f"  复刻 4 类 acc = {results['replication_e4']['acc4']['mean']:.4f}（E4 参考 0.9850）")
    print(f"  复刻 5 类 acc = {P0:.4f}（E4 参考 0.9873）  ← 判据 P0")
    print(f"  对抗集 acc    = {adv_acc:.4f}（多数类基线 {m0:.3f}）")
    print(f"  route_bg_fp   = {results['adversarial']['route_bg_fp']['mean']:.4f}"
          f"（原始 {results['original_test']['route_bg_fp']['mean']:.4f}）")
    print(f"  route_macro   = {results['adversarial']['route_macro']['mean']:.4f}"
          f"（原始 {results['original_test']['route_macro']['mean']:.4f}）")
    print(f"  对照 打乱标签: orig {results['controls']['shuffled_label']['orig']['mean']:.4f}"
          f" / adv {results['controls']['shuffled_label']['adv']['mean']:.4f}")
    print(f"  对照 随机特征: orig {results['controls']['random_feature']['orig']['mean']:.4f}"
          f" / adv {results['controls']['random_feature']['adv']['mean']:.4f}")
    print(f"  对抗集∩E4数据 重复样本 = {len(dup)}")
    print(f"\n[判据] P0={P0:.4f} adv={adv_acc:.4f} drop={(P0-adv_acc)*100:.2f}pt "
          f"M0={m0:.3f} ⇒ {decision}")
    print(f"[probe] 总耗时 {time.time()-t_start:.1f}s -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
