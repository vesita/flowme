#!/usr/bin/env python3
"""E4: 路由可分性上界（线性探针实验）。

测法：
不训练任何新卡，在冻结基座上提取文本表征（如对有效 token 的 doc_memory 做 mean pooling，维度 D=128）。
做线性探针（基于 PyTorch 的多分类逻辑回归，无隐藏层），预测文本来自哪张卡（4 分类任务：pronoun, sentiment, relation, person）。
输入样本为各卡的真实训练集（每类各 600 条训练 + 200 条测试，共 3200 样本）。
同时加入「中性池」进行 5 分类测试（4 卡 + 真实中性背景句）。

指标：Accuracy, Macro-F1, 混淆矩阵，多数类基线对比。
空洞负对照：
1. 真实标签打乱负对照（shuffled labels）：自变量确实被破坏，打乱后应当跌落至多数类/随机猜测基准（~25% / 20%）。
2. 随机高斯特征对照（random features）：特征无任何语言学表征信息。
3. 探讨与反思 dev-notes/11 中的正交变换对照（若采用正交阵旋转，点积与距离精确保持，不构成有效负对照）。
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
from dtseek.tasks.runtime import GenericTaskDataset  # noqa: E402
from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402


def compute_metrics(y_true: list[int], y_pred: list[int], num_classes: int) -> dict:
    cm = [[0] * num_classes for _ in range(num_classes)]
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1
    total = len(y_true)
    correct = sum(cm[i][i] for i in range(num_classes))
    acc = correct / max(1, total)

    f1s = []
    for c in range(num_classes):
        tp = cm[c][c]
        fp = sum(cm[r][c] for r in range(num_classes) if r != c)
        fn = sum(cm[c][p] for p in range(num_classes) if p != c)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec / max(1e-9, prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    macro_f1 = sum(f1s) / len(f1s)
    return {
        "accuracy": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "per_class_f1": f1s,
    }


def train_logistic_regression(X_train: torch.Tensor, y_train: torch.Tensor,
                              X_test: torch.Tensor, y_test: torch.Tensor,
                              num_classes: int, lr: float = 0.05,
                              epochs: int = 200, l2: float = 1e-4) -> dict:
    """标准凸优化多分类逻辑回归线性探针。"""
    D = X_train.shape[1]
    linear = nn.Linear(D, num_classes)
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)
    opt = torch.optim.AdamW(linear.parameters(), lr=lr, weight_decay=l2)

    ds_train = TensorDataset(X_train, y_train)
    loader = DataLoader(ds_train, batch_size=64, shuffle=True)

    for epoch in range(epochs):
        linear.train()
        for bx, by in loader:
            opt.zero_grad()
            logits = linear(bx)
            loss = F.cross_entropy(logits, by)
            loss.backward()
            opt.step()

    linear.eval()
    with torch.no_grad():
        test_logits = linear(X_test)
        preds = test_logits.argmax(dim=-1).tolist()
        y_true = y_test.tolist()

    return compute_metrics(y_true, preds, num_classes)


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", default="checkpoints/base_encoder.pt")
    parser.add_argument("--n-train-per-class", type=int, default=600)
    parser.add_argument("--n-test-per-class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="experiments/rank_budget/e4_results.json")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[E4] 设备: {device}, 随机种子: {args.seed}")

    base_info = read_base(args.base_ckpt)
    tokenizer = NanoCharTokenizer()
    doc_encoder = NanoDocEncoder(
        vocab_size=base_info["vocab_size"],
        hidden_dim=base_info["hidden_dim"],
        **base_info["encoder_kwargs"],
    ).to(device)
    doc_encoder.load_state_dict(base_info["doc_encoder"])
    doc_encoder.eval()

    task_names = ["pronoun", "sentiment", "relation", "person"]
    registered = all_tasks()

    total_need = args.n_train_per_class + args.n_test_per_class
    data_by_class = {}

    for name in task_names:
        card = registered[name]
        raw = card.build_dataset(total_need * 2)
        # 过滤只保留正例或真实任务样本
        pos_samples = [d["text"] for d in raw if len(d.get("spans", [])) > 0]
        if len(pos_samples) < total_need:
            pos_samples = [d["text"] for d in raw]
        random.Random(args.seed).shuffle(pos_samples)
        data_by_class[name] = pos_samples[:total_need]
        print(f"  收集任务 {name:10s} 文本: {len(data_by_class[name])} 条")

    # 中性池
    neutral_texts = mine_neutral_texts(total_need * 2)
    random.Random(args.seed).shuffle(neutral_texts)
    data_by_class["neutral"] = neutral_texts[:total_need]
    print(f"  收集中性池 neutral    文本: {len(data_by_class['neutral'])} 条")

    # 提取特征
    def extract_features(texts: list[str]) -> torch.Tensor:
        feats = []
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            chunk = texts[i : i + batch_size]
            encs = [tokenizer.encode(t, max_length=128, padding=True) for t in chunk]
            max_l = max(len(e["input_ids"]) for e in encs)
            inps = []
            masks = []
            for e in encs:
                pad_l = max_l - len(e["input_ids"])
                inps.append(e["input_ids"] + [0] * pad_l)
                masks.append(e["attention_mask"] + [0] * pad_l)
            tinps = torch.tensor(inps, dtype=torch.long, device=device)
            tmasks = torch.tensor(masks, dtype=torch.bool, device=device)
            with torch.no_grad():
                mem = doc_encoder(tinps, attention_mask=tmasks)  # [B, L, D]
                # mask mean pooling
                m = tmasks.unsqueeze(-1).float()
                pooled = (mem * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
                feats.append(pooled.cpu())
        return torch.cat(feats, dim=0)

    # 1. 四卡分类设置 (4-class)
    X_train_4_list = []
    y_train_4_list = []
    X_test_4_list = []
    y_test_4_list = []

    for idx, name in enumerate(task_names):
        texts = data_by_class[name]
        train_t = texts[: args.n_train_per_class]
        test_t = texts[args.n_train_per_class : total_need]
        f_train = extract_features(train_t)
        f_test = extract_features(test_t)
        X_train_4_list.append(f_train)
        y_train_4_list.append(torch.full((len(train_t),), idx, dtype=torch.long))
        X_test_4_list.append(f_test)
        y_test_4_list.append(torch.full((len(test_t),), idx, dtype=torch.long))

    X_train_4 = torch.cat(X_train_4_list, dim=0)
    y_train_4 = torch.cat(y_train_4_list, dim=0)
    X_test_4 = torch.cat(X_test_4_list, dim=0)
    y_test_4 = torch.cat(y_test_4_list, dim=0)

    # 多数类基线
    majority_acc_4 = 1.0 / len(task_names)

    # 真实线性探针 (4-class)
    real_res_4 = train_logistic_regression(X_train_4, y_train_4, X_test_4, y_test_4, len(task_names))
    print(f"\n[E4] 4-Class 线性探针 (pronoun, sentiment, relation, person):")
    print(f"  Accuracy: {real_res_4['accuracy']*100:.2f}% | Macro-F1: {real_res_4['macro_f1']*100:.2f}%")
    print(f"  多数类基线: {majority_acc_4*100:.2f}%")
    print(f"  混淆矩阵:\n{np.array(real_res_4['confusion_matrix'])}")

    # 对照1: 随机打乱标签负对照 (Shuffled Labels)
    shuffled_idx = torch.randperm(y_train_4.size(0))
    y_train_shuffled = y_train_4[shuffled_idx]
    ctrl_shuffled_res_4 = train_logistic_regression(X_train_4, y_train_shuffled, X_test_4, y_test_4, len(task_names))
    print(f"\n[E4 对照 1] 真实打乱标签负对照 (Shuffled Labels):")
    print(f"  Accuracy: {ctrl_shuffled_res_4['accuracy']*100:.2f}% | Macro-F1: {ctrl_shuffled_res_4['macro_f1']*100:.2f}%")

    # 对照2: 随机高斯特征对照 (Random Gaussian Features)
    X_train_rand = torch.randn_like(X_train_4)
    X_test_rand = torch.randn_like(X_test_4)
    ctrl_randfeat_res_4 = train_logistic_regression(X_train_rand, y_train_4, X_test_rand, y_test_4, len(task_names))
    print(f"\n[E4 对照 2] 随机特征对照 (Random Gaussian Features):")
    print(f"  Accuracy: {ctrl_randfeat_res_4['accuracy']*100:.2f}% | Macro-F1: {ctrl_randfeat_res_4['macro_f1']*100:.2f}%")

    # 2. 五分类设置（4 卡 + 中性池 neutral）
    all_5_names = task_names + ["neutral"]
    neutral_texts = data_by_class["neutral"]
    f_train_neu = extract_features(neutral_texts[: args.n_train_per_class])
    f_test_neu = extract_features(neutral_texts[args.n_train_per_class : total_need])

    X_train_5 = torch.cat([X_train_4, f_train_neu], dim=0)
    y_train_5 = torch.cat([y_train_4, torch.full((len(f_train_neu),), 4, dtype=torch.long)], dim=0)
    X_test_5 = torch.cat([X_test_4, f_test_neu], dim=0)
    y_test_5 = torch.cat([y_test_4, torch.full((len(f_test_neu),), 4, dtype=torch.long)], dim=0)

    majority_acc_5 = 1.0 / len(all_5_names)
    real_res_5 = train_logistic_regression(X_train_5, y_train_5, X_test_5, y_test_5, len(all_5_names))
    print(f"\n[E4] 5-Class 线性探针 (+ 中性背景池):")
    print(f"  Accuracy: {real_res_5['accuracy']*100:.2f}% | Macro-F1: {real_res_5['macro_f1']*100:.2f}%")
    print(f"  多数类基线: {majority_acc_5*100:.2f}%")
    print(f"  混淆矩阵:\n{np.array(real_res_5['confusion_matrix'])}")

    results = {
        "four_classes": {
            "tasks": task_names,
            "majority_acc": majority_acc_4,
            "real_probe": real_res_4,
            "shuffled_label_control": ctrl_shuffled_res_4,
            "random_feature_control": ctrl_randfeat_res_4,
        },
        "five_classes_with_neutral": {
            "tasks": all_5_names,
            "majority_acc": majority_acc_5,
            "real_probe": real_res_5,
        },
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2, ensure_ascii=False)
        print(f"\n[E4] 结果已保存至 {args.out}")


if __name__ == "__main__":
    main()
