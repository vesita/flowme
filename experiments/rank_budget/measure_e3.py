#!/usr/bin/env python3
"""E3: 卡片间子空间冲突分析。

测法：
对每张卡计算「任务向量」：该卡 loss 对基座输出 doc_memory [B, L, D] 的平均梯度（维度 D），
两两计算余弦相似度与主角（Principal Angles）。
同时还可以考察各任务每一步解码器的梯度敏感方向，以及直接在 token 级梯度上计算子空间重叠。
此外，两两算余弦矩阵与夹角矩阵。

判据：|cos| 大 => 两张卡抢同一批特征，联合训练必然互扰。
"""
import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.artifacts import read_base, read_card  # noqa: E402
from dtseek.tasks.plugin import all_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, task_loss  # noqa: E402
from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402


def compute_subspace_principal_angles(mat1: torch.Tensor, mat2: torch.Tensor, top_k: int = 16) -> list[float]:
    """提取两组样本梯度的 top_k 主成分子空间，计算主角（度数）。

    mat1: [N1, D], mat2: [N2, D]
    若直接在 N > D 时对全空间做正交化，两边都张成整个 R^D 空间，任何两组向量的主角都必然是 0 度（假指标！）。
    因此必须取捕获了主要梯度方差的 top_k 主成分方向（如 top 16，约代表 90%+ 能量）。
    """
    _, _, vh1 = torch.linalg.svd(mat1.float(), full_matrices=False)
    _, _, vh2 = torch.linalg.svd(mat2.float(), full_matrices=False)
    V1 = vh1[:top_k].T  # [D, top_k]
    V2 = vh2[:top_k].T  # [D, top_k]
    m = V1.T @ V2
    s = torch.linalg.svdvals(m)
    s = torch.clamp(s, 0.0, 1.0)
    angles_deg = [float(torch.acos(sigma).item() * 180.0 / 3.141592653589793) for sigma in s]
    return sorted(angles_deg)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", default="checkpoints/base_encoder.pt")
    parser.add_argument("--cards-dir", default="checkpoints/cards")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-batches", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="experiments/rank_budget/e3_results.json")
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[E3] 设备: {device}, 随机种子: {args.seed}")

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
    decoders = {}
    cards = {}
    registered = all_tasks()

    for name in task_names:
        c = read_card(f"{args.cards_dir}/{name}.pt")
        cards[name] = registered[name]
        dec = RobustARSliceDecoder(
            hidden_dim=c["hidden_dim"],
            num_classes=len(c["spec"]["classes"]),
            **c.get("decoder_kwargs", {"num_heads": 4, "num_layers": 2}),
        ).to(device)
        dec.load_state_dict(c["decoder"])
        dec.eval()
        decoders[name] = dec

    # 准备各任务数据
    task_loaders = {}
    for name in task_names:
        card = registered[name]
        raw = card.build_dataset(args.batch_size * args.n_batches * 2)
        random.Random(args.seed).shuffle(raw)
        ds = GenericTaskDataset(raw, tokenizer, card.spec)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        task_loaders[name] = loader

    # 1. 计算每个任务的梯度向量与梯度张量矩阵
    # 梯度对基座输出 doc_memory [B, L, D]
    task_mean_grads = {}       # [D] 平均梯度向量
    task_grad_matrices = {}     # [N_samples, D] 样本级平均梯度矩阵，用于主角计算

    for name in task_names:
        loader = task_loaders[name]
        spec = cards[name].spec
        decoder = decoders[name]

        all_grads_list = []
        batch_count = 0

        for batch in loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            with torch.no_grad():
                doc_mem = doc_encoder(inp, attention_mask=mask)
            doc_mem = doc_mem.detach().requires_grad_(True)

            loss = task_loss(decoder, doc_mem, mask, batch, spec, device)
            grad = torch.autograd.grad(loss, doc_mem)[0]  # [B, L, D]

            # 按 attention_mask 取平均
            m = mask.unsqueeze(-1).float()  # [B, L, 1]
            sample_grad = (grad * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # [B, D]
            all_grads_list.append(sample_grad.detach().cpu())

            batch_count += 1
            if batch_count >= args.n_batches:
                break

        full_mat = torch.cat(all_grads_list, dim=0)  # [N, D]
        mean_vec = full_mat.mean(dim=0)               # [D]
        task_mean_grads[name] = mean_vec
        task_grad_matrices[name] = full_mat
        print(f"  任务 {name:10s} | 收集样本数: {full_mat.shape[0]} | 梯度均值范数: {mean_vec.norm().item():.6f}")

    # 2. 两两余弦与主角计算
    cosine_matrix = {}
    abs_cosine_matrix = {}
    principal_angles = {}

    print("\n[E3] 任务向量两两余弦相似度:")
    header = f"{'':12s}" + "".join(f"{n:>12s}" for n in task_names)
    print(header)

    for n1 in task_names:
        row_cos = {}
        row_abs = {}
        row_pa = {}
        v1 = task_mean_grads[n1]
        v1_norm = v1 / (v1.norm() + 1e-12)
        row_str = f"{n1:12s}"
        for n2 in task_names:
            v2 = task_mean_grads[n2]
            v2_norm = v2 / (v2.norm() + 1e-12)
            cos = float(torch.dot(v1_norm, v2_norm).item())
            row_cos[n2] = cos
            row_abs[n2] = abs(cos)
            row_str += f"{cos:12.4f}"

            # 主角计算 (基于样本级梯度矩阵的 top-16 主成分子空间)
            mat1 = task_grad_matrices[n1]
            mat2 = task_grad_matrices[n2]
            pas = compute_subspace_principal_angles(mat1, mat2, top_k=16)
            row_pa[n2] = {
                "min_angle_deg": pas[0] if pas else 0.0,
                "mean_angle_deg": sum(pas) / len(pas) if pas else 0.0,
                "first_5_angles_deg": pas[:5],
            }
        print(row_str)
        cosine_matrix[n1] = row_cos
        abs_cosine_matrix[n1] = row_abs
        principal_angles[n1] = row_pa

    print("\n[E3] 两两子空间最小主角（度数，越小重合越大，90度完全正交）:")
    for n1 in task_names:
        row_str = f"{n1:12s}"
        for n2 in task_names:
            min_a = principal_angles[n1][n2]["min_angle_deg"]
            row_str += f"{min_a:12.2f}"
        print(row_str)

    results = {
        "cosine_similarity": cosine_matrix,
        "abs_cosine_similarity": abs_cosine_matrix,
        "principal_angles": principal_angles,
        "task_grad_norms": {n: float(task_mean_grads[n].norm().item()) for n in task_names},
    }

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fp:
            json.dump(results, fp, indent=2, ensure_ascii=False)
        print(f"\n[E3] 结果已保存至 {args.out}")


if __name__ == "__main__":
    main()
