#!/usr/bin/env python3
"""E1: 基座表征是否坍缩。

测法：
冻结基座，对每张卡的数据取若干 batch 前向，取 embedding / block 0 / block 1 / block 2 / norm(末层) 的隐向量，
对有效 token 的隐向量 (N_tokens, D) 或 (B*L, D) 做 SVD，计算有效秩 erank 与利用率 util = erank / min(M, D)。
同时计算四张卡混合数据下的各层有效秩。

判据：末层 util < 50% => 基座在做低秩压缩，此时再加卡只会更挤。
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

from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.artifacts import read_base  # noqa: E402
from dtseek.tasks.plugin import all_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset  # noqa: E402
from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402


def compute_erank(X: torch.Tensor) -> tuple[float, float]:
    """计算矩阵 X [N, D] 的有效秩 erank 与 util = erank / min(N, D)。"""
    if X.dim() != 2:
        raise ValueError(f"Expected 2D tensor, got shape {X.shape}")
    N, D = X.shape
    s = torch.linalg.svdvals(X.float())
    s = s[s > 1e-9]
    if s.numel() == 0:
        return 0.0, 0.0
    p = s / s.sum()
    er = float(torch.exp(-(p * p.log()).sum()))
    util = er / min(N, D)
    return er, util


def forward_with_layers(doc_encoder: NanoDocEncoder, input_ids: torch.Tensor, attention_mask: torch.Tensor):
    """提取 embedding、每层 block 输出、以及最终 norm 输出。"""
    B, L = input_ids.shape
    h = doc_encoder.embedding(input_ids)
    cos = doc_encoder.rope_cos[:L]
    sin = doc_encoder.rope_sin[:L]
    attn_mask = attention_mask.to(torch.bool).view(B, 1, 1, L) if attention_mask is not None else None

    layer_outputs = {"embedding": h}
    for i, block in enumerate(doc_encoder.blocks):
        h = block(h, cos, sin, attn_mask)
        layer_outputs[f"block_{i}"] = h
    out = doc_encoder.norm(h)
    layer_outputs["output"] = out
    return layer_outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ckpt", default="checkpoints/base_encoder.pt")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--n-batches", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[E1] 设备: {device}, 随机种子: {args.seed}")

    base_info = read_base(args.base_ckpt)
    tokenizer = NanoCharTokenizer()
    doc_encoder = NanoDocEncoder(
        vocab_size=base_info["vocab_size"],
        hidden_dim=base_info["hidden_dim"],
        **base_info["encoder_kwargs"],
    ).to(device)
    doc_encoder.load_state_dict(base_info["doc_encoder"])
    doc_encoder.eval()

    registered = all_tasks()
    task_names = ["pronoun", "sentiment", "relation", "person"]

    # 准备各任务数据
    task_loaders = {}
    for name in task_names:
        card = registered[name]
        raw = card.build_dataset(args.batch_size * args.n_batches * 2)
        random.Random(args.seed).shuffle(raw)
        ds = GenericTaskDataset(raw, tokenizer, card.spec)
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)
        task_loaders[name] = loader

    results = {}
    all_hidden = {k: [] for k in ["embedding", "block_0", "block_1", "block_2", "output"]}

    for name in task_names:
        loader = task_loaders[name]
        layer_hiddens = {k: [] for k in all_hidden}
        collected = 0
        with torch.no_grad():
            for batch in loader:
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                outs = forward_with_layers(doc_encoder, inp, mask)
                # 仅保留有效 token
                valid_mask = mask.bool()
                for k, tensor in outs.items():
                    valid_vecs = tensor[valid_mask].cpu()  # [N_tokens, D]
                    layer_hiddens[k].append(valid_vecs)
                    all_hidden[k].append(valid_vecs)
                collected += 1
                if collected >= args.n_batches:
                    break

        task_res = {}
        for k in layer_hiddens:
            mat = torch.cat(layer_hiddens[k], dim=0)
            er, util = compute_erank(mat)
            task_res[k] = {"n_tokens": mat.shape[0], "dim": mat.shape[1], "erank": er, "util": util}
            print(f"  任务 {name:10s} | {k:10s} | tokens: {mat.shape[0]:5d} | erank: {er:6.2f} / {mat.shape[1]} | util: {util*100:5.1f}%")
        results[name] = task_res

    # 全任务混合
    print("  --- 混合四卡全样本 ---")
    combined_res = {}
    for k in all_hidden:
        mat = torch.cat(all_hidden[k], dim=0)
        er, util = compute_erank(mat)
        combined_res[k] = {"n_tokens": mat.shape[0], "dim": mat.shape[1], "erank": er, "util": util}
        print(f"  {'COMBINED':10s} | {k:10s} | tokens: {mat.shape[0]:5d} | erank: {er:6.2f} / {mat.shape[1]} | util: {util*100:5.1f}%")
    results["combined"] = combined_res

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"[E1] 结果已保存至 {args.out}")


if __name__ == "__main__":
    main()
