"""E-D：在**冻结认知核**上训练「negation 头 + 自己的并联分支」（或只训头的 frozen 对照）。

与 `experiments/additivity/train_bypass_card.py` 同构（那边只读，不改），差异：
  1. `--branch {mlp,block1,block2}` 三档并联分支（B1 MLP / B2 1 块 / B3 2 块）；
  2. 训练末尾加**关闸评估**（空测试③：关闸后 exact 必须明显下降）；
  3. 空测试③ 打印全部零初始化张量的训后漂移（generic，按 zero_init_keys()）。

用法（详见 PREREG.md）：
    uv run python experiments/core_branch/train_branch_card.py \
        --mode bypass --branch mlp --base checkpoints/base_encoder.pt --seed 42 \
        --out experiments/core_branch/cards/negation_mlp_s42.pt
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.tasks.artifacts import load_base_encoder, save_card  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task, task_loss  # noqa: E402

from branch import BranchSet, trainable_report  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="冻结基座 + 并联分支训练（E-D core_branch）")
    ap.add_argument("--mode", choices=("bypass", "frozen"), required=True,
                    help="bypass=分支+头一起训；frozen=分支在场但冻结在零初值（零分支基线）")
    ap.add_argument("--branch", choices=("mlp", "block1", "block2"), default="mlp")
    ap.add_argument("--card", default="negation")
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=84)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--mlp-hidden", type=int, default=128)
    ap.add_argument("--eval-every", type=int, default=4,
                    help=">0 时每 N 个 epoch 在 val 上评估一次（曲线/平台诊断，不计入步时）")
    ap.add_argument("--num-layers", type=int, default=2)
    ap.add_argument("--dim-feedforward", type=int, default=None)
    args = ap.parse_args(argv)

    card = resolve_tasks([args.card])[args.card]
    spec = card.spec
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = NanoCharTokenizer()

    # ---- 1. 基座（冻结、eval）------------------------------------------
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    doc_encoder, base_ck = load_base_encoder(args.base, device)
    hidden_dim = base_ck["hidden_dim"]
    print(f"[setup] mode={args.mode} branch={args.branch} card={args.card} "
          f"base={args.base} seed={args.seed} device={device}")

    # ---- 2. 解码头（同 seed 同序 ⇒ 与 E-B 各 seed 的头初始化逐位相同）---
    decoder = RobustARSliceDecoder(
        hidden_dim=hidden_dim, num_classes=spec.num_classes, num_heads=4,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
    ).to(device)

    # ---- 3. 数据（与 E-B 同口径）----------------------------------------
    data = card.build_dataset(args.samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val, train = data[:n_val], data[n_val:]
    train_loader = DataLoader(GenericTaskDataset(train, tokenizer, spec),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(GenericTaskDataset(val, tokenizer, spec),
                            batch_size=args.batch_size, shuffle=False)
    print(f"[data] train={len(train)} val={len(val)}")

    # ---- 4. 分支（fork_rng 内构造 ⇒ 不扰动全局 RNG 流，见 branch.py）----
    branch = BranchSet(doc_encoder, kind=args.branch, mlp_hidden=args.mlp_hidden,
                       seed=args.seed)
    branch = branch.to(device)
    init_state = branch.snapshot()

    if args.mode == "frozen":
        for p in branch.parameters():
            p.requires_grad_(False)          # 分支在场、冻结在零初值
    branch.enable()                          # 两档都开闸：frozen = 零分支配置本身

    # ---- 自检 2(a)：分支零初值 ⇒ 编码器输出与「无分支」逐位一致 ----------
    batch_ids = train_loader.dataset[0]["input_ids"][None].to(device)
    batch_msk = train_loader.dataset[0]["attention_mask"][None].to(device)
    with torch.no_grad():
        h_on = doc_encoder(batch_ids, attention_mask=batch_msk)
        branch.disable()
        h_off = doc_encoder(batch_ids, attention_mask=batch_msk)
        branch.enable()
        plain, _ = load_base_encoder(args.base, device)   # 全新载入、未挂分支的对照
        h_plain = plain(batch_ids, attention_mask=batch_msk)
    d_on_off = float((h_on - h_off).abs().max())
    d_on_plain = float((h_on - h_plain).abs().max())
    zero_keys = branch.zero_init_keys()
    z0 = all(float(init_state[k].std()) == 0.0 for k in zero_keys)
    print(f"SELFTEST_2a max|Δ|(分支开@零 vs 关)={d_on_off:.3e} "
          f"max|Δ|(分支开@零 vs 无分支基座)={d_on_plain:.3e} 零初值张量全零={z0} "
          f"zero_keys={zero_keys}")

    # ---- 自检 1：生效后的可训参数量（直接数 requires_grad）---------------
    rep = trainable_report(doc_encoder, decoder, branch)
    print(f"SELFTEST_1 trainable_report(mode={args.mode},branch={args.branch})="
          + json.dumps(rep, ensure_ascii=False))

    # ---- 5. 优化器 / 调度（预算与 E-B 逐项对齐）--------------------------
    opt_params = list(decoder.parameters())
    if args.mode == "bypass":
        opt_params += [p for p in branch.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(opt_params, lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))

    # ---- 6. 训练 --------------------------------------------------------
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    step_times: list[float] = []
    t0 = time.perf_counter()
    it = iter(train_loader)
    n_steps = 0
    for epoch in range(1, args.epochs + 1):
        decoder.train()
        doc_encoder.eval()
        running = 0.0
        for _ in range(args.steps_per_epoch):
            ts = time.perf_counter()
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mem = doc_encoder(inp, attention_mask=mask)     # 不 no_grad：分支要梯度
            loss = task_loss(decoder, mem, mask, batch, spec, device, ndb=None)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()                          # 每步同步（与 E-B 同构）
            step_times.append(time.perf_counter() - ts)
            n_steps += 1
        if epoch % 4 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | loss {running/args.steps_per_epoch:.4f}")
        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
            m = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=None)
            print(f"EPOCH_EVAL {epoch} " + json.dumps(
                {k: round(v, 4) for k, v in m.items() if isinstance(v, float)},
                ensure_ascii=False))
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    steady = sorted(step_times[-200:])
    sec_steady = steady[len(steady) // 2]

    # ---- 7. 评估：开闸（主指标）+ 关闸（空测试③功能性）-------------------
    metrics = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=None)
    show = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
    print(f"[eval_gate_on] {json.dumps(show, ensure_ascii=False)}")

    branch.disable()
    metrics_off = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=None)
    branch.enable()
    show_off = {k: round(v, 4) for k, v in metrics_off.items() if isinstance(v, float)}
    drop = metrics["exact_match"] - metrics_off["exact_match"]
    print(f"[eval_gate_off] {json.dumps(show_off, ensure_ascii=False)}")
    print(f"SELFTEST_3b gate_off_drop exact {metrics['exact_match']:.4f} -> "
          f"{metrics_off['exact_match']:.4f} (Δ={-drop:+.4f}, 需 >=0.10)")

    # ---- 自检 3(a)：训后分支权重必须离开零初值 ---------------------------
    fin_state = branch.snapshot()
    drift = {}
    for k in sorted(init_state):
        a0, a1 = init_state[k], fin_state[k]
        drift[k] = {"init_std": float(a0.std()), "final_std": float(a1.std()),
                    "final_norm": float(a1.norm()), "max_abs_delta": float((a1 - a0).abs().max())}
    moved = [k for k in zero_keys if drift[k]["final_std"] > 0.0]
    n_touched = sum(1 for k in drift if drift[k]["max_abs_delta"] > 0.0)
    print("SELFTEST_3a branch_drift=" + json.dumps(drift, ensure_ascii=False))
    print(f"SELFTEST_3a_verdict 零初值张量全部离开零={len(moved) == len(zero_keys)} "
          f"({len(moved)}/{len(zero_keys)}) 张量发生位移={n_touched}/{len(drift)}")

    # ---- 8. 存卡 --------------------------------------------------------
    save_card(out, decoder, task=args.card, spec=spec, hidden_dim=hidden_dim,
              decoder_kwargs={"num_heads": 4, "num_layers": args.num_layers,
                              "dim_feedforward": args.dim_feedforward},
              base_format=base_ck["format"],
              train_args={"mode": args.mode, "branch": args.branch, "epochs": args.epochs,
                          "steps_per_epoch": args.steps_per_epoch, "lr": args.lr,
                          "samples": args.samples, "base": str(args.base),
                          "seed": args.seed, "mlp_hidden": args.mlp_hidden,
                          "metrics": metrics},
              extra={"branch": {"kind": args.branch, "mlp_hidden": args.mlp_hidden,
                                "n_blocks": len(branch.branch),
                                "sites": branch.sites,
                                "state_dict": {k: v.cpu() for k, v in fin_state.items()}}})
    print(f"[save] {out} ({out.stat().st_size/1024:.0f} KB)")

    # 机器可读汇总
    print("AB_METRICS " + json.dumps({
        "mode": args.mode, "branch": args.branch, "seed": args.seed, "card": args.card,
        "base": str(args.base), "mlp_hidden": args.mlp_hidden, "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size,
        "samples": args.samples, "n_steps": n_steps,
        "metrics": metrics, "metrics_gate_off": metrics_off,
        "gate_off_drop": drop,
        "params": rep, "branch_params": branch.n_params(),
        "branch_stats_final": branch.stats(),
        "train_sec": train_sec, "sec_per_step": train_sec / max(1, n_steps),
        "sec_per_step_steady": sec_steady, "peak_mem_mb": peak_mb,
        "selftest_2a": {"max_delta_gate_on_vs_off": d_on_off,
                        "max_delta_gate_on_vs_plain": d_on_plain, "zero_tensors_all_zero": z0},
        "selftest_3a_zero_keys_moved": len(moved) == len(zero_keys),
        "selftest_3b_gate_off_drop": drop,
        "out": str(out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
