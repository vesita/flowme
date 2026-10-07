"""E-B：在**冻结认知核**上训练「negation 头 + 自己的低秩旁路」（或只训头的 frozen 对照）。

与 `training/train_task_card.py` 的差异（不改它，另写）：
  1. 支持 `--mode bypass`：旁路与头一起训；`--mode frozen`：旁路在场但冻结在零（= 零旁路档）；
  2. 前向不走 `torch.no_grad()`（旁路需要梯度穿过冻结基座的激活）；
  3. 自检三条的原始输出直接打在日志里（SELFTEST_1/2/3）。

用法（详见 PREREG.md）：
    uv run python experiments/additivity/train_bypass_card.py \
        --mode bypass --base checkpoints/base_encoder.pt --seed 42 \
        --out experiments/additivity/cards/negation_bypass_base_s42.pt
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

from bypass import BypassSet, trainable_report  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="冻结基座 + per-module 低秩旁路训练（E-B）")
    ap.add_argument("--mode", choices=("bypass", "frozen"), required=True)
    ap.add_argument("--card", default="negation")
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=84)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=16.0)
    ap.add_argument("--eval-every", type=int, default=0,
                    help=">0 时每 N 个 epoch 在 val 上评估一次（诊断欠训/饱和用，不计入训练耗时）")
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
    print(f"[setup] mode={args.mode} card={args.card} base={args.base} seed={args.seed} device={device}")

    # ---- 2. 解码头（同 seed 同序 ⇒ F/B 两档头初始化逐位相同）----------
    decoder = RobustARSliceDecoder(
        hidden_dim=hidden_dim, num_classes=spec.num_classes, num_heads=4,
        num_layers=args.num_layers, dim_feedforward=args.dim_feedforward,
    ).to(device)

    # ---- 3. 数据（与 N5/R 臂同口径）------------------------------------
    data = card.build_dataset(args.samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val, train = data[:n_val], data[n_val:]
    train_loader = DataLoader(GenericTaskDataset(train, tokenizer, spec),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(GenericTaskDataset(val, tokenizer, spec),
                            batch_size=args.batch_size, shuffle=False)
    print(f"[data] train={len(train)} val={len(val)}")

    # ---- 4. 旁路（独立 generator ⇒ 不扰动头初始化的 RNG 流）-----------
    bypass = BypassSet(doc_encoder, rank=args.rank, alpha=args.alpha,
                       generator=torch.Generator().manual_seed(args.seed + 1000))
    bypass = bypass.to(device)
    init_state = {k: v.detach().cpu().clone() for k, v in bypass.state_dict().items()}

    if args.mode == "frozen":
        for p in bypass.parameters():
            p.requires_grad_(False)          # 旁路在场、冻结在初值（B 恒为精确零）
    bypass.enable()                          # 两档都开闸：frozen = 零旁路配置本身

    # ---- 自检 2(a)：旁路初始为零 ⇒ 编码器输出与「无旁路」逐位一致 ------
    batch_ids = train_loader.dataset[0]["input_ids"][None].to(device)
    batch_msk = train_loader.dataset[0]["attention_mask"][None].to(device)
    with torch.no_grad():
        h_on = doc_encoder(batch_ids, attention_mask=batch_msk)
        bypass.disable()
        h_off = doc_encoder(batch_ids, attention_mask=batch_msk)
        bypass.enable()
        plain, _ = load_base_encoder(args.base, device)   # 全新载入、未挂旁路的对照
        h_plain = plain(batch_ids, attention_mask=batch_msk)
    d_on_off = float((h_on - h_off).abs().max())
    d_on_plain = float((h_on - h_plain).abs().max())
    b_zero = all(float(t["B_std"]) == 0.0 for t in bypass.stats().values())
    print(f"SELFTEST_2a max|Δ|(bypass开@零 vs 关)={d_on_off:.3e} "
          f"max|Δ|(bypass开@零 vs 无旁路基座)={d_on_plain:.3e} B全零={b_zero}")

    # ---- 自检 1：生效后的可训参数量（直接数 requires_grad）-------------
    rep = trainable_report(doc_encoder, decoder, bypass)
    print("SELFTEST_1 trainable_report(mode=%s)=%s" % (args.mode, json.dumps(rep, ensure_ascii=False)))

    # ---- 5. 优化器 / 调度（预算与 N5/R 臂逐项对齐）---------------------
    opt_params = list(decoder.parameters())
    if args.mode == "bypass":
        opt_params += list(bypass.parameters())
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
            mem = doc_encoder(inp, attention_mask=mask)     # 不 no_grad：旁路要梯度
            loss = task_loss(decoder, mem, mask, batch, spec, device, ndb=None)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()                          # 每步同步（与 train_task_card 同构）
            step_times.append(time.perf_counter() - ts)
            n_steps += 1
        if epoch % 4 == 0 or epoch == args.epochs:
            print(f"  Epoch {epoch:3d}/{args.epochs} | loss {running/args.steps_per_epoch:.4f}")
        if args.eval_every and (epoch % args.eval_every == 0 or epoch == args.epochs):
            m = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=None)
            print(f"EPOCH_EVAL {epoch} {json.dumps({k: round(v,4) for k,v in m.items() if isinstance(v,float)}, ensure_ascii=False)}")
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    steady = sorted(step_times[-200:])
    sec_steady = steady[len(steady) // 2]

    # ---- 7. 评估（旁路开闸；frozen 档 = 零旁路）-------------------------
    metrics = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=None)
    show = {k: round(v, 4) for k, v in metrics.items() if isinstance(v, float)}
    print(f"[eval] {json.dumps(show, ensure_ascii=False)}")

    # ---- 自检 3：训练后旁路权重范数/标准差必须偏离初值 ------------------
    fin_state = {k: v.detach().cpu().clone() for k, v in bypass.state_dict().items()}
    drift = {}
    # 逐点算（state_dict 键形如 adapters.0.A）
    for i, site in enumerate(bypass.sites):
        a0 = init_state[f"adapters.{i}.A"]; a1 = fin_state[f"adapters.{i}.A"]
        b0 = init_state[f"adapters.{i}.B"]; b1 = fin_state[f"adapters.{i}.B"]
        drift[site] = {
            "A_max|Δ|": float((a1 - a0).abs().max()),
            "B_std_init": float(b0.std()), "B_std_final": float(b1.std()),
            "B_norm_final": float(b1.norm()), "A_norm_final": float(a1.norm()),
        }
    print("SELFTEST_3 bypass_drift=" + json.dumps(drift, ensure_ascii=False))
    b_moved = all(drift[s]["B_std_final"] > 0 for s in drift)
    print(f"SELFTEST_3_verdict B全部离开零初值={b_moved}")

    # ---- 8. 存卡 --------------------------------------------------------
    save_card(out, decoder, task=args.card, spec=spec, hidden_dim=hidden_dim,
              decoder_kwargs={"num_heads": 4, "num_layers": args.num_layers,
                              "dim_feedforward": args.dim_feedforward},
              base_format=base_ck["format"],
              train_args={"mode": args.mode, "epochs": args.epochs,
                          "steps_per_epoch": args.steps_per_epoch, "lr": args.lr,
                          "samples": args.samples, "base": str(args.base), "seed": args.seed,
                          "rank": args.rank, "alpha": args.alpha, "metrics": metrics},
              extra={"bypass": {"rank": args.rank, "alpha": args.alpha,
                                "sites": bypass.sites,
                                "state_dict": {k: v.cpu() for k, v in fin_state.items()}}})
    print(f"[save] {out} ({out.stat().st_size/1024:.0f} KB)")

    # 机器可读汇总
    print("AB_METRICS " + json.dumps({
        "mode": args.mode, "seed": args.seed, "card": args.card, "base": str(args.base),
        "rank": args.rank, "alpha": args.alpha, "epochs": args.epochs,
        "steps_per_epoch": args.steps_per_epoch, "batch_size": args.batch_size,
        "samples": args.samples, "n_steps": n_steps,
        "metrics": metrics,
        "params": rep,
        "bypass_params": bypass.n_params(),
        "bypass_stats_final": bypass.stats(),
        "train_sec": train_sec, "sec_per_step": train_sec / max(1, n_steps),
        "sec_per_step_steady": sec_steady, "peak_mem_mb": peak_mb,
        "selftest_2a": {"max_delta_gate_on_vs_off": d_on_off,
                        "max_delta_gate_on_vs_plain": d_on_plain, "B_all_zero": b_zero},
        "selftest_3_b_moved": b_moved,
        "out": str(out),
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
