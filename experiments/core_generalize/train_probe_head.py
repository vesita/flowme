"""core_generalize 探针：在**冻结核**上训同一个 hold-out 的新头，报 exact_match。

唯一变量 = 核：头、数据、预算、seed 全同（16 ep × 84 = 1344 步、batch 64、lr 1e-3、
AdamW(wd 1e-4)、cosine、clip 1.0）。核 requires_grad=False、eval、前向 no_grad。

    uv run python experiments/core_generalize/train_probe_head.py \
        --core checkpoints/base_encoder.pt --label B0 --holdout idiom --seed 42 \
        --out experiments/core_generalize/probes/B0_idiom_s42.json
"""
from __future__ import annotations

import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))   # 本目录的 prepare.py

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task, task_loss  # noqa: E402

from prepare import CACHE, HOLDOUT_SAMPLES, split_of  # noqa: E402

ENCODER_KWARGS = {
    "num_layers": 3, "num_heads": 4, "max_len": 128, "rope_theta": 10000.0,
    "use_qk_norm": True, "swiglu_scale": 8 / 3,
}
DECODER_KWARGS = {"num_heads": 4, "num_layers": 2}
HIDDEN_DIM = 128
DATA_INFO = HERE / "data_info.json"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="core_generalize 探针头训练（核冻结）")
    ap.add_argument("--core", required=True, help="核 ckpt 路径（含 doc_encoder）")
    ap.add_argument("--label", required=True, help="核臂标签，如 C1_s42 / B0")
    ap.add_argument("--holdout", required=True, choices=tuple(HOLDOUT_SAMPLES))
    ap.add_argument("--seed", type=int, default=42, choices=(42, 43))
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=84)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    seed = args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    n_samples = HOLDOUT_SAMPLES[args.holdout]
    floors = json.loads(DATA_INFO.read_text(encoding="utf-8"))["tasks"][args.holdout]["splits"]
    floor = floors[str(seed)]["floor_exact_empty_predictor"]

    tokenizer = NanoCharTokenizer()
    card = resolve_tasks([args.holdout])[args.holdout]
    spec = card.spec

    # ---- 数据（所有核同一条 eval 集：只由 (holdout, seed) 决定）----
    data = pickle.loads((CACHE / f"{args.holdout}_{n_samples}.pkl").read_bytes())
    vals, train = split_of(data, seed)
    train_loader = DataLoader(GenericTaskDataset(train, tokenizer, spec),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(GenericTaskDataset(vals, tokenizer, spec),
                            batch_size=args.batch_size, shuffle=False)
    print(f"[setup] label={args.label} core={args.core} holdout={args.holdout} seed={seed} "
          f"train={len(train)} val={len(vals)} floor={floor:.4f} device={device}", flush=True)

    # ---- 核：加载 + 冻结 ----
    ck = torch.load(args.core, map_location="cpu", weights_only=False)
    doc_encoder = NanoDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=HIDDEN_DIM,
                                 dropout=0.1, **ENCODER_KWARGS).to(device)
    doc_encoder.load_state_dict(ck["doc_encoder"], strict=True)
    for p in doc_encoder.parameters():
        p.requires_grad_(False)
    doc_encoder.eval()
    core_n = sum(p.numel() for p in doc_encoder.parameters())
    core_train = sum(p.numel() for p in doc_encoder.parameters() if p.requires_grad)
    print(f"SELFTEST_core frozen={core_train == 0} params={core_n:,} "
          f"requires_grad全False={all(not p.requires_grad for p in doc_encoder.parameters())}",
          flush=True)

    # ---- 头：随机初始化（seed 决定）----
    decoder = RobustARSliceDecoder(hidden_dim=HIDDEN_DIM, num_classes=spec.num_classes,
                                   **DECODER_KWARGS).to(device)
    head_train = sum(p.numel() for p in decoder.parameters() if p.requires_grad)
    head_total = sum(p.numel() for p in decoder.parameters())

    # ---- step-0 起跑线 ----
    m0 = evaluate_task(doc_encoder, decoder, val_loader, device, spec)
    print(f"SELFTEST_step0 exact={m0['exact_match']:.4f} cls={m0['cls_acc']:.4f} "
          f"(随机头起跑线) floor={floor:.4f}", flush=True)

    # ---- 训练（只训头）----
    optimizer = torch.optim.AdamW(list(decoder.parameters()), lr=args.lr, weight_decay=1e-4)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    step_times: list[float] = []
    it = iter(train_loader)
    t0 = time.perf_counter()
    n_steps = 0
    hist = []
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
            with torch.no_grad():                       # 核冻结且不存激活
                mem = doc_encoder(inp, attention_mask=mask)
            loss = task_loss(decoder, mem, mask, batch, spec, device, ndb=None)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(decoder.parameters()), 1.0)
            optimizer.step()
            scheduler.step()
            running += float(loss)
            step_times.append(time.perf_counter() - ts)
            n_steps += 1
        hist.append(running / args.steps_per_epoch)
        if epoch % 4 == 0 or epoch == args.epochs or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | loss {hist[-1]:.4f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    steady = sorted(step_times[-200:])
    sec_steady = steady[len(steady) // 2]

    # ---- 评测 ----
    m = evaluate_task(doc_encoder, decoder, val_loader, device, spec)
    exact = m["exact_match"]
    budget = {"epochs": args.epochs, "steps_per_epoch": args.steps_per_epoch,
              "n_steps": n_steps, "batch_size": args.batch_size, "lr": args.lr,
              "n_train": len(train), "n_val": len(vals),
              "sample_draws": n_steps * args.batch_size,
              "core_trainable": core_train, "head_trainable": head_train,
              "head_total": head_total}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rec = {"label": args.label, "core": args.core, "holdout": args.holdout, "seed": seed,
           "metrics": {k: float(v) for k, v in m.items() if isinstance(v, float)},
           "budget": budget, "floor": floor,
           "step0_exact": float(m0["exact_match"]),
           "floor_plus_delta_hint": floor,
           "timing": {"train_sec": train_sec, "sec_per_step": train_sec / max(1, n_steps),
                      "sec_per_step_steady": sec_steady, "peak_mem_mb": peak_mb}}
    out.write_text(json.dumps(rec, ensure_ascii=False, indent=2, default=float),
                   encoding="utf-8")
    print("SELFTEST_budget " + json.dumps(budget, ensure_ascii=False), flush=True)
    print(f"[eval] exact={exact:.4f} cls_acc={m['cls_acc']:.4f} bg_fp={m['bg_fp']:.4f} "
          f"| floor={floor:.4f} | Δfloor={exact - floor:+.4f}", flush=True)
    print("CG_PROBE_METRICS " + json.dumps({
        "label": args.label, "holdout": args.holdout, "seed": seed,
        "exact": exact, "cls_acc": m["cls_acc"], "floor": floor,
        "step0_exact": float(m0["exact_match"]), "budget": budget,
        "train_sec": train_sec, "peak_mem_mb": peak_mb}, ensure_ascii=False), flush=True)
    print(f"[save] {out}")
    print("CG_PROBE_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
