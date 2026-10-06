"""core_generalize 核臂训练：C1 / C3 / C5 / C1x5（+ hold-out 可学性预检 PRE）。

口径 = `experiments/core_keep/train_core_keep.py` 的温启动联合训练（**不改它**，本文件是
它在 `experiments/core_generalize/` 里的子集版）：

  1. 核 ← `checkpoints/base_encoder.pt`，有冻结卡的任务头 ←
     `experiments/capability_map/cards/{task}_frozen_s{S}.pt`（hold-out 头随机）；
  2. 温启动后**训练前**先在集内卡上算 step-0 exact，必须与 `capability_map/eval.json`
     的 `frozen` 逐位一致（|Δ| ≤ 1e-6）否则本臂作废（SELFTEST_init）；
  3. 每步对本臂所有任务各抽一个 batch，损失求和，核与头一起训。

只写 `experiments/core_generalize/` 与其 `logs/`。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "capability_map"))
sys.path.insert(0, str(HERE))   # 本目录的 prepare.py 必须盖过 capability_map 的同名模块

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.artifacts import read_card  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task, task_loss  # noqa: E402

from prepare import CACHE, CORE_SAMPLES, HOLDOUT_SAMPLES, split_of  # noqa: E402

ARMS = ("C1", "C3", "C5", "C1x5", "PRE")
ARM_TASKS = {
    "C1": ["sentiment"],
    "C3": ["pronoun", "sentiment", "relation"],
    "C5": ["pronoun", "sentiment", "relation", "person", "negation"],
    "C1x5": ["sentiment"],
}
#: 有 capability_map 冻结卡的 5 张（hold-out 没有 ⇒ 头随机初始化）
FROZEN_CARD = "experiments/capability_map/cards/{}_frozen_s{}.pt"

ENCODER_KWARGS = {
    "num_layers": 3, "num_heads": 4, "max_len": 128, "rope_theta": 10000.0,
    "use_qk_norm": True, "swiglu_scale": 8 / 3,
}
DECODER_KWARGS = {"num_heads": 4, "num_layers": 2}
HIDDEN_DIM = 128
DEFAULT_BASE = "checkpoints/base_encoder.pt"
SAMPLES = {**CORE_SAMPLES, **HOLDOUT_SAMPLES}


def core_drift(path: Path, base_path: str) -> float:
    """‖core − base‖F / ‖base‖F（核漂移）。"""
    import math
    a = torch.load(path, map_location="cpu", weights_only=False)["doc_encoder"]
    b = torch.load(base_path, map_location="cpu", weights_only=False)["doc_encoder"]
    num = math.sqrt(sum(float(((a[k].float() - b[k].float()) ** 2).sum()) for k in a))
    den = math.sqrt(sum(float((b[k].float() ** 2).sum()) for k in b))
    return num / den


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="core_generalize 核臂训练")
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--holdout", default=None, help="PRE 臂的 hold-out 任务名")
    ap.add_argument("--seed", type=int, default=42, choices=(42, 43))
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--steps-per-epoch", type=int, default=84)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-base", type=float, default=3e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--skip-init-check", action="store_true",
                    help="PRE 臂用（hold-out 没有冻结卡，无 step-0 基线可比）")
    args = ap.parse_args(argv)

    seed = args.seed
    if args.arm == "PRE":
        if not args.holdout:
            raise SystemExit("PRE 臂必须给 --holdout")
        tasks = [args.holdout]
        epochs = args.epochs or 16
    else:
        if args.holdout:
            raise SystemExit("只有 PRE 臂吃 --holdout")
        tasks = ARM_TASKS[args.arm]
        epochs = args.epochs or (80 if args.arm == "C1x5" else 16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"[setup] arm={args.arm} holdout={args.holdout} seed={seed} tasks={tasks} "
          f"epochs={epochs}x{args.steps_per_epoch} bs={args.batch_size} device={device}",
          flush=True)

    tokenizer = NanoCharTokenizer()
    cards = resolve_tasks(tasks)

    # ---- 数据 ----
    print("[1/5] 数据 ...", flush=True)
    raw, vals = {}, {}
    for name in tasks:
        n = SAMPLES[name]
        data = pickle_load(CACHE / f"{name}_{n}.pkl")
        vals[name], train = split_of(data, seed)
        raw[name] = train
        print(f"  {name}: n={len(data)} train={len(train)} val={len(vals[name])}", flush=True)

    # ALIGN 自检：我的 val 必须与 capability_map 的 eval_S 逐条相同（step-0 对账的前提）。
    # capability_map 的 split（probe.split_of）= 读 ordered pkl → shuffle(seed) → 前 max(200,n//10)；
    # 这里**只读复刻**它（不 import 它的模块：两边都有同名 prepare.py，谁在前谁劫持）。
    import copy
    capmap_cache = ROOT / "experiments" / "capability_map" / "cache"
    for name in tasks:
        if name not in CORE_SAMPLES:
            continue
        ordered = pickle_load(capmap_cache / f"{name}_ordered_s{seed}.pkl")
        work = copy.deepcopy(ordered)
        random.Random(seed).shuffle(work)
        ev = work[:max(200, len(work) // 10)]
        same = [x["text"] for x in ev] == [x["text"] for x in vals[name]]
        print(f"ALIGN_CHECK {name} s{seed} eval==val: {same} (n={len(ev)})", flush=True)
        if not same:
            raise SystemExit(f"评测集对齐失败：{name} ⇒ 本臂作废")

    loaders = {n: DataLoader(GenericTaskDataset(raw[n], tokenizer, cards[n].spec),
                             batch_size=args.batch_size, shuffle=True, drop_last=True)
               for n in tasks}
    val_loaders = {n: DataLoader(GenericTaskDataset(vals[n], tokenizer, cards[n].spec),
                                 batch_size=args.batch_size, shuffle=False)
                   for n in tasks}

    # ---- 模块 ----
    print("[2/5] 构建 NanoDocEncoder ...", flush=True)
    doc_encoder = NanoDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=HIDDEN_DIM,
                                 dropout=0.1, **ENCODER_KWARGS).to(device)
    decoders = {n: RobustARSliceDecoder(hidden_dim=HIDDEN_DIM,
                                        num_classes=cards[n].spec.num_classes,
                                        **DECODER_KWARGS).to(device)
                for n in tasks}

    # ---- 温启动 ----
    print(f"[3/5] 温启动：核 ← {args.base}", flush=True)
    bck = torch.load(args.base, map_location="cpu", weights_only=False)
    doc_encoder.load_state_dict(bck["doc_encoder"], strict=True)
    warm_heads = []
    for n in tasks:
        p = Path(FROZEN_CARD.format(n, seed))
        if p.exists():
            decoders[n].load_state_dict(read_card(p)["decoder"], strict=True)
            warm_heads.append(n)
            print(f"  头 {n} ← {p}", flush=True)
        else:
            print(f"  头 {n}：随机初始化（无冻结卡）", flush=True)

    # ---- 自检 1a：真实可训参数量 ----
    core_t = sum(p.numel() for p in doc_encoder.parameters())
    core_r = sum(p.numel() for p in doc_encoder.parameters() if p.requires_grad)
    heads = {n: {"total": sum(p.numel() for p in decoders[n].parameters()),
                 "trainable": sum(p.numel() for p in decoders[n].parameters()
                                  if p.requires_grad)} for n in tasks}
    params = {"core_total": core_t, "core_trainable": core_r, "heads": heads,
              "trainable_total": core_r + sum(h["trainable"] for h in heads.values())}
    print("SELFTEST_init params=" + json.dumps(params, ensure_ascii=False), flush=True)

    # ---- 自检 1b：温启动后 step-0 行为必须与 capability_map 冻结基线逐位一致。
    #      只对**真正从冻结卡载入**的头对账：negation 在 capability_map 里没有冻结卡
    #      （本实验与 core_keep 一样给它随机头），拿它的 frozen 基线比是错的口径。 ----
    init0 = {}
    n_checked = 0
    print("[4/5] step-0 对账 ...", flush=True)
    for name in tasks:
        m = evaluate_task(doc_encoder, decoders[name], val_loaders[name],
                          device, cards[name].spec)
        init0[name] = {k: float(v) for k, v in m.items() if isinstance(v, float)}
        line = f"  [{name}] step0 exact={m['exact_match']:.6f} cls={m['cls_acc']:.6f}"
        if (not args.skip_init_check) and name in warm_heads:
            ref = json.loads((ROOT / "experiments" / "capability_map" / "eval.json")
                             .read_text(encoding="utf-8"))["caps"][name][str(seed)]["frozen"]
            d = abs(m["exact_match"] - ref["exact_match"])
            line += f" | frozen基线={ref['exact_match']:.6f} Δ={d:.3e}"
            print(line, flush=True)
            n_checked += 1
            if d > 1e-6:
                raise SystemExit(f"SELFTEST_init 失败：{name} step-0 与冻结基线差 {d:.3e} ⇒ 本臂作废")
        else:
            why = "随机头（无冻结卡），不参与 step-0 对账" if name not in warm_heads \
                else "hold-out 随机头"
            print(line + f" | {why}", flush=True)
    if args.skip_init_check:
        print("SELFTEST_init verdict: PRE 臂无冻结基线，仅记录 step-0", flush=True)
    else:
        print(f"SELFTEST_init verdict: {n_checked} 张温启动头 step-0 与冻结基线逐位一致 ✅"
              f"（随机头 {len(tasks) - n_checked} 张不参与对账）", flush=True)

    # ---- 优化器 ----
    print("[5/5] 训练 ...", flush=True)
    head_params = [p for n in tasks for p in decoders[n].parameters() if p.requires_grad]
    groups = [{"params": list(doc_encoder.parameters()), "lr": args.lr_base},
              {"params": head_params, "lr": args.lr_head}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    total_steps = epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    clip_targets = list(doc_encoder.parameters()) + head_params

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    iters = {n: iter(loaders[n]) for n in tasks}
    step_times: list[float] = []
    hist: list[dict] = []
    t0 = time.perf_counter()
    n_steps = 0
    for epoch in range(1, epochs + 1):
        doc_encoder.train()
        for d in decoders.values():
            d.train()
        running = {n: 0.0 for n in tasks}
        for _ in range(args.steps_per_epoch):
            ts = time.perf_counter()
            optimizer.zero_grad()
            total = torch.tensor(0.0, device=device)
            for name in tasks:
                try:
                    batch = next(iters[name])
                except StopIteration:
                    iters[name] = iter(loaders[name])
                    batch = next(iters[name])
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                mem = doc_encoder(inp, attention_mask=mask)
                l = task_loss(decoders[name], mem, mask, batch, cards[name].spec, device)
                total = total + l
                running[name] += float(l)
            total.backward()
            torch.nn.utils.clip_grad_norm_(clip_targets, 1.0)
            optimizer.step()
            scheduler.step()
            step_times.append(time.perf_counter() - ts)
            n_steps += 1
        ns = args.steps_per_epoch
        hist.append({"epoch": epoch,
                     "ce": {n: running[n] / ns for n in tasks},
                     "lr": [f"{g['lr']:.3e}" for g in optimizer.param_groups]})
        if epoch % 4 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch {epoch:3d}/{epochs} | ce=" +
                  " ".join(f"{n}={hist[-1]['ce'][n]:.3f}" for n in tasks), flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    steady = sorted(step_times[-200:])
    sec_steady = steady[len(steady) // 2]

    # ---- 集内卡评测（健康检查）----
    print("[eval] 集内卡 ...", flush=True)
    report = {}
    for name in tasks:
        m = evaluate_task(doc_encoder, decoders[name], val_loaders[name],
                          device, cards[name].spec)
        report[name] = m
        print(f"  [{name:9s}] exact={m['exact_match']:.4f} cls_acc={m['cls_acc']:.4f} "
              f"bg_fp={m['bg_fp']:.4f}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "decoders": {k: v.state_dict() for k, v in decoders.items()},
        "hidden_dim": HIDDEN_DIM,
        "encoder": "NanoDocEncoder",
        "encoder_kwargs": ENCODER_KWARGS,
        "decoder_kwargs": DECODER_KWARGS,
        "task_order": tasks,
        "task_specs": {n: cards[n].spec.to_snapshot() for n in tasks},
        "train_args": {"arm": args.arm, "holdout": args.holdout, "seed": seed,
                       "epochs": epochs, "steps_per_epoch": args.steps_per_epoch,
                       "batch_size": args.batch_size, "lr_base": args.lr_base,
                       "lr_head": args.lr_head, "samples": {n: SAMPLES[n] for n in tasks}},
    }, out)
    drift = core_drift(out, args.base)
    extra = {"arm": args.arm, "holdout": args.holdout, "seed": seed, "tasks": tasks,
             "params": params, "init_step0": init0, "warm_heads": warm_heads,
             "core_drift": drift,
             "timing": {"train_sec": train_sec, "n_steps": n_steps,
                        "sec_per_step": train_sec / max(1, n_steps),
                        "sec_per_step_steady": sec_steady, "peak_mem_mb": peak_mb},
             "history": hist}
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(
        {"arm": args.arm, "holdout": args.holdout, "seed": seed,
         "metrics": {n: {k: float(v) for k, v in report[n].items() if isinstance(v, float)}
                     for n in tasks},
         "extra": extra}, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    print("CG_CORE_METRICS " + json.dumps({
        "arm": args.arm, "holdout": args.holdout, "seed": seed,
        "exact": {n: report[n]["exact_match"] for n in tasks},
        "params": params, "core_drift": drift, "n_steps": n_steps,
        "train_sec": train_sec, "peak_mem_mb": peak_mb,
    }, ensure_ascii=False), flush=True)
    print("CG_CORE_DONE", flush=True)
    return 0


def pickle_load(p: Path):
    import pickle
    return pickle.loads(p.read_bytes())


if __name__ == "__main__":
    sys.exit(main())
