"""P4 用的温启动联合臂：四张老卡 + anchored_sel，同批梯度，**核可训、老头冻结**。

口径逐字克隆 `experiments/cumulative_add/train_cumulative.py --mode control`
（那三个 helper 函数直接 import 复用，不复制实现），只把新卡从 `negation/idiom/ownership`
换成本实验的 `anchored_sel`。

    uv run python experiments/anchored_select/train_joint.py --seed 42 \
        --out experiments/anchored_select/cards/joint_s42.pt \
        --metrics experiments/anchored_select/results/joint_s42.json

判据（PREREG §4 P4）：末态 − 起点 的老卡 Δ ≥ −各自噪声带（pronoun 2.83 / sentiment 0.41 /
relation 1.39 / person 0.33 pt），两 seed 都要过。
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
sys.path.insert(0, str(ROOT / "experiments" / "additivity"))
sys.path.insert(0, str(ROOT / "experiments" / "capability_map"))
sys.path.insert(0, str(ROOT / "experiments" / "cumulative_add"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.artifacts import read_card  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task, task_loss  # noqa: E402

import bands as BANDS  # noqa: E402  experiments/cumulative_add/bands.py（只读复用噪声带）
import train_cumulative as TC  # noqa: E402  复用 get_raw / split_val / fingerprint / eval_all / ...
import anchored_card  # noqa: E402  import 即注册 anchored_sel（本进程内）

NEW_CARD = "anchored_sel"
LEGACY_BAND = dict(BANDS.LEGACY_BAND)      # 单位：比例（0.0283 = 2.83pt）
DATA = HERE / "data"


def my_samples() -> list[dict]:
    rows = [json.loads(l) for l in open(DATA / "train.jsonl", encoding="utf-8")]
    if len(rows) != 8000:
        raise SystemExit(f"训练集应 8000 条，实际 {len(rows)} ⇒ 本步作废")
    n_pos = sum(1 for r in rows if r["label"])
    if n_pos * 2 != len(rows):
        raise SystemExit(f"正负不是 1:1（正 {n_pos} / 共 {len(rows)}）⇒ 本步作废")
    print(f"SELFTEST_DATA anchored_sel n={len(rows)} 正={n_pos} 负={len(rows)-n_pos} "
          f"多数类基线={max(n_pos, len(rows)-n_pos)/len(rows):.4f} 盲猜=0.5000", flush=True)
    return [{"text": r["text"], "spans": r["spans"]} for r in rows]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="温启动联合加卡（control 口径）+ anchored_sel")
    ap.add_argument("--seed", type=int, default=42, choices=(42, 43))
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=84)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-base", type=float, default=3e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--base", default=TC.DEFAULT_BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args(argv)

    seed = args.seed
    cards = list(TC.OLD_CARDS) + [NEW_CARD]
    frozen = list(TC.OLD_CARDS)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"[setup] mode=control seed={seed} cards={cards} frozen={frozen} "
          f"trainable={[NEW_CARD]} core=trainable device={device}", flush=True)

    tokenizer = NanoCharTokenizer()
    cards_rt = resolve_tasks(TC.OLD_CARDS + [NEW_CARD])
    spec_of = {n: cards_rt[n].spec for n in cards}

    # ---- 数据 ----
    print("[1/6] 构建任务数据集 ...", flush=True)
    raw, vals = {}, {}
    for name in TC.OLD_CARDS:
        t0 = time.perf_counter()
        data = TC.get_raw(name, cards_rt[name])
        vals[name], raw[name] = TC.split_val(data, seed)
        ev, _tr = TC.legacy_split_of(name, seed)
        same = [x["text"] for x in ev] == [x["text"] for x in vals[name]]
        print(f"ALIGN_CHECK {name} s{seed} eval==val: {same} (n={len(ev)}, "
              f"{time.perf_counter()-t0:.1f}s)", flush=True)
        if not same:
            raise SystemExit(f"SELFTEST_SPLIT {name} 评测集对齐失败 ⇒ 本步作废")
    mine = my_samples()
    vals[NEW_CARD], raw[NEW_CARD] = TC.split_val(mine, seed)
    print(f"  {NEW_CARD}: n={len(mine)} train={len(raw[NEW_CARD])} val={len(vals[NEW_CARD])}",
          flush=True)
    val_sha = {n: TC.texts_sha256(vals[n]) for n in cards}
    print("SELFTEST_SPLIT " + json.dumps(val_sha), flush=True)

    loaders = {n: DataLoader(GenericTaskDataset(raw[n], tokenizer, spec_of[n]),
                             batch_size=args.batch_size, shuffle=True, drop_last=True)
               for n in cards}
    val_loaders = {n: DataLoader(GenericTaskDataset(vals[n], tokenizer, spec_of[n]),
                                 batch_size=args.batch_size, shuffle=False)
                   for n in cards}

    # ---- 模块 ----
    print("[2/6] 构建 NanoDocEncoder ...", flush=True)
    doc_encoder = NanoDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=TC.HIDDEN_DIM,
                                 dropout=0.1, **TC.ENCODER_KWARGS).to(device)
    decoders = {n: RobustARSliceDecoder(hidden_dim=TC.HIDDEN_DIM,
                                        num_classes=spec_of[n].num_classes,
                                        **TC.DECODER_KWARGS).to(device)
                for n in cards}

    # ---- 温启动 ----
    print("[3/6] 温启动 ...", flush=True)
    bck = torch.load(args.base, map_location="cpu", weights_only=False)
    doc_encoder.load_state_dict(bck["doc_encoder"], strict=True)
    for n in TC.OLD_CARDS:
        decoders[n].load_state_dict(
            read_card(ROOT / TC.FROZEN_CARD.format(n, seed))["decoder"], strict=True)
    print(f"  核 ← {args.base}；四张老卡头 ← {TC.FROZEN_CARD.format('*', seed)}；"
          f"{NEW_CARD} 头：随机初始化", flush=True)
    base_sd = bck["doc_encoder"]

    for n in frozen:
        for p in decoders[n].parameters():
            p.requires_grad_(False)

    # ---- 自检 1 / 3：真实可训参数量与冻结实况 ----
    core_t = sum(p.numel() for p in doc_encoder.parameters())
    core_r = sum(p.numel() for p in doc_encoder.parameters() if p.requires_grad)
    heads = {n: {"total": sum(p.numel() for p in decoders[n].parameters()),
                 "trainable": sum(p.numel() for p in decoders[n].parameters()
                                  if p.requires_grad)} for n in cards}
    pre = {"core_total": core_t, "core_trainable": core_r, "heads": heads,
           "trainable_total": core_r + sum(h["trainable"] for h in heads.values())}
    print("SELFTEST_1 " + json.dumps({"mode": "control_joint", "seed": seed,
                                      "init": f"{args.base}+frozen_cards", "params": pre},
                                     ensure_ascii=False), flush=True)
    freeze_ok = True
    for n in cards:
        flags = sorted({p.requires_grad for p in decoders[n].parameters()})
        want = [False] if n in frozen else [True]
        ok = flags == want
        freeze_ok &= ok
        print(f"SELFTEST_3 {n:14s} requires_grad={flags} expect={want} "
              f"{'FROZEN' if n in frozen else 'TRAINABLE'} {'OK' if ok else 'FAIL'}", flush=True)
    if not freeze_ok or core_r == 0:
        raise SystemExit("SELFTEST_3 失败：冻结/可训状态与口径不符 ⇒ 本步作废")

    # ---- 优化器 ----
    head_params = [p for n in cards for p in decoders[n].parameters() if p.requires_grad]
    groups = [{"params": list(doc_encoder.parameters()), "lr": args.lr_base},
              {"params": head_params, "lr": args.lr_head}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * args.steps_per_epoch))
    opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
    leaked = [n for n in frozen for p in decoders[n].parameters() if id(p) in opt_ids]
    if leaked:
        raise SystemExit(f"SELFTEST_OPT 失败：冻结头进了优化器 {set(leaked)} ⇒ 本步作废")
    print(f"SELFTEST_OPT groups={len(optimizer.param_groups)} 冻结头不在优化器内 ✅", flush=True)

    # ---- 起点门禁 G0（老卡起点必须与 capability_map 记录的 frozen_exact 逐位相同）----
    print("[4/6] 起点门禁 G0 ...", flush=True)
    start_report = TC.eval_all(doc_encoder, decoders, val_loaders, device, TC.OLD_CARDS, spec_of)
    recorded = json.loads(TC.CAPMAP_SUMMARY.read_text(encoding="utf-8"))["rows"]
    gate_ok = True
    for n in TC.OLD_CARDS:
        rec = recorded[n][str(seed)]["frozen_exact"]
        cur = start_report[n]["exact_match"]
        same = cur == rec
        gate_ok &= same
        print(f"SELFTEST_GATE G0 s{seed} {n:10s} 现算={cur:.12f} 记录={rec:.12f} "
              f"{'OK' if same else 'GATE_FAIL'}", flush=True)
    print(f"SELFTEST_GATE G0 verdict: {'PASS ✅' if gate_ok else 'GATE_FAIL ❌'}", flush=True)
    if not gate_ok:
        return 3

    # ---- 训练 ----
    print("[5/6] 开始训练 ...", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    iters = {n: iter(loaders[n]) for n in cards}
    step_times: list[float] = []
    hist = []
    t_start = time.perf_counter()
    n_steps = 0
    clip_targets = list(doc_encoder.parameters()) + head_params
    for epoch in range(1, args.epochs + 1):
        doc_encoder.train()
        for n in cards:
            decoders[n].train()
        ce_running = {n: 0.0 for n in cards}
        for _ in range(args.steps_per_epoch):
            ts = time.perf_counter()
            optimizer.zero_grad()
            total = torch.tensor(0.0, device=device)
            for name in cards:
                try:
                    batch = next(iters[name])
                except StopIteration:
                    iters[name] = iter(loaders[name])
                    batch = next(iters[name])
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                mem = doc_encoder(inp, attention_mask=mask)
                l = task_loss(decoders[name], mem, mask, batch, spec_of[name], device)
                total = total + l
                ce_running[name] += l.item()
            total.backward()
            torch.nn.utils.clip_grad_norm_(clip_targets, 1.0)
            optimizer.step()
            scheduler.step()
            step_times.append(time.perf_counter() - ts)
            n_steps += 1
        ns = args.steps_per_epoch
        rec = {"epoch": epoch, "ce": {n: ce_running[n] / ns for n in cards},
               "lr": [f"{g['lr']:.3e}" for g in optimizer.param_groups]}
        hist.append(rec)
        print(f"Epoch {epoch:2d}/{args.epochs} | {time.strftime('%H:%M:%S')} | ce=" +
              " ".join(f"{n}={rec['ce'][n]:.3f}" for n in cards), flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t_start
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    steady = sorted(step_times[-200:])
    timing = {"train_sec": train_sec, "n_steps": n_steps,
              "sec_per_step": train_sec / max(1, n_steps),
              "sec_per_step_steady": steady[len(steady) // 2], "peak_mem_mb": peak_mb}

    # ---- 末态评测 + P4 ----
    print("[6/6] 末态评测 ...", flush=True)
    end_report = TC.eval_all(doc_encoder, decoders, val_loaders, device, cards, spec_of)
    drift = TC.core_drift(doc_encoder.state_dict(), base_sd)
    print(f"SELFTEST_DRIFT s{seed} d(base→end)={drift:.6f}", flush=True)

    p4 = {}
    print(f"P4 老卡 Δ（末态 − 起点，pt）与噪声带比较：")
    for n in TC.OLD_CARDS:
        d_pt = (end_report[n]["exact_match"] - start_report[n]["exact_match"]) * 100
        band = LEGACY_BAND[n] * 100
        ok = d_pt >= -band
        p4[n] = {"delta_pt": round(d_pt, 4), "band_pt": round(band, 4), "pass": bool(ok)}
        print(f"P4 {n:10s} Δ={d_pt:+.2f}pt 带=−{band:.2f}pt "
              f"{'PASS ✅' if ok else 'FAIL ❌'}  (start={start_report[n]['exact_match']:.4f} "
              f"end={end_report[n]['exact_match']:.4f})", flush=True)
    p4_pass = all(v["pass"] for v in p4.values())
    print(f"P4 verdict: {'PASS ✅' if p4_pass else 'FAIL ❌'}", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "decoders": {k: v.state_dict() for k, v in decoders.items()},
        "hidden_dim": TC.HIDDEN_DIM, "encoder": "NanoDocEncoder",
        "encoder_kwargs": TC.ENCODER_KWARGS, "decoder_kwargs": TC.DECODER_KWARGS,
        "task_order": cards, "task_specs": {n: spec_of[n].to_snapshot() for n in cards},
        "train_args": {"mode": "control_joint", "seed": seed, "new_card": NEW_CARD,
                       "num_epochs": args.epochs, "batch_size": args.batch_size,
                       "steps_per_epoch": args.steps_per_epoch,
                       "lr_base": args.lr_base, "lr_head": args.lr_head,
                       "frozen": frozen, "trainable_heads": [NEW_CARD]},
        "extra": {"anchored_select": {"params": pre, "timing": timing, "drift": drift,
                                      "val_sha256": val_sha}},
    }, out)

    met = {"mode": "control_joint", "seed": seed, "cards": cards, "new_card": NEW_CARD,
           "frozen": frozen, "trainable_heads": [NEW_CARD], "gate_g0": gate_ok,
           "start": start_report, "end": end_report, "params": pre, "timing": timing,
           "drift": drift, "val_sha256": val_sha, "p4": p4, "p4_pass": p4_pass,
           "history": hist}
    pm = Path(args.metrics)
    pm.parent.mkdir(parents=True, exist_ok=True)
    pm.write_text(json.dumps(met, ensure_ascii=False, indent=2, default=float),
                  encoding="utf-8")
    print(f"[save] {out}\n[save] {pm}", flush=True)
    print("JOINT_METRICS " + json.dumps({
        "seed": seed,
        "exact": {n: end_report[n]["exact_match"] for n in cards},
        "start_exact": {n: start_report[n]["exact_match"] for n in TC.OLD_CARDS},
        "p4": p4, "p4_pass": p4_pass,
        "params_trainable": pre["trainable_total"], "core_trainable": core_r,
        "timing": timing, "drift": drift}, ensure_ascii=False), flush=True)
    print("JOINT_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
