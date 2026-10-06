"""core_ndb 阶段 2：温启动联合训练 —— 核级 mention 门控（臂 C）vs 卡级 NDB（臂 L）。

两臂除「记忆挂哪一层」外逐项同口径（J3 档：核可训、四张老头冻结、negation 新头可训）：

  C = CoreNDB 挂在**核编码**上：每批 encode → 写（本批自己的段）→ 读 → 交给各卡解码；
      读写与"当前挂了哪些卡"无关（一个批次只 encode 一次，5 张卡共用同一份 doc_memory）。
  L = MentionNDB 挂在 **person 头**上：`task_loss(..., ndb=ndb)` 在解码阶段逐层读写，
      每批 `ndb.reset(B, device)`（部署时引擎更是在每段开头 reset ⇒ 逐段 reset）。

判据见同目录 `PREREG_PHASE2.md`（跑前写死）。只写 `experiments/core_ndb/`、
`logs/core_ndb/`、`/tmp`。
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
for p in ("src", "experiments/additivity", "experiments/capability_map",
          "experiments/core_keep", "experiments/core_ndb"):
    sys.path.insert(0, str(ROOT / p))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.mention_ndb import MentionNDB  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.artifacts import read_card  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task, task_loss  # noqa: E402

from prepare import TASK_SAMPLES  # noqa: E402
from train_core_keep import (  # noqa: E402
    DECODER_KWARGS, ENCODER_KWARGS, FROZEN_CARD, HIDDEN_DIM, TASK_ORDER, OLD_CARDS,
    get_raw, split_val,
)
from core_memory import CoreNDB  # noqa: E402

ARMS = ("C", "L")
#: 与 PREREG_PHASE2 §1 一致
NDB_KWARGS = {"levels": (1, 2), "slots": (8192, 4096), "max_table_gb": 0.5}
VOCAB = 8192


class CoreMemEncoder(nn.Module):
    """核编码 + 核级记忆：encode → 写（本批）→ 关写 → 读。每批 reset（表按输入清零）。"""

    def __init__(self, inner: NanoDocEncoder, core: CoreNDB):
        super().__init__()
        self.inner = inner
        self.core = core

    def forward(self, input_ids, attention_mask=None):
        h = self.inner(input_ids, attention_mask=attention_mask)
        self.core.reset(h.shape[0], h.device)
        self.core.open_write()
        self.core.write(h, input_ids, attention_mask)
        self.core.close_write()
        return self.core.read(h, input_ids, attention_mask)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="core_ndb 阶段 2 温启动联合训练")
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=42, choices=(42, 43))
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=84)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-base", type=float, default=3e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    # 记忆门控单独的学习率（不写进 PREREG 的门槛，属实现口径，见 RESULTS「与 PREREG 的出入」）：
    # 3e-4 下实测 400 步 read_scale 只到 0.0024、相对扰动 ~4e-7 ⇒ 判据量级测不出差异，
    # 那是「门没开」不是「记忆无益」。10× 之后门才有机会在 1344 步内真正打开。
    ap.add_argument("--lr-gate", type=float, default=3e-3)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args(argv)

    arm, seed = args.arm, args.seed
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"[setup] arm={arm} seed={seed} device={device} "
          f"steps={args.epochs * args.steps_per_epoch}", flush=True)

    tokenizer = NanoCharTokenizer()
    cards = resolve_tasks(TASK_ORDER)
    resolved = {n: TASK_SAMPLES[n] for n in TASK_ORDER}

    # ---- 数据（只读复用 core_keep 的缓存，含 4 分钟的 sentiment）----
    print("[1/5] 数据 ...", flush=True)
    raw, vals = {}, {}
    for name in TASK_ORDER:
        data = get_raw(name, cards[name])
        vals[name], train = split_val(data, seed)
        raw[name] = train
        print(f"  {name}: n={len(data)} train={len(train)} val={len(vals[name])}", flush=True)
    from probe import split_of as probe_split_of  # noqa: E402
    for name in TASK_ORDER:
        ev, _ = probe_split_of(name, seed)
        if [x["text"] for x in ev] != [x["text"] for x in vals[name]]:
            raise SystemExit(f"ALIGN_CHECK 失败：{name} 与 capability_map 评测集不同，本档作废")
    print("ALIGN_CHECK 5/5 与 capability_map eval_S 逐条相同 ✅", flush=True)

    loaders = {n: DataLoader(GenericTaskDataset(raw[n], tokenizer, cards[n].spec),
                             batch_size=args.batch_size, shuffle=True, drop_last=True)
               for n in TASK_ORDER}
    val_loaders = {n: DataLoader(GenericTaskDataset(vals[n], tokenizer, cards[n].spec),
                                 batch_size=args.batch_size, shuffle=False)
                   for n in TASK_ORDER}

    # ---- 模块 ----
    print("[2/5] 构建核与头 ...", flush=True)
    inner = NanoDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=HIDDEN_DIM,
                           dropout=0.1, **ENCODER_KWARGS).to(device)
    decoders = {n: RobustARSliceDecoder(hidden_dim=HIDDEN_DIM,
                                        num_classes=cards[n].spec.num_classes,
                                        **DECODER_KWARGS).to(device)
                for n in TASK_ORDER}

    mem_enc: nn.Module = inner
    core = None
    ndb = None
    if arm == "C":
        core = CoreNDB(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB, **NDB_KWARGS).to(device)
        # ---- 门控初始化：「记满、不读」----
        # 阶段 2 首跑实测的梯度死锁（2026-10-06 18:12 那次，8 epoch 后两个尺度仍精确为 0）：
        #   write_scale=0 ⇒ 表恒空 ⇒ mass=0 ⇒ ∂loss/∂read_scale = g_raw·mass·(v−h) = 0
        #   read_scale=0  ⇒ g=0 ⇒ read_gate 权重梯度也恒 0
        # ⇒ 两个总闸互为前提，门永远打不开，记忆等于没接。
        # 解法：write_scale=1（表有内容 ⇒ mass>0 ⇒ read_scale 有梯度方向），
        #       read_scale=0（step-0 输出贡献**恰为 0**，零初始化门禁不变）。
        core.write_scale.data.fill_(1.0)
        core.read_scale.data.fill_(0.0)
        mem_enc = CoreMemEncoder(inner, core).to(device)
        print(f"  [C] CoreNDB 就位：slots={core.slots} 表(批 {args.batch_size}) "
              f"{core.table_gb(args.batch_size):.3f}GB 门控参数 "
              f"{sum(p.numel() for p in core.parameters()):,} | "
              f"init write_scale={float(core.write_scale):.1f} "
              f"read_scale={float(core.read_scale):.1f}（记满不读）", flush=True)
    else:
        ndb = MentionNDB(hidden_dim=HIDDEN_DIM, num_classes=cards["person"].spec.num_classes,
                         vocab_size=VOCAB, levels=(1, 2), slots=(8192, 4096),
                         max_table_gb=0.5, read_true=True).to(device)
        print(f"  [L] person 卡 MentionNDB 就位：门控参数 "
              f"{sum(p.numel() for p in ndb.parameters()):,}", flush=True)

    # ---- 温启动（与 core_keep J3 逐字一致）----
    print(f"[3/5] 温启动：核 ← {args.base}，老头 ← {FROZEN_CARD.format('*', seed)}", flush=True)
    bck = torch.load(args.base, map_location="cpu", weights_only=False)
    inner.load_state_dict(bck["doc_encoder"], strict=True)
    for n in OLD_CARDS:
        decoders[n].load_state_dict(read_card(FROZEN_CARD.format(n, seed))["decoder"], strict=True)
    for n in OLD_CARDS:
        for p in decoders[n].parameters():
            p.requires_grad_(False)
    print("  老头冻结，negation 头可训", flush=True)

    # ---- 自检 ----
    def _rep() -> dict:
        core_t = sum(p.numel() for p in inner.parameters())
        core_r = sum(p.numel() for p in inner.parameters() if p.requires_grad)
        heads = {n: {"total": sum(p.numel() for p in decoders[n].parameters()),
                     "trainable": sum(p.numel() for p in decoders[n].parameters()
                                      if p.requires_grad)}
                 for n in TASK_ORDER}
        mem_t = (sum(p.numel() for p in core.parameters()) if core is not None
                 else (sum(p.numel() for p in ndb.parameters()) if ndb is not None else 0))
        mem_r = (sum(p.numel() for p in core.parameters() if p.requires_grad)
                 if core is not None
                 else (sum(p.numel() for p in ndb.parameters() if p.requires_grad)
                       if ndb is not None else 0))
        return {"core_total": core_t, "core_trainable": core_r, "heads": heads,
                "mem_total": mem_t, "mem_trainable": mem_r,
                "trainable_total": core_r + mem_r
                + sum(h["trainable"] for h in heads.values())}

    pre = _rep()
    print("SELFTEST_1 " + json.dumps({"arm": arm, "params": pre}, ensure_ascii=False),
          flush=True)
    for n in OLD_CARDS:
        if pre["heads"][n]["trainable"] != 0:
            raise SystemExit(f"自检失败：老卡 {n} 头仍有可训参数，本档作废")
    if pre["heads"]["negation"]["trainable"] == 0:
        raise SystemExit("自检失败：negation 新头可训参数为 0")
    if pre["mem_trainable"] == 0:
        raise SystemExit("自检失败：记忆门控可训参数为 0")
    if arm == "C":
        ws, rs = float(core.write_scale.detach()), float(core.read_scale.detach())
        if rs != 0.0:
            raise SystemExit(f"自检失败：read_scale 起点必须为 0（step-0 贡献恰为 0），实为 {rs}")
        if ws != 1.0:
            raise SystemExit(f"自检失败：write_scale 起点必须为 1（否则梯度死锁），实为 {ws}")
    print("SELFTEST_1 verdict 老头全冻结 / 新头可训 / 记忆门控可训 ✅", flush=True)

    # ---- 优化器（口径同 train_multitask / core_keep）----
    print("[4/5] 优化器 ...", flush=True)
    mem_params = ([p for p in core.parameters()] if core is not None
                  else [p for p in ndb.parameters()] if ndb is not None else [])
    head_params = [p for n in TASK_ORDER for p in decoders[n].parameters() if p.requires_grad]
    groups = [{"params": list(inner.parameters()), "lr": args.lr_base},
              {"params": mem_params, "lr": args.lr_gate},
              {"params": head_params, "lr": args.lr_head}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    clip_targets = list(inner.parameters()) + mem_params + head_params
    print(f"  lr_base={args.lr_base} lr_gate={args.lr_gate} lr_head={args.lr_head} "
          f"steps={total_steps} "
          f"trainable={pre['trainable_total']:,}", flush=True)

    # ---- 训练 ----
    print("[5/5] 开始训练 ...", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    iters = {n: iter(loaders[n]) for n in TASK_ORDER}
    step_times: list[float] = []
    hist: list[dict] = []
    t_start = time.perf_counter()
    n_steps = 0
    for epoch in range(1, args.epochs + 1):
        mem_enc.train()
        for d in decoders.values():
            d.train()
        if core is not None:
            core.reset_stats()
        ce_running = {n: 0.0 for n in TASK_ORDER}
        for _ in range(args.steps_per_epoch):
            ts = time.perf_counter()
            optimizer.zero_grad()
            total = torch.tensor(0.0, device=device)
            for name in TASK_ORDER:
                try:
                    batch = next(iters[name])
                except StopIteration:
                    iters[name] = iter(loaders[name])
                    batch = next(iters[name])
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                mem = mem_enc(inp, attention_mask=mask)
                ndb_arg = ndb if (arm == "L" and name == "person") else None
                l = task_loss(decoders[name], mem, mask, batch, cards[name].spec,
                              device, ndb=ndb_arg)
                total = total + l
                ce_running[name] += l.item()
            total.backward()
            torch.nn.utils.clip_grad_norm_(clip_targets, 1.0)
            g_ws = (float(core.write_scale.grad) if core is not None
                    and core.write_scale.grad is not None else None)
            g_rs = (float(core.read_scale.grad) if core is not None
                    and core.read_scale.grad is not None else None)
            optimizer.step()
            scheduler.step()
            if core is not None:
                # 投影梯度：两个总闸参数始终留在 [0,1]。
                # 不投影的话 Adam 会把 read_scale 推到负值，而 clamp 在负区梯度为 0 ⇒ 卡死；
                # 投影后参数恒 ≥0，边界处 PyTorch 的 clamp 仍放行梯度 ⇒ 门可开可关。
                core.write_scale.data.clamp_(0.0, 1.0)
                core.read_scale.data.clamp_(0.0, 1.0)
                if n_steps <= 30 or n_steps % 200 == 0:
                    st = core.stats()
                    print(f"  [gate s{n_steps}] write={float(core.write_scale):+.5f} "
                          f"(g={g_ws:+.3e}) read={float(core.read_scale):+.5f} "
                          f"(g={g_rs:+.3e}) w_sum={st['w_sum']:.2f} "
                          f"mass={st['mass_mean']:.4f} d_l1={st['delta_l1']:.4f}",
                          flush=True)
            step_times.append(time.perf_counter() - ts)
            n_steps += 1
        rec = {"epoch": epoch,
               "ce": {n: ce_running[n] / args.steps_per_epoch for n in TASK_ORDER},
               "total": sum(ce_running.values()) / args.steps_per_epoch,
               "lr": [f"{g['lr']:.3e}" for g in optimizer.param_groups]}
        if core is not None:
            rec["mem"] = core.stats()
        elif ndb is not None:
            st = ndb.stats()          # 键名与 CoreNDB.stats() 不同，取可用子集
            rec["mem"] = {k: st[k] for k in ("n_written", "table_gb",
                                             "write_gate_bias", "read_gate_bias",
                                             "retrieval_top1_hit")}
        hist.append(rec)
        if epoch % 4 == 0 or epoch == 1 or epoch == args.epochs:
            mem_s = " ".join(f"{k}={v:.4f}" for k, v in rec["mem"].items()
                             if isinstance(v, float) and v == v)
            print(f"Epoch {epoch:2d}/{args.epochs} | ce=" +
                  " ".join(f"{n}={rec['ce'][n]:.3f}" for n in TASK_ORDER) +
                  f" | total={rec['total']:.3f} | {mem_s}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t_start
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    steady = sorted(step_times[-200:])
    sec_steady = steady[len(steady) // 2]

    # ---- 验证 ----
    print("[eval] 逐任务 val ...", flush=True)
    report = {}
    for name in TASK_ORDER:
        m = evaluate_task(mem_enc, decoders[name], val_loaders[name], device,
                          cards[name].spec,
                          ndb=(ndb if (arm == "L" and name == "person") else None))
        report[name] = m
        print(f"  [{name:9s}] exact={m['exact_match']:.4f} cls_acc={m['cls_acc']:.4f} "
              f"bg_fp={m['bg_fp']:.4f}", flush=True)

    extra = {"arm": arm, "seed": seed, "freeze_old_heads": True,
             "params": pre,
             "timing": {"train_sec": train_sec, "n_steps": n_steps,
                        "sec_per_step": train_sec / max(1, n_steps),
                        "sec_per_step_steady": sec_steady, "peak_mem_mb": peak_mb},
             "history": hist}
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "doc_encoder": inner.state_dict(),
        "decoders": {k: v.state_dict() for k, v in decoders.items()},
        "core_ndb": (core.state_dict() if core is not None else None),
        "core_ndb_kwargs": ({"hidden_dim": HIDDEN_DIM, "vocab_size": VOCAB, **NDB_KWARGS}
                            if core is not None else None),
        "card_ndb": (ndb.state_dict() if ndb is not None else None),
        "card_ndb_kwargs": ({"hidden_dim": HIDDEN_DIM,
                             "num_classes": cards["person"].spec.num_classes,
                             "vocab_size": VOCAB, "levels": (1, 2),
                             "slots": (8192, 4096), "max_table_gb": 0.5,
                             "read_true": True} if ndb is not None else None),
        "hidden_dim": HIDDEN_DIM,
        "encoder": "NanoDocEncoder",
        "encoder_kwargs": ENCODER_KWARGS,
        "decoder_kwargs": DECODER_KWARGS,
        "task_order": TASK_ORDER,
        "task_specs": {n: cards[n].spec.to_snapshot() for n in TASK_ORDER},
        "train_args": {"arm": arm, "seed": seed, "epochs": args.epochs,
                       "steps_per_epoch": args.steps_per_epoch,
                       "batch_size": args.batch_size, "lr_base": args.lr_base,
                       "lr_head": args.lr_head, "lr_gate": args.lr_gate},
        "extra": {"core_ndb_p2": extra},
    }, out)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(
        {"arm": arm, "seed": seed, "metrics": report, "extra": extra},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"[save] {out}", flush=True)
    print("P2_METRICS " + json.dumps({
        "arm": arm, "seed": seed,
        "exact": {n: report[n]["exact_match"] for n in TASK_ORDER},
        "repeat": {n: (report[n].get("repeat_mention_acc")) for n in TASK_ORDER},
        "timing": extra["timing"], "params": pre,
    }, ensure_ascii=False), flush=True)
    print("CORE_NDB_TRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
