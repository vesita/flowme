#!/usr/bin/env python
"""NDB 空壳检验探针（P4）—— **只读包装**：不改 `src/`、不改 `training/`，只加载产物重跑评估。

做三件事（判据 P4 要求两条缺一不可，这里都给）：

  ① 梯度：取一个训练 batch，`task_loss` → backward，逐参数报 `grad` 是否存在且非零；
  ② 旁路：同一份训练好的权重，(a) 整个记忆模块不参与前向（`ndb=None`）、
     (b) 读门控压死（`read_gate.bias = -50`，g≈0），各评估一次，指标必须变化；
  ③ 同构校验：带 NDB 的评估必须复现训练日志里 `AB_METRICS` 的指标
     （复现不了说明这个探针与训练评估不同构，①②的数字一律作废）。

用法：
    uv run python experiments/ndb_recheck/probe_shell.py \
        --card person --base checkpoints/base_encoder.pt \
        --ck /tmp/ndb_recheck/run1/person_ndb_seed42.pt --seed 42 \
        --ref /tmp/ndb_recheck/run1/person_ndb_seed42.log
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.mention_ndb import MentionNDB  # noqa: E402
from dtseek.tasks.artifacts import (  # noqa: E402
    build_card_decoder, check_card_base_compat, load_base_encoder, read_card,
)
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task, task_loss  # noqa: E402

SAMPLES = {"pronoun": 6000, "sentiment": 32000, "relation": 14000,
           "person": 9000, "idiom": 14000, "ownership": 8000}


def ref_metrics(log: Path | None) -> dict | None:
    """从训练日志里抓 AB_METRICS 行的 metrics（同构校验的参照）。"""
    if log is None or not log.exists():
        return None
    for line in log.read_text(encoding="utf-8").splitlines():
        if line.startswith("AB_METRICS "):
            return json.loads(line[len("AB_METRICS "):])["metrics"]
    return None


def grad_report(ndb: MentionNDB, batches: list[dict], loss_fn) -> dict:
    """① 记忆模块每个参数的梯度状态（**多 batch**：单个 batch 撞上零梯度不算证据）。"""
    acc: dict[str, dict] = {}
    losses = []
    for batch in batches:
        ndb.zero_grad(set_to_none=True)
        loss = loss_fn(batch)
        losses.append(float(loss.detach()))
        loss.backward()
        for name, p in ndb.named_parameters():
            st = acc.setdefault(name, {
                "grad_exists": False, "grad_nonzero_batches": 0,
                "grad_abs_max": 0.0, "grad_l2_max": 0.0, "n_params": p.numel()})
            if p.grad is None:
                continue
            st["grad_exists"] = True
            st["grad_abs_max"] = max(st["grad_abs_max"], float(p.grad.abs().max()))
            st["grad_l2_max"] = max(st["grad_l2_max"], float(p.grad.norm()))
            if int((p.grad != 0).sum()) > 0:
                st["grad_nonzero_batches"] += 1
    for st in acc.values():
        st["grad_nonzero_all"] = (st["grad_nonzero_batches"] == len(batches))
    acc["_n_batches"] = {"value": len(batches)}
    acc["_losses"] = {"value": [round(x, 4) for x in losses]}
    return acc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="NDB 空壳检验（梯度 + 旁路 + 同构校验）")
    ap.add_argument("--card", default="person")
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--ck", required=True, help="训练产出的任务卡（带 extra.ndb）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--ref", default=None, help="该次训练的日志（取 AB_METRICS 做同构校验）")
    ap.add_argument("--grad-batches", type=int, default=10, help="梯度检验用几个训练 batch")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args(argv)

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = NanoCharTokenizer()

    card = resolve_tasks([args.card])[args.card]
    spec = card.spec          # 训练口径就是注册表里的 spec（train_task_card.py:80）
    doc_encoder, base_ck = load_base_encoder(args.base, device)
    card_ck = read_card(args.ck)
    check_card_base_compat(card_ck, base_ck, args.ck)
    decoder, snap_spec = build_card_decoder(card_ck, device)
    # ⚠ 已知产物缺口：to_snapshot（plugin.py:219-230）不序列化 identity_labels /
    # annotate_all，from_snapshot（plugin.py:239-249）把它们填回默认 False ——
    # 用快照重建的 spec 跑 evaluate_task 会**静默丢掉** repeat/first/cluster 全部身份指标。
    # 所以本探针的指标口径必须用注册表 spec（与训练一致），并把这条差异记进结果。
    snapshot_spec_gap = {k: [getattr(snap_spec, k), getattr(spec, k)]
                         for k in ("identity_labels", "annotate_all")
                         if getattr(snap_spec, k) != getattr(spec, k)}
    if snapshot_spec_gap:
        print(f"  ⚠ 快照 spec 缺字段（from_snapshot 填了默认值）：{snapshot_spec_gap}")
    print(f"卡 {args.ck} | 基座 {args.base} | 设备 {device}")

    extra = card_ck.get("extra", {})
    if "ndb" not in extra:
        raise SystemExit(f"{args.ck} 没有 extra.ndb，不是 NDB 臂的产物")
    ndb_kwargs = extra["ndb"]["kwargs"]
    ndb = MentionNDB(**ndb_kwargs)
    ndb.load_state_dict({k: v.clone() for k, v in extra["ndb"]["state_dict"].items()})
    ndb = ndb.to(device)
    print(f"  NDB 重建：levels={ndb.levels} slots={ndb.slots} read_true={ndb.read_true} "
          f"| 门控参数 {sum(p.numel() for p in ndb.parameters())}")

    # ---- 数据切分：与 train_task_card.py:117-126 逐行同构 ----
    samples = SAMPLES.get(args.card, 8000)
    data = card.build_dataset(samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    val, _train = data[:n_val], data[n_val:]
    val_loader = DataLoader(GenericTaskDataset(val, tokenizer, spec),
                            batch_size=args.batch_size, shuffle=False)
    train_loader = DataLoader(GenericTaskDataset(_train, tokenizer, spec),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    print(f"  val {len(val)} 条 | samples={samples} | seed={args.seed}")

    # ---- ③ 同构校验：带 NDB 评估必须复现训练日志 ----
    m_on = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=ndb)
    ref = ref_metrics(Path(args.ref) if args.ref else None)
    same_as_ref = None
    if ref is not None:
        keys = sorted(set(ref) | set(m_on))
        diff = {k: [ref.get(k), m_on.get(k)] for k in keys
                if (ref.get(k) is None) != (m_on.get(k) is None)
                or (ref.get(k) is not None and ref[k] != m_on[k])}
        same_as_ref = not diff
        print(f"  同构校验（vs 训练 AB_METRICS）：{'逐位一致 ✅' if same_as_ref else '❌ 不一致 ' + json.dumps(diff)}")

    # ---- ② 旁路：(a) 模块不参与前向 ----
    m_bypass_module = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=None)

    # ---- ② 旁路：(b) 模块还在前向里，但读门控压死（g≈0）----
    bias_backup = ndb.read_gate.bias.data.clone()
    ndb.read_gate.bias.data.fill_(-50.0)
    m_bypass_gate = evaluate_task(doc_encoder, decoder, val_loader, device, spec, ndb=ndb)
    ndb.read_gate.bias.data.copy_(bias_backup)

    # ---- ① 梯度：连续多个训练 batch 上 backward（单 batch 撞上零梯度不算证据）----
    def loss_fn(batch: dict) -> torch.Tensor:
        inp = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            mem = doc_encoder(inp, attention_mask=mask)
        return task_loss(decoder, mem, mask, batch, spec, device, ndb=ndb)

    it = iter(train_loader)
    batches = [next(it) for _ in range(args.grad_batches)]
    decoder.zero_grad(set_to_none=True)
    grads = grad_report(ndb, batches, loss_fn)
    params = {k: v for k, v in grads.items() if not k.startswith("_")}
    n_all = sum(1 for v in params.values() if v["grad_nonzero_all"])
    n_any = sum(1 for v in params.values() if v["grad_nonzero_batches"] > 0)
    print(f"  ① 梯度（{len(batches)} 个 batch）：{n_all}/{len(params)} 个门控参数**每个 batch**都非零，"
          f"{n_any}/{len(params)} 至少一个 batch 非零")
    for k, v in params.items():
        print(f"       {k:<18} 非零 batch {v['grad_nonzero_batches']}/{len(batches)} | "
              f"|g|max={v['grad_abs_max']:.6g} |g|2max={v['grad_l2_max']:.6g}")

    keys = ["repeat_mention_acc", "first_mention_acc", "cluster_f1", "exact_match",
            "id_acc", "cls_acc"]
    print("\n  口径                " + "  ".join(f"{k:>20}" for k in keys))
    for tag, m in (("带 NDB（同构）", m_on), ("旁路·模块不前向", m_bypass_module),
                   ("旁路·读门控压死", m_bypass_gate)):
        print(f"  {tag:<18}" + "  ".join(f"{m.get(k, float('nan')):>20.4f}" for k in keys))

    result = {
        "ck": args.ck, "seed": args.seed, "device": str(device),
        "metrics_ndb_on": m_on, "metrics_bypass_module": m_bypass_module,
        "metrics_bypass_gate": m_bypass_gate,
        "same_as_train_metrics": same_as_ref, "grad": grads,
        "snapshot_spec_gap": snapshot_spec_gap,
        "n_grad_all_batches": n_all, "n_grad_any_batch": n_any, "n_grad_params": len(params),
        "bypass_delta": {k: m_on.get(k, float("nan")) - m_bypass_module.get(k, float("nan"))
                         for k in keys},
        "bypass_gate_delta": {k: m_on.get(k, float("nan")) - m_bypass_gate.get(k, float("nan"))
                              for k in keys},
    }
    print("\nBYPASS_PROBES " + json.dumps(result, ensure_ascii=False))
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
        print(f"  已写 {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
