"""方案 A：候选槽位嵌入（candidate slot embedding）—— 自包含实验脚本。

假设（dev-notes/13 §10 方案 A）：在 `128 字扁平窗口 + 单查询解码` 下，类别头与指针头
建立不了「哪个区间属于哪个候选」的绑定；给每个候选区间的 token 加一个可学的「候选编号」
嵌入后，绑定应当变容易。

本脚本**不改任何共享模块**：只在 `experiments/selection_cards/` 下工作。
- 数据：`build_reply_pick_dataset`（`spans` = 0-based 半开区间；引擎侧 anchor 是闭区间，别混）；
- 模型：`NanoDocEncoder` →（+`cand_emb(cand_ids)`）→ `RobustARSliceDecoder`；
- 评测：与训练同前向，`_rollout`/`_truth_of` 直接复用 `dtseek.tasks.runtime`（不重写）。

    uv run python -u experiments/selection_cards/exp_cand_embed.py \
        --mode frozen --cand on --seed 42

    # 测量自检（PREREG §3）：nocand 口径下与 runtime.evaluate_task 逐项相等
    uv run python -u experiments/selection_cards/exp_cand_embed.py --parity
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
from dtseek.tasks.artifacts import load_base_encoder
from dtseek.tasks.builtin.reply_pick.dataset import (
    build_reply_pick_dataset,
)
from dtseek.tasks.builtin.reply_pick.frame import SLOT_OVERHEAD
from dtseek.tasks.plugin import DEFAULT_SEED
from dtseek.tasks.runtime import (
    GenericTaskDataset,
    _rollout,
    _truth_of,
    evaluate_task,
    task_loss,
)

CARD = "reply_pick"
N_SLOTS = 4
#: 候选嵌入档位数：0 = 问句/填充，1..4 = 候选 n（标号段 `|n）` + 正文）
N_CAND_ROWS = N_SLOTS + 1

_TOKENIZER = None


def tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from nano_char_tokenizer import NanoCharTokenizer

        _TOKENIZER = NanoCharTokenizer()
    return _TOKENIZER


def get_spec():
    import dtseek.tasks.builtin.reply_pick  # noqa: F401  import 即注册
    from dtseek.tasks.plugin import resolve_tasks

    return resolve_tasks([CARD])[CARD].spec


# ---------------------------------------------------------------------------
# 输入构造：cand_idx 逐字符（= 逐 token）归档
# ---------------------------------------------------------------------------

def cand_ids_of(text: str, cands: tuple, length: int, spans=()) -> list[int]:
    """逐字符给出候选编号：0 = 问句/填充；n = 第 n 个候选的标号段 `|n）` + 正文。

    fail-closed（PREREG_CAND_EMBED §1）：渲染结果与文本尾部对不上、锚点字面不等于候选、
    或**金标区间没有整段落在同编号档位里**，都直接抛错 —— 绑定标签错 = 整个实验白做。
    """
    from dtseek.tasks.builtin.reply_pick.frame import render_choice_list

    lst, offsets = render_choice_list(list(cands))
    head_len = len(text) - len(lst)
    if head_len < 0 or text[head_len:] != lst:
        raise ValueError(f"候选列表与文本尾部不一致：{text!r}")
    ids = [0] * len(text)
    for n, ((s0, e0), cand) in enumerate(zip(offsets, cands), start=1):
        if text[head_len + s0: head_len + e0] != cand:
            raise ValueError(f"锚点错位（候选 {n}）：期望 {cand!r} 实际 "
                             f"{text[head_len + s0: head_len + e0]!r}")
        # 标号段 `|n）` 紧贴正文之前，归入本候选（PREREG_CAND_EMBED §1）
        start = head_len + s0 - SLOT_OVERHEAD
        for i in range(max(0, start), min(len(text), head_len + e0)):
            ids[i] = n
    for sp in spans:                       # 金标（0-based 半开）必须整段归到 label 那一档
        for i in range(sp["start"], min(sp["end"], len(ids))):
            if ids[i] != sp["label"]:
                raise ValueError(
                    f"候选编号与金标不一致：spans={dict(sp)} 第 {i} 位 cand_idx={ids[i]}"
                    f"（文本 {text!r}）")
    out = ids[:length]
    out += [0] * (length - len(out))                  # 填充位归 0
    return out


class CandTaskDataset(GenericTaskDataset):
    """`GenericTaskDataset` + 每样本的 `cand_ids`（其余张量逐字不变）。"""

    def __getitem__(self, idx: int) -> dict:
        item = self.data[idx]
        out = super().__getitem__(idx)
        cands = item.get("meta", {}).get("cands")
        if cands is None:
            raise ValueError(f"样本缺 meta.cands，无法构造候选编号：{item['text']!r}")
        out["cand_ids"] = torch.tensor(
            cand_ids_of(item["text"], cands, out["input_ids"].shape[0], item.get("spans")),
            dtype=torch.long)
        return out


def check_cand_ids(items: list[dict], spec, n: int = 300) -> dict:
    """绑定标签自检：逐样本构造 cand_ids（内含金标归档 fail-closed），并给一条样例。"""
    ds = CandTaskDataset(items, tokenizer(), spec)
    dist: Counter = Counter()
    for i in range(min(n, len(items))):
        out = ds[i]
        c = out["cand_ids"]
        mask = out["attention_mask"]
        for v, m in zip(c.tolist(), mask.tolist()):
            if m:
                dist[v] += 1
    sample = items[0]
    c0 = ds[0]["cand_ids"].tolist()[: len(sample["text"])]
    return {"n_checked": min(n, len(items)), "token_dist": dict(sorted(dist.items())),
            "sample_text": sample["text"],
            "sample_cand_ids": c0,
            "sample_spans": sample["spans"]}


def forward_memory(enc, inp, mask, cand_emb, cand_ids, use_cand: bool):
    """encoder 输出（+ 候选嵌入）。冻结档下 enc 无 requires_grad ⇒ 不建图。"""
    mem = enc(inp, attention_mask=mask)
    if use_cand:
        mem = mem + cand_emb(cand_ids)
    return mem


# ---------------------------------------------------------------------------
# 评测：与训练同前向；前三项与 runtime.evaluate_task 同口径（累加逻辑逐字复制）
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_cand(enc, decoder, cand_emb, loader, device, spec, *, use_cand: bool,
                  collect: bool = False) -> dict:
    was_training = (enc.training, decoder.training)
    enc.eval()
    decoder.eval()

    cls_ok = cls_tot = span_ok = bg_fired = bg_tot = 0
    exact_n = exact_tot = tp = fp = fn = 0
    joint_ok = joint_tot = anchor_ok = 0
    details: list[dict] = []

    for batch in loader:
        inp = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        cand = batch["cand_ids"].to(device)
        mem = forward_memory(enc, inp, mask, cand_emb, cand, use_cand)
        B = inp.shape[0]

        q0 = decoder.bos_query.expand(B, 1, -1)
        out = decoder.forward_step(q0, mem, doc_mask=mask)
        pred_cls = out["cls_logits"].argmax(-1)
        pred_s = out["start_logits"].argmax(-1)
        pred_e = out["end_logits"].argmax(-1)

        t_labels = batch["labels"][:, 0].to(device)
        t_s = batch["starts"][:, 0].to(device)
        t_e = batch["ends"][:, 0].to(device)
        is_bg = batch["is_bg"].to(device)

        real = t_labels > 0
        cls_ok += ((pred_cls == t_labels) & real).sum().item()
        cls_tot += real.sum().item()
        span_ok += (((pred_s == t_s) & (pred_e == t_e)) & real).sum().item()
        bg_fired += ((pred_cls > 0) & (is_bg > 0.5)).sum().item()
        bg_tot += (is_bg > 0.5).sum().item()

        preds = _rollout(decoder, mem, mask, B, spec)
        for b in range(B):
            truth = _truth_of(batch, b, spec)
            got = preds[b]
            exact_tot += 1
            if sorted(got) == sorted(truth):
                exact_n += 1
            ts, gs = set(truth), set(got)
            tp += len(ts & gs)
            fp += len(gs - ts)
            fn += len(ts - gs)
            if truth:                                   # 有正解样本：锚点与联合口径
                joint_tot += 1
                g_lab, g_s, g_e = truth[0]
                p = got[0] if got else (0, -1, -1)
                span_eq = (p[1], p[2]) == (g_s, g_e)
                if span_eq:
                    anchor_ok += 1
                if p[0] == g_lab and span_eq:
                    joint_ok += 1
            if collect:
                details.append({
                    "is_bg": bool(is_bg[b].item()),
                    "gold_cls": int(t_labels[b]),
                    "pred_cls": int(pred_cls[b]),
                })

    if was_training[0]:
        enc.train()
    if was_training[1]:
        decoder.train()
    rep = {
        "cls_acc": cls_ok / max(1, cls_tot),
        "span_hit": span_ok / max(1, cls_tot),
        "bg_fp": bg_fired / max(1, bg_tot),
        "exact_match": exact_n / max(1, exact_tot),
        "slice_precision": tp / max(1, tp + fp),
        "slice_recall": tp / max(1, tp + fn),
        "n_cls": cls_tot,
        "n_bg": bg_tot,
        "anchor_exact": anchor_ok / max(1, joint_tot),
        "cls_and_span": joint_ok / max(1, joint_tot),
    }
    if collect:
        rep["_details"] = details
    return rep


def detail_stats(rep: dict, items: list[dict]) -> dict:
    """按正解位置分层的类别准确率 + 预测类别分布（诊断，不设阈值）。"""
    det = rep.pop("_details")
    hit: dict[int, list[bool]] = {}
    for d, it in zip(det, items):
        if d["is_bg"]:
            continue
        hit.setdefault(it["meta"]["answer_pos"], []).append(d["pred_cls"] == d["gold_cls"])
    by_pos = {p: round(sum(v) / len(v), 4) for p, v in sorted(hit.items())}
    dist = Counter(d["pred_cls"] for d in det)
    return {"cls_acc_by_answer_pos": by_pos,
            "pred_class_dist": {str(k): v for k, v in sorted(dist.items())}}


# ---------------------------------------------------------------------------
# 测量自检（PREREG §3）：nocand 口径下与 runtime.evaluate_task 逐项相等
# ---------------------------------------------------------------------------

@torch.no_grad()
def parity_check(enc, decoder, cand_emb, items, spec, device, n: int = 200) -> dict:
    subset = items[:n]
    loader = DataLoader(CandTaskDataset(subset, tokenizer(), spec),
                        batch_size=64, shuffle=False)
    mine = evaluate_cand(enc, decoder, cand_emb, loader, device, spec, use_cand=False)

    class _Wrap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.e = enc

        def forward(self, input_ids, attention_mask=None):
            return self.e(input_ids, attention_mask=attention_mask)

    wrap = _Wrap().to(device).eval()          # 起点 eval ⇒ evaluate_task 结束时不会把 enc 切回 train
    theirs = evaluate_task(wrap, decoder, loader, device, spec)
    keys = ["cls_acc", "span_hit", "bg_fp", "exact_match", "slice_precision", "slice_recall"]
    diff = {k: [round(mine[k], 6), round(theirs[k], 6)] for k in keys
            if abs(mine[k] - theirs[k]) > 1e-9}
    return {"n": len(subset), "diff": diff, "ok": not diff,
            "mine": {k: round(mine[k], 6) for k in keys},
            "evaluate_task": {k: round(theirs[k], 6) for k in keys}}


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="方案 A：候选槽位嵌入")
    ap.add_argument("--mode", choices=("frozen", "joint"), default="frozen")
    ap.add_argument("--cand", choices=("on", "off"), default="on",
                    help="on=加候选嵌入（实验臂）；off=同配置不加（唯一有效对照）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=None, help="默认 frozen 12 / joint 16")
    ap.add_argument("--steps-per-epoch", type=int, default=150)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--lr-base", type=float, default=3e-4)
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--eval-samples", type=int, default=1500)
    ap.add_argument("--eval-seed", type=int, default=4242)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--out", default=None, help="产物 .pt 路径")
    ap.add_argument("--metrics-out", default=None)
    ap.add_argument("--parity", action="store_true",
                    help="只做测量自检（不训练）：nocand 口径 vs runtime.evaluate_task")
    ap.add_argument("--check-cand", action="store_true",
                    help="只做绑定标签自检（不训练）：cand_ids 与金标逐位一致")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args(argv)

    use_cand = args.cand == "on"
    epochs = args.epochs if args.epochs is not None else (12 if args.mode == "frozen" else 16)
    tag = f"{args.mode}_{'cand' if use_cand else 'nocand'}_s{args.seed}"
    out = Path(args.out) if args.out else ROOT / "experiments" / "selection_cards" / f"candemb_{tag}.pt"
    mout = Path(args.metrics_out) if args.metrics_out else (
        ROOT / "experiments" / "selection_cards" / f"results_cand_embed_{tag}.json")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    spec = get_spec()
    enc, base_ck = load_base_encoder(args.base, device)
    hidden_dim = base_ck["hidden_dim"]
    joint = args.mode == "joint"
    if joint:
        for p in enc.parameters():
            p.requires_grad_(True)
        enc.train()

    decoder = RobustARSliceDecoder(
        hidden_dim=hidden_dim, num_classes=spec.num_classes, num_heads=4,
        num_layers=2, dim_feedforward=None,
    ).to(device)
    # 两臂都构造候选嵌入（同一 RNG 消耗点、同一初始化）—— 唯一变量是接不接入计算图
    cand_emb = torch.nn.Embedding(N_CAND_ROWS, hidden_dim).to(device)
    torch.nn.init.normal_(cand_emb.weight, mean=0.0, std=0.02)
    if not use_cand:
        for p in cand_emb.parameters():
            p.requires_grad_(False)

    n_dec = sum(p.numel() for p in decoder.parameters())
    n_emb = sum(p.numel() for p in cand_emb.parameters())
    n_enc = sum(p.numel() for p in enc.parameters())
    head_params = list(decoder.parameters()) + (list(cand_emb.parameters()) if use_cand else [])
    n_train = sum(p.numel() for p in head_params) + (n_enc if joint else 0)

    print(f"[{tag}] device={device} hidden={hidden_dim} mode={args.mode} cand={args.cand} "
          f"| 参数：decoder {n_dec:,} cand_emb {n_emb:,} encoder {n_enc:,} 可训 {n_train:,}",
          flush=True)

    # —— 数据：训练折 6000（与 §9 同一份，seed=DEFAULT_SEED）+ 评测折（split=eval）——
    train_data = build_reply_pick_dataset(target_samples=args.samples, seed=DEFAULT_SEED)
    random.Random(args.seed).shuffle(train_data)
    n_val = max(200, len(train_data) // 10)
    val, trn = train_data[:n_val], train_data[n_val:]
    eval_data = build_reply_pick_dataset(target_samples=args.eval_samples, seed=args.eval_seed,
                                         split="eval")
    train_loader = DataLoader(CandTaskDataset(trn, tokenizer(), spec),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(CandTaskDataset(val, tokenizer(), spec),
                            batch_size=args.batch_size, shuffle=False)
    train_eval_loader = DataLoader(CandTaskDataset(trn, tokenizer(), spec),
                                   batch_size=args.batch_size, shuffle=False)
    eval_loader = DataLoader(CandTaskDataset(eval_data, tokenizer(), spec),
                             batch_size=args.batch_size, shuffle=False)
    print(f"[{tag}] 数据：训练折 {len(train_data)}（训 {len(trn)} / 验 {len(val)}）"
          f" | 评测折 {len(eval_data)}（split=eval，与训练逐条不相交）", flush=True)

    if args.check_cand:
        chk = check_cand_ids(eval_data, spec)
        print(f"CHECK_CAND_IDS {json.dumps(chk, ensure_ascii=False)}")
        return 0

    if args.parity:
        got = parity_check(enc, decoder, cand_emb, eval_data, spec, device)
        print(f"PARITY {json.dumps(got, ensure_ascii=False)}")
        return 0 if got["ok"] else 3

    groups = [{"params": head_params, "lr": args.lr_head}]
    if joint:
        groups.insert(0, {"params": list(enc.parameters()), "lr": args.lr_base})
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs * args.steps_per_epoch))

    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    it = iter(train_loader)
    n_steps = 0
    loss_curve: list[float] = []
    for epoch in range(1, epochs + 1):
        decoder.train()
        if joint:
            enc.train()
        running = 0.0
        for _ in range(args.steps_per_epoch):
            try:
                batch = next(it)
            except StopIteration:
                it = iter(train_loader)
                batch = next(it)
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            cand = batch["cand_ids"].to(device)
            mem = forward_memory(enc, inp, mask, cand_emb, cand, use_cand)
            loss = task_loss(decoder, mem, mask, batch, spec, device, ndb=None)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [p for g in optimizer.param_groups for p in g["params"]], 1.0)
            optimizer.step()
            scheduler.step()
            running += loss.item()
            n_steps += 1
        loss_curve.append(running / args.steps_per_epoch)
        if epoch % 4 == 0 or epoch == 1 or epoch == epochs:
            print(f"[{tag}] Epoch {epoch:3d}/{epochs} | loss {running / args.steps_per_epoch:.4f}",
                  flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0
    peak_mb = (torch.cuda.max_memory_allocated() / 2 ** 20) if device.type == "cuda" else 0.0

    # —— 三个集合，同一口径（评测折 = 判据来源）——
    m_train = evaluate_cand(enc, decoder, cand_emb, train_eval_loader, device, spec,
                            use_cand=use_cand, collect=True)
    m_val = evaluate_cand(enc, decoder, cand_emb, val_loader, device, spec, use_cand=use_cand)
    m_eval = evaluate_cand(enc, decoder, cand_emb, eval_loader, device, spec,
                           use_cand=use_cand, collect=True)
    eval_detail = detail_stats(m_eval, eval_data)
    train_detail = detail_stats(m_train, trn)

    # 测量自检（PREREG §3）：nocand 臂用**训好的权重**复算 —— 指标非退化才算有效对照
    parity = None
    if not use_cand:
        parity = parity_check(enc, decoder, cand_emb, eval_data, spec, device)
        print(f"[{tag}] PARITY {json.dumps(parity, ensure_ascii=False)}", flush=True)
        if not parity["ok"]:
            print("!! 与 runtime.evaluate_task 口径不一致 —— 指标不可信，先修测量")
            return 3

    res = {
        "parity": parity,
        "cfg": {"mode": args.mode, "cand": args.cand, "seed": args.seed, "epochs": epochs,
                "steps_per_epoch": args.steps_per_epoch, "n_steps": n_steps,
                "batch_size": args.batch_size, "lr_head": args.lr_head,
                "lr_base": args.lr_base if joint else None, "samples": args.samples,
                "data_seed": DEFAULT_SEED, "eval_seed": args.eval_seed,
                "eval_samples": args.eval_samples, "base": args.base},
        "params": {"decoder": n_dec, "cand_emb": n_emb, "encoder": n_enc,
                   "trainable": n_train},
        "cost": {"train_sec": round(train_sec, 2), "n_steps": n_steps,
                 "sec_per_step": round(train_sec / max(1, n_steps), 4),
                 "peak_mem_mb": round(peak_mb, 1)},
        "loss_curve": [round(x, 4) for x in loss_curve],
        "train_metrics": m_train,
        "val_metrics": m_val,
        "eval_metrics": m_eval,
        "eval_detail": eval_detail,
        "train_detail": train_detail,
    }
    mout.parent.mkdir(parents=True, exist_ok=True)
    mout.write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    if not args.no_save:
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "format": "cand_embed_exp_v1",
            "task": CARD, "cfg": res["cfg"], "params": res["params"],
            "decoder": decoder.state_dict(), "cand_emb": cand_emb.state_dict(),
            "use_cand": use_cand,
            "encoder": enc.state_dict() if joint else None,
            "hidden_dim": hidden_dim,
        }, out)

    def fmt(m: dict) -> str:
        return (f"exact {m['exact_match']:.4f} | bg_fp {m['bg_fp']:.4f} "
                f"| span_hit {m['span_hit']:.4f} | cls {m['cls_acc']:.4f} "
                f"| anchor(条件) {m['anchor_exact']:.4f}")

    print(f"[{tag}] 训练集：{fmt(m_train)}")
    print(f"[{tag}] 训练折 val：{fmt(m_val)}")
    print(f"[{tag}] 评测折  ：{fmt(m_eval)}")
    print(f"[{tag}] 评测折按位置 cls：{eval_detail['cls_acc_by_answer_pos']}"
          f" | 预测分布 {eval_detail['pred_class_dist']}")
    print(f"[{tag}] 代价：train_sec {train_sec:.2f} | {n_steps} 步 "
          f"| {train_sec / max(1, n_steps):.4f} s/step | peak {peak_mb:.1f} MB")
    print(f"[{tag}] EXP_RESULT " + json.dumps({
        "cfg": res["cfg"], "cost": res["cost"], "params": res["params"],
        "train": {k: round(v, 4) for k, v in m_train.items() if isinstance(v, float)},
        "val": {k: round(v, 4) for k, v in m_val.items() if isinstance(v, float)},
        "eval": {k: round(v, 4) for k, v in m_eval.items() if isinstance(v, float)},
        "eval_detail": eval_detail,
        "loss_first_last": [loss_curve[0], loss_curve[-1]],
    }, ensure_ascii=False))
    print(f"[{tag}] 指标已写 {mout}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
