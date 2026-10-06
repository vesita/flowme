"""cumulative_add：连续加卡（每步温启动联合）的训练与门禁 —— 口径 = core_keep 的 J3。

三种 mode：

- `gate`    ：步 0 —— 不训练，只建 `base_encoder` + 四张 frozen 老卡，评四卡输出并
              与 `capability_map/summary.json` 的 `frozen_exact`（各自单独跑的记录值）逐位比（G0）。
- `chain`   ：步 i≥1 —— 载入步 i−1 产物温启动（核可训、**已存在卡的头全冻结**、
              新卡头可训），先跑起点门禁 G_i（该步已存在卡的输出必须与上一步结束时逐位相同），
              再训 1344 步，评该步全部卡。
- `control` ：P2 对照 —— 从 `base_encoder` + 四张 frozen 老卡出发，**只加一张卡**，
              同口径 1344 步（`negation` 的对照直接复用 core_keep 的 J3，不重训）。

只写 `experiments/cumulative_add/`、`logs/cumadd_*`、`/tmp`（见 PREREG §5）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "additivity"))
sys.path.insert(0, str(ROOT / "experiments" / "capability_map"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.artifacts import read_card  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import (GenericTaskDataset, _rollout,  # noqa: E402
                                  evaluate_task, task_loss)

from prepare import TASK_SAMPLES as LEGACY_SAMPLES  # noqa: E402
from probe import split_of as legacy_split_of  # noqa: E402

# ---------------- 配置（PREREG §2 写死） ----------------
CARD_ORDER = ["pronoun", "sentiment", "relation", "person", "negation", "idiom", "ownership"]
OLD_CARDS = ["pronoun", "sentiment", "relation", "person"]
#: 步 → 该步在场的卡 / 新加的卡
STEPS = {
    1: {"cards": CARD_ORDER[:5], "new": "negation"},
    2: {"cards": CARD_ORDER[:6], "new": "idiom"},
    3: {"cards": CARD_ORDER[:7], "new": "ownership"},
}
SAMPLES = dict(LEGACY_SAMPLES)
SAMPLES.update({"idiom": 6000, "ownership": 6000})   # PREREG §2

ENCODER_KWARGS = {  # 与 training/train_multitask.py / core_keep 逐字一致
    "num_layers": 3, "num_heads": 4, "max_len": 128, "rope_theta": 10000.0,
    "use_qk_norm": True, "swiglu_scale": 8 / 3,
}
DECODER_KWARGS = {"num_heads": 4, "num_layers": 2}
HIDDEN_DIM = 128
DEFAULT_BASE = "checkpoints/base_encoder.pt"
FROZEN_CARD = "experiments/capability_map/cards/{}_frozen_s{}.pt"
CAPMAP_SUMMARY = ROOT / "experiments" / "capability_map" / "summary.json"

MY_CACHE = HERE / "cache"
SHARED_CACHE = ROOT / "experiments" / "capability_map" / "cache"


# ---------------- 数据（带缓存；老 5 卡只读复用 capability_map 已验证的那一份） ----------------
def get_raw(name: str, card) -> list[dict]:
    n = SAMPLES[name]
    MY_CACHE.mkdir(parents=True, exist_ok=True)
    mine = MY_CACHE / f"{name}_{n}.pkl"
    if mine.exists():
        return pickle.loads(mine.read_bytes())
    src = SHARED_CACHE / f"{name}_{n}_20240927.pkl"      # 老卡：只读复用（含 4 分钟的 sentiment）
    if src.exists():
        mine.write_bytes(src.read_bytes())
        return pickle.loads(src.read_bytes())
    data = card.build_dataset(n)                          # idiom / ownership：本地新构建（<1s）
    mine.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    return data


def split_val(data: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """train_multitask / core_keep 的划分：random.Random(seed).shuffle 后前 max(200, n//10) 是 val。"""
    work = list(data)
    random.Random(seed).shuffle(work)
    n_val = max(200, len(work) // 10)
    return work[:n_val], work[n_val:]


def texts_sha256(samples: list[dict]) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(s["text"].encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


# ---------------- 门禁：输出逐位指纹（PREREG §3） ----------------
def fingerprint(enc, dec, loader, device, spec) -> dict:
    """跑一遍完整自回归发射，返回两枚摘要：
    `pred_sha256` = 每个样本切片序列的 sha256（主判据：输出逐位相同）
    `logit_sha256` = 首步 cls/start/end logits 的 float32 原始字节 sha256（补充观测）
    """
    was = (enc.training, dec.training)
    enc.eval()
    dec.eval()
    hp, hl = hashlib.sha256(), hashlib.sha256()
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            mem = enc(inp, attention_mask=mask)
            B = inp.shape[0]
            q0 = dec.bos_query.expand(B, 1, -1)
            out = dec.forward_step(q0, mem, doc_mask=mask)
            for k in ("cls_logits", "start_logits", "end_logits"):
                hl.update(out[k].detach().float().cpu().numpy().tobytes())
            preds = _rollout(dec, mem, mask, B, spec, ndb=None, input_ids=inp)
            hp.update(repr(preds).encode("utf-8"))
    if was[0]:
        enc.train()
    if was[1]:
        dec.train()
    return {"pred_sha256": hp.hexdigest(), "logit_sha256": hl.hexdigest()}


def eval_all(enc, decs, loaders, device, cards, spec_of) -> dict:
    """每卡一份：全部指标 + 输出指纹。"""
    out = {}
    for n in cards:
        m = evaluate_task(enc, decs[n], loaders[n], device, spec_of[n])
        f = fingerprint(enc, decs[n], loaders[n], device, spec_of[n])
        out[n] = dict(m)
        out[n]["fingerprint"] = f
        print(f"  [{n:10s}] exact={m['exact_match']:.6f} cls_acc={m['cls_acc']:.4f} "
              f"bg_fp={m['bg_fp']:.4f} pred={f['pred_sha256'][:16]} "
              f"logit={f['logit_sha256'][:16]}", flush=True)
    return out


def gate_compare(tag: str, prev: dict, cur: dict, cards: list[str]) -> bool:
    """逐卡逐位比较（PREREG §3）。prev[n] 至少含 fingerprint.pred_sha256 与 exact_match。"""
    ok = True
    for n in cards:
        p, c = prev[n], cur[n]
        pf = p.get("fingerprint", {}).get("pred_sha256")
        cf = c.get("fingerprint", {}).get("pred_sha256")
        pl = p.get("fingerprint", {}).get("logit_sha256")
        cl = c.get("fingerprint", {}).get("logit_sha256")
        same_pred = (pf == cf) if pf else None
        same_logit = (pl == cl) if pl else None
        same_exact = p["exact_match"] == c["exact_match"]
        flag = bool(same_pred) and bool(same_exact)
        ok &= flag
        print(f"SELFTEST_GATE {tag} {n:10s} pred_bitwise={same_pred} "
              f"logit_bitwise={same_logit} exact_equal={same_exact} "
              f"({p['exact_match']:.6f} -> {c['exact_match']:.6f}) "
              f"{'OK' if flag else 'GATE_FAIL'}", flush=True)
    print(f"SELFTEST_GATE {tag} verdict: {'PASS ✅' if ok else 'GATE_FAIL ❌ 该步作废'}", flush=True)
    return ok


def core_drift(core_sd: dict, ref_sd: dict) -> float:
    """‖core − ref‖F / ‖ref‖F（P3）。"""
    num = den = 0.0
    for k in ref_sd:
        a = core_sd[k].detach().float().cpu()
        b = ref_sd[k].detach().float().cpu()
        num += float(((a - b) ** 2).sum())
        den += float((b ** 2).sum())
    return (num ** 0.5) / (den ** 0.5)


# ---------------- 主流程 ----------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="cumulative_add 连续加卡（J3 口径）")
    ap.add_argument("--mode", choices=("gate", "chain", "control"), required=True)
    ap.add_argument("--step", type=int, choices=(0, 1, 2, 3))
    ap.add_argument("--card", choices=("idiom", "ownership", "negation"))
    ap.add_argument("--seed", type=int, default=42, choices=(42, 43))
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=84)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-base", type=float, default=3e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--init", default=None,
                    help="chain 步≥2：上一步产物 .pt；gate/control 省略 = base + 四张 frozen 卡")
    ap.add_argument("--prev-metrics", default=None,
                    help="chain 步≥1 / control：上一步（或步 0 门禁）的 metrics json，用于起点门禁")
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args(argv)

    seed = args.seed
    if args.mode == "gate":
        step, new_card, cards = 0, None, list(OLD_CARDS)
        frozen = list(OLD_CARDS)
    elif args.mode == "control":
        if not args.card:
            raise SystemExit("--mode control 必须给 --card")
        step, new_card = 0, args.card
        cards = OLD_CARDS + [args.card]
        frozen = list(OLD_CARDS)
    else:
        if args.step is None or args.step < 1:
            raise SystemExit("--mode chain 必须给 --step {1,2,3}")
        step = args.step
        new_card = STEPS[step]["new"]
        cards = STEPS[step]["cards"]
        frozen = [c for c in cards if c != new_card]
    trainable_cards = [] if args.mode == "gate" else [new_card]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"[setup] mode={args.mode} step={step} seed={seed} new={new_card} "
          f"cards={cards} frozen={frozen} trainable={trainable_cards} device={device}", flush=True)

    tokenizer = NanoCharTokenizer()
    cards_rt = resolve_tasks(CARD_ORDER)
    spec_of = {n: cards_rt[n].spec for n in CARD_ORDER}

    # ---- 数据 ----
    print("[1/6] 构建任务数据集 ...", flush=True)
    raw, vals = {}, {}
    for name in cards:
        t0 = time.perf_counter()
        data = get_raw(name, cards_rt[name])
        vals[name], train = split_val(data, seed)
        raw[name] = train
        print(f"  {name}: n={len(data)} train={len(train)} val={len(vals[name])} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # 老 5 卡：与 capability_map 的 eval_S 逐条对齐（P1 同一评测集的前提）
    split_ok = True
    for name in cards:
        if name in LEGACY_SAMPLES:
            ev, _tr = legacy_split_of(name, seed)
            same = [x["text"] for x in ev] == [x["text"] for x in vals[name]]
            print(f"ALIGN_CHECK {name} s{seed} eval==val: {same} (n={len(ev)})", flush=True)
            split_ok &= same
        else:
            print(f"ALIGN_CHECK {name} s{seed}: 无外部参照，用 val sha256 记录", flush=True)
    if not split_ok:
        raise SystemExit("SELFTEST_SPLIT 评测集对齐失败 ⇒ 本步作废")

    val_sha = {n: texts_sha256(vals[n]) for n in cards}
    print("SELFTEST_SPLIT " + json.dumps(val_sha), flush=True)

    loaders = {n: DataLoader(GenericTaskDataset(raw[n], tokenizer, spec_of[n]),
                             batch_size=args.batch_size, shuffle=True, drop_last=True)
               for n in cards}
    val_loaders = {n: DataLoader(GenericTaskDataset(vals[n], tokenizer, spec_of[n]),
                                 batch_size=args.batch_size, shuffle=False)
                   for n in cards}

    # ---- 模块（构造顺序固定 CARD_ORDER，保证新卡头初值在各档间可比）----
    print("[2/6] 构建 NanoDocEncoder ...", flush=True)
    doc_encoder = NanoDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=HIDDEN_DIM,
                                 dropout=0.1, **ENCODER_KWARGS).to(device)
    decoders = {n: RobustARSliceDecoder(hidden_dim=HIDDEN_DIM,
                                        num_classes=spec_of[n].num_classes,
                                        **DECODER_KWARGS).to(device)
                for n in CARD_ORDER}

    # ---- 温启动 ----
    print("[3/6] 温启动 ...", flush=True)
    if args.init:
        ck = torch.load(args.init, map_location="cpu", weights_only=False)
        doc_encoder.load_state_dict(ck["doc_encoder"], strict=True)
        for n in frozen:
            decoders[n].load_state_dict(ck["decoders"][n], strict=True)
        print(f"  核 ← {args.init}；已存在卡头 ← 同文件（{len(frozen)} 张）；"
              f"{new_card} 头：随机初始化", flush=True)
        init_src = str(args.init)
    else:
        bck = torch.load(args.base, map_location="cpu", weights_only=False)
        doc_encoder.load_state_dict(bck["doc_encoder"], strict=True)
        for n in OLD_CARDS:
            decoders[n].load_state_dict(read_card(ROOT / FROZEN_CARD.format(n, seed))["decoder"],
                                        strict=True)
        print(f"  核 ← {args.base}；四张老卡头 ← {FROZEN_CARD.format('*', seed)}"
              + (f"；{new_card} 头：随机初始化" if new_card else ""), flush=True)
        init_src = f"{args.base}+frozen_cards"
    base_ck = torch.load(args.base, map_location="cpu", weights_only=False)
    base_sd = base_ck["doc_encoder"]

    # ---- 冻结（J3 口径：已存在卡的头全冻结，只有新卡头 + 核可训）----
    for n in frozen:
        for p in decoders[n].parameters():
            p.requires_grad_(False)
    if new_card is None:
        for p in doc_encoder.parameters():
            p.requires_grad_(False)

    # ---- 自检 1：真实可训参数量 ----
    core_t = sum(p.numel() for p in doc_encoder.parameters())
    core_r = sum(p.numel() for p in doc_encoder.parameters() if p.requires_grad)
    heads = {n: {"total": sum(p.numel() for p in decoders[n].parameters()),
                 "trainable": sum(p.numel() for p in decoders[n].parameters()
                                  if p.requires_grad)}
             for n in CARD_ORDER if n in cards}
    pre = {"core_total": core_t, "core_trainable": core_r, "heads": heads,
           "trainable_total": core_r + sum(h["trainable"] for h in heads.values())}
    print("SELFTEST_1 " + json.dumps({"mode": args.mode, "step": step, "seed": seed,
                                      "init": init_src, "params": pre},
                                     ensure_ascii=False), flush=True)

    # ---- 自检 3：冻结实况 ----
    freeze_ok = True
    for n in cards:
        flags = sorted({p.requires_grad for p in decoders[n].parameters()})
        kind = "FROZEN" if n in frozen else "TRAINABLE"
        want = [False] if n in frozen else [True]
        ok = flags == want
        freeze_ok &= ok
        print(f"SELFTEST_3 {n:10s} requires_grad={flags} expect={want} {kind} "
              f"{'OK' if ok else 'FAIL'}", flush=True)
    if not freeze_ok:
        raise SystemExit("SELFTEST_3 失败：冻结/可训状态与口径不符 ⇒ 本步作废")
    print("SELFTEST_3 verdict: 老头全部 requires_grad=False、新卡头 True ✅", flush=True)

    # ---- 优化器 / 调度 ----
    head_params = [p for n in cards for p in decoders[n].parameters() if p.requires_grad]
    groups, opt_desc = [], []
    if core_r > 0:
        groups.append({"params": list(doc_encoder.parameters()), "lr": args.lr_base})
        opt_desc.append(f"core={core_r:,}@{args.lr_base}")
    if head_params:
        groups.append({"params": head_params, "lr": args.lr_head})
        opt_desc.append(f"heads={sum(p.numel() for p in head_params):,}@{args.lr_head}")
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = None
    if groups:
        optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
        clip_targets = list(doc_encoder.parameters()) + head_params
        n_group_tensors = sum(len(g["params"]) for g in optimizer.param_groups)
        print(f"SELFTEST_OPT groups={len(optimizer.param_groups)} tensors={n_group_tensors} "
              f"({' | '.join(opt_desc)})", flush=True)
        # 冻结头不得出现在优化器里
        opt_ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        leaked = [n for n in frozen
                  for p in decoders[n].parameters() if id(p) in opt_ids]
        if leaked:
            raise SystemExit(f"SELFTEST_OPT 失败：冻结头进了优化器 {set(leaked)} ⇒ 本步作废")
        print("SELFTEST_OPT 冻结头不在优化器内 ✅", flush=True)

    # ---- 起点门禁（只比"该步开始前就已存在"的卡 = frozen）----
    start_report = None
    gate_ok = True
    if args.prev_metrics:
        prev = json.loads(Path(args.prev_metrics).read_text(encoding="utf-8"))
        prev_end = prev["end"]
        missing = [n for n in frozen if n not in prev_end]
        if missing:
            raise SystemExit(f"SELFTEST_GATE 上一步 metrics 缺卡：{missing} ⇒ 本步作废")
        print(f"[4/6] 起点门禁（对照 {args.prev_metrics}）...", flush=True)
        start_report = eval_all(doc_encoder, decoders, val_loaders, device, frozen, spec_of)
        gate_ok = gate_compare(f"step{step}_start", prev_end, start_report, frozen)
        if not gate_ok:
            print("GATE_FAIL 起点门禁未过：本步作废（不训练、不存档）", flush=True)
            return 3
    elif args.mode == "gate":
        # G0：四张老卡基线必须与 capability_map 记录的各自单独跑结果逐位相同
        recorded = json.loads(CAPMAP_SUMMARY.read_text(encoding="utf-8"))["rows"]
        print("[4/6] 起点门禁 G0（vs capability_map frozen_exact）...", flush=True)
        start_report = eval_all(doc_encoder, decoders, val_loaders, device, cards, spec_of)
        gate_ok = True
        for n in OLD_CARDS:
            rec = recorded[n][str(seed)]["frozen_exact"]
            cur = start_report[n]["exact_match"]
            same = cur == rec
            gate_ok &= same
            print(f"SELFTEST_GATE G0 s{seed} {n:10s} 现算={cur:.12f} 记录={rec:.12f} "
                  f"{'OK' if same else 'GATE_FAIL'}", flush=True)
        print(f"SELFTEST_GATE G0 verdict: {'PASS ✅' if gate_ok else 'GATE_FAIL ❌'}", flush=True)
        if not gate_ok:
            return 3
    else:
        print("[4/6] 无起点门禁（control 已带 --prev-metrics 则另走上面分支）", flush=True)

    start_metrics = start_report

    # ---- 训练 ----
    if args.mode == "gate":
        print("[5/6] gate 模式：不训练", flush=True)
        train_sec, n_steps, peak_mb, sec_steady, hist = 0.0, 0, 0.0, 0.0, []
    else:
        print("[5/6] 开始训练 ...", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
        iters = {n: iter(loaders[n]) for n in cards}
        step_times: list[float] = []
        hist = []
        t_start = time.perf_counter()
        n_steps = 0
        for epoch in range(1, args.epochs + 1):
            doc_encoder.train()
            for n in cards:
                decoders[n].train()
            ce_running = {n: 0.0 for n in cards}
            tot_running = 0.0
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
                tot_running += float(total.detach())
                step_times.append(time.perf_counter() - ts)
                n_steps += 1
            ns = args.steps_per_epoch
            rec = {"epoch": epoch,
                   "ce": {n: ce_running[n] / ns for n in cards},
                   "total": tot_running / ns,
                   "lr": [f"{g['lr']:.3e}" for g in optimizer.param_groups]}
            hist.append(rec)
            if epoch % 4 == 0 or epoch == 1 or epoch == args.epochs:
                print(f"Epoch {epoch:2d}/{args.epochs} | ce=" +
                      " ".join(f"{n}={rec['ce'][n]:.3f}" for n in cards) +
                      f" | total={rec['total']:.4f}", flush=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        train_sec = time.perf_counter() - t_start
        peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
        steady = sorted(step_times[-200:])
        sec_steady = steady[len(steady) // 2]

    # ---- 末态评测 ----
    print("[6/6] 末态评测 ...", flush=True)
    end_report = eval_all(doc_encoder, decoders, val_loaders, device, cards, spec_of)

    # ---- 核漂移（P3）----
    drift = {"vs_base_end": core_drift(doc_encoder.state_dict(), base_sd)}
    if args.init:
        drift["vs_prev_end"] = core_drift(doc_encoder.state_dict(), ck["doc_encoder"])
        drift["vs_base_start"] = core_drift(ck["doc_encoder"], base_sd)
    else:
        drift["vs_base_start"] = core_drift(base_sd, base_sd)
        drift["vs_prev_end"] = 0.0
    print(f"SELFTEST_DRIFT step={step} s{seed} d(base→start)={drift['vs_base_start']:.6f} "
          f"d(base→end)={drift['vs_base_end']:.6f} "
          f"d(prev→end)={drift['vs_prev_end']:.6f}", flush=True)

    timing = {"train_sec": train_sec, "n_steps": n_steps,
              "sec_per_step": train_sec / max(1, n_steps),
              "sec_per_step_steady": sec_steady, "peak_mem_mb": peak_mb}

    # ---- 存档 ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "decoders": {k: v.state_dict() for k, v in decoders.items()},
        "hidden_dim": HIDDEN_DIM,
        "encoder": "NanoDocEncoder",
        "encoder_kwargs": ENCODER_KWARGS,
        "decoder_kwargs": DECODER_KWARGS,
        "task_order": cards,
        "task_specs": {n: spec_of[n].to_snapshot() for n in cards},
        "train_args": {"mode": args.mode, "step": step, "seed": seed,
                       "num_epochs": args.epochs, "batch_size": args.batch_size,
                       "steps_per_epoch": args.steps_per_epoch,
                       "lr_base": args.lr_base, "lr_head": args.lr_head,
                       "task_samples": {n: SAMPLES[n] for n in cards},
                       "init": init_src, "frozen": frozen, "trainable_heads": trainable_cards},
        "extra": {"cumulative_add": {"params": pre, "timing": timing, "drift": drift,
                                     "val_sha256": val_sha}},
    }, out)

    met = {
        "mode": args.mode, "step": step, "seed": seed, "cards": cards,
        "new_card": new_card, "frozen": frozen, "trainable_heads": trainable_cards,
        "init": init_src, "prev_metrics": args.prev_metrics,
        "gate": {"ok": gate_ok, "checked": bool(args.prev_metrics) or args.mode == "gate"},
        "start": start_metrics, "end": end_report,
        "params": pre, "timing": timing, "drift": drift,
        "val_sha256": val_sha, "history": hist,
    }
    pm = Path(args.metrics)
    pm.parent.mkdir(parents=True, exist_ok=True)
    pm.write_text(json.dumps(met, ensure_ascii=False, indent=2, default=float),
                  encoding="utf-8")
    print(f"[save] {out}\n[save] {pm}", flush=True)
    print("CUMADD_METRICS " + json.dumps({
        "mode": args.mode, "step": step, "seed": seed,
        "exact": {n: end_report[n]["exact_match"] for n in cards},
        "start_exact": {n: (start_metrics[n]["exact_match"] if start_metrics else None)
                        for n in (start_metrics or {})},
        "params_trainable": pre["trainable_total"],
        "core_trainable": core_r,
        "timing": timing, "drift": drift,
    }, ensure_ascii=False), flush=True)
    print("CUMADD_TRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
