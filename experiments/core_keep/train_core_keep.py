"""core_keep：四档「改核 + 保持老卡行为」联合训练（J0/J1/J2/J3 + 保真档 F）。

与 `training/train_multitask.py` 的关系：**不改它**，本文件是它的联合训练口径在
`experiments/core_keep/` 里的复刻，只多出三件事 ——

  1. **温启动**：核 ← `checkpoints/base_encoder.pt`，四张老卡头 ←
     `experiments/capability_map/cards/{cap}_frozen_s{S}.pt`，negation 头随机；
     于是 step 0 的老卡行为精确等于 P1 参照基线（Δ=0），之后的变化全是"漂移"。
  2. **LwF 蒸馏**（J1/J2）：老卡数据上，student（新核+新头）与 teacher（冻结旧核+冻结旧头）
     在同一条教师强制轨迹上的四个输出分布的 KL。
  3. **老卡 replay**（J2）与 **冻结老卡头**（J3）。

判据见 `PREREG.md`（跑前写死）。只写 `experiments/core_keep/`、`logs/corekeep_*`、`/tmp`。
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
sys.path.insert(0, str(ROOT / "experiments" / "additivity"))
sys.path.insert(0, str(ROOT / "experiments" / "capability_map"))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.artifacts import build_card_decoder, load_base_encoder, read_card  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task, task_loss  # noqa: E402

from prepare import CACHE as CAPMAP_CACHE, TASK_SAMPLES  # noqa: E402

#: 与 checkpoints/arm_neg5_seed{42,43}.pt 的 task_order 逐字一致
TASK_ORDER = ["pronoun", "sentiment", "relation", "person", "negation"]
OLD_CARDS = ["pronoun", "sentiment", "relation", "person"]
ARMS = ("J0", "J1", "J2", "J3", "F")

# 与 training/train_multitask.py 逐字一致
ENCODER_KWARGS = {
    "num_layers": 3, "num_heads": 4, "max_len": 128, "rope_theta": 10000.0,
    "use_qk_norm": True, "swiglu_scale": 8 / 3,
}
DECODER_KWARGS = {"num_heads": 4, "num_layers": 2}
HIDDEN_DIM = 128
DEFAULT_BASE = "checkpoints/base_encoder.pt"
FROZEN_CARD = "experiments/capability_map/cards/{}_frozen_s{}.pt"

MY_CACHE = HERE / "cache"


# ---------------- 数据（带缓存，口径与 train_multitask 逐项一致） ----------------
def get_raw(name: str, card) -> list[dict]:
    n = TASK_SAMPLES[name]
    MY_CACHE.mkdir(parents=True, exist_ok=True)
    mine = MY_CACHE / f"{name}_{n}.pkl"
    if mine.exists():
        return pickle.loads(mine.read_bytes())
    # capability_map 已经把这 5 份数据建好并验过（含 4 分钟的 sentiment），只读复用
    src = CAPMAP_CACHE / f"{name}_{n}_20240927.pkl"
    if src.exists():
        mine.write_bytes(src.read_bytes())          # 复用已验证的同一份数据（只读）
        return pickle.loads(src.read_bytes())
    data = card.build_dataset(n)
    mine.write_bytes(pickle.dumps(data, protocol=pickle.HIGHEST_PROTOCOL))
    return data


def split_val(data: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    """train_multitask 的划分：random.Random(seed).shuffle 后前 max(200, n//10) 是 val。"""
    work = list(data)
    random.Random(seed).shuffle(work)
    n_val = max(200, len(work) // 10)
    return work[:n_val], work[n_val:]


# ---------------- 蒸馏 ----------------
def _kl(student: torch.Tensor, teacher: torch.Tensor, T: float) -> torch.Tensor:
    """逐样本 KL(teacher || student) @ 温度 T，带 T² 缩放。返回 [B]。"""
    s = F.log_softmax(student / T, dim=-1)
    t = F.softmax(teacher / T, dim=-1)
    return F.kl_div(s, t, reduction="none").sum(-1) * (T * T)


def kd_task_loss(decoder, doc_memory, mask, batch, spec, device,
                 t_decoder, t_doc_memory, T: float):
    """`task_loss` 的逐行复刻 + 同一条教师强制轨迹上的 LwF KL。

    返回 (CE, KD)：CE 与 `task_loss` 同口径（逐步 mean-over-batch 再沿步求和），
    KD 结构与 CE 相同 ⇒ λ_kd=1 时两者可直接同权相加。
    """
    max_steps = spec.max_steps
    t_labels = batch["labels"].to(device)
    t_starts = batch["starts"].to(device)
    t_ends = batch["ends"].to(device)
    t_nstarts = batch["norm_starts"].to(device)
    t_nends = batch["norm_ends"].to(device)
    t_actions = batch["actions"].to(device)
    step_mask = batch["step_mask"].to(device)
    B = doc_memory.shape[0]

    cls_w = torch.tensor(spec.cls_weights(), device=device)
    act_w = torch.tensor(list(spec.action_weight), device=device)

    q_seq = decoder.bos_query.expand(B, 1, -1)
    t_q = t_decoder.bos_query.expand(B, 1, -1)
    loss = torch.tensor(0.0, device=device)
    kd = torch.tensor(0.0, device=device)

    for s in range(max_steps):
        step_out = decoder.forward_step(q_seq, doc_memory, doc_mask=mask)
        cls_logits = step_out["cls_logits"]
        m = step_mask[:, s]
        if m.sum() > 0:
            l_cls = (F.cross_entropy(cls_logits, t_labels[:, s],
                                     weight=cls_w, reduction="none") * m).sum() / m.sum()
            l_s = (F.cross_entropy(step_out["start_logits"], t_starts[:, s],
                                   reduction="none") * m).sum() / m.sum()
            l_e = (F.cross_entropy(step_out["end_logits"], t_ends[:, s],
                                   reduction="none") * m).sum() / m.sum()
            l_act = (F.cross_entropy(step_out["action_logits"], t_actions[:, s],
                                     weight=act_w, reduction="none") * m).sum() / m.sum()
            loss = loss + (l_cls + spec.span_weight * l_s + spec.span_weight * l_e + l_act)

            with torch.no_grad():
                t_out = t_decoder.forward_step(t_q, t_doc_memory, doc_mask=mask)
            kl = ( _kl(step_out["cls_logits"], t_out["cls_logits"], T)
                 + _kl(step_out["start_logits"], t_out["start_logits"], T)
                 + _kl(step_out["end_logits"], t_out["end_logits"], T)
                 + _kl(step_out["action_logits"], t_out["action_logits"], T) )
            kd = kd + (kl * m).sum() / m.sum()
        else:
            with torch.no_grad():
                t_out = t_decoder.forward_step(t_q, t_doc_memory, doc_mask=mask)

        next_q = decoder.get_step_input(
            prev_hidden=step_out["last_hidden"],
            prev_cls=t_labels[:, s:s + 1],
            prev_start=t_nstarts[:, s:s + 1, None],
            prev_end=t_nends[:, s:s + 1, None],
        )
        q_seq = torch.cat([q_seq, next_q], dim=1)
        t_next = t_decoder.get_step_input(
            prev_hidden=t_out["last_hidden"],
            prev_cls=t_labels[:, s:s + 1],
            prev_start=t_nstarts[:, s:s + 1, None],
            prev_end=t_nends[:, s:s + 1, None],
        )
        t_q = torch.cat([t_q, t_next], dim=1)

    return loss, kd


# ---------------- 主流程 ----------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="core_keep 四档联合训练")
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--seed", type=int, default=42, choices=(42, 43))
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--steps-per-epoch", type=int, default=84)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr-base", type=float, default=3e-4)
    ap.add_argument("--lr-head", type=float, default=1e-3)
    ap.add_argument("--kd-weight", type=float, default=1.0, help="λ_kd（J1/J2）")
    ap.add_argument("--kd-temp", type=float, default=2.0, help="蒸馏温度 T（J1/J2）")
    ap.add_argument("--rep-weight", type=float, default=1.0, help="λ_rep（J2）")
    ap.add_argument("--warm", choices=("on", "off"), default="on",
                    help="off = 随机初始化（保真档 F）")
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--out", required=True)
    ap.add_argument("--metrics", required=True)
    args = ap.parse_args(argv)

    arm = args.arm
    seed = args.seed
    warm = args.warm == "on"
    use_kd = arm in ("J1", "J2")
    use_rep = arm == "J2"
    freeze_old_heads = arm == "J3"
    if arm == "F" and warm:
        raise SystemExit("F 档必须 --warm off（它就是随机初始化保真对照）")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    random.seed(seed)
    print(f"[setup] arm={arm} seed={seed} warm={warm} kd={use_kd}(λ={args.kd_weight}, "
          f"T={args.kd_temp}) replay={use_rep}(λ={args.rep_weight}) "
          f"freeze_old_heads={freeze_old_heads} device={device}", flush=True)

    tokenizer = NanoCharTokenizer()
    cards = resolve_tasks(TASK_ORDER)
    resolved = {n: TASK_SAMPLES[n] for n in TASK_ORDER}

    # ---- 数据 ----
    print("[1/5] 构建任务数据集 ...", flush=True)
    raw, vals = {}, {}
    for name in TASK_ORDER:
        t0 = time.perf_counter()
        data = get_raw(name, cards[name])
        vals[name], train = split_val(data, seed)
        raw[name] = train
        print(f"  {name}: n={len(data)} train={len(train)} val={len(vals[name])} "
              f"({time.perf_counter()-t0:.1f}s)", flush=True)

    # ALIGN 自检：我的 val 必须与 capability_map 的 eval_S 逐条相同（P1 同一评测集的前提）
    from probe import split_of as probe_split_of          # noqa: E402
    for name in TASK_ORDER:
        ev, _tr = probe_split_of(name, seed)
        same = [x["text"] for x in ev] == [x["text"] for x in vals[name]]
        print(f"ALIGN_CHECK {name} s{seed} eval==val: {same} (n={len(ev)})", flush=True)
        if not same:
            raise SystemExit(f"评测集对齐失败：{name} —— 本档结论作废")

    loaders = {n: DataLoader(GenericTaskDataset(raw[n], tokenizer, cards[n].spec),
                             batch_size=args.batch_size, shuffle=True, drop_last=True)
               for n in TASK_ORDER}
    val_loaders = {n: DataLoader(GenericTaskDataset(vals[n], tokenizer, cards[n].spec),
                                 batch_size=args.batch_size, shuffle=False)
                   for n in TASK_ORDER}
    rep_loaders = {}
    if use_rep:
        rep_loaders = {n: DataLoader(GenericTaskDataset(raw[n], tokenizer, cards[n].spec),
                                     batch_size=args.batch_size, shuffle=True, drop_last=True,
                                     generator=torch.Generator().manual_seed(seed + 777 + i))
                       for i, n in enumerate(OLD_CARDS)}

    # ---- 模块（构造顺序与 train_multitask 逐字一致 ⇒ RNG 流一致）----
    print("[2/5] 构建 NanoDocEncoder 高效基座 ...", flush=True)
    doc_encoder = NanoDocEncoder(vocab_size=tokenizer.vocab_size, hidden_dim=HIDDEN_DIM,
                                 dropout=0.1, **ENCODER_KWARGS).to(device)
    decoders = {n: RobustARSliceDecoder(hidden_dim=HIDDEN_DIM,
                                        num_classes=cards[n].spec.num_classes,
                                        **DECODER_KWARGS).to(device)
                for n in TASK_ORDER}

    # ---- 温启动 ----
    teacher_core = None
    teacher_decs = None
    if warm:
        print(f"[3/5] 温启动：核 ← {args.base}，四张老卡头 ← {FROZEN_CARD.format('*', seed)}",
              flush=True)
        bck = torch.load(args.base, map_location="cpu", weights_only=False)
        doc_encoder.load_state_dict(bck["doc_encoder"], strict=True)
        for n in OLD_CARDS:
            ck = read_card(FROZEN_CARD.format(n, seed))
            decoders[n].load_state_dict(ck["decoder"], strict=True)
            print(f"  老卡头 {n} ← {FROZEN_CARD.format(n, seed)}", flush=True)
        print("  negation 头：随机初始化（未加载 negation_frozen）", flush=True)
    else:
        print("[3/5] 随机初始化（F 档，保真对照）", flush=True)

    if use_kd:
        teacher_core, _ = load_base_encoder(args.base, device)
        teacher_decs = {}
        for n in OLD_CARDS:
            d, _ = build_card_decoder(read_card(FROZEN_CARD.format(n, seed)), device)
            for p in d.parameters():
                p.requires_grad_(False)
            teacher_decs[n] = d
        print(f"[teacher] 冻结旧核 + {len(teacher_decs)} 张冻结旧头已就位（eval, no-grad）",
              flush=True)

    if freeze_old_heads:
        for n in OLD_CARDS:
            for p in decoders[n].parameters():
                p.requires_grad_(False)

    # ---- 自检 1：真实可训参数量 ----
    def _rep() -> dict:
        core_t = sum(p.numel() for p in doc_encoder.parameters())
        core_r = sum(p.numel() for p in doc_encoder.parameters() if p.requires_grad)
        heads = {n: {"total": sum(p.numel() for p in decoders[n].parameters()),
                     "trainable": sum(p.numel() for p in decoders[n].parameters()
                                      if p.requires_grad)}
                 for n in TASK_ORDER}
        return {"core_total": core_t, "core_trainable": core_r, "heads": heads,
                "trainable_total": core_r + sum(h["trainable"] for h in heads.values())}
    pre = _rep()
    print("SELFTEST_1 " + json.dumps({"arm": arm, "params": pre}, ensure_ascii=False),
          flush=True)
    if freeze_old_heads:
        for n in OLD_CARDS:
            if pre["heads"][n]["trainable"] != 0:
                raise SystemExit(f"自检3失败：{n} 头仍有可训参数，本档作废")
    if not freeze_old_heads and any(pre["heads"][n]["trainable"] == 0 for n in TASK_ORDER):
        raise SystemExit("自检1失败：存在本应可训却为 0 的头")

    # ---- 自检 3：J3 老头 requires_grad 实况 ----
    if freeze_old_heads:
        for n in OLD_CARDS:
            flags = sorted({p.requires_grad for p in decoders[n].parameters()})
            print(f"SELFTEST_3 {n}.requires_grad = {flags} "
                  f"(n_params={sum(1 for _ in decoders[n].parameters())})", flush=True)
            if flags != [False]:
                raise SystemExit(f"自检3失败：{n} 头 requires_grad={flags}，本档作废")
        print("SELFTEST_3 verdict: 四张老卡头全部 requires_grad=False ✅", flush=True)
    else:
        print("SELFTEST_3 本档不冻结老卡头（无此项）", flush=True)

    # ---- 优化器 / 调度（与 train_multitask 同口径）----
    print("[4/5] 优化器与调度 ...", flush=True)
    head_params = [p for n in TASK_ORDER for p in decoders[n].parameters() if p.requires_grad]
    groups = [{"params": list(doc_encoder.parameters()), "lr": args.lr_base},
              {"params": head_params, "lr": args.lr_head}]
    optimizer = torch.optim.AdamW(groups, weight_decay=1e-4)
    total_steps = args.epochs * args.steps_per_epoch
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, total_steps))
    clip_targets = list(doc_encoder.parameters()) + head_params
    print(f"  lr_base={args.lr_base} lr_head={args.lr_head} steps={total_steps} "
          f"trainable={pre['trainable_total']:,}", flush=True)

    # ---- 训练 ----
    print("[5/5] 开始训练 ...", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    iters = {n: iter(loaders[n]) for n in TASK_ORDER}
    rep_iters = {n: iter(rep_loaders[n]) for n in rep_loaders}
    step_times: list[float] = []
    kd_first: float | None = None
    hist: list[dict] = []
    t_start = time.perf_counter()
    n_steps = 0
    kd_total_acc = ce_total_acc = rep_total_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        doc_encoder.train()
        for d in decoders.values():
            d.train()
        if teacher_core is not None:
            teacher_core.eval()
            for d in teacher_decs.values():
                d.eval()
        ce_running = {n: 0.0 for n in TASK_ORDER}
        kd_running = rep_running = tot_running = 0.0

        for _ in range(args.steps_per_epoch):
            ts = time.perf_counter()
            optimizer.zero_grad()
            total = torch.tensor(0.0, device=device)
            kd_val = torch.tensor(0.0, device=device)
            rep_val = torch.tensor(0.0, device=device)
            ce_val = torch.tensor(0.0, device=device)

            for name in TASK_ORDER:
                try:
                    batch = next(iters[name])
                except StopIteration:
                    iters[name] = iter(loaders[name])
                    batch = next(iters[name])
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                mem = doc_encoder(inp, attention_mask=mask)

                if use_kd and name in OLD_CARDS:
                    with torch.no_grad():
                        t_mem = teacher_core(inp, attention_mask=mask)
                    l, kd = kd_task_loss(decoders[name], mem, mask, batch,
                                         cards[name].spec, device,
                                         teacher_decs[name], t_mem, args.kd_temp)
                    kd_val = kd_val + kd
                    total = total + l + args.kd_weight * kd
                else:
                    l = task_loss(decoders[name], mem, mask, batch,
                                  cards[name].spec, device)
                    total = total + l
                ce_val = ce_val + l
                ce_running[name] += l.item()

                if use_rep and name in OLD_CARDS:
                    try:
                        b2 = next(rep_iters[name])
                    except StopIteration:
                        rep_iters[name] = iter(rep_loaders[name])
                        b2 = next(rep_iters[name])
                    i2 = b2["input_ids"].to(device)
                    m2 = b2["attention_mask"].to(device)
                    mem2 = doc_encoder(i2, attention_mask=m2)
                    l2 = task_loss(decoders[name], mem2, m2, b2,
                                   cards[name].spec, device)
                    rep_val = rep_val + l2
                    total = total + args.rep_weight * l2

            total.backward()
            torch.nn.utils.clip_grad_norm_(clip_targets, 1.0)
            optimizer.step()
            scheduler.step()

            kd_f = float(kd_val.detach())
            rep_f = float(rep_val.detach())
            tot_f = float(total.detach())
            ce_f = float(ce_val.detach())
            if kd_first is None:
                kd_first = kd_f
            kd_running += kd_f
            rep_running += rep_f
            tot_running += tot_f
            kd_total_acc += kd_f
            ce_total_acc += ce_f
            rep_total_acc += rep_f
            step_times.append(time.perf_counter() - ts)
            n_steps += 1

        ns = args.steps_per_epoch
        rec = {"epoch": epoch,
               "ce": {n: ce_running[n] / ns for n in TASK_ORDER},
               "kd": kd_running / ns, "replay_ce": rep_running / ns,
               "total": tot_running / ns,
               "kd_share": (args.kd_weight * kd_running) / max(1e-9, tot_running),
               "lr": [f"{g['lr']:.3e}" for g in optimizer.param_groups]}
        hist.append(rec)
        if epoch % 4 == 0 or epoch == 1 or epoch == args.epochs:
            print(f"Epoch {epoch:2d}/{args.epochs} | ce=" +
                  " ".join(f"{n}={rec['ce'][n]:.3f}" for n in TASK_ORDER) +
                  f" | kd={rec['kd']:.4f} (占 {rec['kd_share']*100:.2f}%) "
                  f"rep={rec['replay_ce']:.4f} total={rec['total']:.4f}", flush=True)

    if device.type == "cuda":
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t_start
    peak_mb = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    steady = sorted(step_times[-200:])
    sec_steady = steady[len(steady) // 2]

    # ---- 自检 2：蒸馏/正则项的实际数值 ----
    kd_mean_all = kd_total_acc / max(1, n_steps)
    loss_total_acc = ce_total_acc + rep_total_acc + args.kd_weight * kd_total_acc
    kd_share_all = (args.kd_weight * kd_total_acc) / max(1e-9, loss_total_acc)
    if use_kd:
        print(f"SELFTEST_2 kd_first_step={kd_first:.6f} kd_mean_all={kd_mean_all:.6f} "
              f"kd_epoch1={hist[0]['kd']:.6f} kd_last={hist[-1]['kd']:.6f} "
              f"kd_share_first={hist[0]['kd_share']:.6f} kd_share_last={hist[-1]['kd_share']:.6f} "
              f"kd_share_all={kd_share_all:.6f}", flush=True)
        nonzero = abs(kd_first) > 0 and abs(hist[-1]["kd"]) > 0
        print(f"SELFTEST_2 verdict 非零={nonzero}", flush=True)
        if not nonzero:
            raise SystemExit("自检2失败：蒸馏项恒为 0 ⇒ 约束没生效，本档作废")
    else:
        print(f"SELFTEST_2 本档无蒸馏/正则项（kd=0, replay=0）；"
              f"唯一正则是 AdamW weight_decay=1e-4。kd_mean_all={kd_mean_all:.6f}", flush=True)
    if use_rep:
        print(f"SELFTEST_2 replay_mean={rep_running/ns:.6f}", flush=True)

    # ---- 验证（val = eval_S）----
    print("[eval] 逐任务验证 ...", flush=True)
    report = {}
    for name in TASK_ORDER:
        m = evaluate_task(doc_encoder, decoders[name], val_loaders[name],
                          device, cards[name].spec)
        report[name] = m
        print(f"  [{name:9s}] exact={m['exact_match']:.4f} cls_acc={m['cls_acc']:.4f} "
              f"bg_fp={m['bg_fp']:.4f}", flush=True)

    # ---- 存档 ----
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    extra = {"arm": arm, "warm": warm, "kd": use_kd, "kd_weight": args.kd_weight,
             "kd_temp": args.kd_temp, "replay": use_rep, "rep_weight": args.rep_weight,
             "freeze_old_heads": freeze_old_heads,
             "kd_stats": {"first": kd_first, "mean": kd_mean_all,
                          "share_all": kd_share_all,
                          "share_last_epoch": hist[-1]["kd_share"]},
             "params": pre,
             "timing": {"train_sec": train_sec, "n_steps": n_steps,
                        "sec_per_step": train_sec / max(1, n_steps),
                        "sec_per_step_steady": sec_steady, "peak_mem_mb": peak_mb},
             "history": hist}
    torch.save({
        "doc_encoder": doc_encoder.state_dict(),
        "decoders": {k: v.state_dict() for k, v in decoders.items()},
        "hidden_dim": HIDDEN_DIM,
        "encoder": "NanoDocEncoder",
        "encoder_kwargs": ENCODER_KWARGS,
        "decoder_kwargs": DECODER_KWARGS,
        "task_order": TASK_ORDER,
        "task_specs": {n: cards[n].spec.to_snapshot() for n in TASK_ORDER},
        "train_args": {"num_epochs": args.epochs, "batch_size": args.batch_size,
                       "seed": seed, "lr_base": args.lr_base, "lr_head": args.lr_head,
                       "task_samples": resolved, "steps_per_epoch": args.steps_per_epoch,
                       "arm": arm, "warm": warm},
        "extra": {"core_keep": extra},
    }, out)
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(
        {"arm": arm, "seed": seed, "metrics": report, "extra": extra},
        ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(f"[save] {out}", flush=True)

    print("CK_METRICS " + json.dumps({
        "arm": arm, "seed": seed, "warm": warm,
        "exact": {n: report[n]["exact_match"] for n in TASK_ORDER},
        "cls_acc": {n: report[n]["cls_acc"] for n in TASK_ORDER},
        "params": pre,
        "kd_stats": extra["kd_stats"],
        "timing": extra["timing"],
        "n_steps": n_steps,
    }, ensure_ascii=False), flush=True)
    print("COREKEEP_TRAIN_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
