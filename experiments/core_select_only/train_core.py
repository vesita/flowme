#!/usr/bin/env python3
"""core_select_only 训练 + 全量评测（两臂唯一变量 = 每步是否跑老任务批次 / 老卡头是否挂载）。

  --arm joint    : 核对臂，必须**零误差复现** `select_semantic_joint` 主臂
                   （select batch + 4 老任务 batch，老卡头挂载并冻结）—— 代码路径与
                   select_semantic_joint/train_sem.py 的 joint 分支逐行同源（只读 import 其 common.py）。
  --arm selonly  : 主臂，**只有 select batch**；核可训；**训练期间不挂载任何老卡头**
                   （不参与任何计算）；训练结束后只读加载老卡头做 T5 评测（no_grad，不算梯度）。

配方 = PREREG §1（跑前写死）：1800 步、batch 64、lr_core 3e-4、lr_head 1e-3、AdamW wd 1e-4、
cosine T_max=1800、clip 1.0、seed 42/43、核 eval()。数据只读 select_rerank/data/clean。

用法：
  uv run python experiments/core_select_only/train_core.py --arm joint --seed 42
  uv run python experiments/core_select_only/train_core.py --arm selonly --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SSJ = ROOT / "experiments" / "select_semantic_joint"          # 只读
sys.path.insert(0, str(SSJ))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import (  # noqa: E402  (select_semantic_joint/common.py —— 只读 import，不写其目录)
    BAND, BASE_CKPT, BATCH, CLIP, ENC_PARAMS, EXPECTED_N, FP_EXPECTED, HEAD_PARAMS,
    LR_CORE, LR_HEAD, OLD_CARDS, SEEDS, SPLITS, STEPS, TWO_SE, WD,
    SemModel, SemSpec, core_drift, data_fingerprint, eval_old_cards, eval_splits,
    freeze_report_old, inputs_for, load_base_encoder, load_blob, load_old_heads,
    load_rows, old_loaders, old_loss)

RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
R4_BASELINE = SSJ / "results"                                  # 只读（T0/step-0 对账参照）
ARMS = ("joint", "selonly")


class IndexDataset(torch.utils.data.Dataset):
    """只按下标取数（与 select_pool / select_semantic_joint 同）。"""

    def __init__(self, idx: torch.Tensor):
        self.idx = idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        return self.idx[i]


def top_modules(model: torch.nn.Module) -> list[str]:
    """named_modules() 的顶层命名空间（挂载证据：selonly 只应有本模块自己的部件）。"""
    return sorted({name.split(".")[0] for name, _ in model.named_modules() if name})


def train_one(arm: str, seed: int, randlabel: bool, steps: int = STEPS,
              device: str | None = None, tag: str = "") -> dict:
    assert arm in ARMS, arm
    t_start = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    # ---- 模型构造顺序与 select_pool / select_semantic_joint 逐字一致（RNG 流同源） ----
    spec = SemSpec.from_build_spec()
    model = SemModel(spec, freeze_core=False).to(device)      # 两臂核都可训
    freeze = model.freeze_report()
    print(f"[freeze] {json.dumps(freeze, ensure_ascii=False)}", flush=True)

    fp = data_fingerprint()
    assert fp == FP_EXPECTED, f"数据指纹漂移：{fp}"
    rows = load_rows()
    blob = load_blob()
    assert len(rows) == blob["n"], f"行数 {len(rows)} ≠ 缓存 {blob['n']}"

    n_train = EXPECTED_N["train"]
    labels = blob["labels"].clone()
    randlabel_n_diff = None
    if randlabel:                       # T4：只打乱 train 标签，保持 1:1
        g = torch.Generator().manual_seed(seed * 1000 + 7)
        idx = torch.arange(n_train)
        labels[idx] = labels[idx][torch.randperm(n_train, generator=g)]
        cnt = torch.bincount(labels[:n_train], minlength=2).tolist()
        n_diff = int((labels[:n_train] != blob["labels"][:n_train]).sum())
        print(f"[randlabel] train 标签已 randperm，计数 {cnt}，与真标签不同 {n_diff} 条", flush=True)
        assert cnt == [n_train // 2, n_train - n_train // 2], f"随机标签计数不是 1:1：{cnt}"
        assert n_diff > 0, "随机标签没有改变任何标签（空对照）"
        randlabel_n_diff = n_diff

    train_idx = torch.nonzero(blob["splits"] == 0, as_tuple=False).squeeze(-1)
    ds = IndexDataset(train_idx)
    gen = torch.Generator().manual_seed(seed)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True, generator=gen, drop_last=False)

    # ---- 可训参数 / 优化器（两臂同：核 3e-4、头 1e-3） ----
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    assert n_params == HEAD_PARAMS + ENC_PARAMS, f"{arm} 可训参数 {n_params} ≠ {HEAD_PARAMS + ENC_PARAMS}"
    core_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    assert len(core_params) > 0 and len(head_params) > 0
    opt = torch.optim.AdamW([{"params": core_params, "lr": LR_CORE},
                             {"params": head_params, "lr": LR_HEAD}],
                            weight_decay=WD)
    clip_targets = core_params + head_params
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for p in params:
        p.requires_grad_(True)

    # ---- 老卡头：joint 挂载（模型构造之后建 ⇒ 不污染打分头 RNG 流）；selonly 不挂载 ----
    heads, old_dl, old_rep, r4_step0 = {}, {}, None, None
    if arm == "joint":
        heads = load_old_heads(seed, device)
        old_rep = freeze_report_old(heads)
        print(f"[freeze] 老卡头冻结实况 {json.dumps(old_rep, ensure_ascii=False)}", flush=True)
        old_dl = old_loaders(seed)
        r4_step0 = eval_old_cards(model.encoder, heads, seed, device, with_eval_task=False)
        print("R4_STEP0 " + json.dumps({"seed": seed,
                                        "combined": r4_step0["combined_sha256"],
                                        "per_card": {n: r4_step0[n]["sha256"] for n in OLD_CARDS},
                                        "exact": {n: r4_step0[n]["exact"] for n in OLD_CARDS}},
                                       ensure_ascii=False), flush=True)
        base_path = R4_BASELINE / f"r4_baseline_s{seed}.json"
        if base_path.exists():
            base = json.loads(base_path.read_text(encoding="utf-8"))
            ok = base["combined_sha256"] == r4_step0["combined_sha256"]
            print(f"[R4] 与 r4_check 基线比对：{'相等 ✅' if ok else '不相等 ❌'} "
                  f"{r4_step0['combined_sha256']} vs {base['combined_sha256']}", flush=True)
            assert ok, "R4 step-0 与『不接本任务』配置不逐位相同 ⇒ 权重没接对，本 seed 作废"

    # live = 是否把编码接进计算图（两臂核都可训 ⇒ 一律 True；缓存路径核收不到梯度，空测试实测踩过）
    # use_old = 每步是否同时跑 4 个老任务批次 —— **本实验的唯一变量**
    live = True
    use_old = arm == "joint"
    assert use_old == (len(heads) == 4), "挂载与臂不一致"
    if arm == "selonly":
        assert heads == {} and old_dl == {}, "selonly 竟挂了老卡头"
    iters = {n: iter(old_dl[n]) for n in OLD_CARDS} if use_old else {}

    # ---- 挂载证据（PREREG §4.2，训练开始前打印） ----
    mount_ev = {
        "arm": arm,
        "old_heads_mounted_during_training": bool(heads),
        "heads_keys_during_training": sorted(heads.keys()),
        "old_loaders_during_training": sorted(old_dl.keys()),
        "model_top_level_modules": top_modules(model),
        "model_named_modules_n": sum(1 for _ in model.named_modules()),
        "old_forward_count": 0,
    }
    print("MOUNT_EVIDENCE_PRE " + json.dumps(mount_ev, ensure_ascii=False), flush=True)
    if arm == "selonly":
        assert mount_ev["heads_keys_during_training"] == []
        assert not any(k in mount_ev["model_top_level_modules"] for k in OLD_CARDS)

    # ---- 训练 ----
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    step_times: list[float] = []
    losses: list[float] = []
    totals: list[float] = []
    old_forward_count = 0
    core_grad_norm_step1 = None
    step = 0
    t0 = time.perf_counter()
    while step < steps:
        for j in dl:
            j = j[0] if isinstance(j, (list, tuple)) else j
            ts = time.perf_counter()
            c, mc, d, md = inputs_for(model, blob, rows, j, live, device)
            y = labels[j].to(device)
            logits = model.forward_tokens(c, mc, d, md)
            loss = torch.nn.functional.cross_entropy(logits, y)
            total = loss
            if use_old:                  # 旧任务 + 新任务**同批梯度**（core_keep J3 口径）
                for i, n in enumerate(OLD_CARDS):
                    try:
                        b = next(iters[n])
                    except StopIteration:
                        iters[n] = iter(old_dl[n])
                        b = next(iters[n])
                    total = total + old_loss(heads, model.encoder, b, n, device)
                    old_forward_count += 1
            opt.zero_grad(set_to_none=True)
            total.backward()
            if step == 0:                # 第 1 步硬断言：核确实收到梯度（编码路径必须接进图）
                gs = [p.grad.detach().float().norm()
                      for p in model.encoder.parameters() if p.grad is not None]
                core_grad_norm_step1 = float(torch.stack(gs).norm()) if gs else 0.0
                assert core_grad_norm_step1 > 0, \
                    "第 1 步核梯度范数 = 0 ⇒ 核没收到梯度（没走 live 编码或被冻结）"
                print(f"[grad] step1 核梯度范数 = {core_grad_norm_step1:.6e}", flush=True)
            torch.nn.utils.clip_grad_norm_(clip_targets, CLIP)
            opt.step()
            sched.step()
            losses.append(float(loss))
            totals.append(float(total))
            step_times.append(time.perf_counter() - ts)
            step += 1
            if step >= steps:
                break
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    train_sec = time.perf_counter() - t0

    mount_ev["old_forward_count"] = old_forward_count
    mount_ev["core_grad_norm_step1"] = core_grad_norm_step1
    print("MOUNT_EVIDENCE_POST " + json.dumps(mount_ev, ensure_ascii=False), flush=True)
    if arm == "selonly":
        assert old_forward_count == 0, f"selonly 训练期间老任务前向 {old_forward_count} 次"
        assert not heads and not old_dl, "selonly 训练期间挂了老卡头"

    assert step == steps, f"只跑了 {step} 步"
    assert all(math.isfinite(x) for x in losses), "loss 出现非有限值"
    m_first, m_last = sum(losses[:50]) / 50, sum(losses[-50:]) / 50
    if not randlabel:
        assert m_last < m_first, f"loss 没降（前50 {m_first:.4f} / 后50 {m_last:.4f}）"
    m_tot_first = sum(totals[:50]) / 50
    m_tot_last = sum(totals[-50:]) / 50
    print(f"[train] {arm} {step} 步 done，select loss 首/末 {losses[0]:.4f}/{losses[-1]:.4f} "
          f"前50均 {m_first:.4f} / 后50均 {m_last:.4f} | "
          f"total 前50均 {m_tot_first:.4f} / 后50均 {m_tot_last:.4f}", flush=True)

    steady = sorted(step_times[-min(200, len(step_times)):])
    sec_steady = steady[len(steady) // 2]
    peak_mb = (torch.cuda.max_memory_allocated() / 2 ** 20) if device.startswith("cuda") else 0.0

    # ---- 选择任务全量评测 ----
    evals = eval_splits(model, blob, rows, live, device)
    for s in SPLITS:
        e = evals[s]
        print(f"[eval] {arm}{'_rand' if randlabel else ''} seed{seed} {s:5s} "
              f"n={e['n']} acc={e['acc']:.4f} (SE={e['se']:.4f} "
              f"余量/SE={e['margin_over_se']:+.2f} 多数类={e['majority_baseline']:.4f})", flush=True)
    for name, e in evals.get("adv_by_ctype", {}).items():
        print(f"[eval] {arm}{'_rand' if randlabel else ''} seed{seed} adv/{name} "
              f"n={e['n']} acc={e['acc']:.4f} (SE={e['se']:.4f} "
              f"余量/SE={e['margin_over_se']:+.2f} 过2SE={e['passes_2se']})", flush=True)

    # ---- 老卡 T5：训练后（joint 用训练中挂载的头；selonly 此时才只读加载，且只做 no_grad 评测） ----
    r4_post, r5, drift = None, None, None
    if arm == "joint":
        r4_post = eval_old_cards(model.encoder, heads, seed, device, with_eval_task=False)
        r5 = {n: {"step0": r4_step0[n]["exact"], "post": r4_post[n]["exact"],
                  "delta": r4_post[n]["exact"] - r4_step0[n]["exact"],
                  "band": BAND[n]} for n in OLD_CARDS}
        r4_step0_band = {n: BAND[n] for n in OLD_CARDS}
    else:
        # 此时才只读加载老卡头（训练期间从未挂载）；只做 no_grad 评测，不进 optimizer
        t5_heads = heads_seed_heads(seed, device)
        # step-0 参照 = 「不接本任务」配置 B：**独立的新基座核**（不是训练模型的核）+ 冻结老卡头
        base_enc, _ = load_base_encoder(str(BASE_CKPT), device)
        base_enc.eval()
        r4_base = eval_old_cards(base_enc, t5_heads, seed, device, with_eval_task=False)
        ref = R4_BASELINE / f"r4_baseline_s{seed}.json"
        assert ref.exists(), f"缺 step-0 参照：{ref}"
        ref_json = json.loads(ref.read_text(encoding="utf-8"))
        ok = ref_json["combined_sha256"] == r4_base["combined_sha256"]
        print(f"[T5 step-0] 与 r4_check 基线比对：{'相等 ✅' if ok else '不相等 ❌'} "
              f"{r4_base['combined_sha256']} vs {ref_json['combined_sha256']}", flush=True)
        assert ok, "selonly 的 step-0 老卡基线与 r4_check 不逐位相同"
        r4_post = eval_old_cards(model.encoder, t5_heads, seed, device, with_eval_task=False)
        del t5_heads
        r5 = {n: {"step0": r4_base[n]["exact"], "post": r4_post[n]["exact"],
                  "delta": r4_post[n]["exact"] - r4_base[n]["exact"],
                  "band": BAND[n]} for n in OLD_CARDS}
        r4_step0_band = {n: BAND[n] for n in OLD_CARDS}
        del base_enc
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    print("R5_OLD " + json.dumps({n: {"step0": round(v["step0"], 6),
                                      "post": round(v["post"], 6),
                                      "delta": round(v["delta"], 6),
                                      "band": v["band"]}
                                  for n, v in r5.items()}, ensure_ascii=False), flush=True)
    drift = core_drift(model.encoder)
    assert drift > 0, f"核漂移 = {drift} ⇒ 核一步没动（本实验两臂核都必须可训）"
    print(f"R6_DRIFT rel={drift:.6f} sec/step_steady={sec_steady:.4f} "
          f"peak_mem_mb={peak_mb:.1f}", flush=True)

    name = f"{arm}{tag}_s{seed}{'_rand' if randlabel else ''}"
    out = {
        "name": name, "arm": arm, "seed": seed, "data_arm": "clean",
        "randlabel": randlabel, "randlabel_n_diff": randlabel_n_diff,
        "device": device, "freeze": freeze,
        "old_heads_freeze": old_rep,
        "mount_evidence": mount_ev,
        "spec": {"max_len_ctx": spec.max_len_ctx, "max_len_cand": spec.max_len_cand,
                 "head_hidden": spec.head_hidden, "hidden": spec.hidden},
        "recipe": {"steps": steps, "batch": BATCH, "lr_head": LR_HEAD, "lr_core": LR_CORE,
                   "weight_decay": WD, "cosine": True, "grad_clip": CLIP,
                   "n_train": n_train, "old_tasks_in_batch": (OLD_CARDS if use_old else []),
                   "old_heads_mounted_during_training": bool(use_old),
                   "encoder_eval_mode": True},
        "eval": evals,
        "r4_step0": r4_step0, "r4_post": r4_post, "r5": r5, "core_drift": drift,
        "r5_band_ref": r4_step0_band,
        "timing": {"train_sec": round(train_sec, 2), "n_steps": steps,
                   "sec_per_step": round(train_sec / max(1, steps), 4),
                   "sec_per_step_steady": round(sec_steady, 4),
                   "peak_mem_mb": round(peak_mb, 1)},
        "loss_first": round(losses[0], 6), "loss_last": round(losses[-1], 6),
        "loss_select_first50": round(m_first, 6), "loss_select_last50": round(m_last, 6),
        "loss_total_first50": round(m_tot_first, 6), "loss_total_last50": round(m_tot_last, 6),
        "wall_sec": round(time.time() - t_start, 1),
        "data_fingerprint": fp,
        "sp_cache_readonly": True,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    WEIGHTS.mkdir(exist_ok=True)
    torch.save({k: v for k, v in model.state_dict().items()
                if k.startswith(("encoder.", "head."))}, WEIGHTS / f"{name}.pt")
    print(f"[done] {name} → results/{name}.json  ({out['wall_sec']}s)", flush=True)
    return out


def heads_seed_heads(seed: int, device: str) -> dict:
    """selonly 的 T5：训练后只读加载四张冻结老卡头（no_grad 评测用，不进任何 optimizer）。"""
    h = load_old_heads(seed, device)
    rep = freeze_report_old(h)
    print(f"[T5] 老卡头（训练后只读加载，仅评测）冻结实况 "
          f"{json.dumps(rep, ensure_ascii=False)}", flush=True)
    return h


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True, choices=list(SEEDS))
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--tag", default="", help="产物名后缀（探跑用）")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    train_one(a.arm, a.seed, a.randlabel, a.steps, a.device, a.tag)


if __name__ == "__main__":
    main(sys.argv[1:])
