#!/usr/bin/env python3
"""layer_selective 训练 + 全量评测（五臂唯一变量 = 哪些参数 requires_grad=True）。

  --arm F    基线：核冻结、只训打分头、读 token 缓存 ⇒ 必须零误差复现 select_pool 臂 A
  --arm ALL  = core_select_only::selonly：核全可训、只给 select 梯度 ⇒ 必须零误差复现 selonly
  --arm EMB  只训 encoder.embedding.weight（+ 头），其余核冻结
  --arm TOP  只训 encoder.blocks.2.*（最深层，+ 头），其余核冻结
  --arm ADPT 核全冻结 + 私有适配层（核输出之后 / mean-pool 之前，残差零初始化）+ 头

配方 = PREREG §2（跑前写死）：1800 步、batch 64、lr_core 3e-4（含核参数的臂）/ lr_head 1e-3、
AdamW wd 1e-4、cosine T_max=1800、clip 1.0、seed 42/43、核 eval()。数据只读 select_rerank/data/clean。
五臂一律**只跑 select batch**（不挂载任何老卡头、不跑老任务批次）；老卡只在训练后 no_grad 评测。

用法：
  uv run python experiments/layer_selective/train_ls.py --arm F --seed 42
  uv run python experiments/layer_selective/train_ls.py --arm TOP --seed 43 --randlabel
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
CSO = ROOT / "experiments" / "core_select_only"               # 只读
sys.path.insert(0, str(SSJ))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import (  # noqa: E402  (select_semantic_joint/common.py —— 只读 import)
    BAND, BASE_CKPT, BATCH, CLIP, ENC_PARAMS, EXPECTED_N, FP_EXPECTED, HEAD_PARAMS,
    LR_CORE, LR_HEAD, OLD_CARDS, SEEDS, SPLITS, STEPS, WD,
    SemModel, SemSpec, core_drift, data_fingerprint, eval_old_cards, eval_splits,
    freeze_report_old, inputs_for, load_base_encoder, load_blob, load_old_heads,
    load_rows)

RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
R4_BASELINE = SSJ / "results"                                  # 只读（step-0 老卡参照）

ARMS = ("F", "ALL", "EMB", "TOP", "ADPT")
#: 臂 → 核的可训子集（None = 核全冻；"all" = 全可训；否则前缀匹配）
CORE_POLICY = {"F": None, "ADPT": None, "ALL": "all", "EMB": "embedding", "TOP": "blocks.2"}
#: 臂 → 训练/评测是否走 live 编码（核可训才必须 live；核冻结臂读缓存 ⇒ 与臂 A 逐位同输入）
LIVE = {"F": False, "ADPT": False, "ALL": True, "EMB": True, "TOP": True}
#: PREREG §2 跑前写死的可训参数量
EXPECT_TRAINABLE = {"F": 164_353, "ALL": 1_852_813, "EMB": 1_212_929,
                    "TOP": 377_605, "ADPT": 230_273}
#: PREREG §4.7：五臂真标签 step-0 loss 必须同源（同 seed 同头初始化）
EXPECT_LOSS_FIRST = {42: 0.693002, 43: 0.692196}
ADAPTER_MID = 256
ADAPTER_PARAMS = (128 * ADAPTER_MID + ADAPTER_MID) + (ADAPTER_MID * 128 + 128)   # 65,920


class Adapter(nn.Module):
    """私有适配层：核输出之后、mean-pool 之前，token 级；残差 + fc2 零初始化 ⇒ step-0 恒等。"""

    def __init__(self, hidden: int = 128, mid: int = ADAPTER_MID):
        super().__init__()
        self.fc1 = nn.Linear(hidden, mid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid, hidden)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.fc2(self.act(self.fc1(x)))


class LSModel(SemModel):
    """SemModel + 可选私有适配层。适配层在 super().__init__ **之后**建 ⇒ 不碰打分头的 RNG 流。"""

    def __init__(self, spec: SemSpec, with_adapter: bool):
        super().__init__(spec, freeze_core=False)      # 构造顺序与 select_pool 逐字一致
        self.adapter = Adapter(spec.hidden) if with_adapter else None

    def forward_tokens(self, h_ctx, m_ctx, h_cand, m_cand) -> torch.Tensor:
        if self.adapter is not None:
            h_ctx = self.adapter(h_ctx)
            h_cand = self.adapter(h_cand)
        return super().forward_tokens(h_ctx, m_ctx, h_cand, m_cand)


class IndexDataset(torch.utils.data.Dataset):
    """只按下标取数（与 select_pool / select_semantic_joint 同）。"""

    def __init__(self, idx: torch.Tensor):
        self.idx = idx

    def __len__(self) -> int:
        return len(self.idx)

    def __getitem__(self, i: int):
        return self.idx[i]


def apply_policy(model: LSModel, arm: str) -> dict:
    """按臂施加 requires_grad 策略，并返回「冻结实况」（实测，不是声明）。"""
    pol = CORE_POLICY[arm]
    for p in model.encoder.parameters():                  # 先按臂全冻 / 全开
        p.requires_grad_(pol == "all")
    if pol not in (None, "all"):
        n = 0
        for name, p in model.encoder.named_parameters():
            if name.startswith(pol + "."):
                p.requires_grad_(True)
                n += 1
        assert n > 0, f"{arm}: 核子集 {pol} 没匹配到任何参数"
    for p in model.head.parameters():
        p.requires_grad_(True)
    if model.adapter is not None:
        for p in model.adapter.parameters():
            p.requires_grad_(True)
    frozen = [{"name": n, "numel": p.numel()}
              for n, p in model.encoder.named_parameters() if not p.requires_grad]
    train_core = [{"name": n, "numel": p.numel()}
                  for n, p in model.encoder.named_parameters() if p.requires_grad]
    return {"policy": str(pol), "frozen_core": frozen, "trainable_core": train_core}


def inventory(model: LSModel, arm: str) -> dict:
    """逐臂实测打印 requires_grad=True 的参数名 + 数量（PREREG §4.1）。"""
    tr = [{"name": n, "numel": p.numel()} for n, p in model.named_parameters() if p.requires_grad]
    fr = [{"name": n, "numel": p.numel()} for n, p in model.named_parameters() if not p.requires_grad]
    n_tr = sum(x["numel"] for x in tr)
    assert n_tr == EXPECT_TRAINABLE[arm], \
        f"{arm}: 可训参数 {n_tr} ≠ PREREG 预期 {EXPECT_TRAINABLE[arm]}"
    enc_r = sum(p.numel() for p in model.encoder.parameters() if p.requires_grad)
    enc_p = sum(p.numel() for p in model.encoder.parameters())
    head_p = sum(p.numel() for p in model.head.parameters())
    ad_p = sum(p.numel() for p in model.adapter.parameters()) if model.adapter is not None else 0
    assert enc_p == ENC_PARAMS and head_p == HEAD_PARAMS
    if model.adapter is not None:
        assert ad_p == ADAPTER_PARAMS == 65_920, f"适配层参数 {ad_p} ≠ 65,920"
        assert ad_p / ENC_PARAMS <= 0.05, "适配层 > 核的 5%"
    assert not model.encoder.training, "核不在 eval() 模式"
    return {"arm": arm, "policy": CORE_POLICY[arm], "live": LIVE[arm],
            "n_trainable": n_tr, "trainable": tr, "n_frozen": len(fr), "frozen": fr,
            "encoder_params": enc_p, "encoder_trainable": enc_r,
            "head_params": head_p, "adapter_params": ad_p,
            "adapter_pct_of_core": round(100.0 * ad_p / enc_p, 3) if ad_p else 0.0,
            "encoder_training": model.encoder.training}


def grad_stats(model: LSModel) -> dict:
    """step-0 反向之后：可训侧梯度范数 + 冻结侧是否真的一个梯度都没有。"""
    t_norm, t_none, frozen_with_grad = 0.0, [], []
    for n, p in model.named_parameters():
        if p.requires_grad:
            if p.grad is None:
                t_none.append(n)
            else:
                t_norm += float(p.grad.detach().float().norm()) ** 2
        elif p.grad is not None:
            frozen_with_grad.append(n)
    return {"trainable_grad_norm": math.sqrt(t_norm), "trainable_grad_none": t_none,
            "frozen_with_grad": frozen_with_grad}


def train_one(arm: str, seed: int, randlabel: bool, steps: int = STEPS,
              device: str | None = None, tag: str = "") -> dict:
    assert arm in ARMS, arm
    t_start = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)

    # ---- 模型构造顺序与 select_pool / select_semantic_joint / core_select_only 逐字一致 ----
    spec = SemSpec.from_build_spec()
    model = LSModel(spec, with_adapter=(arm == "ADPT")).to(device)
    pol = apply_policy(model, arm)
    inv = inventory(model, arm)
    head_keys = ("arm", "policy", "n_trainable", "n_frozen",
                 "encoder_params", "encoder_trainable", "head_params", "adapter_params")
    print(f"[freeze] {json.dumps({k: inv[k] for k in head_keys}, ensure_ascii=False)}", flush=True)
    print("TRAINABLE_INV " + json.dumps(inv, ensure_ascii=False), flush=True)
    print("FROZEN_INV " + json.dumps(pol, ensure_ascii=False), flush=True)

    # PREREG §4.4：适配层 step-0 恒等（零初始化生效）
    if model.adapter is not None:
        with torch.no_grad():
            x = torch.randn(7, 11, model.spec.hidden, device=device)
            d0 = float((model.adapter(x) - x).abs().max())
        assert d0 == 0.0, f"适配层 step-0 不恒等：max|Δ| = {d0}"
        print(f"[adapter] step-0 恒等 max|Δ| = {d0}，参数 {ADAPTER_PARAMS} "
              f"({100.0 * ADAPTER_PARAMS / ENC_PARAMS:.3f}% of core)", flush=True)

    fp = data_fingerprint()
    assert fp == FP_EXPECTED, f"数据指纹漂移：{fp}"
    rows = load_rows()
    blob = load_blob()
    assert len(rows) == blob["n"], f"行数 {len(rows)} ≠ 缓存 {blob['n']}"

    n_train = EXPECTED_N["train"]
    labels = blob["labels"].clone()
    randlabel_n_diff = None
    if randlabel:                       # X4：只打乱 train 标签，保持 1:1
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

    # ---- 可训参数 / 优化器（PREREG §2：含核参数两组 3e-4/1e-3；不含核参数单组 1e-3） ----
    params = [p for p in model.parameters() if p.requires_grad]
    n_params = sum(p.numel() for p in params)
    assert n_params == EXPECT_TRAINABLE[arm]
    core_train = [p for p in model.encoder.parameters() if p.requires_grad]
    head_params = list(model.head.parameters())
    if core_train:
        opt = torch.optim.AdamW([{"params": core_train, "lr": LR_CORE},
                                 {"params": head_params, "lr": LR_HEAD}], weight_decay=WD)
        clip_targets = core_train + head_params
        lr_groups = {"core_subset": LR_CORE, "head": LR_HEAD}
    else:
        opt = torch.optim.AdamW(params, lr=LR_HEAD, weight_decay=WD)
        clip_targets = params
        n_ad = sum(p.numel() for p in model.adapter.parameters()) if model.adapter is not None else 0
        lr_groups = {"head_and_private": LR_HEAD, "n_adapter": n_ad}
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    for p in params:
        p.requires_grad_(True)

    # ---- 五臂一律不挂载老卡头（老任务不参与任何训练计算） ----
    heads, old_dl = {}, {}
    live = LIVE[arm]
    iters = {}

    mount_ev = {
        "arm": arm,
        "old_heads_mounted_during_training": False,
        "heads_keys_during_training": [],
        "old_loaders_during_training": [],
        "old_tasks_in_batch": [],
        "model_top_level_modules": sorted({n.split(".")[0] for n, _ in model.named_modules() if n}),
        "model_named_modules_n": sum(1 for _ in model.named_modules()),
        "old_forward_count": 0,
        "live": live,
        "lr_groups": lr_groups,
    }
    print("MOUNT_EVIDENCE_PRE " + json.dumps(mount_ev, ensure_ascii=False), flush=True)
    assert mount_ev["heads_keys_during_training"] == []
    assert not any(k in mount_ev["model_top_level_modules"] for k in OLD_CARDS)

    # ---- 训练 ----
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    step_times: list[float] = []
    losses: list[float] = []
    totals: list[float] = []
    step = 0
    g_step0 = None
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
            opt.zero_grad(set_to_none=True)
            total.backward()
            if step == 0:
                g_step0 = grad_stats(model)
                print("GRADSTAT_STEP0 " + json.dumps(
                    {"arm": arm, "trainable_grad_norm": round(g_step0["trainable_grad_norm"], 6),
                     "trainable_grad_none": g_step0["trainable_grad_none"],
                     "frozen_with_grad": g_step0["frozen_with_grad"],
                     "n_trainable": inv["n_trainable"]}, ensure_ascii=False), flush=True)
                assert not g_step0["trainable_grad_none"], \
                    f"可训参数没收到梯度：{g_step0['trainable_grad_none']}"
                assert not g_step0["frozen_with_grad"], \
                    f"冻结参数收到梯度（『只训指定层』是假的）：{g_step0['frozen_with_grad']}"
                if core_train:
                    assert g_step0["trainable_grad_norm"] > 0, "第 1 步可训侧梯度范数 = 0"
                else:
                    assert g_step0["trainable_grad_norm"] > 0, "第 1 步头/适配层梯度范数 = 0"
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

    assert step == steps, f"只跑了 {step} 步"
    assert all(math.isfinite(x) for x in losses), "loss 出现非有限值"
    m_first, m_last = sum(losses[:50]) / 50, sum(losses[-50:]) / 50
    if not randlabel:
        assert m_last < m_first, f"loss 没降（前50 {m_first:.4f} / 后50 {m_last:.4f}）"
        # PREREG §4.7：五臂真标签 step-0 同源
        exp = EXPECT_LOSS_FIRST[seed]
        assert round(losses[0], 6) == exp, \
            f"loss_first {round(losses[0], 6)} ≠ {exp} ⇒ 构造顺序/RNG 流被改过（口径失同源）"
    m_tot_first = sum(totals[:50]) / 50
    m_tot_last = sum(totals[-50:]) / 50
    print(f"[train] {arm}{'_rand' if randlabel else ''} {step} 步 done，select loss 首/末 "
          f"{losses[0]:.6f}/{losses[-1]:.6f} 前50均 {m_first:.4f} / 后50均 {m_last:.4f} | "
          f"前50均 {m_tot_first:.4f} / 后50均 {m_tot_last:.4f}", flush=True)

    steady = sorted(step_times[-min(200, len(step_times)):])
    sec_steady = steady[len(steady) // 2]
    peak_mb = (torch.cuda.max_memory_allocated() / 2 ** 20) if device.startswith("cuda") else 0.0

    # ---- 选择任务全量评测（核冻结臂读缓存 ⇒ 与臂 A 逐位同输入） ----
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

    # ---- 老卡：训练后只读加载冻结头做 no_grad 评测（与 core_select_only::selonly 同一路径） ----
    t_heads = load_old_heads(seed, device)
    old_rep = freeze_report_old(t_heads)
    print(f"[T5] 老卡头（训练后只读加载，仅评测）冻结实况 {json.dumps(old_rep, ensure_ascii=False)}",
          flush=True)
    base_enc, _ = load_base_encoder(str(BASE_CKPT), device)
    base_enc.eval()
    r4_base = eval_old_cards(base_enc, t_heads, seed, device, with_eval_task=False)
    ref = R4_BASELINE / f"r4_baseline_s{seed}.json"
    assert ref.exists(), f"缺 step-0 参照：{ref}"
    ref_json = json.loads(ref.read_text(encoding="utf-8"))
    ok = ref_json["combined_sha256"] == r4_base["combined_sha256"]
    print(f"[T5 step-0] 与 r4_check 基线比对：{'相等 ✅' if ok else '不相等 ❌'} "
          f"{r4_base['combined_sha256']} vs {ref_json['combined_sha256']}", flush=True)
    assert ok, "step-0 老卡基线与 r4_check 不逐位相同 ⇒ 该 seed 作废"
    r4_post = eval_old_cards(model.encoder, t_heads, seed, device, with_eval_task=False)
    del t_heads
    r5 = {n: {"step0": r4_base[n]["exact"], "post": r4_post[n]["exact"],
              "delta": r4_post[n]["exact"] - r4_base[n]["exact"], "band": BAND[n]}
          for n in OLD_CARDS}
    del base_enc
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    print("R5_OLD " + json.dumps({n: {"step0": round(v["step0"], 6),
                                      "post": round(v["post"], 6),
                                      "delta": round(v["delta"], 6),
                                      "band": v["band"]} for n, v in r5.items()},
                                 ensure_ascii=False), flush=True)

    drift = core_drift(model.encoder)
    if CORE_POLICY[arm] is None:
        assert drift == 0.0, f"{arm}: 核全冻结却漂移 {drift} ⇒ 冻结是假的"
    else:
        assert drift > 0, f"{arm}: 核漂移 = {drift} ⇒ 核一步没动（空测试）"
    print(f"R6_DRIFT rel={drift:.6f} sec/step_steady={sec_steady:.4f} peak_mem_mb={peak_mb:.1f}",
          flush=True)

    name = f"{arm}{tag}_s{seed}{'_rand' if randlabel else ''}"
    out = {
        "name": name, "arm": arm, "seed": seed, "data_arm": "clean",
        "randlabel": randlabel, "randlabel_n_diff": randlabel_n_diff,
        "device": device, "inventory": inv, "policy": pol,
        "old_heads_freeze": old_rep, "mount_evidence": mount_ev,
        "spec": {"max_len_ctx": spec.max_len_ctx, "max_len_cand": spec.max_len_cand,
                 "head_hidden": spec.head_hidden, "hidden": spec.hidden},
        "recipe": {"steps": steps, "batch": BATCH, "lr_head": LR_HEAD,
                   "lr_core": (LR_CORE if core_train else None), "lr_groups": lr_groups,
                   "weight_decay": WD, "cosine": True, "grad_clip": CLIP,
                   "n_train": n_train, "old_tasks_in_batch": [],
                   "old_heads_mounted_during_training": False,
                   "encoder_eval_mode": True, "live": live},
        "eval": evals,
        "r4_base": {n: r4_base[n]["exact"] for n in OLD_CARDS},
        "r4_post": {n: r4_post[n]["exact"] for n in OLD_CARDS},
        "r5": r5, "core_drift": drift, "grad_step0": g_step0,
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
                if k.startswith(("encoder.", "head.", "adapter."))}, WEIGHTS / f"{name}.pt")
    print(f"[done] {name} → results/{name}.json  ({out['wall_sec']}s)", flush=True)
    return out


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
