#!/usr/bin/env python3
"""select_semantic_joint —— 训练前自检（PREREG §4 的空测试，**全部实测**）。

1. 两臂可训参数量与**冻结实况**（核对臂核 requires_grad 全 False；
   主臂四张老卡头 requires_grad **全 False**，逐参数打印）；
2. 多数类基线：train/test/adv 与两个 ctype 全部 0.5000（1:1 被破坏即炸）；
3. 随机标签对照**真执行**：randperm 后计数仍 1:1、与真标签不同条数 > 0；
4. R4 基线文件就位（`r4_check.py` 产出，sha256 非空）；
5. **等价性实测**：joint 循环把核冻住跑 N 步 vs 只跑 select 跑 N 步
   ⇒ 打分头权重 `torch.equal`（这是「两臂只差核是否可训」这条的证据）；
6. **live 编码 vs 缓存**：同一批文本，live 与 `select_pool` 缓存的 max|Δ|（只报不判）。

用法：uv run python experiments/select_semantic_joint/selftest.py [--steps 20]
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from common import (  # noqa: E402
    BATCH, CLIP, EXPECTED_N, HEAD_PARAMS, ENC_PARAMS, LR_CORE, LR_HEAD, OLD_CARDS,
    RESULTS, SEEDS, SPLITS, STEPS, WD,
    SemModel, SemSpec, _encode_ids, eval_old_cards, freeze_report_old, inputs_for,
    load_blob, load_old_heads, load_rows, old_loaders, old_loss)
from train_sem import IndexDataset  # noqa: E402


def build(seed: int, device: str, freeze_core: bool) -> SemModel:
    random.seed(seed)
    torch.manual_seed(seed)
    return SemModel(SemSpec.from_build_spec(), freeze_core=freeze_core).to(device)


def run(model: SemModel, joint_loop: bool, blob: dict, rows, labels, train_idx,
        heads, old_dl, steps: int, device: str) -> dict:
    """joint_loop=True ⇒ 每步加 4 个老任务批次（核冻结时它对可训参数梯度恒为 0）。"""
    gen = torch.Generator().manual_seed(42)
    dl = DataLoader(IndexDataset(train_idx), batch_size=BATCH, shuffle=True,
                    generator=gen, drop_last=False)
    core_params = list(model.encoder.parameters())
    head_params = list(model.head.parameters())
    params = [p for p in model.parameters() if p.requires_grad]
    if joint_loop:
        opt = torch.optim.AdamW([{"params": core_params, "lr": LR_CORE},
                                 {"params": head_params, "lr": LR_HEAD}],
                                weight_decay=WD)
        clip_targets = core_params + head_params
    else:
        opt = torch.optim.AdamW(params, lr=LR_HEAD, weight_decay=WD)
        clip_targets = params
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    iters = {n: iter(old_dl[n]) for n in OLD_CARDS}
    step = 0
    while step < steps:
        for j in dl:
            j = j[0] if isinstance(j, (list, tuple)) else j
            c, mc, d, md = inputs_for(model, blob, rows, j, False, device)   # 缓存（两臂同输入）
            y = labels[j].to(device)
            loss = torch.nn.functional.cross_entropy(model.forward_tokens(c, mc, d, md), y)
            total = loss
            if joint_loop:
                for n in OLD_CARDS:
                    try:
                        b = next(iters[n])
                    except StopIteration:
                        iters[n] = iter(old_dl[n])
                        b = next(iters[n])
                    total = total + old_loss(heads, model.encoder, b, n, device)
            opt.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(clip_targets, CLIP)
            opt.step()
            sched.step()
            step += 1
            if step >= steps:
                break
    return {"steps": step, "params": sum(p.numel() for p in params)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42, choices=list(SEEDS))
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed = a.seed
    out: dict = {"device": device, "seed": seed, "checks": {}}

    # ---- 1. 参数量 / 冻结实况（实测，不声明） ----
    m_free = build(seed, device, freeze_core=True)
    rep_f = m_free.freeze_report()
    assert rep_f["encoder_trainable"] == 0 and rep_f["trainable_params"] == HEAD_PARAMS
    m_j = build(seed, device, freeze_core=False)
    rep_j = m_j.freeze_report()
    assert rep_j["encoder_trainable"] == ENC_PARAMS
    assert rep_j["trainable_params"] == HEAD_PARAMS + ENC_PARAMS
    heads = load_old_heads(seed, device)
    old_rep = freeze_report_old(heads)
    out["checks"]["params"] = {"frozen": rep_f, "joint": rep_j, "old_heads": old_rep}
    print(f"[1] frozen 可训={rep_f['trainable_params']}（核 {rep_f['encoder_trainable']}） "
          f"joint 可训={rep_j['trainable_params']}（核 {rep_j['encoder_trainable']}）", flush=True)
    print(f"[1] 老卡头冻结实测 = {json.dumps(old_rep, ensure_ascii=False)}", flush=True)
    del m_j
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # ---- 数据 / 多数类 / 随机标签 ----
    rows = load_rows()
    blob = load_blob()
    labels = blob["labels"].clone()
    maj = {}
    for i, s in enumerate(SPLITS):
        sel = blob["splits"] == i
        n = int(sel.sum())
        assert n == EXPECTED_N[s], f"{s} n={n}"
        cnt = torch.bincount(blob["labels"][sel], minlength=2)
        m = float(cnt.max()) / n
        maj[s] = {"n": n, "counts": cnt.tolist(), "majority": round(m, 6)}
        assert abs(m - 0.5) < 1e-9, f"{s} 多数类 {m} ≠ 0.5"
    for code, name in ((1, "heldout_pair"), (2, "shifted_pos")):
        sel = (blob["splits"] == 2) & (blob["ctype"] == code)
        n = int(sel.sum())
        assert n == 1250, f"{name} n={n}"
        cnt = torch.bincount(blob["labels"][sel], minlength=2).tolist()
        m = float(max(cnt)) / n
        maj[name] = {"n": n, "counts": cnt, "majority": round(m, 6)}
        assert abs(m - 0.5) < 0.01, f"{name} 多数类 {m} 偏离 0.5 太多：{cnt}"
    out["checks"]["majority"] = maj
    print(f"[2] 多数类基线（split=0.5000 / ctype=0.5016，627:623）："
          f"{json.dumps(maj, ensure_ascii=False)}", flush=True)

    n_train = EXPECTED_N["train"]
    g = torch.Generator().manual_seed(seed * 1000 + 7)
    idx = torch.arange(n_train)
    rl = labels.clone()
    rl[idx] = rl[idx][torch.randperm(n_train, generator=g)]
    cnt = torch.bincount(rl[:n_train], minlength=2).tolist()
    n_diff = int((rl[:n_train] != blob["labels"][:n_train]).sum())
    assert cnt == [n_train // 2, n_train - n_train // 2] and n_diff > 0
    out["checks"]["randlabel"] = {"counts": cnt, "n_diff": n_diff, "executed": True}
    print(f"[3] 随机标签真执行：计数 {cnt}，与真标签不同 {n_diff} 条", flush=True)

    # ---- 4. R4 基线就位 ----
    r4 = {}
    for s in SEEDS:
        p = RESULTS / f"r4_baseline_s{s}.json"
        assert p.exists(), f"缺 R4 基线：{p}（先跑 r4_check.py）"
        r4[str(s)] = json.loads(p.read_text(encoding="utf-8"))["combined_sha256"]
        assert r4[str(s)], "R4 基线 sha256 为空"
    out["checks"]["r4_baseline"] = r4
    print(f"[4] R4 基线 sha256 就位：{json.dumps(r4, ensure_ascii=False)}", flush=True)

    # ---- 5. 等价性：joint 循环冻核 ≡ 只跑 select（打分头逐位相同）----
    train_idx = torch.nonzero(blob["splits"] == 0, as_tuple=False).squeeze(-1)
    old_dl = old_loaders(seed)
    ma = build(seed, device, freeze_core=True)
    ra = run(ma, True, blob, rows, labels, train_idx, heads, old_dl, a.steps, device)
    w_joint = {k: v.detach().cpu().clone() for k, v in ma.head.state_dict().items()}
    del ma
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    mb = build(seed, device, freeze_core=True)
    rb = run(mb, False, blob, rows, labels, train_idx, heads, old_dl, a.steps, device)
    w_froz = {k: v.detach().cpu().clone() for k, v in mb.head.state_dict().items()}
    del mb
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    diff = {k: float((w_joint[k] - w_froz[k]).abs().max()) for k in w_joint}
    eq = all(bool(torch.equal(w_joint[k], w_froz[k])) for k in w_joint)
    out["checks"]["equivalence"] = {"steps": a.steps, "joint": ra, "frozen": rb,
                                    "bitwise_equal": eq, "max_abs_diff_per_tensor": diff}
    print(f"[5] 等价性（{a.steps} 步）：joint-冻核 vs 只跑 select ⇒ "
          f"打分头逐位相同 = {eq}，max|Δ| = {max(diff.values()):.3e}", flush=True)
    assert eq, f"等价性不成立（两臂不再只差核是否可训）：{diff}"

    # ---- 6. live vs cache ----
    sample = train_idx[:BATCH]
    h1, m1, d1, md1 = inputs_for(m_free, blob, rows, sample, False, device)
    h2, m2, d2, md2 = inputs_for(m_free, blob, rows, sample, True, device)
    d_h = max(float((h1 - h2).abs().max()), float((d1 - d2).abs().max()))
    d_m = bool(torch.equal(m1, m2)) and bool(torch.equal(md1, md2))
    out["checks"]["live_vs_cache"] = {"max_abs_diff": d_h, "mask_equal": d_m,
                                      "n": int(len(sample))}
    print(f"[6] live vs cache：max|Δ|={d_h:.3e} mask 相等={d_m}", flush=True)

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "selftest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("[save] results/selftest.json")
    print("SELFTEST_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
