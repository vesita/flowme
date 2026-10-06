#!/usr/bin/env python3
"""layer_selective 空测试自检（PREREG §4，跑前必过）。

逐项：
  1) 五臂逐臂**实测打印** requires_grad=True 的参数名 + 数量（与 PREREG §2 预期逐臂对账）
  2) 冻结实况：冻结侧参数名清单 + 反向后冻结参数 p.grad 必须全为 None
  3) 目标层梯度 > 0（EMB 只 embedding、TOP 只 blocks.2、ADPT 核 0 可训）
  4) 适配层 step-0 恒等 max|Δ| == 0（零初始化）
  5) 多数类基线：三个 split == 0.5000；两个 ctype 计数（627/623 ⇒ 0.5016）
  6) 随机标签对照真执行：1:1 不变、与真标签不同条数 > 0
  7) RNG/口径同源：五臂 step-0 loss_first 都 == 0.693002 / 0.692196（同 seed）
  8) 缓存 vs live：核冻结臂读缓存，与 live 编码的 logits max|Δ|（只报不判）

用法：uv run python experiments/layer_selective/selftest.py
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from train_ls import (  # noqa: E402  (先 import 它 ⇒ 把 select_semantic_joint 加进 sys.path)
    ARMS, EXPECT_LOSS_FIRST, EXPECT_TRAINABLE, LIVE, LSModel, apply_policy,
    grad_stats, inventory)
from common import (  # noqa: E402  (select_semantic_joint/common.py —— 只读 import)
    BATCH, EXPECTED_N, SemSpec, data_fingerprint, inputs_for, load_blob, load_rows)

RESULTS = HERE / "results"


def build(arm: str, seed: int, device: str) -> LSModel:
    random.seed(seed)
    torch.manual_seed(seed)
    spec = SemSpec.from_build_spec()
    model = LSModel(spec, with_adapter=(arm == "ADPT")).to(device)
    apply_policy(model, arm)
    model.eval()
    return model


def first_batch(dl) -> torch.Tensor:
    for j in dl:
        j = j[0] if isinstance(j, (list, tuple)) else j
        return j
    raise SystemExit("空 dataloader")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fp = data_fingerprint()
    rows = load_rows()
    blob = load_blob()
    out: dict = {"device": device, "arms": {}, "data": {"fingerprint": fp}}

    n_train = EXPECTED_N["train"]
    train_idx = torch.nonzero(blob["splits"] == 0, as_tuple=False).squeeze(-1)

    # ---- 5) 多数类基线（PREREG §4.5） ----
    maj = {}
    for i, s in enumerate(("train", "test", "adv")):
        idx = torch.nonzero(blob["splits"] == i, as_tuple=False).squeeze(-1)
        y = blob["labels"][idx]
        m = float(torch.bincount(y, minlength=2).float().max() / len(idx))
        assert abs(m - 0.5) < 1e-9, f"{s} 多数类基线 {m} ≠ 0.5000"
        maj[s] = round(m, 6)
    adv_idx = torch.nonzero(blob["splits"] == 2, as_tuple=False).squeeze(-1)
    ctype_n, ctype_pos = {}, {}
    for code in (1, 2):
        sub = adv_idx[blob["ctype"][adv_idx] == code]
        y = blob["labels"][sub]
        ctype_n[code] = int(len(sub))
        ctype_pos[code] = int(y.sum())
    assert set(ctype_n.values()) == {1250}, f"ctype 子集大小变了：{ctype_n}"
    assert sorted(ctype_pos.values()) == [623, 627], f"ctype 正例数变了：{ctype_pos}"
    out["data"]["majority"] = maj
    out["data"]["adv_ctype_n"] = ctype_n
    out["data"]["adv_ctype_pos"] = ctype_pos
    out["data"]["adv_ctype_majority"] = round(max(ctype_pos.values()) / 1250, 6)
    print(f"[5] 多数类基线 {maj}；adv ctype n={ctype_n} 正例={ctype_pos} "
          f"⇒ 多数类 {out['data']['adv_ctype_majority']}", flush=True)

    # ---- 6) 随机标签对照真执行（PREREG §4.6） ----
    labels = blob["labels"].clone()
    g = torch.Generator().manual_seed(42 * 1000 + 7)
    ix = torch.arange(n_train)
    labels[ix] = labels[ix][torch.randperm(n_train, generator=g)]
    cnt = torch.bincount(labels[:n_train], minlength=2).tolist()
    n_diff = int((labels[:n_train] != blob["labels"][:n_train]).sum())
    assert cnt == [n_train // 2, n_train - n_train // 2], f"随机标签计数 {cnt}"
    assert n_diff > 0, "随机标签没有改变任何标签（空对照）"
    out["randlabel"] = {"counts": cnt, "n_diff": n_diff, "seed": 42}
    print(f"[6] 随机标签 1:1 {cnt}，与真标签不同 {n_diff} 条", flush=True)

    # ---- 1~4、7、8：逐臂 ----
    for arm in ARMS:
        for seed in (42, 43):
            model = build(arm, seed, device)
            inv = inventory(model, arm)
            live = LIVE[arm]
            gen = torch.Generator().manual_seed(seed)
            dl = DataLoader(torch.utils.data.TensorDataset(train_idx), batch_size=BATCH,
                            shuffle=True, generator=gen, drop_last=False)
            j = first_batch(dl)

            # 7) step-0 loss 同源（构造顺序 / RNG 流没被改过）
            c, mc, d, md = inputs_for(model, blob, rows, j, live, device)
            y = blob["labels"][j].to(device)
            logits = model.forward_tokens(c, mc, d, md)
            loss_t = torch.nn.functional.cross_entropy(logits, y)
            loss0 = float(loss_t.detach())
            exp = EXPECT_LOSS_FIRST[seed]
            assert round(loss0, 6) == exp, f"{arm} seed{seed}: loss_first {round(loss0, 6)} ≠ {exp}"

            # 4) 适配层 step-0 恒等
            ad_max = None
            if model.adapter is not None:
                with torch.no_grad():
                    x = torch.randn(5, 9, model.spec.hidden, device=device)
                    ad_max = float((model.adapter(x) - x).abs().max())
                assert ad_max == 0.0, f"适配层不恒等：{ad_max}"

            # 2/3) 反向 → 冻结侧必须没有任何梯度、可训侧必须有
            loss_t.backward()
            gs = grad_stats(model)
            assert not gs["trainable_grad_none"], f"{arm}: 可训参数无梯度 {gs['trainable_grad_none']}"
            assert not gs["frozen_with_grad"], f"{arm}: 冻结参数收到梯度 {gs['frozen_with_grad']}"
            assert gs["trainable_grad_norm"] > 0, f"{arm}: 可训侧梯度范数 0"
            model.zero_grad(set_to_none=True)

            # 2b) 核冻结臂（F/ADPT）额外走一次 **live 前向** 证明「核在图里也没梯度」
            live_core_probe = None
            if not live:
                c2, mc2, d2, md2 = inputs_for(model, blob, rows, j, True, device)
                l2 = torch.nn.functional.cross_entropy(
                    model.forward_tokens(c2, mc2, d2, md2), y)
                l2.backward()
                gs2 = grad_stats(model)
                enc_grad = [n for n, p in model.encoder.named_parameters() if p.grad is not None]
                assert enc_grad == [], f"{arm}: live 前向下核收到梯度 {enc_grad[:5]}"
                assert not gs2["trainable_grad_none"] and gs2["trainable_grad_norm"] > 0
                live_core_probe = {"encoder_params_with_grad": enc_grad,
                                   "trainable_grad_norm": round(gs2["trainable_grad_norm"], 6),
                                   "loss_live": round(float(l2.detach()), 6)}
                model.zero_grad(set_to_none=True)

            # 8) 缓存 vs live（只报不判）
            cache_live_delta = None
            if not live:
                with torch.no_grad():
                    a = model.forward_tokens(
                        *(inputs_for(model, blob, rows, j, False, device)))
                    b = model.forward_tokens(
                        *(inputs_for(model, blob, rows, j, True, device)))
                    cache_live_delta = float((a - b).abs().max())

            rec = {"n_trainable": inv["n_trainable"], "expected": EXPECT_TRAINABLE[arm],
                   "trainable_names": [x["name"] for x in inv["trainable"]],
                   "n_frozen": inv["n_frozen"],
                   "encoder_trainable": inv["encoder_trainable"],
                   "adapter_params": inv["adapter_params"],
                   "loss_first": round(loss0, 6), "expected_loss_first": exp,
                   "grad_norm": round(gs["trainable_grad_norm"], 6),
                   "frozen_with_grad": gs["frozen_with_grad"],
                   "adapter_identity_max": ad_max,
                   "live_core_probe": live_core_probe,
                   "cache_vs_live_max_abs_diff": cache_live_delta,
                   "live": live}
            out["arms"].setdefault(arm, {})[f"s{seed}"] = rec
            print(f"[1-4/7/8] {arm} s{seed}: 可训 {inv['n_trainable']} "
                  f"(核可训 {inv['encoder_trainable']} / 头 {inv['head_params']} / "
                  f"适配 {inv['adapter_params']})，冻结 {inv['n_frozen']}，"
                  f"loss_first {round(loss0, 6)}，|g|={gs['trainable_grad_norm']:.4f}，"
                  f"冻结侧有梯度的参数 {len(gs['frozen_with_grad'])} 个，"
                  f"adapter 恒等 {ad_max}，cache-vs-live {cache_live_delta}", flush=True)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "selftest.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
    print("SELFTEST_DONE", flush=True)


if __name__ == "__main__":
    main()
