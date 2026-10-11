#!/usr/bin/env python3
"""S86g · L3-m2「中英模因锚定」驱动：4 张卡 + AaBb/AbBa 映射（口径见 `PREREG_ZH_EN.md`）。

    --prepare   建语料缓存            --smoke     自检（S1–S5）
    --run LAM SEED [--maps AaBb,AbBa] [--loss none|infonce] [--steps N] [--eval-every N]
    --tables    报告表格              --verdict   按 PREREG §3 出判词（相对 λ=0 基线）
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/home/vesita/coding/my/flowme")
sys.path.insert(0, str(ROOT / "experiments" / "meme_anchor"))
import data as D          # noqa: E402
import data_zh_en as DZ   # noqa: E402
import measure as MS      # noqa: E402
import model_zh_en as MZ  # noqa: E402
from model import make_pairs   # noqa: E402

_spec = importlib.util.spec_from_file_location("m1drv", ROOT / "stages" / "86_meme_anchor.py")
DRV = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(DRV)          # 复用 align_extra / slot_matrix / d_slot（同一份实现）

VOCAB, BOS, EOS, PAD = D.VOCAB, D.BOS, D.EOS, D.PAD
LOGS = ROOT / "logs"
TRAIN_N, TEST_N, SPLIT_SEED = 40000, 2245, 20261011
BS_DEFAULT, STEPS_DEFAULT, EVAL_EVERY = 32, 8000, 1000
ALL_MAPS = ("AaBb", "AbBa")


def log(msg: str) -> None:
    print(msg, flush=True)


def js(d: dict) -> dict:
    """与 L3-m1 用**同一份**实现（含"丢掉逐样本张量 `delta_i`"的处理）—— R47：一个事实一个实现。"""
    return DRV.js(d)


_PACK: dict | None = None


def get_pack() -> dict:
    global _PACK
    if _PACK is None:
        pairs = DZ.load_pairs()
        g = torch.Generator().manual_seed(SPLIT_SEED)
        pairs = [pairs[i] for i in torch.randperm(len(pairs), generator=g).tolist()]
        train = pairs[:TRAIN_N]
        test = pairs[TRAIN_N:TRAIN_N + TEST_N]
        tk = D.load_tokenizer()
        enc = lambda s: tk.encode(s).ids                                    # noqa: E731
        _PACK = dict(train=train, test=test,
                     train_ids=[(enc(z), enc(e)) for z, e in train],
                     test_ids=[(enc(z), enc(e)) for z, e in test])
    return _PACK


def pad_ids(ids_list: list[list[int]], device) -> torch.Tensor:
    L = max(len(s) for s in ids_list)
    out = torch.full((len(ids_list), L), PAD, dtype=torch.long)
    for i, s in enumerate(ids_list):
        out[i, :len(s)] = torch.tensor(s)
    return out.to(device)


def _peak_mib(device) -> float:
    """当前 CUDA 峰值显存（MiB）；CPU 或不可用时返回 nan。

    ★ 用途（用户 2026-10-12 强调的"卡片化第三条目标"）：每个 run 都自带显存口径，
      这样"降显存 / 按需加载 ⇒ 等效提升参数量"这条目标不会在结论里漂掉（`MEME_FRAMEWORK §1.1`）。
    """
    if str(device).startswith("cuda") and torch.cuda.is_available():
        return float(torch.cuda.max_memory_allocated()) / 2 ** 20
    return float("nan")


def _rollout_ce(model, crit, lang, meme, t_in, t_out):
    """rollout CE：用**模型自己的 argmax** 当后续位置的输入（stop-gradient），CE 照算。

    - pass 1 在 `model.eval()` + `no_grad()` 下取 argmax（**不消耗 dropout RNG**）
    - pass 2 包在 `torch.random.fork_rng()` 里（dropout 照常生效，但**不消耗全局 RNG**）
    ⇒ 与 `--obj tf` 相比**全局 RNG 消耗逐位相同**（R34；PREREG_GENOBJ §1 / 自检 N6）

    返回 `(ce, r_in, pred, lg1)` —— 后三者只给 `--genobj-selfcheck` 的 N4/N5 断言用。
    """
    was_training = model.training
    model.eval()
    with torch.no_grad():
        lg1 = model.logits(lang, meme, t_in)
        pred = lg1.argmax(-1)
    if was_training:
        model.train()
    r_in = torch.cat([t_in[:, :1], pred[:, :-1]], dim=1)      # [BOS, pred_0, pred_1, ...]
    with torch.random.fork_rng():
        lg2 = model.logits(lang, meme, r_in)
    return crit(lg2.reshape(-1, VOCAB), t_out.reshape(-1)), r_in, pred, lg1


def _terms(model, crit, z_list, e_list, maps, device, obj: str = "tf"):
    """按映射集合算各项 CE。返回 (各项 CE, 输入模因 mz/me, **衔接模因** bz/be, rollout 各项 CE 或 None)。

    ★ 所有输出卡读的都是**衔接处**的模因 `bz/be = t(mz/me)`（`bridge="none"` 时 t 是恒等 ⇒ 与 L3-m2 逐位一致）。
       用户 2026-10-11 的写法 Ata/Btb/Atb/Bta 就是"输入卡 → t → 输出卡"，t 两语言共用。
    ★ `obj=="roll"` 时**额外**算一份 rollout CE（复用同一份 mz/me/bz/be 与同一批数据 ⇒ 不重复前向输入卡）。
    """
    mz = model.meme_zh(pad_ids(z_list, device))
    me = model.meme_en(pad_ids(e_list, device))
    bz, be = model.apply_bridge(mz, "zh"), model.apply_bridge(me, "en")
    tz_in, tz_out = make_pairs(z_list, device)
    te_in, te_out = make_pairs(e_list, device)
    t: dict[str, torch.Tensor] = {}
    if "AaBb" in maps:                                    # 自映射：中→中、英→英
        t["zz"] = crit(model.logits("zh", bz, tz_in).reshape(-1, VOCAB), tz_out.reshape(-1))
        t["ee"] = crit(model.logits("en", be, te_in).reshape(-1, VOCAB), te_out.reshape(-1))
    if "AbBa" in maps:                                    # 互映射：中→英、英→中（金标=配对句）
        t["ze"] = crit(model.logits("en", bz, te_in).reshape(-1, VOCAB), te_out.reshape(-1))
        t["ez"] = crit(model.logits("zh", be, tz_in).reshape(-1, VOCAB), tz_out.reshape(-1))
    roll = None
    if obj == "roll":
        roll = {}
        if "AaBb" in maps:
            roll["zz"] = _rollout_ce(model, crit, "zh", bz, tz_in, tz_out)[0]
            roll["ee"] = _rollout_ce(model, crit, "en", be, te_in, te_out)[0]
        if "AbBa" in maps:
            roll["ze"] = _rollout_ce(model, crit, "en", bz, te_in, te_out)[0]
            roll["ez"] = _rollout_ce(model, crit, "zh", be, tz_in, tz_out)[0]
    return t, mz, me, bz, be, roll


def _combine_loss(t, roll, mz, me, bz, be, loss_kind, lam, abba_weight,
                  obj, roll_w, tau=0.07, mu=1.0, gamma=0.2, align_on="bridge"):
    """训练用的损失装配（单点）：`0.5·(mean_aa + α·mean_ab) + 对齐项 [+ roll_w·mean(rollout 项)]`。

    ★ 抽成函数是为了让 `--genobj-selfcheck` 的 **N1（roll_w=0 ⇒ 与 tf 逐位相同）** 驱动**同一份表达式**，
      而不是另写一遍（"判据写死了 ≠ 判据接上了线"）。
    """
    aa = [v for k, v in t.items() if k in ("zz", "ee")]        # AaBb（自映射）
    ab = [v for k, v in t.items() if k in ("ze", "ez")]        # AbBa（互映射）
    loss = (0.5 * (sum(aa) / max(len(aa), 1) + abba_weight * sum(ab) / max(len(ab), 1))
            + DRV.align_extra(*((bz, be) if align_on == "bridge" else (mz, me)),
                              loss_kind, lam, tau, mu, gamma))
    if obj == "roll":
        loss = loss + roll_w * (sum(roll.values()) / max(len(roll), 1))
    return loss

@torch.no_grad()
@torch.no_grad()
def eval_model(model, pack, device, chunk: int = 256) -> dict:
    """held-out 上评 4 条映射 + **两个衔接点**的模因：输入侧 `m_*`、衔接侧 `b_* = t(m_*)`。"""
    zs = [p[0] for p in pack["test_ids"]]
    es = [p[1] for p in pack["test_ids"]]
    keys = ("ce_zz", "ce_ee", "ce_ze", "ce_ez", "char_zz", "char_ee", "char_ze", "char_ez",
            "exact_zz", "exact_ee", "exact_ze", "exact_ez")
    acc: dict[str, list] = {k: [] for k in keys}
    m_zh, m_en, b_zh, b_en = [], [], [], []
    for i in range(0, len(zs), chunk):
        zl, el = zs[i:i + chunk], es[i:i + chunk]
        _, mz, me, bz, be, _ = _terms(model, nn.CrossEntropyLoss(ignore_index=PAD), zl, el, ALL_MAPS, device)
        m_zh.append(mz.cpu()); m_en.append(me.cpu()); b_zh.append(bz.cpu()); b_en.append(be.cpu())
        tz_in, tz_out = make_pairs(zl, device)
        te_in, te_out = make_pairs(el, device)
        logs = {"zz": (model.logits("zh", bz, tz_in), tz_out),
                "ee": (model.logits("en", be, te_in), te_out),
                "ze": (model.logits("en", bz, te_in), te_out),
                "ez": (model.logits("zh", be, tz_in), tz_out)}
        for nm, (lg, tgt) in logs.items():
            mk = tgt != PAD
            pred = lg.argmax(-1)
            acc[f"ce_{nm}"].append(float(nn.functional.cross_entropy(
                lg.reshape(-1, VOCAB), tgt.reshape(-1)).item()))
            acc[f"char_{nm}"].append(float(((pred == tgt) & mk).sum()) / max(int(mk.sum()), 1))
            acc[f"exact_{nm}"].append(float((((pred == tgt) | ~mk).all(1)).float().mean()))
    out = {k: float(np.mean([x for x in v if x == x])) for k, v in acc.items()}
    out["m_zh"], out["m_en"] = torch.cat(m_zh), torch.cat(m_en)
    out["b_zh"], out["b_en"] = torch.cat(b_zh), torch.cat(b_en)
    # ★ 分化度按**衔接处**算（那是"输出衔接"的模因；输入侧的分化度另有读数）
    out["spread_zh"] = float(out["b_zh"].mean(1).std(dim=0).mean())
    out["spread_en"] = float(out["b_en"].mean(1).std(dim=0).mean())
    out["spread_in_zh"] = float(out["m_zh"].mean(1).std(dim=0).mean())
    out["spread_in_en"] = float(out["m_en"].mean(1).std(dim=0).mean())
    return out

def _junction(A, B, pack, seed: int) -> dict:
    """**一个衔接点**上的全套口径：Δ / 长度匹配(M8) / 逐槽+M3 / CKA / 线性映射 / 地板 / AUROC+阈值 / 繁简子集。"""
    zh_s = [p[0] for p in pack["test"]]
    en_s = [p[1] for p in pack["test"]]
    zh_i = [p[0] for p in pack["test_ids"]]
    en_i = [p[1] for p in pack["test_ids"]]
    pA, pB = MS.pooled(A), MS.pooled(B)
    S = MS.cosmat(pA, pB)
    rep_pool = MS.score_report(S, "pooled")
    rep_slot = MS.score_report(DRV.slot_matrix(A, B), "slot")
    cka = MS.cka_linear(pA, pB)
    cka_m, cka_se = MS.cka_perm_null(pA, pB, 50, seed)
    lin_id, lin_map = MS.linmap(pA, pB)
    _, lin_map_rand = MS.linmap(pA, pB, derange=True)
    rep_shuf = MS.score_report(DRV.slot_matrix(MS.shuffle_slots(A, seed + 1),
                                               MS.shuffle_slots(B, seed + 2)), "slot_shuffled")
    n = pA.size(0)
    rep_rand = MS.score_report(MS.cosmat(MS.pooled(MS.floor_random(n, seed + 3)),
                                         MS.pooled(MS.floor_random(n, seed + 4))), "floor_random")
    cz, ce = MS.floor_counts(zh_s), MS.floor_counts(en_s)
    nz = cz / cz.norm(dim=1, keepdim=True).clamp_min(1e-12)
    ne = ce / ce.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rep_cnt = MS.score_report(MS.cosmat(nz, ne), "floor_counts")
    bz = MS.floor_bag_of_chars(zh_i)
    be = MS.floor_bag_of_chars(en_i)
    bz = bz / bz.norm(dim=1, keepdim=True).clamp_min(1e-12)
    be = be / be.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rep_bag = MS.score_report(MS.cosmat(bz, be), "reference_bag_of_chars")
    m8 = MS.length_matched_delta(S, [len(e) // 10 for e in en_s])       # 按英文长度桶匹配负样本
    drop = rep_slot["delta"] - rep_shuf["delta"]
    drop_se = math.sqrt(rep_slot["delta_se"] ** 2 + rep_shuf["delta_se"] ** 2)
    Sc = S - S.mean(1, keepdim=True)                                     # 行中心化（原始余弦不可用）
    off = Sc[~torch.eye(n, dtype=torch.bool)]
    au = MS.auroc_rank(Sc.diagonal(), off)
    thr = MS.youden_threshold(Sc.diagonal(), off)
    subsets = {}
    for nm, marker in (("simp", DZ.SIMP), ("trad", DZ.TRAD)):
        other = DZ.TRAD if nm == "simp" else DZ.SIMP
        ix = [i for i, z in enumerate(zh_s) if any(c in marker for c in z) and not any(c in other for c in z)]
        if len(ix) >= 50:
            st = js(MS.pair_stats(S[torch.tensor(ix)][:, torch.tensor(ix)]))
            subsets[nm] = dict(n_sub=len(ix), **{k: v for k, v in st.items() if k != "n"})
    return dict(pooled=js(rep_pool), slot=js(rep_slot), slot_shuffled=js(rep_shuf),
                cka=cka, cka_perm=cka_m, cka_perm_se=cka_se, cka_delta=cka - cka_m,
                lin_identity=js(lin_id), lin_mapped=js(lin_map), lin_mapped_rand=js(lin_map_rand),
                floor_random=js(rep_rand), floor_counts=js(rep_cnt), reference_bag=js(rep_bag),
                m8_length_matched=js(m8),
                m3=dict(delta_slot=rep_slot["delta"], delta_shuffled=rep_shuf["delta"], drop=drop,
                        drop_se=drop_se, pass_drop=bool(drop >= max(2 * drop_se, 0.01))),
                surface_gate=bool(rep_pool["delta"] >= rep_cnt["delta"] + max(2 * rep_cnt["delta_se"], 0.01)),
                auroc_centered=au, thr_centered=js(thr), subsets=subsets,
                ctrl_zh_zh_diff_content=float((MS.cosmat(pA, pA) * (1 - torch.eye(n))).sum() /
                                              max(n * (n - 1), 1)))


def full_metrics(ev: dict, pack: dict, seed: int) -> dict:
    """★ **两个衔接点**各算一遍：
    `input` = 输入卡的输出模因（冻结/未冻结都一样测）；`bridge` = t 的输出（用户说的"输出衔接"）。
    """
    out = {"input": _junction(ev["m_zh"], ev["m_en"], pack, seed)}
    if "b_zh" in ev:
        out["bridge"] = _junction(ev["b_zh"], ev["b_en"], pack, seed + 100)
    return out

@torch.no_grad()
def gen_eval(model, pack, device, n: int = 200, bs: int = 50, maxlen: int = 160) -> dict:
    """贪心自回归：4 条映射各自的 EM / 字级准确率（不给任何目标前缀）。

    ★ 目标直接用 `pack["test_ids"]` 里的 **id**（早先版本每样本都重新 load_tokenizer ⇒ 慢到不可用）。
    """
    ids = pack["test_ids"][:n]
    out: dict[str, float] = {}
    for nm, (src_ix, gold_ix, lang) in {"zz": (0, 0, "zh"), "ee": (1, 1, "en"),
                                        "ze": (0, 1, "en"), "ez": (1, 0, "zh")}.items():
        em = cor = tot = 0
        for i in range(0, len(ids), bs):
            part = ids[i:i + bs]
            sl = [p[src_ix] for p in part]
            gl = [p[gold_ix] for p in part]
            meme = (model.meme_zh(pad_ids(sl, device)) if src_ix == 0
                    else model.meme_en(pad_ids(sl, device)))
            meme = model.apply_bridge(meme, "zh" if src_ix == 0 else "en")   # ★ 输出卡读衔接处模因
            cur = torch.full((len(sl), 1), BOS, dtype=torch.long, device=device)
            done = torch.zeros(len(sl), dtype=torch.bool, device=device)
            outs: list[list[int]] = [[] for _ in sl]
            for _ in range(maxlen):
                nxt = model.logits(lang, meme, cur)[:, -1].argmax(-1)
                for r in range(len(sl)):
                    if not done[r]:
                        if int(nxt[r]) == EOS or len(outs[r]) >= 150:
                            done[r] = True
                        else:
                            outs[r].append(int(nxt[r]))
                if bool(done.all()):
                    break
                cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
            for r, g in enumerate(gl):
                em += int(outs[r] == list(g))
                cor += sum(1 for a, b in zip(outs[r], g) if a == b)
                tot += len(g)
        out[f"gen_em_{nm}"] = em / max(len(ids), 1)
        out[f"gen_char_{nm}"] = cor / max(tot, 1)
    out["gen_n"] = len(ids)
    return out


def train_run(lam: float, seed: int, steps: int, maps: str, loss_kind: str, device: str,
              bs: int, eval_every: int, shared_out: bool = False, early_stop: bool = False,
              abba_weight: float = 1.0, bridge: str = "none", bridge_mid: int = 64,
              init_inputs: str = "", freeze_inputs: bool = False, init_outputs: bool = False,
              freeze_bridge: bool = False, align_on: str = "bridge",
              tau: float = 0.07, mu: float = 1.0, gamma: float = 0.2,
              obj: str = "tf", roll_w: float = 0.5, gen_eval_n: int = 1000) -> dict:
    mapset = tuple(m for m in maps.split(",") if m)
    # ★ fail-closed 守卫：位置错位/取值非法必须**当场**报错，而不是跑完 8000 步再崩在保存上
    assert bridge in ("none", "parser", "dual"), f"bridge 非法：{bridge!r}"
    assert align_on in ("bridge", "input"), f"align_on 非法：{align_on!r}（该收到 'bridge'/'input'）"
    assert obj in ("tf", "roll"), f"obj 非法：{obj!r}（该收到 'tf'/'roll'）"
    assert 0.0 <= float(roll_w) <= 10.0, f"roll_w 非法：{roll_w!r}"
    assert int(gen_eval_n) > 0, f"gen_eval_n 非法：{gen_eval_n!r}"
    for _nm, _v in (("freeze_inputs", freeze_inputs), ("init_outputs", init_outputs),
                    ("freeze_bridge", freeze_bridge), ("shared_out", shared_out)):
        assert isinstance(_v, bool), f"{_nm} 应为 bool，收到 {_v!r}"
    assert not init_inputs or Path(str(init_inputs)).is_file(), f"init_inputs 不是文件：{init_inputs!r}"
    pack = get_pack()
    model = MZ.build(seed, shared_out, bridge, bridge_mid).to(device)
    npar = model.n_params()
    log(f"[M2] run lam={lam} seed={seed} steps={steps} maps={mapset} loss={loss_kind} "
        f"abba_w={abba_weight} bridge={bridge}(mid={bridge_mid}) align_on={align_on} bs={bs} shared_out={shared_out} "
        f"params={npar} 总={sum(npar.values())}")
    frozen = []
    if init_inputs:
        blob = torch.load(init_inputs, map_location="cpu", weights_only=False)
        pref = ("zh_in.", "en_in.") + (("zh_out.", "en_out.") if init_outputs else ())
        sd = {k: v for k, v in blob["state"].items() if k.startswith(pref)}
        miss, unexp = model.load_state_dict(sd, strict=False)
        log(f"[M2] 载入输入卡 {init_inputs}：{len(sd)} 张量（missing={len(miss)} unexpected={len(unexp)}）")
    if freeze_inputs:
        for nm in ("zh_in", "en_in"):
            for prm in getattr(model, nm).parameters():
                prm.requires_grad_(False)
        frozen = ["zh_in", "en_in"]
    if freeze_bridge:                       # 臂 IV：随机初始化后**冻结**解析卡（同尺寸、不可训）
        for nm in ("bridge", "bridge_zh", "bridge_en"):
            c = getattr(model, nm, None)
            if c is not None:
                for prm in c.parameters():
                    prm.requires_grad_(False)
                frozen.append(nm)
    if frozen:
        log(f"[M2] 已冻结：{frozen}（可训参数 {sum(p.numel() for p in model.parameters() if p.requires_grad)}）")
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=1e-3, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    # ★ 显存账（`MEME_FRAMEWORK §1.1` 第三条目标）：参数占用 + 训练/评测/生成三段峰值
    mem_params_mib = sum(p.numel() * p.element_size() for p in model.parameters()) / 2 ** 20
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(pack["train_ids"]), generator=g).tolist()
    ptr, snaps, t0 = 0, [], time.time()
    step_time, stopped = float("nan"), "上限步数"
    for step in range(1, steps + 1):
        if ptr + bs > len(order):
            order = torch.randperm(len(pack["train_ids"]), generator=g).tolist(); ptr = 0
        idx = order[ptr:ptr + bs]; ptr += bs
        zl = [pack["train_ids"][j][0] for j in idx]
        el = [pack["train_ids"][j][1] for j in idx]
        t, mz, me, bz, be, roll = _terms(model, crit, zl, el, mapset, device, obj=obj)
        # ★ 0.5·(mean_aa + α·mean_ab)：α 只调"翻译压力 / 重构压力"的比例，且 **α=1 时恰好等于
        #   `sum(四项)/4`**（第一轮的实现）⇒ 复用第一轮 α=1.0 的 run 合法。
        #   若写成 `mean_aa + α·mean_ab`，α=1 会变成原来的 **2 倍**（等于偷偷给 CE 加倍）—— 已避开。
        #   ★ `obj=="roll"` 时由 `_combine_loss` 再加 `roll_w·mean(4 项 rollout CE)`（PREREG_GENOBJ §1）。
        loss = _combine_loss(t, roll, mz, me, bz, be, loss_kind, lam, abba_weight,
                             obj, roll_w, tau, mu, gamma, align_on)
        # ★ 对齐项加在哪一层由 --align-on 决定：bridge=t(m)（L3-m3c 的配置）· input=m（L3-m2 的位置）
        #   两者在"梯度能否穿透到输入卡"上不同 —— 见 PREREG_ALIGN_LOC.md 的 2×2 对照
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step == 20:
            step_time = (time.time() - t0) / 20
            log(f"[SPEED] lam={lam} seed={seed} 前 20 步 {step_time:.3f} s/step "
                f"⇒ {steps} 步预计 {step_time*steps/60:.1f} min")
        if step % eval_every == 0 or step == steps:
            ev = eval_model(model, pack, device)
            snap = dict(step=step, **{k: ev[k] for k in ev if k.startswith(("ce_", "char_", "exact_"))},
                        spread_zh=ev["spread_zh"], spread_en=ev["spread_en"])
            snaps.append(snap)
            log(f"[SNAP] lam={lam} seed={seed} step={step} 中→中 {snap['char_zz']*100:.1f}% "
                f"英→英 {snap['char_ee']*100:.1f}% 中→英 {snap['char_ze']*100:.1f}% "
                f"英→中 {snap['char_ez']*100:.1f}% 分化 {snap['spread_zh']:.3f}/{snap['spread_en']:.3f} "
                f"用时 {(time.time()-t0)/60:.1f}min")
            if early_stop and len(snaps) >= 3:
                # 收敛判据：**四项 CE 之和**（自映射 + 互映射都算，臂之间才可比）
                # ★ 只有"在降、但降得很慢"才算收敛。早先版本写作 `(b-a)/b < 0.01`，
                #   对"CE 上升"（负比值）也成立 ⇒ 6 个 run 全被截在 3000 步（实测 CE 和在升、精度在涨）。
                tot = lambda s: sum(v for k, v in s.items() if k.startswith("ce_"))   # noqa: E731
                slow = lambda x, y: 0.0 <= (x - y) / max(x, 1e-9) < 0.01            # noqa: E731
                if slow(tot(snaps[-2]), tot(snaps[-1])) and slow(tot(snaps[-3]), tot(snaps[-2])):
                    stopped = "收敛"; break
    peak_train_mib = _peak_mib(device)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    ev = eval_model(model, pack, device)
    peak_eval_mib = _peak_mib(device)
    metrics = full_metrics(ev, pack, seed)
    if str(device).startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    gen = gen_eval(model, pack, device, n=int(gen_eval_n))
    peak_gen_mib = _peak_mib(device)
    log(f"[MEM] 参数 {mem_params_mib:.1f} MiB（可训 {n_tr} / 总 {n_all}）· "
        f"峰值 训练 {peak_train_mib:.1f} / 评测 {peak_eval_mib:.1f} / 生成 {peak_gen_mib:.1f} MiB")
    log(f"[GEN] lam={lam} seed={seed} 贪心自回归(n={gen['gen_n']}) EM: 中→中 {gen['gen_em_zz']*100:.1f}% "
        f"英→英 {gen['gen_em_ee']*100:.1f}% 中→英 {gen['gen_em_ze']*100:.1f}% 英→中 {gen['gen_em_ez']*100:.1f}%")
    # ★ 载入标识必须进文件名：否则"同 bridge/同 seed 但载入不同 ckpt"的两次 run 会撞名互相覆盖（实测发生过）
    init_lab = ""
    if init_inputs:
        import re as _re
        init_lab = "_i-" + _re.sub(r"_l[0-9.]+_s\d+$", "", Path(init_inputs).stem).replace("86g_ckpt_", "")
    tag = (f"{loss_kind}_{'_'.join(mapset)}" + ("_shared" if shared_out else "")
           + ("" if abs(abba_weight - 1.0) < 1e-9 else f"_a{abba_weight}")
           + ("" if bridge == "none" else f"_{bridge}{bridge_mid}")
           + ("" if align_on == "bridge" else f"_ao-{align_on}")
           # ★ `o-{obj}` 必须进 tag：否则 tf 臂与 roll 臂的 ckpt/json 会同名**静默互相覆盖**（PREREG_GENOBJ §7）
           + f"_o-{obj}" + ("" if abs(roll_w - 0.5) < 1e-9 else f"_rw{roll_w}")
           + ("_frz" if freeze_inputs else "") + ("_io" if init_outputs else "")
           + ("_frzbr" if freeze_bridge else "") + init_lab)
    ck = LOGS / f"86g_ckpt_{tag}_l{lam}_s{seed}.pt"
    torch.save(dict(state=model.state_dict(), seed=seed, lam=lam, maps=mapset,
                    loss_kind=loss_kind, steps=step, shared_out=shared_out), ck)
    res = dict(lam=lam, seed=seed, steps=step, maps=list(mapset), loss_kind=loss_kind,
               shared_out=shared_out, abba_weight=abba_weight, bridge=bridge,
               bridge_mid=bridge_mid, align_on=align_on,
               obj=obj, roll_w=roll_w, gen_eval_n=int(gen_eval_n),
               mem=dict(params_mib=mem_params_mib, params_trainable=n_tr, params_total=n_all,
                        peak_train_mib=peak_train_mib, peak_eval_mib=peak_eval_mib,
                        peak_gen_mib=peak_gen_mib),
               init_inputs=str(init_inputs), init_outputs=init_outputs,
               freeze_bridge=freeze_bridge, frozen=frozen, tau=tau,
               stopped=stopped, step_time_s=step_time,
               wall_min=(time.time() - t0) / 60, n_params=npar,
               final={k: ev[k] for k in ev if k.startswith(("ce_", "char_", "exact_"))},
               final_spread=dict(zh=ev["spread_zh"], en=ev["spread_en"],
                                 in_zh=ev["spread_in_zh"], in_en=ev["spread_in_en"]),
               gen=gen, metrics=metrics, snaps=snaps)
    (LOGS / f"86g_result_{tag}_l{lam}_s{seed}.json").write_text(
        json.dumps(res, ensure_ascii=False), encoding="utf-8")
    log(f"[RUN-FINAL] lam={lam} seed={seed} 中→中 {ev['char_zz']*100:.2f}% 英→英 {ev['char_ee']*100:.2f}% "
        f"中→英 {ev['char_ze']*100:.2f}% 英→中 {ev['char_ez']*100:.2f}% | "
        f"Δ输入侧={metrics['input']['pooled']['delta']:+.4f} "
        f"**Δ衔接={metrics['bridge']['pooled']['delta']:+.4f}**"
        f"±{metrics['bridge']['pooled']['delta_se']:.4f} "
        f"长度匹配Δ(衔接)={metrics['bridge']['m8_length_matched']['delta']:+.4f} "
        f"| {stopped} | 墙钟 {(time.time()-t0)/60:.1f}min")
    return res


def verdict() -> None:
    """Q1–Q4（PREREG §3）：**按（损失 × 映射集合）分组**，各组相对基线 `none|AaBb`。

    ★ 早先版本把所有 `loss_kind=="none"` 的 run 混成一个"基线组" —— 但"只 AaBb"与"AaBb+AbBa"都是 λ=0，
      混在一起会把候选算进基线（同型错误见 L3-m1 订正 12/14）。
    """
    import glob
    runs = [json.loads(Path(f).read_text(encoding="utf-8")) for f in glob.glob(str(LOGS / "86g_result_*.json"))]
    groups: dict[str, list] = {}
    for r in runs:
        aw = float(r.get("abba_weight", 1.0))
        groups.setdefault(f"{r['loss_kind']}|{'+'.join(r['maps'])}|α={aw}"
                          f"|bridge={r.get('bridge','none')}|frz={bool(r.get('frozen'))}"
                          f"|io={bool(r.get('init_outputs'))}", []).append(r)
    base_key = "none|AaBb|α=1.0|bridge=none|frz=False|io=False"      # α 只作用于 AbBa 项 ⇒ 基线（只 AaBb）与 α 无关
    if base_key not in groups:
        log(f"[VERDICT] 缺基线（{base_key}）⇒ 按 PREREG 不许出判词。现有组: {sorted(groups)}"); return

    def _dig(r, path):
        x = r
        for k in path:
            x = x[k]
        return float(x)

    def mu(rs, path):
        vals = [_dig(r, path) for r in rs]
        n = len(vals)
        return (float(np.mean(vals)),
                float(np.std(vals, ddof=1) / math.sqrt(n)) if n > 1 else float("nan"), vals)

    base = groups[base_key]
    log(f"[VERDICT] 基线 {base_key}（{len(base)} run）；候选组：{[k for k in sorted(groups) if k != base_key]}")
    for key, cand in sorted(groups.items()):
        if key == base_key:
            continue
        rows = []
        for nm, path, margin, ratio in (
                ("Q1a 池化 Δ(衔接)", ("metrics", "bridge", "pooled", "delta"), None, None),
                ("Q1b 长度匹配 Δ(衔接)", ("metrics", "bridge", "m8_length_matched", "delta"), None, None),
                ("Q2a 中→中字准", ("final", "char_zz"), -0.05, None),
                ("Q2b 英→英字准", ("final", "char_ee"), -0.05, None),
                ("Q3 分化度(中)", ("final_spread", "zh"), None, 0.8),
                ("Q4a 生成EM 中→中", ("gen", "gen_em_zz"), -0.05, None),
                ("Q4b 生成EM 英→英", ("gen", "gen_em_ee"), -0.05, None)):
            bm, _, _ = mu(base, path)
            cm, cs, cv = mu(cand, path)
            if nm.startswith("Q1"):
                ok = cm >= bm + 2 * cs and all(v > bm for v in cv)
            elif ratio is not None:
                ok = cm >= bm * ratio
            else:
                ok = cm >= bm + margin
            rows.append((nm, bm, cm, ok, cv))
        log(f"[VERDICT-GROUP] ==== 候选 {key}（{len(cand)} run）vs 基线 ====")
        for nm, bm, cm, ok, cv in rows:
            log(f"[VERDICT-ROW] {nm:<16} 基线 {bm:+.4f} → 候选 {cm:+.4f}  {'✅' if ok else '❌'}  候选逐 seed={[round(v, 4) for v in cv]}")
        bm_t, _, _ = mu(base, ("final", "char_ze"))
        cm_t, _, cv_t = mu(cand, ("final", "char_ze"))
        bm_e, _, _ = mu(base, ("final", "char_ez"))
        cm_e, _, _ = mu(cand, ("final", "char_ez"))
        log(f"[VERDICT-TRANS] 翻译（非门，证明模因承载跨语言内容）：中→英 {bm_t*100:.1f}% → {cm_t*100:.1f}%；"
            f"英→中 {bm_e*100:.1f}% → {cm_e*100:.1f}%  逐 seed={[round(v*100,1) for v in cv_t]}")
        allok = all(r[3] for r in rows)
        log(f"[VERDICT] {key}：{'✅ Q1∧Q2∧Q3∧Q4 全过 ⇒ 跨语言锚定成立' if allok else '❌ 未全过 ⇒ 按上面卡住的那条报读数'}")


def tables() -> None:
    import glob
    rows = [json.loads(Path(f).read_text(encoding="utf-8")) for f in sorted(glob.glob(str(LOGS / "86g_result_*.json")))]
    if not rows:
        log("没有 86g 结果"); return
    print("| 损失 | maps | α | bridge | 冻结 | seed | 步 | 中→中 | 英→英 | 中→英 | 英→中 | "
          "Δ输入侧 | **Δ衔接** | Δ长度匹配(衔接) | 生成EM 中→中/英→英 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        f, m = r["final"], r["metrics"]
        br = m.get("bridge") or m["input"]
        inp = m.get("input", br)
        print(f"| {r['loss_kind']} | {'+'.join(r['maps'])} | {r.get('abba_weight',1.0)} | "
              f"{r.get('bridge','none')} | {'是' if r.get('frozen') else '否'} | {r['seed']} | {r['steps']} | "
              f"{f['char_zz']*100:.1f}% | {f['char_ee']*100:.1f}% | {f['char_ze']*100:.1f}% | {f['char_ez']*100:.1f}% | "
              f"{inp['pooled']['delta']:+.4f} | **{br['pooled']['delta']:+.4f}** | "
              f"{br['m8_length_matched']['delta']:+.4f} | "
              f"{r['gen']['gen_em_zz']*100:.1f}%/{r['gen']['gen_em_ee']*100:.1f}% |")


def genobj_selfcheck() -> None:
    """`--obj roll` 的仪表自检 N1/N2/N4/N5/N6（CPU、合成数据、不需要语料缓存）。

    ★ 为什么这几条必须存在（PREREG_GENOBJ §2）：N1 抓"新通路其实没接上"，N2 抓"pass 2 误用金标"，
      **N4 抓"r_in 首列没接 BOS / 移位错"、N5 抓"pass 1 漏写 no_grad"**（这两条 N1/N2 都抓不住），
      **N6 抓"两臂全局 RNG 消耗不一致"**（dropout 随机流分叉 ⇒ 违反 R34）。
    """
    dev = torch.device("cpu")
    torch.manual_seed(20261012)
    model = MZ.build(1234, False, "parser", 64).to(dev)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    maps = ("AaBb", "AbBa")
    g = torch.Generator().manual_seed(7)
    zl = [torch.randint(5, VOCAB - 1, (17,), generator=g).tolist() for _ in range(6)]
    el = [torch.randint(5, VOCAB - 1, (19,), generator=g).tolist() for _ in range(6)]
    ok = True

    # ---- N4 / N5：单批输入契约与梯度截断（直接调 _rollout_ce 拿 r_in / pred / lg1）----
    t, _mz, _me, bz, _be, _ = _terms(model, crit, zl, el, maps, dev, obj="tf")
    tz_in, tz_out = make_pairs(zl, dev)
    _ce, r_in, pred, lg1 = _rollout_ce(model, crit, "zh", bz, tz_in, tz_out)
    n4 = bool((r_in[:, 0] == BOS).all()) and bool((r_in[:, 1:] == pred[:, :-1]).all())
    n5 = (lg1.grad_fn is None) and (not pred.requires_grad)
    log(f"[N4] r_in 首列=BOS 且逐位移位对齐: {n4}（r_in[0,:1]={int(r_in[0, 0])} BOS={BOS}）")
    log(f"[N5] lg1.grad_fn is None={lg1.grad_fn is None} / pred.requires_grad={bool(pred.requires_grad)} ⇒ {n5}")
    ok = ok and n4 and n5

    # ---- N2：pass 2 的输入**不是金标**（结构性、确定性）。
    #   ★ 订正（2026-10-12，本自检第一次跑时发现）：原写"初始化时 roll CE 严格大于 tf CE"**无判别力** ——
    #     初始化近均匀（CE≈ln V≈9.36 nats），两个前缀给出的 CE 只差 ~0.04 nats，会被 dropout 噪声翻号
    #     （实测 ez 项 tf=9.3638 > roll=9.3250）。改为直接查张量：`r_in` 必须**不等于** `t_in`。
    #     "误用金标"由 N4（`r_in[:,1:] == pred[:,:-1]`）与 N2（`r_in != t_in`）合起来钉死。
    diff_frac = float((r_in != tz_in).float().mean())
    n2 = diff_frac >= 0.5
    log(f"[N2] r_in ≠ t_in 的位置占比 = {diff_frac:.3f}（应 ≥0.5 ⇒ 输入是模型自己的预测而不是金标）⇒ {n2}")
    ok = ok and n2
    _, _, _, _, _, roll_terms = _terms(model, crit, zl, el, maps, dev, obj="roll")
    for k in sorted(t):
        log(f"[N2-读数·不进门] {k}: tf={float(t[k].detach()):.4f} roll={float(roll_terms[k].detach()):.4f} "
            f"（初始化近均匀 ⇒ 该差无判别力，只作参考）")

    # ---- N1：roll_w=0 与 tf 在**同批同 RNG 状态**下 loss 逐位相同 ----
    def _loss(obj: str, roll_w: float) -> float:
        st = torch.get_rng_state()
        tt, mmz, mme, bbz, bbe, rr = _terms(model, crit, zl, el, maps, dev, obj=obj)
        val = float(_combine_loss(tt, rr, mmz, mme, bbz, bbe, "none", 0.0, 1.0, obj, roll_w))
        torch.set_rng_state(st)
        return val
    l_tf, l_r0 = _loss("tf", 0.5), _loss("roll", 0.0)
    n1 = abs(l_tf - l_r0) <= 1e-6
    log(f"[N1] tf loss={l_tf:.8f} | roll(w=0) loss={l_r0:.8f} | max|Δ|={abs(l_tf - l_r0):.3e} ⇒ {n1}")
    ok = ok and n1

    # ---- N6：两臂各走一步之后，全局 RNG 状态逐位相同 ----
    torch.manual_seed(1234)
    _terms(model, crit, zl, el, maps, dev, obj="tf")
    st_tf = torch.get_rng_state().clone()
    torch.manual_seed(1234)
    _terms(model, crit, zl, el, maps, dev, obj="roll")
    st_roll = torch.get_rng_state().clone()
    n6 = bool(torch.equal(st_tf, st_roll))
    log(f"[N6] tf 臂与 roll 臂一步后全局 RNG 逐位相同: {n6}")
    ok = ok and n6

    log(f"[GENOBJ-SELFCHECK] {'OK' if ok else 'FAIL'}（N1/N2/N4/N5/N6 五条）")
    if not ok:
        sys.exit(9)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--run", nargs=2, metavar=("LAM", "SEED"))
    ap.add_argument("--maps", default="AaBb,AbBa")
    ap.add_argument("--loss", default="none", choices=["none", "align", "infonce", "infonce_var"])
    ap.add_argument("--steps", type=int, default=STEPS_DEFAULT)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY)
    ap.add_argument("--bs", type=int, default=BS_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--shared-out", action="store_true")
    ap.add_argument("--bridge", default="none", choices=["none", "parser", "dual"],
                    help="插入 parser 卡 t（语言解析卡）：输入模因 → t → 输出卡（Ata/Btb/Atb/Bta）")
    ap.add_argument("--bridge-mid", type=int, default=64, help="parser 卡的瓶颈维度 d_mid（<128，§13.2）")
    ap.add_argument("--init-inputs", default="", help="从该 ckpt 载入 zh_in/en_in（配合 --freeze-inputs）")
    ap.add_argument("--freeze-inputs", action="store_true", help="冻结 zh_in/en_in（只训 parser 与输出卡）")
    ap.add_argument("--align-on", default="bridge", choices=["bridge", "input"],
                    help="对齐项加在哪一层：bridge=t(m)（默认）· input=m（压力直接落在输入卡上）")
    ap.add_argument("--freeze-bridge", action="store_true",
                    help="冻结解析卡（臂 IV：随机初始化后不可训，同尺寸对照）")
    ap.add_argument("--init-outputs", action="store_true",
                    help="同时载入 zh_out/en_out（让『有/无 parser』两臂只差 parser 一处）")
    ap.add_argument("--abba-weight", type=float, default=1.0,
                    help="AbBa（互映射）两项的权重 α；1.0 = 与两项均值等价（见 PREREG §7）")
    ap.add_argument("--early-stop", action="store_true",
                    help="默认关：跑满 --steps（早停判据曾误触，见 train_run 注释）")
    ap.add_argument("--obj", default="tf", choices=["tf", "roll"],
                    help="输出卡训练目标：tf=现状 4 项 teacher-forced CE；roll=再加 roll_w·mean(4 项 rollout CE)（PREREG_GENOBJ.md）")
    ap.add_argument("--roll-w", type=float, default=0.5, help="rollout 项权重（仅 --obj roll 生效；默认 0.5）")
    ap.add_argument("--gen-eval-n", type=int, default=1000, help="自由生成评测样本数（gen_eval 的 n；PREREG_GENOBJ §3）")
    ap.add_argument("--genobj-selfcheck", action="store_true",
                    help="N1/N2/N4/N5/N6 仪表自检（CPU、合成数据，不需要语料缓存）")
    args = ap.parse_args()
    dev = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    if dev != args.device:
        log(f"[WARN] cuda 不可用 ⇒ 回落 {dev}")

    if args.prepare:
        st = DZ.build()
        log(f"[PREP] 链接 {st['links']} ⇒ 保留 {st['kept']} 对；丢弃 {st['drop']}")
        log(f"[PREP] 分词器往返：{DZ.build_roundtrip_check()}")
        pack = get_pack()
        log(f"[PREP] train={len(pack['train'])} held-out={len(pack['test'])}（seed={SPLIT_SEED}）")
    elif args.smoke:
        smoke(dev)
    elif args.tables:
        tables()
    elif args.verdict:
        verdict()
    elif args.genobj_selfcheck:
        genobj_selfcheck()
    elif args.run:
        lam, seed = float(args.run[0]), int(args.run[1])
        train_run(lam, seed, args.steps, args.maps, args.loss, dev, args.bs,
                  args.eval_every,
                  shared_out=args.shared_out, early_stop=args.early_stop,
                  abba_weight=args.abba_weight, bridge=args.bridge, bridge_mid=args.bridge_mid,
                  align_on=args.align_on, init_inputs=args.init_inputs,
                  freeze_inputs=args.freeze_inputs, init_outputs=args.init_outputs,
                  freeze_bridge=args.freeze_bridge,
                  obj=args.obj, roll_w=args.roll_w, gen_eval_n=args.gen_eval_n)
    else:
        ap.print_help()


def smoke(dev: str) -> None:
    """S1–S5 自检（与 L3-m1 同精神；缺一即作废）。"""
    pack = get_pack()
    tk = D.load_tokenizer()
    z0, e0 = pack["test"][0]
    log(f"[S1] 语料首条：中 {z0!r} / 英 {e0!r}")
    log(f"[S2] 逐字往返：{'✅' if all(tk.decode(tk.encode(s).ids) == s for s in (z0, e0)) else '★失败'}")
    tr = {z for z, _ in pack["train"]}
    te = {z for z, _ in pack["test"]}
    log(f"[S3] train ∩ held-out（中文）= {len(tr & te)}（应 0）；分母 train={len(tr)} test={len(te)}")
    m = MZ.build(1234).to(dev).eval()          # ★ eval()：关掉 dropout，否则 S4 测的是 dropout 噪声
    z_ids = pad_ids([pack["test_ids"][0][0]], dev)
    with torch.no_grad():
        a = m.meme_zh(z_ids)
        b = m.meme_zh(z_ids)
    log(f"[S4] 同进程同批两次前向 max|Δ| = {float((a-b).abs().max()):.3e}（应 0）")
    m2 = MZ.build(1234)
    log(f"[S5] 同 seed 两次构造 max|Δθ| = "
        f"{max(float((p1-p2).abs().max()) for p1, p2 in zip(m.state_dict().values(), m2.state_dict().values())):.3e}（应 0）")
    with torch.no_grad():
        mm = MS.cosmat(MS.pooled(MS.floor_random(64, 7)), MS.pooled(MS.floor_random(64, 8)))
    log(f"[S6] 度量非退化：随机模因余弦矩阵非对角 std = {float(mm[~torch.eye(64,dtype=torch.bool)].std()):.4f}（应 >0）")


if __name__ == "__main__":
    main()
