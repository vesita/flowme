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


def _terms(model, crit, z_list, e_list, maps, device) -> tuple[dict, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """按映射集合算各项 CE。返回 (各项 CE, 输入模因 mz/me, **衔接模因** bz/be)。

    ★ 所有输出卡读的都是**衔接处**的模因 `bz/be = t(mz/me)`（`bridge="none"` 时 t 是恒等 ⇒ 与 L3-m2 逐位一致）。
       用户 2026-10-11 的写法 Ata/Btb/Atb/Bta 就是"输入卡 → t → 输出卡"，t 两语言共用。
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
    return t, mz, me, bz, be

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
        _, mz, me, bz, be = _terms(model, nn.CrossEntropyLoss(ignore_index=PAD), zl, el, ALL_MAPS, device)
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
              freeze_bridge: bool = False,
              tau: float = 0.07, mu: float = 1.0, gamma: float = 0.2) -> dict:
    mapset = tuple(m for m in maps.split(",") if m)
    pack = get_pack()
    model = MZ.build(seed, shared_out, bridge, bridge_mid).to(device)
    npar = model.n_params()
    log(f"[M2] run lam={lam} seed={seed} steps={steps} maps={mapset} loss={loss_kind} "
        f"abba_w={abba_weight} bridge={bridge}(mid={bridge_mid}) bs={bs} shared_out={shared_out} "
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
        t, mz, me, bz, be = _terms(model, crit, zl, el, mapset, device)
        aa = [v for k, v in t.items() if k in ("zz", "ee")]        # AaBb（自映射）
        ab = [v for k, v in t.items() if k in ("ze", "ez")]        # AbBa（互映射）
        # ★ 0.5·(mean_aa + α·mean_ab)：α 只调"翻译压力 / 重构压力"的比例，且 **α=1 时恰好等于
        #   `sum(四项)/4`**（第一轮的实现）⇒ 复用第一轮 α=1.0 的 run 合法。
        #   若写成 `mean_aa + α·mean_ab`，α=1 会变成原来的 **2 倍**（等于偷偷给 CE 加倍）—— 已避开。
        loss = (0.5 * (sum(aa) / max(len(aa), 1) + abba_weight * sum(ab) / max(len(ab), 1))
                + DRV.align_extra(bz, be, loss_kind, lam, tau, mu, gamma))   # ★ 对齐项作用于**衔接处**
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
    ev = eval_model(model, pack, device)
    metrics = full_metrics(ev, pack, seed)
    gen = gen_eval(model, pack, device)
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
           + ("_frz" if freeze_inputs else "") + ("_io" if init_outputs else "")
           + ("_frzbr" if freeze_bridge else "") + init_lab)
    ck = LOGS / f"86g_ckpt_{tag}_l{lam}_s{seed}.pt"
    torch.save(dict(state=model.state_dict(), seed=seed, lam=lam, maps=mapset,
                    loss_kind=loss_kind, steps=step, shared_out=shared_out), ck)
    res = dict(lam=lam, seed=seed, steps=step, maps=list(mapset), loss_kind=loss_kind,
               shared_out=shared_out, abba_weight=abba_weight, bridge=bridge,
               bridge_mid=bridge_mid, init_inputs=str(init_inputs), init_outputs=init_outputs,
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
    ap.add_argument("--freeze-bridge", action="store_true",
                    help="冻结解析卡（臂 IV：随机初始化后不可训，同尺寸对照）")
    ap.add_argument("--init-outputs", action="store_true",
                    help="同时载入 zh_out/en_out（让『有/无 parser』两臂只差 parser 一处）")
    ap.add_argument("--abba-weight", type=float, default=1.0,
                    help="AbBa（互映射）两项的权重 α；1.0 = 与两项均值等价（见 PREREG §7）")
    ap.add_argument("--early-stop", action="store_true",
                    help="默认关：跑满 --steps（早停判据曾误触，见 train_run 注释）")
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
    elif args.run:
        lam, seed = float(args.run[0]), int(args.run[1])
        train_run(lam, seed, args.steps, args.maps, args.loss, dev, args.bs,
                  args.eval_every, args.shared_out, args.early_stop, args.abba_weight,
                  args.bridge, args.bridge_mid, args.init_inputs, args.freeze_inputs,
                  args.init_outputs, args.freeze_bridge)
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
