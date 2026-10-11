#!/usr/bin/env python3
"""S86 · L3-m1 模因锚定试验驱动：预处理 / smoke 自检 / 单臂训练+评估 / 汇总判词。

规格与判据的唯一真相源 = `experiments/meme_anchor/PREREG.md`（判据写死，本文件不许发明判据）。

    python stages/86_meme_anchor.py --prepare                  # 建数据与渲染缓存（CPU）
    python stages/86_meme_anchor.py --smoke                    # S1–S5 + INIT-CHECK + 20 步
    python stages/86_meme_anchor.py --run 0.5 1234 --steps 6000 # 单臂训练 + 全量评估
    python stages/86_meme_anchor.py --verdict                  # 汇总 M1′–M6 判词

只写 `logs/86_*` 与 `experiments/meme_anchor/`；不碰 `cards/` / `harness/` / `nanoSeek/`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path("/home/vesita/coding/my/flowme")
HERE = ROOT / "experiments" / "meme_anchor"
sys.path.insert(0, str(HERE))

import data as DAT          # noqa: E402
import measure as MS        # noqa: E402
import model as MD          # noqa: E402
from data import PAD, VOCAB  # noqa: E402
from model import BOS, EOS   # noqa: E402

LOGS = ROOT / "logs"
EVAL_EVERY_DEFAULT = 500
PRIMARY_ARCH = "detr"          # 主档（PREREG 订正 2 / 订正 8）：只有它参与判词；yolo 只作附加臂
PRIMARY_SEEDS = (1234, 5678)   # ★ 订正 12：判词**只认预注册的 2 个 seed** —— 补救轮（seed 9012）等后补臂
                               #   与主档同 arch/同 λ ⇒ 不按 seed 过滤会把它们当主档臂、判词不可复现。
PRIMARY_LAMS = (0.0, 0.5, 2.0)  # ★ 订正 14：同样只认预注册的 3 个 λ（否则 0.1 等新形状会挤进主档判词）
PRIMARY_LOSS = "align"          # ★ 主档用的是 "align"；对比式/其它形状一律算"非主档读数"


def log(msg: str) -> None:
    print(msg, flush=True)


def sha16(p) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16]


def js(o):
    """把 numpy / torch 标量转成 JSON 可序列化。"""
    if isinstance(o, dict):
        return {k: js(v) for k, v in o.items() if k != "delta_i"}
    if isinstance(o, (list, tuple)):
        return [js(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist() if o.numel() > 1 else float(o)
    return o


# --------------------------------------------------------------------------- 批处理工具
def pad_ids(ids_list, device, pad: int = PAD) -> torch.Tensor:
    L = max(len(a) for a in ids_list)
    x = torch.full((len(ids_list), L), pad, dtype=torch.long, device=device)
    for i, a in enumerate(ids_list):
        x[i, :len(a)] = torch.tensor(a, dtype=torch.long, device=device)
    return x


def img_batch(imgs, idx, device) -> torch.Tensor:
    if isinstance(idx, (list, np.ndarray)):
        arr = np.asarray(imgs)[list(idx)]
    elif isinstance(idx, slice):
        arr = np.asarray(imgs[idx])                     # ★ slice 不能和 int 相加
    else:
        arr = np.asarray(imgs[idx:idx + 1])
    if arr.ndim == 2:
        arr = arr[None]
    return torch.from_numpy(np.ascontiguousarray(arr)).to(device).unsqueeze(1)


def slot_matrix(A: torch.Tensor, B: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """逐槽对齐的相似度矩阵：S[i,j] = mean_p cos(A_i[p], B_j[p])。"""
    a = torch.as_tensor(A).float()
    b = torch.as_tensor(B).float()
    a = a / a.norm(dim=2, keepdim=True).clamp_min(1e-12)
    b = b / b.norm(dim=2, keepdim=True).clamp_min(1e-12)
    out = [torch.einsum("ipd,jpd->ij", a[i:i + chunk], b) / a.size(1)
           for i in range(0, a.size(0), chunk)]
    return torch.cat(out, 0)


def d_slot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """损失里的模因差异 D = 1 − mean_p cos(a_p, b_p)（PREREG §4 写死）。"""
    an = a / a.norm(dim=2, keepdim=True).clamp_min(1e-12)
    bn = b / b.norm(dim=2, keepdim=True).clamp_min(1e-12)
    return 1.0 - (an * bn).sum(-1).mean()


# --------------------------------------------------------------------------- 评估
@torch.no_grad()
def eval_model(model, ids_list, imgs, device, bs: int = 64) -> dict:
    model.eval()
    crit = nn.CrossEntropyLoss(ignore_index=PAD, reduction="sum")
    tot_t = tot_v = 0.0
    cor_t = cor_v = n_tok = 0
    ex_t = ex_v = 0
    cor_t_abl = cor_v_abl = 0
    T, I = [], []
    for i in range(0, len(ids_list), bs):
        chunk = ids_list[i:i + bs]
        ids = pad_ids(chunk, device)
        tgt_in, tgt_out = MD.make_pairs(chunk, device)
        im = img_batch(imgs, slice(i, i + bs), device)
        mt = model.meme_text(ids)
        mi = model.meme_image(im)
        lt = model.logits(mt, tgt_in)
        lv = model.logits(mi, tgt_in)
        mask = tgt_out != PAD
        n_tok += int(mask.sum())
        tot_t += float(crit(lt.reshape(-1, VOCAB), tgt_out.reshape(-1)))
        tot_v += float(crit(lv.reshape(-1, VOCAB), tgt_out.reshape(-1)))
        pt, pv = lt.argmax(-1), lv.argmax(-1)
        cor_t += int(((pt == tgt_out) & mask).sum())
        cor_v += int(((pv == tgt_out) & mask).sum())
        ex_t += int(((pt == tgt_out) | ~mask).all(dim=1).sum())
        ex_v += int(((pv == tgt_out) | ~mask).all(dim=1).sum())
        # ★ M9：模因置零消融 ⇒ 解码器"靠自己（teacher forcing）"能得多少
        abl_v = model.logits(torch.zeros_like(mi), tgt_in)
        abl_t = model.logits(torch.zeros_like(mt), tgt_in)
        cor_v_abl += int(((abl_v.argmax(-1) == tgt_out) & mask).sum())
        cor_t_abl += int(((abl_t.argmax(-1) == tgt_out) & mask).sum())
        T.append(mt.float().cpu())
        I.append(mi.float().cpu())
    n = len(ids_list)
    pT, pI = torch.cat(T).mean(1), torch.cat(I).mean(1)
    ca_t, ca_v = cor_t / max(n_tok, 1), cor_v / max(n_tok, 1)
    ca_t_abl, ca_v_abl = cor_t_abl / max(n_tok, 1), cor_v_abl / max(n_tok, 1)
    return dict(ce_t=tot_t / max(n_tok, 1), ce_v=tot_v / max(n_tok, 1),
                char_acc_t=ca_t, char_acc_v=ca_v, exact_t=ex_t / n, exact_v=ex_v / n, n_tok=n_tok,
                char_acc_t_ablated=ca_t_abl, char_acc_v_ablated=ca_v_abl,
                ablation_gain_t=ca_t - ca_t_abl, ablation_gain_v=ca_v - ca_v_abl,
                spread_t=float(pT.std(dim=0).mean()), spread_v=float(pI.std(dim=0).mean()),
                m_T=torch.cat(T), m_I=torch.cat(I))


def full_metrics(ev: dict, pack: dict, seed: int) -> dict:
    """PREREG §8 的全部口径：三相似度 + 正控 + 顺序自检 + 地板 + 阈值 + 熵。"""
    T, I = ev["m_T"], ev["m_I"]
    pT, pI = MS.pooled(T), MS.pooled(I)

    rep_pool = MS.score_report(MS.cosmat(pT, pI), "pooled")
    rep_slot = MS.score_report(slot_matrix(T, I), "slot")
    cka = MS.cka_linear(pT, pI)
    cka_m, cka_se = MS.cka_perm_null(pT, pI, 50, seed)
    lin_i, lin_iii = MS.linmap(pT, pI)
    _, lin_iii_rand = MS.linmap(pT, pI, derange=True)

    # M2′ 正控：同模态（文本模因 × 文本模因）异内容
    S_tt = MS.cosmat(pT, pT)
    off = S_tt.clone()
    off.fill_diagonal_(float("nan"))
    ctrl_tt = float(torch.nanmean(off))

    # M3 顺序自检：两路各自随机置换 16 槽（不同置换）
    rep_shuf = MS.score_report(slot_matrix(MS.shuffle_slots(T, seed + 1),
                                           MS.shuffle_slots(I, seed + 2)), "slot_shuffled")

    # M4 地板
    n = pT.size(0)
    rep_rand = MS.score_report(MS.cosmat(MS.pooled(MS.floor_random(n, seed + 3)),
                                         MS.pooled(MS.floor_random(n, seed + 4))), "floor_random")
    T_id, I_id = MS.floor_identity_mediator(pack["test_ids"], ev["m_I_src"], seed + 5)
    rep_id = MS.score_report(MS.cosmat(MS.pooled(T_id), MS.pooled(I_id)), "floor_identity")
    cnt = MS.floor_counts(pack["test"])
    cn = cnt / cnt.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rep_cnt = MS.score_report(MS.cosmat(cn, cn), "floor_counts")
    bag = MS.floor_bag_of_chars(pack["test_ids"])
    bn = bag / bag.norm(dim=1, keepdim=True).clamp_min(1e-12)
    rep_bag = MS.score_report(MS.cosmat(bn, bn), "reference_bag_of_chars")

    # ★ M8（订正 7）：同长度、异内容的配对 Δ —— 排除"图片模因只记住句长"
    m8 = MS.length_matched_delta(MS.cosmat(pT, pI), [len(s) for s in pack["test"]])
    # ★ M3：逐槽口径下的"置换掉幅"（主口径池化对置换恒等 ⇒ 不能用主口径判 M3）
    drop = rep_slot["delta"] - rep_shuf["delta"]
    drop_se = math.sqrt(rep_slot["delta_se"] ** 2 + rep_shuf["delta_se"] ** 2)
    m3 = dict(delta_slot=rep_slot["delta"], delta_shuffled=rep_shuf["delta"],
              drop=drop, drop_se=drop_se, pass_drop=bool(drop >= max(2 * drop_se, 0.01)))
    # ★ 免费规则地板必须被超过（R7/R9：地板含计数类）
    surface_gate = bool(rep_pool["delta"] >= rep_cnt["delta"] + max(2 * rep_cnt["delta_se"], 0.01))

    return dict(
        pooled=js(rep_pool), slot=js(rep_slot), slot_shuffled=js(rep_shuf),
        cka=cka, cka_perm=cka_m, cka_perm_se=cka_se, cka_delta=cka - cka_m,
        lin_identity=js(lin_i), lin_mapped=js(lin_iii), lin_mapped_rand=js(lin_iii_rand),
        ctrl_same_modality_diff_content=ctrl_tt,
        floor_random=js(rep_rand), floor_identity=js(rep_id), floor_counts=js(rep_cnt),
        reference_bag=js(rep_bag),
        m8_length_matched=js(m8), surface_gate=surface_gate, m3=m3,
        ablation=dict(char_acc_t_ablated=ev["char_acc_t_ablated"],
                      char_acc_v_ablated=ev["char_acc_v_ablated"],
                      gain_t=ev["ablation_gain_t"], gain_v=ev["ablation_gain_v"]),
        entropy=MS.count_tokens_entropy(pack["test_ids"]),
    )


# --------------------------------------------------------------------------- 单臂训练
def align_extra(mt: torch.Tensor, mi: torch.Tensor, loss_kind: str, lam: float,
                tau: float, mu: float, gamma: float) -> torch.Tensor:
    """对齐 / 对比项（口径见 `experiments/meme_anchor/PREREG_SHAPE.md` §1）。

    · `none`         ：无对齐项（等价 λ=0，只留共享出口 + 重构压力）
    · `align`        ：现有形状 `D = 1 − mean_p cos(m_T[p], m_I[p])`（**只有吸引、没有排斥**）
    · `infonce`      ：池化后的对称 InfoNCE（同句为正、**批内异句为负** ⇒ 加排斥项）
    · `infonce_var`  ：`infonce` + 防塌陷方差项 `clamp(γ − std_batch(z), 0)`
    """
    if loss_kind == "none":
        return mt.sum() * 0.0                      # 保持计算图、数值恒 0
    if loss_kind == "align":
        return lam * d_slot(mt, mi)
    zt = nn.functional.normalize(mt.mean(1), dim=1)
    zi = nn.functional.normalize(mi.mean(1), dim=1)
    logits = zt @ zi.t() / tau
    labels = torch.arange(mt.size(0), device=mt.device)
    nce = 0.5 * (nn.functional.cross_entropy(logits, labels)
                 + nn.functional.cross_entropy(logits.t(), labels))
    out = lam * nce
    if loss_kind == "infonce_var":
        var = (torch.clamp(gamma - mt.mean(1).std(dim=0), min=0).mean()
               + torch.clamp(gamma - mi.mean(1).std(dim=0), min=0).mean())
        out = out + mu * var
    return out


@torch.no_grad()
def gen_eval(model, ids_list, imgs, device, n: int = 200, bs: int = 50, maxlen: int = 64) -> dict:
    """**真还原**（贪心自回归，不给任何目标前缀）：held-out 前 `n` 句，两条路各算 EM 与字级准确率。

    ★ 只在**最终评估**调用一次（每步迭代一次解码，比 teacher forcing 贵 ~60×）；
    ★ 与 `eval_model` 的区别：这里是模型自己往下写 ⇒ 用于 `PREREG_INFONCE.md` 的 **Q4**。
    """
    model.eval()
    idx = list(range(min(n, len(ids_list))))
    out: dict = {}
    for path in ("text", "image"):
        em = cor = tot = 0
        for i in range(0, len(idx), bs):
            js = idx[i:i + bs]
            chunk = [ids_list[j] for j in js]
            meme = (model.meme_text(pad_ids(chunk, device)) if path == "text"
                    else model.meme_image(img_batch(imgs, js, device)))
            cur = torch.full((len(chunk), 1), BOS, dtype=torch.long, device=device)
            done = torch.zeros(len(chunk), dtype=torch.bool, device=device)
            outs: list[list[int]] = [[] for _ in chunk]
            for _ in range(maxlen):
                nxt = model.logits(meme, cur)[:, -1].argmax(-1)
                for r in range(len(chunk)):
                    if not done[r]:
                        if int(nxt[r]) == EOS or len(outs[r]) >= 60:
                            done[r] = True
                        else:
                            outs[r].append(int(nxt[r]))
                if bool(done.all()):
                    break
                cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
            for r, gold in enumerate(chunk):
                pred = outs[r]
                em += int(pred == list(gold))
                cor += sum(1 for a, b in zip(pred, gold) if a == b)
                tot += len(gold)
        out[f"gen_em_{path}"] = em / max(len(idx), 1)
        out[f"gen_char_acc_{path}"] = cor / max(tot, 1)
    out["gen_n"] = len(idx)
    return out


def train_run(lam: float, seed: int, steps: int, vision_arch: str, device: str,
              bs: int, eval_every: int, loss_kind: str = "align",
              tau: float = 0.07, mu: float = 1.0, gamma: float = 0.2,
              mode: str = "joint") -> dict:
    """`mode="joint"` = 两条路都训（主档）；`mode="image"` = **只训图片卡 + 共享输出卡**（诊断用）。

    ★ `mode="image"` 时**跳过文本路**（省一半算力），且**跨模态项无定义** ⇒ 该配置下 `--loss` 只应给 `none`。
      这是 `PREREG_SEEDSTAB.md` 臂①：用来判"图片路的平台是它自己造成的、还是与文本路共适应造成的"。
    """
    pack = get_pack()
    model = MD.build(seed, vision_arch).to(device)
    npar = model.n_params()
    log(f"[M1] run lam={lam} seed={seed} steps={steps} arch={vision_arch} device={device} "
        f"bs={bs} loss={loss_kind} mode={mode} tau={tau} mu={mu} gamma={gamma} params={npar} 总={sum(npar.values())}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    g = torch.Generator().manual_seed(seed)
    N = len(pack["train_ids"])
    order = torch.randperm(N, generator=g).tolist()
    ptr = 0
    snaps: list[dict] = []
    t0 = time.time()
    step_time = float("nan")
    stopped = "上限步数"

    for step in range(1, steps + 1):
        if ptr + bs > N:
            order = torch.randperm(N, generator=g).tolist()
            ptr = 0
        idx = order[ptr:ptr + bs]
        ptr += bs
        chunk = [pack["train_ids"][j] for j in idx]
        ids = pad_ids(chunk, device)
        tgt_in, tgt_out = MD.make_pairs(chunk, device)
        im = img_batch(pack["train_img"], idx, device)

        if mode == "image":
            mi = model.meme_image(im)
            lv = model.logits(mi, tgt_in)
            ce_v = crit(lv.reshape(-1, VOCAB), tgt_out.reshape(-1))
            ce_t = torch.zeros((), device=device)
            loss = ce_v
        else:
            mt = model.meme_text(ids)
            mi = model.meme_image(im)
            lt = model.logits(mt, tgt_in)
            lv = model.logits(mi, tgt_in)
            ce_t = crit(lt.reshape(-1, VOCAB), tgt_out.reshape(-1))
            ce_v = crit(lv.reshape(-1, VOCAB), tgt_out.reshape(-1))
            loss = ce_t + ce_v + align_extra(mt, mi, loss_kind, lam, tau, mu, gamma)

        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        if step == 20:
            step_time = (time.time() - t0) / 20
            log(f"[SPEED] lam={lam} seed={seed} 前 20 步 {step_time:.3f} s/step "
                f"⇒ {steps} 步预计 {step_time*steps/60:.1f} min")

        if step % eval_every == 0 or step == steps:
            ev = eval_model(model, pack["test_ids"], pack["test_img"], device)
            rep = MS.score_report(MS.cosmat(MS.pooled(ev["m_T"]), MS.pooled(ev["m_I"])), "pooled")
            snap = dict(step=step, ce_t=ev["ce_t"], ce_v=ev["ce_v"],
                        char_acc_t=ev["char_acc_t"], char_acc_v=ev["char_acc_v"],
                        exact_t=ev["exact_t"], exact_v=ev["exact_v"],
                        spread_t=ev["spread_t"], spread_v=ev["spread_v"],
                        delta=rep["delta"], delta_se=rep["delta_se"],
                        pass_final=rep["pass_final"])
            snaps.append(snap)
            log(f"[SNAP] lam={lam} seed={seed} step={step} CE_T={snap['ce_t']:.4f} CE_V={snap['ce_v']:.4f} "
                f"字准T={snap['char_acc_t']*100:.1f}% 字准V={snap['char_acc_v']*100:.1f}% "
                f"精T={snap['exact_t']*100:.1f}% 精V={snap['exact_v']*100:.1f}% "
                f"Δ={snap['delta']:+.4f}±{snap['delta_se']:.4f} 过判据={snap['pass_final']} "
                f"分化T={snap['spread_t']:.3f} 分化V={snap['spread_v']:.3f} | 用时 {(time.time()-t0)/60:.1f}min")
            if len(snaps) >= 3:
                c = [s["ce_t"] + s["ce_v"] for s in snaps[-3:]]
                r1 = (c[0] - c[1]) / max(c[0], 1e-9)
                r2 = (c[1] - c[2]) / max(c[1], 1e-9)
                if r1 < 0.01 and r2 < 0.01:
                    stopped = f"收敛（连续两次快照 CE 改善 <1%：{r1*100:.2f}% / {r2*100:.2f}%）"
                    log(f"[STOP] lam={lam} seed={seed} step={step} {stopped}")
                    break
            model.train()

    ev = eval_model(model, pack["test_ids"], pack["test_img"], device)
    ev["m_I_src"] = torch.as_tensor(np.asarray(pack["test_img"]))
    metrics = full_metrics(ev, pack, seed)
    gen = gen_eval(model, pack["test_ids"], pack["test_img"], device)     # ★ Q4：真还原（贪心）
    log(f"[GEN] lam={lam} seed={seed} 贪心自回归（n={gen['gen_n']}）EM: 文本 {gen['gen_em_text']*100:.1f}% · "
        f"图片 {gen['gen_em_image']*100:.1f}% | 字准: 文本 {gen['gen_char_acc_text']*100:.1f}% · "
        f"图片 {gen['gen_char_acc_image']*100:.1f}%")
    # ★ 文件名带"损失形状/模式"标记（否则 infonce 与 infonce_var、joint 与 image 会互相覆盖）
    tag = ("" if loss_kind == "align" else f"_{loss_kind}") + ("" if mode == "joint" else f"_{mode}")
    ck = LOGS / f"86_ckpt_{vision_arch}{tag}_l{lam}_s{seed}.pt"
    torch.save({"state": model.state_dict(), "lam": lam, "seed": seed,
                "vision_arch": vision_arch, "steps": steps}, ck)
    res = dict(lam=lam, seed=seed, steps=steps, vision_arch=vision_arch, bs=bs,
               loss_kind=loss_kind, mode=mode, tau=tau, mu=mu, gamma=gamma,
               stopped=stopped, step_time_s=step_time, wall_min=(time.time() - t0) / 60,
               params=npar, ckpt=str(ck), ckpt_sha16=sha16(ck), snaps=snaps,
               final=dict(ce_t=ev["ce_t"], ce_v=ev["ce_v"], char_acc_t=ev["char_acc_t"],
                          char_acc_v=ev["char_acc_v"], exact_t=ev["exact_t"], exact_v=ev["exact_v"]),
               gen=gen,
               metrics=metrics)
    (LOGS / f"86_result_{vision_arch}{tag}_l{lam}_s{seed}.json").write_text(
        json.dumps(js(res), ensure_ascii=False, indent=1), encoding="utf-8")
    log(f"[RUN-FINAL] lam={lam} seed={seed} CE_T={ev['ce_t']:.4f} CE_V={ev['ce_v']:.4f} "
        f"字准T={ev['char_acc_t']*100:.2f}% 字准V={ev['char_acc_v']*100:.2f}% "
        f"精T={ev['exact_t']*100:.2f}% 精V={ev['exact_v']*100:.2f}% "
        f"分化T={ev['spread_t']:.3f} 分化V={ev['spread_v']:.3f} "
        f"Δ={metrics['pooled']['delta']:+.4f}±{metrics['pooled']['delta_se']:.4f} "
        f"2SE={metrics['pooled']['pass2se']} 下限={metrics['pooled']['pass_floor']} "
        f"过判据={metrics['pooled']['pass_final']} | {stopped} | "
        f"墙钟 {res['wall_min']:.1f}min")
    return res


# --------------------------------------------------------------------------- 数据包（进程内缓存）
_PACK: dict | None = None


def get_pack() -> dict:
    global _PACK
    if _PACK is None:
        _PACK = DAT.prepare(smoke=False)
    return _PACK


# --------------------------------------------------------------------------- 自检
def init_check(seed: int, vision_arch: str) -> float:
    """INIT-CHECK：同 seed 构造两次，逐参数 max|Δθ| 必须为 0（L3_PLAN §9.7）。"""
    a, b = MD.build(seed, vision_arch), MD.build(seed, vision_arch)
    sa, sb = a.state_dict(), b.state_dict()
    assert set(sa) == set(sb)
    return max((float((sa[k].float() - sb[k].float()).abs().max()) for k in sa), default=0.0)


def self_checks(device: str, vision_arch: str) -> dict:
    pack = DAT.prepare(smoke=True)
    out: dict = {}
    tr, te = pack["train"][:100], pack["test"][:100]

    # S1 渲染确定性
    a, b = DAT.render(tr[0]), DAT.render(tr[0])
    s1 = int(np.abs(a.astype(int) - b.astype(int)).max())
    out["S1_render_determinism"] = s1
    log(f"[S1] 渲染确定性 max|Δ| = {s1} ⇒ {'过' if s1 == 0 else '★不过'}")

    # S1b 无裁切（订正 4：旧版按 24px 排行距 ⇒ 60 字句第三行被裁，只裁长句）
    long_s = ("昭通机场是位于中国云南昭通的民用机场始建于一九三五年" * 3)[:60]
    r0, r1, c0, c1 = DAT.ink_bbox(long_s)
    s1b = bool(r1 < DAT.IMG_H - 1 and c1 < DAT.IMG_W - 1)
    out["S1b_no_clip"] = s1b
    log(f"[S1b] 无裁切：60 字句墨迹 bbox=(行 {r0}..{r1}, 列 {c0}..{c1}) 画布 "
        f"{DAT.IMG_H}x{DAT.IMG_W} ⇒ {'过' if s1b else '★不过（内容被切）'}")

    # S3 无泄漏
    meta = DAT.build_sentences()
    leak = len(set(meta["train"]) & set(meta["test"]))
    out["S3_leak"] = leak
    log(f"[S3] train∩test 逐字重叠 = {leak}（{len(meta['train'])}/{len(meta['test'])}），"
        f"去重剔除 {meta['n_dup_dropped']} 条 ⇒ {'过' if leak == 0 else '★不过'}")

    # S4 目标可生成
    tk = DAT.load_tokenizer()
    bad = [s for s in te if tk.decode(tk.encode(s).ids) != s]
    out["S4_bad_targets"] = len(bad)
    log(f"[S4] 目标可逐字还原：坏 {len(bad)} / {len(te)} ⇒ {'过' if not bad else '★不过'}")

    # INIT-CHECK
    ic = init_check(1234, vision_arch)
    out["INIT_CHECK"] = ic
    log(f"[INIT-CHECK] 同 seed 两次构造 逐参数 max|Δθ| = {ic} ⇒ {'过' if ic == 0.0 else '★不过'}")

    # S2 前向确定性 + S5 度量非退化 + 形状契约 + 20 步冒烟
    model = MD.build(1234, vision_arch).to(device).eval()
    ids = pad_ids(pack["train_ids"][:8], device)
    im = img_batch(pack["train_img"], slice(0, 8), device)
    with torch.no_grad():
        m1a, m1b = model.meme_text(ids), model.meme_text(ids)
        i1a, i1b = model.meme_image(im), model.meme_image(im)
    s2 = max(float((m1a - m1b).abs().max()), float((i1a - i1b).abs().max()))
    out["S2_forward_determinism"] = s2
    log(f"[S2] 同进程同批两次前向 max|Δ| = {s2:.3e} ⇒ {'过' if s2 == 0.0 else '★不过'}")
    log(f"[SHAPE] meme_text={tuple(m1a.shape)} meme_image={tuple(i1a.shape)} "
        f"（契约 [b,16,128]）")

    rnd = MS.cosmat(MS.pooled(MS.floor_random(64, 1)), MS.pooled(MS.floor_random(64, 2)))
    off = rnd.clone()
    off.fill_diagonal_(float("nan"))
    s5 = float(off[~torch.isnan(off)].std(unbiased=True))
    out["S5_metric_nondegenerate"] = s5
    log(f"[S5] 随机模因相似度矩阵 非对角 std = {s5:.4f} ⇒ {'过' if s5 > 1e-6 else '★不过'}")

    # 20 步冒烟（测步时）
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    crit = nn.CrossEntropyLoss(ignore_index=PAD)
    t0 = time.time()
    model.train()
    for step in range(20):
        ck = pack["train_ids"][step * 8:(step + 1) * 8]
        i_ = pad_ids(ck, device)
        ti, to = MD.make_pairs(ck, device)
        im_ = img_batch(pack["train_img"], slice(step * 8, (step + 1) * 8), device)
        mt, mi = model.meme_text(i_), model.meme_image(im_)
        loss = crit(model.logits(mt, ti).reshape(-1, VOCAB), to.reshape(-1)) + \
            crit(model.logits(mi, ti).reshape(-1, VOCAB), to.reshape(-1)) + 0.5 * d_slot(mt, mi)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    st = (time.time() - t0) / 20
    out["smoke_step_s"] = st
    log(f"[SPEED] smoke bs=8 前 20 步 {st:.3f} s/step（设备 {device}）")
    out["all_pass"] = all([s1 == 0, s1b, leak == 0, not bad, ic == 0.0, s2 == 0.0, s5 > 1e-6])
    log(f"[SMOKE] {'全部自检通过' if out['all_pass'] else '★有自检不过'}")
    return out


# --------------------------------------------------------------------------- 汇总判词
def verdict() -> dict:
    """汇总判词。★ 只对**主档 arch**（detr）判；其余 arch（如 yolo 附加臂）只列不判。"""
    files = sorted(LOGS.glob("86_result_*.json"))
    if not files:
        raise SystemExit("没有 86_result_*.json")
    all_runs = [json.loads(f.read_text(encoding="utf-8")) for f in files]
    runs = [r for r in all_runs if r.get("vision_arch") == PRIMARY_ARCH
            and int(r.get("seed", -1)) in PRIMARY_SEEDS
            and float(r.get("lam", -1)) in PRIMARY_LAMS
            and r.get("loss_kind", PRIMARY_LOSS) == PRIMARY_LOSS]
    skipped = [r for r in all_runs if r not in runs]
    if skipped:
        log(f"[VERDICT] 非主档读数（不参与判词）："
            f"{[(r.get('vision_arch'), r['lam'], r['seed'], r['steps']) for r in skipped]}")
    if not runs:
        raise SystemExit(f"没有主档 arch={PRIMARY_ARCH} 的结果")
    by_lam: dict[float, list] = {}
    for r in runs:
        by_lam.setdefault(float(r["lam"]), []).append(r)

    table, arm = [], {}
    for lam, rs in sorted(by_lam.items()):
        def seed_metric(key):
            return [r["metrics"][key] for r in rs]

        pooled = seed_metric("pooled")
        m8s = seed_metric("m8_length_matched")
        m3s = seed_metric("m3")
        # G-read（订正 7 后含 M9 模因置零消融：解码器不许自己把句子补出来）
        gr = [bool(r["final"]["char_acc_v"] >= 0.90 * r["final"]["char_acc_t"]
                   and r["final"]["char_acc_v"] >= 0.50
                   and r["metrics"]["ablation"]["gain_v"] >= 0.20) for r in rs]
        # 正控（M2′）：同模态异内容必须显著低于跨模态同内容
        ctrl_ok = [bool(r["metrics"]["ctrl_same_modality_diff_content"]
                        < r["metrics"]["pooled"]["same"] - 2 * r["metrics"]["pooled"]["delta_se"])
                   for r in rs]
        # 饱和（R35 + 复核 A5）：**末两次间隔**都要 <1%（原版只看末一格，会把 12.5%/0.7% 判成饱和）
        sat = []
        for r in rs:
            ce = [s["ce_t"] + s["ce_v"] for s in r["snaps"]]
            ok_sat = False
            if len(ce) >= 3:
                r1 = (ce[-3] - ce[-2]) / max(ce[-3], 1e-9)
                r2 = (ce[-2] - ce[-1]) / max(ce[-2], 1e-9)
                ok_sat = bool(r1 < 0.01 and r2 < 0.01)
            sat.append(ok_sat)
        arm[lam] = dict(
            n_seed=len(rs),
            g_read=all(gr), ctrl=all(ctrl_ok), saturated=all(sat),
            m1=all(bool(x["pass_final"]) for x in pooled),
            m8=all(bool(x["pass_final"]) for x in m8s),
            m3=all(bool(x["pass_drop"]) for x in m3s),
            surface=all(bool(r["metrics"]["surface_gate"]) for r in rs))
        table.append(dict(lam=lam, n_seed=len(rs),
                          delta=[x["delta"] for x in pooled], delta_se=[x["delta_se"] for x in pooled],
                          pass2se=[bool(x["pass2se"]) for x in pooled],
                          pass_floor=[bool(x["pass_floor"]) for x in pooled],
                          pass_final=[bool(x["pass_final"]) for x in pooled],
                          auroc_centered=[x["auroc_centered"] for x in pooled],
                          ctrl=[r["metrics"]["ctrl_same_modality_diff_content"] for r in rs],
                          g_read=gr, saturated=sat, ctrl_ok=ctrl_ok,
                          char_acc_t=[r["final"]["char_acc_t"] for r in rs],
                          char_acc_v=[r["final"]["char_acc_v"] for r in rs],
                          ablation_gain_v=[r["metrics"]["ablation"]["gain_v"] for r in rs],
                          m8_delta=[x["delta"] for x in m8s], m8_pass=[bool(x["pass_final"]) for x in m8s],
                          m3_drop=[x["drop"] for x in m3s], m3_pass=[bool(x["pass_drop"]) for x in m3s],
                          surface_gate=[bool(r["metrics"]["surface_gate"]) for r in rs],
                          slot_delta=[r["metrics"]["slot"]["delta"] for r in rs],
                          slot_shuf_delta=[r["metrics"]["slot_shuffled"]["delta"] for r in rs],
                          cka_delta=[r["metrics"]["cka_delta"] for r in rs]))
        a = arm[lam]
        log(f"[VERDICT-ARM] λ={lam} seeds={len(rs)} Δ={['%+.4f' % x['delta'] for x in pooled]} "
            f"Δ长度匹配={['%+.4f' % x['delta'] for x in m8s]} "
            f"G-read={gr} 饱和={sat} 正控={ctrl_ok} 超地板={a['surface']} "
            f"⇒ M1′={a['m1']} M8={a['m8']} M3={a['m3']}")

    # ---------------- 5 档判决（订正 7 + 复核 A1–A4 的强制点） ----------------
    arms = list(arm.values())
    a0 = arm.get(0.0, {})
    pooled_all = [r["metrics"]["pooled"] for r in runs]
    # ★ A4 完整性门：必须 3 臂 × 2 seed 齐备，且数据口径就是新画布 —— 否则不许出判词
    meta = json.loads((ROOT / "logs" / "86_cache" / "sent_meta.json").read_text(encoding="utf-8"))
    expect_lam = [0.0, 0.5, 2.0]
    complete = (sorted(arm.keys()) == expect_lam) and all(a["n_seed"] == 2 for a in arms)
    canvas_ok = list(meta.get("img_shape", [])) == [128, 584]

    # ★ 订正 13（用户 2026-10-11 裁定）：**M2′ 作废** —— 它的定义是「同模态·异内容 < 跨模态·同内容 − 2SE」，
    #   把**内容差**与**模态偏移**绑在一次比较里（实测 λ=0：同模态异内容 0.482/0.446 > 跨模态同内容 0.236/−0.030，
    #   而同一张矩阵的**行中心化 AUROC = 0.985/0.779**）⇒ 只保留为**报告读数**（`ctrl` 仍在表里），**不再作门**。
    #   旧：M2′ 不过 ⇒ 档 0b（实得档 0b）；新：链上不再有 M2′ 分支。
    if not canvas_ok or not complete:
        v = (f"档 0（**读数不完整，不许出判词**）：臂/seed = "
             f"{ {k: v['n_seed'] for k, v in sorted(arm.items())} }（应 3 臂×2 seed）；"
             f"数据画布 = {meta.get('img_shape')}（应 [128,584]，即订正 4 后的规格）")
    # ★ 顺序（订正 10）：① 仪器有效性（口径）→ ② 训练充分性（饱和）→ ③ 前置门 → ④ 结论
    elif not all(a["m3"] for a in arms):
        v = ("档 3（**判据口径缺陷**）：M3 逐槽置换掉幅不过（<max(2SE,0.01)）⇒ 度量对槽序不敏感 "
             "⇒ 属判据缺陷、**不进 M6**（`experiment-design.md`：判据错 ≠ 数据错）")
    elif not all(a["saturated"] for a in arms):
        n_sat = sum(1 for a in arms if a["saturated"])
        v = (f"档 4（**未饱和**）：{n_sat}/{len(arms)} 臂达收敛（判据=末**两次**间隔 CE 改善均 <1%）"
             f"⇒ 按 R35 **只报方向**；★ 此档下 **G-read 未过不得读成『学不会读图』**"
             f"（读数还没到平台），也不得下失败结论")
    elif not any(a["g_read"] for a in arms):
        v = ("档 0（前置门未过 · 已饱和）：没有任何臂通过 G-read（图片路字准 ≥0.90×文本路 且 ≥0.50，"
             "**且**模因置零后字准必须掉 ≥20pp）⇒ 本实现下『没学会读图 / 解码器靠自己补句子』"
             "⇒ **不构成对模因假设的判决**")
    elif not all(a["surface"] for a in arms):                    # ★ A2：地板升门必须生效
        v = ("档 3b（**与表面统计不可区分**）：Δ_模因未超过计数类地板 + max(2SE,0.01) "
             "⇒ 判词只能写『与表面统计不可区分』，**不得**写『模因锚定』（订正 7）")
    elif a0.get("g_read") and a0.get("m1") and a0.get("m8"):     # ★ A1；M2′ 已作废（订正 13）
        v = ("档 1：**λ=0 也过 M1′+M8+正控、且自己过了 G-read** ⇒ **模因自然锚定**"
             "（在『共享出口』这一耦合下，且已排除句长混杂）—— 强")
    elif any(arm.get(l, {}).get("g_read") and arm.get(l, {}).get("m1")
             and arm.get(l, {}).get("m8")
             for l in arm if l != 0.0):                          # ★ A1；M2′ 已作废（订正 13）
        v = ("档 2：**仅 λ>0 过 M1′+M8+正控（且过 G-read）** ⇒ 对齐可训练、模因空间可被拉齐 —— 中；"
             "阈值只对『训练过对齐』的卡有效")
    elif all(a["m1"] for a in arms) and not all(a["m8"] for a in arms):
        v = ("档 5b（**与句长不可区分**）：M1′（随机配对）过、而 **M8（同长度异内容）不过** "
             "⇒ 跨模态相似可由句长/版式解释 ⇒ 不得写成模因锚定")
    elif all(x.get("unmeasured") for x in pooled_all):
        v = ("档 5a（M6 · **该口径下未测出差异**）：|Δ| ≤ 1SE ⇒ 本实现未达成；"
             "**不是『模因假设被证伪』**（`§10.1` 的存续条件需要 T vs M 对照，本试验没有）")
    elif any((not x.get("pass2se")) and abs(x["delta"]) > x["delta_se"] for x in pooled_all):
        v = ("档 5d（**不可判**）：部分 run 的 Δ 落在 `1SE < |Δ| < 2SE` ⇒ 既不算过也不算未测出 "
             "⇒ 按 §8 M1′ 判『不可判』（不许写成失败）")
    else:
        v = ("档 5c（M6 · **本实现未达成**）：G-read（带地板）过、已饱和、而 M1′ 不过 ⇒ "
             "该口径下未测出跨模态一致性 ⇒ 依 `§0.2` 记为『未决 / 本实现未达成』，"
             "**不是『模因假设被证伪』**")
    out = dict(verdict=v, arms=table, arm=arm,
               runs=[{k: r[k] for k in ("lam", "seed", "steps", "stopped", "wall_min")} for r in runs])
    (LOGS / "86_verdict.json").write_text(json.dumps(js(out), ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    log(f"[VERDICT] {v}")
    return out


def tables() -> None:
    """从 86_result_*.json 生成 markdown 表格（报告用；不判断、不改数字）。"""
    runs = [json.loads(f.read_text(encoding="utf-8"))
            for f in sorted(LOGS.glob("86_result_*.json"))]
    if not runs:
        raise SystemExit("没有 86_result_*.json")
    runs.sort(key=lambda r: (float(r["lam"]), int(r["seed"])))

    print("\n### 表 A · 每 run 终态\n")
    print("| λ | seed | arch | steps | 停止原因 | 墙钟 min | CE_T | CE_V | 字准T | 字准V | 字准V(置零) | 消融增益V | 精T | 精V | Δ(池化) | SE | 过判据 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        f, m = r["final"], r["metrics"]["pooled"]
        abl = r["metrics"]["ablation"]
        print(f"| {r['lam']} | {r['seed']} | {r.get('vision_arch','detr')} | {r['steps']} | {r['stopped']} | "
              f"{r['wall_min']:.1f} | "
              f"{f['ce_t']:.4f} | {f['ce_v']:.4f} | {f['char_acc_t']*100:.2f}% | {f['char_acc_v']*100:.2f}% | "
              f"{abl['char_acc_v_ablated']*100:.2f}% | {abl['gain_v']*100:.1f}pp | "
              f"{f['exact_t']*100:.2f}% | {f['exact_v']*100:.2f}% | {m['delta']:+.4f} | {m['delta_se']:.4f} | "
              f"{m['pass_final']} |")

    print("\n### 表 B · 各口径（PREREG §8，含订正 7 的 M8/M3/地板门）\n")
    print("| λ | seed | Δ池化 | **Δ长度匹配** | Δ逐槽 | Δ逐槽打乱 | M3掉幅 | CKA Δ | 线性映射Δ | 正控(同模态异内容) | 超计数地板 | AUROC中心 | 阈值(中心) | TPR | FPR |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        m = r["metrics"]
        print(f"| {r['lam']} | {r['seed']} | {m['pooled']['delta']:+.4f} | "
              f"**{m['m8_length_matched']['delta']:+.4f}** | {m['slot']['delta']:+.4f} | "
              f"{m['slot_shuffled']['delta']:+.4f} | {m['m3']['drop']:+.4f} | {m['cka_delta']:+.4f} | "
              f"{m['lin_mapped']['delta']:+.4f} | {m['ctrl_same_modality_diff_content']:.4f} | "
              f"{m['surface_gate']} | {m['pooled']['auroc_centered']:.4f} | "
              f"{m['pooled']['thr_centered']['thr']:+.4f} | {m['pooled']['thr_centered']['tpr']:.3f} | "
              f"{m['pooled']['thr_centered']['fpr']:.3f} |")

    print("\n### 表 B2 · 新门明细（M8 长度匹配 / M3 掉幅 / 地板）\n")
    print("| λ | seed | Δ随机 | Δ长度匹配 | 同长度平均伙伴 | M8过 | M3过 | 正控过 | 计数地板Δ | 超地板 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in runs:
        m = r["metrics"]
        ctrl_ok = m["ctrl_same_modality_diff_content"] < m["pooled"]["same"] - 2 * m["pooled"]["delta_se"]
        print(f"| {r['lam']} | {r['seed']} | {m['pooled']['delta']:+.4f} | {m['m8_length_matched']['delta']:+.4f} | "
              f"{m['m8_length_matched']['mean_partners']:.1f} | {m['m8_length_matched']['pass_final']} | "
              f"{m['m3']['pass_drop']} | {ctrl_ok} | {m['floor_counts']['delta']:+.4f} | {m['surface_gate']} |")

    print("\n### 表 C · 地板（λ/seed 取首个 run 作代表，其余 run 同口径）\n")
    r0 = runs[0]
    print("| 地板 | Δ | SE | 2SE |")
    print("|---|---|---|---|")
    for name, key in (("随机模因", "floor_random"), ("恒等中介", "floor_identity"),
                      ("计数类特征", "floor_counts"), ("参考·词袋", "reference_bag")):
        d = r0["metrics"][key]
        print(f"| {name} | {d['delta']:+.4f} | {d['delta_se']:.4f} | {d['pass2se']} |")
    print(f"\n熵/容量：{json.dumps(r0['metrics']['entropy'], ensure_ascii=False)}")

    print("\n### 表 D · 收敛曲线（CE_T+CE_V 与 Δ）\n")
    print("| λ | seed | " + " | ".join(f"@{s['step']}" for s in r0["snaps"]) + " |")
    print("|---" * (len(r0["snaps"]) + 2) + "|")
    for r in runs:
        ce = " | ".join(f"{s['ce_t']+s['ce_v']:.3f}" for s in r["snaps"])
        print(f"| {r['lam']} | {r['seed']} | {ce} |")
    for r in runs:
        dd = " | ".join(f"{s['delta']:+.4f}" for s in r["snaps"])
        print(f"| {r['lam']} | {r['seed']} (Δ) | {dd} |")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--run", nargs=2, metavar=("LAM", "SEED"))
    ap.add_argument("--verdict", action="store_true")
    ap.add_argument("--tables", action="store_true")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--eval-every", type=int, default=EVAL_EVERY_DEFAULT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vision-arch", default="detr", choices=["detr", "yolo"])
    ap.add_argument("--loss", default="align",
                    choices=["none", "align", "infonce", "infonce_var"],
                    help="对齐/对比项形状（见 PREREG_SHAPE.md §1）")
    ap.add_argument("--mode", default="joint", choices=["joint", "image"],
                    help="joint=两条路都训；image=只训图片卡+输出卡（诊断，见 PREREG_SEEDSTAB.md）")
    ap.add_argument("--tau", type=float, default=0.07, help="InfoNCE 温度")
    ap.add_argument("--mu", type=float, default=1.0, help="方差项权重")
    ap.add_argument("--gamma", type=float, default=0.2, help="方差项目标标准差")
    args = ap.parse_args()

    torch.set_num_threads(6)
    dev = args.device
    if dev == "cuda" and not torch.cuda.is_available():
        log("[WARN] cuda 不可用 ⇒ 回落 cpu")
        dev = "cpu"
    if dev == "cuda":
        log(f"[ENV] device={torch.cuda.get_device_name(0)} "
            f"free={torch.cuda.mem_get_info()[0]/2**30:.2f}GiB")

    if args.prepare:
        t0 = time.time()
        pack = DAT.prepare(smoke=False)
        log(f"[PREPARE] train={len(pack['train'])} test={len(pack['test'])} "
            f"meta={pack['meta']} 用时 {(time.time()-t0)/60:.1f} min")
    elif args.smoke:
        self_checks(dev, args.vision_arch)
    elif args.run:
        lam, seed = float(args.run[0]), int(args.run[1])
        train_run(lam, seed, args.steps, args.vision_arch, dev, args.bs, args.eval_every,
                  args.loss, args.tau, args.mu, args.gamma, args.mode)
    elif args.verdict:
        verdict()
    elif args.tables:
        tables()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
