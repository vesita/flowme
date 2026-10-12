#!/usr/bin/env python3
"""S86p · L3-t1（`experiments/meme_anchor/PREREG_THINKCARD.md` **v2**）判据表：单思维卡的容量瓶颈对照。

    .venv/bin/python stages/86p_thinkcard_verdict.py                  # 离线重评 ckpt + 出判词 + 落盘
    .venv/bin/python stages/86p_thinkcard_verdict.py --selfcheck      # N5（R48）：假结果逐档驱动**真实 verdict()**
    .venv/bin/python stages/86p_thinkcard_verdict.py --n3-check       # N3：A 臂前向恒等（同批惰性对照）
    .venv/bin/python stages/86p_thinkcard_verdict.py --n6-check       # N6：同源输入卡（四臂同 seed 同一 ckpt）
    .venv/bin/python stages/86p_thinkcard_verdict.py --n7-check       # N7：梯度流（bridge 参数有非零梯度）

★ 与 PREREG v2 **逐条同构**（复核必改 2/6 的核心要求：文本与代码不许各说各话）：
  主指标 = 翻译 TF char 宏平均（逐样本配对）· Δ_min = max(2×SE_sample, δ_seed) · 等价 margin δ_eq = δ_seed（≥1pp）
  非劣门**只有**重构 TF char 两条（`zz`/`ee`）· gen EM 只并列报 · 档八「无分辨力 ⇒ 不可判」· 操纵门 = 档七条件
  判定表序：七(不完备/操纵门破) → 八(无分辨力) → 一 → 六 → 三 → 三′(瓶颈只把卡拉回基线) → 二 → 四 → 五
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path("/home/vesita/coding/my/flowme")
LOGS = ROOT / "logs"
STEPS = 8000
SEED_FLOOR = 0.01                    # δ 下限 1pp（measurement §2 的 δ = max(地板, 极差) 同构）
N2_TOL = 1e-3                        # 跨进程/跨设备 pred 翻转地板：本机实测 2.2e-5（1 个 token）
SEEDS_DEFAULT = (1234, 5678, 9012, 3456)
ARMS = {"A": ("none", 64), "B": ("parser", 128), "C64": ("parser", 64), "C32": ("parser", 32)}
MIDSUF = {"A": "", "B": "_parser128", "C64": "_parser64", "C32": "_parser32"}
MAPS4 = ("zz", "ee", "ze", "ez")
BRANCH = {
    "1": "一 瓶颈是机制", "2": "二 卡可旁路", "3": "三 卡有益但与瓶颈无关",
    "3p": "三′ 瓶颈只把卡拉回基线（卡未超过旁路）", "4": "四 瓶颈有害",
    "5": "五 带内（该口径下未测出差异）", "6": "六 代价过大",
    "7": "七 作废（读数不完备 / 操纵门破）", "8": "八 无分辨力（不可判）",
}


# ------------------------------------------------------------------ 路径 / 驱动
def run_paths(arm: str, seed: int, steps: int = STEPS) -> tuple[Path, Path]:
    tag = (f"infonce_AaBb_AbBa{MIDSUF[arm]}_o-tf_st{steps}_frz_io_"
           f"i-none_AaBb_AbBa_a0.5_o-tf_st8000")
    return (LOGS / f"86g_result_{tag}_l0.3_s{seed}.json",
            LOGS / f"86g_ckpt_{tag}_l0.3_s{seed}.pt")


def load_driver():
    spec = importlib.util.spec_from_file_location("zhendrv", ROOT / "stages" / "86g_zh_en.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def load_model(drv, ckpt: Path, bridge: str, mid: int, device: str):
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = drv.MZ.build(int(blob["seed"]), bool(blob.get("shared_out", False)), bridge, mid).to(device)
    model.load_state_dict(blob["state"], strict=True)
    return model, blob


# ------------------------------------------------------------------ 离线逐样本重评
@torch.no_grad()
def per_item_tf(drv, model, pack, device, chunk: int = 64) -> tuple[dict, dict, dict]:
    """teacher forcing 的**逐样本** char 准确率 / EM；返回 (acc, em, 全局 char)。"""
    zs = [p[0] for p in pack["test_ids"]]
    es = [p[1] for p in pack["test_ids"]]
    model.eval()                                            # R39
    acc = {k: [] for k in MAPS4}
    em = {k: [] for k in MAPS4}
    cor = {k: 0 for k in MAPS4}
    tok = {k: 0 for k in MAPS4}
    for i in range(0, len(zs), chunk):
        zl, el = zs[i:i + chunk], es[i:i + chunk]
        _, _mz, _me, bz, be, _ = drv._terms(model, nn.CrossEntropyLoss(ignore_index=drv.PAD),
                                            zl, el, drv.ALL_MAPS, device)
        tz_in, tz_out = drv.make_pairs(zl, device)
        te_in, te_out = drv.make_pairs(el, device)
        logs = {"zz": (model.logits("zh", bz, tz_in), tz_out),
                "ee": (model.logits("en", be, te_in), te_out),
                "ze": (model.logits("en", bz, te_in), te_out),
                "ez": (model.logits("zh", be, tz_in), tz_out)}
        for nm, (lg, tgt) in logs.items():
            mk = tgt != drv.PAD
            pred = lg.argmax(-1)
            hit = ((pred == tgt) & mk).sum(1).float()
            cnt = mk.sum(1).float().clamp_min(1)
            acc[nm].extend((hit / cnt).tolist())
            em[nm].extend((((pred == tgt) | ~mk).all(1)).float().tolist())
            cor[nm] += int(hit.sum())
            tok[nm] += int(mk.sum())
    return acc, em, {k: cor[k] / max(tok[k], 1) for k in MAPS4}


@torch.no_grad()
def per_item_gen(drv, model, pack, device, n: int = 1000, bs: int = 50, maxlen: int = 160) -> dict:
    """贪心自回归逐样本 EM（与驱动 `gen_eval` 同一算法；由 N2 对账保证没走偏）。"""
    ids = pack["test_ids"][:n]
    model.eval()
    em = {k: [] for k in MAPS4}
    for nm, (src_ix, gold_ix, lang) in {"zz": (0, 0, "zh"), "ee": (1, 1, "en"),
                                        "ze": (0, 1, "en"), "ez": (1, 0, "zh")}.items():
        for i in range(0, len(ids), bs):
            part = ids[i:i + bs]
            sl = [p[src_ix] for p in part]
            gl = [p[gold_ix] for p in part]
            meme = (model.meme_zh(drv.pad_ids(sl, device)) if src_ix == 0
                    else model.meme_en(drv.pad_ids(sl, device)))
            meme = model.apply_bridge(meme, "zh" if src_ix == 0 else "en")
            cur = torch.full((len(sl), 1), drv.BOS, dtype=torch.long, device=device)
            done = torch.zeros(len(sl), dtype=torch.bool, device=device)
            outs: list[list[int]] = [[] for _ in sl]
            for _ in range(maxlen):
                nxt = model.logits(lang, meme, cur)[:, -1].argmax(-1)
                for r in range(len(sl)):
                    if not done[r]:
                        if int(nxt[r]) == drv.EOS or len(outs[r]) >= 150:
                            done[r] = True
                        else:
                            outs[r].append(int(nxt[r]))
                if bool(done.all()):
                    break
                cur = torch.cat([cur, nxt.unsqueeze(1)], dim=1)
            for r, g in enumerate(gl):
                em[nm].append(float(outs[r] == list(g)))
    return em


@torch.no_grad()
def degeneration_rho(drv, model, pack, device, chunk: int = 64) -> float:
    """N8：`ρ = mean‖t(m)−m‖ / mean‖m‖`（只报不判；A 臂恒为 0）。"""
    if getattr(model, "bridge", None) is None and getattr(model, "bridge_zh", None) is None:
        return 0.0
    zs = [p[0] for p in pack["test_ids"]]
    es = [p[1] for p in pack["test_ids"]]
    model.eval()
    num = den = 0.0
    for i in range(0, len(zs), chunk):
        mz = model.meme_zh(drv.pad_ids(zs[i:i + chunk], device))
        me = model.meme_en(drv.pad_ids(es[i:i + chunk], device))
        for m in (mz, me):
            t = model.apply_bridge(m, "zh")
            num += float((t - m).norm(dim=-1).sum())
            den += float(m.norm(dim=-1).sum())
    return num / max(den, 1e-12)


def evaluate_run(drv, pack, arm: str, seed: int, device: str, gen_n: int, bypass: bool = False) -> dict:
    rj, ck = run_paths(arm, seed)
    bridge, mid = ARMS[arm]
    model, blob = load_model(drv, ck, bridge, mid, device)
    model.bypass_bridge = bool(bypass)
    acc, em, char_global = per_item_tf(drv, model, pack, device)
    gen = per_item_gen(drv, model, pack, device, n=gen_n)
    rho = 0.0 if bypass else degeneration_rho(drv, model, pack, device)
    res = json.loads(rj.read_text(encoding="utf-8"))
    # ---- N2 对账（容差 1e-3；逐项打印）
    problems, diffs = [], {}
    for k in MAPS4:
        d = abs(char_global[k] - float(res["final"][f"char_{k}"]))
        gm = sum(gen[k]) / max(len(gen[k]), 1)
        dg = abs(gm - float(res["gen"][f"gen_em_{k}"]))
        diffs[f"char_{k}"] = d
        diffs[f"gen_em_{k}"] = dg
        if d > N2_TOL:
            problems.append(f"N2 char_{k}: 离线 {char_global[k]:.6f} vs JSON {res['final'][f'char_{k}']:.6f}（Δ={d:.2e}）")
        if dg > N2_TOL:
            problems.append(f"N2 gen_em_{k}: 离线 {gm:.6f} vs JSON {res['gen'][f'gen_em_{k}']:.6f}（Δ={dg:.2e}）")
    b = res["metrics"].get("bridge") or res["metrics"]["input"]
    return dict(acc=acc, em=em, gen=gen, char_global=char_global, json=res, problems=problems, n2_diffs=diffs,
                tr_macro=[(a + bb) / 2 for a, bb in zip(acc["ze"], acc["ez"])],
                recon_macro=[(a + bb) / 2 for a, bb in zip(acc["zz"], acc["ee"])],
                gen_macro=[(gen["zz"][i] + gen["ee"][i] + gen["ze"][i] + gen["ez"][i]) / 4
                           for i in range(len(gen["zz"]))],
                bridge_delta=float(b["pooled"]["delta"]), bridge_delta_se=float(b["pooled"]["delta_se"]),
                n_params=int(res["n_params"]["bridge"]), stopped=res.get("stopped", ""),
                snaps=res.get("snaps", []), rho=rho, bypass=bypass)


# ------------------------------------------------------------------ 统计
def paired(x: list[float], y: list[float]) -> dict:
    assert len(x) == len(y) and x, "配对样本必须等长且非空"
    d = [a - b for a, b in zip(x, y)]
    n = len(d)
    m = sum(d) / n
    var = sum((v - m) ** 2 for v in d) / (n - 1) if n > 1 else float("nan")
    return dict(delta=m, se=math.sqrt(var / n) if n > 1 else float("nan"), n_item=n,
                n_item_pos=sum(1 for v in d if v > 0), n_item_neg=sum(1 for v in d if v < 0))


def contrast(metrics: dict, a: str, b: str, seeds, key: str) -> dict:
    per = []
    for s in seeds:
        st = paired(metrics[a][s][key], metrics[b][s][key])
        st["seed"] = s
        per.append(st)
    delta = sum(p["delta"] for p in per) / len(per)
    se_sample = max(p["se"] for p in per)
    deltas = [p["delta"] for p in per]
    seed_range = (max(deltas) - min(deltas)) if len(deltas) > 1 else float("nan")
    delta_seed = max(SEED_FLOOR, seed_range) if seed_range == seed_range else SEED_FLOOR
    return dict(delta=delta, se_sample=se_sample, delta_seed=delta_seed, seed_range=seed_range,
                delta_min=max(2 * se_sample, delta_seed), delta_eq=delta_seed,
                per_seed=per, n=len(per), n_pos=sum(1 for v in deltas if v > 0),
                n_neg=sum(1 for v in deltas if v < 0))


def sig(c: dict) -> str:
    if c["delta"] >= c["delta_min"] and c["n_pos"] == c["n"]:
        return "pos"
    if c["delta"] <= -c["delta_min"] and c["n_neg"] == c["n"]:
        return "neg"
    if abs(c["delta"]) >= c["se_sample"]:
        return "band"
    return "null"


def approx(c: dict) -> bool:
    return abs(c["delta"]) <= c["delta_eq"]


def approx_effect(c: dict) -> bool:
    """「≈ 无差别」分支的**生效条件**：落在 δ_eq 内 **且不显著**。
    ★ 边界：`|Δ| = Δ_min = δ_eq` 时两者同时成立 ⇒ 按 PREREG v2 §4 的边界规则**归正/负向档**，
      故 `≈` 分支必须排除显著情形（否则 `|Δ| = Δ_min` 的负向结果会被误判成档二『卡可旁路』）。"""
    return approx(c) and sig(c) in ("null", "band")


def noninferior(metrics: dict, arm: str, ref: str, seeds, key: str) -> dict:
    drops, s2, dsd = [], [], []
    for s in seeds:
        st = paired(metrics[arm][s][key], metrics[ref][s][key])
        drops.append(-st["delta"])
        s2.append(2 * st["se"])
    deltas = [-d for d in drops]
    dsd = max(SEED_FLOOR, (max(deltas) - min(deltas)) if len(deltas) > 1 else 0.0)
    drop = sum(drops) / len(drops)
    thr_se, thr_seed = max(s2), dsd
    thr = max(thr_se, thr_seed)
    binding = "2×SE_pair" if thr_se >= thr_seed else "δ_seed"
    return dict(drop=drop, thr=thr, thr_2se=thr_se, thr_seed=thr_seed, binding=binding, ok=bool(drop <= thr))


# ------------------------------------------------------------------ 判词（真实判决函数；R48 用伪造 summary 驱动它）
def verdict(s: dict) -> tuple[str, str]:
    """按 PREREG v2 §4 的**表序**判定：七 → 八 → 一 → 六 → 三 → 三′ → 二 → 四 → 五。"""
    if not s["complete"]:
        return "7", f"读数不完备：{'; '.join(s['problems'])}"
    if not s["manip_ok"]:
        return "7", f"操纵门破（通路未生效）：{s['manip_why']}"
    if not s["resolution"]["ok"]:
        return "8", f"无分辨力 ⇒ 不可判：{s['resolution']['why']}"
    cb, ca = s["contrasts"]["C64-B"], s["contrasts"]["C64-A"]
    k_cb, k_ca = sig(cb), sig(ca)
    if k_cb == "pos" and k_ca == "pos" and s["gates_ok"]:
        return "1", (f"C64−B {cb['delta']*100:+.2f}pp ≥ Δ_min {cb['delta_min']*100:.2f}pp（{cb['n_pos']}/{cb['n']} 同号）"
                     f"且 C64−A {ca['delta']*100:+.2f}pp ≥ Δ_min {ca['delta_min']*100:.2f}pp、非劣门全过 ⇒ 瓶颈带来能力")
    if (k_cb == "pos" or k_ca == "pos") and not s["gates_ok"]:
        bad = ", ".join(f"{k}(掉 {v['drop']*100:+.2f}pp > 门 {v['thr']*100:.2f}pp[{v['binding']}])"
                        for k, v in s["gates"].items() if not v["ok"])
        return "6", f"主判据过但非劣门破：{bad}"
    if k_ca == "pos" and approx_effect(cb):
        return "3", (f"C64−A {ca['delta']*100:+.2f}pp 而 C64−B {cb['delta']*100:+.2f}pp ≤ δ_eq "
                     f"{cb['delta_eq']*100:.2f}pp ⇒ 卡有益但与瓶颈无关 ⇒ 触发 E1（判据见 PREREG §4）")
    if k_cb == "pos" and approx_effect(ca):
        return "3p", (f"C64−B {cb['delta']*100:+.2f}pp 而 C64−A {ca['delta']*100:+.2f}pp ≤ δ_eq "
                      f"{ca['delta_eq']*100:.2f}pp ⇒ 瓶颈只把卡拉回基线，卡未超过旁路（不得写档一）")
    if approx_effect(cb) and approx_effect(ca):
        return "2", (f"C64−B {cb['delta']*100:+.2f}pp、C64−A {ca['delta']*100:+.2f}pp 都 ≤ δ_eq "
                     f"(B {cb['delta_eq']*100:.2f}pp / A {ca['delta_eq']*100:.2f}pp) ⇒ 本任务上卡与瓶颈都不是能力来源")
    if k_cb == "neg":
        return "4", (f"C64−B {cb['delta']*100:+.2f}pp ≤ −Δ_min {cb['delta_min']*100:.2f}pp"
                     f"（{cb['n_neg']}/{cb['n']} 为负）⇒ 瓶颈有害（该配置、该任务下）")
    return "5", (f"带内：C64−B {cb['delta']*100:+.2f}pp（1SE={cb['se_sample']*100:.2f}、Δ_min={cb['delta_min']*100:.2f}）、"
                 f"C64−A {ca['delta']*100:+.2f}pp（Δ_min={ca['delta_min']*100:.2f}）⇒ 该口径下未测出差异")


# ------------------------------------------------------------------ 汇总
def build_summary(metrics: dict, seeds, complete: bool, problems: list[str],
                  manip_ok: bool, manip_why: str, resolution: dict) -> dict:
    need = [("C64", "B"), ("C64", "A"), ("B", "A"), ("C32", "B")]
    contrasts = {f"{a}-{b}": contrast(metrics, a, b, seeds, "tr_macro") for a, b in need}
    gates = {}
    for arm in ("B", "C64", "C32"):
        gates[f"{arm}:重构zz"] = noninferior(metrics, arm, "A", seeds, "acc_zz")
        gates[f"{arm}:重构ee"] = noninferior(metrics, arm, "A", seeds, "acc_ee")
    return dict(complete=complete, problems=problems, seeds=list(seeds), contrasts=contrasts,
                gates=gates, gates_ok=all(v["ok"] for v in gates.values()),
                manip_ok=manip_ok, manip_why=manip_why, resolution=resolution)


def _acc_view(metrics: dict, seeds) -> None:
    """把 per-item char 数组塞进 metrics 以便复用 noninferior（键名 acc_zz / acc_ee）。"""
    for arm, per in metrics.items():
        for s in seeds:
            if s in per:
                per[s]["acc_zz"] = per[s]["acc"]["zz"]
                per[s]["acc_ee"] = per[s]["acc"]["ee"]


def manipulation_gate(metrics: dict, seeds, complete: bool) -> tuple[bool, str]:
    """B/C64/C32 臂的衔接 Δ 必须 ≥ A 臂同 seed 的衔接 Δ + 2SE（λ 项真的生效）。★ 只是仪表门。"""
    if not complete:
        return True, ""
    bad = []
    for arm in ("B", "C64", "C32"):
        for s in seeds:
            a, m = metrics["A"][s], metrics[arm][s]
            need = a["bridge_delta"] + 2 * a["bridge_delta_se"]
            if not (m["bridge_delta"] >= need):
                bad.append(f"{arm}/s{s} 衔接Δ {m['bridge_delta']:+.4f} < A+2SE {need:+.4f}")
    return (not bad), ("；".join(bad) if bad else "")


def resolution_check(metrics: dict, seeds, fl: dict, complete: bool) -> dict:
    """档八：主指标四臂全贴地/贴顶，或四臂全距 < 2×SE_sample ⇒ 无分辨力。"""
    if not complete:
        return dict(ok=True, why="")
    means = {}
    for arm in ARMS:
        vals = [sum(metrics[arm][s]["tr_macro"]) / len(metrics[arm][s]["tr_macro"]) for s in seeds]
        means[arm] = sum(vals) / len(vals)
    floor = max(fl["copy_tr"], fl["step0_tr"])
    ceiling = 1.0
    se = contrast(metrics, "C64", "B", seeds, "tr_macro")["se_sample"]
    hi, lo = ceiling - 2 * se, floor + 2 * se
    all_floor = all(v <= lo for v in means.values())
    all_ceil = all(v >= hi for v in means.values())
    spread = max(means.values()) - min(means.values())
    why = ""
    if all_floor:
        why = f"四臂主指标全 ≤ 地板+2SE（{lo*100:.2f}%，地板={floor*100:.2f}%）⇒ 贴地"
    elif all_ceil:
        why = f"四臂主指标全 ≥ 天花板−2SE（{hi*100:.2f}%）⇒ 贴顶"
    elif spread < 2 * se:
        why = f"四臂全距 {spread*100:.2f}pp < 2×SE_sample {2*se*100:.2f}pp"
    return dict(ok=(why == ""), why=why, means=means, lo=lo, hi=hi, spread=spread)


def floors(drv, pack, device: str, init_ckpt: Path | None) -> dict:
    """地板：copy（输出=输入）逐映射 EM · freq（**逐语言**恒出最高频字符）char · step-0（冻结输入+未训输出）。"""
    from collections import Counter
    zs = [p[0] for p in pack["test_ids"]]
    es = [p[1] for p in pack["test_ids"]]
    out: dict = {"copy_em": {}, "freq_char": {}, "lang_top": {}}
    gold_of = {"zz": 0, "ee": 1, "ze": 1, "ez": 0}
    src_of = {"zz": 0, "ee": 1, "ze": 0, "ez": 1}
    for lang, text_list in (("zh", zs), ("en", es)):
        cnt = Counter()
        for s in text_list:
            cnt.update(s)
        out["lang_top"][lang] = cnt.most_common(1)[0][0]
    for nm in MAPS4:
        gold = [p[gold_of[nm]] for p in pack["test_ids"]]
        src = [p[src_of[nm]] for p in pack["test_ids"]]
        out["copy_em"][nm] = sum(1 for a, b in zip(src, gold) if a == b) / max(len(src), 1)
        top = out["lang_top"]["zh" if gold_of[nm] == 0 else "en"]
        tot = sum(len(g) for g in gold)
        out["freq_char"][nm] = sum(g.count(top) for g in gold) / max(tot, 1)
    # step-0：**只灌冻结的输入卡**、输出卡保持随机初始化（teacher-forced char）——这是"语言模型先验 + 边界"的白拿分
    out["step0_char"] = {}
    if init_ckpt is not None and Path(init_ckpt).exists():
        blob = torch.load(init_ckpt, map_location="cpu", weights_only=False)
        m = drv.MZ.build(int(blob["seed"]), False, "none", 64).to(device)
        sd = {k: v for k, v in blob["state"].items() if k.startswith(("zh_in.", "en_in."))}
        m.load_state_dict(sd, strict=False)          # ★ 只灌输入卡：输出卡保持随机
        acc, _em, glob = per_item_tf(drv, m, pack, device)
        out["step0_char"] = {k: round(glob[k], 6) for k in MAPS4}
    out["copy_tr"] = (out["copy_em"]["ze"] + out["copy_em"]["ez"]) / 2
    out["step0_tr"] = ((out["step0_char"].get("ze", 0.0) + out["step0_char"].get("ez", 0.0)) / 2
                       if out["step0_char"] else 0.0)
    return out


def saturation(metrics: dict, seeds, arm: str) -> dict:
    res = {}
    for s in seeds:
        snaps = metrics[arm][s]["snaps"]
        if len(snaps) < 2:
            res[s] = dict(n=len(snaps))
            continue
        a, b = snaps[-2], snaps[-1]
        tr = lambda d: (d["char_ze"] + d["char_ez"]) / 2                   # noqa: E731
        ce = lambda d: sum(v for k, v in d.items() if k.startswith("ce_"))  # noqa: E731
        res[s] = dict(step_from=a["step"], step_to=b["step"], d_tr=tr(b) - tr(a), d_ce=ce(b) - ce(a))
    return res


# ------------------------------------------------------------------ N5 / N3 / N6 / N7
def selfcheck() -> int:
    def c(delta, se=0.005, npos=4, nneg=0, n=4, dseed=0.01):
        return dict(delta=delta, se_sample=se, delta_seed=dseed, seed_range=dseed,
                    delta_min=max(2 * se, dseed), delta_eq=dseed, n=n, n_pos=npos, n_neg=nneg, per_seed=[])
    def mk(complete=True, cb=(0.05, 4, 0, 0.005), ca=(0.05, 4, 0, 0.005), gates_ok=True, manip_ok=True,
           res_ok=True, problems=(), manip_why="", why=""):
        return dict(complete=complete, problems=list(problems), manip_ok=manip_ok, manip_why=manip_why,
                    resolution=dict(ok=res_ok, why=why),
                    contrasts={"C64-B": c(cb[0], se=cb[3], npos=cb[1], nneg=cb[2]),
                               "C64-A": c(ca[0], se=ca[3], npos=ca[1], nneg=ca[2]),
                               "B-A": c(0.0, npos=2, nneg=2), "C32-B": c(0.0, npos=2, nneg=2)},
                    gates={"B:重构zz": dict(drop=0.0, thr=0.01, ok=True, binding="δ_seed")},
                    gates_ok=gates_ok)
    cases = [
        (mk(), "1", "瓶颈是机制"),
        (mk(cb=(0.001, 2, 2, 0.005), ca=(0.001, 2, 2, 0.005)), "2", "卡可旁路（两者都 ≤ δ_eq）"),
        (mk(cb=(0.001, 2, 2, 0.005), ca=(0.05, 4, 0, 0.005)), "3", "卡有益但与瓶颈无关"),
        (mk(cb=(0.05, 4, 0, 0.005), ca=(0.001, 2, 2, 0.005)), "3p", "瓶颈只把卡拉回基线"),
        (mk(cb=(-0.05, 0, 4, 0.005)), "4", "瓶颈有害"),
        (mk(cb=(0.02, 4, 0, 0.02), ca=(0.02, 4, 0, 0.02)), "5", "带内（Δ_min=0.04 > |Δ|=0.02=1SE > δ_eq=0.01）"),
        (mk(cb=(0.05, 2, 2, 0.005)), "5", "超 Δ_min 但不同号 ⇒ 带内档（不许单次运行出结论）"),
        (mk(gates_ok=False), "6", "代价过大"),
        (mk(cb=(0.05, 4, 0, 0.005), ca=(0.05, 4, 0, 0.005), gates_ok=False), "6", "门破优先于档一"),
        (mk(ca=(0.05, 4, 0, 0.005), gates_ok=False), "6", "C64−A 过但门破 ⇒ 档六优先于档三"),
        (mk(complete=False, problems=["缺 C64/s1234"]), "7", "作废（读数不完备）"),
        (mk(manip_ok=False, manip_why="B/s1234 衔接Δ 未过 A+2SE"), "7", "作废（操纵门破）"),
        (mk(res_ok=False, why="四臂全贴地"), "8", "无分辨力"),
        (mk(cb=(0.01, 4, 0, 0.005), ca=(0.01, 4, 0, 0.005)), "1", "|Δ| 恰等于 Δ_min=0.01 ⇒ 归正/负向档（边界）"),
        (mk(cb=(-0.01, 0, 4, 0.005), ca=(-0.01, 0, 4, 0.005)), "4", "|Δ| 恰等于 −Δ_min ⇒ 归负向档（边界）"),
        (mk(cb=(0.01, 2, 2, 0.02), ca=(0.01, 2, 2, 0.02)), "2", "|Δ| 恰等于 δ_eq=0.01 ⇒ 归『≈』（档二）"),
        (mk(cb=(0.015, 4, 0, 0.01), ca=(0.015, 4, 0, 0.01)), "5", "δ_eq < |Δ| < Δ_min ⇒ 档五"),
    ]
    bad = 0
    for s, want, note in cases:
        got, txt = verdict(s)
        ok = got == want
        bad += 0 if ok else 1
        print(f"[SELFCHECK] 期望 {want:>2s}（{note}）⇒ 实得 {got:>2s} {'OK' if ok else 'FAIL'} | {txt}")
    print(f"[SELFCHECK] {'全部通过' if bad == 0 else f'{bad} 档失败'}（{len(cases)} 例）")
    return 1 if bad else 0


def n3_check() -> int:
    """N3：同一份输入/输出权重下，`parser + bypass_bridge=True` 与 `bridge=none` 的 logits 逐位相同。"""
    drv = load_driver()
    m1 = drv.MZ.build(1234, False, "parser", 64)
    m2 = drv.MZ.build(1234, False, "none", 64)
    sd = {k: v for k, v in m1.state_dict().items() if not k.startswith("bridge.")}
    miss, unexp = m2.load_state_dict(sd, strict=False)
    assert not unexp, f"N3：意外键 {unexp}"
    m1.bypass_bridge = True
    g = torch.Generator().manual_seed(20261012)
    zl = [torch.randint(5, drv.VOCAB - 1, (17,), generator=g).tolist() for _ in range(4)]
    tgt = torch.randint(5, drv.VOCAB - 1, (4, 12), generator=g)
    m1.eval(); m2.eval()
    with torch.no_grad():
        a = m1.logits("zh", m1.apply_bridge(m1.meme_zh(drv.pad_ids(zl, "cpu")), "zh"), tgt)
        b = m2.logits("zh", m2.apply_bridge(m2.meme_zh(drv.pad_ids(zl, "cpu")), "zh"), tgt)
    d = float((a - b).abs().max())
    ok = d <= 1e-6 and bool((a.argmax(-1) == b.argmax(-1)).all())
    print(f"[N3] bypass(parser) vs bridge=none：logits max|Δ|={d:.3e}、pred 全同={bool((a.argmax(-1)==b.argmax(-1)).all())} ⇒ {'OK' if ok else 'FAIL'}")
    print(f"[N3] 结构性依据：`apply_bridge` 在 `bridge is None` 时 `return m`（model_zh_en.py），本次数值兜底通过={ok}")
    return 0 if ok else 1


def n6_check(seeds) -> int:
    """N6：每 seed 的四臂 `init_inputs` 同一文件、`frozen` 含 zh_in/en_in。"""
    import hashlib
    bad = []
    for s in seeds:
        refs, frozen_ok = set(), True
        for arm in ARMS:
            rj, _ = run_paths(arm, s)
            if not rj.exists():
                bad.append(f"{arm}/s{s} 缺 result JSON"); continue
            r = json.loads(rj.read_text(encoding="utf-8"))
            refs.add(str(r.get("init_inputs", "")))
            if "zh_in" not in r.get("frozen", []) or "en_in" not in r.get("frozen", []):
                frozen_ok = False
        if len(refs) > 1:
            bad.append(f"s{s} 四臂 init_inputs 不一致：{sorted(refs)}")
        if not frozen_ok:
            bad.append(f"s{s} 有臂未冻结输入卡")
        if refs and all(refs):
            p = Path(sorted(refs)[0])
            if p.exists():
                print(f"[N6] s{s} init_inputs={p.name} sha256={hashlib.sha256(p.read_bytes()).hexdigest()[:16]} 一致={len(refs)==1}")
    if bad:
        print("[N6] FAIL：" + "；".join(bad))
        return 1
    print("[N6] OK：每 seed 四臂同源输入卡且冻结生效")
    return 0


def n7_check() -> int:
    """N7：`§13.2` 的"梯度必须经过卡" —— 首步 `bridge` 参数梯度非 None 且非零。"""
    drv = load_driver()
    ok = True
    for mid in (128, 64, 32):
        model = drv.MZ.build(7, False, "parser", mid)
        crit = nn.CrossEntropyLoss(ignore_index=drv.PAD)
        g = torch.Generator().manual_seed(3)
        zl = [torch.randint(5, drv.VOCAB - 1, (15,), generator=g).tolist() for _ in range(4)]
        el = [torch.randint(5, drv.VOCAB - 1, (18,), generator=g).tolist() for _ in range(4)]
        t, mz, me, bz, be, _ = drv._terms(model, crit, zl, el, drv.ALL_MAPS, "cpu")
        loss = drv._combine_loss(t, None, mz, me, bz, be, "none", 0.0, 1.0, "tf", 0.5)
        loss.backward()
        gs = [p.grad for p in model.bridge.parameters()]
        nonnull = all(p is not None for p in gs)
        nonzero = nonnull and all(float(p.abs().sum()) > 0 for p in gs)
        print(f"[N7] mid={mid}: bridge 参数 {len(gs)} 个 · grad 全非 None={nonnull} · 全非零={nonzero}")
        ok = ok and nonzero
    print(f"[N7] {'OK' if ok else 'FAIL'}（§13.2『梯度必须经过卡』的直接检查）")
    return 0 if ok else 1


# ------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true", help="N5：伪造结果逐档驱动真实 verdict()")
    ap.add_argument("--n3-check", action="store_true", help="N3：A 臂前向恒等")
    ap.add_argument("--n6-check", action="store_true", help="N6：同源输入卡")
    ap.add_argument("--n7-check", action="store_true", help="N7：梯度流")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_DEFAULT))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gen-n", type=int, default=1000)
    ap.add_argument("--no-d", action="store_true", help="跳过 D 臂（推理期捷径）离线评测")
    ap.add_argument("--out", default="logs/86p_thinkcard_verdict.json")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if a.n3_check:
        return n3_check()
    if a.n7_check:
        return n7_check()
    seeds = tuple(int(x) for x in a.seeds.split(",") if x)
    if a.n6_check:
        return n6_check(seeds)
    dev = a.device if (a.device != "cuda" or torch.cuda.is_available()) else "cpu"
    drv = load_driver()
    pack = drv.get_pack()

    problems: list[str] = []
    metrics: dict[str, dict] = {}
    for arm in ARMS:
        metrics[arm] = {}
        for s in seeds:
            rj, ck = run_paths(arm, s)
            if not rj.exists() or not ck.exists():
                problems.append(f"缺产物：{arm}/s{s}（{rj.name if not rj.exists() else ck.name}）")
                continue
            m = evaluate_run(drv, pack, arm, s, dev, a.gen_n)
            problems.extend(f"{arm}/s{s} {p}" for p in m["problems"])
            metrics[arm][s] = m
            print(f"[EVAL] {arm}/s{s} 桥参 {m['n_params']} · 主(翻译char宏) "
                  f"{sum(m['tr_macro'])/len(m['tr_macro'])*100:.2f}% · 重构 "
                  f"{sum(m['recon_macro'])/len(m['recon_macro'])*100:.2f}% · 生成宏 "
                  f"{sum(m['gen_macro'])/len(m['gen_macro'])*100:.2f}% · N8 ρ={m['rho']:.3f} · {m['stopped']}", flush=True)
    complete = not problems
    _acc_view(metrics, seeds)
    init_ckpt = None
    if metrics.get("A") and seeds[0] in metrics["A"]:
        init_ckpt = Path(str(metrics["A"][seeds[0]]["json"].get("init_inputs", "")))
    fl = floors(drv, pack, dev, init_ckpt)
    manip_ok, manip_why = manipulation_gate(metrics, seeds, complete)
    res_chk = resolution_check(metrics, seeds, fl, complete)
    s = build_summary(metrics, seeds, complete, problems, manip_ok, manip_why, res_chk)
    code, txt = verdict(s)

    payload = dict(prereg="PREREG_THINKCARD.md v2", seeds=list(seeds), steps=STEPS,
                   complete=complete, problems=problems, contrasts=s["contrasts"], gates=s["gates"],
                   gates_ok=s["gates_ok"], manip_ok=manip_ok, manip_why=manip_why,
                   resolution=res_chk, floors=fl, verdict=dict(code=code, label=BRANCH[code], text=txt),
                   per_arm={arm: {str(sd): dict(
                       tr_macro=sum(metrics[arm][sd]["tr_macro"]) / len(metrics[arm][sd]["tr_macro"]),
                       recon_macro=sum(metrics[arm][sd]["recon_macro"]) / len(metrics[arm][sd]["recon_macro"]),
                       gen_macro=sum(metrics[arm][sd]["gen_macro"]) / len(metrics[arm][sd]["gen_macro"]),
                       char_global=metrics[arm][sd]["char_global"], rho=metrics[arm][sd]["rho"],
                       bridge_delta=metrics[arm][sd]["bridge_delta"],
                       bridge_delta_se=metrics[arm][sd]["bridge_delta_se"],
                       n_params=metrics[arm][sd]["n_params"], stopped=metrics[arm][sd]["stopped"],
                       n2_diffs=metrics[arm][sd]["n2_diffs"],
                       saturation=saturation(metrics, seeds, arm).get(sd))
                       for sd in seeds if sd in metrics[arm]} for arm in ARMS},
                   n8_rho={arm: {str(sd): metrics[arm][sd]["rho"] for sd in seeds if sd in metrics[arm]} for arm in ARMS})

    print("\n================ L3-t1 判据表（PREREG_THINKCARD v2） ================")
    print(f"数据：held-out n={len(pack['test_ids'])} · {STEPS} 步 · seeds={list(seeds)}")
    print(f"地板：copy EM " + " ".join(f"{k}={v*100:.1f}%" for k, v in fl["copy_em"].items())
          + " | freq char " + " ".join(f"{k}={v*100:.1f}%" for k, v in fl["freq_char"].items())
          + (f" | step-0(翻译char) {fl['step0_tr']*100:.1f}%" if fl["step0_char"] else ""))
    for k, c in s["contrasts"].items():
        print(f"  {k:7s} Δ={c['delta']*100:+.2f}pp  1SE={c['se_sample']*100:.2f}  δ_seed={c['delta_seed']*100:.2f}  "
              f"Δ_min={c['delta_min']*100:.2f}  δ_eq={c['delta_eq']*100:.2f}  同号 {c['n_pos']}/{c['n']}正 {c['n_neg']}/{c['n']}负 ⇒ {sig(c)}")
    for k, g in s["gates"].items():
        print(f"  非劣门 {k:10s} 掉 {g['drop']*100:+.2f}pp ≤ 门 {g['thr']*100:.2f}pp"
              f"（2SE={g['thr_2se']*100:.2f} / δ_seed={g['thr_seed']*100:.2f} ⇒ 由 {g['binding']} 起作用）{'OK' if g['ok'] else ' **破**'}")
    print(f"操纵门：{'过' if manip_ok else '**破**'} {manip_why}")
    print(f"分辨力：{'足' if res_chk['ok'] else '**无**'} {res_chk.get('why','')} 四臂主指标={ {k: round(v*100,2) for k,v in (res_chk.get('means') or {}).items()} }")
    for arm in ARMS:
        if metrics.get(arm) and all(sd in metrics[arm] for sd in seeds):
            print(f"  饱和 {arm}: {saturation(metrics, seeds, arm)}")
    if not a.no_d and metrics.get("C64") and all(sd in metrics["C64"] for sd in seeds):
        print("  D 臂（推理期捷径：C64 权重 + bypass_bridge）：")
        darm = {}
        for sd in seeds:
            mb = evaluate_run(drv, pack, "C64", sd, dev, a.gen_n, bypass=True)
            st = paired(metrics["C64"][sd]["tr_macro"], mb["tr_macro"])
            deq = s["contrasts"]["C64-B"]["delta_eq"]
            dmin = s["contrasts"]["C64-B"]["delta_min"]
            # ★ 标签口径按 PREREG §4 D 臂**逐条**判（旧版先判"≥ −δ_eq ⇒ 未测出掉分"，会把
            #   `D−C64 ≥ +δ_eq`（捷径反而更好）误标成"未测出掉分" —— 文本与代码不一致，已修）
            if st["delta"] >= deq:
                tag = "捷径反而更好"
            elif st["delta"] >= -deq:
                tag = "未测出掉分"
            elif st["delta"] >= -dmin:
                tag = "带内（补 seed 再判）"
            else:
                tag = "掉分"
            darm[str(sd)] = dict(delta=st["delta"], se=st["se"], delta_eq=deq, tag=tag)
            print(f"    s{sd}: D−C64 = {st['delta']*100:+.2f}pp ± {st['se']*100:.2f}pp（δ_eq={deq*100:.2f}）⇒ {tag}")
        payload["d_arm"] = darm
    print(f"\n【档 {code}】{BRANCH[code]}：{txt}")
    if code == "7":
        for p in problems:
            print(f"  - {p}")
    print("==================================================================")

    outp = ROOT / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[落盘] 关键数字 → {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
