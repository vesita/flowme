#!/usr/bin/env python3
"""P21 评测：held-out 最小对 2AFC（主）+ 40 类 argmax（并列）+ 机会实测 + 地板 + 机制探针。

口径 = `PREREG.md` §1/§4/§5：
  · 2AFC = 受限该对 {s,t} 后两侧都选对（PREREG §1 原文）；40 类 argmax = P4/`skeleton_leak` 同口径；
  · 机会**必须实测**：机会_R（独立均匀随机 MC 2000 轮）/ 机会_P（本臂 logits 置换 1000 次）/ 盲预测器；
  · 地板：naive 电池（fit=train）+ `fw_decision_list`（= R-chain 对应物）+ `n_slots` 规则（L/非 L 分列）；
  · 探针：`shuffle_content`（pair 版）/ `shuffle_all` / `aux_only`（不适用 → n_slots 查表列）。

用法：uv run python experiments/minpair_supervision/eval.py --arm MP --seed 42 [--randlabel]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "funcword_minpair"))
sys.path.insert(0, str(HERE))

from train import (ARMS, CACHE, RESULTS, WEIGHTS, Spec, StructSupModel, check,  # noqa: E402
                   load_blob, load_rows, build_pairs, split, v_bag_of)

import eval_arms as FMEA  # noqa: E402  只读复用（P4）：forward_all / naive_preds / make_content_shuffle

TCH_DATA = ROOT / "experiments" / "two_channel_head" / "data"
LEAK = ROOT / "experiments" / "skeleton_leak" / "results" / "leak_stats.json"
L_SET = {35, 0, 1, 2}
CHANCE_SEED = 20261007
REPS_R, REPS_P = 2000, 1000


# ---------------------------------------------------------------------------
def pair_index(pairs: list[tuple[int, int]], sel: list[int]) -> list[tuple[int, int]]:
    return [pairs[k] for k in sel]


def twoafc_list(la: torch.Tensor, lb: torch.Tensor, gi: list[int], gj: list[int]) -> list[int]:
    n = len(gi)
    ig = torch.tensor(gi)
    jg = torch.tensor(gj)
    a_gi = la.gather(1, ig.view(-1, 1)).squeeze(1)
    a_gj = la.gather(1, jg.view(-1, 1)).squeeze(1)
    b_gi = lb.gather(1, ig.view(-1, 1)).squeeze(1)
    b_gj = lb.gather(1, jg.view(-1, 1)).squeeze(1)
    ok_a = a_gi >= a_gj          # argmax 取靠前索引 ⇒ 平局判 gold 在前为「对」
    ok_b = b_gj > b_gi           # gold 在后 ⇒ 必须严格更大
    return [int(x) for x in (ok_a & ok_b).tolist()]


def pair_metrics(lg: torch.Tensor, gold: list[int], prs: list[tuple[int, int]]) -> dict:
    """lg: [N,40] logits；返回 2AFC 与 40 类 argmax 两个口径（含逐对 0/1 列表）。"""
    idx_a = [i for i, _ in prs]
    idx_b = [j for _, j in prs]
    gi = [gold[i] for i in idx_a]
    gj = [gold[j] for _, j in prs]
    check(all(s != t for s, t in zip(gi, gj)), "对内 gold 相同")
    la, lb = lg[idx_a], lg[idx_b]
    a2 = twoafc_list(la, lb, gi, gj)
    pa = la.argmax(1).tolist()
    pb = lb.argmax(1).tolist()
    p40 = [int(pa[k] == gi[k] and pb[k] == gj[k]) for k in range(len(prs))]
    side = [int(pa[k] == gi[k]) for k in range(len(prs))] + \
           [int(pb[k] == gj[k]) for k in range(len(prs))]
    n = len(prs)
    se = math.sqrt(0.25 / n) if n else 0.0
    m2, m40 = sum(a2) / n if n else 0.0, sum(p40) / n if n else 0.0
    return {"n_pairs": n, "twoafc": round(m2, 6), "pair_success_40": round(m40, 6),
            "per_side_acc": round(sum(side) / (2 * n), 6) if n else 0.0,
            "se_pair": round(se, 6), "pred_differs": round(
                sum(int(pa[k] != pb[k]) for k in range(n)) / n, 6) if n else 0.0,
            "twoafc_list": a2, "pair40_list": p40}


# ---------------------------------------------------------------------------
# 机会（M0，必须实测）
# ---------------------------------------------------------------------------
def chance_uniform(n: int) -> dict:
    g = torch.Generator().manual_seed(CHANCE_SEED)
    vals = []
    for _ in range(REPS_R):
        a = torch.rand(n, generator=g) < 0.5
        b = torch.rand(n, generator=g) < 0.5
        vals.append(float((a & b).float().mean()))
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))
    return {"reps": REPS_R, "seed": CHANCE_SEED, "mean": round(mu, 6),
            "rep_sd": round(sd, 6), "se_of_mean": round(sd / math.sqrt(REPS_R), 6),
            "se_metric": round(math.sqrt(0.25 / n), 6), "n_pairs": n}


def chance_perm(lg: torch.Tensor, gold: list[int], prs: list[tuple[int, int]]) -> dict:
    idx_a = [i for i, _ in prs]
    idx_b = [j for _, j in prs]
    gi = torch.tensor([gold[i] for i in idx_a])
    gj = torch.tensor([gold[j] for _, j in prs])
    LA, LB = lg[idx_a], lg[idx_b]
    n = len(prs)
    g = torch.Generator().manual_seed(CHANCE_SEED)
    vals = []
    for _ in range(REPS_P):
        perm = torch.randperm(n, generator=g)
        la, lb = LA[perm], LB[perm]
        a_gi = la.gather(1, gi.view(-1, 1)).squeeze(1)
        a_gj = la.gather(1, gj.view(-1, 1)).squeeze(1)
        b_gi = lb.gather(1, gi.view(-1, 1)).squeeze(1)
        b_gj = lb.gather(1, gj.view(-1, 1)).squeeze(1)
        vals.append(float(((a_gi >= a_gj) & (b_gj > b_gi)).float().mean()))
    mu = sum(vals) / len(vals)
    sd = math.sqrt(sum((x - mu) ** 2 for x in vals) / (len(vals) - 1))
    # 盲/常值预测器：两侧同 logits ⇒ 同代码路径实测
    blind = sum(twoafc_list(torch.zeros_like(LA), torch.zeros_like(LB),
                            gi.tolist(), gj.tolist())) / n
    return {"reps": REPS_P, "seed": CHANCE_SEED, "mean": round(mu, 6),
            "sd": round(sd, 6), "se_of_mean": round(sd / math.sqrt(REPS_P), 6),
            "blind_const": round(blind, 6), "n_pairs": n}


# ---------------------------------------------------------------------------
def rule_pair_metrics(preds: list, gold: list[int],
                      prs: list[tuple[int, int]]) -> dict:
    """规则（单一类别预测）走同一 2AFC 代码路径：one-hot → 受限 argmax。
    `preds` 为**绝对下标**长列表（非 held-out 行可为 None）。"""
    n = len(prs)
    if not n:
        return {"n_pairs": 0}
    idx = sorted({i for p in prs for i in p})
    check(all(preds[i] is not None for i in idx), "规则预测缺失行")
    lg = torch.full((len(preds), 40), float("-inf"))
    lg[torch.tensor(idx), torch.tensor([int(preds[i]) for i in idx])] = 0.0
    m = pair_metrics(lg, gold, prs)
    return {"n_pairs": n, "twoafc": m["twoafc"], "pair_success_40": m["pair_success_40"],
            "per_side_acc": m["per_side_acc"]}


# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_rows(model, rows: list[dict], spec, device: str, batch: int = 128) -> dict:
    """探针新句编码：只写本目录 cache/。"""
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    CACHE.mkdir(exist_ok=True)
    import hashlib
    key = hashlib.md5("".join(r["sent"] + str(r["bag_span"]) for r in rows)
                      .encode()).hexdigest()[:12]
    path = CACHE / f"probe_{key}_L{spec.max_len_sent}.pt"
    if path.exists():
        b = torch.load(path, map_location="cpu", weights_only=True)
        if b["n"] == len(rows):
            print(f"[enc] 命中 {path.name}（n={b['n']}）", flush=True)
            return b
    v_sent, v_items, mask, skel, assign = [], [], [], [], []
    model.encoder.eval()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        ids, msk = [], []
        for r in chunk:
            e = tok.encode(r["sent"], max_length=spec.max_len_sent, padding=True)
            check(sum(e["attention_mask"]) == len(r["sent"]), "1 字符 1 token 前提被破坏")
            ids.append(e["input_ids"])
            msk.append(e["attention_mask"])
        id_t = torch.tensor(ids, dtype=torch.long, device=device)
        m_t = torch.tensor(msk, dtype=torch.bool, device=device)
        h = model.encoder(id_t, m_t).cpu()
        m3 = m_t.cpu().unsqueeze(-1).to(h.dtype)
        v_sent.append((h * m3).sum(1) / m3.sum(1).clamp(min=1.0))
        items = torch.zeros(len(chunk), spec.max_slots, spec.hidden)
        im = torch.zeros(len(chunk), spec.max_slots, dtype=torch.bool)
        for j, r in enumerate(chunk):
            for s, (a, b) in enumerate(r["bag_span"]):
                check(r["sent"][a:b] == r["bag"][s], "span 与 bag 不一致")
                items[j, s] = h[j, a:b].mean(0)
                im[j, s] = True
        v_items.append(items)
        mask.append(im)
        skel += [r["skel_id"] for r in chunk]
        assign += [r["assign"] + [-1] * (spec.max_slots - len(r["assign"])) for r in chunk]
    blob = {"v_sent": torch.cat(v_sent), "v_items": torch.cat(v_items),
            "item_mask": torch.cat(mask), "skel": torch.tensor(skel),
            "assign": torch.tensor(assign), "n": len(rows)}
    torch.save(blob, path)
    print(f"[enc] 写入 {path.name}（n={len(rows)}）", flush=True)
    return blob


def evaluate(arm: str, seed: int, rand: bool, device: str) -> dict:
    spec = Spec()
    name = f"{arm}_s{seed}{'_rand' if rand else ''}"
    model = StructSupModel("B", seed, spec, vocab=None).to(device)
    sd = torch.load(WEIGHTS / f"{name}.pt", map_location="cpu", weights_only=True)
    miss = [k for k in model.state_dict() if k not in sd]
    check(all(k.startswith("encoder.") for k in miss), f"{name} 缺键异常 {miss[:5]}")
    check(not [k for k in sd if k not in model.state_dict()], f"{name} 多键")
    model.load_state_dict(sd, strict=False)
    model.eval()

    rows = load_rows()
    prs_all = build_pairs(rows)
    tr_rows, ho_rows, st = split(rows, prs_all)          # 跑后分离断言（fail-closed）
    ho_pairs_idx = st["split_heldout_pairs"]
    tr_pairs_idx = st["split_train_pairs"]
    blob = load_blob()
    res = FMEA.forward_all(model, blob, device)
    gold = blob["skel"].tolist()
    lg = res["logits"]
    check(lg.shape[0] == 1120, "logits 行数漂移")

    ho_prs = pair_index(prs_all, ho_pairs_idx)
    full_prs = prs_all
    b1 = [k for k in ho_pairs_idx if rows[prs_all[k][0]]["kind"] == "b1_src"]
    b2 = [k for k in ho_pairs_idx if rows[prs_all[k][0]]["kind"] == "b2_src"]
    out: dict = {"name": name, "arm": arm, "seed": seed, "randlabel": rand,
                 "device": device, "split": {k: v for k, v in st.items()
                                             if k not in ("split_train_pairs",
                                                          "split_heldout_pairs")},
                 "heldout": {}, "chance": {}, "floor": {}, "probe": {}}

    out["heldout"]["b_all"] = pair_metrics(lg, gold, ho_prs)
    out["heldout"]["b1_word"] = pair_metrics(lg, gold, pair_index(prs_all, b1))
    out["heldout"]["b2_order"] = pair_metrics(lg, gold, pair_index(prs_all, b2))
    out["heldout"]["train_half_insample"] = pair_metrics(
        lg, gold, pair_index(prs_all, tr_pairs_idx))
    out["full560_insample"] = pair_metrics(lg, gold, full_prs)

    # ---- 机会（M0）----
    n = len(ho_prs)
    out["chance"]["uniform_random_R"] = chance_uniform(n)
    out["chance"]["permutation_P"] = chance_perm(lg, gold, ho_prs)
    cR, cP = out["chance"]["uniform_random_R"], out["chance"]["permutation_P"]
    thr = max(cR["mean"] + 2 * cR["se_metric"], cP["mean"] + 2 * cP["se_of_mean"])
    out["chance"]["threshold"] = round(thr, 6)
    out["chance"]["gate_expr"] = ("max(机会_R + 2*sqrt(0.25/n), 机会_P + 2*se_of_mean)")
    h = out["heldout"]["b_all"]
    out["chance"]["M1_pass"] = bool(h["twoafc"] > thr)

    # ---- 地板（M3）----
    fit = [json.loads(x) for x in open(TCH_DATA / "train.jsonl", encoding="utf-8")]
    leak = json.loads(LEAK.read_text(encoding="utf-8"))
    tab = leak["fit"]["n_slots_rule_table"]
    check(set(tab.values()) == L_SET, f"n_slots 规则输出 {set(tab.values())} != L")
    glob = leak["fit"]["train_majority"]
    ho_rows_d = [rows[i] for i in ho_rows]                # 绝对下标 → 行 dict（顺序不变）
    P = FMEA.naive_preds(fit, ho_rows_d)                  # 局部顺序 = ho_rows
    P["n_slots_rule"] = [tab.get(str(r["n_slots"]), glob) for r in ho_rows_d]
    hy = [r["skel_id"] for r in ho_rows_d]                # 局部下标（算 acc）

    def to_abs(pl: list[int]) -> list:
        o: list = [None] * 1120
        for local, a in enumerate(ho_rows):
            o[a] = pl[local]
        return o

    L_pairs = [(i, j) for i, j in ho_prs if gold[i] in L_SET or gold[j] in L_SET]
    nonL_pairs = [(i, j) for i, j in ho_prs if gold[i] not in L_SET
                  and gold[j] not in L_SET]
    by_rule: dict = {}
    for k, v in P.items():
        acc = sum(int(p == y) for p, y in zip(v, hy)) / len(hy)
        rec = {"acc": round(acc, 6)}
        rec.update(rule_pair_metrics(to_abs(v), gold, ho_prs))
        if k == "n_slots_rule":
            rec["L_group"] = rule_pair_metrics(to_abs(v), gold, L_pairs)
            rec["nonL_group"] = rule_pair_metrics(to_abs(v), gold, nonL_pairs)
        by_rule[k] = rec
    out["floor"] = {"by_rule": by_rule}
    out["floor"]["max_naive_complete"] = round(
        max(r["acc"] for r in by_rule.values()), 6)
    out["floor"]["max_naive_complete_rule"] = max(by_rule, key=lambda k: by_rule[k]["acc"])
    out["floor"]["R_chain_ref"] = {
        "rule": "fw_decision_list（功能词字面决策链，fit=train 既有口径）",
        "note": "syllogism_card 的字符串链 R-chain 属三段论任务，本骨架任务不适用",
        **by_rule["fw_decision_list"]}
    out["floor"]["max_pair_success_rule"] = max(
        by_rule, key=lambda k: by_rule[k]["pair_success_40"])
    out["floor"]["max_twoafc_rule"] = max(by_rule, key=lambda k: by_rule[k]["twoafc"])
    out["floor"]["L_split"] = {"L_set": sorted(L_SET), "n_L_pairs": len(L_pairs),
                               "n_nonL_pairs": len(nonL_pairs)}

    # ---- 机制探针（M5）----
    hold_rows = [rows[i] for k in ho_pairs_idx for i in prs_all[k]]
    sh_rows, diag = FMEA.make_content_shuffle(hold_rows, 2)
    ok_pairs = [kk for kk, k in enumerate(ho_pairs_idx)
                if sh_rows[2 * kk] is not None and sh_rows[2 * kk + 1] is not None]
    rec_sc: dict = {"build": diag, "n_survivor_pairs": len(ok_pairs),
                    "n_total_pairs": len(ho_pairs_idx)}
    if ok_pairs:
        sub = [x for kk in ok_pairs for x in (sh_rows[2 * kk], sh_rows[2 * kk + 1])]
        blob_sh = encode_rows(model, sub, spec, device)
        r_sh = FMEA.forward_all(model, blob_sh, device)
        prs_sh = [(2 * a, 2 * a + 1) for a in range(len(ok_pairs))]
        rec_sc["shuffled"] = pair_metrics(r_sh["logits"], blob_sh["skel"].tolist(), prs_sh)
        t_idx = [i for kk in ok_pairs for i in ho_prs[kk]]
        rec_sc["true_on_survivors"] = pair_metrics(lg[t_idx], [gold[i] for i in t_idx],
                                                   prs_sh)
        print(f"[{name}] shuffle_content 幸存 {len(ok_pairs)}/{len(ho_pairs_idx)} "
              f"2AFC {rec_sc['true_on_survivors']['twoafc']}→"
              f"{rec_sc['shuffled']['twoafc']}", flush=True)
    out["probe"]["shuffle_content"] = rec_sc

    # shuffle_all：held-out 行输入跨行置换，gold 不动
    hg = torch.tensor(ho_rows, dtype=torch.long)
    blob_ho = {k: (blob[k][hg] if k in ("v_sent", "v_items", "item_mask", "skel", "assign")
                   else blob[k]) for k in ("v_sent", "v_items", "item_mask", "skel", "assign")}
    blob_ho["n"] = len(ho_rows)
    g = torch.Generator().manual_seed(seed * 1000 + 99)
    perm = torch.randperm(blob_ho["n"], generator=g)
    blob_perm = {k: (blob_ho[k][perm] if k in ("v_sent", "v_items", "item_mask")
                     else blob_ho[k]) for k in blob_ho}
    r_all = FMEA.forward_all(model, blob_perm, device)
    rel_prs = [(2 * a, 2 * a + 1) for a in range(len(ho_pairs_idx))]
    out["probe"]["shuffle_all"] = pair_metrics(r_all["logits"], blob_ho["skel"].tolist(),
                                               rel_prs)
    # held-out 真值（相对下标口径，用于同表对比）
    blob_true = {k: blob_ho[k] for k in blob_ho}
    r_true = FMEA.forward_all(model, blob_true, device)
    out["probe"]["shuffle_all_true"] = pair_metrics(r_true["logits"], blob_true["skel"].tolist(),
                                                    rel_prs)
    out["probe"]["aux_only"] = {
        "applicable": False,
        "reason": "本模型（struct_supervision B 口径）无标签通道 ⇒ bag_modules 的 aux_only 原口径不适用",
        "replacement_n_slots_rule": out["floor"]["by_rule"]["n_slots_rule"]}

    del model
    RESULTS.mkdir(exist_ok=True)
    p = RESULTS / f"eval_{name}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[{name}] heldout 2AFC={h['twoafc']:.4f} 40argmax={h['pair_success_40']:.4f} "
          f"门槛={thr:.4f} 机会R={cR['mean']:.4f} 机会P={cP['mean']:.4f} "
          f"blind={cP['blind_const']:.4f} → M1={'PASS' if h['twoafc'] > thr else 'NO'}",
          flush=True)
    print(f"[done] → {p}", flush=True)
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43])
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--device", default="cpu")
    a = ap.parse_args(argv)
    evaluate(a.arm, a.seed, a.randlabel, a.device)


if __name__ == "__main__":
    main(sys.argv[1:])
