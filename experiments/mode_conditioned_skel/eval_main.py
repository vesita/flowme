#!/usr/bin/env python3
"""P8 主评测：同一模型/同一 logits，只换「是否按句式条件化」的解码。PREREG §2。

解码臂：F / C-rule / C-pred / C-pred-ML / C-soft / C-soft-lr /
        C-rule-B / C-pred-B（映射 B 单列）/ 随机标签对照 / 结构保持置换对照。
出口：test、adv2（污染，分 L / 非 L）、a_bal（无泄露，L 组 n=0）。

用法：uv run python experiments/mode_conditioned_skel/eval_main.py [--smoke] [--arm UP]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import modes as MM  # noqa: E402

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
SPLIT_PATH = {
    "test": C.TCH / "data" / "test.jsonl",
    "adv2": C.SS / "data" / "adv2.jsonl",
    "a_bal": C.SLEAK / "data" / "a_bal.jsonl",
}
L_GROUP = (35, 0, 1, 2)
DECODES = ("F", "C_rule", "C_pred", "C_predML", "C_soft", "C_softlr",
           "C_rule_B", "C_pred_B", "C_rule_rand", "C_pred_rand", "C_perm")


def se(n: int) -> float:
    return math.sqrt(0.25 / n)


def mask_argmax(logits: torch.Tensor, keep: list[list[int]] | None) -> list[int]:
    if keep is None:
        return logits.argmax(-1).tolist()
    out = []
    for i, ks in enumerate(keep):
        if not ks:                       # 空子集 ⇒ 不掩码（回退 F），计数另行报出
            out.append(int(logits[i].argmax()))
            continue
        m = torch.full((logits.shape[1],), float("-inf"), dtype=logits.dtype)
        m[ks] = 0.0
        out.append(int((logits[i] + m).argmax()))
    return out


def acc_of(pred: list[int], gold: list[int]) -> float:
    return sum(1 for a, b in zip(pred, gold) if a == b) / len(gold)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--arm", default=None, help="只跑某个臂")
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = C.Spec()
    torch.manual_seed(0)
    tab = MM.skeleton_table()
    print(f"[device] {device}", flush=True)

    mapj = json.loads((RESULTS / "mapping.json").read_text(encoding="utf-8"))
    sA = {m: v for m, v in mapj["subsets_A"].items()}
    sB = {m: v for m, v in mapj["subsets_B"].items()}
    probe = json.loads((RESULTS / "probe_pred.json").read_text(encoding="utf-8"))["seeds"]

    rows = {s: C.load_rows(p) for s, p in SPLIT_PATH.items()}
    if a.smoke:
        for s in rows:
            rows[s] = rows[s][:100]
    blob = {s: C.encode_rows(rows[s], spec, device) for s in SPLIT_PATH}
    labs = {s: C.label_tensors(rows[s]) for s in SPLIT_PATH}

    # ---- 行级句式标签（一次性算好）----
    gold_mode = {s: [MM.MASK_ALIAS[MM.mode_of_skel(tab[r["skel_id"]])] for r in rows[s]]
                 for s in rows}
    rule_mode = {s: [MM.MASK_ALIAS[MM.mode_of_sent(r["sent"])] for r in rows[s]]
                 for s in rows}
    prob = {sd: {s: torch.tensor(probe[str(sd)]["pred"][s]["probs"], dtype=torch.float32)
                 for s in SPLIT_PATH} for sd in C.SEEDS}
    if a.smoke:                       # 冒烟只截断到前 n 行（探针预测是全量算的）
        for sd in C.SEEDS:
            for s in SPLIT_PATH:
                prob[sd][s] = prob[sd][s][:len(rows[s])]
    pmode = {sd: {s: prob[sd][s].argmax(-1).tolist() for s in SPLIT_PATH}
             for sd in C.SEEDS}
    ml_mode = {sd: {s: [MM.MASK_ALIAS[MM.MODES[i]] for i in pmode[sd][s]]
                    for s in SPLIT_PATH} for sd in C.SEEDS}
    # train 先验（C-soft-lr 用，目标 = A(gold) 的 train 分布）
    tr_rows = C.load_rows(C.TCH / "data" / "train.jsonl")
    prior_c = Counter(MM.MASK_ALIAS[MM.mode_of_skel(tab[r["skel_id"]])] for r in tr_rows)
    n_tr = len(tr_rows)
    pi = {m: prior_c.get(m, 0) / n_tr for m in ("陈述", "疑问", "祈使")}
    print(f"[prior] train 模式先验 = { {k: round(v, 4) for k, v in pi.items()} }", flush=True)
    del tr_rows

    # ---- 随机对照的标签（seed*1000+99，同 skeleton_leak Q5 口径）----
    rnd_mode, perm_keep = {}, {}
    for s in SPLIT_PATH:
        n = len(rows[s])
        for sd in C.SEEDS:
            g = torch.Generator().manual_seed(sd * 1000 + 99)
            p = torch.randperm(n, generator=g)
            rnd_mode[(sd, s, "gold")] = [gold_mode[s][int(i)] for i in p]
            rnd_mode[(sd, s, "rule")] = [rule_mode[s][int(i)] for i in p]
        # 结构保持置换：同尺寸随机划分 40 个骨架
        g2 = torch.Generator().manual_seed(77001)
        ids = torch.randperm(40, generator=g2).tolist()
        sizes = [(m, len(sA[m])) for m in MM.MODES]
        part, pos = {}, 0
        for m, sz in sizes:
            part[m] = sorted(ids[pos:pos + sz])
            pos += sz
        perm_keep[s] = part
    print(f"[rand] 结构保持置换子集大小 = "
          f"{ {m: len(v) for m, v in perm_keep['test'].items()} }", flush=True)

    arms = (a.arm,) if a.arm else C.ARMS
    out: dict = {"device": device, "n": {s: len(rows[s]) for s in SPLIT_PATH},
                 "sA": sA, "sB": sB, "pi": pi, "rows": {}}
    fallback = {}

    for arm in arms:
        for seed in C.SEEDS:
            model = C.load_arm(arm, seed, spec, device)
            key = f"{arm}_s{seed}"
            out["rows"][key] = {}
            for s in SPLIT_PATH:
                t0 = time.time()
                with torch.no_grad():
                    vs = blob[s]["v_sent"].to(device)
                    vi = blob[s]["v_items"].to(device)
                    im = blob[s]["item_mask"].to(device)
                    vb = C.v_bag_of(vi, im)
                    lab_in = None
                    if model.lab is not None:
                        lab_in = {k: labs[s][k].to(device) for k in
                                  ("type_t", "role_t", "cls_t", "pos_b")}
                        lab_in["mask"] = labs[s]["mask"].to(device)
                    h_tok = tmask = None
                    if model.pool is not None:
                        h_tok = blob[s]["h"].to(device)
                        tmask = blob[s]["hmask"].to(device)
                    sk_logits, _, _ = model.forward(vs, vb, vi, im, lab_in,
                                                     h_tok, tmask)
                logits = sk_logits.cpu()
                gold = blob[s]["skel"].tolist()

                # --- 各解码臂 ---
                keep_A_g = [sA[m] for m in gold_mode[s]]
                keep_A_r = [sA[m] for m in rule_mode[s]]
                keep_A_m = [sA[m] for m in ml_mode[seed][s]]
                keep_B_g = [sB.get(m, []) for m in gold_mode[s]]
                keep_B_r = [sB.get(m, []) for m in rule_mode[s]]
                keep_A_rg = [sA[m] for m in rnd_mode[(seed, s, "gold")]]
                keep_A_rr = [sA[m] for m in rnd_mode[(seed, s, "rule")]]
                keep_P = [perm_keep[s][m] for m in gold_mode[s]]

                bias = torch.zeros_like(logits)
                bias_lr = torch.zeros_like(logits)
                # 逐骨架：偏置 = log p(A(s) | x) [− log pi(A(s))]，λ=1 写死
                mode_of_skel_ = [MM.MASK_ALIAS[MM.mode_of_skel(tab[i])] for i in range(40)]
                lg_p = torch.log(prob[seed][s].clamp(min=1e-9))          # [n,5]
                for sid in range(40):
                    mi = MM.MODES.index(mode_of_skel_[sid])
                    if mode_of_skel_[sid] not in pi:
                        continue
                    bias[:, sid] = lg_p[:, mi]
                    bias_lr[:, sid] = lg_p[:, mi] - math.log(pi[mode_of_skel_[sid]])

                preds = {
                    "F": mask_argmax(logits, None),
                    "C_rule": mask_argmax(logits, keep_A_g),
                    "C_pred": mask_argmax(logits, keep_A_r),
                    "C_predML": mask_argmax(logits, keep_A_m),
                    "C_soft": mask_argmax(logits + bias, None),
                    "C_softlr": mask_argmax(logits + bias_lr, None),
                    "C_rule_B": mask_argmax(logits, keep_B_g),
                    "C_pred_B": mask_argmax(logits, keep_B_r),
                    "C_rule_rand": mask_argmax(logits, keep_A_rg),
                    "C_pred_rand": mask_argmax(logits, keep_A_rr),
                    "C_perm": mask_argmax(logits, keep_P),
                }
                fb = {d: sum(1 for ks in v if not ks)
                      for d, v in (("C_rule", keep_A_g), ("C_pred", keep_A_r),
                                   ("C_rule_B", keep_B_g), ("C_pred_B", keep_B_r))}
                fallback[f"{key}|{s}"] = fb
                rec = {}
                for d, p in preds.items():
                    corr = [1 if x == y else 0 for x, y in zip(p, gold)]
                    rec[d] = {"acc": round(acc_of(p, gold), 6), "se": round(se(len(gold)), 6),
                              "correct": corr, "pred": p}
                rec["gold"] = gold
                rec["gold_mode"] = gold_mode[s]
                rec["rule_mode"] = rule_mode[s]
                rec["ml_mode"] = ml_mode[seed][s]
                out["rows"][key][s] = rec
                print(f"[eval] {key} {s:6s} F={rec['F']['acc']:.4f} "
                      f"C_rule={rec['C_rule']['acc']:.4f} C_pred={rec['C_pred']['acc']:.4f} "
                      f"C_predML={rec['C_predML']['acc']:.4f} C_soft={rec['C_soft']['acc']:.4f} "
                      f"C_softlr={rec['C_softlr']['acc']:.4f} "
                      f"B(rule)={rec['C_pred_B']['acc']:.4f} "
                      f"rand={rec['C_rule_rand']['acc']:.4f}/{rec['C_pred_rand']['acc']:.4f} "
                      f"perm={rec['C_perm']['acc']:.4f} "
                      f"({time.time() - t0:.1f}s)", flush=True)
            del model
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

    out["fallback_empty_subset"] = fallback
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / "eval.json").write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[done] → {RESULTS / 'eval.json'}", flush=True)


if __name__ == "__main__":
    main()
