#!/usr/bin/env python3
"""P4 评测：held-out 最小对 pair_success + a_bal/test + L 分列 + 机制探针 + 完备免费地图。

口径 = `PREREG.md` §3（跑前写死）：
  · 所有评测集与编码缓存**只读复用**（two_channel_head / skeleton_leak），本目录只写 cache/ 与 results/；
  · `pair_success` = 两侧非受限 40 类 argmax 都对的比例（机会 0.5，= skeleton_leak 同口径）；
    另报 `pair_2afc`（限制到该对 {s,t} 后两侧都对）与 per-side acc / t_vs_0.5；
  · 机制探针 `shuffle_content`（保留骨架字面+mask，跨行换内容词，失败行剔除、true 与 shuf 同批配对）、
    `shuffle_all`（整行输入含 mask 跨行置换）、`pair_shuffle_content`；
    本模型**无标签通道** ⇒ bag_modules 的 `aux_only` 不适用，替代口径 = 只吃 n_slots 的查表读出；
  · `max_naive_complete`（fit=train，含 n_slots + 本集多数类）与 **pair 地板**（逐规则 pair_success 取 max）。

用法：
  uv run python experiments/funcword_minpair/eval_arms.py --arm M --seed 42
  uv run python experiments/funcword_minpair/eval_arms.py --arm MP --seed 43 --randlabel
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "two_channel_head"))
sys.path.insert(0, str(ROOT / "experiments" / "funcword_minpair"))

from build_gen_data import (  # noqa: E402  只读
    FUNCTION_WORDS, PUNCTS, SKELS, _acc, _fit_predict, _majority, _tree_depth,
    feats, match_sentence, naive_skeleton, ok_sentence,
)

from model import (ARMS, StructSupModel, Spec, check, get_blob,  # noqa: E402
                   encode_rows, v_bag_of)

HERE = Path(__file__).resolve().parent
ROOTEX = ROOT / "experiments"
TCH_DATA = ROOTEX / "two_channel_head" / "data"
SL_DATA = ROOTEX / "skeleton_leak" / "data"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
LEAK = ROOTEX / "skeleton_leak" / "results" / "leak_stats.json"

SPLITS = ("test", "a_bal", "a_lit", "b_pairs", "c_pairs")
PROBE_SPLITS = ("test", "a_bal", "b_pairs")
SHUF_SEED_BASE = 1000


def load_rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def get_rows(split: str) -> list[dict]:
    if split == "test":
        return load_rows(TCH_DATA / "test.jsonl")
    return load_rows(SL_DATA / f"{split}.jsonl")


# ---------------------------------------------------------------------------
# 机制探针的行构造
# ---------------------------------------------------------------------------
def _bucket_shift(rows: list[dict], shift: int) -> tuple[list[int], int]:
    """按 n_slots 分桶、桶内循环移位 `shift` ⇒ 返回 (donor[i])；桶 < shift+1 报不可构造。"""
    by: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(rows):
        by[r["n_slots"]].append(i)
    donor = [-1] * len(rows)
    bad = 0
    for n, idx in by.items():
        if len(idx) <= shift:
            bad += len(idx)
            continue
        for t, i in enumerate(idx):
            donor[i] = idx[(t + shift) % len(idx)]
    return donor, bad


def make_content_shuffle(rows: list[dict], shift: int) -> tuple[list[dict | None], dict]:
    """只换内容词：用 donor 的**句序槽文本**灌进本行骨架模板；fail-closed 验证。"""
    donor, bad = _bucket_shift(rows, shift)
    out: list[dict | None] = [None] * len(rows)
    fail: Counter = Counter()
    for i, r in enumerate(rows):
        j = donor[i]
        if j < 0:
            fail["桶太小"] += 1
            continue
        g = [rows[j]["sent"][a:b] for a, b in sorted(rows[j]["bag_span"])]
        if g == [r["sent"][a:b] for a, b in sorted(r["bag_span"])]:
            fail["与原槽相同"] += 1
            continue
        parts_ = __import__("re").split(r"\[\d+\]", SKELS[r["skel_id"]].template)
        v = parts_[0]
        for gi, pi in zip(g, parts_[1:]):
            v += gi + pi
        if not ok_sentence(v):
            fail["ok_sentence拒"] += 1
            continue
        m = match_sentence(v)
        if m is None:
            fail["无匹配"] += 1
            continue
        sid, g2, sp2 = m
        if sid != r["skel_id"]:
            fail[f"错骨架#{sid}"] += 1
            continue
        if g2 != g:
            fail["槽文本不一致"] += 1
            continue
        out[i] = {"sent": v, "skel_id": r["skel_id"], "n_slots": r["n_slots"],
                  "bag": list(g2), "bag_span": [list(x) for x in sp2],
                  "assign": list(range(len(g2)))}
    n_ok = sum(1 for x in out if x is not None)
    return out, {"ok": n_ok, "fail": dict(fail), "bucket_too_small": bad}


# ---------------------------------------------------------------------------
# 模型前向与指标
# ---------------------------------------------------------------------------
@torch.no_grad()
def forward_all(model, blob: dict, device: str, bs: int = 4096) -> dict:
    model.eval()
    logits, sk_ok, slot_frac, joint_ok, preds = [], [], [], [], []
    for i in range(0, blob["n"], bs):
        j = min(i + bs, blob["n"])
        vs = blob["v_sent"][i:j].to(device)
        vi = blob["v_items"][i:j].to(device)
        im = blob["item_mask"][i:j].to(device)
        vb = v_bag_of(vi, im)
        y = blob["skel"][i:j].to(device)
        a = blob["assign"][i:j].to(device)
        sk_l, a_l = model.forward_gen(vs, vb, vi, im)
        p = sk_l.argmax(-1)
        pa = a_l.argmax(-1)
        logits.append(sk_l.cpu())
        preds += p.tolist()
        for r in range(j - i):
            k = int(im[r].sum())
            right = sum(1 for s in range(k) if int(pa[r, s]) == int(a[r][s]))
            slot_frac.append(right / k)
            ok = int(p[r] == int(y[r]))
            sk_ok.append(ok)
            joint_ok.append(int(ok and right == k))
    return {"logits": torch.cat(logits) if logits else torch.zeros(0, 40),
            "skel_ok": sk_ok, "slot_frac": slot_frac, "joint_ok": joint_ok,
            "pred": preds}


def metrics(res: dict) -> dict:
    n = len(res["skel_ok"])
    se = math.sqrt(0.25 / n)
    mu = sum(res["slot_frac"]) / n
    s2 = sum((x - mu) ** 2 for x in res["slot_frac"]) / (n - 1) if n > 1 else 0.0
    se_slot = math.sqrt(s2) / math.sqrt(n)
    return {"n": n,
            "skel": {"acc": round(sum(res["skel_ok"]) / n, 6), "se": round(se, 6)},
            "slot": {"acc": round(mu, 6), "se": round(se_slot, 6)},
            "joint": {"acc": round(sum(res["joint_ok"]) / n, 6), "se": round(se, 6)}}


def paired_delta(a: list, b: list) -> dict:
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    mu = sum(d) / n
    var = sum((x - mu) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    s = math.sqrt(var / n)
    return {"delta": round(mu, 6), "se": round(s, 6),
            "t": round(mu / s, 3) if s > 0 else None, "n": n}


# ---------------------------------------------------------------------------
# 最小对指标
# ---------------------------------------------------------------------------
def pair_index(rows: list[dict], kinds: tuple[str, ...] | None = None) -> list[tuple[int, int]]:
    out = []
    for i, r in enumerate(rows):
        k = r.get("kind", "")
        if not k.endswith("_src"):
            continue
        if kinds is not None and k not in kinds:
            continue
        check(i + 1 < len(rows), "b_pairs 源行无相邻变体")
        out.append((i, i + 1))
    return out


def pair_metrics(res: dict, prs: list[tuple[int, int]], n_classes: int = 40) -> dict:
    if not prs:
        return {"n_pairs": 0}
    lg = res["logits"]
    both = diff = m2 = 0
    for i, j in prs:
        gi, gj = int(res["gold"][i]), int(res["gold"][j])
        pi, pj = res["pred"][i], res["pred"][j]
        both += int(pi == gi and pj == gj)
        diff += int(pi != pj)
        la, lb = lg[i], lg[j]
        two = torch.stack([la[[gi, gj]], lb[[gi, gj]]])
        m2 += int(two[0].argmax() == 0 and two[1].argmax() == 1)
    n = len(prs)
    se = math.sqrt(0.25 / n)
    acc = sum(res["skel_ok"][i] for i, _ in prs
              ) / (2 * n) if n else 0.0
    se_side = math.sqrt(0.25 / (2 * n))
    both_list = [int(res["pred"][i] == res["gold"][i]
                     and res["pred"][j] == res["gold"][j]) for i, j in prs]
    side_list = [res["skel_ok"][i] for i, j in prs] + [res["skel_ok"][j] for i, j in prs]
    return {"n_pairs": n,
            "both_list": both_list, "side_list": side_list,
            "pair_success": round(both / n, 6), "se_pair": round(se, 6),
            "gate_0.5_2se": round(0.5 + 2 * se, 6),
            "over_0.5": round(both / n - 0.5, 6),
            "t_vs_0.5": round((both / n - 0.5) / se, 3) if se else None,
            "pair_2afc": round(m2 / n, 6),
            "pair_pred_differs": round(diff / n, 6),
            "per_side_acc": round(acc, 6), "per_side_se": round(se_side, 6),
            "per_side_t_vs_0.5": round((acc - 0.5) / se_side, 3) if se_side else None}


# ---------------------------------------------------------------------------
# 完备免费地图（fit=train）
# ---------------------------------------------------------------------------
def naive_preds(fit_rows: list[dict], eval_rows: list[dict]) -> dict[str, list[int]]:
    """与 `build_gen_data.naive_skeleton` 逐规则同式，但返回逐行预测（口径校验见下）。"""
    y = [r["skel_id"] for r in eval_rows]
    glob = _majority([r["skel_id"] for r in fit_rows])
    P: dict[str, list[int]] = {}
    P["majority"] = [glob] * len(y)
    P["len_bucket"] = _fit_predict(lambda r: len(r["sent"]) // 6, fit_rows, eval_rows)
    P["first_char"] = _fit_predict(lambda r: r["sent"][0], fit_rows, eval_rows)
    P["last_char"] = _fit_predict(lambda r: r["sent"][-1], fit_rows, eval_rows)
    P["punct_pattern"] = _fit_predict(
        lambda r: "".join(sorted(set(r["sent"]) & set(PUNCTS))), fit_rows, eval_rows)
    cnt = Counter(w for r in fit_rows for w in FUNCTION_WORDS if w in r["sent"])
    order_w = sorted(FUNCTION_WORDS, key=lambda w: -cnt.get(w, 0))
    best: dict[str, int] = {}
    for w in order_w:
        sub = [r["skel_id"] for r in fit_rows if w in r["sent"]]
        if len(sub) >= 30:
            best[w] = _majority(sub)
    preds = []
    for r in eval_rows:
        p = glob
        for w in order_w:
            if w in r["sent"] and w in best:
                p = best[w]
                break
        preds.append(p)
    P["fw_decision_list"] = preds
    Xf = [feats(r) for r in fit_rows]
    yf = [r["skel_id"] for r in fit_rows]
    Xq = [feats(r) for r in eval_rows]
    P["tree_depth2"] = _tree_depth(Xf, yf, Xq, 2)
    P["tree_depth4"] = _tree_depth(Xf, yf, Xq, 4)
    # 口径校验：逐规则 acc 必须 == naive_skeleton 输出（fail-closed）
    ref = naive_skeleton(fit_rows, eval_rows)
    for k in P:
        got = round(_acc(P[k], y), 4)
        check(abs(got - round(ref[k], 4)) < 1e-9, f"naive 预测器口径漂移 {k}: {got} vs {ref[k]}")
    return P


# ---------------------------------------------------------------------------
def build_floor(train: list[dict], splits_rows: dict, b_pairs_pairs: dict) -> dict:
    leak = json.loads(LEAK.read_text(encoding="utf-8"))
    fit_tab = leak["fit"]["n_slots_rule_table"]
    glob = leak["fit"]["train_majority"]
    out = {}
    for name, rows in splits_rows.items():
        y = [r["skel_id"] for r in rows]
        P = naive_preds(train, rows)
        nsl = [fit_tab.get(str(r["n_slots"]), glob) for r in rows]
        P["n_slots_rule"] = nsl
        maj_own = Counter(y).most_common(1)[0][0]
        P["majority_own"] = [maj_own] * len(y)
        accs = {k: round(_acc(v, y), 4) for k, v in P.items()}
        mx = max(accs.values())
        rec = {"acc_by_rule": accs, "max_naive_complete": mx,
               "max_naive_complete_rule": max(accs, key=accs.get),
               "majority_own": round(max(Counter(y).values()) / len(y), 4),
               "leak_stats_ref": {k: leak[name][k] for k in
                                  ("n", "n_slots_rule", "majority_own",
                                   "max_naive_complete", "max_naive_complete_rule")
                                  if k in leak.get(name, {})}}
        # 口径校验：n_slots / majority_own 必须与 skeleton_leak 公布值一致
        if name in leak:
            check(abs(accs["n_slots_rule"] - leak[name]["n_slots_rule"]) < 1e-9,
                  f"{name} n_slots 规则与 leak_stats 不一致")
        # pair 地板
        prs = b_pairs_pairs.get(name)
        if prs:
            pf = {}
            for k, v in P.items():
                both = sum(1 for i, j in prs if v[i] == y[i] and v[j] == y[j])
                pf[k] = round(both / len(prs), 4)
            rec["pair_floor_by_rule"] = pf
            rec["max_naive_pair_success"] = max(pf.values())
            rec["max_naive_pair_rule"] = max(pf, key=pf.get)
        out[name] = rec
    return out


# ---------------------------------------------------------------------------
def evaluate_model(arm: str, seed: int, rand: bool, device: str) -> dict:
    spec = Spec()
    name = f"{arm}_s{seed}{'_rand' if rand else ''}"
    model = StructSupModel("B", seed, spec, vocab=None).to(device)
    sd = torch.load(WEIGHTS / f"{name}.pt", map_location="cpu", weights_only=True)
    miss = [k for k in model.state_dict() if k not in sd]
    check(all(k.startswith("encoder.") for k in miss),
          f"{name} 权重缺键异常：{miss[:5]}")
    extra = [k for k in sd if k not in model.state_dict()]
    check(not extra, f"{name} 权重多键：{extra[:5]}")
    model.load_state_dict(sd, strict=False)
    model.eval()
    train = load_rows(TCH_DATA / "train.jsonl")
    rows = {s: get_rows(s) for s in SPLITS}
    blob = {s: get_blob(model, rows[s], spec, device, s) for s in SPLITS}

    out: dict = {"name": name, "arm": arm, "seed": seed, "randlabel": rand,
                 "splits": {}, "pair": {}, "probe": {}, "L_split": {},
                 "correct": {}}
    res_all: dict = {}
    for s in SPLITS:
        r = forward_all(model, blob[s], device)
        r["gold"] = blob[s]["skel"].tolist()
        res_all[s] = r
        m = metrics(r)
        out["splits"][s] = m
        out["correct"][s] = r["skel_ok"]
        print(f"[{name}] {s:8s} 骨架={m['skel']['acc']:.4f}±{m['skel']['se']:.4f} "
              f"槽位={m['slot']['acc']:.4f} 联合={m['joint']['acc']:.4f}", flush=True)

    # ---- 最小对 ----
    br = rows["b_pairs"]
    prs_all = pair_index(br)
    prs_b1 = pair_index(br, ("b1_src",))
    prs_b2 = pair_index(br, ("b2_src",))
    check(len(prs_b1) == 520 and len(prs_b2) == 40, "held-out 对数漂移")
    cr = rows["c_pairs"]
    grp: dict[str, list[int]] = defaultdict(list)
    for i, x in enumerate(cr):
        grp[x["pair"]].append(i)
    prs_c = []
    for _, idx in grp.items():
        h = len(idx) // 2
        prs_c += [(idx[j], idx[j + h]) for j in range(h)]
    out["pair"]["b1_word"] = pair_metrics(res_all["b_pairs"], prs_b1)
    out["pair"]["b2_order"] = pair_metrics(res_all["b_pairs"], prs_b2)
    out["pair"]["b_all"] = pair_metrics(res_all["b_pairs"], prs_all)
    out["pair"]["c_natural_nonminimal"] = pair_metrics(res_all["c_pairs"], prs_c)
    p1 = out["pair"]["b1_word"]
    print(f"[{name}] pair B1(MP-word) n={p1['n_pairs']} "
          f"pair_success={p1['pair_success']:.4f} 门槛={p1['gate_0.5_2se']:.4f} "
          f"2afc={p1['pair_2afc']:.4f} per_side={p1['per_side_acc']:.4f} "
          f"pred_differs={p1['pair_pred_differs']:.4f}", flush=True)
    p2 = out["pair"]["b2_order"]
    print(f"[{name}] pair B2(MP-order) n={p2['n_pairs']} "
          f"pair_success={p2['pair_success']:.4f} 门槛={p2['gate_0.5_2se']:.4f} "
          f"per_side={p2['per_side_acc']:.4f}", flush=True)

    # ---- L / 非 L 分列（PREREG §3.1）----
    Lset = {0, 1, 2, 35}
    for s in ("test", "a_lit"):
        idx_L = [i for i, r in enumerate(rows[s]) if r["skel_id"] in Lset]
        idx_n = [i for i, r in enumerate(rows[s]) if r["skel_id"] not in Lset]
        for tag, idx in (("L", idx_L), ("nonL", idx_n)):
            if not idx:
                out["L_split"].setdefault(s, {})[tag] = None
                continue
            sk = [res_all[s]["skel_ok"][i] for i in idx]
            out["L_split"].setdefault(s, {})[tag] = {
                "n": len(idx),
                "acc": round(sum(sk) / len(sk), 6),
                "se": round(math.sqrt(0.25 / len(idx)), 6)}
    # b_pairs：L 参与的对 vs 非 L 对
    for tag, prs in (("L", [(i, j) for i, j in prs_all
                            if rows["b_pairs"][i]["skel_id"] in Lset]),
                     ("nonL", [(i, j) for i, j in prs_all
                               if rows["b_pairs"][i]["skel_id"] not in Lset])):
        out["L_split"].setdefault("b_pairs", {})[tag] = pair_metrics(
            res_all["b_pairs"], prs)

    # ---- 机制探针 ----
    for s in PROBE_SPLITS:
        rs = rows[s]
        # b_pairs 行内相邻成对、两侧同袋 ⇒ 位移必须为 2 才能取到「另一对」的槽文本
        sh_rows, diag = make_content_shuffle(rs, 2 if s == "b_pairs" else 1)
        ok_idx = [i for i, x in enumerate(sh_rows) if x is not None]
        sub = [sh_rows[i] for i in ok_idx]
        blob_sh = encode_rows(model, sub, spec, device) if sub else None
        blob_true_sub = {k: (v[ok_idx] if k != "n" else len(ok_idx))
                         for k, v in blob[s].items()}
        r_true = forward_all(model, blob_true_sub, device)
        r_true["gold"] = blob_true_sub["skel"].tolist()
        r_sh = forward_all(model, blob_sh, device) if blob_sh else None
        rec: dict = {"build": diag, "n_survivor": len(ok_idx), "n_total": len(rs)}
        if r_sh is not None:
            r_sh["gold"] = blob_sh["skel"].tolist()
            t, h = metrics(r_true), metrics(r_sh)
            rec["true"] = t["skel"]
            rec["shuffled"] = h["skel"]
            rec["drop"] = round(t["skel"]["acc"] - h["skel"]["acc"], 6)
            rec["paired"] = paired_delta(r_true["skel_ok"], r_sh["skel_ok"])
        # shuffle_all（整行输入含 mask 跨行置换，gold 不动）
        g = torch.Generator().manual_seed(seed * SHUF_SEED_BASE + 99)
        perm = torch.randperm(blob[s]["n"], generator=g)
        blob_all = {k: (v[perm] if k in ("v_sent", "v_items", "item_mask") else v)
                    for k, v in blob[s].items()}
        r_all = forward_all(model, blob_all, device)
        r_all["gold"] = blob[s]["skel"].tolist()
        rec["shuffle_all"] = metrics(r_all)["skel"]
        rec["shuffle_all_paired"] = paired_delta(res_all[s]["skel_ok"], r_all["skel_ok"])
        out["probe"][s] = rec
        print(f"[{name}] probe {s:8s} content {rec.get('true', {}).get('acc')}→"
              f"{rec.get('shuffled', {}).get('acc')}（幸存 {len(ok_idx)}/{len(rs)}）"
              f" | shuffle_all={rec['shuffle_all']['acc']}", flush=True)

    # pair_shuffle_content：两侧同用另一对的槽文本
    sh_b, diag_b = make_content_shuffle(rows["b_pairs"], 2)
    surv_pairs = [(i, j) for i, j in prs_all if sh_b[i] is not None and sh_b[j] is not None]
    rec_p = {"build": diag_b, "n_survivor_pairs": len(surv_pairs),
             "n_total_pairs": len(prs_all)}
    if surv_pairs:
        # **交错**排列：sub[2k]=对k 的 A 侧、sub[2k+1]=对k 的 B 侧 ⇒ 与 prs_t 同构
        sub = [x for i, j in surv_pairs for x in (sh_b[i], sh_b[j])]
        blob_sh = encode_rows(model, sub, spec, device)
        r_sh = forward_all(model, blob_sh, device)
        r_sh["gold"] = [blob_sh["skel"][k] for k in range(len(sub))]
        prs_sh = [(2 * k, 2 * k + 1) for k in range(len(surv_pairs))]
        # true 侧在同一幸存子集上重算（配对比较）
        t_idx = [i for p in surv_pairs for i in p]
        bt = {k: (v[t_idx] if k != "n" else len(t_idx)) for k, v in blob["b_pairs"].items()}
        r_t = forward_all(model, bt, device)
        r_t["gold"] = bt["skel"].tolist()
        prs_t = [(2 * k, 2 * k + 1) for k in range(len(surv_pairs))]
        rec_p["true_pair"] = pair_metrics(r_t, prs_t)
        rec_p["shuffled_pair"] = pair_metrics(r_sh, prs_sh)
        out["pair"]["b1_word_content_shuffled"] = rec_p["shuffled_pair"]
        out["pair"]["b1_word_true_on_survivors"] = rec_p["true_pair"]
        print(f"[{name}] pair_shuffle_content 幸存 {len(surv_pairs)}/{len(prs_all)} "
              f"true={rec_p['true_pair']['pair_success']:.4f} "
              f"shuf={rec_p['shuffled_pair']['pair_success']:.4f}", flush=True)
    out["probe"]["pair_content"] = rec_p

    del model
    torch.cuda.empty_cache() if device.startswith("cuda") else None
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43])
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    RESULTS.mkdir(exist_ok=True)
    out = evaluate_model(a.arm, a.seed, a.randlabel, device)
    # 地板与 aux_only（模型无关，逐格存一份便于对账）
    train = load_rows(TCH_DATA / "train.jsonl")
    rows = {s: get_rows(s) for s in SPLITS}
    prs = {"b_pairs": pair_index(rows["b_pairs"])}
    out["floor"] = build_floor(train, rows, prs)
    out["aux_only"] = {
        "applicable": False,
        "reason": "本模型（struct_supervision B 口径）没有标签通道 ⇒ bag_modules 的 aux_only 不适用",
        "replacement_n_slots_rule": {s: out["floor"][s]["acc_by_rule"]["n_slots_rule"]
                                     for s in SPLITS},
        "replacement_pair_success": out["floor"]["b_pairs"].get("max_naive_pair_success"),
    }
    p = RESULTS / f"eval_{out['name']}.json"
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] → {p}", flush=True)


if __name__ == "__main__":
    main()
