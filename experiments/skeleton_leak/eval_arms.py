#!/usr/bin/env python3
"""skeleton_leak 评测：只读加载 bag_modules 四臂权重，在新集上重测（Q1/Q2/Q5）。

- 权重只读：`experiments/bag_modules/weights/{A,U,P,UP}_s{42,43}.pt`，`strict=False`，
  **断言缺键只许 `encoder.*`**；不重训、不微调、不写对方目录。
- 编码/标签/指标口径与 `bag_modules/train.py` 逐字同式；编码缓存只写本目录 `cache/`。
- **口径校验**：旧 `test`(n=2500) 重评结果必须与 `bag_modules/results/*.json` 逐格一致，
  且 v_sent 与只读缓存逐位一致，否则 fail-closed 停。
- 负向对照（PREREG Q5）：评测期标签行内 randperm（含 mask，seed*1000+99，与
  `bag_modules/analyze.py::shuffle_probe` 同口径）⇒ 期望落回多数类。

用法：uv run python experiments/skeleton_leak/eval_arms.py [--device cpu]
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

HERE = Path(__file__).resolve().parent
BM = ROOT / "experiments" / "bag_modules"
TCH_DATA = ROOT / "experiments" / "two_channel_head" / "data"
TCH_CACHE = ROOT / "experiments" / "two_channel_head" / "cache"
DATA = HERE / "data"
CACHE = HERE / "cache"
RESULTS = HERE / "results"
SS_DATA = ROOT / "experiments" / "struct_supervision" / "data"

POS_BUCKET = 4
ARMS = ("A", "U", "P", "UP")
SEEDS = (42, 43)
SPLITS = ("test", "adv2", "a_bal", "a_lit", "c_pairs", "b_pairs")
SHUFFLE_SPLITS = ("test", "a_bal")     # 负向对照（test 先 ⇒ 与对方口径可对齐）


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load(BM / "model.py", "sleak_bagmod_ro")          # 只读
BL = _load(BM / "build_labels.py", "sleak_labels_ro")  # 只读
Spec, BagModModel, v_bag_of = M.Spec, M.BagModModel, M.v_bag_of
label_row, row_key = BL.label_row, BL.row_key

ARMS_ = ARMS


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[skeleton_leak eval fail-closed] {msg}")


def load_rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def get_rows(split: str) -> list[dict]:
    if split in ("test",):
        return load_rows(TCH_DATA / f"{split}.jsonl")
    if split in ("adv1", "adv2"):
        return load_rows(SS_DATA / f"{split}.jsonl")
    return load_rows(DATA / f"{split}.jsonl")


# ---------------------------------------------------------------------------
# 编码（一次前向同时给出 v_sent / 袋项 / 逐 token 状态；缓存写本目录）
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode(model, rows: list[dict], spec, device: str, split: str) -> dict:
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    CACHE.mkdir(exist_ok=True)
    fp = M._fp(rows)
    path = CACHE / f"enc_{fp}_L{spec.max_len_sent}.pt"
    if path.exists():
        b = torch.load(path, map_location="cpu", weights_only=True)
        if b["n"] == len(rows):
            print(f"[enc] 命中 {path.name}（n={b['n']}）", flush=True)
            return b
    v_sent, v_items, mask, hs, ms, skel, assign = [], [], [], [], [], [], []
    model.encoder.eval()
    bs = 256
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ids, msk = [], []
        for r in chunk:
            e = tok.encode(r["sent"], max_length=spec.max_len_sent, padding=True)
            check(sum(e["attention_mask"]) == len(r["sent"]), f"1 字符 1 token 被破坏：{r['sent']}")
            ids.append(e["input_ids"])
            msk.append(e["attention_mask"])
        id_t = torch.tensor(ids, dtype=torch.long, device=device)
        m_t = torch.tensor(msk, dtype=torch.bool, device=device)
        h = model.encoder(id_t, m_t).cpu()
        m3 = m_t.cpu().unsqueeze(-1).to(h.dtype)
        v_sent.append((h * m3).sum(1) / m3.sum(1).clamp(min=1.0))
        hs.append(h)
        ms.append(m_t.cpu())
        bsz = len(chunk)
        items = torch.zeros(bsz, spec.max_slots, spec.hidden)
        im = torch.zeros(bsz, spec.max_slots, dtype=torch.bool)
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
            "assign": torch.tensor(assign), "h": torch.cat(hs), "hmask": torch.cat(ms),
            "n": len(rows)}
    torch.save(blob, path)
    print(f"[enc] 写入 {path.name}（n={len(rows)}, "
          f"{path.stat().st_size / 1e6:.0f} MB）", flush=True)
    return blob


def label_tensors(rows: list[dict]) -> dict:
    n = len(rows)
    m = max(r["n_slots"] for r in rows)
    tt = torch.zeros(n, m, dtype=torch.long)
    rt = torch.zeros(n, m, dtype=torch.long)
    ct = torch.zeros(n, m, dtype=torch.long)
    pb = torch.zeros(n, m, dtype=torch.long)
    mk = torch.zeros(n, m, dtype=torch.bool)
    labs = []
    for i, r in enumerate(rows):
        lab = label_row(r)
        labs.append(lab)
        check(len(lab) == r["n_slots"], "标签数与槽位不符")
        order = sorted(range(r["n_slots"]), key=lambda k: r["bag_span"][k][0])
        L = max(1, len(r["sent"]))
        for j, k in enumerate(order):
            a, _b = r["bag_span"][k]
            tt[i, j] = lab[k]["t"]
            rt[i, j] = lab[k]["r"]
            ct[i, j] = lab[k]["c"]
            pb[i, j] = min(POS_BUCKET - 1, a * POS_BUCKET // L)
            mk[i, j] = True
    return {"type_t": tt, "role_t": rt, "cls_t": ct, "pos_b": pb, "mask": mk}, labs


# ---------------------------------------------------------------------------
# 评测（口径 = bag_modules/train.py::evaluate）
# ---------------------------------------------------------------------------
def _metrics(skel_ok, slot_frac, joint_ok) -> dict:
    n = len(skel_ok)
    se = math.sqrt(0.25 / n)
    mu = sum(slot_frac) / n
    se_slot = math.sqrt(max(1e-12, sum((x - mu) ** 2 for x in slot_frac) / (n - 1))) / math.sqrt(n)
    return {"skel": {"n": n, "acc": round(sum(skel_ok) / n, 6), "se": round(se, 6)},
            "slot": {"n": n, "acc": round(mu, 6), "se": round(se_slot, 6)},
            "joint": {"n": n, "acc": round(sum(joint_ok) / n, 6), "se": round(se, 6)}}


@torch.no_grad()
def evaluate(model, blob, labs, device, bs=4096) -> tuple[dict, dict, dict]:
    skel_ok, slot_frac, joint_ok, preds = [], [], [], []
    aux_ok, head_ok = [], []
    use_lab = model.lab is not None
    model.eval()
    for i in range(0, blob["n"], bs):
        j = min(i + bs, blob["n"])
        vs = blob["v_sent"][i:j].to(device)
        vi = blob["v_items"][i:j].to(device)
        im = blob["item_mask"][i:j].to(device)
        vb = v_bag_of(vi, im)
        sk_y = blob["skel"][i:j].to(device)
        as_y = blob["assign"][i:j].to(device)
        lab_in = None
        if use_lab:
            lab_in = {k: labs[k][i:j].to(device) for k in
                      ("type_t", "role_t", "cls_t", "pos_b")}
            lab_in["mask"] = labs["mask"][i:j].to(device)
        h_tok = tmask = None
        if model.pool is not None:
            h_tok = blob["h"][i:j].to(device)
            tmask = blob["hmask"][i:j].to(device)
        sk_logits, a_logits, h = model.forward(vs, vb, vi, im, lab_in, h_tok, tmask)
        p = sk_logits.argmax(-1).cpu().tolist()
        pa = a_logits.argmax(-1).cpu()
        gold = as_y.tolist()
        if use_lab:
            aux_ok += (model.aux_only(lab_in).argmax(-1).cpu() == sk_y.cpu()).tolist()
            head_ok += (model.head_only(h).argmax(-1).cpu() == sk_y.cpu()).tolist()
        for r in range(j - i):
            k = int(im[r].sum())
            right = sum(1 for s in range(k) if int(pa[r, s]) == gold[r][s])
            slot_frac.append(right / k)
            ok = int(p[r] == int(sk_y[r]))
            skel_ok.append(ok)
            joint_ok.append(int(ok and right == k))
            preds.append(int(p[r]))
    m = _metrics(skel_ok, slot_frac, joint_ok)
    diag = {"aux_only": round(sum(aux_ok) / len(aux_ok), 6) if aux_ok else None,
            "head_only": round(sum(head_ok) / len(head_ok), 6) if head_ok else None}
    return m, {"skel": skel_ok, "slot_frac": slot_frac, "joint": joint_ok,
               "pred": preds}, diag


def paired(a: list, b: list) -> dict:
    d = [x - y for x, y in zip(b, a)]
    n = len(d)
    mu = sum(d) / n
    var = sum((x - mu) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    s = math.sqrt(var / n)
    return {"delta": round(mu, 6), "se": round(s, 6),
            "t": round(mu / s, 3) if s > 0 else None, "n": n}


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(0)
    spec = Spec()
    print(f"[device] {device}", flush=True)

    rows_all = {s: get_rows(s) for s in SPLITS}
    leak = json.loads((RESULTS / "leak_stats.json").read_text(encoding="utf-8"))

    # ---- 编码 ----
    ref = BagModModel("A", 42, spec).to(device)
    blob, labs, lab_rows = {}, {}, {}
    for s in SPLITS:
        blob[s] = encode(ref, rows_all[s], spec, device, s)
        labs[s], lab_rows[s] = label_tensors(rows_all[s])
    # 口径校验 1：v_sent 与只读缓存逐位一致
    fp_t = M._fp(rows_all["test"])
    cands = list(TCH_CACHE.glob(f"gen_{fp_t}_L{spec.max_len_sent}_M{spec.max_slots}.pt"))
    check(cands, "缺 two_channel_head 只读缓存（口径校验用）")
    ref_blob = torch.load(cands[0], map_location="cpu", weights_only=True)
    d = (ref_blob["v_sent"] - blob["test"]["v_sent"]).abs().max().item()
    print(f"[check] v_sent vs 只读缓存 max|Δ| = {d:.2e}", flush=True)
    check(d < 1e-5, f"v_sent 与 two_channel_head 缓存不一致 {d}")
    check(torch.equal(ref_blob["skel"], blob["test"]["skel"]), "skel 标签不一致")

    out: dict = {"device": device, "n": {s: len(rows_all[s]) for s in SPLITS},
                 "rows": {}, "paired_vs_A": {}, "q1": {}, "neg_control": {}}

    # ---- 逐臂 × seed ----
    for arm in ARMS:
        for seed in SEEDS:
            m = BagModModel(arm, seed, spec).to(device)
            sd = torch.load(BM / "weights" / f"{arm}_s{seed}.pt",
                            map_location="cpu", weights_only=True)
            miss = [k for k in m.state_dict() if k not in sd]
            check(all(k.startswith("encoder.") for k in miss),
                  f"{arm}_s{seed} 缺键异常：{miss[:5]}")
            m.load_state_dict(sd, strict=False)
            m.eval()
            key = f"{arm}_s{seed}"
            out["rows"][key] = {}
            for s in SPLITS:
                met, cor, diag = evaluate(m, blob[s], labs[s], device)
                base = leak[s]
                met["skel"]["majority"] = base["majority_own"]
                met["skel"]["max_naive_complete"] = base["max_naive_complete"]
                met["skel"]["over_max_naive"] = round(
                    met["skel"]["acc"] - base["max_naive_complete"], 4)
                met["slot"]["blind"] = round(
                    sum(1.0 / len(r["assign"]) for r in rows_all[s]) / len(rows_all[s]), 4)
                met["diag"] = diag
                out["rows"][key][s] = {"metrics": met, "correct": cor}
                print(f"[eval] {key} {s:8s} 骨架={met['skel']['acc']:.4f}"
                      f"±{met['skel']['se']:.4f} 槽位={met['slot']['acc']:.4f}"
                      f" 联合={met['joint']['acc']:.4f} "
                      f"−max_naive={met['skel']['over_max_naive']:+.4f} "
                      f"aux={diag['aux_only']} head={diag['head_only']}", flush=True)
            del m
            torch.cuda.empty_cache() if device.startswith("cuda") else None

    # ---- 口径校验 2：与 bag_modules/results 逐格一致 ----
    ref_cmp = {}
    for arm in ARMS:
        for seed in SEEDS:
            p = BM / "results" / f"{arm}_s{seed}.json"
            if not p.exists():
                continue
            refj = json.loads(p.read_text(encoding="utf-8"))
            o = out["rows"][f"{arm}_s{seed}"]["test"]["metrics"]
            r = refj["meta"]["eval"]["test"]
            same = (abs(o["skel"]["acc"] - r["skel"]["acc"]) < 1e-6
                    and abs(o["slot"]["acc"] - r["slot"]["acc"]) < 1e-6
                    and abs(o["joint"]["acc"] - r["joint"]["acc"]) < 1e-6)
            ref_cmp[f"{arm}_s{seed}"] = bool(same)
            check(same, f"{arm}_s{seed} test 复评与 bag_modules 结果不一致："
                        f"{o['skel']['acc']} vs {r['skel']['acc']}")
    out["regress_vs_bag_modules"] = ref_cmp
    print(f"[check] test 复评 == bag_modules results：{ref_cmp}", flush=True)

    # ---- 配对 Δ = 臂 − A ----
    for arm in ARMS:
        if arm == "A":
            continue
        for seed in SEEDS:
            k0, k1 = f"A_s{seed}", f"{arm}_s{seed}"
            for s in SPLITS:
                out["paired_vs_A"].setdefault(f"{arm}_s{seed}", {})[s] = {
                    "skel": paired(out["rows"][k0][s]["correct"]["skel"],
                                   out["rows"][k1][s]["correct"]["skel"]),
                    "slot": paired(out["rows"][k0][s]["correct"]["slot_frac"],
                                   out["rows"][k1][s]["correct"]["slot_frac"]),
                }

    # ---- Q1：最小对 ----
    for s in ("b_pairs", "c_pairs"):
        rows = rows_all[s]
        k_present = len(set(r["skel_id"] for r in rows))
        out["q1"][s] = {"n": len(rows), "k_present": k_present,
                        "blind_uniform_1_40": 0.025,
                        "blind_present_1_k": round(1 / k_present, 4)}
        if s == "b_pairs":
            pairs_idx = []
            for i, r in enumerate(rows):
                if r.get("kind", "").endswith("_src"):
                    check(i + 1 < len(rows) and rows[i + 1].get("kind", "").endswith("_variant"),
                          "B 型源/变体未相邻成对")
                    check(rows[i]["skel_id"] != rows[i + 1]["skel_id"], "B 对骨架相同")
                    check(sorted(rows[i]["bag"]) == sorted(rows[i + 1]["bag"]), "B 对袋不同")
                    pairs_idx.append((i, i + 1))
            check(len(pairs_idx) * 2 == len(rows), "B 对数不匹配")
            out["q1"][s]["n_pairs"] = len(pairs_idx)
            out["q1"][s]["pair_chance"] = 0.5
        else:
            pairs_idx = None
            by = defaultdict(list)
            for i, r in enumerate(rows):
                by[r["pair"]].append(i)
            out["q1"][s]["n_pair_groups"] = len(by)
        for arm in ARMS:
            for seed in SEEDS:
                cor = out["rows"][f"{arm}_s{seed}"][s]["correct"]
                n = len(cor["skel"])
                acc = sum(cor["skel"]) / n
                se = math.sqrt(0.25 / n)
                rec = {"acc": round(acc, 4), "se": round(se, 4),
                       "over_blind_1_40": round(acc - 0.025, 4),
                       "t_vs_blind": round((acc - 0.025) / se, 2)}
                if pairs_idx is not None:
                    both = sum(1 for i, j in pairs_idx
                               if cor["skel"][i] and cor["skel"][j])
                    diff = sum(1 for i, j in pairs_idx if cor["pred"][i] != cor["pred"][j])
                    rec["pair_success"] = round(both / len(pairs_idx), 4)
                    rec["pair_pred_differs"] = round(diff / len(pairs_idx), 4)
                    rec["over_pair_chance_0.5"] = round(acc - 0.5, 4)
                    rec["t_vs_0.5"] = round((acc - 0.5) / se, 2)
                else:
                    rec["over_1k"] = round(acc - 1 / k_present, 4)
                    rec["t_vs_1k"] = round((acc - 1 / k_present) / se, 2)
                out["q1"][s][f"{arm}_s{seed}"] = rec

    # ---- Q5 负向对照：评测期标签打乱（含 mask，seed*1000+99，同 shuffle_probe 口径）----
    for seed in SEEDS:
        for s in SHUFFLE_SPLITS:
            g = torch.Generator().manual_seed(seed * 1000 + 99)
            perm = torch.randperm(len(rows_all[s]), generator=g)
            labs_sh = {k: (v[perm] if v.shape[0] == len(rows_all[s]) else v)
                       for k, v in labs[s].items()}
            for arm in ("U", "UP"):
                m = BagModModel(arm, seed, spec).to(device)
                sd = torch.load(BM / "weights" / f"{arm}_s{seed}.pt",
                                map_location="cpu", weights_only=True)
                m.load_state_dict(sd, strict=False)
                m.eval()
                t_met, _, _ = evaluate(m, blob[s], labs[s], device)
                s_met, _s_cor, s_diag = evaluate(m, blob[s], labs_sh, device)
                base = leak[s]
                rec = {"true": t_met["skel"]["acc"], "shuffled": s_met["skel"]["acc"],
                       "drop": round(t_met["skel"]["acc"] - s_met["skel"]["acc"], 4),
                       "shuffled_slot": s_met["slot"]["acc"],
                       "head_only": s_diag["head_only"], "aux_only_shuf": s_diag["aux_only"],
                       "majority": base["majority_own"],
                       "threshold_majority_2se": base["Q0_threshold"],
                       "pass_fall_back": bool(s_met["skel"]["acc"]
                                              <= base["Q0_threshold"])}
                out["neg_control"].setdefault(f"{arm}_s{seed}", {})[s] = rec
                print(f"[neg] {arm}_s{seed} {s}: true={rec['true']:.4f} "
                      f"shuf={rec['shuffled']:.4f} head_only={rec['head_only']} "
                      f"多数类={rec['majority']} 阈={rec['threshold_majority_2se']} "
                      f"落回={'是' if rec['pass_fall_back'] else '否'}", flush=True)
                del m
                torch.cuda.empty_cache() if device.startswith("cuda") else None

    (RESULTS / "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(f"[done] → {RESULTS / 'eval.json'}", flush=True)


if __name__ == "__main__":
    main()
