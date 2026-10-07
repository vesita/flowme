#!/usr/bin/env python3
"""syllogism_card 全量评测：max_naive 电池 + L0–L6 + 注入反例 + 老卡 Δ。

用法：uv run python experiments/syllogism_card/eval.py [--tag T_s42] [--all]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from gen_data import (  # noqa: E402
    REJECT, rule_label)

SE = math.sqrt(0.25 / 1200)
TWO_SE = 2 * SE

TYPE_LABEL = {"chain": "有效·直链", "variant": "有效·变体",
              "fallacy": "谬误", "distractor": "干扰"}
COUNTED = ["R-chain", "R-chain2", "R-keyword", "majority", "length",
           "extreme", "n_slots"]


# ---------------- naive 电池 ----------------
def fit_rules(exit_: str, items: list[dict]) -> dict:
    tr = [it for it in items if it["split"] == "train" and it["exit"] == exit_]
    lab = Counter(it["label"] for it in tr)
    maj = lab.most_common(1)[0][0]
    nslots = Counter((len(it["premises"][0]), len(it["premises"][1]), it["label"]) for it in tr)
    tab_ns = {}
    for (a, b, y), c in nslots.items():
        tab_ns.setdefault((a, b), Counter())[y] += 1
    tab_ns = {k: v.most_common(1)[0][0] for k, v in tab_ns.items()}
    lenb = defaultdict(Counter)
    for it in tr:
        lenb[(len(it["premises"][0]) + len(it["premises"][1])) // 4][it["label"]] += 1
    tab_len = {k: v.most_common(1)[0][0] for k, v in lenb.items()}

    def f_length(p):
        k = (len(p[0]) + len(p[1])) // 4
        return tab_len.get(k, maj)

    def f_nslots(p):
        return tab_ns.get((len(p[0]), len(p[1])), maj)

    return {
        "R-chain": lambda p: rule_label("R-chain", p),
        "R-chain2": lambda p: rule_label("R-chain2", p),
        "R-keyword": lambda p: rule_label("R-keyword", p),
        "majority": lambda p: maj,
        "length": f_length,
        "extreme": lambda p: maj,
        "n_slots": f_nslots,
        "R-lexdir": lambda p: rule_label("R-lexdir", p),
    }


def eval_rules(rules: dict, test: list[dict]) -> dict:
    out = {}
    for name, f in rules.items():
        ok = 0
        rej = Counter()
        per_type = defaultdict(lambda: [0, 0])
        for it in test:
            y = f(it["premises"])
            good = int(y == it["gold_label"])
            ok += good
            rej[it["type"]] += int(y == REJECT)
            per_type[it["type"]][0] += good
            per_type[it["type"]][1] += 1
        n = len(test)
        out[name] = {
            "acc": round(ok / n, 4), "n": n,
            "reject_rate": {t: round(v / max(1, sum(1 for x in test if x["type"] == t)), 4)
                            for t, v in rej.items()},
            "per_type_acc": {t: round(a / b, 4) for t, (a, b) in per_type.items()},
        }
    counted = {k: v["acc"] for k, v in out.items() if k in COUNTED}
    best = max(counted, key=counted.get)
    out["_max_naive"] = {"rule": best, "acc": counted[best]}
    out["_disclose_lexdir"] = out["R-lexdir"]["acc"]
    return out


# ---------------- L3 引用映射检查 ----------------
def check_refs(item: dict) -> dict:
    """返回 {"struct": 违例数, "sem": 违例数, "reasons": [...]}。

    struct = 结构门（PREREG L3 的「恰好铺满 + span 可回溯」，必须为 0）
    sem    = 归因正确性（ref 的实体 id 与结论参数一致）
    """
    st: list[str] = []
    sem: list[str] = []
    refs = item["refs"]
    lab = item["pred_label"]
    if lab == REJECT:
        if refs is not None:
            st.append("拒答却给出引用映射")
        return {"struct": len(st), "sem": 0, "reasons": st}
    if refs is None:
        return {"struct": 1, "sem": 0, "reasons": ["有结论却无引用映射"]}
    if len(refs) != 2:
        return {"struct": 1, "sem": 0, "reasons": [f"引用条数 {len(refs)} ≠ 2"]}
    if refs[0] is None or refs[1] is None:
        return {"struct": 1, "sem": 0, "reasons": ["引用下标越界（None）"]}
    from gen_data import dec_label
    d = dec_label(lab)
    if d is None:
        return {"struct": 1, "sem": 0, "reasons": ["标签解码失败"]}
    _, _, a, b = d
    prem = item["premises"]
    for k, (r, want) in enumerate(zip(refs, (a, b))):
        s, e = r["start"], r["end"]
        if not (0 <= s < e <= len(prem[r["p"]])):
            st.append(f"ref{k} span {r['p'], s, e} 越界")
        elif prem[r["p"]][s:e] != r["text"]:
            st.append(f"ref{k} 不是输入逐字子串")
        if r["eid"] != want:
            sem.append(f"ref{k} 实体 id {r['eid']} ≠ 结论参数 {want}")
    if refs[0]["ref"] == refs[1]["ref"]:
        sem.append("两条引用指向同一 mention（未分别对应两个结论参数）")
    return {"struct": len(st), "sem": len(sem), "reasons": st + sem}


def tamper_ref(item: dict) -> dict:
    """注入反例 C1：篡改第一条引用（改成越界下标）。"""
    it = dict(item)
    refs = [dict(x) for x in (item["refs"] or [])]
    if not refs:
        return it
    bad = dict(refs[0])
    bad["start"], bad["end"] = 0, 0
    bad["text"] = "ZZ"
    refs[0] = bad
    it["refs"] = refs
    return it


# ---------------- 老卡 L5 ----------------
def old_cards_round(device: str, seed: int) -> dict:
    sys.path.insert(0, str(ROOT / "experiments" / "select_semantic_joint"))
    from common import (  # noqa
        BAND, old_card_baseline, core_drift, load_base_encoder, BASE_CKPT)
    enc, _ = load_base_encoder(str(BASE_CKPT), device)
    enc.eval()
    base = old_card_baseline(seed, device)
    return {"enc": enc, "base": base, "BAND": BAND, "core_drift": core_drift}


# ---------------- 主流程 ----------------
def tag_eval(tag: str, items: list[dict]) -> dict:
    preds = json.loads((HERE / "results" / f"pred_{tag}.json").read_text(encoding="utf-8"))
    meta = json.loads((HERE / "results" / f"meta_{tag}.json").read_text(encoding="utf-8"))
    by_exit = defaultdict(list)
    for p in preds:
        by_exit[p["split"] + "/" + p["exit"]].append(p)

    res = {"meta": meta, "sets": {}}
    for key in ("test/tmpl", "test/para", "train/tmpl", "train/para"):
        test = by_exit.get(key, [])
        if not test:
            continue
        exit_ = key.split("/")[1]
        rules = fit_rules(exit_, items)
        rb = eval_rules(rules, test)
        n = len(test)
        ok = sum(1 for p in test if p["pred_label"] == p["gold_label"])
        acc = ok / n
        mn = rb["_max_naive"]
        rec = {
            "n": n, "acc": round(acc, 4), "se": round(SE, 4),
            "majority_gold": round(Counter(p["gold_label"] for p in test).most_common(1)[0][0] / n, 4),
            "max_naive": mn, "card_minus_naive": round(acc - mn["acc"], 4),
            "card_minus_naive_2se": round(acc - mn["acc"] - TWO_SE, 4),
            "card_minus_lexdir": round(acc - rb["_disclose_lexdir"], 4),
            "parse_ok": round(sum(1 for p in test if p["parse_ok"]) / n, 4),
            "tag_acc": round(sum(p["tag_acc"] for p in test) / n, 4),
            "reject_pred": round(sum(1 for p in test if p["pred_label"] == REJECT) / n, 4),
            "reject_gold": round(sum(1 for p in test if p["gold_label"] == REJECT) / n, 4),
            "rules": {k: v for k, v in rb.items() if not k.startswith("_")},
        }
        # 逐型
        per = {}
        for ty in TYPE_LABEL:
            sub = [p for p in test if p["type"] == ty]
            if not sub:
                continue
            nn = len(sub)
            a = sum(1 for p in sub if p["pred_label"] == p["gold_label"]) / nn
            rp = sum(1 for p in sub if p["pred_label"] == REJECT) / nn
            rg = sum(1 for p in sub if p["gold_label"] == REJECT) / nn
            rr = max(v["reject_rate"].get(ty, 0.0) for k, v in rb.items() if k in COUNTED)
            ra = max(v["per_type_acc"].get(ty, 0.0) for k, v in rb.items() if k in COUNTED)
            per[ty] = {"label": TYPE_LABEL[ty], "n": nn,
                       "acc": round(a, 4), "se": round(math.sqrt(0.25 / nn), 4),
                       "reject_pred": round(rp, 4), "reject_gold": round(rg, 4),
                       "max_naive_reject": rr, "max_naive_acc": ra,
                       "card_minus_naive": round(a - ra, 4),
                       "false_reject": sum(1 for p in sub if p["gold_label"] != REJECT
                                           and p["pred_label"] == REJECT),
                       "false_conclude": sum(1 for p in sub if p["gold_label"] == REJECT
                                             and p["pred_label"] != REJECT)}
        rec["per_type"] = per
        # L3
        viol = semv = 0
        reasons = Counter()
        attr_ok = attr_or = n_with_refs = 0
        for p in test:
            cr = check_refs(p)
            viol += cr["struct"]
            semv += cr["sem"]
            reasons.update(cr["reasons"])
            if p["gold_label"] != REJECT and p["pred_label"] != REJECT:
                from gen_data import dec_label
                d = dec_label(p["pred_label"])
                if d:
                    n_with_refs += 1
                    attr_or += int(p["pred_attr"] == p["gold_attr"])
                    if p["refs"] and p["refs"][0] and p["refs"][1]:
                        _, _, a1, b1 = d
                        attr_ok += int(p["refs"][0]["eid"] == a1 and p["refs"][1]["eid"] == b1)
        rec["l3"] = {"violations": viol, "sem_violations": semv,
                     "reasons": dict(reasons.most_common(6)),
                     "n_gold_concluded_pred_concluded": n_with_refs,
                     "attr_eid_match": round(attr_ok / max(1, n_with_refs), 4),
                     "attr_exact_deployed": attr_or,
                     "n_concluded": sum(1 for p in test if p["pred_label"] != REJECT)}
        res["sets"][key] = rec
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--device", default="cuda" if __import__("torch").cuda.is_available() else "cpu")
    ap.add_argument("--no-oldcards", action="store_true")
    a = ap.parse_args()

    items = json.loads((HERE / "data.json").read_text(encoding="utf-8"))["items"]
    tags = sorted(p.stem[len("pred_"):] for p in (HERE / "results").glob("pred_*.json"))
    if a.tag:
        tags = [a.tag]
    out = {"tags": {}, "surface": json.loads((HERE / "data.json").read_text(encoding="utf-8"))["surface"],
           "stats": json.loads((HERE / "data.json").read_text(encoding="utf-8"))["stats"]}
    for t in tags:
        print(f"[eval] {t}", flush=True)
        out["tags"][t] = tag_eval(t, items)

    # ---- 注入反例 ----
    if tags:
        some = json.loads((HERE / "results" / f"pred_{tags[0]}.json").read_text(encoding="utf-8"))
        concl = [p for p in some if p["pred_label"] != REJECT and p["refs"]]
        c1 = check_refs(tamper_ref(concl[0])) if concl else {
            "struct": 0, "sem": 0, "reasons": ["无可篡改样本"]}
        fall = [p for p in some if p["gold_label"] == REJECT]
        c2 = 0
        for p in fall:
            q = dict(p)
            q["pred_label"] = 1          # 强行给结论
            q["pred_attr"] = 0
            cr = check_refs(q)
            c2 += int((cr["struct"] + cr["sem"]) > 0)
        out["inject"] = {"C1_tamper_ref": {"violations": c1["struct"] + c1["sem"],
                                           "reasons": c1["reasons"]},
                         "C2_skip_fallacy_reject": {"forced_conclude": len(fall),
                                                    "caught": c2}}
        # C3：袋 span 不是输入逐字子串 ⇒ render.py 必须拒
        sys.path.insert(0, str(ROOT / "src"))
        from dtseek.tasks.render import (  # noqa
            BagItem, Instruction, check_structure)
        text = "所有雪原是青田"
        bad = BagItem(1, "雪原", (0, 9), "名", "施事", "c:1", True)
        c3 = check_structure(Instruction("S01", (1, 2, 3)), [bad], text)
        out["inject"]["C3_bag_span_not_substring"] = {"problems": c3, "caught": bool(c3)}

    # ---- L5 老卡 ----
    if not a.no_oldcards:
        try:
            import time
            t0 = time.time()
            sys.path.insert(0, str(ROOT / "experiments" / "select_semantic_joint"))
            from common import (old_card_baseline, core_drift, load_base_encoder,
                                BASE_CKPT, BAND)
            oc = {}
            for seed in (42, 43):
                enc, _ = load_base_encoder(str(BASE_CKPT), a.device)
                enc.eval()
                before = old_card_baseline(seed, a.device)
                # 走一遍本项目训练时对核的唯一操作（no_grad 编码）
                from nano_char_tokenizer import NanoCharTokenizer
                tok = NanoCharTokenizer()
                with __import__("torch").no_grad():
                    e = tok.encode("所有雪原是青田", max_length=24, padding=True)
                    enc(__import__("torch").tensor([e["input_ids"]], device=a.device),
                        __import__("torch").tensor([e["attention_mask"]], dtype=bool,
                                                   device=a.device))
                after = old_card_baseline(seed, a.device)
                drift = core_drift(enc)
                d = {}
                for n in ("pronoun", "sentiment", "relation", "person"):
                    d[n] = {"exact": after[n]["exact"],
                            "delta": round(after[n]["exact"] - before[n]["exact"], 6),
                            "band": BAND[n],
                            "sha_equal": after[n]["sha256"] == before[n]["sha256"]}
                oc[str(seed)] = {"cards": d, "core_drift": drift,
                                 "combined_sha_equal":
                                     after["combined_sha256"] == before["combined_sha256"]}
            out["old_cards"] = {"per_seed": oc, "wall_sec": round(time.time() - t0, 1)}
        except Exception as ex:  # noqa
            out["old_cards"] = {"error": repr(ex)}

    (HERE / "results" / "eval.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                                encoding="utf-8")
    print("[eval] -> results/eval.json", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
