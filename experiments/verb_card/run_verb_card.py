#!/usr/bin/env python3
"""P25 判据驱动：跑 W0–W6，写 `results.json`（判据原文见 `PREREG.md`）。

    cd DTSeek && uv run python experiments/verb_card/run_verb_card.py

一次性跑完，全部数字都是**本次实测**（不引用旧结论）。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline as P  # noqa: E402
import verb_card as VC  # noqa: E402
from fixtures import CF_VERB_SKELETONS, FIXTURES, SKELETON_FIXTURE, W2_SKELETONS  # noqa: E402
from fixtures import gold_spans  # noqa: E402
from dtseek.tasks import render as R  # noqa: E402
from dtseek.tasks.builtin.cloze_fill.dataset import POS_LEXICON  # noqa: E402
from dtseek.tasks.builtin.negation.dataset import NEG_MARKERS  # noqa: E402
from dtseek.tasks.builtin.sentiment.dataset import LEXICON_BY_CAT  # noqa: E402

ROOT = HERE.parents[1]
HAN = re.compile(r"[一-鿿]")

OFFICIAL_16: tuple[str, ...] = tuple(f"S{i:02d}" for i in range(1, 14)) + ("R01", "R02", "R03")
MODES: tuple[str, ...] = ("none", "T1", "T2", "T3", "verb")   # 前 4 个 = 基线/地板，最后 = 动词卡
FLOOR_RULES: tuple[str, ...] = ("T1", "T2", "T3")


# ---- W0 平凡地板（PREREG §3，跑前写死） -----------------------------------------

def _t1_lexicon() -> tuple[str, ...]:
    """T1 词表 = cloze 四类全词表 ∪ 官方 CONTENT_LEXICON 全类 ∪ 代词 ∪ 情绪词 ∪ 否定词。"""
    from proposers import _PRONOUNS

    words: set[str] = set()
    for table in (POS_LEXICON, R.CONTENT_LEXICON, LEXICON_BY_CAT):
        for ws in table.values():
            words.update(ws)
    words.update(_PRONOUNS)
    words.update(NEG_MARKERS)
    words.discard("")
    return tuple(sorted(words, key=lambda w: (-len(w), w)))


T1_LEX: tuple[str, ...] | None = None


def floor_spans(rule: str, text: str) -> list[tuple[int, int]]:
    """地板规则 → 至多 1 个「动词」span（确定性）。"""
    if rule == "T1":                                   # 输入里第一个命中的实义词
        for p in range(len(text)):
            for w in T1_LEX:
                if text.startswith(w, p):
                    return [(p, p + len(w))]
        return []
    han = [i for i, ch in enumerate(text) if HAN.match(ch)]
    if rule == "T2":                                   # 取末字
        return [(han[-1], han[-1] + 1)] if han else []
    if rule == "T3":                                   # 取首字
        return [(han[0], han[0] + 1)] if han else []
    raise ValueError(rule)


def floor_cands(rule: str, text: str) -> list[dict]:
    return [{"card": "verb", "s0": s, "e0": e, "surface": text[s:e], "class_id": 1,
             "class_name": "动词", "theme": None, "type": "动",
             "cid": f"floor:{rule}:{s}:{e}"} for s, e in floor_spans(rule, text)]


def pred_spans(mode: str, text: str) -> set[tuple[int, int]]:
    """某模式在 `text` 上**产出的动槽 span**（W0 的 gold 比对口径）。"""
    if mode == "verb":
        return {(c["s0"], c["e0"]) for c in VC.extract_verb(text)}
    if mode in FLOOR_RULES:
        return set(floor_spans(mode, text))
    return set()


def prf(mode: str) -> dict:
    """微平均 P/R/F1：gold = 夹具人工判定的动词 span（PREREG §4）。"""
    tp = fp = fn = 0
    per: list[dict] = []
    for fix in FIXTURES:
        g, p = gold_spans(fix), pred_spans(mode, fix.text)
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        per.append({"fid": fix.fid, "gold": sorted(g), "pred": sorted(p)})
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"tp": tp, "fp": fp, "fn": fn, "P": round(prec, 4),
            "R": round(rec, 4), "F1": round(f1, 4), "per_fixture": per}


# ---- 跑骨架：一条输入 × 一组骨架 × 一个模式 -------------------------------------

def _assign_text(out: dict, bag: list[R.BagItem],
                 table: dict[str, R.Skeleton]) -> list[str]:
    sk = table[out["instruction"]["skeleton_id"]]
    items = {b.ref: b for b in bag}
    return [f"[{i}]={items[r].text}({items[r].pos}"
            + (f"/{items[r].theta}" if items[r].theta else "") + ")"
            for i, r in enumerate(out["instruction"]["assignment"], 1)]


def try_all(text: str, roles: dict | None, sids: tuple[str, ...] | list[str],
            table: dict[str, R.Skeleton], mode: str) -> list[dict]:
    floor = floor_cands(mode, text) if mode in FLOOR_RULES else None
    bag, _valid = P.make_bag(text, roles, with_verb=(mode == "verb"), floor_verb=floor)
    rows: list[dict] = []
    for sid in sids:
        out = P.fill(text, bag, table[sid], table)
        row: dict = {"sid": sid, "kind": out.get("kind")}
        if out.get("kind") == "text":
            g = P.gates(out, bag, text, table)
            a = P.trace_audit(out, bag, text, table)
            row.update(text=out["text"], assign=_assign_text(out, bag, table), gates=g,
                       audit=len(a), verbs=[list(s) for s in
                                            sorted(P.verb_spans(out, bag, table))])
        else:
            row["reason"] = out.get("reason") or out.get("missing_pos")
        rows.append(row)
    return rows


def _ok(row: dict) -> bool:
    return row["kind"] == "text" and sum(row["gates"].values()) == 0


def _agg(rows: list[dict]) -> dict:
    ok = [r for r in rows if _ok(r)]
    return {
        "n_rows": len(rows), "n_text": sum(1 for r in rows if r["kind"] == "text"),
        "n_unlocked": len(ok),
        "gates_sum": {k: sum(r["gates"][k] for r in rows if r["kind"] == "text")
                      for k in ("rule_a", "slot_schema", "item", "contract", "structure", "deref")},
        "audit_violations": sum(r["audit"] for r in rows if r["kind"] == "text"),
        "verb_gold_spans_used": sorted({tuple(v) for r in ok for v in r["verbs"]}),
    }


# ---- 主流程 ---------------------------------------------------------------------

def main() -> int:
    global T1_LEX
    t0 = time.time()
    T1_LEX = _t1_lexicon()

    # —— 清单与前提对账（实测） ——
    table = dict(R.SKELETONS)
    table.update(P.cf_skeletons())
    p1 = tuple(sid for sid in R.SKELETONS
               if any(sl.pos == "动" for sl in R.SKELETONS[sid].slots))
    p2 = tuple(CF_VERB_SKELETONS)
    p3 = tuple(W2_SKELETONS)
    from skeletons import BY_ID as CF                     # card_flow 25 条（只读）
    kinds = lambda pre: sum(1 for s in CF.values()             # noqa: E731
                            if any(sl.name[0] == pre for sl in s.slots))
    counts = {"动": kinds("v"), "否": kinds("g"), "数": kinds("d"),
              "计数和": kinds("v") + kinds("g") + kinds("d"),
              "去重并集": sum(1 for s in CF.values()
                              if any(sl.name[0] in "vgd" for sl in s.slots)),
              "card_flow 总数": len(CF)}

    # —— W0 平凡地板 ——
    w0 = {"t1_lexicon_size": len(T1_LEX), "verb_lexicon_size": len(VC.VERB_LEXICON),
          "gold_note": "gold = fixtures.py 人工判定的动词 span（20 条夹具）",
          "modes": {}}
    for mode in MODES:
        w0["modes"][mode] = {"prf": prf(mode),
                             "unlock_on_fixtures": {}}

    # —— W1 逐骨架解锁表（夹具：两层题元） ——
    w1: dict = {}
    for mode in MODES:
        tier_a, tier_b = [], []
        for fix in FIXTURES:
            sids = [sid for sid in (p1 + p2) if SKELETON_FIXTURE.get(sid) == fix.fid]
            if not sids:
                continue
            tier_a += try_all(fix.text, None, sids, table, mode)
            tier_b += try_all(fix.text, fix.role_map, sids, table, mode)
        for tier_name, rows in (("tierA_无题元", tier_a), ("tierB_夹具题元", tier_b)):
            w0["modes"][mode]["unlock_on_fixtures"][tier_name] = _agg(rows)
        w1[mode] = {
            "tierA": {r["sid"]: r for r in tier_a},
            "tierB": {r["sid"]: r for r in tier_b},
        }

    # —— W1 语料 120 句（card_flow 同规则抽样） ——
    from run_card_flow import load_samples  # noqa: E402

    samples = load_samples()
    corpus: dict = {"n": len(samples), "skeletons": list(p1), "modes": {}}
    corpus_gates = {k: 0 for k in ("rule_a", "slot_schema", "item", "contract",
                                   "structure", "deref")}
    corpus_audit = 0
    for mode in MODES:
        t1_ = time.time()
        unlocked_pairs = 0
        sentences = 0
        verb_hits: set[tuple[int, int]] = set()
        for s in samples:
            rows = try_all(s, None, p1, table, mode)
            for r in rows:
                if r["kind"] == "text":
                    corpus_audit += r["audit"]
                    for k in corpus_gates:
                        corpus_gates[k] += r["gates"][k]
            k = sum(1 for r in rows if _ok(r))
            unlocked_pairs += k
            sentences += 1 if k else 0
            for r in rows:
                if _ok(r):
                    verb_hits.update(tuple(v) for v in r["verbs"])
        corpus["modes"][mode] = {
            "sentences_with_ge1_unlocked": sentences,
            "unlocked_pairs": unlocked_pairs,
            "possible_pairs": len(samples) * len(p1),
            "verb_spans_used": len(verb_hits),
            "sec": round(time.time() - t1_, 2),
        }

    # —— W0 解锁数（P1∪P2，夹具 tierB） ——
    for mode in MODES:
        w0["modes"][mode]["unlock_total_P1P2_tierB"] = \
            w0["modes"][mode]["unlock_on_fixtures"]["tierB_夹具题元"]["n_unlocked"]
        w0["modes"][mode]["unlock_total_P1P2_tierA"] = \
            w0["modes"][mode]["unlock_on_fixtures"]["tierA_无题元"]["n_unlocked"]

    # —— W2 / W5：等价性实测（由 equiv_official.py --compare 产出） ——
    cmp_ = json.loads((HERE / "equiv_compare.json").read_text(encoding="utf-8"))

    # —— W5：全库 pytest（本次实跑） ——
    proc = subprocess.run([sys.executable, "-m", "pytest", "-q"],
                          cwd=ROOT, capture_output=True, text=True)
    tail = [ln for ln in proc.stdout.strip().splitlines() if ln.strip()][-1]
    m = re.search(r"(\d+) failed", tail)
    m2 = re.search(r"(\d+) passed", tail)
    pytest_res = {"returncode": proc.returncode, "tail": tail,
                  "failed": int(m.group(1)) if m else 0,
                  "passed": int(m2.group(1)) if m2 else 0,
                  "baseline": 258}

    # —— W3/W4 全局合计（夹具两层 + 语料 120 句） ——
    all_rows = [r for mode in MODES for tier in ("tierA", "tierB")
                for r in w1[mode][tier].values()]
    w3 = {k: v + corpus_gates[k]
          for k, v in _agg(all_rows)["gates_sum"].items()}
    w3["all_zero"] = all(v == 0 for v in w3.values())
    w4 = {"violations": _agg(all_rows)["audit_violations"] + corpus_audit,
          "checked_rows": _agg(all_rows)["n_text"] + corpus["n"] * len(MODES) * len(p1)}
    w4["all_zero"] = w4["violations"] == 0

    # —— W6：老卡 Δ（实测 git 状态） ——
    def git(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True).stdout

    status = [ln for ln in git("status", "--porcelain").splitlines() if ln.strip()]
    numstat = [ln for ln in git("diff", "--numstat", "src/").splitlines() if ln.strip()]
    changed_src = [ln.split("\t")[-1] for ln in numstat]
    w6 = {
        "git_status_porcelain": status,
        "src_diff_numstat": numstat,
        "changed_src_files": changed_src,
        "only_render_py": changed_src == ["src/dtseek/tasks/render.py"],
        "tracked_modified": [ln for ln in status if not ln.startswith("??")],
        "untracked": [ln for ln in status if ln.startswith("??")],
        "pytest": pytest_res["tail"],
        "note_老卡Δ": "未碰 card_flow / 其他 experiments 的任何源文件；真卡 GPU 路径未跑（推断："
                      "render.py 改动只影响门禁类型域，`POS_TYPES` 对外取值域未动）",
    }

    # —— 确定性复跑 ——
    rerun = [r for fix in FIXTURES
             for r in try_all(fix.text, fix.role_map,
                              [sid for sid in (p1 + p2) if SKELETON_FIXTURE.get(sid) == fix.fid],
                              table, "verb")]
    first = [r for mode in ("verb",) for tier in ("tierB",) for r in w1[mode][tier].values()]
    deterministic = json.dumps(rerun, ensure_ascii=False, sort_keys=True) == \
        json.dumps(first, ensure_ascii=False, sort_keys=True)

    # —— 判定（PREREG §5 写死；W1 的 17/17 是 **P1（含动槽 17 条）**，P2 单独 7/7） ——
    def _unlock(mode: str, tier: str, ids: tuple[str, ...]) -> int:
        return sum(1 for sid in ids if _ok(w1[mode][tier][sid]))

    tier_a_p1 = _unlock("verb", "tierA", p1)
    tier_a_p2 = _unlock("verb", "tierA", p2)
    tier_b_p1 = _unlock("verb", "tierB", p1)
    tier_b_p2 = _unlock("verb", "tierB", p2)
    corpus_verb = corpus["modes"]["verb"]["sentences_with_ge1_unlocked"]
    corpus_base = corpus["modes"]["none"]["sentences_with_ge1_unlocked"]
    floor_best = max(w0["modes"][m]["prf"]["F1"] for m in FLOOR_RULES)
    verb_f1 = w0["modes"]["verb"]["prf"]["F1"]
    substantive = verb_f1 > floor_best          # 地板 gold F1 达 1.000 时按 PREREG 改判
    w1_pass = (tier_a_p1 >= 4) and (tier_b_p1 == 17) and (tier_b_p2 == 7) \
        and (corpus_verb > corpus_base) and substantive
    w2_pass = cmp_["p3_render_ok_after"] == 9 and cmp_["p3_schema_clean_after"] == 9
    w5_pass = (cmp_["official16_byte_diff"] == 0 and cmp_["cf21_byte_diff"] == 0
               and pytest_res["failed"] == 0 and pytest_res["passed"] == pytest_res["baseline"])
    if w1_pass and w3["all_zero"] and w4["all_zero"] and w5_pass and w2_pass:
        verdict = "可用"
    elif (tier_a_p1 + tier_b_p1 + tier_b_p2) > 0:
        verdict = "部分可用"
    else:
        verdict = "不可用"

    results = {
        "meta": {"script": "experiments/verb_card/run_verb_card.py",
                 "pos_types": list(R.POS_TYPES),
                 "pos_types_all": list(R.POS_TYPES_ALL),
                 "n_fixtures": len(FIXTURES), "sec": round(time.time() - t0, 1)},
        "counts": {"P1_含动槽_生成链": list(p1), "P1_n": len(p1),
                   "P2_含动槽_cardflow": list(p2), "P2_n": len(p2),
                   "P3_含否数槽": list(p3), "P3_n": len(p3),
                   "official_16": list(OFFICIAL_16),
                   "前提对账_cardflow25": counts},
        "w0_floors": w0,
        "w1_unlock": {m: {"tierA": {k: v for k, v in w1[m]["tierA"].items()},
                          "tierB": {k: v for k, v in w1[m]["tierB"].items()}}
                      for m in MODES},
        "w1_corpus": corpus,
        "w2": {"p3_before": cmp_["p3_before"], "p3_after": cmp_["p3_after"],
               "render_ok_before": cmp_["p3_render_ok_before"],
               "render_ok_after": cmp_["p3_render_ok_after"],
               "schema_clean_after": cmp_["p3_schema_clean_after"],
               "text_after": cmp_["p3_text_after"],
               "gates_after": cmp_["p3_gates_after"],
               "audit_after": cmp_["p3_audit_after"]},
        "w3_gates": w3,
        "w4_trace": w4,
        "w5": {"official16_byte_diff": cmp_["official16_byte_diff"],
               "official16_diff_ids": cmp_["official16_diff_ids"],
               "cf21_byte_diff": cmp_["cf21_byte_diff"],
               "pytest": pytest_res},
        "w6": w6,
        "deterministic": deterministic,
        "verdict": {
            "value": verdict,
            "tierA_P1": tier_a_p1, "tierA_P2": tier_a_p2,
            "tierB_P1": tier_b_p1, "tierB_P2": tier_b_p2,
            "tierB_P1P2_total": tier_b_p1 + tier_b_p2,
            "corpus_verb": corpus_verb, "corpus_base": corpus_base,
            "corpus_floors": {m: corpus["modes"][m]["sentences_with_ge1_unlocked"]
                              for m in FLOOR_RULES},
            "verb_F1": verb_f1, "best_floor_F1": floor_best, "substantive": substantive,
            "w1_pass": w1_pass, "w2_pass": w2_pass,
            "w3_pass": w3["all_zero"], "w4_pass": w4["all_zero"], "w5_pass": w5_pass,
        },
    }
    (HERE / "results.json").write_text(json.dumps(results, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    print(json.dumps({"counts": {k: (v if not isinstance(v, list) else len(v))
                                 for k, v in results["counts"].items() if k != "official_16"},
                      "w0_F1": {m: w0["modes"][m]["prf"]["F1"] for m in MODES},
                      "w1": {m: [w0["modes"][m]["unlock_total_P1P2_tierA"],
                                 w0["modes"][m]["unlock_total_P1P2_tierB"]] for m in MODES},
                      "corpus": {m: [corpus["modes"][m]["sentences_with_ge1_unlocked"],
                                     corpus["modes"][m]["unlocked_pairs"]] for m in MODES},
                      "w2": [cmp_["p3_render_ok_before"], cmp_["p3_render_ok_after"]],
                      "w3": w3, "w4": w4, "w5": [cmp_["official16_byte_diff"], pytest_res["tail"]],
                      "deterministic": deterministic, "verdict": results["verdict"],
                      "sec": results["meta"]["sec"]},
                     ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
