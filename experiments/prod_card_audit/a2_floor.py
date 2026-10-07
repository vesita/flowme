#!/usr/bin/env python3
"""A2 —— 生产四卡（pronoun / relation / sentiment / person）的 `max_naive` 完备电池对照。

口径见 PREREG §3：fit = train_S、eval = eval_S、查表逐字复用
`experiments/free_rule_floor/rules.py::lookup`（无 min_support、未见键回退 train 全局多数、报 seen_rate）；
目标 = `labels[:, 0]` 且只取 `t_labels > 0`（与 `evaluate_task.cls_acc` 同分母）；
只用原始 `text`，不读 spans/label/category/pair_key 等标注字段。

用法：
    CUDA_VISIBLE_DEVICES="" uv run python experiments/prod_card_audit/a2_floor.py
"""
from __future__ import annotations

import importlib.util
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from dtseek.tasks.plugin import resolve_tasks  # noqa: E402

from common import (  # noqa: E402
    SEEDS,
    TASK_SAMPLES,
    lookup,
    make_loader,
    split_of,
)

PUNCTS = "，、：；！？…—"          # 逐字取自 two_channel_head/build_gen_data.py:126
FOUR = ["pronoun", "relation", "sentiment", "person"]
RULES = ("majority", "len_bucket", "first_char", "last_char", "punct_pattern",
         "lex_rule", "lex_first_key", "n_cue")
#: 逐规则的**未取整**准确率（地板取 max 用；表里报 4 位小数）
_RAW: dict[str, float] = {}


def load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---- 四卡的闭集 cue 表（全部取自 src/dtseek/tasks/builtin/*/dataset.py，不新造）----

def cue_table(cap: str) -> tuple[list[tuple[int, str]], str]:
    """返回 ([(class_id, surface), ...], 来源说明)。"""
    if cap == "pronoun":
        from dtseek.tasks.builtin.pronoun.dataset import PRONOUN_MAP
        out = [(c, w) for c, ws in PRONOUN_MAP for w in ws]
        return out, "builtin/pronoun/dataset.py::PRONOUN_MAP"
    if cap == "sentiment":
        from dtseek.tasks.builtin.sentiment.dataset import EMOTION_KEYWORDS
        out = [(c, w) for c, ws in EMOTION_KEYWORDS for w in ws]
        return out, "builtin/sentiment/dataset.py::EMOTION_KEYWORDS"
    if cap == "relation":
        from dtseek.tasks.builtin.relation.lexicon import ANTONYM_PAIRS, SYNONYM_PAIRS
        out = [(1, w) for a, b in SYNONYM_PAIRS for w in (a, b)]
        out += [(2, w) for a, b in ANTONYM_PAIRS for w in (a, b)]
        return out, "builtin/relation/lexicon.py::SYNONYM_PAIRS+ANTONYM_PAIRS（类别=词对类型）"
    if cap == "person":
        from dtseek.tasks.builtin.person.dataset import _PERSON_VOCAB
        # person 的类别 = 出场顺序 id，**没有类别词表** ⇒ 首提及 id 恒为 1（结构事实），
        # 故词表只给「是不是人物提及」，类别按首提及恒 1 出（如实披露，见 REPORT）。
        out = [(1, w) for w in sorted(_PERSON_VOCAB)]
        return out, "builtin/person/dataset.py::_PERSON_VOCAB（类别按「首提及 id 恒 1」结构出）"
    raise KeyError(cap)


def cue_regex(words: list[str]) -> re.Pattern:
    return re.compile("|".join(sorted((re.escape(w) for w in words), key=len, reverse=True)))


def first_label(sample: dict, spec) -> int:
    """GenericTaskDataset 的 labels[:,0] 同构计算（用于交叉自检）。"""
    spans = sorted(sample["spans"], key=lambda x: x["start"])
    if spec.pair_emission:
        spans = spans[: len(spans) // 2 * 2]
    if spec.annotate_all and len(spans) > spec.max_steps:
        raise ValueError("annotate_all 样本切片数超过 max_steps")
    spans = spans[: spec.max_steps]
    return spans[0]["label"] if spans else 0


def y_of(samples: list[dict], spec) -> list[int]:
    ds = GenericTaskDataset_from(samples, spec)
    return [int(ds[i]["labels"][0]) for i in range(len(ds))]


def GenericTaskDataset_from(samples: list[dict], spec):
    from nano_char_tokenizer import NanoCharTokenizer
    from dtseek.tasks.runtime import GenericTaskDataset
    return GenericTaskDataset(samples, NanoCharTokenizer(), spec)


def build_features(cap: str, texts: list[str]) -> dict[str, list]:
    table, _src = cue_table(cap)
    words = [w for _, w in table]
    cls_of = {}
    for c, w in table:
        cls_of.setdefault(w, c)
    rx = cue_regex(words)
    feats: dict[str, list] = defaultdict(list)
    for t in texts:
        m = rx.search(t)
        feats["first_char"].append(t[0] if t else "")
        feats["last_char"].append(t[-1] if t else "")
        feats["len_bucket"].append(len(t) // 6)
        feats["punct_pattern"].append("".join(sorted(set(t) & set(PUNCTS))))
        # 词表/字面：长词优先首命中 ⇒ 类别；无命中 ⇒ None（回退 train 全局多数）
        if m is not None:
            feats["lex_rule"].append(cls_of[m.group(0)])
            feats["lex_first_key"].append(m.group(0))
        else:
            feats["lex_rule"].append(None)
            feats["lex_first_key"].append(None)
        feats["n_cue"].append(len(rx.findall(t)))
    return feats


def acc(pred: list, y: list[int]) -> float:
    return sum(1 for a, b in zip(pred, y) if a == b) / max(1, len(y))


def main() -> int:
    torch.set_num_threads(torch.get_num_threads())
    # 卡指标：读 A0 的产物（线上核 = 生产口径），缺失就响亮退出
    a0_path = HERE / "results" / "a0_bases.json"
    if not a0_path.exists():
        raise SystemExit("缺 results/a0_bases.json：先跑 a0_bases.py")
    a0 = json.loads(a0_path.read_text(encoding="utf-8"))

    out: dict = {"prereg": "experiments/prod_card_audit/PREREG.md",
                 "lookup_free_rule_floor": "experiments/free_rule_floor/rules.py::lookup",
                 "puncts": PUNCTS, "rules": list(RULES), "caps": {}}

    for cap in FOUR:
        spec = resolve_tasks([cap])[cap].spec
        _, src_note = cue_table(cap)
        cap_out: dict = {"cue_source": src_note, "seeds": {}}
        for S in SEEDS:
            _RAW.clear()
            ev, tr = split_of(cap, S)
            y_tr = y_of(tr, spec)
            y_ev = y_of(ev, spec)
            real_tr = [i for i, v in enumerate(y_tr) if v > 0]
            real_ev = [i for i, v in enumerate(y_ev) if v > 0]
            yt = [y_tr[i] for i in real_tr]
            ye = [y_ev[i] for i in real_ev]
            tr_txt = [tr[i]["text"] for i in real_tr]
            ev_txt = [ev[i]["text"] for i in real_ev]

            f_tr = build_features(cap, tr_txt)
            f_ev = build_features(cap, ev_txt)

            glob = Counter(yt).most_common(1)[0][0]
            rec: dict = {"n_train": len(yt), "n_eval": len(ye),
                         "n_train_all": len(y_tr), "n_eval_all": len(y_ev),
                         "label_dist_train": dict(Counter(yt)),
                         "label_dist_eval": dict(Counter(ye)),
                         "train_majority": glob,
                         "seen_rate": {}}
            # 1) majority
            _RAW["majority"] = acc([glob] * len(ye), ye)
            rec["majority"] = round(_RAW["majority"], 4)
            # 2-4) 表面字面：free_rule_floor::lookup 查表（无 min_support）
            rec["len_bucket"] = _fit(f_tr["len_bucket"], yt, f_ev["len_bucket"], ye, rec, "len_bucket")
            rec["first_char"] = _fit(f_tr["first_char"], yt, f_ev["first_char"], ye, rec, "first_char")
            rec["last_char"] = _fit(f_tr["last_char"], yt, f_ev["last_char"], ye, rec, "last_char")
            rec["punct_pattern"] = _fit(f_tr["punct_pattern"], yt, f_ev["punct_pattern"], ye, rec, "punct_pattern")
            # 5) lex_rule：纯字面（无拟合、无回退表），无命中回退 train 全局多数
            _RAW["lex_rule"] = acc(
                [glob if v is None else v for v in f_ev["lex_rule"]], ye)
            rec["lex_rule"] = round(_RAW["lex_rule"], 4)
            # 7) lex_first_key：首命中词条 → train 多数（查表，free_rule_floor 口径）
            rec["lex_first_key"] = _fit(f_tr["lex_first_key"], yt, f_ev["lex_first_key"], ye, rec, "lex_first_key")
            # 8) n_cue：计数类（D2）
            rec["n_cue"] = _fit(f_tr["n_cue"], yt, f_ev["n_cue"], ye, rec, "n_cue")

            rec["max_naive_raw"] = max(_RAW[k] for k in RULES)
            rec["max_naive"] = round(rec["max_naive_raw"], 6)
            # 披露：naive_skeleton 的 min_support=30 变体（不计入地板）
            rec["disclosed_min_support30"] = {
                k: round(_fit(f_tr[k], yt, f_ev[k], ye, None, min_support=30), 4)
                for k in ("len_bucket", "first_char", "last_char", "punct_pattern")}

            # 卡指标（生产口径 = 线上核，来自 A0）+ 卡 − 地板
            card_cls = a0["runs"]["combo_full"]["E_online"][cap][str(S)]["cls_acc"]
            card_exact = a0["runs"]["combo_full"]["E_online"][cap][str(S)]["exact_match"]
            rec["card_cls_acc_online"] = card_cls
            rec["card_exact_match_online"] = card_exact
            rec["card_minus_floor_cls"] = round(card_cls - rec["max_naive_raw"], 6)
            rec["below_floor"] = bool(card_cls < rec["max_naive_raw"] - 1e-12)
            rec["marginal_abs_lt_1e3"] = bool(abs(card_cls - rec["max_naive_raw"]) < 1e-3)
            cap_out["seeds"][str(S)] = rec
            print(f"[{cap} s{S}] n_tr={rec['n_train']} n_ev={rec['n_eval']} "
                  f"maj={rec['majority']:.4f} lex_rule={rec['lex_rule']:.4f} "
                  f"n_cue={rec['n_cue']:.4f} max_naive={rec['max_naive']:.4f} | "
                  f"card={card_cls:.4f} 卡−地板={rec['card_minus_floor_cls']:+.4f} "
                  f"{'<低于地板>' if rec['below_floor'] else ''}", flush=True)
        out["caps"][cap] = cap_out

    # ---- 汇总判定（A3）----
    EPS = 1e-6
    verdict, note = {}, {}
    for cap in FOUR:
        d = [out["caps"][cap]["seeds"][str(S)]["card_minus_floor_cls"] for S in SEEDS]
        verdict[cap] = ("未超过免费规则" if all(x <= EPS for x in d)
                        else "超过地板" if all(x > EPS for x in d)
                        else "两 seed 不一致")
        note[cap] = ("并列（|Δ|<1e-3，实质相等）"
                     if all(abs(x) < 1e-3 for x in d) else "")
    out["verdict_card_vs_floor"] = verdict
    out["verdict_note"] = note
    print("\n[A2/A3] 卡 − max_naive(7条) 逐卡（两 seed）")
    for cap in FOUR:
        d = [out["caps"][cap]["seeds"][str(S)]["card_minus_floor_cls"] for S in SEEDS]
        extra = ("；" + note[cap]) if note[cap] else ""
        print(f"  {cap:10s} Δ = {d[0]:+.6f} / {d[1]:+.6f}  ⇒  {verdict[cap]}{extra}")

    res = HERE / "results" / "a2_floor.json"
    res.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {res}")
    print("A2_DONE")
    return 0


def _fit(f_tr, y_tr, f_ev, y_ev, rec, rule: str = "", min_support: int | None = None) -> float:
    """free_rule_floor::lookup 口径（min_support=None）；min_support 给值时按 tch 口径。"""
    glob = Counter(y_tr).most_common(1)[0][0]
    if min_support is None:
        pred, sr = lookup(f_tr, y_tr, f_ev)
        if rec is not None:
            rec.setdefault("seen_rate", {})[rule] = round(sr, 4)
        if rule:
            _RAW[rule] = acc(pred, y_ev)
    else:
        groups: dict = defaultdict(list)
        for k, y in zip(f_tr, y_tr):
            groups[k].append(y)
        tab = {k: Counter(v).most_common(1)[0][0] for k, v in groups.items()
               if len(v) >= min_support}
        pred = [tab.get(k, glob) for k in f_ev]
    return round(acc(pred, y_ev), 4)


if __name__ == "__main__":
    sys.exit(main())
