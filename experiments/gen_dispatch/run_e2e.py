#!/usr/bin/env python3
"""端到端驱动 —— 执行 `PREREG.md` 里跑前写死的判据。

    uv run python experiments/gen_dispatch/run_e2e.py --mode selfcheck   # 假卡 + 契约 + H1a
    uv run python experiments/gen_dispatch/run_e2e.py --mode sample      # 语料过滤 + 抽样（落 samples.json）
    uv run python experiments/gen_dispatch/run_e2e.py --mode random      # 随机批 N=1000（真卡）
    uv run python experiments/gen_dispatch/run_e2e.py --mode enriched    # 富集批 M=400（真卡）
    uv run python experiments/gen_dispatch/run_e2e.py --mode adv         # 对抗组 A1–A14
    uv run python experiments/gen_dispatch/run_e2e.py --mode h1          # H1 逐字可复现
    uv run python experiments/gen_dispatch/run_e2e.py --mode all         # 除 sample 外全部

产出 `results_<mode>.json`；只写 `experiments/gen_dispatch/`、`logs/`、`/tmp`。
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "card_flow"))
sys.path.insert(0, str(HERE))

from dtseek.tasks import dispatch as DP  # noqa: E402
from dtseek.tasks import render as R  # noqa: E402
from dtseek.tasks.corpus import resolve_corpus_files  # noqa: E402

from adv import run_cases  # noqa: E402
from pipeline import (  # noqa: E402
    BAG_CARDS_ALL,
    execute,
    fake_propose_anchors,
    propose_with_anchors,
    verify_record,
)
from rules import RULE_TEXT, contract_problems, rule_audit  # noqa: E402

SENT_SPLIT = re.compile(r"[。！？!?；]")
HANZI = re.compile(r"[一-鿿]")
PREFIXES = ("用户：", "模型：", "user:", "assistant:")

#: 抽样口径（PREREG §4）：阈值、规模、seed 全部跑前写死。
HAN_RATIO_KEEP = 0.5
N_RANDOM = 1000
M_ENRICHED = 400
SEED = 42
NAIVE_SUBSET = 200     # max_naive 的假卡批
H1_SUBSET = 100        # 端到端逐字复现的条数
SENT_MIN, SENT_MAX, HAN_MIN = 6, 40, 6
SAMPLES_JSON = HERE / "samples.json"


# ---- 抽样 -------------------------------------------------------------------

def _sentences(line: str) -> list[str]:
    s = line.strip()
    for pre in PREFIXES:
        if s.startswith(pre):
            s = s[len(pre):]
    return [p.strip() for p in SENT_SPLIT.split(s) if p.strip()]


def _sent_ok(s: str) -> bool:
    return SENT_MIN <= len(s) <= SENT_MAX and len(HANZI.findall(s)) >= HAN_MIN


def _reservoir_add(pool: list[str], cap: int, seen: int, item: str, rng: random.Random) -> None:
    """标准 reservoir sampling（与文件返回顺序无关，只由 seed 决定）。"""
    if len(pool) < cap:
        pool.append(item)
        return
    j = rng.randrange(seen)
    if j < cap:
        pool[j] = item


def build_samples() -> dict:
    """语料入口 → 逐文件自算汉字占比（≥0.5 才留）→ 两个 reservoir。"""
    from lexicon import IDIOM_TYPE

    files = resolve_corpus_files()          # fail-closed：命中为空即抛
    rows: list[dict] = []
    t0 = time.perf_counter()
    for path in files:
        n_lines = han = tot = 0
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                n_lines += 1
                t = line.strip()
                if not t:
                    continue
                han += len(HANZI.findall(t))
                tot += len(t) - t.count(" ") - t.count("\t")
        ratio = han / tot if tot else 0.0
        rows.append({"file": Path(path).name, "lines": n_lines,
                     "han_chars": han, "nonspace_chars": tot,
                     "han_ratio": round(ratio, 4),
                     "keep": ratio >= HAN_RATIO_KEEP})
    kept = [r for r in rows if r["keep"]]
    dropped = [r for r in rows if not r["keep"]]
    if not kept:
        raise SystemExit(f"语料过滤后 0 个文件（阈值 {HAN_RATIO_KEEP}）：fail-closed")

    rng_r, rng_e = random.Random(SEED), random.Random(SEED + 1)
    rand_res: list[str] = []
    idio_res: list[str] = []
    n_sent = n_idio = 0
    idiom_keys = tuple(IDIOM_TYPE)
    by_name = {Path(f).name: f for f in files}
    for row in kept:
        with open(by_name[row["file"]], encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                for sent in _sentences(line):
                    if not _sent_ok(sent):
                        continue
                    n_sent += 1
                    _reservoir_add(rand_res, N_RANDOM, n_sent, sent, rng_r)
                    if any(k in sent for k in idiom_keys):
                        n_idio += 1
                        _reservoir_add(idio_res, M_ENRICHED, n_idio, sent, rng_e)

    def uniq(xs: list[str]) -> list[str]:
        seen: set[str] = set()
        out = []
        for x in xs:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    enriched = uniq(idio_res)
    idio_set = set(enriched)
    random_part = [s for s in uniq(rand_res) if s not in idio_set]
    out = {
        "formula": "汉字占比 = 汉字数 / 非空白字符数（整文件流式）",
        "threshold": HAN_RATIO_KEEP,
        "seed": SEED,
        "files_all": rows,
        "files_used": [r["file"] for r in kept],
        "files_dropped": dropped,
        "n_sentences_pool": n_sent,
        "n_idiom_sentences_pool": n_idio,
        "n_random": len(random_part),
        "n_enriched": len(enriched),
        "random": random_part,
        "enriched": enriched,
        "idiom_lexicon_size": len(idiom_keys),
        "scan_sec": round(time.perf_counter() - t0, 2),
    }
    SAMPLES_JSON.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")
    return out


def load_samples() -> dict:
    if not SAMPLES_JSON.exists():
        raise SystemExit(f"缺 {SAMPLES_JSON}：先跑 --mode sample（fail-closed）")
    data = json.loads(SAMPLES_JSON.read_text(encoding="utf-8"))
    if not data.get("random"):
        raise SystemExit("samples.json 里没有随机样本：fail-closed")
    return data


# ---- 引擎（真卡） -------------------------------------------------------------

def make_engine() -> tuple[dict[str, tuple[Any, list[str]]], list[str], dict]:
    """默认 4 张卡 + 否定卡（默认基座）+ **成语卡（它自带的基座，独立引擎）**。

    成语卡的 `.pt` 是一体训练快照、不是 v1 卡，且它的 `doc_encoder` 与默认基座
    最大绝对差 0.0438 ⇒ 不能挂到默认引擎上（实测换基座会改锚点，见 `idiom_card.py`）。
    """
    import idiom_card as idiom_card_mod
    from dtseek.tasks.engine import MultiTaskEngine

    engine = MultiTaskEngine()
    neg_path = ROOT / "checkpoints" / "negation_accept_card.pt"
    try:
        neg_name = engine.attach(neg_path)
    except Exception as exc:  # noqa: BLE001 —— 挂不上就 fail-closed 中止
        raise SystemExit(f"挂载失败 {neg_path}：{type(exc).__name__}: {exc}")

    idiom_engine, idiom_name, idiom_info = idiom_card_mod.make_idiom_engine()
    jobs: dict[str, tuple[Any, list[str]]] = {
        "main": (engine, sorted(engine.decoders)),
        "idiom": (idiom_engine, [idiom_name]),
    }
    attached = sorted(set(engine.decoders) | {idiom_name})
    missing = sorted(set(BAG_CARDS_ALL) - set(attached))
    if missing:
        raise SystemExit(f"声明袋卡未挂载：{missing}（fail-closed，不许少卡硬跑）")
    extra = {"negation": str(neg_path.relative_to(ROOT)), "negation_name": neg_name,
             "idiom": str(idiom_card_mod.CARD_SRC.relative_to(ROOT)), "idiom_info": idiom_info}
    return jobs, attached, extra


def _make_propose(jobs: dict[str, tuple[Any, list[str]]]):
    def propose(text: str):
        main_eng, main_cards = jobs["main"]
        idiom_eng, idiom_cards = jobs["idiom"]
        return propose_with_anchors(text, main_eng, main_cards,
                                   also=[(idiom_eng, idiom_cards)])

    return propose


# ---- 单批执行 ----------------------------------------------------------------

def _step_dict(s: Any) -> dict:
    return {"module_id": s.module_id,
            "input_span": list(s.input_span) if s.input_span else None,
            "params": dict(s.params)}


def _sample_entry(record: dict, debug: dict) -> dict:
    ref_map = record.get("ref_map") or []
    return {
        "input": debug.get("source"),
        "plan": record.get("plan"),
        "plan_steps": [_step_dict(s) for s in debug.get("turn_steps", [])],
        "plan_step_id": record.get("plan_step_id"),
        "terminal": record.get("terminal"),
        "channel": record.get("channel"),
        "rule": record.get("reason"),
        "cards_run": record.get("cards_run"),
        "instruction": record.get("instruction"),
        "text": record.get("text"),
        "evidence": record.get("evidence"),
        "ref_map_head": [{k: e.get(k) for k in ("unit_id", "cls", "text", "out", "span")}
                         for e in ref_map[:8]],
        "g_stats": {k: debug.get("g_stats", {}).get(k)
                    for k in ("combos", "cap", "skeletons_tried", "why")},
        "bag_pos": (debug.get("bag_stats") or {}).get("pos_count"),
    }


def run_batch(tag: str, sentences: list[str], *, jobs: dict[str, tuple[Any, list[str]]],
              attached: list[str], h1_subset: int = H1_SUBSET,
              naive_subset: int | None = None) -> dict:
    """真卡批：逐条执行 + 独立复核 + H1 复跑 + max_naive 假卡对照。"""
    attached = set(attached)
    propose = _make_propose(jobs)
    st: Counter = Counter()
    rows: list[dict] = []
    gen_samples: list[dict] = []
    ptr_examples: list[dict] = []
    rej_examples: list[dict] = []
    problems_ex: list[dict] = []
    h1_fail: list[dict] = []
    t0 = time.perf_counter()

    for i, sent in enumerate(sentences):
        record, debug = execute(sent, propose=propose, attached=attached, need_g=True)
        st["n"] += 1
        st["ch_" + record.get("channel", "?")] += 1
        st["term_" + str(record.get("terminal"))] += 1
        st["row_" + str(debug.get("rule_row"))] += 1
        st["G" if debug.get("G") else "notG"] += 1
        st["P" if debug.get("P") else "notP"] += 1

        cprob = contract_problems(record, n_turn_steps=debug.get("n_turn_steps"))
        if cprob:
            st["contract_fail"] += 1
            if len(problems_ex) < 5:
                problems_ex.append({"input": sent, "contract": cprob})
        if record["kind"] == "reject":
            st["reject_n"] += 1
            if "text" in record and record["text"] is not None:
                st["reject_with_text"] += 1
            if not (record.get("reason") or "").strip():
                st["reject_reason_empty"] += 1
            # H2a 正例误拒：该通道本有合法输出却拒答
            if (record.get("terminal") == "generate" and debug.get("G")) or \
               (record.get("terminal") == "pointer" and debug.get("P")):
                st["false_reject"] += 1
            if len(rej_examples) < 6:
                rej_examples.append({"input": sent, "reason": record.get("reason"),
                                     "terminal": record.get("terminal"),
                                     "rule": debug.get("rule_row")})
        else:
            st["text_n"] += 1
            vprob = verify_record(record, debug)
            if vprob:
                st["illegal_text"] += 1
                if len(problems_ex) < 5:
                    problems_ex.append({"input": sent, "verify": vprob[:4]})
            if record.get("channel") == "generate":
                st["h3_n"] += 1
                st["h3_struct"] += sum(1 for x in vprob if x.startswith("结构侧"))
                st["h3_deref"] += sum(1 for x in vprob if x.startswith("解引用侧"))
                if len(gen_samples) < 10:
                    gen_samples.append(_sample_entry(record, debug))
            elif len(ptr_examples) < 6:
                ptr_examples.append({"input": sent, "text": record.get("text"),
                                     "evidence": record.get("evidence"),
                                     "terminal": record.get("terminal"),
                                     "plan_step_id": record.get("plan_step_id")})

        rows.append({"i": i, "input": sent, "channel": record.get("channel"),
                     "terminal": record.get("terminal"), "G": bool(debug.get("G")),
                     "P": bool(debug.get("P")), "rule": debug.get("rule_row"),
                     "plan_step_id": record.get("plan_step_id"),
                     "reason": (record.get("reason") or "")[:120]})

        # H1：前 h1_subset 条再跑一遍，计划必须逐字相同
        if i < h1_subset:
            rec2, _dbg2 = execute(sent, propose=propose, attached=attached, need_g=True)
            if rec2.get("plan") != record.get("plan"):
                st["h1_plan_mismatch"] += 1
                if len(h1_fail) < 5:
                    h1_fail.append({"input": sent, "first": record.get("plan"),
                                    "second": rec2.get("plan")})
            if rec2 != record:
                st["h1_record_mismatch"] += 1
            st["h1_checked"] += 1

    # max_naive：N4 假卡（词典）在同一子集上跑同一条管线
    naive_fake: dict[str, Any] = {}
    if naive_subset:
        sub = sentences[:naive_subset]
        fake_st: Counter = Counter()
        for sent in sub:
            rec, dbg = execute(sent, propose=fake_propose_anchors, attached=None,
                               need_g=True)
            fake_st["n"] += 1
            fake_st["G"] += int(bool(dbg.get("G")))
            fake_st["gen_ok"] += int(rec.get("channel") == "generate"
                                     and rec.get("kind") == "text")
            if rec.get("kind") == "text":
                fake_st["illegal"] += int(bool(verify_record(rec, dbg)))
        naive_fake = dict(fake_st)

    n = st["n"] or 1
    naive = {
        "N1_全生成": {"illegal": st["notG"], "legal_yield": st["G"],
                      "legal_rate": round(st["G"] / n, 4)},
        "N2_全指针": {"illegal": st["notP"], "legal_yield": st["P"],
                      "legal_rate": round(st["P"] / n, 4)},
        "N3_按type静态": {
            "plain→generate": {"illegal": st["notG"], "legal_yield": st["G"]},
            "plain→pointer": {"illegal": st["notP"], "legal_yield": st["P"]}},
        "N4_词典假卡": naive_fake,
        "N5_echo复述": {"pass": 0,
                       "why": "无 instruction/ref_map 且 evidence 空 ⇒ 契约与 I3 同时不过"},
        "max_naive_通道合法率": max(round(st["G"] / n, 4), round(st["P"] / n, 4)),
        "本方_非法输出": st["illegal_text"],
    }
    return {
        "tag": tag,
        "n": st["n"],
        "channel": {k[3:]: v for k, v in st.items() if k.startswith("ch_")},
        "terminal": {k[5:]: v for k, v in st.items() if k.startswith("term_")},
        "G_true": st["G"], "P_true": st["P"],
        "reject": {"n": st["reject_n"], "with_text_field": st["reject_with_text"],
                   "reason_empty": st["reject_reason_empty"],
                   "false_reject": st["false_reject"]},
        "text": {"n": st["text_n"], "illegal": st["illegal_text"]},
        "h2_contract_fail": st["contract_fail"],
        "h3": {"n_generated": st["h3_n"], "structure_problems": st["h3_struct"],
               "deref_problems": st["h3_deref"]},
        "h1": {"checked": st["h1_checked"],
               "plan_mismatch": st["h1_plan_mismatch"],
               "record_mismatch": st["h1_record_mismatch"], "fails": h1_fail},
        "rule_rows": {k[4:]: v for k, v in st.items() if k.startswith("row_")},
        "max_naive": naive,
        "wall_sec": round(time.perf_counter() - t0, 2),
        "generate_samples": gen_samples,
        "pointer_examples": ptr_examples,
        "reject_examples": rej_examples,
        "problems_examples": problems_ex,
        "rows": rows,
    }


# ---- H1a：全状态空间 ----------------------------------------------------------

def h1_state_space() -> dict:
    """`dispatch.plan()` 在 4860 个状态上连跑两遍，`as_tuple()` 必须全等。"""
    reg = DP.load_default_registry()
    first = [p.as_tuple() for p in
             (DP.plan(s, registry=reg) for s in DP.enumerate_state_space())]
    second = [p.as_tuple() for p in
              (DP.plan(s, registry=reg) for s in DP.enumerate_state_space())]
    same = first == second
    return {"n_states": DP.STATE_SPACE_SIZE, "runs": 2, "identical": same,
            "n_first": len(first), "n_second": len(second)}


def h1_dialogue_tables() -> dict:
    """注入对话层三张表后再跑一遍（本实验真正走的那条路）。"""
    from dtseek.tasks import dialogue as D

    reg = DP.load_default_registry()
    kw = {"registry": reg, "info_bits": D.DIALOGUE_BITS,
          "required_bits": D.DIALOGUE_REQUIRED_BITS, "bit_kinds": D.DIALOGUE_BIT_KINDS,
          "source_span": (0, 12)}
    a = [DP.plan(s, **kw).as_tuple() for s in DP.enumerate_state_space()]
    b = [DP.plan(s, **kw).as_tuple() for s in DP.enumerate_state_space()]
    return {"n_states": len(a), "identical": a == b}


# ---- 模式 --------------------------------------------------------------------

def mode_selfcheck() -> dict:
    """假卡自检：契约谓词 + 规则审计 + 假卡端到端 + H1a。"""
    fake_sentences = [
        "你说的对，我们明天试试。", "我不喜欢这个方案，太慢了。",
        "他不是坏人，你别生气。", "我们跑一次看看效果。",
        "这个太难了，我做不了。", "你觉得好吗？我觉得好。",
        "用户说我讨厌等待。", "别说了，我不在乎。",
        "张三喜欢李四。", "我们讨论解决这个问题。",
    ]
    st: Counter = Counter()
    samples: list[dict] = []
    rejects: list[dict] = []
    for s in fake_sentences * 3:
        rec, dbg = execute(s, propose=fake_propose_anchors, attached=set(BAG_CARDS_ALL),
                           need_g=True)
        st["n"] += 1
        st["ch_" + rec["channel"]] += 1
        cp = contract_problems(rec, n_turn_steps=dbg.get("n_turn_steps"))
        if cp:
            st["contract_fail"] += 1
        vp = verify_record(rec, dbg)
        if rec["kind"] == "text" and vp:
            st["illegal"] += 1
        if rec["kind"] == "text" and rec["channel"] == "generate" and len(samples) < 5:
            samples.append(_sample_entry(rec, dbg))
        if rec["kind"] == "reject" and len(rejects) < 5:
            rejects.append({"input": s, "reason": rec.get("reason"),
                            "has_text": "text" in rec})
        st["G"] += int(bool(dbg.get("G")))
        st["P"] += int(bool(dbg.get("P")))
    audit = rule_audit()
    return {
        "n": st["n"],
        "channel": {k[3:]: v for k, v in st.items() if k.startswith("ch_")},
        "contract_fail": st["contract_fail"],
        "illegal_text": st["illegal"],
        "G_true": st["G"], "P_true": st["P"],
        "rule_audit": audit,
        "h1a_state_space": h1_state_space(),
        "h1b_dialogue_tables": h1_dialogue_tables(),
        "generate_samples": samples,
        "reject_examples": rejects,
        "rule_text": RULE_TEXT,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["selfcheck", "sample", "random", "enriched", "adv",
                             "h1", "all"])
    args = ap.parse_args()
    t0 = time.perf_counter()
    out: dict[str, Any] = {"mode": args.mode, "started": time.strftime("%F %T")}

    if args.mode in ("selfcheck", "all"):
        out["selfcheck"] = mode_selfcheck()
    if args.mode == "sample":
        out["sample"] = build_samples()
    if args.mode in ("adv", "all"):
        out["adv"] = run_cases()
    if args.mode in ("random", "enriched", "h1", "all"):
        data = load_samples()
        jobs, attached, extra = make_engine()
        out["engine"] = {"cards": attached, "detail": extra}
        out["samples"] = {"files_used": data["files_used"],
                          "files_dropped": data["files_dropped"],
                          "n_random": data["n_random"], "n_enriched": data["n_enriched"],
                          "formula": data["formula"], "threshold": data["threshold"]}
        batches = []
        if args.mode in ("random", "h1", "all"):
            batches.append(("random", data["random"], H1_SUBSET, NAIVE_SUBSET))
        if args.mode in ("enriched", "all"):
            batches.append(("enriched", data["enriched"], H1_SUBSET, None))
        if args.mode == "h1":
            batches = [("h1", data["random"][:H1_SUBSET], H1_SUBSET, None)]
        for tag, sents, h1n, naiven in batches:
            out[tag] = run_batch(tag, sents, jobs=jobs, attached=attached,
                                 h1_subset=h1n, naive_subset=naiven)

    out["wall_sec"] = round(time.perf_counter() - t0, 2)
    path = HERE / f"results_{args.mode}.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items()
                      if k not in ("random", "enriched", "h1")},
                     ensure_ascii=False, indent=1)[:6000])
    print(f"[写出] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
