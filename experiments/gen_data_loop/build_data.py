#!/usr/bin/env python3
"""数据生成模式三步管线：卡片提议 → 可核对谓词筛选 → 非内容编辑重整。

产出（全部落 data/*.jsonl，逐行带 source / kind / preds 可追溯）：
  data/train.jsonl             8000 = 有效 4000 + 无效 4000（提案臂）
  data/test.jsonl              1600 = 800 + 800
  data/adversarial.jsonl       1200 = 600 + 600（不进训练）
  data/ctrl_train.jsonl        8000  注入对照臂（复刻 anchored_select 构造与标签口径）
  data/ctrl_test.jsonl         1600
  data/ctrl_adversarial.jsonl  1200
  data/natural.jsonl           200   手写自然用句（不经本管线）
  data/stats.json              规模 / 接受率 / 朴素规则 / 重叠 / 多样性 / 长度语体

三步（PREREG §1）：
  1) 候选生成：神经指针卡（checkpoints/cards/*.pt，CPU 推理）+ 真值源探测器（规则版指针发射）
     + 同句竞争装配；候选一律是**真实输入上的区间**，不是注入。
  2) 可核对筛选：P1 锚定 = 准入（不锚定就丢弃）；P2 可回溯 / P3 规范类别名 /
     P4 不跨分句 / P5 与真值源一致 = 打标。全程不用学出来的分数。
  3) 重整：只做非内容编辑（补标点 / 插连词 / 分句调序），内容词多重集上锁，
     真值源在完整句上必须给出与原句逐位映射一致的输出。

fail-closed：语料入口只用 dtseek.tasks.corpus.resolve_corpus_files（glob 空即抛错，不自己 glob），
任一断言不满足即抛错，绝不静默降级。
"""
from __future__ import annotations

import json
import math
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.utils.data as tud  # noqa: E402
from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402

from dtseek.tasks.artifacts import build_card_decoder, load_base_encoder, read_card  # noqa: E402
from dtseek.tasks.builtin.negation import SPEC as NEG_SPEC  # noqa: E402
from dtseek.tasks.builtin.negation.dataset import (  # noqa: E402
    NEG_MARKERS, extract_negation_spans, reject_reason,
)
from dtseek.tasks.builtin.pronoun import SPEC as PRO_SPEC  # noqa: E402
from dtseek.tasks.builtin.pronoun.dataset import PRONOUN_MAP, extract_all_spans  # noqa: E402
from dtseek.tasks.builtin.sentiment import SPEC as SENT_SPEC  # noqa: E402
from dtseek.tasks.builtin.sentiment.dataset import (  # noqa: E402
    LEXICON_BY_CAT, extract_emotion_spans,
)
from dtseek.tasks.corpus import CORPUS_GLOB, resolve_corpus_files  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, _rollout  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"

SPLIT_SEED = 20240927

#: 提案臂源句池 / 行数目标（PREREG §3）
POOL_N = {"train": 12000, "test": 3000, "adv": 2500}
N_ROWS = {"train": 8000, "test": 1600, "adv": 1200}
#: 注入对照臂源句池（每句 1 正 + 1 负 ⇒ 行数 = 池 × 2；不足则从溢出池补齐，仍与提案池不相交）
CTRL_POOL_N = {"train": 4000, "test": 800, "adv": 600}
OVERFLOW_N = 3000

#: 负例分层配额（PREREG §3）
NEG_QUOTA = {"neural_err": 0.40, "crosscat": 0.25, "boundary": 0.15,
             "alias": 0.10, "punct_edge": 0.10}
ADV_QUOTA = {"crosscat": 0.50, "samecat": 0.50}

MAX_LEN = 96
PREFIX, SUFFIX, CATP = "原文：", "候选：", "类："
SEP = "|"
SENT_SPLIT = re.compile(r"[。！？\n；;]+")
ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
CJK = re.compile(r"[一-龥]")
BAD = re.compile(r"[�\t｜|]|原文：|候选：|类：")
PUNCT_ALL = frozenset("，、；：。！？!?；;，．,.：:；;")
CLAUSE_SPLIT = re.compile(r"([，、；：。！？]+)")
CONJS = ("然后", "并且", "但是", "所以", "于是", "接着", "而且")
TERMINALS = frozenset("。！？")

NEURAL_CARDS = ("pronoun", "sentiment")   # negation 无对应卡权重 ⇒ 只有规则版提议
RULE_CARDS = ("negation", "sentiment", "pronoun")

# ---------------------------------------------------------------------------
# 类别体系（跑前从 TaskSpec 快照取，不硬编码）
# ---------------------------------------------------------------------------
CARD_OF_CAT: dict[str, str] = {}
CANONICAL: set[str] = set()
NONCANON: dict[str, list[str]] = {}
for _spec, _card in ((NEG_SPEC, "negation"), (SENT_SPEC, "sentiment"), (PRO_SPEC, "pronoun")):
    _nc = [_spec.label] + [c.label for c in _spec.classes if c.label and c.label != c.name]
    NONCANON[_card] = sorted({x for x in _nc if x})
    for _i, _c in enumerate(_spec.classes):
        if _i == 0:
            continue
        CANONICAL.add(_c.name)
        CARD_OF_CAT[_c.name] = _card

LEX_ALL: set[str] = set(NEG_MARKERS)
for _w in PRONOUN_MAP:
    LEX_ALL.update(_w[1])
for _ws in LEXICON_BY_CAT.values():
    LEX_ALL.update(_ws)


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[gen_data_loop fail-closed] {msg}")


def cjk(s: str) -> int:
    return len(CJK.findall(s))


def ok_sentence(s: str) -> bool:
    if not (14 <= len(s) <= 44) or BAD.search(s):
        return False
    if cjk(s) < 0.6 * len(s):
        return False
    if re.search(r"[A-Za-z]{3,}", s):
        return False
    return SEP not in s


# ---------------------------------------------------------------------------
# 语料 → 池（池两两不相交）
# ---------------------------------------------------------------------------
def collect_sentences(n: int, per_file: int = 6000, seed: int = SPLIT_SEED) -> tuple[list[str], list[str]]:
    files = resolve_corpus_files(CORPUS_GLOB)          # fail-closed：空 glob 直接抛
    rnd = random.Random(seed)
    rnd.shuffle(files)
    out: list[str] = []
    seen: set[str] = set()
    used: list[str] = []
    for p in files:
        got_file = 0
        with open(p, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                t = ROLE_PREFIX.sub("", line.strip())
                for s in SENT_SPLIT.split(t):
                    s = s.strip()
                    if s in seen or not ok_sentence(s):
                        continue
                    seen.add(s)
                    out.append(s)
                    got_file += 1
                if len(out) >= n or got_file >= per_file:
                    break
        if got_file:
            used.append(f"{p}({got_file})")
        if len(out) >= n:
            break
    check(len(out) >= n, f"语料可用句子不足：{len(out)} < {n}")
    return out, used


# ---------------------------------------------------------------------------
# 真值源（手写、可审计）与神经卡提议
# ---------------------------------------------------------------------------
def card_truth(card: str, sent: str) -> list[tuple[int, int, str]]:
    if card == "negation":
        if reject_reason(sent) is not None:
            return []
        return [(s["start"], s["end"], NEG_SPEC.classes[1].name) for s in extract_negation_spans(sent)]
    if card == "sentiment":
        lab, spans = extract_emotion_spans(sent)
        if lab == -1:
            return []
        return [(s["start"], s["end"], SENT_SPEC.classes[s["label"]].name) for s in spans]
    return [(s["start"], s["end"], PRO_SPEC.classes[s["label"]].name) for s in extract_all_spans(sent)]


def gold_of(sent: str) -> list[tuple[int, int, str]]:
    out: list[tuple[int, int, str]] = []
    for c in RULE_CARDS:
        out.extend(card_truth(c, sent))
    return sorted(set(out))


def run_neural(sents: list[str]) -> list[list[tuple[int, int, str]]]:
    """现有神经指针卡在真实句子上的发射（半开区间 + 规范类别名）。CPU，不占 GPU。"""
    device = torch.device("cpu")
    enc, _ck = load_base_encoder(ROOT / "checkpoints" / "base_encoder.pt", device)
    tok = NanoCharTokenizer()
    out: list[list[tuple[int, int, str]]] = [[] for _ in sents]
    for card in NEURAL_CARDS:
        dec, spec = build_card_decoder(read_card(ROOT / "checkpoints" / "cards" / f"{card}.pt"), device)
        ds = GenericTaskDataset([{"text": s, "spans": []} for s in sents], tok, spec)
        loader = tud.DataLoader(ds, batch_size=64, shuffle=False)
        bi = 0
        with torch.no_grad():
            for b in loader:
                inp, mask = b["input_ids"], b["attention_mask"]
                mem = enc(inp, attention_mask=mask)
                preds = _rollout(dec, mem, mask, inp.shape[0], spec, ndb=None, input_ids=inp)
                for k, pr in enumerate(preds):
                    for (lab, s0, e0) in pr:
                        if lab <= 0:
                            continue
                        name = spec.classes[lab].name
                        if name not in CANONICAL:
                            continue
                        out[bi + k].append((s0, e0 + 1, name))    # e0 是闭端点 ⇒ 半开
                bi += len(preds)
        del dec
    return [sorted(set(v)) for v in out]


# ---------------------------------------------------------------------------
# 谓词（P1 准入 / P2–P5 打标）
# ---------------------------------------------------------------------------
def p1_anchor(text: str, full: str, s: int, e: int) -> bool:
    return 0 <= s < e <= len(full) and full[s:e] == text


def p2_traceable(text: str) -> bool:
    if not text or not cjk(text):
        return False
    if text[0] in PUNCT_ALL or text[-1] in PUNCT_ALL:
        return False
    return cjk(text) >= 0.6 * len(text)


def p3_canonical(cat: str) -> bool:
    return cat in CANONICAL


def p4_no_punct(full: str, s: int, e: int) -> bool:
    if not (0 <= s < e <= len(full)):
        return False
    return not any(ch in PUNCT_ALL for ch in full[s:e])


def label_of(full: str, text: str, s: int, e: int, cat: str,
             gold_set: set[tuple[int, int, str]]) -> tuple[int, dict]:
    preds = {
        "P2": p2_traceable(text),
        "P3": p3_canonical(cat),
        "P4": p4_no_punct(full, s, e),
        "P5": (s, e, cat) in gold_set,
    }
    return int(all(preds.values())), preds


# ---------------------------------------------------------------------------
# 第一步：候选生成
# ---------------------------------------------------------------------------
def _other_card_cat(cat: str, rng: random.Random) -> str | None:
    others = [c for c in sorted(CANONICAL) if CARD_OF_CAT.get(c) != CARD_OF_CAT.get(cat)]
    return rng.choice(others) if others else None


def _other_same_card_cat(cat: str, rng: random.Random) -> str | None:
    card = CARD_OF_CAT.get(cat)
    same = [c for c in sorted(CANONICAL) if CARD_OF_CAT.get(c) == card and c != cat]
    return rng.choice(same) if same else None


def variant(kind: str, sent: str, gold: list[tuple[int, int, str]],
            rng: random.Random) -> tuple[int, int, str, str] | None:
    """(s, e, cat, text)；该 kind 在此句不可用 ⇒ None。

    区间一律**只向前扩或向后收**，绝不向后退 —— 后退会踩进上一分句的标点，
    导致重整后区间无法逐位映射（PREREG §1.3 内容上锁的前置条件）。
    """
    if not gold:
        return None
    if kind in ("crosscat", "samecat", "alias"):
        order = list(gold)
        rng.shuffle(order)
        for (s, e, cat) in order:
            if kind == "crosscat":
                c2 = _other_card_cat(cat, rng)
            elif kind == "samecat":
                c2 = _other_same_card_cat(cat, rng)
            else:
                opts = NONCANON.get(CARD_OF_CAT.get(cat, ""), [])
                c2 = rng.choice(opts) if opts else None
            if c2 and c2 != cat:
                return s, e, c2, sent[s:e]
        return None
    if kind == "boundary":
        order = list(gold)
        rng.shuffle(order)
        for (s, e, cat) in order:
            for (s2, e2) in ((s, e + 1), (s + 1, e), (s + 1, e + 1)):
                if 0 <= s2 < e2 <= len(sent) and sent[s2:e2] != sent[s:e]:
                    return s2, e2, cat, sent[s2:e2]
        return None
    if kind == "punct_edge":
        order = list(gold)
        rng.shuffle(order)
        for (s, e, cat) in order:
            nxt = []
            if e < len(sent) and sent[e] in PUNCT_ALL:
                nxt.append((s, e + 1))
            if e + 1 < len(sent) and sent[e + 1] in PUNCT_ALL:
                nxt.append((s, e + 2))
            rng.shuffle(nxt)
            for (s2, e2) in nxt:
                if 0 <= s2 < e2 <= len(sent):
                    return s2, e2, cat, sent[s2:e2]
        return None
    raise AssertionError(f"未知 variant kind {kind}")


def sentence_candidates(sent: str, gold: list[tuple[int, int, str]],
                        neural: list[tuple[int, int, str]],
                        kinds: list[str], rng: random.Random) -> list[dict]:
    gold_set = set(gold)
    neural_set = set(neural)
    cands: list[dict] = []
    seen: set[tuple[int, int, str]] = set()

    def add(src: str, kind: str, s: int, e: int, cat: str, text: str) -> None:
        key = (s, e, cat)
        if key in seen or not text:
            return
        seen.add(key)
        cands.append({"src": src, "kind": kind, "s": s, "e": e, "cat": cat, "text": text})

    for (s, e, cat) in gold:
        add("neural_card" if (s, e, cat) in neural_set else "rule_card",
            "gold", s, e, cat, sent[s:e])
    for (s, e, cat) in neural:                      # ≤1 条未命中真值源的神经提议
        if (s, e, cat) in gold_set:
            continue
        if 0 <= s < len(sent):
            add("neural_card", "neural_err", s, e, cat, sent[s:min(e, len(sent))])
        break
    order = kinds[:]
    rng.shuffle(order)
    for k in order[:len(kinds)]:
        v = variant(k, sent, gold, rng)
        if v is not None:
            add("variant", k, v[0], v[1], v[2], v[3])
    return cands


# ---------------------------------------------------------------------------
# 第三步：重整（只做非内容编辑；内容词多重集上锁）
# ---------------------------------------------------------------------------
def _units(sent: str) -> list[tuple[str, str, int]]:
    """[(seg, sep, orig_start), ...]；sep 是紧跟该 seg 的标点串（最后一段可为空）。"""
    parts = CLAUSE_SPLIT.split(sent)
    units, pos = [], 0
    for i in range(0, len(parts), 2):
        seg = parts[i]
        sep = parts[i + 1] if i + 1 < len(parts) else ""
        units.append((seg, sep, pos))
        pos += len(seg) + len(sep)
    return units


def _content_multiset(s: str) -> Counter:
    return Counter(ch for ch in s if ch not in PUNCT_ALL)


def _apply_ops(sent: str, units: list[tuple[str, str, int]], ops: list[str],
               rng: random.Random):
    """返回 (完整句, 插入的连词列表, offset 映射 fn)；本方案不可行 ⇒ None。"""
    order = list(range(len(units)))
    segs = [u[0] for u in units]
    seps = [u[1] for u in units]
    starts = [u[2] for u in units]
    lead_of: dict[int, int] = {}
    if "reorder" in ops and len(order) >= 2:
        order[0], order[1] = order[1], order[0]
    if "conj" in ops:
        cands = [w for w in CONJS if all(ch not in sent for ch in w)]
        if not cands:
            return None
        w = rng.choice(cands)
        b = rng.randrange(1, len(order)) if len(order) >= 2 else 0
        pos = order[b]
        segs[pos] = w + segs[pos]
        lead_of[pos] = len(w)
    if "punct" in ops:
        last = order[-1]
        if seps[last] not in TERMINALS:
            seps[last] = "。"
    inserted = [list(lead_of.values())[i] for i in range(len(lead_of))] if lead_of else []
    inserted = [segs[pos][:lead_of[pos]] for pos in lead_of]

    out_parts: list[str] = []
    unit_map: list[tuple[int, int, int, int]] = []   # (u_start, u_end, new_seg_start, seg_len)
    cur = 0
    for pos in order:
        seg, sep = segs[pos], seps[pos]
        out_parts.append(seg)
        out_parts.append(sep)
        u_start = starts[pos]
        u_len = len(units[pos][0]) + len(units[pos][1])
        unit_map.append((u_start, u_start + u_len, cur + lead_of.get(pos, 0), len(units[pos][0])))
        cur += len(seg) + len(sep)
    out = "".join(out_parts)

    def mapping(o: int) -> int:
        for (a, b, n, glen) in unit_map:
            if a <= o <= b:
                if o <= a + glen:
                    return n + (o - a)
                return n + glen + (o - (a + glen))
        best = None
        for (a, b, n, glen) in unit_map:
            d = min(abs(o - a), abs(o - b))
            cand = n + (0 if o < a else glen + (b - (a + glen)))
            if best is None or d < best[0]:
                best = (d, cand)
        return best[1] if best else o

    return out, inserted, mapping


def restructure(sent: str, rng: random.Random) -> tuple[str, list[str], object]:
    """每句一次（同句所有候选共享同一个完整句）。任一方案都过不了校验 ⇒ 抛错。"""
    units = _units(sent)
    conj_ok = any(all(ch not in sent for ch in w) for w in CONJS)
    plans: list[list[str]] = []
    for mask in range(1, 8):
        ops = [op for i, op in enumerate(("reorder", "conj", "punct")) if mask >> i & 1]
        if "reorder" in ops and len(units) < 2:
            continue
        if "conj" in ops and not conj_ok:
            continue
        plans.append(ops)
    check(bool(plans), f"重整无可用方案：{sent!r}")
    rng.shuffle(plans)
    plans.sort(key=len, reverse=True)              # 强编辑优先，弱编辑兜底；同长度保持洗牌序
    last_err = ""
    gold_src = gold_of(sent)
    for ops in plans:
        built = _apply_ops(sent, units, ops, rng)
        if built is None:
            continue
        out, inserted, mapping = built
        if not (Counter(ch for ch in "".join(_strip_ins(out, inserted)) if ch not in PUNCT_ALL)
                == _content_multiset(sent)):
            last_err = "内容多重集不等"
            continue
        exp = sorted((mapping(s), mapping(e), c) for (s, e, c) in gold_src)
        got = sorted(gold_of(out))
        if got != exp:
            last_err = f"真值源不保序（期望 {len(exp)} 条，实得 {len(got)} 条）"
            continue
        return out, sorted(ops), mapping
    raise AssertionError(f"[gen_data_loop fail-closed] 重整失败：{sent!r}（{last_err}）")


def _strip_ins(out: str, inserted: list[str]) -> str:
    tmp = out
    for w in inserted:
        i = tmp.find(w)
        if i < 0:
            return ""
        tmp = tmp[:i] + tmp[i + len(w):]
    return tmp


# ---------------------------------------------------------------------------
# 行渲染
# ---------------------------------------------------------------------------
def render(full: str, text: str, cat: str) -> str:
    return f"{PREFIX}{full}{SEP}{SUFFIX}{text}{SEP}{CATP}{cat}"


def cand_field_pos(full: str, text: str) -> tuple[int, int]:
    s = len(PREFIX) + len(full) + len(SEP) + len(SUFFIX)
    return s, s + len(text)


def make_row(split: str, idx: int, sent: str, full: str, c: dict, s: int, e: int,
             label: int, preds: dict, ops: list[str]) -> dict:
    text = render(full, c["text"], c["cat"])
    fs, fe = cand_field_pos(full, c["text"])
    return {
        "id": f"{split}-{idx:05d}", "split": split, "label": label,
        "source": c["src"], "kind": c["kind"], "preds": preds,
        "input": sent, "full": full, "candidate": c["text"], "category": c["cat"],
        "gold_span": [s, e], "restructure_ops": ops,
        "text": text, "spans": ([{"label": 1, "start": fs, "end": fe}] if label else []),
        "lex_match": True,
    }


# ---------------------------------------------------------------------------
# 提案臂：一个池 → 提案 → P1 准入 → P2–P5 打标 → 配额下采样
# ---------------------------------------------------------------------------
def build_pool(name: str, pool: list[str], neural_all: list[list[tuple[int, int, str]]],
               n_pos: int, n_neg: int, quota: dict[str, float],
               kinds: list[str], rng: random.Random) -> tuple[list[dict], dict]:
    admitted: list[dict] = []
    n_proposed = n_p1 = 0
    n_p1_drop = n_relocate_drop = 0
    for si, sent in enumerate(pool):
        gold = gold_of(sent)
        if not gold:
            continue                       # 无真分析的句子不产候选（防句子级捷径）
        neural = neural_all[si]
        cands = sentence_candidates(sent, gold, neural, kinds, rng)
        if not cands:
            continue
        full, ops, mapping = restructure(sent, rng)
        gold_full = set(gold_of(full))
        for c in cands:
            n_proposed += 1
            if not p1_anchor(c["text"], sent, c["s"], c["e"]):
                n_p1_drop += 1             # P1 准入①：候选必须锚定在**原始输入**上
                continue
            ns, ne = mapping(c["s"]), mapping(c["e"])
            if ne < ns:
                ns, ne = ne, ns
            if not p1_anchor(c["text"], full, ns, ne):
                n_relocate_drop += 1       # P1 准入②：必须也锚定在模型可见的完整句上
                continue
            n_p1 += 1
            lab, preds = label_of(full, c["text"], ns, ne, c["cat"], gold_full)
            admitted.append(make_row(name, 0, sent, full, c, ns, ne, lab, preds, ops))

    pos = [r for r in admitted if r["label"] == 1]
    neg = [r for r in admitted if r["label"] == 0]
    check(len(pos) >= n_pos, f"{name} 正例不足：{len(pos)} < {n_pos}")
    rng.shuffle(pos)
    rng.shuffle(neg)
    pos_rows = pos[:n_pos]

    by_kind: dict[str, list[dict]] = {}
    for r in neg:
        by_kind.setdefault(r["kind"], []).append(r)
    neg_rows: list[dict] = []
    quota_sum = 0
    for kind, q in quota.items():
        want = int(round(q * n_neg))
        quota_sum += want
        have = by_kind.get(kind, [])
        check(len(have) >= want,
              f"{name} 负例分层 {kind} 不足：可得 {len(have)} < 需要 {want}")
        rng.shuffle(have)
        neg_rows.extend(have[:want])
    check(quota_sum == n_neg, f"{name} 负例配额之和 {quota_sum} != {n_neg}")
    check(len(neg_rows) == n_neg, f"{name} 负例 {len(neg_rows)} != {n_neg}")

    rows = pos_rows + neg_rows
    rng.shuffle(rows)
    for i, r in enumerate(rows):
        r["id"] = f"{name}-{i:05d}"
    return rows, {
        "proposed": n_proposed,
        "p1_dropped_not_anchored": n_p1_drop,
        "p1_dropped_not_in_full": n_relocate_drop,
        "p1_admitted": n_p1,
        "p1_admit_rate": round(n_p1 / max(1, n_proposed), 4),
        "pool_pos": len(pos), "pool_neg": len(neg),
        "accept_rate": round(len(pos) / max(1, len(pos) + len(neg)), 4),
        "neg_kept_by_kind": dict(Counter(r["kind"] for r in neg_rows)),
        "neg_pool_by_kind": {k: len(v) for k, v in sorted(by_kind.items())},
        "source_dist_pool": dict(Counter(r["source"] for r in admitted)),
    }


# ---------------------------------------------------------------------------
# 注入对照臂（复刻 anchored_select 构造与标签口径）
# ---------------------------------------------------------------------------
PUNCT_TAIL = set("，。！？、；：\"'”’）)】》…~—-,.?!:;")
PUNCT_HEAD = set("，。！？、；：\"'“‘（(【《…~—-,.?!:;")
ANT = [("提高", "降低"), ("增加", "减少"), ("上升", "下降"), ("变好", "变坏"),
       ("高兴", "难过"), ("美丽", "丑陋"), ("快速", "缓慢"), ("简单", "复杂"),
       ("安全", "危险"), ("容易", "困难"), ("安静", "吵闹"), ("炎热", "寒冷"),
       ("支持", "反对"), ("同意", "拒绝"), ("接受", "放弃"), ("记得", "忘记"),
       ("表扬", "批评"), ("成功", "失败"), ("前进", "后退"), ("开始", "结束"),
       ("出现", "消失"), ("表扬", "责备"), ("喜欢", "讨厌"), ("勇敢", "胆怯"),
       ("勤奋", "懒惰"), ("大方", "小气"), ("耐心", "急躁"), ("表扬", "赞赏"),
       ("表扬", "夸奖"), ("认可", "否认"), ("开心", "郁闷"), ("满意", "不满")]
SYN = [("看见", "看到"), ("立刻", "马上"), ("高兴", "开心"), ("漂亮", "好看"),
       ("快速", "迅速"), ("帮助", "帮忙"), ("讨论", "商量"), ("拒绝", "回绝"),
       ("开始", "启动"), ("担心", "担忧"), ("安静", "宁静"), ("简单", "容易"),
       ("重要", "关键"), ("温暖", "暖和"), ("觉得", "认为"), ("知道", "了解"),
       ("选择", "挑选"), ("准备", "打算"), ("经常", "常常"), ("特别", "尤其"),
       ("大概", "大约"), ("好像", "仿佛"), ("反正", "总之"), ("干脆", "索性"),
       ("其实", "实际上"), ("马上", "立即"), ("有点", "稍微"), ("一直", "总是")]
WORD_BANK = ["今天", "明天", "昨天", "时间", "问题", "工作", "学习", "生活", "家里",
             "朋友", "孩子", "老师", "同学", "吃饭", "出门", "回来", "觉得", "知道",
             "应该", "可能", "需要", "帮忙", "东西", "地方", "时候", "办法", "方法",
             "结果", "开始", "感觉", "认为", "非常", "特别", "一定", "一起", "现在",
             "还是", "但是", "因为", "所以", "如果", "虽然", "然后", "而且", "已经",
             "正在", "准备", "安排", "注意", "检查", "收拾", "打扫", "休息", "睡觉",
             "周末", "早上", "晚上", "下午", "路上", "房间", "厨房", "公司", "学校",
             "医院", "商场", "公园", "车站", "天气", "下雨", "放假", "加班", "开会",
             "情况", "消息", "消息", "内容", "格式", "进度", "安排", "方案", "文档",
             "记录", "参数", "配置", "环境", "测试", "发布", "上线", "回滚", "复盘"]


def pick_fragment(text: str, rng: random.Random, lo: int = 8, hi: int = 18) -> tuple[int, int] | None:
    if len(text) < lo + 1:
        return None
    for _ in range(60):
        s = rng.randrange(0, len(text) - lo + 1)
        e = s + rng.randint(lo, min(hi, len(text) - s))
        if e - s < lo:
            continue
        f = text[s:e]
        if f[0] in PUNCT_HEAD or f[-1] in PUNCT_TAIL or SEP in f:
            continue
        if cjk(f) < lo - 2 or f == text:
            continue
        return s, e
    return None


def perturb(frag: str, full: str, rule: str, rng: random.Random) -> tuple[str, str] | None:
    """返回 (新文本, 实际用到的 rule)；都不成立 ⇒ None。"""
    order = [rule] + [r for r in ("word_swap", "antonym", "unrelated", "reorder")
                      if r != rule]
    for r in order:
        out = _perturb_once(frag, full, r, rng)
        if out is not None and out not in full and out != frag:
            return out, r
    return None


def _perturb_once(frag: str, full: str, rule: str, rng: random.Random) -> str | None:
    if rule == "word_swap":
        for p in range(0, max(1, len(frag) - 1)):
            cands = [w for w in WORD_BANK if w not in full and w not in frag]
            if not cands:
                continue
            out = frag[:p] + rng.choice(cands) + frag[p + 2:]
            if out != frag and cjk(out) >= 4:
                return out
        return None
    if rule == "antonym":
        for p in range(0, max(1, len(frag) - 1)):
            win = frag[p:p + 2]
            for (a, b) in ANT:
                if win == a:
                    return frag[:p] + b + frag[p + 2:]
        for (a, b) in ANT:
            for (x, y) in ((a, b), (b, a)):
                if x in frag:
                    return frag.replace(x, y, 1)
        return None
    if rule == "unrelated":
        fp = pick_fragment(full, rng)
        if fp is None:
            return None
        return full[fp[0]:fp[1]]
    if rule == "reorder":
        if len(frag) < 10:
            return None
        marks = [i for i, ch in enumerate(frag) if ch in "，、；："]
        if marks:
            a = marks[0]
            chunks = [frag[:a + 1], frag[a + 1:]]
        else:
            a = len(frag) // 2
            chunks = [frag[:a], frag[a:]]
        chunks = [c for c in chunks if c]
        if len(chunks) < 2:
            return None
        out = chunks[1] + chunks[0]
        return out if sorted(out) == sorted(frag) else None
    return None


def build_ctrl(name: str, sent_iter, cats: list[str], rng: random.Random,
               adv: bool, target: int) -> list[dict]:
    """注入对照臂：每句 1 正（逐字片段 / 对抗集=同义改写）+ 1 负（注入扰动）。"""
    rows: list[dict] = []
    neg_rules = ["reorder", "unrelated"] if adv else ["word_swap", "antonym", "unrelated"]
    pos_rule = "paraphrase" if adv else "verbatim"
    n_target = target // 2
    cat_ptr = 0
    guard = 0
    while len(rows) < target:
        guard += 1
        if guard > target * 40:
            break
        sent = next(sent_iter, None)
        if sent is None:
            break
        full, ops, _ = restructure(sent, rng)
        fp = pick_fragment(full, rng)
        if fp is None:
            continue
        fs, fe = fp
        frag = full[fs:fe]
        if adv:
            got = perturb(frag, full, "paraphrase", rng)
            if got is None:
                continue
            ptxt, pos_actual = got, "paraphrase"
            if ptxt in full:                       # 对抗集正例 = 改写，逐字片段不算
                continue
        else:
            ptxt, pos_actual = frag, "verbatim"
        neg_rule = neg_rules[len(rows) // 2 % len(neg_rules)]
        got = perturb(frag, full, neg_rule, rng)
        if got is None:
            continue
        ntxt, neg_actual = got
        pos_ok = (ptxt in full) if not adv else (ptxt not in full)
        neg_ok = ntxt not in full
        if not pos_ok or not neg_ok:
            continue
        for (txt, lab, rul) in ((ptxt, 1, pos_actual), (ntxt, 0, neg_actual)):
            ps, pe = cand_field_pos(full, txt)
            rows.append({
                "id": f"{name}-{len(rows):05d}", "split": name, "label": lab,
                "source": "injected", "kind": rul, "preds": {},
                "input": sent, "full": full, "candidate": txt,
                "category": cats[cat_ptr % len(cats)], "cat_ptr": cat_ptr,
                "gold_span": [fs, fe], "restructure_ops": ops,
                "text": render(full, txt, cats[cat_ptr % len(cats)]),
                "spans": ([{"label": 1, "start": ps, "end": pe}] if lab else []),
                "lex_match": txt in full,
            })
            cat_ptr += 1
    check(len(rows) == target,
          f"ctrl/{name} 只造出 {len(rows)} 行，目标 {target}（源句池不够或 perturb 失败过多）")
    for i, r in enumerate(rows):
        r["id"] = f"{name}-{i:05d}"
    return rows


# ---------------------------------------------------------------------------
# 自然 held-out 集（手写，不经本管线）
# ---------------------------------------------------------------------------
def load_natural(path: Path) -> list[dict]:
    rows: list[dict] = []
    for ln, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        check(len(parts) >= 4, f"natural 第 {ln} 行字段不足：{line!r}")
        lab, sent, txt, cat = int(parts[0]), parts[1].strip(), parts[2].strip(), parts[3].strip()
        check(lab in (0, 1), f"natural 第 {ln} 行 label 非法")
        check(txt in sent, f"natural 第 {ln} 行候选不锚定：{line!r}")
        check(sent.count(txt) == 1, f"natural 第 {ln} 行候选在句中不唯一：{line!r}")
        check(cat in CANONICAL, f"natural 第 {ln} 行类别非规范名：{cat!r}")
        s = sent.index(txt)
        truth_hit = int((s, s + len(txt), cat) in set(gold_of(sent)))
        rows.append({
            "id": f"natural-{len(rows):04d}", "split": "natural", "label": lab,
            "source": "hand", "kind": "hand", "preds": {},
            "input": sent, "full": sent, "candidate": txt, "category": cat,
            "gold_span": [s, s + len(txt)], "restructure_ops": [],
            "text": render(sent, txt, cat),
            "spans": ([{"label": 1, "start": len(PREFIX) + len(sent) + len(SEP) + len(SUFFIX),
                        "end": len(PREFIX) + len(sent) + len(SEP) + len(SUFFIX) + len(txt)}]
                      if lab else []),
            "lex_match": True, "truth_agree": int(truth_hit == lab),
        })
    return rows


# ---------------------------------------------------------------------------
# 统计
# ---------------------------------------------------------------------------
def naive_battery(rows: list[dict]) -> dict:
    """零训练的表层规则；每条取最优方向。**不含**真值源重实现（单列，见 stats）。"""
    n = len(rows)
    check(n > 0, "naive_battery 收到空集")

    def acc(pred: list[int]) -> float:
        return sum(int(p == r["label"]) for p, r in zip(pred, rows)) / n

    def pos_of(r: dict) -> tuple[int, int]:
        i = r["full"].find(r["candidate"])
        return (i if i >= 0 else len(r["full"])), len(r["full"])

    rules = {
        "substring": [int(r["candidate"] in r["full"]) for r in rows],
        "cat_canonical": [int(r["category"] in CANONICAL) for r in rows],
        "lex_member": [int(r["candidate"] in LEX_ALL) for r in rows],
        "no_punct": [int(not any(ch in PUNCT_ALL for ch in r["candidate"])) for r in rows],
        "len_le_3": [int(len(r["candidate"]) <= 3) for r in rows],
        "cjk_ge_90": [int(cjk(r["candidate"]) >= 0.9 * max(1, len(r["candidate"]))) for r in rows],
        "neg_char": [int(any(w in r["candidate"] for w in NEG_MARKERS)) for r in rows],
        "early_pos": [int(pos_of(r)[0] <= pos_of(r)[1] / 2) for r in rows],
    }
    out: dict = {}
    best_name, best_val = "", 0.0
    for k, v in rules.items():
        a = acc(v)
        m = max(a, 1 - a)
        out[k] = {"acc": round(a, 4), "best_dir": round(m, 4)}
        if m > best_val:
            best_name, best_val = k, m
    out["max_naive"] = {"rule": best_name, "best_dir": round(best_val, 4)}
    return out


def style_stats(rows: list[dict]) -> dict:
    lens = sorted(len(r["input"]) for r in rows)
    cand = [len(r["candidate"]) for r in rows]
    n = len(rows)
    chars = set("".join(r["input"] for r in rows))
    return {
        "n": n,
        "input_len_mean": round(sum(lens) / n, 2),
        "input_len_median": lens[n // 2],
        "input_len_p10": lens[max(0, n // 10)],
        "input_len_p90": lens[min(n - 1, 9 * n // 10)],
        "cand_len_mean": round(sum(cand) / n, 2),
        "rate_has_comma": round(sum("，" in r["input"] for r in rows) / n, 4),
        "rate_end_terminal": round(sum(r["input"][-1] in TERMINALS for r in rows) / n, 4),
        "rate_restructured": round(sum(bool(r.get("restructure_ops")) for r in rows) / n, 4),
        "rate_lex_match": round(sum(bool(r["lex_match"]) for r in rows) / n, 4),
        "distinct_chars": len(chars),
        "top_candidate_chars": dict(Counter(
            ch for r in rows for ch in r["candidate"]).most_common(8)),
    }


def diversity(rows: list[dict]) -> dict:
    n = len(rows)
    cats = Counter(r["category"] for r in rows)
    h = -sum((v / n) * math.log(v / n) for v in cats.values())
    return {
        "n": n,
        "category_entropy_nats": round(h, 4),
        "n_categories": len(cats),
        "distinct_candidate": len({r["candidate"] for r in rows}),
        "distinct_candidate_rate": round(len({r["candidate"] for r in rows}) / n, 4),
        "distinct_input": len({r["input"] for r in rows}),
        "distinct_input_rate": round(len({r["input"] for r in rows}) / n, 4),
        "rows_per_input": round(n / max(1, len({r["input"] for r in rows})), 3),
        "category_dist": dict(cats.most_common()),
    }


def overlap(a: list[dict], b: list[dict]) -> dict:
    return {
        "text": len({r["text"] for r in a} & {r["text"] for r in b}),
        "input": len({r["input"] for r in a} & {r["input"] for r in b}),
        "full": len({r["full"] for r in a} & {r["full"] for r in b}),
        "candidate": len({r["candidate"] for r in a} & {r["candidate"] for r in b}),
    }


def report(name: str, rows: list[dict], require_anchor: bool = True) -> dict:
    n = len(rows)
    check(n > 0, f"{name} 空")
    dist = Counter(r["label"] for r in rows)
    texts = [r["text"] for r in rows]
    check(len(set(texts)) == n, f"{name} 行文本重复 {n - len(set(texts))} 条")
    for r in rows:
        check(bool(r["spans"]) == bool(r["label"]), f"标签与 span 不一致：{r['id']}")
        if r["label"]:
            sp = r["spans"][0]
            check(r["text"][sp["start"]:sp["end"]] == r["candidate"], f"span 没指向候选：{r['id']}")
        if require_anchor:
            check(r["candidate"] in r["full"], f"候选不锚定在完整句上：{r['id']}")
    return {
        "n": n,
        "label_dist": {str(k): v for k, v in sorted(dist.items())},
        "majority_baseline": round(max(dist.values()) / n, 4),
        "source_dist": dict(Counter(r["source"] for r in rows)),
        "kind_dist": dict(Counter(r["kind"] for r in rows)),
        "pred_fail": {p: sum(1 for r in rows if not r["preds"].get(p, True))
                      for p in ("P2", "P3", "P4", "P5")},
        "naive": naive_battery(rows),
        "style": style_stats(rows),
        "diversity": diversity(rows),
        **({"truth_agree": round(sum(r["truth_agree"] for r in rows) / n, 4)}
           if name == "natural" else {}),
    }


def no_truncation(rows: list[dict], max_len: int) -> int:
    tok = NanoCharTokenizer()
    bad = 0
    for r in rows:
        ids = tok.encode(r["text"], max_length=4096, padding=False)["input_ids"]
        if len(ids) > max_len:
            bad += 1
    return bad


# ---------------------------------------------------------------------------
def main() -> None:
    rng = random.Random(SPLIT_SEED)
    DATA.mkdir(parents=True, exist_ok=True)

    print("[1/6] 取语料句子（resolve_corpus_files，fail-closed）...", flush=True)
    total = sum(POOL_N.values()) + sum(CTRL_POOL_N.values()) + OVERFLOW_N
    sents, files = collect_sentences(total)
    print(f"  {len(sents)} 句 / {len(files)} 个文件", flush=True)
    shuffled = list(sents)
    rng.shuffle(shuffled)
    pools: dict[str, list[str]] = {}
    at = 0
    for k in ("train", "test", "adv"):
        pools[k] = shuffled[at:at + POOL_N[k]]
        at += POOL_N[k]
    ctrl_pools: dict[str, list[str]] = {}
    for k in ("train", "test", "adv"):
        ctrl_pools[k] = shuffled[at:at + CTRL_POOL_N[k]]
        at += CTRL_POOL_N[k]
    overflow = shuffled[at:]
    check(len(overflow) == OVERFLOW_N, f"溢出池 {len(overflow)} != {OVERFLOW_N}")
    names = list(pools) + [f"ctrl_{k}" for k in ctrl_pools] + ["overflow"]
    all_pools = {**pools, **{f"ctrl_{k}": v for k, v in ctrl_pools.items()}, "overflow": overflow}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = len(set(all_pools[a]) & set(all_pools[b]))
            check(ov == 0, f"源句池 {a}∩{b} 重叠 {ov} 条")
    for k, v in pools.items():
        check(len(v) == POOL_N[k], f"池 {k} 大小 {len(v)}")
    print(f"  四类池两两重叠 = 0（溢出池 {len(overflow)} 句备补）", flush=True)

    print("[2/6] 神经指针卡提议（CPU，不占 GPU）...", flush=True)
    neural: dict[str, list] = {}
    for k in ("train", "test", "adv"):
        neural[k] = run_neural(pools[k])
        print(f"  {k}: {sum(len(v) for v in neural[k])} 条神经发射 / {len(pools[k])} 句", flush=True)

    print("[3/6] 提案臂三集（候选 → P1 准入 → P2–P5 打标 → 重整）...", flush=True)
    kinds_std = ["crosscat", "boundary", "alias", "punct_edge"]
    metas: dict[str, dict] = {}
    prop: dict[str, list[dict]] = {}
    prop["train"], metas["train"] = build_pool(
        "train", pools["train"], neural["train"], N_ROWS["train"] // 2,
        N_ROWS["train"] // 2, NEG_QUOTA, kinds_std, rng)
    prop["test"], metas["test"] = build_pool(
        "test", pools["test"], neural["test"], N_ROWS["test"] // 2,
        N_ROWS["test"] // 2, NEG_QUOTA, kinds_std, rng)
    prop["adversarial"], metas["adversarial"] = build_pool(
        "adversarial", pools["adv"], neural["adv"], N_ROWS["adv"] // 2,
        N_ROWS["adv"] // 2, ADV_QUOTA, ["crosscat", "samecat"], rng)
    for k, v in prop.items():
        m = metas[k]
        print(f"  {k}: {len(v)} 行 | 提案 {m['proposed']} → P1 准入 {m['p1_admitted']}"
              f"（丢弃 {m['p1_dropped_not_anchored'] + m['p1_dropped_not_in_full']}）"
              f" | 接受率 {m['accept_rate']}", flush=True)

    print("[4/6] 注入对照臂（复刻 anchored_select 构造）...", flush=True)
    cats = {k: [r["category"] for r in prop[k]] for k in ("train", "test", "adversarial")}

    def mk_iter(pool: list[str], rest: list[str]):
        it = iter(pool + rest)
        return it

    ctrl: dict[str, list[dict]] = {}
    used_ctrl: dict[str, list[str]] = {}
    for k, key in (("train", "train"), ("test", "test"), ("adversarial", "adv")):
        pool = ctrl_pools[key]
        it = mk_iter(pool, overflow)
        rows = []
        # 记录实际用掉的句子（含溢出），仅用于统计
        it2 = iter(pool + overflow)
        target = N_ROWS[key]
        ctrl[k] = build_ctrl(k, it2, cats[k], rng, adv=(key == "adv"), target=target)
        used_ctrl[k] = []
        # 重放一遍拿实际用到的句子数：从 build_ctrl 内部不好取，改用行里的 input 去重
        used_ctrl[k] = sorted({r["input"] for r in ctrl[k]})
        print(f"  ctrl/{k}: {len(ctrl[k])} 行 | "
              f"正负 {dict(Counter(r['label'] for r in ctrl[k]))} | "
              f"用句 {len(used_ctrl[k])} | 朴素规则 "
              f"{naive_battery(ctrl[k])['max_naive']}", flush=True)

    print("[5/6] 自然 held-out 集 + 守卫断言...", flush=True)
    natural = load_natural(HERE / "natural_hand.txt")
    print(f"  natural: {len(natural)} 行，正负 "
          f"{dict(Counter(r['label'] for r in natural))}，与真值源一致率 "
          f"{sum(r['truth_agree'] for r in natural) / len(natural):.4f}", flush=True)

    all_sets = {**prop, **{f"ctrl_{k}": v for k, v in ctrl.items()}, "natural": natural}
    for k, rows in all_sets.items():
        bad = no_truncation(rows, MAX_LEN)
        check(bad == 0, f"{k} 有 {bad} 条编码长度 > max_len={MAX_LEN}（会被静默截断）")

    print("[6/6] 不相交实测 + 统计 ...", flush=True)
    split_names = ["train", "test", "adversarial"]
    pairs: dict[str, dict] = {}
    for i, a in enumerate(split_names):
        for b in split_names[i + 1:]:
            for tag, src in (("prop", prop), ("ctrl", ctrl)):
                k = f"{tag}:{a}∩{b}"
                pairs[k] = overlap(src[a], src[b])
                check(pairs[k]["text"] == 0 and pairs[k]["input"] == 0 and pairs[k]["full"] == 0,
                      f"{k} 泄漏：{pairs[k]}")
            k = f"cross:{a}∩ctrl_{b}"
            pairs[k] = overlap(prop[a], ctrl[b])
            check(pairs[k]["text"] == 0 and pairs[k]["input"] == 0 and pairs[k]["full"] == 0,
                  f"{k} 泄漏：{pairs[k]}")
    for k in split_names:
        ov = overlap(prop[k], natural)
        pairs[f"{k}∩natural"] = ov
        check(ov["text"] == 0 and ov["input"] == 0, f"{k}∩natural 泄漏：{ov}")
        ov = overlap(ctrl[k], natural)
        pairs[f"ctrl_{k}∩natural"] = ov
        check(ov["text"] == 0 and ov["input"] == 0, f"ctrl_{k}∩natural 泄漏：{ov}")
    for k, v in pairs.items():
        print(f"  {k}: text={v['text']} input={v['input']} full={v['full']} "
              f"candidate={v['candidate']}", flush=True)

    stats = {
        "split_seed": SPLIT_SEED,
        "corpus_files_used": files,
        "pools": {k: len(v) for k, v in all_pools.items()},
        "ctrl_sentences_used": {k: len(v) for k, v in used_ctrl.items()},
        "proposal": {k: report(k, prop[k]) for k in split_names},
        "control": {f"ctrl_{k}": report(f"ctrl_{k}", v, require_anchor=False)
                    for k, v in ctrl.items()},
        "natural": report("natural", natural),
        "pipeline_meta": metas,
        "overlap_check": pairs,
        "predicate_defs": {
            "P1": "锚定 = 0≤s<e≤len 且 input[s:e]==候选文本；**准入**（不通过即丢弃）",
            "P2": "可回溯：非空 / ≥1 汉字 / 不以标点起止 / 中文占比≥60%",
            "P3": "规范类别名 ∈ TaskSpec.classes[1:].name（spec.label / 类展示名不算）",
            "P4": "候选区间内不含标点（不跨分句）",
            "P5": "与真值源一致（手写指针发射器）",
        },
        "label_rule": "label = P2 ∧ P3 ∧ P4 ∧ P5（在通过 P1 准入的候选上）",
        "non_predicate_judgments": [
            "自然集 200 行的人工标签（手写，非谓词）",
            "各集的配额下采样与正负平衡（抽样决定，非谓词）",
            "注入对照臂的标签 = anchored_select 口径（规则但非本管线谓词）",
        ],
        "truth_source_note": "真值源重实现的准确率 = 1.0000（定义即标签，单列，不作 D3 判据）",
        "text_layout": f"{PREFIX}{{full}}{SEP}{SUFFIX}{{candidate}}{SEP}{CATP}{{category}}",
        "canon_categories": sorted(CANONICAL),
        "noncanon_names": NONCANON,
        "max_len": MAX_LEN,
    }
    for k in list(stats["proposal"]) + list(stats["control"]) + ["natural"]:
        blk = stats["proposal"].get(k) or stats["control"].get(k) or stats["natural"]
        check(blk["majority_baseline"] == 0.5, f"{k} 多数类基线 {blk['majority_baseline']} != 0.5")

    files_out = {
        "train": prop["train"], "test": prop["test"], "adversarial": prop["adversarial"],
        "ctrl_train": ctrl["train"], "ctrl_test": ctrl["test"],
        "ctrl_adversarial": ctrl["adversarial"], "natural": natural,
    }
    for name, rows in files_out.items():
        with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    (DATA / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                     encoding="utf-8")

    brief = {
        "proposal": {k: {"n": stats["proposal"][k]["n"],
                         "majority": stats["proposal"][k]["majority_baseline"],
                         "max_naive": stats["proposal"][k]["naive"]["max_naive"],
                         "H": stats["proposal"][k]["diversity"]["category_entropy_nats"],
                         "distinct_cand_rate": stats["proposal"][k]["diversity"]["distinct_candidate_rate"],
                         "pred_fail": stats["proposal"][k]["pred_fail"],
                         "style": stats["proposal"][k]["style"]}
                     for k in stats["proposal"]},
        "control": {k: {"max_naive": stats["control"][k]["naive"]["max_naive"],
                        "H": stats["control"][k]["diversity"]["category_entropy_nats"]}
                    for k in stats["control"]},
        "natural": {kk: stats["natural"][kk]
                    for kk in ("n", "majority_baseline", "truth_agree", "naive", "style", "diversity")},
        "pipeline_meta": metas,
        "overlap_check": {k: v for k, v in pairs.items() if v["candidate"] or k.endswith("natural")},
    }
    print(json.dumps(brief, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
