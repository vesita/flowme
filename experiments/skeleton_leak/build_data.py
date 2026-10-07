#!/usr/bin/env python3
"""skeleton_leak 数据构建：不受 `n_slots` 泄露污染的评测集（A/B/C 三型）。

口径 = `PREREG.md` §1/§2（跑前写死）：
  · 语料入口只用 `dtseek.tasks.corpus.resolve_corpus_files`；文件级汉字比 ≥0.6 保留 14/18、
    打乱文件序（SPLIT_SEED），不依赖 fs 顺序、不写 fs[:N]；
  · 匹配器/过滤器 **只读 import** `two_channel_head/build_gen_data`（与 train/test 逐字同）；
  · 骨架池 = train 出现过的 30 个 id；句子与 train/test/adv1/adv2 互斥、四子集互斥；
  · A 型主集 a_bal：各 (n_slots, skeleton) 组合等量 + **排除泄露骨架 L={每桶 train 多数}**；
    对照 a_lit = a_bal ∪ {L 四组合各 C 行}（只作构造诊断，不进 Q1/Q2）；
  · B 型 b_pairs：同袋同 n_slots 的合成最小对（B1 功能词替换 / B2 因果连词位置交换+槽序交换），
    matcher fail-closed 验证；C 型 c_pairs：孪生骨架对自然句等量采样。

用法：uv run python experiments/skeleton_leak/build_data.py
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "two_channel_head"))

from dtseek.tasks.corpus import CORPUS_GLOB, resolve_corpus_files  # noqa: E402
from build_gen_data import (  # noqa: E402
    ALLOWED_LIT_CHARS, BAG_SEED, GAP_LEN, MIN_CJK_RATIO, PUNCT_NORM, ROLE_PREFIX,
    SENT_LEN, SENT_SPLIT, SPLIT_SEED, SKELS, TRAIL_WEAK, match_sentence, ok_sentence,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOTEX = ROOT / "experiments"
TCH_DATA = ROOTEX / "two_channel_head" / "data"
SS_DATA = ROOTEX / "struct_supervision" / "data"

CAP_RES = 800        # 每骨架蓄水池上限（报告说明）
MIN_COMBO = 10       # 单组合可用 < 10 ⇒ 判不可构造并剔除
MAX_COMBO = 40       # 单组合取样上限
MIN_PAIR_SIDE = 10   # 孪生对单侧可用 < 10 ⇒ 该对不可构造
MAX_PAIR_SIDE = 40
MAX_B_DIR = 20       # B 型每个（对, 方向）最多取源句数

# PREREG §2.1：孪生骨架对（跑前写死，不事后增补）
C_PAIRS = [(6, 11), (6, 26), (11, 26), (7, 9), (7, 10), (9, 10),
           (14, 15), (17, 18), (19, 20), (35, 36), (36, 37), (35, 37), (38, 39)]
# B2：因果对连词位置交换（同袋、槽序相对连词交换 ⇒ 指派也变）
B2_PAIRS = [(7, 6), (6, 7)]


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[skeleton_leak build fail-closed] {msg}")


def h12(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def load_rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def lit_parts(tmpl: str) -> list[str]:
    return re.split(r"\[\d+\]", tmpl)


def twin_ok(a: int, b: int) -> tuple[bool, int, str]:
    """PREREG §2 判据：同槽位数、字面段数相同、恰一段不同且等长、只含形式词。"""
    pa, pb = lit_parts(SKELS[a].template), lit_parts(SKELS[b].template)
    if len(pa) != len(pb):
        return False, -1, "字面段数不同"
    diff = [i for i in range(len(pa)) if pa[i] != pb[i]]
    if len(diff) != 1:
        return False, -1, f"差异段数={len(diff)}（要求恰 1）"
    i = diff[0]
    if len(pa[i]) != len(pb[i]):
        return False, -1, "差异段不等长（增删而非换词）"
    if set(pa[i] + pb[i]) - ALLOWED_LIT_CHARS:
        return False, -1, "差异段含非形式词"
    return True, i, f"{pa[i]!r}↔{pb[i]!r}"


def render(tmpl: str, gaps: list[str]) -> str:
    parts = lit_parts(tmpl)
    out = parts[0]
    for g, p in zip(gaps, parts[1:]):
        out += g + p
    return out


# ---------------------------------------------------------------------------
# 语料扫描（同 struct 口径 + 每骨架蓄水池 + 未命中率计数）
# ---------------------------------------------------------------------------
def scan(trained: set[int], used_hash: set[str]) -> dict:
    files = resolve_corpus_files(CORPUS_GLOB)
    lang: dict[str, dict] = {}
    kept: list[Path] = []
    rnd = random.Random(SPLIT_SEED)
    for p in files:
        txt = Path(p).read_text(encoding="utf-8", errors="ignore")
        ratio = len(re.findall(r"[一-龥]", txt)) / max(1, len(txt))
        lang[Path(p).name] = {"cjk_ratio": round(ratio, 4), "kept": ratio >= MIN_CJK_RATIO}
        if ratio >= MIN_CJK_RATIO:
            kept.append(p)
    check(kept, "语言过滤后一个语料文件都不剩")
    rnd.shuffle(kept)

    res: dict[int, list] = {s: [] for s in trained}
    res_rng = {s: random.Random(SPLIT_SEED + s * 7919) for s in trained}
    cnt, cnt_any = Counter(), Counter()
    seen: set[str] = set()
    n_seg = n_ok = n_match = n_match_tr = n_pre = 0
    for p in kept:
        name = Path(p).name
        for line in Path(p).read_text(encoding="utf-8", errors="ignore").splitlines():
            t = ROLE_PREFIX.sub("", line.strip()).translate(PUNCT_NORM)
            for s in SENT_SPLIT.split(t):
                s = s.strip().strip(TRAIL_WEAK)
                n_seg += 1
                if not ok_sentence(s):
                    continue
                n_pre += 1
                hh = h12(s)
                if hh in seen or hh in used_hash:
                    continue
                seen.add(hh)
                n_ok += 1
                if not (set(s) & ALLOWED_LIT_CHARS):
                    continue
                m = match_sentence(s)
                if m is None:
                    continue
                n_match += 1
                sid, gaps, spans = m
                cnt_any[sid] += 1
                if sid not in trained:
                    continue
                n_match_tr += 1
                cnt[sid] += 1
                i = cnt[sid] - 1
                r = res_rng[sid]
                item = {"sent": s, "gaps": gaps, "spans": spans, "file": name}
                if len(res[sid]) < CAP_RES:
                    res[sid].append(item)
                else:
                    j = r.randint(0, i)
                    if j < CAP_RES:
                        res[sid][j] = item
        print(f"[scan] {name} seg={n_seg:,} ok={n_ok:,} match={n_match:,} "
              f"trained={n_match_tr:,}", flush=True)

    return {"res": res, "cnt": dict(cnt), "cnt_any": dict(cnt_any),
            "n_seg": n_seg, "n_pre": n_pre, "n_ok": n_ok, "n_match": n_match,
            "n_match_tr": n_match_tr, "lang": lang, "files_kept": [Path(p).name for p in kept]}


# ---------------------------------------------------------------------------
# 成行（袋洗牌，规则 0）
# ---------------------------------------------------------------------------
def to_rows(picks: list[dict], split: str, salt: int) -> list[dict]:
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    srng = random.Random(BAG_SEED + salt)
    out = []
    for c in picks:
        gaps, spans = c["gaps"], c["spans"]
        n = len(gaps)
        perm = list(range(n))
        srng.shuffle(perm)
        bag = [gaps[i] for i in perm]
        bag_span = [spans[i] for i in perm]
        assign = [perm.index(slot) for slot in range(n)]
        e = tok.encode(c["sent"], max_length=64, padding=False)
        check(sum(e["attention_mask"]) == len(c["sent"]), f"截断丢弃：{c['sent']}")
        row = {"sent": c["sent"], "skel_id": c["skel_id"],
               "skel": SKELS[c["skel_id"]].template, "n_slots": n, "bag": bag,
               "bag_span": bag_span, "assign": assign, "split": split, "file": c.get("file")}
        for k in ("pair", "kind"):
            if c.get(k) is not None:
                row[k] = c[k]
        out.append(row)
    for r in out:
        check([r["sent"][a:b] for a, b in r["bag_span"]] == r["bag"], "span 与 bag 不一致")
        check(all(GAP_LEN[0] <= len(g) <= GAP_LEN[1] for g in r["bag"]), "槽长越界")
    return out


def pick(res: dict, sid: int, need: int, used: set[str]) -> list[dict]:
    cand = [x for x in res[sid] if h12(x["sent"]) not in used]
    cand.sort(key=lambda x: x["sent"])
    random.Random(SPLIT_SEED + 31 + sid).shuffle(cand)
    check(len(cand) >= need, f"#{sid} 可用不足：{len(cand)} < {need}")
    out = cand[:need]
    for x in out:
        used.add(h12(x["sent"]))
    return out


def _try_variant(v: str, dst: int, gaps: list[str], used: set[str],
                 reasons: Counter) -> tuple[list, list] | None:
    if h12(v) in used:
        reasons["重复"] += 1
        return None
    if not ok_sentence(v):
        reasons["ok_sentence拒"] += 1
        return None
    m = match_sentence(v)
    if m is None:
        reasons["无匹配"] += 1
        return None
    sid2, g2, sp2 = m
    if sid2 != dst:
        reasons[f"错骨架#{sid2}"] += 1
        return None
    if g2 != gaps and sorted(g2) != sorted(gaps):
        reasons["槽文本不一致"] += 1
        return None
    return g2, sp2


# ---------------------------------------------------------------------------
def main() -> None:
    train = load_rows(TCH_DATA / "train.jsonl")
    test = load_rows(TCH_DATA / "test.jsonl")
    adv1 = load_rows(SS_DATA / "adv1.jsonl")
    adv2 = load_rows(SS_DATA / "adv2.jsonl")
    used: set[str] = {h12(r["sent"]) for r in train + test + adv1 + adv2}
    tr_cnt = Counter(r["skel_id"] for r in train)
    trained = set(tr_cnt)
    check(len(trained) == 30, f"训练骨架数应为 30，实得 {len(trained)}")

    bucket_tr: dict[int, Counter] = defaultdict(Counter)
    for r in train:
        bucket_tr[r["n_slots"]][r["skel_id"]] += 1
    L = {n: c.most_common(1)[0][0] for n, c in sorted(bucket_tr.items())}
    check(L == {1: 35, 2: 0, 3: 1, 4: 2}, f"train 泄露骨架漂移：{L}")
    Lset = set(L.values())
    print(f"[L] 泄露骨架（每桶 train 多数）= {L}", flush=True)

    twin_info = {}
    for a, b in C_PAIRS:
        ok, i, desc = twin_ok(a, b)
        check(ok, f"孪生对 ({a},{b}) 不满足 PREREG 判据：{desc}")
        twin_info[f"{a}-{b}"] = {"seg": i, "desc": desc}
    print(f"[twin] {len(C_PAIRS)} 对全部通过机械校验", flush=True)

    sc = scan(trained, used)
    print(f"[scan-done] 段 {sc['n_seg']:,} 通过ok_sentence {sc['n_pre']:,} "
          f"去重去已用 {sc['n_ok']:,} 命中 {sc['n_match']:,} 训练骨架命中 {sc['n_match_tr']:,}",
          flush=True)
    avail = sc["cnt"]

    # ---------------- A 型主集 a_bal ----------------
    by_bucket: dict[int, list[int]] = defaultdict(list)
    for sid in sorted(trained):
        by_bucket[SKELS[sid].template.count("[")].append(sid)
    combos_bal = [(n, sid) for n in sorted(by_bucket) for sid in by_bucket[n]
                  if sid not in Lset]
    dropped = [(n, sid, avail.get(sid, 0)) for n, sid in combos_bal
               if avail.get(sid, 0) < MIN_COMBO]
    combos_bal = [(n, sid) for n, sid in combos_bal if avail.get(sid, 0) >= MIN_COMBO]
    C_bal = min([MAX_COMBO] + [avail[sid] for _, sid in combos_bal])
    check(C_bal >= MIN_COMBO, f"a_bal 单组合取样过小：{C_bal}")
    print(f"[a_bal] 组合 {len(combos_bal)} 个，C={C_bal}，剔除 {dropped}", flush=True)

    a_bal_cands: list[dict] = []
    for n, sid in combos_bal:
        a_bal_cands += [dict(x, skel_id=sid) for x in pick(sc["res"], sid, C_bal, used)]
    a_bal = to_rows(a_bal_cands, "a_bal", 101)

    # ---------------- 对照 a_lit = a_bal ∪ {L 四组合各 C 行}（同袋洗牌 ⇒ 前缀逐行相同）---
    a_lit_cands = [dict(x, skel_id=r["skel_id"]) for x, r in zip(a_bal_cands, a_bal)]
    L_dropped = []
    for n in sorted(L):
        sid = L[n]
        if avail.get(sid, 0) >= MIN_COMBO:
            a_lit_cands += [dict(x, skel_id=sid)
                            for x in pick(sc["res"], sid, min(C_bal, avail[sid]), used)]
        else:
            L_dropped.append([n, sid, avail.get(sid, 0)])
    a_lit = to_rows(a_lit_cands, "a_lit", 101)

    # ---------------- C 型自然句孪生对 ----------------
    c_rows: list[dict] = []
    c_report = {}
    for a, b in C_PAIRS:
        va, vb = avail.get(a, 0), avail.get(b, 0)
        side = min([MAX_PAIR_SIDE, va, vb])
        if side < MIN_PAIR_SIDE:
            c_report[f"{a}-{b}"] = {"ok": False, "reason": f"单侧可用不足 a={va} b={vb}",
                                    "avail": [va, vb], "desc": twin_info[f"{a}-{b}"]["desc"]}
            continue
        for sid in (a, b):
            c_rows += [dict(x, skel_id=sid, pair=f"{a}-{b}", kind="natural")
                       for x in pick(sc["res"], sid, side, used)]
        c_report[f"{a}-{b}"] = {"ok": True, "per_side": side, "avail": [va, vb],
                                "desc": twin_info[f"{a}-{b}"]["desc"]}
    c_pairs = to_rows(c_rows, "c_pairs", 103)

    # ---------------- B 型合成最小对 ----------------
    b_cands: list[dict] = []
    b_report = {}
    b_fail: Counter = Counter()
    for a, b in C_PAIRS:
        i = twin_info[f"{a}-{b}"]["seg"]
        assert i >= 0
        for src, dst in ((a, b), (b, a)):
            src_cand = [x for x in sc["res"][src] if h12(x["sent"]) not in used]
            src_cand.sort(key=lambda x: x["sent"])
            random.Random(SPLIT_SEED + 977 + src * 3 + dst).shuffle(src_cand)
            need = min(MAX_B_DIR, avail.get(src, 0))
            take, reasons = 0, Counter()
            for x in src_cand:
                if take >= need:
                    break
                gaps = x["gaps"]
                v = render(SKELS[dst].template, gaps)      # B1：同槽序换功能词
                if h12(v) == h12(x["sent"]):
                    reasons["与源相同"] += 1
                    continue
                got = _try_variant(v, dst, gaps, used, reasons)
                if got is None:
                    continue
                g2, sp2 = got
                take += 1
                used.add(h12(x["sent"]))
                used.add(h12(v))
                b_cands.append(dict(x, skel_id=src, pair=f"{a}-{b}", kind="b1_src"))
                b_cands.append({"sent": v, "gaps": g2, "spans": sp2, "skel_id": dst,
                                "pair": f"{a}-{b}", "kind": "b1_variant"})
            b_fail += reasons
            b_report[f"b1 {src}->{dst}"] = {"ok": take >= 5, "take": take,
                                            "fail": dict(reasons),
                                            "src_avail": avail.get(src, 0)}
    for src, dst in B2_PAIRS:
        src_cand = [x for x in sc["res"][src] if h12(x["sent"]) not in used]
        src_cand.sort(key=lambda x: x["sent"])
        random.Random(SPLIT_SEED + 1301 + src * 3 + dst).shuffle(src_cand)
        take, reasons = 0, Counter()
        for x in src_cand:
            if take >= MAX_B_DIR:
                break
            # B2：连词位置交换 ⇒ 槽序相对连词交换（袋内容不变，槽1↔槽2 的词面位置对调）
            gaps = x["gaps"][::-1]
            v = render(SKELS[dst].template, gaps)
            got = _try_variant(v, dst, gaps, used, reasons)
            if got is None:
                continue
            g2, sp2 = got
            take += 1
            used.add(h12(x["sent"]))
            used.add(h12(v))
            b_cands.append(dict(x, skel_id=src, pair=f"{src}-{dst}", kind="b2_src"))
            b_cands.append({"sent": v, "gaps": g2, "spans": sp2, "skel_id": dst,
                            "pair": f"{src}-{dst}", "kind": "b2_variant"})
        b_fail += reasons
        b_report[f"b2 {src}->{dst}"] = {"ok": take >= 5, "take": take, "fail": dict(reasons),
                                        "src_avail": avail.get(src, 0)}
    b_pairs = to_rows(b_cands, "b_pairs", 104)

    # ---------------- 断言 ----------------
    sets = {"a_bal": a_bal, "a_lit": a_lit, "c_pairs": c_pairs, "b_pairs": b_pairs}
    base = {h12(r["sent"]) for r in train + test + adv1 + adv2}
    for name, rows in sets.items():
        hs = [h12(r["sent"]) for r in rows]
        check(len(hs) == len(set(hs)), f"{name} 内部句子重复")
        check(not (set(hs) & base), f"{name} 与 train/test/adv1/adv2 句子重叠")
        check(all(r["skel_id"] in trained for r in rows), f"{name} 含训练零样本骨架")
    check(not ({h12(r["sent"]) for r in a_bal} - {h12(r["sent"]) for r in a_lit}),
          "a_lit 必须包含 a_bal")
    check(not (set(r["skel_id"] for r in a_bal) & Lset),
          "a_bal 含泄露骨架 L（PREREG §2 要求排除）")
    ab = {h12(r["sent"]) for r in a_bal}
    for name in ("c_pairs", "b_pairs"):
        check(not ({h12(r["sent"]) for r in sets[name]} & ab),
              f"{name} 与 a_bal 句子重叠")
    extra = {h12(r["sent"]) for r in a_lit} - ab
    check(not (extra & ({h12(r['sent']) for r in c_pairs} |
                        {h12(r['sent']) for r in b_pairs})), "a_lit 增量与 C/B 重叠")

    # ---------------- 落盘 ----------------
    DATA.mkdir(exist_ok=True)
    for name, rows in sets.items():
        with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # ---------------- 统计 ----------------
    def bucket_stat(rows: list[dict]) -> dict:
        by = defaultdict(Counter)
        for r in rows:
            by[r["n_slots"]][r["skel_id"]] += 1
        out = {}
        for n in sorted(by):
            c = by[n]
            tot = sum(c.values())
            h = -sum((v / tot) * math.log(v / tot) for v in c.values())
            out[str(n)] = {"n": tot, "k": len(c), "H_nats": round(h, 4),
                           "H_norm": round(h / math.log(len(c)), 4) if len(c) > 1 else 1.0,
                           "top": [(int(a), b) for a, b in c.most_common(4)]}
        return out

    b_take = sum(v.get("take", 0) for v in b_report.values())
    b_try = b_take + sum(b_fail.values())
    stats = {
        "prereg": "PREREG.md §1/§2",
        "L": {str(k): v for k, v in L.items()},
        "scan": {"segments": sc["n_seg"], "pass_ok_sentence": sc["n_pre"],
                 "dedup_unique": sc["n_ok"], "matched_any": sc["n_match"],
                 "matched_trained": sc["n_match_tr"],
                 "miss_ok_sentence": round(1 - sc["n_pre"] / sc["n_seg"], 4),
                 "miss_match": round(1 - sc["n_match"] / sc["n_ok"], 4),
                 "miss_untrained": round(1 - sc["n_match_tr"] / sc["n_match"], 4),
                 "files_kept": sc["files_kept"], "file_cjk": sc["lang"]},
        "avail_per_skeleton": {str(k): v for k, v in sorted(avail.items())},
        "construct": {
            "A_a_bal": {"status": "可构造" if not dropped else "部分组合不可构造",
                        "n_combos": len(combos_bal), "C": C_bal,
                        "dropped": dropped,
                        "未命中率": {"ok_sentence": sc and round(1 - sc["n_pre"] / sc["n_seg"], 4),
                                     "match": round(1 - sc["n_match"] / sc["n_ok"], 4),
                                     "非训练骨架": round(1 - sc["n_match_tr"] / sc["n_match"], 4)}},
            "A_a_lit": {"status": "可构造", "L_combos_dropped": L_dropped},
            "B_b_pairs": {"status": "可构造" if b_take >= 10 else "不可构造",
                          "pairs_attempted": len(C_PAIRS) * 2 + len(B2_PAIRS),
                          "pairs_ok": sum(1 for v in b_report.values() if v["ok"]),
                          "rows": len(b_pairs), "synth_tried": b_try, "synth_take": b_take,
                          "未命中率": round(sum(b_fail.values()) / max(1, b_try), 4),
                          "fail_reasons": dict(b_fail), "report": b_report},
            "C_c_pairs": {"status": "可构造" if any(v["ok"] for v in c_report.values())
                          else "不可构造",
                          "pairs_total": len(C_PAIRS),
                          "pairs_ok": sum(1 for v in c_report.values() if v["ok"]),
                          "pairs_failed": [k for k, v in c_report.items() if not v["ok"]],
                          "rows": len(c_pairs), "report": c_report},
        },
        "sets": {k: {"n": len(v),
                     "files": dict(Counter(r.get("file") for r in v)),
                     "buckets": bucket_stat(v),
                     "label_dist": dict(sorted(Counter(r["skel_id"] for r in v).items()))}
                 for k, v in sets.items()},
        "twin_table": twin_info,
    }
    (DATA / "stats_build.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print("[construct] " + json.dumps(stats["construct"], ensure_ascii=False)[:3000], flush=True)
    print(f"[done] a_bal={len(a_bal)} a_lit={len(a_lit)} c_pairs={len(c_pairs)} "
          f"b_pairs={len(b_pairs)} → {DATA}", flush=True)
    for name, rows in sets.items():
        print(f"[sample {name}] " + " | ".join(r["sent"] for r in rows[:3]))


if __name__ == "__main__":
    main()
