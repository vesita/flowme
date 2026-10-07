#!/usr/bin/env python3
"""P4 功能词最小对：训练最小对构造 + 可构造性自检 + 训练/held-out 分离断言。

口径 = `PREREG.md` §1（跑前写死）：
  · 训练/测试只读复用 `two_channel_head/data/{train,test}.jsonl`；匹配器/过滤器**只读 import**
    `two_channel_head/build_gen_data`（与 train/test 逐字同口径）；
  · R1 等长换词 / R2 功能词增删 / R3 槽序交换 —— 三条**机械规则**，跑前枚举写死（62/21/1）；
  · 伙伴选择：每行至多 1 个 R1/R2（按 md5(f"P4:{i}:{t}") 升序取第一个通过验证的），
    骨架 ∈{6,7} 额外追加 1 个 R3；fail-closed 验证（ok_sentence ∧ matcher == 目标骨架 ∧ 槽文本同序同文 ∧ 未用过）；
  · 训练/held-out 分离：held-out = `skeleton_leak` 只读复用 b_pairs(B1/B2) + c_pairs，
    四条分离断言全 0，任一非 0 即 fail-closed 停。

用法：uv run python experiments/funcword_minpair/build_data.py
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "two_channel_head"))

from build_gen_data import (  # noqa: E402  只读
    ALLOWED_LIT_CHARS, PUNCTS, SKELS, match_sentence, ok_sentence,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
ROOTEX = ROOT / "experiments"
TCH_DATA = ROOTEX / "two_channel_head" / "data"
SL_DATA = ROOTEX / "skeleton_leak" / "data"
SS_DATA = ROOTEX / "struct_supervision" / "data"

ZERO_IDS = {16, 21, 24, 27, 28, 29, 30, 32, 33, 34}
ORDER_PAIR_IDS = (6, 7)          # R3 唯一可构造骨架对（跑前枚举）

# PREREG §1.4：held-out 只读复用（不重建）
HELDOUT = ("b_pairs", "c_pairs")


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[funcword_minpair build fail-closed] {msg}")


def h12(s: str) -> str:
    return hashlib.md5(s.encode()).hexdigest()[:12]


def load_rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def parts(t: str) -> list[str]:
    return re.split(r"\[\d+\]", t)


def render(tmpl: str, gaps: list[str]) -> str:
    p = parts(tmpl)
    out = p[0]
    for g, q in zip(gaps, p[1:]):
        out += g + q
    return out


def litchars(t: str) -> Counter:
    return Counter(ch for seg in parts(t) for ch in seg)


def fw_chars(seg: str) -> set[str]:
    return set(seg) - set(PUNCTS)


# ---------------------------------------------------------------------------
# 三条机械规则（PREREG §1.2，跑前写死）
# ---------------------------------------------------------------------------
def enumerate_rules(trained: set[int]) -> tuple[list, list, list]:
    r1: list[tuple[int, int, str, str]] = []
    r2: list[tuple[int, int, str, str]] = []
    r3: list[tuple[int, int]] = []
    ids = sorted(trained)
    for i, a in enumerate(ids):
        for b in ids[i + 1:]:
            if SKELS[a].k != SKELS[b].k:
                continue
            pa, pb = parts(SKELS[a].template), parts(SKELS[b].template)
            # ---- R3：字面多重集相等、段序列不同 ----
            if litchars(SKELS[a].template) == litchars(SKELS[b].template) and pa != pb:
                r3.append((a, b))
            if len(pa) != len(pb):
                continue
            d = [k for k in range(len(pa)) if pa[k] != pb[k]]
            if len(d) != 1:
                continue
            k = d[0]
            x, y = pa[k], pb[k]
            if set(x + y) - ALLOWED_LIT_CHARS:        # 字面必须 ⊆ 标点+形式虚词
                continue
            # ---- R1 等长换词（至少含 1 个非标点功能词）----
            if len(x) == len(y) and (fw_chars(x) | fw_chars(y)):
                r1.append((a, b, x, y))
                continue
            # ---- R2 功能词增删（一段是另一段连续子串，增删物非空且不含标点）----
            hit = None
            for short, long_ in ((x, y), (y, x)):
                if short in long_:
                    ins = long_.replace(short, "", 1)
                    if ins and not (set(ins) & set(PUNCTS)):
                        hit = (a, b, x, y)
                        break
            if hit:
                r2.append(hit)
    return r1, r2, r3


def sentence_order_gaps(r: dict) -> list[str]:
    """按 span 升序取**句序**槽文本（render 的输入）。span 是半开区间 [start,end)。"""
    sp = sorted(r["bag_span"], key=lambda x: x[0])
    g = [r["sent"][a:b] for a, b in sp]
    check(all(r["sent"][a:b] for a, b in sp), "span 取到空槽")
    return g


def try_variant(src: dict, dst: int, gaps: list[str], used: set[str],
                reasons: Counter) -> dict | None:
    v = render(SKELS[dst].template, gaps)
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
    if g2 != gaps:
        reasons["槽文本不一致"] += 1
        return None
    n = len(g2)
    return {"sent": v, "skel_id": dst, "skel": SKELS[dst].template,
            "n_slots": n, "bag": list(g2), "bag_span": [list(x) for x in sp2],
            "assign": list(range(n))}


def main() -> None:
    train = load_rows(TCH_DATA / "train.jsonl")
    test = load_rows(TCH_DATA / "test.jsonl")
    adv1 = load_rows(SS_DATA / "adv1.jsonl")
    adv2 = load_rows(SS_DATA / "adv2.jsonl")
    held = {k: load_rows(SL_DATA / f"{k}.jsonl") for k in HELDOUT}

    tr_cnt = Counter(r["skel_id"] for r in train)
    trained = set(tr_cnt)
    check(len(trained) == 30, f"训练骨架应为 30，实得 {len(trained)}")
    check(not (trained & ZERO_IDS), "训练集含零样本 id？口径漂移")

    # L = 每桶 train 多数（PREREG §3.1 断言）
    bucket: dict[int, Counter] = defaultdict(Counter)
    for r in train:
        bucket[r["n_slots"]][r["skel_id"]] += 1
    Lmap = {n: c.most_common(1)[0][0] for n, c in sorted(bucket.items())}
    check(Lmap == {1: 35, 2: 0, 3: 1, 4: 2}, f"L 漂移：{Lmap}")
    Lset = set(Lmap.values())
    print(f"[L] 泄露骨架（每桶 train 多数）= {Lmap} → L={sorted(Lset)}", flush=True)

    # ---------------- 三条机械规则 ----------------
    r1, r2, r3 = enumerate_rules(trained)
    print(f"[rules] R1={len(r1)} R2={len(r2)} R3={len(r3)}", flush=True)
    check(len(r1) == 62, f"R1 应为 62（PREREG 跑前枚举），实得 {len(r1)}")
    check(len(r2) == 21, f"R2 应为 21，实得 {len(r2)}")
    check(len(r3) == 1 and tuple(r3[0]) == (6, 7), f"R3 应为 [(6,7)]，实得 {r3}")

    cand: dict[int, list[int]] = defaultdict(list)
    for a, b, *_ in r1 + r2:               # word 伙伴只来自 R1∪R2（PREREG §1.3）
        cand[a].append(b)
        cand[b].append(a)
    for k in cand:
        cand[k] = sorted(set(cand[k]))
    cov_rows = sum(v for k, v in tr_cnt.items() if k in cand)
    print(f"[cover] 候选骨架 {len(cand)}/30，候选行 {cov_rows}/{len(train)} "
          f"= {cov_rows / len(train):.4f}", flush=True)
    uncovered = {k: tr_cnt[k] for k in sorted(trained) if k not in cand}
    print(f"[cover] 未覆盖骨架 {uncovered}", flush=True)

    # ---------------- 已用句（分离断言用） ----------------
    used: set[str] = set()
    for r in train + test + adv1 + adv2:
        used.add(h12(r["sent"]))
    held_hash = {k: [h12(x["sent"]) for x in v] for k, v in held.items()}

    # ---------------- 逐行构造伙伴 ----------------
    variants: dict[int, dict] = {}          # 行下标 → 变体行
    fail: Counter = Counter()
    n_try = n_ok_row = 0
    n_cand_rows = 0
    per_rule_ok = Counter()
    for i, r in enumerate(train):
        gaps = sentence_order_gaps(r)
        s = r["skel_id"]
        cands = cand.get(s, [])
        if cands:
            n_cand_rows += 1
        # R1/R2 伙伴：候选按 md5(f"P4:{i}:{t}") 升序，取第一个通过验证的
        order = sorted(cands, key=lambda t: hashlib.md5(f"P4:{i}:{t}".encode()).hexdigest())
        got = None
        for t in order:
            n_try += 1
            v = try_variant(r, t, gaps, used, fail)
            if v is None:
                continue
            got = v
            got["type"] = "word"
            got["dst"] = t
            break
        # R3 追加（骨架 ∈ {6,7}）：在**已有 word 伙伴**之外额外追加
        if got is None:
            continue
        used.add(h12(got["sent"]))
        variants[i] = got
        n_ok_row += 1
        per_rule_ok["word"] += 1

    # R3 追加：把 skel∈{6,7} 的行的 word 伙伴**替换**为固定槽序伙伴？
    # PREREG §1.3 写死：**额外追加** ⇒ 这些行有两个伙伴，取「槽序」那个作为第 2 份监督。
    n_r3_ok = n_r3_try = 0
    r3_fail: Counter = Counter()
    r3_rows: dict[int, dict] = {}
    for i, r in enumerate(train):
        if r["skel_id"] not in ORDER_PAIR_IDS:
            continue
        t = 7 if r["skel_id"] == 6 else 6
        gaps_rev = sentence_order_gaps(r)[::-1]
        n_r3_try += 1
        v = try_variant(r, t, gaps_rev, used, r3_fail)
        if v is None:
            continue
        used.add(h12(v["sent"]))
        v["type"] = "order"
        v["dst"] = t
        r3_rows[i] = v
        n_r3_ok += 1

    total_att = n_try + n_r3_try
    total_fail = sum(fail.values()) + sum(r3_fail.values())
    print(f"[build] 候选行 {n_cand_rows}；word 伙伴成功行 {n_ok_row}（尝试 {n_try}，"
          f"失败 {sum(fail.values())} = {sum(fail.values()) / max(1, n_try):.4%}）", flush=True)
    print(f"[build] R3 槽序成功 {n_r3_ok}/{n_r3_try}；失败 {dict(r3_fail)}", flush=True)
    print(f"[build] 失败分类 {dict(fail)}", flush=True)
    print(f"[build] 未命中率（全部尝试）= {total_fail}/{total_att} = "
          f"{total_fail / max(1, total_att):.4%}", flush=True)

    # ---------------- 分离断言（PREREG §1.4 ①②③④） ----------------
    tr_hash = {h12(r["sent"]) for r in train}
    te_hash = {h12(r["sent"]) for r in test} | {h12(r["sent"]) for r in adv1 + adv2}
    var_hash = {h12(v["sent"]) for v in variants.values()} | \
               {h12(v["sent"]) for v in r3_rows.values()}
    b1 = {k for k, x in zip(held_hash["b_pairs"], held["b_pairs"])
          if x.get("kind", "").startswith("b1")}
    b2 = {k for k, x in zip(held_hash["b_pairs"], held["b_pairs"])
          if x.get("kind", "").startswith("b2")}
    cp = set(held_hash["c_pairs"])
    sep = {
        "train ∩ heldout": len(tr_hash & (b1 | b2 | cp)),
        "variant ∩ heldout": len(var_hash & (b1 | b2 | cp)),
        "heldout ∩ (train∪test∪adv1∪adv2)": len((b1 | b2 | cp) & (tr_hash | te_hash)),
        "B1 ∩ B2": len(b1 & b2),
        "b_pairs ∩ c_pairs": len((b1 | b2) & cp),
        "variant ∩ train": len(var_hash & tr_hash),
        "variant 内部重复": len(var_hash) - len(variants) - len(r3_rows),
    }
    for k, v in sep.items():
        print(f"[sep] {k} = {v}", flush=True)
    for k, v in sep.items():
        if k == "variant ∩ train":
            continue                      # 变体落在别的训练行上是允许的（两侧都在 train 内）
        check(v == 0, f"分离断言失败：{k} = {v}")

    # ---------------- held-out 配对结构自检 ----------------
    pairs_b1, pairs_b2 = [], []
    rows_b = held["b_pairs"]
    for i, x in enumerate(rows_b):
        if not x.get("kind", "").endswith("_src"):
            continue
        check(i + 1 < len(rows_b) and rows_b[i + 1].get("kind", "").endswith("_variant"),
              "b_pairs 源/变体未相邻成对")
        y = rows_b[i + 1]
        check(x["skel_id"] != y["skel_id"], "b_pairs 对骨架相同")
        check(sorted(x["bag"]) == sorted(y["bag"]), "b_pairs 对袋不同")
        gx = [x["sent"][a:b] for a, b in sorted(x["bag_span"])]
        gy = [y["sent"][a:b] for a, b in sorted(y["bag_span"])]
        if x.get("kind") == "b1_src":
            check(gx == gy, "B1 两侧槽文本应同序同文")
            pairs_b1.append((i, i + 1))
        else:
            check(gx == gy[::-1], "B2 两侧槽序应相反")
            pairs_b2.append((i, i + 1))
    print(f"[heldout] b_pairs: B1(MP-word)={len(pairs_b1)} 对, B2(MP-order)={len(pairs_b2)} 对",
          flush=True)
    check(len(pairs_b1) == 520 and len(pairs_b2) == 40,
          f"held-out 对数漂移：B1={len(pairs_b1)} B2={len(pairs_b2)}")

    # c_pairs 组内下标配对（仅诊断）
    grp: dict[str, list[int]] = defaultdict(list)
    for i, x in enumerate(held["c_pairs"]):
        grp[x["pair"]].append(i)
    c_pair_idx = []
    for k, idx in grp.items():
        half = len(idx) // 2
        check(len(idx) % 2 == 0, f"c_pairs 组 {k} 行数非偶")
        for j in range(half):
            a, b = idx[j], idx[j + half]
            check(held["c_pairs"][a]["skel_id"] != held["c_pairs"][b]["skel_id"],
                  "c_pairs 配对两侧骨架相同")
            c_pair_idx.append((a, b))
    print(f"[heldout] c_pairs: {len(c_pair_idx)} 组（两侧袋不同 ⇒ 非最小对，仅诊断）",
          flush=True)

    # ---------------- 落盘 ----------------
    DATA.mkdir(exist_ok=True)
    # 与 train 行一一对齐的伙伴表（None = 无伙伴）
    slot: list[dict | None] = [None] * len(train)
    for i, v in variants.items():
        slot[i] = v
    n_both = 0
    order_list = []
    for i, v in r3_rows.items():
        order_list.append({"idx": i, "variant": v, "word": variants.get(i)})
        if i in variants:
            n_both += 1
    with open(DATA / "train_pairs.jsonl", "w", encoding="utf-8") as f:
        for i, v in enumerate(slot):
            f.write(json.dumps({"idx": i, "variant": v}, ensure_ascii=False) + "\n")
    with open(DATA / "train_order_pairs.jsonl", "w", encoding="utf-8") as f:
        for rec in order_list:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats = {
        "rules": {"R1": [[a, b, x, y] for a, b, x, y in r1],
                  "R2": [[a, b, x, y] for a, b, x, y in r2],
                  "R3": [list(x) for x in r3]},
        "cover": {"cand_skeletons": len(cand), "cand_rows": cov_rows,
                  "uncovered_skeletons": uncovered,
                  "covered_rows_with_word_variant": n_ok_row,
                  "covered_rate": round(n_ok_row / len(train), 6),
                  "r3_ok": n_r3_ok, "r3_try": n_r3_try, "rows_with_both": n_both},
        "attempts": {"word": n_try, "order": n_r3_try, "total": total_att},
        "fail": {"word": dict(fail), "order": dict(r3_fail),
                 "total": total_fail,
                 "rate": round(total_fail / max(1, total_att), 6)},
        "sep": sep,
        "heldout": {"b1_pairs": len(pairs_b1), "b2_pairs": len(pairs_b2),
                    "c_pairs": len(c_pair_idx)},
        "L": {str(k): v for k, v in Lmap.items()},
        "n_train": len(train),
        "order_pairs_idx": [r["idx"] for r in order_list],
    }
    (DATA / "stats_build.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] → data/train_pairs.jsonl（{n_ok_row} 个 word 伙伴）、"
          f"data/train_order_pairs.jsonl（{n_r3_ok}）、data/stats_build.json", flush=True)


if __name__ == "__main__":
    main()
