#!/usr/bin/env python3
"""P23 定义勘察（**只用 train 内部 80/20**，不看 a_bal/test/adv2 任何 acc、不看任何 gold 于这些出口）。

**选择规则（本文件 docstring 先于运行写死，PREREG §1.2 原文引用）**：
  对含多个候选构造的信号（③④⑤⑥）：
    (i)  保留 train-holdout gold 覆盖率 ≥ 0.90 的构造；
    (ii) 在保留者中取「中位候选集最小」者；
    (iii) 若无一 ≥0.90 ⇒ 取覆盖率最高者，并在 PREREG/REPORT 标注为「弱构造」；
    (iv) **并列**（本行在 design_scan 跑完之后、run_eval 之前补齐，mtime 可查）：
         比覆盖率 ⇒ 比 p75 ⇒ 比构造列举顺序（靠前胜）。
  信号 ①② 只有唯一构造（P8 规则 / P22 定义），不参与选择。
  **看的是覆盖率与候选集大小，不是任何模型 acc。**

用法：uv run python experiments/orthogonal_tighten/design_scan.py
"""
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
P8 = ROOT / "experiments" / "mode_conditioned_skel"
sys.path.insert(0, str(P8))

import common as C  # noqa: E402  只读复用（P8）
import modes as MM  # noqa: E402  只读复用（P8）

TCH = ROOT / "experiments" / "two_channel_head" / "data"
TAIL = MM._TAIL
FUNC_LAST = "吗么呢吧"


# ---------------------------------------------------------------------------
# 行侧可观测键（全部只用 sent / bag_span / n_slots；无 skel_id、无 gold）
# ---------------------------------------------------------------------------
def k_mode(row: dict) -> str:
    return MM.MASK_ALIAS[MM.mode_of_sent(row["sent"])]


def k_ns(row: dict) -> int:
    return row["n_slots"]


def _split_tail(s: str) -> tuple[str, str]:
    """(去尾末字, 原始末字符号) —— 去尾规则复用 P8 mode_of_sent 的 _TAIL。"""
    t = s.rstrip()
    while t and t[-1] in TAIL:
        t = t[:-1]
    return (t[-1] if t else ""), (s.rstrip()[-1] if s.rstrip() else "")


def k_last(row: dict) -> tuple[str, str]:
    """⑤ 末字/标点型 = (去尾末字类, 原末标点类)。
    去尾末字类：功能字 ∈ 吗么呢吧 ⇒ 该字；否则 'C'（内容字）。
    原末标点类：末字符 ∈ ！!? ⇒ '！'；∈ 。．.…～~）)」』"'”’ ⇒ '。'；
               ∈ ，,；;：:·、 ⇒ '，'；无 ⇒ '∅'。"""
    last, raw = _split_tail(row["sent"])
    cls = last if last in FUNC_LAST else "C"
    if raw in "！!?":
        p = "！"
    elif raw in "。．.…～~）)」』\"'”’":
        p = "。"
    elif raw in "，,；;：:·、":
        p = "，"
    else:
        p = "∅"
    return (cls, p)


def k_firstlast(row: dict) -> tuple[str, tuple[str, str]]:
    """⑤' 变体：首字类 ∘ 末字/标点型。首字类 = 首字本身（去空白后）。"""
    s = row["sent"].strip()
    return (s[0] if s else "", k_last(row))


def k_type_inv(row: dict) -> tuple:
    """⑥ 类型库存 = 排序后的槽位类型多重集（label_row 只读句面 + bag_span）。"""
    return tuple(sorted(l["t"] for l in C.label_row(row)))


def k_type_seq(row: dict) -> tuple:
    """⑥' 变体：按 span 顺序的类型序列（库存的有序版）。"""
    return tuple(l["t"] for l in C.label_row(row))


def sig_c(row: dict, mode: bool) -> tuple:
    """④ 合格候选数的行侧签名：(n_slots, 末字型) 或再 ∘ 句式。"""
    base = (k_ns(row), k_last(row))
    return base + (k_mode(row),) if mode else base


# ---------------------------------------------------------------------------
# 骨架侧（全部来自 train 统计）
# ---------------------------------------------------------------------------
def build_stats(train: list[dict]) -> dict:
    covered = sorted({r["skel_id"] for r in train})
    ns_of: dict[int, set] = defaultdict(set)
    cnt: Counter = Counter()
    sets: dict[str, dict[int, set]] = {k: defaultdict(set) for k in
                                       ("last", "firstlast", "inv", "seq", "c2", "c3")}
    for r in train:
        sid = r["skel_id"]
        ns_of[sid].add(r["n_slots"])
        cnt[sid] += 1
        sets["last"][sid].add(k_last(r))
        sets["firstlast"][sid].add(k_firstlast(r))
        sets["inv"][sid].add(k_type_inv(r))
        sets["seq"][sid].add(k_type_seq(r))
        sets["c2"][sid].add(sig_c(r, False))
        sets["c3"][sid].add(sig_c(r, True))
    for sid, v in ns_of.items():
        assert len(v) == 1, f"骨架 {sid} 的 n_slots 非确定性"
    ns = {sid: next(iter(v)) for sid, v in ns_of.items()}

    # ③(b) cnt 等频 3 分位（ties 按 (cnt, sid) 升序，前 1/3 → 桶 0）
    order = sorted(covered, key=lambda s: (cnt[s], s))
    bq = {sid: min(2, i * 3 // len(order)) for i, sid in enumerate(order)}
    return {"covered": covered, "ns": ns, "cnt": dict(cnt), "bq": bq,
            "sets": {k: dict(v) for k, v in sets.items()}}


def s1(row: dict, sA: dict) -> list[int]:
    return sA[k_mode(row)]


def s2(row: dict, st: dict) -> list[int]:
    return [s for s in st["covered"] if st["ns"][s] == k_ns(row)]


def s3_b1(row: dict, st: dict, q_cnt: dict, bq_of_q: dict) -> list[int]:
    """行侧样本数 = train 中同 (句式,n_slots) 行数；其分位 == 骨架 cnt 分位。"""
    b = bq_of_q.get((k_mode(row), k_ns(row)), 2)
    return [s for s in st["covered"] if st["bq"][s] == b]


def s3_b2(row: dict, st: dict, modal: dict) -> list[int]:
    """行侧键 = (句式,n_slots)；候选 = 该键 train 条件分布的众数分位桶。"""
    b = modal.get((k_mode(row), k_ns(row)), None)
    if b is None:
        return list(st["covered"])
    return [s for s in st["covered"] if st["bq"][s] == b]


def s4(row: dict, st: dict, which: str) -> list[int]:
    key = sig_c(row, which == "c3")
    return [s for s in st["covered"] if key in st["sets"][which][s]]


def s5(row: dict, st: dict, which: str) -> list[int]:
    key = k_last(row) if which == "last" else k_firstlast(row)
    return [s for s in st["covered"] if key in st["sets"][which][s]]


def s6(row: dict, st: dict, which: str) -> list[int]:
    key = k_type_inv(row) if which == "inv" else k_type_seq(row)
    return [s for s in st["covered"] if key in st["sets"][which][s]]


def eval_arm(rows: list[dict], fn) -> dict:
    sizes, hit, empty = [], 0, 0
    for r in rows:
        ks = fn(r)
        sizes.append(len(ks))
        if not ks:
            empty += 1
        if r["skel_id"] in ks:
            hit += 1
    sizes.sort()
    n = len(rows)
    return {"cov": round(hit / n, 4), "empty": round(empty / n, 4),
            "med": sizes[n // 2], "p25": sizes[n // 4], "p75": sizes[3 * n // 4],
            "min": sizes[0], "max": sizes[-1]}


def main() -> None:
    train = C.load_rows(TCH / "train.jsonl")
    st = build_stats(train)
    tab = MM.skeleton_table()
    sA = MM.subsets_A(tab)

    # ③ 的 train 侧辅助统计（只用 train 前 80%）
    idx = list(range(len(train)))
    import torch
    g = torch.Generator().manual_seed(42)
    perm = torch.randperm(len(idx), generator=g).tolist()
    cut = int(0.8 * len(train))
    tr80 = [train[i] for i in perm[:cut]]
    ev20 = [train[i] for i in perm[cut:]]

    q_cnt = Counter((k_mode(r), k_ns(r)) for r in tr80)
    qs = sorted(q_cnt.values())
    b_of_q = {}
    for k, v in q_cnt.items():
        r = qs.index(v) * 3 // max(1, len(qs))
        b_of_q[k] = min(2, r)
    by_key = defaultdict(Counter)
    st80 = build_stats(tr80)
    for r in tr80:
        by_key[(k_mode(r), k_ns(r))][st80["bq"][r["skel_id"]]] += 1
    modal = {}
    for k, c in by_key.items():
        modal[k] = min(c, key=lambda b: (-c[b], b))

    out = {"n_train": len(train), "covered": len(st["covered"]),
           "size_A": {m: len(sA[m]) for m in MM.MODES},
           "ns_buckets": {str(k): [s for s in st["covered"] if st["ns"][s] == k]
                          for k in sorted(set(st["ns"].values()))},
           "cnt_tercile": {str(b): sorted(s for s in st["covered"] if st["bq"][s] == b)
                           for b in (0, 1, 2)},
           "arms": {}}
    arms = {
        "①句式": lambda r: s1(r, sA),
        "②n_slots": lambda r: s2(r, st),
        "③b1_等分位相等": lambda r: s3_b1(r, st, q_cnt, b_of_q),
        "③b2_条件众数": lambda r: s3_b2(r, st, modal),
        "④c2_签名(ns,末字)": lambda r: s4(r, st, "c2"),
        "④c3_签名(句式,ns,末字)": lambda r: s4(r, st, "c3"),
        "⑤last_末字标点": lambda r: s5(r, st, "last"),
        "⑤fl_首字∘末字": lambda r: s5(r, st, "firstlast"),
        "⑥inv_类型库存": lambda r: s6(r, st, "inv"),
        "⑥seq_类型序列": lambda r: s6(r, st, "seq"),
    }
    for name, fn in arms.items():
        out["arms"][name] = {"holdout": eval_arm(ev20, fn), "all_train": eval_arm(train, fn)}
    (HERE / "design_scan.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                           encoding="utf-8")
    print(f"train={len(train)} covered={len(st['covered'])} "
          f"size_A={out['size_A']}")
    for k, v in out["arms"].items():
        h = v["holdout"]
        print(f"  {k:24s} holdout cov={h['cov']:.4f} empty={h['empty']:.4f} "
              f"size med={h['med']} p25={h['p25']} p75={h['p75']} "
              f"[{h['min']},{h['max']}] | all cov={v['all_train']['cov']:.4f}")
    print(f"[done] → {HERE / 'design_scan.json'}")


if __name__ == "__main__":
    main()
