#!/usr/bin/env python3
"""P7 数据构建（口径 = PREREG §1/§2/§3，跑前写死）。

语料口径（`dev-notes/19` 硬要求，逐字对齐 `skeleton_leak/build_data.py`）：
  · 入口只用 `dtseek.tasks.corpus.resolve_corpus_files(CORPUS_GLOB)`；
  · 文件级汉字占比 ≥0.6 过滤，**打乱文件序**（SPLIT_SEED），不依赖 `fs` 顺序、不写 `fs[:N]`；
  · 行级：`用户：`/`用户:` 起首，4≤长度≤80、汉字比 ≥0.6、无 ≥3 字母 ASCII 词；
  · 全池按文本去重（train/test/colloq 三者互斥）。

产物：
  data/train.jsonl, data/test.jsonl        —— 有句末标点的标签池（六类）
  data/colloq_train.jsonl, data/colloq_test.jsonl —— 无句末标点的口语池
  data/stats.json                          —— 文件清单 + 丢弃率 + 逐类 n + 构造性 rule 披露

用法：uv run python experiments/sentence_mode/build_data.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dtseek.tasks.corpus import CORPUS_GLOB, resolve_corpus_files  # noqa: E402

import labels as L  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
RESULTS = HERE / "results"

SPLIT_SEED = 20240930
MIN_CJK_FILE = 0.6
MIN_CJK_SENT = 0.6
MIN_LEN, MAX_LEN = 4, 80
N_TRAIN, N_TEST = 12000, 6000
N_COLLOQ_TRAIN, N_COLLOQ_TEST = 8000, 4000

ASCII_WORD = re.compile(r"[A-Za-z]{3,}")
CJK = re.compile(r"[一-龥]")
PREFIXES = ("用户：", "用户:")


def cjk_ratio(s: str) -> float:
    return len(CJK.findall(s)) / max(1, len(s))


def ok_sentence(s: str) -> bool:
    return (MIN_LEN <= len(s) <= MAX_LEN
            and cjk_ratio(s) >= MIN_CJK_SENT
            and not ASCII_WORD.search(s))


def main() -> None:
    files = resolve_corpus_files(CORPUS_GLOB)
    lang: dict[str, dict] = {}
    kept: list[Path] = []
    rnd = random.Random(SPLIT_SEED)
    for p in files:
        txt = Path(p).read_text(encoding="utf-8", errors="ignore")
        r = cjk_ratio(txt)
        lang[Path(p).name] = {"cjk_ratio": round(r, 4), "lines": txt.count("\n"),
                              "kept": r >= MIN_CJK_FILE}
        if r >= MIN_CJK_FILE:
            kept.append(Path(p))
    assert kept, "语言过滤后一个语料文件都不剩（fail-closed）"
    rnd.shuffle(kept)                       # 打乱文件序，不依赖 fs 顺序

    labeled: list[str] = []
    colloq: list[str] = []
    drop: Counter[str] = Counter()
    seen: set[str] = set()
    n_user = n_ok = 0
    for p in kept:
        with open(p, encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.startswith(PREFIXES):
                    continue
                n_user += 1
                s = line[len(PREFIXES[0]):].strip() if line.startswith("用户：") \
                    else line[len("用户:"):].strip()
                if not ok_sentence(s):
                    drop["行级过滤（长度/汉字比/ASCII词）"] += 1
                    continue
                n_ok += 1
                if s in seen:
                    drop["重复文本"] += 1
                    continue
                seen.add(s)
                fch = L.final_char(s)
                if fch in L.FINAL_PUNCT:
                    labeled.append(s)
                elif fch == "" or fch not in L.MASK_SET:
                    colloq.append(s)
                else:
                    drop[f"末字是其他标点 {fch}"] += 1

    rnd2 = random.Random(SPLIT_SEED + 1)
    rnd2.shuffle(labeled)
    rnd2.shuffle(colloq)

    def pack(rows: list[str], n_tr: int, n_te: int, *, punct_labels: bool) -> tuple[list, list, dict]:
        take = min(len(rows), n_tr + n_te)
        rows = rows[:take]
        tr_n = min(n_tr, take)
        tr, te = rows[:tr_n], rows[tr_n:]
        def build(chunk: list[str]) -> list[dict]:
            out = []
            for i, s in enumerate(chunk):
                mode = L.label_full(s) if punct_labels else L.label_nopunct(s)
                if punct_labels:
                    assert mode is not None, f"句末标点池出现无标签行（PREREG §2 覆盖不全）：{s!r}"
                ex = L.make_exits(s) if punct_labels else {
                    "keep": s, "maskfinal": s, "mask": L.exit_mask(s)}
                if not punct_labels:
                    L.assert_masked(ex["mask"], "colloq")
                else:
                    L.assert_masked(ex["mask"], "labeled")
                    assert ex["keep"] == s
                out.append({
                    "i": i, "text": s, "mode": L.MODE_ID[mode], "mode_name": mode,
                    "len": len(s),
                    "exit": ex,
                    "trig": {k: L.find_trigger(v) for k, v in ex.items()},
                })
            return out
        return build(tr), build(te), {"n_all": len(rows), "n_train": len(tr), "n_test": len(te)}

    tr, te, sz = pack(labeled, N_TRAIN, N_TEST, punct_labels=True)
    ctr, cte, csz = pack(colloq, N_COLLOQ_TRAIN, N_COLLOQ_TEST, punct_labels=False)

    DATA.mkdir(exist_ok=True)
    for name, rows in (("train", tr), ("test", te),
                       ("colloq_train", ctr), ("colloq_test", cte)):
        with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def dist(rows: list[dict]) -> dict:
        c = Counter(r["mode_name"] for r in rows)
        n = len(rows)
        return {k: {"n": v, "pct": round(v / n, 4)} for k, v in c.most_common()}

    lens = sorted(len(s) for s in labeled + colloq)
    def pct(p: float) -> int:
        return lens[min(len(lens) - 1, int(p * len(lens)))]

    n_pool = len(labeled) + len(colloq) + sum(
        v for k, v in drop.items() if k.startswith("末字是其他标点"))
    stats = {
        "corpus": {"files_all": len(files), "files_kept": len(kept),
                   "lang": lang,
                   "kept_sorted_shuffled": [p.name for p in kept],
                   "SPLIT_SEED": SPLIT_SEED},
        "rows": {"用户话轮行数": n_user, "行级 ok": n_ok, "去重后池": n_pool,
                 "有句末标点(标签池)": len(labeled), "无句末标点(口语池)": len(colloq),
                 "丢弃": dict(drop),
                 "丢弃率_主标签池": round(1 - len(labeled) / max(1, n_pool), 4),
                 "有句末标点占比": round(len(labeled) / max(1, n_pool), 4),
                 "长度": {"p10": pct(.10), "p50": pct(.50), "p90": pct(.90)}},
        "sizes": {"labeled": sz, "colloq": csz},
        "dist_train": dist(tr), "dist_test": dist(te),
        "dist_colloq_train": dist(ctr), "dist_colloq_test": dist(cte),
        "rule_by_class": {
            "陈述/是非疑问/感叹(顶部分支)": "句末标点（遮蔽后不可见）→ 与 R-particle 不同源",
            "特指疑问": "标点 + QSET 词表 → 细分与 R-particle **同源**",
            "祈使": "标点 + IMP 词表 → 不同源（IMP 不进 R-particle）",
            "反问": "RHO 词表（不看标点）→ 与 R-particle **同源**",
            "label_full(原文)": 1.0, "label_nopunct(口语出口)": 1.0,
            "note": "以上 1.0 是**定义**不是实测发现（dev-notes/21 D4）",
        },
        "example": {k: [r["text"] for r in rows[:3]] for k, rows in
                    (("train", tr), ("test", te), ("colloq", ctr))},
    }
    DATA.mkdir(exist_ok=True)
    (DATA / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("corpus", "rows", "sizes")},
                     ensure_ascii=False, indent=2)[:4000])
    print("[dist_train]", json.dumps(stats["dist_train"], ensure_ascii=False))
    print("[dist_test ]", json.dumps(stats["dist_test"], ensure_ascii=False))
    print("[dist_coll ]", json.dumps(stats["dist_colloq_test"], ensure_ascii=False))
    print(f"[files_kept] {len(kept)}/{len(files)}: "
          + ", ".join(f"{p.name}:{lang[p.name]['cjk_ratio']}" for p in kept))
    print(f"[done] data/ train={len(tr)} test={len(te)} "
          f"colloq_train={len(ctr)} colloq_test={len(cte)}")


if __name__ == "__main__":
    main()
