#!/usr/bin/env python3
"""S87 · 临时语料适配器：用本机**现有**中文语料，冒充 S86 的 `sent_meta.json` 缓存。

⚠ **这是偏差，不是等价替换**：S86 写死读
`nanoSeek/data/chinese/clean_v3/{wikipedia_cn,c4_zh}.txt`，而本机该目录**只有报告、无语料**
（`.gitignore:34` 把 `clean_v3/*` 整目录忽略 ⇒ 从未入库）。本机现有的中文语料是
`data/chinese/*_dialogue.txt`（对话体，非百科/网页体）。

因此本脚本：
- **只写** `logs/87_cache/sent_meta.json`（S86 会照它继续 tokenize），**不碰** `nanoSeek/`；
- 复用 S86 的**同一套过滤口径**（长度 8..60 · ≥6 汉字 · 逐字可编码且逐字还原）与**同一抽样种子** `SENT_SEED=20261011`；
- 结果必须标「**语料代用**」，任何引用这些读数的结论都要写明偏差。

    python stages/87_data_alt.py --build          # 建代用语料缓存
    python stages/87_data_alt.py --check          # 只核对现有缓存
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path("/home/vesita/coding/my/flowme")
NANOSEEK = Path("/home/vesita/coding/my/nanoSeek/data/chinese")
TOKENIZER = ROOT / "logs" / "86_tokenizer" / "char_tokenizer.json"
CACHE = ROOT / "logs" / "87_cache"

# ★ 代用语料：本机实有（`ls -la` 实测，2026-10-14）
CORPUS_SUB = [
    NANOSEEK / "multi_turn_dialogue.txt",        # 133 MB
    NANOSEEK / "shareai_dialogue.txt",           # 3.3 MB
    NANOSEEK / "agent_dialogue.txt",             # 285 KB
]
BYTES_PER_FILE = 40_000_000
SENT_SEED = 20261011                  # 与 data.py 逐字相同
N_TRAIN, N_TEST = 20000, 2000

HAN = re.compile(r"[\u4e00-\u9fff]")
# ★ 代用语料是对话体：换行是天然的轮次边界，必须一起切（S86 的 SENT_SPLIT 只切标点）
SENT_SPLIT = re.compile(r"(?<=[。！？；，])|\n")


def load_tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(TOKENIZER))


def acceptable(s: str, tk) -> bool:
    """逐字复制 `experiments/meme_anchor/data.py::_acceptable`，**外加换行禁令**。

    ★ 偏差说明：S86 的原语料是百科/网页体，切句后不含换行；代用的对话语料若保留换行，
    会让 60 字上限内的句子变成多行，与 S86 的渲染规格（3 行 × 24 字）不同源 ⇒ 直接排除。
    """
    if "\n" in s or "\r" in s:
        return False
    if not (8 <= len(s) <= 60):
        return False
    if len(HAN.findall(s)) < 6:
        return False
    return tk.decode(tk.encode(s).ids) == s


def build() -> dict:
    if CACHE.exists() and (CACHE / "sent_meta.json").exists():
        log(f"[ALT] 缓存已存在，不覆盖：{CACHE/'sent_meta.json'}")
        return json.loads((CACHE / "sent_meta.json").read_text(encoding="utf-8"))

    CACHE.mkdir(parents=True, exist_ok=True)
    tk = load_tokenizer()
    raw: list[str] = []
    per_file: dict[str, int] = {}
    for p in CORPUS_SUB:
        if not p.exists():
            log(f"[ALT][WARN] 代用语料不存在，跳过：{p}")
            continue
        with open(p, encoding="utf-8", errors="ignore") as f:
            txt = f.read(BYTES_PER_FILE)
        got = [s.strip() for s in SENT_SPLIT.split(txt)]
        keep = [s for s in got if acceptable(s, tk)]
        per_file[p.name] = len(keep)
        raw += keep
        log(f"[ALT] {p.name}: 句 {len(got)} → 合格 {len(keep)}")

    seen: set[str] = set()
    uniq: list[str] = []
    dup = 0
    for s in raw:
        if s in seen:
            dup += 1
            continue
        seen.add(s)
        uniq.append(s)
    log(f"[ALT] 逐字去重：{len(raw)} → {len(uniq)}（重复 {dup}）")

    if len(uniq) < N_TRAIN + N_TEST:
        raise SystemExit(f"[ALT] 合格句不足：去重后 {len(uniq)} < {N_TRAIN + N_TEST}")

    rng = random.Random(SENT_SEED)
    rng.shuffle(uniq)
    train = uniq[:N_TRAIN]
    test = uniq[N_TRAIN:N_TRAIN + N_TEST]
    meta = {"train": train, "test": test,
            "seed": SENT_SEED, "bytes_per_file": BYTES_PER_FILE, "n_per_file": per_file,
            "corpus_substitute": [p.name for p in CORPUS_SUB if p.exists()],
            "note": "语料代用：本机无 clean_v3/{wikipedia_cn,c4_zh}.txt，改用 *_dialogue.txt；过滤口径与 SENT_SEED 与 data.py 逐字相同"}
    (CACHE / "sent_meta.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    log(f"[ALT] 落盘 {CACHE/'sent_meta.json'}：train={len(train)} test={len(test)}")
    log(f"[ALT] 样例：{train[0][:40]}")
    return meta


def log(m: str) -> None:
    print(m, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    if args.build:
        t0 = time.time()
        build()
        log(f"[ALT] 用时 {(time.time()-t0)/60:.1f} min")
    elif args.check:
        p = CACHE / "sent_meta.json"
        if not p.exists():
            log(f"❌ 无缓存：{p}")
            sys.exit(1)
        m = json.loads(p.read_text(encoding="utf-8"))
        log(f"✅ train={len(m['train'])} test={len(m['test'])} corpus={m.get('corpus_substitute')}")
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
