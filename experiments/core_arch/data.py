#!/usr/bin/env python3
"""P20 数据装载（只读复用既有实验）+ 词表 + 张量化。

只读：experiments/sentence_mode/data、experiments/entity_identity/data、
      experiments/two_channel_head/data（仅取字符清单）、experiments/sentence_mode/labels.py（仅 import）。
只写：experiments/core_arch/。
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "experiments" / "sentence_mode"))

SM_DATA = ROOT / "experiments" / "sentence_mode" / "data"
EI_DATA = ROOT / "experiments" / "entity_identity" / "data"
TCH_DATA = ROOT / "experiments" / "two_channel_head" / "data"

MAX_LEN = 128
PAD, UNK, M1, M2, CTX = 0, 1, 2, 3, 4
SPECIALS = ["<pad>", "<unk>", "<m1>", "<m2>", "<ctx>"]
N_MODE = 6
N_ENT = 2

# FUNCWORD（PREREG §1.1 写死，按字符逐一匹配）
FUNCWORD = (
    "因为所以但是但因此于是虽然尽管而且并且如果要是只要除非总之可见由此结果"
    "而则就才也都却并且或者之乎"
    "不没别非未无"
    "吗呢吧啊呀啦嘛哈难道岂何尝"
    "什么啥谁哪哪儿哪里哪些哪个怎么咋怎样如何为什么为啥为何多少几何"
    "请不要不准不许马上赶紧快点记得必须注意小心"
    "的地得着过把被从到对和与在给让使"
)
FUNCWORD_SET = frozenset(FUNCWORD)


def _jsonl(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def load_sentence_mode() -> dict[str, list[dict]]:
    return {"train": _jsonl(SM_DATA / "train.jsonl"),
            "test": _jsonl(SM_DATA / "test.jsonl")}


def load_entity() -> dict[str, list[dict]]:
    return {"train": _jsonl(EI_DATA / "train.jsonl"),
            "test": _jsonl(EI_DATA / "test.jsonl")}


def ent_input(r: dict) -> str:
    """Task T 输入格式（PREREG §2 口径判断）。"""
    return f"<m1>{r['s1']}<m2>{r['s2']}<ctx>{r['text']}"


def ent_label(r: dict) -> int:
    assert r["label"] in ("SAME", "DIFF"), r["label"]
    return 1 if r["label"] == "SAME" else 0


def s_texts(rows: list[dict], exit_key: str) -> list[str]:
    return [r["exit"][exit_key] for r in rows]


# ---------------------------------------------------------------------------
# 词表（只由训练池构建：Task S train ∪ Task T train ∪ two_channel_head train 的字符）
# ---------------------------------------------------------------------------
def build_vocab() -> tuple[dict[str, int], dict]:
    pools: dict[str, list[str]] = {}
    pools["sm_train"] = s_texts(load_sentence_mode()["train"], "mask")
    pools["ei_train"] = [ent_input(r) for r in load_entity()["train"]]
    tch = _jsonl(TCH_DATA / "train.jsonl")
    pools["tch_train"] = [r["sent"] for r in tch]

    order: list[str] = []
    seen = set(SPECIALS)
    counts: dict[str, int] = {}
    for name in ("sm_train", "ei_train", "tch_train"):
        c = 0
        for s in pools[name]:
            for ch in s:
                counts[ch] = counts.get(ch, 0) + 1
                if ch not in seen:
                    seen.add(ch)
                    order.append(ch)
                    c += 1
        counts[f"__new_from_{name}"] = c
    vocab = {ch: i for i, ch in enumerate(SPECIALS + order)}
    meta = {"vocab_size": len(vocab),
            "new_chars_sm_train": counts["__new_from_sm_train"],
            "new_chars_ei_train": counts["__new_from_ei_train"],
            "new_chars_tch_train": counts["__new_from_tch_train"],
            "tch_chars_used": sum(1 for ch in order
                                  if counts.get(ch, 0) > 0)}
    return vocab, meta


def encode(texts: list[str], vocab: dict[str, int]) -> tuple[list[list[int]], dict]:
    ids, trunc, oov_ch, tot_ch = [], 0, 0, 0
    for t in texts:
        raw = list(t)
        tot_ch += len(raw)
        keep = raw[:MAX_LEN]
        trunc += (len(raw) > MAX_LEN)
        row = [vocab.get(ch, UNK) for ch in keep]
        oov_ch += sum(1 for ch in keep if ch not in vocab)
        ids.append(row or [UNK])
    return ids, {"n": len(texts), "trunc": trunc,
                 "trunc_rate": round(trunc / max(1, len(texts)), 6),
                 "oov_char_rate": round(oov_ch / max(1, tot_ch), 6)}


def pad(ids: list[list[int]]) -> list[list[int]]:
    out = []
    for row in ids:
        out.append(row + [PAD] * (MAX_LEN - len(row)))
    return out


def build_dataset() -> dict:
    """返回全部张量所需的数据（python list，训练脚本再转 tensor）。"""
    vocab, vmeta = build_vocab()
    sm = load_sentence_mode()
    ei = load_entity()

    ds: dict = {"vocab": vocab, "vocab_meta": vmeta, "meta": {}}
    # ---- Task S ----
    for split in ("train", "test"):
        rows = sm[split]
        mask = s_texts(rows, "mask")
        keep = s_texts(rows, "keep")
        mf = s_texts(rows, "maskfinal")
        ids_mask, st = encode(mask, vocab)
        ids_keep, st_keep = encode(keep, vocab)
        ids_mf, st_mf = encode(mf, vocab)
        # 逐行零 MASK_SET 标点断言（M0 同口径）
        import labels as L  # 只读导入 sentence_mode/labels.py
        bad = sum(1 for s in mask if any(ch in L.MASK_SET for ch in s))
        assert bad == 0, f"mask 出口残留标点 {bad} 行"
        ds[f"S_{split}"] = {"ids": pad(ids_mask), "y": [r["mode"] for r in rows],
                            "ids_keep": pad(ids_keep), "ids_maskfinal": pad(ids_mf),
                            "stat": st,
                            "stat_keep": st_keep, "stat_maskfinal": st_mf,
                            "mask_punct_residual": bad}
        ds["meta"][f"S_{split}"] = st | {"n_classes": N_MODE}
    # ---- Task T ----
    for split in ("train", "test"):
        rows = ei[split]
        ids, st = encode([ent_input(r) for r in rows], vocab)
        ds[f"T_{split}"] = {"ids": pad(ids), "y": [ent_label(r) for r in rows],
                            "stat": st, "variant": [r["variant"] for r in rows],
                            "type": [r["type"] for r in rows],
                            "label_raw": [r["label"] for r in rows]}
        ds["meta"][f"T_{split}"] = st | {"n_classes": N_ENT}
    # ---- H4 probe 域（只读，不用标签训练）----
    for split in ("train", "test"):
        rows = _jsonl(TCH_DATA / f"{split}.jsonl")
        ids, st = encode([r["sent"] for r in rows], vocab)
        ds[f"P_skel_{split}"] = {"ids": pad(ids), "y": [r["skel_id"] for r in rows],
                                 "stat": st, "sent": [r["sent"] for r in rows]}
        ds["meta"][f"P_skel_{split}"] = st
    for name in ("b_pairs", "c_pairs"):
        rows = _jsonl(ROOT / "experiments" / "skeleton_leak" / "data" / f"{name}.jsonl")
        ids, st = encode([r["sent"] for r in rows], vocab)
        ds[f"P_{name}"] = {"ids": pad(ids), "y": [r["skel_id"] for r in rows],
                           "stat": st, "pair": [r.get("pair") for r in rows],
                           "kind": [r.get("kind") for r in rows]}
        ds["meta"][f"P_{name}"] = st
    # ---- M 机制自检：Task S mask 输入逐字符随机置换（seed*1000+11）----
    return ds


def shuffled_texts(texts: list[str], seed: int) -> list[str]:
    g = random.Random(seed * 1000 + 11)
    out = []
    for t in texts:
        ch = list(t)
        g.shuffle(ch)
        out.append("".join(ch))
    return out


if __name__ == "__main__":
    d = build_dataset()
    print(json.dumps({"vocab": d["vocab_meta"], "meta": d["meta"]},
                     ensure_ascii=False, indent=2))
