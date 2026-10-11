#!/usr/bin/env python3
"""中英平行语料（Tatoeba cmn-eng）的构建与缓存 —— L3-m2「中英模因锚定」的数据层。

**数据来源**（CC-BY 2.0 FR，Tatoeba）：`https://downloads.tatoeba.org/exports/per_language/`
    cmn/cmn_sentences.tsv.bz2       中文句（id \\t cmn \\t text）
    cmn/cmn-eng_links.tsv.bz2       中英翻译链接（id_cmn \\t id_eng）
    eng/eng_sentences.tsv.bz2       英文句（id \\t eng \\t text）
原始文件落在 `logs/86_tatoeba/`（不入库）；本模块产出缓存 `logs/86_cache_zhen/pairs.jsonl`。

    CUDA_VISIBLE_DEVICES="" .venv/bin/python experiments/meme_anchor/data_zh_en.py --build
"""
from __future__ import annotations

import bz2
import json
import re
from pathlib import Path

WORK = Path("/home/vesita/coding/my/flowme")
RAW = WORK / "logs" / "86_tatoeba"
CACHE = WORK / "logs" / "86_cache_zhen"

# ---- 过滤规格（写死；改动要留痕）------------------------------------------------
ZH_MIN, ZH_MAX = 8, 60          # 中文字符数（与 L3-m1 的中文规格一致，便于对照）
EN_MIN, EN_MAX = 10, 140        # 英文字符数
ZH_HAN_MIN = 6                  # 至少 6 个汉字
RATIO_LO, RATIO_HI = 1.2, 6.0   # len(en)/len(zh) 必须在带内（剔除错配对）
HAN = re.compile(r"[\u4e00-\u9fff]")
LAT = re.compile(r"[A-Za-z]")
URLISH = re.compile(r"https?://|www\.|@|\ufffd|[\ue000-\uf8ff]")
PRINTABLE_EN = re.compile(r"^[\x20-\x7e]+$")

# ---- 繁/简判定用的高频"简繁异形"标记字 -----------------------------------------
# ★ 这是**启发式**（不是完整转换表）：句中出现任一"仅繁体"字 ⇒ 判为繁体句；反之判简体。
#   用途只是把语料切成两个可比子集（附加读数：Δ 是否与字形无关），不用于清洗数据。
TRAD = set("們個這來說時會對沒麼應該開關學國語幾點覺讓與為條東馬鳥魚龍買賣錢銀鐵頭體長門問間陽際隊陸陳張導層屬島灣網線練總織統絲綠紅紙級約純組結給絕經續維綜編緣縱於後隻雙車風飛飯館廣義樂習書畫筆寫讀聽")
SIMP = set("们个这来说时会对没么应该开关学国语几点觉让与为条东马鸟鱼龙买卖钱银铁头体长门问间阳际队陆陈张导层属岛湾网线练总织统丝绿红纸级约纯组结给绝经续维综编缘纵于后只双车风飞饭馆广义乐习书画笔写读听")


def _read_sentences(path: Path, need: set[str] | None) -> dict[str, str]:
    out: dict[str, str] = {}
    with bz2.open(path, "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) < 3:
                continue
            if need is None or p[0] in need:
                out[p[0]] = p[2]
    return out


def build() -> dict:
    import sys
    sys.path.insert(0, str(WORK / "experiments" / "meme_anchor"))
    import data as D
    tk = D.load_tokenizer()
    rt = lambda s: tk.decode(tk.encode(s).ids) == s       # noqa: E731

    links = []
    with bz2.open(RAW / "cmn-eng_links.tsv.bz2", "rt", encoding="utf-8", errors="ignore") as f:
        for line in f:
            p = line.rstrip("\n").split("\t")
            if len(p) >= 2:
                links.append((p[0], p[1]))
    zh = _read_sentences(RAW / "cmn_sentences.tsv.bz2", {a for a, _ in links})
    en = _read_sentences(RAW / "eng_sentences.tsv.bz2", {b for _, b in links})

    pairs, seen_z, seen_e = [], set(), set()
    drop = dict(link=0, zh_missing=0, en_missing=0, zh_len=0, en_len=0, han=0, ratio=0,
                en_shape=0, urlish=0, roundtrip=0, dup=0)
    for iz, ie in links:
        z, e = zh.get(iz), en.get(ie)
        if z is None:
            drop["zh_missing"] += 1; continue
        if e is None:
            drop["en_missing"] += 1; continue
        if not (ZH_MIN <= len(z) <= ZH_MAX):
            drop["zh_len"] += 1; continue
        if not (EN_MIN <= len(e) <= EN_MAX):
            drop["en_len"] += 1; continue
        if len(HAN.findall(z)) < ZH_HAN_MIN:
            drop["han"] += 1; continue
        if not (RATIO_LO <= len(e) / len(z) <= RATIO_HI):
            drop["ratio"] += 1; continue
        if not PRINTABLE_EN.match(e) or len(LAT.findall(e)) < len(e) * 0.5:
            drop["en_shape"] += 1; continue
        if URLISH.search(z) or URLISH.search(e):
            drop["urlish"] += 1; continue
        # ★ 逐字往返（词表外的字 —— 全角数字/繁体异体/弯引号等 —— 会在 decode 时被丢掉）
        if not (rt(z) and rt(e)):
            drop["roundtrip"] += 1; continue
        if z in seen_z or e in seen_e:
            drop["dup"] += 1; continue
        seen_z.add(z); seen_e.add(e)
        pairs.append((z, e))

    CACHE.mkdir(parents=True, exist_ok=True)
    with (CACHE / "pairs.jsonl").open("w", encoding="utf-8") as f:
        for z, e in pairs:
            f.write(json.dumps({"zh": z, "en": e}, ensure_ascii=False) + "\n")
    return dict(links=len(links), kept=len(pairs), drop=drop)


def build_roundtrip_check() -> dict:
    """往返检查：每条必须能被项目的字符分词器逐字还原（否则该条不可用于字符级重构）。"""
    import sys
    sys.path.insert(0, str(WORK / "experiments" / "meme_anchor"))
    import data as D
    tk = D.load_tokenizer()
    bad = 0
    n = 0
    for z, e in iter_pairs():
        n += 1
        for s in (z, e):
            if tk.decode(tk.encode(s).ids) != s:
                bad += 1
                break
    return dict(checked=n, bad=bad)


def iter_pairs():
    with (CACHE / "pairs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            yield d["zh"], d["en"]


def load_pairs() -> list[tuple[str, str]]:
    """固定顺序读回（不 shuffle；切分由调用方按固定种子做）。"""
    return list(iter_pairs())


if __name__ == "__main__":
    import sys
    if "--build" in sys.argv:
        st = build()
        print(f"[ZH-EN] 链接 {st['links']} 条 ⇒ **保留 {st['kept']} 对**")
        print("[ZH-EN] 丢弃明细:", {k: v for k, v in st["drop"].items() if v})
        rt = build_roundtrip_check()
        print(f"[ZH-EN] 分词器往返检查：{rt['checked']} 对，**不往返 {rt['bad']}**")
    else:
        print(__doc__)
