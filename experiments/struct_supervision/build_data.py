#!/usr/bin/env python3
"""struct_supervision 数据：只读复用 train/test + 构建对抗集 adv1/adv2 + 词表。

口径（PREREG §2，跑前写死）：
  · 训练/测试 = `experiments/two_channel_head/data/{train,test}.jsonl` **只读复用**，不改不重建；
  · 语料入口只用 `dtseek.tasks.corpus.resolve_corpus_files`（glob 空即抛）；
    **文件级汉字占比 ≥ MIN_CJK_RATIO(0.6)**（严于 dev-notes/19 的 ≥50%，与训练数据同口径），
    **打乱文件序**（SPLIT_SEED），不依赖 `fs` 返回顺序、不取前 N 个文件；
  · 句级过滤、匹配器、GAP/SENT_LEN 全部 import 自 two_channel_head（**只读 import**，口径逐字相同）。

对抗集（两子集，都必须是训练集未出现的句式）：
  adv1 零样本骨架：gold id ∈ 训练零样本的 10 个 id；
  adv2 组合槽型  ：骨架 ∈ 训练 30 类，槽型签名在该骨架 train 中未见（L1/L2/L3 降级，记录级别）。

用法：uv run python experiments/struct_supervision/build_data.py
"""
from __future__ import annotations

import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "two_channel_head"))

from dtseek.tasks.corpus import CORPUS_GLOB, resolve_corpus_files  # noqa: E402

# 只读复用：匹配器 / 过滤器 / 朴素规则电池（口径与训练数据逐字相同）
from build_gen_data import (  # noqa: E402
    ALLOWED_LIT_CHARS, ASCII_WORD, BAD, BAG_SEED, CJK, GAP_LEN, MIN_CJK_RATIO,
    PUNCT_NORM, ROLE_PREFIX, SENT_LEN, SENT_SPLIT, SPLIT_SEED, TRAIL_WEAK,
    check, match_sentence, naive_assign, naive_skeleton, ok_sentence,
)

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
TCH_DIR = ROOT / "experiments" / "two_channel_head"
TCH_DATA = TCH_DIR / "data"

CAP_ADV1_PER_ID = 150      # adv1 每个零样本骨架最多取多少句
CAP_ADV2 = 1000            # adv2 目标句数
ADV2_PREF = ("L1", "L2", "L3")

# ---------------------------------------------------------------------------
# 槽型分类（PREREG §2 写死：疑问 > 否定 > 数字 > 时间 > 其他，首个命中）
# ---------------------------------------------------------------------------
RE_Q = re.compile(r"什么|怎么|怎样|如何|为什么|为何|是否|哪|多少|几")
RE_NEG = re.compile(r"不|没|未|别|无|非")
RE_NUM = re.compile(r"[0-9０-９]|[一二三四五六七八九十百千万亿]")
RE_TIME = re.compile(r"年|月|日|天|周|星期|小时|分钟|秒|现在|今天|明天|昨天|前天|"
                     r"早上|上午|中午|下午|晚上|夜里|当年|时代|时期|季节")
CLASS_ORDER = ("Q", "N", "M", "T", "O")


def slot_class(t: str) -> str:
    if RE_Q.search(t):
        return "Q"
    if RE_NEG.search(t):
        return "N"
    if RE_NUM.search(t):
        return "M"
    if RE_TIME.search(t):
        return "T"
    return "O"


def signature(gaps: list[str]) -> tuple[str, ...]:
    return tuple(slot_class(g) for g in gaps)


def row_signature(r: dict) -> tuple[str, ...]:
    """**句序**槽文本的类型签名（train 行的 bag 是洗牌的 ⇒ 必须按 span 排回句序）。"""
    spans = sorted(r["bag_span"], key=lambda x: x[0])
    return tuple(slot_class(r["sent"][a:b]) for a, b in spans)


def check(cond: bool, msg: str) -> None:  # 覆盖 import 的 check，加上本目录前缀
    if not cond:
        raise AssertionError(f"[struct_supervision build fail-closed] {msg}")


# ---------------------------------------------------------------------------
# 只读读取 train/test
# ---------------------------------------------------------------------------
def load_rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


# ---------------------------------------------------------------------------
# 语料全量扫描（同口径过滤 + 打乱文件序）
# ---------------------------------------------------------------------------
def scan_corpus(zero_ids: set[int], trained_ids: set[int],
                train_sigs: dict[int, set], train_cls: dict[int, set],
                global_cls: set, exclude: set[str]) -> tuple[list, list, dict]:
    files = resolve_corpus_files(CORPUS_GLOB)
    lang: dict[str, dict] = {}
    kept: list[str] = []
    rnd = random.Random(SPLIT_SEED)
    for p in files:
        txt = Path(p).read_text(encoding="utf-8", errors="ignore")
        ratio = len(CJK.findall(txt)) / max(1, len(txt))
        lang[Path(p).name] = {"cjk_ratio": round(ratio, 4), "kept": ratio >= MIN_CJK_RATIO}
        if ratio >= MIN_CJK_RATIO:
            kept.append(p)
    check(kept, "语言过滤后一个语料文件都不剩")
    rnd.shuffle(kept)                      # 打乱文件序，不依赖 fs 顺序

    adv1: list[dict] = []
    adv2: list[dict] = []
    seen: set[str] = set()
    n_sent = n_match = 0
    procs = Counter()
    for p in kept:
        name = Path(p).name
        for line in Path(p).read_text(encoding="utf-8", errors="ignore").splitlines():
            t = ROLE_PREFIX.sub("", line.strip()).translate(PUNCT_NORM)
            for s in SENT_SPLIT.split(t):
                s = s.strip().strip(TRAIL_WEAK)
                if not ok_sentence(s) or s in seen or s in exclude:
                    continue
                seen.add(s)
                n_sent += 1
                # 便宜预筛：句子不含任何合法字面字符 ⇒ 不可能命中任何骨架
                if not (set(s) & ALLOWED_LIT_CHARS):
                    continue
                m = match_sentence(s)
                if m is None:
                    continue
                n_match += 1
                sid, gaps, spans = m
                if sid in zero_ids:
                    adv1.append({"sent": s, "skel_id": sid, "gaps": gaps,
                                 "spans": spans, "file": name})
                    procs[f"adv1/{sid}"] += 1
                elif sid in trained_ids:
                    sig = signature(gaps)
                    if sig in train_sigs[sid]:
                        continue
                    level = None
                    if set(sig) <= train_cls[sid]:
                        level = "L1"
                    elif set(sig) <= global_cls:
                        level = "L2"
                    else:
                        level = "L3"
                    adv2.append({"sent": s, "skel_id": sid, "gaps": gaps,
                                 "spans": spans, "file": name, "sig": list(sig),
                                 "level": level})
                    procs[f"adv2/{sid}"] += 1
        print(f"[scan] {name} done（累计句子 {n_sent:,}，命中 {n_match:,}，"
              f"adv1 {len(adv1)}, adv2 {len(adv2)}）", flush=True)
    lang_stats = {"files_total": len(lang), "files_kept": sorted(
        k for k, v in lang.items() if v["kept"]),
        "files_dropped": {k: v["cjk_ratio"] for k, v in lang.items() if not v["kept"]},
        "min_cjk_ratio": MIN_CJK_RATIO, "shuffle_seed": SPLIT_SEED,
        "sentences_considered": n_sent, "matched": n_match}
    return adv1, adv2, {"lang": lang_stats, "procs": dict(procs)}


# ---------------------------------------------------------------------------
# 采样 + 落盘
# ---------------------------------------------------------------------------
def to_rows(cands: list[dict], split: str) -> list[dict]:
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    srng = random.Random(BAG_SEED + {"adv1": 11, "adv2": 12}[split])
    rows = []
    for c in cands:
        gaps, spans = c["gaps"], c["spans"]
        n = len(gaps)
        perm = list(range(n))
        srng.shuffle(perm)
        bag = [gaps[i] for i in perm]
        bag_span = [spans[i] for i in perm]
        assign = [perm.index(slot) for slot in range(n)]
        e = tok.encode(c["sent"], max_length=64, padding=False)
        check(sum(e["attention_mask"]) == len(c["sent"]),
              f"截断丢弃：{c['sent']}")
        rows.append({"sent": c["sent"], "skel_id": c["skel_id"],
                     "skel": c.get("skel", ""), "n_slots": n, "bag": bag,
                     "bag_span": bag_span, "assign": assign, "split": split,
                     **({"sig": c["sig"], "level": c["level"]} if split == "adv2" else {})})
    for r in rows:
        check([r["sent"][a:b] for a, b in r["bag_span"]] == r["bag"],
              "span 与 bag 文本不一致")
        check(all(GAP_LEN[0] <= len(g) <= GAP_LEN[1] for g in r["bag"]), "槽长越界")
    return rows


def main() -> None:
    train = load_rows(TCH_DATA / "train.jsonl")
    test = load_rows(TCH_DATA / "test.jsonl")
    tr_cnt = Counter(r["skel_id"] for r in train)
    trained_ids = set(tr_cnt)
    zero_ids = {i for i in range(40) if tr_cnt[i] == 0}
    print(f"[train] n={len(train)} 类数={len(trained_ids)} "
          f"零样本 id={sorted(zero_ids)}（n={len(zero_ids)}）", flush=True)
    check(len(zero_ids) == 10, f"零样本骨架 id 应为 10 个，实得 {len(zero_ids)}")

    # 槽型签名（train 口径）
    train_sigs: dict[int, set] = defaultdict(set)
    train_cls: dict[int, set] = defaultdict(set)
    global_cls: set = set()
    for r in train:
        sig = row_signature(r)          # 句序（与 adv 候选的 gaps 口径一致）
        train_sigs[r["skel_id"]].add(sig)
        train_cls[r["skel_id"]].update(sig)
        global_cls.update(sig)
    print(f"[sig] train 签名数={sum(len(v) for v in train_sigs.values())} "
          f"全局槽型类={sorted(global_cls)}", flush=True)

    adv1_c, adv2_c, meta = scan_corpus(zero_ids, trained_ids, train_sigs,
                                       train_cls, global_cls,
                                       {r["sent"] for r in train} |
                                       {r["sent"] for r in test})
    print(f"[adv-cand] adv1={len(adv1_c)} adv2={len(adv2_c)}", flush=True)
    check(adv1_c, "adv1 候选为 0（零样本骨架在全语料里找不到句子）")
    check(adv2_c, "adv2 候选为 0")

    # ---- adv1：每 id 限额、确定性采样 ----
    by_id: dict[int, list] = defaultdict(list)
    for c in adv1_c:
        by_id[c["skel_id"]].append(c)
    rnd = random.Random(SPLIT_SEED + 99)
    adv1_pick = []
    for sid in sorted(by_id):
        v = sorted(by_id[sid], key=lambda c: c["sent"])
        rnd.shuffle(v)
        adv1_pick += v[:CAP_ADV1_PER_ID]
    # ---- adv2：按级别优先 + 按骨架分层 ----
    lv: dict[str, dict[int, list]] = {"L1": defaultdict(list), "L2": defaultdict(list),
                                      "L3": defaultdict(list)}
    for c in adv2_c:
        lv[c["level"]][c["skel_id"]].append(c)
    adv2_pick: list = []
    for level in ADV2_PREF:
        if len(adv2_pick) >= CAP_ADV2:
            break
        remain = CAP_ADV2 - len(adv2_pick)
        pool = []
        for sid in sorted(lv[level]):
            v = sorted(lv[level][sid], key=lambda c: c["sent"])
            rnd.shuffle(v)
            pool += v
        rnd.shuffle(pool)
        take = pool[:remain]
        adv2_pick += take
        print(f"[adv2] 级别 {level} 取 {len(take)}（候选 {sum(len(x) for x in lv[level].values())}）",
              flush=True)

    skel_tpl = {int(k): v for k, v in
                json.loads((TCH_DATA / "stats.json").read_text(encoding="utf-8"))
                ["skeletons"].items()}
    for c in adv1_pick + adv2_pick:
        c["skel"] = skel_tpl[c["skel_id"]]

    rows1 = to_rows(adv1_pick, "adv1")
    rows2 = to_rows(adv2_pick, "adv2")

    # ---- 断言 ----
    tr_sents = {r["sent"] for r in train}
    te_sents = {r["sent"] for r in test}
    both = tr_sents | te_sents
    for name, rows in (("adv1", rows1), ("adv2", rows2)):
        ov = {r["sent"] for r in rows} & both
        check(not ov, f"{name} 与 train/test 句子重叠 {len(ov)}")
    check(all(r["skel_id"] in zero_ids for r in rows1), "adv1 含非零样本骨架")
    check(all(tr_cnt[r["skel_id"]] == 0 for r in rows1), "adv1 的骨架在 train 中非零样本")
    for r in rows2:
        check(tuple(r["sig"]) not in train_sigs[r["skel_id"]],
              f"adv2 签名在 train 中出现过：{r['sent']}")
        check(r["skel_id"] in trained_ids, "adv2 含非训练骨架")

    # ---- 词表 ----
    chars = sorted({ch for r in train + test + rows1 + rows2 for ch in r["sent"]})
    vocab = {"PAD": 0, "BOS": 1, "EOS": 2}
    for ch in chars:
        vocab[ch] = len(vocab)
    oov = [ch for r in train + test + rows1 + rows2 for ch in r["sent"] if ch not in vocab]
    check(not oov, f"词表覆盖失败 OOV={len(oov)}")

    # ---- 朴素规则电池（fit = train）----
    naive = {
        "adv1_skel": {k: round(v, 4) for k, v in naive_skeleton(train, rows1).items()},
        "adv2_skel": {k: round(v, 4) for k, v in naive_skeleton(train, rows2).items()},
        "adv1_assign": {k: round(v, 4) for k, v in naive_assign(train, rows1).items()},
        "adv2_assign": {k: round(v, 4) for k, v in naive_assign(train, rows2).items()},
    }

    stats = {
        "prereg": "PREREG.md §2",
        "corpus": meta["lang"],
        "reused": {"train": len(train), "test": len(test),
                   "path": str(TCH_DATA)},
        "zero_shot_ids": {str(i): skel_tpl[i] for i in sorted(zero_ids)},
        "adv1": {"n": len(rows1),
                 "per_id": dict(sorted(Counter(r["skel_id"] for r in rows1).items())),
                 "slots": dict(sorted(Counter(r["n_slots"] for r in rows1).items())),
                 "files": dict(Counter(c["file"] for c in adv1_pick)),
                 "samples": [r["sent"] for r in rows1[:3]]},
        "adv2": {"n": len(rows2),
                 "levels_used": dict(Counter(r["level"] for r in rows2)),
                 "candidates_by_level": {k: sum(len(x) for x in lv[k].values())
                                         for k in ADV2_PREF},
                 "per_id": dict(sorted(Counter(r["skel_id"] for r in rows2).items())),
                 "slots": dict(sorted(Counter(r["n_slots"] for r in rows2).items())),
                 "sig_dist": dict(Counter(",".join(r["sig"]) for r in rows2)),
                 "samples": [r["sent"] for r in rows2[:3]]},
        "vocab": {"n_chars": len(vocab), "pad/bos/eos": [0, 1, 2]},
        "naive": naive,
        "overlap": {"adv1∩(train|test)": 0, "adv2∩(train|test)": 0},
        "procs": meta["procs"],
    }

    DATA.mkdir(exist_ok=True)
    for name, rows in (("adv1", rows1), ("adv2", rows2)):
        with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    (DATA / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False),
                                     encoding="utf-8")
    (DATA / "stats_adv.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(f"[done] adv1 n={len(rows1)} adv2 n={len(rows2)} vocab={len(vocab)} → {DATA}",
          flush=True)
    print(f"[naive] adv1 骨架 {naive['adv1_skel']}", flush=True)
    print(f"[naive] adv2 骨架 {naive['adv2_skel']}", flush=True)
    print(f"[naive] adv1 指派 {naive['adv1_assign']}", flush=True)
    print(f"[naive] adv2 指派 {naive['adv2_assign']}", flush=True)


if __name__ == "__main__":
    main()
