#!/usr/bin/env python3
"""选择重排（select_rerank）数据集构建，fail-closed。

两臂（每臂 train/test/adv 三个集合，七个源句池两两不相交）：
  臂 S（data/shortcut/）  train/test：正 = 上下文逐字片段、负 = 别句无关片段 ⇒ 逐字规则 100%（复刻 anchored_select）
  臂 N（data/clean/）     train/test：正 = 片段某词换同义词、负 = 同位置换反义词 ⇒ 两候选都不是逐字片段、逐字规则 50%

adv（**两臂同构造**，只换源句池；正 = 同义替换、负 = 反义替换，两 ctype 各半）：
  heldout_pair（换说法）：留出**替换对** —— 词头取自训练词头（保证命中率），
                          两个输出串与训练词表（词头∪同义∪反义）**零交集**（构造期断言）
  shifted_pos （换位置）：训练词表，但被替换的词落在候选**尾部**（train/test 在**前部**）

不变量（构造期断言，任一不满足即抛错）：
  · 候选顺序随机化且两位置各恰 50%（否则位置是捷径）
  · 臂 N 两候选与原词**等长**、都**不是**上下文逐字片段
  · 词表共享汉字状态对称 share(w,syn)==share(w,ant) ⇒「共享字⇒正例」零预测力
  · 逐样本截断守卫：与训练同口径 encode，未截断才留
  · 七池两两重叠 = 0；六集两两 key/context/candidate 重叠 = 0；候选字符串全局唯一

零训练表面基线（stats 必报）：朴素逐字规则 / 长度规则 / align_share_rule / ctx_char_overlap_rule。

语料入口只用 dtseek.tasks.corpus.resolve_corpus_files（glob 空即抛错，不绕过）。

用法：
    uv run python experiments/select_rerank/build_data.py --probe   # 量命中率与构造成功率 → 池子建议
    uv run python experiments/select_rerank/build_data.py           # 落 data/{shortcut,clean}/*.jsonl + data/stats.json
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

from dtseek.tasks.corpus import CORPUS_GLOB, resolve_corpus_files  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SPLIT_SEED = 20240928          # 与 anchored_select(20240927) 不同 ⇒ 源句池独立

#: 截断守卫口径（**与训练同口径**：train_rerank / model 直接 import 这个 dict）
SPEC = {"max_len_ctx": 64, "max_len_cand": 32}

N_TRAIN, N_TEST, N_ADV = 8000, 2500, 2500
#: 池规模按 --probe 出的**构造成功率**算（probe 打 SUGGEST，此处填 1.2 倍余量后的值）。
POOL = {
    "s_train": 9000, "s_test": 3200, "donor": 3000,
    "s_adv": 30000, "n_train": 70000, "n_test": 22000, "n_adv": 30000,
}
NEED_SENT = sum(POOL.values()) + 2000
PER_FILE = 25000                # 每文件最多取句数（供给上限见 collect 的 fail-closed 报错）

SEP = "|"
SENT_SPLIT = re.compile(r"[。！？\n；;]+")
ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
CJK = re.compile(r"[一-龥]")
BAD = re.compile(r"[�\t｜|]|原文：|候选：")
PUNCT_TAIL = set("，。！？、；：\"'”’）)】》…~—-,.?!:;")
PUNCT_HEAD = set("，。！？、；：\"'“‘（(【《…~—-,.?!:;")

# ---------------------------------------------------------------------------
# 词表：三元组 (原词, 同义替换, 反义替换)
# 硬约束（validate_tables 断言）：三者等长、互异、词头唯一、
#   share(w,syn) == share(w,ant) —— 否则零训练规则「替换处仍共享一个字 ⇒ 判正」有预测力。
# ---------------------------------------------------------------------------
TRAIN_PAIRS = [
    ("高兴", "快乐", "难过"), ("美丽", "好看", "丑陋"), ("快速", "敏捷", "缓慢"),
    ("简单", "容易", "复杂"), ("安全", "可靠", "危险"), ("安静", "悄然", "吵闹"),
    ("诚实", "忠厚", "虚伪"), ("支持", "赞成", "反对"), ("同意", "应允", "拒绝"),
    ("接受", "采纳", "放弃"), ("记得", "牢记", "忘记"), ("表扬", "夸奖", "批评"),
    ("喜欢", "热衷", "讨厌"), ("勤奋", "刻苦", "懒惰"), ("完整", "齐全", "残缺"),
    ("富裕", "丰足", "贫穷"), ("强大", "威猛", "弱小"), ("正确", "无误", "错误"),
    ("增加", "扩充", "减少"), ("上升", "抬高", "下降"), ("前进", "迈步", "后退"),
    ("开始", "启动", "结束"), ("出现", "产生", "消失"), ("干净", "清爽", "肮脏"),
    ("温暖", "热乎", "寒冷"), ("热闹", "喧哗", "冷清"), ("积极", "主动", "消沉"),
    ("肯定", "确定", "否定"), ("进步", "提高", "退化"), ("帮助", "支援", "妨碍"),
    ("信任", "托付", "怀疑"), ("优点", "长处", "短板"), ("团结", "齐心", "分裂"),
    ("稳定", "牢靠", "动荡"), ("清楚", "明白", "模糊"), ("热情", "亲切", "冷淡"),
    ("宽敞", "开阔", "狭窄"), ("干燥", "缺水", "潮湿"), ("聪明", "机灵", "愚蠢"),
    ("勇敢", "无畏", "胆怯"), ("认真", "仔细", "马虎"), ("漂亮", "美观", "难看"),
    ("重要", "关键", "琐碎"), ("困难", "棘手", "容易"), ("购买", "添置", "出售"),
    ("温和", "柔顺", "粗暴"), ("舒适", "惬意", "难受"), ("方便", "顺手", "麻烦"),
    ("明显", "突出", "隐蔽"), ("真实", "确凿", "虚假"), ("成功", "顺利", "失败"),
    ("懒散", "松垮", "勤快"), ("礼貌", "文雅", "粗鲁"), ("公平", "均等", "偏心"),
    ("轻松", "安闲", "紧张"), ("密切", "贴近", "疏远"), ("详细", "具体", "粗略"),
    ("丰富", "充足", "匮乏"), ("珍贵", "稀罕", "廉价"), ("晴朗", "明媚", "阴沉"),
    ("宽裕", "富余", "短缺"), ("陡峭", "险峻", "平缓"), ("嘈杂", "喧嚣", "宁静"),
    ("平淡", "索然", "精彩"), ("宽恕", "原谅", "计较"), ("犹豫", "迟疑", "果断"),
    ("制止", "劝阻", "允许"), ("允许", "答应", "禁止"), ("刺耳", "扎耳", "悦耳"),
    ("熟悉", "了解", "陌生"), ("生气", "恼火", "开心"), ("讨厌", "反感", "喜欢"),
    ("担心", "挂心", "放心"), ("着急", "慌忙", "镇定"), ("平静", "安宁", "激动"),
    ("愉快", "欣喜", "悲伤"), ("失望", "灰心", "期待"), ("满意", "舒心", "窝火"),
    ("理解", "领会", "误会"), ("关心", "体贴", "冷落"), ("孤独", "落单", "热闹"),
    ("普通", "平凡", "特别"), ("懒惰", "懈怠", "勤劳"), ("胆小", "怯懦", "勇敢"),
    ("勤劳", "苦干", "懒惰"),
    ("果断", "坚决", "犹豫"), ("认为", "觉得", "否定"), ("可能", "或许", "必定"),
    ("结束", "收尾", "开始"), ("马上", "立刻", "慢慢"), ("赶紧", "尽快", "慢慢"),
    ("全部", "所有", "少量"), ("一定", "必然", "未必"), ("一起", "结伴", "分开"),
    ("不同", "有别", "一样"),
]

#: 留出**替换对**（对抗集 heldout_pair 专用）：词头取自训练词头，但 (同义, 反义) 这对**说法**
#: 训练从未见过 —— 两个输出串与训练词表零交集（断言），于是「背训练词表」在对抗集给不出答案。
HOLDOUT_PAIRS = [
    ("高兴", "喜悦", "痛苦"), ("简单", "好办", "繁杂"), ("安全", "稳妥", "危机"),
    ("喜欢", "倾心", "厌恶"), ("漂亮", "标致", "碍眼"), ("干净", "卫生", "污秽"),
    ("聪明", "机智", "愚笨"), ("安静", "沉寂", "喧闹"), ("勤奋", "用功", "偷闲"),
    ("记得", "想起", "忘怀"), ("增加", "提升", "降低"), ("上升", "攀高", "下沉"),
    ("出现", "露头", "隐匿"), ("开始", "起头", "收场"), ("结束", "告终", "开端"),
    ("正确", "精准", "荒谬"), ("表扬", "称许", "指责"), ("支持", "拥护", "拆台"),
    ("接受", "收下", "谢绝"), ("讨厌", "腻烦", "钟情"), ("困难", "费劲", "顺当"),
    ("舒适", "安逸", "拘束"), ("方便", "省事", "费事"), ("明显", "昭著", "隐晦"),
    ("真实", "确切", "虚幻"), ("热情", "好客", "冷漠"), ("认真", "用心", "敷衍"),
    ("勤劳", "努力", "好逸"), ("诚实", "坦白", "奸诈"), ("勇敢", "大胆", "怯场"),
    ("礼貌", "客套", "蛮横"), ("公平", "正当", "徇私"), ("富裕", "丰饶", "拮据"),
    ("强大", "雄壮", "衰弱"), ("积极", "起劲", "颓废"), ("稳定", "牢固", "飘摇"),
    ("宽敞", "空旷", "拥挤"), ("清楚", "明了", "恍惚"), ("团结", "合力", "对立"),
    ("完整", "无缺", "破损"), ("温和", "斯文", "火爆"), ("成功", "得手", "搁浅"),
    ("详细", "周密", "简略"), ("丰富", "充盈", "贫乏"), ("晴朗", "明净", "阴霾"),
    ("快速", "迅捷", "迟钝"), ("同意", "默许", "不依"), ("懒散", "拖沓", "振作"),
    ("热闹", "欢腾", "死寂"), ("生气", "烦躁", "舒畅"), ("着急", "发慌", "安然"),
    ("平静", "安详", "慌乱"), ("失望", "落寞", "如愿"),
]


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[select_rerank build fail-closed] {msg}")


def share(w: str, x: str) -> bool:
    """替换串与原词是否共享汉字。share(w,syn) != share(w,ant) 时，零训练规则
    「替换处仍与原词共享一个字 ⇒ 判正」就有预测力 ⇒ 全表必须对称。"""
    return bool(set(w) & set(x))


def validate_tables() -> dict:
    def one(name: str, pairs: list[tuple[str, str, str]]) -> set[str]:
        seen_w: set[str] = set()
        for w, syn, ant in pairs:
            check(len(w) == len(syn) == len(ant), f"{name} 不等长：{(w, syn, ant)}")
            check(len({w, syn, ant}) == 3, f"{name} 三者有重复：{(w, syn, ant)}")
            check(w not in seen_w, f"{name} 词头重复：{w}")
            check(share(w, syn) == share(w, ant),
                  f"{name} 共享汉字状态不对称（「共享字⇒正例」捷径）：{(w, syn, ant)}")
            seen_w.add(w)
        return seen_w

    tr_w = one("TRAIN_PAIRS", TRAIN_PAIRS)
    ho_w = one("HOLDOUT_PAIRS", HOLDOUT_PAIRS)
    train_all = {s for p in TRAIN_PAIRS for s in p}
    ho_out = {t for p in HOLDOUT_PAIRS for t in p[1:]}      # 只看两个输出串（词头本就来自训练）
    check(ho_w <= tr_w, f"留出对词头不在训练词头里：{sorted(ho_w - tr_w)}")
    check(not (ho_out & train_all),
          f"留出对的输出串出现在训练词表里：{sorted(ho_out & train_all)}")
    return {"n_train_pairs": len(TRAIN_PAIRS), "n_holdout_pairs": len(HOLDOUT_PAIRS),
            "train_strings": len(train_all), "holdout_output_strings": len(ho_out),
            "share_asymmetric_pairs": 0}


TRAIN_IDX = {w: (syn, ant) for w, syn, ant in TRAIN_PAIRS}
HOLDOUT_IDX = {w: (syn, ant) for w, syn, ant in HOLDOUT_PAIRS}


# ---------------------------------------------------------------------------
# 语料 → 源句池
# ---------------------------------------------------------------------------
def cjk(s: str) -> int:
    return len(CJK.findall(s))


def ok_sentence(s: str) -> bool:
    if not (14 <= len(s) <= 44):
        return False
    if BAD.search(s) or SEP in s:
        return False
    if cjk(s) < 0.6 * len(s):
        return False
    if re.search(r"[A-Za-z]{3,}", s):
        return False
    return True


def iter_sentences(files: list[str], per_file: int):
    """按 SPLIT_SEED 打乱文件序后逐行读，每个文件最多取 per_file 个**可用句**。"""
    rnd = random.Random(SPLIT_SEED)
    rnd.shuffle(files)
    for p in files:
        got = 0
        with open(p, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                t = ROLE_PREFIX.sub("", line.strip())
                for s in SENT_SPLIT.split(t):
                    s = s.strip()
                    if not ok_sentence(s):
                        continue
                    yield s
                    got += 1
                if got >= per_file:
                    break


def collect_sentences(n: int, per_file: int = PER_FILE) -> list[str]:
    files = resolve_corpus_files(CORPUS_GLOB)          # fail-closed：glob 空直接抛
    seen: set[str] = set()
    out: list[str] = []
    for s in iter_sentences(files, per_file):
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
        if len(out) >= n:
            break
    check(len(out) >= n,
          f"语料可用句子不足：{len(out)} < {n}。上限 ≈ 文件数×per_file({per_file})；"
          f"先跑 --probe 看供给，再调 POOL / PER_FILE（不许降低判据所需规模）。")
    return out


# ---------------------------------------------------------------------------
# 片段与替换
# ---------------------------------------------------------------------------
def frag_ok(f: str, lo: int = 8) -> bool:
    if len(f) < lo:
        return False
    if f[0] in PUNCT_HEAD or f[-1] in PUNCT_TAIL:
        return False
    if BAD.search(f) or SEP in f:
        return False
    if cjk(f) < 0.6 * len(f):
        return False
    if re.search(r"[A-Za-z]{3,}", f):
        return False
    return True


def pick_fragment(sent: str, rng: random.Random, lo: int = 8, hi: int = 18,
                  tries: int = 50) -> str | None:
    """臂 S plain 用：从句子里取一个连贯片段（不锚定词表词）。"""
    if len(sent) <= lo:
        return None
    for _ in range(tries):
        start = rng.randrange(0, len(sent) - lo + 1)
        end = start + rng.randint(lo, min(hi, len(sent) - start))
        if end - start < lo:
            continue
        f = sent[start:end]
        if f == sent:
            continue
        if not frag_ok(f, lo):
            continue
        return f
    return None


def occurrences(sent: str, table: dict) -> list[tuple[int, str]]:
    return [(i, sent[i:i + 2]) for i in range(len(sent) - 1) if sent[i:i + 2] in table]


def window_around(sent: str, i: int, wlen: int, rng: random.Random,
                  mode: str, lo: int = 8, hi: int = 24, tries: int = 120):
    """取一个含 [i, i+wlen) 的片段窗口。

    mode="front"：被替换的词落在片段**前部**（距头部 0~2 字）—— train/test 与 heldout_pair；
    mode="back" ：被替换的词落在片段**尾部**（距尾部 0~2 字）—— shifted_pos（换位置）。
    """
    for _ in range(tries):
        if mode == "front":
            start = max(0, i - rng.randint(0, min(2, i)))
            ln_min = max(lo, i + wlen - start)
            ln_max = min(hi, len(sent) - start)
            if ln_min > ln_max:
                continue
            ln = rng.randint(ln_min, ln_max)
            end = start + ln
        else:
            end = min(len(sent), i + wlen + rng.randint(0, 2))
            ln_min = max(lo, wlen)
            ln_max = min(hi, end)
            if ln_min > ln_max:
                continue
            ln = rng.randint(ln_min, ln_max)
            start = end - ln
            if start < 0 or start > i:
                continue
        if not (start <= i and i + wlen <= end <= len(sent)):
            continue
        f = sent[start:end]
        if f == sent:
            continue
        if not frag_ok(f, lo):
            continue
        return f, i - start
    return None, None


def substitute(frag: str, pos: int, word: str, repl: str) -> str:
    check(frag[pos:pos + len(word)] == word, f"替换位不对：{frag} @{pos} ≠ {word}")
    return frag[:pos] + repl + frag[pos + len(word):]


# ---------------------------------------------------------------------------
# 单条样本
# ---------------------------------------------------------------------------
def make_row(ctype: str, arm: str, ctx: str, donor_pool: list[str],
             rng: random.Random) -> dict | None:
    """构造一条样本；返回 {pos, neg, sub, frag}，构造失败返回 None（不消耗配额）。"""
    if arm == "shortcut" and ctype == "plain":
        frag = pick_fragment(ctx, rng)
        if frag is None:
            return None
        pos = frag
        for _ in range(40):
            don = donor_pool[rng.randrange(len(donor_pool))]
            neg = pick_fragment(don, rng)
            if neg is not None and neg not in ctx and neg != pos:
                return {"pos": pos, "neg": neg, "sub": None, "frag": frag}
        return None

    if ctype == "heldout_pair":
        table, mode = HOLDOUT_IDX, "front"
    elif ctype == "shifted_pos":
        table, mode = TRAIN_IDX, "back"
    else:                                   # plain：训练词表、前部
        table, mode = TRAIN_IDX, "front"

    occ = occurrences(ctx, table)
    if not occ:
        return None
    i, w = rng.choice(occ)
    syn, ant = table[w]
    frag, p = window_around(ctx, i, len(w), rng, mode)
    if frag is None:
        return None
    pos = substitute(frag, p, w, syn)
    neg = substitute(frag, p, w, ant)
    if pos == frag or neg == frag or pos == neg or len(pos) != len(neg):
        return None
    if pos in ctx or neg in ctx:            # 两候选都不是逐字片段（臂 N 全部 + 两臂 adv）
        return None
    return {"pos": pos, "neg": neg, "sub": w, "frag": frag}


def flag_list(n: int, kinds: list[tuple[str, int]], rng: random.Random) -> list[str]:
    flags: list[str] = []
    for k, q in kinds:
        flags += [k] * q
    check(len(flags) == n, f"ctype 配额 {len(flags)} != {n}")
    rng.shuffle(flags)
    return flags


def truncated(tok, text: str, max_len: int) -> bool:
    """逐样本截断守卫：与训练同口径 encode 实际编码，并与未截断全长比对。"""
    full = len(tok.tokenizer.encode(text).ids)
    out = tok.encode(text, max_length=max_len, padding=False)
    check(len(out["input_ids"]) == min(full, max_len), "encode 行为与预期不符")
    return full > max_len


def build_split(arm: str, split: str, n: int, kinds: list[tuple[str, int]],
                pool: list[str], donor_pool: list[str], rng: random.Random,
                used_cand: set[str], used_key: set[str], tok) -> tuple[list[dict], dict]:
    """按 ctype 配额造一个集合；候选顺序随机化且**两位置各恰一半**（顺序旗标预生成）。"""
    flags = flag_list(n, kinds, rng)
    order_flags = [1] * (n // 2) + [0] * (n - n // 2)
    rng.shuffle(order_flags)
    order = list(pool)
    rng.shuffle(order)
    ptr = 0
    attempts = 0
    rows: list[dict] = []
    dropped = {"construct": 0, "lex_guard": 0, "trunc_ctx": 0, "trunc_cand": 0, "dupe": 0}

    while len(rows) < n:
        ctype = flags[len(rows)]
        attempts += 1
        check(attempts <= n * 100,
              f"{arm}/{split} 构造失败过多（{attempts} 次只造出 {len(rows)}/{n}）"
              f" —— 池子太小或命中率低于预期，先跑 --probe")
        ctx = order[ptr % len(order)]
        ptr += 1
        got = make_row(ctype, arm, ctx, donor_pool, rng)
        if got is None:
            dropped["construct"] += 1
            continue
        pos, neg = got["pos"], got["neg"]
        if arm == "shortcut" and ctype == "plain":
            if pos not in ctx or neg in ctx or pos == neg:
                dropped["lex_guard"] += 1
                continue
        else:
            if pos in ctx or neg in ctx or pos == neg:
                dropped["lex_guard"] += 1
                continue
        if truncated(tok, ctx, SPEC["max_len_ctx"]):
            dropped["trunc_ctx"] += 1
            continue
        if truncated(tok, pos, SPEC["max_len_cand"]) or truncated(tok, neg, SPEC["max_len_cand"]):
            dropped["trunc_cand"] += 1
            continue
        if pos in used_cand or neg in used_cand:
            dropped["dupe"] += 1
            continue
        swap = bool(order_flags[len(rows)])
        cands = [neg, pos] if swap else [pos, neg]
        label = 1 if swap else 0            # label = **正例**所在下标（cands[1] 是 pos 当且仅当 swap）
        key = ctx + "\x1e" + "\x1e".join(sorted(cands))
        check(cands[label] == pos and cands[1 - label] == neg,
              f"label 不是正例下标：label={label}")
        if key in used_key:
            dropped["dupe"] += 1
            continue
        used_cand.update(cands)
        used_key.add(key)
        rows.append({"id": f"{arm}/{split}/{len(rows):05d}", "arm": arm, "split": split,
                     "ctype": ctype, "context": ctx, "candidates": cands, "label": label,
                     "sub_word": got["sub"], "frag": got["frag"], "key": key,
                     "attempts": attempts})

    lab = Counter(r["label"] for r in rows)
    check(lab[0] == n // 2 and lab[1] == n - n // 2,
          f"{arm}/{split} 顺序不平衡：{dict(lab)}（位置会变成捷径）")
    return rows, dropped


# ---------------------------------------------------------------------------
# 零训练表面基线
# ---------------------------------------------------------------------------
def naive_rule(rows: list[dict]) -> float:
    """候选 ∈ 上下文 ⇒ 判正；恰好一侧命中才选它，否则（双命中/双不中）固定选下标 0。"""
    hit = 0
    for r in rows:
        hits = [i for i, c in enumerate(r["candidates"]) if c in r["context"]]
        pred = hits[0] if len(hits) == 1 else 0
        hit += int(pred == r["label"])
    return hit / max(1, len(rows))


def length_rule(rows: list[dict]) -> float:
    """更长者判正；等长（臂 N 必然）⇒ 固定选下标 0。"""
    hit = 0
    for r in rows:
        a, b = r["candidates"]
        pred = 0 if len(a) > len(b) else (1 if len(b) > len(a) else 0)
        hit += int(pred == r["label"])
    return hit / max(1, len(rows))


def align_share_rule(rows: list[dict]) -> dict:
    """「替换处仍与原词共享汉字 ⇒ 判正」：把候选对齐回 frag 找替换位，再看共享状态。
    只在两候选同源于一个 frag 的行上可算（臂 N 全部）；词表对称时必然判同一侧。"""
    hit = n = 0
    for r in rows:
        if r["sub_word"] is None or r["frag"] is None:
            continue
        w, frag = r["sub_word"], r["frag"]
        shares = []
        for c in r["candidates"]:
            p = next((i for i in range(min(len(c), len(frag))) if c[i] != frag[i]), None)
            if p is None or p + len(w) > len(c):
                shares.append(None)
                continue
            shares.append(bool(set(c[p:p + len(w)]) & set(w)))
        if None in shares or shares[0] == shares[1]:
            pred = 0
        else:
            pred = 0 if shares[0] else 1
        hit += int(pred == r["label"])
        n += 1
    return {"acc": round(hit / n, 4) if n else None, "n_scored": n}


def ctx_char_overlap_rule(rows: list[dict]) -> float:
    """候选里出现在上下文中的**字种数**更多者判正；等值固定选下标 0。"""
    hit = 0
    for r in rows:
        cs = set(r["context"])
        cnt = [len(set(c) & cs) for c in r["candidates"]]
        pred = 0 if cnt[0] > cnt[1] else (1 if cnt[1] > cnt[0] else 0)
        hit += int(pred == r["label"])
    return hit / max(1, len(rows))


def len_stats(rows: list[dict]) -> dict:
    ctx = [len(r["context"]) for r in rows]
    cd = [len(c) for r in rows for c in r["candidates"]]
    return {"ctx_mean": round(sum(ctx) / len(ctx), 2), "ctx_max": max(ctx),
            "cand_mean": round(sum(cd) / len(cd), 2), "cand_max": max(cd),
            "cand_min": min(cd)}


def report(arm: str, name: str, rows: list[dict], dropped: dict) -> dict:
    n = len(rows)
    dist = Counter(r["label"] for r in rows)
    check(n > 0, f"{arm}/{name} 空")
    check(dist[0] * 2 == n and dist[1] * 2 == n, f"{arm}/{name} 不是 1:1：{dict(dist)}")
    keys = [r["key"] for r in rows]
    check(len(set(keys)) == n, f"{arm}/{name} key 重复 {n - len(set(keys))} 条")
    for r in rows:
        a, b = r["candidates"]
        if arm == "clean":
            check(len(a) == len(b), f"clean 臂候选不等长：{r['id']}")
            check(a not in r["context"] and b not in r["context"],
                  f"clean 臂出现逐字片段：{r['id']}")
            check(a != b, f"clean 臂两候选相同：{r['id']}")
        elif r["ctype"] == "plain":
            check(r["sub_word"] is None, f"shortcut/plain 不该有替换：{r['id']}")
    return {
        "n": n,
        "label_dist": {str(k): v for k, v in sorted(dist.items())},
        "ctype_dist": dict(Counter(r["ctype"] for r in rows)),
        "majority_baseline": max(dist.values()) / n,
        "blind_guess": 0.5,
        "naive_substring_rule": round(naive_rule(rows), 4),
        "length_rule": round(length_rule(rows), 4),
        "align_share_rule": align_share_rule(rows),
        "ctx_char_overlap_rule": round(ctx_char_overlap_rule(rows), 4),
        "lengths": len_stats(rows),
        "dropped": dropped,
        "n_sub_words": len({r["sub_word"] for r in rows if r["sub_word"]}),
    }


def overlap_report(sets: dict[str, list[dict]]) -> dict:
    """六集两两实测重叠（key/context/candidate 必须全 0；跨字段只报不断言）。"""
    out: dict[str, dict] = {}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ka = {r["key"] for r in sets[a]}
            kb = {r["key"] for r in sets[b]}
            ca = {r["context"] for r in sets[a]}
            cb = {r["context"] for r in sets[b]}
            va = {c for r in sets[a] for c in r["candidates"]}
            vb = {c for r in sets[b] for c in r["candidates"]}
            out[f"{a}|{b}"] = {"key": len(ka & kb), "context": len(ca & cb),
                               "candidate": len(va & vb),
                               "cross_ctx_cand_reported": len(va & cb) + len(vb & ca)}
    return out


# ---------------------------------------------------------------------------
# 探针：量命中率 / 构造成功率 → 池子建议
# ---------------------------------------------------------------------------
def probe() -> None:
    from nano_char_tokenizer import NanoCharTokenizer

    tok = NanoCharTokenizer()
    tbl = validate_tables()
    print("[probe] 词表:", json.dumps(tbl, ensure_ascii=False))
    sents = collect_sentences(20000, per_file=PER_FILE)
    donors = sents[-3000:]
    ctx_pool = sents[:-3000]
    rng = random.Random(7)
    print(f"[probe] 可用句 {len(sents)}（PER_FILE={PER_FILE}）")
    print(f"[probe] 含训练词头 {sum(1 for s in ctx_pool if occurrences(s, TRAIN_IDX))}"
          f"/{len(ctx_pool)} | 含留出词头 "
          f"{sum(1 for s in ctx_pool if occurrences(s, HOLDOUT_IDX))}/{len(ctx_pool)}")

    def yield_of(arm: str, ctype: str, n: int = 4000) -> float:
        ok = 0
        for _ in range(n):
            ctx = ctx_pool[rng.randrange(len(ctx_pool))]
            got = make_row(ctype, arm, ctx, donors, rng)
            if got is None:
                continue
            pos, neg = got["pos"], got["neg"]
            if arm == "shortcut" and ctype == "plain":       # 臂 S plain：正例**必须**是逐字片段
                if pos not in ctx or neg in ctx or pos == neg:
                    continue
            else:
                if pos in ctx or neg in ctx or pos == neg or len(pos) != len(neg):
                    continue
            if truncated(tok, ctx, SPEC["max_len_ctx"]) or \
               truncated(tok, pos, SPEC["max_len_cand"]) or \
               truncated(tok, neg, SPEC["max_len_cand"]):
                continue
            ok += 1
        return ok / n

    Y = {(a, c): yield_of(a, c) for a in ("clean", "shortcut")
         for c in ("plain", "heldout_pair", "shifted_pos")}
    for k, v in Y.items():
        print(f"[probe] 构造成功率 {k}: {v:.4f}")

    def need(rows: float, y: float) -> int:
        return int(rows / max(y, 1e-6) * 1.2) + 100

    half = N_ADV // 2
    sug = {
        "s_train": need(N_TRAIN, Y[("shortcut", "plain")]),
        "s_test": need(N_TEST, Y[("shortcut", "plain")]),
        "s_adv": need(half, Y[("shortcut", "heldout_pair")])
        + need(N_ADV - half, Y[("shortcut", "shifted_pos")]),
        "n_train": need(N_TRAIN, Y[("clean", "plain")]),
        "n_test": need(N_TEST, Y[("clean", "plain")]),
        "n_adv": need(half, Y[("clean", "heldout_pair")])
        + need(N_ADV - half, Y[("clean", "shifted_pos")]),
        "donor": 3000,
    }
    print("[probe] POOL 建议（1.2 倍余量）:", json.dumps(sug, ensure_ascii=False))
    print(f"[probe] 建议 NEED_SENT = {sum(sug.values()) + 2000}"
          f"；当前 POOL 合计 = {sum(POOL.values()) + 2000}")
    mx = max(len(s) for s in sents)
    print(f"[probe] 最长句 {mx} <= max_len_ctx {SPEC['max_len_ctx']}，超长 "
          f"{sum(1 for s in sents if truncated(tok, s, SPEC['max_len_ctx']))} 条")


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def main(argv: list[str]) -> None:
    if "--probe" in argv:
        probe()
        return

    from nano_char_tokenizer import NanoCharTokenizer

    tok = NanoCharTokenizer()
    tbl = validate_tables()
    rng = random.Random(SPLIT_SEED)
    (DATA / "shortcut").mkdir(parents=True, exist_ok=True)
    (DATA / "clean").mkdir(parents=True, exist_ok=True)

    print("[1/5] 取语料句子（resolve_corpus_files，fail-closed）...", flush=True)
    sents = collect_sentences(NEED_SENT)
    print(f"  {len(sents)} 句（目标 {NEED_SENT}，每文件上限 {PER_FILE}）", flush=True)

    shuffled = list(sents)
    rng.shuffle(shuffled)
    pools: dict[str, list[str]] = {}
    at = 0
    for k in ("s_train", "s_test", "s_adv", "n_train", "n_test", "n_adv", "donor"):
        pools[k] = shuffled[at:at + POOL[k]]
        at += POOL[k]
    spare = shuffled[at:]
    for k, p in pools.items():
        check(len(p) == POOL[k], f"池 {k} 大小 {len(p)} != {POOL[k]}")
    for i, a in enumerate(pools):
        for b in list(pools)[i + 1:]:
            ov = len(set(pools[a]) & set(pools[b]))
            check(ov == 0, f"源句池 {a}∩{b} 重叠 {ov} 条")

    print("[2/5] 构造六个集合 ...", flush=True)
    used_cand: set[str] = set()
    used_key: set[str] = set()
    sets: dict[str, list[dict]] = {}
    drops: dict[str, dict] = {}
    adv_kinds = [("heldout_pair", N_ADV // 2), ("shifted_pos", N_ADV - N_ADV // 2)]

    plan = [
        ("shortcut", "train", N_TRAIN, [("plain", N_TRAIN)], "s_train"),
        ("shortcut", "test", N_TEST, [("plain", N_TEST)], "s_test"),
        ("shortcut", "adv", N_ADV, adv_kinds, "s_adv"),
        ("clean", "train", N_TRAIN, [("plain", N_TRAIN)], "n_train"),
        ("clean", "test", N_TEST, [("plain", N_TEST)], "n_test"),
        ("clean", "adv", N_ADV, adv_kinds, "n_adv"),
    ]
    for arm, split, n, kinds, pool_name in plan:
        rows, dropped = build_split(arm, split, n, kinds, pools[pool_name],
                                    pools["donor"], rng, used_cand, used_key, tok)
        sets[f"{arm}/{split}"] = rows
        drops[f"{arm}/{split}"] = dropped
        print(f"  {arm}/{split}: {len(rows)} 行，末次尝试号 {rows[-1]['attempts']}"
              f"，丢弃 {dropped}", flush=True)

    print("[3/5] 六集两两不相交实测 ...", flush=True)
    ov = overlap_report(sets)
    for k, v in ov.items():
        check(v["key"] == 0 and v["context"] == 0 and v["candidate"] == 0,
              f"{k} 有重叠：{v}")
        print(f"  {k}: {v}", flush=True)

    print("[4/5] 统计与硬判 ...", flush=True)
    stats: dict = {
        "split_seed": SPLIT_SEED,
        "spec": SPEC,
        "vocab": tbl,
        "per_file": PER_FILE,
        "pools": {k: len(v) for k, v in pools.items()} | {"spare": len(spare)},
        "sets": {k: report(k.split("/")[0], k.split("/")[1], v, drops[k])
                 for k, v in sets.items()},
        "overlap_check": ov,
        "union_key_unique": len({r["key"] for rows in sets.values() for r in rows}),
        "total_rows": sum(len(v) for v in sets.values()),
        "label_criterion": "正例 = 语义等价于上下文该处意思的候选；负例 = 不等价（反义替换 / 无关片段）",
        "order_randomized": True,
        "chance": 0.5,
        "se_n2500": round((0.25 / N_TEST) ** 0.5, 6),
        "se_n8000": round((0.25 / N_TRAIN) ** 0.5, 6),
        "truncation_guard": "encode(text, max_length=SPEC[...], padding=False) 与未截断全长比对，超长即丢弃并计数",
    }
    clean_rule = stats["sets"]["clean/train"]["naive_substring_rule"]
    stats["clean_train_rule_ge_90"] = clean_rule >= 0.90
    print(f"  臂 N 训练集朴素逐字规则 = {clean_rule:.4f}"
          + ("  ⚠⚠ ≥ 0.90 ⇒ **该臂数据无效**" if clean_rule >= 0.90 else "（< 0.90，有效）"))
    check(not stats["clean_train_rule_ge_90"], "臂 N 训练集朴素规则 ≥ 0.90，数据无效")
    for k, v in stats["sets"].items():
        check(abs(v["majority_baseline"] - 0.5) < 1e-9, f"{k} 多数类基线不是 50%")
        if k.startswith("clean"):
            check(abs(v["naive_substring_rule"] - 0.5) < 1e-9,
                  f"{k} 逐字规则不是 50%：{v['naive_substring_rule']}")
            check(abs(v["length_rule"] - 0.5) < 1e-9,
                  f"{k} 长度规则不是 50%：{v['length_rule']}")
            check(abs(v["align_share_rule"]["acc"] - 0.5) < 1e-9,
                  f"{k} 共享字规则不是 50%：{v['align_share_rule']}")
    for k in ("shortcut/train", "shortcut/test"):
        check(stats["sets"][k]["naive_substring_rule"] >= 0.9999,
              f"{k} 逐字规则应为 100%：{stats['sets'][k]['naive_substring_rule']}")

    print("[5/5] 落盘 ...", flush=True)
    for k, rows in sets.items():
        arm, split = k.split("/")
        with open(DATA / arm / f"{split}.jsonl", "w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    (DATA / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: {kk: stats["sets"][k][kk] for kk in
                          ("n", "label_dist", "ctype_dist", "naive_substring_rule",
                           "length_rule", "align_share_rule", "ctx_char_overlap_rule",
                           "dropped", "lengths")}
                      for k in stats["sets"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1:])
