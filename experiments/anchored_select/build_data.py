#!/usr/bin/env python3
"""锚定候选打标（anchored_select）数据集构建，fail-closed。

产出（全部落 data/*.jsonl，逐条带 rule，可追溯）：
  data/train.jsonl        8000 = 支持 4000 + 不支持 4000（换词 1334 / 反义 1333 / 无关 1333）
  data/test.jsonl         1600 = 800 + 800（同构造分布、源句池不相交）
  data/adversarial.jsonl   600 = 同义改写 300（正） + 同词表换位置 300（负），**不进训练**
  data/stats.json         规模 / 正负比 / 多数类基线 / 两两不相交实测 / 表层捷径基线

标签口径（PREREG §1，口径判断）：
  支持 = 候选在原文上下文里有等价锚点（逐字片段，或该片段的同义改写）；否则不支持。
  正例 = 发射候选区间（span 指右侧候选字段），负例 = 背景（不发射）。

语料入口只用 dtseek.tasks.corpus.resolve_corpus_files（glob 空即抛错，不绕过）。
任一 assert 不满足即抛错，绝不静默降级。
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
SPLIT_SEED = 20240927

N_TRAIN, N_TEST, N_ADV = 8000, 1600, 600
#: 源句池（按下标切分，三池两两不相交；样本构造时**池内有放回**取上下文，
#: 唯一性由「整条 text 全局不重复」保证 —— 失败的构造不消耗池子）
POOL_N = {"train": 9500, "test": 2100, "adv": 950}
NEED_SENT = sum(POOL_N.values()) + 500      # 余量

#: 分隔符用 ASCII `|` —— 实测 nano_char_tokenizer 里全角 `｜` 是 <unk>，
#: 用 <unk> 当分隔会让两侧结构在编码后不可分（1 字符 = 1 token 的索引契约本身没问题）。
SEP = "|"
PREFIX, SUFFIX = "原文：", "候选："
SENT_SPLIT = re.compile(r"[。！？\n；;]+")
ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
CJK = re.compile(r"[一-龥]")
#: 句子级垃圾/非中文过滤（含结构标记与两种竖线）
BAD = re.compile(r"[�\t｜|]|原文：|候选：")

PUNCT_TAIL = set("，。！？、；：\"'”’）)】》…~—-,.?!:;")
PUNCT_HEAD = set("，。！？、；：\"'“‘（(【《…~—-,.?!:;")

#: 反义/矛盾替换表（双向命中：key→val 与 val→key 都算）
ANT = [
    ("提高", "降低"), ("增加", "减少"), ("上升", "下降"), ("变好", "变坏"),
    ("高兴", "难过"), ("美丽", "丑陋"), ("快速", "缓慢"), ("简单", "复杂"),
    ("安全", "危险"), ("容易", "困难"), ("安静", "吵闹"), ("炎热", "寒冷"),
    ("诚实", "虚伪"), ("支持", "反对"), ("同意", "拒绝"), ("接受", "放弃"),
    ("记得", "忘记"), ("表扬", "批评"), ("成功", "失败"), ("前进", "后退"),
    ("开始", "结束"), ("出现", "消失"), ("增加", "缩减"), ("上升", "回落"),
    ("表扬", "责备"), ("喜欢", "讨厌"), ("勇敢", "胆怯"), ("勤奋", "懒惰"),
    ("大方", "小气"), ("耐心", "急躁"), ("清醒", "糊涂"), ("完整", "残缺"),
    ("光滑", "粗糙"), ("明亮", "昏暗"), ("湿润", "干燥"), ("厚实", "单薄"),
    ("宽敞", "狭窄"), ("笔直", "弯曲"), ("前进", "撤退"), ("奖励", "惩罚"),
    ("得", "失"), ("买", "卖"), ("借", "还"), ("开", "关"),
    ("来", "去"), ("升", "降"), ("加", "减"), ("赚", "赔"),
    ("赢", "输"), ("胜", "负"), ("真", "假"), ("对", "错"),
    ("多", "少"), ("高", "低"), ("长", "短"), ("宽", "窄"),
    ("深", "浅"), ("轻", "重"), ("厚", "薄"), ("冷", "热"),
    ("黑", "白"), ("新", "旧"), ("早", "晚"), ("快", "慢"),
    ("老", "嫩"), ("干", "湿"), ("软", "硬"), ("甜", "苦"),
    ("富", "贫"), ("强", "弱"), ("攻", "守"), ("进", "退"),
]

#: 同义改写表（对抗集正例用；语义等价性 = 口径判断，人工给定）
SYN = [
    ("看见", "看到"), ("立刻", "马上"), ("高兴", "开心"), ("漂亮", "好看"),
    ("快速", "迅速"), ("帮助", "帮忙"), ("讨论", "商量"), ("拒绝", "回绝"),
    ("开始", "启动"), ("担心", "担忧"), ("安静", "宁静"), ("简单", "容易"),
    ("重要", "关键"), ("温暖", "暖和"), ("觉得", "认为"), ("知道", "了解"),
    ("选择", "挑选"), ("准备", "打算"), ("经常", "常常"), ("特别", "尤其"),
    ("大概", "大约"), ("好像", "仿佛"), ("反正", "总之"), ("干脆", "索性"),
    ("也许", "或许"), ("赶紧", "赶快"), ("全部", "所有"), ("依旧", "仍然"),
    ("突然", "忽然"), ("马上", "立刻"), ("原来", "原本"), ("本来", "原本"),
    ("询问", "打听"), ("讲述", "叙述"), ("观看", "观赏"), ("购买", "买"),
    ("居住", "住"), ("忘记", "忘了"), ("明白", "清楚"), ("奇怪", "诧异"),
    ("疲倦", "疲惫"), ("生气", "恼火"), ("着急", "焦急"), ("舒服", "舒适"),
    ("适合", "合适"), ("比较", "较为"), ("更加", "更为"), ("非常", "十分"),
    ("办法", "方法"), ("时候", "时机"), ("地方", "地点"), ("东西", "物件"),
    ("结果", "后果"), ("原因", "缘由"), ("意见", "看法"), ("消息", "音讯"),
]

#: 换词用词库（语料里抽常见词；替换词不得出现在上下文里）
WORD_BANK = [
    "今天", "明天", "昨天", "时间", "问题", "工作", "学习", "生活", "家里",
    "朋友", "孩子", "老师", "同学", "吃饭", "出门", "回来", "觉得", "知道",
    "应该", "可能", "需要", "帮忙", "东西", "地方", "时候", "办法", "方法",
    "结果", "开始", "感觉", "认为", "非常", "特别", "一定", "一起", "现在",
    "还是", "但是", "因为", "所以", "如果", "虽然", "然后", "而且", "已经",
    "正在", "准备", "安排", "注意", "检查", "收拾", "打扫", "休息", "睡觉",
    "周末", "早上", "晚上", "下午", "路上", "房间", "厨房", "公司", "学校",
    "医院", "商场", "公园", "车站", "天气", "下雨", "放假", "加班", "开会",
]

NEG_INSERT = ["不", "没", "从不", "并不"]

#: 正例规则（其余规则一律产负例）
POS_RULES = {"verbatim", "paraphrase"}


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[anchored_select build fail-closed] {msg}")


def cjk(s: str) -> int:
    return len(CJK.findall(s))


def ok_sentence(s: str) -> bool:
    """语料句子可用性：长度、中文占比、无结构标记污染。"""
    if not (14 <= len(s) <= 44):
        return False
    if BAD.search(s):
        return False
    if cjk(s) < 0.6 * len(s):
        return False
    if re.search(r"[A-Za-z]{3,}", s):       # 代码/英文片段不当证据上下文
        return False
    if s.count(SEP):
        return False
    return True


# ---------------------------------------------------------------------------
# 语料 → 三个源句池
# ---------------------------------------------------------------------------
def collect_sentences(n: int, per_file: int = 4000) -> tuple[list[str], list[str]]:
    """按 SPLIT_SEED 打乱文件顺序后逐行读，**每个文件最多取 per_file 句**（保证来源多样）。

    返回 (sentences, 用过的文件列表)。文件顺序与用量都进 stats（可追溯）。
    """
    files = resolve_corpus_files(CORPUS_GLOB)          # fail-closed：空 glob 直接抛
    rnd = random.Random(SPLIT_SEED)
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
    check(len(out) >= n, f"语料可用句子不足：{len(out)} < {n}（过滤太严或语料缺失）")
    return out, used


# ---------------------------------------------------------------------------
# 片段与扰动
# ---------------------------------------------------------------------------
def pick_fragment(sent: str, rng: random.Random, lo: int, hi: int,
                  used: set[str], tries: int = 40) -> str | None:
    """从 sent 里取一个连贯片段：不以标点起止、含汉字、未被用过。"""
    for _ in range(tries):
        if len(sent) < lo + 1:
            return None
        start = rng.randrange(0, len(sent) - lo + 1)
        end = start + rng.randint(lo, min(hi, len(sent) - start))
        if end - start < lo or end > len(sent):
            continue
        f = sent[start:end]
        if f[0] in PUNCT_HEAD or f[-1] in PUNCT_TAIL:
            continue
        if SEP in f or cjk(f) < lo - 2 or f in used:
            continue
        if f == sent:
            continue
        return f
    return None


def frag_positions(f: str) -> list[int]:
    """片段里可做换词/反义替换的窗口起点（优先整词表命中，其次任意 2 字窗口）。"""
    hits = [f.find(w) for w in WORD_BANK if w in f]
    hits = [i for i in hits if i >= 0]
    if hits:
        return hits
    return [i for i in range(0, len(f) - 1)]


def rule_swap_word(context: str, frag: str, rng: random.Random) -> str | None:
    """① 换词：同位置换成别处的词（替换词不在上下文里）。"""
    positions = frag_positions(frag)
    rng.shuffle(positions)
    for p in positions:
        wlen = 2 if (p + 2 <= len(frag) and frag[p:p + 2] in WORD_BANK) else 2
        orig = frag[p:p + wlen]
        cands = [w for w in WORD_BANK if w != orig and w not in context and w not in frag]
        if not cands:
            continue
        rep = rng.choice(cands)
        out = frag[:p] + rep + frag[p + wlen:]
        if out != frag and out not in context and cjk(out) >= 4:
            return out
    return None


def rule_antonym(context: str, frag: str, rng: random.Random) -> str | None:
    """② 反义/矛盾 A：反义词表替换（双向命中）。"""
    rng.shuffle(ANT)
    for k, v in ANT:
        for a, b in ((k, v), (v, k)):
            if a in frag:
                out = frag.replace(a, b, 1)
                if out != frag and out not in context:
                    return out
    return None


def rule_negate(context: str, frag: str, rng: random.Random) -> str | None:
    """② 反义/矛盾 B：插入否定（把命题翻成矛盾）。"""
    for p in sorted(frag_positions(frag), key=lambda x: -x):
        ins = rng.choice(NEG_INSERT)
        out = frag[:p] + ins + frag[p:]
        if out not in context and abs(len(out) - len(frag)) <= 2:
            return out
    # 尾部追加否定词（保证与原文不同锚点）
    for ins in ("并不", "没有"):
        out = frag + ins
        if out not in context:
            return out
    return None


def rule_unrelated(context: str, donor: str, rng: random.Random,
                   used: set[str]) -> str | None:
    """③ 无关片段：从别处取。"""
    f = pick_fragment(donor, rng, 8, 18, used)
    if f is None or f in context:
        return None
    return f


def rule_paraphrase(context: str, frag: str, rng: random.Random) -> str | None:
    """对抗集正例：同义改写（词表与上下文不同，语义等价 = 口径判断）。"""
    rng.shuffle(SYN)
    for a, b in SYN:
        for x, y in ((a, b), (b, a)):
            if x in frag:
                out = frag.replace(x, y, 1)
                if out != frag and out not in context and y not in context:
                    return out
    return None


def rule_reorder(context: str, frag: str, rng: random.Random) -> str | None:
    """对抗集负例：同词表换位置 —— 词表与片段完全一致，只有位置重排。"""
    if len(frag) < 10:
        return None
    marks = [i for i, ch in enumerate(frag) if ch in "，、；："]
    chunks: list[str] = []
    if len(marks) >= 2:
        a, b = marks[0], marks[1]
        chunks = [frag[:a + 1], frag[a + 1:b + 1], frag[b + 1:]]
    elif len(marks) == 1:
        a = marks[0]
        chunks = [frag[:a + 1], frag[a + 1:]]
    else:
        cuts = [i for i in range(5, len(frag) - 5)]
        if not cuts:
            return None
        a = rng.choice(cuts)
        chunks = [frag[:a], frag[a:]]
    chunks = [c for c in chunks if c]
    if len(chunks) < 2:
        return None
    perms = [chunks[1:] + chunks[:1]] if len(chunks) == 2 else [
        list(p) for p in _perms(chunks)]
    rng.shuffle(perms)
    for p in perms:
        if p == chunks:
            continue
        out = "".join(p)
        if out != frag and out not in context and sorted(out) == sorted(frag):
            return out
    return None


def _perms(xs: list[str]):
    if len(xs) <= 1:
        yield list(xs)
        return
    for i in range(len(xs)):
        for rest in _perms(xs[:i] + xs[i + 1:]):
            yield [xs[i]] + rest


# ---------------------------------------------------------------------------
# 单条样本
# ---------------------------------------------------------------------------
def make_text(context: str, cand: str) -> str:
    return f"{PREFIX}{context}{SEP}{SUFFIX}{cand}"


def pos_row(idx: str, split: str, rule: str, context: str, cand: str) -> dict:
    start = len(PREFIX) + len(context) + len(SEP) + len(SUFFIX)
    text = make_text(context, cand)
    return {
        "id": idx, "split": split, "rule": rule, "label": 1,
        "context": context, "candidate": cand, "text": text,
        "spans": [{"label": 1, "start": start, "end": start + len(cand)}],
        "lex_match": cand in context,
    }


def neg_row(idx: str, split: str, rule: str, context: str, cand: str) -> dict:
    return {
        "id": idx, "split": split, "rule": rule, "label": 0,
        "context": context, "candidate": cand, "text": make_text(context, cand),
        "spans": [],
        "lex_match": cand in context,
    }


def build_split(name: str, pos_quotas: dict[str, int], neg_quotas: dict[str, int],
                pool: list[str], rng: random.Random,
                used_text: set[str], used_cand: set[str]) -> list[dict]:
    """按配额造一个集合（正/负规则分开声明，标签由规则决定，不靠事后翻转）。

    上下文在池内**轮转有放回**取（构造失败不消耗池子，否则高失败率规则会把池烧穿）；
    样本唯一性由「整条 text 全局不重复 + candidate 全局不重复」保证。
    """
    order = list(pool)
    rng.shuffle(order)
    ptr = 0
    rows: list[dict] = []
    attempts = 0

    def next_ctx() -> str:
        nonlocal ptr, attempts
        attempts += 1
        if attempts > len(order) * 40:
            raise AssertionError(
                f"[anchored_select build fail-closed] {name} 构造失败过多（{attempts} 次尝试）")
        s = order[ptr % len(order)]
        ptr += 1
        return s

    def emit(rule: str, ctx: str, cand: str, frag: str | None, n: int) -> bool:
        """构造一行并落库；返回是否成功（失败不消耗配额）。"""
        idx = f"{name}-{rule}-{n:05d}"
        if rule in POS_RULES:
            row = pos_row(idx, name, rule, ctx, cand)
            if rule == "verbatim":
                check(row["lex_match"], f"逐字正例必须是上下文片段：{row}")
            else:
                check(not row["lex_match"], f"改写正例不得与上下文逐字相同：{row}")
            check(row["spans"][0]["end"] <= len(row["text"]), "span 越界")
        else:
            row = neg_row(idx, name, rule, ctx, cand)
            check(not row["lex_match"], f"负例不得是上下文逐字片段：{row}")
        if rule == "reorder":
            check(frag is not None and sorted(cand) == sorted(frag), "换位置必须同词表")
        if row["text"] in used_text or cand in used_cand:
            return False
        used_text.add(row["text"])
        used_cand.add(cand)
        rows.append(row)
        return True

    for quotas, kind in ((pos_quotas, "pos"), (neg_quotas, "neg")):
        for rule, want in quotas.items():
            n = 0
            tries = 0
            while n < want:
                tries += 1
                if tries > want * 60:
                    raise AssertionError(
                        f"[anchored_select build fail-closed] {name}/{rule} 只造出 {n}/{want}"
                        f"（{tries} 次尝试）")
                ctx = next_ctx()
                frag = pick_fragment(ctx, rng, 10 if rule == "reorder" else 8, 18, used_cand)
                if frag is None:
                    continue
                cand: str | None
                if rule == "verbatim":
                    cand = frag
                elif rule == "paraphrase":
                    cand = rule_paraphrase(ctx, frag, rng)
                elif rule == "word_swap":
                    cand = rule_swap_word(ctx, frag, rng)
                elif rule == "antonym":
                    cand = rule_antonym(ctx, frag, rng) or rule_negate(ctx, frag, rng)
                elif rule == "unrelated":
                    cand = rule_unrelated(ctx, rng.choice(pool), rng, used_cand)
                elif rule == "reorder":
                    cand = rule_reorder(ctx, frag, rng)
                else:
                    raise AssertionError(f"未知 rule {rule}（{kind}）")
                if cand is None or (rule != "verbatim" and cand == frag):
                    continue
                if emit(rule, ctx, cand, frag, n):
                    n += 1
    return rows


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
def lex_baseline(rows: list[dict]) -> float:
    """零训练基线：候选是上下文的逐字子串 ⇒ 判支持。"""
    hit = sum(1 for r in rows if (r["candidate"] in r["context"]) == bool(r["label"]))
    return hit / max(1, len(rows))


def report(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    dist = Counter(r["label"] for r in rows)
    check(n > 0, f"{name} 空")
    check(dist[1] * 2 == n and dist[0] * 2 == n,
          f"{name} 不是 1:1：{dict(dist)}（多数类基线会偏离 50%）")
    texts = [r["text"] for r in rows]
    check(len(set(texts)) == n, f"{name} 文本重复 {n - len(set(texts))} 条")
    for r in rows:
        check(bool(r["spans"]) == bool(r["label"]), f"标签与 span 不一致：{r['id']}")
        if r["label"]:
            s = r["spans"][0]
            check(r["text"][s["start"]:s["end"]] == r["candidate"], f"span 没指向候选：{r['id']}")
        else:
            check(r["candidate"] not in r["context"], f"负例却是逐字片段：{r['id']}")
    return {
        "n": n,
        "label_dist": {str(k): v for k, v in sorted(dist.items())},
        "rule_dist": dict(Counter(r["rule"] for r in rows)),
        "majority_baseline": max(dist.values()) / n,
        "blind_guess": 1.0 / 2,
        "lex_substring_baseline": round(lex_baseline(rows), 4),
        "mean_text_len": round(sum(len(r["text"]) for r in rows) / n, 2),
        "max_text_len": max(len(r["text"]) for r in rows),
    }


def overlap(a: list[dict], b: list[dict]) -> dict:
    ta, tb = {r["text"] for r in a}, {r["text"] for r in b}
    ca, cb = {r["context"] for r in a}, {r["context"] for r in b}
    ka, kb = {r["candidate"] for r in a}, {r["candidate"] for r in b}
    return {"text": len(ta & tb), "context": len(ca & cb), "candidate": len(ka & kb)}


def main() -> None:
    rng = random.Random(SPLIT_SEED)
    DATA.mkdir(parents=True, exist_ok=True)

    print("[1/4] 取语料句子（resolve_corpus_files，fail-closed）...", flush=True)
    sents, files = collect_sentences(NEED_SENT)
    print(f"  {len(sents)} 句 / {len(files)} 个文件", flush=True)

    shuffled = list(sents)
    rng.shuffle(shuffled)
    pools, at = {}, 0
    for name in ("train", "test", "adv"):
        pools[name] = shuffled[at:at + POOL_N[name]]
        at += POOL_N[name]
    spare = shuffled[at:]
    for name, p in pools.items():
        check(len(p) == POOL_N[name], f"{name} 池大小 {len(p)} != {POOL_N[name]}")
    # 池两两不相交（按构造即不相交，再实测一次）
    for a in pools:
        for b in pools:
            if a < b:
                ov = len(set(pools[a]) & set(pools[b]))
                check(ov == 0, f"源句池 {a}∩{b} 重叠 {ov} 条")

    print("[2/4] 构造三个集合 ...", flush=True)
    used_text: set[str] = set()
    used_cand: set[str] = set()
    train = build_split("train", {"verbatim": N_TRAIN // 2},
                        {"word_swap": 1334, "antonym": 1333, "unrelated": 1333},
                        pools["train"], rng, used_text, used_cand)
    test = build_split("test", {"verbatim": N_TEST // 2},
                       {"word_swap": 267, "antonym": 267, "unrelated": 266},
                       pools["test"], rng, used_text, used_cand)
    # 对抗集：正例只用同义改写（词表与上下文不同），负例只用同词表换位置
    adv = build_split("adv", {"paraphrase": N_ADV // 2}, {"reorder": N_ADV // 2},
                      pools["adv"], rng, used_text, used_cand)
    for nm, rows, want in (("train", train, N_TRAIN), ("test", test, N_TEST),
                           ("adv", adv, N_ADV)):
        check(len(rows) == want, f"{nm} 规模 {len(rows)} != {want}")

    print("[3/4] 两两不相交实测 ...", flush=True)
    pairs = {"train∩test": overlap(train, test),
             "train∩adv": overlap(train, adv),
             "test∩adv": overlap(test, adv)}
    for k, v in pairs.items():
        check(all(x == 0 for x in v.values()), f"{k} 有重叠：{v}")
        print(f"  {k}: text={v['text']} context={v['context']} candidate={v['candidate']}", flush=True)
    all_texts = [r["text"] for rows in (train, test, adv) for r in rows]
    check(len(set(all_texts)) == len(all_texts), "三集并集大小 != 各自之和")

    stats = {
        "split_seed": SPLIT_SEED,
        "corpus_files_used": files,
        "sentence_pools": {k: len(v) for k, v in pools.items()} | {"spare": len(spare)},
        "train": report("train", train),
        "test": report("test", test),
        "adversarial": report("adv", adv),
        "overlap_check": pairs,
        "union_text_unique": len(set(all_texts)),
        "label_criterion": "支持 = 候选在上下文有等价锚点（逐字片段或同义改写）",
        "text_layout": f"{PREFIX}{{context}}{SEP}{SUFFIX}{{candidate}}",
    }
    for k in ("train", "test", "adversarial"):
        check(abs(stats[k]["majority_baseline"] - 0.5) < 1e-9, f"{k} 多数类基线不是 50%")

    print("[4/4] 落盘 ...", flush=True)
    for nm, rows in (("train", train), ("test", test), ("adversarial", adv)):
        with open(DATA / f"{nm}.jsonl", "w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    (DATA / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("train", "test", "adversarial",
                                            "overlap_check", "union_text_unique")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
