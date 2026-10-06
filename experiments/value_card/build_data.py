#!/usr/bin/env python3
"""价值观卡（value_judge）数据集构建，fail-closed。

产出（全部落 data/*.jsonl，逐条带 rule，可追溯）：
  data/train.jsonl        8100 = 背景 3240 + 得体 2430 + 冒犯 2430
  data/test.jsonl         1500 = 背景  600 + 得体  450 + 冒犯  450
  data/adversarial.jsonl   600 = 同义改写冒犯 200 + 同义改写得体 200 + 同词表换语境 200
  data/gold.jsonl          120 = 手写金标准（V3 外部代理真值，40/40/40）
  data/stats.json          规模 / 类别比 / 多数类基线 / 朴素规则基线 / 两两不相交实测

标签口径（PREREG §1，口径判断）：
  冒犯 = span 是**使用中**的贬损/攻击表达；得体 = span 是**使用中**的礼貌/致谢/致歉表达；
  背景 = 不发射 —— 本句没有值判断表达，或词表词只被**提及**（引用/词典/禁止框架）。
  类 id：0=背景（不发射）/ 1=得体 / 2=冒犯（TaskSpec.classes 下标）。

监督来源（PREREG §1）：自建人工词表（OFF/POL/NEU 三张）填进**从语料取的载体句**；
对抗集只用**词表外**的近义表达（adv_syn_*）与**同词表换语境**（adv_mention）。
朴素规则基线（零训练）= 只扫 OFF∪POL 词表：命中⇒按位置发射该类，不命中⇒背景。

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

#: 三集规模与类别配额（跑前算死；多数类 = 背景）
QUOTA = {
    "train": {"bg": 3240, "pol": 2430, "off": 2430},
    "test": {"bg": 600, "pol": 450, "off": 450},
    "adversarial": {"bg": 200, "pol": 200, "off": 200},
}
POOL_N = {"train": 6500, "test": 1500, "adv": 700}
NEED_SENT = sum(POOL_N.values()) + 600

MAX_LEN = 64                      # TaskSpec.max_len，超长直接抛

#: ---------------------------------------------------------------------------
#: 自建小型人工词表（监督来源；三张表两两子串不相交，构建期断言）
#: ---------------------------------------------------------------------------
#: 冒犯/贬损类（**使用**时判冒犯）
OFF = [
    "笨蛋", "傻瓜", "白痴", "混蛋", "王八蛋", "蠢货", "废物", "弱智",
    "脑残", "猪头", "饭桶", "懒虫", "怂包", "胆小鬼", "窝囊废", "软骨头",
    "跟屁虫", "马屁精", "冤大头", "不长进", "没志气", "讨厌鬼", "害人精",
    "扫把星", "铁公鸡", "小气鬼", "泼妇", "无赖", "流氓", "混账",
    "岂有此理", "不像话", "丢人现眼", "不争气", "让人失望", "胡闹",
    "讨人嫌", "碍事", "烦人", "气人",
]
#: 得体/礼貌类（**使用**时判得体）
POL = [
    "谢谢", "感谢", "辛苦", "对不起", "抱歉", "请问", "欢迎", "恭喜",
    "祝贺", "您好", "麻烦", "劳驾", "久仰", "失敬", "打扰", "请教",
    "包涵", "担待", "赏光", "光临", "慢走", "客气", "见笑", "献丑",
    "承让", "过奖", "荣幸", "感激", "有劳", "拜托", "失礼", "海涵",
    "劳您", "留步", "失陪", "慢用", "承蒙", "叨扰", "屈驾", "台安",
]
#: 中性词（背景类注入：有"注入痕迹"但不构成值判断，堵掉"有注入就发射"的捷径）
NEU = [
    "今天", "明天", "时间", "问题", "工作", "学习", "生活", "东西",
    "地方", "时候", "办法", "方法", "结果", "开始", "周末", "早上",
    "晚上", "下午", "路上", "房间", "厨房", "公司", "学校", "医院",
    "商场", "公园", "车站", "天气", "下雨", "放假", "加班", "开会",
    "吃饭", "睡觉", "休息", "出门", "回来", "需要", "准备", "安排",
    "检查",
]
#: 对抗集专用：**词表外**的近义表达（与 OFF/POL/NEU 子串不相交，构建期断言）
SYN_OFF = [
    "脑子进水", "没出息", "不中用", "智商堪忧", "蠢得要命", "啥也不是",
    "不成器", "缺心眼", "二百五", "拎不清", "不着调", "叫人看扁",
    "成事不足", "朽木", "烂泥", "扶不起", "尽添乱", "碍手碍脚",
    "让人头疼", "成何体统", "太不像样", "狗屁不通", "一无是处", "干啥啥差",
]
SYN_POL = [
    "多亏了你", "真贴心", "有心了", "您真好", "想得周到", "受教了",
    "长见识了", "真心不易", "难为你了", "费心了", "实在难得", "受宠若惊",
    "蓬荜生辉", "处处照应", "记挂在心", "太周到了", "用心了", "费神了",
    "让您跑一趟", "真是不好意思", "劳烦您了", "难得您记得", "惭愧得很", "有负所托",
]
#: 对抗集：同词表换语境（**提及**框架）—— 词表词在句中只是被引用/查到/禁止
MENTION_TPL_OFF = [
    "他写作文时用了{w}这个词，老师让他重写。",
    "词典里把{w}解释成骂人的粗话。",
    "妈妈说：不许用{w}骂人。",
    "字幕里出现{w}时被消音了。",
]
MENTION_TPL_POL = [
    "语文课上老师讲了{w}这个词的用法。",
    "他把{w}这个词抄进了生词本。",
    "考试里有一道题考{w}怎么用。",
    "词典里能查到{w}这几个字。",
]

CLS = {"bg": 0, "pol": 1, "off": 2}
INV_CLS = {0: "背景", 1: "得体", 2: "冒犯"}

SENT_SPLIT = re.compile(r"[。！？\n；;]+")
ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
CJK = re.compile(r"[一-龥]")
BAD = re.compile(r"[�\t｜|⟧⟦]|原文：|候选：")
PUNCT_TAIL = set("，。！？、；：\"'”’）)】》…~—-,.?!:;")
PUNCT_HEAD = set("，。！？、；：\"'“‘（(【《…~—-,.?!:;")
GOLD_MARK_OPEN, GOLD_MARK_CLOSE = "⟦", "⟧"


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[value_card build fail-closed] {msg}")


def cjk(s: str) -> int:
    return len(CJK.findall(s))


def ok_sentence(s: str) -> bool:
    """语料句子可用性：长度、中文占比、无结构标记污染、不含任何词表词。"""
    if not (14 <= len(s) <= 40):
        return False
    if BAD.search(s):
        return False
    if cjk(s) < 0.6 * len(s):
        return False
    if re.search(r"[A-Za-z]{3,}", s):
        return False
    if any(w in s for w in OFF + POL):        # 载体自带值词 ⇒ 朴素规则会误命中
        return False
    return True


def lexicon_sanity() -> None:
    """词表自检：表内无互为子串、三张表两两不相交、SYN 与三张表子串不相交。"""
    for name, lst in (("OFF", OFF), ("POL", POL), ("NEU", NEU),
                      ("SYN_OFF", SYN_OFF), ("SYN_POL", SYN_POL)):
        check(len(lst) == len(set(lst)), f"{name} 词表有重复")
        for a in lst:
            for b in lst:
                if a != b:
                    check(a not in b, f"{name} 内互为子串：{a!r} ⊂ {b!r}")
    tri = {"OFF": OFF, "POL": POL, "NEU": NEU}
    names = list(tri)
    for i, x in enumerate(names):
        for y in names[i + 1:]:
            for a in tri[x]:
                for b in tri[y]:
                    check(a not in b and b not in a,
                          f"{x}/{y} 两表相交：{a!r} vs {b!r}")
    for name, lst in (("SYN_OFF", SYN_OFF), ("SYN_POL", SYN_POL)):
        for a in lst:
            for tbl in (OFF, POL, NEU):
                for b in tbl:
                    check(b not in a and a not in b,
                          f"{name} 与词表相交：{a!r} vs {b!r}")
    check(not (set(SYN_OFF) & set(SYN_POL)), "SYN_OFF/SYN_POL 相交")


def all_value_words() -> list[str]:
    return OFF + POL


def naive_rule(text: str) -> tuple[int, int, int]:
    """朴素规则基线（零训练）：只扫 OFF∪POL 词表。

    返回 (pred_cls, start, end)；命中 ⇒ 按**最早出现**的词表词发射该类，
    不命中 ⇒ (0, -1, -1) 即背景、不发射。
    """
    best: tuple[int, int] | None = None
    best_cls = 0
    for cls_id, tbl in ((2, OFF), (1, POL)):
        for w in tbl:
            i = text.find(w)
            if i < 0:
                continue
            if best is None or i < best[0]:
                best = (i, len(w))
                best_cls = cls_id
    if best is None:
        return 0, -1, -1
    return best_cls, best[0], best[0] + best[1]


def expected_value_matches(text: str) -> list[tuple[int, str]]:
    """text 里全部 OFF/POL 命中（位置, 词）—— 构造期用来断言"只该有的一处命中"。"""
    out = []
    for cls_id, tbl in ((2, OFF), (1, POL)):
        for w in tbl:
            i = text.find(w)
            if i >= 0:
                out.append((i, w))
    return sorted(out)


def matches_ok(kind: str, text: str, s: int, entry: str) -> bool:
    """插入边界可能与载体拼出额外的值词（如中性词+邻字 ⇒ 恰好成 OFF 词）。

    命中数不符合本行预期就**重造**（换位置/换载体），而不是把脏样本放行；
    row() 里仍留同一断言作兜底 fail-closed。
    """
    got = expected_value_matches(text)
    if kind in ("neu", "syn"):
        return got == []
    return got == [(s, entry)]


# ---------------------------------------------------------------------------
# 语料 → 三个源句池
# ---------------------------------------------------------------------------
def collect_sentences(n: int, per_file: int = 4000) -> tuple[list[str], list[str]]:
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


def inject(carrier: str, entry: str, rng: random.Random) -> tuple[str, int, int]:
    """把词表条目插进载体句的随机内部位置，返回 (text, span_start, span_end)。"""
    check(entry not in carrier, f"载体已含条目：{carrier!r} / {entry!r}")
    pos = rng.randint(1, max(1, len(carrier) - 1))
    text = carrier[:pos] + entry + carrier[pos:]
    check(len(text) <= MAX_LEN, f"text 超过 max_len={MAX_LEN}：{text!r}")
    return text, pos, pos + len(entry)


def row(idx: str, split: str, rule: str, label: int, text: str,
        span: tuple[int, int], entry: str, kind: str, carrier: str) -> dict:
    s, e = span
    if label == 0:
        spans = []
    else:
        check(text[s:e] == entry or kind == "mention",
              f"span 没指向条目：{idx}")
        spans = [{"label": label, "start": s, "end": e}]
    if kind in ("off", "pol"):
        check(expected_value_matches(text) == [(s, entry)],
              f"{kind} 注入的值词命中数不对：{idx} {expected_value_matches(text)}")
    elif kind == "neu":
        check(expected_value_matches(text) == [], f"背景注入却命中值词表：{idx}")
    elif kind == "syn":
        check(expected_value_matches(text) == [], f"同义改写却命中值词表：{idx}")
    elif kind == "mention":
        check(len(expected_value_matches(text)) == 1, f"提及句值词命中数 != 1：{idx}")
    return {"id": idx, "split": split, "rule": rule, "label": label,
            "cls": INV_CLS[label], "entry": entry, "entry_kind": kind,
            "carrier": carrier, "text": text, "span": [s, e], "spans": spans}


def build_split(name: str, quota: dict[str, int], pool: list[str],
                rng: random.Random, used_text: set[str]) -> list[dict]:
    """按配额造一个集合：bg=中性注入 / pol=得体词表注入 / off=冒犯词表注入。"""
    order = list(pool)
    rng.shuffle(order)
    ptr = 0
    rows: list[dict] = []
    want = [("bg", "neu", NEU, 0), ("pol", "pol", POL, 1), ("off", "off", OFF, 2)]
    for key, kind, tbl, label in want:
        n = 0
        tries = 0
        while n < quota[key]:
            tries += 1
            if tries > quota[key] * 80:
                raise AssertionError(
                    f"[value_card build fail-closed] {name}/{key} 只造出 {n}/{quota[key]}"
                    f"（{tries} 次尝试）")
            carrier = order[ptr % len(order)]
            ptr += 1
            cand_entries = [w for w in tbl if w not in carrier]
            if not cand_entries:
                continue
            entry = rng.choice(cand_entries)
            text, s, e = inject(carrier, entry, rng)
            if text in used_text or not matches_ok(kind, text, s, entry):
                continue
            used_text.add(text)
            rid = f"{name}-{kind}-{n:05d}"
            rows.append(row(rid, name, kind, label, text, (s, e), entry, kind, carrier))
            n += 1
    return rows


def build_adversarial(pool: list[str], rng: random.Random,
                      used_text: set[str]) -> list[dict]:
    """对抗集（**不进训练**）：同义改写冒犯/得体 + 同词表换语境（提及）。"""
    order = list(pool)
    rng.shuffle(order)
    ptr = 0
    rows: list[dict] = []
    for key, tbl, kind, label in (("off", SYN_OFF, "syn", 2),
                                  ("pol", SYN_POL, "syn", 1)):
        n = 0
        tries = 0
        while n < QUOTA["adversarial"][key]:
            tries += 1
            if tries > QUOTA["adversarial"][key] * 80:
                raise AssertionError(f"[value_card build fail-closed] adv/{key} 只造出 {n}")
            carrier = order[ptr % len(order)]
            ptr += 1
            cand_entries = [w for w in tbl if w not in carrier]
            if not cand_entries:
                continue
            entry = rng.choice(cand_entries)
            text, s, e = inject(carrier, entry, rng)
            if text in used_text or not matches_ok(kind, text, s, entry):
                continue
            used_text.add(text)
            rows.append(row(f"adv-syn-{key}-{n:05d}", "adversarial",
                            f"adv_syn_{key}", label, text, (s, e), entry, kind, carrier))
            n += 1
    n = 0
    tries = 0
    while n < QUOTA["adversarial"]["bg"]:
        tries += 1
        if tries > 20000:
            raise AssertionError(f"[value_card build fail-closed] adv/mention 只造出 {n}")
        off_case = n % 2 == 0
        tpl = rng.choice(MENTION_TPL_OFF if off_case else MENTION_TPL_POL)
        entry = rng.choice(OFF if off_case else POL)
        text = tpl.format(w=entry)
        check(len(text) <= MAX_LEN, f"提及句超长：{text!r}")
        s = text.index(entry)
        if text in used_text or not matches_ok("mention", text, s, entry):
            continue
        used_text.add(text)
        rows.append(row(f"adv-mention-{n:05d}", "adversarial", "adv_mention", 0, text,
                        (s, s + len(entry)), entry, "mention", ""))
        n += 1
    return rows


# ---------------------------------------------------------------------------
# 手写金标准（V3 外部代理真值）
# ---------------------------------------------------------------------------
def load_gold(used_text: set[str]) -> list[dict]:
    """读手写金标准：`标签<TAB>带⟦⟧标记的句子`，标记即 span。

    金标准**不经**载体句/注入管线（独立于构造），非背景必须恰一个标记，
    背景允许 0 个（中性句）或 1 个（提及句）。
    """
    src = HERE / "gold" / "gold_hand.txt"
    rows = []
    for i, line in enumerate(src.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        lab_s, text = line.split("\t", 1)
        check(lab_s in INV_CLS.values(), f"gold 第 {i} 行标签非法：{lab_s!r}")
        label = {"背景": 0, "得体": 1, "冒犯": 2}[lab_s]
        # 扫描 ⟦⟧：plain 保留被标记的文字，marks 记 (start, end, 字面)
        plain_chars: list[str] = []
        marks: list[tuple[int, int, str]] = []
        buf = ""
        cur: int | None = None
        pos = 0
        for ch in text:
            if ch == GOLD_MARK_OPEN:
                check(cur is None, f"gold 第 {i} 行标记嵌套：{text!r}")
                cur = pos
                buf = ""
            elif ch == GOLD_MARK_CLOSE:
                check(cur is not None, f"gold 第 {i} 行出现孤立的右标记：{text!r}")
                marks.append((cur, pos, buf))
                cur = None
            else:
                plain_chars.append(ch)
                if cur is not None:
                    buf += ch
                pos += 1
        check(cur is None, f"gold 第 {i} 行有未闭合的标记：{text!r}")
        plain = "".join(plain_chars)
        marked = [m[2] for m in marks]
        if label == 0:
            check(len(marks) <= 1, f"gold 第 {i} 行背景类却有 {len(marks)} 个标记")
        else:
            check(len(marks) == 1, f"gold 第 {i} 行非背景必须恰有 1 个标记，实为 {len(marks)}")
        s, e = (marks[0][0], marks[0][1]) if marks else (-1, -1)
        check(len(plain) <= MAX_LEN, f"gold 第 {i} 行超长：{plain!r}")
        check(plain not in used_text, f"gold 第 {i} 行与构造集文本重复：{plain!r}")
        used_text.add(plain)
        rows.append(row(f"gold-{i:03d}", "gold", "hand", label, plain, (s, e),
                        marked[0] if marked else "", "gold", ""))
    check(len(rows) == 120, f"gold 规模 {len(rows)} != 120（手写文件被改过？）")
    dist = Counter(r["label"] for r in rows)
    check(dist == {0: 40, 1: 40, 2: 40}, f"gold 不是 40/40/40：{dict(dist)}")
    return rows


# ---------------------------------------------------------------------------
# 统计与校验
# ---------------------------------------------------------------------------
def naive_acc(rows: list[dict]) -> float:
    hit = 0
    for r in rows:
        p_cls, p_s, p_e = naive_rule(r["text"])
        if r["label"] == 0:
            ok = p_cls == 0
        else:
            ok = p_cls == r["label"] and [p_s, p_e] == r["span"]
        hit += ok
    return hit / max(1, len(rows))


def naive_cls_acc(rows: list[dict]) -> float:
    hit = sum(1 for r in rows if naive_rule(r["text"])[0] == r["label"])
    return hit / max(1, len(rows))


def report(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    dist = Counter(r["label"] for r in rows)
    check(n > 0, f"{name} 空")
    texts = [r["text"] for r in rows]
    check(len(set(texts)) == n, f"{name} 文本重复 {n - len(set(texts))} 条")
    for r in rows:
        check(bool(r["spans"]) == bool(r["label"]), f"标签与 span 不一致：{r['id']}")
        check(r["label"] in (0, 1, 2), f"标签越界：{r['id']}")
        if r["label"]:
            s = r["spans"][0]
            check(0 <= s["start"] < s["end"] <= len(r["text"]), f"span 越界：{r['id']}")
    return {
        "n": n,
        "class_dist": {INV_CLS[k]: dist.get(k, 0) for k in (0, 1, 2)},
        "rule_dist": dict(Counter(r["rule"] for r in rows)),
        "majority_baseline": round(max(dist.values()) / n, 4),
        "majority_class": INV_CLS[max(dist, key=lambda k: dist[k])],
        "blind_guess_3cls": round(1 / 3, 4),
        "naive_rule_exact": round(naive_acc(rows), 4),
        "naive_rule_cls": round(naive_cls_acc(rows), 4),
        "mean_text_len": round(sum(len(r["text"]) for r in rows) / n, 2),
        "max_text_len": max(len(r["text"]) for r in rows),
    }


def overlap(a: list[dict], b: list[dict]) -> dict:
    ta, tb = {r["text"] for r in a}, {r["text"] for r in b}
    ca, cb = {r["carrier"] for r in a if r["carrier"]}, {r["carrier"] for r in b if r["carrier"]}
    return {"text": len(ta & tb), "carrier": len(ca & cb)}


def main() -> None:
    lexicon_sanity()
    rng = random.Random(SPLIT_SEED)
    DATA.mkdir(parents=True, exist_ok=True)

    print("[1/5] 取语料句子（resolve_corpus_files，fail-closed）...", flush=True)
    sents, files = collect_sentences(NEED_SENT)
    print(f"  {len(sents)} 句 / {len(files)} 个文件", flush=True)

    shuffled = list(sents)
    rng.shuffle(shuffled)
    pools, at = {}, 0
    for name, size in POOL_N.items():
        pools[name] = shuffled[at:at + size]
        at += size
    for nm, p in pools.items():
        check(len(p) == POOL_N[nm], f"{nm} 池大小 {len(p)} != {POOL_N[nm]}")
    for a in pools:
        for b in pools:
            if a < b:
                ov = len(set(pools[a]) & set(pools[b]))
                check(ov == 0, f"源句池 {a}∩{b} 重叠 {ov} 条")

    print("[2/5] 构造 train / test ...", flush=True)
    used_text: set[str] = set()
    train = build_split("train", QUOTA["train"], pools["train"], rng, used_text)
    test = build_split("test", QUOTA["test"], pools["test"], rng, used_text)
    print("[3/5] 构造 adversarial（不进训练）...", flush=True)
    adv = build_adversarial(pools["adv"], rng, used_text)
    gold = load_gold(used_text)
    for nm, rows, want in (("train", train, 8100), ("test", test, 1500),
                           ("adversarial", adv, 600)):
        check(len(rows) == want, f"{nm} 规模 {len(rows)} != {want}")
        want_dist = {0: QUOTA[nm]["bg"], 1: QUOTA[nm]["pol"], 2: QUOTA[nm]["off"]}
        got = Counter(r["label"] for r in rows)
        check(all(got.get(k, 0) == v for k, v in want_dist.items()),
              f"{nm} 类别比不对：{dict(got)} != {want_dist}")

    print("[4/5] 两两不相交实测 ...", flush=True)
    sets = {"train": train, "test": test, "adversarial": adv, "gold": gold}
    pairs: dict[str, dict] = {}
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            ov = overlap(sets[a], sets[b])
            pairs[f"{a}∩{b}"] = ov
            check(all(v == 0 for v in ov.values()), f"{a}∩{b} 有重叠：{ov}")
            print(f"  {a}∩{b}: text={ov['text']} carrier={ov['carrier']}", flush=True)
    all_texts = [r["text"] for rows in sets.values() for r in rows]
    check(len(set(all_texts)) == len(all_texts), "四集并集大小 != 各自之和")

    print("[5/5] 落盘 ...", flush=True)
    for nm, rows in (("train", train), ("test", test), ("adversarial", adv),
                     ("gold", gold)):
        with open(DATA / f"{nm}.jsonl", "w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")

    stats = {
        "split_seed": SPLIT_SEED,
        "corpus_files_used": files,
        "sentence_pools": {k: len(v) for k, v in pools.items()},
        "lexicon": {"OFF": len(OFF), "POL": len(POL), "NEU": len(NEU),
                    "SYN_OFF": len(SYN_OFF), "SYN_POL": len(SYN_POL),
                    "MENTION_TPL": len(MENTION_TPL_OFF) + len(MENTION_TPL_POL)},
        "train": report("train", train),
        "test": report("test", test),
        "adversarial": report("adversarial", adv),
        "gold": report("gold", gold),
        "overlap_check": pairs,
        "union_text_unique": len(set(all_texts)),
        "max_len": MAX_LEN,
        "label_criterion": ("冒犯=使用中的贬损表达；得体=使用中的礼貌表达；"
                            "背景=无值判断表达或词表词仅被提及"),
        "naive_rule": "只扫 OFF∪POL 词表：最早命中⇒发射该类；不命中⇒背景（不发射）",
        "class_id": {"背景": 0, "得体": 1, "冒犯": 2},
    }
    for k in ("train", "test", "adversarial", "gold"):
        check(stats[k]["max_text_len"] <= MAX_LEN, f"{k} 有样本超过 max_len")
    print(json.dumps({k: stats[k] for k in ("train", "test", "adversarial", "gold",
                                            "overlap_check", "lexicon",
                                            "union_text_unique")},
                     ensure_ascii=False, indent=2))
    (DATA / "stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print("BUILD_OK", flush=True)


if __name__ == "__main__":
    main()
