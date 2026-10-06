#!/usr/bin/env python3
"""two_channel_head 生成分支数据集构建（骨架 id 多分类 + 槽位指派，不发字）。

口径来源：
  · `dev-notes/16` §7.7 重整层输出 = **骨架 id + 槽位指派**（不发字），
    规则 0（指派独立于袋序）/ 规则 A（骨架字面只许标点与形式虚词）/ 规则 B（槽位带类型）。
  · 语料入口**只用** `dtseek.tasks.corpus.resolve_corpus_files`（glob 空即抛，不绕过）。

构造（全部确定性，SPLIT_SEED=20240930 / BAG_SEED=77001）：
  1. 语料按**汉字占比 ≥ MIN_CJK_RATIO 过滤文件**（不依赖文件顺序 —— board m_151：
     `code_alpaca_dialogue.txt` 汉字占比 0.000 却排在最前），逐行去角色前缀后按 [。！？；\\n] 切句；
  2. 句子须匹配**一张手写骨架表**（≤40 个，`[n]` 占位，槽位与字面**严格交替** ⇒ 切分唯一）：
     匹配 = 模板字面按序出现、每槽恰占一个非空间隙、句首/句尾间隙按模板强制为空；
     多骨架匹配 ⇒ 取**字面字符数最多**者，平手取 id 小者（跑前写死）；
  3. 正例 = (骨架 id, 槽位指派)；袋 = 各槽的间隙文本，**按 seed 洗牌**（规则 0：指派独立于袋序），
     指派标签 = 槽 j 在洗牌袋中的下标；
  4. 负例由训练时的 CE 隐式给出（骨架 = Ns 类多分类；指派 = 每槽对 n 项的多分类）。

零训练表面基线（stats 必报，W4）：多数类 / 长度桶 / 首字 / 尾字 / 标点型 /
形式虚词决策表 / 深度 2 决策树 —— `max_naive` 在 **train** 上必须 < 90%。
另有 `annotation_inverse`（标注函数的逆：匹配器自复现 + 按位置排序还原指派）单列披露。

用法：
  uv run python experiments/two_channel_head/build_gen_data.py
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

from dtseek.tasks.corpus import CORPUS_GLOB, resolve_corpus_files  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
SPLIT_SEED = 20240930
BAG_SEED = 77001

#: 截断守卫口径：与训练同口径（tokenizer.encode(text, max_length=...)，超长即丢弃）
SPEC = {"max_len_sent": 64}

N_TRAIN, N_TEST = 8000, 2500
MIN_CJK_RATIO = 0.6          # 文件级与句级双重语言过滤（不依赖文件顺序）
SENT_LEN = (14, 44)          # 与 select_rerank 同口径的句长带
GAP_LEN = (2, 30)            # 单个槽位（间隙）长度带
MIN_CLASS = 30               # 语料命中数 < 30 的骨架从表里剔除（否则是永远学不会的孤类）

BAD = re.compile(r"[�\t｜|]|原文：|候选：")
CJK = re.compile(r"[一-龥]")
ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
SENT_SPLIT = re.compile(r"[。！？\n；;]+")
ASCII_WORD = re.compile(r"[A-Za-z]{3,}")
#: 半角 → 全角（语料 63% 的句子带半角标点；不归一化则 `[1]，[2]` 之类的骨架
#: 匹配不到，标签会退化成"哪个虚词恰好出现"的噪声 —— 实测踩过）
PUNCT_NORM = str.maketrans({",": "，", "!": "！", "?": "？", ":": "：", ";": "；"})
#: 句尾残留的弱标点（行被切开留下的），匹配前剥掉
TRAIL_WEAK = ",，、；：;:\""

# ---------------------------------------------------------------------------
# 手写骨架表（规则 A：字面只许标点与形式虚词 / 逻辑虚词；实义词一律进槽）
# 语法：[1][2]... 槽位；槽位与字面**严格交替**（相邻两槽之间必须有非空字面）
# ---------------------------------------------------------------------------
SKELETONS: list[str] = [
    # —— 纯标点结构（最常见）——
    "[1]，[2]",
    "[1]，[2]，[3]",
    "[1]，[2]，[3]，[4]",
    "[1]、[2]",
    "[1]、[2]、[3]",
    "[1]：[2]",
    # —— 因果 / 假设 / 转折 ——
    "因为[1]，[2]",
    "[1]，因为[2]",
    "因为[1]，所以[2]",
    "[1]，所以[2]",
    "[1]，因此[2]",
    "如果[1]，[2]",
    "[1]，就[2]",
    "虽然[1]，但是[2]",
    "[1]，但是[2]",
    "[1]，不过[2]",
    "[1]，然而[2]",
    # —— 并列 / 递进 / 顺承 / 选择 ——
    "[1]，而且[2]",
    "[1]，并且[2]",
    "[1]，然后[2]",
    "[1]，或者[2]",
    "[1]，要不[2]",
    # —— 时间 / 条件从句 ——
    "[1]的时候，[2]",
    "[1]的话，[2]",
    "[1]之前，[2]",
    "[1]之后，[2]",
    "为了[1]，[2]",
    # —— 介词 / 特殊句式（把 / 被 / 是…的 / 从…到） ——
    "[1]把[2]",
    "[1]把[2]了",
    "[1]被[2]",
    "[1]被[2]了",
    "[1]是[2]的",
    "[1]从[2]到[3]",
    "[1]让[2]",
    "[1]比[2]",
    # —— 语气（句尾语气词）——
    "[1]吗",
    "[1]吧",
    "[1]呢",
    "[1]，[2]吗",
    "[1]，[2]吧",
]
assert len(SKELETONS) <= 40, len(SKELETONS)

#: 形式虚词 / 逻辑虚词清单（W4 的朴素规则用；不含实义词）
FUNCTION_WORDS = [
    "因为", "所以", "因此", "如果", "虽然", "但是", "不过", "然而", "而且", "并且",
    "然后", "或者", "要不", "为了", "的话", "的时候", "之前", "之后", "把", "被",
    "让", "比", "从", "到", "是", "的", "了", "吗", "吧", "呢", "就", "要", "不",
    "没", "有", "在", "和", "与", "很", "都", "也", "还", "再", "又", "会", "能",
]
PUNCTS = "，、：；！？…—"
ALLOWED_LIT_CHARS = set(PUNCTS) | set("".join(FUNCTION_WORDS))


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[two_channel_head build fail-closed] {msg}")


# ---------------------------------------------------------------------------
# 骨架解析与匹配
# ---------------------------------------------------------------------------
class Skeleton:
    def __init__(self, sid: int, template: str):
        self.id = sid
        self.template = template
        parts = re.split(r"\[\d+\]", template)
        slots = re.findall(r"\[(\d+)\]", template)
        check([int(s) for s in slots] == list(range(1, len(slots) + 1)),
              f"骨架 #{sid} 槽位编号必须从 1 连续：{template}")
        k = len(slots)
        check(k >= 1, f"骨架 #{sid} 至少 1 个槽：{template}")
        check(k <= 6, f"骨架 #{sid} 槽位数 {k} > 6：{template}")
        # 严格交替：相邻两槽之间的字面必须非空（否则切分不唯一）
        for i in range(1, k):
            check(parts[i], f"骨架 #{sid} 相邻槽之间字面为空（切分歧义）：{template}")
        # 规则 A：字面只许标点与形式虚词
        for p in parts:
            bad = set(p) - ALLOWED_LIT_CHARS
            check(not bad, f"骨架 #{sid} 字面含非形式词 {sorted(bad)}：{template}")
        self.parts = parts                       # k+1 个字面（首尾可为空）
        self.k = k
        self.literal_chars = sum(len(p) for p in parts)
        check(self.literal_chars >= 1,
              f"骨架 #{sid} 全是槽、无字面（= 整句槽，类别会吞掉全部数据）：{template}")
        # 模板 → 正则：^ lit0 (g1) lit1 (g2) ... (gk) litk $
        rx = "^"
        for i in range(k + 1):
            if parts[i]:
                rx += re.escape(parts[i])
            if i < k:
                rx += f"(?P<g{i + 1}>.+?)"
        self.rx = re.compile(rx)

    def match(self, sent: str) -> list[str] | None:
        m = self.rx.fullmatch(sent)
        if m is None:
            return None
        # 规则 A′（锚定）：**含形式虚词的字面**必须被标点或句界锚定 ——
        # 否则它可能只是内容词的一部分（实测：`[1]比[2]` 匹配到"比较方便"、
        # `[1]让[2]` 匹配到"让自己"、`[1]是[2]的` 匹配到"这是…应该的"）。
        pos = 0
        for i in range(self.k + 1):
            part = self.parts[i]
            if part and (set(part) - set(PUNCTS)):
                # 字面自己以标点开头/结尾 ⇒ 天然锚定（如 "，所以"）；
                # 否则要求紧邻标点或句界。
                left_ok = (part[0] in PUNCTS or pos == 0
                           or sent[pos - 1] in PUNCTS)
                right_ok = (part[-1] in PUNCTS or pos + len(part) == len(sent)
                            or sent[pos + len(part)] in PUNCTS)
                if not (left_ok or right_ok):
                    return None
                pos += len(part)
            if i < self.k:
                pos = m.end(f"g{i + 1}")
        gaps = [m.group(f"g{i + 1}") for i in range(self.k)]
        spans = [(m.start(f"g{i + 1}"), m.end(f"g{i + 1}")) for i in range(self.k)]
        if any(len(g) < GAP_LEN[0] or len(g) > GAP_LEN[1] for g in gaps):
            return None
        return gaps, spans


SKELS = [Skeleton(i, t) for i, t in enumerate(SKELETONS)]
#: 匹配优先级（跑前写死）：字面字符数多者优先，平手取 id 小者
ORDER = sorted(range(len(SKELS)), key=lambda i: (-SKELS[i].literal_chars, i))


def match_sentence(sent: str) -> tuple[int, list[str], list[tuple[int, int]]] | None:
    """返回 (骨架 id, 各槽文本, 各槽在句中的 [start,end))；全部按匹配优先级取第一个命中。"""
    for i in ORDER:
        res = SKELS[i].match(sent)
        if res is not None:
            return i, res[0], res[1]
    return None


# ---------------------------------------------------------------------------
# 语料（语言过滤 + fail-closed）
# ---------------------------------------------------------------------------
def ok_sentence(s: str) -> bool:
    if not (SENT_LEN[0] <= len(s) <= SENT_LEN[1]):
        return False
    if BAD.search(s) or "|" in s or ASCII_WORD.search(s):
        return False
    if len(CJK.findall(s)) < MIN_CJK_RATIO * len(s):    # 句级语言过滤
        return False
    return True


def load_corpus(n_want: int) -> tuple[list[str], dict]:
    """按文件级汉字占比过滤后取句；返回 (句子, 每文件语言统计)。"""
    files = resolve_corpus_files(CORPUS_GLOB)           # fail-closed
    lang: dict[str, dict] = {}
    kept: list[str] = []
    rnd = random.Random(SPLIT_SEED)
    for p in files:
        txt = Path(p).read_text(encoding="utf-8", errors="ignore")
        ratio = len(CJK.findall(txt)) / max(1, len(txt))
        lang[Path(p).name] = {"cjk_ratio": round(ratio, 4), "kept": ratio >= MIN_CJK_RATIO}
        if ratio >= MIN_CJK_RATIO:
            kept.append(p)
    check(kept, "语言过滤后一个语料文件都不剩（MIN_CJK_RATIO 太高）")
    rnd.shuffle(kept)                                    # 打乱文件序，不依赖顺序
    seen: set[str] = set()
    out: list[str] = []
    for p in kept:
        for line in Path(p).read_text(encoding="utf-8", errors="ignore").splitlines():
            t = ROLE_PREFIX.sub("", line.strip()).translate(PUNCT_NORM)
            for s in SENT_SPLIT.split(t):
                s = s.strip().strip(TRAIL_WEAK)
                if not ok_sentence(s) or s in seen:
                    continue
                seen.add(s)
                out.append(s)
                if len(out) >= n_want:
                    return out, lang
    check(len(out) >= n_want, f"语料可用句子不足：{len(out)} < {n_want}")
    return out, lang


# ---------------------------------------------------------------------------
# 朴素规则电池（W4）
# ---------------------------------------------------------------------------
def _majority(labels: list[int]) -> int:
    return Counter(labels).most_common(1)[0][0]


def _acc(pred, truth) -> float:
    return sum(1 for a, b in zip(pred, truth) if a == b) / len(truth)


def _fit_predict(feat, fit_rows, eval_rows, min_support=30) -> list[int]:
    """按特征分组取**训练集**多数类；支持数不足的组回退到全局多数类。"""
    groups: dict = defaultdict(list)
    for r in fit_rows:
        groups[feat(r)].append(r["skel_id"])
    glob = _majority([r["skel_id"] for r in fit_rows])
    table = {f: Counter(v).most_common(1)[0][0] for f, v in groups.items()
             if len(v) >= min_support}
    return [table.get(feat(r), glob) for r in eval_rows]


def feats(r: dict) -> tuple:
    t = r["sent"]
    return tuple([int(w in t) for w in FUNCTION_WORDS] +
                 [int(c in t) for c in PUNCTS] + [len(t) // 8])


def _tree_depth(X, y, Xq, depth0: int) -> list[int]:
    """贪心决策树（叶 = 训练集多数类）；末维长度桶按阈值二分。"""
    def _imp(idx):
        if not idx:
            return 0
        return len(idx) - max(Counter(y[i] for i in idx).values())

    def build(idx, depth):
        if depth == 0 or len(idx) < 40:
            return ("leaf", Counter(y[i] for i in idx).most_common(1)[0][0])
        best = None
        nf = len(X[0])
        for f in range(nf):
            if f == nf - 1:
                vals = sorted(set(X[i][f] for i in idx))
                cuts = [(a + b) / 2 for a, b in zip(vals, vals[1:])]
            else:
                cuts = [0.5]
            for c in cuts:
                L = [i for i in idx if X[i][f] <= c]
                R = [i for i in idx if X[i][f] > c]
                if len(L) < 20 or len(R) < 20:
                    continue
                gain = len(idx) - _imp(L) - _imp(R)
                if best is None or gain > best[0]:
                    best = (gain, f, c, L, R)
        if best is None:
            return ("leaf", Counter(y[i] for i in idx).most_common(1)[0][0])
        _, f, c, L, R = best
        return ("node", f, c, build(L, depth - 1), build(R, depth - 1))

    def query(node, x):
        if node[0] == "leaf":
            return node[1]
        _, f, c, L, R = node
        return query(L if x[f] <= c else R, x)

    tree = build(list(range(len(y))), depth0)
    return [query(tree, x) for x in Xq]


def naive_skeleton(fit_rows: list[dict], eval_rows: list[dict]) -> dict:
    """朴素规则电池在 eval_rows 上的骨架准确率（W4 用 fit=train）。"""
    y = [r["skel_id"] for r in eval_rows]
    n = len(y)
    out: dict[str, float] = {}
    glob = _majority([r["skel_id"] for r in fit_rows])
    out["majority"] = sum(1 for v in y if v == glob) / n
    out["len_bucket"] = _acc(_fit_predict(lambda r: len(r["sent"]) // 6,
                                          fit_rows, eval_rows), y)
    out["first_char"] = _acc(_fit_predict(lambda r: r["sent"][0], fit_rows, eval_rows), y)
    out["last_char"] = _acc(_fit_predict(lambda r: r["sent"][-1], fit_rows, eval_rows), y)
    out["punct_pattern"] = _acc(
        _fit_predict(lambda r: "".join(sorted(set(r["sent"]) & set(PUNCTS))),
                     fit_rows, eval_rows), y)
    # 形式虚词决策表：命中数最多的虚词优先，输出该虚词子集上的最优类；无命中回退多数类
    cnt = Counter(w for r in fit_rows for w in FUNCTION_WORDS if w in r["sent"])
    order_w = sorted(FUNCTION_WORDS, key=lambda w: -cnt.get(w, 0))
    best: dict[str, int] = {}
    for w in order_w:
        sub = [r["skel_id"] for r in fit_rows if w in r["sent"]]
        if len(sub) >= 30:
            best[w] = _majority(sub)
    preds = []
    for r in eval_rows:
        p = glob
        for w in order_w:
            if w in r["sent"] and w in best:
                p = best[w]
                break
        preds.append(p)
    out["fw_decision_list"] = _acc(preds, y)
    Xf = [feats(r) for r in fit_rows]
    yf = [r["skel_id"] for r in fit_rows]
    out["tree_depth2"] = _acc(_tree_depth(Xf, yf, [feats(r) for r in eval_rows], 2), y)
    out["tree_depth4"] = _acc(_tree_depth(Xf, yf, [feats(r) for r in eval_rows], 4), y)
    out["max_naive"] = max(out.values())
    return out


def naive_assign(fit_rows: list[dict], eval_rows: list[dict]) -> dict:
    """指派通道的朴素规则（袋序随机 ⇒ 单特征规则应≈盲猜）。"""
    out: dict[str, float] = {}
    ok = tot = 0
    for r in eval_rows:
        for j, idx in enumerate(r["assign"]):
            tot += 1
            ok += int(idx == j)                       # 恒等规则
    out["identity_rule"] = ok / tot
    per_slot: dict[int, Counter] = defaultdict(Counter)
    for r in fit_rows:
        for j, idx in enumerate(r["assign"]):
            per_slot[j][idx] += 1
    best = {j: c.most_common(1)[0][0] for j, c in per_slot.items()}
    ok = tot = 0
    for r in eval_rows:
        for j, idx in enumerate(r["assign"]):
            tot += 1
            ok += int(idx == best.get(j, 0))          # 逐槽多数下标
    out["slot_majority_rule"] = ok / tot
    out["blind_uniform"] = sum(1.0 / len(r["assign"]) for r in eval_rows) / len(eval_rows)
    # 标注函数的逆：按句中位置排序还原指派（**单列披露**，不计入 max_naive）
    ok = tot = 0
    for r in eval_rows:
        pos = [r["sent"].find(it) for it in r["bag"]]
        order = sorted(range(len(pos)), key=lambda i: (pos[i] if pos[i] >= 0 else 10 ** 6))
        pred = [0] * len(pos)
        for slot, bag_i in enumerate(order):
            pred[slot] = bag_i
        for j, idx in enumerate(r["assign"]):
            tot += 1
            ok += int(pred[j] == idx)
    out["annotation_inverse_pos_sort"] = ok / tot
    return out


# ---------------------------------------------------------------------------
# 构建
# ---------------------------------------------------------------------------
def _apportion(counts: dict, total: int) -> dict:
    """按 counts 比例把 total 个名额分到各类（最大余数法），结果和恰为 total。"""
    n_all = sum(counts.values())
    raw = {k: v * total / n_all for k, v in counts.items()}
    out = {k: int(raw[k]) for k in counts}
    left = total - sum(out.values())
    for k in sorted(raw, key=lambda k: (-(raw[k] - int(raw[k])), k))[:left]:
        out[k] += 1
    return out


def stratify(matched: list, n_train: int, n_test: int) -> tuple[list, list]:
    """按骨架分层、比例抽样 train/test（两集合骨架分布一致，且句子互斥）。"""
    by: dict[int, list] = defaultdict(list)
    for rec in matched:
        by[rec[1]].append(rec)
    rnd = random.Random(SPLIT_SEED + 1)
    for v in by.values():
        rnd.shuffle(v)
    check(len(by) >= 20, f"只有 {len(by)} 个骨架被命中，骨架表覆盖不足")
    counts = {k: len(v) for k, v in by.items()}
    quota_train = _apportion(counts, n_train)
    quota_test = _apportion(counts, n_test)
    check(all(quota_train[k] + quota_test[k] <= counts[k] for k in counts),
          f"配额超出各类可得量：{[(k, quota_train[k] + quota_test[k], counts[k]) for k in counts if quota_train[k] + quota_test[k] > counts[k]]}")
    train, test = [], []
    for k in sorted(by):
        n_tr, n_te = quota_train[k], quota_test[k]
        train += by[k][:n_tr]
        test += by[k][n_tr:n_tr + n_te]
    check(len(train) == n_train and len(test) == n_test,
          f"分层抽样不足：train={len(train)} test={len(test)}（匹配到 {len(matched)}）")
    check(not (set(map(id, train)) & set(map(id, test))), "train/test 记录重叠")
    return train, test


def to_rows(recs: list, split: str) -> list[dict]:
    rows = []
    srng = random.Random(BAG_SEED + {"train": 1, "test": 2}[split])
    for sent, sid, gaps, spans in recs:
        n = len(gaps)
        # 规则 0：袋序取**全部置换上的均匀分布**（含恒等）——
        # 曾用「拒绝恒等置换」的写法，n=2 时只剩唯一置换 ⇒ 指派标签恒为 [1,0]，
        # slot_majority 规则在 train 上直接 81%（捷径），已修。
        perm = list(range(n))
        srng.shuffle(perm)
        bag = [gaps[i] for i in perm]                    # 洗牌后的袋
        assign = [perm.index(slot) for slot in range(n)]  # 槽 slot → 洗牌袋下标
        bag_span = [spans[i] for i in perm]                # 洗牌后的袋（带 span）
        rows.append({"sent": sent, "skel_id": sid, "skel": SKELS[sid].template,
                     "n_slots": n, "bag": bag, "bag_span": bag_span,
                     "assign": assign, "split": split})
    return rows


def build() -> dict:
    pool, lang = load_corpus(N_TRAIN + N_TEST + 40000)
    print(f"[corpus] 取到 {len(pool)} 句（文件级汉字比 ≥{MIN_CJK_RATIO}，已打乱文件序）")

    matched: list = []
    n_no_match = 0
    for s in pool:
        m = match_sentence(s)
        if m is None:
            n_no_match += 1
            continue
        matched.append((s, m[0], m[1], m[2]))
    # 剔除语料里几乎命不中的骨架（孤类永远学不会，进测试集只会白扣分）
    pool_cnt = Counter(r[1] for r in matched)
    dead = sorted(k for k, v in pool_cnt.items() if v < MIN_CLASS)
    if dead:
        matched = [r for r in matched if pool_cnt[r[1]] >= MIN_CLASS]
        n_no_match += sum(pool_cnt[k] for k in dead)
        print(f"[filter] 剔除低命中骨架 {[(k, SKELS[k].template, pool_cnt[k]) for k in dead]}")
    print(f"[match] 命中 {len(matched)} / {len(pool)} = "
          f"{len(matched) / max(1, len(pool)):.4f}（无匹配/低命中丢弃 {n_no_match}）")
    check(len(matched) >= N_TRAIN + N_TEST,
          f"匹配到的句子不足：{len(matched)} < {N_TRAIN + N_TEST}")

    train_recs, test_recs = stratify(matched, N_TRAIN, N_TEST)
    rows_train = to_rows(train_recs, "train")
    rows_test = to_rows(test_recs, "test")

    # ---- 截断守卫（与训练同口径 tokenizer）----
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    trunc = 0
    for r in rows_train + rows_test:
        e = tok.encode(r["sent"], max_length=SPEC["max_len_sent"], padding=False)
        m = list(e["attention_mask"])
        if sum(m) < len(r["sent"]):
            trunc += 1
    check(trunc == 0, f"截断丢弃 {trunc} 条（句长带与 max_len 不相容）")

    # ---- 统计 ----
    tr_cnt = Counter(r["skel_id"] for r in rows_train)
    majority = tr_cnt.most_common(1)[0]
    stats: dict = {
        "spec": SPEC,
        "split_seed": SPLIT_SEED,
        "bag_seed": BAG_SEED,
        "n_skeletons": len(SKELS),
        "skeletons": {s.id: s.template for s in SKELS},
        "match_priority": "字面字符数降序 → id 升序（跑前写死）",
        "corpus": {
            "files_total": len(lang),
            "files_kept": sum(1 for v in lang.values() if v["kept"]),
            "files_dropped": {k: v["cjk_ratio"] for k, v in lang.items() if not v["kept"]},
            "min_cjk_ratio": MIN_CJK_RATIO,
            "sentences_considered": len(pool),
            "matched": len(matched),
            "coverage": round(len(matched) / max(1, len(pool)), 4),
            "drop_no_match": n_no_match,
            "dropped_skeletons": {SKELS[k].template: pool_cnt[k] for k in dead},
            "n_skeletons_effective": len(SKELS) - len(dead),
        },
        "sets": {},
    }
    for name, rows in (("train", rows_train), ("test", rows_test)):
        cnt = Counter(r["skel_id"] for r in rows)
        stats["sets"][name] = {
            "n": len(rows),
            "n_classes": len(cnt),
            "label_dist": dict(sorted(cnt.items())),
            "majority_class": majority[0],
            "majority_baseline": round(cnt[majority[0]] / len(rows), 4),
            "blind_guess": round(1 / len(SKELS), 4),
            "slots_dist": dict(sorted(Counter(r["n_slots"] for r in rows).items())),
            "len": {"min": min(len(r["sent"]) for r in rows),
                    "max": max(len(r["sent"]) for r in rows),
                    "mean": round(sum(len(r["sent"]) for r in rows) / len(rows), 2)},
            "trunc_dropped": 0,
        }
    stats["naive_skeleton_train"] = {k: round(v, 4) for k, v in
                                     naive_skeleton(rows_train, rows_train).items()}
    stats["naive_skeleton_test"] = {k: round(v, 4) for k, v in
                                    naive_skeleton(rows_train, rows_test).items()}
    stats["naive_assign"] = {k: round(v, 4) for k, v in
                             naive_assign(rows_train, rows_test).items()}
    stats["annotation_inverse"] = {
        "matcher_self_reproduction": 1.0,
        "note": "标签由确定性匹配器定义（与任何模板数据集同构）；此项为标注函数的逆运算，"
                "按 select_rerank 既有口径不计入 max_naive",
    }
    stats["max_naive_train"] = stats["naive_skeleton_train"]["max_naive"]
    stats["w4_pass"] = bool(stats["max_naive_train"] < 0.90)
    for r in rows_train + rows_test:
        assert [r["sent"][a:b] for a, b in r["bag_span"]] == r["bag"], "span 与 bag 文本不一致"
    stats["overlap"] = {"train_test_sentence": len(
        {r["sent"] for r in rows_train} & {r["sent"] for r in rows_test})}
    check(stats["overlap"]["train_test_sentence"] == 0, "train/test 句子重叠")

    print("[samples]")
    for r in rows_train[:8]:
        print(f"  #{r['skel_id']:02d} {r['skel']:<18} | {r['sent']} | "
              f"bag={r['bag']} assign={r['assign']}")
    print(f"[naive] train 骨架: {stats['naive_skeleton_train']}")
    print(f"[naive] test  骨架: {stats['naive_skeleton_test']}")
    print(f"[naive] 指派   : {stats['naive_assign']}")
    print(f"[W4] max_naive(train) = {stats['max_naive_train']:.4f} < 0.90 ? "
          f"{stats['w4_pass']}")

    DATA.mkdir(exist_ok=True)
    for name, rows in (("train", rows_train), ("test", rows_test)):
        with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as fp:
            for r in rows:
                fp.write(json.dumps(r, ensure_ascii=False) + "\n")
    (DATA / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2),
                                     encoding="utf-8")
    print(f"[done] → {DATA}/train.jsonl, {DATA}/test.jsonl, {DATA}/stats.json")
    return stats


if __name__ == "__main__":
    build()
