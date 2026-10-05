"""完形填空（selection）数据集：带空位的句子 + 4 个候选词，选出该填的那个。

任务定义（dev-notes/13 §4.2）：
  - **正例**：真实语料句子里挖一个实词 → 空位 `__` → 正确词进候选列表；
  - **核心风险是答案不唯一**（多个候选填进去都通顺），所以每条样本都要过
    **共现唯一性校验**：用语料的局部二元组共现统计，正确候选与空位左右邻接
    字的共现必须显著高于全部干扰项，不足就**丢样本**（不靠「答案是对的」就算数）；
  - **干扰项同类同频同字长**：同词性、词频落在 `[f/3, 3f+2]`、字长与正确词相同 ——
    否则「挑长的 / 挑常见的」就是表面捷径；
  - **真·无解样本**：同一个空位配 4 个共现明显偏弱的候选 → `spans=[]` → 类别 0。

三条 fail-closed 不变量：
  1. **符号纪律**：构建前跑 `frame.validate_markers()`（`__`=95、`|`=124、`）`=156，
     设计稿的 `＿＿` 实测是 UNK，已换形）；
  2. **唯一性校验不过就丢**：`score(正确) >= MIN_SCORE` 且
     `score(正确) >= MARGIN * max(score(干扰))`，无解样本反过来要求
     `max(score(干扰)) < MIN_SCORE`；
  3. **空位位置配额**：句首 / 句中 / 句末各 >= 15%，且与标点相邻的空位不占绝对多数
     （dev-notes/13 §4.2：空位不能总是被标点锚定）。

词性来源：本仓库没有分词/词性标注器（jieba 等均未安装），所以词表是**手写闭集**
（`POS_LEXICON`，词性由人工判定），每个词的语料词频在扫描时实测。
"""
from __future__ import annotations

import hashlib
import itertools
import random
import re
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field

from dtseek.tasks.builtin.reply_pick.frame import (
    BLANK,
    LIST_OVERHEAD,
    check_text_length,
    content_problems,
    render_choice_list,
    style_problems,
    validate_markers,
)
from dtseek.tasks.corpus import resolve_corpus_files
from dtseek.tasks.plugin import DEFAULT_SEED

LABEL_BG = 0
N_CANDIDATES = 4

#: 整段文本上限（= spec.max_len - 8，引擎 window 分段阈值）。单测钉住这条等式。
TEXT_LIMIT = 88

#: 「无合适候选」样本占比。
BG_RATIO = 0.15

#: 句子切分与长度粗筛（真正的预算逐样本算）。
#: **只按句末标点切**：句内逗号必须留在片段里，否则空位永远不会与标点相邻，
#: dev-notes/13 §4.2 的「不能总是被标点锚定」就变成空判据（探针里的
#: 标点相邻形态也会落在训练分布之外）。
_SENT_SPLIT = re.compile(r"[。！？!?\n]+")
_ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
_SENT_TAIL = "。！？!?…，,、；;"
_SENT_LEN = (6, 70)

#: 唯一性校验阈值（dev-notes/13 §4.2「共现显著高于干扰项」的可执行定义）。
#: score = 空位左邻字与词首字的共现次数 + 词尾字与空位右邻字的共现次数。
MIN_SCORE = 4
MARGIN = 3

#: 目标词的语料词频下限（共现不足 MIN_SCORE 的词当正解只会把样本逼成噪音）。
TARGET_FREQ_MIN = 4

#: 干扰项重抽次数：干扰项集合是自由设计变量，抽到唯一性达标为止；
#: 这不放宽判据 —— 判据只认 `MIN_SCORE / MARGIN`，重抽只是把「抽签」多抽几次。
DISTRACTOR_TRIES = 6

#: 空位位置配额（占正解样本的比例下限），与标点相邻的空位占比上限。
POS_QUOTA = {"句首": 0.15, "句末": 0.15}
PUNCT_ADJACENT_MAX = 0.70

#: 语料扫描行数上限（18 文件轮转）。测试用 max_lines 覆盖。
MAX_SCAN_LINES = 1_200_000

#: 训练/评测折（按句子内容哈希分 10 折，第 9 折只做评测）。
EVAL_FOLD = 9
N_FOLDS = 10

# ---------------------------------------------------------------------------
# 词表（手写闭集，词性人工判定）。同一词只允许出现在一个词类里 ——
# 否则「同类干扰项」会混进不同词性的词，判据本身就不成立了。
# ---------------------------------------------------------------------------
_SENT_TAIL = "。！？!?…，,、；;"
_SENT_LEN = (6, 70)

#: 唯一性校验阈值（dev-notes/13 §4.2「共现显著高于干扰项」的可执行定义）。
#: score = 空位左邻字与词首字的共现次数 + 词尾字与空位右邻字的共现次数。
MIN_SCORE = 4
MARGIN = 3

#: 目标词的语料词频下限（共现不足 MIN_SCORE 的词当正解只会把样本逼成噪音）。
TARGET_FREQ_MIN = 4

#: 干扰项重抽次数：干扰项集合是自由设计变量，抽到唯一性达标为止；
#: 这不放宽判据 —— 判据只认 `MIN_SCORE / MARGIN`，重抽只是把「抽签」多抽几次。
DISTRACTOR_TRIES = 6

#: 空位位置配额（占正解样本的比例下限），与标点相邻的空位占比上限。
POS_QUOTA = {"句首": 0.15, "句末": 0.15}
PUNCT_ADJACENT_MAX = 0.70

#: 语料扫描行数上限（18 文件轮转）。测试用 max_lines 覆盖。
MAX_SCAN_LINES = 1_200_000

#: 训练/评测折（按句子内容哈希分 10 折，第 9 折只做评测）。
EVAL_FOLD = 9
N_FOLDS = 10

# ---------------------------------------------------------------------------
# 词表（手写闭集，词性人工判定）。同一词只允许出现在一个词类里 ——
# 否则「同类干扰项」会混进不同词性的词，判据本身就不成立了。
# ---------------------------------------------------------------------------
POS_LEXICON: dict[str, tuple[str, ...]] = {
    "名词": (
        "学校", "老师", "学生", "朋友", "公司", "电脑", "手机", "时间", "问题", "办法",
        "结果", "原因", "机会", "天气", "交通", "方案", "文件", "数据", "系统", "服务",
        "质量", "效率", "成绩", "能力", "经验", "知识", "世界", "生活", "家庭", "价格",
        "市场", "客户", "项目", "会议", "需求", "目标", "任务", "资料", "文章", "内容",
        "方法", "情况", "状态", "位置", "房子", "车站", "医院", "饭店", "菜单", "衣服",
        "钱包", "钥匙", "报纸", "电视", "电影", "音乐", "图书", "语言", "历史", "地图",
        "阳台", "厨房", "卧室", "沙发", "窗户", "电梯", "街道", "公园", "湖水", "山峰",
        "云朵", "雨水", "花朵", "树林", "田地", "工人", "警察", "医生", "护士", "司机",
        "厨师", "记者", "演员", "歌手", "教练", "学期", "课本", "作业", "考试", "教室",
        "黑板", "书包", "课程", "图书馆", "办公室", "会议室", "说明书", "自行车", "身份证",
        "计算机", "互联网", "停车场", "便利店", "快递员", "星期天", "天气预报", "毕业证",
        "图书室", "储物柜", "早餐店", "火车站", "飞机场", "体育场", "博物馆", "电影院",
    ),
    "动词": (
        "学习", "工作", "讨论", "研究", "准备", "安排", "检查", "完成", "开始", "结束",
        "参加", "解决", "发现", "记得", "选择", "判断", "比较", "观察", "保持", "改变",
        "提高", "减少", "增加", "支持", "反对", "需要", "希望", "觉得", "认为", "打算",
        "决定", "介绍", "说明", "确认", "商量", "建议", "请求", "帮助", "通知", "报告",
        "等待", "迟到", "出发", "回来", "出去", "进门", "敲门", "开门", "洗衣", "做饭",
        "吃饭", "喝水", "睡觉", "休息", "锻炼", "逛街", "旅行", "拍照", "打扫", "修理",
        "种植", "收获", "讲解", "提问", "回答", "背诵", "朗读", "抄写", "搬运", "折叠",
        "排列", "挑选", "打印", "复印", "填写", "登录", "删除", "保存", "下载", "上传",
        "打电话", "看电影", "吃午饭", "去上班", "走出去", "拿回来", "写下来", "说出来",
        "想起来", "忘不了", "睡不着", "走不动", "用不完", "记不住", "分不清", "来不及",
    ),
    "形容词": (
        "重要", "安全", "简单", "复杂", "困难", "容易", "清楚", "明白", "干净", "整洁",
        "热闹", "安静", "漂亮", "丰富", "及时", "准确", "合理", "现代", "传统", "常见",
        "特别", "熟悉", "陌生", "严格", "轻松", "紧张", "忙碌", "清晰", "模糊", "详细",
        "具体", "温暖", "寒冷", "明亮", "潮湿", "干燥", "宽敞", "拥挤", "新鲜", "陈旧",
        "诚实", "聪明", "顽皮", "温柔", "粗心", "认真", "马虎", "耐心", "着急", "满意",
        "失望", "兴奋", "平静", "感动", "尴尬", "关键", "漫长", "短暂", "稀少", "密集",
        "柔软", "坚硬", "光滑", "粗糙", "香甜", "苦涩", "响亮", "低沉", "鲜艳", "暗淡",
    ),
    "副词": (
        "必须", "应该", "可能", "一定", "突然", "逐渐", "立刻", "马上", "已经", "正在",
        "经常", "偶尔", "一起", "互相", "分别", "重新", "仔细", "自然", "慢慢", "悄悄",
        "渐渐", "一直", "总是", "从来", "常常", "终于", "幸亏", "反正", "到底", "难道",
        "居然", "果然", "也许", "大概", "恰好", "正好", "稍微", "尤其", "甚至", "亲自",
        "单独", "顺便", "故意", "尽快", "尽早", "差点", "几乎", "统统", "连忙",
    ),
}

#: 词表全集（`_lexicon_groups` 会顺带校验跨词类重复）。
_ALL_WORDS: tuple[str, ...] = tuple(w for words in POS_LEXICON.values() for w in words)


@dataclass
class ClozePool:
    """一趟扫描的产物：二元组共现、词表词频、可用句子。按 `max_lines` 缓存。"""

    bg: Counter = field(default_factory=Counter)
    freq: Counter = field(default_factory=Counter)
    sentences: list[str] = field(default_factory=list)
    files: int = 0
    lines: int = 0
    raw_sentences: int = 0
    views: dict[str, list[str]] = field(default_factory=dict)


_POOL_CACHE: dict[int, ClozePool] = {}


def _lexicon_groups() -> dict[str, tuple[str, ...]]:
    """按首字分组：扫描时只需查「行内出现过的首字」那一组，避免逐词全表扫描。"""
    groups: dict[str, list[str]] = {}
    seen: set[str] = set()
    for words in POS_LEXICON.values():
        for w in words:
            if w in seen:
                raise ValueError(f"词表跨词类重复：{w!r} —— 同类干扰项的判据会失效")
            seen.add(w)
            groups.setdefault(w[0], []).append(w)
    return {k: tuple(v) for k, v in groups.items()}


def word_class(word: str) -> str:
    """词所属的手写词类；不在词表里直接抛错（fail-closed）。"""
    for pos, words in POS_LEXICON.items():
        if word in words:
            return pos
    raise ValueError(f"{word!r} 不在 POS_LEXICON 里 —— 无法保证「同类」干扰项")


def load_cloze_pool(max_lines: int = MAX_SCAN_LINES, *, refresh: bool = False) -> ClozePool:
    """一趟扫描同时收集二元组共现、词表词频与可用句子（`resolve_corpus_files` 唯一入口）。"""
    if not refresh and max_lines in _POOL_CACHE:
        return _POOL_CACHE[max_lines]

    files = resolve_corpus_files()
    groups = _lexicon_groups()
    start_chars = frozenset(groups)
    pool = ClozePool(files=len(files))
    seen_sents: set[str] = set()

    with ExitStack() as stack:
        fhs = [stack.enter_context(open(f, encoding="utf-8", errors="ignore")) for f in files]
        active = list(range(len(fhs)))
        n_lines = 0
        while active and n_lines < max_lines:
            nxt = []
            for i in active:
                line = fhs[i].readline()
                if not line:
                    continue
                n_lines += 1
                nxt.append(i)
                line = line.rstrip("\n")
                if not line:
                    continue
                # 去掉角色前缀：`模型：是的，…` 直接进句子池会把「模型：」当成句首内容
                line = _ROLE_PREFIX.sub("", line, count=1)
                pool.bg.update(itertools.pairwise(line))
                # 词频：只查行内出现过的首字对应的词组
                hit = start_chars & set(line)
                if hit:
                    for c in hit:
                        for w in groups[c]:
                            n = line.count(w)
                            if n:
                                pool.freq[w] += n
                # 句子：粗筛后进池，词级检查留到构建阶段（那里知道空位在哪）
                for raw in _SENT_SPLIT.split(line):
                    s = raw.strip().rstrip(_SENT_TAIL)
                    if not (_SENT_LEN[0] <= len(s) <= _SENT_LEN[1]) or s in seen_sents:
                        continue
                    if style_problems(s) or content_problems(s):
                        continue
                    if not (start_chars & set(s)):
                        continue
                    seen_sents.add(s)
                    pool.sentences.append(s)
            active = nxt

    pool.lines = n_lines
    _POOL_CACHE[max_lines] = pool
    return pool


def _fold(sentence: str) -> int:
    return int(hashlib.md5(sentence.encode("utf-8")).hexdigest(), 16) % N_FOLDS


def _score(pool: ClozePool, word: str, prev: str | None, nxt: str | None) -> int:
    """空位上下文与候选词的局部共现分（dev-notes/13 §4.2 唯一性校验的度量）。"""
    score = 0
    if prev:
        score += pool.bg.get((prev, word[0]), 0)
    if nxt:
        score += pool.bg.get((word[-1], nxt), 0)
    return score


def _freq_bucket(freq: int) -> tuple[int, int]:
    """同频桶：`[f/3, 3f+2]`（词频太小时给个下限 1，否则桶是空的）。"""
    return max(1, freq // 3), 3 * freq + 2


def _distractors(pool: ClozePool, *, target: str, sentence: str,
                 budget_left: int, rng: random.Random) -> list[str] | None:
    """抽 3 个同词类、同字长、同频桶、不在句中出现的干扰项。"""
    pos = word_class(target)
    length = len(target)
    f = pool.freq[target]
    lo, hi = _freq_bucket(f)
    pool_words = [w for w in POS_LEXICON[pos]
                  if len(w) == length and w != target and w not in sentence
                  and lo <= pool.freq[w] <= hi and len(w) * N_CANDIDATES <= budget_left]
    if len(pool_words) < N_CANDIDATES - 1:
        return None
    picks: list[str] = []
    for _ in range(60):
        cand = rng.choice(pool_words)
        if cand not in picks:
            picks.append(cand)
        if len(picks) == N_CANDIDATES - 1:
            return picks
    return None


def _uniqueness(pool: ClozePool, *, target: str, prev: str | None, nxt: str | None,
                distract: list[str]) -> tuple[bool, int, list[int]]:
    """唯一性校验（fail-closed）：`score(正确) >= MIN_SCORE >= ...` 且领先 MARGIN 倍。"""
    s_true = _score(pool, target, prev, nxt)
    s_neg = [_score(pool, w, prev, nxt) for w in distract]
    ok = s_true >= MIN_SCORE and s_true >= MARGIN * max(s_neg, default=0)
    return ok, s_true, s_neg


def _rotate_to(candidates: list[str], index: int, target: int) -> list[str]:
    """把 `candidates[index]` 转到 `target` 位（其余保持相对顺序）。

    转置后原位置 `i` 落到 `(i - k) mod n`，所以 `k = index - target`；
    写成 `target - index` 会把**干扰项**转到答案位上（单测抓到过这个方向错误）。
    """
    k = (index - target) % len(candidates)
    return candidates[k:] + candidates[:k]


def _case_strata(pool: ClozePool, view: list[str], rng: random.Random,
                 content_budget: int) -> dict[str, list[tuple[str, str, int]]]:
    """把折内句子展开成 (句子, 目标词, 位置) 案例，按空位位置分三层并各自洗牌。

    分层是 dev-notes/13 §4.2「句首/句中/句末都要覆盖」的**实现点**：随机抽样时
    句首天然只占几个百分点，靠运气到不了配额，只能显式分层再按配额取。
    """
    strata: dict[str, list[tuple[str, str, int]]] = {"句首": [], "句中": [], "句末": []}
    for sentence in view:
        for w in sorted(w for w in _ALL_WORDS if w in sentence):
            if pool.freq[w] < TARGET_FREQ_MIN:
                continue
            if sentence.count(w) != 1:          # 同句出现多次 → 空位指代歧义，丢
                continue
            i = sentence.find(w)
            prefix, suffix = sentence[:i], sentence[i + len(w):]
            if len(prefix) + len(suffix) + len(w) * N_CANDIDATES > content_budget:
                continue
            strata[_blank_kind(prefix, suffix)].append((sentence, w, i))
    for cases in strata.values():
        rng.shuffle(cases)
    return strata


def build_cloze_dataset(target_samples: int = 6000, seed: int = DEFAULT_SEED, *,
                        split: str = "train", fixed_answer_pos: int | None = None,
                        bg_ratio: float = BG_RATIO,
                        max_lines: int = MAX_SCAN_LINES) -> list[dict]:
    """构建完形填空数据集。

    Args:
        target_samples: 目标样本数，最终**恰好**返回该数量。
        seed: 随机种子；决定案例顺序、干扰项与候选位置。
        split: `"train"`（9 折）/ `"eval"`（第 9 折），句子逐条不相交。
        fixed_answer_pos: 给定时正确词固定在该位置（P1 固定序对照组）；
            None = 均匀随机。它与随机序**共用同一串 rng 抽样**，只有位置不同。
        bg_ratio: 「无合适候选」样本占比。
        max_lines: 语料扫描行数上限（测试用小值）。
    """
    if split not in ("train", "eval"):
        raise ValueError(f"split 只能是 train/eval，收到 {split!r}")
    if fixed_answer_pos is not None and not (1 <= fixed_answer_pos <= N_CANDIDATES):
        raise ValueError(f"fixed_answer_pos 必须落在 1..{N_CANDIDATES}")
    if not (0.0 <= bg_ratio < 1.0):
        raise ValueError(f"bg_ratio 必须落在 [0,1)，收到 {bg_ratio}")

    validate_markers()
    # 词表整体过一遍内容门禁：手写词表里若混进词表外字符，"同类干扰项"就是 UNK 簇
    lexicon_problems = [w for w in _ALL_WORDS if content_problems(w)]
    if lexicon_problems:
        raise ValueError(
            f"POS_LEXICON 有 {len(lexicon_problems)} 个词过不了内容门禁："
            f"{lexicon_problems[:8]} —— fail-closed 拒绝出带病数据")
    pool = load_cloze_pool(max_lines)
    view = [s for s in pool.sentences if (_fold(s) == EVAL_FOLD) == (split == "eval")]
    if len(view) < 200:
        raise ValueError(
            f"split={split} 只有 {len(view)} 个句子（<200）—— 扫描窗口 "
            f"max_lines={max_lines} 太小，fail-closed 拒绝出退化数据")
    rng = random.Random(seed)

    n_bg = int(target_samples * bg_ratio)
    n_pos = target_samples - n_bg
    content_budget = TEXT_LIMIT - LIST_OVERHEAD - len(BLANK)   # 前后文 + 4 个候选词
    strata = _case_strata(pool, view, rng, content_budget)
    cursor = {"句首": 0, "句中": 0, "句末": 0}
    drops: Counter = Counter()
    solvable: list[dict] = []
    backgrounds: list[dict] = []
    blank_kinds: Counter = Counter()
    punct_adjacent = 0

    def _make_solvable(case, kind) -> dict | None:
        nonlocal punct_adjacent
        sentence, target, pos_i = case
        prefix, suffix = sentence[:pos_i], sentence[pos_i + len(target):]
        prev = prefix[-1] if prefix else None
        nxt = suffix[0] if suffix else None
        s_true = _score(pool, target, prev, nxt)
        distract, s_neg = None, None
        for _ in range(DISTRACTOR_TRIES):
            got = _distractors(pool, target=target, sentence=sentence,
                               budget_left=content_budget - len(prefix) - len(suffix), rng=rng)
            if got is None:
                break
            # 唯一性判据只有一处实现（`_uniqueness`），测试也复算同一处
            ok, _, got_neg = _uniqueness(pool, target=target, prev=prev, nxt=nxt, distract=got)
            if ok:
                distract, s_neg = got, got_neg
                break
        if distract is None:
            drops["干扰项不足(同词类同频同字长)" if s_true < MIN_SCORE else "唯一性不足"] += 1
            return None
        cands = distract + [target]
        rng.shuffle(cands)
        draw_pos = rng.randrange(N_CANDIDATES)
        target_pos = draw_pos if fixed_answer_pos is None else fixed_answer_pos - 1
        cands = _rotate_to(cands, cands.index(target), target_pos)
        sample = _finalize(prefix, suffix, cands, target_pos + 1,
                           {"kind": "solvable", "answer_pos": target_pos + 1,
                            "blank_kind": kind, "word": target,
                            "freq": pool.freq[target], "score_true": s_true,
                            "score_neg": tuple(s_neg), "split": split})
        blank_kinds[kind] += 1
        if _punct_adjacent(prev, nxt):
            punct_adjacent += 1
        return sample

    def _make_bg(case, kind) -> dict | None:
        sentence, target, pos_i = case
        prefix, suffix = sentence[:pos_i], sentence[pos_i + len(target):]
        prev = prefix[-1] if prefix else None
        nxt = suffix[0] if suffix else None
        # 无合适样本**不放正确词**：4 个候选全部共现偏弱（`< MIN_SCORE`），
        # 同词类 / 同频桶 / 同字长的约束照旧成立（否则「挑弱的」本身就是捷径）。
        s_true = _score(pool, target, prev, nxt)
        distract = extra = s_neg = None
        for _ in range(DISTRACTOR_TRIES):
            got = _distractors(pool, target=target, sentence=sentence,
                               budget_left=content_budget - len(prefix) - len(suffix), rng=rng)
            if got is None:
                break
            got_neg = [_score(pool, w, prev, nxt) for w in got]
            if not (s_true >= MIN_SCORE and max(got_neg, default=MIN_SCORE) < MIN_SCORE):
                continue
            got_extra = _weak_alternatives(pool, target=target, distract=got,
                                           sentence=sentence, prev=prev, nxt=nxt, rng=rng)
            if got_extra:
                distract, s_neg, extra = got, got_neg, got_extra
                break
        if distract is None or extra is None:
            drops["无合适样本未达『全弱共现』" if s_true >= MIN_SCORE else "正解共现不足"] += 1
            return None
        cands = distract + extra
        rng.shuffle(cands)
        return _finalize(prefix, suffix, cands, None,
                         {"kind": "bg", "answer_pos": 0, "blank_kind": kind,
                          "word": target, "score_true": s_true,
                          "score_neg": tuple(s_neg), "split": split})

    def _fill(make, quotas: dict[str, int], sink: list[dict], what: str) -> None:
        """按空位分层配额取样：某层取尽仍不够 → 响亮失败，绝不静默缩配额。"""
        for kind in ("句首", "句中", "句末"):
            need = quotas[kind]
            while need > 0:
                if cursor[kind] >= len(strata[kind]):
                    raise ValueError(
                        f"{what} 的空位 {kind} 层取尽（{quotas[kind]} 个配额，"
                        f"已取 {quotas[kind] - need}）仍不够 —— 丢弃 {dict(drops)}。"
                        "加大 max_lines 或调低 POS_QUOTA，不允许静默缩水")
                case = strata[kind][cursor[kind]]
                cursor[kind] += 1
                sample = make(case, kind)
                if sample is None:
                    continue
                sink.append(sample)
                need -= 1

    _fill(_make_solvable,
          {"句首": int(n_pos * POS_QUOTA["句首"]),
           "句中": n_pos - int(n_pos * POS_QUOTA["句首"]) - int(n_pos * POS_QUOTA["句末"]),
           "句末": int(n_pos * POS_QUOTA["句末"])},
          solvable, "有正解样本")
    _fill(_make_bg,
          {"句首": int(n_bg * POS_QUOTA["句首"]),
           "句中": n_bg - int(n_bg * POS_QUOTA["句首"]) - int(n_bg * POS_QUOTA["句末"]),
           "句末": int(n_bg * POS_QUOTA["句末"])},
          backgrounds, "无合适样本")

    n_pos_built = max(1, len(solvable))
    punct_ratio = punct_adjacent / n_pos_built
    if punct_ratio > PUNCT_ADJACENT_MAX:
        raise ValueError(
            f"与标点相邻的空位占比 {punct_ratio * 100:.1f}% > {PUNCT_ADJACENT_MAX * 100:.0f}%"
            f"（dev-notes/13 §4.2：空位不能总是被标点锚定）；丢弃 {dict(drops)}")

    dataset = solvable + backgrounds
    rng.shuffle(dataset)
    texts = [x["text"] for x in dataset]
    if len(set(texts)) != len(texts):
        raise ValueError("数据集里出现重复文本 —— 句子复用或干扰项失控，fail-closed")

    _report(pool, dataset, split=split, view_n=len(view), drops=drops,
            blank_kinds=blank_kinds, punct_adjacent=punct_adjacent, seed=seed)
    return dataset


def _weak_alternatives(pool: ClozePool, *, target: str, distract: list[str],
                       sentence: str, prev: str | None, nxt: str | None,
                       rng: random.Random) -> list[str]:
    """给无合适样本补第 4 个「同词类 / 同频桶 / 同字长 / 共现偏弱」的候选。"""
    pos = word_class(target)
    length = len(target)
    lo, hi = _freq_bucket(pool.freq[target])
    cand = [w for w in POS_LEXICON[pos]
            if len(w) == length and w not in sentence and w != target
            and w not in distract and lo <= pool.freq[w] <= hi
            and _score(pool, w, prev, nxt) < MIN_SCORE]
    if not cand:
        return []
    for _ in range(40):
        pick = rng.choice(cand)
        if pick not in distract:
            return [pick]
    return []


def _blank_kind(prefix: str, suffix: str) -> str:
    if not prefix:
        return "句首"
    if not suffix:
        return "句末"
    return "句中"


def _punct_adjacent(prev: str | None, nxt: str | None) -> bool:
    return bool(prev and prev in "，,、；;") or bool(nxt and nxt in "，,、；;")


def _finalize(prefix: str, suffix: str, candidates: list[str], answer_pos: int | None,
              meta: dict) -> dict:
    """渲染 `前文__后文|1）…`、打锚点、跑 fail-closed 检查。"""
    head = prefix + BLANK + suffix
    lst, offsets = render_choice_list(candidates)
    text = head + lst
    meta["ctx"] = head          # 原文句子（含空位）—— 折不相交、探针不重合都按它判
    check_text_length(text, TEXT_LIMIT)
    spans: list[dict] = []
    if answer_pos is not None:
        s0, e0 = offsets[answer_pos - 1]
        s0 += len(head)
        e0 += len(head)
        if text[s0:e0] != candidates[answer_pos - 1]:
            raise ValueError(f"锚点错位：期望 {candidates[answer_pos - 1]!r}，实际 {text[s0:e0]!r}")
        spans.append({"label": answer_pos, "start": s0, "end": e0,
                      "word": candidates[answer_pos - 1]})
    return {"text": text, "spans": spans, "meta": dict(meta, cands=tuple(candidates))}


def _report(pool: ClozePool, dataset: list[dict], *, split: str, view_n: int,
            drops: Counter, blank_kinds: Counter, punct_adjacent: int, seed: int) -> None:
    solvable = [x for x in dataset if x["meta"]["kind"] == "solvable"]
    bg = [x for x in dataset if x["meta"]["kind"] == "bg"]
    pos_counter = Counter(x["meta"]["answer_pos"] for x in solvable)
    n_pos = max(1, len(solvable))
    n_sent = len(pool.sentences)
    print(f"  完形填空语料池：扫描 {pool.lines} 行 / {pool.files} 文件 → 可用句子 {n_sent}"
          f"（其中含词表词且通过语体/符号门禁）")
    print(f"  本数据集：{len(dataset)} = 有正解 {len(solvable)} + 无合适 {len(bg)}"
          f" | 占本折可用句子 {len(solvable) / max(1, view_n) * 100:.1f}%（折内 {view_n} 句）"
          f" | split={split} seed={seed}")
    print(f"  正解位置分布：{dict(sorted(pos_counter.items()))} | 空位位置：{dict(blank_kinds)}"
          f"（句首 {blank_kinds.get('句首', 0) / n_pos * 100:.1f}% / 句末 {blank_kinds.get('句末', 0) / n_pos * 100:.1f}%）"
          f" | 与标点相邻 {punct_adjacent / n_pos * 100:.1f}%")
    if drops:
        print(f"  丢弃归因：{dict(drops)}")


def position_report(dataset: list[dict]) -> dict[int, int]:
    return dict(sorted(Counter(x["meta"]["answer_pos"]
                               for x in dataset if x["meta"]["kind"] == "solvable").items()))


def blank_report(dataset: list[dict]) -> dict[str, int]:
    """空位位置构成 —— dev-notes/13 §4.2 要求句首/句中/句末都要覆盖。"""
    return dict(sorted(Counter(x["meta"]["blank_kind"]
                               for x in dataset if x["meta"]["kind"] == "solvable").items()))


if __name__ == "__main__":
    ds = build_cloze_dataset(target_samples=2000, max_lines=120_000)
    print(f"位置 {position_report(ds)} | 空位 {blank_report(ds)}")
    print("样例：")
    for x in ds[:3]:
        print("   ", x["text"], "| spans", [(s["start"], s["end"]) for s in x["spans"]])
