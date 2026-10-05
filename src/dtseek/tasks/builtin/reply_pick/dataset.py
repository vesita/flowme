"""择优回复（selection）数据集：问句 + 4 条候选回复，选出最合适的一条。

任务定义（dev-notes/13 §4.1）：
  - **正例**：多轮对话里的 `(问句, 真实下一条回复)` —— 免费监督；
  - **负例按难度分两档**（先用 1+2，档 3「近义改写」需构造/人工，本轮不做）：
      档 1「跑题」：同一折里随机抽的其它回复，长度与正例相近（±8 字）、
                    与问句二元组重合 ≤ 0.20 —— 只判「哪个跑题」就能拿分；
      档 2「同话题不同轮次」：**先取同一对话块里其它轮的回复**（同一场对话，
                    语义相近但不是这一步该说的）；块里不够时退回
                    「问句与本问共享关键词（二元组重合 ≥ 0.03）的其它轮对的回复」——
                    仍是同话题的回复，只是答给另一个问题的；
  - **真·无解样本**：同一个问句配 4 条都不贴切的候选 → `spans=[]` → 类别 0
    （无合适候选）。它与有正解样本的**唯一区别**就是正例在不在候选里 ——
    拒答出口（P2）因此可判。

三条 fail-closed 不变量（破坏任意一条都响亮抛错，绝不静默出数据）：
  1. **符号纪律**：构建前先跑 `frame.validate_markers()`（词表 id / 无 UNK / 无碰撞）；
  2. **内容纯净 + 长度门禁**：问句与候选不许含标记字符，整段 ≤ TEXT_LIMIT
     （= `spec.max_len - 8`，引擎 window 分段阈值），超限即抛错而不是截断；
  3. **样本完整性**：每样本至多 1 个切片、锚点区间逐字等于被选中候选、
     文本不重复、四个候选位置都要有覆盖。

两条捷径控制（P1 的前提）：
  - **长度捷径**：候选长度一律向正例长度收敛（档 1 ±8 字、档 2 ±10 字），
    无合适样本的候选长度按语料回复长度分布抽样；
  - **位置捷径**：候选顺序在训练里均匀打乱；`fixed_answer_pos` 给定时才固定，
    而且它与随机序**共用同一串 rng 抽样** —— 两组只有位置不同，是配对对照。
"""
from __future__ import annotations

import hashlib
import random
from collections import Counter
from contextlib import ExitStack
from dataclasses import dataclass, field

from dtseek.tasks.builtin.reply_pick.frame import (
    LIST_OVERHEAD,
    QUESTION_HEAD,
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
TEXT_LIMIT = 120

#: 「无合适候选」样本占比（有正解 : 无合适 = 0.85 : 0.15）。
BG_RATIO = 0.15

#: 问句 / 回复的长度粗筛（真正的预算逐样本算）。
Q_LEN_RANGE = (3, 40)
R_LEN_RANGE = (2, 60)

#: 候选长度向参照长度收敛的容差（字）。档 2 稍宽：同话题回复天然长短不一。
LEN_TOL = {1: 8, 2: 10}
#: 档 1 负例与问句的二元组 Jaccard 上限（随机异话题回复实测 99.97% 在 0.20 以下）。
MAX_QUESTION_OVERLAP = 0.20
#: 档 2 退化路径的问句二元组重合下限（实测随机问句对 p95≈0.027，0.03 ≈ 共享关键词）。
MIN_TIER2_OVERLAP = 0.03

#: 负例难度档（dev-notes/13 §4.1）。
TIERS = (1, 2)

#: 语料扫描行数上限（18 文件轮转，取各文件头部）。测试用 max_lines 覆盖。
MAX_SCAN_LINES = 1_200_000

#: 训练/评测折：按 (问句, 回复) 内容哈希分 10 折，第 9 折只做评测 ——
#: 换帧探针与整句级指标不能吃训练折的同一条对话（P3/P4）。
EVAL_FOLD = 9
N_FOLDS = 10


@dataclass
class TurnPair:
    """一条通过门禁的 `(问句, 真实回复)` 轮对。`block` 是所在对话块的全局编号。"""

    q: str
    r: str
    block: int
    qbg: frozenset = frozenset()      # 问句二元组，档 2 抽样的热路径缓存


@dataclass
class CorpusPool:
    """一趟扫描的产物，按 split 派生出「负例只能来自本折」的视图。

    缓存在模块级：评测脚本要用同一份语料构造多套样本，每套都重扫一遍既慢
    又会因为扫描窗口不一致而悄悄换数据。
    """

    pairs: list[TurnPair] = field(default_factory=list)
    raw_pairs: int = 0
    files: int = 0
    lines: int = 0
    views: dict[str, dict] = field(default_factory=dict)


_POOL_CACHE: dict[int, CorpusPool] = {}


def _gate(*texts: str) -> bool:
    """轮对内容门禁：语体 + 标记字符纯净（长度粗筛由调用方做）。"""
    return all(not style_problems(t) and not content_problems(t) for t in texts)


def _flush(state: dict, pool: CorpusPool, seen: set) -> None:
    """收束当前轮：成对 → 过门禁 → 登记进轮对表与负例池。"""
    if state["q"] is None or not state["buf"]:
        return
    q, r = state["q"], " ".join(x.strip() for x in state["buf"] if x.strip())
    pool.raw_pairs += 1
    if not _gate(q, r):
        return
    if not (Q_LEN_RANGE[0] <= len(q) <= Q_LEN_RANGE[1]
            and R_LEN_RANGE[0] <= len(r) <= R_LEN_RANGE[1]):
        return
    if (q, r) in seen:
        return
    seen.add((q, r))
    pool.pairs.append(TurnPair(q, r, state["block"], frozenset(_bigrams(q))))


def load_pool(max_lines: int = MAX_SCAN_LINES, *, refresh: bool = False) -> CorpusPool:
    """扫描语料构造 `CorpusPool`（按 `max_lines` 缓存）。`resolve_corpus_files` 唯一入口。

    块结构：`用户：` 起新轮、`模型：` 起回复缓冲、空行或 `---` 收束整块 ——
    lccc / sharegpt / qwen3 三种排版都由这三条规则覆盖。块内其它轮的回复就是
    档 2 负例的首选来源（同话题不同轮次）。
    """
    if not refresh and max_lines in _POOL_CACHE:
        return _POOL_CACHE[max_lines]

    files = resolve_corpus_files()
    pool = CorpusPool(files=len(files))
    seen: set[tuple[str, str]] = set()
    state: dict[int, dict] = {}
    n_lines = 0

    def _start_block(idx: int, k: int) -> int:
        return idx * 1_000_000 + k

    with ExitStack() as stack:
        fhs = [stack.enter_context(open(f, encoding="utf-8", errors="ignore")) for f in files]
        active = list(range(len(fhs)))
        while active and n_lines < max_lines:
            nxt = []
            for i in active:
                line = fhs[i].readline()
                if not line:
                    continue
                n_lines += 1
                nxt.append(i)
                line = line.rstrip("\n")
                st = state.setdefault(i, {"q": None, "buf": [], "k": 0,
                                          "block": _start_block(i, 0), "raw": None})
                if line.startswith(("用户：", "用户:")):
                    _flush(st, pool, seen)
                    st["q"] = line.split("：", 1)[-1].split(":", 1)[-1].strip()
                    st["buf"] = []
                elif line.startswith(("模型：", "模型:")):
                    if st["q"] is not None:
                        st["buf"] = [line.split("：", 1)[-1].split(":", 1)[-1].strip()]
                elif not line.strip() or line.startswith("---"):
                    _flush(st, pool, seen)
                    st["q"], st["buf"] = None, []
                    st["k"] += 1
                    st["block"] = _start_block(i, st["k"])
                elif st["q"] is not None and st["buf"]:
                    st["buf"].append(line)
            active = nxt

    pool.lines = n_lines
    _POOL_CACHE[max_lines] = pool
    return pool


def _fold(pair: TurnPair) -> int:
    h = hashlib.md5(f"{pair.q}\x00{pair.r}".encode()).hexdigest()
    return int(h, 16) % N_FOLDS


def split_view(pool: CorpusPool, split: str) -> dict:
    """按折派生的负例视图：**负例只能来自本折**，训练折与评测折文本不串。

    返回 `{pairs, replies_by_len, block_replies, lens}`；负例池只登记本折的回复，
    否则评测样本会拿训练折当正例的回复去当负例（文本级串折）。
    """
    if split in pool.views:
        return pool.views[split]
    pairs = [p for p in pool.pairs if (_fold(p) == EVAL_FOLD) == (split == "eval")]
    if not pairs:
        raise ValueError(f"split={split} 的轮对为 0 —— 扫描窗口太小，fail-closed")
    replies_by_len: dict[int, list[str]] = {}
    pairs_by_len: dict[int, list[TurnPair]] = {}
    block_replies: dict[int, list[str]] = {}
    for p in pairs:
        replies_by_len.setdefault(len(p.r), []).append(p.r)
        pairs_by_len.setdefault(len(p.r), []).append(p)
        block_replies.setdefault(p.block, []).append(p.r)
    view = {
        "pairs": pairs,
        "replies_by_len": replies_by_len,
        "pairs_by_len": pairs_by_len,
        "block_replies": block_replies,
        "lens": sorted(replies_by_len),
    }
    pool.views[split] = view
    return view


def _bigrams(s: str) -> set[str]:
    return {s[i:i + 2] for i in range(len(s) - 1)}


def _overlap(a: str, b: str) -> float:
    """二元组 Jaccard：判「跑题」与「同话题」用，纯字面、无模型参与。"""
    sa, sb = _bigrams(a), _bigrams(b)
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _tier1_draw(rng: random.Random, view: dict, *, q: str, ref_len: int,
                used: set[str], budget: int, tries: int = 40) -> str | None:
    """档 1：同长度、与问句低重合的随机回复（跑题负例）。"""
    lo = max(R_LEN_RANGE[0], ref_len - LEN_TOL[1])
    hi = ref_len + LEN_TOL[1]
    lengths = [L for L in range(lo, hi + 1) if view["replies_by_len"].get(L)]
    if not lengths:
        return None
    for _ in range(tries):
        cand = rng.choice(view["replies_by_len"][rng.choice(lengths)])
        if cand in used or len(cand) > budget:
            continue
        if _overlap(q, cand) > MAX_QUESTION_OVERLAP:
            continue
        used.add(cand)
        return cand
    return None


def _tier2_draw(rng: random.Random, view: dict, *, q: str, block: int, ref_len: int,
                used: set[str], budget: int, tries: int = 150) -> tuple[str, str] | None:
    """档 2：同话题不同轮次。返回 `(回复, 来源)`，来源 ∈ {同块, 同话题}。

    先取同一对话块里其它轮的回复（最强形态）；块内不够再取「问句共享关键词的
    其它轮对」的回复 —— 后者是答给另一个问题的，贴着话题但不贴这一步。
    """
    lo = max(R_LEN_RANGE[0], ref_len - LEN_TOL[2])
    hi = ref_len + LEN_TOL[2]
    same_block = [r for r in view["block_replies"].get(block, [])
                  if r not in used and lo <= len(r) <= hi and len(r) <= budget]
    if same_block:
        pick = rng.choice(same_block)
        used.add(pick)
        return pick, "同块"
    lengths = [L for L in range(lo, hi + 1)
               if L <= budget and view["pairs_by_len"].get(L)]
    if not lengths:
        return None
    qbg = _bigrams(q)
    for _ in range(tries):
        p = rng.choice(view["pairs_by_len"][rng.choice(lengths)])
        if p.r in used:
            continue
        inter = len(qbg & p.qbg)
        if inter == 0:
            continue
        if inter / len(qbg | p.qbg) < MIN_TIER2_OVERLAP:
            continue
        used.add(p.r)
        return p.r, "同话题"
    return None


def _rotate_to(candidates: list[str], index: int, target: int) -> list[str]:
    """把 `candidates[index]` 旋转到 `target` 位（其余候选保持相对顺序）。"""
    k = (index - target) % len(candidates)
    return candidates[k:] + candidates[:k]


def _finalize(head: str, candidates: list[str], answer_pos: int | None,
              meta: dict) -> dict:
    """渲染整段 + 打锚点 + 跑 fail-closed 检查，返回样本。"""
    lst, offsets = render_choice_list(candidates)
    text = head + lst
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
    meta = dict(meta, cands=tuple(candidates))
    return {"text": text, "spans": spans, "meta": meta}


def _draw_negatives(rng: random.Random, view: dict, *, q: str, block: int, ref_len: int,
                    n: int, tier: int, used: set[str], budget: int) -> tuple[list, list, int] | None:
    """抽 `n` 条同档负例；返回 `(候选, 来源, 实际难度档)`，抽不齐返回 None。"""
    cands: list[str] = []
    srcs: list[str] = []
    for _ in range(n):
        left = budget - sum(len(c) for c in cands)
        if left <= 0:
            return None
        if tier == 1:
            got = _tier1_draw(rng, view, q=q, ref_len=ref_len, used=used, budget=left)
            if got is None:
                return None
            cands.append(got)
            srcs.append("随机")
        else:
            got2 = _tier2_draw(rng, view, q=q, block=block, ref_len=ref_len,
                               used=used, budget=left)
            if got2 is None:
                return None
            cands.append(got2[0])
            srcs.append(got2[1])
    return cands, srcs, tier


def build_reply_pick_dataset(target_samples: int = 6000, seed: int = DEFAULT_SEED, *,
                             split: str = "train", tiers: tuple[int, ...] = TIERS,
                             fixed_answer_pos: int | None = None,
                             bg_ratio: float = BG_RATIO,
                             max_lines: int = MAX_SCAN_LINES) -> list[dict]:
    """构建择优回复数据集。

    Args:
        target_samples: 目标样本数，最终**恰好**返回该数量。
        seed: 随机种子；决定轮对抽样、负例抽样与候选位置。
        split: `"train"`（9 折）/ `"eval"`（第 9 折），轮对逐条不相交。
        tiers: 允许的负例难度档。`(2,)` = 只留高难负例（同话题不同轮次）。
        fixed_answer_pos: 给定时正解固定在该位置（P1 的「固定序」对照组）；
            None = 均匀随机（训练与「打乱序」评测都用它）。
        bg_ratio: 「无合适候选」样本占比。
        max_lines: 语料扫描行数上限（测试用小值）。
    """
    if split not in ("train", "eval"):
        raise ValueError(f"split 只能是 train/eval，收到 {split!r}")
    if not tiers or any(t not in TIERS for t in tiers):
        raise ValueError(f"tiers 必须取自 {TIERS}，收到 {tiers!r}")
    if fixed_answer_pos is not None and not (1 <= fixed_answer_pos <= N_CANDIDATES):
        raise ValueError(f"fixed_answer_pos 必须落在 1..{N_CANDIDATES}")
    if not (0.0 <= bg_ratio < 1.0):
        raise ValueError(f"bg_ratio 必须落在 [0,1)，收到 {bg_ratio}")

    validate_markers()
    pool = load_pool(max_lines)
    view = split_view(pool, split)
    rng = random.Random(seed)

    n_bg = int(target_samples * bg_ratio)
    n_pos = target_samples - n_bg
    pairs = view["pairs"]
    if len(pairs) < 200:
        raise ValueError(
            f"split={split} 只有 {len(pairs)} 条轮对（<200）—— 扫描窗口 "
            f"max_lines={max_lines} 太小，fail-closed 拒绝出退化数据")
    order = list(range(len(pairs)))
    rng.shuffle(order)

    drops: Counter = Counter()
    neg_src: Counter = Counter()
    solvable: list[dict] = []
    backgrounds: list[dict] = []
    used_q: set[str] = set()
    # 内容预算：整段 ≤ TEXT_LIMIT，扣掉 `问：` 与 4 个标号段
    content_budget = TEXT_LIMIT - LIST_OVERHEAD - len(QUESTION_HEAD)

    # 一条轮对只试一次：抽不齐负例就丢，不允许换个随机数重抽到凑够为止
    # （那等于把「数据不够」藏进重复采样里）。
    for pos_i in order:
        if len(solvable) >= n_pos and len(backgrounds) >= n_bg:
            break
        pair = pairs[pos_i]
        if pair.q in used_q:
            drops["问句重复"] += 1
            continue

        # —— 与 fixed_answer_pos 无关的 rng 抽样必须逐次对齐：
        # 打乱序组与固定序组因此抽到同一批轮对、同一批负例，只差一个位置（P1 配对设计）。
        tier = rng.choice(tiers)
        draw_pos = rng.randrange(N_CANDIDATES)
        want_bg = len(backgrounds) < n_bg and (len(solvable) >= n_pos or rng.random() < bg_ratio)

        if want_bg:
            # 无正解：参照长度按语料回复长度分布抽，候选长度分布与有正解样本对齐
            ref_len = rng.choice(view["lens"])
            got = _draw_negatives(rng, view, q=pair.q, block=pair.block, ref_len=ref_len,
                                  n=N_CANDIDATES, tier=tier, used=set(),
                                  budget=content_budget - len(pair.q))
            if got is None:
                if len(tiers) > 1:
                    alt = next(t for t in TIERS if t != tier)
                    got = _draw_negatives(rng, view, q=pair.q, block=pair.block,
                                          ref_len=ref_len, n=N_CANDIDATES, tier=alt,
                                          used=set(), budget=content_budget - len(pair.q))
                    if got is not None:
                        drops[f"档{tier}不足退档"] += N_CANDIDATES
                if got is None:
                    drops[f"无合适负例不足(档{tier})"] += 1
                    continue
            cands, srcs, used_tier = got
            rng.shuffle(cands)
            sample = _finalize(
                QUESTION_HEAD + pair.q, cands, None,
                {"kind": "bg", "neg_tiers": (used_tier,) * N_CANDIDATES, "neg_src": tuple(srcs),
                 "hard": used_tier == 2, "answer_pos": 0, "q": pair.q, "tiers": tiers})
        else:
            ref_len = len(pair.r)
            got = _draw_negatives(rng, view, q=pair.q, block=pair.block, ref_len=ref_len,
                                  n=N_CANDIDATES - 1, tier=tier, used={pair.r},
                                  budget=content_budget - len(pair.q) - len(pair.r))
            if got is None:
                if len(tiers) > 1:
                    alt = next(t for t in TIERS if t != tier)
                    got = _draw_negatives(rng, view, q=pair.q, block=pair.block,
                                          ref_len=ref_len, n=N_CANDIDATES - 1, tier=alt,
                                          used={pair.r},
                                          budget=content_budget - len(pair.q) - len(pair.r))
                    if got is not None:
                        drops[f"档{tier}不足退档"] += N_CANDIDATES - 1
                if got is None:
                    drops[f"正例负例不足(档{tier})"] += 1
                    continue
            negs, srcs, used_tier = got
            neg_src.update(srcs)
            cands = [pair.r] + negs
            rng.shuffle(cands)
            target = draw_pos if fixed_answer_pos is None else fixed_answer_pos - 1
            cands = _rotate_to(cands, cands.index(pair.r), target)
            sample = _finalize(
                QUESTION_HEAD + pair.q, cands, target + 1,
                {"kind": "solvable", "neg_tiers": (used_tier,) * len(negs),
                 "neg_src": tuple(srcs), "hard": used_tier == 2,
                 "answer_pos": target + 1, "q": pair.q, "true": pair.r, "tiers": tiers})
        used_q.add(pair.q)
        (backgrounds if want_bg else solvable).append(sample)

    if len(solvable) < n_pos or len(backgrounds) < n_bg:
        raise ValueError(
            f"数据集不完整：有正解 {len(solvable)}/{n_pos}，无合适 {len(backgrounds)}/{n_bg}。"
            f" 丢弃原因 {dict(drops)}；split={split} tiers={tiers} —— "
            "加大 max_lines 或放宽参数，不允许静默缩水")

    dataset = solvable + backgrounds
    rng.shuffle(dataset)
    texts = [x["text"] for x in dataset]
    if len(set(texts)) != len(texts):
        raise ValueError("数据集里出现重复文本 —— 问句复用或负例复用失控，fail-closed")

    _report(pool, view, dataset, split=split, drops=drops, neg_src=neg_src, seed=seed)
    return dataset


def _report(pool: CorpusPool, view: dict, dataset: list[dict], *, split: str,
            drops: Counter, neg_src: Counter, seed: int) -> None:
    """构建报告：真实语料占比、负例难度与来源、位置分布、丢弃归因。"""
    solvable = [x for x in dataset if x["meta"]["kind"] == "solvable"]
    bg = [x for x in dataset if x["meta"]["kind"] == "bg"]
    pos_counter = Counter(x["meta"]["answer_pos"] for x in solvable)
    hard = sum(1 for x in solvable if x["meta"]["hard"])
    lens = sorted(len(x["meta"]["true"]) for x in solvable)
    fold_n = len(view["pairs"])
    print(f"  择优回复轮对池：扫描 {pool.lines} 行 / {pool.files} 文件 → 原始 {pool.raw_pairs} 对"
          f" → 通过门禁 {len(pool.pairs)} 对（{len(pool.pairs) / max(1, pool.raw_pairs) * 100:.1f}%）"
          f" | split={split} 折内 {fold_n} 对")
    print(f"  本数据集：{len(dataset)} = 有正解 {len(solvable)} + 无合适 {len(bg)}"
          f" | 采样占本折可用轮对 {len(solvable) / max(1, fold_n) * 100:.1f}%"
          f" | 真实语料门禁通过率 {len(pool.pairs) / max(1, pool.raw_pairs) * 100:.1f}%"
          f" | seed={seed}")
    print(f"  负例来源（有正解样本的负例条数）：{dict(sorted(neg_src.items()))}"
          f" | 纯高难样本 {hard}/{len(solvable)}（{hard / max(1, len(solvable)) * 100:.1f}%）")
    print(f"  正解位置分布：{dict(sorted(pos_counter.items()))}"
          f" | 正例字长 p10/p50/p90 = {lens[len(lens) // 10] if lens else 0}"
          f"/{lens[len(lens) // 2] if lens else 0}/{lens[len(lens) * 9 // 10] if lens else 0}")
    if drops:
        print(f"  丢弃归因：{dict(drops)}")


def position_report(dataset: list[dict]) -> dict[int, int]:
    """按正解位置统计样本量 —— P1 位置偏置的前提（每个位置都要有足量样本）。"""
    return dict(sorted(Counter(x["meta"]["answer_pos"]
                               for x in dataset if x["meta"]["kind"] == "solvable").items()))


def difficulty_report(dataset: list[dict]) -> dict[str, int]:
    """负例难度构成：纯跑题 / 纯高难 / 混合 各多少条。"""
    out: Counter = Counter()
    for x in dataset:
        tiers = x["meta"]["neg_tiers"]
        if all(t == 1 for t in tiers):
            out["档1跑题"] += 1
        elif all(t == 2 for t in tiers):
            out["档2同话题"] += 1
        else:
            out["混合"] += 1
    return dict(out)


if __name__ == "__main__":
    ds = build_reply_pick_dataset(target_samples=2000, max_lines=120_000)
    print(f"位置分布 {position_report(ds)} | 难度构成 {difficulty_report(ds)}")
