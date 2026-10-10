#!/usr/bin/env python3
"""S76 · L3-r3a′ 「选择进入输出」：回复必须含**被选中候选的字母** —— 数据生成器 + 地板
（纯 CPU、确定性 seed、秒~分钟级；不训模型、不碰 GPU、不用 systemd-run）。

设计见 L3_PLAN.md §9（r2 对话数据：多轮 ctx + 五条不变量）、§11.1（r3a 候选层规格）、
§11.5（**机会级 / 捷径级地板必须分开命名**）、§11.6（r3a 结果）、§11.7 选项①（加难度的首选）、
§11.8（本单元规格与 R3a′-a/b/c 判据）。骨架照 stages/74_nl_dialog_cand_gen.py：
importlib 复用 `72` 的族定义 / DET 回复 / 槽位匹配 / 指纹；不变量逐条断言 + 计数；
地板分机会级与捷径级；`[D76]`/`[FLOOR-L]`/`[FLOOR-F]`/`[GEN]` 打印风格；确定性（连跑两遍逐字节相同）。

★★ 相对 r3a（`74`/`78`）的**唯一改动**：回复必须含「被选中候选的字母」。
   候选层（候选行文本、干扰项构造、正确项位置）**逐字节沿用 r3a**：
   同一 `--seed`、同一 `SEED_CAND_XOR=0x74C0`（与 74 同值）、同一 `build_candidates` 调用序
   ⇒ `[ALIGN-74]` 与 logs/{74,78}_dialog_{train,test}.jsonl 的 `cand_line` / `gold_idx` / `n_cands`
     应**逐条一致**（这就是「唯一改动」的机器强制点；不一致 → 计违例、退出码非 0）。

ctx 与 §11.1 完全同形（7 行，第 6 行是候选行）：
    用户：{轮1用户}
    <resp>{轮1系统}<eos>
    用户：{轮2用户}
    <resp>{轮2系统}<eos>
    用户：{轮3用户}                 ← 既不含实体也不含条件（不变量 D，来自 §9）
    候选：A. {实体} / {条件}；B. …；…   ← 新增一层（沿用 r3a，与 74/78 逐字节一致）
    <resp>                          ← 待生成：**DET 句面 + 标签**

--------------------------------------------------------------------------
标签格式（写死，机器可解析；LABEL_TMPL / LABEL_RE，`[LABEL-FMT]` 行打印样例）
    标签          LABEL_TMPL = "（{letter}）"（**全角**括号；字母按**位置**取 CAND_LETTERS）
    插入位置      **末尾句号之前**：reply_with_label = DET句面[:-1] + 标签 + "。"
                  理由：§11.8 的样例形如「没问题，AS975482 会按返厂检测安排（C）。」
    DET 句面      与 §9/§11.1 同形：DET(族, 轮3 句式 index) + (实体, 条件)，模板一律以「。」结尾
    反向解析      split_label(reply)：标签必须**恰出现 1 次**、且紧贴末尾「。」之前；
                  返回 (DET句面, 字母)，否则 None（这是 T1/T2 的判定口径）
样例（k=8，族=售后 / 轮3句式 2 / 实体 AS975482 / 条件 返厂检测 / 正确项位置 index=2 ⇒ 字母 C）：
    DET 句面          没问题，AS975482 会按返厂检测安排。
    标签              （C）
    reply_with_label  没问题，AS975482 会按返厂检测安排（C）。
样例（k=4，同一族/句式，正确项位置 index=0 ⇒ 字母 A）：
    reply_with_label  没问题，AS975482 会按返厂检测安排（A）。
★ **位置随机化下字母随位置变化**：同一 (族, 实体, 条件) 的正确项在不同记录里落在不同位置 ⇒ 字母不同；
  历史行（轮1/轮2/轮3）与候选行里**都不含「（X）」形态的标签**（候选行只有裸字母 "A. "），
  所以 T1（字母准确率）只能靠「先选中、再读它的位置」得到 ⇒ 机会地板 = 1/k。

--------------------------------------------------------------------------
两个可判目标（两条指标**各算一套地板，绝不混在一起**，§11.5 的裁定）
    T1（**选择卡**目标）  gold_letter ⇒ 指标 = **字母准确率**，机会地板 = 1/k
                          （k=8 → 12.50%，k=4 → 25.00%）
    T2（**重整卡**目标）  reply_with_label 逐字 EM（整条回复，含标签与句号）
    新增字段：gold_letter / label_marker / reply_with_label（+ gold_det 便于人工核）
    沿用 r3a 字段：cands / gold_idx / n_cands（以及 §9 的全部字段）

不变量（逐条断言 + 打印计数，违例必须 0，否则退出码非 0）
    ① 恰 1 个候选与历史两个槽位同时匹配（且 gold_idx 指向它、候选互异、不越族、
       该候选的「DET 句面 + 正确标签」== target）；
    ② 正确项位置分布均匀（报每位置计数 + 最大偏差 + 卡方，临界限取 0.999 分位）；
    ③ 继承 §9：A 函数性（ctx→target 是函数）/ B 信息充分 / C 非退化（唯一 ctx 比 >= 0.9）/
       D 必读历史（轮3 不含实体与条件、轮2 不含实体、轮1 不含条件）/
       E 用字逐字往返、**无 <unk>**（ctx 与 target 两侧都查）；
    ④ 位置置换不变性：候选顺序打乱（逆序 + 随机置换各一遍）后**被选中的候选（实体,条件）
       与族/句式不变**；而**字母与 target 必须随位置改**（这是本变体的要点，不是缺陷）⇒
       判的是「选中项不变 + 新位置对应新字母正确」；
    ⑤ **字母唯一可判**：gold_letter == letter_of(gold_idx)；ctx 里同时匹配两个槽位的候选
       **唯一**，其字母 == target 里解析出的标签字母；target 去掉标签 == DET(族,句式,实体,条件)；
       历史行/候选行不含「（X）」标签（标签不能从输入整段抄）；
    ⑥ **参考上限 = 100.00%**：完美选择（只用 ctx 文本按历史匹配候选）+ DET 句面 + 正确标签
       ⇒ 逐条复原 target（不是 100% 即违例）。

地板（L5，两条指标各一套；名字里带「机会级 / 捷径级」）
    字母指标（T1）块 `[FLOOR-L]`：报每条规则的**字母准确率**原始行。机会级（§11.5 三类 + 逐位置扫描）
        ①固定字母 always-A  ②随机字母 1/k  ②b随机候选+DET+其字母（=74 的①随机猜1/k）
        ②c总选第一个候选  ②d总选最后一个候选  + always_p 逐位置扫描（≤「字母位置桶」的核对）；
        捷径级（复制/众数/桶/只出标签）：③复制轮3/轮1用户/轮1系统/轮2用户/轮2系统 ④train 全集众数
        ⑤a..⑤f 桶众数 ⑥只出标签不出内容 ⑥b模板填空只用轮3 ⇒ `max_naive`（**字母指标下这些天然≈1/k**，
        因为字母只有 1 位；给出 `max_all` 与 1/k+2SE 的比较，超了才叫泄露）；
    完整回复指标（T2）块 `[FLOOR-F]`：同样规则的**逐字 EM** 原始行 + 机会级 `max_chance`（≈1/k）
        + 捷径级 `max_naive`（**目标 <= 5%**）+ 两条指标的参考上限 EM。
    两块分开打印、分开汇总，绝不把 1/k 的机会水平混进 `max_naive`（§11.5 的裁定）。

产物（只写 logs/）：
    k=8（主档）→ logs/76_dialog_{train,test}.jsonl
    k=4（对照）→ logs/77_dialog_{train,test}.jsonl
    train 7000 / test 800（--smoke 写 76s_/77s_ 的 200/100）；
    每条 = §9 全部字段 + r3a 的 cands/gold_idx/n_cands/gold_letter + 本单元的
    label_marker / reply_with_label / gold_det / label_pos / gold_letter_pos_count。

确定性：同一 (--k --seed --n-train --n-test) 连跑两次，jsonl 逐字节相同（[OUT] 行报 sha256）。

用法（本机只有 .venv/bin/python 装了 tokenizers；本脚本无 torch、秒级）：
    .venv/bin/python stages/76_letter_cand_gen.py                 # k=8 → 76_*
    .venv/bin/python stages/76_letter_cand_gen.py --k 4           # k=4 → 77_*
    .venv/bin/python stages/76_letter_cand_gen.py --smoke         # 小量冒烟（76s_/77s_）
    .venv/bin/python stages/76_letter_cand_gen.py --k 4 --smoke
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sys
from collections import Counter, defaultdict

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
if NS_ROOT not in sys.path:
    sys.path.insert(0, NS_ROOT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
D72_PATH = os.path.join(ROOT, "stages", "72_nl_dialog_gen.py")
OUT_TMPL = os.path.join(LOG_DIR, "{prefix}_dialog_{split}.jsonl")

N_TRAIN = 7000
N_TEST = 800
N_SMOKE_TRAIN = 200
N_SMOKE_TEST = 100
SEED = 20261010
EOS = "<eos>"
USER_PREFIX = "用户："
RESP_PREFIX = "<resp>"

# ---- 候选层（与 74 逐字相同，保证与 r3a 产物逐字节对齐）----
CAND_PREFIX = "候选："
CAND_SEP = "；"
CAND_LETTERS = "ABCDEFGH"
CAND_ITEM = "{letter}. {ent} / {cond}"
CAND_ITEM_RE = re.compile(r"^([A-H])\. (.+?) / (.+)$")

# ---- 本单元新增：标签 ----
LABEL_TMPL = "（{letter}）"
LABEL_RE = re.compile(r"（([A-H])）")

BUCKET_CHARS = 20          # 回复 / 候选行长度桶宽（字符）
T3_BUCKET_CHARS = 5        # 轮3 长度桶宽（字符）
ENT_POS_BUCKET = 10        # 实体位置桶宽（字符）
MAX_SEQ = 512              # ctx + target 字符 token 上限（与 66/70/72/74 同口径）
MIN_UNIQ_CTX_RATIO = 0.9   # 不变量 C：唯一 ctx / 样本数 下限
N_LINE_CTX = 7             # ctx 行数（§9 的 6 行 + 候选行）
PREFIX_BY_K = {8: "76", 4: "77"}          # 产物前缀（k=8 主档 / k=4 对照档），fail-closed
ALIGN_PREFIX_BY_K = {8: "78", 4: "74"}    # r3a 侧产物前缀（只用于对齐核验）
CHI2_999 = {3: 16.266, 7: 24.322}         # 卡方 0.999 分位（dof 3 / 7）
SEED_CAND_XOR = 0x74C0     # ★ 与 74 同值：候选位置/干扰项 RNG 流（保证候选层逐字节一致）
SEED_PERM_XOR = 0x76FE     # ④ 置换检查 RNG 流（本单元自有）
SEED_FLOOR_XOR = 0x76F1    # 地板「随机」RNG 流（本单元自有）


def _load_path(name: str, path: str):
    """按路径加载同名脚本（文件名以数字开头，不能普通 import；照抄 stages/73/74 的写法）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d72 = _load_path("d72_nl_dialog_gen", D72_PATH)

# 与 §9 生成器同源：族定义 / DET 回复 / 槽位匹配 / 指纹 / sha256 全部复用 d72，避免两处定义漂移。
FAMILIES = d72.FAMILIES
FAM_BY_KEY = d72.FAM_BY_KEY
ALL_ENTS = d72.ALL_ENTS
ALL_CONDS = d72.ALL_CONDS
N_FAMILIES = len(FAMILIES)
N_REPLY_TEMPLATES = d72.N_REPLY_TEMPLATES

assert USER_PREFIX == d72.USER_PREFIX and RESP_PREFIX == d72.RESP_PREFIX and EOS == d72.EOS, "与 72 的渲染约定不一致"
assert hasattr(d72, "_match_slot"), "72 的槽位匹配函数不见了 ⇒ 解析口径无法对齐，停"
assert set(PREFIX_BY_K) == set(ALIGN_PREFIX_BY_K), "两档 k 写死为 8 / 4"
# 标签格式的前置条件：所有 DET 句面都以「。」结尾、且不含全角括号（否则插入/解析有歧义）
_ALL_TPL = [t for f in FAMILIES for t in f["reply"]]
assert all(t.endswith("。") for t in _ALL_TPL), "有回复模板不以「。」结尾 ⇒ 标签插入点不唯一"
assert all("（" not in t and "）" not in t for t in _ALL_TPL), "有回复模板已含全角括号 ⇒ 标签不可解析"
assert all("（" not in t for f in FAMILIES for t in f["t3_user"]), "轮3 句面含全角括号"


# ---------------- 标签：构造与反向解析 ----------------
def label_marker(letter: str) -> str:
    return LABEL_TMPL.format(letter=letter)


def attach_label(det: str, letter: str) -> str:
    """DET 句面 + 标签 ⇒ reply_with_label（标签插在末尾「。」之前）。"""
    assert det.endswith("。") and "（" not in det and "）" not in det, det
    assert letter in CAND_LETTERS, letter
    return det[:-1] + label_marker(letter) + "。"


def split_label(reply: str):
    """reply ⇒ (DET句面, 字母)；标签必须恰 1 个、且紧贴末尾「。」之前；否则 None。"""
    if not reply or not reply.endswith("。"):
        return None
    ms = list(LABEL_RE.finditer(reply))
    if len(ms) != 1:
        return None
    m = ms[0]
    if m.end() != len(reply) - 1:
        return None
    return reply[: m.start()] + "。", m.group(1)


def letters_in(text: str):
    """文本里出现的标签字母列表（地板判定用；空 = 该预测没有任何字母可用）。"""
    return LABEL_RE.findall(text or "")


# ---------------- 渲染：§9 的 6 行 + 候选行（与 74 逐字相同）----------------
def render_cand_line(pairs) -> str:
    """pairs = [(ent, cond), …]（顺序即位置）⇒ 「候选：A. e / c；B. …」一行。"""
    items = [CAND_ITEM.format(letter=CAND_LETTERS[i], ent=e, cond=c) for i, (e, c) in enumerate(pairs)]
    return CAND_PREFIX + CAND_SEP.join(items)


def render_ctx(r1u: str, r1s: str, r2u: str, r2s: str, r3u: str, cand_line: str) -> str:
    return (f"{USER_PREFIX}{r1u}\n{RESP_PREFIX}{r1s}{EOS}\n"
            f"{USER_PREFIX}{r2u}\n{RESP_PREFIX}{r2s}{EOS}\n"
            f"{USER_PREFIX}{r3u}\n{cand_line}\n{RESP_PREFIX}")


def parse_cand_line(line: str):
    """解析候选行 ⇒ [(letter, ent, cond), …]；格式不对返回 None（字母必须按位置 A.. 递增）。"""
    if not line.startswith(CAND_PREFIX):
        return None
    body = line[len(CAND_PREFIX):]
    out = []
    for i, item in enumerate(body.split(CAND_SEP)):
        m = CAND_ITEM_RE.match(item)
        if not m:
            return None
        letter, ent, cond = m.group(1), m.group(2), m.group(3)
        if letter != CAND_LETTERS[i]:
            return None
        out.append((letter, ent, cond))
    return out or None


def parse_ctx(ctx: str):
    """只用 ctx 文本复原 (族, t1_idx, t2_idx, t3_idx, 实体, 条件)；复原不出或歧义返回 None。

    与 72::parse_ctx 同口径，只是多要求第 6 行（index 5）是合法候选行、且候选落在该族词表内。
    """
    lines = ctx.split("\n")
    if len(lines) != N_LINE_CTX:
        return None
    if lines[6] != RESP_PREFIX:
        return None
    for i in (0, 2, 4):
        if not lines[i].startswith(USER_PREFIX):
            return None
    for i in (1, 3):
        if not lines[i].startswith(RESP_PREFIX) or not lines[i].endswith(EOS):
            return None
    cands = parse_cand_line(lines[5])
    if cands is None:
        return None
    r1u = lines[0][len(USER_PREFIX):]
    r2u = lines[2][len(USER_PREFIX):]
    r3u = lines[4][len(USER_PREFIX):]
    found = []
    for fam in FAMILIES:
        m1 = d72._match_slot(r1u, fam["t1_user"], set(fam["ents"]), "ent")
        if len(m1) != 1:
            continue
        i1, ent = m1[0]
        m2 = d72._match_slot(r2u, fam["t2_user"], set(fam["conds"]), "cond")
        if len(m2) != 1:
            continue
        i2, cond = m2[0]
        if r3u not in fam["t3_user"]:
            continue
        i3 = fam["t3_user"].index(r3u)
        found.append((fam["key"], i1, i2, i3, ent, cond))
    if len(found) != 1:
        return None
    fk = found[0][0]
    fam = FAM_BY_KEY[fk]
    for _l, e, c in cands:
        if e not in fam["ents"] or c not in fam["conds"]:
            return None
    return found[0]


# ---------------- 候选构造（与 74 逐字相同 ⇒ 候选层与 r3a 逐字节一致）----------------
def build_candidates(rng, fam, ent: str, cond: str, k: int):
    """返回 (pairs, gold_idx)：恰一个 (ent, cond)；混入同实体/同条件干扰项；正确项位置均匀随机。"""
    conds_other = [c for c in fam["conds"] if c != cond]
    ents_other = [e for e in fam["ents"] if e != ent]
    distract = [(ent, rng.choice(conds_other)), (rng.choice(ents_other), cond)]
    used = {(ent, cond)} | set(distract)
    pool = [(e, c) for e in fam["ents"] for c in fam["conds"] if (e, c) not in used]
    rng.shuffle(pool)
    for p in pool:
        if len(distract) >= k - 1:
            break
        distract.append(p)
    assert len(distract) >= k - 1, f"{fam['key']} 族的组合不足以造 {k} 个候选"
    pairs = [(ent, cond)] + distract[:k - 1]
    assert len(set(pairs)) == k, "候选必须互异"
    rng.shuffle(pairs)
    return pairs, pairs.index((ent, cond))


def _cand_records(pairs):
    return [{"idx": i, "letter": CAND_LETTERS[i], "ent": e, "cond": c,
             "text": CAND_ITEM.format(letter=CAND_LETTERS[i], ent=e, cond=c)}
            for i, (e, c) in enumerate(pairs)]


# ---------------- 生成 ----------------
def gen_records(seed: int, n_train: int, n_test: int, k: int):
    """组合空间 shuffle 后切前 need 条；shuffle 种子与 72/74 相同 ⇒ 对话部分与 §9 逐条对齐、
    候选层与 r3a（74/78）逐字节对齐。**唯一改动**：target = DET 句面 + 标签。"""
    combos = d72.build_combos()
    rng_split = random.Random(seed)
    rng_split.shuffle(combos)
    need = n_train + n_test
    if len(combos) < need:
        raise SystemExit(f"[D76] 组合空间不足：{len(combos)} < {need}")
    rng_cand = random.Random(seed ^ SEED_CAND_XOR)      # ★ 与 74 同流
    recs = []
    for fi, i1, i2, i3, ent, cond in combos[:need]:
        fam = FAMILIES[fi]
        pairs, gold_idx = build_candidates(rng_cand, fam, ent, cond, k)
        cands = _cand_records(pairs)
        cand_line = render_cand_line(pairs)
        r1u = fam["t1_user"][i1].format(ent=ent)
        r1s = fam["t1_sys"][i1].format(ent=ent)
        r2u = fam["t2_user"][i2].format(cond=cond)
        r2s = fam["t2_sys"][i2].format(cond=cond)
        r3u = fam["t3_user"][i3]
        gold_det = d72.det_reply(fam["key"], i3, ent, cond)
        gold_letter = CAND_LETTERS[gold_idx]
        marker = label_marker(gold_letter)
        reply_with_label = attach_label(gold_det, gold_letter)
        ctx = render_ctx(r1u, r1s, r2u, r2s, r3u, cand_line)
        fam_ents, fam_conds = set(fam["ents"]), set(fam["conds"])
        n_same_ent = sum(1 for c in cands if c["ent"] == ent and c["cond"] != cond)
        n_same_cond = sum(1 for c in cands if c["cond"] == cond and c["ent"] != ent)
        n_ref_match = sum(1 for c in cands if c["ent"] == ent and c["cond"] == cond)
        recs.append(dict(
            ctx=ctx, target=reply_with_label, gold=reply_with_label,
            family=fam["key"], ent=ent, cond=cond,
            cond_type=fam["cond_type"], ent_kind=fam["ent_kind"],
            t1_idx=i1, t2_idx=i2, t3_idx=i3,
            tpl_id=f"{fam['key']}#t3_{i3}",
            t1_user=r1u, t1_sys=r1s, t2_user=r2u, t2_sys=r2s, t3=r3u,
            n_turns=3,
            # ---- r3a 的候选层字段（沿用）----
            cands=cands, gold_idx=gold_idx, n_cands=k,
            gold_letter=gold_letter, cand_line=cand_line,
            cand_line_chars=len(cand_line),
            n_distract_same_ent=n_same_ent, n_distract_same_cond=n_same_cond,
            n_cands_in_vocab=sum(1 for c in cands if c["ent"] in fam_ents and c["cond"] in fam_conds),
            # ---- 本单元新增：标签与两个可判目标 ----
            gold_det=gold_det, label_marker=marker, reply_with_label=reply_with_label,
            label_pos=len(gold_det) - 1,                       # 标签起始字符位置（末尾「。」之前）
            gold_letter_pos_count=sum(1 for c in cands if c["letter"] == gold_letter),
            n_hist_label_hits=sum(1 for x in (r1u, r1s, r2u, r2s, r3u, cand_line) if LABEL_RE.search(x)),
            n_ref_match=n_ref_match,                            # 同时匹配两个槽位的候选数（应恒 = 1）
            # ---- §9 的计数类字段 ----
            ent_count=sum(1 for e in ALL_ENTS if e in ctx),
            cond_count=sum(1 for c in ALL_CONDS if c in ctx),
            ent_pos=ctx.index(ent),
            t3_chars=len(r3u),
            resp_chars=len(reply_with_label), resp_bucket=len(reply_with_label) // BUCKET_CHARS,
            det_chars=len(gold_det),
            turn3_has_ent=(ent in r3u), turn3_has_cond=(cond in r3u),
            t2_has_ent=(ent in r2u), t1_has_cond=(cond in r1u),
            ent_in_t1=(ent in r1u), cond_in_t2=(cond in r2u),
            ent_in_target=(ent in reply_with_label), cond_in_target=(cond in reply_with_label),
            ent_in_ctx=(ent in ctx), cond_in_ctx=(cond in ctx),
        ))
    return recs, combos


# ---------------- 不变量（逐条断言 + 计数，违例必须 0）----------------
def _chi2_crit_999(dof: int) -> float:
    return CHI2_999.get(dof, dof + 5.0 * (2.0 * dof) ** 0.5)   # 表外 dof 用保守近似


def _reference_reply(r):
    """⑥ 参考上限：**只用 ctx 文本**复原 (族,句式,实体,条件) → 找唯一同时匹配两槽位的候选
    → DET 句面 + 该候选按位置的字母。返回 (reply|None, 失败原因)。"""
    p = parse_ctx(r["ctx"])
    if p is None:
        return None, "ctx无法复原"
    fk, _i1, _i2, i3, ent, cond = p
    cl = parse_cand_line(r["ctx"].split("\n")[5])
    hit = [q for q in cl if q[1] == ent and q[2] == cond]
    if len(hit) != 1:
        return None, "唯一匹配候选数!=1"
    l, e, c = hit[0]
    return attach_label(d72.det_reply(fk, i3, e, c), l), ""


def check_invariants(recs, tag="all", perm_rng=None):
    """返回 (vio Counter, info dict)。① ② ③(A/B/C/D/E) ④ ⑤ ⑥ 逐条判。"""
    vio = Counter()
    info = {}
    ok = Counter()

    # ---- 指标 A：ctx -> target 是函数 ----
    seen = {}
    for r in recs:
        if r["ctx"] in seen:
            vio["A_重复ctx"] += 1
            if seen[r["ctx"]] != r["target"]:
                vio["A_同ctx不同target"] += 1
        seen[r["ctx"]] = r["target"]
    info["unique_ctx"] = len(seen)
    info["dup_ctx"] = len(recs) - len(seen)

    # ---- 指标 C：非退化（唯一 ctx / 样本数 >= 0.9） ----
    ratio = len(seen) / len(recs)
    info["uniq_ctx_ratio"] = ratio
    vio["C_唯一ctx比不足"] = int(ratio < MIN_UNIQ_CTX_RATIO)

    # ---- 逐条 ① / B / D / E / ⑤ / ⑥ ----
    n_ref_ok, ref_fail = 0, Counter()
    for r in recs:
        # ① 恰 1 个候选与历史两个槽位同时匹配
        match = [c for c in r["cands"] if c["ent"] == r["ent"] and c["cond"] == r["cond"]]
        one = (len(match) == 1)
        vio["①_与历史一致的候选数!=1"] += int(not one)
        g = r["cands"][r["gold_idx"]]
        vio["①_gold_idx不是唯一匹配项"] += int(not (one and g["ent"] == r["ent"] and g["cond"] == r["cond"]))
        vio["①_候选互异坏"] += int(len({(c["ent"], c["cond"]) for c in r["cands"]}) != r["n_cands"])
        vio["①_候选超出族词表"] += int(r["n_cands_in_vocab"] != r["n_cands"])
        vio["①_gold候选(句面+标签)!=target"] += int(
            attach_label(d72.det_reply(r["family"], r["t3_idx"], g["ent"], g["cond"]), g["letter"]) != r["target"])
        ok["①"] += int(one)

        b = (r["ent_in_ctx"] and r["cond_in_ctx"] and r["ent_in_t1"] and r["cond_in_t2"]
             and r["ent_in_target"] and r["cond_in_target"])
        vio["B_实体或条件不在输入里"] += int(not b)
        d = (not r["turn3_has_ent"]) and (not r["turn3_has_cond"]) \
            and (not r["t2_has_ent"]) and (not r["t1_has_cond"])
        vio["D_轮3含实体或条件(含轮2含实体/轮1含条件)"] += int(not d)
        vio["E_逐条往返坏"] += int(not r.get("rt_ok", True))
        vio["E_target含unk"] += int(r.get("unk_in_target", False))
        ok["B"] += int(b)
        ok["D"] += int(d)

        # ⑤ 字母唯一可判（一条一条拆开计，便于定位）
        v5 = 0
        if r["gold_letter"] != CAND_LETTERS[r["gold_idx"]]:
            v5 += 1
            vio["⑤_gold_letter!=letter_of(gold_idx)"] += 1
        if not (one and match[0]["letter"] == r["gold_letter"]):
            v5 += 1
            vio["⑤_唯一匹配候选的字母!=gold_letter"] += 1
        sl = split_label(r["target"])
        if sl is None:
            v5 += 1
            vio["⑤_target标签不可解析(非恰1个/不在末尾句号前)"] += 1
        else:
            det2, letter2 = sl
            if letter2 != r["gold_letter"]:
                v5 += 1
                vio["⑤_target标签字母!=gold_letter"] += 1
            if one and letter2 != match[0]["letter"]:
                v5 += 1
                vio["⑤_target标签字母!=唯一匹配候选字母"] += 1
            if det2 != d72.det_reply(r["family"], r["t3_idx"], r["ent"], r["cond"]):
                v5 += 1
                vio["⑤_target去掉标签!=DET句面"] += 1
        if r["label_marker"] != label_marker(r["gold_letter"]):
            v5 += 1
            vio["⑤_label_marker坏"] += 1
        if r["reply_with_label"] != r["target"]:
            v5 += 1
            vio["⑤_reply_with_label!=target"] += 1
        if r["n_hist_label_hits"] != 0:
            v5 += 1
            vio["⑤_历史行或候选行含（X）标签(可整段抄)"] += 1
        if r["n_ref_match"] != 1:
            v5 += 1
            vio["⑤_同时匹配两槽位的候选数!=1"] += 1
        vio["⑤_字母唯一可判坏"] += v5

        # ⑥ 参考上限（只用 ctx 复原）
        ref, why = _reference_reply(r)
        if ref == r["target"]:
            n_ref_ok += 1
        else:
            ref_fail[why or "回复不一致"] += 1
    info["ok"] = ok
    info["ref_ok"] = n_ref_ok
    info["ref_fail"] = dict(ref_fail)
    vio["⑥_参考上限!=target的条数"] = len(recs) - n_ref_ok
    info["ref_em"] = 100.0 * n_ref_ok / len(recs)

    # ---- ② 正确项位置分布均匀 ----
    k = recs[0]["n_cands"]
    N = len(recs)
    pos = Counter(r["gold_idx"] for r in recs)
    counts = [pos.get(i, 0) for i in range(k)]
    exp = N / k
    max_dev = max(abs(c - exp) / exp * 100.0 for c in counts)
    chi2 = sum((c - exp) ** 2 / exp for c in counts)
    dof = k - 1
    crit = _chi2_crit_999(dof)
    se_pp = ((1.0 / k) * (1.0 - 1.0 / k) / N) ** 0.5 * 100.0   # 单格**比例**的 1SE（百分点）
    se_rel = se_pp * k                                        # 单格**相对偏差**的 1SE（%，= max_dev 的单位）
    vio["②_位置计数为0"] = sum(1 for c in counts if c == 0)
    vio["②_卡方超0.999临界"] = int(chi2 > crit)
    info["pos"] = dict(counts=counts, exp=exp, max_dev=max_dev, chi2=chi2, dof=dof,
                       crit=crit, se_pp=se_pp, se_rel=se_rel)

    # ---- 静态词表级 D ----
    static_ent = sum(1 for f in FAMILIES for t in f["t3_user"] for e in ALL_ENTS if e in t)
    static_cond = sum(1 for f in FAMILIES for t in f["t3_user"] for c in ALL_CONDS if c in t)
    static_t2_ent = sum(1 for f in FAMILIES for t in f["t2_user"] for e in ALL_ENTS if e in t)
    static_t1_cond = sum(1 for f in FAMILIES for t in f["t1_user"] for c in ALL_CONDS if c in t)
    vio["D_static_轮3含实体"] = static_ent
    vio["D_static_轮3含条件"] = static_cond
    vio["D_static_轮2含实体"] = static_t2_ent
    vio["D_static_轮1含条件"] = static_t1_cond
    info["static"] = dict(t3_ent=static_ent, t3_cond=static_cond,
                          t2_ent=static_t2_ent, t1_cond=static_t1_cond)

    # ---- 模板级静态检查：占位符个数 / 去重 / 标签格式前置条件 ----
    shape_vio = 0
    for f in FAMILIES:
        for t in f["t1_user"]:
            shape_vio += int(t.count("{ent}") != 1 or "{cond}" in t)
        for t in f["t1_sys"]:
            shape_vio += int(t.count("{ent}") != 1 or "{cond}" in t)
        for t in f["t2_user"]:
            shape_vio += int(t.count("{cond}") != 1 or "{ent}" in t)
        for t in f["t2_sys"]:
            shape_vio += int(t.count("{cond}") != 1 or "{ent}" in t)
        for t in f["t3_user"]:
            shape_vio += int("{" in t or "}" in t) + int("（" in t)
        for t in f["reply"]:
            shape_vio += int(t.count("{ent}") != 1 or t.count("{cond}") != 1)
            shape_vio += int(not t.endswith("。")) + int("（" in t or "）" in t)
        shape_vio += int(len(f["reply"]) != len(f["t3_user"]))
        for col in ("ents", "conds"):
            shape_vio += int(len(set(f[col])) != len(f[col]))
    vio["SHAPE_模板占位符或去重或标签前置条件坏"] = shape_vio

    # ---- 可复原判定：parse_ctx 必须逐条复原出构造时的六元组 ----
    n_parse_bad = 0
    for r in recs:
        p = parse_ctx(r["ctx"])
        if p is None or p != (r["family"], r["t1_idx"], r["t2_idx"], r["t3_idx"], r["ent"], r["cond"]):
            n_parse_bad += 1
    vio["PARSE_可复原判定失败"] = n_parse_bad

    # ---- ④ 位置置换不变性（选中项不变 + 字母随位置改且正确）----
    n_perm, n_perm_bad, n_perm_idx_moved, n_perm_letter_changed = 0, 0, 0, 0
    if perm_rng is not None:
        for r in recs:
            hist = (r["ent"], r["cond"])
            for kind in ("reverse", "rng"):
                pairs = [(c["ent"], c["cond"]) for c in r["cands"]]
                if kind == "reverse":
                    pairs = pairs[::-1]
                else:
                    perm_rng.shuffle(pairs)
                line = render_cand_line(pairs)
                ctx2 = render_ctx(r["t1_user"], r["t1_sys"], r["t2_user"], r["t2_sys"], r["t3"], line)
                p2 = parse_ctx(ctx2)
                cl2 = parse_cand_line(line)
                hit = [q for q in cl2 if (q[1], q[2]) == hist]
                idx2 = cl2.index(hit[0]) if len(hit) == 1 else -1
                letter2 = CAND_LETTERS[idx2] if idx2 >= 0 else None
                okp = False
                if p2 is not None and len(hit) == 1:
                    sel_same = (p2[0], p2[4], p2[5]) == (r["family"], r["ent"], r["cond"])
                    lab_same = (hit[0][0] == letter2)
                    rep2 = attach_label(d72.det_reply(p2[0], p2[3], hit[0][1], hit[0][2]), letter2)
                    sl2 = split_label(rep2)
                    okp = (sel_same and lab_same and sl2 is not None and sl2[1] == letter2
                           and sl2[0] == d72.det_reply(p2[0], p2[3], r["ent"], r["cond"]))
                n_perm += 1
                n_perm_bad += int(not okp)
                n_perm_idx_moved += int(idx2 != r["gold_idx"])
                n_perm_letter_changed += int(letter2 != r["gold_letter"])
    vio["④_置换后选中项或新字母坏"] = n_perm_bad
    info["perm"] = dict(checked=n_perm, idx_moved=n_perm_idx_moved, letter_changed=n_perm_letter_changed)

    print(f"[INV:{tag}] ① 候选唯一性：通过={ok['①']}/{len(recs)} | 违例 与历史一致数!=1={vio['①_与历史一致的候选数!=1']} "
          f"gold_idx错位={vio['①_gold_idx不是唯一匹配项']} 候选互异坏={vio['①_候选互异坏']} "
          f"候选越族={vio['①_候选超出族词表']} gold候选(句面+标签)!=target={vio['①_gold候选(句面+标签)!=target']}", flush=True)
    print(f"[INV:{tag}] ② 位置均匀（=字母机会地板的来源）：计数={counts} 期望={exp:.1f} 最大偏差={max_dev:.2f}% "
          f"（单格相对偏差 1SE={se_rel:.2f}% ⇒ 3SE={3 * se_rel:.2f}%）"
          f"卡方={chi2:.3f}(dof={dof} 0.999临界={crit:.3f} 单格比例1SE={se_pp:.2f}pp) "
          f"| 违例 位置计数为0={vio['②_位置计数为0']} 卡方超临界={vio['②_卡方超0.999临界']}", flush=True)
    print(f"[INV:{tag}] ③A 函数性：唯一ctx={len(seen)} 重复ctx={info['dup_ctx']} "
          f"同ctx不同target={vio['A_同ctx不同target']}（违例）", flush=True)
    print(f"[INV:{tag}] ③B 信息充分：通过={ok['B']}/{len(recs)} 违例={vio['B_实体或条件不在输入里']}", flush=True)
    print(f"[INV:{tag}] ③C 非退化：唯一ctx/样本数={ratio:.4f}（门槛>={MIN_UNIQ_CTX_RATIO}）"
          f" 违例={vio['C_唯一ctx比不足']}", flush=True)
    print(f"[INV:{tag}] ③D 必读历史：通过={ok['D']}/{len(recs)} 违例={vio['D_轮3含实体或条件(含轮2含实体/轮1含条件)']} "
          f"| 静态词表级 轮3含实体={static_ent} 轮3含条件={static_cond} "
          f"轮2含实体={static_t2_ent} 轮1含条件={static_t1_cond}", flush=True)
    print(f"[INV:{tag}] ③E 用字可编码：坏字符={info.get('bad_chars', [])} "
          f"逐条往返坏={vio['E_逐条往返坏']} target含<unk>={vio['E_target含unk']}", flush=True)
    print(f"[INV:{tag}] ③PARSE 可复原判定（族+三轮句式 index+实体+条件 六元组复原）失败={n_parse_bad} "
          f"| SHAPE 模板占位符/标签前置条件静态违例={shape_vio}", flush=True)
    print(f"[INV:{tag}] ④ 位置置换不变性：检查={n_perm}（逆序+随机 各一遍）违例={n_perm_bad} "
          f"| 置换后正确项位置变了的次数={n_perm_idx_moved} 字母跟着变的次数={n_perm_letter_changed}"
          f"（位置变/字母变是**本变体的要点**；判的是选中项不变+新字母正确）", flush=True)
    print(f"[INV:{tag}] ⑤ 字母唯一可判：违例合计={vio['⑤_字母唯一可判坏']} "
          f"(gold_letter!=letter_of(gold_idx):{vio['⑤_gold_letter!=letter_of(gold_idx)']} "
          f"唯一匹配候选字母!=gold_letter:{vio['⑤_唯一匹配候选的字母!=gold_letter']} "
          f"同时匹配两槽位候选数!=1:{vio['⑤_同时匹配两槽位的候选数!=1']} "
          f"target标签不可解析:{vio['⑤_target标签不可解析(非恰1个/不在末尾句号前)']} "
          f"target标签字母!=gold_letter:{vio['⑤_target标签字母!=gold_letter']} "
          f"target标签字母!=唯一匹配候选字母:{vio['⑤_target标签字母!=唯一匹配候选字母']} "
          f"去掉标签!=DET句面:{vio['⑤_target去掉标签!=DET句面']} "
          f"label_marker坏:{vio['⑤_label_marker坏']} reply_with_label!=target:{vio['⑤_reply_with_label!=target']} "
          f"历史/候选行含（X）标签:{vio['⑤_历史行或候选行含（X）标签(可整段抄)']})", flush=True)
    print(f"[INV:{tag}] ⑥ 参考上限（完美选择+DET句面+正确标签）：复原成功={n_ref_ok}/{len(recs)} "
          f"EM={info['ref_em']:.2f}% 违例={vio['⑥_参考上限!=target的条数']} 失败原因={info['ref_fail']}", flush=True)
    return vio, info


# ---------------- 地板（L5）：两条指标各一套 ----------------
def _em(gold: str, pred: str) -> int:
    return int((gold or "").strip() == (pred or "").strip())


def _infer_fam_i3(t3: str):
    """只用轮3 句面推断 (族, 轮3句式 index)；t3 可能跨族重复 ⇒ 返回候选列表。"""
    out = []
    for f in FAMILIES:
        if t3 in f["t3_user"]:
            out.append((f["key"], f["t3_user"].index(t3)))
    return out


def _pred_position(r, p):
    """总是选第 p 个候选（位置基准，机会级地板）⇒ DET(该候选) + 该候选的字母标签。"""
    if p >= r["n_cands"]:
        return []
    c = r["cands"][p]
    return [attach_label(d72.det_reply(fk, i3, c["ent"], c["cond"]), c["letter"])
            for fk, i3 in _infer_fam_i3(r["t3"])]


def _bucket_specs():
    return (
        ("⑤a回复长度桶(20字)", lambda r: r["resp_bucket"]),
        ("⑤b族桶", lambda r: r["family"]),
        ("⑤c轮3句式桶(族×句式)", lambda r: (r["family"], r["t3_idx"])),
        ("⑤d候选行长度桶(20字)", lambda r: r["cand_line_chars"] // BUCKET_CHARS),
        ("⑤e实体位置桶(10字)", lambda r: r["ent_pos"] // ENT_POS_BUCKET),
        ("⑤f轮3长度桶(5字)", lambda r: r["t3_chars"] // T3_BUCKET_CHARS),
    )


def _stats(train):
    st = {"maj_target": Counter(r["target"] for r in train).most_common(1)[0][0],
          "maj_letter": Counter(r["gold_letter"] for r in train).most_common(1)[0][0],
          "maj_ent": {}, "maj_cond": {}, "bucket": {}}
    for f in FAMILIES:
        sub = [r for r in train if r["family"] == f["key"]]
        st["maj_ent"][f["key"]] = Counter(r["ent"] for r in sub).most_common(1)[0][0]
        st["maj_cond"][f["key"]] = Counter(r["cond"] for r in sub).most_common(1)[0][0]
    for name, keyfn in _bucket_specs():
        per = defaultdict(Counter)
        for r in train:
            per[keyfn(r)][r["target"]] += 1
        st["bucket"][name] = {b: c.most_common(1)[0][0] for b, c in per.items()}
    return st


def _pred_full_ctx_fill(r, s):
    """⑥b 强朴素规则：只用轮3 认出 (族, 句式)，实体/条件/字母一律填 train 族内/全局众数（不看历史）。"""
    return [attach_label(d72.det_reply(fk, i3, s["maj_ent"][fk], s["maj_cond"][fk]), s["maj_letter"])
            for fk, i3 in _infer_fam_i3(r["t3"])]


CHANCE_NAMES = (
    "①固定字母always-A（只出标签）",
    "②随机字母1/k（只出标签）",
    "②b随机候选+DET+其字母（=74 的①随机猜1/k）",
    "②c总选第一个候选（位置0+DET+其字母）",
    "②d总选最后一个候选（位置k-1+DET+其字母）",
)
# 这两条是**自带 RNG 的随机装置**：其准确率的期望值恒为 1/k（与数据的位置分布无关），
# 单次抽样带 sqrt((1/k)(1-1/k)/N) 的抽样噪声 ⇒ 不能拿单次抽样值当「数据是否泄露」的判据。
RANDOM_CHANCE_NAMES = ("②随机字母1/k（只出标签）", "②b随机候选+DET+其字母（=74 的①随机猜1/k）")
SHORTCUT_NAMES = (
    "③复制轮3用户话", "③b复制轮1用户话", "③c复制轮1系统话", "③d复制轮2用户话", "③e复制轮2系统话",
    "④train全集众数（完整回复+其字母）",
    "⑤a回复长度桶(20字)", "⑤b族桶", "⑤c轮3句式桶(族×句式)", "⑤d候选行长度桶(20字)",
    "⑤e实体位置桶(10字)", "⑤f轮3长度桶(5字)",
    "⑥只出标签不出内容（字母=train众数字母）",
    "⑥b模板填空只用轮3（实体/条件/字母全填train众数）",
)
REFERENCE_NAME = "参考:完美选择(按历史匹配候选)+DET句面+正确标签"


def _rule_fns(rng_letter, rng_cand):
    """每条规则返回**预测文本列表**（多条 = 命中任一即算中，对地板更保守）。

    同一套规则在两个指标上各评一次：
      字母指标（T1）：从预测文本里用 LABEL_RE 抽字母（恰 1 个才算给出字母，0 个/多个都算错）；
      完整回复指标（T2）：整串逐字 EM。
    """
    rules = [
        ("①固定字母always-A（只出标签）", lambda r, s: [label_marker("A")]),
        ("②随机字母1/k（只出标签）",
         lambda r, s: [label_marker(rng_letter.choice(CAND_LETTERS[: r["n_cands"]]))]),
        ("②b随机候选+DET+其字母（=74 的①随机猜1/k）",
         lambda r, s: _pred_position(r, rng_cand.randrange(r["n_cands"]))),
        ("②c总选第一个候选（位置0+DET+其字母）", lambda r, s: _pred_position(r, 0)),
        ("②d总选最后一个候选（位置k-1+DET+其字母）", lambda r, s: _pred_position(r, r["n_cands"] - 1)),
        ("③复制轮3用户话", lambda r, s: [r["t3"]]),
        ("③b复制轮1用户话", lambda r, s: [r["t1_user"]]),
        ("③c复制轮1系统话", lambda r, s: [r["t1_sys"]]),
        ("③d复制轮2用户话", lambda r, s: [r["t2_user"]]),
        ("③e复制轮2系统话", lambda r, s: [r["t2_sys"]]),
        ("④train全集众数（完整回复+其字母）", lambda r, s: [s["maj_target"]]),
        ("⑥只出标签不出内容（字母=train众数字母）", lambda r, s: [label_marker(s["maj_letter"])]),
        ("⑥b模板填空只用轮3（实体/条件/字母全填train众数）", _pred_full_ctx_fill),
    ]
    for name, keyfn in _bucket_specs():
        def mk(keyfn=keyfn, name=name):
            def f(r, s):
                return [s["bucket"][name].get(keyfn(r), s["maj_target"])]
            return f
        rules.append((name, mk()))
    return rules


def compute_floors(train, test, tag="test"):
    """在 test 上逐条算 **字母准确率（T1）** 与 **完整回复 EM（T2）**；两块分开打印、分开汇总。"""
    st = _stats(train)
    rng_letter = random.Random(SEED ^ SEED_FLOOR_XOR)
    rng_cand = random.Random(SEED ^ SEED_FLOOR_XOR ^ 0x5A5A)
    k = test[0]["n_cands"]
    rules = _rule_fns(rng_letter, rng_cand)
    names = [n for n, _ in rules]
    assert len(set(names)) == len(names), "地板规则名重复"
    assert set(names) == set(CHANCE_NAMES) | set(SHORTCUT_NAMES), (
        f"地板规则与分类表不一致：多出={sorted(set(names) - set(CHANCE_NAMES) - set(SHORTCUT_NAMES))} "
        f"缺失={sorted(set(CHANCE_NAMES) | set(SHORTCUT_NAMES) - set(names))}")

    def eval_rules(rs):
        rows = {}
        for name, fn in rules:
            hit_l = hit_em = 0
            for r in rs:
                preds = fn(r, st) or [""]
                hit_l += int(any(letters_in(p) and letters_in(p)[0] == r["gold_letter"] and len(letters_in(p)) == 1
                                 for p in preds))
                hit_em += int(any(_em(r["target"], p) for p in preds))
            rows[name] = (100.0 * hit_l / len(rs), 100.0 * hit_em / len(rs))
        return rows

    rows = eval_rules(test)

    # ---- 块 1：字母指标（T1）----
    print(f"[FLOOR-L] {tag} k={k} ★字母指标（T1＝字母准确率）n={len(test)} 原始行：", flush=True)
    for name, _fn in rules:
        lacc = rows[name][0]
        print(f"[FLOOR-L] {tag} k={k}   {name} 字母准确率={lacc:.2f}%", flush=True)
    scan_l, scan_f = [], []
    for p in range(k):
        hl = he = 0
        for r in test:
            preds = _pred_position(r, p) or [""]
            hl += int(any(len(letters_in(x)) == 1 and letters_in(x)[0] == r["gold_letter"] for x in preds))
            he += int(any(_em(r["target"], x) for x in preds))
        scan_l.append(100.0 * hl / len(test))
        scan_f.append(100.0 * he / len(test))
    print(f"[FLOOR-L] {tag} k={k}   位置扫描 always_p 的**字母准确率**：" +
          " ".join(f"p{i}={scan_l[i]:.2f}%" for i in range(k)) +
          f" | min={min(scan_l):.2f}% max={max(scan_l):.2f}% 解析机会水平=100/k={100.0 / k:.2f}%", flush=True)
    chance_l = {n: rows[n][0] for n in CHANCE_NAMES}
    short_l = {n: rows[n][0] for n in SHORTCUT_NAMES}
    mc_l_name = max(chance_l, key=chance_l.get)
    se_l = 2.0 * ((1.0 / k) * (1 - 1.0 / k) / len(test)) ** 0.5 * 100.0
    dev_l = chance_l[mc_l_name] - 100.0 / k
    # 随机装置的重复估计：把「单次抽样的运气」与「数据的性质」分开（R=201 次，各次用独立 RNG 流）
    reps = {n: [] for n in RANDOM_CHANCE_NAMES}
    R_REP = 201
    for t in range(R_REP):
        rl = random.Random((SEED ^ SEED_FLOOR_XOR) + 1000 * (t + 1))
        rc = random.Random((SEED ^ SEED_FLOOR_XOR ^ 0x5A5A) + 1000 * (t + 1))
        hit = {n: 0 for n in RANDOM_CHANCE_NAMES}
        for r in test:
            p = label_marker(rl.choice(CAND_LETTERS[: r["n_cands"]]))
            hit[RANDOM_CHANCE_NAMES[0]] += int(len(letters_in(p)) == 1 and letters_in(p)[0] == r["gold_letter"])
            preds = _pred_position(r, rc.randrange(r["n_cands"])) or [""]
            hit[RANDOM_CHANCE_NAMES[1]] += int(any(len(letters_in(x)) == 1 and letters_in(x)[0] == r["gold_letter"]
                                                   for x in preds))
        for n in RANDOM_CHANCE_NAMES:
            reps[n].append(100.0 * hit[n] / len(test))
    info_rep = {}
    for n in RANDOM_CHANCE_NAMES:
        mu = sum(reps[n]) / R_REP
        sd = (sum((x - mu) ** 2 for x in reps[n]) / (R_REP - 1)) ** 0.5
        info_rep[n] = (mu, sd, min(reps[n]), max(reps[n]))
        print(f"[FLOOR-L] {tag} k={k}   ★{n} 重复{R_REP}次（各次独立 RNG 流）：均值={mu:.2f}% "
              f"标准差={sd:.2f}% min={min(reps[n]):.2f}% max={max(reps[n]):.2f}% | 解析期望值恒=100/k="
              f"{100.0 / k:.2f}%（该规则自带 RNG，期望与数据的位置分布**无关**）| 单次抽样值={rows[n][0]:.2f}%", flush=True)
    print(f"[FLOOR-L] {tag} k={k}   机会级 max_chance={chance_l[mc_l_name]:.2f}% 来自={mc_l_name} "
          f"| 解析 1/k={100.0 / k:.2f}% Δ={dev_l:+.2f}pp 2SE={se_l:.2f}pp ⇒ "
          f"{'与 1/k 不可区分（位置不泄漏字母）' if abs(dev_l) <= se_l else '★超出 2SE ⇒ 查位置/线索泄露'}",
          flush=True)
    ms_l_name = max(short_l, key=short_l.get)
    print(f"[FLOOR-L] {tag} k={k}   非位置类（复制/众数/桶/只出标签）最大={short_l[ms_l_name]:.2f}% 来自={ms_l_name}"
          f"（这类在**字母指标**上天然≈1/k：字母是 1 位、猜对概率就是 1/k；> 1/k+2SE 才叫泄露）", flush=True)
    ma_l_name = max(rows, key=lambda n: rows[n][0])
    print(f"[FLOOR-L] {tag} k={k}   全部规则 max_all={rows[ma_l_name][0]:.2f}% 来自={ma_l_name} ⇒ "
          f"{'无规则超过 1/k+2SE ⇒ 字母不可被任何可见线索预测' if rows[ma_l_name][0] <= 100.0 / k + se_l else '★有规则超过 1/k+2SE'}",
          flush=True)
    fixed_dev = {n: chance_l[n] - 100.0 / k for n in CHANCE_NAMES if n not in RANDOM_CHANCE_NAMES}
    mfd_name = max(fixed_dev, key=lambda n: abs(fixed_dev[n]))
    print(f"[FLOOR-L] {tag} k={k}   ★数据侧判据（**固定**机会规则＝数据位置计数的确定性函数）："
          + " ".join(f"{n}={chance_l[n]:.2f}%(Δ{v:+.2f}pp)" for n, v in fixed_dev.items())
          + f" | 最大偏离={abs(fixed_dev[mfd_name]):.2f}pp 来自={mfd_name} 单条 2SE={se_l:.2f}pp ⇒ "
          + ("全在 2SE 内 ⇒ 字母机会水平=1/k（位置均匀）" if abs(fixed_dev[mfd_name]) <= se_l else "★超 2SE"),
          flush=True)
    __strict_name = max(chance_l, key=chance_l.get)
    if __strict_name in RANDOM_CHANCE_NAMES:
        print(f"[FLOOR-L] {tag} k={k}   ★口径说明：max_chance 的单次抽样最大值来自**自带 RNG 的随机装置**"
              f"（{__strict_name}={chance_l[__strict_name]:.2f}%）；该规则期望值恒=1/k，上面 {R_REP} 次重复已证"
              f"（均值 {info_rep[__strict_name][0]:.2f}%，与 1/k 差 {info_rep[__strict_name][0] - 100.0 / k:+.2f}pp）"
              f"⇒ 它的单次偏离是抽样噪声，**不是**数据泄露；数据侧判据看上一行。", flush=True)

    # ---- 块 2：完整回复指标（T2）----
    print(f"[FLOOR-F] {tag} k={k} ★完整回复指标（T2＝逐字 EM，含标签与句号）n={len(test)} 原始行：", flush=True)
    for name, _fn in rules:
        print(f"[FLOOR-F] {tag} k={k}   {name} EM={rows[name][1]:.2f}%", flush=True)
    print(f"[FLOOR-F] {tag} k={k}   位置扫描 always_p 的**完整回复 EM**：" +
          " ".join(f"p{i}={scan_f[i]:.2f}%" for i in range(k)) +
          f" | min={min(scan_f):.2f}% max={max(scan_f):.2f}% 解析机会水平=100/k={100.0 / k:.2f}%", flush=True)
    chance_f = {n: rows[n][1] for n in CHANCE_NAMES}
    short_f = {n: rows[n][1] for n in SHORTCUT_NAMES}
    mc_f_name = max(chance_f, key=chance_f.get)
    ms_f_name = max(short_f, key=short_f.get)
    mx = short_f[ms_f_name]
    se2 = 2.0 * (mx / 100.0 * (1 - mx / 100.0) / len(test)) ** 0.5 * 100.0
    print(f"[FLOOR-F] {tag} k={k}   机会级 max_chance={chance_f[mc_f_name]:.2f}% 来自={mc_f_name} "
          f"（构造期望=100/k={100.0 / k:.2f}%；随机/固定位置猜候选都属这一类，**不是退化判据**）", flush=True)
    print(f"[FLOOR-F] {tag} k={k}   捷径级 max_naive={mx:.2f}% 来自={ms_f_name} | 2SE={se2:.2f}pp ⇒ "
          f"R3a′-b 门槛=max_naive+2SE={mx + se2:.2f}% | max_naive<=5%: "
          f"{'PASS' if mx <= 5.0 else 'FAIL'}", flush=True)
    ma_f_name = max(rows, key=lambda n: rows[n][1])
    print(f"[FLOOR-F] {tag} k={k}   全部规则（含机会级）max_all={rows[ma_f_name][1]:.2f}% 来自={ma_f_name}"
          f"（★ 受 1/k 机会水平支配，不是退化判据；退化判据看上面的捷径级 max_naive）", flush=True)

    # ---- 参考上限（两条指标都应是 100.00%）----
    ref_l = ref_f = 0
    for r in test:
        rep = _reference_reply(r)[0]
        if rep is None:
            continue
        sl = split_label(rep)
        ref_l += int(sl is not None and sl[1] == r["gold_letter"])
        ref_f += int(_em(r["target"], rep))
    print(f"[FLOOR-L] {tag} k={k}   {REFERENCE_NAME} 字母准确率={100.0 * ref_l / len(test):.2f}% (n={len(test)})", flush=True)
    print(f"[FLOOR-F] {tag} k={k}   {REFERENCE_NAME} EM={100.0 * ref_f / len(test):.2f}% (n={len(test)})"
          f" ⇒ 信息全在输入里（非地板）", flush=True)

    # ---- train 侧同口径（数据是否退化：捷径规则在 train 上 >= 90% ⇒ 数据无效）----
    rows_train = eval_rules(train)
    tr_l = max(short_l, key=lambda n: rows_train[n][0])
    tr_f = max(short_f, key=lambda n: rows_train[n][1])
    print(f"[FLOOR-L] {tag} k={k}   train 非位置类最大字母准确率={rows_train[tr_l][0]:.2f}% 来自={tr_l} ⇒ "
          + ("★数据无效" if rows_train[tr_l][0] >= 90.0 else "有效"), flush=True)
    print(f"[FLOOR-F] {tag} k={k}   train 捷径级 max_naive={rows_train[tr_f][1]:.2f}% 来自={tr_f} ⇒ "
          + ("★数据无效（捷径规则在 train 上>=90%）" if rows_train[tr_f][1] >= 90.0
             else "有效（无捷径规则在 train 上>=90%）"), flush=True)
    return dict(rows=rows, chance_l=chance_l, short_l=short_l, max_chance_l=chance_l[mc_l_name],
                max_all_l=rows[ma_l_name][0], max_all_l_name=ma_l_name, mc_l_name=mc_l_name, dev_l=dev_l, se_l=se_l,
                fixed_dev=fixed_dev, mfd_name=mfd_name, strict_l_name=max(chance_l, key=chance_l.get),
                rep=info_rep, R_REP=R_REP,
                chance_f=chance_f, short_f=short_f, max_chance_f=chance_f[mc_f_name], mc_f_name=mc_f_name,
                max_naive_f=mx, ms_f_name=ms_f_name, se2=se2, ref_l=100.0 * ref_l / len(test),
                ref_f=100.0 * ref_f / len(test), rows_train=rows_train, scan_l=scan_l, scan_f=scan_f)


# ---------------- 与 §9 / r3a 的对齐核验 ----------------
def check_align_72(recs, n_train, smoke=False):
    """对话部分与 §9 产物逐条比对；smoke 下比对 72s_ 产物（没有就跳过，不当违例）。"""
    out = {}
    tmpl = "72s_dialog_{split}.jsonl" if smoke else "72_dialog_{split}.jsonl"
    for split, rs in (("train", recs[:n_train]), ("test", recs[n_train:])):
        fp = os.path.join(LOG_DIR, tmpl.format(split=split))
        if not os.path.exists(fp):
            out[split] = None
            continue
        old = [json.loads(ln) for ln in open(fp, encoding="utf-8") if ln.strip()]
        same = sum(1 for a, b in zip(rs, old)
                   if "\n".join(a["ctx"].split("\n")[:5]) + "\n" + RESP_PREFIX == b["ctx"])
        out[split] = (same, len(rs), len(old))
    return out


def check_align_r3a(recs, n_train, k, smoke):
    """与 r3a（74/78）产物的**候选层**逐条比对：cand_line / gold_idx / n_cands 必须逐字节一致。"""
    prefix = ALIGN_PREFIX_BY_K[k] + ("s" if smoke else "")
    out = {}
    for split, rs in (("train", recs[:n_train]), ("test", recs[n_train:])):
        fp = os.path.join(LOG_DIR, f"{prefix}_dialog_{split}.jsonl")
        if not os.path.exists(fp):
            out[split] = None
            continue
        old = [json.loads(ln) for ln in open(fp, encoding="utf-8") if ln.strip()]
        n = min(len(rs), len(old))
        bad_line = bad_idx = bad_n = 0
        for a, b in zip(rs[:n], old[:n]):
            bad_line += int(a["cand_line"] != b["cand_line"])
            bad_idx += int(a["gold_idx"] != b["gold_idx"])
            bad_n += int(a["n_cands"] != b["n_cands"])
        out[split] = dict(path=fp, sha=d72.file_sha256(fp), n=n, n_old=len(old),
                          bad_line=bad_line, bad_idx=bad_idx, bad_n=bad_n)
    return out


# ---------------- 入口 ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="S76 L3-r3a′ 「选择进入输出」的字母标签对话数据生成器 + 地板（纯 CPU）")
    ap.add_argument("--k", type=int, default=8, help="候选数（只接受 8 / 4：产物名 76_/77_ 写死；默认 8 = 主档）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--smoke", action="store_true", help="小量冒烟（train 200 / test 100，写 76s_/77s_）")
    a = ap.parse_args(argv)

    if a.k not in PREFIX_BY_K:
        raise SystemExit(f"[D76] 只接受 --k in {sorted(PREFIX_BY_K)}（产物名 76_/77_ 写死），收到 {a.k}")
    prefix = PREFIX_BY_K[a.k] + ("s" if a.smoke else "")
    if a.smoke:
        a.n_train, a.n_test = N_SMOKE_TRAIN, N_SMOKE_TEST

    src_sha = hashlib.sha256(open(D72_PATH, "rb").read()).hexdigest()[:16]
    print(f"[SRC] 72_nl_dialog_gen.py sha256[:16]={src_sha}（族定义/DET 回复/槽位匹配/指纹全部复用该文件）", flush=True)
    print(f"[LABEL-FMT] 标签格式：LABEL_TMPL={LABEL_TMPL!r} LABEL_RE={LABEL_RE.pattern!r} "
          f"插入位置=末尾句号之前 ⇒ reply_with_label = DET句面[:-1] + 标签 + '。'", flush=True)
    print(f"[LABEL-FMT] 反向解析 split_label(reply)：标签恰 1 个且紧贴末尾「。」之前 ⇒ (DET句面, 字母)，否则 None", flush=True)
    _svc = FAM_BY_KEY["售后"]
    _det = d72.det_reply("售后", 2, "AS975482", "返厂检测")
    for kk, letter in ((8, "C"), (4, "A")):
        print(f"[LABEL-FMT] 样例 k={kk}：DET句面={_det!r} 标签={label_marker(letter)!r} "
              f"⇒ reply_with_label={attach_label(_det, letter)!r}", flush=True)
    print(f"[LABEL-FMT] 样例位置随机化对照（同一 DET 句面、不同位置的字母）："
          + " | ".join(f"正确项在 index={i} ⇒ {attach_label(_det, CAND_LETTERS[i])!r}" for i in range(min(4, a.k))),
          flush=True)
    print(f"[CAND-FMT] 候选行文本格式（沿用 r3a/74）：CAND_PREFIX={CAND_PREFIX!r} CAND_SEP={CAND_SEP!r} "
          f"CAND_ITEM={CAND_ITEM!r} 字母=CAND_LETTERS[:k]", flush=True)

    recs, combos = gen_records(a.seed, a.n_train, a.n_test, a.k)
    train, test = recs[:a.n_train], recs[a.n_train:]
    print(f"[SPLIT] k={a.k} 组合空间={len(combos)}（无放回抽样后 train={len(train)} test={len(test)}）"
          f" | 族={N_FAMILIES} 轮1/2/3句式和="
          + " ".join(f"{f['key']}[{len(f['t1_user'])}/{len(f['t2_user'])}/{len(f['t3_user'])}]"
                     for f in FAMILIES), flush=True)
    print(f"[TPL] DET 句面表（回复模板 = DET(族, 轮3句式 index) + 标签，回复模板总数={N_REPLY_TEMPLATES}）：", flush=True)
    for f in FAMILIES:
        for i3, tpl in enumerate(f["reply"]):
            print(f"[TPL] {f['key']}#t3_{i3} 轮3={f['t3_user'][i3]!r} ⇒ DET={tpl} 例（正确项在 index 2）⇒ "
                  f"{attach_label(tpl.format(ent=f['ents'][0], cond=f['conds'][0]), CAND_LETTERS[2])}", flush=True)
    print("[VOCAB] " + " | ".join(
        f"{f['key']} 实体[{f['ent_kind']}]={len(f['ents'])} 条件[{f['cond_type']}]={len(f['conds'])}"
        for f in FAMILIES), flush=True)

    al = check_align_72(recs, a.n_train, a.smoke)
    for split, v in al.items():
        print(f"[ALIGN] 与 logs/72{'s' if a.smoke else ''}_dialog_{split}.jsonl 的对话部分逐条一致："
              + ("缺 72 产物，跳过" if v is None else f"{v[0]}/{v[1]}（72 侧 {v[2]} 条）"), flush=True)
    al3 = check_align_r3a(recs, a.n_train, a.k, a.smoke)
    n_align_bad = 0
    for split, v in al3.items():
        if v is None:
            print(f"[ALIGN-74] 与 r3a（{ALIGN_PREFIX_BY_K[a.k]}{'s' if a.smoke else ''}）{split} 产物比对：缺产物，跳过", flush=True)
            continue
        n_align_bad += v["bad_line"] + v["bad_idx"] + v["bad_n"]
        print(f"[ALIGN-74] 与 r3a {os.path.basename(v['path'])} 的候选层逐条一致：n={v['n']} "
              f"cand_line不一致={v['bad_line']} gold_idx不一致={v['bad_idx']} n_cands不一致={v['bad_n']} "
              f"| r3a 产物 sha256={v['sha'][:16]}（★ 唯一改动 = 回复加标签，候选层必须逐字节相同）", flush=True)

    # ---- tokenizer：逐字往返（③E）+ 逐条编码 + 指纹 ----
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    V = tok.get_vocab_size()
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids                 # noqa: E731
    dec = lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=False)  # noqa: E731
    EOS_ID = tok.token_to_id(EOS)
    UNK_ID = tok.token_to_id("<unk>")
    assert enc(RESP_PREFIX) == [140] and enc(EOS) == [EOS_ID], (enc(RESP_PREFIX), enc(EOS))
    for ch in "（）ABCDEFGH。":
        assert dec(enc(ch)) == ch, f"标签/字母字符 {ch!r} 不可逐字往返 ⇒ 停"

    chars = sorted({c for r in recs for s in (r["ctx"], r["target"]) for c in s})
    bad_chars = [(c, enc(c), dec(enc(c))) for c in chars if dec(enc(c)) != c]
    print(f"[E-CHECK] 数据中不同字符数={len(chars)} 逐字往返坏字符={len(bad_chars)} "
          f"{bad_chars} | V={V} EOS={EOS_ID} UNK={UNK_ID}", flush=True)

    for r in recs:
        r["p"] = enc(r["ctx"])
        r["t"] = enc(r["target"] + EOS)
        r["ctx_tok"] = len(r["p"])
        r["resp_tok"] = len(r["t"])
        r["unk_in_ctx"] = UNK_ID in r["p"]
        r["unk_in_target"] = UNK_ID in r["t"]
        r["rt_ok"] = (dec(r["p"]) == r["ctx"] and dec(r["t"]) == r["target"] + EOS)

    n_unk = sum(1 for r in recs if r["unk_in_ctx"] or r["unk_in_target"])
    print(f"[E-CHECK] 含 <unk> 的条数（ctx 或 target）={n_unk}/{len(recs)}（必须 0）", flush=True)
    assert n_unk == 0, "出现 <unk> ⇒ 有字符不在词表里"
    n_additive_bad = sum(1 for r in recs if enc(r["ctx"] + r["target"] + EOS) != r["p"] + r["t"])
    assert max(r["ctx_tok"] for r in recs) + max(r["resp_tok"] for r in recs) <= MAX_SEQ, "ctx+target 超 512"
    print(f"[FMT] 编解码整串往返坏={sum(1 for r in recs if not r['rt_ok'])} | "
          f"enc 可加性破坏={n_additive_bad} | ctx tok 中位="
          f"{sorted(r['ctx_tok'] for r in recs)[len(recs)//2]} 最大={max(r['ctx_tok'] for r in recs)} "
          f"| target tok 中位={sorted(r['resp_tok'] for r in recs)[len(recs)//2]} "
          f"最大={max(r['resp_tok'] for r in recs)}", flush=True)

    tr_fp, te_fp = d72.data_fp(train), d72.data_fp(test)

    # ---- 不变量 ----
    perm_rng = random.Random(a.seed ^ SEED_PERM_XOR)
    vio, info = check_invariants(train + test, tag="all", perm_rng=perm_rng)
    vio["E_坏字符"] = len(bad_chars)
    vio["ALIGN_r3a候选层不一致"] = n_align_bad
    info["bad_chars"] = [c for c, _i, _b in bad_chars]
    total_vio = sum(vio.values())

    pos_te = Counter(r["gold_idx"] for r in test)
    print(f"[DIST] 族分布={dict(Counter(r['family'] for r in recs))} | "
          f"条件类型分布={dict(Counter(r['cond_type'] for r in recs))} | "
          f"轮3句式分布={dict(sorted(Counter(r['tpl_id'] for r in recs).items()))} | "
          f"候选数分布={dict(Counter(r['n_cands'] for r in recs))} | "
          f"字母分布={dict(sorted(Counter(r['gold_letter'] for r in recs).items()))} | "
          f"同实体干扰项个数分布={dict(sorted(Counter(r['n_distract_same_ent'] for r in recs).items()))} | "
          f"同条件干扰项个数分布={dict(sorted(Counter(r['n_distract_same_cond'] for r in recs).items()))} | "
          f"候选行长度(字符) min/中位/max="
          f"{min(r['cand_line_chars'] for r in recs)}/"
          f"{sorted(r['cand_line_chars'] for r in recs)[len(recs)//2]}/"
          f"{max(r['cand_line_chars'] for r in recs)} | "
          f"test 位置计数={[pos_te.get(i, 0) for i in range(a.k)]}", flush=True)

    # ---- 落盘（p/t 只在内存里用于指纹与长度统计，不写进 jsonl）----
    for split, rs in (("train", train), ("test", test)):
        fp = OUT_TMPL.format(prefix=prefix, split=split)
        with open(fp, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("p", "t")},
                                   ensure_ascii=False) + "\n")
        print(f"[OUT] {split} → {fp} ({len(rs)} 条) sha256={d72.file_sha256(fp)}", flush=True)

    # ---- 地板（两条指标各一套，分开打印）----
    fl = compute_floors(train, test, tag="test")
    mx = fl["max_naive_f"]

    # ---- [GEN] 样例（完整 ctx → gold；人工核「字母确实是 gold_idx 那个」）----
    picked, seen_fam = [], set()
    for r in test:
        if r["family"] not in seen_fam:
            seen_fam.add(r["family"])
            picked.append(r)
        if len(picked) == 3:
            break
    if picked and picked[0]["gold_letter"] == "A":        # 尽量展示一个字母非 A 的样例
        alt = next((r for r in test if r["gold_letter"] != "A" and r["family"] != picked[0]["family"]), None)
        if alt is not None:
            picked[0] = alt
    for k2, r in enumerate(picked, 1):
        print(f"[GEN{k2}] family={r['family']} ent={r['ent']!r} cond={r['cond']!r} "
              f"cond_type={r['cond_type']} tpl_id={r['tpl_id']} n_cands={r['n_cands']} "
              f"gold_idx={r['gold_idx']} gold_letter={r['gold_letter']} label_pos={r['label_pos']}", flush=True)
        for i, ln in enumerate(r["ctx"].split("\n"), 1):
            print(f"[GEN{k2}] ctx{i}| {ln}", flush=True)
        for c in r["cands"]:
            em_ = c["ent"] == r["ent"]
            cm_ = c["cond"] == r["cond"]
            print(f"[GEN{k2}]   候选 {c['letter']}({c['ent']} / {c['cond']}) "
                  f"实体对上={em_} 条件对上={cm_} 两个都对上={em_ and cm_}"
                  f"{'  ← 唯一同时匹配两槽位 ⇒ 这就是 gold' if em_ and cm_ else ''}", flush=True)
        sl = split_label(r["target"])
        print(f"[GEN{k2}] gold：{r['target']}", flush=True)
        print(f"[GEN{k2}] 拆解：DET句面={sl[0]!r} 标签={label_marker(sl[1])!r} 标签字母={sl[1]!r} "
              f"== CAND_LETTERS[gold_idx={r['gold_idx']}]={CAND_LETTERS[r['gold_idx']]!r} "
              f"⇒ {'一致' if sl[1] == CAND_LETTERS[r['gold_idx']] else '★不一致'}", flush=True)
        print(f"[GEN{k2}] 同一 DET 句面若正确项在别的位置，标签就会变："
              + " ".join(f"[{i}]{label_marker(CAND_LETTERS[i])}" for i in range(min(4, r["n_cands"]))), flush=True)

    # ---- [D76] 一行汇总（六条不变量）----
    pos = info["pos"]
    v1 = sum(vio[x] for x in ("①_与历史一致的候选数!=1", "①_gold_idx不是唯一匹配项", "①_候选互异坏",
                              "①_候选超出族词表", "①_gold候选(句面+标签)!=target"))
    v2 = vio["②_位置计数为0"] + vio["②_卡方超0.999临界"]
    v3 = sum(vio[x] for x in ("A_同ctx不同target", "B_实体或条件不在输入里", "C_唯一ctx比不足",
                              "D_轮3含实体或条件(含轮2含实体/轮1含条件)", "E_坏字符", "E_逐条往返坏",
                              "E_target含unk", "PARSE_可复原判定失败",
                              "SHAPE_模板占位符或去重或标签前置条件坏",
                              "D_static_轮3含实体", "D_static_轮3含条件",
                              "D_static_轮2含实体", "D_static_轮1含条件"))
    v4 = vio["④_置换后选中项或新字母坏"]
    v5 = vio["⑤_字母唯一可判坏"]
    v6 = vio["⑥_参考上限!=target的条数"]
    print(f"[D76] k={a.k} n_train={len(train)} n_test={len(test)} 池={len(combos)}(组合空间) "
          f"seed={a.seed} prefix={prefix} | 族={N_FAMILIES} 轮3句式/族={[len(f['t3_user']) for f in FAMILIES]} "
          f"回复模板={N_REPLY_TEMPLATES} 标签模板={LABEL_TMPL!r} | "
          f"不变量违例 ①={v1} ②={v2} ③={v3} ④={v4} ⑤={v5} ⑥={v6} 合计={total_vio}"
          f"(① 一致数!=1:{vio['①_与历史一致的候选数!=1']} gold_idx错位:{vio['①_gold_idx不是唯一匹配项']} "
          f"互异坏:{vio['①_候选互异坏']} 越族:{vio['①_候选超出族词表']} "
          f"gold(句面+标签)!=target:{vio['①_gold候选(句面+标签)!=target']} | "
          f"② 位置计数为0:{vio['②_位置计数为0']} 卡方超临界:{vio['②_卡方超0.999临界']} | "
          f"③ A/B/C/D/E={vio['A_同ctx不同target']}/{vio['B_实体或条件不在输入里']}/"
          f"{vio['C_唯一ctx比不足']}/{vio['D_轮3含实体或条件(含轮2含实体/轮1含条件)']}/"
          f"{vio['E_坏字符'] + vio['E_逐条往返坏'] + vio['E_target含unk']}"
          f"(PARSE:{vio['PARSE_可复原判定失败']} SHAPE:{vio['SHAPE_模板占位符或去重或标签前置条件坏']}) | "
          f"④ 置换违例:{v4}(检查{info['perm']['checked']}次 位置变了{info['perm']['idx_moved']}次 "
          f"字母变了{info['perm']['letter_changed']}次) | ⑤ 字母唯一可判:{v5} | ⑥ 参考上限违例:{v6}) | "
          f"位置计数={pos['counts']} 期望={pos['exp']:.1f} 最大偏差={pos['max_dev']:.2f}% "
          f"卡方={pos['chi2']:.3f}/dof{pos['dof']}(临界{pos['crit']:.3f}) | "
          f"唯一ctx比={info['uniq_ctx_ratio']:.4f} 唯一ctx={info['unique_ctx']}/{len(recs)} | "
          f"字母分布={dict(sorted(Counter(r['gold_letter'] for r in recs).items()))} | "
          f"train_fp={tr_fp} test_fp={te_fp} | 参考上限(字母)={fl['ref_l']:.2f}% 参考上限(完整回复)={fl['ref_f']:.2f}%",
          flush=True)

    # ---- 判据 ----
    inv_all_zero = (v1 == 0 and v2 == 0 and v3 == 0 and v4 == 0 and v5 == 0 and v6 == 0
                    and vio["ALIGN_r3a候选层不一致"] == 0)
    # R3a′-a 的「字母机会地板 ≈ 1/k」用**数据侧判据**：固定机会规则（数据位置计数的确定性函数）
    # + 位置卡方/最大偏差。自带 RNG 的随机装置（单次抽样）另报，不参与判定（其期望恒=1/k）。
    r3a_a = (inv_all_zero and pos["chi2"] <= pos["crit"] and pos["max_dev"] <= 3.0 * pos["se_rel"]
             and abs(fl["fixed_dev"][fl["mfd_name"]]) <= fl["se_l"])
    r3a_b = (100.0 / a.k <= 25.0) and (fl["max_naive_f"] <= 5.0)
    align_ok = (all(v is None or v[0] == v[1] for v in al.values())
                and info["ref_em"] >= 99.9999
                and vio["E_坏字符"] == 0 and vio["E_逐条往返坏"] == 0)
    r3a_c = align_ok
    strict_dev = fl["chance_l"][fl["strict_l_name"]] - 100.0 / a.k
    print(f"[D76-VERDICT] R3a′-a（不变量①–⑥ 违例全 0；字母 max_chance≈1/k；位置均匀卡方过）="
          f"{'PASS' if r3a_a else 'FAIL'}"
          f"（六条违例 ①={v1}/②={v2}/③={v3}/④={v4}/⑤={v5}/⑥={v6}；"
          f"r3a 候选层不一致={vio['ALIGN_r3a候选层不一致']}；"
          f"位置卡方={pos['chi2']:.3f}<=临界{pos['crit']:.3f}；最大偏差={pos['max_dev']:.2f}%"
          f"<=3SE {3.0 * pos['se_rel']:.2f}%；固定机会规则最大偏离={abs(fl['fixed_dev'][fl['mfd_name']]):.2f}pp"
          f"<=2SE {fl['se_l']:.2f}pp 来自={fl['mfd_name']}；解析 1/k={100.0 / a.k:.2f}%）", flush=True)
    print(f"[D76-VERDICT]   ↳ ★严格口径（把自带 RNG 的随机装置的单次抽样也算进 max_chance 的 2SE 检验）："
          f"max_chance={fl['chance_l'][fl['strict_l_name']]:.2f}% 来自={fl['strict_l_name']} "
          f"Δ={strict_dev:+.2f}pp 2SE={fl['se_l']:.2f}pp ⇒ "
          f"{'也在 2SE 内' if abs(strict_dev) <= fl['se_l'] else '★超 2SE（该规则期望恒=1/k，'
             + str(fl['R_REP']) + ' 次重复均值=' + format(fl['rep'][fl['strict_l_name']][0], '.2f')
             + '%；属抽样噪声，已在上面单列）'}", flush=True)
    print(f"[D76-VERDICT] R3a′-b（余量可用：机会地板 1/k<=25% 且 完整回复捷径级 max_naive<=5%）="
          f"{'PASS' if r3a_b else 'FAIL'}（1/k={100.0 / a.k:.2f}% 来自构造；max_naive={fl['max_naive_f']:.2f}% "
          f"来自={fl['ms_f_name']}）", flush=True)
    print(f"[D76-VERDICT] R3a′-c（[ALIGN] 与 §9 逐条一致；[E-CHECK] 往返坏 0；参考上限 100.00%）="
          f"{'PASS' if r3a_c else 'FAIL'}（72 对齐=" + " ".join(
              f"{s}:{('缺' if v is None else f'{v[0]}/{v[1]}')}" for s, v in al.items())
          + f"；参考上限(⑥)={info['ref_em']:.2f}%；E坏字符={vio['E_坏字符']} 往返坏={vio['E_逐条往返坏']}）", flush=True)
    print(f"[DONE] exit={0 if total_vio == 0 else 4}", flush=True)
    return 0 if total_vio == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
