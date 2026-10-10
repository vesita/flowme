#!/usr/bin/env python3
"""S74 · L3-r3a 「带候选集的多轮对话」数据生成器 + 地板（CPU、确定性 seed、秒~分钟级）。

设计见 L3_PLAN.md §9（r2 对话数据：多轮 ctx + 五条不变量）、§11.1（本单元规格：候选集）、
§11.2（R3-a 判据）。骨架照 stages/72_nl_dialog_gen.py（不变量断言 + 地板 + [D72]/[FLOOR]/[GEN]
打印风格 + 确定性）。本文件只做**数据面与地板**：不训模型、不碰 GPU、不用 systemd-run。

在 §9 的三轮对话 ctx 上**追加一段候选列表**（共 7 行，第 6 行是新增的候选行）：
    用户：{轮1用户}
    <resp>{轮1系统}<eos>
    用户：{轮2用户}
    <resp>{轮2系统}<eos>
    用户：{轮3用户}                 ← 既不含实体也不含条件（不变量 D，来自 §9）
    候选：A. {实体} / {条件}；B. …；…   ← 新增（本单元）
    <resp>                          ← 待生成

候选列表的**文本格式**（写死，机器可解析；CAND_FMT 常量 + [CAND-FMT] 行打印样例）：
    前缀     CAND_PREFIX = "候选："
    分隔     CAND_SEP    = "；"（全角分号，项间；无尾分隔符）
    单项     CAND_ITEM   = "{letter}. {ent} / {cond}"（字母、句点、半角空格、斜杠、半角空格）
    字母     CAND_LETTERS = "ABCDEFGH"（按**位置**给，A 是第一项）
样例（k=4，取「预约」族词表）：
    候选：A. 张伟 / 周三上午；B. 李娜 / 周五晚上；C. 王芳 / 明天上午；D. 刘洋 / 周四下午
样例（k=8，同上）：
    候选：A. 张伟 / 周三上午；B. 李娜 / 周五晚上；C. 王芳 / 明天上午；D. 刘洋 / 周四下午；
         E. 陈静 / 本月二十号；F. 杨帆 / 后天下午；G. 赵磊 / 周六上午；H. 黄敏 / 下周二上午
（实际落盘是一整行，不带换行与缩进。）

候选构造（k 由 --k 定，只接受 4 / 8）：
    恰 1 个候选 = 历史（轮1 实体 + 轮2 条件）对应的那一对（不变量 ①）；
    其余 k-1 个取自**同族其它组合**，并**故意混入** 1 个「同实体不同条件」+ 1 个「同条件不同实体」
    ⇒ 单看实体、或单看条件都选不出唯一项，必须两个槽位**同时**对上（这是本层的难点所在）；
    正确项位置 = rng.randrange(k) 均匀随机（不变量 ②）。

target 与 §9 **同形**：DET(族, 轮3句式 index) + 实体 + 条件 ⇒ 完美选择时与 72 的 gold 逐字相同。
（回复必须用**被选中的那一对**；正确项唯一 ⇒ 正确回复 = §9 的 gold。）

不变量（逐条断言 + 打印计数，违例必须 0，否则退出码非 0）
    ① 恰 1 个候选与历史一致（且 gold_idx 指向它、候选互异、候选不越出族词表、gold 候选生成的回复 == target）；
    ② 正确项位置分布均匀（报每位置计数 + 最大偏差百分比 + 卡方，卡方临界限取 0.999 分位）；
    ③ 继承 §9：A 函数性 / B 信息充分 / C 非退化（唯一 ctx / 样本数 >= 0.9）/ D 必读历史 / E 用字逐字往返、无 <unk>；
    ④ 位置置换不变性：把候选顺序打乱（逆序 + 随机置换各一遍）后正确项**不变**（机器可查）。

地板（L5，两类分开报，避免把「机会水平」和「捷径」混成一个数）
    机会级 chance（构造决定 ≈ 1/k，用来证实「位置不泄漏答案」）
        ①随机猜 1/k（解析值 100/k + 实测）  ②总选第一个候选  ③总选最后一个候选（+ 逐位置扫描 always_p）；
    捷径级 shortcut（这才是「数据是否退化」的口径，目标 max_naive <= 5%）
        ④复制轮3 / ④b轮1用户 / ④c轮1系统 / ④d轮2用户 / ④e轮2系统  ⑤train 全集众数
        ⑥a..⑥j 长度 / 位置 / 族 / 候选数 / 候选行长 桶众数；
    参考上限 = 完美选择（按历史匹配候选）+ DET 句面，应 = 100%。

产物（只写 logs/）：k=4 → logs/74_dialog_{train,test}.jsonl；k=8 → logs/78_dialog_{train,test}.jsonl
    train 7000 / test 800（--smoke 写 74s_/78s_ 的 200/100）；
    每条字段 = §9 的全部字段 + cands / gold_idx / n_cands（+ 候选行与其计数做审计用）。
    cands = [{"idx","letter","ent","cond","text"}, …]（顺序即候选列表顺序，letter 按位置）。

确定性：同一 (--k --seed --n-train --n-test) 连跑两次，jsonl 逐字节相同（[OUT] 行报 sha256）。

用法（本机只有 .venv/bin/python 装了 tokenizers；本脚本无 torch、秒级）：
    .venv/bin/python stages/74_nl_dialog_cand_gen.py                 # k=4 → 74_*
    .venv/bin/python stages/74_nl_dialog_cand_gen.py --k 8           # k=8 → 78_*
    .venv/bin/python stages/74_nl_dialog_cand_gen.py --smoke
    .venv/bin/python stages/74_nl_dialog_cand_gen.py --k 8 --n-train 7000 --n-test 800
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
CAND_PREFIX = "候选："
CAND_SEP = "；"
CAND_LETTERS = "ABCDEFGH"
CAND_ITEM = "{letter}. {ent} / {cond}"
CAND_ITEM_RE = re.compile(r"^([A-H])\. (.+?) / (.+)$")
BUCKET_CHARS = 20          # 回复 / 候选行长度桶宽（字符）
T3_BUCKET_CHARS = 5        # 轮3 长度桶宽（字符）
ENT_POS_BUCKET = 10        # 实体位置桶宽（字符）
MAX_SEQ = 512              # ctx + target 字符 token 上限（与 66/70/72 同口径）
MIN_UNIQ_CTX_RATIO = 0.9   # 不变量 C：唯一 ctx / 样本数 下限
N_LINE_CTX = 7             # 本单元的 ctx 行数（§9 的 6 行 + 候选行）
PREFIX_BY_K = {4: "74", 8: "78"}   # 只接受这两个 k（文件名写死，fail-closed）
CHI2_999 = {3: 16.266, 7: 24.322}  # 卡方 0.999 分位（dof 3 / 7）
SEED_CAND_XOR = 0x74C0     # 候选位置/干扰项 RNG 流
SEED_PERM_XOR = 0x74FE     # ④ 置换检查 RNG 流
SEED_FLOOR_XOR = 0x74F1    # 地板「随机猜」RNG 流


def _load_path(name: str, path: str):
    """按路径加载同名脚本（文件名以数字开头，不能普通 import；照抄 stages/73 的写法）。"""
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


# ---------------- 渲染：§9 的 6 行 + 候选行 ----------------
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

    与 72::parse_ctx 同口径，只是多要求第 6 行是合法候选行、且候选落在该族词表内。
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


def cands_of(r) -> list:
    return [{"idx": c["idx"], "letter": c["letter"], "ent": c["ent"], "cond": c["cond"], "text": c["text"]}
            for c in r["cands"]]


# ---------------- 候选构造 ----------------
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
    """组合空间 shuffle 后切前 need 条；shuffle 种子与 72 相同 ⇒ 对话部分与 §9 数据逐条对齐。"""
    combos = d72.build_combos()
    rng_split = random.Random(seed)
    rng_split.shuffle(combos)
    need = n_train + n_test
    if len(combos) < need:
        raise SystemExit(f"[D74] 组合空间不足：{len(combos)} < {need}")
    rng_cand = random.Random(seed ^ SEED_CAND_XOR)
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
        gold = d72.det_reply(fam["key"], i3, ent, cond)
        ctx = render_ctx(r1u, r1s, r2u, r2s, r3u, cand_line)
        fam_ents, fam_conds = set(fam["ents"]), set(fam["conds"])
        n_same_ent = sum(1 for c in cands if c["ent"] == ent and c["cond"] != cond)
        n_same_cond = sum(1 for c in cands if c["cond"] == cond and c["ent"] != ent)
        recs.append(dict(
            ctx=ctx, target=gold, gold=gold,
            family=fam["key"], ent=ent, cond=cond,
            cond_type=fam["cond_type"], ent_kind=fam["ent_kind"],
            t1_idx=i1, t2_idx=i2, t3_idx=i3,
            tpl_id=f"{fam['key']}#t3_{i3}",
            t1_user=r1u, t1_sys=r1s, t2_user=r2u, t2_sys=r2s, t3=r3u,
            n_turns=3,
            # ---- 本单元新增 ----
            cands=cands, gold_idx=gold_idx, n_cands=k,
            gold_letter=CAND_LETTERS[gold_idx], cand_line=cand_line,
            cand_line_chars=len(cand_line),
            n_distract_same_ent=n_same_ent, n_distract_same_cond=n_same_cond,
            n_cands_in_vocab=sum(1 for c in cands if c["ent"] in fam_ents and c["cond"] in fam_conds),
            # ---- §9 的计数类字段 ----
            ent_count=sum(1 for e in ALL_ENTS if e in ctx),
            cond_count=sum(1 for c in ALL_CONDS if c in ctx),
            ent_pos=ctx.index(ent),
            t3_chars=len(r3u),
            resp_chars=len(gold), resp_bucket=len(gold) // BUCKET_CHARS,
            turn3_has_ent=(ent in r3u), turn3_has_cond=(cond in r3u),
            t2_has_ent=(ent in r2u), t1_has_cond=(cond in r1u),
            ent_in_t1=(ent in r1u), cond_in_t2=(cond in r2u),
            ent_in_target=(ent in gold), cond_in_target=(cond in gold),
            ent_in_ctx=(ent in ctx), cond_in_ctx=(cond in ctx),
        ))
    return recs, combos


# ---------------- 不变量（逐条断言 + 计数，违例必须 0） ----------------
def _chi2_crit_999(dof: int) -> float:
    return CHI2_999.get(dof, dof + 5.0 * (2.0 * dof) ** 0.5)   # 表外 dof 用保守近似


def check_invariants(recs, tag="all", perm_rng=None):
    """返回 (vio Counter, info dict)。① ② ③(A/B/C/D/E) ④ 逐条判。"""
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

    # ---- 逐条 ① / B / D / E ----
    for r in recs:
        # ① 恰 1 个候选与历史一致
        match = [c for c in r["cands"] if c["ent"] == r["ent"] and c["cond"] == r["cond"]]
        one = (len(match) == 1)
        vio["①_与历史一致的候选数!=1"] += int(not one)
        g = r["cands"][r["gold_idx"]]
        vio["①_gold_idx不是唯一匹配项"] += int(not (one and g["ent"] == r["ent"] and g["cond"] == r["cond"]))
        vio["①_候选互异坏"] += int(len({(c["ent"], c["cond"]) for c in r["cands"]}) != r["n_cands"])
        vio["①_候选超出族词表"] += int(r["n_cands_in_vocab"] != r["n_cands"])
        vio["①_gold候选回复!=target"] += int(
            d72.det_reply(r["family"], r["t3_idx"], g["ent"], g["cond"]) != r["target"])
        ok["①"] += int(one)

        b = (r["ent_in_ctx"] and r["cond_in_ctx"] and r["ent_in_t1"] and r["cond_in_t2"]
             and r["ent_in_target"] and r["cond_in_target"])
        vio["B_实体或条件不在输入里"] += int(not b)
        d = (not r["turn3_has_ent"]) and (not r["turn3_has_cond"]) \
            and (not r["t2_has_ent"]) and (not r["t1_has_cond"])
        vio["D_轮3含实体或条件(含轮2含实体/轮1含条件)"] += int(not d)
        vio["E_逐条往返坏"] += int(not r.get("rt_ok", True))
        ok["B"] += int(b)
        ok["D"] += int(d)
    info["ok"] = ok

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
    se_pp = ((1.0 / k) * (1.0 - 1.0 / k) / N) ** 0.5 * 100.0   # 单格比例的 1SE（百分点）
    vio["②_位置计数为0"] = sum(1 for c in counts if c == 0)
    vio["②_卡方超0.999临界"] = int(chi2 > crit)
    info["pos"] = dict(counts=counts, exp=exp, max_dev=max_dev, chi2=chi2, dof=dof,
                       crit=crit, se_pp=se_pp)

    # ---- 静态词表级 D：任何 (族, 句式) 文本里都不许出现任何实体或条件 ----
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

    # ---- 模板级静态检查：占位符个数 / 去重 ----
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
            shape_vio += int("{" in t or "}" in t)
        for t in f["reply"]:
            shape_vio += int(t.count("{ent}") != 1 or t.count("{cond}") != 1)
        shape_vio += int(len(f["reply"]) != len(f["t3_user"]))
        for col in ("ents", "conds"):
            shape_vio += int(len(set(f[col])) != len(f[col]))
    vio["SHAPE_模板占位符或去重坏"] = shape_vio

    # ---- 可复原判定：parse_ctx 必须逐条复原出构造时的六元组 ----
    n_parse_bad = 0
    for r in recs:
        p = parse_ctx(r["ctx"])
        if p is None or p != (r["family"], r["t1_idx"], r["t2_idx"], r["t3_idx"], r["ent"], r["cond"]):
            n_parse_bad += 1
    vio["PARSE_可复原判定失败"] = n_parse_bad

    # ---- ④ 位置置换不变性 ----
    n_perm, n_perm_bad, n_perm_idx_moved = 0, 0, 0
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
                hit = [q for q in pairs if q == hist]
                idx2 = pairs.index(hist) if hit else -1
                okp = (p2 is not None
                       and (p2[0], p2[4], p2[5]) == (r["family"], r["ent"], r["cond"])
                       and len(hit) == 1
                       and d72.det_reply(p2[0], p2[3], hit[0][0], hit[0][1]) == r["target"])
                n_perm += 1
                n_perm_bad += int(not okp)
                n_perm_idx_moved += int(idx2 != r["gold_idx"])
    vio["④_置换后正确项改变"] = n_perm_bad
    info["perm"] = dict(checked=n_perm, idx_moved=n_perm_idx_moved)

    print(f"[INV:{tag}] ① 候选唯一性：通过={ok['①']}/{len(recs)} | 违例 与历史一致数!=1={vio['①_与历史一致的候选数!=1']} "
          f"gold_idx错位={vio['①_gold_idx不是唯一匹配项']} 候选互异坏={vio['①_候选互异坏']} "
          f"候选越族={vio['①_候选超出族词表']} gold候选回复!=target={vio['①_gold候选回复!=target']}", flush=True)
    print(f"[INV:{tag}] ② 位置均匀：计数={counts} 期望={exp:.1f} 最大偏差={max_dev:.2f}% "
          f"卡方={chi2:.3f}(dof={dof} 0.999临界={crit:.3f} 单格1SE={se_pp:.2f}pp) "
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
          f"逐条往返坏={vio['E_逐条往返坏']}", flush=True)
    print(f"[INV:{tag}] ③PARSE 可复原判定（族+三轮句式 index+实体+条件 六元组复原）失败={n_parse_bad} "
          f"| SHAPE 模板占位符静态违例={shape_vio}", flush=True)
    print(f"[INV:{tag}] ④ 位置置换不变性：检查={n_perm}（逆序+随机 各一遍）违例={n_perm_bad} "
          f"| 置换后正确项位置变了的次数={n_perm_idx_moved}（位置变、指代不变才算过）", flush=True)
    return vio, info


# ---------------- 地板（L5） ----------------
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
    """总是选第 p 个候选（位置基准，机会级地板）。"""
    if p >= r["n_cands"]:
        return []
    c = r["cands"][p]
    return [d72.det_reply(fk, i3, c["ent"], c["cond"]) for fk, i3 in _infer_fam_i3(r["t3"])]


def _bucket_specs():
    return (
        ("⑥a回复长度桶(20字)", lambda r: r["resp_bucket"]),
        ("⑥b轮数桶", lambda r: r["n_turns"]),
        ("⑥c实体个数桶", lambda r: r["ent_count"]),
        ("⑥d条件类型桶", lambda r: r["cond_type"]),
        ("⑥e轮3句式桶(族×句式)", lambda r: (r["family"], r["t3_idx"])),
        ("⑥f轮3长度桶(5字)", lambda r: r["t3_chars"] // T3_BUCKET_CHARS),
        ("⑥g实体位置桶(10字)", lambda r: r["ent_pos"] // ENT_POS_BUCKET),
        ("⑥h条件个数桶", lambda r: r["cond_count"]),
        ("⑥i候选数桶", lambda r: r["n_cands"]),
        ("⑥j候选行长度桶(20字)", lambda r: r["cand_line_chars"] // BUCKET_CHARS),
    )


def _stats(train):
    st = {"maj_target": Counter(r["target"] for r in train).most_common(1)[0][0],
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
    """⑥ 强朴素规则：只用轮3 认出 (族, 句式)，实体/条件一律填 train 族内众数（不看历史）。"""
    return [d72.det_reply(fk, i3, s["maj_ent"][fk], s["maj_cond"][fk])
            for fk, i3 in _infer_fam_i3(r["t3"])]


def _pred_reference(r, s):
    """参考上限（非地板）：按历史匹配候选（完美选择）+ DET 句面。"""
    p = parse_ctx(r["ctx"])
    if p is None:
        s["miss"] += 1
        return ""
    fk, _i1, _i2, i3, ent, cond = p
    hit = [c for c in r["cands"] if c["ent"] == ent and c["cond"] == cond]
    if len(hit) != 1:
        s["sel_bad"] += 1
        return ""
    return d72.det_reply(fk, i3, hit[0]["ent"], hit[0]["cond"])


def _rule_preds():
    """每条规则返回**预测文本列表**（多条 = 命中任一即算中，对地板更保守）。"""
    rules = [
        ("②总选第一个候选", lambda r, s: _pred_position(r, 0)),
        ("③总选最后一个候选", lambda r, s: _pred_position(r, r["n_cands"] - 1)),
        ("④复制轮3用户话", lambda r, s: [r["t3"]]),
        ("④b复制轮1用户话", lambda r, s: [r["t1_user"]]),
        ("④c复制轮1系统话", lambda r, s: [r["t1_sys"]]),
        ("④d复制轮2用户话", lambda r, s: [r["t2_user"]]),
        ("④e复制轮2系统话", lambda r, s: [r["t2_sys"]]),
        ("⑤train全集众数", lambda r, s: [s["maj_target"]]),
        ("⑥模板填空只用轮3(实体/条件填train族内众数)", _pred_full_ctx_fill),
    ]
    for name, keyfn in _bucket_specs():
        def mk(keyfn=keyfn, name=name):
            def f(r, s):
                return [s["bucket"][name].get(keyfn(r), s["maj_target"])]
            return f
        rules.append((name, mk()))
    return rules


CHANCE_NAMES = ("①随机猜1/k", "②总选第一个候选", "③总选最后一个候选")
SHORTCUT_NAMES = tuple(n for n, _ in _rule_preds() if n not in CHANCE_NAMES)
REFERENCE_NAME = "参考:完美选择(按历史匹配候选)+DET句面"
assert set(CHANCE_NAMES) | set(SHORTCUT_NAMES) == {n for n, _ in _rule_preds()} | {"①随机猜1/k"}


def compute_floors(train, test, tag):
    """在 test 上逐条算 EM；机会级与捷径级分开汇总（避免把 1/k 的机会水平混进 max_naive）。"""
    st = _stats(train)
    rng = random.Random(SEED ^ SEED_FLOOR_XOR)

    def _pred_random(r, s):
        return _pred_position(r, rng.randrange(r["n_cands"]))

    rules = [("①随机猜1/k", _pred_random)] + _rule_preds()

    def eval_rules(rs):
        rows = {}
        for name, fn in rules:
            hit = 0
            for r in rs:
                preds = fn(r, st) or [""]
                hit += int(any(_em(r["target"], p) for p in preds))
            rows[name] = 100.0 * hit / len(rs)
        return rows

    rows = eval_rules(test)
    for name, _fn in rules:
        print(f"[FLOOR] {tag} {name} EM={rows[name]:.2f}% (n={len(test)})", flush=True)

    # 逐位置扫描（②③ 的一般化，同时是位置均匀性的旁证）
    k = test[0]["n_cands"]
    scan = []
    for p in range(k):
        hit = sum(int(any(_em(r["target"], x) for x in (_pred_position(r, p) or [""]))) for r in test)
        scan.append(100.0 * hit / len(test))
    print(f"[FLOOR] {tag} 位置扫描 always_p EM=" +
          " ".join(f"p{i}={scan[i]:.2f}%" for i in range(k)) +
          f" | min={min(scan):.2f}% max={max(scan):.2f}% 解析机会水平=100/k={100.0 / k:.2f}%", flush=True)

    # 选择准确率（只看选中项对不对，不看句面）
    ref_sel = 100.0 * sum(
        int(any(c["ent"] == r["ent"] and c["cond"] == r["cond"] for c in r["cands"])) for r in test) / len(test)
    rng2 = random.Random(SEED ^ SEED_FLOOR_XOR)
    picks = {"①随机猜": [r["cands"][rng2.randrange(r["n_cands"])] for r in test],
             "②总选第一": [r["cands"][0] for r in test],
             "③总选最后": [r["cands"][-1] for r in test]}
    sel_line = " ".join(
        f"{nm}={100.0 * sum(int(c['ent'] == r['ent'] and c['cond'] == r['cond']) for c, r in zip(cs, test)) / len(test):.2f}%"
        for nm, cs in picks.items())
    print(f"[FLOOR] {tag} 选择准确率（选对候选，不看句面）：{sel_line} "
          f"参考(按历史匹配)={ref_sel:.2f}% | 解析机会水平=100/k={100.0 / k:.2f}%", flush=True)

    # 参考上限
    s_ref = {"miss": 0, "sel_bad": 0}
    ref_hit = sum(int(_em(r["target"], _pred_reference(r, s_ref))) for r in test)
    ref_em = 100.0 * ref_hit / len(test)
    print(f"[FLOOR] {tag} {REFERENCE_NAME} EM={ref_em:.2f}% (n={len(test)}) "
          f"识别不出ctx={s_ref['miss']} 唯一匹配候选数!=1 的次数={s_ref['sel_bad']} ⇒ 信息全在输入里（非地板）", flush=True)

    # 分类汇总
    chance = {k2: v for k2, v in rows.items() if k2 in CHANCE_NAMES}
    shortcut = {k2: v for k2, v in rows.items() if k2 in SHORTCUT_NAMES}
    mc_name = max(chance, key=chance.get)
    ms_name = max(shortcut, key=shortcut.get)
    mx = shortcut[ms_name]
    se2 = 2.0 * (mx / 100.0 * (1 - mx / 100.0) / len(test)) ** 0.5 * 100.0
    print(f"[FLOOR] {tag} 机会级 max_chance={chance[mc_name]:.2f}% 来自={mc_name} "
          f"（构造期望=100/k={100.0 / k:.2f}%；这三个才是「地板≈25%」的那一类）", flush=True)
    print(f"[FLOOR] {tag} 捷径级 max_naive={mx:.2f}% 来自={ms_name} | 2SE={se2:.2f}pp ⇒ "
          f"R3-a 门槛=max_naive+2SE={mx + se2:.2f}% | max_naive<=5%: "
          f"{'PASS' if mx <= 5.0 else 'FAIL'}", flush=True)
    mx_all_name = max(rows, key=rows.get)
    print(f"[FLOOR] {tag} 全部规则（含机会级）max_all={rows[mx_all_name]:.2f}% 来自={mx_all_name}"
          f"（★ 这个数受 1/k 机会水平支配，不是退化判据；退化判据看上面的捷径级 max_naive）", flush=True)

    # train 侧同口径（判数据是否退化：捷径规则在 train 上 >= 90% ⇒ 数据无效）
    rows_train = eval_rules(train)
    tr_name = max(shortcut, key=lambda x: rows_train[x])
    print(f"[FLOOR] {tag} train 捷径级 max_naive={rows_train[tr_name]:.2f}% 来自={tr_name} ⇒ "
          + ("★数据无效（捷径规则在 train 上>=90%）" if rows_train[tr_name] >= 90.0
             else "有效（无捷径规则在 train 上>=90%）"), flush=True)
    return rows, ms_name, mx, ref_em, rows_train, chance, scan


# ---------------- 与 §9 数据的对齐核验（有 72 产物才算） ----------------
def check_align_72(recs, n_train):
    out = {}
    for split, rs in (("train", recs[:n_train]), ("test", recs[n_train:])):
        fp = os.path.join(LOG_DIR, f"72_dialog_{split}.jsonl")
        if not os.path.exists(fp):
            out[split] = None
            continue
        old = [json.loads(ln) for ln in open(fp, encoding="utf-8") if ln.strip()]
        same = sum(1 for a, b in zip(rs, old)
                   if "\n".join(a["ctx"].split("\n")[:5]) + "\n" + RESP_PREFIX == b["ctx"])
        out[split] = (same, len(rs), len(old))
    return out


# ---------------- 入口 ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="S74 L3-r3a 带候选集的多轮对话数据生成器 + 地板")
    ap.add_argument("--k", type=int, default=4, help="候选数（只接受 4 / 8：文件名 74_/78_ 写死）")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--smoke", action="store_true", help="小量冒烟（train 200 / test 100，写 74s_/78s_）")
    a = ap.parse_args(argv)

    if a.k not in PREFIX_BY_K:
        raise SystemExit(f"[D74] 只接受 --k in {sorted(PREFIX_BY_K)}（产物名 74_/78_ 写死），收到 {a.k}")
    prefix = PREFIX_BY_K[a.k] + ("s" if a.smoke else "")
    if a.smoke:
        a.n_train, a.n_test = N_SMOKE_TRAIN, N_SMOKE_TEST

    src_sha = hashlib.sha256(open(D72_PATH, "rb").read()).hexdigest()[:16]
    print(f"[SRC] 72_nl_dialog_gen.py sha256[:16]={src_sha}（族定义/DET 回复/指纹全部复用该文件）", flush=True)
    print(f"[CAND-FMT] 候选行文本格式：CAND_PREFIX={CAND_PREFIX!r} CAND_SEP={CAND_SEP!r} "
          f"CAND_ITEM={CAND_ITEM!r} 字母=CAND_LETTERS[:k]", flush=True)
    for kk in sorted(PREFIX_BY_K):
        fams = FAMILIES[1]                                   # 「预约」族做格式样例
        pairs = [(fams["ents"][i], fams["conds"][(i * 3) % len(fams["conds"])]) for i in range(kk)]
        print(f"[CAND-FMT] 样例 k={kk}：{render_cand_line(pairs)}", flush=True)

    recs, combos = gen_records(a.seed, a.n_train, a.n_test, a.k)
    train, test = recs[:a.n_train], recs[a.n_train:]
    print(f"[SPLIT] k={a.k} 组合空间={len(combos)}（无放回抽样后 train={len(train)} test={len(test)}）"
          f" | 族={N_FAMILIES} 轮1/2/3句式和="
          + " ".join(f"{f['key']}[{len(f['t1_user'])}/{len(f['t2_user'])}/{len(f['t3_user'])}]"
                     for f in FAMILIES), flush=True)
    print(f"[TPL] DET 句面表（回复模板 = DET(族, 轮3句式 index)，回复模板总数={N_REPLY_TEMPLATES}）：", flush=True)
    for f in FAMILIES:
        for i3, tpl in enumerate(f["reply"]):
            print(f"[TPL] {f['key']}#t3_{i3} 轮3={f['t3_user'][i3]!r} ⇒ {tpl}", flush=True)
    print("[VOCAB] " + " | ".join(
        f"{f['key']} 实体[{f['ent_kind']}]={len(f['ents'])} 条件[{f['cond_type']}]={len(f['conds'])}"
        for f in FAMILIES), flush=True)
    al = check_align_72(recs, a.n_train)
    for split, v in al.items():
        print(f"[ALIGN] 与 logs/72_dialog_{split}.jsonl 的对话部分逐条一致："
              + ("缺 72 产物，跳过" if v is None else f"{v[0]}/{v[1]}（72 侧 {v[2]} 条）"), flush=True)

    # ---- tokenizer：逐字往返（③E）+ 逐条编码 + 指纹 ----
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    V = tok.get_vocab_size()
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids            # noqa: E731
    dec = lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=False)  # noqa: E731
    EOS_ID = tok.token_to_id(EOS)
    UNK_ID = tok.token_to_id("<unk>")
    assert enc(RESP_PREFIX) == [140] and enc(EOS) == [EOS_ID], (enc(RESP_PREFIX), enc(EOS))

    chars = sorted({c for r in recs for s in (r["ctx"], r["target"]) for c in s})
    bad_chars = [(c, enc(c), dec(enc(c))) for c in chars if dec(enc(c)) != c]
    print(f"[E-CHECK] 数据中不同字符数={len(chars)} 逐字往返坏字符={len(bad_chars)} "
          f"{bad_chars} | V={V} EOS={EOS_ID} UNK={UNK_ID}", flush=True)

    for r in recs:
        r["p"] = enc(r["ctx"])
        r["t"] = enc(r["target"] + EOS)
        r["t3_tok"] = len(enc(r["t3"]))
        r["ctx_tok"] = len(r["p"])
        r["resp_tok"] = len(r["t"])
        r["unk_in_ctx"] = UNK_ID in r["p"]
        r["rt_ok"] = (dec(r["p"]) == r["ctx"] and dec(r["t"]) == r["target"] + EOS)

    n_unk = sum(1 for r in recs if r["unk_in_ctx"])
    print(f"[E-CHECK] 含 <unk> 的 ctx 条数={n_unk}/{len(recs)}（必须 0）", flush=True)
    assert n_unk == 0, "ctx 里出现 <unk> ⇒ 有字符不在词表里"
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
    info["bad_chars"] = [c for c, _i, _b in bad_chars]
    total_vio = sum(vio.values())

    pos_te = Counter(r["gold_idx"] for r in test)
    print(f"[DIST] 族分布={dict(Counter(r['family'] for r in recs))} | "
          f"条件类型分布={dict(Counter(r['cond_type'] for r in recs))} | "
          f"轮3句式分布={dict(sorted(Counter(r['tpl_id'] for r in recs).items()))} | "
          f"候选数分布={dict(Counter(r['n_cands'] for r in recs))} | "
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

    # ---- 地板 ----
    rows, ms_name, mx, ref_em, rows_train, chance, scan = compute_floors(train, test, tag="test")

    # ---- 三条 [GEN] 样例（完整 ctx → gold，人工核「候选列表里确实只有一个与历史一致」）----
    picked, seen_fam = [], set()
    for r in test:
        if r["family"] not in seen_fam:
            seen_fam.add(r["family"])
            picked.append(r)
        if len(picked) == 3:
            break
    for k, r in enumerate(picked, 1):
        print(f"[GEN{k}] family={r['family']} ent={r['ent']!r} cond={r['cond']!r} "
              f"cond_type={r['cond_type']} tpl_id={r['tpl_id']} n_cands={r['n_cands']} "
              f"gold_idx={r['gold_idx']} gold_letter={r['gold_letter']}", flush=True)
        for i, ln in enumerate(r["ctx"].split("\n"), 1):
            print(f"[GEN{k}] ctx{i}| {ln}", flush=True)
        for c in r["cands"]:
            em_ = c["ent"] == r["ent"]
            cm_ = c["cond"] == r["cond"]
            print(f"[GEN{k}]   候选 {c['letter']}({c['ent']} / {c['cond']}) "
                  f"实体对上={em_} 条件对上={cm_} 两个都对上={em_ and cm_}", flush=True)
        print(f"[GEN{k}] gold：{r['gold']}  (gold含实体={r['ent_in_target']} gold含条件={r['cond_in_target']})",
              flush=True)

    # ---- [D74] 一行汇总 ----
    pos = info["pos"]
    print(f"[D74] k={a.k} n_train={len(train)} n_test={len(test)} 池={len(combos)}(组合空间) "
          f"seed={a.seed} prefix={prefix} | 族={N_FAMILIES} 轮3句式/族={[len(f['t3_user']) for f in FAMILIES]} "
          f"回复模板={N_REPLY_TEMPLATES} | "
          f"①违例={vio['①_与历史一致的候选数!=1'] + vio['①_gold_idx不是唯一匹配项'] + vio['①_候选互异坏'] + vio['①_候选超出族词表'] + vio['①_gold候选回复!=target']}"
          f"(一致数!=1:{vio['①_与历史一致的候选数!=1']} gold_idx错位:{vio['①_gold_idx不是唯一匹配项']} "
          f"互异坏:{vio['①_候选互异坏']} 越族:{vio['①_候选超出族词表']} gold回复!=target:{vio['①_gold候选回复!=target']}) "
          f"②违例={vio['②_位置计数为0'] + vio['②_卡方超0.999临界']}(位置计数为0:{vio['②_位置计数为0']} "
          f"卡方超临界:{vio['②_卡方超0.999临界']}) 位置计数={pos['counts']} 期望={pos['exp']:.1f} "
          f"最大偏差={pos['max_dev']:.2f}% 卡方={pos['chi2']:.3f}/dof{pos['dof']}(临界{pos['crit']:.3f}) "
          f"③A/B/C/D/E违例={vio['A_同ctx不同target']}/{vio['B_实体或条件不在输入里']}/"
          f"{vio['C_唯一ctx比不足']}/{vio['D_轮3含实体或条件(含轮2含实体/轮1含条件)']}/"
          f"{vio['E_坏字符'] + vio['E_逐条往返坏']}(唯一ctx比={info['uniq_ctx_ratio']:.4f}) "
          f"④置换违例={vio['④_置换后正确项改变']}(检查{info['perm']['checked']}次) | "
          f"train_fp={tr_fp} test_fp={te_fp} | 参考上限={ref_em:.2f}%", flush=True)
    r3a = (vio["①_与历史一致的候选数!=1"] == 0 and vio["①_gold_idx不是唯一匹配项"] == 0
           and vio["①_候选互异坏"] == 0 and vio["①_候选超出族词表"] == 0
           and vio["①_gold候选回复!=target"] == 0
           and vio["②_位置计数为0"] == 0 and vio["②_卡方超0.999临界"] == 0
           and vio["A_同ctx不同target"] == 0 and vio["B_实体或条件不在输入里"] == 0
           and vio["C_唯一ctx比不足"] == 0 and vio["D_轮3含实体或条件(含轮2含实体/轮1含条件)"] == 0
           and vio["E_坏字符"] == 0 and vio["E_逐条往返坏"] == 0
           and vio["④_置换后正确项改变"] == 0
           and mx <= 5.0)
    print(f"[D74-VERDICT] R3-a={'PASS' if r3a else 'FAIL'}"
          f"（不变量违例{'全0' if total_vio == 0 else '>0'}；捷径级 max_naive={mx:.2f}% 来自={ms_name} "
          f"目标<=5%；机会级 max_chance={chance[max(chance, key=chance.get)]:.2f}% "
          f"期望100/k={100.0 / a.k:.2f}%；位置最大偏差={pos['max_dev']:.2f}%）"
          + ("" if total_vio == 0 else " | ★违例>0 ⇒ 数据无效，停。"), flush=True)
    print(f"[DONE] exit={0 if total_vio == 0 else 4}", flush=True)
    return 0 if total_vio == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
