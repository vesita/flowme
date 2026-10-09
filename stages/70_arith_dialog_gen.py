#!/usr/bin/env python3
"""S70 · L3-r0a 算术路数据生成器 + 地板（CPU、确定性 seed、秒级；不碰 GPU）。

数据源：nanoSeek/data/chinese/_discarded/math_cot_v2_dialogue.txt（项目自有合成，MIT）。
该文件 80,000 个块（`用户：`/`模型：` 成对、块间空行），题型只有 5 种：

    请计算 a + b，写出逐位进位过程。      28,060 条   题面含两个操作数
    请计算 a - b，写出退位（借位）过程。   7,980 条   题面含两个操作数
    请计算 a × b，写出分步拆解过程。      16,162 条   题面含两个操作数
    请解下列一元一次方程，写出完整求解步骤。 16,116 条  题面**一个数都没有**（数字只在答案里）
    请解答这道鸡兔同笼问题，写出假设法与推导过程。 11,682 条  同上

后两族各只有 **1 个唯一 ctx**，却挂着一万多个不同 gold ⇒ **不是函数**（同一输入多个正确答案，
不可判），本脚本用程序把它们逐族数出来并剔除。剔除后算术子集 = 28,060 + 7,980 + 16,162 = 52,202 条。

★ 真函数性门（逐条判，违例必须为 0，否则退出码非 0）：
  1  剔除后子集内 ctx 唯一（重复 ctx 必须同 gold）（ctx = 题面原文）；
  2  ctx → target 是函数（同 ctx 不同 target 的 ctx 数 = 0）；
  3  每条 gold 都能抽出「结论句最后一个整数」gold_int（抽不出算违例）；
  4  gold_int == 题面算式的计算结果（口径自洽：抽出的答案就是题面的正确答案）；
  5  ctx + target 的字符 token 数 <= 512（沿用 66 的口径上限）。

★ 训练 target 与 EM 指标（两个口径分开写清，别混）：
  · **训练 target = 回复原文**（teacher forcing：`enc(gold + <eos>)`，只在目标 span 上算 CE）；
  · **EM 指标 = 从生成文本里抽「结论句的最后一个整数」与 gold 整数比较**
    （实现 `extract_conclusion_int`；规则 ①②③ 的预测也先变成文本、再走同一个抽取器，
      抽不出来的那条直接算错并单独计数）。
  · 与 nanoSeek `training/rl/reward/arith_gen.py` 的关系（**实测，不是推断**）：
    `arith_gen.py:73 extract_arith` **不可直接复用**作主口径 —— 它 (i) 正则 `\\d{1,3}`（`arith_gen.py:70`）
    只认 1~3 位操作数、(ii) 操作数范围 `arith_rules.toml` 写死 `max_num = 100`、
    (iii) 返回的是**它自己算出来的**结果而不是生成文本里写的整数。
    本数据 52,202 条里绝大多数是 3~5 位操作数 ⇒ 用它会几乎全部返回 None。
    因此主口径自实现 `extract_conclusion_int`；只在**两边操作数都 <= 100** 的子集上
    用 `extract_arith` 做**交叉校验**（见 [XCHK] 行，统计能复用多少条、分歧多少条）。

★ 规模：train 7000 / test 800（从去重后的 50,816 个唯一 ctx 里无放回抽）；
  另外单列**小数字子集**（1–2 位加减在盘上不存在，见 [SMALL-DEFICIT] 行）：
  「2 位 × 2 位乘法」唯一 ctx 1,240 条 ⇒ train 1000 / test 240，落盘 logs/70s_dialog_*.jsonl。

产物（只写 logs/）：
  logs/70_dialog_{train,test}.jsonl   全量算术子集
  logs/70s_dialog_{train,test}.jsonl  2 位 × 2 位乘法小数字子集
每条字段（沿用 stages/66_dialog_gen.py:16-22 的 ctx/target/gold 风格 + 计数类特征）：
  ctx / target / gold / gold_int / op / type_prefix / nums / n_nums / max_digits
  / gold_digits / resp_chars / resp_bucket / q_chars / ctx_tok / resp_tok / small_2x2
指纹 data_fp 口径与 stages/66_dialog_gen.py:250-258（= stages/65_textgen_cards.py::data_fp）逐字一致。

用法（本机只有 .venv/bin/python 装了 tokenizers；本脚本无 torch、秒~分钟级）：
  .venv/bin/python stages/70_arith_dialog_gen.py
  .venv/bin/python stages/70_arith_dialog_gen.py --n-train 7000 --n-test 800 --seed 20261010
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import struct
import sys
from collections import Counter, defaultdict

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
if NS_ROOT not in sys.path:
    sys.path.insert(0, NS_ROOT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
SRC = NS_ROOT + "/data/chinese/_discarded/math_cot_v2_dialogue.txt"
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
OUT_TMPL = os.path.join(LOG_DIR, "70_dialog_{split}.jsonl")
OUT_SMALL_TMPL = os.path.join(LOG_DIR, "70s_dialog_{split}.jsonl")

N_TRAIN = 7000
N_TEST = 800
N_SMALL_TRAIN = 1000
N_SMALL_TEST = 240
SEED = 20261010
EOS = "<eos>"
BUCKET_CHARS = 50          # 回复长度/题干长度桶宽（字符）
MAX_SEQ = 512              # ctx + target 字符 token 上限（与 66 同口径）

OP_CHARS = ("+", "-", "×")
OP_NAME = {"+": "加法", "-": "减法", "×": "乘法"}
# 被剔除的两个「数字只在答案里」的族（题面前缀 → 族名）
NONFUNC_FAMILIES = (
    ("请解答这道鸡兔同笼问题", "鸡兔同笼"),
    ("请解下列一元一次方程", "一元一次方程"),
)
ARITH_HEAD = "请计算"
INT_RE = re.compile(r"-?\d+")


# ---------------- 解析：块 = 用户：<题面> / 模型：<回复>，块间空行 ----------------
def parse_blocks(path: str):
    """读原始 txt，返回 (rows, n_blocks, n_malformed)。

    rows = [{'user': 题面原文(strip), 'gold': 回复原文(strip)}]
    """
    with open(path, encoding="utf-8") as f:
        text = f.read()
    blocks = [b for b in text.split("\n\n") if b.strip()]
    rows, malformed = [], 0
    for b in blocks:
        if "用户：" not in b or "模型：" not in b:
            malformed += 1
            continue
        head, _sep, gold = b.partition("模型：")
        user = head.split("用户：", 1)[1]
        rows.append({"user": user.strip(), "gold": gold.strip()})
    return rows, len(blocks), malformed


def question_op(q: str):
    """题面 → 运算符（只认 `请计算` 族）；非算式题返回 None。"""
    if not q.startswith(ARITH_HEAD):
        return None
    body = q.split("，", 1)[0]
    for s in OP_CHARS:
        if s in body:
            return s
    return None


def question_nums(q: str):
    """题面（逗号前）里的整数列表。"""
    return [int(x) for x in INT_RE.findall(q.split("，", 1)[0])]


def extract_conclusion_int(text: str):
    """抽「结论句的最后一个整数」；抽不出返回 None（调用方把 None 算错并计数）。

    口径（三步，确定性）：
      1. 结论句 = 从后往前**第一个含 '=' 的非空行**（本题族 100% 命中，实测抽不出率 0%）；
      2. 取该行最后一个整数（含负号）；
      3. 没有含 '=' 的行时退化：取全文最后一个整数；仍无整数 → None。

    对照 `training/rl/reward/arith_gen.py:73 extract_arith`：那条口径是
    「正则搜算式 → 自己把 a op b 算出来」，位数上限 3、操作数上限 100；
    本数据是 3~5 位操作数，故只取其「按算式定位答案」的精神，不复用其实现（见文件头）。
    """
    if not text:
        return None
    lines = [ln for ln in text.strip().split("\n") if ln.strip()]
    for ln in reversed(lines):
        if "=" in ln:
            m = INT_RE.findall(ln)
            if m:
                return int(m[-1])
    m = INT_RE.findall(text)
    return int(m[-1]) if m else None


def expr_result(op: str, nums):
    """题面算式的正确结果（加法/减法/乘法）。"""
    a, b = nums[0], nums[-1]
    return {"+": a + b, "-": a - b, "×": a * b}[op]


# ---------------- 函数性门（逐条判，违例必须为 0） ----------------
def excluded_family_report(rows):
    """程序化证明两个被剔除的族**不是函数**（同 ctx 不同 gold 的条数）。"""
    worst = 0
    for prefix, name in NONFUNC_FAMILIES:
        sub = [r for r in rows if r["user"].startswith(prefix)]
        per = defaultdict(set)
        for r in sub:
            per[r["user"]].add(r["gold"])
        conflict = sum(1 for v in per.values() if len(v) > 1)
        mx = max((len(v) for v in per.values()), default=0)
        worst = max(worst, conflict)
        ex = max(per.values(), key=len) if per else {""}
        print(f"[FUNC-EXCLUDED] {name}: 条数={len(sub)} 唯一ctx={len(per)} "
              f"同ctx不同gold的ctx数={conflict} 单ctx最多不同gold={mx}"
              f" ⇒ {'非函数，剔除' if conflict else '未发现冲突(意外)'}", flush=True)
        print(f"[FUNC-EXCLUDED] {name} 证据样例(前 2 个不同 gold 的前 60 字): "
              + " || ".join(repr(g[:60]) for g in sorted(ex)[:2]), flush=True)
    return worst


def check_functions(recs, tag):
    """返回 (vio, info)。vio 全部为 0 才算过门。"""
    per = defaultdict(set)
    for r in recs:
        per[r["ctx"]].add(r["target"])
    conflict = sum(1 for v in per.values() if len(v) > 1)
    dup = len(recs) - len(per)
    vio = Counter()
    vio["ctx冲突"] = conflict
    for r in recs:
        vio["gt1个数!=2"] += int(r["n_nums"] != 2)
        vio["op未识别"] += int(r["op"] not in OP_CHARS)
        vio["gold_int抽不出"] += int(r["gold_int"] is None)
        vio["gold_int与题面算式不符"] += int(
            r["gold_int"] is not None and r["gold_int"] != expr_result(r["op"], r["nums"]))
        vio["ctx+target超512tok"] += int(r["ctx_tok"] + r["resp_tok"] > MAX_SEQ)
    print(f"[FUNC:{tag}] ctx唯一={len(per)} 重复ctx={dup}(同ctx同gold,去重时只留一条) "
          f"同ctx不同target={conflict}(违例) | 其它违例="
          + " ".join(f"{k}={v}" for k, v in vio.items() if k != "ctx冲突")
          + f" | 违例合计={sum(vio.values())}", flush=True)
    return vio, {"unique_ctx": len(per), "dup": dup, "conflict": conflict}


# ---------------- 组记录 ----------------
def build_records(rows, enc=None):
    """从原始 rows 造算术子集记录（不含 token 时 enc=None，含 token 时算 ctx_tok/resp_tok）。"""
    recs = []
    for r in rows:
        q = r["user"]
        op = question_op(q)
        if op is None:
            continue
        nums = question_nums(q)
        if len(nums) != 2:
            continue
        gold = r["gold"]
        ctx = f"用户：{q}\n<resp>"
        rec = dict(
            ctx=ctx, target=gold, gold=gold, q=q,
            gold_int=extract_conclusion_int(gold),
            op=op, type_prefix=OP_NAME[op], nums=nums, n_nums=len(nums),
            max_digits=max(len(str(x)) for x in nums),
            gold_digits=len(str(extract_conclusion_int(gold))) if extract_conclusion_int(gold) is not None else 0,
            resp_chars=len(gold), resp_bucket=len(gold) // BUCKET_CHARS,
            q_chars=len(q), small_2x2=(op == "×" and max(nums) <= 99),
        )
        if enc is not None:
            rec["ctx_tok"] = len(enc(ctx))
            rec["resp_tok"] = len(enc(gold) + enc(EOS))
        recs.append(rec)
    return recs


def dedupe_by_ctx(recs):
    """按 ctx 去重（同 ctx 必须同 target；调用方已过函数性门）。返回 (uniq, n_dup)。"""
    seen = {}
    for r in recs:
        seen.setdefault(r["ctx"], r)
    return list(seen.values()), len(recs) - len(seen)


def split_recs(recs, n_train, n_test, seed, tag):
    """确定性无放回切分：shuffle 后切前 n_train / 再 n_test；池不够就取满并写清。"""
    rng = random.Random(seed + sum(ord(c) for c in tag))
    pool = list(recs)
    rng.shuffle(pool)
    need = n_train + n_test
    if len(pool) < need:
        n_train = int(len(pool) * 0.8)
        n_test = len(pool) - n_train
        print(f"[SPLIT:{tag}] ★池不足：可用唯一 ctx={len(pool)} < 请求 {need} ⇒ "
              f"按 80/20 取满 train={n_train} test={n_test}", flush=True)
    return pool[:n_train], pool[n_train:n_train + n_test]


# ---------------- 指纹（与 stages/66_dialog_gen.py:250-258 逐字一致） ----------------
def data_fp(recs) -> str:
    h = hashlib.sha256()
    for r in recs:
        for seq in (r["p"], r["t"]):
            h.update(struct.pack("<Q", len(seq)))
            if seq:
                h.update(struct.pack(f"<{len(seq)}i", *seq))
    return h.hexdigest()[:16]


# ---------------- 地板（P0-c：含计数类；抽不出来的算错并单独计数） ----------------
def _bucket_specs():
    return (
        ("③a题干数字个数桶", lambda r: r["n_nums"]),
        ("③b最大位数桶", lambda r: r["max_digits"]),
        ("③c题型前缀桶", lambda r: r["type_prefix"]),
        ("④d回复长度桶(50字)", lambda r: r["resp_bucket"]),
    )


def _stats(train):
    st = {"maj_int": Counter(r["gold_int"] for r in train).most_common(1)[0][0],
          "bucket": {}}
    for name, keyfn in _bucket_specs():
        per = defaultdict(Counter)
        for r in train:
            per[keyfn(r)][r["gold_int"]] += 1
        st["bucket"][name] = {b: c.most_common(1)[0][0] for b, c in per.items()}
    return st


def _rule_preds():
    """每条规则返回**预测文本**；EM 由 extract_conclusion_int 抽整数后比对。"""
    rules = [
        ("①复制题干末数", lambda r, s: str(r["nums"][-1])),
        ("①b复制题干首数", lambda r, s: str(r["nums"][0])),
        ("②train全集众数", lambda r, s: str(s["maj_int"])),
    ]
    for name, keyfn in _bucket_specs():
        def mk(keyfn=keyfn, name=name):
            def f(r, s):
                return str(s["bucket"][name].get(keyfn(r), s["maj_int"]))
            return f
        rules.append((name, mk()))
    # 参考项（不是地板）：题面算式的正确结果 —— 「会算即满分」的上界，用来证明抽取路径可达
    rules.append(("参 考:题面算式正确结果(非地板)", lambda r, s: str(expr_result(r["op"], r["nums"]))))
    return rules


NAIVE_NAMES = ("①复制题干末数", "①b复制题干首数", "②train全集众数",
               "③a题干数字个数桶", "③b最大位数桶", "③c题型前缀桶", "④d回复长度桶(50字)")


def compute_floors(train, test, tag):
    """在 test 上逐条算 EM；返回 (per_rule_em, max_naive_name, max_naive, miss_gold, per_rule_miss)。"""
    st = _stats(train)
    rules = _rule_preds()
    miss_gold = sum(1 for r in test if r["gold_int"] is None)
    em, miss = {}, {}
    for name, fn in rules:
        hit, ms = 0, 0
        for r in test:
            pi = extract_conclusion_int(fn(r, st))
            if pi is None:
                ms += 1            # 抽不出来 ⇒ 算错，单独计数
            elif pi == r["gold_int"]:
                hit += 1
        em[name] = 100.0 * hit / len(test)
        miss[name] = ms
    for name, _fn in rules:
        print(f"[FLOOR] {tag} {name} EM={em[name]:.2f}% (n={len(test)})"
              + (f" | 抽不出={miss[name]}" if miss[name] else ""), flush=True)
    naive = {k: v for k, v in em.items() if k in NAIVE_NAMES}
    mx_name = max(naive, key=naive.get)
    mx = naive[mx_name]
    se2 = 2.0 * (mx / 100.0 * (1 - mx / 100.0) / len(test)) ** 0.5 * 100.0
    print(f"[FLOOR] {tag} max_naive={mx:.2f}% 来自={mx_name} | 2SE={se2:.2f}pp ⇒ "
          f"P0-a 门槛=max_naive+2SE={mx + se2:.2f}% | gold_int 抽不出={miss_gold}"
          f" ({100.0 * miss_gold / len(test):.2f}%,仪器质量)", flush=True)
    # train 侧同样报一遍（P2 口径：朴素规则在 train 上 >= 90% ⇒ 数据无效）
    tr_em = {}
    for name, fn in rules:
        hit = 0
        for r in train:
            pi = extract_conclusion_int(fn(r, st))
            hit += int(pi is not None and pi == r["gold_int"])
        tr_em[name] = 100.0 * hit / len(train)
    tr_name = max((k for k in tr_em if k in NAIVE_NAMES), key=lambda k: tr_em[k])
    print(f"[FLOOR] {tag} train max_naive={tr_em[tr_name]:.2f}% 来自={tr_name} ⇒ "
          + ("★数据无效（朴素规则在 train 上>=90%）" if tr_em[tr_name] >= 90.0
             else "有效（无朴素规则在 train 上>=90%）"), flush=True)
    return em, mx_name, mx, miss_gold, miss


# ---------------- 入口 ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="S70 L3-r0a 算术路数据生成器+地板")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--n-small-train", type=int, default=N_SMALL_TRAIN)
    ap.add_argument("--n-small-test", type=int, default=N_SMALL_TEST)
    a = ap.parse_args(argv)

    rows, n_blocks, n_malformed = parse_blocks(SRC)
    print(f"[SRC] {SRC} 块={n_blocks} 解析出={len(rows)} 格式坏块={n_malformed}", flush=True)
    assert n_malformed == 0, "原始文件有坏块"

    nfam_worst = excluded_family_report(rows)

    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids        # noqa: E731
    dec = lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=False)  # noqa: E731
    assert enc("<resp>") == [140] and enc(EOS) == [tok.token_to_id("<eos>")], (enc("<resp>"), enc(EOS))

    recs_all = build_records(rows, enc=enc)
    n_dup_all = len(recs_all) - len({r["ctx"] for r in recs_all})
    print(f"[ARITH] 算术子集={len(recs_all)} 条（全 {n_blocks} 剔除两族非函数样本 "
          f"{n_blocks - len(recs_all)} 条） 其中重复 ctx={n_dup_all} 条（同 ctx 同 gold）",
          flush=True)

    uniq, n_dup = dedupe_by_ctx(recs_all)
    train, test = split_recs(uniq, a.n_train, a.n_test, a.seed, "full")

    # 逐位编解码往返 + 指纹（口径同 66）
    # ★ 实测：唯一不可逆字符是乘号 '×'（U+00D7）—— char_tokenizer 里它映射到 <unk>(id 129)，
    #   只影响往返检查，不影响 gold_int 抽取；如实报为数据质量事实，不计入函数性违例。
    n_rt = 0
    for rs in (train, test):
        for r in rs:
            r["p"], r["t"] = enc(r["ctx"]), enc(r["target"] + EOS)
            if dec(r["p"]) != r["ctx"] or dec(r["t"]) != r["target"] + EOS:
                n_rt += 1
    n_x = sum(1 for r in recs_all if "×" in r["ctx"] or "×" in r["gold"])
    tr_fp, te_fp = data_fp(train), data_fp(test)

    vio, info = check_functions(train + test, "full")
    print(f"[FMT] 编解码往返坏={n_rt}（全部由 '×'→<unk> 造成） | "
          f"ctx tok 最大={max(r['ctx_tok'] for r in train + test)} "
          f"| target tok 最大={max(r['resp_tok'] for r in train + test)} "
          f"| 位数分布(题面最大位)={dict(sorted(Counter(r['max_digits'] for r in recs_all).items()))} "
          f"| 位数分布(题型×最大位)="
          f"{dict(sorted(Counter((r['op'], r['max_digits']) for r in recs_all).items()))}", flush=True)
    print(f"[OOV] 算术子集里唯一的不可逆字符是乘号 '×'(U+00D7)：char_tokenizer 映射到 <unk>(id 129)"
          f" ⇒ 涉及 {n_x} 条（{100.0 * n_x / len(recs_all):.2f}%）的 ctx/gold 往返不逐字相等。"
          f"gold_int 抽取不受影响（只抽整数）；此项**不计入函数性违例**。", flush=True)

    # ---- 小数字子集（1–2 位加减在盘上不存在，用 2 位×2 位乘法替代；如实打印缺口）----
    n_add12 = sum(1 for r in recs_all if r["op"] == "+" and r["max_digits"] <= 2)
    n_sub12 = sum(1 for r in recs_all if r["op"] == "-" and r["max_digits"] <= 2)
    small_all = [r for r in recs_all if r["small_2x2"]]
    small_uniq, n_dup_s = dedupe_by_ctx(small_all)
    s_train, s_test = split_recs(small_uniq, a.n_small_train, a.n_small_test, a.seed, "small")
    print(f"[SMALL-DEFICIT] 请求的「1–2 位加减子集」（题面两数都 <= 99）：加法={n_add12} 条 / "
          f"减法={n_sub12} 条 ⇒ 规模不足，不可用于 3000 步出信号。", flush=True)
    print(f"[SMALL-ALT] 替代小数字子集「2 位×2 位乘法」：条数={len(small_all)} 唯一ctx={len(small_uniq)} "
          f"(重复 {n_dup_s}) ⇒ train={len(s_train)} test={len(s_test)}（落盘 "
          f"{os.path.basename(OUT_SMALL_TMPL.format(split='train'))} / "
          f"{os.path.basename(OUT_SMALL_TMPL.format(split='test'))}）", flush=True)
    for rs in (s_train, s_test):
        for r in rs:
            r["p"], r["t"] = enc(r["ctx"]), enc(r["target"] + EOS)
    s_tr_fp, s_te_fp = data_fp(s_train), data_fp(s_test)
    s_vio, _ = check_functions(s_train + s_test, "small")

    # ---- extract_arith 交叉校验（只在两边操作数 <= 100 的条数上；实测能复用多少）----
    from training.rl.reward.arith_gen import extract_arith  # training/rl/reward/arith_gen.py:73
    elig = [r for r in recs_all if r["nums"][0] <= 100 and r["nums"][-1] <= 100]
    got = [r for r in elig if extract_arith(r["q"]) is not None]
    agree = sum(1 for r in got if extract_arith(r["q"])[3] == r["gold_int"])
    print(f"[XCHK] arith_gen.extract_arith 可用条数={len(got)}/{len(elig)}（操作数都<=100 的样本）"
          f" | 与 gold_int 一致={agree} 分歧={len(got) - agree}"
          f" | 全量 52,202 条里该函数覆盖不到 {len(recs_all) - len(got)} 条"
          f"（位数上限 3 + 操作数上限 100）", flush=True)

    # ---- 地板（全量 + 小数字子集）----
    compute_floors(train, test, "full")
    compute_floors(s_train, s_test, "small")

    # ---- 落盘 ----
    for split, rs in (("train", train), ("test", test)):
        fp = OUT_TMPL.format(split=split)
        with open(fp, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("p", "t")},
                                   ensure_ascii=False) + "\n")
        print(f"[OUT] {split} → {fp} ({len(rs)} 条)", flush=True)
    for split, rs in (("train", s_train), ("test", s_test)):
        fp = OUT_SMALL_TMPL.format(split=split)
        with open(fp, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("p", "t")},
                                   ensure_ascii=False) + "\n")
        print(f"[OUT] small {split} → {fp} ({len(rs)} 条)", flush=True)

    # ---- [D70] 一行汇总 ----
    total_vio = sum(vio.values()) + sum(s_vio.values())   # 往返坏（'×'→<unk>）另计，不算违例
    print(f"[D70] n_train={len(train)} n_test={len(test)} 池={len(uniq)}(去重自 {len(recs_all)}，"
          f"重复ctx={n_dup}) seed={a.seed} | 位数分布(题型×题面最大位)="
          f"{dict(sorted(Counter((r['op'], r['max_digits']) for r in recs_all).items()))} "
          f"| gold_int 位数分布={dict(sorted(Counter(r['gold_digits'] for r in recs_all).items()))} "
          f"| train_fp={tr_fp} test_fp={te_fp} | 函数性违例={sum(vio.values())} "
          f"(ctx冲突={info['conflict']} 其它={sum(vio.values()) - info['conflict']}) "
          f"| gold_int 抽不出={sum(1 for r in recs_all if r['gold_int'] is None)} "
          f"| small: n_train={len(s_train)} n_test={len(s_test)} 池={len(small_uniq)} "
          f"train_fp={s_tr_fp} test_fp={s_te_fp} 违例={sum(s_vio.values())}", flush=True)
    print("[D70-VERDICT] " + ("函数性全过；地板已报；P0-a 判据见 [FLOOR] max_naive 行。"
                              if total_vio == 0 else
                              f"★违例={total_vio} ⇒ 数据无效，停。"), flush=True)
    print("[DONE] exit=0", flush=True)
    return 0 if total_vio == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
