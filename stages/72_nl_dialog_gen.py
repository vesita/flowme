#!/usr/bin/env python3
"""S72 · L3-r2 自造可判「多轮自然语言对话」数据生成器 + 地板（CPU、确定性 seed、秒~分钟级）。

设计见 L3_PLAN.md §9 与 §9.1；本文件只做**数据面与地板**，不训模型、不碰 GPU、不用 systemd-run。

每条样本 = 3 轮对话（用户话 + 系统话 成对），目标 = 第 3 轮的系统回复：
    轮1 用户话 / 轮1 系统话 / 轮2 用户话 / 轮2 系统话 / 轮3 用户话 / <resp>（待生成）
轮1 给**实体**（运单号 / 姓名 / 座位号 / 工单号 / 会员卡号），轮2 给**条件**（时间 / 数量 / 方式），
轮3 用**指代与省略**提请求 —— 轮3 里既没有实体、也没有条件（不变量 D）。
目标回复必须**同时**含轮1 的实体与轮2 的条件（不变量 B）⇒ 不读历史写不出来。

句面由输入确定（L2 §8.4 的教训：句面若随机抽、输入里看不见，EM 上限会崩）：
    回复模板 = DET(族, 轮3 句式 index)
每族 4 个轮3 句式各自绑**唯一**一个回复模板；轮3 文本在族内唯一 ⇒ 从 ctx 文本可复原 index。
[TPL] 行把这张 DET 表整张打印出来（可复原判定），字段 t3_idx / tpl_id 同步落盘。

5 个语义族（每族 3 种轮1 句式 × 3 种轮2 句式 × 4 种轮3 句式）：
    快递（运单号 + 时间） · 预约（姓名 + 时间） · 订座（座位号 + 数量）
    售后（工单号 + 方式） · 会员（会员卡号 + 方式）

五条不变量（逐条断言 + 打印计数，违例必须 0，否则退出码非 0）：
    A 函数性：同 ctx 不同 target = 0；
    B 信息充分：目标的实体与条件 100% 出现在输入里（实体在轮1、条件在轮2、且都在 target 里）；
    C 非退化：唯一 ctx 数 >= 0.9 × 样本数（禁「唯一题面 × 唯一回复」查找表）；
    D 必读历史：轮3 既无实体也无条件；且轮2 不含实体、轮1 不含条件（逐条 + 静态词表级）；
    E 用字可编码：数据里出现的每个字符都能被 char_tokenizer 逐字往返（decode(encode(c)) == c），
      不许出现 <unk> —— 这是 L3-r1 的 `×`(U+00D7) 教训（它不在词表里，把整个乘法族打成 0 分）。

产物（只写 logs/）：
    logs/72_dialog_{train,test}.jsonl          train 7000 / test 800
    logs/72s_dialog_{train,test}.jsonl         --smoke 小量（train 200 / test 100）
每条字段（沿用 stages/66_dialog_gen.py 与 stages/70_arith_dialog_gen.py 的 ctx/target/gold 风格 + 计数类）：
    ctx / target / gold / family / ent / cond / cond_type / ent_kind
    / t1_idx / t2_idx / t3_idx / tpl_id
    / t1_user / t1_sys / t2_user / t2_sys / t3（地板与「去历史」消融用）
    / n_turns / ent_count / cond_count / ent_pos / t3_chars / t3_tok
    / resp_chars / resp_bucket / ctx_tok / resp_tok
    / turn3_has_ent / turn3_has_cond / ent_in_t1 / cond_in_t2 / ent_in_target / cond_in_target
指纹口径与 stages/66_dialog_gen.py / stages/65_textgen_cards.py::data_fp 逐字一致：
    逐条 (len(p), p..., len(t), t...) 拼接后 sha256[:16]。

用法（本机只有 .venv/bin/python 装了 tokenizers；本脚本无 torch、秒~分钟级）：
    .venv/bin/python stages/72_nl_dialog_gen.py
    .venv/bin/python stages/72_nl_dialog_gen.py --smoke
    .venv/bin/python stages/72_nl_dialog_gen.py --n-train 7000 --n-test 800 --seed 20261010
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import struct
import sys
from collections import Counter, defaultdict

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
if NS_ROOT not in sys.path:
    sys.path.insert(0, NS_ROOT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG_DIR = os.path.join(ROOT, "logs")
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
OUT_TMPL = os.path.join(LOG_DIR, "72_dialog_{split}.jsonl")
OUT_SMOKE_TMPL = os.path.join(LOG_DIR, "72s_dialog_{split}.jsonl")

N_TRAIN = 7000
N_TEST = 800
N_SMOKE_TRAIN = 200
N_SMOKE_TEST = 100
SEED = 20261010
EOS = "<eos>"
USER_PREFIX = "用户："
RESP_PREFIX = "<resp>"
BUCKET_CHARS = 20          # 回复长度桶宽（字符）
T3_BUCKET_CHARS = 5        # 轮3 长度桶宽（字符）
ENT_POS_BUCKET = 10        # 实体位置桶宽（字符）
MAX_SEQ = 512              # ctx + target 字符 token 上限（与 66/70 同口径）
MIN_UNIQ_CTX_RATIO = 0.9   # 不变量 C：唯一 ctx / 样本数 下限


# ---------------- 5 个语义族：实体词表 / 条件词表 / 逐句式的明面模板 ----------------
# 约定（构造即保证不变量）：
#   t1_user 每个模板恰含一个 {ent}；t2_user 恰含一个 {cond}；t3_user 不含任何占位符；
#   回复模板每族 4 个（与 t3_user 的下标一一对应），恰含 {ent} 与 {cond} 各一个。
FAMILIES = (
    {
        "key": "快递",
        "ent_kind": "运单号",
        "cond_type": "时间",
        "ents": (
            "SF10293847", "SF20485716", "SF30917264", "SF41827365", "SF52938476",
            "SF63749518", "SF74651029", "SF85762130", "SF96873241", "SF10794352",
        ),
        "conds": (
            "明天下午", "后天上午", "这周五", "下周一",
            "周六上午", "周日下午", "三天内", "本周内",
        ),
        "t1_user": (
            "你好，我的运单号是{ent}，上周寄的包裹还没到。",
            "麻烦查一下，运单{ent}现在是什么状态。",
            "我这边有个快递，单号{ent}，一直没派送。",
        ),
        "t1_sys": (
            "好的，运单{ent}已登记，正在为您查询。",
            "收到，运单{ent}的物流状态马上为您调出。",
            "明白，运单{ent}已进入重点跟进列表。",
        ),
        "t2_user": (
            "我希望它在{cond}送到。",
            "能不能安排{cond}派送？",
            "时间上我只有{cond}方便收货。",
        ),
        "t2_sys": (
            "好的，已备注{cond}派送。",
            "可以，已把{cond}加入配送备注。",
            "收到，配送时间按{cond}安排。",
        ),
        "t3_user": (
            "那你说大概什么时候能到？",
            "帮我看看到底还要多久。",
            "这样安排没问题吧？",
            "那你再帮我确认一下。",
        ),
        "reply": (
            "{ent} 的包裹预计{cond}送达，请留意短信。",
            "已为您查询，运单 {ent} 预计{cond}送达。",
            "没问题，运单 {ent} 会在{cond}送达。",
            "已确认，运单 {ent} 的派送时间是{cond}。",
        ),
    },
    {
        "key": "预约",
        "ent_kind": "姓名",
        "cond_type": "时间",
        "ents": (
            "张伟", "李娜", "王芳", "刘洋", "陈静", "杨帆",
            "赵磊", "黄敏", "周涛", "吴倩", "徐强", "孙丽",
        ),
        "conds": (
            "周三上午", "周四下午", "周六上午", "周五晚上",
            "下周二上午", "本月二十号", "明天上午", "后天下午",
        ),
        "t1_user": (
            "我叫{ent}，想预约牙科门诊。",
            "我是{ent}，需要挂一个口腔科。",
            "你好，{ent}想约一次门诊检查。",
        ),
        "t1_sys": (
            "好的{ent}，预约通道已为您打开。",
            "收到{ent}，门诊号正在为您生成。",
            "明白{ent}，已进入预约流程。",
        ),
        "t2_user": (
            "时间上我希望安排在{cond}。",
            "能约在{cond}吗？",
            "我只有{cond}有空。",
        ),
        "t2_sys": (
            "好的，已按{cond}为您排号。",
            "可以，号源已锁定在{cond}。",
            "收到，预约时间记为{cond}。",
        ),
        "t3_user": (
            "那具体几点过去合适？",
            "帮我确认一下安排吧。",
            "这样能不能排上？",
            "那你再确认一下。",
        ),
        "reply": (
            "{ent}，您的门诊已约在{cond}，请提前十分钟到。",
            "已为您排号{ent}，就诊时间是{cond}。",
            "{ent}，{cond}的号已经挂好，请准时到。",
            "已确认{ent}的门诊时间为{cond}。",
        ),
    },
    {
        "key": "订座",
        "ent_kind": "座位号",
        "cond_type": "数量",
        "ents": (
            "A12", "A08", "B03", "B17", "C05",
            "C21", "D09", "D14", "E02", "E19",
        ),
        "conds": (
            "三个人", "四个人", "五个人", "六个人",
            "七个人", "八个人", "九个人", "十个人",
        ),
        "t1_user": (
            "你好，我订的座位号是{ent}，靠窗的位置。",
            "我这边有个预订，座位号{ent}。",
            "麻烦查一下我的座位号{ent}。",
        ),
        "t1_sys": (
            "好的，座位{ent}已锁定，靠窗保留。",
            "收到，座位{ent}的预订已调出。",
            "明白，正在核对座位{ent}。",
        ),
        "t2_user": (
            "我们一共{cond}，麻烦加个位子。",
            "人数改成{cond}吧。",
            "这次来的是{cond}。",
        ),
        "t2_sys": (
            "好的，已按{cond}调整桌位。",
            "收到，桌位改为{cond}的规格。",
            "明白，已备注{cond}。",
        ),
        "t3_user": (
            "那还能坐下吗？",
            "帮我看看这样行不行。",
            "位子够不够？",
            "那你再确认一下。",
        ),
        "reply": (
            "{ent} 已改为{cond}的桌子，靠窗保留。",
            "已为您调整，座位 {ent} 现在是{cond}的桌。",
            "没问题，{ent} 会安排成{cond}的位置。",
            "已确认，座位 {ent} 按{cond}保留。",
        ),
    },
    {
        "key": "售后",
        "ent_kind": "工单号",
        "cond_type": "方式",
        "ents": (
            "AS204815", "AS317926", "AS429037", "AS531048", "AS642159",
            "AS753260", "AS864371", "AS975482", "AS186593", "AS297604",
        ),
        "conds": (
            "上门取件", "到店维修", "寄回厂家", "远程指导",
            "换一台新的", "上门检测", "返厂检测", "免费清洗",
        ),
        "t1_user": (
            "你好，我的售后工单号是{ent}。",
            "我报修过，工单号{ent}。",
            "麻烦看下工单{ent}的处理进度。",
        ),
        "t1_sys": (
            "好的，工单{ent}已登记。",
            "收到，工单{ent}已进入处理队列。",
            "明白，正在调出工单{ent}。",
        ),
        "t2_user": (
            "我希望按{cond}的方式处理。",
            "能不能用{cond}的方式？",
            "处理方式就选{cond}吧。",
        ),
        "t2_sys": (
            "好的，已按{cond}登记。",
            "可以，处理方式记为{cond}。",
            "收到，已备注{cond}。",
        ),
        "t3_user": (
            "那大概要多久？",
            "这样能解决吗？",
            "帮我催一下吧。",
            "那你再帮我看看。",
        ),
        "reply": (
            "{ent} 已按{cond}处理，请保持电话畅通。",
            "已为您登记，工单 {ent} 的处理方式是{cond}。",
            "没问题，{ent} 会按{cond}安排。",
            "已确认，工单 {ent} 采用{cond}。",
        ),
    },
    {
        "key": "会员",
        "ent_kind": "会员卡号",
        "cond_type": "方式",
        "ents": (
            "VIP3928471", "VIP4839582", "VIP5740693", "VIP6651704", "VIP7562815",
            "VIP8473926", "VIP9384037", "VIP1295148", "VIP2106259", "VIP3017360",
        ),
        "conds": (
            "兑换十元券", "兑换二十元券", "兑换三十元券", "兑换五十元券",
            "升级为金卡", "延长一年有效期", "兑换一百元券", "升级为白金卡",
        ),
        "t1_user": (
            "你好，我的会员卡号是{ent}。",
            "我是老会员了，卡号{ent}。",
            "麻烦查下会员卡{ent}的权益。",
        ),
        "t1_sys": (
            "欢迎回来，会员卡{ent}已通过验证。",
            "收到，会员{ent}的权益已调出。",
            "明白，正在核对会员卡{ent}。",
        ),
        "t2_user": (
            "我想按{cond}来使用权益。",
            "能不能帮我{cond}？",
            "操作方式就选{cond}。",
        ),
        "t2_sys": (
            "好的，已按{cond}为您办理。",
            "可以，{cond}已加入待办。",
            "收到，方式记为{cond}。",
        ),
        "t3_user": (
            "现在就能办吗？",
            "这样划算吗？",
            "帮我看看还有什么。",
            "那你再确认一下。",
        ),
        "reply": (
            "{ent} 已按{cond}办理，可在下次消费时使用。",
            "已为您处理，会员 {ent} 的权益方式是{cond}。",
            "没问题，{ent} 会按{cond}执行。",
            "已确认，会员卡 {ent} 采用{cond}。",
        ),
    },
)

FAM_BY_KEY = {f["key"]: f for f in FAMILIES}
REPLY_TMPL = {f["key"]: f["reply"] for f in FAMILIES}
ALL_ENTS = tuple(e for f in FAMILIES for e in f["ents"])
ALL_CONDS = tuple(c for f in FAMILIES for c in f["conds"])
N_FAMILIES = len(FAMILIES)
N_REPLY_TEMPLATES = sum(len(f["reply"]) for f in FAMILIES)


# ---------------- 渲染：与 stages/66_dialog_gen.py 同款（用户：/ <resp> / <eos>） ----------------
def render_ctx(r1u: str, r1s: str, r2u: str, r2s: str, r3u: str) -> str:
    return (f"{USER_PREFIX}{r1u}\n{RESP_PREFIX}{r1s}{EOS}\n"
            f"{USER_PREFIX}{r2u}\n{RESP_PREFIX}{r2s}{EOS}\n"
            f"{USER_PREFIX}{r3u}\n{RESP_PREFIX}")


# ---------------- 可复原判定：从 ctx 文本复原 (族, 三个句式 index, 实体, 条件) ----------------
def _match_slot(text: str, tpls, values, key: str):
    """在模板列表里找「head + 槽位 + tail」且槽位落在闭合词表里的模板；返回 [(idx, value)]。"""
    out = []
    ph = "{" + key + "}"
    for idx, tpl in enumerate(tpls):
        head, sep, tail = tpl.partition(ph)
        if not sep:
            continue
        if len(text) < len(head) + len(tail):
            continue
        if not text.startswith(head):
            continue
        if tail:
            if not text.endswith(tail):
                continue
            mid = text[len(head): len(text) - len(tail)]
        else:
            mid = text[len(head):]
        if mid in values:
            out.append((idx, mid))
    return out


def parse_ctx(ctx: str):
    """只用 ctx 文本复原 (fam_key, t1_idx, t2_idx, t3_idx, ent, cond)；复原不出或歧义返回 None。

    这是「句面由输入确定」的强制点：不变量 A/C 与「参考上限」都走这一个函数。
    """
    lines = ctx.split("\n")
    if len(lines) != 6:
        return None
    if lines[5] != RESP_PREFIX:
        return None
    for i in (0, 2, 4):
        if not lines[i].startswith(USER_PREFIX):
            return None
    for i in (1, 3):
        if not lines[i].startswith(RESP_PREFIX) or not lines[i].endswith(EOS):
            return None
    r1u = lines[0][len(USER_PREFIX):]
    r2u = lines[2][len(USER_PREFIX):]
    r3u = lines[4][len(USER_PREFIX):]
    cands = []
    for fam in FAMILIES:
        m1 = _match_slot(r1u, fam["t1_user"], set(fam["ents"]), "ent")
        if len(m1) != 1:
            continue
        i1, ent = m1[0]
        m2 = _match_slot(r2u, fam["t2_user"], set(fam["conds"]), "cond")
        if len(m2) != 1:
            continue
        i2, cond = m2[0]
        if r3u not in fam["t3_user"]:
            continue
        i3 = fam["t3_user"].index(r3u)
        cands.append((fam["key"], i1, i2, i3, ent, cond))
    if len(cands) != 1:
        return None
    return cands[0]


def det_reply(fam_key: str, t3_idx: int, ent: str, cond: str) -> str:
    """DET(族, 轮3 句式 index) + 两个槽位 ⇒ 唯一回复句面。"""
    return REPLY_TMPL[fam_key][t3_idx].format(ent=ent, cond=cond)


# ---------------- 生成 ----------------
def build_combos():
    """组合空间 = 族 × 轮1句式 × 轮2句式 × 轮3句式 × 实体 × 条件（全互异）。"""
    combos = []
    for fi, fam in enumerate(FAMILIES):
        for i1 in range(len(fam["t1_user"])):
            for i2 in range(len(fam["t2_user"])):
                for i3 in range(len(fam["t3_user"])):
                    for ent in fam["ents"]:
                        for cond in fam["conds"]:
                            combos.append((fi, i1, i2, i3, ent, cond))
    return combos


def gen_records(seed: int, n_train: int, n_test: int):
    """确定性无放回抽样：shuffle 组合空间后切前 need 条（train 在前、test 在后）。"""
    combos = build_combos()
    rng = random.Random(seed)
    rng.shuffle(combos)
    need = n_train + n_test
    if len(combos) < need:
        raise SystemExit(f"[D72] 组合空间不足：{len(combos)} < {need}")
    recs = []
    for fi, i1, i2, i3, ent, cond in combos[:need]:
        fam = FAMILIES[fi]
        r1u = fam["t1_user"][i1].format(ent=ent)
        r1s = fam["t1_sys"][i1].format(ent=ent)
        r2u = fam["t2_user"][i2].format(cond=cond)
        r2s = fam["t2_sys"][i2].format(cond=cond)
        r3u = fam["t3_user"][i3]
        gold = det_reply(fam["key"], i3, ent, cond)
        ctx = render_ctx(r1u, r1s, r2u, r2s, r3u)
        recs.append(dict(
            ctx=ctx, target=gold, gold=gold,
            family=fam["key"], ent=ent, cond=cond,
            cond_type=fam["cond_type"], ent_kind=fam["ent_kind"],
            t1_idx=i1, t2_idx=i2, t3_idx=i3,
            tpl_id=f"{fam['key']}#t3_{i3}",
            t1_user=r1u, t1_sys=r1s, t2_user=r2u, t2_sys=r2s, t3=r3u,
            n_turns=3,
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
def check_invariants(recs, tag="all"):
    """返回 (vio Counter, info dict)。A/B/C/D/E 逐条判。"""
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

    # ---- 逐条 B / D / E ----
    for r in recs:
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

    # ---- 模板级静态检查：占位符个数 / 可复原判定 ----
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

    print(f"[INV:{tag}] A 函数性：唯一ctx={len(seen)} 重复ctx={info['dup_ctx']} "
          f"同ctx不同target={vio['A_同ctx不同target']}（违例）", flush=True)
    print(f"[INV:{tag}] B 信息充分：通过={ok['B']}/{len(recs)} 违例={vio['B_实体或条件不在输入里']}", flush=True)
    print(f"[INV:{tag}] C 非退化：唯一ctx/样本数={ratio:.4f}（门槛>={MIN_UNIQ_CTX_RATIO}）"
          f" 违例={vio['C_唯一ctx比不足']}", flush=True)
    print(f"[INV:{tag}] D 必读历史：通过={ok['D']}/{len(recs)} 违例={vio['D_轮3含实体或条件(含轮2含实体/轮1含条件)']} "
          f"| 静态词表级 轮3含实体={static_ent} 轮3含条件={static_cond} "
          f"轮2含实体={static_t2_ent} 轮1含条件={static_t1_cond}", flush=True)
    print(f"[INV:{tag}] E 用字可编码：坏字符={info.get('bad_chars', [])} "
          f"逐条往返坏={vio['E_逐条往返坏']}", flush=True)
    print(f"[INV:{tag}] PARSE 可复原判定（族+三轮句式 index+实体+条件 六元组复原）失败={n_parse_bad} "
          f"| SHAPE 模板占位符静态违例={shape_vio}", flush=True)
    return vio, info


# ---------------- 指纹（与 stages/66::data_fp 逐字一致） ----------------
def data_fp(recs) -> str:
    h = hashlib.sha256()
    for r in recs:
        for seq in (r["p"], r["t"]):
            h.update(struct.pack("<Q", len(seq)))
            if seq:
                h.update(struct.pack(f"<{len(seq)}i", *seq))
    return h.hexdigest()[:16]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read())
    return h.hexdigest()


# ---------------- 地板（L5）：只用输入可见信息 + train 统计 ----------------
def _em(gold: str, pred: str) -> int:
    return int((gold or "").strip() == (pred or "").strip())


def _bucket_specs():
    return (
        ("⑤a回复长度桶(20字)", lambda r: r["resp_bucket"]),
        ("⑤b轮数桶", lambda r: r["n_turns"]),
        ("⑤c实体个数桶", lambda r: r["ent_count"]),
        ("⑤d条件类型桶", lambda r: r["cond_type"]),
        ("⑤e轮3句式桶(族×句式)", lambda r: (r["family"], r["t3_idx"])),
        ("⑤f轮3长度桶(5字)", lambda r: r["t3_chars"] // T3_BUCKET_CHARS),
        ("⑤g实体位置桶(10字)", lambda r: r["ent_pos"] // ENT_POS_BUCKET),
        ("⑤h条件个数桶", lambda r: r["cond_count"]),
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
    """⑥ 强朴素规则：只用轮3 认出 (族, 句式)，实体/条件一律填 train 族内众数。

    这是「只会背句面模板、不会读历史」的地板上界（我们希望的正是它低）。
    """
    cands = []
    for f in FAMILIES:
        if r["t3"] in f["t3_user"]:
            i3 = f["t3_user"].index(r["t3"])
            cands.append(det_reply(f["key"], i3, s["maj_ent"][f["key"]], s["maj_cond"][f["key"]]))
    return cands


def _pred_reference(r, s):
    """参考上限（非地板）：完美实体 + 完美条件 + DET 句面 —— 用来证明信息都在输入里。"""
    p = parse_ctx(r["ctx"])
    if p is None:
        s["miss"] += 1
        return ""
    fk, _i1, _i2, i3, ent, cond = p
    return det_reply(fk, i3, ent, cond)


def _rule_preds():
    """每条规则返回**预测文本**；EM 由整体字符串比较（与卡片管线的回复口径一致）。"""
    rules = [
        ("①复制轮3用户话", lambda r, s: [r["t3"]]),
        ("②复制轮1用户话", lambda r, s: [r["t1_user"]]),
        ("②b复制轮1系统话", lambda r, s: [r["t1_sys"]]),
        ("③复制轮2用户话", lambda r, s: [r["t2_user"]]),
        ("③b复制轮2系统话", lambda r, s: [r["t2_sys"]]),
        ("④train全集众数", lambda r, s: [s["maj_target"]]),
        ("⑥模板填空只用轮3(实体/条件填train族内众数)", _pred_full_ctx_fill),
    ]
    for name, keyfn in _bucket_specs():
        def mk(keyfn=keyfn, name=name):
            def f(r, s):
                return [s["bucket"][name].get(keyfn(r), s["maj_target"])]
            return f
        rules.append((name, mk()))
    return rules


NAIVE_NAMES = tuple(n for n, _ in _rule_preds())
REFERENCE_NAME = "参考:完美实体+完美条件+DET句面"


def compute_floors(train, test, tag):
    """在 test 上逐条算 EM；预测候选可以多条（模板填空的歧义取「命中任一即算中」，对地板更保守）。"""
    st = _stats(train)
    rules = _rule_preds()
    miss = Counter()

    def eval_rules(rs):
        rows = {}
        for name, fn in rules:
            if name not in miss:
                miss[name] = 0
            hit = 0
            for r in rs:
                preds = fn(r, st)
                if preds is None:
                    preds = [""]
                hit += int(any(_em(r["target"], p) for p in preds))
            rows[name] = 100.0 * hit / len(rs)
        return rows

    rows_test = eval_rules(test)
    for name, _fn in rules:
        print(f"[FLOOR] {tag} {name} EM={rows_test[name]:.2f}% (n={len(test)})", flush=True)
    # 参考上限（非地板）
    s_ref = {"miss": 0}
    ref_hit = sum(int(_em(r["target"], _pred_reference(r, s_ref))) for r in test)
    ref_em = 100.0 * ref_hit / len(test)
    print(f"[FLOOR] {tag} {REFERENCE_NAME} EM={ref_em:.2f}% (n={len(test)}) "
          f"识别不出ctx={s_ref['miss']} ⇒ 信息全在输入里（非地板）", flush=True)

    naive = {k: v for k, v in rows_test.items() if k in NAIVE_NAMES}
    mx_name = max(naive, key=naive.get)
    mx = naive[mx_name]
    se2 = 2.0 * (mx / 100.0 * (1 - mx / 100.0) / len(test)) ** 0.5 * 100.0
    print(f"[FLOOR] {tag} max_naive={mx:.2f}% 来自={mx_name} | 2SE={se2:.2f}pp ⇒ "
          f"R2-b 门槛=max_naive+2SE={mx + se2:.2f}% | max_naive<=5%: "
          f"{'PASS' if mx <= 5.0 else 'FAIL'}", flush=True)
    # train 侧同口径（判数据是否退化：朴素规则在 train 上 >= 90% ⇒ 数据无效）
    rows_train = eval_rules(train)
    tr_name = max(rows_train, key=lambda k: rows_train[k])
    print(f"[FLOOR] {tag} train max_naive={rows_train[tr_name]:.2f}% 来自={tr_name} ⇒ "
          + ("★数据无效（朴素规则在 train 上>=90%）" if rows_train[tr_name] >= 90.0
             else "有效（无朴素规则在 train 上>=90%）"), flush=True)
    return rows_test, mx_name, mx, ref_em, rows_train


# ---------------- 入口 ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="S72 L3-r2 自造可判多轮自然语言对话数据生成器+地板")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-train", type=int, default=N_TRAIN)
    ap.add_argument("--n-test", type=int, default=N_TEST)
    ap.add_argument("--smoke", action="store_true", help="小量冒烟（train 200 / test 100，写 72s_*）")
    a = ap.parse_args(argv)

    if a.smoke:
        a.n_train, a.n_test = N_SMOKE_TRAIN, N_SMOKE_TEST
        out_tmpl = OUT_SMOKE_TMPL
    else:
        out_tmpl = OUT_TMPL

    recs, combos = gen_records(a.seed, a.n_train, a.n_test)
    train, test = recs[:a.n_train], recs[a.n_train:]
    print(f"[SPLIT] 组合空间={len(combos)}（无放回抽样后 train={len(train)} test={len(test)}）"
          f" | 族={N_FAMILIES} 轮1/2/3句式和="
          + " ".join(f"{f['key']}[{len(f['t1_user'])}/{len(f['t2_user'])}/{len(f['t3_user'])}]"
                     for f in FAMILIES), flush=True)
    print(f"[TPL] DET 句面表（回复模板 = DET(族, 轮3句式 index)，回复模板总数={N_REPLY_TEMPLATES}）：",
          flush=True)
    for f in FAMILIES:
        for i3, tpl in enumerate(f["reply"]):
            print(f"[TPL] {f['key']}#t3_{i3} 轮3={f['t3_user'][i3]!r} ⇒ {tpl}", flush=True)
    print("[VOCAB] " + " | ".join(
        f"{f['key']} 实体[{f['ent_kind']}]={len(f['ents'])} 条件[{f['cond_type']}]={len(f['conds'])}"
        for f in FAMILIES), flush=True)

    # ---- tokenizer：逐字往返（不变量 E）+ 逐条编码 + 指纹 ----
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    V = tok.get_vocab_size()
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids            # noqa: E731
    dec = lambda ids: tok.decode([int(i) for i in ids], skip_special_tokens=False)  # noqa: E731
    EOS_ID = tok.token_to_id(EOS)
    assert enc(RESP_PREFIX) == [140] and enc(EOS) == [EOS_ID], (enc(RESP_PREFIX), enc(EOS))

    # E：数据里出现的**每个字符**逐一往返
    chars = sorted({c for r in recs for s in (r["ctx"], r["target"]) for c in s})
    bad_chars = [(c, enc(c), dec(enc(c))) for c in chars if dec(enc(c)) != c]
    bad_set = {c for c, _i, _b in bad_chars}
    print(f"[E-CHECK] 数据中不同字符数={len(chars)} 逐字往返坏字符={len(bad_chars)} "
          f"{bad_chars} | V={V} EOS={EOS_ID}", flush=True)

    for r in recs:
        r["p"] = enc(r["ctx"])
        r["t"] = enc(r["target"] + EOS)
        r["t3_tok"] = len(enc(r["t3"]))
        r["ctx_tok"] = len(r["p"])
        r["resp_tok"] = len(r["t"])
        r["rt_ok"] = (dec(r["p"]) == r["ctx"] and dec(r["t"]) == r["target"] + EOS)

    n_additive_bad = sum(1 for r in recs if enc(r["ctx"] + r["target"] + EOS) != r["p"] + r["t"])
    assert max(r["ctx_tok"] for r in recs) + max(r["resp_tok"] for r in recs) <= MAX_SEQ, "ctx+target 超 512"
    print(f"[FMT] 编解码整串往返坏={sum(1 for r in recs if not r['rt_ok'])} | "
          f"enc 可加性破坏={n_additive_bad} | ctx tok 中位="
          f"{sorted(r['ctx_tok'] for r in recs)[len(recs)//2]} 最大={max(r['ctx_tok'] for r in recs)} "
          f"| target tok 中位={sorted(r['resp_tok'] for r in recs)[len(recs)//2]} "
          f"最大={max(r['resp_tok'] for r in recs)}", flush=True)

    tr_fp, te_fp = data_fp(train), data_fp(test)

    # ---- 不变量 ----
    vio, info = check_invariants(train + test, tag="all")
    vio["E_坏字符"] = len(bad_chars)
    info["bad_chars"] = [c for c, _i, _b in bad_chars]
    total_vio = sum(vio.values())

    print(f"[DIST] 族分布={dict(Counter(r['family'] for r in recs))} | "
          f"条件类型分布={dict(Counter(r['cond_type'] for r in recs))} | "
          f"轮3句式分布={dict(sorted(Counter(r['tpl_id'] for r in recs).items()))} | "
          f"实体个数分布={dict(Counter(r['ent_count'] for r in recs))} | "
          f"回复长度(字符) min/中位/max="
          f"{min(r['resp_chars'] for r in recs)}/"
          f"{sorted(r['resp_chars'] for r in recs)[len(recs)//2]}/"
          f"{max(r['resp_chars'] for r in recs)}", flush=True)

    # ---- 落盘（p/t 只在内存里用于指纹与长度统计，不写进 jsonl）----
    for split, rs in (("train", train), ("test", test)):
        fp = out_tmpl.format(split=split)
        with open(fp, "w", encoding="utf-8") as f:
            for r in rs:
                f.write(json.dumps({k: v for k, v in r.items() if k not in ("p", "t")},
                                   ensure_ascii=False) + "\n")
        print(f"[OUT] {split} → {fp} ({len(rs)} 条) sha256={file_sha256(fp)}", flush=True)
    print(f"[OUT] test jsonl sha256={file_sha256(out_tmpl.format(split='test'))}", flush=True)

    # ---- 地板 ----
    floors, mx_name, mx, ref_em, floors_train = compute_floors(train, test, tag="test")

    # ---- 三条 [GEN] 样例（完整 ctx → gold，人工核「轮3 确实没有实体与条件」）----
    picked = []
    seen_fam = set()
    for r in test:
        if r["family"] not in seen_fam:
            seen_fam.add(r["family"])
            picked.append(r)
        if len(picked) == 3:
            break
    for k, r in enumerate(picked, 1):
        print(f"[GEN{k}] family={r['family']} ent={r['ent']!r} cond={r['cond']!r} "
              f"cond_type={r['cond_type']} tpl_id={r['tpl_id']}", flush=True)
        print(f"[GEN{k}] 轮1用户：{r['t1_user']}", flush=True)
        print(f"[GEN{k}] 轮1系统：{r['t1_sys']}", flush=True)
        print(f"[GEN{k}] 轮2用户：{r['t2_user']}", flush=True)
        print(f"[GEN{k}] 轮2系统：{r['t2_sys']}", flush=True)
        print(f"[GEN{k}] 轮3用户：{r['t3']}  "
              f"(轮3含实体={r['turn3_has_ent']} 轮3含条件={r['turn3_has_cond']})", flush=True)
        print(f"[GEN{k}] gold：{r['gold']}  (gold含实体={r['ent_in_target']} gold含条件={r['cond_in_target']})",
              flush=True)
        print(f"[GEN{k}] 原始 ctx 逐行：{r['ctx']!r}", flush=True)

    # ---- [D72] 一行汇总 ----
    print(f"[D72] n_train={len(train)} n_test={len(test)} 池={len(combos)}(组合空间) seed={a.seed} | "
          f"族={N_FAMILIES} 轮1句式/族={[len(f['t1_user']) for f in FAMILIES]} "
          f"轮2句式/族={[len(f['t2_user']) for f in FAMILIES]} "
          f"轮3句式/族={[len(f['t3_user']) for f in FAMILIES]} 回复模板={N_REPLY_TEMPLATES} | "
          f"A违例={vio['A_同ctx不同target']} B违例={vio['B_实体或条件不在输入里']} "
          f"C违例={vio['C_唯一ctx比不足']}(唯一ctx比={info['uniq_ctx_ratio']:.4f}) "
          f"D违例={vio['D_轮3含实体或条件(含轮2含实体/轮1含条件)']}"
          f"(静态:{vio['D_static_轮3含实体']}/{vio['D_static_轮3含条件']}/"
          f"{vio['D_static_轮2含实体']}/{vio['D_static_轮1含条件']}) "
          f"E违例={vio['E_坏字符'] + vio['E_逐条往返坏']}(坏字符={vio['E_坏字符']}) "
          f"PARSE违例={vio['PARSE_可复原判定失败']} | "
          f"train_fp={tr_fp} test_fp={te_fp} | 参考上限={ref_em:.2f}%", flush=True)
    print(f"[D72-VERDICT] R2-a={'PASS' if (vio['A_同ctx不同target'] == 0 and vio['B_实体或条件不在输入里'] == 0) else 'FAIL'}"
          f" R2-b={'PASS' if (mx <= 5.0 and info['uniq_ctx_ratio'] >= MIN_UNIQ_CTX_RATIO) else 'FAIL'}"
          f"（max_naive={mx:.2f}% 来自={mx_name}；唯一ctx比={info['uniq_ctx_ratio']:.4f}）"
          + (" | ★违例>0 ⇒ 数据无效，停。" if total_vio else " | 不变量全过，地板已报。"), flush=True)
    print(f"[DONE] exit={0 if total_vio == 0 else 4}", flush=True)
    return 0 if total_vio == 0 else 4


if __name__ == "__main__":
    sys.exit(main())
