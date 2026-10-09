#!/usr/bin/env python3
"""S49 · P4 门1 ★核心验证（零训练 · 全 CPU · 只前向）：把 stages/ 已训好的 ckpt 灌进 cards/ 路径，
与 stages/ 的 oracle 实现逐位比 logits / 贪心预测 / EM。

只回答一个问题：用 `cards/` 框架（InputCard → TargetCard → OutputCard，走黑板路由）重装的
A/B/C/D/E 五种卡内结构，装入 stages/36、stages/44、stages/38 **同一份已训好的权重**后，
前向是否与实验脚本逐位一致？

做法（★零训练：不 backward、不 opt.step、不构造优化器；全 CPU）：
 ① 数据 / 超参 / 评估口径一律从 `stages/19_residual_scan.py` **原文本 exec**（切到 "# ③ 主循环"
    之前）⇒ DATA(add_3d train4000/test800)、Rng、enc/dec、build_batch、greedy_gen、parse_ans、
    hits_from、em_report、mean_se、sin_pe/enc_layer 全部与 S36/S44 同源（防空测试）。
    S19 头在 device!=cuda 时 sys.exit(3)；本进程 CUDA_VISIBLE_DEVICES=""，只在 exec 期间用
    只读补丁绕过该门，exec 之后把命名空间里的 device 改回 "cpu"（不改源文本、不初始化 CUDA）。
 ② oracle（对照实现）从**原文本 exec**，不重写：
      stages/36 → `FP = 0.1` 常量段 + ③ 五臂卡段（CardA..Cards36；含 [S36-INIT] A≡res 自检）；
      stages/44 → `FP = 0.1` 常量段 + ③ 五臂卡段（CardA..Cards44/Cards38Clone；含 [S44-INIT]）。
 ③ cards 侧（被验对象）：本文件按 `cards/base.py` 契约新写 CardB/C/D/E(Card)（io=d->d，
    apply_card 体逐字抄 stages/36:155-221，只把 h 改名 x —— 参数名保持 36 的 norm/mlp/dw/pw/
    gate/cand/att/do，便于键名映射）；A 直接用 `cards/target.py` 的 TargetCard(residual=True)。
    `StructPipeline(CardPipeline)` **只覆盖 __init__**（按 input→target→output 原地构造，再补
    blackboard/frozen_cards/route_history），route/freeze/snapshot/mv/load/contract 全部继承
    ⇒ 数值路径走框架的黑板路由 + `Card.forward` 的形状契约强制点。
 ④ 键名映射（36/44/38 存的是**裸 state_dict**，无 {format,contract,state} 包）：
      emb.*        → input.emb.*
      in_enc.*     → input.enc.*
      card.layer.* → target.think.*   （36 的 A 臂）
      thought.*    → target.think.*   （38 的 A 臂；44:285-288 的官定映射）
      card.*       → target.*         （36/44 的 B–E 臂）
      head.*       → output.head.*
    strict=True 装载，报张量数 / missing / unexpected。
 ⑤ 判据（写死）：同一批 test 800 条 —— logits `max|Δ| == 0.0`（整批分块 + batch=1 逐样本两条
    路径）、贪心预测逐样本逐字相同；EM(clean n=770) 与 oracle CPU 侧 |Δ|==0，与入库值
    （METRICS.md:18 的 @6000 clean / logs/44_results.jsonl 的 @12000 em_clean）|Δ| ≤ 0.13pp。

★ 门1 只在 CPU 上做：CPU 逐位一致 **不代表**与 GPU 训练逐位一致（§7.12 / R34）。
只允许写：本文件、logs/49_*、/tmp；cards/ 既有文件与 stages/*.py 只读、不改。
运行：`.venv/bin/python cards/exp_struct_ablation.py`
"""
from __future__ import annotations

import contextlib
import hashlib
import io
import json
import math
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ★绝不碰 GPU（P3 在 GPU 上跑 46_*）
ROOT = "/home/vesita/coding/my/flowme"
sys.path.insert(0, ROOT)
T0 = time.time()
LOG = f"{ROOT}/logs/49_gate1.log"
JSON_OUT = f"{ROOT}/logs/49_gate1.json"
SRC19 = f"{ROOT}/stages/19_residual_scan.py"
SRC36 = f"{ROOT}/stages/36_saturation_retest.py"
SRC44 = f"{ROOT}/stages/44_struct_at_12k.py"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

torch.set_num_threads(int(os.environ.get("S49_THREADS", "4")))
# ★线程数=4（实测：batch=1 逐样本贪心在 12 线程下线程颠簸 —— E 臂 50 样本 0.27s@4线程 vs
#   25.86s@12线程，差 96×；两侧模型同进程同线程数，逐位比较不受影响。见
#   logs/49_gate1_partial_12threads.log：E 臂 1 个 case 要 963.6s）

from cards import (Card, IO_D_TO_D, Blackboard, CardPipeline, InputCard,  # noqa: E402
                   OutputCard, TargetCard, state_sha256)

FP = 0.1                      # 由 stages/36:68 与 stages/44:68 的原文本常量段核对（见下）
CHUNK = 200                   # 整批 logits 比对的分块大小（控显存/内存）
N_B1 = 64                     # batch=1 逐样本 logits 比对的样本数
EM_TOL = 0.0013               # 0.13pp = 1/800


# ============================================================================
# 0. 原文本 exec 工具
# ============================================================================
def _span(path: str, a: str, b: str) -> tuple[str, int, int]:
    """取出 path 文本里 [a, b) 的区间（a/b 为字面量标记），返回 (文本, 起行, 止行)。"""
    src = open(path, encoding="utf-8").read()
    i = src.index(a)
    j = src.index(b, i)
    return src[i:j], src.count("\n", 0, i) + 1, src.count("\n", 0, j) + 1


def _exec_span(path: str, a: str, b: str, ns: dict) -> str:
    txt, l0, l1 = _span(path, a, b)
    exec(compile(txt, path, "exec"), ns)          # noqa: S102 - 只读原文本
    return f"{os.path.basename(path)}:{l0}-{l1 - 1}"


def _exec_head(path: str, cut_marker: str, ns: dict) -> str:
    """exec path 到 cut_marker 之前。S19 头有 GPU 门（device!=cuda ⇒ sys.exit(3)），
    CUDA 不可见时只在 exec 期间用只读补丁绕过，exec 后立即还原。"""
    src = open(path, encoding="utf-8").read()
    cut = src.index(cut_marker)
    saved = (torch.cuda.is_available, torch.cuda.get_device_name)
    torch.cuda.is_available = lambda: True
    torch.cuda.get_device_name = lambda *a, **k: "cpu-forced(s49)"
    try:
        exec(compile(src[:cut], path, "exec"), ns)        # noqa: S102
    finally:
        torch.cuda.is_available, torch.cuda.get_device_name = saved
    return f"{os.path.basename(path)}:1-{src.count(chr(10), 0, cut)}"


print(f"[S49] ★P4 门1：stages ckpt → cards/ 路径逐位 XCHECK | device=cpu（零训练，只前向）| "
      f"torch={torch.__version__} threads={torch.get_num_threads()} | "
      f"CUDA_VISIBLE_DEVICES='{os.environ['CUDA_VISIBLE_DEVICES']}'", flush=True)

# ============================================================================
# 1. 原文本 exec：S19 头（数据/超参/评估口径）
# ============================================================================
print(f"\n[S49] ===== ① exec {SRC19}（原文本，到 '# ③ 主循环' 之前）=====", flush=True)
S: dict = {"__name__": "s19_head", "__file__": SRC19}
head_span = _exec_head(SRC19, "# ③ 主循环", S)
S["device"] = "cpu"          # ★exec 后改回 cpu：S19 头里的函数按全局名查它
D, FF, NHEAD, MAXLEN = S["D"], S["FF"], S["NHEAD"], S["MAXLEN"]
V, PAD_ID, EOS_ID = S["V"], S["PAD_ID"], S["EOS_ID"]
DATA, greedy_gen, parse_ans, hits_from = S["DATA"], S["greedy_gen"], S["parse_ans"], S["hits_from"]
TEST = DATA["add_3d"]["test"]
TRAIN = DATA["add_3d"]["train"]
print(f"[S49-EXEC] {head_span} | device(命名空间)={S['device']} | "
      f"D={D} FF={FF} NHEAD={NHEAD} MAXLEN={MAXLEN} V={V} PAD={PAD_ID} EOS={EOS_ID} | "
      f"add_3d train={len(TRAIN)} test={len(TEST)}", flush=True)

# 门0 指纹复认（与 stages/36:110-118 同口径，写死期望值）
_h = hashlib.sha256()
for r in TEST:
    _h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
TEST_SHA = _h.hexdigest()[:16]
_tp_tr = {(r["a"], r["b"]) for r in TRAIN}
for r in TEST:
    r["seen"] = (r["a"], r["b"]) in _tp_tr
LEAK = sum(1 for r in TEST if r["seen"])
BAD_IDX = [i for i, r in enumerate(TEST) if "{expr}" in r["nl"]]
CLEAN_IDX = [i for i in range(len(TEST)) if i not in set(BAD_IDX)]
print(f"[S49-DATA] TEST sha256[:16]={TEST_SHA}（入库 b34e7ea515203227）| (a,b) 泄漏={LEAK}/800"
      f"（入库 16）| bad{{expr}}={len(BAD_IDX)} | clean n={len(CLEAN_IDX)}", flush=True)

# ============================================================================
# 2. 原文本 exec：oracle（stages/36 与 stages/44 的卡定义）
# ============================================================================
print(f"\n[S49] ===== ② exec oracle 段（stages/36 与 stages/44 原文本）=====", flush=True)
N36 = dict(S)
N36.update(ROOT=ROOT, SMOKE36=False)
s36a = _exec_span(SRC36, "FP = 0.1", 'print(f"\\n[S36] ★E6', N36)
s36b = _exec_span(SRC36, "class CardA(nn.Module):", "def train36(", N36)
N44 = dict(S)
N44.update(ROOT=ROOT, SMOKE44=False)
s44a = _exec_span(SRC44, "FP = 0.1", "assert str(device)", N44)
s44b = _exec_span(SRC44, "class CardA(nn.Module):", "def train44(", N44)
print(f"[S49-EXEC] {s36a} + {s36b} ⇒ oracle36: Cards36/Card{{A,B,C,D,E}}/build_card", flush=True)
print(f"[S49-EXEC] {s44a} + {s44b} ⇒ oracle44: Cards44/Cards38Clone/load_a_state", flush=True)
assert N36["FP"] == N44["FP"] == FP, f"FP 漂移：{N36['FP']} {N44['FP']} {FP}"
assert N36["NHEAD"] == N44["NHEAD"] == NHEAD, "NHEAD 漂移"

# ============================================================================
# 3. cards 侧：B/C/D/E 卡内结构（apply_card 体逐字抄 stages/36:155-221，h 改名 x）
# ============================================================================
class CardB(Card):
    """抄 stages/36:155-164：纯逐位置 MLP（零跨位置混合）。"""

    io = IO_D_TO_D
    card_name = "target"

    def __init__(self, d: int, ff: int, maxlen: int):
        Card.__init__(self, d, maxlen)
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(FP),
                                 nn.Linear(ff, d), nn.Dropout(FP))

    def apply_card(self, x, mask=None):
        h = x
        return h + self.mlp(self.norm(h))


class CardC(Card):
    """抄 stages/36:166-179：因果卷积 k=3（左填充）。"""

    io = IO_D_TO_D
    card_name = "target"

    def __init__(self, d: int, maxlen: int, k: int = 3):
        Card.__init__(self, d, maxlen)
        self.k = k
        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, k, groups=d, padding=k - 1)
        self.pw = nn.Linear(d, d)
        self.act = nn.GELU()
        self.do = nn.Dropout(FP)

    def apply_card(self, x, mask=None):
        h = x
        z = self.norm(h).transpose(1, 2)
        z = self.dw(z)[..., : h.size(1)].transpose(1, 2)
        return h + self.do(self.pw(self.act(z)))


class CardD(Card):
    """抄 stages/36:182-193：门控 GRU 风格。"""

    io = IO_D_TO_D
    card_name = "target"

    def __init__(self, d: int, ff: int, maxlen: int):
        Card.__init__(self, d, maxlen)
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Linear(d, d)
        self.cand = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def apply_card(self, x, mask=None):
        h = x
        z = self.norm(h)
        g = torch.sigmoid(self.gate(z))
        c = torch.tanh(self.cand(z))
        return g * c + (1.0 - g) * h


class CardE(Card):
    """抄 stages/36:196-206：仅注意力（去 FFN）。"""

    io = IO_D_TO_D
    card_name = "target"

    def __init__(self, d: int, maxlen: int, nhead: int):
        Card.__init__(self, d, maxlen)
        self.norm = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, nhead, dropout=FP, batch_first=True)
        self.do = nn.Dropout(FP)

    def apply_card(self, x, mask=None):
        h = x
        z = self.norm(h)
        o = self.att(z, z, z, attn_mask=mask, need_weights=False)[0]
        return h + self.do(o)


def make_target(arm: str, d: int, ff: int, nhead: int, maxlen: int,
                dropout: float = 0.1, residual: bool = True) -> Card:
    """arm → 卡内结构。A = 现状（cards/target.py 的 TargetCard，外层残差）。"""
    if arm == "A":
        return TargetCard(d, ff, nhead, maxlen, dropout, residual)
    if arm == "B":
        return CardB(d, ff, maxlen)
    if arm == "C":
        return CardC(d, maxlen)
    if arm == "D":
        return CardD(d, ff, maxlen)
    if arm == "E":
        return CardE(d, maxlen, nhead)
    raise AssertionError(f"未知臂 {arm!r}")


class StructPipeline(CardPipeline):
    """★只覆盖 __init__：按 input → target(=A|B..E) → output 顺序原地构造，
    再补 blackboard/frozen_cards/route_history。其余 route/freeze/snapshot/mv/load/contract
    全部继承 cards/pipeline.py（不重写）。"""

    def __init__(self, arm: str, d: int = 128, vocab_size: int = 8192, ff: int = 512,
                 nhead: int = 4, maxlen: int = 512, pad_id: int = 0, dropout: float = 0.1,
                 residual: bool = True):
        nn.Module.__init__(self)          # 不走父类 __init__：避免多造一张被丢弃的 target 卡
        self.arm = arm
        self.input = InputCard(d, vocab_size, ff, nhead, maxlen, pad_id, dropout)
        self.target = make_target(arm, d, ff, nhead, maxlen, dropout, residual)
        self.output = OutputCard(d, vocab_size, maxlen)
        self.blackboard = Blackboard()
        self.frozen_cards: tuple = ()
        self.route_history: list = []


# ============================================================================
# 4. 键名映射（stages 裸 state_dict → cards 管线键名）
# ============================================================================
MAP_DESC = {
    "layer": "card.layer.* → target.think.*   (36 的 A 臂)",
    "thought": "thought.* → target.think.*     (38 的 A 臂，44:285-288 官定)",
    "card": "card.* → target.*             (36/44 的 B–E 臂)",
}
MAP_COMMON = "emb.* → input.emb.* | in_enc.* → input.enc.* | head.* → output.head.*"


def map_state(sd: dict, style: str) -> dict:
    out = {}
    for k, v in sd.items():
        if k.startswith("emb."):
            nk = "input.emb." + k[4:]
        elif k.startswith("in_enc."):
            nk = "input.enc." + k[7:]
        elif k.startswith("head."):
            nk = "output.head." + k[5:]
        elif style == "thought" and k.startswith("thought."):
            nk = "target.think." + k[8:]
        elif style == "layer" and k.startswith("card.layer."):
            nk = "target.think." + k[11:]
        elif style == "card" and k.startswith("card."):
            nk = "target." + k[5:]
        else:
            raise AssertionError(f"未覆盖的键 {k!r}（style={style}）")
        assert nk not in out, f"映射后键重复 {nk!r}"
        out[nk] = v
    return out


# ============================================================================
# 5. 入库 EM 值（stages 侧，GPU 训练时的入库口径）
# ============================================================================
EM_REF_6000_CLEAN = {                       # METRICS.md:18（E6@6000 clean n=770，1234/5678）
    ("A", 1234): 0.8896, ("A", 5678): 0.7727,
    ("B", 1234): 0.0117, ("B", 5678): 0.0091,
    ("C", 1234): 0.0091, ("C", 5678): 0.0104,
    ("D", 1234): 0.0143, ("D", 5678): 0.0104,
    ("E", 1234): 0.0130, ("E", 5678): 0.0143,
}
EM_REF_12000_CLEAN = {}                     # logs/44_results.jsonl 的 em_clean（GPU 入库）
_em_src = {}
with open(f"{ROOT}/logs/44_results.jsonl", encoding="utf-8") as f:
    for line in f:
        r = json.loads(line)
        EM_REF_12000_CLEAN[(r["arm"], r["seed"])] = r["em_clean"]
        _em_src[(r["arm"], r["seed"])] = r["sha"]
# 交账前的自洽：44 的 A@12000 clean 必须等于 METRICS 的 95.45/90.52
for _s, _v in ((1234, 0.9545), (5678, 0.9052)):
    assert abs(EM_REF_12000_CLEAN[("A", _s)] - _v) < 5e-5, f"A@12000 入库值与 METRICS 不一致 {_s}"

EM_REF_6000_FULL = {                        # logs/e6_sat.log [MAIN-TABLE]（全量 n=800，仅对照）
    ("A", 1234): 0.8562, ("A", 5678): 0.7438,
    ("B", 1234): 0.0112, ("B", 5678): 0.0088,
    ("C", 1234): 0.0088, ("C", 5678): 0.0100,
    ("D", 1234): 0.0138, ("D", 5678): 0.0100,
    ("E", 1234): 0.0125, ("E", 5678): 0.0138,
}

# ============================================================================
# 6. 20 个 ckpt × (oracle, cards) 前向比对
# ============================================================================
CASES = []
for arm in ("A", "B", "C", "D", "E"):
    for seed in (1234, 5678):
        CASES.append(dict(tag=f"36@{arm}_s6000_seed{seed}", arm=arm, seed=seed, steps=6000,
                          ckpt=f"{ROOT}/logs/36_ckpt_{arm}_s6000_seed{seed}.pt",
                          oracle="36", style=("layer" if arm == "A" else "card")))
for arm in ("A", "B", "C", "D", "E"):
    for seed in (1234, 5678):
        if arm == "A":
            CASES.append(dict(tag=f"38@A_s12000_seed{seed}", arm=arm, seed=seed, steps=12000,
                              ckpt=f"{ROOT}/logs/38_ckpt_w1_s12000_seed{seed}.pt",
                              oracle="44A", style="thought"))
        else:
            CASES.append(dict(tag=f"44@{arm}_s12000_seed{seed}", arm=arm, seed=seed, steps=12000,
                              ckpt=f"{ROOT}/logs/44_ckpt_{arm}_s12000_seed{seed}.pt",
                              oracle="44", style="card"))


CASES = CASES[: int(os.environ.get("S49_LIMIT", len(CASES)))]     # 调试用：只跑前 N 个


def build_oracle(case: dict):
    """按 oracle 段原文本构造 stage 侧模型并 strict 装载裸 state_dict。"""
    raw = torch.load(case["ckpt"], map_location="cpu", weights_only=True)
    if case["oracle"] == "36":
        m = N36["Cards36"](case["arm"], D, FF)
        info = m.load_state_dict(raw, strict=True)
        return m, raw, info, None
    if case["oracle"] == "44":
        m = N44["Cards44"](case["arm"], D, FF)
        info = m.load_state_dict(raw, strict=True)
        return m, raw, info, None
    m = N44["Cards44"]("A", D, FF)
    mapped = N44["load_a_state"](case["ckpt"])          # thought. → card.layer.（44 原函数）
    info = m.load_state_dict(mapped, strict=True)
    return m, raw, info, mapped


@torch.no_grad()
def chunk_max(fa, fb, recs, chunk=CHUNK) -> float:
    """同一批 ids 下两条 logits 路径的 max|Δ|（分块，控内存）。"""
    worst = 0.0
    for i in range(0, len(recs), chunk):
        ids, _ = S["build_batch"](recs[i:i + chunk])
        worst = max(worst, float((fa(ids) - fb(ids)).abs().max().item()))
    return worst


@torch.no_grad()
def logits_chunk_max(oracle, pipe, recs, chunk=CHUNK) -> float:
    return chunk_max(oracle.logits, pipe.logits, recs, chunk)


@torch.no_grad()
def logits_b1_max(oracle, pipe, recs) -> float:
    worst = 0.0
    for r in recs:
        ids = torch.tensor([r["p"]], dtype=torch.long)
        worst = max(worst, float((oracle.logits(ids) - pipe.logits(ids)).abs().max().item()))
    return worst


ROWS = []
MV_BY_ARM = {}
for ci, case in enumerate(CASES, 1):
    t0 = time.time()
    oracle, raw, o_info, _ = build_oracle(case)
    oracle.eval()

    pipe = StructPipeline(case["arm"], D, V, FF, NHEAD, MAXLEN, PAD_ID, 0.1, True)
    mapped = map_state(raw, case["style"])
    p_info = pipe.load_state_dict(mapped, strict=True)
    pipe.eval()
    assert p_info.missing_keys == [] and p_info.unexpected_keys == [], \
        f"{case['tag']}: 键映射不齐 missing={p_info.missing_keys} unexpected={p_info.unexpected_keys}"
    assert o_info.missing_keys == [] and o_info.unexpected_keys == []

    # ---- 框架侧：契约一致 + 黑板路由 + MV（证明走的是 cards 路径而不是自己拼的网络）----
    _ = pipe.contract                                     # 三卡契约不一致会当场 assert
    _ = pipe.logits(torch.tensor([TEST[0]["p"]], dtype=torch.long))   # 走一遍黑板路由
    chain = [f"{a}->{b}" for a, b in pipe.who_reads_whom()]
    MV_BY_ARM.setdefault(case["arm"], pipe.mv().interface_sha256[:16])

    # ---- ① 整批 logits（分块，逐位）----
    oracle.eval(); pipe.eval()          # ★greedy_gen 末尾会 model.train()，每次前向比较前必须重设 eval
    d_all = logits_chunk_max(oracle, pipe, TEST)
    # ---- ② batch=1 逐样本 logits（与贪心主口径同形）----
    oracle.eval(); pipe.eval()
    d_b1 = logits_b1_max(oracle, pipe, TEST[:N_B1])
    # ---- ③ 贪心预测（batch=1，全 800）----
    txt_o = greedy_gen(oracle, TEST, batch=1)
    txt_p = greedy_gen(pipe, TEST, batch=1)
    same_txt = sum(a == b for a, b in zip(txt_o, txt_p))
    pred_o = [parse_ans(t) for t in txt_o]
    pred_p = [parse_ans(t) for t in txt_p]
    same_pred = sum(a == b for a, b in zip(pred_o, pred_p))
    # ---- ④ EM（S19 hits_from 口径；clean n=770）----
    hits_o = hits_from(txt_o, TEST)
    hits_p = hits_from(txt_p, TEST)
    so = [h["strict"] for h in hits_o]
    sp = [h["strict"] for h in hits_p]
    em_o = sum(so) / len(so)
    em_p = sum(sp) / len(sp)
    em_o_cl = sum(so[i] for i in CLEAN_IDX) / len(CLEAN_IDX)
    em_p_cl = sum(sp[i] for i in CLEAN_IDX) / len(CLEAN_IDX)
    ref = (EM_REF_6000_CLEAN if case["steps"] == 6000 else EM_REF_12000_CLEAN)[(case["arm"], case["seed"])]
    ref_full = EM_REF_6000_FULL.get((case["arm"], case["seed"])) if case["steps"] == 6000 else None

    # ---- A@12000 额外：oracle 自身两条路自洽（★注意 greedy_gen 末尾 model.train()）----
    xchk = None
    if case["oracle"] == "44A":
        # 第二条 oracle（同为原文本）：S19 自己的 res 臂容器 Cards(identity=False, residual=True)
        # —— 键名 emb/in_enc/thought/head 与 38 ckpt 严格对应（Cards38Clone 正是它的 init 等价物，
        #    44 的 [S44-INIT] 已证），故这是「A@12000 = S19-res」的独立前向复核。
        c19 = S["Cards"](D, FF, identity=False, residual=True)
        c19.load_state_dict(raw, strict=True)
        c19.eval()
        oracle.eval()                   # ★greedy_gen 末尾 model.train() ⇒ 不重设 eval 会带上 dropout
        xchk = chunk_max(oracle.logits, c19.logits, TEST)

    row = dict(tag=case["tag"], arm=case["arm"], seed=case["seed"], steps=case["steps"],
               ckpt=os.path.basename(case["ckpt"]), oracle=case["oracle"], style=case["style"],
               n_ckpt_tensors=len(raw), n_mapped=len(mapped),
               missing=len(p_info.missing_keys), unexpected=len(p_info.unexpected_keys),
               contract_sha16=pipe.contract.sha256()[:16], mv16=MV_BY_ARM[case["arm"]],
               route_chain=str(chain), d_logits_chunk=d_all, d_logits_b1=d_b1,
               pred_same=same_pred, txt_same=same_txt, n=len(TEST),
               em_full_cards=em_p, em_full_oracle=em_o, em_full_ref=ref_full,
               em_clean_cards=em_p_cl, em_clean_oracle=em_o_cl, em_clean_ref=ref,
               d_em_vs_oracle_pp=(em_p_cl - em_o_cl) * 100,
               d_em_vs_ref_pp=(em_p_cl - ref) * 100, oracle_self_delta=xchk,
               wall_s=time.time() - t0)
    ROWS.append(row)
    print(f"[S49-CASE {ci:2d}/{len(CASES)}] {row['tag']:22s} n_t={row['n_ckpt_tensors']:2d} "
          f"missing={row['missing']} unexpected={row['unexpected']} | "
          f"logits max|Δ| 整批={row['d_logits_chunk']:.3e} b1={row['d_logits_b1']:.3e} | "
          f"pred 同={row['pred_same']}/{row['n']} 文本同={row['txt_same']}/{row['n']} | "
          f"EM_clean cards={em_p_cl*100:.2f}% oracle={em_o_cl*100:.2f}% 入库={ref*100:.2f}% "
          f"Δ入库={row['d_em_vs_ref_pp']:+.2f}pp | {row['wall_s']:.1f}s", flush=True)

# ============================================================================
# 6b. MV 独立性（P4_PLAN 风险②的精确检验）：MV/接口快照只覆盖 input+output
#     ⇒ 同一份接口权重下把 target 从 A 换成 B，MV 必须不变（target 权重确实不同）
# ============================================================================
_rA = torch.load(f"{ROOT}/logs/36_ckpt_A_s6000_seed1234.pt", map_location="cpu", weights_only=True)
_rB = torch.load(f"{ROOT}/logs/36_ckpt_B_s6000_seed1234.pt", map_location="cpu", weights_only=True)
_pa = StructPipeline("A", D, V, FF, NHEAD, MAXLEN, PAD_ID, 0.1, True)
_pa.load_state_dict(map_state(_rA, "layer"), strict=True)
_pb = StructPipeline("B", D, V, FF, NHEAD, MAXLEN, PAD_ID, 0.1, True)
_pb.load_state_dict(map_state(_rB, "card"), strict=True)
_pb.input.load_state_dict(_pa.input.state_dict())
_pb.output.load_state_dict(_pa.output.state_dict())
_sha_ta = state_sha256(_pa.target.state_dict())
_sha_tb = state_sha256(_pb.target.state_dict())
_td = (_sha_ta != _sha_tb)          # target 卡确实不同（A 12 张量 / B 6 张量，键名也不同）
_nt = (len(_pa.target.state_dict()), len(_pb.target.state_dict()))
MV_SAME_IO = (_pa.interface_sha256() == _pb.interface_sha256()
              and _pa.mv().sha256() == _pb.mv().sha256())
print(f"[S49-MV] 同接口权重（A 的 input/output）+ target A→B（target 权重 sha 不同 = {_td}，"
      f"张量数 {_nt[0]}→{_nt[1]}）⇒ interface_sha256/MV 同 = {MV_SAME_IO}"
      f"（MV 不认卡内结构，P4_PLAN 风险②）", flush=True)

# ============================================================================
# 7. 判定（判据写死）
# ============================================================================
all_zero = all(r["d_logits_chunk"] == 0.0 and r["d_logits_b1"] == 0.0 for r in ROWS)
all_pred = all(r["pred_same"] == r["n"] == r["txt_same"] for r in ROWS)
all_em_oracle = all(abs(r["d_em_vs_oracle_pp"]) == 0.0 for r in ROWS)
worst_ref = max(abs(r["d_em_vs_ref_pp"]) for r in ROWS)
em_ok = worst_ref <= EM_TOL * 100
bad = [r["tag"] for r in ROWS
       if not (r["d_logits_chunk"] == 0.0 and r["d_logits_b1"] == 0.0
               and r["pred_same"] == r["n"] == r["txt_same"])]

print(f"\n[S49] ===== ★门1 判定（写死：logits max|Δ|==0 且 pred 逐样本同）=====", flush=True)
print(f"[S49-JUDGE] {len(ROWS)}/{len(CASES)} ckpt：整批+batch1 logits max|Δ| 全 0.0 = {all_zero} | "
      f"pred 逐样本同（且逐字同）= {all_pred} | EM(clean) vs oracle CPU 逐位同 = {all_em_oracle} | "
      f"vs 入库最大 |Δ| = {worst_ref:.2f}pp（判据 ≤0.13pp）= {em_ok}", flush=True)
print(f"[S49-JUDGE] 键映射：{len(ROWS)}/{len(CASES)} 张量数吻合、missing=0、unexpected=0 | "
      f"映射表 = {MAP_COMMON} || " + " || ".join(MAP_DESC[k] for k in ("layer", "thought", "card")),
      flush=True)
print(f"[S49-JUDGE] 框架侧：三卡契约一致（无 assert 抛）+ 黑板路由链 "
      f"{ROWS[0]['route_chain']}（多次路由同形）| MV 独立性 = {MV_SAME_IO}", flush=True)
_a = [r for r in ROWS if r["oracle_self_delta"] is not None]
if _a:
    print(f"[S49-JUDGE] A@12000 第二条 oracle 自洽：Cards44-A(load_a_state) vs S19-res "
          f"Cards(residual=True) logits max|Δ| = "
          f"{max(r['oracle_self_delta'] for r in _a):.3e}（0.0 ⇒ 38 ckpt 确实是 A/S19-res 同一图）",
          flush=True)
print(f"[S49-VERDICT] 门1 = {'★通过' if (all_zero and all_pred and all_em_oracle and em_ok) else '★不通过'} "
      f"| 若不过，第一处分岔的第一条 = {bad[:1] if bad else '无'}", flush=True)
print("[S49-NOTE] ★门1 只在 CPU 上做：CPU 逐位一致 ≠ 与 GPU 训练逐位一致（§7.12 / R34）；"
      "EM 与入库的 ≤0.13pp 是 CPU/GPU 浮点同量级的口径差。", flush=True)

out = dict(device="cpu", torch=torch.__version__, threads=torch.get_num_threads(),
           exec_head=head_span, exec_36=[s36a, s36b], exec_44=[s44a, s44b],
           test_sha16=TEST_SHA, leak=LEAK, bad_expr=len(BAD_IDX), clean_n=len(CLEAN_IDX),
           map_common=MAP_COMMON, map_desc=MAP_DESC, fp=FP, chunk=CHUNK, n_b1=N_B1,
           mv_by_arm=MV_BY_ARM, mv_same_io=MV_SAME_IO, mv_target_differs=_td, mv_target_ntensors=list(_nt),
           all_logits_zero=all_zero, all_pred_same=all_pred, all_em_vs_oracle_zero=all_em_oracle,
           worst_em_vs_ref_pp=worst_ref, em_tol_pp=EM_TOL * 100, em_ok=em_ok,
           gate1_pass=bool(all_zero and all_pred and all_em_oracle and em_ok),
           rows=ROWS, wall_s=round(time.time() - T0, 1))
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"[META] 总墙钟={out['wall_s']}s | 20 ckpt × 2 路径 | JSON={JSON_OUT} | [DONE] exit=0",
      flush=True)
