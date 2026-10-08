#!/usr/bin/env python3
"""S35 · E1 ★同参数量同算力下，「三类卡闭环」相比「单体 Transformer」到底有没有增量？

唯一问题（本项目最该回答的一问：判断"卡片化"值不值得继续）：
  现状 B = 三类卡闭环：输入卡(emb+in_enc) → 模因[n,d] → 目标卡(thought) → 模因[n,d] → 输出卡(head)
            forward: h = in_enc(x); h = h + thought(h); logits = head(h)   （S19-res 臂，带跨卡残差）
  对照 A = 单体 Transformer：把三卡合并成一条无卡边界的通路，同一个 head / 同一份 embedding /
            同一层结构（2 层 pre-LN TransformerEncoderLayer），去掉卡模块与卡边界：
            forward: h = enc.layers[0](x); h = enc.layers[1](h); logits = head(h)
            ⇒ 参数量与 B 逐位相同（2,501,888，差额 0，0.0000% ≤ 1%）

★ 归因诚实条款（写在这里，报告里必须原样报）：
  「只去掉卡边界、其余一切不动」在固定参数量下**没有可训练的自由度** —— 卡只是把同一张计算图
  拆成三个模块。本文件用一条**实测等价证据**把它钉死：把 B 的权重装进 A 那种「单体容器」
  （同一个 nn.TransformerEncoder 的 2 层），并保留跨卡残差 ⇒ 与 B 的 logits 逐位相同
  （max|Δlogits| 必须 == 0.0）。⇒ 卡边界本身的增量 ≡ 0（构造性），
  因此 A 取「普通 2 层 Transformer」这一**参数量匹配的对照模型**（这正是题面给的
  emb → in_enc → Think → head 通路），本单元实测到的 A vs B 差距，只能归因于
  「跨卡残差」这一条计算图差异（= S19 的唯一变量），不能归因于"卡片化/卡边界"。

步数 {3000, 6000}（S28/E5 已证 3000 步欠训）× 2 seed（1234, 5678）× 2 臂 = 8 run。
其余与 S19-res 逐字相同：d=128 ff=512 NHEAD=4 lr=1e-3 batch=32 MAXLEN=512(prompt 优先)、
思维卡带残差（B 臂）。数据只用 add_3d（train4000/test800）。

★ 口径自检（缺一即无效）：
  1 EM 走 batch=1（R28），并报 batch=1 vs 16 逐字一致性自检；
  2 SEEN/UNSEEN 分层 EM（泄漏：add_3d 的 (a,b) 数对重叠）；
  3 本桶自己的「多数答案地板」（add_3d 期望 ≈0.2%，不许借用 1.89%）；
  4 选步必须有 dev 集：从 train 切 10%（400 条）做 dev，最佳步只在 dev 上选，绝不在 test 上选；
  5 参数量逐项对齐表（A vs B 的分组与差额）；
  6 R29 任何干预的 Δ 非零比例；TEST 指纹 sha256[:16] 必须 == b34e7ea515203227。
  ★ 口径 4 的代价如实报：切出 dev ⇒ 训练集 = 3600（S19 是 4000），因此 B@3000 与 S19-res
    锚点 37.25/31.13 **不是逐位可复现**，只作参考对照并报偏离量。

判据（写死，分开报）：Δ = EM(A) − EM(B)，逐 test 样本配对 ±SE（配对 d_i ∈ {−1,0,1}）：
  |Δ| ≤ 1SE 且 2 seed 同号 ⇒ 单体不差 ⇒ 卡框架零增量；
  Δ ≥ 2SE 且 2 seed 同号 ⇒ 闭环有增量；
  Δ ≤ −2SE 且 2 seed 同号 ⇒ 单体更好 ⇒ 卡框架是负增量；
  并报 3000/6000 两臂 EM 曲线（是否饱和；都还在升 ⇒ 明确写"不能外推"）。

ckpt：logs/35_ckpt_{mono,cards}_s<步数>_seed<s>.pt（报 sha256 前16）。
结果增量落 logs/35_results.jsonl。只允许写：本文件、logs/e1_mono.log、logs/35_*、/tmp。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
FLOWME = "/home/vesita/coding/my/flowme"
LOGDIR = FLOWME + "/logs"
sys.path.insert(0, NS_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

# ---------------- device 纪律（ROCm 下报成 cuda，正常） ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
if device != "cuda":
    print("GPU 起不来，立即停（不用 CPU 硬跑）。", flush=True)
    print("torch.cuda.is_available() =", torch.cuda.is_available(), flush=True)
    print("HSA_OVERRIDE_GFX_VERSION =", os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)
    sys.exit(3)
print("torch:", torch.__version__, "| HSA_OVERRIDE_GFX_VERSION =",
      os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)

# ---------------- 配置（与 S19-res / S25 / S28 / S34 逐字一致） ----------------
D, FF, NHEAD, MAXLEN = 128, 512, 4, 512
MIN_T, TAIL_KEEP = 48, 24
LR, BATCH, GEN_BATCH, EM_BATCH, CHK_BATCH = 1e-3, 32, 32, 1, 16
N_TRAIN, N_TEST, MAX_GEN = 4000, 800, 48
BUCKET, BI = "add_3d", 2
DEV_FRAC = 0.10
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
STEPS_LIST = (3000, 6000)
SEEDS = (1234, 5678)
ARMS = ("cards", "mono")          # cards = B（三类卡闭环）；mono = A（单体 Transformer）
S19_RES_ANCHOR = {1234: 0.3725, 5678: 0.3113}     # S19 add_3d res 臂 @3000（只读引用）
TEST_SHA_EXPECT = "b34e7ea515203227"              # S31/S34 入库指纹
LEAK_EXPECT = 16                                  # S31/S34 入库 (a,b) 泄漏（对 train4000）

SMOKE = os.environ.get("S35_SMOKE") == "1"
if SMOKE:
    STEPS_LIST, SEEDS = (200,), (1234,)

CKPT_TMPL = LOGDIR + "/35_ckpt_{arm}_s{steps}_seed{seed}.pt"
RESULT_JSONL = "/tmp/35_results_smoke.jsonl" if SMOKE else LOGDIR + "/35_results.jsonl"

t_start = time.time()
print(f"[S35] ★E1 单体(A) vs 三卡闭环(B)：桶={BUCKET} 步数={STEPS_LIST} seeds={SEEDS} "
      f"arms={ARMS} | d={D} ff={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN}"
      f"(prompt优先) | EM主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | smoke={SMOKE}", flush=True)

tok = Tokenizer.from_file(TOK)
V = tok.get_vocab_size()
PAD_ID = tok.token_to_id("<pad>")
EOS_ID = tok.token_to_id("<eos>")


def enc(t: str) -> list[int]:
    return tok.encode(t, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


# ============================================================================
# ① 数据（与 S14/S19/S25/S28/S34 逐字相同；只用 add_3d；同种子 ⇒ 同数据）
# ============================================================================
def build_templates() -> list[tuple[str, bool]]:
    ts: list[tuple[str, bool]] = []
    tails_cn = ["等于多少？", "是多少？", "等于几？", "得多少？", "的结果是多少？",
                "的结果是几？", "的值是多少？", "等于多少", "是多少", "是几"]
    for lead in ("", "请", "帮我", "麻烦"):
        for verb in ("计算", "算出", "算一下", "口算", "快速算", "算"):
            for t in tails_cn:
                ts.append((f"{lead}{verb} {{expr}} {t}", True))
    for t in tails_cn:
        ts.append(("{{expr}} " + t, True))
    for q in ("What is {expr}?", "{expr} equals what?", "How much is {expr}?",
              "What is {expr} equal to?", "{expr} is what?", "What does {expr} make?",
              "What is the result of {expr}?", "{expr} equals how much?",
              "How many is {expr}?", "Tell me {expr}", "I need the answer to {expr}",
              "Do you know {expr}?"):
        ts.append((q, False))
    for lead in ("", "Please ", "Could you ", "Quick: "):
        for verb in ("Calculate", "Compute", "Work out", "Find", "Solve", "Figure out"):
            v = verb if lead == "" else verb.lower()
            for tail in ("", " for me", " please", "."):
                ts.append((f"{lead}{v} {{expr}}{tail}", False))
    assert len({t for t, _ in ts}) == len(ts), "模板有重复"
    return ts


TEMPLATES = build_templates()
CN_OPS, EN_OPS, LOHI = {"add_3d": ("加", "加上")}, {"add_3d": ("plus",)}, {"add_3d": (100, 999)}


class Rng:
    def __init__(self, seed: int):
        self._r = random.Random(seed)

    def randrange(self, n): return self._r.randrange(n)
    def randint(self, a, b): return self._r.randint(a, b)
    def random(self): return self._r.random()
    def shuffle(self, xs): self._r.shuffle(xs)


def answer_values(name):
    lo, hi = LOHI[name]
    return list(range(2 * lo, 2 * hi + 1))


def split_by_answer(name, y, rng):
    lo, hi = LOHI[name]
    a = rng.randint(max(lo, y - hi), min(hi, y - lo))
    return a, y - a


def render(name, a, b, y, rng):
    tpl, cjk = TEMPLATES[rng.randrange(len(TEMPLATES))]
    op = (CN_OPS[name] if cjk else EN_OPS[name])
    op = op[rng.randrange(len(op))]
    tight = cjk and rng.random() < 0.5
    expr = f"{a}{op}{b}" if tight else f"{a} {op} {b}"
    nl = tpl.format(expr=expr)
    return f"题干：{nl} → ", f"#### {y}", nl


def gen_half(name, n, rng, used, tag):
    ys = answer_values(name)[:]
    rng.shuffle(ys)
    items, i, tries = [], 0, 0
    while len(items) < n:
        y = ys[i % len(ys)]
        i += 1
        tries += 1
        if tries > n * 500:
            raise RuntimeError(f"{name}{tag}: 文本空间不足")
        a, b = split_by_answer(name, y, rng)
        prompt, target, nl = render(name, a, b, y, rng)
        full = prompt + target
        if full in used:
            continue
        used.add(full)
        items.append(dict(prompt=prompt, target=target, a=a, b=b, y=y, nl=nl))
    return items


def encode_record(it):
    p_full, t_full = enc(it["prompt"]), enc(it["target"] + "<eos>")
    cut_p = cut_t = False
    if len(p_full) + len(t_full) > MAXLEN:
        p_room = MAXLEN - MIN_T
        if len(p_full) > p_room:
            p_ids = p_full[: p_room - TAIL_KEEP] + p_full[-TAIL_KEEP:]
            cut_p = True
        else:
            p_ids = p_full
        room = MAXLEN - len(p_ids)
        if len(t_full) <= room:
            t_ids = t_full
        else:
            t_ids = t_full[-room:]
            cut_t = True
    else:
        p_ids, t_ids = p_full, t_full
    assert len(p_ids) + len(t_ids) <= MAXLEN
    return dict(p=p_ids, t=t_ids, t_full=t_full, gold=it["target"][5:].strip(),
                cut_p=cut_p, cut_t=cut_t, a=it["a"], b=it["b"], y=it["y"], nl=it["nl"])


_tr_rng, _te_rng = Rng(14000 + BI * 7), Rng(14900 + BI * 7)
_used: set[str] = set()
TRAIN_ALL = [encode_record(x) for x in gen_half(BUCKET, N_TRAIN, _tr_rng, _used, "train")]
TEST = [encode_record(x) for x in gen_half(BUCKET, N_TEST, _te_rng, _used, "test")]
assert len(_used) == len(TRAIN_ALL) + len(TEST), "train/test 文本池计数异常（应零重叠）"

_h = hashlib.sha256()
for r in TEST:
    _h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
TEST_SHA = _h.hexdigest()[:16]
_tp_all = {(r["a"], r["b"]) for r in TRAIN_ALL}
LEAK_AB_ALL = sum(1 for r in TEST if (r["a"], r["b"]) in _tp_all)
_yv = answer_values(BUCKET)
print(f"[S35-DATA] {BUCKET}: train={len(TRAIN_ALL)} test={len(TEST)} 文本零重叠 | "
      f"答案域={min(_yv)}..{max(_yv)}({len(_yv)}个) | target中位="
      f"{sorted(len(r['t_full']) for r in TRAIN_ALL)[len(TRAIN_ALL)//2]}tok", flush=True)
sha_ok = TEST_SHA == TEST_SHA_EXPECT
leak_ok = LEAK_AB_ALL == LEAK_EXPECT
print(f"[S35-DATA-CHECK] TEST 指纹 sha256[:16]={TEST_SHA} vs 入库 {TEST_SHA_EXPECT} ⇒ "
      f"{'逐位一致 ✓' if sha_ok else '★不一致'} | (a,b) 泄漏对 train4000={LEAK_AB_ALL}/800 vs "
      f"入库 {LEAK_EXPECT} ⇒ {'一致 ✓' if leak_ok else '★不一致'}", flush=True)
assert sha_ok and leak_ok, "数据指纹不符 ⇒ 与 S19/S25/S28/S31/S34 不是同一数据集，本单元无效"

# ---- ★口径 4：dev 集（从 train 切 10%）— 只在 dev 上选步，绝不在 test 上选步 ----
_idx = list(range(len(TRAIN_ALL)))
random.Random(3500).shuffle(_idx)
N_DEV = int(round(DEV_FRAC * len(TRAIN_ALL)))
DEV = [TRAIN_ALL[i] for i in _idx[-N_DEV:]]
TRAIN = [TRAIN_ALL[i] for i in _idx[:-N_DEV]]
print(f"[S35-DEV] 口径4：train{len(TRAIN_ALL)} ⇒ 训练 {len(TRAIN)} + dev {len(DEV)} "
      f"(10%, random.Random(3500) 确定性切分) | ★选步只用 dev，test 只报不看 ⇒ "
      f"训练集 {len(TRAIN)} ≠ S19 的 4000（锚点非逐位复现，见报告）", flush=True)

# ---- ★口径 2：SEEN/UNSEEN 分层（按**实际训练集** 3600 判重叠）----
_tp_tr = {(r["a"], r["b"]) for r in TRAIN}
_tp_dev = {(r["a"], r["b"]) for r in DEV}
for r in TEST:
    r["seen"] = (r["a"], r["b"]) in _tp_tr
    r["seen_dev_only"] = ((r["a"], r["b"]) in _tp_dev) and (not r["seen"])
N_SEEN = sum(1 for r in TEST if r["seen"])
LEAK_TR = sum(1 for r in TEST if r["seen"])
N_DEVONLY = sum(1 for r in TEST if r["seen_dev_only"])
print(f"[S35-SEEN] 口径2：test 中 (a,b) 在训练集3600 内 = {LEAK_TR}/{len(TEST)} "
      f"({100*LEAK_TR/len(TEST):.2f}%) ⇒ SEEN={N_SEEN} / UNSEEN={len(TEST)-N_SEEN} | "
      f"仅在 dev 内(未训练)={N_DEVONLY}（按 UNSEEN 算）", flush=True)

# ---- ★口径 3：本桶自己的多数答案地板（不许借用 1.89%）----
_c_tr = Counter(r["gold"] for r in TRAIN)
_top1, _top1n = _c_tr.most_common(1)[0]
_top5 = [v for v, _ in _c_tr.most_common(5)]
FLOOR_1 = sum(1 for r in TEST if r["gold"] == _top1) / len(TEST)
FLOOR_5 = sum(1 for r in TEST if r["gold"] in _top5) / len(TEST)
_c_all = Counter(r["gold"] for r in TRAIN_ALL)
_top5a = [v for v, _ in _c_all.most_common(5)]
FLOOR_5_ALL = sum(1 for r in TEST if r["gold"] in _top5a) / len(TEST)
print(f"[S35-FLOOR] 口径3：{BUCKET} 多数答案地板：单答案 '{_top1}'(train3600 {_top1n}/{len(TRAIN)}) "
      f"覆盖 test {FLOOR_1*100:.2f}% | top5={_top5} 覆盖 test {FLOOR_5*100:.2f}% "
      f"⇒ ★本地板(实际训练集3600)={FLOOR_1*100:.2f}%/机会水平={FLOOR_5*100:.2f}%"
      f"（不借用其它桶的 1.89%）", flush=True)
print(f"[S35-FLOOR] 口径3 交叉核对：同口径算在 train4000 上 top5={_top5a} 覆盖 test "
      f"{FLOOR_5_ALL*100:.2f}% ⇒ 与 S19/S34 入库的 0.2% "
      f"{'一致（≈0，噪声级）' if abs(FLOOR_5_ALL-0.002) <= 0.006 else '★偏离'}；"
      f"分母 800 ⇒ 0.2% 只是 1–2 条样本的噪声", flush=True)


# ============================================================================
# ② 模型：B = 三类卡（三个模块 + 跨卡残差）；A = 单体（一个 nn.TransformerEncoder）
# ============================================================================
def sin_pe(max_len, d):
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def enc_layer(d, ff):
    return nn.TransformerEncoderLayer(d, NHEAD, ff, dropout=0.1, activation="gelu",
                                      batch_first=True, norm_first=True)


def make_mask(ids):
    n = ids.size(1)
    m = torch.full((n, n), float("-inf"), device=ids.device)
    m = torch.triu(m, diagonal=1)
    m = m.unsqueeze(0).expand(ids.size(0), -1, -1).clone()
    m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
    return m.repeat_interleave(NHEAD, dim=0)


class Cards(nn.Module):
    """B · 三类卡闭环：输入卡(emb+in_enc) → 模因[n,d] → 目标卡(thought) → 模因[n,d] → 输出卡(head)。
    与 S19-res 逐字相同：h = in_enc(x); h = h + thought(h)（跨卡残差）。"""

    def __init__(self, d, ff):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        self.thought = enc_layer(d, ff)
        self.head = nn.Linear(d, V)
        self.register_buffer("pe", sin_pe(MAXLEN, d), persistent=False)

    def logits(self, ids):
        m = make_mask(ids)
        pe = self.pe.to(ids.device)
        x = self.emb(ids) + pe[: ids.size(1)].unsqueeze(0)
        h = self.in_enc(x, src_mask=m)            # 输入卡 → 模因
        h = h + self.thought(h, src_mask=m)       # 模因 → (跨卡残差) 目标卡 → 模因
        return self.head(h)                       # 输出卡


class Mono(nn.Module):
    """A · 单体 Transformer：一条无卡边界的通路 emb → 2 层 TransformerEncoder → head。
    与 B 参数逐位同形（同一份 embedding/head/层结构 ⇒ 参数量差 0）。
    cross_skip=True 时保留跨卡残差 ⇒ 用于「卡边界零增量」的逐位等价实测。"""

    def __init__(self, d, ff, cross_skip: bool = False):
        super().__init__()
        self.d = d
        self.cross_skip = cross_skip
        self.emb = nn.Embedding(V, d)
        # ★按与 Cards 完全相同的顺序建同一批参数（emb → 第1层 → 第2层 → head），
        #   容器只是一个平铺的 ModuleList（无卡模块、无卡边界）⇒ 同 seed 初始化逐位相同。
        #   （不能用 nn.TransformerEncoder(...)：它 _get_clones 复制同一层 ⇒ 两层初始权重相同、
        #    RNG 消耗也不同，那会引入"初始化不同"这个额外混淆变量。）
        self.enc = nn.ModuleList([enc_layer(d, ff) for _ in range(2)])
        self.head = nn.Linear(d, V)
        self.register_buffer("pe", sin_pe(MAXLEN, d), persistent=False)

    def logits(self, ids):
        m = make_mask(ids)
        pe = self.pe.to(ids.device)
        x = self.emb(ids) + pe[: ids.size(1)].unsqueeze(0)
        h = self.enc[0](x, src_mask=m)
        if self.cross_skip:
            h = h + self.enc[1](h, src_mask=m)
        else:
            h = self.enc[1](h, src_mask=m)
        return self.head(h)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


def param_groups(model, arm):
    c = {}
    if arm == "cards":
        for n, p in model.named_parameters():
            if n.startswith("emb."):
                c["输入卡(emb)"] = c.get("输入卡(emb)", 0) + p.numel()
            elif n.startswith("in_enc."):
                c["输入卡(in_enc)"] = c.get("输入卡(in_enc)", 0) + p.numel()
            elif n.startswith("thought."):
                c["目标卡(thought)"] = c.get("目标卡(thought)", 0) + p.numel()
            elif n.startswith("head."):
                c["输出卡(head)"] = c.get("输出卡(head)", 0) + p.numel()
    else:
        for n, p in model.named_parameters():
            if n.startswith("emb."):
                c["单体(emb)"] = c.get("单体(emb)", 0) + p.numel()
            elif n.startswith("enc.0."):
                c["单体(enc.0)"] = c.get("单体(enc.0)", 0) + p.numel()
            elif n.startswith("enc.1."):
                c["单体(enc.1)"] = c.get("单体(enc.1)", 0) + p.numel()
            elif n.startswith("head."):
                c["单体(head)"] = c.get("单体(head)", 0) + p.numel()
    return c


def build(arm, seed, cross_skip=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    m = Mono(D, FF, cross_skip=cross_skip) if arm == "mono" else Cards(D, FF)
    return m.to(device)


# ---- ★口径 5：参数量逐项对齐表 + 同 seed 初始化逐位相同 ----
_pB = Cards(D, FF)
_pA = Mono(D, FF)
_NB, _NA = n_params(_pB), n_params(_pA)
_gB, _gA = param_groups(_pB, "cards"), param_groups(_pA, "mono")
print(f"\n[S35-PARAMS] 口径5 · 参数量逐项对齐表（A vs B）", flush=True)
print(f"[S35-PARAMS] 组 | B 三卡闭环 | A 单体 | 差额", flush=True)
_pairs = (("输入卡", "输入卡(emb)", "单体(emb)"), ("输入卡", "输入卡(in_enc)", "单体(enc.0)"),
          ("目标卡", "目标卡(thought)", "单体(enc.1)"), ("输出卡", "输出卡(head)", "单体(head)"))
_gB2 = {"输入卡": _gB["输入卡(emb)"] + _gB["输入卡(in_enc)"], "目标卡": _gB["目标卡(thought)"],
        "输出卡": _gB["输出卡(head)"]}
_gA2 = {"输入卡": _gA["单体(emb)"] + _gA["单体(enc.0)"], "目标卡": _gA["单体(enc.1)"],
        "输出卡": _gA["单体(head)"]}
for k in ("输入卡", "目标卡", "输出卡"):
    print(f"[S35-PARAMS] {k} | {_gB2[k]:,} | {_gA2[k]:,} | {_gB2[k]-_gA2[k]:+,}", flush=True)
print(f"[S35-PARAMS] 总计 | {_NB:,} | {_NA:,} | {_NB-_NA:+,} "
      f"({100.0*abs(_NB-_NA)/_NB:.4f}%) ⇒ {'✓ ≤1% 对齐' if abs(_NB-_NA) <= 0.01*_NB else '★未对齐'}",
      flush=True)
print(f"[S35-PARAMS] 细目 B: {_gB} | A: {_gA}", flush=True)

# 同 seed 初始化逐位相同（两臂唯一差异只能是计算图）
_m0 = build("cards", 999).cpu()
_sd0 = {k: v.clone() for k, v in _m0.state_dict().items()}
_m1 = build("mono", 999).cpu()
_sd1 = _m1.state_dict()
_map = {k: k.replace("in_enc.", "enc.0.").replace("thought.", "enc.1.") for k in _sd0}
_dmax = max(float((_sd0[k] - _sd1[_map[k]]).abs().max()) for k in _sd0)
print(f"[S35-INIT] 同 seed(999) 下两臂初始化逐参数最大差 = {_dmax:.3e} ⇒ "
      f"{'✓ 逐位相同（起点一致，唯一差异=计算图）' if _dmax == 0.0 else '★起点不同'}",
      flush=True)

# ★ 卡边界零增量的实测等价证据：把 B 的三卡权重装进**单体容器**
#   （一个 nn.TransformerEncoder 的 2 层）并保留跨卡残差 ⇒ logits 必须逐位相同
_pB = _pB.cpu()
_pB.eval()
_enc = nn.TransformerEncoder(enc_layer(D, FF), num_layers=2, enable_nested_tensor=False)
_enc.layers = nn.ModuleList([_pB.in_enc, _pB.thought])   # 同一个单体容器，直接复用三卡的层
_n = max(len(TEST[i]["p"]) + len(TEST[i]["t"]) for i in range(16))
_ids = torch.full((16, _n), PAD_ID, dtype=torch.long)
for _i in range(16):
    _row = TEST[_i]["p"] + TEST[_i]["t"]
    _ids[_i, : len(_row)] = torch.tensor(_row)
with torch.no_grad():
    _m = make_mask(_ids).cpu()
    _x = _pB.emb(_ids) + _pB.pe[: _n].unsqueeze(0)
    _h = _enc.layers[0](_x, src_mask=_m)
    _hf = _enc.layers[1](_h, src_mask=_m)          # 单体容器：只有平铺的 2 层
    _lb = _pB.logits(_ids)
    _lp = _pB.head(_h + _hf)                       # 保留跨卡残差 ⇒ 与三卡图逐位同一
    EQ_MAX = float((_lb - _lp).abs().max())
    EQ_BITEQ = bool(torch.equal(_lb, _lp))
print(f"[S35-EQUIV] ★卡边界零增量实测：把 B 的三卡权重装进 A 那种单体容器"
      f"（同一 nn.TransformerEncoder 的 2 层）并保留跨卡残差 ⇒ 16 样本 logits "
      f"max|Δ|={EQ_MAX:.3e} 逐位相同={EQ_BITEQ} ⇒ "
      f"{'卡边界本身既无参数也无算子（构造性增量 ≡ 0）' if EQ_BITEQ else '★不等价'}",
      flush=True)
print(f"[S35-EQUIV] ⇒ 因此 A 取「普通 2 层 Transformer」(无跨卡残差，= 题面 emb→in_enc→Think→head "
      f"通路)：本单元实测的 A vs B 差距只能归因于【跨卡残差】这一条图差异（=S19 唯一变量），"
      f"不能归因于卡片化/卡边界本身。", flush=True)
del _pA, _pB, _enc, _m0, _m1
torch.cuda.empty_cache()


# ============================================================================
# ③ 训练 / 评估（与 S19 逐字同构）
# ============================================================================
def build_batch(recs):
    lens = [len(r["p"]) + len(r["t"]) for r in recs]
    n = max(lens)
    ids = torch.full((len(recs), n), PAD_ID, dtype=torch.long)
    s = torch.zeros(len(recs), dtype=torch.long)
    for i, r in enumerate(recs):
        ids[i, : len(r["p"])] = torch.tensor(r["p"])
        ids[i, len(r["p"]): len(r["p"]) + len(r["t"])] = torch.tensor(r["t"])
        s[i] = len(r["p"])
    return ids.to(device), s


def masked_ce(model, recs):
    ids, s = build_batch(recs)
    logits = model.logits(ids)
    logp = torch.log_softmax(logits, dim=-1)
    nll = torch.zeros((), device=device)
    ntok = 0
    for i, r in enumerate(recs):
        e = s[i] + len(r["t"])
        lp = logp[i, s[i] - 1: e - 1]
        tgt = ids[i, s[i]: e]
        nll = nll - lp.gather(1, tgt.unsqueeze(1)).sum()
        ntok += len(r["t"])
    return nll / ntok


def train_run(model, train, seed, steps, hooks: dict):
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0 = 0, 0, time.time()
    last = 0.0
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(model, [train[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step in hooks:
            hooks[step](model, step, loss.item(), time.time() - t0)
        if step % 500 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train] step={step}/{steps} loss={loss.item():.4f} elapsed={el:.0f}s "
                  f"s/step={(el - last) / (step % 500 or 500):.3f}", flush=True)
            last = el
    return time.time() - t0


@torch.no_grad()
def greedy_gen(model, recs, batch: int = 1) -> list[str]:
    model.eval()
    texts: list[str] = [""] * len(recs)
    for i in range(0, len(recs), batch):
        chunk = recs[i: i + batch]
        lens = [len(r["p"]) for r in chunk]
        n0 = max(lens)
        ids = torch.full((len(chunk), n0), PAD_ID, dtype=torch.long)
        for j, r in enumerate(chunk):
            ids[j, : lens[j]] = torch.tensor(r["p"])
        ids = ids.to(device)
        gen_len = [0] * len(chunk)
        cap = [min(MAXLEN - L, MAX_GEN) for L in lens]
        done = [False] * len(chunk)
        last_col = [L - 1 for L in lens]
        for _ in range(max(cap)):
            if all(done):
                break
            logits = model.logits(ids)
            nxt = []
            for j in range(len(chunk)):
                if done[j]:
                    nxt.append(PAD_ID)
                    continue
                t = int(logits[j, last_col[j]].argmax().item())
                nxt.append(t)
                gen_len[j] += 1
                if t == EOS_ID or gen_len[j] >= cap[j]:
                    done[j] = True
                last_col[j] = n0 + gen_len[j] - 1
            ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long, device=device).unsqueeze(1)], 1)
        for j in range(len(chunk)):
            g_ids = [x for x in ids[j, n0: n0 + gen_len[j]].tolist() if x != EOS_ID]
            texts[i + j] = dec(g_ids)
    model.train()
    return texts


def parse_ans(text: str):
    if "####" not in text:
        return None
    a = text.rsplit("####", 1)[1].split("\n")[0]
    a = a.strip().replace(",", "").replace(" ", "").replace("$", "")
    return a or None


def hits_from(texts, recs):
    hits = []
    for i, r in enumerate(recs):
        pred = parse_ans(texts[i])
        hits.append(dict(idx=i, gold=r["gold"], pred=pred, gen=texts[i],
                         strict=int(pred == r["gold"]),
                         seen=bool(r.get("seen", False))))
    return hits


@torch.no_grad()
def em_on(model, recs) -> list[int]:
    """★口径1：EM 主口径 batch=1。返回逐样本 strict 命中。"""
    txt = greedy_gen(model, recs, batch=EM_BATCH)
    return [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, recs)], txt


NUM_CHARS = set("0123456789+-*/=%$")
TMPL_CHARS = set("【】详细解题思路推理与") | {"<", ">", "#"}
CAT_NAMES = ("①数字与算子", "②模板/格式串", "③中文字符", "④其它")
_SPECIAL = {"<eos>", "<unk>", "<pad>", "<bos>", "<cont>", "<sep>", "<resp>",
            "<call>", "<result>", "<answer>", "<tool>", "<search>", "<topic>"}


def classify(tid):
    s = tok.id_to_token(int(tid)) or ""
    if s in _SPECIAL or (s.startswith("<") and s.endswith(">")):
        return 1
    if s and all(c in NUM_CHARS for c in s):
        return 0
    if any(c in TMPL_CHARS for c in s):
        return 1
    if any("一" <= c <= "鿿" for c in s):
        return 2
    return 3


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n) if n else float("nan")


@torch.no_grad()
def digit_acc(model, recs):
    """数字每步正确率 = exp(−数字/算子 token 的样本级 CE)，口径逐字沿用 S10/S14/S19。"""
    model.eval()
    tok_ce = [[] for _ in range(4)]
    dig_samp = []
    for i in range(0, len(recs), GEN_BATCH):
        chunk = recs[i: i + GEN_BATCH]
        ids, s = build_batch(chunk)
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        for j, r in enumerate(chunk):
            e = s[j] + len(r["t"])
            lp = logp[j, s[j] - 1: e - 1]
            tgt = ids[j, s[j]: e]
            ces = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1).tolist()
            ds = []
            for pos, v in enumerate(ces):
                k = classify(int(tgt[pos]))
                tok_ce[k].append(v)
                if k == 0:
                    ds.append(v)
            if ds:
                dig_samp.append(ds)
    model.train()
    if not dig_samp:
        return float("nan"), float("nan"), float("nan"), 0
    per = [sum(x) / len(x) for x in dig_samp]
    dm, dse = mean_se(per)
    acc = math.exp(-dm)
    return acc, acc * dse, dm, len(per)


def r28_selfcheck(model, recs, tag, k=CHK_BATCH):
    sub = recs[:k]
    t1 = greedy_gen(model, sub, batch=1)
    tN = greedy_gen(model, sub, batch=CHK_BATCH)
    same = sum(a == b for a, b in zip(t1, tN))
    print(f"[R28] {tag}批内一致性 K={len(sub)}: batch=1 vs batch={CHK_BATCH} 逐字一致 {same}/{len(sub)}"
          f" ⇒ {'一致（batch=1 口径自洽）' if same == len(sub) else '★不一致 ⇒ 仅 batch=1 口径可用'}"
          f"（本单元 EM 全部走 batch=1）", flush=True)
    return same, len(sub)


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def eval_dev(model, dev):
    hits, _ = em_on(model, dev)
    return sum(hits) / len(hits)


def add_jsonl(obj):
    with open(RESULT_JSONL, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


GRID = {3000: (500, 1000, 1500, 2000, 2500, 3000), 6000: (1000, 2000, 3000, 4000, 5000, 6000),
        200: (50, 100, 150, 200)}
RESULTS: dict[tuple[int, str, int], dict] = {}

print(f"\n[S35] ===== 主循环：{len(STEPS_LIST)} 步数 × {len(SEEDS)} seed × {len(ARMS)} 臂 "
      f"= {len(STEPS_LIST)*len(SEEDS)*len(ARMS)} run（cards 与 mono 同 seed 紧挨着跑 ⇒ 配对最干净）=====",
      flush=True)

for steps in STEPS_LIST:
    for seed in SEEDS:
        for arm in ARMS:
            t0 = time.time()
            print(f"\n[S35] ===== run arm={arm} steps={steps} seed={seed} | train={len(TRAIN)} "
                  f"dev={len(DEV)} test={len(TEST)} =====", flush=True)
            model = build(arm, seed)
            snaps: dict[int, dict] = {}
            dev_curve: dict[int, float] = {}

            def hook(m, st, loss, el, steps=steps, seed=seed, arm=arm):
                d = eval_dev(m, DEV)
                dev_curve[st] = d
                snaps[st] = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
                print(f"  [dev] arm={arm} steps={steps} seed={seed} @{st} devEM={d*100:.2f}% "
                      f"(loss={loss:.4f}, {el:.0f}s)", flush=True)

            wall_train = train_run(model, TRAIN, seed, steps, {s: hook for s in GRID[steps]})
            # 末点 test（主口径 batch=1）
            strict_fin, txt_fin = em_on(model, TEST)
            em_fin = sum(strict_fin) / len(TEST)
            em_se = math.sqrt(em_fin * (1 - em_fin) / len(TEST))
            seen_idx = [i for i, r in enumerate(TEST) if r["seen"]]
            unseen_idx = [i for i, r in enumerate(TEST) if not r["seen"]]
            em_seen = sum(strict_fin[i] for i in seen_idx) / len(seen_idx)
            em_unseen = sum(strict_fin[i] for i in unseen_idx) / len(unseen_idx)
            acc, acc_se, dsamp, dn = digit_acc(model, TEST)
            r28s, r28k = r28_selfcheck(model, TEST, tag=f"[{arm} s{steps} seed{seed}] ")
            ck = CKPT_TMPL.format(arm=arm, steps=steps, seed=seed)
            if not SMOKE:
                torch.save(model.state_dict(), ck)
                dig = sha16(ck)
                print(f"[CKPT] saved {ck} sha256[:16]={dig}", flush=True)
            else:
                dig = "-"
            # dev 选步 ⇒ 该步的 test EM（test 只报不看；另加 3000 快照点，供 3000/6000 同轨迹对比）
            best_step = max(dev_curve, key=lambda s: (dev_curve[s], s))   # 平局取更晚的步（dev 同分，不多训只是更保守）
            extra = {}
            for st in {best_step, 3000} - {steps}:
                if st not in snaps:
                    continue
                model.load_state_dict({k: v.to(device) for k, v in snaps[st].items()})
                strict_st, _ = em_on(model, TEST)
                extra[st] = sum(strict_st) / len(TEST)
                if st == best_step:
                    extra["strict_at_best"] = strict_st
                print(f"[DEV-SEL] arm={arm} steps={steps} seed={seed} 快照 @{st} testEM="
                      f"{extra[st]*100:.2f}%", flush=True)
            if best_step == steps:
                extra["strict_at_best"] = strict_fin
            model.load_state_dict({k: v.to(device) for k, v in snaps[steps].items()})
            strict_best = extra.get("strict_at_best", strict_fin)
            em_best = sum(strict_best) / len(TEST)
            wall_total = time.time() - t0
            del model, snaps
            torch.cuda.empty_cache()

            r = dict(arm=arm, steps=steps, seed=seed, train_n=len(TRAIN), dev_n=len(DEV),
                     em=em_fin, em_se=em_se, em_seen=em_seen, em_unseen=em_unseen,
                     n_seen=len(seen_idx), n_unseen=len(unseen_idx),
                     best_step=best_step, dev_at_best=dev_curve[best_step], em_devsel=em_best,
                     dev_curve={str(k): v for k, v in sorted(dev_curve.items())},
                     em_at_3000=extra.get(3000), acc=acc, acc_se=acc_se, dsamp=dsamp, dig_n=dn,
                     r28_same=r28s, r28_k=r28k, wall_train=wall_train, wall_total=wall_total,
                     ckpt=ck, sha=dig, strict=strict_fin, strict_best=strict_best,
                     pred_fin=[parse_ans(t) for t in txt_fin])
            RESULTS[(steps, arm, seed)] = r
            add_jsonl({k: v for k, v in r.items()
                       if k not in ("strict", "strict_best", "pred_fin")})
            print(f"[MAIN-TABLE] {arm} s{steps} seed={seed} | EM全量={em_fin*100:.2f}%±"
                  f"{em_se*100:.2f}(n={len(TEST)}) | SEEN={em_seen*100:.2f}%({len(seen_idx)}) "
                  f"UNSEEN={em_unseen*100:.2f}%({len(unseen_idx)}) | 数字每步={acc*100:.2f}%±"
                  f"{acc_se*100:.2f}pp | dev选步=@{best_step}(dev {dev_curve[best_step]*100:.2f}%"
                  f" ⇒ test {em_best*100:.2f}%) | 训练={wall_train/60:.1f}min 合计="
                  f"{wall_total/60:.1f}min | ckpt={os.path.basename(ck)} sha16={dig}", flush=True)

# ============================================================================
# ④ 汇总 + ★配对 Δ + ★判决
# ============================================================================
def paired(xs_a, xs_b):
    d = [a - b for a, b in zip(xs_a, xs_b)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    nz = sum(1 for x in d if x != 0) / n
    return dict(n=n, d=m, se=se, ratio=(m / se if se > 0 else float("inf")), nz=nz)


print("\n[S35] ===== ★两臂主表（2 步数 × 2 seed） =====", flush=True)
print("[S35] arm | steps | seed | EM全量(n=800)±SE | SEEN | UNSEEN | 数字每步±SE | dev选步 | "
      "dev选步后test | ckpt sha16", flush=True)
for steps in STEPS_LIST:
    for seed in SEEDS:
        for arm in ARMS:
            r = RESULTS.get((steps, arm, seed))
            if r is None:
                print(f"[S35] {arm} {steps} {seed}: ★未完成", flush=True)
                continue
            print(f"[S35] {arm} | {steps} | {seed} | {r['em']*100:.2f}%±{r['em_se']*100:.2f} | "
                  f"{r['em_seen']*100:.2f}% | {r['em_unseen']*100:.2f}% | "
                  f"{r['acc']*100:.2f}%±{r['acc_se']*100:.2f}pp | @{r['best_step']}"
                  f"(dev {r['dev_at_best']*100:.2f}%) | {r['em_devsel']*100:.2f}% | {r['sha']}",
                  flush=True)

print("\n[S35] ===== ★Δ = EM(A=mono) − EM(B=cards)（逐 test 样本配对；门槛 1SE / 2SE）=====",
      flush=True)
print("[S35] steps | seed | 口径 | Δ±SE(pp) | Δ/SE | Δ非零比例(R29) | 预测串不同比例 | 判定",
      flush=True)
DELTAS: dict[tuple[int, str], list[dict]] = {}
for steps in STEPS_LIST:
    for seed in SEEDS:
        a = RESULTS.get((steps, "mono", seed))
        b = RESULTS.get((steps, "cards", seed))
        if a is None or b is None:
            print(f"[S35] {steps} {seed}: ★缺 run ⇒ 无法配对", flush=True)
            continue
        for tag, ka, kb in (("末点", "strict", "strict"), ("dev选步", "strict_best", "strict_best")):
            dd = paired(a[ka], b[kb])
            pred_diff = sum(1 for x, y in zip(a["pred_fin"], b["pred_fin"]) if x != y) / len(TEST)
            if dd["se"] == 0:                       # 两臂逐样本完全相同 ⇒ SE=0，按"无差异"处理
                verdict = "Δ 恒为 0（逐样本完全相同）⇒ 等价，无增量" if dd["d"] == 0 else \
                          f"SE=0 且 Δ={dd['d']*100:+.2f}pp（全样本同向）⇒ 看 2 seed 汇总"
            elif dd["d"] >= 2 * dd["se"]:
                verdict = "闭环有增量(B更好)"
            elif dd["d"] <= -2 * dd["se"]:
                verdict = "单体更好(A更好)⇒卡框架负增量"
            elif abs(dd["d"]) <= dd["se"]:
                verdict = "|Δ|≤1SE ⇒ 单体不差 ⇒ 卡框架零增量"
            else:
                verdict = "1SE<|Δ|<2SE ⇒ 未过门槛（不显著）"
            DELTAS.setdefault((steps, tag), []).append(dict(seed=seed, **dd))
            print(f"[S35] {steps} | {seed} | {tag} | {dd['d']*100:+.2f}±{dd['se']*100:.2f} | "
                  f"{dd['ratio']:+.2f} | {dd['nz']*100:.1f}% | {pred_diff*100:.1f}% | {verdict}",
                  flush=True)

print("\n[S35] ===== ★逐条判定（2 seed 同号 + 门槛，判据写死）=====", flush=True)
JUDGE: dict[tuple[int, str], str] = {}
for steps in STEPS_LIST:
    for tag in ("末点", "dev选步"):
        ds = DELTAS.get((steps, tag))
        if not ds or len(ds) < 2:
            JUDGE[(steps, tag)] = f"steps={steps} {tag}: ★run 不全 ⇒ 不判"
            print(f"[S35-JUDGE] {JUDGE[(steps, tag)]}", flush=True)
            continue
        signs = {1 if d["d"] > 0 else (-1 if d["d"] < 0 else 0) for d in ds}
        same = len(signs) == 1
        all2 = all(d["d"] >= 2 * d["se"] and d["d"] > 0 for d in ds)
        allneg2 = all(d["d"] <= -2 * d["se"] and d["d"] < 0 for d in ds)
        all1 = all(abs(d["d"]) <= d["se"] for d in ds)
        txt = " ".join(f"Δ(s{d['seed']})={d['d']*100:+.2f}±{d['se']*100:.2f}pp" for d in ds)
        if signs == {0}:
            v = f"★两臂逐 test 样本完全相同（Δ≡0，SE=0）⇒ 等价 ⇒ 卡框架零增量（{txt}）"
        elif not same:
            v = f"2 seed 不同号 ⇒ 不显著（{txt}）"
        elif all2:
            v = f"★Δ≥2SE 且 2 seed 同号(正) ⇒ 闭环有增量（{txt}）"
        elif allneg2:
            v = f"★Δ≤−2SE 且 2 seed 同号(负) ⇒ 单体更好 ⇒ 卡框架负增量（{txt}）"
        elif all1 and signs == {1}:
            v = f"★|Δ|≤1SE 且 2 seed 同号(正) ⇒ 单体不差 ⇒ 卡框架零增量（{txt}）"
        elif all1 and signs == {-1}:
            v = f"★|Δ|≤1SE 且 2 seed 同号(负) ⇒ 单体不差 ⇒ 卡框架零增量（{txt}）"
        else:
            v = f"未过门槛（{txt}）"
        JUDGE[(steps, tag)] = f"steps={steps} {tag}: {v}"
        print(f"[S35-JUDGE] {JUDGE[(steps, tag)]}", flush=True)

print("\n[S35] ===== ★饱和曲线（dev EM 轨迹 + 3000/6000 末点 test EM）=====", flush=True)
for steps in STEPS_LIST:
    for arm in ARMS:
        for seed in SEEDS:
            r = RESULTS.get((steps, arm, seed))
            if r is None:
                continue
            cur = " ".join(f"{k}:{v*100:.1f}%" for k, v in r["dev_curve"].items())
            print(f"[S35-CURVE] {arm} s{steps} seed={seed} devEM: {cur} | 末点testEM="
                  f"{r['em']*100:.2f}%" + (f" | 同轨迹@3000 testEM={r['em_at_3000']*100:.2f}%"
                                           if r.get("em_at_3000") is not None else ""), flush=True)
for arm in ARMS:
    r3 = RESULTS.get((3000, arm, SEEDS[0]))
    r6 = RESULTS.get((6000, arm, SEEDS[0])) or RESULTS.get((3000, arm, SEEDS[0]))
    if r6 is not None and r6.get("em_at_3000") is not None:
        print(f"[S35-SAT] {arm}: 同一条 6000 步轨迹上 testEM 3000→6000 = "
              f"{r6['em_at_3000']*100:.2f}% → {r6['em']*100:.2f}%", flush=True)
_ris = {}
for arm in ARMS:
    for seed in SEEDS:
        r6 = RESULTS.get((6000, arm, seed))
        if r6 is None:
            continue
        ks = sorted(int(k) for k in r6["dev_curve"])
        _ris[(arm, seed)] = (r6["dev_curve"][str(ks[-1])] - r6["dev_curve"][str(ks[-2])]) * 100
for k, v in _ris.items():
    print(f"[S35-SAT] {k[0]} seed={k[1]}: dev 最后两格(5000→6000) Δ={v:+.2f}pp", flush=True)
print(f"[S35-SAT] 若两臂最后两格 dev EM 都还 >0 ⇒ 不能外推，需再补一格（6000 之后）", flush=True)

# ---------------- 锚点参考 + 收尾 ----------------
print("\n[S35] ===== 锚点参考（口径4 改了训练集 3600 vs S19 的 4000 ⇒ 只作参考，非逐位可复现）=====",
      flush=True)
for seed in SEEDS:
    r = RESULTS.get((3000, "cards", seed))
    if r is None:
        continue
    ref = S19_RES_ANCHOR[seed]
    print(f"[S35-ANCHOR] cards@3000 seed={seed}: S19(训练4000)={ref*100:.2f}% vs "
          f"本次(训练3600)={r['em']*100:.2f}% ⇒ Δ={(r['em']-ref)*100:+.2f}pp", flush=True)

print(f"\n[META] device={device} {torch.cuda.get_device_name(0)} | D={D} FF={FF} NHEAD={NHEAD} "
      f"lr={LR} batch={BATCH} MAXLEN={MAXLEN} | TEST sha16={TEST_SHA} 泄漏对train4000="
      f"{LEAK_AB_ALL}/800 对train3600={LEAK_TR}/800 | 地板(单答案/top5)={FLOOR_1*100:.2f}%/"
      f"{FLOOR_5*100:.2f}% | 参数量 B={_NB} A={_NA} 差={_NB-_NA} | 卡边界逐位等价="
      f"max|Δlogits|={EQ_MAX:.1e}", flush=True)
for steps in STEPS_LIST:
    for arm in ARMS:
        for seed in SEEDS:
            r = RESULTS.get((steps, arm, seed))
            if r is None:
                continue
            print(f"[META] {arm} s{steps} seed={seed}: steps={steps} 训练墙钟={r['wall_train']/60:.1f}min "
                  f"合计={r['wall_total']/60:.1f}min R28={r['r28_same']}/{r['r28_k']} ckpt="
                  f"{os.path.basename(r['ckpt'])} sha256[:16]={r['sha']}", flush=True)
print(f"[META] 总墙钟={(time.time()-t_start)/60:.1f}min | run 数={len(RESULTS)}/"
      f"{len(STEPS_LIST)*len(SEEDS)*len(ARMS)}", flush=True)
print("[DONE] exit=0", flush=True)
