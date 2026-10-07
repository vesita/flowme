#!/usr/bin/env python3
"""S14 · 判决性实验：程序生成的「算术合成集」—— 这套架构到底会不会算？ ★GPU。

只回答一个问题：在一个**答案分布均匀、难度可控**的合成算术集上，这套架构能不能学会「会算」。
（GSM8K 答案高度偏斜 ⇒ S12 的 EM 2.17% 可能只是「多数类先验」；本单元用合成集把这层剥掉。）

数据（本脚本自己程序生成，**不用 GSM8K / nanoSeek 语料**；分词器仍用 A 臂那只 char_tokenizer）：
  格式 `题干：<中文或英文自然语言> → #### <答案>`，**只输出答案、无 CoT** ⇒
  prompt = `题干：<nl> → `，target = `#### <答案>` + <eos>（★A 臂口径逐字一致）。
  五桶：add_1d / add_2d / add_3d / sub_2d / mul_2d；每桶 train 4000 / test 800；
  答案**按取值逐个轮转生成** ⇒ 各半近似均匀（不集中在小整数）；
  同一套模板（程序组合出 ~360 句 × 中文紧凑/空格两种排版），数字随机、文本 train/test 零重叠。

配置与 A 臂逐字一致：d=128 ff=512 NHEAD=4 lr=1e-3 batch=32 3000 步 MAXLEN=512（prompt 优先）
  MIN_T=48 TAIL_KEEP=24 EPS=1e-3 生成主口径 batch=1（R28）。
  每桶单独训练：5 桶 × seed1234；时间允许再给 add_1d / add_3d 补 seed5678。

★主指标（唯一干净指标）= 终局答案数字的每步正确率 = exp(−数字/算子 token CE)（样本级 ±SE，delta 法）。
R28：自回归生成 batch=1 为主口径，每个模型报一次 batch=1 vs batch=16 逐字一致率。
高频整数倾向：每桶抽 20 条生成，报「输出 ∈ 该桶训练集最高频 5 个答案」的比例（并给机会水平）。

ckpt：logs/14_ckpt_<桶>_seed<seed>.pt（报 sha256 前16）。
只允许写：本文件、logs/、/tmp；既有 stages/*.py 与 nanoSeek 只读。
"""
from __future__ import annotations

import hashlib
import math
import os
import sys
import time
from collections import Counter, defaultdict

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
sys.path.insert(0, NS_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402

# ---------------- device 纪律（ROCm 下报成 cuda，正常） ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
if device != "cuda":
    print("GPU 起不来，立即停（不用 CPU 硬跑）。", flush=True)
    print("torch.cuda.is_available() =", torch.cuda.is_available(), flush=True)
    print("HSA_OVERRIDE_GFX_VERSION =", os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)
    sys.exit(3)

# ---------------- 配置（与 A 臂逐字一致） ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512
MIN_T = 48
TAIL_KEEP = 24
STEPS = 3000
LR = 1e-3
EPS = 1e-3
BATCH = 32
GEN_BATCH = 32
EM_BATCH = 1          # ★主口径：batch=1（R28）
CHK_BATCH = 16        # 自检口径
N_TRAIN = 4000
N_TEST = 800
MAX_GEN = 48          # 生成上限（target 最长 ~10 tok；防模型不吐 <eos> 时退化成 472 步×800）

SMOKE = os.environ.get("S14_SMOKE") == "1"
if SMOKE:
    STEPS, N_TRAIN, N_TEST = 30, 300, 60

TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
CKPT_TMPL = "/home/vesita/coding/my/flowme/logs/14_ckpt_{bucket}_seed{seed}.pt"
SEED_PRIMARY = 1234
SEED_EXTRA = (5678,)
EXTRA_BUCKETS = ("add_1d", "add_3d")
BUDGET_MIN = 50.0        # 主 5 桶跑完后，剩余墙钟 < 该值才补第二个 seed

t_start = time.time()
print(f"[S14] ★合成算术判决：5 桶各自单训 d={D} ff={FF} steps={STEPS} lr={LR} batch={BATCH} "
      f"MAXLEN={MAXLEN}(prompt优先) NHEAD={NHEAD} EPS={EPS} 切分=train{N_TRAIN}/test{N_TEST} "
      f"生成主口径=batch{EM_BATCH} | seed={SEED_PRIMARY}"
      f"{'' if SMOKE else f'(+{SEED_EXTRA} 给 {EXTRA_BUCKETS})'} | smoke={SMOKE}", flush=True)

# ---------------- tokenizer（只读） ----------------
from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(TOK)
V = tok.get_vocab_size()
PAD_ID = tok.token_to_id("<pad>")
EOS_ID = tok.token_to_id("<eos>")


def enc(text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


_unk = tok.token_to_id("<unk>")
_arrow_ids = enc("→")
print(f"[TOK] WordLevel V={V} PAD={PAD_ID} EOS={EOS_ID} enc('<eos>')={enc('<eos>')} | "
      f"分隔符 '→' 词表外 ⇒ 单 token <unk>(id={_unk})，实测 enc('→')={_arrow_ids} "
      f"（prompt 里的常量分隔符，不进 target）", flush=True)


# ============================================================================
# ① 数据：程序生成的算术合成集（五桶，答案逐取值轮转 ⇒ 均匀；同一套模板）
# ============================================================================
def _has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


# —— 同一套自然语言模板（程序组合，桶间共用；{expr} 里放 "<a> <op> <b>"） ——
def build_templates() -> list[tuple[str, bool]]:
    ts: list[tuple[str, bool]] = []
    tails_cn = ["等于多少？", "是多少？", "等于几？", "得多少？", "的结果是多少？",
                "的结果是几？", "的值是多少？", "等于多少", "是多少", "是几"]
    for lead in ("", "请", "帮我", "麻烦"):
        for verb in ("计算", "算出", "算一下", "口算", "快速算", "算"):
            for t in tails_cn:
                ts.append((f"{lead}{verb} {{expr}} {t}", True))
    for t in tails_cn:                       # 裸式（最常见的口算句）
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
N_TPL_CN = sum(1 for _, c in TEMPLATES if c)
RENDER_VARIANTS = N_TPL_CN * 2 + (len(TEMPLATES) - N_TPL_CN)   # 中文多一种紧凑排版

CN_OPS = {"add_1d": ("加", "加上"), "add_2d": ("加", "加上"), "add_3d": ("加", "加上"),
          "sub_2d": ("减", "减去"), "mul_2d": ("乘", "乘以")}
EN_OPS = {"add_1d": ("plus",), "add_2d": ("plus",), "add_3d": ("plus",),
          "sub_2d": ("minus",), "mul_2d": ("times",)}

# —— 各桶的取值域与「给定答案反解操作数」（⇒ 答案严格均匀） ——
LOHI = {"add_1d": (0, 9), "add_2d": (10, 99), "add_3d": (100, 999)}

MUL_BY_Y: dict[int, list[tuple[int, int]]] = defaultdict(list)
for _a in range(10, 100):
    for _b in range(1, 10):        # 一位数乘数取 1–9：取 0 会让 10% 样本答案恒为 0，破坏均匀
        MUL_BY_Y[_a * _b].append((_a, _b))
MUL_YS = sorted(MUL_BY_Y)


def answer_values(name: str) -> list[int]:
    if name in LOHI:
        lo, hi = LOHI[name]
        return list(range(2 * lo, 2 * hi + 1))
    if name == "sub_2d":
        return list(range(0, 90))
    return list(MUL_YS)


def split_by_answer(name: str, y: int, rng) -> tuple[int, int]:
    if name in LOHI:
        lo, hi = LOHI[name]
        a = rng.randint(max(lo, y - hi), min(hi, y - lo))
        return a, y - a
    if name == "sub_2d":
        a = rng.randint(max(10, y + 10), 99)
        return a, a - y
    fs = MUL_BY_Y[y]
    return fs[rng.randrange(len(fs))]


def carry_rate(name: str, n: int, rng) -> tuple[float, float]:
    """(至少一次进位的占比, 个位进位的占比)；非加法桶返回 (nan, nan)。"""
    if name not in LOHI:
        return float("nan"), float("nan")
    lo, hi = LOHI[name]
    any_c = unit_c = 0
    for _ in range(n):
        y = rng.randint(2 * lo, 2 * hi)
        a, b = split_by_answer(name, y, rng)
        w = max(len(str(a)), len(str(b)))
        sa, sb = str(a).zfill(w), str(b).zfill(w)
        c = 0
        carry_pos = []
        for i in range(w - 1, -1, -1):
            s = int(sa[i]) + int(sb[i]) + c
            if s >= 10:
                carry_pos.append(w - 1 - i)
            c = s // 10
        if carry_pos:
            any_c += 1
        if 0 in carry_pos:
            unit_c += 1
    return any_c / n, unit_c / n


def borrow_rate(name: str, n: int, rng) -> float:
    if name != "sub_2d":
        return float("nan")
    hit = 0
    for _ in range(n):
        y = rng.randint(0, 89)
        a, b = split_by_answer(name, y, rng)
        w = max(len(str(a)), len(str(b)))
        sa, sb = str(a).zfill(w), str(b).zfill(w)
        c, bor = 0, False
        for i in range(w - 1, -1, -1):
            d = int(sa[i]) - int(sb[i]) - c
            if d < 0:
                bor = True
                c = 1
            else:
                c = 0
        hit += int(bor)
    return hit / n


class Rng:
    """可复现的桶内随机源（python random 的 MT199337 同族，避免占用 torch 全局种子）。"""

    def __init__(self, seed: int):
        import random
        self._r = random.Random(seed)

    def randrange(self, n: int) -> int:
        return self._r.randrange(n)

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def random(self) -> float:
        return self._r.random()

    def shuffle(self, xs: list) -> None:
        self._r.shuffle(xs)


def render(name: str, a: int, b: int, y: int, rng: Rng) -> tuple[str, str, str]:
    tpl, cjk = TEMPLATES[rng.randrange(len(TEMPLATES))]
    op = (CN_OPS[name] if cjk else EN_OPS[name])
    op = op[rng.randrange(len(op))]
    tight = cjk and rng.random() < 0.5
    expr = f"{a}{op}{b}" if tight else f"{a} {op} {b}"
    nl = tpl.format(expr=expr)
    return f"题干：{nl} → ", f"#### {y}", nl


def gen_half(name: str, n: int, rng: Rng, used: set[str], tag: str) -> list[dict]:
    """按答案取值逐个轮转生成 n 条 ⇒ 答案近似均匀；与已用文本零重叠。"""
    ys = answer_values(name)
    order = ys[:]
    rng.shuffle(order)
    items: list[dict] = []
    i = tries = 0
    while len(items) < n:
        y = order[i % len(order)]
        i += 1
        tries += 1
        if tries > n * 500:
            raise RuntimeError(f"{name}{tag}: 文本空间不足 got={len(items)}/{n}")
        a, b = split_by_answer(name, y, rng)
        prompt, target, nl = render(name, a, b, y, rng)
        full = prompt + target
        if full in used:
            continue
        used.add(full)
        items.append(dict(prompt=prompt, target=target, a=a, b=b, y=y, nl=nl))
    return items


def encode_record(it: dict) -> dict:
    """prompt 优先的截断（与 A 臂逐字相同）；本数据极短 ⇒ 实测应全程 0 截断。"""
    p_full = enc(it["prompt"])
    t_full = enc(it["target"] + "<eos>")
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
                text=it["target"], cut_p=cut_p, cut_t=cut_t, a=it["a"], b=it["b"],
                y=it["y"], nl=it["nl"])


BUCKETS = ("add_1d", "add_2d", "add_3d", "sub_2d", "mul_2d")
DATA: dict[str, dict[str, list[dict]]] = {}
BUCKET_INFO: dict[str, dict] = {}

for bi, name in enumerate(BUCKETS):
    tr_rng, te_rng = Rng(14000 + bi * 7), Rng(14900 + bi * 7)
    used: set[str] = set()
    raw_tr = gen_half(name, N_TRAIN, tr_rng, used, "train")
    raw_te = gen_half(name, N_TEST, te_rng, used, "test")
    train = [encode_record(x) for x in raw_tr]
    test = [encode_record(x) for x in raw_te]
    assert len(used) == len(raw_tr) + len(raw_te), "train/test 文本池计数异常（应零重叠）"

    ys = answer_values(name)
    ctr_tr = Counter(r["y"] for r in train)
    ctr_te = Counter(r["y"] for r in test)
    pair_tr = len({(r["a"], r["b"]) for r in train})
    pair_te = len({(r["a"], r["b"]) for r in test})
    top5 = [v for v, _ in Counter(r["gold"] for r in train).most_common(5)]
    prior = sum(1 for r in test if r["gold"] in top5) / max(1, len(test))
    cut = sum(1 for r in train + test if r["cut_p"] or r["cut_t"])
    anyc, unitc = carry_rate(name, 300, Rng(77 + bi))
    bor = borrow_rate(name, 300, Rng(97 + bi))
    tpl_used = len({r["nl"] for r in train + test})
    tgt_len = [len(r["t_full"]) for r in train]
    tgt_len.sort()

    DATA[name] = dict(train=train, test=test)
    BUCKET_INFO[name] = dict(top5=top5, prior=prior, ans_n=len(ys),
                             ctr_tr=ctr_tr, ctr_te=ctr_te,
                             pair_tr=pair_tr, pair_te=pair_te, tpl_used=tpl_used,
                             anyc=anyc, unitc=unitc, bor=bor)
    print(f"[DATA] {name}: train={len(train)} test={len(test)} 文本零重叠 | "
          f"答案域={min(ys)}..{max(ys)}({len(ys)}个) 计数train min/max="
          f"{min(ctr_tr.values())}/{max(ctr_tr.values())} test min/max="
          f"{min(ctr_te.values())}/{max(ctr_te.values())} | "
          f"最大占比={100.0*max(ctr_tr.values())/len(train):.1f}%", flush=True)
    print(f"[DATA] {name}: 模板覆盖={tpl_used}/{len(TEMPLATES)}句(渲染{RENDER_VARIANTS}种) "
          f"| distinct(a,b) train={pair_tr} test={pair_te} | 截断={cut} | "
          f"target中位={tgt_len[len(tgt_len)//2]}tok | "
          f"进位率(至少一次/个位)={anyc:.2f}/{unitc:.2f} 借位率={bor:.2f} | "
          f"★最高频5答案={top5} 在test的机会水平={100.0*prior:.1f}%", flush=True)

print(f"[TPL] 同一套模板 {len(TEMPLATES)} 句（中文 {N_TPL_CN} × 排版2 + 英文 "
      f"{len(TEMPLATES)-N_TPL_CN}）⇒ 渲染 {RENDER_VARIANTS} 种；五桶共用，只换操作数与操作符。",
      flush=True)


# ============================================================================
# ② 模型（与 A 臂逐字相同）
# ============================================================================
def sin_pe(max_len: int, d: int) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def enc_layer(d: int, ff: int) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d, NHEAD, ff, dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )


class Cards(nn.Module):
    def __init__(self, d: int, ff: int, identity: bool = False):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        self.thought = None if identity else enc_layer(d, ff)
        self.head = nn.Linear(d, V)
        self.register_buffer("pe", sin_pe(MAXLEN, d), persistent=False)

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        n = ids.size(1)
        m = torch.full((n, n), float("-inf"), device=ids.device)
        m = torch.triu(m, diagonal=1)
        m = m.unsqueeze(0).expand(ids.size(0), -1, -1).clone()
        m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
        m = m.repeat_interleave(NHEAD, dim=0)
        pe = self.pe if n <= self.pe.size(0) else sin_pe(n, self.d).to(self.pe.device)
        assert n <= pe.size(0), f"PE 长度 {pe.size(0)} < 序列长 {n}"
        x = self.emb(ids) + pe[:n].unsqueeze(0)
        h = self.in_enc(x, src_mask=m)
        if self.thought is not None:
            h = self.thought(h, src_mask=m)
        return self.head(h)


def n_params(d: int, ff: int) -> int:
    m = Cards(d, ff)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


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


def masked_ce(model: Cards, recs) -> torch.Tensor:
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


def train_run(seed: int, train, steps: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards(D, FF).to(device)
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
        if step % 500 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train] step={step}/{steps} loss={loss.item():.4f} "
                  f"elapsed={el:.0f}s s/step={(el - last) / (step % 500 or 500):.3f}",
                  flush=True)
            last = el
    return model, time.time() - t0


@torch.no_grad()
def eval_ce(model: Cards, recs) -> list[float]:
    model.eval()
    per = []
    for i in range(0, len(recs), GEN_BATCH):
        chunk = recs[i: i + GEN_BATCH]
        ids, s = build_batch(chunk)
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        for j, r in enumerate(chunk):
            e = s[j] + len(r["t"])
            lp = logp[j, s[j] - 1: e - 1]
            tgt = ids[j, s[j]: e]
            per.append(float((-lp.gather(1, tgt.unsqueeze(1)).sum() / len(r["t"])).item()))
    model.train()
    return per


# ---------------- 数字/算子 CE 分类口径（逐字沿用 S10） ----------------
NUM_CHARS = set("0123456789+-*/=%$")
TMPL_CHARS = set("【】详细解题思路推理与") | {"<", ">", "#"}
CAT_NAMES = ("①数字与算子", "②模板/格式串", "③中文字符", "④其它")
_SPECIAL = {"<eos>", "<unk>", "<pad>", "<bos>", "<cont>", "<sep>", "<resp>",
            "<call>", "<result>", "<answer>", "<tool>", "<search>", "<topic>"}


def classify(tid: int) -> int:
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


@torch.no_grad()
def ce_categories(model: Cards, recs):
    model.eval()
    tok_ce: list[list[float]] = [[] for _ in range(4)]
    samp_all: list[float] = []
    dig_samp: list[list[float]] = []
    for i in range(0, len(recs), GEN_BATCH):
        chunk = recs[i: i + GEN_BATCH]
        ids, s = build_batch(chunk)
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        for j, r in enumerate(chunk):
            e = s[j] + len(r["t"])
            lp = logp[j, s[j] - 1: e - 1]
            tgt = ids[j, s[j]: e]
            ces = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1).tolist()
            dsamp = []
            for pos, v in enumerate(ces):
                k = classify(int(tgt[pos]))
                tok_ce[k].append(v)
                if k == 0:
                    dsamp.append(v)
            if dsamp:
                dig_samp.append(dsamp)
            samp_all.append(sum(ces) / len(ces))
    model.train()
    return tok_ce, samp_all, dig_samp


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n) if n else float("nan")


def report_ce(tag: str, model: Cards, recs) -> dict:
    tok_ce, samp_all, dig_samp = ce_categories(model, recs)
    ntok = sum(len(v) for v in tok_ce)
    ce_m, ce_se = mean_se(samp_all)
    dig_ce = dig_n = None
    for k in range(4):
        xs = tok_ce[k]
        if not xs:
            print(f"[CE] {tag} {CAT_NAMES[k]}: n=0", flush=True)
            continue
        km = sum(xs) / len(xs)
        print(f"[CE] {tag} {CAT_NAMES[k]}: n={len(xs)} 占比={100.0*len(xs)/ntok:.1f}% "
              f"CE={km:.4f} nats", flush=True)
        if k == 0:
            dig_ce, dig_n = km, len(xs)
    out = dict(ce_m=ce_m, ce_se=ce_se, dig_ce=dig_ce, dig_n=dig_n)
    print(f"[CE] {tag} ★整体配对CE(样本级)={ce_m:.4f}±{ce_se:.4f} n={len(samp_all)}", flush=True)
    if dig_ce is not None:
        per = [sum(x) / len(x) for x in dig_samp]
        dm, dse = mean_se(per)
        acc = math.exp(-dm)
        acc_se = acc * dse
        print(f"[CE] {tag} ★数字CE(token加权)={dig_ce:.4f} n={dig_n} | "
              f"★数字CE(样本级)={dm:.4f}±{dse:.4f} n={len(per)} ⇒ "
              f"★数字每步正确率={acc*100:.2f}%±{acc_se*100:.2f}pp", flush=True)
        out.update(dig_samp_m=dm, dig_samp_se=dse, acc=acc, acc_se=acc_se)
    return out


# ---------------- R28：贪心生成（主口径 batch=1） ----------------
@torch.no_grad()
def greedy_gen(model: Cards, recs, batch: int = 1) -> list[str]:
    """查询列始终指向「最后一个真实/已生成 token 所在列」（正确写法）；
    batch=1 时 n0==lens[j] ⇒ 与旧写法逐字节相同（R28 自洽的原因）。"""
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


def hits_from(texts: list[str], recs) -> list[dict]:
    hits = []
    for i, r in enumerate(recs):
        if r["gold"] is None:
            continue
        pred = parse_ans(texts[i])
        hits.append(dict(idx=i, gold=r["gold"], pred=pred, gen=texts[i],
                         strict=int(pred == r["gold"]),
                         bucket=("B" if r["cut_t"] else "A"),
                         free=(not r["cut_t"] and not r["cut_p"])))
    return hits


def r28_selfcheck(model: Cards, recs, tag: str, k: int = CHK_BATCH) -> dict:
    sub = recs[:k]
    t1 = greedy_gen(model, sub, batch=1)
    tN = greedy_gen(model, sub, batch=CHK_BATCH)
    same = sum(a == b for a, b in zip(t1, tN))
    print(f"[R28] {tag}批内一致性 K={len(sub)}: batch=1 vs batch={CHK_BATCH} 逐字一致 "
          f"{same}/{len(sub)} ⇒ {'一致（batch=1 口径自洽）' if same == len(sub) else '★不一致 ⇒ batch=N 口径作废'}",
          flush=True)
    shown = 0
    for i, (a, b) in enumerate(zip(t1, tN)):
        if a == b:
            continue
        d = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print(f"[R28] {tag}差异[{i}] 首分歧位={d} | batch1={a[:50]!r} | batchN={b[:50]!r}",
              flush=True)
        shown += 1
        if shown >= 3:
            break
    return dict(same=same, k=len(sub))


def em_report(tag: str, hits: list[dict]) -> dict:
    out = {}
    for name, sel in (("桶A(目标未截)", [h for h in hits if h["bucket"] == "A"]),
                      ("桶A1(都未截)", [h for h in hits if h["free"]]),
                      ("全量", list(hits))):
        key = name.split("(")[0]
        n = len(sel)
        s = sum(h["strict"] for h in sel) / n if n else float("nan")
        se = math.sqrt(s * (1 - s) / n) if n else float("nan")
        out[key] = dict(n=n, em=s, se=se)
        print(f"[EM] {tag} {name}: n={n} 严格EM={s*100:.2f}%±{se*100:.2f}", flush=True)
    return out


def sample20(tag: str, hits: list[dict], top5: list[str]) -> dict:
    pool = sorted(hits[:20], key=lambda h: h["idx"])
    n_freq = sum(1 for h in pool if h["pred"] in top5)
    vals = [h["pred"] for h in pool]
    print(f"[FREQ] {tag} ★抽20条(batch=1贪心): 输出∈训练集最高频5答案{top5} = "
          f"{n_freq}/{len(pool)} ({100.0*n_freq/len(pool):.0f}%) | 抽取值分布="
          f"{dict(Counter(vals).most_common(8))}", flush=True)
    for i, h in enumerate(pool[:5]):
        print(f"[GEN] {tag}[{i}] {h['gen'][:60]!r} | gold={h['gold']} | 抽取={h['pred']} "
              f"| {'对' if h['strict'] else '错'}", flush=True)
    allp = [h["pred"] for h in hits]
    allf = sum(1 for p in allp if p in top5)
    print(f"[FREQ] {tag} 参考·全test(n={len(hits)}): ∈高频5 = {allf}/{len(hits)} "
          f"({100.0*allf/len(hits):.1f}%) | Top6={dict(Counter(allp).most_common(6))}", flush=True)
    return dict(n20=len(pool), freq20=n_freq,
                all_freq=allf, all_n=len(hits))


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ============================================================================
# ③ 主循环：5 桶 × seed1234（时间允许补 seed5678 给 add_1d / add_3d）
# ============================================================================
JOBS: list[tuple[str, int]] = [(b, SEED_PRIMARY) for b in BUCKETS]
RESULTS: dict[tuple[str, int], dict] = {}

print(f"\n[S14] ===== params d={D} ff={FF} ⇒ {n_params(D, FF)/1e6:.2f}M | "
      f"五桶={list(BUCKETS)} 主 seed={SEED_PRIMARY} =====", flush=True)


def run_one(bucket: str, seed: int, first: bool) -> None:
    train, test = DATA[bucket]["train"], DATA[bucket]["test"]
    info = BUCKET_INFO[bucket]
    t0 = time.time()
    print(f"\n[S14] ===== 桶 {bucket} seed={seed} steps={STEPS} train={len(train)} "
          f"test={len(test)} =====", flush=True)
    model, wall = train_run(seed, train, STEPS)

    ce_list = eval_ce(model, test)
    cm, cs = mean_se(ce_list)
    txt = greedy_gen(model, test, batch=EM_BATCH)
    hits = hits_from(txt, test)
    rep = report_ce(f"{bucket} seed={seed}", model, test)
    em = em_report(f"{bucket} seed={seed} (batch={EM_BATCH})", hits)
    r28 = r28_selfcheck(model, test, tag=f"[{bucket} seed={seed}] ")
    freq = sample20(f"{bucket} seed={seed}", hits, info["top5"])

    ck = CKPT_TMPL.format(bucket=bucket, seed=seed)
    if not SMOKE:
        torch.save(model.state_dict(), ck)
        dig = sha16(ck)
        print(f"[CKPT] saved {ck} sha256[:16]={dig}", flush=True)
    else:
        dig = "-"
    total = time.time() - t0
    del model
    torch.cuda.empty_cache()

    RESULTS[(bucket, seed)] = dict(
        bucket=bucket, seed=seed, steps=STEPS, wall_train=wall, wall_total=total,
        ce=cm, ce_se=cs, acc=rep.get("acc"), acc_se=rep.get("acc_se"),
        dsamp=rep.get("dig_samp_m"), dsamp_se=rep.get("dig_samp_se"),
        dig_tok=rep.get("dig_ce"),
        em=em["全量"]["em"], em_se=em["全量"]["se"], em_n=em["全量"]["n"],
        em_a1=em["桶A1"]["em"], em_a1_n=em["桶A1"]["n"],
        r28=r28, freq=freq, ckpt=ck, sha=dig,
        prior=info["prior"], ans_n=info["ans_n"],
    )
    r = RESULTS[(bucket, seed)]
    print(f"[MAIN-TABLE] {bucket} seed={seed} | 数字每步正确率={r['acc']*100:.2f}%"
          f"±{r['acc_se']*100:.2f}pp | 整体CE={r['ce']:.4f}±{r['ce_se']:.4f} | "
          f"EM全量={r['em']*100:.2f}%(n={r['em_n']}) 桶A1={r['em_a1']*100:.2f}%(n={r['em_a1_n']}) | "
          f"R28={r['r28']['same']}/{r['r28']['k']} | 高频5={r['freq']['freq20']}/20 | "
          f"训练={wall/60:.1f}min 合计={total/60:.1f}min | ckpt={os.path.basename(ck)} sha16={dig}",
          flush=True)


for j, (bucket, seed) in enumerate(JOBS):
    run_one(bucket, seed, first=(j == 0))

# —— 时间允许再补第二个 seed ——
elapsed_min = (time.time() - t_start) / 60
if not SMOKE and elapsed_min < BUDGET_MIN:
    for bucket in EXTRA_BUCKETS:
        run_one(bucket, SEED_EXTRA[0], first=False)
    print(f"[S14] 补跑 {EXTRA_BUCKETS} × seed{SEED_EXTRA[0]}"
          f"（主循环结束时已用 {elapsed_min:.1f}min < {BUDGET_MIN}min 预算）", flush=True)
else:
    print(f"[S14] 跳过补跑 seed{SEED_EXTRA[0]}：主循环已用 {elapsed_min:.1f}min "
          f"{'≥' if not SMOKE else '或 smoke 模式'} {BUDGET_MIN}min 预算", flush=True)

# ============================================================================
# ④ 汇总 + ★判决
# ============================================================================
print("\n[S14] ===== ★五桶主表（seed1234 为主判据） =====", flush=True)
print("[S14] 桶 | 答案域(个数) | 数字每步正确率±SE | 数字CE(样本级) | 整体CE±SE | "
      "EM全量(n) | 桶A1(n) | R28 | 高频5比例 | 机会水平", flush=True)
for b in BUCKETS:
    r = RESULTS.get((b, SEED_PRIMARY))
    if r is None:
        print(f"[S14] {b}: ★未完成（时间不够）", flush=True)
        continue
    print(f"[S14] {b} | {min(answer_values(b))}..{max(answer_values(b))}({r['ans_n']}) | "
          f"{r['acc']*100:.2f}%±{r['acc_se']*100:.2f}pp | {r['dsamp']:.4f}±{r['dsamp_se']:.4f} | "
          f"{r['ce']:.4f}±{r['ce_se']:.4f} | {r['em']*100:.2f}%({r['em_n']}) | "
          f"{r['em_a1']*100:.2f}%({r['em_a1_n']}) | {r['r28']['same']}/{r['r28']['k']} | "
          f"{r['freq']['freq20']}/20 ({100.0*r['freq']['freq20']/20:.0f}%) | "
          f"{100.0*r['prior']:.1f}%", flush=True)

for b in EXTRA_BUCKETS:
    for s in SEED_EXTRA:
        r = RESULTS.get((b, s))
        if r:
            print(f"[S14-2seed] {b} seed={s}: 数字每步正确率={r['acc']*100:.2f}%"
                  f"±{r['acc_se']*100:.2f}pp | EM={r['em']*100:.2f}%(n={r['em_n']}) | "
                  f"整体CE={r['ce']:.4f}±{r['ce_se']:.4f} | R28={r['r28']['same']}/{r['r28']['k']} | "
                  f"高频5={r['freq']['freq20']}/20", flush=True)

print("\n[VERDICT-S14] ===== ★判决（判据写死） =====", flush=True)
have = {b: RESULTS.get((b, SEED_PRIMARY)) for b in BUCKETS}
if have["add_1d"] is None:
    verdict = "★未跑出 add_1d ⇒ 无法判决（不许空报告：看日志尾部与退出码）"
elif have["add_1d"]["em"] < 0.90:
    verdict = (f"★架构性问题：add_1d EM={have['add_1d']['em']*100:.2f}% < 90% "
               f"⇒ 连一位数加法都学不会（不是难度问题）")
else:
    rest = {b: (have[b]["em"] if have[b] else float("nan")) for b in BUCKETS
            if b != "add_1d"}
    if all(v >= 0.90 for v in rest.values() if v == v):
        verdict = "★会算：五桶 EM 全部 ≥90% ⇒ GSM8K 的失败是分布偏斜/模板稀释问题，不是算术能力"
    else:
        curve = " ".join(f"{b}={have[b]['em']*100:.1f}%" for b in BUCKETS if have[b])
        verdict = f"★会算但随难度衰减：add_1d={have['add_1d']['em']*100:.1f}% ≥90%，其余桶 → {curve}"
print(f"[VERDICT-S14] 判决 = {verdict}", flush=True)
print("[VERDICT-S14] 衰减曲线（EM%）: " + " | ".join(
    f"{b}={have[b]['em']*100:.2f}% (n={have[b]['em_n']})" for b in BUCKETS if have[b]), flush=True)
print("[VERDICT-S14] 对照：均匀答案下的机会水平（最高频5答案先验）= " + " | ".join(
    f"{b}={100.0*BUCKET_INFO[b]['prior']:.1f}%" for b in BUCKETS), flush=True)

# ---------------- 收尾元数据 ----------------
print(f"\n[META] device={device} {torch.cuda.get_device_name(0)} | D={D} FF={FF} "
      f"NHEAD={NHEAD} lr={LR} batch={BATCH} steps={STEPS} MAXLEN={MAXLEN}(prompt优先) "
      f"EPS={EPS} | 生成主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH}", flush=True)
for key in sorted(RESULTS, key=lambda k: (BUCKETS.index(k[0]), k[1])):
    r = RESULTS[key]
    print(f"[META] {r['bucket']} seed={r['seed']}: steps={r['steps']} "
          f"训练墙钟={r['wall_train']/60:.1f}min 本seed合计={r['wall_total']/60:.1f}min "
          f"ckpt={r['ckpt']} sha256[:16]={r['sha']} R28={r['r28']['same']}/{r['r28']['k']}",
          flush=True)
print(f"[META] 总墙钟={(time.time()-t_start)/60:.1f}min", flush=True)
print("[DONE] exit=0", flush=True)
