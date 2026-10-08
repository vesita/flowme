#!/usr/bin/env python3
"""S19 · 判决：思维卡带残差是不是【普遍增益】？—— 5 桶 × {res/nores} × 2 seed ★GPU。

只回答一个问题：「思维卡带残差」是普遍增益，还是只对"深层推理"（进位链 add_3d）有效？

唯一变量 = 思维卡是否带残差（k=1 固定，不做循环迭代 —— S15 已证 k≥4 会崩）：
    无残差（现状 / S14 逐字相同）： h = Think(h)
    带残差（新）                  ： h = h + Think(h)      ← 只改这一行
输入卡 / 输出卡 / 参数量 / 其它一切不动（残差是纯加法，不新增参数）。

数据生成器与 S14 逐字相同（同种子 ⇒ 同 train/test），其余超参与 S14 逐字相同：
  d=128 ff=512 NHEAD=4 lr=1e-3 batch=32 3000 步 MAXLEN=512（prompt 优先）
  MIN_T=48 TAIL_KEEP=24 EPS=1e-3 生成主口径 batch=1（R28），自检 batch=16。
  五桶 add_1d / add_2d / add_3d / sub_2d / mul_2d；train 4000 / test 800。

指标：严格 EM（n=800，二项 ±SE）与数字每步正确率 = exp(−数字/算子 token 的 CE)。
配对 Δ = EM(res) − EM(nores)，逐 test 样本配对（d_i ∈ {−1,0,1}）⇒ SE = sd(d)/√800；
门槛 = 各自 2SE，且要求 2 seed 同号（都 >0 且 Δ ≥ 2SE）。

判决（写死）：
  ≥4/5 桶显著正 ⇒ 残差是普遍增益；只有 add_3d 显著 ⇒ 只救深层推理；
  都不显著 ⇒ 残差不成立（并指出与 S15 的矛盾）。
锚点：无残差 arm 必须复现 S14 的 EM（差 >1pp 点名）。

ckpt：logs/19_ckpt_<桶>_<res|nores>_seed<seed>.pt（报 sha256 前16）；
结果增量落盘 logs/19_results.jsonl。只允许写：本文件、logs/、/tmp；既有 stages/*.py 只读。
"""
from __future__ import annotations

import hashlib
import json
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

# ---------------- 配置（与 S14 逐字一致） ----------------
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
MAX_GEN = 48

SMOKE = os.environ.get("S19_SMOKE") == "1"
if SMOKE:
    STEPS, N_TRAIN, N_TEST = 30, 300, 60

TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
CKPT_TMPL = "/home/vesita/coding/my/flowme/logs/19_ckpt_{bucket}_{arm}_seed{seed}.pt"
RESULT_JSONL = "/home/vesita/coding/my/flowme/logs/19_results.jsonl"
SEEDS = (1234, 5678)
ARMS = ("nores", "res")     # nores = S14 现状；res = h + Think(h)

# S14 锚（无残差 arm 必须复现；只报 S14 实际跑过的 7 个）
S14_ANCHOR = {
    ("add_1d", 1234): 0.9662, ("add_2d", 1234): 0.7188, ("add_3d", 1234): 0.0250,
    ("sub_2d", 1234): 0.6138, ("mul_2d", 1234): 0.8262,
    ("add_1d", 5678): 0.9825, ("add_3d", 5678): 0.0962,
}

t_start = time.time()
print(f"[S19] ★残差普遍性判决：5 桶 × 2 arm(res/nores) × 2 seed = 20 run × {STEPS} 步 "
      f"| 唯一变量=思维卡是否带残差（k=1，无循环）| d={D} ff={FF} NHEAD={NHEAD} lr={LR} "
      f"batch={BATCH} MAXLEN={MAXLEN}(prompt优先) EPS={EPS} 切分=train{N_TRAIN}/test{N_TEST} "
      f"生成主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | seeds={SEEDS} | smoke={SMOKE}",
      flush=True)

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
# ① 数据：与 S14 逐字相同（同种子 ⇒ 同 train/test）
# ============================================================================
def _has_cjk(s: str) -> bool:
    return any("一" <= c <= "鿿" for c in s)


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
N_TPL_CN = sum(1 for _, c in TEMPLATES if c)
RENDER_VARIANTS = N_TPL_CN * 2 + (len(TEMPLATES) - N_TPL_CN)

CN_OPS = {"add_1d": ("加", "加上"), "add_2d": ("加", "加上"), "add_3d": ("加", "加上"),
          "sub_2d": ("减", "减去"), "mul_2d": ("乘", "乘以")}
EN_OPS = {"add_1d": ("plus",), "add_2d": ("plus",), "add_3d": ("plus",),
          "sub_2d": ("minus",), "mul_2d": ("times",)}

LOHI = {"add_1d": (0, 9), "add_2d": (10, 99), "add_3d": (100, 999)}

MUL_BY_Y: dict[int, list[tuple[int, int]]] = defaultdict(list)
for _a in range(10, 100):
    for _b in range(1, 10):
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
    """可复现的桶内随机源（避免占用 torch 全局种子）。"""

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
    tgt_len = sorted(len(r["t_full"]) for r in train)

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
# ② 模型：与 S14 唯一差异 = 思维卡是否带残差（一行）
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
    """与 S14 逐字相同，唯一差异：residual=True 时 h = h + Think(h)（k=1，不循环）。"""

    def __init__(self, d: int, ff: int, identity: bool = False, residual: bool = False):
        super().__init__()
        self.d = d
        self.residual = residual
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
            if self.residual:
                h = h + self.thought(h, src_mask=m)   # ★唯一改动：带残差
            else:
                h = self.thought(h, src_mask=m)       # ★S14 现状：无残差
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


def train_run(seed: int, train, steps: int, residual: bool):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards(D, FF, residual=residual).to(device)
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


# ---------------- 数字/算子 CE 分类口径（逐字沿用 S10/S14） ----------------
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
          f"{same}/{len(sub)} ⇒ {'一致（batch=1 口径自洽）' if same == len(sub) else '★不一致 ⇒ 仅 batch=1 口径可用（本单元 EM 全部走 batch=1）'}",
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


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ============================================================================
# ③ 主循环：5 桶 × 2 seed × 2 arm（nores 与 res 紧挨着跑 ⇒ 配对最干净）
# ============================================================================
JOBS: list[tuple[str, str, int]] = [(b, arm, s) for b in BUCKETS for s in SEEDS
                                    for arm in ARMS]
RESULTS: dict[tuple[str, str, int], dict] = {}

print(f"\n[S19] ===== params d={D} ff={FF} ⇒ {n_params(D, FF)/1e6:.2f}M | "
      f"五桶={list(BUCKETS)} | arms={ARMS} seeds={SEEDS} | JOBS={len(JOBS)} =====", flush=True)


def append_jsonl(obj: dict) -> None:
    with open(RESULT_JSONL, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def run_one(bucket: str, arm: str, seed: int) -> None:
    train, test = DATA[bucket]["train"], DATA[bucket]["test"]
    info = BUCKET_INFO[bucket]
    t0 = time.time()
    print(f"\n[S19] ===== 桶 {bucket} arm={arm} seed={seed} steps={STEPS} train={len(train)} "
          f"test={len(test)} =====", flush=True)
    model, wall = train_run(seed, train, STEPS, residual=(arm == "res"))

    cm, cs = mean_se(eval_ce(model, test))
    txt = greedy_gen(model, test, batch=EM_BATCH)          # ★R28：主口径 batch=1
    hits = hits_from(txt, test)
    rep = report_ce(f"{bucket} {arm} seed={seed}", model, test)
    em = em_report(f"{bucket} {arm} seed={seed} (batch={EM_BATCH})", hits)
    r28 = r28_selfcheck(model, test, tag=f"[{bucket} {arm} seed={seed}] ")

    ck = CKPT_TMPL.format(bucket=bucket, arm=arm, seed=seed)
    if not SMOKE:
        torch.save(model.state_dict(), ck)
        dig = sha16(ck)
        print(f"[CKPT] saved {ck} sha256[:16]={dig}", flush=True)
    else:
        dig = "-"
    total = time.time() - t0
    del model
    torch.cuda.empty_cache()

    r = dict(bucket=bucket, arm=arm, seed=seed, steps=STEPS, wall_train=wall,
             wall_total=total, ce=cm, ce_se=cs, acc=rep.get("acc"),
             acc_se=rep.get("acc_se"), dsamp=rep.get("dig_samp_m"),
             dsamp_se=rep.get("dig_samp_se"), dig_tok=rep.get("dig_ce"),
             em=em["全量"]["em"], em_se=em["全量"]["se"], em_n=em["全量"]["n"],
             em_a1=em["桶A1"]["em"], r28_same=r28["same"], r28_k=r28["k"],
             ckpt=ck, sha=dig, prior=info["prior"], ans_n=info["ans_n"],
             strict=[h["strict"] for h in hits])
    RESULTS[(bucket, arm, seed)] = r
    append_jsonl({k: v for k, v in r.items() if k != "strict"})
    print(f"[MAIN-TABLE] {bucket} {arm} seed={seed} | EM全量={r['em']*100:.2f}%±{r['em_se']*100:.2f}"
          f"(n={r['em_n']}) | 数字每步正确率={r['acc']*100:.2f}%±{r['acc_se']*100:.2f}pp | "
          f"整体CE={r['ce']:.4f}±{r['ce_se']:.4f} | R28={r['r28_same']}/{r['r28_k']} | "
          f"训练={wall/60:.1f}min 合计={total/60:.1f}min | ckpt={os.path.basename(ck)} sha16={dig}",
          flush=True)


for job in JOBS:
    run_one(*job)

# ============================================================================
# ④ 汇总 + ★配对 Δ + ★判决
# ============================================================================
def paired_delta(bucket: str, seed: int) -> dict | None:
    a = RESULTS.get((bucket, "res", seed))
    b = RESULTS.get((bucket, "nores", seed))
    if a is None or b is None:
        return None
    xs = [ra - rb for ra, rb in zip(a["strict"], b["strict"])]
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return dict(n=n, d=m, se=se, ratio=m / se if se > 0 else float("inf"),
                sig=(m > 0 and m >= 2 * se))


print("\n[S19] ===== ★五桶主表（2 arm × 2 seed） =====", flush=True)
print("[S19] 桶 | arm | seed | EM全量(n)±SE | 数字每步正确率±SE | 整体CE±SE | R28 | ckpt sha16",
      flush=True)
for b in BUCKETS:
    for arm in ARMS:
        for s in SEEDS:
            r = RESULTS.get((b, arm, s))
            if r is None:
                print(f"[S19] {b} {arm} seed={s}: ★未完成", flush=True)
                continue
            print(f"[S19] {b} | {arm} | {s} | {r['em']*100:.2f}%±{r['em_se']*100:.2f}(n={r['em_n']}) | "
                  f"{r['acc']*100:.2f}%±{r['acc_se']*100:.2f}pp | "
                  f"{r['ce']:.4f}±{r['ce_se']:.4f} | {r['r28_same']}/{r['r28_k']} | {r['sha']}",
                  flush=True)

print("\n[S19] ===== ★每桶配对 Δ = EM(res) − EM(nores)（逐 test 样本配对，门槛 2×SE） =====",
      flush=True)
print("[S19] 桶 | seed | Δ±SE(pp) | Δ/SE | 判定 | nores EM | res EM", flush=True)
DELTA: dict[str, list[dict]] = {}
for b in BUCKETS:
    DELTA[b] = []
    for s in SEEDS:
        d = paired_delta(b, s)
        if d is None:
            print(f"[S19] {b} seed={s}: ★缺 run ⇒ 无法配对", flush=True)
            continue
        nores = RESULTS[(b, "nores", s)]["em"]
        res = RESULTS[(b, "res", s)]["em"]
        DELTA[b].append(dict(seed=s, **d))
        print(f"[S19] {b} | {s} | {d['d']*100:+.2f}±{d['se']*100:.2f}pp | "
              f"{d['ratio']:.2f} | {'★显著正' if d['sig'] else '未过 2SE 门槛'} | "
              f"{nores*100:.2f}% → {res*100:.2f}%", flush=True)

print("\n[S19] ===== ★每桶同号判定（2 seed） =====", flush=True)
BUCKET_SIG: dict[str, bool] = {}
for b in BUCKETS:
    ds = DELTA[b]
    if len(ds) < len(SEEDS):
        BUCKET_SIG[b] = False
        print(f"[S19] {b}: run 不全 ⇒ 不判显著", flush=True)
        continue
    signs = [1 if d["d"] > 0 else (-1 if d["d"] < 0 else 0) for d in ds]
    same_sign = len(set(signs)) == 1 and signs[0] == 1
    sig = same_sign and all(d["sig"] for d in ds)
    BUCKET_SIG[b] = sig
    print(f"[S19] {b}: Δ(1234)={ds[0]['d']*100:+.2f}pp(SE {ds[0]['se']*100:.2f}) "
          f"Δ(5678)={ds[1]['d']*100:+.2f}pp(SE {ds[1]['se']*100:.2f}) | "
          f"{'同号(正)' if same_sign else '不同号或非正'} | "
          f"{'★两 seed 均 Δ≥2SE ⇒ 该桶显著' if sig else '未达两 seed Δ≥2SE'}", flush=True)

# ---------------- 锚点复核（无残差 arm vs S14） ----------------
print("\n[S19] ===== ★锚点复核：无残差 arm 必须复现 S14 =====", flush=True)
ANCHOR_OK = True
if SMOKE:
    print("[S19] smoke 模式（30 步）⇒ 跳过锚点复核（与 3000 步的 S14 不可比）", flush=True)
for (b, s), ref in ({} if SMOKE else S14_ANCHOR.items()):
    r = RESULTS.get((b, "nores", s))
    if r is None:
        print(f"[S19] 锚 {b} seed={s}: ★无 run", flush=True)
        ANCHOR_OK = False
        continue
    dev = (r["em"] - ref) * 100
    ok = abs(dev) <= 1.0
    ANCHOR_OK &= ok
    print(f"[S19] 锚 {b} seed={s}: S14={ref*100:.2f}% vs nores={r['em']*100:.2f}% "
          f"Δ={dev:+.2f}pp ⇒ {'一致(|Δ|≤1pp)' if ok else '★偏离 >1pp（点名）'}", flush=True)

# ---------------- ★判决（判据写死） ----------------
n_sig = sum(1 for b in BUCKETS if BUCKET_SIG[b])
sig_list = [b for b in BUCKETS if BUCKET_SIG[b]]
print("\n[VERDICT-S19] ===== ★判决（判据写死） =====", flush=True)
print(f"[VERDICT-S19] 显著正的桶 = {sig_list} （{n_sig}/5）；判据：Δ≥2×SE(配对) 且 2 seed 同号(正)",
      flush=True)
if not all((b, arm, s) in RESULTS for b in BUCKETS for arm in ARMS for s in SEEDS):
    verdict = "★未跑满 20 run ⇒ 无法判决（不许空报告：看日志尾部与退出码）"
elif n_sig >= 4:
    verdict = (f"★残差是【普遍增益】：{n_sig}/5 桶显著正（{sig_list}，均 Δ≥2SE 且 2 seed 同号）"
               f"⇒ 应写进框架规范（思维卡必须带残差）")
elif n_sig == 1 and BUCKET_SIG["add_3d"]:
    d3d = " / ".join(f"{d['d']*100:+.2f}±{d['se']*100:.2f}pp" for d in DELTA["add_3d"])
    others = [b for b in BUCKETS if b != "add_3d"]
    otxt = " ".join(f"{b}={'显著' if BUCKET_SIG[b] else '不显著'}" for b in others)
    verdict = (f"★残差【只救深层推理】：仅 add_3d 显著（Δ={d3d}），其余四桶不显著（{otxt}）"
               f"⇒ 如实报：不写成普遍增益")
elif n_sig == 0:
    nores_txt = " ".join(f"{b}Δ(1234)={DELTA[b][0]['d']*100:+.2f}pp" for b in BUCKETS)
    verdict = (f"★残差【不成立】：5 桶均未过 2SE 门槛（{nores_txt}）"
               f"⇒ 与 S15（add_3d 残差单变量 +34.75/+21.51pp）矛盾：S15 那次是 "
               f"after_first↔always 的口径差，本单元两 arm 均 k=1 且其余与 S14 逐字同 —— "
               f"矛盾点需点名，如实报")
else:
    verdict = (f"★部分桶显著（{n_sig}/5：{sig_list}）：既不足 4 桶判「普遍增益」，"
               f"也不是「只有 add_3d」⇒ 如实报：残差增益局限于 {sig_list}")
print(f"[VERDICT-S19] 判决 = {verdict}", flush=True)
print(f"[VERDICT-S19] 锚点复核 = {'全部一致（|Δ|≤1pp）' if ANCHOR_OK else '★有锚点偏离 >1pp（见上）'}",
      flush=True)

# ---------------- 收尾元数据 ----------------
print(f"\n[META] device={device} {torch.cuda.get_device_name(0)} | D={D} FF={FF} "
      f"NHEAD={NHEAD} lr={LR} batch={BATCH} steps={STEPS} MAXLEN={MAXLEN}(prompt优先) "
      f"EPS={EPS} | 生成主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | 残差=res arm 唯一变量",
      flush=True)
for key in sorted(RESULTS, key=lambda k: (BUCKETS.index(k[0]), k[1], k[2])):
    r = RESULTS[key]
    print(f"[META] {r['bucket']} {r['arm']} seed={r['seed']}: steps={r['steps']} "
          f"训练墙钟={r['wall_train']/60:.1f}min 本run合计={r['wall_total']/60:.1f}min "
          f"ckpt={r['ckpt']} sha256[:16]={r['sha']} R28={r['r28_same']}/{r['r28_k']}",
          flush=True)
print(f"[META] 总墙钟={(time.time()-t_start)/60:.1f}min | run数={len(RESULTS)}/20", flush=True)
print("[DONE] exit=0", flush=True)
