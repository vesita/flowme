#!/usr/bin/env python3
"""S15 · 判决性实验：思维卡「按位迭代」能不能救三位数加法？（进位链 = 跨位状态传递）★GPU。

只回答一个问题：在 add_3d（三位数加法，含进位）上，把思维卡的迭代次数 k 从 1 加到 2/4/8，EM 会不会显著上升？

唯一变量 = 思维卡迭代次数 k ∈ {1,2,4,8}；数据/模型/超参与 S14 **逐字全同**：
    h = Enc_in(x)                      # 输入卡 → 模因
    for i in range(k): h = Think(h)+h  # ★同一份 Think 参数循环 k 次（权重共享，残差）
    out = head(h)                      # 输出卡
参数量与 k 无关（权重共享）⇒ 2.50M。

数据：add_3d（+ add_2d 对照），train4000/test800，**同 S14 生成器同种子**（桶序/`Rng(14000+bi*7)` 不变）。
超参：d=128 ff=512 NHEAD=4 lr=1e-3 batch=32 3000 步 MAXLEN=512(prompt优先) EPS=1e-3 生成主口径 batch=1（R28）。

锚点门（k=1 必须复现 S14 add_3d 基线）：seed1234 EM≈2.50%，|Δ|>5pp ⇒ 自动切换残差口径
复测一次，仍不过 ⇒ exit 4 停（不往下跑）。残差口径：
    always      = 每轮都加残差（规格字面写法）
    after_first = 首轮不加（k=1 与 S14 逐位等价 ⇒ 严格可比）
S14 基线：add_3d EM 2.50%(seed1234) / 9.62%(seed5678)、数字每步 22.05% / 28.72%；
          add_2d EM 71.88%(seed1234)。

ckpt：logs/15_ckpt_<add3d|add2d>_k<k>_seed<seed>.pt（报 sha256 前16）；结果增量落盘 logs/15_results.jsonl。
只允许写：本文件、logs/、/tmp；既有 stages/*.py 与 nanoSeek 只读。
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

# ---------------- 配置（与 A 臂 / S14 逐字一致，唯一变量 = K） ----------------
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

K_VALUES = (1, 2, 4, 8)
SEEDS = (1234, 5678)
BUCKET = "add_3d"          # 主判据桶
CTRL_BUCKET = "add_2d"     # 对照桶（看迭代是否只对进位链有用）

SMOKE = os.environ.get("S15_SMOKE") == "1"
if SMOKE:
    STEPS, N_TRAIN, N_TEST = 30, 300, 60

TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
CKPT_TAG = {"add_3d": "add3d", "add_2d": "add2d"}
CKPT_TMPL = "/home/vesita/coding/my/flowme/logs/15_ckpt_{tag}{sfx}_k{k}_seed{seed}.pt"
RESULT_JSONL = "/home/vesita/coding/my/flowme/logs/15_results.jsonl"

# 锚点（S14 实测基线）与停止边界
ANCHOR_EM = {1234: 0.0250, 5678: 0.0962}
ANCHOR_TOL = 0.05          # |EM − S14| > 5pp ⇒ 停

# 残差口径：由锚点门决定（见模块 docstring）；S15_MODE 给定时固定该口径（补充扫描用）
MODE = "always"
MODE_ENV = os.environ.get("S15_MODE")
if MODE_ENV:
    MODE = MODE_ENV
# 补充扫描的 ckpt 加口径后缀，避免覆盖主扫描（规格名 15_ckpt_<桶>_k<k>_seed<s>.pt 属主扫描）
CKPT_SFX = f"_{MODE}" if MODE_ENV else ""
t_start = time.time()
print(f"[S15] ★判决性实验：思维卡按位迭代 k={K_VALUES}（唯一变量=k，数据/超参全同） d={D} ff={FF} "
      f"steps={STEPS} lr={LR} batch={BATCH} MAXLEN={MAXLEN}(prompt优先) NHEAD={NHEAD} EPS={EPS} "
      f"切分=train{N_TRAIN}/test{N_TEST} 生成主口径=batch{EM_BATCH} | 主桶={BUCKET} 对照={CTRL_BUCKET} "
      f"| 残差口径={MODE} | smoke={SMOKE}", flush=True)

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
      f"分隔符 '→' 词表外 ⇒ 单 token <unk>(id={_unk})，实测 enc('→')={_arrow_ids}", flush=True)


# ============================================================================
# ① 数据：程序生成的算术合成集（与 S14 逐字同源同种子；只取 add_3d / add_2d）
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


class Rng:
    """可复现的桶内随机源（同 S14）。"""

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


BUCKETS = ("add_1d", "add_2d", "add_3d", "sub_2d", "mul_2d")   # 桶序必须与 S14 一致（决定数据种子）
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
    top5 = [v for v, _ in Counter(r["gold"] for r in train).most_common(5)]
    prior = sum(1 for r in test if r["gold"] in top5) / max(1, len(test))
    cut = sum(1 for r in train + test if r["cut_p"] or r["cut_t"])
    anyc, unitc = carry_rate(name, 300, Rng(77 + bi))

    DATA[name] = dict(train=train, test=test)
    BUCKET_INFO[name] = dict(top5=top5, prior=prior, ans_n=len(ys),
                             ctr_tr=ctr_tr, ctr_te=ctr_te, anyc=anyc, unitc=unitc)
    if name in (BUCKET, CTRL_BUCKET):
        print(f"[DATA] {name}: train={len(train)} test={len(test)} 文本零重叠 | "
              f"答案域={min(ys)}..{max(ys)}({len(ys)}个) 计数train min/max="
              f"{min(ctr_tr.values())}/{max(ctr_tr.values())} | 截断={cut} | "
              f"进位率(至少一次/个位)={anyc:.2f}/{unitc:.2f} | "
              f"★最高频5答案={top5} 机会水平={100.0*prior:.1f}%", flush=True)

# 与 S14 的数据一致性自检（同一生成器同一种子 ⇒ 关键统计必须逐字一致）
_S14_CHECK = {   # bucket: (答案域个数, 机会水平(%), 进位率(至少一次/个位), 最高频5答案) —— 均取 S14 日志显示口径
    "add_3d": (1799, "0.2", ("0.87", "0.43"), ['706', '1130', '977', '537', '926']),
    "add_2d": (179, "2.8", ("0.76", "0.46"), ['167', '68', '115', '102', '151']),
}
for _b, (_n, _p1, (_ac, _uc), _t5) in _S14_CHECK.items():
    _i = BUCKET_INFO[_b]
    _got = (f"{100*_i['prior']:.1f}", f"{_i['anyc']:.2f}", f"{_i['unitc']:.2f}")
    _ok = ((SMOKE and _i["ans_n"] == _n)          # smoke 下 test 仅 60 条 ⇒ 机会水平/top5 不可比
           or (_i["ans_n"] == _n and _i["top5"] == _t5 and _got == (_p1, _ac, _uc)))
    print(f"[DATA-CHECK] {_b}: 答案域={_i['ans_n']}(S14={_n}) 机会水平={_got[0]}%(S14={_p1}%) "
          f"进位率={_got[1]}/{_got[2]}(S14={_ac}/{_uc}) top5与S14一致={_i['top5'] == _t5} "
          f"⇒ {'一致 ✓' if _ok else '★不一致 ⇒ 立即停'}", flush=True)
    if not _ok:
        sys.exit(5)


# ============================================================================
# ② 模型：与 A 臂同骨架，思维卡改成「权重共享循环」k 次
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
    """唯一改动：思维卡从「一次前向」改成权重共享的循环 k 次（残差）。"""

    def __init__(self, d: int, ff: int, k: int = 1, mode: str = "always"):
        super().__init__()
        self.d = d
        self.k = k
        self.mode = mode
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        self.thought = enc_layer(d, ff)          # ★同一份参数循环 k 次
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
        for i in range(self.k):                   # ★按位迭代（权重共享）
            out = self.thought(h, src_mask=m)
            h = out + h if (self.mode == "always" or i > 0) else out
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


def train_run(seed: int, train, steps: int, k: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards(D, FF, k=k, mode=MODE).to(device)
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


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ============================================================================
# ③ 主循环：add_3d × k∈{1,2,4,8} × 2 seed（锚点 k=1 先跑先判），再补 add_2d 对照
# ============================================================================
RESULTS: dict[tuple[str, int, int], dict] = {}


def save_jsonl(rec: dict) -> None:
    with open(RESULT_JSONL, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_one(bucket: str, k: int, seed: int) -> dict:
    train, test = DATA[bucket]["train"], DATA[bucket]["test"]
    info = BUCKET_INFO[bucket]
    t0 = time.time()
    print(f"\n[S15] ===== 桶 {bucket} k={k} seed={seed} steps={STEPS} 残差口径={MODE} "
          f"train={len(train)} test={len(test)} =====", flush=True)
    model, wall = train_run(seed, train, STEPS, k)

    rep = report_ce(f"{bucket} k={k} seed={seed}", model, test)
    txt = greedy_gen(model, test, batch=EM_BATCH)
    hits = hits_from(txt, test)
    em = em_report(f"{bucket} k={k} seed={seed} (batch={EM_BATCH})", hits)
    r28 = r28_selfcheck(model, test, tag=f"[{bucket} k={k} seed={seed}] ")

    ck = CKPT_TMPL.format(tag=CKPT_TAG[bucket], sfx=CKPT_SFX, k=k, seed=seed)
    if not SMOKE:
        torch.save(model.state_dict(), ck)
        dig = sha16(ck)
        print(f"[CKPT] saved {ck} sha256[:16]={dig}", flush=True)
    else:
        dig = "-"
    total = time.time() - t0
    del model
    torch.cuda.empty_cache()

    rec = dict(bucket=bucket, k=k, seed=seed, mode=MODE, steps=STEPS,
               wall_train=round(wall, 1), wall_total=round(total, 1),
               ce=rep["ce_m"], ce_se=rep["ce_se"],
               acc=rep.get("acc"), acc_se=rep.get("acc_se"),
               dsamp=rep.get("dig_samp_m"), dsamp_se=rep.get("dig_samp_se"),
               em=em["全量"]["em"], em_se=em["全量"]["se"], em_n=em["全量"]["n"],
               r28_same=r28["same"], r28_k=r28["k"], ckpt=ck, sha=dig,
               prior=info["prior"])
    RESULTS[(bucket, k, seed)] = rec
    save_jsonl(rec)
    print(f"[MAIN-TABLE] {bucket} k={k} seed={seed} | EM全量={rec['em']*100:.2f}%"
          f"±{rec['em_se']*100:.2f}(n={rec['em_n']}) | 数字每步正确率={rec['acc']*100:.2f}%"
          f"±{rec['acc_se']*100:.2f}pp | 整体CE={rec['ce']:.4f}±{rec['ce_se']:.4f} | "
          f"R28={rec['r28_same']}/{rec['r28_k']} | 训练={wall/60:.1f}min 合计={total/60:.1f}min | "
          f"ckpt={os.path.basename(ck)} sha16={dig}", flush=True)
    return rec


# —— 锚点门：k=1 必须复现 S14 add_3d 基线（seed1234 EM≈2.50%，|Δ|>5pp ⇒ 停） ——
def anchor_gate() -> None:
    global MODE
    rec = run_one(BUCKET, 1, 1234)
    em = rec["em"]
    if SMOKE:
        print(f"[ANCHOR] smoke 模式 ⇒ 跳过锚点判定（EM={em*100:.2f}%）", flush=True)
        return
    if abs(em - ANCHOR_EM[1234]) <= ANCHOR_TOL:
        print(f"[ANCHOR] ✓ 通过：k=1 seed1234 EM={em*100:.2f}% vs S14 2.50% "
              f"（|Δ|={abs(em-ANCHOR_EM[1234])*100:.2f}pp ≤ {ANCHOR_TOL*100:.0f}pp）"
              f" ⇒ 残差口径={MODE}，继续往下跑", flush=True)
        return
    print(f"[ANCHOR] ★ 未通过：k=1 seed1234 EM={em*100:.2f}% vs S14 2.50% "
          f"（|Δ|={abs(em-ANCHOR_EM[1234])*100:.2f}pp > {ANCHOR_TOL*100:.0f}pp）"
          f" ⇒ 「每轮加残差」破坏了 k=1 的可比性，切换为 after_first（k=1 ≡ 原设定）复测", flush=True)
    MODE = "after_first"
    rec = run_one(BUCKET, 1, 1234)
    em = rec["em"]
    if abs(em - ANCHOR_EM[1234]) <= ANCHOR_TOL:
        print(f"[ANCHOR] ✓ 复测通过：k=1 seed1234 EM={em*100:.2f}% vs S14 2.50% "
              f"（|Δ|={abs(em-ANCHOR_EM[1234])*100:.2f}pp）⇒ 残差口径={MODE}，继续往下跑"
              f"（注意：该口径下 k=1 与 S14 逐位等价，k≥2 才加残差）", flush=True)
        return
    print(f"[ANCHOR] ★★ 复测仍不通过：EM={em*100:.2f}% vs S14 2.50% ⇒ 停止，不往下跑。", flush=True)
    print("[ANCHOR] ★★ 中止（可比性锚点不成立）：exit=4", flush=True)
    sys.exit(4)


print(f"\n[S15] ===== params d={D} ff={FF} ⇒ {n_params(D, FF)/1e6:.2f}M（权重共享 ⇒ 与 k 无关）"
      f" | 桶={BUCKET} k={K_VALUES} seeds={SEEDS} =====", flush=True)

JOBS: list[tuple[str, int, int]] = [(BUCKET, 1, 1234)]          # ① 锚点（先跑先判）
JOBS.append((BUCKET, 1, 5678))                                  # ② 锚点第二 seed（优先 k=1,4,8）
for _k in (4, 8, 2):                                            # ③ 优先 k=4/8，最后补 k=2
    for _s in SEEDS:
        JOBS.append((BUCKET, _k, _s))
for _s in SEEDS:                                                # ④ 对照：add_2d k=1 vs k=4
    JOBS.append((CTRL_BUCKET, 1, _s))
    JOBS.append((CTRL_BUCKET, 4, _s))

if SMOKE:
    JOBS = [(BUCKET, 1, 1234), (BUCKET, 4, 1234), (CTRL_BUCKET, 1, 1234)]

# 补充扫描：S15_JOBS="桶:k:seed,..." 指定 job（此时跳过锚点门 —— 口径已由主扫描锚定）
JOBS_ENV = os.environ.get("S15_JOBS")
if JOBS_ENV:
    JOBS = [(p[0], int(p[1]), int(p[2]))
            for p in (j.split(":") for j in JOBS_ENV.split(","))]
    print(f"[S15] ★补充扫描：残差口径={MODE} JOBS={JOBS}（跳过锚点门，ckpt 后缀={CKPT_SFX!r}）",
          flush=True)

if JOBS_ENV:
    for bucket, k, seed in JOBS:
        run_one(bucket, k, seed)
else:
    anchor_gate()
    for bucket, k, seed in JOBS[1:]:
        run_one(bucket, k, seed)

# 第二个锚点（seed5678）复现核对：不作硬停止，只报偏离
a5678 = RESULTS.get((BUCKET, 1, 5678))
if a5678 and not SMOKE:
    d5 = abs(a5678["em"] - ANCHOR_EM[5678]) * 100
    print(f"[ANCHOR] k=1 seed5678 复现核对：EM={a5678['em']*100:.2f}% vs S14 9.62% "
          f"（|Δ|={d5:.2f}pp）⇒ {'一致 ✓' if d5 <= ANCHOR_TOL*100 else '★偏离 >5pp（已记录，见报告）'}",
          flush=True)

# ============================================================================
# ④ 汇总 + ★判决
# ============================================================================
def g(bucket: str, k: int, seed: int) -> dict | None:
    return RESULTS.get((bucket, k, seed))


print("\n[S15] ===== ★主表 add_3d（唯一变量 = k） =====", flush=True)
print("[S15] k | seed | EM全量(n=800)±SE | 数字每步正确率±SE | 整体CE±SE | R28 | 训练墙钟 | ckpt sha16",
      flush=True)
for k in K_VALUES:
    for s in SEEDS:
        r = g(BUCKET, k, s)
        if r is None:
            print(f"[S15] k={k} seed={s} | ★未完成（时间不够）", flush=True)
            continue
        print(f"[S15] k={k} | {s} | {r['em']*100:.2f}%±{r['em_se']*100:.2f} (n={r['em_n']}) | "
              f"{r['acc']*100:.2f}%±{r['acc_se']*100:.2f}pp | "
              f"{r['ce']:.4f}±{r['ce_se']:.4f} | {r['r28_same']}/{r['r28_k']} | "
              f"{r['wall_train']/60:.1f}min | {r['sha']}", flush=True)

print("\n[S15] ===== ★对照 add_2d（迭代是否只对进位链有用） =====", flush=True)
for k in (1, 4):
    for s in SEEDS:
        r = g(CTRL_BUCKET, k, s)
        if r is None:
            print(f"[S15] add_2d k={k} seed={s} | ★未完成", flush=True)
            continue
        print(f"[S15] add_2d k={k} | {s} | EM={r['em']*100:.2f}%±{r['em_se']*100:.2f} "
              f"(n={r['em_n']}) | 数字每步={r['acc']*100:.2f}%±{r['acc_se']*100:.2f}pp | "
              f"R28={r['r28_same']}/{r['r28_k']} | {r['wall_train']/60:.1f}min | {r['sha']}", flush=True)

print("\n[VERDICT-S15] ===== ★判决（判据写死：k=8 的 EM ≥ k=1 + 2×SE 且 2 seed 同号） =====", flush=True)
need = [(BUCKET, k, s) for k in K_VALUES for s in SEEDS]
missing = [x for x in need if x not in RESULTS]
if missing:
    print(f"[VERDICT-S15] ★未跑全（缺 {len(missing)} 项）⇒ 不作最终判决；已完成项见上表。", flush=True)
else:
    deltas, se_thr, pass_seed = {}, {}, {}
    for s in SEEDS:
        r1, r8 = g(BUCKET, 1, s), g(BUCKET, 8, s)
        thr = 2 * r1["em_se"]
        deltas[s] = r8["em"] - r1["em"]
        se_thr[s] = thr
        pass_seed[s] = r8["em"] >= r1["em"] + thr
        print(f"[VERDICT-S15] seed={s}: EM(k=8)={r8['em']*100:.2f}% vs EM(k=1)={r1['em']*100:.2f}% "
              f"⇒ Δ={deltas[s]*100:+.2f}pp，门槛 2×SE={thr*100:.2f}pp ⇒ "
              f"{'过门槛 ✓' if pass_seed[s] else '未过门槛 ✗'}", flush=True)
    ks = [k for k in K_VALUES
          if all(g(BUCKET, k, s)["em"] >= g(BUCKET, 1, s)["em"] + se_thr[s] for s in SEEDS)]
    if all(pass_seed.values()):
        kstar = min(ks) if ks else 8
        print(f"[VERDICT-S15] 判决 = ★「缺迭代」成立：k=8 显著高于 k=1 且 2 seed 同号；"
              f"拐点 k*={kstar}；Δ8 = " +
              " / ".join(f"seed{s} {deltas[s]*100:+.2f}pp" for s in SEEDS), flush=True)
    else:
        same_sign = (deltas[1234] > 0) == (deltas[5678] > 0)
        print(f"[VERDICT-S15] 判决 = ★「迭代不是解」：k=8 未达 k=1+2×SE "
              f"（Δ8 = " + " / ".join(f"seed{s} {deltas[s]*100:+.2f}pp" for s in SEEDS) +
              f"）｜2 seed {'同号' if same_sign else '异号'} ⇒ 全程无显著变化，如实报。",
              flush=True)

# —— add_2d 对照结论 ——
c1 = [(g(CTRL_BUCKET, 1, s), g(CTRL_BUCKET, 4, s)) for s in SEEDS]
if all(a and b for a, b in c1):
    for s in SEEDS:
        a, b = g(CTRL_BUCKET, 1, s), g(CTRL_BUCKET, 4, s)
        thr = 2 * a["em_se"]
        print(f"[VERDICT-S15] add_2d 对照 seed={s}: k=4 {b['em']*100:.2f}% vs k=1 "
              f"{a['em']*100:.2f}% ⇒ Δ={(b['em']-a['em'])*100:+.2f}pp，门槛={thr*100:.2f}pp ⇒ "
              f"{'过门槛 ✓' if b['em'] >= a['em'] + thr else '未过门槛 ✗'}", flush=True)

# ---------------- 收尾元数据 ----------------
print(f"\n[META] device={device} {torch.cuda.get_device_name(0)} | D={D} FF={FF} NHEAD={NHEAD} "
      f"lr={LR} batch={BATCH} steps={STEPS} MAXLEN={MAXLEN}(prompt优先) EPS={EPS} | "
      f"生成主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | 权重共享(参数={n_params(D,FF)/1e6:.2f}M) "
      f"| 残差口径={MODE} | 唯一变量=k{K_VALUES}", flush=True)
for key in sorted(RESULTS, key=lambda x: (x[0], x[1], x[2])):
    r = RESULTS[key]
    print(f"[META] {r['bucket']} k={r['k']} seed={r['seed']}: steps={r['steps']} "
          f"训练墙钟={r['wall_train']/60:.1f}min 本run合计={r['wall_total']/60:.1f}min "
          f"ckpt={r['ckpt']} sha256[:16]={r['sha']} R28={r['r28_same']}/{r['r28_k']} "
          f"EM={r['em']*100:.2f}% 数字每步={r['acc']*100:.2f}%±{r['acc_se']*100:.2f}pp", flush=True)
print(f"[META] 完成 {len(RESULTS)}/{len(JOBS)} run | 总墙钟={(time.time()-t_start)/60:.1f}min", flush=True)
print("[DONE] exit=0", flush=True)
