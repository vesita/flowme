#!/usr/bin/env python3
"""S13 · 容量 vs 训练量：把 A 臂「终局答案数字的每步正确率 ≈11%」的瓶颈分开。

★ 只改两个变量，其余全部沿用 A 臂（stages/12_answer_only.py）设定：
    ① 纯容量   key=`d256_s3000`  : D=256（FF=4d=1024），**步数仍 3000**
    ② 纯训练量 key=`d128_s10000` : D=128（FF=512），**步数按墙钟预案决定**（10000 或 6000）
  两点 × 2 seed（1234、5678）⇒ 与已有基线（d=128 / 3000 步，logs/12_answer.log）比。
  不跑 2×2 全网格；**不重跑门 B**（门 B 效应量强依赖输出形态）；不做路由/不做 scheduled sampling。

沿用 A 臂不动：NHEAD=4、MAXLEN=512、MIN_T=48、TAIL_KEEP=24、LR=1e-3、BATCH=32、
EPS=1e-3、9:1 切分（seed42）、截断优先级（prompt 优先）、四类 token 分类口径、
整体配对 CE / 数字 CE（token 加权 + 样本级 ±SE）/ 每步正确率 / 分桶 EM / 贪心解码实现。

★主指标（唯一干净指标）= 终局答案数字的每步正确率 = exp(−数字/算子 CE)，用**样本级** ±SE 判。

墙钟预案（写死，不许悄悄改别的）：
  ② 若单 seed >60min ⇒ 降到 6000 步。本脚本**先实测** d=128 的 s/step（300 步同构探针，
  只用形状/批量/优化器同构的训练步），外推 10000 步 + 评估开销；>60min 即落 6000 并打印推导。

R28（自回归生成仪器）：生成主口径 **batch=1**（正确写法）；每个训练完的模型各报一次
batch=1 vs batch=16 逐字一致率 + 一次随机初始化的结构自检；batch=N 只作参考、不进结论。

高频整数倾向：抽 20 条生成，报「抽取答案 ∈ {10,100,12,20,24}」的比例（另报全 test 比例作参考）。

ckpt：logs/13_ckpt_{d256_s3000,d128_s10000}_seed{1234,5678}.pt
只允许写：本文件、logs/、/tmp；nanoSeek / 既有 stages/*.py 只读。
"""
from __future__ import annotations

import math
import os
import re
import sys
import time
from collections import Counter

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

# ---------------- 配置（★只改 d 与步数，其余与 A 臂逐字相同） ----------------
NHEAD = 4
MAXLEN = 512
MIN_T = 48
TAIL_KEEP = 24
LR = 1e-3
SEEDS = (1234, 5678)
EPS = 1e-3
BATCH = 32
GEN_BATCH = 32
EM_BATCH = 1          # ★主口径：batch=1（R28）
CHK_BATCH = 16        # 自检/参考口径

CONFIGS = [
    dict(key="d256_s3000", d=256, ff=1024, steps=3000, var="纯容量 d=256, 步数仍 3000"),
    dict(key="d128_s10000", d=128, ff=512, steps=10000, var="纯训练量 d=128, 步数 10000(或 6000)"),
]

DATA = NS_ROOT + "/data/chinese/clean_v3/gsm8k_cot_dialogue.txt"
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
CKPT_TMPL = "/home/vesita/coding/my/flowme/logs/13_ckpt_{key}_seed{seed}.pt"

# 高频整数集合（本单元要报的比例口径）
FREQ_SET = frozenset({"10", "100", "12", "20", "24"})

# ---------------- 基线（logs/12_answer.log，d=128 / 3000 步，已入库不重跑） ----------------
BASE = {
    1234: dict(dig_tok=2.1282, dsamp=2.1947, dsamp_se=0.0198,
               ce=0.6924, ce_se=0.0072, em=0.0217, em_n=739, em_a1=0.0207, em_a1_n=726),
    5678: dict(dig_tok=2.1427, dsamp=2.1975, dsamp_se=0.0228,
               ce=0.6844, ce_se=0.0083, em=0.0217, em_n=739, em_a1=0.0207, em_a1_n=726),
}
for _s in BASE.values():
    _s["acc"] = math.exp(-_s["dsamp"])            # 样本级数字每步正确率
    _s["acc_se"] = _s["acc"] * _s["dsamp_se"]     # delta 法
    _s["acc_tok"] = math.exp(-_s["dig_tok"])      # token 加权（只作参考，无基线 SE）

PROBE = os.environ.get("S13_PROBE") == "1"
t_start = time.time()

print(f"[S13] ★容量 vs 训练量：两点各×2 seed。变量化 = D 与 STEPS；"
      f"其余 A 臂设定不动：NHEAD={NHEAD} MAXLEN={MAXLEN} MIN_T={MIN_T} TAIL_KEEP={TAIL_KEEP} "
      f"lr={LR} batch={BATCH} eps={EPS} 切分=seed42(9:1) 生成主口径=batch{EM_BATCH}。"
      f" 配置={[(c['key'], c['d'], c['steps']) for c in CONFIGS]} | 门B不重跑 | 网格只 2 点。",
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


# ---------------- 切分器（只读自 nanoSeek） ----------------
import training.dialogue_stream as ds_mod  # noqa: E402
from training.dialogue_stream import (  # noqa: E402
    DialogueStream,
    EOS,
    read_corpus,
)

assert ds_mod.__file__.startswith(NS_ROOT), ds_mod.__file__
stream_helper = DialogueStream(enc, window=10 ** 9)


def parse_ans(text: str):
    """#### 终局答案解析（与 S09c/S11/S12 逐字相同）。"""
    if "####" not in text:
        return None
    a = text.rsplit("####", 1)[1].split("\n")[0]
    a = a.strip().replace(",", "").replace(" ", "").replace("$", "")
    return a or None


EXTRACT_STAT = Counter()

MARK_RE = re.compile(r"</?(?:eos|pad|resp|cont|bos|unk)>", re.IGNORECASE)


def strip_mark(s):
    if s is None:
        return None
    return MARK_RE.sub("", str(s))


_UNIT_RE = re.compile(
    r"(美元|美分|人民币|元|毛|个|人|名|天|日|小时|分钟|秒"
    r"|米|厘米|千米|公里|米|克|千克|公斤|吨|升|毫升|度|倍|次"
    r"|辆|本|张|件|条|只|道|题|percent|dollars?|cents?|hours?"
    r"|minutes?|seconds?|days?|meters?|kilometers?|km|kg|cm|%|％)$",
    re.IGNORECASE,
)


def extract_ans(text: str):
    if "####" in text:
        a = text.rsplit("####", 1)[1].split("\n")[0].strip()
        if a:
            EXTRACT_STAT["####"] += 1
            return a
    ms = re.findall(r"\\boxed\{([^{}]*)\}", text)
    if ms:
        EXTRACT_STAT["boxed兜底"] += 1
        return ms[-1].strip()
    EXTRACT_STAT["取不到"] += 1
    return None


def norm_ans(a):
    if a is None:
        return None
    s = str(a).strip()
    m = re.search(r"\\boxed\{([^{}]*)\}", s)
    if m:
        s = m.group(1)
    s = strip_mark(s)
    s = re.sub(r"[\s,，]", "", s)
    s = s.replace("$", "").replace("￥", "").replace("€", "").replace("£", "")
    s = s.replace("−", "-").replace("—", "-")
    for _ in range(3):
        t = _UNIT_RE.sub("", s)
        if t == s:
            break
        s = t
    return s or None


def _as_num(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def em_norm(gold, pred) -> int:
    if gold is None or pred is None:
        return 0
    g, p = norm_ans(gold), norm_ans(pred)
    if g is None or p is None:
        return 0
    ng, np_ = _as_num(g), _as_num(p)
    if ng is not None and np_ is not None:
        return int(abs(ng - np_) <= 1e-6 * max(1.0, abs(ng)))
    return int(g == p)


def em_strict(gold, text) -> int:
    if gold is None:
        return 0
    p = parse_ans(text)
    if p is None:
        return 0
    return int(strip_mark(p) == strip_mark(gold))


# ---------------- 数据（与 A 臂逐字相同：target = `#### <答案>`） ----------------
blocks = read_corpus(DATA)
raw_pairs = []
bad = 0
for b in blocks:
    parts = re.split(r"\n模型：", b, maxsplit=1)
    if len(parts) != 2 or not parts[0].startswith("用户："):
        bad += 1
        continue
    q, r = parts[0][len("用户："):], parts[1]
    if not q.strip() or not r.strip():
        bad += 1
        continue
    raw_pairs.append((q, r))

records = []
dropped = span_mismatch = encode_nonadditive = 0
trunc_p = trunc_t = both_cut = 0
no_ans_kept_full = 0
tgt_len_hist = Counter()

for q, r in raw_pairs:
    stream = DialogueStream(enc, window=10 ** 9)
    stream.append("用户", q)
    stream.enforce_window()
    prompt = stream.prompt()
    p_full = enc(prompt)
    if "####" in r:
        ans_raw = r.rsplit("####", 1)[1].split("\n")[0].strip()
        target_text = "#### " + ans_raw + EOS
    else:
        target_text = r + EOS
        no_ans_kept_full += 1
    t_full = enc(target_text)
    tgt_len_hist[len(t_full)] += 1
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
    assert len(p_ids) + len(t_ids) <= MAXLEN, (len(p_ids), len(t_ids))
    if len(p_ids) < 1:
        dropped += 1
        continue
    if cut_p:
        trunc_p += 1
    if cut_t:
        trunc_t += 1
    if cut_p and cut_t:
        both_cut += 1
    if not (cut_p or cut_t):
        spans = stream_helper.loss_token_spans(prompt + target_text)
        if not spans or spans[0][0] != len(p_ids):
            span_mismatch += 1
            if enc(prompt + target_text) != p_ids + enc(target_text):
                encode_nonadditive += 1
    records.append(
        dict(p=p_ids, t=t_ids, t_full=t_full, gold=parse_ans(target_text),
             text=target_text, cut_p=cut_p, cut_t=cut_t)
    )

n_total = len(records)
g42 = torch.Generator().manual_seed(42)
perm = torch.randperm(n_total, generator=g42).tolist()
n_test = int(round(n_total * 0.1))
test_idx = set(perm[:n_test])
train = [r for i, r in enumerate(records) if i not in test_idx]
test = [r for i, r in enumerate(records) if i in test_idx]
n_em = sum(1 for r in test if r["gold"] is not None)

n_parsed = len(raw_pairs)
n_cut_any = trunc_p + trunc_t - both_cut
print(f"[DATA] blocks={len(blocks)} parsed={n_parsed} bad={bad} records={n_total} "
      f"| split 9:1 train={len(train)} test={len(test)} (seed42) "
      f"| 任一截断={n_cut_any}/{n_parsed} ({100.0*n_cut_any/n_parsed:.1f}%) "
      f"| dropped={dropped} | 无####沿用原文={no_ans_kept_full}", flush=True)
print(f"[TRUNC] 截prompt={trunc_p}/{n_parsed} ({100.0*trunc_p/n_parsed:.1f}%) | "
      f"截target={trunc_t}/{n_parsed} ({100.0*trunc_t/n_parsed:.1f}%) | "
      f"两者都截={both_cut}/{n_parsed} | 丢弃={dropped}", flush=True)
_tl = sorted(tgt_len_hist.items())
print(f"[TGT] target=`#### <答案>`+EOS 长度: 中位={_tl[len(_tl)//2][0]}tok "
      f"min={_tl[0][0]} max={_tl[-1][0]}", flush=True)
print(f"[SPAN] span_mismatch={span_mismatch} nonadditive={encode_nonadditive} "
      f"| test有gold(EM分母)={n_em}", flush=True)
print(f"[TOK] WordLevel V={V} PAD={PAD_ID} EOS={EOS_ID} | import {ds_mod.__file__}", flush=True)

nA = sum(1 for r in test if not r["cut_t"])
nB = sum(1 for r in test if r["cut_t"])
n_free = sum(1 for r in test if not r["cut_t"] and not r["cut_p"])
print(f"[BUCKET] test={len(test)} | 桶A n={nA} | 桶B n={nB} | 桶A1(都未截) n={n_free} "
      f"⇒ 桶A1 n>=100 → {'满足,可判' if n_free >= 100 else '不满足'}", flush=True)


# ---------------- 模型（与 A 臂逐字相同，仅 D/FF 参数化） ----------------
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

    def cards(self):
        return {
            "输入卡": list(self.emb.parameters()) + list(self.in_enc.parameters()),
            "思维卡": [] if self.thought is None else list(self.thought.parameters()),
            "输出卡": list(self.head.parameters()),
        }


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


def train_run(seed: int, d: int, ff: int, steps: int, identity: bool,
              batch_size: int, gate_a: bool = False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards(d, ff, identity).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, gate_res = 0, 0, None
    t0 = time.time()
    while step < steps:
        idx = []
        for _ in range(batch_size):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        recs = [train[i] for i in idx]
        loss = masked_ce(model, recs)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        step += 1
        if gate_a and step == 20:
            gate_res = {k: sum(p.grad is not None for p in ps)
                        for k, ps in model.cards().items()}
            tot = {k: len(ps) for k, ps in model.cards().items()}
            print(f"[GATE A] step20 grad is not None 计数 {gate_res} / 参数总数 {tot} "
                  f"=> {'PASS 全>0' if all(v > 0 for v in gate_res.values()) else 'FAIL'}",
                  flush=True)
        opt.step()
        if step % 500 == 0 or step == steps:
            print(f"  [train] seed={seed} d={d} step={step}/{steps} "
                  f"loss={loss.item():.4f} elapsed={time.time()-t0:.0f}s", flush=True)
    return model, gate_res, time.time() - t0


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


# ---------------- 数字/算子 CE（分类口径逐字沿用 stages/10_ce_diag.py） ----------------
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
    """目标 span 上的 per-token CE 按 S10 四类拆开。

    返回 (各类 token CE 列表, 每条样本整体均值, 每条样本的数字token CE 列表)。
    第三项 = 样本级数字 CE ⇒ 用于算「数字每步正确率的 ±SE」（判据要 2SE）。
    """
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
    """★四项：整体配对 CE + 数字/算子 CE（token 加权 & 样本级 ±SE）+ 每步正确率。"""
    tok_ce, samp_all, dig_samp = ce_categories(model, recs)
    ntok = sum(len(v) for v in tok_ce)
    ce_m, ce_se = mean_se(samp_all)
    print(f"[CE] {tag} 整体配对CE(样本级,同09口径)={ce_m:.4f}±{ce_se:.4f} n={len(samp_all)} "
          f"| ntok={ntok} | ★每步正确率=exp(−CE)={math.exp(-ce_m)*100:.1f}%", flush=True)
    dig_ce = dig_n = None
    for k in range(4):
        xs = tok_ce[k]
        if not xs:
            print(f"[CE] {tag} {CAT_NAMES[k]}: n=0 占比=0.0% CE=无样本(该类不在目标 span 内)",
                  flush=True)
            continue
        km = sum(xs) / len(xs)
        print(f"[CE] {tag} {CAT_NAMES[k]}: n={len(xs)} 占比={100.0*len(xs)/ntok:.1f}% "
              f"CE={km:.4f} nats", flush=True)
        if k == 0:
            dig_ce, dig_n = km, len(xs)
    out = dict(ce_m=ce_m, ce_se=ce_se, dig_ce=dig_ce, dig_n=dig_n)
    if dig_ce is not None:
        print(f"[CE] {tag} ★数字/算子CE(token加权)={dig_ce:.4f} n={dig_n} "
              f"占比={100.0*dig_n/ntok:.1f}% | ★数字每步正确率=exp(−数字CE)="
              f"{math.exp(-dig_ce)*100:.1f}%", flush=True)
        per = [sum(x) / len(x) for x in dig_samp]
        dm, dse = mean_se(per)
        acc = math.exp(-dm)
        acc_se = acc * dse                      # delta 法：SE(exp(−CE)) ≈ exp(−CE)·SE(CE)
        print(f"[CE] {tag} ★数字CE(样本级,每条≥1数字token)={dm:.4f}±{dse:.4f} n={len(per)} "
              f"⇒ 数字每步正确率={acc*100:.1f}%±{acc_se*100:.2f}pp （判据 2SE 用这个）",
              flush=True)
        out.update(dig_samp_m=dm, dig_samp_se=dse, acc=acc, acc_se=acc_se)
    return out


# ---------------- ★R28：贪心生成（正确写法 + 批内一致性自检） ----------------
@torch.no_grad()
def greedy_gen(model: Cards, recs, batch: int = 1, mode: str = "correct") -> list[str]:
    """贪心解码。mode='correct' 查询列指向最后真实/已生成 token 列；mode='s11' 用旧索引
    （`k=lens+gen-1`，批内落在 PAD 缺口 ⇒ 非自回归，仅作自检对照）。
    batch=1 时 n0 == lens[j] ⇒ 两种写法逐字节相同（这就是 batch=1 自洽的原因）。
    """
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
        cap = [MAXLEN - L for L in lens]
        done = [False] * len(chunk)
        last_col = [L - 1 for L in lens]        # ★正确写法：最后真实 token 所在列
        for _ in range(max(cap)):
            if all(done):
                break
            logits = model.logits(ids)
            nxt = []
            for j in range(len(chunk)):
                if done[j]:
                    nxt.append(PAD_ID)
                    continue
                if mode == "correct":
                    k = last_col[j]
                else:
                    k = lens[j] + gen_len[j] - 1      # 旧写法（已知仪器 bug）
                t = int(logits[j, k].argmax().item())
                nxt.append(t)
                gen_len[j] += 1
                if t == EOS_ID or gen_len[j] >= cap[j]:
                    done[j] = True
                last_col[j] = n0 + gen_len[j] - 1      # 新 token 落在列 n0+gen-1
            ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long, device=device).unsqueeze(1)], 1)
        for j in range(len(chunk)):
            g_ids = ids[j, n0: n0 + gen_len[j]].tolist()   # ★切片从 n0 起
            g_ids = [x for x in g_ids if x != EOS_ID]
            texts[i + j] = dec(g_ids)
    model.train()
    return texts


def hits_from(texts: list[str], recs) -> list[dict]:
    hits = []
    for i, r in enumerate(recs):
        if r["gold"] is None:
            continue
        text = texts[i]
        pred = extract_ans(text)
        hits.append(dict(idx=i, gold=r["gold"], pred=pred, gen=text,
                         strict=em_strict(r["gold"], text),
                         norm=em_norm(r["gold"], pred),
                         bucket=("B" if r["cut_t"] else "A"),
                         free=(not r["cut_t"] and not r["cut_p"])))
    return hits


def r28_selfcheck(model: Cards, recs, k: int = 16, tag: str = "") -> dict:
    """★R28 批内一致性自检：同一 prompt，batch=1 vs batch=N 是否逐字一致。"""
    sub = recs[:k]
    t1 = greedy_gen(model, sub, batch=1, mode="correct")
    tN = greedy_gen(model, sub, batch=CHK_BATCH, mode="correct")
    sameN = sum(a == b for a, b in zip(t1, tN))
    print(f"[R28] {tag}批内一致性自检 K={len(sub)}: "
          f"batch=1 vs batch={CHK_BATCH}(正确写法) 逐字一致 {sameN}/{len(sub)}", flush=True)
    shown = 0
    for i, (a, b) in enumerate(zip(t1, tN)):
        if a == b:
            continue
        d = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print(f"[R28] {tag}差异[{i}] 首分歧字符位={d} len1={len(a)} lenN={len(b)} "
              f"| batch1={a[:60]!r} | batchN={b[:60]!r}", flush=True)
        shown += 1
        if shown >= 3:
            break
    verdict = "一致" if sameN == len(sub) else "★不一致 ⇒ batch=N 口径作废，EM 主口径只用 batch=1"
    print(f"[R28] {tag}结论：{verdict}", flush=True)
    return dict(sameN=sameN, k=len(sub))


def em_buckets(tag: str, hits: list[dict]):
    out = {}
    rows = (("桶A(目标未截)", [h for h in hits if h["bucket"] == "A"]),
            ("桶A1(两者都未截)", [h for h in hits if h["free"]]),
            ("全量", list(hits)))
    for name, sel in rows:
        key = name.split("(")[0]
        n = len(sel)
        s = sum(h["strict"] for h in sel) / n if n else float("nan")
        m = sum(h["norm"] for h in sel) / n if n else float("nan")
        se = math.sqrt(s * (1 - s) / n) if n else float("nan")
        out[key] = (n, s, se, m)
        if n == 0:
            print(f"[EM] {tag} {name}: n=0 无样本", flush=True)
            continue
        judge = ""
        if key == "桶A1":
            judge = (f" ⇒ 判据 n>=100 {'满足' if n >= 100 else '不满足'} | "
                     f"vs 地板1.89% → {'显著更高' if s - 1.96 * se > 0.0189 else '未显著高于地板/≈地板'}")
        print(f"[EM] {tag} {name}: n={n} 严格EM={s*100:.2f}%±{se*100:.2f}  "
              f"归一化EM={m*100:.2f}%{judge}", flush=True)
    return out


def sample20(tag: str, hits: list[dict], recs, k: int = 20) -> dict:
    """抽 20 条生成：打印原文 + 统计「抽取答案 ∈ 高频整数集合」的比例。"""
    pool = sorted(hits[: k], key=lambda h: h["idx"])
    n_freq = n_pred = n_ok = 0
    vals = []
    print(f"[GEN] {tag} 抽样 {len(pool)} 条（贪心 batch={EM_BATCH}，test 前 k 条按 idx 排序）",
          flush=True)
    for i, h in enumerate(pool):
        ptxt = dec(recs[h["idx"]]["p"]).replace("\n", "⏎")
        g = h["gen"].replace("\n", "⏎")
        pred = norm_ans(h["pred"])
        vals.append(pred)
        if pred is not None:
            n_pred += 1
            if pred in FREQ_SET:
                n_freq += 1
        n_ok += h["norm"]
        print(f"[GEN] [{i}] {ptxt[-55:]} → {g[:80]!r} | gold={strip_mark(h['gold'])} "
              f"| 抽取={strip_mark(h['pred'])} | {'对' if h['norm'] else '错'}", flush=True)
    all_pred = [norm_ans(h["pred"]) for h in hits]
    all_freq = sum(1 for p in all_pred if p in FREQ_SET)
    print(f"[FREQ] {tag} ★抽20条: 输出∈{sorted(FREQ_SET)} 的比例 = {n_freq}/{len(pool)} "
          f"({100.0*n_freq/len(pool):.0f}%) | 有可抽取答案 {n_pred}/{len(pool)} | "
          f"这20条 EM={n_ok}/{len(pool)} | 抽取值分布={dict(Counter(vals).most_common(8))}",
          flush=True)
    print(f"[FREQ] {tag} 参考·全 test(n={len(hits)}): 输出∈{sorted(FREQ_SET)} = {all_freq}/{len(hits)} "
          f"({100.0*all_freq/len(hits):.1f}%) | 抽取值 Top6={dict(Counter(all_pred).most_common(6))}",
          flush=True)
    return dict(n20=len(pool), freq20=n_freq, pred20=n_pred, em20=n_ok,
                all_freq=all_freq, all_n=len(hits), dist=dict(Counter(all_pred).most_common(8)))


# ---------------- 地板（门 C）：三条平凡基线 ----------------
train_ans = [r["gold"] for r in train if r["gold"] is not None]
common_ans = Counter(train_ans).most_common(1)[0][0]
tok_counter = Counter()
for r in train:
    tok_counter.update(r["t_full"])
freq_tok = tok_counter.most_common(1)[0][0]


def floor_b_ids(kind: str, r) -> list[int]:
    L = len(r["t"])
    if kind == "copy_tail":
        p = r["p"]
        return (p[-L:] if len(p) >= L else [p[0]] * (L - len(p)) + p)
    if kind == "common_ans":
        return enc("#### " + common_ans)
    return [freq_tok] * L


def floor_ce(kind: str, r) -> float:
    b = floor_b_ids(kind, r)
    ce = 0.0
    for pos, gold in enumerate(r["t"]):
        if pos >= len(b):
            ce += math.log(V)
        elif b[pos] == gold:
            ce += -math.log(1 - EPS)
        else:
            ce += -math.log(EPS / (V - 1))
    return ce / len(r["t"])


def floor_text(kind: str, r) -> str:
    if kind == "common_ans":
        return "#### " + common_ans
    if kind == "freq_tok":
        return dec([freq_tok])
    return dec(floor_b_ids(kind, r))


def floor_em(kind: str, recs) -> list[int]:
    hits = []
    for r in recs:
        if r["gold"] is None:
            continue
        hits.append(em_strict(r["gold"], floor_text(kind, r)))
    return hits


FLOORS = [("copy_tail", "①复制prompt尾部"), ("common_ans", "②最常见####答案"),
          ("freq_tok", "③最常出现token")]

floor_res = {}
for kind, name in FLOORS:
    ces = [floor_ce(kind, r) for r in test]
    ems = floor_em(kind, test)
    ce_m, ce_se = mean_se(ces)
    em_m, em_se = mean_se(ems)
    floor_res[kind] = dict(ce_m=ce_m, ce_se=ce_se, em_m=em_m, em_se=em_se)
    print(f"[GATE C] {name}: 配对CE={ce_m:.4f}±{ce_se:.4f} nats  "
          f"EM={em_m*100:.2f}%±{em_se*100:.2f} (n={len(ems)})", flush=True)

# 结构性自检（随机初始化：索引写法的性质，不依赖权重）——d=128 与 A 臂同维
_probe = Cards(128, 512).to(device)
torch.manual_seed(0)
_r28_init = r28_selfcheck(_probe, test, k=CHK_BATCH, tag="[随机init d128] ")
del _probe
torch.cuda.empty_cache()


# ---------------- ★计时/显存探针（与训练同构：batch=32、AdamW、masked_ce） ----------------
def timing_probe(steps: int = 60, burn: int = 20) -> None:
    print("[PROBE] ---- 双配置计时/显存探针（与训练同构，只测 s/step 与 OOM） ----", flush=True)
    for cfg in CONFIGS:
        d, ff = cfg["d"], cfg["ff"]
        torch.manual_seed(0)
        torch.cuda.reset_peak_memory_stats()      # 逐配置重置，否则峰显存是整进程累计值
        try:
            m = Cards(d, ff).to(device)
            opt = AdamW(m.parameters(), lr=LR)
            order = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))
            pos, t_prev, dt = 0, None, []
            for k in range(steps):
                idx = []
                for _ in range(BATCH):
                    if pos >= len(order):
                        order = torch.randperm(len(train))
                        pos = 0
                    idx.append(int(order[pos]))
                    pos += 1
                loss = masked_ce(m, [train[i] for i in idx])
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
                now = time.time()
                if t_prev is not None and k >= burn:
                    dt.append(now - t_prev)
                t_prev = now
            rate = sum(dt) / len(dt)
            alloc = torch.cuda.max_memory_allocated() / 2 ** 30
            print(f"[PROBE] {cfg['key']}: d={d} ff={ff} params={n_params(d, ff)/1e6:.2f}M "
                  f"s/step={rate:.3f}（steps {burn+1}~{steps} 均值, n={len(dt)}） "
                  f"峰显存={alloc:.2f}GiB | 外推 train({cfg['steps']}步)="
                  f"{rate*cfg['steps']/60:.1f}min", flush=True)
            del m, opt, loss
        except torch.cuda.OutOfMemoryError:
            print(f"[PROBE] ★OOM d={d} batch={BATCH} ⇒ 该配置 batch=32 装不下", flush=True)
        torch.cuda.empty_cache()
    print("[PROBE DONE] exit=0", flush=True)
    sys.exit(0)


if PROBE:
    timing_probe()

# ---------------- 显存探针（与训练同构，逐配置） ----------------
for _cfg in CONFIGS:
    try:
        _pm = Cards(_cfg["d"], _cfg["ff"]).to(device)
        _loss = masked_ce(_pm, train[:BATCH])
        _loss.backward()
        del _loss, _pm
        torch.cuda.empty_cache()
        print(f"[META] 显存探针 OK: {_cfg['key']} d={_cfg['d']} batch={BATCH}", flush=True)
    except torch.cuda.OutOfMemoryError:
        print(f"[META] ★OOM: {_cfg['key']} d={_cfg['d']} batch={BATCH} —— "
              f"不静默改 batch（会破坏「只改两个变量」），直接退出让上层判定。", flush=True)
        sys.exit(5)


# ---------------- ② 的步数预案：实测 s/step → 外推 10000 步墙钟 ----------------
def decide_steps(cfg: dict) -> int:
    if cfg["d"] != 128 or cfg["steps"] != 10000:
        return cfg["steps"]
    rate_probe_steps, burn = 300, 100
    print(f"[RATE] ②步数预案：先实测 d=128/batch={BATCH} 的 s/step"
          f"（{rate_probe_steps} 步同构探针，丢弃权重，不进任何结论）", flush=True)
    torch.manual_seed(0)
    m = Cards(128, 512).to(device)
    opt = AdamW(m.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))
    pos, t_prev, dt = 0, None, []
    for k in range(rate_probe_steps):
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(m, [train[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        now = time.time()
        if t_prev is not None and k >= burn:
            dt.append(now - t_prev)
        t_prev = now
    rate = sum(dt) / len(dt)
    del m, opt, loss
    torch.cuda.empty_cache()
    EVAL_SEC = 120.0          # 每 seed 评估+生成+R28 的经验上界（S12 全跑总62.2min−训练59.6min≈2.6min/5项）
    proj10000 = rate * 10000 + EVAL_SEC
    proj6000 = rate * 6000 + EVAL_SEC
    s12_rate = 1222.0 / 3000
    print(f"[RATE] 实测 s/step={rate:.4f}（n={len(dt)}，burn={burn}；S12 同配置实测 "
          f"{s12_rate:.4f} s/step 作交叉核对）", flush=True)
    print(f"[RATE] 外推单 seed：10000 步 = {proj10000/60:.1f}min | 6000 步 = {proj6000/60:.1f}min "
          f"| 阈值 60min ⇒ 10000 {'★超阈值 ⇒ 落到 6000 步' if proj10000 > 3600 else '未超 ⇒ 保持 10000 步'}",
          flush=True)
    if proj10000 > 3600:
        print("[RATE] ★按写死预案：② 步数 10000 → **6000**（唯一改动，显式写明；"
              "d/ff/batch/lr/MAXLEN/切分/口径全部不变）。", flush=True)
        return 6000
    return 10000


# ---------------- 主循环：两配置 × 2 seed ----------------
RESULTS: dict[str, dict[int, dict]] = {}

for cfg in CONFIGS:
    steps = decide_steps(cfg)
    cfg["steps_run"] = steps
    key, d, ff = cfg["key"], cfg["d"], cfg["ff"]
    RESULTS[key] = {}
    print(f"\n[S13] ===== 配置 {key}: d={d} ff={ff} steps={steps} "
          f"（{cfg['var']}） params={n_params(d, ff)/1e6:.2f}M =====", flush=True)
    for si, seed in enumerate(SEEDS):
        t_run = time.time()
        tag = f"{key} seed={seed} d={d} steps={steps}"
        model, gate, wall = train_run(seed, d, ff, steps, False, BATCH, gate_a=(si == 0))
        if si == 0 and (gate is None or not all(v > 0 for v in gate.values())):
            print("[GATE A] 不过 ⇒ 梯度设计无效，停。", flush=True)
            sys.exit(4)

        ce_list = eval_ce(model, test)
        cm, cs = mean_se(ce_list)
        txt = greedy_gen(model, test, batch=EM_BATCH, mode="correct")
        hits = hits_from(txt, test)
        print(f"[MAIN] {tag}: 整体配对CE={cm:.4f}±{cs:.4f} n={len(hits)}", flush=True)
        rep = report_ce(tag, model, test)
        em = em_buckets(f"{tag} (batch={EM_BATCH}正确写法)", hits)
        r28 = r28_selfcheck(model, test, k=CHK_BATCH, tag=f"[{tag}] ")
        freq = sample20(tag, hits, test, k=20)

        ck = CKPT_TMPL.format(key=key, seed=seed)
        torch.save(model.state_dict(), ck)
        print(f"[CKPT] saved {ck}", flush=True)
        del model
        torch.cuda.empty_cache()

        RESULTS[key][seed] = dict(
            d=d, ff=ff, steps=steps, seed=seed,
            dig_tok=rep.get("dig_ce"), acc=rep.get("acc"), acc_se=rep.get("acc_se"),
            dsamp=rep.get("dig_samp_m"), dsamp_se=rep.get("dig_samp_se"),
            ce=cm, ce_se=cs, em=em.get("全量", (0, 0, 0, 0))[1], em_se=em.get("全量", (0, 0, 0, 0))[2],
            em_n=em.get("全量", (0, 0, 0, 0))[0],
            em_a1=em.get("桶A1", (0, 0, 0, 0))[1], em_a1_se=em.get("桶A1", (0, 0, 0, 0))[2],
            em_a1_n=em.get("桶A1", (0, 0, 0, 0))[0],
            r28=r28["sameN"], r28_k=r28["k"], freq=freq,
            wall_train=wall, wall_total=time.time() - t_run, ckpt=ck,
        )
        print(f"[CLOCK] {tag} 训练墙钟={wall/60:.1f}min 本 seed 合计={(time.time()-t_run)/60:.1f}min",
              flush=True)

print(f"\n[EXTRACT] 抽取方式计数 {dict(EXTRACT_STAT)}", flush=True)

# ---------------- ★Δ±SE 对照表（逐项 vs 基线） ----------------
print("\n[S13] ===== ★与基线（d=128/3000步, logs/12_answer.log）的 Δ±SE 对照 =====", flush=True)
print(f"[S13] 基线逐 seed: seed1234 数字CE(样本级)=2.1947±0.0198 ⇒ "
      f"每步正确率={BASE[1234]['acc']*100:.1f}%±{BASE[1234]['acc_se']*100:.2f}pp "
      f"| 整体CE=0.6924±0.0072 | EM全量=2.17%(n=739) 桶A1=2.07%(n=726)", flush=True)
print(f"[S13] 基线逐 seed: seed5678 数字CE(样本级)=2.1975±0.0228 ⇒ "
      f"每步正确率={BASE[5678]['acc']*100:.1f}%±{BASE[5678]['acc_se']*100:.2f}pp "
      f"| 整体CE=0.6844±0.0083 | EM全量=2.17%(n=739) 桶A1=2.07%(n=726)", flush=True)

print("[S13] 配置 | seed | Δ数字CE(样本级)±SE | Δ数字每步正确率(pp)±SE | Δ整体CE±SE | "
      "ΔEM全量(pp)±SE | ΔEM桶A1(pp)±SE", flush=True)
SIG: dict[str, list[bool]] = {}
for key, per in RESULTS.items():
    SIG[key] = []
    for seed in SEEDS:
        r, b = per[seed], BASE[seed]
        se_ds = math.hypot(r["dsamp_se"], b["dsamp_se"])
        d_ds = r["dsamp"] - b["dsamp"]
        se_acc = math.hypot(r["acc_se"], b["acc_se"])
        d_acc = (r["acc"] - b["acc"]) * 100
        se_ce = math.hypot(r["ce_se"], b["ce_se"])
        d_ce = r["ce"] - b["ce"]
        se_em = math.hypot(r["em_se"], math.sqrt(b["em"] * (1 - b["em"]) / b["em_n"]))
        d_em = (r["em"] - b["em"]) * 100
        se_a1 = math.hypot(r["em_a1_se"], math.sqrt(b["em_a1"] * (1 - b["em_a1"]) / b["em_a1_n"]))
        d_a1 = (r["em_a1"] - b["em_a1"]) * 100
        thr = b["acc"] + 2 * b["acc_se"]
        hit = (r["acc"] >= thr) and (d_acc > 2 * se_acc * 100)
        SIG[key].append(hit)
        print(f"[S13] {key} | {seed} | {d_ds:+.4f}±{se_ds:.4f} | "
              f"{d_acc:+.2f}±{se_acc*100:.2f} (达标线 {thr*100:.1f}% "
              f"{'≥达标' if r['acc'] >= thr else '<达标'}; 2SE判据 {'显著' if d_acc > 2*se_acc*100 else '不显著'}) | "
              f"{d_ce:+.4f}±{se_ce:.4f} | {d_em:+.2f}±{se_em*100:.2f} | "
              f"{d_a1:+.2f}±{se_a1*100:.2f}", flush=True)
        print(f"[S13] {key} | {seed} 绝对值: 数字CE(token加权)={r['dig_tok']:.4f} "
              f"(基线 {b['dig_tok']:.4f}, Δ={r['dig_tok']-b['dig_tok']:+.4f} 无基线token级SE) | "
              f"数字每步正确率(token加权)={math.exp(-r['dig_tok'])*100:.1f}% "
              f"vs 基线 {b['acc_tok']*100:.1f}% | 整体CE={r['ce']:.4f}±{r['ce_se']:.4f} "
              f"| EM全量={r['em']*100:.2f}% (n={r['em_n']}) 桶A1={r['em_a1']*100:.2f}% (n={r['em_a1_n']})",
              flush=True)

# ---------------- ★判决 ----------------
print("\n[VERDICT-S13] ===== 判决（判据写死） =====", flush=True)
cap_ok = all(SIG["d256_s3000"])
step_ok = all(SIG["d128_s10000"])
for key in ("d256_s3000", "d128_s10000"):
    per = RESULTS[key]
    a1 = [f"seed{s}: {per[s]['acc']*100:.1f}%±{per[s]['acc_se']*100:.2f}pp "
          f"(Δ={((per[s]['acc']-BASE[s]['acc'])*100):+.1f}pp, "
          f"{'达标' if SIG[key][SEEDS.index(s)] else '不达标'})" for s in SEEDS]
    print(f"[VERDICT-S13] {key} 数字每步正确率 {'  '.join(a1)} ⇒ "
          f"{'2 seed 均 ≥11%+2SE ⇒ 显著' if all(SIG[key]) else '未同时达标 ⇒ 不显著'}", flush=True)

if cap_ok and step_ok:
    verdict = "容量与训练量都是瓶颈"
elif cap_ok:
    verdict = "★容量是瓶颈"
elif step_ok:
    verdict = "★训练量是瓶颈"
else:
    verdict = "★两者都不显著 ⇒ 架构不会做算术（给三条新方向，见下）"
print(f"[VERDICT-S13] 判决 = {verdict}", flush=True)
if not cap_ok and not step_ok:
    print("[VERDICT-S13] 三条新方向（本单元不执行，仅提出）："
          "(a) 换目标形态/损失权重——把数字 token 加权或先教『数字→算子→进位』的独立子任务，"
          "使 CE 不被 `#### ` 模板 token 稀释；"
          "(b) 换结构归纳偏置——为等长数字串加 copying/进位状态（逐位卷积或显式计算器槽），"
          "而非纯因果 Transformer 复读高频整数；"
          "(c) 换数据/课程——用可程序生成、答案分布均匀的加减乘除合成集先过『会算』关，"
          "再回 GSM8K，同时把输出收敛到纯数字（去掉 `####` 前缀），排除抽取口径噪声。",
          flush=True)

# ---------------- 高频整数汇总 ----------------
print("\n[FREQ-S13] ===== 高频整数倾向（抽20条/配置/seed） =====", flush=True)
tot20 = totf = 0
for key, per in RESULTS.items():
    for seed in SEEDS:
        f = per[seed]["freq"]
        tot20 += f["n20"]
        totf += f["freq20"]
        print(f"[FREQ-S13] {key} seed={seed}: 抽20条中输出∈{sorted(FREQ_SET)} = "
              f"{f['freq20']}/{f['n20']} ({100.0*f['freq20']/f['n20']:.0f}%) | "
              f"有可抽取答案={f['pred20']}/{f['n20']} | 全test {f['all_freq']}/{f['all_n']} "
              f"({100.0*f['all_freq']/f['all_n']:.1f}%) | Top={dict(list(f['dist'].items())[:5])}",
              flush=True)
print(f"[FREQ-S13] 四个模型合计: 抽20条×4={tot20} 条中 ∈{sorted(FREQ_SET)} = {totf}/{tot20} "
      f"({100.0*totf/tot20:.0f}%) | 对照 A 臂基线（logs/12_answer.log seed1234 的 20 条）"
      f"= 18/20 (90%)", flush=True)

# ---------------- 收尾元数据 ----------------
print(f"\n[META] device={device} {torch.cuda.get_device_name(0)} | "
      f"MAXLEN={MAXLEN} batch={BATCH} lr={LR} NHEAD={NHEAD} 切分=seed42(9:1) | "
      f"生成主口径=batch{EM_BATCH} R28自检(batch1 vs batch{CHK_BATCH})=" +
      " / ".join(f"{key}seed{s}={RESULTS[key][s]['r28']}/{RESULTS[key][s]['r28_k']}"
                 for key in RESULTS for s in SEEDS) +
      f" | 随机init={_r28_init['sameN']}/{_r28_init['k']}", flush=True)
for key in RESULTS:
    for s in SEEDS:
        r = RESULTS[key][s]
        print(f"[META] {key} seed={s}: d={r['d']} ff={r['ff']} steps={r['steps']} batch={BATCH} "
              f"训练墙钟={r['wall_train']/60:.1f}min seed合计={r['wall_total']/60:.1f}min "
              f"ckpt={r['ckpt']}", flush=True)
print(f"[META] 总墙钟={(time.time()-t_start)/60:.1f}min", flush=True)
print("[DONE] exit=0", flush=True)
