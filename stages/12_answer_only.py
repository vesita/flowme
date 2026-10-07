#!/usr/bin/env python3
"""S12 · 归因判决两臂之 A 臂：**只改输出形态**（目标裁成 `#### <答案>`，去掉 CoT）—— ★GPU 训练。

唯一变量 = 输出形态：
  A（本臂）目标 = `#### <答案>` + <eos>（无 CoT）；prompt（题干）与 S11 逐字相同；
  B（对照） = S11（完整 CoT + `#### <答案>`），**直接引用 logs/11_textgen.log 的数字，不重跑**。

其余全部与 stages/11_window512.py 同源、不动：D/FF/NHEAD/MAXLEN/MIN_T/TAIL_KEEP/
STEPS/LR/SEEDS/BATCH、9:1 切分（seed42）、截断优先级（prompt 优先）、四类 token 分类口径、
整体配对 CE / 每步正确率 / 分桶 EM / 门 A·B·C / 贪心解码实现。

本臂要报的七件事：
  1) 只改输出形态（prompt 不变、MAXLEN=512、其余全同）；
  2) A 四项（2 seed）：整体配对 CE · 数字/算子 CE · 数字每步正确率 · EM（含桶A1 且 n>=100）；
  3) A 截断三项 + 被截情况（答案很短 ⇒ 截 target 应大幅下降，没降要说明）；
  4) 门 B 复跑（恒等中介）：配对 ΔCE；
  5) ★R28 生成自检：同一 prompt 在 batch=1 与 batch=N 下生成是否一致（不一致 ⇒ 报差异并作废该口径）；
  6) **正确写法**的 EM + 自由生成 20 条原文；
  7) 墙钟 / ckpt 路径 / 对本文件与 stages/12*.py 跑字面量 `"dt"+"seek"` 的 grep（结果必须为空）。

生成口径纪律（R28 / 已知仪器 bug）：
  * 旧写法 `k = lens[j] + gen_len[j] - 1` 在批内（先把每行垫到 n0=max(prompt)）第 2 步起
    落在 PAD 缺口 ⇒ 非自回归 ⇒ **作废**；
  * 正确写法：查询列始终指向「最后一个真实/已生成 token 所在列」
    （gen==0 → lens[j]-1；gen>=1 → n0+gen-1），生成文本从列 n0 起切；
  * **batch=1 时 n0 == lens[j]，两种写法逐字节相同** ⇒ batch=1 是天然自洽口径；
  * 本单元主口径 = batch=1（正确写法）；batch=N 只作并列参考，且先过 R28 自检。

ckpt：logs/12_ckpt_seed{1234,5678}.pt；只允许写本文件与 logs/、/tmp；nanoSeek 只读。
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

# ---------------- 配置（★只改输出形态，其余与 S11 逐字相同） ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512          # 与 S11 相同
MIN_T = 48            # 与 S11 相同：prompt 上限 = MAXLEN−48 = 464
TAIL_KEEP = 24        # 与 S11 相同
STEPS = 3000
LR = 1e-3
SEEDS = (1234, 5678)
EPS = 1e-3
BATCH = 32
GEN_BATCH = 32
EM_BATCH = 1          # ★主口径：batch=1（R28 自检通过的唯一自洽口径）
CHK_BATCH = 16        # 自检/并列参考口径

DATA = NS_ROOT + "/data/chinese/clean_v3/gsm8k_cot_dialogue.txt"
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
CKPT_TMPL = "/home/vesita/coding/my/flowme/logs/12_ckpt_seed{seed}.pt"

PROBE = os.environ.get("S12_PROBE") == "1"
t_start = time.time()

print(f"[S12] ★A臂 只改一个变量=输出形态：target = `#### <答案>`+EOS（去 CoT）；"
      f"prompt=题干 与 S11 逐字相同；MAXLEN={MAXLEN} 截断优先级=prompt优先(MIN_T={MIN_T})。"
      f" 其余不动：steps={STEPS} batch={BATCH} lr={LR} d={D} ff={FF} seeds={SEEDS} 切分=seed42(9:1)。"
      f" B臂=引用 S11 不重跑；生成主口径=batch={EM_BATCH}", flush=True)

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
    """#### 终局答案解析（与 S09c/S11 逐字相同）。"""
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


# ---------------- 数据（★唯一改动：target = `#### <答案>`） ----------------
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
no_ans_kept_full = 0          # ★无 #### 的样本（保持条数 ⇒ 保持同一 test 切分）
tgt_len_hist = Counter()

for q, r in raw_pairs:
    stream = DialogueStream(enc, window=10 ** 9)
    stream.append("用户", q)
    stream.enforce_window()
    prompt = stream.prompt()
    p_full = enc(prompt)
    # ★S12 唯一改动：目标 = `#### <答案>` + <eos>（CoT 去掉）；prompt 不动。
    if "####" in r:
        ans_raw = r.rsplit("####", 1)[1].split("\n")[0].strip()
        target_text = "#### " + ans_raw + EOS
    else:
        target_text = r + EOS        # 4/7394 条无 ####：沿用原文，保证条数与切分不变
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
n_free_all = n_parsed - n_cut_any
print(f"[TRUNC] ★A臂 target=`####<答案>` prompt优先(MAXLEN={MAXLEN} MIN_T={MIN_T}) "
      f"截prompt={trunc_p}/{n_parsed} ({100.0*trunc_p/n_parsed:.1f}%) | "
      f"截target={trunc_t}/{n_parsed} ({100.0*trunc_t/n_parsed:.1f}%) | "
      f"两者都截={both_cut}/{n_parsed} ({100.0*both_cut/n_parsed:.1f}%) | "
      f"两者都未截={n_free_all}/{n_parsed} ({100.0*n_free_all/n_parsed:.1f}%) | "
      f"丢弃={dropped}", flush=True)
_tl = sorted(tgt_len_hist.items())
print(f"[TGT] ★target形态=`#### <答案>`+EOS 目标长度: 中位={_tl[len(_tl)//2][0]}tok "
      f"min={_tl[0][0]} max={_tl[-1][0]} 分布(长度:条数,前8)={dict(_tl[:8])} "
      f"| S11(B臂) 目标长度中位≈156tok", flush=True)
print(f"[SPAN] span_mismatch={span_mismatch} nonadditive={encode_nonadditive} "
      f"| test有gold(EM分母)={n_em}", flush=True)
print(f"[TOK] WordLevel V={V} PAD={PAD_ID} EOS={EOS_ID} | import {ds_mod.__file__}", flush=True)

nA = sum(1 for r in test if not r["cut_t"])
nB = sum(1 for r in test if r["cut_t"])
nA_pc = sum(1 for r in test if not r["cut_t"] and r["cut_p"])
n_free = sum(1 for r in test if not r["cut_t"] and not r["cut_p"])
print(f"[BUCKET] test={len(test)} | 桶A(目标未截) n={nA} | 桶B(目标被截) n={nB} "
      f"| 桶A中prompt仍被截 n={nA_pc} | 两者都未截(=桶A1) n={n_free} "
      f"⇒ 桶A n>=30 → {'满足' if nA >= 30 else '不满足'} | "
      f"桶A1 n>=100 → {'满足,可判' if n_free >= 100 else '不满足,本单元无法判'}",
      flush=True)


# ---------------- 模型：三张卡（与 S11 逐字相同） ----------------
def sin_pe(max_len: int, d: int) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def enc_layer() -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        D, NHEAD, FF, dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )


class Cards(nn.Module):
    def __init__(self, identity: bool = False):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.in_enc = enc_layer()
        self.thought = None if identity else enc_layer()
        self.head = nn.Linear(D, V)
        self.register_buffer("pe", sin_pe(MAXLEN, D), persistent=False)

    def logits(self, ids: torch.Tensor) -> torch.Tensor:
        n = ids.size(1)
        m = torch.full((n, n), float("-inf"), device=ids.device)
        m = torch.triu(m, diagonal=1)
        m = m.unsqueeze(0).expand(ids.size(0), -1, -1).clone()
        m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
        m = m.repeat_interleave(NHEAD, dim=0)
        pe = self.pe if n <= self.pe.size(0) else sin_pe(n, D).to(self.pe.device)
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


def train_run(seed: int, identity: bool, batch_size: int, gate_a: bool):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards(identity).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, gate_res = 0, 0, None
    t0 = time.time()
    while step < STEPS:
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
            gate_res = {
                k: sum(p.grad is not None for p in ps)
                for k, ps in model.cards().items()
            }
            tot = {k: len(ps) for k, ps in model.cards().items()}
            print(f"[GATE A] step20 grad is not None 计数 {gate_res} / 参数总数 {tot} "
                  f"=> {'PASS 全>0' if all(v > 0 for v in gate_res.values()) else 'FAIL'}",
                  flush=True)
        opt.step()
        if step % 500 == 0 or step == STEPS:
            print(f"  [train] seed={seed} identity={identity} step={step}/{STEPS} "
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
    第三项 = ★S12 新增：用于算「数字 CE 的样本级 ±SE」（判据要 2SE）。
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


def report_ce(tag: str, model: Cards, recs) -> dict:
    """★A 四项之三：整体配对 CE + 数字/算子 CE + 两条每步正确率（附样本级 ±SE）。"""
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
                    k = lens[j] + gen_len[j] - 1      # S11/10 旧写法（已知 bug）
                t = int(logits[j, k].argmax().item())
                nxt.append(t)
                gen_len[j] += 1
                if t == EOS_ID or gen_len[j] >= cap[j]:
                    done[j] = True
                last_col[j] = n0 + gen_len[j] - 1      # 新 token 落在列 n0+gen-1
            ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long, device=device).unsqueeze(1)], 1)
        for j in range(len(chunk)):
            g_ids = ids[j, n0: n0 + gen_len[j]].tolist()   # ★切片从 n0 起（S10③ 已修口径）
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


def r28_selfcheck(model: Cards, recs, k: int = 16) -> dict:
    """★R28 批内一致性自检：同一 prompt，batch=1 vs batch=N 是否逐字一致。"""
    sub = recs[:k]
    t1 = greedy_gen(model, sub, batch=1, mode="correct")
    tN = greedy_gen(model, sub, batch=CHK_BATCH, mode="correct")
    tS = greedy_gen(model, sub, batch=CHK_BATCH, mode="s11")
    sameN = sum(a == b for a, b in zip(t1, tN))
    sameS = sum(a == b for a, b in zip(t1, tS))
    print(f"[R28] 批内一致性自检 K={len(sub)}: "
          f"batch=1 vs batch={CHK_BATCH}(正确写法) 逐字一致 {sameN}/{len(sub)} | "
          f"batch=1 vs batch={CHK_BATCH}(S11旧索引) 逐字一致 {sameS}/{len(sub)}", flush=True)
    shown = 0
    for i, (a, b) in enumerate(zip(t1, tN)):
        if a == b:
            continue
        d = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print(f"[R28] 差异[{i}] 首分歧字符位={d} len(batch1)={len(a)} len(batchN)={len(b)} "
              f"| batch1={a[:60]!r} | batchN={b[:60]!r}", flush=True)
        shown += 1
        if shown >= 4:
            break
    verdict = "一致" if sameN == len(sub) else "★不一致 ⇒ batch=N 口径作废，EM 主口径只用 batch=1"
    print(f"[R28] 结论：{verdict}", flush=True)
    return dict(sameN=sameN, sameS=sameS, k=len(sub))


def em_buckets(tag: str, hits: list[dict]):
    out = {}
    rows = (("桶A(目标未截)", [h for h in hits if h["bucket"] == "A"]),
            ("桶A1(两者都未截)", [h for h in hits if h["free"]]),
            ("桶B(目标被截)", [h for h in hits if h["bucket"] == "B"]),
            ("全量", list(hits)))
    for name, sel in rows:
        key = name.split("(")[0]
        n = len(sel)
        s = sum(h["strict"] for h in sel) / n if n else float("nan")
        m = sum(h["norm"] for h in sel) / n if n else float("nan")
        out[key] = (n, s, m)
        if n == 0:
            print(f"[EM] {tag} {name}: n=0 无样本", flush=True)
            continue
        judge = ""
        if key == "桶A1":
            judge = (f" ⇒ 判据 n>=100 {'满足' if n >= 100 else '不满足(本单元无法判)'} | "
                     f"vs 地板1.89% → {'显著更高' if s > 0.05 else '未显著高于地板/≈地板'}")
        print(f"[EM] {tag} {name}: n={n} 严格EM={s*100:.2f}%  归一化EM={m*100:.2f}%{judge}",
              flush=True)
    return out


def free_gen_report(tag: str, hits: list[dict], recs, k: int = 20):
    pool = [h for h in hits if h["bucket"] == "A"] + [h for h in hits if h["bucket"] == "B"]
    pool = sorted(pool[:k], key=lambda h: h["idx"])
    print(f"[GEN] {tag} 自由生成 {len(pool)} 条（贪心 batch={EM_BATCH}，优先桶A）", flush=True)
    for i, h in enumerate(pool):
        ptxt = dec(recs[h["idx"]]["p"]).replace("\n", "⏎")
        g = h["gen"].replace("\n", "⏎")
        head = g[:120]
        ok = "对" if h["norm"] else "错"
        ok_s = "" if h["norm"] == h["strict"] else f"(严格={'对' if h['strict'] else '错'})"
        print(f"[GEN] [{i}] {ptxt[-60:]} → {head} "
              f"| gold={strip_mark(h['gold'])} | 抽取={strip_mark(h['pred'])} | {ok}{ok_s}",
              flush=True)


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n) if n else float("nan")


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

# ---------------- 地板 + ★S12 自检/探针（训练前先做） ----------------
floor_res = {}
for kind, name in FLOORS:
    ces = [floor_ce(kind, r) for r in test]
    ems = floor_em(kind, test)
    ce_m, ce_se = mean_se(ces)
    em_m, em_se = mean_se(ems)
    floor_res[kind] = dict(ce=ces, em=ems, ce_m=ce_m, ce_se=ce_se, em_m=em_m, em_se=em_se)
    print(f"[GATE C] {name}: 配对CE={ce_m:.4f}±{ce_se:.4f} nats  "
          f"EM={em_m*100:.2f}%±{em_se*100:.2f} (n={len(ems)})", flush=True)

# 结构性自检（随机初始化模型也必须过：这是索引写法的性质，不依赖权重）
probe_model = Cards().to(device)
torch.manual_seed(0)
_r28_init = r28_selfcheck(probe_model, test, k=CHK_BATCH)

if PROBE:
    t0 = time.time()
    _sub = test[:8]
    _txt = greedy_gen(probe_model, _sub, batch=1, mode="correct")
    dt = time.time() - t0
    caps = sum(MAXLEN - len(r["p"]) for r in test)
    print(f"[PROBE] batch=1 贪心 {len(_sub)} 条耗时={dt:.1f}s "
          f"⇒ 推算全 test({len(test)}) batch=1 最坏 {dt/len(_sub)*len(test)/60:.1f}min "
          f"(cap 总和={caps} tok, 未命中的最长路径)", flush=True)
    for a in _txt[:3]:
        print(f"[PROBE] gen={a[:80]!r}", flush=True)
    print(f"[PROBE] 训练前数据/截断/自检均就绪，device={device}", flush=True)
    print("[PROBE DONE] exit=0", flush=True)
    del probe_model
    torch.cuda.empty_cache()
    sys.exit(0)

del probe_model
torch.cuda.empty_cache()

# ---------------- 显存探针（与训练同构） ----------------
gate_a_res = None
try:
    _pm = Cards().to(device)
    _loss = masked_ce(_pm, train[:BATCH])
    _loss.backward()
    del _loss, _pm
    torch.cuda.empty_cache()
except torch.cuda.OutOfMemoryError:
    BATCH = 16
    print(f"[META] OOM 回退 batch=32 -> {BATCH}", flush=True)

# ---------------- 主流程 ----------------
model_s1, gate_a_res, wall_s1 = train_run(SEEDS[0], False, BATCH, gate_a=True)
if gate_a_res is None or not all(v > 0 for v in gate_a_res.values()):
    print("[GATE A] 不过 ⇒ 梯度设计无效，停。", flush=True)
    sys.exit(4)

# ★A 四项（seed1）
ce_s1 = eval_ce(model_s1, test)
cm1, cs1 = mean_se(ce_s1)
txt_s1_b1 = greedy_gen(model_s1, test, batch=EM_BATCH, mode="correct")
hits_s1 = hits_from(txt_s1_b1, test)
print(f"[MAIN] A臂(只输出答案) seed={SEEDS[0]}: 配对CE={cm1:.4f}±{cs1:.4f}  n={len(hits_s1)}",
      flush=True)
rep1 = report_ce(f"seed={SEEDS[0]} A臂(无CoT)", model_s1, test)
em_buckets(f"seed={SEEDS[0]} A臂(无CoT,batch={EM_BATCH}正确写法)", hits_s1)
free_gen_report(f"seed={SEEDS[0]}", hits_s1, test, k=20)

# 并列参考：batch=N（先过 R28 自检；不一致即作废该口径）
_r28_trained = r28_selfcheck(model_s1, test, k=CHK_BATCH)
txt_s1_bN = greedy_gen(model_s1, test, batch=CHK_BATCH, mode="correct")
hits_s1_N = hits_from(txt_s1_bN, test)
em_buckets(f"seed={SEEDS[0]} A臂(batch={CHK_BATCH}参考,自检{_r28_trained['sameN']}"
           f"/{_r28_trained['k']}一致)", hits_s1_N)

torch.save(model_s1.state_dict(), CKPT_TMPL.format(seed=SEEDS[0]))
print(f"[CKPT] saved {CKPT_TMPL.format(seed=SEEDS[0])}", flush=True)

# ---------------- 门 B：恒等中介（同 seed、同步数）—— 只报配对 ΔCE ----------------
model_id, _, wall_id = train_run(SEEDS[0], True, BATCH, gate_a=False)
ce_id = eval_ce(model_id, test)
del model_id
torch.cuda.empty_cache()
ce_id_m, ce_id_se = mean_se(ce_id)
diffs = [a - b for a, b in zip(ce_id, ce_s1)]
d_m, d_se = mean_se(diffs)
gate_b_pass = d_m > 2 * d_se
print(f"[GATE B] 恒等中介 CE={ce_id_m:.4f}±{ce_id_se:.4f} | A臂完整闭环 CE={cm1:.4f} "
      f"| 配对ΔCE(恒等-完整)={d_m:+.4f}±{d_se:.4f} ⇒ "
      f"{'PASS A臂显著更优(思维卡在干活)' if gate_b_pass else 'FAIL 差<2SE'}", flush=True)
print(f"[GATE B] 参照 S11(B臂) 同门配对ΔCE=+0.6640±0.0073 ⇒ 本臂与其差="
      f"{d_m - 0.6640:+.4f}", flush=True)
if not gate_b_pass:
    print("[GATE B] 不过 ⇒ 如实报（继续跑 seed2，主表供参考）。", flush=True)

# ---------------- seed2 ----------------
model_s2, _, wall_s2 = train_run(SEEDS[1], False, BATCH, gate_a=False)
ce_s2 = eval_ce(model_s2, test)
cm2, cs2 = mean_se(ce_s2)
txt_s2_b1 = greedy_gen(model_s2, test, batch=EM_BATCH, mode="correct")
hits_s2 = hits_from(txt_s2_b1, test)
print(f"[MAIN] A臂(只输出答案) seed={SEEDS[1]}: 配对CE={cm2:.4f}±{cs2:.4f}  n={len(hits_s2)}",
      flush=True)
rep2 = report_ce(f"seed={SEEDS[1]} A臂(无CoT)", model_s2, test)
em_buckets(f"seed={SEEDS[1]} A臂(无CoT,batch={EM_BATCH}正确写法)", hits_s2)
free_gen_report(f"seed={SEEDS[1]}", hits_s2, test, k=20)
txt_s2_bN = greedy_gen(model_s2, test, batch=CHK_BATCH, mode="correct")
hits_s2_N = hits_from(txt_s2_bN, test)
em_buckets(f"seed={SEEDS[1]} A臂(batch={CHK_BATCH}参考)", hits_s2_N)
torch.save(model_s2.state_dict(), CKPT_TMPL.format(seed=SEEDS[1]))
print(f"[CKPT] saved {CKPT_TMPL.format(seed=SEEDS[1])}", flush=True)
del model_s2
torch.cuda.empty_cache()
print(f"[EXTRACT] 抽取方式计数 {dict(EXTRACT_STAT)}", flush=True)

# ---------------- ★判决：容量 vs CoT 长度 ----------------
B_DIG = (0.9676, 0.9579)          # S11(B臂) 数字/算子 CE（引用，不重跑）
B_ACC = (38.0, 38.4)              # S11(B臂) 数字每步正确率 %
accs = [rep1["acc"] * 100, rep2["acc"] * 100]
ses = [rep1["acc_se"] * 100, rep2["acc_se"] * 100]
diffs_a = [a - 38.2 for a in accs]
z = [(d / se if se else float("inf")) for d, se in zip(diffs_a, ses)]
both_up = all(d > 2 * s for d, s in zip(diffs_a, ses))
print(f"[B-REF] ★B臂(S11,引用不重跑) 整体CE=0.7705/0.7388 每步正确率=46.3%/47.8% | "
      f"数字CE={B_DIG[0]}/{B_DIG[1]} 数字每步正确率={B_ACC[0]}%/{B_ACC[1]}% "
      f"→ 均值38.2%", flush=True)
print(f"[VERDICT-S12] A臂数字每步正确率={accs[0]:.1f}%±{ses[0]:.2f} / {accs[1]:.1f}%±{ses[1]:.2f} "
      f"vs B臂38.2% | Δ={diffs_a[0]:+.1f}pp/{diffs_a[1]:+.1f}pp "
      f"⇒ 2SE 判据 {'A 显著更高 ⇒ CoT长度/暴露偏差是主因' if both_up else '无显著差别(仍≈38%) ⇒ 容量/架构是主因' if all(abs(d) <= 2*s for d, s in zip(diffs_a, ses)) else '两者都不是 ⇒ 如实报'}",
      flush=True)
print(f"[VERDICT-S12] 说明：B 臂未报 token 级 SE，2SE 用 A 臂样本级 SE 作单侧近似；"
      f"A 整体CE={rep1['ce_m']:.4f}/{rep2['ce_m']:.4f} 每步正确率="
      f"{math.exp(-rep1['ce_m'])*100:.1f}%/{math.exp(-rep2['ce_m'])*100:.1f}%", flush=True)

print(f"[META] steps={STEPS} batch={BATCH} lr={LR} d={D} ff={FF} MAXLEN={MAXLEN} prompt优先 "
      f"| 生成主口径=batch{EM_BATCH}(正确写法) 自检(batch1 vs batch{CHK_BATCH})="
      f"{_r28_trained['sameN']}/{_r28_trained['k']}一致 "
      f"| seed1墙钟={wall_s1:.0f}s seed2墙钟={wall_s2:.0f}s 门B墙钟={wall_id:.0f}s "
      f"| 总墙钟={(time.time()-t_start)/60:.1f}min 任一截断={100.0*n_cut_any/n_parsed:.1f}%",
      flush=True)
print(f"[META] ckpt: {CKPT_TMPL.format(seed=SEEDS[0])} / {CKPT_TMPL.format(seed=SEEDS[1])}",
      flush=True)
print("[DONE] exit=0", flush=True)
