#!/usr/bin/env python3
"""S11 · 决定性一枪：窗口 256→512 + 截断优先级反转（prompt 优先）—— ★GPU 训练。

架构、超参、步数、数据切分、EM/CE 评估口径**全部与 stages/09_textgen_gsm8k.py 同源**，
本文件只改「窗口与截断优先级」这一个变量（与 S09b/S09c 可比）：

  ★改动1  MAXLEN: 256 → 512（模型 PE 缓冲随 MAXLEN 建，属窗口的一部分）；
  ★改动2  截断优先级反转：**prompt 优先**（题干完整留下），
          目标放不下时才截目标**头部**（保含 `####` 的尾部）；
          与上一版（目标优先）相反。
          若 prompt 本身 > MAXLEN−MIN_T，则按 head+tail 截 prompt（保头 + 保尾
          TAIL_KEEP=24 的问句尾 cue），并计入截断率。
  ★改动3  新截断三项（截 prompt / 截 target / 两者都截）+「两者都未截」n
          （判据：test 内 n≥100 才可判，否则如实报「本单元无法判」）。
  ★改动4  报 CE 新增两行：数字/算子 token 的 CE（按 S10 的 `[0-9+-*/=%$]` 分类口径）
          与「每步正确率 = exp(−CE)」（整体 + 数字 token 单列）。
  ★改动5  EM 桶改为 全量 / 桶A(目标未截) / 桶A1(两者都未截)，各带 n。

不变的东西（硬约束）：D/FF/NHEAD/STEPS/LR/SEEDS/BATCH、9:1 切分（seed42）、
地板三条、门 A/B/C、EM 严格与归一化口径、贪心解码实现（含其批内查询列索引，
**一处未改**，以保与 S09c 可比）。

ckpt 落盘：logs/11_ckpt_seed{1234,5678}.pt（S09C_EVAL_ONLY / S11_EVAL_ONLY=1 只评估）。
只允许写本文件与 logs/、/tmp；nanoSeek 一律只读。
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

# ---------------- 配置（★只动窗口与截断优先级这一件事） ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512          # ★S11 改动1：256 → 512
MIN_T = 48            # ★S11 改动2：prompt 优先 ⇒ 目标至少留 48（prompt 上限 = MAXLEN−48 = 464）
TAIL_KEEP = 24        # prompt 自身超额时 head+tail 截断：保头 + 保尾 24（问句尾 + <resp> cue）
STEPS = 3000
LR = 1e-3
SEEDS = (1234, 5678)
EPS = 1e-3          # 地板基线的平滑：p(命中)=1-EPS，其余均分 EPS
BATCH = 32          # OOM 时回退 16
GEN_BATCH = 32

DATA = NS_ROOT + "/data/chinese/clean_v3/gsm8k_cot_dialogue.txt"
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
# ★S11：两个 seed 的模型落盘（同 S09C_EVAL_ONLY 的可复现模式）
CKPT_TMPL = "/home/vesita/coding/my/flowme/logs/11_ckpt_seed{seed}.pt"

t_start = time.time()

print(f"[S11] ★只改一个变量=窗口与截断优先级：① MAXLEN 256→{MAXLEN}；"
      f"② 截断优先级反转=prompt优先（题干完整留下；目标放不下才截**头部**、保含 #### 的尾部；"
      f"prompt 自身 > MAXLEN−{MIN_T} 时按 head+tail 截 prompt 并计入截断率）；"
      f"③ 报新截断三项 + 「两者都未截」n（判据 test n>=100）。"
      f" 其余不动：steps={STEPS} batch={BATCH} lr={LR} d={D} ff={FF} seeds={SEEDS} 切分=seed42(9:1)",
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
    """#### 终局答案解析（nanoSeek 里没有 ⇒ 自研，自包含）。

    ★S09c：这就是「严格精确匹配」口径 = S09b 原口径（逐字符；两边都去过
    空白/千分位逗号/`$`），保留不动，保证与 S09b 可比。
    """
    if "####" not in text:
        return None
    a = text.rsplit("####", 1)[1].split("\n")[0]
    a = a.strip().replace(",", "").replace(" ", "").replace("$", "")
    return a or None


# ---------------- ★S09c：鲁棒抽取 + 归一化（只改测量，不改架构/判据） ----------------
EXTRACT_STAT = Counter()          # 抽取方式计数（#### / boxed兜底 / 都取不到）

# ★S09c 修测量（根因）：语料里 target = 回复 + EOS，而 EOS = '<eos>'
#   （nanoSeek/training/dialogue_stream.py:93）⇒ parse_ans 出来的 gold 形如 '72<eos>'，
#   但生成文本按设计滤掉了 EOS_ID、永不含 '<eos>' ⇒ 旧口径下严格 EM **结构性恒为 0**
#   （S09b EM 全 0 的根因 = 测量问题）。哨兵不是答案的一部分 ⇒ 抽取/归一化时剥掉它；
#   floor 两侧同样剥，floor EM 数值不变（已核：14/739 = 1.89% 不变）。
MARK_RE = re.compile(r"</?(?:eos|pad|resp|cont|bos|unk)>", re.IGNORECASE)


def strip_mark(s):
    """剥掉 <eos>/<pad>/... 轮末哨兵；None 原样返回。"""
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
    """鲁棒抽取：优先 `####` 后的答案；没有 `####` 时兜底取最后一个 `\\boxed{}`。"""
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
    """归一化口径：去空白/千分位逗号/货币符号/尾部单位，认 \\boxed{}，返回可比字符串。"""
    if a is None:
        return None
    s = str(a).strip()
    m = re.search(r"\\boxed\{([^{}]*)\}", s)
    if m:
        s = m.group(1)
    s = strip_mark(s)                                   # ★剥 <eos> 等轮末哨兵
    s = re.sub(r"[\s,，]", "", s)                       # 空白 + 千分位逗号
    s = s.replace("$", "").replace("￥", "").replace("€", "").replace("£", "")
    s = s.replace("−", "-").replace("—", "-")
    for _ in range(3):                                  # 单位可能叠（"$5美元"）
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
    """归一化 EM：纯数字按数值比（1,234 == 1234），非数字按归一化字符串比。"""
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
    """严格精确匹配口径 = S09b 原口径（parse_ans 后逐字符）+ ★剥掉 <eos> 哨兵。

    只剥哨兵、不做任何数值/单位宽容 ⇒ 仍然是「精确匹配」；不剥哨兵时该口径
    结构性恒为 0（见上 MARK_RE 处的说明）。
    """
    if gold is None:
        return 0
    p = parse_ans(text)
    if p is None:
        return 0
    return int(strip_mark(p) == strip_mark(gold))


# ---------------- 数据 ----------------
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


for q, r in raw_pairs:
    # ★S11 改动2（prompt 优先，与上一版「目标优先」相反）：
    #   题干要完整留下 ⇒ 只有当 len(prompt)+len(target) > MAXLEN 才动刀；
    #   放不下时先给 prompt 让路（prompt 上限 MAXLEN−MIN_T，超额走 head+tail 截中部），
    #   剩下的 room 给目标；目标仍放不下 ⇒ 只截目标**头部**（尾部含 #### 答案必在窗内）。
    stream = DialogueStream(enc, window=10 ** 9)
    stream.append("用户", q)
    stream.enforce_window()
    prompt = stream.prompt()
    p_full = enc(prompt)
    target_text = r + EOS
    t_full = enc(target_text)
    cut_p = cut_t = False
    if len(p_full) + len(t_full) > MAXLEN:
        p_room = MAXLEN - MIN_T                 # prompt 允许占的上限（464）
        if len(p_full) > p_room:                # prompt 自身超额 ⇒ head+tail 截中部，计入截断率
            p_ids = p_full[: p_room - TAIL_KEEP] + p_full[-TAIL_KEEP:]
            cut_p = True
        else:
            p_ids = p_full                      # ★prompt 完整保留
        room = MAXLEN - len(p_ids)              # >= MIN_T = 48
        if len(t_full) <= room:
            t_ids = t_full                      # 目标整体放得下 ⇒ 一点都不截
        else:
            t_ids = t_full[-room:]              # 截目标只截头部 ⇒ 尾部(含####答案)必在窗内
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
    # 切分器口径 vs 拼接口径：只在未截断时校验（截断后按设计与全文口径不同）
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
      f"| dropped={dropped}", flush=True)
# ★S11 改动3：新三项（截prompt / 截target / 两者都截）+「两者都未截」n
n_free_all = n_parsed - n_cut_any
print(f"[TRUNC] ★S11 prompt优先(MAXLEN={MAXLEN} MIN_T={MIN_T} prompt头尾保={TAIL_KEEP}) "
      f"截prompt={trunc_p}/{n_parsed} ({100.0*trunc_p/n_parsed:.1f}%) | "
      f"截target={trunc_t}/{n_parsed} ({100.0*trunc_t/n_parsed:.1f}%) | "
      f"两者都截={both_cut}/{n_parsed} ({100.0*both_cut/n_parsed:.1f}%) | "
      f"两者都未截={n_free_all}/{n_parsed} ({100.0*n_free_all/n_parsed:.1f}%) | "
      f"丢弃={dropped}", flush=True)
print(f"[SPAN] span_mismatch={span_mismatch} nonadditive={encode_nonadditive} "
      f"| test有gold(EM分母)={n_em}", flush=True)
print(f"[TOK] WordLevel V={V} PAD={PAD_ID} EOS={EOS_ID} "
      f"| import {ds_mod.__file__}", flush=True)

# ★S09c 分桶（PS2 RULER 口径）：桶A=目标未被截断，桶B=目标被截断
nA = sum(1 for r in test if not r["cut_t"])
nB = sum(1 for r in test if r["cut_t"])
nA_pc = sum(1 for r in test if not r["cut_t"] and r["cut_p"])   # 桶A里 prompt 仍被截
n_free = sum(1 for r in test if not r["cut_t"] and not r["cut_p"])  # prompt∧target 都未截 = 桶A1
print(f"[BUCKET] test={len(test)} | 桶A(目标未截) n={nA} | 桶B(目标被截) n={nB} "
      f"| 桶A中prompt仍被截 n={nA_pc} | 两者都未截(=桶A1) n={n_free} "
      f"⇒ 桶A n>=30 → {'满足' if nA >= 30 else '不满足'} | "
      f"★S11 判据 桶A1 n>=100 → {'满足,可判' if n_free >= 100 else '不满足,本单元无法判'}",
      flush=True)


# ---------------- 模型：三张卡 ----------------
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
        # F.multi_head_attention_forward 要求 [B*num_heads, n, n]
        m = m.repeat_interleave(NHEAD, dim=0)
        # ★修1（off-by-one）：pe 只建到 MAXLEN，而生成批宽 = max_prompt + (MAXLEN - min_prompt)
        #   会超过 MAXLEN（上次正是 n=257 崩在 pe[:256]）。按 n 动态扩展 + 断言兜底。
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
    """只在目标 span 上算 loss（teacher forcing：位置 s-1..e-1 预测 t）。"""
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


# ---------------- ★S11 改动4：数字/算子 CE（分类口径逐字沿用 stages/10_ce_diag.py） ----------
NUM_CHARS = set("0123456789+-*/=%$")                 # S10 的 `[0-9+-*/=%$]`
TMPL_CHARS = set("【】详细解题思路推理与") | {"<", ">", "#"}
CAT_NAMES = ("①数字与算子", "②模板/格式串", "③中文字符", "④其它")
_SPECIAL = {"<eos>", "<unk>", "<pad>", "<bos>", "<cont>", "<sep>", "<resp>",
            "<call>", "<result>", "<answer>", "<tool>", "<search>", "<topic>"}


def classify(tid: int) -> int:
    """token 类别：0=数字与算子 1=模板/格式串 2=中文字符 3=其它（与 S10 完全同序）。"""
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
def ce_categories(model: Cards, recs) -> tuple[list[list[float]], list[float]]:
    """目标 span 上的 per-token CE 按 S10 四类拆开；返回 (各类 token CE 列表, 每条样本整体均值)。"""
    model.eval()
    tok_ce: list[list[float]] = [[] for _ in range(4)]
    samp_all: list[float] = []
    for i in range(0, len(recs), GEN_BATCH):
        chunk = recs[i: i + GEN_BATCH]
        ids, s = build_batch(chunk)
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        for j, r in enumerate(chunk):
            e = s[j] + len(r["t"])
            lp = logp[j, s[j] - 1: e - 1]
            tgt = ids[j, s[j]: e]
            ces = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1).tolist()
            for pos, v in enumerate(ces):
                tok_ce[classify(int(tgt[pos]))].append(v)
            samp_all.append(sum(ces) / len(ces))
    model.train()
    return tok_ce, samp_all


def report_ce(tag: str, model: Cards, recs) -> None:
    """★报：整体配对 CE + 数字/算子 CE + 两条「每步正确率」= exp(−CE)。"""
    tok_ce, samp_all = ce_categories(model, recs)
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
    if dig_ce is not None:
        print(f"[CE] {tag} ★数字/算子CE(token加权)={dig_ce:.4f} n={dig_n} "
              f"占比={100.0*dig_n/ntok:.1f}% | ★数字每步正确率=exp(−数字CE)="
              f"{math.exp(-dig_ce)*100:.1f}%", flush=True)


@torch.no_grad()
def greedy_em(model: Cards, recs) -> tuple[list[dict], list[str]]:
    """贪心解码 → ★S09c 鲁棒抽取 → 严格/归一化两口径 EM + 分桶标记。

    返回 (hits, texts)：hits 只含「有 gold」的样本（口径与 S09b 一致，
    便于与 floor_em 按序号配对）；texts 与 recs 等长、含无 gold 样本。
    """
    model.eval()
    hits: list[dict] = []
    texts: list[str] = [""] * len(recs)
    for i in range(0, len(recs), GEN_BATCH):
        chunk = recs[i: i + GEN_BATCH]
        lens = [len(r["p"]) for r in chunk]
        n0 = max(lens)
        ids = torch.full((len(chunk), n0), PAD_ID, dtype=torch.long)
        for j, r in enumerate(chunk):
            ids[j, : lens[j]] = torch.tensor(r["p"])
        ids = ids.to(device)
        gen_len = [0] * len(chunk)
        cap = [MAXLEN - L for L in lens]
        done = [False] * len(chunk)
        for _ in range(max(cap)):
            if all(done):
                break
            logits = model.logits(ids)
            nxt = []
            for j in range(len(chunk)):
                if done[j]:
                    nxt.append(PAD_ID)
                    continue
                k = lens[j] + gen_len[j] - 1
                t = int(logits[j, k].argmax().item())
                nxt.append(t)
                gen_len[j] += 1
                if t == EOS_ID or gen_len[j] >= cap[j]:
                    done[j] = True
            ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long, device=device).unsqueeze(1)], 1)
        for j, r in enumerate(chunk):
            g_ids = ids[j, lens[j]: lens[j] + gen_len[j]].tolist()
            g_ids = [x for x in g_ids if x != EOS_ID]
            text = dec(g_ids)
            texts[i + j] = text
            if r["gold"] is None:
                continue
            pred = extract_ans(text)
            hits.append(dict(idx=i + j, gold=r["gold"], pred=pred, gen=text,
                             strict=em_strict(r["gold"], text),
                             norm=em_norm(r["gold"], pred),
                             bucket=("B" if r["cut_t"] else "A"),
                             free=(not r["cut_t"] and not r["cut_p"])))
    model.train()
    return hits, texts


def em_buckets(tag: str, hits: list[dict]):
    """★S09c 分桶 EM 报告 + ★S11 改动5：桶A / 桶A1(两者都未截) / 桶B / 全量，各带 n。"""
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
    """★S09c 自由生成 k 条（优先桶A），贪心，逐条打印原文。"""
    pool = [h for h in hits if h["bucket"] == "A"] + [h for h in hits if h["bucket"] == "B"]
    pool = sorted(pool[:k], key=lambda h: h["idx"])
    print(f"[GEN] {tag} 自由生成 {len(pool)} 条（贪心，优先桶A）", flush=True)
    for i, h in enumerate(pool):
        ptxt = dec(recs[h["idx"]]["p"]).replace("\n", "⏎")
        g = h["gen"].replace("\n", "⏎")
        head = g[:120]
        tail = ""
        if "####" in g:
            i0 = g.index("####")
            if i0 >= 120:
                tail = " …" + g[i0: i0 + 40]
        ok = "对" if h["norm"] else "错"
        ok_s = "" if h["norm"] == h["strict"] else f"(严格={'对' if h['strict'] else '错'})"
        print(f"[GEN] [{i}] {ptxt[-60:]} → {head}{tail} "
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
    if kind == "copy_tail":                       # ① 复制 prompt 尾部
        p = r["p"]
        return (p[-L:] if len(p) >= L else [p[0]] * (L - len(p)) + p)
    if kind == "common_ans":                      # ② 只输出最常见 #### 答案
        return enc("#### " + common_ans)
    return [freq_tok] * L                         # ③ 最常出现的 token


def floor_ce(kind: str, r) -> float:
    b = floor_b_ids(kind, r)
    ce = 0.0
    for pos, gold in enumerate(r["t"]):
        if pos >= len(b):
            ce += math.log(V)                     # 超出基线输出长度 ⇒ 均匀
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
        # 与模型侧同一口径（剥 <eos> 哨兵）；floor 文本与 gold 同源 ⇒ 数值不变
        hits.append(em_strict(r["gold"], floor_text(kind, r)))
    return hits


FLOORS = [("copy_tail", "①复制prompt尾部"), ("common_ans", "②最常见####答案"),
          ("freq_tok", "③最常出现token")]

# ---------------- ★S11：只评估模式（改口径不用重训，靠 logs/11_ckpt_seed*.pt） ----------------
if os.environ.get("S11_EVAL_ONLY", os.environ.get("S09C_EVAL_ONLY")) == "1":
    print("[EVAL-ONLY] 只做评估：载入 logs/11_ckpt_seed*.pt，不训练任何东西", flush=True)
    for kind, name in FLOORS:                       # 门 C 复算（确认地板口径没动）
        ces = [floor_ce(kind, r) for r in test]
        ems = floor_em(kind, test)
        c_m, c_se = mean_se(ces)
        e_m, e_se = mean_se(ems)
        print(f"[GATE C] {name}: 配对CE={c_m:.4f}±{c_se:.4f} nats  "
              f"EM={e_m*100:.2f}%±{e_se*100:.2f} (n={len(ems)})", flush=True)
    for seed in SEEDS:
        _p = CKPT_TMPL.format(seed=seed)
        m = Cards(False).to(device)
        m.load_state_dict(torch.load(_p, map_location=device, weights_only=True))
        print(f"[EVAL-ONLY] loaded {_p}", flush=True)
        ce = eval_ce(m, test)
        hits, _ = greedy_em(m, test)
        em = [h["strict"] for h in hits]
        cm, cs = mean_se(ce)
        emm, ems = mean_se(em)
        print(f"[MAIN] 完整闭环 seed={seed}: 配对CE={cm:.4f}±{cs:.4f}  "
              f"EM={emm*100:.2f}%±{ems*100:.2f} (n={len(em)})", flush=True)
        report_ce(f"seed={seed} 完整闭环", m, test)
        em_buckets(f"seed={seed} 完整闭环", hits)
        free_gen_report(f"seed={seed}", hits, test, k=20)
        del m
        torch.cuda.empty_cache()
    print(f"[EXTRACT] 抽取方式计数 {dict(EXTRACT_STAT)}", flush=True)
    print(f"[META] EVAL_ONLY MAXLEN={MAXLEN} prompt优先 steps={STEPS} batch={BATCH} lr={LR} "
          f"d={D} ff={FF} | 耗时={(time.time()-t_start)/60:.1f}min", flush=True)
    print("[DONE] exit=0", flush=True)
    sys.exit(0)

# ---------------- 主流程（三道门按序） ----------------
gate_a_res = None
gate_b_pass = None
results = {}

# 显存探针（与训练同构：同 batch、同长度）
probe_model = Cards().to(device)
try:
    _loss = masked_ce(probe_model, train[:BATCH])
    _loss.backward()
    del _loss, probe_model
    torch.cuda.empty_cache()
except torch.cuda.OutOfMemoryError:
    BATCH = 16
    print(f"[META] OOM 回退 batch=32 -> {BATCH}", flush=True)
    del probe_model

# 门 A + 完整闭环 seed1
model_s1, gate_a_res, wall_s1 = train_run(SEEDS[0], False, BATCH, gate_a=True)
if gate_a_res is None or not all(v > 0 for v in gate_a_res.values()):
    print("[GATE A] 不过 ⇒ 梯度设计无效，停。", flush=True)
    sys.exit(4)

# 门 C：三条地板（配对 CE + EM）
floor_res = {}
for kind, name in FLOORS:
    ces = [floor_ce(kind, r) for r in test]
    ems = floor_em(kind, test)
    ce_m, ce_se = mean_se(ces)
    em_m, em_se = mean_se(ems)
    floor_res[kind] = dict(ce=ces, em=ems, ce_m=ce_m, ce_se=ce_se, em_m=em_m, em_se=em_se)
    print(f"[GATE C] {name}: 配对CE={ce_m:.4f}±{ce_se:.4f} nats  "
          f"EM={em_m*100:.2f}%±{em_se*100:.2f} (n={len(ems)})", flush=True)

best_floor_ce = min(FLOORS, key=lambda kv: floor_res[kv[0]]["ce_m"])
best_floor_em = max(FLOORS, key=lambda kv: floor_res[kv[0]]["em_m"])

# 完整闭环 seed1 评测
ce_s1 = eval_ce(model_s1, test)
hits_s1, _ = greedy_em(model_s1, test)
em_s1 = [h["strict"] for h in hits_s1]          # ★CRIT/GATE B 仍用严格口径（不改判据）
ce_m, ce_se = mean_se(ce_s1)
em_m, em_se = mean_se(em_s1)
results[SEEDS[0]] = dict(ce=ce_s1, em=em_s1, ce_m=ce_m, ce_se=ce_se, em_m=em_m,
                         em_se=em_se, hits=hits_s1)
print(f"[MAIN] 完整闭环 seed={SEEDS[0]}: 配对CE={ce_m:.4f}±{ce_se:.4f}  "
      f"EM={em_m*100:.2f}%±{em_se*100:.2f} (n={len(em_s1)})", flush=True)
report_ce(f"seed={SEEDS[0]} 完整闭环", model_s1, test)
em_buckets(f"seed={SEEDS[0]} 完整闭环", hits_s1)
free_gen_report(f"seed={SEEDS[0]}", hits_s1, test, k=20)

# ★S09c 存 ckpt（以后改评估口径不用重训）
torch.save(model_s1.state_dict(), CKPT_TMPL.format(seed=SEEDS[0]))
print(f"[CKPT] saved {CKPT_TMPL.format(seed=SEEDS[0])}", flush=True)

# 门 B：恒等中介（同 seed、同步数）—— 报配对 CE + EM
model_id, _, wall_id = train_run(SEEDS[0], True, BATCH, gate_a=False)
ce_id = eval_ce(model_id, test)
hits_id, _ = greedy_em(model_id, test)
em_id = [h["strict"] for h in hits_id]
del model_id
torch.cuda.empty_cache()
ce_id_m, ce_id_se = mean_se(ce_id)
em_id_m, em_id_se = mean_se(em_id)
diffs = [a - b for a, b in zip(ce_id, ce_s1)]      # 恒等 − 完整（正 = 完整更好）
d_m, d_se = mean_se(diffs)
em_id_map = {i: h for i, h in enumerate(em_id)}
em_s1_map = {i: h for i, h in enumerate(em_s1)}
common_b = sorted(set(em_id_map) & set(em_s1_map))
dem_b = [em_id_map[i] - em_s1_map[i] for i in common_b]
b_m, b_se = mean_se(dem_b)
gate_b_pass = d_m > 2 * d_se
print(f"[GATE B] 恒等中介 CE={ce_id_m:.4f}±{ce_id_se:.4f} EM={em_id_m*100:.2f}%±{em_id_se*100:.2f} "
      f"| 完整闭环 CE={ce_m:.4f} EM={em_m*100:.2f}% "
      f"| 配对ΔCE(恒等-完整)={d_m:+.4f}±{d_se:.4f} ⇒ "
      f"{'PASS 完整显著更优' if gate_b_pass else 'FAIL 差<2SE 中介冗余/任务无效'}", flush=True)
print(f"[GATE B] 配对ΔEM(恒等-完整)={b_m*100:+.2f}±{b_se*100:.2f}pp (n={len(dem_b)})", flush=True)
em_buckets(f"seed={SEEDS[0]} 恒等中介", hits_id)
if not gate_b_pass:
    print("[GATE B] 不过 ⇒ 中介冗余/任务无效（如实报；仍继续跑 seed2，主表供参考）。",
          flush=True)

# 完整闭环 seed2
model_s2, _, wall_s2 = train_run(SEEDS[1], False, BATCH, gate_a=False)
ce_s2 = eval_ce(model_s2, test)
hits_s2, _ = greedy_em(model_s2, test)
em_s2 = [h["strict"] for h in hits_s2]
torch.save(model_s2.state_dict(), CKPT_TMPL.format(seed=SEEDS[1]))
print(f"[CKPT] saved {CKPT_TMPL.format(seed=SEEDS[1])}", flush=True)
ce_m2, ce_se2 = mean_se(ce_s2)
em_m2, em_se2 = mean_se(em_s2)
results[SEEDS[1]] = dict(ce=ce_s2, em=em_s2, ce_m=ce_m2, ce_se=ce_se2, em_m=em_m2,
                         em_se=em_se2, hits=hits_s2)
print(f"[MAIN] 完整闭环 seed={SEEDS[1]}: 配对CE={ce_m2:.4f}±{ce_se2:.4f}  "
      f"EM={em_m2*100:.2f}%±{em_se2*100:.2f} (n={len(em_s2)})", flush=True)
report_ce(f"seed={SEEDS[1]} 完整闭环", model_s2, test)
del model_s2
torch.cuda.empty_cache()
em_buckets(f"seed={SEEDS[1]} 完整闭环", hits_s2)
free_gen_report(f"seed={SEEDS[1]}", hits_s2, test, k=20)
print(f"[EXTRACT] 抽取方式计数 {dict(EXTRACT_STAT)}", flush=True)

# ---------------- 主判据 ----------------
fk, fname = best_floor_ce
ek, ename = best_floor_em
floor_ce_list, floor_em_list = floor_res[fk]["ce"], floor_res[ek]["em"]
print(f"[CRIT] 对比地板：CE地板={fname} ({floor_res[fk]['ce_m']:.4f})  "
      f"EM地板={ename} ({floor_res[ek]['em_m']*100:.2f}%)", flush=True)
ce_ok = em_ok = True
for seed in SEEDS:
    r = results[seed]
    dce = [a - b for a, b in zip(r["ce"], floor_ce_list)]
    m_ce, se_ce = mean_se(dce)
    em_map = {i: h for i, h in enumerate(r["em"])}
    fem_map = {i: h for i, h in enumerate(floor_em_list)}
    common = sorted(set(em_map) & set(fem_map))
    dem = [em_map[i] - fem_map[i] for i in common]
    m_em, se_em = mean_se(dem)
    ok_ce = m_ce < -2 * se_ce
    ok_em = m_em > 2 * se_em
    ce_ok &= ok_ce
    em_ok &= ok_em
    print(f"[CRIT] seed={seed}: ΔCE(闭环-地板)={m_ce:+.4f}±{se_ce:.4f} "
          f"{'显著低' if ok_ce else '不显著'} | ΔEM={m_em*100:+.2f}±{se_em*100:.2f}pp "
          f"{'显著高' if ok_em else '不显著'} (同号要求两 seed 均满足)", flush=True)

verdict = "PASS 主判据过 ⇒ 三类卡闭环在文本生成上成立" if (ce_ok and em_ok) \
    else "FAIL 主判据不过 ⇒ 如实报（最可能原因：数据少/步数不够/d=128 太小）"
print(f"[VERDICT] {verdict}", flush=True)
print(f"[S11-SUM] MAXLEN={MAXLEN} 256→512 + 截断优先级反转(prompt优先, MIN_T={MIN_T}) | "
      f"截prompt={trunc_p}/{n_parsed}({100.0*trunc_p/n_parsed:.1f}%) "
      f"截target={trunc_t}/{n_parsed}({100.0*trunc_t/n_parsed:.1f}%) "
      f"两者都截={both_cut}/{n_parsed}({100.0*both_cut/n_parsed:.1f}%) "
      f"两者都未截(全量)={n_free_all} test桶A1 n={n_free} "
      f"⇒ 判据n>=100 {'满足' if n_free >= 100 else '不满足'}", flush=True)
print(f"[META] steps={STEPS} batch={BATCH} lr={LR} d={D} ff={FF} MAXLEN={MAXLEN} prompt优先 "
      f"| seed1墙钟={wall_s1:.0f}s seed2墙钟={wall_s2:.0f}s 门B墙钟={wall_id:.0f}s "
      f"| 总墙钟={(time.time()-t_start)/60:.1f}min 任一截断={100.0*n_cut_any/n_parsed:.1f}%",
      flush=True)
print("[DONE] exit=0", flush=True)
