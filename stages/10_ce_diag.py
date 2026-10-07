#!/usr/bin/env python3
"""S10 · 零训练 CE 诊断（GSM8K-CoT）—— ★只载 ckpt，不训练、不改超参。

三个问题，全部用 logs/09c_ckpt_seed{1234,5678}.pt：
  Q1 目标 span 内 per-token CE 按四类拆开（①数字与算子 ②模板/格式串 ③中文字符 ④其它），
     报均值±SE 与占比，并给出配对 Δ(①−②)（两 seed 同号才判「CE 被模板主导」成立）；
  Q2 贪心解码下「第几个 token 与 gold 不一致」的前缀匹配分布（桶A 前 50 条）；
  Q3 prompt∧target 都未截子集（=桶A1）与 桶A2（prompt 被截）的 EM 与 n（n<30 只报不下结论）。

只允许写本文件、logs/、/tmp；nanoSeek 只读。数据切分/模型/评估口径全部与 09 号文件
逐行同源 ⇒ 配对 CE 应复现 S09c 的 0.8688 / 0.8791（复现失败即如实报、不继续下结论）。
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

# ---------------- 配置 ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 256
MIN_P = 48           # ★修2：联合窗口截断时 prompt 至少保留的 token 数（目标优先 ⇒ 目标最多占 208）
TAIL_KEEP = 24       # ★修2：prompt 超额时保「头 + 尾(问句尾+<resp> cue)」，只截中部
STEPS = 3000
LR = 1e-3
SEEDS = (1234, 5678)
EPS = 1e-3          # 地板基线的平滑：p(命中)=1-EPS，其余均分 EPS
BATCH = 32          # OOM 时回退 16
GEN_BATCH = 32

DATA = NS_ROOT + "/data/chinese/clean_v3/gsm8k_cot_dialogue.txt"
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
# ★S09c：两个 seed 的模型落盘（以后改评估口径不用重训）
CKPT_TMPL = "/home/vesita/coding/my/flowme/logs/09c_ckpt_seed{seed}.pt"

t_start = time.time()

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
    # ★修2（保尾截断）：不再独立弹句截 prompt；当 len(prompt)+len(target) > MAXLEN 时
    #   目标（CoT 尾 + #### 答案行 + <eos>）优先占窗，prompt 保头部(+尾部 cue)、截中部。
    stream = DialogueStream(enc, window=10 ** 9)
    stream.append("用户", q)
    stream.enforce_window()
    prompt = stream.prompt()
    p_full = enc(prompt)
    target_text = r + EOS
    t_full = enc(target_text)
    cut_p = cut_t = False
    if len(p_full) + len(t_full) > MAXLEN:
        budget_t = MAXLEN - MIN_P
        if len(t_full) <= budget_t:
            t_ids = t_full                      # 目标整体放得下 ⇒ 一点都不截
        else:
            t_ids = t_full[-budget_t:]          # 截目标也只截头部 ⇒ 尾部(含####答案)必在窗内
            cut_t = True
        room = MAXLEN - len(t_ids)              # >= MIN_P = 48
        if len(p_full) > room:
            p_ids = p_full[: room - TAIL_KEEP] + p_full[-TAIL_KEEP:]
            cut_p = True
        else:
            p_ids = p_full
    else:
        p_ids, t_ids = p_full, t_full
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
print(f"[TRUNC] ★保尾截断(MIN_P={MIN_P} prompt保尾={TAIL_KEEP}) "
      f"截prompt={trunc_p}/{n_parsed} ({100.0*trunc_p/n_parsed:.1f}%) | "
      f"截target={trunc_t}/{n_parsed} ({100.0*trunc_t/n_parsed:.1f}%) | "
      f"两者都截={both_cut}/{n_parsed} ({100.0*both_cut/n_parsed:.1f}%) | "
      f"丢弃={dropped}", flush=True)
print(f"[SPAN] span_mismatch={span_mismatch} nonadditive={encode_nonadditive} "
      f"| test有gold(EM分母)={n_em}", flush=True)
print(f"[TOK] WordLevel V={V} PAD={PAD_ID} EOS={EOS_ID} "
      f"| import {ds_mod.__file__}", flush=True)

# ★S09c 分桶（PS2 RULER 口径）：桶A=目标未被截断，桶B=目标被截断
nA = sum(1 for r in test if not r["cut_t"])
nB = sum(1 for r in test if r["cut_t"])
nA_pc = sum(1 for r in test if not r["cut_t"] and r["cut_p"])   # 桶A里 prompt 仍被截
n_free = sum(1 for r in test if not r["cut_t"] and not r["cut_p"])  # prompt∧target 都未截
print(f"[BUCKET] test={len(test)} | 桶A(目标未截) n={nA} | 桶B(目标被截) n={nB} "
      f"| 桶A中prompt仍被截 n={nA_pc} | 两者都未截 n={n_free} "
      f"⇒ 区分度判据 桶A n>=30 → {'满足' if nA >= 30 else '不满足'}", flush=True)


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


@torch.no_grad()
def greedy_em(model: Cards, recs) -> tuple[list[dict], list[str], list[list[int]]]:
    """贪心解码 → ★S09c 鲁棒抽取 → 严格/归一化两口径 EM + 分桶标记。

    返回 (hits, texts, raws)：hits 只含「有 gold」的样本；texts 与 recs 等长、含无
    gold 样本；★S10 新增 raws——每条的原始生成 id（含触发停止的 <eos>），供 Q2 算漂移点。
    """
    model.eval()
    hits: list[dict] = []
    texts: list[str] = [""] * len(recs)
    raws: list[list[int]] = [[] for _ in recs]
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
            # ★S10 修正：批内行先垫到 n0=max(prompt)，生成 token 从列 n0 开始追加。
            #   S09c 旧写法 ids[j, lens[j]: lens[j]+gen_len[j]] 会把右侧 PAD 算进序列、
            #   并丢掉尾部 (n0-lens[j]) 个真实生成 token ⇒ Q2/Q3 一律用修正后的切片。
            raw = ids[j, n0: n0 + gen_len[j]].tolist()
            raws[i + j] = raw                      # 含触发停止的 <eos>，供 Q2 算漂移点
            g_ids = [x for x in raw if x != EOS_ID]
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
    return hits, texts, raws


def em_buckets(tag: str, hits: list[dict]):
    """★S09c 分桶 EM 报告：桶A / 桶B / 全量 × 严格与归一化两口径（带 n）。"""
    out = {}
    rows = (("桶A(目标未截)", [h for h in hits if h["bucket"] == "A"]),
            ("桶B(目标被截)", [h for h in hits if h["bucket"] == "B"]),
            ("全量", list(hits)),
            ("附·两者都未截(参考,预期n<30)", [h for h in hits if h["free"]]))
    for name, sel in rows:
        key = name.split("(")[0]
        n = len(sel)
        s = sum(h["strict"] for h in sel) / n if n else float("nan")
        m = sum(h["norm"] for h in sel) / n if n else float("nan")
        out[key] = (n, s, m)
        if n == 0:
            print(f"[EM] {tag} {name}: n=0 无样本", flush=True)
            continue
        print(f"[EM] {tag} {name}: n={n} 严格EM={s*100:.2f}%  归一化EM={m*100:.2f}%",
              flush=True)
    return out


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n) if n else float("nan")


# ---------------- ★S10 主流程：只载 ckpt、只评估（不训练、不改超参） ----------------
import statistics  # noqa: E402

NUM_CHARS = set("0123456789+-*/=%$")                       # Q1 ①：数字与算子
TMPL_CHARS = set("【】详细解题思路推理与") | {"<", ">", "#"}     # Q1 ②：固定头部串 + #### + <<>> 构件
CAT_NAMES = ("①数字与算子", "②模板/格式串", "③中文字符", "④其它")
_SPECIAL = {"<eos>", "<unk>", "<pad>", "<bos>", "<cont>", "<sep>", "<resp>",
            "<call>", "<result>", "<answer>", "<tool>", "<search>", "<topic>"}
REF_CE = {1234: 0.8688, 5678: 0.8791}                      # S09c 实测，用于核对 ckpt 评估一致
Q2_N = 50


def classify(tid: int) -> int:
    """token 类别：0=数字与算子 1=模板/格式串 2=中文字符 3=其它。判序按 Q1 规格（算子先于模板）。"""
    s = tok.id_to_token(int(tid)) or ""
    if s in _SPECIAL or (s.startswith("<") and s.endswith(">")):
        return 1
    if s and all(c in NUM_CHARS for c in s):
        return 0
    if any(c in TMPL_CHARS for c in s):
        return 1
    if any("一" <= c <= "鿿" for c in s):        # CJK 统一表意文字
        return 2
    return 3


@torch.no_grad()
def ce_by_category(model: Cards, recs):
    """目标 span 上的 per-token CE 按四类拆开。

    返回 (各类 token CE 列表, 每条样本各类均值, 每条样本整体均值)。
    最后一项的 mean_se 与 09 号文件 eval_ce 完全同口径 ⇒ 可核对 ckpt 评估一致。
    """
    model.eval()
    tok_ce: list[list[float]] = [[] for _ in range(4)]
    samp: list[dict[int, float]] = []
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
            g: dict[int, list[float]] = {0: [], 1: [], 2: [], 3: []}
            for pos, v in enumerate(ces):
                k = classify(int(tgt[pos]))
                tok_ce[k].append(v)
                g[k].append(v)
            samp.append({k: sum(v) / len(v) for k, v in g.items() if v})
            samp_all.append(sum(ces) / len(ces))
    model.train()
    return tok_ce, samp, samp_all


def drift_of(raw: list[int], gold: list[int]) -> tuple[int, float]:
    """Q2：前缀匹配长度（与 gold 一致的 token 数）与占比；漂移点 = 第 prefix+1 个 token。"""
    n = min(len(raw), len(gold))
    p = n
    for i in range(n):
        if raw[i] != gold[i]:
            p = i
            break
    return p, p / max(1, len(gold))


verd: dict[int, dict] = {}
t_diag = time.time()
print("[S10] ★贪心文本切片已修正：生成 token 从列 n0=max(prompt) 起算"
      "（S09c 旧口径 ids[lens:...] 混入右侧 PAD 并截掉尾部 n0-lens 个生成 token；"
      "首步 top5 已用 batch=1 隔离复算：首 token=【(id115) 概率≈1，是模型真实输出）", flush=True)

for seed in SEEDS:
    path = CKPT_TMPL.format(seed=seed)
    m = Cards(False).to(device)
    m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
    print(f"[S10] loaded {path}", flush=True)

    # ---------- Q1：目标 span 的 per-token CE 按四类拆开 ----------
    tok_ce, samp, samp_all = ce_by_category(m, test)
    ntok = sum(len(v) for v in tok_ce)
    ce_m, ce_se = mean_se(samp_all)                       # ★与 09 号文件 eval_ce 完全同口径
    ce_w = sum(sum(v) for v in tok_ce) / ntok
    ref = REF_CE[seed]
    print(f"[Q1] seed={seed} 配对CE(样本级,同09口径)={ce_m:.4f}±{ce_se:.4f} n={len(samp_all)} | "
          f"S09c参照={ref:.4f} ⇒ "
          f"{'一致(ckpt 评估复现)' if abs(ce_m - ref) < 5e-4 else '★不一致,下面结论作废'}", flush=True)
    print(f"[Q1] seed={seed} CE(token加权)={ce_w:.4f} ntok={ntok}｜下表均值=token加权均值，±SE=样本级(739条)",
          flush=True)
    for k in range(4):
        xs = tok_ce[k]
        if not xs:
            note = ("（目标 span 内没有该类 token：GSM8K-CoT 的 target 是英文推理+模板头，"
                    "中文只出现在【详细解题思路与推理】里、已归入② ⇒ 此处是「无样本」，不是 CE=0）"
                    if k == 2 else "")
            print(f"[Q1] seed={seed} {CAT_NAMES[k]}: n=0 占比=0.0% CE=无法估计{note}", flush=True)
            continue
        km = sum(xs) / len(xs)
        _, kse = mean_se([s[k] for s in samp if k in s])
        print(f"[Q1] seed={seed} {CAT_NAMES[k]}: n={len(xs)} 占比={100.0*len(xs)/ntok:.1f}% "
              f"CE={km:.4f}±{kse:.4f} nats", flush=True)
    d = [s[0] - s[1] for s in samp if 0 in s and 1 in s]
    dm, dse = mean_se(d)
    print(f"[Q1] seed={seed} 配对Δ(①−②, 每条样本内均值之差)={dm:+.4f}±{dse:.4f} n={len(d)} ⇒ "
          f"{'①数字更高' if dm > 0 else '②模板更高'}"
          f"{' (显著, >2SE)' if dm > 2 * dse else ' (不显著)'}", flush=True)

    # ---------- 贪心解码（Q2/Q3 共用，一次跑完） ----------
    hits, texts, raws = greedy_em(m, test)

    # ---------- Q2：漂移点分布（桶A 前 50 条） ----------
    sel = [k for k, r in enumerate(test) if not r["cut_t"]][:Q2_N]
    ratios, prefs, glens = [], [], []
    for k in sel:
        p, ratio = drift_of(raws[k], test[k]["t"])
        prefs.append(p)
        ratios.append(ratio)
        glens.append(len(test[k]["t"]))
    med = statistics.median(ratios)
    c_lo = sum(1 for x in ratios if x < 0.10)
    c_mid = sum(1 for x in ratios if 0.10 <= x <= 0.50)
    c_hi = sum(1 for x in ratios if x > 0.50)
    print(f"[Q2] seed={seed} n={len(ratios)}(桶A·贪心) 漂移起点中位=第{statistics.median(prefs)+1:.0f}个token "
          f"前缀占比中位={med*100:.1f}% (目标长度中位={statistics.median(glens):.0f}tok) | "
          f"<10%:{c_lo}条 10–50%:{c_mid}条 >50%:{c_hi}条", flush=True)
    for _i, k in enumerate(sel):                       # 逐条明细（只进日志，不进报告）
        _p, _r = drift_of(raws[k], test[k]["t"])
        print(f"[Q2-D] seed={seed} [{_i}] idx={k} 前缀={_p}/{len(test[k]['t'])} ({_r*100:.1f}%) "
              f"gen={raws[k][:8]}|{dec(raws[k][:8])!r} gold={test[k]['t'][:8]}|{dec(test[k]['t'][:8])!r}",
              flush=True)
    if seed == SEEDS[0]:
        # 对照：单条(batch=1, 无右侧 PAD)重算首步 top5，判「首 token=<pad>」是模型行为还是批内伪影
        m.eval()
        for k in sel[:8]:
            ids1, s1 = build_batch([test[k]])
            lg = torch.log_softmax(m.logits(ids1), dim=-1)[0, int(s1[0]) - 1]
            top = torch.topk(lg, 5)
            print(f"[Q2-P] idx={k} 隔离首步top5={list(zip(top.indices.tolist(), [round(float(x), 3) for x in top.values.tolist()]))} "
                  f"| 贪心(批内)首token={raws[k][:4]}", flush=True)

    # ---------- Q3：未截断子集 / 桶A1 / 桶A2 的 EM ----------
    def _em(sel, key):
        n = len(sel)
        return (sum(h[key] for h in sel) / n * 100 if n else float("nan")), n

    rows = (("Q3·两者都未截(=桶A1,prompt与target全)", [h for h in hits if h["free"]]),
            ("Q3·桶A2(prompt被截,target未截)", [h for h in hits if h["bucket"] == "A" and not h["free"]]),
            ("参考·桶A", [h for h in hits if h["bucket"] == "A"]),
            ("参考·桶B", [h for h in hits if h["bucket"] == "B"]),
            ("参考·全量", list(hits)))
    n_a1 = None
    for name, sel_h in rows:
        ms, n = _em(sel_h, "strict")
        mn, _ = _em(sel_h, "norm")
        tag = "  ⇒ n<30 样本不足以判(不下结论)" if n < 30 else ""
        print(f"[Q3] seed={seed} {name}: n={n} 严格EM={ms:.2f}% 归一化EM={mn:.2f}%{tag}", flush=True)
        if n_a1 is None:
            n_a1 = n
    verd[seed] = dict(d=(dm, dse), med=med, n_a1=n_a1, c=(c_lo, c_mid, c_hi))

    del m
    torch.cuda.empty_cache()

# ---------------- 判定（三条分开报） ----------------
q1 = all(v["d"][0] > 2 * v["d"][1] for v in verd.values())
q2 = all(v["med"] < 0.10 for v in verd.values())
n_a1 = verd[SEEDS[0]]["n_a1"]
print(f"[VERDICT] Q1 {'成立：CE 被模板 token 主导（①显著高于②，两 seed 同号）' if q1 else '不成立：① 未显著高于 ② ⇒ CE 低可能是真的学会了'} "
      f"| Δ(①−②)=" + " ".join(f"seed{s}={verd[s]['d'][0]:+.4f}±{verd[s]['d'][1]:.4f}" for s in SEEDS), flush=True)
print(f"[VERDICT] Q2 {'成立：暴露偏差（漂移中位 <10%）' if q2 else '不成立：漂移中位 ≥10%'} | "
      + " ".join(f"seed{s}={verd[s]['med']*100:.1f}% " + str(verd[s]["c"]) for s in SEEDS), flush=True)
print(f"[VERDICT] Q3 全未截子集 n={n_a1} {'<30 ⇒ 样本不足以判，不许下结论' if n_a1 < 30 else '≥30 可判'}", flush=True)
print(f"[META] steps={STEPS} batch={BATCH} lr={LR} d={D} ff={FF} test={len(test)} "
      f"| 诊断段耗时={(time.time()-t_diag)/60:.1f}min 总墙钟={(time.time()-t_start)/60:.1f}min", flush=True)
print("[DONE] exit=0", flush=True)

