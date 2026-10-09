#!/usr/bin/env python3
"""S65 · 语言版 P4 门：把 S09–S12 的 GSM8K 文本生成闭环接进 cards/ 的 CardPipeline。

目标（NEXT_MAINLINE.md:5-8）：确认「框架能承载语言生成任务」——只要求跑通 + 数值可复现，
不要求提升能力。判据写死见 P6_PLAN.md：门 0 数据路径 / 门 1 前向逐位 / 门 2 训练路径逐位 /
REPRO 零训练复现 / 门 3 全量重训逐位。

数据、切分、保尾截断、span loss、贪心 EM 全部逐字搬运自 stages/12_answer_only.py；
唯一改动 = 模型换成 cards.CardPipeline（输入卡/思维卡/输出卡）。
★ residual=False：cards/target.py:28-30 的 TargetCard 默认带外层残差，而 stages/09:337-339
  的 thought 是裸层 —— Lead 已 CPU 实测：residual=False 时三个 ckpt 的 max|Δlogits| = 0.0e+00，
  residual=True 时 14.08/11.91/9.63。

用法：
  python stages/65_textgen_cards.py --gates   # 门 0/1/2 + REPRO（分钟级、零训练）
  python stages/65_textgen_cards.py --gate3   # 门 3（~21min，必须 systemd-run）
只允许写本文件与 logs/、/tmp；nanoSeek 只读。
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

# ★S65：cards 框架（仓库根 = 本文件上一级）
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402

import hashlib  # noqa: E402
import struct  # noqa: E402

from cards import CardPipeline  # noqa: E402

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


def _arg_value(flag: str, default: str) -> str:
    a = sys.argv[1:]
    return a[a.index(flag) + 1] if flag in a and a.index(flag) + 1 < len(a) else default


# ★记录集变体：s12 = target `#### <答案>`（去 CoT，默认）· s11 = target 完整 CoT 回复
#   —— 复现门必须用**与被评 ckpt 同口径**的记录集，否则不可比（见 P6_PLAN.md §4.3）
VARIANT = _arg_value("--variant", "s12")
assert VARIANT in ("s11", "s12"), f"未知 variant {VARIANT!r}"

DATA = NS_ROOT + "/data/chinese/clean_v3/gsm8k_cot_dialogue.txt"
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
CKPT_TMPL = "/home/vesita/coding/my/flowme/logs/12_ckpt_seed{seed}.pt"

PROBE = os.environ.get("S12_PROBE") == "1"
t_start = time.time()

print(f"[S12] ★A臂 只改一个变量=输出形态：target = `#### <答案>`+EOS（去 CoT）；"
      f"prompt=题干 与 S11 逐字相同；MAXLEN={MAXLEN} 截断优先级=prompt优先(MIN_T={MIN_T})。"
      f" 其余不动：steps={STEPS} batch={BATCH} lr={LR} d={D} ff={FF} seeds={SEEDS} 切分=seed42(9:1)。"
      f" B臂=引用 S11 不重跑；生成主口径=batch={EM_BATCH}", flush=True)
print(f"[S65] ★variant={VARIANT}：" + (
    "target=`#### <答案>`+EOS（S12 口径，复现 12_ckpt）" if VARIANT == "s12"
    else "target=完整 CoT 回复+EOS（S11 口径，复现 11_ckpt）"), flush=True)

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
    # ★s11 变体：目标 = 完整 CoT 回复 + <eos>（= stages/11 的口径，供复现 11_ckpt 用）。
    if VARIANT == "s11":
        target_text = r + EOS
    elif "####" in r:
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
_tgt_label = ("target=`#### <答案>`+EOS（S12 口径）" if VARIANT == "s12"
              else "target=完整 CoT 回复+EOS（S11 口径）")
print(f"[TGT] ★{_tgt_label} 目标长度（未截断）: 中位={_tl[len(_tl)//2][0]}tok "
      f"min={_tl[0][0]} max={_tl[-1][0]} 分布(长度:条数,前8)={dict(_tl[:8])}", flush=True)
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


def card_param_groups(model) -> dict:
    """同一形状的参数分组（Cards.cards() / CardPipeline 三张卡）。"""
    if isinstance(model, CardPipeline):
        return {n: list(model.card(n).parameters()) for n in model.CARD_NAMES}
    return model.cards()


G3_MODEL = os.environ.get("S65_MODEL", "cards")   # cards | stages（stages = A/A 地板，原路径）


def make_model(identity: bool):
    """模型类别由 `S65_MODEL` 选：cards = `CardPipeline`（residual=False）；stages = `Cards` 原类。

    ★residual=False 才与 stages/09 的裸 thought 逐位等价（见 `NL_CARDS_REF.md` §2）。
    """
    if identity:
        raise NotImplementedError("恒等臂未接 cards（门 B 本单元不跑）")
    if G3_MODEL == "stages":
        return Cards().to(device)
    return CardPipeline(d=D, vocab_size=V, ff=FF, nhead=NHEAD, maxlen=MAXLEN,
                        pad_id=PAD_ID, dropout=0.1, residual=False).to(device)


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
    model = make_model(identity)
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
                for k, ps in card_param_groups(model).items()
            }
            tot = {k: len(ps) for k, ps in card_param_groups(model).items()}
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




# ================= ★S65：把 S09–S12 的文本生成闭环接到 cards/ 的 CardPipeline =================
# 判据写死见 P6_PLAN.md；数据/切分/span loss/贪心 EM 全部逐字搬运自 stages/12_answer_only.py。
S65_LOG = os.path.join(ROOT, "logs")
S65_CKPT = os.path.join(S65_LOG, "65_ckpt_s12_seed1234.pt")
S12_CKPT_TMPL = os.path.join(S65_LOG, "12_ckpt_seed{seed}.pt")
REF_FLOOR_EM = 0.0189      # ★引用（不是本单元实测）：logs/12_answer.log 的门 C 地板
REF_S12_EM = 2.17          # ★引用：logs/12_answer.log 的 A 臂全量严格 EM（seed1234）
REF_S11_EM = 0.00          # ★引用：logs/11_textgen.log 的完整闭环全量严格 EM


def data_fp(recs) -> str:
    """对 (len(p),p...,len(t),t...) 求 sha256[:16]：数据路径一变动指纹就变。"""
    h = hashlib.sha256()
    for r in recs:
        for seq in (r["p"], r["t"]):
            h.update(struct.pack("<Q", len(seq)))
            if seq:
                h.update(struct.pack(f"<{len(seq)}i", *seq))
    return h.hexdigest()[:16]


def keys_to_cards(ck) -> dict:
    """stages `Cards` 的 state_dict 键 → CardPipeline 键（四组一一对应）。"""
    remap = {}
    for k, v in ck.items():
        if k == "emb.weight":
            nk = "input.emb.weight"
        elif k.startswith("in_enc."):
            nk = "input.enc." + k[len("in_enc."):]
        elif k.startswith("thought."):
            nk = "target.think." + k[len("thought."):]
        elif k.startswith("head."):
            nk = "output." + k
        else:
            raise KeyError(f"未知 ckpt 键 {k!r}")
        remap[nk] = v
    return remap


def load_cards_ckpt(model, path) -> tuple:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    r = model.load_state_dict(keys_to_cards(ck), strict=False)
    return len(r.missing_keys), len(r.unexpected_keys)


def new_pipe() -> CardPipeline:
    """★residual=False：cards/target.py:28-30 默认带外层残差，S09 的 thought 是裸层。"""
    return CardPipeline(d=D, vocab_size=V, ff=FF, nhead=NHEAD, maxlen=MAXLEN,
                        pad_id=PAD_ID, dropout=0.1, residual=False).to(device)


def weight_delta_vs_s12(model, path) -> tuple:
    """逐张量比对（R43：不许用文件 sha）。"""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = model.state_dict()
    worst, wname = 0.0, ""
    for k, v in ck.items():
        nk = next(iter(keys_to_cards({k: v})))
        key = nk if nk in sd else k        # stages `Cards` 模型：键名与 ckpt 相同
        d = (v.to(sd[key].device) - sd[key]).abs().max().item()
        if d > worst:
            worst, wname = d, k
    return worst, wname


# ---------------- 门 0：数据路径指纹 ----------------
def gate0():
    print(f"[G0] train_fp={data_fp(train)} test_fp={data_fp(test)} "
          f"n_train={len(train)} n_test={len(test)}", flush=True)


# ---------------- 门 1：前向逐位（零训练，灌既有 ckpt） ----------------
def gate1():
    ids_batch, _ = build_batch(test[:8])
    for tag, fn in (("s09c", "09c_ckpt_seed1234.pt"), ("s11", "11_ckpt_seed1234.pt"),
                    ("s12", "12_ckpt_seed1234.pt")):
        path = os.path.join(S65_LOG, fn)
        if not os.path.exists(path):
            print(f"[G1] {tag} 缺 ckpt {path} ⇒ 跳过", flush=True)
            continue
        ref = Cards().to(device)
        ref.load_state_dict(torch.load(path, map_location="cpu", weights_only=False))
        ref.eval()                                          # R39
        pipe = new_pipe()
        miss, unexp = load_cards_ckpt(pipe, path)
        pipe.eval()
        with torch.no_grad():
            a = ref.logits(ids_batch)
            b = pipe.logits(ids_batch)
        d = (a - b).abs().max().item()
        ok = (d == 0.0 and miss == 0 and unexp == 0)
        print(f"[G1] {tag} max|Δlogits|={d:.3e} missing={miss} unexpected={unexp} "
              f"n={ids_batch.size(1)} ⇒ {'PASS' if ok else 'FAIL'}", flush=True)
        del ref, pipe
        torch.cuda.empty_cache()


# ---------------- 门 2：训练路径逐位（同 seed、同 200 步） ----------------
def _rng_states() -> tuple:
    return torch.get_rng_state(), torch.cuda.get_rng_state()


def _rng_same(a: tuple, b: tuple) -> tuple:
    return bool(torch.equal(a[0], b[0])), bool(torch.equal(a[1], b[1]))


def _train_probe(model, n: int, snap_at=()) -> dict:
    """与 train_run 逐字同构的 n 步循环（数据序 = Generator(1234)）；返回末步 loss + 快照。

    快照存的是解码后的 state_dict（detach().clone()），键名按各自模型的命名。
    """
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(1234))
    pos, loss, snaps = 0, None, {}
    for step in range(1, n + 1):
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))     # ★重洗吃全局流（R34 的坑）
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(model, [train[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if step in snap_at:
            snaps[step] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    return dict(loss=loss.item(), snaps=snaps)


def gate2_align(ref, pipe, n_steps: int) -> None:
    """★S51-CHK2 同款：构造期全局流 / dropout 流消耗 / 数据序 —— 先对齐流，再谈逐位（R34）。"""
    ids_batch, _ = build_batch(train[:8])
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    _ = Cards().to(device)
    s_a = _rng_states()
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)
    _ = new_pipe()
    s_b = _rng_states()
    c, g = _rng_same(s_a, s_b)
    print(f"[G2-RNG1] 构造后全局流 ref-vs-pipe：CPU 逐位同={c} | CUDA 逐位同={g}", flush=True)

    after_ref = None
    for tag, m in (("ref", ref), ("pipe", pipe)):
        m.train()
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        before = _rng_states()
        with torch.no_grad():
            _ = m.logits(ids_batch)
        after = _rng_states()
        print(f"[G2-RNG2] {tag} train 前向：CPU 流被消耗={not torch.equal(after[0], before[0])} "
              f"CUDA 流被消耗={not torch.equal(after[1], before[1])}", flush=True)
        if tag == "ref":
            after_ref = after
        else:
            c2, g2 = _rng_same(after_ref, after)
            print(f"[G2-RNG2] 消耗后 ref-vs-pipe：CPU 逐位同={c2} | CUDA 逐位同={g2}"
                  f"（CUDA 同 ⇒ 同一批 dropout 掩码）", flush=True)

    n_idx = n_steps * BATCH
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(1234))
    reshuffles = "不触发重洗（数据序纯由 seed 决定）" if n_idx <= len(train) \
        else f"★触发重洗（{n_idx} > {len(train)}）⇒ 数据序依赖全局流"
    fp = hashlib.sha256(struct.pack(f"<{min(n_idx, len(train))}i",
                                    *order[:min(n_idx, len(train))].tolist())).hexdigest()[:16]
    print(f"[G2-RNG3] {n_steps} 步 = {n_idx} 样本 / len(train)={len(train)} ⇒ {reshuffles} | "
          f"前段索引 sha256[:16]={fp}", flush=True)


def _delta_same(a: dict, b: dict) -> tuple:
    """同键名逐张量 max|Δ|（A/A 对照用）。"""
    worst, wname = 0.0, ""
    for k, v in a.items():
        d = (v - b[k]).abs().max().item()
        if d > worst:
            worst, wname = d, k
    return worst, wname


def _delta_cards_pipe(a: dict, b: dict) -> tuple:
    """Cards 键 vs CardPipeline 键逐张量 max|Δ|（映射见 keys_to_cards）。"""
    worst, wname = 0.0, ""
    for k, v in a.items():
        nk = next(iter(keys_to_cards({k: v})))
        d = (v - b[nk]).abs().max().item()
        if d > worst:
            worst, wname = d, k
    return worst, wname


def gate2(n_steps: int = 200) -> float:
    """★按 S51 的顺序：先流结构对账（R34）→ 再 A/A（仪器）→ 最后才比 cards vs pipeline。"""
    def build(kind):
        torch.manual_seed(1234)
        torch.cuda.manual_seed_all(1234)
        return Cards().to(device) if kind == "cards" else new_pipe()

    ref_a, ref_b, pipe = build("cards"), build("cards"), build("pipe")
    gate2_align(ref_a, pipe, n_steps)
    w_aa0, _ = _delta_same(ref_a.state_dict(), ref_b.state_dict())
    w_ab0, n0 = _delta_cards_pipe(ref_a.state_dict(), pipe.state_dict())
    print(f"[G2-INIT] A/A max|Δθ|={w_aa0:.3e} | 框架 max|Δθ|={w_ab0:.3e} (worst={n0}) ⇒ "
          f"{'PASS' if (w_aa0 == 0.0 and w_ab0 == 0.0) else 'FAIL'}", flush=True)

    SNAP = (1, n_steps)
    ra = _train_probe(ref_a, n_steps, SNAP)
    rb = _train_probe(ref_b, n_steps, SNAP)
    rp = _train_probe(pipe, n_steps, SNAP)
    out = {}
    for step in SNAP:
        aa, _ = _delta_same(ra["snaps"][step], rb["snaps"][step])
        ab, nab = _delta_cards_pipe(ra["snaps"][step], rp["snaps"][step])
        out[step] = (aa, ab)
        print(f"[G2-{step}step] A/A max|Δθ|={aa:.3e} | 框架 Cards-vs-Pipeline "
              f"max|Δθ|={ab:.3e} (worst={nab}) | loss refA={ra['loss']:.6f} "
              f"refB={rb['loss']:.6f} pipe={rp['loss']:.6f}", flush=True)
    aa_n, ab_n = out[n_steps]
    if aa_n == 0.0 and ab_n == 0.0:
        verdict = "两条路径逐位等价（A/A=0 且框架=0）"
    elif aa_n == 0.0:
        verdict = "★框架真差异：A/A=0（训练可逐位重复）但 cards≠pipeline ⇒ 必须定位"
    else:
        verdict = "训练本身非确定（A/A≠0）⇒ 逐位不可验证，只能报统计判据"
    print(f"[G2-VERDICT] A/A={aa_n:.3e} 框架={ab_n:.3e} ⇒ {verdict}", flush=True)
    del ref_a, ref_b, pipe
    torch.cuda.empty_cache()
    return aa_n, ab_n


# ---------------- 复现门：零训练，既有 ckpt 走 cards 路径 ----------------
def repro():
    """★只评「记录集口径匹配」的 ckpt：s12 记录集 ⇒ 12_ckpt（answer-only）；
    s11 记录集 ⇒ 11_ckpt（CoT）。拿 CoT 模型评 answer-only 记录集不可比（P6_PLAN.md §4.3）。
    """
    plan = (("12", REF_S12_EM, SEEDS), ("11", REF_S11_EM, SEEDS))
    for tag, ref_em, seeds in plan:
        want = "s12" if tag == "12" else "s11"
        if VARIANT != want:
            print(f"[REPRO] {tag}_ckpt 需要 variant={want} 的记录集（当前 {VARIANT}）"
                  f"⇒ 跳过，避免口径不可比", flush=True)
            continue
        for seed in seeds:
            path = os.path.join(S65_LOG, f"{tag}_ckpt_seed{seed}.pt")
            if not os.path.exists(path):
                print(f"[REPRO] {tag} seed={seed} 缺 ckpt ⇒ 跳过", flush=True)
                continue
            pipe = new_pipe()
            miss, unexp = load_cards_ckpt(pipe, path)
            pipe.eval()
            ce = eval_ce(pipe, test)
            cm, cse = mean_se(ce)
            txt = greedy_gen(pipe, test, batch=EM_BATCH, mode="correct")
            hits = hits_from(txt, test)
            em = 100.0 * sum(h["strict"] for h in hits) / len(hits)
            ok = abs(em - ref_em) <= 0.13
            print(f"[REPRO] {tag}_ckpt_seed{seed} 配对CE={cm:.4f}±{cse:.4f} EM={em:.2f}% "
                  f"(n={len(hits)}) vs 日志 {ref_em:.2f}% Δ={em - ref_em:+.2f}pp "
                  f"missing={miss} unexpected={unexp} ⇒ {'PASS' if ok else 'FAIL'}",
                  flush=True)
            em_buckets(f"REPRO {tag}_ckpt_seed{seed}", hits)
            del pipe
            torch.cuda.empty_cache()


# ---------------- 门 3：cards 路径全量重训（3000 步） ----------------
def gate3():
    model, gate_a_res, wall = train_run(1234, False, BATCH, gate_a=True)
    if gate_a_res is None or not all(v > 0 for v in gate_a_res.values()):
        print("[G3-GATE-A] 不过 ⇒ 梯度没到卡，停。", flush=True)
        sys.exit(4)
    model.eval()
    ce = eval_ce(model, test)
    cm, cse = mean_se(ce)
    txt = greedy_gen(model, test, batch=EM_BATCH, mode="correct")
    hits = hits_from(txt, test)
    em = 100.0 * sum(h["strict"] for h in hits) / len(hits)
    print(f"[G3-MAIN] {G3_MODEL}路径 seed=1234 配对CE={cm:.4f}±{cse:.4f}", flush=True)
    em_buckets(f"G3 {G3_MODEL}路径 seed=1234", hits)
    ckpt_out = (S65_CKPT if G3_MODEL == "cards"
                else S65_CKPT.replace(".pt", "_aa_stages.pt"))
    print(f"[G3-MODEL] 模型类别={G3_MODEL}（cards=CardPipeline / stages=Cards 原类）| "
          f"ckpt={ckpt_out}", flush=True)
    torch.save(model.state_dict(), ckpt_out)
    print(f"[G3-CKPT] saved {ckpt_out}", flush=True)
    path12 = S12_CKPT_TMPL.format(seed=1234)
    if os.path.exists(path12):
        worst, wname = weight_delta_vs_s12(model, path12)
        print(f"[G3-CMP] vs {path12} 逐张量 max|Δθ|={worst:.3e} (worst={wname}) ⇒ "
              f"{'PASS 逐位一致' if worst == 0.0 else 'FAIL'}", flush=True)
    print(f"[G3-EM] {G3_MODEL}路径 EM={em:.2f}% (n={len(hits)}) vs S12 日志 {REF_S12_EM:.2f}% "
          f"Δ={em - REF_S12_EM:+.2f}pp ⇒ "
          f"{'PASS' if abs(em - REF_S12_EM) <= 0.13 else 'FAIL'}", flush=True)
    print(f"[G3-META] 模型类别={G3_MODEL} 墙钟={wall:.0f}s steps={STEPS} batch={BATCH} "
          f"lr={LR} d={D} ff={FF} MAXLEN={MAXLEN} gate_a={gate_a_res}", flush=True)


# ---------------- 入口 ----------------
if __name__ == "__main__":
    ARGS = sys.argv[1:]
    MODE = "gates"
    if "--gate3" in ARGS:
        MODE = "gate3"
    elif "--gate2" in ARGS:
        MODE = "gate2"
    elif "--repro" in ARGS:
        MODE = "repro"
    if MODE == "repro":
        gate0()
        repro()
        print("[S65-VERDICT] 复现门完成", flush=True)
        print("[DONE] exit=0", flush=True)
        sys.exit(0)
    gate0()
    gate1()
    aa2, ab2 = gate2(200)
    # ★门 2 判据 v3（留痕见 P6_PLAN.md §4.1）：v1「逐位」不可达；v2「单次 框架 ≤ A/A」是两次含噪估计
    #   的逐次比较，会随机翻面 —— 实测四次 A/A ∈ {2.902,1.850,1.654,1.525}e-3、
    #   框架 ∈ {1.225,2.024,1.511,1.816}e-3，两组重叠 ⇒ 只能判"同量级"。
    AA_HIST = (1.525e-03, 2.902e-03)     # 历史 A/A 观测区间（本机、本设置）
    g2_ok = ab2 <= AA_HIST[1] * 1.3
    print(f"[S65-G2JUDGE] 框架差={ab2:.3e} vs 历史 A/A 区间=[{AA_HIST[0]:.3e},{AA_HIST[1]:.3e}] "
          f"⇒ {'PASS（同量级，未测出框架差异）' if g2_ok else 'FAIL（超出区间上限 30%）'}", flush=True)
    if MODE == "gate2":
        print(f"[S65-VERDICT] 门2 三跑对照完成（对齐→A/A→框架）| 框架 max|Δθ|={ab2:.3e} "
              f"（A/A={aa2:.3e}）", flush=True)
    elif MODE == "gates":
        repro()
        print(f"[S65-VERDICT] 门0/1/2 + REPRO 完成 | 门2 框架={ab2:.3e} vs A/A={aa2:.3e} ⇒ "
              f"{'PASS' if g2_ok else 'FAIL'}", flush=True)
    else:
        if not g2_ok:
            print("[S65] 门2 超出历史 A/A 区间 ⇒ 记录但不阻断门 3（门 3 自有 EM/CE 判据）",
                  flush=True)
        gate3()
    print("[DONE] exit=0", flush=True)
