#!/usr/bin/env python3
"""S28 · 独立复核：S25「只训一张 198K 的卡 ⇒ EM 比端到端高 43pp」是真的吗？

本文件是**独立复核者**写的，**不复用 S25 的评估代码**：
  · 数据：按 S14/S15 生成器规格**重写**，并用 S15 里已入库的指纹常量
    （答案域 1799 / 机会水平 0.2% / 进位率 0.87,0.43 / top5 = 706,1130,977,537,926）
    独立证明"我重建的 test 集 == S25 用的那个 test 集"；同时打印前 3 条题面+gold
    与整个 test 集的 sha256。
  · 模型：按 ckpt 的 state_dict 键自己重建（架构必须一致，否则载不进权重）。
  · 生成/EM：自己写的逐样本贪心解码（batch=1 by construction）+ 自己写的答案解析
    与数字每步正确率（exp(−数字 token CE)）定义。
  · S25 的 jsnol 只用来**取它对外的报告值**做逐位比对，不参与任何计算。

模式（env）：
  S28_MODE=eval   零训练复核（默认）
  S28_MODE=train  等算力对照：臂1b 端到端 6000 步 ×2 seed（GPU）；S28_EXTRA=1 时
                  额外重训臂2 并每 750 步评一次 ⇒ 给"训练末 vs 最佳"
  S28_DEV=cpu|cuda   默认 cpu
  S28_ONLY=e2e,anchor,...   只评若干臂
  S28_SEEDS=1234,5678
  S28_LIMIT=N   只评 test 前 N 条（冒烟计时用）

只允许写：本文件、logs/、/tmp。
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

MODE = os.environ.get("S28_MODE", "eval")
REQ_DEV = os.environ.get("S28_DEV", "cpu")
SEEDS = tuple(int(x) for x in os.environ.get("S28_SEEDS", "1234,5678").split(","))
ONLY = tuple(x for x in os.environ.get("S28_ONLY", "").split(",") if x)
LIMIT = int(os.environ.get("S28_LIMIT", "0")) or None
NTHREADS = int(os.environ.get("S28_THREADS", "4"))
EXTRA = os.environ.get("S28_EXTRA") == "1"
TRAIN_STEPS = int(os.environ.get("S28_TRAIN_STEPS", "6000"))

torch.set_num_threads(NTHREADS)

if REQ_DEV == "cuda":
    dev = "cuda" if torch.cuda.is_available() else "cpu"
else:
    dev = "cpu"
print(f"[S28] mode={MODE} req_dev={REQ_DEV} dev={dev} threads={NTHREADS} "
      f"seeds={SEEDS} only={ONLY or 'all'} limit={LIMIT} extra={EXTRA}", flush=True)
if dev == "cuda":
    print(f"[S28] GPU={torch.cuda.get_device_name(0)}", flush=True)

# ---------------- 配置：必须与 S25/S19-res 逐字一致（否则 ckpt 无意义） -------------
D, FF, NHEAD, MAXLEN = 128, 512, 4, 512
MIN_T, TAIL_KEEP = 48, 24
LR, BATCH, GEN_BATCH, EM_BATCH = 1e-3, 32, 32, 1
N_TRAIN, N_TEST, MAX_GEN = 4000, 800, 48
BUCKET, BI = "add_3d", 2
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"

tok = Tokenizer.from_file(TOK)
V = tok.get_vocab_size()
PAD_ID = tok.token_to_id("<pad>")
EOS_ID = tok.token_to_id("<eos>")


def enc(t: str) -> list[int]:
    return tok.encode(t, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


# ============================================================================
# ① 数据：独立重建（规格来自 S14/S15），并用 S15 已入库指纹自证同一性
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
    assert len({t for t, _ in ts}) == len(ts)
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


def carry_rate(name, n, rng):
    lo, hi = LOHI[name]
    any_c = unit_c = 0
    for _ in range(n):
        y = rng.randint(2 * lo, 2 * hi)
        a, b = split_by_answer(name, y, rng)
        w = max(len(str(a)), len(str(b)))
        sa, sb = str(a).zfill(w), str(b).zfill(w)
        c, cp = 0, []
        for i in range(w - 1, -1, -1):
            s = int(sa[i]) + int(sb[i]) + c
            if s >= 10:
                cp.append(w - 1 - i)
            c = s // 10
        any_c += bool(cp)
        unit_c += (0 in cp)
    return any_c / n, unit_c / n


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
            raise RuntimeError(f"{name}{tag} 文本空间不足")
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


_t0 = time.time()
_tr_rng, _te_rng = Rng(14000 + BI * 7), Rng(14900 + BI * 7)
_used: set[str] = set()
TRAIN = [encode_record(x) for x in gen_half(BUCKET, N_TRAIN, _tr_rng, _used, "train")]
TEST = [encode_record(x) for x in gen_half(BUCKET, N_TEST, _te_rng, _used, "test")]
assert len(_used) == len(TRAIN) + len(TEST)

_ys = answer_values(BUCKET)
_ctr_tr = Counter(r["y"] for r in TRAIN)
_top5 = [v for v, _ in Counter(r["gold"] for r in TRAIN).most_common(5)]
_prior = sum(1 for r in TEST if r["gold"] in _top5) / len(TEST)
_anyc, _unitc = carry_rate(BUCKET, 300, Rng(77 + BI))
_tpl_used = len({r["nl"] for r in TRAIN + TEST})
_tgt_len = sorted(len(r["t_full"]) for r in TRAIN)
_cut = sum(1 for r in TRAIN + TEST if r["cut_p"] or r["cut_t"])

print(f"[S28-DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)} 文本零重叠 | "
      f"答案域={min(_ys)}..{max(_ys)}({len(_ys)}个) 计数train min/max="
      f"{min(_ctr_tr.values())}/{max(_ctr_tr.values())} | 模板覆盖={_tpl_used}/{len(TEMPLATES)}句 | "
      f"截断={_cut} | target中位={_tgt_len[len(_tgt_len)//2]}tok | "
      f"进位率={_anyc:.2f}/{_unitc:.2f} | ★最高频5答案={_top5} 机会水平={100.0*_prior:.1f}%",
      flush=True)

# —— 与 S15 已入库指纹逐位比对（证明"我重建的 == 它们用的那个数据集"）——
_S14_FP = (1799, "0.2", "0.87", "0.43", ['706', '1130', '977', '537', '926'])
_got = (len(_ys), f"{100*_prior:.1f}", f"{_anyc:.2f}", f"{_unitc:.2f}", _top5)
_fp_ok = (_got == _S14_FP) if LIMIT is None else (len(_ys) == 1799)
print(f"[S28-DATA-CHECK] 我的重建 vs S15 入库指纹 {_S14_FP} ⇒ 实测{_got} ⇒ "
      f"{'逐位一致 ✓（同一生成器同一种子 ⇒ 同一个 test 集）' if _fp_ok else '★不一致 ⇒ 数据集不是同一个，复核无效'}"
      + ("" if LIMIT is None else "（LIMIT 模式只比答案域）"), flush=True)

# test 集整体指纹 + 前 3 条题面/gold（口径同一性的可复现凭据）
_h = hashlib.sha256()
for r in TEST:
    _h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
TEST_SHA = _h.hexdigest()[:16]
print(f"[S28-DATA] TEST 集指纹 sha256[:16]={TEST_SHA}  n={len(TEST)}", flush=True)
for i in range(3):
    r = TEST[i]
    print(f"[S28-DATA] test[{i}] 题面={r['nl']!r} gold={r['gold']!r} (a={r['a']} b={r['b']} "
          f"prompt_tok={len(r['p'])} target_tok={len(r['t'])})", flush=True)
_tp = {(r["a"], r["b"]) for r in TRAIN}
_ov = sum(1 for r in TEST if (r["a"], r["b"]) in _tp)
_txt_ov = len({r["nl"] + "||" + r["gold"] for r in TRAIN} & {r["nl"] + "||" + r["gold"]
                                                              for r in TEST})
print(f"[S28-LEAK] train∩test: 整串重叠={_txt_ov} 条 | (a,b) 数对重叠={_ov}/{len(TEST)} 条 "
      f"（题面不同但数字对撞车，属于「同题不同模板」）", flush=True)
assert _txt_ov == 0, "train/test 有整串重叠 ⇒ 泄漏"
print(f"[S28] 数据重建耗时 {time.time()-_t0:.1f}s", flush=True)


# ============================================================================
# ② 模型（按 ckpt 键重建；架构必须一致否则载不入权重）
# ============================================================================
def sin_pe(max_len, d):
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def mk_layer():
    return nn.TransformerEncoderLayer(D, NHEAD, FF, dropout=0.1, activation="gelu",
                                      batch_first=True, norm_first=True)


class RepCards(nn.Module):
    """h = h + Think(h)（S19-res / S25 逐字架构）。"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.in_enc = mk_layer()
        self.thought = mk_layer()
        self.head = nn.Linear(D, V)
        self.register_buffer("pe", sin_pe(MAXLEN, D), persistent=False)

    def forward(self, ids):
        n = ids.size(1)
        m = torch.full((n, n), float("-inf"), device=ids.device)
        m = torch.triu(m, diagonal=1).unsqueeze(0).expand(ids.size(0), -1, -1).clone()
        m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
        m = m.repeat_interleave(NHEAD, dim=0)
        x = self.emb(ids) + self.pe[:n].unsqueeze(0).to(ids.device)
        h = self.in_enc(x, src_mask=m)
        h = h + self.thought(h, src_mask=m)
        return self.head(h)


def group_of(name):
    if name.startswith("emb.") or name.startswith("in_enc."):
        return "输入卡"
    if name.startswith("thought."):
        return "目标卡"
    if name.startswith("head."):
        return "输出卡"
    raise KeyError(name)


def load(ckpt):
    m = RepCards()
    sd = torch.load(ckpt, map_location="cpu")
    m.load_state_dict(sd)
    return m.to(dev).eval()


# ============================================================================
# ③ 我自己的评估路径
# ============================================================================
@torch.no_grad()
def gen_bs1(model, recs):
    """逐样本贪心（batch=1 by construction，与 S25 主口径同）。"""
    model.eval()                      # ★必须 eval：否则 dropout=0.1 打开，EM 会被砸低
    outs = []
    for r in recs:
        ids = torch.tensor([r["p"]], dtype=torch.long, device=dev)
        cap = min(MAXLEN - len(r["p"]), MAX_GEN)
        got = []
        for _ in range(cap):
            nxt = int(model(ids)[0, -1].argmax().item())
            if nxt == EOS_ID:
                break
            got.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=dev)], 1)
        outs.append(dec(got))
    return outs


@torch.no_grad()
def gen_batchN(model, recs, bs):
    """batch=N 贪心（自检用）。注意：本模型无 padding-aware 位置编码，
    混长度 batch 会把「生成位置」从 lens 推到 n0（右边补 PAD），PE 随 batch 组成而变，
    所以与 batch=1 不逐位等价 —— 这正是 R28 的现象。"""
    model.eval()
    outs = [""] * len(recs)
    for i in range(0, len(recs), bs):
        ch = recs[i: i + bs]
        lens = [len(r["p"]) for r in ch]
        n0 = max(lens)
        ids = torch.full((len(ch), n0), PAD_ID, dtype=torch.long, device=dev)
        for j, r in enumerate(ch):
            ids[j, :lens[j]] = torch.tensor(r["p"], dtype=torch.long, device=dev)
        cap = [min(MAXLEN - L, MAX_GEN) for L in lens]
        gen = [[] for _ in ch]
        done = [False] * len(ch)
        last_col = [L - 1 for L in lens]
        for _ in range(max(cap)):
            if all(done):
                break
            logits = model(ids)
            nxt = []
            for j in range(len(ch)):
                if done[j]:
                    nxt.append(PAD_ID)
                    continue
                t = int(logits[j, last_col[j]].argmax().item())
                gen[j].append(t)
                nxt.append(t)
                if t == EOS_ID or len(gen[j]) >= cap[j]:
                    done[j] = True
                last_col[j] = n0 + len(gen[j]) - 1
            ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long,
                                               device=dev).unsqueeze(1)], 1)
        for j in range(len(ch)):
            outs[i + j] = dec([t for t in gen[j] if t != EOS_ID])
    return outs


def my_parse(text):
    if "####" not in text:
        return None
    seg = text.rsplit("####", 1)[1].split("\n")[0]
    seg = seg.strip().replace(",", "").replace(" ", "").replace("$", "")
    return seg or None


NUM_CHARS = set("0123456789+-*/=%$")


def digit_token(tid) -> bool:
    s = tok.id_to_token(int(tid)) or ""
    if s.startswith("<") and s.endswith(">"):
        return False
    return bool(s) and all(c in NUM_CHARS for c in s)


@torch.no_grad()
def ce_and_acc(model, recs):
    """自己写的 CE：整体样本级 CE + 数字每步正确率 = exp(−数字 per-sample CE 均值)。"""
    model.eval()
    samp_all, dig_per = [], []
    for i in range(0, len(recs), GEN_BATCH):
        ch = recs[i: i + GEN_BATCH]
        lens = [len(r["p"]) + len(r["t"]) for r in ch]
        n = max(lens)
        ids = torch.full((len(ch), n), PAD_ID, dtype=torch.long, device=dev)
        for j, r in enumerate(ch):
            ids[j, :len(r["p"])] = torch.tensor(r["p"], dtype=torch.long, device=dev)
            ids[j, len(r["p"]):len(r["p"]) + len(r["t"])] = torch.tensor(
                r["t"], dtype=torch.long, device=dev)
        logp = torch.log_softmax(model(ids), dim=-1)
        for j, r in enumerate(ch):
            s = len(r["p"])
            lp = logp[j, s - 1: s + len(r["t"]) - 1]
            tgt = torch.tensor(r["t"], dtype=torch.long, device=dev)
            ces = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1).tolist()
            samp_all.append(sum(ces) / len(ces))
            d = [c for c, tk in zip(ces, r["t"]) if digit_token(tk)]
            if d:
                dig_per.append(sum(d) / len(d))
    model.train()
    ce = sum(samp_all) / len(samp_all)
    dm = sum(dig_per) / len(dig_per)
    return ce, math.exp(-dm), dm


def strict_of(texts, recs):
    return [int(my_parse(t) == r["gold"]) for t, r in zip(texts, recs)]


def binomial_se(xs):
    n = len(xs)
    s = sum(xs) / n
    return math.sqrt(s * (1 - s) / n)


# ============================================================================
# ④ S25 对外的报告值（只读它的 jsnol 取数，不参与计算）
# ============================================================================
S25 = {}
try:
    with open(LOGDIR + "/25_results.jsonl") as f:
        for line in f:
            r = json.loads(line)
            S25[(r["arm"], r["seed"])] = r
except FileNotFoundError:
    print("[S28] ★找不到 logs/25_results.jsonl", flush=True)

ARM_CKPT = {"e2e": "e2e", "anchor": "anchor", "randtgt": "randtgt",
            "randall": "randall", "alt": "alt"}
ARMS = ONLY or ("e2e", "anchor", "randtgt", "alt", "randall")
test_set = TEST if LIMIT is None else TEST[:LIMIT]


# ============================================================================
# ⑤ 零训练复核主流程
# ============================================================================
def run_eval():
    t_all = time.time()
    res = {}
    for arm in ARMS:
        for seed in SEEDS:
            ck = f"{LOGDIR}/25_ckpt_{ARM_CKPT[arm]}_seed{seed}.pt"
            if not os.path.exists(ck):
                print(f"[S28] ★缺 {ck}", flush=True)
                continue
            t0 = time.time()
            model = load(ck)
            txt = gen_bs1(model, test_set)
            st = strict_of(txt, test_set)
            em, se = sum(st) / len(st), binomial_se(st)
            ce, acc, dm = ce_and_acc(model, test_set)
            # 泄漏探针：训练集前 100 条（若 test 被见过，train EM 不会显著高于 test）
            tr_sub = TRAIN[:100]
            st_tr = strict_of(gen_bs1(model, tr_sub), tr_sub)
            em_tr = sum(st_tr) / len(st_tr)
            exp = S25.get((arm, seed))
            d_em = (em - exp["em"]) * 100 if exp else float("nan")
            d_ac = (acc - exp["acc"]) * 100 if exp else float("nan")
            res[(arm, seed)] = dict(em=em, em_se=se, acc=acc, ce=ce, dm=dm,
                                    strict=st, em_train=em_tr, sha=hashlib.sha256(
                                        open(ck, "rb").read()).hexdigest()[:16])
            print(f"[S28-EM] {arm:8s} seed={seed} n={len(st)} 我的EM={em*100:.2f}%±{se*100:.2f} | "
                  f"S25报={exp['em']*100:.2f}% Δ={d_em:+.2f}pp | 我的数字每步={acc*100:.2f}% "
                  f"S25报={exp['acc']*100:.2f}% Δ={d_ac:+.2f}pp | 我的CE={ce:.4f} "
                  f"S25报={exp['ce']:.4f} | ★train100-EM={em_tr*100:.1f}% | "
                  f"{time.time()-t0:.0f}s", flush=True)
            for i in range(3):
                print(f"    [{arm} s{seed} 样例{i}] 题面={test_set[i]['nl']!r} gold="
                      f"{test_set[i]['gold']!r} 我生成={txt[i]!r}", flush=True)
            del model

    # ---- R28：batch=1 vs batch=16（同一 ckpt、同一 16 条）----
    print("\n[S28-R28] batch=1 vs batch=16 逐字一致率（前 16 条 test）", flush=True)
    for arm in ARMS:
        for seed in SEEDS:
            ck = f"{LOGDIR}/25_ckpt_{ARM_CKPT[arm]}_seed{seed}.pt"
            if not os.path.exists(ck):
                continue
            m = load(ck)
            sub = TEST[:16]
            a = gen_bs1(m, sub)
            b = gen_batchN(m, sub, 16)
            same = sum(x == y for x, y in zip(a, b))
            # 判因实验：取 16 条【prompt 等长】的样本 ⇒ 无 PAD 空档 ⇒ 与 batch=1 应逐位一致
            bylen = defaultdict(list)
            for r in TEST:
                bylen[len(r["p"])].append(r)
            grp = max(bylen.values(), key=len)[:16]
            gs = len(grp)
            c = gen_bs1(m, grp)
            d = gen_batchN(m, grp, gs)
            same_eq = sum(x == y for x, y in zip(c, d))
            print(f"[S28-R28] {arm} seed={seed}: 混长度16条 一致 {same}/16 | "
                  f"等长(prompt_tok={len(grp[0]['p'])}){gs}条 一致 {same_eq}/{gs} ⇒ "
                  f"{'batch 效应来自「右边补 PAD 使生成位置(n0+gen)偏离 lens+gen，PE 随 batch 组成变化」' if same_eq == gs and same < 16 else '需再查'}"
                  f" | 例 batch1={a[0][:30]!r} batch16={b[0][:30]!r}", flush=True)
            del m

    # ---- 冻结验证：臂2 vs 臂1 逐参数 ----
    print("\n[S28-FREEZE] 臂2(anchor) vs 臂1(e2e) 逐参数 max|Δ|（应：输入/输出卡 == 0，目标卡 > 0）",
          flush=True)
    for seed in SEEDS:
        p1 = f"{LOGDIR}/25_ckpt_e2e_seed{seed}.pt"
        p2 = f"{LOGDIR}/25_ckpt_anchor_seed{seed}.pt"
        p3 = f"{LOGDIR}/25_ckpt_randtgt_seed{seed}.pt"
        if not (os.path.exists(p1) and os.path.exists(p2)):
            continue
        s1, s2 = torch.load(p1, map_location="cpu"), torch.load(p2, map_location="cpu")
        s3 = torch.load(p3, map_location="cpu") if os.path.exists(p3) else None
        agg = defaultdict(float)
        npar = defaultdict(int)
        for k in s1:
            g = group_of(k)
            d = float((s1[k].float() - s2[k].float()).abs().max())
            agg[g] = max(agg[g], d)
            npar[g] += s1[k].numel()
        line = " | ".join(f"{g}: max|Δ|={agg[g]:.3e}" for g in ("输入卡", "目标卡", "输出卡"))
        ok = agg["输入卡"] == 0.0 and agg["输出卡"] == 0.0 and agg["目标卡"] > 0.0
        print(f"[S28-FREEZE] seed={seed} 臂2 vs 臂1: {line} ⇒ "
              f"{'冻结成立 ✓（接口逐位不动、只有目标卡变）' if ok else '★冻结不成立'}", flush=True)
        if s3 is not None:
            a3 = defaultdict(float)
            for k in s1:
                a3[group_of(k)] = max(a3[group_of(k)],
                                      float((s1[k].float() - s3[k].float()).abs().max()))
            print(f"[S28-FREEZE] seed={seed} 臂3 vs 臂1: "
                  + " | ".join(f"{g}: max|Δ|={a3[g]:.3e}" for g in ("输入卡", "目标卡", "输出卡"))
                  + "（应只有目标卡不同）", flush=True)

    # ---- 臂1 是否逐位复现 S19-res 的入库 ckpt ----
    print("\n[S28-S19] 臂1(e2e) vs S19-res 入库 ckpt 逐参数 max|Δ|", flush=True)
    for seed in SEEDS:
        p1 = f"{LOGDIR}/25_ckpt_e2e_seed{seed}.pt"
        p19 = f"{LOGDIR}/19_ckpt_add_3d_res_seed{seed}.pt"
        if not os.path.exists(p19):
            print(f"[S28-S19] ★缺 {p19}", flush=True)
            continue
        s1, s2 = torch.load(p1, map_location="cpu"), torch.load(p19, map_location="cpu")
        mx = max(float((s1[k].float() - s2[k].float()).abs().max()) for k in s1)
        print(f"[S28-S19] seed={seed}: max|Δ|={mx:.3e} | 我的文件sha16 S25="
              f"{res[(('e2e'), seed)]['sha'] if ('e2e', seed) in res else '-'} vs S19="
              f"{hashlib.sha256(open(p19,'rb').read()).hexdigest()[:16]} ⇒ "
              f"{'逐位同一模型 ✓' if mx == 0 else '★不是同一个模型'}", flush=True)

    # ---- R29：臂2 − 臂1 的 Δ 非零比例（逐样本重算）----
    print("\n[S28-R29] 臂2 − 臂1 逐样本 Δ 非零比例 / 配对 Δ±SE", flush=True)
    for seed in SEEDS:
        if ("anchor", seed) not in res or ("e2e", seed) not in res:
            continue
        a = res[("anchor", seed)]["strict"]
        b = res[("e2e", seed)]["strict"]
        xs = [x - y for x, y in zip(a, b)]
        m = sum(xs) / len(xs)
        sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))
        nz = sum(1 for x in xs if x != 0) / len(xs)
        exp = S25.get(("anchor", seed))
        print(f"[S28-R29] seed={seed}: Δ={m*100:+.2f}±{sd/math.sqrt(len(xs))*100:.2f}pp | "
              f"Δ非零={nz*100:.1f}%（S25 未直接入库此值，见其日志）| 两份 strict 向量长度="
              f"{len(a)}/{len(b)}", flush=True)

    # ---- 一致性汇总（判据：逐位一致 ≤0.5pp）----
    print("\n[S28] ===== ★逐 ckpt × seed 对照表（S25 报 vs 我算） =====", flush=True)
    print("[S28] arm | seed | S25 EM | 我 EM | Δ | S25 数字每步 | 我 数字每步 | Δ | 判定", flush=True)
    worst = 0.0
    for arm in ARMS:
        for seed in SEEDS:
            if (arm, seed) not in res:
                continue
            r = res[(arm, seed)]
            e = S25.get((arm, seed))
            d1 = (r["em"] - e["em"]) * 100
            d2 = (r["acc"] - e["acc"]) * 100
            worst = max(worst, abs(d1), abs(d2))
            print(f"[S28] {arm} | {seed} | {e['em']*100:.2f}% | {r['em']*100:.2f}% | {d1:+.2f}pp | "
                  f"{e['acc']*100:.2f}% | {r['acc']*100:.2f}% | {d2:+.2f}pp | "
                  f"{'✓一致' if max(abs(d1),abs(d2)) <= 0.5 else '★不一致'}", flush=True)
    print(f"[S28] ★最大偏离={worst:.2f}pp ⇒ "
          f"{'复核通过（全部 ≤0.5pp）' if worst <= 0.5 else '★复核不通过，需定位'}", flush=True)

    # ---- 等算力对照的锚（供 train 模式比对）----
    with open("/tmp/28_eval_summary.json", "w") as f:
        json.dump({"test_sha": TEST_SHA,
                   "em": {f"{a}|{s}": res[(a, s)]["em"] for a in ARMS for s in SEEDS
                          if (a, s) in res},
                   "acc": {f"{a}|{s}": res[(a, s)]["acc"] for a in ARMS for s in SEEDS
                           if (a, s) in res},
                   "em_train": {f"{a}|{s}": res[(a, s)]["em_train"] for a in ARMS for s in SEEDS
                                if (a, s) in res}}, f, indent=1)
    print(f"[S28] eval 段墙钟={time.time()-t_all:.0f}s", flush=True)


# ============================================================================
# ⑥ 训练段：臂1b 端到端 6000 步（等算力对照）+ 可选臂2 轨迹
# ============================================================================
def build_batch(recs):
    lens = [len(r["p"]) + len(r["t"]) for r in recs]
    n = max(lens)
    ids = torch.full((len(recs), n), PAD_ID, dtype=torch.long)
    s = torch.zeros(len(recs), dtype=torch.long)
    for i, r in enumerate(recs):
        ids[i, :len(r["p"])] = torch.tensor(r["p"])
        ids[i, len(r["p"]):len(r["p"]) + len(r["t"])] = torch.tensor(r["t"])
        s[i] = len(r["p"])
    return ids.to(dev), s


def masked_ce(model, recs):
    ids, s = build_batch(recs)
    logp = torch.log_softmax(model(ids), dim=-1)
    nll, ntok = torch.zeros((), device=dev), 0
    for i, r in enumerate(recs):
        e = s[i] + len(r["t"])
        lp = logp[i, s[i] - 1: e - 1]
        tgt = ids[i, s[i]: e]
        nll = nll - lp.gather(1, tgt.unsqueeze(1)).sum()
        ntok += len(r["t"])
    return nll / ntok


def eval_quick(model, recs):
    model.eval()
    txt = gen_bs1(model, recs)
    st = strict_of(txt, recs)
    ce, acc, _ = ce_and_acc(model, recs)
    model.train()
    return sum(st) / len(st), acc, ce


def train_e2e(steps, seed, evals_at=()):
    """臂1b：与 S25 的 e2e 逐字同一代码路径，只是步数更多。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = RepCards().to(dev)
    for p in model.parameters():
        p.requires_grad_(True)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    order = torch.randperm(len(TRAIN), generator=torch.Generator().manual_seed(seed))
    pos = 0
    model.train()
    curve = []
    t0 = time.time()
    for k in range(1, steps + 1):
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(TRAIN))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(model, [TRAIN[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if k % 500 == 0 or k in evals_at:
            print(f"  [S28-train] e2e6000 seed={seed} step={k}/{steps} loss={loss.item():.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
        if k in evals_at:
            em, acc, ce = eval_quick(model, test_set)
            curve.append((k, em, acc, ce))
            print(f"  [S28-eval@] seed={seed} step={k} EM={em*100:.2f}% 数字每步={acc*100:.2f}% "
                  f"CE={ce:.4f}", flush=True)
    return model, curve, time.time() - t0


def train_frozen(seed, steps, evals_at=(), ckpt=None):
    """臂2 复跑：从臂1 ckpt 分叉，冻结输入/输出卡，只训目标卡。"""
    ckpt = ckpt or f"{LOGDIR}/25_ckpt_e2e_seed{seed}.pt"
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = RepCards()
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(dev)
    for n, p in model.named_parameters():
        p.requires_grad_(group_of(n) == "目标卡")
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    order = torch.randperm(len(TRAIN), generator=torch.Generator().manual_seed(seed))
    pos = 0
    model.train()
    curve = []
    t0 = time.time()
    for k in range(1, steps + 1):
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(TRAIN))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(model, [TRAIN[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if k in evals_at:
            em, acc, ce = eval_quick(model, test_set)
            curve.append((k, em, acc, ce))
            print(f"  [S28-eval@] frozen seed={seed} step={k} EM={em*100:.2f}% "
                  f"数字每步={acc*100:.2f}% CE={ce:.4f}", flush=True)
    return model, curve, time.time() - t0


def run_train():
    out = {}
    for seed in SEEDS:
        print(f"\n[S28-TRAIN] ===== 臂1b 端到端 {TRAIN_STEPS} 步 seed={seed}（等算力对照）=====",
              flush=True)
        model, curve, wall = train_e2e(TRAIN_STEPS, seed, evals_at=(3750, 4500, 5250, TRAIN_STEPS))
        ck = f"{LOGDIR}/28_ckpt_e2e{TRAIN_STEPS}_seed{seed}.pt"
        torch.save(model.state_dict(), ck)
        print(f"[S28-TRAIN] seed={seed} 墙钟={wall/60:.1f}min ckpt={ck}", flush=True)
        out[f"e2e{TRAIN_STEPS}|{seed}"] = dict(wall=wall, curve=curve, ckpt=ck)
        with open(f"{LOGDIR}/28_results.jsonl", "a") as f:
            f.write(json.dumps(dict(kind="e2e6000", seed=seed, steps=TRAIN_STEPS,
                                    wall=wall, curve=curve, ckpt=ck),
                               ensure_ascii=False) + "\n")
        del model
        if dev == "cuda":
            torch.cuda.empty_cache()
        if EXTRA:
            print(f"\n[S28-TRAIN] ===== 臂2 复跑（冻接口只训目标卡）seed={seed} "
                  f"每 750 步评一次 ===== ", flush=True)
            m2, c2, w2 = train_frozen(seed, 3000, evals_at=(750, 1500, 2250, 3000))
            best = max(c2, key=lambda x: x[1]) if c2 else None
            print(f"[S28-TRAIN] 臂2 seed={seed} 训练末 EM={c2[-1][1]*100:.2f}% "
                  f"最佳 EM={best[1]*100:.2f}%@step{best[0]}（差 "
                  f"{(c2[-1][1]-best[1])*100:+.2f}pp） 墙钟={w2/60:.1f}min", flush=True)
            out[f"frozen_traj|{seed}"] = dict(curve=c2, wall=w2,
                                              final=c2[-1][1], best=best[1], best_step=best[0])
            with open(f"{LOGDIR}/28_results.jsonl", "a") as f:
                f.write(json.dumps(dict(kind="frozen_traj", seed=seed, curve=c2,
                                        wall=w2), ensure_ascii=False) + "\n")
            del m2
            if dev == "cuda":
                torch.cuda.empty_cache()
    with open("/tmp/28_train_summary.json", "w") as f:
        json.dump(out, f, indent=1)
    print("\n[S28-TRAIN] 完成", flush=True)


if MODE == "train":
    run_train()
elif MODE == "trainset":
    # 泄漏探针加强版：在【全 4000 条训练集】上评 EM。若模型背下来了 ⇒ 训练集 EM 应 ≈100%；
    # 若训练集 EM 也远低于 100% ⇒ 模型根本没记住训练数据（= 欠训），test 的 EM 不可能是泄漏来的。
    for spec in (os.environ.get("S28_CKPT", "e2e|1234")).split(","):
        if "|" in spec:
            arm, sd = spec.split("|")
            ck = f"{LOGDIR}/25_ckpt_{ARM_CKPT[arm]}_seed{sd}.pt"
        else:
            ck = spec
        m = load(ck)
        t0 = time.time()
        for name, recs in (("全train4000", TRAIN), ("test800", TEST)):
            txt = gen_bs1(m, recs)
            st = strict_of(txt, recs)
            print(f"[S28-TRAINSET] {os.path.basename(ck)} {name}: EM={100*sum(st)/len(st):.2f}% "
                  f"(n={len(st)}) {time.time()-t0:.0f}s", flush=True)
        del m
else:
    run_eval()
print("[S28-DONE]", flush=True)
