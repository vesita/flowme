#!/usr/bin/env python3
"""S31 · 最便宜也最致命的一枪：泄漏到底影响了多少？SEEN/UNSEEN 拆解。

独立复核（零训练 · CPU · batch=1）。**不复用 S25/S28 的评估代码**：数据生成、模型、
贪心解码、答案解析、EM/SE、聚类 bootstrap 全部本文件自己写（规格来自 stages/14_synth_arith.py）。

四问：
  §0 泄漏率自查（两种口径）：test 条的 (a,b) 数对是否在 train 内；以及更严的 (a,b,y) 三元组。
  §1 按 SEEN/UNSEEN 分层报 EM ±SE（naive 按样本）与 n。
  §2 同桶配对 Δ=EM(SEEN)−EM(UNSEEN)（按 (a,b) 聚类配对 bootstrap）＋跨桶 (泄漏率,EM) 关系
     ＋ add_3d 的「无泄漏干净数字」。
  §3 聚类 bootstrap（重采样 (a,b) 数对，1000 次）的 SE vs 朴素按样本 SE。
  §4 logs/28_ckpt_e2e6000_seed{1234,5678}.pt 在与 S25 同口径（batch=1, n=800, add_3d）下补评。

只允许写：本文件、logs/31_seen_unseen.log、logs/31_results.jsonl、/tmp。
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

FLOWME = "/home/vesita/coding/my/flowme"
NS_ROOT = "/home/vesita/coding/my/nanoSeek"
LOGDIR = FLOWME + "/logs"
sys.path.insert(0, NS_ROOT)

# ---------------- CPU 纪律（本单元全程 CPU，零训练） ----------------
os.environ["CUDA_VISIBLE_DEVICES"] = ""
NTHREADS = int(os.environ.get("S31_THREADS", "4"))
LIMIT = int(os.environ.get("S31_LIMIT", "0")) or None
NBOOT = int(os.environ.get("S31_NBOOT", "1000"))
SEEDS = tuple(int(x) for x in os.environ.get("S31_SEEDS", "1234,5678").split(","))
ONLY = tuple(x for x in os.environ.get("S31_ONLY", "").split(",") if x)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

torch.set_num_threads(NTHREADS)
dev = "cpu"
if torch.cuda.is_available():
    print("★CPU 纪律被破坏：torch.cuda.is_available()=True，立即停。", flush=True)
    sys.exit(3)

D, FF, NHEAD, MAXLEN = 128, 512, 4, 512
MIN_T, TAIL_KEEP = 48, 24
N_TRAIN, N_TEST, MAX_GEN = 4000, 800, 48
BUCKETS = ("add_1d", "add_2d", "add_3d", "sub_2d", "mul_2d")  # ★顺序逐字同 S14（决定 RNG 种子）
BI = {b: i for i, b in enumerate(BUCKETS)}
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"

t_start = time.time()
print(f"[S31] device={dev} cuda_avail={torch.cuda.is_available()} CUDA_VISIBLE_DEVICES="
      f"{os.environ.get('CUDA_VISIBLE_DEVICES')!r} threads={NTHREADS} seeds={SEEDS} "
      f"limit={LIMIT} nboot={NBOOT} only={ONLY or 'all'}", flush=True)

tok = Tokenizer.from_file(TOK)
V = tok.get_vocab_size()
PAD_ID = tok.token_to_id("<pad>")
EOS_ID = tok.token_to_id("<eos>")
print(f"[S31] tokenizer V={V} PAD={PAD_ID} EOS={EOS_ID}", flush=True)


def enc(t: str) -> list[int]:
    return tok.encode(t, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


# ============================================================================
# §0-a 自己的数据生成（规格逐字来自 stages/14_synth_arith.py；种子同 S14 ⇒ 同数据）
# ============================================================================
def build_templates() -> list[tuple[str, bool]]:
    ts: list[tuple[str, bool]] = []
    tails_cn = ["等于多少？", "是多少？", "等于几？", "得多少？", "的结果是多少？",
                "的结果是几？", "的值是多少？", "等于多少", "是多少", "是几"]
    # ★逐字复刻 S14 build_templates（含它的一个 bug）：S14 里模板由 f-string 构造，
    #   包装式写成 f"...{{{{expr}}}}..." ⇒ 构造后是 "{expr}"（会被替换）；
    #   裸式写成 "{{expr}} " + t ⇒ 构造后仍是 "{{expr}}" ⇒ 那次 str.format 只把 "{{"
    #   折成字面 "{"、**不做替换**（裸式 10 句的已知 bug，必须一起复刻，否则 prompt
    #   字符串与 S14 数据不同、ckpt 评估无效）。这里用 <W>/<B> 标记模拟「构造后」的两种状态。
    for lead in ("", "请", "帮我", "麻烦"):
        for verb in ("计算", "算出", "算一下", "口算", "快速算", "算"):
            for t in tails_cn:
                ts.append((f"{lead}{verb} " + "<W>" + f" {t}", True))
    for t in tails_cn:
        ts.append(("<B> " + t, True))
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
                ts.append((f"{lead}{v} " + "<W>" + f"{tail}", False))
    assert len({t for t, _ in ts}) == len(ts), "模板有重复"
    return ts


TEMPLATES = build_templates()
# ★S14 的模板由 f-string 构造：f"...{{expr}}..." 在构造期把 "{{" 折成 "{..."。裸式那 10 句
#   写成 "{{expr}} " + t（双括号），f-string 后仍是 "{{expr}}" ⇒ render() 里的一次 str.format
#   只把 "{{" 折成字面 "{"、**不做替换**（S14 生成器已知的 bug）。必须连这个 bug 一起复刻，
#   否则 prompt 字符串与 S14 数据不同、ckpt 评估无效。
TEMPLATES = [(t.replace("<W>", "{expr}").replace("<B>", "{{expr}}"), c) for t, c in TEMPLATES]
N_TPL_CN = sum(1 for _, c in TEMPLATES if c)
RENDER_VARIANTS = N_TPL_CN * 2 + (len(TEMPLATES) - N_TPL_CN)
CN_OPS = {"add_1d": ("加", "加上"), "add_2d": ("加", "加上"), "add_3d": ("加", "加上"),
          "sub_2d": ("减", "减去"), "mul_2d": ("乘", "乘以")}
EN_OPS = {"add_1d": ("plus",), "add_2d": ("plus",), "add_3d": ("plus",),
          "sub_2d": ("minus",), "mul_2d": ("times",)}
LOHI = {"add_1d": (0, 9), "add_2d": (10, 99), "add_3d": (100, 999)}

MUL_BY_Y: dict[int, list[tuple[int, int]]] = defaultdict(list)
for _a in range(10, 100):
    for _b in range(1, 10):   # 一位数乘数取 1–9（与 S14 逐字一致）
        MUL_BY_Y[_a * _b].append((_a, _b))
MUL_YS = sorted(MUL_BY_Y)
print(f"[S31] 模板={len(TEMPLATES)}句(中文{N_TPL_CN}×2排版+英文{len(TEMPLATES)-N_TPL_CN}) "
      f"⇒ 渲染{RENDER_VARIANTS}种 | 乘法答案域={len(MUL_YS)}个", flush=True)


class Rng:
    def __init__(self, seed: int):
        self._r = random.Random(seed)

    def randrange(self, n): return self._r.randrange(n)
    def randint(self, a, b): return self._r.randint(a, b)
    def random(self): return self._r.random()
    def shuffle(self, xs): self._r.shuffle(xs)


def answer_values(name: str) -> list[int]:
    if name in LOHI:
        lo, hi = LOHI[name]
        return list(range(2 * lo, 2 * hi + 1))
    if name == "sub_2d":
        return list(range(0, 90))
    return list(MUL_YS)


def split_by_answer(name: str, y: int, rng: Rng) -> tuple[int, int]:
    if name in LOHI:
        lo, hi = LOHI[name]
        a = rng.randint(max(lo, y - hi), min(hi, y - lo))
        return a, y - a
    if name == "sub_2d":
        a = rng.randint(max(10, y + 10), 99)
        return a, a - y
    fs = MUL_BY_Y[y]
    return fs[rng.randrange(len(fs))]


def render(name, a, b, y, rng: Rng) -> tuple[str, str, str]:
    tpl, cjk = TEMPLATES[rng.randrange(len(TEMPLATES))]
    op = (CN_OPS[name] if cjk else EN_OPS[name])
    op = op[rng.randrange(len(op))]
    tight = cjk and rng.random() < 0.5
    expr = f"{a}{op}{b}" if tight else f"{a} {op} {b}"
    nl = tpl.format(expr=expr)
    return f"题干：{nl} → ", f"#### {y}", nl


def gen_half(name, n, rng: Rng, used: set[str], tag: str) -> list[dict]:
    ys = answer_values(name)[:]
    rng.shuffle(ys)
    items, i, tries = [], 0, 0
    while len(items) < n:
        y = ys[i % len(ys)]
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
    return dict(p=p_ids, t=t_ids, gold=it["target"][5:].strip(), a=it["a"], b=it["b"],
                y=it["y"], nl=it["nl"], cut_p=cut_p, cut_t=cut_t)


DATA: dict[str, dict] = {}
print(f"\n[S31] ===== ① 数据重建（种子同 S14）=====", flush=True)
for name in BUCKETS:
    bi = BI[name]
    tr_rng, te_rng = Rng(14000 + bi * 7), Rng(14900 + bi * 7)
    used: set[str] = set()
    tr = [encode_record(x) for x in gen_half(name, N_TRAIN, tr_rng, used, "train")]
    te = [encode_record(x) for x in gen_half(name, N_TEST, te_rng, used, "test")]
    assert len(used) == len(tr) + len(te), "train/test 整串重叠"
    ys = answer_values(name)
    ctr = Counter(r["y"] for r in tr)
    cut = sum(1 for r in tr + te if r["cut_p"] or r["cut_t"])
    top5 = [v for v, _ in Counter(r["gold"] for r in tr).most_common(5)]
    prior = sum(1 for r in te if r["gold"] in top5) / len(te)
    DATA[name] = dict(train=tr, test=te, ys=ys, top5=top5, prior=prior,
                      n_pair_tr=len({(r["a"], r["b"]) for r in tr}),
                      n_pair_te=len({(r["a"], r["b"]) for r in te}))
    print(f"[S31-DATA] {name}: train={len(tr)} test={len(te)} 文本零重叠 | "
          f"答案域={min(ys)}..{max(ys)}({len(ys)}个) 计数train min/max="
          f"{min(ctr.values())}/{max(ctr.values())} | distinct(a,b) train="
          f"{DATA[name]['n_pair_tr']} test={DATA[name]['n_pair_te']} | 截断={cut} | "
          f"★最高频5={top5} 机会水平={100.0*prior:.1f}%", flush=True)

# 自证同一性：对 S14/S19/S22 日志里的数据行逐项比对
_EXP = {  # (train distinct(a,b), test distinct(a,b), 机会水平%) 来自 logs/19_res.log:6-18 / 22_card.log
    "add_1d": (100, 100, "27.5"), "add_2d": (2868, 732, "2.8"), "sub_2d": (2302, 683, "5.6"),
    "mul_2d": (781, 561, "1.1"), "add_3d": (3983, 800, "0.2"),
}
_ok_all = True
for _b, (_ptr, _pte, _pri) in _EXP.items():
    _d = DATA[_b]
    _g = (_d["n_pair_tr"], _d["n_pair_te"], f"{100*_d['prior']:.1f}")
    _ok = (_g == (_ptr, _pte, _pri))
    _ok_all &= _ok
    print(f"[S31-DATA-CHECK] {_b:7s} 我的 distinct(a,b) train/test={_g[0]}/{_g[1]} "
          f"机会水平={_g[2]}% vs S14/S19 入库={(_ptr, _pte, _pri)} ⇒ "
          f"{'逐位一致 ✓' if _ok else '★不一致'}", flush=True)
_h = hashlib.sha256()
for r in DATA["add_3d"]["test"]:
    _h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
print(f"[S31-DATA-CHECK] 五桶逐位一致={_ok_all} | add_3d TEST 指纹 sha256[:16]="
      f"{_h.hexdigest()[:16]} n={len(DATA['add_3d']['test'])}", flush=True)

# ---------------------------------------------------------------- §0-b 泄漏率
print(f"\n[S31] ===== §0 泄漏率：两种口径 =====", flush=True)
LEAK: dict[str, dict] = {}
for name in BUCKETS:
    tr, te = DATA[name]["train"], DATA[name]["test"]
    pair_tr = {(r["a"], r["b"]) for r in tr}
    trip_tr = {(r["a"], r["b"], r["y"]) for r in tr}
    pair_tr_set = {(r["a"], r["b"]): set() for r in tr}
    for r in tr:
        pair_tr_set[(r["a"], r["b"])].add(r["y"])
    n_te = len(te)
    n_pair = sum(1 for r in te if (r["a"], r["b"]) in pair_tr)
    n_trip = sum(1 for r in te if (r["a"], r["b"], r["y"]) in trip_tr)
    n_pair_mult = sum(1 for r in te if len(pair_tr_set.get((r["a"], r["b"]), ())) > 1)
    LEAK[name] = dict(pair=n_pair, pair_rate=n_pair / n_te,
                      trip=n_trip, trip_rate=n_trip / n_te, n=n_te,
                      n_pair_distinct_tr=len(pair_tr), n_trip_distinct_tr=len(trip_tr),
                      pair_mult=n_pair_mult)
    print(f"[S31-LEAK] {name:7s} n={n_te} | (a,b) 数对泄漏={n_pair}/{n_te}="
          f"{100.0*n_pair/n_te:.1f}% | ★(a,b,y) 三元组泄漏={n_trip}/{n_te}="
          f"{100.0*n_trip/n_te:.1f}% | train distinct (a,b)={len(pair_tr)} "
          f"三元组={len(trip_tr)} | train 内同 (a,b) 有多个 y 的 test 条={n_pair_mult}",
          flush=True)

# ============================================================================
# §1/§2/§3 模型 + 自己的评估路径
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


class Cards31(nn.Module):
    """按 ckpt 的键重建；residual=True 时 h = h + Think(h)（S19-res/S25/S28 架构）。"""

    def __init__(self, residual: bool = True):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.in_enc = mk_layer()
        self.thought = mk_layer()
        self.head = nn.Linear(D, V)
        self.residual = residual
        self.register_buffer("pe", sin_pe(MAXLEN, D), persistent=False)

    def forward(self, ids):
        n = ids.size(1)
        m = torch.full((n, n), float("-inf"), device=ids.device)
        m = torch.triu(m, diagonal=1).unsqueeze(0).expand(ids.size(0), -1, -1).clone()
        m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
        m = m.repeat_interleave(NHEAD, dim=0)
        pe = self.pe if n <= self.pe.size(0) else sin_pe(n, D).to(self.pe.device)
        x = self.emb(ids) + pe[:n].unsqueeze(0)
        h = self.in_enc(x, src_mask=m)
        h = h + self.thought(h, src_mask=m) if self.residual else self.thought(h, src_mask=m)
        return self.head(h)


def load_ckpt(path: str, residual: bool = True):
    m = Cards31(residual=residual)
    sd = torch.load(path, map_location="cpu")
    m.load_state_dict(sd)
    return m.to(dev).eval()


@torch.no_grad()
def gen_bs1(model, recs) -> list[str]:
    """逐样本贪心解码（batch=1 by construction；必须 eval 否则 dropout 打开）。"""
    model.eval()
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


def parse_ans(text):
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
def digit_step_acc(model, recs):
    """数字每步正确率 = exp(−数字 token 的 per-sample CE 均值)（自己写的口径）。"""
    model.eval()
    per = []
    for i in range(0, len(recs), 32):
        ch = recs[i: i + 32]
        n = max(len(r["p"]) + len(r["t"]) for r in ch)
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
            d = [c for c, tk in zip(ces, r["t"]) if digit_token(tk)]
            if d:
                per.append(sum(d) / len(d))
    return math.exp(-(sum(per) / len(per))) if per else float("nan")


def binomial_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan")
    s = sum(xs) / n
    return math.sqrt(s * (1 - s) / n)


def mean_se(xs):
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    v = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(v / n)


# ---------------- §3 聚类 bootstrap（重采样 (a,b) 数对，非样本） ----------------
def cluster_bootstrap(recs, hits, nboot, rng):
    """按 (a,b) 聚类重采样；返回 overall EM 的聚类 SE 与 Δ=EM(SEEN)−EM(UNSEEN) 的聚类 SE。"""
    groups: dict[tuple, list[int]] = defaultdict(list)
    for i, r in enumerate(recs):
        groups[(r["a"], r["b"])].append(i)
    keys = list(groups)
    ems, deltas = [], []
    for _ in range(nboot):
        pick = [keys[rng.randrange(len(keys))] for _ in range(len(keys))]
        idx = [i for k in pick for i in groups[k]]
        ems.append(sum(hits[i] for i in idx) / len(idx))
        s_idx = [i for i in idx if _seen_flag[i]]
        u_idx = [i for i in idx if not _seen_flag[i]]
        if s_idx and u_idx:
            deltas.append(sum(hits[i] for i in s_idx) / len(s_idx)
                          - sum(hits[i] for i in u_idx) / len(u_idx))
    se_em = (math.sqrt(sum((x - sum(ems) / len(ems)) ** 2 for x in ems) / (len(ems) - 1))
             if len(ems) > 1 else float("nan"))
    if len(deltas) > 1:
        dm = sum(deltas) / len(deltas)
        se_d = math.sqrt(sum((x - dm) ** 2 for x in deltas) / (len(deltas) - 1))
    else:
        dm, se_d = float("nan"), float("nan")
    return se_em, (dm, se_d)


_seen_flag: list[bool] = []


# ============================================================================
# 主循环
# ============================================================================
RES: list[dict] = []
summary: dict[str, dict] = {}
boot_rng = random.Random(31337)

print(f"\n[S31] ===== §1 SEEN/UNSEEN 分层 EM（19_ckpt_*_res_seed*，batch=1）=====", flush=True)
ck_files = []
for name in BUCKETS:
    if ONLY and name not in ONLY:
        continue
    for seed in SEEDS:
        p = f"{LOGDIR}/19_ckpt_{name}_res_seed{seed}.pt"
        if os.path.exists(p):
            ck_files.append((name, seed, p))
        else:
            print(f"[S31] ★缺 {p}", flush=True)

for name, seed, p in ck_files:
    t0 = time.time()
    tr, te = DATA[name]["train"], DATA[name]["test"]
    if LIMIT:
        te = te[:LIMIT]
    pair_tr = {(r["a"], r["b"]) for r in tr}
    trip_tr = {(r["a"], r["b"], r["y"]) for r in tr}
    seen = [i for i, r in enumerate(te) if (r["a"], r["b"]) in pair_tr]
    unseen = [i for i, r in enumerate(te) if (r["a"], r["b"]) not in pair_tr]
    trip_seen = [i for i, r in enumerate(te) if (r["a"], r["b"], r["y"]) in trip_tr]

    model = load_ckpt(p)
    txt = gen_bs1(model, te)
    hits = [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, te)]
    acc = digit_step_acc(model, te)

    def em(idx):
        return sum(hits[i] for i in idx) / len(idx) if idx else float("nan")

    em_all, se_all = em(range(len(te))), binomial_se(hits)
    em_s, em_u = em(seen), em(unseen)
    se_s, se_u = binomial_se([hits[i] for i in seen]), binomial_se([hits[i] for i in unseen])
    d_naive = em_s - em_u
    se_d_naive = math.sqrt(se_s ** 2 + se_u ** 2) if seen and unseen else float("nan")

    # §3 聚类 bootstrap（配对：同一重采样内取 Δ）
    _seen_flag[:] = [False] * len(te)
    for i in seen:
        _seen_flag[i] = True
    se_clu, (d_clu, se_d_clu) = cluster_bootstrap(te, hits, NBOOT, boot_rng)

    # 泄漏对 EM 的贡献分解：整体 EM 与 UNSEEN 之差 = (n_seen/n)·Δ
    contrib = (len(seen) / len(te)) * d_naive if seen else float("nan")

    r = dict(bucket=name, seed=seed, n=len(te), n_seen=len(seen), n_unseen=len(unseen),
             n_trip_seen=len(trip_seen), em=em_all, se_naive=se_all,
             em_seen=em_s, se_seen=se_s, em_unseen=em_u, se_unseen=se_u,
             delta_naive=d_naive, se_delta_naive=se_d_naive,
             se_cluster=se_clu, delta_cluster=d_clu, se_delta_cluster=se_d_clu,
             contrib_leak=contrib, digit_acc=acc,
             ckpt=os.path.basename(p))
    RES.append(r)
    print(f"[S31-EM] {name:7s} seed={seed} n={len(te)} | 全量EM={100*em_all:.2f}%±"
          f"{100*se_all:.2f}(朴素) ±{100*se_clu:.2f}(聚类) | SEEN n={len(seen)} "
          f"EM={100*em_s:.2f}%±{100*se_s:.2f} | UNSEEN n={len(unseen)} "
          f"EM={100*em_u:.2f}%±{100*se_u:.2f} | ★Δ=EM(SEEN)−EM(UNSEEN)="
          f"{100*d_naive:+.2f}pp±{100*se_d_naive:.2f}(朴素)±{100*se_d_clu:.2f}(聚类) | "
          f"泄漏对整体的贡献=(n_seen/n)·Δ={100*contrib:+.2f}pp | 数字每步={100*acc:.2f}% | "
          f"{time.time()-t0:.0f}s", flush=True)
    for i in range(3):
        print(f"    [{name} s{seed} 样例{i}] gold={te[i]['gold']!r} 生成={txt[i][:40]!r} "
              f"(a={te[i]['a']} b={te[i]['b']} {'SEEN' if i in seen else 'UNSEEN'})", flush=True)
    del model

# ---------------- §2 跨桶关系 ----------------
print(f"\n[S31] ===== §2 跨桶：泄漏率 vs EM（每桶 seed1234 为准）=====", flush=True)
tab = []
for name in BUCKETS:
    rr = [r for r in RES if r["bucket"] == name and r["seed"] == SEEDS[0]]
    if not rr:
        continue
    r = rr[0]
    tab.append((name, LEAK[name]["pair_rate"], LEAK[name]["trip_rate"],
                r["em"], r["delta_naive"]))
tab.sort(key=lambda x: x[1], reverse=True)
for name, lk, lt, emv, dl in tab:
    print(f"[S31-CROSS] {name:7s} 泄漏率(a,b)={100*lk:.1f}% ({lt*100:.1f}% 三元组) | "
          f"全量EM={100*emv:.2f}% | Δ={100*dl:+.2f}pp", flush=True)
_pr = [x[1] for x in tab]
_em = [x[3] for x in tab]
_n = len(_pr)
_mpr, _mem = sum(_pr) / _n, sum(_em) / _n
_cov = sum((a - _mpr) * (b - _mem) for a, b in zip(_pr, _em))
_vp = math.sqrt(sum((a - _mpr) ** 2 for a in _pr))
_ve = math.sqrt(sum((b - _mem) ** 2 for b in _em))
_pear = _cov / (_vp * _ve) if _vp > 0 and _ve > 0 else float("nan")
print(f"[S31-CROSS] Pearson r(泄漏率, EM) over {_n} 桶 = {_pear:.3f} | "
      f"排序后是否单调（按泄漏率降序的 EM）={[f'{b*100:.1f}%' for b in _em]} ⇒ "
      f"{'单调正相关' if all(_em[i] >= _em[i+1] for i in range(_n-1)) else '★非单调 ⇒ 泄漏不是唯一解释'}",
      flush=True)
_a3r = [r for r in RES if r["bucket"] == "add_3d"]
if _a3r:
    print(f"[S31-CLEAN] add_3d（泄漏 {100*LEAK['add_3d']['pair_rate']:.1f}%）UNSEEN EM="
          f"{100*_a3r[0]['em_unseen']:.2f}%±{100*_a3r[0]['se_unseen']:.2f} (n={_a3r[0]['n_unseen']}) "
          f"| SEEN n={_a3r[0]['n_seen']} ⇒ ★无泄漏条件下的三位数加法 EM（干净难度信号）",
          flush=True)

# ============================================================================
# §2b 补充：nores 臂（S19 里 2.50% 那个 ckpt）+ 全部 19_*_res 的 §3 SE 汇总
# ============================================================================
print(f"\n[S31] ===== §2b nores 臂对照（S19 的 2.50% 来自 add_3d **nores**）=====", flush=True)
NR = []
for name in BUCKETS:
    for seed in SEEDS:
        p = f"{LOGDIR}/19_ckpt_{name}_nores_seed{seed}.pt"
        if not os.path.exists(p):
            continue
        t0 = time.time()
        tr, te = DATA[name]["train"], DATA[name]["test"]
        pair_tr = {(r["a"], r["b"]) for r in tr}
        seen = [i for i, r in enumerate(te) if (r["a"], r["b"]) in pair_tr]
        us = [i for i in range(len(te)) if i not in seen]
        model = load_ckpt(p)
        txt = gen_bs1(model, te)
        hits = [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, te)]
        em = sum(hits) / len(hits)
        em_s = sum(hits[i] for i in seen) / len(seen) if seen else float("nan")
        em_u = sum(hits[i] for i in us) / len(us) if us else float("nan")
        NR.append(dict(bucket=name, seed=seed, em=em, se=binomial_se(hits),
                       em_seen=em_s, em_unseen=em_u, n_seen=len(seen), n_unseen=len(us)))
        print(f"[S31-NORES] {name:7s} seed={seed} n={len(te)} EM={100*em:.2f}%±"
              f"{100*binomial_se(hits):.2f} | SEEN(n={len(seen)})={100*em_s:.2f}% "
              f"UNSEEN(n={len(us)})={100*em_u:.2f}% | {time.time()-t0:.0f}s", flush=True)
        del model
_a3n = [r for r in NR if r["bucket"] == "add_3d"]
for r in _a3n:
    print(f"[S31-CLEAN-NORES] add_3d(泄漏 {100*LEAK['add_3d']['pair_rate']:.1f}%, nores) "
          f"seed={r['seed']}: ★UNSEEN EM={100*r['em_unseen']:.2f}% (n={r['n_unseen']}) "
          f"= 无泄漏条件下的三位数加法 EM（S19 报 2.50%/9.62% 即此臂）", flush=True)


print(f"\n[S31] ===== §4 28_ckpt_e2e6000 补评（add_3d, batch=1, 与 S25 同口径）=====", flush=True)
EXPECT = {1234: 85.625, 5678: 74.375}
S4 = []
for seed in SEEDS:
    p = f"{LOGDIR}/28_ckpt_e2e6000_seed{seed}.pt"
    if not os.path.exists(p):
        print(f"[S31] ★缺 {p}", flush=True)
        continue
    t0 = time.time()
    tr, te = DATA["add_3d"]["train"], DATA["add_3d"]["test"]
    model = load_ckpt(p)
    txt = gen_bs1(model, te)
    hits = [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, te)]
    em = sum(hits) / len(hits)
    acc = digit_step_acc(model, te)
    pair_tr = {(r["a"], r["b"]) for r in tr}
    seen = [i for i, r in enumerate(te) if (r["a"], r["b"]) in pair_tr]
    em_s = sum(hits[i] for i in seen) / len(seen) if seen else float("nan")
    us = [i for i in range(len(te)) if i not in seen]
    em_u = sum(hits[i] for i in us) / len(us) if us else float("nan")
    d = (em * 100 - EXPECT[seed])
    print(f"[S31-S4] e2e6000 seed={seed} n={len(te)} 我的EM={100*em:.2f}%±"
          f"{100*binomial_se(hits):.2f} | 28_train.log 报={EXPECT[seed]:.2f}% | Δ={d:+.2f}pp "
          f"⇒ {'复现 ✓（≤1pp）' if abs(d) <= 1.0 else '★不复现'} | 数字每步={100*acc:.2f}% | "
          f"SEEN(n={len(seen)})={100*em_s:.2f}% UNSEEN(n={len(us)})={100*em_u:.2f}% | "
          f"{time.time()-t0:.0f}s", flush=True)
    S4.append(dict(seed=seed, em=em, se=binomial_se(hits), expect=EXPECT[seed], d_pp=d,
                   digit_acc=acc, em_seen=em_s, em_unseen=em_u, n=len(te)))
    del model

# ============================================================================
# 落盘
# ============================================================================
out = dict(device=dev, seeds=list(SEEDS), leak=LEAK, results=RES, nores=NR, s4=S4,
           pearson=_pear, wall=time.time() - t_start)
with open(LOGDIR + "/31_results.jsonl", "w") as f:
    f.write(json.dumps(dict(kind="leak", leak=LEAK), ensure_ascii=False) + "\n")
    for r in RES:
        f.write(json.dumps(dict(kind="em", **r), ensure_ascii=False) + "\n")
    for r in NR:
        f.write(json.dumps(dict(kind="em_nores", **r), ensure_ascii=False) + "\n")
    for r in S4:
        f.write(json.dumps(dict(kind="e2e6000", **r), ensure_ascii=False) + "\n")
print(f"\n[S31] 汇总已写 logs/31_results.jsonl | 墙钟={time.time()-t_start:.1f}s", flush=True)
print("[S31-DONE]", flush=True)
