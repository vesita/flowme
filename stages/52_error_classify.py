#!/usr/bin/env python3
"""S52 · E9（零训练 · CPU · batch=1）：add_3d clean(n=770) 错例分类 + UNSEEN/SEEN 分层 + B 类难度分布。

问题（NEXT_MAINLINE.md Q2 零训练格）：add_3d clean EM ≈ 91–95% 是【口径顶】还是【模型顶】？
判据（写死）：
  · 解析错为主 ⇒ 口径顶（评估口径吃掉能力）；数值错为主 ⇒ 模型顶。
  · UNSEEN ≈ 全量（Δ ≤ 1SE）⇒ 不是记忆缺口造成。
  · B 类（数值错）若集中在长进位链/4 位数 ⇒ 难度墙；均匀 ⇒ 普遍精度不足（优化/容量）。

口径与来源（数据/模型/生成逐字复用 stages/43_clean_em.py，不另写一套）：
  数据 S14：Rng(14000+2*7)/Rng(14900+2*7) 复刻 add_3d train=4000 / test=800；
            clean = 剔 nl 含字面量 "{expr}" 或 nl 无数字的 30 条 ⇒ n=770（与 S42 核对）。
  ckpt S38：logs/38_ckpt_w1_s{12000,15000}_seed{1234,5678}.pt
            （唯一臂 w1 = res 臂 = Cards(d=128, ff=512, residual=True)）。
  R28 贪心 batch=1；R39 生成前显式 model.eval()；R40 torch.set_num_threads(4)。
只允许写：本文件、logs/52_error_classify.log、logs/52_error_classify.json、/tmp；其余只读。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

FLOWME = "/home/vesita/coding/my/flowme"
NS_ROOT = "/home/vesita/coding/my/nanoSeek"
LOG_DIR = FLOWME + "/logs"
sys.path.insert(0, NS_ROOT)

os.environ["CUDA_VISIBLE_DEVICES"] = ""     # ★硬约束：全程 CPU
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

NTHREADS = 4
torch.set_num_threads(NTHREADS)             # R40：batch=1 限线程
DEVICE = "cpu"
SMOKE = os.environ.get("S52_SMOKE") == "1"

# ---- 超参（与 S43/S38 逐字一致）----
D, FF, NHEAD = 128, 512, 4
MAXLEN, MIN_T, TAIL_KEEP = 512, 48, 24
MAX_GEN = 48
EM_BATCH = 1                                # R28
N_TRAIN, N_TEST = 4000, 800
BUCKETS = ("add_1d", "add_2d", "add_3d", "sub_2d", "mul_2d")
BUCKET, BI = "add_3d", 2                    # add_3d 的 rng 种子 = 14000+BI*7 / 14900+BI*7
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
S42_JSON = LOG_DIR + "/42_data_hygiene.json"
OUT_JSON = LOG_DIR + "/52_error_classify.json"
OUT_LOG = LOG_DIR + "/52_error_classify.log"
CKPTS = [(st, sd, f"{LOG_DIR}/38_ckpt_w1_s{st}_seed{sd}.pt")
         for st in (12000, 15000) for sd in (1234, 5678)]
# 与 logs/43_clean_em.log:31-34 相同的 ckpt 指纹（锚点）
EXP_SHA = {"38_ckpt_w1_s12000_seed1234.pt": "639360d078db2493",
           "38_ckpt_w1_s12000_seed5678.pt": "ee99239df44334e6",
           "38_ckpt_w1_s15000_seed1234.pt": "a855df724c8f1004",
           "38_ckpt_w1_s15000_seed5678.pt": "539adcd14c846fa8"}
# 与 logs/43_clean_em.log 相同的 clean EM（锚点）
EXP_EM = {("12000", 1234): 95.45, ("12000", 5678): 90.52,
          ("15000", 1234): 94.81, ("15000", 5678): 90.65}

t_start = time.time()
_log_f = open(OUT_LOG, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log_f.write(s + "\n")
    _log_f.flush()


say(f"[S52] device={DEVICE} torch={torch.__version__} cuda.is_available()={torch.cuda.is_available()} "
    f"threads={torch.get_num_threads()}(R40) | 零训练 · EM batch={EM_BATCH}(R28) · eval() 显式(R39) | "
    f"桶={BUCKET} ckpt=38_ckpt_w1_s{{12000,15000}}_seed{{1234,5678}}")

# ============================================================================
# ① 数据：逐字复刻 stages/14_synth_arith.py（只读），只建 add_3d
# ============================================================================
tok = Tokenizer.from_file(TOK)
V = tok.get_vocab_size()
PAD_ID = tok.token_to_id("<pad>")
EOS_ID = tok.token_to_id("<eos>")


def enc(text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


def build_templates() -> list[tuple[str, bool]]:
    ts: list[tuple[str, bool]] = []
    tails_cn = ["等于多少？", "是多少？", "等于几？", "得多少？", "的结果是多少？",
                "的结果是几？", "的值是多少？", "等于多少", "是多少", "是几"]
    for lead in ("", "请", "帮我", "麻烦"):
        for verb in ("计算", "算出", "算一下", "口算", "快速算", "算"):
            for t in tails_cn:
                ts.append((f"{lead}{verb} {{expr}} {t}", True))
    for t in tails_cn:                       # ★裸式：.format() 后残留字面量 {expr}
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


def split_by_answer(name: str, y: int, rng):
    if name in LOHI:
        lo, hi = LOHI[name]
        a = rng.randint(max(lo, y - hi), min(hi, y - lo))
        return a, y - a
    if name == "sub_2d":
        a = rng.randint(max(10, y + 10), 99)
        return a, a - y
    fs = MUL_BY_Y[y]
    return fs[rng.randrange(len(fs))]


class Rng:
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


def render(name: str, a: int, b: int, y: int, rng: Rng):
    tpl, cjk = TEMPLATES[rng.randrange(len(TEMPLATES))]
    op = (CN_OPS[name] if cjk else EN_OPS[name])
    op = op[rng.randrange(len(op))]
    tight = cjk and rng.random() < 0.5
    expr = f"{a}{op}{b}" if tight else f"{a} {op} {b}"
    nl = tpl.format(expr=expr)
    return f"题干：{nl} → ", f"#### {y}", nl


def gen_half(name: str, n: int, rng: Rng, used: set, tag: str) -> list[dict]:
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
                y=it["y"], nl=it["nl"], prompt=it["prompt"])


say(f"\n[S52] ===== §1 复刻 {BUCKET} train/test + clean 索引 =====")
t0 = time.time()
tr_rng, te_rng = Rng(14000 + BI * 7), Rng(14900 + BI * 7)
used: set = set()
TRAIN = [encode_record(x) for x in gen_half(BUCKET, N_TRAIN, tr_rng, used, "train")]
TEST = [encode_record(x) for x in gen_half(BUCKET, N_TEST, te_rng, used, "test")]
BAD_IDX = [i for i, r in enumerate(TEST) if "{expr}" in r["nl"] or not any(c.isdigit() for c in r["nl"])]
CLEAN_IDX = [i for i in range(len(TEST)) if i not in set(BAD_IDX)]
say(f"[S52-DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)} bad={len(BAD_IDX)} "
    f"clean={len(CLEAN_IDX)} | 生成墙钟={time.time()-t0:.1f}s")

_h = hashlib.sha256()
for r in TEST:
    _h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
say(f"[S52-DATA-CHK] add_3d TEST 指纹 sha256[:16]={_h.hexdigest()[:16]} n={len(TEST)} "
    f"（S31 入库=b34e7ea515203227）⇒ {'一致 ✓' if _h.hexdigest()[:16] == 'b34e7ea515203227' else '★不一致'}")
try:
    _s42 = json.load(open(S42_JSON, encoding="utf-8"))
    _b42 = _s42["s1_template_bug"]["add_3d"]["bad_test"]
    say(f"[S52-DATA-CHK] bad_test 本机={len(BAD_IDX)} vs S42={_b42} ⇒ "
        f"{'一致 ✓' if _b42 == len(BAD_IDX) else '★不一致'}")
except FileNotFoundError:
    say("[S52-DATA-CHK] ★找不到 logs/42_data_hygiene.json ⇒ 无法核对（如实报）")

# ---- UNSEEN / SEEN：与 S31 同定义（(a,b) 数对是否出现在 train）----
PAIR_TR = {(r["a"], r["b"]) for r in TRAIN}
SEEN_FULL = [i for i, r in enumerate(TEST) if (r["a"], r["b"]) in PAIR_TR]
UNSEEN_FULL = [i for i, r in enumerate(TEST) if (r["a"], r["b"]) not in PAIR_TR]
CLEAN_SET = set(CLEAN_IDX)
SEEN_CLEAN = [i for i in SEEN_FULL if i in CLEAN_SET]
UNSEEN_CLEAN = [i for i in UNSEEN_FULL if i in CLEAN_SET]
say(f"[S52-SPLIT] 全量 n=800: SEEN={len(SEEN_FULL)} UNSEEN={len(UNSEEN_FULL)} | "
    f"clean n={len(CLEAN_IDX)}: SEEN={len(SEEN_CLEAN)} UNSEEN={len(UNSEEN_CLEAN)}")

# ============================================================================
# ② 模型（逐字同构 S43/S38：w1 = res 臂）
# ============================================================================
def sin_pe(max_len: int, d: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def enc_layer(d: int, ff: int) -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(
        d, NHEAD, ff, dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )


class Cards(nn.Module):
    """S19/S38 容器；w1 ckpt = residual=True。"""

    def __init__(self, d: int, ff: int, residual: bool = False):
        super().__init__()
        self.residual = residual
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        self.thought = enc_layer(d, ff)
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
        x = self.emb(ids) + pe[:n].unsqueeze(0)
        h = self.in_enc(x, src_mask=m)
        t = self.thought(h, src_mask=m)
        h = h + t if self.residual else t
        return self.head(h)


def load_ckpt(path: str):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model = Cards(D, FF, residual=True).to(DEVICE)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "pe" not in k], f"{path} 缺键 {missing[:5]}"
    assert not unexpected, f"{path} 多键 {unexpected[:5]}"
    model.eval()                                     # R39：评估前显式 eval()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return model, h.hexdigest()[:16]


@torch.no_grad()
def greedy_gen(model, recs, batch: int = 1):
    """S14/S43 greedy_gen 逐字复刻；额外回传是否由 <eos> 正常收尾（未收尾⇒截断）。"""
    model.eval()                                     # R39
    texts: list[str] = [""] * len(recs)
    ended_eos: list[bool] = [False] * len(recs)
    for i in range(0, len(recs), batch):
        chunk = recs[i: i + batch]
        lens = [len(r["p"]) for r in chunk]
        n0 = max(lens)
        ids = torch.full((len(chunk), n0), PAD_ID, dtype=torch.long)
        for j, r in enumerate(chunk):
            ids[j, : lens[j]] = torch.tensor(r["p"])
        ids = ids.to(DEVICE)
        gen_len = [0] * len(chunk)
        cap = [min(MAXLEN - L, MAX_GEN) for L in lens]
        done = [False] * len(chunk)
        last_col = [L - 1 for L in lens]
        last_tok = [PAD_ID] * len(chunk)
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
                last_tok[j] = t
                gen_len[j] += 1
                if t == EOS_ID or gen_len[j] >= cap[j]:
                    done[j] = True
                last_col[j] = n0 + gen_len[j] - 1
            ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long, device=DEVICE).unsqueeze(1)], 1)
        for j in range(len(chunk)):
            g_ids = [x for x in ids[j, n0: n0 + gen_len[j]].tolist() if x != EOS_ID]
            texts[i + j] = dec(g_ids)
            ended_eos[i + j] = (last_tok[j] == EOS_ID)
    return texts, ended_eos


def parse_ans(text: str):
    if "####" not in text:
        return None
    a = text.rsplit("####", 1)[1].split("\n")[0]
    a = a.strip().replace(",", "").replace(" ", "").replace("$", "")
    return a or None


def int_of(parsed):
    if parsed is None:
        return None
    s = parsed[1:] if parsed[:1] in "+-" else parsed
    return int(parsed) if s.isdigit() else None


# ============================================================================
# ③ 误差分类（写死口径）
# ============================================================================
CLASS_NAME = {"A": "A 解析错(有数字但规范化后≠gold)", "B": "B 数值错(parse 得整数≠gold)",
              "C": "C 无输出/无数字", "D": "D 截断/未出 eos", "E": "E 其它"}


def classify_err(gen: str, gold: str, ended_eos: bool) -> str:
    has_digit = any(c.isdigit() for c in gen)
    parsed = parse_ans(gen)
    pint = int_of(parsed)
    if not ended_eos:                       # D 优先：长度上限截断
        return "D"
    if not has_digit:
        return "C"
    if pint is None or str(pint) == gold:   # 有数字但无法归一成整数（含前导零）
        return "A"
    return "B"


def binom_se(p: float, n: int) -> float:
    return math.sqrt(p * (1 - p) / n) if n else float("nan")


# ============================================================================
# ④ 难度特征：进位链 / 位数
# ============================================================================
def carry_feats(a: int, b: int) -> dict:
    x, y = a, b
    carries: list[int] = []
    c = 0
    while x > 0 or y > 0:
        s = (x % 10) + (y % 10) + c
        c = 1 if s >= 10 else 0
        carries.append(c)
        x //= 10
        y //= 10
    n_carry = sum(carries)
    best = cur = 0
    for v in carries:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return dict(n_carry=n_carry, chain=best, n_digits=len(str(a + b)))


FEAT = {i: carry_feats(r["a"], r["b"]) for i, r in enumerate(TEST)}
say(f"[S52-FEAT] clean 内难度分布: 进位次数 {dict(sorted(Counter(FEAT[i]['n_carry'] for i in CLEAN_IDX).items()))} "
    f"| 最长进位链 {dict(sorted(Counter(FEAT[i]['chain'] for i in CLEAN_IDX).items()))} "
    f"| 位数 {dict(sorted(Counter(FEAT[i]['n_digits'] for i in CLEAN_IDX).items()))}")

# ============================================================================
# ⑤ 主循环：逐 ckpt 生成 + 分类 + 分层
# ============================================================================
say(f"\n[S52] ===== §2 逐 ckpt：clean(n={len(CLEAN_IDX)}) 贪心 batch=1 生成 + 错例分类 =====")
RES: list[dict] = []
VEC: list[dict] = []
RECS_CLEAN = [TEST[i] for i in CLEAN_IDX]
for st, sd, path in CKPTS:
    label = f"w1_s{st}_seed{sd}"
    if not os.path.exists(path):
        say(f"[SKIP] {label}: ★ckpt 缺失 {path}")
        continue
    model, sha = load_ckpt(path)
    t0 = time.time()
    txt, eos = greedy_gen(model, RECS_CLEAN, batch=EM_BATCH)
    wall = time.time() - t0
    del model

    hits: dict[int, int] = {}
    parsed_of: dict[int, str | None] = {}
    err_of: dict[int, str] = {}
    ex: dict[str, list[str]] = defaultdict(list)
    for j, i in enumerate(CLEAN_IDX):
        gold = TEST[i]["gold"]
        p = parse_ans(txt[j])
        parsed_of[i] = p
        ok = int(p == gold)
        hits[i] = ok
        if not ok:
            cls = classify_err(txt[j], gold, eos[j])
            err_of[i] = cls
            if len(ex[cls]) < 4:
                ex[cls].append(f"a={TEST[i]['a']} b={TEST[i]['b']} gold='{gold}' "
                               f"gen='{txt[j][:60]}' parsed={p!r}")

    # ---- EM：clean 与全量（bug 样本 = 全错，S43 已验证 only_bug=0.00%）----
    def em(sel, hits=hits):        # 显式绑定本轮的 hits（避免闭包引用下一轮的对象）
        n = len(sel)
        k = sum(hits.get(i, 0) for i in sel)
        p = k / n if n else float("nan")
        return dict(n=n, k=k, em=p, se=binom_se(p, n))

    all_clean, all_full = em(CLEAN_IDX), em(range(len(TEST)))
    seen_c, unseen_c = em(SEEN_CLEAN), em(UNSEEN_CLEAN)
    seen_f, unseen_f = em(SEEN_FULL), em(UNSEEN_FULL)

    cnt = Counter(err_of.values())
    n_err = len(err_of)
    d_em = (all_clean["em"] - EXP_EM[(str(st), sd)] / 100.0) * 100
    say(f"\n[EM] {label} sha={sha}{'（与 S43 一致 ✓）' if EXP_SHA.get(os.path.basename(path)) == sha else '（★sha 与 S43 不符）'} "
        f"| clean EM={all_clean['em']*100:.2f}%±{all_clean['se']*100:.2f} (n={all_clean['n']}, "
        f"与 S43 Δ={d_em:+.2f}pp) | 全量={all_full['em']*100:.2f}% | 错例={n_err} "
        f"| 生成墙钟={wall:.1f}s")
    say(f"     全量口径: SEEN n={seen_f['n']} EM={seen_f['em']*100:.2f}%±{seen_f['se']*100:.2f} | "
        f"UNSEEN n={unseen_f['n']} EM={unseen_f['em']*100:.2f}%±{unseen_f['se']*100:.2f} | "
        f"Δ(UNSEEN−全量)={((unseen_f['em']-all_full['em'])*100):+.2f}pp (1SE={unseen_f['se']*100:.2f}pp)")
    say(f"     clean 口径: SEEN n={seen_c['n']} EM={seen_c['em']*100:.2f}%±{seen_c['se']*100:.2f} | "
        f"UNSEEN n={unseen_c['n']} EM={unseen_c['em']*100:.2f}%±{unseen_c['se']*100:.2f} | "
        f"Δ(UNSEEN−clean)={((unseen_c['em']-all_clean['em'])*100):+.2f}pp (1SE={unseen_c['se']*100:.2f}pp) "
        f"| Δ(SEEN−UNSEEN)={((seen_c['em']-unseen_c['em'])*100):+.2f}pp")
    say(f"     误差分类 (n_err={n_err}): " + " | ".join(
        f"{c}={cnt.get(c,0)}({100.0*cnt.get(c,0)/n_err:.1f}%)" for c in "ABCDE" if cnt.get(c, 0)))
    for c in "ABCDE":
        for e in ex[c]:
            say(f"       [{c}] {e}")

    # ---- B 类难度 ----
    b_idx = [i for i, c in err_of.items() if c == "B"]
    RES.append(dict(label=label, steps=st, seed=sd, sha=sha, wall_gen=wall,
                    clean=all_clean, full=all_full,
                    seen_full=seen_f, unseen_full=unseen_f,
                    seen_clean=seen_c, unseen_clean=unseen_c,
                    n_err=n_err, err_counts={c: cnt.get(c, 0) for c in "ABCDE"},
                    err_share={c: (cnt.get(c, 0) / n_err if n_err else 0.0) for c in "ABCDE"},
                    b_idx=b_idx, examples={c: ex[c] for c in "ABCDE"}))
    VEC.append(dict(label=label, hits=hits, err_of=err_of, b_idx=b_idx))

# ============================================================================
# ⑥ §3 B 类（数值错）难度分布
# ============================================================================
say("\n[S52] ===== §3 B 类（数值错）的难度分布：进位次数 / 位数 =====")
pool_b = [i for v in VEC for i in v["b_idx"]]
base = [FEAT[i] for i in CLEAN_IDX]
n_base = len(base)


def share(xs, pred):
    return sum(1 for x in xs if pred(x)) / len(xs) if xs else float("nan")


# 全错例（不只 B）也报一份，便于看梯度
pool_all_err = [i for v in VEC for i in v["err_of"]]
DIST = {}
for key, pred, name in (("n_carry>=2", lambda f: f["n_carry"] >= 2, "进位次数≥2"),
                        ("n_carry==3", lambda f: f["n_carry"] == 3, "进位次数=3"),
                        ("chain>=2", lambda f: f["chain"] >= 2, "最长进位链≥2"),
                        ("digits==4", lambda f: f["n_digits"] == 4, "结果为 4 位数")):
    p0 = share(base, pred)
    pB = share([FEAT[i] for i in pool_b], pred)
    pE = share([FEAT[i] for i in pool_all_err], pred)
    nb = len(pool_b)
    z = (pB - p0) / math.sqrt(p0 * (1 - p0) / nb) if nb and 0 < p0 < 1 else float("nan")
    DIST[key] = dict(name=name, base=p0, b=pB, all_err=pE, n_b=nb, z=z)
    say(f"[S52-DIFF] {name}: 基准(clean)={p0*100:.1f}% | 全部错例={pE*100:.1f}% | "
        f"★B类(数值错)={pB*100:.1f}% | n_B={nb} z={(z):+.1f}")

# ★去重口径：同一 test 样本可能在 4 个 ckpt 里各错一次 ⇒ 上面的 z 偏乐观；再按【样本】去重
uniq_b = sorted({i for v in VEC for i in v["b_idx"]})
DEID = {}
for key, pred, name in (("n_carry>=2", lambda f: f["n_carry"] >= 2, "进位次数≥2"),
                        ("chain>=2", lambda f: f["chain"] >= 2, "最长进位链≥2"),
                        ("digits==4", lambda f: f["n_digits"] == 4, "结果为 4 位数")):
    p0 = share(base, pred)
    pu = share([FEAT[i] for i in uniq_b], pred)
    nu = len(uniq_b)
    z = (pu - p0) / math.sqrt(p0 * (1 - p0) / nu) if nu and 0 < p0 < 1 else float("nan")
    DEID[key] = dict(name=name, base=p0, b_uniq=pu, n_uniq=nu, z=z)
    say(f"[S52-DIFF-DEID] {name}: 基准={p0*100:.1f}% | ★去重 B 类={pu*100:.1f}% | n_uniq={nu} z={z:+.1f}")
for v in VEC:
    p0c = share(base, lambda f: f["chain"] >= 2)
    pvc = share([FEAT[i] for i in v["b_idx"]], lambda f: f["chain"] >= 2)
    nvc = len(v["b_idx"])
    zc = (pvc - p0c) / math.sqrt(p0c * (1 - p0c) / nvc) if nvc and 0 < p0c < 1 else float("nan")
    say(f"[S52-DIFF-PER-CKPT] {v['label']}: 进位链≥2 基准={p0c*100:.1f}% B类={pvc*100:.1f}% "
        f"(n_B={nvc}) z={zc:+.1f}")
_rep = Counter(sum(1 for v in VEC if i in set(v["b_idx"])) for i in uniq_b)
say(f"[S52-DIFF-DEID] 去重后 {len(uniq_b)} 个错样本在 4 个 ckpt 中的复现次数分布: "
    f"{dict(sorted(_rep.items()))}")

m_base = sum(f["n_carry"] for f in base) / n_base
m_b = sum(FEAT[i]["n_carry"] for i in pool_b) / len(pool_b) if pool_b else float("nan")
m_e = sum(FEAT[i]["n_carry"] for i in pool_all_err) / len(pool_all_err) if pool_all_err else float("nan")
say(f"[S52-DIFF] 平均进位次数: 基准={m_base:.3f} | 全部错例={m_e:.3f} | ★B类={m_b:.3f}")

# 按进位次数分桶的 EM（每 ckpt + 汇总）
say("\n[S52-DIFF] 按进位次数分桶的 clean EM（看是否单调下降）:")
EM_BY_CARRY = {}
for f in range(4):
    sel = [i for i in CLEAN_IDX if FEAT[i]["n_carry"] == f]
    row = {}
    for v in VEC:
        n = len(sel)
        k = sum(v["hits"].get(i, 0) for i in sel)
        row[v["label"]] = (k / n if n else float("nan"), n)
    EM_BY_CARRY[f] = row
    say(f"   进位={f} (n={len(sel)}): " + " | ".join(
        f"{lb}={row[lb][0]*100:.1f}%" for lb in row))
say("[S52-DIFF] 按结果位数分桶的 clean EM:")
EM_BY_DIG = {}
for dgt in (3, 4):
    sel = [i for i in CLEAN_IDX if FEAT[i]["n_digits"] == dgt]
    row = {}
    for v in VEC:
        n = len(sel)
        k = sum(v["hits"].get(i, 0) for i in sel)
        row[v["label"]] = (k / n if n else float("nan"), n)
    EM_BY_DIG[dgt] = row
    say(f"   {dgt} 位数 (n={len(sel)}): " + " | ".join(
        f"{lb}={row[lb][0]*100:.1f}%" for lb in row))

# ============================================================================
# ⑦ 判据结论 + 落盘
# ============================================================================
say("\n[S52] ===== §4 ★ 判据 =====")
totA = sum(r["err_counts"]["A"] for r in RES)
totB = sum(r["err_counts"]["B"] for r in RES)
totC = sum(r["err_counts"]["C"] for r in RES)
totD = sum(r["err_counts"]["D"] for r in RES)
totE = sum(r["err_counts"]["E"] for r in RES)
tot = totA + totB + totC + totD + totE
say(f"[S52-VERDICT] 四 ckpt 合计错例={tot}: A={totA}({100*totA/tot:.1f}%) B={totB}({100*totB/tot:.1f}%) "
    f"C={totC}({100*totC/tot:.1f}%) D={totD}({100*totD/tot:.1f}%) E={totE}({100*totE/tot:.1f}%)")
say(f"[S52-VERDICT] ★ 数值错 B 占 {100*totB/tot:.1f}% vs 解析/格式错 A 占 {100*totA/tot:.1f}% ⇒ "
    f"{'模型顶（数值错为主）' if totB > totA else '口径顶（解析错为主）'}")
say(f"[S52-VERDICT] 非数值类(A+C+D+E)={100*(tot-totB)/tot:.1f}% ⇒ "
    f"{'数值错仍是主因 ⇒ 模型顶' if totB > (tot - totB) else '非数值类为主 ⇒ 口径/生成顶'}")
for r in RES:
    d = (r["unseen_full"]["em"] - r["full"]["em"]) * 100
    say(f"[S52-VERDICT] {r['label']}: Δ(UNSEEN−全量)={d:+.2f}pp vs 1SE={r['unseen_full']['se']*100:.2f}pp "
        f"⇒ {'UNSEEN≈全量 ✓（非记忆缺口）' if abs(d) <= r['unseen_full']['se']*100 else '★超 1SE，需注意'}")

out = dict(unit="S52", device=DEVICE, threads=NTHREADS, batch=EM_BATCH, bucket=BUCKET,
           n_clean=len(CLEAN_IDX), n_bad=len(BAD_IDX), n_test=len(TEST),
           data_sha=_h.hexdigest()[:16],
           split=dict(seen_full=len(SEEN_FULL), unseen_full=len(UNSEEN_FULL),
                      seen_clean=len(SEEN_CLEAN), unseen_clean=len(UNSEEN_CLEAN)),
           rows=[{k: v for k, v in r.items() if k != "b_idx"} for r in RES],
           error_totals=dict(A=totA, B=totB, C=totC, D=totD, E=totE, n=tot),
           diff=DIST, diff_deid=DEID, uniq_b=uniq_b, repeat_hist=dict(sorted(_rep.items())), mean_carry=dict(base=m_base, all_err=m_e, b=m_b),
           em_by_carry={str(k): {lb: v[0] for lb, v in row.items()} for k, row in EM_BY_CARRY.items()},
           em_by_digits={str(k): {lb: v[0] for lb, v in row.items()} for k, row in EM_BY_DIG.items()},
           b_idx_per_ckpt={v["label"]: v["b_idx"] for v in VEC},
           wall_s=time.time() - t_start)
json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
say(f"\n[S52] ★墙钟={time.time()-t_start:.1f}s | 写出 {OUT_LOG} 与 {OUT_JSON}")
_log_f.close()
