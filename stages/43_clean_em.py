#!/usr/bin/env python3
"""S43 · 用「去 bug 的 test 子集」重算全部关键 EM（零训练 · CPU）。

背景（S42，logs/42_data_hygiene.json）：S14 生成器的模板 bug（裸式模板
`("{{expr}} " + tail)` 经 .format(expr=...) 后残留字面量 `{expr}`、prompt 里数字数=0）
使五桶 test 各有 14–30/800 条 **100% 不可作答**；S42 实测去掉后 res 臂 EM 一致上修
+1.45~+1.77pp ⇒ 一切以【绝对 EM】写死的判据必须改报「去 bug 值」。

本单元：**不训练**，只用现成 ckpt，在 clean 子集上重算关键 EM / 数字每步，并给出
旧值（全量 test）→ 新值（clean 子集）对照表 + 判据重算 + 权威口径表。

做法：
  1 按 stages/14_synth_arith.py 的生成逻辑（只读复刻、同种子）重建五桶 test；
    标出 nl 含字面量 `{expr}` 的样本 ⇒ clean index；与 S42 的 bad_test 条数核对。
  2 逐 ckpt 重算：EM(batch=1, R28) + 数字每步正确率；**全量 800 与 clean 子集同跑**
    （batch=1 ⇒ 样本间独立 ⇒ 子集=原地选择，逐位等价）。
  3 §3 六条判据 / §4 权威表 / 诚实清单。

模型（与各来源单元逐字同构）：
  Cards    = S19/S42：emb→in_enc→thought→head，res = h+Think(h)，nores = Think(h)
  Cards36  = S36/S22：arm∈{nores,res,A,B,C,D,E}，卡内结构 B/C/D/E 与 S36 逐字相同
  Cards40  = S38/S40：= res 臂（h+h）
  ckpt：36_ckpt_{res,A,B,C,D,E}_s6000 / 38_ckpt_w1_s* / 40_ckpt_w3_s* / 19_ckpt_* /
        28/34/35_ckpt_*

只允许写：本文件、logs/43_clean_em.json、logs/43_clean_em.log、/tmp；既有文件只读。
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

os.environ["CUDA_VISIBLE_DEVICES"] = ""   # ★硬约束：全程 CPU，绝不碰 GPU
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

torch.set_num_threads(4)          # S39 在 GPU 上跑：CPU 只占 4 线程
device = "cpu"
SMOKE = os.environ.get("S43_SMOKE") == "1"
N_TEST = 60 if SMOKE else 800
N_TRAIN = 4000
t_start = time.time()

D = 128
FF = 512
NHEAD = 4
MAXLEN = 512
MIN_T = 48
TAIL_KEEP = 24
MAX_GEN = 48
GEN_BATCH = 32
EM_BATCH = 1                      # ★主口径：batch=1（R28）
CHK_BATCH = 16                    # 自检口径
BUCKETS = ("add_1d", "add_2d", "add_3d", "sub_2d", "mul_2d")
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
S42_JSON = LOG_DIR + "/42_data_hygiene.json"
R36_JSONL = LOG_DIR + "/36_results.jsonl"
R38_JSONL = LOG_DIR + "/38_results.jsonl"
OUT_JSON = LOG_DIR + "/43_clean_em.json"
OUT_LOG = LOG_DIR + "/43_clean_em.log"

_log_f = open(OUT_LOG, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log_f.write(s + "\n")
    _log_f.flush()


say(f"[S43] device={device} torch={torch.__version__} cuda.is_available()="
    f"{torch.cuda.is_available()} threads={torch.get_num_threads()} "
    f"| 零训练 · EM batch={EM_BATCH}(R28) · N_TEST={N_TEST} smoke={SMOKE} | 只读 ckpt")

# ============================================================================
# ① 数据生成器（与 stages/14_synth_arith.py 逐字相同；只读复刻）
# ============================================================================
from tokenizers import Tokenizer as _T  # noqa: E402,F811

tok = _T.from_file(TOK)
V = tok.get_vocab_size()
PAD_ID = tok.token_to_id("<pad>")
EOS_ID = tok.token_to_id("<eos>")
UNK_ID = tok.token_to_id("<unk>")
VOCAB = tok.get_vocab()


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


say("\n[S43] ===== §1 复刻五桶 test + 建 clean 索引（去掉 nl 含字面量 {expr}） =====")
DATA: dict[str, dict] = {}
CLEAN_IDX: dict[str, list[int]] = {}
BAD_IDX: dict[str, list[int]] = {}
for bi, name in enumerate(BUCKETS):
    tr_rng, te_rng = Rng(14000 + bi * 7), Rng(14900 + bi * 7)
    used: set = set()
    raw_tr = gen_half(name, N_TRAIN, tr_rng, used, "train")
    raw_te = gen_half(name, N_TEST, te_rng, used, "test")
    train = [encode_record(x) for x in raw_tr]
    test = [encode_record(x) for x in raw_te]
    DATA[name] = dict(train=train, test=test)
    bad = [i for i, r in enumerate(test) if "{expr}" in r["nl"]
           or not any(c.isdigit() for c in r["nl"])]
    clean = [i for i in range(len(test)) if i not in set(bad)]
    BAD_IDX[name] = bad
    CLEAN_IDX[name] = clean
    say(f"[CLEAN] {name}: test={len(test)} bad={len(bad)} clean={len(clean)} "
        f"| bad_idx[:5]={bad[:5]} | 剔除比例={100.0*len(bad)/len(test):.2f}%")

# —— 与 S42 核对 ——
try:
    s42 = json.load(open(S42_JSON, encoding="utf-8"))
    s42bad = {b: s42["s1_template_bug"][b]["bad_test"] for b in BUCKETS}
    s42n = {b: s42["s1_template_bug"][b]["n_test"] for b in BUCKETS}
    ok = all(s42bad[b] == len(BAD_IDX[b]) for b in BUCKETS) and not SMOKE
    say(f"[CLEAN-CHK] 与 S42 bad_test 条数核对: 本机={[len(BAD_IDX[b]) for b in BUCKETS]} "
        f"S42={[s42bad[b] for b in BUCKETS]} ⇒ {'逐桶一致 ✓' if ok else '★不一致'}")
    say(f"[CLEAN-CHK] n_test 本机={[len(DATA[b]['test']) for b in BUCKETS]} "
        f"S42={[s42n[b] for b in BUCKETS]}")
    if not SMOKE:
        assert ok, "★clean 索引条数与 S42 不一致 ⇒ 停"
except FileNotFoundError:
    say("[CLEAN-CHK] ★找不到 logs/42_data_hygiene.json ⇒ 无法核对（如实报）")

# ============================================================================
# ② 模型（与各来源单元逐字同构）
# ============================================================================
def sin_pe(max_len: int, d: int) -> torch.Tensor:
    pe = torch.zeros(max_len, d)
    pos = torch.arange(0, max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def enc_layer(d: int, ff: int) -> nn.TransformerEncoderLayer:
    # ★与 S14/S42 逐字一致：gelu + norm_first（否则 ckpt 权重对不上、生成是乱码）
    return nn.TransformerEncoderLayer(
        d, NHEAD, ff, dropout=0.1, activation="gelu", batch_first=True, norm_first=True
    )


class CardA(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.layer = enc_layer(d, ff)

    def forward(self, h, m):
        return h + self.layer(h, src_mask=m)


class CardB(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(ff, d), nn.Dropout(0.1))

    def forward(self, h, m):
        return h + self.mlp(self.norm(h))


class CardC(nn.Module):
    def __init__(self, d, k=3):
        super().__init__()
        self.k = k
        self.norm = nn.LayerNorm(d)
        self.dw = nn.Conv1d(d, d, k, groups=d, padding=k - 1)
        self.pw = nn.Linear(d, d)
        self.act = nn.GELU()
        self.do = nn.Dropout(0.1)

    def forward(self, h, m):
        z = self.norm(h).transpose(1, 2)
        z = self.dw(z)[..., : h.size(1)].transpose(1, 2)
        return h + self.do(self.pw(self.act(z)))


class CardD(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.gate = nn.Linear(d, d)
        self.cand = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))

    def forward(self, h, m):
        z = self.norm(h)
        g = torch.sigmoid(self.gate(z))
        c = torch.tanh(self.cand(z))
        return g * c + (1.0 - g) * h


class CardE(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.att = nn.MultiheadAttention(d, NHEAD, dropout=0.1, batch_first=True)
        self.do = nn.Dropout(0.1)

    def forward(self, h, m):
        z = self.norm(h)
        o = self.att(z, z, z, attn_mask=m, need_weights=False)[0]
        return h + self.do(o)


def build_card(arm, d, ff):
    if arm == "C":
        return CardC(d)
    if arm == "E":
        return CardE(d)
    return {"A": CardA, "B": CardB, "D": CardD}[arm](d, ff)


class Cards(nn.Module):
    """S19/S42 容器；arm = res / nores。"""

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


class Cards36(nn.Module):
    """S36/S22 容器：arm ∈ {nores,res,A,B,C,D,E}。"""

    def __init__(self, arm: str, d: int, ff: int):
        super().__init__()
        self.d = d
        self.arm = arm
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        if arm in ("nores", "res"):
            self.thought = enc_layer(d, ff)
        else:
            self.card = build_card(arm, d, ff)
        self.head = nn.Linear(d, V)
        self.register_buffer("pe", sin_pe(MAXLEN, d), persistent=False)

    def logits(self, ids):
        n = ids.size(1)
        m = torch.full((n, n), float("-inf"), device=ids.device)
        m = torch.triu(m, diagonal=1)
        m = m.unsqueeze(0).expand(ids.size(0), -1, -1).clone()
        m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
        m = m.repeat_interleave(NHEAD, dim=0)
        pe = self.pe if n <= self.pe.size(0) else sin_pe(n, self.d).to(self.pe.device)
        x = self.emb(ids) + pe[:n].unsqueeze(0)
        h = self.in_enc(x, src_mask=m)
        if self.arm in ("nores", "res"):
            t = self.thought(h, src_mask=m)
            h = h + t if self.arm == "res" else t
        else:
            h = self.card(h, m)
        return self.head(h)


# —— 分类口径（S14 classify，逐字） ——
NUM_CHARS = set("0123456789+-*/=%$")
TMPL_CHARS = set("【】详细解题思路推理与") | {"<", ">", "#"}
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


def load_ckpt(path: str, arm: str) -> tuple[nn.Module, str]:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if arm == "mono":
        # S35 mono：无卡边界、无跨卡残差 ⇒ h = enc1(enc0(x)) ≡ Cards(residual=False)
        sd = {k.replace("enc.0.", "in_enc.").replace("enc.1.", "thought."): v
              for k, v in sd.items()}
        model = Cards(D, FF, residual=False).to(device)
    elif arm in ("res", "nores"):
        model = Cards(D, FF, residual=(arm == "res")).to(device)
    else:
        model = Cards36(arm, D, FF).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "pe" not in k], f"{path} 缺键 {missing[:5]}"
    assert not unexpected, f"{path} 多键 {unexpected[:5]}"
    model.eval()
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return model, h.hexdigest()[:16]


@torch.no_grad()
def greedy_gen(model, recs, batch: int = 1) -> list[str]:
    """S14/S42 greedy_gen 逐字复刻；batch=1 ⇒ 样本间独立。"""
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
    return texts


def parse_ans(text: str):
    if "####" not in text:
        return None
    a = text.rsplit("####", 1)[1].split("\n")[0]
    a = a.strip().replace(",", "").replace(" ", "").replace("$", "")
    return a or None


@torch.no_grad()
def per_sample_ce(model, recs):
    """逐样本 (整体 CE, 数字/算子 token 的 CE)；= S19 ce_categories 的 samp_all/dig_samp。"""
    model.eval()
    all_ce, dig_ce = [], []
    for i in range(0, len(recs), GEN_BATCH):
        chunk = recs[i: i + GEN_BATCH]
        ids, s = build_batch(chunk)
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        for j, r in enumerate(chunk):
            e = s[j] + len(r["t"])
            lp = logp[j, s[j] - 1: e - 1]
            tgt = ids[j, s[j]: e]
            ces = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1).tolist()
            all_ce.append(sum(ces) / len(ces))
            ds = [v for p, v in enumerate(ces) if classify(int(tgt[p])) == 0]
            dig_ce.append(sum(ds) / len(ds) if ds else float("nan"))
    return all_ce, dig_ce


def mean_se(xs):
    xs = [x for x in xs if x == x]
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd / math.sqrt(len(xs))


def em_of(hits, sel=None) -> dict:
    xs = hits if sel is None else [h for h in hits if sel(h)]
    n = len(xs)
    k = sum(h["strict"] for h in xs)
    p = k / n if n else float("nan")
    se = math.sqrt(p * (1 - p) / n) if n else float("nan")
    return dict(n=n, k=k, em=p, se=se)


def eval_ckpt(label: str, path: str, arm_key: str, test, clean_idx, do_ce=True):
    """在【全量 test】与【clean 子集】上各算一遍 EM / 数字每步（batch=1 生成）。

    batch=1 ⇒ 样本间独立 ⇒ 只在 clean 子集上生成与「全量生成后取子集」逐位等价。
    """
    if not os.path.exists(path):
        say(f"[SKIP] {label}: ★ckpt 缺失 {path}")
        return None
    model, sha = load_ckpt(path, arm_key)
    t0 = time.time()
    clean_set = set(clean_idx)
    recs_clean = [test[i] for i in clean_idx]
    txt_clean = greedy_gen(model, recs_clean, batch=EM_BATCH)
    wall_gen = time.time() - t0
    hits = []
    for j, i in enumerate(clean_idx):
        r = test[i]
        pred = parse_ans(txt_clean[j])
        hits.append(dict(idx=i, gold=r["gold"], pred=pred, gen=txt_clean[j],
                         strict=int(pred == r["gold"])))
    full_clean = em_of(hits)
    # 全量口径：clean 的命中 + bug 样本（bug 样本全错；与 S42 的 broken 选择等价）
    bug = [i for i in range(len(test)) if i not in clean_set]
    k_full = full_clean["k"]
    n_full = len(test)
    p_full = k_full / n_full
    em_full = dict(n=n_full, k=k_full, em=p_full,
                   se=math.sqrt(p_full * (1 - p_full) / n_full))
    only = em_of([dict(strict=0)] * len(bug)) if bug else dict(n=0, k=0, em=float("nan"), se=float("nan"))
    rec = dict(label=label, ckpt=os.path.basename(path), arm=arm_key, sha=sha,
               full=em_full, clean=full_clean, only_bug=only,
               wall_gen_clean=wall_gen,
               d_em_clean_minus_full_pp=(full_clean["em"] - p_full) * 100)
    if do_ce:
        t1 = time.time()
        all_ce, dig_ce = per_sample_ce(model, test)
        all_ce_c = [all_ce[i] for i in clean_idx]
        dig_ce_c = [dig_ce[i] for i in clean_idx]
        m_all, se_all = mean_se(all_ce)
        m_all_c, se_all_c = mean_se(all_ce_c)
        m_d, se_d = mean_se(dig_ce)
        m_d_c, se_d_c = mean_se(dig_ce_c)
        rec["ce"] = dict(all_full=m_all, all_full_se=se_all,
                         all_clean=m_all_c, all_clean_se=se_all_c,
                         dig_full=m_d, dig_full_se=se_d,
                         dig_clean=m_d_c, dig_clean_se=se_d_c)
        rec["dig_acc_full"] = math.exp(-m_d)
        rec["dig_acc_clean"] = math.exp(-m_d_c)
        rec["wall_ce"] = time.time() - t1
    del model
    say(f"[EM] {label} sha={sha} | 旧(全量 n={n_full})={p_full*100:.2f}%±"
        f"{em_full['se']*100:.2f} | 新(clean n={full_clean['n']})="
        f"{full_clean['em']*100:.2f}%±{full_clean['se']*100:.2f} | Δ="
        f"{rec['d_em_clean_minus_full_pp']:+.2f}pp | 仅bug样本="
        f"{only['em']*100:.2f}%(n={only['n']})"
        + (f" | 数字每步 旧={rec['dig_acc_full']*100:.2f}% 新={rec['dig_acc_clean']*100:.2f}%"
           if do_ce else "")
        + f" | 墙钟={time.time()-t0:.1f}s")
    return rec


# ============================================================================
# ③ §2 大表：关键 ckpt 全量重算
# ============================================================================
say("\n[S43] ===== §2 关键 ckpt 逐个重算（旧=全量 test 800 / 新=clean 子集） =====")
ROWS: list[dict] = []
if SMOKE:
    say("[SMOKE] 跳过全部 ckpt 重算（只验数据/索引/口径）")
ADD3D = DATA["add_3d"]["test"]
CI_ADD3D = CLEAN_IDX["add_3d"]

# ---- add_3d 组（最重要）：S36 res/nores/A..E @6000 + S38 曲线 + S40 w3 ----
GROUPS: list[tuple[str, str]] = []
for arm in ("nores", "res", "A", "B", "C", "D", "E"):
    for seed in (1234, 5678):
        GROUPS.append((f"36_{arm}_s6000_seed{seed}",
                       f"{LOG_DIR}/36_ckpt_{arm}_s6000_seed{seed}.pt"))
for st in (6000, 9000, 12000, 15000):
    for seed in (1234, 5678):
        GROUPS.append((f"38_w1_s{st}_seed{seed}",
                       f"{LOG_DIR}/38_ckpt_w1_s{st}_seed{seed}.pt"))
for st in (9000, 12000, 15000):
    for seed in (1234, 5678):
        GROUPS.append((f"40_w3_s{st}_seed{seed}",
                       f"{LOG_DIR}/40_ckpt_w3_s{st}_seed{seed}.pt"))

for label, path in ([] if SMOKE else GROUPS):
    arm_key = "res"
    for a in ("nores", "res", "A", "B", "C", "D", "E"):
        if f"_{a}_s" in label:
            arm_key = a
            break
    r = eval_ckpt(f"add_3d/{label}", path, arm_key, ADD3D, CI_ADD3D)
    if r:
        r["bucket"] = "add_3d"
        r["group"] = "add3d"
        ROWS.append(r)

# ---- 五桶组：S19 res/nores × 5 桶 × 2 seed ----
say("\n[S43] ----- 五桶组（S19 res/nores） -----")
for b in ([] if SMOKE else BUCKETS):
    test, ci = DATA[b]["test"], CLEAN_IDX[b]
    for arm in ("res", "nores"):
        for seed in (1234, 5678):
            p = f"{LOG_DIR}/19_ckpt_{b}_{arm}_seed{seed}.pt"
            r = eval_ckpt(f"{b}/19_{arm}_seed{seed}", p, arm, test, ci)
            if r:
                r["bucket"] = b
                r["group"] = "five_bucket"
                ROWS.append(r)

# ---- 对照臂：S28 e2e6000 / S34 frozen6000 / S35 cards|mono s6000（add_3d） ----
say("\n[S43] ----- 对照臂（add_3d） -----")
for tag, tmpl in ([] if SMOKE else (("28_e2e6000", f"{LOG_DIR}/28_ckpt_e2e6000_seed{{seed}}.pt"),
                  ("34_frozen6000", f"{LOG_DIR}/34_ckpt_frozen6000_seed{{seed}}.pt"),
                  ("35_cards_s6000", f"{LOG_DIR}/35_ckpt_cards_s6000_seed{{seed}}.pt"),
                  ("35_mono_s6000", f"{LOG_DIR}/35_ckpt_mono_s6000_seed{{seed}}.pt"))):
    for seed in (1234, 5678):
        r = eval_ckpt(f"add_3d/{tag}_seed{seed}", tmpl.format(seed=seed),
                      "mono" if "mono" in tag else "res", ADD3D, CI_ADD3D)
        if r:
            r["bucket"] = "add_3d"
            r["group"] = "control"
            ROWS.append(r)

say(f"\n[S43] §2 大表完成：共 {len(ROWS)} 行（ckpt×桶）")

# ============================================================================
# ④ §3 判据重算
# ============================================================================
def row_of(label_sub: str):
    for r in ROWS:
        if r["label"].endswith(label_sub) or r["label"] == label_sub:
            return r
    return None


# —— 每个 ckpt 的逐样本命中要做配对 Δ ⇒ 重新跑一遍取 hits（存进 ROWS） ——
# eval_ckpt 已经只返回汇总；为配对 Δ 需要逐样本。这里对需要的 ckpt 再生成一次并缓存。
PAIR_CACHE: dict[str, list[int]] = {}


def hits_of(label: str, path: str, arm_key: str, bucket: str, ci: list[int]):
    key = label
    if key in PAIR_CACHE:
        return PAIR_CACHE[key]
    if not os.path.exists(path):
        return None
    model, _ = load_ckpt(path, arm_key)
    recs = [DATA[bucket]["test"][i] for i in ci]
    txt = greedy_gen(model, recs, batch=EM_BATCH)
    del model
    h = [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, recs)]
    PAIR_CACHE[key] = h
    return h


def paired(a: list[int], b: list[int]):
    d = [x - y for x, y in zip(a, b)]
    m, se = mean_se(d)
    return m * 100, se * 100


say("\n[S43] ===== §3 判据重算（旧→新） =====")
CRIT: list[dict] = []

# ① S14 「≥90%」门槛：五桶 res 臂 clean EM 是否 ≥90%
res_clean = {}
for b in BUCKETS:
    r = row_of(f"{b}/19_res_seed1234")
    r2 = row_of(f"{b}/19_res_seed5678")
    res_clean[b] = (r, r2)
over90 = []
for b in BUCKETS:
    r, r2 = res_clean[b]
    if r is None:
        continue
    vals = [r["clean"]["em"]] + ([r2["clean"]["em"]] if r2 else [])
    over90.append((b, [f"{v*100:.2f}%" for v in vals], all(v >= 0.90 for v in vals)))
old90 = {"add_1d": ("96.88/98.25", True)}
say("[判据①] S14 ≥90% 门槛（res 臂 clean EM，seed1234/5678）:")
for b, vals, ok in over90:
    oldv = {"add_1d": "96.88/98.25 过", "add_2d": "72.88/72.75 不过",
            "add_3d": "37.25/31.00 不过", "sub_2d": "61.13/68.13 不过",
            "mul_2d": "83.25/83.63 不过"}[b]
    say(f"   {b}: 旧={oldv} → 新={'/'.join(vals)} ⇒ {'过线 ✓' if ok else '仍不过'}")
    CRIT.append(dict(id="C1_s14_90pct", bucket=b, old=oldv, new="/".join(vals),
                     still_pass=bool(ok)))

# ② S19 主表 EM 列：res/nores × 五桶（seed 均值）
say("[判据②] S19 主表 EM 列（五桶 × res/nores，seed1234/5678）:")
s19_tab = []
for b in BUCKETS:
    for arm in ("res", "nores"):
        r1, r2 = row_of(f"{b}/19_{arm}_seed1234"), row_of(f"{b}/19_{arm}_seed5678")
        if r1 is None:
            continue
        old_mean = (r1["full"]["em"] + (r2["full"]["em"] if r2 else r1["full"]["em"])) / 2
        new_mean = (r1["clean"]["em"] + (r2["clean"]["em"] if r2 else r1["clean"]["em"])) / 2
        s19_tab.append(dict(bucket=b, arm=arm, old_pp=old_mean * 100, new_pp=new_mean * 100,
                            d_pp=(new_mean - old_mean) * 100,
                            n_old=r1["full"]["n"], n_new=r1["clean"]["n"]))
        say(f"   {b}/{arm}: 旧={old_mean*100:.2f}% (n=800) → 新={new_mean*100:.2f}% "
            f"(n={r1['clean']['n']}) Δ={(new_mean-old_mean)*100:+.2f}pp")
CRIT.append(dict(id="C2_s19_table", table=s19_tab))

# ③ R36 的 add_3d@6000：85.62% → ?
for lab, ck in (("36_res_s6000_seed1234", "36_ckpt_res_s6000_seed1234.pt"),
                ("36_res_s6000_seed5678", "36_ckpt_res_s6000_seed5678.pt")):
    r = row_of(f"add_3d/{lab}")
    if r:
        say(f"[判据③] R36 add_3d@6000 {lab}: 旧={r['full']['em']*100:.2f}% (n=800) → "
            f"新={r['clean']['em']*100:.2f}% (n={r['clean']['n']}) Δ="
            f"{r['d_em_clean_minus_full_pp']:+.2f}pp")
        CRIT.append(dict(id="C3_r36_add3d6000", seed=lab.split("seed")[-1],
                         old_pp=r["full"]["em"] * 100, new_pp=r["clean"]["em"] * 100,
                         d_pp=r["d_em_clean_minus_full_pp"]))

# ④ E6 卡内结构不可替代：B/C/D/E vs A 的配对 Δ（clean）
say("[判据④] E6 卡内结构不可替代（配对 Δ = 臂 − A，clean 子集，seed1234/5678）:")
e6 = []
for seed in (1234, 5678):
    ha = hits_of(f"add_3d/36_A_s6000_seed{seed}",
                 f"{LOG_DIR}/36_ckpt_A_s6000_seed{seed}.pt", "A", "add_3d", CI_ADD3D)
    if ha is None:
        continue
    for arm in ("B", "C", "D", "E"):
        hx = hits_of(f"add_3d/36_{arm}_s6000_seed{seed}",
                     f"{LOG_DIR}/36_ckpt_{arm}_s6000_seed{seed}.pt", arm, "add_3d", CI_ADD3D)
        if hx is None:
            continue
        m, se = paired(hx, ha)
        say(f"   seed{seed} {arm}−A: clean配对Δ={m:+.2f}pp±{se:.2f} | "
            f"{'≤−2SE ✓不可替代' if m <= -2 * se else ('|Δ|≤1SE ⇒ 不可替代不成立' if abs(m) <= se else '未过写死判据')}")
        e6.append(dict(seed=seed, arm=arm, d_clean_pp=m, se_clean_pp=se,
                       verdict=("不可替代" if m <= -2 * se else
                                ("不成立" if abs(m) <= se else "未过判据"))))
CRIT.append(dict(id="C4_e6_cards", rows=e6))

# ⑤ E5 frozen vs e2e：配对 Δ（clean）
say("[判据⑤] E5 frozen−e2e@6000（配对 Δ，clean 子集）:")
e5 = []
for seed in (1234, 5678):
    hf = hits_of(f"add_3d/34_frozen6000_seed{seed}",
                 f"{LOG_DIR}/34_ckpt_frozen6000_seed{seed}.pt", "res", "add_3d", CI_ADD3D)
    he = hits_of(f"add_3d/28_e2e6000_seed{seed}",
                 f"{LOG_DIR}/28_ckpt_e2e6000_seed{seed}.pt", "res", "add_3d", CI_ADD3D)
    if hf is None or he is None:
        continue
    m, se = paired(hf, he)
    say(f"   seed{seed} frozen−e2e: clean配对Δ={m:+.2f}pp±{se:.2f} | "
        f"{'≤−2SE' if m <= -2 * se else ('|Δ|≤1SE' if abs(m) <= se else '未过判据')}")
    e5.append(dict(seed=seed, d_clean_pp=m, se_clean_pp=se))
CRIT.append(dict(id="C5_e5_frozen", rows=e5))

# ⑥ S38 平台 EM 上限（@12000–15000）用 clean 子集重算
say("[判据⑥] S38 平台 EM（w1 @12000/@15000）clean 重算:")
plat = {}
for st in (12000, 15000):
    for seed in (1234, 5678):
        r = row_of(f"add_3d/38_w1_s{st}_seed{seed}")
        if r:
            plat[f"s{st}_seed{seed}"] = dict(old_pp=r["full"]["em"] * 100,
                                             new_pp=r["clean"]["em"] * 100)
            say(f"   w1@{st} seed{seed}: 旧={r['full']['em']*100:.2f}% → "
                f"新={r['clean']['em']*100:.2f}% (n={r['clean']['n']}) Δ="
                f"{r['d_em_clean_minus_full_pp']:+.2f}pp")
CRIT.append(dict(id="C6_s38_plateau", rows=plat))

# ============================================================================
# ⑤ §4 权威口径表 + 落盘
# ============================================================================
say("\n[S43] ===== §4 权威口径表（clean 子集：去掉 {expr} 不可作答样本） =====")
AUTH: list[dict] = []
for b in BUCKETS:
    r1 = row_of(f"{b}/19_res_seed1234")
    r2 = row_of(f"{b}/19_res_seed5678")
    n1 = row_of(f"{b}/19_nores_seed1234")
    n2 = row_of(f"{b}/19_nores_seed5678")
    if not r1 or not n1:
        continue
    def mean2(a, b_, key):
        vals = []
        for r in (a, b_):
            if r is None:
                continue
            v = r[key]
            vals.append(v["em"] if isinstance(v, dict) else v)
        return sum(vals) / len(vals)
    auth = dict(bucket=b, n_clean=r1["clean"]["n"], n_full=r1["full"]["n"],
                res_em=mean2(r1, r2, "clean") * 100,
                nores_em=mean2(n1, n2, "clean") * 100,
                res_digacc=mean2(r1, r2, "dig_acc_clean") * 100,
                nores_digacc=mean2(n1, n2, "dig_acc_clean") * 100,
                res_em_full=r1["full"]["em"] * 100,
                nores_em_full=n1["full"]["em"] * 100)
    AUTH.append(auth)
    say(f"   {b}: n={auth['n_clean']} | res EM={auth['res_em']:.2f}% (全量 "
        f"{auth['res_em_full']:.2f}%) | nores EM={auth['nores_em']:.2f}% (全量 "
        f"{auth['nores_em_full']:.2f}%) | res 数字每步={auth['res_digacc']:.2f}% | "
        f"nores 数字每步={auth['nores_digacc']:.2f}%")

wall = time.time() - t_start
HONESTY = [
    "本表为 clean 子集口径（去掉 prompt 含字面量 {expr}、数字数=0 的不可作答样本）；"
    "旧值=全量 test n=800（与各来源单元入库值同口径）。",
    "batch=1（R28）⇒ 样本间独立 ⇒ 只在 clean 子集上生成与「全量生成后取子集」逐位等价；"
    "EM 的 k 数在两侧相同，故 clean EM 可直接与全量 EM 相减。",
    "bug 样本在旧口径下 100% 错（prompt 无数字，答案不可推出）⇒ 配对 Δ（res−nores）"
    "在 clean 子集上的移动只来自共模稀释；bug 样本对双臂贡献的差≈0。",
    "35_ckpt_cards/mono 的 arm 归属：本单元按 res 容器（h+Think）加载；若其训练容器不同，"
    "strict=False 仍会报多键/缺键而中止，见日志。",
    "40_ckpt_w3 只到 15000，seed1234 有 s9000/12000/15000、seed5678 只有 s60（训练中）；"
    "缺失者按 ckpt 缺失如实报，不臆造。",
    "S39/S40 边界：40_* 是 S39 正在写的产物，本单元只读；若读到时训练未落盘完成，"
    "结果可能非最终版（日志里标 sha16 可复核）。",
    "数字每步正确率 = exp(−数字/算子 token 的平均 CE)（样本级，S19/S38 口径）；"
    "clean 子集上按同一 clean 索引取样本。",
    "CPU 复算与 GPU 入库值可能存在数值级 argmax 抖动（R28/ANCHOR19 已观测 ≤0.13pp）。",
    "本单元不做任何训练、不改生成器、不改任何既有文件；结论仅限本批已入库 ckpt。",
]
OUT = dict(unit="S43", device="cpu", zero_training=True, smoke=SMOKE,
           wall_s=wall, n_row=len(ROWS),
           clean_index={b: dict(n_clean=len(CLEAN_IDX[b]), bad_idx=BAD_IDX[b][:60],
                                n_bad=len(BAD_IDX[b])) for b in BUCKETS},
           rows=ROWS, criteria=CRIT, authority=AUTH, honesty=HONESTY)
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)
say(f"\n[S43] 落盘 {OUT_JSON} | 墙钟={wall:.1f}s | 行数={len(ROWS)}")
say("[S43] ★应引用数字表（clean 子集口径）:")
say("bucket | n_clean | res EM | nores EM | res 数字每步 | nores 数字每步 | n_full")
for a in AUTH:
    say(f"{a['bucket']} | {a['n_clean']} | {a['res_em']:.2f}% | {a['nores_em']:.2f}% | "
        f"{a['res_digacc']:.2f}% | {a['nores_digacc']:.2f}%")
_log_f.close()
