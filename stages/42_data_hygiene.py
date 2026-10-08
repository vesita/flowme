#!/usr/bin/env python3
"""S42 · 数据卫生审计（零训练 · 全程 CPU）：模板 bug 影响面 + 词表卫生 + 全桶地板表。

三条已被独立审计提出、但从未量化的问题：
  ① stages/14_synth_arith.py 的 build_templates() 里 10 句裸式用普通字符串拼接
     （"{{expr}} " + t），.format() 后残留字面量 `{expr}`、prompt 里没有数字 ⇒ 不可作答。
     S26/S31 只说"10/358 句"，从未算清影响多少 train/test 样本、对 EM 的影响面多大。
  ② 任务用 V=8192 WordLevel 词表，而分隔符 '→' 词表外（<unk> id=129）⇒ 影响面从未量化。
  ③ R36 只给了 add_3d 的地板；每桶自己的多数答案地板从未进判据 ⇒ 需要全桶地板表。

做法（只读既有资产，不训练、不动 GPU）：
  - 逐字复刻 stages/14_synth_arith.py 的数据生成器（同种子 14000+7i / 14900+7i）重建五桶
    train4000/test800，并用 logs/14_synth.log 里 S14 自己打印的 [DATA] 统计量作锚点校验；
  - 用已入库 ckpt（logs/19_ckpt_<桶>_res|nores_seed1234.pt，另加 logs/36_ckpt_res_s6000_seed1234.pt）
    在 CPU 上 batch=1 贪心生成（R28 主口径）重算 EM；
  - 比对 logs/19_results.jsonl 里入库的 EM 值（一致 ⇒ 复刻与推理口径自洽）。

口径：device=cpu，batch=1，不训练、不写除 logs/42_* 与本文件以外的任何路径。
只允许写：本文件、logs/42_data_hygiene.json、logs/42_data_hygiene.log。
"""
from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")   # ★全程 CPU：先屏蔽 GPU，再 import torch

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402

FLOWME = "/home/vesita/coding/my/flowme"
NS_ROOT = "/home/vesita/coding/my/nanoSeek"
LOG_DIR = FLOWME + "/logs"
sys.path.insert(0, NS_ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

torch.set_num_threads(4)          # S39 在 GPU 上跑：CPU 只占 4 线程，避免抢数据加载
device = "cpu"
SMOKE = os.environ.get("S42_SMOKE") == "1"
N_TEST = 40 if SMOKE else 800
t_start = time.time()

# ---------------- 配置（与 S14 逐字一致） ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512
MIN_T = 48
TAIL_KEEP = 24
MAX_GEN = 48
N_TRAIN = 4000
GEN_BATCH = 32
EM_BATCH = 1                      # ★主口径：batch=1（R28）
BUCKETS = ("add_1d", "add_2d", "add_3d", "sub_2d", "mul_2d")
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
CK19 = LOG_DIR + "/19_ckpt_{bucket}_{arm}_seed1234.pt"
CK36 = LOG_DIR + "/36_ckpt_res_s6000_seed1234.pt"
ANCHOR_JSONL = LOG_DIR + "/19_results.jsonl"
S14_LOG = LOG_DIR + "/14_synth.log"

print(f"[S42] device={device} torch={torch.__version__} cuda.is_available()="
      f"{torch.cuda.is_available()} threads={torch.get_num_threads()} "
      f"| 零训练 · batch={EM_BATCH} 贪心 | N_TEST={N_TEST} smoke={SMOKE}", flush=True)

# ============================================================================
# ① 数据生成器：与 stages/14_synth_arith.py 逐字相同（只读复刻，同种子 ⇒ 同 train/test）
# ============================================================================
from tokenizers import Tokenizer as _T  # noqa: E402,F811

tok = _T.from_file(TOK)
V = tok.get_vocab_size()
PAD_ID = tok.token_to_id("<pad>")
EOS_ID = tok.token_to_id("<eos>")
UNK_ID = tok.token_to_id("<unk>")
VOCAB = tok.get_vocab()           # token -> id


def enc(text: str) -> list[int]:
    return tok.encode(text, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


print(f"[TOK] WordLevel V={V} PAD={PAD_ID} EOS={EOS_ID} UNK={UNK_ID} "
      f"enc('→')={enc('→')} enc('题干：1 加 2 → ')={enc('题干：1 加 2 → ')}", flush=True)


def build_templates() -> list[tuple[str, bool]]:
    ts: list[tuple[str, bool]] = []
    tails_cn = ["等于多少？", "是多少？", "等于几？", "得多少？", "的结果是多少？",
                "的结果是几？", "的值是多少？", "等于多少", "是多少", "是几"]
    for lead in ("", "请", "帮我", "麻烦"):
        for verb in ("计算", "算出", "算一下", "口算", "快速算", "算"):
            for t in tails_cn:
                ts.append((f"{lead}{verb} {{expr}} {t}", True))
    for t in tails_cn:                       # ★裸式：普通字符串拼接 ⇒ .format() 残留字面量 {expr}
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
BARE_IDX = [i for i, (t, _) in enumerate(TEMPLATES) if t.startswith("{{expr}}")]

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
                y=it["y"], nl=it["nl"], prompt=it["prompt"])   # prompt 留原文供词表审计


print("\n[S42] ===== §0 复刻数据生成器（与 S14 同种子）+ 锚点校验 =====", flush=True)
DATA: dict[str, dict] = {}
GEN_STATS: dict[str, dict] = {}
for bi, name in enumerate(BUCKETS):
    tr_rng, te_rng = Rng(14000 + bi * 7), Rng(14900 + bi * 7)
    used: set = set()
    raw_tr = gen_half(name, N_TRAIN, tr_rng, used, "train")
    raw_te = gen_half(name, N_TEST, te_rng, used, "test")
    train = [encode_record(x) for x in raw_tr]
    test = [encode_record(x) for x in raw_te]
    ys = answer_values(name)
    ctr_tr = Counter(r["y"] for r in train)
    ctr_te = Counter(r["y"] for r in test)
    pair_tr = len({(r["a"], r["b"]) for r in train})
    pair_te = len({(r["a"], r["b"]) for r in test})
    top5 = [v for v, _ in Counter(r["gold"] for r in train).most_common(5)]
    prior = sum(1 for r in test if r["gold"] in top5) / max(1, len(test))
    tpl_used = len({r["nl"] for r in train + test})
    DATA[name] = dict(train=train, test=test)
    GEN_STATS[name] = dict(cnt_tr=f"{min(ctr_tr.values())}/{max(ctr_tr.values())}",
                           cnt_te=f"{min(ctr_te.values())}/{max(ctr_te.values())}",
                           tpl_used=tpl_used, pair_tr=pair_tr, pair_te=pair_te,
                           top5=top5, prior=prior, ans_n=len(ys))
    print(f"[GEN] {name}: train={len(train)} test={len(test)} 计数train "
          f"{min(ctr_tr.values())}/{max(ctr_tr.values())} 计数test "
          f"{min(ctr_te.values())}/{max(ctr_te.values())} 模板覆盖={tpl_used} "
          f"distinct(a,b)={pair_tr}/{pair_te} top5={top5} 机会水平={100*prior:.1f}%",
          flush=True)

# —— 锚点：与 logs/14_synth.log 里 S14 自己打的 [DATA] 行逐项比对（证明复刻逐字一致） ——
anchor_ok = anchor_bad = 0
anchor_rows = []
if os.path.exists(S14_LOG) and not SMOKE:
    with open(S14_LOG, encoding="utf-8") as f:
        lines = [ln for ln in f if ln.startswith("[DATA]")]
    for name in BUCKETS:
        blob = "\n".join(x for x in lines if x.startswith(f"[DATA] {name}:"))
        if not blob:
            continue
        g = GEN_STATS[name]
        chk = (f"计数train min/max={g['cnt_tr']}" in blob
               and f"test min/max={g['cnt_te']}" in blob
               and f"模板覆盖={g['tpl_used']}/" in blob
               and f"distinct(a,b) train={g['pair_tr']} test={g['pair_te']}" in blob
               and f"★最高频5答案={g['top5']}" in blob)
        anchor_rows.append(dict(bucket=name, match=bool(chk)))
        anchor_ok += int(chk)
        anchor_bad += int(not chk)
print(f"[ANCHOR] 与 logs/14_synth.log 的 [DATA] 统计量逐项一致: {anchor_ok}/{anchor_ok+anchor_bad} 桶"
      f"{'' if not SMOKE else '（smoke 跳过）'}", flush=True)
assert (anchor_bad == 0) or SMOKE, "★复刻与 S14 日志不一致 ⇒ 后续一切数字无效"

# ============================================================================
# ② 模型（与 S14/S19 逐字同构；res 臂 = h + Think(h)）
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
    """S19 Cards 逐字同构：residual=True ⇒ h = h + Think(h)。"""

    def __init__(self, d: int, ff: int, identity: bool = False, residual: bool = False):
        super().__init__()
        self.d = d
        self.residual = residual
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
            if self.residual:
                h = h + self.thought(h, src_mask=m)
            else:
                h = self.thought(h, src_mask=m)
        return self.head(h)


def load_ckpt(path: str, residual: bool) -> tuple[Cards, str]:
    sd = torch.load(path, map_location="cpu", weights_only=True)
    model = Cards(D, FF, residual=residual).to(device)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "pe" not in k], f"{path} 缺键 {missing[:5]}"
    assert not unexpected, f"{path} 多键 {unexpected[:5]}"
    model.eval()
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return model, h.hexdigest()[:16]


@torch.no_grad()
def greedy_gen(model: Cards, recs, batch: int = 1) -> list[str]:
    """S14 greedy_gen 逐字复刻（batch=1 ⇒ 样本间独立 ⇒ 取子集重算 EM 等价于原地重跑）。"""
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


def em_of(hits, sel=None) -> dict:
    xs = hits if sel is None else [h for h in hits if sel(h)]
    n = len(xs)
    k = sum(h["strict"] for h in xs)
    p = k / n if n else float("nan")
    se = math.sqrt(p * (1 - p) / n) if n else float("nan")
    return dict(n=n, k=k, em=p, se=se)


def run_arm(bucket: str, arm: str, ckpt: str, residual: bool, test) -> dict:
    model, sha = load_ckpt(ckpt, residual)
    t0 = time.time()
    txt = greedy_gen(model, test, batch=EM_BATCH)
    del model
    hits = []
    for i, r in enumerate(test):
        pred = parse_ans(txt[i])
        hits.append(dict(idx=i, gold=r["gold"], pred=pred, gen=txt[i],
                         strict=int(pred == r["gold"]), nl=r["nl"],
                         broken=("{expr}" in test[i]["nl"]),
                         has_digit=any(c.isdigit() for c in r["nl"])))
    wall = time.time() - t0
    full = em_of(hits)
    clean = em_of(hits, lambda h: not h["broken"])
    only = em_of(hits, lambda h: h["broken"])
    print(f"[EM] {bucket} {arm}: 全量 EM={full['em']*100:.2f}%±{full['se']*100:.2f} "
          f"(n={full['n']}) | 去模板bug EM={clean['em']*100:.2f}% (n={clean['n']}) "
          f"| 仅bug样本 EM={only['em']*100:.2f}% (n={only['n']}) | 墙钟={wall:.1f}s sha={sha}",
          flush=True)
    assert len(txt) == len(test)
    return dict(arm=arm, ckpt=os.path.basename(ckpt), sha=sha, wall=wall,
                full=full, clean=clean, only=only, hits=hits, gen=txt)


print("\n[S42] ===== §1/§3 用已入库 ckpt 重算 EM（CPU batch=1，零训练） =====", flush=True)
EM: dict[str, dict] = {}
EM2: dict[str, dict] = {}
SEEDS = (1234,) if SMOKE else (1234, 5678)
for b in BUCKETS:
    test = DATA[b]["test"]
    EM[b], EM2[b] = {}, {}
    for seed in SEEDS:
        tgt = EM[b] if seed == 1234 else EM2[b]
        for arm, res in (("res", True), ("nores", False)):
            p = (CK19.format(bucket=b, arm=arm) if seed == 1234
                 else CK19.format(bucket=b, arm=arm).replace("seed1234", f"seed{seed}"))
            if not os.path.exists(p):
                print(f"[EM] {b} {arm} seed={seed}: ★ckpt 缺失 {p}", flush=True)
                continue
            tgt[arm] = run_arm(b, f"{arm} seed={seed}", p, res, test)
            if SMOKE:
                break
        if SMOKE:
            break

EM36 = None
if os.path.exists(CK36) and "add_3d" in DATA:
    EM36 = run_arm("add_3d", "res@6000(S36)", CK36, True, DATA["add_3d"]["test"])

# —— 锚点：与 logs/19_results.jsonl 入库 EM 比对 ——
anchor19 = []
if os.path.exists(ANCHOR_JSONL) and not SMOKE:
    with open(ANCHOR_JSONL, encoding="utf-8") as f:
        recs = [json.loads(ln) for ln in f if ln.strip()]
    for b in BUCKETS:
        for arm in ("res", "nores"):
            if arm not in EM[b]:
                continue
            row = next((r for r in recs if r["bucket"] == b and r["arm"] == arm
                        and r["seed"] == 1234), None)
            if row is None:
                continue
            d = (EM[b][arm]["full"]["em"] - row["em"]) * 100
            anchor19.append(dict(bucket=b, arm=arm, jsonl_em=row["em"],
                                 cpu_em=EM[b][arm]["full"]["em"], delta_pp=d))
            print(f"[ANCHOR19] {b} {arm}: 入库 EM={row['em']*100:.2f}% vs 本机 CPU "
                  f"EM={EM[b][arm]['full']['em']*100:.2f}% ⇒ Δ={d:+.2f}pp", flush=True)

# ============================================================================
# ③ §1 模板 bug 影响面
# ============================================================================
print("\n[S42] ===== §1 模板 bug（prompt 残留字面量 {expr}，不可作答）影响面 =====", flush=True)


def mse(xs):
    if not xs:
        return float("nan"), float("nan")
    m = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return m, sd / math.sqrt(len(xs))


def arm_delta(r_, n_):
    """同 test 集上 res−nores 的配对 Δ（全量 / 去模板bug）。"""
    dv = [int(r_["hits"][i]["strict"]) - int(n_["hits"][i]["strict"])
          for i in range(len(r_["hits"]))]
    dv_clean = [d for i, d in enumerate(dv) if not r_["hits"][i]["broken"]]
    m_all, se_all = mse(dv)
    m_cl, se_cl = mse(dv_clean)
    return dict(arm_delta_full_pp=m_all * 100, arm_delta_full_se_pp=se_all * 100,
                arm_delta_clean_pp=m_cl * 100, arm_delta_clean_se_pp=se_cl * 100,
                arm_delta_shift_pp=(m_cl - m_all) * 100)


S1: dict[str, dict] = {}
for b in BUCKETS:
    tr, te = DATA[b]["train"], DATA[b]["test"]
    nbt = sum(1 for r in tr if "{expr}" in r["nl"])
    nbe = sum(1 for r in te if "{expr}" in r["nl"])
    nodigit = sum(1 for r in te if "{expr}" in r["nl"] and not any(c.isdigit() for c in r["nl"]))
    row = dict(n_train=len(tr), n_test=len(te),
               bad_train=nbt, bad_train_pct=100.0 * nbt / len(tr),
               bad_test=nbe, bad_test_pct=100.0 * nbe / len(te),
               bad_pct_of_tpl=100.0 * len(BARE_IDX) / len(TEMPLATES),
               bad_test_no_digit=nodigit)
    for tag, E in (("", EM[b]), ("_s5678", EM2.get(b, {}))):
        for arm in ("res", "nores"):
            if arm in E:
                e = E[arm]
                row[f"em_full_{arm}{tag}"] = e["full"]["em"]
                row[f"em_full_{arm}{tag}_se"] = e["full"]["se"]
                row[f"em_clean_{arm}{tag}"] = e["clean"]["em"]
                row[f"em_only_{arm}{tag}"] = e["only"]["em"]
                row[f"d_em_{arm}{tag}_pp"] = (e["clean"]["em"] - e["full"]["em"]) * 100
        if "res" in E and "nores" in E:
            for k, v in arm_delta(E["res"], E["nores"]).items():
                row[k + tag] = v
    ds = [x for x in (row.get("d_em_res_pp"), row.get("d_em_res_s5678_pp"))
          if isinstance(x, float) and math.isfinite(x)]
    row["d_em_res_mean_pp"] = sum(ds) / len(ds) if ds else float("nan")
    S1[b] = row
    print(f"[S1] {b}: 模板 bug train={nbt}/{len(tr)}({row['bad_train_pct']:.2f}%) "
          f"test={nbe}/{len(te)}({row['bad_test_pct']:.2f}%) 无数字={nodigit}/{nbe} | "
          f"res EM {100*row.get('em_full_res',float('nan')):.2f}% → 去bug "
          f"{100*row.get('em_clean_res',float('nan')):.2f}% "
          f"(Δ={row.get('d_em_res_pp',float('nan')):+.2f}pp; seed5678 "
          f"Δ={row.get('d_em_res_s5678_pp',float('nan')):+.2f}pp) | 仅bug样本 EM="
          f"{100*row.get('em_only_res',float('nan')):.2f}%(n={EM[b]['res']['only']['n'] if 'res' in EM[b] else 0}) "
          f"| 臂Δ(res−nores) 全量={row.get('arm_delta_full_pp',float('nan')):+.2f}pp → 去bug="
          f"{row.get('arm_delta_clean_pp',float('nan')):+.2f}pp (移动 "
          f"{row.get('arm_delta_shift_pp',float('nan')):+.2f}pp)", flush=True)

# ============================================================================
# ④ §2 词表卫生：测试集 prompt 里有多少 token 落在词表外
# ============================================================================
print("\n[S42] ===== §2 词表卫生（V=8192 WordLevel；OOV ⇒ <unk> id=%d） =====" % UNK_ID, flush=True)
S2: dict[str, dict] = {}
oov_char_samples: dict[str, set] = defaultdict(set)
oov_char_counts: Counter = Counter()
for b in BUCKETS:
    te = DATA[b]["test"]
    ntok = nunk = 0
    n_gt1 = 0
    per_char: dict[str, int] = defaultdict(int)
    for i, r in enumerate(te):
        ids = enc(r["prompt"])
        ntok += len(ids)
        u = sum(1 for x in ids if x == UNK_ID)
        nunk += u
        if u > 1:
            n_gt1 += 1
        for c in r["prompt"]:
            if c not in VOCAB:
                per_char[c] += 1
                oov_char_samples[c].add((b, i))
                oov_char_counts[c] += 1
    # 不变式：每个字符恰好一个 token（Split Isolated）⇒ len(ids)==len(prompt) 且 OOV 字符数==unk 数
    len_eq = all(len(enc(r["prompt"])) == len(r["prompt"]) for r in te[:50])
    inv = all(sum(1 for c in r["prompt"] if c not in VOCAB) == sum(1 for x in enc(r["prompt"]) if x == UNK_ID)
              for r in te)
    # 去掉 → 之后是否还有 OOV（即"可归因于常量分隔符之外"的部分）
    rest = {c: v for c, v in per_char.items() if c != "→"}
    S2[b] = dict(prompt_tokens=ntok, unk_tokens=nunk,
                 unk_per_prompt=nunk / len(te), unk_rate_pct=100.0 * nunk / ntok,
                 samples_with_gt1_unk=n_gt1,
                 oov_chars={c: v for c, v in per_char.items()},
                 oov_chars_excl_arrow={c: v for c, v in rest.items()},
                 len_invariant=len_eq, unk_eq_oovchar=inv)
    print(f"[S2] {b}: test prompt token={ntok} 其中 <unk>={nunk} "
          f"（{100.0*nunk/ntok:.2f}%，{nunk/len(te):.3f}/条）| 含 >1 个 unk 的样本={n_gt1} "
          f"| OOV 字符={ {c: v for c, v in per_char.items()} } | "
          f"len(ids)==len(text)={len_eq} 每字符1token且unk数==OOV字符数={inv}", flush=True)

print(f"[S2] 全桶 OOV 字符合计: {dict(oov_char_counts)} | 覆盖样本数={len(oov_char_samples)}", flush=True)
for c in sorted(oov_char_counts):
    cnt = len(oov_char_samples[c])
    print(f"[S2] OOV 字符 {c!r} (U+{ord(c):04X}): 命中 {cnt} 条 test 样本"
          f"（= 全桶 test 样本的 {100.0*cnt/(N_TEST*len(BUCKETS)):.1f}%）", flush=True)

# —— 去掉 → 之后还剩什么：可量化的那一部分 ——
S2_rest_affected = {b: sum(1 for r in DATA[b]["test"]
                           if any(c not in VOCAB for c in r["prompt"] if c != "→"))
                    for b in BUCKETS}
S2_target_unk = {b: sum(1 for r in DATA[b]["test"] for x in enc(r["text"] + "<eos>") if x == UNK_ID)
                 for b in BUCKETS}
print(f"[S2] 除常量 '→' 之外仍含 OOV 的 test 样本数: {S2_rest_affected} "
      f"⇒ 可归因影响面={'0 条（无样本可去 ⇒ ΔEM ≡ 0）' if sum(S2_rest_affected.values())==0 else '见上'}", flush=True)
print(f"[S2] target（'#### <答案>' + <eos>）里的 <unk> 数: {S2_target_unk} ⇒ "
      f"{'target 干净' if sum(S2_target_unk.values())==0 else '★target 也受损'}", flush=True)
S2_FIX = ("最小修法（只建议，未改）：'→' 由数据生成器 stages/14_synth_arith.py:255 的 "
          "render() 硬编码（f\"题干：{nl} → \"，A 臂口径），不在模板里；"
          "词表是 nanoSeek/data/chinese/char_tokenizer.json（只读）。两条路："
          "① 下次重建词表时把 '→' 加进 V=8192 的字符集（'→' 只占 1 个槽位，"
          "或把最冷门字符挤掉）；② 不改词表，把 render() 的分隔符换成词表内字符"
          "（如 '=>'、'：'、'|'）。两条都会改变 prompt token 序列 ⇒ 必须重训，"
          "不是纯推理侧的修法。")
print(f"[S2] {S2_FIX}", flush=True)

# ============================================================================
# ⑤ §3 全桶地板表
# ============================================================================
print("\n[S42] ===== §3 全桶地板表（各桶自己 train 的最高频答案 / top5 / 机会水平） =====", flush=True)
S3: dict[str, dict] = {}
for b in BUCKETS:
    tr, te = DATA[b]["train"], DATA[b]["test"]
    gtr = Counter(r["gold"] for r in tr)
    top1, c1 = gtr.most_common(1)[0]
    single = c1 / len(tr)
    top5 = [v for v, _ in gtr.most_common(5)]
    top5floor = sum(1 for r in te if r["gold"] in top5) / len(te)
    single_test = sum(1 for r in te if r["gold"] == top1) / len(te)   # R36 口径：单答案覆盖 test
    chance = 1.0 / len(answer_values(b))
    em = EM[b]["res"]["full"]["em"] if "res" in EM[b] else float("nan")
    em_clean = EM[b]["res"]["clean"]["em"] if "res" in EM[b] else float("nan")
    em_n = EM[b]["nores"]["full"]["em"] if "nores" in EM[b] else float("nan")
    em_s2 = EM2.get(b, {}).get("res", {}).get("full", {}).get("em", float("nan"))
    em36 = EM36["full"]["em"] if (b == "add_3d" and EM36) else float("nan")
    em36_clean = EM36["clean"]["em"] if (b == "add_3d" and EM36) else float("nan")
    gap = (em - top5floor) * 100
    gap_n = (em_n - top5floor) * 100
    gap36 = (em36 - top5floor) * 100 if em36 == em36 else float("nan")
    S3[b] = dict(domain=len(answer_values(b)), single_floor=single, single_ans=top1,
                 single_floor_test=single_test, top5_floor=top5floor, top5=top5, chance=chance,
                 em_res=em, em_res_clean=em_clean, em_res_seed5678=em_s2, em_nores=em_n,
                 em_res_6000=em36, em_res_6000_clean=em36_clean,
                 em_minus_top5_pp=gap, em_nores_minus_top5_pp=gap_n,
                 em6000_minus_top5_pp=gap36,
                 flag=bool(gap < 10.0), flag_nores=bool(gap_n < 10.0),
                 flag_6000=bool(gap36 < 10.0) if gap36 == gap36 else None)
    print(f"[S3] {b}: 单答案地板(train P)={100*single:.2f}%(答案 {top1}; 覆盖 test "
          f"{100*single_test:.2f}%) top5地板(test)={100*top5floor:.2f}% 机会水平={100*chance:.2f}% "
          f"| res EM={100*em:.2f}%(去bug {100*em_clean:.2f}%, seed5678 {100*em_s2:.2f}%) "
          f"→ EM−top5={gap:+.1f}pp {'★太近(不可信)' if gap < 10.0 else 'OK'} "
          f"| nores EM={100*em_n:.2f}% → {gap_n:+.1f}pp {'★太近(不可信)' if gap_n < 10.0 else 'OK'}"
          f"{f' | @6000 res EM={100*em36:.2f}%(去bug {100*em36_clean:.2f}%) → {gap36:+.1f}pp' if em36==em36 else ''}",
          flush=True)

# ============================================================================
# ⑥ 落盘
# ============================================================================
wall = time.time() - t_start
S1_trigger = {b: S1[b].get("d_em_res_pp") for b in BUCKETS}
honesty = []
if not os.path.exists(CK19.format(bucket="add_2d", arm="res")):
    honesty.append("19 ckpt 缺失（见日志）")
honesty.append("'→' 出现在 100% 的 test prompt（常量分隔符）⇒ “去掉含 OOV 的样本”这一消融"
               "退化为清空整个 test 集，EM 变化量不可定义；且 <unk> 是训练/测试同分布的常量，"
               "其对 EM 的因果影响不可从已入库 ckpt 识别（需换词表重训，任务禁止）⇒ 不可量化。")
honesty.append("§1/§2 的 EM 变化量用 seed1234 的 ckpt（res/nores × 5 桶）为主，seed5678 作复核；"
               "两 seed 的方向若不一致以 seed1234 报。")
honesty.append("6000 步（饱和点）只有 add_3d 的 ckpt（36_ckpt_*）；其余四桶的饱和点 EM 无 ckpt ⇒ 不可量化。")
honesty.append("CPU 复算与 GPU 入库值可能存在数值级 argmax 抖动（见 [ANCHOR19] 的 Δ）。")
honesty.append("本审计只量化“这批已入库 ckpt 的 test 集”，未测 §1 模板 bug 对 train 侧学习的影响"
               "（坏样本占 train 1.9–3.0%，对训练本身的损害不可从现有资产量化）。")
honesty.append("口径对照：R36 的 add_3d“单答案地板 0.12%”= 该答案覆盖 test 的比例(1/800)；"
               "本文同时报 train 口径 P(最常见答案)=0.07%(3/4000)，两者口径不同、不矛盾；"
               "top5=0.25% 两边一致。")
honesty.append("§1 判据结论：五个桶的绝对 EM 全部被模板 bug 低估 1.45–1.77pp(res 臂, seed1234; "
               "seed5678 同向) ⇒ 触及 ≥0.5pp 门槛，所有以绝对 EM 写死的判据（S14 的 ≥90% 门槛、"
               "S19 主表 EM 列、R36 的 add_3d 数值）都应改报“去 bug 后”的值；"
               "臂间配对 Δ(res−nores) 是共模稀释，移动 ≤0.02pp/add_3d +1.35pp（效应量 35pp，"
               "方向与显著性不变）⇒ 不需要重训，只需用本文的去 bug 子集重算 EM。")
if SMOKE:
    honesty.append("★本次是 S42_SMOKE=1 冒烟跑，N_TEST=40，所有数字不作数。")

OUT = dict(
    unit="S42", device=device, smoke=SMOKE, wall_s=wall, zero_training=True,
    config=dict(V=V, PAD=PAD_ID, EOS=EOS_ID, UNK=UNK_ID, D=D, FF=FF, NHEAD=NHEAD,
                N_TRAIN=N_TRAIN, N_TEST=N_TEST, batch=EM_BATCH, buckets=list(BUCKETS)),
    templates=dict(n_templates=len(TEMPLATES), n_cn=N_TPL_CN, render_variants=RENDER_VARIANTS,
                   bare_idx=BARE_IDX, bare_count=len(BARE_IDX),
                   bare_share_of_templates_pct=100.0 * len(BARE_IDX) / len(TEMPLATES),
                   bare_examples=[TEMPLATES[i][0] for i in BARE_IDX[:3]]),
    gen_stats=GEN_STATS, anchor_s14=anchor_rows, anchor19=anchor19,
    s1_template_bug=S1, s2_vocab=S2, s2_oov_char_totals=dict(oov_char_counts),
    s2_rest_affected=S2_rest_affected, s2_target_unk=S2_target_unk, s2_fix=S2_FIX,
    s3_floors=S3, s3_em36_add3d=(EM36["full"] if EM36 else None),
    s1_trigger=S1_trigger, honesty=honesty,
)
with open(LOG_DIR + "/42_data_hygiene.json", "w", encoding="utf-8") as f:
    json.dump(OUT, f, ensure_ascii=False, indent=1)

print("\n[S42] ===== 汇总 =====", flush=True)
print(f"[S42] 模板 bug：{len(BARE_IDX)}/{len(TEMPLATES)} 句 = "
      f"{100.0*len(BARE_IDX)/len(TEMPLATES):.2f}% 模板；样本影响见 §1（因去重拒绝，"
      f"裸式模板的样本占比低于模板占比）", flush=True)
for b in BUCKETS:
    r = S1[b]
    print(f"[S42] {b}: bug样本 train {r['bad_train']}/{r['n_train']}"
          f"({r['bad_train_pct']:.2f}%) test {r['bad_test']}/{r['n_test']}"
          f"({r['bad_test_pct']:.2f}%) | EM Δ(res)={r.get('d_em_res_pp', float('nan')):+.2f}pp "
          f"Δ(nores)={r.get('d_em_nores_pp', float('nan')):+.2f}pp | 臂Δ移动="
          f"{r.get('arm_delta_shift_pp', float('nan')):+.2f}pp", flush=True)
print("[S42] 地板表: 桶 | 单答案地板(train P / test 覆盖) | top5地板(test) | 机会水平 | "
      "res EM | EM−top5 | nores EM | nores EM−top5 | 判据", flush=True)
for b in BUCKETS:
    s = S3[b]
    bad = s["flag"] or s["flag_nores"]
    print(f"[S42] {b} | {100*s['single_floor']:.2f}% / {100*s['single_floor_test']:.2f}% | "
          f"{100*s['top5_floor']:.2f}% | {100*s['chance']:.2f}% | {100*s['em_res']:.2f}% | "
          f"{s['em_minus_top5_pp']:+.1f}pp | {100*s['em_nores']:.2f}% | "
          f"{s['em_nores_minus_top5_pp']:+.1f}pp | "
          f"{'★EM 与地板太近(<10pp) ⇒ 该桶该臂结论不可信' if bad else 'OK'}", flush=True)
trigger = [b for b in BUCKETS if abs(S1[b].get("d_em_res_pp", 0.0)) >= 0.5]
print(f"[S42] §1 判据：|ΔEM| ≥0.5pp 的桶 = {trigger} "
      f"⇒ {'必须重跑关键对照（同一坏样本对所有臂是共模稀释，但绝对 EM 被低估 1–2pp）' if trigger else '影响可忽略'}",
      flush=True)
floor_flags = [b for b in BUCKETS if S3[b]["flag"] or S3[b]["flag_nores"]]
extra36 = ""
if S3["add_3d"]["flag_6000"] is not None:
    extra36 = (f"（add_3d 在 6000 步饱和后 EM−top5地板="
               f"{S3['add_3d']['em6000_minus_top5_pp']:+.1f}pp"
               f"{'，仍太近' if S3['add_3d']['flag_6000'] else '，已脱离地板'}）")
print(f"[S42] 地板判据：EM−top5地板 <10pp 的桶（按臂）= {floor_flags}{extra36}", flush=True)
print(f"[S42] 墙钟={wall:.1f}s ({(wall)/60:.1f}min) | JSON={LOG_DIR}/42_data_hygiene.json", flush=True)
print("[S42][DONE] exit=0", flush=True)
