#!/usr/bin/env python3
"""S44 · ★P0 R36 强制：卡内结构「不可替代」在 ≥12000 步重做（A 复用，只训 B/C/D/E）。

背景：E6（logs/e6_sat.log）@6000 步测「卡内结构 = 自注意力 + 逐位置 FFN 不可替代」，
B/C/D/E vs A 的 Δ = −84.5…−73.0pp（8/8 ≤−2SE）；但 R36 明确 @6000 是欠训区，
真饱和/最低门槛 = ≥12000 步 ⇒ 本单元在 12000 步把这个结论钉死（或推翻）。

设计：
  A（复用，不训）= logs/38_ckpt_w1_s12000_seed{1234,5678}.pt（S38 已训，EM 91.88/87.12%）；
  B/C/D/E 各训 12000 步（8 run）；其余与 S22/E6（S36）逐字相同
  （d=128 ff=512 NHEAD=4 lr=1e-3 batch=32 MAXLEN=512 思维卡带残差 add_3d train4000/test800）；
  每 750 步取一次权重快照（S38 已证「单条轨迹 @N 快照 == 独立跑 N 步」）⇒ 收敛曲线；
  ckpt 落 logs/44_ckpt_<臂>_s12000_seed<s>.pt。

★ 口径（缺一即无效，全部照抄 S36/S38）：
  1 EM 一律 batch=1（R28）+ batch=1 vs 16 逐字一致性自检；
  2 SEEN/UNSEEN 分层（train4000 对 test 的 (a,b) 泄漏 16/800）；
  3 地板用本桶 add_3d：单答案 / top5；
  4 ★锚点复核：A 复用 ckpt 必须复现 S38 的 91.88/87.12%（Δ>1pp 要点名）；
  5 ★饱和检查（R35）：B/C/D/E 在 9000→12000 段的增量；末段 ≥1pp ⇒ 未饱和、结论不可下；
  6 R29 非零比例；TEST sha256[:16] 必须 == b34e7ea515203227；
  7 ★两种口径都报：全量 test（n=800）与 clean 子集（剔除 nl 含字面量 {expr} 的 30 条，n=770）。

判据（写死）：
  Δ = EM(臂) − EM(A) @12000，配对 ±SE（全量与 clean 两种口径都算）：
    B/C/D/E 全部 ≤−2SE 且 2 seed 同号 ⇒「不可替代」在 ≥12000 步仍成立（钉死）；
    有任一臂 |Δ|≤1SE（追平 A）⇒ @6000 的「不可替代」是欠训假象（如实报，比 E6 更重）；
  并报「是否饱和」；未饱和不得下结论。

复用：exec stages/19_residual_scan.py 到 "# ③ 主循环" 之前（⇒ 同 tokenizer、同 add_3d
train4000/test800、同 enc_layer/greedy_gen/parse_ans/classify/r28_selfcheck 口径）。
五臂卡实现逐字复制 S36（stages/36_saturation_retest.py）。
只允许写：本文件、logs/p0_struct.log、logs/44_*、/tmp；既有 stages/*.py 只读。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

ROOT = "/home/vesita/coding/my/flowme"
SRC19 = f"{ROOT}/stages/19_residual_scan.py"

SMOKE44 = os.environ.get("S44_SMOKE") == "1"
if SMOKE44:
    os.environ["S19_SMOKE"] = "1"          # 让 S19 头跳过落盘（冒烟）

# ============================================================================
# ① 逐字复用 S19 前半段（⇒ 同 tokenizer、同 add_3d train4000/test800、同评估口径）
# ============================================================================
_src = open(SRC19, encoding="utf-8").read()
_cut = _src.index("# ③ 主循环")
NS = {"__name__": "s19_head", "__file__": SRC19}
exec(compile(_src[:_cut], SRC19, "exec"), NS)          # noqa: S102
for _k, _v in NS.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
del NS

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402

T0 = time.time()
FP = 0.1
BUCKET = "add_3d"
SMOKE_SEEDS = (1234,)
SEEDS44 = SMOKE_SEEDS if SMOKE44 else (1234, 5678)
STEPS44 = 30 if SMOKE44 else 12000
GRID = tuple(range(3, 31, 3)) if SMOKE44 else tuple(range(750, 12001, 750))
FINAL = GRID[-1]
REPORT_AT = tuple(st for st in (3000, 6000, 9000, 12000) if st in GRID) or (FINAL,)
TRAIN_ARMS = ("B", "C", "D", "E")
ALL_ARMS = ("A",) + TRAIN_ARMS
N_EVAL = 60 if SMOKE44 else N_TEST                   # 800
CKPT_TMPL = f"{ROOT}/logs/44_ckpt_{{arm}}_s{FINAL}_seed{{seed}}.pt"
JSONL = "/tmp/44_results_smoke.jsonl" if SMOKE44 else f"{ROOT}/logs/44_results.jsonl"
TEST_SHA_EXPECT = "b34e7ea515203227"
LEAK_EXPECT = 16
CLEAN_BAD_EXPECT = 30
CLEAN_BAD_43 = [51, 113, 164, 197, 209, 230, 248, 249, 324, 370, 420, 438, 444, 463,
                540, 550, 562, 583, 604, 608, 627, 634, 638, 653, 680, 682, 706, 712, 751, 762]
A_CKPT = {s: f"{ROOT}/logs/38_ckpt_w1_s12000_seed{s}.pt" for s in SEEDS44}
ANCHOR_A_S38 = {1234: 0.9188, 5678: 0.8712}          # S38 @12000 全量 EM（batch=1）
ANCHOR_CURVE_S38 = {1234: {3000: 0.3725, 6000: 0.8562, 9000: 0.8800, 12000: 0.9188},
                    5678: {3000: 0.3113, 6000: 0.7438, 9000: 0.8362, 12000: 0.8712}}
ARM_DESC = {
    "A": "现状=S19-res：h + TransformerEncoderLayer(MHA4+FFN512/gelu+2×LN)（复用 S38 ckpt）",
    "B": "纯逐位置 MLP：h + MLP(LN(h))（零跨位置混合）",
    "C": "因果卷积 k=3：h + Linear(GELU(DepthwiseConv1d(128,k=3,左填充)))",
    "D": "门控 GRU 风格：g=sigmoid(Wg·LN(h)), c=tanh(MLP(LN(h))); g⊙c+(1−g)⊙h",
    "E": "仅注意力（去 FFN）：h + Dropout(MHA(LN(h)))",
}

assert str(device).startswith("cuda"), f"device={device} ⇒ 本单元必须在 cuda 上跑"
print(f"\n[S44] ★R36 强制：卡内结构@≥12000 步重做 | 桶={BUCKET} 步数={STEPS44} "
      f"曲线格={GRID[0]}..{GRID[-1]}/{GRID[1]-GRID[0]}步 seeds={SEEDS44} | "
      f"臂= A(复用 38_ckpt_w1_s12000) + 训练 {list(TRAIN_ARMS)} | d={D} ff={FF} NHEAD={NHEAD} "
      f"lr={LR} batch={BATCH} MAXLEN={MAXLEN} | EM主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | "
      f"★不切 dev、固定 {STEPS44} 步末点（未用 dev 选步）| smoke={SMOKE44}", flush=True)
print(f"[S44-DEV] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"V={V} PAD={PAD_ID} EOS={EOS_ID}", flush=True)

# ============================================================================
# ② 数据核对（口径 2/3/6/7 + TEST 指纹 + clean 子集）
# ============================================================================
TRAIN = DATA[BUCKET]["train"]
TEST = DATA[BUCKET]["test"]
TEST_EVAL = TEST[:N_EVAL]

_h = hashlib.sha256()
for r in TEST:
    _h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
TEST_SHA = _h.hexdigest()[:16]
_tp_tr = {(r["a"], r["b"]) for r in TRAIN}
for r in TEST:
    r["seen"] = (r["a"], r["b"]) in _tp_tr
LEAK = sum(1 for r in TEST if r["seen"])
N_SEEN, N_UNSEEN = LEAK, len(TEST) - LEAK

_c_tr = Counter(r["gold"] for r in TRAIN)
_top1, _top1n = _c_tr.most_common(1)[0]
_top5 = [v for v, _ in _c_tr.most_common(5)]
FLOOR_1 = sum(1 for r in TEST if r["gold"] == _top1) / len(TEST)
FLOOR_5 = sum(1 for r in TEST if r["gold"] in _top5) / len(TEST)

BAD_IDX = [i for i, r in enumerate(TEST) if "{expr}" in r["nl"]]
CLEAN_IDX = [i for i in range(len(TEST_EVAL)) if i not in set(BAD_IDX)]
N_CLEAN = len(CLEAN_IDX)

sha_ok = TEST_SHA == TEST_SHA_EXPECT
leak_ok = LEAK == LEAK_EXPECT
clean_ok = len(BAD_IDX) == CLEAN_BAD_EXPECT and BAD_IDX == CLEAN_BAD_43
print(f"[S44-DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)}(eval {len(TEST_EVAL)}) 文本零重叠 | "
      f"答案域={min(answer_values(BUCKET))}..{max(answer_values(BUCKET))}"
      f"({len(answer_values(BUCKET))}个) | ★不切 dev ⇒ 训练集={len(TRAIN)}（与 S22/S36/S38 逐位同源）",
      flush=True)
print(f"[S44-CHK6] TEST 指纹 sha256[:16]={TEST_SHA} vs 入库 {TEST_SHA_EXPECT} ⇒ "
      f"{'逐位一致 ✓' if sha_ok else '★不一致'} | (a,b) 泄漏对 train4000={LEAK}/{len(TEST)} vs "
      f"入库 {LEAK_EXPECT} ⇒ {'一致 ✓' if leak_ok else '★不一致'}", flush=True)
print(f"[S44-CHK2] SEEN={N_SEEN} / UNSEEN={N_UNSEEN}（train4000 对 test 的 (a,b) 泄漏 "
      f"{100.0*LEAK/len(TEST):.2f}%，分层报 EM）", flush=True)
print(f"[S44-CHK3] add_3d 本桶地板：单答案 '{_top1}'(train4000 {_top1n}/4000) 覆盖 test "
      f"{FLOOR_1*100:.2f}% | top5={_top5} 覆盖 test {FLOOR_5*100:.2f}% ⇒ ★本地板="
      f"{FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}%（不借用其它桶）", flush=True)
print(f"[S44-CHK7] clean 子集（口径 7，METRICS 权威口径）：剔除 nl 含字面量 '{{expr}}' 的 "
      f"{len(BAD_IDX)} 条 ⇒ n_clean={N_CLEAN} | 与 S43 入库 bad_idx "
      f"{'逐位一致 ✓' if clean_ok else '★不一致'} | 30 条 only_bug 在 S43 实测 EM=0%",
      flush=True)
if not SMOKE44:
    assert sha_ok and leak_ok, "数据指纹不符 ⇒ 与 S22/S36/S38 不是同一数据集，本单元无效"
    assert clean_ok, "clean 索引与 S43 不一致 ⇒ 口径 7 无效"


# ============================================================================
# ③ 五臂卡实现（逐字复制 S36/S22）+ 统一容器 Cards44
# ============================================================================
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
        self.mlp = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(FP),
                                 nn.Linear(ff, d), nn.Dropout(FP))

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
        self.do = nn.Dropout(FP)

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
        self.att = nn.MultiheadAttention(d, NHEAD, dropout=FP, batch_first=True)
        self.do = nn.Dropout(FP)

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


HOOK_IN = {"fn": None}          # 思维卡【输入】侧干预（R29 用）


class Cards44(nn.Module):
    """与 S19 Cards / S22 Cards22 / S36 Cards36 逐字同构：emb → in_enc → card → head。"""

    def __init__(self, arm, d, ff):
        super().__init__()
        self.d = d
        self.arm = arm
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
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
        assert n <= pe.size(0), f"PE 长度 {pe.size(0)} < 序列长 {n}"
        x = self.emb(ids) + pe[:n].unsqueeze(0)
        h = self.in_enc(x, src_mask=m)
        if HOOK_IN["fn"] is not None:
            h = HOOK_IN["fn"](h)
        h = self.card(h, m)
        return self.head(h)


class Cards38Clone(nn.Module):
    """S38 的计算图（thought 命名），只用于证明「A ≡ S38 ckpt 的同一计算图」。"""

    def __init__(self, d, ff):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        self.thought = enc_layer(d, ff)
        self.head = nn.Linear(d, V)
        self.register_buffer("pe", sin_pe(MAXLEN, d), persistent=False)


def n_params_arm(arm):
    mdl = Cards44(arm, D, FF)
    n = sum(p.numel() for p in mdl.parameters())
    del mdl
    return n


def load_a_state(path):
    """S38 ckpt 的 thought.* → Cards44-A 的 card.layer.*。"""
    sd = torch.load(path, map_location="cpu", weights_only=True)
    return {k.replace("thought.", "card.layer."): v for k, v in sd.items()}


# ---- ★一致性自检：A 与 S38 是同一计算图（构造顺序相同 ⇒ 同 seed 初始化应逐位相同）----
torch.manual_seed(999)
_a44 = Cards44("A", D, FF)
torch.manual_seed(999)
_a38 = Cards38Clone(D, FF)
_s44, _s38 = _a44.state_dict(), _a38.state_dict()
_map = {k: k.replace("card.layer.", "thought.") for k in _s44}
INIT_DIFF = max(float((_s44[k] - _s38[_map[k]]).abs().max()) for k in _s44)
del _a44, _a38
print(f"[S44-INIT] 臂 A 与 S38 Cards38 同 seed(999) 初始化逐参数最大差={INIT_DIFF:.3e} "
      f"⇒ {'✓ 同一计算图（可逐位复用 S38 ckpt）' if INIT_DIFF == 0.0 else '★不同起点'}",
      flush=True)
if not SMOKE44:
    assert INIT_DIFF == 0.0, "A 与 S38 不是同一计算图 ⇒ 复用无效"


def train44(arm, seed, train, steps, grid):
    """与 S36 train36 唯一差别：每 750 步取快照（其余逐字相同）。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards44(arm, D, FF).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0 = 0, 0, time.time()
    last, snaps = 0.0, {}
    gset = set(grid)
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(model, [train[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step in gset:
            snaps[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step % 1500 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train:{arm}] step={step}/{steps} loss={loss.item():.4f} elapsed={el:.0f}s "
                  f"s/step={(el - last) / (step % 1500 or 1500):.3f}", flush=True)
            last = el
    assert set(grid) <= set(snaps), "未取到全部曲线格点快照"
    return model, time.time() - t0, snaps


@torch.no_grad()
def em_at(model, recs):
    txt = greedy_gen(model, recs, batch=EM_BATCH)          # ★R28 主口径 batch=1
    return [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, recs)], txt


@torch.no_grad()
def per_sample_ce(model, recs, hook=None):
    """逐样本 (整体CE, 数字/算子 token 的 CE)；口径 = S19 report_ce 的 samp_all / dig_samp。"""
    model.eval()
    HOOK_IN["fn"] = hook
    all_ce, dig_ce, dig_tok = [], [], []
    try:
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
                dig_tok.extend(ds)
                dig_ce.append(sum(ds) / len(ds) if ds else float("nan"))
    finally:
        HOOK_IN["fn"] = None
    model.train()
    return all_ce, dig_ce, dig_tok


def dig_acc(dig_ce):
    ys = [x for x in dig_ce if x == x]
    m, se = mean_se(ys)
    acc = math.exp(-m)
    return acc, acc * se, m, se, len(ys)


def iv_zero(x, *rest):
    return torch.zeros_like(x)


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def add_jsonl(obj):
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def frac(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sub(xs, idx):
    return [xs[i] for i in idx]


def hit_se(hits):
    p = frac(hits)
    return p, math.sqrt(p * (1 - p) / len(hits))


def is_catch(d):
    """追平：Δ≡0（SE=0）或 |Δ|≤1SE。"""
    return (d["se"] == 0 and d["d"] == 0) or (d["se"] > 0 and abs(d["d"]) <= d["se"])


def is_neg2(d):
    return d["se"] > 0 and d["d"] < 0 and d["d"] <= -2 * d["se"]


def is_pos2(d):
    return d["se"] > 0 and d["d"] > 0 and d["d"] >= 2 * d["se"]


def delta_tag(d):
    if d["se"] == 0:
        return "Δ≡0（逐样本完全相同，SE=0）" if d["d"] == 0 else \
               f"SE=0 且 Δ={d['d']*100:+.2f}pp（全样本同向）"
    if is_neg2(d):
        return "★≤−2SE（显著更低）"
    if is_pos2(d):
        return "★≥+2SE（显著更高）"
    if is_catch(d):
        return "|Δ|≤1SE（追平）"
    return "1SE<|Δ|<2SE（未过门槛）"


def paired_stats(xa, xb):
    d = [a - b for a, b in zip(xa, xb)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return dict(n=n, d=m, se=se, nz=sum(1 for x in d if x != 0) / n,
                ratio=(m / se if se > 0 else float("inf")))


# ============================================================================
# ④ 评估一个 state_dict：全量/clean EM + 曲线用 CE
# ============================================================================
@torch.no_grad()
def eval_state(arm, sd):
    ev = Cards44(arm, D, FF).to(device)
    ev.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=True)
    hits, _ = em_at(ev, TEST_EVAL)
    all_ce, dig_ce, _ = per_sample_ce(ev, TEST_EVAL)
    acc, acc_se, dce, dce_se, dn = dig_acc(dig_ce)
    ev.train()
    return ev, hits, dict(em_full=frac(hits), em_clean=frac(sub(hits, CLEAN_IDX)),
                          acc=acc, acc_se=acc_se, dig_ce=dce, ce=frac(all_ce))


PARAMS = {a: n_params_arm(a) for a in ALL_ARMS}
print(f"[S44-PARAMS] " + " | ".join(f"{a}={PARAMS[a]/1e6:.3f}M" for a in ALL_ARMS), flush=True)

RESULTS: dict = {}
open(JSONL, "w").close()

# ============================================================================
# ⑤ 臂 A（复用 S38 ckpt，不训练）：锚点复核 + 曲线参考点
# ============================================================================
print(f"\n[S44] ===== ★臂 A（复用 38_ckpt_w1_s12000，不训练）=====", flush=True)
for seed in SEEDS44:
    t0 = time.time()
    r = dict(arm="A", seed=seed, trained=False, params=PARAMS["A"], wall_train=0.0,
             curve={}, ckpt=A_CKPT[seed])
    # 曲线参考点：S38 已落盘的 6000/9000/12000（不存在 3000 ckpt）
    for st, path in ((6000, f"{ROOT}/logs/38_ckpt_w1_s6000_seed{seed}.pt"),
                     (9000, f"{ROOT}/logs/38_ckpt_w1_s9000_seed{seed}.pt"),
                     (12000, A_CKPT[seed])):
        if SMOKE44 and st != 12000:
            continue
        if not os.path.exists(path):
            continue
        ev, hits, m = eval_state("A", load_a_state(path))
        r["curve"][st] = m
        if st == 12000:
            r.update(**m)
            r["hits"] = hits
            r["sha"] = sha16(path)
            seen = [hits[i] for i in range(len(TEST_EVAL)) if TEST_EVAL[i]["seen"]]
            unseen = [hits[i] for i in range(len(TEST_EVAL)) if not TEST_EVAL[i]["seen"]]
            r["em_seen"], r["em_unseen"] = frac(seen), frac(unseen)
            r["n_seen"], r["n_unseen"] = len(seen), len(unseen)
            _r28 = r28_selfcheck(ev, TEST_EVAL, tag=f"[A s{FINAL} seed{seed}] ",
                                 k=min(CHK_BATCH, len(TEST_EVAL)))
            r["r28_same"], r["r28_k"] = _r28["same"], _r28["k"]
            r["em_se"] = math.sqrt(r["em_full"] * (1 - r["em_full"]) / len(TEST_EVAL))
            _all, dig_iv, _ = per_sample_ce(ev, TEST_EVAL, hook=iv_zero)
            a_iv = [math.exp(-x) if x == x else float("nan") for x in dig_iv]
            # 数字每步基线（样本级 exp(−CE)）：R29 的非零比例以它为对照
            _, dig_b, _ = per_sample_ce(ev, TEST_EVAL)
            a_b = [math.exp(-x) if x == x else float("nan") for x in dig_b]
            r["r29_iv_nz"] = frac([1.0 if (x == x and y == y and abs(x - y) > 1e-3) else 0.0
                                   for x, y in zip(a_iv, a_b)])
        del ev
        torch.cuda.empty_cache()
    r["wall_total"] = time.time() - t0
    RESULTS[("A", seed)] = r
    a_s38 = ANCHOR_A_S38[seed]
    dd = (r["em_full"] - a_s38) * 100
    print(f"[S44-ANCHOR] A seed={seed}: 本次@12000 EM全量={r['em_full']*100:.2f}% vs S38入库="
          f"{a_s38*100:.2f}% ⇒ Δ={dd:+.2f}pp {'✓ 复现（≤1pp）' if abs(dd) <= 1.0 else '★>1pp，点名'}",
          flush=True)
    print(f"[S44-MAIN] A seed={seed} | EM全量={r['em_full']*100:.2f}%±{r['em_se']*100:.2f}"
          f"(n={len(TEST_EVAL)}) clean={r['em_clean']*100:.2f}%(n={N_CLEAN}) | "
          f"SEEN={r['em_seen']*100:.2f}%({r['n_seen']}) UNSEEN={r['em_unseen']*100:.2f}%"
          f"({r['n_unseen']}) | 数字每步={r['acc']*100:.2f}% | R29非零比={r['r29_iv_nz']*100:.1f}% | "
          f"R28={r['r28_same']}/{r['r28_k']} | 本run={r['wall_total']/60:.2f}min | sha16={r['sha']}",
          flush=True)
    add_jsonl({k: v for k, v in r.items() if k != "hits"})

# ============================================================================
# ⑥ 主循环：B/C/D/E × 2 seed，各 12000 步 + 每 750 步快照 ⇒ 曲线
# ============================================================================
for seed in SEEDS44:
    for arm in TRAIN_ARMS:
        t0 = time.time()
        print(f"\n[S44] ===== arm={arm} seed={seed} steps={STEPS44} | train={len(TRAIN)} "
              f"test={len(TEST_EVAL)} | {ARM_DESC[arm]} =====", flush=True)
        model, wall_train, snaps = train44(arm, seed, TRAIN, STEPS44, GRID)
        ck = CKPT_TMPL.format(arm=arm, seed=seed)
        dig = "-"
        if not SMOKE44:
            torch.save(model.state_dict(), ck)
            dig = sha16(ck)
            print(f"[CKPT] saved {ck} sha256[:16]={dig}", flush=True)

        r = dict(arm=arm, seed=seed, trained=True, params=PARAMS[arm], wall_train=wall_train,
                 curve={}, ckpt=ck, sha=dig)
        ev = Cards44(arm, D, FF).to(device)
        for st in GRID:
            ev.load_state_dict({k: v.to(device) for k, v in snaps[st].items()}, strict=True)
            hits, _ = em_at(ev, TEST_EVAL)
            all_ce, dig_ce, _ = per_sample_ce(ev, TEST_EVAL)
            acc, acc_se, dce, _, _ = dig_acc(dig_ce)
            r["curve"][st] = dict(em_full=frac(hits), em_clean=frac(sub(hits, CLEAN_IDX)),
                                  acc=acc, acc_se=acc_se, dig_ce=dce, ce=frac(all_ce))
            if st in REPORT_AT or st == FINAL:
                print(f"  [S44-PT] {arm} seed={seed} step={st} EM全量={frac(hits)*100:.2f}% "
                      f"clean={frac(sub(hits, CLEAN_IDX))*100:.2f}% | 数字每步={acc*100:.2f}%",
                      flush=True)
        fin = r["curve"][FINAL]
        r.update(em_full=fin["em_full"], em_clean=fin["em_clean"], acc=fin["acc"],
                 acc_se=fin["acc_se"], dig_ce=fin["dig_ce"], ce=fin["ce"])
        ev.load_state_dict({k: v.to(device) for k, v in snaps[FINAL].items()}, strict=True)
        hits_fin, _ = em_at(ev, TEST_EVAL)
        r["hits"] = hits_fin
        seen = [hits_fin[i] for i in range(len(TEST_EVAL)) if TEST_EVAL[i]["seen"]]
        unseen = [hits_fin[i] for i in range(len(TEST_EVAL)) if not TEST_EVAL[i]["seen"]]
        r["em_seen"], r["em_unseen"] = frac(seen), frac(unseen)
        r["n_seen"], r["n_unseen"] = len(seen), len(unseen)
        _r28 = r28_selfcheck(ev, TEST_EVAL, tag=f"[{arm} s{FINAL} seed{seed}] ",
                             k=min(CHK_BATCH, len(TEST_EVAL)))
        r["r28_same"], r["r28_k"] = _r28["same"], _r28["k"]
        _, dig_iv, _ = per_sample_ce(ev, TEST_EVAL, hook=iv_zero)
        a_iv = [math.exp(-x) if x == x else float("nan") for x in dig_iv]
        _, dig_b, _ = per_sample_ce(ev, TEST_EVAL)
        a_b = [math.exp(-x) if x == x else float("nan") for x in dig_b]
        r["r29_iv_nz"] = frac([1.0 if (x == x and y == y and abs(x - y) > 1e-3) else 0.0
                               for x, y in zip(a_iv, a_b)])
        r["wall_total"] = time.time() - t0
        del ev, snaps, model
        torch.cuda.empty_cache()
        RESULTS[(arm, seed)] = r
        add_jsonl({k: v for k, v in r.items() if k != "hits"})
        em_se = math.sqrt(r["em_full"] * (1 - r["em_full"]) / len(TEST_EVAL))
        print(f"[S44-MAIN] {arm} seed={seed} | EM全量={r['em_full']*100:.2f}%±{em_se*100:.2f}"
              f"(n={len(TEST_EVAL)}) clean={r['em_clean']*100:.2f}%(n={N_CLEAN}) | "
              f"SEEN={r['em_seen']*100:.2f}%({r['n_seen']}) UNSEEN={r['em_unseen']*100:.2f}%"
              f"({r['n_unseen']}) | 数字每步={r['acc']*100:.2f}%±{r['acc_se']*100:.2f} | "
              f"R29非零比={r['r29_iv_nz']*100:.1f}% | R28={r['r28_same']}/{r['r28_k']} | "
              f"训练={wall_train/60:.2f}min 本run={r['wall_total']/60:.2f}min | sha16={dig}",
              flush=True)

# ============================================================================
# ⑦ 口径自检汇总 1–6
# ============================================================================
print("\n[S44] ===== ★口径自检 1–6 =====", flush=True)
_r28_all = [(a, s, RESULTS[(a, s)]["r28_same"], RESULTS[(a, s)]["r28_k"])
            for a in ALL_ARMS for s in SEEDS44 if (a, s) in RESULTS]
_r28_ok = all(x[2] == x[3] for x in _r28_all)
print(f"[S44-CHK1] EM 主口径=batch{EM_BATCH}；batch=1 vs batch{CHK_BATCH} 逐字一致性："
      f"{'全部一致 ✓ ' if _r28_ok else '★有不一致 '}"
      f"({sum(x[2] for x in _r28_all)}/{sum(x[3] for x in _r28_all)} 条逐字相同)", flush=True)
print(f"[S44-CHK2] SEEN/UNSEEN 分层见主表（泄漏 {LEAK}/800 = {100.0*LEAK/len(TEST):.2f}%）",
      flush=True)
print(f"[S44-CHK3] 地板（本桶 add_3d）：单答案={FLOOR_1*100:.2f}% / top5={FLOOR_5*100:.2f}%",
      flush=True)
_anchor = []
for s in SEEDS44:
    dd = (RESULTS[("A", s)]["em_full"] - ANCHOR_A_S38[s]) * 100
    _anchor.append(abs(dd) <= 1.0)
    for st, ref in ANCHOR_CURVE_S38[s].items():
        if st in RESULTS[("A", s)]["curve"]:
            cur = RESULTS[("A", s)]["curve"][st]["em_full"]
            print(f"[S44-ANCHOR-CURVE] A seed={s} @{st}: 本次={cur*100:.2f}% vs S38入库="
                  f"{ref*100:.2f}% ⇒ Δ={(cur-ref)*100:+.2f}pp", flush=True)
print(f"[S44-CHK4] 锚点复核 @12000（A 复用 ckpt）：{'全部复现 ✓（≤1pp）' if all(_anchor) else '★有 >1pp，已点名'}",
      flush=True)
print(f"[S44-CHK6] TEST sha16={TEST_SHA}（入库 {TEST_SHA_EXPECT}）| R29 Δ非零比见主表/Δ 表",
      flush=True)

# ============================================================================
# ⑧ 主表 + 收敛曲线 + Δ±SE 与逐条判定
# ============================================================================
print("\n[S44] ===== ★主表：五臂 × 2 seed @12000（全量/clean，EM 主口径 batch=1）=====", flush=True)
print("[S44-TABLE] 臂 | seed | EM全量 | EMclean | SEEN | UNSEEN | 数字每步 | 训练min | sha16",
      flush=True)
for a in ALL_ARMS:
    for s in SEEDS44:
        r = RESULTS.get((a, s))
        if not r:
            continue
        print(f"[S44-TABLE] {a} | {s} | {r['em_full']*100:.2f}% | {r['em_clean']*100:.2f}% | "
              f"{r['em_seen']*100:.2f}% | {r['em_unseen']*100:.2f}% | {r['acc']*100:.2f}% | "
              f"{r['wall_train']/60:.2f} | {r.get('sha', '-')}", flush=True)

print(f"\n[S44] ===== ★收敛曲线（EM 全量/clean，每 {GRID[1]-GRID[0]} 步）=====", flush=True)
for a in ALL_ARMS:
    for s in SEEDS44:
        r = RESULTS.get((a, s))
        if not r:
            continue
        cells = []
        for st in REPORT_AT:
            c = r["curve"].get(st)
            cells.append(f"@{st}=" + (f"{c['em_full']*100:.2f}/{c['em_clean']*100:.2f}%"
                                      if c else "-"))
        print(f"[S44-CURVE] {a} seed={s}: " + " ".join(cells), flush=True)
if not SMOKE44:
    print("[S44-CURVE] ★A 的 @3000 参考值（S38 入库日志，无 ckpt）：seed1234=37.25% seed5678=31.13%",
          flush=True)
    for s in SEEDS44:
        r = RESULTS[("A", s)]
        cells = []
        for st in (3000, 6000, 9000, 12000):
            cur = r["curve"].get(st, {}).get("em_full")
            cells.append(f"@{st}=" + (f"{cur*100:.2f}%" if cur is not None else
                                      f"{ANCHOR_CURVE_S38[s][st]*100:.2f}%(S38入库)"))
        print(f"[S44-CURVE] A seed={s}: " + " ".join(cells), flush=True)

print("\n[S44] ===== ★Δ = EM(臂) − EM(A) @12000（配对 ±SE，两种口径）=====", flush=True)
D_FULL: dict = {}
D_CLEAN: dict = {}
for a in TRAIN_ARMS:
    D_FULL[a], D_CLEAN[a] = {}, {}
    for s in SEEDS44:
        if (a, s) not in RESULTS or ("A", s) not in RESULTS:
            continue
        ra, rA = RESULTS[(a, s)], RESULTS[("A", s)]
        for tag, store, idx in (("FULL", D_FULL[a], None), ("CLEAN", D_CLEAN[a], CLEAN_IDX)):
            xa = ra["hits"] if idx is None else sub(ra["hits"], idx)
            xA = rA["hits"] if idx is None else sub(rA["hits"], idx)
            d = paired_stats(xa, xA)
            store[s] = d
            print(f"[S44-DELTA-{tag}] {a}−A seed={s}: Δ={d['d']*100:+.2f}±{d['se']*100:.2f}pp "
                  f"Δ/SE={d['ratio']:.2f} | R29 Δ非零比={d['nz']*100:.1f}% | A="
                  f"{frac(xA)*100:.2f}% → {a}={frac(xa)*100:.2f}% | {delta_tag(d)} | n={d['n']}",
                  flush=True)

print("\n[S44] ===== ★逐条判定（写死：全部 ≤−2SE 且 2 seed 同号 / 任一 |Δ|≤1SE 追平）=====",
      flush=True)
_neg2_ok, _catch_any, _unclear = [], [], []
for a in TRAIN_ARMS:
    lines = []
    ok_both = True
    for tag, store in (("全量", D_FULL[a]), ("clean", D_CLEAN[a])):
        if len(store) != len(SEEDS44):
            lines.append(f"{tag}=run不全")
            ok_both = False
            continue
        same_sign = all(d["d"] < 0 for d in store.values())
        neg2 = all(is_neg2(d) for d in store.values())
        catch = any(is_catch(d) for d in store.values())
        unclear = any((not is_neg2(d)) and (not is_catch(d)) for d in store.values())
        lines.append(f"{tag}: 全≤−2SE={'是' if neg2 else '否'} 同号={'是' if same_sign else '否'} "
                     f"追平={'有' if catch else '无'} 未过门槛={'有' if unclear else '无'} "
                     f"Δ均值={sum(d['d'] for d in store.values())/len(store)*100:+.2f}pp")
        ok_both &= (neg2 and same_sign)
        if catch:
            _catch_any.append(a)
        if unclear:
            _unclear.append(a)
    if ok_both:
        _neg2_ok.append(a)
    print(f"[S44-JUDGE] 臂 {a}: " + " || ".join(lines) +
          f" ⇒ {'★钉死（不可替代成立）' if ok_both else '★未过写死判据（如实报）'}", flush=True)

print("\n[S44] ===== ★饱和检查（R35）：9000→12000 段 EM 增量（全量）=====", flush=True)
_sat_bad = []
for a in ALL_ARMS:
    for s in SEEDS44:
        r = RESULTS.get((a, s))
        if not r or 9000 not in r["curve"] or 12000 not in r["curve"]:
            continue
        inc = (r["curve"][12000]["em_full"] - r["curve"][9000]["em_full"]) * 100
        inc_clean = (r["curve"][12000]["em_clean"] - r["curve"][9000]["em_clean"]) * 100
        note = "★未饱和（≥1pp）" if inc >= 1.0 else "已平（<1pp）"
        if inc >= 1.0:
            _sat_bad.append(f"{a}/s{s}")
        print(f"[S44-SAT] {a} seed={s}: EM 9000→12000 = {r['curve'][9000]['em_full']*100:.2f}→"
              f"{r['curve'][12000]['em_full']*100:.2f}% (Δ{inc:+.2f}pp) | clean "
              f"{r['curve'][9000]['em_clean']*100:.2f}→{r['curve'][12000]['em_clean']*100:.2f}% "
              f"(Δ{inc_clean:+.2f}pp) ⇒ {note}", flush=True)
for s in SEEDS44:
    r = RESULTS.get(("A", s))
    if r and 6000 in r["curve"] and 9000 in r["curve"] and 12000 in r["curve"]:
        print(f"[S44-SAT] A seed={s}（复用 ckpt 实测）：EM 6000→9000→12000 = "
              f"{r['curve'][6000]['em_full']*100:.2f}→{r['curve'][9000]['em_full']*100:.2f}→"
              f"{r['curve'][12000]['em_full']*100:.2f}% | 9000→12000="
              f"{(r['curve'][12000]['em_full']-r['curve'][9000]['em_full'])*100:+.2f}pp", flush=True)
_sat_b_arms = sorted({x.split("/")[0] for x in _sat_bad})
print(f"[S44-SAT-VERDICT] B/C/D/E 中 9000→12000 ≥1pp 的（臂,seed）={_sat_bad if _sat_bad else '无'}"
      f" ⇒ {'★有臂仍在上『未饱和』，按 R35 不得下结论' if _sat_bad else '★B/C/D/E 在 12000 步已饱和，可下结论'}"
      f" | 涉及臂={_sat_b_arms if _sat_b_arms else '无'}", flush=True)

# ---- 总判决 ----
if len(_neg2_ok) == 4 and not _catch_any:
    VERDICT = ("★B/C/D/E 全部 ≤−2SE 且 2 seed 同号（全量与 clean 两种口径都过）"
               "⇒「卡内结构不可替代」在 ≥12000 步仍成立（结论钉死）")
elif _catch_any:
    VERDICT = (f"★有臂在 12000 步追平 A（|Δ|≤1SE）：{sorted(set(_catch_any))}"
               f"⇒ @6000 的「不可替代」是欠训假象（如实报，比 E6 更重）")
else:
    VERDICT = (f"★未过写死判据：≤−2SE 且同号的臂={_neg2_ok}；追平={sorted(set(_catch_any))}；"
               f"未过门槛={sorted(set(_unclear))} ⇒ 如实报")
print(f"\n[S44-VERDICT] {VERDICT}", flush=True)

print("\n[S44] ===== 成本与 ckpt =====", flush=True)
for a in ALL_ARMS:
    for s in SEEDS44:
        r = RESULTS.get((a, s))
        if not r:
            continue
        print(f"[META] {a} seed={s}: trained={r['trained']} steps={STEPS44 if r['trained'] else 12000} "
              f"训练墙钟={r['wall_train']/60:.2f}min 本run={r['wall_total']/60:.2f}min "
              f"ckpt={os.path.basename(r['ckpt'])} sha256[:16]={r.get('sha', '-')}", flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"D={D} FF={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} | EM口径=batch{EM_BATCH} "
      f"自检=batch{CHK_BATCH} | TEST sha16={TEST_SHA} 泄漏={LEAK}/{len(TEST)} | 地板(单/top5)="
      f"{FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% | clean n={N_CLEAN} | 总墙钟={(time.time()-T0)/60:.1f}min "
      f"| run 数={len(RESULTS)}/{len(ALL_ARMS)*len(SEEDS44)}", flush=True)
print("[DONE] exit=0", flush=True)
