#!/usr/bin/env python3
"""S34 · E5：补上缺的那一格 —— frozen@6000 vs e2e@6000（同数据 · 同 seed · 等总步数 6000）。

唯一问题：等总步数 6000 下，「冻结接口 + 只训目标卡」(frozen) 与「端到端全参」(e2e) 谁更好？
  A = frozen@6000：从 S25 臂1 的 3000 步 ckpt 分叉，冻结输入卡/输出卡，只训目标卡 3000 步
      ⇒ 总 6000 步 = S25 臂2（anchor）的逐位同一路径；本单元重跑它，拿曲线/墙钟/配对 Δ，
      并与 logs/25_ckpt_anchor_seed*.pt 逐参数比对（同一实现的可核验凭据）。
  B = e2e@6000：logs/28_ckpt_e2e6000_seed*.pt（已有 ckpt），先复核能否复现 28_train.log 的
      85.62%(s1234) / 74.38%(s5678) 与数字每步 84.27% / 76.33%。

同时把「算力对齐」做实（这是裁判判 A8 只部分成立的点）：
  · Phase 0 在同一实现、同一会话里重算 e2e 前 3000 步底座，测它的纯训练墙钟与 s/step；
  · Phase A 测 frozen 3000 步的纯训练墙钟与 s/step；
  · 两臂总步数 = 6000 = 6000，总前向次数 = 6000 = 6000（每步 batch=32 一次前向）；
    反传不同：e2e 每步反传 2,501,888 参数，frozen 每步只反传目标卡 198,272 参数
    ⇒ 本单元的「等算力」定义 = 等步数 + 等前向次数；墙钟作为经验校验一并报。
  · 因此 frozen 的总墙钟必须 = 底座(3000 步) + 冻结段(3000 步)，而不能只报冻结段那 2.7min。

自检（缺一即无效）：
  1 门 A：被冻结卡 max|Δθ| == 0（精确）、目标卡有梯度且更新；报参数量分组；
  2 锚点复现：frozen 段每 750 步 EM 需与 S28 已入库的 frozen_traj 一致，末点需复现 S25 臂2
    （80.38/61.12，≤1pp）；Phase 0 的 e2e@3000 需复现 37.25/31.13 且与 S25 臂1 ckpt 逐位一致；
    B 需复现 28_train.log 的 6000 步 EM（≤1pp）；
  3 R28：EM 主口径 batch=1；报一次 batch=1 vs 16 一致性；
  4 算力对齐：见上。

数据：只用 add_3d（train4000/test800），必须复现 S31 指纹 TEST sha256[:16]=b34e7ea515203227、
  泄漏 16/800。数据生成器与超参与 S19-res/S25/S28 逐字相同（d=128 ff=512 NHEAD=4 lr=1e-3
  batch=32 MAXLEN=512 思维卡带残差）。
ckpt：logs/34_ckpt_frozen6000_seed{1234,5678}.pt；结果增量落 logs/34_results.jsonl。
只允许写：本文件、logs/e5_frozen.log、logs/34_*、/tmp。
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

# ---------------- device 纪律（ROCm 下报成 cuda，正常） ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
if device != "cuda":
    print("GPU 起不来，立即停（不用 CPU 硬跑）。", flush=True)
    print("torch.cuda.is_available() =", torch.cuda.is_available(), flush=True)
    print("HSA_OVERRIDE_GFX_VERSION =", os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)
    sys.exit(3)

print("torch:", torch.__version__, "| HSA_OVERRIDE_GFX_VERSION =",
      os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)

# ---------------- 配置（与 S19-res / S25 / S28 逐字一致） ----------------
D, FF, NHEAD, MAXLEN = 128, 512, 4, 512
MIN_T, TAIL_KEEP = 48, 24
LR, BATCH, GEN_BATCH, EM_BATCH, CHK_BATCH = 1e-3, 32, 32, 1, 16
N_TRAIN, N_TEST, MAX_GEN = 4000, 800, 48
BUCKET, BI = "add_3d", 2
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
BASE_STEPS = 3000          # e2e 底座（= S25 臂1 的 3000 步）
FROZEN_STEPS = 3000        # frozen 段（= S25 臂2 的第二段）⇒ 总 6000
EVAL_AT = (750, 1500, 2250, 3000)
SEEDS = tuple(int(x) for x in os.environ.get("S34_SEEDS", "1234,5678").split(","))
SMOKE = os.environ.get("S34_SMOKE") == "1"
if SMOKE:
    BASE_STEPS, FROZEN_STEPS, N_TRAIN, N_TEST = 30, 30, 300, 60
    EVAL_AT = (15, 30)
CKPT_TMPL = LOGDIR + "/34_ckpt_frozen6000_seed{seed}.pt"
RESULT_JSONL = "/tmp/34_results_smoke.jsonl" if SMOKE else LOGDIR + "/34_results.jsonl"

S25_E2E = LOGDIR + "/25_ckpt_e2e_seed{seed}.pt"
S25_ANCHOR = LOGDIR + "/25_ckpt_anchor_seed{seed}.pt"
S28_E2E6000 = LOGDIR + "/28_ckpt_e2e6000_seed{seed}.pt"

# 入库锚点
E2E3000_ANCHOR = {1234: 0.3725, 5678: 0.3113}
FROZEN6000_ANCHOR = {1234: 0.8038, 5678: 0.6112}          # S25 臂2（= frozen@6000）
E2E6000_ANCHOR = {1234: 0.8562, 5678: 0.7438}             # 28_train.log
FROZEN_TRAJ_S28 = {1234: (0.6188, 0.7275, 0.7612, 0.8038),
                   5678: (0.5212, 0.5562, 0.5700, 0.6112)}
S25_WALL_BASE = {1234: 141.3, 5678: 142.2}                # S25 臂1 纯训练墙钟（只读引用）
S28_WALL_E2E6000 = {1234: 303.5, 5678: 317.3}             # 28_results.jsonl（含 4 次中途评估）

t_start = time.time()
print(f"[S34] ★E5 frozen@6000 vs e2e@6000：桶={BUCKET} seeds={SEEDS} "
      f"底座={BASE_STEPS}步 + 冻结段={FROZEN_STEPS}步 ⇒ 总 {BASE_STEPS + FROZEN_STEPS} 步 | "
      f"d={D} ff={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN}(prompt优先) "
      f"| EM主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | smoke={SMOKE}", flush=True)

tok = Tokenizer.from_file(TOK)
V = tok.get_vocab_size()
PAD_ID = tok.token_to_id("<pad>")
EOS_ID = tok.token_to_id("<eos>")


def enc(t: str) -> list[int]:
    return tok.encode(t, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


# ============================================================================
# ① 数据（与 S14/S19/S25/S28 逐字相同；只用 add_3d）
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
    assert len({t for t, _ in ts}) == len(ts), "模板有重复"
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
            raise RuntimeError(f"{name}{tag}: 文本空间不足")
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


_tr_rng, _te_rng = Rng(14000 + BI * 7), Rng(14900 + BI * 7)
_used: set[str] = set()
TRAIN = [encode_record(x) for x in gen_half(BUCKET, N_TRAIN, _tr_rng, _used, "train")]
TEST = [encode_record(x) for x in gen_half(BUCKET, N_TEST, _te_rng, _used, "test")]
assert len(_used) == len(TRAIN) + len(TEST), "train/test 文本池计数异常（应零重叠）"

_ys = answer_values(BUCKET)
_top5 = [v for v, _ in Counter(r["gold"] for r in TRAIN).most_common(5)]
_prior = sum(1 for r in TEST if r["gold"] in _top5) / len(TEST)
_h = hashlib.sha256()
for r in TEST:
    _h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
TEST_SHA = _h.hexdigest()[:16]
_tp = {(r["a"], r["b"]) for r in TRAIN}
LEAK_AB = sum(1 for r in TEST if (r["a"], r["b"]) in _tp)
print(f"[S34-DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)} 文本零重叠 | "
      f"答案域={min(_ys)}..{max(_ys)}({len(_ys)}个) | 最高频5答案={_top5} 机会水平={100*_prior:.1f}%",
      flush=True)
if not SMOKE:
    ok_sha = TEST_SHA == "b34e7ea515203227"
    ok_leak = LEAK_AB == 16
    print(f"[S34-DATA-CHECK] TEST 指纹 sha256[:16]={TEST_SHA} vs S31 入库 b34e7ea515203227 ⇒ "
          f"{'逐位一致 ✓' if ok_sha else '★不一致'} | (a,b) 泄漏={LEAK_AB}/800 vs S31 入库 16 ⇒ "
          f"{'一致 ✓' if ok_leak else '★不一致'}", flush=True)
    assert ok_sha and ok_leak, "数据指纹不符 ⇒ 与 S25/S28/S31 不是同一数据集，本单元无效"
else:
    print(f"[S34-DATA-CHECK] SMOKE 模式跳过指纹断言（sha16={TEST_SHA} 泄漏={LEAK_AB}）", flush=True)


# ============================================================================
# ② 模型（S25 的 Cards 逐字拷贝：h = h + Think(h)）
# ============================================================================
def sin_pe(max_len, d):
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def enc_layer(d, ff):
    return nn.TransformerEncoderLayer(d, NHEAD, ff, dropout=0.1, activation="gelu",
                                      batch_first=True, norm_first=True)


class Cards(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        self.thought = enc_layer(d, ff)
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
        h = h + self.thought(h, src_mask=m)
        return self.head(h)


def group_of(name):
    if name.startswith("emb.") or name.startswith("in_enc."):
        return "输入卡"
    if name.startswith("thought."):
        return "目标卡"
    if name.startswith("head."):
        return "输出卡"
    raise KeyError(f"未知参数分组：{name}")


GROUP_ORDER = ("输入卡", "目标卡", "输出卡")


def set_trainable(model, groups):
    for n, p in model.named_parameters():
        p.requires_grad_(group_of(n) in groups)


def group_counts(model):
    c = defaultdict(int)
    for n, p in model.named_parameters():
        c[group_of(n)] += p.numel()
    return c


def snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def check_grads(model, trainable, where):
    for n, p in model.named_parameters():
        g = group_of(n)
        if g in trainable:
            if p.grad is None:
                raise SystemExit(f"[门A-FAIL] {where}: 可训组 {g} 的参数 {n} grad is None")
        else:
            if p.grad is not None:
                raise SystemExit(f"[门A-FAIL] {where}: 冻结组 {g} 的参数 {n} grad 非 None")


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


def masked_ce(model, recs):
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


def ckpt_max_delta(pa, pb):
    """两个 state_dict 的逐参数 max|Δ|（按卡分组）。"""
    sa = torch.load(pa, map_location="cpu")
    sb = torch.load(pb, map_location="cpu")
    agg = defaultdict(float)
    for k in sa:
        agg[group_of(k)] = max(agg[group_of(k)], float((sa[k].float() - sb[k].float()).abs().max()))
    return max(agg.values()), dict(agg)


# ============================================================================
# ③ 训练：底座 e2e（Phase 0，用于墙钟与同一实现核验）与冻结段（Phase A）
# ============================================================================
def perm_and_pos(seed):
    order = torch.randperm(len(TRAIN), generator=torch.Generator().manual_seed(seed))
    return order, 0


def train_steps(model, opt, steps, seed, groups, tag, do_gate=False):
    """与 S25 run_arm 的训练循环逐字同构。返回 (墙钟秒, 每步 last loss, 是否门A通过)。"""
    order, pos = perm_and_pos(seed)
    trainable = set(groups)
    if do_gate:
        for p in model.parameters():
            p.grad = None
        before = snapshot(model)
    model.train()
    t0 = time.time()
    last_loss = float("nan")
    gate_ok = None
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
        if do_gate and (k % 250 == 0 or k == steps):
            check_grads(model, trainable, f"{tag} step={k}")
        opt.step()
        last_loss = loss.item()
        if k % 500 == 0 or k == steps:
            el = time.time() - t0
            print(f"  [train] {tag} step={k}/{steps} loss={loss.item():.4f} elapsed={el:.0f}s "
                  f"s/step={el/k:.4f}", flush=True)
    wall = time.time() - t0
    if do_gate:
        rep = {}
        for g in GROUP_ORDER:
            mx, n_grad = 0.0, 0
            for n, p in model.named_parameters():
                if group_of(n) != g:
                    continue
                mx = max(mx, (before[n] - p.detach()).abs().max().item())
                n_grad += 1 if p.grad is not None else 0
            rep[g] = dict(max_delta=mx, n_grad=n_grad)
        line = []
        for g in GROUP_ORDER:
            r = rep[g]
            if g in trainable:
                assert r["n_grad"] > 0, f"门A失败：可训组 {g} 无梯度"
                assert r["max_delta"] > 0.0, f"门A失败：可训组 {g} 未更新"
                line.append(f"{g}: max|Δθ|={r['max_delta']:.3e}(>0 ✓) grad≠None({r['n_grad']}个) ✓")
            else:
                assert r["max_delta"] == 0.0, f"门A失败：冻结组 {g} 被改动 {r['max_delta']:.3e}"
                assert r["n_grad"] == 0, f"门A失败：冻结组 {g} 有梯度"
                line.append(f"{g}: max|Δθ|={r['max_delta']:.3e}(==0 ✓) grad=None ✓")
        gate_ok = rep
        print(f"[门A] {tag} 步数={steps} ⇒ " + " | ".join(line), flush=True)
    return wall, last_loss, gate_ok


def make_e2e(seed):
    """与 S25 臂1 逐字同构：manual_seed → Cards → 全参可训。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards(D, FF).to(device)
    set_trainable(model, set(GROUP_ORDER))
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=LR)
    return model, opt


def make_frozen_from(ckpt, seed):
    """与 S25 臂2 逐字同构：manual_seed → Cards → 载入臂1 ckpt → 只训目标卡。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards(D, FF)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model = model.to(device)
    set_trainable(model, {"目标卡"})
    params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(params, lr=LR)
    return model, opt


# ============================================================================
# ④ 评估（R28 主口径 batch=1；CE/数字每步与 S25/S28 同定义）
# ============================================================================
@torch.no_grad()
def greedy_bs1(model, recs):
    model.eval()
    outs = []
    for r in recs:
        ids = torch.tensor([r["p"]], dtype=torch.long, device=device)
        cap = min(MAXLEN - len(r["p"]), MAX_GEN)
        got = []
        for _ in range(cap):
            nxt = int(model.logits(ids)[0, -1].argmax().item())
            if nxt == EOS_ID:
                break
            got.append(nxt)
            ids = torch.cat([ids, torch.tensor([[nxt]], dtype=torch.long, device=device)], 1)
        outs.append(dec(got))
    return outs


@torch.no_grad()
def greedy_bsN(model, recs, bs):
    model.eval()
    outs = [""] * len(recs)
    for i in range(0, len(recs), bs):
        ch = recs[i: i + bs]
        lens = [len(r["p"]) for r in ch]
        n0 = max(lens)
        ids = torch.full((len(ch), n0), PAD_ID, dtype=torch.long, device=device)
        for j, r in enumerate(ch):
            ids[j, :lens[j]] = torch.tensor(r["p"], dtype=torch.long)
        cap = [min(MAXLEN - L, MAX_GEN) for L in lens]
        gen = [[] for _ in ch]
        done = [False] * len(ch)
        last_col = [L - 1 for L in lens]
        for _ in range(max(cap)):
            if all(done):
                break
            lg = model.logits(ids)
            nxt = []
            for j in range(len(ch)):
                if done[j]:
                    nxt.append(PAD_ID)
                    continue
                t = int(lg[j, last_col[j]].argmax().item())
                gen[j].append(t)
                nxt.append(t)
                if t == EOS_ID or len(gen[j]) >= cap[j]:
                    done[j] = True
                last_col[j] = n0 + len(gen[j]) - 1
            ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long,
                                               device=device).unsqueeze(1)], 1)
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
    model.eval()
    samp_all, dig_per = [], []
    for i in range(0, len(recs), GEN_BATCH):
        ch = recs[i: i + GEN_BATCH]
        lens = [len(r["p"]) + len(r["t"]) for r in ch]
        n = max(lens)
        ids = torch.full((len(ch), n), PAD_ID, dtype=torch.long, device=device)
        for j, r in enumerate(ch):
            ids[j, :len(r["p"])] = torch.tensor(r["p"], dtype=torch.long)
            ids[j, len(r["p"]): len(r["p"]) + len(r["t"])] = torch.tensor(r["t"], dtype=torch.long)
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        for j, r in enumerate(ch):
            s = len(r["p"])
            lp = logp[j, s - 1: s + len(r["t"]) - 1]
            tgt = torch.tensor(r["t"], dtype=torch.long, device=device)
            ces = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1).tolist()
            samp_all.append(sum(ces) / len(ces))
            d = [c for c, tk in zip(ces, r["t"]) if digit_token(tk)]
            if d:
                dig_per.append(sum(d) / len(d))
    ce = sum(samp_all) / len(samp_all)
    dm = sum(dig_per) / len(dig_per)
    return ce, math.exp(-dm), dm


def strict_of(texts, recs):
    return [int(my_parse(t) == r["gold"]) for t, r in zip(texts, recs)]


def binomial_se(xs):
    n = len(xs)
    s = sum(xs) / n
    return math.sqrt(s * (1 - s) / n)


def full_eval(model, tag, do_r28=False):
    t0 = time.time()
    txt = greedy_bs1(model, TEST)
    st = strict_of(txt, TEST)
    em, se = sum(st) / len(st), binomial_se(st)
    ce, acc, dm = ce_and_acc(model, TEST)
    out = dict(em=em, em_se=se, acc=acc, dig_ce=dm, ce=ce, strict=st,
               em_raw=sum(st), n=len(st))
    print(f"[EM] {tag}: EM={em*100:.2f}%±{se*100:.2f}(n={len(st)}) | "
          f"数字每步={acc*100:.2f}% | 整体CE={ce:.4f} | 耗时={time.time()-t0:.0f}s", flush=True)
    if do_r28:
        sub = TEST[:CHK_BATCH]
        t1 = greedy_bs1(model, sub)
        tN = greedy_bsN(model, sub, CHK_BATCH)
        same = sum(a == b for a, b in zip(t1, tN))
        out["r28_same"], out["r28_k"] = same, len(sub)
        print(f"[R28] {tag}: batch=1 vs batch={CHK_BATCH} 逐字一致 {same}/{len(sub)} ⇒ "
              f"{'一致' if same == len(sub) else '★不一致（本单元 EM 全部走 batch=1 口径）'}",
              flush=True)
    return out


def paired(xa, xb):
    n = len(xa)
    xs = [a - b for a, b in zip(xa, xb)]
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    se = math.sqrt(var / n)
    nz = sum(1 for x in xs if x != 0) / n
    return dict(n=n, d=m, se=se, ratio=(m / se if se > 0 else float("inf")), nz=nz,
                cover0=(abs(m) <= se), ge2se=(m >= 2 * se), le_neg2se=(m <= -2 * se))


def append_jsonl(obj):
    with open(RESULT_JSONL, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


# ============================================================================
# ⑤ 主循环
# ============================================================================
RES: dict[int, dict] = {}
for seed in SEEDS:
    print(f"\n[S34] ================ seed={seed} ================", flush=True)
    r: dict = dict(seed=seed)
    for p in (S25_E2E.format(seed=seed), S25_ANCHOR.format(seed=seed),
              S28_E2E6000.format(seed=seed)):
        if not os.path.exists(p):
            print(f"[STOP] ★缺前置 ckpt {p}", flush=True)
            print("[DONE] exit=5", flush=True)
            sys.exit(5)

    # ---- Phase 0：底座 e2e@3000（同一实现复算 + 墙钟 + 逐位核验 S25 臂1）----
    print(f"[S34-P0] 底座 e2e {BASE_STEPS} 步（全参；与 S25 臂1 同实现）seed={seed}", flush=True)
    m0, o0 = make_e2e(seed)
    cnt = group_counts(m0)
    print(f"[门A] 参数量分组: " + " | ".join(f"{g}={cnt[g]:,}" for g in GROUP_ORDER)
          + f" | 合计={sum(cnt.values()):,}（输入卡=emb+in_enc，目标卡=thought，输出卡=head）",
          flush=True)
    r["counts"] = {g: cnt[g] for g in GROUP_ORDER}
    r["n_params"] = sum(cnt.values())
    wall0, loss0, _ = train_steps(m0, o0, BASE_STEPS, seed, GROUP_ORDER, f"e2e{BASE_STEPS} seed={seed}")
    r["wall_e2e_base"] = wall0
    s_per_step = wall0 / BASE_STEPS
    r["s_per_step_e2e"] = s_per_step
    mx0 = float("nan")
    if not SMOKE:
        torch.save(m0.state_dict(), LOGDIR + f"/34_ckpt_e2e3000_seed{seed}.pt")
        mx0, _ = ckpt_max_delta(LOGDIR + f"/34_ckpt_e2e3000_seed{seed}.pt",
                                S25_E2E.format(seed=seed))
        print(f"[S34-P0] 与 S25 臂1 ckpt 逐参数 max|Δθ|={mx0:.3e} ⇒ "
              f"{'逐位同一模型 ✓（同一实现可核验）' if mx0 == 0 else '★不完全逐位（见下）'}", flush=True)
    ev0 = full_eval(m0, f"{BUCKET} e2e@{BASE_STEPS} seed={seed}")
    dev_a = (ev0["em"] - E2E3000_ANCHOR[seed]) * 100
    print(f"[锚点-P0] e2e@{BASE_STEPS} seed={seed}: 入库={E2E3000_ANCHOR[seed]*100:.2f}% vs "
          f"本次={ev0['em']*100:.2f}% Δ={dev_a:+.2f}pp ⇒ "
          f"{'一致(|Δ|≤1pp) ✓' if abs(dev_a) <= 1.0 else '★偏离 >1pp'}", flush=True)
    r["e2e3000_em"], r["e2e3000_acc"] = ev0["em"], ev0["acc"]
    r["e2e3000_anchor_dev"] = dev_a
    r["e2e3000_bitwise"] = mx0
    del m0, o0
    torch.cuda.empty_cache()

    # ---- Phase A：frozen@6000 = 从 S25 臂1 ckpt 分叉，只训目标卡 3000 步 ----
    print(f"[S34-PA] frozen 段：载入 S25 臂1 ckpt（3000 步）→ 冻结输入/输出卡 → 只训目标卡 "
          f"{FROZEN_STEPS} 步 ⇒ 总 {BASE_STEPS + FROZEN_STEPS} 步", flush=True)
    mA, oA = make_frozen_from(S25_E2E.format(seed=seed), seed)
    n_opt = sum(p.numel() for p in oA.param_groups[0]["params"])
    print(f"[门A] 优化器只含目标卡参数：张量 {len(oA.param_groups[0]['params'])} 个 "
          f"/ 参数量 {n_opt:,}（= 目标卡 {cnt['目标卡']:,}）⇒ "
          f"{'✓' if n_opt == cnt['目标卡'] else '★不符'}", flush=True)
    r["opt_params"] = n_opt

    order, pos = perm_and_pos(seed)          # 与训练循环共用同一排列（曲线评估不打断）
    before = snapshot(mA)
    mA.train()
    curve = []
    t0 = time.time()
    wall_evals = 0.0
    last_loss = float("nan")
    tr_marks = []
    for k in range(1, FROZEN_STEPS + 1):
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(TRAIN))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(mA, [TRAIN[i] for i in idx])
        oA.zero_grad(set_to_none=True)
        loss.backward()
        if k % 250 == 0 or k == FROZEN_STEPS:
            check_grads(mA, {"目标卡"}, f"frozen seed={seed} step={k}")
        oA.step()
        last_loss = loss.item()
        if k % 500 == 0 or k == FROZEN_STEPS:
            print(f"  [train] frozen seed={seed} step={k}/{FROZEN_STEPS} loss={loss.item():.4f} "
                  f"elapsed={time.time()-t0-wall_evals:.0f}s "
                  f"s/step={(time.time()-t0-wall_evals)/k:.4f}", flush=True)
            tr_marks.append((k, loss.item()))
        if k in EVAL_AT:
            te = time.time()
            ev = full_eval(mA, f"frozen seed={seed} 冻结段step={k}（总{BASE_STEPS+k}步）")
            wall_evals += time.time() - te
            curve.append(dict(frozen_step=k, total_step=BASE_STEPS + k, em=ev["em"],
                              acc=ev["acc"], ce=ev["ce"]))
            mA.train()
    wall_frozen = (time.time() - t0) - wall_evals
    r["wall_frozen"] = wall_frozen
    r["s_per_step_frozen"] = wall_frozen / FROZEN_STEPS
    r["wall_frozen_with_evals"] = time.time() - t0
    r["curve"] = curve
    r["train_marks"] = tr_marks

    # 门 A（段末逐位）
    rep = {}
    for g in GROUP_ORDER:
        mx, n_grad = 0.0, 0
        for n, p in mA.named_parameters():
            if group_of(n) != g:
                continue
            mx = max(mx, (before[n] - p.detach()).abs().max().item())
            n_grad += 1 if p.grad is not None else 0
        rep[g] = dict(max_delta=mx, n_grad=n_grad)
    line = []
    for g in GROUP_ORDER:
        rr = rep[g]
        if g == "目标卡":
            assert rr["n_grad"] > 0 and rr["max_delta"] > 0.0, "门A失败：目标卡未更新/无梯度"
            line.append(f"{g}: max|Δθ|={rr['max_delta']:.3e}(>0 ✓) grad≠None({rr['n_grad']}个) ✓")
        else:
            assert rr["max_delta"] == 0.0 and rr["n_grad"] == 0, f"门A失败：{g} 被改动"
            line.append(f"{g}: max|Δθ|={rr['max_delta']:.3e}(==0 ✓) grad=None ✓")
    r["gate"] = rep
    print(f"[门A] frozen@6000 seed={seed} 段末 ⇒ " + " | ".join(line), flush=True)

    frozen_ck = CKPT_TMPL.format(seed=seed)
    torch.save(mA.state_dict(), frozen_ck)
    r["ckpt"] = frozen_ck
    r["sha"] = sha16(frozen_ck)
    print(f"[CKPT] saved {frozen_ck} sha256[:16]={r['sha']}", flush=True)

    mxA, aggA = ckpt_max_delta(frozen_ck, S25_ANCHOR.format(seed=seed))
    print(f"[锚点-PA] frozen@6000 与 S25 臂2 ckpt 逐参数 max|Δθ|={mxA:.3e}（分组 "
          + " | ".join(f"{g}:{aggA.get(g, float('nan')):.2e}" for g in GROUP_ORDER)
          + f"）⇒ {'逐位同一 ✓（与 S25 臂2 逐位可比）' if mxA == 0 else '★不完全逐位'}", flush=True)
    r["anchor_bitwise"] = mxA

    # 末点主结果 + R28
    evA = full_eval(mA, f"{BUCKET} frozen@6000 seed={seed}", do_r28=True)
    dev_f = (evA["em"] - FROZEN6000_ANCHOR[seed]) * 100
    print(f"[锚点-PA] frozen@6000 seed={seed}: S25 臂2 入库={FROZEN6000_ANCHOR[seed]*100:.2f}% vs "
          f"本次={evA['em']*100:.2f}% Δ={dev_f:+.2f}pp ⇒ "
          f"{'一致(|Δ|≤1pp) ✓' if abs(dev_f) <= 1.0 else '★偏离 >1pp'}", flush=True)
    r["frozen"] = {k: v for k, v in evA.items() if k != "strict"}
    r["frozen_strict"] = evA["strict"]
    r["frozen_anchor_dev"] = dev_f
    del mA, oA
    torch.cuda.empty_cache()

    # ---- Phase B：e2e@6000 已有 ckpt 复核 ----
    mB = Cards(D, FF)
    mB.load_state_dict(torch.load(S28_E2E6000.format(seed=seed), map_location="cpu"))
    mB = mB.to(device)
    for p in mB.parameters():
        p.requires_grad_(False)
    evB = full_eval(mB, f"{BUCKET} e2e@6000 seed={seed}", do_r28=True)
    dev_b = (evB["em"] - E2E6000_ANCHOR[seed]) * 100
    print(f"[锚点-PB] e2e@6000 seed={seed}: 28_train.log 入库={E2E6000_ANCHOR[seed]*100:.2f}% vs "
          f"本次复核={evB['em']*100:.2f}% Δ={dev_b:+.2f}pp ⇒ "
          f"{'一致(|Δ|≤1pp) ✓' if abs(dev_b) <= 1.0 else '★偏离 >1pp'}", flush=True)
    r["e2e6000"] = {k: v for k, v in evB.items() if k != "strict"}
    r["e2e6000_strict"] = evB["strict"]
    r["e2e6000_anchor_dev"] = dev_b
    r["e2e6000_sha"] = sha16(S28_E2E6000.format(seed=seed))
    del mB
    torch.cuda.empty_cache()

    d = paired(evA["strict"], evB["strict"])
    r["paired"] = d
    print(f"[S34-Δ] seed={seed}: Δ=EM(frozen@6000)−EM(e2e@6000)={d['d']*100:+.2f}pp±{d['se']*100:.2f} "
          f"| Δ/SE={d['ratio']:.2f} | Δ非零={d['nz']*100:.1f}% | 1SE覆盖0={d['cover0']}", flush=True)
    RES[seed] = r
    append_jsonl({k: v for k, v in r.items()
                  if k not in ("frozen_strict", "e2e6000_strict")})

# ============================================================================
# ⑥ 汇总：主表 / Δ 判定 / 曲线 / 算力对齐
# ============================================================================
print("\n[S34] ===== ★主表：两臂 × 2 seed（EM 主口径 batch=1，n=800） =====", flush=True)
print("[S34] 臂 | seed | EM±SE | 数字每步 | 整体CE | R28 | 总步数 | sha16", flush=True)
for seed in SEEDS:
    r = RES.get(seed)
    if not r:
        continue
    f, b = r["frozen"], r["e2e6000"]
    print(f"[S34] A=frozen@6000 | {seed} | {f['em']*100:.2f}%±{f['em_se']*100:.2f} | "
          f"{f['acc']*100:.2f}% | {f['ce']:.4f} | {f.get('r28_same','-')}/{f.get('r28_k','-')} | "
          f"{BASE_STEPS+FROZEN_STEPS} | {r['sha']}", flush=True)
    print(f"[S34] B=e2e@6000    | {seed} | {b['em']*100:.2f}%±{b['em_se']*100:.2f} | "
          f"{b['acc']*100:.2f}% | {b['ce']:.4f} | {b.get('r28_same','-')}/{b.get('r28_k','-')} | "
          f"6000 | {r['e2e6000_sha']}", flush=True)

print("\n[S34] ===== ★配对 Δ = EM(frozen@6000) − EM(e2e@6000)（逐 test 样本配对） =====",
      flush=True)
signs = []
ok2se_pos = ok2se_neg = ok1se = True
for seed in SEEDS:
    r = RES.get(seed)
    if not r:
        continue
    d = r["paired"]
    signs.append(1 if d["d"] > 0 else (-1 if d["d"] < 0 else 0))
    ok2se_pos &= d["ge2se"]
    ok2se_neg &= d["le_neg2se"]
    ok1se &= d["cover0"]
    print(f"[S34-Δ] seed={seed}: Δ={d['d']*100:+.2f}pp±{d['se']*100:.2f} | Δ/SE={d['ratio']:.2f} | "
          f"Δ非零={d['nz']*100:.1f}% | |Δ|≤1SE={'是' if d['cover0'] else '否'} | "
          f"Δ≥2SE={'是' if d['ge2se'] else '否'} | Δ≤−2SE={'是' if d['le_neg2se'] else '否'}",
          flush=True)
same_pos = len(set(signs)) == 1 and signs and signs[0] == 1
same_neg = len(set(signs)) == 1 and signs and signs[0] == -1
if same_neg and ok2se_neg:
    verdict = "★e2e 更好（2 seed 同号为负且 Δ≤−2SE）⇒ A8 成立、B⑦（训练量不是瓶颈）被证伪"
elif same_pos and ok2se_pos:
    verdict = "★frozen 更好（2 seed 同号为正且 Δ≥2SE）⇒ S28 的『反超』被推翻"
elif ok1se:
    verdict = "两者等价（2 seed |Δ|≤1SE）⇒ M2 的核心主张（单卡训练）不成立"
else:
    verdict = "未落入任一写死判定（见上逐 seed 数值）"
print(f"[判定] 同号正={same_pos} 同号负={same_neg} 都≤1SE={ok1se} 都≥2SE={ok2se_pos} "
      f"都≤−2SE={ok2se_neg} ⇒ {verdict}", flush=True)

print("\n[S34] ===== ★frozen 的 3000→6000 曲线（每 750 步，EM / 数字每步） =====", flush=True)
for seed in SEEDS:
    r = RES.get(seed)
    if not r:
        continue
    pts = [("3000(底座e2e)", r["e2e3000_em"])] + \
          [(str(c["total_step"]), c["em"]) for c in r["curve"]]
    s = " → ".join(f"{k}:{v*100:.2f}%" for k, v in pts)
    inc = [f"{r['curve'][i]['total_step']}:{(r['curve'][i]['em']-prev)*100:+.2f}pp"
           for i, prev in enumerate([r["e2e3000_em"]] + [c["em"] for c in r["curve"][:-1]])]
    rising = all(RES[seed]["curve"][i]["em"] > RES[seed]["curve"][i - 1]["em"]
                 for i in range(1, len(RES[seed]["curve"])))
    print(f"[S34-CURVE] seed={seed}: {s}", flush=True)
    print(f"[S34-CURVE] seed={seed}: 相邻增量 " + " | ".join(inc)
          + f" ⇒ 冻结段{'仍在单调上升' if rising else '已非单调（出现回落）'}"
          + (f" | 与 S28 入库 frozen_traj 逐点差="
             + " | ".join(f"{(c['em']-a)*100:+.2f}pp"
                          for c, a in zip(RES[seed]["curve"], FROZEN_TRAJ_S28[seed]))
             if len(RES[seed]["curve"]) == 4 and not SMOKE else ""), flush=True)

print("\n[S34] ===== ★算力对齐（等步数 / 等前向 / 反传不同） =====", flush=True)
for seed in SEEDS:
    r = RES.get(seed)
    if not r:
        continue
    tot_frozen_path = r["wall_e2e_base"] + r["wall_frozen"]
    e2e6000_est = r["s_per_step_e2e"] * 6000
    print(f"[S34-算力] seed={seed}: frozen 路径 = 底座{BASE_STEPS}步 {r['wall_e2e_base']:.0f}s"
          f"({r['s_per_step_e2e']:.4f}s/步) + 冻结段{FROZEN_STEPS}步 {r['wall_frozen']:.0f}s"
          f"({r['s_per_step_frozen']:.4f}s/步) ⇒ 总 {tot_frozen_path:.0f}s "
          f"({tot_frozen_path/60:.2f}min) / 6000 步", flush=True)
    print(f"[S34-算力] seed={seed}: e2e@6000 = 6000 步（本会话实测 s/步×6000 ≈ {e2e6000_est:.0f}s "
          f"= {e2e6000_est/60:.2f}min；28_results.jsonl 实录 {S28_WALL_E2E6000[seed]:.0f}s 含 4 次"
          f"中途评估）| S25 臂1 实录底座 {S25_WALL_BASE[seed]:.0f}s", flush=True)
    print(f"[S34-算力] seed={seed}: 总步数 6000=6000 ✓ | 总前向次数 6000=6000 ✓ | "
          f"反传参数量/步 frozen={r['opt_params']:,} vs e2e={r['n_params']:,}（frozen 反传更少）"
          f" ⇒ 墙钟差={tot_frozen_path-e2e6000_est:+.0f}s（{100*(tot_frozen_path/e2e6000_est-1):+.1f}%）",
          flush=True)

# 门 A / 自检汇总
print("\n[S34] ===== ★自检汇总 =====", flush=True)
for seed in SEEDS:
    r = RES.get(seed)
    if not r:
        continue
    g = r["gate"]
    print(f"[S34-自检] seed={seed}: 门A 冻结卡 max|Δθ| 输入卡={g['输入卡']['max_delta']:.1e} "
          f"输出卡={g['输出卡']['max_delta']:.1e}（须精确 0）| 目标卡 max|Δθ|="
          f"{g['目标卡']['max_delta']:.3e}（须 >0）| P0 逐位={r['e2e3000_bitwise']:.1e} | "
          f"PA 逐位={r['anchor_bitwise']:.1e} | P0 锚点Δ={r['e2e3000_anchor_dev']:+.2f}pp | "
          f"PA 锚点Δ={r['frozen_anchor_dev']:+.2f}pp | PB 锚点Δ={r['e2e6000_anchor_dev']:+.2f}pp",
          flush=True)

meta = dict(device=device, gpu=torch.cuda.get_device_name(0), D=D, FF=FF, NHEAD=NHEAD, lr=LR,
            batch=BATCH, maxlen=MAXLEN, base_steps=BASE_STEPS, frozen_steps=FROZEN_STEPS,
            total_steps=BASE_STEPS + FROZEN_STEPS, test_sha16=TEST_SHA, leak_ab=LEAK_AB,
            em_batch=EM_BATCH, chk_batch=CHK_BATCH, seeds=list(SEEDS),
            wall_total_min=(time.time() - t_start) / 60)
with open("/tmp/34_summary.json", "w") as f:
    json.dump(meta, f, ensure_ascii=False, indent=1)
print(f"[META] {meta}", flush=True)
print(f"[META] 总墙钟={(time.time()-t_start)/60:.1f}min", flush=True)
print("[DONE] exit=0", flush=True)
