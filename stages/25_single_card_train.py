#!/usr/bin/env python3
"""S25 · 单卡能不能【独立训练】、能不能【搬】？—— 五臂（M2 的 L1/L2 + RB4 的判据 c/d）★GPU。

判据全部已入库（只引用，不改）：
  M2-L1（speculation/SINGLE_CARD_TRAINING.md §2）：冻输入/输出卡只训目标卡 ⇒ 目标卡 EM ≥ 随机初始化
        且与端到端差 ≤ 5pp；
  M2-L2：交替训练 ⇒ ≥ 端到端 − 2pp；
  RB4-a（experiments/lit_survey/module_composition.md §2）：配对 Δ±SE；|Δ|≤1SE 且 2 seed 同号 = 过；
        Δ < −3pp 且 SE 不覆盖 0 = 不过；
  RB4-c ★：从同一联合 ckpt 分叉应可组合；随机初值各训应失败；
  RB4-d：拼装后【不再训任何参数】仍达标才算真组合；
  I3（§7）：冻结输入/输出卡时中间卡 ≥ 端到端 − 2pp。

臂（每臂 × 2 seed = 1234/5678，除额外对照 3b）：
  1 e2e      端到端联合训（= S19-res add_3d，必须逐位复现 37.25/31.13）
  2 anchor   从臂 1 ckpt 分叉 ⇒ 冻结输入/输出卡，只训目标卡
  3 randtgt  ★目标卡随机初始化，输入/输出卡同样冻结（= 锚点接口 + 随机目标卡，单变量对照）
  3b randall 额外对照：输入/输出卡也随机冻结（= RB4-c "随机初值各训" 的字面口径）
  4 alt      交替训练：冻输出卡训输入卡 → 冻输入卡训输出卡 ×2 轮，目标卡全程可训

门 A（核心自检，训练中逐步断言）：目标卡 grad is not None 且 max|Δθ| > 0；
  被冻结卡 grad is None 且 max|Δθ| == 0（精确）；打印参数量分组。

卡分组（= ABI 的工程实现）：输入卡 = emb + in_enc（token→模因），输出卡 = head（模因→logits），
  目标卡 = thought；外层 h = h + Think(h)（S19-res 逐字相同）。

其余与 S19-res 逐字相同：d=128 ff=512 NHEAD=4 lr=1e-3 batch=32 MAXLEN=512 prompt 优先，
3000 步，数据只用 add_3d（train4000/test800，同生成器同种子），EM/生成主口径 batch=1（R28），
自检 batch=16；任何干预报 Δ 非零比例（R29）。

ckpt：logs/25_ckpt_<arm>_seed<seed>.pt；结果增量落盘 logs/25_results.jsonl。
只允许写：本文件、logs/、/tmp；既有 stages/*.py / speculation/ / experiments/ 只读。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

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

# ---------------- 配置（与 S19-res 逐字一致） ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512
MIN_T = 48
TAIL_KEEP = 24
STEPS = 3000
LR = 1e-3
EPS = 1e-3
BATCH = 32
GEN_BATCH = 32
EM_BATCH = 1          # ★主口径：batch=1（R28）
CHK_BATCH = 16        # 自检口径
N_TRAIN = 4000
N_TEST = 800
MAX_GEN = 48

SMOKE = os.environ.get("S25_SMOKE") == "1"
if SMOKE:
    STEPS, N_TRAIN, N_TEST = 30, 300, 60
# 仅用于自检的临时覆盖（默认不设 ⇒ 与规格逐字一致）
if os.environ.get("S25_STEPS"):
    STEPS = int(os.environ["S25_STEPS"])

BUCKET = "add_3d"
BI = 2                # add_3d 在 S19 BUCKETS 里的下标 ⇒ 同生成器种子（14000+2*7 / 14900+2*7）
TOK = NS_ROOT + "/data/chinese/char_tokenizer.json"
LOGDIR = "/home/vesita/coding/my/flowme/logs"
CKPT_DIR = "/tmp" if SMOKE else LOGDIR      # smoke 不污染 logs/
CKPT_TMPL = CKPT_DIR + "/25_ckpt_{arm}_seed{seed}.pt"
RESULT_JSONL = ("/tmp/25_results_smoke.jsonl" if SMOKE else LOGDIR + "/25_results.jsonl")
if os.environ.get("S25_JSONL"):
    RESULT_JSONL = os.environ["S25_JSONL"]
SEEDS = (1234, 5678)
_only_env = [x for x in os.environ.get("S25_ONLY", "").split(",") if x]
ARM_ORDER = tuple(_only_env) if _only_env else ("e2e", "anchor", "randtgt", "alt", "randall")
if os.environ.get("S25_SEEDS"):
    SEEDS = tuple(int(x) for x in os.environ["S25_SEEDS"].split(","))

# 臂 1 必须复现的入库锚点（logs/15_iter_always.log / logs/19_results.jsonl）
E2E_ANCHOR = {1234: 0.3725, 5678: 0.3113}
E2E_ANCHOR_ACC = {1234: 0.5259, 5678: 0.4963}

ARM_LABEL = {
    "e2e": "臂1 端到端",
    "anchor": "臂2 共享锚点+冻结",
    "randtgt": "臂3 随机初值+冻结",
    "randall": "臂3b 全随机+冻结",
    "alt": "臂4 交替训练",
}
ARM_CKPT = {"e2e": "e2e", "anchor": "anchor", "randtgt": "randtgt",
            "randall": "randall", "alt": "alt"}
ALL_GROUPS = ("输入卡", "目标卡", "输出卡")
GROUP_ORDER = ("输入卡", "目标卡", "输出卡")

t_start = time.time()
print(f"[S25] ★单卡独立训练四臂（+3b 对照）：桶={BUCKET} 2 seed={SEEDS} steps={STEPS} "
      f"| d={D} ff={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN}(prompt优先) "
      f"EPS={EPS} 切分=train{N_TRAIN}/test{N_TEST} 生成主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} "
      f"| smoke={SMOKE}", flush=True)

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


_unk = tok.token_to_id("<unk>")
_arrow_ids = enc("→")
print(f"[TOK] WordLevel V={V} PAD={PAD_ID} EOS={EOS_ID} enc('<eos>')={enc('<eos>')} | "
      f"分隔符 '→' 词表外 ⇒ 单 token <unk>(id={_unk})，实测 enc('→')={_arrow_ids}", flush=True)


# ============================================================================
# ① 数据：与 S14/S19 逐字相同（同种子 ⇒ 同 train/test；只用 add_3d）
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
N_TPL_CN = sum(1 for _, c in TEMPLATES if c)
RENDER_VARIANTS = N_TPL_CN * 2 + (len(TEMPLATES) - N_TPL_CN)

CN_OPS = {"add_3d": ("加", "加上")}
EN_OPS = {"add_3d": ("plus",)}
LOHI = {"add_3d": (100, 999)}


def answer_values(name: str) -> list[int]:
    lo, hi = LOHI[name]
    return list(range(2 * lo, 2 * hi + 1))


def split_by_answer(name: str, y: int, rng) -> tuple[int, int]:
    lo, hi = LOHI[name]
    a = rng.randint(max(lo, y - hi), min(hi, y - lo))
    return a, y - a


def carry_rate(name: str, n: int, rng) -> tuple[float, float]:
    lo, hi = LOHI[name]
    any_c = unit_c = 0
    for _ in range(n):
        y = rng.randint(2 * lo, 2 * hi)
        a, b = split_by_answer(name, y, rng)
        w = max(len(str(a)), len(str(b)))
        sa, sb = str(a).zfill(w), str(b).zfill(w)
        c = 0
        carry_pos = []
        for i in range(w - 1, -1, -1):
            s = int(sa[i]) + int(sb[i]) + c
            if s >= 10:
                carry_pos.append(w - 1 - i)
            c = s // 10
        if carry_pos:
            any_c += 1
        if 0 in carry_pos:
            unit_c += 1
    return any_c / n, unit_c / n


class Rng:
    """可复现的桶内随机源（避免占用 torch 全局种子）。"""

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


def render(name: str, a: int, b: int, y: int, rng: Rng) -> tuple[str, str, str]:
    tpl, cjk = TEMPLATES[rng.randrange(len(TEMPLATES))]
    op = (CN_OPS[name] if cjk else EN_OPS[name])
    op = op[rng.randrange(len(op))]
    tight = cjk and rng.random() < 0.5
    expr = f"{a}{op}{b}" if tight else f"{a} {op} {b}"
    nl = tpl.format(expr=expr)
    return f"题干：{nl} → ", f"#### {y}", nl


def gen_half(name: str, n: int, rng: Rng, used: set[str], tag: str) -> list[dict]:
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
                y=it["y"], nl=it["nl"])


_tr_rng, _te_rng = Rng(14000 + BI * 7), Rng(14900 + BI * 7)
_used: set[str] = set()
raw_tr = gen_half(BUCKET, N_TRAIN, _tr_rng, _used, "train")
raw_te = gen_half(BUCKET, N_TEST, _te_rng, _used, "test")
TRAIN = [encode_record(x) for x in raw_tr]
TEST = [encode_record(x) for x in raw_te]
assert len(_used) == len(raw_tr) + len(raw_te), "train/test 文本池计数异常（应零重叠）"

_ys = answer_values(BUCKET)
_ctr_tr = Counter(r["y"] for r in TRAIN)
_ctr_te = Counter(r["y"] for r in TEST)
_top5 = [v for v, _ in Counter(r["gold"] for r in TRAIN).most_common(5)]
_prior = sum(1 for r in TEST if r["gold"] in _top5) / max(1, len(TEST))
_cut = sum(1 for r in TRAIN + TEST if r["cut_p"] or r["cut_t"])
_anyc, _unitc = carry_rate(BUCKET, 300, Rng(77 + BI))
_tpl_used = len({r["nl"] for r in TRAIN + TEST})
_tgt_len = sorted(len(r["t_full"]) for r in TRAIN)
print(f"[DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)} 文本零重叠 | "
      f"答案域={min(_ys)}..{max(_ys)}({len(_ys)}个) 计数train min/max="
      f"{min(_ctr_tr.values())}/{max(_ctr_tr.values())} | 模板覆盖={_tpl_used}/{len(TEMPLATES)}句"
      f"(渲染{RENDER_VARIANTS}种) | 截断={_cut} | target中位={_tgt_len[len(_tgt_len)//2]}tok | "
      f"进位率(至少一次/个位)={_anyc:.2f}/{_unitc:.2f} | ★最高频5答案={_top5} "
      f"在test的机会水平={100.0*_prior:.1f}%", flush=True)


# ============================================================================
# ② 模型：与 S19-res 逐字相同（h = h + Think(h)）
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
    """与 S19-res 逐字相同：h = h + Think(h)（k=1，无循环）。"""

    def __init__(self, d: int, ff: int):
        super().__init__()
        self.d = d
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
        assert n <= pe.size(0), f"PE 长度 {pe.size(0)} < 序列长 {n}"
        x = self.emb(ids) + pe[:n].unsqueeze(0)
        h = self.in_enc(x, src_mask=m)
        h = h + self.thought(h, src_mask=m)          # ★与 S19-res 逐字相同
        return self.head(h)


def group_of(name: str) -> str:
    if name.startswith("emb.") or name.startswith("in_enc."):
        return "输入卡"
    if name.startswith("thought."):
        return "目标卡"
    if name.startswith("head."):
        return "输出卡"
    raise KeyError(f"未知参数分组：{name}")


def set_trainable(model: Cards, groups) -> None:
    for n, p in model.named_parameters():
        p.requires_grad_(group_of(n) in groups)


def group_counts(model: Cards) -> dict:
    c = defaultdict(int)
    for n, p in model.named_parameters():
        c[group_of(n)] += p.numel()
    return c


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


# ============================================================================
# ③ 门 A：训练中的梯度 / 更新断言
# ============================================================================
def snapshot(model: Cards) -> dict:
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def check_grads(model: Cards, trainable, where: str) -> None:
    """门 A 逐步断言：可训组每个参数都得有梯度；冻结组每个参数 grad 必须为 None。"""
    for n, p in model.named_parameters():
        g = group_of(n)
        if g in trainable:
            if p.grad is None:
                raise SystemExit(f"[门A-FAIL] {where}: 可训组 {g} 的参数 {n} grad is None")
        else:
            if p.grad is not None:
                raise SystemExit(f"[门A-FAIL] {where}: 冻结组 {g} 的参数 {n} grad 非 None")


def arm_model(arm: str, seed: int) -> Cards:
    """建模型：臂 1 逐字复现 S19-res 的初始化顺序（manual_seed → Cards）。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if arm == "randall":
        return Cards(D, FF).to(device)          # 全随机初值（输入/输出/目标卡都随机）
    model = Cards(D, FF)
    if arm != "e2e":
        ck = CKPT_TMPL.format(arm="e2e", seed=seed)
        if not os.path.exists(ck):
            raise SystemExit(f"[臂-前置] 缺臂 1 ckpt {ck} ⇒ 不能分叉")
        model.load_state_dict(torch.load(ck, map_location="cpu"))
        if arm == "randtgt":
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            model.thought = enc_layer(D, FF)    # ★目标卡随机初始化（输入/输出卡保持锚点值）
    return model.to(device)


def phases_for(arm: str):
    if arm == "e2e":
        return [("联合训练(全参)", STEPS, set(ALL_GROUPS))]
    if arm == "anchor":
        return [("冻输入/输出 → 训目标卡", STEPS, {"目标卡"})]
    if arm == "randtgt":
        return [("冻输入/输出(锚点) → 训随机目标卡", STEPS, {"目标卡"})]
    if arm == "randall":
        return [("冻输入/输出(随机) → 训随机目标卡", STEPS, {"目标卡"})]
    if arm == "alt":
        h = STEPS // 4
        sizes = [h, h, h, STEPS - 3 * h]
        return [("R1 冻输出 → 训输入+目标", sizes[0], {"输入卡", "目标卡"}),
                ("R1 冻输入 → 训输出+目标", sizes[1], {"输出卡", "目标卡"}),
                ("R2 冻输出 → 训输入+目标", sizes[2], {"输入卡", "目标卡"}),
                ("R2 冻输入 → 训输出+目标", sizes[3], {"输出卡", "目标卡"})]
    raise KeyError(arm)


def run_arm(arm: str, seed: int) -> dict:
    model = arm_model(arm, seed)
    cnt = group_counts(model)
    print(f"[门A] {ARM_LABEL[arm]} seed={seed} 参数量分组: "
          + " | ".join(f"{g}={cnt[g]:,}" for g in GROUP_ORDER)
          + f" | 合计={sum(cnt.values()):,} (输入卡=V*d+编码层, 目标卡=编码层, 输出卡=d*V)", flush=True)

    phases = phases_for(arm)
    order = torch.randperm(len(TRAIN), generator=torch.Generator().manual_seed(seed))
    pos = 0
    wall0 = time.time()
    gate_log = []
    step_global = 0
    for pname, psteps, groups in phases:
        if psteps <= 0:
            continue
        set_trainable(model, groups)
        params = [p for p in model.parameters() if p.requires_grad]
        opt = AdamW(params, lr=LR)
        for p in model.parameters():
            p.grad = None
        before = snapshot(model)
        model.train()
        t0, last = time.time(), 0.0
        for k in range(1, psteps + 1):
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
            check_grads(model, groups, f"{arm} seed={seed} {pname} step={k}")   # ★门 A 逐步
            opt.step()
            step_global += 1
            if k % 500 == 0 or k == psteps:
                el = time.time() - t0
                print(f"  [train] {arm} seed={seed} {pname} step={k}/{psteps} "
                      f"loss={loss.item():.4f} elapsed={el:.0f}s "
                      f"s/step={(el - last) / (k % 500 or 500):.3f}", flush=True)
                last = el
        # 阶段末：逐位比对（冻结组必须精确 0；可训组必须 > 0）
        rep = {}
        for g in GROUP_ORDER:
            names = [n for n, _ in model.named_parameters() if group_of(n) == g]
            mx = 0.0
            for n, p in model.named_parameters():
                if group_of(n) != g:
                    continue
                mx = max(mx, (before[n] - p.detach()).abs().max().item())
            n_grad = sum(1 for n, p in model.named_parameters()
                         if group_of(n) == g and p.grad is not None)
            rep[g] = dict(max_delta=mx, n_param=len(names), n_grad=n_grad)
        line = []
        for g in GROUP_ORDER:
            r = rep[g]
            if g in groups:
                assert r["n_grad"] > 0, f"门A失败：可训组 {g} 无梯度"
                assert r["max_delta"] > 0.0, f"门A失败：可训组 {g} 未更新（max|Δθ|==0）"
                line.append(f"{g}: max|Δθ|={r['max_delta']:.3e}(>0 ✓) grad≠None({r['n_grad']}个) ✓")
            else:
                assert r["max_delta"] == 0.0, \
                    f"门A失败：冻结组 {g} 被改动 max|Δθ|={r['max_delta']:.3e}"
                assert r["n_grad"] == 0, f"门A失败：冻结组 {g} 有梯度（{r['n_grad']}个）"
                line.append(f"{g}: max|Δθ|={r['max_delta']:.3e}(==0 ✓) grad=None ✓")
        print(f"[门A] {ARM_LABEL[arm]} seed={seed} 阶段[{pname}] 步数={psteps} ⇒ " + " | ".join(line),
              flush=True)
        gate_log.append(dict(phase=pname, steps=psteps, trainable=sorted(groups),
                             groups={g: rep[g] for g in GROUP_ORDER}))
    wall_train = time.time() - wall0

    return dict(model=model, wall_train=wall_train, gate=gate_log,
                n_params=sum(cnt.values()), counts={g: cnt[g] for g in GROUP_ORDER},
                steps=step_global)


# ============================================================================
# ④ 评估（与 S19 逐字相同：R28 主口径 batch=1）
# ============================================================================
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


NUM_CHARS = set("0123456789+-*/=%$")
TMPL_CHARS = set("【】详细解题思路推理与") | {"<", ">", "#"}
CAT_NAMES = ("①数字与算子", "②模板/格式串", "③中文字符", "④其它")
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


@torch.no_grad()
def ce_categories(model: Cards, recs):
    model.eval()
    tok_ce: list[list[float]] = [[] for _ in range(4)]
    samp_all: list[float] = []
    dig_samp: list[list[float]] = []
    for i in range(0, len(recs), GEN_BATCH):
        chunk = recs[i: i + GEN_BATCH]
        ids, s = build_batch(chunk)
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        for j, r in enumerate(chunk):
            e = s[j] + len(r["t"])
            lp = logp[j, s[j] - 1: e - 1]
            tgt = ids[j, s[j]: e]
            ces = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1).tolist()
            dsamp = []
            for pos, v in enumerate(ces):
                kk = classify(int(tgt[pos]))
                tok_ce[kk].append(v)
                if kk == 0:
                    dsamp.append(v)
            if dsamp:
                dig_samp.append(dsamp)
            samp_all.append(sum(ces) / len(ces))
    model.train()
    return tok_ce, samp_all, dig_samp


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n) if n else float("nan")


def report_ce(tag: str, model: Cards, recs) -> dict:
    tok_ce, samp_all, dig_samp = ce_categories(model, recs)
    ntok = sum(len(v) for v in tok_ce)
    ce_m, ce_se = mean_se(samp_all)
    dig_ce = dig_n = None
    for kk in range(4):
        xs = tok_ce[kk]
        if not xs:
            print(f"[CE] {tag} {CAT_NAMES[kk]}: n=0", flush=True)
            continue
        km = sum(xs) / len(xs)
        print(f"[CE] {tag} {CAT_NAMES[kk]}: n={len(xs)} 占比={100.0*len(xs)/ntok:.1f}% "
              f"CE={km:.4f} nats", flush=True)
        if kk == 0:
            dig_ce, dig_n = km, len(xs)
    out = dict(ce_m=ce_m, ce_se=ce_se, dig_ce=dig_ce, dig_n=dig_n)
    print(f"[CE] {tag} ★整体配对CE(样本级)={ce_m:.4f}±{ce_se:.4f} n={len(samp_all)}", flush=True)
    if dig_ce is not None:
        per = [sum(x) / len(x) for x in dig_samp]
        dm, dse = mean_se(per)
        acc = math.exp(-dm)
        acc_se = acc * dse
        print(f"[CE] {tag} ★数字CE(token加权)={dig_ce:.4f} n={dig_n} | "
              f"★数字CE(样本级)={dm:.4f}±{dse:.4f} n={len(per)} ⇒ "
              f"★数字每步正确率={acc*100:.2f}%±{acc_se*100:.2f}pp", flush=True)
        out.update(dig_samp_m=dm, dig_samp_se=dse, acc=acc, acc_se=acc_se)
    return out


@torch.no_grad()
def greedy_gen(model: Cards, recs, batch: int = 1) -> list[str]:
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


def hits_from(texts: list[str], recs) -> list[dict]:
    hits = []
    for i, r in enumerate(recs):
        if r["gold"] is None:
            continue
        pred = parse_ans(texts[i])
        hits.append(dict(idx=i, gold=r["gold"], pred=pred, gen=texts[i],
                         strict=int(pred == r["gold"]),
                         bucket=("B" if r["cut_t"] else "A"),
                         free=(not r["cut_t"] and not r["cut_p"])))
    return hits


def r28_selfcheck(model: Cards, recs, tag: str, k: int = CHK_BATCH) -> dict:
    sub = recs[:k]
    t1 = greedy_gen(model, sub, batch=1)
    tN = greedy_gen(model, sub, batch=CHK_BATCH)
    same = sum(a == b for a, b in zip(t1, tN))
    print(f"[R28] {tag}批内一致性 K={len(sub)}: batch=1 vs batch={CHK_BATCH} 逐字一致 "
          f"{same}/{len(sub)} ⇒ "
          f"{'一致（batch=1 口径自洽）' if same == len(sub) else '★不一致 ⇒ 仅 batch=1 口径可用（本单元 EM 全部走 batch=1）'}",
          flush=True)
    shown = 0
    for i, (a, b) in enumerate(zip(t1, tN)):
        if a == b:
            continue
        d = next((j for j in range(min(len(a), len(b))) if a[j] != b[j]), min(len(a), len(b)))
        print(f"[R28] {tag}差异[{i}] 首分歧位={d} | batch1={a[:50]!r} | batchN={b[:50]!r}",
              flush=True)
        shown += 1
        if shown >= 3:
            break
    return dict(same=same, k=len(sub))


def em_report(tag: str, hits: list[dict]) -> dict:
    out = {}
    for name, sel in (("桶A(目标未截)", [h for h in hits if h["bucket"] == "A"]),
                      ("桶A1(都未截)", [h for h in hits if h["free"]]),
                      ("全量", list(hits))):
        key = name.split("(")[0]
        n = len(sel)
        s = sum(h["strict"] for h in sel) / n if n else float("nan")
        se = math.sqrt(s * (1 - s) / n) if n else float("nan")
        out[key] = dict(n=n, em=s, se=se)
        print(f"[EM] {tag} {name}: n={n} 严格EM={s*100:.2f}%±{se*100:.2f}", flush=True)
    return out


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def evaluate(model: Cards, tag: str) -> dict:
    cm, cs = mean_se(eval_ce(model, TEST))
    txt = greedy_gen(model, TEST, batch=EM_BATCH)          # ★R28：主口径 batch=1
    hits = hits_from(txt, TEST)
    rep = report_ce(tag, model, TEST)
    em = em_report(f"{tag} (batch={EM_BATCH})", hits)
    r28 = r28_selfcheck(model, TEST, tag=f"[{tag}] ")
    return dict(ce=cm, ce_se=cs, acc=rep.get("acc"), acc_se=rep.get("acc_se"),
                dsamp=rep.get("dig_samp_m"), dsamp_se=rep.get("dig_samp_se"),
                dig_tok=rep.get("dig_ce"),
                em=em["全量"]["em"], em_se=em["全量"]["se"], em_n=em["全量"]["n"],
                em_a1=em["桶A1"]["em"], r28_same=r28["same"], r28_k=r28["k"],
                strict=[h["strict"] for h in hits])


def paired(xa, xb) -> dict:
    n = len(xa)
    xs = [a - b for a, b in zip(xa, xb)]
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    nz = sum(1 for x in xs if x != 0) / n
    return dict(n=n, d=m, se=se, ratio=(m / se if se > 0 else float("inf")), nz=nz,
                cover0=(abs(m) <= se))


def append_jsonl(obj: dict) -> None:
    with open(RESULT_JSONL, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# ============================================================================
# ⑤ 主循环：按 seed 逐臂（臂 1 先跑 ⇒ 分叉有源）
# ============================================================================
JOBS = [(arm, s) for s in SEEDS for arm in ARM_ORDER]
RESULTS: dict[tuple[str, int], dict] = {}
ANCHOR_OK = True

for arm, seed in JOBS:
    t0 = time.time()
    print(f"\n[S25] ===== {ARM_LABEL[arm]} arm={arm} seed={seed} steps={STEPS} "
          f"train={len(TRAIN)} test={len(TEST)} =====", flush=True)
    out = run_arm(arm, seed)
    model = out["model"]
    ev = evaluate(model, f"{BUCKET} {arm} seed={seed}")
    ck = CKPT_TMPL.format(arm=ARM_CKPT[arm], seed=seed)
    torch.save(model.state_dict(), ck)
    dig = sha16(ck)
    print(f"[CKPT] saved {ck} sha256[:16]={dig}", flush=True)
    total = time.time() - t0

    r = dict(arm=arm, seed=seed, steps=out["steps"], wall_train=out["wall_train"],
             wall_total=total, n_params=out["n_params"], counts=out["counts"],
             gate=out["gate"], ckpt=ck, sha=dig, **{k: v for k, v in ev.items()})
    RESULTS[(arm, seed)] = r
    append_jsonl({k: v for k, v in r.items() if k != "strict"})
    print(f"[MAIN-TABLE] {arm} seed={seed} | EM全量={r['em']*100:.2f}%±{r['em_se']*100:.2f}"
          f"(n={r['em_n']}) | 数字每步正确率={r['acc']*100:.2f}%±{r['acc_se']*100:.2f}pp | "
          f"整体CE={r['ce']:.4f}±{r['ce_se']:.4f} | R28={r['r28_same']}/{r['r28_k']} | "
          f"训练={out['wall_train']/60:.1f}min 合计={total/60:.1f}min | "
          f"ckpt={os.path.basename(ck)} sha16={dig}", flush=True)

    if arm == "e2e" and not SMOKE:
        ref = E2E_ANCHOR[seed]
        dev = (r["em"] - ref) * 100
        ok = abs(dev) <= 1.0
        ANCHOR_OK &= ok
        print(f"[锚点] 臂1 seed={seed}: 入库={ref*100:.2f}% vs 本次={r['em']*100:.2f}% "
              f"Δ={dev:+.2f}pp ⇒ {'一致(|Δ|≤1pp)' if ok else '★偏离 >1pp（点名并停）'}", flush=True)
        if not ok:
            print(f"[STOP] ★臂 1 未复现入库锚点 37.25/31.13（seed={seed} Δ={dev:+.2f}pp）"
                  f"⇒ 按判据 1 停，不跑下游臂。", flush=True)
            print("[DONE] exit=4", flush=True)
            sys.exit(4)
    del model
    torch.cuda.empty_cache()

# ---------------- 判据 d：臂 2 训好后，再冻结【全部卡】不再训任何参数 → 评估（拼装后不训） ----
print("\n[S25] ===== ★判据 d：臂 2 拼装后【不再训任何参数】评估（所有卡 requires_grad=False） =====",
      flush=True)
D_EM: dict[int, dict] = {}
if all(("anchor", s) in RESULTS for s in SEEDS):
    for s in SEEDS:
        ck = RESULTS[("anchor", s)]["ckpt"]
        torch.manual_seed(777)
        m = Cards(D, FF)
        m.load_state_dict(torch.load(ck, map_location="cpu"))
        m = m.to(device)
        for p in m.parameters():
            p.requires_grad_(False)
        assert not any(p.requires_grad for p in m.parameters()), "拼装后仍有可训参数"
        ev = evaluate(m, f"[判据d] {BUCKET} anchor seed={s} 全冻结")
        same = sum(a == b for a, b in zip(ev["strict"], RESULTS[("anchor", s)]["strict"]))
        D_EM[s] = dict(em=ev["em"], acc=ev["acc"], strict=ev["strict"], same=same,
                       em_se=ev["em_se"])
        print(f"[判据d] seed={s}: 拼装后不训 EM={ev['em']*100:.2f}%±{ev['em_se']*100:.2f} "
              f"数字每步={ev['acc']*100:.2f}% | 与臂 2 训练末逐样本一致 {same}/{len(ev['strict'])} "
              f"| 训练中 EM={RESULTS[('anchor', s)]['em']*100:.2f}%", flush=True)
        del m
        torch.cuda.empty_cache()
else:
    print("[判据d] ★臂 2 缺失 ⇒ 无法评估", flush=True)

# ============================================================================
# ⑥ 汇总表 + 配对 Δ + ★逐条判定
# ============================================================================
print("\n[S25] ===== ★四臂（+3b）主表 × 2 seed =====", flush=True)
print("[S25] 臂 | seed | EM全量(n)±SE | 数字每步±SE | 整体CE±SE | R28 | sha16", flush=True)
for arm in ("e2e", "anchor", "randtgt", "alt", "randall"):
    for s in SEEDS:
        r = RESULTS.get((arm, s))
        if r is None:
            print(f"[S25] {arm} seed={s}: ★未完成", flush=True)
            continue
        print(f"[S25] {ARM_LABEL[arm]} | {s} | {r['em']*100:.2f}%±{r['em_se']*100:.2f}"
              f"(n={r['em_n']}) | {r['acc']*100:.2f}%±{r['acc_se']*100:.2f}pp | "
              f"{r['ce']:.4f}±{r['ce_se']:.4f} | {r['r28_same']}/{r['r28_k']} | {r['sha']}",
              flush=True)

PAIRS = (("anchor", "e2e", "臂2 − 臂1（L1：冻接口单卡训 vs 端到端）"),
         ("anchor", "randtgt", "臂2 − 臂3（★判据 c：共享锚点 vs 随机目标卡）"),
         ("anchor", "randall", "臂2 − 臂3b（★判据 c 字面口径：共享锚点 vs 全随机）"),
         ("alt", "e2e", "臂4 − 臂1（L2：交替训练 vs 端到端）"))
print("\n[S25] ===== ★配对 Δ±SE（逐 test 样本配对；R29 报 Δ 非零比例） =====", flush=True)
print("[S25] 配对 | seed | Δ±SE(pp) | Δ/SE | Δ非零比 | 1SE 覆盖 0?", flush=True)
PD: dict[tuple[str, str, int], dict] = {}
for a, b, desc in PAIRS:
    for s in SEEDS:
        ra, rb = RESULTS.get((a, s)), RESULTS.get((b, s))
        if ra is None or rb is None:
            print(f"[S25] {desc} seed={s}: ★缺 run ⇒ 无法配对", flush=True)
            continue
        d = paired(ra["strict"], rb["strict"])
        PD[(a, b, s)] = d
        print(f"[S25] {desc} | {s} | {d['d']*100:+.2f}±{d['se']*100:.2f}pp | {d['ratio']:.2f} | "
              f"{d['nz']*100:.1f}% | {'是(≤1SE)' if d['cover0'] else '否(>1SE)'}", flush=True)


def two_seed_same_sign(ps):
    ds = [PD.get(p) for p in ps]
    if any(d is None for d in ds):
        return None
    signs = [1 if d["d"] > 0 else (-1 if d["d"] < 0 else 0) for d in ds]
    return signs


print("\n[S25] ===== ★逐条判定（判据写死，不改） =====", flush=True)

# 判据 1：臂 1 复现
print(f"[判定-1] 臂 1 复现 37.25/31.13："
      + " | ".join(f"seed{s}={RESULTS[('e2e', s)]['em']*100:.2f}%(Δ"
                   f"{(RESULTS[('e2e', s)]['em']-E2E_ANCHOR[s])*100:+.2f}pp)"
                   for s in SEEDS if ("e2e", s) in RESULTS)
      + f" ⇒ {'✓ 过（|Δ|≤1pp）' if ANCHOR_OK else '★不过'}", flush=True)

# 判据 2：L1（臂 2 vs 臂 1）—— |Δ|≤1SE 且 2 seed 同号 = 过；Δ<−3pp 且 SE 不覆盖 0 = 不过
d21 = [PD.get(("anchor", "e2e", s)) for s in SEEDS]
if all(d is not None for d in d21):
    sg = two_seed_same_sign([("anchor", "e2e", s) for s in SEEDS])
    same_sign = len(set(sg)) == 1
    within1 = all(abs(d["d"]) <= d["se"] for d in d21)
    fail = any(d["d"] < -0.03 and not d["cover0"] for d in d21)
    if fail:
        v1 = "★不过（Δ < −3pp 且 SE 不覆盖 0）"
    elif within1 and same_sign:
        v1 = "✓ 过（|Δ| ≤ 1SE 且 2 seed 同号）"
    else:
        v1 = "未过（未同时满足 |Δ|≤1SE 与 2 seed 同号，也未触发 <−3pp 不过线）"
    print(f"[判定-L1] 臂2 vs 臂1："
          + " | ".join(f"seed{s} Δ={d['d']*100:+.2f}±{d['se']*100:.2f}pp" for s, d in zip(SEEDS, d21))
          + f" | 同号={same_sign} | 都≤1SE={within1} ⇒ {v1}", flush=True)
else:
    print("[判定-L1] ★缺 run", flush=True)

# 判据 3 ★：判据 c（臂 2 应显著优于臂 3 / 3b）
for other, tag in (("randtgt", "臂3"), ("randall", "臂3b")):
    ds = [PD.get(("anchor", other, s)) for s in SEEDS]
    if all(d is not None for d in ds):
        sg = two_seed_same_sign([("anchor", other, s) for s in SEEDS])
        same_pos = len(set(sg)) == 1 and sg[0] == 1
        sig2 = all(d["d"] >= 2 * d["se"] for d in ds)
        verdict = ("★臂2 显著优于 " + tag) if (same_pos and sig2) else \
                  (f"臂2 与 {tag} 无显著差（未达 2SE 且 2 seed 同号正）")
        print(f"[判定-c] 臂2 vs {tag}："
              + " | ".join(f"seed{s} Δ={d['d']*100:+.2f}±{d['se']*100:.2f}pp" for s, d in zip(SEEDS, ds))
              + f" | 2seed同号正={same_pos} | 都Δ≥2SE={sig2} ⇒ {verdict}"
              + ("（★若臂3≈臂2 ⇒ 共享锚点不必要，如实报）" if not (same_pos and sig2) else ""),
              flush=True)
    else:
        print(f"[判定-c] 臂2 vs {tag}: ★缺 run", flush=True)

# 判据 4：L2（臂 4 ≥ 臂 1 − 2pp？）
if all(("alt", s) in RESULTS and ("e2e", s) in RESULTS for s in SEEDS):
    parts = []
    ok_all = True
    for s in SEEDS:
        gap = (RESULTS[("alt", s)]["em"] - RESULTS[("e2e", s)]["em"]) * 100
        parts.append(f"seed{s} Δ={gap:+.2f}pp({'✓' if gap >= -2.0 else '★'})")
        ok_all &= gap >= -2.0
    print(f"[判定-L2] 臂4 交替 vs 臂1 端到端（门槛 −2pp）：" + " | ".join(parts)
          + f" ⇒ {'✓ 过' if ok_all else '★不过'}", flush=True)
else:
    print("[判定-L2] ★缺 run", flush=True)

# 判据 5：d（拼装后不训）
if D_EM:
    parts = []
    ok_d = True
    for s in SEEDS:
        if s not in D_EM:
            continue
        gap = (D_EM[s]["em"] - RESULTS[("e2e", s)]["em"]) * 100
        ok_d &= gap >= -2.0
        parts.append(f"seed{s} EM={D_EM[s]['em']*100:.2f}% vs 端到端="
                     f"{RESULTS[('e2e', s)]['em']*100:.2f}%（Δ={gap:+.2f}pp，"
                     f"{'✓≥−2pp' if gap >= -2.0 else ('仅 ≥−5pp(L1线)' if gap >= -5.0 else '★<−5pp')}）")
    print("[判定-d] 拼装后不再训任何参数：" + " | ".join(parts)
          + (" ⇒ ✓ 达标（无需融合层）" if ok_d else " ⇒ ★未达 −2(I3) 线 ⇒ 需要融合层（判据 d 被证伪）"),
          flush=True)
else:
    print("[判定-d] ★臂 2 缺 run", flush=True)

# ---------------- 收尾元数据 ----------------
print("\n[META] device=%s %s | D=%d FF=%d NHEAD=%d lr=%g batch=%d steps=%d MAXLEN=%d EPS=%g "
      "| 生成主口径=batch%d 自检=batch%d | 臂=1 e2e / 2 anchor(冻输入+输出) / 3 randtgt / "
      "3b randall / 4 alt" % (device, torch.cuda.get_device_name(0), D, FF, NHEAD, LR, BATCH,
                              STEPS, MAXLEN, EPS, EM_BATCH, CHK_BATCH), flush=True)
for arm in ("e2e", "anchor", "randtgt", "alt", "randall"):
    for s in SEEDS:
        r = RESULTS.get((arm, s))
        if r is None:
            continue
        print(f"[META] {ARM_LABEL[arm]} seed={s}: steps={r['steps']} "
              f"训练墙钟={r['wall_train']/60:.1f}min 本run合计={r['wall_total']/60:.1f}min "
              f"ckpt={r['ckpt']} sha256[:16]={r['sha']} R28={r['r28_same']}/{r['r28_k']} "
              f"参数量={r['n_params']:,} 分组={r['counts']}", flush=True)
print(f"[META] 总墙钟={(time.time()-t_start)/60:.1f}min | run数={len(RESULTS)}/{len(JOBS)}", flush=True)
print("[DONE] exit=0", flush=True)
