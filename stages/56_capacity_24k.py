#!/usr/bin/env python3
"""S55 · ★把容量轴判死或救活：S54 的容量臂（目标卡 ff 512→1024）步数翻倍到 24000。

未决问题（S54，stages/54_capacity_target.py，提交 acebbf4）：
  S53 ff512：s1234/s5678 @12000 全量 93.62% / clean 97.27%（两 seed 完全相同，极差 0.00pp）。
  容量 ff1024 @12000：s1234 93.62/97.27（Δ +0.00pp）· s5678 62.88/65.32（Δ −30.75pp）
    ⇒ 2 seed 不同号；且 s5678 在 @12000 仍陡升（@6000 仅 27.62%，测试 CE 0.140 vs S53 0.048）
    ⇒ 饱和检查未过（9000→12000 均值 +2.06/+2.14pp ≥ 1pp）⇒ S54 只给了【条件性】结论。
  Lead 裁断：不能判「容量无戏」，只能判「12000 步不够」——cosine 能救 ff512（极差 0）
    却救不了 ff1024 ⇒ 容量使优化变难。
⇒ 本单元 = 把 ff1024 跑到 @24000，看它收敛后追平/超越 97.27%（容量有戏）还是仍不收敛（容量无戏）。

设计（★只加步数，结构逐字同 S54 的容量臂）：
  臂 = ff1024（目标卡 ff=1024，d=128/nhead=4/1 层思维卡残差/in_enc ff=512 全同 S54）。
  seed = 1234 / 5678 ⇒ 2 run；步数 24000（S54 的两倍）。
  调度 = warmup500 + cosine→0，总步数 24000 ⇒ cosine 周期随总步数拉长（@12000 处 lr 高于 S53）。
  数据/batch/MAXLEN/add_3d/R39/R40 与 S54/S53 逐字同。
  每 750 步内存快照（32 格）⇒ 同时给出 @12000/15000/18000/21000/24000 曲线；
  ckpt 落盘 @12000/15000/18000/21000/24000 = logs/56_ckpt_ff1024_s{steps}_seed{s}.pt。

★判据（写死，同 S54）：
  1 主判据：Δ = EM(ff1024@24000) − EM(S53 cosine ff512@12000)（S53 基线 93.62/97.27），
    逐 test 样本配对 ±SE，全量与 clean（n=770）都报，2 seed 同号才算数：
      Δ ≥ +2SE 同号 ⇒ 容量有增益；|Δ| ≤ 1SE 同号 ⇒ 追平；Δ ≤ −2SE 同号 ⇒ 有害。
  2 同 step 对照：ff1024@12000 vs S53@12000（复核 S54）⇒ 分离「容量效应」与「步数效应」；
    另报 ff1024@24000 vs ff1024@12000（纯步数效应）。
  3 ★饱和检查（R35，本单元核心）：@21000→@24000 增量 <1pp ⇒ 已饱和 ⇒ 结论可下；
    ≥1pp ⇒ 再标【条件性】并给「还需多少步」的线性外推（明确标注是外推）。
  4 两 seed 极差在 @12000/15000/18000/21000/24000 的变化 ⇒ 是否随训练收窄。
  5 CE 曲线：@3000…@24000 的训练 CE 与测试整体 CE，与 S53 对照 ⇒ 容量是否只是需要更长步数。

★口径：EM 一律 batch=1（R28）+ batch=1 vs 16 自检；R39 显式 eval()；R40 batch=1 前向 4 线程；
  地板 0.12%/0.25%；SEEN/UNSEEN 分层；TEST sha256[:16] == b34e7ea515203227（R29：无干预 ⇒ 不适用）。

复用：exec stages/19_residual_scan.py 到 "# ③ 主循环" 之前（同 tokenizer、同 add_3d
  train4000/test800、同 build_batch/greedy_gen(EM_BATCH=1)/parse_ans/r28_selfcheck）。
对照曲线：读 logs/53_results.jsonl（既有日志，只读）取 S53 臂的逐格 EM/CE/训练 loss。
只允许写：本文件、logs/s55_cap24k.log、logs/56_*、/tmp；既有 stages/*.py 只读。
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

SMOKE = os.environ.get("S56_SMOKE") == "1"

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
# ★坑：S19 头里也有一个同名 SMOKE（读 S19_SMOKE），上面那次 globals() 拷贝会把它带进来
#   并覆盖本单元的 S56_SMOKE ⇒ 必须在拷贝之后重新取一次。
SMOKE = os.environ.get("S56_SMOKE") == "1"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402

torch.set_num_threads(4)                # ★R40：batch=1 逐样本前向统一 4 线程

T0 = time.time()
BUCKET = "add_3d"
SEEDS56 = (1234, 5678)
FF_IN = FF                              # 输入卡 ff = 512（不动）
FF_TGT = FF * 2                         # ★唯一结构变量：目标卡 ff 512 → 1024（与 S54 逐字同）
STEPS56 = 90 if SMOKE else 24000        # ★本单元的唯一变化：12000 → 24000
GRID = tuple(range(30, 91, 30)) if SMOKE else tuple(range(750, 24001, 750))
WARMUP = 5 if SMOKE else 500
BASE_LR = LR                            # 1e-3（与 S53/S54 同一个 base lr）
SAVE_ATS = (30, 60, 90) if SMOKE else (12000, 15000, 18000, 21000, 24000)
N_EVAL = 60 if SMOKE else N_TEST        # 800
CKPT_TMPL = (f"/tmp/56_ckpt_ff1024_s{{steps}}_seed{{seed}}.pt" if SMOKE
             else f"{ROOT}/logs/56_ckpt_ff1024_s{{steps}}_seed{{seed}}.pt")
BASE_CKPT = f"{ROOT}/logs/53_ckpt_lrcosine_s{{steps}}_seed{{seed}}.pt"
BASE_JSONL = f"{ROOT}/logs/53_results.jsonl"
JSONL = "/tmp/56_results_smoke.jsonl" if SMOKE else f"{ROOT}/logs/56_results.jsonl"
TEST_SHA_EXPECT = "b34e7ea515203227"
LEAK_EXPECT = 16
# S53 入库锚点（logs/s53_lrcos.log: 2 seed 相同）—— 重载 ckpt 必须复现，差 ≤0.2pp
ANCHOR_S53 = {1234: dict(full=0.9362, clean=0.9727), 5678: dict(full=0.9362, clean=0.9727)}
# S54 入库的容量臂 @12000（logs/s54_cap.log）—— 同 step 对照的复核基准
S54_AT_12K = {1234: dict(full=0.9362, clean=0.9727), 5678: dict(full=0.6288, clean=0.6532)}
REPORT_STEPS = tuple(st for st in (12000, 15000, 18000, 21000, 24000) if st <= STEPS56)


def lr_at(t: int, steps: int, warmup: int, base: float) -> float:
    """与 S53/S54 逐字相同：warmup 线性 0→base，随后 cosine 衰减到 0（t = 0-based）。"""
    if t < warmup:
        return base * (t / warmup)
    prog = (t - warmup) / max(1, steps - warmup)
    prog = min(1.0, max(0.0, prog))
    return 0.5 * base * (1.0 + math.cos(math.pi * prog))


print(f"\n[S56] ★容量轴续跑：目标卡 ff {FF_IN}→{FF_TGT}，步数 {STEPS56}（S54 的两倍，"
      f"warmup{WARMUP}+cosine→0 周期随总步数拉长）｜桶={BUCKET} 曲线格={GRID[0]}..{GRID[-1]}/"
      f"{GRID[1]-GRID[0]}步（{len(GRID)} 格）seeds={SEEDS56} | d={D} nhead={NHEAD} "
      f"层数=1(思维卡残差) batch={BATCH} MAXLEN={MAXLEN} base_lr={BASE_LR} | "
      f"EM主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | 线程={torch.get_num_threads()}(R40) | "
      f"smoke={SMOKE}", flush=True)
print(f"[S56-LR] 采样 lr：t=0/{WARMUP//2}/{WARMUP}/{STEPS56//4}/{STEPS56//2}/{3*STEPS56//4}/"
      f"{STEPS56-1} ⇒ " + " / ".join(f"{lr_at(t, STEPS56, WARMUP, BASE_LR):.3e}"
                                     for t in (0, WARMUP // 2, WARMUP, STEPS56 // 4,
                                               STEPS56 // 2, 3 * STEPS56 // 4, STEPS56 - 1)),
      flush=True)

# ============================================================================
# ② 数据核对（口径自检 + TEST 指纹 + clean 索引）
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

BAD_IDX = [i for i, r in enumerate(TEST) if "{expr}" in r["nl"]]
CLEAN_IDX = [i for i in range(len(TEST)) if i not in set(BAD_IDX)]
CLEAN_EVAL = [i for i in CLEAN_IDX if i < N_EVAL]
CLEAN_SET_EVAL = set(CLEAN_EVAL)
_c_tr = Counter(r["gold"] for r in TRAIN)
_top1, _top1n = _c_tr.most_common(1)[0]
_top5 = [v for v, _ in _c_tr.most_common(5)]
FLOOR_1 = sum(1 for r in TEST if r["gold"] == _top1) / len(TEST)
FLOOR_5 = sum(1 for r in TEST if r["gold"] in _top5) / len(TEST)

sha_ok = TEST_SHA == TEST_SHA_EXPECT
leak_ok = LEAK == LEAK_EXPECT
clean_ok = (len(BAD_IDX), len(CLEAN_IDX)) == (30, 770)
print(f"[S56-DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)}(eval {len(TEST_EVAL)}) "
      f"答案域={min(answer_values(BUCKET))}..{max(answer_values(BUCKET))} | 不切 dev", flush=True)
print(f"[S56-CHK1] TEST 指纹 sha256[:16]={TEST_SHA} vs 入库 {TEST_SHA_EXPECT} ⇒ "
      f"{'逐位一致 ✓' if sha_ok else '★不一致'} | (a,b) 泄漏对={LEAK}/{len(TEST)} vs {LEAK_EXPECT} "
      f"⇒ {'一致 ✓' if leak_ok else '★不一致'}", flush=True)
print(f"[S56-CHK2] clean 口径：nl 含字面量 '{{expr}}' 的 bug 样本={len(BAD_IDX)}（全在 test）"
      f"⇒ clean n={len(CLEAN_IDX)} vs S42/S43 入库 30/770 ⇒ "
      f"{'一致 ✓' if clean_ok else '★不一致'}", flush=True)
print(f"[S56-CHK3] SEEN={N_SEEN} / UNSEEN={N_UNSEEN} | 地板：单答案 '{_top1}'={FLOOR_1*100:.2f}% "
      f"top5={_top5}={FLOOR_5*100:.2f}%（本地板，不借 1.89%）| 干预面=无 ⇒ R29 不适用", flush=True)
if not SMOKE:
    assert sha_ok and leak_ok and clean_ok, "数据口径不符 ⇒ 本单元无效"


# ============================================================================
# ③ 模型：与 S54 的 Cards54 逐字同构（★唯一变量 = 目标卡 ff=1024）
# ============================================================================
class Cards56(nn.Module):
    """h = h + Think(h)（k=1），w1 = 全部 token 权重 1（纯 masked_ce）。
    ff_in = 输入卡宽度（512，不动）；ff_tgt = ★目标卡宽度（1024）。"""

    def __init__(self, d, ff_in, ff_tgt):
        super().__init__()
        self.d = d
        self.ff_in, self.ff_tgt = ff_in, ff_tgt
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff_in)
        self.thought = enc_layer(d, ff_tgt)          # ★唯一变量
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


def train56(seed, train, steps, grid, save_ats):
    """★与 S54 的 train54 逐字相同（同数据顺序、同 dropout 消耗），唯一差别 = steps/grid/save_ats。
    因此 step=12000 的状态应与 S54 的 ckpt 轨迹同源（但 cosine 周期不同 ⇒ 逐点数值会有差）。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards56(D, FF_IN, FF_TGT).to(device)
    opt = AdamW(model.parameters(), lr=BASE_LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0 = 0, 0, time.time()
    last, snaps, ckpts = 0.0, {}, {}
    gset = set(grid)
    inst, blk = {}, {}
    wsum, wcnt = 0.0, 0
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        for g in opt.param_groups:
            g["lr"] = lr_at(step, steps, WARMUP, BASE_LR)
        loss = masked_ce(model, [train[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        wsum += float(loss.item())
        wcnt += 1
        if step in gset:
            snaps[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step in save_ats:
                ck = CKPT_TMPL.format(steps=step, seed=seed)
                torch.save(snaps[step], ck)
                ckpts[step] = ck
        if step % 1500 == 0 or step == steps:
            el = time.time() - t0
            inst[step], blk[step] = float(loss.item()), wsum / wcnt
            wsum = wcnt = 0
            print(f"  [train:ff{FF_TGT}] step={step}/{steps} loss={inst[step]:.4f} "
                  f"块均值={blk[step]:.4f} lr={opt.param_groups[0]['lr']:.3e} "
                  f"elapsed={el:.0f}s s/step={(el - last) / 1500:.3f}", flush=True)
            last = el
    assert set(grid) <= set(snaps), "未取到全部曲线格点快照"
    return model, time.time() - t0, snaps, ckpts, inst, blk


@torch.no_grad()
def em_at(model, recs):
    model.eval()                                            # ★R39 显式 eval()
    txt = greedy_gen(model, recs, batch=EM_BATCH)            # ★R28 主口径 batch=1
    return [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, recs)], txt


@torch.no_grad()
def per_sample_ce(model, recs):
    """逐样本 (整体CE, 数字/算子 token 的 CE)；口径 = S19 ce_categories 的 samp_all / dig_samp。"""
    model.eval()                                            # ★R39
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
    model.train()
    return all_ce, dig_ce


def dig_acc(dig_ce):
    ys = [x for x in dig_ce if x == x]
    m, se = mean_se(ys)
    acc = math.exp(-m)
    return acc, acc * se, m, se, len(ys)


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def frac(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sub(hits, idx):
    return [hits[i] for i in idx]


def paired_delta(a_hits, b_hits, idx=None):
    """配对 Δ = a − b（逐 test 样本），SE = sd(d_i)/√n。"""
    if idx is not None:
        a_hits, b_hits = sub(a_hits, idx), sub(b_hits, idx)
    d = [x - y for x, y in zip(a_hits, b_hits)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n), n


def verdict(delta, se):
    if delta >= 2 * se and delta > 0:
        return "★提升（≥+2SE）"
    if delta <= -2 * se and delta < 0:
        return "★有害（≤−2SE）"
    if abs(delta) <= se:
        return "无增益（|Δ|≤1SE）"
    return f"落在 1–2SE 之间（不判定，Δ/SE={delta/se:+.2f}）"


def pooled_verdict(ms, ses, same_sign):
    m = sum(ms) / len(ms)
    se = math.sqrt(sum(x * x for x in ses)) / len(ses)
    if not same_sign:
        return f"★2 seed 不同号 ⇒ 不判定（逐 seed 判定" \
               f"{' / '.join(f's{i}:{verdict(ms[i], ses[i])}' for i in range(len(ms)))}）"
    if m >= 2 * se and m > 0:
        return "★容量有增益（Δ≥+2SE，2 seed 同号）"
    if m <= -2 * se and m < 0:
        return "★容量有害（Δ≤−2SE，2 seed 同号）"
    if abs(m) <= se:
        return "★追平（|Δ|≤1SE，2 seed 同号）—— 容量能追平但无增益"
    return f"落在 1–2SE 之间（不判定，Δ/SE={m/se:+.2f}）"


# ============================================================================
# ④ 参数量对照
# ============================================================================
def card_params(ff):
    m = enc_layer(D, ff)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


N_BASE = sum(p.numel() for p in Cards56(D, FF_IN, FF_IN).parameters())
N_CAP = sum(p.numel() for p in Cards56(D, FF_IN, FF_TGT).parameters())
CARD_BASE, CARD_CAP = card_params(FF_IN), card_params(FF_TGT)
D_TGT = CARD_CAP - CARD_BASE
D_TOT = N_CAP - N_BASE
PCT_TGT = 100.0 * D_TGT / CARD_BASE
PCT_TOT = 100.0 * D_TOT / N_BASE
print(f"\n[S56-PARAMS] 目标卡：ff {FF_IN}→{FF_TGT} ⇒ {CARD_BASE/1e6:.4f}M → {CARD_CAP/1e6:.4f}M "
      f"（Δ=+{D_TGT/1e6:.4f}M = +{PCT_TGT:.1f}%）| 总量 {N_BASE/1e6:.4f}M → {N_CAP/1e6:.4f}M "
      f"（+{PCT_TOT:.2f}%）⇒ 与 S54 同构（+5.26%）", flush=True)

_ck_probe = torch.load(BASE_CKPT.format(steps=12000, seed=SEEDS56[0]),
                       map_location="cpu", weights_only=True)
_sd_probe = Cards56(D, FF_IN, FF_IN).state_dict()
_kk = sorted(set(_ck_probe) | set(_sd_probe))
_shape_bad = [k for k in _kk
              if k not in _ck_probe or k not in _sd_probe
              or tuple(_ck_probe[k].shape) != tuple(_sd_probe[k].shape)]
print(f"[S56-CHK4] S53 ckpt 键兼容自检：键数={len(_ck_probe)} vs Cards56(ff={FF_IN})="
      f"{len(_sd_probe)} | 形状不符={len(_shape_bad)}{_shape_bad} ⇒ "
      f"{'逐位兼容 ✓（可安全重载对照臂）' if not _shape_bad else '★不兼容'}", flush=True)
assert not _shape_bad, "S53 ckpt 与对照臂结构不匹配 ⇒ 本单元无效"
del _ck_probe, _sd_probe


# ============================================================================
# ⑤ 对照臂（S53 的 lr_cosine @12000）= 重载 53 ckpt（★锚点复核）
# ============================================================================
@torch.no_grad()
def eval_states(ev, states, steps, tag, do_ce=True):
    out = {}
    for st in steps:
        ev.load_state_dict({k: v.to(device) for k, v in states[st].items()})
        ev.eval()                                                # ★R39
        hits, _ = em_at(ev, TEST_EVAL)
        full, clean = frac(hits), frac([hits[i] for i in CLEAN_EVAL])
        seen = frac([hits[i] for i, r in enumerate(TEST_EVAL) if r["seen"]])
        unseen = frac([hits[i] for i, r in enumerate(TEST_EVAL) if not r["seen"]])
        rec = dict(em_full=full, em_clean=clean, em_seen=seen, em_unseen=unseen, hits=hits)
        if do_ce:
            all_ce, dig_ce = per_sample_ce(ev, TEST_EVAL)
            acc, acc_se, dce, dce_se, _n = dig_acc(dig_ce)
            rec.update(ce=frac(all_ce), dig_acc=acc, dig_ce=dce,
                       dig_acc_clean=frac([math.exp(-x) for i, x in enumerate(dig_ce)
                                           if x == x and i in CLEAN_SET_EVAL]))
        out[st] = rec
        print(f"  [{tag}] step={st} EM全量={full*100:.2f}%  clean={clean*100:.2f}% "
              f"SEEN={seen*100:.2f}% UNSEEN={unseen*100:.2f}%"
              + (f" | 数字每步={rec['dig_acc']*100:.2f}% 整体CE={rec['ce']:.4f}" if do_ce else ""),
              flush=True)
    return out


print(f"\n[S56] ===== 对照臂（S53 lr_cosine ff={FF_IN} @12000）= 重载 53 ckpt（★锚点复核）=====",
      flush=True)
ev = Cards56(D, FF_IN, FF_IN).to(device)
ev_cap = Cards56(D, FF_IN, FF_TGT).to(device)   # ★容量臂专属评估模型（目标卡 ff=1024）
BASE_EV, BASE_SHA = {}, {}
for s in SEEDS56:
    pp = BASE_CKPT.format(steps=12000, seed=s)
    bst = torch.load(pp, map_location="cpu", weights_only=True)
    BASE_SHA[s] = sha16(pp)
    BASE_EV[s] = eval_states(ev, {12000: bst}, [12000], f"BASE53 s{s}")
    r = BASE_EV[s][12000]
    if not SMOKE:
        ref = ANCHOR_S53[s]
        dd = (r["em_full"] - ref["full"]) * 100
        dc = (r["em_clean"] - ref["clean"]) * 100
        print(f"[S56-ANCHOR] 对照臂 s{s}@12000: 本次全量={r['em_full']*100:.2f}% vs S53 入库 "
              f"{ref['full']*100:.2f}% ⇒ Δ={dd:+.2f}pp | clean={r['em_clean']*100:.2f}% vs "
              f"{ref['clean']*100:.2f}% ⇒ Δ={dc:+.2f}pp ⇒ "
              f"{'复现 ✓' if abs(dd) <= 0.2 and abs(dc) <= 0.2 else '★不复现，点名'}", flush=True)

# S53 逐格曲线（既有 JSONL，只读）⇒ CE / EM 曲线对照
BASE53_CURVE, BASE53_TRAINLOSS = {}, {}
try:
    for ln in open(BASE_JSONL, encoding="utf-8"):
        d = json.loads(ln)
        BASE53_CURVE[d["seed"]] = {int(k): v for k, v in d["curve"].items()}
        BASE53_TRAINLOSS[d["seed"]] = {int(k): v for k, v in d["train_loss"].items()}
    assert all(s in BASE53_CURVE for s in SEEDS56), "53_results.jsonl 缺 seed"
    print(f"[S56-CHK5] 对照曲线来源 logs/53_results.jsonl：2 seed × {len(BASE53_CURVE[1234])} 格"
          f"（S53 只到 12000 ⇒ >12000 无 S53 对照）", flush=True)
except Exception as _e:                                          # noqa: BLE001
    BASE53_CURVE, BASE53_TRAINLOSS = {}, {}
    print(f"[S56-WARN] 读 logs/53_results.jsonl 失败({_e})", flush=True)


# ============================================================================
# ⑥ 容量臂：warmup+cosine（周期 = 24000），只把目标卡 ff→1024
# ============================================================================
RESULTS, CURVE = {}, {}
if not SMOKE:
    open(JSONL, "w").close()
for s in SEEDS56:
    t0 = time.time()
    print(f"\n[S56] ===== seed={s} 容量臂 ff{FF_IN}→{FF_TGT} 步数={STEPS56} | "
          f"train={len(TRAIN)} test={len(TEST_EVAL)} 曲线格={len(GRID)} 个 =====", flush=True)
    model, wall_train, snaps, ckpts, inst, blk = train56(s, TRAIN, STEPS56, GRID, SAVE_ATS)
    print(f"[S56] seed={s} 训练结束 墙钟={wall_train/60:.2f}min | ckpt 落盘 {sorted(ckpts)}",
          flush=True)
    curve = eval_states(ev_cap, snaps, GRID, f"CAP s{s}")
    fin = GRID[-1]
    seen = [curve[fin]["hits"][i] for i, r in enumerate(TEST_EVAL) if r["seen"]]
    unseen = [curve[fin]["hits"][i] for i, r in enumerate(TEST_EVAL) if not r["seen"]]
    ev_cap.load_state_dict({k: v.to(device) for k, v in snaps[fin].items()})
    ev_cap.eval()                                                # ★R39
    _r28 = r28_selfcheck(ev_cap, TEST_EVAL, tag=f"[cap ff{FF_TGT} s{fin} seed{s}] ",
                         k=min(CHK_BATCH, len(TEST_EVAL)))
    ckpt_meta = {st: dict(path=ckpts[st], sha=sha16(ckpts[st])) for st in ckpts}
    RESULTS[s] = dict(seed=s, steps=STEPS56, wall_train=wall_train,
                      wall_total=time.time() - t0, n_eval=len(TEST_EVAL),
                      em_final=curve[fin]["em_full"], em_final_clean=curve[fin]["em_clean"],
                      em_seen=frac(seen), em_unseen=frac(unseen), n_seen=len(seen),
                      n_unseen=len(unseen), dig_acc_final=curve[fin]["dig_acc"],
                      ce_final=curve[fin]["ce"], r28_same=_r28["same"], r28_k=_r28["k"],
                      ckpt=ckpt_meta, train_loss=inst, train_loss_blk=blk)
    CURVE[s] = curve
    del snaps, model
    torch.cuda.empty_cache()
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(seed=s, steps=STEPS56, wall_train=wall_train,
                                curve={str(st): {k: v for k, v in curve[st].items()
                                                 if k != "hits"} for st in GRID},
                                train_loss={str(k): v for k, v in inst.items()},
                                train_loss_blk={str(k): v for k, v in blk.items()},
                                r28_same=_r28["same"], r28_k=_r28["k"],
                                ckpt={str(k): v for k, v in ckpt_meta.items()}),
                   ensure_ascii=False) + "\n")
    print(f"[S56-RUN] seed={s}: EM全量@{fin}={curve[fin]['em_full']*100:.2f}% "
          f"clean={curve[fin]['em_clean']*100:.2f}% SEEN={frac(seen)*100:.2f}%({len(seen)}) "
          f"UNSEEN={frac(unseen)*100:.2f}%({len(unseen)}) | 数字每步={curve[fin]['dig_acc']*100:.2f}% "
          f"| R28={_r28['same']}/{_r28['k']} | 合计墙钟={RESULTS[s]['wall_total']/60:.2f}min",
          flush=True)

FIN = GRID[-1]
BASE_FIN = 12000          # 对照臂恒为 S53 @12000


# ============================================================================
# ⑦ 主判据：Δ（配对）— 全量与 clean 两种口径 × 2 seed
# ============================================================================
print(f"\n[S56] ===== ★主判据：Δ = EM(ff{FF_TGT}@{FIN}) − EM(S53 ff{FF_IN}@{BASE_FIN})，"
      f"逐样本配对 ±SE（S53 基线 93.62/97.27）=====", flush=True)
DELTA = {}
for s in SEEDS56:
    a = CURVE[s][FIN]["hits"]
    b = BASE_EV[s][BASE_FIN]["hits"]
    for name, idx in (("全量", None), ("clean", CLEAN_EVAL)):
        m, se, n = paired_delta(a, b, idx)
        DELTA[(s, name)] = (m, se, n)
        print(f"[S56-DELTA] seed={s} {name}(n={n}): Δ={m*100:+.2f}pp ± {se*100:.2f}pp "
              f"({m/se if se else float('nan'):+.2f}SE) ⇒ {verdict(m, se)}", flush=True)
for name in ("全量", "clean"):
    ms = [DELTA[(s, name)][0] for s in SEEDS56]
    ses = [DELTA[(s, name)][1] for s in SEEDS56]
    same_sign = all(m > 0 for m in ms) or all(m < 0 for m in ms)
    pooled = sum(ms) / len(ms)
    pooled_se = math.sqrt(sum(x * x for x in ses)) / len(ses)
    print(f"[S56-VERDICT] {name}口径：2 seed 均值 Δ={pooled*100:+.2f}pp ± {pooled_se*100:.2f}pp"
          f"（{ms[0]*100:+.2f}/{ms[1]*100:+.2f}）| 同号={'是' if same_sign else '否'} ⇒ "
          f"{pooled_verdict(ms, ses, same_sign)}", flush=True)

# ============================================================================
# ⑧ ★同 step 对照（分离「容量效应」与「步数效应」）
# ============================================================================
MID = 12000 if 12000 in GRID else GRID[len(GRID) // 2]
print(f"\n[S56] ===== ★同 step 对照：分离容量效应 / 步数效应（同 step 点 = @{MID}）=====", flush=True)
for name, idx in (("全量", None), ("clean", CLEAN_EVAL)):
    # (a) @MID: ff1024@12000 vs S53@12000（复核 S54）
    rows = []
    for s in SEEDS56:
        m, se, n = paired_delta(CURVE[s][MID]["hits"], BASE_EV[s][BASE_FIN]["hits"], idx)
        rows.append(f"s{s}: {m*100:+.2f}±{se*100:.2f}pp")
        if name == "全量" and not SMOKE:
            ref = S54_AT_12K[s]
            print(f"[S56-SAMESTEP] seed={s} @{MID} ff1024 vs S53@{BASE_FIN} {name}: "
                  f"{m*100:+.2f}pp ± {se*100:.2f}pp | 本次 ff1024@{MID} 全量="
                  f"{CURVE[s][MID]['em_full']*100:.2f}% clean={CURVE[s][MID]['em_clean']*100:.2f}% "
                  f"vs S54 入库 {ref['full']*100:.2f}/{ref['clean']*100:.2f}%（cosine 周期不同，"
                  f"数值不必逐位相同）", flush=True)
    print(f"[S56-SAMESTEP] @{MID} 容量 vs S53（{name}，复核 S54）：" + " | ".join(rows), flush=True)
    # (b) 步数效应（容量臂内部）：@FIN vs @MID
    rows = []
    for s in SEEDS56:
        m, se, n = paired_delta(CURVE[s][FIN]["hits"], CURVE[s][MID]["hits"], idx)
        rows.append(f"s{s}: {m*100:+.2f}±{se*100:.2f}pp")
    print(f"[S56-STEPEFFECT] 容量臂 @{FIN} vs @{MID}（纯步数效应，{name}）：" + " | ".join(rows),
          flush=True)

# ============================================================================
# ⑨ ★饱和检查（R35，本单元核心：@21000→@24000）
# ============================================================================
print(f"\n[S56] ===== ★饱和检查（R35，本单元核心）：@21000→@24000 =====", flush=True)
SAT_STEPS = [st for st in (21000, 24000) if st in GRID]
if len(SAT_STEPS) == 2:
    sat_full, sat_cl = [], []
    for s in SEEDS56:
        d_full = (CURVE[s][24000]["em_full"] - CURVE[s][21000]["em_full"]) * 100
        d_cl = (CURVE[s][24000]["em_clean"] - CURVE[s][21000]["em_clean"]) * 100
        sat_full.append(d_full)
        sat_cl.append(d_cl)
        print(f"[S56-SAT] seed={s}: 21000→24000 Δ全量={d_full:+.2f}pp / clean={d_cl:+.2f}pp "
              f"（@21000 全量={CURVE[s][21000]['em_full']*100:.2f}% → @24000 "
              f"{CURVE[s][24000]['em_full']*100:.2f}%）", flush=True)
    mf, mc = sum(sat_full) / len(sat_full), sum(sat_cl) / len(sat_cl)
    print(f"[S56-SAT-VERDICT] 2 seed 均值 @21000→@24000 Δ全量={mf:+.2f}pp / clean={mc:+.2f}pp ⇒ "
          f"{'★已饱和（<1pp）⇒ 结论可下' if max(mf, mc) < 1.0 else '★仍未饱和（≥1pp）⇒ 结论标【条件性】'}",
          flush=True)
    # 外推（明确标注：线性外推，非实测）
    if max(mf, mc) >= 1.0:
        TGT = ANCHOR_S53[1234]["clean"] * 100
        for s in SEEDS56:
            cur = CURVE[s][24000]["em_clean"] * 100
            rate = (CURVE[s][24000]["em_clean"] - CURVE[s][21000]["em_clean"]) * 100 / 3000.0
            need = (TGT - cur) / rate if rate > 1e-9 else float("inf")
            print(f"[S56-EXTRAP] seed={s} clean @24000={cur:.2f}% vs S53 平台={TGT:.2f}% | "
                  f"最近 3000 步速率={rate*1000:+.2f}pp/1k步 ⇒ ★线性外推还需≈{need:.0f} 步"
                  f"（外推，非实测；速率在 cosine 尾部会继续衰减 ⇒ 实际更长）", flush=True)
        print(f"[S56-EXTRAP] 均值外推：按 2 seed 均值速率 ⇒ 还需"
              f"≈{sum(((ANCHOR_S53[1234]['clean']*100 - CURVE[s][24000]['em_clean']*100) / ((CURVE[s][24000]['em_clean'] - CURVE[s][21000]['em_clean']) / 3000.0)) for s in SEEDS56) / 2:.0f} "
              f"步（★外推；且 clean 平台 97.27% 本身是 S53 的值，非容量臂上界）", flush=True)

# ============================================================================
# ⑩ 两 seed 极差（随训练是否收窄）+ EM 曲线表
# ============================================================================
print(f"\n[S56] ===== ★两 seed 极差随训练的变化 =====", flush=True)
for name, key in (("全量", "em_full"), ("clean", "em_clean")):
    print(f"[S56-SPREAD] {name}：" + " | ".join(
        f"@{st}={max(CURVE[s][st][key] for s in SEEDS56)*100 - min(CURVE[s][st][key] for s in SEEDS56)*100:.2f}pp"
        f"({min(CURVE[s][st][key] for s in SEEDS56)*100:.2f}–{max(CURVE[s][st][key] for s in SEEDS56)*100:.2f})"
        for st in REPORT_STEPS if st in GRID), flush=True)
b53 = [BASE_EV[s][BASE_FIN]["em_full"] for s in SEEDS56]
b53c = [BASE_EV[s][BASE_FIN]["em_clean"] for s in SEEDS56]
print(f"[S56-SPREAD] 对照 S53@{BASE_FIN}：全量极差={(max(b53)-min(b53))*100:.2f}pp / "
      f"clean={(max(b53c)-min(b53c))*100:.2f}pp（入库 0.00pp）", flush=True)
print(f"\n[S56-TABLE] ★ff1024 曲线 EM 全量/clean × 2 seed（@12000/15000/18000/21000/24000）",
      flush=True)
for st in REPORT_STEPS:
    if st not in GRID:
        continue
    row = " | ".join(f"s{s}: {CURVE[s][st]['em_full']*100:.2f}/{CURVE[s][st]['em_clean']*100:.2f}"
                     for s in SEEDS56)
    span_f = (max(CURVE[s][st]["em_full"] for s in SEEDS56)
              - min(CURVE[s][st]["em_full"] for s in SEEDS56)) * 100
    print(f"[S56-TABLE] @{st} | {row} | 极差={span_f:.2f}pp", flush=True)
print(f"[S56-TABLE-FULL] 逐格全曲线（每 750 步）：", flush=True)
for st in GRID:
    row = " | ".join(f"{CURVE[s][st]['em_full']*100:.2f}/{CURVE[s][st]['em_clean']*100:.2f}"
                     for s in SEEDS56)
    print(f"[S56-GRID] {st} | {row}", flush=True)

# ============================================================================
# ⑪ CE 曲线（训练 CE 同协议 + 测试整体 CE）@3000…@24000
# ============================================================================
CE_STEPS = [st for st in range(3000, STEPS56 + 1, 3000) if st in GRID]
print(f"\n[S56] ===== ★CE 曲线（训练单批 CE 与测试整体 CE）@3000…@24000 vs S53 =====", flush=True)
for st in CE_STEPS:
    row = []
    for s in SEEDS56:
        bi = RESULTS[s]["train_loss"].get(st, float("nan"))
        b53 = BASE53_TRAINLOSS.get(s, {}).get(st, float("nan"))
        row.append(f"s{s}: cap={bi:.4f} vs S53={b53:.4f} Δ={bi-b53:+.4f}"
                   if b53 == b53 else f"s{s}: cap={bi:.4f} vs S53(>12000 无)")
    print(f"[S56-TRAINCE] step={st} | " + " | ".join(row), flush=True)
for st in CE_STEPS:
    row = []
    for s in SEEDS56:
        cv = CURVE[s][st]["ce"]
        b53 = BASE53_CURVE.get(s, {}).get(st, {}).get("ce", float("nan"))
        row.append(f"s{s}: cap={cv:.4f} vs S53={b53:.4f} Δ={cv-b53:+.4f}"
                   if isinstance(b53, float) and b53 == b53 else f"s{s}: cap={cv:.4f} vs S53(无)")
    print(f"[S56-TESTCE] step={st} 测试整体CE | " + " | ".join(row), flush=True)
print(f"[S56-CE-VERDICT] 训练 CE：@12000 " + " ; ".join(
    f"s{s}: cap={RESULTS[s]['train_loss'].get(12000, float('nan')):.4f} vs "
    f"S53={BASE53_TRAINLOSS.get(s, {}).get(12000, float('nan')):.4f}" for s in SEEDS56)
    + f" | @24000 " + " ; ".join(
    f"s{s}: cap={RESULTS[s]['train_loss'].get(24000, float('nan')):.4f}" for s in SEEDS56),
    flush=True)

# ============================================================================
# ⑫ 成本、ckpt 与总表
# ============================================================================
print(f"\n[S56] ===== 成本与 ckpt =====", flush=True)
for s in SEEDS56:
    r = RESULTS[s]
    for st in sorted(r["ckpt"]):
        print(f"[META] seed={s}: steps={st} ckpt={os.path.basename(r['ckpt'][st]['path'])} "
              f"sha256[:16]={r['ckpt'][st]['sha']}", flush=True)
    print(f"[META] seed={s}: 训练墙钟={r['wall_train']/60:.2f}min 本run={r['wall_total']/60:.2f}min",
          flush=True)
print(f"[S56-EM@FIN] " + " | ".join(
    f"s{s}: 容量臂 全量={CURVE[s][FIN]['em_full']*100:.2f}% clean={CURVE[s][FIN]['em_clean']*100:.2f}% "
    f"SEEN={CURVE[s][FIN]['em_seen']*100:.2f}% UNSEEN={CURVE[s][FIN]['em_unseen']*100:.2f}% "
    f"数字每步={CURVE[s][FIN]['dig_acc']*100:.2f}% 测试CE={CURVE[s][FIN]['ce']:.4f}  ‖ S53@12000 "
    f"全量={BASE_EV[s][BASE_FIN]['em_full']*100:.2f}% clean={BASE_EV[s][BASE_FIN]['em_clean']*100:.2f}%"
    for s in SEEDS56), flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"线程={torch.get_num_threads()}(R40) | D={D} NHEAD={NHEAD} 层数=1 batch={BATCH} "
      f"MAXLEN={MAXLEN} | ★唯一结构变量=目标卡 ff {FF_IN}→{FF_TGT} | lr=warmup{WARMUP}"
      f"(0→{BASE_LR:g})+cos→0 @{STEPS56}步 | EM口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | "
      f"TEST sha16={TEST_SHA} 泄漏={LEAK}/{len(TEST)} clean n={len(CLEAN_IDX)} | "
      f"地板(单/top5)={FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% | 参数 基线={N_BASE/1e6:.3f}M → "
      f"容量={N_CAP/1e6:.3f}M（+{PCT_TOT:.2f}%）| 总墙钟={(time.time()-T0)/60:.1f}min | "
      f"run={len(RESULTS)}/{len(SEEDS56)}", flush=True)
print("[DONE] exit=0", flush=True)
sys.exit(0)
