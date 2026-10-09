#!/usr/bin/env python3
"""S53 · ★下一轮第一枪（优化轴）：warmup 500 + cosine LR → 0，@12000 步。

依据（已入库）：
  E9（stages/52_error_classify.py，提交 d027ab4）：平台是【模型顶】不是口径顶
    —— 220 个错例：解析错 0.0% / 数值错 100.0%；UNSEEN≈全量（4/4 ≤1SE）⇒ 非记忆缺口；
    ★ 且不是「难度墙」（按进位分桶 clean EM 无单调下降、位数零梯度）
    ⇒ 判据：普遍精度不足 + 长进位链轻度偏置。
  S38 实测平台仍有抖动（两 seed 平台差 ~4.8pp；12000→15000 仅 −0.64/+0.13pp）
    ⇒ 不是步数问题，可能是优化调度问题。

设计（★单变量：只改 LR schedule，结构/数据/batch 逐字不动）：
  臂 lr_cosine：warmup 500 步（线性 0→1e-3）+ cosine 衰减到 0，总 12000 步；
  对照 = S38 的 w1@12000（logs/38_ckpt_w1_s12000_seed{1234,5678}.pt，常量 lr=1e-3）
    ⇒ 直接复用 ckpt（不重跑），本单元用同一评估路径重载并复核锚点；
  seed = 1234 / 5678 ⇒ 2 run；每 750 步存内存快照（与 S38 同一套格点）；
  ckpt 落盘 logs/53_ckpt_lrcosine_s12000_seed<s>.pt（另存 9000 用于饱和检查）。

★判据（写死）：
  1 主判据：Δ = EM(lr_cosine) − EM(基线 w1@12000) @12000，逐 test 样本配对 ±SE，
    全量与 clean（去 30 条 `{expr}`，n=770）两种口径都报：
      Δ ≥ +2SE 且 2 seed 同号 ⇒ 优化调度提升能力；|Δ| ≤ 1SE ⇒ 无影响；
      Δ ≤ −2SE 且 2 seed 同号 ⇒ 有害。
  2 辅判据：两 seed 极差是否收窄 ≥1pp（基线极差 4.76pp 全量）。
  3 饱和检查：报 lr_cosine 在 9000→12000 的增量 ⇒ ≥1pp ⇒ 声明「未饱和、结论条件性」。
  4 训练 CE 曲线：lr_cosine 与基线在 @3000/6000/9000/12000 的训练 CE（单批 loss，
     与 S38 日志同协议）+ 每 1500 步块均值；另附测试整体 CE @6000/9000/12000。

★口径：EM 一律 batch=1（R28）＋ batch=1 vs 16 自检；R39 显式 eval()；
  R40 batch=1 逐样本前向 4 线程；地板 0.12%/0.25%；SEEN/UNSEEN 分层；
  TEST sha256[:16] == b34e7ea515203227（R29：无干预 ⇒ 不适用）。
  锚点：基线重载必须复现 S38 的 91.88% 全量 / 95.45% clean（报 Δ）。

复用：exec stages/19_residual_scan.py 到 "# ③ 主循环" 之前（同 tokenizer、同 add_3d
  train4000/test800、同 build_batch/greedy_gen(EM_BATCH=1)/parse_ans/r28_selfcheck）。
只允许写：本文件、logs/s53_lrcos.log、logs/53_*、/tmp；既有 stages/*.py 只读。
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

SMOKE = os.environ.get("S53_SMOKE") == "1"
if SMOKE:
    os.environ["S19_SMOKE"] = "1"

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

torch.set_num_threads(4)                # ★R40：batch=1 逐样本前向统一 4 线程

T0 = time.time()
BUCKET = "add_3d"
SEEDS53 = (1234, 5678)
STEPS53 = 90 if SMOKE else 12000
GRID = tuple(range(30, 91, 30)) if SMOKE else tuple(range(750, 12001, 750))
WARMUP = 5 if SMOKE else 500
BASE_LR = LR                            # 1e-3（与 S38 常量臂同一个 base lr）
SAVE_ATS = (30, 60, 90) if SMOKE else (9000, 12000)
N_EVAL = 60 if SMOKE else N_TEST        # 800
CKPT_TMPL = f"{ROOT}/logs/53_ckpt_lrcosine_s{{steps}}_seed{{seed}}.pt"
BASE_CKPT = f"{ROOT}/logs/38_ckpt_w1_s{{steps}}_seed{{seed}}.pt"
BASE_TRACE_LOG = f"{ROOT}/logs/s38_sat.log"
JSONL = "/tmp/53_results_smoke.jsonl" if SMOKE else f"{ROOT}/logs/53_results.jsonl"
TEST_SHA_EXPECT = "b34e7ea515203227"
LEAK_EXPECT = 16
# S43（logs/43_clean_em.log:31-32）入库的基线 w1@12000（重载后必须复现，差 ≤0.2pp）
ANCHOR_S38 = {1234: dict(full=0.9188, clean=0.9545, n_clean=770),
              5678: dict(full=0.8712, clean=0.9052, n_clean=770)}
BASE_CE_REF = {1234: {6000: 0.0642, 9000: 0.0667, 12000: 0.0548},
               5678: {6000: 0.1010, 9000: 0.0846, 12000: 0.0773}}
# S38 日志里的单批训练 loss（同协议对照值，也由脚本从日志解析校验）
BASE_TRAIN_LOSS_REF = {1234: {3000: 0.4071, 6000: 0.1335, 9000: 0.0740, 12000: 0.0952},
                       5678: {3000: 0.3338, 6000: 0.1257, 9000: 0.0962, 12000: 0.0246}}


def lr_at(t: int, steps: int, warmup: int, base: float) -> float:
    """★本单元唯一变量：warmup 线性 0→base，随后 cosine 衰减到 0（t = 0-based 步号）。"""
    if t < warmup:
        return base * (t / warmup)
    prog = (t - warmup) / max(1, steps - warmup)
    prog = min(1.0, max(0.0, prog))
    return 0.5 * base * (1.0 + math.cos(math.pi * prog))


print(f"\n[S53] ★优化轴：warmup{WARMUP}+cosine→0 vs 常量 lr（复用 S38 w1@12000）｜桶={BUCKET} "
      f"步数={STEPS53} 曲线格={GRID[0]}..{GRID[-1]}/{GRID[1]-GRID[0]}步 seeds={SEEDS53} | "
      f"d={D} ff={FF} NHEAD={NHEAD} base_lr={BASE_LR} batch={BATCH} MAXLEN={MAXLEN} 思维卡残差 | "
      f"EM主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | 线程={torch.get_num_threads()}(R40) | "
      f"smoke={SMOKE}", flush=True)
print(f"[S53-LR] 采样 lr：t=0/{WARMUP//2}/{WARMUP}/{STEPS53//4}/{STEPS53//2}/{3*STEPS53//4}/"
      f"{STEPS53-1} ⇒ " + " / ".join(f"{lr_at(t, STEPS53, WARMUP, BASE_LR):.3e}"
                                     for t in (0, WARMUP // 2, WARMUP, STEPS53 // 4,
                                               STEPS53 // 2, 3 * STEPS53 // 4, STEPS53 - 1)),
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

BAD_IDX = [i for i, r in enumerate(TEST) if "{expr}" in r["nl"]]        # S42/S43：模板 bug
CLEAN_IDX = [i for i in range(len(TEST)) if i not in set(BAD_IDX)]
_c_tr = Counter(r["gold"] for r in TRAIN)
_top1, _top1n = _c_tr.most_common(1)[0]
_top5 = [v for v, _ in _c_tr.most_common(5)]
FLOOR_1 = sum(1 for r in TEST if r["gold"] == _top1) / len(TEST)
FLOOR_5 = sum(1 for r in TEST if r["gold"] in _top5) / len(TEST)

sha_ok = TEST_SHA == TEST_SHA_EXPECT
leak_ok = LEAK == LEAK_EXPECT
clean_ok = (len(BAD_IDX), len(CLEAN_IDX)) == (30, 770)
print(f"[S53-DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)}(eval {len(TEST_EVAL)}) "
      f"答案域={min(answer_values(BUCKET))}..{max(answer_values(BUCKET))} | 不切 dev", flush=True)
print(f"[S53-CHK1] TEST 指纹 sha256[:16]={TEST_SHA} vs 入库 {TEST_SHA_EXPECT} ⇒ "
      f"{'逐位一致 ✓' if sha_ok else '★不一致'} | (a,b) 泄漏对={LEAK}/{len(TEST)} vs {LEAK_EXPECT} "
      f"⇒ {'一致 ✓' if leak_ok else '★不一致'}", flush=True)
print(f"[S53-CHK2] clean 口径：nl 含字面量 '{{expr}}' 的 bug 样本={len(BAD_IDX)}（全在 test）"
      f"⇒ clean n={len(CLEAN_IDX)} vs S42/S43 入库 30/770 ⇒ "
      f"{'一致 ✓' if clean_ok else '★不一致'}", flush=True)
print(f"[S53-CHK3] SEEN={N_SEEN} / UNSEEN={N_UNSEEN} | 地板：单答案 '{_top1}'={FLOOR_1*100:.2f}% "
      f"top5={_top5}={FLOOR_5*100:.2f}%（本地板，不借 1.89%）| 干预面=无 ⇒ R29 不适用", flush=True)
if not SMOKE:
    assert sha_ok and leak_ok and clean_ok, "数据口径不符 ⇒ 本单元无效"


# ============================================================================
# ③ 模型：与 S38 Cards38 逐字同构（⇒ 与基线臂同初始化/同参数量）
# ============================================================================
class Cards53(nn.Module):
    """h = h + Think(h)（k=1），w1 = 全部 token 权重 1（纯 masked_ce）。"""

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


def train53(seed, train, steps, grid, save_ats):
    """★与 S38 train38 唯一差别：优化器 lr 由 lr_at() 调度（其余逐字相同）。
    ⇒ 同 seed 下初始化/数据顺序/dropout 消耗完全一致，唯一自由变量是 LR schedule。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards53(D, FF).to(device)
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
        for g in opt.param_groups:                     # ★唯一变量：写 lr
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
            print(f"  [train:lrcos] step={step}/{steps} loss={inst[step]:.4f} "
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
    all_ce, dig_ce, dig_tok = [], [], []
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
    return all_ce, dig_ce, dig_tok


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
        return "无影响（|Δ|≤1SE）"
    return f"落在 1–2SE 之间（不判定，Δ/SE={delta/se:+.2f}）"


def parse_base_trace(path):
    """从 S38 日志解析 (seed → {step: 单批 loss})：同协议对照，避免手抄。
    S38 日志按 seed 顺序打印（先 1234 全部 10 行，再 5678）⇒ 对半切。"""
    cur = []
    for ln in open(path, encoding="utf-8"):
        if "[train:w1]" in ln:
            cur.append(ln)
    assert len(cur) == 2 * (15000 // 1500), f"S38 日志 train:w1 行数={len(cur)}（期望 20）"
    half = len(cur) // 2
    out = {}
    for si, seed in enumerate(SEEDS53):
        d = {}
        for ln in cur[si * half:(si + 1) * half]:
            p = ln.split()
            st = int(next(x for x in p if x.startswith("step=")).split("=")[1].split("/")[0])
            d[st] = float(next(x for x in p if x.startswith("loss=")).split("=")[1])
        out[seed] = d
    return out


try:
    BASE_TRACE = parse_base_trace(BASE_TRACE_LOG)
except Exception as _e:                                          # noqa: BLE001
    BASE_TRACE = {s: dict(BASE_TRAIN_LOSS_REF[s]) for s in SEEDS53}
    print(f"[S53-WARN] 解析 S38 训练日志失败({_e}) ⇒ 用硬编码对照值", flush=True)
for s in SEEDS53:                                                # 与硬编码表交叉核对
    for st, v in BASE_TRAIN_LOSS_REF[s].items():
        assert abs(BASE_TRACE[s].get(st, v) - v) < 1e-4, f"S38 日志解析与硬编码不一致 s={s} st={st}"

n_par = sum(p.numel() for p in Cards53(D, FF).parameters())
print(f"[S53-PARAMS] Cards53(res)={n_par/1e6:.3f}M（与 S38 w1 臂同构）", flush=True)


# ============================================================================
# ④ 评估：把「重载的基线 ckpt」与「本次 cosine 快照」放进同一条评估路径
# ============================================================================
@torch.no_grad()
def eval_states(ev, states, steps, tag, do_ce=True):
    """states: {step: state_dict}；返回 {step: dict(em_full, em_clean, hits, ce, dig_acc...)}。"""
    out = {}
    for st in steps:
        ev.load_state_dict({k: v.to(device) for k, v in states[st].items()})
        ev.eval()                                                # ★R39
        hits, _ = em_at(ev, TEST_EVAL)
        full, clean = frac(hits), frac([hits[i] for i in CLEAN_IDX[:N_EVAL]])
        seen = frac([hits[i] for i, r in enumerate(TEST_EVAL) if r["seen"]])
        unseen = frac([hits[i] for i, r in enumerate(TEST_EVAL) if not r["seen"]])
        rec = dict(em_full=full, em_clean=clean, em_seen=seen, em_unseen=unseen, hits=hits)
        if do_ce:
            all_ce, dig_ce, _ = per_sample_ce(ev, TEST_EVAL)
            acc, acc_se, dce, dce_se, _n = dig_acc(dig_ce)
            rec.update(ce=frac(all_ce), dig_acc=acc, dig_ce=dce, dig_acc_clean=frac(
                [math.exp(-x) for i, x in enumerate(dig_ce)
                 if x == x and i in set(CLEAN_IDX)]))
        out[st] = rec
        print(f"  [{tag}] step={st} EM全量={full*100:.2f}%  clean={clean*100:.2f}% "
              f"SEEN={seen*100:.2f}% UNSEEN={unseen*100:.2f}%"
              + (f" | 数字每步={rec['dig_acc']*100:.2f}% 整体CE={rec['ce']:.4f}" if do_ce else ""),
              flush=True)
    return out


# ---- ④a 基线：重载 S38 的 w1@12000（并附 @6000/@9000 供 CE 曲线对照） ----
print(f"\n[S53] ===== 基线臂（常量 lr=1e-3）= 重载 S38 ckpt（★锚点复核）=====", flush=True)
BASE_STEPS = [12000] if SMOKE else [6000, 9000, 12000]
BASE_STATES, BASE_SHA = {}, {}
for st in BASE_STEPS:
    p = BASE_CKPT.format(steps=st, seed="{seed}")
    BASE_SHA[st] = {}
    for s in SEEDS53:
        pp = p.format(seed=s)
        BASE_STATES[(s, st)] = torch.load(pp, map_location="cpu")
        BASE_SHA[st][s] = sha16(pp)
ev = Cards53(D, FF).to(device)
BASE_EV = {}
for s in SEEDS53:
    print(f"[S53-BASE] seed={s} 重载 {os.path.basename(BASE_CKPT.format(steps=BASE_STEPS[-1], seed=s))}",
          flush=True)
    BASE_EV[s] = eval_states(ev, {st: BASE_STATES[(s, st)] for st in BASE_STEPS},
                             BASE_STEPS, f"BASE s{s}")
    for st in BASE_STEPS:
        r = BASE_EV[s][st]
        if st == 12000:
            ref = ANCHOR_S38[s]
            dd = (r["em_full"] - ref["full"]) * 100
            dc = (r["em_clean"] - ref["clean"]) * 100
            print(f"[S53-ANCHOR] 基线 s{s}@12000: 本次全量={r['em_full']*100:.2f}% vs S43 入库 "
                  f"{ref['full']*100:.2f}% ⇒ Δ={dd:+.2f}pp | clean={r['em_clean']*100:.2f}% vs "
                  f"{ref['clean']*100:.2f}% ⇒ Δ={dc:+.2f}pp ⇒ "
                  f"{'复现 ✓' if abs(dd) <= 0.2 and abs(dc) <= 0.2 else '★不复现，点名'} | "
                  f"整体CE={r['ce']:.4f} vs S38={BASE_CE_REF[s][st]:.4f}", flush=True)

# ---- ④b 本臂：warmup+cosine，跑满 STEPS53 并在每 750 步评估 ----
RESULTS, CURVE = {}, {}
if not SMOKE:
    open(JSONL, "w").close()
for s in SEEDS53:
    t0 = time.time()
    print(f"\n[S53] ===== seed={s} lr_cosine(warmup{WARMUP}+cos→0) 步数={STEPS53} | "
          f"train={len(TRAIN)} test={len(TEST_EVAL)} 曲线格={len(GRID)} 个 =====", flush=True)
    model, wall_train, snaps, ckpts, inst, blk = train53(s, TRAIN, STEPS53, GRID, SAVE_ATS)
    print(f"[S53] seed={s} 训练结束 墙钟={wall_train/60:.2f}min | ckpt 落盘 {sorted(ckpts)}", flush=True)
    curve = eval_states(ev, snaps, GRID, f"COS s{s}")
    fin = GRID[-1]
    seen = [curve[fin]["hits"][i] for i, r in enumerate(TEST_EVAL) if r["seen"]]
    unseen = [curve[fin]["hits"][i] for i, r in enumerate(TEST_EVAL) if not r["seen"]]
    ev.load_state_dict({k: v.to(device) for k, v in snaps[fin].items()})
    ev.eval()                                                    # ★R39
    _r28 = r28_selfcheck(ev, TEST_EVAL, tag=f"[cos s{fin} seed{s}] ",
                         k=min(CHK_BATCH, len(TEST_EVAL)))
    ckpt_meta = {st: dict(path=ckpts[st], sha=sha16(ckpts[st])) for st in ckpts}
    RESULTS[s] = dict(seed=s, steps=STEPS53, wall_train=wall_train,
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
        f.write(json.dumps(dict(seed=s, steps=STEPS53, wall_train=wall_train,
                                curve={str(st): {k: v for k, v in curve[st].items()
                                                 if k != "hits"} for st in GRID},
                                train_loss={str(k): v for k, v in inst.items()},
                                train_loss_blk={str(k): v for k, v in blk.items()},
                                r28_same=_r28["same"], r28_k=_r28["k"],
                                ckpt={str(k): v for k, v in ckpt_meta.items()}),
                   ensure_ascii=False) + "\n")
    print(f"[S53-RUN] seed={s}: EM全量@{fin}={curve[fin]['em_full']*100:.2f}% "
          f"clean={curve[fin]['em_clean']*100:.2f}% SEEN={frac(seen)*100:.2f}%({len(seen)}) "
          f"UNSEEN={frac(unseen)*100:.2f}%({len(unseen)}) | 数字每步={curve[fin]['dig_acc']*100:.2f}% "
          f"| R28={_r28['same']}/{_r28['k']} | 合计墙钟={RESULTS[s]['wall_total']/60:.2f}min",
          flush=True)

# ============================================================================
# ⑤ 主判据：Δ（配对）— 全量与 clean 两种口径 × 2 seed
# ============================================================================
print(f"\n[S53] ===== ★主判据：Δ = EM(lr_cosine) − EM(基线 w1@12000)，逐样本配对 ±SE =====",
      flush=True)
FIN = GRID[-1] if not SMOKE else GRID[-1]
DELTA = {}
for s in SEEDS53:
    a = CURVE[s][FIN]["hits"]
    b = BASE_EV[s][12000]["hits"]
    for name, idx in (("全量", None), ("clean", CLEAN_IDX[:N_EVAL])):
        m, se, n = paired_delta(a, b, idx)
        DELTA[(s, name)] = (m, se, n)
        print(f"[S53-DELTA] seed={s} {name}(n={n}): Δ={m*100:+.2f}pp ± {se*100:.2f}pp "
              f"({m/se if se else float('nan'):+.2f}SE) ⇒ {verdict(m, se)}", flush=True)
for name in ("全量", "clean"):
    ms = [DELTA[(s, name)][0] for s in SEEDS53]
    ses = [DELTA[(s, name)][1] for s in SEEDS53]
    same_sign = all(m > 0 for m in ms) or all(m < 0 for m in ms)
    pooled = sum(ms) / len(ms)
    print(f"[S53-VERDICT] {name}口径：2 seed 均值 Δ={pooled*100:+.2f}pp（"
          f"{ms[0]*100:+.2f}/{ms[1]*100:+.2f}）| 同号={'是' if same_sign else '否'} | "
          f"逐 seed 判定=" + " ; ".join(f"s{s}: {verdict(DELTA[(s, name)][0], DELTA[(s, name)][1])}"
                                        for s in SEEDS53), flush=True)

# ============================================================================
# ⑥ 辅判据：两 seed 极差（基线 vs lr_cosine）
# ============================================================================
print(f"\n[S53] ===== ★辅判据：两 seed 极差（@{FIN}）=====", flush=True)
for name, key in (("全量", "em_full"), ("clean", "em_clean")):
    bvals = [BASE_EV[s][12000][key] for s in SEEDS53]
    cvals = [CURVE[s][FIN][key] for s in SEEDS53]
    bspan, cspan = max(bvals) - min(bvals), max(cvals) - min(cvals)
    print(f"[S53-SPREAD] {name}: 基线={bspan*100:.2f}pp（{min(bvals)*100:.2f}–{max(bvals)*100:.2f}）"
          f" | lr_cosine={cspan*100:.2f}pp（{min(cvals)*100:.2f}–{max(cvals)*100:.2f}）"
          f" ⇒ 收窄={((bspan-cspan)*100):+.2f}pp "
          f"{'★≥1pp 收窄（有价值）' if (bspan-cspan) >= 0.01 else '未达 1pp'}", flush=True)

# ============================================================================
# ⑦ 训练 CE 曲线对照 + 测试 CE + 饱和检查
# ============================================================================
CE_STEPS = [st for st in (3000, 6000, 9000, 12000) if st <= STEPS53]
print(f"\n[S53] ===== ★训练 CE 曲线（单批 loss，与 S38 同协议）+ 测试整体 CE =====", flush=True)
for st in CE_STEPS:
    row = []
    for s in SEEDS53:
        bi = RESULTS[s]["train_loss"].get(st, float("nan"))
        bb = RESULTS[s]["train_loss_blk"].get(st, float("nan"))
        row.append(f"s{s}: cos={bi:.4f}(块均{bb:.4f}) vs 基线={BASE_TRACE[s].get(st, float('nan')):.4f}")
    print(f"[S53-TRAINCE] step={st} | " + " | ".join(row), flush=True)
for st in [x for x in (6000, 9000, 12000) if x in BASE_STEPS and x in GRID]:
    row = [f"s{s}: cos={CURVE[s][st]['ce']:.4f} vs 基线={BASE_EV[s][st]['ce']:.4f}"
           for s in SEEDS53]
    print(f"[S53-TESTCE] step={st} 测试整体CE | " + " | ".join(row), flush=True)

if not SMOKE:
    print(f"\n[S53] ===== ★饱和检查（R35）：lr_cosine 在 9000→12000 的 EM 增量 =====", flush=True)
    for s in SEEDS53:
        d_full = (CURVE[s][12000]["em_full"] - CURVE[s][9000]["em_full"]) * 100
        d_cl = (CURVE[s][12000]["em_clean"] - CURVE[s][9000]["em_clean"]) * 100
        db = (BASE_EV[s][12000]["em_full"] - BASE_EV[s][9000]["em_full"]) * 100
        print(f"[S53-SAT] seed={s}: cos 9000→12000 Δ全量={d_full:+.2f}pp / clean={d_cl:+.2f}pp "
              f"| 基线同段={db:+.2f}pp ⇒ "
              f"{'★未饱和（≥1pp）⇒ 结论条件性' if max(d_full, d_cl) >= 1.0 else '已平（<1pp）'}",
              flush=True)
    sat_mean = sum((CURVE[s][12000]["em_full"] - CURVE[s][9000]["em_full"]) * 100
                   for s in SEEDS53) / len(SEEDS53)
    print(f"[S53-SAT-VERDICT] 2 seed 均值 9000→12000 Δ={sat_mean:+.2f}pp ⇒ "
          f"{'★未饱和：本单元结论按 [条件性] 记（同 S38 基线亦未饱和）' if sat_mean >= 1.0 else '已饱和'}",
          flush=True)

# ============================================================================
# ⑧ 成本、ckpt 与总表
# ============================================================================
print(f"\n[S53] ===== 成本与 ckpt =====", flush=True)
for s in SEEDS53:
    r = RESULTS[s]
    for st in sorted(r["ckpt"]):
        print(f"[META] seed={s}: steps={r['steps']} ckpt={os.path.basename(r['ckpt'][st]['path'])} "
              f"sha256[:16]={r['ckpt'][st]['sha']}", flush=True)
    print(f"[META] seed={s}: 训练墙钟={r['wall_train']/60:.2f}min 本run={r['wall_total']/60:.2f}min",
          flush=True)
print(f"[S53-TABLE] step | " + " | ".join(f"s{s} EM全量/clean" for s in SEEDS53) +
      "  ‖ 基线@12000 全量/clean", flush=True)
for st in GRID:
    row = " | ".join(f"{CURVE[s][st]['em_full']*100:.2f}/{CURVE[s][st]['em_clean']*100:.2f}"
                     for s in SEEDS53)
    tail = ("  ‖ " + " / ".join(f"{BASE_EV[s][12000]['em_full']*100:.2f}/"
                                f"{BASE_EV[s][12000]['em_clean']*100:.2f}" for s in SEEDS53)
            if st == FIN else "")
    print(f"[S53-TABLE] {st} | {row}{tail}", flush=True)
print(f"[S53-TABLE-ACC] 数字每步正确率（样本级 exp(−CE)）", flush=True)
for st in GRID:
    row = " | ".join(f"{CURVE[s][st]['dig_acc']*100:.2f}%" for s in SEEDS53)
    b = " | ".join(f"{BASE_EV[s][12000]['dig_acc']*100:.2f}%" for s in SEEDS53) if st == FIN else ""
    print(f"[S53-TABLE-ACC] {st} | {row}" + (f"  ‖ 基线 {b}" if b else ""), flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"线程={torch.get_num_threads()}(R40) | D={D} FF={FF} NHEAD={NHEAD} batch={BATCH} "
      f"MAXLEN={MAXLEN} | lr_cosine=warmup{WARMUP}(0→{BASE_LR:g})+cos→0 @{STEPS53}步 | "
      f"EM口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | TEST sha16={TEST_SHA} 泄漏={LEAK}/{len(TEST)} "
      f"clean n={len(CLEAN_IDX)} | 地板(单/top5)={FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% | "
      f"参数={n_par/1e6:.3f}M | 总墙钟={(time.time()-T0)/60:.1f}min | run={len(RESULTS)}/{len(SEEDS53)}",
      flush=True)
print("[DONE] exit=0", flush=True)
sys.exit(0)
