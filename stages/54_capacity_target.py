#!/usr/bin/env python3
"""S54 · ★容量轴（唯一未试的容量方向）：只把「目标卡」的 ff 512 → 1024，@12000 步。

依据（已入库）：
  S53（stages/53_lr_schedule.py，logs/s53_lrcos.log，提交 b6f2547）★已是新基线：
    warmup500 + cosine→0 @12000，两 seed 完全相同 全量 93.62% / clean 97.27%
    （基线 w1 常量 lr = 91.88/95.45；Δ 均值 +4.12pp，2 seed 同号 ≥2SE）；
    饱和检查通过（cos 9000→12000 = +0.62/+0.75pp < 1pp）；收敛左移一倍。
  ⇒ ★容量轴必须在 cosine 调度之上测，否则会把「调度收益」误记成「容量收益」。
  E9（stages/52_error_classify.py）：平台是【模型顶】（220 错例：解析 0.0% / 数值 100.0%，
    非记忆缺口，非难度墙）⇒ 容量是自然下一步。
  S13 试的是【d+ff 全局宽度】且只在 3000 步欠训区 ⇒ 与本次不同（本次只动目标卡 ff）。
  余量提示：clean 平台 97.27% ⇒ 名义余量 2.73pp ⇒ 结论用 clean 口径 + 2 seed 同号 ≥2SE 才判。

设计（★单变量：只扩目标卡 ff）：
  对照臂（基线）= S53 的 lr_cosine：结构 Cards53(d=128, ff=512, nhead=4, 1 层思维卡残差)，
    warmup500+cosine→0，@12000 ⇒ ★直接重载 logs/53_ckpt_lrcosine_s12000_seed{1234,5678}.pt
    （不重跑；重载后必须复现 93.62% / 97.27%）。
  容量臂 = 逐字同构，★只把 thought（目标卡）的 ff 512→1024；in_enc（输入卡）、d、nhead、
    层数、head、数据、batch、lr 调度全不动；同 warmup500+cosine→0。
  seed = 1234 / 5678 ⇒ 2 run；每 750 步内存快照；ckpt 落盘
    logs/54_ckpt_ff1024_s12000_seed<s>.pt（另存 9000 供饱和检查）。

★判据（写死）：
  1 主判据：Δ = EM(ff1024) − EM(S53 lr_cosine) @12000，逐 test 样本配对 ±SE，
    全量与 clean（去 30 条 `{expr}`，n=770）都报：
      Δ ≥ +2SE 且 2 seed 同号 ⇒ 扩目标卡容量有增益；|Δ| ≤ 1SE ⇒ 无增益；Δ ≤ −2SE 同号 ⇒ 有害。
  2 饱和检查（R35）：容量臂 9000→12000 增量 ≥1pp ⇒ 声明「未饱和、结论条件性」。
  3 两 seed 极差：容量臂 vs S53（全量/clean 都是 0.00pp）。
  4 训练 CE（单批 loss，与 S38/S53 同协议）与测试整体 CE @3000/6000/9000/12000
     ⇒ 看容量是否改变优化难度（E9 说普遍精度不足，容量若有效应表现为 CE 下降）。
  5 参数量对照：目标卡与总量（ff 512→1024 的增量）；增量 >5% 要说明「非同规模对照」。

★口径：EM 一律 batch=1（R28）+ batch=1 vs 16 自检；R39 显式 eval()；R40 batch=1 前向 4 线程；
  地板 0.12%/0.25%；SEEN/UNSEEN 分层；TEST sha256[:16] == b34e7ea515203227（R29：无干预 ⇒ 不适用）。

复用：exec stages/19_residual_scan.py 到 "# ③ 主循环" 之前（同 tokenizer、同 add_3d
  train4000/test800、同 build_batch/greedy_gen(EM_BATCH=1)/parse_ans/r28_selfcheck）。
对照曲线：读 logs/53_results.jsonl（既有日志，只读）取 S53 臂的逐格 EM/CE/训练 loss。
只允许写：本文件、logs/s54_cap.log、logs/54_*、/tmp；既有 stages/*.py 只读。
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
SRC53 = f"{ROOT}/stages/53_lr_schedule.py"

SMOKE = os.environ.get("S54_SMOKE") == "1"

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
#   并覆盖本单元的 S54_SMOKE ⇒ 必须在拷贝之后重新取一次，否则 SMOKE 永远由 S19 决定。
SMOKE = os.environ.get("S54_SMOKE") == "1"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402

torch.set_num_threads(4)                # ★R40：batch=1 逐样本前向统一 4 线程

T0 = time.time()
BUCKET = "add_3d"
SEEDS54 = (1234, 5678)
FF_IN = FF                              # 输入卡 ff = 512（不动）
FF_TGT = FF * 2                         # ★唯一变量：目标卡 ff 512 → 1024
STEPS54 = 90 if SMOKE else 12000
GRID = tuple(range(30, 91, 30)) if SMOKE else tuple(range(750, 12001, 750))
WARMUP = 5 if SMOKE else 500
BASE_LR = LR                            # 1e-3（与 S53/S38 同一个 base lr）
SAVE_ATS = (30, 60, 90) if SMOKE else (9000, 12000)
N_EVAL = 60 if SMOKE else N_TEST        # 800
CKPT_TMPL = (f"/tmp/54_ckpt_ff1024_s{{steps}}_seed{{seed}}.pt" if SMOKE
             else f"{ROOT}/logs/54_ckpt_ff1024_s{{steps}}_seed{{seed}}.pt")
BASE_CKPT = f"{ROOT}/logs/53_ckpt_lrcosine_s{{steps}}_seed{{seed}}.pt"
BASE_JSONL = f"{ROOT}/logs/53_results.jsonl"
JSONL = "/tmp/54_results_smoke.jsonl" if SMOKE else f"{ROOT}/logs/54_results.jsonl"
TEST_SHA_EXPECT = "b34e7ea515203227"
LEAK_EXPECT = 16
# S53 入库锚点（logs/s53_lrcos.log: 2 seed 相同）—— 重载 ckpt 必须复现，差 ≤0.2pp
ANCHOR_S53 = {1234: dict(full=0.9362, clean=0.9727), 5678: dict(full=0.9362, clean=0.9727)}


def lr_at(t: int, steps: int, warmup: int, base: float) -> float:
    """与 S53 逐字相同：warmup 线性 0→base，随后 cosine 衰减到 0（t = 0-based）。"""
    if t < warmup:
        return base * (t / warmup)
    prog = (t - warmup) / max(1, steps - warmup)
    prog = min(1.0, max(0.0, prog))
    return 0.5 * base * (1.0 + math.cos(math.pi * prog))


print(f"\n[S54] ★容量轴：只扩目标卡 ff {FF_IN}→{FF_TGT}（建立在 S53 的 warmup{WARMUP}+cosine→0 之上）"
      f"｜桶={BUCKET} 步数={STEPS54} 曲线格={GRID[0]}..{GRID[-1]}/{GRID[1]-GRID[0]}步 "
      f"seeds={SEEDS54} | d={D} nhead={NHEAD} 层数=1(思维卡残差) batch={BATCH} MAXLEN={MAXLEN} "
      f"base_lr={BASE_LR} | EM主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | "
      f"线程={torch.get_num_threads()}(R40) | smoke={SMOKE}", flush=True)
print(f"[S54-LR] 采样 lr：t=0/{WARMUP//2}/{WARMUP}/{STEPS54//4}/{STEPS54//2}/{3*STEPS54//4}/"
      f"{STEPS54-1} ⇒ " + " / ".join(f"{lr_at(t, STEPS54, WARMUP, BASE_LR):.3e}"
                                     for t in (0, WARMUP // 2, WARMUP, STEPS54 // 4,
                                               STEPS54 // 2, 3 * STEPS54 // 4, STEPS54 - 1)),
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
CLEAN_EVAL = [i for i in CLEAN_IDX if i < N_EVAL]      # 评估子集内的 clean 样本索引
CLEAN_SET_EVAL = set(CLEAN_EVAL)
_c_tr = Counter(r["gold"] for r in TRAIN)
_top1, _top1n = _c_tr.most_common(1)[0]
_top5 = [v for v, _ in _c_tr.most_common(5)]
FLOOR_1 = sum(1 for r in TEST if r["gold"] == _top1) / len(TEST)
FLOOR_5 = sum(1 for r in TEST if r["gold"] in _top5) / len(TEST)

sha_ok = TEST_SHA == TEST_SHA_EXPECT
leak_ok = LEAK == LEAK_EXPECT
clean_ok = (len(BAD_IDX), len(CLEAN_IDX)) == (30, 770)
print(f"[S54-DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)}(eval {len(TEST_EVAL)}) "
      f"答案域={min(answer_values(BUCKET))}..{max(answer_values(BUCKET))} | 不切 dev", flush=True)
print(f"[S54-CHK1] TEST 指纹 sha256[:16]={TEST_SHA} vs 入库 {TEST_SHA_EXPECT} ⇒ "
      f"{'逐位一致 ✓' if sha_ok else '★不一致'} | (a,b) 泄漏对={LEAK}/{len(TEST)} vs {LEAK_EXPECT} "
      f"⇒ {'一致 ✓' if leak_ok else '★不一致'}", flush=True)
print(f"[S54-CHK2] clean 口径：nl 含字面量 '{{expr}}' 的 bug 样本={len(BAD_IDX)}（全在 test）"
      f"⇒ clean n={len(CLEAN_IDX)} vs S42/S43 入库 30/770 ⇒ "
      f"{'一致 ✓' if clean_ok else '★不一致'}", flush=True)
print(f"[S54-CHK3] SEEN={N_SEEN} / UNSEEN={N_UNSEEN} | 地板：单答案 '{_top1}'={FLOOR_1*100:.2f}% "
      f"top5={_top5}={FLOOR_5*100:.2f}%（本地板，不借 1.89%）| 干预面=无 ⇒ R29 不适用", flush=True)
if not SMOKE:
    assert sha_ok and leak_ok and clean_ok, "数据口径不符 ⇒ 本单元无效"


# ============================================================================
# ③ 模型：与 S53 的 Cards53 逐字同构，★唯一差别 = 目标卡 ff 可独立指定
#    （属性名 emb / in_enc / thought / head / pe 与 S53 完全一致 ⇒ 可直接 load S53 ckpt）
# ============================================================================
class Cards54(nn.Module):
    """h = h + Think(h)（k=1），w1 = 全部 token 权重 1（纯 masked_ce）。
    ff_in = 输入卡宽度（不动）；ff_tgt = ★目标卡宽度（唯一变量，512→1024）。"""

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


def train54(seed, train, steps, grid, save_ats):
    """★与 S53 的 train53 逐字相同（同 lr 调度、同数据顺序、同 dropout 消耗），
    唯一差别 = 模型的目标卡 ff = ff_tgt。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards54(D, FF_IN, FF_TGT).to(device)
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
        for g in opt.param_groups:                     # ★唯一变量：写 lr（与 S53 同调度）
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


# ============================================================================
# ④ 参数量对照（★必须报：目标卡增量 + 总量变化 + 是否同规模）
# ============================================================================
def card_params(ff):
    """单张卡（TransformerEncoderLayer, d=D, nhead=NHEAD）的参数量。"""
    m = enc_layer(D, ff)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


N_BASE = sum(p.numel() for p in Cards54(D, FF_IN, FF_IN).parameters())
N_CAP = sum(p.numel() for p in Cards54(D, FF_IN, FF_TGT).parameters())
CARD_BASE, CARD_CAP = card_params(FF_IN), card_params(FF_TGT)
D_TGT = CARD_CAP - CARD_BASE
D_TOT = N_CAP - N_BASE
PCT_TGT = 100.0 * D_TGT / CARD_BASE
PCT_TOT = 100.0 * D_TOT / N_BASE
print(f"\n[S54-PARAMS] ★参数量对照（目标卡 = thought，输入卡 in_enc 不动）", flush=True)
print(f"[S54-PARAMS] 目标卡：ff {FF_IN}→{FF_TGT} ⇒ {CARD_BASE/1e6:.4f}M → {CARD_CAP/1e6:.4f}M "
      f"（Δ=+{D_TGT/1e6:.4f}M = +{PCT_TGT:.1f}%）", flush=True)
print(f"[S54-PARAMS] 总量：{N_BASE/1e6:.4f}M → {N_CAP/1e6:.4f}M （Δ=+{D_TOT/1e6:.4f}M = "
      f"+{PCT_TOT:.2f}%）| 输入卡 {card_params(FF_IN)/1e6:.4f}M 不动 | emb+head="
      f"{(N_BASE - 2 * CARD_BASE)/1e6:.4f}M 不动 ⇒ "
      f"{'★总增量 >5% ⇒ 严格说不是「同规模对照」，仅结构同构、只动一个旋钮' if PCT_TOT > 5 else '总增量 ≤5%'}"
      f" | S53 入库参数=2.502M vs 本次复算基线={N_BASE/1e6:.3f}M ⇒ "
      f"{'一致 ✓' if abs(N_BASE/1e6 - 2.502) < 0.002 else '★不一致'}", flush=True)

# 载入 S53 ckpt 前先验证 state_dict 键/形状逐位兼容（防止静默错载）
_ck_probe = torch.load(BASE_CKPT.format(steps=12000, seed=SEEDS54[0]),
                       map_location="cpu", weights_only=True)
_sd_probe = Cards54(D, FF_IN, FF_IN).state_dict()
_kk = sorted(set(_ck_probe) | set(_sd_probe))
_shape_bad = [k for k in _kk
              if k not in _ck_probe or k not in _sd_probe
              or tuple(_ck_probe[k].shape) != tuple(_sd_probe[k].shape)]
print(f"[S54-CHK4] S53 ckpt 键兼容自检：键数={len(_ck_probe)} vs Cards54(ff={FF_IN})={len(_sd_probe)} "
      f"| 形状不符={len(_shape_bad)}{_shape_bad} ⇒ "
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


print(f"\n[S54] ===== 对照臂（S53 lr_cosine ff={FF_IN}）= 重载 53 ckpt（★锚点复核）=====", flush=True)
ev = Cards54(D, FF_IN, FF_IN).to(device)
ev_cap = Cards54(D, FF_IN, FF_TGT).to(device)   # ★容量臂专属评估模型（目标卡 ff=1024）
BASE_EV, BASE_SHA = {}, {}
for s in SEEDS54:
    pp = BASE_CKPT.format(steps=12000, seed=s)      # 对照臂恒为 S53 @12000 的 ckpt
    bst = torch.load(pp, map_location="cpu", weights_only=True)
    BASE_SHA[s] = sha16(pp)
    BASE_EV[s] = eval_states(ev, {12000: bst}, [12000], f"BASE53 s{s}")
    r = BASE_EV[s][12000]
    if not SMOKE:
        ref = ANCHOR_S53[s]
        dd = (r["em_full"] - ref["full"]) * 100
        dc = (r["em_clean"] - ref["clean"]) * 100
        print(f"[S54-ANCHOR] 对照臂 s{s}@12000: 本次全量={r['em_full']*100:.2f}% vs S53 入库 "
              f"{ref['full']*100:.2f}% ⇒ Δ={dd:+.2f}pp | clean={r['em_clean']*100:.2f}% vs "
              f"{ref['clean']*100:.2f}% ⇒ Δ={dc:+.2f}pp ⇒ "
              f"{'复现 ✓' if abs(dd) <= 0.2 and abs(dc) <= 0.2 else '★不复现，点名'}", flush=True)

# S53 逐格曲线（从既有 JSONL 读，只读）⇒ 供 CE / EM 曲线对照
BASE53_CURVE, BASE53_TRAINLOSS = {}, {}
try:
    for ln in open(BASE_JSONL, encoding="utf-8"):
        d = json.loads(ln)
        BASE53_CURVE[d["seed"]] = {int(k): v for k, v in d["curve"].items()}
        BASE53_TRAINLOSS[d["seed"]] = {int(k): v for k, v in d["train_loss"].items()}
    assert all(s in BASE53_CURVE for s in SEEDS54), "53_results.jsonl 缺 seed"
    print(f"[S54-CHK5] 对照曲线来源 logs/53_results.jsonl：2 seed × {len(BASE53_CURVE[1234])} 格；"
          f"@12000 EM全量={BASE53_CURVE[1234][12000]['em_full']*100:.2f}%/"
          f"{BASE53_CURVE[5678][12000]['em_full']*100:.2f}% ⇒ 与锚点一致 ✓", flush=True)
except Exception as _e:                                          # noqa: BLE001
    BASE53_CURVE, BASE53_TRAINLOSS = {}, {}
    print(f"[S54-WARN] 读 logs/53_results.jsonl 失败({_e}) ⇒ 对照曲线仅用重载点的实测值", flush=True)


# ============================================================================
# ⑥ 容量臂：warmup+cosine（与 S53 同调度），只把目标卡 ff→{FF_TGT}
# ============================================================================
RESULTS, CURVE = {}, {}
if not SMOKE:
    open(JSONL, "w").close()
for s in SEEDS54:
    t0 = time.time()
    print(f"\n[S54] ===== seed={s} 容量臂 ff{FF_IN}→{FF_TGT} 步数={STEPS54} | "
          f"train={len(TRAIN)} test={len(TEST_EVAL)} 曲线格={len(GRID)} 个 =====", flush=True)
    model, wall_train, snaps, ckpts, inst, blk = train54(s, TRAIN, STEPS54, GRID, SAVE_ATS)
    print(f"[S54] seed={s} 训练结束 墙钟={wall_train/60:.2f}min | ckpt 落盘 {sorted(ckpts)}",
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
    RESULTS[s] = dict(seed=s, steps=STEPS54, wall_train=wall_train,
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
        f.write(json.dumps(dict(seed=s, steps=STEPS54, wall_train=wall_train,
                                curve={str(st): {k: v for k, v in curve[st].items()
                                                 if k != "hits"} for st in GRID},
                                train_loss={str(k): v for k, v in inst.items()},
                                train_loss_blk={str(k): v for k, v in blk.items()},
                                r28_same=_r28["same"], r28_k=_r28["k"],
                                ckpt={str(k): v for k, v in ckpt_meta.items()}),
                   ensure_ascii=False) + "\n")
    print(f"[S54-RUN] seed={s}: EM全量@{fin}={curve[fin]['em_full']*100:.2f}% "
          f"clean={curve[fin]['em_clean']*100:.2f}% SEEN={frac(seen)*100:.2f}%({len(seen)}) "
          f"UNSEEN={frac(unseen)*100:.2f}%({len(unseen)}) | 数字每步={curve[fin]['dig_acc']*100:.2f}% "
          f"| R28={_r28['same']}/{_r28['k']} | 合计墙钟={RESULTS[s]['wall_total']/60:.2f}min",
          flush=True)

FIN = GRID[-1]
BASE_FIN = 12000          # 对照臂恒为 S53 @12000 的 ckpt（smoke 下 FIN 会变小）

# ============================================================================
# ⑦ 主判据：Δ（配对）— 全量与 clean 两种口径 × 2 seed
# ============================================================================
print(f"\n[S54] ===== ★主判据：Δ = EM(ff{FF_TGT}) − EM(S53 lr_cosine ff{FF_IN})，逐样本配对 ±SE =====",
      flush=True)
DELTA = {}
for s in SEEDS54:
    a = CURVE[s][FIN]["hits"]
    b = BASE_EV[s][BASE_FIN]["hits"]
    for name, idx in (("全量", None), ("clean", CLEAN_EVAL)):
        m, se, n = paired_delta(a, b, idx)
        DELTA[(s, name)] = (m, se, n)
        print(f"[S54-DELTA] seed={s} {name}(n={n}): Δ={m*100:+.2f}pp ± {se*100:.2f}pp "
              f"({m/se if se else float('nan'):+.2f}SE) ⇒ {verdict(m, se)}", flush=True)
for name in ("全量", "clean"):
    ms = [DELTA[(s, name)][0] for s in SEEDS54]
    ses = [DELTA[(s, name)][1] for s in SEEDS54]
    same_sign = all(m > 0 for m in ms) or all(m < 0 for m in ms)
    pooled = sum(ms) / len(ms)
    pooled_se = math.sqrt(sum(x * x for x in ses)) / len(ses)
    print(f"[S54-VERDICT] {name}口径：2 seed 均值 Δ={pooled*100:+.2f}pp（"
          f"{ms[0]*100:+.2f}/{ms[1]*100:+.2f}）| 同号={'是' if same_sign else '否'} | "
          f"逐 seed 判定=" + " ; ".join(f"s{s}: {verdict(DELTA[(s, name)][0], DELTA[(s, name)][1])}"
                                        for s in SEEDS54), flush=True)

# ============================================================================
# ⑧ 两 seed 极差（S53 对照 vs 容量臂）与饱和检查（R35）
# ============================================================================
print(f"\n[S54] ===== ★两 seed 极差（@{FIN}）与饱和检查（R35）=====", flush=True)
for name, key in (("全量", "em_full"), ("clean", "em_clean")):
    bvals = [BASE_EV[s][BASE_FIN][key] for s in SEEDS54]
    cvals = [CURVE[s][FIN][key] for s in SEEDS54]
    bspan, cspan = max(bvals) - min(bvals), max(cvals) - min(cvals)
    print(f"[S54-SPREAD] {name}: S53={bspan*100:.2f}pp（{min(bvals)*100:.2f}–{max(bvals)*100:.2f}）"
          f" | 容量臂={cspan*100:.2f}pp（{min(cvals)*100:.2f}–{max(cvals)*100:.2f}） ⇒ "
          f"极差变化={(cspan-bspan)*100:+.2f}pp ⇒ "
          f"{'★容量重新引入方差（>1pp）' if cspan - bspan >= 0.01 else '未重新引入方差（<1pp）'}",
          flush=True)
if not SMOKE:
    sats = []
    for s in SEEDS54:
        d_full = (CURVE[s][12000]["em_full"] - CURVE[s][9000]["em_full"]) * 100
        d_cl = (CURVE[s][12000]["em_clean"] - CURVE[s][9000]["em_clean"]) * 100
        sats.append((d_full, d_cl))
        print(f"[S54-SAT] seed={s}: 容量臂 9000→12000 Δ全量={d_full:+.2f}pp / clean={d_cl:+.2f}pp "
              f"| S53 同段=+0.62/+0.65（s1234）、+0.75/+0.78（s5678）⇒ "
              f"{'★未饱和（≥1pp）⇒ 结论条件性' if max(d_full, d_cl) >= 1.0 else '已平（<1pp）'}",
              flush=True)
    sat_mean = sum(d for d, _ in sats) / len(sats)
    sat_mean_cl = sum(d for _, d in sats) / len(sats)
    print(f"[S54-SAT-VERDICT] 2 seed 均值 9000→12000 Δ全量={sat_mean:+.2f}pp / "
          f"clean={sat_mean_cl:+.2f}pp ⇒ "
          f"{'★未饱和：本单元结论按 [条件性] 记' if max(sat_mean, sat_mean_cl) >= 1.0 else '已饱和'}",
          flush=True)

# ============================================================================
# ⑨ CE 曲线（训练 CE 同协议 + 测试整体 CE）@3000/6000/9000/12000
# ============================================================================
CE_STEPS = [st for st in (3000, 6000, 9000, 12000) if st <= STEPS54 and st in GRID]
print(f"\n[S54] ===== ★训练 CE 曲线（单批 loss，与 S38/S53 同协议）+ 测试整体 CE =====", flush=True)
for st in CE_STEPS:
    row = []
    for s in SEEDS54:
        bi = RESULTS[s]["train_loss"].get(st, float("nan"))
        bb = RESULTS[s]["train_loss_blk"].get(st, float("nan"))
        b53 = BASE53_TRAINLOSS.get(s, {}).get(st, float("nan"))
        row.append(f"s{s}: cap={bi:.4f}(块均{bb:.4f}) vs S53={b53:.4f} Δ={bi-b53:+.4f}")
    print(f"[S54-TRAINCE] step={st} | " + " | ".join(row), flush=True)
for st in CE_STEPS:
    row = []
    for s in SEEDS54:
        cv = CURVE[s][st]["ce"]
        b53 = BASE53_CURVE.get(s, {}).get(st, {}).get("ce", float("nan"))
        row.append(f"s{s}: cap={cv:.4f} vs S53={b53:.4f} Δ={cv-b53:+.4f}")
    print(f"[S54-TESTCE] step={st} 测试整体CE | " + " | ".join(row), flush=True)
print(f"[S54-CE-VERDICT] 训练 CE @12000：" + " ; ".join(
    f"s{s}: cap={RESULTS[s]['train_loss'].get(12000, float('nan')):.4f} vs "
    f"S53={BASE53_TRAINLOSS.get(s, {}).get(12000, float('nan')):.4f}" for s in SEEDS54)
    + " ⇒ 容量改变优化难度与否看 Δ 符号/量级", flush=True)


# ============================================================================
# ⑩ 成本、ckpt 与总表
# ============================================================================
print(f"\n[S54] ===== 成本与 ckpt =====", flush=True)
for s in SEEDS54:
    r = RESULTS[s]
    for st in sorted(r["ckpt"]):
        print(f"[META] seed={s}: steps={r['steps']} ckpt={os.path.basename(r['ckpt'][st]['path'])} "
              f"sha256[:16]={r['ckpt'][st]['sha']}", flush=True)
    print(f"[META] seed={s}: 训练墙钟={r['wall_train']/60:.2f}min 本run={r['wall_total']/60:.2f}min",
          flush=True)
print(f"[S54-TABLE] step | " + " | ".join(f"s{s} EM全量/clean" for s in SEEDS54) +
      "  ‖ S53@12000 全量/clean", flush=True)
for st in GRID:
    row = " | ".join(f"{CURVE[s][st]['em_full']*100:.2f}/{CURVE[s][st]['em_clean']*100:.2f}"
                     for s in SEEDS54)
    tail = ("  ‖ " + " / ".join(f"{BASE_EV[s][BASE_FIN]['em_full']*100:.2f}/"
                                f"{BASE_EV[s][BASE_FIN]['em_clean']*100:.2f}" for s in SEEDS54)
            if st == FIN else "")
    print(f"[S54-TABLE] {st} | {row}{tail}", flush=True)
print(f"[S54-EM@12000] " + " | ".join(
    f"s{s}: 容量臂 全量={CURVE[s][FIN]['em_full']*100:.2f}% clean={CURVE[s][FIN]['em_clean']*100:.2f}% "
    f"数字每步={CURVE[s][FIN]['dig_acc']*100:.2f}%  ‖ S53 全量={BASE_EV[s][BASE_FIN]['em_full']*100:.2f}% "
    f"clean={BASE_EV[s][BASE_FIN]['em_clean']*100:.2f}% 数字每步={BASE_EV[s][BASE_FIN]['dig_acc']*100:.2f}%"
    for s in SEEDS54), flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"线程={torch.get_num_threads()}(R40) | D={D} NHEAD={NHEAD} 层数=1 batch={BATCH} "
      f"MAXLEN={MAXLEN} | ★唯一变量=目标卡 ff {FF_IN}→{FF_TGT} | lr=warmup{WARMUP}"
      f"(0→{BASE_LR:g})+cos→0 @{STEPS54}步（与 S53 同） | EM口径=batch{EM_BATCH} "
      f"自检=batch{CHK_BATCH} | TEST sha16={TEST_SHA} 泄漏={LEAK}/{len(TEST)} clean n={len(CLEAN_IDX)} "
      f"| 地板(单/top5)={FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% | 参数 基线={N_BASE/1e6:.3f}M → "
      f"容量={N_CAP/1e6:.3f}M（+{PCT_TOT:.2f}%）| 总墙钟={(time.time()-T0)/60:.1f}min | "
      f"run={len(RESULTS)}/{len(SEEDS54)}", flush=True)
print("[DONE] exit=0", flush=True)
sys.exit(0)
