#!/usr/bin/env python3
"""S38 · ★解锁一切的一枪：找到【真正的饱和点】—— 只加步数，其它一切不变。

为什么必须做：
  E6：@3000=33.88% → @6000=90.75%（自己声明「可能仍略欠训」）；
  S37：6/6 臂在 4500→6000 段仍在上升 ⇒ 未饱和（R35）⇒ 一切「必要性/有效性」结论不可下；
  ⇒ 至今【从未在真正的饱和点上测过任何东西】。

设计（★只加步数，不改结构/数据/超参/loss）：
  唯一臂 w1 = 全部 token 权重 1（= E6/S36 的 res 臂，逐字同构，见 S37 w1）；
  seed = 1234 / 5678；数据只用 add_3d（train4000 / test800，★不切 dev，固定末点）；
  步数做到 15000；★每 750 步取一次权重快照 ⇒ EM / 数字每步正确率的完整曲线（@750…@15000）。
  两个 ckpt 落盘点集：6000/9000/12000/15000 ⇒ logs/38_ckpt_w1_s<步数>_seed<s>.pt。
  等价性：快照只在 step 边界取（opt.step() 之后），训练轨迹只由 (seed, 数据顺序, 步数上界)
  决定 ⇒ 单条 15000 步轨迹的 @N 快照与「独立跑 N 步」逐位相同；
  @3000/@6000 复现 E6 即是对这条等价性的实测校验（差 >1pp 要点名）。

★ 判据（写死；本单元唯一产出目标）：
  1 饱和定义：最后两格（13500→15000）EM 增量 < 1pp，且【两 seed 都是】⇒ 判「已饱和」；
    另报每 750 步增量、以及「从哪一格起后续所有格增量都 <1pp」= 饱和步数；
  2 EM 上限 = 饱和点 EM（2 seed 均值 ± 跨度）；
  3 到饱和需要多少步（给全项目的口径）；
  4 若 @15000 仍未饱和 ⇒ 明写「仍未饱和，需要更大预算」+ 按末段趋势线性外推的 EM（标明外推）。

★ 口径（缺一即无效）：
  1 EM 一律 batch=1（R28）＋ batch=1 vs 16 逐字一致性自检；
  2 SEEN/UNSEEN 分层（train 对 test 的 (a,b) 泄漏 16/800）；
  3 地板用本桶：单答案 0.12% / top5 0.25%（不用 1.89%）；
  4 锚点：本次 @3000/@6000 必须复现 E6 的 37.25/31.13 与 85.62/74.38（差 ≤1pp）；
  5 TEST sha256[:16] == b34e7ea515203227；（R29 非零比例：本单元不做干预 ⇒ 不适用）
  6 数字每步正确率 = exp(−数字/算子 token CE)（样本级），每 750 步都报。

复用：exec stages/19_residual_scan.py 到 "# ③ 主循环" 之前（⇒ 同 tokenizer、同 add_3d
train4000/test800、同 build_batch/greedy_gen(EM_BATCH=1)/classify/parse_ans/r28_selfcheck）。
模型：与 S37 Cards37 逐字同构（emb → in_enc → thought(残差) → head，构造顺序相同 ⇒ 同 seed 同初始化）。
只允许写：本文件、logs/s38_sat.log、logs/38_*、/tmp；既有 stages/*.py 只读。
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

SMOKE38 = os.environ.get("S38_SMOKE") == "1"
if SMOKE38:
    os.environ["S19_SMOKE"] = "1"          # 让 S19 头只造小数据（冒烟）

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
BUCKET = "add_3d"
SEEDS38 = (1234, 5678)
STEPS38 = 60 if SMOKE38 else 15000
GRID = tuple(range(30, 61, 30)) if SMOKE38 else tuple(range(750, 15001, 750))
SAVE_ATS = (30, 60) if SMOKE38 else (6000, 9000, 12000, 15000)
N_EVAL = 60 if SMOKE38 else N_TEST                      # 800
CKPT_TMPL = f"{ROOT}/logs/38_ckpt_w1_s{{steps}}_seed{{seed}}.pt"
JSONL = "/tmp/38_results_smoke.jsonl" if SMOKE38 else f"{ROOT}/logs/38_results.jsonl"
TEST_SHA_EXPECT = "b34e7ea515203227"
LEAK_EXPECT = 16
# E6/S36 res 入库值：@3000 与 @6000（本次必须复现，差 ≤1pp）
ANCHOR_E6 = {1234: {3000: 0.3725, 6000: 0.8562},
             5678: {3000: 0.3113, 6000: 0.7438}}
ANCHOR_REF_4500 = {1234: 0.8162, 5678: 0.6088}          # 仅作形状参考

print(f"\n[S38] ★找真正的饱和点：桶={BUCKET} 唯一臂=w1(全 token 权重 1) 步数={STEPS38} "
      f"曲线格={GRID[0]}..{GRID[-1]}/{GRID[1]-GRID[0]}步 seeds={SEEDS38} | ★不切 dev、固定末点 | "
      f"d={D} ff={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} 思维卡残差 | "
      f"EM主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | smoke={SMOKE38}", flush=True)

# ============================================================================
# ② 数据核对（口径 2/3/4/5/6 + TEST 指纹）
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

sha_ok = TEST_SHA == TEST_SHA_EXPECT
leak_ok = LEAK == LEAK_EXPECT
print(f"[S38-DATA] {BUCKET}: train={len(TRAIN)} test={len(TEST)}(eval {len(TEST_EVAL)}) "
      f"文本零重叠 | 答案域={min(answer_values(BUCKET))}..{max(answer_values(BUCKET))}"
      f"({len(answer_values(BUCKET))}个) | ★不切 dev ⇒ 训练集={len(TRAIN)}", flush=True)
print(f"[S38-CHK6] TEST 指纹 sha256[:16]={TEST_SHA} vs 入库 {TEST_SHA_EXPECT} ⇒ "
      f"{'逐位一致 ✓' if sha_ok else '★不一致'} | (a,b) 泄漏对 train={LEAK}/{len(TEST)} vs "
      f"入库 {LEAK_EXPECT} ⇒ {'一致 ✓' if leak_ok else '★不一致'}", flush=True)
print(f"[S38-CHK4] SEEN={N_SEEN} / UNSEEN={N_UNSEEN}（train 对 test 的 (a,b) 泄漏 "
      f"{100.0*LEAK/len(TEST):.2f}%，终点点分层报 EM）", flush=True)
print(f"[S38-CHK5] add_3d 本桶地板：单答案 '{_top1}'(train {_top1n}/4000) 覆盖 test "
      f"{FLOOR_1*100:.2f}% | top5={_top5} 覆盖 test {FLOOR_5*100:.2f}% ⇒ ★本地板="
      f"{FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}%（不借用 1.89%）", flush=True)
print(f"[S38-CHK7] 干预面=无（唯一臂 w1 全权重 1）⇒ R29 Δ 非零比例不适用；本单元只测曲线",
      flush=True)
if not SMOKE38:
    assert sha_ok and leak_ok, "数据指纹不符 ⇒ 与 S19/S36/S37 不是同一数据集，本单元无效"


# ============================================================================
# ③ 模型：与 S37 Cards37 / S36 res 臂逐字同构（emb → in_enc → thought → head）
# ============================================================================
class Cards38(nn.Module):
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


def n_params_38():
    mdl = Cards38(D, FF)
    n = sum(p.numel() for p in mdl.parameters())
    del mdl
    return n


def train38(seed, train, steps, grid, save_ats):
    """★与 S37 train37 唯一差别：snapshot 格点更密 + 在 save_ats 直接落盘 ckpt。
    计算图 / 数据顺序 / 优化器 / 步进顺序逐字相同 ⇒ @N 快照 == 独立跑 N 步。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards38(D, FF).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0 = 0, 0, time.time()
    last, snaps, ckpts = 0.0, {}, {}
    gset = set(grid)
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(model, [train[i] for i in idx])          # ★w1：全 token 权重 1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step in gset:
            snaps[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if step in save_ats:
                ck = CKPT_TMPL.format(steps=step, seed=seed)
                torch.save(snaps[step], ck)
                ckpts[step] = ck
        if step % 1500 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train:w1] step={step}/{steps} loss={loss.item():.4f} "
                  f"elapsed={el:.0f}s s/step={(el - last) / (step % 1500 or 1500):.3f}", flush=True)
            last = el
    assert set(grid) <= set(snaps), "未取到全部曲线格点快照"
    return model, time.time() - t0, snaps, ckpts


@torch.no_grad()
def em_at(model, recs):
    txt = greedy_gen(model, recs, batch=EM_BATCH)          # ★R28 主口径 batch=1
    return [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, recs)], txt


@torch.no_grad()
def per_sample_ce(model, recs):
    """逐样本 (整体CE, 数字/算子 token 的 CE)；口径 = S19 ce_categories 的 samp_all / dig_samp。"""
    model.eval()
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
    model.train()
    return all_ce, dig_ce, dig_tok


def dig_acc(dig_ce):
    """样本级：a_i = exp(−CE_i)，主口径返回 (均值, SE, CE均值, CE的SE, n)。"""
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


def add_jsonl(obj):
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def frac(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def linfit(xs, ys):
    """最小二乘直线 y = a + b·x（x 用真实步数）。"""
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx if sxx else 0.0
    return my - b * mx, b


# ============================================================================
# ④ 主循环：唯一臂 w1，逐 seed 跑满 15000 步并在每 750 步评估
# ============================================================================
print(f"[S38-PARAMS] Cards38(res)={n_params_38()/1e6:.3f}M（= S36/S37 w1 臂，唯一变量是步数上限）",
      flush=True)
RESULTS: dict = {}
CURVE: dict = {}
open(JSONL, "w").close()

for seed in SEEDS38:
    t0 = time.time()
    print(f"\n[S38] ===== seed={seed} w1 步数={STEPS38} | train={len(TRAIN)} test={len(TEST_EVAL)} "
          f"曲线格={len(GRID)} 个 | 全部 token 权重 1（不改 loss 权重）=====", flush=True)
    model, wall_train, snaps, ckpts = train38(seed, TRAIN, STEPS38, GRID, SAVE_ATS)
    print(f"[S38] seed={seed} 训练结束 墙钟={wall_train/60:.2f}min | ckpt 落盘 "
          f"{sorted(ckpts)} ⇒ {[os.path.basename(v) for v in ckpts.values()]}", flush=True)

    ev = Cards38(D, FF).to(device)
    curve = {}
    for st in GRID:
        ev.load_state_dict({k: v.to(device) for k, v in snaps[st].items()})
        hits, _ = em_at(ev, TEST_EVAL)
        em = frac(hits)
        all_ce, dig_ce, dig_tok = per_sample_ce(ev, TEST_EVAL)
        acc, acc_se, dce, dce_se, dn = dig_acc(dig_ce)
        curve[st] = dict(em=em, em_se=math.sqrt(em * (1 - em) / len(TEST_EVAL)),
                         acc=acc, acc_se=acc_se, dig_ce=dce, ce=frac(all_ce),
                         hits=hits, acc_samp=[math.exp(-x) if x == x else float("nan")
                                              for x in dig_ce])
        print(f"  [S38-PT] seed={seed} step={st} EM={em*100:.2f}%±{curve[st]['em_se']*100:.2f} | "
              f"数字每步={acc*100:.2f}% | 数字CE={dce:.4f} | 整体CE={curve[st]['ce']:.4f}",
              flush=True)

    # ---- 终点点扩展口径：SEEN/UNSEEN + R28 自检 ----
    fin = GRID[-1]
    hits = curve[fin]["hits"]
    seen = [hits[i] for i, r in enumerate(TEST_EVAL) if r["seen"]]
    unseen = [hits[i] for i, r in enumerate(TEST_EVAL) if not r["seen"]]
    _r28 = r28_selfcheck(ev, TEST_EVAL, tag=f"[w1 s{fin} seed{seed}] ",
                         k=min(CHK_BATCH, len(TEST_EVAL)))
    # ---- 锚点格点 ----
    anchors = {}
    for st in (3000, 6000):
        if st in curve:
            anchors[st] = curve[st]["em"]
    del snaps, model, ev
    torch.cuda.empty_cache()

    RESULTS[seed] = dict(seed=seed, steps=STEPS38, wall_train=wall_train,
                         wall_total=time.time() - t0, n_eval=len(TEST_EVAL),
                         em_final=curve[fin]["em"], em_seen=frac(seen), em_unseen=frac(unseen),
                         n_seen=len(seen), n_unseen=len(unseen),
                         acc_final=curve[fin]["acc"], acc_se_final=curve[fin]["acc_se"],
                         ce_final=curve[fin]["ce"], r28_same=_r28["same"], r28_k=_r28["k"],
                         anchors=anchors,
                         ckpt={st: dict(path=ckpts[st], sha=sha16(ckpts[st])) for st in ckpts})
    CURVE[seed] = curve
    add_jsonl(dict(seed=seed, steps=STEPS38, wall_train=wall_train,
                   curve={str(st): {k: v for k, v in curve[st].items()
                                    if k not in ("hits", "acc_samp")} for st in GRID},
                   em_seen=frac(seen), em_unseen=frac(unseen), n_seen=len(seen),
                   n_unseen=len(unseen), r28_same=_r28["same"], r28_k=_r28["k"],
                   anchors=anchors,
                   ckpt={str(st): dict(path=ckpts[st], sha=sha16(ckpts[st])) for st in ckpts}))
    print(f"[S38-RUN] seed={seed}: EM@{fin}={curve[fin]['em']*100:.2f}%±"
          f"{curve[fin]['em_se']*100:.2f} SEEN={frac(seen)*100:.2f}%({len(seen)}) "
          f"UNSEEN={frac(unseen)*100:.2f}%({len(unseen)}) | 数字每步@{fin}="
          f"{curve[fin]['acc']*100:.2f}% | R28={_r28['same']}/{_r28['k']} | "
          f"合计墙钟={RESULTS[seed]['wall_total']/60:.2f}min", flush=True)

# ============================================================================
# ⑤ 口径自检 4：锚点（@3000 / @6000 复现 E6）
# ============================================================================
print("\n[S38] ===== ★口径自检 4：锚点 @3000 / @6000 必须复现 E6（差 ≤1pp）=====", flush=True)
ANCHOR_OK = True
for s in SEEDS38:
    for st in (3000, 6000):
        if st not in CURVE[s]:
            continue
        cur = CURVE[s][st]["em"]
        ref = ANCHOR_E6[s][st]
        dd = (cur - ref) * 100
        ok = abs(dd) <= 1.0
        ANCHOR_OK &= ok
        print(f"[S38-ANCHOR] w1 seed={s} @{st}: 本次={cur*100:.2f}% vs E6={ref*100:.2f}% ⇒ "
              f"Δ={dd:+.2f}pp {'✓ 复现' if ok else '★>1pp，点名'}", flush=True)
for s in SEEDS38:
    if 4500 in CURVE[s]:
        print(f"[S38-ANCHOR-REF] w1 seed={s} @4500: 本次={CURVE[s][4500]['em']*100:.2f}% vs "
              f"S37={ANCHOR_REF_4500[s]*100:.2f}%（形状参考，非写死判据）", flush=True)

# ============================================================================
# ⑥ 曲线表 + 饱和判定（写死）
# ============================================================================
W = GRID[1] - GRID[0]
print(f"\n[S38] ===== ★曲线表（唯一臂 w1，EM 主口径 batch={EM_BATCH}，每 {W} 步一格）=====",
      flush=True)
print("[S38-CURVE] step | " + " | ".join(f"s{s}" for s in SEEDS38), flush=True)
for st in GRID:
    row = " | ".join(f"{CURVE[s][st]['em']*100:.2f}%" for s in SEEDS38)
    print(f"[S38-CURVE] {st} | {row}", flush=True)
print(f"[S38-CURVE-ACC] 数字每步正确率（样本级 exp(−CE)）", flush=True)
for st in GRID:
    row = " | ".join(f"{CURVE[s][st]['acc']*100:.2f}%" for s in SEEDS38)
    print(f"[S38-CURVE-ACC] {st} | {row}", flush=True)
print(f"[S38-CURVE-INC] 每格 EM 增量（pp）", flush=True)
for i in range(1, len(GRID)):
    row = " | ".join(f"{(CURVE[s][GRID[i]]['em']-CURVE[s][GRID[i-1]]['em'])*100:+.2f}"
                     for s in SEEDS38)
    print(f"[S38-CURVE-INC] {GRID[i-1]}→{GRID[i]} | {row}", flush=True)

print(f"\n[S38] ===== ★饱和判定（写死：最后两格 {GRID[-2]}→{GRID[-1]} 增量 <1pp，两 seed 都是 "
      f"⇒ 已饱和）=====", flush=True)
SAT_STEP = {}
for s in SEEDS38:
    c = CURVE[s]
    last_inc = (c[GRID[-1]]["em"] - c[GRID[-2]]["em"]) * 100
    # 从哪一格起，后续所有格的增量都 <1pp
    sat_from = None
    for i in range(1, len(GRID)):
        if all((c[GRID[j]]["em"] - c[GRID[j-1]]["em"]) * 100 < 1.0 for j in range(i, len(GRID))):
            sat_from = GRID[i - 1]
            break
    SAT_STEP[s] = sat_from
    print(f"[S38-SAT] seed={s}: 末段（{GRID[-2]}→{GRID[-1]}）增量={last_inc:+.2f}pp ⇒ "
          f"{'已平（<1pp）✓' if last_inc < 1.0 else '★仍在升（≥1pp）'} | 从 @{sat_from} 起"
          f"后续每格增量都 <1pp", flush=True)
EM_CEIL = {s: CURVE[s][GRID[-1]]["em"] for s in SEEDS38}
ceil_mean = sum(EM_CEIL.values()) / len(EM_CEIL)
ceil_span = max(EM_CEIL.values()) - min(EM_CEIL.values())
both_flat = all((CURVE[s][GRID[-1]]["em"] - CURVE[s][GRID[-2]]["em"]) * 100 < 1.0
                for s in SEEDS38)
if both_flat:
    sat_step_proj = max(v for v in SAT_STEP.values() if v is not None)
    print(f"[S38-SAT-VERDICT] ★已饱和：EM 上限={ceil_mean*100:.2f}%（2 seed 均值）"
          f"跨度={ceil_span*100:.2f}pp（{min(EM_CEIL.values())*100:.2f}–"
          f"{max(EM_CEIL.values())*100:.2f}%，@{GRID[-1]} 实测）| 到饱和需要 ≤{sat_step_proj} 步"
          f"（口径：从该步起每 {W} 步增量都 <1pp）", flush=True)
else:
    _xs = list(GRID[-4:])
    ext_lines = []
    exs = []
    for s in SEEDS38:
        ys = [CURVE[s][st]["em"] for st in _xs]
        a, b = linfit(_xs, ys)
        ex_step = GRID[-1] + 6000
        ex = a + b * ex_step
        exs.append(ex)
        print(f"[S38-EXTRAP] seed={s}: 用末 4 格（{_xs[0]}..{_xs[-1]}）线性拟合 ⇒ 斜率="
              f"{b*750*100:+.2f}pp/{W}步 | ★外推 @{ex_step}={ex*100:.2f}%（外推、非实测）",
              flush=True)
    print(f"[S38-SAT-VERDICT] ★仍未饱和（@{GRID[-1]} 前两格增量 ≥1pp）：需要更大预算 | "
          f"按当前趋势外推 @{GRID[-1]+6000} ≈ {sum(exs)/len(exs)*100:.2f}%"
          f"（2 seed 均值，外推、非实测）| 当前 @{GRID[-1]} 实测={ceil_mean*100:.2f}%"
          f"（跨度 {min(EM_CEIL.values())*100:.2f}–{max(EM_CEIL.values())*100:.2f}%）", flush=True)

# ============================================================================
# ⑦ 成本与 ckpt
# ============================================================================
print("\n[S38] ===== 成本与 ckpt =====", flush=True)
for s in SEEDS38:
    r = RESULTS[s]
    for st in sorted(r["ckpt"]):
        print(f"[META] seed={s}: steps={r['steps']} ckpt={os.path.basename(r['ckpt'][st]['path'])} "
              f"sha256[:16]={r['ckpt'][st]['sha']}", flush=True)
    print(f"[META] seed={s}: 训练墙钟={r['wall_train']/60:.2f}min 本run={r['wall_total']/60:.2f}min",
          flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"D={D} FF={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} | EM口径=batch{EM_BATCH} "
      f"自检=batch{CHK_BATCH} | TEST sha16={TEST_SHA} 泄漏={LEAK}/{len(TEST)} | 地板(单/top5)="
      f"{FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% | 锚点复现={'是' if ANCHOR_OK else '否(点名)'} | "
      f"总墙钟={(time.time()-T0)/60:.1f}min | run 数={len(RESULTS)}/{len(SEEDS38)}", flush=True)
print(f"[S38-SUMMARY] 饱和={SAT_STEP} | EM上限={ceil_mean*100:.2f}% 跨度="
      f"{min(EM_CEIL.values())*100:.2f}–{max(EM_CEIL.values())*100:.2f}% @{GRID[-1]} | "
      f"锚点复现={'是' if ANCHOR_OK else '否'}", flush=True)
print("[DONE] exit=0", flush=True)
sys.exit(0)
