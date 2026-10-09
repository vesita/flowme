#!/usr/bin/env python3
"""S64 · S63 后续 ★ 位置绑定 @24000 的残留增益 +0.94pp：真残留还是种子噪声（加 2 个 seed）。

S63（stages/63_binding_24k.py，本地）已钉死饱和（@21000→@24000 两臂 4/4 <1pp），
但残留 Δ@24000 = +0.94pp ± 0.56（t=1.67，n=2，全量）落中间带（1SE=0.56 < 0.94 < 2SE=1.12），
两 seed 同号为正（+0.37/+1.50）⇒ 需加 seed 才能判「是否真残留一个小增益」。

本单元唯一新增变量 = seed：新增 2 个 seed（7890、2468），与 S63 的 9012/3456 合并成 n=4；
其余（臂定义 / 实现 / 数据 / 超参 / lr / batch / 步数 / 快照格点）与 S63 逐字相同。

实现：本文件 exec stages/50_binding_at_saturation.py 的【前 ④ 段】（切在 "# ⑤ 主循环" 之前），
  逐字复用 BoundMHA / Cards50 / train50 / em_eval / swap_sensitivity / r28_selfcheck / 参数自检；
  随后只把 CKPT_TMPL 改指 logs/64_*（★绝不覆盖 logs/50_*、logs/62_*、logs/63_*）。
  S63 的既有结果【不重训】：现评 logs/63_ckpt_{base,bind}_s24000_seed{9012,3456}.pt 取 @24000，
  其 @15000/@21000 格点从 logs/63_curve_*.json 读入（同代码同 batch=1 口径）⇒ 合成 n=4。

判据（写死，同任务书）：
  1 主判据：Δ@24000 = EM(bind) − EM(base)，n=4 逐 seed 配对（合并 S63 的 9012/3456），全量与 clean 都报：
    Δ ≥ +2SE 且 4/4 同号为正 ⇒ 饱和点仍有小增益（"绑定 = 加速收敛 + 小增益"）；
    |Δ| ≤ 1SE ⇒ 纯加速收敛（"绑定 = 加速收敛"，与残差同型）；
    否则 ⇒ 仍不可判（报 n 与 SE，并说明还需多少 seed）。
  2 ★必报 n=2（S63）与 n=4 并列（SE 是否收窄、均值是否稳定）。
  3 ★逐 seed 明细（4 个 seed 的 base/bind/Δ）。
  4 ★机制复核：4 个 seed 的 @15000 与 @24000 的 Δ + base 在 15000→24000 的增量
    ⇒ 验证「base 后来追上」在 2 个新 seed 上是否重现。
  5 ★饱和检查（R35）：每个新 seed 报 @21000→@24000 增量（两臂）⇒ 任何一格 ≥1pp 点名。
  6 ★(a) 置换敏感性：每个新 ckpt 报 max|Δlogits|（对照 base）⇒ 绑定生效。

口径：旧数据（TEST sha256[:16] = b34e7ea515203227，与 P3b/P3c/S63 同版）；batch=1（R28）；
  生成前显式 eval()（R39）；torch.set_num_threads(4)（R40）；地板 / SEEN / UNSEEN 照报。
  ★锚点复核：现评 S63 的 @24000 ckpt，期望 base9012=92.00 / bind9012=92.38 / base3456=91.00 /
    bind3456=92.50（全量，Δ>0.5pp 点名）。
ckpt：logs/64_ckpt_{base|bind}_s{15000,24000}_seed{7890,2468}.pt（报 sha256[:16]）；
  曲线落 logs/64_curve_{arm}_seed<s>.json；结果落 logs/64_results.jsonl。可续跑（ckpt 在则复用）。
只允许写：本文件（新建）、logs/64_*、logs/s64_bind_n4.log、/tmp；既有 stages/*.py 只读。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from statistics import median

ROOT = "/home/vesita/coding/my/flowme"
SRC50 = f"{ROOT}/stages/50_binding_at_saturation.py"

SMOKE64 = os.environ.get("S64_SMOKE") == "1"
if SMOKE64:
    os.environ["S50_SMOKE"] = "1"          # 必须在 exec 之前：S50 前缀随之切小数据/短训

# ============================================================================
# ① 逐字复用 S50 前 ④ 段（其内部又 exec S19 前半段：同 tokenizer/同 add_3d 数据/
#    同 build_batch/masked_ce/greedy_gen/parse_ans/r28_selfcheck/sin_pe/enc_layer）
# ============================================================================
_src50 = open(SRC50, encoding="utf-8").read()
assert _src50.count("# ⑤ 主循环") == 1
_cut50 = _src50.index("# ⑤ 主循环")
G: dict = {"__name__": "s64_s50head", "__file__": SRC50}
exec(compile(_src50[:_cut50], SRC50, "exec"), G)          # noqa: S102
# ★ G 就是 S50 前缀里所有函数/类的 __globals__：改 G 才能改到 train50 实际用的常量。

import torch  # noqa: E402
torch.set_num_threads(4)                                  # ★R40：batch=1 逐样本前向统一 4 线程

device = G["device"]
D, FF, NHEAD = G["D"], G["FF"], G["NHEAD"]
LR, BATCH, MAXLEN = G["LR"], G["BATCH"], G["MAXLEN"]
EM_BATCH, CHK_BATCH = G["EM_BATCH"], G["CHK_BATCH"]
N_EVAL = G["N_EVAL"]
TEST, TRAIN = G["TEST"], G["TRAIN"]
CLEAN_IDX = G["CLEAN_IDX"]
Cards50, train50 = G["Cards50"], G["train50"]
em_eval, swap_sensitivity = G["em_eval"], G["swap_sensitivity"]
r28_selfcheck, sha16, mean_se = G["r28_selfcheck"], G["sha16"], G["mean_se"]

assert device == "cuda", "★本单元必须 cuda"
assert EM_BATCH == 1, "★主口径必须 batch=1（R28）"

# ---- 只改 ckpt 落点（S50 前缀默认写 logs/50_*，本单元绝不覆盖它们）----
G["CKPT_TMPL"] = ("/tmp/64_ckpt_{arm}_s{steps}_seed{seed}.pt" if SMOKE64
                  else f"{ROOT}/logs/64_ckpt_{{arm}}_s{{steps}}_seed{{seed}}.pt")
JSONL64 = "/tmp/64_results_smoke.jsonl" if SMOKE64 else f"{ROOT}/logs/64_results.jsonl"
CURVE_TMPL = "/tmp/64_curve_{arm}_seed{seed}.json" if SMOKE64 else f"{ROOT}/logs/64_curve_{{arm}}_seed{{seed}}.json"

T0 = time.time()
BUCKET = "add_3d"
ARMS64 = ("base", "bind")

if SMOKE64:
    STEPS64, STEP_SAVE64 = 60, 30
    GRID64 = (30, 60)
    SEEDS64 = (7890,)
else:
    STEPS64, STEP_SAVE64 = 24000, 750
    GRID64 = tuple(range(3000, STEPS64 + 1, 1500))        # 3000,4500,…,24000（含 15000/21000/24000）
    _env_seeds = os.environ.get("S64_SEEDS", "").strip()
    SEEDS64 = tuple(int(x) for x in _env_seeds.split(",") if x.strip()) if _env_seeds else (7890, 2468)
SNAP64 = tuple(range(STEP_SAVE64, STEPS64 + 1, STEP_SAVE64))
assert set(GRID64) <= set(SNAP64), "报告格点必须落在快照格点上"

# ---- S63（63）入库锚：既有 4 个 @24000 ckpt + 曲线（本单元只现评 / 只读，绝不重训）----
S63_SEEDS = (9012, 3456)
S63_CKPT = {(arm, s): f"{ROOT}/logs/63_ckpt_{arm}_s24000_seed{s}.pt"
            for s in S63_SEEDS for arm in ARMS64}
S63_CURVE = {(arm, s): f"{ROOT}/logs/63_curve_{arm}_seed{s}.json"
             for s in S63_SEEDS for arm in ARMS64}
S63_EM = {("base", 9012): (92.00, 95.58), ("bind", 9012): (92.38, 95.97),
          ("base", 3456): (91.00, 94.55), ("bind", 3456): (92.50, 96.10)}
S63_SHA = {("base", 9012): "481d622638f661e4", ("bind", 9012): "df4edfdbaaab8234",
           ("base", 3456): "3826899cb3955f69", ("bind", 3456): "1d7e2277db10ce45"}
ALL_SEEDS = tuple(S63_SEEDS) + tuple(SEEDS64)

print(f"\n[S64] ★S63 后续：只加 seed {SEEDS64}（合并 S63 的 {S63_SEEDS} ⇒ n={len(ALL_SEEDS)}）："
      f"臂={ARMS64} 步数={STEPS64}（每 {STEP_SAVE64} 步快照，EM 格点={GRID64}）| d={D} ff={FF} "
      f"nhead={NHEAD} lr={LR} batch={BATCH} maxlen={MAXLEN} | 数据=旧版 add_3d | EM口径=batch{EM_BATCH} | "
      f"threads={torch.get_num_threads()} | smoke={SMOKE64}", flush=True)
print(f"[S64-DEV] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__}", flush=True)

# ============================================================================
# ② 数据版与口径复核（旧数据 TEST sha / 泄漏 / clean n / 地板）
# ============================================================================
print(f"[S64-CHK6] 数据版=【旧】数据（与 P3b/P3c/S63 同版）| TEST sha256[:16]={G['TEST_SHA']} vs "
      f"入库 {G['TEST_SHA_EXPECT']} ⇒ "
      f"{'逐位一致 ✓' if G['TEST_SHA'] == G['TEST_SHA_EXPECT'] else '★不一致'} | "
      f"(a,b) 泄漏={G['LEAK']}/{len(TEST)} vs {G['LEAK_EXPECT']} | clean n={len(CLEAN_IDX)}/{N_EVAL}",
      flush=True)
print(f"[S64-CHK4] SEEN={G['N_SEEN']} / UNSEEN={G['N_UNSEEN']} | 地板(单答案/top5)="
      f"{G['FLOOR_1']*100:.2f}%/{G['FLOOR_5']*100:.2f}%", flush=True)

# ============================================================================
# ③ 参数量 + 基线权重逐位相同（复用 S50 前缀已算好的自检）
# ============================================================================
P_BASE, P_BIND = G["P_BASE"], G["P_BIND"]
PARAM_DIFF = abs(P_BIND - P_BASE) / P_BASE
print(f"[S64-PARAMS] 参数量：base={P_BASE} bind={P_BIND} | ①vs② 绝对差={P_BIND-P_BASE} "
      f"相对差={PARAM_DIFF*100:.4f}% ⇒ {'≤5% ✓' if PARAM_DIFF <= 0.05 else '★>5% 点名'}",
      flush=True)
print(f"[S64-CHK7] ①② 基线权重（emb/in_enc/thought 基线部分/head/pe）逐位相同："
      f"max|Δθ|={G['_wbin']['bind']:.3e} ⇒ "
      f"{'max|Δθ|=0 ✓' if G['_wbin']['bind'] == 0.0 else '★非零，点名'}", flush=True)


# ============================================================================
# ④ 公共工具
# ============================================================================
def report_hits(hits):
    full = sum(hits) / len(hits)
    clean = sum(hits[i] for i in CLEAN_IDX) / len(CLEAN_IDX)
    seen = [hits[i] for i, r in enumerate(TEST[:N_EVAL]) if r["seen"]]
    uns = [hits[i] for i, r in enumerate(TEST[:N_EVAL]) if not r["seen"]]
    return (full, clean, (sum(seen) / len(seen)) if seen else float("nan"),
            (sum(uns) / len(uns)) if uns else float("nan"))


_PAIRS = None


def swap_pairs():
    global _PAIRS
    if _PAIRS is None:
        lo_s = min(len(r["p"]) for r in TEST[:32])
        ps = [(0, 1), (2, 5), (1, lo_s - 3), (3, lo_s - 4), (4, lo_s // 2), (0, lo_s - 2)]
        _PAIRS = sorted({(min(i, j), max(i, j)) for i, j in ps}), lo_s
    return _PAIRS


def quick_swap(model):
    pairs, lo_s = swap_pairs()
    sw, lsc = swap_sensitivity(model, TEST[:32], pairs)
    best = max(sw.values(), key=lambda v: v["mx"])
    return dict(best), lsc, pairs, lo_s


def swap_verdict(mx):
    return ("★绑定生效（≥1e-3）" if mx >= 1e-3 else
            ("绑定生效但弱（≥1e-4）" if mx >= 1e-4 else "★未生效（fp 噪声量级）"))


# ============================================================================
# ④b 判定：Δ = bind − base 逐 seed 配对（样本 sd/√n；t = Δ/SE）
# ============================================================================
def verdict_of(m, se):
    return m / se if se > 0 else float("nan")


def judge(ds, m, se):
    both_pos, both_neg = all(d > 0 for d in ds), all(d < 0 for d in ds)
    if m >= 2 * se and both_pos:
        return "★提升（Δ≥+2SE 且全同号）"
    if abs(m) <= se:
        return "无影响（|Δ|≤1SE）"
    if m <= -2 * se and both_neg:
        return "★有害（Δ≤−2SE 且全同号）"
    return "仍不可判（中间带）"


def need_more_seeds(m, se):
    """若想以 2SE 判「提升」暂需的 n（样本 sd 不变、SE=sd/√n 的估计；标注为外推）。"""
    sd = se * math.sqrt(N_LAST)
    if m <= 0 or sd <= 0:
        return float("nan")
    n = (2 * sd / m) ** 2
    return n


def emit_verdict(seeds, tag):
    """★主判据：Δ ≥ +2SE 且全同号 ⇒ 提升；|Δ| ≤ 1SE ⇒ 无影响；否则仍不可判。"""
    global N_LAST
    for mode, tagm in (("full", "全量"), ("clean", "clean")):
        ds = [ALL[(s, "bind")][mode] - ALL[(s, "base")][mode] for s in seeds]
        m, se = mean_se(ds)
        t = verdict_of(m, se)
        v = judge(ds, m, se)
        N_LAST = len(seeds)
        extra = ""
        if v.startswith("仍不可判") and se > 0:
            n_need = need_more_seeds(m, se)
            if n_need == n_need:
                extra = f" | 以 2SE 判定约需 n≈{math.ceil(n_need)}（外推）"
        print(f"[S64-N@{tag}] n={len(seeds)} {tagm}: Δ(bind−base)={m*100:+.2f}pp ± {se*100:.2f} "
              f"(t={t:.2f}) 逐seed={['%+.2f' % (d*100) for d in ds]} ⇒ {v}{extra}", flush=True)


N_LAST = 1

# ============================================================================
# ⑤ 锚点复核：现评 S63 的 4 个 @24000 ckpt（独立于本次训练）+ 读 S63 曲线
# ============================================================================
ANCHOR63: dict = {}
CURVE63: dict = {}
print("\n[S64] ===== ★锚点复核：现评 S63(63) @24000 ckpt（独立于本次训练）=====", flush=True)
for s in S63_SEEDS:
    for arm in ARMS64:
        p = S63_CKPT[(arm, s)]
        assert os.path.exists(p), f"★锚点 ckpt 不存在：{p}"
        m = Cards50(D, FF, arm).to(device)
        m.load_state_dict({k: v.to(device) for k, v in torch.load(p, map_location="cpu").items()})
        m.eval()                                        # ★R39
        h = em_eval(m, TEST[:N_EVAL])
        f, c, sn, un = report_hits(h)
        ah, ac = S63_EM[(arm, s)]
        d_f, d_c = f * 100 - ah, c * 100 - ac
        w, lsc, _, _ = quick_swap(m)
        ANCHOR63[(arm, s)] = dict(full=f, clean=c, seen=sn, unseen=un, sha=sha16(p),
                                  swap_best=w, lsc=lsc)
        print(f"[S64-ANCHOR] {arm} seed={s} @24000: 全量={f*100:.2f}% vs S63 入库 {ah:.2f}%（Δ={d_f:+.2f}pp）"
              f" | clean={c*100:.2f}% vs {ac:.2f}%（Δ={d_c:+.2f}pp）⇒ "
              f"{'复现 ✓（≤0.5pp）' if max(abs(d_f), abs(d_c)) <= 0.5 else '★>0.5pp，点名'} | "
              f"ckpt sha16={ANCHOR63[(arm, s)]['sha']} vs 入库 {S63_SHA[(arm, s)]} ⇒ "
              f"{'一致 ✓' if ANCHOR63[(arm, s)]['sha'] == S63_SHA[(arm, s)] else '★不一致'} | "
              f"max|Δlogits|={w['mx']:.3e}", flush=True)
        del m
        torch.cuda.empty_cache()
        cp = S63_CURVE[(arm, s)]
        assert os.path.exists(cp), f"★S63 曲线 json 不存在：{cp}"
        CURVE63[(arm, s)] = json.load(open(cp, encoding="utf-8"))["curve"]

# ============================================================================
# ⑥ 单 (seed, arm)：训练到 24000 + 逐格评测；ckpt 已在则复用
# ============================================================================
RES: dict = {}


def eval_grid(seed, arm, snaps, curve):
    ev = Cards50(D, FF, arm).to(device)
    fin = None
    for st in GRID64:
        ev.load_state_dict({k: v.to(device) for k, v in snaps[st].items()})
        ev.eval()                                       # ★R39
        hits = em_eval(ev, TEST[:N_EVAL])
        f, c, sn, un = report_hits(hits)
        curve[str(st)] = dict(full=f, clean=c, seen=sn, unseen=un)
        print(f"  [S64-PT] {arm} seed={seed} step={st} EM全量={f*100:.2f}% clean={c*100:.2f}% "
              f"SEEN={sn*100:.2f}% UNSEEN={un*100:.2f}%", flush=True)
        if st == GRID64[-1]:
            fin = dict(full=f, clean=c, seen=sn, unseen=un)
    ev.load_state_dict({k: v.to(device) for k, v in snaps[GRID64[-1]].items()})
    ev.eval()                                           # ★R39
    r28 = r28_selfcheck(ev, TEST[:N_EVAL], tag=f"[S64 {arm} seed{seed}] ",
                        k=min(CHK_BATCH, N_EVAL))
    w, lsc, _, _ = quick_swap(ev)
    del ev
    torch.cuda.empty_cache()
    return fin, r28, w, lsc


def run_train(seed, arm):
    t0 = time.time()
    ck, wall_train, snaps = train50(seed, arm, TRAIN, STEPS64, SNAP64, STEPS64)
    if not SMOKE64 and 15000 in snaps:
        ck15 = G["CKPT_TMPL"].format(arm=arm, steps=15000, seed=seed)
        torch.save(snaps[15000], ck15)
    curve: dict = {}
    fin, r28, w, lsc = eval_grid(seed, arm, snaps, curve)
    del snaps
    torch.cuda.empty_cache()
    with open(CURVE_TMPL.format(arm=arm, seed=seed), "w", encoding="utf-8") as f:
        json.dump(dict(arm=arm, seed=seed, steps=STEPS64, grid=list(GRID64), curve=curve),
                  f, ensure_ascii=False)
    return dict(**fin, curve=curve, wall_train=wall_train, wall_total=time.time() - t0,
                ckpt=ck, ckpt_sha=sha16(ck), reused=False, src="新训", r28=r28,
                swap_best=w, lsc=lsc)


def run_load(seed, arm):
    """复用已训好的 logs/64_ckpt_*_s24000_*：只评测；曲线从 json 读（若有）。"""
    p = G["CKPT_TMPL"].format(arm=arm, steps=STEPS64, seed=seed)
    t0 = time.time()
    assert os.path.exists(p), f"★ckpt 不存在：{p}"
    m = Cards50(D, FF, arm).to(device)
    m.load_state_dict({k: v.to(device) for k, v in torch.load(p, map_location="cpu").items()})
    m.eval()                                            # ★R39
    hits = em_eval(m, TEST[:N_EVAL])
    f, c, sn, un = report_hits(hits)
    m.eval()
    r28 = r28_selfcheck(m, TEST[:N_EVAL], tag=f"[S64 复用 {arm} seed{seed}] ", k=min(CHK_BATCH, N_EVAL))
    w, lsc, _, _ = quick_swap(m)
    del m
    torch.cuda.empty_cache()
    cp = CURVE_TMPL.format(arm=arm, seed=seed)
    curve = json.load(open(cp, encoding="utf-8"))["curve"] if os.path.exists(cp) else {}
    return dict(full=f, clean=c, seen=sn, unseen=un, curve=curve, wall_train=float("nan"),
                wall_total=time.time() - t0, ckpt=p, ckpt_sha=sha16(p), reused=True, src="复用64",
                r28=r28, swap_best=w, lsc=lsc)


def run_auto(seed, arm):
    p = G["CKPT_TMPL"].format(arm=arm, steps=STEPS64, seed=seed)
    if not SMOKE64 and os.path.exists(p):
        return run_load(seed, arm)
    return run_train(seed, arm)


def dump(seed, arm, r):
    with open(JSONL64, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(stage="64", bucket=BUCKET, seed=int(seed), arm=arm, steps=STEPS64,
                                full=r["full"], clean=r["clean"], seen=r["seen"], unseen=r["unseen"],
                                curve=r["curve"], reused=r["reused"], src=r.get("src", "?"),
                                ckpt=os.path.basename(r["ckpt"]), ckpt_sha=r["ckpt_sha"],
                                wall_train=r["wall_train"], r28=r["r28"], swap_best=r["swap_best"]),
                         ensure_ascii=False) + "\n")


# ============================================================================
# ⑦ 主循环：新 seed 7890 两臂 → 新 seed 2468 两臂
#   ★不打印 n=1 的中间统计（S63 出现过 SE=0 的退化假阳）；判定只在 §⑧ 末尾做。
# ============================================================================
open(JSONL64, "w").close()
for seed in SEEDS64:
    for arm in ARMS64:
        r = run_auto(seed, arm)
        RES[(seed, arm)] = r
        dump(seed, arm, r)
        if r["reused"]:
            print(f"[S64-RUN] 复用 {arm} seed={seed}: 全量={r['full']*100:.2f}% clean={r['clean']*100:.2f}% "
                  f"| R28={r['r28']['same']}/{r['r28']['k']} | 评测墙钟={r['wall_total']/60:.2f}min", flush=True)
        else:
            print(f"[S64-RUN] 新 {arm} seed={seed}: EM@{STEPS64}全量={r['full']*100:.2f}% "
                  f"clean={r['clean']*100:.2f}% SEEN={r['seen']*100:.2f}% UNSEEN={r['unseen']*100:.2f}% "
                  f"| R28={r['r28']['same']}/{r['r28']['k']} | 训练墙钟={r['wall_train']/60:.2f}min "
                  f"本 run={r['wall_total']/60:.2f}min | ckpt={os.path.basename(r['ckpt'])} "
                  f"sha16={r['ckpt_sha']}", flush=True)

# ---- 合并 n=4 视图：老 seed 用现评 @24000 + S63 曲线；新 seed 用本单元 ----
ALL: dict = {}


def _add(seed, arm, full, clean, seen, unseen, curve, src):
    ALL[(seed, arm)] = dict(full=full, clean=clean, seen=seen, unseen=unseen, curve=curve, src=src)


for s in S63_SEEDS:
    for arm in ARMS64:
        A = ANCHOR63[(arm, s)]
        _add(s, arm, A["full"], A["clean"], A["seen"], A["unseen"], CURVE63[(arm, s)], "S63现评")
for s in SEEDS64:
    for arm in ARMS64:
        r = RES[(s, arm)]
        _add(s, arm, r["full"], r["clean"], r["seen"], r["unseen"], r["curve"], r["src"])

# ============================================================================
# ⑧ 明细表 + 判定（n=2 vs n=4）+ 机制复核 + 饱和检查 + 置换敏感性
# ============================================================================
print(f"\n[S64] ===== ★逐 seed 明细（EM batch=1，clean n={len(CLEAN_IDX)}/{N_EVAL}；@{STEPS64}）=====",
      flush=True)
print("[S64-TBL] seed | 来源 | base全量 | base clean | bind全量 | bind clean | Δ全量(pp) | Δclean(pp)", flush=True)
for s in ALL_SEEDS:
    b, d = ALL[(s, "base")], ALL[(s, "bind")]
    print(f"[S64-TBL] {s} | {b['src']} | {b['full']*100:.2f}% | {b['clean']*100:.2f}% | "
          f"{d['full']*100:.2f}% | {d['clean']*100:.2f}% | {(d['full']-b['full'])*100:+.2f} | "
          f"{(d['clean']-b['clean'])*100:+.2f}", flush=True)

print(f"\n[S64] ===== ★主判据：n=2（S63）vs n=4（S63+本单元）Δ@24000 =====", flush=True)
print(f"[S64-NOTE] n=2 = S63 的 {S63_SEEDS}；n=4 = 再加本单元 {SEEDS64}", flush=True)
emit_verdict(S63_SEEDS, "N2")
emit_verdict(ALL_SEEDS, "N4")

print(f"\n[S64] ===== ★机制复核：Δ@15000 vs Δ@24000 + base 15000→24000 增量 =====", flush=True)
print("[S64-MECH] seed | Δ@15000全量(clean) | Δ@24000全量(clean) | base 15000→24000 | bind 15000→24000",
      flush=True)
for s in ALL_SEEDS:
    cb15, cd15 = ALL[(s, "base")]["curve"].get("15000"), ALL[(s, "bind")]["curve"].get("15000")
    cb24, cd24 = ALL[(s, "base")]["curve"].get("24000"), ALL[(s, "bind")]["curve"].get("24000")
    d15 = (cd15["full"] - cb15["full"]) * 100 if (cb15 and cd15) else float("nan")
    d15c = (cd15["clean"] - cb15["clean"]) * 100 if (cb15 and cd15) else float("nan")
    d24 = (cd24["full"] - cb24["full"]) * 100 if (cb24 and cd24) else float("nan")
    d24c = (cd24["clean"] - cb24["clean"]) * 100 if (cb24 and cd24) else float("nan")
    incb = (cb24["full"] - cb15["full"]) * 100 if (cb15 and cb24) else float("nan")
    incd = (cd24["full"] - cd15["full"]) * 100 if (cd15 and cd24) else float("nan")
    print(f"[S64-MECH] {s} | {d15:+.2f}pp ({d15c:+.2f}) | {d24:+.2f}pp ({d24c:+.2f}) | "
          f"base {incb:+.2f}pp | bind {incd:+.2f}pp", flush=True)

print(f"\n[S64-SAT] ★饱和检查（R35）：末段增量（两臂 × 4 seed）", flush=True)
for s in ALL_SEEDS:
    for arm in ARMS64:
        cv = ALL[(s, arm)]["curve"]
        if not cv:
            print(f"[S64-SAT] {arm} seed={s}: 无曲线", flush=True)
            continue
        for a, b in ((21000, 24000), (15000, 24000)):
            if str(a) in cv and str(b) in cv:
                inc = (cv[str(b)]["full"] - cv[str(a)]["full"]) * 100
                incc = (cv[str(b)]["clean"] - cv[str(a)]["clean"]) * 100
                bad = max(inc, incc) >= 1.0
                flag = ("★≥1pp 未饱和 ⇒ 点名" if bad else "已平（<1pp）")
                print(f"[S64-SAT] {arm} seed={s}: {a}→{b} 全量={inc:+.2f}pp clean={incc:+.2f}pp ⇒ {flag}",
                      flush=True)

print(f"\n[S64] ===== ★(a) 置换敏感性（@24000，4 seed × 2 臂；只换被读列之前的两整行）=====", flush=True)
print("[S64-SWAP-TBL] seed | arm | max|Δlogits| | 中位|Δ| | Δ>1e-4 | logits量级 | 判定", flush=True)
for s in ALL_SEEDS:
    for arm in ARMS64:
        if s in S63_SEEDS:
            w = ANCHOR63[(arm, s)]["swap_best"]
            lsc = ANCHOR63[(arm, s)]["lsc"]
        else:
            w = RES[(s, arm)]["swap_best"]
            lsc = RES[(s, arm)]["lsc"]
        print(f"[S64-SWAP-TBL] {s} | {arm} | {w['mx']:.3e} | {w['med']:.3e} | {w['nz4']*100:.1f}% | "
              f"{lsc:.2f} | {swap_verdict(w['mx'])}", flush=True)

print(f"\n[S64] ===== 成本与 ckpt =====", flush=True)
for s in SEEDS64:
    for arm in ARMS64:
        r = RES[(s, arm)]
        wt = f"{r['wall_train']/60:.2f}min" if not r["reused"] else "复用"
        print(f"[META64] {arm} seed={s}: {r.get('src', '?')} ckpt={os.path.basename(r['ckpt'])} "
              f"sha256[:16]={r['ckpt_sha']} 训练墙钟={wt} R28={r['r28']['same']}/{r['r28']['k']}",
              flush=True)
for s in S63_SEEDS:
    for arm in ARMS64:
        A = ANCHOR63[(arm, s)]
        print(f"[META64] {arm} seed={s}: S63现评(未重训) ckpt={os.path.basename(S63_CKPT[(arm, s)])} "
              f"sha256[:16]={A['sha']}", flush=True)
print(f"[META64] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"threads={torch.get_num_threads()} | D={D} FF={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} "
      f"MAXLEN={MAXLEN} | EM口径=batch{EM_BATCH} | TEST sha16={G['TEST_SHA']} | 数据=旧版 | "
      f"总墙钟={(time.time()-T0)/60:.1f}min | 新训 run 数={len(RES)}/{len(SEEDS64)*len(ARMS64)} | "
      f"合并 n={len(ALL_SEEDS)}", flush=True)
print("[DONE] exit=0", flush=True)
sys.exit(0)
