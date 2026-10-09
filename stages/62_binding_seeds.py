#!/usr/bin/env python3
"""S62 · P3c ★ 把「位置绑定 +3.37pp 中间带」钉死：加 seed 到 n=4–6（只加 seed，其余逐字同 P3b）。

背景（P3b = stages/50_binding_at_saturation.py，提交 aed7ff6，2 seed @15000，旧数据）：
  ①base 91.25/94.81 · 87.25/90.65 ｜ ②bind 92.38/95.97 · 92.88/96.49（全量/clean）
  Δ(②−①) = +3.37pp ± 2.25（逐 seed +1.12 / +5.62）⇒ 2 seed 同号为正但 < 2SE(4.50) ⇒ 中间带；
  Δ(③−②) = −2.00pp ± 1.00 ⇒ ③rand 显著差于 ② ⇒ 位置绑定是有效信号（P3b 已下、本单元不重做）；
  饱和检查 12000→15000 全部 <1pp ⇒ 结论可下。
⇒ 本单元唯一变量 = seed：复用 P3b 的 ①② ckpt（1234/5678，不重训，先复现入库 EM），
  再新增 4 个 seed（9012/3456/7890/2468），只跑 ①base 与 ②bind 两臂（③rand 不做）。

实现：本文件 exec stages/50_binding_at_saturation.py 的【前 ④ 段】（切在 "# ⑤ 主循环" 之前），
  从而逐字复用它的 BoundMHA / Cards50 / train50 / em_eval / swap_sensitivity / 参数与初始化自检；
  随后把 CKPT_TMPL 改指 logs/62_ckpt_*（★绝不覆盖 logs/50_*），数据/超参/lr/batch/15000 步逐字不动。

判据（写死）：
  Δ = EM(bind) − EM(base)，逐 seed 配对，全量与 clean 都报 n=2/4/6 的 均值±SE 与 t：
    Δ ≥ +2SE 且全部 seed 同号 ⇒ 绑定提升能力；|Δ| ≤ 1SE ⇒ 无影响；否则 ⇒ 仍不可判。
  饱和检查（R35）：每个新 seed 报 12000→15000 增量，任一格 ≥1pp 点名。
  (a) 置换敏感性：每个 seed 的 ①② 都报 max|Δlogits| ⇒ 确认绑定在每个 seed 上都生效。

口径：旧数据（TEST sha256[:16] = b34e7ea515203227，与 P3b 同版）；batch=1（R28）；
  生成前显式 eval()（R39）；torch.set_num_threads(4)（R40）；地板/SEEN/UNSEEN 照报。
ckpt：logs/62_ckpt_{base|bind}_s15000_seed<s>.pt（报 sha256[:16]）；结果落 logs/62_results.jsonl。
只允许写：本文件（新建）、logs/62_*、logs/p3c_bind.log、/tmp；既有 stages/*.py 只读。
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

SMOKE62 = os.environ.get("S62_SMOKE") == "1"
if SMOKE62:
    os.environ["S50_SMOKE"] = "1"          # 必须在 exec 之前：S50 前缀随之切小数据/短训

# ============================================================================
# ① 逐字复用 S50 前 ④ 段（其内部又 exec S19 前半段：同 tokenizer/同 add_3d 数据/
#    同 build_batch/masked_ce/greedy_gen/parse_ans/r28_selfcheck/sin_pe/enc_layer）
# ============================================================================
_src50 = open(SRC50, encoding="utf-8").read()
assert _src50.count("# ⑤ 主循环") == 1
_cut50 = _src50.index("# ⑤ 主循环")
G: dict = {"__name__": "s62_s50head", "__file__": SRC50}
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
G["CKPT_TMPL"] = ("/tmp/62_ckpt_{arm}_s{steps}_seed{seed}.pt" if SMOKE62
                  else f"{ROOT}/logs/62_ckpt_{{arm}}_s{{steps}}_seed{{seed}}.pt")
JSONL62 = "/tmp/62_results_smoke.jsonl" if SMOKE62 else f"{ROOT}/logs/62_results.jsonl"

T0 = time.time()
BUCKET = "add_3d"
ARMS62 = ("base", "bind")
NEW_SEEDS_ALL = (9012, 3456, 7890, 2468)
_env_seeds = os.environ.get("S62_SEEDS", "").strip()
NEW_SEEDS = tuple(int(x) for x in _env_seeds.split(",") if x.strip()) if _env_seeds else NEW_SEEDS_ALL
REUSE_SEEDS = () if SMOKE62 else (1234, 5678)

if SMOKE62:
    STEPS62, STEP_SAVE62, GRID62 = 60, 30, (30, 60)
else:
    STEPS62, STEP_SAVE62, GRID62 = 15000, 750, (12000, 15000)
SNAP62 = tuple(range(STEP_SAVE62, STEPS62 + 1, STEP_SAVE62))
assert set(GRID62) <= set(SNAP62), "报告格点必须落在快照格点上"

# ---- P3b 入库锚（logs/p3b_bind.log [S50-EM] / [META]；旧数据）----
P3B_CKPT = {(arm, s): f"{ROOT}/logs/50_ckpt_{arm}_s15000_seed{s}.pt"
            for s in (1234, 5678) for arm in ARMS62}
P3B_SHA = {("base", 1234): "d6865dd0bc198c18", ("base", 5678): "84ce98bce275a7a0",
           ("bind", 1234): "9a800fb42f499092", ("bind", 5678): "d7798022d02590c7"}
P3B_EM = {("base", 1234): (91.25, 94.81), ("bind", 1234): (92.38, 95.97),
          ("base", 5678): (87.25, 90.65), ("bind", 5678): (92.88, 96.49)}

print(f"\n[S62] ★P3c 只加 seed：臂={ARMS62} 复用 seeds={REUSE_SEEDS} 新增 seeds={NEW_SEEDS} "
      f"@ {STEPS62} 步（每 {STEP_SAVE62} 步快照，EM 格点={GRID62}）| d={D} ff={FF} nhead={NHEAD} "
      f"lr={LR} batch={BATCH} maxlen={MAXLEN} | 数据=旧版 add_3d | EM口径=batch{EM_BATCH} | "
      f"threads={torch.get_num_threads()} | smoke={SMOKE62}", flush=True)
print(f"[S62-DEV] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__}", flush=True)

# ============================================================================
# ② 数据版与口径复核（旧数据 TEST sha / 泄漏 / clean n / 地板）
# ============================================================================
print(f"[S62-CHK6] 数据版=【旧】数据（生成器修 stages/14_synth_arith.py:126 之前；本单元沿用 P3b 同版）"
      f"| TEST sha256[:16]={G['TEST_SHA']} vs 入库 {G['TEST_SHA_EXPECT']} ⇒ "
      f"{'逐位一致 ✓' if G['TEST_SHA'] == G['TEST_SHA_EXPECT'] else '★不一致（冒烟/异常）'} | "
      f"(a,b) 泄漏={G['LEAK']}/{len(TEST)} vs {G['LEAK_EXPECT']} | clean n={len(CLEAN_IDX)}/{N_EVAL}",
      flush=True)
print(f"[S62-CHK4] SEEN={G['N_SEEN']} / UNSEEN={G['N_UNSEEN']} | 地板(单答案/top5)="
      f"{G['FLOOR_1']*100:.2f}%/{G['FLOOR_5']*100:.2f}%", flush=True)

# ============================================================================
# ③ 参数量 + 基线权重逐位相同（复用 S50 前缀已算好的自检；R31/R39 口径 5）
# ============================================================================
P_BASE, P_BIND = G["P_BASE"], G["P_BIND"]
PARAM_DIFF = abs(P_BIND - P_BASE) / P_BASE
print(f"[S62-PARAMS] 参数量：base={P_BASE} bind={P_BIND} | ①vs② 绝对差={P_BIND-P_BASE} "
      f"相对差={PARAM_DIFF*100:.4f}% ⇒ {'≤5% ✓' if PARAM_DIFF <= 0.05 else '★>5% 点名'}"
      f"（★本单元实测差 {P_BIND-P_BASE}；任务书写 +5248，与 S50/P3b 实测 5184 不符，按实测）", flush=True)
print(f"[S62-CHK7] ①② 基线权重（emb/in_enc/thought 基线部分/head/pe）逐位相同："
      f"max|Δθ|={G['_wbin']['bind']:.3e} ⇒ "
      f"{'max|Δθ|=0 ✓' if G['_wbin']['bind'] == 0.0 else '★非零，点名'}", flush=True)


# ============================================================================
# ④ 单 (seed, arm) 评测：复用 ckpt 直接载入；新 seed 先训 15000 步
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


def run_new(seed, arm):
    t0 = time.time()
    ck, wall_train, snaps = train50(seed, arm, TRAIN, STEPS62, SNAP62, STEPS62)
    ev = Cards50(D, FF, arm).to(device)
    curve = {}
    for st in GRID62:
        ev.load_state_dict({k: v.to(device) for k, v in snaps[st].items()})
        ev.eval()                                      # ★R39
        hits = em_eval(ev, TEST[:N_EVAL])
        f, c, sn, un = report_hits(hits)
        curve[st] = dict(full=f, clean=c, seen=sn, unseen=un)
        print(f"  [S62-PT] {arm} seed={seed} step={st} EM全量={f*100:.2f}% clean={c*100:.2f}% "
              f"SEEN={sn*100:.2f}% UNSEEN={un*100:.2f}%", flush=True)
        if st == STEPS62:
            fin_hits = hits
    ev.load_state_dict({k: v.to(device) for k, v in snaps[STEPS62].items()})
    del snaps
    torch.cuda.empty_cache()
    ev.eval()                                          # ★R39
    r28 = r28_selfcheck(ev, TEST[:N_EVAL], tag=f"[S62 {arm} seed{seed}] ",
                        k=min(CHK_BATCH, N_EVAL))
    pairs, lo_s = swap_pairs()
    sw, lsc = swap_sensitivity(ev, TEST[:32], pairs)
    best = max(sw.values(), key=lambda v: v["mx"])
    return dict(full=curve[STEPS62]["full"], clean=curve[STEPS62]["clean"],
                seen=curve[STEPS62]["seen"], unseen=curve[STEPS62]["unseen"],
                curve={str(k): v for k, v in curve.items()}, wall_train=wall_train,
                wall_total=time.time() - t0, ckpt=ck, ckpt_sha=sha16(ck), reused=False, src="新训",
                r28=r28, swap={f"{i}-{j}": v for (i, j), v in sw.items()},
                swap_best=dict(best), lsc=lsc, lo_s=lo_s)


def run_load(seed, arm, path, label):
    """复用既有 ckpt（P3b 的 logs/50_* 或本单元已训好的 logs/62_*），只评测不重训。"""
    t0 = time.time()
    assert os.path.exists(path), f"★ckpt 不存在：{path}"
    m = Cards50(D, FF, arm).to(device)
    m.load_state_dict({k: v.to(device) for k, v in torch.load(path, map_location="cpu").items()})
    m.eval()                                           # ★R39
    hits = em_eval(m, TEST[:N_EVAL])
    f, c, sn, un = report_hits(hits)
    m.eval()
    r28 = r28_selfcheck(m, TEST[:N_EVAL], tag=f"[S62 {label} {arm} seed{seed}] ",
                        k=min(CHK_BATCH, N_EVAL))
    pairs, lo_s = swap_pairs()
    sw, lsc = swap_sensitivity(m, TEST[:32], pairs)
    best = max(sw.values(), key=lambda v: v["mx"])
    if label == "复用P3b":
        ah, ac = P3B_EM[(arm, seed)]
        d_f, d_c = (f * 100 - ah), (c * 100 - ac)
        print(f"[S62-ANCHOR] 复用 P3b {arm} seed={seed}: 全量={f*100:.2f}% vs 入库 {ah:.2f}%（Δ={d_f:+.2f}pp）"
              f" | clean={c*100:.2f}% vs {ac:.2f}%（Δ={d_c:+.2f}pp）⇒ "
              f"{'复现 ✓（≤0.5pp）' if max(abs(d_f), abs(d_c)) <= 0.5 else '★>0.5pp，点名'} | "
              f"ckpt sha16={sha16(path)} vs 入库 {P3B_SHA[(arm, seed)]} ⇒ "
              f"{'一致 ✓' if sha16(path) == P3B_SHA[(arm, seed)] else '★不一致'}", flush=True)
    return dict(full=f, clean=c, seen=sn, unseen=un, curve=None, wall_train=float("nan"),
                wall_total=time.time() - t0, ckpt=path, ckpt_sha=sha16(path), reused=True,
                src=label, r28=r28, swap={f"{i}-{j}": v for (i, j), v in sw.items()},
                swap_best=dict(best), lsc=lsc, lo_s=lo_s)


def run_auto(seed, arm):
    """可续跑：本单元 62 ckpt 已在 ⇒ 直接复用；P3b ckpt 在 ⇒ 复用；否则训练。"""
    p62 = G["CKPT_TMPL"].format(arm=arm, steps=STEPS62, seed=seed)
    if not SMOKE62 and os.path.exists(p62):
        return run_load(seed, arm, p62, "复用62")
    if (arm, seed) in P3B_CKPT and os.path.exists(P3B_CKPT[(arm, seed)]):
        return run_load(seed, arm, P3B_CKPT[(arm, seed)], "复用P3b")
    return run_new(seed, arm)


# ============================================================================
# ④b 判定：Δ = bind − base 逐 seed 配对（均值 ± SE 用样本 sd/√n；t = Δ/SE）
# ============================================================================
def emit_verdict(seeds):
    """★主判据：Δ ≥ +2SE 且全同号 ⇒ 提升；|Δ| ≤ 1SE ⇒ 无影响；否则仍不可判。"""
    out = {}
    for mode, tag in (("full", "全量"), ("clean", "clean")):
        ds = [RES[(s, "bind")][mode] - RES[(s, "base")][mode] for s in seeds]
        m, se = mean_se(ds)
        t = m / se if se > 0 else float("nan")
        both_pos, both_neg = all(d > 0 for d in ds), all(d < 0 for d in ds)
        if m >= 2 * se and both_pos:
            v = "★提升（Δ≥+2SE 且全同号）"
        elif abs(m) <= se:
            v = "无影响（|Δ|≤1SE）"
        elif m <= -2 * se and both_neg:
            v = "★有害（Δ≤−2SE 且全同号）"
        else:
            v = "仍不可判（中间带）"
        out[mode] = (m, se, ds, t, v)
        print(f"[S62-N] n={len(seeds)} {tag}: Δ(bind−base)={m*100:+.2f}pp ± {se*100:.2f} "
              f"(t={t:.2f}) 逐seed={['%+.2f' % (d*100) for d in ds]} ⇒ {v}", flush=True)
    return out


# ============================================================================
# ⑤ 主循环：复用 2 seed（快）→ 新增 seed 逐个训（base/bind 配对）
# ============================================================================
RES: dict = {}
open(JSONL62, "w").close()
ALL_SEEDS_ORDER = tuple(REUSE_SEEDS) + tuple(NEW_SEEDS)


def dump(seed, arm, r):
    with open(JSONL62, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(stage="62", bucket=BUCKET, seed=seed, arm=arm,
                                full=r["full"], clean=r["clean"], seen=r["seen"],
                                unseen=r["unseen"], curve=r["curve"], reused=r["reused"],
                                src=r.get("src", "?"),
                                ckpt=os.path.basename(r["ckpt"]), ckpt_sha=r["ckpt_sha"],
                                wall_train=r["wall_train"], r28=r["r28"],
                                swap=r["swap"], lsc=r["lsc"]),
                           ensure_ascii=False) + "\n")


for seed in ALL_SEEDS_ORDER:
    for arm in ARMS62:
        r = run_auto(seed, arm)
        RES[(seed, arm)] = r
        dump(seed, arm, r)
        if r["reused"]:
            print(f"[S62-RUN] {r['src']} {arm} seed={seed}: 全量={r['full']*100:.2f}% "
                  f"clean={r['clean']*100:.2f}% | R28={r['r28']['same']}/{r['r28']['k']} | "
                  f"评测墙钟={r['wall_total']/60:.2f}min", flush=True)
        else:
            print(f"[S62-RUN] 新 {arm} seed={seed}: EM@{STEPS62}全量={r['full']*100:.2f}% "
                  f"clean={r['clean']*100:.2f}% SEEN={r['seen']*100:.2f}% UNSEEN={r['unseen']*100:.2f}% "
                  f"| R28={r['r28']['same']}/{r['r28']['k']} | 训练墙钟={r['wall_train']/60:.2f}min "
                  f"本 run={r['wall_total']/60:.2f}min | ckpt={os.path.basename(r['ckpt'])} "
                  f"sha16={r['ckpt_sha']}", flush=True)
        # ★每完成一个 seed 的配对就出一次累积判决（n=2/3/4/5/6）
        done = [s for s in ALL_SEEDS_ORDER if (s, "base") in RES and (s, "bind") in RES]
        if len(done) >= 2:
            print(f"\n[S62-PROGRESS] 已完成配对 seeds={done}（n={len(done)}）", flush=True)
            emit_verdict(done)
            print("", flush=True)

# ============================================================================
# ⑥ 逐 seed 明细表 + 饱和检查 + 置换敏感性表
# ============================================================================
print(f"\n[S62] ===== ★逐 seed 明细（EM batch=1，clean n={len(CLEAN_IDX)}/{N_EVAL}；@{STEPS62}）=====",
      flush=True)
print("[S62-TBL] seed | 复用? | base全量 | base clean | bind全量 | bind clean | "
      "Δ全量(pp) | Δclean(pp)", flush=True)
for s in ALL_SEEDS_ORDER:
    b, d = RES[(s, "base")], RES[(s, "bind")]
    print(f"[S62-TBL] {s} | {b.get('src', '?')} | {b['full']*100:.2f}% | "
          f"{b['clean']*100:.2f}% | {d['full']*100:.2f}% | {d['clean']*100:.2f}% | "
          f"{(d['full']-b['full'])*100:+.2f} | {(d['clean']-b['clean'])*100:+.2f}", flush=True)

print(f"\n[S62-SAT] ★饱和检查（R35）：12000→15000 增量（新 seed；复用 seed 无曲线）", flush=True)
for s in NEW_SEEDS:
    for arm in ARMS62:
        c = RES[(s, arm)]["curve"]
        inc = (c[str(GRID62[-1])]["full"] - c[str(GRID62[0])]["full"]) * 100
        incc = (c[str(GRID62[-1])]["clean"] - c[str(GRID62[0])]["clean"]) * 100
        flag = "★≥1pp 未饱和 ⇒ 能力结论不可下" if max(inc, incc) >= 1.0 else "已平（<1pp）"
        print(f"[S62-SAT] {arm} seed={s}: {GRID62[0]}→{GRID62[-1]} 全量={inc:+.2f}pp "
              f"clean={incc:+.2f}pp ⇒ {flag}", flush=True)
for s in REUSE_SEEDS:
    for arm in ARMS62:
        print(f"[S62-SAT] {arm} seed={s}: 复用 P3b ckpt（P3b 入库 12000→15000 全量 = base "
              f"−0.62/+0.13pp、bind +0.87/−0.38pp，全部 <1pp）", flush=True)

print(f"\n[S62] ===== ★(a) 置换敏感性（只换被读列之前的两整行；被读列 = target 列）=====", flush=True)
print("[S62-SWAP-TBL] arm | seed | max|Δlogits| | 中位|Δ| | Δ>1e-4 | logits量级 | 判定", flush=True)
for s in ALL_SEEDS_ORDER:
    for arm in ARMS62:
        w = RES[(s, arm)]["swap_best"]
        verdict = ("★绑定生效（≥1e-3）" if w["mx"] >= 1e-3 else
                   ("绑定生效但弱（≥1e-4）" if w["mx"] >= 1e-4 else "★未生效（fp 噪声量级）"))
        print(f"[S62-SWAP-TBL] {arm} | {s} | {w['mx']:.3e} | {w['med']:.3e} | {w['nz4']*100:.1f}% | "
              f"{RES[(s, arm)]['lsc']:.2f} | {verdict}", flush=True)

print(f"\n[S62] ===== ★n=2 / n=4 / n=6 的均值±SE 与判定（判据写死）=====", flush=True)
FINAL_N: dict = {}
for n in (2, 4, 6):
    if len(ALL_SEEDS_ORDER) >= n:
        FINAL_N[n] = emit_verdict(ALL_SEEDS_ORDER[:n])

print(f"\n[S62] ===== 成本与 ckpt =====", flush=True)
for s in ALL_SEEDS_ORDER:
    for arm in ARMS62:
        r = RES[(s, arm)]
        wt = f"{r['wall_train']/60:.2f}min" if r["reused"] is False else "复用"
        print(f"[META62] {arm} seed={s}: {r.get('src', '?')} ckpt="
              f"{os.path.basename(r['ckpt'])} sha256[:16]={r['ckpt_sha']} 训练墙钟={wt} "
              f"R28={r['r28']['same']}/{r['r28']['k']}", flush=True)
print(f"[META62] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"threads={torch.get_num_threads()} | D={D} FF={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} "
      f"MAXLEN={MAXLEN} | EM口径=batch{EM_BATCH} | TEST sha16={G['TEST_SHA']} | 数据=旧版 | "
      f"总墙钟={(time.time()-T0)/60:.1f}min | run 数={len(RES)}/{len(ALL_SEEDS_ORDER)*len(ARMS62)}",
      flush=True)
print("[DONE] exit=0", flush=True)
sys.exit(0)
