#!/usr/bin/env python3
"""S63 · P3c 后续 ★ 把「位置绑定 +3.37pp」钉到饱和：①base / ②bind 推到 24000 步。

背景（P3c = stages/62_binding_seeds.py，本地提交）：
  n=4 @15000：Δ(bind−base)=+3.37pp ± 0.92（t=3.65），4/4 同号且 ≥+2SE ⇒ 提升；
  但 R35 附注：新 bind seed 9012 在 12000→15000 仍 +1.38pp、3456 +1.12pp（clean +1.43/+1.17）
  ⇒ 那两个格点未饱和 ⇒ +3.37pp 只是【下界】。P3c 自己的建议：加步数（n=6 不解决未饱和）。

本单元唯一变量 = 步数：只把 ①② 两臂从 15000 推到 24000 步（同 seed 同数据同超参同实现），
  seed 取 P3c 里未饱和的那两个（9012 / 3456；1234/5678 已饱和，不做）。
  每 750 步取权重快照（S38 已证「快照 = 独立跑 N 步」等价）；报告格点 = 每 1500 步（3000…24000）。

实现：本文件 exec stages/50_binding_at_saturation.py 的【前 ④ 段】（切在 "# ⑤ 主循环" 之前），
  从而逐字复用它的 BoundMHA / Cards50 / train50 / em_eval / swap_sensitivity / r28_selfcheck /
  参数与初始化自检；随后只把 CKPT_TMPL 改指 logs/63_*（★绝不覆盖 logs/50_*、logs/62_*），
  数据/超参/lr/batch/实现逐字不动（步数 15000→24000）。

判据（写死）：
  1 ★饱和检查（R35，本单元核心）：报 @21000→@24000 两臂增量；<1pp ⇒ 已饱和 ⇒ 结论可下；
    若 bind 仍 ≥1pp ⇒ 结论标条件性，并给「还需多少步」的线性外推（明确标注是外推）。
  2 主判据：Δ = EM(bind) − EM(base) @24000，逐 seed 配对，全量与 clean 都报：
    Δ ≥ +2SE 且 2 seed 同号 ⇒ 绑定在饱和点上仍提升能力；
    |Δ| ≤ 1SE ⇒ 饱和后增益消失（P3c 的 +3.37pp 属「加速收敛」⇒ 重要更正）；
    否则 ⇒ 仍不可判（报 n 与 SE）。
  3 ★与 P3c 对照：Δ@15000 vs Δ@24000 并列（维持 / 放大 / 消失）。
  4 ★(a) 置换敏感性：每个 ckpt 报 max|Δlogits|（对照 base）⇒ 绑定仍生效。
  5 ★两臂曲线（@3000…@24000）⇒ 看 bind 是「更早到平台且平台更高」还是「更快到同一平台」。

口径：旧数据（TEST sha256[:16] = b34e7ea515203227，与 P3b/P3c 同版）；batch=1（R28）；
  生成前显式 eval()（R39）；torch.set_num_threads(4)（R40）；地板/SEEN/UNSEEN 照报。
  ★锚点复核：复用 P3c 的 ckpt（logs/62_ckpt_{base,bind}_s15000_seed{9012,3456}.pt）现评，
  期望 base 9012=90.38 / bind 9012=93.50 / base 3456=88.50 / bind 3456=92.12（全量，Δ>0.5pp 点名）。
  ★确定性复核：63 在 @15000 的快照应与 62 的 @15000 ckpt 逐位相同（同一 RNG/数据序/无 lr 调度）。
ckpt：logs/63_ckpt_{base|bind}_s{15000,24000}_seed<s>.pt（报 sha256[:16]）；
  曲线落 logs/63_curve_{arm}_seed<s>.json；结果落 logs/63_results.jsonl。可续跑（ckpt 在则复用）。
只允许写：本文件（新建）、logs/63_*、logs/s63_bind24k.log、/tmp；既有 stages/*.py 只读。
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

SMOKE63 = os.environ.get("S63_SMOKE") == "1"
if SMOKE63:
    os.environ["S50_SMOKE"] = "1"          # 必须在 exec 之前：S50 前缀随之切小数据/短训

# ============================================================================
# ① 逐字复用 S50 前 ④ 段（其内部又 exec S19 前半段：同 tokenizer/同 add_3d 数据/
#    同 build_batch/masked_ce/greedy_gen/parse_ans/r28_selfcheck/sin_pe/enc_layer）
# ============================================================================
_src50 = open(SRC50, encoding="utf-8").read()
assert _src50.count("# ⑤ 主循环") == 1
_cut50 = _src50.index("# ⑤ 主循环")
G: dict = {"__name__": "s63_s50head", "__file__": SRC50}
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
G["CKPT_TMPL"] = ("/tmp/63_ckpt_{arm}_s{steps}_seed{seed}.pt" if SMOKE63
                  else f"{ROOT}/logs/63_ckpt_{{arm}}_s{{steps}}_seed{{seed}}.pt")
JSONL63 = "/tmp/63_results_smoke.jsonl" if SMOKE63 else f"{ROOT}/logs/63_results.jsonl"
CURVE_TMPL = "/tmp/63_curve_{arm}_seed{seed}.json" if SMOKE63 else f"{ROOT}/logs/63_curve_{{arm}}_seed{{seed}}.json"

T0 = time.time()
BUCKET = "add_3d"
ARMS63 = ("base", "bind")

if SMOKE63:
    STEPS63, STEP_SAVE63 = 60, 30
    GRID63 = (30, 60)
    SEEDS63 = (9012,)
else:
    STEPS63, STEP_SAVE63 = 24000, 750
    GRID63 = tuple(range(3000, STEPS63 + 1, 1500))        # 3000,4500,…,24000（含 15000/21000/24000）
    _env_seeds = os.environ.get("S63_SEEDS", "").strip()
    SEEDS63 = tuple(int(x) for x in _env_seeds.split(",") if x.strip()) if _env_seeds else (9012, 3456)
SNAP63 = tuple(range(STEP_SAVE63, STEPS63 + 1, STEP_SAVE63))
assert set(GRID63) <= set(SNAP63), "报告格点必须落在快照格点上"

# ---- P3c（62）入库锚：@15000 的 4 个 ckpt（旧数据；任务书给定值）----
P3C_CKPT = {(arm, s): f"{ROOT}/logs/62_ckpt_{arm}_s15000_seed{s}.pt"
            for s in (9012, 3456) for arm in ARMS63}
P3C_EM = {("base", 9012): (90.38, 93.90), ("bind", 9012): (93.50, 97.14),
          ("base", 3456): (88.50, 91.95), ("bind", 3456): (92.12, 95.71)}
P3C_SHA = {("base", 9012): "0d871ed2d7423e8d", ("bind", 9012): "5747f915f249fb5d",
           ("base", 3456): "53f3a60f690f4a0e", ("bind", 3456): "13485ac9f2f63bd4"}

print(f"\n[S63] ★P3c 后续：只加步数 15000→{STEPS63}：臂={ARMS63} seeds={SEEDS63} "
      f"（每 {STEP_SAVE63} 步快照，EM 格点={GRID63}）| d={D} ff={FF} nhead={NHEAD} "
      f"lr={LR} batch={BATCH} maxlen={MAXLEN} | 数据=旧版 add_3d | EM口径=batch{EM_BATCH} | "
      f"threads={torch.get_num_threads()} | smoke={SMOKE63}", flush=True)
print(f"[S63-DEV] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__}", flush=True)

# ============================================================================
# ② 数据版与口径复核（旧数据 TEST sha / 泄漏 / clean n / 地板）
# ============================================================================
print(f"[S63-CHK6] 数据版=【旧】数据（与 P3b/P3c 同版）| TEST sha256[:16]={G['TEST_SHA']} vs "
      f"入库 {G['TEST_SHA_EXPECT']} ⇒ "
      f"{'逐位一致 ✓' if G['TEST_SHA'] == G['TEST_SHA_EXPECT'] else '★不一致'} | "
      f"(a,b) 泄漏={G['LEAK']}/{len(TEST)} vs {G['LEAK_EXPECT']} | clean n={len(CLEAN_IDX)}/{N_EVAL}",
      flush=True)
print(f"[S63-CHK4] SEEN={G['N_SEEN']} / UNSEEN={G['N_UNSEEN']} | 地板(单答案/top5)="
      f"{G['FLOOR_1']*100:.2f}%/{G['FLOOR_5']*100:.2f}%", flush=True)

# ============================================================================
# ③ 参数量 + 基线权重逐位相同（复用 S50 前缀已算好的自检）
# ============================================================================
P_BASE, P_BIND = G["P_BASE"], G["P_BIND"]
PARAM_DIFF = abs(P_BIND - P_BASE) / P_BASE
print(f"[S63-PARAMS] 参数量：base={P_BASE} bind={P_BIND} | ①vs② 绝对差={P_BIND-P_BASE} "
      f"相对差={PARAM_DIFF*100:.4f}% ⇒ {'≤5% ✓' if PARAM_DIFF <= 0.05 else '★>5% 点名'}",
      flush=True)
print(f"[S63-CHK7] ①② 基线权重（emb/in_enc/thought 基线部分/head/pe）逐位相同："
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


def sd_maxdiff(sd_a, sd_b):
    """两个 state_dict 的逐键 max|Δ|（只在同名键上比；键集合不同 ⇒ -1）。"""
    if set(sd_a) != set(sd_b):
        return float("nan")
    return max(float((sd_a[k].float() - sd_b[k].float()).abs().max()) for k in sd_a)


def quick_swap(model):
    pairs, lo_s = swap_pairs()
    sw, lsc = swap_sensitivity(model, TEST[:32], pairs)
    best = max(sw.values(), key=lambda v: v["mx"])
    return dict(best), lsc, pairs, lo_s


# ============================================================================
# ④b 判定：Δ = bind − base 逐 seed 配对（样本 sd/√n；t = Δ/SE）
# ============================================================================
def emit_verdict(seeds, tag_step):
    """★主判据：Δ ≥ +2SE 且全同号 ⇒ 提升；|Δ| ≤ 1SE ⇒ 无影响；否则仍不可判。"""
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
        print(f"[S63-N@{tag_step}] n={len(seeds)} {tag}: Δ(bind−base)={m*100:+.2f}pp ± {se*100:.2f} "
              f"(t={t:.2f}) 逐seed={['%+.2f' % (d*100) for d in ds]} ⇒ {v}", flush=True)


# ============================================================================
# ⑤ 锚点复核：现评 P3c 的 4 个 @15000 ckpt（独立于本次训练）
# ============================================================================
ANCHOR: dict = {}
print("\n[S63] ===== ★锚点复核：现评 P3c(62) @15000 ckpt（独立于本次训练）=====", flush=True)
for s in (9012, 3456):
    for arm in ARMS63:
        p = P3C_CKPT[(arm, s)]
        assert os.path.exists(p), f"★锚点 ckpt 不存在：{p}"
        m = Cards50(D, FF, arm).to(device)
        m.load_state_dict({k: v.to(device) for k, v in torch.load(p, map_location="cpu").items()})
        m.eval()                                        # ★R39
        h = em_eval(m, TEST[:N_EVAL])
        f, c, sn, un = report_hits(h)
        ah, ac = P3C_EM[(arm, s)]
        d_f, d_c = f * 100 - ah, c * 100 - ac
        w, lsc, _, _ = quick_swap(m)
        ANCHOR[(arm, s)] = dict(full=f, clean=c, seen=sn, unseen=un, sha=sha16(p),
                                swap_best=w, lsc=lsc)
        print(f"[S63-ANCHOR] {arm} seed={s} @15000: 全量={f*100:.2f}% vs 入库 {ah:.2f}%（Δ={d_f:+.2f}pp）"
              f" | clean={c*100:.2f}% vs {ac:.2f}%（Δ={d_c:+.2f}pp）⇒ "
              f"{'复现 ✓（≤0.5pp）' if max(abs(d_f), abs(d_c)) <= 0.5 else '★>0.5pp，点名'} | "
              f"ckpt sha16={ANCHOR[(arm, s)]['sha']} vs 入库 {P3C_SHA[(arm, s)]} ⇒ "
              f"{'一致 ✓' if ANCHOR[(arm, s)]['sha'] == P3C_SHA[(arm, s)] else '★不一致'} | "
              f"max|Δlogits|={w['mx']:.3e}", flush=True)
        del m
        torch.cuda.empty_cache()


# ============================================================================
# ⑥ 单 (seed, arm)：训练到 24000 + 逐格评测；ckpt 已在则复用
# ============================================================================
RES: dict = {}
CURVES: dict = {}


def eval_grid(seed, arm, snaps, curve):
    ev = Cards50(D, FF, arm).to(device)
    for st in GRID63:
        ev.load_state_dict({k: v.to(device) for k, v in snaps[st].items()})
        ev.eval()                                       # ★R39
        hits = em_eval(ev, TEST[:N_EVAL])
        f, c, sn, un = report_hits(hits)
        curve[str(st)] = dict(full=f, clean=c, seen=sn, unseen=un)
        print(f"  [S63-PT] {arm} seed={seed} step={st} EM全量={f*100:.2f}% clean={c*100:.2f}% "
              f"SEEN={sn*100:.2f}% UNSEEN={un*100:.2f}%", flush=True)
        if st == GRID63[-1]:
            fin = dict(full=f, clean=c, seen=sn, unseen=un)
    ev.load_state_dict({k: v.to(device) for k, v in snaps[GRID63[-1]].items()})
    ev.eval()                                           # ★R39
    r28 = r28_selfcheck(ev, TEST[:N_EVAL], tag=f"[S63 {arm} seed{seed}] ",
                        k=min(CHK_BATCH, N_EVAL))
    w, lsc, _, _ = quick_swap(ev)
    del ev
    torch.cuda.empty_cache()
    return fin, r28, w, lsc


def run_train(seed, arm):
    t0 = time.time()
    ck, wall_train, snaps = train50(seed, arm, TRAIN, STEPS63, SNAP63, STEPS63)
    # ★确定性复核：63@15000 快照 vs P3c(62)@15000 ckpt（逐键 max|Δθ|；同 RNG/数据序/无调度）
    det = ""
    if not SMOKE63 and 15000 in snaps:
        p62 = P3C_CKPT[(arm, seed)]
        if os.path.exists(p62):
            sd62 = torch.load(p62, map_location="cpu")
            dd = sd_maxdiff(snaps[15000], sd62)
            print(f"[S63-DET] {arm} seed={seed}: 63@15000 快照 vs 62@15000 ckpt max|Δθ|={dd:.3e} ⇒ "
                  f"{'逐位相同 ✓（同一轨迹）' if dd == 0.0 else '★不同（轨迹分叉）'}", flush=True)
            det = dd
            del sd62
        ck15 = G["CKPT_TMPL"].format(arm=arm, steps=15000, seed=seed)
        torch.save(snaps[15000], ck15)
    curve: dict = {}
    fin, r28, w, lsc = eval_grid(seed, arm, snaps, curve)
    del snaps
    torch.cuda.empty_cache()
    with open(CURVE_TMPL.format(arm=arm, seed=seed), "w", encoding="utf-8") as f:
        json.dump(dict(arm=arm, seed=seed, steps=STEPS63, grid=list(GRID63), curve=curve, det=det),
                  f, ensure_ascii=False)
    return dict(**fin, curve=curve, wall_train=wall_train, wall_total=time.time() - t0,
                ckpt=ck, ckpt_sha=sha16(ck), reused=False, src="新训", r28=r28,
                swap_best=w, lsc=lsc, det=det)


def run_load(seed, arm):
    """复用已训好的 logs/63_ckpt_*_s24000_*：只评测；曲线从 json 读（若有）。"""
    p = G["CKPT_TMPL"].format(arm=arm, steps=STEPS63, seed=seed)
    t0 = time.time()
    assert os.path.exists(p), f"★ckpt 不存在：{p}"
    m = Cards50(D, FF, arm).to(device)
    m.load_state_dict({k: v.to(device) for k, v in torch.load(p, map_location="cpu").items()})
    m.eval()                                            # ★R39
    hits = em_eval(m, TEST[:N_EVAL])
    f, c, sn, un = report_hits(hits)
    m.eval()
    r28 = r28_selfcheck(m, TEST[:N_EVAL], tag=f"[S63 复用 {arm} seed{seed}] ", k=min(CHK_BATCH, N_EVAL))
    w, lsc, _, _ = quick_swap(m)
    del m
    torch.cuda.empty_cache()
    cp = CURVE_TMPL.format(arm=arm, seed=seed)
    curve = json.load(open(cp, encoding="utf-8"))["curve"] if os.path.exists(cp) else {}
    return dict(full=f, clean=c, seen=sn, unseen=un, curve=curve, wall_train=float("nan"),
                wall_total=time.time() - t0, ckpt=p, ckpt_sha=sha16(p), reused=True, src="复用63",
                r28=r28, swap_best=w, lsc=lsc, det=float("nan"))


def run_auto(seed, arm):
    p = G["CKPT_TMPL"].format(arm=arm, steps=STEPS63, seed=seed)
    if not SMOKE63 and os.path.exists(p):
        return run_load(seed, arm)
    return run_train(seed, arm)


def dump(seed, arm, r):
    with open(JSONL63, "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(stage="63", bucket=BUCKET, seed=seed, arm=arm, steps=STEPS63,
                                full=r["full"], clean=r["clean"], seen=r["seen"], unseen=r["unseen"],
                                curve=r["curve"], reused=r["reused"], src=r.get("src", "?"),
                                ckpt=os.path.basename(r["ckpt"]), ckpt_sha=r["ckpt_sha"],
                                wall_train=r["wall_train"], r28=r["r28"], swap_best=r["swap_best"],
                                det=r.get("det", float("nan"))), ensure_ascii=False) + "\n")


# ============================================================================
# ⑦ 主循环：seed 9012 两臂 → seed 3456 两臂（9012 先跑完 ⇒ 时间紧时可先报）
# ============================================================================
open(JSONL63, "w").close()
for seed in SEEDS63:
    for arm in ARMS63:
        r = run_auto(seed, arm)
        RES[(seed, arm)] = r
        dump(seed, arm, r)
        if r["reused"]:
            print(f"[S63-RUN] 复用 {arm} seed={seed}: 全量={r['full']*100:.2f}% clean={r['clean']*100:.2f}% "
                  f"| R28={r['r28']['same']}/{r['r28']['k']} | 评测墙钟={r['wall_total']/60:.2f}min", flush=True)
        else:
            print(f"[S63-RUN] 新 {arm} seed={seed}: EM@{STEPS63}全量={r['full']*100:.2f}% "
                  f"clean={r['clean']*100:.2f}% SEEN={r['seen']*100:.2f}% UNSEEN={r['unseen']*100:.2f}% "
                  f"| R28={r['r28']['same']}/{r['r28']['k']} | 训练墙钟={r['wall_train']/60:.2f}min "
                  f"本 run={r['wall_total']/60:.2f}min | ckpt={os.path.basename(r['ckpt'])} "
                  f"sha16={r['ckpt_sha']}", flush=True)
    done = [s for s in SEEDS63 if (s, "base") in RES and (s, "bind") in RES]
    if len(done) >= 1:
        print(f"\n[S63-PROGRESS] 已完成配对 seeds={done}（n={len(done)}）", flush=True)
        emit_verdict(done, STEPS63)
        print("", flush=True)

# ============================================================================
# ⑧ 明细表 + 曲线 + 饱和检查 + Δ@15000 vs @24000 + 置换敏感性
# ============================================================================
print(f"\n[S63] ===== ★逐 seed 明细（EM batch=1，clean n={len(CLEAN_IDX)}/{N_EVAL}；@{STEPS63}）=====",
      flush=True)
print("[S63-TBL] seed | base全量 | base clean | bind全量 | bind clean | Δ全量(pp) | Δclean(pp)", flush=True)
for s in SEEDS63:
    b, d = RES[(s, "base")], RES[(s, "bind")]
    print(f"[S63-TBL] {s} | {b['full']*100:.2f}% | {b['clean']*100:.2f}% | {d['full']*100:.2f}% | "
          f"{d['clean']*100:.2f}% | {(d['full']-b['full'])*100:+.2f} | {(d['clean']-b['clean'])*100:+.2f}",
          flush=True)

print(f"\n[S63] ===== ★两臂曲线（EM 全量% / clean%；格点 {'/'.join(str(x) for x in GRID63)}）=====", flush=True)
for s in SEEDS63:
    for arm in ARMS63:
        cv = RES[(s, arm)]["curve"]
        if not cv:
            print(f"[S63-CURVE] {arm} seed={s}: （无曲线，复用且无 json）", flush=True)
            continue
        fs = " ".join(f"{cv[str(st)]['full']*100:.2f}" for st in GRID63)
        cs = " ".join(f"{cv[str(st)]['clean']*100:.2f}" for st in GRID63)
        print(f"[S63-CURVE] {arm} seed={s} 全量: {fs}", flush=True)
        print(f"[S63-CURVE] {arm} seed={s} clean: {cs}", flush=True)
    # Δ 曲线（bind − base）
    cb, cd = RES[(s, "base")]["curve"], RES[(s, "bind")]["curve"]
    if cb and cd:
        ds = " ".join(f"{(cd[str(st)]['full']-cb[str(st)]['full'])*100:+.2f}" for st in GRID63)
        print(f"[S63-CURVE] Δ(bind−base) seed={s} 全量: {ds}", flush=True)

print(f"\n[S63-SAT] ★饱和检查（R35）：末段增量", flush=True)
for s in SEEDS63:
    for arm in ARMS63:
        cv = RES[(s, arm)]["curve"]
        if not cv:
            print(f"[S63-SAT] {arm} seed={s}: 无曲线", flush=True)
            continue
        for a, b in ((21000, 24000), (15000, 24000)):
            if str(a) in cv and str(b) in cv:
                inc = (cv[str(b)]["full"] - cv[str(a)]["full"]) * 100
                incc = (cv[str(b)]["clean"] - cv[str(a)]["clean"]) * 100
                flag = ("★≥1pp 未饱和 ⇒ 能力结论不可下" if (arm == "bind" and max(inc, incc) >= 1.0)
                        else "已平（<1pp）")
                print(f"[S63-SAT] {arm} seed={s}: {a}→{b} 全量={inc:+.2f}pp clean={incc:+.2f}pp ⇒ {flag}",
                      flush=True)

print(f"\n[S63-D1500] ★Δ@15000（P3c 入库）vs Δ@24000（本单元）对照", flush=True)
for s in SEEDS63:
    ah = P3C_EM[("bind", s)][0] - P3C_EM[("base", s)][0]
    ac = P3C_EM[("bind", s)][1] - P3C_EM[("base", s)][1]
    r15f = RES[(s, "bind")]["curve"].get("15000", {}).get("full", float("nan"))
    r15b = RES[(s, "base")]["curve"].get("15000", {}).get("full", float("nan"))
    a15 = (r15f - r15b) * 100 if r15f == r15f else float("nan")
    d24 = (RES[(s, "bind")]["full"] - RES[(s, "base")]["full"]) * 100
    d24c = (RES[(s, "bind")]["clean"] - RES[(s, "base")]["clean"]) * 100
    print(f"[S63-D1500] seed={s}: Δ@15000(P3c 入库)={ah:+.2f}pp（clean {ac:+.2f}）"
          f" | Δ@15000(63 轨迹内)={a15:+.2f}pp | Δ@24000={d24:+.2f}pp（clean {d24c:+.2f}）", flush=True)

print(f"\n[S63] ===== ★(a) 置换敏感性（每个 ckpt；只换被读列之前的两整行）=====", flush=True)
print("[S63-SWAP-TBL] arm | seed | step | max|Δlogits| | 中位|Δ| | Δ>1e-4 | logits量级 | 判定", flush=True)
for s in (9012, 3456):
    for arm in ARMS63:
        w = ANCHOR[(arm, s)]["swap_best"]
        vd = ("★绑定生效（≥1e-3）" if w["mx"] >= 1e-3 else
              ("绑定生效但弱（≥1e-4）" if w["mx"] >= 1e-4 else "★未生效（fp 噪声量级）"))
        print(f"[S63-SWAP-TBL] {arm} | {s} | 15000(锚) | {w['mx']:.3e} | {w['med']:.3e} | "
              f"{w['nz4']*100:.1f}% | {ANCHOR[(arm, s)]['lsc']:.2f} | {vd}", flush=True)
    for arm in ARMS63:
        if s in SEEDS63:
            w = RES[(s, arm)]["swap_best"]
            vd = ("★绑定生效（≥1e-3）" if w["mx"] >= 1e-3 else
                  ("绑定生效但弱（≥1e-4）" if w["mx"] >= 1e-4 else "★未生效（fp 噪声量级）"))
            print(f"[S63-SWAP-TBL] {arm} | {s} | {STEPS63} | {w['mx']:.3e} | {w['med']:.3e} | "
                  f"{w['nz4']*100:.1f}% | {RES[(s, arm)]['lsc']:.2f} | {vd}", flush=True)

print(f"\n[S63] ===== 成本与 ckpt =====", flush=True)
for s in SEEDS63:
    for arm in ARMS63:
        r = RES[(s, arm)]
        wt = f"{r['wall_train']/60:.2f}min" if not r["reused"] else "复用"
        print(f"[META63] {arm} seed={s}: {r.get('src', '?')} ckpt={os.path.basename(r['ckpt'])} "
              f"sha256[:16]={r['ckpt_sha']} 训练墙钟={wt} R28={r['r28']['same']}/{r['r28']['k']}",
              flush=True)
print(f"[META63] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"threads={torch.get_num_threads()} | D={D} FF={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} "
      f"MAXLEN={MAXLEN} | EM口径=batch{EM_BATCH} | TEST sha16={G['TEST_SHA']} | 数据=旧版 | "
      f"总墙钟={(time.time()-T0)/60:.1f}min | run 数={len(RES)}/{len(SEEDS63)*len(ARMS63)}", flush=True)
print("[DONE] exit=0", flush=True)
sys.exit(0)
