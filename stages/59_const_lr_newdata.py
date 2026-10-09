#!/usr/bin/env python3
"""S59 · ★补上 2×2 的缺格：常量 lr=1e-3（无 warmup、无 cosine）+【新数据】。

背景（F29）：四格只有三格 ——（常量 lr, 新数据）缺。
  本单元只改 LR 调度（去掉 warmup500+cosine→0，全程恒 1e-3），其余逐字同 S58 的 FIX 臂：
  数据 = 已修 stages/14_synth_arith.py:126 后重生成的 add_3d；结构 / batch / 步数 / seed /
  评估口径全同（模型与训练函数直接 exec S53 源码 ⇒ 同初始化、同数据顺序、同 dropout 消耗）。

★做法（全因子 × 配对，规避「跨数据版不可比」）：
  4 组 ckpt = {常量 lr, warmup+cosine} × {旧数据, 新数据}，全部在**同一 test 集**上逐样本配对：
    (常量,旧) = logs/38_ckpt_w1_s12000_seed{s}.pt          （S38，已入库）
    (cos ,旧) = logs/53_ckpt_lrcosine_s12000_seed{s}.pt    （S53，已入库）
    (cos ,新) = logs/58_ckpt_fixeddata_s12000_seed{s}.pt   （S58 FIX，已入库）
    (常量,新) = 本臂（本次训练）
  ⇒ 调度效应 / 数据效应 / 交互 全部用**配对 Δ**（逐样本），绝不比跨数据版的绝对 EM。

★判据（写死，见任务书）：
  1 主判据：本臂绝对 EM（新 test，n=800，bad=0 ⇒ clean≡全量）放进 2×2；
    同数据版内比：新数据上「cos − 常量」= 调度效应（新数据），与旧数据上的 +1.82pp(clean, s1234) 比；
  2 交互 = [EM(cos,新) − EM(常量,新)] − [EM(cos,旧) − EM(常量,旧)]，配对 ±SE；
  3 饱和检查 R35：本臂 9000→12000 增量 ⇒ ≥1pp 标条件性；
  4 曲线对照：本臂 @3000/6000/9000/12000 全量 EM vs FIX 同点；
  5 报新数据绝对 EM（供 METRICS 新数据栏）。

★口径：EM 一律 batch=1（R28）+ batch=1 vs 16 自检；R39 每次前向显式 eval()；
  R40 4 线程；地板 0.12%/0.25%；SEEN/UNSEEN 分层；新 TEST sha256[:16]=fa03ef9287567510（复认）。
★锚点复核（Δ>0.5pp 点名）：58_ckpt@新test=97.38/96.88；53_ckpt@新test=96.25；
  38_ckpt@旧test=91.88/95.45（s1234）；53_ckpt@旧test=93.62/97.27。

只允许写：本文件、logs/s58_constlr.log（systemd 重定向）、logs/59_*、/tmp。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

ROOT = "/home/vesita/coding/my/flowme"
T0 = time.time()

# ============================================================================
# ① 旧世界：exec S53 源码到 ④a 之前（内部 exec S19 前半段 ⇒ 同 tokenizer / 同 build_batch /
#    greedy_gen(EM_BATCH=1) / parse_ans / 旧 add_3d test 复刻）
# ============================================================================
_src53 = open(f"{ROOT}/stages/53_lr_schedule.py", encoding="utf-8").read()
_cut53 = _src53.index("# ---- ④a")
NS: dict = {"__name__": "s53_head", "__file__": f"{ROOT}/stages/53_lr_schedule.py"}
exec(compile(_src53[:_cut53], "s53_head", "exec"), NS)              # noqa: S102

import torch  # noqa: E402

device = NS["device"]
assert device == "cuda", f"★device 必须 cuda，实得 {device}"
SEEDS = tuple(NS["SEEDS53"])
SMOKE = bool(NS["SMOKE"])
STEPS = NS["STEPS53"]
GRID = tuple(NS["GRID"])
SAVE_ATS = tuple(NS["SAVE_ATS"])
N_EVAL = NS["N_EVAL"]
D, FF, NHEAD, MAXLEN = NS["D"], NS["FF"], NS["NHEAD"], NS["MAXLEN"]
BATCH = NS["BATCH"]
EM_BATCH, CHK_BATCH = NS["EM_BATCH"], NS["CHK_BATCH"]
Cards = NS["Cards53"]
em_at = NS["em_at"]
frac, sha16, paired_delta, verdict = NS["frac"], NS["sha16"], NS["paired_delta"], NS["verdict"]
eval_states, r28_selfcheck = NS["eval_states"], NS["r28_selfcheck"]
OLD_TEST, OLD_TEST_SHA = NS["TEST"], NS["TEST_SHA"]
OLD_CLEAN_IDX = list(NS["CLEAN_IDX"])
OLD_BAD_IDX = list(NS["BAD_IDX"])
OLD_EVAL = OLD_TEST[:N_EVAL]
if not SMOKE:
    assert OLD_TEST_SHA == "b34e7ea515203227", f"旧 test 指纹不符：{OLD_TEST_SHA}"
NEW_SHA_EXPECT = "fa03ef9287567510"

LOG = "/tmp/59_smoke.log" if SMOKE else f"{ROOT}/logs/59_const_lr_newdata.log"
JSONL = "/tmp/59_results_smoke.jsonl" if SMOKE else f"{ROOT}/logs/59_results.jsonl"
CKPT_TMPL = ("/tmp/59smoke_ckpt_s{steps}_seed{seed}.pt" if SMOKE
             else f"{ROOT}/logs/59_ckpt_constlr_newdata_s{{steps}}_seed{{seed}}.pt")
CK38 = f"{ROOT}/logs/38_ckpt_w1_s{{steps}}_seed{{seed}}.pt"
CK53 = f"{ROOT}/logs/53_ckpt_lrcosine_s{{steps}}_seed{{seed}}.pt"
CK58 = f"{ROOT}/logs/58_ckpt_fixeddata_s{{steps}}_seed{{seed}}.pt"
FIX_JSONL = f"{ROOT}/logs/58_results.jsonl"

_log_f = open(LOG, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log_f.write(s + "\n")
    _log_f.flush()


say(f"[S59] device={device} {torch.cuda.get_device_name(0)} torch={torch.__version__} | "
    f"d={D} ff={FF} NHEAD={NHEAD} batch={BATCH} MAXLEN={MAXLEN} | 线程={torch.get_num_threads()}(R40) | "
    f"EM口径=batch{EM_BATCH}(R28) 自检=batch{CHK_BATCH} | smoke={SMOKE} | seeds={SEEDS}")
say(f"[S59-DESIGN] ★唯一变量 = LR 调度：常量 1e-3（无 warmup、无 cosine）｜数据=已修 S14 新 add_3d｜"
    f"结构/batch/步数={STEPS}/seed/评估口径逐字同 S58 FIX 臂")
say(f"[S59-OLD] 修前 add_3d test 复刻 sha256[:16]={OLD_TEST_SHA}（bad={len(OLD_BAD_IDX)} "
    f"clean={len(OLD_CLEAN_IDX)}）⇒ 与 S53/S38 同一旧 test 集")

# ============================================================================
# ② 新世界数据：exec 已修 S14 到 ② 模型 之前（五桶重生成）⇒ 覆盖 S53 的数据口径全局量
# ============================================================================
_src14 = open(f"{ROOT}/stages/14_synth_arith.py", encoding="utf-8").read()
_cut14 = _src14.index("# ② 模型")
NS14: dict = {"__name__": "s14_head", "__file__": f"{ROOT}/stages/14_synth_arith.py"}
exec(compile(_src14[:_cut14], "s14_head", "exec"), NS14)            # noqa: S102
NEW = NS14["DATA"]
BUCKETS = list(NS["BUCKETS"])
_bad_tr = {b: sum(1 for r in NEW[b]["train"] if "{expr}" in r["nl"]) for b in BUCKETS}
_bad_te = {b: sum(1 for r in NEW[b]["test"] if "{expr}" in r["nl"]) for b in BUCKETS}
NEW_TRAIN: list = NEW["add_3d"]["train"]
NEW_TEST: list = NEW["add_3d"]["test"]


def prot_sha(recs) -> str:
    h = hashlib.sha256()
    for r in recs:
        h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
    return h.hexdigest()[:16]


NEW_SHA = prot_sha(NEW_TEST)
BAD_IDX = [i for i, r in enumerate(NEW_TEST) if "{expr}" in r["nl"]]
CLEAN_IDX = [i for i in range(len(NEW_TEST)) if i not in set(BAD_IDX)]
_pairs = {(r["a"], r["b"]) for r in NEW_TRAIN}
for r in NEW_TEST:
    r["seen"] = (r["a"], r["b"]) in _pairs
N_SEEN = sum(1 for r in NEW_TEST if r["seen"])
_c = NS["Counter"](r["gold"] for r in NEW_TRAIN)
_top1, _ = _c.most_common(1)[0]
_top5 = [v for v, _ in _c.most_common(5)]
FLOOR_1 = sum(1 for r in NEW_TEST if r["gold"] == _top1) / len(NEW_TEST)
FLOOR_5 = sum(1 for r in NEW_TEST if r["gold"] in _top5) / len(NEW_TEST)
NEW_EVAL = NEW_TEST[:N_EVAL]

say(f"[S59-BAD] 修后五桶 bad_train={[_bad_tr[b] for b in BUCKETS]} "
    f"bad_test={[_bad_te[b] for b in BUCKETS]} ⇒ "
    f"{'✓ 全 0' if all(v == 0 for v in list(_bad_tr.values()) + list(_bad_te.values())) else '★非 0'}")
say(f"[S59-CHK1] ★新 TEST sha256[:16]={NEW_SHA} vs 任务书 {NEW_SHA_EXPECT} ⇒ "
    f"{'逐位一致 ✓' if NEW_SHA == NEW_SHA_EXPECT else '★不一致'} | (a,b) 泄漏对={N_SEEN}/"
    f"{len(NEW_TEST)} | SEEN={N_SEEN} UNSEEN={len(NEW_TEST)-N_SEEN}")
say(f"[S59-CHK2] 新 clean 口径：nl 含 '{{expr}}' 者={len(BAD_IDX)} ⇒ clean n={len(CLEAN_IDX)} "
    f"= 全量 {len(NEW_TEST)} ⇒ ★clean 口径已无意义（全量≡clean）")
say(f"[S59-CHK3] 新数据地板：单答案 '{_top1}'={FLOOR_1*100:.2f}% top5={FLOOR_5*100:.2f}% "
    f"（旧 S53 口径 0.12%/0.25%）")
if not SMOKE:
    assert NEW_SHA == NEW_SHA_EXPECT, "★新数据口径不符 ⇒ 本单元无效"
    assert (len(NEW_TRAIN), len(NEW_TEST), len(BAD_IDX)) == (4000, 800, 0), "新数据形状异常"

# ---- 覆盖 S53 的数据全局量（本臂在新数据上训练/评估）；并把 lr_at 换成常量 ----
NS.update(TEST=NEW_TEST, TEST_EVAL=NEW_EVAL, TRAIN=NEW_TRAIN, BAD_IDX=BAD_IDX,
          CLEAN_IDX=CLEAN_IDX, TEST_SHA=NEW_SHA, FLOOR_1=FLOOR_1, FLOOR_5=FLOOR_5,
          CKPT_TMPL=CKPT_TMPL, SAVE_ATS=SAVE_ATS, GRID=GRID, JSONL=JSONL)


def lr_at_const(t: int, steps: int, warmup: int, base: float) -> float:
    """★本单元唯一变量：恒定为 base（= 1e-3），无 warmup、无 cosine。"""
    return base


NS["lr_at"] = lr_at_const
assert NS["train53"].__globals__ is NS, "train53 的全局命名空间不是 NS ⇒ lr_at 覆盖会失效"
_base = NS["BASE_LR"]
assert _base == 1e-3, f"base lr 非 1e-3：{_base}"
assert all(NS["lr_at"](t, STEPS, 500, _base) == 1e-3 for t in (0, 1, 500, 6000, STEPS - 1))
say(f"[S59-LR] 采样 lr：t=0/1/500/{STEPS//4}/{STEPS//2}/{3*STEPS//4}/{STEPS-1} ⇒ "
    + " / ".join(f"{NS['lr_at'](t, STEPS, 500, _base):.3e}"
                 for t in (0, 1, 500, STEPS // 4, STEPS // 2, 3 * STEPS // 4, STEPS - 1))
    + "（★常量；train53 日志标签 'lrcos' 来自未改的 S53 源码，实际无 cosine）")
say(f"[S59-DATA] 新 add_3d: train={len(NEW_TRAIN)} test={len(NEW_TEST)}(eval {len(NEW_EVAL)}) | "
    f"旧 test: n={len(OLD_EVAL)} clean={len(OLD_CLEAN_IDX)}")

# ============================================================================
# ③ 已入库的 3 组 ckpt × 2 test 集 × 2 seed：逐样本配对（batch=1，显式 eval）
# ============================================================================
ev = Cards(D, FF).to(device)


def load_state(path: str):
    return torch.load(path, map_location="cpu", weights_only=True)


def hits_on(state, recs) -> list:
    ev.load_state_dict({k: v.to(device) for k, v in state.items()})
    ev.eval()                                                       # ★R39
    h, _ = em_at(ev, recs)                                          # ★R28 batch=1
    return h


H: dict = {}
for tag, tmpl in (("const_old", CK38), ("cos_old", CK53), ("cos_new", CK58)):
    for s in SEEDS:
        p = tmpl.format(steps=12000, seed=s)
        assert os.path.exists(p), f"缺 ckpt {p}"
        st = load_state(p)
        old_h = hits_on(st, OLD_EVAL)
        new_h = hits_on(st, NEW_EVAL)
        H[(tag, s, "old")] = old_h
        H[(tag, s, "new")] = new_h
        old_cl = frac([old_h[i] for i in OLD_CLEAN_IDX if i < len(OLD_EVAL)])
        say(f"[S59-EVAL] {tag} seed={s} {os.path.basename(p)} sha256[:16]={sha16(p)} | "
            f"旧test 全量={frac(old_h)*100:.2f}% clean={old_cl*100:.2f}% | "
            f"新test 全量={frac(new_h)*100:.2f}%")


def anchor(name, got, exp, tol=0.5):
    d = (got - exp) * 100
    say(f"[S59-ANCHOR] {name}: 本次={got*100:.2f}% vs 入库 {exp*100:.2f}% ⇒ Δ={d:+.2f}pp ⇒ "
        f"{'复现 ✓' if abs(d) <= tol else '★不复现，点名'}")


ANCHOR_OK = True
if not SMOKE:
    for s in SEEDS:
        anchor(f"53_ckpt@新test 全量 s{s}", frac(H[("cos_old", s, "new")]), 0.9625)
    anchor("58_ckpt@新test 全量 s1234", frac(H[("cos_new", 1234, "new")]), 0.97375)
    anchor("58_ckpt@新test 全量 s5678", frac(H[("cos_new", 5678, "new")]), 0.96875)
    anchor("38_ckpt@旧test 全量 s1234", frac(H[("const_old", 1234, "old")]), 0.9188)
    anchor("38_ckpt@旧test 全量 s5678", frac(H[("const_old", 5678, "old")]), 0.8712)
    for s in SEEDS:
        anchor(f"53_ckpt@旧test 全量 s{s}", frac(H[("cos_old", s, "old")]), 0.9362)

# ---- FIX（cosine+新数据，S58 入库）曲线，供同点对照 ----
FIX = {}
if os.path.exists(FIX_JSONL):
    for ln in open(FIX_JSONL, encoding="utf-8"):
        d = json.loads(ln)
        FIX[d["seed"]] = d

# ============================================================================
# ④ 本臂：常量 lr × 新数据，2 seed，跑满 STEPS（train53 逐字同款，仅 lr_at 与 ckpt 路径不同）
# ============================================================================
RESULTS: dict = {}
ARM_NEW_HITS: dict = {}
ARM_OLD_HITS: dict = {}
open(JSONL, "w").close()
for s in SEEDS:
    t0 = time.time()
    say(f"\n[S59] ===== seed={s} 常量 lr=1e-3 × 新数据：{STEPS} 步 "
        f"train={len(NEW_TRAIN)} test={len(NEW_EVAL)} 曲线格={len(GRID)} =====")
    model, wall_train, snaps, ckpts, inst, blk = NS["train53"](s, NEW_TRAIN, STEPS, GRID, SAVE_ATS)
    say(f"[S59] seed={s} 训练结束 墙钟={wall_train/60:.2f}min | ckpt={sorted(ckpts)}")
    curve = eval_states(ev, snaps, GRID, f"CONST s{s}")
    fin = GRID[-1]
    hits_new = curve[fin]["hits"]
    ARM_NEW_HITS[s] = hits_new
    ARM_OLD_HITS[s] = hits_on(snaps[fin], OLD_EVAL)
    ev.load_state_dict({k: v.to(device) for k, v in snaps[fin].items()})
    ev.eval()                                                       # ★R39
    _r28 = r28_selfcheck(ev, NEW_EVAL, tag=f"[const s{fin} seed{s}] ",
                         k=min(CHK_BATCH, len(NEW_EVAL)))
    ckpt_meta = {st: dict(path=ckpts[st], sha=sha16(ckpts[st])) for st in ckpts}
    _s9 = 9000 if 9000 in GRID else GRID[max(0, len(GRID) - 2)]
    _s12 = 12000 if 12000 in GRID else GRID[-1]
    sat = (curve[_s12]["em_full"] - curve[_s9]["em_full"]) * 100
    n_ev = len(NEW_EVAL)
    emf = curve[fin]["em_full"]
    RESULTS[s] = dict(
        seed=s, steps=STEPS, wall_train=wall_train, wall_total=time.time() - t0, n_eval=n_ev,
        em_final=emf, em_final_clean=curve[fin]["em_clean"],
        em_se=math.sqrt(max(emf * (1 - emf), 0.0) / n_ev),
        em_seen=curve[fin]["em_seen"], em_unseen=curve[fin]["em_unseen"],
        dig_acc=curve[fin]["dig_acc"], ce=curve[fin]["ce"], sat=sat,
        arm_old=frac(ARM_OLD_HITS[s]),
        r28_same=_r28["same"], r28_k=_r28["k"], ckpt=ckpt_meta,
        train_loss={str(k): v for k, v in inst.items()},
        train_loss_blk={str(k): v for k, v in blk.items()},
        curve={str(st): {k: v for k, v in curve[st].items() if k != "hits"} for st in GRID})
    say(f"[S59-RUN] seed={s}: ★新数据 EM全量@{fin}={emf*100:.2f}%±{RESULTS[s]['em_se']*100:.2f}pp "
        f"clean={curve[fin]['em_clean']*100:.2f}% SEEN={curve[fin]['em_seen']*100:.2f}% "
        f"UNSEEN={curve[fin]['em_unseen']*100:.2f}% | 数字每步={curve[fin]['dig_acc']*100:.2f}% "
        f"整体CE={curve[fin]['ce']:.4f} | 本臂@旧test={RESULTS[s]['arm_old']*100:.2f}% | "
        f"R28={_r28['same']}/{_r28['k']} | 饱和9000→12000={sat:+.2f}pp | "
        f"合计墙钟={RESULTS[s]['wall_total']/60:.2f}min")
    del snaps, model
    torch.cuda.empty_cache()
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(RESULTS[s], ensure_ascii=False) + "\n")

for s in SEEDS:
    H[("const_new", s, "new")] = ARM_NEW_HITS[s]
    H[("const_new", s, "old")] = ARM_OLD_HITS[s]

# ============================================================================
# ⑤ 汇总：2×2 表（含交叉格）+ 调度效应 / 数据效应 / 交互（全部配对 Δ）
# ============================================================================
TAGS = {"const_old": "常量lr×旧数据(38)", "cos_old": "cosine×旧数据(53)",
        "const_new": "常量lr×新数据(本臂)", "cos_new": "cosine×新数据(58FIX)"}


def cells(tag: str, tname: str) -> list:
    return [H[(tag, s, tname)] for s in SEEDS]


def pair_summary(name, a_list, b_list, tname):
    ms, ses = [], []
    for a, b in zip(a_list, b_list):
        m, se, n = paired_delta(a, b)
        ms.append(m)
        ses.append(se)
    same = all(m > 0 for m in ms) or all(m < 0 for m in ms)
    mean = sum(ms) / len(ms)
    return dict(name=name, tname=tname, ms=ms, ses=ses, mean=mean, same=same,
                txt=" ; ".join(f"s{SEEDS[i]}: {ms[i]*100:+.2f}pp±{ses[i]*100:.2f}pp "
                               f"({verdict(ms[i], ses[i])})" for i in range(len(ms))))


say(f"\n[S59-2x2] ★四格（各自在【同数据版】的 test 上；格式 s1234/s5678 全量/clean，bad=0 ⇒ clean≡全量）")
for tag in ("const_old", "cos_old", "const_new", "cos_new"):
    row = []
    for s in SEEDS:
        h = H[(tag, s, "old")] if tag in ("const_old", "cos_old") else H[(tag, s, "new")]
        cl = OLD_CLEAN_IDX if tag in ("const_old", "cos_old") else CLEAN_IDX
        row.append(f"{frac(h)*100:.2f}/{frac([h[i] for i in cl if i < len(h)])*100:.2f}")
    meanv = sum(frac(H[(tag, s, "old" if tag in ("const_old", "cos_old") else "new")])
                for s in SEEDS) / len(SEEDS)
    say(f"[S59-2x2] {TAGS[tag]:<20} | {row[0]} | {row[1]} | 均值全量={meanv*100:.2f}%")

say(f"\n[S59-CROSS] ★交叉格（同一 ckpt 换到【另一版本】的 test 上；跨数据版不比绝对 EM，仅供配对用）")
for tag in ("const_old", "cos_old", "const_new", "cos_new"):
    other = "new" if tag in ("const_old", "cos_old") else "old"
    vals = [frac(H[(tag, s, other)]) for s in SEEDS]
    say(f"[S59-CROSS] {TAGS[tag]:<20} @{'新' if other == 'new' else '旧'}test 全量 = "
        f"{vals[0]*100:.2f}% / {vals[1]*100:.2f}%")

# ---- 调度效应 = cos − 常量（同一训练数据、同一 test 集） ----
say(f"\n[S59-SCHED] ★调度效应 = EM(cosine) − EM(常量lr)，同数据版、同 test 集、逐样本配对")
SCHED = {}
for tname, tlabel in (("new", "新数据版（在【新】test 上）"), ("old", "旧数据版（在【旧】test 上）")):
    r = pair_summary("sched", cells("cos_new", tname), cells("const_new", tname), tname) \
        if tname == "new" else pair_summary("sched", cells("cos_old", tname),
                                            cells("const_old", tname), tname)
    SCHED[tname] = r
    say(f"[S59-SCHED] {tlabel}: {r['txt']} | 均值={r['mean']*100:+.2f}pp | "
        f"同号={'是' if r['same'] else '否'}")
# 任务书口径：旧数据上的调度效应用 S53/S38 的 clean 口径表述
_s_old_clean = []
for i, s in enumerate(SEEDS):
    a = [H[("cos_old", s, "old")][j] for j in OLD_CLEAN_IDX if j < len(OLD_EVAL)]
    b = [H[("const_old", s, "old")][j] for j in OLD_CLEAN_IDX if j < len(OLD_EVAL)]
    m, se, n = paired_delta(a, b)
    _s_old_clean.append((s, m, se))
say("[S59-SCHED] 旧数据版 clean(n=770，任务书写死的对照口径，来自 S53−S38 同源): "
    + " ; ".join(f"s{s}: {m*100:+.2f}pp±{se*100:.2f}pp" for s, m, se in _s_old_clean)
    + f" | 均值={sum(m for _, m, _ in _s_old_clean)/len(_s_old_clean)*100:+.2f}pp")

# ---- 数据效应 = 新数据 − 旧数据（同一调度、同一 test 集） ----
say(f"\n[S59-DATA] ★数据效应 = EM(新数据 ckpt) − EM(旧数据 ckpt)，同调度、同 test 集、逐样本配对")
DATA_EFF = {}
for tname, tlabel in (("new", "在【新】test 上"), ("old", "在【旧】test 上")):
    rc = pair_summary("data_cos", cells("cos_new", tname), cells("cos_old", tname), tname)
    rk = pair_summary("data_const", cells("const_new", tname), cells("const_old", tname), tname)
    DATA_EFF[("cos", tname)] = rc
    DATA_EFF[("const", tname)] = rk
    say(f"[S59-DATA] {tlabel} | cosine 下的数据效应: {rc['txt']} | 均值={rc['mean']*100:+.2f}pp")
    say(f"[S59-DATA] {tlabel} | 常量lr 下的数据效应: {rk['txt']} | 均值={rk['mean']*100:+.2f}pp")

# ---- 交互 = 调度效应(新数据) − 调度效应(旧数据)，逐样本四格对比（配对 SE） ----
say(f"\n[S59-INTER] ★交互 = [EM(cos,新)−EM(常量,新)] − [EM(cos,旧)−EM(常量,旧)]，逐样本配对")
for tname, tlabel in (("new", "四格全在【新】test 上（严格可配）"),
                      ("old", "四格全在【旧】test 上（严格可配）")):
    ms, ses = [], []
    for s in SEEDS:
        dA = [a - b for a, b in zip(H[("cos_new", s, tname)], H[("const_new", s, tname)])]
        dB = [a - b for a, b in zip(H[("cos_old", s, tname)], H[("const_old", s, tname)])]
        m, se, n = paired_delta(dA, dB)
        ms.append(m)
        ses.append(se)
    same = all(m > 0 for m in ms) or all(m < 0 for m in ms)
    sig = all(abs(m) >= 2 * se for m, se in zip(ms, ses))
    say(f"[S59-INTER] {tlabel}: " + " ; ".join(
        (f"s{SEEDS[i]}: {ms[i]*100:+.2f}pp±{ses[i]*100:.2f}pp "
         f"({(ms[i]/ses[i]) if ses[i] > 0 else float('nan'):+.2f}SE)") for i in range(2))
        + f" | 均值={sum(ms)/2*100:+.2f}pp | 同号={'是' if same else '否'} | "
        f"逐 seed 均≥2SE={'是' if sig else '否'} ⇒ "
        f"{'★存在交互' if same and sig else ('无显著交互（可加）' if not sig else '异号/不显著 ⇒ 不判定')}")
# 任务书字面口径（跨 test 集，不可比 ⇒ 仅作参考，SE 用独立近似）
ms, ses = [], []
for i, s in enumerate(SEEDS):
    m1 = SCHED["new"]["ms"][i]
    se1 = SCHED["new"]["ses"][i]
    m2 = _s_old_clean[i][1]
    se2 = _s_old_clean[i][2]
    ms.append(m1 - m2)
    ses.append(math.sqrt(se1 ** 2 + se2 ** 2))
say("[S59-INTER] 任务书字面口径（新数据@新test − 旧数据@旧test clean；★跨 test 集不可比，仅参考）: "
    + " ; ".join(f"s{SEEDS[i]}: {ms[i]*100:+.2f}pp±{ses[i]*100:.2f}pp" for i in range(2))
    + f" | 均值={sum(ms)/2*100:+.2f}pp")

# ---- 曲线对照（本臂 vs FIX，同格点） ----
say(f"\n[S59-CURVE] ★曲线对照（全量 EM，每 750 步一格中的 4 个关键格点）")
for st in (3000, 6000, 9000, 12000):
    if st not in GRID:
        continue
    a = [RESULTS[s]["curve"][str(st)]["em_full"] * 100 for s in SEEDS]
    fx = [FIX[s]["curve"][str(st)]["em_full"] * 100 if s in FIX and str(st) in FIX[s]["curve"] else float("nan")
          for s in SEEDS]
    say(f"[S59-CURVE] step={st:>5} | 本臂(常量,新)={a[0]:.2f}/{a[1]:.2f}% (均值 {sum(a)/2:.2f}%) | "
        f"FIX(cos,新)={fx[0]:.2f}/{fx[1]:.2f}% (均值 {sum(fx)/2:.2f}%) | 差(cos−const)="
        f"{fx[0]-a[0]:+.2f}/{fx[1]-a[1]:+.2f}pp")
_front = {"本臂(常量,新)": [RESULTS[s]["curve"] for s in SEEDS],
          "FIX(cos,新)": [FIX[s]["curve"] for s in SEEDS] if FIX else []}
for name, cs in _front.items():
    if not cs:
        continue
    first = []
    for c in cs:
        f = next((st for st in GRID
                  if c.get(str(st), {}).get("em_full", -1.0) >= 0.5), None)
        first.append(f)
    say(f"[S59-SPEED] {name}: 首次 EM≥50% 的格点 = {first}（越小=收敛越快）")

# ---- 饱和检查 R35 ----
say(f"\n[S59-SAT] ★饱和检查（R35）：本臂 9000→12000 全量 EM 增量")
for s in SEEDS:
    c = RESULTS[s]["curve"]
    _a = str(9000) if str(9000) in c else str(GRID[max(0, len(GRID) - 2)])
    _b = str(12000) if str(12000) in c else str(GRID[-1])
    d = (c[_b]["em_full"] - c[_a]["em_full"]) * 100
    say(f"[S59-SAT] seed={s}: {_a} 全量={c[_a]['em_full']*100:.2f}% → {_b} "
        f"全量={c[_b]['em_full']*100:.2f}% | Δ={d:+.2f}pp ⇒ "
        f"{'★未饱和(≥1pp) ⇒ 结论条件性' if d >= 1.0 else '已平(<1pp)'}")
_sat_mean = sum(RESULTS[s]["sat"] for s in SEEDS) / len(SEEDS)
say(f"[S59-SAT-VERDICT] 2 seed 均值 Δ={_sat_mean:+.2f}pp ⇒ "
    f"{'★未饱和：本单元结论按 [条件性] 记' if _sat_mean >= 1.0 else '已饱和'}")

# ---- 成本 / ckpt / 元数据 ----
say(f"\n[S59] ===== 成本与 ckpt =====")
for s in SEEDS:
    r = RESULTS[s]
    for st in sorted(r["ckpt"]):
        say(f"[META] seed={s}: ckpt={os.path.basename(r['ckpt'][st]['path'])} "
            f"sha256[:16]={r['ckpt'][st]['sha']}")
    say(f"[META] seed={s}: 训练墙钟={r['wall_train']/60:.2f}min 本run={r['wall_total']/60:.2f}min")
say(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
    f"线程={torch.get_num_threads()}(R40) | 新TEST sha16={NEW_SHA}(bad={len(BAD_IDX)}) "
    f"旧TEST sha16={OLD_TEST_SHA}(bad={len(OLD_BAD_IDX)}) | 地板={FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% | "
    f"总墙钟={(time.time()-T0)/60:.1f}min")
say("[DONE] exit=0")
sys.exit(0)
