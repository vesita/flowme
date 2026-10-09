#!/usr/bin/env python3
"""S60 · ★把「新数据 + warmup500+cosine→0」钉死到饱和：@24000（每 750 步快照）。

背景（S59 暴露的未决问题）：
  - 现有最好成绩 = 新数据 + warmup500+cosine→0 @12000（S58 FIX）⇒ 2 seed 97.38 / 96.88（均值 97.12%）。
  - 但 S57/S58 的饱和检查未过：s1234 9000→12000 = +1.38pp（≥1pp ⇒ 它自己标了【条件性】）。
  - S59 的（常量 lr × 新数据 @12000）也未饱和（+2.38pp 均值）。
  ⇒ 所以现在不能回答「97.12% 是不是新数据的上限」，也不能可靠地做 2×2。
  ⇒ 本单元 = 把 cosine 这条跑到 @24000（每 750 步快照）⇒ 回答：
    ① 新数据上 cosine 的饱和点与上限；② 97.12% 是否是天花板；
    ③ 补齐 2×2 的「新数据」列（@12000 已有，@24000 给「已饱和」版）。

设计（只加步数；其余配置逐字同 S58 的 FIX 臂）：
  - 臂 = 新数据 + warmup500 + cosine→0，总步数 24000 ⇒ cosine 周期随总步数走
    （lr_at(t, 24000, 500, 1e-3)；t=24000 时 lr≈0）。
  - seed = 1234 / 5678 ⇒ 2 run；数据 = 已修 stages/14_synth_arith.py 后重生成的 add_3d
    （新 TEST sha256[:16] = fa03ef9287567510）。
  - 结构 d=128 / ff=512 / nhead=4 / batch=32 / MAXLEN=512 / add_3d 与 S57/S53 逐字同
    （模型与训练函数直接 exec S53 源码头 ⇒ 同初始化、同数据顺序、同 dropout 消耗）。
  - ★ 与 S58 完全同轨迹（同 seed ⇒ 同初始化 + 同 torch.randperm 数据顺序），
    差别只在 lr 调度：cosine 周期 12000 → 24000，从 step 500 之后开始分叉。

判据（写死）：
  1 主判据（饱和 R35）：@21000→@24000 增量 <1pp ⇒ 已饱和 ⇒ 可答「新数据 cosine 的上限」；
    ≥1pp ⇒ 标条件性 + 给「还需多少步」的外推估计。
  2 上限：@12000/15000/18000/21000/24000 的 EM 曲线（2 seed）+ 最好格点的 EM +
    两 seed 极差随步数的变化（S57 FIX @12000 = 0.50pp；S53 旧数据 @12000 = 0.00pp）。
  3 Δ = EM(@24000) − EM(@12000)（★本臂内，逐样本配对 ±SE）：≥+2SE 且 2 seed 同号 ⇒ 12000 步不够；
    |Δ|≤1SE ⇒ 已到顶。另报 Δ = EM(本臂@24000) − EM(58 ckpt@12000)（现有最好成绩的终点）。
  4 2×2「新数据」列更新；★R35：只有已饱和的格才能用于交互判断（如实写，不拿未饱和格做交互）。
  5 新数据绝对 EM 上限（供 METRICS.md 新数据栏）。

口径：EM = batch=1（R28）+ 显式 eval()（R39）+ 4 线程（R40）；新 test n=800、bad=0 ⇒ clean ≡ 全量；
  TEST sha256[:16] 复认 fa03ef9287567510；地板 0.12%/0.25%；SEEN/UNSEEN 分层；
  ★锚点复核：58_ckpt_fixeddata_s12000 在新 test 上应为 97.38 / 96.88（Δ>0.5pp 点名）。

★写键卫生（S59 的 KeyError: 12000 教训）：曲线字典键一律 str()，读取一律走 rec_of()。
只允许写：本文件、logs/s60_newdata24k.log（systemd 重定向）、logs/60_*、/tmp。
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

SMOKE = os.environ.get("S60_SMOKE") == "1"
if SMOKE:
    os.environ["S53_SMOKE"] = "1"
    os.environ["S19_SMOKE"] = "1"

# ============================================================================
# ① 旧世界的「评估机器」：exec S53 源码到 ④a 之前
#    （内部 exec S19 前半段 ⇒ 同 tokenizer / build_batch / greedy_gen(EM_BATCH=1) /
#     parse_ans / paired_delta / r28_selfcheck；同时复刻修前 add_3d test）
# ============================================================================
_src53 = open(f"{ROOT}/stages/53_lr_schedule.py", encoding="utf-8").read()
_cut53 = _src53.index("# ---- ④a")
NS: dict = {"__name__": "s53_head", "__file__": f"{ROOT}/stages/53_lr_schedule.py"}
exec(compile(_src53[:_cut53], "s53_head", "exec"), NS)              # noqa: S102

import torch  # noqa: E402

device = NS["device"]
assert device == "cuda", f"★device 必须 cuda，实得 {device}"
SEEDS = tuple(NS["SEEDS53"])
D, FF, NHEAD, MAXLEN, BATCH = NS["D"], NS["FF"], NS["NHEAD"], NS["MAXLEN"], NS["BATCH"]
EM_BATCH, CHK_BATCH = NS["EM_BATCH"], NS["CHK_BATCH"]
WARMUP, BASE_LR = NS["WARMUP"], NS["BASE_LR"]
N_EVAL = NS["N_EVAL"]
Cards, em_at = NS["Cards53"], NS["em_at"]
frac, sha16 = NS["frac"], NS["sha16"]
paired_delta, verdict = NS["paired_delta"], NS["verdict"]
eval_states, r28_selfcheck = NS["eval_states"], NS["r28_selfcheck"]
OLD_TEST_SHA, OLD_BAD_IDX = NS["TEST_SHA"], list(NS["BAD_IDX"])
assert NS["train53"].__globals__ is NS, "train53 的全局命名空间不是 NS"
NEW_SHA_EXPECT = "fa03ef9287567510"


def say(*a):
    print(" ".join(str(x) for x in a), flush=True)


# ---- 本单元配置（唯一改动 = 总步数 12000 → 24000） ----
STEPS = 90 if SMOKE else 24000
GRID = tuple(range(30, 91, 30)) if SMOKE else tuple(range(750, 24001, 750))
SAVE_ATS = (30, 60, 90) if SMOKE else (9000, 12000, 15000, 18000, 21000, 24000)
KEY_STEPS = tuple(s for s in (12000, 15000, 18000, 21000, 24000) if s in GRID) or tuple(GRID[-3:])
DELTA_A, DELTA_B = (12000, 24000) if not SMOKE else (GRID[0], GRID[-1])
SAT_A, SAT_B = (21000, 24000) if not SMOKE else (GRID[-2], GRID[-1])
CKPT_TMPL = ("/tmp/s60smoke_ckpt_s{steps}_seed{seed}.pt" if SMOKE
             else f"{ROOT}/logs/60_ckpt_newdata24k_s{{steps}}_seed{{seed}}.pt")
JSONL = "/tmp/60_results_smoke.jsonl" if SMOKE else f"{ROOT}/logs/60_results.jsonl"
C58 = f"{ROOT}/logs/58_ckpt_fixeddata_s{{steps}}_seed{{seed}}.pt"
C53 = f"{ROOT}/logs/53_ckpt_lrcosine_s{{steps}}_seed{{seed}}.pt"

for _st in (DELTA_A, DELTA_B, SAT_A, SAT_B):
    assert _st in GRID, f"格点 {_st} 不在曲线格里"

say(f"[S60] device={device} {torch.cuda.get_device_name(0)} torch={torch.__version__} | "
    f"d={D} ff={FF} NHEAD={NHEAD} batch={BATCH} MAXLEN={MAXLEN} | 线程={torch.get_num_threads()}(R40) | "
    f"EM口径=batch{EM_BATCH}(R28) 自检=batch{CHK_BATCH} | smoke={SMOKE} | seeds={SEEDS}")
say(f"[S60-DESIGN] ★唯一改动 = 总步数 {STEPS}（cosine 周期随总步数走）｜数据=已修 S14 新 add_3d｜"
    f"结构/batch/warmup{WARMUP}/base_lr={BASE_LR}/seed/评估口径 逐字同 S58 FIX 臂｜"
    f"曲线格={GRID[0]}..{GRID[-1]}/{GRID[1]-GRID[0]}步（{len(GRID)} 格）｜存盘步={SAVE_ATS}")
say(f"[S60-OLD] 修前 add_3d test 复刻 sha256[:16]={OLD_TEST_SHA}（bad={len(OLD_BAD_IDX)}）"
    f" ⇒ 与 S53/S58 同一旧 test 集")
_lr_probe = [(t, NS["lr_at"](t, STEPS, WARMUP, BASE_LR))
             for t in (0, WARMUP // 2, WARMUP, 3000, 12000, 18000, 21000, STEPS - 1)]
say("[S60-LR] 采样 lr（★cosine 周期 = 总步数 " + str(STEPS) + "）："
    + " / ".join(f"t={t}:{v:.3e}" for t, v in _lr_probe)
    + f" ⇒ t={STEPS - 1} lr={_lr_probe[-1][1]:.2e}（→0 ✓）")
if not SMOKE:
    assert NS["lr_at"](STEPS - 1, STEPS, WARMUP, BASE_LR) < 1e-9, \
        "★末步 lr 未衰减到 0 ⇒ 周期没随总步数走"
    assert NS["lr_at"](12000, STEPS, WARMUP, BASE_LR) > 1e-4, \
        "★@12000 的 lr 已到 0 ⇒ 周期没随总步数走（必须仍在上行/中段）"

# ============================================================================
# ② 新世界数据：exec 已修 stages/14_synth_arith.py 到 ② 模型 之前（五桶重生成）
# ============================================================================
_src14 = open(f"{ROOT}/stages/14_synth_arith.py", encoding="utf-8").read()
_cut14 = _src14.index("# ② 模型")
NS14: dict = {"__name__": "s14_head", "__file__": f"{ROOT}/stages/14_synth_arith.py"}
say(f"\n[S60-NEW] ===== 已修 S14 重生成五桶（新 [DATA] 见下）=====")
exec(compile(_src14[:_cut14], "s14_head", "exec"), NS14)            # noqa: S102
NEW = NS14["DATA"]
NEW_INFO = NS14["BUCKET_INFO"]
BUCKETS = list(NS["BUCKETS"])
NEW_TRAIN: list = NEW["add_3d"]["train"]
NEW_TEST: list = NEW["add_3d"]["test"]

_bad_tr = {b: sum(1 for r in NEW[b]["train"] if "{expr}" in r["nl"]) for b in BUCKETS}
_bad_te = {b: sum(1 for r in NEW[b]["test"] if "{expr}" in r["nl"]) for b in BUCKETS}


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

say(f"[S60-BAD] 修后五桶 bad_train={[_bad_tr[b] for b in BUCKETS]} "
    f"bad_test={[_bad_te[b] for b in BUCKETS]} ⇒ "
    f"{'✓ 全 0' if all(v == 0 for v in list(_bad_tr.values()) + list(_bad_te.values())) else '★非 0'}")
say(f"[S60-CHK1] ★新 TEST sha256[:16]={NEW_SHA} vs 任务书 {NEW_SHA_EXPECT} ⇒ "
    f"{'逐位一致 ✓' if NEW_SHA == NEW_SHA_EXPECT else '★不一致'} | (a,b) 泄漏对={N_SEEN}/"
    f"{len(NEW_TEST)} | SEEN={N_SEEN} UNSEEN={len(NEW_TEST)-N_SEEN}")
say(f"[S60-CHK2] 新 clean 口径：nl 含 '{{expr}}' 者={len(BAD_IDX)} ⇒ clean n={len(CLEAN_IDX)} "
    f"= 全量 {len(NEW_TEST)} ⇒ ★clean 口径已无意义（全量≡clean）")
say(f"[S60-CHK3] 新数据地板：单答案 '{_top1}'={FLOOR_1*100:.2f}% top5={FLOOR_5*100:.2f}% "
    f"（旧 S53 口径 0.12%/0.25%）| 干预面=无 ⇒ R29 不适用")
say(f"[S60-DATA] 新 add_3d: train={len(NEW_TRAIN)} test={len(NEW_TEST)}(eval {len(NEW_EVAL)}) "
    f"答案域={min(r['y'] for r in NEW_TRAIN)}..{max(r['y'] for r in NEW_TRAIN)}"
    f"({NEW_INFO['add_3d']['ans_n']}个) | 模板覆盖={NEW_INFO['add_3d']['tpl_used']}/358 "
    f"distinct(a,b)train={NEW_INFO['add_3d']['pair_tr']} test={NEW_INFO['add_3d']['pair_te']}")
assert NEW_SHA == NEW_SHA_EXPECT, "★新数据口径不符 ⇒ 本单元无效"
assert (len(NEW_TRAIN), len(NEW_TEST), len(BAD_IDX)) == (4000, 800, 0), "新数据形状异常"

# ---- 覆盖 S53 的数据全局量（本臂在新数据上训练/评估）；lr_at 保持 S53 原样（cosine） ----
NS.update(TEST=NEW_TEST, TEST_EVAL=NEW_EVAL, TRAIN=NEW_TRAIN, BAD_IDX=BAD_IDX,
          CLEAN_IDX=CLEAN_IDX, TEST_SHA=NEW_SHA, FLOOR_1=FLOOR_1, FLOOR_5=FLOOR_5,
          CKPT_TMPL=CKPT_TMPL, SAVE_ATS=SAVE_ATS, GRID=GRID, JSONL=JSONL)

ev = Cards(D, FF).to(device)


def rec_of(curve: dict, st: int) -> dict:
    """★统一 str 键读取（S59 的 KeyError: 12000 教训）。"""
    k = str(st)
    assert k in curve, (f"曲线缺格点 {st}；现有键（前 3）={sorted(curve)[:3]} "
                        f"共 {len(curve)} 个；★键必须统一 str")
    return curve[k]


# ============================================================================
# ③ 锚点复核：58 ckpt @12000（现有最好成绩）与 53 ckpt @12000 在新 test 上
# ============================================================================
def load_state(path: str):
    return torch.load(path, map_location="cpu", weights_only=True)


def hits_on(state, recs) -> list:
    ev.load_state_dict({k: v.to(device) for k, v in state.items()})
    ev.eval()                                                       # ★R39
    h, _ = em_at(ev, recs)                                          # ★R28 batch=1
    return h


ANCHOR_EXP = {("58", 1234): 0.97375, ("58", 5678): 0.96875,
              ("53", 1234): 0.9625, ("53", 5678): 0.9625}
H58: dict = {}
say(f"\n[S60-ANCHOR] ★锚点复核（新 test，batch=1，显式 eval）")
for s in SEEDS:
    for tag, tmpl in (("58", C58), ("53", C53)):
        p = tmpl.format(steps=12000, seed=s)
        assert os.path.exists(p), f"缺 ckpt {p}"
        h = hits_on(load_state(p), NEW_EVAL)
        if tag == "58":
            H58[s] = h
        got, exp = frac(h), ANCHOR_EXP[(tag, s)]
        d = (got - exp) * 100
        ok = "★不复现，点名" if abs(d) > 0.5 else "复现 ✓"
        say(f"[S60-ANCHOR] {tag}_ckpt@12000 seed={s} sha16={sha16(p)} | 新test EM={got*100:.2f}% "
            f"vs 入库 {exp*100:.2f}% ⇒ Δ={d:+.2f}pp ⇒ "
            f"{'（smoke：仅 60 样本，不作锚点判定）' if SMOKE else ok}")

# ============================================================================
# ④ 主实验：新数据 + warmup500+cosine→0 @24000，2 seed
# ============================================================================
RESULTS: dict = {}
open(JSONL, "w").close()
for s in SEEDS:
    t0 = time.time()
    say(f"\n[S60] ===== seed={s} 新数据 warmup{WARMUP}+cos→0 @{STEPS} "
        f"train={len(NEW_TRAIN)} test={len(NEW_EVAL)} 曲线格={len(GRID)} =====")
    model, wall_train, snaps, ckpts, inst, blk = NS["train53"](s, NEW_TRAIN, STEPS, GRID, SAVE_ATS)
    say(f"[S60] seed={s} 训练结束 墙钟={wall_train/60:.2f}min | ckpt={sorted(ckpts)}")
    curve = eval_states(ev, snaps, GRID, f"NEW24K s{s}")
    C = {str(st): rec for st, rec in curve.items()}                  # ★str 键
    fin = GRID[-1]
    hits_fin = C[str(fin)]["hits"]
    ev.load_state_dict({k: v.to(device) for k, v in snaps[fin].items()})
    ev.eval()                                                       # ★R39
    _r28 = r28_selfcheck(ev, NEW_EVAL, tag=f"[new24k s{fin} seed{s}] ",
                         k=min(CHK_BATCH, len(NEW_EVAL)))
    d_in = paired_delta(hits_fin, C[str(DELTA_A)]["hits"])
    d_sat = paired_delta(hits_fin, C[str(SAT_A)]["hits"])
    d_58 = paired_delta(hits_fin, H58[s])
    em_fin = C[str(fin)]["em_full"]
    ckpt_meta = {st: dict(path=ckpts[st], sha=sha16(ckpts[st])) for st in ckpts}
    RESULTS[s] = dict(
        seed=s, steps=STEPS, wall_train=wall_train, wall_total=time.time() - t0,
        n_eval=len(NEW_EVAL), em_final=em_fin, em_final_clean=C[str(fin)]["em_clean"],
        em_se=math.sqrt(max(em_fin * (1 - em_fin), 0.0) / len(NEW_EVAL)),
        em_seen=C[str(fin)]["em_seen"], em_unseen=C[str(fin)]["em_unseen"],
        dig_acc=C[str(fin)]["dig_acc"], ce=C[str(fin)]["ce"],
        d_in=(d_in[0], d_in[1]), d_sat=(d_sat[0], d_sat[1]),
        d_58=(d_58[0], d_58[1]), r28_same=_r28["same"], r28_k=_r28["k"], ckpt=ckpt_meta,
        train_loss={str(k): v for k, v in inst.items()},
        train_loss_blk={str(k): v for k, v in blk.items()},
        curve={st: {k: v for k, v in C[st].items() if k != "hits"} for st in C})
    say(f"[S60-RUN] seed={s}: EM全量@{fin}={em_fin*100:.2f}%±{RESULTS[s]['em_se']*100:.2f}pp "
        f"SEEN={C[str(fin)]['em_seen']*100:.2f}% UNSEEN={C[str(fin)]['em_unseen']*100:.2f}% | "
        f"Δ(@{DELTA_B}−@{DELTA_A})={d_in[0]*100:+.2f}pp±{d_in[1]*100:.2f}pp ⇒ "
        f"{verdict(d_in[0], d_in[1])} | "
        f"Δ({DELTA_B}−58@12000)={d_58[0]*100:+.2f}pp±{d_58[1]*100:.2f}pp ⇒ "
        f"{verdict(d_58[0], d_58[1])} | "
        f"饱和@{SAT_A}→@{SAT_B}={d_sat[0]*100:+.2f}pp±{d_sat[1]*100:.2f}pp | 数字每步="
        f"{C[str(fin)]['dig_acc']*100:.2f}% 整体CE={C[str(fin)]['ce']:.4f} | "
        f"R28={_r28['same']}/{_r28['k']} | 合计墙钟={RESULTS[s]['wall_total']/60:.2f}min")
    del snaps, model
    torch.cuda.empty_cache()
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(RESULTS[s], ensure_ascii=False) + "\n")

# ============================================================================
# ⑤ 汇总：曲线 / Δ / 饱和 / 上限与极差 / 2×2
# ============================================================================
def load_jsonl(p: str) -> dict:
    out: dict = {}
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln:
                d = json.loads(ln)
                out[d["seed"]] = d
    return out


S53 = load_jsonl(f"{ROOT}/logs/53_results.jsonl")
S58 = load_jsonl(f"{ROOT}/logs/58_results.jsonl")
S59 = load_jsonl(f"{ROOT}/logs/59_results.jsonl")

say(f"\n[S60-CURVE] ★全曲线（EM 全量%，batch=1；极差=两 seed 差）")
say("[S60-CURVE] step    | s1234  | s5678  | 均值   | 极差")
for st in GRID:
    v = [rec_of(RESULTS[s]["curve"], st)["em_full"] * 100 for s in SEEDS]
    mark = " ★" if st in KEY_STEPS else ""
    say(f"[S60-CURVE] {st:>7} | {v[0]:6.2f} | {v[1]:6.2f} | {sum(v)/2:6.2f} | {abs(v[0]-v[1]):5.2f}{mark}")

say(f"\n[S60-KEY] ★关键格点（@{DELTA_A}…@{DELTA_B}）与逐步增量")
prev = None
for st in KEY_STEPS:
    v = [rec_of(RESULTS[s]["curve"], st)["em_full"] * 100 for s in SEEDS]
    d = "" if prev is None else f" | Δ均值 vs @{prev}={sum(v)/2 - sum(pv)/2:+.2f}pp"
    if prev is not None:
        d += f" (s1234 {v[0]-pv[0]:+.2f} / s5678 {v[1]-pv[1]:+.2f})"
    say(f"[S60-KEY] @{st:>5} | s1234={v[0]:.2f}% s5678={v[1]:.2f}% | 均值={sum(v)/2:.2f}% "
        f"| 极差={abs(v[0]-v[1]):.2f}pp{d}")
    pv = v
    prev = st

# ---- 判据 3：Δ = EM(@24000) − EM(@12000)（本臂内，配对 ±SE） ----
say(f"\n[S60-DELTA] ★Δ = EM(@{DELTA_B}) − EM(@{DELTA_A})（本臂内，逐样本配对）")
ms = [RESULTS[s]["d_in"][0] for s in SEEDS]
ses = [RESULTS[s]["d_in"][1] for s in SEEDS]
same = all(m > 0 for m in ms) or all(m < 0 for m in ms)
ge2 = all(abs(m) >= 2 * x for m, x in zip(ms, ses))
le1 = all(abs(m) <= x for m, x in zip(ms, ses))
say(f"[S60-DELTA] " + " ; ".join(f"s{s}: {RESULTS[s]['d_in'][0]*100:+.2f}pp±"
                                f"{RESULTS[s]['d_in'][1]*100:.2f}pp ({verdict(*RESULTS[s]['d_in'])})"
                                for s in SEEDS)
    + f" | 均值={sum(ms)/2*100:+.2f}pp | 同号={'是' if same else '否'}")
say(f"[S60-DELTA-VERDICT] " + ("★12000 步不够（≥+2SE 且 2 seed 同号 ⇒ @24000 仍在上行）"
                               if (same and ge2 and ms[0] > 0) else
                               ("已到顶（|Δ|≤1SE ⇒ 12000 步足够）" if le1 else
                                "落在 1–2SE 之间/异号 ⇒ 不判定（不给二值结论）")))
say(f"[S60-DELTA58] ★Δ = EM(本臂@24000) − EM(58_ckpt@12000 现有最好成绩)（配对）: "
    + " ; ".join(f"s{s}: {RESULTS[s]['d_58'][0]*100:+.2f}pp±{RESULTS[s]['d_58'][1]*100:.2f}pp "
                 f"({verdict(*RESULTS[s]['d_58'])})" for s in SEEDS)
    + f" | 均值={sum(RESULTS[s]['d_58'][0] for s in SEEDS)/2*100:+.2f}pp | 同号="
    + ("是" if all(RESULTS[s]['d_58'][0] > 0 for s in SEEDS) or
       all(RESULTS[s]['d_58'][0] < 0 for s in SEEDS) else "否"))

# ---- 判据 1：饱和 R35 ----
say(f"\n[S60-SAT] ★饱和检查（R35，核心）：@{SAT_A}→@{SAT_B}")
for s in SEEDS:
    a = rec_of(RESULTS[s]["curve"], SAT_A)["em_full"] * 100
    b = rec_of(RESULTS[s]["curve"], SAT_B)["em_full"] * 100
    say(f"[S60-SAT] seed={s}: @{SAT_A} 全量={a:.2f}% → @{SAT_B} 全量={b:.2f}% | 增量={b-a:+.2f}pp "
        f"（配对 {RESULTS[s]['d_sat'][0]*100:+.2f}pp±{RESULTS[s]['d_sat'][1]*100:.2f}pp）⇒ "
        f"{'已饱和(<1pp)' if abs(b-a) < 1.0 else '★未饱和(≥1pp) ⇒ 标条件性'}")
_msat = sum(rec_of(RESULTS[s]["curve"], SAT_B)["em_full"] - rec_of(RESULTS[s]["curve"], SAT_A)["em_full"]
            for s in SEEDS) / len(SEEDS) * 100
_extrap = "—"
if abs(_msat) >= 1.0 and _msat > 0:
    _per750 = _msat / ((SAT_B - SAT_A) / 750)
    _extrap = f"按近 3000 步 {_msat:+.2f}pp/3000步 ≈ {_per750:+.2f}pp/750步 线性外推，再拿 <1pp 需 ≈ {math.ceil(1.0/abs(_per750))*750} 步（★纯外推，非实测）"
say(f"[S60-SAT-VERDICT] 2 seed 均值 @{SAT_A}→@{SAT_B} Δ={_msat:+.2f}pp ⇒ "
    f"{'已饱和 ⇒ 可答「新数据 cosine 的上限」' if abs(_msat) < 1.0 else '★未饱和 ⇒ 所有结论标【条件性】'}"
    f" | 若未饱和，还需步数估计：{_extrap}")

# ---- 判据 2：上限与两 seed 极差 ----
say(f"\n[S60-LIMIT] ★上限与极差变化")
_all = {s: max(rec_of(RESULTS[s]["curve"], st)["em_full"] for st in GRID) * 100 for s in SEEDS}
_arg = {s: max(GRID, key=lambda st: (rec_of(RESULTS[s]["curve"], st)["em_full"], st)) for s in SEEDS}
say(f"[S60-LIMIT] 每 seed 最好格点：s1234={_all[1234]:.2f}% @{_arg[1234]} | "
    f"s5678={_all[5678]:.2f}% @{_arg[5678]} | 2 seed 均值最好={sum(_all.values())/2:.2f}%")
_lim = f"[S60-LIMIT] @{DELTA_B} 均值={sum(RESULTS[s]['em_final'] for s in SEEDS)/2*100:.2f}%"
if S58:
    _lim += (f" | 对照 S58 @12000 均值="
             f"{sum(rec_of(S58[s]['curve'], 12000)['em_full'] for s in SEEDS)/2*100:.2f}%")
say(_lim)
_sp = {st: abs(rec_of(RESULTS[SEEDS[0]]["curve"], st)["em_full"]
               - rec_of(RESULTS[SEEDS[1]]["curve"], st)["em_full"]) * 100 for st in GRID}
_sp_max = max(_sp, key=lambda st: _sp[st])
_sp580 = (abs(rec_of(S58[SEEDS[0]]['curve'], 12000)['em_full']
              - rec_of(S58[SEEDS[1]]['curve'], 12000)['em_full']) * 100) if S58 else float('nan')
_sp530 = (abs(rec_of(S53[SEEDS[0]]['curve'], 12000)['em_full']
              - rec_of(S53[SEEDS[1]]['curve'], 12000)['em_full']) * 100) if S53 else float('nan')
say(f"[S60-SPREAD] 极差随步数（关键格点）："
    + " / ".join(f"@{st}={_sp[st]:.2f}pp" for st in KEY_STEPS)
    + f" | 全程最大极差={_sp[_sp_max]:.2f}pp @{_sp_max}")
say(f"[S60-SPREAD] 对照：S57/S58 FIX @12000 极差={_sp580:.2f}pp | S53 旧数据 @12000 极差={_sp530:.2f}pp "
    f"| 本臂 @12000={_sp[12000] if 12000 in _sp else float('nan'):.2f}pp "
    f"→ @{DELTA_B}={_sp[DELTA_B]:.2f}pp")

# ---- 判据 4：2×2「新数据」列 + 交互可判性 ----
say(f"\n[S60-2x2] ★2×2「新数据」列更新（格式 s1234/s5678 全量；bad=0 ⇒ clean≡全量）")
_CELLS: dict = {}
for tag, src, satk in (("常量lr×新数据(59)@12000", S59, "sat"),
                       ("cosine×新数据(58)@12000", S58, "sat"),
                       ("cosine×新数据(本臂)@24000", None, None)):
    if src is None:
        vals = [RESULTS[s]["em_final"] * 100 for s in SEEDS]
        sat = None
    elif src and "em_final" in src[SEEDS[0]]:
        vals = [src[s]["em_final"] * 100 for s in SEEDS]
        sat = [src[s][satk] for s in SEEDS] if satk and satk in src[SEEDS[0]] else None
    else:
        say(f"[S60-2x2] {tag:<26} | ★数据缺失（results.jsonl 不可读）")
        continue
    line = f"[S60-2x2] {tag:<26} | {vals[0]:.2f}/{vals[1]:.2f} | 均值={sum(vals)/2:.2f}%"
    if sat is not None:
        line += (f" | 该格 9000→12000 Δ均值={sum(sat)/2:+.2f}pp ⇒ "
                 f"{'未饱和（R35：不可用于交互）' if abs(sum(sat)/2) >= 1.0 else '已饱和'}")
    say(line)
    _CELLS[tag] = dict(vals=vals, sat=sat)
_cn = _CELLS.get("常量lr×新数据(59)@12000", {}).get("sat")
say(f"[S60-2x2-INTER] ★交互可判性（R35）：新数据列的常量格 Δ均值="
    f"{(sum(_cn)/len(_cn) if _cn else float('nan')):+.2f}pp"
    f"（{'★未饱和' if not _cn or abs(sum(_cn)/len(_cn)) >= 1.0 else '已饱和'}）⇒ 该列的『调度效应』"
    f"仍被未饱和的常量格污染 ⇒ **交互仍不可判**（不拿未饱和格做交互）；要判交互须把"
    f"常量 lr × 新数据 也跑到饱和（同 @{DELTA_B}）。")

# ---- 判据 5：新数据绝对 EM 上限 ----
say(f"\n[S60-NEWBASE] ★新数据绝对 EM 上限（供 METRICS.md 新数据栏）：@" + str(DELTA_B) + " "
    + " | ".join(f"s{s} 全量={RESULTS[s]['em_final']*100:.2f}% SEEN={RESULTS[s]['em_seen']*100:.2f}% "
                 f"UNSEEN={RESULTS[s]['em_unseen']*100:.2f}%" for s in SEEDS)
    + f" | 均值全量={sum(RESULTS[s]['em_final'] for s in SEEDS)/2*100:.2f}%"
    + f" | 最好格点均值={sum(_all.values())/2:.2f}%")

# ============================================================================
# ⑥ 成本 / ckpt / 元数据 / DONE
# ============================================================================
say(f"\n[S60] ===== 成本与 ckpt =====")
for s in SEEDS:
    r = RESULTS[s]
    for st in sorted(r["ckpt"]):
        say(f"[META] seed={s}: ckpt={os.path.basename(r['ckpt'][st]['path'])} "
            f"sha256[:16]={r['ckpt'][st]['sha']}")
    say(f"[META] seed={s}: 训练墙钟={r['wall_train']/60:.2f}min 本run={r['wall_total']/60:.2f}min")
say(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
    f"线程={torch.get_num_threads()}(R40) | 新TEST sha16={NEW_SHA}(bad={len(BAD_IDX)}) | "
    f"地板={FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% | 总墙钟={(time.time()-T0)/60:.1f}min")
say("[DONE] exit=0")
sys.exit(0)
