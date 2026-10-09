#!/usr/bin/env python3
"""S58 · ★数据轴主实验：在「修好 {expr} 的数据」上重跑 S53 臂（warmup500+cosine→0 @12000）。

- 唯一变量 = 数据（已修 stages/14_synth_arith.py:126 后重生成的 add_3d train4000/test800）；
  结构 / LR 调度 / batch / 步数 / 评估口径与 S53 逐字相同（模型与训练函数直接 exec S53 源码）。
- A0 对照 = 复用既有 logs/53_ckpt_lrcosine_s12000_seed{1234,5678}.pt（★不重训），
  两臂在【同一批新 test 样本】上逐样本配对。
- 判据 C：Δ = EM(新数据训练) − EM(53 ckpt @新 test)，±SE，全量与 clean 都报
  （新 data bad=0 ⇒ clean==全量，clean 口径已无意义）。
- 饱和检查 R35：新臂 9000→12000 增量。
- 口径：EM=batch1(R28) + 显式 eval()(R39) + 4 线程(R40)。

只允许写：本文件、logs/s58_fixdata.log（外部由 systemd 重定向到 logs/s57_fixdata.log）、
          logs/58_*、/tmp。
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
# ① 旧世界的「评估机器」：exec S53 源码到 ④a 之前
#    （内部 exec S19 前半段 ⇒ 同 tokenizer / 同 build_batch / 同 greedy_gen(EM=1) /
#     同 parse_ans / 同 paired_delta；同时复刻修前 add_3d test 作为对照）
# ============================================================================
_src53 = open(f"{ROOT}/stages/53_lr_schedule.py", encoding="utf-8").read()
_cut53 = _src53.index("# ---- ④a")
NS: dict = {"__name__": "s53_head", "__file__": f"{ROOT}/stages/53_lr_schedule.py"}
exec(compile(_src53[:_cut53], "s53_head", "exec"), NS)          # noqa: S102

import torch  # noqa: E402

device = NS["device"]
SEEDS = NS["SEEDS53"]
SMOKE = NS["SMOKE"]
STEPS = NS["STEPS53"]
GRID = tuple(NS["GRID"])
SAVE_ATS = tuple(NS["SAVE_ATS"])
CKPT_TMPL = (f"/tmp/s58smoke_ckpt_s{{steps}}_seed{{seed}}.pt" if SMOKE
             else f"{ROOT}/logs/58_ckpt_fixeddata_s{{steps}}_seed{{seed}}.pt")
JSONL = "/tmp/58_results_smoke.jsonl" if SMOKE else f"{ROOT}/logs/58_results.jsonl"
OLD_TEST = NS["TEST"]
OLD_TEST_SHA = NS["TEST_SHA"]

print(f"\n[S58] ★数据轴主实验：修后数据重跑 S53 臂（warmup500+cosine→0 @{STEPS}）| "
      f"device={device} {torch.cuda.get_device_name(0)} | d={NS['D']} ff={NS['FF']} "
      f"NHEAD={NS['NHEAD']} batch={NS['BATCH']} MAXLEN={NS['MAXLEN']} | 线程="
      f"{torch.get_num_threads()}(R40) | EM口径=batch{NS['EM_BATCH']}(R28) 自检="
      f"batch{NS['CHK_BATCH']} | 判据=配对Δ vs 53 ckpt | seeds={SEEDS}", flush=True)
print(f"[S58-OLD] 修前 add_3d test 复刻 sha256[:16]={OLD_TEST_SHA} vs 入库 b34e7ea515203227 ⇒ "
      f"{'复现 ✓' if OLD_TEST_SHA == 'b34e7ea515203227' else '★不复现'}", flush=True)

# ============================================================================
# ② 新世界数据：exec 已修 stages/14_synth_arith.py 到 ② 模型 之前（五桶重生成）
# ============================================================================
_src14 = open(f"{ROOT}/stages/14_synth_arith.py", encoding="utf-8").read()
_cut14 = _src14.index("# ② 模型")
NS14: dict = {"__name__": "s14_head", "__file__": f"{ROOT}/stages/14_synth_arith.py"}
print(f"\n[S58-NEW] ===== 已修 S14 重生成五桶（新 [DATA] 见下）=====", flush=True)
exec(compile(_src14[:_cut14], "s14_head", "exec"), NS14)        # noqa: S102
NEW = NS14["DATA"]
NEW_INFO = NS14["BUCKET_INFO"]
BUCKETS = NS["BUCKETS"]

# 新五桶 bad 计数（判据 A）
_bad_tr = {b: sum(1 for r in NEW[b]["train"] if "{expr}" in r["nl"]) for b in BUCKETS}
_bad_te = {b: sum(1 for r in NEW[b]["test"] if "{expr}" in r["nl"]) for b in BUCKETS}
print(f"[S58-BAD] 修后五桶 bad_train={[ _bad_tr[b] for b in BUCKETS]} "
      f"bad_test={[_bad_te[b] for b in BUCKETS]}（旧=75/119/103/110/103 与 14/19/30/20/14）⇒ "
      f"{'✓ 全 0' if all(v == 0 for v in list(_bad_tr.values()) + list(_bad_te.values())) else '★非 0'}",
      flush=True)

# ============================================================================
# ③ 覆盖 S53 的「数据口径全局量」为：新 add_3d train/test
# ============================================================================
NEW_TRAIN = NEW["add_3d"]["train"]
NEW_TEST = NEW["add_3d"]["test"]


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
_c = __import__("collections").Counter(r["gold"] for r in NEW_TRAIN)
_top1, _ = _c.most_common(1)[0]
_top5 = [v for v, _ in _c.most_common(5)]
FLOOR_1 = sum(1 for r in NEW_TEST if r["gold"] == _top1) / len(NEW_TEST)
FLOOR_5 = sum(1 for r in NEW_TEST if r["gold"] in _top5) / len(NEW_TEST)

NEW_EVAL = NEW_TEST[: NS["N_EVAL"]]
CLEAN_EVAL = CLEAN_IDX[: len(NEW_EVAL)]
NS.update(TEST=NEW_TEST, TEST_EVAL=NEW_EVAL, TRAIN=NEW_TRAIN, BAD_IDX=BAD_IDX,
          CLEAN_IDX=CLEAN_IDX, TEST_SHA=NEW_SHA, FLOOR_1=FLOOR_1, FLOOR_5=FLOOR_5,
          CKPT_TMPL=CKPT_TMPL, SAVE_ATS=SAVE_ATS, GRID=GRID, JSONL=JSONL)

print(f"[S58-DATA] 新 add_3d: train={len(NEW_TRAIN)} test={len(NEW_TEST)}(eval {len(NEW_EVAL)}) "
      f"答案域={min(r['y'] for r in NEW_TRAIN)}..{max(r['y'] for r in NEW_TRAIN)}"
      f"({NEW_INFO['add_3d']['ans_n']}个) | 模板覆盖={NEW_INFO['add_3d']['tpl_used']}/358 "
      f"distinct(a,b)train={NEW_INFO['add_3d']['pair_tr']} test={NEW_INFO['add_3d']['pair_te']}",
      flush=True)
print(f"[S58-CHK1] ★新 TEST sha256[:16]={NEW_SHA}（旧={OLD_TEST_SHA}）| (a,b) 泄漏对="
      f"{N_SEEN}/{len(NEW_TEST)} | SEEN={N_SEEN} UNSEEN={len(NEW_TEST)-N_SEEN}", flush=True)
print(f"[S58-CHK2] ★新 clean 口径：nl 含 '{{expr}}' 的不可作答样本={len(BAD_IDX)} ⇒ clean n="
      f"{len(CLEAN_IDX)} = 全量 {len(NEW_TEST)} ⇒ ★clean 口径已无意义（全量==clean）", flush=True)
print(f"[S58-CHK3] 新数据地板：单答案 '{_top1}'={FLOOR_1*100:.2f}% top5={_top5}="
      f"{FLOOR_5*100:.2f}%（旧 S53 口径 0.12%/0.25%）| 干预面=无 ⇒ R29 不适用", flush=True)
assert len(NEW_TRAIN) == 4000 and len(NEW_TEST) == 800 and len(BAD_IDX) == 0, "新数据口径异常"

# ============================================================================
# ④ A0：53 ckpt（不重训）在新 test 上
# ============================================================================
em_at, frac, sha16 = NS["em_at"], NS["frac"], NS["sha16"]
per_sample_ce, dig_acc = NS["per_sample_ce"], NS["dig_acc"]
paired_delta, verdict = NS["paired_delta"], NS["verdict"]
eval_states, r28_selfcheck = NS["eval_states"], NS["r28_selfcheck"]
Cards53 = NS["Cards53"]

ev = Cards53(NS["D"], NS["FF"]).to(device)
print(f"\n[S58] ===== A0 对照：既有 53_ckpt_lrcosine_s12000（不重训）在新 test 上 =====",
      flush=True)
A0 = {}
for s in SEEDS:
    p = f"{ROOT}/logs/53_ckpt_lrcosine_s12000_seed{s}.pt"
    ev.load_state_dict({k: v.to(device) for k, v in torch.load(p, map_location="cpu").items()})
    ev.eval()                                                   # ★R39
    hits, _ = em_at(ev, NEW_EVAL)
    all_ce, dig_ce, _ = per_sample_ce(ev, NEW_EVAL)
    acc, _se, _dce, _dse, _n = dig_acc(dig_ce)
    A0[s] = dict(hits=hits, em=frac(hits), clean=frac([hits[i] for i in CLEAN_EVAL]),
                 em_seen=frac([hits[i] for i, r in enumerate(NEW_EVAL) if r["seen"]]),
                 em_unseen=frac([hits[i] for i, r in enumerate(NEW_EVAL) if not r["seen"]]),
                 ce=frac(all_ce), dig_acc=acc, sha=sha16(p))
    print(f"[S58-A0] seed={s} 53@12000 sha16={A0[s]['sha']} | 新test EM全量={A0[s]['em']*100:.2f}% "
          f"clean={A0[s]['clean']*100:.2f}% SEEN={A0[s]['em_seen']*100:.2f}% "
          f"UNSEEN={A0[s]['em_unseen']*100:.2f}% | 编号每步={acc*100:.2f}% 整体CE={A0[s]['ce']:.4f}",
          flush=True)

# ============================================================================
# ⑤ 主实验：新数据训练 2 seed（S53 逐字同款 train53，仅数据与 ckpt 路径不同）
# ============================================================================
train53 = NS["train53"]
RESULTS = {}
open(JSONL, "w").close()
for s in SEEDS:
    t0 = time.time()
    print(f"\n[S58] ===== seed={s} 新数据训练：warmup{NS['WARMUP']}+cos→0 @{STEPS} "
          f"train={len(NEW_TRAIN)} test={len(NEW_EVAL)} 曲线格={len(GRID)} =====", flush=True)
    model, wall_train, snaps, ckpts, inst, blk = train53(s, NEW_TRAIN, STEPS, GRID, SAVE_ATS)
    print(f"[S58] seed={s} 训练结束 墙钟={wall_train/60:.2f}min | ckpt={sorted(ckpts)}", flush=True)
    curve = eval_states(ev, snaps, GRID, f"FIX s{s}")
    fin = GRID[-1]
    hits_new = curve[fin]["hits"]
    d_full = paired_delta(hits_new, A0[s]["hits"])
    d_clean = paired_delta(hits_new, A0[s]["hits"], CLEAN_EVAL)
    ev.load_state_dict({k: v.to(device) for k, v in snaps[fin].items()})
    ev.eval()                                                   # ★R39
    _r28 = r28_selfcheck(ev, NEW_EVAL, tag=f"[fix s{fin} seed{s}] ",
                         k=min(NS["CHK_BATCH"], len(NEW_EVAL)))
    ckpt_meta = {st: dict(path=ckpts[st], sha=sha16(ckpts[st])) for st in ckpts}
    _s9 = 9000 if 9000 in GRID else GRID[max(0, len(GRID) - 2)]
    _s12 = 12000 if 12000 in GRID else GRID[-1]
    sat = (curve[_s12]["em_full"] - curve[_s9]["em_full"]) * 100
    SAT_STEPS = (_s9, _s12)
    RESULTS[s] = dict(seed=s, steps=STEPS, wall_train=wall_train, wall_total=time.time() - t0,
                      n_eval=len(NEW_EVAL), em_final=curve[fin]["em_full"],
                      em_final_clean=curve[fin]["em_clean"], hits=hits_new,
                      em_seen=frac([hits_new[i] for i, r in enumerate(NEW_EVAL) if r["seen"]]),
                      em_unseen=frac([hits_new[i] for i, r in enumerate(NEW_EVAL) if not r["seen"]]),
                      dig_acc=curve[fin]["dig_acc"], ce=curve[fin]["ce"],
                      delta=d_full, delta_clean=d_clean, sat=sat,
                      r28_same=_r28["same"], r28_k=_r28["k"], ckpt=ckpt_meta,
                      train_loss=inst, train_loss_blk=blk,
                      curve={str(st): {k: v for k, v in curve[st].items() if k != "hits"}
                             for st in GRID})
    print(f"[S58-RUN] seed={s}: EM全量@{fin}={curve[fin]['em_full']*100:.2f}% "
          f"clean={curve[fin]['em_clean']*100:.2f}% | Δ全量={d_full[0]*100:+.2f}pp±"
          f"{d_full[1]*100:.2f}pp ⇒ {verdict(d_full[0], d_full[1])} | Δclean={d_clean[0]*100:+.2f}pp±"
          f"{d_clean[1]*100:.2f}pp ⇒ {verdict(d_clean[0], d_clean[1])} | 数字每步={curve[fin]['dig_acc']*100:.2f}% "
          f"| R28={_r28['same']}/{_r28['k']} | 饱和9000→12000={sat:+.2f}pp | "
          f"合计墙钟={RESULTS[s]['wall_total']/60:.2f}min", flush=True)
    del snaps, model
    torch.cuda.empty_cache()
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in RESULTS[s].items() if k != "hits"},
                           ensure_ascii=False) + "\n")

# ============================================================================
# ⑥ 主判据汇总
# ============================================================================
print(f"\n[S58] ===== ★判据 C：Δ = EM(新数据训练) − EM(53 ckpt @新 test)，逐样本配对 =====",
      flush=True)
for name, key in (("全量", "delta"), ("clean(=全量)", "delta_clean")):
    ms = [RESULTS[s][key][0] for s in SEEDS]
    ses = [RESULTS[s][key][1] for s in SEEDS]
    same_sign = all(m > 0 for m in ms) or all(m < 0 for m in ms)
    print(f"[S58-VERDICT] {name}口径：2 seed Δ={[f'{m*100:+.2f}pp' for m in ms]} "
          f"±SE={[f'{x*100:.2f}pp' for x in ses]} | 同号={'是' if same_sign else '否'} | "
          f"逐 seed=" + " ; ".join(f"s{s}: {verdict(RESULTS[s][key][0], RESULTS[s][key][1])}" for s in SEEDS) +
          f" | 均值Δ={sum(ms)/2*100:+.2f}pp | {'★提升(≥+2SE 且同号)' if same_sign and all(m >= 2*x for m, x in zip(ms, ses)) and ms[0] > 0 else ('★有害(≤−2SE 且同号)' if same_sign and all(m <= -2*x for m, x in zip(ms, ses)) and ms[0] < 0 else ('无影响(|Δ|≤1SE)' if all(abs(m) <= x for m, x in zip(ms, ses)) else '落在 1–2SE 之间/异号 ⇒ 不判定'))}",
          flush=True)
print(f"[S58-NEWBASE] ★新数据绝对 EM（新基线，供 METRICS 更新）：" + " | ".join(
    f"s{s} 全量={RESULTS[s]['em_final']*100:.2f}% clean={RESULTS[s]['em_final_clean']*100:.2f}%"
    for s in SEEDS) + f" | 均值全量={sum(RESULTS[s]['em_final'] for s in SEEDS)/2*100:.2f}%", flush=True)
print(f"[S58-SPREAD] 新臂两 seed 极差（全量）="
      f"{(max(RESULTS[s]['em_final'] for s in SEEDS)-min(RESULTS[s]['em_final'] for s in SEEDS))*100:.2f}pp"
      f" | A0 两 seed 极差="
      f"{(max(A0[s]['em'] for s in SEEDS)-min(A0[s]['em'] for s in SEEDS))*100:.2f}pp", flush=True)

print(f"\n[S58] ===== ★饱和检查（R35）：新臂 9000→12000 =====", flush=True)
for s in SEEDS:
    c9, c12 = RESULTS[s]["curve"][str(SAT_STEPS[0])], RESULTS[s]["curve"][str(SAT_STEPS[1])]
    print(f"[S58-SAT] seed={s}: {SAT_STEPS[0]} 全量={c9['em_full']*100:.2f}% "
          f"clean={c9['em_clean']*100:.2f}% → {SAT_STEPS[1]} 全量={c12['em_full']*100:.2f}% "
          f"clean={c12['em_clean']*100:.2f}% | Δ={RESULTS[s]['sat']:+.2f}pp ⇒ "
          f"{'★未饱和(≥1pp) ⇒ 结论条件性' if RESULTS[s]['sat'] >= 1.0 else '已平(<1pp)'}", flush=True)
_sat_mean = sum(RESULTS[s]["sat"] for s in SEEDS) / 2
print(f"[S58-SAT-VERDICT] 2 seed 均值 9000→12000 Δ={_sat_mean:+.2f}pp ⇒ "
      f"{'★未饱和：本单元结论按 [条件性] 记' if _sat_mean >= 1.0 else '已饱和'}", flush=True)

# ============================================================================
# ⑦ 成本 / ckpt / 元数据
# ============================================================================
print(f"\n[S58] ===== 成本与 ckpt =====", flush=True)
for s in SEEDS:
    r = RESULTS[s]
    for st in sorted(r["ckpt"]):
        print(f"[META] seed={s}: ckpt={os.path.basename(r['ckpt'][st]['path'])} "
              f"sha256[:16]={r['ckpt'][st]['sha']}", flush=True)
    print(f"[META] seed={s}: 训练墙钟={r['wall_train']/60:.2f}min 本run={r['wall_total']/60:.2f}min",
          flush=True)
print(f"[S58-TABLE] 臂 | s1234 全量/clean | s5678 全量/clean", flush=True)
print(f"[S58-TABLE] A0(53@新test) | {A0[1234]['em']*100:.2f}/{A0[1234]['clean']*100:.2f} | "
      f"{A0[5678]['em']*100:.2f}/{A0[5678]['clean']*100:.2f}", flush=True)
print(f"[S58-TABLE] FIX(新数据训练) | {RESULTS[1234]['em_final']*100:.2f}/"
      f"{RESULTS[1234]['em_final_clean']*100:.2f} | {RESULTS[5678]['em_final']*100:.2f}/"
      f"{RESULTS[5678]['em_final_clean']*100:.2f}", flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"线程={torch.get_num_threads()}(R40) | 新TEST sha16={NEW_SHA} 泄漏={N_SEEN}/{len(NEW_TEST)} "
      f"clean n={len(CLEAN_IDX)}(bad={len(BAD_IDX)}) | 地板={FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% | "
      f"总墙钟={(time.time()-T0)/60:.1f}min", flush=True)
print("[DONE] exit=0", flush=True)
sys.exit(0)
