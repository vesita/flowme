#!/usr/bin/env python3
"""S61 · ★正则轴第一枪：weight decay（单变量），主判据走 **loss/CE 口径**（R41）。

为什么用 loss 口径：
  S60 已证 EM 在新数据上饱和（@21000→@24000 均值 −0.06pp，平台均值 97.87%）；
  E11 已证过拟合在 EM 口径看不出、在 loss 口径成立（loss gap +0.0664 vs +0.0287 nats = 2.3×）；
  ⇒ 按 R41：主指标饱和时判决必须换 loss 口径 ⇒ **本单元主判据 = 新 test 整体 CE（nats）**，EM 只作辅助报告。

★基线的实际 weight decay（本单元开工先查，任务书的兜底条款触发）：
  train53（被 58_ckpt 使用的那条训练路径，逐字未改）写的是
      opt = AdamW(model.parameters(), lr=BASE_LR)        ← 没有 weight_decay 形参
  torch 2.9.1 的 AdamW 默认 weight_decay = 0.01 ⇒ **基线 A0（=58_ckpt）的 wd 就是 1e-2，不是 0**。
  ⇒ 任务书设想的「臂 wd=1e-2 vs 对照 wd=0（复用 58_ckpt）」在事实上退化为同一配置。
  ⇒ 处理：**两臂都真跑**（单变量、同 seed、同数据、同调度，只有 wd 不同）：
       arm wd1e2 : weight_decay = 1e-2（显式传；应与 58_ckpt 逐位相同 ⇒ 反向证死基线 wd）
       arm wd0   : weight_decay = 0    （任务书里的「A0」配置，本单元新训）
     主判据仍按任务书定义：ΔCE = CE(wd=1e-2) − CE(wd=0)。

设计（单变量：只动 wd）：
  新数据（已修 S14 重生成的 add_3d train4000/test800，bad=0，TEST sha16=fa03ef9287567510）
  + warmup500 + cosine→0 @12000 + d=128/ff=512/nhead=4/batch=32/seed=1234,5678，逐字复用 S53 的 train53。

★判据（写死）：
  1 主判据（loss，R41）：ΔCE = 测试整体 CE(wd1e-2) − 测试整体 CE(wd0)，同批新 test 逐样本配对 ±SE；
      ΔCE ≤ −2SE 且 2 seed 同号 ⇒ 正则降低泛化误差；|ΔCE| ≤ 1SE ⇒ 无影响；
      ΔCE ≥ +2SE 且 2 seed 同号 ⇒ 有害。
  2 辅判据（EM，仅报告）：EM 与 ΔEM ±SE；★EM 已饱和 ⇒ 不作为判决依据（R41）。
  3 泛化 gap（loss 版）：gap = 测试 CE − 训练 CE（eval 模式、同一代码路径、同批）；
      gap(wd) < gap(A0) 且 2 seed 同向 ⇒ 正则确实压小泛化 gap。
  4 饱和检查（R35）：9000→12000 的 CE 与 EM 增量；EM ≥1pp ⇒ 标条件性。
  5 曲线：@3000/6000/9000/12000 的 train CE 与 test CE（2 seed），与 A0 对照。

★口径：EM 一律 batch=1（R28）+ batch=1 vs 16 自检；R39 每次前向显式 eval()；
  R40 batch=1 逐样本前向 4 线程；新 test n=800 bad=0（clean≡全量）；
  锚点复核：58_ckpt 在新 test 上应 97.38/96.88（均值 97.12）且 CE 可复现。

只允许写：本文件、logs/s61_wd.log（systemd 重定向）、logs/61_*、/tmp。
"""
from __future__ import annotations

import hashlib
import inspect
import json
import math
import os
import sys
import time

ROOT = "/home/vesita/coding/my/flowme"
T0 = time.time()
SMOKE = os.environ.get("S61_SMOKE") == "1"
if SMOKE:
    os.environ["S53_SMOKE"] = "1"
    os.environ["S14_SMOKE"] = "1"

# ============================================================================
# ① 旧世界：exec S53 源码到 ④a 之前（内部 exec S19 前半段 ⇒ 同 tokenizer / build_batch /
#    greedy_gen(EM_BATCH=1) / parse_ans / r28_selfcheck / paired_delta / Cards53 / train53）
# ============================================================================
SRC53 = f"{ROOT}/stages/53_lr_schedule.py"
_src53 = open(SRC53, encoding="utf-8").read()
_cut53 = _src53.index("# ---- ④a")
NS: dict = {"__name__": "s53_head", "__file__": SRC53}
exec(compile(_src53[:_cut53], "s53_head", "exec"), NS)              # noqa: S102

import torch  # noqa: E402
from torch.optim import AdamW as RealAdamW  # noqa: E402

device = NS["device"]
assert device == "cuda", f"★device 必须 cuda，实得 {device}"
SEEDS = tuple(NS["SEEDS53"])
S53_SMOKE = bool(NS["SMOKE"])
STEPS = NS["STEPS53"]
GRID = tuple(NS["GRID"])
SAVE_ATS = tuple(NS["SAVE_ATS"])
N_EVAL = NS["N_EVAL"]
D, FF, NHEAD, MAXLEN, BATCH = NS["D"], NS["FF"], NS["NHEAD"], NS["MAXLEN"], NS["BATCH"]
EM_BATCH, CHK_BATCH = NS["EM_BATCH"], NS["CHK_BATCH"]
Cards53, train53 = NS["Cards53"], NS["train53"]
em_at, per_sample_ce, frac = NS["em_at"], NS["per_sample_ce"], NS["frac"]
paired_delta, mean_se, r28_selfcheck = NS["paired_delta"], NS["mean_se"], NS["r28_selfcheck"]
sha16 = NS["sha16"]
OLD_TEST_SHA = NS["TEST_SHA"]

LOG = "/tmp/61_smoke.log" if SMOKE else f"{ROOT}/logs/61_weight_decay.log"
JSONL = "/tmp/61_results_smoke.jsonl" if SMOKE else f"{ROOT}/logs/61_results.jsonl"
CK58 = f"{ROOT}/logs/58_ckpt_fixeddata_s12000_seed{{seed}}.pt"
ARMS = [("wd1e2", 1e-2, ("/tmp/61smoke_ckpt_wd1e2_s{steps}_seed{seed}.pt" if SMOKE
                         else f"{ROOT}/logs/61_ckpt_wd1e2_s{{steps}}_seed{{seed}}.pt")),
        ("wd0", 0.0, ("/tmp/61smoke_ckpt_wd0_s{steps}_seed{seed}.pt" if SMOKE
                      else f"{ROOT}/logs/61_ckpt_wd0_s{{steps}}_seed{{seed}}.pt"))]

_log_f = open(LOG, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log_f.write(s + "\n")
    _log_f.flush()


# ============================================================================
# ② ★基线实际 wd：静态证据（源码行）+ 运行时证据（torch 默认值）
# ============================================================================
_wd_default = inspect.signature(RealAdamW.__init__).parameters["weight_decay"].default
_opt_lines = [l.strip() for l in _src53.splitlines() if "AdamW(" in l]
say(f"[S61] device={device} {torch.cuda.get_device_name(0)} torch={torch.__version__} | "
    f"线程={torch.get_num_threads()}(R40) | EM口径=batch{EM_BATCH}(R28) 自检=batch{CHK_BATCH} | "
    f"d={D} ff={FF} NHEAD={NHEAD} batch={BATCH} MAXLEN={MAXLEN} | smoke={SMOKE} | seeds={SEEDS}")
say(f"[S61-WD] ★基线实际 wd：stages/53_lr_schedule.py 的优化器构造 = {_opt_lines} "
    f"⇒ 未显式传 weight_decay ⇒ 实际 wd = torch AdamW 默认值 {_wd_default}"
    f" ⇒ {'★基线 wd = 1e-2（≠0，触发任务书兜底条款）' if abs(_wd_default - 1e-2) < 1e-12 else '基线 wd=0'}")
say(f"[S61-WD] ⇒ 任务书设想的『对照 wd=0 = 复用 58_ckpt』不成立（58_ckpt 本身就是 wd=1e-2）；"
    f"本单元改为两臂都真跑：wd1e2(1e-2) 与 wd0(0)，单变量只有 wd 不同；"
    f"wd1e2 臂同时用来与 58_ckpt 逐位对账（证死基线 wd）")

# ============================================================================
# ③ 新数据：exec 已修 stages/14_synth_arith.py 到 ② 模型 之前（五桶重生成）
# ============================================================================
_src14 = open(f"{ROOT}/stages/14_synth_arith.py", encoding="utf-8").read()
_cut14 = _src14.index("# ② 模型")
NS14: dict = {"__name__": "s14_head", "__file__": f"{ROOT}/stages/14_synth_arith.py"}
exec(compile(_src14[:_cut14], "s14_head", "exec"), NS14)            # noqa: S102
NEW = NS14["DATA"]
BUCKETS = list(NS["BUCKETS"])
_bad_tr = {b: sum(1 for r in NEW[b]["train"] if "{expr}" in r["nl"]) for b in BUCKETS}
_bad_te = {b: sum(1 for r in NEW[b]["test"] if "{expr}" in r["nl"]) for b in BUCKETS}
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
NEW_EVAL = NEW_TEST[:N_EVAL]
NEW_SHA_EXPECT = "fa03ef9287567510"

NS.update(TEST=NEW_TEST, TEST_EVAL=NEW_EVAL, TRAIN=NEW_TRAIN, BAD_IDX=BAD_IDX,
          CLEAN_IDX=CLEAN_IDX, TEST_SHA=NEW_SHA, TEST_SHA_EXPECT=NEW_SHA_EXPECT,
          CKPT_TMPL=ARMS[0][2], SAVE_ATS=SAVE_ATS, GRID=GRID, JSONL=JSONL)

say(f"[S61-BAD] 修后五桶 bad_train={[str(_bad_tr[b]) for b in BUCKETS]} "
    f"bad_test={[str(_bad_te[b]) for b in BUCKETS]} ⇒ "
    f"{'✓ 全 0' if all(v == 0 for v in list(_bad_tr.values()) + list(_bad_te.values())) else '★非 0'}")
say(f"[S61-CHK1] ★新 TEST sha256[:16]={NEW_SHA} vs 任务书 {NEW_SHA_EXPECT} ⇒ "
    f"{'逐位一致 ✓' if NEW_SHA == NEW_SHA_EXPECT else '★不一致'} | 旧(修前) sha16={OLD_TEST_SHA} | "
    f"(a,b) 泄漏对={N_SEEN}/{len(NEW_TEST)}")
say(f"[S61-CHK2] 口径自检：新 test n={len(NEW_TEST)}（eval {len(NEW_EVAL)}）bad={len(BAD_IDX)} "
    f"⇒ clean n={len(CLEAN_IDX)} ≡ 全量 | train n={len(NEW_TRAIN)} | "
    f"device={device}(cuda ✓) 线程={torch.get_num_threads()}(R40 ✓) EM=batch{EM_BATCH}(R28)")
if not SMOKE:
    assert NEW_SHA == NEW_SHA_EXPECT, "★新数据口径不符 ⇒ 本单元无效"
    assert (len(NEW_TRAIN), len(NEW_TEST), len(BAD_IDX)) == (4000, 800, 0), "新数据形状异常"
    assert torch.get_num_threads() == 4, "★R40：必须 4 线程"

# ============================================================================
# ④ 评估工具（R39 显式 eval；token 加权整体 CE = 主判据；per-sample CE 供配对）
# ============================================================================
N_PARAMS = sum(p.numel() for p in Cards53(D, FF).parameters())


@torch.no_grad()
def ce_eval(model, recs) -> dict:
    """eval 模式：token 加权整体 CE（主判据）+ 逐样本 CE（配对用）+ 逐样本 token 数。"""
    model.eval()                                                    # ★R39
    ps, dig, _ = per_sample_ce(model, recs)                          # 内部再置 eval（R39）
    nts = [len(r["t"]) for r in recs]
    tot = sum(nts)
    ce_tok = sum(x * n for x, n in zip(ps, nts)) / tot
    return dict(ce_tok=ce_tok, ce_samp=frac(ps), per_sample=ps, ntoks=tot)


def l2_norm(sd) -> float:
    tot = 0.0
    for v in sd.values():
        if v.is_floating_point():
            tot += float((v.float() ** 2).sum())
    return math.sqrt(tot)


def max_abs_diff(a, b) -> float:
    if set(a) != set(b):
        return float("inf")
    m = 0.0
    for k in a:
        m = max(m, float((a[k].float() - b[k].float()).abs().max()))
    return m


def judge_ce(delta, se):
    if delta <= -2 * se and delta < 0:
        return "★正则降低泛化误差（ΔCE ≤ −2SE）"
    if abs(delta) <= se:
        return "无影响（|ΔCE| ≤ 1SE）"
    if delta >= 2 * se and delta > 0:
        return "★有害（ΔCE ≥ +2SE）"
    return f"落在 1–2SE 之间（不判定，Δ/SE={delta/se if se else float('nan'):+.2f}）"


ev = Cards53(D, FF).to(device)
CURVE_STEPS = [st for st in (3000, 6000, 9000, 12000) if st in set(GRID)]
if not CURVE_STEPS:
    CURVE_STEPS = [GRID[-1]]
SAT9 = 9000 if 9000 in set(GRID) else GRID[max(0, len(GRID) - 2)]
SAT12 = 12000 if 12000 in set(GRID) else GRID[-1]
EVAL_STEPS = sorted(set(CURVE_STEPS) | {SAT9, SAT12, GRID[-1]})
say(f"[S61-PARAMS] Cards53(d={D},ff={FF}) 参数量={N_PARAMS/1e6:.4f}M（两臂同构、只差 wd ⇒ 参数量完全相同）"
    f" | 曲线格点={CURVE_STEPS} | 评估格点={EVAL_STEPS} | 饱和格点 {SAT9}→{SAT12}")

# ============================================================================
# ⑤ ★锚点复核：58_ckpt 在新 test 上（应 97.38/96.88，均值 97.12）
# ============================================================================
say(f"\n[S61] ===== ★锚点复核：logs/58_ckpt_fixeddata_s12000_seed<s>.pt 在新 test 上 =====")
A58 = {}
for s in SEEDS:
    p = CK58.format(seed=s)
    assert os.path.exists(p), f"缺 ckpt {p}"
    sd = torch.load(p, map_location="cpu")
    ev.load_state_dict({k: v.to(device) for k, v in sd.items()})
    ev.eval()                                                       # ★R39
    hits, _ = em_at(ev, NEW_EVAL)                                   # ★R28 batch=1
    cem = ce_eval(ev, NEW_EVAL)
    A58[s] = dict(hits=hits, em=frac(hits), ce_tok=cem["ce_tok"], ce_samp=cem["ce_samp"],
                  per_sample=cem["per_sample"], sd=sd, sha=sha16(p), l2=l2_norm(sd))
    say(f"[S61-ANCHOR] 58_ckpt s{s} sha16={A58[s]['sha']} | 新test EM={A58[s]['em']*100:.2f}% "
        f"（入库 97.38/96.88，均值 97.12）| 整体CE(tok)={cem['ce_tok']:.4f} "
        f"CE(samp)={cem['ce_samp']:.4f}（S58 入库 0.0076/0.0090）| ||θ||={A58[s]['l2']:.3f}")
    del sd
_anchor_mean = sum(A58[s]["em"] for s in SEEDS) / len(SEEDS)
say(f"[S61-ANCHOR] 2 seed 均值 EM={_anchor_mean*100:.2f}% vs 入库 97.12% ⇒ "
    f"Δ={(_anchor_mean-0.9712)*100:+.2f}pp ⇒ "
    f"{'复现 ✓' if abs(_anchor_mean-0.9712) <= 0.005 else '★不复现，点名'}")

# ============================================================================
# ⑥ 主实验：两臂 × 2 seed 训练（train53 逐字复用，只把 AdamW 的 wd 与 ckpt 路径换掉）
# ============================================================================
REC: dict = {}
ORDER = [(a[0], a[1], a[2], s) for a in ARMS for s in SEEDS]


def make_adamw(wd: float, tag: str):
    def _factory(params, *a, **kw):
        kw["weight_decay"] = wd
        opt = RealAdamW(params, *a, **kw)
        _LAST[tag] = dict(wd=float(opt.param_groups[0]["weight_decay"]),
                          groups=len(opt.param_groups))
        return opt
    return _factory


_LAST: dict = {}
say(f"\n[S61] ===== 主实验：{len(ORDER)} run（两臂×2 seed，{STEPS} 步）=====")
open(JSONL, "w").close()
for aname, wd, tmpl, s in ORDER:
    NS["AdamW"] = make_adamw(wd, aname)
    NS["CKPT_TMPL"] = tmpl
    assert train53.__globals__ is NS, "train53 的全局命名空间不是 NS ⇒ wd/ckpt 覆盖会失效"
    t0 = time.time()
    say(f"\n[S61] ===== arm={aname}(wd={wd:g}) seed={s} @{STEPS} 步 train={len(NEW_TRAIN)} "
        f"test={len(NEW_EVAL)} =====")
    model, wall_train, snaps, ckpts, inst, blk = train53(s, NEW_TRAIN, STEPS, GRID, SAVE_ATS)
    say(f"[S61] arm={aname} seed={s} 训练结束 墙钟={wall_train/60:.2f}min | "
        f"实际生效 wd={_LAST[aname]['wd']:g}（{_LAST[aname]['groups']} 个 param_group）| "
        f"ckpt={sorted(ckpts)}")
    assert abs(_LAST[aname]["wd"] - wd) < 1e-15, f"★wd 未生效：{_LAST[aname]} vs {wd}"
    CURVE: dict = {}
    fin = GRID[-1]
    for st in EVAL_STEPS:
        ev.load_state_dict({k: v.to(device) for k, v in snaps[st].items()})
        ev.eval()                                                   # ★R39
        hits, _ = em_at(ev, NEW_EVAL)                               # ★R28 batch=1
        te = ce_eval(ev, NEW_EVAL)
        tr = ce_eval(ev, NEW_TRAIN)
        CURVE[str(st)] = dict(em=frac(hits), hits=hits,
                              ce_test=te["ce_tok"], ce_test_samp=te["ce_samp"],
                              ce_train=tr["ce_tok"], ce_train_samp=tr["ce_samp"],
                              per_sample=te["per_sample"],
                              gap=te["ce_tok"] - tr["ce_tok"],
                              l2=l2_norm(snaps[st]))
        say(f"[{aname} s{s}] step={st:>5} EM={frac(hits)*100:.2f}% | "
            f"train CE={tr['ce_tok']:.4f} test CE={te['ce_tok']:.4f} "
            f"gap={te['ce_tok']-tr['ce_tok']:+.4f} nats | ||θ||={CURVE[str(st)]['l2']:.3f}")
    fin_key = str(fin)
    ev.load_state_dict({k: v.to(device) for k, v in snaps[fin].items()})
    ev.eval()                                                       # ★R39
    _r28 = r28_selfcheck(ev, NEW_EVAL, tag=f"[{aname} s{fin} seed{s}] ",
                         k=min(CHK_BATCH, len(NEW_EVAL)))
    ckpt_meta = {str(st): dict(path=ckpts[st], sha=sha16(ckpts[st])) for st in ckpts}
    d58 = max_abs_diff(A58[s]["sd"], snaps[fin]) if not SMOKE else float("nan")
    REC[(aname, s)] = dict(arm=aname, wd=wd, seed=s, steps=STEPS, wall_train=wall_train,
                           wall_total=time.time() - t0, n_eval=len(NEW_EVAL),
                           r28_same=_r28["same"], r28_k=_r28["k"], ckpt=ckpt_meta,
                           train_loss={str(k): v for k, v in inst.items()},
                           train_loss_blk={str(k): v for k, v in blk.items()},
                           curve={k: {kk: vv for kk, vv in v.items()
                                      if kk not in ("hits", "per_sample")}
                                  for k, v in CURVE.items()},
                           em=frac(CURVE[fin_key]["hits"]),
                           ce_test=CURVE[fin_key]["ce_test"],
                           ce_train=CURVE[fin_key]["ce_train"],
                           gap=CURVE[fin_key]["gap"],
                           l2=CURVE[fin_key]["l2"], diff_vs_58=d58,
                           per_sample=CURVE[fin_key]["per_sample"],
                           hits=CURVE[fin_key]["hits"])
    say(f"[S61-RUN] arm={aname} seed={s}: EM@{fin}={REC[(aname, s)]['em']*100:.2f}% | "
        f"★test CE(tok)={REC[(aname, s)]['ce_test']:.4f} train CE={REC[(aname, s)]['ce_train']:.4f} "
        f"gap={REC[(aname, s)]['gap']:+.4f} | ||θ||={REC[(aname, s)]['l2']:.3f} | R28={_r28['same']}/"
        f"{_r28['k']} | 与 58_ckpt 最大逐元素差={d58:.3e}（sha16={ckpt_meta[str(fin)]['sha']} vs "
        f"58={A58[s]['sha']}）| 本 run={REC[(aname, s)]['wall_total']/60:.2f}min")
    del snaps, model
    torch.cuda.empty_cache()
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps({k: v for k, v in REC[(aname, s)].items()
                            if k not in ("hits", "per_sample")}, ensure_ascii=False) + "\n")

# ============================================================================
# ⑦ ★主判据（loss 口径，R41）：ΔCE = CE(wd1e2) − CE(wd0)，同批新 test 逐样本配对
# ============================================================================
say(f"\n[S61] ===== ★主判据（loss/R41）：ΔCE = 测试整体 CE(wd=1e-2) − CE(wd=0)，"
    f"逐样本配对 ±SE =====")
D_CE = {}
for s in SEEDS:
    a, b = REC[("wd1e2", s)], REC[("wd0", s)]
    d_tok = a["ce_test"] - b["ce_test"]
    diffs = [x - y for x, y in zip(a["per_sample"], b["per_sample"])]
    m, se = mean_se(diffs)
    D_CE[s] = (m, se, d_tok)
    say(f"[S61-DELTA-CE] s{s}: 整体CE(tok) wd1e2={a['ce_test']:.4f} vs wd0={b['ce_test']:.4f} "
        f"⇒ ΔCE(tok)={d_tok:+.4f} | 逐样本配对 ΔCE={m:+.4f} ± {se:.4f} nats "
        f"({(m/se) if se else float('nan'):+.2f}SE) ⇒ {judge_ce(m, se)}")
ms = [D_CE[s][0] for s in SEEDS]
ses = [D_CE[s][1] for s in SEEDS]
same = all(x > 0 for x in ms) or all(x < 0 for x in ms)
_sig_neg = same and all(m <= -2 * se for m, se in zip(ms, ses))
_sig_pos = same and all(m >= 2 * se for m, se in zip(ms, ses))
_noeff = all(abs(m) <= se for m, se in zip(ms, ses))
VERDICT_CE = ("★正则降低泛化误差（ΔCE ≤ −2SE 且 2 seed 同号）" if _sig_neg else
              "★有害（ΔCE ≥ +2SE 且 2 seed 同号）" if _sig_pos else
              "无影响（|ΔCE| ≤ 1SE，2 seed）" if _noeff else
              "不判定（落 1–2SE 之间或 2 seed 异号）")
say(f"[S61-VERDICT-CE] 2 seed 配对 ΔCE 均值={sum(ms)/len(ms):+.4f} nats "
    f"（{'/'.join(f'{x:+.4f}' for x in ms)}）± {sum(ses)/len(ses):.4f} | 同号={'是' if same else '否'} "
    f"| 逐 seed 判定=" + " ; ".join(f"s{s}: {judge_ce(D_CE[s][0], D_CE[s][1])}" for s in SEEDS)
    + f" ⇒ 【{VERDICT_CE}】")

# ============================================================================
# ⑧ 辅判据（EM，★仅报告，不作判决 —— R41）
# ============================================================================
say(f"\n[S61] ===== 辅判据（EM，★仅报告）：ΔEM = EM(wd1e2) − EM(wd0) =====")
D_EM = {}
for s in SEEDS:
    a, b = REC[("wd1e2", s)], REC[("wd0", s)]
    m, se, n = paired_delta(a["hits"], b["hits"])
    D_EM[s] = (m, se)
    say(f"[S61-DELTA-EM] s{s}: EM wd1e2={a['em']*100:.2f}% vs wd0={b['em']*100:.2f}% "
        f"⇒ ΔEM={m*100:+.2f}pp ± {se*100:.2f}pp （n={n}）")
_em_abs = [REC[(an, s)]["em"] for an in ("wd1e2", "wd0") for s in SEEDS]
say(f"[S61-EM-NOTE] ★EM 已饱和：四臂 EM={['%.2f' % (x*100) for x in _em_abs]}%"
    f"（S60 平台均值 97.87%、上限 97.75–98.06%）⇒ 按 R41 不作判决依据；"
    f"2 seed 配对 ΔEM 均值={sum(x[0] for x in D_EM.values())/len(SEEDS)*100:+.2f}pp"
    f"（同号={'是' if all(x[0] > 0 for x in D_EM.values()) or all(x[0] < 0 for x in D_EM.values()) else '否'}）")

# ============================================================================
# ⑨ ★泛化 gap（loss 版）对照：gap = test CE − train CE（eval 模式、同协议）
# ============================================================================
say(f"\n[S61] ===== ★泛化 gap（loss 版）对照 =====")
D_GAP, D_TR, D_TE = {}, {}, {}
for s in SEEDS:
    a, b = REC[("wd1e2", s)], REC[("wd0", s)]
    dg = a["gap"] - b["gap"]
    D_GAP[s], D_TR[s], D_TE[s] = dg, a["ce_train"] - b["ce_train"], a["ce_test"] - b["ce_test"]
    say(f"[S61-GAP] s{s}: gap(wd1e2)={a['gap']:+.4f} nats（test {a['ce_test']:.4f} − train "
        f"{a['ce_train']:.4f}）‖ gap(wd0)={b['gap']:+.4f} nats（test {b['ce_test']:.4f} − train "
        f"{b['ce_train']:.4f}）⇒ Δgap={dg:+.4f} nats | Δtrain CE={D_TR[s]:+.4f} Δtest CE={D_TE[s]:+.4f}")
dg_s = [D_GAP[s] for s in SEEDS]
same_g = all(x > 0 for x in dg_s) or all(x < 0 for x in dg_s)
say(f"[S61-GAP-VERDICT] 2 seed Δgap 均值={sum(dg_s)/len(dg_s):+.4f} nats（同向="
    f"{'是' if same_g else '否'}）⇒ "
    f"{'★gap(wd=1e-2) 更小 ⇒ 正则确实压小了泛化 gap（最直接的机理证据）' if same_g and sum(dg_s) < 0 else ('★gap(wd=1e-2) 更大 ⇒ 与『正则压小 gap』方向相反' if same_g else '2 seed 不同向 ⇒ 不成立')}"
    f" | gap 比 = {REC[('wd1e2', SEEDS[0])]['gap']/REC[('wd0', SEEDS[0])]['gap']:.2f}× "
    f"/ {REC[('wd1e2', SEEDS[1])]['gap']/REC[('wd0', SEEDS[1])]['gap']:.2f}×")

# ============================================================================
# ⑩ 曲线对照 + 饱和检查（R35）+ 参数量
# ============================================================================
say(f"\n[S61] ===== ★CE 曲线对照（eval 模式；train CE vs test CE）=====")
for st in CURVE_STEPS:
    k = str(st)
    row = []
    for s in SEEDS:
        a, b = REC[("wd1e2", s)]["curve"][k], REC[("wd0", s)]["curve"][k]
        row.append(f"s{s}: wd1e2 train={a['ce_train']:.4f}/test={a['ce_test']:.4f} "
                   f"‖ wd0 train={b['ce_train']:.4f}/test={b['ce_test']:.4f} "
                   f"‖ Δtest={a['ce_test']-b['ce_test']:+.4f} ΔEM={100*(a['em']-b['em']):+.2f}pp")
    say(f"[S61-CURVE] step={st:>5} | " + " | ".join(row))

say(f"\n[S61] ===== ★饱和检查（R35）：{SAT9}→{SAT12} =====")
for an in ("wd1e2", "wd0"):
    for s in SEEDS:
        c = REC[(an, s)]["curve"]
        a9, a12 = c[str(SAT9)], c[str(SAT12)]
        d_em = (a12["em"] - a9["em"]) * 100
        say(f"[S61-SAT] {an} s{s}: {SAT9}→{SAT12} ΔEM={d_em:+.2f}pp | ΔtestCE="
            f"{a12['ce_test']-a9['ce_test']:+.4f} nats | ΔtrainCE={a12['ce_train']-a9['ce_train']:+.4f} "
            f"⇒ {'★EM 未饱和(≥1pp) ⇒ 结论条件性' if d_em >= 1.0 else 'EM 已平(<1pp)'}")

# ============================================================================
# ⑪ 成本 / ckpt / 元数据 / 落盘
# ============================================================================
say(f"\n[S61] ===== 成本与 ckpt =====")
for an, wd, _t, s in ORDER:
    r = REC[(an, s)]
    say(f"[META] arm={an} seed={s}: 训练墙钟={r['wall_train']/60:.2f}min 本 run={r['wall_total']/60:.2f}min")
    for st in sorted(r["ckpt"], key=lambda x: int(x)):
        say(f"[META] arm={an} seed={s}: ckpt={os.path.basename(r['ckpt'][st]['path'])} "
            f"sha256[:16]={r['ckpt'][st]['sha']}")
say(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
    f"线程={torch.get_num_threads()}(R40) | 新TEST sha16={NEW_SHA}(n={len(NEW_TEST)},bad={len(BAD_IDX)}) | "
    f"参数={N_PARAMS/1e6:.4f}M×2 臂（只差 wd）| 基线(58) wd={_wd_default:g} | "
    f"总墙钟={(time.time()-T0)/60:.1f}min")

with open(f"{'/tmp/61_meta_smoke.json' if SMOKE else ROOT + '/logs/61_meta.json'}", "w",
          encoding="utf-8") as f:
    json.dump(dict(unit="S61", device=device, torch=torch.__version__,
                   threads=torch.get_num_threads(), em_batch=EM_BATCH, chk_batch=CHK_BATCH,
                   params=N_PARAMS, baseline_wd=float(_wd_default), arms={a[0]: a[1] for a in ARMS},
                   new_test_sha=NEW_SHA, n_test=len(NEW_TEST), bad=len(BAD_IDX),
                   anchor58={str(s): dict(em=A58[s]["em"], ce_tok=A58[s]["ce_tok"],
                                          ce_samp=A58[s]["ce_samp"], sha=A58[s]["sha"])
                             for s in SEEDS},
                   main_delta_ce={str(s): dict(paired=d_ce[0], se=d_ce[1], tok=d_ce[2])
                                  for s, d_ce in D_CE.items()},
                   verdict_ce=VERDICT_CE,
                   gap={str(s): dict(ce_train=REC[("wd1e2", s)]["ce_train"],
                                     ce_test=REC[("wd1e2", s)]["ce_test"],
                                     ce_train_a0=REC[("wd0", s)]["ce_train"],
                                     ce_test_a0=REC[("wd0", s)]["ce_test"]) for s in SEEDS},
                   runs={f"{an}_{s}": {kk: vv for kk, vv in REC[(an, s)].items()
                                       if kk not in ("hits", "per_sample")}
                         for an, _w, _t, s in ORDER},
                   wall=(time.time() - T0)), f, ensure_ascii=False, indent=1)
say("[DONE] exit=0")
sys.exit(0)
