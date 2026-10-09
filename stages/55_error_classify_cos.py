#!/usr/bin/env python3
"""S55 · E10（零训练 · CPU · batch=1）：在 cosine 基线上重做 E9 的误差分类与难度分布。

问题（F25 最高优先级）：S53 改写了基线（常量 lr clean 95.45% → cosine warmup500+cos→0 97.27%）
  ⇒ E9 的结论（平台是【模型顶】：错例 A 解析错 0.0% / B 数值错 100.0%；非难度墙）必须在 cosine 基线上复核。

口径（★与 E9 完全同口径）：本文件不另写一套 ——
  exec stages/52_error_classify.py 到 "# ⑤ 主循环" 之前（逐字复用 S52 的数据复刻、
  clean 索引、Cards、load_ckpt、greedy_gen、parse_ans/int_of、classify_err、carry_feats），
  只把 §1 里两个【输出路径】改写到 /tmp（S52 的 OUT_LOG 是 "w" 打开，绝不能覆盖 logs/52_*），
  然后把 CKPTS 换成 logs/53_ckpt_lrcosine_s12000_seed{1234,5678}.pt。
  R28 batch=1；R39 显式 model.eval()；R40 torch.set_num_threads(4)；CUDA_VISIBLE_DEVICES=""。

判据（写死）：
  1 误差分类：B 数值错仍 ≈100% ⇒ cosine 未改变瓶颈性质（仍数值精度）⇒ 容量/优化方向继续有效；
             A 解析错显著上升（如 >10%）⇒ 瓶颈转向口径/格式 ⇒ 下一步修评估口径。
  2 UNSEEN≈全量（Δ ≤ 1SE）⇒ 与 E9 一致（非记忆缺口）；clean EM 对照 METRICS 97.27%（Δ>0.13pp 点名）。
  3 B 类难度分布：长进位链偏置减弱/消失 ⇒ cosine 部分解决"难度"；不变 ⇒ 瓶颈性质未变 ⇒ 容量轴更值得做。

只允许写：本文件、logs/55_*、/tmp；既有 stages/*.py、logs/ 其它文件一律只读。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import Counter, defaultdict

FLOWME = "/home/vesita/coding/my/flowme"
SRC52 = f"{FLOWME}/stages/52_error_classify.py"
LOG_DIR = f"{FLOWME}/logs"
OUT_LOG = f"{LOG_DIR}/55_error_classify_cos.log"
OUT_JSON = f"{LOG_DIR}/55_error_classify_cos.json"
HEAD_TMP = "/tmp/55_s52head.log"
HEAD_JSON_TMP = "/tmp/55_s52head.json"

T_START = time.time()

# ============================================================================
# ① 逐字复用 S52（=E9）前半段；★仅把输出路径改写到 /tmp，避免覆盖 logs/52_*
# ============================================================================
_src = open(SRC52, encoding="utf-8").read()
_cut = _src.index("# ⑤ 主循环")
_head = _src[:_cut]
for _old, _new in ((f'OUT_LOG = LOG_DIR + "/52_error_classify.log"', f'OUT_LOG = "{HEAD_TMP}"'),
                   (f'OUT_JSON = LOG_DIR + "/52_error_classify.json"', f'OUT_JSON = "{HEAD_JSON_TMP}"')):
    assert _old in _head, f"未能改写 S52 输出路径（口径漂移风险）：{_old}"
    _head = _head.replace(_old, _new)
assert ('OUT_LOG = LOG_DIR + "/52_error_classify.log"' not in _head
        and 'OUT_JSON = LOG_DIR + "/52_error_classify.json"' not in _head), "★S52 输出路径未改写"
NS = {"__name__": "s52_head", "__file__": SRC52}
exec(compile(_head, SRC52, "exec"), NS)                     # noqa: S102
for _k, _v in NS.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
del NS, _src, _head
# ★exec 会把 S52 的 OUT_LOG/OUT_JSON 带进 globals（已指向 /tmp）⇒ 此处重新钉死本单元输出路径
OUT_LOG = f"{LOG_DIR}/55_error_classify_cos.log"
OUT_JSON = f"{LOG_DIR}/55_error_classify_cos.json"
assert OUT_LOG.startswith(f"{LOG_DIR}/55_") and OUT_JSON.startswith(f"{LOG_DIR}/55_")
assert DEVICE == "cpu" and torch.get_num_threads() == 4, "口径不符：需 CPU + 4 线程(R40)"
assert torch.cuda.is_available() is False, "★CUDA 可见（硬约束：全程 CPU）"

_log_f = open(OUT_LOG, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log_f.write(s + "\n")
    _log_f.flush()


say(f"[S55] device={DEVICE} torch={torch.__version__} cuda.is_available()={torch.cuda.is_available()} "
    f"threads={torch.get_num_threads()}(R40) | 零训练 · EM batch={EM_BATCH}(R28) · eval() 显式(R39) | "
    f"bucket={BUCKET} | 口径 = 逐字复用 stages/52_error_classify.py(E9) 到 §5 之前（仅改写到 /tmp）")
say(f"[S55] E9 基线锚点（常量 lr @12000）= clean 95.45/90.52% | 新基线（cosine @12000）= METRICS 97.27%")

# ============================================================================
# ② ckpt：(cosine) S53 @12000 × 2 seed；sha 与 logs/s53_lrcos.log:131,134 对照
# ============================================================================
CKPTS55 = [(12000, sd, f"{LOG_DIR}/53_ckpt_lrcosine_s12000_seed{sd}.pt") for sd in (1234, 5678)]
EXP_SHA55 = {"53_ckpt_lrcosine_s12000_seed1234.pt": "1857ea042a00ceb0",
             "53_ckpt_lrcosine_s12000_seed5678.pt": "ea8417641087d820"}
EXP_CLEAN_EM55 = 97.27                                     # METRICS.md:8 / s53_lrcos.log:64,96
EXP_E9 = {1234: dict(n_err=35, A=0, B=35, clean=95.45, full=91.88),
          5678: dict(n_err=73, A=0, B=73, clean=90.52, full=87.12)}   # E9 @12000

# 地板（与 S53 同法：train gold 的单答案 / top5 频率）
_c_tr = Counter(r["gold"] for r in TRAIN)
_top1, _ = _c_tr.most_common(1)[0]
_top5 = [v for v, _ in _c_tr.most_common(5)]
FLOOR_1 = sum(1 for r in TEST if r["gold"] == _top1) / len(TEST)
FLOOR_5 = sum(1 for r in TEST if r["gold"] in _top5) / len(TEST)
say(f"[S55-SPLIT] 全量 n={len(TEST)}: SEEN={len(SEEN_FULL)} UNSEEN={len(UNSEEN_FULL)} | "
    f"clean n={len(CLEAN_IDX)}: SEEN={len(SEEN_CLEAN)} UNSEEN={len(UNSEEN_CLEAN)} | "
    f"地板：单答案'{_top1}'={FLOOR_1*100:.2f}% top5={FLOOR_5*100:.2f}%")

# ============================================================================
# ③ 主循环（与 S52 §5 同逻辑）：clean(n=770) 贪心 batch=1 生成 + 错例分类 + 分层
# ============================================================================
say(f"\n[S55] ===== §2 逐 ckpt：clean(n={len(CLEAN_IDX)}) 贪心 batch=1 生成 + 错例分类 =====")
RES: list[dict] = []
VEC: list[dict] = []
RECS_CLEAN = [TEST[i] for i in CLEAN_IDX]
for st, sd, path in CKPTS55:
    label = f"lrcosine_s{st}_seed{sd}"
    if not os.path.exists(path):
        say(f"[SKIP] {label}: ★ckpt 缺失 {path}")
        continue
    model, sha = load_ckpt(path)
    t0 = time.time()
    txt, eos = greedy_gen(model, RECS_CLEAN, batch=EM_BATCH)     # R28 + R39（内部 model.eval()）
    wall = time.time() - t0
    del model

    hits: dict[int, int] = {}
    err_of: dict[int, str] = {}
    ex: dict[str, list[str]] = defaultdict(list)
    for j, i in enumerate(CLEAN_IDX):
        gold = TEST[i]["gold"]
        p = parse_ans(txt[j])
        ok = int(p == gold)
        hits[i] = ok
        if not ok:
            cls = classify_err(txt[j], gold, eos[j])             # ★E9 逐字同一函数
            err_of[i] = cls
            if len(ex[cls]) < 4:
                ex[cls].append(f"a={TEST[i]['a']} b={TEST[i]['b']} gold='{gold}' "
                               f"gen='{txt[j][:60]}' parsed={p!r}")

    def em(sel, hits=hits):
        n = len(sel)
        k = sum(hits.get(i, 0) for i in sel)
        p = k / n if n else float("nan")
        return dict(n=n, k=k, em=p, se=binom_se(p, n))

    all_clean, all_full = em(CLEAN_IDX), em(range(len(TEST)))
    seen_c, unseen_c = em(SEEN_CLEAN), em(UNSEEN_CLEAN)
    seen_f, unseen_f = em(SEEN_FULL), em(UNSEEN_FULL)

    cnt = Counter(err_of.values())
    n_err = len(err_of)
    sha_ok = EXP_SHA55.get(os.path.basename(path)) == sha
    d_em = (all_clean["em"] - EXP_CLEAN_EM55 / 100.0) * 100
    say(f"\n[EM] {label} sha={sha}{'（与 s53 log 一致 ✓）' if sha_ok else '（★sha 与 s53 log 不符）'} | "
        f"clean EM={all_clean['em']*100:.2f}%±{all_clean['se']*100:.2f} (n={all_clean['n']}, "
        f"与 METRICS {EXP_CLEAN_EM55:.2f}% Δ={d_em:+.2f}pp"
        f"{'  ★>0.13pp 点名' if abs(d_em) > 0.13 else ' ✓'}) | 全量={all_full['em']*100:.2f}% "
        f"| 错例={n_err} | 生成墙钟={wall:.1f}s")
    say(f"     全量口径: SEEN n={seen_f['n']} EM={seen_f['em']*100:.2f}%±{seen_f['se']*100:.2f} | "
        f"UNSEEN n={unseen_f['n']} EM={unseen_f['em']*100:.2f}%±{unseen_f['se']*100:.2f} | "
        f"Δ(UNSEEN−全量)={((unseen_f['em']-all_full['em'])*100):+.2f}pp (1SE={unseen_f['se']*100:.2f}pp)")
    say(f"     clean 口径: SEEN n={seen_c['n']} EM={seen_c['em']*100:.2f}%±{seen_c['se']*100:.2f} | "
        f"UNSEEN n={unseen_c['n']} EM={unseen_c['em']*100:.2f}%±{unseen_c['se']*100:.2f} | "
        f"Δ(UNSEEN−clean)={((unseen_c['em']-all_clean['em'])*100):+.2f}pp (1SE={unseen_c['se']*100:.2f}pp)")
    say(f"     误差分类 (n_err={n_err}): " + " | ".join(
        f"{c}={cnt.get(c,0)}({100.0*cnt.get(c,0)/n_err:.1f}%)" for c in "ABCDE" if cnt.get(c, 0)))
    e9 = EXP_E9[sd]
    say(f"     ★E9 vs E10 (seed{sd}): E9 常量lr n_err={e9['n_err']} A={100*e9['A']/e9['n_err']:.1f}% "
        f"B={100*e9['B']/e9['n_err']:.1f}%  ‖  E10 cosine n_err={n_err} "
        f"A={100.0*cnt.get('A',0)/n_err:.1f}% B={100.0*cnt.get('B',0)/n_err:.1f}%")
    for c in "ABCDE":
        for e in ex[c]:
            say(f"       [{c}] {e}")

    b_idx = [i for i, c in err_of.items() if c == "B"]
    RES.append(dict(label=label, steps=st, seed=sd, sha=sha, sha_ok=sha_ok, wall_gen=wall,
                    clean=all_clean, full=all_full, seen_full=seen_f, unseen_full=unseen_f,
                    seen_clean=seen_c, unseen_clean=unseen_c, n_err=n_err,
                    err_counts={c: cnt.get(c, 0) for c in "ABCDE"},
                    err_share={c: (cnt.get(c, 0) / n_err if n_err else 0.0) for c in "ABCDE"},
                    b_idx=b_idx, examples={c: ex[c] for c in "ABCDE"}))
    VEC.append(dict(label=label, hits=hits, err_of=err_of, b_idx=b_idx))

# ============================================================================
# ④ §3 B 类（数值错）难度分布（与 S52 §6 同逻辑：池化 + ★样本去重）
# ============================================================================
say("\n[S55] ===== §3 B 类（数值错）的难度分布（与 E9 同口径）=====")
pool_b = [i for v in VEC for i in v["b_idx"]]
base = [FEAT[i] for i in CLEAN_IDX]
n_base = len(base)


def share(xs, pred):
    return sum(1 for x in xs if pred(x)) / len(xs) if xs else float("nan")


pool_all_err = [i for v in VEC for i in v["err_of"]]
DIST = {}
for key, pred, name in (("n_carry>=2", lambda f: f["n_carry"] >= 2, "进位次数≥2"),
                        ("n_carry==3", lambda f: f["n_carry"] == 3, "进位次数=3"),
                        ("chain>=2", lambda f: f["chain"] >= 2, "最长进位链≥2"),
                        ("digits==4", lambda f: f["n_digits"] == 4, "结果为 4 位数")):
    p0, pB, pE = share(base, pred), share([FEAT[i] for i in pool_b], pred), \
        share([FEAT[i] for i in pool_all_err], pred)
    nb = len(pool_b)
    z = (pB - p0) / math.sqrt(p0 * (1 - p0) / nb) if nb and 0 < p0 < 1 else float("nan")
    DIST[key] = dict(name=name, base=p0, b=pB, all_err=pE, n_b=nb, z=z)
    say(f"[S55-DIFF] {name}: 基准(clean)={p0*100:.1f}% | 全部错例={pE*100:.1f}% | "
        f"★B类={pB*100:.1f}% | n_B={nb} z={z:+.1f}")

uniq_b = sorted({i for v in VEC for i in v["b_idx"]})
DEID = {}
E9_DEID = {"n_carry>=2": 57.4, "chain>=2": 48.5, "digits==4": 51.5}    # E9 去重口径（logs/52_*）
for key, pred, name in (("n_carry>=2", lambda f: f["n_carry"] >= 2, "进位次数≥2"),
                        ("chain>=2", lambda f: f["chain"] >= 2, "最长进位链≥2"),
                        ("digits==4", lambda f: f["n_digits"] == 4, "结果为 4 位数")):
    p0 = share(base, pred)
    pu = share([FEAT[i] for i in uniq_b], pred)
    nu = len(uniq_b)
    z = (pu - p0) / math.sqrt(p0 * (1 - p0) / nu) if nu and 0 < p0 < 1 else float("nan")
    DEID[key] = dict(name=name, base=p0, b_uniq=pu, n_uniq=nu, z=z, e9=E9_DEID[key])
    say(f"[S55-DIFF-DEID] {name}: 基准={p0*100:.1f}% | ★去重 B 类={pu*100:.1f}% (n_uniq={nu}, z={z:+.1f}) "
        f"‖ E9 去重 B 类={E9_DEID[key]:.1f}% ⇒ Δ={((pu-E9_DEID[key]/100)*100):+.1f}pp")

m_base = sum(f["n_carry"] for f in base) / n_base
m_b = sum(FEAT[i]["n_carry"] for i in pool_b) / len(pool_b) if pool_b else float("nan")
say(f"[S55-DIFF] 平均进位次数: 基准={m_base:.3f} | ★B类={m_b:.3f}（E9 B类=1.627, 基准=1.500）")

say("\n[S55-DIFF] 按进位次数分桶的 clean EM（看是否单调下降；E9：无单调下降）:")
EM_BY_CARRY = {}
for f in range(4):
    sel = [i for i in CLEAN_IDX if FEAT[i]["n_carry"] == f]
    row = {}
    for v in VEC:
        n = len(sel)
        k = sum(v["hits"].get(i, 0) for i in sel)
        row[v["label"]] = (k / n if n else float("nan"), n)
    EM_BY_CARRY[f] = row
    say(f"   进位={f} (n={len(sel)}): " + " | ".join(f"{lb}={row[lb][0]*100:.1f}%" for lb in row))
say("[S55-DIFF] 按结果位数分桶的 clean EM:")
EM_BY_DIG = {}
for dgt in (3, 4):
    sel = [i for i in CLEAN_IDX if FEAT[i]["n_digits"] == dgt]
    row = {}
    for v in VEC:
        n = len(sel)
        k = sum(v["hits"].get(i, 0) for i in sel)
        row[v["label"]] = (k / n if n else float("nan"), n)
    EM_BY_DIG[dgt] = row
    say(f"   {dgt} 位数 (n={len(sel)}): " + " | ".join(f"{lb}={row[lb][0]*100:.1f}%" for lb in row))

# ============================================================================
# ⑤ ★判据
# ============================================================================
say("\n[S55] ===== §4 ★ 判据 =====")
totA = sum(r["err_counts"]["A"] for r in RES)
totB = sum(r["err_counts"]["B"] for r in RES)
tot = sum(r["n_err"] for r in RES)
say(f"[S55-VERDICT] 2 ckpt 合计错例={tot}: " + " ".join(
    f"{c}={sum(r['err_counts'][c] for r in RES)}({100.0*sum(r['err_counts'][c] for r in RES)/tot:.1f}%)"
    for c in "ABCDE"))
say(f"[S55-VERDICT] ★1 B(数值错)={100*totB/tot:.1f}% A(解析错)={100*totA/tot:.1f}% ⇒ "
    f"{'B≈100% ⇒ cosine 未改变瓶颈性质（仍是数值精度）⇒ 容量/优化方向继续有效' if totB/tot > 0.9 else ('A>10% ⇒ 瓶颈转向口径/格式 ⇒ 下一步修评估口径' if totA/tot > 0.1 else 'B 非压倒性 ⇒ 按上表具体判')}")
for r in RES:
    d = (r["unseen_full"]["em"] - r["full"]["em"]) * 100
    say(f"[S55-VERDICT] ★2 {r['label']}: Δ(UNSEEN−全量)={d:+.2f}pp vs 1SE={r['unseen_full']['se']*100:.2f}pp "
        f"⇒ {'UNSEEN≈全量 ✓（与 E9 一致，非记忆缺口）' if abs(d) <= r['unseen_full']['se']*100 else '★超 1SE，需注意'} "
        f"| clean EM={r['clean']['em']*100:.2f}% vs METRICS 97.27%")
e9c = share([FEAT[i] for i in uniq_b], lambda f: f["chain"] >= 2)
pm = m_b
say(f"[S55-VERDICT] ★3 B 类长进位链≥2: E9={E9_DEID['chain>=2']:.1f}% → E10={e9c*100:.1f}% "
    f"(Δ={((e9c-E9_DEID['chain>=2']/100)*100):+.1f}pp)；平均进位 E9=1.627 → E10={pm:.3f} ⇒ "
    f"{'长进位链偏置减弱/消失 ⇒ cosine 部分解决难度' if e9c < 0.45 else '偏置不变 ⇒ 瓶颈性质未变（普遍精度不足）⇒ 容量轴更值得做'}")

out = dict(unit="S55", device=DEVICE, threads=NTHREADS, batch=EM_BATCH, bucket=BUCKET,
           n_clean=len(CLEAN_IDX), n_test=len(TEST),
           split=dict(seen_full=len(SEEN_FULL), unseen_full=len(UNSEEN_FULL),
                      seen_clean=len(SEEN_CLEAN), unseen_clean=len(UNSEEN_CLEAN)),
           floor=dict(single=FLOOR_1, top5=FLOOR_5, top1=_top1),
           rows=[{k: v for k, v in r.items() if k != "b_idx"} for r in RES],
           error_totals=dict(A=totA, B=totB, n=tot), e9_ref=EXP_E9,
           diff=DIST, diff_deid=DEID, uniq_b=uniq_b, mean_carry=dict(base=m_base, b=m_b),
           em_by_carry={str(k): {lb: v[0] for lb, v in row.items()} for k, row in EM_BY_CARRY.items()},
           em_by_digits={str(k): {lb: v[0] for lb, v in row.items()} for k, row in EM_BY_DIG.items()},
           b_idx_per_ckpt={v["label"]: v["b_idx"] for v in VEC}, wall_s=time.time() - T_START)
json.dump(out, open(OUT_JSON, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
say(f"\n[S55] ★墙钟={time.time()-T_START:.1f}s | 写出 {OUT_LOG} 与 {OUT_JSON}")
_log_f.close()
