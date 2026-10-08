#!/usr/bin/env python3
"""S29 · 判决：模因是「任务无关的内容坐标系」还是「带任务倾向」？—— 零训练 · 全程 CPU。

唯一问题：不同任务桶（add_1d/add_2d/add_3d/sub_2d/mul_2d）各自端到端训出的整模型，
在**受控内容**下产出的模因，是否落在同一个语义坐标系里？
  ⇒ 若是 ⇒ 模因任务无关（数据驱动路线成立）；若否 ⇒ 模因带任务倾向（初始卡片定义的偏差被证实）。

构造（★先把「内容」控制住，才测得到「任务」）：
  (III)  基线：同桶跨 seed（1234 vs 5678）——内容逐字相同，只差训练随机性。
  (II-S) 严格「同一组 (a,b) 数字」：核心三桶 {add_2d, sub_2d, mul_2d}，三者共享操作数域
         a∈10..99, b∈1..9 ⇒ 同数字、各桶自己的算子词、同一句模板 ⇒ 差异只剩「任务」。
         add_1d(a,b∈0..9) 与 add_3d(a,b∈100..999) 的操作数**位数域与其余桶不相交**
         （S14 的 LOHI 写死：add_1d 0..9 / add_2d 10..99 / add_3d 100..999），
         无法共享同一组数字 ⇒ 只能进 (II-N)。
  (II-N) 尺度归一内容轴：同一批归一化幅度对 (u,v)∈[0,1)^2，各桶按自身定义域线性映射
         ⇒ 覆盖全 5 桶，给出完整 5×5 矩阵（内容 = 相对量级，非逐字同一题 ⇒ 比 (II-S) 弱一档）。
度量（沿用 S26 三口径）：(i) mean-pool 余弦 · (ii) 线性核 CKA · (iii) 线性映射 W 后余弦；
  每个都配「随机配对」对照 ⇒ 报 Δ±SE。
正控：同桶·同内容异模板（应高）vs 同桶·异内容同模板（应低）；R29 随机配对必须非零差异。

零训练、零 GPU：只前向取模因。复用方式 = exec stages/26_meme_anchoring.py 的「# ⑤ 主流程」之前
那一段（其内部又 exec stages/17 与 stages/14 的前半段 ⇒ 生成器/Cards/模因提取/CKA/线性映射全现成）。
只允许写：本文件、logs/、/tmp。不碰 09…28、不碰 speculation/methodology/src/tests。
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ★先于 torch 导入：强制关 GPU
import numpy as np  # noqa: E402
import torch  # noqa: E402

ROOT = "/home/vesita/coding/my/flowme"
SRC26 = f"{ROOT}/stages/26_meme_anchoring.py"
T0 = time.time()

os.environ.setdefault("S26_THREADS", "8")
os.environ.setdefault("S26_K", "128")
os.environ.setdefault("S26_NPERM", "50")
os.environ.setdefault("S26_BATCH", "64")

# ============================================================================
# ① 逐字复用 S26 的「# ⑤ 主流程」之前那一段（⇒ S17 前半段 ⇒ S14 前半段）
# ============================================================================
_s26 = open(SRC26, encoding="utf-8").read()
_c26 = _s26.index("# ⑤ 主流程")
exec(compile(_s26[:_c26], SRC26, "exec"), globals())          # noqa: S102
assert "extract26" in globals() and "cka_linear" in globals() and "DATA" in globals()

BUCKETS = ("add_1d", "add_2d", "add_3d", "sub_2d", "mul_2d")
LOHI = NS14["LOHI"]
assert LOHI == {"add_1d": (0, 9), "add_2d": (10, 99), "add_3d": (100, 999)}, LOHI
CANON = "请计算 {expr} 等于多少？"          # 全桶同一句模板
CANON2 = "帮我算一下 {expr} 的结果是几？"   # 正控用第二句模板
_tset = {t for t, _cjk in TEMPLATES}
assert CANON in _tset and CANON2 in _tset, "模板不在 S14 TEMPLATES 里"
DEV = "cpu"
assert DEV == "cpu" and torch.cuda.is_available() is False, "必须全程 CPU"

Kk = K
rng29 = np.random.default_rng(int(os.environ.get("S29_SEED", "20261029")))
PERM = [torch.tensor(rng29.permutation(Kk)) for _ in range(NPERM)]   # (ii) 随机行置换对照

# ---- (II-S) 严格同 (a,b)：核心三桶共享域 a∈10..99, b∈1..9（不重复取 K 组）----
_all = [(a, b) for a in range(10, 100) for b in range(1, 10)]
apairs = [_all[i] for i in rng29.permutation(len(_all))[:Kk]]
CORE = ("add_2d", "sub_2d", "mul_2d")

# ---- (II-N) 尺度归一：同一批 (u,v) ∈ [0,1)^2 ----
un, vn = rng29.random(Kk), rng29.random(Kk)


def ops_of(bucket: str, mode: str, i: int) -> tuple[int, int]:
    if mode == "S":
        a, b = apairs[i]
        return a, b
    u, v = float(un[i]), float(vn[i])
    if bucket == "add_1d":
        return int(round(u * 9)), int(round(v * 9))
    if bucket == "add_2d":
        return 10 + int(round(u * 89)), 10 + int(round(v * 89))
    if bucket == "add_3d":
        return 100 + int(round(u * 899)), 100 + int(round(v * 899))
    if bucket == "sub_2d":
        a = 10 + int(round(u * 89))
        return a, max(0, a - int(round(v * 89)))
    if bucket == "mul_2d":
        return 10 + int(round(u * 89)), 1 + int(round(v * 8))
    raise KeyError(bucket)


def y_of(bucket: str, a: int, b: int) -> int:
    if bucket.startswith("add"):
        return a + b
    if bucket == "sub_2d":
        return a - b
    return a * b


def prompts_of(bucket: str, mode: str, tpl: str = CANON, shift: int = 0) -> list[list[int]]:
    opw = CN_OPS[bucket][0]
    out = []
    for i in range(Kk):
        j = (i + shift) % Kk
        a, b = ops_of(bucket, mode, j)
        out.append(enc(rend(tpl, a, b, opw, False)))
    return out


# ============================================================================
# ② 三口径 + 随机配对对照
# ============================================================================
def compare(X: torch.Tensor, Y: torch.Tensor) -> dict:
    st_i = pair_stats(cosmat(X, Y))                       # (i) 原始 mean-pool
    st_ic = pair_stats(cosmat(center(X), center(Y)))      # (i') 去均值 mean-pool
    cka = cka_linear(X, Y)                                # (ii)
    pk = [cka_linear(X, Y[p]) for p in PERM]              # (ii) 随机行置换
    cm, cs = float(np.mean(pk)), float(np.std(pk, ddof=1) / math.sqrt(len(pk)))
    st_i2, st_iii = linmap(X, Y)                          # (iii) train 拟合 W / test 评估
    _, st_iii_r = linmap(X, Y, derange=True)              # 对照：W 用随机配对拟合
    return dict(i=st_i, ic=st_ic, cka=cka, cka_perm=cm, cka_perm_se=cs, cka_d=cka - cm,
                lin_i=st_i2, lin_iii=st_iii, lin_iii_r=st_iii_r,
                gain_i=st_iii["same"] - st_i2["same"], gain_d=st_iii["d"] - st_i2["d"])


def load_bucket(bucket: str, seed: int):
    m = build_model26(bucket, seed)
    return m


def memes(model, prompts):
    rh, ri, _ = extract26(model, prompts, BATCH)
    return pool(rh), pool(ri)


print(f"\n[S29] ===== 跨任务桶模因坐标系判决 | device={DEV} | "
      f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')!r} | "
      f"torch.cuda.is_available()={torch.cuda.is_available()} | threads={torch.get_num_threads()} "
      f"| K={Kk} NPERM={NPERM} batch={BATCH} =====", flush=True)
print(f"[S29] 尺度约束（S14 LOHI 写死）：add_1d 操作数 0..9 · add_2d 10..99 · add_3d 100..999 "
      f"⇒ 三者操作数域互不相交 ⇒ 严格同 (a,b) 只能用核心三桶 {CORE}", flush=True)

# ============================================================================
# ③ 自检：恒等 / R28 batch 依赖 / 模因尺度
# ============================================================================
m0 = load_bucket("add_2d", 1234)
sp = prompts_of("add_2d", "S")[:16]
h1, i1, _ = extract26(m0, sp, 1)
h2, i2, _ = extract26(m0, sp, 1)
id_h = max(float((a - b).abs().max()) for a, b in zip(h1, h2))
h64, i64, _ = extract26(m0, sp, BATCH)
r28 = max(float((a - b).abs().max()) for a, b in zip(h1, h64))
rms_h = float(torch.cat([r.reshape(-1) for r in h1]).pow(2).mean().sqrt())
print(f"[SELF] 同 prompt 两次前向 模因 max|Δ|={id_h:.3e} ⇒ {'精确 0 ✓' if id_h == 0 else '★非 0'} | "
      f"batch=1 vs {BATCH} max|Δ|={r28:.3e}（RMS|模因|={rms_h:.3f}，相对={r28 / rms_h:.1e}）"
      f" ⇒ {'无 batch 依赖' if r28 == 0 else '★只用同一 batch 口径'}", flush=True)

# ============================================================================
# ④ 提模因：10 个 ckpt × {S, N} 两套内容 × (模因 h, 输入卡输出 h_in)
# ============================================================================
P = {}          # (bucket, seed, mode) -> prompts
NEED = [(b, "N") for b in BUCKETS] + [(b, "S") for b in CORE]
for bucket, mode in NEED:
    for seed_ in (1234, 5678):
        P[(bucket, seed_, mode)] = prompts_of(bucket, mode)

MH, MI, SH = {}, {}, {}
for bucket, mode in NEED:
    for seed_ in (1234, 5678):
        m = load_bucket(bucket, seed_)
        Xh, Xi = memes(m, P[(bucket, seed_, mode)])
        MH[(bucket, seed_, mode)] = Xh
        MI[(bucket, seed_, mode)] = Xi
        SH[(bucket, seed_)] = sha16(ckpt_path(bucket, seed_))
    del m

# ============================================================================
# ⑤ 正控：同桶·同内容异模板（应高） vs 同桶·异内容同模板（应低）
# ============================================================================
CTRL = {}
for bucket in CORE:
    m = load_bucket(bucket, 1234)
    Xa, _ = memes(m, prompts_of(bucket, "S", CANON))
    Xb, _ = memes(m, prompts_of(bucket, "S", CANON2))
    Xc, _ = memes(m, prompts_of(bucket, "S", CANON, shift=1))     # 内容整体错位 1 ⇒ 异内容同模板
    same_ct = pair_stats(cosmat(Xa, Xb))                          # 同内容(数字同)、异模板
    diff_ct = pair_stats(cosmat(Xa, Xc))                          # 异内容(数字异)、同模板
    CT = pair_stats(cosmat(center(Xa), center(Xb)))
    DT = pair_stats(cosmat(center(Xa), center(Xc)))
    CTRL[bucket] = dict(same_content_diff_tpl=same_ct["same"], same_d=same_ct["d"],
                        same_d_se=same_ct["d_se"],
                        diff_content_same_tpl=diff_ct["same"], diff_d=diff_ct["d"],
                        rnd=same_ct["rand"], same_ct_ic=CT["same"], diff_ct_ic=DT["same"])
    print(f"[CTRL-正控] {bucket} 同内容异模板 cos={same_ct['same']:.4f} (Δ={same_ct['d']:+.4f}"
          f"±{same_ct['d_se']:.4f}) | 异内容同模板 cos={diff_ct['same']:.4f} (Δ={diff_ct['d']:+.4f}"
          f"±{diff_ct['d_se']:.4f}) | 随机配对 cos={same_ct['rand']:.4f} ⇒ 区分度 "
          f"{same_ct['same'] - diff_ct['same']:+.4f} | (i')去均值 同内容={CT['same']:.4f} "
          f"异内容={DT['same']:.4f}", flush=True)

# ============================================================================
# ⑥ 基线（III）同桶跨 seed
# ============================================================================
BASE = {}
for mode in ("S", "N"):
    for bucket in BUCKETS:
        if mode == "S" and bucket not in CORE:
            continue
        r = compare(MH[(bucket, 1234, mode)], MH[(bucket, 5678, mode)])
        r_in = compare(MI[(bucket, 1234, mode)], MI[(bucket, 5678, mode)])
        BASE[(bucket, mode)] = r
        print(f"[BASE-{mode}] {bucket:7s} 跨seed(s1234 vs s5678) sha {SH[(bucket, 1234)]}/"
              f"{SH[(bucket, 5678)]} | (i) cos={r['i']['same']:.4f} Δ={r['i']['d']:+.4f}±{r['i']['d_se']:.4f}"
              f" | (i') 去均值 cos={r['ic']['same']:.4f} Δ={r['ic']['d']:+.4f}±{r['ic']['d_se']:.4f}"
              f" | (ii) CKA={r['cka']:.4f} vs 置换 {r['cka_perm']:.4f}±{r['cka_perm_se']:.4f} "
              f"(Δ={r['cka_d']:+.4f}) | (iii) 映射后 cos={r['lin_iii']['same']:.4f} "
              f"Δ={r['lin_iii']['d']:+.4f}±{r['lin_iii']['d_se']:.4f} [输入卡 (ii) CKA={r_in['cka']:.4f}]",
              flush=True)

# ============================================================================
# ⑦ 跨桶矩阵：对角 = 同桶跨 seed（基线 III），非对角 = 跨桶同 seed=1234（II）
# ============================================================================
MAT = {}
for mode, bl in (("S", CORE), ("N", BUCKETS)):
    M = {}
    for bi in bl:
        for bj in bl:
            X = MH[(bi, 5678, mode)] if bi == bj else MH[(bi, 1234, mode)]
            Y = MH[(bj, 1234, mode)]
            M[(bi, bj)] = compare(X, Y)
            XI = MI[(bi, 5678, mode)] if bi == bj else MI[(bi, 1234, mode)]
            YI = MI[(bj, 1234, mode)]
            M[(bi, bj)]["cka_in"] = cka_linear(XI, YI)
    MAT[mode] = M
    print(f"[MAT-{mode}-输入卡输出 CKA 5x5（次级口径：输入卡本身，非思维卡模因）]", flush=True)
    print("        " + " ".join(f"{b:>8s}" for b in bl), flush=True)
    for bi in bl:
        print("  " + f"{bi:>6s} " + " ".join(f"{M[(bi, bj)]['cka_in']:8.3f}" for bj in bl), flush=True)
    print(f"[MAT-{mode}-输入卡 CKA 对角={float(np.mean([M[(b, b)]['cka_in'] for b in bl])):.3f} "
          f"非对角={float(np.mean([M[(bi, bj)]['cka_in'] for bi in bl for bj in bl if bi != bj])):.3f}]",
          flush=True)
    diag = [M[(b, b)].copy() for b in bl]
    off = [M[(bi, bj)] for bi in bl for bj in bl if bi != bj]
    dd = [float(np.mean([M[(bi, bj)]["cka"] for bj in bl if bj != bi])) for bi in bl]
    delta = [M[(b, b)]["cka"] - d for b, d in zip(bl, dd)]
    dm = float(np.mean(delta))
    dse = float(np.std(delta, ddof=1) / math.sqrt(len(delta))) if len(delta) > 1 else float("nan")
    doi = [float(np.mean([M[(bi, bj)]["lin_iii"]["same"] for bj in bl if bj != bi])) for bi in bl]
    diii = [M[(b, b)]["lin_iii"]["same"] for b in bl]
    d3 = [a - b for a, b in zip(diii, doi)]
    d3m = float(np.mean(d3))
    d3se = float(np.std(d3, ddof=1) / math.sqrt(len(d3))) if len(d3) > 1 else float("nan")
    MAT[mode + "_sum"] = dict(
        diag_cka=[M[(b, b)]["cka"] for b in bl],
        off_cka=[M[(bi, bj)]["cka"] for bi in bl for bj in bl if bi != bi],
        diag_mean=float(np.mean([M[(b, b)]["cka"] for b in bl])),
        off_mean=float(np.mean([M[(bi, bj)]["cka"] for bi in bl for bj in bl if bi != bj])),
        off_se=float(np.std([M[(bi, bj)]["cka"] for bi in bl for bj in bl if bi != bj],
                            ddof=1) / math.sqrt(len(bl) * (len(bl) - 1))),
        delta_diag_vs_off=dm, delta_se=dse,
        diag_lin3=float(np.mean(diii)), off_lin3=float(np.mean(doi)),
        delta_lin3=d3m, delta_lin3_se=d3se,
        diag_i=[M[(b, b)]["i"]["same"] for b in bl],
        off_i=[M[(bi, bj)]["i"]["same"] for bi in bl for bj in bl if bi != bj],
        diag_i_mean=float(np.mean([M[(b, b)]["i"]["same"] for b in bl])),
        off_i_mean=float(np.mean([M[(bi, bj)]["i"]["same"] for bi in bl for bj in bl if bi != bj])),
        diag_ic_mean=float(np.mean([M[(b, b)]["ic"]["same"] for b in bl])),
        off_ic_mean=float(np.mean([M[(bi, bj)]["ic"]["same"] for bi in bl for bj in bl if bi != bj])),
    )
    s = MAT[mode + "_sum"]
    print(f"\n[S29-MAT-{mode}] 桶序 {list(bl)} | CKA 对角(同桶跨seed)={s['diag_mean']:.4f} "
          f"非对角(跨桶)={s['off_mean']:.4f}±{s['off_se']:.4f} ⇒ Δ(对角-非对角)={s['delta_diag_vs_off']:+.4f}"
          f"±{s['delta_se']:.4f} | (iii) 对角={s['diag_lin3']:.4f} 非对角={s['off_lin3']:.4f} "
          f"Δ={s['delta_lin3']:+.4f}±{s['delta_lin3_se']:.4f} | (i) 对角={s['diag_i_mean']:.4f} "
          f"非对角={s['off_i_mean']:.4f} | (i') 去均值 对角={s['diag_ic_mean']:.4f} "
          f"非对角={s['off_ic_mean']:.4f}", flush=True)
    hdr = "        " + " ".join(f"{b:>8s}" for b in bl)
    print(f"[MAT-{mode}-CKA]", flush=True)
    print(hdr, flush=True)
    for bi in bl:
        print("  " + f"{bi:>6s} " + " ".join(f"{M[(bi, bj)]['cka']:8.3f}" for bj in bl), flush=True)
    print(f"[MAT-{mode}-CKA 非对角明细] " + " ".join(
        f"{bi}/{bj}={M[(bi, bj)]['cka']:.3f}" for bi in bl for bj in bl if bi < bj), flush=True)
    print(f"[MAT-{mode}-(iii)映射后cos 5x5]", flush=True)
    print(hdr, flush=True)
    for bi in bl:
        print("  " + f"{bi:>6s} " + " ".join(
            f"{M[(bi, bj)]['lin_iii']['same']:8.3f}" for bj in bl), flush=True)
    print(f"[MAT-{mode}-(i')去均值cos 非对角] " + " ".join(
        f"{bi}/{bj}={M[(bi, bj)]['ic']['same']:.3f}" for bi in bl for bj in bl if bi < bj),
        flush=True)
    print(f"[MAT-{mode}-(iii)映射后cos 非对角] " + " ".join(
        f"{bi}/{bj}={M[(bi, bj)]['lin_iii']['same']:.3f}" for bi in bl for bj in bl if bi < bj),
        flush=True)
    print(f"[MAT-{mode}-gain=(iii)-(i) 非对角] " + " ".join(
        f"{bi}/{bj}={M[(bi, bj)]['gain_i']:+.3f}" for bi in bl for bj in bl if bi < bj), flush=True)

# ============================================================================
# ⑧ R29：随机配对必须给出非零差异（度量非常数）
# ============================================================================
X0 = MH[("add_2d", 1234, "S")]
Y0 = MH[("mul_2d", 1234, "S")]
C0 = cosmat(X0, Y0)
off0 = C0.clone()
off0.fill_diagonal_(float("nan"))
off0 = off0[~torch.isnan(off0)]
r29 = dict(off_std=float(off0.std(unbiased=True)), off_min=float(off0.min()),
           off_max=float(off0.max()), off_mean=float(off0.mean()),
           nz_frac=float((off0.abs() > 1e-6).float().mean()),
           cka_perm_std=float(np.std([cka_linear(X0, Y0[p]) for p in PERM], ddof=1)))
print(f"\n[R29] 随机配对非零性：add_2d/mul_2d 余弦非对角 n={off0.numel()} std={r29['off_std']:.4f} "
      f"range=[{r29['off_min']:.3f},{r29['off_max']:.3f}] |非零占比={r29['nz_frac']:.2f} | "
      f"CKA 置换对照 std={r29['cka_perm_std']:.4f} ⇒ {'度量非常数 ✓' if r29['off_std'] > 1e-6 else '★常数'}",
      flush=True)

WALL = time.time() - T0
summary = dict(device=DEV, cuda=torch.cuda.is_available(), K=Kk, nperm=NPERM, batch=BATCH,
               threads=torch.get_num_threads(), wall_s=WALL, sha={f"{b}@{s}": v for (b, s), v in SH.items()},
               ctrl={k: {kk: float(vv) for kk, vv in v.items()} for k, v in CTRL.items()},
               base={f"{b}@{m}": v for (b, m), v in BASE.items()}, mat={}, r29=r29)
for mode in ("S", "N"):
    summary["mat"][mode] = {f"{a}|{c}": w for (a, c), w in MAT[mode].items()}
    summary["mat"][mode + "_sum"] = MAT[mode + "_sum"]
OUT = os.environ.get("S29_OUT", "/tmp/29_cross_task_meme_summary.json")
with open(OUT, "w", encoding="utf-8") as f:
    json.dump(summary, f, ensure_ascii=False, indent=1)
print(f"\n[S29] 墙钟={WALL:.1f}s（{WALL / 60:.1f}min）| device={DEV} | 全程零训练零生成 | "
      f"结果 JSON={OUT}", flush=True)
