#!/usr/bin/env python3
"""E4 · 独立证据账本：把 logs/ 下全部 ckpt 按「逐参数权重指纹」去重，重算受影响统计量。
（★全程 CPU · ★零训练 · ★只读现成 ckpt；只写 logs/33_* 或 /tmp/33_*）

五步：
  ① 去重键 = (stage, variant, seed, steps, 结构指纹(sig), 逐参数权重指纹)；
     权重指纹 = 两两 state_dict 逐张量 max|Δ|（键名前缀映射：thought.inner./card.layer./thought./card. 等），
     max|Δ| == 0 ⇒ 同一权重。★不是文件 sha（前缀不同 sha 不同但权重可逐位相同）。
  ② 把引用多 ckpt 互证的主要结论，换成唯一权重后报「真实独立样本数 n」。
  ③ S30 的 ρ：用 /tmp/s30/points.jsonl 重算集中度，分别在 n=23（原文件全格）与 n=18（去重后）上算
     ρ(PR_pool(h), 集中度) · ρ(PR, EM) · ρ(集中度, EM)（Spearman + p）。
  ④ 给结论重新定价：强（独立 n≥6 且 2 seed 同号）/ 中 / 弱（独立 n≤2）/ 不可判。
  ⑤ 诚实清单：无法判定身份的文件（缺 ckpt、被覆写的同名 ckpt、只有日志没有权重）。
"""
from __future__ import annotations

import glob
import json
import os
import re
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ★先于 torch 导入：强制 CPU
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

ROOT = "/home/vesita/coding/my/flowme"
LOGS = os.path.join(ROOT, "logs")
S30 = "/tmp/s30"
T0 = time.time()
DEV = "cpu"
torch.set_num_threads(int(os.environ.get("S33_THREADS", "12")))
assert not torch.cuda.is_available(), "必须全程 CPU"


def pick_outdir() -> str:
    for d in (LOGS, "/tmp"):
        try:
            p = os.path.join(d, "33_writetest.tmp")
            with open(p, "w", encoding="utf-8") as f:
                f.write("x")
            os.remove(p)
            return d
        except OSError:
            continue
    raise SystemExit("[FATAL] logs/ 与 /tmp 都不可写")


OUT = pick_outdir()
print(f"[S33] device={DEV} cuda_avail={torch.cuda.is_available()} out={OUT} "
      f"threads={torch.get_num_threads()}", flush=True)

# ============================================================================
# ① 去重键：结构指纹 + 逐参数权重指纹
# ============================================================================
PREFIXES = ("thought.inner.", "card.layer.", "thought.", "card.", "think.", "inner.", "layer.")


def canon(k: str) -> str:
    """键名前缀归一：把 thought.* / card.layer.* / thought.inner.* 映射到同一裸参数名。"""
    for p in PREFIXES:
        if k.startswith(p):
            return k[len(p):]
    return k


def parse_identity(name: str) -> dict:
    base = name[:-3] if name.endswith(".pt") else name
    m = re.match(r"^(\d+[a-z]?)_ckpt_(.*)$", base)
    stage, body = (m.group(1), m.group(2)) if m else ("?", base)
    seed = steps = None
    ms = re.search(r"seed(\d+)", body)
    if ms:
        seed = int(ms.group(1))
        body = body[: ms.start()].rstrip("_")
    mt = re.search(r"_s(\d+)$", body)          # 如 d128_s10000
    if mt:
        steps = int(mt.group(1))
        body = body[: mt.start()]
    return dict(stage=stage, variant=body, seed=seed, steps=steps)


files = sorted(glob.glob(os.path.join(LOGS, "*.pt")))
RECS, BAD = [], []
for f in files:
    nm = os.path.basename(f)
    try:
        sd = torch.load(f, map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        BAD.append(dict(file=nm, why=f"{type(e).__name__}: {e}"))
        continue
    if not isinstance(sd, dict):
        BAD.append(dict(file=nm, why=f"top-level={type(sd).__name__}"))
        continue
    ck, clash = {}, []
    for k, v in sd.items():
        c = canon(k)
        if c in ck and c != k:
            clash.append(k)
        ck[c] = v
    sig = tuple(sorted((k, tuple(v.shape)) for k, v in ck.items()))
    RECS.append(dict(
        file=nm, path=f, ident=parse_identity(nm),
        prefixes=sorted({k.split(".")[0] for k in sd}),
        ck=ck, sig=sig, n_param=int(sum(v.numel() for v in ck.values())),
        clash=clash,
    ))
print(f"[①] ckpt 文件 {len(files)} 个 ⇒ 可加载 {len(RECS)} 个 | 读取失败 {len(BAD)} 个", flush=True)

# 结构指纹（sig）分组：同一签名才可能逐位相同
groups: dict = {}
for i, r in enumerate(RECS):
    groups.setdefault(r["sig"], []).append(i)

parent = list(range(len(RECS)))


def find(a):
    while parent[a] != a:
        parent[a] = parent[parent[a]]
        a = parent[a]
    return a


def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)


PAIRS = []
BREAK_AT = 1e-2       # 判异阈值：一旦某张量 max|Δ|>1e-2 就停（该值记为下界），只对近邻对算全局 max
for g in groups.values():
    if len(g) < 2:
        continue
    for ai in range(len(g)):
        for bi in range(ai + 1, len(g)):
            a, b = RECS[g[ai]], RECS[g[bi]]
            mx, where, stopped = 0.0, None, False
            for k, va in a["ck"].items():
                vb = b["ck"][k]
                if torch.equal(va, vb):
                    continue
                d = float((va.float() - vb.float()).abs().max().item())
                if d > mx:
                    mx, where = d, k
                if mx > BREAK_AT:
                    stopped = True
                    break
            PAIRS.append(dict(a=a["file"], b=b["file"], max_abs_d=mx,
                              at=where, lower_bound=stopped))
            if not stopped and mx == 0.0:
                union(g[ai], g[bi])

root2id, WGROUPS = {}, {}
for i, r in enumerate(RECS):
    root2id.setdefault(find(i), []).append(i)
for gi, members in enumerate(sorted(root2id.values(), key=lambda m: RECS[m[0]]["file"]), 1):
    wid = f"W{gi:02d}"
    for i in members:
        WGROUPS[RECS[i]["file"]] = wid
        RECS[i]["weight_id"] = wid
print(f"[①] {len(RECS)} 个 ckpt ⇒ ★唯一权重 {len(root2id)} 个（sha 会高估：前缀不同 sha 不同）", flush=True)

dup_groups = [sorted(RECS[i]["file"] for i in m) for m in root2id.values() if len(m) > 1]
for g in dup_groups:
    pre = {}
    for f in g:
        pre.setdefault(tuple(RECS[[r["file"] for r in RECS].index(f)]["prefixes"]), []).append(f)
    print(f"[①·同一权重] {g}  键前缀={[list(p) for p in pre]}", flush=True)

# 同名 ckpt 被覆写（同一路径出现两个 sha16）⇒ 磁盘上的权重只对应最后一次
saved = {}
for lg in glob.glob(os.path.join(LOGS, "*.log")):
    try:
        txt = open(lg, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    for m in re.finditer(r"\[CKPT\] saved (\S+\.pt) sha256\[:16\]=([0-9a-f]{16})", txt):
        saved.setdefault(os.path.basename(m.group(1)), set()).add(m.group(2))
overwritten = {k: sorted(v) for k, v in saved.items() if len(v) > 1}
print(f"[⑤] 同名 ckpt 被覆写（日志里 ≥2 个 sha16）: {overwritten}", flush=True)

# ============================================================================
# ② 结论 → ckpt 引用 → 唯一权重数 n
# ============================================================================
BUCKETS = ("add_1d", "add_2d", "add_3d", "sub_2d", "mul_2d")
S19_20 = [f"19_ckpt_{b}_{arm}_seed{s}.pt" for b in BUCKETS for arm in ("res", "nores")
          for s in (1234, 5678)]
TAG2FILE = {
    "S14/add_1d/1234": "14_ckpt_add_1d_seed1234.pt", "S14/add_1d/5678": "14_ckpt_add_1d_seed5678.pt",
    "S14/add_2d/1234": "14_ckpt_add_2d_seed1234.pt", "S14/add_3d/1234": "14_ckpt_add_3d_seed1234.pt",
    "S14/add_3d/5678": "14_ckpt_add_3d_seed5678.pt", "S14/sub_2d/1234": "14_ckpt_sub_2d_seed1234.pt",
    "S14/mul_2d/1234": "14_ckpt_mul_2d_seed1234.pt",
    "S19/add_1d/res/1234": "19_ckpt_add_1d_res_seed1234.pt", "S19/add_2d/res/1234": "19_ckpt_add_2d_res_seed1234.pt",
    "S19/add_3d/res/1234": "19_ckpt_add_3d_res_seed1234.pt", "S19/sub_2d/res/5678": "19_ckpt_sub_2d_res_seed5678.pt",
    "S19/mul_2d/res/5678": "19_ckpt_mul_2d_res_seed5678.pt", "S19/add_1d/nores/1234": "19_ckpt_add_1d_nores_seed1234.pt",
    "S19/add_3d/nores/1234": "19_ckpt_add_3d_nores_seed1234.pt", "S19/add_3d/nores/5678": "19_ckpt_add_3d_nores_seed5678.pt",
    "S20/add3d/pos/1234": "20_ckpt_add3d_pos_seed1234.pt", "S20/add3d/nopos/1234": "20_ckpt_add3d_nopos_seed1234.pt",
    "S20/add3d/pos/5678": "20_ckpt_add3d_pos_seed5678.pt", "S25/e2e/1234": "25_ckpt_e2e_seed1234.pt",
    "S25/anchor/1234": "25_ckpt_anchor_seed1234.pt", "S25/alt/5678": "25_ckpt_alt_seed5678.pt",
    "S25/randtgt/1234": "25_ckpt_randtgt_seed1234.pt", "S25/randall/1234": "25_ckpt_randall_seed1234.pt",
}
A22 = [f"22_ckpt_add_3d_{a}_seed{s}.pt" for a in "ABCDE" for s in (1234, 5678)]
A25 = [f"25_ckpt_{a}_seed{s}.pt" for a in ("e2e", "anchor", "randtgt", "alt", "randall") for s in (1234, 5678)]

CONC = [
    dict(id="K1", name="残差必要性·『S15 与 S19 两次独立复现』",
         files=["15_ckpt_add3d_always_k1_seed1234.pt", "15_ckpt_add3d_always_k1_seed5678.pt",
                "19_ckpt_add_3d_res_seed1234.pt", "19_ckpt_add_3d_res_seed5678.pt",
                "19_ckpt_add_3d_nores_seed1234.pt", "19_ckpt_add_3d_nores_seed5678.pt"]),
    dict(id="K2", name="S22 五臂卡内结构对照（A/B/C/D/E ×2 seed）", files=A22),
    dict(id="K3", name="S25 单卡四臂+randall（e2e/anchor/randtgt/alt/randall ×2 seed）", files=A25),
    dict(id="K4", name="S19 五桶 × res/nores（20 run，桶普遍性）", files=S19_20),
    dict(id="K5", name="S30 ρ(PR,EM) / ρ(集中度,EM)", files=sorted(set(TAG2FILE.values()))),
    dict(id="K6", name="S31 SEEN/UNSEEN 残差效应（add_3d res/nores + 28_e2e6000）",
         files=["19_ckpt_add_3d_res_seed1234.pt", "19_ckpt_add_3d_res_seed5678.pt",
                "19_ckpt_add_3d_nores_seed1234.pt", "19_ckpt_add_3d_nores_seed5678.pt",
                "28_ckpt_e2e6000_seed1234.pt", "28_ckpt_e2e6000_seed5678.pt"]),
    dict(id="K7", name="S22-A『逐位复现』S19-res（当作独立互证）",
         files=["22_ckpt_add_3d_A_seed1234.pt", "19_ckpt_add_3d_res_seed1234.pt"]),
    dict(id="K8", name="S25 锚点增益（臂2 anchor vs 臂3 randtgt，+27.0/+9.4pp）",
         files=["25_ckpt_anchor_seed1234.pt", "25_ckpt_anchor_seed5678.pt",
                "25_ckpt_randtgt_seed1234.pt", "25_ckpt_randtgt_seed5678.pt"]),
    dict(id="K9", name="S13 容量/步数不是瓶颈（d×2.3、步数×2）",
         files=["13_ckpt_d128_s10000_seed1234.pt", "13_ckpt_d128_s10000_seed5678.pt",
                "13_ckpt_d256_s3000_seed1234.pt", "13_ckpt_d256_s3000_seed5678.pt"]),
    dict(id="K10", name="S22-A 集中度（前20% 41.2/41.5%）",
         files=["22_ckpt_add_3d_A_seed1234.pt", "22_ckpt_add_3d_A_seed5678.pt"]),
]
LEDGER = []
for c in CONC:
    exist = [f for f in c["files"] if f in WGROUPS]
    missing = [f for f in c["files"] if f not in WGROUPS]
    wids = sorted({WGROUPS[f] for f in exist})
    per_seed = {s: len({WGROUPS[f] for f in exist if f.endswith(f"seed{s}.pt")}) for s in (1234, 5678)}
    LEDGER.append(dict(id=c["id"], name=c["name"], n_files=len(exist), n_missing=len(missing),
                       n_unique_weights=len(wids), weight_ids=wids, per_seed=per_seed,
                       missing=missing))
    print(f"[②] {c['id']} {c['name']}：文件 {len(exist)} → ★唯一权重 n={len(wids)} "
          f"(seed1234:{per_seed[1234]}, seed5678:{per_seed[5678]})"
          + (f" | 缺 {missing}" if missing else ""), flush=True)

# ============================================================================
# ③ S30 的 ρ：n=23（原文件全格）vs n=18（去重后）
# ============================================================================
RHO = dict(available=os.path.exists(f"{S30}/points.jsonl") and os.path.exists(f"{S30}/summary.json"))
if RHO["available"]:
    rows23 = json.load(open(f"{S30}/summary.json", encoding="utf-8"))["rows"]
    pts = [json.loads(l) for l in open(f"{S30}/points.jsonl", encoding="utf-8") if l.strip()]
    tag_order = [r["tag"] for r in rows23]

    def share20(ce, p=0.2):
        ce = np.asarray(ce, float)
        cs = ce[np.argsort(-ce, kind="stable")]
        cc = np.cumsum(cs) / cs.sum()
        return float(cc[max(1, int(round(p * len(ce)))) - 1])

    conc_re = {t: share20([x["ce"] for x in pts if x["tag"] == t]) for t in tag_order}
    dmax = max(abs(conc_re[r["tag"]] - r["conc"]) for r in rows23)
    print(f"[③] points.jsonl 重算集中度 vs summary.json：max|Δ|={dmax:.2e} "
          f"({'一致 ⇒ 可用 points 独立复算' if dmax < 1e-9 else '★不一致'})", flush=True)

    # 23 个 tag 各自对应哪个权重
    tag_wid = {t: WGROUPS.get(TAG2FILE[t]) for t in tag_order if t in TAG2FILE}
    uniq_wids = sorted({w for w in tag_wid.values() if w})
    seen_w, keep_tags = set(), []
    for t in tag_order:                       # 每个唯一权重只保留一行（代表）
        w = tag_wid.get(t)
        if w in seen_w:
            continue
        seen_w.add(w)
        keep_tags.append(t)
    dup_map = {}
    for t, w in tag_wid.items():
        dup_map.setdefault(w, []).append(t)
    dup_map = {w: v for w, v in dup_map.items() if len(v) > 1}
    print(f"[③] S30 的 23 行 ⇒ 唯一权重 {len(uniq_wids)} 个；重复组 {dup_map}", flush=True)

    def calc(rows, label):
        pr = np.array([r["PR"] for r in rows], float)
        cc = np.array([conc_re[r["tag"]] for r in rows], float)
        em = np.array([r["em"] for r in rows], float)
        out = {}
        for nm, x, y in (("PR-conc", pr, cc), ("PR-EM", pr, em), ("conc-EM", cc, em)):
            r = spearmanr(x, y)
            out[nm] = dict(n=len(x), rho=float(r.statistic), p=float(r.pvalue))
        m20, m5 = em >= 0.20, em < 0.05
        for nm, m in (("PR-conc|EM>=20%", m20), ("PR-conc|EM<5%", m5)):
            r = spearmanr(pr[m], cc[m])
            out[nm] = dict(n=int(m.sum()), rho=float(r.statistic), p=float(r.pvalue))
        print(f"[③] {label}: " + " | ".join(f"{k} n={v['n']} ρ={v['rho']:+.4f} p={v['p']:.4f}"
                                            for k, v in out.items()), flush=True)
        return out

    RHO["n23_all"] = calc(rows23, "n=23（summary.json 全格，未去重）")
    RHO["n18_dedup"] = calc([r for r in rows23 if r["tag"] in keep_tags], "n=18（唯一权重去重后）")
    RHO["reported"] = dict(PR_conc=-0.135, PR_EM=+0.575, conc_EM=-0.046)
    RHO["tag2weight"] = tag_wid
    RHO["dup_map"] = {w: v for w, v in dup_map.items()}
    RHO["conc_recheck_maxabs"] = dmax
else:
    print(f"[③] ★ /tmp/s30/points.jsonl 或 summary.json 不存在 ⇒ S30 无法重算", flush=True)

# ============================================================================
# 写产物
# ============================================================================
du = dict(
    device=DEV, n_files=len(files), n_loadable=len(RECS), n_unique_weights=len(root2id),
    bad=BAD, overwritten_names=overwritten,
    same_weight_groups=[[dict(file=f, prefix=RECS[[r["file"] for r in RECS].index(f)]["prefixes"])
                         for f in g] for g in dup_groups],
    files=[dict(file=r["file"], stage=r["ident"]["stage"], variant=r["ident"]["variant"],
                seed=r["ident"]["seed"], steps=r["ident"]["steps"], prefixes=r["prefixes"],
                n_param=r["n_param"], weight_id=r["weight_id"], key_clash=r["clash"]) for r in RECS],
    close_pairs=[p for p in PAIRS if p["max_abs_d"] <= 1e-2][:50],
    n_pairs_compared=len(PAIRS),
)
with open(f"{OUT}/33_dedupe.json", "w", encoding="utf-8") as f:
    json.dump(du, f, ensure_ascii=False, indent=1)
with open(f"{OUT}/33_rho.json", "w", encoding="utf-8") as f:
    json.dump(dict(RHO, ledger=LEDGER, wall_s=time.time() - T0), f, ensure_ascii=False, indent=1)

lines = []
lines.append(f"# E4 独立证据账本（device={DEV}）")
lines.append(f"文件 {len(files)} → 唯一权重 {len(root2id)}；两两逐参数比较 {len(PAIRS)} 对")
for g in dup_groups:
    lines.append("同一权重: " + " == ".join(g))
for e in LEDGER:
    lines.append(f"{e['id']} n_files={e['n_files']} n_unique={e['n_unique_weights']} {e['name']}")
if RHO.get("available"):
    for lab in ("n23_all", "n18_dedup"):
        lines.append(lab + ": " + json.dumps({k: round(v["rho"], 4) for k, v in RHO[lab].items() if k in
                                              ("PR-conc", "PR-EM", "conc-EM")}, ensure_ascii=False))
lines.append(f"无法判定: {BAD} | 覆写: {overwritten}")
lines.append(f"墙钟={time.time()-T0:.1f}s")
with open(f"{OUT}/33_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")
print(f"[DONE] out={OUT}/33_dedupe.json,33_rho.json,33_report.txt 墙钟={time.time()-T0:.1f}s exit=0", flush=True)
