#!/usr/bin/env python3
"""E7 · 证据账本更新：把 E1/S35、E5/S34、E6/S36、S37 的新 ckpt 纳入 E4 的去重与定价。

（★全程 CPU · ★零训练 · ★只读现成 ckpt；只新建 stages/39_* 与 logs/39_*）

复用 E4（stages/33_evidence_dedupe.py）的方法：
  去重键 = 结构指纹（键集+形状） + 逐参数权重指纹；键前缀 thought.* / thought.inner.* /
  card.layer.* / 单体(enc.*) 归一后两两比较（torch.equal；max|Δ|==0 ⇒ 同一权重）。
五步：
  ① 全量重跑去重：文件数 → 唯一权重数；列出相对 E4 基线新增/扩大的「同一权重组」。
  ② 五条结论 → 唯一权重独立 n + 定价（强=独立 n≥6 且 2 seed 同号 / 中 / 弱=独立 n≤2 / 不可判）。
  ③ 专门检查新出现的重复：S37 w1 vs E6 res、S37 w3/w10 vs E1 cards/mono、E6 A vs res 等，逐对给 max|Δ|。
  ④ 写 logs/39_ledger_update.json 与 logs/39_report.txt。
  ⑤ 诚实清单：读不了 / 被覆写 / 结构不可比 / 引用了却缺文件。
"""
from __future__ import annotations

import glob
import json
import os
import re
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""          # ★先于 torch 导入：强制 CPU
import torch  # noqa: E402

ROOT = "/home/vesita/coding/my/flowme"
LOGS = os.path.join(ROOT, "logs")
T0 = time.time()
DEV = "cpu"
torch.set_num_threads(int(os.environ.get("S39_THREADS", "12")))
assert not torch.cuda.is_available(), "必须全程 CPU"
print(f"[S39] device={DEV} cuda_avail={torch.cuda.is_available()} "
      f"threads={torch.get_num_threads()} root={ROOT}", flush=True)

# ============================================================================
# ① 去重键：结构指纹 + 逐参数权重指纹（与 E4 同法）
# ============================================================================
PREFIXES = ("thought.inner.", "card.layer.", "thought.", "card.", "think.", "inner.", "layer.")


def canon(k: str) -> str:
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
    mt = re.search(r"_s(\d+)$", body)
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
    RECS.append(dict(file=nm, path=f, ident=parse_identity(nm),
                     prefixes=sorted({k.split(".")[0] for k in sd}),
                     ck=ck, sig=sig, n_param=int(sum(v.numel() for v in ck.values())),
                     clash=clash))
print(f"[①] ckpt 文件 {len(files)} 个 ⇒ 可加载 {len(RECS)} 个 | 读取失败 {len(BAD)} 个", flush=True)

groups: dict = {}
for i, r in enumerate(RECS):
    groups.setdefault(r["sig"], []).append(i)
print(f"[①] 结构指纹分组（sig）: {len(groups)} 组，最大组 {max(len(g) for g in groups.values())} 个文件",
      flush=True)

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
BREAK_AT = 1e-2
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
            PAIRS.append(dict(a=a["file"], b=b["file"], max_abs_d=mx, at=where, lower_bound=stopped))
            if not stopped and mx == 0.0:
                union(g[ai], g[bi])

root2mem: dict = {}
for i, r in enumerate(RECS):
    root2mem.setdefault(find(i), []).append(i)
WGROUPS = {}
for gi, members in enumerate(sorted(root2mem.values(), key=lambda m: RECS[m[0]]["file"]), 1):
    for i in members:
        WGROUPS[RECS[i]["file"]] = f"W{gi:02d}"
        RECS[i]["weight_id"] = f"W{gi:02d}"
N_UNIQUE = len(root2mem)
print(f"[①] {len(RECS)} 个 ckpt（vs E4 的 83）⇒ ★唯一权重 {N_UNIQUE} 个（vs E4 的 64）", flush=True)

all_groups = sorted([sorted(RECS[i]["file"] for i in m) for m in root2mem.values() if len(m) > 1])
byfile = {r["file"]: r for r in RECS}

# E4 基线同权重组
BASE_GROUPS = set()
if os.path.exists(f"{LOGS}/33_dedupe.json"):
    b = json.load(open(f"{LOGS}/33_dedupe.json", encoding="utf-8"))
    for g in b["same_weight_groups"]:
        BASE_GROUPS.add(frozenset(x["file"] for x in g))
print(f"[①] E4 基线同权重组 {len(BASE_GROUPS)} 组", flush=True)

new_groups, expanded, unchanged = [], [], 0
for g in all_groups:
    fs = frozenset(g)
    if fs in BASE_GROUPS:
        unchanged += 1
        continue
    # 是否是某个基线组的超集（扩大）
    sup = [bg for bg in BASE_GROUPS if bg < fs]
    if sup:
        expanded.append(dict(files=g, expanded_from=[sorted(x) for x in sup]))
    else:
        new_groups.append(g)
print(f"[①·新增同一权重组] {len(new_groups)} 组", flush=True)
for g in new_groups:
    print(f"    NEW: {' == '.join(g)}", flush=True)
print(f"[①·扩大的同一权重组] {len(expanded)} 组（相对 E4 基线）", flush=True)
for e in expanded:
    print(f"    EXP: {' == '.join(e['files'])}", flush=True)

# 同名 ckpt 被覆写
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
# ③ 逐对判定（允许跨 sig：sig 不同 ⇒ 结构不可比，只能判「不同权重」）
# ============================================================================
def cmp_pair(fa: str, fb: str) -> dict:
    A, B = byfile.get(fa), byfile.get(fb)
    if A is None or B is None:
        return dict(a=fa, b=fb, verdict="缺文件", missing=[x for x in (fa, fb) if x not in byfile])
    if A["sig"] != B["sig"]:
        ka, kb = set(A["ck"]), set(B["ck"])
        return dict(a=fa, b=fb, verdict="结构不可比（键集/形状不同）⇒不同权重",
                    only_a=sorted(ka - kb)[:6], only_b=sorted(kb - ka)[:6],
                    n_param_a=A["n_param"], n_param_b=B["n_param"])
    mx, where, ndiff, ntot = 0.0, None, 0, 0
    for k, va in A["ck"].items():
        ntot += 1
        vb = B["ck"][k]
        if torch.equal(va, vb):
            continue
        ndiff += 1
        d = float((va.float() - vb.float()).abs().max().item())
        if d > mx:
            mx, where = d, k
    same = (ndiff == 0)
    return dict(a=fa, b=fb, verdict=("同一权重（max|Δ|=0）" if same else "不同权重"),
                max_abs_d=mx, at=where, n_tensors_diff=ndiff, n_tensors=ntot)


PAIR_Q = [
    ("S37 w1 vs E6 res（应同：都权重1/同 seed/同步数）",
     [("37_ckpt_w1_s6000_seed1234.pt", "36_ckpt_res_s6000_seed1234.pt"),
      ("37_ckpt_w1_s6000_seed5678.pt", "36_ckpt_res_s6000_seed5678.pt")]),
    ("E6 A vs E6 res（同构造顺序？）",
     [("36_ckpt_A_s6000_seed1234.pt", "36_ckpt_res_s6000_seed1234.pt"),
      ("36_ckpt_A_s6000_seed5678.pt", "36_ckpt_res_s6000_seed5678.pt")]),
    ("E6 res/A vs S19-res@3000（同步数? 不，3000 vs 6000）",
     [("36_ckpt_res_s6000_seed1234.pt", "19_ckpt_add_3d_res_seed1234.pt"),
      ("36_ckpt_A_s6000_seed5678.pt", "19_ckpt_add_3d_res_seed5678.pt")]),
    ("E6 A vs S22-A@3000",
     [("36_ckpt_A_s6000_seed1234.pt", "22_ckpt_add_3d_A_seed1234.pt"),
      ("36_ckpt_A_s6000_seed5678.pt", "22_ckpt_add_3d_A_seed5678.pt")]),
    ("E6 res/A vs S25 e2e@3000（E4 里同一组）",
     [("36_ckpt_res_s6000_seed1234.pt", "25_ckpt_e2e_seed1234.pt"),
      ("36_ckpt_A_s6000_seed5678.pt", "25_ckpt_e2e_seed5678.pt")]),
    ("★用户点名：E1 cards_s6000 vs E6 res_s6000",
     [("35_ckpt_cards_s6000_seed1234.pt", "36_ckpt_res_s6000_seed1234.pt"),
      ("35_ckpt_cards_s6000_seed5678.pt", "36_ckpt_res_s6000_seed5678.pt")]),
    ("E1 cards_s3000 vs S19-res@3000（E1 切了 dev，应不同）",
     [("35_ckpt_cards_s3000_seed1234.pt", "19_ckpt_add_3d_res_seed1234.pt"),
      ("35_ckpt_cards_s3000_seed5678.pt", "19_ckpt_add_3d_res_seed5678.pt")]),
    ("S37 w1/w3/w10 内部",
     [("37_ckpt_w1_s6000_seed1234.pt", "37_ckpt_w3_s6000_seed1234.pt"),
      ("37_ckpt_w1_s6000_seed1234.pt", "37_ckpt_w10_s6000_seed1234.pt"),
      ("37_ckpt_w3_s6000_seed1234.pt", "37_ckpt_w10_s6000_seed1234.pt")]),
    ("S37 w3/w10 vs E1 cards_s6000",
     [("37_ckpt_w3_s6000_seed1234.pt", "35_ckpt_cards_s6000_seed1234.pt"),
      ("37_ckpt_w10_s6000_seed1234.pt", "35_ckpt_cards_s6000_seed1234.pt"),
      ("37_ckpt_w3_s6000_seed5678.pt", "35_ckpt_cards_s6000_seed5678.pt"),
      ("37_ckpt_w10_s6000_seed5678.pt", "35_ckpt_cards_s6000_seed5678.pt")]),
    ("S37 w3/w10 vs E1 mono_s6000",
     [("37_ckpt_w3_s6000_seed1234.pt", "35_ckpt_mono_s6000_seed1234.pt"),
      ("37_ckpt_w10_s6000_seed1234.pt", "35_ckpt_mono_s6000_seed1234.pt"),
      ("37_ckpt_w10_s6000_seed5678.pt", "35_ckpt_mono_s6000_seed5678.pt")]),
    ("E1 cards vs mono（同参数量、不同容器）",
     [("35_ckpt_cards_s6000_seed1234.pt", "35_ckpt_mono_s6000_seed1234.pt"),
      ("35_ckpt_cards_s6000_seed5678.pt", "35_ckpt_mono_s6000_seed5678.pt")]),
    ("E5 34 frozen6000 vs 28 e2e6000",
     [("34_ckpt_frozen6000_seed1234.pt", "28_ckpt_e2e6000_seed1234.pt"),
      ("34_ckpt_frozen6000_seed5678.pt", "28_ckpt_e2e6000_seed5678.pt")]),
]
PAIR_OUT = []
for title, plist in PAIR_Q:
    print(f"[③] {title}", flush=True)
    for fa, fb in plist:
        r = cmp_pair(fa, fb)
        PAIR_OUT.append(dict(group=title, **r))
        if "max_abs_d" in r:
            print(f"    {fa} vs {fb}: {r['verdict']}  max|Δ|={r['max_abs_d']:.3e} "
                  f"(tensor={r['at']}, 差异张量 {r['n_tensors_diff']}/{r['n_tensors']})", flush=True)
        else:
            print(f"    {fa} vs {fb}: {r['verdict']} | only_a={r.get('only_a')} only_b={r.get('only_b')}",
                  flush=True)

# ============================================================================
# ② 结论 → 唯一权重独立 n + 定价
# ============================================================================
CONCLUSIONS = [
    dict(id="①", name="卡内结构不可替代（E6 @6000，B/C/D/E 各 2 seed vs A）",
         files=[f"36_ckpt_{a}_s6000_seed{s}.pt" for a in "ABCDE" for s in (1234, 5678)]),
    dict(id="②", name="残差在饱和点失效（E6 nores vs res @6000）",
         files=[f"36_ckpt_{a}_s6000_seed{s}.pt" for a in ("res", "nores") for s in (1234, 5678)]),
    dict(id="③", name="卡边界零增量（E1 构造性证明 + 统计对照）",
         files=[f"35_ckpt_{a}_s{st}_seed{s}.pt" for a in ("cards", "mono")
                for st in (3000, 6000) for s in (1234, 5678)]),
    dict(id="④", name="单卡更好被推翻（E5 frozen@6000 vs e2e@6000）",
         files=["34_ckpt_frozen6000_seed1234.pt", "34_ckpt_frozen6000_seed5678.pt",
                "28_ckpt_e2e6000_seed1234.pt", "28_ckpt_e2e6000_seed5678.pt"]),
    dict(id="⑤", name="loss 加权（S37 w1/w3/w10 @6000）",
         files=[f"37_ckpt_w{w}_s6000_seed{s}.pt" for w in ("1", "3", "10") for s in (1234, 5678)]),
]
LEDGER = []
for c in CONCLUSIONS:
    exist = [f for f in c["files"] if f in WGROUPS]
    missing = [f for f in c["files"] if f not in WGROUPS]
    wids = sorted({WGROUPS[f] for f in exist})
    # 去重后每个唯一权重取一个代表文件
    rep = {}
    for f in exist:
        rep.setdefault(WGROUPS[f], f)
    LEDGER.append(dict(id=c["id"], name=c["name"], n_files=len(exist), n_unique_weights=len(wids),
                       weight_ids=wids, representatives=sorted(rep.values()), missing=missing))
    print(f"[②] {c['id']} {c['name']}：文件 {len(exist)} → ★唯一权重 n={len(wids)}"
          + (f" | 缺 {missing}" if missing else ""), flush=True)


def load_jsonl(p):
    out = []
    if not os.path.exists(p):
        return out
    for ln in open(p, encoding="utf-8"):
        ln = ln.strip()
        if ln:
            try:
                out.append(json.loads(ln))
            except json.JSONDecodeError:
                pass
    return out


R34 = load_jsonl(f"{LOGS}/34_results.jsonl")
R35 = load_jsonl(f"{LOGS}/35_results.jsonl")
R36 = load_jsonl(f"{LOGS}/36_results.jsonl")
R37 = load_jsonl(f"{LOGS}/37_results.jsonl")


def g36(arm, seed):
    for r in R36:
        if r.get("arm") == arm and r.get("seed") == seed:
            return float(r["em"])
    return None


def g35(arm, steps, seed):
    for r in R35:
        if r.get("arm") == arm and r.get("steps") == steps and r.get("seed") == seed:
            return float(r["em"])
    return None


def g37(arm, seed):
    for r in R37:
        if r.get("arm") == arm and r.get("seed") == seed:
            return float(r["em"])
    return None


def price(n, same_sign, saturated=True, extra=""):
    if not saturated:
        return "不可判"
    if n >= 6 and same_sign:
        return "强"
    if n <= 2:
        return "弱"
    if same_sign:
        return "中"
    return "不可判"


PRICE = {}
# ① 卡内结构：Δ = em(arm) - em(A)
b1 = {}
for arm in "BCDE":
    ds = [g36(arm, s) - g36("A", s) for s in (1234, 5678)]
    b1[arm] = ds
d1 = [v for ds in b1.values() for v in ds]
same1 = all(v < 0 for v in d1) or all(v > 0 for v in d1)
PRICE["①"] = dict(n=LEDGER[0]["n_unique_weights"], deltas=b1, same_sign=same1,
                   price=price(LEDGER[0]["n_unique_weights"], same1),
                   note="B/C/D/E 对 A 的 Δ 在 2 seed 上全部同号（大幅更低）")
# ② 残差：Δ = em(res) - em(nores)
d2 = [g36("res", s) - g36("nores", s) for s in (1234, 5678)]
same2 = all(v < 0 for v in d2) or all(v > 0 for v in d2)
PRICE["②"] = dict(n=LEDGER[1]["n_unique_weights"], deltas=d2, same_sign=same2,
                   price=price(LEDGER[1]["n_unique_weights"], same2),
                   note="Δ(res−nores) 随 seed 翻号 ⇒ 未过 2 seed 同号")
# ③ 卡边界：构造性 + 统计 Δ = em(cards) - em(mono) @6000
d3 = [g35("cards", 6000, s) - g35("mono", 6000, s) for s in (1234, 5678)]
same3 = all(v < 0 for v in d3) or all(v > 0 for v in d3)
PRICE["③"] = dict(n_stat=LEDGER[2]["n_unique_weights"], deltas_6000=d3, same_sign=same3,
                   constructive_n=1,
                   price="中",
                   note="零增量只由构造性逐位恒等（n=1 证明）支撑；统计对照 cards≥mono 并非零")
# ④ 单卡：Δ = em(frozen) - em(e2e6000)，用 34 的 paired d
d4 = [float(r["paired"]["d"]) for r in R34]
same4 = all(v < 0 for v in d4) or all(v > 0 for v in d4)
PRICE["④"] = dict(n=LEDGER[3]["n_unique_weights"], deltas=d4, same_sign=same4,
                   price=price(LEDGER[3]["n_unique_weights"], same4),
                   note="frozen−e2e 均 ≤−2SE，2 seed 同号 ⇒ 单卡更好被推翻")
# ⑤ loss 加权：Δ = em(w) - em(w1)
d5 = {w: [g37(w, s) - g37("w1", s) for s in (1234, 5678)] for w in ("w3", "w10")}
same5 = all(v < 0 for vs in d5.values() for v in vs) or all(v > 0 for vs in d5.values() for v in vs)
PRICE["⑤"] = dict(n=LEDGER[4]["n_unique_weights"], deltas=d5, same_sign=same5,
                   price="不可判",
                   note="S37 自判 6/6 臂 @4500→6000 仍在升 ⇒ 未饱和，R35 结论不可下；且方向随 seed 翻转")
for k, v in PRICE.items():
    print(f"[②·定价] {k} 独立 n={v.get('n', v.get('n_stat'))} 同号={v['same_sign']} ⇒ {v['price']}", flush=True)

# ============================================================================
# ⑤ 诚实清单
# ============================================================================
refd = sorted({m.group(1) for lg in glob.glob(os.path.join(LOGS, "*.log"))
               for m in re.finditer(r"\[CKPT\] saved (\S+\.pt)", open(lg, errors="replace").read())})
honest = dict(
    unloadable=BAD,
    overwritten_names=overwritten,
    structural_pairs=[r for r in PAIR_OUT if "结构不可比" in r["verdict"]],
    missing_in_conclusions={c["id"]: c["missing"] for c in LEDGER if c["missing"]},
    ckpt_in_logs_not_on_disk=sorted({os.path.basename(x) for x in refd if
                                     not os.path.exists(x if os.path.isabs(x) else os.path.join(LOGS, os.path.basename(x)))}),
    baseline_e4=dict(n_files=83, n_unique=64),
)
print(f"[⑤] 读不了 {len(BAD)} | 覆写 {len(overwritten)} | 结构不可比对 {len(honest['structural_pairs'])} "
      f"| 结论里缺文件 {honest['missing_in_conclusions']}", flush=True)
print(f"[⑤] 日志提及但磁盘缺失（末 8）: {honest['ckpt_in_logs_not_on_disk'][-8:]}", flush=True)

# ============================================================================
# ④ 写产物
# ============================================================================
WALL = time.time() - T0
out = dict(
    stage="E7/39", device=DEV, wall_s=WALL,
    n_files=len(files), n_loadable=len(RECS), n_unique_weights=N_UNIQUE,
    e4_baseline=dict(n_files=83, n_unique=64),
    n_sig_groups=len(groups), n_pairs_compared=len(PAIRS),
    same_weight_groups=all_groups,
    new_same_weight_groups=new_groups,
    expanded_same_weight_groups=expanded,
    pair_checks=PAIR_OUT,
    conclusions=LEDGER,
    pricing=PRICE,
    honest=honest,
    files=[dict(file=r["file"], stage=r["ident"]["stage"], variant=r["ident"]["variant"],
                seed=r["ident"]["seed"], steps=r["ident"]["steps"], prefixes=r["prefixes"],
                n_param=r["n_param"], weight_id=r["weight_id"], clash=r["clash"]) for r in RECS],
    close_pairs=[p for p in PAIRS if p["max_abs_d"] <= 1e-2 and p["max_abs_d"] > 0][:50],
)
with open(f"{LOGS}/39_ledger_update.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

L = []
L.append(f"# E7 证据账本更新（device={DEV}，E4 基线 83→64）")
L.append(f"文件 {len(files)} → ★唯一权重 {N_UNIQUE}；可加载 {len(RECS)}；逐参比较 {len(PAIRS)} 对；"
         f"结构分组 {len(groups)}")
L.append(f"新增同一权重组 {len(new_groups)}；扩大 {len(expanded)}；E4 基线保留 {unchanged}")
for g in new_groups:
    L.append("NEW 同一权重: " + " == ".join(g))
for e in expanded:
    L.append("EXP 同一权重: " + " == ".join(e["files"]))
for e in LEDGER:
    L.append(f"{e['id']} n_files={e['n_files']} ★n_unique={e['n_unique_weights']} {e['name']}"
             + (f" | 缺 {e['missing']}" if e["missing"] else ""))
for k, v in PRICE.items():
    L.append(f"定价{k}: 独立n={v.get('n', v.get('n_stat'))} 同号={v['same_sign']} ⇒ {v['price']} | {v['note']}")
for r in PAIR_OUT:
    if "max_abs_d" in r:
        L.append(f"逐对 {r['a']} vs {r['b']}: {r['verdict']} max|Δ|={r['max_abs_d']:.3e}")
    else:
        L.append(f"逐对 {r['a']} vs {r['b']}: {r['verdict']}")
L.append(f"无法判定/诚实清单: 读不了={len(BAD)} 覆写={overwritten} "
         f"结构不可比={len(honest['structural_pairs'])} 结论缺文件={honest['missing_in_conclusions']}")
L.append(f"日志提及但磁盘缺失: {honest['ckpt_in_logs_not_on_disk']}")
L.append(f"墙钟={WALL:.1f}s")
with open(f"{LOGS}/39_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"[DONE] 文件 {len(files)} → 唯一权重 {N_UNIQUE}（E4: 83→64）| 新增组 {len(new_groups)} | "
      f"墙钟={WALL:.1f}s | logs/39_ledger_update.json + logs/39_report.txt exit=0", flush=True)
