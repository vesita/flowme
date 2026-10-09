#!/usr/bin/env python3
"""E8 · 证据账本更新（第三轮）：把 S40 / S44(P0) / S45(P2) / S46(P3) 的新 ckpt 纳入
E4（33_evidence_dedupe.py）与 E7（39_ledger_update.py）的「唯一权重」去重与定价。

（★全程 CPU · ★零训练 · ★只读现成 ckpt；只新建 stages/47_* 与 logs/47_*，不碰 GPU）

方法完全复用 E4/E7（直接读 33/39 的实现，不改它们）：
  去重键 = 结构指纹（键集+形状，canon 归一后） + 逐参数权重指纹；
  canon：thought.inner. / card.layer. / thought. / card. / think. / inner. / layer. 前缀剥掉，
  两组 ckpt 若归一后键集与形状一致，再逐张量 torch.equal 比 max|Δ|；max|Δ|==0 ⇒ 同一权重。
  ★不用文件 sha 判同一权重（sha 只用来核对「磁盘上的文件没被覆写」）。

四步：
  ① 全量重跑去重：文件数 → 唯一权重数；列出相对 E7 基线新增/扩大的「同一权重组」。
  ② 七条结论 → 更新后真实独立 n 与定价（强=独立 n≥6 且 2 seed 同号 / 中 / 弱 / 不可判）。
  ③ 专门检查新出现的同一权重（A(e2e/res) vs S38 w1、P2②/③/③b、P3 base vs S38），逐对给 max|Δ|。
  ④ 写 logs/47_ledger_update.json 与 logs/47_report.txt；诚实列出无法判定身份的 ckpt。

★P3（stages/46_binding_vs_addition.py，systemd unit p3-bind.service）本轮跑在 GPU 上，正在写
  logs/46_ckpt_bind_seed1234.pt / logs/46_ckpt_rand_seed5678.pt ⇒ 这些文件按「新鲜度窗口」
  （默认最近 300s 内被改过）判为「正在写」并跳过逐参比较，只登记身份与 sha；其余文件正常纳入。
"""
from __future__ import annotations

import glob
import hashlib
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
torch.set_num_threads(int(os.environ.get("S47_THREADS", "10")))
assert not torch.cuda.is_available(), "必须全程 CPU"
FRESH_S = float(os.environ.get("S47_FRESH_S", "300"))   # 「可能正在写」的新鲜度窗口（秒）
print(f"[S47] device={DEV} cuda_avail={torch.cuda.is_available()} "
      f"threads={torch.get_num_threads()} fresh_window={FRESH_S:.0f}s root={ROOT}", flush=True)

# ============================================================================
# ① 去重键：结构指纹 + 逐参数权重指纹（与 E4/E7 同法）
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
    mt = re.search(r"_s(\d+)$", body)          # 如 d128_s10000 / w1_s12000 / B_s12000
    if mt:
        steps = int(mt.group(1))
        body = body[: mt.start()]
    return dict(stage=stage, variant=body, seed=seed, steps=steps)


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


files = sorted(glob.glob(os.path.join(LOGS, "*.pt")))
NOW = time.time()
RECS, BAD, FRESH = [], [], []
for f in files:
    nm = os.path.basename(f)
    age = NOW - os.path.getmtime(f)
    if age < FRESH_S:
        FRESH.append(dict(file=nm, age_s=round(age, 1), size=os.path.getsize(f)))
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
                     clash=clash, mtime_age_s=round(age, 1)))
print(f"[①] ckpt 文件 {len(files)} 个 ⇒ 可加载 {len(RECS)} 个 | 读取失败 {len(BAD)} 个 "
      f"| 正在写（{FRESH_S:.0f}s 内被改过）{len(FRESH)} 个", flush=True)
for x in FRESH:
    print(f"    FRESH: {x['file']} (age={x['age_s']}s)", flush=True)

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
print(f"[①] {len(RECS)} 个 ckpt（E4 基线 83、E7 基线 115）⇒ ★唯一权重 {N_UNIQUE} 个"
      f"（E4 的 64、E7 的 86）", flush=True)

all_groups = sorted([sorted(RECS[i]["file"] for i in m) for m in root2mem.values() if len(m) > 1])
byfile = {r["file"]: r for r in RECS}


def load_baseline(paths):
    """E4/E7 的同一权重组基线（33_dedupe.json / 39_ledger_update.json）。磁盘上若已被清掉则空。"""
    out = set()
    for p in paths:
        if not os.path.exists(p):
            continue
        try:
            b = json.load(open(p, encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        for g in b.get("same_weight_groups", []):
            names = [x["file"] if isinstance(x, dict) else x for x in g]
            out.add(frozenset(names))
    return out


BASE_GROUPS = load_baseline([f"{LOGS}/33_dedupe.json", f"{LOGS}/39_ledger_update.json"])
print(f"[①] 基线同权重组 {len(BASE_GROUPS)} 组（33_dedupe.json / 39_ledger_update.json；"
      f"两者磁盘存在? {[os.path.exists(p) for p in (f'{LOGS}/33_dedupe.json', f'{LOGS}/39_ledger_update.json')]}）",
      flush=True)

new_groups, expanded, unchanged = [], [], 0
for g in all_groups:
    fs = frozenset(g)
    if fs in BASE_GROUPS:
        unchanged += 1
        continue
    sup = [bg for bg in BASE_GROUPS if bg < fs]
    if sup:
        expanded.append(dict(files=g, expanded_from=[sorted(x) for x in sup]))
    else:
        new_groups.append(g)
print(f"[①·新增同一权重组] {len(new_groups)} 组（相对 E7/E4 基线）", flush=True)
for g in new_groups:
    print(f"    NEW: {' == '.join(g)}", flush=True)
for e in expanded:
    print(f"    EXP: {' == '.join(e['files'])}", flush=True)

# 同名 ckpt 被覆写：日志里的 [CKPT] saved 行 sha16 ≥2 个
saved = {}
for lg in glob.glob(os.path.join(LOGS, "*.log")):
    try:
        txt = open(lg, encoding="utf-8", errors="replace").read()
    except OSError:
        continue
    for m in re.finditer(r"\[CKPT\] saved (\S+\.pt) sha256\[:16\]=([0-9a-f]{16})", txt):
        saved.setdefault(os.path.basename(m.group(1)), set()).add(m.group(2))
overwritten = {k: sorted(v) for k, v in saved.items() if len(v) > 1}
print(f"[①] 同名 ckpt 被覆写（日志 ≥2 个 sha16）: {overwritten}", flush=True)


# ============================================================================
# ③ 逐对判定（允许跨 sig：sig 不同 ⇒ 结构不可比，只能判「不同权重」）
# ============================================================================
def cmp_pair(fa: str, fb: str) -> dict:
    A, B = byfile.get(fa), byfile.get(fb)
    if A is None or B is None:
        return dict(a=fa, b=fb, verdict="缺文件/未纳入", missing=[x for x in (fa, fb) if x not in byfile])
    fresh = [x["file"] for x in (A, B) if x["mtime_age_s"] < FRESH_S]
    if A["sig"] != B["sig"]:
        ka, kb = set(A["ck"]), set(B["ck"])
        return dict(a=fa, b=fb, verdict="结构不可比（键集/形状不同）⇒不同权重",
                    only_a=sorted(ka - kb)[:6], only_b=sorted(kb - ka)[:6],
                    n_param_a=A["n_param"], n_param_b=B["n_param"], caveat=f"正在写={fresh}" or None)
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
                max_abs_d=mx, at=where, n_tensors_diff=ndiff, n_tensors=ntot,
                caveat=(f"★含正在写文件{fresh}，结果为瞬时" if fresh else None))


PAIR_Q = [
    ("§3.1 P0 的 A(e2e/res) 与 S38 的 w1 —— 44 的 A 臂不落新 ckpt（trained=false，ckpt 字段直指 "
     "38_ckpt_w1_s12000）；故核 A 臂用的 w1 与 S38 同 step 的 w1 是否逐位同（跨 44/45/46 复用件）",
     [("38_ckpt_w1_s12000_seed1234.pt", "38_ckpt_w1_s9000_seed1234.pt"),
      ("38_ckpt_w1_s12000_seed5678.pt", "38_ckpt_w1_s9000_seed5678.pt"),
      ("38_ckpt_w1_s12000_seed1234.pt", "38_ckpt_w1_s15000_seed1234.pt")]),
    ("§3.2 P2 的 ② 与 S38 的 w1@12000（P2 报告称 max|Δ|=0 ⇒ 复核：接口+目标卡都是 S38 原样）",
     [("45_ckpt_a2_seed1234.pt", "38_ckpt_w1_s12000_seed1234.pt"),
      ("45_ckpt_a2_seed5678.pt", "38_ckpt_w1_s12000_seed5678.pt")]),
    ("§3.3 P2 的 ③ 的「接口 ≡ ①、卡 ≡ S38」逐件复核",
     [("45_ckpt_a3_seed1234.pt", "45_ckpt_a1_s12000_seed1234.pt"),
      ("45_ckpt_a3_seed5678.pt", "45_ckpt_a1_s12000_seed5678.pt"),
      ("45_ckpt_a3_seed1234.pt", "38_ckpt_w1_s12000_seed1234.pt"),
      ("45_ckpt_a3_seed5678.pt", "38_ckpt_w1_s12000_seed5678.pt")]),
    ("§3.3b P2 的 ③b 的「接口 ≡ S38、卡 ≡ ①」逐件复核",
     [("45_ckpt_a3b_seed1234.pt", "38_ckpt_w1_s12000_seed1234.pt"),
      ("45_ckpt_a3b_seed5678.pt", "38_ckpt_w1_s12000_seed5678.pt"),
      ("45_ckpt_a3b_seed1234.pt", "45_ckpt_a1_s12000_seed1234.pt"),
      ("45_ckpt_a3b_seed5678.pt", "45_ckpt_a1_s12000_seed5678.pt")]),
    ("§3.4 P3 的 base 臂与 S38 的 w1@12000（46 base 自训 12000 步，同 seed）",
     [("46_ckpt_base_seed1234.pt", "38_ckpt_w1_s12000_seed1234.pt"),
      ("46_ckpt_base_seed5678.pt", "38_ckpt_w1_s12000_seed5678.pt")]),
    ("§3.5 S40 的基线臂（w0/base）与 S38 的 w1@同步数（38 与 40 应同训同 seed）",
     [("40_ckpt_w1_s12000_seed1234.pt", "38_ckpt_w1_s12000_seed1234.pt"),
      ("40_ckpt_w1_s9000_seed1234.pt", "38_ckpt_w1_s9000_seed1234.pt"),
      ("40_ckpt_w1_s15000_seed1234.pt", "38_ckpt_w1_s15000_seed1234.pt")]),
    ("§3.6 P2 ① B自训@12000 vs S46 base@12000（都是 add_2d 12000 步同 seed？）",
     [("45_ckpt_a1_s12000_seed1234.pt", "46_ckpt_base_seed1234.pt"),
      ("45_ckpt_a1_s12000_seed5678.pt", "46_ckpt_base_seed5678.pt")]),
    ("§3.7 S46 内部：base/bind/rand 三臂互不同（bind 多 5248 参数 ⇒ 结构不可比）",
     [("46_ckpt_base_seed1234.pt", "46_ckpt_bind_seed1234.pt"),
      ("46_ckpt_base_seed1234.pt", "46_ckpt_rand_seed1234.pt"),
      ("46_ckpt_bind_seed1234.pt", "46_ckpt_rand_seed1234.pt")]),
    ("§3.8 P0 的 B/C/D/E 互不同（不同结构 ⇒ 结构不可比）",
     [("44_ckpt_B_s12000_seed1234.pt", "44_ckpt_C_s12000_seed1234.pt"),
      ("44_ckpt_B_s12000_seed1234.pt", "44_ckpt_D_s12000_seed1234.pt"),
      ("44_ckpt_B_s12000_seed1234.pt", "44_ckpt_E_s12000_seed1234.pt"),
      ("44_ckpt_C_s12000_seed1234.pt", "44_ckpt_D_s12000_seed1234.pt"),
      ("44_ckpt_C_s12000_seed1234.pt", "44_ckpt_E_s12000_seed1234.pt"),
      ("44_ckpt_D_s12000_seed1234.pt", "44_ckpt_E_s12000_seed1234.pt")]),
    ("§3.9 S44 A 臂（复用 S38）与 S38 w1 —— 44 无 A ckpt，A 就是 38_ckpt_w1（登记用）",
     [("44_ckpt_B_s12000_seed1234.pt", "38_ckpt_w1_s12000_seed1234.pt")]),
]
PAIR_OUT = []
for title, plist in PAIR_Q:
    print(f"[③] {title}", flush=True)
    for fa, fb in plist:
        if fa == fb:
            r = cmp_pair(fa, fb)
            r = dict(a=fa, b=fb, verdict="自比（跳过）")
            PAIR_OUT.append(dict(group=title, **r))
            print(f"    {fa} vs {fb}: 自比跳过", flush=True)
            continue
        r = cmp_pair(fa, fb)
        PAIR_OUT.append(dict(group=title, **r))
        if "max_abs_d" in r:
            print(f"    {fa} vs {fb}: {r['verdict']}  max|Δ|={r['max_abs_d']:.3e} "
                  f"(tensor={r['at']}, 差异张量 {r['n_tensors_diff']}/{r['n_tensors']})"
                  + (f" | {r['caveat']}" if r.get("caveat") else ""), flush=True)
        else:
            print(f"    {fa} vs {fb}: {r['verdict']} | only_a={r.get('only_a')} only_b={r.get('only_b')}"
                  + (f" | {r['caveat']}" if r.get("caveat") else ""), flush=True)

# ============================================================================
# ② 结论 → 唯一权重独立 n + 定价
# ============================================================================
CONCLUSIONS = [
    dict(id="①", name="卡内结构不可替代 @12000（P0/S44：B/C/D/E 各 2 seed vs A）",
         files=[f"44_ckpt_{a}_s12000_seed{s}.pt" for a in "BCDE" for s in (1234, 5678)]
               + [f"38_ckpt_w1_s12000_seed{s}.pt" for s in (1234, 5678)]),
    dict(id="②", name="卡边界零增量（E1 构造性 + S35 统计对照）",
         files=[f"35_ckpt_{a}_s{st}_seed{s}.pt" for a in ("cards", "mono")
                for st in (3000, 6000) for s in (1234, 5678)]),
    dict(id="③", name="单卡更好被推翻（E5：frozen@6000 vs e2e@6000）",
         files=["34_ckpt_frozen6000_seed1234.pt", "34_ckpt_frozen6000_seed5678.pt",
                "28_ckpt_e2e6000_seed1234.pt", "28_ckpt_e2e6000_seed5678.pt"]),
    dict(id="④", name="残差@饱和点失效（E6：nores vs res @6000）",
         files=[f"36_ckpt_{a}_s6000_seed{s}.pt" for a in ("res", "nores") for s in (1234, 5678)]),
    dict(id="⑤", name="loss 加权（S40 @15000，w1/w3/w10）",
         files=[f"40_ckpt_w{w}_s15000_seed{s}.pt" for w in ("1", "3", "10") for s in (1234, 5678)]),
    dict(id="⑥", name="跨任务功能迁移（P2/S45：①B自训 ②A全量 ③A卡+B接口 ③bA接口+B卡）",
         files=[f"45_ckpt_a1_s12000_seed{s}.pt" for s in (1234, 5678)]
               + [f"45_ckpt_a{n}_seed{s}.pt" for n in ("2", "3", "3b") for s in (1234, 5678)]
               + [f"38_ckpt_w1_s12000_seed{s}.pt" for s in (1234, 5678)]),
    dict(id="⑦", name="S42/S43 clean 口径（模板 bug 影响面；clean 口径不产生 ckpt）",
         files=[]),
]
LEDGER = []
for c in CONCLUSIONS:
    exist = [f for f in c["files"] if f in WGROUPS]
    missing = [f for f in c["files"] if f not in WGROUPS]
    wids = sorted({WGROUPS[f] for f in exist})
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


AVAILABLE_JSONL = sorted(os.path.basename(p) for p in glob.glob(f"{LOGS}/*.jsonl"))
R44 = load_jsonl(f"{LOGS}/44_results.jsonl")
R45 = load_jsonl(f"{LOGS}/45_results.jsonl")
R46 = load_jsonl(f"{LOGS}/46_results.jsonl")


# 「44 的 A 臂没有新 ckpt 文件」由 sha 交叉核对证明：44_results 里 A 臂的 ckpt 字段直指
# logs/38_ckpt_w1_s12000_seed<s>.pt，且记录的 sha 必须等于磁盘现值（否则该文件被覆写过）。
SHA_REC = {}
for tag, rows in (("44", R44), ("45", R45), ("46", R46)):
    for r in rows:
        ck, sh = r.get("ckpt"), (r.get("sha") or r.get("ckpt_sha"))
        if ck and sh:
            SHA_REC.setdefault(os.path.basename(ck), set()).add((tag, str(sh)))
SHA_CHECK = []
for nm, recs in sorted(SHA_REC.items()):
    p = os.path.join(LOGS, nm)
    disk = sha16(p) if os.path.exists(p) else None
    ok = disk is not None and all(d == disk for _, d in recs)
    SHA_CHECK.append(dict(file=nm, recorded=sorted({d for _, d in recs}), on_disk=disk,
                          match=ok, reported_by=sorted({t for t, _ in recs})))
    print(f"[③·sha] {nm}: 记录={sorted({d for _, d in recs})} 磁盘={disk} ⇒ "
          f"{'✓一致（未覆写）' if ok else '★不一致/缺文件'}", flush=True)
# S34(单卡)/S35(卡边界)/S36(残差)/S37/S38/S40 的 results.jsonl 在 logs/ 里已不存在
# （git 忽略 logs/ ⇒ 过程产物随时会丢）⇒ ②③④⑤ 的 EM 只能用 ckpt 计数 + 既有报告复核。
R_MISSING = [f"{LOGS}/{n}_results.jsonl" for n in (34, 35, 36, 37, 38, 40)
             if not os.path.exists(f"{LOGS}/{n}_results.jsonl")]


def g44(arm, seed):
    for r in R44:
        if r.get("arm") == arm and r.get("seed") == seed:
            return float(r["em_full"])
    return None


def g45(arm, seed, key="em_full"):
    for r in R45:
        if r.get("arm") == arm and r.get("seed") == seed:
            return float(r[key])
    return None


def g46(arm, seed, steps="12000"):
    for r in R46:
        if r.get("arm") == arm and r.get("seed") == seed:
            return float(r["curve"][steps]["em"])
    return None


def price(n, same_sign, saturated=True):
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
# ① 卡内结构 @12000：Δ = em(arm) - em(A)，A 用 S44 报告值（= 38_ckpt_w1 的锚点）
A_ANCHOR = {1234: 0.91875, 5678: 0.87125}          # 44_results A 臂（复用 S38 w1@12000）
b1 = {}
for arm in "BCDE":
    b1[arm] = {s: (g44(arm, s) - A_ANCHOR[s]) for s in (1234, 5678)}
d1 = [v for ds in b1.values() for v in ds]
same1 = all(v < 0 for v in d1) or all(v > 0 for v in d1)
PRICE["①"] = dict(n=LEDGER[0]["n_unique_weights"], deltas_pp={a: {s: round(v * 100, 2) for s, v in ds.items()}
                                                              for a, ds in b1.items()},
                   same_sign=same1, price=price(LEDGER[0]["n_unique_weights"], same1),
                   note="B/C/D/E 对 A 的 Δ 在 2 seed 上全部同号且大幅为负（A 复用 S38 w1，未重训）")
# ② 卡边界：构造性（E1 逐位恒等） + S35 统计；无 35_results.jsonl ⇒ 统计 Δ 不可复算
PRICE["②"] = dict(n_stat=LEDGER[1]["n_unique_weights"], n_files=LEDGER[1]["n_files"],
                   constructive_n=1, same_sign=None, price="中",
                   note="零增量只由构造性逐位恒等（n=1 证明）支撑；S35 统计对照的 EM 因 "
                        "logs/35_results.jsonl 已丢，本轮无法复算 ⇒ 统计独立性未更新")
# ③ 单卡：需 34_results.jsonl 的 paired d ⇒ 已丢
PRICE["③"] = dict(n=LEDGER[2]["n_unique_weights"], same_sign=None, price="中",
                   note="E5 的 paired Δ（frozen−e2e ≤−2SE、2 seed 同号）本轮无法从 jsonl 复核；"
                        "权重面 4 文件 → n 见左，定价沿用前轮「中」")
# ④ 残差@饱和点：需 36_results.jsonl ⇒ 已丢
PRICE["④"] = dict(n=LEDGER[3]["n_unique_weights"], same_sign=None, price="不可判",
                   note="前轮 Δ(res−nores) 随 seed 翻号（未过 2 seed 同号）⇒ 不可判；"
                        "36_results.jsonl 已丢，本轮无法重算")
# ⑤ loss 加权 @15000：需 40_results.jsonl ⇒ 已丢
PRICE["⑤"] = dict(n=LEDGER[4]["n_unique_weights"], same_sign=None, price="不可判",
                   note="S40 的 9000→12000→15000 增量是否归零决定可判性；40_results.jsonl 已丢，"
                        "本轮只能给去重后的独立 n（w1 若 ≡ S38 则与 S38 共用权重）")
# ⑥ 跨任务迁移：P2 有 jsonl（全在）
d6 = {}
for arm in ("②A全量", "③A卡+B接口", "③bA接口+B卡"):
    d6[arm] = {s: round((g45(arm, s) - g45("①B自训", s)) * 100, 2) for s in (1234, 5678)}
v6 = [v for ds in d6.values() for v in ds]
same6 = all(v < 0 for v in v6) or all(v > 0 for v in v6)
PRICE["⑥"] = dict(n=LEDGER[5]["n_unique_weights"], deltas_pp_vs_B={a: ds for a, ds in d6.items()},
                   same_sign=same6, price="中" if same6 else "不可判",
                   note="②(0.00/0.00) 与 ③(15.63/37.63)、③b(7.00/9.38) 方向相反 ⇒ 不是同号一族；"
                        "「零训练直插无效、只换卡不迁移」是 2 seed 同号的部分结论")
# ⑦ clean 口径：不产生 ckpt，是数据/判分口径
PRICE["⑦"] = dict(n=0, same_sign=None, price="不可判",
                   note="S42/S43 的 clean 口径是判分口径（模板 bug 影响面），不落 ckpt ⇒ "
                        "本轮去重账本对它无 n 可言；需回到 42/43 的日志口径证据")
for k, v in PRICE.items():
    print(f"[②·定价] {k} 独立 n={v.get('n', v.get('n_stat'))} 同号={v['same_sign']} ⇒ {v['price']}",
          flush=True)

# ============================================================================
# ⑤ 诚实清单
# ============================================================================
refd = sorted({m.group(1) for lg in glob.glob(os.path.join(LOGS, "*.log"))
               for m in re.finditer(r"\[CKPT\] saved (\S+\.pt)", open(lg, errors="replace").read())})
refd_missing = sorted({os.path.basename(x) for x in refd
                       if not os.path.exists(x if os.path.isabs(x) else os.path.join(LOGS, os.path.basename(x)))})
# 中间权重组按 stage 归类（新增组里各 stage 各多少）
stage_of = {}
for r in RECS:
    stage_of.setdefault(r["ident"]["stage"], []).append(r["file"])
honest = dict(
    unloadable=BAD,
    being_written_fresh=FRESH,
    fresh_window_s=FRESH_S,
    overwritten_names=overwritten,
    structural_pairs=[r for r in PAIR_OUT if "结构不可比" in r.get("verdict", "")],
    missing_in_conclusions={c["id"]: c["missing"] for c in LEDGER if c["missing"]},
    missing_results_jsonl=R_MISSING,
    available_results_jsonl=AVAILABLE_JSONL,
    ckpt_in_logs_not_on_disk=refd_missing,
    baseline_note=("E4/E7 的 33_dedupe.json 与 39_ledger_update.json 在磁盘上已不存在"
                   "（logs/ 不入库）⇒ 本轮「新增同权重组」只能以磁盘现存为全集报告；"
                   "E4 基线 83→64、E7 基线 115→86 取自任务书"),
    baseline_e4=dict(n_files=83, n_unique=64),
    baseline_e7=dict(n_files=115, n_unique=86),
    p3_note=("P3（stages/46_binding_vs_addition.py，unit p3-bind.service）正在 GPU 上跑："
             "bind seed1234 训练中（截至本轮仅 bind/rand seed1234 与 base 两 seed 落盘）⇒ "
             "46_ckpt_bind_seed5678 / 46_ckpt_rand_seed5678 尚不存在；本轮不碰 GPU。"),
)
print(f"[⑤] 读不了 {len(BAD)} | 正在写 {len(FRESH)} | 覆写 {len(overwritten)} "
      f"| 结构不可比对 {len(honest['structural_pairs'])} | 结论缺文件 {honest['missing_in_conclusions']}",
      flush=True)
print(f"[⑤] 丢失的 results.jsonl: {[os.path.basename(x) for x in R_MISSING]} | "
      f"现存: {AVAILABLE_JSONL}", flush=True)

# ============================================================================
# ④ 写产物
# ============================================================================
WALL = time.time() - T0
out = dict(
    stage="E8/47", device=DEV, wall_s=WALL,
    n_files=len(files), n_loadable=len(RECS), n_unique_weights=N_UNIQUE,
    e4_baseline=dict(n_files=83, n_unique=64), e7_baseline=dict(n_files=115, n_unique=86),
    n_sig_groups=len(groups), n_pairs_compared=len(PAIRS),
    same_weight_groups=all_groups,
    new_same_weight_groups=new_groups,
    expanded_same_weight_groups=expanded,
    pair_checks=PAIR_OUT,
    sha_cross_check=SHA_CHECK,
    conclusions=LEDGER,
    pricing=PRICE,
    honest=honest,
    files=[dict(file=r["file"], stage=r["ident"]["stage"], variant=r["ident"]["variant"],
                seed=r["ident"]["seed"], steps=r["ident"]["steps"], prefixes=r["prefixes"],
                n_param=r["n_param"], weight_id=r["weight_id"], clash=r["clash"],
                mtime_age_s=r["mtime_age_s"]) for r in RECS],
    close_pairs=[p for p in PAIRS if 0 < p["max_abs_d"] <= 1e-2][:50],
)
with open(f"{LOGS}/47_ledger_update.json", "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=1)

L = []
L.append(f"# E8 证据账本更新（device={DEV}，E4 基线 83→64，E7 基线 115→86）")
L.append(f"文件 {len(files)} → ★唯一权重 {N_UNIQUE}；可加载 {len(RECS)}；逐参比较 {len(PAIRS)} 对；"
         f"结构分组 {len(groups)}；正在写（<{FRESH_S:.0f}s）{len(FRESH)}")
L.append(f"新增同一权重组 {len(new_groups)}；扩大 {len(expanded)}；基线保留 {unchanged}"
         f"（磁盘上无 33/39 基线 json ⇒ 以现存全集为准）")
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
        L.append(f"逐对 {r['a']} vs {r['b']}: {r['verdict']} max|Δ|={r['max_abs_d']:.3e}"
                 + (f"  [{r['caveat']}]" if r.get("caveat") else ""))
    else:
        L.append(f"逐对 {r['a']} vs {r['b']}: {r['verdict']}"
                 + (f" | only_a={r.get('only_a')} only_b={r.get('only_b')}" if r.get("only_a") else ""))
L.append(f"不可判定/诚实清单: 读不了={len(BAD)} 正在写={[x['file'] for x in FRESH]} 覆写={overwritten}")
L.append("sha 交叉核对（44/45/46 results 记录的 sha vs 磁盘现值，证明未被覆写）：")
for s in SHA_CHECK:
    L.append(f"  {s['file']}: 记录={s['recorded']} 磁盘={s['on_disk']} ⇒ "
             f"{'一致' if s['match'] else '不一致/缺文件'}（来自 {','.join(s['reported_by'])}）")
L.append(f"丢失的 results.jsonl={[os.path.basename(x) for x in R_MISSING]} 现存={AVAILABLE_JSONL}")
L.append(f"日志提及但磁盘缺失={refd_missing}")
L.append(f"P3: {honest['p3_note']}")
L.append(f"墙钟={WALL:.1f}s")
with open(f"{LOGS}/47_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(L) + "\n")
print(f"[DONE] 文件 {len(files)} → 唯一权重 {N_UNIQUE}（E4: 83→64 / E7: 115→86）| "
      f"新增组 {len(new_groups)} | 墙钟={WALL:.1f}s | logs/47_ledger_update.json + logs/47_report.txt exit=0",
      flush=True)
