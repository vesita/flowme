#!/usr/bin/env python3
"""S51 · ★P4 门2（训练路径等价性）：用 `cards/` 从零训臂 A，与 `stages/` 的 oracle 路径从零训臂 A 比 EM。

门0（数据路径）与门1（前向/评估，灌已训 ckpt ⇒ logits max|Δ|==0.0）已过（提交 193de67 / 27c7b39）。
本单元只回答：**从零训练**时，两条路径是否等价（门1 只检了前向）。

设计（最小、单臂 A、单 seed 1234、12000 步 = R36 门槛）：
  · 臂 A = 现状结构：`TargetCard(residual=True)`（卡内 TransformerEncoderLayer + 外层残差）；
  · oracle 路径 = `stages/44_struct_at_12k.py` 的 **原文本 exec**（CardA / Cards44）+ `stages/19` 的
    `masked_ce`（emb → in_enc → card.layer → head）；
  · cards 路径 = `cards/example_add3d.py` 的 **原文本 exec**（数据 + train_run 的两条随机流结构）
    的 `CardPipeline`（input.emb/enc → target.think → output.head），device=cuda（同设备才谈逐位）；
  · 训练循环由本文件写一份（逐字镜像 `stages/19:452-471` / `stages/44:317-338` /
    `cards/example_add3d.py:316-336`），两条路径共用**同一个循环体**，只换模型与其 `masked_ce`
    ⇒ 循环本身不是变量（见 §③ 的 30 步保真度控制：本循环 vs S19 真 train_run 逐位比）。

★诚实边界（写死）：
  1 门0 复认：两侧 TEST sha256[:16] == b34e7ea515203227、泄漏 16/800、130 batch 索引序列逐位同；
  2 两条随机流结构（R34）：两侧「模型构造后的 CPU 全局状态」必须逐位同（⇒ 重洗同源）；
    GPU 上 dropout 抽 CUDA 流（`torch.manual_seed` 已一并设 CUDA 种子），两侧前向后
    CPU 全局流都必须原封不动、CUDA 流都必须被消耗且消耗后状态相同；
  3 ★「从零训练的逐位等价」**不预设成立**：先做 30 步的重复性控制（同一段 oracle 代码跑两次）。
    若 oracle-vs-oracle 就逐位不同 ⇒ GPU 训练本身非确定 ⇒ **明确写「训练路径的逐位等价不可验证」**，
    只报统计判据（EM Δ）与收敛曲线形状，**不许声称逐位一致**；
    若 oracle-vs-oracle 逐位同，再看 cards-vs-oracle（同则报「逐位同」，不同则定位）。

判据（写死）：|ΔEM| ≤ 0.13pp ⇒ 等价；≤ 2pp ⇒ 近似等价（须报曲线形状差异）；> 2pp ⇒ 不等价（定位）。
并报收敛曲线 @3000/6000/9000/12000 与饱和检查（9000→12000 增量 ≥1pp ⇒ 未饱和、结论条件性）。

只允许写：本文件、logs/p4_gate2.log、logs/51_*、/tmp。cards/ 与 stages/ 其它脚本只读、不改。
运行（device 必须 cuda）：
  systemd-run --user --unit=p4-gate2 --collect --property=WorkingDirectory=<ROOT> \
    --property=StandardOutput=append:<ROOT>/logs/p4_gate2.log \
    --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
    uv run --no-sync python stages/51_p4_gate2.py
调试：S51_SMOKE=1 ⇒ 30 步 / 60 条评测 / 不落 ckpt（落 /tmp）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

ROOT = "/home/vesita/coding/my/flowme"
sys.path.insert(0, ROOT)
os.environ.setdefault("HSA_OVERRIDE_GFX_VERSION", "10.3.0")
os.environ["CARDS_DEVICE"] = "cuda"                 # ★同设备：cards 侧也走 GPU
os.environ.setdefault("CARDS_RNG", "s19-two-stream")  # ★R34：dropout 独立流 + 重洗全局流

SRC19 = f"{ROOT}/stages/19_residual_scan.py"
SRC44 = f"{ROOT}/stages/44_struct_at_12k.py"
SRCCARDS = f"{ROOT}/cards/example_add3d.py"
SMOKE = os.environ.get("S51_SMOKE") == "1"

SEED = 1234
STEPS = 30 if SMOKE else 12000
GRID = (10, 20, 30) if SMOKE else (3000, 6000, 9000, 12000)
REPORT_AT = tuple(st for st in (3000, 6000, 9000, 12000) if st in GRID) or (GRID[-1],)
TOL_EQ = 0.13          # pp = 1/800
TOL_NEAR = 2.0
PROBE_STEPS = 30
TEST_SHA_EXPECT = "b34e7ea515203227"
LEAK_EXPECT = 16
CLEAN_BAD_EXPECT = 30
JSON_OUT = "/tmp/51_gate2_smoke.json" if SMOKE else f"{ROOT}/logs/51_gate2.json"
CKPT_TMPL = ("/tmp/51_ckpt_{kind}_s{steps}_seed{seed}.pt" if SMOKE
             else f"{ROOT}/logs/51_ckpt_{{kind}}_s{STEPS}_seed{{seed}}.pt")

T0 = time.time()
print(f"[S51] ★P4 门2：cards/ 从零训臂 A vs stages/ oracle 从零训臂 A | steps={STEPS} "
      f"grid={GRID} seed={SEED} | smoke={SMOKE} | cards DEVICE={os.environ['CARDS_DEVICE']} "
      f"RNG={os.environ['CARDS_RNG']} | HSA_OVERRIDE_GFX_VERSION="
      f"{os.environ.get('HSA_OVERRIDE_GFX_VERSION')}", flush=True)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402

# ============================================================================
# 0. 原文本 exec 工具（不改源文本；S19 头要求 device==cuda，本单元本来就在 GPU 上跑）
# ============================================================================
def _span(path: str, a: str, b: str):
    src = open(path, encoding="utf-8").read()
    i = src.index(a)
    j = src.index(b, i)
    return src[i:j], src.count("\n", 0, i) + 1, src.count("\n", 0, j) + 1


def _exec_span(path: str, a: str, b: str, ns: dict) -> str:
    txt, l0, l1 = _span(path, a, b)
    exec(compile(txt, path, "exec"), ns)          # noqa: S102 - 只读原文本
    return f"{os.path.basename(path)}:{l0}-{l1 - 1}"


def _exec_head(path: str, cut_marker: str, ns: dict) -> str:
    src = open(path, encoding="utf-8").read()
    cut = src.index(cut_marker)
    exec(compile(src[:cut], path, "exec"), ns)    # noqa: S102
    return f"{os.path.basename(path)}:1-{src.count(chr(10), 0, cut)}"


def _sha_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def sd_maxdiff(a: dict, b: dict):
    """同一组键的张量逐位比较 ⇒ (max|Δ|, 不同张量数, 不同元素数)。"""
    if list(a.keys()) != list(b.keys()):
        i = next((j for j, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
        raise AssertionError(f"键序不同 @{i}：{list(a)[i:i + 2]} vs {list(b)[i:i + 2]}")
    worst, nbad, nel = 0.0, 0, 0
    for k in a:
        assert a[k].shape == b[k].shape, f"{k} 形状不同 {a[k].shape} vs {b[k].shape}"
        if a[k].numel() == 0:
            continue
        d = (a[k].float() - b[k].float()).abs()
        worst = max(worst, float(d.max()))
        nb = int((d > 0).sum())
        nbad += int(nb > 0)
        nel += nb
    return worst, nbad, nel


CARDS2ORACLE = (("input.emb.", "emb."), ("input.enc.", "in_enc."),
                ("target.think.", "card.layer."), ("output.head.", "head."))


def cards_to_oracle(sd: dict) -> dict:
    """cards 管线键名 → stages Cards44 键名（门1 官定映射；顺序 = AdamW 参数顺序）。"""
    out = {}
    for k, v in sd.items():
        for a, b in CARDS2ORACLE:
            if k.startswith(a):
                out[b + k[len(a):]] = v
                break
        else:
            raise AssertionError(f"未覆盖的 cards 键 {k!r}")
    return out


# ============================================================================
# ① 原文本 exec：S19 头（数据/超参/评估口径，与 S36/S44 同源）
# ============================================================================
print(f"\n[S51] ===== ① exec {SRC19}（到 '# ③ 主循环'）=====", flush=True)
S: dict = {"__name__": "s19_head", "__file__": SRC19}
head19 = _exec_head(SRC19, "# ③ 主循环", S)
D, FF, NHEAD, MAXLEN = S["D"], S["FF"], S["NHEAD"], S["MAXLEN"]
V, PAD_ID, EOS_ID = S["V"], S["PAD_ID"], S["EOS_ID"]
device, LR, BATCH = S["device"], S["LR"], S["BATCH"]
EM_BATCH, CHK_BATCH, GEN_BATCH, MAX_GEN = S["EM_BATCH"], S["CHK_BATCH"], S["GEN_BATCH"], S["MAX_GEN"]
DATA, build_batch, greedy_gen = S["DATA"], S["build_batch"], S["greedy_gen"]
parse_ans, hits_from, r28_selfcheck = S["parse_ans"], S["hits_from"], S["r28_selfcheck"]
ce_categories, eval_ce, mean_se = S["ce_categories"], S["eval_ce"], S["mean_se"]
TRAIN, TEST = DATA["add_3d"]["train"], DATA["add_3d"]["test"]
N_EVAL = 60 if SMOKE else len(TEST)
TEST_EVAL = TEST[:N_EVAL]
assert str(device).startswith("cuda"), f"device={device} ⇒ 门2 必须在 cuda 上跑（同设备才谈逐位）"
print(f"[S51-EXEC] {head19} ⇒ d={D} ff={FF} NHEAD={NHEAD} MAXLEN={MAXLEN} V={V} PAD={PAD_ID} "
      f"EOS={EOS_ID} lr={LR} batch={BATCH} EM口径=batch{EM_BATCH} | add_3d train={len(TRAIN)} "
      f"test={len(TEST)} | device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__}",
      flush=True)

# ============================================================================
# ② 原文本 exec：oracle（stages/44 的卡与容器）+ cards 侧（cards/example_add3d.py 的前半段）
# ============================================================================
print(f"\n[S51] ===== ② exec oracle（{SRC44}）与 cards 侧（{SRCCARDS}）=====", flush=True)
N44 = dict(S)
N44.update(ROOT=ROOT, SMOKE44=False)
s44a = _exec_span(SRC44, "FP = 0.1", "assert str(device)", N44)
s44b = _exec_span(SRC44, "class CardA(nn.Module):", "def train44(", N44)
assert N44["SMOKE_SEEDS"] == (1234,) and N44["STEPS44"] == 12000, "S44 常量漂移"
Cards44 = N44["Cards44"]

C: dict = {"__name__": "cards_head", "__file__": SRCCARDS}
cards_head = _exec_head(SRCCARDS, "# ③ 交叉核对", C)
assert C["DEVICE"] == "cuda" and C["RNG_MODE"] == "s19-two-stream", "cards 侧 device/RNG 结构漂移"
CardPipeline, cards_masked_ce, DROP_GEN = C["CardPipeline"], C["masked_ce"], C["DROP_GEN"]
print(f"[S51-EXEC] {s44a} + {s44b} ⇒ oracle: Cards44/CardA + [S44-INIT] 自检 | "
      f"{cards_head} ⇒ cards: CardPipeline/masked_ce(两条流)", flush=True)
print(f"[S51-EXEC] 臂 A 描述 = {N44['ARM_DESC']['A']} | cards 侧 = "
      f"InputCard(emb+enc) → TargetCard(think, residual=True) → OutputCard(head)", flush=True)

# ============================================================================
# ③ 门0 复认（数据路径）+ 两条随机流结构（R34）
# ============================================================================
print(f"\n[S51] ===== ③ 前置检查 1–2：门0 复认 + 两条随机流 =====", flush=True)
_h = hashlib.sha256()
for r in TEST:
    _h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
SHA19 = _h.hexdigest()[:16]
_tp = {(r["a"], r["b"]) for r in TRAIN}
for r in TEST:
    r["seen"] = (r["a"], r["b"]) in _tp
LEAK19 = sum(1 for r in TEST if r["seen"])
BAD_IDX = [i for i, r in enumerate(TEST) if "{expr}" in r["nl"]]
CLEAN_IDX = [i for i in range(N_EVAL) if i not in set(BAD_IDX)]

train_c, test_c = C["build_data"]()
_hc = hashlib.sha256()
for r in test_c:
    _hc.update((r["nl"] + "||" + r["gold"] + "\n").encode())
SHA_C = _hc.hexdigest()[:16]
_tpc = {(r["a"], r["b"]) for r in train_c}
LEAK_C = sum(1 for r in test_c if (r["a"], r["b"]) in _tpc)
FIELDS = ("p", "t", "t_full", "gold", "a", "b", "y", "nl", "cut_p", "cut_t", "text")
bitdiff = -1
if (len(train_c), len(test_c)) == (len(TRAIN), len(TEST)):
    bitdiff = sum(1 for A_, B_ in zip(train_c + test_c, TRAIN + TEST) for f in FIELDS
                  if A_[f] != B_[f])
print(f"[S51-CHK1] 门0 复认 | TEST sha256[:16]：stages={SHA19} cards={SHA_C} 入库={TEST_SHA_EXPECT} "
      f"⇒ {'✓两侧同且=入库' if SHA19 == SHA_C == TEST_SHA_EXPECT else '★不一致'} | (a,b) 泄漏："
      f"stages={LEAK19}/800 cards={LEAK_C}/800 入库={LEAK_EXPECT} ⇒ "
      f"{'✓' if LEAK19 == LEAK_C == LEAK_EXPECT else '★不一致'} | clean bad{{expr}}={len(BAD_IDX)}"
      f"（入库 {CLEAN_BAD_EXPECT}）| train/test 11 字段逐位不同点={bitdiff}", flush=True)
assert SHA19 == SHA_C == TEST_SHA_EXPECT and LEAK19 == LEAK_C == LEAK_EXPECT and bitdiff == 0

# 模型构造后的两条流状态（S48 口径）
torch.manual_seed(SEED)
DROP_GEN.manual_seed(SEED)
_pc = CardPipeline(d=D, vocab_size=V, ff=FF, nhead=NHEAD, maxlen=MAXLEN, pad_id=PAD_ID,
                   dropout=0.1, residual=True).to(device)
ST_C, GC_C = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
_po = Cards44("A", D, FF).to(device)
ST_O, GC_O = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
same_st = bool(torch.equal(ST_C, ST_O))
same_gc = bool(torch.equal(GC_C, GC_O))
print(f"[S51-CHK2] 两条流结构（R34）：构造后 CPU 全局流 cards vs oracle "
      f"{'逐位同 ✓' if same_st else '★不同'} | CUDA 流 {'逐位同 ✓' if same_gc else '★不同'} "
      f"| cards 侧 = torch.manual_seed + DROP_GEN.manual_seed（dropout 独立）+ 重洗吃全局；"
      f"stages 侧 = torch.manual_seed(+cuda.manual_seed_all) + 重洗吃全局", flush=True)

N = len(TRAIN)


def _draw(state):
    torch.random.set_rng_state(state)
    order = torch.randperm(N, generator=torch.Generator().manual_seed(SEED))
    out, pos = [], 0
    for _ in range(130):
        row = []
        for _ in range(BATCH):
            if pos >= N:
                order = torch.randperm(N)
                pos = 0
            row.append(int(order[pos]))
            pos += 1
        out.append(row)
    return out


idx_c, idx_o = _draw(ST_C), _draw(ST_O)
div = next((b for b in range(130) if idx_c[b] != idx_o[b]), None)
print(f"[S51-CHK2] 130 batch 索引序列（含第 125 batch 首次重洗）："
      f"{'逐位相同 ✓' if div is None else f'★首个分岔 batch={div}'} | 前三 batch 前3索引 cards="
      f"{[r[:3] for r in idx_c[:3]]} oracle={[r[:3] for r in idx_o[:3]]}", flush=True)

# dropout 换流实测（两侧都必须在 GPU 上抽 CUDA 流、且不动 CPU 全局流）


def _stream_probe(kind, model):
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    ids, _ = build_batch(TRAIN[:8])
    c0, g0 = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    model.train()
    with torch.no_grad():
        if kind == "cards":
            C["logits_train"](model, ids)
        else:
            model.logits(ids)
    c1, g1 = torch.random.get_rng_state().clone(), torch.cuda.get_rng_state().clone()
    return dict(cpu_untouched=bool(torch.equal(c0, c1)), cuda_moved=not bool(torch.equal(g0, g1)),
                g_after=g1, out=None)


pc = _stream_probe("cards", _pc)
po = _stream_probe("oracle", _po)
same_consume = bool(torch.equal(pc["g_after"], po["g_after"]))
print(f"[S51-CHK2] dropout 流实测（train 前向）：cards CPU流不动={pc['cpu_untouched']} "
      f"CUDA流被消耗={pc['cuda_moved']} | oracle CPU流不动={po['cpu_untouched']} "
      f"CUDA流被消耗={po['cuda_moved']} | 消耗后 CUDA 流状态两侧"
      f"{'逐位同 ✓（=同一批 dropout 掩码）' if same_consume else '★不同'}", flush=True)

# ============================================================================
# ④ 逐位可行性探针（★不预设成立）：初始化 / 前向(eval) / 前向(train) / 30 步重复性
# ============================================================================
print(f"\n[S51] ===== ④ 前置检查 3：逐位对齐可行性 =====", flush=True)
sd_o0 = {k: v.detach().cpu().clone() for k, v in _po.state_dict().items()}
sd_c0 = cards_to_oracle({k: v.detach().cpu().clone() for k, v in _pc.state_dict().items()})
d_init, nb_init, ne_init = sd_maxdiff(sd_o0, sd_c0)
MODEL_INIT_SAME = (d_init == 0.0)

ids_eval, _ = build_batch(TEST_EVAL[:200])
_po.eval(); _pc.eval()
with torch.no_grad():
    lo, lc = _po.logits(ids_eval), _pc.logits(ids_eval)
D_FWD_EVAL = float((lo - lc).abs().max().item())
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)      # ★每次前向各自重置种子
_po.train()
with torch.no_grad():
    to_ = _po.logits(ids_eval)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
_pc.train()
with torch.no_grad():
    tl_ = C["logits_train"](_pc, ids_eval)
D_FWD_TRAIN = float((to_ - tl_).abs().max().item())
del lo, lc, to_, tl_, ids_eval
print(f"[S51-CHK3] 同 seed 初始化：oracle vs cards 键序逐位比 max|Δθ|={d_init:.3e}"
      f"（不同张量 {nb_init}/{len(sd_o0)}，不同元素 {ne_init}）⇒ "
      f"{'★逐位同' if MODEL_INIT_SAME else '★不同'} | eval 前向 logits max|Δ|={D_FWD_EVAL:.3e} | "
      f"train 前向（dropout 开）logits max|Δ|={D_FWD_TRAIN:.3e} ⇒ "
      f"{'dropout 掩码也同' if D_FWD_TRAIN == 0.0 else 'dropout/核有差'}", flush=True)
del _pc, _po
torch.cuda.empty_cache()

# ---- 30 步控制：本循环(oracle) vs S19 真 train_run(oracle) vs 本循环(cards) ----
def train_loop(kind: str, steps: int, grid, seed: int = SEED, trace: bool = False):
    """逐字镜像 stages/19:452-471 = stages/44:317-338 = cards/example_add3d.py:316-336 的训练循环；
    两条路径共用此循环体，只换模型与其 masked_ce。"""
    torch.manual_seed(seed)
    if kind == "oracle":
        torch.cuda.manual_seed_all(seed)
        model = Cards44("A", D, FF).to(device)
        ce = S["masked_ce"]
    else:
        DROP_GEN.manual_seed(seed)
        model = CardPipeline(d=D, vocab_size=V, ff=FF, nhead=NHEAD, maxlen=MAXLEN,
                             pad_id=PAD_ID, dropout=0.1, residual=True).to(device)
        ce = cards_masked_ce
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(TRAIN), generator=torch.Generator().manual_seed(seed))
    step, pos, t0, last = 0, 0, time.time(), 0.0
    gset, snaps, losses = set(grid), {}, {}
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(TRAIN))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = ce(model, [TRAIN[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if trace:
            losses[step] = float(loss.item())
        if step in gset:
            snaps[step] = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
            losses[step] = float(loss.item())
        if step % 1500 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train:{kind}] step={step}/{steps} loss={loss.item():.4f} elapsed={el:.0f}s",
                  flush=True)
            last = el
    assert set(grid) <= set(snaps), f"未取到全部曲线格点 {sorted(set(grid) - set(snaps))}"
    return model, time.time() - t0, snaps, losses


print(f"[S51] ----- 30 步控制：oracle 跑两次（本循环 vs S19 真 train_run）+ cards 本循环 -----",
      flush=True)
m_o1, _, sn_o1, lo_o1 = train_loop("oracle", PROBE_STEPS, (PROBE_STEPS,), trace=True)
m_o2, _ = S["train_run"](SEED, TRAIN, PROBE_STEPS, True)               # S19 原函数，同 30 步
m_c1, _, sn_c1, lo_c1 = train_loop("cards", PROBE_STEPS, (PROBE_STEPS,), trace=True)
P_O1 = {k: v.detach().cpu().clone() for k, v in m_o1.state_dict().items()}
P_O2 = {k.replace("thought.", "card.layer."): v.detach().cpu().clone()
        for k, v in m_o2.state_dict().items()}     # S19 Cards 的 thought.* → Cards44 的 card.layer.*
P_C1 = cards_to_oracle({k: v.detach().cpu().clone() for k, v in m_c1.state_dict().items()})
d_oo, nb_oo, ne_oo = sd_maxdiff(P_O1, P_O2)
d_co, nb_co, ne_co = sd_maxdiff(P_C1, P_O1)
LOSS_TRACE_EQ = all(abs(lo_o1[s] - lo_c1[s]) == 0.0 for s in lo_o1)
ORACLE_REPRO = (d_oo == 0.0)
print(f"[S51-CHK3] 30 步重复性（同一段 oracle 代码跑两次：本循环 vs S19 train_run）："
      f"max|Δθ|={d_oo:.3e}（不同张量 {nb_oo}/{len(P_O1)}，元素 {ne_oo}）⇒ "
      f"{'逐位可复现' if ORACLE_REPRO else '★GPU 训练本身逐位不可复现'}", flush=True)
print(f"[S51-CHK3] 30 步 cards vs oracle：max|Δθ|={d_co:.3e}（不同张量 {nb_co}，元素 {ne_co}）| "
      f"30 步 loss 逐位同={LOSS_TRACE_EQ}（cards step1={lo_c1[1]:.6f} oracle step1="
      f"{lo_o1[1]:.6f}）", flush=True)
BITWISE_TRAIN_VERIFIABLE = bool(ORACLE_REPRO and MODEL_INIT_SAME and D_FWD_TRAIN == 0.0)
if BITWISE_TRAIN_VERIFIABLE:
    BITWISE_MSG = ("oracle 自身逐位可复现 ⇒ 训练路径的逐位比较**可做**：cards vs oracle 30 步 "
                   f"max|Δθ|={d_co:.3e}（0 ⇒ 逐位同；非 0 ⇒ 定位）")
else:
    BITWISE_MSG = ("★训练路径的逐位等价**不可验证**：同一段 oracle 代码跑两次就 max|Δθ|"
                   f"={d_oo:.3e}（GPU 核/atomics 非确定）⇒ 本单元只报统计判据（EM Δ）与收敛"
                   "曲线形状，不声称逐位一致")
print(f"[S51-CHK3] ⇒ {BITWISE_MSG}", flush=True)
del m_o1, m_o2, m_c1, P_O1, P_O2, P_C1, sn_o1, sn_c1
torch.cuda.empty_cache()

# ============================================================================
# ⑤ 主运行：cards 12000 步 + oracle 12000 步（各取 3000/6000/9000/12000 快照）
# ============================================================================
def make_eval_model(kind: str):
    return (Cards44("A", D, FF) if kind == "oracle"
            else CardPipeline(d=D, vocab_size=V, ff=FF, nhead=NHEAD, maxlen=MAXLEN,
                              pad_id=PAD_ID, dropout=0.1, residual=True)).to(device)


@torch.no_grad()
def eval_at(model, sd):
    model.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=True)
    model.eval()
    txt = greedy_gen(model, TEST_EVAL, batch=EM_BATCH)
    hits = [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, TEST_EVAL)]
    tok_ce, samp_all, dig_samp = ce_categories(model, TEST_EVAL)
    dig = [sum(x) / len(x) for x in dig_samp]
    dm = sum(dig) / len(dig)
    tr_ce = eval_ce(model, TRAIN[:256])
    return dict(em_full=sum(hits) / len(hits),
                em_clean=sum(hits[i] for i in CLEAN_IDX) / len(CLEAN_IDX),
                dig_acc=math.exp(-dm), ce_train=sum(tr_ce) / len(tr_ce), hits=hits)


RESULT = {}
for kind in ("oracle", "cards"):
    print(f"\n[S51] ===== ⑤ 主运行 {kind}：{STEPS} 步（臂 A，seed={SEED}）=====", flush=True)
    model, wall_train, snaps, losses = train_loop(kind, STEPS, GRID)
    ck = CKPT_TMPL.format(kind=kind, steps=STEPS, seed=SEED)
    sha = "-"
    if not SMOKE:
        torch.save(model.state_dict(), ck)
        sha = _sha_file(ck)
        print(f"[S51-CKPT] {os.path.basename(ck)} sha256[:16]={sha}", flush=True)
    ev = make_eval_model(kind)
    curve, hits_fin, r28 = {}, None, None
    for st in GRID:
        t1 = time.time()
        m = eval_at(ev, snaps[st])
        m["loss"] = losses.get(st)
        curve[st] = m
        if st == GRID[-1]:
            hits_fin = m["hits"]
            ev.load_state_dict({k: v.to(device) for k, v in snaps[st].items()}, strict=True)
            ev.eval()
            r28 = r28_selfcheck(ev, TEST_EVAL, tag=f"[{kind} s{STEPS} seed{SEED}] ",
                                k=min(CHK_BATCH, len(TEST_EVAL)))
        print(f"  [S51-PT] {kind} @{st}: EM全量={m['em_full']*100:.2f}% clean={m['em_clean']*100:.2f}%"
              f"(n={len(CLEAN_IDX)}) | 数字每步={m['dig_acc']*100:.2f}% | 训练CE(train256)="
              f"{m['ce_train']:.4f} | 该步 loss={m['loss']:.4f} | 本格={time.time()-t1:.1f}s",
              flush=True)
    RESULT[kind] = dict(kind=kind, wall_train=wall_train, ckpt=ck, sha=sha, curve=curve,
                        hits=hits_fin, r28_same=r28["same"], r28_k=r28["k"], snaps=snaps)
    del ev, model, snaps
    torch.cuda.empty_cache()

# ============================================================================
# ⑥ Δ / 判据 / 曲线 / 饱和检查 / ckpt
# ============================================================================
def paired_stats(xa, xb):
    d = [a - b for a, b in zip(xa, xb)]
    n = len(d)
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    return dict(n=n, d=m, se=se, nz=sum(1 for x in d if x != 0) / n)


FINAL = GRID[-1]
ro, rc = RESULT["oracle"], RESULT["cards"]
print(f"\n[S51] ===== ⑥ 判定（写死：|ΔEM| ≤ {TOL_EQ}pp 等价 / ≤ {TOL_NEAR}pp 近似等价 / "
      f"> {TOL_NEAR}pp 不等价）=====", flush=True)
print(f"[S51-TABLE] 路径 | EM全量@{FINAL} | EMclean@{FINAL} | 数字每步 | R28 | 训练min | ckpt sha16",
      flush=True)
for kind in ("oracle", "cards"):
    r = RESULT[kind]
    f = r["curve"][FINAL]
    print(f"[S51-TABLE] {kind} | {f['em_full']*100:.2f}% | {f['em_clean']*100:.2f}% | "
          f"{f['dig_acc']*100:.2f}% | {r['r28_same']}/{r['r28_k']} | {r['wall_train']/60:.2f} | "
          f"{r['sha']}", flush=True)

BITWISE_STEPS = {}
for st in GRID:
    d_sd, nb_sd, ne_sd = sd_maxdiff(RESULT["oracle"]["snaps"][st],
                                    cards_to_oracle(RESULT["cards"]["snaps"][st]))
    BITWISE_STEPS[st] = dict(maxdiff=d_sd, n_tensors_diff=nb_sd, n_elem_diff=ne_sd)
    print(f"[S51-BITWISE] @{st} cards vs oracle 权重逐位：max|Δθ|={d_sd:.3e}"
          f"（不同张量 {nb_sd}/{len(RESULT['oracle']['snaps'][st])}，不同元素 {ne_sd}）", flush=True)
ALL_BITWISE = all(v["maxdiff"] == 0.0 for v in BITWISE_STEPS.values())
print(f"[S51-BITWISE] ⇒ 训练全程（@{REPORT_AT}）两条路径权重"
      f"{'逐位一致 ✓' if ALL_BITWISE else '★不逐位一致（差异见表）'}", flush=True)

DELTAS = {}
for tag, key in (("全量", "em_full"), ("clean", "em_clean")):
    d_full = (rc["curve"][FINAL][key] - ro["curve"][FINAL][key]) * 100
    st = paired_stats(rc["hits"], ro["hits"]) if tag == "全量" else \
        paired_stats([rc["curve"][FINAL]["hits"][i] for i in CLEAN_IDX],
                     [ro["curve"][FINAL]["hits"][i] for i in CLEAN_IDX])
    DELTAS[tag] = (d_full, st)
    print(f"[S51-DELTA] {tag}：EM(oracle)={ro['curve'][FINAL][key]*100:.2f}% → "
          f"EM(cards)={rc['curve'][FINAL][key]*100:.2f}% ⇒ Δ={d_full:+.2f}pp | 配对 "
          f"Δ={st['d']*100:+.2f}±{st['se']*100:.2f}pp 逐样本不同比={st['nz']*100:.1f}% | n={st['n']}",
          flush=True)
d_main = DELTAS["全量"][0]
verdict = ("★等价（≤0.13pp）" if abs(d_main) <= TOL_EQ else
           f"★近似等价（≤{TOL_NEAR}pp，须看曲线形状差异）" if abs(d_main) <= TOL_NEAR else
           "★不等价（>2pp，必须定位：数据RNG → 初始化顺序 → 超参 → 评估）")
print(f"[S51-JUDGE] 主判据（全量 EM Δ）={d_main:+.2f}pp ⇒ {verdict} | 逐位可验证="
      f"{BITWISE_TRAIN_VERIFIABLE} | 全程权重逐位一致={ALL_BITWISE}", flush=True)

print(f"\n[S51] ===== ★收敛曲线（EM全量/clean，@{REPORT_AT}）=====", flush=True)
for kind in ("oracle", "cards"):
    cells = []
    for st in REPORT_AT:
        c = RESULT[kind]["curve"].get(st)
        cells.append(f"@{st}=" + (f"{c['em_full']*100:.2f}/{c['em_clean']*100:.2f}%" if c else "-"))
    print(f"[S51-CURVE] {kind}: " + " ".join(cells), flush=True)
shp = []
for st in REPORT_AT:
    a = ro["curve"][st]["em_full"] * 100
    b = rc["curve"][st]["em_full"] * 100
    shp.append(f"@{st} Δ={b-a:+.2f}pp")
print(f"[S51-CURVE] 形状逐点差（cards−oracle，全量）：" + " ".join(shp), flush=True)
for tag, key in (("训练CE(train256)", "ce_train"), ("数字每步", "dig_acc")):
    line = []
    for st in REPORT_AT:
        v = (rc["curve"][st][key] - ro["curve"][st][key])
        line.append(f"@{st}={v*100:+.2f}pp" if key == "dig_acc" else f"@{st}={v:+.4f}")
    print(f"[S51-CURVE] {tag} 差（cards−oracle）：" + " ".join(line), flush=True)

print(f"\n[S51] ===== ★饱和检查（R35）：9000→12000 EM 增量 =====", flush=True)
unsat = []
for kind in ("oracle", "cards"):
    if 9000 in RESULT[kind]["curve"] and FINAL in RESULT[kind]["curve"]:
        inc = (RESULT[kind]["curve"][FINAL]["em_full"]
               - RESULT[kind]["curve"][9000]["em_full"]) * 100
        note = "★未饱和（≥1pp）" if inc >= 1.0 else "已平（<1pp）"
        if inc >= 1.0:
            unsat.append(kind)
        print(f"[S51-SAT] {kind}: EM {RESULT[kind]['curve'][9000]['em_full']*100:.2f}→"
              f"{RESULT[kind]['curve'][FINAL]['em_full']*100:.2f}% (Δ{inc:+.2f}pp) ⇒ {note}",
              flush=True)
print(f"[S51-SAT] 未饱和路径={unsat if unsat else '无'} ⇒ "
      f"{'★结论条件性（未饱和）' if unsat else '已平'}", flush=True)

print(f"\n[S51] ===== ⑦ 成本与 ckpt =====", flush=True)
for kind in ("oracle", "cards"):
    r = RESULT[kind]
    print(f"[META] {kind}: steps={STEPS} 训练墙钟={r['wall_train']/60:.2f}min ckpt="
          f"{os.path.basename(r['ckpt'])} sha256[:16]={r['sha']}", flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} torch={torch.__version__} | "
      f"d={D} ff={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} seed={SEED} | "
      f"TEST sha16={SHA19} 泄漏={LEAK19}/{len(TEST)} | 总墙钟={(time.time()-T0)/60:.1f}min", flush=True)

out = dict(device=device, gpu=torch.cuda.get_device_name(0), torch=torch.__version__,
           steps=STEPS, grid=list(GRID), seed=SEED, smoke=SMOKE,
           exec=dict(s19=head19, s44=[s44a, s44b], cards=cards_head),
           chk1=dict(sha_stages=SHA19, sha_cards=SHA_C, sha_expect=TEST_SHA_EXPECT,
                     leak_stages=LEAK19, leak_cards=LEAK_C, leak_expect=LEAK_EXPECT,
                     clean_bad=len(BAD_IDX), field_diffs=bitdiff),
           chk2=dict(construct_cpu_state_same=same_st, construct_cuda_state_same=same_gc,
                     index_130_batches_divergence=div,
                     cards=dict(cpu_untouched=pc["cpu_untouched"], cuda_moved=pc["cuda_moved"]),
                     oracle=dict(cpu_untouched=po["cpu_untouched"], cuda_moved=po["cuda_moved"]),
                     dropout_cuda_state_same_after_forward=same_consume),
           chk3=dict(init_maxdiff=d_init, init_bitwise_same=MODEL_INIT_SAME,
                     fwd_eval_maxdiff=D_FWD_EVAL, fwd_train_maxdiff=D_FWD_TRAIN,
                     probe30_oracle_vs_oracle_maxdiff=d_oo,
                     probe30_cards_vs_oracle_maxdiff=d_co,
                     probe30_loss_bitwise_same=LOSS_TRACE_EQ,
                     oracle_reproducible=ORACLE_REPRO,
                     bitwise_train_verifiable=BITWISE_TRAIN_VERIFIABLE, msg=BITWISE_MSG),
           table={k: dict(em_full=RESULT[k]["curve"][FINAL]["em_full"],
                          em_clean=RESULT[k]["curve"][FINAL]["em_clean"],
                          dig_acc=RESULT[k]["curve"][FINAL]["dig_acc"],
                          ce_train=RESULT[k]["curve"][FINAL]["ce_train"],
                          r28=[RESULT[k]["r28_same"], RESULT[k]["r28_k"]],
                          wall_train=RESULT[k]["wall_train"], sha=RESULT[k]["sha"],
                          ckpt=os.path.basename(RESULT[k]["ckpt"]))
                  for k in RESULT},
           curve={k: {st: {kk: vv for kk, vv in RESULT[k]["curve"][st].items() if kk != "hits"}
                      for st in GRID} for k in RESULT},
           delta_pp={tag: dict(em=d, paired=st) for tag, (d, st) in DELTAS.items()},
           bitwise_steps=BITWISE_STEPS, all_steps_bitwise_same=ALL_BITWISE,
           verdict=verdict, unsaturated=unsat, wall_total_min=round((time.time() - T0) / 60, 1))
with open(JSON_OUT, "w", encoding="utf-8") as f:
    json.dump(out, f, ensure_ascii=False, indent=2)
print(f"[S51-JSON] {JSON_OUT} | [DONE] exit=0 总墙钟={out['wall_total_min']}min", flush=True)
