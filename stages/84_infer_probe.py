#!/usr/bin/env python3
"""S84 · L3-r7「推理专用探针 + 顺序执行（动态加载的离线等价）」。

预注册：`L3_PLAN.md` §19（方向 VI+III）。骨架/仪器**逐件复用** 80/82（只读 importlib 加载 79，不改它）：
数据前缀 `76`（k=8 带字母）、`load_tok/load_jsonl/prep_record/pack`、固定探测批置换常量
（`PROBE_PERM_SEED = 0x52340820`，与 80/82 同一指纹 ⇒ 「同一探测批」跨配置逐位相同）。

★ 要回答的问题（§19）：**顺序执行 N 张卡时，驻留显存能否 ≈ `max(单卡)+接口`（而不是 N 张卡之和），
  而等效参数量 = N·P？**

★ 协议（写死）
  - **只跑前向**：`torch.no_grad()`，**无优化器、无反向**（本文件里没有 backward/opt）；
  - 三要素恒定并逐条打印（`[CONST]` 行）：① **同一探测批**（跨档逐位相同 + sha256 指纹）；
    ② `batch=32`；③ `maxlen=512`；**dtype 单列** —— fp32 主档，fp16 是**独立的另一档**
    （每次调用只跑一个 dtype，落盘到各自的 JSON；`--report-r7` 里**不跨 dtype 混比**）。
  - 每档报 **峰值分配 `max_memory_allocated`** / **峰值预留 `max_memory_reserved`** /
    **墙钟（per-batch、per-token）** / **驻留（步前 `memory_allocated`）**。

★ 三档（配置）
  | 档 | 内容 |
  |---|---|
  | **A**  | 单卡：input(1 层) → **1 个 encoder 层卡** → output |
  | **B**  | **N 张同构层卡顺序执行**（N=2,3,4）：**卡间只传 d 维接口**（h ∈ R^{b×L×d}）；每张卡用完即释放，下一张再载入 |
  | **C**  | 对照：把 N 张卡当成**一个整体图**（全部常驻、等价计算）|

  B 与 C 的权重来自**同一份 CPU 参考卡**（逐位相同）⇒ 输出可直接逐条比。

★ 判据（§19，跑前写死；判定在 `--report-r7`）
  - **R7-a 保真**：B（顺序）与 C（整体图）的输出**逐条相同** —— 报**实测 max|Δ|** 与 per-item argmax
    不一致数；容差按 `methodology/measurement.md` §4 的 ROCm 口径（同进程同批 logits 可逐位相同，
    跨进程实测 ~1e-6 非确定性）⇒ 本文件用 **TOL=1e-5** 作容差，并同时给"是否逐位为 0"。
  - **R7-b**：驻留 ≈ `max(单卡)+接口`，而**不是** N 卡之和 —— 给出**实测 vs 两种解析预测**的对照
    （预测用实测张量字节数逐项相加：`params(input)+params(output)+N·params(card)+h+m`）。
  - **R7-c**：**等效参数量 ÷ 驻留** 随 N 的曲线。
  - **R7-d**：延迟 per-batch / per-token 随 N 的形状（B 与 C 并列）。
  - **R7-e**：参数 / 接口宽度 / 峰值（分配+预留）/ dtype 并列。

★ 诚实边界（§19 预告，照抄不软化）
  - 本探针的「卡」是**同构同尺寸**的 encoder 层卡（我们目前只有这一种尺寸）⇒ **不同尺寸/不同功能的卡未测**；
  - **未实现真正的磁盘 paging**：顺序执行是它的**极限情形**（最坏延迟、最小驻留）；
  - C 档的"整体图"在 `no_grad` 下**不保留反向激活** ⇒ 它是"全常驻"的**推理**画像，不是训练画像。

用法（本机只有 `.venv/bin/python` 装了 torch/tokenizers）
  .venv/bin/python stages/84_infer_probe.py --dtype fp32        # GPU：主档（~几分钟）
  .venv/bin/python stages/84_infer_probe.py --dtype fp16        # GPU：独立另一档
  .venv/bin/python stages/84_infer_probe.py --report-r7         # CPU：出 [CRIT] R7-a..e
GPU 长跑必须：systemd-run --user --unit=<名> --collect --working-directory=<repo>
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 .venv/bin/python stages/84_infer_probe.py ...
只允许写本文件与 `logs/84_*`。
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def _load_path(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


m79 = _load_path("m79_cand_cards_v2", os.path.join(ROOT, "stages", "79_cand_cards_v2.py"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# ---------------- 常量（沿用 79/80 的口径） ----------------
D = m79.D
FF = m79.FF
NHEAD = m79.NHEAD
MAXLEN = m79.MAXLEN
BATCH = m79.BATCH
DROPOUT = m79.DROPOUT
LOG_DIR = m79.LOG_DIR
PREFIX = "76"
K = 8
MIB = 1024 ** 2
IO_D_TO_D = m79.IO_D_TO_D

# 固定探测批：与 80/82 **同一个置换常量** ⇒ 指纹应逐位相同（跨文件交叉核对）
PROBE_PERM_SEED = 0x52_34_08_20
N_BATCHES = 10          # = 80 的 PS（探测批条数）
REPEATS = 2             # 计时重复遍数（驻留/峰值取全程）
WARMUP_BATCHES = 1
NS = (1, 2, 3, 4)
DTYPES = ("fp32", "fp16")
# ★ 容差口径：methodology/measurement.md §4 —— 同进程同批 logits 可逐位相同，跨进程 ~1e-6。
#   本探针 B/C 在**同进程同批**下比较 ⇒ 期望 max|Δ|=0；TOL 取 1e-5（10× 跨进程实测尺度）作闸门。
TOL_ABS = 1e-5


# ---------------- 卡：1 个 encoder 层（与 79 的 SelCard/RealignCard 同款） ----------------
class LayerCard(m79.Card):
    """d→d 的同构层卡：唯一的标准卡内层（`cards.encoder_layer`，不许自己发明架构）。"""

    io = IO_D_TO_D
    card_name = "layer"

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT):
        super().__init__(d, maxlen)
        self.enc = m79.encoder_layer(d, ff, nhead, dropout)

    def apply_card(self, x, mask=None):
        return self.enc(x, src_mask=mask)


def param_bytes(mod) -> int:
    return sum(p.numel() * p.element_size() for p in mod.parameters())


def tensor_bytes(t) -> int:
    return int(t.numel() * t.element_size())


def param_bytes_dtype(mod, dtype: str) -> int:
    size = 2 if dtype == "fp16" else 4
    return sum(p.numel() * size for p in mod.parameters())


# ---------------- 数据 / 固定探测批（与 80 逐字同口径） ----------------
def load_data():
    train = m79.load_jsonl(os.path.join(LOG_DIR, f"{PREFIX}_dialog_train.jsonl"))
    test = m79.load_jsonl(os.path.join(LOG_DIR, f"{PREFIX}_dialog_test.jsonl"))
    for r in train:
        m79.prep_record(r)
    for r in test:
        m79.prep_record(r)
    return train, test


def build_probe_batches(train, n_steps):
    g = torch.Generator().manual_seed(PROBE_PERM_SEED)
    perm = torch.randperm(len(train), generator=g)
    recs = [[train[int(perm[i * BATCH + j])] for j in range(BATCH)] for i in range(n_steps)]
    packed = [m79.pack(b) for b in recs]
    return recs, packed


def probe_fp(packed) -> str:
    h = hashlib.sha256()
    for ids, s, eidx, nc in packed:
        for t in (ids, s, eidx, nc):
            h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()[:16]


def probe_len_stats(packed):
    lens = [int(ids.size(1)) for ids, _s, _e, _n in packed]
    return min(lens), max(lens), sum(lens) / len(lens), len(lens)


# ---------------- 前向（B = 顺序，C = 整体图） ----------------
def load_card(ref, dtype: str, device: str):
    """从 CPU 参考卡拷到 GPU（= 动态加载时的一次换入；权重与 C 档逐位相同）。

    ★ `eval()`：推理探针必须关掉 dropout（否则 B/C 两次前向的 dropout 掩码不同 ⇒ 保真判据失效）。
    """
    c = copy.deepcopy(ref)
    c = c.to(device).eval()
    if dtype == "fp16":
        c = c.half()
    return c


def make_mask(input_card, ids, dtype):
    m = input_card.causal_mask(ids)
    if dtype == "fp16":
        m = m.half()
    return m


@torch.no_grad()
def forward_seq(n, packed, input_card, output_card, ref_cards, dtype, device):
    """B 档：逐卡载入→前向→释放；卡间只传 h（d 维接口）。返回 (logits, h, m, resident, t_load, t_fwd, hb, mb)。"""
    ids, _s, _e, _n = packed
    m = make_mask(input_card, ids, dtype)
    h = input_card(ids, mask=m)
    hb, mb = tensor_bytes(h), tensor_bytes(m)
    resident, t_load, t_fwd = 0, 0.0, 0.0
    for i in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        card = load_card(ref_cards[i], dtype, device)
        torch.cuda.synchronize()
        t_load += time.perf_counter() - t0
        resident = max(resident, torch.cuda.memory_allocated())
        t1 = time.perf_counter()
        h = card(h, mask=m)
        torch.cuda.synchronize()
        t_fwd += time.perf_counter() - t1
        del card
    t2 = time.perf_counter()
    logits = output_card(h)
    torch.cuda.synchronize()
    t_fwd += time.perf_counter() - t2
    return logits, h, m, resident, t_load, t_fwd, hb, mb


@torch.no_grad()
def forward_graph(mods, packed, input_card, output_card, dtype):
    """C 档：N 张卡全部常驻，串成一张图；返回 (logits, h, m, resident, t_fwd)。"""
    ids, _s, _e, _n = packed
    m = make_mask(input_card, ids, dtype)
    h = input_card(ids, mask=m)
    torch.cuda.synchronize()
    resident = torch.cuda.memory_allocated()
    t0 = time.perf_counter()
    for c in mods:
        h = c(h, mask=m)
    logits = output_card(h)
    torch.cuda.synchronize()
    return logits, h, m, resident, time.perf_counter() - t0


# ---------------- 测量（显存 + 墙钟） ----------------
def measure_seq(label, n, batches, input_card, output_card, ref_cards, dtype, device):
    forward_seq(n, batches[0], input_card, output_card, ref_cards, dtype, device)   # 预热
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    resident_max, t_load, t_fwd, tokens = 0, 0.0, 0.0, 0
    hb = mb = 0
    t0 = time.perf_counter()
    for packed in batches:
        logits, h, m, resident, tl, tf, hbi, mbi = forward_seq(
            n, packed, input_card, output_card, ref_cards, dtype, device)
        resident_max = max(resident_max, resident)
        t_load += tl
        t_fwd += tf
        hb, mb = hbi, mbi
        tokens += int(packed[0].numel())
        del logits, h, m
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    row = {"cfg": label, "N": n, "mode": "seq" if n > 1 else "single",
           "resident_mib": resident_max / MIB,
           "peak_mib": torch.cuda.max_memory_allocated() / MIB,
           "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
           "wall_s": wall, "t_load_s": t_load, "t_fwd_s": t_fwd,
           "per_batch_s": wall / len(batches), "per_batch_fwd_s": t_fwd / len(batches),
           "per_token_s": wall / tokens, "tokens": tokens, "n_batches": len(batches),
           "h_bytes": hb, "m_bytes": mb, "card_fwd_per_batch": n + 2}
    return row


def measure_graph(label, n, batches, input_card, output_card, ref_cards, dtype, device):
    mods = [load_card(ref_cards[i], dtype, device) for i in range(n)]
    forward_graph(mods, batches[0], input_card, output_card, dtype)                # 预热
    del mods
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    mods = [load_card(ref_cards[i], dtype, device) for i in range(n)]
    torch.cuda.synchronize()
    t_load = time.perf_counter() - t0
    resident_max, t_fwd, tokens = 0, 0.0, 0
    hb = mb = 0
    t1 = time.perf_counter()
    for packed in batches:
        logits, h, m, resident, tf = forward_graph(mods, packed, input_card, output_card, dtype)
        resident_max = max(resident_max, resident)
        t_fwd += tf
        hb, mb = tensor_bytes(h), tensor_bytes(m)
        tokens += int(packed[0].numel())
        del logits, h, m
    torch.cuda.synchronize()
    wall = time.perf_counter() - t1
    del mods
    row = {"cfg": label, "N": n, "mode": "graph",
           "resident_mib": resident_max / MIB,
           "peak_mib": torch.cuda.max_memory_allocated() / MIB,
           "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
           "wall_s": wall, "t_load_s": t_load, "t_fwd_s": t_fwd,
           "per_batch_s": t_fwd / len(batches), "per_batch_fwd_s": t_fwd / len(batches),
           "per_token_s": t_fwd / tokens, "tokens": tokens, "n_batches": len(batches),
           "h_bytes": hb, "m_bytes": mb, "card_fwd_per_batch": n + 2}
    return row


# ---------------- 保真（R7-a） ----------------
@torch.no_grad()
def fidelity(n, batches, input_card, output_card, ref_cards, dtype, device):
    """同一批上 B（顺序）与 C（整体图）逐条比：logits max|Δ|、接口 max|Δ|、per-item argmax 不一致数。"""
    mods = [load_card(ref_cards[i], dtype, device) for i in range(n)]
    worst, worst_iface = 0.0, 0.0
    elem_mism, item_mism, n_elem, n_item = 0, 0, 0, 0
    worst_batch = None
    for bi, packed in enumerate(batches):
        lb, hb_, _mb, _r, _tl, _tf, _hbi, _mbi = forward_seq(
            n, packed, input_card, output_card, ref_cards, dtype, device)
        lg, hg_, _mg, _r2, _tf2 = forward_graph(mods, packed, input_card, output_card, dtype)
        d = float((lb.float() - lg.float()).abs().max().item())
        di = float((hb_.float() - hg_.float()).abs().max().item())
        if d > worst:
            worst, worst_batch = d, bi
        worst_iface = max(worst_iface, di)
        ab = lb.argmax(-1)
        ag = lg.argmax(-1)
        ne = int((ab != ag).sum().item())
        elem_mism += ne
        n_elem += int(ab.numel())
        row_mism = (ab != ag).any(dim=-1).sum().item()
        item_mism += int(row_mism)
        n_item += int(ab.size(0))
        del lb, lg, hb_, hg_
    del mods
    return {"N": n, "max_abs_delta": worst, "worst_batch": worst_batch,
            "iface_max_abs_delta": worst_iface,
            "elem_argmax_mismatch": elem_mism, "n_elem": n_elem,
            "item_argmax_mismatch": item_mism, "n_item": n_item,
            "tol_abs": TOL_ABS,
            "pass": bool(worst <= TOL_ABS and elem_mism == 0)}


# ---------------- GPU 主流程 ----------------
def run_probe(a) -> int:
    global N_BATCHES, REPEATS, NS
    if a.smoke:
        N_BATCHES, REPEATS, NS = 2, 1, (1, 2)
    if not torch.cuda.is_available():
        print("GPU 起不来，立即停（不用 CPU 硬跑）。", flush=True)
        print("HSA_OVERRIDE_GFX_VERSION =", os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)
        return 3
    device = "cuda"
    m79.device = device
    dtype = a.dtype
    print(f"device: {device} {torch.cuda.get_device_name(0)} | dtype={dtype}（单列，本进程只跑这一档）",
          flush=True)
    train, _test = load_data()
    probe_recs, probe_packed = build_probe_batches(train, N_BATCHES)
    batches = probe_packed * REPEATS
    lo, hi, mean, nb = probe_len_stats(probe_packed)
    fp = probe_fp(probe_packed)
    print(f"[DATA] prefix={PREFIX} k={K} train={len(train)} | 固定探测批 {nb} 批×batch{BATCH}（"
          f"各重复 {REPEATS} 遍计时）| pad 后长度 min/max/mean={lo}/{hi}/{mean:.0f} | 指纹={fp}", flush=True)

    # ---- 参考卡（CPU，一份，B/C 共用 ⇒ 权重逐位相同） ----
    torch.manual_seed(a.seed)
    input_ref = m79.InputCard(D, m79.V, FF, NHEAD, MAXLEN, m79.PAD_ID, DROPOUT)
    output_ref = m79.OutputCard(D, m79.V, MAXLEN)
    ref_cards = [LayerCard() for _ in range(max(NS))]
    n_card = sum(p.numel() for p in ref_cards[0].parameters())
    n_in = sum(p.numel() for p in input_ref.parameters())
    n_out = sum(p.numel() for p in output_ref.parameters())
    print(f"[PARAMS] input={n_in} 单卡={n_card} output={n_out} | 1 卡合计={n_in + n_card + n_out} "
          f"| N 卡等效={n_in + n_card * max(NS) + n_out}(N=4) | 参数块大小 dtype={dtype} 单卡="
          f"{param_bytes_dtype(ref_cards[0], dtype)} B | 接口宽度 d={D}（张量形状 (b,L,d)，"
          f"精度={'fp32' if dtype == 'fp32' else 'fp16'}）", flush=True)

    input_card = input_ref.to(device).eval()
    output_card = output_ref.to(device).eval()
    if dtype == "fp16":
        input_card, output_card = input_card.half(), output_card.half()

    rows, fid_rows = [], []
    cfgs = [("A", 1, "seq")]
    for n in NS[1:]:
        cfgs.append((f"B{n}", n, "seq"))
    for n in NS[1:]:
        cfgs.append((f"C{n}", n, "graph"))

    for label, n, mode in cfgs:
        torch.cuda.empty_cache()
        if mode == "seq":
            row = measure_seq(label, n, batches, input_card, output_card, ref_cards, dtype, device)
        else:
            row = measure_graph(label, n, batches, input_card, output_card, ref_cards, dtype, device)
        rows.append(row)
        print(f"[CONST] dtype={dtype} 档={label} N={n} mode={mode} | ① 同一探测批=是（指纹={fp}，"
              f"跨档逐位相同）② batch={BATCH} ③ maxlen={MAXLEN} | ★dtype 单列={dtype}"
              f"（不跨 dtype 对比）| 接口=卡间只传 h(b,L,d)，d={D}", flush=True)
        print(f"[MEM] dtype={dtype} 档={label} N={n} mode={mode} | 峰值分配={row['peak_mib']:.1f} MiB"
              f" | 峰值预留={row['peak_reserved_mib']:.1f} MiB | 驻留(卡前/步前)={row['resident_mib']:.3f} MiB"
              f" | 每批墙钟={row['per_batch_s']:.4f}s（仅前向 {row['per_batch_fwd_s']:.4f}s）"
              f" | 每 token={row['per_token_s'] * 1e6:.1f} µs | 换入累计={row['t_load_s']:.3f}s"
              f" | 前向调用/批={row['card_fwd_per_batch']}", flush=True)
        if mode == "seq" and n > 1:
            fr = fidelity(n, probe_packed, input_card, output_card, ref_cards, dtype, device)
            fid_rows.append(fr)
            print(f"[FIDELITY] dtype={dtype} N={n} B(顺序) vs C(整体图) | max|Δ logits|={fr['max_abs_delta']:.3e}"
                  f"（最差批={fr['worst_batch'] if fr['worst_batch'] is not None else '—（所有批均为 0）'}）"
                  f" | 接口 max|Δ|={fr['iface_max_abs_delta']:.3e}"
                  f" | argmax 不一致 逐位置={fr['elem_argmax_mismatch']}/{fr['n_elem']}"
                  f" 逐条(item)={fr['item_argmax_mismatch']}/{fr['n_item']}"
                  f" | 容差={TOL_ABS:.0e}（同进程同批；measurement.md §4）⇒ "
                  f"{'PASS 保真' if fr['pass'] else '★FAIL 不保真'}", flush=True)

    # ---- 预测（R7-b 的两把尺） ----
    #   ①「max(单卡)+接口」= **实测 A(N=1) 的驻留**（直接实测基线，最保守）
    #   ②「N 卡之和」    = ① + (N−1)×单卡参数块（解析）
    #   另报**解析式**（params(input)+params(output)+N·card+h+m）与其**实测常数偏移 C0**：
    #   C0 是首次编码层前向在 ROCm 上一次性申请、之后常驻的库工作区（实测 ~32 MiB，跨 N 不变 ⇒ 抵消）。
    a_row = next(r for r in rows if r["cfg"] == "A")
    base_res = a_row["resident_mib"]
    pb_mib = param_bytes_dtype(ref_cards[0], dtype) / MIB
    preds = []
    for (label, n, mode), row in zip(cfgs, rows):
        pin = param_bytes_dtype(input_ref, dtype)
        pout = param_bytes_dtype(output_ref, dtype)
        pb = param_bytes_dtype(ref_cards[0], dtype)
        pred_1_an = (pin + pout + pb + row["h_bytes"] + row["m_bytes"]) / MIB
        pred_n_an = (pin + pout + n * pb + row["h_bytes"] + row["m_bytes"]) / MIB
        c0 = base_res - pred_1_an
        e_params = n_in + n * n_card + n_out
        preds.append({"cfg": label, "N": n, "mode": mode, "resident_mib": row["resident_mib"],
                      "pred_1card_mib": base_res, "pred_ncard_mib": base_res + (n - 1) * pb_mib,
                      "pred_1card_analytic_mib": pred_1_an, "pred_ncard_analytic_mib": pred_n_an,
                      "c0_mib": c0, "card_mib": pb_mib, "equiv_params": e_params,
                      "ratio_params_per_mib": e_params / row["resident_mib"] if row["resident_mib"] else None,
                      "ratio_params_per_mib_peak": e_params / row["peak_mib"] if row["peak_mib"] else None})
        print(f"[PRED] dtype={dtype} 档={label} | 实测驻留={row['resident_mib']:.3f} MiB | "
              f"预测①max(单卡)+接口={base_res:.3f} MiB（实测基线 A；解析 {pred_1_an:.3f}，"
              f"实测常数 C0={c0:+.3f}）⇒ 差 {row['resident_mib'] - base_res:+.3f} | "
              f"预测②N 卡之和={base_res + (n - 1) * pb_mib:.3f} MiB（=①+(N−1)×{pb_mib:.3f} MiB）"
              f"⇒ 差 {row['resident_mib'] - (base_res + (n - 1) * pb_mib):+.3f}", flush=True)

    out = {"dtype": dtype, "seed": a.seed, "probe_fp": fp, "batch": BATCH, "maxlen": MAXLEN,
           "n_batches": N_BATCHES, "repeats": REPEATS, "interface_d": D, "tol_abs": TOL_ABS,
           "params": {"input": n_in, "card": n_card, "output": n_out, "card_bytes": param_bytes_dtype(ref_cards[0], dtype)},
           "len_stats": {"min": lo, "max": hi, "mean": mean},
           "rows": rows, "predictions": preds, "fidelity": fid_rows,
           "device": torch.cuda.get_device_name(0)}
    p = os.path.join(LOG_DIR, f"84_infer_{dtype}{'_smoke' if a.smoke else ''}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump(out, f)
    print(f"[META] dtype={dtype} 指纹={fp} 档数={len(rows)} 落盘={p}", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


# ---------------- [CRIT] R7-a..e（CPU，读两个 dtype 的 JSON） ----------------
def load_json(p):
    if os.path.exists(p):
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return None


def report_r7(a) -> int:
    data = {}
    for dt in DTYPES:
        d = load_json(os.path.join(LOG_DIR, f"84_infer_{dt}.json"))
        if d is not None:
            data[dt] = d
    print(f"[PAIR] 读到 dtype：{sorted(data)}（每档只在自己的 dtype 内比较；★不跨 dtype 混比）", flush=True)
    if not data:
        print("[CRIT] R7：没有读数 ⇒ 未判（如实报）", flush=True)
        return 1
    for dt, d in data.items():
        print(f"[READ] dtype={dt} 指纹={d['probe_fp']} batch={d['batch']} maxlen={d['maxlen']} "
              f"探测批={d['n_batches']} 接口 d={d['interface_d']} | 参数 input={d['params']['input']} "
              f"单卡={d['params']['card']} output={d['params']['output']}", flush=True)
    fps = {dt: d["probe_fp"] for dt, d in data.items()}

    # ---- R7-a 保真 ----
    print("[CRIT] R7-a 保真：B（顺序执行）与 C（整体图）输出逐条相同", flush=True)
    ok_a = True
    for dt, d in data.items():
        if not d["fidelity"]:
            print(f"[CRIT] R7-a dtype={dt}：无保真行（只跑了 N=1）⇒ 未判", flush=True)
            continue
        for fr in d["fidelity"]:
            ok_a = ok_a and fr["pass"]
            print(f"[CRIT] R7-a dtype={dt} N={fr['N']} | 实测 max|Δ logits|={fr['max_abs_delta']:.3e}"
                  f" | 接口 max|Δ|={fr['iface_max_abs_delta']:.3e}"
                  f" | argmax 不一致 逐位置={fr['elem_argmax_mismatch']}/{fr['n_elem']} "
                  f"逐条={fr['item_argmax_mismatch']}/{fr['n_item']}"
                  f" | 容差={fr['tol_abs']:.0e}（ROCm 口径：同进程同批可逐位，跨进程 ~1e-6）⇒ "
                  f"{'PASS' if fr['pass'] else '★FAIL'}", flush=True)
    print(f"[CRIT] R7-a ⇒ {'PASS（所有 dtype×N 的 B≡C 逐条相同）' if ok_a else '★未过（如实报）'}"
          f"；⇒ 「动态加载不改语义」在**同构同尺寸层卡**上成立", flush=True)

    # ---- R7-b 驻留 vs 两种预测 ----
    print("[CRIT] R7-b 驻留：实测 vs 「max(单卡)+接口」（实测基线 A） vs 「N 卡之和」（①+(N−1)×单卡）", flush=True)
    for dt, d in data.items():
        print(f"[CRIT] R7-b dtype={dt}（预测①②都锚在**实测** A(N=1) 上；解析式还需加一个跨 N 不变的"
              f"实测常数 C0 —— ROCm 上首次编码层前向一次性申请的库工作区）", flush=True)
        for pr in d["predictions"]:
            m, p1, pn = pr["resident_mib"], pr["pred_1card_mib"], pr["pred_ncard_mib"]
            near1 = abs(m - p1) <= 0.5
            near_n = abs(m - pn) <= 0.5
            tag = ("≈max(单卡)+接口" if near1 and not near_n else
                   ("≈N 卡之和" if near_n and not near1 else ("两者都近（N 小、卡太小）" if near1 and near_n
                                                              else "★两者都不近（如实报）")))
            print(f"[CRIT] R7-b dtype={dt} 档={pr['cfg']} mode={pr['mode']} | 实测驻留={m:.3f} MiB | "
                  f"预测①={p1:.3f}（差 {m - p1:+.3f}）| 预测②={pn:.3f}（差 {m - pn:+.3f}）"
                  f" | 解析式①②={pr['pred_1card_analytic_mib']:.3f}/{pr['pred_ncard_analytic_mib']:.3f}"
                  f"（实测常数 C0={pr['c0_mib']:+.3f} MiB，跨 N 相同 ⇒ 在差值里抵消）⇒ {tag}", flush=True)
        seqs = [r for r in d["rows"] if r["mode"] == "seq" and r["N"] > 1]
        a_row = next((r for r in d["rows"] if r["cfg"] == "A"), None)
        if seqs and a_row:
            spread = max(r["resident_mib"] for r in seqs) - min(r["resident_mib"] for r in seqs)
            off = max(r["resident_mib"] for r in seqs) - a_row["resident_mib"]
            print(f"[CRIT] R7-b dtype={dt} ⇒ 顺序执行档 B2/B3/B4 驻留极差={spread:.3f} MiB"
                  f"（单卡参数块={d['params']['card_bytes'] / MIB:.3f} MiB）⇒ "
                  f"{'驻留不随 N 增长（= 单卡+接口）' if spread <= 0.05 else '★驻留随 N 增长（如实报）'}"
                  f"；A→B 的一次性偏置={off:+.3f} MiB（fp16 下可见的分配器块粒度，不随 N 复利）"
                  f"；★诚实边界：本探针只有**同构同尺寸**层卡，不同尺寸/功能的卡未测；未实现真正的磁盘 paging",
                  flush=True)

    # ---- R7-c 等效参数量 ÷ 驻留 ----
    print("[CRIT] R7-c ★等效参数量 ÷ 驻留 随 N 的曲线（用户设想的核心量化）", flush=True)
    for dt, d in data.items():
        print(f"[CRIT] R7-c dtype={dt}", flush=True)
        for pr in d["predictions"]:
            rb = pr["ratio_params_per_mib"]
            rp = pr["ratio_params_per_mib_peak"]
            print(f"[CRIT] R7-c dtype={dt} N={pr['N']} 档={pr['cfg']} mode={pr['mode']} | 等效参数={pr['equiv_params']}"
                  f" | 驻留={pr['resident_mib']:.3f} MiB ⇒ 等效参数/驻留={rb:.0f} 参数/MiB"
                  f" | 用峰值分配算={rp:.0f} 参数/MiB", flush=True)

    # ---- R7-d 延迟 ----
    print("[CRIT] R7-d 延迟 per-batch / per-token 随 N 的形状", flush=True)
    for dt, d in data.items():
        base = next((r for r in d["rows"] if r["cfg"] == "A"), None)
        for r in d["rows"]:
            xb = r["per_batch_s"] / base["per_batch_s"] if base and base["per_batch_s"] else float("nan")
            xt = r["per_token_s"] / base["per_token_s"] if base and base["per_token_s"] else float("nan")
            print(f"[CRIT] R7-d dtype={dt} 档={r['cfg']} N={r['N']} mode={r['mode']} | "
                  f"per-batch={r['per_batch_s'] * 1e3:.2f} ms（仅前向 {r['per_batch_fwd_s'] * 1e3:.2f} ms）"
                  f" ×{xb:.2f} vs A | per-token={r['per_token_s'] * 1e6:.1f} µs ×{xt:.2f} | "
                  f"换入累计={r['t_load_s']:.3f}s | 前向调用/批={r['card_fwd_per_batch']}", flush=True)

    # ---- R7-e 并列 ----
    print("[CRIT] R7-e 参数 / 接口宽度 / 双量峰值 / dtype 并列", flush=True)
    for dt, d in data.items():
        for r in d["rows"]:
            print(f"[CRIT] R7-e dtype={dt} 档={r['cfg']} N={r['N']} mode={r['mode']} | 接口 d={d['interface_d']} "
                  f"| 峰值分配={r['peak_mib']:.1f} MiB | 峰值预留={r['peak_reserved_mib']:.1f} MiB "
                  f"| 驻留={r['resident_mib']:.3f} MiB | 等效参数={d['params']['input'] + r['N'] * d['params']['card'] + d['params']['output']}"
                  f" | 探测批指纹={d['probe_fp']}", flush=True)
    if len(fps) == 2 and len(set(fps.values())) == 1:
        print(f"[CRIT] R7-e 两个 dtype 的探测批指纹相同={fps} ⇒ 数据侧恒定；★但 dtype 是**两列**，"
              f"跨 dtype 的任何差值都没有单变量含义（§19 写死）", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="S84 L3-r7 推理专用探针 + 顺序执行（R7-a..e）")
    ap.add_argument("--dtype", choices=DTYPES, default="fp32")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--smoke", action="store_true", help="2 批×1 遍、N∈{1,2}（探路，不作判据）")
    ap.add_argument("--report-r7", action="store_true", help="CPU：读两个 dtype 的 JSON 出 [CRIT] R7-a..e")
    a = ap.parse_args(argv)
    if a.report_r7:
        return report_r7(a)
    m79.load_tok()
    return run_probe(a)


if __name__ == "__main__":
    with torch.no_grad():
        sys.exit(main())
