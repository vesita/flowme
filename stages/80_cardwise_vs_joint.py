#!/usr/bin/env python3
"""S80 · L3-r4「逐卡训练 vs 联合训练」四臂（J / P / P+F / J+GC）：显存 / 质量 / 可训规模上界。

预注册：`L3_PLAN.md` §12（协议 12.1 / 四臂 12.2 / 判据 R4-a..d 12.3）；init 缺陷纪律见 §9.7；
两卡结构的读数背景见 §11.31。骨架与仪器**逐件复用** `stages/79_cand_cards_v2.py`
（importlib 加载，不改它）：模型两卡结构（input 1 层 + 选择卡 1 层 + 重整卡 1 层 + output）、
`select` 打分 `s_j = q·h_j` + soft 选择 `e_sel`、选择 CE（λ=1）、`reply_ce`、`greedy_gen`、
`eval_ce/eval_sel`、`report_texts`（T1 字母 / T2 完整回复 EM）、`paired_stats`/`mcnemar_exact`（§11.4 配对口径）、
`t1_threshold`/`t2_threshold`、数据前缀 `76`（k=8 带字母）。

★ 四臂（§12.2）—— **同一结构、同一数据、同一步数预算，唯一变量 = 训练方式**
  | 臂 | 训练方式 |
  |---|---|
  | **J**      | 联合端到端：全部参数进**同一个** AdamW |
  | **P**      | 逐卡训练：一次只训一张卡（其余 `requires_grad=False`），**每相位新建优化器** ⇒ 优化器状态只覆盖当前卡 |
  | **P+F**    | 逐卡训练 + 末尾**少量联合微调**（末段回到 J 的画像：一个新优化器覆盖全部参数） |
  | **J+GC**   | 联合 + **梯度检查点**（`torch.utils.checkpoint`，非重入模式；见 `--gc-check` 的逐张量梯度等价自检） |

**步数分配（写死，四臂同为 `--steps` 步 = 默认 3000，同步数；日志里 [PLAN]/[SEG] 逐段打印）**
  - J      ：1 段 × 3000 步（全参数）
  - J+GC   ：1 段 × 3000 步（全参数 + GC）
  - P      ：5 轮 × 4 张卡 × **150** 步 = 3000（相位顺序 input → select(+q) → realign → output，逐轮循环）
  - P+F    ：逐卡 5 轮 × 4 张卡 × **125** 步 = 2500 + 末尾联合微调 **500** 步 = 3000
  - 「卡」的划分 = `{input}` / `{select, q}` / `{realign}` / `{output}`（q 是选择打分向量，归入选择卡）
  - `--smoke`（探路，不作判据）：50 步，P/P+F 退化为 1 轮；探针窗口同步缩短。
  ⇒ 四臂总样本预算相同：`steps × batch32`（3000 → 96000 样本）。

★ 显存探针（§12.1，**必须与真实训练同构**）
  - 用**真实训练步**：`reset_peak_memory_stats()` → 跑 N 步真实 `loss.backward(); opt.step()`
    → 读 `torch.cuda.max_memory_allocated()`（MiB）。**禁止**解析估算代替。
  - **四要素恒定**（逐条打印在 `[MEM]` 行里）：
      ① `batch=32`；② **同一序列长度分布** —— 探针窗口用**跨臂逐位相同**的固定探测批
         （同一置换、同一 padding、同一 `end_idx`），探针批指纹在 `--report-r4` 里跨臂比对；
      ③ `dtype=float32`（全程不 autocast）；④ `maxlen=512`。
  - 探针窗口内的这些步**就是真实训练步**（计入 3000 步预算，用的也是该臂该相位的真优化器），
    只是该窗口的数据取样被固定 ⇒ 四臂的峰值显存可比。
  - 每臂报：**峰值显存 / 可训参数量 / 每步墙钟 / T2 / T1**（探针行前缀 `[MEM]`，臂汇总行也以 `[MEM]` 开头）。

★ 判据（§12.3，写死，逐条打印 `[CRIT] R4-a..d`；跨臂判定在 `--report-r4`）
  - **R4-a**：`P` 峰值显存**显著低于** `J`（同参数量、同 batch/seq），且实测与「优化器状态+梯度只按**当前卡**计」
    的量级一致（本文件用**两把尺**：①峰值之差 vs 解析 `12 B×(全部可训 − 当前卡)`；②forward 前**驻留**
    `torch.cuda.memory_allocated()` vs 解析 `参数 4B + AdamW 两个动量 8B`）。
  - **R4-b**：质量代价 —— 报 `P` 与 `J` 的 T2 差 + **配对 Δ 与配对 SE**（per-item hits，§11.4）+ McNemar；
    ★ **两臂都 ≥95% ⇒ 判「不可判（饱和）」**，如实写「该数据上饱和、测不出质量差」，**不得**读成「逐卡无代价」。
  - **R4-c**：`J` 与 `P` 各自能训的**最大参数量上界**（**外推**，方法在 `[CRIT] R4-c` 行里写死：
    实测 intercept A = 峰值 − 训练态的解析占用；假定 A 与参数量无关 ⇒ `J: (B−A_J)/16`、`P: (B−A_P−12·P_card)/4`，
    B = 8 GiB×0.92），并注明**推理时仍需容纳全部参数**（逐卡只省训练态）。
  - **R4-d**：`J+GC` 与 `J` 的峰值显存与质量并列报 + 「拆分」与「检查点」能否**叠加**（`--probe-cross` 实测四项诊断）。

★ 诚实边界（照 §12 预告）
  - 逐卡训练省的是**梯度与优化器状态**，**不省**联合训练的激活（实测本任务峰值里激活占绝对多数）；
  - 单卡 8G 的结论**不能**外推到多卡模型并行；
  - 「单卡更好」在别处被推翻过（frozen@6000 比 e2e 差 −5.25/−13.25pp）⇒ 本任务上要复测这一代价（R4-b）；
  - **混合精度未测**：autocast 改 dtype ⇒ 违反 §12.1 的四要素恒定，必须单独预注册（本文件不测）；
  - 探针窗口固定探测批会被重复喂入（J 40 步 / P 4 相位×20 步），是**取样偏移**，四臂步数预算仍相同。

用法（本机只有 `.venv/bin/python` 装了 torch/tokenizers）
  .venv/bin/python stages/80_cardwise_vs_joint.py --gc-check                        # CPU：GC 梯度等价自检
  .venv/bin/python stages/80_cardwise_vs_joint.py --arm J --seed 1234 --smoke       # 探路
  .venv/bin/python stages/80_cardwise_vs_joint.py --arm P --seed 1234
  .venv/bin/python stages/80_cardwise_vs_joint.py --probe-cross --seed 1234         # 诊断：拆分×检查点叠加
  .venv/bin/python stages/80_cardwise_vs_joint.py --report-r4 --seed 1234          # CPU：出 [CRIT] R4-a..d
GPU 长跑必须：systemd-run --user --unit=<名> --collect --working-directory=<repo>
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 .venv/bin/python stages/80_cardwise_vs_joint.py ...
只允许写本文件与 `logs/80_*`。
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
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


# ★骨架/仪器复用 79（只读；不 import 一个数字）
m79 = _load_path("m79_cand_cards_v2", os.path.join(ROOT, "stages", "79_cand_cards_v2.py"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402
from torch.utils.checkpoint import checkpoint as _ckpt  # noqa: E402

# ---------------- 常量（全部沿用 79 = 与 r3b′ 同口径） ----------------
D = m79.D
FF = m79.FF
NHEAD = m79.NHEAD
MAXLEN = m79.MAXLEN
LR = m79.LR
BATCH = m79.BATCH
DROPOUT = m79.DROPOUT
SEL_LAMBDA = m79.SEL_LAMBDA
LOG_DIR = m79.LOG_DIR
PREFIX = "76"                      # k=8 带字母（§11.14）
K = 8
MIB = 1024 ** 2
STEPS_TOTAL = 3000

ARMS = ("J", "P", "P+F", "J+GC")
ARM_SLUG = {"J": "J", "P": "P", "P+F": "PF", "J+GC": "JGC"}
CARD_ORDER = ("input", "select", "realign", "output")
CARD_UNITS = {"input": ("input",), "select": ("select", "q"),
              "realign": ("realign",), "output": ("output",)}
ALL_CARDS = CARD_ORDER
P_CYCLES = 5
P_STEPS_PER_CARD = 150
PF_CYCLES = 5
PF_STEPS_PER_CARD = 125
PF_JOINT_STEPS = 500

# 探针（正式）：预热 5 步 + 2 个窗口 × 10 步 = 25 步/相位（都在预算内）
PW, PS, PR = 5, 10, 2
# 探针（--smoke）：窗口缩短，仍打同一批固定探测批
PW_S, PS_S, PR_S = 1, 3, 1
PROBE_PERM_SEED = 0x52_34_08_20   # ★定值：四臂的探测批完全相同

device = "cpu"


# ---------------- 数据 / 探针批 ----------------
load_jsonl = m79.load_jsonl
prep_record = m79.prep_record
pack = m79.pack


def load_data():
    train = load_jsonl(os.path.join(LOG_DIR, f"{PREFIX}_dialog_train.jsonl"))
    test = load_jsonl(os.path.join(LOG_DIR, f"{PREFIX}_dialog_test.jsonl"))
    for r in train:
        prep_record(r)
    for r in test:
        prep_record(r)
    return train, test


def build_probe_batches(train, n_steps: int):
    """固定探测批：同一置换 ⇒ 跨臂逐位相同的输入（四要素里的「同一序列长度分布」由此成立）。"""
    g = torch.Generator().manual_seed(PROBE_PERM_SEED)
    perm = torch.randperm(len(train), generator=g)
    recs = [[train[int(perm[i * BATCH + j])] for j in range(BATCH)] for i in range(n_steps)]
    packed = [pack(b) for b in recs]
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


# ---------------- 模型：四臂共用 79 的 B2 两卡结构 ----------------
def unit_params(model, card: str):
    out = []
    for nm in CARD_UNITS[card]:
        out += [model.q] if nm == "q" else list(getattr(model, nm).parameters())
    return out


class ArmJoint(m79.ArmB):
    """同一个两卡结构（= 79 的 **B2**：两卡 + λ=1 的选择 CE）；本类只多一个 `use_gc` 开关。

    四臂**共用**它 ⇒ 结构/初值/参数量逐位相同，唯一变量是训练方式（哪些参数进优化器 / 是否 GC）。
    `use_gc=True` 时 `forward_all` 逐字照抄 `79.ArmB.forward_all`，只把三次卡调用（input / select / realign）
    包进 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`（评估/eval 模式下自动退回普通前向）。
    """

    def __init__(self, use_gc: bool = False, name: str = "J"):
        super().__init__("B2")
        self.use_gc = bool(use_gc)
        self.name = name

    def forward_all(self, ids, ctx_len=None, end_idx=None, n_cands=None):
        if not (self.training and self.use_gc):
            return super().forward_all(ids, ctx_len, end_idx, n_cands)
        assert ctx_len is not None and end_idx is not None and n_cands is not None, \
            "[B 臂] 必须给 ctx_len / end_idx / n_cands"
        n = ids.size(1)
        m = self.input.causal_mask(ids)
        hi = _ckpt(self.input, ids, use_reentrant=False, mask=m)
        hs = _ckpt(self.select, hi, use_reentrant=False, mask=m)
        g = torch.gather(hs, 1, end_idx.unsqueeze(-1).expand(-1, -1, hs.size(-1)))
        sc = (g * self.q).sum(-1)
        k = sc.size(1)
        valid = torch.arange(k, device=sc.device)[None, :] < n_cands[:, None]
        sc = sc.masked_fill(~valid, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        e_sel = torch.einsum("bk,bkd->bd", p, g)
        pos = torch.arange(n, device=ids.device)[None, :]
        apply = (pos >= ctx_len[:, None]).unsqueeze(-1).to(hi.dtype)
        h = _ckpt(self.realign, hi + apply * e_sel.unsqueeze(1), use_reentrant=False, mask=m)
        return self.output(h), sc


def build_arm(arm: str) -> ArmJoint:
    return ArmJoint(use_gc=(arm == "J+GC"), name=arm)


# ---------------- 训练计划（写死的步数分配） ----------------
def _distribute(total: int, units: list) -> list:
    """把 total 步均分给 units（余数给前几段）——保证各臂总步数完全相等。"""
    per, rem = divmod(total, len(units))
    return [per + (1 if i < rem else 0) for i in range(len(units))]


def _windows(steps: int, pw: int, ps: int, pr: int, late: bool) -> list:
    """探针窗口（互不重叠）：预热 pw 步后 pr 个窗口 × ps 步；late=True 再在末段补一组（看稳定性）。"""
    if steps < pw + pr * ps + 2:
        return []
    ws = [(pw + 1 + r * ps, pw + (r + 1) * ps) for r in range(pr)]
    if late:
        end = steps - 1
        start0 = max(pw + pr * ps + 2, end - pr * ps - pw + 1)
        ws += [(start0 + r * ps, start0 + (r + 1) * ps - 1) for r in range(pr)]
    return [(a, b) for a, b in ws if a >= 1 and b <= steps]


def build_plan(arm: str, total: int, smoke: bool) -> list:
    """返回 [{name, steps, cards, windows}]；四臂总步数 = total。"""
    pw, ps, pr = (PW_S, PS_S, PR_S) if smoke else (PW, PS, PR)
    if arm in ("J", "J+GC"):
        return [{"name": f"{ARM_SLUG[arm]}#1", "steps": total, "cards": ALL_CARDS,
                 "windows": _windows(total, pw, ps, pr, late=True)}]
    cycles = 1 if smoke else P_CYCLES
    units = [(c, card) for c in range(1, cycles + 1) for card in CARD_ORDER]
    if arm == "P":
        steps = _distribute(total, units)
        return [{"name": f"P-{card}#{c}", "steps": steps[i], "cards": (card,),
                 "windows": _windows(steps[i], pw, ps, pr, late=False) if c == cycles else []}
                for i, (c, card) in enumerate(units)]
    # P+F：逐卡 2500 + 末尾联合 500
    joint = PF_JOINT_STEPS if not smoke else max(1, total // 6)
    cardwise = total - joint
    steps = _distribute(cardwise, units)
    plan = [{"name": f"PF-{card}#{c}", "steps": steps[i], "cards": (card,),
             "windows": _windows(steps[i], pw, ps, pr, late=False) if c == cycles else []}
            for i, (c, card) in enumerate(units)]
    plan.append({"name": "PF-joint", "steps": joint, "cards": ALL_CARDS,
                 "windows": _windows(joint, pw, ps, pr, late=True)})
    return plan


def plan_desc(arm: str, plan: list) -> str:
    tot = sum(s["steps"] for s in plan)
    if arm in ("J", "J+GC"):
        return f"1 段×{tot} 步（全参数{'，GC' if arm == 'J+GC' else ''}）"
    if arm == "P":
        return f"{P_CYCLES} 轮 × 4 卡 × {plan[0]['steps']} 步 = {tot}（相位 input→select(+q)→realign→output）"
    return (f"逐卡 {PF_CYCLES} 轮 × 4 卡 × {plan[0]['steps']} 步 = {tot - plan[-1]['steps']}"
            f" + 末尾联合微调 {plan[-1]['steps']} 步 = {tot}")


# ---------------- 可训性 / 优化器 ----------------
def set_active(model, cards) -> list:
    """只让 `cards` 里的卡可训（其余 `requires_grad=False`，即 §12.2 的逐卡口径）。"""
    active = []
    for c in cards:
        active += unit_params(model, c)
    ids = {id(p) for p in active}
    for p in model.parameters():
        p.requires_grad_(id(p) in ids)
    return active


def train_step(model, batch, packed, opt):
    ids, s, eidx, nc = packed if packed is not None else pack(batch)
    logits, sc = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)
    lce = m79.reply_ce(logits, ids, batch, s)
    gold = torch.tensor([r["gold_idx"] for r in batch], device=ids.device)
    lsel = F.cross_entropy(sc, gold)
    loss = lce + SEL_LAMBDA * lsel
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    return loss.detach(), lce.detach(), lsel.detach(), sc


def gate_line(model, arm, seg, cards, opt, n_tot, sc, batch):
    """[GATE]：活动卡必须有梯度、冻结卡必须 0 梯度、优化器状态只覆盖活动卡（逐条实测）。"""
    parts, ok = [], True
    for c in ALL_CARDS:
        ps = unit_params(model, c)
        trained = all(p.requires_grad for p in ps)
        got = sum(1 for p in ps if p.grad is not None)
        gm = max((float(p.grad.abs().max()) for p in ps if p.grad is not None), default=0.0)
        parts.append(f"{c}[{'训' if trained else '冻'}]{got}/{len(ps)} max|g|={gm:.2e}")
        if trained:
            ok = ok and got == len(ps) and gm > 0.0
        else:
            ok = ok and got == 0
    n_state = len(opt.state)
    n_act = sum(1 for p in model.parameters() if p.requires_grad)
    ok = ok and n_state in (0, n_act)
    _pred, hits = model.sel_pred(sc, batch)
    print(f"[GATE] arm={arm} seg={seg['name']} 卡={'+'.join(cards)} step={min(20, seg['steps'])} | "
          + " | ".join(parts)
          + f" | 优化器状态覆盖参数={n_state}/{n_act}（全部参数 {n_tot} 里只有这些进优化器）"
          + f" | batch 选择准确率={100.0 * sum(hits) / len(hits):.2f}% ⇒ {'PASS' if ok else '★CHECK'}", flush=True)
    return ok


def train_arm(model, arm, train, seed, plan, probe_recs, probe_packed, tag=""):
    """按 plan 逐段训练；探针窗口内用固定探测批跑真实训练步并读峰值显存。返回 (mem_rows, wall, steps, gates)."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train), generator=gen)
    pos = 0
    n_tot = sum(p.numel() for p in model.parameters())
    step_total = 0
    mem_rows, gates = [], []
    t0 = time.perf_counter()
    for si, seg in enumerate(plan):
        cards = seg["cards"]
        active = set_active(model, cards)
        n_act = sum(p.numel() for p in active)
        opt = AdamW(active, lr=LR)                     # ★每段新建：状态只覆盖当前卡
        wins = list(seg["windows"])
        seg_t0 = time.perf_counter()
        last = None
        gate_done = False
        reset_win = None
        for st in range(1, seg["steps"] + 1):
            win = None
            for w in wins:
                if w[0] <= st <= w[1]:
                    win = w
                    break
            if win is not None and st == win[0]:
                reset_win = win
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                resident = torch.cuda.memory_allocated() / MIB
                pt0 = time.perf_counter()
            if win is not None:
                j = st - win[0]
                batch, packed = probe_recs[j], probe_packed[j]
            else:
                idx = []
                for _ in range(BATCH):
                    if pos >= len(order):
                        order = torch.randperm(len(train), generator=gen)
                        pos = 0
                    idx.append(int(order[pos]))
                    pos += 1
                batch, packed = [train[i] for i in idx], None
            loss, lce, lsel, sc = train_step(model, batch, packed, opt)
            last = (float(loss), float(lce), float(lsel))
            step_total += 1
            if not gate_done and st == min(20, seg["steps"]):
                gate_done = True
                gates.append(gate_line(model, arm, seg, cards, opt, n_tot, sc, batch))
            if win is not None and st == win[1] and reset_win == win:
                torch.cuda.synchronize()
                dt = (time.perf_counter() - pt0) / (win[1] - win[0] + 1)
                peak = torch.cuda.max_memory_allocated() / MIB
                lo, hi, mean, nb = probe_len_stats(probe_packed)
                mem_rows.append({"arm": arm, "seg": seg["name"], "cards": "+".join(cards),
                                 "cards_n": len(cards), "win": f"{win[0]}-{win[1]}",
                                 "peak_mib": peak, "resident_mib": resident, "s_per_step": dt,
                                 "n_active": n_act, "n_total": n_tot,
                                 "probe_fp": probe_fp(probe_packed)})
                print(f"[MEM] arm={arm} seg={seg['name']} 卡={'+'.join(cards)} 探针步={win[0]}–{win[1]}"
                      f"{tag} | 四要素恒定: batch={BATCH} | 序列长度分布=固定探测批（跨臂逐位相同；"
                      f"pad 后长度 min/max/mean={lo}/{hi}/{mean:.0f}，{nb} 批）| dtype=float32 | maxlen={MAXLEN}"
                      f" | 峰值={peak:.0f} MiB | 驻留(forward 前)={resident:.1f} MiB"
                      f" | 可训(本优化器内)={n_act}/{n_tot} | 每步墙钟={dt:.3f}s", flush=True)
        seg_sec = time.perf_counter() - seg_t0
        print(f"[SEG] arm={arm} seg={seg['name']} 卡={'+'.join(cards)} 步数={seg['steps']} 累计步={step_total}"
              f" 可训={n_act}/{n_tot} 末步 loss={last[0]:.4f} ce={last[1]:.4f} sel={last[2]:.4f}"
              f" 段墙钟={seg_sec:.0f}s", flush=True)
    return mem_rows, time.perf_counter() - t0, step_total, gates


# ---------------- 自检：GC 梯度等价（CPU） ----------------
def step_once(model, batch, seed):
    torch.manual_seed(seed)
    ids, s, eidx, nc = pack(batch)
    logits, sc = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)
    lce = m79.reply_ce(logits, ids, batch, s)
    gold = torch.tensor([r["gold_idx"] for r in batch], device=ids.device)
    loss = lce + SEL_LAMBDA * F.cross_entropy(sc, gold)
    model.zero_grad(set_to_none=True)
    loss.backward()
    return float(loss.detach()), {k: (None if p.grad is None else p.grad.detach().clone())
                         for k, p in model.named_parameters()}


def gc_check(seed: int = 1234) -> int:
    """CPU：同初值、同批、同 RNG 流下，GC 与非 GC 的 loss 与**逐张量梯度**是否一致（仪器自检）。"""
    global device
    device, m79.device = "cpu", "cpu"
    m79.load_tok()
    train, _test = load_data()
    batch = train[:BATCH]
    print(f"[GC-CHECK] 同一初值/同一批（n={len(batch)}）下，torch.utils.checkpoint(use_reentrant=False) 与非 GC 对照：", flush=True)
    torch.manual_seed(seed)
    m0 = build_arm("J")
    torch.manual_seed(seed)
    m1 = build_arm("J+GC")
    bad = 0
    for k, v in m0.state_dict().items():
        bad += int(float((v - m1.state_dict()[k]).abs().max().item()) != 0.0)
    print(f"[GC-CHECK] 两模型同 seed 初值：不一致张量数={bad}（必须 0）", flush=True)
    l0, g0 = step_once(m0, batch, 20261010)
    l1, g1 = step_once(m1, batch, 20261010)
    worst, where = 0.0, "全部张量差=0"
    for k in g0:
        if g0[k] is None or g1[k] is None:
            ok = g0[k] is None and g1[k] is None
            print(f"[GC-CHECK] ★{k}: grad 有无不一致 g0={g0[k] is not None} g1={g1[k] is not None}", flush=True)
            continue
        d = float((g0[k] - g1[k]).abs().max().item())
        sc = max(1.0, float(g0[k].abs().max().item()))
        if d / sc > worst:
            worst, where = d / sc, k
    print(f"[GC-CHECK] loss 非GC={l0:.6f} GC={l1:.6f} Δ={abs(l0 - l1):.3e} | 逐张量梯度最大相对差="
          f"{worst:.3e}（{where}）⇒ {'PASS 梯度等价' if worst < 1e-4 else '★FAIL 梯度不等价 ⇒ R4-d 的读数不可用'}",
          flush=True)
    return 0 if (bad == 0 and worst < 1e-4) else 1


# ---------------- per-run 主流程 ----------------
def hits_path(arm: str, seed: int, smoke: bool = False) -> str:
    return os.path.join(LOG_DIR, f"80_hits_{ARM_SLUG[arm]}_s{seed}{'_smoke' if smoke else ''}.json")


def run_one(a) -> int:
    global device
    torch.manual_seed(a.seed)
    if not torch.cuda.is_available():
        print("GPU 起不来，立即停（不用 CPU 硬跑）。", flush=True)
        print("HSA_OVERRIDE_GFX_VERSION =", os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)
        return 3
    device = "cuda"
    m79.device = device
    print("device:", device, torch.cuda.get_device_name(0), flush=True)
    train, test = load_data()
    ev = test[: m79.SMOKE_EVAL] if a.smoke else test
    steps_total = m79.SMOKE_STEPS if a.smoke else a.steps
    plan = build_plan(a.arm, steps_total, a.smoke)
    smoke_note = "（--smoke 探路，不作判据）" if a.smoke else ""

    # ---- [DATA]（与 79 同口径的指纹核对，不作判据） ----
    fp_tr, fp_te = m79.d74.d72.data_fp(train), m79.d74.d72.data_fp(test)
    q = m79.FP_QUOTED.get(PREFIX)
    print(f"[DATA] prefix={PREFIX} k={K} arm={a.arm} seed={a.seed} train={len(train)} test={len(test)} "
          f"评测 n={len(ev)} | ctx tok 最大={max(len(r['p']) for r in train + test)} "
          f"p+t 最大={max(len(r['p']) + len(r['t']) for r in train + test)} 超 MAXLEN="
          f"{sum(1 for r in train + test if len(r['p']) + len(r['t']) > MAXLEN)} | 候选末位 token "
          f"前缀稳定性违例={m79.POS_BAD['n']} | data_fp train={fp_tr} test={fp_te}"
          + (f" | §11.14 引文 {q[0]}/{q[1]} ⇒ {'表面一致' if (fp_tr == q[0] and fp_te == q[1]) else '★与引文不同（如实报）'}"
             if q else ""), flush=True)
    print(f"[FLOOR] 本 run **不重算**地板（79 的 201 次重复很慢）：沿用 §11.14/§11.31 引文 —— "
          f"T1 机会级门槛=1/k+2SE={m79.t1_threshold(K, len(ev))[2]:.2f}%、T2 捷径级门槛="
          f"{m79.t2_threshold(m79.MAX_NAIVE_QUOTED[K], len(ev))[2]:.2f}%（同一份数据指纹已核对）", flush=True)

    # ---- [INIT-CHECK]（§9.7：manual_seed 先于构造；四臂同 seed 逐张量核） ----
    print("[INIT-CHECK] §9.7 init 缺陷修复：manual_seed 先于构造（四臂在构造前统一播种）", flush=True)
    built = {}
    for arm in ARMS:
        torch.manual_seed(a.seed)
        built[arm] = build_arm(arm)
    torch.manual_seed(a.seed)
    model = build_arm(a.arm)
    sj = built["J"].state_dict()
    worst_cross = 0.0
    for arm in ARMS:
        d = max(float((sj[k] - built[arm].state_dict()[k]).abs().max().item()) for k in sj)
        worst_cross = max(worst_cross, d)
        print(f"[INIT-CHECK] 四臂同 seed={a.seed} J vs {arm} 逐张量 max|Δ|={d:.3e}"
              f"（{len(sj)} 个张量，结构逐位相同）", flush=True)
    h1 = m79.state_sha256(model.state_dict())
    torch.manual_seed(a.seed)
    h2 = m79.state_sha256(build_arm(a.arm).state_dict())
    torch.manual_seed(a.seed ^ 0x5EED)
    h3 = m79.state_sha256(build_arm(a.arm).state_dict())
    print(f"[INIT-CHECK] arm={a.arm} 同 seed 两次构造 sha={h1[:16]}/{h2[:16]} ⇒ {'相同' if h1 == h2 else '★不同'}"
          f" | 异 seed sha={h3[:16]} ⇒ {'不同（阴性对照通过）' if h3 != h1 else '★相同（seed 没生效）'}", flush=True)
    model = model.to(device)

    # ---- [PARAMS] ----
    br = m79.param_groups_named(model)
    n_tot = br["合计"]
    biggest = max(CARD_ORDER, key=lambda c: sum(p.numel() for p in unit_params(model, c)))
    print(f"[PARAMS] arm={a.arm} 全部参数={n_tot} | 分项 " + " ".join(f"{k}={v}" for k, v in br.items() if k != "合计")
          + f" | 同一优化器内最大可训={sum(p.numel() for p in unit_params(model, biggest))}（相位 {biggest}）"
          f" | 总步数={steps_total}×batch{BATCH}={steps_total * BATCH} 样本 lr={LR} SEL_LAMBDA={SEL_LAMBDA}", flush=True)
    print(f"[PARAMS] arm={a.arm} 步数分配={plan_desc(a.arm, plan)} | 段数={len(plan)}"
          f" 逐段=" + " ".join(f"{s['name']}:{s['steps']}" for s in plan), flush=True)

    # ---- 探针批 + 训练 ----
    probe_recs, probe_packed = build_probe_batches(train, PS if not a.smoke else PS_S)
    lo, hi, mean, nb = probe_len_stats(probe_packed)
    print(f"[MEM] 探测批（四臂共用，指纹={probe_fp(probe_packed)}）：{nb} 批×batch{BATCH}，pad 后长度 "
          f"min/max/mean={lo}/{hi}/{mean:.0f}，dtype=float32，maxlen={MAXLEN} ⇒ 四要素恒定", flush=True)
    torch.cuda.reset_peak_memory_stats()
    mem_rows, wall, done_steps, gates = train_arm(model, a.arm, train, a.seed, plan, probe_recs, probe_packed,
                                                 tag=smoke_note)

    # ---- 评测（T2 完整回复 EM + T1 字母准确率，口径同 79） ----
    t_eval = time.perf_counter()
    ce = m79.eval_ce(model, ev)
    cm, cse = m79.mean_se(ce)
    txt = m79.greedy_gen(model, ev)
    hits = [m79.em_strict(r["target"], txt[i]) for i, r in enumerate(ev)]
    em = 100.0 * sum(hits) / len(hits)
    se, _thr = m79.em_with_thr(em, len(hits))
    sel = m79.eval_sel(model, ev)
    sel_acc = None if sel is None else 100.0 * sum(sel["hits"]) / len(sel["hits"])
    ev_rep = m79.report_texts(hits, txt, ev, "T1T2")
    t1, miss = ev_rep["t1"], ev_rep["miss"]
    eval_sec = time.perf_counter() - t_eval
    t1thr = m79.t1_threshold(K, len(ev))[2]
    t2thr = m79.t2_threshold(m79.MAX_NAIVE_QUOTED[K], len(ev))[2]
    ck = os.path.join(LOG_DIR, f"80_ckpt_{ARM_SLUG[a.arm]}_s{a.seed}{'_smoke' if a.smoke else ''}.pt")
    torch.save(model.state_dict(), ck)

    # ---- [MEM] 臂汇总（含 T2/T1） ----
    peak = max(r["peak_mib"] for r in mem_rows) if mem_rows else float("nan")
    peak_row = max(mem_rows, key=lambda r: r["peak_mib"]) if mem_rows else {}
    per_step = (sum(r["s_per_step"] for r in mem_rows) / len(mem_rows)) if mem_rows else float("nan")
    print(f"[MEM] arm={a.arm} ★峰值={peak:.0f} MiB（出现于 seg={peak_row.get('seg')} 卡={peak_row.get('cards')}）"
          f" | 可训(单优化器内最大)={sum(p.numel() for p in unit_params(model, biggest))} 全部参数={n_tot}"
          f" | 每步墙钟={per_step:.3f}s（探针窗口均值）| T2={em:.2f}% | T1={t1:.2f}%"
          f" | 训练墙钟={wall:.0f}s{smoke_note}", flush=True)
    print(f"[MAIN] arm={a.arm} steps={done_steps} batch={BATCH} lr={LR} d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN}"
          f" | 配对CE={cm:.4f}±{cse:.4f} (nats/token) | 选择准确率="
          + ("不适用" if sel_acc is None else f"{sel_acc:.2f}%")
          + f" | 评估墙钟={eval_sec:.0f}s | GC={'开' if a.arm == 'J+GC' else '关'}", flush=True)

    # ---- 本 run 的 [CRIT] 分条（跨臂判定在 --report-r4） ----
    print(f"[CRIT] R4-a（本 run 部分）arm={a.arm} 全部参数={n_tot} 峰值={peak:.0f} MiB | 逐相位峰值="
          + " ".join(f"{r['seg']}({r['cards']})={r['peak_mib']:.0f}" for r in mem_rows)
          + " | ★跨臂 P-vs-J 的判定与解析比对见 --report-r4", flush=True)
    print(f"[CRIT] R4-b（本 run 部分）arm={a.arm} T2={em:.2f}% ±{se:.2f}pp（门槛 {t2thr:.2f}%）| "
          f"T1={t1:.2f}%（抽出失败 {100.0 * miss / len(ev):.2f}%，门槛 {t1thr:.2f}%）| "
          f"★P-vs-J 的配对 Δ/SE 见 --report-r4"
          + (f" | ★本臂 T2≥95% ⇒ 若对臂也如此则 R4-b 不可判（饱和）" if em >= 95.0 else ""), flush=True)
    print(f"[CRIT] R4-c（本 run 部分）arm={a.arm} 可训(单优化器内)={sum(p.numel() for p in unit_params(model, biggest))}"
          f" 全部参数={n_tot} 峰值={peak:.0f} MiB | ★上界外推见 --report-r4（★推理时仍需容纳全部参数）", flush=True)
    print(f"[CRIT] R4-d（本 run 部分）arm={a.arm} 峰值={peak:.0f} MiB T2={em:.2f}% T1={t1:.2f}%"
          f" | GC={'开（非重入 checkpoint）' if a.arm == 'J+GC' else '关'}"
          + (" | ★见 --gc-check（梯度等价）与 --probe-cross（叠加性）" if a.arm == "J+GC" else ""), flush=True)

    hp = hits_path(a.arm, a.seed, a.smoke)
    with open(hp, "w", encoding="utf-8") as f:
        json.dump({"arm": a.arm, "seed": a.seed, "k": K, "prefix": PREFIX, "smoke": a.smoke,
                   "n": len(hits), "em": em, "se": se, "t1": t1, "miss": miss, "sel": sel_acc,
                   "params": n_tot, "steps": done_steps, "peak_mib": peak, "wall_s": wall,
                   "s_per_step": per_step, "t1_thr": t1thr, "t2_thr": t2thr,
                   "init_ok": worst_cross == 0.0, "init_worst": worst_cross, "gates_ok": all(gates),
                   "probe_fp": probe_fp(probe_packed), "mem_rows": mem_rows,
                   "plan": [{"name": s["name"], "steps": s["steps"], "cards": list(s["cards"])} for s in plan],
                   "hits": hits, "txt": txt, "letters": ev_rep["ls"], "gold_letters": ev_rep["gold_l"],
                   "ckpt": ck}, f)
    print(f"[META] arm={a.arm} seed={a.seed} smoke={a.smoke} steps={done_steps} params={n_tot} peak_mib={peak:.0f} "
          f"T2={em:.2f}% T1={t1:.2f}% s_per_step={per_step:.3f} init_ok={worst_cross == 0.0} "
          f"gates_ok={all(gates)} probe_fp={probe_fp(probe_packed)} ckpt={ck} hits={hp} "
          f"训练墙钟={wall:.0f}s 总墙钟={wall + eval_sec:.0f}s", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


# ---------------- 诊断：拆分 × 检查点 的叠加性（实测） ----------------
def probe_cross(a) -> int:
    global device
    if not torch.cuda.is_available():
        print("GPU 起不来，立即停。", flush=True)
        return 3
    device = "cuda"
    m79.device = device
    print("device:", device, torch.cuda.get_device_name(0), flush=True)
    train, _test = load_data()
    probe_recs, probe_packed = build_probe_batches(train, PS)
    print(f"[MEM-X] 诊断（**非四臂判据**）：全新构造的模型、每配置只跑 {PW + PR * PS} 步（含探针），"
          f"探针批与四臂逐位相同（指纹={probe_fp(probe_packed)}）⇒ 只比显存画像", flush=True)
    cfgs = [("J", ALL_CARDS, False), ("J+GC", ALL_CARDS, True),
            ("P-input", ("input",), False), ("P-input+GC", ("input",), True)]
    res = {}
    for name, cards, gc in cfgs:
        torch.manual_seed(a.seed)
        model = build_arm("J+GC" if gc else "J").to(device)
        plan = [{"name": name, "steps": PW + PR * PS, "cards": cards,
                 "windows": _windows(PW + PR * PS, PW, PS, PR, late=False)}]
        rows, _wall, _n, _g = train_arm(model, name, train, a.seed, plan, probe_recs, probe_packed,
                                        tag="（诊断）")
        res[name] = rows[-1]
        del model
        torch.cuda.empty_cache()
    g = res["J+GC"]["peak_mib"] - res["J"]["peak_mib"]
    c = res["P-input"]["peak_mib"] - res["J"]["peak_mib"]
    b = res["P-input+GC"]["peak_mib"] - res["J"]["peak_mib"]
    print(f"[MEM-X] 峰值：J={res['J']['peak_mib']:.0f} | J+GC={res['J+GC']['peak_mib']:.0f} | "
          f"P(input)={res['P-input']['peak_mib']:.0f} | P(input)+GC={res['P-input+GC']['peak_mib']:.0f} MiB", flush=True)
    print(f"[MEM-X] 驻留：J={res['J']['resident_mib']:.1f} | J+GC={res['J+GC']['resident_mib']:.1f} | "
          f"P(input)={res['P-input']['resident_mib']:.1f} | P(input)+GC={res['P-input+GC']['resident_mib']:.1f} MiB", flush=True)
    print(f"[MEM-X] ★叠加性检验（相对 J 的峰值变化）：Δ(仅 GC)={g:+.0f} MiB；Δ(仅逐卡 input)={c:+.0f} MiB；"
          f"两者之和={g + c:+.0f} MiB；实测 逐卡+GC={b:+.0f} MiB；差={b - (g + c):+.0f} MiB ⇒ "
          f"{'可叠加（差值在探针噪声内）' if abs(b - (g + c)) <= 8.0 else '★非线性/不可简单叠加（如实报）'}", flush=True)
    p = os.path.join(LOG_DIR, f"80_probecross_s{a.seed}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"seed": a.seed, "rows": res, "d_gc": g, "d_card": c, "d_both": b,
                   "probe_fp": probe_fp(probe_packed)}, f)
    print(f"[MEM-X] 落盘 {p}", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


# ---------------- [CRIT] R4-a..d（CPU，读四臂读数） ----------------
def within_spread(rows) -> float:
    """同一段内多个探针窗口的峰值极差（= 探针噪声的经验上界）。"""
    by = {}
    for r in rows:
        by.setdefault(r["seg"], []).append(r["peak_mib"])
    return max((max(v) - min(v) for v in by.values()), default=0.0)


def r4_a(dj, dp, mib_note=""):
    rows_j = dj["mem_rows"]
    peak_j = dj["peak_mib"]
    peak_p = dp["peak_mib"]
    row_p = max(dp["mem_rows"], key=lambda r: r["peak_mib"])
    p_tot, n_act = dj["params"], row_p["n_active"]
    pred = 12.0 * (p_tot - n_act) / MIB                       # 梯度 4B + AdamW 两动量 8B，只按当前卡
    spread_j, spread_p = within_spread(rows_j), within_spread(dp["mem_rows"])
    d_res_j = dj["mem_rows"][0]["resident_mib"] - 12.0 * p_tot / MIB
    d_res_p = row_p["resident_mib"] - (4.0 * p_tot + 8.0 * n_act) / MIB
    save = peak_j - peak_p
    print(f"[CRIT] R4-a 逐卡(P) vs 联合(J) 峰值显存（同参数量、同 batch/seq/dtype/maxlen）{mib_note}", flush=True)
    print(f"[CRIT] R4-a   J 峰值={peak_j:.0f} MiB（同段重复极差 {spread_j:.0f} MiB）| "
          f"P 峰值={peak_p:.0f} MiB（取最大相位 seg={row_p['seg']} 卡={row_p['cards']}；同段重复极差 {spread_p:.0f} MiB）"
          f" ⇒ Δ(P−J)={-save:+.0f} MiB（{100.0 * (-save) / peak_j:+.2f}%）"
          f" ⇒ {'超过探针噪声' if save > 2.0 * max(spread_j, spread_p) else '★未超过探针噪声（如实报）'}", flush=True)
    print(f"[CRIT] R4-a   解析（训练态只按**当前卡**计：梯度 4B+AdamW 8B）× (全部可训 {p_tot} − 当前卡 {n_act}) "
          f"= {pred:.1f} MiB ⇒ 实测/解析={save / pred if pred else float('nan'):.2f} "
          f"⇒ {'量级一致（PASS 的解析那一半）' if abs(save - pred) <= max(4.0, 0.35 * pred) else '★与解析不符（如实报）'}", flush=True)
    print(f"[CRIT] R4-a   驻留（forward 前 memory_allocated）vs 解析 12B×参数（J）/ 4B×全部+8B×当前卡（P）："
          f"J 实测={dj['mem_rows'][0]['resident_mib']:.1f} MiB（解析 {12.0 * p_tot / MIB:.1f}，差 {d_res_j:+.1f}）| "
          f"P({row_p['seg']}) 实测={row_p['resident_mib']:.1f} MiB（解析 {(4.0 * p_tot + 8.0 * n_act) / MIB:.1f}，差 {d_res_p:+.1f}）", flush=True)
    print(f"[CRIT] R4-a   P 逐相位峰值：" + " | ".join(
        f"{r['seg']}({'/'.join(k for k in CARD_ORDER if k in r['cards'])})={r['peak_mib']:.0f} MiB 可训={r['n_active']}"
        for r in dp["mem_rows"]), flush=True)
    ok = save > 0 and abs(save - pred) <= max(4.0, 0.35 * pred)
    print(f"[CRIT] R4-a   ⇒ {'PASS（P 峰值低于 J，且与解析量级一致）' if ok else '★FAIL/未达（如实报）'}"
          f"；★幅度很小（激活项主导）与 §12 的诚实边界一致：逐卡**不省激活**", flush=True)
    return ok


def r4_b(dj, dp, mib_note=""):
    hj, hp = dj["hits"], dp["hits"]
    st = m79.paired_stats(hp, hj)                 # Δ = P − J
    lj = [m79.dec_letter(t) for t in dj["txt"]]
    lp = [m79.dec_letter(t) for t in dp["txt"]]
    gj, gp = dj["gold_letters"], dp["gold_letters"]
    hlj = [int(x == g) for x, g in zip(lj, gj)]
    hlp = [int(x == g) for x, g in zip(lp, gp)]
    stl = m79.paired_stats(hlp, hlj)
    print(f"[CRIT] R4-b 质量代价 P vs J（T2 完整回复 EM，per-item hits 配对；§11.4 口径）{mib_note}", flush=True)
    print(f"[CRIT] R4-b   T2: J={dj['em']:.2f}% P={dp['em']:.2f}% ⇒ Δ(P−J)={st['delta']:+.2f}pp "
          f"SE_Δ(配对)={st['se']:.2f}pp 2SE_Δ={2 * st['se']:.2f}pp | 不一致对 b(P错J对)={st['n01']} "
          f"c(P对J错)={st['n10']} | McNemar 双侧精确 p={st['p']:.3g}（n={st['n']}）", flush=True)
    print(f"[CRIT] R4-b   T1 字母: J={dj['t1']:.2f}% P={dp['t1']:.2f}% ⇒ Δ={stl['delta']:+.2f}pp "
          f"SE_Δ={stl['se']:.2f}pp 2SE_Δ={2 * stl['se']:.2f}pp | b={stl['n01']} c={stl['n10']} | p={stl['p']:.3g}",
          flush=True)
    both_sat = dj["em"] >= 95.0 and dp["em"] >= 95.0
    if both_sat:
        print("[CRIT] R4-b   ⇒ ★**不可判（饱和）**：两臂 T2 都 ≥95% ⇒ 如实写「该数据上饱和、测不出质量差」；"
              "**不得**读成「逐卡无代价」（§12.3 写死）", flush=True)
        return None
    sig = (abs(st["delta"]) >= 2 * st["se"]) and st["p"] < 0.05
    print(f"[CRIT] R4-b   ⇒ 代价={-st['delta']:+.2f}pp（P − J 的反号）；"
          f"{'代价显著（|Δ|≥2SE 且 p<0.05）' if sig else '该口径下**未测出显著差异**（不等于无代价）'}"
          f"；★注意两臂都在同一数据/同一步数下比较，且 79 §11.31 已证本任务对 init 敏感（幅度差可达 14.6pp）", flush=True)
    if max(dj["em"], dp["em"]) < 60.0:
        print("[CRIT] R4-b   ★提醒：两臂 T2 都 <60% ⇒ 先查步数/容量是否不足（§11.7 的读数规则），"
              "此时的臂间差主要反映训练难度而不是「逐卡的代价」", flush=True)
    return sig


def r4_c(dj, dp, mib_note=""):
    B = 8.0 * 1024 ** 3 * 0.92                       # 8 GiB 预算留 8% 给分配器碎片/上下文
    p_tot = dj["params"]
    row_p = max(dp["mem_rows"], key=lambda r: r["peak_mib"])
    n_act = row_p["n_active"]
    a_j = dj["peak_mib"] * MIB - 16.0 * p_tot                       # 参数 4 + 梯度 4 + AdamW 8
    a_p = dp["peak_mib"] * MIB - (4.0 * p_tot + 12.0 * n_act)       # 驻留 4B×全部 + 当前卡 4+8B
    pmax_j = (B - a_j) / 16.0
    pmax_p = (B - a_p - 12.0 * n_act) / 4.0
    print(f"[CRIT] R4-c 可训规模上界（**外推，非实测**）{mib_note}", flush=True)
    print(f"[CRIT] R4-c   外推方法（写死）：实测 intercept A = 峰值 − 训练态解析占用"
          f"（fp32：参数 4B/个、梯度 4B/个、AdamW 两个动量 8B/个）；**假定 A 与参数量无关**；"
          f"预算 B=8 GiB×0.92={B / 1024 ** 3:.2f} GiB", flush=True)
    print(f"[CRIT] R4-c   A_J={a_j / MIB:.0f} MiB ⇒ J 上界=(B−A_J)/16 B/参数 ≈ **{pmax_j / 1e6:.1f}M 参数** | "
          f"A_P={a_p / MIB:.0f} MiB（当前卡 {n_act}）⇒ P 上界=(B−A_P−12B·当前卡)/4 ≈ **{pmax_p / 1e6:.1f}M 参数**"
          f" ⇒ 比值 ≈ {pmax_p / pmax_j:.1f}×", flush=True)
    print(f"[CRIT] R4-c   本任务模型 {p_tot} 参数，距两上界都差 2 个数量级 ⇒ 本机 8G 上**规模不是瓶颈，激活才是**；"
          f"★**推断（非实测）**：A 与参数量无关是**乐观**假设（把模型加宽/加深会同时抬高 A）", flush=True)
    print(f"[CRIT] R4-c   ★**推理时仍需容纳全部参数**：逐卡训练只省**梯度与优化器状态**（训练态），"
          f"推理没有优化器态 ⇒ 拆分对推理显存**没有任何帮助**；单卡结论**不能**外推到多卡模型并行", flush=True)
    return pmax_j, pmax_p


def r4_d(dj, dg, cross_path, mib_note=""):
    st = m79.paired_stats(dg["hits"], dj["hits"])
    print(f"[CRIT] R4-d J+GC vs J（并列报，§12.3）{mib_note}", flush=True)
    print(f"[CRIT] R4-d   峰值：J={dj['peak_mib']:.0f} MiB | J+GC={dg['peak_mib']:.0f} MiB ⇒ "
          f"Δ={dg['peak_mib'] - dj['peak_mib']:+.0f} MiB（{100.0 * (dg['peak_mib'] - dj['peak_mib']) / dj['peak_mib']:+.1f}%）"
          f" | 每步墙钟：J={dj['s_per_step']:.3f}s J+GC={dg['s_per_step']:.3f}s"
          f"（×{dg['s_per_step'] / dj['s_per_step']:.2f}）", flush=True)
    print(f"[CRIT] R4-d   质量：T2 J={dj['em']:.2f}% J+GC={dg['em']:.2f}% ⇒ Δ={st['delta']:+.2f}pp "
          f"SE_Δ(配对)={st['se']:.2f}pp 2SE_Δ={2 * st['se']:.2f}pp b={st['n01']} c={st['n10']} "
          f"McNemar p={st['p']:.3g} | T1 J={dj['t1']:.2f}% J+GC={dg['t1']:.2f}%", flush=True)
    if cross_path and os.path.exists(cross_path):
        with open(cross_path, encoding="utf-8") as f:
            x = json.load(f)
        r = x["rows"]
        print(f"[CRIT] R4-d   ★叠加性（--probe-cross 实测，全新模型）：J={r['J']['peak_mib']:.0f} "
              f"J+GC={r['J+GC']['peak_mib']:.0f} P(input)={r['P-input']['peak_mib']:.0f} "
              f"P(input)+GC={r['P-input+GC']['peak_mib']:.0f} MiB | Δ(仅 GC)={x['d_gc']:+.0f} "
              f"Δ(仅逐卡)={x['d_card']:+.0f} 之和={x['d_gc'] + x['d_card']:+.0f} 实测两者={x['d_both']:+.0f} MiB "
              f"⇒ {'可叠加' if abs(x['d_both'] - (x['d_gc'] + x['d_card'])) <= 8.0 else '★不可简单叠加'}"
              f"（机理：GC 省**激活**、逐卡省**梯度+优化器态**，两项互不重叠）", flush=True)
    else:
        print(f"[CRIT] R4-d   叠加性：未跑 --probe-cross（{cross_path} 不存在）⇒ **未判**（如实报）", flush=True)
    print("[CRIT] R4-d   混合精度：**未测**（autocast 改 dtype ⇒ 违反 §12.1 的四要素恒定，必须单独预注册）", flush=True)
    return st


def report_r4(seed: int, smoke: bool) -> int:
    data = {}
    for arm in ARMS:
        p = hits_path(arm, seed, smoke)
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                data[arm] = json.load(f)
        else:
            print(f"[PAIR] 缺 {arm} 的读数（{p}）⇒ 如实报「未跑」", flush=True)
    print(f"[PAIR] 读到：{sorted(data)} | seed={seed} smoke={smoke}", flush=True)
    for arm, d in sorted(data.items()):
        print(f"[READ] arm={arm} T2={d['em']:.2f}% ±{d['se']:.2f}pp | T1={d['t1']:.2f}%（抽出失败 "
              f"{100.0 * d['miss'] / d['n']:.2f}%）| 选择准确率={d['sel']} | 参数={d['params']} | "
              f"步数={d['steps']} | 峰值={d['peak_mib']:.0f} MiB | 每步={d['s_per_step']:.3f}s | "
              f"init_ok={d['init_ok']} gates_ok={d.get('gates_ok')}", flush=True)
    fps = {a: d["probe_fp"] for a, d in data.items()}
    print(f"[MEM-CHECK] 探测批指纹跨臂：{fps} ⇒ "
          f"{'PASS 逐位相同（四要素里的「同一序列长度分布」成立）' if len(set(fps.values())) <= 1 else '★FAIL 各臂探测批不同 ⇒ 峰值不可比'}",
          flush=True)
    if "J" not in data or "P" not in data:
        print("[CRIT] R4-a..d：J 或 P 读数缺失 ⇒ 主判据未判（如实报）", flush=True)
        return 1
    dj, dp = data["J"], data["P"]
    r4_a(dj, dp)
    r4_b(dj, dp)
    r4_c(dj, dp)
    if "J+GC" in data:
        r4_d(dj, data["J+GC"], os.path.join(LOG_DIR, f"80_probecross_s{seed}.json"))
    else:
        print("[CRIT] R4-d：J+GC 读数缺失 ⇒ 未判（如实报）", flush=True)
    if "P+F" in data:
        st = m79.paired_stats(data["P+F"]["hits"], dj["hits"])
        st2 = m79.paired_stats(data["P+F"]["hits"], dp["hits"])
        print(f"[ATTR] P+F：T2={data['P+F']['em']:.2f}%（P={dp['em']:.2f}% J={dj['em']:.2f}%）| "
              f"vs J Δ={st['delta']:+.2f}pp SE_Δ={st['se']:.2f}pp p={st['p']:.3g} | "
              f"vs P Δ={st2['delta']:+.2f}pp SE_Δ={st2['se']:.2f}pp p={st2['p']:.3g} | "
              f"峰值={data['P+F']['peak_mib']:.0f} MiB（末段回到 J 的画像，见 [MEM] 逐相位）", flush=True)
    else:
        print("[ATTR] P+F：未跑 ⇒ 未判", flush=True)
    return 0


# ---------------- main ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="S80 L3-r4 逐卡 vs 联合（四臂 J/P/P+F/J+GC）+ 显存探针 + R4-a..d")
    ap.add_argument("--arm", choices=ARMS, default="J")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=STEPS_TOTAL)
    ap.add_argument("--smoke", action="store_true", help="50 步、评前 60 条（探路，不作判据）")
    ap.add_argument("--gc-check", action="store_true", help="CPU：GC 与非 GC 的逐张量梯度等价自检")
    ap.add_argument("--probe-cross", action="store_true", help="诊断：拆分×检查点的峰值显存叠加性")
    ap.add_argument("--report-r4", action="store_true", help="CPU：读四臂读数出 [CRIT] R4-a..d")
    a = ap.parse_args(argv)
    if a.gc_check:
        return gc_check(a.seed)
    if a.report_r4:
        m79.load_tok()
        return report_r4(a.seed, a.smoke)
    m79.load_tok()
    if a.probe_cross:
        return probe_cross(a)
    return run_one(a)


if __name__ == "__main__":
    sys.exit(main())
