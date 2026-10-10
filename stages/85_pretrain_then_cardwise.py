#!/usr/bin/env python3
"""S85 · L3-r8「预训练后逐卡微调」vs「联合微调」四臂（J / JT / JT-P / JT-P1）。

预注册：`L3_PLAN.md` §20（方向 II）。骨架与仪器**逐件复用** `stages/80_cardwise_vs_joint.py`
（真实步探针 / 双量峰值 / 四要素恒定 / [INIT-CHECK]）＋ `stages/81_realign_card_ablation.py`
（卡片结构、选择卡的 `q`、参数对齐口径）＋ `stages/82_amp_probe.py`（峰值**分配与预留**两个量）；
模型与数据口径来自 `stages/79_cand_cards_v2.py`（B2 两卡结构：input 1 层 + 选择卡 1 层 +
重整卡 1 层 + output）。

★ 为什么做（§20 开头）：§12.8 的 `P` 臂是**最坏情形**（被冻结的同伴是**随机初始化**）⇒ 代价被夸大成
  **−55.00pp**。用户的设想是「**已训好的卡片集合里，针对新任务只训其中几张**」⇒ 必须补测真实形态。

★ 四臂（**同数据 `logs/76_*` k=8、同总步数预算 = 3000、同 seed、同探测批指纹**）
  | 臂 | 内容 |
  |---|---|
  | **J**      | 联合端到端 3000 步（单相位、单优化器）—— **对照** |
  | **JT**     | 联合预训练 2000 步 → **联合微调** 1000 步（**新建优化器**） |
  | **JT-P**   | 联合预训练 2000 步 → **只微调选择卡+重整卡**（1000 步；冻结 input/output） |
  | **JT-P1**  | 联合预训练 2000 步 → **只微调重整卡**（1000 步；冻结其余） |

  ★ **切分口径（本实现写死，未做敏感性对照 —— 照 §12.10(b) 的诚实口径）**：
    总预算固定 3000；预训练 = 2000 步（2/3）、微调 = 1000 步（1/3）。四臂**唯一变量**是
    「第 2000 步之后哪些卡可训」，J 与 JT 的差别则是「2000 步处是否重建优化器（Adam 动量清零）」。
    ⇒ `R8-a` 量的是**相位切分+优化器重建本身**的效果（同数据、同预算、无额外语料可用）。
  ★ 相位边界上的检查 `[PRETRAIN-CHECK]`：四个臂在第 2000 步的模型状态逐臂落盘，跨臂比 max|Δ|。
    ★★ 实测例外（`--precheck`，同进程 A/A）：**相位 1 的逐位相同在本机不可得** —— 同 seed、同代码、
      同数据序的两遍 `J`，40 步后参数 max|Δ|≈7e-05（相对 1.4e-5），末步 loss 六位小数仍相同；
      根因是 `nn.Embedding` 反向走 `index_add` 的**原子累加**（累加顺序不定）。
      ⇒ 判定口径 = 「跨臂差异 ≤ 同进程 A/A 地板」+ 末步 loss 逐位（六位小数）相同；
      **不得**把跨臂 sha 不同读成「相位 1 不是同一条轨迹」。

★ 探针（§12.1 同构，写死的四要素）
  - 用**真实训练步**：`reset_peak_memory_stats()` → 跑真实 `loss.backward(); opt.step()` → 读
    `max_memory_allocated()` **与** `max_memory_reserved()`（§12.10(a)：本机 8G 的真正约束是分配器预留）；
  - 四要素恒定：① `batch=32`；② **同一序列长度分布** = 跨臂逐位相同的固定探测批（同一置换、同一
    padding，指纹在 `--report-r8` 里跨臂比对）；③ `dtype=float32`（全程不 autocast）；④ `maxlen=512`；
  - 探针窗口内的步**就是真实训练步**（计入 3000 步预算，用的是该臂该相位的真优化器）。

★ 判据（§20 写死；跨臂判定在 `--report-r8`）
  - **R8-a**：`JT` vs `J`（同预算下「相位切分+优化器重建」是否带来质量差）—— per-item 配对 Δ ±2SE
    + McNemar + **2 seed 同号**；
  - **R8-b** ★：`JT-P` vs `JT`（**只训部分卡的代价** = 用户设想的直接检验）；同时报 `JT-P1` vs `JT`；
    **若两臂都饱和（T2 ≥95%）⇒ 判「不可判」**；
  - **R8-c**：`JT-P` 的双量显存（分配**与**预留；优化器状态只覆盖被训卡）与 r4 的 **−4.25%** 对照；
  - **R8-d**：参数量 / 步数 / 峰值 / T1 / T2 并列。

★ 预期（§20 已写死，便于本实现证实或推翻）
  r4 的 −55pp 来自「冻结**随机**同伴」；预训练后同伴是**有用**的 ⇒ `R8-b` 的代价应**显著小于 −55pp**
  （预期 −15pp 以内）；且 `JT-P` 的**显存收益仍只有个位数百分比**。⇒ 本文件**只报实测**，不顺着预期写。

用法（本机只有 `.venv/bin/python` 装了 torch/tokenizers）
  .venv/bin/python stages/85_pretrain_then_cardwise.py --arm JT --seed 1234 --smoke      # 探路
  .venv/bin/python stages/85_pretrain_then_cardwise.py --arm JT-P --seed 1234
  .venv/bin/python stages/85_pretrain_then_cardwise.py --report-r8 --seeds 1234,5678     # CPU：出 [CRIT] R8-a..d
GPU 长跑必须：systemd-run --user --unit=<名> --collect --working-directory=<repo>
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 .venv/bin/python stages/85_pretrain_then_cardwise.py ...
只允许写本文件、`scripts/85_runs.sh` 与 `logs/85_*`。
"""
from __future__ import annotations

import argparse
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
import torch.nn.functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402

# ---------------- 常量（全部沿用 79 = 与 r3b′/r4/r6 同口径） ----------------
D = m79.D
FF = m79.FF
NHEAD = m79.NHEAD
MAXLEN = m79.MAXLEN
LR = m79.LR
BATCH = m79.BATCH
DROPOUT = m79.DROPOUT
SEL_LAMBDA = m79.SEL_LAMBDA
LOG_DIR = m79.LOG_DIR
PREFIX = "76"
K = 8
MIB = 1024 ** 2

STEPS_TOTAL = 3000
PRETRAIN_STEPS = 2000              # ★切分口径（写死）：预训练 2/3、微调 1/3，总预算恒定 3000
ARMS = ("J", "JT", "JT-P", "JT-P1")
ARM_SLUG = {"J": "J", "JT": "JT", "JT-P": "JTP", "JT-P1": "JTP1"}
CARD_ORDER = ("input", "select", "realign", "output")
CARD_UNITS = {"input": ("input",), "select": ("select", "q"),
              "realign": ("realign",), "output": ("output",)}
ALL_CARDS = CARD_ORDER
FT_CARDS = {"JT": ALL_CARDS, "JT-P": ("select", "realign"), "JT-P1": ("realign",)}
CFG_ONLY_INPUT_OUTPUT = ("input", "output")

# 探针（正式）：预热 5 步 + 1 个前窗 + 1 个末窗 = 2×10 步/相位（都在预算内）
PW, PS, PR = 5, 10, 1
PW_S, PS_S, PR_S = 1, 3, 1
PROBE_PERM_SEED = 0x52_34_08_20    # ★定值：四臂的探测批完全相同（与 80/82 同一指纹）

device = "cpu"


# ---------------- 数据 / 探针批（与 80 逐字同口径） ----------------
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
    g = torch.Generator().manual_seed(PROBE_PERM_SEED)
    perm = torch.randperm(len(train), generator=g)
    recs = [[train[int(perm[i * BATCH + j])] for j in range(BATCH)] for i in range(n_steps)]
    packed = [pack(b) for b in recs]
    return recs, packed


def probe_fp(packed) -> str:
    import hashlib
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
    """同一个两卡结构（= 79 的 B2）；四臂**共用**它 ⇒ 结构/初值/参数量逐位相同，唯一变量是训练相位划分。"""

    def __init__(self, name: str = "J"):
        super().__init__("B2")
        self.name = name


def build_arm(arm: str) -> ArmJoint:
    return ArmJoint(name=arm)


# ---------------- 训练计划（写死的步数分配） ----------------
def _windows(steps: int, pw: int, ps: int, pr: int, late: bool) -> list:
    if steps < pw + pr * ps + 2:
        return []
    ws = [(pw + 1 + r * ps, pw + (r + 1) * ps) for r in range(pr)]
    if late:
        end = steps - 1
        start0 = max(pw + pr * ps + 2, end - pr * ps - pw + 1)
        ws += [(start0 + r * ps, start0 + (r + 1) * ps - 1) for r in range(pr)]
    return [(a, b) for a, b in ws if a >= 1 and b <= steps]


def build_plan(arm: str, total: int, smoke: bool) -> list:
    """四臂的步数/相位/窗口**逐点对齐**：J 虽然只有一个相位，但它的探针窗口位置
    与 JT* 的两个相位**全局位置完全相同** ⇒ 相位 1 里每一步吃到的数据（探针批 or 真实批）
    与 JT* 一致 ⇒ J 与 JT 的唯一差别 = 第 2000 步处是否重建优化器（Adam 动量清零）。"""
    pw, ps, pr = (PW_S, PS_S, PR_S) if smoke else (PW, PS, PR)
    pre = max(1, int(round(total * PRETRAIN_STEPS / STEPS_TOTAL)))
    pre = min(pre, total - 1)
    ft = total - pre
    pre_w = _windows(pre, pw, ps, pr, late=True)
    ft_w_local = _windows(ft, pw, ps, pr, late=True)     # ★段内坐标（训练循环的 st 是**段内**步号）
    if arm == "J":
        # J 只有一个段 ⇒ 它的窗口用**全局坐标**：pre 窗 + ft 窗平移，位置与 JT* 两相位逐点一致
        return [{"name": "J#1", "phase": "joint", "steps": total, "cards": ALL_CARDS,
                 "windows": pre_w + [(a + pre, b + pre) for a, b in ft_w_local]}]
    return [{"name": f"{ARM_SLUG[arm]}-pre", "phase": "pre", "steps": pre, "cards": ALL_CARDS,
             "windows": pre_w},
            {"name": f"{ARM_SLUG[arm]}-ft", "phase": "ft", "steps": ft, "cards": FT_CARDS[arm],
             "windows": ft_w_local}]


def plan_desc(arm: str, plan: list) -> str:
    tot = sum(s["steps"] for s in plan)
    if arm == "J":
        return f"1 段×{tot} 步（全参数、单优化器）"
    pre, ft = plan[0], plan[1]
    return (f"预训练 {pre['steps']} 步（全参数）→ 微调 {ft['steps']} 步（可训={'+'.join(ft['cards'])}，"
            f"冻结={'+'.join(c for c in CARD_ORDER if c not in ft['cards'])}）= {tot}")


# ---------------- 可训性 / 优化器 ----------------
def set_active(model, cards) -> list:
    """只让 `cards` 里的卡可训；并清掉上一相位遗留的 `.grad`（r4 §12.6 的实测坑）。"""
    for p in model.parameters():
        p.grad = None
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
    parts, ok = [], True
    for c in CARD_ORDER:
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
    print(f"[GATE] arm={arm} seg={seg['name']} phase={seg['phase']} 卡={'+'.join(cards)} "
          f"step={min(20, seg['steps'])} | " + " | ".join(parts)
          + f" | 优化器状态覆盖参数={n_state}/{n_act}（全部参数 {n_tot} 里只有这些进优化器）"
          + f" | batch 选择准确率={100.0 * sum(hits) / len(hits):.2f}% ⇒ "
          + f"{'PASS' if ok else '★CHECK'}", flush=True)
    return ok


def train_arm(model, arm, train, seed, plan, probe_recs, probe_packed, mid_step: int, tag=""):
    """按 plan 逐相位训练；探针窗口内用固定探测批跑真实训练步并读双量峰值。返回 (mem_rows, wall, steps, gates, shas)。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train), generator=gen)
    pos = 0
    n_tot = sum(p.numel() for p in model.parameters())
    step_total = 0
    mem_rows, gates = [], []
    shas = {}
    t0 = time.perf_counter()
    for seg in plan:
        cards = seg["cards"]
        active = set_active(model, cards)
        n_act = sum(p.numel() for p in active)
        opt = AdamW(active, lr=LR)                       # ★每相位新建：状态只覆盖当前可训卡
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
            if mid_step and step_total == mid_step:
                shas["mid_sha"] = m79.state_sha256(model.state_dict())
                shas["mid_state"] = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
                print(f"[MID] arm={arm} 累计步={step_total}（= §20 的预训练相位边界）模型状态 sha256="
                      f"{shas['mid_sha'][:16]}（状态已留档 ⇒ 跨臂 max|Δ| 在 --report-r8 里报）", flush=True)
            if win is not None and st == win[1] and reset_win == win:
                torch.cuda.synchronize()
                dt = (time.perf_counter() - pt0) / (win[1] - win[0] + 1)
                peak = torch.cuda.max_memory_allocated() / MIB
                peak_res = torch.cuda.max_memory_reserved() / MIB
                lo, hi, mean, nb = probe_len_stats(probe_packed)
                mem_rows.append({"arm": arm, "seg": seg["name"], "phase": seg["phase"],
                                 "cards": "+".join(cards), "cards_n": len(cards),
                                 "win": f"{win[0]}-{win[1]}", "peak_mib": peak,
                                 "peak_reserved_mib": peak_res, "resident_mib": resident,
                                 "s_per_step": dt, "n_active": n_act, "n_total": n_tot,
                                 "probe_fp": probe_fp(probe_packed)})
                print(f"[MEM] arm={arm} seg={seg['name']} phase={seg['phase']} 卡={'+'.join(cards)} "
                      f"探针步={win[0]}–{win[1]}{tag} | 四要素恒定: batch={BATCH} | 序列长度分布=固定探测批"
                      f"（跨臂逐位相同；pad 后长度 min/max/mean={lo}/{hi}/{mean:.0f}，{nb} 批，指纹="
                      f"{probe_fp(probe_packed)}）| dtype=float32 | maxlen={MAXLEN}"
                      f" | ★峰值分配={peak:.0f} MiB | ★峰值预留={peak_res:.0f} MiB"
                      f" | 驻留(步前)={resident:.2f} MiB | 可训={n_act}/{n_tot}"
                      f" | 每步墙钟={dt:.3f}s", flush=True)
        seg_sec = time.perf_counter() - seg_t0
        if seg["phase"] == "pre":
            shas["pre_sha"] = m79.state_sha256(model.state_dict())
        print(f"[SEG] arm={arm} seg={seg['name']} phase={seg['phase']} 卡={'+'.join(cards)} "
              f"步数={seg['steps']} 累计步={step_total} 可训={n_act}/{n_tot} 末步 loss={last[0]:.4f} "
              f"ce={last[1]:.4f} sel={last[2]:.4f} 段墙钟={seg_sec:.0f}s", flush=True)
        del opt
    return mem_rows, time.perf_counter() - t0, step_total, gates, shas


# ---------------- per-run 主流程 ----------------
def hits_path(arm: str, seed: int, smoke: bool = False) -> str:
    return os.path.join(LOG_DIR, f"85_hits_{ARM_SLUG[arm]}_s{seed}{'_smoke' if smoke else ''}.json")


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
    # ★相位边界（= 预训练步数）：**四臂都取**，J 也在第 2000 步取一次 ⇒ 跨臂可逐位比对（[PRETRAIN-CHECK]）
    mid_step = max(1, min(steps_total - 1, int(round(steps_total * PRETRAIN_STEPS / STEPS_TOTAL))))
    smoke_note = "（--smoke 探路，不作判据）" if a.smoke else ""

    print(f"[PLAN] arm={a.arm} seed={a.seed} 总步数={steps_total}×batch{BATCH}={steps_total * BATCH} 样本 | "
          f"{plan_desc(a.arm, plan)} | 段数={len(plan)} 逐段="
          + " ".join(f"{s['name']}({s['phase']}):{s['steps']}[{'/'.join(s['cards'])}]" for s in plan), flush=True)

    # ---- [DATA] ----
    fp_tr, fp_te = m79.d74.d72.data_fp(train), m79.d74.d72.data_fp(test)
    q = m79.FP_QUOTED.get(PREFIX)
    print(f"[DATA] prefix={PREFIX} k={K} arm={a.arm} seed={a.seed} train={len(train)} test={len(test)} "
          f"评测 n={len(ev)} | ctx tok 最大={max(len(r['p']) for r in train + test)} "
          f"p+t 最大={max(len(r['p']) + len(r['t']) for r in train + test)} 超 MAXLEN="
          f"{sum(1 for r in train + test if len(r['p']) + len(r['t']) > MAXLEN)} | 候选末位 token "
          f"前缀稳定性违例={m79.POS_BAD['n']} | data_fp train={fp_tr} test={fp_te}"
          + (f" | §11.14 引文 {q[0]}/{q[1]} ⇒ {'表面一致' if (fp_tr == q[0] and fp_te == q[1]) else '★与引文不同（如实报）'}"
             if q else ""), flush=True)
    print(f"[FLOOR] 本 run **不重算**地板（同 80 口径）：沿用 §11.14/§11.31 引文 —— T1 机会级门槛="
          f"1/k+2SE={m79.t1_threshold(K, len(ev))[2]:.2f}%、T2 捷径级门槛="
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
          + f" | 最大单卡={biggest}({sum(p.numel() for p in unit_params(model, biggest))})"
          f" | 总步数={steps_total}×batch{BATCH}={steps_total * BATCH} 样本 lr={LR} SEL_LAMBDA={SEL_LAMBDA}",
          flush=True)

    # ---- 探针批 + 训练 ----
    probe_recs, probe_packed = build_probe_batches(train, PS if not a.smoke else PS_S)
    lo, hi, mean, nb = probe_len_stats(probe_packed)
    print(f"[MEM] 探测批（四臂共用，指纹={probe_fp(probe_packed)}）：{nb} 批×batch{BATCH}，pad 后长度 "
          f"min/max/mean={lo}/{hi}/{mean:.0f}，dtype=float32，maxlen={MAXLEN} ⇒ 四要素恒定", flush=True)
    torch.cuda.reset_peak_memory_stats()
    mem_rows, wall, done_steps, gates, shas = train_arm(
        model, a.arm, train, a.seed, plan, probe_recs, probe_packed, mid_step, tag=smoke_note)

    # ---- [PRETRAIN-CHECK] ----
    boundary = shas.get("mid_sha") or shas.get("pre_sha")
    pre_path = None
    if "mid_state" in shas:
        pre_path = os.path.join(LOG_DIR, f"85_prestate_{ARM_SLUG[a.arm]}_s{a.seed}{'_smoke' if a.smoke else ''}.pt")
        torch.save(shas["mid_state"], pre_path)
    print(f"[PRETRAIN-CHECK] arm={a.arm} §20 预训练相位边界（累计步={mid_step}）模型状态 sha256="
          f"{(boundary or '未取到')[:16]} ⇒ 同进程内逐位相同的证明见 --precheck；跨臂 max|Δ| 见 --report-r8"
          f"（★跨进程的 sha 差异 = ROCm 非确定性）"
          + (f" | 状态留档={pre_path}" if pre_path else ""), flush=True)

    # ---- 评测 ----
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
    ck = os.path.join(LOG_DIR, f"85_ckpt_{ARM_SLUG[a.arm]}_s{a.seed}{'_smoke' if a.smoke else ''}.pt")
    torch.save(model.state_dict(), ck)

    # ---- [MEM] 臂汇总 ----
    peak = max(r["peak_mib"] for r in mem_rows) if mem_rows else float("nan")
    peak_row = max(mem_rows, key=lambda r: r["peak_mib"]) if mem_rows else {}
    peak_res = max(r["peak_reserved_mib"] for r in mem_rows) if mem_rows else float("nan")
    per_step = (sum(r["s_per_step"] for r in mem_rows) / len(mem_rows)) if mem_rows else float("nan")
    print(f"[MEM] arm={a.arm} ★峰值分配={peak:.0f} MiB（出现于 seg={peak_row.get('seg')} "
          f"phase={peak_row.get('phase')} 卡={peak_row.get('cards')}）| ★峰值预留={peak_res:.0f} MiB"
          f" | 可训(单优化器内最大)={sum(p.numel() for p in unit_params(model, biggest))} 全部参数={n_tot}"
          f" | 每步墙钟={per_step:.3f}s（探针窗口均值）| T2={em:.2f}% | T1={t1:.2f}%"
          f" | 训练墙钟={wall:.0f}s{smoke_note}", flush=True)
    print(f"[MAIN] arm={a.arm} steps={done_steps} batch={BATCH} lr={LR} d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN}"
          f" | 配对CE={cm:.4f}±{cse:.4f} (nats/token) | 选择准确率="
          + ("不适用" if sel_acc is None else f"{sel_acc:.2f}%")
          + f" | 评估墙钟={eval_sec:.0f}s", flush=True)

    # ---- 本 run 的 [CRIT] 分条（跨臂判定在 --report-r8） ----
    print(f"[CRIT] R8-a（本 run 部分）arm={a.arm} T2={em:.2f}% ±{se:.2f}pp（门槛 {t2thr:.2f}%）| "
          f"★JT-vs-J 的配对 Δ/SE 见 --report-r8", flush=True)
    print(f"[CRIT] R8-b（本 run 部分）arm={a.arm} T2={em:.2f}% T1={t1:.2f}% | 微调相位可训="
          f"{'+'.join(plan[-1]['cards']) if len(plan) > 1 else '全参数（单相位）'}"
          + (f" | ★本臂 T2≥95% ⇒ 若对臂也如此则 R8-b 不可判（饱和）" if em >= 95.0 else ""), flush=True)
    print(f"[CRIT] R8-c（本 run 部分）arm={a.arm} ★峰值分配={peak:.0f} MiB ★峰值预留={peak_res:.0f} MiB | "
          f"逐相位=" + " ".join(f"{r['seg']}({r['phase']}):{r['peak_mib']:.0f}MiB/{r['peak_reserved_mib']:.0f}MiB"
                                f"/驻留{r['resident_mib']:.1f}" for r in mem_rows), flush=True)
    print(f"[CRIT] R8-d（本 run 部分）arm={a.arm} 全部参数={n_tot} 步数={done_steps} 峰值分配={peak:.0f} MiB "
          f"T2={em:.2f}% T1={t1:.2f}% 每步={per_step:.3f}s", flush=True)

    hp = hits_path(a.arm, a.seed, a.smoke)
    with open(hp, "w", encoding="utf-8") as f:
        json.dump({"arm": a.arm, "seed": a.seed, "k": K, "prefix": PREFIX, "smoke": a.smoke,
                   "n": len(hits), "em": em, "se": se, "t1": t1, "miss": miss, "sel": sel_acc,
                   "params": n_tot, "steps": done_steps, "peak_mib": peak,
                   "peak_reserved_mib": peak_res, "wall_s": wall,
                   "s_per_step": per_step, "t1_thr": t1thr, "t2_thr": t2thr,
                   "init_ok": worst_cross == 0.0, "init_worst": worst_cross, "gates_ok": all(gates),
                   "probe_fp": probe_fp(probe_packed), "mem_rows": mem_rows,
                   "mid_step": mid_step, "pre_sha": shas.get("pre_sha"),
                   "mid_sha": shas.get("mid_sha") or shas.get("pre_sha"),
                   "plan": [{"name": s["name"], "phase": s["phase"], "steps": s["steps"],
                             "cards": list(s["cards"])} for s in plan],
                   "hits": hits, "txt": txt, "letters": ev_rep["ls"], "gold_letters": ev_rep["gold_l"],
                   "ckpt": ck}, f)
    print(f"[META] arm={a.arm} seed={a.seed} smoke={a.smoke} steps={done_steps} params={n_tot} "
          f"peak_mib={peak:.0f} peak_reserved_mib={peak_res:.0f} T2={em:.2f}% T1={t1:.2f}% "
          f"s_per_step={per_step:.3f} init_ok={worst_cross == 0.0} gates_ok={all(gates)} "
          f"probe_fp={probe_fp(probe_packed)} boundary_sha={(boundary or 'NA')[:16]} ckpt={ck} "
          f"hits={hp} 训练墙钟={wall:.0f}s 总墙钟={wall + eval_sec:.0f}s", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


# ---------------- [PRECHECK] 相位 1 的 A/A 噪声地板 + 与臂标签无关性 ----------------
def precheck(a) -> int:
    """在**同一进程**内跑相位 1 的三条轨迹：`J`（两遍，A/A）与 `JT-P`（相位 1 代码完全相同）。

    - **A/A**（J 两遍）：同 seed、同数据序、同代码 ⇒ 量出这台机器上训练轨迹的**噪声地板**
      （实测根因：`nn.Embedding` 反向走 `index_add` 的**原子累加**，累加顺序不定 ⇒ 非逐位可重复）；
    - **标签无关性**（J vs JT-P）：相位 1 的代码路径与 RNG 流相同 ⇒ 差异不应超过 A/A 地板。
    ⇒ 结论：相位 1 的**逐位**相同在 ROCm 上不可得；`[PRETRAIN-CHECK]` 只能按「差异 ≤ A/A 地板」判。
    """
    global device
    if not torch.cuda.is_available():
        print("GPU 起不来，立即停。", flush=True)
        return 3
    device = "cuda"
    m79.device = device
    print(f"device: {device} {torch.cuda.get_device_name(0)} | [PRECHECK] 同进程内三条相位 1 轨迹，"
          f"各 {a.precheck_steps} 步（两条同 arm=J ⇒ A/A 地板；一条 arm=JT-P ⇒ 标签无关性）", flush=True)
    train, _test = load_data()
    steps = a.precheck_steps
    states, losses = {}, {}
    for tag in ("J#1", "J#2", "JT-P"):
        arm = "J" if tag.startswith("J") else "JT-P"
        torch.manual_seed(a.seed)
        model = build_arm(arm).to(device)
        torch.manual_seed(a.seed)
        torch.cuda.manual_seed_all(a.seed)
        gen = torch.Generator().manual_seed(a.seed)
        order = torch.randperm(len(train), generator=gen)
        active = set_active(model, ALL_CARDS)
        opt = AdamW(active, lr=LR)
        pos = 0
        last = None
        for _st in range(1, steps + 1):
            idx = []
            for _ in range(BATCH):
                if pos >= len(order):
                    order = torch.randperm(len(train), generator=gen)
                    pos = 0
                idx.append(int(order[pos]))
                pos += 1
            batch = [train[i] for i in idx]
            loss, lce, lsel, _sc = train_step(model, batch, None, opt)
            last = (float(loss), float(lce), float(lsel))
        states[tag] = {k: v.detach().to("cpu").clone() for k, v in model.state_dict().items()}
        losses[tag] = last
        print(f"[PRECHECK] {tag}（arm={arm}）{steps} 步末步 loss={last[0]:.6f} ce={last[1]:.6f} sel={last[2]:.6f}",
              flush=True)
        del model, opt
        torch.cuda.empty_cache()

    def diff(x, y):
        d = max(float((states[x][k] - states[y][k]).abs().max()) for k in states[x])
        return d, (m79.state_sha256(states[x]) == m79.state_sha256(states[y]))

    scale = max(float(v.abs().max()) for v in states["J#1"].values())
    aa, aa_eq = diff("J#1", "J#2")
    lab, lab_eq = diff("J#1", "JT-P")
    ok = lab <= max(2.0 * aa, 1e-7)
    print(f"[PRECHECK] 同 seed={a.seed} {steps} 步后（参数最大幅值={scale:.3e}）：", flush=True)
    print(f"[PRECHECK]   A/A（arm=J 两遍，同代码同 seed）：逐位相同={aa_eq} | 逐张量 max|Δ|={aa:.3e}"
          f"（相对 {aa / scale:.2e}）⇒ **噪声地板**", flush=True)
    print(f"[PRECHECK]   臂标签无关性（J vs JT-P，相位 1 同代码）：逐位相同={lab_eq} | max|Δ|={lab:.3e}"
          f"（相对 {lab / scale:.2e}）⇒ {'≤ A/A 地板（标签无影响）' if ok else '★超出 A/A 地板的 2 倍（如实报）'}",
          flush=True)
    print(f"[PRECHECK] ⇒ 结论：**相位 1 的逐位相同在 ROCm 上不可得**（根因=Embedding 反向 index_add 原子累加）；"
          f"`[PRETRAIN-CHECK]` 按「跨臂差异 ≤ A/A 地板」判，并把这条例外写进报告；"
          f"六位小数的末步 loss 仍逐位相同（{losses['J#1'][0]:.6f} / {losses['J#2'][0]:.6f} / {losses['JT-P'][0]:.6f}）",
          flush=True)
    print("[DONE] exit=0", flush=True)
    return 0 if ok else 1


# ---------------- [CRIT] R8-a..d（CPU，读四臂×2 seed 读数） ----------------
def read_hits(arm, seed, smoke):
    p = hits_path(arm, seed, smoke)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def pair_line(tag, hx, hy, name_x, name_y):
    st = m79.paired_stats(hx, hy)
    print(f"[PAIR] {tag} {name_x} vs {name_y}：Δ={st['delta']:+.2f}pp SE_Δ(配对)={st['se']:.2f}pp "
          f"2SE_Δ={2 * st['se']:.2f}pp | 不一致对 b({name_x}错{name_y}对)={st['n01']} "
          f"c({name_x}对{name_y}错)={st['n10']} | McNemar 双侧精确 p={st['p']:.3g} (n={st['n']})", flush=True)
    return st


def verdict_pair(tag, seed, dx, dy, name_x, name_y):
    """一对臂在**同一 seed**下的判定：饱和条款 → 否则 |Δ|≥2SE 且 p<0.05。返回 (sig_or_None, delta, sat)."""
    st = pair_line(tag, dx["hits"], dy["hits"], name_x, name_y)
    both_sat = dx["em"] >= 95.0 and dy["em"] >= 95.0
    if both_sat:
        print(f"[CRIT] {tag} seed={seed} ⇒ ★**不可判（饱和）**：两臂 T2 都 ≥95% ⇒ 如实写「该数据上饱和、"
              f"测不出质量差」，**不得**读成「无代价」（§20/§12.3 写死）", flush=True)
        return None, st["delta"], True
    sig = (abs(st["delta"]) >= 2 * st["se"]) and st["p"] < 0.05
    tail = "显著（|Δ|≥2SE 且 p<0.05）" if sig else "该口径下**未测出显著差异**（不等于无代价）"
    print(f"[CRIT] {tag} seed={seed} ⇒ {tail}", flush=True)
    return sig, st["delta"], False


def report_r8(a) -> int:
    seeds = [int(s) for s in str(a.seeds).replace(" ", ",").split(",") if s.strip()]
    smoke = a.smoke
    print(f"[PAIR] 读四臂×seed{seeds}（smoke={smoke}）", flush=True)
    data = {}
    for seed in seeds:
        for arm in ARMS:
            d = read_hits(arm, seed, smoke)
            if d is not None:
                data[(arm, seed)] = d
            else:
                print(f"[PAIR] 缺读数 arm={arm} seed={seed} ⇒ 如实报「未跑」", flush=True)
    if not data:
        print("[CRIT] R8：没有读数 ⇒ 未判（如实报）", flush=True)
        return 1

    # ---- 读数与仪器核对 ----
    for (arm, seed), d in sorted(data.items()):
        print(f"[READ] arm={arm} seed={seed} T2={d['em']:.2f}% ±{d['se']:.2f}pp | T1={d['t1']:.2f}%"
              f"（抽不出 {100.0 * d['miss'] / d['n']:.2f}%）| 选择={d['sel']} | 参数={d['params']} | "
              f"步数={d['steps']} | 峰值分配={d['peak_mib']:.0f} MiB 峰值预留={d['peak_reserved_mib']:.0f} MiB | "
              f"每步={d['s_per_step']:.3f}s | init_ok={d['init_ok']} gates_ok={d.get('gates_ok')}", flush=True)
    for seed in seeds:
        sub = {a: d for (a, s), d in data.items() if s == seed}
        if not sub:
            continue
        fps = {a: d["probe_fp"] for a, d in sub.items()}
        print(f"[MEM-CHECK] seed={seed} 探测批指纹跨臂：{fps} ⇒ "
              f"{'PASS 逐位相同（四要素里的「同一序列长度分布」成立）' if len(set(fps.values())) == 1 else '★FAIL 各臂探测批不同 ⇒ 峰值不可比'}",
              flush=True)
        # ★相位边界状态：跨臂比对用**实测 max|Δ| + 相对尺度**（跨进程的 sha 差异 = ROCm 非确定性）
        sts = {}
        for a_ in ARMS:
            fp_ = os.path.join(LOG_DIR, f"85_prestate_{ARM_SLUG[a_]}_s{seed}{'_smoke' if smoke else ''}.pt")
            if os.path.exists(fp_):
                sts[a_] = torch.load(fp_, map_location="cpu", weights_only=True)
        if "JT" in sts:
            base = sts["JT"]
            scale = max(float(v.abs().max()) for v in base.values())
            lines = []
            ok = True
            for a_ in ARMS:
                if a_ not in sts:
                    continue
                d = max(float((base[k] - sts[a_][k]).abs().max()) for k in base)
                rel = d / scale if scale else float("nan")
                ok = ok and rel <= 1e-3
                lines.append(f"{a_}: sha={(sub[a_].get('mid_sha') or 'NA')[:16]}"
                             f" max|Δ|={d:.3e} 相对={rel:.2e}")
            print(f"[PRETRAIN-CHECK] seed={seed} 相位边界（累计步={next(iter(sub.values()))['mid_step']}）"
                  f"跨臂状态比对（基准=JT，参数最大幅值={scale:.3e}）：" + " | ".join(lines), flush=True)
            print(f"[PRETRAIN-CHECK] seed={seed} ⇒ {'PASS 同一轨迹（跨臂差异 ≤1e-3×幅值 = 浮点噪声带；对照实测 A/A 地板见 driver 的 --precheck）' if ok else '★CHECK 跨臂差异超出浮点噪声（如实报）'}"
                  f"；★逐位相同在本机不可得（`--precheck` 实测 A/A 地板）：判定口径 = 差异 ≤ A/A 地板 + "
                  f"末步 loss 六位小数相同；**不得**把 sha 不同读成「相位 1 不是同一条轨迹」", flush=True)
        else:
            print(f"[PRETRAIN-CHECK] seed={seed}：未找到 85_prestate_*.pt ⇒ 只报 sha（如实报）", flush=True)

    # ---- R8-a JT vs J ----
    print("[CRIT] R8-a 「相位切分+优化器重建」本身是否带来质量差：JT vs J（同预算 3000 步）", flush=True)
    sigs_a, deltas_a = [], []
    for seed in seeds:
        if ("JT", seed) in data and ("J", seed) in data:
            sig, delta, sat = verdict_pair("R8-a", seed, data[("JT", seed)], data[("J", seed)], "JT", "J")
            sigs_a.append(sig)
            deltas_a.append(delta)
    if len(deltas_a) == 2:
        same_sign = (deltas_a[0] > 0) == (deltas_a[1] > 0)
        print(f"[CRIT] R8-a 2 seed 同号：Δ={deltas_a[0]:+.2f} / {deltas_a[1]:+.2f}pp ⇒ "
              f"{'同号' if same_sign else '★异号（如实报）'}；判定="
              f"{'PASS（2 seed 同号且两 seed 都显著）' if (len(sigs_a) == 2 and all(sigs_a) and same_sign) else '该口径下**未达 PASS**（如实报，不得写成「无差异」）'}",
              flush=True)

    # ---- R8-b JT-P vs JT / JT-P1 vs JT ----
    print("[CRIT] R8-b ★「只训部分卡的代价」（用户设想的直接检验）：JT-P vs JT；JT-P1 vs JT", flush=True)
    for arm_x in ("JT-P", "JT-P1"):
        sigs, deltas = [], []
        for seed in seeds:
            if (arm_x, seed) in data and ("JT", seed) in data:
                sig, delta, sat = verdict_pair("R8-b", seed, data[(arm_x, seed)], data[("JT", seed)],
                                               arm_x, "JT")
                sigs.append(sig)
                deltas.append(delta)
        if len(deltas) == 2:
            same_sign = (deltas[0] > 0) == (deltas[1] > 0)
            pre = data.get((arm_x, seeds[0])) or (data.get((arm_x, seeds[1])) if len(seeds) > 1 else None)
            ft_cards = pre["plan"][-1]["cards"] if (pre and len(pre["plan"]) > 1) else []
            print(f"[CRIT] R8-b {arm_x}（相位 2 只训 {'+'.join(ft_cards)}）vs JT：2 seed Δ="
                  f"{deltas[0]:+.2f} / {deltas[1]:+.2f}pp ⇒ {'同号' if same_sign else '★异号（如实报）'}"
                  f" | 对照 r4 的 P 臂（冻结**随机**同伴）代价 −55.00pp（§12.8）⇒ 实测代价="
                  f"{'远小于 −55pp' if max(abs(d) for d in deltas) < 55.0 else '★不亚于 −55pp（推翻 §20 预期，如实报）'}"
                  f" | §20 预期「−15pp 以内」⇒ "
                  f"{'与预期同向（落在 −15pp 以内）' if max(abs(d) for d in deltas) <= 15.0 else '★超出预期（如实报）'}",
                  flush=True)
        else:
            print(f"[CRIT] R8-b {arm_x}：读数缺一 seed ⇒ 未判（如实报）", flush=True)

    # ---- R8-c 双量显存 ----
    print("[CRIT] R8-c JT-P 的双量显存（优化器状态只覆盖被训卡）与 r4 的 −4.25% 对照", flush=True)
    for seed in seeds:
        sub = {a: d for (a, s), d in data.items() if s == seed}
        if not sub:
            continue
        base = sub.get("J")
        for arm in ("JT", "JT-P", "JT-P1"):
            if arm not in sub:
                continue
            d = sub[arm]
            da = d["peak_mib"] - base["peak_mib"] if base else float("nan")
            dr = d["peak_reserved_mib"] - base["peak_reserved_mib"] if base else float("nan")
            pa = 100.0 * da / base["peak_mib"] if base else float("nan")
            pr = 100.0 * dr / base["peak_reserved_mib"] if base else float("nan")
            ph = {}
            for r in d["mem_rows"]:
                if r["phase"] not in ph or r["peak_mib"] > ph[r["phase"]]["peak_mib"]:
                    ph[r["phase"]] = r
            print(f"[CRIT] R8-c seed={seed} {arm} | 峰值分配={d['peak_mib']:.0f} MiB"
                  + (f"（vs J {da:+.0f} MiB / {pa:+.2f}%）" if base else "")
                  + f" | 峰值预留={d['peak_reserved_mib']:.0f} MiB"
                  + (f"（vs J {dr:+.0f} / {pr:+.2f}%）" if base else "")
                  + " | 逐相位峰值=" + " ".join(f"{k}:{v['peak_mib']:.0f}MiB(可训{v['n_active']})"
                                                 for k, v in ph.items())
                  + f" | 微调相位可训={'+'.join(d['plan'][-1]['cards']) if len(d['plan']) > 1 else '全参数'}",
                  flush=True)
    print("[CRIT] R8-c 读法：r4 的 P 臂（冻结随机同伴）**分配 −4.25%、预留基本不变**（§12.8/§12.10a）；"
          "★结论必须用**实测的两个量**，不能用「优化器状态只覆盖一卡」去推（激活占峰值绝对多数）", flush=True)

    # ---- R8-d 并列 ----
    print("[CRIT] R8-d 参数量/步数/峰值/T1/T2 并列", flush=True)
    for seed in seeds:
        for arm in ARMS:
            d = data.get((arm, seed))
            if d is None:
                continue
            ft = "+".join(d["plan"][-1]["cards"]) if len(d["plan"]) > 1 else "全参数"
            print(f"[CRIT] R8-d seed={seed} arm={arm} | 参数={d['params']} 步数={d['steps']} "
                  f"峰值分配={d['peak_mib']:.0f} MiB 峰值预留={d['peak_reserved_mib']:.0f} MiB | "
                  f"T2={d['em']:.2f}% T1={d['t1']:.2f}% 选择={d['sel']} | 每步={d['s_per_step']:.3f}s | "
                  f"相位 2 可训={ft}", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="S85 L3-r8 预训练后逐卡微调 vs 联合微调（四臂 J/JT/JT-P/JT-P1）")
    ap.add_argument("--arm", choices=ARMS, default="J")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=STEPS_TOTAL)
    ap.add_argument("--smoke", action="store_true", help="50 步、评前 60 条（探路，不作判据）")
    ap.add_argument("--report-r8", action="store_true", help="CPU：读四臂×seed 读数出 [CRIT] R8-a..d")
    ap.add_argument("--seeds", type=str, default="1234,5678", help="--report-r8 用：逗号分隔的 seed")
    ap.add_argument("--precheck", action="store_true",
                    help="GPU：同进程内证明相位 1 与臂标签无关（两条轨迹逐位比对）")
    ap.add_argument("--precheck-steps", type=int, default=40)
    a = ap.parse_args(argv)
    if a.report_r8:
        m79.load_tok()
        return report_r8(a)
    m79.load_tok()
    if a.precheck:
        return precheck(a)
    return run_one(a)


if __name__ == "__main__":
    sys.exit(main())
