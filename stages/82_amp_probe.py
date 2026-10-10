#!/usr/bin/env python3
"""S82 · L3-r6 混合精度（AMP）四臂：J-FP32 / J-AMP-bf16 / J-AMP-fp16 / J+GC-AMP。

预注册：`L3_PLAN.md` §15（本单元的判据 R6-a..e 与「dtype 作为声明变量」的协议）；
探针同构四要素见 §12.1；J / J+GC 的已测旁证见 §12.8/§12.9（2074 / 1484 MiB、T2 61.62%、
叠加性做法 = `--probe-cross` 实测四项）；配对 SE 纪律见 §11.4；init 纪律见 §9.7。

★ 骨架逐件复用 `stages/80_cardwise_vs_joint.py`（真实训练步探针 + 四臂 + [INIT-CHECK] 已就绪），
   **不改 80**；本文件只保留 80 的「联合」那一支，把它复制成四个**数值口径**不同的臂。

★ 四臂（§15 写死）——**同结构、同数据 `logs/76_*`（k=8 带字母）、同步数 3000、同 seed；
  batch=32 / maxlen=512 / 同一探测批（跨臂逐位相同，打指纹核对）**

  | 臂 | 训练数值口径（本文件实测的口径） | GC |
  |---|---|---|
  | `J-FP32`      | 无 autocast（全程 float32）；无 scaler | 关 |
  | `J-AMP-bf16`  | `torch.autocast(device_type="cuda", dtype=torch.bfloat16)`；**无 scaler** | 关 |
  | `J-AMP-fp16`  | `torch.autocast(device_type="cuda", dtype=torch.float16)` + `torch.amp.GradScaler("cuda")` | 关 |
  | `J+GC-AMP`    | **较优的那个 AMP 档**（`--amp-dtype {bf16,fp16}`，由 `--best-amp` 依据峰值/可用性选） | **开**（`torch.utils.checkpoint(use_reentrant=False)`） |

  ⇒ dtype 是**声明变量**：其余三要素（batch / 探测批 / maxlen）全部固定，**每个臂的峰值都在它自己的
     run 内测**，绝不在一个臂里混 dtype 后再当同构比较（§15 的协议；这不违反 §12.1 —— 同构原则禁止的是
    「**无意中**变了 dtype 却当成同构比较」）。

★ 显存探针（§12.1/§15，**必须与真实训练同构**）
  - `torch.cuda.reset_peak_memory_stats()` → 真实 `loss.backward(); opt.step()` → 读
    `torch.cuda.max_memory_allocated()`（MiB；ROCm 下走 hip，API 同名）。**禁止**解析估算代替。
  - 探测批 = `PROBE_PERM_SEED` 定值置换下的同一批样本（与 80 同 seed ⇒ 指纹应同为 `6f0df644384b6422`），
    跨臂逐位相同；`[MEM-CHECK]` 逐臂核对指纹。
  - 探针窗口内这些步**就是真实训练步**（计入 3000 步预算），只是取样被固定。
  - 每臂报：**峰值显存 / 可训参数量 / 每步墙钟 / T2 / T1**（探针行前缀 `[MEM]`）。

★ ROCm / RX 6600（gfx1030 = RDNA2）特定要求：**实测，不许假设**（§15）
  - `[DTYPE]` 行实测**参与计算的 dtype**：用 `TorchDispatchMode` 在**一次真实 forward+backward** 时
    记录 `aten::addmm/mm/bmm/linear/_scaled_dot_product_attention` 等算子**实收**的输入/输出 dtype
    （autocast 的 cast 发生在 Autocast key、位于 Python key 之上 ⇒ 本模式看到的是**内核实收**的 dtype），
    并**明确报「是否发生回退」**（判据：声明 bf16/fp16 时，GEMM 类算子里是否出现 float32）；
  - 旁证：`torch.cuda.is_bf16_supported()`、`allow_bf16_reduced_precision_reduction`、
    `allow_fp16_reduced_precision_reduction`、`gcnArchName`，以及一次**实测 GEMM 微基准**
    （fp32/bf16/fp16 同时长 + 同时长的相对误差）；回退/转换实现会表现为明显慢于 fp32（实测数据说话）；
  - fp16 的 `GradScaler`：报 **scale 初值/末值/最小/变化次数（升/降）** 与
    **被跳过的步数**（实测：`跳过 = 尝试步数 − AdamW 的 state['step'] 末值`，即优化器真正走了几步），
    另报 loss 出现 inf/NaN 的步数（= 溢出触发次数）；
  - 任一档出现 **NaN/Inf**（loss 或梯度）⇒ 如实报该档「**不可用**」，**不得**靠调 lr/加 eps 掩饰。

★ 判据（§15 写死，逐条打印 `[CRIT] R6-a..e`；跨臂判定在 `--report-r6`，支持 `--seeds 1234,5678`）
  - **R6-a**：AMP 档峰值 vs `J-FP32`（同 batch/seq/maxlen）⇒ 降 ≥10% 记「有实质收益」，<10% 如实写
    「本任务上 AMP 不显著省显存」。
  - **R6-b**：质量 Δ（**配对 per-item Δ + 配对 SE（§11.4）+ McNemar**，**2 seed 同号**）；
    两臂都饱和（T2 ≥95%）⇒ 判「**不可判**」（本数据 J≈62% 不饱和 ⇒ 可判）。
  - **R6-c**：与梯度检查点的**叠加性**：★**实测** vs 两项相加（在 `--probe-cross-amp` 里同一次
    全新构造、同一探测批、四种配置各实测一次），报差值；**不得**用相加代替实测。并列报 3000 步
    满程臂的一致性核对（那句里 GC-仅-FP32 那一项是 80 的**跨单元引文**，单独标注）。
  - **R6-d**：数值保真：loss 尺度（首/末 loss 与 CE，nats）、scaler 触发/跳过步数、NaN/Inf。
  - **R6-e**：每步墙钟 / 峰值 / 参数 / T1 / T2 并列。

★ 诚实边界（照 §15 预告）
  - 单卡 RDNA2（gfx1030）的结论**不能**外推到别的架构/厂商；任何"bf16 是原生算的"这类话必须由
    `[DTYPE]` 的实测行支撑，不能只凭 `autocast` 开了没开；
  - 若某档因回退或溢出而**无法公平比较** ⇒ 如实报「**不可判**」，不得硬凑；
  - **评测口径**：四个臂的 T2/T1 **一律在 float32 下评测**（默认 `--eval-amp` 关）⇒ 读数反映**权重**
    而不是评测期的数值路径；该选择写进 `[DTYPE]` 与 JSON 的 `eval_mode`；
  - 探针窗口固定探测批会被重复喂入（40 步 = 2 窗口 × 10 步 × 2 组），是**取样偏移**，
    但四臂步数预算相同（3000×batch32）。
  - 推断与实测分开：`[DTYPE]` 里凡标「推断」的行都不是本机实测。

用法（本机只有 `.venv/bin/python` 装了 torch/tokenizers）
  .venv/bin/python stages/82_amp_probe.py --dtype-probe --seed 1234          # GPU：dtype/回退/微基准/SDPA 开关（~1min）
  .venv/bin/python stages/82_amp_probe.py --sdp-flag-probe --amp-dtype fp16 --seed 1234   # 只量 SDPA 开关的显存效应
  .venv/bin/python stages/82_amp_probe.py --arm J-AMP-bf16 --seed 1234 --smoke
  .venv/bin/python stages/82_amp_probe.py --arm J-FP32 --seed 1234
  .venv/bin/python stages/82_amp_probe.py --arm J+GC-AMP --amp-dtype fp16 --seed 1234
  .venv/bin/python stages/82_amp_probe.py --probe-cross-amp --amp-dtype fp16 --seed 1234   # 叠加性实测
  .venv/bin/python stages/82_amp_probe.py --best-amp --seed 1234             # CPU：选较优 AMP 档（供 runner）
  .venv/bin/python stages/82_amp_probe.py --report-r6 --seeds 1234,5678      # CPU：出 [CRIT] R6-a..e
GPU 长跑必须：systemd-run --user --unit=flowme_l3r6 --collect --working-directory=<repo>
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 bash scripts/82_runs.sh
只允许写本文件与 `logs/82_*`、`scripts/82_runs.sh`。

★ 本机（RX 6600 / gfx1030 / torch 2.9.1+rocm6.4）**实测**的两条机制（`--dtype-probe` 的原始行；不是假设）
  1. **注意力核在 AMP 下仍是 fp32**：`nn.MultiheadAttention` 一旦 `torch.is_autocast_enabled()` 就**禁用
     fast path**（`torch/nn/modules/activation.py:1391-1392`），改走 SDPA 的 math 后端；而该后端在
     `allow_fp16_bf16_reduction_math_sdp=False`（本机默认）时把 q/k/v **上转到 fp32** ⇒ 实测
     `aten::bmm|in=float32|out=float32 ×6`（每层 2 次：QK^T 与 attn@V）。只翻这一个开关（诊断，非四臂）后
     同一模型实测变成 `bmm|in=bfloat16 ×7`、且峰值下降（见 `[DTYPE] ★SDPA 开关的显存效应`）。
  2. **bf16 在本 GPU 上不是"免费"的**：GEMM 微基准实测 bf16 比 fp32 **慢约 2×**（fp16 反而快约 2×），
     与 `is_bf16_supported()=True` 无关（那只说明 API 层可用）。⇒ 报「bf16 是否真在算 bf16」用两层证据：
     **dtype 直方图**（算子在算 bf16）+ **耗时**（是否被内部转换/模拟拖慢）。
  3. `[MEM]` 行并列报**峰值分配**与**峰值预留**（§12.10(a)：本机 8G 的真正约束是分配器预留）。
"""
from __future__ import annotations

import argparse
import contextlib
import gc
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
from torch.utils._python_dispatch import TorchDispatchMode  # noqa: E402

# ---------------- 常量（全部沿用 79 = 与 r3b′/r4 同口径） ----------------
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
PROBE_QUOTED_80 = "6f0df644384b6422"     # §12.8 的四臂探测批指纹（本文件应与它相同 = 同一探测批）

ARMS = ("J-FP32", "J-AMP-bf16", "J-AMP-fp16", "J+GC-AMP")
ARM_SLUG = {"J-FP32": "JFP32", "J-AMP-bf16": "Jbf16", "J-AMP-fp16": "Jfp16", "J+GC-AMP": "JGCAMP"}
ARM_AMP_NAME = {"J-FP32": "fp32", "J-AMP-bf16": "bf16", "J-AMP-fp16": "fp16", "J+GC-AMP": None}
JOINT_CARDS = ("input", "select", "realign", "output")
CARD_UNITS = {"input": ("input",), "select": ("select", "q"),
              "realign": ("realign",), "output": ("output",)}

# 探针（正式）：预热 5 步 + 2 个窗口 × 10 步（都在 3000 步预算内）；--smoke 同 80 缩短
PW, PS, PR = 5, 10, 2
PW_S, PS_S, PR_S = 1, 3, 1
PROBE_PERM_SEED = 0x52_34_08_20   # ★定值：四臂的探测批完全相同（与 80 同值）

# dtype 名 → torch dtype 对象
DTYPE_OBJ = {"fp32": None, "bf16": torch.bfloat16, "fp16": torch.float16}

WATCH_OPS = ("aten::addmm", "aten::mm", "aten::bmm", "aten::matmul", "aten::linear",
             "aten::native_layer_norm", "aten::layer_norm", "aten::softmax", "aten::_softmax",
             "aten::gelu", "aten::conv1d")
GEMMISH = ("addmm", "mm", "bmm", "matmul", "linear", "scaled_dot_product", "_scaled")

device = "cpu"


# ---------------- 数据 / 探测批（逐字沿用 80） ----------------
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


# ---------------- 数值口径（★ dtype 是声明变量） ----------------
def arm_cfg(arm: str, amp_dtype: str = "bf16") -> dict:
    """把「臂名」翻译成**声明**的数值口径：dtype / scaler / GC。

    `J+GC-AMP` 的 dtype 由 `--amp-dtype` 指定（较优的那档），其余臂写死。
    """
    if arm == "J-FP32":
        name, gc = "fp32", False
    elif arm == "J-AMP-bf16":
        name, gc = "bf16", False
    elif arm == "J-AMP-fp16":
        name, gc = "fp16", False          # ★fp16 只需 GradScaler，**不**开 GC（GC 只属于 J+GC-AMP）
    elif arm == "J+GC-AMP":
        name, gc = amp_dtype, True
    else:
        raise SystemExit(f"未知臂 {arm!r}")
    if name not in DTYPE_OBJ:
        raise SystemExit(f"未知 amp dtype {name!r}")
    return {"arm": arm, "dtype_name": name, "dtype": DTYPE_OBJ[name],
            "scaler": (name == "fp16"), "gc": gc}


def amp_ctx(cfg: dict):
    if cfg["dtype"] is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=cfg["dtype"])


def cfg_line(cfg: dict) -> str:
    if cfg["dtype"] is None:
        amp_desc = "无 autocast，全程 float32"
    else:
        amp_desc = "torch.autocast(device_type='cuda', dtype=" + cfg["dtype_name"] + ")"
    return (f"arm={cfg['arm']} 声明 dtype={cfg['dtype_name']}（{amp_desc}）"
            f" | GradScaler={'开' if cfg['scaler'] else '关'}"
            f" | 梯度检查点={'开（use_reentrant=False）' if cfg['gc'] else '关'}")


# ---------------- 模型：四臂共用 79 的 B2 两卡结构（逐字照抄 80 的 ArmJoint） ----------------
def unit_params(model, card: str):
    out = []
    for nm in CARD_UNITS[card]:
        out += [model.q] if nm == "q" else list(getattr(model, nm).parameters())
    return out


class ArmJoint(m79.ArmB):
    """同一个两卡结构（= 79 的 B2）；本类只多 `use_gc` 开关 + `last_dtypes`（[DTYPE] 的仪器）。

    `use_gc=True` 时 `forward_all` 逐字照抄 `79.ArmB.forward_all`，只把三次卡调用（input / select /
    realign）包进 `torch.utils.checkpoint.checkpoint(..., use_reentrant=False)`（eval 模式自动退回普通前向）。
    """

    def __init__(self, use_gc: bool = False, name: str = "J"):
        super().__init__("B2")
        self.use_gc = bool(use_gc)
        self.name = name
        self.last_dtypes = {}

    def forward_all(self, ids, ctx_len=None, end_idx=None, n_cands=None):
        """逐字照抄 `79.ArmB.forward_all`；`training and use_gc` 时把三处卡调用包进 checkpoint。

        ★与非 GC 分支逐行等价（→ 同一 RNG 消耗、同一梯度），只多把中间张量 dtype 记进 `last_dtypes`
        （[DTYPE] 的仪器用；GC 与非 GC 都要能读到）。
        """
        assert ctx_len is not None and end_idx is not None and n_cands is not None, \
            "[B 臂] 必须给 ctx_len / end_idx / n_cands"
        n = ids.size(1)
        m = self.input.causal_mask(ids)
        gc_on = bool(self.training and self.use_gc)
        if gc_on:
            hi = _ckpt(self.input, ids, use_reentrant=False, mask=m)
            hs = _ckpt(self.select, hi, use_reentrant=False, mask=m)
        else:
            hi = self.input(ids, mask=m)
            hs = self.select(hi, mask=m)
        g = torch.gather(hs, 1, end_idx.unsqueeze(-1).expand(-1, -1, hs.size(-1)))
        sc = (g * self.q).sum(-1)
        k = sc.size(1)
        valid = torch.arange(k, device=sc.device)[None, :] < n_cands[:, None]
        sc = sc.masked_fill(~valid, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        e_sel = torch.einsum("bk,bkd->bd", p, g)
        pos = torch.arange(n, device=ids.device)[None, :]
        apply = (pos >= ctx_len[:, None]).unsqueeze(-1).to(hi.dtype)
        if gc_on:
            h = _ckpt(self.realign, hi + apply * e_sel.unsqueeze(1), use_reentrant=False, mask=m)
        else:
            h = self.realign(hi + apply * e_sel.unsqueeze(1), mask=m)
        logits = self.output(h)
        self.last_dtypes = {"gc": gc_on, "mask": str(m.dtype), "hi": str(hi.dtype), "hs": str(hs.dtype),
                            "g": str(g.dtype), "sc": str(sc.dtype), "e_sel": str(e_sel.dtype),
                            "h": str(h.dtype), "logits": str(logits.dtype), "q": str(self.q.dtype)}
        return logits, sc


def build_arm(arm: str, amp_dtype: str = "bf16") -> ArmJoint:
    return ArmJoint(use_gc=arm_cfg(arm, amp_dtype)["gc"], name=arm)


# ---------------- 训练计划（四臂同为 1 段 × steps，全参数） ----------------
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
    pw, ps, pr = (PW_S, PS_S, PR_S) if smoke else (PW, PS, PR)
    return [{"name": f"{ARM_SLUG[arm]}#1", "steps": total, "cards": JOINT_CARDS,
             "windows": _windows(total, pw, ps, pr, late=True)}]


def plan_desc(plan: list) -> str:
    tot = sum(s["steps"] for s in plan)
    return f"1 段 × {tot} 步（全参数，单一 AdamW）"


def set_active(model, cards) -> list:
    """只让 `cards` 里的卡可训（联合档 = 全部卡；逻辑照抄 80，保持口径一致）。"""
    for p in model.parameters():
        p.grad = None
    active = []
    for c in cards:
        active += unit_params(model, c)
    ids = {id(p) for p in active}
    for p in model.parameters():
        p.requires_grad_(id(p) in ids)
    return active


# ---------------- 真实训练步（含 AMP / scaler / 非有限计数） ----------------
def new_counters():
    return {"attempts": 0, "opt_steps_done": 0, "skipped": 0, "scale_down": 0, "scale_up": 0,
            "scale_init": None, "scale_final": None, "scale_min": None,
            "inf_loss_steps": 0, "grad_nf_probe": 0.0, "grad_max_probe": 0.0,
            "_nf_t": None}


def opt_step_count(opt) -> int:
    """★实测「优化器真正走了几步」：读 AdamW 的 `state[p]['step']`（被跳过的步不会让它 +1）。

    不用「包装 opt.step」：那会形成 `opt → 包装函数 → opt` 的引用环，把**已死模型**的
    参数+优化器态拖到 gc 之前才释放 —— 实测会让同会话里**后跑的配置**驻留与峰值凭空 +83 MiB。
    """
    n = 0
    for _p, v in opt.state.items():
        if isinstance(v, dict) and "step" in v:
            st = v["step"]
            n = max(n, int(st.item()) if hasattr(st, "item") else int(st))
    return n


def train_step(model, batch, packed, opt, cfg, scaler, counters):
    """一步真实训练：forward（在声明的 autocast 内）→ backward → （scaler.）step。

    非有限计数用**设备端**累加（不逐步入 `.item()`，避免把每步墙钟拖慢）。
    """
    ids, s, eidx, nc = packed if packed is not None else pack(batch)
    with amp_ctx(cfg):
        logits, sc = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)
        lce = m79.reply_ce(logits, ids, batch, s)
        gold = torch.tensor([r["gold_idx"] for r in batch], device=ids.device)
        lsel = F.cross_entropy(sc, gold)
        loss = lce + SEL_LAMBDA * lsel
    opt.zero_grad(set_to_none=True)
    nf = (~torch.isfinite(loss.detach())).to(torch.float32).sum()
    counters["_nf_t"] = nf if counters["_nf_t"] is None else counters["_nf_t"] + nf
    counters["attempts"] += 1
    if scaler is not None:
        scaler.scale(loss).backward()
        prev = float(scaler.get_scale())
        scaler.step(opt)
        scaler.update()
        post = float(scaler.get_scale())
        if counters["scale_init"] is None:
            counters["scale_init"] = prev
        counters["scale_min"] = post if counters["scale_min"] is None else min(counters["scale_min"], post)
        counters["scale_final"] = post
        if post < prev:
            counters["scale_down"] += 1
        elif post > prev:
            counters["scale_up"] += 1
    else:
        loss.backward()
        opt.step()
    return loss.detach(), lce.detach(), lsel.detach(), sc


def grad_stats(model):
    """只在探针窗口/段末调用（全参数扫一遍，不进每步）：返回 (非有限元素数, max|g|, 元素数)。"""
    nf, mx, tot = 0.0, 0.0, 0
    for p in model.parameters():
        if p.grad is None:
            continue
        g = p.grad.detach()
        nf += float((~torch.isfinite(g)).sum().item())
        if g.numel():
            mx = max(mx, float(g.abs().max().item()))
        tot += g.numel()
    return nf, mx, tot


def gate_line(model, arm, seg, cards, opt, n_tot, sc, batch):
    """[GATE]：活动卡必须有梯度、冻结卡必须 0 梯度、优化器状态只覆盖活动卡（逐条实测）。"""
    parts, ok = [], True
    for c in JOINT_CARDS:
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
          + f" | 优化器状态覆盖参数={n_state}/{n_act}（全部参数 {n_tot}）"
          + f" | batch 选择准确率={100.0 * sum(hits) / len(hits):.2f}% ⇒ {'PASS' if ok else '★CHECK'}", flush=True)
    return ok


def train_arm(model, arm, train, seed, plan, probe_recs, probe_packed, cfg, tag=""):
    """按 plan 训练；探针窗口内用固定探测批跑真实训练步并读峰值显存。

    返回 (mem_rows, wall, steps, gates, loss_first, loss_last, counters)。
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train), generator=gen)
    pos = 0
    n_tot = sum(p.numel() for p in model.parameters())
    counters = new_counters()
    mem_rows, gates = [], []
    loss_first = loss_last = None
    t0 = time.perf_counter()
    for seg in plan:
        cards = seg["cards"]
        active = set_active(model, cards)
        n_act = sum(p.numel() for p in active)
        opt = AdamW(active, lr=LR)
        scaler = torch.amp.GradScaler("cuda", enabled=True) if cfg["scaler"] else None
        # ★不包装 opt.step（引用环会把死模型拖住 ⇒ 峰值虚高）；跳过步数在段末用 state['step'] 反推
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
            loss, lce, lsel, sc = train_step(model, batch, packed, opt, cfg, scaler, counters)
            if st == 1:
                loss_first = (float(loss), float(lce), float(lsel))
            last = (float(loss), float(lce), float(lsel))
            if win is not None:
                gnf, gmx, _gt = grad_stats(model)
                counters["grad_nf_probe"] += gnf
                counters["grad_max_probe"] = max(counters["grad_max_probe"], gmx)
            if not gate_done and st == min(20, seg["steps"]):
                gate_done = True
                gates.append(gate_line(model, arm, seg, cards, opt, n_tot, sc, batch))
            if win is not None and st == win[1] and reset_win == win:
                torch.cuda.synchronize()
                dt = (time.perf_counter() - pt0) / (win[1] - win[0] + 1)
                peak = torch.cuda.max_memory_allocated() / MIB
                peak_res = torch.cuda.max_memory_reserved() / MIB
                lo, hi, mean, nb = probe_len_stats(probe_packed)
                mem_rows.append({"arm": arm, "seg": seg["name"], "cards": "+".join(cards),
                                 "win": f"{win[0]}-{win[1]}", "peak_mib": peak,
                                 "peak_reserved_mib": peak_res, "resident_mib": resident,
                                 "s_per_step": dt, "n_active": n_act, "n_total": n_tot,
                                 "dtype": cfg["dtype_name"], "scaler": cfg["scaler"],
                                 "gc": cfg["gc"], "probe_fp": probe_fp(probe_packed)})
                print(f"[MEM] arm={arm} 声明dtype={cfg['dtype_name']} gc={int(cfg['gc'])}"
                      f" seg={seg['name']} 探针步={win[0]}–{win[1]}{tag} | 四要素恒定: batch={BATCH}"
                      f" | 序列长度分布=固定探测批（跨臂逐位相同；pad 后长度 min/max/mean="
                      f"{lo}/{hi}/{mean:.0f}，{nb} 批，指纹={probe_fp(probe_packed)}）"
                      f" | maxlen={MAXLEN} | ★峰值分配={peak:.0f} MiB | 峰值预留={peak_res:.0f} MiB"
                      f" | 驻留(步前)={resident:.1f} MiB | 可训={n_act}/{n_tot}"
                      f" | 每步墙钟={dt:.3f}s", flush=True)
        seg_sec = time.perf_counter() - seg_t0
        counters["opt_steps_done"] += opt_step_count(opt)
        print(f"[SEG] arm={arm} seg={seg['name']} 卡={'+'.join(cards)} 步数={seg['steps']}"
              f" 可训={n_act}/{n_tot} 末步 loss={last[0]:.4f} ce={last[1]:.4f} sel={last[2]:.4f}"
              f" 段墙钟={seg_sec:.0f}s", flush=True)
        del opt                     # ★显式释放优化器态（不进引用环、不等 gc）
    loss_last = last
    gnf, gmx, _gt = grad_stats(model)
    counters["grad_nf_final"] = gnf
    counters["grad_max_final"] = gmx
    return mem_rows, time.perf_counter() - t0, sum(s["steps"] for s in plan), gates, loss_first, loss_last, counters


# ---------------- [DTYPE] 仪器：实测参与计算的 dtype ----------------
def _flat_tensors(x, out=None):
    if out is None:
        out = []
    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for y in x:
            _flat_tensors(y, out)
    elif isinstance(x, dict):
        for y in x.values():
            _flat_tensors(y, out)
    return out


def _op_name(func) -> str:
    """torch 2.9 下 `str(OpOverload)` = `aten.mm.default` ⇒ 归一成 `aten::mm`。"""
    s = str(func)
    if s.startswith("aten."):
        return "aten::" + s.split(".")[1]
    return s


def _dt_name(t) -> str:
    return str(t.dtype).split(".")[-1]


class DtypeWatch(TorchDispatchMode):
    """在一次真实 forward+backward 上，逐算子记录**内核实收**的输入/输出 dtype 直方图。

    autocast 的 cast 挂在 Autocast key（优先级高于 Python/TorchDispatchMode key）⇒ 本模式看到的是
    算子**实际执行**用的 dtype。若某算子在此处仍是 float32 而声明是 bf16/fp16，那就是回退/未被 cast。
    """

    def __init__(self):
        self.rows = {}

    def _hit(self, func, args, kwargs):
        name = _op_name(func)
        if not (name in WATCH_OPS or "scaled_dot_product" in name):
            return func(*args, **kwargs)          # ★未观测的算子必须原样调用（否则返回 None）
        ins = "/".join(sorted({_dt_name(t) for t in _flat_tensors(args)}))
        out = func(*args, **kwargs)
        outs = "/".join(sorted({_dt_name(t) for t in _flat_tensors(out)}))
        key = f"{name}|in={ins}|out={outs}"
        self.rows[key] = self.rows.get(key, 0) + 1
        return out

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return self._hit(func, args, kwargs or {})


def watch_step(model, batch, cfg):
    """用 DtypeWatch 包一次真实 forward+backward（不 step，不改权重；只为读 dtype）。"""
    ids, s, eidx, nc = pack(batch)
    w = DtypeWatch()
    with w, amp_ctx(cfg):
        logits, sc = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)
        lce = m79.reply_ce(logits, ids, batch, s)
        gold = torch.tensor([r["gold_idx"] for r in batch], device=ids.device)
        loss = lce + SEL_LAMBDA * F.cross_entropy(sc, gold)
    loss.backward()
    model.zero_grad(set_to_none=True)
    return w.rows, model.last_dtypes, float(loss.detach())


def dtype_verdict(rows: dict, declared: str) -> dict:
    """回退判定（只取 GEMM 类算子）：声明 bf16/fp16 时出现 float32 ⇒ 回退/未 cast，并点名是哪几个算子。"""
    gemm = {k: v for k, v in rows.items() if any(g in k.split("|")[0] for g in GEMMISH)}
    dts = {}
    fp32_ops = {}
    for k, v in gemm.items():
        ins = k.split("|in=")[1].split("|")[0]
        for d in ins.split("/"):
            dts[d] = dts.get(d, 0) + v
        if "float32" in ins.split("/"):
            fp32_ops[k] = v
    fallback = declared in ("bf16", "fp16") and dts.get("float32", 0) > 0
    return {"gemm_hist": dts, "fp32_gemm_ops": fp32_ops, "fallback": bool(fallback),
            "verdict": (("★有回退/未 cast：GEMM 类算子里出现 float32 —— " + "；".join(
                f"{k} ×{v}" for k, v in sorted(fp32_ops.items(), key=lambda kv: -kv[1])))
                        if fallback else
                        ("未发现回退：GEMM 类算子全部按声明 dtype 执行"
                         if declared in ("bf16", "fp16") else "对照档：声明即 float32"))}


def gemm_microbench(iters: int = 20) -> dict:
    """实测 GEMM 微基准：fp32 / bf16 / fp16 的同时长、输出 dtype、相对误差（回退/转换会慢于 fp32）。

    形状取两种：① 4096×4096×512（够大，看原生吞吐）② 模型实际规模 (32·512)×512×128。
    """
    out = {}
    for tag, (M, N, Kd) in {"big": (4096, 4096, 512), "model": (32 * 512, 512, 128)}.items():
        a = torch.randn(M, Kd, device=device)
        b = torch.randn(Kd, N, device=device)
        ref = a.double() @ b.double()
        den = float(ref.abs().max().item()) or 1.0
        rec = {}
        for nm, dt in (("fp32", None), ("bf16", torch.bfloat16), ("fp16", torch.float16)):
            x = a.to(dt) if dt is not None else a
            y = b.to(dt) if dt is not None else b
            ctx = torch.autocast(device_type="cuda", dtype=dt) if dt is not None else contextlib.nullcontext()
            for _ in range(2):
                with ctx:
                    _ = x @ y
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                with ctx:
                    z = x @ y
            torch.cuda.synchronize()
            ms = 1000.0 * (time.perf_counter() - t0) / iters
            rec[nm] = {"ms": ms, "out_dtype": str(z.dtype),
                       "rel_err": float(((z.float().double() - ref).abs().max().item()) / den)}
            del x, y, z
        del a, b, ref
        torch.cuda.empty_cache()
        out[tag] = rec
    return out


def gpu_env() -> dict:
    p = torch.cuda.get_device_properties(0)
    return {"torch": torch.__version__, "hip": torch.version.hip,
            "device": p.name, "gcn_arch": getattr(p, "gcnArchName", None),
            "capability": list(torch.cuda.get_device_capability(0)),
            "total_mem_gib": round(p.total_memory / 1024 ** 3, 2),
            "is_bf16_supported": bool(torch.cuda.is_bf16_supported()),
            "allow_bf16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction),
            "allow_fp16_reduced_precision_reduction": bool(torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction),
            "fp16_bf16_reduction_math_sdp_allowed": bool(torch.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()),
            "math_sdp_enabled": bool(torch.backends.cuda.math_sdp_enabled()),
            "flash_sdp_enabled": bool(torch.backends.cuda.flash_sdp_enabled()),
            "mem_efficient_sdp_enabled": bool(torch.backends.cuda.mem_efficient_sdp_enabled()),
            "HSA_OVERRIDE_GFX_VERSION": os.environ.get("HSA_OVERRIDE_GFX_VERSION")}


def print_env(env: dict):
    print(f"[DTYPE] 环境：torch={env['torch']} hip={env['hip']} device={env['device']} "
          f"gcnArchName={env['gcn_arch']} capability={env['capability']} VRAM={env['total_mem_gib']}GiB "
          f"HSA_OVERRIDE_GFX_VERSION={env['HSA_OVERRIDE_GFX_VERSION']}", flush=True)
    print(f"[DTYPE] 开关：is_bf16_supported={env['is_bf16_supported']} "
          f"allow_bf16_reduced_precision_reduction={env['allow_bf16_reduced_precision_reduction']} "
          f"allow_fp16_reduced_precision_reduction={env['allow_fp16_reduced_precision_reduction']} | "
          f"SDPA backend(math/flash/mem_eff)={env['math_sdp_enabled']}/{env['flash_sdp_enabled']}/"
          f"{env['mem_efficient_sdp_enabled']} | ★fp16_bf16_reduction_math_sdp_allowed="
          f"{env['fp16_bf16_reduction_math_sdp_allowed']}（False ⇒ math SDPA 会把 q/k/v **上转到 fp32**）",
          flush=True)


def sdp_flag_sensitivity(seed: int, train, ampmap) -> dict:
    """诊断（**不是四臂**）：同一 bf16/fp16 声明下，只翻 `allow_fp16_bf16_reduction_math_sdp` 一个开关，
    看注意力核的 dtype 与探测窗峰值怎么变 —— 用来定量「回退吃掉了多少 AMP 的收益」。"""
    out = {}
    batch = train[:BATCH]
    probe_recs, probe_packed = build_probe_batches(train, PS)
    for name in ampmap:
        cfg = {"arm": f"sdpflag-{name}", "dtype_name": name, "dtype": DTYPE_OBJ[name],
               "scaler": False, "gc": False}
        for flag in (False, True):
            torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(flag)
            torch.manual_seed(seed)
            model = ArmJoint(use_gc=False, name=f"sdpflag-{name}").to(device)
            rows, last_dt, loss = watch_step(model, batch, cfg)
            v = dtype_verdict(rows, name)
            plan = [{"name": f"probe-flag{int(flag)}", "steps": PW + PR * PS + 5, "cards": JOINT_CARDS,
                     "windows": _windows(PW + PR * PS + 5, PW, PS, PR, late=False)}]
            mem_rows, wall, _n, _g, _lf, _ll, _c = train_arm(
                model, cfg["arm"], train, seed, plan, probe_recs, probe_packed, cfg, tag="（诊断）")
            peak = max(r["peak_mib"] for r in mem_rows)
            out[f"{name}|flag={int(flag)}"] = {"peak_mib": peak, "s_per_step": mem_rows[0]["s_per_step"],
                                               "gemm_hist": v["gemm_hist"], "fallback": v["fallback"],
                                               "verdict": v["verdict"], "last_dtypes": last_dt, "loss": loss}
            print(f"[DTYPE] ★SDPA 开关敏感性（诊断，非四臂）声明={name} "
                  f"allow_fp16_bf16_reduction_math_sdp={flag}：峰值={peak:.0f} MiB "
                  f"每步={mem_rows[0]['s_per_step']:.3f}s | GEMM 类 dtype={v['gemm_hist']} | {v['verdict']}",
                  flush=True)
            del model
            gc.collect()
            torch.cuda.empty_cache()
        torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    if len(out) >= 2:
        for name in ampmap:
            a, b = out.get(f"{name}|flag=0"), out.get(f"{name}|flag=1")
            if a and b:
                print(f"[DTYPE] ★SDPA 开关的显存效应（实测）声明={name}：flag=0 峰值={a['peak_mib']:.0f} MiB "
                      f"→ flag=1 峰值={b['peak_mib']:.0f} MiB ⇒ {b['peak_mib'] - a['peak_mib']:+.0f} MiB"
                      f"（每步 {a['s_per_step']:.3f}s → {b['s_per_step']:.3f}s）", flush=True)
    return out


def dtype_probe_all(seed: int, train) -> dict:
    """对三种 dtype 各做一次真实 forward+backward 的 dtype 实测 + GEMM 微基准。"""
    env = gpu_env()
    print_env(env)
    bench = gemm_microbench()
    print(f"[DTYPE] ★GEMM 微基准（实测，本机 GPU）：" + " | ".join(
        f"{tag}: " + " ".join(f"{nm}={v['ms']:.2f}ms(out={v['out_dtype']},rel_err={v['rel_err']:.2e})"
                              for nm, v in rec.items())
        for tag, rec in bench.items()), flush=True)
    print(f"[DTYPE] ★回退线索（推断，需与上面的 dtype 直方图合读）：big 档 bf16/fp32 耗时比="
          f"{bench['big']['bf16']['ms'] / bench['big']['fp32']['ms']:.2f}×、"
          f"fp16/fp32={bench['big']['fp16']['ms'] / bench['big']['fp32']['ms']:.2f}×"
          f"（≤1 表示原生支持；明显 >1 提示内部转换/回退，**这是耗时线索不是直接证据**）", flush=True)
    ev = {}
    batch = train[:BATCH]
    for name in ("fp32", "bf16", "fp16"):
        cfg = {"arm": f"probe-{name}", "dtype_name": name, "dtype": DTYPE_OBJ[name],
               "scaler": False, "gc": False}
        torch.manual_seed(seed)
        model = ArmJoint(use_gc=False, name=f"probe-{name}").to(device)
        rows, last_dt, loss = watch_step(model, batch, cfg)
        v = dtype_verdict(rows, name)
        ev[name] = {"rows": rows, "verdict": v, "last_dtypes": last_dt, "loss": loss}
        print(f"[DTYPE] 声明={name}：一次真实 forward+backward 的算子实收 dtype 直方图"
              f"（TorchDispatchMode，共 {len(rows)} 类）:", flush=True)
        for k, c in sorted(rows.items(), key=lambda kv: -kv[1])[:14]:
            print(f"[DTYPE]    {k} ×{c}", flush=True)
        print(f"[DTYPE] 声明={name}：中间张量实测 dtype {last_dt} | loss={loss:.4f}", flush=True)
        print(f"[DTYPE] 声明={name}：★回退判定（只看 GEMM 类算子）= {v['verdict']}；"
              f"GEMM 类实收 dtype 分布={v['gemm_hist']}", flush=True)
        del model
        torch.cuda.empty_cache()
    print(f"[DTYPE] ★下面这一节是**诊断**：只翻 `allow_fp16_bf16_reduction_math_sdp` 一个开关，"
          f"量它对注意力核 dtype 与峰值的作用（**四臂一律跑默认值 False**）", flush=True)
    sdp = sdp_flag_sensitivity(seed, train, ("bf16", "fp16"))
    return {"env": env, "bench": bench, "per_dtype": ev, "sdp_flag_sensitivity": sdp}


# ---------------- per-run 主流程 ----------------
def hits_path(arm: str, seed: int, smoke: bool = False) -> str:
    return os.path.join(LOG_DIR, f"82_hits_{ARM_SLUG[arm]}_s{seed}{'_smoke' if smoke else ''}.json")


def run_one(a) -> int:
    global device
    torch.manual_seed(a.seed)
    if not torch.cuda.is_available():
        print("GPU 起不来，立即停（不用 CPU 硬跑）。", flush=True)
        print("HSA_OVERRIDE_GFX_VERSION =", os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)
        return 3
    device = "cuda"
    m79.device = device
    cfg = arm_cfg(a.arm, a.amp_dtype)
    # ★四臂一律跑在**默认**开关下（RDNA2 实测：math SDPA 会把注意力核上转 fp32；这只在诊断里翻）
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    print("device:", device, torch.cuda.get_device_name(0), flush=True)
    print(f"[PLAN] {cfg_line(cfg)} | 评测口径={'FP32（默认）' if not a.eval_amp else '各臂自己的 dtype'}"
          f" | seed={a.seed} steps={'SMOKE' if a.smoke else a.steps} batch={BATCH} maxlen={MAXLEN}", flush=True)
    train, test = load_data()
    ev = test[: m79.SMOKE_EVAL] if a.smoke else test
    steps_total = m79.SMOKE_STEPS if a.smoke else a.steps
    plan = build_plan(a.arm, steps_total, a.smoke)
    smoke_note = "（--smoke 探路，不作判据）" if a.smoke else ""

    # ---- [DATA]（与 80 同口径的指纹核对，不作判据） ----
    fp_tr, fp_te = m79.d74.d72.data_fp(train), m79.d74.d72.data_fp(test)
    q = m79.FP_QUOTED.get(PREFIX)
    print(f"[DATA] prefix={PREFIX} k={K} arm={a.arm} seed={a.seed} train={len(train)} test={len(test)} "
          f"评测 n={len(ev)} | ctx tok 最大={max(len(r['p']) for r in train + test)} "
          f"p+t 最大={max(len(r['p']) + len(r['t']) for r in train + test)} 超 MAXLEN="
          f"{sum(1 for r in train + test if len(r['p']) + len(r['t']) > MAXLEN)} | data_fp "
          f"train={fp_tr} test={fp_te}"
          + (f" | §11.14 引文 {q[0]}/{q[1]} ⇒ {'表面一致' if (fp_tr == q[0] and fp_te == q[1]) else '★与引文不同（如实报）'}"
             if q else ""), flush=True)

    # ---- [DTYPE]（本 run 的数值口径 + 实测参与计算的 dtype） ----
    print(f"[DTYPE] {cfg_line(cfg)} | torch.is_autocast_enabled()={torch.is_autocast_enabled()}"
          f"（此处在 autocast 上下文**之外** ⇒ False 属正常；真实训练步在 amp_ctx 内跑）", flush=True)
    dtype_ev = None
    if a.dtype_probe:
        dtype_ev = dtype_probe_all(a.seed, train)
    else:
        print_env(gpu_env())
        # 轻量版：只为本臂做一次真实 forward+backward 的 dtype 实测（不跑微基准）
        torch.manual_seed(a.seed)
        pm = ArmJoint(use_gc=cfg["gc"], name=a.arm).to(device)
        rows, last_dt, ploss = watch_step(pm, train[:BATCH], cfg)
        v = dtype_verdict(rows, cfg["dtype_name"])
        dtype_ev = {"env": gpu_env(), "per_dtype": {cfg["dtype_name"]: {"rows": rows, "verdict": v,
                                                                      "last_dtypes": last_dt, "loss": ploss}}}
        print(f"[DTYPE] 本臂一次真实 forward+backward 的算子实收 dtype 直方图（TorchDispatchMode，"
              f"共 {len(rows)} 类）:", flush=True)
        for k, c in sorted(rows.items(), key=lambda kv: -kv[1])[:14]:
            print(f"[DTYPE]    {k} ×{c}", flush=True)
        print(f"[DTYPE] 中间张量实测 dtype {last_dt} | loss={ploss:.4f} | ★回退判定={v['verdict']}；"
              f"GEMM 类实收 dtype 分布={v['gemm_hist']}", flush=True)
        del pm
        torch.cuda.empty_cache()

    # ---- [INIT-CHECK]（§9.7：manual_seed 先于构造；四臂同 seed 逐张量核） ----
    print("[INIT-CHECK] §9.7 init 缺陷修复：manual_seed 先于构造（四臂在构造前统一播种）", flush=True)
    built = {}
    for arm in ARMS:
        torch.manual_seed(a.seed)
        built[arm] = build_arm(arm, a.amp_dtype)
    torch.manual_seed(a.seed)
    model = build_arm(a.arm, a.amp_dtype)
    sj = built["J-FP32"].state_dict()
    worst_cross = 0.0
    for arm in ARMS:
        d = max(float((sj[k] - built[arm].state_dict()[k]).abs().max().item()) for k in sj)
        worst_cross = max(worst_cross, d)
        print(f"[INIT-CHECK] 四臂同 seed={a.seed} J-FP32 vs {arm} 逐张量 max|Δ|={d:.3e}"
              f"（{len(sj)} 个张量，结构逐位相同；参数全为 float32 —— AMP 不改参数 dtype）", flush=True)
    h1 = m79.state_sha256(model.state_dict())
    torch.manual_seed(a.seed)
    h2 = m79.state_sha256(build_arm(a.arm, a.amp_dtype).state_dict())
    torch.manual_seed(a.seed ^ 0x5EED)
    h3 = m79.state_sha256(build_arm(a.arm, a.amp_dtype).state_dict())
    print(f"[INIT-CHECK] arm={a.arm} 同 seed 两次构造 sha={h1[:16]}/{h2[:16]} ⇒ {'相同' if h1 == h2 else '★不同'}"
          f" | 异 seed sha={h3[:16]} ⇒ {'不同（阴性对照通过）' if h3 != h1 else '★相同（seed 没生效）'}", flush=True)
    model = model.to(device)
    dtypes_params = sorted({str(p.dtype) for p in model.parameters()})
    print(f"[INIT-CHECK] arm={a.arm} 参数 dtype 集合={dtypes_params} ⇒ "
          f"{'全 float32（AMP 的主权重口径）' if dtypes_params == ['torch.float32'] else '★非全 float32（如实报）'}",
          flush=True)

    # ---- [PARAMS] ----
    br = m79.param_groups_named(model)
    n_tot = br["合计"]
    biggest = max(JOINT_CARDS, key=lambda c: sum(p.numel() for p in unit_params(model, c)))
    print(f"[PARAMS] arm={a.arm} 全部参数={n_tot} | 分项 " + " ".join(f"{k}={v}" for k, v in br.items() if k != "合计")
          + f" | 单一 AdamW 内可训={n_tot} | 总步数={steps_total}×batch{BATCH}={steps_total * BATCH} 样本"
          f" lr={LR} SEL_LAMBDA={SEL_LAMBDA} | dtype={cfg['dtype_name']} scaler={cfg['scaler']} gc={cfg['gc']}",
          flush=True)
    print(f"[PARAMS] arm={a.arm} 步数分配={plan_desc(plan)} | 段数={len(plan)}"
          f" 逐段=" + " ".join(f"{s['name']}:{s['steps']}" for s in plan), flush=True)

    # ---- 探针批 + 训练 ----
    probe_recs, probe_packed = build_probe_batches(train, PS if not a.smoke else PS_S)
    lo, hi, mean, nb = probe_len_stats(probe_packed)
    pfp = probe_fp(probe_packed)
    print(f"[MEM] 探测批（四臂共用，指纹={pfp}"
          + (f"；与 §12.8 的 {PROBE_QUOTED_80} {'相同 ⇒ 与 r4 同一探测批' if pfp == PROBE_QUOTED_80 else '★不同（如实报）'}"
             if not a.smoke else "；--smoke 另起置换，不作判据")
          + f"）：{nb} 批×batch{BATCH}，pad 后长度 min/max/mean={lo}/{hi}/{mean:.0f}，maxlen={MAXLEN}"
          f" ⇒ 四要素恒定（dtype 是声明变量）", flush=True)
    torch.cuda.reset_peak_memory_stats()
    mem_rows, wall, done_steps, gates, loss_first, loss_last, counters = train_arm(
        model, a.arm, train, a.seed, plan, probe_recs, probe_packed, cfg, tag=smoke_note)

    # ---- 评测（T2 完整回复 EM + T1 字母准确率；默认 FP32 评测口径） ----
    eval_cfg = cfg if a.eval_amp else {"arm": cfg["arm"], "dtype_name": "fp32", "dtype": None,
                                       "scaler": False, "gc": False}
    t_eval = time.perf_counter()
    with amp_ctx(eval_cfg):
        ce = m79.eval_ce(model, ev)
        cm, cse = m79.mean_se(ce)
        sel = m79.eval_sel(model, ev)
        txt = m79.greedy_gen(model, ev)
    hits = [m79.em_strict(r["target"], txt[i]) for i, r in enumerate(ev)]
    em = 100.0 * sum(hits) / len(hits)
    se, _thr = m79.em_with_thr(em, len(hits))
    sel_acc = None if sel is None else 100.0 * sum(sel["hits"]) / len(sel["hits"])
    ev_rep = m79.report_texts(hits, txt, ev, "T1T2")
    t1, miss = ev_rep["t1"], ev_rep["miss"]
    eval_sec = time.perf_counter() - t_eval
    t1thr = m79.t1_threshold(K, len(ev))[2]
    t2thr = m79.t2_threshold(m79.MAX_NAIVE_QUOTED[K], len(ev))[2]
    ck = os.path.join(LOG_DIR, f"82_ckpt_{ARM_SLUG[a.arm]}_s{a.seed}{'_smoke' if a.smoke else ''}.pt")
    torch.save(model.state_dict(), ck)

    # ---- scaler / 非有限 的实测汇总 ----
    attempts = counters["attempts"]
    skipped = attempts - counters["opt_steps_done"]
    nf_loss_steps = int(counters["_nf_t"].item()) if counters["_nf_t"] is not None else 0
    scal = {"scaler": cfg["scaler"], "attempts": attempts,
            "opt_step_called": counters["opt_steps_done"], "skipped": skipped,
            "skipped_frac": (skipped / attempts) if attempts else float("nan"),
            "scale_init": counters["scale_init"], "scale_final": counters["scale_final"],
            "scale_min": counters["scale_min"], "scale_down": counters["scale_down"],
            "scale_up": counters["scale_up"], "inf_loss_steps": nf_loss_steps,
            "grad_nf_in_probe": counters["grad_nf_probe"],
            "grad_nf_final": counters.get("grad_nf_final", 0.0),
            "grad_max_in_probe": counters["grad_max_probe"],
            "grad_max_final": counters.get("grad_max_final", 0.0)}
    if cfg["scaler"]:
        usable = (skipped < attempts) and (counters.get("grad_nf_final", 0.0) == 0.0)
        why = ("全步被跳过（梯度一直溢出）" if skipped >= attempts else
               ("末态有非有限梯度" if counters.get("grad_nf_final", 0.0) else ""))
    else:
        usable = (nf_loss_steps == 0) and (counters.get("grad_nf_final", 0.0) == 0.0)
        why = ("出现非有限 loss/梯度" if not usable else "")
    print(f"[NUM] arm={a.arm} dtype={cfg['dtype_name']} scaler={'开' if cfg['scaler'] else '关'}"
          f" | loss 首步={loss_first[0]:.4f}(ce={loss_first[1]:.4f},sel={loss_first[2]:.4f})"
          f" 末步={loss_last[0]:.4f}(ce={loss_last[1]:.4f},sel={loss_last[2]:.4f}) nats"
          f" | 非有限 loss 步数={nf_loss_steps}/{attempts}"
          f" | 梯度非有限元素（探针窗口+末态）={counters['grad_nf_probe']:.0f}/"
          f"{counters.get('grad_nf_final', 0.0):.0f} max|g|={counters['grad_max_probe']:.3e}"
          f" | ★可用性={'可用' if usable else '★不可用：' + why}", flush=True)
    if cfg["scaler"]:
        print(f"[SCALER] arm={a.arm} fp16+GradScaler 实测：scale 初值={scal['scale_init']:.0f}"
              f" 末值={scal['scale_final']:.0f} 最小={scal['scale_min']:.0f}"
              f" | 缩放变化次数：降={scal['scale_down']} 升={scal['scale_up']}"
              f" | ★被跳过的步数（AdamW 实走 {scal['opt_step_called']}/{attempts} 步）="
              f"{skipped}（{100.0 * scal['skipped_frac']:.2f}%）"
              f" | 溢出触发（loss 非有限）步数={nf_loss_steps}", flush=True)
    else:
        print(f"[SCALER] arm={a.arm} 无 scaler（{'fp32 对照' if cfg['dtype_name'] == 'fp32' else 'bf16 按 §15 不加 scaler'}）"
              f" | 无 scaler ⇒ 「跳过步数」不适用（但实测 AdamW 实走 {scal['opt_step_called']}/{attempts} 步，"
              f"若两者不等即为仪器异常）"
              f" | 非有限 loss 步数={nf_loss_steps}/{attempts}", flush=True)

    # ---- [MEM] 臂汇总（含 T2/T1） ----
    peak = max(r["peak_mib"] for r in mem_rows) if mem_rows else float("nan")
    peak_res = max(r["peak_reserved_mib"] for r in mem_rows) if mem_rows else float("nan")
    peak_row = max(mem_rows, key=lambda r: r["peak_mib"]) if mem_rows else {}
    per_step = (sum(r["s_per_step"] for r in mem_rows) / len(mem_rows)) if mem_rows else float("nan")
    print(f"[MEM] arm={a.arm} 声明dtype={cfg['dtype_name']} ★峰值={peak:.0f} MiB（预留 {peak_res:.0f} MiB）"
          f"（出现于 seg={peak_row.get('seg')} 窗={peak_row.get('win')}）"
          f" | 全部参数={n_tot} | 每步墙钟={per_step:.3f}s（探针窗口均值）| T2={em:.2f}% | T1={t1:.2f}%"
          f" | 训练墙钟={wall:.0f}s{smoke_note}", flush=True)
    print(f"[MAIN] arm={a.arm} steps={done_steps} batch={BATCH} lr={LR} d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN}"
          f" | 配对CE={cm:.4f}±{cse:.4f} (nats/token) | 选择准确率="
          + ("不适用" if sel_acc is None else f"{sel_acc:.2f}%")
          + f" | 评估墙钟={eval_sec:.0f}s | 评测口径={eval_cfg['dtype_name']}"
          f" | GC={'开' if cfg['gc'] else '关'} | scaler={'开' if cfg['scaler'] else '关'}", flush=True)

    # ---- 本 run 的 [CRIT] 分条（跨臂判定在 --report-r6） ----
    print(f"[CRIT] R6-a（本 run 部分）arm={a.arm} dtype={cfg['dtype_name']} 峰值={peak:.0f} MiB"
          f"（预留 {peak_res:.0f} MiB）| 逐窗口峰值="
          + " ".join(f"{r['win']}={r['peak_mib']:.0f}" for r in mem_rows)
          + " | ★vs J-FP32 的判定见 --report-r6", flush=True)
    print(f"[CRIT] R6-b（本 run 部分）arm={a.arm} T2={em:.2f}% ±{se:.2f}pp（门槛 {t2thr:.2f}%）| "
          f"T1={t1:.2f}%（抽出失败 {100.0 * miss / len(ev):.2f}%，门槛 {t1thr:.2f}%）| "
          f"选择准确率={sel_acc} | ★配对 Δ/SE 见 --report-r6", flush=True)
    print(f"[CRIT] R6-c（本 run 部分）arm={a.arm} GC={'开' if cfg['gc'] else '关'} 峰值={peak:.0f} MiB"
          f" | ★AMP×GC 的叠加性见 --probe-cross-amp（实测四项）", flush=True)
    print(f"[CRIT] R6-d（本 run 部分）arm={a.arm} loss 末步={loss_last[0]:.4f} nats | scaler="
          f"{'开' if cfg['scaler'] else '关'} 跳过={skipped}/{attempts} 非有限loss步数={nf_loss_steps}"
          f" | ★可用性={'可用' if usable else '不可用：' + why}", flush=True)
    print(f"[CRIT] R6-e（本 run 部分）arm={a.arm} 峰值={peak:.0f} MiB 每步={per_step:.3f}s 参数={n_tot}"
          f" T1={t1:.2f}% T2={em:.2f}% 训练墙钟={wall:.0f}s", flush=True)

    hp = hits_path(a.arm, a.seed, a.smoke)
    with open(hp, "w", encoding="utf-8") as f:
        json.dump({"arm": a.arm, "seed": a.seed, "k": K, "prefix": PREFIX, "smoke": a.smoke,
                   "dtype_declared": cfg["dtype_name"], "amp_dtype": cfg["dtype_name"],
                   "scaler": cfg["scaler"], "gc": cfg["gc"], "eval_mode": eval_cfg["dtype_name"],
                   "batch": BATCH, "maxlen": MAXLEN,
                   "n": len(hits), "em": em, "se": se, "t1": t1, "miss": miss, "sel": sel_acc,
                   "params": n_tot, "steps": done_steps, "peak_mib": peak,
                   "peak_reserved_mib": peak_res, "wall_s": wall, "eval_s": eval_sec,
                   "s_per_step": per_step, "t1_thr": t1thr, "t2_thr": t2thr,
                   "loss_first": loss_first, "loss_last": loss_last,
                   "scaler_stats": scal, "usable": bool(usable), "unusable_why": why,
                   "init_ok": worst_cross == 0.0, "init_worst": worst_cross, "gates_ok": all(gates),
                   "probe_fp": pfp, "probe_fp_quoted_80": PROBE_QUOTED_80, "mem_rows": mem_rows,
                   "plan": [{"name": s["name"], "steps": s["steps"], "cards": list(s["cards"])} for s in plan],
                   "dtype_evidence": dtype_ev,
                   "hits": hits, "txt": txt, "letters": ev_rep["ls"], "gold_letters": ev_rep["gold_l"],
                   "ckpt": ck}, f)
    print(f"[META] arm={a.arm} seed={a.seed} smoke={a.smoke} steps={done_steps} params={n_tot} "
          f"dtype={cfg['dtype_name']} scaler={cfg['scaler']} gc={cfg['gc']} peak_mib={peak:.0f} "
          f"T2={em:.2f}% T1={t1:.2f}% s_per_step={per_step:.3f} init_ok={worst_cross == 0.0} "
          f"gates_ok={all(gates)} usable={usable} probe_fp={pfp} ckpt={ck} hits={hp} "
          f"训练墙钟={wall:.0f}s 总墙钟={wall + eval_sec:.0f}s", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


# ---------------- 诊断：AMP × 检查点 的峰值叠加性（实测四项） ----------------
def probe_cross_amp(a) -> int:
    global device
    if not torch.cuda.is_available():
        print("GPU 起不来，立即停。", flush=True)
        return 3
    device = "cuda"
    m79.device = device
    torch.backends.cuda.allow_fp16_bf16_reduction_math_sdp(False)
    print("device:", device, torch.cuda.get_device_name(0), flush=True)
    train, _test = load_data()
    probe_recs, probe_packed = build_probe_batches(train, PS)
    n_steps = PW + PR * PS + 5
    amp_name = a.amp_dtype
    print(f"[MEM-X] AMP×GC 叠加性（**实测**，§15 R6-c）：全新构造模型、每配置只跑 {n_steps} 步（含探针），"
          f"探测批与四臂逐位相同（指纹={probe_fp(probe_packed)}）⇒ 只比显存画像；"
          f"较优 AMP 档（--amp-dtype）={amp_name}", flush=True)
    cfgs = [("J-FP32", "fp32", False, False), (f"J-AMP-{amp_name}", amp_name, False, False),
            ("J+GC-FP32", "fp32", True, False), (f"J+GC-AMP-{amp_name}", amp_name, True, False)]
    res = {}
    for name, dname, use_gc, scaler in cfgs:
        cfg = {"arm": name, "dtype_name": dname, "dtype": DTYPE_OBJ[dname],
               "scaler": (dname == "fp16"), "gc": use_gc}
        torch.manual_seed(a.seed)
        model = ArmJoint(use_gc=use_gc, name=name).to(device)
        plan = [{"name": name, "steps": n_steps, "cards": JOINT_CARDS,
                 "windows": _windows(n_steps, PW, PS, PR, late=False)}]
        rows, _wall, _n, _g, _lf, _ll, _c = train_arm(model, name, train, a.seed, plan,
                                                      probe_recs, probe_packed, cfg, tag="（诊断）")
        res[name] = max(rows, key=lambda r: r["peak_mib"])
        print(f"[MEM-X]   {name}：驻留(步前)={res[name]['resident_mib']:.1f} MiB "
              f"峰值={res[name]['peak_mib']:.0f} MiB（★下一步配置起跑前的驻留应回到同一水位，"
              f"否则说明上一配置有活引用残留 → 峰值虚高）", flush=True)
        del model
        gc.collect()                      # ★断开引用环：不 collect 会把死模型拖到 gc 才释放 ⇒ +83 MiB 虚高
        torch.cuda.empty_cache()
    base = res["J-FP32"]["peak_mib"]
    d_amp = res[f"J-AMP-{amp_name}"]["peak_mib"] - base
    d_gc = res["J+GC-FP32"]["peak_mib"] - base
    d_both = res[f"J+GC-AMP-{amp_name}"]["peak_mib"] - base
    diff = d_both - (d_amp + d_gc)
    print(f"[MEM-X] 峰值（实测，同一次会话、同一探测批）：J-FP32={base:.0f} | "
          f"J-AMP-{amp_name}={res[f'J-AMP-{amp_name}']['peak_mib']:.0f} | "
          f"J+GC-FP32={res['J+GC-FP32']['peak_mib']:.0f} | "
          f"J+GC-AMP-{amp_name}={res[f'J+GC-AMP-{amp_name}']['peak_mib']:.0f} MiB", flush=True)
    print(f"[MEM-X] ★叠加性（相对 J-FP32）：Δ(仅 AMP)={d_amp:+.0f} MiB；Δ(仅 GC,FP32)={d_gc:+.0f} MiB；"
          f"两项之和={d_amp + d_gc:+.0f} MiB；**实测**两者={d_both:+.0f} MiB；差={diff:+.0f} MiB ⇒ "
          f"{'可叠加（差值在探针噪声内）' if abs(diff) <= 8.0 else '★非线性/不可简单叠加（如实报，且必须用实测值）'}",
          flush=True)
    p = os.path.join(LOG_DIR, f"82_probecross_s{a.seed}_{amp_name}.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"seed": a.seed, "amp_dtype": amp_name, "rows": res, "d_amp": d_amp, "d_gc": d_gc,
                   "d_both": d_both, "diff": diff, "probe_fp": probe_fp(probe_packed)}, f)
    print(f"[MEM-X] 落盘 {p}", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


# ---------------- [CRIT] R6-a..e（CPU，读四臂读数） ----------------
def load_hits(arm, seed, smoke=False):
    p = hits_path(arm, seed, smoke)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def within_spread(rows) -> float:
    by = {}
    for r in rows:
        by.setdefault(r["seg"], []).append(r["peak_mib"])
    return max((max(v) - min(v) for v in by.values()), default=0.0)


def r6_a(data, seed, mib_note=""):
    """AMP 档峰值 vs J-FP32（同 batch/seq/maxlen）。"""
    dj = data["J-FP32"]
    base = dj["peak_mib"]
    print(f"[CRIT] R6-a AMP 档峰值 vs J-FP32（同 batch={dj['batch']}/maxlen={dj['maxlen']}/"
          f"同一探测批 {dj['probe_fp']}；dtype 是声明变量）seed={seed}{mib_note}", flush=True)
    print(f"[CRIT] R6-a   J-FP32 峰值={base:.0f} MiB（窗口极差 {within_spread(dj['mem_rows']):.0f} MiB）", flush=True)
    out = {}
    for arm in ("J-AMP-bf16", "J-AMP-fp16", "J+GC-AMP"):
        d = data.get(arm)
        if d is None:
            print(f"[CRIT] R6-a   {arm}：未跑 ⇒ 未判（如实报）", flush=True)
            continue
        pct = 100.0 * (d["peak_mib"] - base) / base
        ok = pct <= -10.0
        out[arm] = pct
        print(f"[CRIT] R6-a   {arm}（dtype={d['dtype_declared']}，GC={int(d['gc'])}）：峰值={d['peak_mib']:.0f} MiB"
              f" ⇒ Δ={d['peak_mib'] - base:+.0f} MiB（{pct:+.2f}%）| 每步={d['s_per_step']:.3f}s"
              f" vs {dj['s_per_step']:.3f}s（×{d['s_per_step'] / dj['s_per_step']:.2f}）⇒ "
              f"{'★有实质收益（≥10%）' if ok else '本任务上 AMP 不显著省显存（<10%）'}"
              f"{'' if d['usable'] else '；★但该档不可用（' + d['unusable_why'] + '）'}", flush=True)
        if arm == "J+GC-AMP":
            print(f"[CRIT] R6-a   ★注意 J+GC-AMP 混了两个变量（AMP + 检查点）⇒ 它的降幅不是「AMP 单独的收益」，"
                  f"AMP 单独的收益只看 J-AMP-*（R6-c 用实测拆开）", flush=True)
    return out


def r6_b(data, seed, mib_note=""):
    """质量 Δ：配对 per-item Δ + 配对 SE（§11.4）+ McNemar。"""
    dj = data["J-FP32"]
    print(f"[CRIT] R6-b 质量 Δ（T2 完整回复 EM，per-item 配对；§11.4 口径）seed={seed}{mib_note}", flush=True)
    for arm in ("J-AMP-bf16", "J-AMP-fp16", "J+GC-AMP"):
        d = data.get(arm)
        if d is None:
            print(f"[CRIT] R6-b   {arm}：未跑 ⇒ 未判（如实报）", flush=True)
            continue
        st = m79.paired_stats(d["hits"], dj["hits"])          # Δ = arm − J-FP32
        lj = [m79.dec_letter(t) for t in dj["txt"]]
        la = [m79.dec_letter(t) for t in d["txt"]]
        gj, ga = dj["gold_letters"], d["gold_letters"]
        hlj = [int(x == g) for x, g in zip(lj, gj)]
        hla = [int(x == g) for x, g in zip(la, ga)]
        stl = m79.paired_stats(hla, hlj)
        both_sat = dj["em"] >= 95.0 and d["em"] >= 95.0
        sig = (abs(st["delta"]) >= 2 * st["se"]) and st["p"] < 0.05
        print(f"[CRIT] R6-b   {arm}：T2={d['em']:.2f}% vs J-FP32={dj['em']:.2f}% ⇒ Δ={st['delta']:+.2f}pp "
              f"SE_Δ(配对)={st['se']:.2f}pp 2SE_Δ={2 * st['se']:.2f}pp | b(amp错对照对)={st['n01']} "
              f"c(amp对对照错)={st['n10']} | McNemar 双侧精确 p={st['p']:.3g}（n={st['n']}） ⇒ "
              + ("★不可判（两臂都饱和 ≥95%）" if both_sat else
                 ("Δ 达 2SE 且 p<0.05" if sig else "该口径下未测出显著差异（不等于无差异）")), flush=True)
        print(f"[CRIT] R6-b   {arm}：T1 字母 {d['t1']:.2f}% vs {dj['t1']:.2f}% ⇒ Δ={stl['delta']:+.2f}pp "
              f"SE_Δ={stl['se']:.2f}pp 2SE_Δ={2 * stl['se']:.2f}pp b={stl['n01']} c={stl['n10']} "
              f"p={stl['p']:.3g} | 可用性={d['usable']}{'' if d['usable'] else '（' + d['unusable_why'] + '）'}", flush=True)
    return None


def r6_c(data, seed, mib_note=""):
    """叠加性：实测（--probe-cross-amp）vs 两项相加。"""
    print(f"[CRIT] R6-c AMP × 梯度检查点 的叠加性 seed={seed}{mib_note}", flush=True)
    best = None
    for nm in ("bf16", "fp16"):
        p = os.path.join(LOG_DIR, f"82_probecross_s{seed}_{nm}.json")
        if os.path.exists(p):
            best = (nm, p)
    if best is None:
        print(f"[CRIT] R6-c 未跑 --probe-cross-amp（logs/82_probecross_s{seed}_*.json 不存在）"
              f" ⇒ **未判（如实报）**；★不得用「两项相加」代替实测", flush=True)
    else:
        nm, p = best
        with open(p, encoding="utf-8") as f:
            x = json.load(f)
        r = x["rows"]
        print(f"[CRIT] R6-c ★实测四项（同一次全新构造，各 {PW + PR * PS + 5} 步，探测批 {x['probe_fp']}）："
              f"J-FP32={r['J-FP32']['peak_mib']:.0f} | J-AMP-{nm}={r[f'J-AMP-{nm}']['peak_mib']:.0f} | "
              f"J+GC-FP32={r['J+GC-FP32']['peak_mib']:.0f} | J+GC-AMP-{nm}={r[f'J+GC-AMP-{nm}']['peak_mib']:.0f} MiB",
              flush=True)
        print(f"[CRIT] R6-c Δ(仅 AMP)={x['d_amp']:+.0f} | Δ(仅 GC,FP32)={x['d_gc']:+.0f} | 和={x['d_amp'] + x['d_gc']:+.0f} | "
              f"**实测** AMP+GC={x['d_both']:+.0f} MiB ⇒ 差={x['diff']:+.0f} MiB ⇒ "
              f"{'可叠加（差值在探针噪声内）' if abs(x['diff']) <= 8.0 else '★不可简单叠加（如实报，且以实测为准）'}",
              flush=True)
    dj = data.get("J-FP32")
    dgc = data.get("J+GC-AMP")
    if dj and dgc:
        full = dgc["peak_mib"] - dj["peak_mib"]
        amp_arm = min([a for a in ("J-AMP-bf16", "J-AMP-fp16") if a in data],
                      key=lambda a: data[a]["peak_mib"], default=None)
        if amp_arm:
            full_amp = data[amp_arm]["peak_mib"] - dj["peak_mib"]
            print(f"[CRIT] R6-c 3000 步满程臂的一致性核对：Δ(J+GC-AMP vs J-FP32)={full:+.0f} MiB；"
                  f"Δ({amp_arm} vs J-FP32)={full_amp:+.0f} MiB；若 GC 单独的降幅沿用 §12.8/§12.9 的引文 "
                  f"（J+GC −590 MiB，**跨单元引文，本 run 未复测**）⇒ 相加={full_amp - 590:+.0f} MiB，"
                  f"实测−相加={full - (full_amp - 590):+.0f} MiB ⇒ 读法与上面 `--probe-cross-amp` 的实测一致时才算一致；"
                  f"★两项相加**不得**当作实测的替代", flush=True)
    else:
        print(f"[CRIT] R6-c 满程臂 J-FP32 / J+GC-AMP 有缺失 ⇒ 一致性核对未判（如实报）", flush=True)


def r6_d(data, seed, mib_note=""):
    """数值保真：loss 尺度、scaler 触发/跳过、NaN/Inf。"""
    print(f"[CRIT] R6-d 数值保真 seed={seed}{mib_note}", flush=True)
    for arm in ARMS:
        d = data.get(arm)
        if d is None:
            print(f"[CRIT] R6-d   {arm}：未跑 ⇒ 未判（如实报）", flush=True)
            continue
        s = d["scaler_stats"]
        lf, ll = d["loss_first"], d["loss_last"]
        skip_txt = (f"★跳过步数={s['skipped']}/{s['attempts']}（{100.0 * s['skipped_frac']:.2f}%）"
                    if d["scaler"] else
                    f"跳过步数=不适用（无 scaler；实测 AdamW 实走 {s['opt_step_called']}/{s['attempts']} 步）")
        print(f"[CRIT] R6-d   {arm}（声明 dtype={d['dtype_declared']} scaler={int(d['scaler'])}）："
              f"loss 首步={lf[0]:.4f}(ce={lf[1]:.4f}) 末步={ll[0]:.4f}(ce={ll[1]:.4f}) nats | "
              f"非有限 loss 步数={s['inf_loss_steps']}/{s['attempts']} | 梯度非有限（探针窗口/末态）="
              f"{s['grad_nf_in_probe']:.0f}/{s['grad_nf_final']:.0f} | max|g|={s['grad_max_final']:.3e} | "
              f"scaler: 初={s['scale_init']} 末={s['scale_final']} 最小={s['scale_min']} "
              f"降={s['scale_down']} 升={s['scale_up']} {skip_txt} | "
              f"★{'可用' if d['usable'] else '不可用：' + d['unusable_why']}", flush=True)
    dts = {}
    for arm, d in data.items():
        v = (d.get("dtype_evidence") or {}).get("per_dtype", {})
        for k, rec in v.items():
            dts[f"{arm}:{k}"] = rec.get("verdict", {}).get("gemm_hist", {})
    print(f"[CRIT] R6-d ★参与计算的 dtype（GEMM 类算子实收 dtype 分布，TorchDispatchMode 实测）：{dts}", flush=True)
    for arm, d in data.items():
        v = (d.get("dtype_evidence") or {}).get("per_dtype", {})
        for k, rec in v.items():
            print(f"[CRIT] R6-d   回退判定 {arm}/{k}：{rec.get('verdict', {}).get('verdict')}"
                  f" | 中间张量 dtype={rec.get('last_dtypes')}", flush=True)


def r6_e(data, seed, mib_note=""):
    """每步墙钟 / 峰值 / 参数 / T1 / T2 并列。"""
    print(f"[CRIT] R6-e 并列表 seed={seed}{mib_note}", flush=True)
    print(f"[CRIT] R6-e   {'臂':<12}{'dtype':<7}{'GC':<4}{'峰值MiB':>9}{'预留MiB':>9}{'每步s':>8}"
          f"{'参数':>10}{'T1%':>8}{'T2%':>8}{'墙钟s':>7}  可用", flush=True)
    for arm in ARMS:
        d = data.get(arm)
        if d is None:
            print(f"[CRIT] R6-e   {arm:<12}{'—':<7}未跑 ⇒ 未判", flush=True)
            continue
        print(f"[CRIT] R6-e   {arm:<12}{d['dtype_declared']:<7}{int(d['gc']):<4}{d['peak_mib']:>9.0f}"
              f"{d['peak_reserved_mib']:>9.0f}{d['s_per_step']:>8.3f}{d['params']:>10}{d['t1']:>8.2f}"
              f"{d['em']:>8.2f}{d['wall_s']:>7.0f}  {d['usable']}", flush=True)


def report_r6(seeds, smoke: bool, amp_dtype: str) -> int:
    rc = 0
    sign_tab = {}       # arm → [(seed, delta, sig, both_sat)]
    for seed in seeds:
        data = {}
        for arm in ARMS:
            d = load_hits(arm, seed, smoke)
            if d is not None:
                data[arm] = d
        print(f"[PAIR] === seed={seed} smoke={smoke} 读到：{sorted(data)} ===", flush=True)
        for arm, d in sorted(data.items()):
            print(f"[READ] arm={arm} dtype={d['dtype_declared']} scaler={d['scaler']} gc={d['gc']} "
                  f"T2={d['em']:.2f}% ±{d['se']:.2f}pp T1={d['t1']:.2f}% 选择={d['sel']} 参数={d['params']} "
                  f"步数={d['steps']} batch={d['batch']} maxlen={d['maxlen']} 峰值={d['peak_mib']:.0f} MiB "
                  f"预留={d['peak_reserved_mib']:.0f} MiB 每步={d['s_per_step']:.3f}s "
                  f"init_ok={d['init_ok']} gates_ok={d['gates_ok']} usable={d['usable']}", flush=True)
        fps = {a: d["probe_fp"] for a, d in data.items()}
        print(f"[MEM-CHECK] 探测批指纹跨臂：{fps or '★无读数'} ⇒ "
              + ("★无读数 ⇒ 未判" if not fps else
                 ("PASS 逐位相同（四要素成立）" if len(set(fps.values())) <= 1
                  else "★FAIL 各臂探测批不同 ⇒ 峰值不可比"))
              + f" | 与 §12.8 的 {PROBE_QUOTED_80} "
              + ("相同 ⇒ 同一探测批" if fps and set(fps.values()) == {PROBE_QUOTED_80}
                 else "不同（如实报；--smoke 用的是另一套置换）"), flush=True)
        steps = {a: (d["steps"], d["batch"], d["maxlen"], d["params"]) for a, d in data.items()}
        print(f"[PLAN-CHECK] 同步数/同 batch/同 maxlen/同参数：{steps or '★无读数'} ⇒ "
              f"{'PASS' if steps and len(set(steps.values())) <= 1 else '★未判/FAIL（读数不全或有差异）'}",
              flush=True)
        print(f"[INIT-CHECK] 各臂 init_ok={ {a: (d['init_ok'], d['init_worst']) for a, d in data.items()} }",
              flush=True)
        if all(a in data for a in ("J-AMP-bf16", "J-AMP-fp16", "J+GC-AMP")):
            pk = {nm: data[f"J-AMP-{nm}"]["peak_mib"] for nm in ("bf16", "fp16")}
            us = {nm: data[f"J-AMP-{nm}"]["usable"] for nm in ("bf16", "fp16")}
            pool = [nm for nm in pk if us[nm]] or list(pk)
            lo = min(pk[n] for n in pool)
            near = [n for n in pool if pk[n] <= lo * 1.01]
            best = min(near, key=lambda n: data[f"J-AMP-{n}"]["s_per_step"])
            print(f"[BEST-CHECK] 按规则（可用 → 峰值最低 → 同档取更快）推出的较优 AMP 档={best}"
                  f"（峰值={pk}，同档候选={near}，可用性={us}）；J+GC-AMP 实际用的 dtype="
                  f"{data['J+GC-AMP']['dtype_declared']} ⇒ "
                  f"{'一致' if best == data['J+GC-AMP']['dtype_declared'] else '★不一致（如实报，叠加性那条要按实际 dtype 读）'}",
                  flush=True)
        if "J-FP32" not in data:
            print(f"[CRIT] R6-a..e seed={seed}：J-FP32 读数缺失 ⇒ 主判据未判（如实报）", flush=True)
            rc = 1
            continue
        r6_a(data, seed)
        r6_b(data, seed)
        r6_c(data, seed)
        r6_d(data, seed)
        r6_e(data, seed)
        dj = data["J-FP32"]
        for arm in ("J-AMP-bf16", "J-AMP-fp16", "J+GC-AMP"):
            d = data.get(arm)
            if d is None:
                continue
            st = m79.paired_stats(d["hits"], dj["hits"])
            sign_tab.setdefault(arm, []).append((seed, st["delta"], st["se"], st["p"],
                                                 dj["em"] >= 95.0 and d["em"] >= 95.0))
    # ---- 跨 seed 汇总（2 seed 同号） ----
    if len(seeds) > 1:
        print(f"[CRIT] R6-b 跨 seed 汇总（★「2 seed 同号」是 §15 的写死要求）", flush=True)
        for arm, rows in sign_tab.items():
            if len(rows) < 2:
                print(f"[CRIT] R6-b   {arm}：只有 {len(rows)} 个 seed ⇒ 未达「2 seed」要求（如实报）", flush=True)
                continue
            if all(r[4] for r in rows):
                print(f"[CRIT] R6-b   {arm}：★两 seed 都饱和（≥95%）⇒ **不可判**（§15 写死）", flush=True)
                continue
            sgn = {('+' if r[1] > 0 else ('-' if r[1] < 0 else '0')) for r in rows}
            print(f"[CRIT] R6-b   {arm}：" + " | ".join(
                f"s{r[0]} Δ={r[1]:+.2f}pp SE={r[2]:.2f} p={r[3]:.3g} "
                f"{'显著' if abs(r[1]) >= 2 * r[2] and r[3] < 0.05 else '不显著'}" for r in rows)
                + f" ⇒ {'★2 seed 同号' if len(sgn) == 1 and sgn != {'0'} else '★两 seed 不同号（如实报）'}", flush=True)
        print(f"[CRIT] R6-a 跨 seed 汇总（峰值降幅）", flush=True)
        for arm in ("J-AMP-bf16", "J-AMP-fp16", "J+GC-AMP"):
            pcts = []
            for seed in seeds:
                dj, d = load_hits("J-FP32", seed, smoke), load_hits(arm, seed, smoke)
                if dj and d:
                    pcts.append((seed, 100.0 * (d["peak_mib"] - dj["peak_mib"]) / dj["peak_mib"]))
            if pcts:
                print(f"[CRIT] R6-a   {arm}：" + " | ".join(f"s{s} {p:+.2f}%" for s, p in pcts)
                      + f" ⇒ {'★两 seed 都 ≥10%（有实质收益）' if all(p <= -10.0 for _s, p in pcts) and len(pcts) >= 2 else ('两 seed 都 <10% ⇒ 本任务上 AMP 不显著省显存' if all(p > -10.0 for _s, p in pcts) and len(pcts) >= 2 else '结论不一致/不足 2 seed（如实报）')}",
                      flush=True)
    return rc


def best_amp(seed: int, smoke: bool = False) -> int:
    """CPU：选较优 AMP 档供 runner 用（规则：可用性 → 峰值最低 → 同档取每步更快）；最后一行只打 dtype 名。"""
    cands = {}
    for arm, nm in (("J-AMP-bf16", "bf16"), ("J-AMP-fp16", "fp16")):
        d = load_hits(arm, seed, smoke)
        if d is None:
            print(f"[BEST-AMP] {arm}：未跑", flush=True)
            continue
        cands[nm] = d
    if not cands:
        print("[BEST-AMP] 两档都未跑 ⇒ 默认 bf16（如实报：无依据）", flush=True)
        print("bf16")
        return 0
    usable = {nm: d["usable"] for nm, d in cands.items()}
    print(f"[BEST-AMP] 可用性={usable} | 峰值={{ "
          + ", ".join(f"{nm}: {d['peak_mib']:.0f} MiB(每步{d['s_per_step']:.3f}s)" for nm, d in cands.items())
          + " }", flush=True)
    pool = [nm for nm, d in cands.items() if d["usable"]] or list(cands)
    if len(pool) < len(cands):
        print(f"[BEST-AMP] ★有档被判不可用 ⇒ 只在可用档里选（这是「较优」的硬前提）", flush=True)
    pk = {nm: cands[nm]["peak_mib"] for nm in pool}
    lo = min(pk.values())
    near = [nm for nm in pool if pk[nm] <= lo * 1.01]      # 峰值差 ≤1% 视为同档
    best = min(near, key=lambda nm: cands[nm]["s_per_step"])
    print(f"[BEST-AMP] 选 {best}（规则：可用性优先 → 峰值最低 → 峰值同档（≤1%）时取每步墙钟更低者；"
          f"同档候选={near}）", flush=True)
    print(best)
    return 0


# ---------------- main ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="S82 L3-r6 AMP 四臂（J-FP32/J-AMP-bf16/J-AMP-fp16/J+GC-AMP）+ R6-a..e")
    ap.add_argument("--arm", choices=ARMS, default="J-FP32")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=STEPS_TOTAL)
    ap.add_argument("--amp-dtype", choices=("bf16", "fp16"), default="bf16",
                    help="J+GC-AMP 用哪一档 AMP（较优档，由 --best-amp 决定）")
    ap.add_argument("--smoke", action="store_true", help="50 步、评前 60 条（探路，不作判据）")
    ap.add_argument("--eval-amp", action="store_true", help="评测也用本臂 dtype（默认 FP32 评测口径）")
    ap.add_argument("--dtype-probe", action="store_true", help="只跑 dtype/回退/微基准/SDPA 开关诊断（不训练）")
    ap.add_argument("--sdp-flag-probe", action="store_true",
                    help="诊断：只量 allow_fp16_bf16_reduction_math_sdp 对峰值的作用（非四臂）")
    ap.add_argument("--probe-cross-amp", action="store_true", help="诊断：AMP×GC 的峰值叠加性（实测四项）")
    ap.add_argument("--report-r6", action="store_true", help="CPU：读四臂读数出 [CRIT] R6-a..e")
    ap.add_argument("--seeds", type=str, default="1234,5678", help="--report-r6 用：逗号分隔")
    ap.add_argument("--best-amp", action="store_true", help="CPU：依据读数选较优 AMP 档（最后一行打 dtype 名）")
    a = ap.parse_args(argv)
    seeds = [int(x) for x in a.seeds.split(",") if x.strip()]
    m79.load_tok()
    if a.report_r6:
        return report_r6(seeds, a.smoke, a.amp_dtype)
    if a.best_amp:
        return best_amp(a.seed, a.smoke)
    if a.dtype_probe or a.sdp_flag_probe:
        if not torch.cuda.is_available():
            print("GPU 起不来，立即停。", flush=True)
            return 3
        globals()["device"] = "cuda"
        m79.device = "cuda"
        train, _test = load_data()
        if a.dtype_probe:
            torch.manual_seed(a.seed)
            dtype_probe_all(a.seed, train)
        else:
            print_env(gpu_env())
            torch.manual_seed(a.seed)
            sdp_flag_sensitivity(a.seed, train, (a.amp_dtype,))
        print("[DONE] exit=0", flush=True)
        return 0
    if a.probe_cross_amp:
        return probe_cross_amp(a)
    return run_one(a)


if __name__ == "__main__":
    sys.exit(main())
