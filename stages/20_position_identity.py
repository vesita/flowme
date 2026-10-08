#!/usr/bin/env python3
"""S20 · 判决：给模因加【位置信号】能不能救三位数加法？能不能把「位置身份」绑进模因？★GPU。

只回答两个问题：
  Q1 给模因加位置信号后，交换模因两行的不变性会不会被打破？（S18 口径的交换干预冒烟）
  Q2 加了之后，add_3d 的 EM 会不会提升（相对 S19 的 res 臂 37.25%/31.13%）？

★唯一变量 = 思维卡（模因）**输入侧**是否带位置信号；其余与 S19 的 res 臂逐字相同。
注入方式选 **(C)**：现有实现里 PE 本就有，但只加在【输入卡】上——
  `Cards.logits`: x = emb(ids) + pe[:n] → in_enc → 思维卡输入 h 上**没有** PE。
  即「思维卡看不到位置」是现状而非设计 ⇒ 本单元把同一个 PE 表**也加到思维卡输入**：
      h_i = in_enc(x)_i                       （模因，位置无关）
      hi  = h + PE[:n]                        （★选 C：同一个 sin_pe 表，不新增参数）
      out = h + Think(hi)                     （残余臂 h + Think(h) 里的 Think 换输入，仅此一处）
  ⇒ 纯"修 bug"， диагностика价值最高；(A)/(B) 会顺带改变模因幅值与参数量。

干预挂点（两个，见 ThoughtPos）：
  post : 加完 PE 之后、进注意力之前 = 送进思维卡注意力的**整行**（含位置标签）互换
         —— 结构性等价于 S18 的 _InPatch（hook 直接包住注意力层）。
  pre  : 加 PE 之前 = 只换**内容**、位置标签留在原槽（"模因两行"的字面读法）。
         nopos 臂 pre 与 post 是同一个张量 ⇒ 只跑一次。

实测判据（分开报）：
  Q1：被读列（= 最后一个 prompt 位 s-1 及其后真实 target 位）|Δlogits| 与 Δ 非零比例。
      仍 ≈ 浮点噪声 ⇒ 不变性没被打破（如实报，说明假说错）；显著变大 ⇒ 被打破。
  Q2：add_3d EM ≥ S19-res 基线 + 2×SE 且 2 seed 同号 ⇒ 提升（报 Δ±SE）。
  R29：交换干预非零比例为 0 时，该口径作废 ⇒ 用「逐位置清零」(S18 side=in) 做正对照冒烟。
  R28：自回归生成评估主口径只用 batch=1；报 batch=1 vs batch=16 批内一致自检。

运行：add_3d 单桶（train4000/test800，与 S14/S19 同生成器同种子），
      2 arm × 2 seed × 3000 步，d=128 ff=512 NHEAD=4 lr=1e-3 batch=32 MAXLEN=512，
      思维卡带残差（两个 arm 都带）。device 必须 cuda。
ckpt：logs/20_ckpt_add3d_<pos|nopos>_seed<s>.pt（报 sha256 前16）；增量落 logs/20_results.jsonl。
只允许写：本文件（新建）、logs/、/tmp；stages/09..19*.py 只读（本文件按 S18 的办法 exec 其前半段复用）。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

ROOT = "/home/vesita/coding/my/flowme"
SRC19 = f"{ROOT}/stages/19_residual_scan.py"
SMOKE20 = os.environ.get("S20_SMOKE") == "1"
if SMOKE20:
    os.environ.setdefault("S19_SMOKE", "1")   # 必须在 exec 之前设：数据切分缩到 train300/test60

# ============================================================================
# ① 逐字复用 S19 前半段（到「# ③ 主循环」之前）
#    = device 门禁 + 配置 + tokenizer + 数据生成器(5桶) + Cards + train_run/eval_ce/
#      greedy_gen/hits_from/r28_selfcheck/ce_categories/report_ce/em_report/sha16
#    训练主循环在切片之外 ⇒ 模型与训练循环逐字复用，不重写。
# ============================================================================
_src = open(SRC19, encoding="utf-8").read()
assert _src.count("# ③ 主循环") == 1
_cut = _src.index("# ③ 主循环")
NS19 = {"__name__": "s19_head", "__file__": SRC19}
exec(compile(_src[:_cut], SRC19, "exec"), NS19)
for _k, _v in NS19.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
SMOKE = SMOKE20

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

T0 = time.time()
print(f"\n[S20] 复用链：exec(stages/19_residual_scan.py 前 {_src[:_cut].count(chr(10))} 行，"
      f"到「# ③ 主循环」之前) ⇒ 数据/模型/训练/评估逐字复用 | smoke={SMOKE}", flush=True)

# ============================================================================
# ② 位置信号注入（选 C）：同一个 PE 表，只多加在思维卡输入上；零新增参数
# ============================================================================
HOOK = {"pre": None, "post": None}
_CUR = {"pos": False}
_OrigCards = NS19["Cards"]


class ThoughtPos(nn.Module):
    """思维卡包装：pre 挂点 →（可选）加 PE → post 挂点 → 原注意力层。

    pos_signal=False 时是纯直通（logits 与原 Cards 逐位一致，见 CHK-C 自检）。
    PE 用 self._pe（non-persistent buffer）⇒ 会被 .to(device) 搬走，且不进 state_dict、
    不进 parameters()（AdamW 的参数集合与 S19 完全相同）。
    """

    def __init__(self, inner: nn.Module, d: int, pos_signal: bool, pe: torch.Tensor):
        super().__init__()
        self.inner = inner
        self.d = d
        self.pos_signal = pos_signal
        self.register_buffer("_pe", pe, persistent=False)

    def forward(self, x, src_mask=None, **kw):
        f = HOOK["pre"]
        if f is not None:
            x = f(x)
        if self.pos_signal:
            x = x + self._pe[: x.size(1)].unsqueeze(0)     # ★选 C：位置信号进思维卡
        g = HOOK["post"]
        if g is not None:
            x = g(x)
        return self.inner(x, src_mask=src_mask, **kw)


def Cards20(d: int, ff: int, residual: bool = False, identity: bool = False) -> nn.Module:
    """与 S19 的 Cards 唯一差异 = 思维卡外面包一层 ThoughtPos（读 _CUR['pos']）。"""
    m = _OrigCards(d, ff, identity=identity, residual=residual)
    if m.thought is not None:
        m.thought = ThoughtPos(m.thought, d, _CUR["pos"], m.pe)
    return m


NS19["Cards"] = Cards20     # train_run / n_params 内部引用 NS19 的 Cards ⇒ 同样被替换
Cards = Cards20

print(f"[S20] 注入方式=(C) 修 bug：S19 的 Cards.logits 只在输入卡加 PE（x=emb+pe[:n] → in_enc），"
      f"思维卡输入 h=in_enc(x) 上没有 PE ⇒ 本单元把同一个 sin_pe(MAXLEN,{D}) 也加到思维卡输入："
      f"h + Think(h + PE[:n])。参数量不变。", flush=True)


# ---------------- 自检 CHK-C：nopos 包装必须是逐位直通 ----------------
def passthrough_check() -> bool:
    _CUR["pos"] = False
    m0 = _OrigCards(D, FF, residual=True).to(device).eval()
    m1 = Cards20(D, FF, residual=True).to(device).eval()
    m1.emb.load_state_dict(m0.emb.state_dict())
    m1.in_enc.load_state_dict(m0.in_enc.state_dict())
    m1.head.load_state_dict(m0.head.state_dict())
    m1.thought.inner.load_state_dict(m0.thought.state_dict())
    ids, _ = build_batch(DATA["add_3d"]["test"][:8])
    with torch.no_grad():
        same = bool(torch.equal(m0.logits(ids), m1.logits(ids)))
        same2 = bool(torch.equal(m0.logits(ids).exp(), m1.logits(ids).exp()))
    n0 = sum(p.numel() for p in _OrigCards(D, FF, residual=True).parameters())
    _CUR["pos"] = True
    n1 = sum(p.numel() for p in Cards20(D, FF, residual=True).parameters())
    print(f"[CHK-C] nopos 包装 vs 原 Cards：logits 逐位一致={same}（exp 后也={same2}）| "
          f"参数量 nopos={n0} pos={n1} ⇒ {'位置信号零新增参数' if n0 == n1 else '★参数量变了'}", flush=True)
    del m0, m1
    torch.cuda.empty_cache()
    return same and n0 == n1


# ============================================================================
# ③ 干预函数与冒烟口径（S18 逐字口径：只动模因的一小部分）
# ============================================================================
def swp_fn(i: int):
    """交换第 i 与 i+1 行。"""

    def f(x, *rest):
        if i + 1 >= x.size(1):
            return x
        x = x.clone()
        a = x[:, i, :].clone()
        x[:, i, :] = x[:, i + 1, :]
        x[:, i + 1, :] = a
        return x
    return f


def zero_fn(i: int):
    """逐位置清零（S18 side=in 口径；R29 的正对照）。"""

    def f(x, *rest):
        if i >= x.size(1):
            return x
        x = x.clone()
        x[:, i, :] = 0.0
        return x
    return f


@torch.no_grad()
def logits_hooked(model, recs, key=None, fn=None):
    model.eval()          # ★必须：ce_categories 结尾会 model.train()，不显式 eval 会把 dropout 放进干预测量
    ids, s = build_batch(recs)
    s = s.to(ids.device)
    if key is not None:
        HOOK[key] = fn
    try:
        lg = model.logits(ids)
    finally:
        if key is not None:
            HOOK[key] = None
    return lg, ids, s


def read_col_max(dlg: torch.Tensor, ids: torch.Tensor, s: torch.Tensor, recs) -> torch.Tensor:
    """被读列 = 最后一个 prompt 位 s-1 及其后所有真实 target 位（不含 PAD）⇒ 逐样本 max|Δlogits|。"""
    N = ids.size(1)
    pos = torch.arange(N, device=ids.device).unsqueeze(0)
    L = torch.tensor([len(r["p"]) + len(r["t"]) for r in recs], device=ids.device).unsqueeze(1)
    m = (pos >= (s - 1).unsqueeze(1)) & (pos < L)
    d = dlg.abs().amax(dim=-1).masked_fill(~m, 0.0)
    return d.max(dim=1).values


@torch.no_grad()
def digit_ce_of(model, recs, key=None, fn=None) -> float:
    """数字/算子 token 的 CE（S10/S14 分类口径）⇒ exp(-CE) = 数字每步正确率。"""
    if key is not None:
        HOOK[key] = fn
    try:
        tok_ce, _, _ = ce_categories(model, recs)   # 结尾会 model.train()
    finally:
        if key is not None:
            HOOK[key] = None
        model.eval()                                # ★复位，避免 dropout 污染后续干预测量
    xs = tok_ce[0]
    return (sum(xs) / len(xs)) if xs else float("nan")


def _summ(vals: list[torch.Tensor], n_rec: int, tag: str) -> dict:
    allv = torch.cat(vals) if vals else torch.zeros(1)
    n_pair = len(vals) * n_rec
    return dict(tag=tag, n_pts=len(vals), n_pair=n_pair, mx=float(allv.max()),
                med=float(allv.median()), mean=float(allv.mean()),
                nz_1e6=float((allv > 1e-6).float().mean()),
                nz_1e4=float((allv > 1e-4).float().mean()),
                nz_1e2=float((allv > 1e-2).float().mean()))


def swap_smoke(model, recs, arm: str) -> dict:
    """★Q1 判据 + R29：交换模因两行 ⇒ 报被读列 |Δlogits| 与 Δ 非零比例。"""
    model.eval()
    base, ids, s = logits_hooked(model, recs)
    lsc = float(base.abs().max())          # logits 量级 ⇒ 用来判 |Δ| 是不是 1 ulp（fp32 ulp = 2^(floor(log2|L|))-23）
    ulp = math.ldexp(1.0, int(math.floor(math.log2(lsc))) - 23) if lsc > 0 else float("nan")
    lo_s = min(len(r["p"]) for r in recs)
    # 严格下游：i+1 <= lo_s-2 < 被读列起点 s-1（对每条样本都成立）⇒ 置换不变性若成立应逐位不变
    idxs = list(range(0, max(0, lo_s - 2)))
    variants = ["pre"] if arm == "nopos" else ["pre", "post"]
    base_ce = digit_ce_of(model, recs)
    picks = idxs[:: max(1, len(idxs) // 6)][:6]
    out = {}
    for vk in variants:
        vals = []
        for i in idxs:
            lg, _, _ = logits_hooked(model, recs, vk, swp_fn(i))
            vals.append(read_col_max(lg - base, ids, s, recs))
        d = _summ(vals, len(recs), vk)
        dacc = []
        for i in picks:
            ce = digit_ce_of(model, recs, vk, swp_fn(i))
            dacc.append((math.exp(-ce) - math.exp(-base_ce)) * 100.0)
        d["dacc_absmax_pp"] = max(abs(v) for v in dacc) if dacc else float("nan")
        d["dacc_nz"] = sum(1 for v in dacc if abs(v) > 1e-9)
        d["dacc_n"] = len(dacc)
        out[vk] = d
        print(f"[SMOKE-SWP] {arm} 挂点={vk} 交换点数={d['n_pts']}（i=0..{idxs[-1] if idxs else -1}, "
              f"lo_s={lo_s}，两个被换位都在被读列之前）| 被读列 max|Δlogits|={d['mx']:.3e} "
              f"中位={d['med']:.3e} 均值={d['mean']:.3e} | Δ非零比例 >1e-6:{d['nz_1e6']*100:.1f}% "
              f">1e-4:{d['nz_1e4']*100:.1f}% >1e-2:{d['nz_1e2']*100:.1f}% "
              f"| (点×样本={d['n_pair']}) | Δ数字每步 |max|={d['dacc_absmax_pp']:.4f}pp "
              f"非零 {d['dacc_nz']}/{d['dacc_n']} | logits量级={lsc:.1f} 1ulp={ulp:.3e} "
              f"⇒ max|Δ|/ulp={d['mx']/ulp:.2f}", flush=True)
    # S18 原口径的"末对" i=lo_s-2（最短样本上被读列起点就落在被换位上）单独报
    if lo_s >= 2:
        lg, _, _ = logits_hooked(model, recs, variants[-1], swp_fn(lo_s - 2))
        v = read_col_max(lg - base, ids, s, recs)
        print(f"[SMOKE-SWP] {arm} 挂点={variants[-1]} 末对 i={lo_s - 2}(S18 原口径：最短样本的被读列起点 "
              f"s-1 正好是被换位 ⇒ 该行不属'严格下游') max|Δlogits|={float(v.max()):.3e} "
              f"中位={float(v.median()):.3e} 非零>1e-4={float((v > 1e-4).float().mean())*100:.1f}%", flush=True)
        out["last_pair"] = dict(mx=float(v.max()), nz_1e4=float((v > 1e-4).float().mean()))
    off = all(out[k]["nz_1e4"] == 0.0 for k in variants)
    print(f"[R29] {arm} 交换干预非零性冒烟：{'★非零比例为 0 ⇒ 该口径作废，改用逐位置清零' if off else '非零 ⇒ 口径有效'}"
          f"（判据：|Δlogits| > 1e-4 的比例）", flush=True)
    return out


def zero_smoke(model, recs, arm: str) -> dict:
    """R29 正对照：逐位置清零（S18 side=in）——证明评估通路看得见改动。"""
    model.eval()
    base, ids, s = logits_hooked(model, recs)
    lo_s = min(len(r["p"]) for r in recs)
    idxs = sorted({0, lo_s // 4, lo_s // 2, (3 * lo_s) // 4, max(0, lo_s - 1)})
    vals = []
    for i in idxs:
        lg, _, _ = logits_hooked(model, recs, "pre", zero_fn(i))
        vals.append(read_col_max(lg - base, ids, s, recs))
    d = _summ(vals, len(recs), "zero")
    print(f"[SMOKE-ZERO] {arm} 逐位置清零点={idxs} | 被读列 max|Δlogits|={d['mx']:.3e} "
          f"中位={d['med']:.3e} | 非零比例 >1e-4:{d['nz_1e4']*100:.1f}%", flush=True)
    return d


# ============================================================================
# ④ 配置（只跑 add_3d；其余与 S19 res 臂逐字相同）
# ============================================================================
BUCKET = "add_3d"
TRAIN, TEST = DATA[BUCKET]["train"], DATA[BUCKET]["test"]
SEEDS = (1234, 5678)
ARMS = ("nopos", "pos")
CKPT_TMPL = ROOT + "/logs/20_ckpt_add3d_{arm}_seed{seed}.pt"
RESULT_JSONL = ROOT + "/logs/20_results.jsonl"
S19_EM = {1234: 0.3725, 5678: 0.3113}      # S19 res 臂 add_3d 实测（EM 全量, n=800）
S19_DIG = {1234: 0.5259, 5678: 0.4963}     # S19 res 臂 add_3d 数字每步正确率
N_SMOKE_RECS = 16

print(f"[S20] ===== 桶={BUCKET} train={len(TRAIN)} test={len(TEST)} | arms={ARMS} seeds={SEEDS} "
      f"steps={STEPS} | d={D} ff={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} "
      f"| 两 arm 思维卡均带残差，唯一变量=思维卡输入是否加 PE | device={device} "
      f"{torch.cuda.get_device_name(0)} =====", flush=True)

CHK_C = passthrough_check()


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def append_jsonl(obj: dict) -> None:
    with open(RESULT_JSONL, "a") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def s19_key_map(k: str) -> str:
    return "thought.inner." + k[len("thought."):] if k.startswith("thought.") else k


# ============================================================================
# ⑤ 锚点 A：把 S19 的 res ckpt 装进本单元的 nopos 模型 ⇒ 验证评估口径逐字一致
# ============================================================================
print(f"\n[S20] ===== ★锚点 A：装载 S19 的 add_3d res ckpt，验证本单元评估口径 =====", flush=True)
for _s in SEEDS:
    _p = ROOT + f"/logs/19_ckpt_add_3d_res_seed{_s}.pt"
    if not os.path.exists(_p):
        print(f"[ANCHOR-A] seed={_s}: ★缺 {_p}", flush=True)
        continue
    _CUR["pos"] = False
    _m = Cards20(D, FF, residual=True).to(device)
    _sd = torch.load(_p, map_location="cpu", weights_only=True)
    _m.load_state_dict({s19_key_map(k): v for k, v in _sd.items()}, strict=True)
    _m.eval()
    _txt = greedy_gen(_m, TEST, batch=1)
    _hits = hits_from(_txt, TEST)
    _em = sum(h["strict"] for h in _hits) / len(_hits)
    print(f"[ANCHOR-A] seed={_s} S19-ckpt 装入 nopos 模型 ⇒ EM(batch=1,n={len(_hits)})={_em*100:.2f}% "
          f"| S19 报 {S19_EM[_s]*100:.2f}% | Δ={(_em - S19_EM[_s])*100:+.2f}pp ⇒ "
          f"{'逐字复现（评估口径一致）' if abs(_em - S19_EM[_s]) < 1e-9 else '★不一致（点名）'}", flush=True)
    del _m
    torch.cuda.empty_cache()

# ============================================================================
# ⑥ 主循环：2 arm × 2 seed（每 run：训练 → 干预冒烟 → R28 → EM(batch=1) → CE → ckpt）
# ============================================================================
RES: dict[tuple[str, int], dict] = {}
SWAP_SMOKE: dict[str, dict] = {}

for arm in ARMS:
    for seed in SEEDS:
        t0 = time.time()
        print(f"\n[S20] ===== arm={arm} seed={seed} steps={STEPS} "
              f"（思维卡输入位置信号={'加' if arm == 'pos' else '不加'}，两臂均带残差）=====", flush=True)
        _CUR["pos"] = (arm == "pos")
        model, wall = train_run(seed, TRAIN, STEPS, residual=True)
        model.eval()

        # ---- R29 + Q1：交换干预冒烟（先做）----
        sub = TEST[:N_SMOKE_RECS]
        sw = swap_smoke(model, sub, arm)
        zs = zero_smoke(model, sub, arm)
        SWAP_SMOKE[arm] = sw

        # ---- R28 自检（batch=1 vs batch=16 逐字一致）----
        r28 = r28_selfcheck(model, TEST, tag=f"[{BUCKET} {arm} seed={seed}] ")

        # ---- Q2：EM 主口径 batch=1（R28）----
        txt = greedy_gen(model, TEST, batch=EM_BATCH)
        hits = hits_from(txt, TEST)
        em = em_report(f"{BUCKET} {arm} seed={seed} (batch={EM_BATCH})", hits)
        strict = [h["strict"] for h in hits]

        # ---- 数字每步正确率（CE 口径）----
        rep = report_ce(f"{BUCKET} {arm} seed={seed}", model, TEST)

        ck = CKPT_TMPL.format(arm=arm, seed=seed)
        if not SMOKE:
            torch.save(model.state_dict(), ck)
            dig = sha16(ck)
        else:
            dig = "-"
        total = time.time() - t0
        emv, emse = em["全量"]["em"], em["全量"]["se"]
        print(f"[MAIN-TABLE] {BUCKET} {arm} seed={seed} | EM全量={emv*100:.2f}%±{emse*100:.2f}(n={em['全量']['n']}) "
              f"| 数字每步={rep['acc']*100:.2f}%±{rep['acc_se']*100:.2f}pp | 整体CE={rep['ce_m']:.4f}"
              f" | R28={r28['same']}/{r28['k']} | 训练={wall/60:.1f}min 合计={total/60:.1f}min "
              f"| ckpt={os.path.basename(ck)} sha16={dig}", flush=True)
        RES[(arm, seed)] = dict(arm=arm, seed=seed, steps=STEPS, em=emv, em_se=emse,
                                acc=rep.get("acc"), acc_se=rep.get("acc_se"),
                                ce=rep.get("ce_m"), wall_train=wall, wall_total=total,
                                r28_same=r28["same"], r28_k=r28["k"],
                                ckpt=ck, sha=dig, strict=strict,
                                swap={k: v for k, v in sw.items() if k != "last_pair"},
                                zero=zs)
        append_jsonl({k: v for k, v in RES[(arm, seed)].items() if k != "strict"}) if not SMOKE else None
        del model
        torch.cuda.empty_cache()

# ============================================================================
# ⑦ ★Q2 判决：配对 Δ = EM(pos) − EM(nopos)，逐 test 样本配对（门槛 2×SE，2 seed 同号）
# ============================================================================
def paired_delta(a: list[int], b: list[int]):
    xs = [x - y for x, y in zip(a, b)]
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return n, m, math.sqrt(var / n)


print(f"\n[S20] ===== ★Q2 主表：add_3d EM（batch=1，n=800）=====", flush=True)
print("[S20] arm | seed | EM全量±SE | 数字每步±SE | 整体CE | R28 | ckpt sha16", flush=True)
for arm in ARMS:
    for seed in SEEDS:
        r = RES.get((arm, seed))
        if r is None:
            print(f"[S20] {arm} | {seed} | ★未完成", flush=True)
            continue
        print(f"[S20] {arm} | {seed} | {r['em']*100:.2f}%±{r['em_se']*100:.2f} | "
              f"{r['acc']*100:.2f}%±{r['acc_se']*100:.2f}pp | {r['ce']:.4f} | "
              f"{r['r28_same']}/{r['r28_k']} | {r['sha']}", flush=True)

print(f"\n[S20] ===== ★Q2 配对 Δ = EM(pos) − EM(nopos)（逐 test 样本配对，门槛 2×SE）=====", flush=True)
DELTAS: list[dict] = []
Q2_OK = False
if all((a, s) in RES for a in ARMS for s in SEEDS):
    for seed in SEEDS:
        n, d, se = paired_delta(RES[("pos", seed)]["strict"], RES[("nopos", seed)]["strict"])
        sig = (d > 0) and (d >= 2 * se)
        base_em = RES[("nopos", seed)]["em"]
        pos_em = RES[("pos", seed)]["em"]
        ref_dev = (base_em - S19_EM[seed]) * 100
        DELTAS.append(dict(seed=seed, n=n, d=d, se=se, sig=sig, base_em=base_em, pos_em=pos_em))
        print(f"[S20] Δ seed={seed}: {d*100:+.2f}±{se*100:.2f}pp (Δ/SE={d/se if se > 0 else float('inf'):.2f}) "
              f"| nopos {base_em*100:.2f}% → pos {pos_em*100:.2f}% | "
              f"{'★显著正(Δ≥2SE)' if sig else '未过 2SE 门槛'} | "
              f"nopos 对 S19 基线 {S19_EM[seed]*100:.2f}% 差={ref_dev:+.2f}pp", flush=True)
    same_sign = all(dd["d"] > 0 for dd in DELTAS)
    Q2_OK = same_sign and all(dd["sig"] for dd in DELTAS)
    print(f"[S20] 同号判定：2 seed 均 Δ>0 = {same_sign}；均 Δ≥2SE = "
          f"{all(dd['sig'] for dd in DELTAS)} ⇒ {'★提升' if Q2_OK else '无显著提升'}", flush=True)
    for seed in SEEDS:
        print(f"[S20] 基线复现核对 seed={seed}: 无位置 arm EM={RES[('nopos', seed)]['em']*100:.2f}% "
              f"vs S19-res {S19_EM[seed]*100:.2f}% (Δ={(RES[('nopos', seed)]['em']-S19_EM[seed])*100:+.2f}pp) "
              f"| 数字每步={RES[('nopos', seed)]['acc']*100:.2f}% vs S19 {S19_DIG[seed]*100:.2f}%", flush=True)
else:
    print("[S20] ★run 不全 ⇒ 无法配对", flush=True)

# ============================================================================
# ⑧ ★Q1 判决（分开报）
# ============================================================================
print(f"\n[S20] ===== ★Q1 判决：交换干预的不变性 =====", flush=True)
_LBL = {"pre": "只换内容/位置标签留原槽", "post": "整行互换(含位置标签)"}
for arm in ARMS:
    sw = SWAP_SMOKE.get(arm)
    if sw is None:
        print(f"[S20] Q1 {arm}: ★无冒烟", flush=True)
        continue
    for vk in ("pre", "post"):
        if vk not in sw:
            continue
        d = sw[vk]
        broken = d["nz_1e4"] > 0.5        # 判据：|Δlogits| > 1e-4（比浮点噪声高 ~2 个量级）的占比过半
        print(f"[Q1] {arm} {vk}({_LBL[vk]}): max|Δlogits|={d['mx']:.3e} "
              f"非零(>1e-4)={d['nz_1e4']*100:.1f}% (点×样本={d['n_pair']}) ⇒ "
              f"{'★不变性被打破' if broken else '不变（≈浮点噪声/1 ulp）'}", flush=True)

print(f"\n[META] device={device} {torch.cuda.get_device_name(0)} | 桶={BUCKET} | 唯一变量=思维卡输入是否加 PE(方式C) "
      f"| CHK-C 直通自检={CHK_C} | R28 主口径=batch{EM_BATCH}(自检 batch{CHK_BATCH}) | "
      f"arm 数={len(RES)}/4 | 总墙钟={(time.time()-T0)/60:.1f}min | smoke={SMOKE}", flush=True)
print("[DONE] exit=0", flush=True)
