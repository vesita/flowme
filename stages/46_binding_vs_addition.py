#!/usr/bin/env python3
"""S46 · P3 ★ 绑定 vs 相加：解 S18/S20 的置换不变性异常（用 S27 给的 C1）。

异常（S18 logs/18_finegrained.log / S20 logs/20_pos.log 实测）：
  互换思维卡（模因）两整行 ⇒ 下游 logits 的 |Δ| 只有 2–6e-6 ⇒ 对【行置换】代数上不变；
  代数原因：softmax 分母 = keys 的多重集 ⇒ 置换 keys/values 不改变 ΣαV；
  且把【加性 PE 也加到思维卡输入】会 −34.9pp（崩），但仍不打破置换不变性
  ⇒ 结论：位置身份未绑定 ⇒ 需要"绑定（binding）"而不是"相加"。
本单元只回答一个问题：引入"位置绑定"后，(a) 置换不变性是否被打破？(b) 能力是否受影响(@12000)？

三臂 × 2 seed（其余与 S38/E6 逐字相同；只改思维卡的自注意力）：
  ① base 现状：思维卡 = TransformerEncoderLayer + 外层残差 h + Think(h)，无位置绑定；
  ② bind 绑定版：★选 (i) KV 外积绑定（S27 的 C1，nanoSeek/model/attention.py:573 的
     `S ← r⊙S + β·k·vᵀ` 外积思想的静态版）：思维卡自注意力里对 key 与 value 各加一项
     【低秩双线性/外积】位置绑定
         a_i = tanh(P·p_i) ∈ R^r ;  t'_i = t_i + W_o·vec( a_i ⊗ (W_c·t_i) )   (t ∈ {k, v})
     ⇒ 每行的 (key,value) 与【绝对位置】绑死 ⇒ 交换两行后同一位置上的 key 不再相同
     ⇒ 不再只是 keys 的多重集 ⇒ ΣαV 对行置换不再不变。形状严格 [n,d]→[n,d]（卡内契约）；
     新增参数 5248（+0.21%），远低于 5% 上限。
     为什么选 (i)：S27 已把 C1 列为候选，且它是三项里唯一"外积/双线性"绑定
     （(ii) 对 key 加位置本质仍是相加，(iii) 只门控 value，二者都只覆盖一半）。
  ③ rand 随机位置对照：与 ② 同结构、同参数量，但位置向量在样本内按【固定种子的随机
     置换 π】打乱（列 i 用 p_{π(i)}）⇒ 若 ③ 不差于 ②，则 (a) 的 Δ 只是"多参数"效应。

判据（写死）：
  (a) 交换【只互换模因两整行、两个位置都在被读列之前】⇒ 报 |Δlogits| 绝对值与相对量。
      ② 若 |Δlogits| ≥1e-3（≫ 基线 1e-6 量级）⇒ 绑定生效；仍 ~1e-6 ⇒ 未生效（如实报）。
  (b) Δ = EM(臂) − EM(①)，配对 ±SE（全量 n=800 与 clean n=770 两种口径）；
      ②≥①+2SE 且 2 seed 同号 ⇒ 提升；|Δ|≤1SE ⇒ 无影响（(a) 成立则仍是机理发现）；≤−2SE ⇒ 有害；
      ③ 应显著差于 ②。
  (c) ① 与 ② 的 9000→12000 增量 ≥1pp ⇒ 未饱和、能力结论不可下（(a) 不受饱和影响）。

口径：batch=1（R28，报 batch=1 vs 16 一致性）；clean=add_3d 剔除 30 条 nl 含字面量 {expr}（n=770）；
  锚点 ①@12000 必须复现 S38 w1 = 91.88/87.12%（全量）/ 95.45/90.52%（clean），Δ>1pp 点名；
  地板 0.12%/0.25%；SEEN/UNSEEN 分层；R29 非零比例（绑定项幅值比 + 交换干预非零比例）；
  TEST sha256[:16] == b34e7ea515203227。

★ 交换挂点说明：与 S18/S20 的 `_InPatch` 一致 —— 换的是【送进思维卡注意力的那两整行】
  （h = in_enc(x) 之后、Think 之前）。这才能暴露"卡内无位置绑定"：若绑定只在卡外
  （S20 的加性 PE，加在 hook 之后被整行换走），换行 = 换标签 ⇒ 仍不变（S20 已实测）。

运行：systemd-run … stages/46_binding_vs_addition.py（S46_SMOKE=1 = 冒烟）。
ckpt：logs/46_ckpt_{base|bind|rand}_seed{s}.pt（报 sha256[:16]）；曲线落 logs/46_results.jsonl。
只允许写：本文件（新建）、logs/46_*、logs/p3_bind.log、/tmp；既有 stages/*.py 只读。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from statistics import median

ROOT = "/home/vesita/coding/my/flowme"
SRC19 = f"{ROOT}/stages/19_residual_scan.py"

SMOKE46 = os.environ.get("S46_SMOKE") == "1"
if SMOKE46:
    os.environ["S19_SMOKE"] = "1"          # 必须在 exec 之前：数据切到 train300/test60

# ============================================================================
# ① 逐字复用 S19 前半段（同 tokenizer / 同 add_3d train4000(test800) / 同 build_batch /
#    masked_ce(w1) / greedy_gen(batch=1) / parse_ans / r28_selfcheck / sin_pe / enc_layer）
# ============================================================================
_src = open(SRC19, encoding="utf-8").read()
assert _src.count("# ③ 主循环") == 1
_cut = _src.index("# ③ 主循环")
NS = {"__name__": "s19_head", "__file__": SRC19}
exec(compile(_src[:_cut], SRC19, "exec"), NS)          # noqa: S102
for _k, _v in NS.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
del NS

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402

assert device == "cuda", "★本单元必须 cuda"

T0 = time.time()
BUCKET = "add_3d"
ARMS46 = ("base", "bind", "rand")
SEEDS46 = (1234,) if SMOKE46 else (1234, 5678)
STEPS46 = 60 if SMOKE46 else 12000
GRID46 = (30, 60) if SMOKE46 else (3000, 6000, 9000, 12000)
SAVE46 = GRID46[-1]
CKPT_TMPL = ("/tmp/46_ckpt_{arm}_seed{seed}.pt" if SMOKE46
             else f"{ROOT}/logs/46_ckpt_{{arm}}_seed{{seed}}.pt")
JSONL = "/tmp/46_results_smoke.jsonl" if SMOKE46 else f"{ROOT}/logs/46_results.jsonl"
DP_BIND, R_BIND, S_BIND = 16, 4, 4          # 位置维度 / 外积两侧秩
TEST_SHA_EXPECT = "b34e7ea515203227"
LEAK_EXPECT = 16
ANCHOR_S38 = {1234: dict(full=0.9188, clean=0.9545), 5678: dict(full=0.8712, clean=0.9052)}

print(f"\n[S46] ★绑定 vs 相加：桶={BUCKET} 臂={ARMS46} seeds={SEEDS46} 步数={STEPS46} "
      f"曲线格={GRID46} | d={D} ff={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} | "
      f"EM口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | smoke={SMOKE46}", flush=True)

# ============================================================================
# ② 数据核对（口径 2/3/4/6）
# ============================================================================
TRAIN = DATA[BUCKET]["train"]
TEST = DATA[BUCKET]["test"]
N_EVAL = 60 if SMOKE46 else N_TEST

_h = hashlib.sha256()
for r in TEST:
    _h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
TEST_SHA = _h.hexdigest()[:16]
CLEAN_IDX = [i for i, r in enumerate(TEST[:N_EVAL]) if "{expr}" not in r["nl"]]
_tp_tr = {(r["a"], r["b"]) for r in TRAIN}
for r in TEST:
    r["seen"] = (r["a"], r["b"]) in _tp_tr
LEAK = sum(1 for r in TEST if r["seen"])
N_SEEN, N_UNSEEN = LEAK, len(TEST) - LEAK
_c_tr = Counter(r["gold"] for r in TRAIN)
_top1, _top1n = _c_tr.most_common(1)[0]
_top5 = [v for v, _ in _c_tr.most_common(5)]
FLOOR_1 = sum(1 for r in TEST if r["gold"] == _top1) / len(TEST)
FLOOR_5 = sum(1 for r in TEST if r["gold"] in _top5) / len(TEST)
sha_ok = TEST_SHA == TEST_SHA_EXPECT
leak_ok = LEAK == LEAK_EXPECT
print(f"[S46-CHK6] TEST sha256[:16]={TEST_SHA} vs 入库 {TEST_SHA_EXPECT} ⇒ "
      f"{'逐位一致 ✓' if sha_ok else '★不一致'} | (a,b) 泄漏={LEAK}/{len(TEST)} vs {LEAK_EXPECT} ⇒ "
      f"{'一致 ✓' if leak_ok else '★不一致'}", flush=True)
print(f"[S46-CHK2] clean 口径：nl 含字面量 '{{expr}}' 剔除 ⇒ n={len(CLEAN_IDX)}/{N_EVAL}"
      f"{'（★冒烟数，非 770）' if SMOKE46 else ''}", flush=True)
print(f"[S46-CHK4] SEEN={N_SEEN} / UNSEEN={N_UNSEEN} | 地板(单答案 '{_top1}'/top5)= "
      f"{FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}%（不借 1.89%）", flush=True)
if not SMOKE46:
    assert sha_ok and leak_ok, "数据指纹不符 ⇒ 本单元无效"


# ============================================================================
# ③ 绑定模块：(i) KV 外积绑定 —— 思维卡自注意力里给 k/v 各加一项低秩双线性位置绑定
# ============================================================================
class BoundMHA(nn.Module):
    """包住原 nn.MultiheadAttention：参数/初始化与基线逐位相同（inner 原样持有）。

    arm='off' 时不加绑定 ⇒ 与 MHA.forward 数值等价（见 §⑥ CHK-EQ 自检）。
    arm='bind' 用真实位置 p_i；arm='rand' 用固定随机置换 π 的 p_{π(i)}（同参数量）。
    形状 [n,d]→[n,d]；位置表为 sin_pe（非持久 buffer、非参数）。
    ★ _qkv_same_embed_dim=False / batch_first=True：让 TransformerEncoderLayer 在
      eval 下【不走】fused 快路径，保证一定调用本 forward（否则绑定被静默跳过）。
    """

    def __init__(self, inner: nn.MultiheadAttention, d: int, nhead: int, arm: str,
                 dp: int = 16, r: int = 4, s: int = 4, perm_seed: int = 31337):
        super().__init__()
        self.inner = inner
        self.d, self.nhead, self.dh = d, nhead, d // nhead
        self.dropout = inner.dropout
        self.batch_first = True
        self._qkv_same_embed_dim = False
        self.arm = arm
        self.r, self.s, self.dp = r, s, dp
        self.record = False
        self.stats: list[float] = []
        self.register_buffer("pe", sin_pe(MAXLEN, dp), persistent=False)
        self.p_p = nn.Linear(dp, r, bias=False)
        self.k_c = nn.Linear(d, s, bias=False)
        self.k_o = nn.Linear(r * s, d, bias=False)
        self.v_c = nn.Linear(d, s, bias=False)
        self.v_o = nn.Linear(r * s, d, bias=False)
        if arm == "rand":
            g = torch.Generator().manual_seed(perm_seed)
            self.register_buffer("perm", torch.randperm(MAXLEN, generator=g), persistent=False)
        else:
            self.register_buffer("perm", torch.arange(MAXLEN), persistent=False)

    @property
    def in_proj_bias(self):
        return self.inner.in_proj_bias          # 让 encoder layer 的快路径检查能读到

    def _bind(self, t, pos, c_proj, o_proj):
        p = self.pe[pos]                                         # (L, dp)
        a = torch.tanh(self.p_p(p)).view(p.size(0), 1, self.r, 1)   # (L, 1, r, 1)
        c = c_proj(t).unsqueeze(-2)                              # (L, N, 1, s)
        outer = a * c                                            # (L, N, r, s)  ← 外积
        return o_proj(outer.reshape(*t.shape[:-1], self.r * self.s))

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None,
                need_weights=False, is_causal=False, average_attn_weights=True, **kw):
        assert key_padding_mask is None, "本单元把 padding 折进 attn_mask，不用 key_padding_mask"
        q, k, v = (x.transpose(0, 1) for x in (query, key, value))       # (N,L,E)->(L,N,E)
        L, N, E = q.shape
        q, k, v = F.linear(q, self.inner.in_proj_weight, self.inner.in_proj_bias).chunk(3, -1)
        if self.arm != "off":
            pos = self.perm[:L]
            k0n = k.norm(dim=-1)
            bk = self._bind(k, pos, self.k_c, self.k_o)
            bv = self._bind(v, pos, self.v_c, self.v_o)
            k = k + bk
            v = v + bv
            if self.record:
                with torch.no_grad():
                    self.stats.append(float((bk.norm(dim=-1) / (k0n + 1e-9)).mean()))
        nh, dh = self.nhead, self.dh
        q = q.reshape(L, N * nh, dh).transpose(0, 1) * (1.0 / math.sqrt(dh))
        k = k.reshape(L, N * nh, dh).transpose(0, 1)
        v = v.reshape(L, N * nh, dh).transpose(0, 1)
        attn = torch.bmm(q, k.transpose(1, 2))
        if attn_mask is not None:
            attn = attn + (attn_mask if attn_mask.dim() == 3 else attn_mask.unsqueeze(0))
        attn = F.softmax(attn, dim=-1)
        if self.dropout > 0.0:
            attn = F.dropout(attn, p=self.dropout, training=self.training)
        out = torch.bmm(attn, v).transpose(0, 1).reshape(L, N, E)
        out = self.inner.out_proj(out).transpose(0, 1)
        return out, None


class Cards46(nn.Module):
    """与 S38 Cards38 逐字同构（emb → in_enc → thought(残差) → head）。
    arm=base 只有现状；bind/rand 把【思维卡的 self_attn】换成 BoundMHA。
    ★ 绑定模块在 RNG 状态保护下创建 ⇒ ①②③ 的基线权重/数据顺序/dropout 流逐位相同。"""

    def __init__(self, d, ff, arm="base"):
        super().__init__()
        self.d = d
        self.arm = arm
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        self.thought = enc_layer(d, ff)
        self.head = nn.Linear(d, V)
        self.register_buffer("pe", sin_pe(MAXLEN, d), persistent=False)
        if arm != "base":
            st = torch.get_rng_state()
            cst = torch.cuda.get_rng_state_all()
            self.thought.self_attn = BoundMHA(self.thought.self_attn, d, NHEAD, arm=arm,
                                              dp=DP_BIND, r=R_BIND, s=S_BIND)
            torch.set_rng_state(st)
            torch.cuda.set_rng_state_all(cst)

    def logits(self, ids, swap=None):
        n = ids.size(1)
        m = torch.full((n, n), float("-inf"), device=ids.device)
        m = torch.triu(m, diagonal=1)
        m = m.unsqueeze(0).expand(ids.size(0), -1, -1).clone()
        m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
        m = m.repeat_interleave(NHEAD, dim=0)
        pe = self.pe if n <= self.pe.size(0) else sin_pe(n, self.d).to(self.pe.device)
        x = self.emb(ids) + pe[:n].unsqueeze(0)
        h = self.in_enc(x, src_mask=m)
        if swap is not None:                       # ★与 S18/S20 的 _InPatch 同挂点
            i, j = swap
            perm = list(range(n))
            perm[i], perm[j] = perm[j], perm[i]
            h = h[:, perm]
        h = h + self.thought(h, src_mask=m)
        return self.head(h)


def build_model(seed, arm):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return Cards46(D, FF, arm).to(device)


def n_params46(arm):
    torch.manual_seed(0)
    mdl = Cards46(D, FF, arm)
    n = sum(p.numel() for p in mdl.parameters())
    del mdl
    return n


# ============================================================================
# ④ 训练（与 S38 train38 逐字相同的顺序/优化器/数据序；只多取 4 个格点快照）
# ============================================================================
def train46(seed, arm, train, steps, grid, save_at):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards46(D, FF, arm).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0, last = 0, 0, time.time(), 0.0
    snaps: dict[int, dict] = {}
    gset = set(grid)
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(model, [train[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if step in gset:
            snaps[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step % 3000 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train:{arm}] step={step}/{steps} loss={loss.item():.4f} "
                  f"elapsed={el:.0f}s s/step={(el - last) / (step % 3000 or 3000):.3f}", flush=True)
            last = el
    assert set(grid) <= set(snaps)
    ck = CKPT_TMPL.format(arm=arm, seed=seed)
    torch.save(snaps[save_at], ck)
    del model
    torch.cuda.empty_cache()
    return ck, time.time() - t0, snaps


@torch.no_grad()
def em_eval(model, recs):
    model.eval()
    txt = greedy_gen(model, recs, batch=EM_BATCH)          # ★R28 主口径 batch=1
    return [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, recs)]


@torch.no_grad()
def swap_sensitivity(model, recs, pairs):
    """★(a) 只互换模因两整行（都在被读列之前）⇒ 被读列 |Δlogits| 绝对值 + 相对量 + R29。

    被读列 = 预测 target 的列 [s-1, s+len(t)-1]（真实下游）。
    """
    model.eval()
    ids, s = build_batch(recs)
    base = model.logits(ids)
    lsc = float(base.abs().max())
    out = {}
    for (i, j) in pairs:
        d = (model.logits(ids, swap=(i, j)) - base).abs()
        mx, med, rel, nz4, nz6 = [], [], [], [], []
        for b in range(len(recs)):
            a, e = int(s[b]), int(s[b]) + len(recs[b]["t"])
            seg = d[b, a - 1: e]
            ref = float(base[b, a - 1: e].abs().max())
            m = float(seg.max())
            mx.append(m)
            med.append(float(seg.median()))
            rel.append(m / max(ref, 1e-12))
            nz4.append(float((seg > 1e-4).float().mean()))
            nz6.append(float((seg > 1e-6).float().mean()))
        out[(i, j)] = dict(mx=max(mx), med=median(med), rel=max(rel),
                           nz4=sum(nz4) / len(nz4), nz6=sum(nz6) / len(nz6))
    return out, lsc


def binding_r29(model):
    """R29（绑定面）：绑定项相对 key 的幅值比 + 非零比例。"""
    mods = [m for m in model.modules() if isinstance(m, BoundMHA) and m.arm != "off"]
    if not mods:
        return dict(ratio=float("nan"), n=0)
    for m in mods:
        m.record, m.stats = True, []
    ids, _ = build_batch(TEST[:8])
    with torch.no_grad():
        model.eval()
        model.logits(ids)
    vals = [v for m in mods for v in m.stats]
    for m in mods:
        m.record = False
    return dict(ratio=sum(vals) / len(vals) if vals else float("nan"), n=len(vals))


def paired_delta(x, y):
    """配对 (seed 级) Δ = x − y：mean ± SE（n=2 ⇒ SE=|d1−d2|/2）。"""
    ds = [a - b for a, b in zip(x, y)]
    m, se = mean_se(ds)
    return m, se, ds


# ============================================================================
# ⑤ 主循环：3 臂 × 2 seed × 12000 步
# ============================================================================
print(f"[S46-PARAMS] 参数量：base={n_params46('base')/1e6:.4f}M "
      f"bind={n_params46('bind')/1e6:.4f}M rand={n_params46('rand')/1e6:.4f}M", flush=True)
open(JSONL, "w").close()
RES: dict = {}
SWAP: dict = {}
t_all = time.time()

for seed in SEEDS46:
    for arm in ARMS46:
        t0 = time.time()
        ck, wall_train, snaps = train46(seed, arm, TRAIN, STEPS46, GRID46, SAVE46)
        ev = Cards46(D, FF, arm).to(device)
        curve, hits_by_step = {}, {}
        for st in GRID46:
            ev.load_state_dict({k: v.to(device) for k, v in snaps[st].items()})
            hits = em_eval(ev, TEST[:N_EVAL])
            hits_by_step[st] = hits
            em = sum(hits) / len(hits)
            emc = sum(hits[i] for i in CLEAN_IDX) / len(CLEAN_IDX)
            seen = [hits[i] for i, r in enumerate(TEST[:N_EVAL]) if r["seen"]]
            uns = [hits[i] for i, r in enumerate(TEST[:N_EVAL]) if not r["seen"]]
            curve[st] = dict(em=em, em_clean=emc,
                             em_se=math.sqrt(em * (1 - em) / len(hits)),
                             seen=(sum(seen) / len(seen)) if seen else float("nan"),
                             unseen=(sum(uns) / len(uns)) if uns else float("nan"))
            print(f"  [S46-PT] {arm} seed={seed} step={st} EM全量={em*100:.2f}%±"
                  f"{curve[st]['em_se']*100:.2f} clean={emc*100:.2f}% "
                  f"SEEN={curve[st]['seen']*100:.2f}% UNSEEN={curve[st]['unseen']*100:.2f}%",
                  flush=True)
        fin = GRID46[-1]
        ev.load_state_dict({k: v.to(device) for k, v in snaps[fin].items()})
        _r28 = r28_selfcheck(ev, TEST[:N_EVAL], tag=f"[S46 {arm} s{fin} seed{seed}] ",
                             k=min(CHK_BATCH, N_EVAL))
        lo_s = min(len(r["p"]) for r in TEST[:32])
        pairs = [(0, 1), (2, 5), (1, lo_s - 3), (3, lo_s - 4), (4, lo_s // 2), (0, lo_s - 2)]
        pairs = sorted({(min(i, j), max(i, j)) for i, j in pairs})
        sw, lsc = swap_sensitivity(ev, TEST[:32], pairs)
        r29 = binding_r29(ev)
        SWAP[(arm, seed)] = dict(sw={f"{i}-{j}": v for (i, j), v in sw.items()}, lsc=lsc, r29=r29)
        best = max(sw.values(), key=lambda v: v["mx"])
        print(f"[S46-SWAP] {arm} seed={seed} 交换 {len(pairs)} 对（都在被读列之前, lo_s={lo_s}）| "
              f"max|Δlogits|={best['mx']:.3e}（相对 {best['rel']:.3e}）| 中位|Δ|="
              f"{best['med']:.3e} | Δ>1e-4 比例={best['nz4']*100:.1f}% | Δ>1e-6 比例="
              f"{best['nz6']*100:.1f}% | logits量级={lsc:.2f} | R29 绑定幅值比={r29['ratio']:.4f}",
              flush=True)
        RES[(arm, seed)] = dict(curve=curve, wall_train=wall_train, wall_total=time.time() - t0,
                                r28=_r28, ckpt=ck, ckpt_sha=sha16(ck), n_eval=N_EVAL)
        with open(JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(dict(arm=arm, seed=seed, curve={str(k): v for k, v in curve.items()},
                                    r28=_r28, swap={f"{i}-{j}": v for (i, j), v in sw.items()},
                                    lsc=lsc, r29=r29, ckpt=ck, ckpt_sha=sha16(ck),
                                    wall_train=wall_train), ensure_ascii=False) + "\n")
        del snaps, ev
        torch.cuda.empty_cache()
        print(f"[S46-RUN] {arm} seed={seed}: EM@{fin}全量={curve[fin]['em']*100:.2f}% "
              f"clean={curve[fin]['em_clean']*100:.2f}% | R28={_r28['same']}/{_r28['k']} | "
              f"训练墙钟={wall_train/60:.2f}min 本 run={RES[(arm, seed)]['wall_total']/60:.2f}min",
              flush=True)

# ============================================================================
# ⑥ 口径自检：CHK-EQ（绑定关=基线？）+ 锚点（①@12000 复现 S38）
# ============================================================================
print("\n[S46] ===== ★CHK-EQ：bound-off 包装 vs 原始 enc_layer（数值等价性）=====", flush=True)
ids_chk, _ = build_batch(TEST[:8])
m_ref = build_model(111, "base")
st_ref = {k: v.clone() for k, v in m_ref.state_dict().items()}
m_wrap = build_model(222, "bind")
_pre = "thought.self_attn."
_sd_map = {(f"{_pre}inner.{k[len(_pre):]}" if k.startswith(_pre) else k): v
           for k, v in st_ref.items()}
m_wrap.load_state_dict(_sd_map, strict=False)     # 基线权重逐位进 wrapper 的 inner MHA
for mod in m_wrap.modules():
    if isinstance(mod, BoundMHA):
        mod.arm = "off"
_wsd = m_wrap.state_dict()
d_sd = max(float((st_ref[k].float().cpu() - _wsd[(f"{_pre}inner." + k[len(_pre):])
               if k.startswith(_pre) else k].float().cpu()).abs().max()) for k in st_ref)
with torch.no_grad():
    m_ref.eval()
    m_wrap.eval()
    d_eval = float((m_ref.logits(ids_chk) - m_wrap.logits(ids_chk)).abs().max())
    st = torch.get_rng_state()
    cst = torch.cuda.get_rng_state_all()
    m_ref.train()
    torch.set_rng_state(st)
    torch.cuda.set_rng_state_all(cst)
    a = m_ref.logits(ids_chk)
    torch.set_rng_state(st)
    torch.cuda.set_rng_state_all(cst)
    m_wrap.train()
    d_train = float((a - m_wrap.logits(ids_chk)).abs().max())
print(f"[S46-CHK-EQ] 同权重下 bound-off 包装 vs 原层：eval max|Δlogits|={d_eval:.3e} | "
      f"train max|Δlogits|={d_train:.3e}（含 dropout 的 RNG 流）| 基线权重逐位差={d_sd:.3e}", flush=True)
_bind_fx = max(max(SWAP[("bind", s)]["sw"].values(), key=lambda v: v["mx"])["mx"] for s in SEEDS46)
print(f"[S46-CHK-EQ-VERDICT] 绑定效应={_bind_fx:.3e} ⇒ eval 路径差/绑定效应="
      f"{d_eval/_bind_fx:.2e}，train={d_train/_bind_fx:.2e}"
      f" ⇒ {'包装不引入混淆 ✓（②③ 的差异只来自绑定项）' if d_eval < _bind_fx/20 and d_train < _bind_fx/20 else '★路径差不可忽略，点名'}"
      f"（eval 差来自 fused 核 vs 手工核，①用默认快路径、②③被迫走手工路径）", flush=True)
del m_ref, m_wrap
torch.cuda.empty_cache()

print("\n[S46] ===== ★口径自检 3：锚点 ①@12000 必须复现 S38 w1 =====", flush=True)
ANCHOR_OK = True
for s in SEEDS46:
    cur_f, cur_c = RES[("base", s)]["curve"][GRID46[-1]]["em"], \
        RES[("base", s)]["curve"][GRID46[-1]]["em_clean"]
    ref = ANCHOR_S38[s]
    df, dc = (cur_f - ref["full"]) * 100, (cur_c - ref["clean"]) * 100
    ok = abs(df) <= 1.0 and abs(dc) <= 1.0
    ANCHOR_OK &= ok
    print(f"[S46-ANCHOR] base seed={s} @12000 全量={cur_f*100:.2f}% vs S38 {ref['full']*100:.2f}% "
          f"Δ={df:+.2f}pp | clean={cur_c*100:.2f}% vs {ref['clean']*100:.2f}% Δ={dc:+.2f}pp ⇒ "
          f"{'复现 ✓' if ok else '★>1pp，点名'}", flush=True)

# ============================================================================
# ⑦ (a) 置换敏感性表
# ============================================================================
print("\n[S46] ===== ★(a) 置换敏感性（只互换模因两整行，均在【被读列之前】；被读列 = target 列）=====",
      flush=True)
print("[S46-SWAP-TBL] arm | seed | max|Δlogits| | 中位|Δ| | 相对(max|Δ|/max|logits|) | "
      "Δ>1e-4 | Δ>1e-6 | 判定（基线量级 2–6e-6）", flush=True)
for arm in ARMS46:
    for s in SEEDS46:
        w = SWAP[(arm, s)]
        best = max(w["sw"].values(), key=lambda v: v["mx"])
        verdict = ("★绑定生效（≥1e-3）" if best["mx"] >= 1e-3 else
                   ("绑定生效但弱（≥1e-4）" if best["mx"] >= 1e-4 else "★未生效（仍 fp 噪声量级）"))
        print(f"[S46-SWAP-TBL] {arm} | {s} | {best['mx']:.3e} | {best['med']:.3e} | "
              f"{best['rel']:.3e} | {best['nz4']*100:.1f}% | {best['nz6']*100:.1f}% | {verdict}",
              flush=True)
print(f"[S46-R29] 绑定项相对 key 的幅值比（非零 ⇒ 绑定面有效）："
      + " | ".join(f"{arm}/s{s}={SWAP[(arm, s)]['r29']['ratio']:.4f}"
                   for arm in ARMS46 for s in SEEDS46), flush=True)

# ============================================================================
# ⑧ (b) 能力表 + Δ±SE 与判定
# ============================================================================
fin = GRID46[-1]
print(f"\n[S46] ===== ★(b) 能力 @{fin} 步（EM batch=1；clean n={len(CLEAN_IDX)}）=====", flush=True)
for s in SEEDS46:
    for arm in ARMS46:
        c = RES[(arm, s)]["curve"][fin]
        print(f"[S46-EM] seed={s} {arm}: 全量={c['em']*100:.2f}% clean={c['em_clean']*100:.2f}% "
              f"SEEN={c['seen']*100:.2f}% UNSEEN={c['unseen']*100:.2f}%", flush=True)
VERDICT = {}
for arm in ("bind", "rand"):
    for mode in ("em", "em_clean"):
        x = [RES[(arm, s)]["curve"][fin][mode] for s in SEEDS46]
        y = [RES[("base", s)]["curve"][fin][mode] for s in SEEDS46]
        m, se, ds = paired_delta(x, y)
        tag = "全量" if mode == "em" else "clean"
        both_pos = all(d > 0 for d in ds)
        both_neg = all(d < 0 for d in ds)
        if m >= 2 * se and both_pos:
            v = "提升（≥①+2SE 且 2 seed 同号）"
        elif abs(m) <= se:
            v = "不影响能力（|Δ|≤1SE）"
        elif m <= -2 * se:
            v = "有害（≤①−2SE）"
        else:
            v = "不显著（落在中间带）"
        VERDICT[(arm, tag)] = (m, se, ds, v)
        print(f"[S46-DELTA] {tag}: Δ({arm}−base)={m*100:+.2f}pp ± {se*100:.2f} "
              f"(逐 seed {['%+.2f' % (d*100) for d in ds]}) ⇒ {v}", flush=True)
m_b, se_b, ds_b, _v_b = VERDICT[("bind", "全量")]
m_r, se_r, ds_r, _v_r = VERDICT[("rand", "全量")]
print(f"[S46-DELTA-RAND] ③ vs ②（控制参数效应）：Δ(rand−bind)="
      f"{(m_r - m_b)*100:+.2f}pp ⇒ {'③ 显著差于 ② ✓（绑定的 p 是有效信号）' if m_r < m_b - se_r - se_b else '★③ 不差于 ② ⇒ 可能只是参数效应'}",
      flush=True)

# ============================================================================
# ⑨ (c) 饱和曲线 + 判定
# ============================================================================
print(f"\n[S46] ===== ★(c) 饱和曲线（@{GRID46[0]}..@{fin}）=====", flush=True)
for arm in ARMS46:
    for s in SEEDS46:
        row = " | ".join(f"@{st}={RES[(arm, s)]['curve'][st]['em']*100:.2f}%"
                         f"(c{RES[(arm, s)]['curve'][st]['em_clean']*100:.2f}%)" for st in GRID46)
        print(f"[S46-SAT-CURVE] {arm} seed={s}: {row}", flush=True)
if len(GRID46) >= 2:
    for s in SEEDS46:
        for arm in ("base", "bind"):
            inc = (RES[(arm, s)]["curve"][GRID46[-1]]["em"]
                   - RES[(arm, s)]["curve"][GRID46[-2]]["em"]) * 100
            print(f"[S46-SAT] {arm} seed={s}: {GRID46[-2]}→{GRID46[-1]} 增量={inc:+.2f}pp ⇒ "
                  f"{'★未饱和（≥1pp）⇒ 能力结论不可下' if inc >= 1.0 else '已平（<1pp）'}",
                  flush=True)

print("\n[S46] ===== 成本与 ckpt =====", flush=True)
for arm in ARMS46:
    for s in SEEDS46:
        r = RES[(arm, s)]
        print(f"[META] {arm} seed={s}: steps={STEPS46} ckpt={os.path.basename(r['ckpt'])} "
              f"sha256[:16]={r['ckpt_sha']} 训练墙钟={r['wall_train']/60:.2f}min "
              f"R28={r['r28']['same']}/{r['r28']['k']}", flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"D={D} FF={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} | EM口径=batch{EM_BATCH} "
      f"| TEST sha16={TEST_SHA} 泄漏={LEAK}/{len(TEST)} | 地板={FLOOR_1*100:.2f}%/{FLOOR_5*100:.2f}% "
      f"| 锚点复现={'是' if ANCHOR_OK else '否(点名)'} | CHK-EQ eval={d_eval:.2e} train={d_train:.2e} "
      f"| 总墙钟={(time.time()-t_all)/60:.1f}min | run 数={len(RES)}/{len(ARMS46)*len(SEEDS46)}",
      flush=True)
print("[DONE] exit=0", flush=True)
sys.exit(0)
