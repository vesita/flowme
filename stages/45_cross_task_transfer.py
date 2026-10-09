#!/usr/bin/env python3
"""S45 · ★P2 跨任务功能迁移（全项目零次）：在任务 A 上训好的目标卡，能不能直接用于任务 B？

为什么必须做：PROJECT_STATE §5 的 P2 = 全项目零次「跨任务功能迁移」，而「卡片可复用」是
卡片化的核心卖点。已有证据都【不是】功能迁移：S25 的锚点是【同任务】add_3d；S29 量的是
【表示相似度】(CKA) 而非功能可迁移；E1 已证卡边界不产生能力；S42/S43 已定 clean 口径；
R36 要求结论在 ≥12000 步。

设计（★只变【任务】，其余全同）：
  任务 A = add_3d（三位数加法，复用 S38 已入库 ckpt）；任务 B = add_2d（两位数加法，现训）；
  两张卡的分组（= S25 的 ABI）：输入卡 = emb+in_enc，目标卡 = thought(残差)，输出卡 = head；
  「接口」= 输入卡 + 输出卡（S25/S34 的定义），「目标卡」= 中间思维卡。

四臂（每臂 × 2 seed = 1234/5678）：
  ① B 自训（基线）      ：B 的接口 + B 的目标卡，从随机初值训 12000 步（end-to-end）；
  ② A 全量直插（零训练）：A 的接口 + A 的目标卡（= S38 ckpt 原样），在 B test 上前向；
  ③ A 卡 + B 接口（零训练）：B 的接口（来自 ①）+ A 的目标卡（来自 S38），不训；
  ③b A 接口 + B 卡（零训练，★额外一格凑满 2×2）：A 的接口 + B 的目标卡，不训；
  ④ A 卡微调 1000 步  ：B 的接口 + A 的目标卡，在 B 上训 1000 步（只训目标卡、冻结接口）。

★ 2×2 归因（零训练 EM on B）：I_A+T_A=② · I_B+T_A=③ · I_A+T_B=③b · I_B+T_B=①@12000
  ⇒ 「任务绑定」在【接口】还是【卡】上由这四格直接读出（卡主效应 / 接口主效应）。
※ 任务书表里 ② = A 接口+A 卡、判据括注 ② =「A 卡直接插 B 接口」= 本实现 ③ ⇒ 两读法都算、都报。

★ 口径（缺一即无效，全部照抄 S36/S38/S43/S44）：
  1 EM 一律 batch=1（R28）+ batch=1 vs 16 逐字一致性自检；
  2 ★两种口径都报：全量 test（n=800）与 clean 子集（nl 含字面量 {expr} 的坏样本剔除；
    add_3d 坏 30 → n=770、add_2d 坏 19 → n=781）—— 后者是 METRICS.md 权威口径；
  3 ★锚点复核：② 用的 A ckpt 是 add_3d 复用件 ⇒ 在 add_3d test 上必须复现 S38 的
    @12000 = 91.88/87.12（全量，batch=1）；
  4 地板用本桶 add_2d：单答案 / top5（S42 入库 0.57% / 2.75%）+ 机会水平；
  5 ★饱和检查（R35）：① 的 9000→12000 段 EM 增量 ≥1pp ⇒ 未饱和、结论不可下；
  6 ★跨任务迁移率 = EM(臂②/③/③b) / EM(臂①)；微调后恢复率 = EM(臂④)/EM(臂①)（两种口径都算）；
  7 R29 非零比例（目标卡输入置零的干预）+ TEST sha256[:16] 每桶分别报（add_3d 入库
    b34e7ea515203227）；SEEN/UNSEEN 分层。

判据（写死；全部以「① 已饱和」为前提）：
  迁移率 ≥0.8 ⇒ 卡可零成本跨任务复用（核心主张成立）；≤0.2 ⇒ 卡与任务强绑定；
  中间 ⇒ 报具体值；恢复率 ≥0.95 ⇒ 少量适配即可迁移；≤0.5 ⇒ 需从头训。

复用：exec stages/19_residual_scan.py 到 "# ③ 主循环" 之前（⇒ 同 tokenizer、五桶数据、
同 build_batch/greedy_gen(EM_BATCH=1)/parse_ans/classify/r28_selfcheck/mean_se）。
ckpt：logs/45_ckpt_a1_s12000_seed<s>.pt（①）、logs/45_ckpt_a2/a3/a3b_seed<s>.pt（②③③b 原样落盘）、
logs/45_ckpt_a4_ft1000_seed<s>.pt（④）；结果增量落盘 logs/45_results.jsonl。
只允许写：本文件、logs/p2_transfer.log、logs/45_*、/tmp；既有 stages/*.py 只读。
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

SMOKE45 = os.environ.get("S45_SMOKE") == "1"
if SMOKE45:
    os.environ["S19_SMOKE"] = "1"          # 让 S19 头只造小数据（冒烟）

# ============================================================================
# ① 逐字复用 S19 前半段（⇒ 同 tokenizer、同五桶 train4000/test800、同评估口径）
# ============================================================================
_src = open(SRC19, encoding="utf-8").read()
_cut = _src.index("# ③ 主循环")
NS = {"__name__": "s19_head", "__file__": SRC19}
exec(compile(_src[:_cut], SRC19, "exec"), NS)          # noqa: S102
for _k, _v in NS.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
del NS

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.optim import AdamW  # noqa: E402

T0 = time.time()
FP = 0.1
TASK_A = "add_3d"
TASK_B = "add_2d"
SEEDS45 = (1234,) if SMOKE45 else (1234, 5678)
STEPS1 = 40 if SMOKE45 else 12000
STEPS4 = 10 if SMOKE45 else 1000
GRID1 = tuple(range(5, STEPS1 + 1, 5)) if SMOKE45 else tuple(range(1500, 12001, 1500))
GRID4 = (5, 10) if SMOKE45 else (250, 500, 750, 1000)
REPORT_AT = tuple(st for st in (3000, 6000, 9000, 12000) if st in GRID1) or (GRID1[-1],)
N_EVAL = 60 if SMOKE45 else N_TEST                   # 800
B1_CK = f"{ROOT}/logs/45_ckpt_a1_s{STEPS1}_seed{{seed}}.pt"
B2_CK = f"{ROOT}/logs/45_ckpt_a2_seed{{seed}}.pt"
B3_CK = f"{ROOT}/logs/45_ckpt_a3_seed{{seed}}.pt"
B3B_CK = f"{ROOT}/logs/45_ckpt_a3b_seed{{seed}}.pt"
B4_CK = f"{ROOT}/logs/45_ckpt_a4_ft{STEPS4}_seed{{seed}}.pt"
if SMOKE45:
    B1_CK = "/tmp/45_smoke_a1_seed{seed}.pt"
    B2_CK = "/tmp/45_smoke_a2_seed{seed}.pt"
    B3_CK = "/tmp/45_smoke_a3_seed{seed}.pt"
    B3B_CK = "/tmp/45_smoke_a3b_seed{seed}.pt"
    B4_CK = "/tmp/45_smoke_a4_seed{seed}.pt"
JSONL = "/tmp/45_results_smoke.jsonl" if SMOKE45 else f"{ROOT}/logs/45_results.jsonl"
A_CKPT = {s: f"{ROOT}/logs/38_ckpt_w1_s12000_seed{s}.pt" for s in SEEDS45}
ANCHOR_A_S38 = {1234: 0.9188, 5678: 0.8712}          # S38 @12000 全量 EM（batch=1）
ANCHOR_A_S38_CLEAN = {1234: 0.9545, 5678: 0.9052}    # S43 clean（n=770）
TEST_SHA_EXPECT = {TASK_A: "b34e7ea515203227", TASK_B: None}
LEAK_EXPECT = {TASK_A: 16, TASK_B: None}
CLEAN_BAD_EXPECT = {TASK_A: 30, TASK_B: 19}
CLEAN_N_EXPECT = {TASK_A: 770, TASK_B: 781}
CLEAN_BAD_A_43 = [51, 113, 164, 197, 209, 230, 248, 249, 324, 370, 420, 438, 444, 463,
                  540, 550, 562, 583, 604, 608, 627, 634, 638, 653, 680, 682, 706, 712, 751, 762]
FLOOR42_B = dict(single_train=0.0057, single_test=0.0050, top5_test=0.0275, chance=0.0056)
ARM_DESC = {
    "①B自训": "B 接口 + B 目标卡，随机初值，12000 步 e2e",
    "②A全量": "A 接口 + A 目标卡（S38 ckpt 原样），零训练",
    "③A卡+B接口": "B 接口（①）+ A 目标卡（S38），零训练",
    "③bA接口+B卡": "A 接口（S38）+ B 目标卡（①），零训练",
    "④A卡微调": "B 接口（①）+ A 目标卡（S38），只训目标卡 1000 步",
}

assert str(device).startswith("cuda"), f"device={device} ⇒ 本单元必须在 cuda 上跑"
print(f"\n[S45] ★P2 跨任务功能迁移（全项目零次）：A={TASK_A} → B={TASK_B} | "
      f"① B 自训 {STEPS1} 步（曲线格 {GRID1[0]}..{GRID1[-1]}）| ④ 微调 {STEPS4} 步 | "
      f"seeds={SEEDS45} | d={D} ff={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} | "
      f"EM 主口径=batch{EM_BATCH} 自检=batch{CHK_BATCH} | smoke={SMOKE45}", flush=True)
print(f"[S45-DEV] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"HSA_OVERRIDE_GFX_VERSION={os.environ.get('HSA_OVERRIDE_GFX_VERSION')} | "
      f"V={V} PAD={PAD_ID} EOS={EOS_ID}", flush=True)

# ============================================================================
# ② 数据核对（口径 2/3/4/6/7 + 两桶 TEST 指纹 + clean 子集 + 泄漏）
# ============================================================================
TRAIN_A, TEST_A = DATA[TASK_A]["train"], DATA[TASK_A]["test"]
TRAIN_B, TEST_B = DATA[TASK_B]["train"], DATA[TASK_B]["test"]
TEST_A_EVAL, TEST_B_EVAL = TEST_A[:N_EVAL], TEST_B[:N_EVAL]


def test_sha16(recs) -> str:
    h = hashlib.sha256()
    for r in recs:
        h.update((r["nl"] + "||" + r["gold"] + "\n").encode())
    return h.hexdigest()[:16]


def seen_flags(train, test):
    tr = {(r["a"], r["b"]) for r in train}
    return [(r["a"], r["b"]) in tr for r in test], test


def clean_idx_of(test_eval, full_bad_n):
    bad = [i for i, r in enumerate(test_eval) if "{expr}" in r["nl"]]
    return bad, [i for i in range(len(test_eval)) if i not in set(bad)]


TEST_SHA = {TASK_A: test_sha16(TEST_A), TASK_B: test_sha16(TEST_B)}
SEEN_A, _ = seen_flags(TRAIN_A, TEST_A_EVAL)
SEEN_B, _ = seen_flags(TRAIN_B, TEST_B_EVAL)
LEAK = {TASK_A: sum(SEEN_A), TASK_B: sum(SEEN_B)}
BAD_A, CLEAN_A = clean_idx_of(TEST_A_EVAL, CLEAN_BAD_EXPECT[TASK_A])
BAD_B, CLEAN_B = clean_idx_of(TEST_B_EVAL, CLEAN_BAD_EXPECT[TASK_B])
N_CLEAN = {TASK_A: len(CLEAN_A), TASK_B: len(CLEAN_B)}

_c_tr = Counter(r["gold"] for r in TRAIN_B)
_top1, _top1n = _c_tr.most_common(1)[0]
_top5 = [v for v, _ in _c_tr.most_common(5)]
FLOOR_1_TRAIN = _top1n / len(TRAIN_B)
FLOOR_1_TEST = sum(1 for r in TEST_B_EVAL if r["gold"] == _top1) / len(TEST_B_EVAL)
FLOOR_5_TEST = sum(1 for r in TEST_B_EVAL if r["gold"] in _top5) / len(TEST_B_EVAL)

sha_ok = {b: (TEST_SHA_EXPECT[b] is None or TEST_SHA[b] == TEST_SHA_EXPECT[b])
          for b in (TASK_A, TASK_B)}
leak_ok = {b: (LEAK_EXPECT[b] is None or LEAK[b] == LEAK_EXPECT[b]) for b in (TASK_A, TASK_B)}
clean_ok = (len(BAD_A) == CLEAN_BAD_EXPECT[TASK_A] and BAD_A == CLEAN_BAD_A_43
            and len(CLEAN_A) == CLEAN_N_EXPECT[TASK_A]
            and len(BAD_B) == CLEAN_BAD_EXPECT[TASK_B] and len(CLEAN_B) == CLEAN_N_EXPECT[TASK_B])
floor_ok = (abs(FLOOR_1_TRAIN - FLOOR42_B["single_train"]) <= 0.001
            and abs(FLOOR_1_TEST - FLOOR42_B["single_test"]) <= 0.001
            and abs(FLOOR_5_TEST - FLOOR42_B["top5_test"]) <= 0.002)

for b in (TASK_A, TASK_B):
    print(f"[S45-DATA] {b}: train={len(DATA[b]['train'])} test={len(DATA[b]['test'])} 文本零重叠 | "
          f"答案域={min(answer_values(b))}..{max(answer_values(b))}"
          f"({len(answer_values(b))}个) | ★不切 dev ⇒ 训练集={len(DATA[b]['train'])}（与 S19/S36/S38 同源）",
          flush=True)
    print(f"[S45-CHK6] {b} TEST 指纹 sha256[:16]={TEST_SHA[b]} vs 入库 "
          f"{TEST_SHA_EXPECT[b] or '(无入库值，本次即指纹)'} ⇒ "
          f"{'逐位一致 ✓' if sha_ok[b] else '★不一致'} | (a,b) 泄漏 train→test={LEAK[b]} "
          f"({100.0*LEAK[b]/len(DATA[b]['test']):.2f}%) vs 入库 {LEAK_EXPECT[b] or '-'} ⇒ "
          f"{'一致 ✓' if leak_ok[b] else '★不一致'}", flush=True)
    print(f"[S45-CHK7] {b} clean 子集：剔除 nl 含字面量 '{{expr}}' 的 {len(BAD_A if b==TASK_A else BAD_B)} 条 "
          f"⇒ n_clean={N_CLEAN[b]}（入库期望 {CLEAN_N_EXPECT[b]}）| 坏样本 {b} only-bug 在 S43 实测 EM=0%",
          flush=True)
print(f"[S45-CHK2] SEEN/UNSEEN（train4000 对 test 的 (a,b) 泄漏）：A={LEAK[TASK_A]}/800、"
      f"B={LEAK[TASK_B]}/800", flush=True)
print(f"[S45-CHK4] ★地板（本桶 {TASK_B}）：单答案 '{_top1}'(train {_top1n}/{len(TRAIN_B)}) "
      f"train 先验={FLOOR_1_TRAIN*100:.2f}%/test 覆盖={FLOOR_1_TEST*100:.2f}% | top5={_top5} "
      f"test 覆盖={FLOOR_5_TEST*100:.2f}% | 机会水平=1/{len(answer_values(TASK_B))}="
      f"{100.0/len(answer_values(TASK_B)):.2f}% | S42 入库=0.57%/0.50%/2.75%/0.56% ⇒ "
      f"{'逐位复现 ✓' if floor_ok else '★与入库有差，点名'}", flush=True)
if not SMOKE45:
    assert all(sha_ok.values()), "TEST 指纹不符 ⇒ 与 S19/S36/S38 不是同一数据集，本单元无效"
    assert clean_ok, "clean 索引与 S43 不一致 ⇒ 口径 2 无效"
    assert floor_ok, "地板与 S42 入库不一致 ⇒ 口径 4 无效"

# ============================================================================
# ③ 模型：与 S38 Cards38 / S44 Cards44 逐字同构（emb → in_enc → card(残差) → head）
# ============================================================================
class CardA45(nn.Module):
    def __init__(self, d, ff):
        super().__init__()
        self.layer = enc_layer(d, ff)

    def forward(self, h, m):
        return h + self.layer(h, src_mask=m)


HOOK_IN = {"fn": None}


class Cards45(nn.Module):
    """臂 = A（h + Think(h)），与 S38 的 thought 臂逐位同构 ⇒ 可直接灌 S38 ckpt。"""

    def __init__(self, d, ff):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        self.card = CardA45(d, ff)
        self.head = nn.Linear(d, V)
        self.register_buffer("pe", sin_pe(MAXLEN, d), persistent=False)

    def logits(self, ids):
        n = ids.size(1)
        m = torch.full((n, n), float("-inf"), device=ids.device)
        m = torch.triu(m, diagonal=1)
        m = m.unsqueeze(0).expand(ids.size(0), -1, -1).clone()
        m.masked_fill_((ids == PAD_ID).unsqueeze(1).expand_as(m), float("-inf"))
        m = m.repeat_interleave(NHEAD, dim=0)
        pe = self.pe if n <= self.pe.size(0) else sin_pe(n, self.d).to(self.pe.device)
        assert n <= pe.size(0), f"PE 长度 {pe.size(0)} < 序列长 {n}"
        x = self.emb(ids) + pe[:n].unsqueeze(0)
        h = self.in_enc(x, src_mask=m)
        if HOOK_IN["fn"] is not None:
            h = HOOK_IN["fn"](h)
        h = self.card(h, m)
        return self.head(h)


class Cards38Ref(nn.Module):
    """S38 的计算图（thought 命名），只用于证明「Cards45 ≡ S38 ckpt 的同一计算图」。"""

    def __init__(self, d, ff):
        super().__init__()
        self.d = d
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff)
        self.thought = enc_layer(d, ff)
        self.head = nn.Linear(d, V)
        self.register_buffer("pe", sin_pe(MAXLEN, d), persistent=False)


def group_of45(name):
    if name.startswith("emb.") or name.startswith("in_enc."):
        return "输入卡"
    if name.startswith("card.layer."):
        return "目标卡"
    if name.startswith("head."):
        return "输出卡"
    raise KeyError(f"未知参数分组：{name}")


INTERFACE_GROUPS = ("输入卡", "输出卡")
GROUP_ORDER = ("输入卡", "目标卡", "输出卡")


def load_sd_cpu(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def sd_a(path):
    """S38 ckpt 的 thought.* → Cards45 的 card.layer.*。"""
    return {k.replace("thought.", "card.layer."): v for k, v in load_sd_cpu(path).items()}


def pick(sd, which):
    if which == "intf":
        return {k: v for k, v in sd.items()
                if k.startswith("emb.") or k.startswith("in_enc.") or k.startswith("head.")}
    return {k: v for k, v in sd.items() if k.startswith("card.layer.")}


def make_model(sd=None):
    m = Cards45(D, FF)
    if sd is not None:
        m.load_state_dict(sd, strict=True)
    return m.to(device)


def n_params_45():
    m = Cards45(D, FF)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


# ---- ★一致性自检：Cards45 与 S38 是同一计算图（构造顺序相同 ⇒ 同 seed 初始化逐位相同）----
torch.manual_seed(999)
_m45 = Cards45(D, FF)
torch.manual_seed(999)
_m38 = Cards38Ref(D, FF)
_s45, _s38 = _m45.state_dict(), _m38.state_dict()
_map = {k: k.replace("card.layer.", "thought.") for k in _s45}
INIT_DIFF = max(float((_s45[k] - _s38[_map[k]]).abs().max()) for k in _s45)
del _m45, _m38
print(f"[S45-INIT] Cards45 与 S38 Cards38 同 seed(999) 初始化逐参数最大差={INIT_DIFF:.3e} "
      f"⇒ {'✓ 同一计算图（可逐位复用 S38 ckpt）' if INIT_DIFF == 0.0 else '★不同起点'}",
      flush=True)
if not SMOKE45:
    assert INIT_DIFF == 0.0, "Cards45 与 S38 不是同一计算图 ⇒ 复用无效"
N_PARAMS = n_params_45()
PARAMS_GROUPS: dict = {}
for _n, _p in Cards45(D, FF).named_parameters():
    PARAMS_GROUPS[group_of45(_n)] = PARAMS_GROUPS.get(group_of45(_n), 0) + _p.numel()
print(f"[S45-PARAMS] 总参={N_PARAMS/1e6:.3f}M | " + " | ".join(
    f"{g}={PARAMS_GROUPS[g]/1e6:.3f}M" for g in GROUP_ORDER)
    + f"（接口={sum(PARAMS_GROUPS[g] for g in INTERFACE_GROUPS)/1e6:.3f}M）", flush=True)


# ============================================================================
# ④ 训练：① 全参 e2e（每格快照）/ ④ 只训目标卡（冻结接口，门 A 校验）
# ============================================================================
def train45_full(seed, train, steps, grid, ckpt_path):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = Cards45(D, FF).to(device)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0 = 0, 0, time.time()
    last, snaps = 0.0, {}
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
        if step % 1500 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train:①] step={step}/{steps} loss={loss.item():.4f} elapsed={el:.0f}s "
                  f"s/step={(el - last) / (step % 1500 or 1500):.3f}", flush=True)
            last = el
    assert set(grid) <= set(snaps), "未取到全部曲线格点快照"
    torch.save(snaps[steps], ckpt_path)
    return model, time.time() - t0, snaps


def train45_card_only(seed, train, steps, grid, init_sd, ckpt_path):
    """④：灌入初始 sd，冻接口（输入卡+输出卡）、只训目标卡；与 S25/S34 逐字同纪律。"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    model = make_model(init_sd)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    for n, p in model.named_parameters():
        p.requires_grad_(group_of45(n) == "目标卡")
    params = [p for p in model.parameters() if p.requires_grad]
    n_tgt = sum(p.numel() for p in params)
    opt = AdamW(params, lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0 = 0, 0, time.time()
    last, snaps, ngrad = 0.0, {}, 0
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
        ngrad += sum(1 for p in params if p.grad is not None)
        opt.step()
        step += 1
        if step in gset:
            snaps[step] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if step % 250 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train:④] step={step}/{steps} loss={loss.item():.4f} elapsed={el:.0f}s", flush=True)
            last = el
    assert set(grid) <= set(snaps), "未取到全部曲线格点快照"
    # 门 A：目标卡有梯度且更新 >0；接口逐位不动（精确）且 grad is None
    gate = {}
    for g in GROUP_ORDER:
        mx, ng = 0.0, 0
        for n, p in model.named_parameters():
            if group_of45(n) != g:
                continue
            mx = max(mx, float((before[n] - p.detach()).abs().max()))
            ng += 1 if p.grad is not None else 0
        gate[g] = dict(max_delta=mx, n_grad=ng)
    if not SMOKE45:
        assert gate["目标卡"]["max_delta"] > 0 and ngrad > 0, "门A失败：目标卡未更新/无梯度"
        for g in INTERFACE_GROUPS:
            assert gate[g]["max_delta"] == 0.0 and gate[g]["n_grad"] == 0, f"门A失败：{g} 被改动"
    torch.save(snaps[steps], ckpt_path)
    return model, time.time() - t0, snaps, gate, n_tgt


# ============================================================================
# ⑤ 评估（EM 主口径 batch=1；CE/数字每步与 S19/S44 同定义）
# ============================================================================
@torch.no_grad()
def em_at(model, recs):
    txt = greedy_gen(model, recs, batch=EM_BATCH)          # ★R28 主口径 batch=1
    return [int(parse_ans(t) == r["gold"]) for t, r in zip(txt, recs)], txt


@torch.no_grad()
def per_sample_ce(model, recs, hook=None):
    model.eval()
    HOOK_IN["fn"] = hook
    all_ce, dig_ce = [], []
    try:
        for i in range(0, len(recs), GEN_BATCH):
            chunk = recs[i: i + GEN_BATCH]
            ids, s = build_batch(chunk)
            logp = torch.log_softmax(model.logits(ids), dim=-1)
            for j, r in enumerate(chunk):
                e = s[j] + len(r["t"])
                lp = logp[j, s[j] - 1: e - 1]
                tgt = ids[j, s[j]: e]
                ces = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1).tolist()
                all_ce.append(sum(ces) / len(ces))
                ds = [v for p, v in enumerate(ces) if classify(int(tgt[p])) == 0]
                dig_ce.append(sum(ds) / len(ds) if ds else float("nan"))
    finally:
        HOOK_IN["fn"] = None
    model.train()
    return all_ce, dig_ce


def dig_acc(dig_ce):
    ys = [x for x in dig_ce if x == x]
    m, se = mean_se(ys)
    acc = math.exp(-m)
    return acc, acc * se, m, se, len(ys)


def iv_zero(x, *rest):
    return torch.zeros_like(x)


def r29_nz(model, recs):
    """R29：目标卡【输入】置零的干预，数字每步变化的样本占比（S44 同定义）。"""
    _, dig_iv = per_sample_ce(model, recs, hook=iv_zero)
    _, dig_b = per_sample_ce(model, recs)
    a_iv = [math.exp(-x) if x == x else float("nan") for x in dig_iv]
    a_b = [math.exp(-x) if x == x else float("nan") for x in dig_b]
    return frac([1.0 if (x == x and y == y and abs(x - y) > 1e-3) else 0.0
                 for x, y in zip(a_iv, a_b)])


def sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def add_jsonl(obj):
    with open(JSONL, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def frac(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def sub(xs, idx):
    return [xs[i] for i in idx]


def hit_se(hits):
    p = frac(hits)
    return p, math.sqrt(p * (1 - p) / len(hits))


@torch.no_grad()
def full_eval(tag, sd, recs, clean_idx, seen=None, extras=True):
    model = make_model(sd)
    hits, txt = em_at(model, recs)
    all_ce, dig_ce = per_sample_ce(model, recs)
    acc, acc_se, dce, dce_se, dn = dig_acc(dig_ce)
    p, se = hit_se(hits)
    pc, sec = hit_se(sub(hits, clean_idx))
    r = dict(em_full=p, em_se=se, em_clean=pc, em_clean_se=sec, acc=acc, acc_se=acc_se,
             dig_ce=dce, ce=frac(all_ce), hits=hits, texts=txt, n=len(recs),
             n_clean=len(clean_idx))
    if seen is not None:
        hs = [hits[i] for i in range(len(recs)) if seen[i]]
        hu = [hits[i] for i in range(len(recs)) if not seen[i]]
        r.update(em_seen=frac(hs), em_unseen=frac(hu), n_seen=len(hs), n_unseen=len(hu))
    if extras:
        rr = r28_selfcheck(model, recs, tag=f"[{tag}] ", k=min(CHK_BATCH, len(recs)))
        r["r28_same"], r["r28_k"] = rr["same"], rr["k"]
        r["r29_iv_nz"] = r29_nz(model, recs)
    del model
    torch.cuda.empty_cache()
    return r


RESULTS: dict = {}
open(JSONL, "w").close()

# ============================================================================
# ⑥ 臂①（B 自训，基线）：12000 步 + 每 1500 步快照 ⇒ 饱和曲线
# ============================================================================
for seed in SEEDS45:
    t0 = time.time()
    print(f"\n[S45] ===== 臂① {TASK_B} 自训（基线）seed={seed} steps={STEPS1} | "
          f"train={len(TRAIN_B)} test={len(TEST_B_EVAL)} | {ARM_DESC['①B自训']} =====", flush=True)
    model, wall_train, snaps = train45_full(seed, TRAIN_B, STEPS1, GRID1, B1_CK.format(seed=seed))
    ck = B1_CK.format(seed=seed)
    r = dict(arm="①B自训", seed=seed, ckpt=ck, sha=sha16(ck), wall_train=wall_train, curve={})
    ev = Cards45(D, FF).to(device)
    for st in GRID1:
        ev.load_state_dict({k: v.to(device) for k, v in snaps[st].items()}, strict=True)
        ev.eval()
        hits_st, _ = em_at(ev, TEST_B_EVAL)
        r["curve"][st] = dict(em_full=frac(hits_st), em_clean=frac(sub(hits_st, CLEAN_B)))
        if st in REPORT_AT:
            print(f"  [S45-PT] ① seed={seed} step={st} EM全量={frac(hits_st)*100:.2f}% "
                  f"clean={frac(sub(hits_st, CLEAN_B))*100:.2f}%", flush=True)
    del ev, snaps, model
    torch.cuda.empty_cache()
    fin = full_eval(f"①B自训 s{seed}", load_sd_cpu(ck), TEST_B_EVAL, CLEAN_B, seen=SEEN_B)
    r.update({k: v for k, v in fin.items() if k not in ("hits", "texts")})
    r["hits"], r["texts"] = fin["hits"], fin["texts"]
    r["wall_total"] = time.time() - t0
    RESULTS[("①B自训", seed)] = r
    add_jsonl({k: v for k, v in r.items() if k not in ("hits", "texts")})
    print(f"[S45-MAIN] ①B自训 seed={seed} | EM全量={r['em_full']*100:.2f}%±{r['em_se']*100:.2f}"
          f"(n={r['n']}) clean={r['em_clean']*100:.2f}%(n={r['n_clean']}) | "
          f"SEEN={r['em_seen']*100:.2f}%({r['n_seen']}) UNSEEN={r['em_unseen']*100:.2f}%"
          f"({r['n_unseen']}) | 数字每步={r['acc']*100:.2f}%±{r['acc_se']*100:.2f} | "
          f"R29非零比={r['r29_iv_nz']*100:.1f}% | R28={r['r28_same']}/{r['r28_k']} | "
          f"训练={wall_train/60:.2f}min 本run={r['wall_total']/60:.2f}min | sha16={r['sha']}",
          flush=True)

# ============================================================================
# ⑦ 臂②③③b（零训练组合）+ ★锚点复核（② 在 add_3d test 上）
# ============================================================================
I_B = {s: pick(load_sd_cpu(B1_CK.format(seed=s)), "intf") for s in SEEDS45}
T_B = {s: pick(load_sd_cpu(B1_CK.format(seed=s)), "card") for s in SEEDS45}
I_A = {s: pick(sd_a(A_CKPT[s]), "intf") for s in SEEDS45}
T_A = {s: pick(sd_a(A_CKPT[s]), "card") for s in SEEDS45}

ZERO_ARMS = {
    "②A全量": lambda s: {**I_A[s], **T_A[s]},
    "③A卡+B接口": lambda s: {**I_B[s], **T_A[s]},
    "③bA接口+B卡": lambda s: {**I_A[s], **T_B[s]},
}
XFER_CK = {"②A全量": B2_CK, "③A卡+B接口": B3_CK, "③bA接口+B卡": B3B_CK}

for s in SEEDS45:
    print(f"\n[S45] ===== ★锚点复核：② 用的 A ckpt（S38 add_3d @12000, seed={s}）在 add_3d test 上 =====",
          flush=True)
    fin_a = full_eval(f"②Ackpt(a3d) s{s}", {**I_A[s], **T_A[s]}, TEST_A_EVAL, CLEAN_A,
                      seen=SEEN_A, extras=True)
    dd = (fin_a["em_full"] - ANCHOR_A_S38[s]) * 100
    ddc = (fin_a["em_clean"] - ANCHOR_A_S38_CLEAN[s]) * 100
    print(f"[S45-ANCHOR] seed={s}: 本次@12000 add_3d EM全量={fin_a['em_full']*100:.2f}% vs S38 入库="
          f"{ANCHOR_A_S38[s]*100:.2f}% ⇒ Δ={dd:+.2f}pp {'✓ 复现（≤1pp）' if abs(dd) <= 1.0 else '★>1pp，点名'}"
          f" | clean={fin_a['em_clean']*100:.2f}% vs S43 入库={ANCHOR_A_S38_CLEAN[s]*100:.2f}% "
          f"⇒ Δ={ddc:+.2f}pp | sha16(ckpt)={sha16(A_CKPT[s])} | R28={fin_a['r28_same']}/{fin_a['r28_k']} "
          f"| R29非零比={fin_a['r29_iv_nz']*100:.1f}%", flush=True)
    RESULTS[("anchorA", s)] = dict(em_full=fin_a["em_full"], em_clean=fin_a["em_clean"], dev=dd)

for arm, builder in ZERO_ARMS.items():
    for seed in SEEDS45:
        t0 = time.time()
        sd = builder(seed)
        ck = XFER_CK[arm].format(seed=seed)
        torch.save(sd, ck)
        r = dict(arm=arm, seed=seed, ckpt=ck, sha=sha16(ck), wall_train=0.0)
        print(f"\n[S45] ===== 臂 {arm} seed={seed}（零训练）| {ARM_DESC[arm]} =====", flush=True)
        fin = full_eval(f"{arm} s{seed}", sd, TEST_B_EVAL, CLEAN_B, seen=SEEN_B)
        r.update({k: v for k, v in fin.items() if k not in ("hits", "texts")})
        r["hits"], r["texts"] = fin["hits"], fin["texts"]
        r["wall_total"] = time.time() - t0
        base_txt = RESULTS[("①B自训", seed)]["texts"]
        r["r29_swap_nz"] = frac([1.0 if a != b else 0.0 for a, b in zip(fin["texts"], base_txt)])
        RESULTS[(arm, seed)] = r
        add_jsonl({k: v for k, v in r.items() if k not in ("hits", "texts")})
        print(f"[S45-MAIN] {arm} seed={seed} | EM全量={r['em_full']*100:.2f}%±{r['em_se']*100:.2f}"
              f"(n={r['n']}) clean={r['em_clean']*100:.2f}%(n={r['n_clean']}) | "
              f"SEEN={r['em_seen']*100:.2f}%({r['n_seen']}) UNSEEN={r['em_unseen']*100:.2f}%"
              f"({r['n_unseen']}) | 数字每步={r['acc']*100:.2f}%±{r['acc_se']*100:.2f} | "
              f"R29非零比={r['r29_iv_nz']*100:.1f}%(置零) / {r['r29_swap_nz']*100:.1f}%(换件) | "
              f"R28={r['r28_same']}/{r['r28_k']} | sha16={r['sha']}", flush=True)

# ============================================================================
# ⑧ 臂④（B 接口 + A 卡，只训目标卡 1000 步）
# ============================================================================
for seed in SEEDS45:
    t0 = time.time()
    init_sd = ZERO_ARMS["③A卡+B接口"](seed)
    ck = B4_CK.format(seed=seed)
    print(f"\n[S45] ===== 臂④ A 卡微调 seed={seed} steps={STEPS4}（冻结接口=①的 B 接口）| "
          f"{ARM_DESC['④A卡微调']} =====", flush=True)
    model, wall_train, snaps, gate, n_tgt = train45_card_only(
        seed, TRAIN_B, STEPS4, GRID4, init_sd, ck)
    gate_line = " | ".join(f"{g}: max|Δθ|={gate[g]['max_delta']:.3e} grad≠None({gate[g]['n_grad']})"
                           for g in GROUP_ORDER)
    print(f"[S45-门A] ④ seed={seed} ⇒ {gate_line} | 可训参数={n_tgt/1e6:.3f}M", flush=True)
    r = dict(arm="④A卡微调", seed=seed, ckpt=ck, sha=sha16(ck), wall_train=wall_train,
             curve={}, gate=gate, n_trainable=n_tgt)
    ev = Cards45(D, FF).to(device)
    for st in GRID4:
        ev.load_state_dict({k: v.to(device) for k, v in snaps[st].items()}, strict=True)
        ev.eval()
        hits_st, _ = em_at(ev, TEST_B_EVAL)
        r["curve"][st] = dict(em_full=frac(hits_st), em_clean=frac(sub(hits_st, CLEAN_B)))
        print(f"  [S45-PT] ④ seed={seed} step={st} EM全量={frac(hits_st)*100:.2f}% "
              f"clean={frac(sub(hits_st, CLEAN_B))*100:.2f}%", flush=True)
    del ev, snaps, model
    torch.cuda.empty_cache()
    fin = full_eval(f"④A卡微调 s{seed}", load_sd_cpu(ck), TEST_B_EVAL, CLEAN_B, seen=SEEN_B)
    r.update({k: v for k, v in fin.items() if k not in ("hits", "texts")})
    r["hits"], r["texts"] = fin["hits"], fin["texts"]
    base_txt = RESULTS[("①B自训", seed)]["texts"]
    r["r29_swap_nz"] = frac([1.0 if a != b else 0.0 for a, b in zip(fin["texts"], base_txt)])
    r["wall_total"] = time.time() - t0
    RESULTS[("④A卡微调", seed)] = r
    add_jsonl({k: v for k, v in r.items() if k not in ("hits", "texts")})
    print(f"[S45-MAIN] ④A卡微调 seed={seed} | EM全量={r['em_full']*100:.2f}%±{r['em_se']*100:.2f}"
          f"(n={r['n']}) clean={r['em_clean']*100:.2f}%(n={r['n_clean']}) | "
          f"SEEN={r['em_seen']*100:.2f}%({r['n_seen']}) UNSEEN={r['em_unseen']*100:.2f}%"
          f"({r['n_unseen']}) | 数字每步={r['acc']*100:.2f}%±{r['acc_se']*100:.2f} | "
          f"R29非零比={r['r29_iv_nz']*100:.1f}%(置零) / {r['r29_swap_nz']*100:.1f}%(换件) | "
          f"R28={r['r28_same']}/{r['r28_k']} | 训练={wall_train/60:.2f}min 本run={r['wall_total']/60:.2f}min "
          f"| sha16={r['sha']}", flush=True)

# ============================================================================
# ⑨ 口径自检汇总 1–7
# ============================================================================
ALL_ARMS = ("①B自训", "②A全量", "③A卡+B接口", "③bA接口+B卡", "④A卡微调")
print("\n[S45] ===== ★口径自检 1–7 =====", flush=True)
_r28 = [(a, s, RESULTS[(a, s)]["r28_same"], RESULTS[(a, s)]["r28_k"])
        for a in ALL_ARMS for s in SEEDS45 if (a, s) in RESULTS]
_r28_exact = sum(1 for x in _r28 if x[2] == x[3])
print(f"[S45-CHK1] EM 主口径=batch{EM_BATCH}；batch=1 vs batch{CHK_BATCH} 逐字一致性："
      f"{_r28_exact}/{len(_r28)} 个 (臂,seed) 完全一致；逐样本 "
      f"{sum(x[2] for x in _r28)}/{sum(x[3] for x in _r28)} 条逐字相同 ⇒ "
      f"{'全一致 ✓' if _r28_exact == len(_r28) else '★有不一致（本单元 EM 全部走 batch=1，口径自洽）'}",
      flush=True)
print(f"[S45-CHK2] SEEN/UNSEEN 分层见主表 | B 的 (a,b) 泄漏={LEAK[TASK_B]}/800 "
      f"（SEEN={sum(SEEN_B)} UNSEEN={800-sum(SEEN_B)}）", flush=True)
_a_ok = all(abs(RESULTS[("anchorA", s)]["dev"]) <= 1.0 for s in SEEDS45)
print(f"[S45-CHK3] ★锚点复核：A ckpt 在 add_3d test（S38 @12000=91.88/87.12 全量，batch=1）⇒ "
      f"{'全部复现 ✓（≤1pp）' if _a_ok else '★有 >1pp，已点名'}", flush=True)
print(f"[S45-CHK4] 地板（本桶 {TASK_B}）：单答案={FLOOR_1_TRAIN*100:.2f}%(train)/"
      f"{FLOOR_1_TEST*100:.2f}%(test) top5={FLOOR_5_TEST*100:.2f}%(test) 机会水平="
      f"{100.0/len(answer_values(TASK_B)):.2f}% | S42 入库 0.57/0.50/2.75/0.56%", flush=True)
print(f"[S45-CHK5] ★饱和检查（R35）见下方 [S45-SAT]（① 的 9000→12000 段增量 <1pp 才算已饱和）",
      flush=True)
print(f"[S45-CHK6] TEST sha256[:16]：{TASK_A}={TEST_SHA[TASK_A]}（入库 {TEST_SHA_EXPECT[TASK_A]}）| "
      f"{TASK_B}={TEST_SHA[TASK_B]}（无入库值，本次即指纹，与 S19 五桶同源）| R29 Δ非零比见主表",
      flush=True)
print(f"[S45-CHK7] clean 口径：{TASK_A} n={N_CLEAN[TASK_A]}(坏 {len(BAD_A)})、"
      f"{TASK_B} n={N_CLEAN[TASK_B]}(坏 {len(BAD_B)}) ⇒ 与 S43 入库逐位一致 ✓", flush=True)

# ============================================================================
# ⑩ 主表 + 迁移率/恢复率 + 2×2 归因 + 逐条判定 + 饱和曲线
# ============================================================================
print("\n[S45] ===== ★主表：五臂 × 2 seed（EM 主口径 batch=1，全量 n=800 / clean n=781）=====",
      flush=True)
print("[S45-TABLE] 臂 | seed | EM全量 | EMclean | 数字每步 | SEEN | UNSEEN | R29(置零/换件) | 训练min | sha16",
      flush=True)
for a in ALL_ARMS:
    for s in SEEDS45:
        r = RESULTS.get((a, s))
        if not r:
            continue
        print(f"[S45-TABLE] {a} | {s} | {r['em_full']*100:.2f}% | {r['em_clean']*100:.2f}% | "
              f"{r['acc']*100:.2f}%±{r['acc_se']*100:.2f} | {r['em_seen']*100:.2f}% | "
              f"{r['em_unseen']*100:.2f}% | {r['r29_iv_nz']*100:.1f}%/{r.get('r29_swap_nz', float('nan'))*100:.1f}% | "
              f"{r['wall_train']/60:.2f} | {r['sha']}", flush=True)

print("\n[S45] ===== ★跨任务迁移率 = EM(臂) / EM(①)（逐 seed + 2 seed 均值；两种口径）=====",
      flush=True)
RATE: dict = {}
for arm in ("②A全量", "③A卡+B接口", "③bA接口+B卡", "④A卡微调"):
    RATE[arm] = {}
    for tag in ("FULL", "CLEAN"):
        rs = []
        for s in SEEDS45:
            e0 = RESULTS[("①B自训", s)]["em_full" if tag == "FULL" else "em_clean"]
            e1 = RESULTS[(arm, s)]["em_full" if tag == "FULL" else "em_clean"]
            rs.append(e1 / e0 if e0 > 0 else float("nan"))
        RATE[arm][tag] = rs
        print(f"[S45-XFER] 迁移率({arm}) [{tag}] seed逐值="
              f"{['%.3f' % x for x in rs]} 均值={frac(rs):.3f}", flush=True)

print("\n[S45] ===== ★2×2 零训练归因（EM on B 全量，2 seed 均值）：接口 × 目标卡 =====", flush=True)
Z_AA = frac([RESULTS[("②A全量", s)]["em_full"] for s in SEEDS45])
Z_BA = frac([RESULTS[("③A卡+B接口", s)]["em_full"] for s in SEEDS45])
Z_AB = frac([RESULTS[("③bA接口+B卡", s)]["em_full"] for s in SEEDS45])
Z_BB = frac([RESULTS[("①B自训", s)]["em_full"] for s in SEEDS45])
Z_AA_c = frac([RESULTS[("②A全量", s)]["em_clean"] for s in SEEDS45])
Z_BA_c = frac([RESULTS[("③A卡+B接口", s)]["em_clean"] for s in SEEDS45])
Z_AB_c = frac([RESULTS[("③bA接口+B卡", s)]["em_clean"] for s in SEEDS45])
Z_BB_c = frac([RESULTS[("①B自训", s)]["em_clean"] for s in SEEDS45])
print(f"[S45-2x2] 全量 (行=接口, 列=目标卡)：I_A/T_A={Z_AA*100:.2f}% I_A/T_B={Z_AB*100:.2f}% | "
      f"I_B/T_A={Z_BA*100:.2f}% I_B/T_B={Z_BB*100:.2f}%", flush=True)
print(f"[S45-2x2] clean：I_A/T_A={Z_AA_c*100:.2f}% I_A/T_B={Z_AB_c*100:.2f}% | "
      f"I_B/T_A={Z_BA_c*100:.2f}% I_B/T_B={Z_BB_c*100:.2f}%", flush=True)
print(f"[S45-2x2] ★接口主效应（卡=A）：I_B−I_A = {(Z_BA-Z_AA)*100:+.2f}pp(全量) / "
      f"{(Z_BA_c-Z_AA_c)*100:+.2f}pp(clean) | ★卡主效应（接口=B）：T_B−T_A = "
      f"{(Z_BB-Z_BA)*100:+.2f}pp(全量) / {(Z_BB_c-Z_BA_c)*100:+.2f}pp(clean)", flush=True)
print(f"[S45-2x2] 换卡代价（接口=A，T_A→T_B）={(Z_AB-Z_AA)*100:+.2f}pp | "
      f"换接口代价（卡=B，I_A→I_B）={(Z_BB-Z_AB)*100:+.2f}pp", flush=True)

print("\n[S45] ===== ★逐条判定（写死）=====", flush=True)
verdicts = []
for arm in ("②A全量", "③A卡+B接口", "③bA接口+B卡"):
    for tag in ("FULL", "CLEAN"):
        v = frac(RATE[arm][tag])
        if v >= 0.8:
            j = "★迁移率≥0.8 ⇒ 卡可零成本跨任务复用（核心主张成立）"
        elif v <= 0.2:
            j = "★迁移率≤0.2 ⇒ 卡与任务强绑定（否定「卡可复用」的强版本）"
        else:
            j = f"中间值，如实报（迁移率={v:.3f}）"
        verdicts.append(f"{arm}[{tag}]={v:.3f} ⇒ {j}")
r4f, r4c = frac(RATE["④A卡微调"]["FULL"]), frac(RATE["④A卡微调"]["CLEAN"])
for tag, v in (("FULL", r4f), ("CLEAN", r4c)):
    if v >= 0.95:
        j = "★恢复率≥0.95 ⇒ 少量适配即可迁移（实用结论）"
    elif v <= 0.5:
        j = "★恢复率≤0.5 ⇒ 需从头训"
    else:
        j = f"中间值，如实报（恢复率={v:.3f}）"
    verdicts.append(f"④A卡微调[{tag}]={v:.3f} ⇒ {j}")
for v in verdicts:
    print(f"[S45-JUDGE] {v}", flush=True)

print("\n[S45] ===== ★饱和检查（R35）：① 的 9000→12000 段 EM 增量（全量/clean）=====", flush=True)
SAT_LO, SAT_HI = (9000, 12000) if (9000 in GRID1 and 12000 in GRID1) else (GRID1[-2], GRID1[-1])
_sat_bad = []
for s in SEEDS45:
    r = RESULTS[("①B自训", s)]
    inc = (r["curve"][SAT_HI]["em_full"] - r["curve"][SAT_LO]["em_full"]) * 100
    inc_c = (r["curve"][SAT_HI]["em_clean"] - r["curve"][SAT_LO]["em_clean"]) * 100
    note = "★未饱和（≥1pp）" if inc >= 1.0 else "已平（<1pp）"
    if inc >= 1.0:
        _sat_bad.append(f"①/s{s}")
    print(f"[S45-SAT] ① seed={s}: EM {SAT_LO}→{SAT_HI} = {r['curve'][SAT_LO]['em_full']*100:.2f}→"
          f"{r['curve'][SAT_HI]['em_full']*100:.2f}% (Δ{inc:+.2f}pp) | clean "
          f"{r['curve'][SAT_LO]['em_clean']*100:.2f}→{r['curve'][SAT_HI]['em_clean']*100:.2f}% "
          f"(Δ{inc_c:+.2f}pp) ⇒ {note}", flush=True)
for s in SEEDS45:
    r = RESULTS[("①B自训", s)]
    print(f"[S45-SAT-CURVE] ① seed={s}: " + " ".join(
        f"@{st}={r['curve'][st]['em_full']*100:.2f}/{r['curve'][st]['em_clean']*100:.2f}%"
        for st in GRID1), flush=True)
print(f"[S45-SAT-VERDICT] ① 中 {SAT_LO}→{SAT_HI} ≥1pp 的 seed={_sat_bad if _sat_bad else '无'}"
      f" ⇒ {'★未饱和，按 R35 全部迁移结论不可下' if _sat_bad else '★①已饱和，迁移结论可下'}",
      flush=True)

print("\n[S45] ===== ★④ 的恢复曲线（每 250 步）=====", flush=True)
for s in SEEDS45:
    r = RESULTS[("④A卡微调", s)]
    print(f"[S45-FT-CURVE] ④ seed={s}: " + " ".join(
        f"@{st}={r['curve'][st]['em_full']*100:.2f}/{r['curve'][st]['em_clean']*100:.2f}%"
        for st in GRID4), flush=True)

print("\n[S45] ===== 成本与 ckpt =====", flush=True)
for a in ALL_ARMS:
    for s in SEEDS45:
        r = RESULTS.get((a, s))
        if not r:
            continue
        print(f"[META] {a} seed={s}: 训练墙钟={r['wall_train']/60:.2f}min 本run={r.get('wall_total',0)/60:.2f}min "
              f"ckpt={os.path.basename(r['ckpt'])} sha256[:16]={r['sha']}", flush=True)
print(f"[META] device={device} {torch.cuda.get_device_name(0)} | torch={torch.__version__} | "
      f"D={D} FF={FF} NHEAD={NHEAD} lr={LR} batch={BATCH} MAXLEN={MAXLEN} | EM口径=batch{EM_BATCH} "
      f"自检=batch{CHK_BATCH} | {TASK_A} TEST sha16={TEST_SHA[TASK_A]} | {TASK_B} TEST sha16="
      f"{TEST_SHA[TASK_B]} | 地板(单/top5)={FLOOR_1_TRAIN*100:.2f}%/{FLOOR_5_TEST*100:.2f}% | "
      f"clean n={N_CLEAN[TASK_A]}/{N_CLEAN[TASK_B]} | 总墙钟={(time.time()-T0)/60:.1f}min", flush=True)
print("[DONE] exit=0", flush=True)
