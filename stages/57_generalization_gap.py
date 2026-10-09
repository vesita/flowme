#!/usr/bin/env python3
"""S57 · E11（零训练 · 全程 CPU）★独立验证 S55 的「过拟合」诊断：train/test 泛化 gap。

被验证的诊断（S55，`stages/56_capacity_24k.py` + `logs/s55_cap24k.log`）：
  Δ = EM(ff1024@24000) − EM(S53 ff512@12000) = −0.69±0.41pp ⇒ 容量轴判死（无增益）；
  诊断依据 = 训练 CE @24000 0.0112/0.0260 **低于** S53@12000 的 0.0511/0.0287，
             测试 CE @24000 0.0737/0.0836 **高于** S53@12000 的 0.0432/0.0475
  ⇒ 结论「容量被用来记忆训练集 = 过拟合，不是欠训」。

本单元只做测量，不训练，判据写死在下面。
  ★1 直接量 **train 集上的 EM**（不只是 CE）⇒ gap = train EM − test EM，S53 vs ff1024：
       gap(ff1024) 明显大于 gap(S53)（≥3pp 且 2 seed 同向）⇒ 过拟合诊断成立；
       gap 相当 ⇒ 该诊断不成立（并指出 S55 的 CE 反差还能怎么解释）；
       train EM≈100% 而 test 停在 97% ⇒ 直接证据。
  ★2 用已有 ckpt 快照做「gap 随步数」（ff1024 的 5 格 ×2 seed，零训练、只评估）：
       单调增大 ⇒ 过拟合随时间加重；平稳 ⇒ 不是过拟合。
  ★3 正则潜力探测：eval / train 模式下的 test EM（dropout 开关，只记录不定判）；
       每参数 test EM（容量的边际收益是否在下降）；train 上 loss vs test 上 loss（gap 的 loss 版）。

★口径（硬约束）：
  R28 EM 主口径 batch=1（并做 batch=1 vs 16 批内一致性自检）；
  R39 每次前向前**显式 model.eval()**（本文件的 greedy57 故意不改 mode ⇒ 由调用方显式置）；
  R40 torch.set_num_threads(4)，报告里写明线程数；CUDA_VISIBLE_DEVICES="" 强制 CPU。
  train 集无 clean 口径（`{expr}` 模板 bug 只按 test 的 30/770 定义）⇒ train EM 报**原口径**，
  同时附「train 内剔除 `{expr}` 行」的并列读数，避免拿 train 全量去比 test clean。
  ★loss 版 gap 一律用**同一协议**（eval 模式、同批大小、同代码路径）重测：
  S55 的「训练 CE」是训练中单批 + dropout 开启，「测试 CE」是 eval + dropout 关闭 ⇒ 不同协议。
  另跑一个 train 模式（dropout 开）的复刻读数，量化这个协议差。

只允许写：本文件、logs/57_*、/tmp；既有 stages/*.py 与 logs/ 其它文件一律只读。
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time

FLOWME = "/home/vesita/coding/my/flowme"
SRC52 = f"{FLOWME}/stages/52_error_classify.py"
LOG_DIR = f"{FLOWME}/logs"
OUT_LOG = f"{LOG_DIR}/57_generalization_gap.log"
OUT_JSON = f"{LOG_DIR}/57_generalization_gap.json"
T0 = time.time()
SMOKE = os.environ.get("S57_SMOKE") == "1"

# ============================================================================
# ① 逐字复用 S52（=E9/E10）前半段：同 tokenizer、同 add_3d train4000/test800、同 clean 索引、
#    同 greedy_gen、同 parse_ans。★只把 S52 的两个输出路径改写到 /tmp（其主循环以 "w" 打开）。
# ============================================================================
_src = open(SRC52, encoding="utf-8").read()
_cut = _src.index("# ⑤ 主循环")
_head = _src[:_cut]
for _old, _new in ((f'OUT_LOG = LOG_DIR + "/52_error_classify.log"', 'OUT_LOG = "/tmp/57_s52head.log"'),
                   (f'OUT_JSON = LOG_DIR + "/52_error_classify.json"', 'OUT_JSON = "/tmp/57_s52head.json"')):
    assert _old in _head, f"未能改写 S52 输出路径（口径漂移风险）：{_old}"
    _head = _head.replace(_old, _new)
NS = {"__name__": "s52_head", "__file__": SRC52}
exec(compile(_head, SRC52, "exec"), NS)                 # noqa: S102
for _k, _v in NS.items():
    if not _k.startswith("__"):
        globals()[_k] = _v
del NS, _src, _head
# ★坑（S56 已记录）：S52 头里也有一个同名 SMOKE（读 S52_SMOKE），上面那次 globals() 拷贝会把它
#   带进来并覆盖本单元的 S57_SMOKE ⇒ 必须在拷贝之后重新取一次。
SMOKE = os.environ.get("S57_SMOKE") == "1"

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

# ★exec 把 S52 的 OUT_LOG/OUT_JSON 带进了 globals，这里钉死本单元自己的输出路径
OUT_LOG = f"{LOG_DIR}/57_generalization_gap.log"
OUT_JSON = f"{LOG_DIR}/57_generalization_gap.json"
assert OUT_LOG.startswith(f"{LOG_DIR}/57_") and OUT_JSON.startswith(f"{LOG_DIR}/57_")
assert DEVICE == "cpu" and torch.get_num_threads() == 4, "口径不符：需 CPU + 4 线程(R40)"
assert torch.cuda.is_available() is False, "★CUDA 可见（硬约束：全程 CPU）"

_log_f = open(OUT_LOG, "w", encoding="utf-8")


def say(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    _log_f.write(s + "\n")
    _log_f.flush()


say(f"[S57] device={DEVICE} torch={torch.__version__} cuda.is_available()={torch.cuda.is_available()} "
    f"threads={torch.get_num_threads()}(R40) | 零训练 · EM batch={EM_BATCH}(R28) · eval() 显式(R39) | "
    f"底座 = 逐字复用 stages/52_error_classify.py 到 §5 之前（仅输出改写到 /tmp）")
say(f"[S57] 被验诊断（S55）：训练 CE 0.0112/0.0260 vs S53 0.0511/0.0287（低）· 测试 CE "
    f"0.0737/0.0836 vs S53 0.0432/0.0475（高）⇒ S55 判「容量被用来记忆训练集 = 过拟合」")

# ============================================================================
# ② 评估集（train 原口径 + test 全量/clean）与地板
# ============================================================================
N_TR = 64 if SMOKE else len(TRAIN)                 # ★全 4000（实测 ≈5ms/样本 ⇒ 够快）
N_TE = 64 if SMOKE else len(TEST)                  # 800（全量；clean 取其中 770）
TR_EVAL = TRAIN[:N_TR]
TE_EVAL = TEST[:N_TE]
TR_BUG = [i for i, r in enumerate(TR_EVAL) if "{expr}" in r["nl"]]
TR_CLEAN = [i for i in range(len(TR_EVAL)) if i not in set(TR_BUG)]
TE_CLEAN = [i for i in CLEAN_IDX if i < N_TE]
TE_CLEAN_SET = set(TE_CLEAN)

_c_tr = Counter(r["gold"] for r in TRAIN)
_top1, _ = _c_tr.most_common(1)[0]
_top5 = [v for v, _ in _c_tr.most_common(5)]
FLOOR_1 = sum(1 for r in TEST if r["gold"] == _top1) / len(TEST)
FLOOR_5 = sum(1 for r in TEST if r["gold"] in _top5) / len(TEST)
say(f"[S57-DATA] train 评估 n={len(TR_EVAL)}（原口径，含字面量 '{{expr}}' 行 {len(TR_BUG)} 条 ⇒ "
    f"无 clean 口径）| test n={len(TE_EVAL)} 全量 / clean n={len(TE_CLEAN)}（库里 30/770）| "
    f"地板：单答案'{_top1}'={FLOOR_1*100:.2f}% top5={FLOOR_5*100:.2f}%")

# ============================================================================
# ③ 模型：S56 的 Cards56 逐字同构（in_enc ff=512 固定，thought 卡 ff = 512 或 1024）
# ============================================================================
class Cards57(nn.Module):
    def __init__(self, d, ff_in, ff_tgt):
        super().__init__()
        self.d = d
        self.ff_in, self.ff_tgt = ff_in, ff_tgt
        self.emb = nn.Embedding(V, d)
        self.in_enc = enc_layer(d, ff_in)
        self.thought = enc_layer(d, ff_tgt)
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
        x = self.emb(ids) + pe[:n].unsqueeze(0)
        h = self.in_enc(x, src_mask=m)
        h = h + self.thought(h, src_mask=m)       # 与 S53/S56 同：带残差
        return self.head(h)


def build_batch57(recs):
    lens = [len(r["p"]) + len(r["t"]) for r in recs]
    n = max(lens)
    ids = torch.full((len(recs), n), PAD_ID, dtype=torch.long)
    s = torch.zeros(len(recs), dtype=torch.long)
    for i, r in enumerate(recs):
        ids[i, : len(r["p"])] = torch.tensor(r["p"])
        ids[i, len(r["p"]): len(r["p"]) + len(r["t"])] = torch.tensor(r["t"])
        s[i] = len(r["p"])
    return ids.to(DEVICE), s


def sha16(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def load57(path: str, ff_tgt: int):
    sd = torch.load(path, map_location="cpu", weights_only=True)
    m = Cards57(D, FF, ff_tgt).to(DEVICE)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    assert not [k for k in missing if "pe" not in k], f"{path} 缺键 {missing[:5]}"
    assert not unexpected, f"{path} 多键 {unexpected[:5]}"
    m.eval()                                              # ★R39 显式 eval()
    return m, sha16(path)


# ---------------- 贪心生成（★不改 mode：调用方负责显式 eval()，R39） ----------------
@torch.no_grad()
def greedy57(model, recs, batch: int = 1):
    texts: list[str] = [""] * len(recs)
    eos_end: list[bool] = [False] * len(recs)
    for i in range(0, len(recs), batch):
        chunk = recs[i: i + batch]
        lens = [len(r["p"]) for r in chunk]
        n0 = max(lens)
        ids = torch.full((len(chunk), n0), PAD_ID, dtype=torch.long)
        for j, r in enumerate(chunk):
            ids[j, : lens[j]] = torch.tensor(r["p"])
        ids = ids.to(DEVICE)
        gen_len = [0] * len(chunk)
        cap = [min(MAXLEN - L, MAX_GEN) for L in lens]
        done = [False] * len(chunk)
        last_col = [L - 1 for L in lens]
        last_tok = [PAD_ID] * len(chunk)
        for _ in range(max(cap)):
            if all(done):
                break
            logits = model.logits(ids)
            nxt = []
            for j in range(len(chunk)):
                if done[j]:
                    nxt.append(PAD_ID)
                    continue
                t = int(logits[j, last_col[j]].argmax().item())
                nxt.append(t)
                last_tok[j] = t
                gen_len[j] += 1
                if t == EOS_ID or gen_len[j] >= cap[j]:
                    done[j] = True
                last_col[j] = n0 + gen_len[j] - 1
            ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long, device=DEVICE).unsqueeze(1)], 1)
        for j in range(len(chunk)):
            g_ids = [x for x in ids[j, n0: n0 + gen_len[j]].tolist() if x != EOS_ID]
            texts[i + j] = dec(g_ids)
            eos_end[i + j] = (last_tok[j] == EOS_ID)
    return texts, eos_end


@torch.no_grad()
def ce_seq57(model, recs, gen_batch: int = 32, mode: str = "eval") -> float:
    """token 加权整体 CE。mode='eval' 显式 eval()（R39）；mode='train' 仅用于复刻训练时口径。"""
    if mode == "eval":
        model.eval()                                      # ★R39
    else:
        model.train()                                     # ★仅 dropout 探测/训练口径复刻
    nll, ntok = 0.0, 0
    for i in range(0, len(recs), gen_batch):
        chunk = recs[i: i + gen_batch]
        ids, s = build_batch57(chunk)
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        for j, r in enumerate(chunk):
            e = s[j] + len(r["t"])
            lp = logp[j, s[j] - 1: e - 1]
            tgt = ids[j, s[j]: e]
            nll += float((-lp.gather(1, tgt.unsqueeze(1))).sum().item())
            ntok += len(r["t"])
    return nll / ntok


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n) if n > 1 else 0.0


def binom_se(p, n):
    return math.sqrt(p * (1 - p) / n) if n else float("nan")


def frac(xs):
    return sum(xs) / len(xs) if xs else float("nan")


# ============================================================================
# ④ 参数量（每参数 EM 用）
# ============================================================================
def card_params57(ff):
    m = enc_layer(D, ff)
    n = sum(p.numel() for p in m.parameters())
    del m
    return n


N_BASE = sum(p.numel() for p in Cards57(D, FF, FF).parameters())
N_CAP = sum(p.numel() for p in Cards57(D, FF, FF * 2).parameters())
CARD_BASE, CARD_CAP = card_params57(FF), card_params57(FF * 2)
say(f"[S57-PARAMS] 总量 ff512(基线)={N_BASE/1e6:.4f}M · ff1024(容量)={N_CAP/1e6:.4f}M"
    f"（+{100.0*(N_CAP-N_BASE)/N_BASE:.2f}%）| 目标卡 {CARD_BASE/1e6:.4f}M→{CARD_CAP/1e6:.4f}M"
    f"（+{100.0*(CARD_CAP-CARD_BASE)/CARD_BASE:.1f}%）")

# ============================================================================
# ⑤ ckpt 清单：S53(ff512) @9000/@12000 ×2 seed（基线，@9000 供基线趋势参照）
#                S56(ff1024) @12000/15000/18000/21000/24000 ×2 seed（容量臂，5 格）
# ============================================================================
MODELS: list[dict] = []
for _sd in (1234, 5678):
    for _st in (9000, 12000):
        MODELS.append(dict(arm="S53_ff512", seed=_sd, steps=_st, ff_tgt=FF,
                           path=f"{LOG_DIR}/53_ckpt_lrcosine_s{_st}_seed{_sd}.pt"))
    for _st in (12000, 15000, 18000, 21000, 24000):
        MODELS.append(dict(arm="S56_ff1024", seed=_sd, steps=_st, ff_tgt=FF * 2,
                           path=f"{LOG_DIR}/56_ckpt_ff1024_s{_st}_seed{_sd}.pt"))
if SMOKE:
    MODELS = [m for m in MODELS
              if (m["arm"] == "S53_ff512" and m["steps"] == 12000 and m["seed"] == 1234)
              or (m["arm"] == "S56_ff1024" and m["steps"] == 24000 and m["seed"] == 1234)]
for m in MODELS:
    assert os.path.exists(m["path"]), f"★ckpt 缺失 {m['path']}"
say(f"[S57-CKPTS] {len(MODELS)} 个 ckpt：S53 2 seed×2 步 + S56 2 seed×5 步 | "
    f"train 评估 n={len(TR_EVAL)} test n={len(TE_EVAL)} ⇒ 预估 ≈{len(MODELS)*len(TR_EVAL)*5.5/1000:.0f}s 生成墙钟")

# ============================================================================
# ⑥ 主循环：逐 ckpt = train EM + test EM（全量/clean）+ 同协议 CE
# ============================================================================
RES: list[dict] = []
R28_DONE: dict = {}
say(f"\n[S57] ===== §1/§2 逐 ckpt：train EM（batch=1，原口径）+ test EM（全量/clean）+ CE =====")
for mi, M in enumerate(MODELS):
    model, sha = load57(M["path"], M["ff_tgt"])
    model.eval()                                          # ★R39 显式 eval()
    t0 = time.time()
    txt_tr, eos_tr = greedy57(model, TR_EVAL, batch=EM_BATCH)     # ★R28 batch=1
    txt_te, eos_te = greedy57(model, TE_EVAL, batch=EM_BATCH)     # ★R28 batch=1
    wall_gen = time.time() - t0
    h_tr = [int(parse_ans(t) == r["gold"]) for t, r in zip(txt_tr, TR_EVAL)]
    h_te = [int(parse_ans(t) == r["gold"]) for t, r in zip(txt_te, TE_EVAL)]
    # R28 批内一致性自检：每个 arm 的第一个 ckpt
    if M["arm"] not in R28_DONE:
        model.eval()
        _k = min(16, len(TE_EVAL))
        a1, _ = greedy57(model, TE_EVAL[:_k], batch=1)
        aN, _ = greedy57(model, TE_EVAL[:_k], batch=16)
        R28_DONE[M["arm"]] = sum(x == y for x, y in zip(a1, aN))
        say(f"[R28] {M['arm']} s{M['seed']}@{M['steps']}: batch=1 vs batch=16 逐字一致 "
            f"{R28_DONE[M['arm']]}/{_k} ⇒ "
            f"{'自洽 ✓' if R28_DONE[M['arm']] == _k else '★不一致（主口径仍只用 batch=1）'}")
    # 同协议 CE（eval 模式，train 与 test 走同一代码路径）
    ce_tr = ce_seq57(model, TR_EVAL, mode="eval")
    ce_te = ce_seq57(model, TE_EVAL, mode="eval")
    drop_ce = None
    if (M["arm"] == "S53_ff512" and M["steps"] == 12000) or \
       (M["arm"] == "S56_ff1024" and M["steps"] == 24000):
        torch.manual_seed(0)
        drop_ce = ce_seq57(model, TR_EVAL, mode="train")   # ★训练口径复刻（dropout 开）
    model.eval()
    em_tr = frac(h_tr)
    em_tr_clean = frac([h_tr[i] for i in TR_CLEAN])
    em_te = frac(h_te)
    em_te_clean = frac([h_te[i] for i in TE_CLEAN])
    gap_full = em_tr - em_te
    gap_clean = em_tr_clean - em_te_clean
    se_tr, se_te = binom_se(em_tr, len(h_tr)), binom_se(em_te, len(h_te))
    rec = dict(arm=M["arm"], seed=M["seed"], steps=M["steps"], ff_tgt=M["ff_tgt"],
               sha=sha, path=os.path.basename(M["path"]), path_full=M["path"],
               n_train=len(h_tr), n_test=len(h_te), n_test_clean=len(TE_CLEAN),
               n_train_bug=len(TR_BUG),
               em_train=em_tr, em_train_nobug=em_tr_clean,
               em_test_full=em_te, em_test_clean=em_te_clean,
               gap_full=gap_full, gap_clean=gap_clean,
               gap_full_se=math.sqrt(se_tr ** 2 + se_te ** 2),
               ce_train_eval=ce_tr, ce_test_eval=ce_te, gap_ce=ce_te - ce_tr,
               ce_train_trainmode=drop_ce,
               eos_tr=frac([int(x) for x in eos_tr]), eos_te=frac([int(x) for x in eos_te]),
               wall_gen=wall_gen, hits_test=h_te)
    RES.append(rec)
    say(f"[{mi+1}/{len(MODELS)}] {M['arm']} s{M['seed']}@{M['steps']} sha={sha} | "
        f"train EM={em_tr*100:.2f}%(n={len(h_tr)}, 剔除bug={em_tr_clean*100:.2f}%) | "
        f"test EM 全量={em_te*100:.2f}%±{se_te*100:.2f} clean={em_te_clean*100:.2f}% | "
        f"★gap 全量={gap_full*100:+.2f}pp clean={gap_clean*100:+.2f}pp | "
        f"CE train(eval)={ce_tr:.4f} test={ce_te:.4f} gap={ce_te-ce_tr:+.4f}"
        + (f" | CE train(train模式)={drop_ce:.4f}" if drop_ce is not None else "")
        + f" | eos率 {frac([int(x) for x in eos_tr])*100:.1f}/{frac([int(x) for x in eos_te])*100:.1f}%"
        + f" | 生成 {wall_gen:.1f}s")
    del model, txt_tr, txt_te, h_tr

# ============================================================================
# ⑦ 判据 1：gap(S53@12000) vs gap(ff1024@24000) —— ★EM 版
# ============================================================================
def get(arm, steps, seed):
    for r in RES:
        if r["arm"] == arm and r["steps"] == steps and r["seed"] == seed:
            return r
    return None


say(f"\n[S57] ===== ★判据 1：gap = train EM − test EM，S53@12000 vs ff1024@24000 =====")
FOCAL = {s: (get("S53_ff512", 12000, s), get("S56_ff1024", 24000, s)) for s in (1234, 5678)}
gap_diffs = []
for s in (1234, 5678):
    b, c = FOCAL[s]
    if b is None or c is None:
        continue
    for name, kb, kc in (("全量", "gap_full", "gap_full"), ("clean", "gap_clean", "gap_clean")):
        d = c[kc] - b[kb]
        se = math.sqrt(c["gap_full_se"] ** 2 + b["gap_full_se"] ** 2)
        gap_diffs.append((s, name, d, se))
        say(f"[S57-GAP] {name}口径 s{s}: gap(S53@12000 train EM={b['em_train']*100:.2f}% / "
            f"test={b['em_test_full' if name=='全量' else 'em_test_clean']*100:.2f}%)={b[kb]*100:+.2f}pp"
            f" ‖ gap(ff1024@24000 train EM={c['em_train']*100:.2f}% / "
            f"test={c['em_test_full' if name=='全量' else 'em_test_clean']*100:.2f}%)={c[kc]*100:+.2f}pp"
            f" ⇒ Δgap={d*100:+.2f}pp ± {se*100:.2f}pp")
for name in ("全量", "clean"):
    ds = [d for _s, n, d, _se in gap_diffs if n == name]
    if not ds:
        continue
    same = all(x > 0 for x in ds) or all(x < 0 for x in ds)
    m = sum(ds) / len(ds)
    verdict = ("★gap 明显更大（≥3pp 且 2 seed 同向）⇒ 过拟合诊断成立"
               if same and m >= 0.03
               else ("★gap 明显更小（≤−3pp 且 2 seed 同向）⇒ 与过拟合诊断方向相反"
                     if same and m <= -0.03
                     else "★gap 相当（<3pp 或 2 seed 不同向）⇒ 按写死判据，过拟合诊断【不成立】"))
    say(f"[S57-VERDICT-EM] {name}口径：2 seed 均值 Δgap={m*100:+.2f}pp（"
        + "/".join(f"{x*100:+.2f}" for x in ds) + f"，2 seed 同向={'是' if same else '否'}）⇒ {verdict}")

sat = [r for r in RES if r["arm"] == "S56_ff1024"]
if sat:
    say(f"[S57-CEILING] train EM 天花板检查：全部 {len(RES)} 个 ckpt 里 train EM=100% 的有 "
        f"{sum(1 for r in RES if r['em_train'] >= 0.9999)} 个，最低 {min(r['em_train'] for r in RES)*100:.2f}% "
        f"⇒ {'★train EM 已饱和 ⇒ EM 版 gap 只能由 test 侧差异驱动（天花板效应）' if min(r['em_train'] for r in RES) >= 0.99 else 'train EM 未饱和，gap 可比'}")

# ============================================================================
# ⑧ 判据 1 附：gap 的 loss 版（同一协议：eval 模式、同代码路径、同批大小）
# ============================================================================
say(f"\n[S57] ===== ★判据 1 附：loss 版 gap = CE(test) − CE(train)，同协议（eval 模式）=====")
for s in (1234, 5678):
    b, c = FOCAL[s]
    if b is None or c is None:
        continue
    say(f"[S57-CEGAP] s{s}: S53@12000 CE train={b['ce_train_eval']:.4f} test={b['ce_test_eval']:.4f} "
        f"⇒ gap={b['gap_ce']:+.4f} nats ‖ ff1024@24000 CE train={c['ce_train_eval']:.4f} "
        f"test={c['ce_test_eval']:.4f} ⇒ gap={c['gap_ce']:+.4f} nats ⇒ Δgap="
        f"{(c['gap_ce']-b['gap_ce']):+.4f} nats（{(c['gap_ce']/b['gap_ce'] if b['gap_ce'] else float('nan')):.2f}×）")
    if b.get("ce_train_trainmode") is not None and c.get("ce_train_trainmode") is not None:
        say(f"[S57-CEPROTO] s{s}: train 集 CE —— eval 模式 {b['ce_train_eval']:.4f}/{c['ce_train_eval']:.4f}"
            f" vs train 模式(dropout 开) {b['ce_train_trainmode']:.4f}/{c['ce_train_trainmode']:.4f}"
            f"（S53/ff1024）⇒ dropout 项 = +{b['ce_train_trainmode']-b['ce_train_eval']:.4f}/"
            f"+{c['ce_train_trainmode']-c['ce_train_eval']:.4f} nats"
            f"（S55 的『训练 CE』是 train 模式单批 ⇒ 与 eval 测试 CE 不同协议）")

# ============================================================================
# ⑨ 判据 2：gap 随步数（ff1024 的 5 格 × 2 seed）
# ============================================================================
say(f"\n[S57] ===== ★判据 2：gap 随步数（ff1024 的 5 格 × 2 seed；S53 @9000/@12000 仅作基线参照）=====")
for s in (1234, 5678):
    row = [r for r in RES if r["arm"] == "S56_ff1024" and r["seed"] == s]
    row.sort(key=lambda r: r["steps"])
    base = [r for r in RES if r["arm"] == "S53_ff512" and r["seed"] == s]
    base.sort(key=lambda r: r["steps"])
    if not row:
        continue
    say(f"[S57-TREND] s{s} ff1024：" + " | ".join(
        f"@{r['steps']} train={r['em_train']*100:.2f} test={r['em_test_full']*100:.2f}"
        f"(clean {r['em_test_clean']*100:.2f}) gap={r['gap_full']*100:+.2f}/{r['gap_clean']*100:+.2f}pp"
        for r in row))
    inc = all(row[i + 1]["gap_full"] > row[i]["gap_full"] for i in range(len(row) - 1))
    inc_clean = all(row[i + 1]["gap_clean"] > row[i]["gap_clean"] for i in range(len(row) - 1))
    monotone = inc and inc_clean
    d0, d1 = row[0]["gap_full"], row[-1]["gap_full"]
    say(f"[S57-TREND-VERDICT] s{s}: 全量 gap @{row[0]['steps']}={d0*100:+.2f}pp → "
        f"@{row[-1]['steps']}={d1*100:+.2f}pp（Δ={100*(d1-d0):+.2f}pp）| 逐格单调增="
        f"{'是' if monotone else '否'}（全量 {inc} / clean {inc_clean}）⇒ "
        f"{'★gap 随步数单调增大 ⇒ 过拟合随时间加重' if monotone else '★gap 非单调 ⇒ 不是「越训越过拟合」的形态'}"
        + ("｜S53 基线参照：" + " | ".join(
            f"@{r['steps']} gap={r['gap_full']*100:+.2f}pp" for r in base) if base else ""))

# ============================================================================
# ⑩ 判据 3：每参数 test EM + dropout 开关探测
# ============================================================================
say(f"\n[S57] ===== ★判据 3：每参数 test EM（容量边际收益）+ dropout 开关探测 =====")
for s in (1234, 5678):
    b, c = FOCAL[s]
    if b is None or c is None:
        continue
    for name, key in (("全量", "em_test_full"), ("clean", "em_test_clean")):
        pb = b[key] / N_BASE * 1e6 * 100      # %/M
        pc = c[key] / N_CAP * 1e6 * 100
        say(f"[S57-PERPARAM] {name} s{s}: S53 {b[key]*100:.2f}% / {N_BASE/1e6:.4f}M = {pb:.2f}%/M ‖ "
            f"ff1024 {c[key]*100:.2f}% / {N_CAP/1e6:.4f}M = {pc:.2f}%/M ⇒ Δ={pc-pb:+.2f}%/M"
            f"（{pc/pb:.4f}×）")

say(f"\n[S57] ===== dropout 开关探测（★只记录，不作判据）：S53 ckpt 在 eval / train 模式下的 test EM =====")
for s in (1234, 5678):
    b = FOCAL[s][0]
    if b is None:
        continue
    model, _sha = load57(b["path_full"], FF)
    model.eval()                                          # ★R39
    t_ev, _ = greedy57(model, TE_EVAL, batch=EM_BATCH)
    em_ev = frac([int(parse_ans(t) == r["gold"]) for t, r in zip(t_ev, TE_EVAL)])
    torch.manual_seed(0)
    model.train()                                        # ★故意开 dropout（这就是探测本身）
    t_tr, _ = greedy57(model, TE_EVAL, batch=EM_BATCH)
    em_drop = frac([int(parse_ans(t) == r["gold"]) for t, r in zip(t_tr, TE_EVAL)])
    model.eval()
    say(f"[S57-DROPOUT] S53@12000 s{s}: eval 模式(dropout 关) test EM={em_ev*100:.2f}% vs "
        f"train 模式(dropout 开) test EM={em_drop*100:.2f}% ⇒ Δ={100*(em_drop-em_ev):+.2f}pp"
        f" ⇒ {'推理时本来就关着 dropout（读数不受影响）' if em_drop <= em_ev else '★开 dropout 反而更高（记录，不作判据）'}")
    del model, t_ev, t_tr

# ============================================================================
# ⑪ 对照 S55 入库数字（交叉核验）
# ============================================================================
say(f"\n[S57] ===== 与 S55 入库数字对照（S55：ff1024@24000 Δ全量=−0.69±0.41pp，2 seed 同号）=====")
for s in (1234, 5678):
    b, c = FOCAL[s]
    if b is None or c is None:
        continue
    d = [x - y for x, y in zip(c["hits_test"], b["hits_test"])]
    m, se = mean_se([float(x) for x in d])
    say(f"[S57-S55XCHECK] s{s}: 配对 Δ(ff1024@24000 − S53@12000) test 全量 = {m*100:+.2f}pp "
        f"± {se*100:.2f}pp（S55 入库 −0.69±0.41pp 是 2 seed 均值）| S53 本次 test 全量="
        f"{b['em_test_full']*100:.2f}% clean={b['em_test_clean']*100:.2f}%（METRICS 93.62/97.27）")

# ============================================================================
# ⑫ 落盘 + 元信息
# ============================================================================
with open(OUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dict(unit="S57", device=DEVICE, threads=torch.get_num_threads(),
                   em_batch=EM_BATCH, bucket=BUCKET, n_train_eval=len(TR_EVAL),
                   n_test_eval=len(TE_EVAL), n_train_bug=len(TR_BUG),
                   n_test_clean=len(TE_CLEAN), r28=R28_DONE,
                   params=dict(base=N_BASE, cap=N_CAP, card_base=CARD_BASE, card_cap=CARD_CAP),
                   records=[{k: v for k, v in r.items() if k != "hits_test"} for r in RES],
                   wall=time.time() - T0), f, ensure_ascii=False, indent=1)
say(f"\n[S57-META] device={DEVICE} threads={torch.get_num_threads()}(R40) torch={torch.__version__} | "
    f"零训练（只 torch.load + 前向）| EM 口径=batch{EM_BATCH}(R28) R28 自检={R28_DONE} | "
    f"train 评估 n={len(TR_EVAL)} test n={len(TE_EVAL)} clean n={len(TE_CLEAN)} | "
    f"参数量 {N_BASE/1e6:.4f}M→{N_CAP/1e6:.4f}M | 总墙钟={(time.time()-T0)/60:.2f}min")
say("[DONE] exit=0")
sys.exit(0)
