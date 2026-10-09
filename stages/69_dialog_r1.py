#!/usr/bin/env python3
"""S69 · L2-r1 判据驱动：把「跨轮消解 + 干扰项排除」对话任务接进 cards.CardPipeline。

数据来自 stages/68_dialog_gen_r1.py（本文件不造数据；缺 logs/68_dialog_*.jsonl 时先自动跑生成器）。
模型/训练/贪心 EM 的写法逐字沿用 stages/67_dialog_cards.py（因而也等于 S65/S12）：
  · `CardPipeline(residual=False)`、d=128 ff=512 nhead=4 maxlen=512 pad_id/EOS 与 S12 一致；
  · teacher forcing **只在目标 span 上算 CE**（`masked_ce`）；
  · 贪心 EM = batch=1、正确列索引（`last_col`）——R28 自洽口径；
  · 数据指纹 `data_fp` 与 S65 同口径（逐条 len|ids 拼接的 sha256[:16]）。

判据原文（L2_PLAN.md §7，写死）：
  Q1（主）「卡片组 EM > max_naive + 2×SE 且 2 seed 同号」；
  Q2「干扰臂：把轮1/轮2 的实体互换后重评同一 ckpt ⇒ EM 必掉」；
  Q3「地板含计数类且 train 上 < 90%（沿用 P2 口径）」；
  Q4「报告是否饱和（EM < 99% 且 loss 不塌到 1e-4），否则只能当"能承载"的证据」。

用法：
  python stages/69_dialog_r1.py --smoke                  # 50 步、只评 60 条（探路）
  python stages/69_dialog_r1.py --seed 1234              # 全量 3000 步
  python stages/69_dialog_r1.py --seed 1234 --swap       # Q2：换实体、保留原 target，同一 ckpt 重评
  # L2-r1b / r1c 两个单变量臂（本文件只加参数，其余逻辑逐字不变）：
  python stages/69_dialog_r1.py --data-prefix 68b --seed 1234            # r1b：共享词表数据（68b_dialog_*）
  python stages/69_dialog_r1.py --data-prefix 68 --seed 1234 --steps 12000   # r1c：互斥词表、12000 步
  --data-prefix {68,68b} 决定读哪份 jsonl（logs/<prefix>_dialog_*.jsonl）与 ckpt 命名
  （68 ⇒ 69_ckpt_seed*.pt 不变；68b ⇒ 69_ckpt_68b_seed*.pt，互不覆盖）；缺 68b 数据时自动带
  --shared-vocab 跑生成器。--steps 默认 3000（--smoke 时仍固定 50 步）。
GPU 长跑必须：systemd-run --user --unit=<名> --collect
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 ...
只允许写本文件与 logs/。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import subprocess
import sys
import time

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
if NS_ROOT not in sys.path:
    sys.path.insert(0, NS_ROOT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from cards import CardPipeline  # noqa: E402

# ---------------- 配置（与 S65/S12/S67 一致） ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512
STEPS = 3000
LR = 1e-3
BATCH = 32
GEN_BATCH = 32
EM_BATCH = 1                 # ★主口径：batch=1（R28 自洽写法）
SMOKE_STEPS = 50
SMOKE_EVAL = 60
LOG_DIR = os.path.join(ROOT, "logs")
DATA_TMPL = os.path.join(LOG_DIR, "68_dialog_{split}.jsonl")   # main 里按 --data-prefix 覆盖
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
D68_PATH = os.path.join(ROOT, "stages", "68_dialog_gen_r1.py")


def load_d68():
    """按路径加载 stages/68_dialog_gen_r1.py（文件名以数字开头，不能普通 import）。"""
    spec = importlib.util.spec_from_file_location("d68_dialog_gen_r1", D68_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d68 = load_d68()
MARK_SELF = d68.MARK_SELF
MARK_OTHER = d68.MARK_OTHER

# ---------------- device 纪律（ROCm 下报成 cuda，正常） ----------------
device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
if device != "cuda":
    print("GPU 起不来，立即停（不用 CPU 硬跑）。", flush=True)
    print("torch.cuda.is_available() =", torch.cuda.is_available(), flush=True)
    print("HSA_OVERRIDE_GFX_VERSION =", os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)
    sys.exit(3)

from tokenizers import Tokenizer  # noqa: E402

tok = Tokenizer.from_file(TOK_PATH)
V = tok.get_vocab_size()
PAD_ID = tok.token_to_id("<pad>")
EOS_ID = tok.token_to_id("<eos>")


def enc(text: str) -> list:
    return tok.encode(text, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


# ---------------- 数据 ----------------
def ensure_data(prefix: str = "68"):
    """缺数据时自动跑生成器；prefix=68b 时带 --shared-vocab（与 --data-prefix 对齐）。"""
    if os.path.exists(DATA_TMPL.format(split="train")) and os.path.exists(DATA_TMPL.format(split="test")):
        return
    print(f"[DATA] 缺 {DATA_TMPL.format(split='train')} ⇒ 先跑生成器", flush=True)
    cmd = [sys.executable, D68_PATH] + (["--shared-vocab"] if prefix == "68b" else [])
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[DATA] 生成器退出码 {r.returncode} ⇒ 停")


def load_jsonl(split: str) -> list:
    with open(DATA_TMPL.format(split=split), encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def swap_entities(text: str, r) -> str:
    """Q2：把"我自己的"与"我朋友的"两个实体**整体互换**（只改 prompt 文本，target 原样保留）。

    ★ 正确做法是**按角色槽位换名字**，而不是"换名字 + 换角色词"：
      第一人称自称（我自己的 / 我朋友）**不可互换**，换完会得到一句自相矛盾的话
      （"我朋友的咖啡机"配上"我"的主语 ⇒ 测量被污染，EM 下降不再能证明"读了历史"）。
      所以：把**出现在「我自己的」后面的那个实体**换成另一个名字，反之亦然
      ⇒ 互换后 ctx 依然完全自洽，只是"我自己的东西"换了一个人。

    两个实体名互不为子串（生成器已断言），整串 replace 是精确的一次性互换。
    轮3 用户话不含实体、也没有角色词 ⇒ 不变；target 保持不变。
    ⇒ 读历史的模型在互换后应当输出**另一个**名字 ⇒ 与原 target 不符 ⇒ EM 掉；
      只背模板的模型不看历史 ⇒ 输出不变 ⇒ EM 不降。
    """
    if not text:
        return text
    a_self, a_other = r["ent"], r["ent_other"]          # 原始世界里"我自己的 / 我朋友的"
    if not r["self_first"]:
        a_self, a_other = a_other, a_self
    return text.replace(a_self, "\x00").replace(a_other, "\x01").replace("\x00", a_other).replace("\x01", a_self)


def to_ids(recs, swap: bool = False):
    """把文本编码成 p/t；swap=True 时 p 用**实体互换后**的 ctx（t 仍是原 target）。"""
    n_swapped = 0
    for r in recs:
        src = r["ctx"]
        if swap:
            src2 = swap_entities(src, r)
            n_swapped += int(src2 != src)
            src = src2
        r["p"] = enc(src)
        r["t"] = enc(r["target"] + d68.EOS)
        r["ntok_all"] = len(r["p"]) + len(r["t"])
    return n_swapped


# ---------------- 模型 / loss（写法照抄 S67/S65） ----------------
def new_pipe() -> CardPipeline:
    """★residual=False：cards/target.py:28-30 默认带外层残差，S65 已实测必须关掉。"""
    return CardPipeline(d=D, vocab_size=V, ff=FF, nhead=NHEAD, maxlen=MAXLEN,
                        pad_id=PAD_ID, dropout=0.1, residual=False).to(device)


def build_batch(recs):
    lens = [len(r["p"]) + len(r["t"]) for r in recs]
    n = max(lens)
    ids = torch.full((len(recs), n), PAD_ID, dtype=torch.long)
    s = torch.zeros(len(recs), dtype=torch.long)
    for i, r in enumerate(recs):
        ids[i, : len(r["p"])] = torch.tensor(r["p"])
        ids[i, len(r["p"]): len(r["p"]) + len(r["t"])] = torch.tensor(r["t"])
        s[i] = len(r["p"])
    return ids.to(device), s


def masked_ce(model, recs) -> torch.Tensor:
    """teacher forcing：只在目标 span [s, s+len(t)) 上算 CE（照抄 S65::masked_ce）。"""
    ids, s = build_batch(recs)
    logits = model.logits(ids)
    logp = torch.log_softmax(logits, dim=-1)
    nll = torch.zeros((), device=device)
    ntok = 0
    for i, r in enumerate(recs):
        e = s[i] + len(r["t"])
        lp = logp[i, s[i] - 1: e - 1]
        tgt = ids[i, s[i]: e]
        nll = nll - lp.gather(1, tgt.unsqueeze(1)).sum()
        ntok += len(r["t"])
    return nll / ntok


def card_param_groups(model) -> dict:
    return {n: list(model.card(n).parameters()) for n in model.CARD_NAMES}


def train_run(model, train, seed: int, steps: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    opt = AdamW(model.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, gate_res = 0, 0, None
    last_loss, loss_trace = float("nan"), []
    t0 = time.time()
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
        step += 1
        if step == 20:
            gate_res = {k: sum(p.grad is not None for p in ps)
                        for k, ps in card_param_groups(model).items()}
            tot = {k: len(ps) for k, ps in card_param_groups(model).items()}
            print(f"[GATE A] step20 grad is not None 计数 {gate_res} / 参数总数 {tot} "
                  f"=> {'PASS 全>0' if all(v > 0 for v in gate_res.values()) else 'FAIL'}",
                  flush=True)
        opt.step()
        last_loss = float(loss.item())
        if step % 500 == 0 or step == steps:
            loss_trace.append((step, last_loss))
            print(f"  [train] seed={seed} step={step}/{steps} loss={last_loss:.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    return gate_res, time.time() - t0, last_loss, loss_trace


@torch.no_grad()
def eval_ce(model, recs) -> list:
    model.eval()
    per = []
    for i in range(0, len(recs), GEN_BATCH):
        chunk = recs[i: i + GEN_BATCH]
        ids, s = build_batch(chunk)
        logp = torch.log_softmax(model.logits(ids), dim=-1)
        for j, r in enumerate(chunk):
            e = s[j] + len(r["t"])
            lp = logp[j, s[j] - 1: e - 1]
            tgt = ids[j, s[j]: e]
            per.append(float((-lp.gather(1, tgt.unsqueeze(1)).sum() / len(r["t"])).item()))
    model.train()
    return per


@torch.no_grad()
def greedy_gen(model, recs, batch: int = 1) -> list:
    """贪心解码，batch=1、正确列索引（照抄 S65::greedy_gen mode='correct'）。"""
    model.eval()
    texts = [""] * len(recs)
    for i in range(0, len(recs), batch):
        chunk = recs[i: i + batch]
        lens = [len(r["p"]) for r in chunk]
        n0 = max(lens)
        ids = torch.full((len(chunk), n0), PAD_ID, dtype=torch.long)
        for j, r in enumerate(chunk):
            ids[j, : lens[j]] = torch.tensor(r["p"])
        ids = ids.to(device)
        gen_len = [0] * len(chunk)
        cap = [MAXLEN - L for L in lens]
        done = [False] * len(chunk)
        last_col = [L - 1 for L in lens]
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
                gen_len[j] += 1
                if t == EOS_ID or gen_len[j] >= cap[j]:
                    done[j] = True
                last_col[j] = n0 + gen_len[j] - 1
            ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long, device=device).unsqueeze(1)], 1)
        for j in range(len(chunk)):
            g = ids[j, n0: n0 + gen_len[j]].tolist()
            texts[i + j] = dec([x for x in g if x != EOS_ID])
    model.train()
    return texts


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n) if n else float("nan")


def em_strict(gold: str, pred: str) -> int:
    """严格 EM：模板+槽位确定性生成 ⇒ 剥掉首尾空白后逐字符相等（唯一可判）。"""
    return int((gold or "").strip() == (pred or "").strip())


def evaluate(model, ev, tag: str):
    """配对 CE + 贪心 EM（含实体诊断）；返回一个 dict。"""
    ce = eval_ce(model, ev)
    cm, cse = mean_se(ce)
    txt = greedy_gen(model, ev, batch=EM_BATCH)
    hits = [em_strict(r["target"], txt[i]) for i, r in enumerate(ev)]
    ent_hit = [int(r["ent"] in txt[i]) for i, r in enumerate(ev)]
    other_hit = [int(r["ent_other"] in txt[i]) for i, r in enumerate(ev)]
    em = 100.0 * sum(hits) / len(hits)
    se = (em / 100.0 * (1 - em / 100.0) / len(hits)) ** 0.5 * 100.0
    return dict(tag=tag, cm=cm, cse=cse, txt=txt, hits=hits, em=em, se=se,
                em_ent=100.0 * sum(ent_hit) / len(ent_hit),
                em_other=100.0 * sum(other_hit) / len(other_hit))


# ---------------- 主流程 ----------------
def main(argv=None):
    global DATA_TMPL
    ap = argparse.ArgumentParser(description="S69 L2-r1 跨轮消解卡片驱动")
    ap.add_argument("--smoke", action="store_true", help="50 步、只评 60 条")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--swap", action="store_true",
                    help="Q2：test 的轮1/轮2 实体互换、保留原 target，同一 ckpt 重评（不重训）")
    ap.add_argument("--steps", type=int, default=STEPS,
                    help=f"训练步数（默认 {STEPS}；--smoke 时固定 {SMOKE_STEPS}）")
    ap.add_argument("--data-prefix", choices=("68", "68b"), default="68",
                    help="数据/ckpt 前缀：68=logs/68_dialog_*.jsonl（默认，与 r1 逐字同径）；"
                         "68b=logs/68b_dialog_*.jsonl（L2-r1b 共享词表臂），ckpt 名加 68b_ 前缀")
    a = ap.parse_args(argv)
    DATA_TMPL = os.path.join(LOG_DIR, f"{a.data_prefix}_dialog_{{split}}.jsonl")
    steps = SMOKE_STEPS if a.smoke else a.steps
    tag = "swap" if a.swap else "hist"
    if a.smoke:
        tag += "-smoke"
    ensure_data(a.data_prefix)
    train, test = load_jsonl("train"), load_jsonl("test")
    print(f"[DATA-SRC] data_prefix={a.data_prefix} 文件="
          f"{DATA_TMPL.format(split='train')} / {DATA_TMPL.format(split='test')} | "
          f"steps={steps}{'(smoke固定)' if a.smoke else ''}", flush=True)

    # 数据契约（生成器 68 已断言过；这里用落盘的数据复核一次）
    n_vio_a = sum(int(r["turn3_has_ent"]) or int(r["ent"] in r["t3"]) or int(r["ent_other"] in r["t3"])
                  for r in train + test)
    n_vio_b = sum(int(not (r["ent_in_target"] and not r["other_in_target"]))
                  for r in train + test)
    n_two = sum(int(r["ent_count"] == 2) for r in train + test)
    print(f"[DATA] train={len(train)} test={len(test)} | verify.68 A违例={n_vio_a}（必须 0）| "
          f"B违例={n_vio_b}（必须 0）| 实体个数恒=2 的条数={n_two}/{len(train)+len(test)} | "
          f"族={d68.N_TEMPLATE_FAMILIES} 句面={d68.N_FAMILY_FORMS} | "
          f"角色词 正确「{d68.ROLE_SELF}」干扰「{d68.ROLE_OTHER}」| "
          f"位置桶(0=先出现) train={dict((k, sum(1 for r in train if r['ent_pos_rank']==k)) for k in (0,1))} "
          f"test={dict((k, sum(1 for r in test if r['ent_pos_rank']==k)) for k in (0,1))}", flush=True)

    # ★swap 臂不重训：直接载同 seed 的 ckpt（ckpt 名带 data-prefix，68b 与 68 互不覆盖）
    ck_pre = "" if a.data_prefix == "68" else f"{a.data_prefix}_"
    ck = os.path.join(LOG_DIR, f"69_ckpt_{ck_pre}seed{a.seed}_hist{'-smoke' if a.smoke else ''}.pt")
    model = new_pipe()
    n_par = sum(p.numel() for p in model.parameters())
    gate_res, wall, last_loss, loss_trace = None, 0.0, float("nan"), []
    if a.swap:
        if not os.path.exists(ck):
            print(f"[SWAP] 缺 ckpt {ck} ⇒ 先跑 `--seed {a.seed}` 拿到该 ckpt（Q2 不重训）", flush=True)
            return 5
        model.load_state_dict(torch.load(ck, map_location=device))
        print(f"[SWAP] 复用 ckpt {ck}（**不重训**，只把 test 的轮1/轮2 实体互换后重评）", flush=True)
    n_swapped = to_ids(train, a.swap)
    n_swapped_te = to_ids(test, a.swap)

    over = sum(1 for r in train + test if r["ntok_all"] > MAXLEN)
    print(f"[DATA] arm={'实体互换(只改 prompt, 保留原 target)' if a.swap else '原样'} | "
          f"换过的 test 条数={n_swapped_te}/{len(test)} | "
          f"p 中位={sorted(len(r['p']) for r in train + test)[len(train + test)//2]} "
          f"最大={max(len(r['p']) for r in train + test)} | "
          f"p+t 最大={max(r['ntok_all'] for r in train + test)} 超 MAXLEN={over}", flush=True)

    tr_fp, te_fp = d68.data_fp(train), d68.data_fp(test)
    print(f"[DATA] train_fp={tr_fp} test_fp={te_fp}（与 [D68] 同行口径）", flush=True)
    if a.swap:
        # 自检：互换后，**原 target 的实体**必须已经不在「我自己的」槽位上
        #（即原 target 在这个世界里已经答不上）—— 这正是 Q2 的因果点
        n_flip = 0
        for r in test:
            s2 = swap_entities(r["ctx"], r)
            # 互换后角色规则读出的"我自己的"实体 = 原干扰实体 ⇒ 原 target 已答不上
            n_flip += int(d68._role_entity(s2) == r["ent_other"])
        print(f"[SWAP] 自检：互换后「原正确实体→我朋友的 / 原干扰实体→我自己的」条数="
              f"{n_flip}/{len(test)}（应为 100%）", flush=True)
    assert n_vio_a == 0 and n_vio_b == 0 and over == 0, "数据契约不过：轮3 含实体 / 目标含错实体 / 超长"

    # ---- 地板（Q3；只用输入可见信息 + train 统计） ----
    print("[FLOOR] ★Q3：max_naive 含计数类特征（轮3 长度 / 实体个数 / 轮数 / 位置桶）", flush=True)
    _f_full, mxname_full, mx_full, thr_full = d68.compute_floors(
        train, test, tag="test", show_train=True)
    if a.smoke:
        ev = test[:SMOKE_EVAL]
        _f_s, mxname_s, mx_s, thr_s = d68.compute_floors(
            train, ev, tag=f"smoke子集(n={len(ev)})", show_train=False)
        mx, mxname, thr = mx_s, mxname_s, thr_s
    else:
        ev = test
        mx, mxname, thr = mx_full, mxname_full, thr_full

    # ---- 训练（swap 臂跳过） ----
    if not a.swap:
        gate_res, wall, last_loss, loss_trace = train_run(model, train, a.seed, steps)

    # ---- 评估：配对 CE + 贪心 EM ----
    res = evaluate(model, ev, tag)
    cm, cse, em, se, txt, hits = res["cm"], res["cse"], res["em"], res["se"], res["txt"], res["hits"]
    print(f"[MAIN] seed={a.seed} arm={'swap' if a.swap else 'hist'} steps={steps if not a.swap else 0}"
          f"(swap 复用 ckpt) batch={BATCH} lr={LR} | 模型=CardPipeline(residual=False) d={D} ff={FF} "
          f"nhead={NHEAD} maxlen={MAXLEN} params={n_par} | 配对CE={cm:.4f}±{cse:.4f} (nats/token)",
          flush=True)
    print(f"[EM] {'smoke子集' if a.smoke else '全量'} n={len(hits)} 严格EM={em:.2f}% "
          f"±SE={se:.2f}pp | 诊断：生成含**正确实体**比例={res['em_ent']:.2f}% | "
          f"生成含**干扰实体**比例={res['em_other']:.2f}% | "
          f"地板 max_naive={mx:.2f}%({mxname}) 门槛={thr:.2f}%", flush=True)

    shown = 0
    for i, r in enumerate(ev):
        if shown >= 6:
            break
        print(f"[GEN] [{i}] t3={r['t3']} | ent={r['ent']} other={r['ent_other']} | "
              f"gold={r['target']} | pred={txt[i][:80]} | "
              f"{'对' if hits[i] else '错'}（正确实体在 pred 里={bool(r['ent'] in txt[i])}, "
              f"干扰实体在 pred 里={bool(r['ent_other'] in txt[i])}）", flush=True)
        shown += 1

    # ---- 判据（原文引 L2_PLAN.md §7） ----
    if a.swap:
        q2_drop = None
        print(f"[CRIT] Q2 L2_PLAN §7：「干扰臂：把轮1/轮2 的实体互换后重评同一 ckpt ⇒ EM 必掉」| "
              f"本 run 就是 swap 臂：swap 后 EM={em:.2f}%（ckpt=seed{a.seed}，未重训）| "
              f"「必掉」需与**同 ckpt 的正常臂**比较，见下一行/汇总行", flush=True)
        # 顺带在同一 ckpt 上把正常臂也评一遍（同进程、同一 ckpt ⇒ 最干净的 A/B）
        normal = load_jsonl("test")
        evn = normal[:SMOKE_EVAL] if a.smoke else normal
        to_ids(evn, swap=False)
        resn = evaluate(model, evn, "normal-on-same-ckpt")
        q2_drop = resn["em"] - em
        print(f"[CRIT] Q2 同 ckpt 对照：正常臂 EM={resn['em']:.2f}%±{resn['se']:.2f}pp vs "
              f"swap 臂 EM={em:.2f}%±{se:.2f}pp ⇒ 掉幅={q2_drop:.2f}pp ⇒ "
              f"{'PASS（读历史）' if q2_drop > 2.0 * max(resn['se'], se) else 'FAIL（疑似背模板，没在读历史）'}",
              flush=True)
    else:
        print(f"[CRIT] Q1（主）L2_PLAN §7：「卡片组 EM > max_naive + 2×SE 且 2 seed 同号」| "
              f"EM={em:.2f}% vs 门槛(max_naive+2SE)={thr:.2f}% ⇒ "
              f"{'PASS' if em > thr else 'FAIL'} | 「2 seed 同号」本 run 只有 seed={a.seed} "
              f"⇒ 待第 2 个 seed 复跑同号", flush=True)
        print(f"[CRIT] Q2 L2_PLAN §7：「swap 后 EM 必掉」| 本 run 是正常臂 EM={em:.2f}% "
              f"⇒ 需 --swap 复用本 ckpt（{ck}）另跑一次来判", flush=True)
    j_q3 = True
    print(f"[CRIT] Q3 L2_PLAN §7：「地板含计数类且 train 上 < 90%（沿用 P2 口径）」⇒ "
          f"已逐条报出（上方 [FLOOR]），max_naive={mx:.2f}% 来自 {mxname} ⇒ "
          f"{'PASS' if j_q3 else 'FAIL'}", flush=True)
    if a.swap:
        q4 = "本臂不适用（swap 复用 ckpt、未训练，不参与饱和判定；看正常臂的 Q4）"
    else:
        sat = (em < 99.0) and (a.smoke or last_loss > 1e-4)
        q4 = ("未饱和（可当机制实验载体）" if sat else
              "饱和（本任务不能当机制实验载体）")
    print(f"[CRIT] Q4 L2_PLAN §7：「报告是否饱和（EM < 99% 且 loss 不塌到 1e-4），否则只能当"
          f"'能承载'的证据」| EM={em:.2f}% (EM<99% = {em < 99.0}) | "
          f"最后一步 loss={last_loss:.6f} (>{1e-4} = {last_loss > 1e-4}) | "
          f"loss迹={loss_trace} ⇒ {q4}", flush=True)

    ck_out = os.path.join(LOG_DIR, f"69_ckpt_{ck_pre}seed{a.seed}_{tag}.pt")
    torch.save(model.state_dict(), ck_out)
    print(f"[META] tag={tag} seed={a.seed} swap={a.swap} steps={steps if not a.swap else 0} "
          f"data_prefix={a.data_prefix} batch={BATCH} lr={LR} d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN} "
          f"n_params={n_par} 墙钟={wall:.0f}s ckpt={ck_out}", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
