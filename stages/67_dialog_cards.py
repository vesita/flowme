#!/usr/bin/env python3
"""S67 · L2 判据驱动：把「跨轮指代式单轮回复」对话任务接进 cards.CardPipeline。

数据来自 stages/66_dialog_gen.py（本文件不造数据；缺 logs/66_dialog_*.jsonl 时先自动跑生成器）。
模型/训练/贪心 EM 的写法逐字沿用 stages/65_textgen_cards.py：
  · `CardPipeline(residual=False)`、d=128 ff=512 nhead=4 maxlen=512 pad_id/EOS 与 S12 一致；
  · teacher forcing **只在目标 span 上算 CE**（`masked_ce`）；
  · 贪心 EM = batch=1、正确列索引（`last_col`）——R28 自洽口径；
  · 数据指纹 `data_fp` 与 S65 同口径（逐条 len|ids 拼接的 sha256[:16]）。

判据原文（L2_PLAN.md §4）：
  P1（主）「卡片组 EM > max_naive + 2×SE，且 2 seed 同号」；
  P2「max_naive 必须含计数类特征（长度 / 实体个数 / 轮数 / 位置桶），且逐条报出」；
  P3「契约：数据指纹 [G0] + 三卡梯度非空（[GATE A]）」；
  P4「CE 与 EM 并列报」；
  P5「消融：去掉轮1/轮2 历史（只喂轮3）⇒ EM 必掉」。

用法：
  python stages/67_dialog_cards.py --smoke                 # 50 步、只评 60 条（探路）
  python stages/67_dialog_cards.py --seed 1234             # 全量 3000 步
  python stages/67_dialog_cards.py --seed 1234 --no-history  # P5 消融
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

# ---------------- 配置（与 S65/S12 一致） ----------------
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
DATA_TMPL = os.path.join(LOG_DIR, "66_dialog_{split}.jsonl")
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
D66_PATH = os.path.join(ROOT, "stages", "66_dialog_gen.py")


def load_d66():
    """按路径加载 stages/66_dialog_gen.py（文件名以数字开头，不能普通 import）。"""
    spec = importlib.util.spec_from_file_location("d66_dialog_gen", D66_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d66 = load_d66()

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
def ensure_data():
    if os.path.exists(DATA_TMPL.format(split="train")) and os.path.exists(DATA_TMPL.format(split="test")):
        return
    print(f"[DATA] 缺 {DATA_TMPL.format(split='train')} ⇒ 先跑生成器", flush=True)
    r = subprocess.run([sys.executable, D66_PATH], cwd=ROOT)
    if r.returncode != 0:
        raise SystemExit(f"[DATA] 生成器退出码 {r.returncode} ⇒ 停")


def load_jsonl(split: str) -> list:
    with open(DATA_TMPL.format(split=split), encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def no_history_ctx(r) -> str:
    """P5 消融输入：只喂轮3（不喂轮1/轮2），格式仍是 DialogueStream 的「用户：… / <resp>」。"""
    return f"用户：{r['t3']}\n<resp>"


def to_ids(recs, no_history: bool):
    """把文本编码成 p/t；no_history=True 时 p 只用轮3。就地写回 recs。"""
    for r in recs:
        p_src = no_history_ctx(r) if no_history else r["ctx"]
        r["p"] = enc(p_src)
        r["t"] = enc(r["target"] + d66.EOS)
        r["ntok_all"] = len(r["p"]) + len(r["t"])


# ---------------- 模型 / loss（写法照抄 S65） ----------------
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
        if step % 500 == 0 or step == steps:
            print(f"  [train] seed={seed} step={step}/{steps} loss={loss.item():.4f} "
                  f"elapsed={time.time()-t0:.0f}s", flush=True)
    return gate_res, time.time() - t0


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


# ---------------- 主流程 ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="S67 L2 对话卡片驱动")
    ap.add_argument("--smoke", action="store_true", help="50 步、只评 60 条")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--no-history", action="store_true", help="P5 消融：只喂轮3")
    a = ap.parse_args(argv)
    steps = SMOKE_STEPS if a.smoke else STEPS
    tag = ("nohist" if a.no_history else "hist") + ("-smoke" if a.smoke else "")
    ensure_data()
    train, test = load_jsonl("train"), load_jsonl("test")
    n_vio = sum(int(r["turn3_has_ent"]) for r in train + test)
    print(f"[DATA] train={len(train)} test={len(test)} | verify.66 turn3_has_ent 违例={n_vio}"
          f"（必须 0）| 模板数={d66.N_TEMPLATE_FAMILIES} 实体词表={len(d66.NAMES)}", flush=True)

    to_ids(train, a.no_history)
    to_ids(test, a.no_history)
    over = sum(1 for r in train + test if r["ntok_all"] > MAXLEN)
    print(f"[DATA] arm={'只喂轮3(P5 消融)' if a.no_history else '轮1+轮2+轮3(全历史)'} | "
          f"p 中位={sorted(len(r['p']) for r in train + test)[len(train + test)//2]} "
          f"最大={max(len(r['p']) for r in train + test)} | "
          f"p+t 最大={max(r['ntok_all'] for r in train + test)} 超 MAXLEN={over}", flush=True)

    tr_fp, te_fp = d66.data_fp(train), d66.data_fp(test)
    print(f"[DATA] train_fp={tr_fp} test_fp={te_fp}（与 [D66] 同行口径，arm 不同则指纹不同）",
          flush=True)
    assert n_vio == 0 and over == 0, "数据契约不过：turn3 含实体或超长"

    # ---- 地板（P2；只用输入可见信息 + train 统计） ----
    print("[FLOOR] ★P2：max_naive 含计数类特征（轮3 长度 / 实体个数 / 轮数 / 位置桶）", flush=True)
    _f_full, mxname_full, mx_full, thr_full = d66.compute_floors(
        train, test, tag="test", show_train=True)
    if a.smoke:
        ev = test[:SMOKE_EVAL]
        _f_s, mxname_s, mx_s, thr_s = d66.compute_floors(
            train, ev, tag=f"smoke子集(n={len(ev)})", show_train=False)
        mx, mxname, thr = mx_s, mxname_s, thr_s
    else:
        ev = test
        mx, mxname, thr = mx_full, mxname_full, thr_full

    # ---- 训练 ----
    model = new_pipe()
    n_par = sum(p.numel() for p in model.parameters())
    gate_res, wall = train_run(model, train, a.seed, steps)

    # ---- 评估：配对 CE + 贪心 EM ----
    ce = eval_ce(model, ev)
    cm, cse = mean_se(ce)
    txt = greedy_gen(model, ev, batch=EM_BATCH)
    hits = [em_strict(r["target"], txt[i]) for i, r in enumerate(ev)]
    ent_hit = [int(r["ent"] in txt[i]) for i, r in enumerate(ev)]
    em = 100.0 * sum(hits) / len(hits)
    em_ent = 100.0 * sum(ent_hit) / len(ent_hit)
    se = (em / 100.0 * (1 - em / 100.0) / len(hits)) ** 0.5 * 100.0
    print(f"[MAIN] seed={a.seed} arm={'只喂轮3' if a.no_history else '全历史'} steps={steps} "
          f"batch={BATCH} lr={LR} | 模型=CardPipeline(residual=False) d={D} ff={FF} "
          f"nhead={NHEAD} maxlen={MAXLEN} params={n_par} | 配对CE={cm:.4f}±{cse:.4f} (nats/token)",
          flush=True)
    print(f"[EM] {'smoke子集' if a.smoke else '全量'} n={len(hits)} 严格EM={em:.2f}% "
          f"±SE={se:.2f}pp | 诊断：生成里含正确实体 ent 的比例={em_ent:.2f}% | "
          f"地板 max_naive={mx:.2f}%({mxname}) 门槛={thr:.2f}%", flush=True)

    shown = 0
    for i, r in enumerate(ev):
        if shown >= 6:
            break
        print(f"[GEN] [{i}] t3={r['t3']} | gold={r['target']} | pred={txt[i][:80]} | "
              f"{'对' if hits[i] else '错'}（实体在 pred 里={bool(ent_hit[i])}）", flush=True)
        shown += 1

    # ---- 判据（原文引 L2_PLAN.md §4） ----
    j_p1 = em > thr
    print(f"[CRIT] P1（主）L2_PLAN §4：「卡片组 EM > max_naive + 2×SE，且 2 seed 同号」| "
          f"EM={em:.2f}% vs 门槛(max_naive+2SE)={thr:.2f}% ⇒ {'PASS' if j_p1 else 'FAIL'} | "
          f"「2 seed 同号」本 run 只有 seed={a.seed} ⇒ 待第 2 个 seed 复跑同号", flush=True)
    print(f"[CRIT] P2 L2_PLAN §4：「max_naive 必须含计数类特征（长度 / 实体个数 / 轮数 / 位置桶），"
          f"且逐条报出」⇒ 已逐条报出（上方 [FLOOR]），max_naive={mx:.2f}% 来自 {mxname} ⇒ PASS",
          flush=True)
    j_p3 = gate_res is not None and all(v > 0 for v in gate_res.values())
    print(f"[CRIT] P3 L2_PLAN §4：「契约：数据指纹 [G0] + 三卡梯度非空（[GATE A]）」| "
          f"[DATA] train_fp={tr_fp} | [GATE A] {gate_res} ⇒ "
          f"{'PASS' if j_p3 else 'FAIL'}", flush=True)
    print(f"[CRIT] P4 L2_PLAN §4：「CE 与 EM 并列报」| 配对CE={cm:.4f}±{cse:.4f} nats/token 与 "
          f"EM={em:.2f}% 已并列报出 ⇒ PASS", flush=True)
    if a.no_history:
        print(f"[CRIT] P5 L2_PLAN §4：「去掉轮1/轮2 历史（只喂轮3）⇒ EM 必掉」| "
              f"本 run 就是消融臂：EM={em:.2f}% ⇒ 与同 seed 全历史臂比较（掉 = PASS）", flush=True)
    else:
        print(f"[CRIT] P5 L2_PLAN §4：「去掉轮1/轮2 历史（只喂轮3）⇒ EM 必掉」| "
              f"本 run 是全历史臂 EM={em:.2f}% ⇒ 需 --no-history 同 seed 另跑一次来判", flush=True)

    ck = os.path.join(LOG_DIR, f"67_ckpt_seed{a.seed}_{tag}.pt")
    torch.save(model.state_dict(), ck)
    print(f"[META] tag={tag} seed={a.seed} no_history={a.no_history} steps={steps} "
          f"batch={BATCH} lr={LR} d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN} "
          f"n_params={n_par} 墙钟={wall:.0f}s ckpt={ck}", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
