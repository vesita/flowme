#!/usr/bin/env python3
"""S71 · L3-r1 算术路基线：把 S70 的算术对话数据接进 cards.CardPipeline。

派生自 stages/67_dialog_cards.py（去掉实体/换位/轮次消融逻辑，其余训练与贪心口径逐字沿用）：
  · `CardPipeline(residual=False)`、d=128 ff=512 nhead=4 maxlen=512、batch=32、lr=1e-3、3000 步；
  · teacher forcing **只在目标 span 上算 CE**（`masked_ce`）；
  · 贪心 EM = batch=1、正确列索引（`last_col`）——R28 自洽口径（EM_BATCH=1）。

数据（只读，不造）：`logs/{prefix}_dialog_{train,test}.jsonl`（prefix=70 full / 70s small），
字段沿用 stages/70_arith_dialog_gen.py:43-46 的 ctx/target/gold/gold_int/op/max_digits/...
（本文件先读一眼字段再假设，见 [DATA] 行）。

指标（L3_PLAN.md §8.3 写死）：**从生成文本里抽「结论句最后一个整数」与 gold 整数比较**，
抽不出算错并单独报抽不出比例（仪器质量，R1-c ≤ 1%）。

抽取器来源（实测，不是推断）：
  · 主口径 = `stages/70_arith_dialog_gen.py:131-152 extract_conclusion_int`（importlib 载入，
    **不重写**）：结论句 = 从后往前第一个含 '=' 的非空行 → 该行最后一个整数；
    退化 = 全文最后一个整数；无整数 → None。
  · 交叉校验 = nanoSeek `training/rl/reward/arith_gen.py:73 extract_arith`（importlib 载入）：
    它 (i) 正则 `\\d{1,3}`（arith_gen.py:70）只认 1~3 位操作数、(ii) `arith_rules.toml` 写死
    操作数 <= 100、(iii) 返回的是**它自己算出来的**结果，而不是生成文本里写的整数
    ⇒ 本数据绝大多数是 3~5 位 / 多位中间结果，**不可作主口径**；本 run 用 [XCHK] 行实测
    「能返回非 None 的比例 + 与主口径不一致的比例」，把这条判断钉在本次生成上。

地板：importlib 载入 `stages/70_arith_dialog_gen.py:311 compute_floors`（**不重写**），
在同样的 test 上重算一遍，并与 L3_PLAN.md §8.1 的 r0 实测值（full 0.00% / small 0.00%）对齐报出。

判据（L3_PLAN.md §8.3，逐条打印，**不许为好看而改**）：
  R1-a full：EM > 门槛 + 2SE（预期不高：4–5 位算术对 d=128 很难）
  R1-b（主）small：EM >= 50% 且 2 seed 同号
  R1-c 抽不出比例 <= 1%
  R1-d 目标 span CE 与位数分桶 EM 并列报

用法（本机只有 .venv/bin/python 装了 torch/tokenizers）：
  .venv/bin/python stages/71_arith_dialog_cards.py --data-prefix 70s --seed 1234 --smoke
  .venv/bin/python stages/71_arith_dialog_cards.py --data-prefix 70s --seed 1234
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

# ---------------- 配置（与 S65/S67/S12 一致） ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512
STEPS = 3000
LR = 1e-3
BATCH = 32
GEN_BATCH = 32
EM_BATCH = 1                 # ★主口径：batch=1（R28 自洽写法，照抄 S67）
SMOKE_STEPS = 50
SMOKE_EVAL = 60
LOG_DIR = os.path.join(ROOT, "logs")
DATA_TMPL = os.path.join(LOG_DIR, "{prefix}_dialog_{split}.jsonl")
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
D70_PATH = os.path.join(ROOT, "stages", "70_arith_dialog_gen.py")
ARITH_GEN_PATH = NS_ROOT + "/training/rl/reward/arith_gen.py"

# L3_PLAN.md §8.1 的 r0 实测值（对齐用；引文，不是本脚本算的）
R0_FLOOR = {"70": 0.00, "70s": 0.00}
R0_FP = {"70": ("b5e3e29bd0a8072d", "462a9bf3ad91fa4d"),
         "70s": ("2ce9114fca7d7232", "f52a31145c526521")}


def _load_path(name: str, path: str):
    """按路径加载（文件名以数字开头，不能普通 import）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d70 = _load_path("d70_arith_dialog_gen", D70_PATH)     # extract_conclusion_int / compute_floors / data_fp / EOS
ag = _load_path("ns_arith_gen", ARITH_GEN_PATH)        # extract_arith（仅交叉校验）

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
def load_jsonl(prefix: str, split: str) -> list:
    fp = DATA_TMPL.format(prefix=prefix, split=split)
    with open(fp, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def to_ids(recs):
    """p = enc(ctx)、t = enc(target + <eos>)（与 S70 落盘前的编码口径逐字一致）。"""
    for r in recs:
        r["p"] = enc(r["ctx"])
        r["t"] = enc(r["target"] + d70.EOS)
        r["ntok_all"] = len(r["p"]) + len(r["t"])


# ---------------- 模型 / loss（写法照抄 S67） ----------------
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
    """teacher forcing：只在目标 span [s, s+len(t)) 上算 CE（照抄 S67::masked_ce）。"""
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
            el = time.time() - t0
            print(f"  [train] seed={seed} step={step}/{steps} loss={loss.item():.4f} "
                  f"elapsed={el:.0f}s eta={el / step * (steps - step):.0f}s", flush=True)
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
    """贪心解码，batch=1、正确列索引（照抄 S67::greedy_gen）。"""
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


def em_with_thr(p: float, n: int):
    """二项比例的标准误与 2SE 门槛（口径同 S70::compute_floors）。"""
    se = (p / 100.0 * (1 - p / 100.0) / n) ** 0.5 * 100.0 if n else float("nan")
    return se, p + 2.0 * se


# ---------------- 主流程 ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="S71 L3-r1 算术路卡片基线")
    ap.add_argument("--data-prefix", choices=("70", "70s"), default="70",
                    help="70=full 7000/800；70s=small 1000/240 两位×两位")
    ap.add_argument("--smoke", action="store_true", help="50 步、只评前 60 条（探路）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args(argv)
    steps = SMOKE_STEPS if a.smoke else a.steps
    prefix = a.data_prefix
    mask_tag = "small" if prefix == "70s" else "full"

    train, test = load_jsonl(prefix, "train"), load_jsonl(prefix, "test")
    keys = sorted(train[0].keys())
    print(f"[DATA] prefix={prefix} ({mask_tag}) train={len(train)} test={len(test)} | 字段={keys}",
          flush=True)
    # 字段契约：抽取器与地板函数都依赖这几个键
    need = ("ctx", "target", "gold", "gold_int", "op", "max_digits", "type_prefix",
            "nums", "n_nums", "resp_bucket", "resp_chars")
    miss_keys = [k for k in need if k not in train[0]]
    assert not miss_keys, f"数据缺字段 {miss_keys}"

    to_ids(train)
    to_ids(test)
    over = sum(1 for r in train + test if r["ntok_all"] > MAXLEN)
    tr_fp, te_fp = d70.data_fp(train), d70.data_fp(test)
    fp_ok = (tr_fp, te_fp) == R0_FP[prefix]
    _mid = (len(train) + len(test)) // 2
    print(f"[DATA] p 中位={sorted(len(r['p']) for r in train + test)[_mid]} "
          f"最大={max(len(r['p']) for r in train + test)} | p+t 中位="
          f"{sorted(r['ntok_all'] for r in train + test)[_mid]} "
          f"最大={max(r['ntok_all'] for r in train + test)} 超 MAXLEN={over}", flush=True)
    print(f"[DATA] train_fp={tr_fp} test_fp={te_fp} | r0 实测={R0_FP[prefix][0]}/"
          f"{R0_FP[prefix][1]} ⇒ {'一致' if fp_ok else '★不一致（口径或数据版本不同）'}"
          f" | gold_int 抽不出={sum(1 for r in train + test if r['gold_int'] is None)}",
          flush=True)
    assert over == 0, "数据超长"

    # ---- 地板（R1 的 门槛；importlib 复用 S70，不重写） ----
    print(f"[FLOOR] ★重算 r0 地板（复用 S70::compute_floors，同 test 同口径）", flush=True)
    if a.smoke:
        ev = test[:SMOKE_EVAL]
        _em_s, mxname, mx, miss_gold, _miss = d70.compute_floors(train, ev, f"{prefix}-smoke(n={len(ev)})")
    else:
        ev = test
        _em_f, mxname, mx, miss_gold, _miss = d70.compute_floors(train, test, f"{prefix}-test")
    se_mx, thr = em_with_thr(mx, len(ev))
    r0v = R0_FLOOR[prefix]
    print(f"[FLOOR] {prefix} max_naive={mx:.2f}%({mxname}) vs r0 实测={r0v:.2f}% ⇒ "
          f"{'一致' if abs(mx - r0v) < 1e-9 else f'★差异 {mx - r0v:+.2f}pp（如实报）'}"
          f" | 2SE={se_mx:.2f}pp 门槛=max_naive+2SE={thr:.2f}% | gold_int 抽不出={miss_gold}",
          flush=True)

    # ---- 训练 ----
    model = new_pipe()
    n_par = sum(p.numel() for p in model.parameters())
    gate_res, wall = train_run(model, train, a.seed, steps)

    # ---- 评估：配对 CE + 贪心 EM（抽「结论句最后一个整数」） ----
    t_eval = time.time()
    ce = eval_ce(model, ev)
    cm, cse = mean_se(ce)
    txt = greedy_gen(model, ev, batch=EM_BATCH)
    eval_sec = time.time() - t_eval

    gold_int = [r["gold_int"] for r in ev]
    pred_int, n_miss = [], 0
    for t in txt:
        pi = d70.extract_conclusion_int(t)
        pred_int.append(pi)
        n_miss += int(pi is None)
    hits = [int(pi is not None and pi == gi) for pi, gi in zip(pred_int, gold_int)]
    em = 100.0 * sum(hits) / len(hits)
    se, thr_em = em_with_thr(em, len(hits))
    miss_rate = 100.0 * n_miss / len(hits)

    print(f"[MAIN] prefix={prefix}({mask_tag}) seed={a.seed} steps={steps} batch={BATCH} lr={LR} "
          f"| 模型=CardPipeline(residual=False) d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN} "
          f"params={n_par} | 配对CE={cm:.4f}±{cse:.4f} (nats/token) | 评估墙钟={eval_sec:.0f}s",
          flush=True)
    print(f"[EM] {'smoke子集' if a.smoke else '全量'} n={len(hits)} 严格EM={em:.2f}% ±SE={se:.2f}pp "
          f"| 抽不出={n_miss}（{miss_rate:.2f}%）算错 | 主口径="
          f"70_arith_dialog_gen.py:131 extract_conclusion_int", flush=True)

    # ---- [XCHK] nanoSeek extract_arith 在主口径生成的文本上能复用多少（实测，不是推断） ----
    x_ok = 0
    x_dis = 0
    for pi, t in zip(pred_int, txt):
        r = ag.extract_arith(t)          # arith_gen.py:73
        if r is not None:
            x_ok += 1
            if pi is None or r[3] != pi:
                x_dis += 1
    print(f"[XCHK] arith_gen.py:73 extract_arith 在生成文本上返回非 None={x_ok}/{len(txt)}"
          f"（{100.0 * x_ok / len(txt):.2f}%）| 其中与主口径结论整数不一致={x_dis}"
          f" ⇒ 它返回的是自己按首个算式算出的中间值（\\d{{1,3}} + max_num=100），"
          f"不可作主口径（与 S70 文件头实测一致）", flush=True)

    # ---- [DIAG] 位数分桶 EM（题面最大位 / 运算符） ----
    print("[DIAG] 位数分桶 EM（题面最大位数 max_digits；抽不出算错）:", flush=True)
    for keyname, keyfn in (("max_digits", lambda r: r["max_digits"]),
                           ("op", lambda r: r["op"])):
        buckets = {}
        for i, r in enumerate(ev):
            b = keyfn(r)
            h, n = buckets.get(b, (0, 0))
            buckets[b] = (h + hits[i], n + 1)
        line = " | ".join(f"{keyname}={k}: {100.0 * h / n:.2f}%(n={n})"
                          for k, (h, n) in sorted(buckets.items(), key=lambda kv: str(kv[0])))
        print(f"[DIAG] {line}", flush=True)

    # ---- [GEN] 前 3 条 ----
    for i in range(min(3, len(ev))):
        r = ev[i]
        q = r["q"] if "q" in r else r["ctx"].replace("\n<resp>", "")
        print(f"[GEN] [{i}] 题干={q[:90]} | gold_int={r['gold_int']} | "
              f"gold={r['target'][:60]!r} | pred={txt[i][:90]!r} | pred_int={pred_int[i]} | "
              f"{'对' if hits[i] else '错'}", flush=True)

    # ---- 判据（原文引 L3_PLAN.md §8.3） ----
    if prefix == "70":
        j_a = em > thr
        print(f"[CRIT] R1-a L3_PLAN §8.3「full：EM > 门槛 + 2SE」| EM={em:.2f}% vs "
              f"门槛={thr:.2f}% ⇒ {'PASS' if j_a else 'FAIL'}"
              f"（预期不高：4–5 位算术对 d=128 很难；不过也如实报，不调参）", flush=True)
        print(f"[CRIT] R1-b L3_PLAN §8.3「small：EM >= 50% 且 2 seed 同号」| 本 run 前缀=70(full)，"
              f"不是 small 口径 ⇒ 不判（R1-b 由 --data-prefix 70s 的两次 run 判）", flush=True)
    else:
        j_b = em >= 50.0
        print(f"[CRIT] R1-b（主）L3_PLAN §8.3「small：EM >= 50% 且 2 seed 同号」| 本 run "
              f"seed={a.seed} EM={em:.2f}% vs 门槛 50.00% ⇒ {'PASS' if j_b else 'FAIL'}"
              f"（单 seed 判定；「2 seed 同号」需 70s 的第二个 seed 复跑同号后再判）", flush=True)
        print(f"[CRIT] R1-a L3_PLAN §8.3「full：EM > 门槛 + 2SE」| 本 run 前缀=70s(small)，"
              f"不是 full 口径 ⇒ 不判（R1-a 由 --data-prefix 70 的 run 判）", flush=True)
    j_c = miss_rate <= 1.0
    print(f"[CRIT] R1-c L3_PLAN §8.3「抽不出比例 <= 1%」| 抽不出={n_miss}/{len(hits)}="
          f"{miss_rate:.2f}% ⇒ {'PASS' if j_c else 'FAIL'}（仪器质量）", flush=True)
    print(f"[CRIT] R1-d L3_PLAN §8.3「目标 span CE 与位数分桶 EM 并列报」| 配对CE={cm:.4f}"
          f"±{cse:.4f} nats/token 与上方 [DIAG] 位数/运算符分桶 EM 已并列报出 ⇒ PASS(必报项)",
          flush=True)
    print(f"[CRIT] 数据契约：train_fp={tr_fp} test_fp={te_fp} 与 r0 "
          f"{'一致' if fp_ok else '★不一致'}；[GATE A] {gate_res} ⇒ "
          f"{'PASS' if fp_ok and gate_res and all(v > 0 for v in gate_res.values()) else 'FAIL'}",
          flush=True)

    ck = os.path.join(LOG_DIR, f"71_ckpt_{prefix}_seed{a.seed}{'_smoke' if a.smoke else ''}.pt")
    torch.save(model.state_dict(), ck)
    print(f"[META] prefix={prefix}({mask_tag}) seed={a.seed} steps={steps} batch={BATCH} lr={LR} "
          f"d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN} n_params={n_par} EM_BATCH={EM_BATCH} "
          f"smoke={a.smoke} n_eval={len(hits)} 训练墙钟={wall:.0f}s 评估墙钟={eval_sec:.0f}s "
          f"总墙钟={wall + eval_sec:.0f}s ckpt={ck}", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
