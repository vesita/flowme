#!/usr/bin/env python3
"""S73 · L3-r2b 自然语言多轮对话卡片基线：跨轮整合 + 历史消融。

数据（只读，不造）：`logs/72_dialog_{train,test}.jsonl`（7000/800；--data-prefix 72s ⇒ 200/100）。
  字段见 stages/72_nl_dialog_gen.py:32-38；本文件只用 ctx / target / t3 / ent / cond / family / t3_idx。
  地板与指纹 importlib 载入 `stages/72_nl_dialog_gen.py`（**不重写**）：compute_floors / data_fp / file_sha256。

模型/口径（L3_PLAN.md §9.4 跑前已落盘，骨架逐字沿用 stages/71_arith_dialog_cards.py）：
  `CardPipeline(residual=False)`、d=128 ff=512 nhead=4 maxlen=512、batch=32、lr=1e-3、3000 步；
  teacher forcing **只在目标 span 上算 CE**；贪心 EM = batch=1、正确列索引（R28 自洽口径，EM_BATCH=1）。

指标（主）= **回复文本严格 EM**（strip 后逐字相等，照 L2 的 `em_strict`，stages/67_dialog_cards.py:268）。
  另报（R2-e）：① 生成含正确实体比例 ② 生成含正确条件比例 ③ 按族分桶 EM ④ 按轮3句式分桶 EM
  （含 族×轮3句式 20 格）。

历史消融 `--no-history`（R2-d）：ctx 换成**只有轮3**（`用户：<t3>\\n<resp>`），其余逐字不变；
  并打印「消融后实体/条件是否还在输入里」的**实测比例**（文本级 + token 连续子序列级，应为 0/0）。

判据（L3_PLAN.md §9.4，逐条打印 [CRIT]，**不许为好看而调参 / 改口径 / 放宽 EM**）：
  R2-c test EM > 门槛 0.81%（=§9.3 max_naive 0.38% + 2SE 0.43pp）+ 2SE，且 2 seed 同号；
  R2-d 历史消融 EM 必掉（掉幅 ≥ 2SE 或落到地板附近）—— 本任务的设计就是不许背模板拿分；
  R2-e 实体命中率 / 条件命中率 / 分桶 EM（族 / 轮3句式）并列报。

用法（本机只有 .venv/bin/python 装了 torch/tokenizers）：
  .venv/bin/python stages/73_nl_dialog_cards.py --smoke
  .venv/bin/python stages/73_nl_dialog_cards.py --seed 1234
  .venv/bin/python stages/73_nl_dialog_cards.py --seed 5678 --pair-log logs/73_s1234.log
  .venv/bin/python stages/73_nl_dialog_cards.py --seed 1234 --no-history --pair-log logs/73_s1234.log
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
import re
import sys
import time
from collections import Counter

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
if NS_ROOT not in sys.path:
    sys.path.insert(0, NS_ROOT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from cards import CardPipeline  # noqa: E402

# ---------------- 配置（与 S65/S67/S71 一致） ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512
STEPS = 3000
LR = 1e-3
BATCH = 32
GEN_BATCH = 32
EM_BATCH = 1                 # ★主口径：batch=1（R28 自洽写法，照抄 S71）
SMOKE_STEPS = 50
SMOKE_EVAL = 60
LOG_DIR = os.path.join(ROOT, "logs")
DATA_TMPL = os.path.join(LOG_DIR, "{prefix}_dialog_{split}.jsonl")
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
D72_PATH = os.path.join(ROOT, "stages", "72_nl_dialog_gen.py")

# L3_PLAN.md §9.3 的 r2 数据判定实测值（引文，用作对齐；不是本脚本算的）
R2_MAX_NAIVE = 0.38          # ⑤e/⑥ 模板填空只用轮3（实体/条件填 train 族内众数）
R2_FLOOR_2SE = 0.43          # pp
R2_THRESHOLD = 0.81          # % = max_naive + 2SE（§9.3 写死的 R2-b 门槛）
R2_REF_UPPER = 100.00        # 参考上限（完美实体+完美条件+DET句面）
R2_FP = ("5e040a84804fefe0", "8f54637ea07fcde8")     # train / test（§9.3）
R2_SHA_PREFIX = ("af327cfd", "50f527af")             # train / test jsonl sha256 前 8 位（§9.3）


def _load_path(name: str, path: str):
    """按路径加载（文件名以数字开头，不能普通 import）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d72 = _load_path("d72_nl_dialog_gen", D72_PATH)   # compute_floors / data_fp / file_sha256 / EOS

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
UNK_ID = tok.token_to_id("<unk>")
RESP_ID = tok.token_to_id("<resp>")
BOS_ID = tok.token_to_id("<bos>")
SPECIAL_NAMES = {PAD_ID: "<pad>", EOS_ID: "<eos>", UNK_ID: "<unk>", RESP_ID: "<resp>", BOS_ID: "<bos>"}


def enc(text: str) -> list:
    return tok.encode(text, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


# ---------------- 数据 ----------------
def load_jsonl(prefix: str, split: str) -> list:
    fp = DATA_TMPL.format(prefix=prefix, split=split)
    with open(fp, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def no_history_src(r) -> str:
    """R2-d 消融输入：只喂轮3（不喂轮1/轮2），格式仍是「用户：… / <resp>」（照 stages/67:118-120）。"""
    return f"用户：{r['t3']}\n<resp>"


def prompt_src(r, no_history: bool) -> str:
    return no_history_src(r) if no_history else r["ctx"]


def to_ids(recs, no_history: bool):
    """p = enc(输入文本)、t = enc(target + <eos>)（与 S72 落盘前的编码口径逐字一致）。

    `p_hist` 永远是**全历史 ctx** 的编码：数据指纹 data_fp 依赖 p，若指纹跟着消融臂变就没法
    与 §9.3 对齐（消融臂的 fp 不是数据变了，是 prompt 构造变了）⇒ 指纹/长度一律走 p_hist。
    """
    for r in recs:
        r["p_hist"] = enc(r["ctx"])
        r["p"] = enc(no_history_src(r)) if no_history else r["p_hist"]
        r["t"] = enc(r["target"] + d72.EOS)
        r["ntok_all"] = len(r["p"]) + len(r["t"])
        r["ntok_hist"] = len(r["p_hist"]) + len(r["t"])


def _is_subseq(needle: list, hay: list) -> bool:
    """needle 是否为 hay 的**连续**子序列（token 级二次核验，防「字面包含但编码丢失」）。"""
    if not needle:
        return False
    n = len(needle)
    return any(hay[i: i + n] == needle for i in range(len(hay) - n + 1))


def in_input_ratios(recs, no_history: bool):
    """实测：实体 / 条件 是否还在**真正喂进去的输入文本**里（文本级 + token 连续子序列级）。"""
    n = len(recs)
    ent_t = cond_t = ent_k = cond_k = 0
    for r in recs:
        src = prompt_src(r, no_history)
        pid = enc(src)
        ent_t += int(r["ent"] in src)
        cond_t += int(r["cond"] in src)
        ent_k += int(_is_subseq(enc(r["ent"]), pid))
        cond_k += int(_is_subseq(enc(r["cond"]), pid))
    return n, ent_t, cond_t, ent_k, cond_k


# ---------------- 模型 / loss（写法照抄 S67/S71） ----------------
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
def greedy_gen(model, recs, batch: int = 1, id_counter: Counter = None) -> list:
    """贪心解码，batch=1、正确列索引（照抄 S67/S71::greedy_gen）。

    id_counter：非 EOS 生成 token 的计数（仪器行用：查 <unk>/<resp> 等特殊 token 有没有被
    `dec(skip_special_tokens=True)` 悄悄吃掉 —— L3_PLAN §9.3 的探针坑）。
    """
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
            g = [x for x in g if x != EOS_ID]
            texts[i + j] = dec(g)
            if id_counter is not None:
                id_counter.update(g)
    model.train()
    return texts


def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, math.sqrt(var / n) if n else float("nan")


def em_strict(gold: str, pred: str) -> int:
    """严格 EM：模板+槽位确定性生成 ⇒ 剥掉首尾空白后逐字符相等（照 L2 stages/67:268-270）。"""
    return int((gold or "").strip() == (pred or "").strip())


def em_with_thr(p: float, n: int):
    """二项比例的标准误与 2SE 门槛（口径同 S71::em_with_thr / S72::compute_floors）。"""
    se = (p / 100.0 * (1 - p / 100.0) / n) ** 0.5 * 100.0 if n else float("nan")
    return se, p + 2.0 * se


def near_quoted(a: float, b: float, tol: float = 0.005) -> bool:
    """与 §9.3 的**两位小数引文**比较：|a-b| < 半个最小位（0.005pp）。

    §9.3 的 0.38% / 0.81% 是四舍五入后的引文（原始值是 3/800=0.375% 等），
    用 1e-9 严格相等去比会把「引文精度」误报成数据差异 ⇒ 只允许在引文精度上判一致性。
    """
    return abs(a - b) < tol + 1e-12


def parse_em_from_log(path: str):
    """从另一份 73 日志里读 [EM] 行的 严格EM / SE（R2-d 配对判定用；读不到返回 None）。"""
    try:
        with open(path, encoding="utf-8") as f:
            for ln in f:
                if ln.startswith("[EM] ") and "严格EM=" in ln:
                    m = re.search(r"严格EM=([\d.]+)% ±SE=([\d.]+)pp", ln)
                    if m:
                        return float(m.group(1)), float(m.group(2))
    except OSError:
        return None
    return None


# ---------------- 主流程 ----------------
def main(argv=None):
    ap = argparse.ArgumentParser(description="S73 L3-r2b 自然语言多轮对话卡片基线 + 历史消融")
    ap.add_argument("--data-prefix", choices=("72", "72s"), default="72",
                    help="72=full 7000/800；72s=smoke 小量 200/100")
    ap.add_argument("--smoke", action="store_true", help="50 步、只评前 60 条（探路）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--no-history", action="store_true",
                    help="R2-d 消融：ctx 只喂轮3（用户：<t3>\\n<resp>），其余逐字不变")
    ap.add_argument("--pair-log", default=None,
                    help="R2-d 配对用：同 seed 全历史臂的 73 日志路径（读它的 [EM] 行算掉幅；可选）")
    a = ap.parse_args(argv)
    steps = SMOKE_STEPS if a.smoke else a.steps
    prefix = a.data_prefix
    arm = "nohist" if a.no_history else "hist"
    smoke_note = "（--smoke 探路，不作判据）" if a.smoke else ""

    train, test = load_jsonl(prefix, "train"), load_jsonl(prefix, "test")
    keys = sorted(train[0].keys())
    print(f"[DATA] prefix={prefix} arm={arm} train={len(train)} test={len(test)} | 字段={keys}",
          flush=True)
    need = ("ctx", "target", "t3", "ent", "cond", "family", "tpl_id", "t3_idx")
    miss_keys = [k for k in need if k not in train[0]]
    assert not miss_keys, f"数据缺字段 {miss_keys}"

    to_ids(train, a.no_history)
    to_ids(test, a.no_history)
    over = sum(1 for r in train + test if r["ntok_hist"] > MAXLEN)
    over_arm = sum(1 for r in train + test if r["ntok_all"] > MAXLEN)
    fp_recs = [{"p": r["p_hist"], "t": r["t"]} for r in train + test]
    tr_fp = d72.data_fp(fp_recs[:len(train)])
    te_fp = d72.data_fp(fp_recs[len(train):])
    fp_ok = (tr_fp, te_fp) == R2_FP
    sha_tr = d72.file_sha256(DATA_TMPL.format(prefix=prefix, split="train"))
    sha_te = d72.file_sha256(DATA_TMPL.format(prefix=prefix, split="test"))
    sha_ok = (sha_tr.startswith(R2_SHA_PREFIX[0]), sha_te.startswith(R2_SHA_PREFIX[1]))
    _mid = (len(train) + len(test)) // 2
    print(f"[DATA] 全历史 ctx（指纹口径）p 中位="
          f"{sorted(len(r['p_hist']) for r in train + test)[_mid]} "
          f"最大={max(len(r['p_hist']) for r in train + test)} | p+t 中位="
          f"{sorted(r['ntok_hist'] for r in train + test)[_mid]} "
          f"最大={max(r['ntok_hist'] for r in train + test)} 超 MAXLEN={over}", flush=True)
    if a.no_history:
        print(f"[DATA] 消融臂实际输入 p 中位="
              f"{sorted(len(r['p']) for r in train + test)[_mid]} "
              f"最大={max(len(r['p']) for r in train + test)} | p+t 最大="
              f"{max(r['ntok_all'] for r in train + test)} | 消融臂自己超 MAXLEN={over_arm}",
              flush=True)
    print(f"[DATA] train_fp={tr_fp} test_fp={te_fp} vs §9.3 {R2_FP[0]}/{R2_FP[1]} ⇒ "
          f"{'一致' if fp_ok else '★不一致（口径或数据版本不同，如实报）'} | "
          f"（指纹一律按全历史 ctx 算，消融臂不改指纹）| "
          f"jsonl sha256 前 8 位 train={sha_tr[:8]} test={sha_te[:8]} vs §9.3 "
          f"{R2_SHA_PREFIX[0]}/{R2_SHA_PREFIX[1]} ⇒ "
          f"{'一致' if all(sha_ok) else '★不一致（如实报）'}", flush=True)
    assert over == 0, "数据超长"

    # ---- [NO-HISTORY] R2-d 的仪器自检：消融后实体/条件还在不在输入里（实测，不是推断） ----
    print("[NO-HISTORY] 输入文本实测（文本级包含 / token 连续子序列级）——「信息真的被拿掉了吗」:",
          flush=True)
    for split, recs in (("test", test), ("train", train)):
        n0, e0, c0, ek0, ck0 = in_input_ratios(recs, False)
        n1, e1, c1, ek1, ck1 = in_input_ratios(recs, True)
        print(f"[NO-HISTORY] {split} n={n0} | 全历史臂: 实体在输入={e0}/{n0}"
              f"（{100.0 * e0 / n0:.2f}%）条件在输入={c0}/{n0}（{100.0 * c0 / n0:.2f}%）"
              f" [token级 {100.0 * ek0 / n0:.2f}%/{100.0 * ck0 / n0:.2f}%] || "
              f"消融臂: 实体在输入={e1}/{n1}（{100.0 * e1 / n1:.2f}%）"
              f"条件在输入={c1}/{n1}（{100.0 * c1 / n1:.2f}%）"
              f" [token级 {100.0 * ek1 / n1:.2f}%/{100.0 * ck1 / n1:.2f}%]"
              f" ⇒ {'消融生效(0/0)' if e1 == 0 and c1 == 0 else '★消融没拿掉信息（如实报）'}",
              flush=True)
    print(f"[NO-HISTORY] 本 run arm={arm}：输入={('只喂轮3 `用户：<t3>\\n<resp>`' if a.no_history else '轮1+轮2+轮3 全历史 ctx')}",
          flush=True)

    # ---- 地板（R2-c 的门槛；importlib 复用 S72::compute_floors，不重写） ----
    print("[FLOOR] ★重算 r2 地板（复用 stages/72_nl_dialog_gen.py::compute_floors，同 train 同 test 同口径）",
          flush=True)
    ev = test[:SMOKE_EVAL] if a.smoke else test
    rows_test, mx_name, mx, ref_em, _rows_train = d72.compute_floors(train, ev, f"{prefix}-{arm}-smoke(n={len(ev)})" if a.smoke else f"{prefix}-{arm}")
    se_mx, thr = em_with_thr(mx, len(ev))
    two_se = 2.0 * se_mx
    if a.smoke:
        print(f"[FLOOR] 复算(本 run 口径 n={len(ev)}) max_naive={mx:.3f}%({mx_name}) | "
              f"1SE={se_mx:.3f}pp 2SE={two_se:.2f}pp 门槛=max_naive+2SE={thr:.3f}% | "
              f"参考上限={ref_em:.2f}%", flush=True)
        print(f"[FLOOR] §9.3 引文：max_naive={R2_MAX_NAIVE:.2f}%(⑤e/⑥) 2SE={R2_FLOOR_2SE:.2f}pp ⇒ "
              f"R2-b 门槛={R2_THRESHOLD:.2f}% 参考上限={R2_REF_UPPER:.2f}% ⇒ "
              f"★本行是 smoke 子集口径（n={len(ev)}），不作与 §9.3 的对齐判定", flush=True)
    else:
        ok_mx = near_quoted(mx, R2_MAX_NAIVE)
        ok_se = near_quoted(two_se, R2_FLOOR_2SE)
        ok_thr = near_quoted(thr, R2_THRESHOLD)
        ok_ref = near_quoted(ref_em, R2_REF_UPPER)
        print(f"[FLOOR] 复算 max_naive={mx:.3f}%（打印两位 {mx:.2f}%，来自 {mx_name}，"
              f"计数={round(mx / 100.0 * len(ev))}/{len(ev)}）vs §9.3 引文 {R2_MAX_NAIVE:.2f}% ⇒ "
              f"{'一致（差异在引文精度内）' if ok_mx else f'★差异 {mx - R2_MAX_NAIVE:+.3f}pp（如实报）'} | "
              f"复算 1SE={se_mx:.3f}pp 2SE={two_se:.2f}pp vs §9.3 2SE {R2_FLOOR_2SE:.2f}pp ⇒ "
              f"{'一致' if ok_se else f'★差异 {two_se - R2_FLOOR_2SE:+.2f}pp（如实报）'} | "
              f"复算 门槛={thr:.3f}%（打印两位 {thr:.2f}%）vs §9.3 引文 {R2_THRESHOLD:.2f}% ⇒ "
              f"{'一致（差异在引文精度内）' if ok_thr else f'★差异 {thr - R2_THRESHOLD:+.3f}pp（如实报）'} | "
              f"复算 参考上限={ref_em:.2f}% vs §9.3 {R2_REF_UPPER:.2f}% ⇒ "
              f"{'一致' if ok_ref else f'★差异 {ref_em - R2_REF_UPPER:+.2f}pp（如实报）'} | "
              f"（对齐只在 §9.3 的两位小数精度上判；原始值是 3/800=0.375% 这类）", flush=True)

    # ---- 训练 ----
    model = new_pipe()
    n_par = sum(p.numel() for p in model.parameters())
    gate_res, wall = train_run(model, train, a.seed, steps)

    # ---- 评估：配对 CE + 贪心严格 EM + 仪器计数 ----
    t_eval = time.time()
    ce = eval_ce(model, ev)
    cm, cse = mean_se(ce)
    id_counter = Counter()
    txt = greedy_gen(model, ev, batch=EM_BATCH, id_counter=id_counter)
    eval_sec = time.time() - t_eval

    hits = [em_strict(r["target"], txt[i]) for i, r in enumerate(ev)]
    em = 100.0 * sum(hits) / len(hits)
    se, thr_strict = em_with_thr(em, len(hits))
    ent_hits = [int(r["ent"] in txt[i]) for i, r in enumerate(ev)]
    cond_hits = [int(r["cond"] in txt[i]) for i, r in enumerate(ev)]
    em_ent = 100.0 * sum(ent_hits) / len(ent_hits)
    em_cond = 100.0 * sum(cond_hits) / len(cond_hits)

    spec_hits = {SPECIAL_NAMES[i]: c for i, c in id_counter.items() if i in SPECIAL_NAMES}
    print(f"[MAIN] prefix={prefix} arm={'只喂轮3(R2-d 消融)' if a.no_history else '全历史'} seed={a.seed} "
          f"steps={steps} batch={BATCH} lr={LR} | 模型=CardPipeline(residual=False) d={D} ff={FF} "
          f"nhead={NHEAD} maxlen={MAXLEN} params={n_par} | 配对CE={cm:.4f}±{cse:.4f} (nats/token) | "
          f"评估墙钟={eval_sec:.0f}s", flush=True)
    print(f"[EM] {'smoke子集' if a.smoke else '全量'} n={len(hits)} 严格EM={em:.2f}% ±SE={se:.2f}pp "
          f"| 主口径=strip 后逐字相等（照 L2 stages/67:268 em_strict）| 地板 max_naive={mx:.2f}% "
          f"2SE={se_mx:.2f}pp 门槛={thr:.2f}% | 生成里被 decode 丢掉的特殊 token={spec_hits}",
          flush=True)

    # ---- [DIAG] R2-e：实体/条件命中 + 族 × 轮3句式分桶 EM ----
    print(f"[DIAG] 实体命中（生成含正确实体，文本包含）={em_ent:.2f}%（{sum(ent_hits)}/{len(hits)}）| "
          f"条件命中（生成含正确条件）={em_cond:.2f}%（{sum(cond_hits)}/{len(hits)}）", flush=True)

    def bucket_line(name, keyfn, fmt=str):
        buckets = {}
        for i, r in enumerate(ev):
            b = keyfn(r)
            h, n = buckets.get(b, (0, 0))
            buckets[b] = (h + hits[i], n + 1)
        line = " | ".join(f"{name}={fmt(k)}: {100.0 * h / n:.2f}%(n={n})"
                          for k, (h, n) in sorted(buckets.items(), key=lambda kv: str(kv[0])))
        print(f"[DIAG] {line}", flush=True)
        return buckets

    bucket_line("族", lambda r: r["family"])
    bucket_line("轮3句式", lambda r: r["t3_idx"])
    cells = bucket_line("族#轮3句式", lambda r: r["tpl_id"])
    exp_cells = len({(r["family"], r["t3_idx"]) for r in train})
    print(f"[DIAG] 族×轮3句式格子覆盖={len(cells)}/{exp_cells}（train 侧出现过）⇒ "
          f"{'PASS 并列报齐' if len(cells) == exp_cells else '★有空格子（如实报）'}", flush=True)

    # ---- [GEN] 前 3 条 ----
    for i in range(min(3, len(ev))):
        r = ev[i]
        print(f"[GEN] [{i}] t3={r['t3']!r} | ent={r['ent']} cond={r['cond']} family={r['family']} "
              f"tpl_id={r['tpl_id']} | gold={r['target']!r} | pred={txt[i]!r} | "
              f"实体{'对' if ent_hits[i] else '错'}/条件{'对' if cond_hits[i] else '错'}/"
              f"EM{'对' if hits[i] else '错'}", flush=True)

    # ---- 判据（原文引 L3_PLAN.md §9.4，逐条打印，不许为好看而改） ----
    if a.no_history:
        print(f"[CRIT] R2-c L3_PLAN §9.4「test EM > 门槛(0.81%) + 2SE，且 2 seed 同号」| 本 run 是 "
              f"R2-d 消融臂（--no-history）⇒ **不判 R2-c**（R2-c 由 seed1234/seed5678 两个全历史臂判）"
              f"{smoke_note}", flush=True)
    else:
        j_c = em > R2_THRESHOLD
        j_c_strict = em > thr_strict
        print(f"[CRIT] R2-c L3_PLAN §9.4「test EM > 门槛(0.81%) + 2SE，且 2 seed 同号」| 本 run "
              f"seed={a.seed} 全历史 arm EM={em:.2f}% vs 门槛={R2_THRESHOLD:.2f}%"
              f"（=§9.3 max_naive {R2_MAX_NAIVE:.2f}% + 2SE {R2_FLOOR_2SE:.2f}pp）"
              f" ⇒ {'PASS' if j_c else 'FAIL'} | 严读（门槛 + 本 run 2SE={2 * se:.2f}pp = "
              f"{thr_strict:.2f}%）⇒ {'PASS' if j_c_strict else 'FAIL'}；"
              f"「2 seed 同号」需 seed1234 与 seed5678 两份日志并列后判（单 run 只给一半）"
              f"{smoke_note}", flush=True)
        paired_c = parse_em_from_log(a.pair_log) if a.pair_log else None
        if paired_c is not None:
            em_o, se_o = paired_c
            same = (em > R2_THRESHOLD) and (em_o > R2_THRESHOLD)
            print(f"[CRIT] R2-c「2 seed 同号」配对（--pair-log={a.pair_log}）：另一臂 EM={em_o:.2f}%"
                  f"±{se_o:.2f}pp 与本 run EM={em:.2f}%±{se:.2f}pp ⇒ 两臂都 > 门槛 "
                  f"{R2_THRESHOLD:.2f}%：{'PASS' if same else 'FAIL'}（同号）{smoke_note}", flush=True)

    if a.no_history:
        near_floor = em <= max(mx, R2_THRESHOLD) + 2.0 * se
        paired = parse_em_from_log(a.pair_log) if a.pair_log else None
        if paired is None:
            print(f"[CRIT] R2-d L3_PLAN §9.4「历史消融（只喂轮3）⇒ EM 必掉（掉幅 >= 2SE 或落到地板附近）」"
                  f"| 本 run seed={a.seed} 消融臂 EM={em:.2f}% ±SE={se:.2f}pp | 地板 max_naive="
                  f"{mx:.2f}%/门槛={thr:.2f}% ⇒ 自身{'已落到地板附近' if near_floor else '未落到地板附近'}；"
                  f"「掉幅 >= 2SE」需与 seed={a.seed} 全历史臂 EM 并列后算"
                  f"（本 run {'未给 --pair-log，只给消融侧' if not a.pair_log else f'给不出 --pair-log={a.pair_log} 的可读 [EM] 行'}）"
                  f"{smoke_note}", flush=True)
        else:
            em_h, se_h = paired
            drop = em_h - em
            se_diff = math.hypot(se_h, se)
            j_drop = drop >= 2.0 * se_diff
            j_floor = near_floor
            print(f"[CRIT] R2-d L3_PLAN §9.4「历史消融（只喂轮3）⇒ EM 必掉（掉幅 >= 2SE 或落到地板附近）」"
                  f"| 配对（--pair-log={a.pair_log}）：全历史臂 EM={em_h:.2f}%±{se_h:.2f}pp vs "
                  f"消融臂 EM={em:.2f}%±{se:.2f}pp ⇒ 掉幅={drop:.2f}pp | 对照 2SE："
                  f"差值SE=sqrt({se_h:.2f}^2+{se:.2f}^2)={se_diff:.2f}pp ⇒ 2SE差值={2 * se_diff:.2f}pp、"
                  f"2SE(全历史臂)={2 * se_h:.2f}pp | 地板 max_naive={mx:.2f}%/门槛={thr:.2f}% ⇒ "
                  f"消融臂{'已落到地板附近' if j_floor else '未落到地板附近'} | "
                  f"掉幅>=2SE差值：{'PASS' if j_drop else 'FAIL'}；落到地板附近："
                  f"{'PASS' if j_floor else 'FAIL'} ⇒ R2-d "
                  f"{'PASS' if (j_drop or j_floor) else 'FAIL'}{smoke_note}", flush=True)
    else:
        print(f"[CRIT] R2-d L3_PLAN §9.4「历史消融（只喂轮3）⇒ EM 必掉（掉幅 >= 2SE 或落到地板附近）」"
              f"| 本 run seed={a.seed} 是全历史臂 EM={em:.2f}% ⇒ **本 run 不判 R2-d**"
              f"（需同 seed 跑 --no-history；本单元由 seed1234 --no-history 那次给消融侧）"
              f"{smoke_note}", flush=True)

    r2e_ok = len(cells) == exp_cells
    print(f"[CRIT] R2-e L3_PLAN §9.4「实体命中率与条件命中率并列报；分桶 EM 并列报（族 / 轮3句式）」"
          f"| 实体命中={em_ent:.2f}% 条件命中={em_cond:.2f}% | 按族 "
          f"{len({r['family'] for r in ev})} 桶、按轮3句式 {len({r['t3_idx'] for r in ev})} 桶、"
          f"族×轮3句式 {len(cells)}/{exp_cells} 格 已并列报出 ⇒ "
          f"{'PASS(必报项)' if r2e_ok else '★PARTIAL(有空格子，如实报)'}{smoke_note}", flush=True)

    print(f"[CRIT] 数据契约：train_fp={tr_fp} test_fp={te_fp} 与 §9.3 "
          f"{'一致' if fp_ok else '★不一致'}；jsonl sha256 "
          f"{'一致' if all(sha_ok) else '★不一致'}；[GATE A] {gate_res} ⇒ "
          f"{'PASS' if fp_ok and all(sha_ok) and gate_res and all(v > 0 for v in gate_res.values()) else 'FAIL'}",
          flush=True)

    ck = os.path.join(LOG_DIR, f"73_ckpt_{prefix}_seed{a.seed}_{arm}{'_smoke' if a.smoke else ''}.pt")
    torch.save(model.state_dict(), ck)
    print(f"[META] prefix={prefix} arm={arm} seed={a.seed} steps={steps} batch={BATCH} lr={LR} "
          f"d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN} n_params={n_par} EM_BATCH={EM_BATCH} "
          f"no_history={a.no_history} smoke={a.smoke} n_eval={len(hits)} EM={em:.2f}% "
          f"SE={se:.2f}pp 实体命中={em_ent:.2f}% 条件命中={em_cond:.2f}% 配对CE={cm:.4f}±{cse:.4f} "
          f"地板max_naive={mx:.3f}% 门槛={thr:.3f}% pair_log={a.pair_log} train_fp={tr_fp} test_fp={te_fp} "
          f"训练墙钟={wall:.0f}s 评估墙钟={eval_sec:.0f}s 总墙钟={wall + eval_sec:.0f}s ckpt={ck}",
          flush=True)
    print("[DONE] exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
