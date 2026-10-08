#!/usr/bin/env python3
"""example_add3d —— 卡片框架的端到端可跑例子：合成算术 add_3d 上训一遍，打印 EM（batch=1）。

复用（**不许自己发明架构**）：
  · 卡内结构 = TransformerEncoderLayer(d=128, nhead=4, ff=512, dropout=0.1, gelu,
    norm_first=True) + 【外层残差 h = h + Think(h)】（S19/S22 已验证）；
  · 接通方式 = 输入卡 → 模因[n,d] → 目标卡 → 模因[n,d] → 输出卡（框架契约）；
  · 数据 / 训练超参 = `stages/19_residual_scan.py` 的 add_3d **res 臂**逐字相同
    （d=128, ff=512, NHEAD=4, lr=1e-3, batch=32, MAXLEN=512, MAXLEN 切分同源, seed=1234）；
  · 判据 = ★**饱和点（6000 步，E6/S36 口径）**：EM(batch=1, n=800) 与 E6 res 臂
    85.62%（seed1234）/ 74.38%（seed5678）差 ≤ 3pp；并**同时打印 @3000 / @6000 两栏**
    （数字每步正确率也两栏），日志里标「是否饱和」。

★步数：`CARDS_STEPS` 默认 **6000**（= E6 饱和点口径，与 R35 同口径）；3000 仍可用。
★种子：`CARDS_SEED` 默认 1234，可设 5678 对照 E6 的第二个 seed。

★已知坑（复刻 S14 数据必须连两个特性一起复刻，本文件都保留）：
  ① BUCKETS 顺序决定 RNG 种子（add_1d, add_2d, add_3d, sub_2d, mul_2d 逐字）；
  ② build_templates 的裸式 10 句有 bug（`{{expr}}` 不是 f-string ⇒ `{expr}` 不替换）。
★已知坑（R34）：`CARDS_RNG` 必须保留「dropout 独立流 + 重洗全局流」**两条流结构**
  （照抄 GPU 那段 CPU 代码 ⇒ −18.5pp，见 logits_train）；`coupled` 只用于复现那个坑。

本文件**不 import stages/**（独立）；默认跑 **CPU**（GPU 上有别的实验在跑，不抢）。
运行：`.venv/bin/python cards/example_add3d.py`
      （env：CARDS_STEPS / CARDS_SEED / CARDS_THREADS / CARDS_RNG / CARDS_SNAP_AT）
另：`CARDS_XCHECK=1` 会把 logs/ 里的 S19 GPU ckpt 灌进本管线复核（证明数据与评测口径逐位一致）。
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch                                            # noqa: E402
from torch.optim import AdamW                           # noqa: E402
from tokenizers import Tokenizer                        # noqa: E402

from cards import CardPipeline                          # noqa: E402

# ---------------------------------------------------------------- 配置（与 S19 逐字）
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512
MIN_T = 48
TAIL_KEEP = 24
STEPS = int(os.environ.get("CARDS_STEPS", "6000"))      # ★默认 6000 = E6 饱和点口径
SNAP_AT = int(os.environ.get("CARDS_SNAP_AT", "3000"))  # @3000 快照（两栏对照；≥STEPS 则不取）
LR = 1e-3
BATCH = 32
GEN_BATCH = 32
EM_BATCH = 1            # ★主口径：batch=1（R28）
N_TRAIN = 4000
N_TEST = 800
MAX_GEN = 48
SEED = int(os.environ.get("CARDS_SEED", "1234"))
BUCKET = "add_3d"
S19_RES_EM = 0.3725     # S19 add_3d res seed=1234 @3000（logs/19_results.jsonl）
S19_NORES_EM = 0.0250
# ★E6（stages/36_saturation_retest.py）res 臂饱和点锚：@6000 主锚 + @3000 同轨迹快照
E6_RES_EM = {1234: 0.85625, 5678: 0.74375}
E6_RES_DIG = {1234: 0.8427, 5678: 0.7633}
E6_RES_EM3000 = {1234: 0.3725, 5678: 0.3113}
E6_RES_DIG3000 = {1234: 0.5259, 5678: 0.4963}
TOL_PP = 3.0            # ★判据：与 E6 res 饱和点锚差 ≤3pp

# ★坑①：BUCKETS 的顺序决定每桶 RNG 种子（bi 进种子），必须逐字保持
BUCKETS = ("add_1d", "add_2d", "add_3d", "sub_2d", "mul_2d")
BI = BUCKETS.index(BUCKET)

TOK_PATH = "/home/vesita/coding/my/nanoSeek/data/chinese/char_tokenizer.json"
S19_CKPT = ROOT / "logs" / "19_ckpt_add_3d_res_seed1234.pt"

DEVICE = os.environ.get("CARDS_DEVICE", "cpu")          # ★默认 CPU：不抢 GPU 上的实验
# ★RNG 结构：s19-two-stream = 镜像 S19/GPU 的「dropout 独立流 + 重洗全局流」；
#            coupled = 照抄那段 CPU 代码（两条流耦合）⇒ 用来复现样本重洗分岔的坑。
RNG_MODE = os.environ.get("CARDS_RNG", "s19-two-stream")
DROP_GEN = torch.Generator()                            # dropout 专用流（train_run 里按 seed 重置）
torch.set_num_threads(int(os.environ.get("CARDS_THREADS", "4")))

tok = Tokenizer.from_file(TOK_PATH)
V = tok.get_vocab_size()
PAD_ID = tok.token_to_id("<pad>")
EOS_ID = tok.token_to_id("<eos>")


def enc(text: str) -> list:
    return tok.encode(text, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


# ============================================================================
# ① 数据：与 S19（⇒S14）逐字相同
# ============================================================================
def build_templates() -> list:
    ts = []
    tails_cn = ["等于多少？", "是多少？", "等于几？", "得多少？", "的结果是多少？",
                "的结果是几？", "的值是多少？", "等于多少", "是多少", "是几"]
    for lead in ("", "请", "帮我", "麻烦"):
        for verb in ("计算", "算出", "算一下", "口算", "快速算", "算"):
            for t in tails_cn:
                ts.append((f"{lead}{verb} {{expr}} {t}", True))
    for t in tails_cn:                                  # ★坑②：裸式 10 句（`{{expr}}`
        ts.append(("{{expr}} " + t, True))              #   非 f-string ⇒ 不替换）
    for q in ("What is {expr}?", "{expr} equals what?", "How much is {expr}?",
              "What is {expr} equal to?", "{expr} is what?", "What does {expr} make?",
              "What is the result of {expr}?", "{expr} equals how much?",
              "How many is {expr}?", "Tell me {expr}", "I need the answer to {expr}",
              "Do you know {expr}?"):
        ts.append((q, False))
    for lead in ("", "Please ", "Could you ", "Quick: "):
        for verb in ("Calculate", "Compute", "Work out", "Find", "Solve", "Figure out"):
            v = verb if lead == "" else verb.lower()
            for tail in ("", " for me", " please", "."):
                ts.append((f"{lead}{v} {{expr}}{tail}", False))
    assert len({t for t, _ in ts}) == len(ts), "模板有重复"
    return ts


TEMPLATES = build_templates()
CN_OPS = {"add_1d": ("加", "加上"), "add_2d": ("加", "加上"), "add_3d": ("加", "加上"),
          "sub_2d": ("减", "减去"), "mul_2d": ("乘", "乘以")}
EN_OPS = {"add_1d": ("plus",), "add_2d": ("plus",), "add_3d": ("plus",),
          "sub_2d": ("minus",), "mul_2d": ("times",)}
LOHI = {"add_1d": (0, 9), "add_2d": (10, 99), "add_3d": (100, 999)}

MUL_BY_Y = defaultdict(list)
for _a in range(10, 100):
    for _b in range(1, 10):
        MUL_BY_Y[_a * _b].append((_a, _b))
MUL_YS = sorted(MUL_BY_Y)


def answer_values(name: str) -> list:
    if name in LOHI:
        lo, hi = LOHI[name]
        return list(range(2 * lo, 2 * hi + 1))
    if name == "sub_2d":
        return list(range(0, 90))
    return list(MUL_YS)


def split_by_answer(name: str, y: int, rng):
    if name in LOHI:
        lo, hi = LOHI[name]
        a = rng.randint(max(lo, y - hi), min(hi, y - lo))
        return a, y - a
    if name == "sub_2d":
        a = rng.randint(max(10, y + 10), 99)
        return a, a - y
    fs = MUL_BY_Y[y]
    return fs[rng.randrange(len(fs))]


class Rng:
    """可复现的桶内随机源（与 S19 逐字相同；避免占用 torch 全局种子）。"""

    def __init__(self, seed: int):
        self._r = random.Random(seed)

    def randrange(self, n: int) -> int:
        return self._r.randrange(n)

    def randint(self, a: int, b: int) -> int:
        return self._r.randint(a, b)

    def random(self) -> float:
        return self._r.random()

    def shuffle(self, xs: list) -> None:
        self._r.shuffle(xs)


def render(name: str, a: int, b: int, y: int, rng: Rng):
    tpl, cjk = TEMPLATES[rng.randrange(len(TEMPLATES))]
    op = (CN_OPS[name] if cjk else EN_OPS[name])
    op = op[rng.randrange(len(op))]
    tight = cjk and rng.random() < 0.5
    expr = f"{a}{op}{b}" if tight else f"{a} {op} {b}"
    nl = tpl.format(expr=expr)
    return f"题干：{nl} → ", f"#### {y}", nl


def gen_half(name: str, n: int, rng: Rng, used: set, tag: str) -> list:
    ys = answer_values(name)
    order = ys[:]
    rng.shuffle(order)
    items, i, tries = [], 0, 0
    while len(items) < n:
        y = order[i % len(order)]
        i += 1
        tries += 1
        if tries > n * 500:
            raise RuntimeError(f"{name}{tag}: 文本空间不足 got={len(items)}/{n}")
        a, b = split_by_answer(name, y, rng)
        prompt, target, nl = render(name, a, b, y, rng)
        if prompt + target in used:
            continue
        used.add(prompt + target)
        items.append(dict(prompt=prompt, target=target, a=a, b=b, y=y, nl=nl))
    return items


def encode_record(it: dict) -> dict:
    p_full, t_full = enc(it["prompt"]), enc(it["target"] + "<eos>")
    cut_p = cut_t = False
    if len(p_full) + len(t_full) > MAXLEN:
        p_room = MAXLEN - MIN_T
        if len(p_full) > p_room:
            p_ids = p_full[: p_room - TAIL_KEEP] + p_full[-TAIL_KEEP:]
            cut_p = True
        else:
            p_ids = p_full
        room = MAXLEN - len(p_ids)
        if len(t_full) <= room:
            t_ids = t_full
        else:
            t_ids = t_full[-room:]
            cut_t = True
    else:
        p_ids, t_ids = p_full, t_full
    assert len(p_ids) + len(t_ids) <= MAXLEN
    return dict(p=p_ids, t=t_ids, t_full=t_full, gold=it["target"][5:].strip(),
                text=it["target"], cut_p=cut_p, cut_t=cut_t, a=it["a"], b=it["b"],
                y=it["y"], nl=it["nl"])


def build_data():
    tr_rng, te_rng = Rng(14000 + BI * 7), Rng(14900 + BI * 7)
    used: set = set()
    raw_tr = gen_half(BUCKET, N_TRAIN, tr_rng, used, "train")
    raw_te = gen_half(BUCKET, N_TEST, te_rng, used, "test")
    train = [encode_record(x) for x in raw_tr]
    test = [encode_record(x) for x in raw_te]
    assert len(used) == len(raw_tr) + len(raw_te), "train/test 文本池计数异常（应零重叠）"
    return train, test


# ============================================================================
# ② 训练 / 评测（与 S19 逐字相同，只是模型换成 CardPipeline）
# ============================================================================
def build_batch(recs):
    n = max(len(r["p"]) + len(r["t"]) for r in recs)
    ids = torch.full((len(recs), n), PAD_ID, dtype=torch.long)
    s = torch.zeros(len(recs), dtype=torch.long)
    for i, r in enumerate(recs):
        ids[i, : len(r["p"])] = torch.tensor(r["p"])
        ids[i, len(r["p"]): len(r["p"]) + len(r["t"])] = torch.tensor(r["t"])
        s[i] = len(r["p"])
    return ids.to(DEVICE), s


def logits_train(pipe: CardPipeline, ids: torch.Tensor) -> torch.Tensor:
    """★训练前向：镜像 S19 在 GPU 上的 RNG 结构（两条流）。

    S19（GPU）：dropout 抽 CUDA 流（`torch.cuda.manual_seed_all(seed)`，独立），
                每个 epoch 的 `torch.randperm` 重洗抽 CPU 全局流（此时没被 dropout 动过）。
    CPU 上若照抄那段代码，dropout 与重洗会共用**同一条** CPU 全局流 ⇒ 从 step 125（第一个
    epoch 结束）起样本重洗顺序就跑偏、与 S19 分岔（实测 EM 37.25% → 18.75%，R34）。
    所以这里把 dropout 的 RNG 流独立出来（`CARDS_RNG=coupled` 可关掉，用来复现那个坑）。
    """
    if RNG_MODE == "coupled":
        return pipe.logits(ids)
    saved = torch.random.get_rng_state()
    torch.random.set_rng_state(DROP_GEN.get_state())
    try:
        return pipe.logits(ids)
    finally:
        DROP_GEN.set_state(torch.random.get_rng_state())
        torch.random.set_rng_state(saved)


def masked_ce(pipe: CardPipeline, recs) -> torch.Tensor:
    ids, s = build_batch(recs)
    logp = torch.log_softmax(logits_train(pipe, ids), dim=-1)
    nll = torch.zeros((), device=DEVICE)
    ntok = 0
    for i, r in enumerate(recs):
        e = s[i] + len(r["t"])
        lp = logp[i, s[i] - 1: e - 1]
        tgt = ids[i, s[i]: e]
        nll = nll - lp.gather(1, tgt.unsqueeze(1)).sum()
        ntok += len(r["t"])
    return nll / ntok


def train_run(train, steps: int, seed: int = SEED, snap_at: int | None = None):
    """训 steps 步；snap_at 落在 (0, steps) 内时额外取一份该步的 state_dict 快照（@3000 用）。

    ★R34：两条随机流的结构不许动 —— torch.manual_seed(seed)（重洗/初始化流）
    与 DROP_GEN.manual_seed(seed)（dropout 流）各自独立。
    """
    torch.manual_seed(seed)
    DROP_GEN.manual_seed(seed)                          # ★镜像 torch.cuda.manual_seed_all(seed)
    pipe = CardPipeline(d=D, vocab_size=V, ff=FF, nhead=NHEAD, maxlen=MAXLEN,
                        pad_id=PAD_ID, dropout=0.1, residual=True).to(DEVICE)
    opt = AdamW(pipe.parameters(), lr=LR)
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(seed))
    step, pos, t0, last = 0, 0, time.time(), 0.0
    snap = None
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train))
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        loss = masked_ce(pipe, [train[i] for i in idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        step += 1
        if snap_at is not None and step == snap_at:
            snap = {k: v.detach().to("cpu").clone() for k, v in pipe.state_dict().items()}
        if step % 500 == 0 or step == steps:
            el = time.time() - t0
            print(f"  [train] step={step}/{steps} loss={loss.item():.4f} "
                  f"elapsed={el:.0f}s s/step={(el - last) / (step % 500 or 500):.3f}", flush=True)
            last = el
    return pipe, time.time() - t0, snap


@torch.no_grad()
def greedy_gen(pipe: CardPipeline, recs, batch: int = 1) -> list:
    """S19 的 R28 贪心生成（主口径 batch=1）逐字相同。"""
    pipe.eval()
    texts = [""] * len(recs)
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
        for _ in range(max(cap)):
            if all(done):
                break
            logits = pipe.logits(ids)
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
            ids = torch.cat([ids, torch.tensor(nxt, dtype=torch.long,
                                                device=DEVICE).unsqueeze(1)], 1)
        for j in range(len(chunk)):
            g = [x for x in ids[j, n0: n0 + gen_len[j]].tolist() if x != EOS_ID]
            texts[i + j] = dec(g)
    pipe.train()
    return texts


def parse_ans(text: str):
    if "####" not in text:
        return None
    a = text.rsplit("####", 1)[1].split("\n")[0]
    a = a.strip().replace(",", "").replace(" ", "").replace("$", "")
    return a or None


def em_score(pipe: CardPipeline, test) -> dict:
    texts = greedy_gen(pipe, test, batch=EM_BATCH)
    hits = [int(parse_ans(texts[i]) == r["gold"]) for i, r in enumerate(test)]
    n = len(hits)
    em = sum(hits) / n
    se = math.sqrt(em * (1 - em) / n)
    sample = list(zip(texts[:3], [r["gold"] for r in test[:3]]))
    return dict(em=em, se=se, n=n, hits=hits, sample=sample)


# ---- 数字每步正确率（口径逐字复用 stages/19_residual_scan.py:492-573 的 ①数字与算子）----
SPECIAL_TOKENS = {"<eos>", "<unk>", "<pad>", "<bos>", "<cont>", "<sep>", "<resp>",
                  "<call>", "<result>", "<answer>", "<tool>", "<search>", "<topic>"}
NUM_CHARS = set("0123456789+-*/=%$")
TMPL_CHARS = set("【】详细解题思路推理与") | {"<", ">", "#"}


def classify(tid: int) -> int:
    """token 分类：0=数字与算子（=数字每步正确率的分母）1=模板/格式 2=中文 3=其它。"""
    s = tok.id_to_token(int(tid)) or ""
    if s in SPECIAL_TOKENS or (s.startswith("<") and s.endswith(">")):
        return 1
    if s and all(c in NUM_CHARS for c in s):
        return 0
    if any(c in TMPL_CHARS for c in s):
        return 1
    if any("一" <= c <= "鿿" for c in s):
        return 2
    return 3


def digit_report(pipe: CardPipeline, recs, batch: int = GEN_BATCH) -> dict:
    """数字每步正确率 = exp(−数字 CE 的样本级均值)；不落梯度、不改 eval/train 之外的任何状态。"""
    was_training = pipe.training
    pipe.eval()
    per: list = []
    with torch.no_grad():
        for i in range(0, len(recs), batch):
            chunk = recs[i: i + batch]
            ids, s = build_batch(chunk)
            logp = torch.log_softmax(pipe.logits(ids), dim=-1)
            for j, r in enumerate(chunk):
                e = s[j] + len(r["t"])
                lp = logp[j, s[j] - 1: e - 1]
                tgt = ids[j, s[j]: e]
                ces = (-lp.gather(1, tgt.unsqueeze(1))).squeeze(1).tolist()
                ds = [v for p, v in enumerate(ces) if classify(int(tgt[p])) == 0]
                if ds:
                    per.append(sum(ds) / len(ds))
    if was_training:
        pipe.train()
    n = len(per)
    m = sum(per) / n
    var = sum((x - m) ** 2 for x in per) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    acc = math.exp(-m)
    return dict(acc=acc, acc_se=acc * se, ce=m, ce_se=se, n=n)


# ============================================================================
# ③ 交叉核对：把 logs/ 里的 S19 GPU ckpt 灌进本管线（证明数据/评测口径逐位一致）
# ============================================================================
def load_into_pipeline(sd: dict, ckpt_prefixes=None) -> CardPipeline:
    """把 S19/S36 容器（emb./in_enc./thought./head.）的 state_dict 映射进 CardPipeline。"""
    pref = ckpt_prefixes or {"emb.": "input.emb.", "in_enc.": "input.enc.",
                             "thought.": "target.think.", "head.": "output.head."}
    mapped = {}
    for k, v in sd.items():
        for src, dst in pref.items():
            if k.startswith(src):
                mapped[dst + k[len(src):]] = v
                break
    pipe = CardPipeline(d=D, vocab_size=V, ff=FF, nhead=NHEAD, maxlen=MAXLEN,
                        pad_id=PAD_ID, dropout=0.1, residual=True)
    _missing, unexpected = pipe.load_state_dict(mapped, strict=False)
    assert not unexpected, f"ckpt 有本管线不认的键：{unexpected[:3]}"
    return pipe


def xcheck_s19(test) -> None:
    if not S19_CKPT.exists():
        print(f"[XCHECK] 跳过：{S19_CKPT} 不存在", flush=True)
        return
    blob = torch.load(S19_CKPT, map_location="cpu", weights_only=False)
    sd = blob if isinstance(blob, dict) and "emb.weight" in blob else blob.get("state", blob)
    pipe = load_into_pipeline(sd)
    print(f"[XCHECK] S19 ckpt 灌入本管线：映射 {len(pipe.state_dict())} 张量", flush=True)
    r = em_score(pipe, test)
    print(f"[XCHECK] ★同一份数据 + 同一个贪心评测，跑 S19 的 GPU 权重："
          f"EM={r['em']*100:.2f}%±{r['se']*100:.2f}(n={r['n']}) vs S19 记录 37.25% | "
          f"Δ={(r['em']-S19_RES_EM)*100:+.2f}pp", flush=True)


# ============================================================================
# ④ 主流程
# ============================================================================
def main() -> int:
    t0 = time.time()
    print(f"[CFG] device={DEVICE}（torch.cuda.is_available={torch.cuda.is_available()}，"
          f"★不用 GPU）threads={torch.get_num_threads()} | d={D} ff={FF} "
          f"NHEAD={NHEAD} lr={LR} batch={BATCH} steps={STEPS} MAXLEN={MAXLEN} "
          f"train={N_TRAIN}/test={N_TEST} seed={SEED} 桶={BUCKET}(BI={BI}) "
          f"RNG={RNG_MODE} snap@{SNAP_AT}", flush=True)
    naked = sum(1 for t, _ in TEMPLATES if "{expr}" in t.format(expr="X"))
    print(f"[CFG] V={V} PAD={PAD_ID} EOS={EOS_ID} | 模板={len(TEMPLATES)} 句（★裸式 "
          f"{naked} 句 `{{expr}}` 不替换 = S19 的已知 bug，原样保留）| "
          f"E6 饱和点锚(res@{STEPS}/seed)：EM={E6_RES_EM.get(SEED, float('nan'))*100:.2f}% "
          f"数字每步={E6_RES_DIG.get(SEED, float('nan'))*100:.2f}%", flush=True)

    train, test = build_data()
    print(f"[DATA] {BUCKET}: BI={BI} ⇒ tr_rng=seed{14000 + BI * 7} te_rng=seed{14900 + BI * 7}"
          f"（★坑①：BUCKETS 顺序决定种子）", flush=True)
    print(f"[DATA] {BUCKET}: train={len(train)} test={len(test)} | "
          f"gold 前5答案={[r['gold'] for r in test[:5]]} | "
          f"prompt[-1]示例={dec(test[0]['p'])[:60]!r}", flush=True)

    snap_at = SNAP_AT if 0 < SNAP_AT < STEPS else None
    pipe, wall, snap3 = train_run(train, STEPS, snap_at=snap_at)
    print(f"[TRAIN] 完成 {STEPS} 步，墙钟={wall/60:.2f}min "
          f"（{wall/STEPS*1000:.0f}ms/step）| 参数量={pipe.n_params()/1e6:.3f}M "
          f"| @{SNAP_AT} 快照={'已取' if snap3 else '未取'}", flush=True)

    # ---- @6000（主口径）：EM + 数字每步 ----
    r6 = em_score(pipe, test)
    d6 = digit_report(pipe, test)
    final = {k: v.detach().to("cpu").clone() for k, v in pipe.state_dict().items()}

    # ---- @3000（同一条轨迹的快照，★不是 dev 选步）：EM + 数字每步 ----
    r3 = d3 = None
    if snap3 is not None:
        pipe.load_state_dict({k: v.to(DEVICE) for k, v in snap3.items()})
        r3 = em_score(pipe, test)
        d3 = digit_report(pipe, test)
        pipe.load_state_dict({k: v.to(DEVICE) for k, v in final.items()})   # 还原末点权重

    em6, dig6 = E6_RES_EM.get(SEED), E6_RES_DIG.get(SEED)
    print(f"[EM] ★{BUCKET} test 严格EM(batch=1, n={r6['n']}) @{STEPS}="
          f"{r6['em']*100:.2f}%±{r6['se']*100:.2f} | 数字每步正确率="
          f"{d6['acc']*100:.2f}%±{d6['acc_se']*100:.2f}（n={d6['n']}，CE={d6['ce']:.4f}）", flush=True)
    if r3 is not None:
        print(f"[EM] @{SNAP_AT}（同轨迹快照）EM={r3['em']*100:.2f}%±{r3['se']*100:.2f} | "
              f"数字每步={d3['acc']*100:.2f}%±{d3['acc_se']*100:.2f}（CE={d3['ce']:.4f}）", flush=True)
    else:
        print(f"[EM] @{SNAP_AT} 两栏对照跳过（CARDS_STEPS={STEPS} ≤ 快照点 {SNAP_AT}）", flush=True)

    # ---- ★两栏表 + 饱和判定（R35 口径）----
    if r3 is not None:
        de = (r6["em"] - r3["em"]) * 100
        sat = "已平（=饱和）" if de <= 1.0 else ("仍在升" if de <= 10.0 else "仍在大幅上升（未饱和）")
        print(f"[TABLE] {BUCKET} seed={SEED} | 栏@{SNAP_AT}: EM={r3['em']*100:.2f}% "
              f"数字每步={d3['acc']*100:.2f}% | 栏@{STEPS}: EM={r6['em']*100:.2f}% "
              f"数字每步={d6['acc']*100:.2f}% | ΔEM={de:+.2f}pp Δ数字="
              f"{(d6['acc']-d3['acc'])*100:+.2f}pp ⇒ 是否饱和：{sat}（batch=1, n={r6['n']}）", flush=True)

    # ---- ★V1 判据：与 E6 res 臂饱和点锚比（≤3pp）----
    if em6 is not None:
        dev = (r6["em"] - em6) * 100
        print(f"[JUDGE] vs E6/S36 res 臂 @6000 seed={SEED}：EM {em6*100:.2f}% → 本机 "
              f"{r6['em']*100:.2f}% ⇒ Δ={dev:+.2f}pp | 数字每步 {dig6*100:.2f}% → "
              f"{d6['acc']*100:.2f}% ⇒ Δ={(d6['acc']-dig6)*100:+.2f}pp | 判据 |ΔEM| ≤ {TOL_PP}pp "
              f"⇒ {'一致 ✓' if abs(dev) <= TOL_PP else '★偏离 >%gpp（要定位：CPU/GPU 浮点 / RNG 流结构 / 数据 RNG 顺序）' % TOL_PP}",
              flush=True)
    else:
        print(f"[JUDGE] seed={SEED} 不在 E6 锚表 {sorted(E6_RES_EM)} 里 ⇒ 只报观测值", flush=True)
    e30 = E6_RES_EM3000.get(SEED)
    if r3 is not None and e30 is not None:
        print(f"[JUDGE] @{SNAP_AT} 对照 S19/S36 res 臂同轨迹快照 {e30*100:.2f}%：本机 "
              f"{r3['em']*100:.2f}% ⇒ Δ={(r3['em']-e30)*100:+.2f}pp（数字每步 "
              f"{E6_RES_DIG3000[SEED]*100:.2f}% → {d3['acc']*100:.2f}%）", flush=True)
    print(f"[JUDGE] 对照：nores 臂 S19 @3000 是 {S19_NORES_EM*100:.2f}%（残差是唯一变量，"
          f"本例子用 residual=True）；★E1 结论：卡边界不产生能力，本例子只证明「能跑通 + 口径一致」",
          flush=True)
    print("[SAMPLES] " + " | ".join(f"pred={parse_ans(t)!r} gold={g!r}"
                                    for t, g in r6["sample"]), flush=True)

    # ---- 框架能力现场演示（不影响上面的 EM）----
    print(f"[黑板] 谁读了谁 = {pipe.who_reads_whom()}", flush=True)
    print(f"[黑板] 最近一次路由 = {pipe.blackboard.read_chain()}", flush=True)
    print(f"[MV] {pipe.mv().short()} | sha256={pipe.mv().sha256()[:16]} | "
          f"接口快照={pipe.interface_sha256()[:16]} | W 版本={pipe.mv().w_version}", flush=True)

    before = pipe.theta()
    print(f"[冻结] freeze(['input','output']) → {pipe.freeze(['input', 'output'])}；"
          f"可训参数={sum(p.numel() for p in pipe.trainable_parameters())}", flush=True)
    opt = AdamW(pipe.trainable_parameters(), lr=LR)
    pipe.train()
    for _ in range(5):
        loss = masked_ce(pipe, train[:BATCH])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    delta = pipe.assert_frozen_unchanged(before)
    print(f"[冻结] 再训 5 步后 max|Δθ|：input={delta['input']!r} output={delta['output']!r}"
          f"（★精确 0.0，断言已过）| target={delta['target']:.3e}（在动）", flush=True)

    tmp = Path(tempfile.mkdtemp(prefix="cards_mv_"))
    s1, s2 = pipe.snapshot(tmp / "a.json"), pipe.snapshot(tmp / "b.json")
    print(f"[快照] {s1.path} | sha256={s1.sha256[:16]} 只读={'是' if (os.stat(s1.path).st_mode & 0o222) == 0 else '否'} | "
          f"同权重两次快照 sha 相同={s1.sha256 == s2.sha256}", flush=True)
    print(f"[MV] 快照载荷 MV={json.dumps(s1.payload['mv'], ensure_ascii=False)}", flush=True)

    if os.environ.get("CARDS_XCHECK") == "1":
        xcheck_s19(test)

    print(f"[DONE] 总墙钟={(time.time()-t0)/60:.2f}min exit=0", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
