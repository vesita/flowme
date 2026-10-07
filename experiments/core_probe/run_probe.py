#!/usr/bin/env python3
"""P1 核功能可分离性探针：**只读前向 + 线性 probe**（P0–P5）。

判据全部写死在 PREREG.md（mtime 早于本脚本任何一次运行）。本脚本：
  1. 抽表征：只读 `checkpoints/base_encoder.pt`，hook 捕 `emb / b0 / b1 / b2 / pool`
     五层输出，masked mean 池化（`idr` = 提及 span 内 mean）；核 eval + no_grad。
  2. P0 双向对照（随机标签 / 构造可分）→ 门禁。
  3. P1/P3 分层单任务线性 probe（核冻结，只训 probe），2 seed 配对。
  4. P2 可分离性：k=8 瓶颈联合 probe、方向余弦分布、交叉干扰。
  5. P5 逐任务 max_naive 并列；判定三选一（PREREG §6）。

不训核、不训卡、不写只读目录；产物只落 `experiments/core_probe/results/`。
"""
from __future__ import annotations

import argparse
import itertools
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402

from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402
from dtseek.tasks.builtin.person import SPEC as PERSON_SPEC  # noqa: E402
from dtseek.tasks.builtin.sentiment import SPEC as SENT_SPEC  # noqa: E402
from dtseek.tasks.builtin.sentiment.dataset import extract_emotion_spans  # noqa: E402

# ── 口径常量（PREREG §1/§2 写死）──────────────────────────────────────────
LAYERS = ("emb", "b0", "b1", "b2", "pool")
SEEDS = (42, 43)
LOGIC_WORDS = ("因为", "所以", "但是", "但", "因此", "于是", "虽然", "尽管", "而且",
               "并且", "如果", "要是", "只要", "除非", "总之", "可见", "由此", "结果")
MAX_LEN = {"emo": SENT_SPEC.max_len, "skel": 64, "logic": 64, "idr": PERSON_SPEC.max_len}
BATCH = 256
STEPS = 1500          # 主 probe 拟合步数（--steps 可覆盖）
DIR_STEPS = 800       # 方向（余弦用）拟合步数
LR = 0.2
K_BOTTLE = 8
N_INIT = 4            # 每 seed 的方向初始化数 ⇒ 每任务每层 8 条方向
MARKER_N = 16         # P0② 标记前缀长度
P0_MARKER_ACC = 0.99
SYNTH_N = 4000
SYNTH_N2 = 20000      # 功效修正（AMENDMENTS.md）：p=128 下 n=4000 的 0.99 门槛不可达
P0_MARKER_SPLIT = 3000

EMO_CACHE = ROOT / "experiments" / "core_keep" / "cache" / "sentiment_32000.pkl"
PERSON_CACHE = ROOT / "experiments" / "core_keep" / "cache" / "person_6000.pkl"
SKEL_TRAIN = ROOT / "experiments" / "two_channel_head" / "data" / "train.jsonl"
SKEL_TEST = ROOT / "experiments" / "two_channel_head" / "data" / "test.jsonl"
SKEL_STATS = ROOT / "experiments" / "two_channel_head" / "data" / "stats.json"
BASE = ROOT / "checkpoints" / "base_encoder.pt"

TASKS = ("emo", "skel", "logic", "idr")
PRIM_PAIRS = (("emo", "skel"), ("emo", "idr"), ("skel", "idr"))     # 三类信息的主对
AUX_PAIRS = (("emo", "logic"), ("skel", "logic"), ("logic", "idr"))


class NPEncoder(json.JSONEncoder):
    """numpy 标量/数组 → Python 原生（否则 json.dumps 直接炸）。"""

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def dump(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, cls=NPEncoder)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ══════════════════════════════════════════════════════════════════════════
# 1. 数据与标签（全部现成资产）
# ══════════════════════════════════════════════════════════════════════════
def build_sets() -> dict:
    out: dict = {}

    # —— 情绪：cache == build_sentiment_dataset(32000)（md5 已对账）——
    rows = pickle.load(open(EMO_CACHE, "rb"))
    per: dict = {c: [] for c in range(4)}
    for r in rows:
        per[r["label"]].append(r)
    rng = random.Random(42)
    tr, te = [], []
    for c in range(4):
        idx = list(range(len(per[c])))
        rng.shuffle(idx)
        sel = [per[c][i] for i in idx[:2000]]
        tr += sel[:1000]
        te += sel[1000:]
    random.Random(43).shuffle(tr)          # 打乱类别分块（分层比例不变）
    random.Random(43).shuffle(te)
    out["emo"] = {
        "kind": "text",
        "tr_text": [r["text"] for r in tr], "tr_y": [r["label"] for r in tr],
        "te_text": [r["text"] for r in te], "te_y": [r["label"] for r in te],
        "classes": 4,
        "source": "experiments/core_keep/cache/sentiment_32000.pkl（每类抽 2000 → 1000/1000，rng42）",
    }

    # —— 结构-逻辑：骨架 id（官方 train/test 切分）+ 逻辑词（同一批句子）——
    sk_tr = [json.loads(l) for l in open(SKEL_TRAIN)]
    sk_te = [json.loads(l) for l in open(SKEL_TEST)]
    random.Random(43).shuffle(sk_tr)         # 只打乱行序，split/标签组成不变
    random.Random(43).shuffle(sk_te)
    sent_tr = [r["sent"] for r in sk_tr]
    sent_te = [r["sent"] for r in sk_te]
    out["skel"] = {
        "kind": "text",
        "tr_text": sent_tr, "tr_y": [r["skel_id"] for r in sk_tr],
        "te_text": sent_te, "te_y": [r["skel_id"] for r in sk_te],
        "classes": max(max(r["skel_id"] for r in sk_tr), max(r["skel_id"] for r in sk_te)) + 1,
        "source": "experiments/two_channel_head/data/{train,test}.jsonl::skel_id（官方 split）",
    }
    out["logic"] = {
        "kind": "text",
        "tr_text": sent_tr, "tr_y": [int(any(w in s for w in LOGIC_WORDS)) for s in sent_tr],
        "te_text": sent_te, "te_y": [int(any(w in s for w in LOGIC_WORDS)) for s in sent_te],
        "classes": 2,
        "source": "同 skel 句子；标签 = 逻辑词表匹配（PREREG §2 词表）",
    }

    # —— 身份-回忆：提及级 is_repeat（= 线上 repeat_mention_acc 分母口径），按文本分组切分 ——
    prows = [r for r in pickle.load(open(PERSON_CACHE, "rb")) if r["spans"]]
    rng = random.Random(42)
    idx = list(range(len(prows)))
    rng.shuffle(idx)
    idr: dict = {"kind": "mention", "classes": 2,
                 "source": "experiments/core_keep/cache/person_6000.pkl（提及级 is_repeat，按文本分组 2100/2100，rng42）"}
    for split, sel in (("tr", idx[:2100]), ("te", idx[2100:])):
        t_list, s_list, y_list = [], [], []
        for i in sel:
            r = prows[i]
            seen: set = set()
            for s in r["spans"]:
                y_list.append(1 if s["label"] in seen else 0)
                seen.add(s["label"])
                t_list.append(r["text"])
                s_list.append((s["start"], s["end"]))
        idr[f"{split}_text"] = t_list
        idr[f"{split}_span"] = s_list
        idr[f"{split}_y"] = y_list
    out["idr"] = idr
    return out


def naive_stats(sets: dict) -> dict:
    """P5：逐任务 max_naive@majority（train 多数类在 test 上）与 max_naive@rule（实测）。"""
    res = {}
    for t, d in sets.items():
        ytr, yte = np.array(d["tr_y"]), np.array(d["te_y"])
        classes = np.unique(np.concatenate([ytr, yte]))
        cnt = {int(c): int((ytr == c).sum()) for c in classes}
        maj = max(cnt, key=lambda c: cnt[c])
        res[t] = {
            "n_train": int(len(ytr)), "n_test": int(len(yte)),
            "n_classes": int(d["classes"]),
            "train_dist": {str(k): v for k, v in sorted(cnt.items())},
            "test_dist": {str(int(c)): int((yte == c).sum()) for c in classes},
            "majority_class": int(maj),
            "max_naive@majority": float((yte == maj).mean()),
            "source": d["source"],
        }
    # —— 免费规则基线（逐类实测）——
    emo = pickle.load(open(EMO_CACHE, "rb"))
    ok = abst = 0
    for r in emo:
        dom, _ = extract_emotion_spans(r["text"])
        pred = dom if dom in (0, 1, 2, 3) else 0
        abst += int(dom == -1)
        ok += int(pred == r["label"])
    res["emo"]["max_naive@rule"] = ok / len(emo)
    res["emo"]["rule_desc"] = f"extract_emotion_spans 词表规则（混杂/否定弃权判 0，弃权 n={abst}）"
    res["emo"]["rule_n"] = len(emo)

    st = json.load(open(SKEL_STATS))
    res["skel"]["max_naive@rule"] = float(st["max_naive_train"])
    res["skel"]["rule_desc"] = (f"two_channel_head 自带免费规则套件 max_naive="
                                f"{st['max_naive_train']}（其 stats.json::max_naive_train 实测）")
    res["logic"]["max_naive@rule"] = 1.0
    res["logic"]["rule_desc"] = "标签定义 = 词表匹配 ⇒ 免费规则按定义 100%（probe 不构成增量）"

    tot = good = 0
    for r in pickle.load(open(PERSON_CACHE, "rb")):
        seen_lab: set = set()
        seen_word: set = set()
        for s in r["spans"]:
            gold = 1 if s["label"] in seen_lab else 0
            pred = 1 if s["word"] in seen_word else 0
            tot += 1
            good += int(gold == pred)
            seen_lab.add(s["label"])
            seen_word.add(s["word"])
    res["idr"]["max_naive@rule"] = good / tot
    res["idr"]["rule_desc"] = f"字面重复规则（表面词在前文出现过 ⇒ repeat），n={tot}"
    res["idr"]["rule_n"] = tot
    return res


# ══════════════════════════════════════════════════════════════════════════
# 2. 只读前向抽表征（hook 五层）
# ══════════════════════════════════════════════════════════════════════════
def extract(enc, tok, texts, max_len, device, spans=None, batch=BATCH):
    """文本去重前向（同文本只过一次核）；返回 (X_text [n,5,D], X_span 或 None, meta)。

    - X_text：每行的 masked mean 池化（重复文本共享同一向量）；
    - spans=[(start,end)] 与 texts 逐行对应时另给提及 span 内的各层均值
      （char↔token 字符级 1:1；span 超出有效长度 ⇒ 该行全 NaN，由调用方丢弃）；
    - 构造性对账：hook 捕获的 norm 输出必须与同一次前向返回值逐位相同。
    """
    n = len(texts)
    D = enc.embedding.embedding_dim
    order: list = []
    uid: dict = {}
    inv = np.empty(n, dtype=np.int64)
    for i, t in enumerate(texts):
        if t not in uid:
            uid[t] = len(order)
            order.append(t)
        inv[i] = uid[t]
    nu = len(order)

    ids, masks, over = [], [], 0
    for t in order:
        if len(t) > max_len:
            over += 1
        e = tok.encode(t, max_length=max_len, padding=True)
        ids.append(e["input_ids"])
        masks.append(e["attention_mask"])
    ids_t = torch.tensor(np.array(ids), dtype=torch.long, device=device)
    mask_t = torch.tensor(np.array(masks), dtype=torch.bool, device=device)

    Xu = np.zeros((nu, len(LAYERS), D), dtype=np.float32)
    Xs = np.zeros((n, len(LAYERS), D), dtype=np.float32) if spans is not None else None
    mentions: dict = {}
    if spans is not None:
        for i, (a, b) in enumerate(spans):
            mentions.setdefault(int(inv[i]), []).append((i, a, b))

    cap: dict = {}
    handles = [mod.register_forward_hook(
        lambda m, i, o, nm=name: cap.__setitem__(nm, o))
        for name, mod in (("emb", enc.embedding), ("b0", enc.blocks[0]),
                          ("b1", enc.blocks[1]), ("b2", enc.blocks[2]), ("pool", enc.norm))]
    dropped = 0
    with torch.no_grad():
        for i0 in range(0, nu, batch):
            sl = slice(i0, min(i0 + batch, nu))
            inp, m = ids_t[sl], mask_t[sl]
            doc = enc(inp, attention_mask=m)
            if i0 == 0:
                assert torch.equal(cap["pool"], doc), "hook 捕获的 norm 输出与前向返回值不一致"
            m3 = m.unsqueeze(-1).float()
            for li, name in enumerate(LAYERS):
                Xu[sl, li, :] = ((cap[name] * m3).sum(1) / m3.sum(1)).cpu().numpy()
            if spans is not None:
                valid = m.sum(1).cpu().numpy()
                for ui in range(i0, i0 + int(m.shape[0])):
                    for row, a, b in mentions.get(ui, ()):
                        s = max(0, int(a))
                        e2 = min(int(b), int(valid[ui - i0]))
                        if e2 - s < 1:
                            dropped += 1
                            Xs[row, :, :] = np.nan
                            continue
                        for li, name in enumerate(LAYERS):
                            Xs[row, li, :] = cap[name][ui - i0, s:e2].mean(0).cpu().numpy()
    for hh in handles:
        hh.remove()

    meta = {"n_rows": n, "n_unique": nu, "truncated": over,
            "truncated_frac": round(over / max(1, nu), 4),
            "mention_dropped": dropped}
    return Xu[inv], Xs, meta


def extract_texts(enc, tok, texts, max_len, device, batch=BATCH):
    """文本级池化：返回 (X [n,5,D], meta)。"""
    X, _, meta = extract(enc, tok, texts, max_len, device, spans=None, batch=batch)
    return X, meta


# ══════════════════════════════════════════════════════════════════════════
# 3. 线性 probe
# ══════════════════════════════════════════════════════════════════════════
def standardize(A, B, stats=None):
    """按 train 统计标准化；B=None 表示只变换 A（eval 侧复用 train 统计）。"""
    if stats is None:
        mu = A.mean(0)
        sd = A.std(0)
        stats = (mu, np.where(sd < 1e-6, 1.0, sd))
    mu, sd = stats
    A2 = (A - mu) / sd
    B2 = None if B is None else (B - mu) / sd
    return A2, B2, stats


def _fit(A2, ytr, ncls, seed, steps, lr, device):
    At = torch.tensor(A2, device=device)
    y1 = torch.tensor(np.asarray(ytr), dtype=torch.long, device=device)
    torch.manual_seed(seed)
    lin = nn.Linear(A2.shape[1], ncls).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin(At), y1).backward()
        opt.step()
    return lin


def fit_probe(A, ytr, B, yte, ncls, seed, steps=STEPS, lr=LR, device="cpu", stats=None):
    """线性（softmax 回归）；返回 acc、逐样本预测、权重 [C,D]（标准化空间）。"""
    A2, B2, _ = standardize(np.asarray(A, np.float32), np.asarray(B, np.float32), stats)
    lin = _fit(A2, ytr, ncls, seed, steps, lr, device)
    with torch.no_grad():
        pred = lin(torch.tensor(B2, device=device)).argmax(1).cpu().numpy()
    acc = float((pred == np.asarray(yte)).mean())
    return acc, pred, lin.weight.detach().cpu().numpy()


class Bottleneck(nn.Module):
    """联合 probe：共享投影 P（k=None 表示零共享）+ 各任务独立线性头。"""

    def __init__(self, d: int, k: int | None, dims: list[int], device):
        super().__init__()
        self.P = nn.Linear(d, k).to(device) if k else None
        self.heads = nn.ModuleList([nn.Linear(k or d, c).to(device) for c in dims])

    def forward(self, x, t):
        h = self.P(x) if self.P is not None else x
        return self.heads[t](h)


def fit_joint(datas, k, seed, device, steps=STEPS, lr=LR, shared=True):
    """datas = [(A_tr, y_tr), ...]；loss = 各任务 CE 均值的等权平均。
    返回 (model, stats_list)：逐任务用**自己的 train 统计**标准化（任务头独立）。"""
    dims = [int(np.asarray(y).max()) + 1 for _, y in datas]   # 头维度必须 > 最大标签 id
    torch.manual_seed(seed)
    model = Bottleneck(datas[0][0].shape[1], k if shared else None, dims, device)
    opt = torch.optim.Adam([p for p in model.parameters()], lr=lr)
    ten, stats_list = [], []
    for A, y in datas:
        A2, _, st = standardize(np.asarray(A, np.float32), np.asarray(A, np.float32))
        stats_list.append(st)
        ten.append((torch.tensor(A2, device=device),
                    torch.tensor(np.asarray(y), dtype=torch.long, device=device)))
    for _ in range(steps):
        opt.zero_grad()
        loss = sum(F.cross_entropy(model(x, i), yy) for i, (x, yy) in enumerate(ten)) / len(ten)
        loss.backward()
        opt.step()
    return model, stats_list


def eval_joint(model, datas_te, stats_list, device):
    """→ [(acc, preds), ...]"""
    res = []
    for i, (A, y) in enumerate(datas_te):
        A2, _, _ = standardize(np.asarray(A, np.float32), None, stats_list[i])
        with torch.no_grad():
            pred = model(torch.tensor(A2, device=device), i).argmax(1).cpu().numpy()
        res.append((float((pred == np.asarray(y)).mean()), pred))
    return res


def direction(W: np.ndarray) -> np.ndarray:
    """多类权重 [C,D] 的主判别方向 = 中心化后最大奇异向量（C=2 即两类差向量）。"""
    Wc = W - W.mean(axis=0, keepdims=True)
    if Wc.shape[0] < 2:
        return Wc[0]
    _, _, vt = np.linalg.svd(Wc, full_matrices=False)
    return vt[0]


def pred1d(Atr, ytr, Ate, w) -> np.ndarray:
    """方向 w 的一维最近质心判别（各任务同一协议 ⇒ 隔离「方向不对」这一变量）。"""
    s1 = np.asarray(Atr, dtype=np.float64) @ w
    s2 = np.asarray(Ate, dtype=np.float64) @ w
    ytr = np.asarray(ytr)
    classes = np.unique(ytr)
    cents = {int(c): float(s1[ytr == c].mean()) for c in classes}
    return np.array([min(classes, key=lambda c: abs(v - cents[int(c)])) for v in s2])


def paired_se(cj, cs) -> float:
    d = np.asarray(cj, dtype=np.float64) - np.asarray(cs, dtype=np.float64)
    return float(d.std(ddof=1) / math.sqrt(len(d))) if len(d) > 1 else 0.0


def se_binom(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


# ══════════════════════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════════════════════
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="full", choices=("p0", "full"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true", help="缩小样本/步数，只验代码通不通")
    args = ap.parse_args()
    steps = args.steps

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    preg = HERE / "PREREG.md"
    log(f"device={device} | phase={args.phase} steps={steps} | PREREG mtime="
        f"{time.strftime('%F %T', time.localtime(preg.stat().st_mtime))}")

    tok = NanoCharTokenizer()
    enc, base_ck = load_base_encoder(str(BASE), device)
    enc.eval()
    log(f"核参数 {sum(p.numel() for p in enc.parameters()):,} | hidden={base_ck['hidden_dim']} | eval={not enc.training}")

    sets = build_sets()
    if args.smoke:
        cap = {"emo": 400, "skel": 400, "logic": 400, "idr": 600}
        for t, k in cap.items():
            for split in ("tr", "te"):
                for key in (f"{split}_text", f"{split}_y", f"{split}_span"):
                    if key in sets[t]:
                        sets[t][key] = sets[t][key][:k]
        steps = min(steps, 60)
        globals()["P0_MARKER_SPLIT"] = 600
        globals()["SYNTH_N"] = 800
        globals()["SYNTH_N2"] = 20000
        log(f"SMOKE：每任务行数截到 {cap}，marker 800 / synth 800，steps={steps}")

    # ── 抽表征 ─────────────────────────────────────────────────────────
    F: dict = {}
    ext_meta: dict = {}
    for t in TASKS:
        if t == "logic":                      # 与 skel 同一批句子 ⇒ 复用同一份表征
            F["logic"] = F["skel"]
            ext_meta["logic"] = {**ext_meta["skel"], "shared_with": "skel"}
            continue
        d = sets[t]
        ml = MAX_LEN[t]
        if d["kind"] == "text":
            Xtr, m1 = extract_texts(enc, tok, d["tr_text"], ml, device)
            Xte, m2 = extract_texts(enc, tok, d["te_text"], ml, device)
        else:
            # 提及级：文本去重前向 + 按 span 取各层均值（无效 span 的行丢弃）
            Xtr, Xsp_tr, m1 = extract(enc, tok, d["tr_text"], ml, device, spans=d["tr_span"])
            Xte, Xsp_te, m2 = extract(enc, tok, d["te_text"], ml, device, spans=d["te_span"])
            keep_tr = ~np.isnan(Xsp_tr[:, 0, 0])
            keep_te = ~np.isnan(Xsp_te[:, 0, 0])
            Xtr, Xte = Xsp_tr[keep_tr], Xsp_te[keep_te]
            d["tr_y"] = [y for y, k in zip(d["tr_y"], keep_tr) if k]
            d["te_y"] = [y for y, k in zip(d["te_y"], keep_te) if k]
            m1["mention_dropped"] += int((~keep_tr).sum() + (~keep_te).sum())
        F[t] = {"tr": Xtr, "te": Xte}
        ext_meta[t] = {**m1, **{f"te_{k}": v for k, v in m2.items()}}
        log(f"  抽取 {t}: tr={F[t]['tr'].shape} te={F[t]['te'].shape} "
            f"截断 {ext_meta[t].get('truncated')}/{ext_meta[t].get('n_unique')} 行 {ext_meta[t].get('n_rows')}")

    naive = naive_stats(sets)
    for t in TASKS:
        s = naive[t]
        log(f"  任务 {t}: n_tr={s['n_train']} n_te={s['n_test']} C={s['n_classes']} "
            f"maj={s['max_naive@majority']:.4f} rule={s['max_naive@rule']:.4f}")

    # 全局标准化统计（每层一份，四任务共用 ⇒ 方向可比）
    GSTATS = {}
    for li in range(len(LAYERS)):
        alltr = np.concatenate([F[t]["tr"][:, li, :] for t in TASKS], axis=0)
        mu, sd = alltr.mean(0), alltr.std(0)
        GSTATS[li] = (mu, np.where(sd < 1e-6, 1.0, sd))

    results: dict = {
        "meta": {"device": str(device), "steps": steps, "dir_steps": DIR_STEPS,
                 "layers": list(LAYERS), "seeds": list(SEEDS), "k_bottle": K_BOTTLE,
                 "max_len": MAX_LEN, "lr": LR, "n_init_dir": N_INIT,
                 "extraction": ext_meta,
                 "prereg_mtime": time.strftime("%F %T", time.localtime(preg.stat().st_mtime))},
        "labels": naive, "p0": {}, "p1": {}, "p3": {}, "p2": {}}

    # ══════════ P0 ① 随机标签 ══════════
    log("== P0 ① 随机标签 ==")
    p0_rand = {}
    rng = np.random.default_rng(42)
    for t in TASKS:
        d, X = sets[t], F[t]
        ytr = np.array(d["tr_y"])
        yte = np.array(d["te_y"])
        rtr = rng.permutation(ytr)
        rte = rng.permutation(yte)
        maj = naive[t]["max_naive@majority"]
        p0_rand[t] = {}
        for li, ln in enumerate(LAYERS):
            accs = [fit_probe(X["tr"][:, li, :], rtr, X["te"][:, li, :], rte,
                              d["classes"], seed, steps=steps, device=device)[0]
                    for seed in SEEDS]
            thr = maj + 2 * se_binom(float(np.mean(accs)), len(yte))
            p0_rand[t][ln] = {"acc": [round(a, 4) for a in accs], "threshold": round(thr, 4),
                              "pass": bool(max(accs) <= thr)}
            log(f"  {t}/{ln}: rand={['%.4f' % a for a in accs]} thr={thr:.4f} "
                f"{'PASS' if max(accs) <= thr else 'FAIL'}")
    results["p0"]["random_label"] = p0_rand

    # ══════════ P0 ② 构造可分 ══════════
    log("== P0 ② 构造可分（输入内标记 / 合成张量） ==")
    n_marker = min(P0_MARKER_SPLIT * 4 // 3, len(sets["emo"]["tr_text"]) // 2 * 4 // 3)
    mk_split = min(P0_MARKER_SPLIT, n_marker // 2)
    emo_pool = sets["emo"]["tr_text"][:n_marker]
    perm = np.random.default_rng(42).permutation(len(emo_pool))
    emo_pool = [emo_pool[i] for i in perm]
    g = np.array([i % 2 for i in range(len(emo_pool))])
    marked = [(("甲" * MARKER_N) if gv == 0 else ("乙" * MARKER_N)) + t[:48]
              for t, gv in zip(emo_pool, g)]
    Xm_tr, meta_m1 = extract_texts(enc, tok, marked[:mk_split], 64, device)
    Xm_te, meta_m2 = extract_texts(enc, tok, marked[mk_split:], 64, device)
    p0_mark = {}
    for li, ln in enumerate(LAYERS):
        accs = [fit_probe(Xm_tr[:, li, :], g[:mk_split],
                          Xm_te[:, li, :], g[mk_split:], 2, seed,
                          steps=steps, device=device)[0] for seed in SEEDS]
        p0_mark[ln] = {"acc": [round(a, 4) for a in accs],
                       "pass_best_layer": None,
                       "pass": bool(min(accs) >= P0_MARKER_ACC)}
        log(f"  marker/{ln}: {['%.4f' % a for a in accs]} "
            f"{'PASS' if min(accs) >= P0_MARKER_ACC else 'FAIL'}")
    best_mark = max(LAYERS, key=lambda ln: max(p0_mark[ln]["acc"]))
    p0_mark["_best_layer"] = best_mark
    p0_mark["_best_pass"] = p0_mark[best_mark]["pass"]
    results["p0"]["marker_in_input"] = {**p0_mark, "extraction": [meta_m1, meta_m2]}

    def synth_run(n: int, tag: str) -> dict:
        rs = np.random.default_rng(42)
        Xs = rs.normal(size=(n, 128)).astype(np.float32)
        w_true = rs.normal(size=128)
        ys = (Xs @ w_true > 0).astype(np.int64)
        sp = int(n * 0.75)
        out = {"n": n, "n_train": sp, "d": 128, "label": "sign(X·w_true)"}
        for seed in SEEDS:
            acc, _, _ = fit_probe(Xs[:sp], ys[:sp], Xs[sp:], ys[sp:], 2, seed,
                                  steps=steps, device=device)
            out[str(seed)] = round(acc, 4)
        out["pass"] = bool(min(out[str(s)] for s in SEEDS) >= P0_MARKER_ACC)
        log(f"  synthetic[{tag}]: {out}")
        return out

    p0_syn = synth_run(SYNTH_N, "PREREG 字面 n=4000")
    p0_syn2 = synth_run(SYNTH_N2, "功效修正 n=" + str(SYNTH_N2))
    results["p0"]["synthetic_linear"] = p0_syn
    results["p0"]["synthetic_linear_scaled"] = p0_syn2
    results["p0"]["synthetic_note"] = (
        "p=128 时方向估计误差 θ≈sqrt(p/n) ⇒ n=4000 的 0.99 门槛被统计下限卡住"
        "（实测 ~0.977）；门槛不改，另跑 n=20000 确认测量本身可达 ≈100%。见 AMENDMENTS.md。")

    rand_ok = all(p0_rand[t][ln]["pass"] for t in TASKS for ln in LAYERS)
    p0_pass = bool(rand_ok and p0_mark["_best_pass"] and p0_syn["pass"])
    p0_pass_eff = bool(rand_ok and p0_mark["_best_pass"] and p0_syn2["pass"])
    results["p0"]["pass"] = p0_pass                    # PREREG 字面口径
    results["p0"]["pass_effective"] = p0_pass_eff      # 功效修正口径（AMENDMENTS.md）
    results["p0"]["gate_used_for_continue"] = "pass_effective"
    results["p0"]["read_marker_pass_rule"] = ("判据：最佳层 acc ≥ 0.99（PREREG §3 未限定层，"
                                              "取最佳层判定并逐层全报）")
    log(f"P0 门禁：字面={'PASS' if p0_pass else 'FAIL'} 有效="
        f"{'PASS' if p0_pass_eff else 'FAIL'}（rand={rand_ok} "
        f"marker@best={p0_mark['_best_pass']} synth4000={p0_syn['pass']} "
        f"synth{SYNTH_N2}={p0_syn2['pass']}）")

    dst = HERE / "results"
    dst.mkdir(exist_ok=True)
    if args.phase == "p0":
        (dst / f"p0{args.tag}.json").write_text(dump(results))
        return 0 if p0_pass_eff else 4
    if not p0_pass_eff:
        (dst / f"probe{args.tag}_p0failed.json").write_text(dump(results))
        log("P0 有效口径未过 ⇒ 按 PREREG §3 停；结果已写。")
        return 4
    if not p0_pass:
        log("⚠ P0 字面口径（synth n=4000 < 0.99）不过，但有效口径过 ⇒ 按 AMENDMENTS.md 继续，明示偏离。")

    # ══════════ P1 / P3 分层单任务 probe ══════════
    log("== P1/P3 分层单任务 probe ==")
    fits: dict = {}
    for t in TASKS:
        d, X = sets[t], F[t]
        ytr, yte = np.array(d["tr_y"]), np.array(d["te_y"])
        maj = naive[t]["max_naive@majority"]
        fits[t] = {}
        p1_t = {}
        for li, ln in enumerate(LAYERS):
            fits[t][ln] = {}
            for seed in SEEDS:
                fits[t][ln][seed] = fit_probe(
                    X["tr"][:, li, :], ytr, X["te"][:, li, :], yte, d["classes"],
                    seed, steps=steps, device=device, stats=GSTATS[li])
            accs = [fits[t][ln][s][0] for s in SEEDS]
            mean = float(np.mean(accs))
            se = se_binom(mean, len(yte))
            thr = maj + 2 * se
            p1_t[ln] = {"acc": [round(a, 5) for a in accs], "mean": round(mean, 5),
                        "se": round(se, 5), "thr_maj+2se": round(thr, 5),
                        "pass": bool(mean > thr),
                        "seed_same_side": bool(all(a > thr for a in accs) or
                                               all(a <= thr for a in accs))}
        best = max(LAYERS, key=lambda ln: p1_t[ln]["mean"])
        p1_t["_best_layer"] = best
        p1_t["_pass"] = bool(p1_t[best]["pass"])
        results["p1"][t] = p1_t
        results["p3"][t] = {ln: {k: p1_t[ln][k] for k in
                                 ("acc", "mean", "se", "thr_maj+2se", "pass",
                                  "seed_same_side")} for ln in LAYERS}
        log(f"  {t}: best={best} acc={p1_t[best]['acc']} mean={p1_t[best]['mean']:.4f} "
            f"maj={maj:.4f} thr={p1_t[best]['thr_maj+2se']:.4f} "
            f"{'PASS' if p1_t[best]['pass'] else 'FAIL'}")

    readable = [t for t in TASKS if results["p1"][t]["_pass"]]
    results["p1"]["readable"] = readable
    results["p1"]["n_readable"] = len(readable)
    log(f"可读任务：{readable}")

    # ══════════ P2-1 方向余弦 ══════════
    log("== P2 方向余弦（8 init/任务/层） ==")
    dirs: dict = {}
    for t in TASKS:
        d, X = sets[t], F[t]
        ytr, yte = np.array(d["tr_y"]), np.array(d["te_y"])
        dirs[t] = {}
        for li, ln in enumerate(LAYERS):
            dl = []
            for seed in SEEDS:
                for k in range(N_INIT):
                    _, _, W = fit_probe(X["tr"][:, li, :], ytr, X["te"][:, li, :], yte,
                                        d["classes"], seed * 100 + k, steps=DIR_STEPS,
                                        device=device, stats=GSTATS[li])
                    dl.append(direction(W))
            dirs[t][ln] = dl

    def cos_stats(v1, v2, self_pairs=False):
        pairs = [(a, b) for a, b in itertools.product(v1, v2)]
        if not self_pairs:
            pairs = [(v1[i], v1[j]) for i in range(len(v1)) for j in range(i + 1, len(v1))] \
                if v1 is v2 else pairs
        if v1 is v2 and self_pairs:
            pairs = [(v1[i], v1[j]) for i in range(len(v1)) for j in range(i + 1, len(v1))]
        arr = np.array([abs(float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)))
                        for a, b in pairs])
        return {"n": int(len(arr)), "mean": round(float(arr.mean()), 4),
                "p25": round(float(np.percentile(arr, 25)), 4),
                "p50": round(float(np.percentile(arr, 50)), 4),
                "p75": round(float(np.percentile(arr, 75)), 4)}

    cos = {"within": {}, "between": {}}
    for t in TASKS:
        cos["within"][t] = {ln: cos_stats(dirs[t][ln], dirs[t][ln], self_pairs=True)
                            for ln in LAYERS}
    for a, b in PRIM_PAIRS + AUX_PAIRS:
        cos["between"][f"{a}|{b}"] = {ln: cos_stats(dirs[a][ln], dirs[b][ln]) for ln in LAYERS}
    results["p2"]["cosine"] = cos
    for k, v in cos["between"].items():
        log(f"  between {k}: " + " ".join(f"{ln}={v[ln]['p50']}" for ln in LAYERS))
    log("  within: " + " ".join(
        f"{t}={cos['within'][t]['pool']['p50']}" for t in TASKS))

    # ══════════ P2-2 交叉干扰 ══════════
    log("== P2 交叉干扰（一维最近质心 + 配对 SE） ==")
    cross: dict = {}
    # 双向都要：src 的判别方向拿去读 dst 的数据，X = acc(自己的方向) − acc(src 的方向)
    pair_dirs = []
    for a, b in PRIM_PAIRS + AUX_PAIRS:
        pair_dirs += [(a, b), (b, a)]
    for src_t, dst_t in pair_dirs:
        key = f"{src_t}->{dst_t}"
        cross[key] = {}
        d_dst = sets[dst_t]
        ytr_d, yte_d = np.array(d_dst["tr_y"]), np.array(d_dst["te_y"])
        for li, ln in enumerate(LAYERS):
            Atr, Ate, _ = standardize(F[dst_t]["tr"][:, li, :].astype(np.float32),
                                      F[dst_t]["te"][:, li, :].astype(np.float32))
            row = {}
            for seed in SEEDS:
                w_dst = direction(fits[dst_t][ln][seed][2])
                self_pred = pred1d(Atr, ytr_d, Ate, w_dst)
                self_ok = (self_pred == yte_d)
                xs, ses = [], []
                for w_src in dirs[src_t][ln]:
                    cr_pred = pred1d(Atr, ytr_d, Ate, w_src)
                    cr_ok = (cr_pred == yte_d)
                    xs.append(float(self_ok.mean() - cr_ok.mean()))
                    ses.append(paired_se(self_ok, cr_ok))
                row[str(seed)] = {
                    "acc_self_1d": round(float(self_ok.mean()), 4),
                    "acc_cross_1d_mean": round(float(self_ok.mean() - float(np.mean(xs))), 4),
                    "X_drop": round(float(np.mean(xs)), 4),
                    "X_drop_each": [round(x, 4) for x in xs],
                    "se_paired": round(float(np.mean(ses)), 5),
                    "sig": bool(np.mean(xs) > 2 * np.mean(ses)),
                    "seed_sign_same": None,
                }
            row["seed_sign_same"] = bool(
                (row["42"]["X_drop"] > 0) == (row["43"]["X_drop"] > 0))
            cross[key][ln] = row
        log(f"  {key}: " + " ".join(
            f"{ln}={cross[key][ln]['42']['X_drop']:+.3f}" for ln in LAYERS))
    results["p2"]["cross_interference"] = cross

    # ══════════ P2-3 联合 probe（k=8 瓶颈） ══════════
    log("== P2 联合 probe（共享瓶颈 k=8 vs 单任务同瓶颈） ==")
    joint: dict = {}
    for li, ln in enumerate(LAYERS):
        datas_tr = [(F[t]["tr"][:, li, :], np.array(sets[t]["tr_y"])) for t in TASKS]
        datas_te = [(F[t]["te"][:, li, :], np.array(sets[t]["te_y"])) for t in TASKS]
        for seed in SEEDS:
            model, _ = fit_joint(datas_tr, K_BOTTLE, seed, device, steps=steps, shared=True)
            jres = eval_joint(model, datas_te, [st for st in
                                                [standardize(np.asarray(A, np.float32),
                                                             np.asarray(A, np.float32))[2]
                                                 for A, _ in datas_tr]], device)
            key = f"{ln}_s{seed}"
            joint[key] = {}
            for i, t in enumerate(TASKS):
                m_t, st_t = fit_joint([datas_tr[i]], K_BOTTLE, seed, device,
                                      steps=steps, shared=True)
                s_acc, s_pred = eval_joint(m_t, [datas_te[i]], st_t, device)[0]
                j_acc, j_pred = jres[i]
                se = paired_se(np.asarray(j_pred) == np.asarray(datas_te[i][1]),
                               np.asarray(s_pred) == np.asarray(datas_te[i][1]))
                joint[key][t] = {"joint": round(j_acc, 4), "single_k8": round(s_acc, 4),
                                 "delta": round(j_acc - s_acc, 4), "se_paired": round(se, 5),
                                 "sig_drop": bool((j_acc - s_acc) < -2 * se)}
            log(f"  joint {key}: " + " ".join(
                f"{t}={joint[key][t]['delta']:+.3f}({'*' if joint[key][t]['sig_drop'] else '-'})"
                for t in TASKS))
    results["p2"]["joint_k8"] = joint

    # ══════════ P2-4 零共享 sanity ══════════
    log("== P2 零共享 sanity（构造等价） ==")
    zshare: dict = {}
    for li, ln in enumerate(LAYERS):
        datas_tr = [(F[t]["tr"][:, li, :], np.array(sets[t]["tr_y"])) for t in TASKS]
        datas_te = [(F[t]["te"][:, li, :], np.array(sets[t]["te_y"])) for t in TASKS]
        stats_l = [standardize(np.asarray(A, np.float32), np.asarray(A, np.float32))[2]
                   for A, _ in datas_tr]
        model, _ = fit_joint(datas_tr, None, 42, device, steps=steps, shared=False)
        zres = eval_joint(model, datas_te, stats_l, device)
        zshare[ln] = {}
        for i, t in enumerate(TASKS):
            j_acc, j_pred = zres[i]
            # 同口径基线：单任务、无瓶颈、同一份 per-task 标准化、同一 seed
            # （原先用的是 P1 的全局标准化 probe ⇒ Δ 混入标准化差异，口径不等价）
            m_s, st_s = fit_joint([datas_tr[i]], None, 42, device, steps=steps, shared=True)
            s_acc, s_pred = eval_joint(m_s, [datas_te[i]], st_s, device)[0]
            se = paired_se(np.asarray(j_pred) == np.asarray(datas_te[i][1]),
                           np.asarray(s_pred) == np.asarray(datas_te[i][1]))
            zshare[ln][t] = {"zero_shared": round(j_acc, 4), "single": round(s_acc, 4),
                             "delta": round(j_acc - s_acc, 4), "se_paired": round(se, 5),
                             "ok": bool(abs(j_acc - s_acc) <= 2 * max(se, 1e-6))}
        log(f"  zero-share {ln}: " + " ".join(
            f"{t}={zshare[ln][t]['delta']:+.4f}{'✓' if zshare[ln][t]['ok'] else '✗'}"
            for t in TASKS))
    results["p2"]["zero_shared_sanity"] = zshare

    # ══════════ 判定（PREREG §6） ══════════
    log("== 判定（PREREG §6） ==")
    verdict_src = {}
    for a, b in PRIM_PAIRS + AUX_PAIRS:
        acca = np.array([results["p1"][a][ln]["mean"] for ln in LAYERS])
        accb = np.array([results["p1"][b][ln]["mean"] for ln in LAYERS])
        li = int(np.argmax(np.minimum(acca / acca.max(), accb / accb.max())))
        ln = LAYERS[li]
        c = cos["between"][f"{a}|{b}"][ln]["p50"]
        within_ok = all(cos["within"][t][ln]["p50"] >= 0.5 for t in (a, b))
        # 联合下降：取该对两个任务里更差的那个（保守）
        d_t = {}
        for t in (a, b):
            ds = [joint[f"{ln}_s{s}"][t]["delta"] for s in SEEDS]
            ses = [joint[f"{ln}_s{s}"][t]["se_paired"] for s in SEEDS]
            d_t[t] = {"delta": ds, "se": ses,
                      "sig_drop": bool(np.mean(ds) < -2 * np.mean(ses)),
                      "seed_sign_same": bool((np.mean(ds) < 0) == all(d < 0 for d in ds))}
        worse = min(d_t, key=lambda t: np.mean(d_t[t]["delta"]))
        xb = cross[f"{a}->{b}"][ln]
        xa = cross[f"{b}->{a}"][ln]

        def x_stat(row: dict) -> tuple[float, float, bool]:
            """两 seed 合并的 (X 均值, 2SE 均值, 是否显著)。"""
            xs = [row[str(s)]["X_drop"] for s in SEEDS]
            ses = [row[str(s)]["se_paired"] for s in SEEDS]
            return float(np.mean(xs)), float(2 * np.mean(ses)), bool(np.mean(xs) > 2 * np.mean(ses))

        xb_m, xb_2se, xb_sig = x_stat(xb)
        xa_m, xa_2se, xa_sig = x_stat(xa)
        x_val = max(xb_m, xa_m)
        x_2se = max(xb_2se, xa_2se)
        x_sig = bool(xb_sig or xa_sig)
        verdict_src[f"{a}|{b}"] = {
            "L_star": ln, "cos_p50": c, "within_ok": within_ok,
            "joint": d_t, "worse_task": worse,
            "joint_sig_drop": bool(d_t[worse]["sig_drop"]),
            "cross_X": {f"{a}->{b}": {s: xb[s]["X_drop"] for s in ("42", "43")},
                        f"{b}->{a}": {s: xa[s]["X_drop"] for s in ("42", "43")}},
            "cross_sig": x_sig, "cross_X_max": x_val, "cross_2SE_max": x_2se,
            "cross_seed_sign_same": bool(xb["seed_sign_same"] and xa["seed_sign_same"]),
        }
        log(f"  pair {a}|{b} @ {ln}: cos={c:.3f} jointΔ={d_t[worse]['delta']} "
            f"X={x_val:+.3f} sig={x_sig} within_ok={within_ok}")

    results["p2"]["pairs"] = verdict_src
    readable_pairs = [k for k in verdict_src
                      if k.split("|")[0] in readable and k.split("|")[1] in readable]
    sep = bool(readable_pairs and all(
        abs(verdict_src[k]["cos_p50"]) <= 0.3
        and not verdict_src[k]["joint_sig_drop"]
        and not verdict_src[k]["cross_sig"] for k in readable_pairs))
    ent_pairs = [k for k in readable_pairs
                 if abs(verdict_src[k]["cos_p50"]) >= 0.6
                 and verdict_src[k]["joint_sig_drop"] and verdict_src[k]["cross_sig"]]
    within_gate = all(cos["within"][t][results["p1"][t]["_best_layer"]]["p50"] >= 0.5
                      for t in readable)
    gate = bool(p0_pass_eff and len(readable) >= 2 and within_gate)
    if not gate:
        verdict = "证据不足"
    elif sep:
        verdict = "可分离"
    elif len(ent_pairs) >= 2:
        verdict = "缠绕"
    else:
        verdict = "证据不足"
    results["verdict"] = {
        "verdict": verdict, "gate": gate, "p0_pass_prereg_literal": p0_pass,
        "p0_pass_effective": p0_pass_eff, "readable": readable,
        "n_readable": len(readable), "within_gate": within_gate,
        "readable_pairs": readable_pairs, "separable_ok": sep, "entangled_pairs": ent_pairs,
        "within_stable": {t: {ln: cos["within"][t][ln]["p50"] for ln in LAYERS}
                          for t in TASKS},
    }
    log(f"判定 = {verdict}（可读={readable} 可读对={readable_pairs} 缠绕对={ent_pairs}）")

    outp = dst / f"probe{args.tag}.json"
    outp.write_text(dump(results))
    log(f"结果写入 {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
