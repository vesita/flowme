#!/usr/bin/env python3
"""select_semantic_joint 公共件：口径、数据（只读）、模型、评测、老卡 step-0 对账。

两臂唯一变量 = **核是否可训**（PREREG §2）：
  frozen = select_pool 臂 A 逐行复刻（读其 token 级缓存，**只读**）
  joint  = 同配方 + 核可训（warm start）+ 四张老卡头挂载并冻结 + 每步 4 个老任务批次

所有路径只读他人目录；写入仅限本目录与 logs/、/tmp。
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import pickle
import sys
from pathlib import Path

sys.dont_write_bytecode = True          # 绝不在只读目录落 __pycache__

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "capability_map"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.tasks.artifacts import load_base_encoder, read_card  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import (  # noqa: E402
    GenericTaskDataset, _rollout, _truth_of, evaluate_task, task_loss)

from prepare import TASK_SAMPLES  # noqa: E402  (capability_map，只读常量)

# ---------------- 路径（他人目录一律只读） ----------------
DATA = ROOT / "experiments" / "select_rerank" / "data"            # 只读
SR_CACHE = ROOT / "experiments" / "select_rerank" / "cache"        # 只读（Q4a 对照）
SP_CACHE = ROOT / "experiments" / "select_pool" / "cache"          # 只读（token 级缓存）
CAPMAP_CACHE = ROOT / "experiments" / "capability_map" / "cache"   # 只读
CAPMAP_SUMMARY = ROOT / "experiments" / "capability_map" / "summary.json"
BASE_CKPT = ROOT / "checkpoints" / "base_encoder.pt"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
CACHE = HERE / "cache"
LOGS = ROOT / "logs"

# ---------------- 配方（PREREG §2/§3 写死） ----------------
STEPS = 1800
BATCH = 64
LR_HEAD = 1e-3
LR_CORE = 3e-4                 # = core_keep --lr-base（§15 已验证），跑前定死不扫
WD = 1e-4
CLIP = 1.0
SEEDS = (42, 43)
DATA_ARM = "clean"
SPLITS = ("train", "test", "adv")
CTYPE = {0: "-", 1: "heldout_pair", 2: "shifted_pos"}
EXPECTED_N = {"train": 8000, "test": 2500, "adv": 2500}
FP_EXPECTED = "28988c64fa9145967fee225dc4afb37d"      # select_pool/results/A_s42.json
#: R0：select_pool 臂 A（= B 族基线）
R0_EXPECT = {42: {"heldout_pair": 49.52, "shifted_pos": 92.56},
             43: {"heldout_pair": 51.04, "shifted_pos": 92.48}}
#: R5 噪声带（= core_keep/eval_core_keep.py::BAND）
BAND = {"pronoun": 0.0283, "sentiment": 0.0041, "relation": 0.0139, "person": 0.0033}
OLD_CARDS = ["pronoun", "sentiment", "relation", "person"]
HEAD_PARAMS = 164_353
ENC_PARAMS = 1_688_460
SE_1250 = math.sqrt(0.25 / 1250)          # 0.0141421...
TWO_SE = 2 * SE_1250                      # 0.0282842... = 2.828pt


# ================= 数据（只读） =================
def load_rows(arm: str = DATA_ARM) -> list[dict]:
    """行序逐行复刻 select_rerank/train_rerank.py::load_rows（train→test→adv）。"""
    rows: list[dict] = []
    for s in SPLITS:
        p = DATA / arm / f"{s}.jsonl"
        if not p.exists():
            raise SystemExit(f"缺数据文件：{p}（只读复用 select_rerank/data）")
        with open(p, encoding="utf-8") as fp:
            for line in fp:
                r = json.loads(line)
                r["split_name"] = s
                rows.append(r)
    return rows


def data_fingerprint(arm: str = DATA_ARM) -> str:
    h = hashlib.md5()
    for s in SPLITS:
        h.update((DATA / arm / f"{s}.jsonl").read_bytes())
    return h.hexdigest()


def load_blob() -> dict:
    """只读加载 select_pool 的 token 级缓存（本目录不重建、不改写）。"""
    fp = data_fingerprint()
    if fp != FP_EXPECTED:
        raise SystemExit(f"数据指纹漂移：{fp} ≠ {FP_EXPECTED}（select_rerank/data 被改过？）")
    key = f"{DATA_ARM}_{fp[:12]}_tok_ctx64_cand32.pt"
    path = SP_CACHE / key
    if not path.exists():
        raise SystemExit(f"缺 select_pool 缓存（只读复用）：{path}")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    if blob["fingerprint"] != fp:
        raise SystemExit("缓存指纹与数据对不上")
    print(f"[cache] 只读命中 {path}（n={blob['n']}）", flush=True)
    return blob


# ================= 口径（与 select_pool/model_pool.py 逐位同式） =================
def _axis(h: torch.Tensor, m: torch.Tensor) -> int:
    assert h.dim() == m.dim() + 1 and h.dim() in (3, 4), f"h {h.dim()} m {m.dim()}"
    ax = h.dim() - 2
    assert ax == m.dim() - 1
    return ax


def masked_mean(h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    ax = _axis(h, m)
    mf = m.unsqueeze(-1).to(h.dtype)
    return (h * mf).sum(ax) / mf.sum(ax).clamp(min=1.0)


class ScoringHead(nn.Module):
    """与 select_rerank/model.py::ScoringHead、select_pool/model_pool.py 逐行同构。"""

    def __init__(self, hidden: int = 128, head_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 4, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(self, v_ctx: torch.Tensor, v_cand: torch.Tensor) -> torch.Tensor:
        B, K, D = v_cand.shape
        c = v_ctx.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([c, v_cand, c * v_cand, (c - v_cand).abs()], dim=-1)
        return self.net(x).squeeze(-1)


class SemSpec:
    """截断与结构口径 = select_rerank/build_data.py::SPEC（文本读，不 import ⇒ 不落 pyc）。"""

    max_len_ctx: int = 64
    max_len_cand: int = 32
    hidden: int = 128
    head_hidden: int = 256
    k: int = 2
    base_path: str = str(BASE_CKPT)

    @staticmethod
    def from_build_spec() -> "SemSpec":
        p = ROOT / "experiments" / "select_rerank" / "build_data.py"
        txt = p.read_text(encoding="utf-8")
        import re
        m = re.search(r'^SPEC\s*=\s*\{"max_len_ctx":\s*(\d+),\s*"max_len_cand":\s*(\d+)\}', txt, re.M)
        if not m:
            raise SystemExit(f"读不到 SPEC：{p}")
        sp = SemSpec()
        sp.max_len_ctx, sp.max_len_cand = int(m.group(1)), int(m.group(2))
        if (sp.max_len_ctx, sp.max_len_cand) != (64, 32):
            raise SystemExit(f"SPEC 漂移：{sp.max_len_ctx}/{sp.max_len_cand} ≠ 64/32")
        return sp


class SemModel(nn.Module):
    """核 + mean-pool + 打分头。构造顺序与 select_pool PoolModel **逐字一致**（RNG 流同源）。"""

    def __init__(self, spec: SemSpec | None = None, freeze_core: bool = True):
        super().__init__()
        self.spec = spec or SemSpec()
        self.arm = "A"                       # 本实验固定 mean-pool
        self.encoder, self.ckpt = load_base_encoder(self.spec.base_path)
        self.encoder.eval()                       # 两臂全程 eval（dropout=0 ⇒ 与 train 逐位同）
        for p in self.encoder.parameters():
            p.requires_grad_(not freeze_core)     # frozen: False / joint: True
        self.head = ScoringHead(self.spec.hidden, self.spec.head_hidden)

    # ---- 聚合（臂 A：两侧都是 masked mean-pool） ----
    def aggregate(self, h_ctx, m_ctx, h_cand, m_cand):
        return masked_mean(h_ctx, m_ctx), masked_mean(h_cand, m_cand)

    def forward_tokens(self, h_ctx, m_ctx, h_cand, m_cand) -> torch.Tensor:
        v_ctx, v_cand = self.aggregate(h_ctx, m_ctx, h_cand, m_cand)
        return self.head(v_ctx, v_cand)

    def freeze_report(self) -> dict:
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        enc_r = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_p = sum(p.numel() for p in self.head.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert enc_p == ENC_PARAMS, f"核参数量变了：{enc_p}"
        assert head_p == HEAD_PARAMS, f"打分头参数量变了：{head_p}"
        assert not self.encoder.training, "核不在 eval() 模式"
        assert trainable == enc_r + head_p, f"可训参数对不上：{trainable} != {enc_r}+{head_p}"
        return {"arm": self.arm, "encoder_params": enc_p, "encoder_trainable": enc_r,
                "encoder_training": self.encoder.training, "head_params": head_p,
                "trainable_params": trainable, "total_params": enc_p + head_p,
                "hidden": self.spec.hidden}


# ================= live 编码（joint 臂：核可训 ⇒ 不用缓存） =================
@torch.no_grad()
def _encode_ids(enc, texts: list[str], max_len: int, device: str, bs: int = 256):
    tok = NanoCharTokenizer()
    hs, ms = [], []
    for i in range(0, len(texts), bs):
        chunk = texts[i:i + bs]
        ids, mask = [], []
        for t in chunk:
            e = tok.encode(t, max_length=max_len, padding=True)
            ids.append(e["input_ids"])
            mask.append(e["attention_mask"])
        id_t = torch.tensor(ids, dtype=torch.long, device=device)
        m_t = torch.tensor(mask, dtype=torch.bool, device=device)
        h = enc(id_t, m_t)
        hs.append(h)
        ms.append(m_t)
    if not hs:
        raise SystemExit("空编码批次")
    return torch.cat(hs, 0), torch.cat(ms, 0)


def encode_train(enc, texts: list[str], max_len: int, device: str):
    """训练用：不进 no_grad，梯度流回核。与 `_encode_ids` 同一条编码路径。"""
    tok = NanoCharTokenizer()
    ids, mask = [], []
    for t in texts:
        e = tok.encode(t, max_length=max_len, padding=True)
        ids.append(e["input_ids"])
        mask.append(e["attention_mask"])
    id_t = torch.tensor(ids, dtype=torch.long, device=device)
    m_t = torch.tensor(mask, dtype=torch.bool, device=device)
    return enc(id_t, m_t), m_t


def inputs_for(model: SemModel, blob: dict, rows: list[dict], j: torch.Tensor,
               live: bool, device: str):
    """live=False ⇒ 读缓存（frozen 臂，与 select_pool 逐位同输入）；live=True ⇒ 现算。"""
    if not live:
        return (blob["h_ctx"][j].to(device), blob["m_ctx"][j].to(device),
                blob["h_cand"][j].to(device), blob["m_cand"][j].to(device))
    idx = [int(x) for x in j.tolist()]
    h_c, m_c = encode_train(model.encoder, [rows[i]["context"] for i in idx],
                            model.spec.max_len_ctx, device)
    flat = [c for i in idx for c in rows[i]["candidates"]]
    h_d, m_d = encode_train(model.encoder, flat, model.spec.max_len_cand, device)
    n = len(idx)
    return (h_c, m_c, h_d.view(n, model.spec.k, model.spec.max_len_cand, -1),
            m_d.view(n, model.spec.k, model.spec.max_len_cand))


# ================= 选择任务评测 =================
def evaluate(model: SemModel, blob: dict, idx: torch.Tensor, rows: list[dict],
             live: bool, device: str, bs: int | None = None) -> dict:
    # live 要现算核前向 ⇒ 批小一点（8GB iGPU），缓存路径保持 select_pool 的 1024
    bs = bs or (512 if live else 1024)
    model.eval()
    y_all = blob["labels"][idx]
    correct, logit_var = [], []
    loss_sum = 0.0
    with torch.no_grad():
        for i in range(0, len(idx), bs):
            sub = idx[i:i + bs]
            c, mc, d, md = inputs_for(model, blob, rows, sub, live, device)
            t = y_all[i:i + bs].to(device)
            logits = model.forward_tokens(c, mc, d, md)
            assert torch.isfinite(logits).all(), "logits 出现非有限值"
            loss_sum += torch.nn.functional.cross_entropy(logits, t, reduction="sum").item()
            correct.append((logits.argmax(-1) == t).cpu())
            logit_var.append(float(logits.float().var()))
    ok = torch.cat(correct)
    n = len(idx)
    assert n > 0, "空评测集"
    assert min(logit_var) > 0, "logits 是常数（空跑/塌缩）"
    acc = float(ok.float().mean())
    se = math.sqrt(0.25 / n)
    maj = float(torch.bincount(y_all, minlength=2).float().max() / n)
    return {"n": n, "acc": round(acc, 6), "ce": round(loss_sum / n, 6),
            "se": round(se, 6), "margin": round(acc - 0.5, 6),
            "margin_over_se": round((acc - 0.5) / se, 3),
            "passes_2se": bool(acc - 0.5 > 2 * se),
            "majority_baseline": round(maj, 6),
            "logit_var_min": round(min(logit_var), 8),
            "_correct": ok}


def eval_splits(model: SemModel, blob: dict, rows: list[dict], live: bool,
                device: str) -> dict:
    out = {}
    for i, s in enumerate(SPLITS):
        idx = torch.nonzero(blob["splits"] == i, as_tuple=False).squeeze(-1)
        assert len(idx) == EXPECTED_N[s], f"{s} n={len(idx)} ≠ {EXPECTED_N[s]}"
        e = evaluate(model, blob, idx, rows, live, device)
        assert abs(e["majority_baseline"] - 0.5) < 1e-9, \
            f"{s} 多数类基线不是 0.5000：{e['majority_baseline']}"
        out[s] = e
    adv_idx = torch.nonzero(blob["splits"] == 2, as_tuple=False).squeeze(-1)
    adv = out["adv"]
    correct = adv.pop("_correct")
    for code, name in ((1, "heldout_pair"), (2, "shifted_pos")):
        sub = adv_idx[blob["ctype"][adv_idx] == code]
        assert len(sub) == 1250, f"{name} n={len(sub)} ≠ 1250"
        c = correct[blob["ctype"][adv_idx] == code]
        acc = float(c.float().mean())
        out.setdefault("adv_by_ctype", {})[name] = {
            "n": 1250, "acc": round(acc, 6), "se": round(SE_1250, 6),
            "margin": round(acc - 0.5, 6),
            "margin_over_se": round((acc - 0.5) / SE_1250, 3),
            "passes_2se": bool(acc - 0.5 > TWO_SE)}
    for s in ("train", "test"):
        out[s].pop("_correct")
    return out


# ================= 老卡：step-0 对账（R4）与训练后（R5） =================
def eval_old_cards(enc, heads: dict, seed: int, device: str,
                   with_eval_task: bool = False) -> dict:
    """四张老卡逐样本 pred（完整发射）+ exact ⇒ sha256；同时给 exact_match。

    `with_eval_task=True` 时另跑一遍 `evaluate_task` 对账（自检：两口径 exact 必须相等）。
    """
    cards = old_cards()
    tok = NanoCharTokenizer()
    from probe import split_of
    out: dict = {}
    for n in OLD_CARDS:
        ev, _ = split_of(n, seed)
        loader = DataLoader(GenericTaskDataset(ev, tok, cards[n].spec),
                            batch_size=64, shuffle=False)
        was = (enc.training, heads[n].training)
        enc.eval()
        heads[n].eval()
        lines: list[str] = []
        exact_n = 0
        gi = 0
        with torch.no_grad():
            for batch in loader:
                inp = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                mem = enc(inp, attention_mask=mask)
                B = inp.shape[0]
                preds = _rollout(heads[n], mem, mask, B, cards[n].spec)
                for b in range(B):
                    got = preds[b]
                    truth = _truth_of(batch, b, cards[n].spec)
                    ex = int(sorted(got) == sorted(truth))
                    exact_n += ex
                    lines.append(f"{n}|{gi}|{got}|{truth}|{ex}")
                    gi += 1
        assert gi == len(ev), f"{n}: {gi} != {len(ev)}"
        blob = ("\n".join(lines) + "\n").encode("utf-8")
        rec = {"n": len(ev), "exact": exact_n / len(ev),
               "sha256": hashlib.sha256(blob).hexdigest(),
               "eval_task_exact": None}
        if with_eval_task:
            m = evaluate_task(enc, heads[n], loader, device, cards[n].spec)
            rec["eval_task_exact"] = m["exact_match"]
            assert abs(m["exact_match"] - rec["exact"]) < 1e-12, \
                f"{n}: 自写口径 exact {rec['exact']} ≠ evaluate_task {m['exact_match']}"
        if was[0]:
            enc.train()
        if was[1]:
            heads[n].train()
        out[n] = rec
    out["combined_sha256"] = hashlib.sha256(
        "".join(out[n]["sha256"] for n in OLD_CARDS).encode()).hexdigest()
    return out


def load_old_heads(seed: int, device: str) -> dict:
    """四张老卡头 ← capability_map/cards/*_frozen_s{seed}.pt（只读），并冻结。"""
    from dtseek.tasks.artifacts import build_card_decoder
    heads = {}
    for n in OLD_CARDS:
        p = ROOT / "experiments" / "capability_map" / "cards" / f"{n}_frozen_s{seed}.pt"
        if not p.exists():
            raise SystemExit(f"缺老卡：{p}")
        d, _ = build_card_decoder(read_card(p), device)
        for q in d.parameters():
            q.requires_grad_(False)
        d.eval()
        heads[n] = d
    return heads


def freeze_report_old(heads: dict) -> dict:
    rep = {}
    for n, d in heads.items():
        flags = sorted({bool(p.requires_grad) for p in d.parameters()})
        assert flags == [False], f"老卡头 {n} requires_grad={flags}（必须全 False）"
        rep[n] = {"params": sum(p.numel() for p in d.parameters()),
                  "requires_grad_flags": flags,
                  "training": d.training}
    return rep


_OLD_CARDS_OBJ = None


def old_cards():
    """resolve_tasks 每步调用 4 次太贵 ⇒ 进程内只解析一次。"""
    global _OLD_CARDS_OBJ
    if _OLD_CARDS_OBJ is None:
        _OLD_CARDS_OBJ = resolve_tasks(OLD_CARDS)
    return _OLD_CARDS_OBJ


# ================= 老任务数据与批次（joint 臂） =================
def old_raw(name: str) -> list[dict]:
    n = TASK_SAMPLES[name]
    p = CAPMAP_CACHE / f"{name}_{n}_20240927.pkl"
    if not p.exists():
        raise SystemExit(f"缺老任务数据（只读复用 capability_map）：{p}")
    return pickle.loads(p.read_bytes())


def split_val(data: list[dict], seed: int) -> tuple[list[dict], list[dict]]:
    import copy
    import random
    work = copy.deepcopy(data)
    random.Random(seed).shuffle(work)
    n_val = max(200, len(work) // 10)
    return work[:n_val], work[n_val:]


def old_loaders(seed: int, bs: int = BATCH) -> dict:
    """4 张老卡的训练 loader（各用自己的 generator，不碰全局 RNG）。"""
    cards = old_cards()
    tok = NanoCharTokenizer()
    loaders = {}
    for i, n in enumerate(OLD_CARDS):
        _, tr = split_val(old_raw(n), seed)
        g = torch.Generator().manual_seed(seed * 1000 + 7 + i)
        loaders[n] = DataLoader(GenericTaskDataset(tr, tok, cards[n].spec),
                                batch_size=bs, shuffle=True, drop_last=True, generator=g)
    return loaders


def old_loss(heads: dict, enc, batch, name: str, device: str) -> torch.Tensor:
    cards = old_cards()
    inp = batch["input_ids"].to(device)
    mask = batch["attention_mask"].to(device)
    mem = enc(inp, attention_mask=mask)
    return task_loss(heads[name], mem, mask, batch, cards[name].spec, device)


# ================= R6：核漂移 =================
def core_drift(enc) -> float:
    """‖core−base‖F / ‖base‖F（相对 checkpoints/base_encoder.pt::doc_encoder）。"""
    import math as _math
    ck = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
    base = ck["doc_encoder"]
    num = den = 0.0
    for k, p in enc.state_dict().items():
        a = p.detach().float().cpu()
        b = base[k].float()
        num += float(((a - b) ** 2).sum())
        den += float((b ** 2).sum())
    return _math.sqrt(num) / max(_math.sqrt(den), 1e-12)


def old_card_baseline(seed: int, device: str) -> dict:
    """R4 的「不接本任务」配置 B：核 ← base_encoder，四张老卡头 ← frozen 卡，无任何本任务部件。"""
    enc, _ = load_base_encoder(str(BASE_CKPT), device)
    enc.eval()
    heads = load_old_heads(seed, device)
    rep = eval_old_cards(enc, heads, seed, device, with_eval_task=True)
    # 与 capability_map 记录对账
    summary = json.loads(CAPMAP_SUMMARY.read_text(encoding="utf-8"))["rows"]
    for n in OLD_CARDS:
        rec = summary[n][str(seed)]["frozen_exact"]
        rep[n]["capmap_frozen_exact"] = rec
        rep[n]["delta_vs_capmap"] = rep[n]["exact"] - rec
    return rep
