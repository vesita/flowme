"""P16 e2e_d1 共用：口径（与 `capability_map` / `prod_card_audit` 逐字同公式）+ sha + 配对 Δ。

只读 src/ 与其它实验目录；只写本目录。
"""
from __future__ import annotations

import copy
import hashlib
import os
import pickle
import random
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
CAPMAP = ROOT / "experiments" / "capability_map"
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

CACHE = CAPMAP / "cache"
SEEDS = (42, 43)
CAPS = ("pronoun", "relation", "sentiment", "person", "negation")
EVAL_FILES = [f"{c}_ordered_s{s}.pkl" for c in CAPS for s in SEEDS]

E_BASES = {
    "A_online": "checkpoints/base_encoder.pt",                       # 引擎 A：线上核（在产默认）
    "B_own": "experiments/compose_ops/artifacts/base_encoder.pt",    # 引擎 B：negation 卡来源核
}


def split_of(cap: str, seed: int) -> list[dict]:
    """与 `capability_map/probe.py::split_of` / `prod_card_audit/common.py` 逐字相同。"""
    ordered = pickle.loads((CACHE / f"{cap}_ordered_s{seed}.pkl").read_bytes())
    data = copy.deepcopy(ordered)
    random.Random(seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    return data[:n_val]


def strip_offset(text: str) -> int:
    """`respond` 内部 `source = text.strip()` ⇒ span 偏移相对**去空白后**的文本。"""
    return len(text) - len(text.lstrip())


def truth_first(item: dict) -> dict:
    """真值首切片（与 `GenericTaskDataset.__getitem__` 同序：按 start 升序）。

    返回 {label, sub, s, e, aligned}：`sub` = 原文逐字子串（内容口径，避开 strip 偏移）；
    `s/e` = 已换算到 strip 后坐标系的区间（`aligned=False` 表示换算越界，span 指标跳过）。
    无 span ⇒ label=0。
    """
    text = item["text"]
    off = strip_offset(text)
    core = text.strip()
    spans = sorted(item.get("spans") or [], key=lambda x: x["start"])
    if not spans:
        return {"label": 0, "sub": "", "s": -1, "e": -1, "aligned": True}
    sp = spans[0]
    st, en = int(sp["start"]) - off, int(sp["end"]) - off
    aligned = 0 <= st <= en <= len(core)
    return {"label": int(sp["label"]), "sub": text[int(sp["start"]):int(sp["end"])],
            "s": st, "e": en, "aligned": aligned}


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_sha(sd: dict) -> str:
    """按 state_dict 原始键序拼接张量字节 ⇒ sha256（与 A1 / CATALOG 同一方案）。"""
    h = hashlib.sha256()
    for k in sd:                      # 保持插入序，不排序
        t = sd[k].detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def paired_delta(a: list[int], b: list[int]) -> dict:
    """配对 Δ = mean(b − a)，SE = std(b−a, ddof=1)/√n（与 A0 / free_rule_floor F1 同一套）。"""
    assert len(a) == len(b), (len(a), len(b))
    ds = [bb - aa for aa, bb in zip(a, b)]
    n = len(ds)
    if n == 0:
        return {"n": 0, "delta": float("nan"), "se": float("nan"), "significant": False}
    m = sum(ds) / n
    if n > 1:
        var = sum((d - m) ** 2 for d in ds) / (n - 1)
        se = (var ** 0.5) / (n ** 0.5)
    else:
        se = 0.0
    return {"n": n, "delta": m, "se": se, "z": (m / se) if se > 0 else (0.0 if m == 0 else float("inf")),
            "significant": bool(abs(m) > 2 * se)}


def verdict_two_seed(d42: dict, d43: dict) -> str:
    """E1（写死）：两 seed 同号且 |Δ| > 2SE（两个 seed 都要）。"""
    m1, m2 = d42["delta"], d43["delta"]
    sig = d42["significant"] and d43["significant"]
    same_neg = m1 < 0 and m2 < 0
    same_pos = m1 > 0 and m2 > 0
    if sig and same_neg:
        return "显著变差"
    if sig and same_pos:
        return "显著变好"
    return "未测出显著差异"


def neg_evidence(rec: dict) -> list[dict]:
    return [e for e in rec.get("evidence") or [] if e["card"] == "negation"]


def mood_evidence(rec: dict) -> list[dict]:
    return [e for e in rec.get("evidence") or [] if e["card"] == "sentiment"]


def anchor_key(rec: dict, card: str, src_text: str) -> tuple:
    """一张卡在回复里的首发射指纹：(逐字子串, class_name)；没发射 = ()。

    span 落在**输入**上（`respond` 的 `source`），所以子串必须从输入取，
    不能从 `rec["text"]`（那是回复）取。
    """
    ev = [e for e in rec.get("evidence") or [] if e["card"] == card]
    if not ev:
        return ()
    e = ev[0]
    s, t = e["span"]
    return (src_text[s:t + 1], e["class_name"])
