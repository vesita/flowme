"""P19 fix_negation 共用：口径逐字复用既有实验，不自己造轮子。

出处（只读，不改任何既有目录）：
  * `experiments/e2e_d1/common.py` —— `split_of` / `truth_first` / `doc_sha` / `paired_delta` /
    `verdict_two_seed` / `anchor_key`（端到端 respond 口径，P16）；
  * `experiments/prod_card_audit/common.py` —— `split_of`（返回 train 侧）/ `make_loader` /
    `eval_persample`（`evaluate_task` 逐样本版）/ `lookup`（免费规则查表口径，P15）。
为避免在这些目录里落 `__pycache__`，本文件第一行即 `sys.dont_write_bytecode = True`。
"""
from __future__ import annotations

import os
import re
import sys

sys.dont_write_bytecode = True          # 不在别的实验目录里写 __pycache__
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "4")))

from dtseek.tasks.dialogue import CARD_TEMPLATES, DEFAULT_ATTACH, respond  # noqa: E402
from dtseek.tasks.engine import MultiTaskEngine  # noqa: E402

# ---- 既有实验口径（只读 import）------------------------------------------------

_E2E_DIR = ROOT / "experiments" / "e2e_d1"
_AUD_DIR = ROOT / "experiments" / "prod_card_audit"

import importlib.util  # noqa: E402


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


_e2e_common = _load(_E2E_DIR / "common.py", "fixneg_e2e_common")   # e2e_d1/common.py
_audit = _load(_AUD_DIR / "common.py", "fixneg_audit_common")       # prod_card_audit/common.py

split_of_e2e = _e2e_common.split_of                 # (cap, seed) -> list[dict]
truth_first = _e2e_common.truth_first
doc_sha = _e2e_common.doc_sha
paired_delta = _e2e_common.paired_delta
verdict_two_seed = _e2e_common.verdict_two_seed
anchor_key = _e2e_common.anchor_key

split_of = _audit.split_of                           # (cap, seed) -> (eval, train)
make_loader = _audit.make_loader
eval_persample = _audit.eval_persample
lookup = _audit.lookup

# ---- 常量 ----------------------------------------------------------------------

SEEDS = (42, 43)
CAPS = ("pronoun", "relation", "sentiment", "person", "negation")
NEG = "negation"
ONLINE_BASE = "checkpoints/base_encoder.pt"
ONLINE_DOC_SHA = "dc27db5337d16f185dd08cc50505390e53d03606b9d307457828f4ecbfdf6b1a"
OWN_DOC_SHA = "801657fb40a803a23d18950947cc9a772448628911adcc67b68374d8c5754671"
#: PREREG §0 写死的回退值（原 DEFAULT_ATTACH）
FALLBACK_CARD = "experiments/compose_ops/artifacts/cards/negation.pt"
FALLBACK_FILE_SHA = "ba1a80f9af81b1a2"
#: PREREG §2 写死的主候选（先评它；不达标才评 e12，避免在评测集上挑模型）
PRIMARY_CANDIDATE = "checkpoints/negation_accept_card.pt"
SECOND_CANDIDATE = "checkpoints/negation_accept_card_e12.pt"
#: F5 老卡噪声带（pt，取自 anchored_select/PREREG.md、additivity/PREREG.md、two_channel_head/REPORT.md W5）
OLD_BANDS = {"pronoun": 0.0283, "sentiment": 0.0041, "relation": 0.0139, "person": 0.0033}

_NEG_CLAUSE_RE = re.compile(r"并且有否定成分『[^』]*』")


# ---- 引擎 ----------------------------------------------------------------------

def build_engine(neg_path: str, base_path: str = ONLINE_BASE,
                 auto_attach: bool = True) -> MultiTaskEngine:
    """与 `examples/example_dialogue.py:54-56` 同构：线上核 + 默认四卡 + negation 卡。"""
    eng = MultiTaskEngine(base_path=str(ROOT / base_path), auto_attach=auto_attach)
    if auto_attach:
        eng.attach(ROOT / neg_path)
    else:
        for c in ("pronoun", "relation", "sentiment", "person"):
            eng.attach(ROOT / f"checkpoints/cards/{c}.pt")
        eng.attach(ROOT / neg_path)
    return eng


def run_respond(eng, texts: list[str]) -> list[dict]:
    out = []
    for t in texts:
        try:
            out.append(respond(eng, t, type_="plain"))
        except Exception as exc:  # noqa: BLE001 —— respond 本应 fail-closed；抛了要记下来
            out.append({"kind": "CRASH", "text": f"{type(exc).__name__}: {exc}",
                        "evidence": [], "cards_run": [], "plan": [], "terminal": "crash",
                        "reason": str(exc), "type": "plain"})
    return out


# ---- negation 三项指标（口径 = e2e_d1/w0_e2e.py::neg_metrics 的 rates 部分）-------

def neg_vectors(items: list[dict], recs: list[dict], class_map: dict[str, int]) -> dict:
    """返回 {'detect':[..], 'cls':[..], 'span':[..], 'confusion': {...}}（逐样本 0/1）。"""
    truths = [truth_first(it) for it in items]
    cores = [it["text"].strip() for it in items]
    real = [t["label"] > 0 for t in truths]
    v: dict[str, list[int]] = {"detect": [], "cls": [], "span": []}
    tp = fn = fp = tn = 0
    for i, rec in enumerate(recs):
        ev = [e for e in rec["evidence"] if e["card"] == NEG]
        has = bool(ev)
        v["detect"].append(int(has == real[i]))
        if real[i] and has:
            tp += 1
        elif real[i]:
            fn += 1
        elif has:
            fp += 1
        else:
            tn += 1
        if not real[i]:
            continue
        pred_id = class_map.get(ev[0]["class_name"], -1) if has else 0
        v["cls"].append(int(pred_id == truths[i]["label"]))
        if truths[i]["aligned"]:
            if has:
                s_, e_ = ev[0]["span"]
                v["span"].append(int(cores[i][s_:e_ + 1] == truths[i]["sub"]))
            else:
                v["span"].append(0)
    v["confusion"] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn}
    return v


def rate(vec: list[int]) -> float:
    return sum(vec) / max(1, len(vec))


def binom_se(p: float, n: int) -> float:
    import math
    return math.sqrt(max(p * (1 - p), 0.0) / max(1, n))


def neg_rate_summary(vec: dict) -> dict:
    return {m: {"rate": rate(vec[m]), "n": len(vec[m]),
                "se": binom_se(rate(vec[m]), len(vec[m]))}
            for m in ("detect", "cls", "span")}


# ---- F3 投影 -------------------------------------------------------------------

def project(rec: dict, strict: bool = False) -> dict:
    """PREREG §3 写死的投影：去掉 negation 卡证据及其派生文本子串。

    strict=True 时**不**动 text（披露项）。
    """
    ev = [e for e in rec.get("evidence") or [] if e["card"] != NEG]
    out = {"type": rec.get("type"), "kind": rec.get("kind"),
           "terminal": rec.get("terminal"), "reason": rec.get("reason"),
           "plan": rec.get("plan"), "cards_run": rec.get("cards_run"),
           "evidence": ev}
    text = rec.get("text") or ""
    out["text"] = text if strict else _NEG_CLAUSE_RE.sub("", text).replace("；。", "。")
    return out


def neg_clause(rec: dict) -> str:
    m = _NEG_CLAUSE_RE.search(rec.get("text") or "")
    return m.group(0) if m else ""
