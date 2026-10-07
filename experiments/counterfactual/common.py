"""共用件：引擎装配（纯 CPU）、单次前向、输出身份口径、样本入口。

只读 import `experiments/card_flow/*` 与 `src/dtseek/*`；本目录是唯一写入点。
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CARD_FLOW = ROOT / "experiments" / "card_flow"
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(CARD_FLOW))

PREREG = HERE / "PREREG.md"
RESULTS = HERE / "results"
LOGS = HERE / "logs"
RESULTS.mkdir(exist_ok=True)
LOGS.mkdir(exist_ok=True)

SEED0 = 20261007
MAX_CAND = 30          # 每 (样本, 族) 搜索预算 K（PREREG §3 写死）


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, path
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# 只读加载 card_flow 的五段流与样本入口
FLOW = _load(CARD_FLOW / "flow.py", "cf10_flow_ro")
PROP = _load(CARD_FLOW / "proposers.py", "cf10_proposers_ro")
RCF = _load(CARD_FLOW / "run_card_flow.py", "cf10_runcardflow_ro")

run_flow = FLOW.run_flow
real_propose = PROP.real_propose
load_samples = RCF.load_samples

ENGINE = None
CARDS: list[str] = []
NEGATTACHED = False


def build_engine():
    """显式锁 CPU（PREREG §0 硬约束），挂否定卡（失败则 4 张卡并如实记）。"""
    global ENGINE, CARDS, NEGATTACHED
    if ENGINE is not None:
        return ENGINE
    from dtseek.tasks.engine import MultiTaskEngine
    ENGINE = MultiTaskEngine(device="cpu")
    try:
        ENGINE.attach(ROOT / "checkpoints" / "negation_accept_card.pt")
        NEGATTACHED = True
    except Exception as exc:                                   # noqa: BLE001
        NEGATTACHED = False
        print(f"[warn] 否定卡挂载失败，本轮用 {len(ENGINE.attached)} 张卡: "
              f"{type(exc).__name__}: {exc}", flush=True)
    CARDS = sorted(ENGINE.attached)
    return ENGINE


def run_once(text: str) -> dict:
    """真跑一次端到端流，返回**输出身份**记录（主口径 Y）。"""
    eng = build_engine()
    cands, class_index = real_propose(text, eng, CARDS)
    rec = run_flow(text, cands, class_index)
    if rec["status"] == "ok":
        return {"y": ["ok", rec["text"]], "ch": "ok",
                "skeleton": rec.get("skeleton"), "n_valid": rec.get("n_valid")}
    return {"y": ["reject", rec.get("stage"), rec.get("reason")], "ch": "reject",
            "reason": rec.get("reason"), "stage": rec.get("stage"),
            "n_valid": rec.get("n_valid")}


def ykey(out: dict) -> tuple:
    """主口径可哈希键。"""
    return tuple(out["y"])
