"""成语卡：只读复用 `experiments/cumulative_add/cards/ctrl_idiom_s42.pt`。

它**不是** v1 任务卡产物（是训练时的一体 ckpt：`doc_encoder` + `decoders` + `task_specs`），
`engine.attach()` 会直接拒（`format=None`）。这里按 `artifacts.save_base / save_card`
的口径把它拆成一对产物落在本目录 `cache/` 下，再挂进一个**只带它自己那张卡**的引擎。

为什么必须是**独立引擎**：这份 ckpt 的 `doc_encoder` 是 `cumulative_add` 那次训练自己的
（与 `checkpoints/base_encoder.pt` 最大绝对差 **0.0438**，同结构不同权重）。实测
（`logs/` 的两次探针）把同一张卡换到默认基座上，40 句富集样本里 18 句的锚点会变，
10 句理想成语句的定域也更差 ⇒ 卡和编码器必须成对用。锚点是**原文上的区间**，
与编码器无关，所以两个引擎的锚点可以并进同一个袋。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from dtseek.tasks.artifacts import BASE_FORMAT, CARD_FORMAT, save_base  # noqa: E402
from dtseek.tasks.engine import DEFAULT_BASE, MultiTaskEngine  # noqa: E402

CARD_SRC = ROOT / "experiments" / "cumulative_add" / "cards" / "ctrl_idiom_s42.pt"
CACHE = HERE / "cache"
BASE_OUT = CACHE / "idiom_base.pt"
CARD_OUT = CACHE / "idiom_card.pt"
#: 只放它自己那张卡：这些卡是配默认基座训的，挂到成语基座上只查 hidden_dim/format，
#: 会**静默通过**（见 `check_card_base_compat`），所以干脆给一个不存在的目录。
NO_BUILTIN = CACHE / "no_builtin_cards_here"


def ensure_artifacts() -> tuple[Path, Path, dict]:
    """按 ckpt 现场重建一对产物（源只读）。已存在且比源新 ⇒ 直接复用。"""
    CACHE.mkdir(parents=True, exist_ok=True)
    src_mtime = CARD_SRC.stat().st_mtime
    info = {}
    if BASE_OUT.exists() and CARD_OUT.exists() \
            and BASE_OUT.stat().st_mtime > src_mtime and CARD_OUT.stat().st_mtime > src_mtime:
        return BASE_OUT, CARD_OUT, {"rebuilt": False}

    ck = torch.load(CARD_SRC, map_location="cpu", weights_only=False)
    if "doc_encoder" not in ck or "idiom" not in ck.get("decoders", {}):
        raise SystemExit(f"{CARD_SRC} 里没有 doc_encoder/decoders[idiom]：fail-closed")

    from dtseek.encoder.nano_doc_encoder import NanoDocEncoder

    vocab = ck["doc_encoder"]["embedding.weight"].shape[0]
    enc = NanoDocEncoder(vocab_size=vocab, hidden_dim=ck["hidden_dim"], dropout=0.0,
                         **ck["encoder_kwargs"])
    enc.load_state_dict(ck["doc_encoder"])
    save_base(BASE_OUT, enc, hidden_dim=ck["hidden_dim"], vocab_size=vocab,
              encoder_kwargs=ck["encoder_kwargs"],
              meta={"from": str(CARD_SRC.relative_to(ROOT)), "note": "成语卡自带基座，非默认基座"})

    torch.save({
        "format": CARD_FORMAT,
        "task": "idiom",
        "spec": ck["task_specs"]["idiom"],
        "decoder": ck["decoders"]["idiom"],
        "hidden_dim": ck["hidden_dim"],
        "decoder_kwargs": ck["decoder_kwargs"],
        "base_format": BASE_FORMAT,
        "train_args": ck.get("train_args", {}),
        "extra": {},
    }, CARD_OUT)

    base_ck = torch.load(DEFAULT_BASE, map_location="cpu", weights_only=False)
    md = max(float((a.float() - b.float()).abs().max())
             for a, b in zip(ck["doc_encoder"].values(), base_ck["doc_encoder"].values()))
    info = {"rebuilt": True, "vocab": vocab, "hidden_dim": ck["hidden_dim"],
            "max_abs_diff_vs_default_base": round(md, 6),
            "task_order": ck.get("task_order"), "built_at": time.strftime("%F %T")}
    (CACHE / "idiom_card_info.json").write_text(
        __import__("json").dumps(info, ensure_ascii=False, indent=1), encoding="utf-8")
    return BASE_OUT, CARD_OUT, info


def make_idiom_engine():
    """只带成语卡、用它**自己那张基座**的引擎。"""
    base_path, card_path, info = ensure_artifacts()
    engine = MultiTaskEngine(base_path=str(base_path), cards_dir=str(NO_BUILTIN))
    name = engine.attach(card_path)
    if sorted(engine.decoders) != ["idiom"]:
        raise SystemExit(f"成语引擎挂了意外的卡：{sorted(engine.decoders)}（fail-closed）")
    return engine, name, info
