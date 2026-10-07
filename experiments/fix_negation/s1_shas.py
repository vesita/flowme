#!/usr/bin/env python3
"""S1 —— F0 sha 自校验 + 全树 negation 卡扫描（PREREG §2 步骤 1）。

用法：
    CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1 \
      uv run python experiments/fix_negation/s1_shas.py

只读 src/ checkpoints/ experiments/ training/；只写 stdout 与 results/s1_shas.json（本目录）。
sha 方案 = `experiments/prod_card_audit/a1_shas.py::doc_sha`（state_dict 原始键序拼张量字节）。
F0：线上核必须复现 `dc27db5337d16f18…`，否则 exit 3。
"""
from __future__ import annotations

import hashlib
import json
import sys

sys.dont_write_bytecode = True
from pathlib import Path  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from dtseek.tasks.artifacts import read_base  # noqa: E402
from dtseek.tasks.dialogue import DEFAULT_ATTACH  # noqa: E402
from dtseek.tasks.engine import DEFAULT_BASE  # noqa: E402

EXPECT_ONLINE = "dc27db5337d16f185dd08cc50505390e53d03606b9d307457828f4ecbfdf6b1a"
OWN = "801657fb40a803a23d18950947cc9a772448628911adcc67b68374d8c5754671"


def file_sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_sha(sd: dict) -> str:
    """a1_shas.py::doc_sha 同一方案：按原始键序拼接张量 CPU 连续字节 ⇒ sha256。"""
    h = hashlib.sha256()
    for k in sd:
        t = sd[k].detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def core_of(rel: str) -> str | None:
    """给一个核文件相对路径 ⇒ doc sha；文件缺/无 doc_encoder ⇒ None。"""
    p = ROOT / rel
    if not p.exists():
        return None
    r = torch.load(p, map_location="cpu", weights_only=False)
    return doc_sha(r["doc_encoder"]) if "doc_encoder" in r else None


def main() -> int:
    out: dict = {"prereg": "experiments/fix_negation/PREREG.md", "sha_scheme":
                 "prod_card_audit/a1_shas.py::doc_sha（state_dict 原始键序拼张量字节）"}

    # ---- F0 自校验：先证明本脚本能复现 CATALOG 的线上核 sha ----
    online = core_of(DEFAULT_BASE)
    out["online"] = {"path": DEFAULT_BASE, "doc_sha256": online, "file_sha256": file_sha(ROOT / DEFAULT_BASE)}
    selfcheck = online == EXPECT_ONLINE
    out["f0_selfcheck"] = {"expect": EXPECT_ONLINE, "got": online, "pass": selfcheck}
    print(f"[F0] {DEFAULT_BASE}\n     doc_sha256 = {online}\n     复现 dc27db53… = {selfcheck}")
    if not selfcheck:
        print("F0_FAIL")
        return 3

    # ---- 现默认卡与其来源核 ----
    cur = DEFAULT_ATTACH[0]
    cur_raw = torch.load(ROOT / cur, map_location="cpu", weights_only=False)
    cur_ta = cur_raw.get("train_args") or {}
    cur_prov = cur_ta.get("split_from") or cur_ta.get("base")
    cur_sha = core_of(cur_prov) if cur_prov else None
    out["current_default"] = {"path": cur, "file_sha256": file_sha(ROOT / cur),
                              "prov_how": "split_from" if cur_ta.get("split_from") else "base",
                              "prov_src": cur_prov, "prov_doc_sha256": cur_sha,
                              "prov_is_online": cur_sha == EXPECT_ONLINE}
    print(f"[current] {cur}\n          file_sha256 = {out['current_default']['file_sha256']}\n"
          f"          来源核 {cur_prov} doc_sha256 = {cur_sha}  线上核? {out['current_default']['prov_is_online']}")

    # ---- 扫全树：所有 task == "negation" 的 .pt ----
    files = sorted(set(
        [str(p) for p in (ROOT / "checkpoints").rglob("*.pt")]
        + [str(p) for p in (ROOT / "experiments").rglob("*.pt")]
        + [str(p) for p in (ROOT / "training").rglob("*.pt")]))
    cards = []
    n_loaded = 0
    for f in files:
        rel = str(Path(f).relative_to(ROOT))
        try:
            r = torch.load(f, map_location="cpu", weights_only=False)
        except Exception:  # noqa: BLE001
            continue
        n_loaded += 1
        if r.get("task") != "negation":
            continue
        ta = r.get("train_args") or {}
        how = "train_args.split_from" if ta.get("split_from") else (
              "train_args.base" if ta.get("base") else None)
        prov = ta.get("split_from") or ta.get("base")
        sha = core_of(prov) if prov else None
        spec = r.get("spec") or {}
        cards.append({
            "path": rel, "file_sha256": file_sha(f), "format": r.get("format"),
            "epochs": ta.get("epochs"), "seed": ta.get("seed"),
            "prov_how": how, "prov_src": prov, "prov_doc_sha256": sha,
            "prov_is_online": sha == EXPECT_ONLINE, "prov_is_own": sha == OWN,
            "classes": [c.get("name") if isinstance(c, dict) else str(c)
                        for c in (spec.get("classes") or [])],
            "hidden_dim": r.get("hidden_dim"), "extra_keys": sorted((r.get("extra") or {}).keys()),
            "split_from": ta.get("split_from"), "base": ta.get("base"),
        })
        print(f"[card] {rel}\n        file={cards[-1]['file_sha256'][:16]} "
              f"via={how} src={prov} doc_sha={(sha or 'NONE')[:16]} "
              f"online={cards[-1]['prov_is_online']} epochs={ta.get('epochs')} "
              f"classes={cards[-1]['classes']}")

    out["negation_cards"] = cards
    out["n_pt_loaded"] = n_loaded
    out["verdict"] = {
        "n_negation_cards": len(cards),
        "n_online_core": sum(1 for c in cards if c["prov_is_online"]),
        "online_core_cards": [c["path"] for c in cards if c["prov_is_online"]],
        "current_is_online": out["current_default"]["prov_is_online"],
    }
    print(f"\n[VERDICT] negation 卡 {len(cards)} 张；线上核来源 {out['verdict']['n_online_core']} 张 → "
          f"{out['verdict']['online_core_cards']}；现默认卡是否线上核来源 = "
          f"{out['current_default']['prov_is_online']}")
    res = HERE / "results" / "s1_shas.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {res}\nS1_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
