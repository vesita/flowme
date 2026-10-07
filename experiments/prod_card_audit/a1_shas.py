#!/usr/bin/env python3
"""A1 —— 把「覆盖缺口」钉成事实：打印两侧实际加载的基座 sha256。

用法：
    CUDA_VISIBLE_DEVICES="" uv run python experiments/prod_card_audit/a1_shas.py

只读 src/ checkpoints/ experiments/；只写 stdout 与 results/a1_shas.json（本目录）。
sha 方案（跑前写死，见 PREREG §1）：对 state_dict **按原始键序**依次拼接每个张量的
CPU 连续字节，取 sha256。自校验：线上基座必须复现 CATALOG.md:11 的 dc27db5337d16f18…。
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from dtseek.tasks.artifacts import read_base  # noqa: E402
from dtseek.tasks.dialogue import DEFAULT_ATTACH  # noqa: E402
from dtseek.tasks.engine import DEFAULT_BASE, MultiTaskEngine  # noqa: E402

HERE = Path(__file__).resolve().parent
EXPECT_ONLINE = "dc27db5337d16f185dd08cc50505390e53d03606b9d307457828f4ecbfdf6b1a"


def file_sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_sha(sd: dict) -> str:
    """按 state_dict 原始键序拼接张量字节 ⇒ sha256（与 CATALOG 同一方案）。"""
    h = hashlib.sha256()
    for k in sd:                      # 保持插入序，不排序
        t = sd[k].detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def base_entry(path: str | Path) -> dict:
    p = str(path)
    ck = read_base(p)
    return {"path": p, "file_sha256": file_sha(p), "doc_sha256": doc_sha(ck["doc_encoder"])}


def main() -> int:
    out: dict = {"prereg": "experiments/prod_card_audit/PREREG.md", "items": {}}

    # ---- 1. 源码字面值（不引报告，直接 import 打印）----
    src = {
        "dialogue.DEFAULT_ATTACH": list(DEFAULT_ATTACH),
        "engine.DEFAULT_BASE": DEFAULT_BASE,
        "test_dialogue.py:111-116 fixture": "MultiTaskEngine(); for path in DEFAULT_ATTACH: eng.attach(path)",
    }
    out["src_literals"] = src
    print("[src] dialogue.DEFAULT_ATTACH =", src["dialogue.DEFAULT_ATTACH"])
    print("[src] engine.DEFAULT_BASE     =", DEFAULT_BASE)

    # ---- 2. 线上默认路径实际加载的基座 ----
    online = base_entry(ROOT / DEFAULT_BASE)
    out["items"]["online_default_base"] = online
    print(f"[online] {online['path']}")
    print(f"         file_sha256 = {online['file_sha256']}")
    print(f"         doc_sha256  = {online['doc_sha256']}")

    # ---- 3. DEFAULT_ATTACH 那张卡：文件 sha + 来源核 ----
    for rel in DEFAULT_ATTACH:
        card_path = ROOT / rel
        raw = torch.load(card_path, map_location="cpu", weights_only=False)
        ta = raw.get("train_args") or {}
        entry = {
            "path": str(card_path.relative_to(ROOT)),
            "file_sha256": file_sha(card_path),
            "format": raw.get("format"),
            "task": raw.get("task"),
            "train_args.split_from": ta.get("split_from"),
            "has_embedded_doc_encoder": "doc_encoder" in raw,
        }
        print(f"[attach] {entry['path']}")
        print(f"         file_sha256 = {entry['file_sha256']}")
        print(f"         format={entry['format']} task={entry['task']}")
        print(f"         train_args.split_from = {entry['train_args.split_from']}")
        print(f"         自带 doc_encoder = {entry['has_embedded_doc_encoder']}")

        # 沿 split_from 解析来源核
        sf = ta.get("split_from")
        if sf:
            sfck = torch.load(ROOT / sf, map_location="cpu", weights_only=False)
            entry["split_from_path"] = sf
            entry["split_from_doc_sha256"] = doc_sha(sfck["doc_encoder"])
            print(f"         来源核 {sf}")
            print(f"                   doc_sha256 = {entry['split_from_doc_sha256']}")
        art = ROOT / "experiments/compose_ops/artifacts/base_encoder.pt"
        a = base_entry(art)
        entry["artifacts_base"] = a
        print(f"[artifacts] {a['path']}  doc_sha256 = {a['doc_sha256']}")
        out["items"]["default_attach_card"] = entry

    # ---- 4. 测试 fixture 实际加载的基座（照 tests/test_dialogue.py 的写法跑一遍）----
    eng = MultiTaskEngine()
    test_side = {
        "engine_base_path": eng.base_path,
        "engine_doc_sha256": doc_sha(eng._base["doc_encoder"]),
        "attached": sorted(eng.attached),
        "card_paths": {k: v for k, v in sorted(eng.card_paths.items())},
    }
    print(f"[test-fixture] MultiTaskEngine().base_path = {test_side['engine_base_path']}")
    print(f"               doc_sha256 = {test_side['engine_doc_sha256']}")
    print(f"               attached   = {test_side['attached']}")
    out["items"]["test_fixture_base"] = test_side
    del eng

    # ---- 5. 四默认卡的 split_from 核（PREREG §2 对照臂自检）----
    four = {}
    for name in ("pronoun", "relation", "sentiment", "person"):
        p = ROOT / f"checkpoints/cards/{name}.pt"
        raw = torch.load(p, map_location="cpu", weights_only=False)
        sf = (raw.get("train_args") or {}).get("split_from")
        rec = {"path": f"checkpoints/cards/{name}.pt", "file_sha256": file_sha(p),
               "split_from": sf, "has_embedded_doc_encoder": "doc_encoder" in raw}
        if sf and (ROOT / sf).exists():
            sfr = torch.load(ROOT / sf, map_location="cpu", weights_only=False)
            if "doc_encoder" in sfr:
                rec["split_from_doc_sha256"] = doc_sha(sfr["doc_encoder"])
        four[name] = rec
        print(f"[four] {name}: split_from={sf} doc_sha={rec.get('split_from_doc_sha256')}")
    out["items"]["default_four_cards"] = four

    # multitask_v2 与线上是否逐位相同（复核 CATALOG §5-4）
    mv = ROOT / "checkpoints/multitask_v2_dtseek.pt"
    if mv.exists():
        mvr = torch.load(mv, map_location="cpu", weights_only=False)
        mv_sha = doc_sha(mvr["doc_encoder"])
        diff, mx = 0, 0.0
        a = read_base(ROOT / DEFAULT_BASE)["doc_encoder"]
        b = mvr["doc_encoder"]
        for k in a:
            d = (a[k].float() - b[k].float()).abs().max().item()
            mx = max(mx, d)
            if not torch.equal(a[k], b[k]):
                diff += 1
        out["items"]["multitask_v2"] = {"doc_sha256": mv_sha, "neq_tensors_vs_online": diff,
                                        "max_abs_delta": mx, "same_sha_as_online": mv_sha == online["doc_sha256"]}
        print(f"[multitask_v2] doc_sha256={mv_sha} neq={diff} max|Δ|={mx} "
              f"same_as_online={mv_sha == online['doc_sha256']}")

    # ---- 6. 判定（用打印结果说话）----
    online_sha = online["doc_sha256"]
    test_sha = test_side["engine_doc_sha256"]
    attach_card = out["items"]["default_attach_card"]
    src_sha = attach_card.get("split_from_doc_sha256")
    verdict = {
        "online_base_doc_sha256": online_sha,
        "test_fixture_base_doc_sha256": test_sha,
        "online_vs_test_same": online_sha == test_sha,
        "card_provenance_core_doc_sha256": src_sha,
        "card_provenance_vs_online_same": src_sha == online_sha,
        "gap": ("测试侧基座 == 线上基座，但 DEFAULT_ATTACH 卡的来源核 ≠ 线上基座"
                if (online_sha == test_sha and src_sha != online_sha) else "见打印"),
        "selfcheck_online_matches_catalog": online_sha == EXPECT_ONLINE,
    }
    out["verdict"] = verdict
    print("\n[VERDICT]")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))

    res = HERE / "results" / "a1_shas.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    res.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {res}")
    print("A1_DONE")
    ok = verdict["selfcheck_online_matches_catalog"]
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
