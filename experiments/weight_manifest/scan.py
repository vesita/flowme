#!/usr/bin/env python3
"""P18 —— 全仓 .pt 权重清单扫描（只读）。

用法（仓库根 DTSeek/ 下）：
    CUDA_VISIBLE_DEVICES="" uv run python experiments/weight_manifest/scan.py

只读扫描 *.pt；只写本目录（experiments/weight_manifest/）三个产物：
    manifest.json / manifest.sha256 / MANIFEST.md
sha 方案照 experiments/prod_card_audit/a1_shas.py：
    doc_sha256 = state_dict 按**原始键序**拼接每个张量的 CPU 连续字节 ⇒ sha256。
自校验（M0）：必须复现线上基座 dc27db5337d16f18…，否则 exit 3。
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
EXPECT_ONLINE = "dc27db5337d16f185dd08cc50505390e53d03606b9d307457828f4ecbfdf6b1a"
ONLINE_BASE = ROOT / "checkpoints/base_encoder.pt"


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fp:
        for chunk in iter(lambda: fp.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def doc_sha(sd: dict) -> str:
    h = hashlib.sha256()
    for k in sd:                      # 插入序，不排序
        t = sd[k].detach().cpu().contiguous()
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def diff_vs(a: dict, b: dict):
    """与线上基座逐张量比对 ⇒ (共同键数, neq, max|Δ|)；键集不同则 neq=不适用。"""
    ka, kb = set(a), set(b)
    common = sorted(ka & kb)
    neq, mx = 0, 0.0
    for k in common:
        ta, tb = a[k].float(), b[k].float()
        if ta.shape != tb.shape:
            neq += 1
            continue
        d = (ta - tb).abs().max().item()
        mx = max(mx, d)
        if not torch.equal(a[k], b[k]):
            neq += 1
    return len(common), neq, mx, sorted(ka ^ kb)


def main() -> int:
    t0 = time.time()
    # ---- M0 自校验 ----
    online_sd = torch.load(ONLINE_BASE, map_location="cpu", weights_only=False)
    online_doc_sha = doc_sha(online_sd["doc_encoder"])
    m0_ok = online_doc_sha == EXPECT_ONLINE
    print(f"[M0] checkpoints/base_encoder.pt doc_sha256 = {online_doc_sha}")
    print(f"[M0] 期望                              = {EXPECT_ONLINE}")
    print(f"[M0] 复现 = {m0_ok}")
    if not m0_ok:
        print("M0_FAIL: sha 写法与线上不一致，先修脚本再继续", file=sys.stderr)
        return 3
    online_enc = online_sd["doc_encoder"]
    del online_sd

    # ---- 扫描 ----
    paths = sorted(
        p for p in ROOT.rglob("*.pt")
        if ".venv" not in p.parts and p.is_file()
    )
    print(f"[scan] 扫描到 .pt = {len(paths)}")

    # 既有 sha 记录出处（用于「无任何校验和有记录」判据）
    legacy_json = ROOT / "experiments/prod_card_audit/results/a1_shas.json"
    legacy = json.loads(legacy_text := legacy_json.read_text(encoding="utf-8")) if legacy_json.exists() else {}
    legacy_paths = set()
    def walk(o):
        if isinstance(o, dict):
            for k, v in o.items():
                if k in ("path", "engine_base_path") and isinstance(v, str):
                    legacy_paths.add(v)
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)
    walk(legacy)
    del legacy_text
    # scan_raw 的 kind 分类（沿用，别重复造分类逻辑）
    raw = json.loads((ROOT / "experiments/card_catalog/scan_raw.json").read_text(encoding="utf-8"))
    raw_by_path = {r["path"]: r for r in raw}

    entries = []
    for p in paths:
        rel = str(p.relative_to(ROOT))
        e = {
            "path": rel,
            "bytes": p.stat().st_size,
            "file_sha256": file_sha(p),
            "in_cache_dir": "/cache/" in rel,
        }
        try:
            ck = torch.load(p, map_location="cpu", weights_only=False)
            e["format"] = ck.get("format") if isinstance(ck, dict) else type(ck).__name__
            e["task"] = ck.get("task") if isinstance(ck, dict) else None
            ta = (ck.get("train_args") or {}) if isinstance(ck, dict) else {}
            e["train_args.base"] = ta.get("base")
            e["train_args.split_from"] = ta.get("split_from")
            sd = ck.get("doc_encoder") if isinstance(ck, dict) else None
            if isinstance(sd, dict) and sd:
                e["has_doc_encoder"] = True
                e["doc_sha256"] = doc_sha(sd)
                e["same_as_online_base"] = e["doc_sha256"] == online_doc_sha
                n_common, neq, mx, sym = diff_vs(sd, online_enc)
                e["n_common_tensors"] = n_common
                e["neq_vs_online"] = neq
                e["max_abs_delta_vs_online"] = round(mx, 6)
                e["key_symdiff"] = sym[:6]
            else:
                e["has_doc_encoder"] = False
                e["doc_sha256"] = None
                e["same_as_online_base"] = None
            del sd, ck
        except Exception as exc:                      # noqa: BLE001
            e["load_error"] = f"{type(exc).__name__}: {exc}"
            e["has_doc_encoder"] = False
            e["doc_sha256"] = None
            e["same_as_online_base"] = None
        raw_rec = raw_by_path.get(rel)
        e["catalog_kind"] = raw_rec["kind"] if raw_rec else None
        e["catalog_neq"] = raw_rec.get("neq") if raw_rec else None
        e["catalog_maxd"] = raw_rec.get("maxd") if raw_rec else None
        e["catalog_enc_eq"] = raw_rec.get("enc_eq") if raw_rec else None
        e["legacy_sha_record"] = rel in legacy_paths
        entries.append(e)
        print(f"  [{len(entries):3d}/{len(paths)}] {rel} "
              f"doc={str(e.get('doc_sha256'))[:16]} same={e['same_as_online_base']}", flush=True)

    # ---- 汇总 ----
    same = [e for e in entries if e["same_as_online_base"] is True]
    diff = [e for e in entries if e["same_as_online_base"] is False]
    na = [e for e in entries if e["same_as_online_base"] is None]
    load_err = [e for e in entries if e.get("load_error")]

    # 物理副本：同 file_sha256 出现在 ≥2 个路径
    by_sha: dict[str, list[str]] = {}
    for e in entries:
        by_sha.setdefault(e["file_sha256"], []).append(e["path"])
    dup_groups = {s: ps for s, ps in by_sha.items() if len(ps) > 1}
    for e in entries:
        e["physical_copies"] = len(by_sha[e["file_sha256"]])

    # 引用证据：全仓 .py/.sh/.md 正文里出现该 .pt 文件名（排除本目录与权重文件自身）
    ref_files = [p for p in ROOT.rglob("*")
                 if p.suffix in (".py", ".sh", ".md", ".json", ".yaml", ".toml")
                 and ".venv" not in p.parts and HERE not in p.parents and p.is_file()]
    texts = {}
    for p in ref_files:
        try:
            texts[str(p.relative_to(ROOT))] = p.read_text(encoding="utf-8", errors="ignore")
        except Exception:                              # noqa: BLE001
            pass
    for e in entries:
        base = Path(e["path"]).name
        hits = sorted({fp for fp, tx in texts.items() if base in tx})
        e["referenced_by"] = hits
        e["ref_count"] = len(hits)
        e["ref_experiments"] = sorted({fp.split("/")[1] if fp.startswith("experiments/") and len(fp.split("/")) > 1
                                       else (fp.split("/")[0] if "/" in fp else fp) for fp in hits})

    summary = {
        "n_pt_scanned": len(paths),
        "n_pt_manifested": len(entries),
        "total_bytes": sum(e["bytes"] for e in entries),
        "same_as_online": len(same),
        "diff_from_online": len(diff),
        "not_applicable": len(na),
        "load_errors": len(load_err),
        "load_error_paths": [e["path"] for e in load_err],
        "in_cache_dir": sum(1 for e in entries if e["in_cache_dir"]),
        "duplicate_sha_groups": len(dup_groups),
        "files_with_physical_copy": sum(1 for e in entries if e["physical_copies"] > 1),
        "unique_sha_files": sum(1 for e in entries if e["physical_copies"] == 1),
        "online_doc_sha256": online_doc_sha,
        "elapsed_sec": round(time.time() - t0, 1),
    }

    # ---- 落盘 ----
    manifest = {
        "generator": "experiments/weight_manifest/scan.py",
        "sha_scheme": "doc_sha256 = state_dict 原始键序拼接张量 CPU 连续字节 ⇒ sha256（同 prod_card_audit/a1_shas.py）",
        "online_base": {"path": "checkpoints/base_encoder.pt", "doc_sha256": online_doc_sha},
        "m0_selfcheck_pass": m0_ok,
        "summary": summary,
        "entries": entries,
    }
    (HERE / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(HERE / "manifest.sha256", "w", encoding="utf-8") as fp:
        for e in entries:
            fp.write(f"{e['file_sha256']}  {e['path']}\n")

    # MANIFEST.md（按目录分组，人读）
    groups: dict[str, list[dict]] = {}
    for e in entries:
        d = str(Path(e["path"]).parent)
        groups.setdefault(d, []).append(e)
    out = ["# 全仓 .pt 权重清单（sha256）", "",
           f"生成：`uv run python experiments/weight_manifest/scan.py`（只读扫描）",
           f"扫描时间：{time.strftime('%Y-%m-%d %H:%M:%S %z')}",
           f"线上基座 doc_sha256 = `{online_doc_sha}`（M0 自校验通过 = {m0_ok}）", "",
           f"**{summary['n_pt_scanned']} 个 .pt / {summary['total_bytes']:,} 字节**；"
           f"与线上 base 一致 **{len(same)}** / 不一致 **{len(diff)}** / 不适用 **{len(na)}**（含加载失败 {len(load_err)}）", "",
           "列：路径 | 字节 | 文件 sha256 前16 | 内含 encoder? | doc_encoder sha256 前16 | 与线上 base 一致? | 来源线索（base / split_from）", ""]
    for d in sorted(groups):
        es = groups[d]
        out.append(f"## {d}/  （{len(es)} 个，{sum(x['bytes'] for x in es):,} 字节）")
        out.append("| 文件 | 字节 | 文件 sha16 | encoder? | doc_sha16 | 同线上? | 来源线索 |")
        out.append("|---|---:|---|---|---|---|---|")
        for e in sorted(es, key=lambda x: x["path"]):
            prov = []
            if e.get("train_args.base"):
                prov.append(f"base={e['train_args.base']}")
            if e.get("train_args.split_from"):
                prov.append(f"split_from={e['train_args.split_from']}")
            same_s = "—" if e["same_as_online_base"] is None else ("是" if e["same_as_online_base"] else "否")
            out.append(f"| `{Path(e['path']).name}` | {e['bytes']:,} | `{e['file_sha256'][:16]}` | "
                       f"{'是' if e['has_doc_encoder'] else '否'} | "
                       f"`{str(e['doc_sha256'])[:16]}` | {same_s} | {'；'.join(prov) or '—'} |")
        out.append("")
    (HERE / "MANIFEST.md").write_text("\n".join(out) + "\n", encoding="utf-8")

    (HERE / "summary.json").write_text(json.dumps({
        "summary": summary,
        "diff_paths": [e["path"] for e in diff],
        "dup_groups": dup_groups,
        "legacy_sha_record_paths": sorted(legacy_paths),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"[save] {HERE}/manifest.json  manifest.sha256  MANIFEST.md")
    print("SCAN_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
