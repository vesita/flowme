#!/usr/bin/env python3
"""W5/W2 等价性驱动：`src/dtseek/tasks/render.py::POS_TYPES` 改动**前后各跑一次**，逐字比。

    uv run python experiments/verb_card/equiv_official.py --tag before   # 改 src 之前（基线）
    uv run python experiments/verb_card/equiv_official.py --tag after    # 改 src 之后
    uv run python experiments/verb_card/equiv_official.py --compare      # 逐字比对

三件事：
  1. **W5**：既有 16 条官方骨架（+`CF21`）在固定夹具上的渲染记录 —— 前后必须**逐字相同**；
  2. **W2**：card_flow 9 条 `否`/`数` 槽骨架的 `slot_schema_problems` + 渲染 ——
     改前必须全红（`否/数 ∉ POS_TYPES`）、改后必须全绿且渲染正确；
  3. 改动面：`POS_TYPES` 取值 + 关键文件 sha256（W6 的实测证据）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import pipeline as P  # noqa: E402
from fixtures import BY_FID, SKELETON_FIXTURE, W2_SKELETONS  # noqa: E402
from dtseek.tasks import render as R  # noqa: E402

ROOT = HERE.parents[1]

#: W5 的比对对象：**既有 16 条官方骨架**（+ 迁移族里唯一含动槽的 `CF21`，单列）。
OFFICIAL_16: tuple[str, ...] = tuple(f"S{i:02d}" for i in range(1, 14)) + ("R01", "R02", "R03")
WATCH_FILES: tuple[str, ...] = (
    "src/dtseek/tasks/render.py",
    "src/dtseek/tasks/card_contract.py",
    "src/dtseek/tasks/dialogue.py",
    "src/dtseek/tasks/dispatch.py",
    "src/dtseek/tasks/engine.py",
    "src/dtseek/tasks/plugin.py",
    "src/dtseek/tasks/runtime.py",
    "src/dtseek/tasks/compose.py",
    "experiments/card_flow/skeletons.py",
    "experiments/card_flow/lexicon.py",
    "experiments/card_flow/proposers.py",
    "experiments/card_flow/predicates.py",
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _run_one(sid: str, table: dict[str, R.Skeleton]) -> dict:
    fix = BY_FID[SKELETON_FIXTURE[sid]]
    sk = table[sid]
    bag, _valid = P.make_bag(fix.text, fix.role_map)
    out = P.fill(fix.text, bag, sk, table)
    rec = {
        "fid": fix.fid,
        "input": fix.text,
        "kind": out.get("kind"),
        # ⚠️ 记录里**不放** `POS_TYPES`：那正是本次改动本身；W5 只比渲染输出与门禁结果。
        "schema": R.slot_schema_problems(sk),
        "rule_a": R.rule_a_problems(sk),
        "validate": R.validate_skeleton_table(),
    }
    if out.get("kind") == "text":
        rec.update({
            "text": out["text"],
            "instruction": out["instruction"],
            "evidence": out["evidence"],
            "ref_map": out["ref_map"],
            "gates": P.gates(out, bag, fix.text, table),
            "audit": P.trace_audit(out, bag, fix.text, table),
        })
    else:
        rec["reason"] = out.get("reason") or out.get("missing_pos")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", choices=["before", "after"])
    ap.add_argument("--compare", action="store_true")
    args = ap.parse_args()

    if args.compare:
        a = json.loads((HERE / "equiv_before.json").read_text(encoding="utf-8"))
        b = json.loads((HERE / "equiv_after.json").read_text(encoding="utf-8"))
        diff16 = [sid for sid in OFFICIAL_16
                  if json.dumps(a["official16"][sid], sort_keys=True, ensure_ascii=False)
                  != json.dumps(b["official16"][sid], sort_keys=True, ensure_ascii=False)]
        diff_cf21 = int(json.dumps(a["cf21"], sort_keys=True, ensure_ascii=False)
                        != json.dumps(b["cf21"], sort_keys=True, ensure_ascii=False))
        p3_before = {s: (a["p3_nine"][s]["kind"], len(a["p3_nine"][s]["schema"])) for s in W2_SKELETONS}
        p3_after = {s: (b["p3_nine"][s]["kind"], len(b["p3_nine"][s]["schema"])) for s in W2_SKELETONS}
        out = {
            "official16_byte_diff": len(diff16),
            "official16_diff_ids": diff16,
            "cf21_byte_diff": diff_cf21,
            "p3_before": p3_before,
            "p3_after": p3_after,
            "p3_render_ok_before": sum(1 for k, n in p3_before.values() if k == "text"),
            "p3_render_ok_after": sum(1 for k, n in p3_after.values() if k == "text"),
            "p3_schema_clean_after": sum(1 for k, n in p3_after.values() if n == 0),
            "pos_types_before": a["pos_types"],
            "pos_types_after": b["pos_types"],
            "sha_changed": {k: [a["sha256"][k], b["sha256"][k]]
                            for k in a["sha256"] if a["sha256"][k] != b["sha256"][k]},
            "gates_after": [b["official16"][s].get("gates") for s in OFFICIAL_16],
            "audit_after": [len(b["official16"][s].get("audit") or []) for s in OFFICIAL_16],
            "p3_gates_after": [b["p3_nine"][s].get("gates") for s in W2_SKELETONS],
            "p3_audit_after": [len(b["p3_nine"][s].get("audit") or []) for s in W2_SKELETONS],
            "p3_text_after": {s: b["p3_nine"][s].get("text") for s in W2_SKELETONS},
        }
        path = HERE / "equiv_compare.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(json.dumps(out, ensure_ascii=False, indent=1))
        print(f"[写出] {path}")
        return 0

    table = dict(R.SKELETONS)
    cf = P.cf_skeletons()
    payload = {
        "tag": args.tag,
        "pos_types": list(R.POS_TYPES),
        "official16": {sid: _run_one(sid, table) for sid in OFFICIAL_16},
        "cf21": _run_one("CF21", table),
        "p2_cardflow7": {sid: _run_one(sid, cf) for sid in P.CF_VERB_IDS},
        "p3_nine": {sid: _run_one(sid, cf) for sid in W2_SKELETONS},
        "sha256": {rel: _sha(ROOT / rel) for rel in WATCH_FILES},
    }
    path = HERE / f"equiv_{args.tag}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    ok16 = sum(1 for v in payload["official16"].values() if v["kind"] == "text")
    print(f"[{args.tag}] 官方 16 条渲染成功 {ok16}/16；"
          f"CF21={payload['cf21']['kind']}；"
          f"P2(7)={sum(1 for v in payload['p2_cardflow7'].values() if v['kind']=='text')}/7；"
          f"P3(9)={sum(1 for v in payload['p3_nine'].values() if v['kind']=='text')}/9；"
          f"POS_TYPES={list(R.POS_TYPES)}")
    print(f"[写出] {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
