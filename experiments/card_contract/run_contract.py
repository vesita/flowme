#!/usr/bin/env python3
"""P13 卡片契约统一 —— 驱动（判据见 `PREREG.md`，mtime 早于任何改动）。

    uv run python experiments/card_contract/run_contract.py --phase baseline   # 改动前快照
    uv run python experiments/card_contract/run_contract.py --phase final      # 改动后 + V0–V5

产物：`baseline_render.json`、`after_render.json`、`results.json`。
纯 CPU、只读既有产物、不训练、不碰 GPU。
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for _p in ("src", "experiments/card_flow", "experiments/gen_dispatch",
           "experiments/explain_card", "experiments/pointer_explain"):
    if str(ROOT / _p) not in sys.path:
        sys.path.insert(0, str(ROOT / _p))

from dtseek.tasks import card_contract as CC  # noqa: E402
from dtseek.tasks import render as R  # noqa: E402

import fixtures as FX  # noqa: E402
import migration as M  # noqa: E402
import replay as RP  # noqa: E402

PREREG = HERE / "PREREG.md"
BASE = HERE / "baseline_render.json"
AFTER = HERE / "after_render.json"
RESULTS = HERE / "results.json"

SRC_CHANGED = ["src/dtseek/tasks/render.py", "src/dtseek/tasks/card_contract.py"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gate(final: bool = False) -> dict:
    """PREREG 先于改动、先于产物 —— 驱动里的一道断言。

    `final=True`（改动后）时额外断言：每个被改动的 `src/` 文件 mtime **晚于** PREREG
    （= 判据确实写在改动之前）；`baseline` 阶段还没改，只断言 PREREG 早于产物。
    """
    p = PREREG.stat().st_mtime
    srcs = {f: (ROOT / f).stat().st_mtime for f in SRC_CHANGED if (ROOT / f).exists()}
    after = AFTER.stat().st_mtime if AFTER.exists() else None
    if final:
        bad = [f for f, t in srcs.items() if t < p]
        if bad:
            raise SystemExit(f"PREREG 不早于改动：{bad}")
    if after is not None and after < p:
        raise SystemExit("PREREG 不早于首个结果产物")
    return {"prereg_mtime": p, "src_mtime": srcs, "after_mtime": after,
            "prereg_iso": time.strftime("%F %T", time.localtime(p)),
            "src_iso": {f: time.strftime("%F %T", time.localtime(t))
                        for f, t in srcs.items()}}


def snapshot() -> dict:
    """改动前/后都要跑的同一份快照（V0/V3 就比它）。"""
    g = RP.load_g()
    a = RP.load_a()
    return {
        "official_fixtures": FX.official_fix(),
        "cf_fixtures": FX.cf_fix(),
        "g_replay": [RP.replay_g(x) for x in g],
        "a_replay": [RP.replay_a(x) for x in a],
        "raw_rejects": [{"src": r["src"], "rec": r["rec"]} for r in RP.raw_rejects()],
        "table": {sid: M.sign(sid) for sid in sorted(R.SKELETONS)},
        "n_skeletons": len(R.SKELETONS),
        "validate": R.validate_skeleton_table(),
    }


def phase_baseline() -> None:
    gk = gate()
    snap = snapshot()
    BASE.write_text(json.dumps(snap, ensure_ascii=False, sort_keys=True, indent=1))
    log(f"baseline 写出 {BASE.name}（官方夹具 {len(snap['official_fixtures'])}、"
        f"G 重放 {len(snap['g_replay'])}、A 重放 {len(snap['a_replay'])}、"
        f"raw 拒答 {len(snap['raw_rejects'])}、表 {snap['n_skeletons']} 条）")
    log(f"gate={json.dumps(gk, ensure_ascii=False)}")
    counts = _count(snap["g_replay"]) , _count(snap["a_replay"])
    log(f"baseline 状态计数 G={counts[0]} A={counts[1]}")


def _count(rows: list[dict]) -> dict:
    out: dict[str, int] = {}
    for r in rows:
        out[r["status"]] = out.get(r["status"], 0) + 1
    return out


def diff_snapshots(before: dict, after: dict) -> dict:
    """逐键比对：只允许**新增**（CF 夹具 / A 集重放），官方族必须逐字节相同。"""
    out: dict = {}
    # ① 官方夹具：全等
    off_diff = {k: {"before": before["official_fixtures"].get(k),
                    "after": after["official_fixtures"].get(k)}
                for k in before["official_fixtures"]
                if before["official_fixtures"].get(k) != after["official_fixtures"].get(k)}
    out["official_fixture_diff"] = off_diff
    # ② G 记录重放：状态与文本都必须不变
    bg = {r["rid"]: r for r in before["g_replay"]}
    ag = {r["rid"]: r for r in after["g_replay"]}
    g_diff = {k: {"before": bg.get(k), "after": ag.get(k)}
              for k in bg if bg.get(k) != ag.get(k)}
    out["g_replay_diff"] = g_diff
    # ③ raw 拒答记录：逐字节不变（拒答语义的"默认行为"证据）
    br = {r["src"]: r["rec"] for r in before["raw_rejects"]}
    ar = {r["src"]: r["rec"] for r in after["raw_rejects"]}
    out["raw_reject_diff"] = {k: {"before": br.get(k), "after": ar.get(k)}
                              for k in br if br.get(k) != ar.get(k)}
    # ④ 表：官方族 id/pattern/签名必须逐字不变
    tb, ta = before["table"], after["table"]
    out["official_table_diff"] = {k: {"before": tb.get(k), "after": ta.get(k)}
                                  for k in tb if tb.get(k) != ta.get(k)}
    out["table_added"] = sorted(set(ta) - set(tb))
    out["g_before"] = _count(before["g_replay"])
    out["g_after"] = _count(after["g_replay"])
    out["a_before"] = _count(before["a_replay"])
    out["a_after"] = _count(after["a_replay"])
    return out


# ---- V0–V5 --------------------------------------------------------------------

def pos_check() -> dict:
    """G 集 pos 回填的**弱佐证**：回填出的 pos 计数 ≤ 原跑整袋的 `bag_pos` 直方图。

    等式佐证不可得（`bag_pos` 不按 ref 对齐），只能给不等式；结论按推断级标注。
    """
    from collections import Counter
    rows = []
    for g in RP.load_g():
        sk = R.SKELETONS[g["instruction"]["skeleton_id"]]
        assign = g["instruction"]["assignment"]
        back = Counter(sk.slots[k].pos for k in range(len(assign)))
        bp = g.get("bag_pos") or {}
        bad = {p: c for p, c in back.items() if c > bp.get(p, 0)}
        rows.append({"rid": g["rid"], "backfilled": dict(back), "bag_pos": bp,
                     "violations": bad})
    return {"n": len(rows), "violations": [r for r in rows if r["violations"]],
            "note": "bag_pos = 整袋 pos 直方图（不按 ref）⇒ 只能做 ⊆ 不等式佐证（推断级）"}


def v0(after: dict, d: dict) -> dict:
    g, a = after["g_replay"], after["a_replay"]
    return {
        "official_fixture_diff_n": len(d["official_fixture_diff"]),
        "official_table_diff_n": len(d["official_table_diff"]),
        "g_n": len(g),
        "g_mismatch": sum(1 for r in g if r["status"] == "diff"),
        "g_reject": sum(1 for r in g if r["status"] == "reject"),
        "g_status": d["g_after"],
        "a_n": len(a),
        "a_mismatch": sum(1 for r in a if r["status"] == "diff"),
        "a_match": sum(1 for r in a if r["status"] == "match"),
        "a_deprecated": sum(1 for r in a if r["status"] == "deprecated"),
        "a_reject": [r for r in a if r["status"] == "reject"],
        "a_status": d["a_after"],
        "raw_reject_diff_n": len(d["raw_reject_diff"]),
        "g_replay_diff_n": len(d["g_replay_diff"]),
        "g_pos_check": pos_check(),
    }


def v1() -> dict:
    rep = M.v1_report()
    disp = M.record_disposition(RP.load_a())
    by_old: dict[str, dict] = {}
    for r in disp:
        e = by_old.setdefault(r["old"], {"old": r["old"], "new": r["new"],
                                         "kind": r["kind"], "records": 0,
                                         "reason": r["reason"]})
        e["records"] += 1
    rep["disposition_per_id"] = [by_old[k] for k in sorted(by_old)]
    rep["n_records_disposed"] = len(disp)
    rep["records_naive_clash"] = sum(1 for r in disp if r["same_id_diff_sign_naive"])
    rep["gate"] = gate(final=True)
    return rep


def v2() -> dict:
    """拒答契约：归一前 vs 归一后（同一批记录，同一套谓词）。"""
    import rules as G  # gen_dispatch 的既有契约谓词（只读 import）

    raws = RP.raw_rejects() + RP.stored_rejects()
    before = []
    for r in raws:
        before.append({"src": r["src"],
                       "ours": CC.contract_problems(r["rec"]),
                       "theirs": G.contract_problems(r["rec"])})
    after = []
    for r in raws:
        rec = CC.to_contract(r.get("as_record", r["rec"]))
        d = CC.to_dict(rec)
        after.append({"src": r["src"],
                      "has_text_key": "text" in d,
                      "text": d.get("text"),
                      "reason_empty": not str(d.get("reason") or "").strip(),
                      "ours": CC.contract_problems(rec),
                      "theirs": G.contract_problems(d),
                      "record": d})
    return {
        "n_raw": len(raws),
        "raw_with_text": sum(1 for r in raws if r["rec"].get("text") is not None),
        "raw_ours_violate": sum(1 for r in before if r["ours"]),
        "raw_theirs_violate": sum(1 for r in before if r["theirs"]),
        "raw_both_agree": sum(1 for r in before if bool(r["ours"]) == bool(r["theirs"])),
        "raw_examples": [r for r in before if r["ours"]][:3],
        "norm_with_text": sum(1 for r in after if r["has_text_key"]),
        "norm_reason_empty": sum(1 for r in after if r["reason_empty"]),
        "norm_ours_violate": sum(1 for r in after if r["ours"]),
        "norm_theirs_violate": sum(1 for r in after if r["theirs"]),
        "norm_violation_detail": [r for r in after if r["ours"] or r["theirs"]],
        "norm_agree": sum(1 for r in after if bool(r["ours"]) == bool(r["theirs"])),
    }


def v3(pytest_log: str) -> dict:
    """既有 pytest + 非拒答记录逐字一致（重放部分由 V0 覆盖，这里补测试与改动清单）。"""
    txt = Path(pytest_log).read_text() if Path(pytest_log).exists() else ""
    tail = [l for l in txt.strip().splitlines()[-6:]]
    status = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                            capture_output=True, text=True).stdout.strip().splitlines()
    diff = subprocess.run(["git", "diff", "--stat", "--", "src/"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()
    return {"pytest_tail": tail, "pytest_exit": _exit_of(txt),
            "git_status": status, "git_diff_stat_src": diff}


def _exit_of(txt: str) -> str:
    for line in txt.strip().splitlines()[::-1]:
        if line.startswith("EXIT="):
            return line.split("=", 1)[1]
    return "?"


def pytest_run(path: str) -> str:
    """跑一次全库 pytest，把原始输出存进 logs/。"""
    out = ROOT / "logs" / "p13" / path
    out.parent.mkdir(parents=True, exist_ok=True)
    p = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=ROOT,
                       capture_output=True, text=True)
    out.write_text(p.stdout + p.stderr + f"\nEXIT={p.returncode}\n")
    log(f"pytest → {out.name}（exit={p.returncode}）")
    return str(out)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["baseline", "final"], required=True)
    ap.add_argument("--pytest-log", default="")
    args = ap.parse_args()

    if args.phase == "baseline":
        phase_baseline()
        return 0

    gk = gate(final=True)
    if not BASE.exists():
        raise SystemExit("缺 baseline_render.json：先跑 --phase baseline")
    before = json.loads(BASE.read_text())
    # 同口径：两边都经 JSON 往返（tuple → list），否则元组/列表的 `!=` 会把全表误判成差异
    after = json.loads(json.dumps(snapshot(), ensure_ascii=False, sort_keys=True))
    AFTER.write_text(json.dumps(after, ensure_ascii=False, sort_keys=True, indent=1))
    d = diff_snapshots(before, after)
    log(f"after 写出 {AFTER.name}；官方夹具差 {len(d['official_fixture_diff'])}、"
        f"G 重放差 {len(d['g_replay_diff'])}、raw 拒答差 {len(d['raw_reject_diff'])}")

    res: dict = {"gate": gk, "diff": d, "V0": v0(after, d), "V1": v1(),
                 "V2": v2()}
    if args.pytest_log:
        res["V3"] = v3(args.pytest_log)
    import explain_ops as EO  # noqa: E402
    res["V4"] = EO.v4()
    res["V5"] = EO.v5()
    res["verdict_inputs"] = verdict(res)
    RESULTS.write_text(json.dumps(res, ensure_ascii=False, sort_keys=True, indent=1))
    log(f"results 写出 {RESULTS.name}")
    print(json.dumps({k: res[k] for k in ("V0", "V1", "V2")}, ensure_ascii=False,
                     indent=1)[:4000])
    return 0


def verdict(res: dict) -> dict:
    v0ok = (res["V0"]["official_fixture_diff_n"] == 0
            and res["V0"]["official_table_diff_n"] == 0
            and res["V0"]["g_mismatch"] == 0 and res["V0"]["g_reject"] == 0
            and res["V0"]["a_mismatch"] == 0
            and res["V0"]["a_deprecated"] == 1
            and res["V0"]["raw_reject_diff_n"] == 0)
    v1ok = (not res["V1"]["home_same_id_diff_sign"]
            and not res["V1"]["namespace_overlap"]
            and not res["V1"]["migrated_sig_mismatch"]
            and not res["V1"]["unresolved"]
            and res["V1"]["records_naive_clash"] == 36)
    v2ok = (res["V2"]["norm_with_text"] == 0 and res["V2"]["norm_reason_empty"] == 0)
    v3ok = res.get("V3", {}).get("pytest_exit") == "0"
    v4ok = (res["V4"]["explainable_all"] and res["V4"]["pointer_covered"] > 0
            and res["V4"]["pointer_baseline_covered"] == 0
            and res["V4"]["pointer_contract_violate_with_structure"] == 0)
    v5ok = res["V5"]["missed"] == 0
    flags = {"V0": v0ok, "V1": v1ok, "V2": v2ok, "V3": v3ok, "V4": v4ok, "V5": v5ok}
    if all(flags.values()):
        name = "契约统一成立"
    elif not v0ok or not v3ok:
        name = "不成立"
    else:
        name = "部分成立"
    return {"flags": flags, "verdict": name}


if __name__ == "__main__":
    raise SystemExit(main())
