"""阶段 1 判据测量：G0（零门控逐位不变）/ G1（与挂卡无关）/ G2（非空 + 反面控制）。

判据与口径**逐字来自 `PREREG.md`（跑前写死）**；本文件只负责执行与记录，不改门槛。

    uv run python -u experiments/core_ndb/run_phase1.py --out experiments/core_ndb/results_phase1.json

只写 `experiments/core_ndb/`、`logs/`、`/tmp`。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from pipeline import CoreNDBEngine, strip_display  # noqa: E402

CARDS = ["person", "pronoun", "relation", "sentiment"]
N_TEXTS = 16
PERSON_CACHE = ROOT / "experiments" / "core_keep" / "cache" / "person_6000.pkl"
CARDS_DIR = ROOT / "checkpoints" / "cards"
BASE = ROOT / "checkpoints" / "base_encoder.pt"

#: 条件 → (write_scale, read_scale, card_write)；S0 关记忆
CONDS = {
    "S0":  dict(memory=False, ws=None, rs=None, card_write=False),
    "S1":  dict(memory=True, ws=0.0, rs=0.0, card_write=False),
    "S2r": dict(memory=True, ws=0.0, rs=1.0, card_write=False),
    "S2":  dict(memory=True, ws=1.0, rs=1.0, card_write=False),
    "S3":  dict(memory=True, ws=1.0, rs=1.0, card_write=True),
}


def load_texts() -> list[str]:
    """PREREG §3：person 缓存按序取前 16 条 spans 非空的样本 text（不重排）。"""
    rows = pickle.load(open(PERSON_CACHE, "rb"))
    out = []
    for x in rows:
        sp = x["spans"] if not isinstance(x["spans"], str) else json.loads(x["spans"])
        if sp:
            out.append(x["text"])
        if len(out) == N_TEXTS:
            break
    if len(out) != N_TEXTS:
        raise SystemExit(f"文本集不足：只要到 {len(out)} 条")
    return out


def sample_hash(engine: CoreNDBEngine, text: str, card: str) -> str:
    r = engine.predict(text, tasks=[card])
    blob = json.dumps(strip_display(r), sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def aggregate(hs: list[str]) -> str:
    return hashlib.sha256("\n".join(hs).encode("utf-8")).hexdigest()


def norm_hash(engine: CoreNDBEngine, text: str, card: str,
              tasks: list[str] | None) -> str:
    """跨**调用形态**可比的归一哈希：只含 `{text, tasks:{card}}`。

    `num_segments` 是「所有任务里最大段数」，随挂了哪些卡变，故不入 payload —— 否则
    `tasks=[c]` 与 `tasks=全部` 两种形态永远不等，比较就失去意义。
    """
    r = engine.predict(text, tasks=tasks)
    payload = {"text": r.get("text"),
               "tasks": {card: strip_display(r["tasks"].get(card, []))}}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()


def anchors_of(engine: CoreNDBEngine, text: str, card: str) -> list[dict]:
    r = engine.predict(text, tasks=[card])
    return strip_display(r)["tasks"].get(card, [])


def struct_key(anchors: list[dict]) -> list[tuple]:
    """结构口径（不含 confidence）：类别 + 起止，用来区分「只动了小数位」与「锚点变了」。"""
    return [(a["class_id"], a["s0"], a["e0"]) for a in anchors]


def sym_diff(a: list, b: list) -> int:
    from collections import Counter
    ca, cb = Counter(a), Counter(b)
    return sum((ca - cb).values()) + sum((cb - ca).values())


def set_cond(engine: CoreNDBEngine, name: str) -> None:
    c = CONDS[name]
    engine.memory = c["memory"]
    engine.card_write = c["card_write"]
    if c["ws"] is not None:
        engine.core.write_scale.data.fill_(c["ws"])
    if c["rs"] is not None:
        engine.core.read_scale.data.fill_(c["rs"])


def run_cond(engine: CoreNDBEngine, texts: list[str], cards: list[str], name: str) -> dict:
    """跑一个条件：返回逐卡哈希、诊断聚合、耗时。"""
    set_cond(engine, name)
    hashes: dict[str, list[str]] = {c: [] for c in cards}
    diag = {"w_sum": 0.0, "delta_l1": 0.0, "n_write_pos": 0, "g_mean": [], "mass_mean": [],
            "n_slot_touched_last": 0, "pre_segments": []}
    t0 = time.time()
    n_pred = 0
    for t in texts:
        for c in cards:
            hashes[c].append(sample_hash(engine, t, c))
            n_pred += 1
            s = engine.core.stats()
            diag["w_sum"] += s["w_sum"]
            diag["delta_l1"] += s["delta_l1"]
            diag["n_write_pos"] += s["n_write_pos"]
            if s["g_mean"] == s["g_mean"]:        # 非 NaN
                diag["g_mean"].append(s["g_mean"])
                diag["mass_mean"].append(s["mass_mean"])
            diag["n_slot_touched_last"] = s["n_slot_touched"]
            diag["pre_segments"].append(getattr(engine, "pre_segments", 0))
    dt = time.time() - t0
    if diag["g_mean"]:
        diag["g_mean"] = sum(diag["g_mean"]) / len(diag["g_mean"])
        diag["mass_mean"] = sum(diag["mass_mean"]) / len(diag["mass_mean"])
    else:
        diag["g_mean"] = diag["mass_mean"] = float("nan")
    diag["pre_segments_mean"] = (sum(diag["pre_segments"]) / len(diag["pre_segments"])
                                 if diag["pre_segments"] else 0.0)
    return {
        "cond": name,
        "hashes": hashes,
        "agg": {c: aggregate(hashes[c]) for c in cards},
        "diags": diag,
        "n_predict": n_pred,
        "wall_s": round(dt, 3),
        "ms_per_predict": round(dt * 1000.0 / max(1, n_pred), 2),
    }


def cmp_agg(a: dict, b: dict, cards: list[str]) -> dict:
    out = {}
    for c in cards:
        ha, hb = a["hashes"][c], b["hashes"][c]
        diff = [i for i, (x, y) in enumerate(zip(ha, hb)) if x != y]
        out[c] = {
            "equal": a["agg"][c] == b["agg"][c],
            "agg_a": a["agg"][c], "agg_b": b["agg"][c],
            "n_diff_texts": len(diff), "first_diff_idx": (diff[0] if diff else None),
            "hashes_a": ha, "hashes_b": hb,
        }
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="core_ndb 阶段1：G0/G1/G2 实测")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out", default=str(HERE / "results_phase1.json"))
    args = ap.parse_args(argv)

    texts = load_texts()
    print(f"[texts] {len(texts)} 条；长度 {sorted(len(t) for t in texts)}", flush=True)

    res: dict = {
        "prereg": "experiments/core_ndb/PREREG.md",
        "cards": CARDS,
        "n_texts": len(texts),
        "texts": texts,
        "conds": {k: v for k, v in CONDS.items()},
        "runs": {},
    }

    # ---- 4 卡引擎（G0 / G2a / G2a' / G2b / R0 的载体）---------------------
    print("[engine] 4-card", flush=True)
    e4 = CoreNDBEngine(base_path=str(BASE), cards_dir=str(CARDS_DIR),
                       device=args.device, auto_attach=True)
    assert e4.attached == sorted(CARDS), e4.attached

    runs: dict[str, dict] = {}
    runs["S0_a"] = run_cond(e4, texts, CARDS, "S0")
    runs["S0_b"] = run_cond(e4, texts, CARDS, "S0")     # R0 重复性
    runs["S1"] = run_cond(e4, texts, CARDS, "S1")       # G0
    runs["S2r"] = run_cond(e4, texts, CARDS, "S2r")     # G2a'
    runs["S2_a"] = run_cond(e4, texts, CARDS, "S2")     # G2a / G1 左端
    runs["S2_b"] = run_cond(e4, texts, CARDS, "S2")     # R0 重复性
    runs["S3"] = run_cond(e4, texts, CARDS, "S3")       # G2b 反面控制

    # ---- 单卡引擎（G1 右端）---------------------------------------------
    single: dict[str, dict] = {}
    for c in CARDS:
        print(f"[engine] single {c}", flush=True)
        e1 = CoreNDBEngine(base_path=str(BASE), device=args.device, auto_attach=False)
        e1.attach(str(CARDS_DIR / f"{c}.pt"))
        assert e1.attached == [c], e1.attached
        single[f"{c}/S2"] = run_cond(e1, texts, [c], "S2")
        single[f"{c}/S1"] = run_cond(e1, texts, [c], "S1")
        del e1
        import torch
        torch.cuda.empty_cache()

    # ---- 跨调用形态：`tasks=[c]` vs `tasks=全部` 同跑（G1b / G2b 跨卡 / G0 跨形态）
    # 这是 §4 意图的字面形态：**同一趟 predict 里挂着 4 张卡**，person 写共享表，
    # 后跑的老卡读到被污染的表 —— 「加一张卡 ⇒ 老卡输出被改变」。
    print("[cross] 跨调用形态 S0/S1/S2/S3（全部卡同跑）+ S2 单卡形态", flush=True)
    norm: dict[str, dict[str, list[str]]] = {}
    for cname in ("S0", "S1", "S2", "S3"):
        set_cond(e4, cname)
        norm[f"{cname}_all"] = {c: [norm_hash(e4, t, c, None) for t in texts]
                                for c in CARDS}
    set_cond(e4, "S2")
    norm["S2_single"] = {c: [norm_hash(e4, t, c, [c]) for t in texts] for c in CARDS}

    def all_struct(cname: str) -> dict:
        set_cond(e4, cname)
        out = {c: [] for c in CARDS}
        for t in texts:
            r = strip_display(e4.predict(t))
            for c in CARDS:
                out[c].append(struct_key(r["tasks"].get(c, [])))
        return out

    struct_all_2 = all_struct("S2")
    struct_all_3 = all_struct("S3")

    # ---- 结构差异（G2a / G2b 的「锚点本身变了」口径）-----------------------
    print("[struct] 结构口径复算（S0/S2/S3）", flush=True)
    set_cond(e4, "S0"); struct0 = {c: [struct_key(anchors_of(e4, t, c)) for t in texts]
                                   for c in CARDS}
    set_cond(e4, "S2"); struct2 = {c: [struct_key(anchors_of(e4, t, c)) for t in texts]
                                   for c in CARDS}
    set_cond(e4, "S3"); struct3 = {c: [struct_key(anchors_of(e4, t, c)) for t in texts]
                                   for c in CARDS}

    def struct_cmp(a, b):
        out = {}
        for c in CARDS:
            d = [i for i, (x, y) in enumerate(zip(a[c], b[c])) if x != y]
            changed = sum(sym_diff(a[c][i], b[c][i]) for i in d)
            out[c] = {"n_diff_texts": len(d), "n_anchor_symdiff": changed,
                      "first_diff_idx": (d[0] if d else None)}
        return out

    res["runs"] = {k: v for k, v in runs.items()}
    res["single"] = single
    res["R0"] = {
        "S0_repeat": cmp_agg(runs["S0_a"], runs["S0_b"], CARDS),
        "S2_repeat": cmp_agg(runs["S2_a"], runs["S2_b"], CARDS),
    }
    res["G0"] = cmp_agg(runs["S1"], runs["S0_a"], CARDS)
    res["G1"] = {
        "S2_active": {c: {
            "e4_agg": runs["S2_a"]["agg"][c],
            "e1_agg": single[f"{c}/S2"]["agg"][c],
            "equal": runs["S2_a"]["agg"][c] == single[f"{c}/S2"]["agg"][c],
            "hashes_e4": runs["S2_a"]["hashes"][c],
            "hashes_e1": single[f"{c}/S2"]["hashes"][c],
        } for c in CARDS},
        "S1_zero": {c: {
            "e4_agg": runs["S1"]["agg"][c],
            "e1_agg": single[f"{c}/S1"]["agg"][c],
            "equal": runs["S1"]["agg"][c] == single[f"{c}/S1"]["agg"][c],
        } for c in CARDS},
    }
    res["G2a"] = dict(cmp_agg(runs["S2_a"], runs["S0_a"], CARDS),
                      struct=struct_cmp(struct0, struct2))
    res["G2ap"] = cmp_agg(runs["S2r"], runs["S0_a"], CARDS)
    res["G2b"] = dict(cmp_agg(runs["S3"], runs["S2_a"], CARDS),
                      struct=struct_cmp(struct2, struct3))

    # ---- 跨调用形态（§4 意图的字面形态：4 卡同跑，person 写、老卡读）--------
    def cross_cmp(a, b) -> dict:
        out = {}
        for c in CARDS:
            ha, hb = a[c], b[c]
            diff = [i for i, (x, y) in enumerate(zip(ha, hb)) if x != y]
            out[c] = {"equal": ha == hb, "n_diff_texts": len(diff),
                      "first_diff_idx": (diff[0] if diff else None),
                      "agg_a": aggregate(ha), "agg_b": aggregate(hb),
                      "hashes_a": ha, "hashes_b": hb}
        return out

    res["cross"] = {
        "G0_forms": cross_cmp(norm["S0_all"], norm["S1_all"]),
        "G1_forms": cross_cmp(norm["S2_single"], norm["S2_all"]),
        "G2b_cross": dict(cross_cmp(norm["S2_all"], norm["S3_all"]),
                          struct=struct_cmp(struct_all_2, struct_all_3)),
    }
    res["diags"] = {k: v["diags"] for k, v in runs.items()}
    res["timing"] = {k: {"wall_s": v["wall_s"], "ms_per_predict": v["ms_per_predict"]}
                     for k, v in runs.items()}

    # ---- 判定（逐字来自 PREREG §4）---------------------------------------
    verdict = {}
    verdict["R0"] = all(v["equal"] for v in res["R0"]["S0_repeat"].values()) and \
                    all(v["equal"] for v in res["R0"]["S2_repeat"].values())
    verdict["G0"] = all(v["equal"] for v in res["G0"].values()) and \
                    all(v["equal"] for v in res["cross"]["G0_forms"].values())
    verdict["G1"] = all(v["equal"] for v in res["G1"]["S2_active"].values()) and \
                    all(v["equal"] for v in res["G1"]["S1_zero"].values()) and \
                    all(v["equal"] for v in res["cross"]["G1_forms"].values())
    verdict["G2a"] = all(res["G2a"][c]["n_diff_texts"] > 0 for c in CARDS)
    verdict["G2ap"] = all(v["equal"] for v in res["G2ap"].values())
    verdict["G2b"] = all(res["G2b"][c]["n_diff_texts"] > 0
                         for c in CARDS if c != "person") and \
                     all(res["cross"]["G2b_cross"][c]["n_diff_texts"] > 0
                         for c in CARDS if c != "person")
    verdict["all_pass"] = all(verdict.values())
    res["verdict"] = verdict

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\n===== 判定 =====", flush=True)
    for k in ("R0", "G0", "G1", "G2a", "G2ap", "G2b"):
        print(f"  {k}: {'PASS' if verdict[k] else 'FAIL'}", flush=True)
    for c in CARDS:
        print(f"  G0 {c}: equal={res['G0'][c]['equal']}", flush=True)
        print(f"  G1 {c}: equal(S2)={res['G1']['S2_active'][c]['equal']} "
              f"equal(S1)={res['G1']['S1_zero'][c]['equal']}", flush=True)
        print(f"  G2a {c}: diff_texts={res['G2a'][c]['n_diff_texts']}/16 "
              f"struct_diff={res['G2a']['struct'][c]['n_diff_texts']}", flush=True)
        print(f"  G2ap {c}: equal={res['G2ap'][c]['equal']}", flush=True)
        print(f"  G2b {c}: diff_texts={res['G2b'][c]['n_diff_texts']}/16 "
              f"struct_diff={res['G2b']['struct'][c]['n_diff_texts']} "
              f"anchor_symdiff={res['G2b']['struct'][c]['n_anchor_symdiff']}", flush=True)
        print(f"  G0_x {c}: equal={res['cross']['G0_forms'][c]['equal']} | "
              f"G1_x {c}: equal={res['cross']['G1_forms'][c]['equal']} | "
              f"G2b_x {c}: diff_texts={res['cross']['G2b_cross'][c]['n_diff_texts']}/16 "
              f"struct_diff={res['cross']['G2b_cross']['struct'][c]['n_diff_texts']} "
              f"anchor_symdiff={res['cross']['G2b_cross']['struct'][c]['n_anchor_symdiff']}",
              flush=True)
    print("  diags:", json.dumps({k: {kk: (round(vv, 6) if isinstance(vv, float) else vv)
                                      for kk, vv in v.items() if kk in
                                      ("w_sum", "delta_l1", "g_mean", "mass_mean",
                                       "n_write_pos", "n_slot_touched_last")}
                                  for k, v in res["diags"].items()}, ensure_ascii=False),
          flush=True)
    print("  timing:", json.dumps(res["timing"], ensure_ascii=False), flush=True)
    print(f"  out={args.out}", flush=True)
    return 0 if verdict["all_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
