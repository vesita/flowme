"""core_ndb 阶段 2 评测：P1 跨段收益 / P2 温启动不伤老卡 / P3 代价。

判据见 `PREREG_PHASE2.md`（跑前写死）。只写 `experiments/core_ndb/`、`logs/core_ndb/`、`/tmp`。

- P1：强制按句切段（`predict(..., max_chunk_len=30)` 走 else 分支 ⇒ `split_with_global_offsets`），
  C 臂核级记忆按样本 reset（预编码只写 → 卡片阶段读），L 臂 person 卡的 `MentionNDB`
  每段 `reset(1)`（= 现引擎逐段清零的行为）。指标口径逐字复刻 `runtime.evaluate_task`。
- P2：4 张老卡 `exact_match` vs 冻结参照（同一 val 集、同一份代码现算），噪声带按卡。
- P3：训练耗时/显存 + 推理 ms_per_predict + 记忆表大小。
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
for p in ("src", "experiments/additivity", "experiments/capability_map",
          "experiments/core_keep", "experiments/core_ndb"):
    sys.path.insert(0, str(ROOT / p))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.mention_ndb import MentionNDB  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.artifacts import read_card  # noqa: E402
from dtseek.tasks.engine import MultiTaskEngine  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, _truth_of, evaluate_task  # noqa: E402

from prepare import TASK_SAMPLES  # noqa: E402
from train_core_keep import (  # noqa: E402
    DECODER_KWARGS, ENCODER_KWARGS, FROZEN_CARD, HIDDEN_DIM, TASK_ORDER, OLD_CARDS,
    get_raw, split_val,
)
from probe import split_of  # noqa: E402
from pipeline import CoreNDBEngine  # noqa: E402
from train_core_ndb import CoreMemEncoder  # noqa: E402
from core_memory import CoreNDB  # noqa: E402

BAND = {"pronoun": 0.0283, "sentiment": 0.0041, "relation": 0.0139, "person": 0.0033}
CARDS = ["pronoun", "sentiment", "relation", "person"]
BASE = "checkpoints/base_encoder.pt"
SEEDS = (42, 43)
ARMS = ("C", "L")
#: P1：强制按句切段（这个长度 < 任意样本长度 ⇒ 必走 else 分支的 split）
P1_CHUNK = 30


# ---------------------------------------------------------------- 指标
def metrics_of(pairs: list[tuple[list, list]]) -> dict:
    """`runtime.evaluate_task` 的逐样本身份指标，逐字复刻（got/truth = [(label,s0,e0)]）。"""
    exact_n = exact_tot = 0
    loc_n = loc_ok = first_n = first_ok = repeat_n = repeat_ok = 0
    pair_same_ok = pair_same_tot = pair_same_pred = 0
    for got, truth in pairs:
        exact_tot += 1
        if sorted(got) == sorted(truth):
            exact_n += 1
        pred_at = {(s0, e0): lab for lab, s0, e0 in got}
        seen: set[int] = set()
        for lab, s0, e0 in truth:
            if (s0, e0) not in pred_at:
                seen.add(lab)
                continue
            ok = pred_at[(s0, e0)] == lab
            loc_n += 1
            loc_ok += int(ok)
            if lab in seen:
                repeat_n += 1
                repeat_ok += int(ok)
            else:
                first_n += 1
                first_ok += int(ok)
            seen.add(lab)
        spans = [(lab, s0, e0) for lab, s0, e0 in truth if (s0, e0) in pred_at]
        for i in range(len(spans)):
            for j in range(i + 1, len(spans)):
                same_true = spans[i][0] == spans[j][0]
                same_pred = (pred_at[(spans[i][1], spans[i][2])]
                             == pred_at[(spans[j][1], spans[j][2])])
                if same_true and same_pred:
                    pair_same_ok += 1
                if same_true:
                    pair_same_tot += 1
                if same_pred:
                    pair_same_pred += 1
    return {
        "exact_match": exact_n / max(1, exact_tot),
        "id_acc": loc_ok / max(1, loc_n),
        "first_mention_acc": first_ok / max(1, first_n),
        "repeat_mention_acc": repeat_ok / max(1, repeat_n),
        "cluster_f1": 2 * pair_same_ok / max(1, pair_same_tot + pair_same_pred),
        "n_exact": exact_tot, "n_repeat": repeat_n, "n_first": first_n, "n_loc": loc_n,
    }


# ---------------------------------------------------------------- 装载
def load_ckpt(path: str) -> dict:
    return torch.load(path, map_location="cpu", weights_only=False)


def build_engine(ckpt_path: str, arm: str, seed: int, device) -> object:
    """按臂造推理引擎：C 臂用 CoreNDBEngine（核级记忆），L 臂用普通引擎（person 卡级 NDB）。"""
    ck = load_ckpt(ckpt_path)
    if arm == "C":
        eng = CoreNDBEngine(base_path=BASE, auto_attach=False, device=str(device),
                            memory=True, levels=(1, 2), slots=(8192, 4096),
                            max_table_gb=0.5)
        inner = eng._inner_encoder
        eng.core.load_state_dict(ck["core_ndb"], strict=True)
    else:
        eng = MultiTaskEngine(base_path=BASE, auto_attach=False, device=str(device))
        inner = eng.doc_encoder
    for n in OLD_CARDS:
        eng.attach(FROZEN_CARD.format(n, seed))
    inner.load_state_dict(ck["doc_encoder"], strict=True)
    for n in OLD_CARDS:
        eng.decoders[n].load_state_dict(ck["decoders"][n], strict=True)
    if arm == "L":
        ndb = MentionNDB(**ck["card_ndb_kwargs"]).to(device)
        ndb.load_state_dict(ck["card_ndb"], strict=True)
        ndb.eval()
        eng.ndbs["person"] = ndb
    return eng


def val_batches(name: str, seed: int, tokenizer) -> tuple[DataLoader, list[dict]]:
    """val = capability_map 的 eval_S（与 core_keep 的 ALIGN_CHECK 同一批）。"""
    ev, _ = split_of(name, seed)
    spec = resolve_tasks([name])[name].spec
    ds = GenericTaskDataset(ev, tokenizer, spec)
    return DataLoader(ds, batch_size=1, shuffle=False), ev


# ---------------------------------------------------------------- P1
def p1_one(eng, loader, ev: list[dict], spec, device) -> dict:
    """逐条 val 样本强制按句切段解码（batch=1、不打乱 ⇒ 与 ev 逐条对齐）。"""
    pairs = []
    t0 = time.perf_counter()
    n_pred = 0
    for i, batch in enumerate(loader):
        text = ev[i]["text"]
        truth = _truth_of(batch, 0, spec)
        r = eng.predict(text, tasks=["person"], max_chunk_len=P1_CHUNK)
        n_pred += 1
        got = [(a["class_id"], a["s0"], a["e0"]) for a in r["tasks"].get("person", [])]
        pairs.append((got, truth))
    dt = time.perf_counter() - t0
    m = metrics_of(pairs)
    m["ms_per_predict"] = 1000 * dt / max(1, n_pred)
    m["n_samples"] = len(pairs)
    return m


def p1_texts(name: str, seed: int) -> list[str]:
    ev, _ = split_of(name, seed)
    return [x["text"] for x in ev]


# ---------------------------------------------------------------- P2
def p2_for(ckpt_path: str | None, arm: str | None, seed: int, tokenizers, device) -> dict:
    """ckpt=None ⇒ 冻结参照（base + capability_map 冻结卡）。

    有 ckpt 时按**该臂部署形态**测：C 臂 = 核级记忆开着（编码即写读）；
    L 臂 = person 卡的 `MentionNDB` 接进 `evaluate_task`；其余三张卡无记忆。
    """
    if ckpt_path is None:
        enc = NanoDocEncoder(vocab_size=tokenizers.vocab_size, hidden_dim=HIDDEN_DIM,
                             dropout=0.1, **ENCODER_KWARGS).to(device)
        sd = torch.load(BASE, map_location="cpu", weights_only=False)
        enc.load_state_dict(sd["doc_encoder"], strict=True)
        core = enc
        decs = {}
        for n in CARDS:
            ck = read_card(FROZEN_CARD.format(n, seed))
            d = RobustARSliceDecoder(hidden_dim=HIDDEN_DIM,
                                     num_classes=resolve_tasks([n])[n].spec.num_classes,
                                     **DECODER_KWARGS).to(device)
            d.load_state_dict(ck["decoder"], strict=True)
            decs[n] = d
    else:
        ck = load_ckpt(ckpt_path)
        inner = NanoDocEncoder(vocab_size=tokenizers.vocab_size, hidden_dim=HIDDEN_DIM,
                               dropout=0.1, **ENCODER_KWARGS).to(device)
        inner.load_state_dict(ck["doc_encoder"], strict=True)
        core = inner
        ndb_person = None
        if arm == "C":
            cn = CoreNDB(**ck["core_ndb_kwargs"]).to(device)
            cn.load_state_dict(ck["core_ndb"], strict=True)
            cn.eval()
            core = CoreMemEncoder(inner, cn).to(device)
        elif arm == "L":
            ndb_person = MentionNDB(**ck["card_ndb_kwargs"]).to(device)
            ndb_person.load_state_dict(ck["card_ndb"], strict=True)
            ndb_person.eval()
        decs = {}
        for n in CARDS:
            spec = resolve_tasks([n])[n].spec
            d = RobustARSliceDecoder(hidden_dim=HIDDEN_DIM,
                                     num_classes=spec.num_classes,
                                     **DECODER_KWARGS).to(device)
            d.load_state_dict(ck["decoders"][n], strict=True)
            decs[n] = d
    cards = resolve_tasks(CARDS)
    out = {}
    for n in CARDS:
        spec = cards[n].spec
        loader = DataLoader(GenericTaskDataset(split_of(n, seed)[0], tokenizers, spec),
                            batch_size=64, shuffle=False)
        m = evaluate_task(core, decs[n], loader, device, spec,
                          ndb=(ndb_person if (ckpt_path and n == "person") else None))
        out[n] = {"exact_match": m["exact_match"], "cls_acc": m["cls_acc"],
                  "bg_fp": m["bg_fp"],
                  "repeat_mention_acc": m.get("repeat_mention_acc"),
                  "first_mention_acc": m.get("first_mention_acc")}
    return out


# ---------------------------------------------------------------- main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="core_ndb 阶段 2 评测")
    ap.add_argument("--out", default=str(HERE / "results_phase2.json"))
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = NanoCharTokenizer()
    res: dict = {"p1_chunk": P1_CHUNK, "band": BAND, "arms": {}, "baseline": {},
                 "timing": {}}

    # ---- P2 参照基线（每个 seed 各算一次）----
    print("[P2] 冻结参照 ...", flush=True)
    for s in SEEDS:
        res["baseline"][str(s)] = p2_for(None, None, s, tokenizer, device)
        print("  s%d %s" % (s, json.dumps(
            {k: round(v["exact_match"], 4) for k, v in
             res["baseline"][str(s)].items()})), flush=True)

    for arm in ARMS:
        res["arms"][arm] = {}
        for s in SEEDS:
            ckpt = str(HERE / "cards" / f"p2_{arm}_s{s}.pt")
            print(f"[{arm}/s{s}] 装载 {ckpt}", flush=True)
            # ---- P2 ----
            got = p2_for(ckpt, arm, s, tokenizer, device)
            base = res["baseline"][str(s)]
            delta = {n: got[n]["exact_match"] - base[n]["exact_match"] for n in CARDS}
            p2_ok = all(delta[n] >= -BAND[n] for n in CARDS)
            # ---- P1 ----
            eng = build_engine(ckpt, arm, s, device)
            loader, ev = val_batches("person", s, tokenizer)
            spec = eng.specs["person"]
            p1 = p1_one(eng, loader, ev, spec, device)
            # 对照：同一引擎、关掉记忆（C 臂 memory=False；L 臂把 ndb 摘掉）
            if arm == "C":
                eng.memory = False
            else:
                ndb_keep = eng.ndbs.pop("person", None)
            p1_off = p1_one(eng, loader, ev, spec, device)
            if arm == "L" and ndb_keep is not None:
                eng.ndbs["person"] = ndb_keep
            res["arms"][arm][str(s)] = {
                "ckpt": ckpt, "p2_exact": got, "p2_delta": delta, "p2_ok": p2_ok,
                "p1_on": p1, "p1_off": p1_off,
            }
            print(f"  [{arm}/s{s}] P2 Δ={json.dumps({k: round(v,4) for k,v in delta.items()})} "
                  f"ok={p2_ok} | P1 repeat={p1['repeat_mention_acc']:.4f} "
                  f"exact={p1['exact_match']:.4f} (关记忆 "
                  f"repeat={p1_off['repeat_mention_acc']:.4f} "
                  f"exact={p1_off['exact_match']:.4f}) "
                  f"{p1['ms_per_predict']:.1f} ms/段样本", flush=True)

    # ---- 判据 ----
    verdict = {}
    def spread(metric: str) -> float:
        c = [res["arms"]["C"][str(s)]["p1_on"][metric] for s in SEEDS]
        l = [res["arms"]["L"][str(s)]["p1_on"][metric] for s in SEEDS]
        return max(abs(c[0] - c[1]), abs(l[0] - l[1]))

    def strict_spread(metric: str) -> float:
        vals = [res["arms"][a][str(s)]["p1_on"][metric] for a in ARMS for s in SEEDS]
        return max(vals) - min(vals)

    verdict["p1"] = {}
    for metric in ("repeat_mention_acc", "exact_match"):
        sp, sps = spread(metric), strict_spread(metric)
        deltas = {str(s): (res["arms"]["C"][str(s)]["p1_on"][metric]
                           - res["arms"]["L"][str(s)]["p1_on"][metric]) for s in SEEDS}
        same_sign = (deltas["42"] > 0) and (deltas["43"] > 0)
        verdict["p1"][metric] = {
            "deltas": deltas, "spread": sp, "spread_strict": sps,
            "d_ge_spread": all(deltas[str(s)] >= sp for s in SEEDS),
            "same_sign_positive": same_sign,
            "pass": all(deltas[str(s)] >= sp for s in SEEDS) and same_sign,
        }
    verdict["p1_pass"] = all(v["pass"] for v in verdict["p1"].values())
    verdict["p2_pass"] = all(res["arms"][a][str(s)]["p2_ok"] for a in ARMS for s in SEEDS)
    verdict["p2_by_seed"] = {f"{a}/s{s}": res["arms"][a][str(s)]["p2_ok"]
                             for a in ARMS for s in SEEDS}
    verdict["all_pass"] = verdict["p1_pass"] and verdict["p2_pass"]
    res["verdict"] = verdict

    # ---- P3 ----
    for a in ARMS:
        for s in SEEDS:
            m = json.loads((HERE / "results" / f"p2_{a}_s{s}.json").read_text())["extra"]
            res["timing"][f"{a}/s{s}"] = m["timing"]
    print("P3 " + json.dumps(res["timing"], ensure_ascii=False), flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=1,
                                         default=float), encoding="utf-8")
    print("VERDICT " + json.dumps(verdict, ensure_ascii=False, default=float), flush=True)
    print(f"out={args.out}", flush=True)
    print("CORE_NDB_EVAL_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
