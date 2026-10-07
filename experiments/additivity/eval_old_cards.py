"""E-B P1 复核：挂新卡（bypass + negation 卡）是否干扰四张老卡 —— 用 MultiTaskEngine 实测。

三个条件（同一基座、同一固定评测集 = 各卡 val split，seed=--seed）：

- **A 不挂新卡**：`MultiTaskEngine(base, cards_dir=cards)`，4 张老卡，无旁路；
- **B 挂新卡**：同引擎 + 旁路（装 bypass 卡的权重）+ attach negation 卡；
  老卡评测时**旁路关闸**（结构性保证：老卡路径不经过旁路），negation 评测时开闸；
- **C 敏感性对照**：同 B，但旁路对老卡**强制开闸** —— 若 A vs B 的"逐位一致"是空的，
  这一档应当看不出差别；C 与 A 不同 ⇒ 一致性检查确实测到了东西。

输出：
  1. 逐卡 exact_match 的 A/B/C 与 Δ（P1 逐卡对噪声带）；
  2. 逐样本 `engine.predict` 输出哈希的 A vs B（要求 100% 一致）与 A vs C（预期不一致）；
  3. negation 在引擎路径上的指标（与训练脚本的评估交叉核对）。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.tasks.engine import MultiTaskEngine  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task  # noqa: E402
from bypass import BypassSet  # noqa: E402

# 四张老卡的固定评测集口径 = A′/N5 训练时的 task_samples（val = 前 max(200, n//10)）
OLD_SAMPLES = {"pronoun": 6000, "sentiment": 32000, "relation": 6000, "person": 6000}
NOISE_BAND = {"pronoun": 0.0283, "relation": 0.0139, "sentiment": 0.0041, "person": 0.0033}


def val_split(card, samples: int, seed: int):
    data = card.build_dataset(samples)
    random.Random(seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    return data[:n_val]


def hashes(engine, task: str, texts: list[str], bypass=None, gate: bool | None = None) -> list[str]:
    if bypass is not None and gate is not None:
        bypass.enabled = gate
    out = []
    for t in texts:
        text = t if isinstance(t, str) else t["text"]   # val split 是样本 dict
        r = engine.predict(text, tasks=[task])
        out.append(hashlib.sha256(
            json.dumps(r, sort_keys=True, ensure_ascii=False).encode()).hexdigest())
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="E-B P1：挂新卡是否干扰老卡（MultiTaskEngine 实测）")
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--cards-dir", default="checkpoints/cards")
    ap.add_argument("--bypass-card", required=True, help="训练产出的 negation 卡（含 extra.bypass）")
    ap.add_argument("--seed", type=int, default=42, help="评测集划分 seed（各卡 val split）")
    ap.add_argument("--negation-samples", type=int, default=6000)
    ap.add_argument("--max-predict", type=int, default=0, help=">0 时逐样本哈希只跑前 N 条")
    ap.add_argument("--skip-predict", action="store_true", help="只跑指标，不跑逐样本哈希")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cards = resolve_tasks(list(OLD_SAMPLES))
    neg_card = resolve_tasks(["negation"])["negation"]

    # 固定评测集（A/B/C 三条件共用同一份，逐位可比）
    vals = {name: val_split(cards[name], n, args.seed) for name, n in OLD_SAMPLES.items()}
    vals["negation"] = val_split(neg_card, args.negation_samples, args.seed)
    print(f"[data] seed={args.seed} " + " ".join(f"{k}={len(v)}" for k, v in vals.items()))

    def loaders(names):
        tok = NanoCharTokenizer()
        return {n: DataLoader(GenericTaskDataset(vals[n], tok, cards[n].spec),
                              batch_size=64, shuffle=False) for n in names}

    result: dict = {"seed": args.seed, "conditions": {}}
    pred_hash: dict[str, dict[str, list[str]]] = {"A": {}, "B": {}, "C": {}}
    metric_store: dict[str, dict] = {}

    # ---------------- 条件 A：不挂新卡 ----------------
    print("=== 条件 A：不挂新卡（4 张老卡，无旁路）===")
    eng_a = MultiTaskEngine(args.base, cards_dir=args.cards_dir, device=str(device))
    print(f"  attached={eng_a.attached}")
    old_loaders = loaders(OLD_SAMPLES)
    metric_store["A"] = {}
    for name in OLD_SAMPLES:
        t = time.perf_counter()
        metric_store["A"][name] = evaluate_task(eng_a.doc_encoder, eng_a.decoders[name],
                                                old_loaders[name], device, eng_a.specs[name])
        print(f"  A {name}: em={metric_store['A'][name]['exact_match']:.4f} "
              f"({time.perf_counter()-t:.1f}s)")
    if not args.skip_predict:
        for name in OLD_SAMPLES:
            tx = vals[name][: args.max_predict or None]
            t = time.perf_counter()
            pred_hash["A"][name] = hashes(eng_a, name, tx, None)
            print(f"  A predict-hash {name}: n={len(tx)} ({time.perf_counter()-t:.1f}s)")
    del eng_a
    if device.type == "cuda":
        torch.cuda.empty_cache()

    # ---------------- 条件 B / C：挂新卡 ----------------
    ck = torch.load(args.bypass_card, map_location="cpu", weights_only=False)
    meta = ck.get("extra", {}).get("bypass")
    if meta is None:
        raise SystemExit(f"{args.bypass_card} 里没有 extra.bypass，这不是 bypass 卡")
    print(f"=== 条件 B：挂新卡（{args.bypass_card}，rank={meta['rank']} alpha={meta['alpha']}）===")
    eng = MultiTaskEngine(args.base, cards_dir=args.cards_dir, device=str(device))
    byp = BypassSet(eng.doc_encoder, rank=meta["rank"], alpha=meta["alpha"])
    byp.load(meta["state_dict"])
    byp = byp.to(device)
    byp.disable()                       # 默认关闸（老卡路径不经过旁路）
    neg_name = eng.attach(args.bypass_card)
    print(f"  attached={eng.attached} | bypass sites={byp.sites} "
          f"params={byp.n_params()} | 关闸状态 enabled={byp.enabled}")

    metric_store["B"] = {}
    for name in OLD_SAMPLES:
        byp.disable()                   # 老卡：结构性关闸
        t = time.perf_counter()
        metric_store["B"][name] = evaluate_task(eng.doc_encoder, eng.decoders[name],
                                                old_loaders[name], device, eng.specs[name])
        print(f"  B {name}: em={metric_store['B'][name]['exact_match']:.4f} "
              f"({time.perf_counter()-t:.1f}s)")
    # negation（开闸）
    byp.enable()
    neg_loader = DataLoader(GenericTaskDataset(vals["negation"], eng.tokenizer,
                                               eng.specs[neg_name]), batch_size=64, shuffle=False)
    metric_store["B"]["negation"] = evaluate_task(eng.doc_encoder, eng.decoders[neg_name],
                                                  neg_loader, device, eng.specs[neg_name])
    print(f"  B negation(开闸): em={metric_store['B']['negation']['exact_match']:.4f} "
          f"bg_fp={metric_store['B']['negation']['bg_fp']:.4f}")

    if not args.skip_predict:
        for name in OLD_SAMPLES:
            tx = vals[name][: args.max_predict or None]
            t = time.perf_counter()
            pred_hash["B"][name] = hashes(eng, name, tx, bypass=byp, gate=False)   # 关闸逐样本
            print(f"  B predict-hash {name}: n={len(tx)} ({time.perf_counter()-t:.1f}s)")

    # 条件 C：敏感性对照（老卡也开闸）
    print("=== 条件 C：敏感性对照（旁路对老卡强制开闸）===")
    metric_store["C"] = {}
    for name in OLD_SAMPLES:
        byp.enable()
        metric_store["C"][name] = evaluate_task(eng.doc_encoder, eng.decoders[name],
                                                old_loaders[name], device, eng.specs[name])
        print(f"  C {name}: em={metric_store['C'][name]['exact_match']:.4f}")
    if not args.skip_predict:
        for name in OLD_SAMPLES:
            tx = vals[name][: args.max_predict or None]
            pred_hash["C"][name] = hashes(eng, name, tx, bypass=byp, gate=True)
            print(f"  C predict-hash {name}: n={len(tx)}")
    byp.disable()

    # ---------------- 汇总 ----------------
    print("\n=== P1：逐卡 Δ（B − A）对**各自**噪声带 ===")
    p1 = {}
    for name in OLD_SAMPLES:
        a = metric_store["A"][name]["exact_match"]
        b = metric_store["B"][name]["exact_match"]
        c = metric_store["C"][name]["exact_match"]
        band = NOISE_BAND[name]
        delta = b - a
        ident = (pred_hash["A"].get(name) == pred_hash["B"].get(name)
                 if name in pred_hash["A"] else None)
        n_diff_c = (sum(x != y for x, y in zip(pred_hash["A"][name], pred_hash["C"][name]))
                    if name in pred_hash["A"] else None)
        p1[name] = {"A": a, "B": b, "C": c, "delta_B-A": delta, "noise_band": band,
                    "within_band": abs(delta) <= band,
                    "metrics_equal": metric_store["A"][name] == metric_store["B"][name],
                    "pred_identical": ident, "C_differs_n": n_diff_c}
        print(f"  {name:10s} A={a:.4f} B={b:.4f} Δ={delta:+.6f} (带 {band}) "
              f"带内={abs(delta) <= band} 指标逐位同={metric_store['A'][name] == metric_store['B'][name]} "
              f"预测逐位同={ident} | C 与 A 不同的样本数={n_diff_c}")

    result["conditions"] = {k: {n: {kk: vv for kk, vv in m.items()}
                                for n, m in v.items()} for k, v in metric_store.items()}
    result["p1"] = p1
    result["bypass_card"] = args.bypass_card
    result["negation"] = metric_store["B"]["negation"]

    # 引擎路径的 negation 与训练脚本评估的交叉核对（同权重同 val 同实现 ⇒ 应当一致）
    print(f"[negation@engine] {json.dumps({k: v for k, v in metric_store['B']['negation'].items() if isinstance(v, float)}, ensure_ascii=False)}")

    out = Path(args.out) if args.out else Path(__file__).parent / "old_card_check.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[save] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
