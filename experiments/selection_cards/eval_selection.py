"""按 `PREREG.md §3` 逐条算 P1–P6：部署口径（`MultiTaskEngine.predict`）逐样本推理。

用法：
    uv run python experiments/selection_cards/eval_selection.py \
        --card reply_pick --ckpt checkpoints/cards/reply_pick.pt --sets E1,E2,E3,E4
    uv run python experiments/selection_cards/eval_selection.py \
        --card cloze_fill --ckpt checkpoints/cards/cloze_fill.pt --sets C1,C2

三条口径纪律：
  1. 部署口径与批量口径**对账**（PREREG §4）：前 200 条用 `evaluate_task` 同款
     批量前向复算，(类别, s0, e0) 三元组必须逐条一致，一致率 < 100% 直接退出；
  2. 先跑 `sanity_check`（探针帧已知答案对照），不过就别报指标；
  3. 指标分层打印：全量 / 按答案位置 / 按空位位置 / 按难度档（E3 vs E4）。
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from dtseek.tasks.builtin.cloze_fill.dataset import build_cloze_dataset
from dtseek.tasks.builtin.reply_pick.dataset import build_reply_pick_dataset
from dtseek.tasks.engine import MultiTaskEngine
from dtseek.tasks.plugin import resolve_tasks
from dtseek.tasks.probe import run_probe, sanity_check
from dtseek.tasks.runtime import GenericTaskDataset

#: 评测集定义（与 PREREG §2 逐字对应）
SET_DEFS: dict[str, dict] = {
    "E1": {"card": "reply_pick", "split": "eval", "seed": 4242, "target_samples": 1500},
    "E2": {"card": "reply_pick", "split": "eval", "seed": 4242, "target_samples": 1500,
           "fixed_answer_pos": 1},
    "E3": {"card": "reply_pick", "split": "eval", "seed": 5252, "target_samples": 800,
           "tiers": (2,)},
    "E4": {"card": "reply_pick", "split": "eval", "seed": 6262, "target_samples": 800,
           "tiers": (1,)},
    "C1": {"card": "cloze_fill", "split": "eval", "seed": 4242, "target_samples": 1500},
    "C2": {"card": "cloze_fill", "split": "eval", "seed": 4242, "target_samples": 1500,
           "fixed_answer_pos": 1},
}


def build_set(name: str, max_lines: int) -> list[dict]:
    kw = dict(SET_DEFS[name])
    card = kw.pop("card")
    if card == "reply_pick":
        return build_reply_pick_dataset(max_lines=max_lines, **kw)
    return build_cloze_dataset(max_lines=max_lines, **kw)


def gold_of(item: dict) -> tuple[int, tuple[int, int] | None]:
    """(真值类别, 真值区间闭区间)。背景样本 = (0, None)。"""
    if not item["spans"]:
        return 0, None
    sp = item["spans"][0]
    return sp["label"], (sp["start"], sp["end"] - 1)


def engine_triple(engine: MultiTaskEngine, text: str, task: str) -> tuple[int, tuple | None]:
    out = engine.predict(text, tasks=[task])
    anchors = out["tasks"][task]
    if not anchors:
        return 0, None
    a = anchors[0]
    return a["class_id"], (a["s0"], a["e0"])


def batch_triples(engine: MultiTaskEngine, items: list[dict], task: str,
                  device: torch.device, batch_size: int = 64) -> list[tuple]:
    """批量前向口径（`evaluate_task` 首步同款），与部署口径对账。"""
    spec = engine.specs[task]
    ds = GenericTaskDataset(items, engine.tokenizer, spec)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    engine.doc_encoder.eval()
    out_triples: list[tuple] = []
    with torch.no_grad():
        for batch in loader:
            inp = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            doc_memory = engine.doc_encoder(inp, attention_mask=mask)
            dec = engine.decoders[task]
            dec.eval()
            B = inp.shape[0]
            out = dec.forward_step(dec.bos_query.expand(B, 1, -1), doc_memory, doc_mask=mask)
            cls = out["cls_logits"].argmax(-1)
            s = out["start_logits"].argmax(-1)
            e = out["end_logits"].argmax(-1)
            L = doc_memory.shape[1]
            for b in range(B):
                c = int(cls[b])
                if c == 0:
                    out_triples.append((0, None))
                    continue
                s0 = int(min(s[b], e[b]).clamp(0, L - 1))
                e0 = int(max(s[b], e[b]).clamp(0, L - 1))
                out_triples.append((c, (s0, e0)))
    return out_triples


def score_set(items: list[dict], triples: list[tuple[int, tuple | None]],
              card: str) -> dict:
    """PREREG §3 的全部逐样本指标。"""
    solvable, backgrounds = [], []
    for item, tri in zip(items, triples):
        g_cls, g_span = gold_of(item)
        (solvable if g_cls > 0 else backgrounds).append((item, tri, g_cls, g_span))

    # 1) 类别准确率（只看正解样本）
    cls_ok = sum(1 for _, (p, _), g, _ in solvable if p == g)
    cls_acc = cls_ok / max(1, len(solvable))

    # 2) 按答案位置分组（P1a）
    by_pos: dict[int, list[bool]] = defaultdict(list)
    for item, (p, _), g, _ in solvable:
        by_pos[item["meta"]["answer_pos"]].append(p == g)
    pos_acc = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_pos.items())}

    # 3) 拒答：误触发（正解样本被判成 0）与召回（背景判成 0）
    false_trigger = sum(1 for _, (p, _), _, _ in solvable if p == 0) / max(1, len(solvable))
    bg_recall = sum(1 for _, (p, _), _, _ in backgrounds if p == 0) / max(1, len(backgrounds))

    # 4) 锚点（P5）：类别对的样本里，区间逐字等于被选中候选的比例
    cls_right = [(t, g_span) for _, t, g, g_span in solvable if t[0] == g]
    anchor_hit = sum(1 for (p, sp), gs in cls_right if sp == gs) / max(1, len(cls_right))
    joint = sum(1 for _, (p, sp), g, gs in solvable if p == g and sp == gs) / max(1, len(solvable))

    # 5) 整句级（P4a）：切片多重集口径
    tp = fp = fn = exact = 0
    for item, tri in zip(items, triples):
        g_cls, g_span = gold_of(item)
        pred = [] if tri[0] == 0 else [tri[1]]
        gold = [] if g_cls == 0 else [g_span]
        ts, gs = set(gold), set(pred)
        exact += int(sorted(map(str, gs)) == sorted(map(str, ts)))
        tp += len(ts & gs)
        fp += len(gs - ts)
        fn += len(ts - gs)

    # 6) 预测类别分布（看是否恒选某个槽位）
    dist = Counter(t[0] for t in triples)

    # 7) cloze 额外：按空位位置分层（探索性）
    by_blank: dict[str, list[bool]] = defaultdict(list)
    for item, (p, _), g, _ in solvable:
        kind = item["meta"].get("blank_kind")
        if kind is not None:
            by_blank[kind].append(p == g)
    blank_acc = {k: round(sum(v) / len(v), 4) for k, v in sorted(by_blank.items())} \
        if card == "cloze_fill" else {}

    return {
        "n": len(items),
        "n_solvable": len(solvable),
        "n_background": len(backgrounds),
        "cls_acc": round(cls_acc, 4),
        "cls_acc_by_answer_pos": pos_acc,
        "pred_class_dist": {str(k): v for k, v in sorted(dist.items())},
        "bg_false_trigger": round(false_trigger, 4),
        "bg_recall": round(bg_recall, 4),
        "anchor_exact_given_cls_right": round(anchor_hit, 4),
        "joint_cls_and_span": round(joint, 4),
        "n_cls_right": len(cls_right),
        "exact_match": round(exact / max(1, len(items)), 4),
        "slice_precision": round(tp / max(1, tp + fp), 4),
        "slice_recall": round(tp / max(1, tp + fn), 4),
        **({"cls_acc_by_blank": blank_acc} if blank_acc else {}),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True, choices=["reply_pick", "cloze_fill"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--sets", required=True, help="逗号分隔，如 E1,E2,E3,E4")
    ap.add_argument("--max-lines", type=int, default=1_200_000)
    ap.add_argument("--agree", type=int, default=200, help="对账样本数（PREREG §4）")
    ap.add_argument("--strict-sanity", action="store_true",
                    help="sanity 不过就中止出数（默认只记录、继续出指标）")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    resolve_tasks([args.card])
    engine = MultiTaskEngine(auto_attach=False)
    name = engine.attach(args.ckpt)
    assert name == args.card, f"ckpt 是 {name}，不是 {args.card}"
    device = engine.device

    sanity = sanity_check(engine, args.card)
    if sanity["passed"]:
        print(f"sanity_check：passed={sanity['passed']}（已知答案对照全对）")
    else:
        print(f"sanity_check：passed={sanity['passed']}"
              "（模型答错已知答案 —— 与低于盲猜的指标互为佐证，继续出指标）")
    if not sanity["passed"]:
        print(json.dumps(sanity["cases"], ensure_ascii=False, indent=1))
        if args.strict_sanity:
            print("   !! --strict-sanity 下中止：先怀疑探针/权重，不报指标")
            return 2

    report: dict = {"card": args.card, "ckpt": args.ckpt,
                    "sanity_passed": sanity["passed"], "sets": {}}

    probe = run_probe(engine, args.card)
    report["probe"] = {k: probe[k] for k in
                       ("n_units", "class_acc", "span_acc", "failures") if k in probe}
    print(f"探针：n={probe['n_units']} class_acc={probe['class_acc']:.4f} "
          f"span_acc={probe['span_acc']:.4f}")

    for set_name in args.sets.split(","):
        set_name = set_name.strip()
        if set_name not in SET_DEFS or SET_DEFS[set_name]["card"] != args.card:
            print(f"!! 评测集 {set_name} 不属于 {args.card}，跳过")
            continue
        t0 = time.perf_counter()
        items = build_set(set_name, args.max_lines)
        triples = [engine_triple(engine, it["text"], args.card) for it in items]
        per_sample_sec = (time.perf_counter() - t0) / max(1, len(items))

        # 部署口径 vs 批量口径对账（PREREG §4）
        n_chk = min(args.agree, len(items))
        batch = batch_triples(engine, items[:n_chk], args.card, device)
        agree = sum(1 for a, b in zip(triples[:n_chk], batch) if a == b)
        agreement = agree / n_chk
        print(f"[{set_name}] n={len(items)} | 对账 {agree}/{n_chk} = {agreement:.4f} "
              f"| 单条 {per_sample_sec * 1000:.1f} ms")
        if agreement < 1.0:
            for i, (a, b) in enumerate(zip(triples[:n_chk], batch)):
                if a != b:
                    print(f"   口径不一致 #{i}: engine={a} batch={b} "
                          f"text={items[i]['text'][:40]!r}")
            print("   !! 部署口径与批量口径不一致 —— 指标不可信，先修口径")
            return 3

        sc = score_set(items, triples, args.card)
        sc["per_sample_sec"] = round(per_sample_sec, 5)
        sc["agreement"] = agreement
        report["sets"][set_name] = sc
        print(json.dumps(sc, ensure_ascii=False, indent=1))

    out = Path(args.out) if args.out else Path(
        f"experiments/selection_cards/results_{args.card}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"结果已写 {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
