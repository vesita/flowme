"""E-B 机制探针：训练好的旁路到底"改了多少表征"、"干了多少活"。

三件事（纯前向 + 评估，不再训练）：
  1. 逐插入点的**相对扰动**：||scale·(h Aᵀ Bᵀ)|| / ||h||（在 negation val 批上平均）
     —— 太小（<1%）⇒ 旁路没起作用 / 优化没到位；很大但指标不动 ⇒ 容量/插入点问题；
  2. 最终 doc_memory 的 gate on vs off 差异（整体改变了多少）；
  3. 同一张训练好的卡：**开闸 vs 关闸**的指标 —— 关闸若掉回 frozen 水平，
     说明那几个点的提升确实是旁路在干活（而不是头自己训得更好）。
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task  # noqa: E402
from bypass import BypassSet  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--card", required=True)
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--seed", type=int, default=42, help="val 划分 seed（与训练一致）")
    ap.add_argument("--samples", type=int, default=6000)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, base_ck = load_base_encoder(args.base, device)
    ck = torch.load(args.card, map_location="cpu", weights_only=False)
    meta = ck["extra"]["bypass"]
    # auto_hooks=False：先挂统计 hook，再挂真实旁路 hook ⇒ 统计拿到的是**未加工**的残差流
    byp = BypassSet(enc, rank=meta["rank"], alpha=meta["alpha"], auto_hooks=False)
    byp.load(meta["state_dict"])
    byp = byp.to(device)

    from dtseek.tasks.artifacts import build_card_decoder
    decoder, spec = build_card_decoder(ck, device)

    card = resolve_tasks(["negation"])["negation"]
    data = card.build_dataset(args.samples)
    random.Random(args.seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    tok = NanoCharTokenizer()
    loader = DataLoader(GenericTaskDataset(data[:n_val], tok, spec), batch_size=64, shuffle=False)

    # ---- 1/2. 扰动统计 ---------------------------------------------------
    per_site: dict[str, list[float]] = {s: [] for s in byp.sites}
    rel_mem = []

    def make_hook(idx: int):
        def hook(_m, _i, out):
            d = byp.adapters[idx](out) - out
            if byp.enabled:
                per_site[byp.sites[idx]].append(
                    float((d.norm(dim=-1) / out.norm(dim=-1).clamp_min(1e-9)).mean()))
            return out if not byp.enabled else byp.adapters[idx](out)
        return hook

    handles = []
    # 统计 hook 先注册（拿到的是**未加工**的残差流），BypassSet 的真实变换 hook 后注册 ⇒ 先统计后变换
    for i, blk in enumerate(enc.blocks):
        handles.append(blk.register_forward_hook(make_hook(i)))
    handles.append(enc.norm.register_forward_hook(make_hook(len(enc.blocks))))
    byp.attach_hooks()

    byp.disable()
    with torch.no_grad():
        mem_off = enc(next(iter(loader))["input_ids"].to(device),
                      attention_mask=next(iter(loader))["attention_mask"].to(device))
    byp.enable()
    with torch.no_grad():
        batch = next(iter(loader))
        mem_on = enc(batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
    for h in handles:
        h.remove()              # 只摘统计 hook；旁路自身的 hook 留着给评估用

    rel = {s: (sum(v) / len(v) if v else 0.0) for s, v in per_site.items()}
    rel_all = float((mem_on - mem_off).norm() / mem_off.norm())

    # ---- 3. 开闸 / 关闸 指标 --------------------------------------------
    res = {}
    for gate in (True, False):
        byp.enabled = gate
        m = evaluate_task(enc, decoder, loader, device, spec)
        res["on" if gate else "off"] = m
        print(f"[probe] gate={'on' if gate else 'off'} em={m['exact_match']:.4f} "
              f"cls={m['cls_acc']:.4f} span={m['span_hit']:.4f} bg_fp={m['bg_fp']:.4f}")
    byp.disable()

    out = {
        "card": args.card, "base": args.base, "seed": args.seed,
        "per_site_rel_perturbation": rel,
        "final_memory_rel_change": rel_all,
        "metrics": res,
        "delta_em_gate_on_minus_off": res["on"]["exact_match"] - res["off"]["exact_match"],
        "bypass_stats": byp.stats(),
    }
    print("PROBE " + json.dumps(out, ensure_ascii=False))
    if args.out:
        Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2))
        print(f"[save] {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
