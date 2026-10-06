"""core_keep 统一评测：四档 × 2 seed 的五卡 exact，与冻结基线配对，判 P1/P2。

- 评测集 = `capability_map/probe.split_of(cap, S)`（= 训练 val = `eval_S`，同源）
- **基线在同一份运行里现算**（`base_encoder` + `capability_map/cards/*_frozen_s{S}.pt`），
  并与 `capability_map/summary.json` 的 `frozen_exact` 对账（自检 4）；
- 判据逐字来自 `PREREG.md`：P1 逐卡用自己的噪声带，P2 = negation 0.870/0.831。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "capability_map"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.decoder.robust_ar_model import RobustARSliceDecoder  # noqa: E402
from dtseek.encoder.nano_doc_encoder import NanoDocEncoder  # noqa: E402
from dtseek.tasks.artifacts import build_card_decoder, read_card  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task  # noqa: E402

from prepare import TASK_SAMPLES  # noqa: E402
from probe import split_of  # noqa: E402
from train_core_keep import ARMS, ENCODER_KWARGS, FROZEN_CARD, OLD_CARDS, TASK_ORDER  # noqa: E402

#: PREREG §4.1 写死的逐卡噪声带（4 卡同配置仅换 seed 的实测差）
BAND = {"pronoun": 0.0283, "sentiment": 0.0041, "relation": 0.0139, "person": 0.0033}
#: PREREG §4 写死的 P2 线
P2_LINE = {42: 0.870, 43: 0.831}
SEEDS = (42, 43)


def enc_from(ck: dict, device) -> NanoDocEncoder:
    enc = NanoDocEncoder(vocab_size=NanoCharTokenizer().vocab_size,
                         hidden_dim=ck["hidden_dim"], dropout=0.0,
                         **ck["encoder_kwargs"]).to(device)
    enc.load_state_dict(ck["doc_encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards-dir", default=str(HERE / "cards"))
    ap.add_argument("--out", default=str(HERE / "results.json"))
    ap.add_argument("--arms", default=",".join(ARMS))
    args = ap.parse_args(argv)
    arms = [a for a in args.arms.split(",") if a]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cards = resolve_tasks(TASK_ORDER)
    tok = NanoCharTokenizer()
    recorded = json.loads((ROOT / "experiments" / "capability_map" / "summary.json")
                          .read_text(encoding="utf-8"))["rows"]

    res: dict = {"band": BAND, "p2_line": P2_LINE, "seeds": {}}
    ok_all = True

    for S in SEEDS:
        vals = {n: split_of(n, S)[0] for n in TASK_ORDER}
        loaders = {n: DataLoader(GenericTaskDataset(vals[n], tok, cards[n].spec),
                                 batch_size=64, shuffle=False) for n in TASK_ORDER}
        print(f"[seed {S}] " + " ".join(f"{n}={len(vals[n])}" for n in TASK_ORDER), flush=True)

        # ---- 自检 4：基线现算 vs capability_map 记录 ----
        bck = torch.load(ROOT / "checkpoints" / "base_encoder.pt",
                         map_location="cpu", weights_only=False)
        enc = NanoDocEncoder(vocab_size=tok.vocab_size, hidden_dim=bck["hidden_dim"],
                             dropout=0.0, **bck["encoder_kwargs"]).to(device)
        enc.load_state_dict(bck["doc_encoder"])
        enc.eval()
        base = {}
        for n in OLD_CARDS:
            dec, _ = build_card_decoder(read_card(ROOT / FROZEN_CARD.format(n, S)), device)
            m = evaluate_task(enc, dec, loaders[n], device, cards[n].spec)
            base[n] = m["exact_match"]
            rec = recorded[n][str(S)]["frozen_exact"]
            same = abs(base[n] - rec) < 1e-12
            print(f"SELFTEST_4 s{S} {n:9s} 基线现算={base[n]:.6f} 记录={rec:.6f} "
                  f"一致={same}", flush=True)
            ok_all &= same
            del dec
        del enc
        if device.type == "cuda":
            torch.cuda.empty_cache()

        # ---- 从头联合锚点（复用 capability_map 的 align 产物）----
        anchor = {n: recorded[n][str(S)]["joint_exact"] for n in TASK_ORDER}

        seed_rows = {"baseline": base, "anchor_scratch_joint": anchor, "arms": {}}
        for arm in arms:
            p = Path(args.cards_dir) / f"{arm}_s{S}.pt"
            if not p.exists():
                print(f"[skip] {p} 不存在", flush=True)
                continue
            ck = torch.load(p, map_location="cpu", weights_only=False)
            e = enc_from(ck, device)
            decs = {}
            for n in TASK_ORDER:
                d = RobustARSliceDecoder(hidden_dim=ck["hidden_dim"],
                                         num_classes=cards[n].spec.num_classes,
                                         **ck["decoder_kwargs"]).to(device)
                d.load_state_dict(ck["decoders"][n], strict=True)
                decs[n] = d
            ex, dd = {}, {}
            for n in TASK_ORDER:
                m = evaluate_task(e, decs[n], loaders[n], device, cards[n].spec)
                ex[n] = m["exact_match"]
                if n in OLD_CARDS:
                    dd[n] = ex[n] - base[n]
            p1 = all(abs(dd[n]) <= BAND[n] for n in OLD_CARDS)
            p2 = ex["negation"] >= P2_LINE[S]
            seed_rows["arms"][arm] = {"exact": ex, "delta_vs_frozen": dd,
                                      "p1": p1, "p2": p2,
                                      "pass": p1 and p2}
            ok_all &= p1 and p2
            del e, decs
            if device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[P] s{S} {arm}: P1={p1} P2={p2} | " +
                  " ".join(f"{n}={ex[n]:.4f}(Δ{dd.get(n, float('nan')):+.4f})"
                           for n in TASK_ORDER if n in OLD_CARDS) +
                  f" | negation={ex['negation']:.4f} (线 {P2_LINE[S]})", flush=True)

        res["seeds"][str(S)] = seed_rows

    res["selftest4_ok"] = ok_all
    Path(args.out).write_text(json.dumps(res, ensure_ascii=False, indent=2, default=float),
                              encoding="utf-8")
    print(f"[save] {args.out}", flush=True)

    # ---- 主表 ----
    print("\n=== 主表（exact；Δ = 相对冻结基线）===")
    hdr = ["arm", "seed"] + [f"{n}" for n in TASK_ORDER]
    print("| " + " | ".join(hdr) + " | P1 | P2 |")
    print("|" + "---|" * len(hdr) + "---|---|")
    for S in SEEDS:
        row = res["seeds"].get(str(S))
        if not row:
            continue
        b, a = row["baseline"], row["anchor_scratch_joint"]
        print("| (冻结基线) | " + str(S) + " | " +
              " | ".join(f"{b[n]:.4f}" if n in b else "—" for n in TASK_ORDER) +
              " | — | — |")
        print("| (从头联合锚点) | " + str(S) + " | " +
              " | ".join(f"{a[n]:.4f}" for n in TASK_ORDER) + " | — | — |")
        for arm, r in row["arms"].items():
            print(f"| {arm} | {S} | " +
                  " | ".join(f"{r['exact'][n]:.4f}" for n in TASK_ORDER) +
                  f" | {'✅' if r['p1'] else '❌'} | {'✅' if r['p2'] else '❌'} |")
    print("EVAL_DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
