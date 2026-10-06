"""三档评测：冻结档 / 旁路档 / 联合档，全部落在同一份 held-out `eval_S` 上。

- 冻结档、旁路档：`experiments/capability_map/cards/{cap}_{mode}_s{S}.pt`
  （negation 复用 `experiments/additivity/cards/negation_{mode}_base_s{S}.pt`）
- 联合档：`checkpoints/arm_neg5_seed{S}.pt`（**复用已有 ckpt**，只重新评一遍）

对齐自检（PREREG §3）：联合档在本脚本算出的 `cls_acc/n_cls/n_bg` 必须与
`checkpoints/arm_neg5_seed{S}_metrics.json` **逐位相同**，否则判「对齐失败」。
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "additivity"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402

from dtseek.tasks.artifacts import build_card_decoder, load_base_encoder, read_card  # noqa: E402
from dtseek.tasks.plugin import resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import evaluate_task  # noqa: E402

from bypass import BypassSet  # noqa: E402
from prepare import CAPS, SEEDS  # noqa: E402
from probe import build_compose_ood, make_loader, split_of  # noqa: E402

CARDS_DIR = HERE / "cards"
NEG_DIR = ROOT / "experiments" / "additivity" / "cards"


def card_path(cap: str, mode: str, seed: int) -> Path:
    if cap == "negation":
        return NEG_DIR / f"negation_{mode}_base_s{seed}.pt"
    return CARDS_DIR / f"{cap}_{mode}_s{seed}.pt"


def attach_bypass(enc, ck: dict, device) -> BypassSet | None:
    meta = ck.get("extra", {}).get("bypass")
    if meta is None:
        return None
    byp = BypassSet(enc, rank=meta["rank"], alpha=meta["alpha"])
    byp.load(meta["state_dict"])
    byp = byp.to(device)
    byp.enable()          # frozen 档 B 恒为 0 ⇒ 开闸也逐位等于冻结基线
    return byp


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--joint", default="checkpoints/arm_neg5_seed{}.pt")
    ap.add_argument("--caps", default=",".join(CAPS))
    ap.add_argument("--out", default=str(HERE / "eval.json"))
    args = ap.parse_args(argv)
    caps = [c for c in args.caps.split(",") if c]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[eval] device={device} caps={caps} seeds={SEEDS}", flush=True)
    cards = resolve_tasks(caps)
    ood_neg, ood_sent, ood_meta = build_compose_ood()
    print(f"[ood] {ood_meta}", flush=True)

    result: dict = {"config": {"base": args.base, "joint": args.joint},
                    "ood_meta": ood_meta, "caps": {}}
    align_ok = True

    for cap in caps:
        spec = cards[cap].spec
        entry: dict = {}
        for S in SEEDS:
            ev, _tr = split_of(cap, S)
            loader = make_loader(ev, spec, shuffle=False)
            row: dict = {"n_eval": len(ev)}

            # ---- 冻结档 / 旁路档（同一 base_enc，bypass 开闸）----
            enc, _ = load_base_encoder(args.base, device)
            for mode in ("frozen", "bypass"):
                p = card_path(cap, mode, S)
                if not p.exists():
                    row[mode] = {"missing": str(p)}
                    print(f"[eval] 缺 {p}", flush=True)
                    continue
                ck = read_card(p)
                dec, cspec = build_card_decoder(ck, device)
                byp = attach_bypass(enc, ck, device)
                m = evaluate_task(enc, dec, loader, device, cspec)
                row[mode] = {k: v for k, v in m.items()}
                print(f"[eval] {cap} s{S} {mode}: cls_acc={m['cls_acc']:.4f} "
                      f"exact={m['exact_match']:.4f} bg_fp={m['bg_fp']:.4f}", flush=True)
                if byp is not None:
                    byp.close(enc)
                del dec
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            del enc
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # ---- 联合档（复用已有 ckpt）----
            jck = torch.load(args.joint.format(S), map_location="cpu", weights_only=False)
            jenc = _enc_from_joint(jck, device)
            dec = _dec_from_joint(jck, cap, spec, device)
            m = evaluate_task(jenc, dec, loader, device, spec)
            row["joint"] = {k: v for k, v in m.items()}
            print(f"[eval] {cap} s{S} joint: cls_acc={m['cls_acc']:.4f} exact={m['exact_match']:.4f}",
                  flush=True)

            # ---- 对齐自检：与 arm_neg5 原始 metrics.json 逐位比 ----
            mj = json.loads((ROOT / "checkpoints" / f"arm_neg5_seed{S}_metrics.json")
                            .read_text(encoding="utf-8"))[cap]
            same = all(m[k] == mj[k] for k in ("cls_acc", "n_cls", "n_bg", "exact_match"))
            align_ok &= same
            row["align_check"] = {"match": same,
                                  "recomputed": {k: m[k] for k in ("cls_acc", "n_cls", "n_bg", "exact_match")},
                                  "from_json": {k: mj[k] for k in ("cls_acc", "n_cls", "n_bg", "exact_match")}}
            print(f"[ALIGN] {cap} s{S} match={same}", flush=True)
            del jenc, dec
            if device.type == "cuda":
                torch.cuda.empty_cache()

            # ---- OOD：compose_ops 60（有金标跨度，三档都能评）----
            ood = ood_neg if cap == "negation" else (ood_sent if cap == "sentiment" else [])
            if ood:
                ol = make_loader(ood, spec, shuffle=False)
                orow: dict = {"n": len(ood)}
                enc, _ = load_base_encoder(args.base, device)
                for mode in ("frozen", "bypass"):
                    p = card_path(cap, mode, S)
                    if not p.exists():
                        continue
                    ck = read_card(p)
                    dec2, cspec = build_card_decoder(ck, device)
                    byp = attach_bypass(enc, ck, device)
                    mo = evaluate_task(enc, dec2, ol, device, cspec)
                    orow[mode] = {k: v for k, v in mo.items()}
                    if byp is not None:
                        byp.close(enc)
                    del dec2
                del enc
                jck = torch.load(args.joint.format(S), map_location="cpu", weights_only=False)
                jenc = _enc_from_joint(jck, device)
                dj = _dec_from_joint(jck, cap, spec, device)
                mo = evaluate_task(jenc, dj, ol, device, spec)
                orow["joint"] = {k: v for k, v in mo.items()}
                del jenc, dj
                row["ood_compose60"] = orow
                print(f"[eval] {cap} s{S} OOD60: " +
                      " ".join(f"{k}={v['cls_acc']:.4f}/{v['exact_match']:.4f}"
                               for k, v in orow.items() if isinstance(v, dict)), flush=True)
                if device.type == "cuda":
                    torch.cuda.empty_cache()

            entry[str(S)] = row
        result["caps"][cap] = entry

    result["align_all_ok"] = align_ok
    print(f"[ALIGN] all_ok={align_ok}", flush=True)
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2, default=float))
    print(f"[save] {args.out}")
    print("EVAL_DONE")
    return 0 if align_ok else 2


def _dec_from_joint(jck: dict, cap: str, spec, device):
    """从 5 卡联合 ckpt 重建某张任务卡的解码头（ckpt 里存的是 state_dict，不是模块）。"""
    from dtseek.decoder.robust_ar_model import RobustARSliceDecoder
    sd = jck["decoders"][cap]
    dec = RobustARSliceDecoder(hidden_dim=jck["hidden_dim"], num_classes=spec.num_classes,
                               **dict(jck["decoder_kwargs"])).to(device)
    dec.load_state_dict(sd, strict=True)   # 对不上就响亮抛错
    assert dec.cls_head.out_features == spec.num_classes, (
        f"{cap}: ckpt 头类别 {dec.cls_head.out_features} != spec {spec.num_classes}")
    dec.eval()
    return dec


def _enc_from_joint(jck: dict, device):
    from dtseek.encoder.nano_doc_encoder import NanoDocEncoder
    from nano_char_tokenizer import NanoCharTokenizer
    enc = NanoDocEncoder(vocab_size=NanoCharTokenizer().vocab_size,
                         hidden_dim=jck["hidden_dim"], dropout=0.0,
                         **jck["encoder_kwargs"]).to(device)
    enc.load_state_dict(jck["doc_encoder"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc


if __name__ == "__main__":
    sys.exit(main())
