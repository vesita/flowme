"""core_keep 自检 0（一次性）：温启动起点是否**精确等于** P1 参照基线。

不训练任何东西，只做两件事：
  1. 按 train_core_keep 的温启动方式（核 ← base_encoder，四张老卡头 ← capability_map frozen 卡）
     重建模型，在同一份 `eval_S` 上评四张老卡的 exact，与
     `experiments/capability_map/summary.json` 的 `frozen_exact` 逐卡对比；
  2. 打印每个老卡任务在 **eval 模式**与 **train 模式（dropout=0.1 开）** 下的
     `task_loss` 均值 —— 用来解释"起点 CE 为什么高"（train_multitask 的核 dropout=0.1，
     而 frozen 卡是在 dropout=0 的冻结核上训的）。
"""
from __future__ import annotations

import json
import pickle
import random
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
from dtseek.tasks.runtime import GenericTaskDataset, evaluate_task, task_loss  # noqa: E402

from prepare import CACHE as CAPMAP_CACHE, TASK_SAMPLES  # noqa: E402
from probe import split_of  # noqa: E402
from train_core_keep import (  # noqa: E402
    DECODER_KWARGS, ENCODER_KWARGS, HIDDEN_DIM, OLD_CARDS, TASK_ORDER, FROZEN_CARD,
)

BASELINE = json.loads((ROOT / "experiments" / "capability_map" / "summary.json")
                      .read_text(encoding="utf-8"))["rows"]


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cards = resolve_tasks(TASK_ORDER)
    tok = NanoCharTokenizer()
    out = {}
    for seed in (42, 43):
        torch.manual_seed(seed)
        random.seed(seed)
        enc = NanoDocEncoder(vocab_size=tok.vocab_size, hidden_dim=HIDDEN_DIM,
                             dropout=0.1, **ENCODER_KWARGS).to(device)
        decs = {n: RobustARSliceDecoder(hidden_dim=HIDDEN_DIM,
                                        num_classes=cards[n].spec.num_classes,
                                        **DECODER_KWARGS).to(device)
                for n in TASK_ORDER}
        bck = torch.load(ROOT / "checkpoints" / "base_encoder.pt",
                         map_location="cpu", weights_only=False)
        enc.load_state_dict(bck["doc_encoder"], strict=True)
        for n in OLD_CARDS:
            decs[n].load_state_dict(read_card(ROOT / FROZEN_CARD.format(n, seed))["decoder"],
                                    strict=True)

        row = {}
        for n in OLD_CARDS:
            ev, _ = split_of(n, seed)
            loader = DataLoader(GenericTaskDataset(ev, tok, cards[n].spec),
                                batch_size=64, shuffle=False)
            m = evaluate_task(enc, decs[n], loader, device, cards[n].spec)
            # eval 模式 CE（与 frozen 卡训练时同口径：核 dropout 关）
            enc.eval()
            decs[n].eval()
            ce_eval = _ce(enc, decs[n], loader, device, cards[n].spec)
            # train 模式 CE（train_multitask 训练时的口径：核 dropout=0.1 开）
            enc.train()
            decs[n].train()
            with torch.no_grad():
                ce_train = _ce(enc, decs[n], loader, device, cards[n].spec)
            enc.eval()
            base = BASELINE[n][str(seed)]["frozen_exact"]
            row[n] = {"exact": m["exact_match"], "baseline": base,
                      "delta": m["exact_match"] - base,
                      "ce_eval": ce_eval, "ce_train": ce_train}
            print(f"SELFTEST_0 s{seed} {n:9s} exact={m['exact_match']:.6f} "
                  f"baseline={base:.6f} Δ={m['exact_match']-base:+.2e} "
                  f"ce_eval={ce_eval:.4f} ce_train={ce_train:.4f}", flush=True)
        out[str(seed)] = row
    (HERE / "init_check.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print("[save] experiments/core_keep/init_check.json")
    print("INIT_CHECK_DONE")
    return 0


def _ce(enc, dec, loader, device, spec) -> float:
    tot = 0.0
    n = 0
    for batch in loader:
        inp = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        mem = enc(inp, attention_mask=mask)
        tot += float(task_loss(dec, mem, mask, batch, spec, device))
        n += 1
    return tot / max(1, n)


if __name__ == "__main__":
    sys.exit(main())
