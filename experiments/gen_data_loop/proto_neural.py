#!/usr/bin/env python3
"""原型探针：现有**神经指针卡**在真实语料句子上提议得怎么样（CPU，只读）。

§7 提案要求「候选由现有指针卡产出（区间 + 类别）」。本脚本测两件事：
  1. 速度：CPU 上跑多少句/秒（决定数据生成要不要等 GPU）；
  2. 质量：神经卡的发射与该卡**真值源探测器**的逐句一致率（≈ 在这批句上的精确率）。
只打印，不写文件。
"""
from __future__ import annotations

import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402

from dtseek.tasks.artifacts import build_card_decoder, load_base_encoder, read_card  # noqa: E402
from dtseek.tasks.builtin.negation.dataset import extract_negation_spans  # noqa: E402
from dtseek.tasks.builtin.pronoun.dataset import extract_all_spans  # noqa: E402
from dtseek.tasks.builtin.sentiment.dataset import extract_emotion_spans  # noqa: E402
from dtseek.tasks.corpus import CORPUS_GLOB, resolve_corpus_files  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, _rollout  # noqa: E402

SENT_SPLIT = re.compile(r"[。！？\n；;]+")
ROLE_PREFIX = re.compile(r"^(用户|模型|系统|提问|回答|User|Assistant)[:：]\s*")
CJK = re.compile(r"[一-龥]")
BAD = re.compile(r"[�\t｜|]|原文：|候选：")

N = 512
BATCH = 64


def cjk(s: str) -> int:
    return len(CJK.findall(s))


def ok(s: str) -> bool:
    if not (14 <= len(s) <= 44) or BAD.search(s):
        return False
    return cjk(s) >= 0.6 * len(s)


def sentences(n: int) -> list[str]:
    files = resolve_corpus_files(CORPUS_GLOB)
    random.Random(7).shuffle(files)
    out, seen = [], set()
    for p in files:
        with open(p, encoding="utf-8", errors="ignore") as fp:
            for line in fp:
                t = ROLE_PREFIX.sub("", line.strip())
                for s in SENT_SPLIT.split(t):
                    s = s.strip()
                    if s in seen or not ok(s):
                        continue
                    seen.add(s)
                    out.append(s)
                if len(out) >= n:
                    return out
    return out


def truth(card: str, text: str) -> list[tuple[int, int, int]]:
    if card == "pronoun":
        return [(s["label"], s["start"], s["end"]) for s in extract_all_spans(text)]
    if card == "sentiment":
        lab, spans = extract_emotion_spans(text)
        if lab == -1:
            return "REJECT"
        return [(s["label"], s["start"], s["end"]) for s in spans]
    return [(s["label"], s["start"], s["end"]) for s in extract_negation_spans(text)]


def main() -> int:
    sents = sentences(N)
    print(f"[proto] {len(sents)} 句", flush=True)
    device = torch.device("cpu")
    enc, bck = load_base_encoder("checkpoints/base_encoder.pt", device)
    tok = NanoCharTokenizer()
    print(f"[proto] encoder params={sum(p.numel() for p in enc.parameters()):,} "
          f"hidden={bck['hidden_dim']}", flush=True)

    for card in ("pronoun", "sentiment"):
        path = f"checkpoints/cards/{card}.pt"
        dec, spec = build_card_decoder(read_card(path), device)
        print(f"[proto] {card}: classes={tuple(c.name for c in spec.classes)} "
              f"max_steps={spec.max_steps} max_len={spec.max_len}", flush=True)
        n_out = 0
        trunc = 0
        agree = exact = total_sent = 0
        cls_hist: dict[str, int] = {}
        t0 = time.perf_counter()
        ds = GenericTaskDataset([{"text": s, "spans": []} for s in sents], tok, spec)
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH, shuffle=False)
        trunc = 0
        with torch.no_grad():
            bi = 0
            for batch in loader:
                inp = batch["input_ids"]
                mask = batch["attention_mask"]
                mem = enc(inp, attention_mask=mask)
                preds = _rollout(dec, mem, mask, inp.shape[0], spec, ndb=None, input_ids=inp)
                for k, pr in enumerate(preds):
                    s = sents[bi + k]
                    n_out += len(pr)
                    gt = truth(card, s)
                    if gt == "REJECT":
                        continue
                    total_sent += 1
                    got = sorted(pr)
                    want = sorted(gt)
                    cls_hist[str(len(pr))] = cls_hist.get(str(len(pr)), 0) + 1
                    if got == want:
                        agree += 1
                    if pr and gt and pr[0][0] == gt[0][0]:
                        exact += 1
                bi += len(preds)
        dt = time.perf_counter() - t0
        print(f"[proto] {card}: {len(sents)} 句 / {dt:.1f}s = {len(sents)/dt:.1f} 句/s "
              f"| 截断 {trunc} | 平均发射 {n_out/max(1,len(sents)):.2f}", flush=True)
        print(f"[proto] {card}: 与真值源**逐句完全一致** {agree}/{total_sent} "
              f"= {agree/max(1,total_sent):.4f} | 首条类别一致 {exact}/{total_sent} "
              f"= {exact/max(1,total_sent):.4f} | 发射数分布 {cls_hist}", flush=True)
        del dec
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
