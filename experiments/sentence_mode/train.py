#!/usr/bin/env python3
"""P7 语句模式卡训练 + 评测（**核全程冻结、只训头**；PREREG §5/§6）。

  --arm KEEP|MASK|MIX   训练输入的标点版本（MIX = 逐样本 seed 掷币 50/50）
  --seed 42|43
  --randlabel           训练侧 mode 标签跨行 randperm（M4 负对照；输入不动）
  --steps N             默认 800（PREREG 写死）

评测出口：`keep / maskfinal / mask`（同一批 test 行，逐行配对）
        ＋ `colloq / colloq_mask`（独立池）。
每个出口都报 mode acc ± SE、触发词 span acc、结构侧通过率，并落逐行预测（配对 SE 用）。

用法：uv run python experiments/sentence_mode/train.py --arm MASK --seed 42
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402

import labels as L  # noqa: E402
import mode_render as MR  # noqa: E402

DATA = HERE / "data"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
BATTERY = json.loads((RESULTS / "battery.json").read_text(encoding="utf-8"))

BASE = ROOT / "checkpoints" / "base_encoder.pt"
MAX_LEN = 128
STEPS, BATCH, LR, WD, CLIP = 800, 64, 1e-3, 1e-4, 1.0
ARMS = ("KEEP", "MASK", "MIX")
SEEDS = (42, 43)
#: 池 → 出口评估对（(池, 行内 exit/trig 键, 出口名)）
EVAL_KEYS = (("train_pool", "keep", "keep"), ("train_pool", "maskfinal", "maskfinal"),
             ("train_pool", "mask", "mask"),
             ("colloq_pool", "keep", "colloq"), ("colloq_pool", "mask", "colloq_mask"))
EXIT_TO_KEY = {"keep": "keep", "maskfinal": "maskfinal", "mask": "mask",
               "colloq": "keep", "colloq_mask": "mask"}
POOLS = {"train_pool": ("train", "test"), "colloq_pool": ("colloq_train", "colloq_test")}


def se(p: float, n: int) -> float:
    return math.sqrt(max(p * (1 - p), 1e-12) / max(n, 1))


def load(name: str) -> list[dict]:
    with open(DATA / f"{name}.jsonl", encoding="utf-8") as f:
        return [json.loads(x) for x in f]


# ---------------------------------------------------------------------------
# 数据 → 张量（一次预分词，训练/评测都读缓存张量）
# ---------------------------------------------------------------------------
_TOK = NanoCharTokenizer()


def bio_labels(rows: list[dict], keys: list[str]) -> torch.Tensor:
    """BIO：0=O, 1=B, 2=I；填充位 -100（CE ignore_index）。"""
    y = torch.full((len(rows), MAX_LEN), -100, dtype=torch.long)
    for i, (r, key) in enumerate(zip(rows, keys)):
        t = r["exit"][key]
        y[i, :len(t)] = 0
        sp = r["trig"].get(key)
        if sp:
            a, b = sp
            y[i, a] = 1
            if b - a > 1:
                y[i, a + 1:b] = 2
    return y


def decode_bio(seq: list[int]) -> tuple[int, int] | None:
    start = None
    for i, v in enumerate(seq):
        if v == 1:
            start = i
            break
    if start is None:
        return None
    end = start + 1
    while end < len(seq) and seq[end] == 2:
        end += 1
    return (start, end)


# ---------------------------------------------------------------------------
# 模型（核冻结 + 头）
# ---------------------------------------------------------------------------
class ModeCard(nn.Module):
    def __init__(self, enc, n_mode: int = 6, n_bio: int = 3):
        super().__init__()
        self.encoder = enc
        self.trunk = nn.Sequential(nn.Linear(128, 256), nn.GELU(),
                                   nn.Linear(256, 128), nn.GELU())
        self.mode_head = nn.Linear(128, n_mode)
        self.bio = nn.Sequential(nn.Linear(128, 128), nn.GELU(), nn.Linear(128, n_bio))

    def forward(self, ids, mask):
        with torch.no_grad():                     # 核不训 ⇒ 不留激活
            h = self.encoder(ids, mask)
        m = mask.unsqueeze(-1).to(h.dtype)
        pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.mode_head(self.trunk(pooled)), self.bio(h)


def build_model(seed: int) -> ModeCard:
    torch.manual_seed(seed)
    enc, ck = load_base_encoder(BASE)
    for n, p in enc.named_parameters():
        assert not p.requires_grad, f"核参数 {n} 竟然 requires_grad=True"
    enc.eval()
    model = ModeCard(enc)
    frozen = [n for n, p in model.named_parameters() if not p.requires_grad]
    trainab = [n for n, p in model.named_parameters() if p.requires_grad]
    print(f"[freeze] 冻结 {len(frozen)} 个张量（全部来自 encoder）；"
          f"可训 {len(trainab)}：{trainab}", flush=True)
    return model


# ---------------------------------------------------------------------------
# 臂选择：训练行用哪个出口
# ---------------------------------------------------------------------------
def arm_keys(rows: list[dict], arm: str, seed: int) -> list[str]:
    if arm == "KEEP":
        return ["keep"] * len(rows)
    if arm == "MASK":
        return ["mask"] * len(rows)
    g = random.Random(seed * 7919 + 13)
    return ["keep" if g.random() < 0.5 else "mask" for _ in rows]


def stacked(rows: list[dict], keys: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    """逐行预分词（1 字符 1 token；越长已由 `build_data` 的 80 字上限挡住）。"""
    ids = torch.empty(len(rows), MAX_LEN, dtype=torch.long)
    msk = torch.empty(len(rows), MAX_LEN, dtype=torch.bool)
    for i, (r, k) in enumerate(zip(rows, keys)):
        t = r["exit"][k]
        e = _TOK.encode(t, max_length=MAX_LEN, padding=True)
        assert sum(e["attention_mask"]) == len(t), "1 字符 1 token 被破坏"
        ids[i] = torch.tensor(e["input_ids"], dtype=torch.long)
        msk[i] = torch.tensor(e["attention_mask"], dtype=torch.bool)
    return ids, msk


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_one(arm: str, seed: int, rand: bool, steps: int, device: str) -> dict:
    t0 = time.time()
    torch.manual_seed(seed)
    random.seed(seed)
    tr, te = load("train"), load("test")
    ctr, cte = load("colloq_train"), load("colloq_test")

    keys = arm_keys(tr, arm, seed)
    ids, msk = stacked(tr, keys)
    mode_y = torch.tensor([r["mode"] for r in tr], dtype=torch.long)
    bio_y = bio_labels(tr, keys)
    if rand:
        g = torch.Generator().manual_seed(seed * 1000 + 17)
        idx = torch.randperm(len(tr), generator=g)
        before = torch.bincount(mode_y).tolist()
        mode_y = mode_y[idx]
        after = torch.bincount(mode_y).tolist()
        print(f"[randlabel] mode 分布不变 {before} == {after}；输入不动", flush=True)

    model = build_model(seed).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[params] 可训参数 {n_params}", flush=True)

    ds = TensorDataset(ids, msk, mode_y, bio_y)
    dl = DataLoader(ds, batch_size=BATCH, shuffle=True,
                    generator=torch.Generator().manual_seed(seed))
    head = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    ce = nn.CrossEntropyLoss()

    model.train()
    losses, mode_accs = [], []
    it = iter(dl)
    def nxt():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(dl)
            return next(it)
    for step in range(steps):
        b = [x.to(device) for x in nxt()]
        ml, bl = model(b[0], b[1])
        loss = ce(ml, b[2]) + ce(bl.reshape(-1, 3), b[3].reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head, CLIP)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        mode_accs.append(float((ml.argmax(-1) == b[2]).float().mean()))
    print(f"[train] {steps} 步 done；loss {losses[0]:.4f}→{losses[-1]:.4f}；"
          f"train mode acc 末 100 步均值 {sum(mode_accs[-100:]) / min(100, len(mode_accs)):.4f}",
          flush=True)

    # ---- 评测 ----
    model.eval()
    pools = {"train_pool": (te, "test"), "colloq_pool": (cte, "colloq_test")}
    eval_out: dict = {}
    with torch.no_grad():
        for pname, ekey, ename in EVAL_KEYS:
            rows, splitname = pools[pname]
            ei, em = stacked(rows, [ekey] * len(rows))
            gold_m = [r["mode"] for r in rows]
            gold_t = [tuple(r["trig"][ekey]) if r["trig"][ekey] else None for r in rows]
            pred_m, pred_t, kinds = [], [], []
            for i in range(0, len(rows), 512):
                ml, bl = model(ei[i:i + 512].to(device), em[i:i + 512].to(device))
                pm = ml.argmax(-1).cpu().tolist()
                pb = bl.argmax(-1).cpu().tolist()
                for j, seq in enumerate(pb):
                    L_ = int(em[i + j].sum())
                    sp = decode_bio(seq[:L_])          # 只看真实长度内（跳过 pad 位）
                    pred_t.append(sp)
                    pred_m.append(pm[j])
            n = len(rows)
            for k, (r, sp) in enumerate(zip(rows, pred_t)):
                rec = MR.make_record(pred_m[k], sp, r["exit"][ekey],
                                     plan_step_id=f"sm7:{ename}:{k}",
                                     input_text=r["exit"][ekey])
                kinds.append(rec["kind"])
            acc = sum(1 for a, b in zip(pred_m, gold_m) if a == b) / n
            tacc = sum(1 for a, b in zip(pred_t, gold_t) if a == b) / n
            bat = BATTERY["exits"][ename]["pool"]
            rec = {
                "exit": ename, "n": n, "mode_acc": round(acc, 4),
                "mode_se": round(se(acc, n), 4),
                "mode_correct": [1 if a == b else 0 for a, b in zip(pred_m, gold_m)],
                "pred_mode": pred_m, "gold_mode": gold_m,
                "trig_acc": round(tacc, 4), "trig_se": round(se(tacc, n), 4),
                "trig_none_gold": round(sum(1 for t in gold_t if t is None) / n, 4),
                "record_text_rate": round(sum(1 for k in kinds if k == "text") / n, 4),
                "max_naive": bat["max_naive"], "max_naive_rule": bat["max_naive_rule"],
                "rule_R1_simple": bat["R1_punct_simple"],
                "rule_R2_particle": bat["R2_particle"],
                "rule_R3_final_char": bat["R3_final_char"],
                "rule_R6_labeldef": bat["R6_labeldef"],
                "majority": bat["R0_majority"],
                "over_max_naive": round(acc - bat["max_naive"], 4),
                "over_max_naive_2se": round(acc - bat["max_naive"] - 2 * se(acc, n), 4),
            }
            eval_out[ename] = rec
            print(f"[eval] {arm}{'_rand' if rand else ''} s{seed} {ename:11s} "
                  f"mode={acc:.4f}±{se(acc, n):.4f} trig={tacc:.4f} "
                  f"max_naive={bat['max_naive']:.4f} Δ={acc - bat['max_naive']:+.4f} "
                  f"rec_text={rec['record_text_rate']:.4f}", flush=True)

    name = f"{arm}_s{seed}{'_rand' if rand else ''}"
    if steps != STEPS:
        name += f"_steps{steps}"
    out = {"name": name, "arm": arm, "seed": seed, "randlabel": rand,
           "device": device, "steps": steps, "batch": BATCH, "lr": LR,
           "n_train": len(tr), "trainable_params": n_params,
           "loss_first": round(losses[0], 6), "loss_last": round(losses[-1], 6),
           "train_mode_acc_last100": round(
               sum(mode_accs[-100:]) / min(100, len(mode_accs)), 4),
           "eval": eval_out, "wall_sec": round(time.time() - t0, 1),
           "prereg": "PREREG.md"}
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(
        json.dumps(out, ensure_ascii=False), encoding="utf-8")
    print(f"[done] {name} → results/{name}.json（{out['wall_sec']}s）", flush=True)
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--seed", type=int, required=True, choices=list(SEEDS))
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device.startswith("cuda"):
        print(f"[gpu] {torch.cuda.get_device_name(0)}", flush=True)
    train_one(a.arm, a.seed, a.randlabel, a.steps, device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
