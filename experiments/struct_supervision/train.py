#!/usr/bin/env python3
"""struct_supervision 三臂训练 + 评测（核全程冻结，只训头）。

配方 = PREREG §1（跑前写死）：**1800 步**、batch 64、AdamW lr 1e-3 / wd 1e-4、cosine、
grad clip 1.0、seed 42/43；DataLoader `generator=manual_seed(seed)` ⇒ 与 two_channel_head
的生成分支 **batch 序列逐位相同**（B 臂应复现 0.6396/0.6424，作为同口径校验）。

  A：loss = L_sent（整句 teacher-forced CE）；结构指标 = **自由解码 → match_sentence 解析**（fail-closed）
  B：loss = L_gen（骨架 CE + 指派 CE）
  C：loss = L_gen + L_sent（λ=1）

随机标签对照（门禁）：`--randlabel`
  · 骨架臂（B）：train 骨架 randperm + 每行指派值行内 randperm（与 two_channel_head 同写法）；
  · 整句臂（A）：train **目标句**按行 randperm；评测仍用真金句；
  跑前打印 randperm 前后计数。

用法：
  uv run python experiments/struct_supervision/train.py --arm B --seed 42
  uv run python experiments/struct_supervision/train.py --arm A --seed 42 --randlabel
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from model import (BOS, PAD, StructSupModel, Spec, encode_rows, load_vocab,  # noqa: E402
                   parse_sentence, sentence_ids, v_bag_of)

TCH_DATA = ROOT / "experiments" / "two_channel_head" / "data"
TCH_CACHE = ROOT / "experiments" / "two_channel_head" / "cache"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"

STEPS = 1800
BATCH = 64
LR = 1e-3
WD = 1e-4
CLIP = 1.0
SPLITS = ("test", "adv1", "adv2")


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fp:
        return [json.loads(line) for line in fp]


def get_rows(split: str) -> list[dict]:
    if split in ("train", "test"):
        return load_rows(TCH_DATA / f"{split}.jsonl")
    p = HERE / "data" / f"{split}.jsonl"
    if not p.exists():
        raise SystemExit(f"缺对抗集：{p}（先跑 build_data.py）")
    return load_rows(p)


def get_blob(model, rows, spec, device, split) -> dict:
    """train/test 优先只读复用 two_channel_head 的缓存；其余用本目录同口径编码器。"""
    import hashlib
    fp = hashlib.md5("".join(r["sent"] + str(r["bag_span"]) for r in rows)
                     .encode()).hexdigest()[:12]
    if split in ("train", "test"):
        p = TCH_CACHE / f"gen_{fp}_L{spec.max_len_sent}_M{spec.max_slots}.pt"
        if p.exists():
            blob = torch.load(p, map_location="cpu", weights_only=True)
            if blob["n"] == len(rows):
                print(f"[blob] 只读复用 two_channel_head 缓存 {p.name}（n={blob['n']}）",
                      flush=True)
                return blob
        print(f"[blob] 未命中 two_channel_head 缓存（{p.name}），用本目录编码器",
              flush=True)
    return encode_rows(model, rows, spec, device)


def randlabel_gen(skel, assign, mask, n_train, seed):
    skel, assign = skel.clone(), assign.clone()
    g = torch.Generator().manual_seed(seed * 1000 + 13)
    idx = torch.arange(n_train)
    skel[idx] = skel[idx][torch.randperm(n_train, generator=g)]
    n = mask.sum(1).long()
    for i in range(n_train):
        k = int(n[i])
        perm = torch.randperm(k, generator=g)
        assign[i, :k] = assign[i, :k][perm]
    return skel, assign


def randlabel_sent(ids: torch.Tensor, n_train: int, seed: int) -> torch.Tensor:
    ids = ids.clone()
    g = torch.Generator().manual_seed(seed * 1000 + 7)
    idx = torch.arange(n_train)
    ids[idx] = ids[idx][torch.randperm(n_train, generator=g)]
    return ids


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def _metrics(skel_ok, slot_frac, joint_ok, blind_skel=0.025, blind_slot=None) -> dict:
    n = len(skel_ok)
    se = math.sqrt(0.25 / n)
    mu = sum(slot_frac) / n
    se_slot = math.sqrt(max(1e-12, sum((x - mu) ** 2 for x in slot_frac)) / (n - 1)) / math.sqrt(n)
    out = {
        "skel": {"n": n, "acc": round(sum(skel_ok) / n, 6), "se": round(se, 6)},
        "slot": {"n": n, "acc": round(mu, 6), "se": round(se_slot, 6)},
        "joint": {"n": n, "acc": round(sum(joint_ok) / n, 6), "se": round(se, 6)},
    }
    if blind_slot is not None:
        out["slot"]["blind"] = blind_slot
        out["slot"]["margin_over_2se"] = round((mu - blind_slot) / se_slot, 3)
    out["skel"]["blind"] = blind_skel
    out["skel"]["margin_over_2se"] = round((sum(skel_ok) / n - blind_skel) / se, 3)
    return out


def evaluate_heads(model, blob, rows, device, bs=4096):
    """结构头直出（B/C 的结构指标）。返回 (指标, 逐行 0/1)。"""
    skel_ok, slot_frac, joint_ok = [], [], []
    n_rows = blob["n"]
    model.eval()
    with torch.no_grad():
        for i in range(0, n_rows, bs):
            j = min(i + bs, n_rows)
            vs = blob["v_sent"][i:j].to(device)
            vi = blob["v_items"][i:j].to(device)
            im = blob["item_mask"][i:j].to(device)
            vb = v_bag_of(vi, im)
            sk_y = blob["skel"][i:j].to(device)
            as_y = blob["assign"][i:j].to(device)
            sk_logits, a_logits = model.forward_gen(vs, vb, vi, im)
            p_sk = sk_logits.argmax(-1).cpu().tolist()
            p_as = a_logits.argmax(-1).cpu()
            gold_as = as_y.tolist()
            gold_sk = sk_y.tolist()
            for r in range(j - i):
                k = int(im[r].sum())
                right = sum(1 for s in range(k) if int(p_as[r, s]) == gold_as[r][s])
                slot_frac.append(right / k)
                ok = int(p_sk[r] == gold_sk[r])
                skel_ok.append(ok)
                joint_ok.append(int(ok and right == k))
    metrics = _metrics(skel_ok, slot_frac, joint_ok)
    return metrics, {"skel": skel_ok, "slot_frac": slot_frac, "joint": joint_ok}


def evaluate_decoded(model, blob, rows, ids_all, device, bs=128, with_tf=True):
    """A/C 的结构出口：自由解码 → 解析（fail-closed）；附 TF 字准确率与句子诊断。"""
    skel_ok, slot_frac, joint_ok = [], [], []
    decoded: list[str] = []
    tf_ok = tf_tot = 0
    exact = parseable = 0
    n_rows = blob["n"]
    model.eval()
    with torch.no_grad():
        for i in range(0, n_rows, bs):
            j = min(i + bs, n_rows)
            vs = blob["v_sent"][i:j].to(device)
            vi = blob["v_items"][i:j].to(device)
            im = blob["item_mask"][i:j].to(device)
            vb = v_bag_of(vi, im)
            if with_tf:
                o, t = model.tf_char_stats(ids_all[i:j].to(device), vs, vb, vi, im)
                tf_ok += o
                tf_tot += t
            sents = model.decode(vs, vb, vi, im)
            decoded += sents
            for r, ds in enumerate(sents):
                gold = rows[i + r]
                exact += int(ds == gold["sent"])
                p = parse_sentence(ds, gold["bag"])
                if p is None:
                    skel_ok.append(0)
                    slot_frac.append(0.0)
                    joint_ok.append(0)
                    continue
                parseable += 1
                sid, pasg = p
                ok = int(sid == gold["skel_id"])
                k = len(gold["assign"])
                right = sum(int(pasg[s]) == gold["assign"][s] for s in range(k))
                slot_frac.append(right / k)
                skel_ok.append(ok)
                joint_ok.append(int(ok and right == k))
    n = len(skel_ok)
    metrics = _metrics(skel_ok, slot_frac, joint_ok)
    diag = {"tf_char_acc": round(tf_ok / max(1, tf_tot), 6),
            "tf_char_n": tf_tot,
            "sent_exact": round(exact / n, 6),
            "parseable": round(parseable / n, 6),
            "n": n}
    correct = {"skel": skel_ok, "slot_frac": slot_frac, "joint": joint_ok}
    return metrics, correct, diag, decoded[:3]


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_one(arm: str, seed: int, randlabel: bool = False, device: str | None = None,
              steps: int = STEPS) -> dict:
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)
    spec = Spec()
    vocab = load_vocab()
    model = StructSupModel(arm, seed, spec, vocab).to(device)
    freeze = model.freeze_report()
    params = model.param_report()
    print(f"[freeze] {json.dumps(freeze, ensure_ascii=False)}", flush=True)
    print(f"[params] {json.dumps(params, ensure_ascii=False)}", flush=True)

    # ---- 数据 ----
    rows_tr = get_rows("train")
    blob_tr = get_blob(model, rows_tr, spec, device, "train")
    ids_tr = sentence_ids(rows_tr, vocab)
    n_train = len(rows_tr)
    skel_t = blob_tr["skel"]
    asg_t = blob_tr["assign"]
    ids_t = ids_tr
    if randlabel:
        if arm in ("B", "C"):
            before = torch.bincount(skel_t[:n_train]).tolist()
            skel_t, asg_t = randlabel_gen(skel_t, asg_t, blob_tr["item_mask"],
                                          n_train, seed)
            after = torch.bincount(skel_t[:n_train]).tolist()
            print(f"[randlabel] 骨架 randperm 前 6 类 {before[:6]} → {after[:6]}"
                  f"（分布不变）；每行指派值行内 randperm", flush=True)
        if arm in ("A", "C"):
            head_before = ["".join(vocab_ids_to_chars(ids_t[i], vocab))
                           for i in range(3)]
            ids_t = randlabel_sent(ids_t, n_train, seed)
            head_after = ["".join(vocab_ids_to_chars(ids_t[i], vocab))
                          for i in range(3)]
            print(f"[randlabel] 目标句按行 randperm：前 3 行首 16 字\n"
                  f"    前 {head_before}\n    后 {head_after}", flush=True)

    vs, vb, vi, im = (blob_tr["v_sent"], v_bag_of(blob_tr["v_items"], blob_tr["item_mask"]),
                      blob_tr["v_items"], blob_tr["item_mask"])
    ds = TensorDataset(vs, vb, vi, im, skel_t, asg_t, ids_t)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    losses = []
    part_hist: dict[str, list] = {}
    model.train()
    for step in range(steps):
        b = next_batch()
        b = tuple(x.to(device) for x in b)
        loss, parts = model.loss(b[0], b[1], b[2], b[3], b[4], b[5], b[6])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, CLIP)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        for k, v in parts.items():
            part_hist.setdefault(k, []).append(float(v))
    print(f"[train] {steps} 步 done，loss 首/末 {losses[0]:.4f}/{losses[-1]:.4f}",
          flush=True)

    # ---- 评测（test + adv1 + adv2）----
    eval_out: dict = {}
    correct: dict = {}
    examples: dict = {}
    for split in SPLITS:
        rows = rows_tr if split == "train" else get_rows(split)
        blob = blob_tr if split == "train" else get_blob(model, rows, spec, device, split)
        ids_all = ids_tr if split == "train" else sentence_ids(rows, vocab)
        # 盲猜 = 逐行 1/槽数 的均值（与 two_channel_head 的 blind_uniform 同式）
        blind_slot = sum(1.0 / len(r["assign"]) for r in rows) / len(rows)
        def _t(acc, base, se):
            """(acc−基线)/SE；se=0（逐行全同）时不除零，返回 None 并标记。"""
            if se > 0:
                return round((acc - base) / se, 3)
            return None
        if arm in ("B", "C"):
            m, c = evaluate_heads(model, blob, rows, device)
            m["slot"]["blind"] = round(blind_slot, 4)
            m["slot"]["t"] = _t(m["slot"]["acc"], blind_slot, m["slot"]["se"])
            m["slot"]["all_identical"] = m["slot"]["se"] == 0
            eval_out[f"{split}_head"] = m
            correct[f"{split}_head"] = c
        if arm in ("A", "C"):
            m, c, diag, ex = evaluate_decoded(model, blob, rows, ids_all, device)
            m["slot"]["blind"] = round(blind_slot, 4)
            m["slot"]["t"] = _t(m["slot"]["acc"], blind_slot, m["slot"]["se"])
            m["slot"]["all_identical"] = m["slot"]["se"] == 0
            m["sent_diag"] = diag
            eval_out[f"{split}_decoded"] = m
            correct[f"{split}_decoded"] = c
            examples[split] = [{"gold": r["sent"], "pred": p}
                               for r, p in zip(rows[:3], ex)]
        key = f"{split}_decoded" if arm == "A" else f"{split}_head"
        mm = eval_out[key]
        print(f"[eval] {arm}{'_rand' if randlabel else ''} s{seed} {split:5s} "
              f"骨架={mm['skel']['acc']:.4f} 槽位={mm['slot']['acc']:.4f} "
              f"联合={mm['joint']['acc']:.4f} (槽位SE={mm['slot']['se']:.4f} "
              f"盲猜={mm['slot'].get('blind')})", flush=True)
        if f"{split}_decoded" in eval_out:
            d = eval_out[f"{split}_decoded"]["sent_diag"]
            print(f"        整句诊断 TF字={d['tf_char_acc']:.4f} 逐字复现={d['sent_exact']:.4f} "
                  f"可解析={d['parseable']:.4f}", flush=True)

    name = f"{arm}_s{seed}{'_rand' if randlabel else ''}"
    if steps != STEPS:
        name += f"_steps{steps}"
    out = {
        "name": name, "arm": arm, "seed": seed, "randlabel": randlabel,
        "device": device, "freeze": freeze, "params": params,
        "recipe": {"steps": steps, "batch": BATCH, "lr": LR, "weight_decay": WD,
                   "cosine": True, "grad_clip": CLIP, "n_train": n_train,
                   "loss": {"A": "L_sent", "B": "L_gen",
                            "C": "L_gen + L_sent (λ=1)"}[arm],
                   "vocab_size": len(vocab)},
        "eval": eval_out,
        "loss_first": round(losses[0], 6), "loss_last": round(losses[-1], 6),
        "loss_parts": {k: {"first": round(v[0], 6), "last": round(v[-1], 6),
                           "mean": round(sum(v) / len(v), 6)}
                       for k, v in part_hist.items()},
        "wall_sec": round(time.time() - t0, 1),
        "examples": examples,
    }
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(
        json.dumps({"meta": out, "correct": correct}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (RESULTS / f"{name}.summary.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    WEIGHTS.mkdir(exist_ok=True)
    torch.save({k: v for k, v in model.state_dict().items()
                if not k.startswith("encoder")}, WEIGHTS / f"{name}.pt")
    print(f"[done] {name} → results/{name}.json（{out['wall_sec']}s）", flush=True)
    return out


def vocab_ids_to_chars(ids: torch.Tensor, vocab: dict) -> str:
    rev = {i: c for c, i in vocab.items()}
    out = []
    for t in ids.tolist():
        if t in (PAD, BOS):
            continue
        if t == 2:                      # EOS
            break
        out.append(rev.get(t, "?"))
        if len(out) >= 16:
            break
    return "".join(out)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list("ABC"))
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43])
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=STEPS,
                    help="仅冒烟用；非默认值在结果名后缀 _stepsN")
    a = ap.parse_args(argv)
    RESULTS.mkdir(exist_ok=True)
    WEIGHTS.mkdir(exist_ok=True)
    train_one(a.arm, a.seed, a.randlabel, a.device, a.steps)


if __name__ == "__main__":
    main(sys.argv[1:])
