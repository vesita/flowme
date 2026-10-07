#!/usr/bin/env python3
"""bag_modules 四臂训练 + 评测（核全程冻结，只训头）。

配方 = PREREG §5（跑前写死）：1800 步、batch 64、AdamW lr 1e-3 / wd 1e-4、cosine、
grad clip 1.0、seed 42/43；DataLoader `generator=manual_seed(seed)` ⇒ batch 序列与
`struct_supervision` **逐位相同**（臂 A 必须复现其 B 臂 0.6396/0.6424，作同口径校验）。

  --arm A|U|P|UP        四臂（PREREG §1）
  --mods type,role,cls  U 臂「杀模块」消融（默认三个全开）
  --randlabel           训练标签跨行 randperm（B3 门禁；评测仍用真标签，与 struct 同口径）

用法：
  uv run python experiments/bag_modules/train.py --arm A --seed 42
  uv run python experiments/bag_modules/train.py --arm U --seed 42 --randlabel
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

from model import (ARMS, MODS, BagModModel, Spec, _fp, encode_tokens,  # noqa: E402
                   mean_of, v_bag_of)

TCH_DATA = ROOT / "experiments" / "two_channel_head" / "data"
TCH_CACHE = ROOT / "experiments" / "two_channel_head" / "cache"
SS_DIR = ROOT / "experiments" / "struct_supervision"
SS_CACHE = SS_DIR / "cache"
DATA = HERE / "data"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"

STEPS = 1800
BATCH = 64
LR = 1e-3
WD = 1e-4
CLIP = 1.0
SPLITS = ("test", "adv1", "adv2")
POS_BUCKET = 4

# PREREG §4 B4/B3 基线（写死）
MAX_NAIVE = {"test": 0.5329, "adv1": 0.0, "adv2": 0.182}
MAJORITY = {"test": 0.4169, "adv1": 0.1230, "adv2": 0.2880}


def load_rows(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fp:
        return [json.loads(x) for x in fp]


def get_rows(split: str) -> list[dict]:
    if split in ("train", "test"):
        return load_rows(TCH_DATA / f"{split}.jsonl")
    p = SS_DIR / "data" / f"{split}.jsonl"
    if not p.exists():
        raise SystemExit(f"缺对抗集：{p}")
    return load_rows(p)


def get_blob(rows: list[dict], spec: Spec, device: str, split: str) -> dict:
    """train/test/adv 优先只读复用现有缓存；miss 才现算（写本目录）。"""
    fp = _fp(rows)
    for base, ro in ((TCH_CACHE, True), (SS_CACHE, True)):
        p = base / f"gen_{fp}_L{spec.max_len_sent}_M{spec.max_slots}.pt"
        if p.exists():
            blob = torch.load(p, map_location="cpu", weights_only=True)
            if blob["n"] == len(rows):
                print(f"[blob] 只读复用 {p}（n={blob['n']}）", flush=True)
                return blob
    # miss：用本目录编码器（写 experiments/bag_modules/cache）
    model = BagModModel("A", seed=42, spec=spec, )
    return _encode(model, rows, spec, device)


@torch.no_grad()
def _encode(model, rows, spec, device, batch: int = 256) -> dict:
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    (HERE / "cache").mkdir(exist_ok=True)
    path = HERE / "cache" / f"gen_{_fp(rows)}_L{spec.max_len_sent}_M{spec.max_slots}.pt"
    v_sent, v_items, mask, skel, assign = [], [], [], [], []
    model.encoder.eval()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        ids, msk = [], []
        for r in chunk:
            e = tok.encode(r["sent"], max_length=spec.max_len_sent, padding=True)
            assert sum(e["attention_mask"]) == len(r["sent"]), "1 字符 1 token 被破坏"
            ids.append(e["input_ids"])
            msk.append(e["attention_mask"])
        id_t = torch.tensor(ids, dtype=torch.long, device=device)
        m_t = torch.tensor(msk, dtype=torch.bool, device=device)
        h = model.encoder(id_t, m_t).cpu()
        m3 = m_t.cpu().unsqueeze(-1).to(h.dtype)
        v_sent.append((h * m3).sum(1) / m3.sum(1).clamp(min=1.0))
        bsz = len(chunk)
        items = torch.zeros(bsz, spec.max_slots, spec.hidden)
        im = torch.zeros(bsz, spec.max_slots, dtype=torch.bool)
        for j, r in enumerate(chunk):
            for s, (a, b) in enumerate(r["bag_span"]):
                assert r["sent"][a:b] == r["bag"][s]
                items[j, s] = h[j, a:b].mean(0)
                im[j, s] = True
        v_items.append(items)
        mask.append(im)
        skel += [r["skel_id"] for r in chunk]
        assign += [r["assign"] + [-1] * (spec.max_slots - len(r["assign"])) for r in chunk]
    blob = {"v_sent": torch.cat(v_sent), "v_items": torch.cat(v_items),
            "item_mask": torch.cat(mask), "skel": torch.tensor(skel),
            "assign": torch.tensor(assign), "n": len(rows)}
    torch.save(blob, path)
    print(f"[blob] 写入 {path.name}", flush=True)
    return blob


# ---------------------------------------------------------------------------
# 标签张量（**span 句序**；与袋序无关）
# ---------------------------------------------------------------------------
def label_tensors(rows: list[dict], split: str) -> dict:
    obj = json.loads((DATA / f"labels_{split}.json").read_text(encoding="utf-8"))
    assert obj["fp"] == [_fp_one(r) for r in rows], f"{split} 标签文件与数据行不对齐"
    assert obj["skel_id"] == [r["skel_id"] for r in rows], f"{split} skel_id 不对齐"
    n = len(rows)
    m = max(r["n_slots"] for r in rows)
    tt = torch.zeros(n, m, dtype=torch.long)
    rt = torch.zeros(n, m, dtype=torch.long)
    ct = torch.zeros(n, m, dtype=torch.long)
    pb = torch.zeros(n, m, dtype=torch.long)
    mk = torch.zeros(n, m, dtype=torch.bool)
    for i, (r, lab) in enumerate(zip(rows, obj["labels"])):
        assert len(lab) == r["n_slots"]
        order = sorted(range(r["n_slots"]), key=lambda k: r["bag_span"][k][0])  # 句序
        L = max(1, len(r["sent"]))
        for j, k in enumerate(order):
            a, _b = r["bag_span"][k]
            tt[i, j] = lab[k]["t"]
            rt[i, j] = lab[k]["r"]
            ct[i, j] = lab[k]["c"]
            pb[i, j] = min(POS_BUCKET - 1, a * POS_BUCKET // L)
            mk[i, j] = True
    return {"type_t": tt, "role_t": rt, "cls_t": ct, "pos_b": pb, "mask": mk}


def _fp_one(r: dict) -> str:
    import hashlib
    return hashlib.md5((r["sent"] + str(r["bag_span"])).encode()).hexdigest()[:12]


def randlabel(labs: dict, n_train: int, seed: int) -> dict:
    g = torch.Generator().manual_seed(seed * 1000 + 17)
    idx = torch.randperm(n_train, generator=g)
    out = dict(labs)
    for k in ("type_t", "role_t", "cls_t", "pos_b"):
        v = labs[k].clone()
        v[:n_train] = v[:n_train][idx]
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# 评测
# ---------------------------------------------------------------------------
def _metrics(skel_ok, slot_frac, joint_ok, blind_skel) -> dict:
    n = len(skel_ok)
    se = math.sqrt(0.25 / n)
    mu = sum(slot_frac) / n
    se_slot = math.sqrt(max(1e-12, sum((x - mu) ** 2 for x in slot_frac)) / (n - 1)) / math.sqrt(n)
    return {"skel": {"n": n, "acc": round(sum(skel_ok) / n, 6), "se": round(se, 6),
                     "blind": blind_skel},
            "slot": {"n": n, "acc": round(mu, 6), "se": round(se_slot, 6)},
            "joint": {"n": n, "acc": round(sum(joint_ok) / n, 6), "se": round(se, 6)}}


def evaluate(model, blob, rows, labs, tok, device, bs=4096) -> tuple[dict, dict, dict]:
    """结构头直出 + **只读标签通道** + **只读 h 通道**（机制①/③）。"""
    skel_ok, slot_frac, joint_ok = [], [], []
    aux_ok, head_ok = [], []
    use_lab = model.lab is not None
    model.eval()
    with torch.no_grad():
        for i in range(0, blob["n"], bs):
            j = min(i + bs, blob["n"])
            vs = blob["v_sent"][i:j].to(device)
            vi = blob["v_items"][i:j].to(device)
            im = blob["item_mask"][i:j].to(device)
            vb = v_bag_of(vi, im)
            sk_y = blob["skel"][i:j].to(device)
            as_y = blob["assign"][i:j].to(device)
            lab_in = None
            if use_lab:
                lab_in = {k: labs[k][i:j].to(device) for k in
                          ("type_t", "role_t", "cls_t", "pos_b")}
                lab_in["mask"] = labs["mask"][i:j].to(device)
            h_tok = tmask = None
            if tok is not None:
                h_tok = tok["h"][i:j].to(device)
                tmask = tok["mask"][i:j].to(device)
            sk_logits, a_logits, h = model.forward(vs, vb, vi, im, lab_in, h_tok, tmask)
            p = sk_logits.argmax(-1).cpu().tolist()
            pa = a_logits.argmax(-1).cpu()
            gold = as_y.tolist()
            if use_lab:
                aux_ok += (model.aux_only(lab_in).argmax(-1).cpu() == sk_y.cpu()).tolist()
                head_ok += (model.head_only(h).argmax(-1).cpu() == sk_y.cpu()).tolist()
            for r in range(j - i):
                k = int(im[r].sum())
                right = sum(1 for s in range(k) if int(pa[r, s]) == gold[r][s])
                slot_frac.append(right / k)
                ok = int(p[r] == int(sk_y[r]))
                skel_ok.append(ok)
                joint_ok.append(int(ok and right == k))
    metrics = _metrics(skel_ok, slot_frac, joint_ok, 0.0)
    diag = {"aux_only": round(sum(aux_ok) / len(aux_ok), 6) if aux_ok else None,
            "head_only": round(sum(head_ok) / len(head_ok), 6) if head_ok else None}
    return metrics, {"skel": skel_ok, "slot_frac": slot_frac, "joint": joint_ok}, diag


# ---------------------------------------------------------------------------
# 训练
# ---------------------------------------------------------------------------
def train_one(arm: str, seed: int, rand: bool = False, mods: tuple[str, ...] = MODS,
              device: str | None = None, steps: int = STEPS) -> dict:
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)
    spec = Spec()
    model = BagModModel(arm, seed, spec, mods).to(device)
    freeze = model.freeze_report()
    params = model.param_report()
    print(f"[freeze] {json.dumps(freeze, ensure_ascii=False)}", flush=True)
    print(f"[params] {json.dumps(params, ensure_ascii=False)}", flush=True)

    rows_tr = get_rows("train")
    n_train = len(rows_tr)
    blob_tr = get_blob(rows_tr, spec, device, "train")
    labs = label_tensors(rows_tr, "train")
    if rand:
        before = torch.bincount(blob_tr["skel"][:n_train]).tolist()
        labs = randlabel(labs, n_train, seed)
        after = torch.bincount(blob_tr["skel"][:n_train]).tolist()
        print(f"[randlabel] 骨架分布不变 {before[:6]} == {after[:6]}；"
              f"标签张量跨行 randperm（seed*1000+17）", flush=True)

    tok_tr = encode_tokens(model, rows_tr, spec, device) if model.pool is not None else None
    if tok_tr is not None:                     # 口径对账：H 池化 == 缓存 v_sent
        d = (mean_of(tok_tr["h"], tok_tr["mask"]) - blob_tr["v_sent"]).abs().max().item()
        assert d < 1e-4, f"H 池化与缓存 v_sent 不一致 {d}"
        print(f"[tok] H 池化 == 缓存 v_sent（max|Δ|={d:.2e}）", flush=True)

    vs, vb = blob_tr["v_sent"], v_bag_of(blob_tr["v_items"], blob_tr["item_mask"])
    ds = [vs, vb, blob_tr["v_items"], blob_tr["item_mask"], blob_tr["skel"],
          blob_tr["assign"], labs["type_t"], labs["role_t"], labs["cls_t"],
          labs["pos_b"], labs["mask"]]
    if tok_tr is not None:
        ds += [tok_tr["h"], tok_tr["mask"]]
    dataset = TensorDataset(*ds)
    loader = DataLoader(dataset, batch_size=BATCH, shuffle=True,
                        generator=torch.Generator().manual_seed(seed))

    head_params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(head_params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    it = iter(loader)

    def nxt():
        nonlocal it
        try:
            return next(it)
        except StopIteration:
            it = iter(loader)
            return next(it)

    losses, parts_hist = [], {}
    model.train()
    for step in range(steps):
        b = [x.to(device) for x in nxt()]
        lab_in = None
        if model.lab is not None:
            lab_in = {"type_t": b[6], "role_t": b[7], "cls_t": b[8], "pos_b": b[9],
                      "mask": b[10]}
        h_tok = tmask = None
        if model.pool is not None:
            h_tok, tmask = b[11], b[12]
        loss, sk, asg, _ = model.loss(b[0], b[1], b[2], b[3], b[4], b[5], lab_in,
                                      h_tok, tmask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head_params, CLIP)
        opt.step()
        sched.step()
        losses.append(float(loss.detach()))
        for k, v in (("gen", loss), ("skel", sk), ("assign", asg)):
            parts_hist.setdefault(k, []).append(float(v))
    print(f"[train] {steps} 步 done，loss 首/末 {losses[0]:.4f}/{losses[-1]:.4f}",
          flush=True)

    eval_out, correct, diag = {}, {}, {}
    for split in SPLITS:
        rows = get_rows(split)
        blob = get_blob(rows, spec, device, split)
        labs_s = label_tensors(rows, split)
        tok_s = encode_tokens(model, rows, spec, device) if model.pool is not None else None
        m, c, d = evaluate(model, blob, rows, labs_s, tok_s, device)
        blind_slot = sum(1.0 / len(r["assign"]) for r in rows) / len(rows)
        m["slot"]["blind"] = round(blind_slot, 4)
        m["skel"]["majority"] = MAJORITY[split]
        m["skel"]["max_naive"] = MAX_NAIVE[split]
        m["skel"]["over_max_naive"] = round(m["skel"]["acc"] - MAX_NAIVE[split], 4)
        m["diag"] = d
        eval_out[split] = m
        correct[split] = c
        diag[split] = d
        print(f"[eval] {arm}{'_rand' if rand else ''} s{seed} {split:5s} "
              f"骨架={m['skel']['acc']:.4f} 槽位={m['slot']['acc']:.4f} "
              f"联合={m['joint']['acc']:.4f} (骨架SE={m['skel']['se']:.4f} "
              f"max_naive={MAX_NAIVE[split]}) aux_only={d['aux_only']} "
              f"head_only={d['head_only']}", flush=True)

    name = f"{arm}_s{seed}{'_rand' if rand else ''}"
    if tuple(mods) != MODS:
        name += "_m" + "-".join(mods)
    if steps != STEPS:
        name += f"_steps{steps}"
    out = {"name": name, "arm": arm, "seed": seed, "randlabel": rand,
           "mods": list(mods), "device": device, "freeze": freeze, "params": params,
           "recipe": {"steps": steps, "batch": BATCH, "lr": LR, "weight_decay": WD,
                      "cosine": True, "grad_clip": CLIP, "n_train": n_train,
                      "loss": "L_gen", "label_seq_order": "span 句序"},
           "eval": eval_out, "loss_first": round(losses[0], 6),
           "loss_last": round(losses[-1], 6),
           "loss_parts": {k: {"first": round(v[0], 6), "last": round(v[-1], 6)}
                          for k, v in parts_hist.items()},
           "wall_sec": round(time.time() - t0, 1)}
    RESULTS.mkdir(exist_ok=True)
    WEIGHTS.mkdir(exist_ok=True)
    (RESULTS / f"{name}.json").write_text(
        json.dumps({"meta": out, "correct": correct}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    torch.save({k: v for k, v in model.state_dict().items()
                if not k.startswith("encoder")}, WEIGHTS / f"{name}.pt")
    print(f"[done] {name} → results/{name}.json（{out['wall_sec']}s）", flush=True)
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True, choices=[42, 43])
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--mods", default="type,role,cls")
    ap.add_argument("--device", default=None)
    ap.add_argument("--steps", type=int, default=STEPS)
    a = ap.parse_args(argv)
    mods = tuple(x for x in a.mods.split(",") if x)
    assert all(m in MODS for m in mods) and mods, mods
    out = train_one(a.arm, a.seed, a.randlabel, mods, a.device, a.steps)
    print(json.dumps({k: out[k] for k in ("name", "wall_sec", "params")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
