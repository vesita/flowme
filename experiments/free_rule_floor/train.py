#!/usr/bin/env python3
"""free_rule_floor 四臂训练 + 评测（核全程冻结，只训头）。

配方 = PREREG §1（跑前写死，**逐字继承 bag_modules**）：1800 步、batch 64、
AdamW lr 1e-3 / wd 1e-4、cosine、grad clip 1.0、seed 42/43、DataLoader
`generator=manual_seed(seed)` ⇒ 与 `bag_modules` 同批序列。

  --arm A|N|U|UP     A 基线 / **N 只给 n_slots** / U 模块标签 / UP 标签+位置
  --randlabel        训练标签跨行 randperm（负对照；**mask 不动**）

模型、评测、行读取全部**只读复用** `experiments/bag_modules/{model,train}.py`
（importlib 按路径加载，不改源码）；只有 N 臂与缓存落点是本目录的。

用法：
  uv run python experiments/free_rule_floor/train.py --arm A --seed 42
  uv run python experiments/free_rule_floor/train.py --arm N --seed 42
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

import narm as NA  # noqa: E402

bm = NA.bm                                          # bag_modules/model.py（只读）
bt = NA._load(NA.BAG_DIR / "train.py", "frf_bagmod_train_ro")   # bag_modules/train.py（只读）
Spec = NA.Spec
ARMS = NA.ARMS
MODS = NA.MODS
v_bag_of = bm.v_bag_of
encode_tokens_ref = bm.encode_tokens

get_rows = bt.get_rows                              # TCH train/test + SS adv1/adv2
evaluate = bt.evaluate                              # 逐字同源（含 aux/head 诊断）
_metrics = bt._metrics
_fp = bm._fp

TCH_CACHE = ROOT / "experiments" / "two_channel_head" / "cache"
SS_CACHE = ROOT / "experiments" / "struct_supervision" / "cache"
BAG_CACHE = ROOT / "experiments" / "bag_modules" / "cache"   # 只读复用
DATA = HERE / "data"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
CACHE = HERE / "cache"

STEPS = 1800
BATCH = 64
LR = 1e-3
WD = 1e-4
CLIP = 1.0
SPLITS = ("test", "adv1", "adv2")
POS_BUCKET = 4

# ---- 地板：新电池（P0 results/battery.json，跑前算好）+ 旧电池（对照，两个都留）----
_BAT = json.loads((RESULTS / "battery.json").read_text(encoding="utf-8"))
MAX_NAIVE = {s: _BAT["splits"][s]["max_naive_new"] for s in SPLITS}
MAX_NAIVE_OLD = {"test": 0.5329, "adv1": 0.0, "adv2": 0.182}
MAX_NAIVE_OLD8 = {s: _BAT["splits"][s]["max_naive_old8"] for s in SPLITS}
MAJORITY = {s: _BAT["splits"][s]["majority"] for s in SPLITS}
assert _BAT["align_old_battery"]["aligned"], "旧电池口径未对齐（PREREG §3）"


# ---------------------------------------------------------------------------
# 标签张量（**span 句序**；与 bag_modules/label_tensors 逐字同式，只换 DATA 路径）
# ---------------------------------------------------------------------------
def label_tensors(split: str, rows: list[dict]) -> dict:
    obj = json.loads((DATA / f"labels_{split}.json").read_text(encoding="utf-8"))
    assert obj["fp"] == [bm_lbl_fp(r) for r in rows], f"{split} 标签文件与数据行不对齐"
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
        order = sorted(range(r["n_slots"]), key=lambda k: r["bag_span"][k][0])
        L = max(1, len(r["sent"]))
        for j, k in enumerate(order):
            a, _b = r["bag_span"][k]
            tt[i, j] = lab[k]["t"]
            rt[i, j] = lab[k]["r"]
            ct[i, j] = lab[k]["c"]
            pb[i, j] = min(POS_BUCKET - 1, a * POS_BUCKET // L)
            mk[i, j] = True
    return {"type_t": tt, "role_t": rt, "cls_t": ct, "pos_b": pb, "mask": mk}


def bm_lbl_fp(r: dict) -> str:
    import hashlib
    return hashlib.md5((r["sent"] + str(r["bag_span"])).encode()).hexdigest()[:12]


def randlabel(labs: dict, n_train: int, seed: int) -> dict:
    """训练标签跨行 randperm（**mask 不动**）—— 与 bag_modules 逐字同式。"""
    g = torch.Generator().manual_seed(seed * 1000 + 17)
    idx = torch.randperm(n_train, generator=g)
    out = dict(labs)
    for k in ("type_t", "role_t", "cls_t", "pos_b"):
        v = labs[k].clone()
        v[:n_train] = v[:n_train][idx]
        out[k] = v
    return out


# ---------------------------------------------------------------------------
# 缓存：优先只读复用既有（tch / ss / bag_modules），miss 才写**本目录**
# ---------------------------------------------------------------------------
def get_blob(rows: list[dict], spec: Spec, device: str, split: str) -> dict:
    fp = _fp(rows)
    for base in (TCH_CACHE, SS_CACHE, BAG_CACHE, CACHE):
        p = base / f"gen_{fp}_L{spec.max_len_sent}_M{spec.max_slots}.pt"
        if p.exists():
            blob = torch.load(p, map_location="cpu", weights_only=True)
            if blob["n"] == len(rows):
                print(f"[blob] 只读复用 {p}（n={blob['n']}）", flush=True)
                return blob
    model = NA.build_model("A", seed=42, spec=spec)
    return _encode(model, rows, spec, device)


@torch.no_grad()
def _encode(model, rows, spec, device, batch: int = 256) -> dict:
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"gen_{_fp(rows)}_L{spec.max_len_sent}_M{spec.max_slots}.pt"
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
        assign += [r["assign"] + [-1] * (spec.max_slots - len(r["assign"]))
                   for r in chunk]
    blob = {"v_sent": torch.cat(v_sent), "v_items": torch.cat(v_items),
            "item_mask": torch.cat(mask), "skel": torch.tensor(skel),
            "assign": torch.tensor(assign), "n": len(rows)}
    torch.save(blob, path)
    print(f"[blob] 写入 {path.name}", flush=True)
    return blob


def encode_tokens(model, rows, spec, device, batch: int = 256) -> dict:
    """P/UP 用：只读复用 bag_modules/tch/ss 的 tok 缓存，miss 才写本目录。"""
    fp = _fp(rows)
    for base in (BAG_CACHE, TCH_CACHE, SS_CACHE, CACHE):
        p = base / f"tok_{fp}_L{spec.max_len_sent}.pt"
        if p.exists():
            blob = torch.load(p, map_location="cpu", weights_only=True)
            if blob["n"] == len(rows):
                print(f"[tok-cache] 只读复用 {p}（n={blob['n']}）", flush=True)
                return blob
    return encode_tokens_ref(model, rows, spec, device)   # 写 bag_modules/cache（不应到达）


# ---------------------------------------------------------------------------
# 训练（与 bag_modules/train_one 逐字同式，只换模型构造与地板数字）
# ---------------------------------------------------------------------------
def train_one(arm: str, seed: int, rand: bool = False, mods: tuple[str, ...] = MODS,
              device: str | None = None, steps: int = STEPS) -> dict:
    t0 = time.time()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(seed)
    torch.manual_seed(seed)
    spec = Spec()
    model = NA.build_model(arm, seed, spec, mods).to(device)
    freeze = model.freeze_report()
    params = model.param_report()
    print(f"[freeze] {json.dumps(freeze, ensure_ascii=False)}", flush=True)
    print(f"[params] {json.dumps(params, ensure_ascii=False)}", flush=True)
    if arm == "N":
        print("[N] 旁路只读 mask（GRU 输入恒 0）；type/role/cls/pos_b 不进", flush=True)

    rows_tr = get_rows("train")
    n_train = len(rows_tr)
    blob_tr = get_blob(rows_tr, spec, device, "train")
    labs = label_tensors("train", rows_tr)
    if rand:
        before = torch.bincount(blob_tr["skel"][:n_train]).tolist()
        labs = randlabel(labs, n_train, seed)
        after = torch.bincount(blob_tr["skel"][:n_train]).tolist()
        print(f"[randlabel] 骨架分布不变 {before[:6]} == {after[:6]}；"
              f"标签张量跨行 randperm（seed*1000+17），mask 不动", flush=True)

    tok_tr = encode_tokens(model, rows_tr, spec, device) if model.pool is not None else None
    if tok_tr is not None:
        d = (bm.mean_of(tok_tr["h"], tok_tr["mask"]) - blob_tr["v_sent"]).abs().max().item()
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
        labs_s = label_tensors(split, rows)
        tok_s = encode_tokens(model, rows, spec, device) if model.pool is not None else None
        m, c, d = evaluate(model, blob, rows, labs_s, tok_s, device)
        blind_slot = sum(1.0 / len(r["assign"]) for r in rows) / len(rows)
        m["slot"]["blind"] = round(blind_slot, 4)
        m["skel"]["majority"] = MAJORITY[split]
        m["skel"]["max_naive"] = MAX_NAIVE[split]
        m["skel"]["max_naive_old"] = MAX_NAIVE_OLD[split]
        m["skel"]["max_naive_old8"] = MAX_NAIVE_OLD8[split]
        m["skel"]["over_max_naive"] = round(m["skel"]["acc"] - MAX_NAIVE[split], 4)
        m["skel"]["over_max_naive_old"] = round(
            m["skel"]["acc"] - MAX_NAIVE_OLD[split], 4)
        m["diag"] = d
        eval_out[split] = m
        correct[split] = c
        diag[split] = d
        print(f"[eval] {arm}{'_rand' if rand else ''} s{seed} {split:5s} "
              f"骨架={m['skel']['acc']:.4f} 槽位={m['slot']['acc']:.4f} "
              f"联合={m['joint']['acc']:.4f} (骨架SE={m['skel']['se']:.4f} "
              f"max_naive新={MAX_NAIVE[split]} 旧={MAX_NAIVE_OLD[split]}) "
              f"aux_only={d['aux_only']} head_only={d['head_only']}", flush=True)

    name = f"{arm}_s{seed}{'_rand' if rand else ''}"
    if steps != STEPS:
        name += f"_steps{steps}"
    out = {"name": name, "arm": arm, "seed": seed, "randlabel": rand,
           "mods": list(mods), "device": device, "freeze": freeze, "params": params,
           "recipe": {"steps": steps, "batch": BATCH, "lr": LR, "weight_decay": WD,
                      "cosine": True, "grad_clip": CLIP, "n_train": n_train,
                      "loss": "L_gen", "label_seq_order": "span 句序",
                      "prereg": "PREREG.md"},
           "max_naive": {"new": MAX_NAIVE, "old_reported": MAX_NAIVE_OLD,
                         "old8_recomputed": MAX_NAIVE_OLD8},
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
    if a.randlabel:
        assert a.arm in ("U", "UP"), "randlabel 是标签臂的负对照"
    out = train_one(a.arm, a.seed, a.randlabel, mods, a.device, a.steps)
    print(json.dumps({k: out[k] for k in ("name", "wall_sec", "params")},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
