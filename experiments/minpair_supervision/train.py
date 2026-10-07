#!/usr/bin/env python3
"""P21 三臂训练（核全程冻结，只训头；数据 = skeleton_leak/data/b_pairs.jsonl 的 TRAIN 半）。

配方 = PREREG §3（跑前写死）：1800 步、batch = 32 对 = 64 行、AdamW lr 1e-3 / wd 1e-4、
cosine、grad clip 1.0、seed 42/43；pair 下标打乱 ⇒ 同 seed 三臂 batch 序逐位相同。

  M   = mean_row CE40 + CE_assign          （既有口径）
  MP  = mean_pair ½[CE{s,t}(A)+CE{s,t}(B)] + CE_assign
  MP+ = M + 1.0 × MP_pair                  （λ = 1 写死）

用法：
  uv run python experiments/minpair_supervision/train.py --selfcheck
  uv run python experiments/minpair_supervision/train.py --arm M --seed 42
  uv run python experiments/minpair_supervision/train.py --arm MP+ --seed 43 --randlabel
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

FWMP = ROOT / "experiments" / "funcword_minpair"
SL_DIR = ROOT / "experiments" / "skeleton_leak"
TCH_DATA = ROOT / "experiments" / "two_channel_head" / "data"
RESULTS = HERE / "results"
WEIGHTS = HERE / "weights"
CACHE = HERE / "cache"
DATA_BP = SL_DIR / "data" / "b_pairs.jsonl"

ARMS = ("M", "MP", "MP+")
STEPS = 1800
PAIRS_PER_BATCH = 32          # = 64 行（既有口径 batch 64 行）
LR = 1e-3
WD = 1e-4
CLIP = 1.0
SEEDS = (42, 43)


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FW = _load(FWMP / "model.py", "p21_fw_model")        # 只读复用（其 encode_rows 只写自己目录，本单元不调）
Spec, StructSupModel, v_bag_of = FW.Spec, FW.StructSupModel, FW.v_bag_of
check, pair_ce, assign_loss, fp = FW.check, FW.pair_ce, FW.assign_loss, FW.fp


# ---------------------------------------------------------------------------
# 切分（PREREG §2，跑前写死、确定性）
# ---------------------------------------------------------------------------
def load_rows() -> list[dict]:
    with open(DATA_BP, encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def build_pairs(rows: list[dict]) -> list[tuple[int, int]]:
    prs = [(i, i + 1) for i, r in enumerate(rows)
           if str(r.get("kind", "")).endswith("_src")]
    for i, j in prs:
        check(str(rows[j]["kind"]).endswith("_variant"), "相邻行不是 _variant")
        check(sorted(rows[i]["bag"]) == sorted(rows[j]["bag"]), "对内袋（多重集）不同")
        check(rows[i]["skel_id"] != rows[j]["skel_id"], "对内骨架相同")
        check(rows[i]["sent"] != rows[j]["sent"], "对内句面相同")
    check(len(prs) * 2 == len(rows), "行数与对数不符")
    return prs


def split(rows: list[dict], prs: list[tuple[int, int]]) -> tuple[list[int], list[int], dict]:
    """按方向分层、组内按 md5("P21:"+源句) 排序，前 10 TRAIN / 后 10 HELDOUT。"""
    dirs: dict[tuple, list[int]] = {}
    for k, (i, j) in enumerate(prs):
        key = (rows[i]["skel_id"], rows[j]["skel_id"], rows[i]["kind"])
        dirs.setdefault(key, []).append(k)
    tr_pair: list[int] = []
    ho_pair: list[int] = []
    for key in sorted(dirs, key=str):
        idx = sorted(dirs[key], key=lambda k: hashlib.md5(
            f"P21:{rows[prs[k][0]]['sent']}".encode()).hexdigest())
        check(len(idx) == 20, f"方向 {key} 对数 {len(idx)} != 20")
        tr_pair += idx[:10]
        ho_pair += idx[10:]
    tr_pair.sort()
    ho_pair.sort()
    tr_rows = [r for k in tr_pair for r in prs[k]]
    ho_rows = [r for k in ho_pair for r in prs[k]]

    st = {"n_dirs": len(dirs), "n_pairs_train": len(tr_pair),
          "n_pairs_heldout": len(ho_pair), "n_rows_train": len(tr_rows),
          "n_rows_heldout": len(ho_rows)}
    # ① 句面不交叉
    check(not (set(rows[i]["sent"] for i in tr_rows)
               & set(rows[i]["sent"] for i in ho_rows)), "① 句面交叉")
    # ② 每对两侧同 split（构造性）
    # ③ 袋元组不交叉
    bagk = lambda i: tuple(sorted(rows[i]["bag"]))
    check(not (set(bagk(i) for i in tr_rows) & set(bagk(i) for i in ho_rows)), "③ 袋元组交叉")
    # ④ 每方向 10/10
    dtr: dict[tuple, int] = {}
    dho: dict[tuple, int] = {}
    for k in tr_pair:
        dtr[(rows[prs[k][0]]["skel_id"], rows[prs[k][1]]["skel_id"])] = \
            dtr.get((rows[prs[k][0]]["skel_id"], rows[prs[k][1]]["skel_id"]), 0) + 1
    for k in ho_pair:
        dho[(rows[prs[k][0]]["skel_id"], rows[prs[k][1]]["skel_id"])] = \
            dho.get((rows[prs[k][0]]["skel_id"], rows[prs[k][1]]["skel_id"]), 0) + 1
    check(dtr == dho and set(dtr.values()) == {10}, "④ 方向未 10/10 分层")
    # ⑤ 骨架 id 集合相同
    ids_tr = set(rows[i]["skel_id"] for i in tr_rows)
    ids_ho = set(rows[i]["skel_id"] for i in ho_rows)
    check(ids_tr == ids_ho, f"⑤ 骨架集合不一致 {ids_tr ^ ids_ho}")
    st["n_skel_ids"] = len(ids_tr)
    st["skel_ids"] = sorted(ids_tr)
    st["split_train_pairs"] = tr_pair
    st["split_heldout_pairs"] = ho_pair
    return tr_rows, ho_rows, st


def load_blob() -> dict:
    """只读复用 skeleton_leak 编码缓存（n=1120）。"""
    path = SL_DIR / "cache" / f"enc_{fp(load_rows())}_L64.pt"
    check(path.exists(), f"只读编码缓存缺失 {path}")
    b = torch.load(path, map_location="cpu", weights_only=True)
    check(b["n"] == 1120, "缓存 n 漂移")
    print(f"[blob] ← 只读缓存 {path}（n={b['n']}）", flush=True)
    return b


def randlabel(y: torch.Tensor, asg: torch.Tensor, mask: torch.Tensor, seed: int):
    """类级双射重标（TRAIN 出现的骨架 id 上 randperm）+ 每行 assign 值行内 randperm。"""
    trained = sorted(set(y.tolist()))
    g = torch.Generator().manual_seed(seed * 1000 + 13)
    perm = torch.randperm(len(trained), generator=g)
    lut = torch.zeros(max(trained) + 1, dtype=torch.long)
    for src, dst in zip(trained, [trained[int(k)] for k in perm]):
        lut[src] = dst
    before = torch.bincount(y, minlength=40).tolist()
    y2 = lut[y]
    after = torch.bincount(y2, minlength=40).tolist()
    check(sorted(before) == sorted(after), "randlabel 改变了标签分布")
    n = mask.sum(1).long()
    a2 = asg.clone()
    for i in range(len(a2)):
        k = int(n[i])
        if k <= 1:
            continue
        p = torch.randperm(k, generator=g)
        a2[i, :k] = a2[i, :k][p]
    return y2, a2, before, after, dict(zip(trained, [int(lut[s]) for s in trained]))


# ---------------------------------------------------------------------------
def train_one(arm: str, seed: int, rand: bool, steps: int, device: str) -> dict:
    t0 = time.time()
    torch.manual_seed(seed)
    spec = Spec()
    model = StructSupModel("B", seed, spec, vocab=None).to(device)
    freeze = model.freeze_report()
    check(freeze["encoder_trainable"] == 0, "核没冻住")
    check(not freeze["encoder_training"], "核不在 eval()")
    print(f"[freeze] {json.dumps(freeze, ensure_ascii=False)}", flush=True)

    rows = load_rows()
    prs = build_pairs(rows)
    tr_rows, ho_rows, st = split(rows, prs)
    print(f"[split] {st['n_pairs_train']} 对 TRAIN / {st['n_pairs_heldout']} 对 HELDOUT，"
          f"{st['n_dirs']} 方向、骨架 {st['n_skel_ids']} 类", flush=True)
    blob = load_blob()

    # TRAIN 行按「对内相邻」顺序排列（2k = A 侧、2k+1 = B 侧）
    tr_idx = torch.tensor(tr_rows, dtype=torch.long)
    vs = blob["v_sent"][tr_idx]
    vi = blob["v_items"][tr_idx]
    im = blob["item_mask"][tr_idx]
    vb = v_bag_of(vi, im)
    y = blob["skel"][tr_idx].clone()
    asg = blob["assign"][tr_idx].clone()
    # 对内骨架必须不同
    check(bool((y[0::2] != y[1::2]).all()), "TRAIN 对内 gold 相同")

    if rand:
        y, asg, before, after, lut = randlabel(y, asg, im, seed)
        print(f"[randlabel] 类级 randperm：{json.dumps(lut, ensure_ascii=False)[:200]}；"
              f"计数不变={sorted(before) == sorted(after)}；assign 行内 randperm", flush=True)

    n_pairs = len(tr_rows) // 2
    head_params = [p for p in model.parameters() if p.requires_grad]
    n_head = sum(p.numel() for p in head_params)
    print(f"[params] 可训参数 {n_head}（trunk+gen），核冻结；train 行 {len(tr_rows)}、对 {n_pairs}",
          flush=True)
    opt = torch.optim.AdamW(head_params, lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    def batches():
        g = torch.Generator().manual_seed(seed)
        while True:
            perm = torch.randperm(n_pairs, generator=g)
            for s in range(0, n_pairs, PAIRS_PER_BATCH):
                pp = perm[s:s + PAIRS_PER_BATCH]
                yield torch.stack([pp * 2, pp * 2 + 1], 1).reshape(-1)

    it = batches()
    losses, parts_hist = {}, {}
    model.train()
    for step in range(steps):
        bi = next(it)
        h = model.trunk_h(vs[bi].to(device), vb[bi].to(device))
        sk, al = model.gen(h, vi[bi].to(device), im[bi].to(device))
        asg_l = assign_loss(al, asg[bi].to(device), im[bi].to(device))
        yb = y[bi].to(device)
        pair = pair_ce(sk[0::2], sk[1::2], yb[0::2], yb[1::2])
        sk_ce = F.cross_entropy(sk, yb)
        if arm == "M":
            total = sk_ce + asg_l
        elif arm == "MP":
            total = pair + asg_l
        elif arm == "MP+":
            total = sk_ce + pair + asg_l          # λ = 1.0 写死
        else:
            raise ValueError(arm)
        check(bool(torch.isfinite(total)), f"{arm} loss 非有限")
        opt.zero_grad(set_to_none=True)
        total.backward()
        torch.nn.utils.clip_grad_norm_(head_params, CLIP)
        opt.step()
        sched.step()
        losses[step] = float(total.detach())
        for k, v in (("total", total), ("skel", sk_ce), ("pair", pair), ("assign", asg_l)):
            parts_hist.setdefault(k, []).append(float(v.detach()))
        if step == 0:
            print(f"[step0] {arm} s{seed} loss={losses[0]:.4f} skel={float(sk_ce):.4f} "
                  f"pair={float(pair):.4f} assign={float(asg_l):.4f}", flush=True)
    model.eval()
    print(f"[train] {steps} 步 done，loss 首/末 {losses[0]:.4f}/{losses[steps - 1]:.4f}",
          flush=True)

    name = f"{arm}_s{seed}{'_rand' if rand else ''}"
    if steps != STEPS:
        name += f"_steps{steps}"
    RESULTS.mkdir(exist_ok=True)
    WEIGHTS.mkdir(exist_ok=True)
    torch.save({k: v for k, v in model.state_dict().items()
                if not k.startswith("encoder")}, WEIGHTS / f"{name}.pt")
    out = {"name": name, "arm": arm, "seed": seed, "randlabel": rand,
           "device": device, "freeze": freeze,
           "recipe": {"steps": steps, "batch_rows": PAIRS_PER_BATCH * 2,
                      "pairs_per_batch": PAIRS_PER_BATCH, "lr": LR, "weight_decay": WD,
                      "cosine": True, "grad_clip": CLIP, "n_rows_train": len(tr_rows),
                      "n_pairs_train": n_pairs, "head_trainable": n_head,
                      "epochs_approx": round(steps * PAIRS_PER_BATCH * 2 / len(tr_rows), 1),
                      "loss": {"M": "mean_row CE40 + CE_assign",
                               "MP": "mean_pair ½[CE{s,t}(A)+CE{s,t}(B)] + CE_assign",
                               "MP+": "mean_row CE40 + 1.0*pair + CE_assign"}[arm],
                      "lam": 1.0},
           "split": {k: v for k, v in st.items()
                     if k not in ("split_train_pairs", "split_heldout_pairs")},
           "loss_first": round(losses[0], 6), "loss_last": round(losses[steps - 1], 6),
           "loss_parts": {k: {"first": round(v[0], 6), "last": round(v[-1], 6),
                              "mean": round(sum(v) / len(v), 6)}
                          for k, v in parts_hist.items()},
           "wall_sec": round(time.time() - t0, 1)}
    (RESULTS / f"{name}.train.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[done] {name} → weights/{name}.pt（{out['wall_sec']}s）", flush=True)
    return out


def selfcheck(device: str) -> dict:
    spec = Spec()
    inits = {}
    for arm in ARMS:
        m = StructSupModel("B", 42, spec, vocab=None).to(device)
        fr = m.freeze_report()
        check(fr["encoder_trainable"] == 0, "核没冻住")
        check(not fr["encoder_training"], "核不在 eval()")
        inits[arm] = {k: v.detach().clone() for k, v in m.state_dict().items()
                      if not k.startswith("encoder")}
        del m
    ref = StructSupModel("B", 42, spec, vocab=None).to(device)
    ref_sd = {k: v.detach().clone() for k, v in ref.state_dict().items()
              if not k.startswith("encoder")}
    same = all(all(torch.equal(inits[a][k], ref_sd[k]) for k in ref_sd) for a in ARMS)
    check(same, "三臂初值不一致")
    rows = load_rows()
    prs = build_pairs(rows)
    tr_rows, ho_rows, st = split(rows, prs)
    blob = load_blob()
    check(len(tr_rows) == 560 and len(ho_rows) == 560, "切分行数漂移")
    check(blob["skel"].shape[0] == 1120, "blob 行数漂移")
    out = {"init_identical": bool(same), "encoder_trainable": 0,
           "split": {k: v for k, v in st.items()
                     if k not in ("split_train_pairs", "split_heldout_pairs")}}
    print(f"[selfcheck] 全通过：{json.dumps(out, ensure_ascii=False)}", flush=True)
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=list(ARMS))
    ap.add_argument("--seed", type=int, choices=list(SEEDS))
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args(argv)
    if a.selfcheck:
        selfcheck(a.device)
        return
    check(a.arm is not None and a.seed is not None, "需要 --arm 与 --seed")
    train_one(a.arm, a.seed, a.randlabel, a.steps, a.device)


if __name__ == "__main__":
    main(sys.argv[1:])
