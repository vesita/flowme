#!/usr/bin/env python3
"""P4 功能词最小对：模型与三种骨架监督 loss（唯一变量 = 监督形式）。

- 模型 = **只读复用** `experiments/struct_supervision/model.py::StructSupModel("B", seed)`
  ⇒ 输入特征 / Trunk / GenHead / 构造顺序 / RNG 与 `struct_supervision` B 臂逐位相同；
  三臂都用 `arm="B"` 构造 ⇒ **三臂初值逐位 `torch.equal`**。
- 核 1,688,460 全程冻结 + `eval()`，只训 `trunk`+`gen`（selfcheck 断言）。
- loss（PREREG §2.2，λ=1 写死）：
    M   = CE_skel(row) + CE_assign(row)                     （= struct B `gen_loss` 逐位同式）
    MP  = mean_{有伙伴行}( mean_{该行伙伴} ½[CE_{{s,t}}(row) + CE_{{s,t}}(var)] ) + CE_assign(row)
    MP+ = M + 1.0 × MP_pair
  `CE_{{s,t}}` = logits 限制到该对两个 gold 后的 2 类交叉熵（判「哪侧对应哪个骨架」）。

用法：uv run python experiments/funcword_minpair/model.py --selfcheck
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

HERE = Path(__file__).resolve().parent
SS_DIR = ROOT / "experiments" / "struct_supervision"
TCH_DIR = ROOT / "experiments" / "two_channel_head"
TCH_CACHE = TCH_DIR / "cache"
SL_DIR = ROOT / "experiments" / "skeleton_leak"
SL_CACHE = SL_DIR / "cache"
CACHE = HERE / "cache"

ARMS = ("M", "MP", "MP+")


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SS = _load(SS_DIR / "model.py", "fwmp_ss_model")      # 只读
Spec = SS.Spec
StructSupModel = SS.StructSupModel
v_bag_of = SS.v_bag_of
encode_rows_ss = SS.encode_rows                        # 只读参考（不调用：它写对方目录）


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[funcword_minpair model fail-closed] {msg}")


def fp(rows: list[dict]) -> str:
    return hashlib.md5("".join(r["sent"] + str(r["bag_span"]) for r in rows)
                       .encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------
# 编码：只读复用既有缓存；未命中才前向，且**只写本目录 cache/**
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_rows(model, rows: list[dict], spec: Spec, device: str,
                batch: int = 256) -> dict:
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"gen_{fp(rows)}_L{spec.max_len_sent}_M{spec.max_slots}.pt"
    if path.exists():
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob["n"] == len(rows):
            print(f"[enc] 命中本目录 {path.name}（n={blob['n']}）", flush=True)
            return blob
    v_sent, v_items, mask, skel, assign = [], [], [], [], []
    model.encoder.eval()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        ids, msk = [], []
        for r in chunk:
            e = tok.encode(r["sent"], max_length=spec.max_len_sent, padding=True)
            check(sum(e["attention_mask"]) == len(r["sent"]), "1 字符 1 token 前提被破坏")
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
            check(r["n_slots"] <= spec.max_slots, f"槽位数越界 {r['n_slots']}")
            for s, (a, b) in enumerate(r["bag_span"]):
                check(r["sent"][a:b] == r["bag"][s], "span 与 bag 文本不一致")
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
    print(f"[enc] 写入本目录 {path.name}（n={len(rows)}, "
          f"{path.stat().st_size / 1e6:.0f} MB）", flush=True)
    return blob


def _from_cache(path: Path, n: int) -> dict | None:
    if not path.exists():
        return None
    b = torch.load(path, map_location="cpu", weights_only=True)
    if b.get("n") != n:
        return None
    return b


def get_blob(model, rows: list[dict], spec: Spec, device: str, split: str) -> dict:
    """train/test → 只读 two_channel_head 缓存；skeleton_leak 集 → 只读其缓存；否则本目录编码。"""
    f = fp(rows)
    n = len(rows)
    cands = [TCH_CACHE / f"gen_{f}_L{spec.max_len_sent}_M{spec.max_slots}.pt",
             SL_CACHE / f"enc_{f}_L{spec.max_len_sent}.pt",
             CACHE / f"gen_{f}_L{spec.max_len_sent}_M{spec.max_slots}.pt"]
    for p in cands:
        b = _from_cache(p, n)
        if b is not None:
            print(f"[blob] {split} ← 只读缓存 {p}（n={b['n']}）", flush=True)
            return {k: b[k] for k in
                    ("v_sent", "v_items", "item_mask", "skel", "assign")} | {"n": b["n"]}
    print(f"[blob] {split} 缓存未命中（fp={f}），本目录编码", flush=True)
    return encode_rows(model, rows, spec, device)


# ---------------------------------------------------------------------------
# loss（PREREG §2.2）
# ---------------------------------------------------------------------------
def assign_loss(a_logits: torch.Tensor, assign_y: torch.Tensor,
                item_mask: torch.Tensor) -> torch.Tensor:
    """与 `struct_supervision/model.py::gen_loss` 的指派项**逐字同式**。"""
    n = item_mask.sum(1).long()
    B, M, _ = a_logits.shape
    slots = torch.arange(M, device=a_logits.device)
    valid = (slots.view(1, -1) < n.view(-1, 1)) & item_mask
    flat = a_logits.reshape(B * M, M)
    tgt = assign_y.clamp(min=0).reshape(B * M)
    ce = F.cross_entropy(flat, tgt, reduction="none")
    vm = valid.reshape(B * M).float()
    return (ce * vm).sum() / vm.sum().clamp(min=1.0)


def skel_ce(skel_logits: torch.Tensor, skel_y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(skel_logits, skel_y)


def pair_ce(logits_a: torch.Tensor, logits_b: torch.Tensor,
            ya: torch.Tensor, yb: torch.Tensor) -> torch.Tensor:
    """两侧各自在**该对两个 gold {s,t}** 上做 2 类交叉熵，返回逐行均值（未掩码）。

    logits_a → 目标 = ya（索引 0）；logits_b → 目标 = yb（索引 1）。
    """
    la = logits_a.gather(1, ya.view(-1, 1)).squeeze(1)
    lb = logits_a.gather(1, yb.view(-1, 1)).squeeze(1)
    ta = F.cross_entropy(torch.stack([la, lb], 1),
                         torch.zeros(len(ya), dtype=torch.long, device=ya.device))
    lc = logits_b.gather(1, ya.view(-1, 1)).squeeze(1)
    ld = logits_b.gather(1, yb.view(-1, 1)).squeeze(1)
    tb = F.cross_entropy(torch.stack([lc, ld], 1),
                         torch.ones(len(yb), dtype=torch.long, device=yb.device))
    return 0.5 * (ta + tb)


def arm_loss(model, arm: str, b: dict) -> tuple[torch.Tensor, dict]:
    """b 的键见 `train.py::make_dataset`。返回 (total, parts)。"""
    h_row = model.trunk_h(b["vs"], b["vb"])
    sk_row, a_row = model.gen(h_row, b["vi"], b["im"])
    asg = assign_loss(a_row, b["asg"], b["im"])
    parts: dict = {"assign": asg.detach()}

    # 伙伴变体前向（三臂都做 ⇒ 输入序列完全相同）
    h_var = model.trunk_h(b["vsv"], b["vbv"])
    sk_var, _ = model.gen(h_var, b["viv"], b["imv"])
    h_ord = model.trunk_h(b["vso"], b["vbo"])
    sk_ord, _ = model.gen(h_ord, b["vio"], b["imo"])

    hw = b["has_word"].float()
    ho = b["has_order"].float()
    hp = ((hw + ho) > 0).float()
    nrow = hp.sum().clamp(min=1.0)

    ce_w = pair_ce(sk_row, sk_var, b["y"], b["yv_word"])
    ce_o = pair_ce(sk_row, sk_ord, b["y"], b["yv_order"])
    per_row = (ce_w * hw + ce_o * ho) / (hw + ho).clamp(min=1.0)
    pair = (per_row * hp).sum() / nrow

    parts["pair"] = pair.detach()
    if arm == "M":
        total = skel_ce(sk_row, b["y"]) + asg
        parts["skel"] = total.detach() - asg.detach()
    elif arm == "MP":
        total = pair + asg
        parts["skel"] = pair.detach()
    elif arm == "MP+":
        sk = skel_ce(sk_row, b["y"])
        total = sk + asg + pair                      # λ = 1 写死
        parts["skel"] = (sk + pair).detach()
    else:
        raise ValueError(arm)
    assert torch.isfinite(total), f"{arm} loss 非有限"
    return total, parts


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def selfcheck(device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = Spec()
    out: dict = {}

    # 1) 核冻结 + 三臂初值逐位相同 + == struct B 同 seed
    inits = {}
    for arm in ARMS:
        m = StructSupModel("B", 42, spec, vocab=None).to(device)
        m.encoder.eval()
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
    check(same, "三臂初值与 struct B 同 seed 初值不一致")
    out["init_identical_across_arms_and_structB"] = bool(same)

    # 2) M 臂 loss == struct B `gen_loss` 逐位相等
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    rows = [json.loads(x) for x in
            open(ROOT / "experiments/two_channel_head/data/train.jsonl",
                 encoding="utf-8")][:64]
    model = StructSupModel("B", 42, spec, vocab=None).to(device)
    ids, msk = [], []
    for r in rows:
        e = tok.encode(r["sent"], max_length=spec.max_len_sent, padding=True)
        ids.append(e["input_ids"])
        msk.append(e["attention_mask"])
    id_t = torch.tensor(ids, dtype=torch.long, device=device)
    m_t = torch.tensor(msk, dtype=torch.bool, device=device)
    with torch.no_grad():
        hh = model.encoder(id_t, m_t)
    m3 = m_t.unsqueeze(-1).to(hh.dtype)
    vs = (hh * m3).sum(1) / m3.sum(1).clamp(min=1.0)
    vi = torch.zeros(len(rows), spec.max_slots, spec.hidden, device=device)
    im = torch.zeros(len(rows), spec.max_slots, dtype=torch.bool, device=device)
    for j, r in enumerate(rows):
        for s, (a, b) in enumerate(r["bag_span"]):
            vi[j, s] = hh[j, a:b].mean(0)
            im[j, s] = True
    vb = v_bag_of(vi, im)
    y = torch.tensor([r["skel_id"] for r in rows], device=device)
    asg = torch.tensor([r["assign"] + [-1] * (spec.max_slots - len(r["assign"]))
                        for r in rows], device=device)
    with torch.no_grad():
        ref_tot, ref_sk, ref_asg = model.gen_loss(vs, vb, vi, im, y, asg)
        h = model.trunk_h(vs, vb)
        sk_l, a_l = model.gen(h, vi, im)
        mine = skel_ce(sk_l, y) + assign_loss(a_l, asg, im)
    d = float((ref_tot - mine).abs())
    check(d == 0.0, f"M 臂 loss 与 struct B gen_loss 不等：{d}")
    out["M_loss_equals_structB_maxdiff"] = d
    del model, ref
    torch.cuda.empty_cache() if device.startswith("cuda") else None

    # 3) 最小对构造自检（读 build_data 产物）
    st = json.loads((HERE / "data" / "stats_build.json").read_text(encoding="utf-8"))
    check(len(st["rules"]["R1"]) == 62 and len(st["rules"]["R2"]) == 21
          and len(st["rules"]["R3"]) == 1, "规则枚举数与 PREREG 不符")
    check(st["cover"]["cand_rows"] == 4274, "候选行数与 PREREG 不符")
    check(st["cover"]["covered_rows_with_word_variant"] == 4205, "word 伙伴行数与 PREREG 不符")
    check(all(v == 0 for k, v in st["sep"].items() if k != "variant ∩ train"),
          f"分离断言非 0：{st['sep']}")
    check(st["heldout"]["b1_pairs"] == 520 and st["heldout"]["b2_pairs"] == 40,
          "held-out 对数与 PREREG 不符")
    out["build"] = {"R1": len(st["rules"]["R1"]), "R2": len(st["rules"]["R2"]),
                    "R3": len(st["rules"]["R3"]),
                    "cand_rows": st["cover"]["cand_rows"],
                    "word_rows": st["cover"]["covered_rows_with_word_variant"],
                    "r3_ok": st["cover"]["r3_ok"], "sep": st["sep"],
                    "b1": st["heldout"]["b1_pairs"], "b2": st["heldout"]["b2_pairs"]}

    # 4) 三臂 loss 数值不同（仅 smoke，不作结论）
    print(f"[selfcheck] 全通过：{json.dumps(out, ensure_ascii=False)}", flush=True)
    return out


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args(argv)
    if a.selfcheck:
        selfcheck(a.device)
    else:
        ap.error("需要 --selfcheck")


if __name__ == "__main__":
    main(sys.argv[1:])
