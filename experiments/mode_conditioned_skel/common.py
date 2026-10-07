#!/usr/bin/env python3
"""P8 公共件：只读加载模型/标签口径 + 骨架编码（优先复用 skeleton_leak 只读缓存）。

- 权重只读 `experiments/bag_modules/weights/{A,U,P,UP}_s{42,43}.pt`（strict=False，
  断言缺键只许 `encoder.*`）；**不写任何既有目录**。
- 编码口径与 `skeleton_leak/eval_arms.py::encode` 逐字同式；缓存查找顺序：
  ① `experiments/skeleton_leak/cache/enc_<fp>_L64.pt`（只读复用）
  ② 本目录 `cache/`（写这里）。
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
HERE = Path(__file__).resolve().parent
BM = ROOT / "experiments" / "bag_modules"
TCH = ROOT / "experiments" / "two_channel_head"
SS = ROOT / "experiments" / "struct_supervision"
SLEAK = ROOT / "experiments" / "skeleton_leak"
CACHE = HERE / "cache"
SLEAK_CACHE = SLEAK / "cache"
ARMS = ("A", "U", "P", "UP")
SEEDS = (42, 43)


def load_mod(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


M = load_mod(BM / "model.py", "msk_bagmod_ro")            # 只读
BL = load_mod(BM / "build_labels.py", "msk_labels_ro")    # 只读
Spec, BagModModel, v_bag_of = M.Spec, M.BagModModel, M.v_bag_of
label_row = BL.label_row


def check(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(f"[mode_conditioned_skel fail-closed] {msg}")


def load_rows(p: Path) -> list[dict]:
    with open(p, encoding="utf-8") as f:
        return [json_loads(x) for x in f]


def json_loads(s: str):
    import json
    return json.loads(s)


def fp_of(rows: list[dict]) -> str:
    return hashlib.md5("".join(r["sent"] + str(r["bag_span"]) for r in rows)
                       .encode()).hexdigest()[:12]


@torch.no_grad()
def encode_rows(rows: list[dict], spec, device: str) -> dict:
    """返回 v_sent/v_items/item_mask/skel/assign/h/hmask/n（口径同 skeleton_leak）。"""
    from nano_char_tokenizer import NanoCharTokenizer
    fp = fp_of(rows)
    name = f"enc_{fp}_L{spec.max_len_sent}.pt"
    for p in (SLEAK_CACHE / name, CACHE / name):
        if p.exists():
            b = torch.load(p, map_location="cpu", weights_only=True)
            if b.get("n") == len(rows):
                print(f"[enc] 命中 {p}（n={b['n']}）", flush=True)
                return b
    tok = NanoCharTokenizer()
    ref = BagModModel("A", 42, spec).to(device)
    ref.encoder.eval()
    v_sent, v_items, mask, hs, ms, skel, assign = [], [], [], [], [], [], []
    bs = 256
    for i in range(0, len(rows), bs):
        chunk = rows[i:i + bs]
        ids, msk = [], []
        for r in chunk:
            e = tok.encode(r["sent"], max_length=spec.max_len_sent, padding=True)
            check(sum(e["attention_mask"]) == len(r["sent"]), "1 字符 1 token 被破坏")
            ids.append(e["input_ids"])
            msk.append(e["attention_mask"])
        id_t = torch.tensor(ids, dtype=torch.long, device=device)
        m_t = torch.tensor(msk, dtype=torch.bool, device=device)
        h = ref.encoder(id_t, m_t).cpu()
        m3 = m_t.cpu().unsqueeze(-1).to(h.dtype)
        v_sent.append((h * m3).sum(1) / m3.sum(1).clamp(min=1.0))
        hs.append(h)
        ms.append(m_t.cpu())
        items = torch.zeros(len(chunk), spec.max_slots, spec.hidden)
        im = torch.zeros(len(chunk), spec.max_slots, dtype=torch.bool)
        for j, r in enumerate(chunk):
            for s, (a, b) in enumerate(r["bag_span"]):
                check(r["sent"][a:b] == r["bag"][s], "span 与 bag 不一致")
                items[j, s] = h[j, a:b].mean(0)
                im[j, s] = True
        v_items.append(items)
        mask.append(im)
        skel += [r["skel_id"] for r in chunk]
        assign += [r["assign"] + [-1] * (spec.max_slots - len(r["assign"])) for r in chunk]
    blob = {"v_sent": torch.cat(v_sent), "v_items": torch.cat(v_items),
            "item_mask": torch.cat(mask), "skel": torch.tensor(skel),
            "assign": torch.tensor(assign), "h": torch.cat(hs), "hmask": torch.cat(ms),
            "n": len(rows)}
    CACHE.mkdir(exist_ok=True)
    torch.save(blob, CACHE / name)
    print(f"[enc] 写入 {CACHE / name}（n={len(rows)}）", flush=True)
    return blob


def label_tensors(rows: list[dict], pos_bucket: int = 4) -> dict:
    n = len(rows)
    m = max(r["n_slots"] for r in rows)
    tt = torch.zeros(n, m, dtype=torch.long)
    rt = torch.zeros(n, m, dtype=torch.long)
    ct = torch.zeros(n, m, dtype=torch.long)
    pb = torch.zeros(n, m, dtype=torch.long)
    mk = torch.zeros(n, m, dtype=torch.bool)
    for i, r in enumerate(rows):
        lab = label_row(r)
        check(len(lab) == r["n_slots"], "标签数与槽位不符")
        order = sorted(range(r["n_slots"]), key=lambda k: r["bag_span"][k][0])
        L = max(1, len(r["sent"]))
        for j, k in enumerate(order):
            a, _b = r["bag_span"][k]
            tt[i, j] = lab[k]["t"]
            rt[i, j] = lab[k]["r"]
            ct[i, j] = lab[k]["c"]
            pb[i, j] = min(pos_bucket - 1, a * pos_bucket // L)
            mk[i, j] = True
    return {"type_t": tt, "role_t": rt, "cls_t": ct, "pos_b": pb, "mask": mk}


def load_arm(arm: str, seed: int, spec, device: str) -> BagModModel:
    m = BagModModel(arm, seed, spec).to(device)
    sd = torch.load(BM / "weights" / f"{arm}_s{seed}.pt",
                    map_location="cpu", weights_only=True)
    miss = [k for k in m.state_dict() if k not in sd]
    check(all(k.startswith("encoder.") for k in miss), f"{arm}_s{seed} 缺键异常：{miss[:5]}")
    m.load_state_dict(sd, strict=False)
    m.eval()
    return m
