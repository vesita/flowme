"""A/B 两档线性探针（冻结认知核）+ OOD —— 能力可加性筛查的「便宜先验」。

口径见 PREREG.md §4：
  A 档 `probe_cls`：样本级 mask mean-pool 特征 → 零初始化单层逻辑回归 → **首切片类别**口径
                    （分母 = 首切片标签 > 0 的样本，与 `evaluate_task.cls_acc` 完全一致）；
  B 档 `probe_exact`：逐位置零初始化线性探针（C 类 + 2 起点）→ 解码跨度 → **整句 exact** 口径。

两个口径**不许混**：A 对标卡的 `cls_acc`，B 对标卡的 `exact_match`。
"""
from __future__ import annotations

import argparse
import copy
import json
import pickle
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments" / "additivity"))
sys.path.insert(0, str(HERE))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402
from dtseek.tasks.plugin import probe_units_of, resolve_tasks  # noqa: E402
from dtseek.tasks.runtime import GenericTaskDataset, _truth_of  # noqa: E402

from prepare import CACHE, CAPS, TASK_SAMPLES, SEEDS  # noqa: E402

# ---- E4 探针的固定超参（照抄 experiments/adversarial_routing/probe_e4_style.py）----
LR, EPOCHS, WD, BS = 0.05, 200, 1e-4, 64
CAP_POS_TRAIN = 6000          # B 档逐位置特征的训练集上限（PREREG §4）
OOD_DROP_GT4 = 0              # 计数器：negation OOD 因 annotate_all 超 4 切片被丢弃的样本数


# ======================= 数据 / 划分 =======================
def split_of(cap: str, seed: int) -> tuple[list[dict], list[dict]]:
    ordered = pickle.loads((CACHE / f"{cap}_ordered_s{seed}.pkl").read_bytes())
    data = copy.deepcopy(ordered)
    random.Random(seed).shuffle(data)
    n_val = max(200, len(data) // 10)
    return data[:n_val], data[n_val:]


# ======================= 特征抽取 =======================
def extract_pooled(encoder, texts: list[str], max_len: int, device, bs: int = 64) -> torch.Tensor:
    """与 `GenericTaskDataset` 同一条编码路径：encode(max_length=max_len, padding=True)
    → collate 到 batch 最大长度 → 冻结核前向 → 按 attention_mask mean-pool。"""
    tok = NanoCharTokenizer()
    out = []
    for i in range(0, len(texts), bs):
        chunk = [tok.encode(t, max_length=max_len, padding=True) for t in texts[i:i + bs]]
        L = max(len(e["input_ids"]) for e in chunk)
        inps, masks = [], []
        for e in chunk:
            pad = L - len(e["input_ids"])
            inps.append(e["input_ids"] + [0] * pad)
            masks.append(e["attention_mask"] + [0] * pad)
        ti = torch.tensor(inps, dtype=torch.long, device=device)
        tm = torch.tensor(masks, dtype=torch.bool, device=device)
        with torch.no_grad():
            mem = encoder(ti, attention_mask=tm)
            m = tm.unsqueeze(-1).float()
            out.append(((mem * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu())
    return torch.cat(out, dim=0) if out else torch.empty(0, 128)


def make_loader(samples: list[dict], spec, bs: int = 64, shuffle: bool = False) -> DataLoader:
    return DataLoader(GenericTaskDataset(samples, NanoCharTokenizer(), spec),
                      batch_size=bs, shuffle=shuffle, drop_last=False)


def pass_pooled(encoder, loader, device) -> tuple[torch.Tensor, torch.Tensor]:
    """一遍前向，产出 (mean-pool 特征, 首切片标签)。"""
    xs, ys = [], []
    for batch in loader:
        inp = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            mem = encoder(inp, attention_mask=mask)
            m = mask.unsqueeze(-1).float()
            xs.append(((mem * m).sum(1) / m.sum(1).clamp(min=1.0)).cpu())
        ys.append(batch["labels"][:, 0].clone())
    return torch.cat(xs), torch.cat(ys).long()


def pass_positions(encoder, loader, device, spec) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """一遍前向，产出 (逐位置特征 [N,L,d] f16, 逐位置类别 [N,L], 逐位置起点 [N,L], mask)。"""
    f, lab, sta, msk = [], [], [], []
    for batch in loader:
        inp = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        with torch.no_grad():
            mem = encoder(inp, attention_mask=mask).cpu()
        f.append(mem.to(torch.float16))
        L = mem.shape[1]
        B = mem.shape[0]
        y_lab = torch.zeros(B, L, dtype=torch.long)
        y_sta = torch.zeros(B, L, dtype=torch.long)
        pos = torch.arange(L).unsqueeze(0)
        for i in range(spec.max_steps):
            s = batch["starts"][:, i]
            e = batch["ends"][:, i]
            lab_i = batch["labels"][:, i]
            m = (batch["step_mask"][:, i] > 0.5) & (lab_i > 0)
            if not bool(m.any()):
                continue
            inside = (pos >= s[:, None]) & (pos <= e[:, None]) & m[:, None]
            y_lab = torch.where(inside, lab_i[:, None].expand_as(y_lab), y_lab)
            y_sta = torch.where(inside & (pos == s[:, None]) & m[:, None],
                                torch.ones_like(y_sta), y_sta)
        lab.append(y_lab)
        sta.append(y_sta)
        msk.append(mask.cpu().bool())
    return (torch.cat(f), torch.cat(lab), torch.cat(sta), torch.cat(msk))


# ======================= 探针 =======================
def new_linear(d: int, k: int) -> nn.Linear:
    lin = nn.Linear(d, k)
    nn.init.zeros_(lin.weight)
    nn.init.zeros_(lin.bias)
    return lin


def probe_selfcheck_probeA(lin: nn.Linear, Xev: torch.Tensor, tag: str) -> dict:
    """自检 1/2（A 档）：可训参数量闭式对账 + 零初始化 ⇒ 未训练预测恒定。"""
    n_par = sum(p.numel() for p in lin.parameters() if p.requires_grad)
    closed = lin.weight.numel() + lin.bias.numel()
    with torch.no_grad():
        pred0 = lin(Xev).argmax(-1)
        uniq = sorted(set(pred0.tolist()))
    print(f"[SELFTEST_A1/{tag}] trainable={n_par} closed_form={closed} "
          f"match={n_par == closed} shape={tuple(lin.weight.shape)}", flush=True)
    print(f"[SELFTEST_A2/{tag}] untrained_pred_set={uniq} "
          f"(零权重零偏置 ⇒ argmax 恒为 0)", flush=True)
    return {"trainable": n_par, "closed_form": closed, "untrained_pred_set": uniq}


def train_A(Xtr, ytr, Xev, n_cls, seed, tag) -> tuple[nn.Linear, dict]:
    torch.manual_seed(seed)
    lin = new_linear(Xtr.shape[1], n_cls)
    # 自检 1 + 2 必须在**训练之前**做，否则"零初始化预测恒定"这条是空的
    sc = probe_selfcheck_probeA(lin, Xev, tag)
    opt = torch.optim.AdamW(lin.parameters(), lr=LR, weight_decay=WD)
    loader = DataLoader(TensorDataset(Xtr, ytr), batch_size=BS, shuffle=True)
    for _ in range(EPOCHS):
        lin.train()
        for bx, by in loader:
            opt.zero_grad()
            loss = F.cross_entropy(lin(bx), by)
            loss.backward()
            opt.step()
    lin.eval()
    return lin, sc


def train_B(ftr, ylab, ysta, mask, Xev, n_cls, seed, tag) -> tuple[nn.Linear, dict]:
    torch.manual_seed(seed)
    d = ftr.shape[2]
    lin = new_linear(d, n_cls + 2)
    n_par = sum(p.numel() for p in lin.parameters() if p.requires_grad)
    closed = lin.weight.numel() + lin.bias.numel()
    with torch.no_grad():
        o0 = lin(Xev.float())
        u_lab = sorted(set(o0[..., :n_cls].argmax(-1).reshape(-1).tolist()))
        u_sta = sorted(set(o0[..., n_cls:].argmax(-1).reshape(-1).tolist()))
        d0 = decode_positions(o0[..., :n_cls].argmax(-1).numpy(),
                              o0[..., n_cls:].argmax(-1).numpy(), n_cls)
    n_span0 = sum(len(x) for x in d0)
    print(f"[SELFTEST_B1/{tag}] trainable={n_par} closed_form={closed} match={n_par == closed} "
          f"shape={tuple(lin.weight.shape)}", flush=True)
    print(f"[SELFTEST_B2/{tag}] untrained_lab_set={u_lab} untrained_start_set={u_sta} "
          f"decoded_spans={n_span0} (零初始化 ⇒ 全 0 ⇒ 不解出任何跨度)", flush=True)
    sc = {"trainable": n_par, "closed_form": closed, "untrained_lab_set": u_lab,
          "untrained_start_set": u_sta, "decoded_spans_at_init": n_span0}
    # 逆频率权重（防全 0 退化解）
    cnt_l = torch.bincount(ylab.reshape(-1), minlength=n_cls).float().clamp(min=1.0)
    cnt_s = torch.bincount(ysta.reshape(-1), minlength=2).float().clamp(min=1.0)
    w_l = (cnt_l.sum() / (n_cls * cnt_l))
    w_s = (cnt_s.sum() / (2 * cnt_s))
    opt = torch.optim.AdamW(lin.parameters(), lr=LR, weight_decay=WD)
    N = ftr.shape[0]
    idx_all = torch.arange(N)
    for _ in range(EPOCHS):
        lin.train()
        perm = idx_all[torch.randperm(N)]
        for i in range(0, N, BS):
            sel = perm[i:i + BS]
            bx = ftr[sel].float()
            m = mask[sel]
            out = lin(bx)
            loss = (F.cross_entropy(out[..., :n_cls].reshape(-1, n_cls),
                                    ylab[sel].reshape(-1), weight=w_l, reduction="none")
                    * m.reshape(-1).float()).sum() / m.sum().clamp(min=1)
            loss = loss + (F.cross_entropy(out[..., n_cls:].reshape(-1, 2),
                                           ysta[sel].reshape(-1), weight=w_s, reduction="none")
                           * m.reshape(-1).float()).sum() / m.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            opt.step()
    lin.eval()
    return lin, sc


def decode_positions(pred_lab, pred_sta, n_cls) -> list[list[tuple[int, int, int]]]:
    out = []
    for row_l, row_s in zip(pred_lab, pred_sta):
        L = len(row_l)
        spans, t = [], 0
        while t < L:
            if row_s[t] == 1 and row_l[t] > 0:
                lab = int(row_l[t])
                e = t
                while e + 1 < L and int(row_l[e + 1]) == lab and int(row_s[e + 1]) == 0:
                    e += 1
                spans.append((lab, t, e))
                t = e + 1
            else:
                t += 1
        out.append(sorted(spans))
    return out


# ======================= OOD =======================
def ood_probe_units(cap: str, card) -> list[tuple[str, int]]:
    """各卡 `probe_units()` 的载体句（dev-notes/06 §7：与训练载体句式不重合）。"""
    rows = []
    for u in probe_units_of(card):
        if u.expected_class <= 0:
            continue
        for s in u.sentences():
            rows.append((s, int(u.expected_class)))
    return rows


def ood_compose() -> list[dict]:
    d = json.loads((ROOT / "experiments" / "compose_ops" / "dataset.json").read_text(encoding="utf-8"))
    return [c for c in d["cases"] if c["set"] == "negation"]


def build_compose_ood() -> tuple[list[dict], list[dict], dict]:
    """compose_ops 的 60 条「词典外否定表达」→ 两套金标样本。

    - negation 档：金标跨度 = `extract_negation_spans`（该函数是「什么算否定标记」的唯一真相源）；
      按 negation 卡的三条 fail-closed 不变量剔除（annotate_all 超 4 切片 / reject_reason 命中）。
    - sentiment 档：金标跨度 = 自带 `cue`（`cue_end` 是**闭区间**末位，转成半开要 +1），
      类别 = 自带 `label`（悲伤→3、愤怒→2，与 sentiment 卡 `classes` 下标一致）。
    """
    from dtseek.tasks.builtin.negation.dataset import extract_negation_spans, reject_reason

    compose = ood_compose()
    neg, sent = [], []
    dropped = 0
    for c in compose:
        if reject_reason(c["text"]) is not None:
            dropped += 1
            continue
        spans = [s for s in extract_negation_spans(c["text"]) if s["end"] <= 64]
        if len(spans) > 4:
            dropped += 1
            continue
        neg.append({"text": c["text"],
                    "spans": [{"label": s["label"], "start": s["start"], "end": s["end"]}
                              for s in spans]})
        sent.append({"text": c["text"],
                     "spans": [{"label": 3 if c["label"] == "悲伤" else 2,
                                "start": c["cue_start"], "end": c["cue_end"] + 1}]})
    meta = {"compose_n": len(compose), "compose_neg_n": len(neg),
            "compose_sent_n": len(sent), "compose_dropped": dropped}
    return neg, sent, meta


# ======================= 主流程 =======================
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="checkpoints/base_encoder.pt")
    ap.add_argument("--caps", default=",".join(CAPS))
    ap.add_argument("--out", default=str(HERE / "probe.json"))
    args = ap.parse_args(argv)
    caps = [c for c in args.caps.split(",") if c]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device} caps={caps} seeds={SEEDS}", flush=True)
    enc, _ck = load_base_encoder(args.base, device)
    enc.eval()
    cards = resolve_tasks(caps)

    result: dict = {"config": {"base": args.base, "lr": LR, "epochs": EPOCHS, "wd": WD,
                               "batch": BS, "cap_pos_train": CAP_POS_TRAIN,
                               "feature": "mask mean-pool of 冻结核 doc_memory",
                               "probeA": "Linear(128,C) 零初始化 AdamW lr0.05 200ep bs64",
                               "probeB": "Linear(128,C+2) 零初始化 逐位置 CE+CE(起点) 逆频率加权"},
                    "caps": {}}

    # ---- OOD 集合（与 seed 无关，先备好）----
    compose_neg, compose_sent, ood_meta = build_compose_ood()
    print(f"[ood] {ood_meta}", flush=True)
    result["ood_meta"] = ood_meta

    for cap in caps:
        card = cards[cap]
        spec = card.spec
        C = spec.num_classes
        t0 = time.perf_counter()
        pu = ood_probe_units(cap, card)
        pu_texts = [t for t, _ in pu]
        pu_y = torch.tensor([y for _, y in pu], dtype=torch.long)
        # negation OOD：60 条词典外否定表达（委托指定）；sentiment OOD：同 60 条按 cue 金标
        if cap == "negation":
            ood_spans = compose_neg
        elif cap == "sentiment":
            ood_spans = compose_sent
        else:
            ood_spans = []

        cap_entry: dict = {"num_classes": C, "class_names": list(spec.class_names)}
        for S in SEEDS:
            ev, tr = split_of(cap, S)
            ev_loader = make_loader(ev, spec, shuffle=False)
            tr_loader = make_loader(tr, spec, shuffle=False)

            # ---------- A 档 ----------
            Xev, yev = pass_pooled(enc, ev_loader, device)
            Xtr, ytr = pass_pooled(enc, tr_loader, device)
            lin, sc = train_A(Xtr, ytr, Xev, C, S, f"{cap}/s{S}")
            with torch.no_grad():
                pred = lin(Xev).argmax(-1)
            real = yev > 0
            cls_acc = float((pred[real] == yev[real]).float().mean()) if bool(real.any()) else None
            acc_all = float((pred == yev).float().mean())
            bg = ~real
            bg_acc = float((pred[bg] == yev[bg]).float().mean()) if bool(bg.any()) else None
            # 自检 3：权重离开初值
            with torch.no_grad():
                wmax = float(lin.weight.abs().max()); wstd = float(lin.weight.std())
            print(f"[SELFTEST_A3/{cap}/s{S}] W_max={wmax:.4e} W_std={wstd:.4e} (初值全 0)", flush=True)

            # ---------- B 档 ----------
            random.Random(S).shuffle(tr)
            cap_tr = tr[:CAP_POS_TRAIN]
            cap_loader = make_loader(cap_tr, spec, shuffle=False)
            Ftr, Ltr, Str, Mtr = pass_positions(enc, cap_loader, device, spec)
            Fev, Lev, Sev, Mev = pass_positions(enc, ev_loader, device, spec)
            linB, scB = train_B(Ftr, Ltr, Str, Mtr, Fev, C, S, f"{cap}/s{S}")
            n_par_b = scB["trainable"]
            closed_b = scB["closed_form"]
            with torch.no_grad():
                o = linB(Fev.float())
                pl = o[..., :C].argmax(-1)
                ps = o[..., C:].argmax(-1)
                wb = float(linB.weight.abs().max()); wsb = float(linB.weight.std())
            print(f"[SELFTEST_B3/{cap}/s{S}] W_max={wb:.4e} W_std={wsb:.4e} (初值全 0)", flush=True)
            got = decode_positions(pl.numpy(), ps.numpy(), C)
            probe_exact = _exact_count(ev, got, spec) / max(1, len(ev))

            entry = {
                "probe_cls_acc": cls_acc, "probe_acc_all": acc_all, "probe_bg_acc": bg_acc,
                "n_real": int(real.sum()), "n_bg": int(bg.sum()),
                "probe_exact": probe_exact, "n_eval": len(ev),
                "selftest": sc | scB | {"A_W_max": wmax, "A_W_std": wstd,
                                       "B_W_max": wb, "B_W_std": wsb},
                "probeA_n_train": int(Xtr.shape[0]), "probeB_n_train": int(Ftr.shape[0]),
            }

            # ---- OOD：probe_units 载体（只有 expected_class，只评 A 档口径）----
            if pu_texts:
                Xp = extract_pooled(enc, pu_texts, spec.max_len, device)
                with torch.no_grad():
                    pp = lin(Xp).argmax(-1)
                entry["ood_probe_units"] = {
                    "n": len(pu_texts), "acc": float((pp == pu_y).float().mean()),
                    "in_dist_acc_real": cls_acc,
                    "drop_pt": (cls_acc - float((pp == pu_y).float().mean())) * 100 if cls_acc is not None else None,
                }
            # ---- OOD：compose_ops 60（有跨度 ⇒ A/B 两档 + 卡都能评）----
            if ood_spans:
                ol = make_loader(ood_spans, spec, shuffle=False)
                Xo, yo = pass_pooled(enc, ol, device)
                with torch.no_grad():
                    po = lin(Xo).argmax(-1)
                real_o = yo > 0
                acc_o = float((po[real_o] == yo[real_o]).float().mean()) if bool(real_o.any()) else None
                Fo, Lo, So, Mo = pass_positions(enc, ol, device, spec)
                with torch.no_grad():
                    oo = linB(Fo.float())
                    go = decode_positions(oo[..., :C].argmax(-1).numpy(),
                                          oo[..., C:].argmax(-1).numpy(), C)
                ex_o = _exact_count(ood_spans, go, spec) / max(1, len(ood_spans))
                entry["ood_compose60"] = {
                    "n": len(ood_spans), "cls_acc": acc_o, "exact": ex_o,
                    "n_real": int(real_o.sum()),
                    "in_dist_cls_acc": cls_acc, "in_dist_exact": probe_exact,
                    "drop_cls_pt": (cls_acc - acc_o) * 100 if (cls_acc is not None and acc_o is not None) else None,
                    "drop_exact_pt": (probe_exact - ex_o) * 100,
                }
            cap_entry[str(S)] = entry
            print(f"[probe] {cap} s{S}: cls_acc={cls_acc:.4f} exact={probe_exact:.4f} "
                  f"(n_real={int(real.sum())}) {time.perf_counter()-t0:.1f}s", flush=True)

        result["caps"][cap] = cap_entry

    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2))
    print(f"[save] {args.out}")
    print("PROBE_DONE")
    return 0


def _exact_count(samples: list[dict], got: list, spec) -> int:
    """按 `evaluate_task` 的多重集规则算整句 exact。"""
    tok = NanoCharTokenizer()
    ds = GenericTaskDataset(samples, tok, spec)
    n = 0
    for i in range(len(ds)):
        item = ds[i]
        batch = {k: v.unsqueeze(0) for k, v in item.items()}
        truth = sorted(_truth_of(batch, 0, spec))
        if sorted(got[i]) == truth:
            n += 1
    return n


if __name__ == "__main__":
    sys.exit(main())
