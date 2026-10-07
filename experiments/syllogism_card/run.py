#!/usr/bin/env python3
"""syllogism_card：两阶段训练 + 全量预测（核全程冻结，三张头核外私有）。

  stage1 识别卡（字符级标注）  →  结构指令 → render.py 两道检查（fail-closed）
  stage2 推理头 + 归因卡（输入 = 识别卡规范化文本）

用法：
  uv run python experiments/syllogism_card/run.py --arm T  --seed 42
  uv run python experiments/syllogism_card/run.py --arm TA --seed 43 --randlabel
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from nano_char_tokenizer import NanoCharTokenizer  # noqa: E402
from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402
from dtseek.tasks.render import (  # noqa: E402
    BagItem, Instruction, check_deref, check_structure, render)

sys.path.insert(0, str(HERE))
from gen_data import (  # noqa: E402
    N_ATTR, N_LABEL, REJECT, ATTR_NONE, TAGS, base_of, parse_tags, r_chain,
    r_keyword, r_lexdir, Q_LEX)

BASE_CKPT = ROOT / "checkpoints" / "base_encoder.pt"
DATA = HERE / "data.json"
OUT = HERE / "results"
WEIGHTS = HERE / "weights"
LOGS = HERE / "logs"
for p in (OUT, WEIGHTS, LOGS):
    p.mkdir(exist_ok=True)

STEPS = 1000
BATCH = 64
LR = 1e-3
WD = 1e-4
CLIP = 1.0
MAXLEN_PREM = 24
MAXLEN_CANON = 48
ARMS = {"T": ("tmpl",), "TA": ("tmpl", "para"), "A": ("para",)}


def seed_all(s: int) -> None:
    random.seed(s)
    torch.manual_seed(s)


def load_items() -> list[dict]:
    d = json.loads(DATA.read_text(encoding="utf-8"))
    return d["items"]


# ---------------- 编码（核冻结 ⇒ 可缓存） ----------------
_tok = NanoCharTokenizer()


@torch.no_grad()
def encode(enc, texts: list[str], max_len: int, device: str, bs: int = 512):
    hs, ms = [], []
    for i in range(0, len(texts), bs):
        ids, msk = [], []
        for t in texts[i:i + bs]:
            e = _tok.encode(t, max_length=max_len, padding=True)
            ids.append(e["input_ids"])
            msk.append(e["attention_mask"])
        id_t = torch.tensor(ids, dtype=torch.long, device=device)
        m_t = torch.tensor(msk, dtype=torch.bool, device=device)
        hs.append(enc(id_t, m_t))
        ms.append(m_t)
    return torch.cat(hs, 0), torch.cat(ms, 0)


def masked_mean(h, m):
    mf = m.unsqueeze(-1).to(h.dtype)
    return (h * mf).sum(1) / mf.sum(1).clamp(min=1.0)


# ---------------- 模型（核外私有） ----------------
class Adapter(nn.Module):
    """核外私有适配层：核输出之后，残差 + fc2 零初始化 ⇒ step-0 逐位恒等。"""

    def __init__(self, hidden=128, mid=256):
        super().__init__()
        self.fc1 = nn.Linear(hidden, mid)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mid, hidden)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        return x + self.fc2(self.act(self.fc1(x)))


def mlp(a, b, c):
    return nn.Sequential(nn.Linear(a, b), nn.GELU(), nn.Linear(b, c))


class Tagger(nn.Module):
    def __init__(self):
        super().__init__()
        self.pos = nn.Embedding(32, 128)
        self.adapter = Adapter()
        self.head = mlp(128, 256, len(TAGS))

    def forward(self, h, m):
        L = h.size(1)
        idx = torch.arange(L, device=h.device)
        x = h + self.pos(idx).unsqueeze(0)
        return self.head(self.adapter(x))


class Reasoner(nn.Module):
    def __init__(self):
        super().__init__()
        self.adapter = Adapter()
        self.head_c = mlp(128, 256, N_LABEL)
        self.emb_c = nn.Embedding(N_LABEL, 32)
        self.head_a = mlp(128 + 32, 256, N_ATTR)

    def feat(self, h, m):
        return masked_mean(self.adapter(h), m)

    def forward(self, h, m, concl: torch.Tensor):
        v = self.feat(h, m)
        lc = self.head_c(v)
        ea = self.emb_c(concl)
        la = self.head_a(torch.cat([v, ea], -1))
        return lc, la


# ---------------- 识别卡输出 → 结构指令 → render.py 两道检查 ----------------
def build_structures(item: dict, pred_tags: list[list[int]]) -> dict | None:
    premises = item["premises"]
    structs = [parse_tags(p, t) for p, t in zip(premises, pred_tags)]
    if any(s is None for s in structs):
        return None
    records, bags = [], []
    for pi, (premise, st) in enumerate(zip(premises, structs)):
        bag: list[BagItem] = []
        if st["kind"] == "binary":
            n1, rel, n2 = st["names"][0], st["rel"], st["names"][1]
            bag = [
                BagItem(1, premise[n1["start"]:n1["end"]], (n1["start"], n1["end"]),
                        "名", "施事", f"rec:{item['id']}:{pi}:0", True),
                BagItem(2, premise[rel["start"]:rel["end"]], (rel["start"], rel["end"]),
                        "动", None, f"rec:{item['id']}:{pi}:1", True),
                BagItem(3, premise[n2["start"]:n2["end"]], (n2["start"], n2["end"]),
                        "名", "受事", f"rec:{item['id']}:{pi}:2", True),
            ]
            instr = Instruction("S01", (1, 2, 3))
        else:
            n1, rel = st["names"][0], st["rel"]
            bag = [
                BagItem(1, premise[n1["start"]:n1["end"]], (n1["start"], n1["end"]),
                        "名", "施事", f"rec:{item['id']}:{pi}:0", True),
                BagItem(2, premise[rel["start"]:rel["end"]], (rel["start"], rel["end"]),
                        "动", None, f"rec:{item['id']}:{pi}:1", True),
            ]
            instr = Instruction("S04", (1, 2))
        probs = check_structure(instr, bag, premise)
        if probs:
            return None
        rec = render(instr, bag, premise, plan_step_id=f"item:{item['id']}:p{pi}")
        if rec["kind"] != "text":
            return None
        if check_deref(rec, bag, premise):
            return None
        records.append(rec)
        bags.append(bag)

    mentions: list[dict] = []
    for pi, st in enumerate(structs):
        for nm in st["names"]:
            s, e = nm["start"], nm["end"]
            mentions.append({"p": pi, "start": s, "end": e,
                             "text": premises[pi][s:e], "base": base_of(premises[pi][s:e])})
    ids: dict[str, int] = {}
    for mi, m in enumerate(mentions):
        m["ref"] = mi
        if m["base"] not in ids:
            ids[m["base"]] = len(ids)
        m["eid"] = ids[m["base"]]
    return {"canonical": "".join(r["text"] for r in records),
            "mentions": mentions, "ids": ids, "records": [r for r in records],
            "n_eid": len(ids)}


# ---------------- 主流程 ----------------
def main() -> int:
    global STEPS
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=list(ARMS))
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--randlabel", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--steps", type=int, default=STEPS,
                    help="冒烟用；正式跑用默认 1000（PREREG §3 写死）")
    a = ap.parse_args()
    STEPS = a.steps
    t0 = time.time()
    seed_all(a.seed)
    tag = f"{a.arm}_s{a.seed}" + ("_rand" if a.randlabel else "")
    log = open(LOGS / f"run_{tag}.log", "w", buffering=1)

    def say(*x):
        print(*x, flush=True)
        log.write(" ".join(str(y) for y in x) + "\n")

    items = load_items()
    train_exits = set(ARMS[a.arm])
    tr = [it for it in items if it["split"] == "train" and it["exit"] in train_exits]
    say(f"[{tag}] items={len(items)} train={len(tr)} device={a.device}")

    enc, _ = load_base_encoder(str(BASE_CKPT), a.device)
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    enc_p = sum(p.numel() for p in enc.parameters())
    say(f"[core] params={enc_p} training={enc.training} trainable="
        f"{sum(p.numel() for p in enc.parameters() if p.requires_grad)}")

    # ---- 阶段 1：识别卡 ----
    prem_all, prem_owner, tag_all = [], [], []
    for it in items:
        for k, p in enumerate(it["premises"]):
            prem_all.append(p)
            prem_owner.append((it["id"], k))
            tag_all.append(it["gold_tags"][k])
    say(f"[enc] encode {len(prem_all)} premises ...")
    h_p, m_p = encode(enc, prem_all, MAXLEN_PREM, a.device)
    say(f"[enc] h_p={tuple(h_p.shape)}")

    tr_ids = {x["id"] for x in tr}
    idx_tr = [i for i, (iid, k) in enumerate(prem_owner) if iid in tr_ids]
    say(f"[s1] stage1 premises={len(idx_tr)}")
    tg = Tagger().to(a.device)
    n_tag = sum(p.numel() for p in tg.parameters())
    opt = torch.optim.AdamW(tg.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    tg.train()
    g = torch.Generator().manual_seed(a.seed * 1000 + 7)
    first_loss = None
    for step in range(STEPS):
        sel = torch.tensor(idx_tr)[torch.randint(len(idx_tr), (BATCH,), generator=g)]
        hh, mm = h_p[sel], m_p[sel]
        tgt = torch.full((BATCH, hh.size(1)), -100, dtype=torch.long, device=a.device)
        for bi, si in enumerate(sel.tolist()):
            gold = torch.tensor(tag_all[si], dtype=torch.long)
            tgt[bi, :len(gold)] = gold.to(a.device)
        logits = tg(hh, mm)
        loss = F.cross_entropy(logits.reshape(-1, len(TAGS)), tgt.reshape(-1))
        if first_loss is None:
            first_loss = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(tg.parameters(), CLIP)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == STEPS - 1:
            say(f"[s1] step {step} loss {float(loss.detach()):.4f}")
    tg.eval()
    say(f"[s1] loss_first={first_loss:.6f} loss_last={float(loss.detach()):.4f}")

    # 识别卡预测（全量）
    with torch.no_grad():
        logits = tg(h_p, m_p)
        pred_tags = logits.argmax(-1)
    struct_by_id: dict[int, dict | None] = {}
    tagok: dict[tuple, float] = {}
    owner_idx = {(iid, k): i for i, (iid, k) in enumerate(prem_owner)}
    for i, (iid, k) in enumerate(prem_owner):
        gold = torch.tensor(tag_all[i])
        tagok[(iid, k)] = float(
            (pred_tags[i, :len(gold)].cpu() == gold).float().mean())
    for it in items:
        pts = [pred_tags[owner_idx[(it["id"], k)]][:len(it["gold_tags"][k])].tolist()
               for k in range(2)]
        struct_by_id[it["id"]] = build_structures(it, pts)
    n_ok = sum(1 for v in struct_by_id.values() if v is not None)
    say(f"[s1] 结构可解析(过两道检查) {n_ok}/{len(items)}")

    # ---- 阶段 2：推理头 + 归因卡 ----
    tr_ok = [it for it in tr if struct_by_id[it["id"]] is not None]
    say(f"[s2] 可训条数={len(tr_ok)} (train={len(tr)})")
    can_tr = [struct_by_id[it["id"]]["canonical"] for it in tr_ok]
    h_c, m_c = encode(enc, can_tr, MAXLEN_CANON, a.device)
    y_c = torch.tensor([it["label"] for it in tr_ok], dtype=torch.long)
    y_a = torch.tensor([it["attr"] for it in tr_ok], dtype=torch.long)
    if a.randlabel:
        pr = torch.Generator().manual_seed(a.seed * 1000 + 7)
        y_c = y_c[torch.randperm(len(y_c), generator=pr)]
        y_a = y_a[torch.randperm(len(y_a), generator=pr)]
        say(f"[s2] 随机标签对照：train 标签已 randperm（分布="
            f"{dict(Counter(y_c.tolist()).most_common(3))}）")

    rs = Reasoner().to(a.device)
    n_priv = sum(p.numel() for p in tg.parameters()) + sum(p.numel() for p in rs.parameters())
    say(f"[priv] tagger={sum(p.numel() for p in tg.parameters())} "
        f"reasoner={sum(p.numel() for p in rs.parameters())} 合计={n_priv} "
        f"核={enc_p} 占比={n_priv / enc_p * 100:.2f}%")
    opt = torch.optim.AdamW(rs.parameters(), lr=LR, weight_decay=WD)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=STEPS)
    rs.train()
    g = torch.Generator().manual_seed(a.seed * 1000 + 13)
    first2 = None
    for step in range(STEPS):
        sel = torch.randint(len(tr_ok), (BATCH,), generator=g)
        lc, la = rs(h_c[sel], m_c[sel], y_c[sel])
        loss = F.cross_entropy(lc, y_c[sel].to(a.device)) + \
            F.cross_entropy(la, y_a[sel].to(a.device))
        if first2 is None:
            first2 = float(loss.detach())
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(rs.parameters(), CLIP)
        opt.step()
        sched.step()
        if step % 200 == 0 or step == STEPS - 1:
            say(f"[s2] step {step} loss {float(loss.detach()):.4f}")
    rs.eval()
    say(f"[s2] loss_first={first2:.6f} loss_last={float(loss.detach()):.4f}")

    # ---- 全量预测（test 两出口 + train 两出口） ----
    all_ok = [it for it in items if struct_by_id[it["id"]] is not None]
    h_all, m_all = encode(enc, [struct_by_id[it["id"]]["canonical"] for it in all_ok],
                          MAXLEN_CANON, a.device)
    idx_of = {it["id"]: i for i, it in enumerate(all_ok)}
    preds: dict[int, dict] = {}
    with torch.no_grad():
        for i, it in enumerate(all_ok):
            st = struct_by_id[it["id"]]
            hg = h_all[i:i + 1]
            mg = m_all[i:i + 1]
            g_lab = torch.tensor([it["label"]], device=a.device)
            lc, la_gold = rs(hg, mg, g_lab)
            pl = int(lc.argmax(-1))
            if pl != REJECT:
                _, la_dep = rs(hg, mg, torch.tensor([pl], device=a.device))
                pa = int(la_dep.argmax(-1))
            else:
                pa = ATTR_NONE
            pa_oracle = int(la_gold.argmax(-1)) if it["label"] != REJECT else ATTR_NONE
            refs = None
            if pa != ATTR_NONE:
                r1, r2 = pa // 6, pa % 6
                mm = st["mentions"]
                refs = ([mm[r1]] if r1 < len(mm) else [None]) + \
                       ([mm[r2]] if r2 < len(mm) else [None])
            preds[it["id"]] = {"pred_label": pl, "pred_attr": pa,
                               "pred_attr_oracle": pa_oracle, "refs": refs,
                               "canonical": st["canonical"]}
    out = []
    for it in items:
        p = preds.get(it["id"])
        tg_acc = sum(tagok[(it["id"], k)] for k in range(2)) / 2
        out.append({
            "id": it["id"], "exit": it["exit"], "split": it["split"],
            "type": it["type"], "sub": it["sub"],
            "premises": it["premises"],
            "gold_label": it["label"], "gold_attr": it["attr"],
            "pred_label": p["pred_label"] if p else REJECT,
            "pred_attr": p["pred_attr"] if p else ATTR_NONE,
            "pred_attr_oracle": p["pred_attr_oracle"] if p else ATTR_NONE,
            "refs": p["refs"] if p else None,
            "canonical": p["canonical"] if p else "",
            "parse_ok": p is not None,
            "tag_acc": round(tg_acc, 4),
        })
    (OUT / f"pred_{tag}.json").write_text(json.dumps(out, ensure_ascii=False),
                                          encoding="utf-8")
    meta = {"tag": tag, "arm": a.arm, "seed": a.seed, "randlabel": a.randlabel,
            "steps": STEPS, "batch": BATCH, "lr": LR, "wd": WD,
            "loss_s1_first": first_loss, "loss_s2_first": first2,
            "core_params": enc_p, "private_params": n_priv,
            "n_train": len(tr), "n_train_ok": len(tr_ok),
            "struct_ok": n_ok, "n_items": len(items),
            "device": a.device, "wall_sec": round(time.time() - t0, 1)}
    (OUT / f"meta_{tag}.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    torch.save({"tagger": tg.state_dict(), "reasoner": rs.state_dict()},
               WEIGHTS / f"{tag}.pt")
    say(f"[done] {tag} wall={meta['wall_sec']}s -> results/pred_{tag}.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
