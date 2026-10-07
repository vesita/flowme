#!/usr/bin/env python3
"""struct_supervision 三臂模型（同一模型类，只换监督形式）。

臂（PREREG §1）：
  A = 整句 loss（teacher-forced 字级 CE）→ 结构只能由「自由解码 + match_sentence 解析」给出
  B = 槽位监督（骨架 id + 逐槽指派，= two_channel_head 现有口径）
  C = L_gen + L_sent（λ=1）

共用：同一输入特征、同一 Trunk、同一 GenHead（与 two_channel_head **逐项同构同宽**）；
构造顺序 `(trunk, 占位 Linear(128→1), gen, sent)` + manual_seed(seed) ⇒ **trunk/gen 初值 == two_channel_head**
（占位层只为对齐对方的 RNG 序列）；核 1,688,460 全程冻结 + eval()。

只读复用 two_channel_head 的 `model.py` / `build_gen_data.py`（importlib 按路径加载成别名，
避免与本目录同名文件冲突）；**不写对方任何文件**（编码缓存落本目录 `cache/`）。

用法：uv run python experiments/struct_supervision/model.py --selfcheck
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

HERE = Path(__file__).resolve().parent
TCH_DIR = ROOT / "experiments" / "two_channel_head"
CACHE = HERE / "cache"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


tch_model = _load(TCH_DIR / "model.py", "tch_model")        # 只读
tch_build = _load(TCH_DIR / "build_gen_data.py", "tch_build")  # 只读

Spec = tch_model.Spec
Trunk = tch_model.Trunk
GenHead = tch_model.GenHead
load_base_encoder = __import__("dtseek.tasks.artifacts", fromlist=["x"]).load_base_encoder
match_sentence = tch_build.match_sentence

DATA = HERE / "data"
PAD, BOS, EOS = 0, 1, 2


def load_vocab() -> dict:
    p = DATA / "vocab.json"
    if not p.exists():
        raise SystemExit(f"缺词表：{p}（先跑 build_data.py）")
    return json.loads(p.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 整句头（A / C 用）
# ---------------------------------------------------------------------------
class SentHead(nn.Module):
    """teacher-forced 自回归字头：每步输入 [emb(前一字); h; 袋项注意力和] → GRU → 字分布。

    注意力只读**池化袋向量**（与 B/C 同一信息接口），不含句的逐字状态 ⇒ 无「复述输入」捷径。
    """

    def __init__(self, vocab: int, hidden: int = 128, emb_dim: int = 64):
        super().__init__()
        self.vocab = vocab
        self.emb_dim = emb_dim
        self.hidden_dim = hidden
        self.emb = nn.Embedding(vocab, emb_dim, padding_idx=PAD)
        self.attn_q = nn.Linear(emb_dim, hidden, bias=False)
        self.gru = nn.GRU(emb_dim + hidden + hidden, hidden, batch_first=True)
        self.out = nn.Linear(hidden, vocab)

    def attend(self, e: torch.Tensor, v_items: torch.Tensor,
               mask: torch.Tensor) -> torch.Tensor:
        """e [B,E] 或 [B,L,E] → [B,H] 或 [B,L,H]：对有效袋项的加权和。"""
        q = self.attn_q(e)
        if q.dim() == 2:
            scores = torch.einsum("bh,bmh->bm", q, v_items)
        else:
            scores = torch.einsum("blh,bmh->blm", q, v_items)
        scores = scores / math.sqrt(self.hidden_dim)
        scores = scores.masked_fill(~mask.unsqueeze(1) if scores.dim() == 3
                                    else ~mask, -1e9)
        attn = scores.softmax(-1)
        if e.dim() == 2:
            return torch.einsum("bm,bmh->bh", attn, v_items)
        return torch.einsum("blm,bmh->blh", attn, v_items)

    def forward(self, prev: torch.Tensor, h: torch.Tensor, v_items: torch.Tensor,
                mask: torch.Tensor, hidden: torch.Tensor | None = None):
        """prev [B,L]（或逐步解码时 [B]）→ logits [B,L,V] / [B,1,V]。"""
        if prev.dim() == 1:
            prev = prev.unsqueeze(1)
        e = self.emb(prev)                                    # [B,L,E]
        a = self.attend(e, v_items, mask)                     # [B,L,H]
        hh = h.unsqueeze(1).expand(-1, prev.size(1), -1)
        x = torch.cat([e, hh, a], dim=-1)
        y, hidden = self.gru(x, hidden)
        return self.out(y), hidden

    def step(self, prev: torch.Tensor, h: torch.Tensor, v_items: torch.Tensor,
             mask: torch.Tensor, hidden: torch.Tensor | None = None):
        logits, hidden = self.forward(prev, h, v_items, mask, hidden)
        return logits[:, -1, :], hidden


# ---------------------------------------------------------------------------
# 三臂模型
# ---------------------------------------------------------------------------
class StructSupModel(nn.Module):
    ARMS = ("A", "B", "C")

    def __init__(self, arm: str, seed: int = 42, spec: Spec | None = None,
                 vocab: dict | None = None):
        super().__init__()
        assert arm in self.ARMS, arm
        self.arm = arm
        self.spec = spec or Spec()
        self.vocab = vocab or load_vocab()
        self.v_size = len(self.vocab)
        self.encoder, self.ckpt = load_base_encoder(self.spec.base_path)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        # 固定构造顺序（对齐 two_channel_head 的 RNG 序列）
        torch.manual_seed(seed)
        trunk = Trunk(self.spec.hidden, self.spec.trunk_hidden)
        _ptr_placeholder = nn.Linear(self.spec.hidden, 1)      # 占位：对齐 RNG，不参与训练
        gen = GenHead(self.spec.hidden, self.spec.n_skel, self.spec.max_slots,
                      self.spec.item_dim)
        sent = SentHead(self.v_size, self.spec.hidden, 64)
        self.trunk = trunk
        self.gen = gen
        self.sent = sent
        self._placeholder = _ptr_placeholder

        # 只训该臂的头（PREREG §1）：A = trunk+sent；B = trunk+gen；C = trunk+gen+sent
        for p in self._placeholder.parameters():          # RNG 占位层永不训练
            p.requires_grad_(False)
        for p in self.gen.parameters():
            p.requires_grad_(arm in ("B", "C"))
        for p in self.sent.parameters():
            p.requires_grad_(arm in ("A", "C"))

    # ---- 冻结 / 参数 ----
    def freeze_report(self) -> dict:
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        enc_train = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        assert enc_train == 0, f"核没冻住：{enc_train} 个可训参数"
        assert not self.encoder.training, "核不在 eval() 模式"
        ph = sum(p.numel() for p in self._placeholder.parameters() if p.requires_grad)
        assert ph == 0, "RNG 占位层不应可训"
        return {"encoder_params": enc_p, "encoder_trainable": enc_train,
                "encoder_training": self.encoder.training,
                "placeholder_trainable": ph}

    def param_report(self) -> dict:
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {"arm": self.arm, "trunk": n(self.trunk), "gen_skel": n(self.gen.skel_out),
                "gen_assign": n(self.gen) - n(self.gen.skel_out),
                "sent_head": n(self.sent),
                "head_trainable": sum(p.numel() for p in self.parameters()
                                      if p.requires_grad),
                "heads_frozen": [k for k, g in (("gen", self.gen), ("sent", self.sent))
                                 if not any(p.requires_grad for p in g.parameters())]}

    # ---- 前向 / 损失 ----
    def gen_feature(self, v_sent, v_bag):
        return torch.cat([v_sent, v_bag, v_sent * v_bag, (v_sent - v_bag).abs()], dim=-1)

    def trunk_h(self, v_sent, v_bag) -> torch.Tensor:
        return self.trunk(self.gen_feature(v_sent, v_bag))

    def forward_gen(self, v_sent, v_bag, v_items, item_mask):
        h = self.trunk_h(v_sent, v_bag)
        return self.gen(h, v_items, item_mask)

    def gen_loss(self, v_sent, v_bag, v_items, item_mask, skel_y, assign_y):
        """与 two_channel_head.TwoChannelModel.gen_loss 逐字同式（selfcheck 断言等值）。"""
        h = self.trunk_h(v_sent, v_bag)
        skel_logits, a_logits = self.gen(h, v_items, item_mask)
        n = item_mask.sum(1).long()
        B, M, _ = a_logits.shape
        slots = torch.arange(M, device=a_logits.device)
        valid = (slots.view(1, -1) < n.view(-1, 1)) & item_mask
        flat = a_logits.reshape(B * M, M)
        tgt = assign_y.clamp(min=0).reshape(B * M)
        ce = F.cross_entropy(flat, tgt, reduction="none")
        vm = valid.reshape(B * M).float()
        assign = (ce * vm).sum() / vm.sum().clamp(min=1.0)
        skel = F.cross_entropy(skel_logits, skel_y)
        assert torch.isfinite(assign) and torch.isfinite(skel)
        return skel + assign, skel.detach(), assign.detach()

    def sent_loss(self, ids: torch.Tensor, v_sent, v_bag, v_items, item_mask):
        """teacher-forced 整句 CE：ids = [BOS, c1..cn, EOS, PAD...]，目标 = 右移一位。"""
        h = self.trunk_h(v_sent, v_bag)
        logits, _ = self.sent(ids[:, :-1], h, v_items, item_mask)     # [B,L,V]
        tgt = ids[:, 1:]
        ce = F.cross_entropy(logits.reshape(-1, self.v_size), tgt.reshape(-1),
                             reduction="none").reshape(tgt.shape)
        m = (tgt != PAD).float()
        loss = (ce * m).sum() / m.sum().clamp(min=1.0)
        assert torch.isfinite(loss)
        return loss

    def loss(self, v_sent, v_bag, v_items, item_mask, skel_y, assign_y, ids):
        """按臂返回 (total, parts dict)。λ=1 写死（PREREG §1）。"""
        parts: dict = {}
        if self.arm in ("B", "C"):
            tot, sk, asg = self.gen_loss(v_sent, v_bag, v_items, item_mask,
                                         skel_y, assign_y)
            parts["gen"], parts["skel"], parts["assign"] = tot, sk, asg
        if self.arm in ("A", "C"):
            sl = self.sent_loss(ids, v_sent, v_bag, v_items, item_mask)
            parts["sent"] = sl
            tot = sl if self.arm == "A" else tot + sl
        return tot, parts

    # ---- 解码（A/C 的结构出口）----
    @torch.no_grad()
    def decode(self, v_sent, v_bag, v_items, item_mask, max_len: int = 64) -> list[str]:
        h = self.trunk_h(v_sent, v_bag)
        B = h.size(0)
        dev = h.device
        prev = torch.full((B,), BOS, dtype=torch.long, device=dev)
        hidden = None
        done = torch.zeros(B, dtype=torch.bool, device=dev)
        toks: list[torch.Tensor] = []
        for _ in range(max_len):
            logits, hidden = self.sent.step(prev, h, v_items, item_mask, hidden)
            logits[:, PAD] = -1e9
            logits[:, BOS] = -1e9
            nxt = logits.argmax(-1)
            nxt = torch.where(done, torch.full_like(nxt, PAD), nxt)
            toks.append(nxt)
            done = done | (nxt == EOS)
            prev = torch.where(done, torch.full_like(prev, PAD), nxt)
            if bool(done.all()):
                break
        seq = torch.stack(toks, 1).cpu().tolist()
        rev = {i: c for c, i in self.vocab.items()}
        out = []
        for row in seq:
            buf = []
            for t in row:
                if t == EOS:
                    break
                if t in (PAD, BOS):
                    continue
                buf.append(rev.get(t, "�"))
            out.append("".join(buf))
        return out

    @torch.no_grad()
    def tf_char_stats(self, ids: torch.Tensor, v_sent, v_bag, v_items, item_mask):
        """teacher-forced 字准确率（不含 PAD），诊断 A/C 的监督是否学到句面分布。"""
        h = self.trunk_h(v_sent, v_bag)
        logits, _ = self.sent(ids[:, :-1], h, v_items, item_mask)
        pred = logits.argmax(-1)
        tgt = ids[:, 1:]
        m = tgt != PAD
        ok = ((pred == tgt) & m).sum().item()
        return ok, int(m.sum().item())


# ---------------------------------------------------------------------------
# 结构恢复：句面 → (骨架, 槽位指派)（fail-closed）
# ---------------------------------------------------------------------------
def parse_sentence(sent: str, bag: list[str]) -> tuple[int, list[int]] | None:
    """解析句面并把各槽文本映射回洗牌袋下标；任一环节失败 ⇒ None（该行全错）。"""
    m = match_sentence(sent)
    if m is None:
        return None
    sid, gaps, _spans = m
    remaining = list(range(len(bag)))
    assign: list[int] = []
    for g in gaps:
        hit = None
        for k in remaining:
            if bag[k] == g:
                hit = k
                break
        if hit is None:
            return None
        remaining.remove(hit)
        assign.append(hit)
    if len(assign) != len(bag):
        return None
    return sid, assign


# ---------------------------------------------------------------------------
# 编码（与 two_channel_head 同口径；缓存落本目录）
# ---------------------------------------------------------------------------
def _fp(rows: list[dict]) -> str:
    return hashlib.md5("".join(r["sent"] + str(r["bag_span"]) for r in rows)
                       .encode()).hexdigest()[:12]


def encode_rows(model: StructSupModel, rows: list[dict], spec: Spec,
                device: str, batch: int = 256) -> dict:
    """与 two_channel_head.model.encode_gen 同一段逻辑（逐字复刻），缓存写本目录。"""
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"gen_{_fp(rows)}_L{spec.max_len_sent}_M{spec.max_slots}.pt"
    if path.exists():
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob["n"] == len(rows):
            print(f"[cache] 命中 {path.name}（n={blob['n']}）", flush=True)
            return blob

    v_sent, v_items, mask, skel, assign = [], [], [], [], []
    model.encoder.eval()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        ids, msk = [], []
        for r in chunk:
            e = tok.encode(r["sent"], max_length=spec.max_len_sent, padding=True)
            assert sum(e["attention_mask"]) == len(r["sent"]), "1 字符 1 token 前提被破坏"
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
            n = r["n_slots"]
            assert n <= spec.max_slots, f"槽位数 {n} > {spec.max_slots}"
            for s, (a, b) in enumerate(r["bag_span"]):
                assert r["sent"][a:b] == r["bag"][s], "span 与 bag 文本不一致"
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
    print(f"[cache] 写入 {path.name}（n={len(rows)}）", flush=True)
    return blob


def sentence_ids(rows: list[dict], vocab: dict, max_len: int = 66) -> torch.Tensor:
    """句子 → [BOS]+字+[EOS]+PAD 的 id 矩阵（词表跑前固定，OOV 即断言）。"""
    out = []
    for r in rows:
        ids = [BOS] + [vocab[c] for c in r["sent"]] + [EOS]
        assert len(ids) <= max_len, f"句长越界：{r['sent']}"
        out.append(ids + [PAD] * (max_len - len(ids)))
    return torch.tensor(out, dtype=torch.long)


def v_bag_of(v_items: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (v_items * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def selfcheck(device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = Spec()
    vocab = load_vocab()
    rep: dict = {"device": device, "vocab_size": len(vocab)}

    # 1) 三臂冻结实况 + 可训参数逐项
    arms = {a: StructSupModel(a, seed=42, spec=spec, vocab=vocab).to(device)
            for a in "ABC"}
    rep["freeze"] = arms["B"].freeze_report()
    rep["params"] = {a: arms[a].param_report() for a in "ABC"}
    assert rep["freeze"]["encoder_trainable"] == 0
    assert rep["params"]["A"]["head_trainable"] == rep["params"]["A"]["trunk"] + rep["params"]["A"]["sent_head"]
    assert rep["params"]["B"]["head_trainable"] == rep["params"]["B"]["trunk"] + \
        rep["params"]["B"]["gen_skel"] + rep["params"]["B"]["gen_assign"]
    assert rep["params"]["C"]["head_trainable"] == rep["params"]["C"]["trunk"] + \
        rep["params"]["C"]["gen_skel"] + rep["params"]["C"]["gen_assign"] + rep["params"]["C"]["sent_head"]
    assert rep["params"]["A"]["heads_frozen"] == ["gen"]
    assert rep["params"]["B"]["heads_frozen"] == ["sent"]

    # 2) 同口径校验：trunk/gen 初值 == two_channel_head 同 seed 初值（CPU 上逐位比）
    cpuB = StructSupModel("B", seed=42, spec=spec, vocab=vocab)   # 同 seed ⇒ 同初值
    tchB = tch_model.TwoChannelModel("B", seed=42, spec=spec)
    same_trunk = all(torch.equal(a, b) for a, b in
                     zip(cpuB.trunk.state_dict().values(),
                         tchB.trunk_gen.state_dict().values()))
    same_gen = all(torch.equal(a, b) for a, b in
                   zip(cpuB.gen.state_dict().values(),
                       tchB.gen.state_dict().values()))
    rep["init_identical_to_two_channel"] = {"trunk": same_trunk, "gen": same_gen}
    assert same_trunk and same_gen, rep["init_identical_to_two_channel"]

    # 3) gen 前向/损失与 two_channel_head 等值（随机输入）
    g = {"v_sent": torch.randn(16, spec.hidden),
         "v_items": torch.randn(16, spec.max_slots, spec.hidden),
         "item_mask": torch.ones(16, spec.max_slots, dtype=torch.bool),
         "skel": torch.randint(0, spec.n_skel, (16,)),
         "assign": torch.randint(0, spec.max_slots, (16, spec.max_slots))}
    g["v_bag"] = v_bag_of(g["v_items"], g["item_mask"])
    ours = cpuB.gen_loss(g["v_sent"], g["v_bag"], g["v_items"], g["item_mask"],
                         g["skel"], g["assign"])
    theirs = tchB.gen_loss(g["v_sent"], g["v_bag"], g["v_items"], g["item_mask"],
                           g["skel"], g["assign"])
    rep["gen_loss_equal"] = [round(float(x), 8) for x in ours[:2]] == \
        [round(float(x), 8) for x in theirs[:2]]
    assert rep["gen_loss_equal"], (float(ours[0]), float(theirs[0]))

    # 4) 解析往返：train 8000 行金句 → 结构还原 == 标签
    train = [json.loads(l) for l in open(TCH_DIR / "data" / "train.jsonl",
                                         encoding="utf-8")]
    ok = bad = 0
    for r in train:
        m = match_sentence(r["sent"])
        if m is None or m[0] != r["skel_id"]:
            bad += 1
            continue
        gaps, spans = m[1], m[2]
        if [r["sent"][a:b] for a, b in spans] != gaps:
            bad += 1
            continue
        # 槽位 → 袋下标（按文本消费）
        rem = list(range(len(r["bag"])))
        got = []
        for gp in gaps:
            hit = next((k for k in rem if r["bag"][k] == gp), None)
            if hit is None:
                break
            rem.remove(hit)
            got.append(hit)
        if got == r["assign"]:
            ok += 1
        else:
            bad += 1
    rep["parse_roundtrip"] = {"ok": ok, "bad": bad}
    assert bad == 0, f"解析往返失败 {bad} 行"

    # 5) 对抗集断言（重建 build 的关键不变量）
    tr_cnt = {int(k): v for k, v in
              __import__("collections").Counter(r["skel_id"] for r in train).items()}
    tr_sents = {r["sent"] for r in train}
    te_sents = {json.loads(l)["sent"] for l in open(
        TCH_DIR / "data" / "test.jsonl", encoding="utf-8")}
    adv: dict = {}
    for name in ("adv1", "adv2"):
        p = DATA / f"{name}.jsonl"
        rows = [json.loads(l) for l in open(p, encoding="utf-8")]
        adv[name] = {"n": len(rows),
                     "overlap": len({r["sent"] for r in rows} & (tr_sents | te_sents)),
                     "ids": sorted({r["skel_id"] for r in rows})}
        assert adv[name]["overlap"] == 0, f"{name} 与 train/test 重叠"
    assert all(tr_cnt.get(i, 0) == 0 for i in adv["adv1"]["ids"]), "adv1 含已训骨架"
    from build_data import row_signature  # 槽型签名口径单一来源（句序）
    train_sig: dict[int, set] = {}
    for r in train:
        train_sig.setdefault(r["skel_id"], set()).add(row_signature(r))
    adv2_novel = 0
    for l in open(DATA / "adv2.jsonl", encoding="utf-8"):
        r = json.loads(l)
        assert r.get("sig"), "adv2 缺签名"
        assert tuple(r["sig"]) not in train_sig[r["skel_id"]], \
            f"adv2 签名在 train 中出现过：{r['sent']}"
        assert tuple(r["sig"]) == row_signature(r), "签名与槽文本不一致"
        adv2_novel += 1
    rep["adv2_signatures_novel"] = adv2_novel
    rep["adv"] = adv

    # 6) 词表覆盖（0 OOV）
    oov = sum(1 for l in open(DATA / "adv2.jsonl", encoding="utf-8")
              for c in json.loads(l)["sent"] if c not in vocab)
    oov += sum(1 for l in open(DATA / "adv1.jsonl", encoding="utf-8")
               for c in json.loads(l)["sent"] if c not in vocab)
    rep["vocab_oov"] = oov
    assert oov == 0

    # 7) 句序不变量 + 袋序不变量
    ids = sentence_ids(train[:8], vocab)
    assert ids[0, 0].item() == BOS
    rep["sent_ids_shape"] = list(ids.shape)

    # 8) 编码同口径：本目录编码器 vs two_channel_head 的 train 缓存（前 64 行逐张量比）
    sub = train[:64]
    import contextlib, io
    _buf = io.StringIO()
    with contextlib.redirect_stdout(_buf):        # 别让 [cache] 日志污染 selfcheck 的 JSON
        ours_blob = encode_rows(arms["B"], sub, spec, device)
    rep["encode_log"] = _buf.getvalue().strip()
    fp_all = hashlib.md5("".join(r["sent"] + str(r["bag_span"]) for r in train)
                         .encode()).hexdigest()[:12]
    tch_path = TCH_DIR / "cache" / f"gen_{fp_all}_L{spec.max_len_sent}_M{spec.max_slots}.pt"
    rep["tch_cache"] = str(tch_path.name)
    if tch_path.exists():
        ref = torch.load(tch_path, map_location="cpu", weights_only=True)
        diffs = {
            "v_sent": float((ours_blob["v_sent"] - ref["v_sent"][:64]).abs().max()),
            "v_items": float((ours_blob["v_items"] - ref["v_items"][:64]).abs().max()),
            "item_mask_equal": bool(torch.equal(ours_blob["item_mask"],
                                                ref["item_mask"][:64])),
            "skel_equal": bool(torch.equal(ours_blob["skel"], ref["skel"][:64])),
            "assign_equal": bool(torch.equal(ours_blob["assign"], ref["assign"][:64])),
        }
        rep["encode_same_as_two_channel"] = diffs
        # 批大小不同可能改变浮点归约顺序 ⇒ 标签/掩码要求逐位相等，向量用 1e-4 容差
        assert diffs["v_sent"] < 1e-4 and diffs["v_items"] < 1e-4 \
            and diffs["item_mask_equal"] and diffs["skel_equal"] and diffs["assign_equal"], diffs
    else:
        rep["encode_same_as_two_channel"] = "缓存未找到（two_channel_head 未跑过 train 编码）"
    rep["encode_rows_ok"] = bool(ours_blob["n"] == 64)
    assert rep["encode_rows_ok"]
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.selfcheck:
        print(json.dumps(selfcheck(a.device), ensure_ascii=False, indent=2,
                         default=str))
    else:
        ap.error("用 --selfcheck")
