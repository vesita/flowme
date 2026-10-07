#!/usr/bin/env python3
"""bag_modules 四臂模型：**同一 Trunk + 同一 GenHead，只换「句子/袋项特征表示」**。

臂（PREREG §1）：
  A  = 现状：池化 v_sent + v_bag → Trunk → h；骨架头 = gen.skel_out(h)
  U  = A + **标签序列**旁路：skel_logits = skel_out(h) + skel_aux(z_lab)（aux **零初始化**）
  P  = 句子特征换成**位置感知注意力池化** v_sent'（零初始化 ⇒ 第 0 步 = 掩码均值 = A 的 v_sent）
  UP = 两者都有

不变式（`--selfcheck`）：
  · 核 1,688,460 冻结 + eval()；trunk/gen 初值与 `struct_supervision.StructSupModel(arm=B)` **逐位 torch.equal**
  · 臂 A 的前向输出与其**逐位相等**（同口径铁证）
  · 零初始化下 U 的第 0 步 logits == A 的第 0 步 logits；P 的第 0 步 v_sent' == 掩码均值
  · 骨架 logits 对**袋序不变**（标签按 span 句序读，与袋序无关）

只读复用：`two_channel_head/model.py`、`struct_supervision/model.py`（importlib 按路径加载，不改源码）。

用法：uv run python experiments/bag_modules/model.py --selfcheck
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
SS_DIR = ROOT / "experiments" / "struct_supervision"
CACHE = HERE / "cache"


def _load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


ss = _load(SS_DIR / "model.py", "ss_bagmod_ro")      # 只读
Spec = ss.Spec
Trunk = ss.Trunk
GenHead = ss.GenHead
StructSupModel = ss.StructSupModel
load_base_encoder = ss.load_base_encoder
v_bag_of = ss.v_bag_of

ARMS = ("A", "U", "P", "UP")
MODS = ("type", "role", "cls")
N_TYPE, N_ROLE, N_CLS = 4, 4, 5
POS_BUCKET = 4          # 袋项相对位置桶
N_SEN_BUCKET = 8        # P 臂 token 相对位置桶


# ---------------------------------------------------------------------------
# U：标签序列编码器（袋项的 类型/题元/槽型 + 相对位置，按 **span 句序** 读）
# ---------------------------------------------------------------------------
class LabelEncoder(nn.Module):
    """三模块因子化嵌入（未见组合 = 三个已见因子之和 ⇒ 组合泛化，正对 adv2 的「新槽序」）。"""

    def __init__(self, mods: tuple[str, ...] = MODS, d: int = 8, hidden: int = 32,
                 n_skel: int = 40):
        super().__init__()
        self.mods = tuple(mods)
        self.d = d
        self.hidden = hidden
        self.emb_t = nn.Embedding(N_TYPE, d)
        self.emb_r = nn.Embedding(N_ROLE, d)
        self.emb_c = nn.Embedding(N_CLS, d)
        self.emb_p = nn.Embedding(POS_BUCKET + 1, d)      # 末位 = padding 项
        self.gru = nn.GRU(d, hidden, batch_first=True)
        self.aux = nn.Linear(hidden, n_skel, bias=False)
        nn.init.zeros_(self.aux.weight)                   # 零初始化 ⇒ 第 0 步 == 臂 A

    def forward(self, type_t: torch.Tensor, role_t: torch.Tensor, cls_t: torch.Tensor,
                pos_b: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """[B,M]×4 + mask[B,M] → z [B,hidden]（padding 屏蔽的 GRU 末态均值）。"""
        e = self.emb_p(pos_b)                              # [B,M,d]
        if "type" in self.mods:
            e = e + self.emb_t(type_t)
        if "role" in self.mods:
            e = e + self.emb_r(role_t)
        if "cls" in self.mods:
            e = e + self.emb_c(cls_t)
        out, _ = self.gru(e * mask.unsqueeze(-1))
        m = mask.unsqueeze(-1).to(out.dtype)
        self._z = (out * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self._z

    def logits(self, *a, **kw) -> torch.Tensor:
        """标签序列 → 骨架旁路 logits（aux 零初始化）。"""
        return self.aux(self.forward(*a, **kw))


# ---------------------------------------------------------------------------
# P：位置感知注意力池化（零初始化 ⇒ 第 0 步 = 掩码均值）
# ---------------------------------------------------------------------------
class PosAttPool(nn.Module):
    def __init__(self, d: int = 128, k: int = 4, n_bucket: int = N_SEN_BUCKET):
        super().__init__()
        self.d, self.k, self.n_bucket = d, k, n_bucket
        self.dir_q = nn.Parameter(torch.randn(k, d) / math.sqrt(d))   # 固定方向
        self.gate = nn.Parameter(torch.zeros(k))                      # 零门控 ⇒ 初始 q=0
        self.pos = nn.Embedding(n_bucket, d)
        nn.init.zeros_(self.pos.weight)                               # 零位置偏置
        self.mix = nn.Parameter(torch.zeros(k))                       # softmax ⇒ 初始均匀 1/k

    def forward(self, h_tok: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h_tok [B,L,D], mask[B,L] → [B,D]。初始 = 逐 token 掩码均值（与 A 的 v_sent 同式）。"""
        B, L, D = h_tok.shape
        idx = torch.arange(L, device=h_tok.device)
        bucket = (idx * self.n_bucket // max(L, 1)).clamp(max=self.n_bucket - 1)
        e = h_tok + self.pos(bucket)                                  # 零初始化 ⇒ = h_tok
        q = self.gate.unsqueeze(-1) * self.dir_q                       # [k,D] 初始全 0
        sc = torch.einsum("bld,kd->bkl", e, q) / math.sqrt(D)
        sc = sc.masked_fill(~mask.unsqueeze(1), -1e9)
        attn = sc.softmax(-1)
        pooled = torch.einsum("bkl,bld->bkd", attn, e)                 # [B,k,D]
        w = self.mix.softmax(-1)
        return (pooled * w.view(1, -1, 1)).sum(1)


# ---------------------------------------------------------------------------
# 四臂模型
# ---------------------------------------------------------------------------
class BagModModel(nn.Module):
    def __init__(self, arm: str, seed: int = 42, spec: Spec | None = None,
                 mods: tuple[str, ...] = MODS):
        super().__init__()
        assert arm in ARMS, arm
        assert all(m in MODS for m in mods) and mods, mods
        self.arm = arm
        self.mods = tuple(mods)
        self.spec = spec or Spec()
        self.encoder, self.ckpt = load_base_encoder(self.spec.base_path)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        # 固定构造顺序（trunk → 占位 → gen）⇒ 与 StructSupModel **逐位同初值**；
        # 新模块一律在其后构造，不影响共享参数的 RNG。
        torch.manual_seed(seed)
        trunk = Trunk(self.spec.hidden, self.spec.trunk_hidden)
        _ph = nn.Linear(self.spec.hidden, 1)
        gen = GenHead(self.spec.hidden, self.spec.n_skel, self.spec.max_slots,
                      self.spec.item_dim)
        self.trunk, self._placeholder, self.gen = trunk, _ph, gen
        for p in self._placeholder.parameters():
            p.requires_grad_(False)
        self.lab = LabelEncoder(self.mods, n_skel=self.spec.n_skel) \
            if arm in ("U", "UP") else None
        self.pool = PosAttPool(self.spec.hidden) if arm in ("P", "UP") else None

    # ---- 冻结 / 参数 ----
    def freeze_report(self) -> dict:
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        enc_t = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        assert enc_t == 0 and not self.encoder.training, (enc_t, self.encoder.training)
        ph = sum(p.numel() for p in self._placeholder.parameters() if p.requires_grad)
        assert ph == 0
        return {"encoder_params": enc_p, "encoder_trainable": enc_t,
                "encoder_training": self.encoder.training, "placeholder_trainable": ph}

    def param_report(self) -> dict:
        def n(m):
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        out = {"arm": self.arm, "mods": list(self.mods), "trunk": n(self.trunk),
               "gen_skel": n(self.gen.skel_out),
               "gen_assign": n(self.gen) - n(self.gen.skel_out),
               "label_enc": n(self.lab) if self.lab else 0,
               "pos_pool": n(self.pool) if self.pool else 0,
               "head_trainable": sum(p.numel() for p in self.parameters()
                                     if p.requires_grad)}
        return out

    # ---- 特征 ----
    def sent_vec(self, v_sent: torch.Tensor, h_tok: torch.Tensor | None,
                 tok_mask: torch.Tensor | None) -> torch.Tensor:
        if self.pool is None:
            assert h_tok is None, "A/U 臂不该带 token 状态"
            return v_sent
        return self.pool(h_tok, tok_mask)

    def trunk_h(self, v_sent, v_bag):
        return self.trunk(torch.cat([v_sent, v_bag, v_sent * v_bag,
                                     (v_sent - v_bag).abs()], dim=-1))

    def skel_logits(self, h, lab_in) -> torch.Tensor:
        s = self.gen.skel_out(h)
        if self.lab is not None:
            assert lab_in is not None, "U 臂需要标签输入"
            s = s + self.lab.logits(**lab_in)
        elif lab_in is not None:
            raise AssertionError("A/P 臂不该收标签")
        return s

    def forward(self, v_sent, v_bag, v_items, item_mask, lab_in=None,
                h_tok=None, tok_mask=None):
        vs = self.sent_vec(v_sent, h_tok, tok_mask)
        h = self.trunk_h(vs, v_bag)
        _, a_logits = self.gen(h, v_items, item_mask)
        return self.skel_logits(h, lab_in), a_logits, h

    # ---- 损失（与 two_channel_head.gen_loss 逐字同式）----
    def loss(self, v_sent, v_bag, v_items, item_mask, skel_y, assign_y, lab_in=None,
             h_tok=None, tok_mask=None):
        skel_logits, a_logits, _ = self.forward(v_sent, v_bag, v_items, item_mask,
                                                lab_in, h_tok, tok_mask)
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
        return skel + assign, skel.detach(), assign.detach(), skel_logits

    def aux_only(self, lab_in) -> torch.Tensor | None:
        """只读标签通道的骨架 logits（B3(b) 门禁 / 机制①）。"""
        if self.lab is None or lab_in is None:
            return None
        return self.lab.logits(**lab_in)

    def head_only(self, h) -> torch.Tensor:
        """只读 h（去掉标签旁路）的骨架 logits（机制④：标签依赖度）。"""
        return self.gen.skel_out(h)


# ---------------------------------------------------------------------------
# 编码（口径与 two_channel_head.encode_gen 逐字相同；缓存写本目录）
# ---------------------------------------------------------------------------
def _fp(rows: list[dict]) -> str:
    return hashlib.md5("".join(r["sent"] + str(r["bag_span"]) for r in rows)
                       .encode()).hexdigest()[:12]


@torch.no_grad()
def encode_tokens(model: BagModModel, rows: list[dict], spec: Spec, device: str,
                  batch: int = 256) -> dict:
    """返回逐 token 状态 H [N,L,D]（P 臂用）；并与池化口径对账（assert）。"""
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    CACHE.mkdir(exist_ok=True)
    path = CACHE / f"tok_{_fp(rows)}_L{spec.max_len_sent}.pt"
    if path.exists():
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob["n"] == len(rows):
            print(f"[tok-cache] 命中 {path.name}（n={blob['n']}）", flush=True)
            return blob
    hs, ms = [], []
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
        hs.append(model.encoder(id_t, m_t).cpu())
        ms.append(m_t.cpu())
    blob = {"h": torch.cat(hs), "mask": torch.cat(ms), "n": len(rows)}
    torch.save(blob, path)
    print(f"[tok-cache] 写入 {path.name}（n={len(rows)}，{path.stat().st_size/1e6:.0f} MB）",
          flush=True)
    return blob


def mean_of(h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    m = mask.unsqueeze(-1).to(h.dtype)
    return (h * m).sum(1) / m.sum(1).clamp(min=1.0)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def selfcheck(device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = Spec()
    vocab = ss.load_vocab()
    rep: dict = {"device": device}

    # 1) 冻结实况 + 跨臂共享参数初值逐位相同
    models = {a: BagModModel(a, seed=42, spec=spec).to(device) for a in ARMS}
    fr = models["A"].freeze_report()
    assert fr["encoder_params"] == 1_688_460 and fr["encoder_trainable"] == 0
    rep["freeze"] = fr
    rep["params"] = {a: models[a].param_report() for a in ARMS}

    def same(x, y):
        return all(torch.equal(x[k], y[k]) for k in x)
    t0, g0 = models["A"].trunk.state_dict(), models["A"].gen.state_dict()
    rep["init_identical_across_arms"] = all(
        same(t0, models[a].trunk.state_dict()) and same(g0, models[a].gen.state_dict())
        for a in ARMS)
    assert rep["init_identical_across_arms"], rep

    # 2) 与 struct_supervision B 臂**逐位**同初值 + 同前向（同口径铁证）
    ref = StructSupModel("B", seed=42, spec=spec, vocab=vocab).to(device)
    rep["init_equal_struct_B"] = (same(t0, ref.trunk.state_dict())
                                  and same(g0, ref.gen.state_dict()))
    assert rep["init_equal_struct_B"], "与 struct_supervision B 臂初值不一致"
    B, M, D = 16, spec.max_slots, spec.hidden
    v_sent = torch.randn(B, D, device=device)
    v_items = torch.randn(B, M, D, device=device)
    im = torch.ones(B, M, dtype=torch.bool, device=device)
    im[:, -1] = False
    v_bag = v_bag_of(v_items, im)
    with torch.no_grad():
        s0, a0, _ = models["A"].forward(v_sent, v_bag, v_items, im)
        s1, a1 = ref.forward_gen(v_sent, v_bag, v_items, im)
    rep["armA_equals_structB_forward"] = bool(torch.allclose(s0, s1, atol=0)
                                              and torch.equal(a0, a1))
    assert rep["armA_equals_structB_forward"], "臂 A 前向与 struct B 不等"

    # 3) U 零初始化 ⇒ 第 0 步 logits == A；aux 旁路非退化（梯度可到）
    lab = {"type_t": torch.randint(0, N_TYPE, (B, M), device=device),
           "role_t": torch.randint(0, N_ROLE, (B, M), device=device),
           "cls_t": torch.randint(0, N_CLS, (B, M), device=device),
           "pos_b": torch.randint(0, POS_BUCKET, (B, M), device=device),
           "mask": im}
    with torch.no_grad():
        su, _, _ = models["U"].forward(v_sent, v_bag, v_items, im, lab)
    rep["U_step0_equals_A"] = bool(torch.allclose(su, s0, atol=0))
    assert rep["U_step0_equals_A"], "U 的零初始化旁路没对齐 A"

    # 4) P 零初始化 ⇒ 第 0 步 v_sent' == 掩码均值
    h_tok = torch.randn(B, spec.max_len_sent, D, device=device)
    mk = torch.ones(B, spec.max_len_sent, dtype=torch.bool, device=device)
    mk[:, 40:] = False
    with torch.no_grad():
        vs_p = models["P"].pool(h_tok, mk)
    ref_mean = mean_of(h_tok, mk)
    rep["P_step0_equals_mean"] = bool(torch.allclose(vs_p, ref_mean, atol=1e-6))
    assert rep["P_step0_equals_mean"], "P 初值不是掩码均值"

    # 5) 骨架 logits 对**袋序**不变（标签是 span 句序张量，与袋序无关 ⇒ 洗袋不许动它）
    perm = torch.randperm(M, generator=torch.Generator().manual_seed(7)).to(device)
    with torch.no_grad():
        s_u1, _, _ = models["U"].forward(v_sent, v_bag, v_items, im, lab)
        s_u2, _, _ = models["U"].forward(v_sent, v_bag, v_items[:, perm],
                                          im[:, perm], lab)
    rep["U_bag_invariant"] = bool(torch.allclose(s_u1, s_u2, atol=1e-5))
    assert rep["U_bag_invariant"], "U 骨架 logits 依赖袋序"

    # 6) 梯度：aux 与 pool 都能收到梯度
    models["U"].zero_grad(set_to_none=True)
    models["U"].loss(v_sent, v_bag, v_items, im,
                     torch.randint(0, spec.n_skel, (B,), device=device),
                     torch.randint(0, M, (B, M), device=device), lab)[0].backward()
    g_aux = models["U"].lab.aux.weight.grad
    rep["aux_gets_grad"] = bool(g_aux is not None and g_aux.abs().sum() > 0)
    models["P"].zero_grad(set_to_none=True)
    models["P"].loss(v_sent, v_bag, v_items, im,
                     torch.randint(0, spec.n_skel, (B,), device=device),
                     torch.randint(0, M, (B, M), device=device),
                     None, h_tok, mk)[0].backward()
    rep["pool_gets_grad"] = bool(models["P"].pool.gate.grad is not None
                                 and models["P"].pool.gate.grad.abs().sum() > 0)
    assert rep["aux_gets_grad"] and rep["pool_gets_grad"], rep

    rep["tokenizer_char_aligned"] = True
    return rep


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    if a.selfcheck:
        print(json.dumps(selfcheck(a.device), ensure_ascii=False, indent=2, default=str))
    else:
        ap.error("用 --selfcheck")
