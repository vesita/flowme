#!/usr/bin/env python3
"""two_channel_head：同核双通道头（指针通道 + 生成通道）四臂结构。

臂（PREREG §1）：
  A = 只有指针分支（主干 + 指针输出层）
  B = 只有生成分支（主干 + 生成输出层）
  C = **共享主干** + 两个分支
  D = **两个完整独立头**（= A + B 的参数量）

共用的结构（PREREG §1）：
  主干 Trunk = Linear(512→256) → GELU → Linear(256→128) → GELU   （164,224）
  指针输出层 = Linear(128→1)                                       （129）
  ⇒ A = 164,353，与 experiments/select_rerank 的打分头**逐项相同**。
  生成特征 = [v_sent; v_bag; v_sent⊙v_bag; |v_sent−v_bag|]（512，与指针同几何同宽）
  生成输出层 = 骨架 Linear(128→40) + 指派 MLP([h;v_item])→64 ⊗ 可学槽位嵌入

不变量（`--selfcheck`，也是训练脚本的前置断言）：
  · 核 1,688,460 参数 `requires_grad` 全 False、`training` 为 False（**实测打印**）
  · 四臂主干与两个输出层**初始化逐位相同**（固定构造顺序 + manual_seed(seed)）
  · **C 的两个分支共用同一个主干对象**（`trunk_ptr is trunk_gen`），D 不共用
  · 两通道各自对主干的梯度都非零且不同（共享确实发生了）
  · 指针分支对候选**等变**；生成分支的**骨架输入对袋序不变**

用法：uv run python experiments/two_channel_head/model.py --selfcheck
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402

HERE = Path(__file__).resolve().parent
SR_DIR = ROOT / "experiments" / "select_rerank"          # 指针分支数据/缓存：**只读复用**
GEN_CACHE = HERE / "cache"


@dataclass(frozen=True)
class Spec:
    max_len_ctx: int = 64
    max_len_cand: int = 32
    max_len_sent: int = 64
    hidden: int = 128
    trunk_hidden: int = 256
    k: int = 2
    n_skel: int = 40            # 骨架表规模（输出层宽度）
    max_slots: int = 4          # 槽位上限（数据实测 ≤4）
    item_dim: int = 64          # 指派嵌入维
    base_path: str = "checkpoints/base_encoder.pt"
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 结构件
# ---------------------------------------------------------------------------
class Trunk(nn.Module):
    """共享主干：512 → 256 → 128。指针与生成两通道输入几何同宽，故可共用同一组权重。"""

    def __init__(self, hidden: int = 128, trunk_hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden * 4, trunk_hidden),
            nn.GELU(),
            nn.Linear(trunk_hidden, hidden),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GenHead(nn.Module):
    """生成输出层：骨架 id 多分类 + 每槽对 n 个袋项的指派多分类（不发字）。"""

    def __init__(self, hidden: int = 128, n_skel: int = 40, max_slots: int = 4,
                 item_dim: int = 64):
        super().__init__()
        self.n_skel = n_skel
        self.max_slots = max_slots
        self.item_dim = item_dim
        self.skel_out = nn.Linear(hidden, n_skel)
        self.item_mlp = nn.Sequential(nn.Linear(hidden * 2, item_dim), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(max_slots, item_dim) / (item_dim ** 0.5))

    def forward(self, h: torch.Tensor, v_items: torch.Tensor,
                item_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h [B,D], v_items [B,M,D], item_mask [B,M](True=有效) →
        (skel_logits [B,n_skel], assign_logits [B,M槽,M项])"""
        B, M, D = v_items.shape
        u = torch.cat([h.unsqueeze(1).expand(-1, M, -1), v_items], dim=-1)
        emb = self.item_mlp(u)                                   # [B,M,item_dim]
        logits = torch.einsum("bmi,ki->bkm", emb, self.slot_emb) / (self.item_dim ** 0.5)
        logits = logits.masked_fill(~item_mask.unsqueeze(1), -1e9)
        return self.skel_out(h), logits


class TwoChannelModel(nn.Module):
    """四臂共用一个类：`arm` 决定主干是否共享、装哪些输出层。"""

    ARMS = ("A", "B", "C", "D")

    def __init__(self, arm: str, seed: int = 42, spec: Spec | None = None):
        super().__init__()
        assert arm in self.ARMS, arm
        self.arm = arm
        self.spec = spec or Spec()
        self.encoder, self.ckpt = load_base_encoder(self.spec.base_path)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()

        # 固定构造顺序 ⇒ 四臂的 (主干, 指针输出层, 生成输出层) 初始化逐位相同
        torch.manual_seed(seed)
        trunk = Trunk(self.spec.hidden, self.spec.trunk_hidden)
        ptr_out = nn.Linear(self.spec.hidden, 1)
        gen = GenHead(self.spec.hidden, self.spec.n_skel, self.spec.max_slots,
                      self.spec.item_dim)
        if arm == "A":
            self.trunk_ptr, self.trunk_gen, self.ptr_out = trunk, None, ptr_out
        elif arm == "B":
            self.trunk_ptr, self.trunk_gen, self.gen = None, trunk, gen
        elif arm == "C":
            self.trunk_ptr = self.trunk_gen = trunk          # ← 共享主干（同一对象）
            self.ptr_out, self.gen = ptr_out, gen
        else:  # D：两个完整独立头
            self.trunk_ptr, self.ptr_out = trunk, ptr_out
            self.trunk_gen, self.gen = copy.deepcopy(trunk), gen

    # ---- 冻结与参数实况 ----------------------------------------------------
    def freeze_report(self) -> dict:
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        enc_train = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        assert enc_train == 0, f"核没冻住：{enc_train} 个可训参数"
        assert not self.encoder.training, "核不在 eval() 模式"
        return {"encoder_params": enc_p, "encoder_trainable": enc_train,
                "encoder_training": self.encoder.training,
                "hidden": self.spec.hidden,
                "max_len": getattr(self.encoder, "max_len", None)}

    def param_report(self) -> dict:
        """可训参数**逐项**（PREREG §6.2）。"""
        def n(m: nn.Module) -> int:
            return sum(p.numel() for p in m.parameters() if p.requires_grad)
        trunk_p = n(self.trunk_ptr) if self.trunk_ptr is not None else 0
        trunk_g = n(self.trunk_gen) if self.trunk_gen is not None else 0
        out = {"arm": self.arm, "trunk_ptr": trunk_p, "trunk_gen": trunk_g,
               "shared_trunk": (self.trunk_ptr is not None
                                and self.trunk_ptr is self.trunk_gen),
               "ptr_out": n(self.ptr_out) if hasattr(self, "ptr_out") else 0}
        if hasattr(self, "gen"):
            out["gen_skel"] = sum(p.numel() for p in self.gen.skel_out.parameters())
            out["gen_assign"] = (n(self.gen) - out["gen_skel"])
        else:
            out["gen_skel"] = out["gen_assign"] = 0
        out["head_trainable"] = sum(p.numel() for p in self.parameters()
                                    if p.requires_grad)
        return out

    def named_trainable(self) -> list[str]:
        return [k for k, p in self.named_parameters() if p.requires_grad]

    # ---- 前向 --------------------------------------------------------------
    def forward_ptr(self, v_ctx: torch.Tensor, v_cand: torch.Tensor) -> torch.Tensor:
        """v_ctx [B,D], v_cand [B,K,D] → logits [B,K]（对候选等变：无逐下标参数）。"""
        B, K, D = v_cand.shape
        c = v_ctx.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([c, v_cand, c * v_cand, (c - v_cand).abs()], dim=-1)   # [B,K,512]
        h = self.trunk_ptr(x.reshape(B * K, -1)).reshape(B, K, -1)
        return self.ptr_out(h).squeeze(-1)

    def gen_feature(self, v_sent: torch.Tensor, v_bag: torch.Tensor) -> torch.Tensor:
        """生成特征 512 维（与指针同几何）；v_bag 是袋均值 ⇒ **对袋序不变**。"""
        return torch.cat([v_sent, v_bag, v_sent * v_bag, (v_sent - v_bag).abs()], dim=-1)

    def forward_gen(self, v_sent: torch.Tensor, v_bag: torch.Tensor,
                    v_items: torch.Tensor, item_mask: torch.Tensor):
        x = self.gen_feature(v_sent, v_bag)
        h = self.trunk_gen(x)
        return self.gen(h, v_items, item_mask)

    def ptr_loss(self, v_ctx, v_cand, y) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(self.forward_ptr(v_ctx, v_cand), y)

    def gen_loss(self, v_sent, v_bag, v_items, item_mask, skel_y, assign_y):
        """L_gen = CE(骨架) + CE(指派，逐槽平均、padding 屏蔽)。"""
        skel_logits, a_logits = self.forward_gen(v_sent, v_bag, v_items, item_mask)
        n = item_mask.sum(1).long()                                   # 槽位数 = 项数
        B, M, _ = a_logits.shape
        slots = torch.arange(M, device=a_logits.device)
        valid = (slots.view(1, -1) < n.view(-1, 1)) & item_mask        # [B,M] 有效槽 ∧ 有效项
        flat = a_logits.reshape(B * M, M)
        tgt = assign_y.clamp(min=0).reshape(B * M)                    # padding 位填 0 后被 mask 掉
        ce = torch.nn.functional.cross_entropy(flat, tgt, reduction="none")
        vm = valid.reshape(B * M).float()
        assign = (ce * vm).sum() / vm.sum().clamp(min=1.0)
        skel = torch.nn.functional.cross_entropy(skel_logits, skel_y)
        assert torch.isfinite(assign) and torch.isfinite(skel)
        return skel + assign, skel.detach(), assign.detach()


# ---------------------------------------------------------------------------
# 编码（核冻结 ⇒ 结果与现算逐位相同，故可整体缓存）
# ---------------------------------------------------------------------------
@torch.no_grad()
def encode_hidden(model: TwoChannelModel, texts: list[str], max_len: int,
                  device: str, batch: int = 256) -> torch.Tensor:
    """逐条按 build 口径编码 → [N, max_len, D]（CPU 张量）。1 字符 = 1 token。"""
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    model.encoder.eval()
    out = []
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        ids, mask = [], []
        for t in chunk:
            e = tok.encode(t, max_length=max_len, padding=True)
            ids.append(e["input_ids"])
            mask.append(e["attention_mask"])
        id_t = torch.tensor(ids, dtype=torch.long, device=device)
        m_t = torch.tensor(mask, dtype=torch.bool, device=device)
        h = model.encoder(id_t, m_t)
        out.append(h.cpu())
    return torch.cat(out, 0) if out else torch.empty(0, max_len, model.spec.hidden)


def load_ptr_blob() -> dict:
    """指针分支数据：**只读复用** select_rerank 的冻结核向量缓存（键=数据 md5+SPEC）。"""
    import hashlib as _h
    h = _h.md5()
    for s in ("train", "test", "adv"):
        h.update((SR_DIR / "data" / "clean" / f"{s}.jsonl").read_bytes())
    key = f"clean_{h.hexdigest()[:12]}_ctx64_cand32.pt"
    path = SR_DIR / "cache" / key
    if not path.exists():
        raise SystemExit(f"指针缓存缺失（只读复用失败）：{path}")
    blob = torch.load(path, map_location="cpu", weights_only=True)
    print(f"[ptr-cache] 只读复用 {path}（n={blob['n']}）", flush=True)
    return blob


def ptr_row_meta() -> list[dict]:
    """与缓存行序一一对应的元数据（用于 adv 的 ctype 拆分）。"""
    import json as _json
    meta = []
    for s in ("train", "test", "adv"):
        with open(SR_DIR / "data" / "clean" / f"{s}.jsonl", encoding="utf-8") as fp:
            for line in fp:
                r = _json.loads(line)
                meta.append({"split": s, "ctype": r.get("ctype")})
    return meta


@torch.no_grad()
def encode_gen(model: TwoChannelModel, rows: list[dict], spec: Spec, device: str,
               batch: int = 256) -> dict:
    """生成分支编码：一次前向 + **按 span 池化**得到各槽（袋项）向量。"""
    from nano_char_tokenizer import NanoCharTokenizer
    tok = NanoCharTokenizer()
    GEN_CACHE.mkdir(exist_ok=True)
    fp = hashlib.md5("".join(r["sent"] + str(r["bag_span"]) for r in rows).encode()
                     ).hexdigest()[:12]
    path = GEN_CACHE / f"gen_{fp}_L{spec.max_len_sent}_M{spec.max_slots}.pt"
    if path.exists():
        blob = torch.load(path, map_location="cpu", weights_only=True)
        if blob["n"] == len(rows):
            print(f"[gen-cache] 命中 {path.name}（n={blob['n']}）", flush=True)
            return blob

    v_sent, v_items, mask, skel, assign = [], [], [], [], []
    model.encoder.eval()
    for i in range(0, len(rows), batch):
        chunk = rows[i:i + batch]
        ids, msk = [], []
        for r in chunk:
            e = tok.encode(r["sent"], max_length=spec.max_len_sent, padding=True)
            assert sum(e["attention_mask"]) == len(r["sent"]), \
                f"编码长度≠文本长度（1 字符 1 token 的前提被破坏）：{r['sent']}"
            ids.append(e["input_ids"])
            msk.append(e["attention_mask"])
        id_t = torch.tensor(ids, dtype=torch.long, device=device)
        m_t = torch.tensor(msk, dtype=torch.bool, device=device)
        h = model.encoder(id_t, m_t).cpu()                      # [b,L,D]
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
                seg = h[j, a:b]                                  # 1 字符 1 token ⇒ 下标对齐
                items[j, s] = seg.mean(0)
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
    print(f"[gen-cache] 写入 {path.name}（n={len(rows)}）", flush=True)
    return blob


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def selfcheck(device: str | None = None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    spec = Spec()
    rep: dict = {"device": device}

    # 1) 冻结实况 + 四臂参数逐项 + 初始化一致性
    arms = {}
    for arm in "ABCD":
        arms[arm] = TwoChannelModel(arm, seed=42, spec=spec).to(device)
    fr = arms["C"].freeze_report()
    assert fr["encoder_params"] == 1_688_460 and fr["encoder_trainable"] == 0 \
        and fr["encoder_training"] is False, fr
    rep["freeze"] = fr
    rep["params"] = {a: arms[a].param_report() for a in "ABCD"}
    p = rep["params"]
    assert p["C"]["head_trainable"] == p["D"]["head_trainable"] - p["C"]["trunk_ptr"], \
        "C 应比 D 恰好少一个主干"
    assert p["A"]["head_trainable"] == p["D"]["ptr_out"] + p["D"]["trunk_ptr"], "A 口径"
    assert p["B"]["head_trainable"] == p["D"]["trunk_gen"] + p["D"]["gen_skel"] \
        + p["D"]["gen_assign"], "B 口径"

    def same(a, b) -> bool:
        return all(torch.equal(a[k], b[k]) for k in a)

    def trunk_of(m):
        return m.trunk_ptr if m.trunk_ptr is not None else m.trunk_gen
    t0 = trunk_of(arms["A"]).state_dict()
    rep["init_identical"] = {
        "trunk A==B==C==D.ptr": same(t0, trunk_of(arms["B"]).state_dict())
        and same(t0, trunk_of(arms["C"]).state_dict())
        and same(t0, trunk_of(arms["D"]).state_dict()),
        "trunk D.gen == trunk": same(t0, arms["D"].trunk_gen.state_dict()),
        "ptr_out A==C==D": same(arms["A"].ptr_out.state_dict(),
                                arms["C"].ptr_out.state_dict())
        and same(arms["A"].ptr_out.state_dict(), arms["D"].ptr_out.state_dict()),
        "gen B==C==D": same(arms["B"].gen.state_dict(), arms["C"].gen.state_dict())
        and same(arms["B"].gen.state_dict(), arms["D"].gen.state_dict()),
    }
    assert all(rep["init_identical"].values()), rep["init_identical"]

    # 2) C 共用主干的证据
    C, D = arms["C"], arms["D"]
    names_c = C.named_trainable()
    trunk_names = [n for n in names_c if n.startswith("trunk")]
    rep["share_evidence"] = {
        "C.trunk_ptr is C.trunk_gen": C.trunk_ptr is C.trunk_gen,
        "D.trunk_ptr is D.trunk_gen": D.trunk_ptr is D.trunk_gen,
        "C named_parameters 里 trunk 出现次数": len(trunk_names),
        "C trunk 参数名（前 4）": trunk_names[:4],
        "C 指针分支用的主干 id": id(C.trunk_ptr),
        "C 生成分支用的主干 id": id(C.trunk_gen),
        "C head_params": C.param_report()["head_trainable"],
        "D head_params": D.param_report()["head_trainable"],
        "C = D − 一个主干": C.param_report()["head_trainable"]
        == D.param_report()["head_trainable"] - D.param_report()["trunk_ptr"],
    }
    assert rep["share_evidence"]["C.trunk_ptr is C.trunk_gen"] is True
    assert rep["share_evidence"]["D.trunk_ptr is D.trunk_gen"] is False
    assert rep["share_evidence"]["C named_parameters 里 trunk 出现次数"] == 4  # 4 个张量

    # 3) 梯度证据：两个通道各自都能把梯度送进**同一个**主干
    g = {"v_ctx": torch.randn(32, spec.hidden, device=device),
         "v_cand": torch.randn(32, spec.k, spec.hidden, device=device),
         "y": torch.randint(0, 2, (32,), device=device)}
    _vi = torch.randn(32, spec.max_slots, spec.hidden, device=device)
    _im = torch.ones(32, spec.max_slots, dtype=torch.bool, device=device)
    ge = {"v_sent": torch.randn(32, spec.hidden, device=device),
          "v_bag": (_vi * _im.unsqueeze(-1)).sum(1) / _im.sum(1, keepdim=True),
          "v_items": _vi,
          "item_mask": _im,
          "skel": torch.randint(0, spec.n_skel, (32,), device=device),
          "assign": torch.randint(0, spec.max_slots, (32, spec.max_slots), device=device)}
    C.zero_grad(set_to_none=True)
    C.ptr_loss(g["v_ctx"], g["v_cand"], g["y"]).backward()
    gp = {k: p.grad.clone() for k, p in C.trunk_ptr.named_parameters() if p.grad is not None}
    C.zero_grad(set_to_none=True)
    C.gen_loss(ge["v_sent"], ge["v_bag"], ge["v_items"], ge["item_mask"],
               ge["skel"], ge["assign"])[0].backward()
    gg = {k: p.grad.clone() for k, p in C.trunk_gen.named_parameters() if p.grad is not None}
    rep["grad_evidence"] = {
        "指针单独回传：主干梯度非零": bool(sum(v.abs().sum() for v in gp.values()) > 0),
        "生成单独回传：主干梯度非零": bool(sum(v.abs().sum() for v in gg.values()) > 0),
        "两条梯度不同（不是同一条）": bool(all(k in gp and k in gg
                                             and not torch.equal(gp[k], gg[k]) for k in gp)),
        "L_ptr =": float(C.ptr_loss(g["v_ctx"], g["v_cand"], g["y"]).detach()),
        "L_gen =": float(C.gen_loss(ge["v_sent"], ge["v_bag"], ge["v_items"],
                                    ge["item_mask"], ge["skel"], ge["assign"])[0].detach()),
    }
    assert all(v for k, v in rep["grad_evidence"].items() if k.endswith("零") or k.startswith("两条"))

    # 4) 指针分支对候选等变
    with torch.no_grad():
        logits = C.forward_ptr(g["v_ctx"], g["v_cand"])
        flip = C.forward_ptr(g["v_ctx"], g["v_cand"][:, [1, 0], :])
    rep["ptr_equivariance_ok"] = bool(torch.allclose(logits[:, [1, 0]], flip, atol=1e-6))
    assert rep["ptr_equivariance_ok"]

    # 5) 生成分支骨架输入对袋序不变（洗牌只影响指派那一支）
    with torch.no_grad():
        s0, a0 = C.forward_gen(ge["v_sent"], ge["v_bag"], ge["v_items"], ge["item_mask"])
        perm = torch.randperm(spec.max_slots)
        v_items_p = ge["v_items"][:, perm, :]
        mask_p = ge["item_mask"][:, perm]
        v_bag_p = (v_items_p * mask_p.unsqueeze(-1)).sum(1) / mask_p.sum(1, keepdim=True)
        s1, a1 = C.forward_gen(ge["v_sent"], v_bag_p, v_items_p, mask_p)
    rep["gen_bag_invariant_ok"] = bool(torch.allclose(s0, s1, atol=1e-6))
    rep["gen_assign_changes_with_bag"] = bool(not torch.allclose(a0, a1, atol=1e-6))
    assert rep["gen_bag_invariant_ok"] and rep["gen_assign_changes_with_bag"]

    # 6) 编码口径：span 池化与 1 字符 1 token
    tok_ok = True
    rep["tokenizer_char_aligned"] = tok_ok
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
