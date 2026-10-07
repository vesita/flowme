#!/usr/bin/env python3
"""P20 三臂核（唯一变量 = 结构信息通路；全部随机初始化从头训）。

A：字符级双向编码 → masked mean 池化 → trunk → 头（无位置编码）
B：A + 位置编码 + attention pooling
C：B + 显式功能词/位置通道（功能词身份与位置直给头）

构造顺序按「子模块」固定播种 ⇒ 三臂**同名子模块初值逐位相同**（H0 可比性前提）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from data import FUNCWORD_SET, MAX_LEN, N_ENT, N_MODE, PAD

ARMS = ("A", "B", "C")
ARM_DESC = {
    "A": "双向编码→masked mean 池化→头；无位置编码（结构盲基线）",
    "B": "A + 位置编码 nn.Embedding(128,64) + attention pooling（单 query 加权和）",
    "C": "B + 功能词通道 f_fw(功能词位置嵌入均值)+f_fp(末个功能词位置编码)→Linear(128→64) 拼进 trunk",
}
D = 64
NLAYERS = 2
NHEAD = 4
FFN = 128
DROPOUT = 0.1
#: 子模块固定播种槽位（跨臂一致 ⇒ 同名子模块初值一致）
SEED_SLOT = {"emb": 1, "pos": 2, "enc": 3, "pool": 4, "fw": 5, "trunk": 6, "head": 7}


def _seed(base: int, slot: str) -> None:
    torch.manual_seed(base * 1000 + SEED_SLOT[slot])


def _layer() -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(d_model=D, nhead=NHEAD, dim_feedforward=FFN,
                                      dropout=DROPOUT, activation="gelu",
                                      batch_first=True)


class Core(nn.Module):
    def __init__(self, arm: str, vocab_size: int, fw_ids: set[int], base_seed: int = 42):
        super().__init__()
        assert arm in ARMS, arm
        self.arm = arm
        self.base_seed = base_seed
        self.fw_ids = sorted(fw_ids)

        _seed(base_seed, "emb")
        self.emb = nn.Embedding(vocab_size, D, padding_idx=PAD)
        _seed(base_seed, "pos")
        self.pos = nn.Embedding(MAX_LEN, D) if arm in ("B", "C") else None
        _seed(base_seed, "enc")
        self.enc = nn.ModuleList([_layer() for _ in range(NLAYERS)])
        if arm in ("B", "C"):
            _seed(base_seed, "pool")
            self.pool_w = nn.Linear(D, D)
            self.pool_v = nn.Linear(D, 1, bias=False)
        else:
            self.pool_w = self.pool_v = None
        if arm == "C":
            _seed(base_seed, "fw")
            self.fw_proj = nn.Linear(2 * D, D)
        else:
            self.fw_proj = None
        trunk_in = D * 2 if arm == "C" else D
        _seed(base_seed, "trunk")
        self.trunk = nn.Sequential(nn.Linear(trunk_in, D), nn.GELU(), nn.Linear(D, D))
        _seed(base_seed, "head")
        self.head_s = nn.Linear(D, N_MODE)
        self.head_t = nn.Linear(D, N_ENT)

    # ---- 池化（唯一变量之一）----
    def pool(self, h: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """h [B,L,D], mask [B,L] True=有效 → [B,D]"""
        if self.arm == "A":
            m = mask.unsqueeze(-1).float()
            return (h * m).sum(1) / m.sum(1).clamp(min=1.0)
        score = self.pool_v(torch.tanh(self.pool_w(h))).squeeze(-1)   # [B,L]
        score = score.masked_fill(~mask, -1e9)
        a = torch.softmax(score, dim=-1)
        return (a.unsqueeze(-1) * h).sum(1)

    def fw_channel(self, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """C 专用：功能词身份均值 + 末个功能词位置编码。"""
        if self.fw_ids:
            fw = torch.isin(ids, torch.as_tensor(self.fw_ids, device=ids.device)) & mask
        else:
            fw = torch.zeros_like(mask)
        m = fw.unsqueeze(-1).float()
        f_fw = (self.emb(ids) * m).sum(1) / m.sum(1).clamp(min=1.0)
        f_fw = f_fw * (m.sum(1) > 0).float()                        # 无功能词 → 0（m.sum(1) 已是 [B,1]）
        idx = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0).expand_as(ids)
        last = torch.where(fw, idx, torch.full_like(idx, -1)).max(dim=1).values
        has = last >= 0
        f_fp = torch.where(has.unsqueeze(-1),
                           self.pos(last.clamp(min=0)),
                           torch.zeros(ids.shape[0], D, device=ids.device))
        return self.fw_proj(torch.cat([f_fw, f_fp], dim=-1))

    def encode(self, ids: torch.Tensor) -> torch.Tensor:
        """→ 池化后表示（C 为 2D 拼接）；H4 probe 抽这一层。"""
        mask = ids != PAD
        x = self.emb(ids)
        if self.pos is not None:
            x = x + self.pos(torch.arange(ids.shape[1], device=ids.device)).unsqueeze(0)
        for layer in self.enc:
            x = layer(x, src_key_padding_mask=~mask)
        z = self.pool(x, mask)
        if self.arm == "C":
            z = torch.cat([z, self.fw_channel(ids, mask)], dim=-1)
        return z

    def forward(self, ids: torch.Tensor, task: str) -> torch.Tensor:
        h = self.trunk(self.encode(ids))
        return self.head_s(h) if task == "S" else self.head_t(h)


def count_params(model: Core) -> dict:
    per: dict[str, int] = {}
    total = 0
    for name, p in model.named_parameters():
        grp = name.split(".")[0]
        per[grp] = per.get(grp, 0) + p.numel()
        total += p.numel()
    return {"total": total, "per_group": dict(sorted(per.items()))}


def build_all(vocab_size: int, fw_ids: set[int], base_seed: int = 42) -> dict[str, Core]:
    return {a: Core(a, vocab_size, fw_ids, base_seed) for a in ARMS}


def assert_arm_isolation(models: dict[str, Core]) -> dict:
    """构造性自检：同名子模块跨臂初值逐位相同；差异只在声明部位；A 池化置换不变。"""
    a, b, c = models["A"], models["B"], models["C"]
    rep: dict = {}
    rep["emb_A==B"] = bool(torch.equal(a.emb.weight, b.emb.weight))
    rep["emb_A==C"] = bool(torch.equal(a.emb.weight, c.emb.weight))
    for i in range(NLAYERS):
        for nm in ("self_attn.in_proj_weight", "linear1.weight", "linear2.weight"):
            pa = dict(a.enc[i].named_parameters())[nm]
            pb = dict(b.enc[i].named_parameters())[nm]
            pc = dict(list(c.enc[i].named_parameters()))[nm]
            rep[f"enc{i}.{nm}_A==B"] = bool(torch.equal(pa, pb))
            rep[f"enc{i}.{nm}_A==C"] = bool(torch.equal(pa, pc))
    rep["A_has_pos"] = a.pos is None and b.pos is not None and c.pos is not None
    rep["A_pool_is_mean"] = (a.pool_w is None and b.pool_w is not None
                             and c.pool_w is not None)
    rep["C_has_fw_channel"] = c.fw_proj is not None and a.fw_proj is None
    rep["head_s_A==B==C"] = bool(torch.equal(a.head_s.weight, b.head_s.weight)
                                 and torch.equal(a.head_s.weight, c.head_s.weight))
    # A 的池化对输入置换不变（构造性数值验证）
    torch.manual_seed(0)
    ids = torch.randint(5, 40, (4, 32))
    h0 = a.pool(a.emb(ids), ids != PAD)
    perm = torch.randperm(32)
    h1 = a.pool(a.emb(ids[:, perm]), ids[:, perm] != PAD)
    rep["A_pool_perm_inv_maxdiff"] = float((h0 - h1).detach().abs().max())
    assert all(v for k, v in rep.items()
               if k.endswith(("_A==B", "_A==C", "_A==B==C"))), rep
    assert rep["A_has_pos"] and rep["A_pool_is_mean"] and rep["C_has_fw_channel"], rep
    assert rep["A_pool_perm_inv_maxdiff"] < 1e-6, rep
    return rep
