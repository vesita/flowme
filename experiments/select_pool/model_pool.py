#!/usr/bin/env python3
"""select_pool —— 四臂聚合消融（mean / max / attention / cross-attention）+ 参数量对齐对照臂。

唯一问题（PREREG §0）：B 族 `heldout_pair ≈ 50%` 那堵墙是**池化抹掉了信号**，还是**表示里没有语义等价**？
差别**只在「token 级隐藏 → 向量」的聚合方式**，其余逐项同 B 族（`experiments/select_rerank`）：
冻结核、不发射锚点、同样的打分头与训练配方。

结构：
  冻结核 NanoDocEncoder 各编码 1 次 context 与 K=2 次 candidate（**候选不拼进上下文**）
  → 【本实验的唯一变量：聚合方式】→ 打分头 MLP([ctx; cand; ctx⊙cand; |ctx−cand|]) → softmax over K + CE

不变量（`__main` --selfcheck，也是训练脚本的前置断言）：
  · 核参数 requires_grad 全 False、training 为 False（eval）
  · 每臂：交换两候选 ⇒ 两个 logit 同步交换（下标不是特征）
  · 打分头参数量 = 164,353（与 B 族逐位同构）、核参数量 = 1,688,460
  · 编码口径与 select_rerank/build_data.py::SPEC 同源（截断守卫的另一半）

用法：uv run python experiments/select_pool/model_pool.py --selfcheck
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402

ARMS = ("A", "B", "C", "D", "Aplus")
ARM_DESC = {
    "A": "mean-pool（B 族基线）",
    "B": "max-pool",
    "C": "可学 attention-pool（可学 query ×2）",
    "D": "cross-attention（候选 token → 上文 token，ctx 侧仍 mean）",
    "Aplus": "mean-pool + 可学逐维缩放（与 C 参数量对齐的对照臂）",
}
HEAD_PARAMS_B族 = 164_353
ENC_PARAMS_B族 = 1_688_460


@dataclass(frozen=True)
class PoolSpec:
    """截断与结构口径 —— 与 select_rerank/build_data.py::SPEC 逐字段一致。"""

    max_len_ctx: int = 64
    max_len_cand: int = 32
    hidden: int = 128
    head_hidden: int = 256
    k: int = 2
    base_path: str = "checkpoints/base_encoder.pt"
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_build_spec() -> "PoolSpec":
        """直接读 select_rerank/build_data.py 里的 SPEC（只读），杜绝两处口径漂移。"""
        import importlib.util
        p = HERE.parent / "select_rerank" / "build_data.py"
        spec = importlib.util.spec_from_file_location("sp_build_data", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return PoolSpec(max_len_ctx=mod.SPEC["max_len_ctx"],
                        max_len_cand=mod.SPEC["max_len_cand"])


# ---------------------------------------------------------------------------
# 打分头（与 select_rerank/model.py::ScoringHead 逐行同构）
# ---------------------------------------------------------------------------
class ScoringHead(nn.Module):
    """同一个 MLP 作用在 K 个候选上：score = MLP([ctx; cand; ctx⊙cand; |ctx−cand|])。"""

    def __init__(self, hidden: int = 128, head_hidden: int = 256):
        super().__init__()
        in_dim = hidden * 4
        self.net = nn.Sequential(
            nn.Linear(in_dim, head_hidden),
            nn.GELU(),
            nn.Linear(head_hidden, head_hidden // 2),
            nn.GELU(),
            nn.Linear(head_hidden // 2, 1),
        )

    def forward(self, v_ctx: torch.Tensor, v_cand: torch.Tensor) -> torch.Tensor:
        """v_ctx [B,D], v_cand [B,K,D] → logits [B,K]"""
        B, K, D = v_cand.shape
        c = v_ctx.unsqueeze(1).expand(-1, K, -1)
        x = torch.cat([c, v_cand, c * v_cand, (c - v_cand).abs()], dim=-1)
        return self.net(x).squeeze(-1)


# ---------------------------------------------------------------------------
# 四臂 + 对照臂的聚合（唯一变量；padding 一律 masked）
# ---------------------------------------------------------------------------
def _axis(h: torch.Tensor, m: torch.Tensor) -> int:
    """token 轴：ctx h=[B,L,D] → 1；cand h=[B,K,L,D] → 2（mask 同号）。"""
    assert h.dim() == m.dim() + 1 and h.dim() in (3, 4), f"h {tuple(h.dim())} m {tuple(m.dim())}"
    ax = h.dim() - 2
    assert ax == m.dim() - 1
    return ax


def masked_mean(h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """按 token 轴 masked mean：ctx [B,L,D]→[B,D]，cand [B,K,L,D]→[B,K,D]。"""
    ax = _axis(h, m)
    mf = m.unsqueeze(-1).to(h.dtype)
    return (h * mf).sum(ax) / mf.sum(ax).clamp(min=1.0)


def masked_max(h: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """按 token 轴 masked max（无效位不参与，逐维取最大）。"""
    ax = _axis(h, m)
    assert bool(m.any(dim=ax).all()), "有整行全 padding 的样本（max-pool 会取到 -inf）"
    neg_inf = torch.finfo(h.dtype).min
    return h.masked_fill(~m.unsqueeze(-1), neg_inf).max(ax).values


class Aggregator(nn.Module):
    """把 (ctx token 序列, cand token 序列) 聚合成打分头要的两个向量。"""

    def __init__(self, arm: str, hidden: int = 128):
        super().__init__()
        assert arm in ARMS, f"未知臂：{arm}"
        self.arm = arm
        self.hidden = hidden
        if arm == "C":
            # 两侧各一个可学 query（256 参数）
            self.q_ctx = nn.Parameter(torch.randn(hidden) * 0.02)
            self.q_cand = nn.Parameter(torch.randn(hidden) * 0.02)
        elif arm == "D":
            # 可学温度（1 参数）：t = exp(theta)，theta 初值 = -0.5*ln(128) ⇒ t = 1/sqrt(128)
            self.theta = nn.Parameter(torch.full((), -0.5 * math.log(hidden)))
        elif arm == "Aplus":
            # 与 C 参数量逐个对齐（256 参数），初值全 1 ⇒ 起点等价于 A
            self.g_ctx = nn.Parameter(torch.ones(hidden))
            self.g_cand = nn.Parameter(torch.ones(hidden))

    @property
    def params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ---- 单侧聚合 ---------------------------------------------------------
    def _pool(self, h: torch.Tensor, m: torch.Tensor, side: str) -> torch.Tensor:
        if self.arm == "B":
            v = masked_max(h, m)
        elif self.arm == "C":
            q = self.q_ctx if side == "ctx" else self.q_cand
            logits = (h @ q) / math.sqrt(self.hidden)          # token 轴上的打分
            logits = logits.masked_fill(~m, torch.finfo(h.dtype).min)
            a = torch.softmax(logits, dim=-1)
            v = (a.unsqueeze(-1) * h).sum(_axis(h, m))
        else:
            # A / Aplus（两侧都 mean）与 D 的 ctx 侧：mean-pool（与 B 族逐位同式）
            assert self.arm in ("A", "Aplus") or (self.arm == "D" and side == "ctx"), \
                f"臂 {self.arm} 侧 {side}：候选侧必须走 _cross_cand"
            v = masked_mean(h, m)
        return v

    def _cross_cand(self, h_cand: torch.Tensor, m_cand: torch.Tensor,
                    h_ctx: torch.Tensor, m_ctx: torch.Tensor) -> torch.Tensor:
        """臂 D：候选的每个 token 对上下文 token 做单头注意力（无投影、raw dot × 可学温度），
        再按候选 mask 在 token 维 mean-pool。ctx 侧不参与改动（仍是 mean-pool）。"""
        assert bool(m_ctx.any(dim=1).all()), "有整行全 padding 的上下文"
        t = torch.exp(self.theta)                               # 标量温度
        scores = torch.einsum("bkcd,bld->bkcl", h_cand, h_ctx) * t   # [B,K,Lc,Lx]
        scores = scores.masked_fill(~m_ctx[:, None, None, :],
                                    torch.finfo(h_cand.dtype).min)
        a = torch.softmax(scores, dim=-1)                        # 在上下文 token 维归一
        out = torch.einsum("bkcl,bld->bkcd", a, h_ctx)           # [B,K,Lc,D]
        return masked_mean(out, m_cand)                          # [B,K,D]

    # ---- 两侧一起 ---------------------------------------------------------
    def aggregate(self, h_ctx: torch.Tensor, m_ctx: torch.Tensor,
                  h_cand: torch.Tensor, m_cand: torch.Tensor):
        """返回 (v_ctx [B,D], v_cand [B,K,D])"""
        v_ctx = self._pool(h_ctx, m_ctx, "ctx")
        if self.arm == "D":
            v_cand = self._cross_cand(h_cand, m_cand, h_ctx, m_ctx)
        else:
            v_cand = self._pool(h_cand, m_cand, "cand")
        if self.arm == "Aplus":
            v_ctx = v_ctx * self.g_ctx
            v_cand = v_cand * self.g_cand
        return v_ctx, v_cand


# ---------------------------------------------------------------------------
# 模型
# ---------------------------------------------------------------------------
class PoolModel(nn.Module):
    def __init__(self, arm: str, spec: PoolSpec | None = None):
        super().__init__()
        self.arm = arm
        self.spec = spec or PoolSpec()
        self.encoder, self.ckpt = load_base_encoder(self.spec.base_path)
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.agg = Aggregator(arm, self.spec.hidden)
        self.head = ScoringHead(self.spec.hidden, self.spec.head_hidden)

    # ---- 冻结与参数量实况 ---------------------------------------------------
    def freeze_report(self) -> dict:
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        enc_train = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        agg_p = self.agg.params
        head_p = sum(p.numel() for p in self.head.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        assert enc_train == 0, f"核没冻住：{enc_train} 个可训参数"
        assert not self.encoder.training, "核不在 eval() 模式"
        assert head_p == HEAD_PARAMS_B族, f"打分头与 B 族不同构：{head_p}"
        assert enc_p == ENC_PARAMS_B族, f"核与 B 族不同：{enc_p}"
        assert trainable == agg_p + head_p, \
            f"可训参数对不上：{trainable} != 聚合 {agg_p} + 头 {head_p}"
        return {"arm": self.arm, "arm_desc": ARM_DESC[self.arm],
                "encoder_params": enc_p, "encoder_trainable": enc_train,
                "encoder_training": self.encoder.training,
                "agg_params": agg_p, "head_params": head_p,
                "trainable_params": trainable, "total_params": enc_p + agg_p + head_p,
                "hidden": self.spec.hidden,
                "max_len": getattr(self.encoder, "max_len", None)}

    # ---- 编码（核冻结 ⇒ 与 B 族 encode_texts 同式同序同 batch） ---------------
    @torch.no_grad()
    def encode_tokens(self, texts: list[str], max_len: int, device: str = "cpu",
                      batch: int = 256) -> tuple[torch.Tensor, torch.Tensor]:
        """逐条按 build 口径编码 → token 级隐藏（不池化）→ (h [N,L,D] cpu, m [N,L] cpu bool)。"""
        from nano_char_tokenizer import NanoCharTokenizer
        tok = NanoCharTokenizer()
        self.encoder.eval()
        hs: list[torch.Tensor] = []
        ms: list[torch.Tensor] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            ids, mask = [], []
            for t in chunk:
                e = tok.encode(t, max_length=max_len, padding=True)
                ids.append(e["input_ids"])
                mask.append(e["attention_mask"])
            id_t = torch.tensor(ids, dtype=torch.long, device=device)
            m_t = torch.tensor(mask, dtype=torch.bool, device=device)
            h = self.encoder(id_t, m_t)                          # [B, L, D]
            hs.append(h.cpu())
            ms.append(m_t.cpu())
        if not hs:
            return (torch.empty(0, max_len, self.spec.hidden),
                    torch.empty(0, max_len, dtype=torch.bool))
        return torch.cat(hs, 0), torch.cat(ms, 0)

    def forward_tokens(self, h_ctx, m_ctx, h_cand, m_cand) -> torch.Tensor:
        v_ctx, v_cand = self.agg.aggregate(h_ctx, m_ctx, h_cand, m_cand)
        return self.head(v_ctx, v_cand)

    def forward(self, v_ctx: torch.Tensor, v_cand: torch.Tensor) -> torch.Tensor:
        """打分头直连（只有臂 A/Aplus 与聚合结果一致；训练/评测一律走 forward_tokens）。"""
        return self.head(v_ctx, v_cand)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def selfcheck(device: str | None = None, parity: bool = False) -> dict:
    from nano_char_tokenizer import NanoCharTokenizer
    spec = PoolSpec.from_build_spec()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    out: dict = {"device": device, "spec": {"max_len_ctx": spec.max_len_ctx,
                                            "max_len_cand": spec.max_len_cand,
                                            "head_hidden": spec.head_hidden,
                                            "hidden": spec.hidden},
                 "arms": {}}

    # 0) 编码与截断守卫：正文必须原样编码（未被静默截断）
    tok = NanoCharTokenizer()
    t = "随和的他马上跟少林寺里一名七岁的小和尚清淳结下友谊之情"
    e = tok.encode(t, max_length=spec.max_len_ctx, padding=True)
    m = list(e["attention_mask"])
    assert len(e["input_ids"]) == spec.max_len_ctx and sum(m) == len(t) and m[:len(t)] == [1] * len(t), \
        f"编码口径不对：len={len(e['input_ids'])} sum(mask)={sum(m)} 文本长={len(t)}"

    B = 64
    torch.manual_seed(0)
    for arm in ARMS:
        model = PoolModel(arm, spec).to(device)
        rep = model.freeze_report()

        # 1) token 级编码形状 + 池化有限
        h_ctx, m_ctx = model.encode_tokens([t], spec.max_len_ctx, device=device)
        assert h_ctx.shape == (1, spec.max_len_ctx, spec.hidden) and torch.isfinite(h_ctx).all()

        # 2) 等变性：随机 token 序列，交换两候选 ⇒ logit 同样交换
        Lx, Lc = 24, 16
        r_ctx = torch.randn(B, Lx, spec.hidden, device=device)
        r_mc = torch.rand(B, Lx, device=device) > 0.3
        r_mc[:, 0] = True
        r_cd = torch.randn(B, 2, Lc, spec.hidden, device=device)
        r_md = torch.rand(B, 2, Lc, device=device) > 0.3
        r_md[:, :, 0] = True
        logits = model.forward_tokens(r_ctx, r_mc, r_cd, r_md)
        flipped = model.forward_tokens(r_ctx, r_mc, r_cd[:, [1, 0], :], r_md[:, [1, 0], :])
        assert torch.allclose(logits[:, [1, 0]], flipped, atol=1e-6), \
            f"臂 {arm} 打分不是候选等变的"
        assert logits.shape == (B, spec.k) and torch.isfinite(logits).all()
        assert logits.float().var() > 0, f"臂 {arm} logits 是常数（空跑）"

        # 3) 聚合层确实有梯度（非空测试：可训聚合参数拿得到梯度）
        v_ctx, v_cand = model.agg.aggregate(r_ctx, r_mc, r_cd, r_md)
        loss = model.head(v_ctx, v_cand).sum()
        loss.backward()
        gagg = [p.grad is not None for p in model.agg.parameters()]
        if model.agg.params:
            assert all(gagg), f"臂 {arm} 聚合层有参数拿不到梯度：{gagg}"
        rep["agg_grad_ok"] = True
        rep["equivariant_ok"] = True
        out["arms"][arm] = rep
        del model
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    # 4) 臂 A 的池化与 B 族逐位同式（口径校验在 train 脚本里对真实缓存做）
    out["parity_checked_in_train"] = bool(parity)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    assert a.selfcheck, "用 --selfcheck 运行"
    print(json.dumps(selfcheck(), ensure_ascii=False, indent=2))
