#!/usr/bin/env python3
"""方案 B 选择重排器（select_rerank）。

结构（`dev-notes/13` §9 方案 B，绕开 A 族失败的锚点环节）：
  冻结核 NanoDocEncoder 各编码一次 context 与 K 个 candidate（候选**不**拼进上下文，
  身份就是下标，不发射锚点）→ masked mean-pool → 共享 MLP 打分头（对 K 个候选**等变**）
  → softmax over K + CE。

不变量（`__main__` 自检，也是训练脚本的前置断言）：
  · 核参数 `requires_grad` 全 False、`training` 为 False（eval）
  · 交换两个候选 ⇒ 两个 logit **同样交换**（下标不是特征）
  · 编码口径与 `build_data.py::SPEC` 同源（截断守卫的另一半）

用法：uv run python experiments/select_rerank/model.py --selfcheck
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from dtseek.tasks.artifacts import load_base_encoder  # noqa: E402


@dataclass(frozen=True)
class RerankSpec:
    """截断与结构口径 —— 与 build_data.py::SPEC 逐字段一致（构建期已用同值守卫）。"""

    max_len_ctx: int = 64
    max_len_cand: int = 32
    hidden: int = 128
    head_hidden: int = 256
    k: int = 2
    base_path: str = "checkpoints/base_encoder.pt"
    extra: dict = field(default_factory=dict)

    @staticmethod
    def from_build_spec() -> "RerankSpec":
        """直接读 build_data.py 里的 SPEC，杜绝两处口径漂移。"""
        import importlib.util
        p = Path(__file__).resolve().parent / "build_data.py"
        spec = importlib.util.spec_from_file_location("sr_build_data", p)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return RerankSpec(max_len_ctx=mod.SPEC["max_len_ctx"],
                          max_len_cand=mod.SPEC["max_len_cand"])


class ScoringHead(nn.Module):
    """同一个 MLP 作用在 K 个候选上：score = MLP([ctx; cand; ctx⊙cand; |ctx−cand|])。

    对候选维**等变**（无逐下标参数）⇒ 下标不是特征；候选序打乱只翻转 logit 序。
    """

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


class RerankModel(nn.Module):
    def __init__(self, spec: RerankSpec | None = None):
        super().__init__()
        self.spec = spec or RerankSpec()
        self.encoder, self.ckpt = load_base_encoder(self.spec.base_path)
        # 冻结实况：load_base_encoder 已置 requires_grad=False + eval()，这里再钉一次并断言
        for p in self.encoder.parameters():
            p.requires_grad_(False)
        self.encoder.eval()
        self.head = ScoringHead(self.spec.hidden, self.spec.head_hidden)

    # ---- 冻结实况 ---------------------------------------------------------
    def freeze_report(self) -> dict:
        enc_p = sum(p.numel() for p in self.encoder.parameters())
        enc_train = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        head_p = sum(p.numel() for p in self.head.parameters())
        assert enc_train == 0, f"核没冻住：{enc_train} 个可训参数"
        assert not self.encoder.training, "核不在 eval() 模式"
        return {"encoder_params": enc_p, "encoder_trainable": enc_train,
                "encoder_training": self.encoder.training,
                "head_params": head_p, "total_params": enc_p + head_p,
                "hidden": self.spec.hidden,
                "max_len": getattr(self.encoder, "max_len", None)}

    # ---- 编码（核冻结 ⇒ 结果与现算逐位相同，故可整体缓存） ----------------
    @torch.no_grad()
    def encode_texts(self, texts: list[str], max_len: int, device: str = "cpu",
                     batch: int = 256) -> torch.Tensor:
        """逐条按 build 口径编码 → masked mean-pool → [N, D]（CPU 张量）。"""
        from nano_char_tokenizer import NanoCharTokenizer
        tok = NanoCharTokenizer()
        self.encoder.eval()
        out: list[torch.Tensor] = []
        for i in range(0, len(texts), batch):
            chunk = texts[i:i + batch]
            ids, mask = [], []
            for t in chunk:
                e = tok.encode(t, max_length=max_len, padding=True)
                ids.append(e["input_ids"])
                mask.append(e["attention_mask"])
            id_t = torch.tensor(ids, dtype=torch.long, device=device)
            m_t = torch.tensor(mask, dtype=torch.bool, device=device)
            h = self.encoder(id_t, m_t)                       # [B, L, D]
            m = m_t.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1.0)      # masked mean-pool
            out.append(v.cpu())
        return torch.cat(out, 0) if out else torch.empty(0, self.spec.hidden)

    def forward(self, v_ctx: torch.Tensor, v_cand: torch.Tensor) -> torch.Tensor:
        return self.head(v_ctx, v_cand)


# ---------------------------------------------------------------------------
# 自检
# ---------------------------------------------------------------------------
def selfcheck(device: str | None = None) -> dict:
    from nano_char_tokenizer import NanoCharTokenizer
    spec = RerankSpec.from_build_spec()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = RerankModel(spec).to(device)
    rep = model.freeze_report()

    # 1) 编码与截断守卫：正文必须原样编码（未被静默截断）
    tok = NanoCharTokenizer()
    t = "随和的他马上跟少林寺里一名七岁的小和尚清淳结下友谊之情"
    e = tok.encode(t, max_length=spec.max_len_ctx, padding=True)
    m = list(e["attention_mask"])
    assert len(e["input_ids"]) == spec.max_len_ctx and sum(m) == len(t) and m[:len(t)] == [1] * len(t), \
        f"编码口径不对：len={len(e['input_ids'])} sum(mask)={sum(m)} 文本长={len(t)}"
    v = model.encode_texts([t], spec.max_len_ctx, device=device)
    assert v.shape == (1, spec.hidden), v.shape
    assert torch.isfinite(v).all()

    # 2) 等变性：交换两候选 ⇒ logit 同样交换（下标不是特征）
    torch.manual_seed(0)
    v_ctx = torch.randn(64, spec.hidden, device=device)
    v_cand = torch.randn(64, spec.k, spec.hidden, device=device)
    logits = model(v_ctx, v_cand)
    flipped = model(v_ctx, v_cand[:, [1, 0], :])
    assert torch.allclose(logits[:, [1, 0]], flipped, atol=1e-6), "打分头不是候选等变的"
    assert logits.shape == (64, spec.k)

    # 3) 参数量与冻结
    assert rep["encoder_trainable"] == 0 and rep["encoder_training"] is False
    rep["equivariance_ok"] = True
    rep["spec"] = {"max_len_ctx": spec.max_len_ctx, "max_len_cand": spec.max_len_cand,
                   "head_hidden": spec.head_hidden}
    rep["device"] = device
    return rep


if __name__ == "__main__":
    import json
    print(json.dumps(selfcheck(), ensure_ascii=False, indent=2))
