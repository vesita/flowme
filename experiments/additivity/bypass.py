"""per-module 低秩旁路适配器（冻结认知核上的插件旁路）—— E-B 实验。

形式：`h ← h + scale * (h @ Aᵀ @ Bᵀ)`，`A ∈ R^{r×d}` 小初始化 `N(0, 1/d)`、
`B = 0` 精确零初始化 ⇒ **初始行为与冻结基座逐位一致**（自检 2 的前提）。

gate 语义（老模块零损伤的结构性保证）：
- `enabled = False` → hook 直接返回原张量，**不加任何算术** ⇒ 老模块的计算路径
  与「没装旁路」严格同一条（逐位一致不是巧合，是这条分支保证的）；
- `enabled = True` → 加上旁路项。旁路挂在**编码器**上，但开关是按模块（任务）设的：
  谁的旁路谁打开，别人的不动。
"""
from __future__ import annotations

import torch
from torch import nn


class LoRABypass(nn.Module):
    """单点低秩旁路：h + scale * (h @ Aᵀ @ Bᵀ)。"""

    def __init__(self, dim: int, rank: int, alpha: float, generator: torch.Generator | None = None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.A = nn.Parameter(torch.randn(rank, dim, generator=generator) / (dim ** 0.5))
        self.B = nn.Parameter(torch.zeros(dim, rank))   # 精确零 ⇒ 初始恒等

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + self.scale * ((h @ self.A.T) @ self.B.T)

    @torch.no_grad()
    def stats(self) -> dict:
        a, b = self.A, self.B
        return {
            "A_std": float(a.std()), "A_norm": float(a.norm()),
            "B_std": float(b.std()), "B_norm": float(b.norm()),
        }


class BypassSet(nn.Module):
    """挂到 NanoDocEncoder 上的一组旁路 + 显式开关。

    插入点（本实验预注册的 4 处）：
      blocks[0]、blocks[1]、blocks[2] 的输出（残差流）与 norm 的输出（进解码头之前）。
    """

    def __init__(self, encoder: nn.Module, rank: int = 16, alpha: float = 16.0,
                 generator: torch.Generator | None = None, auto_hooks: bool = True):
        super().__init__()
        dim = encoder.hidden_dim
        n_blocks = len(encoder.blocks)
        self.rank, self.alpha = rank, alpha
        self.sites = [f"blocks[{i}]" for i in range(n_blocks)] + ["norm"]
        self.adapters = nn.ModuleList(
            [LoRABypass(dim, rank, alpha, generator) for _ in self.sites])
        self.enabled = False

        # 不用 self._encoder = encoder：nn.Module.__setattr__ 会把编码器注册成子模块
        # ⇒ state_dict 出现 encoder.bypass.… 互相递归（RecursionError，已实测踩过一次）
        object.__setattr__(self, "_encoder", encoder)
        self._handles = []
        # 注册为编码器的子模块：参数随 .to(device)/.eval()/parameters() 走
        encoder.bypass = self
        if auto_hooks:
            self.attach_hooks()

    def attach_hooks(self) -> None:
        """把旁路挂上编码器（可重复调用前先 close）。"""
        assert not self._handles, "hooks 已挂载"
        enc = self._encoder

        def make_hook(idx: int):
            def hook(_mod, _inp, out):
                # 关闸：原样返回，不做任何算术（老模块路径严格不变）
                return out if not self.enabled else self.adapters[idx](out)
            return hook

        n_blocks = len(enc.blocks)
        for i, blk in enumerate(enc.blocks):
            self._handles.append(blk.register_forward_hook(make_hook(i)))
        self._handles.append(enc.norm.register_forward_hook(make_hook(n_blocks)))

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    # ---- 生命周期 / 存取 ---------------------------------------------------
    def close(self, encoder: nn.Module) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        if getattr(encoder, "bypass", None) is self:
            delattr(encoder, "bypass")

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @torch.no_grad()
    def stats(self) -> dict:
        return {site: ad.stats() for site, ad in zip(self.sites, self.adapters)}

    @torch.no_grad()
    def snapshot(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    @torch.no_grad()
    def load(self, state: dict) -> None:
        self.load_state_dict(state)


def trainable_report(encoder: nn.Module, decoder: nn.Module, bypass: BypassSet | None) -> dict:
    """自检 1 用：打印**生效后**的可训参数量（不看假设，直接按 requires_grad 数）。

    注意：旁路被注册成 encoder 的子模块，所以数编码器时要按名字前缀把 `bypass.` 排掉，
    否则旁路参数会被同时算进「基座」和「旁路」两栏（双计）。
    """
    enc_core = [(n, p) for n, p in encoder.named_parameters() if not n.startswith("bypass.")]

    def cnt(pairs, pred):
        return sum(p.numel() for _, p in pairs if pred(p))

    byp_pairs = ([(n, p) for n, p in bypass.named_parameters()] if bypass is not None else [])
    head_pairs = [(n, p) for n, p in decoder.named_parameters()]
    return {
        "base_core_total": sum(p.numel() for _, p in enc_core),
        "base_core_frozen": cnt(enc_core, lambda p: not p.requires_grad),
        "base_core_trainable": cnt(enc_core, lambda p: p.requires_grad),   # 必须为 0
        "head_trainable": cnt(head_pairs, lambda p: p.requires_grad),
        "head_total": sum(p.numel() for _, p in head_pairs),
        "bypass_total": sum(p.numel() for _, p in byp_pairs),
        "bypass_trainable": cnt(byp_pairs, lambda p: p.requires_grad),
        "trainable_total": (cnt(enc_core, lambda p: p.requires_grad)
                            + cnt(head_pairs, lambda p: p.requires_grad)
                            + cnt(byp_pairs, lambda p: p.requires_grad)),
    }
