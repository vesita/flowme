"""并联分支适配器（E-D / core_branch）—— 非线性或更深的可训分支，挂在冻结认知核上。

三档（kind）：
  mlp    —— 4 个插入点（blocks[0..2] 输出 + norm 输出）上的 MLP 旁路：
             h ← h + W2 · SiLU(W1 · h)，W1 ~ N(0, 1/d)、**W2 精确零** ⇒ 初始恒等；
  block1 —— 1 个完整 NanoTransformerBlock（Pre-RMSNorm 残差式）作用于 doc_memory（norm 输出），
             **attn.proj 与 mlp.c_proj 精确零** ⇒ block(x) ≡ x 初始恒等；
  block2 —— 2 个块堆叠，两个块的输出投影都零初始化。

gate 语义（老卡零损伤的结构性保证，与 E-B `bypass.py` 同款）：
  - 关闸 → forward hook **直接返回原张量对象**，不加任何算术 ⇒ 老卡路径与"没装分支"严格同一条；
  - 开闸 → 才计算分支；block 档用 forward_pre_hook 记录 RoPE/掩码上下文，
    它只读输入与 buffer、不触碰返回张量（关闸路径的输出值与无 hook 时逐位相同）。

不改 `src/`：块直接 import 自 dtseek.encoder.nano_doc_encoder.NanoTransformerBlock。
分支构造包在 `fork_rng(devices=[])` 里、只用 CPU `default_generator` 播种 ⇒
**不扰动全局 RNG 流**，头初始化与数据顺序与 E-B 同 seed 逐位对齐（配对可比）。

空测试打印（SELFTEST_1）的 `trainable_report` 复用 `experiments/additivity/bypass.py` 的口径，
仅把前缀 `bypass.` 换成 `branch.`（那边不改）。
"""
from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from dtseek.encoder.nano_doc_encoder import NanoTransformerBlock


# ---------------------------------------------------------------------------
# B1：MLP 旁路
# ---------------------------------------------------------------------------

class MLPBypass(nn.Module):
    """单点 MLP 旁路：h + W2 · SiLU(W1 · h)。W2 精确零 ⇒ 初始恒等。"""

    def __init__(self, dim: int, hidden: int, generator: torch.Generator | None = None):
        super().__init__()
        self.hidden = hidden
        # W1: [hidden, dim]；W2: [dim, hidden]
        self.W1 = nn.Parameter(torch.randn(hidden, dim, generator=generator) / (dim ** 0.5))
        self.W2 = nn.Parameter(torch.zeros(dim, hidden))   # 精确零 ⇒ 初始恒等

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        return h + (F.silu(h @ self.W1.T) @ self.W2.T)

    @torch.no_grad()
    def stats(self) -> dict:
        return {"W1_std": float(self.W1.std()), "W1_norm": float(self.W1.norm()),
                "W2_std": float(self.W2.std()), "W2_norm": float(self.W2.norm())}


# ---------------------------------------------------------------------------
# B2/B3：完整 Transformer 块
# ---------------------------------------------------------------------------

def _build_block(encoder: nn.Module) -> NanoTransformerBlock:
    """按编码器现有块的结构参数构造一个新块（结构同源，权重全新）。"""
    ref = encoder.blocks[0]
    blk = NanoTransformerBlock(
        hidden_dim=encoder.hidden_dim,
        num_heads=ref.attn.num_heads,
        dropout=ref.attn.dropout_p,
        rope_theta=ref.attn.rope_theta,
        use_qk_norm=ref.attn.use_qk_norm,
        swiglu_clamp=ref.mlp.clamp,
    )
    # fail-closed：SwiGLU 宽度必须与基座一致（scale 不可从权重反推，只能验终值）
    assert blk.mlp.c_fc.out_features == ref.mlp.c_fc.out_features, \
        f"分支块 SwiGLU 宽度 {blk.mlp.c_fc.out_features} != 基座 {ref.mlp.c_fc.out_features}"
    return blk


def _zero_output_projections(blk: NanoTransformerBlock) -> None:
    """零初始化两处输出投影 ⇒ block(x) ≡ x（初始恒等，空测试②的前提）。"""
    nn.init.zeros_(blk.attn.proj.weight)
    nn.init.zeros_(blk.mlp.c_proj.weight)


# ---------------------------------------------------------------------------
# BranchSet：挂到 NanoDocEncoder 上的一组分支 + 显式开关
# ---------------------------------------------------------------------------

class BranchSet(nn.Module):
    """并联分支 + 显式开关。属性注册名沿用 `encoder.branch`（E-B 口径里对应 `bypass.`）。"""

    def __init__(self, encoder: nn.Module, kind: str = "mlp", mlp_hidden: int = 128,
                 seed: int = 0, auto_hooks: bool = True):
        super().__init__()
        assert kind in ("mlp", "block1", "block2"), kind
        dim = encoder.hidden_dim
        n_enc_blocks = len(encoder.blocks)
        self.kind = kind
        self.mlp_hidden = mlp_hidden
        self.seed = seed

        if kind == "mlp":
            self.sites = [f"blocks[{i}]" for i in range(n_enc_blocks)] + ["norm"]
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 1000)
                self.branch = nn.ModuleList(
                    [MLPBypass(dim, mlp_hidden) for _ in self.sites])
        else:
            n = 1 if kind == "block1" else 2
            self.sites = [f"norm+{n}blk"]
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(seed + 1000)
                blocks = []
                for _ in range(n):
                    blk = _build_block(encoder)
                    _zero_output_projections(blk)
                    blocks.append(blk)
                self.branch = nn.ModuleList(blocks)

        self.enabled = False

        # 不用 self._encoder = encoder：nn.Module.__setattr__ 会把编码器注册成子模块
        # ⇒ state_dict 互相递归（E-B 实测踩过 RecursionError，见其 REPORT §8）
        object.__setattr__(self, "_encoder", encoder)
        self._handles = []
        self._ctx: dict = {}
        encoder.branch = self            # 参数随 .to(device)/.eval()/parameters() 走
        if auto_hooks:
            self.attach_hooks()

    # ---- hook ------------------------------------------------------------
    def attach_hooks(self) -> None:
        assert not self._handles, "hooks 已挂载"
        enc = self._encoder
        if self.kind == "mlp":
            # 与 E-B BypassSet 逐行同款：4 处 forward hook，关闸原样返回
            def make_hook(idx: int):
                def hook(_mod, _inp, out):
                    return out if not self.enabled else self.branch[idx](out)
                return hook
            for i, blk in enumerate(enc.blocks):
                self._handles.append(blk.register_forward_hook(make_hook(i)))
            self._handles.append(
                enc.norm.register_forward_hook(make_hook(len(enc.blocks))))
        else:
            # 上下文捕获：只读输入与 buffer，不碰返回张量
            self._handles.append(
                enc.register_forward_pre_hook(self._capture_ctx, with_kwargs=True))

            def hook(_mod, _inp, out):
                if not self.enabled:
                    return out                 # 关闸：原样返回，零算术
                x = out
                ctx = self._ctx
                for blk in self.branch:
                    x = blk(x, ctx["cos"], ctx["sin"], ctx["mask"])
                return x
            self._handles.append(enc.norm.register_forward_hook(hook))

    def _capture_ctx(self, module: nn.Module, args, kwargs) -> None:
        """记下本前向的 RoPE 表切片与 SDPA 掩码（与 encoder 正文同款切法）。"""
        input_ids = args[0]
        mask = kwargs.get("attention_mask")
        if mask is None and len(args) > 1:
            mask = args[1]
        B, L = input_ids.shape
        self._ctx = {
            "cos": module.rope_cos[:L],
            "sin": module.rope_sin[:L],
            "mask": None if mask is None else mask.to(torch.bool).view(B, 1, 1, L),
        }

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    # ---- 生命周期 / 存取 ---------------------------------------------------
    def close(self, encoder: nn.Module) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []
        if getattr(encoder, "branch", None) is self:
            delattr(encoder, "branch")

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def zero_init_keys(self) -> list[str]:
        """零初始化的 state_dict 键（空测试③：这些键训后必须离开零）。"""
        if self.kind == "mlp":
            return [f"branch.{i}.W2" for i in range(len(self.sites))]
        keys = []
        for i in range(len(self.branch)):
            keys += [f"branch.{i}.attn.proj.weight", f"branch.{i}.mlp.c_proj.weight"]
        return keys

    @torch.no_grad()
    def stats(self) -> dict:
        if self.kind == "mlp":
            return {site: ad.stats() for site, ad in zip(self.sites, self.branch)}
        out = {}
        for i, blk in enumerate(self.branch):
            out[f"blk{i}"] = {
                "attn_proj_norm": float(blk.attn.proj.weight.norm()),
                "attn_proj_std": float(blk.attn.proj.weight.std()),
                "c_proj_norm": float(blk.mlp.c_proj.weight.norm()),
                "c_proj_std": float(blk.mlp.c_proj.weight.std()),
            }
        return out

    @torch.no_grad()
    def snapshot(self) -> dict:
        return {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}

    @torch.no_grad()
    def load(self, state: dict) -> None:
        self.load_state_dict(state)


# ---------------------------------------------------------------------------
# 空测试①：生效后的可训参数量（口径 = experiments/additivity/bypass.py#trainable_report）
# ---------------------------------------------------------------------------

def trainable_report(encoder: nn.Module, decoder: nn.Module, branch: BranchSet | None) -> dict:
    """按 requires_grad 实数（不看估算）。

    分支注册为 encoder 的子模块 ⇒ 数编码器时按名字前缀把 `branch.` 排掉，避免双计。
    """
    enc_core = [(n, p) for n, p in encoder.named_parameters() if not n.startswith("branch.")]

    def cnt(pairs, pred):
        return sum(p.numel() for _, p in pairs if pred(p))

    br_pairs = ([(n, p) for n, p in branch.named_parameters()] if branch is not None else [])
    head_pairs = [(n, p) for n, p in decoder.named_parameters()]
    return {
        "base_core_total": sum(p.numel() for _, p in enc_core),
        "base_core_frozen": cnt(enc_core, lambda p: not p.requires_grad),
        "base_core_trainable": cnt(enc_core, lambda p: p.requires_grad),   # 必须为 0
        "head_trainable": cnt(head_pairs, lambda p: p.requires_grad),
        "head_total": sum(p.numel() for _, p in head_pairs),
        "branch_total": sum(p.numel() for _, p in br_pairs),
        "branch_trainable": cnt(br_pairs, lambda p: p.requires_grad),
        "trainable_total": (cnt(enc_core, lambda p: p.requires_grad)
                            + cnt(head_pairs, lambda p: p.requires_grad)
                            + cnt(br_pairs, lambda p: p.requires_grad)),
    }
