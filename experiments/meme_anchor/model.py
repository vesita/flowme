"""M1 三张卡：文本入卡 / 图片入卡 / 共享文本出卡（模因 = [16,128] 固定槽）。

规格见 `PREREG.md` §3。两张输入卡**同构**：`主干 → Transformer encoder（全注意力）→ 16 槽头`，
模态差异只来自前端（char 嵌入 vs CNN）。

图片卡的 16 槽头有**两种可切换**实现（用户 2026-10-11 提出 DETR / YOLO 两条参考）：
  · `detr`：DETR 式 —— 16 个**可学 query** 做「自注意力 + 对特征图交叉注意力」（全局、槽序学出来）
  · `yolo`：YOLO 式 —— 特征图 **自适应池到固定空间网格**（默认 4×4 = 16 格），每格一个槽（空间序）
★ 只借架构，不借 DETR 的集合匹配损失（每个槽没有独立目标）。

形状契约（本试验自己钉，不接 `cards/`）：
    ids [b,L] ──T_in──▶ meme [b,16,128]
    img [b,1,88,584] ──V_in──▶ meme [b,16,128]
    meme [b,16,128] + tgt_in [b,Lt] ──Out──▶ logits [b,Lt,8192]

★ 初始化必须发生在 `torch.manual_seed(seed)` **之后**（`L3_PLAN.md` §9.7）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from data import IMG_SHAPE, PAD, VOCAB

D, FF, NHEAD, SLOTS, DROPOUT, MAXLEN = 128, 512, 4, 16, 0.1, 160   # MAXLEN 160：L3-m2 英文上限 140 字（pe 为非持久 buffer ⇒ 旧 ckpt 不受影响，L≤128 时逐位相同）
YOLO_GRID = (4, 4)                      # 4×4 = 16 槽（与 SLOTS 一致）
POOL = 2                                # ★ 订正 8：进 encoder 前把特征图池化 2×（读字分辨率不变）


def sin_pe(max_len: int, d: int) -> torch.Tensor:
    pos = torch.arange(max_len).unsqueeze(1).float()
    div = torch.exp(torch.arange(0, d, 2).float() * (-math.log(10000.0) / d))
    pe = torch.zeros(max_len, d)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe


def sin_pe_2d(h: int, w: int, d: int) -> torch.Tensor:
    """二维固定正弦位置编码 [h*w, d]：前半维编码行坐标、后半维编码列坐标。"""
    d2 = d // 2
    div = torch.exp(torch.arange(0, d2, 2).float() * (-math.log(10000.0) / d2))
    ii = torch.arange(h).float().unsqueeze(1) * div
    jj = torch.arange(w).float().unsqueeze(1) * div
    pe = torch.zeros(h, w, d)
    pe[:, :, 0:d2:2] = torch.sin(ii).unsqueeze(1)
    pe[:, :, 1:d2:2] = torch.cos(ii).unsqueeze(1)
    pe[:, :, d2::2] = torch.sin(jj).unsqueeze(0)
    pe[:, :, d2 + 1::2] = torch.cos(jj).unsqueeze(0)
    return pe.reshape(h * w, d)


def encoder_layer() -> nn.TransformerEncoderLayer:
    return nn.TransformerEncoderLayer(D, NHEAD, FF, dropout=DROPOUT, activation="gelu",
                                      batch_first=True, norm_first=True)


def decoder_layer() -> nn.TransformerDecoderLayer:
    return nn.TransformerDecoderLayer(D, NHEAD, FF, dropout=DROPOUT, activation="gelu",
                                      batch_first=True, norm_first=True)


class QuerySlots(nn.Module):
    """DETR 式 16 槽头：可学 query 的「自注意力 + 交叉注意力」decoder。"""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.q = nn.Parameter(torch.randn(SLOTS, D) * 0.02)
        self.layers = nn.ModuleList([decoder_layer() for _ in range(n_layers)])
        self.ln = nn.LayerNorm(D)

    def forward(self, memory: torch.Tensor,
                memory_key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        tgt = self.q.unsqueeze(0).expand(memory.size(0), -1, -1)
        for layer in self.layers:
            tgt = layer(tgt, memory, memory_key_padding_mask=memory_key_padding_mask)
        return self.ln(tgt)


class GridSlots(nn.Module):
    """YOLO 式 16 槽头：特征图自适应池到固定空间网格，每格一个槽（无 query、无跨位置注意力）。"""

    def __init__(self, grid: tuple[int, int] = YOLO_GRID):
        super().__init__()
        assert grid[0] * grid[1] == SLOTS, f"[YOLO] 网格 {grid} 的格数 != SLOTS={SLOTS}"
        self.grid = grid
        self.proj = nn.Linear(D, D)
        self.ln = nn.LayerNorm(D)

    def forward(self, feat_map: torch.Tensor) -> torch.Tensor:
        f = F.adaptive_avg_pool2d(feat_map, self.grid)          # [b,D,gh,gw]
        s = f.flatten(2).transpose(1, 2)                        # [b,16,D]（行优先 = 空间序）
        return self.ln(self.proj(s))


def text_encoder() -> tuple[nn.Embedding, nn.ModuleList]:
    return nn.Embedding(VOCAB, D, padding_idx=PAD), nn.ModuleList([encoder_layer() for _ in range(2)])


class TextInputCard(nn.Module):
    """char ids [b,L] → 模因 [b,16,128]。"""

    def __init__(self):
        super().__init__()
        self.emb, self.layers = text_encoder()
        self.register_buffer("pe", sin_pe(MAXLEN, D), persistent=False)
        self.slots = QuerySlots(2)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        L = ids.size(1)
        pad = ids == PAD
        x = self.emb(ids) + self.pe[:L].unsqueeze(0)
        for layer in self.layers:
            x = layer(x, src_key_padding_mask=pad)
        return self.slots(x, memory_key_padding_mask=pad)


class ImageInputCard(nn.Module):
    """灰度渲染图 [b,1,88,584] → 模因 [b,16,128]；`slots_kind` ∈ {detr, yolo}。"""

    def __init__(self, slots_kind: str = "detr"):
        super().__init__()
        assert slots_kind in ("detr", "yolo"), f"[V_in] 未知槽头 {slots_kind!r}"
        self.slots_kind = slots_kind
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2), nn.GroupNorm(4, 32), nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GroupNorm(4, 64), nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.GroupNorm(4, 128), nn.GELU(),
        )
        hh, ww = IMG_SHAPE[0] // 8, IMG_SHAPE[1] // 8            # 16 × 73（订正 4 后画布 128×584）
        assert (hh, ww) == (16, 73), f"[V_in] 下采样网格 {(hh, ww)} 与预期不符"
        # ★ 订正 8：profile 实测图片卡 138.6ms/步（文本卡 2.2ms、解码器 2.8ms）⇒ 热点是 encoder 的
        #   1168-token 全注意力。CNN 仍按 stride 8 算（**读字分辨率不动**），只在进 encoder 前池化 2×
        #   ⇒ token 数 1168 → 288，注意力便宜 ~15×。
        self.gh, self.gw = hh // POOL, ww // POOL                # 8 × 36
        self.register_buffer("pe2d", sin_pe_2d(self.gh, self.gw, D), persistent=False)
        self.layers = nn.ModuleList([encoder_layer() for _ in range(2)])
        self.slots = QuerySlots(2) if slots_kind == "detr" else GridSlots()

    def forward(self, img_u8: torch.Tensor) -> torch.Tensor:
        x = img_u8.float().div_(255.0)                           # [b,1,H,W] ∈ 0..1
        f = self.cnn(x)                                          # [b,D,16,73]
        f = F.adaptive_avg_pool2d(f, (self.gh, self.gw))          # [b,D,8,36] = 288 token（订正 8）
        seq = f.flatten(2).transpose(1, 2) + self.pe2d.unsqueeze(0)
        for layer in self.layers:
            seq = layer(seq)
        if self.slots_kind == "yolo":
            # 订正 9（复核 A15）：yolo 只换**槽头**；encoder + 2D PE 一样要过，否则那 2 层是死参数
            b, n, d = seq.shape
            m = seq.transpose(1, 2).reshape(b, d, self.gh, self.gw)
            return self.slots(F.adaptive_avg_pool2d(m, self.slots.grid))
        return self.slots(seq)


class TextOutputCard(nn.Module):
    """模因 [b,16,128] + 目标前缀 [b,Lt] → logits [b,Lt,8192]（共享输出卡，teacher forcing）。"""

    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB, D, padding_idx=PAD)
        self.register_buffer("pe", sin_pe(MAXLEN, D), persistent=False)
        self.layers = nn.ModuleList([decoder_layer() for _ in range(2)])
        self.head = nn.Linear(D, VOCAB)

    def forward(self, meme: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        Lt = tgt_in.size(1)
        pad = tgt_in == PAD
        t = self.emb(tgt_in) + self.pe[:Lt].unsqueeze(0)
        causal = torch.full((Lt, Lt), float("-inf"), device=tgt_in.device)
        causal = torch.triu(causal, diagonal=1)
        for layer in self.layers:
            t = layer(t, meme, tgt_mask=causal, tgt_key_padding_mask=pad)
        return self.head(t)


class M1Model(nn.Module):
    def __init__(self, vision_arch: str = "detr"):
        super().__init__()
        self.vision_arch = vision_arch
        self.text_in = TextInputCard()
        self.image_in = ImageInputCard(vision_arch)
        self.output = TextOutputCard()

    def meme_text(self, ids: torch.Tensor) -> torch.Tensor:
        return self.text_in(ids)

    def meme_image(self, img_u8: torch.Tensor) -> torch.Tensor:
        return self.image_in(img_u8)

    def logits(self, meme: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        return self.output(meme, tgt_in)

    def n_params(self) -> dict:
        return {n: sum(p.numel() for p in c.parameters())
                for n, c in (("text_in", self.text_in), ("image_in", self.image_in),
                             ("output", self.output))}


def make_pairs(ids_list: list[list[int]], device) -> tuple[torch.Tensor, torch.Tensor]:
    """变长句子 → (decoder 输入, 目标)：输入 = [BOS]+ids，目标 = ids+[EOS]，右侧补 PAD。"""
    seqs = [[BOS] + s + [EOS] for s in ids_list]
    L = max(len(s) for s in seqs)
    tgt_in = torch.full((len(seqs), L), PAD, dtype=torch.long, device=device)
    tgt_out = torch.full((len(seqs), L), PAD, dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        tgt_in[i, :len(s)] = torch.tensor(s, device=device)
        tgt_out[i, :len(s) - 1] = torch.tensor(s[1:], device=device)
    return tgt_in, tgt_out


def build(seed: int, vision_arch: str = "detr") -> M1Model:
    """★ 先播种再构造（`L3_PLAN.md` §9.7）。"""
    torch.manual_seed(seed)
    return M1Model(vision_arch)


from data import BOS, EOS                                                  # noqa: E402

__all__ = ["M1Model", "build", "make_pairs", "sin_pe", "sin_pe_2d", "QuerySlots", "GridSlots",
           "D", "FF", "NHEAD", "SLOTS", "DROPOUT", "MAXLEN", "BOS", "EOS", "PAD", "VOCAB",
           "YOLO_GRID"]
