"""Card —— 卡片式模型的抽象基类（框架的唯一形状契约强制点）。

框架契约（写死，不许各卡自己发明）：

    forward(x) 的形状规则由 `Card.io` 声明，且**只有** `Card.forward` 一个强制点：
      - ``ids->d``    ：x = token ids (n,) / (b, n)  →  out [n, d]（输入卡，引入 d 维）
      - ``d->d``      ：x [n, d]  →  out [n, d]（目标卡，**形状保持**是框架硬约束）
      - ``d->logits`` ：x [n, d]  →  out [n, V]（输出卡，消掉 d 维）

n 的语义 = 序列位置维（batch 维在左，逐位置对齐）；模因 [n, d] 在卡间传递时不改形状。

契约版本：任何改变接口语义的改动都必须让 `CONTRACT_VERSION` +1（MV 因此变化）。
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.nn as nn

# 契约版本：接口语义（d / n 语义 / mask / 位置编码 / MAXLEN 的解释方式）的版本号
CONTRACT_VERSION = 1

IO_IDS_TO_D = "ids->d"
IO_D_TO_D = "d->d"
IO_D_TO_LOGITS = "d->logits"

N_SEMANTICS = "n = 序列位置维（batch 维在左）；逐位置对齐，位置 i 的含义由位置编码算入"
MASK_CONVENTION = ("加性 float src_mask (b*nhead, n, n)：上三角(不含对角)=-inf 为因果；"
                   "padding 行整行=-inf；不接受 bool 掩码混用")
PE_CONVENTION = ("固定正弦位置编码：偶数维 sin(pos*div)、奇数维 cos(pos*div)，"
                 "取前 n 行，作为非持久 buffer 不进 state_dict、不训")


@dataclass(frozen=True)
class Contract:
    """接口契约 = d · n 语义 · mask 约定 · 位置编码 · MAXLEN（§7.9②）。"""

    d: int
    n_semantics: str
    mask: str
    pe: str
    maxlen: int
    version: int = CONTRACT_VERSION

    def to_dict(self) -> dict:
        return asdict(self)

    def sha256(self) -> str:
        blob = json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


class ShapeContractError(AssertionError):
    """形状契约被违反。assert 抛的就是它（框架硬约束，必须当场炸）。"""


def tensor_bytes(t: torch.Tensor) -> bytes:
    """张量的确定性字节表示（跨进程/跨次调用逐位一致）。"""
    tt = t.detach().to("cpu").contiguous()
    head = f"{tt.dtype}|{tuple(tt.shape)}|".encode("utf-8")
    try:
        raw = tt.numpy().tobytes()
    except TypeError:            # bfloat16 等 numpy 不认的 dtype
        raw = tt.float().numpy().tobytes()
    return head + raw


def state_sha256(state: dict) -> str:
    """对 state_dict 的键名 + dtype/shape + 原始字节求 sha256（键名排序 ⇒ 确定性）。"""
    h = hashlib.sha256()
    for k in sorted(state):
        h.update(k.encode("utf-8"))
        h.update(tensor_bytes(state[k]))
    return h.hexdigest()


class Card(nn.Module):
    """所有卡片的基类。子类只实现 `apply_card`，形状契约由 `forward` 统一强制。"""

    io: str = IO_D_TO_D
    card_name: str = "card"

    def __init__(self, d: int, maxlen: int, pad_id: int | None = None):
        super().__init__()
        assert isinstance(d, int) and d > 0, f"d 必须是正整数，实得 {d!r}"
        assert isinstance(maxlen, int) and maxlen > 0, f"maxlen 必须是正整数，实得 {maxlen!r}"
        self.d = int(d)
        self.maxlen = int(maxlen)
        self.pad_id = pad_id
        self._is_frozen = False

    # ------------------------------------------------------------------ 契约
    @property
    def contract(self) -> Contract:
        return Contract(d=self.d, n_semantics=N_SEMANTICS, mask=MASK_CONVENTION,
                        pe=PE_CONVENTION, maxlen=self.maxlen)

    @property
    def contract_sha256(self) -> str:
        return self.contract.sha256()

    # ------------------------------------------- forward：唯一的形状强制点
    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        self.assert_input(x)                      # ★契约是门禁：先验输入，再算
        out = self.apply_card(x, mask)
        self.assert_output(x, out)
        return out

    def apply_card(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError(f"{type(self).__name__} 必须实现 apply_card")

    def out_last_dim(self) -> int:
        """输出的最后一维大小（d->d 卡 = d；输出卡 = V）。"""
        return self.d

    def assert_input(self, x: torch.Tensor) -> None:
        if self.io == IO_D_TO_D:
            assert x.dim() >= 2 and x.shape[-1] == self.d, (
                f"[契约] {self.card_name}: 输入最后两维必须是 (n, d={self.d})，"
                f"实得 {tuple(x.shape)}")
        elif self.io == IO_IDS_TO_D:
            assert x.dim() == 2 and not x.is_floating_point(), (
                f"[契约] {self.card_name}: 输入必须是 (b, n) 的 int token ids，实得 "
                f"{tuple(x.shape)} dtype={x.dtype}")
        elif self.io == IO_D_TO_LOGITS:
            assert x.dim() >= 2 and x.shape[-1] == self.d, (
                f"[契约] {self.card_name}: 输入最后两维必须是 (n, d={self.d})，"
                f"实得 {tuple(x.shape)}")
        else:                                     # pragma: no cover - 防呆
            raise AssertionError(f"[契约] 未声明的卡类型 io={self.io!r}")

    def assert_output(self, x: torch.Tensor, out: torch.Tensor) -> None:
        """输出形状契约。d->d 卡在无 batch 时就是逐字的一行断言。"""
        n = int(x.shape[-2])
        if self.io == IO_D_TO_D:
            if x.dim() == 2:
                assert x.shape == out.shape == (n, self.d), (
                    f"[契约] {self.card_name}: 形状保持要求 x.shape == out.shape == (n, d)"
                    f"={(n, self.d)}，实得 x={tuple(x.shape)} out={tuple(out.shape)}")
            else:
                assert out.shape == x.shape, (
                    f"[契约] {self.card_name}: 形状保持要求 out.shape == x.shape（含 batch 维），"
                    f"实得 x={tuple(x.shape)} out={tuple(out.shape)}")
        elif self.io == IO_IDS_TO_D:
            assert out.shape == (*x.shape, self.d), (
                f"[契约] {self.card_name}: 输出必须是 {(*x.shape, self.d)}，"
                f"实得 {tuple(out.shape)}")
        elif self.io == IO_D_TO_LOGITS:
            want = (*x.shape[:-1], self.out_last_dim())
            assert out.shape == want, (
                f"[契约] {self.card_name}: 输出必须是 {want}，实得 {tuple(out.shape)}")

    def assert_contract(self, x: torch.Tensor, out: torch.Tensor) -> None:
        """输入 + 输出的完整形状契约（forward 内分两步调用，这里给整体入口）。"""
        self.assert_input(x)
        self.assert_output(x, out)

    # ------------------------------------------------------------- 参数量
    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------ save/load
    def save(self, path: str | Path) -> Path:
        p = Path(path)
        torch.save({"format": "card-v1", "name": self.card_name,
                    "contract": self.contract.to_dict(), "state": self.state_dict()}, p)
        return p

    def load(self, path: str | Path, map_location: str = "cpu") -> Contract:
        blob = torch.load(Path(path), map_location=map_location, weights_only=False)
        self.load_state_dict(blob["state"])
        got = Contract(**blob["contract"])
        assert got.sha256() == self.contract.sha256(), (
            f"[契约] {self.card_name}: 快照契约与当前卡不一致（假同版本）：\n"
            f"  快照={got.to_dict()}\n  当前={self.contract.to_dict()}")
        return got

    # -------------------------------------------------------- 快照 / 版本
    def weights_sha256(self) -> str:
        return state_sha256(self.state_dict())

    def snapshot_sha256(self) -> str:
        """接口快照 sha256 = hash(契约, 本卡权重)（同权重两次调用必然相同）。"""
        h = hashlib.sha256()
        h.update(b"card-interface-v1|")
        h.update(self.contract_sha256.encode("utf-8"))
        h.update(b"|")
        h.update(self.weights_sha256().encode("utf-8"))
        return h.hexdigest()

    # ------------------------------------------------------------- 软冻结
    def freeze(self) -> "Card":
        """软冻结：requires_grad=False 且 eval（权重不动，但无跨实验快照）。"""
        for p in self.parameters():
            p.requires_grad_(False)
        self.eval()
        self._is_frozen = True
        return self

    def thaw(self) -> "Card":
        for p in self.parameters():
            p.requires_grad_(True)
        self._is_frozen = False
        return self

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    def theta(self) -> dict:
        """当前权重的 CPU 副本（用于「冻结卡逐位不变」的精确比对）。"""
        return {k: v.detach().to("cpu").clone() for k, v in self.state_dict().items()}

    def max_delta(self, before: dict) -> float:
        """max|Δθ|：与 before（theta() 的产物）逐张量取最大绝对差。"""
        cur = self.state_dict()
        assert set(cur) == set(before), "[校验] 参数集合变了，无法比较 Δθ"
        worst = 0.0
        for k in cur:
            d = float((cur[k].detach().to("cpu") - before[k]).abs().max().item()) if cur[k].numel() else 0.0
            worst = max(worst, d)
        return worst

    def __repr__(self) -> str:                             # pragma: no cover
        return (f"<{type(self).__name__} {self.card_name} io={self.io} d={self.d} "
                f"maxlen={self.maxlen} params={self.n_params()} frozen={self._is_frozen}>")
