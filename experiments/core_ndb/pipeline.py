"""核级 NDB 的推理管线：把 CoreNDB 挂到 MultiTaskEngine 的**核编码阶段**（不改 src/）。

三件事，全部在 `experiments/core_ndb/` 本地实现：

1. `_CoreEncoder`：核编码器外面包一层 —— 编码返回前做核级读（或核级写）。
   这是「读写只发生在核编码阶段」的**机械保证**：卡的解码路径一个字没动。
2. `CoreNDBEngine.predict()`：入口按输入 `reset` 表，然后**先做一遍预编码 + 只写**
   （任何卡运行之前），写完关写；随后 `super().predict()` 走原引擎，卡片阶段读到的表是冻结的。
   ⇒ 某张卡跑过、跑几张卡，都不会改变表内容 —— G1 由构造保证，再由实测把关。
3. `card_write` 反面控制：卡解码出锚点后把 `(提及起点, 该卡判的类)` 写进**同一张共享表**。
   这是被 §4 明令禁止的形态，只用于 G2b：证明「共享表 + 卡写入」确实会改掉老卡的输出，
   从而证明 G1 的约束不是空话。

预编码的切段/窗口是**输入侧常量**（`PRE_CHUNK` / `PRE_MAX_LEN`），**不取自挂载卡的 spec**：
否则挂 1 张卡与挂 4 张卡会产生不同的核级写入序列，G1 从构造上就不可能成立。
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from dtseek.encoder.segmenter import split_with_global_offsets  # noqa: E402
from dtseek.tasks.engine import DEFAULT_BASE, MultiTaskEngine  # noqa: E402

from core_memory import CoreNDB  # noqa: E402


class _CoreEncoder(nn.Module):
    """核编码器包装：`h = inner(...)` 之后按当前阶段做核级读 / 核级写。"""

    def __init__(self, inner: nn.Module, engine: "CoreNDBEngine"):
        super().__init__()
        self.inner = inner
        self._engine = engine

    def forward(self, input_ids: torch.Tensor,
                attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        h = self.inner(input_ids, attention_mask=attention_mask)
        eng = self._engine
        mode = eng.core_mode
        if mode == "write":
            eng.core.write(h, input_ids, attention_mask)   # 只写不读
            return h
        if mode == "read":
            return eng.core.read(h, input_ids, attention_mask)
        return h


class CoreNDBEngine(MultiTaskEngine):
    """MultiTaskEngine + 核级情节记忆（读写都在核编码阶段、与挂卡无关）。"""

    #: 输入侧常量（**不取自挂载卡的 spec**，见模块 docstring）
    PRE_CHUNK = 64
    PRE_MAX_LEN = 128

    def __init__(self, base_path: str = DEFAULT_BASE, cards_dir: str | None = None,
                 device: str | None = None, auto_attach: bool = True,
                 memory: bool = True, levels=(1, 2), slots=(8192, 4096),
                 card_write: bool = False, max_table_gb: float = 0.25):
        super().__init__(base_path=base_path, cards_dir=cards_dir,
                         device=device, auto_attach=auto_attach)
        self.core = CoreNDB(hidden_dim=int(self._base["hidden_dim"]),
                            vocab_size=int(self._base["vocab_size"]),
                            levels=levels, slots=slots,
                            max_table_gb=max_table_gb).to(self.device)
        self.core.eval()
        self._inner_encoder = self.doc_encoder
        self.doc_encoder = _CoreEncoder(self.doc_encoder, self)   # 读写挂点
        self.memory = bool(memory)
        self.core_mode = "off"        # off | write | read
        self.card_write = bool(card_write)
        self.pre_segments = 0
        # 钩子常装、按开关生效（关的时候只是一个布尔分支，不改任何数值）
        self._install_card_write_hook()

    # ---- 预编码（核级写，任何卡运行之前）--------------------------------
    @torch.no_grad()
    def _prewrite(self, text: str) -> int:
        """对输入按句切段、逐段核编码并写入核级表；返回段数。调用时表已 reset。"""
        n = 0
        self.core_mode = "write"
        self.core.open_write()
        try:
            for seg in split_with_global_offsets(text, max_chunk_len=self.PRE_CHUNK):
                enc = self.tokenizer.encode(seg["text"], max_length=self.PRE_MAX_LEN,
                                            padding=True)
                inp = torch.tensor([enc["input_ids"]], device=self.device)
                mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool,
                                    device=self.device)
                self.doc_encoder(inp, attention_mask=mask)     # 编码 + 写
                n += 1
        finally:
            self.core.close_write()        # 卡片阶段表冻结
        return n

    @torch.no_grad()
    def predict(self, text: str, tasks: list[str] | None = None,
                max_chunk_len: int | None = None) -> dict:
        text = (text or "").strip()
        if not text:
            return {"error": "输入为空"}
        # 硬约束：表按输入 reset，绝不跨请求残留
        self.core.reset(1, self.device)
        self.pre_segments = self._prewrite(text) if self.memory else 0
        self.core_mode = "read" if self.memory else "off"
        try:
            return super().predict(text, tasks=tasks, max_chunk_len=max_chunk_len)
        finally:
            self.core_mode = "off"

    # ---- 反面控制：让卡往共享表里写 -------------------------------------
    def _install_card_write_hook(self) -> None:
        orig = MultiTaskEngine._run_segment

        def hooked(engine_self, *args, **kwargs):
            # 签名用 *args 透传：src 侧 `_run_segment` 正在被别的会话演进（已多出
            # `encode_cache` 参数），钩子不锁签名，只在返回后追加「卡写共享表」这一步
            anchors = orig(engine_self, *args, **kwargs)
            if getattr(engine_self, "card_write", False) and anchors:
                task = args[0] if args else kwargs["task"]
                segment_text = args[1] if len(args) > 1 else kwargs["segment_text"]
                engine_self._write_card_anchors(task, segment_text, anchors)
            return anchors

        self._run_segment = types.MethodType(hooked, self)   # type: ignore[method-assign]

    @torch.no_grad()
    def _write_card_anchors(self, task: str, segment_text: str,
                            anchors: list[dict]) -> None:
        """卡把自己的锚点（提及起点 + 它判的类）写进共享表 —— §4 禁止的形态，仅供 G2b。"""
        spec = self.specs[task]
        enc = self.tokenizer.encode(segment_text, max_length=spec.max_len, padding=True)
        inp = torch.tensor([enc["input_ids"]], device=self.device)
        mask = torch.tensor([enc["attention_mask"]], dtype=torch.bool, device=self.device)
        prev_mode, self.core_mode = self.core_mode, "off"
        try:
            h = self.doc_encoder(inp, attention_mask=mask)     # 纯编码（不读）
        finally:
            self.core_mode = prev_mode
        D = self.core.hidden_dim
        pos = torch.tensor([[a["local_s0"] for a in anchors]], device=self.device)
        cls = torch.tensor([a["class_id"] for a in anchors], device=self.device)
        proto = F.one_hot(cls % D, D).to(h.dtype)              # 类 → 固定原型向量（无 RNG）
        vals = h.gather(1, pos.unsqueeze(-1).expand(1, len(anchors), D))   # [1, N, D]
        self.core.open_write()
        try:
            self.core.write_points(vals + proto, inp, pos)     # vals 已是 [1, N, D]
        finally:
            self.core.close_write()


# ---- 结果处理（判据口径）-------------------------------------------------
_DROP_KEYS = ("display", "color")


def strip_display(obj):
    """递归剥掉 `display` / `color`：判据比较的是**全部其余字段**。"""
    if isinstance(obj, dict):
        return {k: strip_display(v) for k, v in obj.items() if k not in _DROP_KEYS}
    if isinstance(obj, list):
        return [strip_display(v) for v in obj]
    return obj
