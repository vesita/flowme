"""核级提及记忆（Core-NDB）—— 把情节记忆从**卡级**上移到**核编码阶段**。

与 `src/dtseek/decoder/mention_ndb.py`（卡级）的对应关系：

    mention_ndb（卡级）                    CoreNDB（核级，本文件）
    ─────────────────────────────────      ─────────────────────────────────────
    键：token n-gram 哈希（同）             键：**同一份代码**（借 `_keys`/`_ensure_flat`）
    值：身份槽 id 的加权计数                值：核隐向量 h 的加权和 + 权重和
    写：解码每步、按预测起点写               写：核编码后、**任何卡运行之前**写
    读：按 start 指针分布聚合               读：按**逐位置 n-gram 槽**聚合
    表：每 batch 一张                       表：**每次输入**一张（predict 入口 reset）
    门控：write/read/level 三个 nn.Parameter 同左，另加**零初始化的 write_scale / read_scale 总闸**

## 形式（本实验的核心，别改）

**读写只在核编码阶段完成，且与「当前挂了哪些卡」无关。**
若由各卡写共享表 ⇒ 加一张卡会改变记忆内容 ⇒ 改变其它卡的读 ⇒
破坏中心不变量「加卡后老卡 pred/exact 逐位不变」（dev-notes §15/§17 的 30/30）。

因此本模块的调用点只有两处，都在核编码阶段：

1. `write()`：`CoreNDBEngine.predict()` 在**任何卡运行之前**做一遍「预编码 + 只写」，
   写完立刻关写 —— 卡片阶段表**冻结**，各卡读到的表内容与挂了几张卡无关；
2. `read()`：核编码器外面那层包装（`_CoreEncoder`）在编码返回前调用，
   得到的 doc_memory 交给各卡解码（各卡读到的是**同一份**结果）。

## 三条硬约束（沿用 mention_ndb）

1. **表按输入 reset**：`predict()` 入口 `reset(1, device)`，绝不跨请求残留。
2. **表是 no_grad 数据统计量**：不注册 buffer、不进 state_dict；只有门控是 nn.Parameter。
3. **零初始化总闸 ⇒ 起点贡献恰为 0**：`write_scale = read_scale = 0`（标量 Parameter），
   于是 `w ≡ 0`（表恒空）且 `g ≡ 0`（读恒直通）⇒ 与「完全不接记忆」逐位一致（G0 的实现点）。
   门控线性层本身沿用 mention_ndb 的 bias 口径（write `+1`、read `−2`），
   阶段 2 训练时由总闸先打开、线性层再学会**按位置**选写点（§4 的「mention 门控头」）。

## 聚合为什么用 one-hot × bmm 而不是 index_add_

`index_add_` 在 GPU 上走 atomics ⇒ 同一槽位的累加顺序每次可能不同 ⇒ fp32 末位会抖，
而本实验的判据是**逐位**比较。one-hot × `bmm` 是固定归约顺序的 GEMM，同形状同输入复跑同结果
（由 `run_phase1.py` 的 R0 重复性前提实测把关）。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from dtseek.decoder.mention_ndb import MentionNDB

#: 与 mention_ndb 同源的归一化分母（`log1p(100)`）
_LOG1P_100 = math.log1p(100.0)


class _Keyer:
    """借 mention_ndb 的键机制：`_keys` / `_ensure_flat` 直接绑定它的方法，不复制代码。"""

    def __init__(self, levels, slots):
        self.levels = tuple(int(x) for x in levels)
        if isinstance(slots, int):
            slots = [slots] * len(self.levels)
        self.slots = [int(s) for s in slots]
        if len(self.slots) != len(self.levels):
            raise ValueError("slots 与 levels 长度必须一致")
        offs, acc = [], 0
        for s in self.slots:
            offs.append(acc)
            acc += s
        self._offsets = offs
        self._total_slots = acc
        self.n_levels = len(self.levels)
        self._flat: torch.Tensor | None = None
        self._flat_src: torch.Tensor | None = None


# 同一份键代码（多项式滚动哈希 + 文档级扁平缓存），绑定到本类上
_Keyer._keys = MentionNDB._keys                 # type: ignore[attr-defined]
_Keyer._ensure_flat = MentionNDB._ensure_flat   # type: ignore[attr-defined]


class CoreNDB(nn.Module):
    """核级情节记忆：no_grad 的「n-gram 槽 → 核隐向量」表 + 零初始化读写总闸。"""

    def __init__(self, hidden_dim: int, vocab_size: int,
                 levels=(1, 2), slots=(8192, 4096), max_table_gb: float = 0.25):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.vocab_size = int(vocab_size)
        self.max_table_gb = float(max_table_gb)
        self.n_stats = 3
        self.keyer = _Keyer(levels, slots)
        self.levels = self.keyer.levels
        self.slots = self.keyer.slots
        self._total_slots = self.keyer._total_slots

        # ---- 门控（只有这些是参数）；bias 初始化与 mention_ndb 同口径 ----
        self.write_gate = nn.Linear(self.hidden_dim, 1)
        nn.init.zeros_(self.write_gate.weight)
        nn.init.constant_(self.write_gate.bias, 1.0)       # σ(1)≈0.73：起点均匀地写
        self.read_gate = nn.Linear(self.hidden_dim + self.n_stats + 1, 1)
        nn.init.zeros_(self.read_gate.weight)
        nn.init.constant_(self.read_gate.bias, -2.0)       # σ(−2)≈0.12：起点先轻信
        self.level_weight = nn.Parameter(torch.zeros(len(self.levels)))
        # 零初始化总闸：起点贡献恰为 0（G0 / G2a′ 的实现点）
        self.write_scale = nn.Parameter(torch.zeros(()))
        self.read_scale = nn.Parameter(torch.zeros(()))

        # ---- 表（故意不注册 buffer）：no_grad、不进优化器、不进 state_dict ----
        self._tab_cnt: torch.Tensor | None = None          # [B, Σslots, D]
        self._tab_tot: torch.Tensor | None = None          # [B, Σslots]
        self._batch = 0
        self._write_on = False
        # 诊断量（设备标量，stats() 里才物化，避免每步同步）
        self._w_sum_t: torch.Tensor | None = None          # 写入总权重（S1 必须恒 0）
        self._n_pos_t: torch.Tensor | None = None          # 参与写的位置数
        self._delta_l1_t: torch.Tensor | None = None       # 读带来的 |out−h| 之和（S1 必须恒 0）
        self._g_mean_t: torch.Tensor | None = None
        self._mass_mean_t: torch.Tensor | None = None
        self._read_n: int = 0                       # read() 被调次数（均值分母）

    # ==================================================================
    # 表生命周期（硬约束：按输入 reset，绝不跨请求残留）
    # ==================================================================
    def table_bytes(self, batch_size: int) -> int:
        return int(batch_size) * (self.hidden_dim + 1) * self._total_slots * 4

    def table_gb(self, batch_size: int | None = None) -> float:
        return self.table_bytes(batch_size or self._batch) / 2**30

    def reset(self, batch_size: int, device) -> None:
        need = self.table_bytes(batch_size) / 2**30
        if need > self.max_table_gb:
            raise ValueError(
                f"Core-NDB 表需要 {need:.3f}GB（batch={batch_size}, slots={self.slots}, "
                f"D={self.hidden_dim}），超过上限 {self.max_table_gb}GB。"
                " 缩小 slots / batch，或提高 max_table_gb。")
        self._tab_cnt = torch.zeros(batch_size, self._total_slots, self.hidden_dim, device=device)
        self._tab_tot = torch.zeros(batch_size, self._total_slots, device=device)
        self._batch = int(batch_size)
        self.keyer._flat = None
        self.keyer._flat_src = None
        self._w_sum_t = self._n_pos_t = None
        self._delta_l1_t = self._g_mean_t = self._mass_mean_t = None
        self._read_n = 0

    @torch.no_grad()
    def reset_stats(self) -> None:
        self._w_sum_t = self._n_pos_t = None
        self._delta_l1_t = self._g_mean_t = self._mass_mean_t = None
        self._read_n = 0

    @property
    def table_ready(self) -> bool:
        return self._tab_cnt is not None

    # 写开关（卡片阶段表冻结 —— 这是「与挂了哪些卡无关」的机械保证）
    def open_write(self) -> None:
        self._write_on = True

    def close_write(self) -> None:
        self._write_on = False

    @property
    def write_active(self) -> bool:
        return self._write_on

    # ==================================================================
    # 写
    # ==================================================================
    def _scatter(self, idx: torch.Tensor, values: torch.Tensor, weight: torch.Tensor) -> None:
        """把 `[B, P]` 的扁平槽位、`[B, P, D]` 的值、`[B, P]` 的权重聚合成表增量。

        one-hot × bmm（固定归约顺序，无 atomics）—— 见模块 docstring。
        """
        B, P = idx.shape
        S, D = self._total_slots, self.hidden_dim
        oh = F.one_hot(idx, S).to(values.dtype)                    # [B, P, S]
        wv = (weight.unsqueeze(-1) * values).reshape(B, P, D)      # [B, P, D]
        self._tab_cnt += torch.bmm(oh.transpose(1, 2), wv)         # [B, S, D]
        self._tab_tot += torch.bmm(oh.transpose(1, 2),
                                   weight.reshape(B, P, 1)).squeeze(-1)

    @torch.no_grad()
    def write(self, h: torch.Tensor, input_ids: torch.Tensor,
              mask: torch.Tensor | None = None) -> None:
        """核编码后**逐位置**写：键 = 该处起始的 token n-gram，值 = 核隐 h。

        权重 `w = σ(write_gate(h)) · clamp(write_scale, 0)`：零初始化时恒为 +0.0
        ⇒ 表保持全 0（`tot` 恒 0 ⇒ 读侧 `covered` 恒 False）。
        """
        if not self._write_on:
            return None
        if self._tab_cnt is None:
            raise RuntimeError("write() 前必须先 reset(batch_size, device)")
        B, L, _ = h.shape
        flat = self.keyer._ensure_flat(input_ids)                  # [B, L, nlev]
        P = L * self.keyer.n_levels
        idx = flat.reshape(B, P)
        scale = torch.clamp(self.write_scale, min=0.0)
        w = torch.sigmoid(self.write_gate(h)).squeeze(-1) * scale    # [B, L]
        if mask is not None:
            w = w * mask.to(w.dtype)
        with torch.no_grad():
            self._w_sum_t = (w.sum() if self._w_sum_t is None else self._w_sum_t + w.sum())
            n_pos = (mask.sum() if mask is not None else torch.tensor(L, device=h.device))
            self._n_pos_t = (n_pos if self._n_pos_t is None else self._n_pos_t + n_pos)
        # 每个位置在每一级写同一个权重/值
        w_lv = w.unsqueeze(-1).expand(B, L, self.keyer.n_levels).reshape(B, P)
        vals = h.unsqueeze(2).expand(B, L, self.keyer.n_levels, self.hidden_dim) \
                 .reshape(B, P, self.hidden_dim)
        self._scatter(idx, vals, w_lv)
        return None

    @torch.no_grad()
    def write_points(self, h_vals: torch.Tensor, input_ids: torch.Tensor,
                     starts: torch.Tensor) -> None:
        """**反面控制专用**：由卡在指定位置写入（起点 `starts[B, N]`，值 `h_vals[B, N, D]`）。

        正式管线绝不用它 —— 它正是被禁止的「共享表 + 由卡写入」形态。
        """
        if not self._write_on:
            return None
        if self._tab_cnt is None:
            raise RuntimeError("write_points() 前必须先 reset(batch_size, device)")
        B, N, _ = h_vals.shape
        L = input_ids.shape[1]
        flat = self.keyer._ensure_flat(input_ids)                  # [B, L, nlev]
        pos = starts.clamp(0, L - 1).view(B, N, 1).expand(B, N, self.keyer.n_levels)
        idx = flat.gather(1, pos).reshape(B, N * self.keyer.n_levels)
        scale = torch.clamp(self.write_scale, min=0.0)
        w = torch.sigmoid(self.write_gate(h_vals)).squeeze(-1) * scale   # [B, N]
        P2 = N * self.keyer.n_levels
        w = w.unsqueeze(-1).expand(B, N, self.keyer.n_levels).reshape(B, P2)
        vals = h_vals.unsqueeze(2).expand(B, N, self.keyer.n_levels, self.hidden_dim) \
                     .reshape(B, P2, self.hidden_dim)
        self._scatter(idx, vals, w)
        with torch.no_grad():
            self._w_sum_t = (w.sum() if self._w_sum_t is None else self._w_sum_t + w.sum())
        return None

    # ==================================================================
    # 读
    # ==================================================================
    def read(self, h: torch.Tensor, input_ids: torch.Tensor,
             mask: torch.Tensor | None = None) -> torch.Tensor:
        """按逐位置 n-gram 槽查表，把「同槽此前写入的核隐」按门控混回核输出。

        `out = h + mask · g · mass · (v − h)`
        - `v`   ：覆盖到的各级 `cnt/tot` 的 `α`-加权平均（未覆盖的级不参与，再按覆盖质量归一）
        - `mass`：`Σ_li α_li·covered_li`（没有任何级覆盖时恰为 0）
        - `g`   ：`σ(read_gate([h, feat, w_t])) · clamp(read_scale, 0, 1)`

        零初始化时 `g ≡ +0.0` ⇒ `out = h + mask·0·… = h` 逐位成立（G0 的实现点）；
        表为空时 `mass ≡ 0` ⇒ 即使 `read_scale=1` 也逐位直通（G2a′ 的实现点）。
        """
        if self._tab_cnt is None:
            raise RuntimeError("read() 前必须先 reset(batch_size, device)")
        B, L, D = h.shape
        nlev = self.keyer.n_levels
        alpha = torch.softmax(self.level_weight, dim=0)               # [nlev]
        flat = self.keyer._ensure_flat(input_ids)                     # [B, L, nlev]
        P = L * nlev
        idx = flat.reshape(B, P)
        cnt = self._tab_cnt.gather(1, idx.unsqueeze(-1).expand(B, P, D)) \
                 .reshape(B, L, nlev, D)
        tot = self._tab_tot.gather(1, idx).reshape(B, L, nlev)
        cov = (tot > 0).to(h.dtype)                                   # [B, L, nlev]
        frac = cnt / tot.clamp_min(1e-6).unsqueeze(-1)                # [B, L, nlev, D]
        wl = alpha.view(1, 1, nlev) * cov                             # [B, L, nlev]
        denom = wl.sum(-1, keepdim=True).clamp_min(1e-9)              # [B, L, 1]
        v = (wl.unsqueeze(-1) * frac).sum(2) / denom                  # [B, L, D]
        mass = wl.sum(-1, keepdim=True)                               # [B, L, 1]

        # 与 mention_ndb 同口径的诊断特征：log1p(总字数) / top1 占比 / 覆盖质量
        tot_agg = (wl * tot).sum(-1)                                  # [B, L]
        f_total = torch.log1p(tot_agg) / _LOG1P_100
        f_share = v.amax(-1)
        feat = torch.stack([f_total, f_share, mass.squeeze(-1)], dim=-1)   # [B, L, 3]
        # w_t 喂进读门控 = **实际写进去的那份权重**（`σ·clamp(write_scale)`）：
        # 表本身在 no_grad 下是常量，梯度进不了 write_scale —— 照 mention_ndb 的做法，
        # 把写门控的值同时喂进读门控，write_scale 才有梯度路径（阶段 2 训练要用）。
        # 在阶段 1 的尺度取值（0 或 1）下 `clamp(s,0) == s`，这条改动对 G0/G1/G2 恒等。
        w_t = (torch.sigmoid(self.write_gate(h)).squeeze(-1)
               * torch.clamp(self.write_scale, min=0.0))                  # [B, L]

        g = torch.sigmoid(self.read_gate(torch.cat([h, feat, w_t.unsqueeze(-1)], dim=-1)))
        g = g * torch.clamp(self.read_scale, min=0.0, max=1.0)                  # [B, L, 1]
        delta = g * mass * (v - h)
        if mask is not None:
            delta = delta * mask.to(delta.dtype).unsqueeze(-1)
        out = h + delta

        with torch.no_grad():
            self._g_mean_t = (g.mean() if self._g_mean_t is None else self._g_mean_t + g.mean())
            self._mass_mean_t = (mass.mean() if self._mass_mean_t is None
                                 else self._mass_mean_t + mass.mean())
            dl = delta.abs().sum()
            self._delta_l1_t = (dl if self._delta_l1_t is None else self._delta_l1_t + dl)
            self._read_n += 1
        return out

    # ==================================================================
    # 监控
    # ==================================================================
    @staticmethod
    def _f(t: torch.Tensor | None) -> float:
        return float("nan") if t is None else float(t.item())

    @staticmethod
    def _i(t: torch.Tensor | None) -> int:
        return 0 if t is None else int(t.item())

    def stats(self) -> dict:
        return {
            "table_gb": self.table_gb(),
            "batch": self._batch,
            "write_active": self._write_on,
            "write_scale": float(self.write_scale.detach()),
            "read_scale": float(self.read_scale.detach()),
            "w_sum": self._f(self._w_sum_t),
            "n_write_pos": self._i(self._n_pos_t),
            "delta_l1": self._f(self._delta_l1_t),
            # g / mass 是**每次 read 调用的均值再对调用次数取均值**（跨段可比）
            "g_mean": (self._f(self._g_mean_t) / self._read_n) if self._read_n else float("nan"),
            "mass_mean": (self._f(self._mass_mean_t) / self._read_n
                          if self._read_n else float("nan")),
            "n_read": self._read_n,
            "n_slot_touched": (int((self._tab_tot > 0).sum().item())
                               if self._tab_tot is not None else 0),
            "alpha": torch.softmax(self.level_weight, 0).detach().cpu().tolist(),
        }

    def extra_repr(self) -> str:
        return (f"levels={self.levels} slots={self.slots} D={self.hidden_dim} "
                f"table(batch={self._batch})={self.table_gb():.3f}GB "
                f"write_scale={float(self.write_scale.detach())} "
                f"read_scale={float(self.read_scale.detach())}")
