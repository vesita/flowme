"""harness.pipeline —— 按「卡片集合 + 拓扑」装配**可推理**流水线（顺序执行；卡间接口 = d 维隐态）。

本模块是 harness 的 canonical core，三件事都在这里、且**可复用**（不是脚本内联）：

  ① **接口契约的唯一实现点**：卡间只传 `Packet`（`h` [b,n,d] + `mask` + `ctx_len` + `aux`）；
  ② **模型构造走 `cards/` 的正规路径**：`cards.InputCard` / `cards.OutputCard` /
     `cards.base.encoder_layer` / `cards.TargetCard(residual=False)` —— 本文件不自己发明层；
  ③ **载 ckpt / 贪心生成 / 评测口径**：与 `stages/79_cand_cards_v2.py` 出记录值时**同一条代码路径**
     （`prep_record` / `pack` / `greedy_gen` / `eval_sel` / `em_strict` / `dec_letter` 的语义逐条对齐）。

★★ 两条硬纪律（本文件的实现点）
  · **先 `manual_seed` 再构造**：`build_pipeline()` 是唯一构造入口，它在建卡之前先播种
    （§9.7 的缺陷就是"构造发生在播种之前"）；构造完打印参数哈希供 `[INIT-CHECK]` 比对。
  · **接口张量不含位置**：位置只由序列轴 + `mask`（因果上三角 + padding）承载；
    唯一的绝对位置编码在 `InputCard` 内部（固定正弦），**不在卡间传递**。

用法（最小例子）
    from harness.pipeline import build_pipeline, load_ckpt, greedy_gen
    pipe = build_pipeline("B2", seed=1234, device="cpu")   # 顺序：input → select → realign → output
    load_ckpt(pipe, "logs/79_ckpt_B2_k8_s1234.pt")
    print(greedy_gen(pipe, [rec])[0])
"""
from __future__ import annotations

import hashlib
import importlib.util
import math
import os
import struct
import sys
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Sequence

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NS_ROOT = "/home/vesita/coding/my/nanoSeek"
for _p in (ROOT, NS_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from cards import InputCard, OutputCard, TargetCard, encoder_layer  # noqa: E402
from cards.base import Card, tensor_bytes  # noqa: E402

# ---------------------------------------------------------------------------
# 配置（与 S65/S67/S71/S73/S79 一致；改这里等于改所有卡的形状契约）
# ---------------------------------------------------------------------------
D = 128                 # ★接口宽度 w 的默认值（= 卡间隐态维度）
FF = 512
NHEAD = 4
MAXLEN = 512
DROPOUT = 0.1
SEL_LAMBDA = 1.0        # B2 选择 CE 权重（写死，§11.11）

EOS = "<eos>"
CAND_PREFIX = "候选："
CAND_SEP = "；"
CAND_LETTERS = "ABCDEFGH"
CAND_ITEM = "{letter}. {ent} / {cond}"
LOG_DIR = os.path.join(ROOT, "logs")
TOK_PATH = os.path.join(NS_ROOT, "data", "chinese", "char_tokenizer.json")
STAGES = os.path.join(ROOT, "stages")

CKPT_B2_K8_S1234 = os.path.join(LOG_DIR, "79_ckpt_B2_k8_s1234.pt")
CKPT_B2_K8_S5678 = os.path.join(LOG_DIR, "79_ckpt_B2_k8_s5678.pt")
DATA_TEST_K8 = os.path.join(LOG_DIR, "76_dialog_test.jsonl")
DATA_TRAIN_K8 = os.path.join(LOG_DIR, "76_dialog_train.jsonl")

# 记录值（L3_PLAN §11.26 / §11.28 / §11.24；B2 s1234 与 A s1234，k=8 带字母数据）
RECORDS = {
    "B2": {"t2": 75.88, "t1": 80.00, "sel": 88.38,
           "src": "L3_PLAN.md §11.26/§11.28（B2 s1234，k=8 带字母 76_*）"},
    "A": {"t2": 12.38, "t1": 12.62, "sel": None,
          "src": "L3_PLAN.md §11.24/§11.26（A s1234，k=8 带字母 76_*）"},
}

_TOK = None
_V = 0
_PAD_ID = 0
_EOS_ID = 0
_UNK_ID = 0


# ---------------------------------------------------------------------------
# 接口契约（H-e 的机器可读形式；README 是它的散文版）
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Interface:
    """一张卡对外的接口声明。

    d_in / d_out   接口宽度（本框架里恒为 d=128；`probe --interface w` 会改它）
    injection      "none" = 只读上游 h；"add" = 在回复区做**加性注入**（重整卡）
    emits          本卡往 `Packet.aux` 里放的张量名（下游可读；不是接口的一部分）
    carries_position  ★ 恒为 False：位置不塞进 h，只由序列轴 + mask + ctx_len 承载
    """
    d_in: int
    d_out: int
    injection: str = "none"
    emits: tuple = ()
    carries_position: bool = False
    doc: str = ""


@dataclass(frozen=True)
class JudgedTarget:
    """一张卡声明的**可判中间目标**（§18.2 方向 IV：没有它，加卡只是加延迟）。"""
    name: str
    metric: str
    needs: tuple          # 需要记录里哪些字段才能算这个目标
    chance: str = ""      # 机会水平（写清，避免把机会级当增益）
    doc: str = ""


@dataclass
class Packet:
    """★卡间唯一载荷（顺序执行的流水线里逐卡传递，就地更新）。

    tokens [b,n] long      仅输入卡读
    mask   [b*nhead,n,n]   因果 + padding（由输入卡的 causal_mask 产生，逐卡透传）
    h      [b,n,d]         卡间隐态（= "接口"）
    ctx_len [b] long       回复区起点（可判目标 / 加性注入的分界）
    aux    dict            各卡 emit 的附加张量（选择卡 → sel_scores/sel_probs/e_sel）
    logits [b,n,V]         输出卡写
    """
    tokens: torch.Tensor | None = None
    mask: torch.Tensor | None = None
    h: torch.Tensor | None = None
    ctx_len: torch.Tensor | None = None
    aux: dict = field(default_factory=dict)
    logits: torch.Tensor | None = None


class Meter:
    """活张量计量（**确定性**：只看 shape/dtype，不看时间与分配器）。

    口径（写死，别混）：在每个卡边界上遍历**实际存在的张量对象**，把
      `常驻参数字节`（全部卡的 parameters+buffers）与 `包内活载荷字节`（packet 里的张量）
    相加，取全流程最大值。它不是 `torch.cuda.max_memory_allocated`（CPU 无该口径），
    也不含框架内部临时量 ⇒ 是**下界**。GPU 档才有真分配器读数（见 probe）。
    """

    def __init__(self) -> None:
        self.stages: list = []
        self.peak_live = 0
        self.peak_stage = None

    def stage(self, iid: str, card: nn.Module, pkt: Packet, resident_params: int) -> None:
        payload = 0
        for t in (pkt.tokens, pkt.mask, pkt.h, pkt.logits):
            if isinstance(t, torch.Tensor):
                payload += t.numel() * t.element_size()
        for t in pkt.aux.values():
            if isinstance(t, torch.Tensor):
                payload += t.numel() * t.element_size()
        card_p = param_bytes(card)
        live = resident_params + payload
        self.stages.append({"stage": iid, "card_params_bytes": card_p,
                            "payload_bytes": payload, "live_bytes": live})
        if live > self.peak_live:
            self.peak_live, self.peak_stage = live, iid


def param_bytes(mod: nn.Module) -> int:
    """模块实测字节数（parameters + buffers，遍历实际张量对象）。"""
    n = 0
    for p in mod.parameters():
        n += p.numel() * p.element_size()
    for b in mod.buffers():
        n += b.numel() * b.element_size()
    return n


# ---------------------------------------------------------------------------
# 分词器 / 记录预处理（口径 = stages/79_cand_cards_v2.py）
# ---------------------------------------------------------------------------
def load_tok(path: str = TOK_PATH):
    """载入字符分词器并设全局 V/PAD/EOS（只做一次）。"""
    global _TOK, _V, _PAD_ID, _EOS_ID, _UNK_ID
    if _TOK is not None:
        return _TOK
    from tokenizers import Tokenizer
    _TOK = Tokenizer.from_file(path)
    _V = _TOK.get_vocab_size()
    _PAD_ID = _TOK.token_to_id("<pad>")
    _EOS_ID = _TOK.token_to_id("<eos>")
    _UNK_ID = _TOK.token_to_id("<unk>")
    return _TOK


def vocab_size() -> int:
    load_tok()
    return _V


def eos_id() -> int:
    load_tok()
    return _EOS_ID


def pad_id() -> int:
    load_tok()
    return _PAD_ID


def enc(text: str) -> list:
    load_tok()
    return _TOK.encode(text, add_special_tokens=False).ids


def dec(ids) -> str:
    load_tok()
    return _TOK.decode([int(i) for i in ids], skip_special_tokens=True)


def load_jsonl(path: str) -> list:
    import json
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


POS_BAD = {"n": 0}


def prep_record(r: dict, with_target: bool = True) -> dict:
    """文本 → token：p=enc(ctx)、t=enc(target+<eos>)、cend=每个候选**末位 token** 的位置。

    逐字复刻 `stages/79_cand_cards_v2.py::prep_record`（含"前缀编码稳定"核验）。
    """
    r["p"] = enc(r["ctx"])
    if with_target and "t" not in r:
        r["t"] = enc(r["target"] + EOS)
    base = r["ctx"].index(CAND_PREFIX)
    ends = []
    for c in r["cands"]:
        pos = r["ctx"].index(c["text"], base)
        pref = enc(r["ctx"][: pos + len(c["text"])])
        if r["p"][: len(pref)] != pref:
            POS_BAD["n"] += 1
        ends.append(len(pref) - 1)
    r["cend"] = ends
    r["n_cands"] = len(r["cands"])
    return r


def pack(recs: Sequence[dict], with_target: bool = True, device=None):
    """(ids, ctx_len, end_idx, n_cands)；ctx 在左、回复在右，右 padding。口径 = 79::pack。"""
    dev = device if device is not None else torch.device("cpu")
    K = max(r["n_cands"] for r in recs)
    lens = [len(r["p"]) + (len(r["t"]) if with_target else 0) for r in recs]
    n = max(lens)
    ids = torch.full((len(recs), n), pad_id(), dtype=torch.long)
    s = torch.zeros(len(recs), dtype=torch.long)
    eidx = torch.zeros((len(recs), K), dtype=torch.long)
    nc = torch.zeros(len(recs), dtype=torch.long)
    for i, r in enumerate(recs):
        ids[i, : len(r["p"])] = torch.tensor(r["p"])
        if with_target:
            ids[i, len(r["p"]): len(r["p"]) + len(r["t"])] = torch.tensor(r["t"])
        s[i] = len(r["p"])
        for j, e in enumerate(r["cend"]):
            eidx[i, j] = e
        nc[i] = r["n_cands"]
    return ids.to(dev), s.to(dev), eidx.to(dev), nc.to(dev)


def data_fp(recs: Sequence[dict]) -> str:
    """数据指纹（与 `stages/72::data_fp` 逐字一致；selfcheck 里会与 72 的定义对拍）。"""
    h = hashlib.sha256()
    for r in recs:
        for seq in (r["p"], r["t"]):
            h.update(struct.pack("<Q", len(seq)))
            if seq:
                h.update(struct.pack(f"<{len(seq)}i", *seq))
    return h.hexdigest()[:16]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_digest(t: torch.Tensor) -> str:
    """张量的确定性指纹（口径 = cards.base.tensor_bytes）。"""
    return hashlib.sha256(tensor_bytes(t)).hexdigest()[:16]


def state_digest(mod: nn.Module) -> str:
    """参数哈希（[INIT-CHECK] 用：同 seed 两次构造必须相同）。"""
    h = hashlib.sha256()
    for k, v in sorted(mod.state_dict().items()):
        h.update(k.encode())
        h.update(tensor_bytes(v))
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# 卡片基类
# ---------------------------------------------------------------------------
class HarnessCardMixin:
    """harness 卡片的统一接口（mixin：让 `cards/` 里已有的卡**原样**变成 harness 卡）。

    **新卡只需**：声明下面 5 个类属性、实现 `forward_card(pkt)`。

    spec_name    注册表里的名字（`registry.py` 用它登记）
    interface    Interface（接口宽度/注入方式/emit 的张量）
    deps         依赖声明：本卡读 Packet 里的哪些东西（"tokens"/"h"/"mask"/"ctx_len"/"aux:e_sel"…）
    judged       可判中间目标（None = 这张卡没有可判目标 ⇒ 按 §18.1 它只是延迟）
    ckpt_prefix  ckpt 里属于本卡的 key 前缀（None = 不参与 ckpt 加载）
    ckpt_root_keys  ckpt 里属于本卡但**不带前缀**的 key（如 79 的选择卡打分向量 `q`）
    slot_in/slot_out 拓扑声明：读哪个接口槽、写不写主接口（`None` = 侧支）

    ★ 拓扑不是"一条链"：**选择卡是侧支** —— 它读输入卡的隐态、只把 `e_sel` 放进 `aux`，
    主接口上的隐态仍是**输入卡的输出**；重整卡读的正是它（`h_input + 注入`）⇒ 见 `Pipeline.topology()`。
    """
    spec_name = "card"
    interface = Interface(0, 0)
    deps: tuple = ()
    judged: JudgedTarget | None = None
    ckpt_prefix: str | None = None
    ckpt_root_keys: tuple = ()
    slot_in: str = "h"                  # 读哪个接口槽（本框架里恒为 "h"：卡间只有一个 d 维隐态）
    slot_out: str | None = "h"          # "h" = 写主接口；None = **侧支**（只 emit aux，不改主接口）

    def forward_card(self, pkt: Packet) -> Packet:      # pragma: no cover - 抽象
        raise NotImplementedError(f"{type(self).__name__} 必须实现 forward_card(pkt)")

    def param_bytes(self) -> int:
        return param_bytes(self)


class HarnessCard(HarnessCardMixin, nn.Module):
    """纯 harness 卡（== `cards/` 里没有现成实现的那种）的基类；用法同 `HarnessCardMixin`。"""


class InputHarnessCard(InputCard, HarnessCardMixin):
    """输入卡：token ids → 隐态（emb + 固定正弦 PE + 1 层 encoder，causal+padding mask）。

    ★它就是 `cards.InputCard` 本体（harness 卡 = cards/ 的卡），只是多一个 `forward_card` 入口，
    所以 ckpt 里的 `input.*` 可以直接映射到它的 `emb.* / enc.*`。
    """
    spec_name = "input"
    interface = Interface(d_in=0, d_out=D,
                          doc="token ids → h[b,n,d]；★唯一施加绝对位置编码的地方")
    deps = ("tokens",)
    ckpt_prefix = "input."

    def __init__(self, d: int = D, vocab_size: int = 8192, ff: int = FF, nhead: int = NHEAD,
                 maxlen: int = MAXLEN, pad: int | None = None, dropout: float = DROPOUT):
        # ★pad 必须来自分词器（实测 <pad>=131，不是 0）；=None 时自动取
        super().__init__(d, vocab_size, ff, nhead, maxlen, pad_id() if pad is None else pad, dropout)

    def forward_card(self, pkt: Packet) -> Packet:
        m = self.causal_mask(pkt.tokens)
        w = self.emb.weight
        if m.dtype != w.dtype:                       # fp16/bf16 档：mask 跟随参数 dtype
            m = m.to(w.dtype)
        pkt.mask = m
        pkt.h = self(pkt.tokens, mask=m)
        return pkt


class EncLayerHarnessCard(HarnessCard):
    """通用 encoder 层卡：`h = enc_layer(h, mask)`（= `cards.TargetCard(residual=False)` 的语义）。"""
    spec_name = "enc_layer"
    interface = Interface(d_in=D, d_out=D, doc="d→d 裸 encoder 层（无外层残差）")
    deps = ("h", "mask")
    ckpt_prefix = None

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, dropout: float = DROPOUT):
        super().__init__()
        self.enc = encoder_layer(d, ff, nhead, dropout)

    def forward_card(self, pkt: Packet) -> Packet:
        pkt.h = self.enc(pkt.h, src_mask=pkt.mask)
        return pkt

    def topology_note(self) -> str:
        return f"{self.slot_in} → {self.slot_out or 'aux'}"


class TargetHarnessCard(TargetCard, HarnessCardMixin):
    """`cards.TargetCard(residual=False)` 原样接入（★模型构造的正规路径的**字面**证据）。

    与 `EncLayerHarnessCard` 数值等价（都是裸层），但它用的是 `cards.target.TargetCard` 本体
    ⇒ `native3` 卡组 = `cards.CardPipeline(residual=False)` 的同一条参数路径。
    """
    spec_name = "target"
    interface = Interface(d_in=D, d_out=D, doc="cards.TargetCard(residual=False)：d→d 裸层")
    deps = ("h", "mask")
    ckpt_prefix = "target."

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT):
        super().__init__(d, ff, nhead, maxlen, dropout, residual=False)

    def forward_card(self, pkt: Packet) -> Packet:
        pkt.h = self(pkt.h, mask=pkt.mask)
        return pkt


class SelectHarnessCard(EncLayerHarnessCard):
    """选择卡：取候选**末位 token** 的隐态 h_j，打分 s_j = q·h_j（q 零初始化、128 参数），
    softmax → p，soft 选择向量 e_sel = Σ_j p_j h_j（**加性注入**到回复区，见重整卡）。

    可判中间目标 = 「在候选层里挑出与历史 (实体,条件) 一致的那一项」（字母准确率，机会水平 1/k）。
    """
    spec_name = "select"
    interface = Interface(d_in=D, d_out=D, injection="none",
                          emits=("select_hidden", "cand_hidden", "sel_scores", "sel_probs", "e_sel"),
                          doc="读 h + 候选末位下标；emit 选择打分/概率/e_sel")
    deps = ("h", "mask", "aux:cand_end", "aux:n_cands")
    judged = JudgedTarget(
        name="选择卡：在候选层里挑出与历史 (实体,条件) 一致的那一项",
        metric="字母准确率（argmax 位置的字母 == 候选集里唯一双槽位匹配项的字母）",
        needs=("cands", "cend", "n_cands", "ent", "cond"),
        chance="1/k（k=8 ⇒ 12.50%）",
        doc="训练时 q 用金标位置做 CE（中间监督）；推理时用模型自己的 p。")
    ckpt_prefix = "select."
    ckpt_root_keys = ("q",)          # 79 的 q 挂在根上（`q`），不是 `select.q`
    slot_out = None                  # ★侧支：只 emit aux，主接口仍是输入卡的 h

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, dropout: float = DROPOUT):
        super().__init__(d, ff, nhead, dropout)
        self.q = nn.Parameter(torch.zeros(d))     # ★零初始化：不消耗 RNG（与 79 逐位同初值）

    def forward_card(self, pkt: Packet) -> Packet:
        hs = self.enc(pkt.h, src_mask=pkt.mask)   # 侧支：读主接口，但**不覆盖**它
        end_idx = pkt.aux["cand_end"]
        nc = pkt.aux["n_cands"]
        g = torch.gather(hs, 1, end_idx.unsqueeze(-1).expand(-1, -1, hs.size(-1)))   # (b,k,d)
        sc = (g * self.q).sum(-1)                                                   # (b,k)
        k = sc.size(1)
        valid = torch.arange(k, device=sc.device)[None, :] < nc[:, None]
        sc = sc.masked_fill(~valid, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        e_sel = torch.einsum("bk,bkd->bd", p, g)                                    # (b,d)
        pkt.aux.update(select_hidden=hs, cand_hidden=g, sel_scores=sc, sel_probs=p, e_sel=e_sel)
        return pkt


class RealignHarnessCard(EncLayerHarnessCard):
    """重整卡：`h = enc(h + apply ⊙ e_sel)`，apply = 「位置 >= ctx_len」（只注入回复区）。

    ★接口契约里唯一的"加性注入"点（`injection="add"`）。可判目标 = T2 逐字 EM。
    """
    spec_name = "realign"
    interface = Interface(d_in=D, d_out=D, injection="add",
                          doc="读上游 h + 上游 emit 的 e_sel；在回复区加性注入后重整")
    deps = ("h", "mask", "ctx_len", "aux:e_sel")
    judged = JudgedTarget(
        name="重整卡：把 soft 选择向量加性注入回复区并重整",
        metric="T2 完整回复逐字 EM",
        needs=("ctx_len", "target"),
        chance="",
        doc="注入置零 ⇒ 通路断开（harness probe 的 inject=off 对照）。")
    ckpt_prefix = "realign."

    slot_in = "h"        # ★读的是**输入卡的输出**（选择卡是侧支，不改主接口）
    slot_out = "h"

    def forward_card(self, pkt: Packet) -> Packet:
        e_sel = pkt.aux["e_sel"]
        n = pkt.h.size(1)
        pos = torch.arange(n, device=pkt.h.device)[None, :]
        apply = (pos >= pkt.ctx_len[:, None]).unsqueeze(-1).to(pkt.h.dtype)
        pkt.h = self.enc(pkt.h + apply * e_sel.unsqueeze(1), src_mask=pkt.mask)
        return pkt


class OutputHarnessCard(OutputCard, HarnessCardMixin):
    """输出卡：h → logits（一个线性头）—— 就是 `cards.OutputCard` 本体。"""
    spec_name = "output"
    interface = Interface(d_in=D, d_out=0, doc="h → logits[b,n,V]（线性头）")
    deps = ("h",)
    ckpt_prefix = "output."

    def __init__(self, d: int = D, vocab_size: int = 8192, maxlen: int = MAXLEN):
        super().__init__(d, vocab_size, maxlen)

    def forward_card(self, pkt: Packet) -> Packet:
        pkt.logits = self(pkt.h)
        return pkt


# ---------------------------------------------------------------------------
# 流水线
# ---------------------------------------------------------------------------
class Pipeline(nn.Module):
    """顺序执行的卡片流水线：`order` 里的卡逐张跑，卡间只传 `Packet`（= d 维隐态 + mask + ctx_len）。"""

    def __init__(self, instances: Sequence[tuple], d: int = D, maxlen: int = MAXLEN,
                 vocab: int = 8192, arch: str = "custom", note: str = ""):
        super().__init__()
        self.cards = nn.ModuleDict()
        self.order = []
        for iid, card in instances:
            assert iid not in self.order, f"[管线] 实例名重复：{iid!r}"
            assert "." not in iid, f"[管线] 实例名不能含 '.'（ModuleDict 限制）：{iid!r}"
            self.cards[iid] = card
            self.order.append(iid)
        self.d = int(d)
        self.maxlen = int(maxlen)
        self.vocab = int(vocab)
        self.arch = arch
        self.note = note

    # ------------------------------------------------------------- 元信息
    def role(self, iid: str) -> str:
        c = self.cards[iid]
        j = "可判目标：" + c.judged.metric if c.judged else "无可判中间目标"
        return f"{type(c).__name__} | {j}"

    def interface_summary(self) -> list:
        out = []
        for iid in self.order:
            c = self.cards[iid]
            it = c.interface
            out.append({"instance": iid, "spec": c.spec_name,
                        "slot_in": c.slot_in, "slot_out": c.slot_out,
                        "side_branch": c.slot_out is None,
                        "w": self.d,
                        "d_in": self.d if it.d_in else 0,
                        "d_out": self.d if it.d_out else 0,
                        "emits_logits": it.d_out == 0,
                        "injection": it.injection,
                        "emits": list(it.emits), "carries_position": it.carries_position,
                        "deps": list(c.deps),
                        "judged": None if c.judged is None else {
                            "name": c.judged.name, "metric": c.judged.metric,
                            "needs": list(c.judged.needs), "chance": c.judged.chance},
                        "params": sum(p.numel() for p in c.parameters()),
                        "params_bytes": c.param_bytes(),
                        "doc": it.doc})
        return out

    def topology(self) -> list:
        """拓扑（不是"一条链"）：逐卡 读槽 → 写槽；`slot_out=None` 的卡是侧支。"""
        out = []
        for iid in self.order:
            c = self.cards[iid]
            out.append({"instance": iid, "spec": c.spec_name,
                        "reads": c.slot_in, "writes": c.slot_out,
                        "emits": list(c.interface.emits),
                        "side_branch": c.slot_out is None})
        return out

    def topology_line(self) -> str:
        parts = []
        for t in self.topology():
            arrow = f"{t['reads']}→{t['writes'] or 'aux'}"
            tag = f"{t['instance']}[{arrow}]"
            if t["side_branch"]:
                tag += "⤵侧支"
            if t["emits"]:
                tag += "{" + ",".join(t["emits"]) + "}"
            parts.append(tag)
        return "  ".join(parts)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def n_params_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def param_table(self) -> dict:
        per = {iid: sum(p.numel() for p in self.cards[iid].parameters()) for iid in self.order}
        per["合计"] = self.n_params()
        return per

    def param_bytes_total(self) -> int:
        return param_bytes(self)

    # ------------------------------------------------------------- 前向
    @torch.no_grad()
    def run(self, tokens: torch.Tensor, cand_end: torch.Tensor | None = None,
            n_cands: torch.Tensor | None = None, ctx_len: torch.Tensor | None = None,
            meter: Meter | None = None) -> Packet:
        """跑一遍：按顺序逐卡执行；只跑前向（`torch.no_grad()`）。"""
        pkt = Packet(tokens=tokens, ctx_len=ctx_len)
        if cand_end is not None:
            pkt.aux["cand_end"] = cand_end
            pkt.aux["n_cands"] = n_cands
        resident = param_bytes(self) if meter is not None else 0
        for iid in self.order:
            card = self.cards[iid]
            pkt = card.forward_card(pkt)
            if meter is not None:
                meter.stage(iid, card, pkt, resident)
        return pkt

    @torch.no_grad()
    def logits(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.run(tokens).logits

    # ------------------------------------------------------------- 载 ckpt
    def load_ckpt(self, path: str, map_location="cpu", verbose: bool = False) -> dict:
        """把外部 ckpt（79 的 `input.*/select.*/realign.*/output.*/q` 布局）灌进本管线。

        返回逐卡 loaded/missing/unexpected 报告（不允许静默丢 key）。
        """
        blob = torch.load(path, map_location=map_location, weights_only=False)
        if isinstance(blob, dict) and "state" in blob and "format" in blob:
            blob = blob["state"]                     # cards.CardPipeline.save() 的包
        used = set()
        report = {}
        for iid in self.order:
            card = self.cards[iid]
            if card.ckpt_prefix is None and not card.ckpt_root_keys:
                report[iid] = {"loaded": [], "missing": [], "unexpected": [], "skipped": True}
                continue
            local = card.state_dict()
            got, unexpected_ext = {}, []
            for k, v in blob.items():
                if k in card.ckpt_root_keys:
                    lk = k
                elif card.ckpt_prefix and k.startswith(card.ckpt_prefix):
                    lk = k[len(card.ckpt_prefix):]
                else:
                    unexpected_ext.append(k)
                    continue
                if lk in local:
                    got[lk] = v
                    used.add(k)
            missing = [k for k in local if k not in got]
            bad = [k for k, v in got.items() if tuple(v.shape) != tuple(local[k].shape)]
            card.load_state_dict(got, strict=False)
            report[iid] = {"loaded": sorted(got), "missing": sorted(missing),
                           "unexpected": sorted(unexpected_ext),
                           "shape_mismatch": sorted(bad), "skipped": False}
            if verbose:
                print(f"[CKPT] {iid}: 载入 {len(got)} | 缺 {sorted(missing)} | "
                      f"不属本卡 {sorted(unexpected_ext)[:3]}{'…' if len(unexpected_ext) > 3 else ''}")
        self.eval()
        unused = sorted(set(blob.keys()) - used)
        return {"path": path, "sha256": file_sha256(path), "per_card": report,
                "unused_keys": unused,
                "missing_any": sorted({m for r in report.values() for m in r.get("missing", [])}),
                "shape_mismatch_any": sorted({m for r in report.values()
                                              for m in r.get("shape_mismatch", [])})}


# ---------------------------------------------------------------------------
# 唯一构造入口（★先 manual_seed 再构造）
# ---------------------------------------------------------------------------
def build_pipeline(arch: str = "B2", seed: int = 1234, d: int = D, ff: int = FF,
                   nhead: int = NHEAD, maxlen: int = MAXLEN, dropout: float = DROPOUT,
                   device: str = "cpu", dtype=torch.float32, verbose: bool = False) -> Pipeline:
    """按**卡组名**装配管线（先播种、再构造；构造完打印 [INIT-CHECK] 摘要）。

    卡组（arch）：
      "B2"      input → select → realign → output   （选择卡+重整卡；记录值 75.88% 的那一个）
      "B1"      同上（架构相同，只是训练时没有选择 CE —— 载 ckpt 才有区别）
      "A"       input → target → output             （单卡直出：2 层 target 合成一张卡）
      "native3" input → target(residual=False) → output  （= cards.CardPipeline(residual=False)）
      "custom"  由 `--cards` 指定的 (实例名:spec) 列表
    """
    from harness.registry import REGISTRY, default_layers
    v = vocab_size()
    torch.manual_seed(seed)                       # ★播种在构造之前（§9.7）
    layers = default_layers(arch)
    instances = []
    for iid, spec in layers:
        spec_obj = REGISTRY[spec]
        kw = dict(d=d, ff=ff, nhead=nhead, maxlen=maxlen, dropout=dropout, vocab_size=v,
                  pad=pad_id())
        card = spec_obj.build(**spec_obj.filter_kwargs(kw))
        instances.append((iid, card))
    pipe = Pipeline(instances, d=d, maxlen=maxlen, vocab=v, arch=arch,
                    note=" ".join(f"{i}:{s}" for i, s in layers))
    pipe = pipe.to(device=device, dtype=dtype) if dtype != torch.float32 else pipe.to(device=device)
    pipe.eval()
    if verbose:
        print(f"[INIT-CHECK] arch={arch} seed={seed} d={d} 参数={pipe.n_params()} "
              f"state_sha256[0:16]={state_digest(pipe)} "
              f"（同 seed 两次构造必须相同 ⇒ 确定性构造）", file=sys.stderr, flush=True)
    return pipe


def _layers_for(arch: str) -> list:
    """（保留给外部调用）卡组 → [(实例名, 注册名)]；真相源在 `harness/registry.py::CARD_SETS`。"""
    from harness.registry import default_layers
    return default_layers(arch)


def build_custom(layers: Sequence[tuple], seed: int = 1234, device: str = "cpu", **kw) -> Pipeline:
    """按 `[(实例名, 注册名), …]` 装配（H-d：加卡不改本函数）。"""
    from harness.registry import REGISTRY
    v = vocab_size()
    torch.manual_seed(seed)
    instances = []
    for iid, spec in layers:
        spec_obj = REGISTRY[spec]
        kw2 = dict(d=kw.get("d", D), ff=kw.get("ff", FF), nhead=kw.get("nhead", NHEAD),
                   maxlen=kw.get("maxlen", MAXLEN), dropout=kw.get("dropout", DROPOUT),
                   vocab_size=v, pad=pad_id())
        instances.append((iid, spec_obj.build(**spec_obj.filter_kwargs(kw2))))
    pipe = Pipeline(instances, d=kw.get("d", D), maxlen=kw.get("maxlen", MAXLEN), vocab=v,
                    arch="custom", note=" ".join(f"{i}:{s}" for i, s in layers))
    pipe.to(device)
    pipe.eval()
    return pipe


# ---------------------------------------------------------------------------
# 贪心生成（口径 = 79::greedy_gen，batch=1）
# ---------------------------------------------------------------------------
@torch.no_grad()
def greedy_gen(pipe: Pipeline, recs: Sequence[dict], maxlen: int = MAXLEN) -> list:
    """贪心解码（每步 argmax，不采样）；生成时用模型自己的 p（不是金标）。逐字对齐 79::greedy_gen。"""
    pipe.eval()
    dev = next(pipe.parameters()).device
    texts = []
    for r in recs:
        ids = torch.tensor([r["p"]], dtype=torch.long, device=dev)
        s = torch.tensor([len(r["p"])], dtype=torch.long, device=dev)
        eidx = torch.tensor([r["cend"]], dtype=torch.long, device=dev)
        nc = torch.tensor([r["n_cands"]], dtype=torch.long, device=dev)
        gen = []
        for _ in range(maxlen - len(r["p"])):
            logits = pipe.run(ids, cand_end=eidx, n_cands=nc, ctx_len=s).logits
            t = int(logits[0, -1].argmax().item())
            if t == eos_id():
                break
            gen.append(t)
            ids = torch.cat([ids, torch.tensor([[t]], dtype=torch.long, device=dev)], dim=1)
        texts.append(dec(gen))
    return texts


@torch.no_grad()
def greedy_trace(pipe: Pipeline, rec: dict, maxlen: int = MAXLEN, show_n: int = 12) -> dict:
    """贪心解码**逐 token 留痕**（同一路径；额外记每步概率，供 chat/演示打印）。"""
    pipe.eval()
    dev = next(pipe.parameters()).device
    ids = torch.tensor([rec["p"]], dtype=torch.long, device=dev)
    s = torch.tensor([len(rec["p"])], dtype=torch.long, device=dev)
    eidx = torch.tensor([rec["cend"]], dtype=torch.long, device=dev)
    nc = torch.tensor([rec["n_cands"]], dtype=torch.long, device=dev)
    steps, gen = [], []
    for _ in range(max(1, maxlen - len(rec["p"]))):
        logits = pipe.run(ids, cand_end=eidx, n_cands=nc, ctx_len=s).logits
        lp = torch.log_softmax(logits[0, -1], dim=-1)
        t = int(logits[0, -1].argmax().item())
        prob = float(lp[t].exp().item())
        if t == eos_id():
            steps.append({"i": len(steps), "token": "<eos>", "id": t, "p": prob, "eos": True})
            break
        steps.append({"i": len(steps), "token": dec([t]), "id": t, "p": prob, "eos": False})
        gen.append(t)
        ids = torch.cat([ids, torch.tensor([[t]], dtype=torch.long, device=dev)], dim=1)
    return {"steps": steps[:show_n], "n_steps": len(steps), "text": dec(gen),
            "all_steps": steps}


# ---------------------------------------------------------------------------
# 评测口径（= 79::eval_arms 的三件事；T1 仪器直接复用 stages/76 的唯一定义）
# ---------------------------------------------------------------------------
def _d76():
    """按路径加载 `stages/76_letter_cand_gen.py`（标签仪器**唯一真相源**，不另写一份）。"""
    global _D76
    try:
        return _D76
    except NameError:
        pass
    spec = importlib.util.spec_from_file_location(
        "h_d76_label_ref", os.path.join(STAGES, "76_letter_cand_gen.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    _D76 = mod
    return mod


def letters_in(text: str) -> list:
    return _d76().letters_in(text or "")


def dec_letter(text: str):
    """T1 口径：文本里的标签字母；**恰 1 个**才算抽出（0 个 / 多个都算抽不出 = None）。"""
    ls = letters_in(text)
    return ls[0] if len(ls) == 1 else None


def split_label(reply: str):
    return _d76().split_label(reply)


def label_marker(letter: str) -> str:
    """标签渲染（全角括号）；口径来自 `stages/76` 的唯一定义。"""
    return _d76().LABEL_TMPL.format(letter=letter)


def correct_letter_of(r: dict):
    """候选集里「唯一同时匹配历史两个槽位」的候选的**位置字母**（动态金标；§11.17）。"""
    hit = [c for c in r["cands"] if c["ent"] == r["ent"] and c["cond"] == r["cond"]]
    if len(hit) != 1:
        return None
    pos = r["cands"].index(hit[0])
    return CAND_LETTERS[pos] if pos < len(CAND_LETTERS) else None


def em_strict(gold: str, pred: str) -> int:
    return int((gold or "").strip() == (pred or "").strip())


@torch.no_grad()
def select_readout(pipe: Pipeline, recs: Sequence[dict], batch: int = 32) -> dict | None:
    """选择卡读数（只喂 ctx，与训练同口径的 p）。管线里没有选择卡 ⇒ None。"""
    if "select" not in pipe.order:
        return None
    dev = next(pipe.parameters()).device
    pipe.eval()
    hits, lhits, probs, preds = [], [], [], []
    for i in range(0, len(recs), batch):
        chunk = recs[i: i + batch]
        ids, s, eidx, nc = pack(chunk, with_target=False, device=dev)
        pkt = pipe.run(ids, cand_end=eidx, n_cands=nc, ctx_len=s)
        sc = pkt.aux["sel_scores"]
        pp = pkt.aux["sel_probs"]
        for j, r in enumerate(chunk):
            pj = int(sc[j].argmax().item())
            preds.append(pj)
            c = r["cands"][pj] if pj < r["n_cands"] else None
            hits.append(int(c is not None and c["ent"] == r["ent"] and c["cond"] == r["cond"]))
            gl = correct_letter_of(r)
            pl = CAND_LETTERS[pj] if (gl is not None and pj < r["n_cands"]) else None
            lhits.append(int(gl is not None and pl == gl))
            probs.append(float(pp[j, pj].item()))
    return {"hits": hits, "letter_hits": lhits, "pred": preds, "p_max": probs}


@torch.no_grad()
def eval_records(pipe: Pipeline, recs: Sequence[dict], maxlen: int = MAXLEN,
                 batch: int = 32) -> dict:
    """T2 完整回复 EM / T1 字母准确率 / 选择准确率 —— 与 79 出记录值时同一口径。"""
    pipe.eval()
    txt = greedy_gen(pipe, recs, maxlen=maxlen)
    hits = [em_strict(r["target"], txt[i]) for i, r in enumerate(recs)]
    ls = [dec_letter(t) for t in txt]
    gl = [r.get("gold_letter") for r in recs]
    usable = [(g, x) for g, x in zip(gl, ls) if g is not None]
    t1 = 100.0 * sum(1 for g, x in usable if x == g) / len(usable) if usable else float("nan")
    t2 = 100.0 * sum(hits) / len(hits) if hits else float("nan")
    miss = sum(1 for x in ls if x is None)
    sel = select_readout(pipe, recs, batch=batch)
    return {"n": len(recs), "t2": t2, "t1": t1, "miss": miss,
            "sel": None if sel is None else 100.0 * sum(sel["hits"]) / len(sel["hits"]),
            "sel_letter": None if sel is None else 100.0 * sum(sel["letter_hits"]) / len(sel["letter_hits"]),
            "hits": hits, "texts": txt, "letters": ls,
            "sel_hits": None if sel is None else sel["hits"],
            "sel_pred": None if sel is None else sel["pred"],
            "sel_p_max": None if sel is None else sel["p_max"]}


def compare_records(arm: str, m: dict) -> list:
    """与 §11.26/§11.28 的记录值逐项比（口径 = 双方按 `:.2f` 显示后相等）。不一致只报 Δ，不改记录值。"""
    rec = RECORDS.get(arm)
    if rec is None:
        return []
    rows = []
    for key, label in (("t2", "T2 完整回复 EM"), ("t1", "T1 字母准确率"), ("sel", "选择准确率(argmax)")):
        want, got = rec.get(key), m.get(key)
        if want is None:
            rows.append({"metric": key, "label": label, "record": None, "measured": got,
                         "delta": None, "ok": got is None, "note": "记录值不适用（该臂无选择卡）"})
            continue
        ok = (got is not None) and (f"{got:.2f}" == f"{want:.2f}")
        rows.append({"metric": key, "label": label, "record": want, "measured": got,
                     "delta": None if got is None else got - want, "ok": bool(ok),
                     "note": "" if ok else "★与记录值不一致 ⇒ 必须查因，不许改记录值迁就"})
    return rows


# ---------------------------------------------------------------------------
# 载模型（可复用入口；80/81/83/85 那一类脚本可以直接调它）
# ---------------------------------------------------------------------------
def load_model(arch: str = "B2", ckpt: str = CKPT_B2_K8_S1234, seed: int = 1234,
               device: str = "cpu", dtype=torch.float32, verbose: bool = False) -> tuple:
    """构建 + 载 ckpt（唯一入口）。返回 (pipe, report)。"""
    pipe = build_pipeline(arch=arch, seed=seed, device=device, dtype=dtype, verbose=verbose)
    rep = pipe.load_ckpt(ckpt, map_location="cpu", verbose=verbose)
    pipe.to(device=device)
    pipe.eval()
    return pipe, rep


def param_groups_named(pipe: Pipeline) -> dict:
    return pipe.param_table()


# ---------------------------------------------------------------------------
# ★对拍：harness 的卡组 == cards/ 的正规路径（H-e/硬约束 2 的机器证据）
# ---------------------------------------------------------------------------
def cross_check_cards_pipeline(device: str = "cpu", d: int = D, maxlen: int = MAXLEN) -> dict:
    """`native3`（input → target(residual=False) → output）对拍 `cards.CardPipeline(residual=False)`：
    参数量必须相等、**同一权重下前向必须逐位相同**。"""
    from cards import CardPipeline
    v = vocab_size()
    torch.manual_seed(1234)
    ref = CardPipeline(d=d, vocab_size=v, ff=FF, nhead=NHEAD, maxlen=maxlen, pad_id=pad_id(),
                       dropout=DROPOUT, residual=False)
    ref.eval()
    pipe = build_pipeline("native3", seed=1234, device=device)
    pipe.cards["input"].load_state_dict(ref.input.state_dict(), strict=True)
    pipe.cards["target"].load_state_dict(ref.target.state_dict(), strict=True)
    pipe.cards["output"].load_state_dict(ref.output.state_dict(), strict=True)
    pipe.eval()
    ids = torch.randint(0, v, (3, 48), generator=torch.Generator().manual_seed(7))
    with torch.no_grad():
        a = ref.route(ids, record=False)["logits"]
        b = pipe.run(ids).logits
    m = float((a - b).abs().max().item())
    return {"pipeline_params": ref.n_params(), "native_params": pipe.n_params(),
            "params_equal": ref.n_params() == pipe.n_params(),
            "max_abs": m, "bitwise_equal": bool(torch.equal(a, b)),
            "residual": False, "ok": (ref.n_params() == pipe.n_params()) and m == 0.0}
