#!/usr/bin/env python3
"""S81 · L3-r5「重整卡的独立验收」：B0-a（重整卡单独、无可判目标）vs B0-b（重整卡 + 有序模因可判目标）。

★本文件是 `stages/79_cand_cards_v2.py` 的**副本**（79 已入库、其读数被本单元当对照引用 ⇒ 一律不改 79）。
  跨臂模型 / 训练循环 / 配对统计 / 反事实骨架逐字沿用 79；本文件只做下面这些改动（逐条可 diff）：

1. 新增两臂 **B0a / B0b**（L3_PLAN §14）：
     B0 = input(1 层) + **重整卡**(1 层)，**无选择卡**；候选经**非选择性**汇总
          （候选末位 token 的隐态在 **input 卡输出**上取**均值**，固定、无参数、不学）加性注入重整卡
          （注入位置 = 回复区，即位置 >= ctx 长；B0b 的 mid 区也在回复区里）；
     B0a：**无**中间目标（只有回复损失）；
     B0b：**有** —— 中间目标 = 先输出匹配到的**有序模因项** `（X）实体/条件`（X = 候选字母），
          再输出回复（**同一自回归序列** ⇒ 回复位置通过因果注意力看到模型**自己**写出的中间项）。

2. ★**参数对齐（§11.11 的「目标层切半」口径）**：
     A = input(1) + target(2) = 3 个 encoder 层；B1/B2 = input(1) + select(1) + realign(1) = 3 层；
     B0 = input(1) + realign(1) = **2 个 encoder 层** —— B0 没有选择卡，按 §14 的结构定义**少 1 层**
     （实测 -198272 ≈ -7.34%，见 [PARAMS]）；这是结构定义的直接后果，不是实现偏差。
     ★为保住「参与比较的臂共享子模块逐位同初值」这一**配对前提**，B0 的构造顺序做成
       input → 第 1 个 encoder 层（**丢弃**，只为与 A/B1/B2 的 select 层消耗同一条 RNG 流）
       → 第 2 个 encoder 层（= realign，与 A.target.layers.1 / B1,B2.realign **逐位相同**）→ output
       ⇒ input / realign / output 三个共享子模块与 A、B1、B2 逐张量 max|Δ|=0（[INIT-CHECK] 实测）。
     ⇒ B0 vs A 的共享集 = {input, realign↔A.target.layers.1, output}；B0 vs B1/B2 的共享集
       = {input, realign↔realign, output}。**不共享**的是 B1/B2 的 select 层（B0 结构上就没有它）。

3. **B0b 的中间目标仪器**：`mid = 「（X）实体/条件」`（X = 该候选在候选行里的**位置字母**，即「有序」）。
     训练序列 = enc(mid) + [SEP] + enc(回复 + <eos>)，SEP = "\n"（单 token、id=0、非 pad/eos/unk；
     7800/7800 条实测「按片段拼 token == 整串 encode」、mid 往返解码违例 0、SEP 与 mid/回复无冲突）。
     评测 = **两段贪心**：① 从 ctx 生成 mid（遇 SEP/EOS 或满 MID_CAP 停）② **强制补 SEP** 后生成回复
     （回复只依赖模型**自己**写出的 mid；与 §11.11「生成时用模型自己的 p，不是金标」同口径）。
     读数：**T_mid**（mid 逐字 EM / mid 字母命中 / mid 端到端）+ **T1**（回复里的字母）+ **T2**（回复逐字 EM）。
     B0a 走 79 的单段贪心（逐字同 79 的 A/B 臂）⇒ B0a 的 T2/T1 与 A/B1/B2 **同口径可比**。

4. 修 79 的一处**仪器缺陷**（本文件核查确认，79 未改，如实记）：
     `stages/79_cand_cards_v2.py:755` 的 `shared_init_check` **返回逐张量 Δ 的 dict**，而
     `:1122` 写的是 `init_worst == 0.0` ⇒ 布尔**恒 False** ⇒ 79 每次 run 都打印
     `[CRIT] R3′-a … ★FAIL ⇒ 配对作废`、hits JSON 里 `init_ok=false`，
     **尽管它紧邻的上一行 [INIT-CHECK] 实测是 `max|Δ|=0.000e+00 … PASS 逐位同初值`**。
     ⇒ 79 的**配对统计本身仍然有效**（初值确实逐位相同），失效的只是那个布尔判词与 JSON 字段。
     本文件让 `shared_init_check` 返回 worst float，判定与 JSON 一致（R5-a 只认实测行）。

5. **[INIT-CHECK] 增加跨文件断言**：用 `_load_path` **只读**载入 `stages/79_cand_cards_v2.py`，
     同 seed 下构造 79 的 A/B1/B2，与 81 的 B0a/B0b 逐张量核对共享子模块；另核对 81 自己那份
     A/B1/B2 副本与 79 的构造**逐位相同**（`[COPY-CHECK]` ⇒ 证明副本没漂移）。

6. 判据换成 §14 的 **R5-a..e**（`--report-r5`）；**三种结局的判词写死在 `report_r5()` 里**，
   机制化打印（B0a≈A 且 B0b≫B0a / B0a>A / B0b≈B0a），不事后找说法；天花板条款（任一臂 T2 ≥95%
   ⇒ 判「不可测」）与两套门槛（T1 机会级 1/k+2SE、T2 捷径级 max_naive+2SE）逐条打印、**不混**。
   ★R5-c（B0b vs B0a）与 §11.10 的 B1/B2 一样是**极干净的单变量**：两臂**参数/初值逐位相同**
   （同 sha，只差训练目标里有没有 mid）—— [INIT-CHECK] 实测。

数据（只读，不造）：`logs/76_dialog_{train,test}.jsonl`（k=8，7000/800）——
  回复**必须含被选中候选的字母**（`没问题，AS975482 会按返厂检测安排（E）。`）；
  T1 机会地板 = 1/k = 12.50%（门槛 14.84%），T2 捷径级 `max_naive` = 0.00%（§11.14）。
对照（**已测，直接引用，不重跑**）：A = 12.38%/12.50%（T2，两 seed）、B1 = 12.12%/12.75%、
  B2 = 75.88%/61.25%（§11.24 / §11.29 / §11.28），hits 从其 `logs/79_hits_*.json` 读。

★诚实边界（跑前写死）：合成模板数据、k=8、单卡规模；**B0b 的中间目标形式（`（X）实体/条件`）是本单元
  选的口径，未做形式敏感性对照**（换形式/换 SEP/换 mid 长度都未测）⇒ 结论只限该形式；B0 比 A/B1/B2
  **少 1 个 encoder 层**（结构定义所致）⇒ 参数量比较必须带着这个前提读；两 seed 只是两次运行。

用法（本机只有 .venv/bin/python 装了 torch/tokenizers）：
  .venv/bin/python stages/81_realign_card_ablation.py --check-data                 # 只读数据+复算门槛（CPU）
  .venv/bin/python stages/81_realign_card_ablation.py --param-table                # 参数/初值对齐表（CPU）
  .venv/bin/python stages/81_realign_card_ablation.py --arm B0a --seed 1234 --smoke --cf
  .venv/bin/python stages/81_realign_card_ablation.py --arm B0b --seed 1234 --cf \
      --pair-hits logs/81_hits_B0a_s1234.json,logs/79_hits_B2_s1234.json
  .venv/bin/python stages/81_realign_card_ablation.py --report-r5                  # 读数齐后出 R5-a..e（CPU）
GPU 长跑必须：systemd-run --user --unit=<名> --collect
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 ... .venv/bin/python ...
只允许写本文件、`scripts/81_runs.sh` 与 logs/81_*。
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import random
import sys
import time

NS_ROOT = "/home/vesita/coding/my/nanoSeek"
if NS_ROOT not in sys.path:
    sys.path.insert(0, NS_ROOT)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.optim import AdamW  # noqa: E402

from cards import Card, InputCard, OutputCard, encoder_layer  # noqa: E402
from cards.base import IO_D_TO_D, state_sha256  # noqa: E402

# ---------------- 配置（与 S65/S67/S71/S73 一致） ----------------
D = 128
FF = 512
NHEAD = 4
MAXLEN = 512
STEPS = 3000
LR = 1e-3
BATCH = 32
SMOKE_STEPS = 50
SMOKE_EVAL = 60
DROPOUT = 0.1
SEL_LAMBDA = 1.0                       # B2 选择 CE 的权重（写死）
LOG_DIR = os.path.join(ROOT, "logs")
TOK_PATH = NS_ROOT + "/data/chinese/char_tokenizer.json"
D74_PATH = os.path.join(ROOT, "stages", "74_nl_dialog_cand_gen.py")
D76_PATH = os.path.join(ROOT, "stages", "76_letter_cand_gen.py")
D79_PATH = os.path.join(ROOT, "stages", "79_cand_cards_v2.py")   # ★只读载入：跨文件初值/参数核对

# ---- S81 新增（§14）：B0 两臂 + B0b 的中间目标 ----
B0_ARMS = ("B0a", "B0b")            # r5（§14）：候选**非选择性均值**注入
B0X_ARMS = ("B0pa", "B0pb")         # r5′（§14.2）：重整卡对候选**交叉注意**（保留身份与位置）
REALIGN_LAYERS = 1                  # 重整卡的 self-attn 层数（--realign-layers {1,2}；2 = 参数对齐档）
# B0b 的中间目标 = 「（X）实体/条件」+ SEP + 回复；SEP 必须是**单 token 且非 pad/eos/unk**：
#   实测 "\n" → [0]（PAD=131 EOS=128 UNK=129）⇒ 安全；7800/7800 条「片段拼 token == 整串 encode」
B0_SEP = "\n"
MID_CAP = 24            # 第一段贪心最多生成的 mid token 数（mid 实测 16 token）

# ★数据默认档 = 76（k=8，带字母，§11.14）；74 为回融档（k=4，无标签），78 为 r3a 的 k=8 无标签档
PREFIX_BY_K = {4: "74", 8: "76"}
# 各档的「回复里是否含标签字母」——由数据字段实测判定，这里的表只用来提前给出期望值
LABELED_PREFIXES = ("76", "77")
# §11.6 的引文（r3a 档的机会级门槛；原始行在 L3_PLAN.md §11.6）
MAX_CHANCE_QUOTED = {4: 27.12, 8: 13.12}
# §11.14 的引文：完整回复的捷径级 max_naive（k=8 实测 0.00%，来自 ③复制轮3；k=4 为 0.12%）
MAX_NAIVE_QUOTED = {8: 0.00, 4: 0.12}
# §11.6/§11.14 引文里的指纹（仅供参考性对齐，不作判据）：prefix → (train_fp, test_fp)
FP_QUOTED = {"74": ("213a203507787631", "25ce4966054660d0"),
             "76": ("efeb1ee872f4b6be", "1ae76682bd7cc997")}

CAND_PREFIX = "候选："
CAND_SEP = "；"
CAND_LETTERS = "ABCDEFGH"


def _load_path(name: str, path: str):
    """按路径加载（文件名以数字开头，不能普通 import）。"""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


d74 = _load_path("d74_nl_dialog_cand_gen", D74_PATH)   # EOS / compute_floors / data_fp / file_sha256
d76 = _load_path("d76_letter_cand_gen", D76_PATH)      # LABEL_RE / split_label / letters_in / 地板（T1+T2）
d79 = _load_path("d79_cand_cards_v2", D79_PATH)        # ★只读：79 的 A/B1/B2 构造（跨文件 [INIT-CHECK]）

# ★标签工具直接复用 76（口径唯一真相源，避免本文件另写一份漂移）
LABEL_RE = d76.LABEL_RE
label_marker = d76.label_marker
split_label = d76.split_label
letters_in = d76.letters_in

tok = None
V = 0
PAD_ID = EOS_ID = UNK_ID = RESP_ID = SEP_ID = 0


def load_tok():
    """载入分词器并设全局 V/PAD/EOS/SEP（CPU 也要，参数表与配对报表不碰 GPU）。"""
    global tok, V, PAD_ID, EOS_ID, UNK_ID, RESP_ID, SEP_ID
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(TOK_PATH)
    V = tok.get_vocab_size()
    PAD_ID = tok.token_to_id("<pad>")
    EOS_ID = tok.token_to_id("<eos>")
    UNK_ID = tok.token_to_id("<unk>")
    RESP_ID = tok.token_to_id("<resp>")
    SEP_ID = tok.token_to_id(B0_SEP)
    assert SEP_ID is not None and enc(B0_SEP) == [SEP_ID], (B0_SEP, SEP_ID, enc(B0_SEP))
    assert SEP_ID not in (PAD_ID, EOS_ID, UNK_ID), \
        f"[B0b] SEP={B0_SEP!r} 撞上特殊 token（SEP={SEP_ID} PAD={PAD_ID} EOS={EOS_ID} UNK={UNK_ID}）"
    # ★跨文件核对的前提：79 的 ArmA/ArmB 构造时读的是 **79 模块自己的** V/PAD_ID（import 不会跑它的
    #   main ⇒ 那里还是 0）⇒ 必须把 V/PAD 同步过去，否则 79 的臂会是 0 维 embedding/head（实测踩过）。
    d79.tok, d79.V, d79.PAD_ID, d79.EOS_ID, d79.UNK_ID, d79.RESP_ID = tok, V, PAD_ID, EOS_ID, UNK_ID, RESP_ID


def enc(text: str) -> list:
    return tok.encode(text, add_special_tokens=False).ids


def dec(ids) -> str:
    return tok.decode([int(i) for i in ids], skip_special_tokens=True)


# ---------------- 标签（T1 的仪器，口径 = d76） ----------------
def letters_of(text: str) -> list:
    """生成文本里出现的标签字母（`（X）`）；空 = 该预测没给出任何字母 ⇒ 抽不出。"""
    return letters_in(text or "")


def dec_letter(text: str) -> str:
    """T1 口径：文本里的标签字母；**恰 1 个**才算抽出（0 个 / 多个都算抽不出 = None）。"""
    ls = letters_of(text)
    return ls[0] if len(ls) == 1 else None


def gold_letter_of(r: dict):
    """金标字母：优先用数据字段；无标签档（74/78）返回 None。"""
    return r.get("gold_letter")


def correct_letter_of(r: dict):
    """本条 ctx 的候选集里「唯一同时匹配历史两槽位」的那个候选的**位置字母**；没有则 None。

    这是反事实下的动态金标（shuffle/replace/delete 都会改候选集 ⇒ 不能沿用 base 的 gold_letter）。
    """
    hit = [c for c in r["cands"] if c["ent"] == r["ent"] and c["cond"] == r["cond"]]
    if len(hit) != 1:
        return None
    pos = r["cands"].index(hit[0])
    return CAND_LETTERS[pos] if pos < len(CAND_LETTERS) else None


def target_letter_em(gold: str, pred: str) -> int:
    """T2 口径的一个分支：pred 的标签字母 == gold 的标签字母（句面另算）。"""
    sl = split_label(gold or "")
    pl = dec_letter(pred)
    return int(sl is not None and pl is not None and sl[1] == pl)


# ---------------- B0b 的中间目标（§14 的「有序模因项」） ----------------
def mid_text_of(ent: str, cond: str, letter: str) -> str:
    """中间项字面形式（**写死**）：`（X）实体/条件`，X = 候选字母 = 该候选在候选行里的位置（「有序」）。"""
    return f"（{letter}）{ent}/{cond}"


def mid_gold_of(r: dict):
    """本条的中间目标。★**动态金标**：字母取「当前候选集里唯一双槽位匹配项」的位置
    （§11.17）⇒ shuffle/replace/delete 后自动跟着变；无唯一匹配项则 None（该题不可判）。"""
    letter = correct_letter_of(r)
    if letter is None:
        return None
    return mid_text_of(r["ent"], r["cond"], letter)


# ---------------- 数据 ----------------
def load_jsonl(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(ln) for ln in f if ln.strip()]


def fp_reply(recs_raw) -> str:
    """数据身份指纹（按 79/§11.14 的**回复目标**口径算 p+t）。

    ★必须在 prep_record **之前**调用：B0b 的 r["t"] 含 mid 段 ⇒ 直接对 prep 后的记录算指纹会变，
    那样就掩盖了「数据本身是否与引文一致」这件事（本函数只回答数据身份）。
    """
    return d74.d72.data_fp([{"p": enc(r["ctx"]), "t": enc(r["target"] + d74.EOS)} for r in recs_raw])


POS_BAD = {"n": 0}
MID_BAD = {"add": 0, "rt": 0}          # B0b 中间目标的编码自检计数（实测必须 0）


def prep_record(r: dict, with_target: bool = True, variant: str = "A") -> dict:
    """把文本 → token：p=enc(ctx)、t=目标序列、cend=每个候选**末位 token** 的位置。

    variant="A"  ：t = enc(回复 + <eos>)（与 79 逐字相同）；
    variant="B0b"：t = enc(mid) + [SEP] + enc(回复 + <eos>)，mid =「（X）实体/条件」（字母取**动态金标**）
                   ⇒ 逐 token 拼接（不整串 encode），边界不稳时记 MID_BAD（7800/7800 实测 0 违例）；
                   `mid_len` = mid+SEP 的 token 数（= 回复段在 t 里的起始偏移，供分段 CE 用）。
    cend 的算法：候选文本在 ctx 里的字符区间 → 该前缀的 token 数 − 1；同时核验「前缀编码稳定」
    （enc(前缀) == enc(ctx)[:len]），违例计数进 POS_BAD（实测，不是推断）。
    """
    r["p"] = enc(r["ctx"])
    if with_target and "t" not in r:
        r["treply"] = enc(r["target"] + d74.EOS)
        if variant == "B0b":
            mid = mid_gold_of(r)
            r["mid"] = mid
            r["tsep"] = [SEP_ID]
            if mid is None:                      # 反事实端可能「无正确对」（§11.25）⇒ mid 不可判
                r["tmid"] = []
            else:
                r["tmid"] = enc(mid)
                if dec(r["tmid"]) != mid:
                    MID_BAD["rt"] += 1
                if enc(mid + B0_SEP + r["target"] + d74.EOS) != r["tmid"] + r["tsep"] + r["treply"]:
                    MID_BAD["add"] += 1
            r["t"] = r["tmid"] + r["tsep"] + r["treply"]
            r["mid_len"] = len(r["tmid"]) + 1    # mid + SEP ⇒ 回复在 t 里的起始偏移
        else:
            r["t"] = r["treply"]
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


def pack(recs, with_target: bool = True):
    """(ids, ctx_len, end_idx, n_cands)：ctx 在左、回复在右，右 padding（不加任何位置偏移）。"""
    K = max(r["n_cands"] for r in recs)
    lens = [len(r["p"]) + (len(r["t"]) if with_target else 0) for r in recs]
    n = max(lens)
    ids = torch.full((len(recs), n), PAD_ID, dtype=torch.long)
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
    return ids.to(device), s.to(device), eidx.to(device), nc.to(device)


# ---------------- 三臂模型 ----------------
class DeepTargetCard(Card):
    """A 臂的目标部分：把单卡的 1 层 think 堆到 n_layers 层（照 cards/target.py 的 residual=False 口径）。"""

    io = IO_D_TO_D
    card_name = "target_deep"

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT, n_layers: int = 2):
        super().__init__(d, maxlen)
        self.layers = nn.ModuleList([encoder_layer(d, ff, nhead, dropout) for _ in range(n_layers)])

    def apply_card(self, x, mask=None):
        for L in self.layers:
            x = L(x, src_mask=mask)          # residual=False：裸层，不加外层残差
        return x


class SelCard(Card):
    io = IO_D_TO_D
    card_name = "select"

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT):
        super().__init__(d, maxlen)
        self.enc = encoder_layer(d, ff, nhead, dropout)

    def apply_card(self, x, mask=None):
        return self.enc(x, src_mask=mask)


class RealignCard(Card):
    io = IO_D_TO_D
    card_name = "realign"

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT):
        super().__init__(d, maxlen)
        self.enc = encoder_layer(d, ff, nhead, dropout)

    def apply_card(self, x, mask=None):
        return self.enc(x, src_mask=mask)


class ArmA(nn.Module):
    """单卡直出：input(1 层) → target(2 层) → output。"""

    name = "A"
    use_sel_loss = False

    def __init__(self):
        super().__init__()
        self.input = InputCard(D, V, FF, NHEAD, MAXLEN, PAD_ID, DROPOUT)
        self.target = DeepTargetCard()
        self.output = OutputCard(D, V, MAXLEN)

    def forward_all(self, ids, ctx_len=None, end_idx=None, n_cands=None):
        m = self.input.causal_mask(ids)
        h = self.input(ids, mask=m)
        h = self.target(h, mask=m)
        return self.output(h), None

    def groups(self):
        return {"input": self.input, "target2": self.target, "output": self.output}


class ArmB(nn.Module):
    """B1/B2：input(1 层) → 选择卡(1 层) → 重整卡(1 层) → output；B2 额外有选择 CE。"""

    def __init__(self, arm: str = "B1"):
        super().__init__()
        self.name = arm
        self.use_sel_loss = (arm == "B2")
        self.input = InputCard(D, V, FF, NHEAD, MAXLEN, PAD_ID, DROPOUT)
        self.select = SelCard()
        self.realign = RealignCard()
        self.output = OutputCard(D, V, MAXLEN)
        # ★零初始化：不消耗 RNG ⇒ 与 A 臂的 output/head 初值逐位相同（[INIT-CHECK] 实测）
        self.q = nn.Parameter(torch.zeros(D))

    def forward_all(self, ids, ctx_len=None, end_idx=None, n_cands=None):
        assert ctx_len is not None and end_idx is not None and n_cands is not None, \
            "[B 臂] 必须给 ctx_len / end_idx / n_cands"
        n = ids.size(1)
        m = self.input.causal_mask(ids)
        hi = self.input(ids, mask=m)
        hs = self.select(hi, mask=m)
        g = torch.gather(hs, 1, end_idx.unsqueeze(-1).expand(-1, -1, hs.size(-1)))   # (b,k,d)
        sc = (g * self.q).sum(-1)                                                  # (b,k)
        k = sc.size(1)
        valid = torch.arange(k, device=sc.device)[None, :] < n_cands[:, None]
        sc = sc.masked_fill(~valid, float("-inf"))
        p = torch.softmax(sc, dim=-1)
        e_sel = torch.einsum("bk,bkd->bd", p, g)                                   # (b,d) soft 选择
        pos = torch.arange(n, device=ids.device)[None, :]
        apply = (pos >= ctx_len[:, None]).unsqueeze(-1).to(hi.dtype)                # 只注入回复位置
        h = self.realign(hi + apply * e_sel.unsqueeze(1), mask=m)
        return self.output(h), sc

    def groups(self):
        return {"input": self.input, "select": self.select, "realign": self.realign, "output": self.output}

    def sel_pred(self, sc, recs):
        """argmax 选择 + 与「历史 (ent,cond) 一致」的命中判定（不看句面）。"""
        pred = [int(x) for x in sc.argmax(dim=-1).tolist()]
        hits = []
        for i, r in enumerate(recs):
            c = r["cands"][pred[i]] if pred[i] < r["n_cands"] else None
            hits.append(int(c is not None and c["ent"] == r["ent"] and c["cond"] == r["cond"]))
        return pred, hits

    def sel_argmax_letter_hits(self, sc, recs):
        """R3′-d 的另一半：**选择卡的 argmax 位置换算成字母后是否正确**（= pred 位置 == 正确项位置）。"""
        pred = [int(x) for x in sc.argmax(dim=-1).tolist()]
        out = []
        for i, r in enumerate(recs):
            gl = correct_letter_of(r)
            pl = CAND_LETTERS[pred[i]] if (gl is not None and pred[i] < r["n_cands"]) else None
            out.append(int(gl is not None and pl == gl))
        return pred, out


class RealignCard2(Card):
    """2 层重整卡（`--realign-layers 2` 的**参数对齐档**；0 个 RNG 占位 ⇒ layers.0/1 == A.target.layers.0/1）。"""

    io = IO_D_TO_D
    card_name = "realign2"

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT):
        super().__init__(d, maxlen)
        self.layers = nn.ModuleList([encoder_layer(d, ff, nhead, dropout) for _ in range(2)])

    def apply_card(self, x, mask=None):
        for L in self.layers:
            x = L(x, src_mask=mask)
        return x


class ArmB0(nn.Module):
    """B0a / B0b（§14）：input(1 层) → **重整卡**(1 层) → output，**无选择卡**。

    结构（两臂**逐字相同**，只差训练目标里有没有 mid ⇒ 与 §11.10 的 B1/B2 一样是干净单变量）：
      ① h0 = input(ids)（因果 mask）；
      ② 候选隐态 g_j = h0[候选末位 token 位置]（★取自 **input 卡输出**；B1/B2 取自**选择层**输出）；
      ③ **非选择性汇总**：cm = mean_j g_j（显式除以有效候选数；**固定、无参数、不学** —— §14 要求）；
      ④ h = realign(h0 + apply·cm)（apply = 位置 >= ctx 长的回复区，含 B0b 的 mid 区）；
      ⑤ logits = output(h)。
    ⇒ 两臂都没有 argmax 选择/选择 CE；mid 只是**中间目标**（B0b 由同一自回归序列输出）。
    ★RNG 对齐（配对前提）：构造顺序 = input → 第 1 个 encoder 层（**丢弃**，只为消耗与 A/B1/B2
      的 select 层同一条 RNG 流）→ 第 2 个 encoder 层（= realign，与 A.target.layers.1 / B1,B2.realign
      逐位相同）→ output ⇒ input / realign / output 与 79 的 A、B1、B2 逐张量 max|Δ|=0。
    """

    def __init__(self, arm: str = "B0a", n_layers: int = 1):
        super().__init__()
        assert arm in B0_ARMS, f"未知 B0 臂 {arm!r}"
        assert n_layers in (1, 2), n_layers
        self.name = arm
        self.variant = arm                       # "B0a"（无 mid）/ "B0b"（有 mid）
        self.use_sel_loss = False
        self.n_realign_layers = int(n_layers)
        self.inject = "mean"
        self.input = InputCard(D, V, FF, NHEAD, MAXLEN, PAD_ID, DROPOUT)
        if n_layers == 1:
            _ = RealignCard()                    # ★丢弃层：RNG 占位（对齐 A/B1/B2 的 select 层）
            self.realign = RealignCard()         # = A.target.layers.1 = B1/B2.realign
        else:
            self.realign = RealignCard2()        # = A.target.layers.0 + .1（0 个占位）
        self.output = OutputCard(D, V, MAXLEN)
        self.disable_inject = False              # [GATE] 探针用：置 True ⇒ 注入置零（通路死活检查）
        self._last_cm_norm = 0.0

    def forward_all(self, ids, ctx_len=None, end_idx=None, n_cands=None):
        assert ctx_len is not None and end_idx is not None and n_cands is not None, \
            "[B0 臂] 必须给 ctx_len / end_idx / n_cands"
        n = ids.size(1)
        m = self.input.causal_mask(ids)
        h0 = self.input(ids, mask=m)
        g = torch.gather(h0, 1, end_idx.unsqueeze(-1).expand(-1, -1, h0.size(-1)))   # (b,k,d)
        k = g.size(1)
        valid = (torch.arange(k, device=g.device)[None, :] < n_cands[:, None]).unsqueeze(-1).to(g.dtype)
        cm = (g * valid).sum(1) / n_cands[:, None].clamp(min=1).to(g.dtype)          # (b,d) 固定均值
        self._last_cm_norm = float(cm.norm(dim=-1).mean().item())
        if self.disable_inject:
            cm = torch.zeros_like(cm)
        pos = torch.arange(n, device=ids.device)[None, :]
        apply = (pos >= ctx_len[:, None]).unsqueeze(-1).to(h0.dtype)                 # 只注入回复区
        h = self.realign(h0 + apply * cm.unsqueeze(1), mask=m)
        return self.output(h), None

    def groups(self):
        return {"input": self.input, "realign": self.realign, "output": self.output}


class XRealignCard(Card):
    """r5′（§14.2）的重整卡：n_layers 个 self-attn 层 + **对候选的交叉注意**（candidates → realign）。

    ★**身份与位置保留**：key/value = **每个候选自己的隐态** g_j（j=候选在候选行里的位置），
      不是均值 ⇒ 回复区可以逐候选打分/挑中某一个（这正是 §14.1 判定"均值注入 ⇒ 空测试"的那一点）。
    ★交叉注意输出**只加在回复区**（apply 掩码：位置 >= ctx 长）；候选 padding 用 key_padding_mask 屏蔽。
    ★RNG 对齐：`pad_draws` 个 encoder 层占位（丢弃），使 output 之前恰好消耗 2 个 encoder draw
      ⇒ input / output 与 A/B1/B2 **逐位同初值**；交叉注意模块在 **output 之后**由 `build_xattn()`
      构造 ⇒ 它的初值**不会**扰动 output（实测 [INIT-CHECK] 判定）。
    """

    io = IO_D_TO_D
    card_name = "realign_x"

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, maxlen: int = MAXLEN,
                 dropout: float = DROPOUT, n_layers: int = 2, pad_draws: int = 0):
        super().__init__(d, maxlen)
        self.nhead = int(nhead)
        self.layers = nn.ModuleList([encoder_layer(d, ff, nhead, dropout) for _ in range(n_layers)])
        for _ in range(pad_draws):               # ★RNG 占位（丢弃）：对齐 output 与 A 的初值
            encoder_layer(d, ff, nhead, dropout)
        self.xattn = None
        self.xnorm = None

    def build_xattn(self):
        """在 output 卡构造**之后**调用（否则会扰动 output 的初值）。"""
        self.xattn = nn.MultiheadAttention(self.d, self.nhead, dropout=DROPOUT, batch_first=True)
        self.xnorm = nn.LayerNorm(self.d)

    def forward(self, x, mask=None, cand=None, cand_valid=None, apply=None):
        self.assert_input(x)
        out = self.apply_card(x, mask, cand, cand_valid, apply)
        self.assert_output(x, out)
        return out

    def apply_card(self, x, mask=None, cand=None, cand_valid=None, apply=None):
        for L in self.layers:
            x = L(x, src_mask=mask)
        if cand is not None and self.xattn is not None:
            key_pad = None if cand_valid is None else ~cand_valid
            a, _ = self.xattn(x, cand, cand, key_padding_mask=key_pad, need_weights=False)
            if apply is not None:
                a = a * apply
            x = self.xnorm(x + a)
        return x


class ArmB0x(nn.Module):
    """r5′：input(1 层) → **对候选交叉注意的重整卡**(n_layers 层) → output；无选择卡。

    与 ArmB0 的唯一结构差别 = 候选注入方式（**逐候选交叉注意**取代均值）⇒ §14.2 的可判设计。
    两臂 B0pa / B0pb **逐字相同**（参数与初值逐位相同），只差训练目标里有没有 mid ⇒ 干净单变量。
    """

    def __init__(self, arm: str = "B0pa", n_layers: int = 2):
        super().__init__()
        assert arm in B0X_ARMS, f"未知 r5′ 臂 {arm!r}"
        assert n_layers in (1, 2), n_layers
        self.name = arm
        self.variant = "B0b" if arm == "B0pb" else "A"
        self.use_sel_loss = False
        self.n_realign_layers = int(n_layers)
        self.inject = "xattn"
        self.input = InputCard(D, V, FF, NHEAD, MAXLEN, PAD_ID, DROPOUT)
        # ★与 A 对齐：output 之前恰好消耗 2 个 encoder 层 draw（用 n_layers 个，其余占位丢弃）
        self.realign = XRealignCard(n_layers=n_layers, pad_draws=max(0, 2 - n_layers))
        self.output = OutputCard(D, V, MAXLEN)
        self.realign.build_xattn()               # ★必须在 output 之后
        self.disable_inject = False
        self._last_cm_norm = 0.0

    def forward_all(self, ids, ctx_len=None, end_idx=None, n_cands=None):
        assert ctx_len is not None and end_idx is not None and n_cands is not None, \
            "[B0′ 臂] 必须给 ctx_len / end_idx / n_cands"
        n = ids.size(1)
        m = self.input.causal_mask(ids)
        h0 = self.input(ids, mask=m)
        g = torch.gather(h0, 1, end_idx.unsqueeze(-1).expand(-1, -1, h0.size(-1)))   # (b,k,d) 逐候选
        k = g.size(1)
        valid = torch.arange(k, device=g.device)[None, :] < n_cands[:, None]         # (b,k) bool
        pos = torch.arange(n, device=ids.device)[None, :]
        apply = (pos >= ctx_len[:, None]).unsqueeze(-1).to(h0.dtype)                 # 只注入回复区
        if self.disable_inject:                                                      # [GATE] 探针用
            apply = torch.zeros_like(apply)
        h = self.realign.forward(h0, mask=m, cand=g, cand_valid=valid, apply=apply)
        self._last_cm_norm = float(g.norm(dim=-1).mean().item())
        return self.output(h), None

    def groups(self):
        return {"input": self.input, "realign": self.realign, "output": self.output}


def build_arm(arm: str, n_layers: int | None = None) -> nn.Module:
    """构造顺序三臂一致：input → 第 1 层 → 第 2 层 → output（B 的 q 零初始化不消耗 RNG）。

    B0a/B0b 用**丢弃层**占住第 1 层的位置（见 ArmB0 的 RNG 对齐说明）⇒ 共享子模块逐位同初值。
    """
    nl = REALIGN_LAYERS if n_layers is None else int(n_layers)
    if arm == "A":
        return ArmA()
    if arm in B0_ARMS:
        return ArmB0(arm, n_layers=nl)
    if arm in B0X_ARMS:
        return ArmB0x(arm, n_layers=nl)
    assert arm in ("B1", "B2"), f"未知臂 {arm!r}"
    return ArmB(arm)


def param_groups_named(model) -> dict:
    g = {k: sum(p.numel() for p in mod.parameters()) for k, mod in model.groups().items()}
    if isinstance(model, ArmB):
        g["q"] = model.q.numel()
    g["合计"] = sum(p.numel() for p in model.parameters())
    return g


# ---------------- 训练 / 评测骨架（写法照抄 S73） ----------------
def reply_ce(logits, ids, recs, s) -> torch.Tensor:
    logp = torch.log_softmax(logits, dim=-1)
    nll = torch.zeros((), device=logits.device)
    ntok = 0
    for i, r in enumerate(recs):
        a, e = int(s[i]), int(s[i]) + len(r["t"])
        lp = logp[i, a - 1: e - 1]
        tgt = ids[i, a: e]
        nll = nll - lp.gather(1, tgt.unsqueeze(1)).sum()
        ntok += len(r["t"])
    return nll / ntok


def train_run(model, train, seed: int, steps: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    opt = AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    gen = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(train), generator=gen)
    pos, step, gate = 0, 0, None
    t0 = time.time()
    while step < steps:
        idx = []
        for _ in range(BATCH):
            if pos >= len(order):
                order = torch.randperm(len(train), generator=gen)
                pos = 0
            idx.append(int(order[pos]))
            pos += 1
        batch = [train[i] for i in idx]
        ids, s, eidx, nc = pack(batch)
        logits, sc = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)
        lce = reply_ce(logits, ids, batch, s)
        loss = lce
        lsel = None
        if model.use_sel_loss:
            gold = torch.tensor([r["gold_idx"] for r in batch], device=device)
            lsel = F.cross_entropy(sc, gold)
            loss = lce + SEL_LAMBDA * lsel
        opt.zero_grad(set_to_none=True)
        loss.backward()
        step += 1
        if step == 20:
            parts = []
            for name, mod in model.groups().items():
                ps = list(mod.parameters())
                got = sum(p.grad is not None for p in ps)
                gm = max((float(p.grad.abs().max()) for p in ps if p.grad is not None), default=0.0)
                parts.append(f"{name}: {got}/{len(ps)} 张量有梯度 max|g|={gm:.2e}")
            if isinstance(model, ArmB):
                qg = model.q.grad
                parts.append(f"q: {'有' if qg is not None else '无'}梯度"
                             + (f" max|g|={float(qg.abs().max()):.2e}" if qg is not None else ""))
                if model.use_sel_loss:
                    pred, hits = model.sel_pred(sc, batch)
                    parts.append(f"batch 选择准确率={100.0 * sum(hits) / len(hits):.2f}%")
                else:
                    selg = max((float(p.grad.abs().max()) for p in model.select.parameters()
                                if p.grad is not None), default=0.0)
                    parts.append(f"选择卡梯度来源=仅回复损失（无中间监督）max|g|={selg:.2e}"
                                 f" ⇒ {'通路活' if selg > 0 else '★通路死'}")
            elif isinstance(model, (ArmB0, ArmB0x)):
                # ★通路活性（§14 的仪器自检）：把**非选择性注入**置零，比较 logits 变化。
                # no_grad + eval：不消耗 dropout RNG、不动训练随机流，也不建图。
                # ★只喂 **1 条**样本：本探针在训练步内做，而本步的 logits/log_softmax（32×n×V）
                #   仍被引用 ⇒ 用整批会把峰值显存抬高 ~350MB（实测 2106→2524MiB），污染 R5-e 的峰值对比。
                was_training = model.training
                model.eval()
                with torch.no_grad():
                    lg_on = model.forward_all(ids[:1], ctx_len=s[:1], end_idx=eidx[:1], n_cands=nc[:1])[0]
                    model.disable_inject = True
                    lg_off = model.forward_all(ids[:1], ctx_len=s[:1], end_idx=eidx[:1], n_cands=nc[:1])[0]
                    model.disable_inject = False
                if was_training:
                    model.train()
                dinj = float((lg_on - lg_off).abs().max().item())
                del lg_on, lg_off
                inj = getattr(model, "inject", "mean")
                parts.append(f"{'非选择性注入(固定均值|无参数)' if inj == 'mean' else '候选交叉注意(逐候选 K/V|保留身份)'}"
                             f" 通路 max|Δlogits|(1 条样本)={dinj:.3e}"
                             f" ⇒ {'通路活' if dinj > 0 else '★通路死'}")
            gate = " | ".join(parts)
            # ★不用子串判据（"30/30" 也含 "0/" ⇒ 79 那版写法在张量数 >=10 的组上会假 ★CHECK）；
            #   改成逐组精确比较「有梯度的张量数 == 张量数」。
            ok = all(sum(p.grad is not None for p in mod.parameters()) == len(list(mod.parameters()))
                     for mod in model.groups().values()) and ("通路死" not in gate)
            print(f"[GATE] step20 {gate} ⇒ {'PASS' if ok else '★CHECK'}", flush=True)
        opt.step()
        if step % 500 == 0 or step == steps:
            el = time.time() - t0
            extra = f" sel={float(lsel):.4f}" if lsel is not None else ""
            print(f"  [train] seed={seed} step={step}/{steps} loss={loss.item():.4f}{extra} "
                  f"ce={lce.item():.4f} elapsed={el:.0f}s eta={el / step * (steps - step):.0f}s", flush=True)
    return gate, time.time() - t0


@torch.no_grad()
def eval_ce(model, recs) -> list:
    model.eval()
    per = []
    for i in range(0, len(recs), BATCH):
        chunk = recs[i: i + BATCH]
        ids, s, eidx, nc = pack(chunk)
        logits = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)[0]
        logp = torch.log_softmax(logits, dim=-1)
        for j, r in enumerate(chunk):
            a, e = int(s[j]), int(s[j]) + len(r["t"])
            lp = logp[j, a - 1: e - 1]
            tgt = ids[j, a: e]
            per.append(float((-lp.gather(1, tgt.unsqueeze(1)).sum() / len(r["t"])).item()))
    model.train()
    return per


@torch.no_grad()
def eval_sel(model, recs):
    """选择卡读数（只喂 ctx；与训练同口径的 p）。A 臂返回 None。

    返回 dict：hits=argmax 命中的候选对 == 历史 (ent,cond)（不看句面）；
               letter_hits=argmax 位置换算的字母 == 「候选集里唯一双槽位匹配项的字母」（R3′-d）。
    """
    if not isinstance(model, ArmB):
        return None
    model.eval()
    hits, lhits = [], []
    for i in range(0, len(recs), BATCH):
        chunk = recs[i: i + BATCH]
        ids, s, eidx, nc = pack(chunk, with_target=False)
        sc = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)[1]
        _pred, h = model.sel_pred(sc, chunk)
        _p2, lh = model.sel_argmax_letter_hits(sc, chunk)
        hits += h
        lhits += lh
    model.train()
    return {"hits": hits, "letter_hits": lhits}


@torch.no_grad()
def greedy_gen(model, recs) -> list:
    """贪心解码 batch=1（照 S73/S71 的 EM_BATCH=1 自洽口径）；生成时用模型自己的选择（非金标）。"""
    model.eval()
    texts = []
    for r in recs:
        ids = torch.tensor([r["p"]], dtype=torch.long, device=device)
        s = torch.tensor([len(r["p"])], dtype=torch.long, device=device)
        eidx = torch.tensor([r["cend"]], dtype=torch.long, device=device)
        nc = torch.tensor([r["n_cands"]], dtype=torch.long, device=device)
        cap = MAXLEN - len(r["p"])
        gen = []
        for _ in range(cap):
            logits = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)[0]
            t = int(logits[0, -1].argmax().item())
            if t == EOS_ID:
                break
            gen.append(t)
            ids = torch.cat([ids, torch.tensor([[t]], dtype=torch.long, device=device)], dim=1)
        texts.append(dec(gen))
    model.train()
    return texts


@torch.no_grad()
def greedy_gen_b0b(model, recs):
    """B0b 的两段贪心（§14：「先输出匹配到的有序模因项，再输出回复」）。

    ① 从 ctx 生成 mid（遇 SEP/EOS 或满 MID_CAP 停；SEP **不算**进 mid 文本）；
    ② **强制补一个 SEP**，再生成回复（到 EOS 停）—— 回复只依赖模型**自己**写出的 mid token
       （与 §11.11「生成时用模型自己的 p，不是金标」同口径）。
    返回 (mids, replies, sep_emitted)：mid 文本 / 回复文本 / 第一段是否自己写出了 SEP（仪器读数）。
    """
    model.eval()
    mids, replies, sep_ok = [], [], []
    for r in recs:
        ids = torch.tensor([r["p"]], dtype=torch.long, device=device)
        s = torch.tensor([len(r["p"])], dtype=torch.long, device=device)
        eidx = torch.tensor([r["cend"]], dtype=torch.long, device=device)
        nc = torch.tensor([r["n_cands"]], dtype=torch.long, device=device)
        mid_ids, got_sep = [], 0
        for _ in range(MID_CAP):
            logits = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)[0]
            t = int(logits[0, -1].argmax().item())
            if t == SEP_ID:
                got_sep = 1
                break
            if t == EOS_ID:
                break
            mid_ids.append(t)
            ids = torch.cat([ids, torch.tensor([[t]], dtype=torch.long, device=device)], dim=1)
        ids = torch.cat([ids, torch.tensor([[SEP_ID]], dtype=torch.long, device=device)], dim=1)
        gen = []
        for _ in range(MAXLEN - ids.size(1)):
            logits = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)[0]
            t = int(logits[0, -1].argmax().item())
            if t == EOS_ID:
                break
            gen.append(t)
            ids = torch.cat([ids, torch.tensor([[t]], dtype=torch.long, device=device)], dim=1)
        mids.append(dec(mid_ids))
        replies.append(dec(gen))
        sep_ok.append(got_sep)
    model.train()
    return mids, replies, sep_ok


@torch.no_grad()
def eval_ce_split(model, recs):
    """B0b 专用：**分段 CE**（mid+SEP 段 / 回复段），nats/token，逐条；用于看中间目标学得怎样。"""
    model.eval()
    mids, reps = [], []
    for i in range(0, len(recs), BATCH):
        chunk = recs[i: i + BATCH]
        ids, s, eidx, nc = pack(chunk)
        logits = model.forward_all(ids, ctx_len=s, end_idx=eidx, n_cands=nc)[0]
        logp = torch.log_softmax(logits, dim=-1)
        for j, r in enumerate(chunk):
            a, ml, e = int(s[j]), int(r.get("mid_len", 1)), int(s[j]) + len(r["t"])
            lp_m = logp[j, a - 1: a + ml - 1]
            tgt_m = ids[j, a: a + ml]
            lp_r = logp[j, a + ml - 1: e - 1]
            tgt_r = ids[j, a + ml: e]
            mids.append(float((-lp_m.gather(1, tgt_m.unsqueeze(1)).sum() / max(1, ml)).item()))
            reps.append(float((-lp_r.gather(1, tgt_r.unsqueeze(1)).sum() / max(1, len(r["t"]) - ml)).item()))
    model.train()
    return mids, reps


# ---------------- 统计口径 ----------------
def mean_se(xs):
    n = len(xs)
    m = sum(xs) / n
    var = sum((x - m) ** 2 for x in xs) / (n - 1) if n > 1 else 0.0
    return m, (math.sqrt(var / n) if n else float("nan"))


def em_strict(gold: str, pred: str) -> int:
    return int((gold or "").strip() == (pred or "").strip())


def em_with_thr(p: float, n: int):
    se = (p / 100.0 * (1 - p / 100.0) / n) ** 0.5 * 100.0 if n else float("nan")
    return se, p + 2.0 * se


def mcnemar_exact(n01: int, n10: int) -> float:
    """McNemar 精确双侧 p（只看不一致对；n01=臂Y对而臂X错，n10=臂X对而臂Y错）。"""
    nd = n01 + n10
    if nd == 0:
        return 1.0
    m = min(n01, n10)
    tail = sum(math.comb(nd, i) for i in range(m + 1)) * (0.5 ** nd)
    return min(1.0, 2.0 * tail)


def paired_stats(hx, hy):
    """§11.4 的配对口径：d_i = 1[臂X 对 i] − 1[臂Y 对 i]；SE_Δ = std(d)/sqrt(n)（不是独立样本公式）。"""
    n = len(hx)
    d = [a - b for a, b in zip(hx, hy)]
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1) if n > 1 else 0.0
    se = math.sqrt(var / n)
    n01 = sum(1 for a, b in zip(hx, hy) if a == 0 and b == 1)     # X 错、Y 对
    n10 = sum(1 for a, b in zip(hx, hy) if a == 1 and b == 0)     # X 对、Y 错
    return {"delta": 100.0 * m, "se": 100.0 * se, "n01": n01, "n10": n10,
            "p": mcnemar_exact(n01, n10), "n": n}


def paired_line(tag: str, hx, hy, name_x: str, name_y: str):
    st = paired_stats(hx, hy)
    print(f"[PAIR] {tag} {name_x} vs {name_y}：Δ={st['delta']:+.2f}pp SE_Δ(配对)={st['se']:.2f}pp "
          f"2SE_Δ={2 * st['se']:.2f}pp | 不一致对 b({name_x}错{name_y}对)={st['n01']} "
          f"c({name_x}对{name_y}错)={st['n10']} | McNemar 双侧精确 p={st['p']:.3g} (n={st['n']})", flush=True)
    return st


def paired_letters(hx, hy):
    """把「生成文本」按 T1 口径抽字母后逐条配对（抽不出 = None ⇒ 记为错；同时报抽不出计数）。"""
    lx, ly = [dec_letter(t) for t in hx], [dec_letter(t) for t in hy]
    mx = [int(x is None) for x in lx]
    my = [int(y is None) for y in ly]
    return lx, ly, mx, my


# ---------------- §11.5 的两套门槛（T1 用机会级 1/k；T2 用捷径级 max_naive） ----------------
def t1_threshold(k: int, n: int):
    """T1（字母准确率）门槛 = 机会级 1/k + 2SE(1/k)。"""
    p = 100.0 / k
    se = (p / 100.0 * (1 - p / 100.0) / n) ** 0.5 * 100.0 if n else float("nan")
    return p, se, p + 2.0 * se


def t2_threshold(max_naive: float, n: int):
    """T2（完整回复 EM）门槛 = 捷径级 max_naive + 2SE(max_naive)。"""
    se = (max_naive / 100.0 * (1 - max_naive / 100.0) / n) ** 0.5 * 100.0 if n else float("nan")
    return max_naive, se, max_naive + 2.0 * se


def report_texts(hits, txt, recs, tag: str):
    """T1 + T2 的一次汇总（原始行）；返回 dict（含逐条字母，供配对与落盘）。"""
    n = len(txt)
    ls = [dec_letter(t) for t in txt]
    miss = sum(1 for x in ls if x is None)
    gold_l = [gold_letter_of(r) for r in recs]
    usable = [(g, x) for g, x in zip(gold_l, ls) if g is not None]
    t1 = 100.0 * sum(1 for g, x in usable if x == g) / len(usable) if usable else float("nan")
    t2 = 100.0 * sum(hits) / n
    strict_ok = sum(1 for g, x in usable if x == g) if usable else 0
    print(f"[{tag}] n={n} | T1 字母准确率={t1:.2f}%（抽出成功 {n - miss}/{n}，"
          f"★抽不出比例={100.0 * miss / n:.2f}%）| T1 明细 字母命中={strict_ok}/{len(usable)}"
          f" | T2 完整回复 EM={t2:.2f}%（逐字，含标签与句号）", flush=True)
    return {"ls": ls, "miss": miss, "t1": t1, "t2": t2, "gold_l": gold_l}


def report_mid(mids, recs, sep_ok=None, tag: str = "T_MID"):
    """B0b 的**中间目标**读数（T_mid）：mid 逐字 EM + mid 字母命中 + SEP 自出率。

    动态金标口径（§11.17）：gold mid 的字母 = 当前候选集里唯一双槽位匹配项的**位置字母**。
    """
    n = len(mids)
    gold = [mid_gold_of(r) for r in recs]
    usable = [(g, m) for g, m in zip(gold, mids) if g is not None]
    em_mid = (100.0 * sum(1 for g, m in usable if (g or "").strip() == (m or "").strip()) / len(usable)
              if usable else float("nan"))
    gl = [correct_letter_of(r) for r in recs]
    ls = [dec_letter(m) for m in mids]
    u2 = [(g, x) for g, x in zip(gl, ls) if g is not None]
    mid_letter = 100.0 * sum(1 for g, x in u2 if x == g) / len(u2) if u2 else float("nan")
    miss = sum(1 for x in ls if x is None)
    sep_note = f" | SEP 自出率={100.0 * sum(sep_ok) / len(sep_ok):.2f}%（其余为强制补 SEP）" if sep_ok else ""
    print(f"[{tag}] n={n} | mid 逐字 EM={em_mid:.2f}%（可判 {len(usable)}/{n}）| mid 字母命中率={mid_letter:.2f}%"
          f"（抽不出={100.0 * miss / n:.2f}%）{sep_note}", flush=True)
    return {"em_mid": em_mid, "mid_letter": mid_letter, "miss": miss, "gold_mid": gold, "ls": ls}


# ---------------- 反事实（R3-c，沿用 79 三条；B0b 增加 mid 口径） ----------------
CF_MODE_ID = {"shuffle": 1, "replace": 2, "delete": 3}
def cf_record(r: dict, mode: str, rng: random.Random, variant: str = "A") -> dict:
    """按模式重建候选行与 ctx；返回新记录（cend/n_cands 由 prep_record 重算）。"""
    cands = [dict(c) for c in r["cands"]]
    gold = r["gold_idx"]
    new, gi = cands, gold
    if mode == "shuffle":
        perm = list(range(len(cands)))
        rng.shuffle(perm)
        new = [cands[j] for j in perm]
        gi = perm.index(gold)
    elif mode == "replace":
        # 正确候选换成「历史不一致」的对：优先选两个槽位都对不上的组合
        pool = []
        for c in cands:
            for c2 in cands:
                pair = (c["ent"], c2["cond"])
                if pair != (r["ent"], r["cond"]):
                    pool.append(pair)
        strong = [p for p in pool if p[0] != r["ent"] and p[1] != r["cond"]]
        pick = rng.choice(strong if strong else pool)
        new = [dict(c) for c in cands]
        new[gold]["ent"], new[gold]["cond"] = pick
        gi = gold
    elif mode == "delete":
        new = [c for j, c in enumerate(cands) if j != gold]
        gi = -1
    else:                                                       # pragma: no cover
        raise AssertionError(f"未知反事实模式 {mode!r}")
    for pos, c in enumerate(new):
        c["letter"] = CAND_LETTERS[pos]
        c["text"] = f"{c['letter']}. {c['ent']} / {c['cond']}"
    line = CAND_PREFIX + CAND_SEP.join(c["text"] for c in new)
    out = dict(r)
    out["ctx"] = r["ctx"].replace(r["cand_line"], line)
    out["cands"] = new
    out["cand_line"] = line
    out["gold_idx"] = gi
    out["n_cands"] = len(new)
    out.pop("t", None)                                          # 回复目标不变，但强制重算 token（ctx 变了）
    for kk in ("mid", "tmid", "tsep", "treply", "mid_len"):      # ★B0b：mid 也必须按新候选集重算（动态金标）
        out.pop(kk, None)
    return prep_record(out, variant=variant)


def run_cf(model, test, base_txt, mode: str, seed: int, tag: str, variant: str = "A", base_mid=None):
    """对同一 ckpt 做一次反事实评测并打印原始行（★按**标签口径**复算，见文件头改动清单 3）。

    要点：replace/delete 之后「原 target」已**不是**该候选集的正确回复（原 target 里写的是被改动/
    被删除那个候选的字母）⇒ 动态金标 = 用「候选集里唯一双槽位匹配项」按位置复算的 (DET句面, 该字母)。
    ★shuffle 下**回复逐字必然要变**（正确项换位置 ⇒ 它带着自己的字母一起换）⇒ 「答案不变」在**带字母的
    数据上**的正确读法是「**选中的还是同一个（实体,条件）**」，即 `dyn` 不塌：
    `dyn`（对动态金标的逐字 EM，**回复段**）就是「还选中同一个、且按新位置报了它的字母」；逐字不变率只作旁证。
    ★B0b：mid 与回复都按同一个动态金标重算（`dyn_mid` / `dyn_e2e`）；`dyn` 仍是**回复段**口径
    ⇒ 与 B0a 及 79 的 A/B1/B2 **可比**。
    返回 dict（hits=对原 target 的逐字 EM；dyn=对动态金标的回复段逐字 EM；letters/sel/…）。
    """
    rng = random.Random(0xC0FFEE ^ seed ^ CF_MODE_ID[mode])   # ★跨臂确定：同 (mode,seed) 各臂见同一扰动输入
    recs = [cf_record(r, mode, rng, variant) for r in test]
    moved = sum(1 for r0, r1 in zip(test, recs) if r0["gold_idx"] != r1["gold_idx"])
    cand_changed = sum(1 for r0, r1 in zip(test, recs) if r0["cand_line"] != r1["cand_line"])
    dyn_gold = [(d76._reference_reply(r)[0] or "") for r in recs]
    dyn_letter = [correct_letter_of(r) for r in recs]
    dyn_mid = [mid_gold_of(r) for r in recs]
    n_noright = sum(1 for x in dyn_letter if x is None)
    sep_ok = None
    if variant == "B0b":
        mids, txt, sep_ok = greedy_gen_b0b(model, recs)
    else:
        txt = greedy_gen(model, recs)
        mids = [None] * len(recs)
    hits = [em_strict(r["target"], txt[i]) for i, r in enumerate(recs)]
    dyn = [em_strict(dyn_gold[i], txt[i]) for i in range(len(recs))]
    dyn_e2e = [em_strict(((dyn_mid[i] or "") + B0_SEP + dyn_gold[i]), (mids[i] or "") + B0_SEP + txt[i])
               for i in range(len(recs))] if variant == "B0b" else []
    dyn_mid_hit = [em_strict(dyn_mid[i] or "", mids[i] or "") for i in range(len(recs))] if variant == "B0b" else []
    letters = [dec_letter(t) for t in txt]
    miss = sum(1 for x in letters if x is None)
    same_letter = sum(1 for i in range(len(recs))
                      if letters[i] is not None and dec_letter(base_txt[i]) is not None
                      and letters[i] == dec_letter(base_txt[i]))
    dyn_letter_hit = sum(1 for i in range(len(recs))
                         if dyn_letter[i] is not None and letters[i] == dyn_letter[i])
    n_dyn_valid = sum(1 for x in dyn_letter if x is not None)
    changed = sum(1 for i in range(len(recs)) if txt[i].strip() != base_txt[i].strip())
    em = 100.0 * sum(hits) / len(hits)
    em_dyn = 100.0 * sum(dyn) / len(dyn)
    sel = eval_sel(model, recs)
    mid_note = ""
    if variant == "B0b":
        n_valid_mid = sum(1 for x in dyn_mid if x is not None)
        mid_note = (f" | ★EM(mid,对动态金标)={100.0 * sum(dyn_mid_hit) / max(1, len(dyn_mid_hit)):.2f}%"
                    f"（可判 {n_valid_mid}/{len(recs)}）| ★端到端 EM(对动态金标)="
                    f"{100.0 * sum(dyn_e2e) / max(1, len(dyn_e2e)):.2f}%"
                    f" | SEP 自出率={100.0 * sum(sep_ok) / len(sep_ok):.2f}%")
    print(f"[CF] 模式={mode} ({tag}) n={len(recs)} | 候选行被改动={cand_changed} | 正确项位置发生变化={moved} | "
          f"候选集里没有正确对的条数={n_noright} | EM(对原 target)={em:.2f}% | ★EM(对动态金标)={em_dyn:.2f}%"
          f"{'（动态金标不可判：无正确对）' if n_dyn_valid == 0 else ''} | ★回复文本变化率(vs 本 run 原始)="
          f"{100.0 * changed / len(recs):.2f}% | 标签字母抽不出={miss} | ★字母不变率={100.0 * same_letter / len(recs):.2f}%"
          f" | 动态正确字母命中率="
          + ("不适用(无正确对)" if n_dyn_valid == 0 else
             f"{100.0 * dyn_letter_hit / n_dyn_valid:.2f}%（{dyn_letter_hit}/{n_dyn_valid}）")
          + mid_note
          + " | 选择准确率(选对候选对)="
          + ("不适用(本臂无选择卡)" if sel is None else f"{100.0 * sum(sel['hits']) / len(sel['hits']):.2f}%")
          + " | 选择 argmax 字母正确率="
          + ("不适用(本臂无选择卡)" if sel is None else
             f"{100.0 * sum(sel['letter_hits']) / len(sel['letter_hits']):.2f}%"), flush=True)
    return {"hits": hits, "dyn": dyn, "txt": txt, "letters": letters, "dyn_letter": dyn_letter,
            "mids": mids, "dyn_mid": dyn_mid, "dyn_mid_hit": dyn_mid_hit, "dyn_e2e": dyn_e2e,
            "sep_ok": sep_ok, "sel": sel, "moved": moved, "cand_changed": cand_changed, "em": em,
            "em_dyn": em_dyn, "changed": changed, "same_letter": same_letter, "miss": miss,
            "dyn_letter_hit": dyn_letter_hit, "n_dyn_valid": n_dyn_valid, "recs": recs}


# ---------------- 参数表 / 初值核对（CPU） ----------------
def _state_pairs(sa: dict, sb: dict, mapping: list):
    """按 (前缀 A, 前缀 B) 列表逐张量求 max|Δ|；返回 {说明: Δ}。"""
    out = {}
    for ka, kb in mapping:
        for k, v in sa.items():
            if k.startswith(ka):
                tail = k[len(ka):]
                out[f"{ka}{tail} ↔ {kb}{tail}"] = float((v - sb[kb + tail]).abs().max().item())
    return out


def build_all_arms(seed: int, n_layers: int = REALIGN_LAYERS) -> dict:
    """同 seed 构造 81 的 B0a/B0b/B0pa/B0pb（本档层数）+ 81 自己的 A/B1/B2 + **79 文件里的** A/B1/B2。"""
    built = {}
    for arm in ("B0a", "B0b", "B0pa", "B0pb", "A", "B1", "B2"):
        torch.manual_seed(seed)
        built[arm] = build_arm(arm, n_layers=n_layers)
    for arm in ("A", "B1", "B2"):
        torch.manual_seed(seed)
        built[f"79.{arm}"] = d79.build_arm(arm)
    return built


def shared_init_check(seed: int, n_layers: int = REALIGN_LAYERS) -> float:
    """★R5-a / R5′-a 的仪器（本文件修了 79 的判词缺陷：返回 **worst float**，不再返回 dict）。

    逐张量核对三组**参与比较的**共享子模块初值（同 seed 构造）：
      ① B0a ↔ B0b（r5）与 B0pa ↔ B0pb（r5′）：**全部**张量（各自两臂结构逐字相同）；
      ② 均值档 B0a ↔ 79 的 A / B1 / B2：input / output / realign 层（n_layers=1 时 realign.enc，
         n_layers=2 时 realign.layers.0/1 ↔ target.layers.0/1 或 select.enc/realign.enc）；
      ③ 交叉注意档 B0pa ↔ 79 的 A（input / output / 自注意层 i ↔ target.layers.i）与
         B0pa ↔ 79 的 B1/B2（**只** input / output；r5′ 的 realign 是含交叉注意的**不同结构** ⇒ 无对应张量）。
    另做 §9.7 的同 seed 两次构造（sha 相同、Δ=0）与异 seed 阴性对照；并核对 81 自己的 A/B1/B2 副本
    与 79 的构造**逐位相同**（[COPY-CHECK]）。
    ★RNG 对齐的做法：B0/B0′ 在 output 之前恰好消耗 **2 个 encoder 层 draw**（用了 n_layers 个，
      其余占位丢弃）；r5′ 的交叉注意模块在 output **之后**构造 ⇒ 不扰动 output 初值。
    """
    built = build_all_arms(seed, n_layers)
    s = {k: v.state_dict() for k, v in built.items()}
    worst = 0.0
    for p1, p2 in (("B0a", "B0b"), ("B0pa", "B0pb")):
        d_full = max((float((a - b).abs().max().item())
                      for a, b in zip(s[p1].values(), s[p2].values())), default=0.0)
        worst = max(worst, d_full)
        print(f"[INIT-CHECK] {p1} ↔ {p2} 全张量 max|Δ|={d_full:.3e}（{len(s[p1])} 个张量）⇒ "
              f"{'PASS 逐位相同（只差训练目标）' if d_full == 0.0 else '★FAIL'}", flush=True)
    if n_layers == 1:
        mean_b = [("input.", "input."), ("realign.enc.", "realign.enc."), ("output.", "output.")]
        mean_a = [("input.", "input."), ("realign.enc.", "target.layers.1."), ("output.", "output.")]
    else:
        mean_b = [("input.", "input."), ("realign.layers.0.", "select.enc."),
                  ("realign.layers.1.", "realign.enc."), ("output.", "output.")]
        mean_a = [("input.", "input."), ("realign.layers.0.", "target.layers.0."),
                  ("realign.layers.1.", "target.layers.1."), ("output.", "output.")]
    x_map_a = [("input.", "input."), ("output.", "output.")] + \
              [(f"realign.layers.{i}.", f"target.layers.{i}.") for i in range(n_layers)]
    for name in ("79.A", "79.B1", "79.B2"):
        dd = _state_pairs(s["B0a"], s[name], mean_a if name == "79.A" else mean_b)
        w = max(dd.values()) if dd else float("nan")
        worst = max(worst, 0.0 if w != w else w)
        print(f"[INIT-CHECK] B0a ↔ {name} 共享子模块逐张量 max|Δ|={w:.3e}（{len(dd)} 个张量）⇒ "
              f"{'PASS 逐位同初值' if w == 0.0 else '★FAIL 共享子模块初值不同 ⇒ 配对作废'}", flush=True)
    dd = _state_pairs(s["B0pa"], s["79.A"], x_map_a)
    w = max(dd.values()) if dd else float("nan")
    worst = max(worst, 0.0 if w != w else w)
    print(f"[INIT-CHECK] B0pa ↔ 79.A 共享子模块（input/output/自注意层）逐张量 max|Δ|={w:.3e}"
          f"（{len(dd)} 个张量）⇒ {'PASS 逐位同初值' if w == 0.0 else '★FAIL'}", flush=True)
    for name in ("79.B1", "79.B2"):
        dd = _state_pairs(s["B0pa"], s[name], [("input.", "input."), ("output.", "output.")])
        w = max(dd.values()) if dd else float("nan")
        worst = max(worst, 0.0 if w != w else w)
        print(f"[INIT-CHECK] B0pa ↔ {name} 共享子模块（**只** input/output；r5′ 的 realign 是含交叉注意的"
              f"不同结构）逐张量 max|Δ|={w:.3e}（{len(dd)} 个张量）⇒ "
              f"{'PASS 逐位同初值' if w == 0.0 else '★FAIL'}", flush=True)
    rng_note = (f"（xattn 在 output 之后构造 ⇒ 不扰动 output；占位 draw={max(0, 2 - n_layers)}）"
                if n_layers == 1 else "（xattn 在 output 之后构造；占位 draw=0）")
    for arm in ("B0a", "B0b", "B0pa", "B0pb", "A", "B1", "B2"):
        torch.manual_seed(seed)
        m1 = build_arm(arm, n_layers=n_layers)
        h1 = state_sha256(m1.state_dict())
        torch.manual_seed(seed)
        m2 = build_arm(arm, n_layers=n_layers)
        h2 = state_sha256(m2.state_dict())
        dm = max((float((a - b).abs().max().item()) for a, b in
                  zip(m1.state_dict().values(), m2.state_dict().values())), default=0.0)
        torch.manual_seed(seed ^ 0x5EED)
        h3 = state_sha256(build_arm(arm, n_layers=n_layers).state_dict())
        print(f"[INIT-CHECK] arm={arm}(L{n_layers}) 同 seed 两次构造 sha={h1[:16]}/{h2[:16]} ⇒ "
              f"{'相同' if h1 == h2 else '★不同'} max|Δ|={dm:.1e} | 异 seed sha={h3[:16]} ⇒ "
              f"{'不同（阴性对照通过）' if h3 != h1 else '★相同（seed 没生效）'}{rng_note}", flush=True)
    for arm in ("A", "B1", "B2"):
        dcp = max((float((a - b).abs().max().item())
                   for a, b in zip(s[arm].values(), s[f"79.{arm}"].values())), default=0.0)
        print(f"[COPY-CHECK] 81 的 {arm} 副本 ↔ 79 的 {arm} 逐张量 max|Δ|={dcp:.3e}"
              f"（sha 81={state_sha256(s[arm])[:16]} 79={state_sha256(s[f'79.{arm}'])[:16]}）⇒ "
              f"{'PASS 副本未漂移' if dcp == 0.0 else '★FAIL 副本漂移 ⇒ 骨架改动越界'}", flush=True)
    return float(worst)


def param_table() -> int:
    print("[PARAMS] 参数对齐表（同 seed=1234 构造；A/B1/B2 与 --realign-layers 无关）", flush=True)
    rows = {}
    for arm in ("A", "B1", "B2"):
        torch.manual_seed(1234)
        rows[(arm, 1)] = param_groups_named(build_arm(arm, n_layers=1))
    for nl in (1, 2):
        for arm in ("B0a", "B0b", "B0pa", "B0pb"):
            torch.manual_seed(1234)
            rows[(arm, nl)] = param_groups_named(build_arm(arm, n_layers=nl))
    base = rows[("A", 1)]["合计"]
    keys = ["input", "target2", "select", "realign", "output", "q", "合计"]
    for nl in (1, 2):
        arms = ("A", "B1", "B2", "B0a", "B0b", "B0pa", "B0pb")
        print(f"[PARAMS] ---- realign-layers={nl}（判据档={2 if nl == 2 else '旁证'}）----", flush=True)
        print("[PARAMS] 子模块 | " + " | ".join(arms), flush=True)
        for k in keys:
            vals = [rows[(a, nl if a in ("B0a", "B0b", "B0pa", "B0pb") else 1)].get(k) for a in arms]
            if all(v is None for v in vals):
                continue
            print("[PARAMS] " + k + " | " + " | ".join("-" if v is None else f"{v}" for v in vals), flush=True)
        for a in arms:
            r = rows[(a, nl if a in ("B0a", "B0b", "B0pa", "B0pb") else 1)]
            d = r["合计"] - base
            print(f"[PARAMS] {a}(L{nl}) vs A：可训参数 Δ={d:+d}（{100.0 * d / base:+.4f}%）| "
                  f"步数 各臂同为 {STEPS}×batch{BATCH}={STEPS * BATCH} 样本", flush=True)
    for nl in (1, 2):
        shared_init_check(1234, nl)
    return 0


def _hits_path(arm: str, seed: int, smoke: bool = False, n_layers: int = 1) -> str:
    """L1 沿用旧名（与已跑的 r5 4 run 一致）；L2 加 `_L2` 后缀 ⇒ 两档不互相覆盖。"""
    return os.path.join(LOG_DIR, f"81_hits_{arm}_s{seed}{'' if n_layers == 1 else f'_L{n_layers}'}"
                                 f"{'_smoke' if smoke else ''}.json")


QUOTED_T2 = {("A", 1234): 12.38, ("A", 5678): 12.50, ("B1", 1234): 12.12, ("B1", 5678): 12.75,
             ("B2", 1234): 75.88, ("B2", 5678): 61.25}      # §11.24/§11.29/§11.28 的已测引文


def _read_hits(prefix: str, arm: str, seed: int):
    p = os.path.join(LOG_DIR, f"{prefix}_hits_{arm}_s{seed}.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def report_r5() -> int:
    """CPU：读 logs/81_hits_B0{a,b}_s*.json（+ 79 的 A/B1/B2 作对照）出 §14 的 R5-a..e。

    ★三种结局的判词**写死在这里**（§14），机制化打印，不事后找说法；另印天花板条款与两套门槛。
    ★措辞纪律（§9.7/§11.28）：「2 seed 同号」实为「**2 次运行同号**」，每个读数带 run 间 init 噪声。
    """
    seeds = (1234, 5678)
    data = {}
    for arm in B0_ARMS:
        for s in seeds:
            d = _read_hits("81", arm, s)
            if d is not None:
                data[(arm, s)] = d
    for arm in ("A", "B1", "B2"):
        for s in seeds:
            d = _read_hits("79", arm, s)
            if d is not None:
                data[(arm, s)] = d
    print(f"[PAIR] 读到 hits：{sorted(f'{a}_s{s}' for (a, s) in data)}（81_* 为本轮；79_* 为对照，只读引用）",
          flush=True)
    print("[PAIR] 对照引文核对（§11.24/§11.29/§11.28）：" + " | ".join(
        f"{a}_s{s}: 引文 {QUOTED_T2[(a, s)]:.2f}% vs hits {data[(a, s)]['em']:.2f}%"
        f"{'一致' if abs(data[(a, s)]['em'] - QUOTED_T2[(a, s)]) < 0.01 else '★不一致（如实报）'}"
        for (a, s) in QUOTED_T2 if (a, s) in data), flush=True)

    # ---------------- R5-a ----------------
    inits = [(a, s, data[(a, s)].get("init_ok")) for a in B0_ARMS for s in seeds if (a, s) in data]
    print(f"[CRIT] R5-a（必过，§11.10）：B0a/B0b 逐 run init_ok={[x[2] for x in inits]} ⇒ "
          f"{'PASS' if inits and all(x[2] for x in inits) else '★FAIL/未判（配对作废）'}", flush=True)
    print(f"[CRIT] R5-a 说明：跨文件 [INIT-CHECK] 原始行在各 run 日志（B0a/B0b 全张量 max|Δ|=0、"
          f"与 79.A/B1/B2 共享子模块逐张量 max|Δ|=0、[COPY-CHECK] A/B1/B2 副本未漂移）。"
          f"★79 的 hits 里 init_ok **恒 false** 是 79 的仪器缺陷（shared_init_check 返回 dict、"
          f"却与 0.0 比较：stages/79_cand_cards_v2.py:755 返回 / :1122 比较）⇒ 79 那个布尔无效，"
          f"但它日志里紧邻的 [INIT-CHECK] 实测 max|Δ|=0.000e+00 ⇒ **79 的配对统计仍有效**（本文件已修）。",
          flush=True)

    # ---------------- R5-b / R5-c / R5-d：配对 ----------------
    comps = [("R5-b", "B0a", "A", "重整卡**单独**（无可判目标）vs 单卡 A"),
             ("R5-c", "B0b", "B0a", "给重整卡挂**可判中间目标** vs 同架构无 mid"),
             ("R5-d", "B0b", "B2", "重整卡上的**有序目标** vs 选择卡上的**选择目标**（能否替代）")]
    res = {}
    for key, ax, ay, desc in comps:
        print(f"[CRIT] {key}（配对，T2 完整回复 EM；§11.4 口径）{ax} vs {ay} —— {desc}", flush=True)
        for s in seeds:
            dx, dy = data.get((ax, s)), data.get((ay, s))
            if dx is None or dy is None:
                print(f"[CRIT] {key} seed={s}：{ax if dx is None else ay} 臂 hits 缺失 ⇒ 未判", flush=True)
                continue
            st = paired_line(f"{key} seed={s}", dx["hits"], dy["hits"], ax, ay)
            ok = st["delta"] > 0 and st["delta"] >= 2 * st["se"]
            neg = st["delta"] < 0 and st["delta"] <= -2 * st["se"]
            res.setdefault(key, {})[s] = dict(st, ok=ok, neg=neg, emx=dx["em"], emy=dy["em"])
            print(f"[CRIT] {key} seed={s}：EM({ax})={dx['em']:.2f}% EM({ay})={dy['em']:.2f}% "
                  f"Δ={st['delta']:+.2f}pp 2SE_Δ={2 * st['se']:.2f}pp McNemar p={st['p']:.3g} ⇒ "
                  f"{'PASS（显著更高：Δ>=2SE_Δ）' if ok else ('★显著**更低**（Δ<=-2SE_Δ）' if neg else '未达显著（≈）')}",
                  flush=True)
        v = res.get(key, {})
        both = [v[s]["ok"] for s in seeds if s in v]
        print(f"[CRIT] {key}「2 次运行同号」（措辞纪律 §9.7：实为 2 次运行）="
              f"{'PASS' if len(both) == 2 and all(both) else 'FAIL/未判'}", flush=True)

    # ---------------- 天花板 / 地板条款 ----------------
    ems = {a: [data[(a, s)]["em"] for s in seeds if (a, s) in data] for a in ("A", "B1", "B2", "B0a", "B0b")}
    ceilings = [a for a, v in ems.items() if v and max(v) >= 95.0]
    print(f"[CRIT] 天花板条款（§14/§11.7）：任一臂 T2 >=95% ⇒ 判「**不可测**」。实测 max T2 = "
          + " ".join(f"{a}={max(v):.2f}%" for a, v in ems.items() if v)
          + f" ⇒ {'★触发：不可测（' + '/'.join(ceilings) + '）' if ceilings else '未触发（A 在地板 ≈12.4% ⇒ 余量充足）'}", flush=True)
    for a in ("A", "B1", "B2", "B0a", "B0b"):
        for s in seeds:
            d = data.get((a, s))
            if d is None:
                continue
            t1 = d.get("t1", float("nan"))
            print(f"[CRIT] 绝对口径 {a}_s{s}：T2={d['em']:.2f}% vs T2门槛(捷径级 max_naive "
                  f"{d.get('max_naive', 0.0):.2f}%+2SE)={d.get('t2_thr', 0.0):.2f}% ⇒ "
                  f"{'过' if d['em'] > d.get('t2_thr', 0.0) else '★未过（如实报）'} | T1={t1:.2f}% vs "
                  f"T1门槛(机会级 1/k+2SE)={d.get('t1_thr', float('nan')):.2f}% ⇒ "
                  f"{'过' if t1 > d.get('t1_thr', float('nan')) else '★未过（如实报）'}", flush=True)

    # ---------------- 三种结局的判词（§14，写死） ----------------
    print("[CRIT] 绝对口径说明：T1（字母）只在**有选择卡的臂**（B1/B2）上是能力判据；A/B0a/B0b 没有选择卡，"
          "T1 落在机会水平 1/k 附近是**预期**（它们的主读数是 T2）。T2 门槛=捷径级 max_naive、T1 门槛="
          "机会级 1/k+2SE，两者**不混**（§11.5）。", flush=True)
    def sig(key, s):
        return bool(res.get(key, {}).get(s, {}).get("ok"))

    def dl(key, s):
        return res.get(key, {}).get(s, {}).get("delta")

    def sig_neg(key, s):
        """显著**为负**（反方向）：Δ<0 且 |Δ| >= 2SE。"""
        v = res.get(key, {}).get(s)
        return bool(v) and v["delta"] < 0 and v["delta"] <= -2 * v["se"]

    e1 = (not any(sig("R5-b", s) for s in seeds)) and all(sig("R5-c", s) for s in seeds)
    e2 = all(sig("R5-b", s) for s in seeds)
    e3 = (not any(sig("R5-c", s) or sig_neg("R5-c", s) for s in seeds))       # 两个方向都不显著 = ≈
    print(f"[CRIT] 结局判词脚本（§14 写死；读 R5-b（B0a−A）与 R5-c（B0b−B0a））："
          f" R5-b Δ={[dl('R5-b', s) for s in seeds]} 显著={[sig('R5-b', s) for s in seeds]} | "
          f"R5-c Δ={[dl('R5-c', s) for s in seeds]} 显著={[sig('R5-c', s) for s in seeds]}", flush=True)
    complete = all(len(res.get(k, {})) == 2 for k in ("R5-b", "R5-c"))
    if not complete:
        # ★读数不齐时**不得**套用判词：缺数据会让「没有显著差异」被误读成结局 3（本文件跑前就防这一条）
        print("[CRIT] ⇒ **读数不齐**（缺 B0a/B0b 的 hits 文件或某一 seed）⇒ **判词未定**；"
              "三条写死的结局一律不套用。", flush=True)
        e1 = e2 = e3 = False
    if e1:
        print("[CRIT] ⇒ 命中**结局 1**：B0-a ≈ A 且 B0-b ≫ B0-a ⇒ 「**重整卡本身零增量**、"
              "挂在它上面的**可判目标**有增益」⇒ 把 §11.29 的结论**推广到另一类卡**："
              "增益来自「可判中间目标」这一机制，而不是某一种卡。", flush=True)
    if e2:
        print("[CRIT] ⇒ 命中**结局 2**：B0-a > A（2 次运行同号显著）⇒ 与「卡边界零增量」冲突 ⇒ "
              "**先排查实现差异**（参数/步数/通路口；注意 B0 比 A 少 1 个 encoder 层），"
              "再谈「重整卡本身有增量」。", flush=True)
    if e3:
        print("[CRIT] ⇒ 命中**结局 3**：B0-b ≈ B0-a ⇒ 「有序模因」作为中间目标**不成立**（该设计点被否定）⇒ "
              "如实写清是**目标设计**问题（形式/可判性）还是**任务不需要有序性**。", flush=True)
    if complete and not (e1 or e2 or e3):
        print("[CRIT] ⇒ 不命中任何一条写死的结局（读数混合，例如 B0a<A 或 B0b>B2）⇒ 按原始读数如实报，"
              "不得套用三条判词。", flush=True)
    if any(sig_neg("R5-c", s) for s in seeds):
        print("[CRIT] ⇒ 另注（方向）：R5-c 出现**显著为负**（B0-b 显著**低于** B0-a）⇒ 不只是「没增益」，"
              "而是「挂上该中间目标反而更差」⇒ 按结局 3 的口径如实写清，并标明方向。", flush=True)
    if any(sig_neg("R5-b", s) for s in seeds):
        print("[CRIT] ⇒ 另注（方向）：R5-b 出现**显著为负**（B0-a 显著**低于** A）⇒ 重整卡单独比单卡更差；"
              "先查实现差异（参数少 1 层是**预期**差异，不是解释）。", flush=True)
    d_neg = any(res.get("R5-d", {}).get(s, {}).get("neg") for s in seeds)
    d_ok = any(sig("R5-d", s) for s in seeds)
    if any(x is not None for x in [dl("R5-d", s) for s in seeds]):
        dvals = [dl("R5-d", s) for s in seeds if dl("R5-d", s) is not None]
        print(f"[CRIT] R5-d 解读（能否用「重整卡上的有序目标」替代「选择卡上的选择目标」，§14）："
              f"Δ(B0b−B2)={dvals} 显著为负={d_neg} ⇒ "
              + ("替代**不成立**：B2 显著更高 ⇒ 选择卡上的显式**选择**目标仍不可被「重整卡上的文本式有序目标」替代"
                 if d_neg else
                 "读数反向：B0b 显著更高（须先排查 B2 的读数/实现，不得直接宣称替代成立）" if d_ok else
                 "替代成立（本口径下无显著差异）")
              + "；★该替代判定的前提是 B2 的 61–76% 不是天花板（余量见上）。", flush=True)

    # ---------------- R5-e：并列报 ----------------
    print("[CRIT] R5-e（必报）参数量 / 步数 / 峰值显存 / T1 / T2 / 中间目标并列：", flush=True)
    for arm in ("A", "B1", "B2", "B0a", "B0b"):
        row = []
        for s in seeds:
            d = data.get((arm, s))
            if d is None:
                row.append("未跑")
                continue
            extra = ""
            if d.get("mid_em") is not None:
                extra = (f" midEM={d['mid_em']:.2f}% mid字母={d.get('mid_letter', float('nan')):.2f}%"
                         f" SEP自出={d.get('mid_sep_rate', float('nan')):.1f}%")
            row.append(f"s{s}: T2={d['em']:.2f}% T1={d.get('t1', float('nan')):.2f}% "
                       f"params={d.get('params')} steps={d.get('steps')} peak={d.get('peak_mib', 0):.0f}MiB"
                       f" 抽不出={100.0 * (d.get('miss') or 0) / d['n']:.2f}%{extra}")
        print(f"[CRIT] R5-e arm={arm} | " + " | ".join(row), flush=True)

    # ---------------- B0b 中间目标解剖（★派生读数，训练后追加；非预注册判据，只为定位失败点） ----------------
    def _mid_letter(t):
        ls = letters_in(t or "")
        return ls[0] if ls else None

    def _mid_content(t):
        return LABEL_RE.sub("", t or "")          # 去掉 `（X）` 标签，只留 `实体/条件`

    for s in seeds:
        d = data.get(("B0b", s))
        if not d or d.get("mids") is None or d.get("gold_mids") is None:
            continue
        gms, pms, hh = d["gold_mids"], d["mids"], d["hits"]
        n = len(gms)
        cont = [1 if _mid_content(g) == _mid_content(p) else 0 for g, p in zip(gms, pms)]
        let = [1 if (_mid_letter(g) is not None and _mid_letter(p) == _mid_letter(g)) else 0
               for g, p in zip(gms, pms)]
        nlet, ncont = sum(let), sum(cont)
        t2_cond = (100.0 * sum(hh[i] for i in range(n) if let[i]) / nlet) if nlet else float("nan")
        chance = 100.0 / max(1, int(d.get("k", 8)))
        ok_form = (100.0 * ncont / n) >= 90.0 and (100.0 * nlet / n) <= 1.5 * chance
        print(f"[CRIT] B0b 中间目标解剖 seed={s}（**派生读数**，从 hits 的 mids/gold_mids 复算；★非预注册判据）："
              f"mid 逐字 EM={d['mid_em']:.2f}% | ★mid **去字母后**（只比 `实体/条件`）EM="
              f"{100.0 * ncont / n:.2f}%（{ncont}/{n}）| mid 字母命中={100.0 * nlet / n:.2f}%（机会 {chance:.2f}%）| "
              f"T2（在 mid 字母命中条件下）={t2_cond:.2f}%（n={nlet}）⇒ "
              + ("模型学会了 mid 的**内容**（实体/条件）、没学会**字母/位置**（≈机会 1/k）⇒ 失败点在"
                 "「在候选集里定位」这一步（与 A/B1 同源的 pointer 缺口，§11.24）⇒ 属**目标形式/机制**问题"
                 "（自由文本式目标 vs B2 那种**对候选位置的结构化分类**），不是「任务不需要有序性」"
                 if ok_form else
                 "内容也没学会 ⇒ 先怀疑**目标形式/监督口径**（文本式 mid 的 token 级 CE），而不是任务性质"),
              flush=True)
    return 0



def _read81(arm: str, seed: int, nl: int):
    p = _hits_path(arm, seed, False, nl)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def report_r5p(judged: int = 2) -> int:
    """CPU：§14.2 的 `R5′-a..e`（**判据档 = --realign-layers 2**；L1 档作旁证并量「少一层」影响）。

    ★地板条款（§14.1 新增）内置：若 B0′-a 与 B0′-b 的 T2 都落在地板带（机会 1/k ± 2SE）⇒ `R5′-c`
    判「**不可判（空测试）**」，**不得**写成「可判目标无效」。
    """
    seeds = (1234, 5678)
    J = {a: {s: _read81(a, s, judged) for s in seeds} for a in B0X_ARMS}
    P = {a: {s: _read81(a, s, 1) for s in seeds} for a in B0X_ARMS}
    ref = {a: {s: _read_hits("79", a, s) for s in seeds} for a in ("A", "B2")}
    print(f"[PAIR] R5′ 读到：判据档 L{judged} = "
          f"{sorted(f'{a}_s{s}' for a in B0X_ARMS for s in seeds if J[a][s])} | 旁证档 L1 = "
          f"{sorted(f'{a}_s{s}' for a in B0X_ARMS for s in seeds if P[a][s])} | 对照 79.A/79.B2", flush=True)
    inits = [(a, s, J[a][s].get("init_ok")) for a in B0X_ARMS for s in seeds if J[a][s]]
    print(f"[CRIT] R5′-a（必过，§14.2）：逐 run init_ok={[x[2] for x in inits]}；[INIT-CHECK]/[PARAMS]/[GATE] "
          f"原始行在各 run 日志 ⇒ {'PASS' if inits and all(x[2] for x in inits) else '★FAIL/未判（配对作废）'}",
          flush=True)
    res = {}
    for key, ax, ay, desc in (("R5′-b", "B0pa", "A", "给重整卡候选身份通路（无可判目标）vs 单卡 A"),
                              ("R5′-c", "B0pb", "B0pa", "可判中间目标挂重整卡 vs 同结构无 mid"),
                              ("R5′-d", "B0pb", "B2", "重整卡上的目标 vs 选择卡上的目标（能否替代）")):
        print(f"[CRIT] {key}（配对，T2 完整回复 EM；§11.4 口径）{ax} vs {ay} —— {desc}", flush=True)
        for s in seeds:
            dx = J[ax][s] if ax in B0X_ARMS else ref.get(ax, {}).get(s)
            dy = J[ay][s] if ay in B0X_ARMS else ref.get(ay, {}).get(s)
            if dx is None or dy is None:
                print(f"[CRIT] {key} seed={s}：读数缺失 ⇒ 未判", flush=True)
                continue
            st = paired_line(f"{key} seed={s}", dx["hits"], dy["hits"], ax, ay)
            ok = st["delta"] > 0 and st["delta"] >= 2 * st["se"]
            neg = st["delta"] < 0 and st["delta"] <= -2 * st["se"]
            res.setdefault(key, {})[s] = dict(st, ok=ok, neg=neg, emx=dx["em"], emy=dy["em"])
            print(f"[CRIT] {key} seed={s}：EM({ax})={dx['em']:.2f}% EM({ay})={dy['em']:.2f}% "
                  f"Δ={st['delta']:+.2f}pp 2SE_Δ={2 * st['se']:.2f}pp McNemar p={st['p']:.3g} ⇒ "
                  f"{'PASS（显著更高）' if ok else ('★显著**更低**' if neg else '未达显著（≈）')}", flush=True)
        v = res.get(key, {})
        both = [v[s]["ok"] for s in seeds if s in v]
        print(f"[CRIT] {key}「2 次运行同号」={'PASS' if len(both) == 2 and all(both) else 'FAIL/未判'}", flush=True)

    # ---- 地板条款（§14.1/§14.2） ----
    d0 = J["B0pa"][seeds[0]] or J["B0pb"][seeds[0]]
    n = int(d0["n"]) if d0 else 800
    k = int(d0.get("k", 8)) if d0 else 8
    chance = 100.0 / k
    se_ch = (chance / 100.0 * (1 - chance / 100.0) / n) ** 0.5 * 100.0
    band = (chance - 2 * se_ch, chance + 2 * se_ch)

    def at_floor(a, s):
        d = J[a][s]
        return d is not None and band[0] <= d["em"] <= band[1]

    fl = {f"{a}_s{s}": (round(J[a][s]["em"], 2) if J[a][s] else None) for a in B0X_ARMS for s in seeds}
    have_any = any(J[a][s] for a in B0X_ARMS for s in seeds)
    both_floor = have_any and all(at_floor(a, s) for a in B0X_ARMS for s in seeds if J[a][s])
    print(f"[CRIT] 地板条款（§14.1/§14.2）：T2 地板带 = 机会 1/k={chance:.2f}% ±2SE {2 * se_ch:.2f}pp = "
          f"[{band[0]:.2f}%, {band[1]:.2f}%]；实测 T2 = {fl} ⇒ "
          + ("★两臂都落在地板带 ⇒ `R5′-c` 判「**不可判（空测试）**」，不得写成「可判目标无效」"
             if both_floor else "未全部落在地板带 ⇒ 该对比可判（按配对判据读）"), flush=True)

    # ---- 「少一层」/参数对齐档的影响（L2 − L1，同臂同 seed 配对） ----
    for a in B0X_ARMS:
        for s in seeds:
            d2, d1 = J[a][s], P[a][s]
            if d2 is None or d1 is None:
                print(f"[PAIR] 参数对齐档影响 {a}_s{s}：缺 L1 或 L2 的 hits ⇒ 未判", flush=True)
                continue
            st = paired_line(f"{a}_s{s}（L{judged} − L1）", d2["hits"], d1["hits"],
                             f"{a}(L{judged})", f"{a}(L1)")
            print(f"[CRIT] 参数对齐档影响（少一层）{a}_s{s}：params L{judged}={d2.get('params')} vs L1="
                  f"{d1.get('params')} | T2(L{judged})={d2['em']:.2f}% T2(L1)={d1['em']:.2f}% Δ={st['delta']:+.2f}pp "
                  f"2SE_Δ={2 * st['se']:.2f}pp McNemar p={st['p']:.3g}", flush=True)

    # ---- R5′-e：并列报 ----
    print("[CRIT] R5′-e（必报）参数量/步数/峰值显存/T1/T2/中间目标并列：", flush=True)
    for tag, src in ((f"L{judged}(判据档)", J), ("L1(旁证)", P)):
        for a in ("B0pa", "B0pb"):
            row = []
            for s in seeds:
                d = src[a][s]
                row.append("未跑" if d is None else
                           f"s{s}: T2={d['em']:.2f}% T1={d.get('t1', float('nan')):.2f}% "
                           f"params={d.get('params')} steps={d.get('steps')} peak={d.get('peak_mib', 0):.0f}MiB"
                           + (f" midEM={d['mid_em']:.2f}% SEP自出={d.get('mid_sep_rate', float('nan')):.1f}%"
                              if d.get("mid_em") is not None else ""))
            print(f"[CRIT] R5′-e {tag} arm={a} | " + " | ".join(row), flush=True)
        for a in ("A", "B2"):
            row = []
            for s in seeds:
                d = ref[a][s]
                row.append("未跑(79)" if d is None else
                           f"s{s}: T2={d['em']:.2f}% T1={d.get('t1', float('nan')):.2f}% "
                           f"params={d.get('params')} peak={d.get('peak_mib', 0):.0f}MiB")
            print(f"[CRIT] R5′-e {tag} 对照 arm={a}（79 只读引用）| " + " | ".join(row), flush=True)

    # ---- 三种结局的判词（§14.2 写死） ----
    def sig(key, s):
        return bool(res.get(key, {}).get(s, {}).get("ok"))

    def sne(key, s):
        return bool(res.get(key, {}).get(s, {}).get("neg"))

    def dl(key, s):
        return res.get(key, {}).get(s, {}).get("delta")

    complete = all(len(res.get(key, {})) == 2 for key in ("R5′-b", "R5′-c"))
    print(f"[CRIT] R5′ 结局判词脚本（§14.2 写死）：R5′-b Δ={[dl('R5′-b', s) for s in seeds]} "
          f"显著={[sig('R5′-b', s) for s in seeds]} | R5′-c Δ={[dl('R5′-c', s) for s in seeds]} "
          f"显著={[sig('R5′-c', s) for s in seeds]}", flush=True)
    if not complete:
        print("[CRIT] ⇒ 读数不齐 ⇒ **判词未定**（三条结局一律不套用）", flush=True)
        return 0
    if both_floor:
        print("[CRIT] ⇒ 地板条款生效：B0′-a 与 B0′-b 都在地板带 ⇒ `R5′-c` 判「**不可判（空测试）**」；"
              "**不得**写成「可判目标无效」或「重整卡零增量」（§14.1/§14.2）。", flush=True)
    if all(sig("R5′-b", s) for s in seeds):
        print("[CRIT] ⇒ R5′-b 命中**结局①**：B0′-a ≫ A ⇒ 「卡（+候选身份通路）」本身就能带来能力。", flush=True)
    else:
        print("[CRIT] ⇒ R5′-b 命中**结局②**：B0′-a ≈ A（都在地板）⇒ **只有身份通路还不够，缺「匹配动作」**"
              "⇒ 支持 §11.24 的 pointer 缺口（A 有身份无动作；B0 均值档连身份都没有，见 §14.1 的真收获）。",
              flush=True)
    if all(sig("R5′-c", s) for s in seeds):
        print("[CRIT] ⇒ R5′-c 命中**结局①**：B0′-b ≫ B0′-a ⇒ 「可判中间目标」这一机制与**挂在哪张卡无关**"
              "（把 §11.29 推广到重整卡）。", flush=True)
    elif not both_floor:
        print("[CRIT] ⇒ R5′-c 命中**结局②**：B0′-b ≈ B0′-a ⇒ **目标必须挂在「专门做匹配的卡」上**，"
              "卡的角色划分有意义（在非空测试前提下）。", flush=True)
    d_neg = any(sne("R5′-d", s) for s in seeds)
    d_ok = any(sig("R5′-d", s) for s in seeds)
    dvals = [dl("R5′-d", s) for s in seeds]
    print(f"[CRIT] R5′-d 解读（重整卡上的目标能否替代选择卡上的目标，§14.2）：Δ(B0′-b − B2)={dvals} ⇒ "
          + ("**不可替代**（B2 显著更高）" if d_neg else
             ("读数反向（B0′-b 显著更高）⇒ 先排查 B2 侧读数/实现，不得直接宣称可替代" if d_ok else
              "**等价**（本口径下无显著差异 ⇒ 可替代）")), flush=True)
    return 0


# ---------------- 主流程 ----------------
device = "cpu"


def has_label(recs) -> bool:
    return bool(recs) and ("gold_letter" in recs[0])


def check_data(prefix: str, a) -> int:
    """CPU：只读数据 + 复算两套地板（不建模型、不碰 GPU）——`81` 的管线自检入口。

    核：①标签可判（gold_letter == CAND_LETTERS[gold_idx]；split_label(target) == (gold_det, gold_letter)）；
        ②T1 门槛 = 1/k+2SE 与 T2 门槛 = max_naive+2SE **各算各的**；
        ③反事实三条的候选集改动量 + 动态金标可判性；
        ④★B0b 的中间目标：mid 往返解码 /「片段拼 token == 整串 encode」/ SEP 冲突 / p+t 是否超 MAXLEN。
    """
    train = load_jsonl(os.path.join(LOG_DIR, f"{prefix}_dialog_train.jsonl"))
    test = load_jsonl(os.path.join(LOG_DIR, f"{prefix}_dialog_test.jsonl"))
    fp_tr, fp_te = fp_reply(train), fp_reply(test)      # ★数据身份（回复目标口径，prep 之前）
    for r in train + test:
        prep_record(r, variant="B0b")                    # ★按 B0b 造目标序列（含 mid + SEP）
    labs = has_label(test)
    fp_b0b_tr, fp_b0b_te = d74.d72.data_fp(train), d74.d72.data_fp(test)
    pt_b0b = max(len(r["p"]) + len(r["t"]) for r in train + test)
    pt_b0a = max(len(r["p"]) + len(r["treply"]) for r in train + test)
    mid_max = max(len(r["tmid"]) for r in train + test)
    bad_len = sum(1 for r in train + test if len(r["p"]) + len(r["t"]) > MAXLEN)
    print(f"[CHECK-DATA] prefix={prefix} k={a.k} 带标签={labs} train={len(train)} test={len(test)} | "
          f"data_fp(回复目标口径=引文口径) train={fp_tr} test={fp_te} | B0b 训练目标指纹 train={fp_b0b_tr} "
          f"test={fp_b0b_te}（含 mid ⇒ 与上面必然不同，非数据差异）| ctx tok 最大={max(len(r['p']) for r in test)} | "
          f"p+t 最大 B0a={pt_b0a} B0b={pt_b0b}（mid 最长 {mid_max} token）超 MAXLEN={bad_len} | "
          f"候选末位 token 前缀稳定性违例={POS_BAD['n']}", flush=True)
    sep_conf = sum(1 for r in train + test
                   if (r.get("mid") and B0_SEP in r["mid"]) or (B0_SEP in r["target"]))
    mid_none = sum(1 for r in train + test if r.get("mid") is None)
    print(f"[CHECK-DATA] B0b 中间目标编码：mid 往返解码违例={MID_BAD['rt']} | "
          f"「片段拼 token == 整串 encode」违例={MID_BAD['add']} | mid 不可判条数={mid_none} | "
          f"SEP={B0_SEP!r}(id={SEP_ID}) 与 mid/回复冲突={sep_conf} ⇒ "
          f"{'PASS' if (MID_BAD['rt'] == MID_BAD['add'] == mid_none == sep_conf == 0 and bad_len == 0) else '★FAIL'}"
          f" | mid 形式样例={test[0]['mid']!r}", flush=True)
    if labs:
        bad_letter = sum(1 for r in test if r["gold_idx"] < 0
                         or r.get("gold_letter") != CAND_LETTERS[r["gold_idx"]])
        bad_split = 0
        for r in test:
            sl = split_label(r["target"])
            if sl is None or sl[1] != r.get("gold_letter"):
                bad_split += 1
        bad_unique = sum(1 for r in test if correct_letter_of(r) != r.get("gold_letter"))
        leak = sum(1 for r in test if letters_of(r["ctx"]))
        print(f"[CHECK-DATA] 标签可判：gold_letter!=CAND_LETTERS[gold_idx]={bad_letter} | "
              f"split_label(target) 与 gold_letter 不一致={bad_split} | 唯一双槽位匹配项不是 gold={bad_unique} | "
              f"ctx 里出现「（X）」形态的条数={leak} ⇒ "
              f"{'PASS' if (bad_letter == bad_split == bad_unique == leak == 0) else '★FAIL'}", flush=True)
    fl = d76.compute_floors(train, test, f"{prefix}-k{a.k}-CHECK")
    t1p, t1se, t1thr = t1_threshold(test[0]["n_cands"], len(test))
    t2se, t2thr = t2_threshold(fl["max_naive_f"], len(test))[1:]
    print(f"[CHECK-DATA] 门槛（两套**分开**，§11.5）：**T1 机会级 1/k={t1p:.2f}%+2SE {2 * t1se:.2f}pp="
          f"{t1thr:.2f}%**（实测 max_chance(T1)={fl['max_chance_l']:.2f}%）| **T2 捷径级 max_naive="
          f"{fl['max_naive_f']:.2f}%+2SE {t2se:.2f}pp={t2thr:.2f}%**（§11.14 引文 "
          f"{MAX_NAIVE_QUOTED.get(a.k, float('nan')):.2f}%）", flush=True)
    for mode in ("shuffle", "replace", "delete"):
        rr = random.Random(0xC0FFEE ^ a.seed ^ CF_MODE_ID[mode])
        recs = [cf_record(r, mode, rr, "B0b") for r in test]
        changed = sum(1 for r0, r1 in zip(test, recs) if r0["cand_line"] != r1["cand_line"])
        no_right = sum(1 for r in recs if correct_letter_of(r) is None)
        no_mid = sum(1 for r in recs if mid_gold_of(r) is None)
        print(f"[CHECK-DATA] 反事实 mode={mode}：候选行被改动={changed}/{len(test)} | "
              f"动态无正确对的条数={no_right} | 动态 mid 不可判={no_mid} | 参考回复可复算="
              f"{sum(1 for r in recs if d76._reference_reply(r)[0])}/{len(recs)}", flush=True)
    return 0


def main(argv=None):
    global device
    ap = argparse.ArgumentParser(
        description="S81 L3-r5 重整卡独立验收（B0a=重整卡无可判目标 / B0b=重整卡+有序模因中间目标）")
    ap.add_argument("--arm", choices=("B0a", "B0b", "B0pa", "B0pb", "A", "B1", "B2"), default="B0pa",
                    help="B0a/B0b = r5（候选**均值**注入，§14）；B0pa/B0pb = r5′（重整卡对候选**交叉注意**，§14.2）；"
                         "A/B1/B2 只为本文件自带的回归/仪器核对保留")
    ap.add_argument("--realign-layers", type=int, choices=(1, 2), default=2,
                    help="重整卡 self-attn 层数（2 = 与 A/B1/B2 的**参数对齐档**，m_383/m_384；1 = 旁证档）")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--k", type=int, choices=(4, 8), default=8)
    ap.add_argument("--data-prefix", default=None,
                    help="数据前缀（默认按 --k：8→76（带字母，§11.14）/ 4→74（回融，无标签））")
    ap.add_argument("--t2-thr", type=float, default=None,
                    help="T2 门槛（百分点）：默认用本 run 复算的捷径级 max_naive；给数则改用该写死引文值并并列")
    ap.add_argument("--smoke", action="store_true", help="50 步、只评前 60 条（探路，不作判据）")
    ap.add_argument("--steps", type=int, default=STEPS)
    ap.add_argument("--cf", action="store_true", help="训练后对同一 ckpt 做反事实评测（shuffle/replace/delete）")
    ap.add_argument("--pair-hits", default=None,
                    help="逗号分隔的其它臂 hits（本 run 出配对原始行）："
                         "B0a→logs/79_hits_A_s<seed>.json；B0b→logs/81_hits_B0a_s<seed>.json,logs/79_hits_B2_s<seed>.json")
    ap.add_argument("--load", default=None, help="载入 ckpt（跳过训练）")
    ap.add_argument("--param-table", action="store_true", help="只打参数/初值对齐表（CPU）")
    ap.add_argument("--report-r5", action="store_true", help="只读 hits 文件出 §14 的 R5-a..e（CPU）")
    ap.add_argument("--report-r5p", action="store_true", help="只读 hits 文件出 §14.2 的 R5′-a..e（CPU）")
    ap.add_argument("--check-data", action="store_true",
                    help="只读数据 + 复算两套门槛（CPU，不占 GPU）")
    a = ap.parse_args(argv)

    load_tok()
    if a.param_table:
        return param_table()
    if a.report_r5:
        return report_r5()
    if a.report_r5p:
        return report_r5p(a.realign_layers)

    prefix = a.data_prefix or PREFIX_BY_K[a.k]
    steps = SMOKE_STEPS if a.smoke else a.steps
    smoke_note = "（--smoke 探路，不作判据）" if a.smoke else ""
    variant = "B0b" if a.arm in ("B0b", "B0pb") else "A"   # 只有 *b 臂的目标序列带 mid

    if a.check_data:
        return check_data(prefix, a)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device, torch.cuda.get_device_name(0) if device == "cuda" else "", flush=True)
    if device != "cuda":
        print("GPU 起不来，立即停（不用 CPU 硬跑）。", flush=True)
        print("HSA_OVERRIDE_GFX_VERSION =", os.environ.get("HSA_OVERRIDE_GFX_VERSION"), flush=True)
        sys.exit(3)

    train = load_jsonl(os.path.join(LOG_DIR, f"{prefix}_dialog_train.jsonl"))
    test = load_jsonl(os.path.join(LOG_DIR, f"{prefix}_dialog_test.jsonl"))
    fp_tr, fp_te = fp_reply(train), fp_reply(test)      # ★数据身份 = 引文口径（prep 之前算）
    for r in train:
        prep_record(r, variant=variant)
    for r in test:
        prep_record(r, variant=variant)
    ev = test[:SMOKE_EVAL] if a.smoke else test
    keys = sorted(train[0].keys())
    labeled = has_label(test)
    sha_tr = d74.d72.file_sha256(os.path.join(LOG_DIR, f"{prefix}_dialog_train.jsonl"))
    sha_te = d74.d72.file_sha256(os.path.join(LOG_DIR, f"{prefix}_dialog_test.jsonl"))
    fp_tgt_tr = d74.d72.data_fp(train)                  # 本 variant 的**训练目标**指纹（B0b 含 mid）
    fp_tgt_te = d74.d72.data_fp(test)
    print(f"[DATA] k={a.k} prefix={prefix} arm={a.arm} variant={variant} seed={a.seed} train={len(train)} "
          f"test={len(test)} 评测 n={len(ev)} | 候选数={train[0]['n_cands']} | 带标签={labeled} | ctx tok 最大="
          f"{max(len(r['p']) for r in train + test)} p+t 最大="
          f"{max(len(r['p']) + len(r['t']) for r in train + test)} 超 MAXLEN="
          f"{sum(1 for r in train + test if len(r['p']) + len(r['t']) > MAXLEN)} | 候选末位 token "
          f"前缀稳定性违例={POS_BAD['n']} | 字段数={len(keys)}", flush=True)
    if variant == "B0b":
        print(f"[DATA] B0b 中间目标：t = enc(mid)+[SEP {B0_SEP!r}(id={SEP_ID})]+enc(回复+<eos>) | mid 最长="
              f"{max(len(r['tmid']) for r in train + test)} token | 拼接违例={MID_BAD['add']} "
              f"往返违例={MID_BAD['rt']} | mid 样例={train[0]['mid']!r}", flush=True)
    q = FP_QUOTED.get(prefix)
    print(f"[DATA] jsonl sha256 前 8 位 train={sha_tr[:8]} test={sha_te[:8]} | data_fp(回复目标口径) "
          f"train={fp_tr} test={fp_te} | 本 variant 训练目标指纹 train={fp_tgt_tr} test={fp_tgt_te}"
          f"{'（B0b 含 mid ⇒ 与回复目标口径必然不同）' if variant == 'B0b' else ''}"
          + (f" | §11.6/§11.14 引文 train={q[0]} test={q[1]} ⇒ "
             f"{'表面一致' if (fp_tr == q[0] and fp_te == q[1]) else '★与引文不同（如实报，指纹口径见引文）'}"
             f"（仅供参考性对齐，不作判据）" if q else ""), flush=True)
    if not labeled:
        print(f"[DATA] ★本档（prefix={prefix}）无标签字段 ⇒ T1 字母指标不可判（未判）", flush=True)

    # ---- 两套门槛（§11.5 裁定：机会级 ≠ 捷径级；T1 与 T2 各算各的） ----
    fl = d76.compute_floors(train, ev, f"{prefix}-k{a.k}-{a.arm}{'-smoke' if a.smoke else ''}")
    t1p, t1se, t1thr = t1_threshold(train[0]["n_cands"], len(ev))
    t2q = MAX_NAIVE_QUOTED.get(a.k, float("nan"))
    t2base = fl["max_naive_f"] if a.t2_thr is None else a.t2_thr
    t2se, t2thr = t2_threshold(t2base, len(ev))[1:]
    print(f"[FLOOR] 复算（d76 口径，两块分开）k={a.k} | 机会级 T1：1/k={t1p:.2f}% 1SE={t1se:.3f}pp "
          f"2SE={2 * t1se:.2f}pp ⇒ **T1 门槛(机会级)=1/k+2SE={t1thr:.2f}%** | 实测 max_chance(T1)="
          f"{fl['max_chance_l']:.2f}%（来自 {fl['mc_l_name']}）", flush=True)
    print(f"[FLOOR] 复算 捷径级 T2：max_naive={fl['max_naive_f']:.2f}%（来自 {fl['ms_f_name']}；"
          f"§11.14 引文 {t2q:.2f}%）2SE={t2se:.2f}pp ⇒ **T2 门槛(捷径级)={t2thr:.2f}%**"
          f"{'（用本 run 复算值）' if a.t2_thr is None else f'（用 --t2-thr 写死值 {a.t2_thr:.2f}%）'}"
          f" | 参考上限 T1={fl['ref_l']:.2f}% T2={fl['ref_f']:.2f}%", flush=True)

    # ---- [INIT-CHECK]（§9.7：必须先用同 seed 构造两次核验，再构造真模型） ----
    print("[INIT-CHECK] §9.7 init 缺陷修复：manual_seed 先于构造（并修 79 的 init_ok 判词缺陷）", flush=True)
    init_worst = shared_init_check(a.seed, a.realign_layers)
    torch.manual_seed(a.seed)
    torch.cuda.manual_seed_all(a.seed)
    model = build_arm(a.arm, n_layers=a.realign_layers).to(device)
    named = param_groups_named(model)
    n_par = named["合计"]
    print(f"[PARAMS] arm={a.arm} 注入={getattr(model, 'inject', '-')} 重整卡层数={a.realign_layers} "
          f"可训参数={n_par} | 分项=" +
          " ".join(f"{k}={v}" for k, v in named.items() if k != "合计") +
          f" | 步数={steps}×batch{BATCH}={steps * BATCH} 样本 lr={LR}"
          + (f"（B0 无选择卡 ⇒ 比 A 少 1 个 encoder 层，见 --param-table）" if a.arm in B0_ARMS else ""),
          flush=True)

    torch.cuda.reset_peak_memory_stats()
    if a.load:
        model.load_state_dict(torch.load(a.load, map_location=device, weights_only=False))
        gate, wall = None, 0.0
        print(f"[TRAIN] --load={a.load} ⇒ 跳过训练", flush=True)
    else:
        gate, wall = train_run(model, train, a.seed, steps)
    peak = torch.cuda.max_memory_allocated() / (1024 ** 2)
    ltag = "" if a.realign_layers == 1 else f"_L{a.realign_layers}"
    ck = os.path.join(LOG_DIR, f"81_ckpt_{a.arm}_k{a.k}_s{a.seed}{ltag}{'_smoke' if a.smoke else ''}.pt")
    torch.save(model.state_dict(), ck)

    # ---- 主评测（T1 与 T2 各算各的；B0b 走两段贪心） ----
    t_eval = time.time()
    ce = eval_ce(model, ev)
    cm, cse = mean_se(ce)
    mids, sep_ok, mid_rep, mid_e2e = None, None, None, None
    cmid_m = cmid_se = crep_m = crep_se = None
    if variant == "B0b":
        mids, txt, sep_ok = greedy_gen_b0b(model, ev)
    else:
        txt = greedy_gen(model, ev)
    hits = [em_strict(r["target"], txt[i]) for i, r in enumerate(ev)]
    em = 100.0 * sum(hits) / len(hits)
    se, thr_run = em_with_thr(em, len(hits))
    sel = eval_sel(model, ev)
    sel_acc = None if sel is None else 100.0 * sum(sel["hits"]) / len(sel["hits"])
    sel_lacc = None if sel is None else 100.0 * sum(sel["letter_hits"]) / len(sel["letter_hits"])
    ev_rep = report_texts(hits, txt, ev, "T1T2")
    t1, miss = ev_rep["t1"], ev_rep["miss"]
    if variant == "B0b":
        mid_rep = report_mid(mids, ev, sep_ok)
        _ce_m, _ce_r = eval_ce_split(model, ev)
        cmid_m, cmid_se = mean_se(_ce_m)
        crep_m, crep_se = mean_se(_ce_r)
        mid_e2e = 100.0 * sum(1 for i, r in enumerate(ev)
                              if em_strict((mid_rep["gold_mid"][i] or "") + B0_SEP + r["target"],
                                           (mids[i] or "") + B0_SEP + txt[i])) / len(ev)
    eval_sec = time.time() - t_eval

    print(f"[MAIN] arm={a.arm} variant={variant} k={a.k} prefix={prefix} seed={a.seed} steps={steps} "
          f"batch={BATCH} lr={LR} | d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN} params={n_par} | "
          f"配对CE={cm:.4f}±{cse:.4f} (nats/token)"
          + (f" | B0b 分段 CE：mid+SEP={cmid_m:.4f}±{cmid_se:.4f} 回复={crep_m:.4f}±{crep_se:.4f}"
             if variant == "B0b" else "")
          + f" | 评估墙钟={eval_sec:.0f}s", flush=True)
    print(f"[EM] {'smoke子集' if a.smoke else '全量'} n={len(hits)} T2 严格EM={em:.2f}% ±SE={se:.2f}pp | "
          f"主口径=strip 后逐字相等（含标签与句号，照 em_strict）| **T2 门槛(捷径级 max_naive {t2base:.2f}%"
          f"+2SE {t2se:.2f}pp)={t2thr:.2f}%**", flush=True)
    print(f"[SEL] " + (f"选择准确率(argmax 候选对==历史 (ent,cond))={sel_acc:.2f}% | "
                       f"选择 argmax 字母正确率={sel_lacc:.2f}%（n={len(sel['hits'])}）"
                       if sel is not None else
                       "不适用：本臂没有选择卡（B0 = 候选隐态**非选择性均值**注入，固定不学）")
          + f" | 参考:完美选择+DET 句面 EM={fl['ref_f']:.2f}% | 本 run T1 字母准确率门槛={t1thr:.2f}%",
          flush=True)
    for i in range(min(3, len(ev))):
        r = ev[i]
        gen_line = (f"[GEN] [{i}] gold_idx={r['gold_idx']} gold_letter={r.get('gold_letter')} "
                    f"gold={r['target']!r} | pred={txt[i]!r} | T2 EM{'对' if hits[i] else '错'} | T1 字母="
                    f"{ev_rep['ls'][i]!r}{'对' if ev_rep['ls'][i] == r.get('gold_letter') else '错'}")
        if variant == "B0b":
            gen_line += (f" | gold_mid={mid_rep['gold_mid'][i]!r} gen_mid={mids[i]!r} mid EM"
                         f"{'对' if (mid_rep['gold_mid'][i] or '').strip() == (mids[i] or '').strip() else '错'}"
                         f" SEP自出={'是' if sep_ok[i] else '否'}")
        if sel is not None:
            gen_line += (f" | 选中对{'对' if sel['hits'][i] else '错'} "
                         f"选中字母{'对' if sel['letter_hits'][i] else '错'}")
        print(gen_line, flush=True)

    # ---- R5-a：init 前提（本文件修了 79 的判词缺陷 ⇒ 这个布尔现在是实测的） ----
    print(f"[CRIT] {'R5′-a（必过，§14.2）' if a.arm in B0X_ARMS else 'R5-a（必过，§11.10/§14）'}："
          f"本 run 共享子模块逐张量 max|Δ|={init_worst:.3e}；同 seed 两次构造"
          f" hash 相同、异 seed 不同（见上方 [INIT-CHECK]/[COPY-CHECK]）⇒ "
          f"{'PASS' if init_worst == 0.0 else '★FAIL ⇒ 配对作废，改保守口径'}{smoke_note}", flush=True)

    # ---- R5-e（本 run 部分）：参数量/步数/峰值显存/T1/T2 并列 ----
    print(f"[CRIT] R5-e（必报）arm={a.arm} 注入={getattr(model, 'inject', '-')} 重整卡层数={a.realign_layers} | "
          f"可训参数={n_par}（分项 "
          + " ".join(f"{k}={v}" for k, v in named.items() if k != "合计")
          + f"）| 步数={steps}×batch{BATCH}={steps * BATCH} 样本 | 峰值显存={peak:.0f}MiB | "
          f"**T2 完整回复 EM={em:.2f}% ±SE={se:.2f}pp** | **T1 字母准确率="
          + (f"{t1:.2f}%（n={len(ev)}；抽不出比例={100.0 * miss / len(ev):.2f}%）" if labeled
             else "未判（本档无标签）")
          + "**"
          + (f" | **T_mid mid 逐字 EM={mid_rep['em_mid']:.2f}%**（mid 字母命中率="
             f"{mid_rep['mid_letter']:.2f}%，抽不出={100.0 * mid_rep['miss'] / len(ev):.2f}%）"
             f" | 端到端 EM(动态金标)={mid_e2e:.2f}%"
             if variant == "B0b" else "")
          + f" | 选择准确率=" + (f"{sel_acc:.2f}%" if sel is not None else "不适用(本臂无选择卡)")
          + f"{smoke_note}", flush=True)
    print(f"[CRIT] R5-e(绝对口径) 本 run EM={em:.2f}% vs **T2 门槛(捷径级 max_naive {t2base:.2f}%+2SE "
          f"{t2se:.2f}pp)={t2thr:.2f}%** ⇒ {'PASS 击败捷径级地板' if em > t2thr else '★FAIL（未击败捷径级地板，如实报）'}"
          f" | T1 字母准确率="
          + (f"{t1:.2f}% vs **T1 门槛(机会级 1/k {t1p:.2f}%+2SE {2 * t1se:.2f}pp)={t1thr:.2f}%** ⇒ "
             + ("PASS 击败机会水平" if t1 > t1thr else "★FAIL（未击败机会水平，如实报）") if labeled
             else "未判（本档无标签）"), flush=True)
    if labeled and em >= 95.0:
        print(f"[CRIT] 天花板条款（§14/§11.7）：臂 {a.arm} 的 T2 EM={em:.2f}% >=95% ⇒ 判「**不可测（天花板）**」；"
              f"**不得**写成「无增益」", flush=True)

    # ---- 反事实（shuffle/replace/delete，B0b 走两段贪心 + 动态 mid） ----
    cf = {}
    if a.cf:
        for mode, tag in (("shuffle", "①打乱候选顺序（位置不变性）"),
                          ("replace", "②替换正确候选（★任务真有选择的验证）"),
                          ("delete", "③删除正确候选（应落到机会水平）")):
            res = run_cf(model, ev, txt, mode, a.seed, tag, variant=variant, base_mid=mids)
            st = paired_stats(res["hits"], hits)
            st_dyn = paired_stats(res["dyn"], hits)
            st_l = None
            if labeled:
                _lx, _ly, _mx, _my = paired_letters(res["txt"], txt)
                lx_hit = [int(x == r.get("gold_letter")) for x, r in zip(_lx, ev)]
                ly_hit = [int(y == r.get("gold_letter")) for y, r in zip(_ly, ev)]
                st_l = paired_stats(lx_hit, ly_hit)
            print(f"[PAIR] CF({mode}) 配对（T2，对原 target）：Δ(反事实 − 原始)={st['delta']:+.2f}pp "
                  f"SE_Δ(配对)={st['se']:.2f}pp 2SE_Δ={2 * st['se']:.2f}pp | b(反事实错原对)={st['n01']} "
                  f"c(反事实对原错)={st['n10']} | McNemar 双侧精确 p={st['p']:.3g}", flush=True)
            print(f"[PAIR] CF({mode}) 配对（T2，对动态金标）：Δ={st_dyn['delta']:+.2f}pp "
                  f"SE_Δ(配对)={st_dyn['se']:.2f}pp 2SE_Δ={2 * st_dyn['se']:.2f}pp | b={st_dyn['n01']} "
                  f"c={st_dyn['n10']} | McNemar p={st_dyn['p']:.3g}", flush=True)
            if st_l is not None:
                print(f"[PAIR] CF({mode}) 配对（T1 字母，附加）：Δ={st_l['delta']:+.2f}pp "
                      f"SE_Δ(配对)={st_l['se']:.2f}pp 2SE_Δ={2 * st_l['se']:.2f}pp | b={st_l['n01']} "
                      f"c={st_l['n10']} | McNemar p={st_l['p']:.3g}", flush=True)
            if variant == "B0b":
                print(f"[PAIR] CF({mode}) 中间目标（B0b，动态金标）：mid 逐字 EM="
                      f"{100.0 * sum(res['dyn_mid_hit']) / len(ev):.2f}% | 端到端 EM="
                      f"{100.0 * sum(res['dyn_e2e']) / len(ev):.2f}% | mid 不可判="
                      f"{sum(1 for x in res['dyn_mid'] if x is None)}", flush=True)
            cf[mode] = {"hits": res["hits"], "dyn": res["dyn"], "txt": res["txt"],
                        "letters": res["letters"], "dyn_letter": res["dyn_letter"],
                        "dyn_mid": res["dyn_mid"], "dyn_mid_hit": res["dyn_mid_hit"],
                        "dyn_e2e": res["dyn_e2e"], "mids": res["mids"],
                        "sel": None if res["sel"] is None else res["sel"]["hits"],
                        "sel_letters": None if res["sel"] is None else res["sel"]["letter_hits"],
                        "em": res["em"], "em_dyn": res["em_dyn"], "moved": res["moved"],
                        "cand_changed": res["cand_changed"], "changed": res["changed"],
                        "text_change_rate": 100.0 * res["changed"] / len(ev),
                        "letter_same_rate": 100.0 * res["same_letter"] / len(ev),
                        "changed_among_lettered": (100.0 * sum(
                            1 for i in range(len(ev))
                            if dec_letter(res["txt"][i]) is not None and dec_letter(txt[i]) is not None
                            and res["txt"][i].strip() != txt[i].strip())
                            / max(1, sum(1 for i in range(len(ev)) if dec_letter(txt[i]) is not None))),
                        "dyn_letter_hit_rate": (100.0 * res["dyn_letter_hit"] / res["n_dyn_valid"]
                                                if res["n_dyn_valid"] else float("nan")),
                        "miss": res["miss"], "n": len(ev), "em_base": em, "thr_t1": t1thr}

            if mode == "shuffle":
                agree_txt = 100.0 * sum(1 for a_, b_ in zip(res["txt"], txt) if a_.strip() == b_.strip()) / len(ev)
                if labeled:
                    dl_hit = cf[mode]["dyn_letter_hit_rate"]
                    se_dl = (dl_hit / 100.0 * (1 - dl_hit / 100.0) / len(ev)) ** 0.5 * 100.0
                    ok = (dl_hit >= t1 - 2.0 * se_dl) and (res["em_dyn"] >= em - 2.0 * st_dyn["se"] - 1e-9)
                    print(f"[CRIT] CF① shuffle（§11.17 **动态金标**）★带字母口径：「选中同一对、按新位置报它的"
                          f"字母」= T2(动态金标)={res['em_dyn']:.2f}%（base T2={em:.2f}%）| 动态正确字母命中率="
                          f"{dl_hit:.2f}% vs base T1={t1:.2f}%（2SE={2 * se_dl:.2f}pp）| 旁证：字母不变率="
                          f"{100.0 * res['same_letter'] / len(ev):.2f}% 逐字不变率={agree_txt:.2f}%"
                          f"（★带字母数据上**应当**变）| 正确项位置移动={res['moved']}/{len(ev)} ⇒ "
                          f"{'PASS 无位置泄漏' if ok else '★FAIL 位置泄漏 ⇒ 结论作废'}{smoke_note}", flush=True)
                else:
                    ok = (st["delta"] >= -2 * st["se"]) and (st["p"] >= 0.05 or st["n01"] == st["n10"] == 0)
                    print(f"[CRIT] CF① shuffle：回复逐字不变率={agree_txt:.2f}% | T2(原target) {em:.2f}% → "
                          f"{res['em']:.2f}% Δ={st['delta']:+.2f}pp 2SE_Δ={2 * st['se']:.2f}pp McNemar "
                          f"p={st['p']:.3g} ⇒ {'PASS' if ok else '★FAIL'}{smoke_note}", flush=True)
            elif mode == "replace":
                ok = res["changed"] / len(ev) >= 0.5
                print(f"[CRIT] CF② replace（§11.25 口径：原 target 已不可达 ⇒ 只看方向）| ★回复变化率="
                      f"{100.0 * res['changed'] / len(ev):.2f}%（{res['changed']}/{len(ev)}）| T2(原target)="
                      f"{res['em']:.2f}% | 字母不变率={100.0 * res['same_letter'] / len(ev):.2f}% ⇒ "
                      f"{'PASS 答案随候选集变' if ok else '★CHECK 答案几乎不随候选集变 ⇒ 与 r3a 同病（§11.12）'}"
                      f"{smoke_note}", flush=True)
            else:
                se_cf = (res["em"] / 100.0 * (1 - res["em"] / 100.0) / len(ev)) ** 0.5 * 100.0
                near = res["em"] <= max(t2thr, t1thr) + 2.0 * se_cf
                print(f"[CRIT] CF③ delete：删除正确项的条数={res['cand_changed']}/{len(ev)} | 回复变化率="
                      f"{100.0 * res['changed'] / len(ev):.2f}% | T2(原target)={res['em']:.2f}% ⇒ "
                      f"{'落到机会水平附近' if near else '★未落到机会水平附近（如实报）'}{smoke_note}", flush=True)

    # ---- 本 run 的配对原始行（--pair-hits，逗号分隔；方向恒为「本臂 − 对臂」） ----
    if a.pair_hits:
        for ph in [x.strip() for x in a.pair_hits.split(",") if x.strip()]:
            if not os.path.exists(ph):
                print(f"[CRIT] R5 配对：{ph} 不存在 ⇒ 跳过（如实报未判）{smoke_note}", flush=True)
                continue
            with open(ph, encoding="utf-8") as f:
                other = json.load(f)
            ox = other.get("arm", "?")
            st = paired_line(f"seed={a.seed}（T2 完整回复 EM，{a.arm} − {ox}）", hits, other["hits"], a.arm, ox)
            ok = st["delta"] > 0 and st["delta"] >= 2 * st["se"]
            print(f"[CRIT] R5 本 run 配对（T2）：{a.arm} vs {ox} Δ={st['delta']:+.2f}pp SE_Δ(配对)={st['se']:.2f}pp "
                  f"2SE_Δ={2 * st['se']:.2f}pp 不一致对 b={st['n01']} c={st['n10']} McNemar p={st['p']:.3g} ⇒ "
                  f"{'PASS（Δ>0 且 Δ>=2SE_Δ）' if ok else 'FAIL（未达显著）'}"
                  f"（「2 次运行同号」要另一 seed 的 hits，用 --report-r5 出总判）{smoke_note}", flush=True)
            if labeled and "txt" in other:
                lx, ly, _mx2, _my2 = paired_letters(txt, other["txt"])
                gy = other.get("gold_letters") or ev_rep["gold_l"]
                if other.get("gold_letters") and other["gold_letters"] != ev_rep["gold_l"]:
                    print("[CRIT] ★两者 gold_letters 不一致 ⇒ T1 配对不可比（如实报）", flush=True)
                hxl = [int(x == g) for x, g in zip(lx, ev_rep["gold_l"])]
                hyl = [int(y == g) for y, g in zip(ly, gy)]
                stl = paired_line(f"seed={a.seed}（T1 字母，附加）", hxl, hyl, a.arm, ox)
                print(f"[CRIT] R5(附加) 本 run 的 T1 字母配对：Δ={stl['delta']:+.2f}pp 2SE_Δ={2 * stl['se']:.2f}pp "
                      f"McNemar p={stl['p']:.3g}（非 R5 主判据；主判据看 T2）{smoke_note}", flush=True)
    else:
        print(f"[CRIT] R5 本 run 未给 --pair-hits ⇒ 只出单臂读数；配对判定（Δ / SE_Δ(配对) / b,c / McNemar p）"
              f"用 --report-r5 出{smoke_note}", flush=True)

    hp = _hits_path(a.arm, a.seed, a.smoke, a.realign_layers)
    with open(hp, "w", encoding="utf-8") as f:
        json.dump({"arm": a.arm, "variant": variant, "seed": a.seed, "k": a.k, "prefix": prefix,
                   "smoke": a.smoke, "n": len(hits),
                   "em": em, "se": se, "t1": t1, "miss": miss, "sel": sel_acc, "sel_letter": sel_lacc,
                   "mid_em": None if mid_rep is None else mid_rep["em_mid"],
                   "mid_letter": None if mid_rep is None else mid_rep["mid_letter"],
                   "mid_miss": None if mid_rep is None else mid_rep["miss"],
                   "mid_sep_rate": None if sep_ok is None else 100.0 * sum(sep_ok) / len(sep_ok),
                   "mid_e2e": mid_e2e,
                   "params": n_par, "steps": steps, "peak_mib": peak, "wall_s": wall,
                   "thr": t2thr, "t2_thr": t2thr, "t1_thr": t1thr, "max_chance": fl["max_chance_l"],
                   "max_naive": fl["max_naive_f"], "init_ok": init_worst == 0.0,
                   "hits": hits, "sel_hits": None if sel is None else sel["hits"],
                   "sel_letter_hits": None if sel is None else sel["letter_hits"],
                   "txt": txt, "mids": mids, "letters": ev_rep["ls"], "gold_letters": ev_rep["gold_l"],
                   "gold_mids": None if mid_rep is None else mid_rep["gold_mid"],
                   "cf": {m: {kk: v.get(kk) for kk in ("hits", "dyn", "txt", "letters", "em", "em_dyn", "moved",
                                                       "cand_changed", "changed", "text_change_rate",
                                                       "letter_same_rate", "dyn_letter_hit_rate",
                                                       "changed_among_lettered", "miss", "n", "em_base",
                                                       "thr_t1", "sel", "sel_letters", "mids", "dyn_mid",
                                                       "dyn_mid_hit", "dyn_e2e")} for m, v in cf.items()}}, f)
    print(f"[META] arm={a.arm} variant={variant} inject={getattr(model, 'inject', '-')} "
          f"realign_layers={a.realign_layers} k={a.k} prefix={prefix} seed={a.seed} steps={steps} batch={BATCH} "
          f"lr={LR} d={D} ff={FF} nhead={NHEAD} maxlen={MAXLEN} n_params={n_par} "
          f"可训={sum(p.numel() for p in model.parameters() if p.requires_grad)} peak_mib={peak:.0f} "
          f"T2_EM={em:.2f}% SE={se:.2f}pp T1={t1:.2f}% T1抽不出={100.0 * miss / len(ev):.2f}% "
          f"sel={sel_acc if sel_acc is None else round(sel_acc, 2)} "
          + (f"midEM={(mid_rep['em_mid'] if mid_rep else float('nan')):.2f}% "
             f"mid字母={(mid_rep['mid_letter'] if mid_rep else float('nan')):.2f}% "
             f"SEP自出={(100.0 * sum(sep_ok) / len(sep_ok)) if sep_ok else float('nan'):.2f}% "
             f"midCE厚={cmid_m if cmid_m is None else round(cmid_m, 4)} 回复CE={crep_m if crep_m is None else round(crep_m, 4)} "
             if variant == "B0b" else "")
          + f"配对CE={cm:.4f}±{cse:.4f} T2门槛={t2thr:.3f}% T1门槛={t1thr:.3f}% max_naive={fl['max_naive_f']:.3f}% "
          f"init_ok={init_worst == 0.0} smoke={a.smoke} cf={a.cf} pair_hits={a.pair_hits} ckpt={ck} hits={hp} "
          f"训练墙钟={wall:.0f}s 评估墙钟={eval_sec:.0f}s 总墙钟={wall + eval_sec:.0f}s", flush=True)
    print("[DONE] exit=0", flush=True)
    return 0



if __name__ == "__main__":
    sys.exit(main())
