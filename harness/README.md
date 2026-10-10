# harness —— flowme 的推理入口

> L3_PLAN.md §21 的落地物。**不是 demo 脚本**，而是"下一阶段的共同测量面"：
> 换卡 / 加卡 / 精度 / 检查点 / 探针都从同一个入口走，结论才累积得下来。

```
harness/registry.py   卡片注册表：名字 →（构造器, 接口声明, 可判目标声明, 依赖张量, ckpt 加载器）
harness/pipeline.py   按「卡片集合 + 拓扑」装配可推理流水线；载 ckpt / 贪心生成 / 评测口径
harness/chat.py       REPL：读自然语言 → 逐轮生成回复（历史累积）
harness/probe.py      推理探针：卡数 N / 接口宽度 w / dtype → 驻留·延迟·保真
harness/cli.py        子命令：chat / say / selfcheck / probe / list-cards
harness/README.md     本文件（接口契约 + 如何加一张卡）
```

## 0. 一分钟上手

```bash
PY=.venv/bin/python        # 本机只有这个解释器装了 torch/tokenizers

$PY -m harness.cli list-cards                     # 注册表与卡组（不改任何文件也能看全）
$PY -m harness.cli chat                           # 交互 REPL（默认 CPU）
$PY -m harness.cli say --turn '你好，我的售后工单号是AS975482。' \
                       --turn '我希望按返厂检测的方式处理。' \
                       --turn '帮我催一下吧。'
$PY -m harness.cli selfcheck --n 800 --json       # 复现 B2 s1234 记录值
$PY -m harness.cli probe --n 1,2,3,4 --interface 128
```

`--device cpu`（默认）时，`cli.py` 在 **import torch 之前**清空 `CUDA_VISIBLE_DEVICES` /
`HIP_VISIBLE_DEVICES` ⇒ 连设备枚举都不发生（本机 GPU 上有别的单元在跑）。

## 1. ★ 接口契约（卡与卡之间传什么）

**卡间只传一个 `Packet`**（`pipeline.py`，唯一实现点）：

| 字段 | 形状 | 谁写 | 说明 |
|---|---|---|---|
| `tokens` | `[b,n]` int64 | 外部 | 只有输入卡读 |
| `mask` | `[b*nhead,n,n]` | 输入卡 | 因果上三角 + padding 行全 `-inf`（`cards.InputCard.causal_mask` 是唯一实现点），逐卡透传 |
| `h` | **`[b,n,d]`** | 每张"写主接口"的卡 | ★**这就是"接口"**：一个 **d 维隐态**，默认 `d=128`（`--interface w` 改的就是它） |
| `ctx_len` | `[b]` int64 | 外部 | 回复区起点；**加性注入**与可判目标的分界 |
| `aux` | dict | 各卡 `emit` | 附属张量（选择卡 → `sel_scores/sel_probs/e_sel`）；**不是接口的一部分**，下游按名字读 |
| `logits` | `[b,n,V]` | 输出卡 | 只在最后出现 |

四条口径（写死，改它们等于改契约）：

1. **接口是 d 维隐态，且不含位置**：位置只由序列轴 + `mask` + `ctx_len` 承载；
   唯一的绝对位置编码在输入卡内部（固定正弦，`cards.input.sin_pe`）⇒ 换卡/换宽度不需要重算位置。
2. **加性注入**只有一个点：重整卡把 `e_sel = Σ_j p_j h_j` **加到回复区**（`位置 >= ctx_len`），
   然后过自己的层：`h = enc(h + apply ⊙ e_sel)`。除此之外没有任何跨卡改写。
3. **拓扑不是"一条链"**：**选择卡是侧支** —— 它读输入卡的隐态、只 `emit e_sel`，
   主接口上的 `h` 仍是输入卡的输出，重整卡读的正是它。`Pipeline.topology_line()` 会打印这张图：

   ```
   input[h→h]  select[h→aux]⤵侧支{...,e_sel}  realign[h→h]  output[h→h]
   ```

   每张卡用 `slot_in` / `slot_out` 声明：`slot_out="h"` 写主接口，`slot_out=None` 表示**侧支**。
4. **顺序执行**：`Pipeline.order` 里逐卡跑，卡间不并行；这就是"动态加载"的**极限情形**
   （最坏延迟、最小驻留）—— 见 §4。

`Pipeline.interface_summary()` 把上面的声明逐卡打成机读结构；`/cards` 与 `list-cards` 都从它出。

## 2. ★ 如何加一张新卡（H-d：三步）

**第 1 步：实现约定接口。** 继承 `HarnessCard`，声明 5 个类属性、实现 `forward_card(pkt)`：

```python
from harness.pipeline import D, DROPOUT, FF, HarnessCard, Interface, JudgedTarget, NHEAD
import torch, torch.nn as nn

class GainCard(HarnessCard):
    """最小示例卡：h → h + tanh(W h) ⊙ g（g 零初始化 ⇒ 未训练时逐位恒等）。"""
    spec_name = "example_gain"                       # 注册表里的名字
    interface = Interface(d_in=D, d_out=D, injection="none",
                          doc="示例卡：h + tanh(W h) ⊙ g（g 零初始化 ⇒ 未训练时恒等）")
    deps = ("h",)                                    # 依赖张量：只读主接口
    judged = JudgedTarget(name="示例卡：没有可判目标（故意留空）",
                          metric="无（对照读数：插进 B2 后输出应逐位不变）",
                          needs=(), chance="")
    ckpt_prefix = "example_gain."                    # 本卡的 ckpt key 前缀（无 ckpt 就写 None）
    slot_in, slot_out = "h", "h"                     # 拓扑：读主接口、写主接口

    def __init__(self, d: int = D, ff: int = FF, nhead: int = NHEAD, dropout: float = DROPOUT):
        super().__init__()
        self.proj = nn.Linear(d, d)
        self.g = nn.Parameter(torch.zeros(d))        # ★零初始化：不消耗 RNG（构造确定性）

    def forward_card(self, pkt):
        pkt.h = pkt.h + torch.tanh(self.proj(pkt.h)) * self.g
        return pkt
```

**第 2 步：注册**（`registry.py`，一行）：

```python
REGISTRY.register(CardSpec(
    name="example_gain", build=GainCard, kind="example",
    interface=GainCard.interface, judged=GainCard.judged, deps=GainCard.deps,
    accepts=("d", "ff", "nhead", "dropout"), ckpt_prefix="example_gain.",
    doc="★最小示例卡：照它抄一遍 + register 就能加卡"))
```

**第 3 步：用**（不改 `pipeline.py` / `chat.py` / `probe.py` / `cli.py`）：

```bash
# 插进 B2（选择卡之后、重整卡之前）—— --cards 就是"用户侧拓扑"
$PY -m harness.cli say --arch B2+ex --turn '你好，我的售后工单号是AS975482。' \
      --turn '我希望按返厂检测的方式处理。' --turn '帮我催一下吧。'
# 或者自己拼拓扑名
$PY -m harness.cli list-cards
$PY -m harness.cli chat --cards input:input,select:select,example:example_gain,realign:realign,output:output
```

**这张示例卡已经注册好**（`registry.py::ExampleGainHarnessCard`，与上面逐字同形），
所以现在就能跑，也可以直接拿它当模板抄。跑出来的读数：

```
$PY -m harness.cli probe --n 1,2,3,4 --hd      # --hd 附带跑这一条
[H-d] 示例卡 example_gain 插进 B2（未训练，g 零初始化）：
      logits max|Δ| = 0.000e+00 | 贪心文本逐条相同 = True | 参数 2700288 → 2716928（+16640）
      ⇒ 加卡本身零增量、纯延迟（§18.1 的实测复现）
```

### 声明**可判中间目标**（`judged`）

```python
judged = JudgedTarget(
    name="选择卡：在候选层里挑出与历史 (实体,条件) 一致的那一项",
    metric="字母准确率（argmax 位置的字母 == 候选集里唯一双槽位匹配项的字母）",
    needs=("cands", "cend", "n_cands", "ent", "cond"),   # 算这个目标需要记录里哪些字段
    chance="1/k（k=8 ⇒ 12.50%）",                          # ★机会水平要写清，别把机会级当增益
    doc="训练时 q 用金标位置做 CE（中间监督）；推理时用模型自己的 p。")
```

`judged=None` 是**合法且诚实**的声明：它等于说"这张卡现在没有可判目标"，
按 §18.1 那张卡就只是**延迟**（示例卡正是这个对照）。`list-cards` 会把 `—` 打出来。

### 声明**依赖哪些张量**（`deps`）

`deps` 是给**冻结 / 微调 / 显存计算**用的显式声明，取值：

| 写法 | 含义 |
|---|---|
| `"tokens"` | 读 `Packet.tokens` |
| `"h"` / `"mask"` / `"ctx_len"` | 读卡间接口隐态 / 掩码 / 回复区起点 |
| `"aux:<名>"` | 读上游卡 `emit` 的张量（如 `"aux:e_sel"`） |

例：`select.deps = ("h","mask","aux:cand_end","aux:n_cands")`、`realign.deps = ("h","mask","ctx_len","aux:e_sel")`。
**只读 `h` 的卡可以被冻结或单独微调而完全不改别的卡**（示例卡就是这样）。

### ckpt 装载约定

每张卡声明 `ckpt_prefix`（本卡在外部 ckpt 里的 key 前缀）与 `ckpt_root_keys`
（不带前缀的 key，如 79 的选择卡打分向量 `q`）。`Pipeline.load_ckpt()` 会逐卡报
`loaded / missing / unexpected / shape_mismatch` —— **不允许静默丢 key**。

## 3. 子命令与判据落点

| 子命令 | 做什么 | 判据 |
|---|---|---|
| `chat` | REPL：逐轮输入 → `flowme>` 回复；`/cards` `/reset` `/examples` `/quit` | H-a |
| `say` | 非交互版：`--turn` 依次喂进**同一个会话**（历史累积） | H-a |
| `selfcheck` | 800 条测试集上复现 B2 s1234 = T2 75.88% / T1 80.00% / 选择 88.38%；另跑 `[INIT-CHECK]` 与 `cards/` 正规路径对拍 | H-b / H-f |
| `probe` | `--n 1,2,3,4 --interface 128` → 驻留·延迟·保真（R7-b/c/d/e） | H-c |
| `list-cards` | 注册表 + 卡组 + 逐卡完整声明（`--json` 机读） | H-d / H-e |

**H-f（可核验）**：`selfcheck` / `say` / `chat` / `list-cards` 的 stdout **逐字节可复现**
（贪心解码、固定 seed、stdout 不含耗时；耗时只走 stderr）。自查：

```bash
$PY -m harness.cli selfcheck --n 800 > /tmp/r1.txt 2>/dev/null
$PY -m harness.cli selfcheck --n 800 > /tmp/r2.txt 2>/dev/null
sha256sum /tmp/r1.txt /tmp/r2.txt      # 两行必须一样
```

`probe` 的输出分两层：**Tier 1**（参数字节 / 接口 / 活载荷峰值 / 保真 / 批指纹）是确定性算术，
`--no-measure` 时整个 stdout 逐字节可复现；**Tier 2**（墙钟 / 进程 RSS）**物理上不可重复**，
每次都会变，`[DET]` 行会点明 —— 这是对 H-f 的落地口径，不是"悄悄放过"。

### 实测读数（本机 CPU，rocm torch 2.9.1，threads=6；供引用时对齐口径）

探测批 = `logs/76_dialog_test.jsonl` 前 32 条 ctx-only，右补 PAD 到 512，批指纹 `5c2dd0453edeaf0a`；
链 = `input → N×enc_layer → output`；**只前向**。

`--n 1,2,3,4 --interface 128`（fp32，trained 权重）：

| N | 等效参数 | 层参数 | 单卡最大 | 接口 | 活载荷峰值 | 驻留·顺序(推断) | 驻留·全常驻 | 等效/驻留(顺序) | per-batch | per-token | 保真 max\|Δ\| |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 2501888 | 198272 | 774.5KiB | 8.0MiB | 648.1MiB | 657.9MiB | 657.9MiB | 0.015 | 298.6ms | 18.2us | 0.0 |
| 2 | 2700160 | 396544 | 774.5KiB | 8.0MiB | 648.1MiB | 657.9MiB | 658.7MiB | 0.016 | 392.6ms | 24.0us | 0.0 |
| 3 | 2898432 | 594816 | 774.5KiB | 8.0MiB | 648.1MiB | 657.9MiB | 659.4MiB | 0.017 | 500.4ms | 30.5us | 0.0 |
| 4 | 3096704 | 793088 | 774.5KiB | 8.0MiB | 648.1MiB | 657.9MiB | 660.2MiB | 0.018 | 576.1ms | 35.2us | 0.0 |

* **R7-a 保真**：4 档全部**逐位相同**（`max|Δ| = 0.0`）⇒ "顺序执行不改语义"成立（fp32、同进程同批）。
* **R7-b**：参数口径 顺序执行**恒为 9.8 MiB**（= max(单卡) + interface 卡），全常驻 9.8 → 12.1 MiB（N=4 时 1.232×）。
* **R7-c**：`等效参数量 ÷ 驻留` = 0.015 → 0.018（几乎不涨）——因为**驻留由激活主导**
  （活载荷 648.1 MiB 里 537 MiB 是 logits、134 MiB 是 mask），不是由卡参数主导。
* **R7-d**：延迟 per-batch 298.6 → 576.1 ms（每加一张层卡约 +92 ms），per-token 18.2 → 35.2 us。

### 接口税（`--interface 64,128,256`，N=1）

| w | 等效参数 | 单卡最大 | 接口 | 驻留·全常驻 | 等效/驻留 | per-batch | per-token |
|---|---|---|---|---|---|---|---|
| 64 | 1222784 | 324.2KiB | 4.0MiB | 648.9MiB | 0.007 | 295.5ms | 18.0us |
| 128 | 2501888 | 774.5KiB | 8.0MiB | 657.9MiB | 0.015 | 352.4ms | 21.5us |
| 256 | 5256704 | 2.0MiB | 16.0MiB | 676.7MiB | 0.030 | 491.7ms | 30.0us |

w 64→256：参数量 ×4.3、接口字节 ×4、**参数口径驻留 4.8 → 20.6 MiB（×4.3）**、
活载荷只 +1.9%、延迟 ×1.66。⇒ 本档"接口税"主要落在**参数与延迟**上，不在总驻留上（激活占绝对多数）。
★ **w ≠ 128 的档用的是 fresh 权重**（79 的 ckpt 形状不符）⇒ 只量了**形状/驻留/延迟**，
**"接口宽度 w 与质量"这件事仍未测**（这是 §18.2 方向 I 的最小可判实验，留给下一单元）。

### dtype（`--dtype fp16`，N=1,2，CPU）

| dtype | 等效参数 | 单卡 | 接口 | 活载荷峰值 | 驻留·全常驻 | per-batch | per-token |
|---|---|---|---|---|---|---|---|
| fp32 | 2501888 | 774.5KiB | 8.0MiB | 648.1MiB | 657.9MiB | 298.6ms | 18.2us |
| fp16 | 2501888 | 387.2KiB | 4.0MiB | 324.1MiB | 329.0MiB | 3483.5ms | 212.6us |

fp16 让**驻留减半**（657.9 → 329.0 MiB，因为激活与参数一起减半），但在**这块 CPU 上慢 ~10×**
（CPU 没有快的 fp16 通路）⇒ 与 r6 在 GPU 上的结论（fp16 省一样多还不慢）**不能互相搬运**。
保真在 fp16 档仍然 `max|Δ| = 0.0`（顺序 vs 整体图各自在 fp16 下逐位相同）。

## 4. ★ 诚实边界（做不到的事，逐条写明）

1. **当前唯一的"卡"是**同构 encoder 层卡**。我们没有第二种尺寸/功能的卡 ⇒
   `probe` 的 N>2 是把 `select.enc` / `realign.enc` 交替复制出来的（`--weights trained`）；
   参数量 2700288（input 1246848 + select 198400 + realign 198272 + output 1056768）。
2. **对话能力限于自造合成分布**：5 族 × 20 条 DET 模板（共 100 条；`logs/76_*`，k=8 带字母，
   回复必须含被选中候选的字母）。⇒ **分布内流畅正确；分布外会给"看着像但内容错"的回复**。
   REPL 里分布外会打 `[OOD?]`，但这个标记是**文法级**的（词表外实体/条件、非模板问法、"先条件后实体"、
   混族），**不是学出来的 OOD 检测器**。它只能提示"别只看流畅度"，不能替你判定内容对错。
   实测样例（`say` 的第 4 轮）：`帮我取消这个工单，现在就退钱。` ⇒
   `[OOD?] 问法不在 4 条轮3模板里 ⇒ 已用模板问法替代`，回复
   `AS975482 已按返厂处理，请保持电话畅通（B）。` —— 句子通顺、字母正确，**但没有回答"取消/退钱"**。
3. **动态加载尚未实现**：`probe` 的顺序执行是它的**极限情形**（最坏延迟、最小驻留）；
   探针里的"驻留 ≈ max(单卡)+接口"是**推断**（假设同时只实例化一张层卡），
   **没有实测磁盘 paging**。CPU 上进程 RSS 的分辨率（~1 MiB）也**高于**一张层卡的参数差（0.79 MiB）
   ⇒ 跨 N 的驻留差别在 CPU 上**测不出来**：`--n 1,2,3,4` 的独立子进程 VmHWM 实测**逐档完全一样**
   （4488.7 MiB，逐卡增量 0.0/0.0/0.0 MiB）。如实报"该口径下未测出差异"，
   **不等于**"没有差异"——参数口径的差异是精确可算的（下表 9.8 → 12.1 MiB）。
   另注：`活载荷峰值 648.1 MiB` 里 **537 MiB 是 logits（[32,512,8192]）、134 MiB 是 causal mask**
   ⇒ 这个尺度上"驻留"由激活主导，不由卡参数主导；读 R7-b 请读**参数口径**那一列。
4. **REPL 的历史应答句是 harness 的模板**（尾标 `(模板)`）：卡组只被训练生成**最后那句回复**
   （数据里的历史 `<resp>…<eos>` 是数据给的）。卡组真正在做的事只有三步：
   **读历史取 (实体,条件) → 在候选层里匹配 → 报出该候选的字母并套 DET 句面**。
5. **`stages/83_card_demo.py` 未改动**（本次硬约束是"不改任何既有文件"）⇒
   "83 改成 harness 的薄封装"这件事**还没做**，留给下一个单元。
   现在 harness 自己已完整包含"载模型 + 生成 + self-check"，不依赖 83。
6. 唯一被 harness **读**的既有文件是数据/文法一侧的
   `stages/72_nl_dialog_gen.py`、`stages/74_nl_dialog_cand_gen.py`、`stages/76_letter_cand_gen.py`
   （族定义 / 候选构造 / 标签仪器 —— 复用而不另写一份，避免口径漂移）。
   模型构造 100% 走 `cards/`（`cards.InputCard` / `cards.OutputCard` / `cards.TargetCard(residual=False)`
   / `cards.base.encoder_layer`），`selfcheck` 里有机器对拍：
   `native3 == cards.CardPipeline(residual=False)`（参数相等、同一权重下前向 max|Δ| = 0）。

## 5. 加载一张已训好的卡组（0 依赖脚本）

```python
from harness.pipeline import load_model, greedy_gen, prep_record, load_jsonl, pack

pipe, rep = load_model("B2", "logs/79_ckpt_B2_k8_s1234.pt", seed=1234)   # ★先 manual_seed 再构造
recs = load_jsonl("logs/76_dialog_test.jsonl")[:4]
for r in recs:
    prep_record(r)
print(greedy_gen(pipe, recs))
```

`load_model(arch, ckpt, seed, device, dtype)` 是唯一入口：内部先 `torch.manual_seed(seed)`
再建卡（§9.7 的缺陷就是"构造发生在播种之前"），然后逐卡灌 ckpt 并 `eval()`；
返回 `(pipe, ckpt_report)`。想换卡组就用 `arch=` 或 `build_custom([(实例名, 注册名), …])`。
