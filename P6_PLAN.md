# P6_PLAN —— 自然语言卡片落地

> 判据在本文件先写死，再跑（`methodology/RULES.md` 的预注册纪律）。
> 本文件只写现状 + 本轮要跑的门；结果另起 `## 结果` 段，跑完补。

## 0. 目标与边界

**目标**：把**自然语言**文本生成闭环接进 `cards/` 三类卡（输入卡 → 模因 → 思维卡 → 模因 → 输出卡）。

**本轮只做第一步**：确认**框架能承载语言生成任务** —— 只要求跑通与数值可复现，**不要求提升能力**（原文：`NEXT_MAINLINE.md:5-8`）。

边界：不动 `cards/*.py` 的接口（形状契约已经够用）；不动 `stages/09-12`；nanoSeek 只读。

## 1. 现状（一句话）

- `cards/` 只落地了合成算术 `add_3d`（`cards/example_add3d.py`、`example_attribution.py`、`example_versioning.py`、`exp_struct_ablation.py`）。
- 文本生成闭环只在 `stages/09_textgen_gsm8k.py`–`12_answer_only.py`，**未走 `CardPipeline`**；`cards/` 里没有文本侧驱动、没有 tokenizer/生成/评估。
- **形状层零冲突**：`stages/09` 的 `ids(b,n) → (b,n,d) → (b,n,d) → (b,n,V)` 与 `cards/base.py:5-10` 写死的契约一致（`InputCard` = emb+正弦PE+1层 Enc，`TargetCard` = 模因→模因，`OutputCard` = `Linear(d,V)`）。
- **唯一的逐位陷阱**：`cards/target.py:28-30` 的 `TargetCard` 默认带外层残差 `x + think(x)`，而 `stages/09:337-339` 的 `thought` 是**裸层** ⇒ 必须 `residual=False` 才逐位等价。
- M1（同一闭环的主判据）**判决不通过**（`MILESTONES.md:69`）：触线 = ③反事实未测、④未过地板（S12 A 臂 EM 2.17% vs 地板 1.89%）。

## 2. 第一步：语言版门 0/1/2 + 复现门（判据写死）

产物：`stages/65_textgen_cards.py`（新）、`logs/65_*`。

| # | 门 | 判据（写死） | 成本 |
|---|---|---|---|
| 门 0 | 数据路径 | `[DATA]/[TRUNC]/[SPAN]/[BUCKET]` 与 `logs/12_answer.log:5-9` 一致（blocks=7396 parsed=7394 bad=2 records=7394；train=6655 test=739；任一截断=152/7394=2.1%；span_mismatch=0），并落 train/test 指纹 sha256 | 分钟级（零训练） |
| 门 1 | 前向逐位 | 三个既有 ckpt（`logs/09c_ckpt_seed1234.pt` / `11_ckpt_seed1234.pt` / `12_ckpt_seed1234.pt`）灌进 `CardPipeline(residual=False)`，`max|Δlogits| == 0.0e+00`，键映射 missing=unexpected=0 | 分钟级（零训练） |
| 门 2 | 训练路径逐位 | 同 seed：`Cards` 与 `CardPipeline` 初值 `max|Δθ| == 0`；同一 200 步训练循环后仍 `max|Δθ| == 0`、末步 loss 相同 | 分钟级 |
| 复现门 | 既有读数 | 零训练：`11_ckpt` 走 cards 路径 EM = 0.00%（`logs/11_textgen.log`）、`12_ckpt` EM = 2.17%（`logs/12_answer.log`）；`|ΔEM| ≤ 0.13pp` 或逐位一致 | 分钟级 |
| 门 3 | 全量重训 | cards 路径 + s12 配置（seed=1234、3000 步）重训，与 `logs/12_ckpt_seed1234.pt` **逐张量** `max|Δθ| == 0`（不许用文件 sha，R43）且 EM 复现 2.17% | ~21min（S12 实测 1222s/seed） |

**判死线**：门 1 或门 2 不为 0 ⇒ 停在诊断、**不跑门 3**；不许改口径硬凑。
**失败时的动作**：定位到「卡接口 / 数据 / 训练循环」的哪一处差异（`NEXT_MAINLINE.md:8` 的判据②）。

## 3. 第二步（未跑）：让语言卡片出非平凡能力

现状原因（现状读数）：S12 A 臂 3000 步 EM 2.17% ≈ 地板 1.89%（`logs/12_answer.log:107-109`）；S11 完整 CoT 臂 EM 0.00%（`logs/11_textgen.log:109`）。最可能原因 = 数据少 / 步数少 / d=128 太小（脚本自己写的结论，未拆开检验）。

待定项（`MEME_FRAMEWORK.md:195-200`）：① `d` 取值；② 第一张输入卡模态（建议文本）；③ 脑区卡数；④ `n` 上限与变长处理。

可用的既有判据（引用，不重写）：保真/可用四门 `MEME_FRAMEWORK.md:180-187`；"强制经过卡 + 容量瓶颈"四臂 A/B/C/D `MEME_FRAMEWORK.md:347-362`。

## 4. 结果（2026-10-09 22:2x · S65）

产物：`stages/65_textgen_cards.py`（cards 路径驱动；数据 / 切分 / span loss / 贪心 EM 逐字搬运自 `stages/12_answer_only.py`，唯一改动 = 模型换成 `CardPipeline`）。
日志：`logs/65_gates.log`、`logs/65_gate2.log`（`logs/` 不入库 ⇒ **关键数字复述在下面**，R32）。

| 门 | 判据 | 实测 | 判定 |
|---|---|---|---|
| 门 0 数据路径 | 与 `logs/12_answer.log` 逐字同 | `blocks=7396 parsed=7394 bad=2 records=7394`；`train=6655 test=739`；任一截断 `152/7394 (2.1%)`；`span_mismatch=0`；指纹 `train_fp=3c2caed4037013bc` / `test_fp=eb1245b165c6ae71` | **PASS** |
| 门 1 前向 | 三个既有 ckpt 灌 `CardPipeline(residual=False)`，**同进程同批** `max|Δlogits|=0` | s09c / s11 / s12 全部 `0.000e+00`，`missing=unexpected=0`（n=479）；CPU 侧独立复算同为 0 | **PASS** |
| 门 2-a 随机流对账（R34） | 构造期全局流 / dropout 流消耗 / 数据序 | 构造后 CPU+CUDA 全局流逐位同；两侧都消耗 CUDA 流且**消耗后逐位同**（= 同一批 dropout 掩码）；200 步 = 6400 样本 ≤ 6655 ⇒ 不触发重洗 | **PASS** |
| 门 2-b 训练路径 | **旧判据**：200 步后 `max|Δθ| == 0`（逐位） | 第 1 步：A/A = 0、框架 = 0；第 200 步：**A/A = 2.902e-03**、框架 = 1.225e-03 | **逐位口径不可用** |
| 复现门 / 门 3 | — | 未跑 | — |

### 4.1 判据订正（留痕：旧 ❌ 与新 ✅ 两个数字都报，依据 `methodology/experiment-design.md:46-50`）

- **旧判据 ❌**：门 2 = 训练 200 步后 `max|Δθ| == 0`。
- **新判据 ✅**：门 2 = ① 随机流结构逐位对齐（已 PASS）；② 先测 **A/A 噪声地板**（同代码、同 seed、同过程跑两次）；③ 框架差 **≤ A/A 噪声** ⇒ 判「该口径下未测出框架差异」，**不许写成"没有差异"**。
- **订正理由（实测，不是"跑不过就算了"）**：A/A = 2.902e-03 > 框架 = 1.225e-03 ⇒ 差异由**训练本身不可逐位重复**解释，与 cards / stages 无关。
- **时间**：订正写于 2026-10-09 22:2x，**晚于**首次门 2 实测（旧版本 `[G2-200] max|Δθ|=1.562e-03 ⇒ FAIL`），**早于**门 3 首训 ⇒ 门 3 未用新判据回头看数据。

### 4.2 边界（必须一起读）

- A/A ≠ 0 只在**本设置**（GSM8K 序列 ≈480–500 token）观测到；**S51 的算术设置**（序列 ~30 token、d=128/ff=512/batch=32）在 GPU 上 @3000/6000/9000/12000 步仍 **逐位 0**（`stages/51_p4_gate2.py` + 提交 `f4a8f1a`）。
  ⇒ **"训练可逐位重复"与序列长度 / 归约规模有关**。【推断，本次**未**做单变量扫描（同模型不同序列长度）】
- 门 1 的 `0.000e+00` 是**同进程同批**读数；`methodology/measurement.md:32-38` 明确 ROCm 前向有 ~1e-6 级**跨进程**非确定性 ⇒ 不许写成跨进程结论。
- 门 3 从此只按统计判据走（EM Δ），权重差**并列** A/A 噪声报告。
