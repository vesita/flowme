# AGENTS —— flowme 的 AI 入口（先读这里，再动代码）

> 给**任何接手本仓库的 AI 会话**：本文件写「先读什么 + 动手前必守什么」。
> 现状与结论在 `PROJECT_STATE.md`，纪律表体在 `methodology/RULES.md`，本文件不复述它们。

## 1. 必读顺序（读完再动手，别跳）

| # | 文件 | 读它回答什么 |
|---|---|---|
| 1 | `README.md` | 这是什么 · 目录地图 · 怎么跑 |
| 2 | `PROJECT_STATE.md` | **现状**：阶段 · 站得住的结论 · 已判死 · 待办 |
| 3 | `MEME_FRAMEWORK.md` | **设计唯一真相源**：三类卡 · 三层规范 · 黑板调度 · 度量 L0–L5 |
| 4 | `NEXT_MAINLINE.md` | **当前缺口与优先级**（判据写死在这里） |
| 5 | `methodology/RULES.md` | 纪律总表 R1–R43（**唯一表体**） |
| 6 | `methodology/*.md` | 准绳：`measurement` · `experiment-design` · `platform` · `collaboration` · `findings` |
| 7 | `DEVLOG.md` | 最近一轮做了什么 · 推翻了什么 · 踩了什么坑 |
| 8 | `METRICS.md` | 绝对数字口径（引用绝对 EM **必须注明数据版 + 调度**） |

按需：**`L3_PLAN.md`（当前主线：用卡片组出自然语言对话框架 · 选择卡/重整卡 · 判据写死）** · `L2_PLAN.md`（**已完成**：可对话卡片组，单条 `CardPipeline` 达信息上限 0.98~1.00） · `P6_PLAN.md`（语言生成接进 `cards/` 的落地判据与结果） · `NL_CARDS_REF.md`（语言卡片参考手册） · `MILESTONES.md`（里程碑判决） · `ROADMAP.md`（队列） · `speculation/`（**推测**，引用须标"推测"） · `dev-notes/`（旧框架经验，效力低于 `methodology/`）。

## 2. 动手前必守（每条都有实测出处，不是形式）

- **判据先写死再跑**：预注册落盘早于首训；要改判据必须留痕 + **旧新两个数字都报**（`methodology/experiment-design.md:46-50`）。
- **★ 写死的判据本身也要交叉复核**：判据会错，而且错法有规律 —— L3 线实证 **3 次**（①把**机会级** `1/k` 与**捷径级**地板混进 `max_naive` ⇒ 多选任务上 `≤5%` **永不可过**；②把**模型侧尺度**（随机装置的 1/k 偏离）当**数据侧判据**；③带**位置依赖标签**的数据沿用"位置不变性"判据 ⇒ 把**答对判成错**）。⇒ **判据在开跑前由另一个单元独立读一遍**；改动照上一条留痕、旧新都报（`L3_PLAN.md` §11.5 / §11.13 / §11.17 / §11.19）。
- **「没测出差异」≠「没有效果」**：只许写"该口径下未测出差异"（`methodology/README.md:12-14`）。
- **逐位口径**：logits 逐位只在**同进程同批**下可承诺；跨进程只承诺 pred/exact —— ROCm 前向有 ~1e-6 级非确定性（`methodology/measurement.md:32-38`）。
- **★ 训练轨迹不是无条件逐位可重复**：本机实测同 seed 重跑，**长序列**（~326 token）训练 40 步后逐张量 max|Δ| 已达 **5e-5~9e-5**（相对 1e-5~1.7e-5），根因是 `nn.Embedding` 反向 `index_add` 的**原子累加顺序不定**（`L3_PLAN.md` §20.1）；短序列（~30 token）@12000 步可逐位 0 ⇒ **口径**：主张"同一条轨迹"必须先跑 A/A 地板并报量级；「2 seed 同号」的准确说法是「**2 次运行同号**」。
- **比训练路径前先对齐随机流**：构造期 RNG 状态 / dropout 流消耗 / 数据序全部逐位对齐后，才谈"等价"（R34；范例 `stages/51_p4_gate2.py`）。
- **★ 卡片化是本项目前进的基础**（用户 2026-10-11 定）：推进过程中**不要试图推翻卡片化**（例如不得以"改成单体大模型"作为解法）；**可以质疑这条路**；若认为确实走不通 ⇒ **先停下、把证据摆出来、由用户决定是否放弃**；只要还有希望，**不得违背这个方向**（`L3_PLAN.md` §27）。
- **长跑**用 `systemd-run --user --unit=<名> --collect`，不用 `setsid nohup`；单卡忙则排队，**绝不杀别人的进程**（`methodology/platform.md:22-27,31-41`）。
- **`logs/` 不入库**；结论性数字必须复述进 REPORT / 提交信息（`README.md:116-117`）。
- **提交按路径显式列出**，禁止 `git add -A`（R18）。
- **动手前先 `collab_lock op=claim`**，只写自己声明的路径；不绕过别人持有的锁（`methodology/collaboration.md:8-20`）。
- **先 grep 既有脚本与文档再写新脚本**：重复造轮子会掩盖已有结论。
- **`logs/` 里的行号在仓库内不可核实**（R32）⇒ 引用只写已入库文件或提交号。
- 本机 GPU = 单卡 RX 6600 / 8G / ROCm 6.4，**每个 GPU 进程必带 `HSA_OVERRIDE_GFX_VERSION=10.3.0`**。

## 4. 最容易抓瞎的地方 → 见参考手册

语言卡片这条线上最容易卡住的几处，已整理成**参考式**文档 [`NL_CARDS_REF.md`](NL_CARDS_REF.md)：术语对照（`stages` 的 `emb/in_enc/thought/head` ↔ `cards` 的 `input.emb/input.enc/target.think/output.head`）、既有 ckpt 与各自口径、S65 的四个模式与输出行读法、训练可重复性的实测边界、复现门的口径陷阱、多会话现场。

两条最常踩的在这里各留一句：

- `cards/target.py:28-30` 的 `TargetCard` 默认带外层残差，而 `stages/09:337-339` 的 `thought` 是裸层 ⇒ 互灌 ckpt 要 `residual=False`（实测 logits 差 9.6~14.1）。
- 训练路径**不是无条件可逐位重复**：算术（序列 ~30 token）@12000 步逐位 0；GSM8K（≈500 token）200 步 A/A 就差 2~3e-3 ⇒ 比训练路径前先跑 A/A 噪声地板（`stages/65_textgen_cards.py --gate2` 内置）。

## 5. 目录地图（哪个是家）

| 路径 | 是什么 |
|---|---|
| `stages/` | **主线实验**（`NN_<问题>.py`） |
| `cards/` | **新框架骨架**：三类卡 + 黑板 + 版本化（只依赖 torch） |
| `methodology/` | 结论 / 纪律 / 准绳的家（R1–R43 在这里） |
| `speculation/` | 待验证推测（**不许与 `methodology/` 混写**） |
| `logs/` | 过程日志（不入库） |
| `dev-notes/` `experiments/` `src/` `tests/` `checkpoints/` | **旧框架 DTSeek**，只作证据与经验来源（`README.md:95-105`） |
