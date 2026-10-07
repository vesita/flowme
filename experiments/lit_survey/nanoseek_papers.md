# nanoSeek 论文原文 / 文献清单侦察（LP1）

**判定：有可用材料（给清单）** —— 6 篇原文 + 4 份文献型 md；其中直接对准 flowme 缺口的 2 份。

## 1. 文件清单（实测 find + ls + pdfinfo/pdftotext）
| 路径（相对 nanoSeek/） | 大小 | 类型 |
|---|---|---|
| data/paper/2506.07900v2.pdf | 2,590,544 B | PDF 8 页 |
| data/paper/2601.20994v1.pdf | 1,258,510 B | PDF 22 页 |
| data/paper/2602.09003v1.pdf | 994,340 B | PDF 4 页 |
| data/paper/2607.24653v2.pdf | 1,790,685 B | PDF 47 页 |
| data/paper/deepseek.md | 159,518 B | arXiv:2606.19348 全文 HTML→md |
| data/paper/GLM.md | 169,202 B | arXiv:2602.15763 全文 HTML→md |
文献型 md：dev-notes/58（3,714B 调研）、dev-notes/70（2,360B）/71（2,218B）、pre-research/DESIGN.md（10,064B）、dev-notes/77（3,504B）/79（14,419B）、README.md:482（借鉴论文清单）
无 .bib/.tex（find 实测 0 命中）；.venv 内 11 个 matplotlib 图标 pdf 与 huggingface papers.py 与文献无关（实测，已剔除）。仓内无 doc/ docs/ notes/，文献笔记在 dev-notes/。

## 2. 逐篇（标题由 pdfinfo + pdftotext 首页实读；md 由 read 实读）
| 材料 | 主题一句话 | 分类 |
|---|---|---|
| MiniCPM4（2506.07900） | 端侧高效 LLM：InfLLM v2 稀疏注意力 + UltraClean 数据 + ModelTunnel 搜索 + chunk-wise rollout RL | 其它（效率/训练算法） |
| The Depth Delusion（2601.20994） | 架构条件缩放律：宽应比深快 2.8×，超 D_crit 加层反而升 loss | 其它（架构缩放） |
| Data Science…Part I（2602.09003） | 面向 AGI 的分层数据管理 | 其它（数据管理） |
| Kimi K3（2607.24653） | 2.8T MoE + Kimi Delta Attention + Attention Residuals + agentic RL | 表示与通信 |
| DeepSeek-V4（deepseek.md） | mHC 超连接（多残差流）+ CSA/HCA 混合注意力 + Muon | 表示与通信 |
| GLM-5（GLM.md） | DSA 稀疏注意力 + 异步 agent RL 基建 | 其它（RL 基建） |
| dev-notes/58 调研 | BLT / MobileLLM / Titans / RWKV-7·DeltaNet / BitNet / SimPO / MTP 八项调研 | 文献清单·表示与通信为主 |
| pre-research/DESIGN.md | n-bit 指令直通层：前向二值化→2^n 指令、反向 STE 热身、GRPO 逐位信用、NOP 旁路路由 | 离散瓶颈 + 组合与路由（设计文档，非论文） |
| dev-notes/70 / 71 | 微型算术模型进位神经元点二列相关探针 + Scratchpad 进位链 | 解释性 |
| dev-notes/77 / 79 | PKM / Titans 惊喜度写入 / RETRO-lite 引用 + 本仓实测 | 表示与通信 |
无「未识别」项：6 篇原文标题全部实读。

## 3. ★ 可落地清单（论文 → 方法 → flowme 环节 → 可判实验）
1. `pre-research/DESIGN.md` → 前向二值化 + 反向 STE 热身（GRPO 段不穿离散）→ 离散瓶颈（E16）→ 卡选择头 STE 热身 vs 冷启动 argmax 同预算 A/B，判据 = 任务 acc 差 > 2SE 才判 STE 有效。
2. `pre-research/DESIGN.md` → 2^n 指令可学路由 + NOP 旁路 → 手写调度 → 把手写调度表参数化为可学路由，同预算 A/B，判据 = 可学臂 ≥ 手写臂 − 2SE 才算「手写可替代」，否则确认必须手写。
3. `DeepSeek-V4` mHC 多残差通道 → 黑板/卡间共享空间 → 黑板加 mHC 式 mixing，判据 = 消融后下游 CE 不升（≤2SE）即「黑板是装饰」。
4. `dev-notes/70` 点二列相关探针 → 「读出 ≠ 会用」→ 思维卡做探针 + 激活擦除，判据 = 探针高而擦除后任务 CE Δ ≤ 2SE ⇒ 证实读出≠会用并给出数值。
5. `Kimi K3` Attention Residuals → 卡间信息流 → 卡间加残差直连头，判据 = 长链任务 CE Δ > 2SE 才保留。
6. `MiniCPM4` chunk-wise rollout 负载均衡 RL → 增益对训练量敏感 → 同 token 预算下均匀 vs 不均匀卡使用率，判据 = 均匀臂 acc 提升是否 > 2SE。
7. `dev-notes/58` MobileLLM 深窄 vs `Depth Delusion` 浅宽（dev-notes/83:813 已标「冲突未解释」）→ 卡片数 vs 卡内深度 → 固定参数下 多卡浅 vs 少卡深，判据 = 两臂 val CE + 任务 acc 配对差。

## 4. 实测 / 推断边界
- 实测：文件存在与大小（find/ls）、PDF 标题作者（pdfinfo + pdftotext 首页）、md 内容（read/grep）。
- 推断：4 篇 PDF 与 flowme 缺口的相关性（只读首页摘要，方法与实验章节未读）；清单 1–7 的「环节」对应是本单元判断，非原文结论。
- 仓内无 logit-lens / SAE / activation patching 类解释性文献（grep 实测 0 命中）；nanoSeek 只读前后 git status --porcelain 均为空。
