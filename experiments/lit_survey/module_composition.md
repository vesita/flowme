# 独立训练的模块能否组合？—— 条件性正面 + 无条件否定（调研）

> **一句话结论**：文献给「独立训练后组合」的是**条件性正面 + 无条件否定**：
> **正面（都带条件）**：共享底模的外挂专家（Branch-Train-Merge / Train Separately Merge Together）·
> 近线性区的 task arithmetic · 严格同步的流水线（GPipe）⇒ 可做到接近等价，
> 条件分别是「**共享联合锚点**」「**小步长**」「**严格同步**」。
> **否定（无条件）**：组合性泛化（SCAN/COGS 未见组合 ≈ 0）· 置换对称/basin（Git Re-Basin 朴素平均失败）·
> 本地漂移（SCAFFOLD 收敛点 ≠ 集中训练）。
> ⇒ **可行的唯一已证形态 = 「共享联合底模 + 独立外挂模块 + 显式组合步骤」。**
> 对 `[n,d]` 瓶颈：**QSGD 的收敛误差界随维度 d 增长** ⇒ 瓶颈维度会直接进误差（这是一条可画的曲线）。

## 1. 方向总表

| # | 方向 | 机制 | 代表工作（标题/首作者/年份/arXiv 或会议） | 链接 | 证明了什么 | 边界/何时失效 |
|---|---|---|---|---|---|---|
| 1 | 模块化 DL 综述 | 模块 + 路由 + 聚合 | Modular Deep Learning, Pfeiffer, 2023, arXiv:2302.11529 | https://arxiv.org/abs/2302.11529 | 形式化「独立模块 + 路由 + 聚合」，组合靠显式聚合层 | 综述自认组合性是开放难点，非定理 |
| 2 | 组合性泛化 | OOD 组合 | SCAN: Generalization without Systematicity, Lake & Baroni, 2018, arXiv:1711.00350 | https://arxiv.org/abs/1711.00350 | 未见组合准确率 ≈ 0（系统性失败） | 训练覆盖过的组合仍能过 |
| 3 | 语义组合 | 同上 | COGS, Kim & Linzen, 2020 EMNLP, arXiv:2010.05465 | https://aclanthology.org/2020.emnlp-main.731/ | 换结构即崩 | 中介表示不必是组合的 |
| 4 | Adapter 组合 | 门控融合 | AdapterFusion, Pfeiffer, 2020, arXiv:2005.00247 | https://arxiv.org/abs/2005.00247 | 独立 adapter 直接拼不行，**须再训融合层** | 融合层要数据、要端到端 |
| 5 | LoRA 组合 | 搜系数 | LoraHub, Huang, 2023, arXiv:2307.13269；Composing PEFT Modules, Ostapenko, 2023 NeurIPS, arXiv:2306.14870 | https://arxiv.org/abs/2307.13269 ; https://arxiv.org/abs/2306.14870 | 可算术组合，**但要搜系数** | 非零训练的组合 |
| 6 | task arithmetic | 权重差加减 | Editing Models with Task Arithmetic, Ilharco, 2022, arXiv:2212.04089 | https://arxiv.org/abs/2212.04089 | 表面等价端到端 | ↓ 见 #7 |
| 7 | TA 的边界 | 线性化 | Task Arithmetic in the Tangent Space, Ortiz-Jimenez, 2023 NeurIPS, arXiv:2305.12827 | https://arxiv.org/abs/2305.12827 | **只在小步长近线性区成立** | 大 lr / 多步即崩 |
| 8 | 参数平均 | model soup | Model soups, Wortsman, 2022 ICML, arXiv:2203.05482 | https://arxiv.org/abs/2203.05482 | 同初始化 + 同超参可平均不掉点 | **必须共享初始点** |
| 9 | 平均失效 | 置换对称 / basin | Git Re-Basin, Ainsworth, 2022, arXiv:2209.04836；LMC, Frankle, 2019, arXiv:1912.05671 | https://arxiv.org/abs/2209.04836 ; https://arxiv.org/abs/1912.05671 | **独立训练落不同 basin，朴素平均失败** | 只对齐置换，仍需同架构 |
| 10 | 合并干扰 | 符号冲突 | TIES-Merging, Yadav, 2023 NeurIPS, arXiv:2306.01708；sur., Song, 2026, arXiv:2603.09938 | https://arxiv.org/abs/2306.01708 | 冲突机制；修剪 + 选举可缓解 | **缓解 ≠ 等价** |
| 11 | MoE 专家 | 稀疏升级 | Sparse Upcycling, Komatsuzaki, 2022, arXiv:2212.05055 | https://arxiv.org/abs/2212.05055 | 专家从**联合训好**的 dense ckpt 分叉 | **没有联合起点，收益消失** |
| 12 | MoE 独立训（正面）| 先独立后合并 | Branch-Train-Merge, Li, 2022, arXiv:2208.03306；Train Separately, Merge Together, Morrison, 2026, arXiv:2604.18473 | https://arxiv.org/abs/2208.03306 ; https://arxiv.org/abs/2604.18473 | **专家可完全并行独立训 + 组合，LM 持平/略胜** | **只对专家层；共享底座仍联合训** |
| 13 | pipeline | 微批同步 / 有界延迟 | GPipe, Huang, 2019, arXiv:1811.06965；One-Step Gradient Delay, Zmushko, 2026 ICML, arXiv:2606.30634 | https://arxiv.org/abs/1811.06965 | **同步流水线数学等价端到端**；一步延迟不是障碍 | 需 micro-batch 重算；延迟超界即偏离 |
| 14 | 通信瓶颈 | 量化压缩 | QSGD, Alistarh, 2016, arXiv:1610.02132；Deep Gradient Compression, Lin, 2017, arXiv:1712.01887 | https://arxiv.org/abs/1610.02132 ; https://arxiv.org/abs/1712.01887 | **收敛误差界随维度 d 增长 ⇒ 瓶颈维度直接进误差** | 降带宽有代价 |
| 15 | 明确否证 | 本地漂移 / 模块化计算 | SCAFFOLD, Karimireddy, 2020 ICML (PMLR v119)；Routing Networks, Rosenbaum, 2019, arXiv:1904.12774 | https://arxiv.org/abs/1904.12774 | **各自本地训练 ⇒ 漂移，收敛点 ≠ 集中训练** | 差异可缩小、不可消除 |

## 2. 转成可判实验的判据（**S25 直接采用**）

| # | 判据 | 写法 |
|---|---|---|
| **a** | **主判据（配对）** | 同算力「独立训 → 拼装」vs「端到端」的 EM，配对 Δ ± SE；**\|Δ\| ≤ 1SE 且 2 seed 同号 = 过**；**Δ < −3pp 且 SE 不覆盖 0 = 不过** |
| **b** | 线性化阈值 | 扫 `‖Δθ‖/‖θ‖` 找 EM 崩塌拐点，须**高于预期训练步数**（机制见 #7） |
| **c** | ★ **共享锚点** | 各卡从**同一联合 ckpt** 分叉（#11/#12 预测**可组合**）；**随机初值各训 ⇒ #9 预测失败** |
| **d** | 接口匹配 | 冻结上游只训下游 vs 联合训的 EM 差；**★「拼装后不再训任何参数仍达标」才算真组合**，否则"需要融合层"即被证伪 |
| **e** | 带宽 | 画 **EM-vs-bit** 曲线，须在达到 **T2 门槛**前仍可用（#14） |
| **f** | OOD 组合 | 留出组合上 **EM ≈ 0** ⇒ 「**可搬运 ≠ 可组合**」（#2/#3，与本项目 Identity 换卡 EM = 0 同构） |
| **g** | 已有否证性证据 | **#9 + #2/#3 已明确否证「任意独立训练后直接组合 = 端到端」** |

## 3. 未验证项

- **所有 `arxiv.org` 链接未逐条点开**（`web_fetch` 全域不可用），只到「同 id + 同标题 ≥2 来源一致」；
- **SCAFFOLD 的 arXiv id 1910.06378 未两来源确证**（仅会议 / PMLR 佐证）；
- **MergeME（ACL 2025.naacl-long.117）** 标题两来源一致但**首作者未核实** ⇒ 未入表；
- arXiv **2602.18960 / 2605.08292 / 2606.07881** 为 2026 新 id，**仅标题两处一致，未读正文**；
- **2604.18473 的具体数值未读原文**（仅检索摘要级）；**1809.02839 标题单来源已弃用**。
