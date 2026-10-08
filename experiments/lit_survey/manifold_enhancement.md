# 「流形增强」的网络设计与消融协议（调研）

> 一句话结论：最便宜且判据最硬的落点是「**训练期隐层插值 + 谱/有效秩惩罚 + Jacobian 等距惩罚**」三件 **loss-only** 改造，
> 全部保持 `[n,d]→[n,d]`；**双曲/黎曼/流**要先证明其映射不等于「残差 + LayerNorm」才值得付代价；
> **VSA 循环卷积**是唯一 **0 参数、天然卡内即插**的算子基型。

## 1. 方向总表

| 方向 | 机制 | 代表工作（标题/首作者/年/arXiv 或会议） | 链接 | 能否落进 `[n,d]→[n,d]` 卡内 | 代价 | 可判判据 |
|---|---|---|---|---|---|---|
| 1 流形正则 | 图拉普拉斯/近邻平滑：惩罚近邻样本表示差 | Manifold Regularization, Belkin & Niyogi, JMLR 2006 | https://jmlr.csail.mit.edu/beta/papers/v7/belkin06a.html | 能（只加 loss）| 0 参数，训练期近邻对 | retrain-without 配对 ΔEM±SE；PR 不降 ⇒ 仅当权重衰减 |
| 1 流形正则 | Jacobian 收缩 / 谱 1-Lipschitz | Contractive AE, Rifai†, ICML 2011；Spectral Norm, Miyato, ICLR 2018, arXiv:1802.05957；Parseval, Cissé, ICML 2017, arXiv:1704.08847 | https://dl.acm.org/doi/10.5555/3104482.3104587 ; https://arxiv.org/abs/1802.05957 ; https://arxiv.org/abs/1704.08847 | 能 | 幂迭代 / 1 次额外反传 | ΔLipschitz 与 ΔEM 同报；Lipschitz 降而 PR 骤降 ⇒ 塌缩 |
| 2 增强 | 隐层插值 manifold mixup（训练期 only）| Manifold Mixup, Verma, ICML 2019, arXiv:1806.05236 | https://arxiv.org/abs/1806.05236 | 能（不动推理图）| 每步 2 组前向 | retrain-without ΔEM≥+2pp 且 2 seed 同号；插值轴先过 R29 |
| 2 增强 | 插值/一致性目标（ICT, UDA）| ICT, Verma, arXiv:1903.03825；UDA, Xie†, NeurIPS 2020, arXiv:1904.12848 | https://arxiv.org/abs/1903.03825 ; https://arxiv.org/abs/1904.12848 | 能 | 2 前向（可加无标注）| 换行干预 Δ=0 ⇒ 轴无效；有标注时 ΔEM |
| 3 几何 | Poincaré / Lorentz 双曲表示 | Poincaré Embeddings, Nickel & Kiela, NeurIPS 2017, arXiv:1705.08039；Lorentz, ICML 2018, arXiv:1806.03417 | https://arxiv.org/abs/1705.08039 ; https://arxiv.org/abs/1806.03417 | 部分（exp/log 后仍 `[n,d]`，破坏欧氏残差语义）| 每次 exp/log + 裁剪 | 与残差基型配对 ΔEM；不涨即弃 |
| 3 几何 | 黎曼优化 / 超球归一化 | Riemannian Adaptive Opt., Bécigneul & Ganea, ICLR 2019, arXiv:1810.00760；Hyperspherical VAE, Davidson†, arXiv:1804.00891 | https://arxiv.org/abs/1810.00760 ; https://arxiv.org/abs/1804.00891 | 归一化能；黎曼优化部分 | 优化器改写 / 1 次除法 | 若与 LayerNorm 等价 ⇒ 装饰 |
| 4 维度 | 内在维度约束自编码器 | Zheng, arXiv:2304.07686 | https://arxiv.org/abs/2304.07686 | 能（加 ID 惩罚）| 每 eval 一次 TwoNN | ID 与 PR 同向降且 ΔEM>0 |
| 4 维度 | 有效秩 / 核范数谱惩罚 | WERank†, arXiv:2402.09586；Nuclear Norm Reg., Scarvelis & Solomon, arXiv:2405.14544 | https://arxiv.org/abs/2402.09586 ; https://arxiv.org/abs/2405.14544 | 能 | 幂迭代/SVD，便宜 | d_eff 降且前 20% 因果占比从 50.7% 至少降 5pp |
| 5 等距/解耦/流 | 等距 AE；正交解耦 | Isometric AE†, arXiv:2006.09289；Orthogonal Disentangled Rep., ECCV 2020, arXiv:2003.05707 | https://arxiv.org/abs/2006.09289 ; https://arxiv.org/abs/2003.05707 | 能（loss-only）| 1 次局部扰动/Jacobian | 等距误差 + ΔEM；R29 先证扰动非零 |
| 5 等距/解耦/流 | 流形学习 + 密度流 | Flows for simultaneous manifold learning..., Brehmer†, NeurIPS 2020, arXiv:2003.13913 | https://arxiv.org/abs/2003.13913 | 部分（需额外 `[n,d]→[n,d]` 映射，参数贵）| O(d²)+，最贵 | 只涨似然不涨 EM ⇒ 装饰，弃 |
| 6 神经流形 | 层内 ID 先升后降 + TwoNN | Ansuini, NeurIPS 2019, arXiv:1905.12784；Facco, Sci Rep 2017（arXiv id 未核到）| https://arxiv.org/abs/1905.12784 ; https://doi.org/10.1038/s41598-017-11873-y | 能（纯读数，0 参数）| 0 参数，eval 期 | ID 曲线本身即读数；降而 EM 不涨 ⇒ 未增强 |
| 6 神经流形 | 流形容量的几何 | Chung, Lee & Sompolinsky, PRX 2018（arXiv id 未核到）| https://journals.aps.org/prx/abstract/10.1103/PhysRevX.8.031003 | 能（读数）| 0 参数 | 流形半径/维度与 ΔEM 的相关方向 |
| 7 VSA/HRR | 循环卷积绑定 + 叠加（固定算子）| HRR, Plate, IJCAI 1991 / IEEE TNN 1995 | https://www.ijcai.org/Proceedings/91-1/Papers/006.pdf ; https://dl.acm.org/doi/abs/10.1109/72.377968 | 能（0 参数、形状不变）| FFT O(d log d) | 解绑保真度随叠加数 k 的曲线 + ΔEM |
| 7 VSA/HRR | 容量/编码分析；resonator 分解 | Kanerva, Cognitive Computation 2009；Schlegel†, AIR 2022, arXiv:2001.11797；Mirus, IJCNN 2020, arXiv:2010.00055；Resonator, Frady†, arXiv:2007.03748；Hrrformer, Bricken†, NeurIPS 2021, arXiv:2111.05498 | https://arxiv.org/abs/2001.11797 ; https://arxiv.org/abs/2010.00055 ; https://arxiv.org/abs/2007.03748 ; https://arxiv.org/abs/2111.05498 | 能（定 d 与叠加数；分解 = 迭代 = 共享权重循环，注意 k≥4 崩）| 0 参数 / k 次迭代 | 可靠解绑最大 k；k≥4 崩即复现已有实测 |

（† = 首作者本次未双源核到，见 §5）

## 2. 消融协议（可直接照做）

1. **R16**：每个候选模块做 `retrain-without`，随机种子配对 **≥2** 起；**推理期删/置零不算**，报配对 ΔEM±SE。
2. **门 B**：把模块换成 Identity（`y = x`），**其余不变重训**；报 ΔEM、**Δ前 20% 因果占比**、**ΔPR/ΔID**（每步含 SE）。
3. **R29**：网格干预先报 **Δ≠0 比例**；"互换两行"已知为**无效轴**，若模块只在换行上非零 ⇒ 判**无效轴剔除**。
4. **R28**：生成/评测一律 **`batch=1`**；**PR/ID/谱熵也逐样本算再平均**，避免 batch 统计串味。
5. **★ 流形读数**：`PR = (Σλ)² / Σλ²`（二阶、线性、对整体缩放不敏感）· TwoNN 内在维度（**需局部均匀、样本 <~100 不稳、高 d 易高估**）· 谱熵 `−Σpᵢlog pᵢ`（**与 PR 高度相关，不可当独立证据**）。**三者同向才算"流形真被增强"**。
6. **集中度**：S21 口径的**前 20% 因果占比**（基线 **50.7%**）；**≥2 seed 同向下降**才说"更多单元参与"。
7. **成本**：每卡报**参数增量 + 同构墙钟**（`batch=1`、同形状），并入**五维报告卡**。
8. **统计**：**2 seed 只写"同号/异号"**；**≥3 seed 才给均值±SE**。

## 3. 最该先试的三个（排序 + 判据）

1. **有效秩 / 核范数 / WERank 类谱惩罚** —— **0 参数、直连 S21 集中度**。
   判据：`retrain-without ΔEM ≥ +2pp 且 2 seed 同号`，同时 **PR 下降**、**前 20% 因果占比至少降 5pp**；**PR 不动 ⇒ 未生效，弃**。
2. **隐层插值 manifold mixup（仅训练期）** —— 判据最清晰，**与"带残差必要"同源**（都在改表示空间结构）。
   判据：`ΔEM ≥ +2pp 且 2 seed 同号`；**R29 先证插值轴 Δ≠0 比例 >90%**；**EM 涨而 PR/ID 不动 ⇒ 记"未归因"**。
3. **Jacobian 收缩 / 等距惩罚** —— **最便宜**（1 次额外反传），能把**塌缩与平滑分开**。
   判据：`ΔEM 与 ΔJacobian 同报`；**Jacobian 降而 PR 骤降 ⇒ 塌缩（负）**；**Jacobian 降、PR 平或升、ΔEM>0 ⇒ 支持**。

## 4. 与本项目已知实测的接口

- **卡内硬约束**：`[n,d] → [n,d]`（模因进、模因出）—— 这是筛选上表"能否落进卡内"的依据；
- **已有实测**（只引用结论）：思维卡**带残差必要**（add_3d 2.50% → 37.25%，+34.75pp）· 共享权重循环 **k≥4 崩**（k=8 EM 0.25%）· **加性位置编码有害**（−34.9pp）· **宽度翻倍无收益** · **集中度基线 = 前 20% 单元因果占比 50.7%**（S21）；
- **判据里用到的五维报告卡**：能力 / 因果(门 B) / 集中度 / 成本 / 自检(R28/R29)。

## 5. 未核实项

- **未逐条 HTTP 点开**（`web_fetch` 全域不可用）⇒ 可点性只到**检索级**；链接可能失效。
- **首作者本次未双源核到**（表中 †）：Rifai(CAE)、Xie(UDA)、Davidson(超球 VAE)、WERank、Brehmer(流形-密度流)、
  Isometric Autoencoders、Frady(resonator)、Bricken(Hrrformer)、Schlegel。
- Facco 2017 的 arXiv id（1610.04782）**未核到** ⇒ 只给 Sci Rep DOI；Chung 2018 的 arXiv id（1710.05246）**未核到** ⇒ 只给 PRX DOI。
- **Lorentz 标题两源不一致**（ICML 页 "Learning Continuous Hierarchies..." vs ar5iv "Learning Hierarchical Representations..."）。
- 单来源、**已剔除未采信**：`DREG arXiv:2606.23942`。
- 表中「**能否落进卡内**」与「**代价**」为**推断，非实测**。
