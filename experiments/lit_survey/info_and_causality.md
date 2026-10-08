# 表示中的「信息量」与「因果可用性」：方法族与因果证据（调研）

> 一句话：四族方法没有任何一族能同时给出「能装多少」与「下游用多少」；
> "含更多信息"只在信息论族不误导，且必须与因果数分开报。

- 口径：**web_fetch 全域不可用** ⇒ 全程仅检索；**可点性只到检索级交叉核对（同一 arXiv id + 标题 ≥2 来源一致）、未逐条点开**（见未核实项）。
- 本调研只落盘结论与文献表，不含任何新实测；对照数字全部引自项目已有实测（P-3 / T1 / S17 / S20）。

## 1. 度量"信息量"的四族

| 族 | 方法 | 代表工作 | 链接 | 度量什么 | 是否混淆可读性 vs 可用性 |
|---|---|---|---|---|---|
| 探针族 | linear/MLP probe + control task + selectivity | Hewitt & Liang 2019 EMNLP | [arXiv:1909.03368](https://arxiv.org/abs/1909.03368) | probe 能解码标签到何种程度；selectivity = 任务准确率 − 控制任务准确率 | 只惩罚 probe 容量、不测因果；高准确率常只说明"线性可读" |
| 探针族 | MDL probing | Voita & Titov 2020 EMNLP | [arXiv:2003.12298](https://arxiv.org/abs/2003.12298) | 复杂度—准确率曲线取最优，量表示替 probe 省下多少描述长度 | 仍是 probe 可达性，无因果项 |
| 探针族 | amnesic probing | Elazar 等 2021 TACL | [arXiv:2006.00995](https://arxiv.org/abs/2006.00995) | 移除属性方向后看下游行为变化 | 最接近"可用性"，但移除是后处理（off-distribution），≠ retrain-without |
| 探针族 | structural probe | Hewitt & Manning 2019 NAACL N19-1419（**注：非 Tenney**；Tenney 2019 是分层 edge probing [arXiv:1905.05950](https://arxiv.org/abs/1905.05950)） | NAACL N19-1419 | 线性变换后距离/范数能恢复句法树 | 至多证明"线性可及" |
| 探针族 | V-usable / conditional probing | Ethayarajh 等 2022 ICML [arXiv:2110.08420](https://arxiv.org/abs/2110.08420)；Hewitt 等 2021 EMNLP [arXiv:2109.09234](https://arxiv.org/abs/2109.09234) | [arXiv:2110.08420](https://arxiv.org/abs/2110.08420) + [arXiv:2109.09234](https://arxiv.org/abs/2109.09234) | 用 baseline 差值定义"超出基线仍可解码" | 仍是"可解码"；baseline 选择敏感 |
| 信息论族 | MI 估计（MINE 类） | Belghazi 等 2018 ICML oral（**arXiv id 未核实**）；局限见 McAllester & Stratos 2019 [arXiv:1811.04251](https://arxiv.org/abs/1811.04251) | [arXiv:1811.04251](https://arxiv.org/abs/1811.04251) | 估计 I(X;Z) | 高维方差大、可被容量灌水；I 大 ≠ 任务有用 |
| 信息论族 | InfoNCE | van den Oord 等 2018 | [arXiv:1807.03748](https://arxiv.org/abs/1807.03748) | I 的变分下界 | 随 batch/负采样漂移；度量可判别性 |
| 信息论族 | I(M;Y) 上下界 + 复杂度—准确率 | Pimentel 等 2020 ACL | ACL 2020 | V-IB 给上界、MDL 给下界，分离"信息"与"probe 复杂度" | 上界依赖 probe 家族 |
| 信息论族 | 最小充分统计量/压缩 | Strouse & Schwab 2017 [arXiv:1604.00268](https://arxiv.org/abs/1604.00268)；2019 [arXiv:1905.07822](https://arxiv.org/abs/1905.07822) | [arXiv:1604.00268](https://arxiv.org/abs/1604.00268) + [arXiv:1905.07822](https://arxiv.org/abs/1905.07822) | 保 Y 下最小化对 X 的描述长度 | 需生成分布假设 |
| 几何/谱族 | 有效维度/participation ratio | Gao 等 2019 ICLR [arXiv:1907.12009](https://arxiv.org/abs/1907.12009)；Roy & Vetterli 2007 EUSIPCO | [arXiv:1907.12009](https://arxiv.org/abs/1907.12009) | 奇异值谱集中度 | 纯几何；高有效维度可能只是各向同性 ⇒ **最易被误当"信息多=有用"** |
| 几何/谱族 | SVD 谱熵/RankMe | Garrido 等 2023 ICML | ICML 2023 | 谱熵（有效秩）预测下游质量 | 仅经验相关、非因果 |
| 几何/谱族 | intrinsic dimension（TwoNN） | Facco 等 2017 Sci Rep 7:12140；Ansuini 等 2019 NeurIPS | Sci Rep 7:12140；NeurIPS 2019 | 流形局部自由度 | 估计器有偏 |
| 几何/谱族 | CKA / RSA | Kornblith 等 2019 ICML [arXiv:1905.00414](https://arxiv.org/abs/1905.00414)；Kriegeskorte 等 2008 | [arXiv:1905.00414](https://arxiv.org/abs/1905.00414) | 表示结构相似度 | 相似 ≠ 等效 |
| 几何/谱族 | 线性可及性 | Park 等 2023 | [arXiv:2311.03658](https://arxiv.org/abs/2311.03658) | 概念是否由线性方向表示 | 正是"可读性"本身 |
| 容量/叠加族 | superposition | Elhage 等 2022 | [arXiv:2209.10652](https://arxiv.org/abs/2209.10652) | d 维能塞多少稀疏特征 | "能编码" ≠ "被用到" |
| 容量/叠加族 | SAE | Bricken 等 2023 transformer-circuits；Cunningham 等 2023 [arXiv:2309.08600](https://arxiv.org/abs/2309.08600)；Templeton 等 2024 | [arXiv:2309.08600](https://arxiv.org/abs/2309.08600) | 稀疏字典分解激活 | 重建好 ≠ 因果重要 |
| 容量/叠加族 | VSA/HRP 绑定容量 | Frady 等 2021 IEEE TNNLS [arXiv:2009.06734](https://arxiv.org/abs/2009.06734)（HRP 原始 Plate 1991/1995，**容量标度未核实**） | [arXiv:2009.06734](https://arxiv.org/abs/2009.06734) | 绑定多少次后串扰淹没信号 | "能存"上限，与效用无关 |
| 容量/叠加族 | 表示瓶颈 | Tishby & Zaslavsky 2015 [arXiv:1503.02406](https://arxiv.org/abs/1503.02406)；Saxe 等 2018 ICLR（**arXiv id 未核实**） | [arXiv:1503.02406](https://arxiv.org/abs/1503.02406) | 压缩 I(X;T) 与保真 I(T;Y) 折中 | 实测难复现 |

## 2. 证明"信息被用于输出"的方法

| # | 方法 | 代表工作 | 链接 | 证明什么 | 如何避免混杂 | 局限 |
|---|---|---|---|---|---|---|
| 1 | activation patching / causal tracing | Meng 等 2022 | [arXiv:2202.05262](https://arxiv.org/abs/2202.05262) | clean/corrupt/restore 配对反事实 | 配对设计避样本难度混杂 | noising 基线低会失真 |
| 2 | noising vs denoising 不可比 | Zhang & Nanda 2024 ICLR | [arXiv:2309.16042](https://arxiv.org/abs/2309.16042) | 两者是两个不同问题、数值不可比 | 明确区分方向 | 混用会造伪结论 |
| 3 | path patching | Goldowsky-Dill 等 2023 [arXiv:2304.05969](https://arxiv.org/abs/2304.05969)；操作规范 Heimersheim & Nanda 2024 [arXiv:2404.15255](https://arxiv.org/abs/2404.15255) | [arXiv:2304.05969](https://arxiv.org/abs/2304.05969) + [arXiv:2404.15255](https://arxiv.org/abs/2404.15255) | 收窄干预集避冗余补偿 | 只干预指定路径 | 路径选择依赖先验 |
| 4 | causal abstraction / DAS | Geiger 等 JMLR 2025 [arXiv:2301.04709](https://arxiv.org/abs/2301.04709)；DAS [arXiv:2112.00826](https://arxiv.org/abs/2112.00826) 与 [arXiv:2303.02536](https://arxiv.org/abs/2303.02536) | [arXiv:2301.04709](https://arxiv.org/abs/2301.04709) + [arXiv:2112.00826](https://arxiv.org/abs/2112.00826) + [arXiv:2303.02536](https://arxiv.org/abs/2303.02536) | 形式化"高维表示忠实抽象低维因果变量"，只有 II 下反事实一致才算对应 | 对齐梯度就搜在同一干预轴 | 需假设因果模型 |
| 5 | amnesic probing | Elazar 等 2021 TACL [arXiv:2006.00995](https://arxiv.org/abs/2006.00995)；改进 Mean Projection/LEACE ACL 2025 Findings [arXiv:2506.11673](https://arxiv.org/abs/2506.11673) | [arXiv:2006.00995](https://arxiv.org/abs/2006.00995) + [arXiv:2506.11673](https://arxiv.org/abs/2506.11673) | 同轴删信息再重训探针测任务掉分 | 同轴 + 重训探针 | off-distribution |
| 6 | causal scrubbing / attribution patching | Chan 等 2022 Alignment Forum；Syed/Rager/Conmy 2024 BlackboxNLP | Alignment Forum 2022；BlackboxNLP 2024 | 与红鲱鱼备择假设对比；梯度近似降本 | 备择假设显式对比 | 近似需与真 patching 对齐验证 |

## 3. 与本项目实测的对照（**最重要**）

- P-3 已确证：rho(局部可读性, |局部因果效应|) = 0.316（n=255, p=0.0000）；
  同 ckpt 内：可读性 98.5% → 因果 0.013pp；可读性 69.2% → 因果 65.20pp（因果差 ~5000×）
- T1：I(M;Y|X)=0（中介不产生信息 ⇒ 价值在"计算"不在"保真"）
- S17 实测陷阱：破坏性干预下 |Δ| ≈ 基线成绩（差 ≤0.01pp）⇒ 那个 rho 测的是"模型好坏 vs 探针准度"；
  文献点名：Canby & Davies arXiv:2408.15510 · Mueller arXiv:2407.04690 · Zhang & Nanda arXiv:2309.16042
- S20 实测：互换两行模因 ⇒ (K,V) 成对互换 ⇒ 下游 ΣαV 不变（|Δlogits| = 2–6e-6）⇒ 该干预轴无效
  （R29 要求：非零比例过低 ⇒ 该轴作废并剔除）

## 4. 落地的实验协议（可直接照做）

- 同一干预轴 · 同一 clean 基线 · **非破坏性单点** · **成对记 (Δ信息, Δ损失)**
- 方向用 **denoising**（不是 noising）；**R29 非零性冒烟**先做；加**轴内置换检验**与**阳性对照**
- 要下"结构必需"结论 ⇒ **必须另做 retrain-without**（R16）
  （文献支持：Liu 2018 arXiv:1810.05270《Rethinking the Value of Network Pruning》；LoFiT NeurIPS 2024）

## 5. 未核实项（**必须原样保留，不许补全**）

- web_fetch 全域不可用 ⇒ 全部为检索级交叉核对（同一 arXiv id + 标题 ≥2 来源一致），**未逐条点开**
- MINE（Belghazi 2018）arXiv id 未核实；Saxe 2018 arXiv id 未核实；HRP 容量标度（Plate 1991/1995）未核实
- 《Refit the Probe》arXiv id 与另一标题冲突 ⇒ **未核实**
- LoFiT 未核到 arXiv id ⇒ 只引 NeurIPS 2024
- CausalGym id 2402.12560 vs 2402.12659 两处不一致 ⇒ 取 ar5iv 的 2402.12560
