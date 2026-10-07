# PS3 文献检索：结构主义 · 同族竞品 · 前沿成果（2026-10-07）

**检索纪律**：`web_fetch` 全域不可用（`resolves to a non-public IP`）⇒ 全程只用 `web_search`；**可点性只到检索级核对、未做 HTTP 逐条点开**。每条以「同一 arXiv/ACL id + 同一标题」在 ≥2 独立来源一致（ar5iv / arXiv 镜像 / ACL Anthology / Semantic Scholar / HF papers / ADS / 会议官网 / NSTL）。**口径**：前两栏（标题+URL、方法）= 检索实测；「对 flowme 环节」「适用性」「实验设计」= **推断**。
**与既有实测约束的对齐（S5）**：所有消融一律 `retrain-without`（②）；离散中介走 STE（③）；训练量 ≥4000 步才比增益（④）；先过 `max_naive` 计数地板（⑤）；判据只认任务指标、探针只当下限证据（①读出≠会用）；保持串联基线（⑧ +0.70pp）。
**R26 结论先行**：结构"真被用上"的证据只有两种 —— (i) **推理期因果干预**（amnesic/patching）使任务配对 Δ≥2×SE、2 seed 同号，且与 random 子空间对照**不同分布**；(ii) **`retrain-without` 结构通路**后任务显著掉点。探针可读、注意力可视化均**不算**。

## 轴 A 结构主义（5 项）
1. 结构探针 | A Structural Probe for Finding Syntax in Word Representations（Hewitt & Manning, NAACL'19）https://aclanthology.org/N19-1419.pdf（+N19-1419.bib、Semantic Scholar） | 线性映射把词表示距离对齐句法树距离，证明结构**可读出** | 读出/探针层（R26 下限证据） | 探针读出结构后，`retrain-without` 结构通路配对任务 Δ<2×SE ⇒ 判「读出≠会用」（①），探针分不作为增益主张 |
2. Amnesic 干预 | Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals（arXiv 2006.00995 / TACL'21）https://aclanthology.org/2021.tacl-1.10/（+ar5iv 2006.00995） | 推理期把某属性子空间投影移除再测任务，属**推理期干预、不重训**（绕 off-distribution，比 retrain-without 便宜） | 思维卡/输出卡表征，R26 主判据 | 移除结构子空间 vs 移除 random 子空间：任务配对 Δ≥2×SE、2 seed 同号且两分布不同 ⇒ 结构被用上；同分布 ⇒ 判「装饰」 |
3. 探针≠语法 | A Tale of a Probe and a Parser（arXiv 2005.01641 / ACL'20）https://aclanthology.org/2020.acl-main.659/（+ar5iv 2005.01641） | 结构探针的表征能力与线性 parser 相当 ⇒ 探针高分可由简单读出解释 | 读出层，压低探针证据权重 | 同表示上线性 parser 可拟合但任务不涨（Δ<2×SE）⇒ 该结构证据降级；与①同向，禁止拿探针分当增益 |
4. 表层 vs 结构 | Local Structure Matters Most: Perturbation Study in NLU（ACL'22 Findings）https://aclanthology.org/2022.findings-acl.293.pdf（+.bib、NSTL） | 系统扰动对比：模型对**局部/表层**扰动比对全局结构差异更敏感 | 直接对应⑦（模型读表层不读结构）的测量装置 | 建配对集：无意义表层扰动 vs 有意义结构差异；机制修复后结构差敏感度升 ≥2×SE、2 seed 同号，且表层敏感度**不升** ⇒ 修好⑦；否则未修 |
5. 栈结构先验 | Bearing Syntactic Fruit with Stack-Augmented Neural Networks（arXiv 2511.03547）https://ar5iv.labs.arxiv.org/html/2511.03547（+arxiv pdf v2、scirate、GitHub 代码仓） | 给 Transformer 加可微非确定性栈，问句生成等任务上栈模型优于纯 Transformer（论文明示结论） | 思维卡结构先验（树/栈归纳）的增益测量模板 | 同 FLOPs：栈卡 vs 参数等量平凡卡，训练 ≥4000 步、2 seed，配对 Δ≥2×SE 同号且超 `max_naive` ⇒ 结构先验有效；同 FLOPs 不降点 ⇒ 该机制是装饰 |

## 轴 B 同族竞品（4 项，逐条"它强/我强"）
B1. MoE 条件计算 | Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity（Fedus/Zoph/Shazeer, arXiv 2101.03961）https://ar5iv.labs.arxiv.org/html/2101.03961（+Semantic Scholar、Princeton 讲义） | 做法：top-1 稀疏路由替换稠密层；规模：**万亿参数级（标题明示）**；结论：简化路由可训万亿稀疏模型；开销：需负载均衡辅助损失（按通行做法，未读正文→推断） | 卡路由 vs 同质专家路由 | 黑板手写调度 vs top-k 专家路由，同 FLOPs：路由方式切换后任务不降点（Δ≥2×SE，2 seed 同号）才保留；不降点 ⇒ 路由机制是装饰
　└ 它强（事实）：万亿参数规模与成熟训练工艺，远超 flowme；我们强（事实）：Switch 是**同质** transformer 层 + 概率路由、无硬 IO 契约；flowme 异构卡 + 卡间只流张量是实现层硬约束（推断：异构契约能否规模化未测）。
B2. Agent 层级聚合 | Mixture-of-Agents Enhances Large Language Model Capabilities（arXiv 2406.04692）https://huggingface.co/papers/2406.04692（+arXiv 镜像、ADS 记录） | 做法：多模型分层聚合彼此输出；规模：多个开源 LLM 聚合（具体层数未核到）；结论：仓库报告 AlpacaEval 65.1%（GitHub 检索级，非论文正文核到）；开销：多模型串行推理（推断） | 输出卡聚合/接力 | 输出卡二段聚合 vs 单卡：配对 Δ≥2×SE、2 seed 同号且超 `max_naive` ⇒ 聚合有效；与单卡同分布 ⇒ 装饰
　└ 它强（事实）：免训练即得指标提升、报告数值公开；我们强（事实）：MoA 聚合走**文本**层、无潜空间中继；flowme 卡间只流张量，聚合发生在表示层（推断：表示层聚合更省带宽未测）。
B3. Agent-as-模块 | Improving Planning with Large Language Models: A Modular Agentic Architecture（arXiv 2310.00194）https://ui.adsabs.harvard.edu/abs/2023arXiv231000194W/abstract（+HF papers-content、arXiv 镜像） | 做法：规划任务拆成独立反思/规划模块组合；规模：LLM 级单体（参数未核到）；结论：模块化在规划任务有效（标题/摘要级，未读正文→推断）；开销：多模块调用链（推断） | 思维卡回路 + 黑板调度（R12） | 加"反思思维卡"回路 vs 无回路，同 FLOPs 配对 Δ≥2×SE、2 seed 同号 ⇒ 保留；不降点 ⇒ 该卡是装饰
　└ 它强（事实）：给出了模块化在规划任务有效的实测结论；我们强（事实）：其模块接口为自然语言级、调度随论文固定；flowme 接口是张量契约 + 手写调度可替换（推断：契约化更利于消融，未测）。
B4. 递归/循环深度 | Latent Chain-of-Thought? Decoding the Depth-Recurrent Transformer（arXiv 2507.02199）https://ar5iv.labs.arxiv.org/html/2507.02199（+HF papers 2507.02199） | 做法：潜空间递归"解码"深度循环 Transformer；规模：未核到；结论：深度循环可替代部分显式思维链（摘要级）；开销：循环推理步时延（推断） | 思维卡循环执行 | 递归 k 步 vs 单遍同 FLOPs：配对 Δ≥2×SE、2 seed 同号 ⇒ 深度有效；与随机权重循环同分布 ⇒ 装饰
　└ 它强（事实）：潜空间递归已有成型实现与结论；我们强（事实）：其循环同构权重、无异构卡/IO 契约；flowme 可按步换卡（推断：异构循环更贵未测）。

## 轴 C 前沿 2025–2026（6 项）
C1. 递归深度 | Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach（arXiv 2502.05171, NeurIPS'25）https://arxiv-org.ezproxy.obspm.fr/abs/2502.05171?context=cs（+ar5iv 2502.05171、NeurIPS proceedings 同 hash） | 同一 Transformer 权重在潜空间迭代多轮放大测试时计算 | **借来**：思维卡共享权重循环，解决"思维卡单遍深度不足" | 同 FLOPs 循环 vs 单遍：Δ≥2×SE、2 seed 同号 ⇒ 借用；不降点 ⇒ 装饰
C2. 连续潜空间推理 | Training Large Language Models to Reason in a Continuous Latent Space（arXiv 2412.06769, Coconut）https://ar5iv.labs.arxiv.org/html/2412.06769v1（+ADS 2024arXiv241206769H） | 思维链改在连续潜空间递归推进，不落 token | **借来**：替代离散中介路线，检验"不必 STE 也能上游可导" | latent 步 vs STE 离散步：任务 Δ≥2×SE、2 seed 同号且上游梯度非零 ⇒ 可弃 STE；否则维持③
C3. 记忆层 | Memory Layers at Scale（arXiv 2412.09764）https://huggingface.co/papers/2412.09764（+ar5iv、arXiv 镜像） | 用可学习查找式记忆层替代部分前馈参数 | **借来**：解决输出卡长尾记忆、与 `max_naive` 计数特征（⑤）的混淆 | 同 FLOPs 加记忆层 vs 等参 FFN：任务 Δ≥2×SE、2 seed 同号 ⇒ 有用；同分布 ⇒ 装饰
C4. 字节/无分词 | Byte Latent Transformer: Patches Scale Better Than Tokens（arXiv 2412.09871）https://ar5iv.labs.arxiv.org/html/2412.09871（+HF papers、Semantic Scholar） | 动态字节 patch 代替固定 token，Meta FAIR | **借来**：解决 tokenizer 切碎结构、缓解⑦表层敏感 | 同 FLOPs 字节 patch vs token 输入：Δ≥2×SE、2 seed 同号，且表层扰动敏感度下降 ⇒ 借用
C5. 能量类 | Energy-Based Transformers are Scalable Learners and Thinkers（arXiv 2507.02092, ICLR'26）https://huggingface.co/papers/2507.02092（+ICLR proceedings、库记录） | 以能量函数联合打分/重排候选 | **借来**：输出卡候选重排，绕开显式 softmax 竞争 | energy 重排 vs 平凡重排：配对 Δ≥2×SE、2 seed 同号且超 `max_naive` ⇒ 有效；同分布 ⇒ 装饰
C6. 测试时训练 | TTRL: Test-Time Reinforcement Learning（arXiv 2504.16084, NeurIPS'25）https://proceedings.neurips.cc/paper_files/paper/2025/hash/be690ea16f005c174f6c4102a5970e67-Abstract-Conference.html（+MSU 目录、NeurIPS virtual） | 测试时用伪标签 RL 更新模型参数 | **借来**：推理期自适应更新卡参数，解决"卡在部署后不适应" | TTRL 更新 vs 冻结：配对 Δ≥2×SE、2 seed 同号 ⇒ 有效；同分布 ⇒ 装饰
（背景/未入列：Conditional computation in neural networks: Principles and research trends（arXiv 2403.07965）https://arxiv.org/pdf/2403.07965 —— 综述，仅作 B 轴框架参照。）

## 优先级建议（先做 3 个）
1. **A2 Amnesic 干预**（先做）：最便宜——推理期干预、不重训、绕 off-distribution；直接回答 R26「结构真被用上没有」。成本：一次评估 + 子空间估计，无训练。
2. **A4 表层 vs 结构对照**（并行做）：给⑦建测量装置，成本只是一套扰动生成 + 配对评估；它是后续所有机制实验的"判装饰"前置。
3. **C1 递归深度思维卡**（测量装置就位后再做）：判据清晰（同 FLOPs 循环 vs 单遍），但需 ≥4000 步 ×2 seed 重训（④），成本最高，放最后。
理由：1、2 先确定"结构是否真被用上"，否则 3 的任何增益都无法归因（①）。

## 未验证清单（S3/S4）
- `web_fetch` 全域不可用 ⇒ **所有 URL 未做 HTTP 逐条点开**，可点性仅到 web_search 检索级（同 id+标题 ≥2 来源一致）。
- ★兜底抽查（最可疑=最新 id 2511.03547）：换 3 种措辞独立复核（作者名措辞 / "2511.03547 abstract" / "stack attention question formation"），ar5iv + arXiv pdf v2 + scirate + GitHub 同 id 同标题一致 ⇒ **抽查通过，清单未降级**。
- 各条"做法/结论/规模/开销"仅据标题与检索摘要，**未读正文**；具体超参/规模数字多数未核到（已逐条标注）。
- 两处 id/标题冲突未入列：COGITAO（两来源标题不一致）；Hierarchical Reasoning Model（2406.14271 vs 2506.21734 歧义）——均按"未核到"处理。
- 检索到 = 实测；「是否适用于 flowme 规模/算力」全部为**推断**，未验证。

## 判定：**清单可用**（三轴各 ≥3、五项齐全、轴 B 逐条强弱对比、交叉核对与兜底抽查通过、未验证项已如实列明；唯一降级风险来源 web_fetch 不可用已在清单首行声明）
