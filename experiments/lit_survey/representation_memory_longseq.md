# PS2 文献检索：集合/槽表示 · 经验注入与记忆 · 长序列与目标定位

- 口径：**web_fetch 全域不可用**（报 non-public IP）⇒ 全程仅 `web_search`；**可点性只到检索级核对、未做 HTTP 逐条点开**（见未验证清单）。
- 交叉核对：同一篇须「同 arXiv id + 同标题」≥2 来源（HF papers / ar5iv·arxiv 镜像 / Semantic Scholar / PMLR / ADS）才收录；★兜底抽查见文末。
- 分栏：**实测**＝检索到的标题·id·来源一致；**推断**＝标 `[推断]`（是否适用我方规模未测）。
- 与 7 条既有实测约束的总落法：消融＝retrain-without 或推理期 patch；离散中介走 STE（禁 argmax）；≥4000 步才判增益；先比 max_naive；探针读出≠被用。

## 轴 A：变长集合 / 槽表示（2 条）

- **A1 位置感知注意力读出** | Set Transformer: A Framework for Attention-based Permutation-Invariant Neural Networks — https://huggingface.co/papers/1810.00825 + https://ar5iv.labs.arxiv.org/html/1810.00825 | 用注意力池化（ISAB/PMA）替代 mean/max 池化：变长+mask 可读出，**key 显式挂位置**才破置换不变 | 模因张量 [n,d]+mask → 思维卡的读出口 | 可判实验：两变体各自重训（=retrain-without 口径），2 seed，构造"交换两槽则答案翻转"任务；判据＝读出后任务配对 Δ≥2×SE 且 2 seed 同号；若与旧池化同分布 ⇒ 该读出是装饰。注意：论文原生置换不变，不加位置＝没改。
- **A2 语序丢失的零训练判定** | On the Limitations of Representing Functions on Sets（Wagstaff 等） — https://ar5iv.labs.arxiv.org/html/1901.09006 + http://proceedings.mlr.press/v97/wagstaff19a.html | 证明有限精度下某些集合函数不可表示 ⇒ 池化丢序是结构性的，调参救不回 | 实测⑥的解释项；决定读出口径动不动 | 可判实验：交换模因顺序、比读出向量与输出分布（零训练）；判据＝配对差 <2×SE（同分布）⇒ 表示不含序，先修读出；有序 ⇒ ⑥由别处引起，A1 降级。

## 轴 B：经验注入与记忆（5 条）

- **B1 快速权重通道** | Linear Transformers Are Secretly Fast Weight Programmers — https://huggingface.co/papers/2102.11174 + https://ar5iv.labs.arxiv.org/html/2102.11174 | 注意力写入的矩阵即"住在参数里的经验"（快速权重），写入与复用同一通道 | 模因张量写入参数的通道（T1 的具体机制） | 可判实验：retrain-without 该通路 vs 保留，同 FLOPs、≥4000 步、2 seed；判据＝配对 Δ≥2×SE 且同号；**同 FLOPs 不降点 ⇒ 该机制是装饰**；写入侧离散选择必须 STE。
- **B2 冻结记忆检索** | Memorizing Transformers — https://huggingface.co/papers/2203.08913 + https://arxiv.org/abs/2203.08913 | 外挂 kNN 记忆按需读回注意力，经验不进权重 | 思维卡的外部记忆读出 | 可判实验：推理期打乱/置空检索邻域（外挂记忆非权重，不构成删权重式消融，避开 off-distribution）；判据＝EM 配对 Δ≥2×SE、2 seed 同号；不降 ⇒ 记忆读出是装饰（"读出≠会用"的直接检验）。
- **B3 上下文经验注入** | ExpeL: LLM Agents Are Experiential Learners — https://arxiv.org/abs/2308.10144 + https://ar5iv.labs.arxiv.org/html/2308.10144 | 把历史成败蒸馏成文字经验注入 prompt，纯上下文、不改参数 | 输入卡/思维卡前挂经验文本 | 可判实验：真实经验 vs 同长度 shuffle 经验 vs 随机文本（平凡地板）；判据＝**与 random 对照同分布 ⇒ 判为装饰**；真实胜出须 Δ≥2×SE、2 seed 同号。
- **B4 因果定位（廉价替代重训）** | Locating and Editing Factual Associations in GPT — https://huggingface.co/papers/2202.05262 + https://ar5iv.labs.arxiv.org/html/2202.05262 | causal tracing：激活/路径反事实 patch，定位某知识真被哪层用到 | "经验是否被用"的第一道口径，先于花钱 retrain-without | 可判实验：patch 关键槽 vs 同位置随机 patch；判据＝配对 Δ≥2×SE、2 seed 同号 ⇒ 被用；patch 不降但探针读得出 ⇒ 读出≠会用，判装饰。
- **B5 电路级 patch 清单** | Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small — https://huggingface.co/papers/2211.00593 + https://ar5iv.labs.arxiv.org/html/2211.00593 | 对单头/单神经元做消融与替换的功能干预清单，复现功能电路 | 把"被用"判定落成可重复的 patch 清单 | 可判实验：逐元素 patch 读出口；判据＝patch 后任务降 <2×SE 而 probe 高 ⇒ 该读出是装饰；降且 2 seed 同号 ⇒ 计入"被用"。（先验 id 2210.07516 与检索不符，以 2211.00593 为准）

## 轴 C：长序列与目标定位（3 条）

- **C1 答案定位与保尾截断** | Lost in the Middle: How Language Models Use Long Contexts — https://huggingface.co/papers/2307.03172 + https://arxiv.org/abs/2307.03172 | 位置决定可用性：U 形曲线、尾部近期信息最可用 ⇒ "保尾"方向有据 | 截断策略（98.2% 被截 → 答案锚定窗/保尾混合） | 可判实验：同一 checkpoint 三臂 头截/尾截/随机截，各 2 seed；判据＝尾截−随机截 配对 Δ≥2×SE 且同号；不显著 ⇒ "位置红利"是装饰。
- **C2 分桶评测口径（先修 EM）** | RULER: What's the Real Context Size of Your Long-Context Language Models? — https://huggingface.co/papers/2404.06654 + https://ui.adsabs.harvard.edu/abs/2024arXiv240406654H/abstract | 合成 needle 按目标深度分桶测有效上下文长度 | 评测口径：全集 EM 换成"未截断子集 + 按答案位置分桶 EM" | 可判实验（零训练）：分桶 EM 若随深度陡降且未截断子集 EM>0 ⇒ 现全集 EM 是截断伪影；桶间差 <2×SE ⇒ 截断不是主因，改查训练侧；同时报 max_naive 地板。
- **C3 位置课程（PoSE）** | PoSE: Efficient Context Window Extension of LLMs via Positional Skip-wise Training — https://huggingface.co/papers/2309.10400 + https://arxiv.org/abs/2309.10400 | 训练时随机偏移/跳位采样位置分段，短序列算力覆盖长位置 | 课程长度：让"答案在尾"的样本在训练中真出现 | 可判实验：位置课程 vs 固定位置，同步数同 FLOPs、2 seed；判据＝配对 Δ≥2×SE 且同号；不涨 ⇒ 课程是装饰；≥4000 步才下结论。

## 优先级建议（先做 3 个）

1. **C2 分桶口径**（先做）：只改评测、零训练，成本最低；不做它，C1/C3 的信号都被 98.2% 截断结构性压死（先报未截断子集 + max_naive 地板）。
2. **B4/B5 推理期 patch**：比 retrain-without 便宜且避开 off-distribution 污染；成本＝推理期干预，先判"经验真被用"再花钱重训。`[推断]`
3. **C1 保尾混合截断**（随后）：改数据管线 + 一次重训，≥4000 步、2 seed 才判；A1/A2 属表示级重训最贵，排最后，且先过 A2 的零训练交换测试。`[推断]`

## 未验证清单

- **web_fetch 全域不可用**（报 non-public IP）⇒ 全程仅 `web_search`；**可点性只到检索级核对、未做 HTTP 逐条点开**，未读任何摘要/正文。
- 交叉核对＝搜索结果中 URL 与标题共现，非页面内容逐字比对；仅单来源给出 id+标题的候选（"Fast Weight Programmers" 2006.11668、RETRO 2112.04426、Shaw 相对位置 1803.02155 的 ACL 条目）**未收录**。
- ★兜底抽查：最可疑＝PoSE id（先验 2310.16188 与检索冲突）→ 换措辞（positional skip-wise + 作者）独立复核 ⇒ 2309.10400 在 HF papers、arxiv 镜像与模型卡一致 ⇒ **通过**；同法自查发现 IOI 先验 id 2210.07516 错误，已改 2211.00593（HF+ar5iv 一致）。故清单**未降级**。
- `[推断]` 各法在我方小模型/千步级训练量下的收益未实测；轴 A 顺序敏感集合学习的近期工作只核到 2 条，第三候选未完成核对。

## 判定

**清单可用** —— Q1 轴 A2/B5/C3（≥2）✓；Q2 每条五项齐全 ✓；Q3 ≥2 来源交叉核对 + ★抽查通过 ✓；Q4 未验证清单含 web_fetch 不可用 ✓；Q5 七条既有约束逐条内嵌、无冲突 ✓。附带代价：可点性仅到检索级。
