# PS1 检索报告：潜向量序列做生成中介 + 模块因果贡献的可判证明（flowme 文本生成主线）

**检索口径**：只用 `web_search`；本环境 `web_fetch` 全域不可用（报 `resolves to a non-public IP`）⇒ **全部 URL 未做 HTTP 逐条点开，可点性只到检索级核对**。每条均用「同一 arXiv/ACL id + 同一标题」≥2 来源一致才收录；标 **（推断）** 者为"是否适用 flowme 规模"的判断，非检索实测。

## 轴 A：潜向量中介做生成（5 条）

| # | 方案 | 论文 + URL | 方法一句话 | flowme 环节 | 可判实验（判据） |
|---|---|---|---|---|---|
| A1 | 连续思维链（latent reasoning） | Training Large Language Models to Reason in a Continuous Latent Space (Coconut) https://arxiv.org/abs/2412.06769 | CoT 不再输出 token，改把连续 hidden state 循环喂回模型，在潜空间多步推理 | 思维卡（模因→模因）连续化，替代 argmax | 同一黑板 3 配置×2 seed×同步数（≥4000 步，约束④）：连续 / 离散+STE / argmax 负控；判据：连续对离散+STE 配对 Δ≥2×SE 且 2 seed 同号才算有效；若与 argmax 负控同分布 ⇒ 连续化是装饰 |
| A2 | 文本 latent diffusion | Diffusion-LM Improves Controllable Text Generation https://arxiv.org/abs/2205.14217 | 在连续 embedding 空间做扩散去噪再解码回 token，生成全程留在连续空间 | 输出卡：模因序列→目标 token 的连续中介 | 训后置换/换批重采样潜向量序列：若任务分与 intact 差 <2×SE ⇒ 输出卡没真用潜序列；intact 对正常基线 Δ≥2×SE、2 seed 同号且先过 max_naive 地板（约束⑤）才算数 |
| A3 | 高效离散瓶颈（FSQ 码本） | Finite Scalar Quantization: VQ-VAE Made Simple https://arxiv.org/abs/2309.15505 | 逐维缩放+取整替代 EMA 码本，免 codebook collapse、免额外库参数 | 卡间离散码本中介（梯度仍走 STE，不许 argmax，约束③） | 参数对齐后（约束⑥）FSQ vs VQ-EMA 各 2 seed；判据：活跃码字占比≥50% 且 retrain-without-该瓶颈 Δ≥2×SE 同号；若与 argmax 负控同分布 ⇒ 瓶颈是装饰 |
| A4 | 连续潜变量文本生成（层式 VAE） | Fuse It More Deeply! A Variational Transformer with Layer-Wise Latent Variable Inference for Text Generation https://aclanthology.org/2022.naacl-main.51/ | 逐层推断潜变量 z_l 并注入解码，缓解 posterior collapse、保持序列连续性 | 输入卡变长模因序列 [n,d] 上挂连续潜变量 | 两道判据：① 重采样/置换 z 后输出分布不变且 KL≈0 ⇒ 潜变量是装饰；② 加 z 后任务 Δ≥2×SE、2 seed 同号且先过 max_naive 地板 |
| A5 | 离散码本 + 会用性判据 | Codebook Features: Sparse and Discrete Interpretability for Neural Networks https://arxiv.org/abs/2310.17230 | 训练中学离散码字，码字可读出语义，压缩与解释一体 | 白板上卡间流动的离散码字 | 严格按约束①：探针读出高**不算数**；判据 = retrain-without-该码本层 Δ≥2×SE、2 seed 同号才判"被使用"；探针高而 Δ≈0 ⇒ 即已知的读出≠会用 |

## 轴 B：模块因果贡献的可判证明（6 条，先便宜后贵）

| # | 方案 | 论文 + URL | 方法一句话 | flowme 环节 | 可判实验（判据） |
|---|---|---|---|---|---|
| B1 | path patching（推理期因果干预） | Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small https://arxiv.org/abs/2211.00593 | 用配对样本替换某卡输出激活，看任务分是否塌，定位因果路径 | 白板上任意两卡之间的张量流 | 判据：卡 i 被 patch 的 Δ 与 random-patch（同形状随机张量）基线同分布 ⇒ 卡 i 判装饰；显著超 random-patch 95 分位 ⇒ 进候选，再由 retrain-without 终审（约束②）。零训练，最便宜 |
| B2 | causal tracing（逐点恢复干预） | Locating and Editing Factual Associations in GPT https://arxiv.org/abs/2202.05262 | 先整体破坏、再逐点恢复中间激活，画每层每卡的因果贡献图 | 黑板上每张卡输出与各层中介 | 判据：trace 分与随机替换基线同分布的卡 ⇒ 装饰；trace top-k 与 retrain-without Δ 的 Spearman ρ≥0.6 且 2 seed 同号才承认归因，否则只当筛查 |
| B3 | attribution patching（梯度廉价筛查） | Attribution Patching Outperforms Automated Circuit Discovery https://aclanthology.org/2024.blackboxnlp-1.25/ | 用梯度近似替代海量 patching（约千倍提速）做粗筛 | 每轮组合实验的第一道廉价闸门 | 判据：attribution top-k 与真实 patching top-k 的 ρ≥0.7 且 2 seed 同号 ⇒ 才允许用廉价版筛；否则必须逐卡真 patching |
| B4 | patching 统一口径（防假阳性） | Towards Best Practices of Activation Patching in Language Models: Metrics and Methods https://arxiv.org/abs/2309.16042 | 指标与替换方法的选择会翻转结论，须固定口径并带对照 | 轴 B 全部实验的公共口径 | 判据：同一实验至少两种指标结论同向、且都对 random-patch 对照显著（Δ≥2×SE、2 seed 同号）才下结论；方向不一致 ⇒ 该结论记"未定"，不进结论栏 |
| B5 | Shapley 类归因（按卡分贡献） | Who Does What in Deep Learning? Multidimensional Game-Theoretic Attribution of Function of Neural Units https://arxiv.org/abs/2506.19732 | 对神经单元/模块算多维 Shapley 值，量化每个单元的边际贡献 | 黑板上"哪张卡值得留"的取舍 | 判据：Shapley 符号与 retrain-without Δ 同号率≥80%（或 ρ≥0.6）才信归因；Shapley≈0 且 retrain-without Δ<2×SE ⇒ 双证据判装饰 |
| B6 | 路由/模块是否被使用 | Are Sixteen Heads Really Better than One? https://arxiv.org/abs/1905.10650 | 推理期逐头/逐单元 ablate，测分数变化与权重冗余，识别真正被用的头 | 黑板调度是否真的走到每张卡 | 在线 ablation **仅作筛查**（\|Δ\|<2×SE ⇒ 疑似未用，再按 retrain-without 复核）；最终结论只认 retrain-without Δ≥2×SE、2 seed 同号（约束②：训练后删组件是 off-distribution，不作证据） |

## 轴 C：训练量敏感 ⇒ 省算力（4 条）

| # | 方案 | 论文 + URL | 方法一句话 | flowme 环节 | 可判实验（判据） |
|---|---|---|---|---|---|
| C1 | 课程 / 数据顺序 | Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning https://arxiv.org/abs/2506.11300 | 按难度/质量排序喂数据替代均匀随机采样，同数据下更快收敛 | 输入卡训练的数据管道（只改顺序、不加参数） | 判据：同 FLOPs、2 seed，课程组达到基线 4000 步水平所需步数减少≥20% ⇒ 真省步；仅终分高但步数不少 ⇒ 是质量增益不是省算力 |
| C2 | 数据重复上限 | Scaling Data-Constrained Language Models https://arxiv.org/abs/2305.16264 | 数据受限时重复有次数上限，超限增益按幂律衰减 | 判"4000 步才显现"是步数问题还是数据问题 | 判据：固定 FLOPs 比多样新数据 vs 重复数据；若重复组增益 ≈0 而新数据组 Δ≥2×SE ⇒ 瓶颈是数据多样性，堆步数无效 |
| C3 | 算力最优配比 | Training Compute-Optimal Large Language Models (Chinchilla) https://arxiv.org/abs/2203.15556 | 给定 FLOPs 时参数量与训练 token 数应同比例缩放 | 决定"加大模型"还是"多训步"兑现 4000 步后的增益 | 判据：同 FLOPs 两种配比点数差 <2×SE ⇒ 配比不敏感，省算力转向 C1/C2；差显著 ⇒ 按 compute-optimal 重分配预算 |
| C4 | 用 scaling law 预判步数 | Scaling Laws for Neural Language Models https://arxiv.org/abs/2001.08361 | loss 随参数量/数据/步数呈幂律，可用前段曲线外推 | 预判增益显现步数、排训练预算 | 判据：前 1/3 曲线外推 4000 步后分数，实测落 95% CI 外 ⇒ 存在非平滑相变（与约束④一致），不能靠外推省步数，必须实测 |

## 优先级建议（先做这 3 个）
1. **B3→B1→B2 干预链（最便宜、先做）**：零训练成本，直接回答"每张卡是不是装饰"；attribution 粗筛（每次前/反向）→ path patching 精查（每卡 1 次配对 forward）→ 可疑卡 retrain-without 终审。**成本 ≈ 平时若干次推理 + 1 次 4000 步训练**，应在一切新机制之前完成（约束②要求终审必须训练）。
2. **A3（FSQ+STE 离散瓶颈）与 A1（连续中介）同批对照**：直接服务文本生成主线，判据已内置 argmax 负控、max_naive 地板、retrain-without 终审。**成本 ≈ 3 配置 × 2 seed 次满 4000 步训练**（受约束④必须跑满），是三者里最贵的一项。
3. **C1 课程/数据顺序**：只改采样不加参数，2 seed 对照，**成本最低且可与 2 并行**；直接检验"增益 4000 步才显现"能否提前。若 C1、C2 同为阴性 ⇒ 瓶颈在数据不在调度，省算力方向要改。

## 未验证清单
1. `web_fetch` 全域不可用（`resolves to a non-public IP`，本环境硬限制）⇒ **所有 URL 未做 HTTP 逐条点开**，可点性只到检索级核对。
2. 未核对任何论文的具体数值（分数/超参），报告不引用原文数字。
3. arXiv 版本号（v1/v2/…）与最终发表版是否一致未逐条确认；A3、B1、B6 等部分条目的两来源同属 arXiv 系镜像（ar5iv + arxiv.org），独立性弱于 HF papers/ACL/S2 组合。
4. B3（Attribution Patching）以 ACL Anthology id `2024.blackboxnlp-1.25` 跨 ACL Anthology + Semantic Scholar 两源核到，**未核到独立 arXiv id**（未找到）。
5. ★ 兜底抽查（换措辞独立复核）：最可疑条 **B5 / arXiv 2506.19732**（最新 id + 最小众标题），改用"作者名 + Shapley 方法词"查询 ⇒ ar5iv、Semantic Scholar、arXiv 列表三处 id+标题一致，**通过**；另抽查 C1 / 2506.11300 改措辞 ⇒ ACL Anthology `2026.eacl-long.271` 与研究机构库同标题一致，**通过**。故不触发整单降级。
6. 实测/推断分栏：表格"论文+URL、方法一句话"= 检索实测；"flowme 环节"与适用性判断（含规模、4000 步）= **（推断）**，未经本环境训练验证。

## 判定：**清单可用**
P1（A=5 / B=6 / C=4，均 ≥2）、P2（15 条五项齐全）、P3（每条 ≥2 来源一致 + ★ 抽查通过、未降级）、P4（未验证清单含 web_fetch 不可用）、P5（无"训练后删组件当消融"：在线 ablation 一律只作筛查、终审一律 retrain-without；离散中介走 STE；新机制先过 max_naive；读出≠会用写进 A5 判据）均满足。残余风险仅"无法 HTTP 点开"，属环境限制且已列。
