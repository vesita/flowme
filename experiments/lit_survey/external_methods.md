# LP2 外部文献调研：为 flowme 方法论找可落地方案

> 检索：`web_search` 可用；**`web_fetch` 全域不可用**（example.com / arxiv.org / api.semanticscholar.org 均报 `resolves to a non-public IP`）⇒ URL 核验 = 每条用 web_search 命中「同一 arXiv id + 同一标题」的独立来源交叉确认，**未做 HTTP 逐条点开**。

## 轴 A · 解释模型（因果）：验证「某卡/某表示真被使用」

| 方案 | 代表论文（标题 + URL） | 方法一句话 | flowme 环节 | 可判实验（含判据） |
|---|---|---|---|---|
| A1 Activation patching | [Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small](https://arxiv.org/abs/2211.00593) | 前向时把某处激活替换成 clean/corrupt 对照样本的激活，看任务指标回落多少 | 思维卡输出（黑板上的模因张量） | 干预卡 k：Δ = acc(clean) − acc(patch k)；判据：2 seed 同号且 abs(Δ) ≥ 2×SE ⇒ 卡被使用；全部卡的 Δ 与 random-patch 基线同分布 ⇒ 卡是装饰。须与 retrain-without 删卡同向，否则记不一致 |
| A2 Path patching | [Localizing Model Behavior With Path Patching](https://arxiv.org/abs/2304.05969) | 只替换指定「起点→终点」路径上的中间激活，其余边保持 clean | 卡 k → 黑板 → 卡 j 的通信边 | 只 patch 路径 k→j：Δ ≥ 2×SE 且 2 seed 同号 ⇒ 该通信边真实；只有全局 patch 有效、路径 patch 全无效 ⇒ 卡间共享空间是装饰（直接回应「卡间共享未验证」） |
| A3 自动电路发现 ACDC | [Towards Automated Circuit Discovery for Mechanistic Interpretability](https://arxiv.org/abs/2304.14997) | 逐边做反事实替换、按 KL 增量剪出贡献子图 | 手写调度的裁剪与核对 | retrain-without：删 ACDC top-k 边配对 Δ < −2×SE、2 seed 同号；删 bottom-k 边 abs(Δ) ≤ SE ⇒ 调度可裁剪；若删 top 边不掉点 ⇒ 发现的图 ≠ 实际用途 |
| A4 SAE 特征因果必要性 | [Are Single-Token Sparse Autoencoder Features Causally Necessary? Layer-Depth and SAE-Family Effects](https://arxiv.org/abs/2607.20596) | 得到 SAE 特征后不只测解码/重构，而是 ablate 该特征测输出变化 | 模因张量的语义特征（直击「读出 ≠ 会用」「语义理由给不出」） | 线性探针可解码但 ablate 特征的 Δacc ≤ SE ⇒ 判「只可读出、无因果用途」；ablate Δ ≥ 2×SE 且 2 seed 同号 ⇒ 才算有语义理由。消融另走 retrain-without，不混用 |

## 轴 B · 卡片组合与路由：多卡组合、路由、迭代

| 方案 | 代表论文（标题 + URL） | 方法一句话 | flowme 环节 | 可判实验（含判据） |
|---|---|---|---|---|
| B1 MoE top-k 卡路由 | [Switch Transformers: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) | 门控按输入选 top-2 专家，其余不激活 | 手写调度 → 学习到的卡路由 | Δ = acc(学习路由) − acc(手写调度)，2 seed 同号且不劣（abs(Δ) ≤ 1×SE 或更好）+ 路由熵非零 ⇒ 可替换手写调度；某卡路由占比趋 0 且 retrain-without 删它不掉点 ⇒ 该卡是装饰 |
| B2 Mixture-of-Depths 卡门 | [Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/abs/2404.02258) | 每层带 capacity 限制的保留门，动态跳过部分计算 | 串联段选几张卡（呼应「串联优于并行」） | 恒激活 100% 的对照：同 FLOPs 下 acc 差 ≤ SE ⇒ 门是装饰；否则按实际激活卡数 n 分桶，低 n 桶 acc 显著低于高 n 桶 ⇒ 学到按需计算 |
| B3 自适应计算时间 ACT | [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983) | 可微 halting 概率决定同一模块迭代多少步 | 思维卡迭代复用（替代固定串联长度） | 难度与平均步数 n̄ 正相关（秩相关显著）且截断一半步数后 abs(Δ) ≤ SE ⇒ 真自适应；n̄ 恒定 ⇒ 退化为固定深度（装饰）。须在 ≥4000 步处比较（增益对训练量敏感） |
| B4 神经模块网络 + 组合泛化评测 | [Deep Compositional Question Answering with Neural Module Networks](https://arxiv.org/abs/1511.02799)；评测划分用 [COGS](https://arxiv.org/abs/2010.05465) | 按程序动态堆叠可复用模块 | 卡 = 模块、调度 = 程序 | 训练只见部分卡序、测试见新卡序的划分下比固定手写调度提升 ≥ 1 点且 2 seed 同号 ⇒ 组合泛化有效；只在随机划分上等价 ⇒ 模块化是装饰 |

## 轴 C · 表示与通信：共享空间、通信、离散瓶颈

| 方案 | 代表论文（标题 + URL） | 方法一句话 | flowme 环节 | 可判实验（含判据） |
|---|---|---|---|---|
| C1 可微 agent 通信 DIAL | [Learning to Communicate with Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1605.06676) | agent 经可微信道交换连续消息，端到端学通信内容 | 卡间经黑板传递的模因消息 | 推理期 mask 消息（换噪声）：Δ ≥ 2×SE 且 2 seed 同号 ⇒ 消息被用；mask 不掉点 ⇒ 通信是装饰；须与 retrain-without 去通道同向 |
| C2 VQ-VAE + commitment loss | [Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) | 离散码本量化 + commitment loss，梯度走直通估计 | 模因张量的离散瓶颈（现有 (hard-soft).detach()+soft STE 的同族方案） | 双判据：有效码字使用率（codebook perplexity）坍缩到 5% 以下 ⇒ 瓶颈退化；互换两张卡的码本条目后 acc 差 ≤ SE ⇒ 离散瓶颈是装饰 |
| C3 表示对齐度量 CKA | [Similarity of Neural Network Representations Revisited](https://arxiv.org/abs/1905.00414) | linear CKA 量化两组激活共享子空间的比例 | 卡间共享空间的「先发现、后验证」 | CKA 高的卡对做 retrain-without 删其一：另一卡在该子空间表示塌陷（CKA 降 > 0.1）⇒ 真共享；CKA 高但删谁都不掉点 ⇒ 冗余非共享。CKA 本身只相关不因果，必须联用 retrain-without 才可判 |

## 优先级建议（先做这 3 个）

1. **A1 activation patching**：推理期干预、不改训练，最便宜，直接回答「思维卡是否被旁路」（已实测：恒等中介即可 acc 0.90）；判据形态（配对 Δ ± SE、2 seed 同号）现成可写。
2. **A4 SAE 因果必要性**：直击「读出 ≠ 会用」「语义理由给不出」，给每个模因方向一条必要性证据；复用 A1 的干预管线。
3. **C1 消息 mask**：回应「卡间共享/通信未验证」，只需加一个推理期 mask 开关，比 retrain-without 便宜；与 retrain-without 方向一致才升级为结论。
   B 轴放后面：路由/门控都要重训，且增益对训练量敏感（4000 步才显现），先用 A/C 小成本判断 B 值不值得投。

## 未验证清单

- `web_fetch` 全域不可用 ⇒ 上述 URL **未做 HTTP 逐条点开**，只经 web_search 多来源交叉核对 id 与标题；「可点」未验证。
- 各方法是否适用于 flowme 的规模与非标准结构，未验证。
- 2607.20596 是 2026 预印本，结论稳定性未验证。
- 在「4000 步才显现增益」的训练量下，patching / 门控 / CKA 的信号强度未验证。
- A2/A3 默认 transformer 前向；flowme 的 STE 离散瓶颈是否改变干预与梯度语义，未验证。
- **未找到**：卡间「共享张量空间」的专门因果验证论文；神经黑板架构的现代可判消融论文（暂以 DIAL 类通信 + retrain-without 替代，替代是否等价未验证）。

## 判定：部分可用

三轴各 ≥2 方案、每方案 5 项齐全、URL 均经检索交叉核对（无编造）；但 `web_fetch` 不可用导致 URL 可点性未做 HTTP 级核验，故不给「清单可用」。
