# nanoSeek 网络层盘点与卡片结构候选

> 只读实测 2026-10-08（`nanoSeek/` 未改一字，`git status --short` 空）。路径相对 `nanoSeek/`。
> 参数量为**按 `configs/base_v2.yaml` 手算的解析值**（`n_embd=512 · n_head=8 · n_layer=12 · kv_lora_rank=96 · n_experts=4 · moe_hidden_scale=1.333`），
> 未运行 python 实测 ⇒ 标 **[推断]**；行号、形状、调用点均为现场读到。
> "训练中被调用"依据 `configs/*.yaml` + `training/train.py` 默认值判读，未跑训练验证。

---

## 1. 盘点总表（文件:行 | 类名 | 是什么 | 参数量 | [n,d]→[n,d]? | 训练中是否被实际调用）

| # | 文件:行 | 类名 / 函数 | 是什么（一句话） | 参数量 [推断] | `[n,d]→[n,d]`? | 训练中实际调用？ |
|---|---|---|---|---|---|---|
| 1 | `model/utils.py:32` | `RMSNorm` | 逐位置 RMS 归一化（零均值去除、无 bias） | 512 / 处 | ✅ 逐位置 | ✅ 每层 `ln_1`/`ln_2` + `ln_f`（`block.py:34`、`gpt.py:45`） |
| 2 | `model/utils.py:23` | `logsumexp_residual` | 对数域 soft-max 合并两残差分支，替代线性相加（零参数） | 0 | ✅ | ⚠️ 配置开关：`use_lse_residual`（`base_v2.yaml:120` = false）；`dev-notes` 有 A/B 历史 |
| 3 | `model/utils.py:93` | `sinkhorn_knopp` | 交替行列归一化 → 双重随机矩阵（谱范数 ≤1，非扩张） | 0 | n/a（hc×hc） | ⚠️ 仅 mHC 开时（`block.py:127`）；`base_v2.yaml:121` false |
| 4 | `model/utils.py:63,76` | `precompute_rope_freqs` / `apply_rotary_pos_emb` | RoPE 相对位置注入（对 q/k 旋转，v 不旋转） | 0（buffer） | ✅ 保持 q/k 形状 | ✅ `use_rope: true`（`base_v2.yaml:126`） |
| 5 | `model/utils.py:114` | `zeropower_via_newtonschulz` | Muon 优化器的 Newton-Schulz 正交化（优化器算子，非前向层） | 0 | n/a | ✅ `use_muon: true`（`base_v2.yaml:125`） |
| 6 | `model/utils.py:162` | `zeropower_via_newtonschulz_split` | 同上，按注意力头分块正交化（GLM-5 Muon Split） | 0 | n/a | ✅ `muon_split: true`（`base_v2.yaml:127`） |
| 7 | `model/attention.py:10` | `CausalSelfAttention` | 因果自注意力总成（标准 MHA / MLA / CSA / KV 记忆四路互斥，由 config 选） | 0.67M(MLA) / 1.05M(MHA) | ✅ | ✅ (`Block.attn`) |
| 8 | `model/attention.py:143` | `_apply_qk_norm` | 对 q/k 做 L2 归一 + 每头可学习 scale（`nn.Parameter`, nh 个） | 8 | ✅ | ✅ `use_qk_norm: true`（`base_v2.yaml:126`） |
| 9 | `model/attention.py:110` | `attn_sink`（`nn.Parameter`） | 每头一个标量，作为 softmax 的"垃圾桶"列（value 为零向量） | 8 | ✅（softmax 多一列） | ✅ `use_attn_sink: true`（`base_v2.yaml:97`） |
| 10 | `model/attention.py:95-102` | MLA（`q_proj`/`kv_down`/`kv_act`/`k_up`/`v_up`） | 多头潜在注意力：KV 共享低秩潜在再展开 | 0.67M | ✅ | ✅ `use_mla: true`（`base_v2.yaml:122`） |
| 11 | `model/attention.py:298` | `_csa_forward` | 压缩稀疏注意力：块级 KV 池化 + top-k 块选择 + 因果滑窗 + HCA 全局摘要 | ~1.18M | ⚠️ 内部 n→nb 压块，输出回 [n,d] | ❌ `use_csa: false`（`base_v2.yaml:99`；`train.py:170` 默认也是 false） |
| 12 | `model/attention.py:562` | `_compress_block` | 可学习门控池化：块内 m 个 token → 1 个潜在 | 131K | ❌ m→1 压块 | ❌ 仅 CSA + `use_csa_learnable` 路径 |
| 13 | `model/attention.py:573` | `_kv_memory_forward` | GLA 式线性注意力状态：逐头外积写入 `A←r⊙A+βkvᵀ`、遗忘门 + 读 | ~0.33M/层 | ✅（`mem_up`: nh·l→C） | ❌ `base_v2.yaml:105` false（⚠️ `train.py:178` 默认 true，被 yaml 覆盖） |
| 14 | `model/attention.py:669` | `_kv_memory_forward_delta` | DeltaNet 擦写：先擦后写 `S←r⊙S+w(v−Sk)kᵀ`（消键冲突） | 同 #13 | ✅ | ❌ `kv_memory_delta: false` |
| 15 | `model/attention.py:727` | `_kv_memory_forward_block` | 块级写入 + 逐 token 读取（P1 chunk 公式块级化） | 同 #13 | ✅ | ❌ `kv_memory_block: 1`（等于关闭块级） |
| 16 | `model/attention.py:711,781` | `_mem_delta_chunk` / `_mem_chunk_step` | 块内顺序/并行递推步（供梯度检查点重算） | 0 | ✅ | ❌ 同上两路 |
| 17 | `model/mlp.py:7` | `SwiGLU` | 门控 FFN：`SiLU(xW1)⊙(xW2)`，可选输出钳制 | 1.05M(`moe_hidden_scale`) / 2.1M(默认) | ✅ | ✅ `use_moe: true` 时作为专家/共享专家；否则作整层 FFN |
| 18 | `model/mlp.py:38` | `MoE` | top-k 专家路由 + 共享专家 + aux-free bias 均衡 + Router Z-Loss | ~5.24M/层 | ✅ | ✅ `use_moe: true`（`base_v2.yaml:123`） |
| 19 | `model/block.py:12` | `Block` | 双子层残差块：pre-norm + attn + FFN，可选残差合并模式/顺序/skip_attn/mHC | ~5.91M/层 | ✅ | ✅ `gpt.py:44,189` |
| 20 | `model/block.py:32` | `skip_attn` 布线（`no_attn_layers`） | 指定层的注意力槽换成宽 SwiGLU 作等参对照 | ≈同注意力 | ✅ | ❌ `no_attn_layers` 空；`dev-notes/87` 已配对否决 |
| 21 | `model/block.py:47-52,95,114` | mHC（`raw_A/B/C_*`） | 多流并行残差超连接：A 压流 → F 跑 1 次 → C 展开 → B(双随机) 混合 | 16（hc=2） | ✅（hc 流） | ❌ `use_mhc: false`（`base_v2.yaml:121`） |
| 22 | `model/block.py:58,74-76` | `raw_gate`（LSE 门控混合 v2） | `α·x+(1−α)·LSE(x,F)`，每层一个可学习标量 | 1/层 | ✅ | ❌ `use_lse_gate: false` |
| 23 | `model/block.py:141` | `MTPModule` | 多 token 预测：`hidden_proj+emb_proj` 融合后过一层 Block 预测 t+2 | ~6.44M | ✅ | ❌ `use_mtp: false`（`base_v2.yaml:124`；`n_mtp: 1` 保留） |
| 24 | `model/gpt.py:34` | `GPT` | 主干：Embedding + N×Block + `ln_f` + 输出头 | ~75M（全模型） | ✅ | ✅ |
| 25 | `model/gpt.py:69` | `memory_tokens`（`nn.Parameter`） | 显式记忆 token 前缀，拼到序列前、过完所有层再剥离 | K×512 | ⚠️ 改变 n（K+t） | ❌ `n_memory_tokens: 0` |
| 26 | `model/gpt.py:51-53` | byte_level（`byte_emb`/`byte_agg`/`byte_unagg`） | 字节直入 + 3:1 可学习软聚合，无 BPE | — | ❌ 3T→T 改 token 轴 | ❌ `byte_level: false`（`base_v2.yaml:10`）；`char_level: true` |
| 27 | `model/gpt.py:59-61` | factorized emb（`emb_proj`/`head_down`） | ALBERT 式因式分解嵌入，低秩后再升维 | — | ❌ 改 d | ❌ `factorized_emb_dim: 0`（`base_v2.yaml:27`） |
| 28 | `model/gpt.py:92-96,201-209` | CD 头（`cd_block`+`cd_adapters`） | 共享权重 Block 循环 `cd_iters` 圈，每圈独立适配器残差注入 | 5.91M + k×0.26M | ✅ | ❌ `use_cd` 仅 `configs/base_v2_rec.yaml:13` 开；`dev-notes/87` 已否决 |
| 29 | `model/ngram_ndb.py:117` | `NgramNDB` | 外挂后缀 n-gram 库：no_grad numpy 表 + 可学习读/写门控，改 logits 不改 hidden | 门控 ~1.0K + 表 ~3.0GB(slots=2²⁸) | ⚠️ 输入 h[n,d]，输出改 p(V) | ✅ 训练**默认组件**（`train.py:785`、`base_v2.yaml` ndb_slots） | 
| 30 | `model/byte_tokenizer.py:14` | `ByteTokenizer` | 字节级 encode/decode（无 `nn.Module`） | 0 | n/a | ❌ 当前 char_level |
| 31 | `nano_arith/model.py:68,103,120,133` | `CausalSelfAttention`/`FeedForward`/`TransformerBlock`/`NanoArithTransformer` | 紧凑 Transformer 四件套（算术子项目，独立 config） | ~161K（默认 cfg） | ✅ | ✅ `nano_arith/train.py` |
| 32 | `nano_arith/model.py:40` | `RotaryEmbedding` | RoPE cos/sin 缓存（buffer，非参数） | 0 | n/a | ✅ |
| 33 | `local/proto_external_memory.py:33` | `ExternalMemory` | no_grad 外部记忆库（top-k 余弦检索 + EMA 写 + GC 复活） | 2×4096×128 冻结 | ✅ 检索回 [N,d] | ❌ 原型脚本 |
| 34 | `local/proto_external_memory.py:138` | `InterfaceNetwork` | 接口网络：`encode_query`/`judge`(surprise 门)/`decode` 三个 Linear | ~132K | ✅ | ❌ 原型脚本 |
| 35 | `local/sync_overhead_test.py:38` | `DB` | `ExternalMemory` 的极简版（同步开销测试） | 冻结 | ✅ | ❌ 测试脚本 |
| 36 | `local/sync_overhead_test.py:27` | `Iface` | 接口网络极简版（q_proj/q_norm/v_proj） | — | ✅ | ❌ 测试脚本 |
| 37 | `pre-research/interface/command_bridge.py:55` | `CommandBridge` | n-bit 指令直通桥：旁路网络出 8 位指令 + payload 头 + 读出解码接回主干 | ~18.7K | ✅（dec 加性接回） | ❌ 尝试期，`pre-research/` |
| 38 | `pre-research/interface/command_bridge.py:43` | `StraightThroughBin` | STE 二值化：前向 `p>0.5`、反向梯度直通 | 0 | ✅ | ❌ 同上 |
| 39 | `pre-research/db_engine/vdb.py:40` | `VDB` | KV+Vec 混合记忆库（非 `nn.Module`） | 0 | n/a | ❌ 同上 |
| 40 | `inference/runtime/src/{attention,model}.rs` | Rust `CausalSelfAttention`/`Config` | 前向逻辑的 Rust 推理重实现（candle） | 同 Python 主干 | ✅ | ❌ 推理专用，非训练层 |

**覆盖的模型定义文件**：`model/{utils,attention,mlp,block,gpt,ngram_ndb,byte_tokenizer}.py`、`nano_arith/model.py`、
`local/proto_external_memory.py`、`local/sync_overhead_test.py`、`pre-research/interface/command_bridge.py`、
`pre-research/db_engine/vdb.py`、`inference/runtime/src/*.rs`。
全仓 `grep "class .*nn\.Module"` 只有 **19 个类**，本表全覆盖。

---

## 2. 与 flowme 已试结论的对照

### 已试（重复，不要再试）

| flowme 已试项 | nanoSeek 对应资产 | 证据 |
|---|---|---|
| ❌ 纯逐位置 MLP（0.00%）| `model/mlp.py:7` `SwiGLU`、`nano_arith/model.py:103` `FeedForward` —— 单独作卡内结构已试 | 结论表；两个 FFN 都是逐位置算子 |
| ❌ 因果卷积 k=3（0.12%）| **库内无 `nn.Conv` 任何形态**（全仓 grep 实测无输出）| 无新角度可借 |
| ❌ GRU 式门控（0.50%）| **库内无 GRU/LSTM**；`SwiGLU` 的乘性门是逐位置门、非递推门 | 无递推门资产 |
| ❌ 仅注意力去 FFN（0.62%）| `model/block.py:36` FFN 一行可换/可关；`model/gpt.py:189-194` 逐层调用 | 同结构，已覆盖 |
| ❌ 共享权重循环 k≥4（k=8 EM 0.25%，不收敛）| `model/gpt.py:92-96,201-209` CD 头；`dev-notes/87` 配对短跑无增益（+41% 墙钟/+9.9% 参数）| 两处独立否决 |
| ❌ 有害：加性位置编码进思维卡（−34.9pp）| `model/gpt.py:72-73` `use_rope=false` 时的 `wpe` 加性 PE；两条路径互斥 | RoPE 是旋转、不是相加 |

### 未试（候选）

- **残差加在哪一层 / 怎么加** —— `model/block.py:72-78` 三种合并（加法 / 纯 LSE / 门控 LSE）+ `block_order: ffn_attn` 顺序重排。
- **Norm 位置（Pre/Post/无）** —— 现有固定 pre-norm（`block.py:34-35,81,89-92`），Post/无未实现但改一行可换。
- **卡内深度 L=1/2/4** —— `Block` 是单层可堆叠单元，`n_layer` 可控；`MTPModule`(`block.py:157`) 内部复用一层 `Block`。
- **输出卡读出方式** —— `compute_logits`(`gpt.py:145`) 逐位置头；`generate` 只取末位(`gpt.py:239`)。读出头本身未有替代实现。
- **绑定 `h_i ⊛ p_i`（而非相加）** —— ★ 库内最接近的原生实现是 **KV 记忆的外积绑定**（`attention.py:570-571` 状态 `S` 是 key-value 外积矩阵，检索 `S·q` 即绑定后读取）；另有 `n_memory_tokens` 前缀（`gpt.py:69`）是"位置身份"，但它是拼接不是绑定。

---

## 3. Top-6 卡片结构候选（含判据写法）

> 五维报告卡口径见 `speculation/CARD_PRIMITIVES.md:41-63`：能力 EM / 因果门 B 恒等中介 Δ / 集中度（前 20% 单元因果占比） / 成本（参数+训练墙钟+推理墙钟） / 自检 R28+R29。

**C1 · KV 记忆的外积绑定（GLA 式线性注意力状态）** — `model/attention.py:573`（+ delta 版 `:669`）
- **为什么值得试**：S18/S20 的异常（互换两行 ⇒ 下游 `ΣαV` 不变）指向"卡内只有相加、没有绑定"；这里是库内**唯一的原生 key-value 外积绑定**（`S ← r⊙S + β·k·vᵀ`，读取 `S·q`），且状态是可解释的 `nh×l×l` 矩阵、可整块关闭。形状严格保持 `[n,d]→[n,d]`（`mem_up`: `nh·l→C`）。
- **判据**：能力 = add_3d 同任务 EM + 数字每步正确率按难度桶；门 B = `mem_up` 输出置零（恒等中介）跑配对 Δ，Δ 不显著 ⇒ 该块不必要；集中度 = 新 ckpt 前 20% 单元因果占比 vs add_3d 的 44.4%/49.6%；成本 = 参数 ~0.33M/层（可只挂最后 1–2 层，`kv_memory_layers`）、训练墙钟（chunk=32 已实测 1.36 it/s）、推理墙钟；自检 R28 = batch=1 vs batch=N 逐字一致（注意状态 fp32 累加，bf16 会丢）、R29 = 保留率 r 与写入门 β 的非零/非饱和比例。

**C2 · LSE 残差 / 门控 LSE 残差（残差怎么加）** — `model/utils.py:23` + `model/block.py:72-78`
- **为什么值得试**：flowme 已定"要有残差"但不知道"怎么加"；这是**零参数、规模严格不变**的连接改变（对照口径干净），且 nanoSeek 已实现三档（加法 / 纯 LSE / 门控混合），可直接搬。
- **判据**：能力 EM 及分桶；门 B = 三种合并模式两两配对 Δ（同 seed、同步数）；集中度 = LSE 的有界收缩是否让更多单元参与（占比应下降）；成本 = 0 参数、墙钟逐位可比；R28/R29 与 C1 同。

**C3 · mHC 多流残差（残差拓扑 + 双随机混合）** — `model/block.py:95,114` + `model/utils.py:93`
- **为什么值得试**：把"残差加在哪一层"推进到"残差本身有 hc 条流、由 Sinkhorn 双随机矩阵混合"（谱范数 ≤1 ⇒ 非扩张、深堆稳定）；参数量仅 16（hc=2），hc=1 退化为标准残差 ⇒ 天然恒等中介消融。
- **判据**：能力 EM；门 B = hc=1 vs hc=2 配对 Δ；集中度（双随机约束防单流主导）；成本 = +16 参数、墙钟因 4 流张量放大需实测；R29 = B 矩阵不得退化成单位阵/均匀阵（否则等于没混合）。

**C4 · Attention Sink（输出读出的"垃圾桶"列）** — `model/attention.py:110`（+ 注入路径 `:232-247`, `:276-282`）
- **为什么值得试**：8 参数/层，base_v2 **已实测开启**（`base_v2.yaml:97`），是"输出卡读出方式"里最便宜的候选——把无处可去的 softmax 质量导向零向量，直接改变"集中度"。
- **判据**：能力 EM；门 B = sink 关/开配对；集中度 = 注意力权重按因果效应排序的前 20% 占比；成本 ≈0；R28/R29 = sink 是否非零、是否被推到极端。

**C5 · QK-Norm（每头 L2 归一 + 可学习 scale）** — `model/attention.py:143`（参数 `:37`）
- **为什么值得试**：pre-norm 之外的**归一化位置**实验（flowme 未试项"Norm 位置"），8 参数/层，形状保持，且与 RoPE 顺序无关（norm 保持范数）。
- **判据**：能力 EM；门 B = `use_qk_norm` 关/开配对；集中度；成本 = 8 参数；R28/R29 = scale 各头是否分化（全相等 ⇒ 未起作用）。

**C6 · SwiGLU 门控钳制（抗尖刺）** — `model/mlp.py:29-32`（`swiglu_clamp`，base_v2 = 10.0）
- **为什么值得试**：零参数、一行开关，**从源头压制异常值**（异常值就产生在门控乘积里），直接对症 EM 与集中度里"少数单元扛全部因果"；也是 `logsumexp_residual` 注释里点名的尖刺来源。
- **判据**：能力 EM 分桶 + 每步正确率；门 B = clamp 开/关配对；集中度（若 clamp 有效，前 20% 占比应下降）；成本 = 0；R29 = 被钳位置比例（若≈0 则等于没开）。

---

## 4. 排除清单（看起来像候选、实际不适合）

| 排除项 | 位置 | 一句理由 |
|---|---|---|
| CSA/HCA 压缩稀疏注意力 | `model/attention.py:298` | 内部把 m 个 token 压成 1 个块（`nb≠n`），n 轴被改变；且 6 个开关组合未验证、`base_v2` 关闭 |
| `_compress_block` 门控池化 | `model/attention.py:562` | 显式 m→1 降维，不满足卡内形状保持 |
| byte_level 3:1 聚合 | `model/gpt.py:51-53` | token 轴 3T→T 改变，词表与评测口径全换，混淆归因 |
| factorized embedding | `model/gpt.py:59-61` | 改 d（低秩），是参数预算技巧不是卡内计算 |
| `memory_tokens` 前缀 | `model/gpt.py:69` | 改变 n（K+t），属于卡间黑板/工作区，不是卡内结构 |
| CD 共享权重循环 | `model/gpt.py:92-96` | 已证伪：k=8 不收敛（flowme）；nanoSeek `dev-notes/87` 配对也无增益 |
| MTPModule | `model/block.py:141` | 训练期辅助头、推理不使用，引入额外监督与参数会混淆卡内归因 |
| MoE | `model/mlp.py:38` | 5.24M/层（占主干约 70%），条件计算使"整块 retrain-without"不可比 |
| `skip_attn` 布线 | `model/block.py:32` | nanoSeek `dev-notes/87` 实测：等参数注释是标准注意力口径，MLA 下净增 13% 参数且无增益 |
| NgramNDB | `model/ngram_ndb.py:117` | 外挂、改 logits 不改 hidden，表 3.0GB numpy，收益 Δ≤0.0013 nats 不迁移 val |
| ExternalMemory / `DB` | `local/proto_external_memory.py:33`、`local/sync_overhead_test.py:38` | 参数冻结 + no_grad 规则写，是库不是层；原型/测试脚本 |
| InterfaceNetwork / `Iface` | `local/proto_external_memory.py:138`、`local/sync_overhead_test.py:27` | 接口网络原型，形状保持但无训练入口与评测口径 |
| CommandBridge / StraightThroughBin | `pre-research/interface/command_bridge.py:55,43` | STE 是不可微近似、依赖 VDB 外部引擎，`pre-research/` 尝试期 |
| VDB | `pre-research/db_engine/vdb.py:40` | 纯存储引擎，非 `nn.Module` |
| Rust runtime | `inference/runtime/src/*.rs` | 推理重实现，不可训练 |
| Newton-Schulz 两函数 | `model/utils.py:114,162` | 优化器算子，不在前向图上 |
| `logsumexp_residual`/`sinkhorn_knopp` 单用 | `model/utils.py:23,93` | 零参数连接算子，必须挂进 C2/C3 才有消融对象 |

---

## 5. 未覆盖 / 不确定项

1. **参数量全部是解析手算 [推断]**，未运行 `.venv/bin/python` 验证（任务禁 GPU/训练，且不想误触环境）。
2. **"训练中被调用"依据配置与默认值判读**，未跑训练核实；`train.py` 的默认值与 `configs/base_v2.yaml` 有冲突项（`use_csa`/`use_kv_memory`），实际以 yaml 为准。
3. **`data/external/TheAlgorithms-Python`（数千文件）未逐文件读** —— 属第三方示例库，不在本项目模型定义范围；本报告的全仓 `nn.Module` grep 排除了 `data/`、`out/`、`temp/`。
4. **`inference/scripts/*.py`（12 个）未逐个通读** —— 采样/评估脚本，grep 确认无 `nn.Module` 类；可能有 hook 式注入逻辑未覆盖。
5. **`out/` 下 4 万余文件（含 100+ 组 run 产物与 `logs_archive/`）未读** —— 不进 git，按 `AGENTS.md` 规则状态现场读，本报告只作层盘点的来源旁证。
6. **`model/probe.py`（观测台，464 行）未逐行读** —— grep 确认无 `nn.Module` 类，只做 hook 记录，故未进 §1。
7. **Rust `inference/runtime/src/attention.rs` 未读**，只读了 `model.rs`/`main.rs` 前 40 行；Rust 侧是否有 Python 没有的算子（如 anticipatory routing）**不确定**。
8. **`dev-notes/` 未通读**：LSE 残差、mHC、KV 记忆各条的历史 A/B 数字未逐条核对，只引用了 `AGENTS.md` 已收录的 `dev-notes/87` 结论。
9. **hf/`nano_arith` 与主模型是否共用卡内结构未验证**：它是独立子项目（独立 `ModelConfig`），本报告只作并列入表。
