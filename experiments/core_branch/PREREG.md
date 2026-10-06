# E-D 预注册：冻结认知核 + 非线性/更深的并联分支（core_branch）

落盘时间：2026-10-06 03:37（+08:00）。**训练开始之前写死；判据事后不改。**
前置阅读：`dev-notes/15` §8（E-B）、§10（E-C）；对照实验 `experiments/additivity/PREREG.md`（E-B）。

## 0. 唯一要回答的问题与假设

**问题**：把 E-B 的"线性低秩旁路"换成"非线性/更深的并联分支"，能否在**不改动既有核权重**
（老卡路径逐位不变）的前提下，把新能力（`negation`）补到联合档水平？

**假设**：缺的不是适配容量，而是**非线性/分层的特征变换能力**。残差式可训分支能模拟联合训练对核的
17% 改写，从而补上缺口；老卡路径被 gate 关掉 ⇒ 结构上零损伤。

**背景（已实测，不重测）**：E-B r16 旁路 0.7267/0.7433、全秩 0.7383、r32×64ep 0.7967（~0.80 平台）、
frozen 0.6933/0.6500、joint 0.9667/0.9233；秩/步数已被排除为瓶颈；联合档改写基座 17.2%。

## 1. 三档分支（同一冻结基座、同一头、同一步数预算、同数据）

| 档 | `--branch` | 形态 | 插入点 | 零初始化方式 |
|---|---|---|---|---|
| **B1** | `mlp` | MLP 旁路 `h + W2·SiLU(W1·h)`，hidden=128（≈d），无 bias | E-B 同款 4 处：blocks[0..2] 输出 + norm 输出 | `W2` 精确零；`W1 ~ N(0, 1/d)` |
| **B2** | `block1` | 1 个完整 `NanoTransformerBlock`（Pre-RMSNorm 残差式，作用于 doc_memory 序列） | norm 输出（进解码头前） | `attn.proj` 与 `mlp.c_proj` 精确零 ⇒ `block(x) ≡ x` |
| **B3** | `block2` | 2 个块堆叠（看是否还需要更深） | 同上 | 两个块的输出投影都精确零 |

- 参数量**实测打印**（SELFTEST_1，不是估算）。事先估算：B1 = 4×2×128×128 = **131,072**
  （与 E-B 全秩线性旁路**逐一等参** ⇒ 隔离"非线性"这一个变量）；B2 ≈ 213k；B3 ≈ 426k。
- 可训：**分支 + 头**；基座 core `requires_grad=False`（SELFTEST_1 必须打印 `base_core_trainable=0`）。
- 块直接 `import` 自 `src/dtseek/encoder/nano_doc_encoder.py`（**不改 src/**）；
  `bypass.py` / `train_bypass_card.py` 只读复用其结构与空测试打印（**不改 additivity/**）。
- 分支构造在 `torch.random.fork_rng(devices=[])` + CPU `default_generator.manual_seed(seed+1000)` 内进行
  ⇒ **不扰动全局 RNG 流**，头初始化与数据顺序与 E-B 各 seed 逐位对齐（配对可比）。

### gate：零算术短路（P1 的结构性前提）

- forward hook 关闸分支：`return out if not self.enabled else ...` —— 关闸时**直接返回原张量对象**，
  不做任何算术 ⇒ 老卡计算路径与"没装分支"严格同一条。
- block 档额外用 `forward_pre_hook` 记录 RoPE cos/sin 与掩码上下文（只读输入与 buffer，不触碰返回张量）；
  该上下文不影响输出值。
- B1 的 4 处 hook 结构与 E-B `BypassSet` 逐行同款。

## 2. 预算与对齐（与 E-B 主配置逐项相同）

`1344 步 = 16 ep × 84`；batch 64；lr 1e-3；AdamW wd 1e-4；Cosine 到 T_max=1344；grad clip 1.0；
negation `samples=6000`（train 5400 / val 600，seed 划分）；头 `num_heads=4, num_layers=2`；
seed **42 与 43**；`--eval-every 4` 打 `EPOCH_EVAL` 曲线（诊断是否仍压 0.80 平台；
eval 时间计入 train_sec，配对用稳态步时）。

## 3. 预注册判据

- **P1（老卡零损伤）**：四张老卡（pronoun/sentiment/relation/person）Δ exact = 0 **且**逐样本
  `engine.predict` sha256 一致，复刻 E-B `eval_old_cards.py` 的 A（不挂）/ B（挂且关闸）/ C（**强制开闸
  敏感性正对照**）三条件；C 必须明显改变老卡（E-B 量级：pronoun 0.96→0.56），否则检查视为无效。
- **P2（新卡到位）**：`negation` exact ≥ **0.870（s42）/ 0.831（s43）**（E-B 预注册线，joint 档 0.9667/0.9233）。
- **P3（代价）**：报可训参数量、稳态步时、峰值显存，与 E-B r16 配对比较
  （+16,384 参数 / 0.0272 s·step⁻¹ / 259.5 MB；frozen 0.0207 / 172.3 MB）。
- **≥2 seed，两 seed 同号才下结论**（P2 过线与否须两 seed 一致；P1 两 seed 都要逐位成立）。

## 4. 空测试三条（缺一条结论作废）

1. **打印真实可训参数量**：SELFTEST_1 直接按 `requires_grad` 数（非估算），`base_core_trainable` 必须为 0。
2. **分支零初始化时 ≈ 冻结基线**：
   - 结构层：`max|Δ|(分支开@零 vs 关)` 与 `max|Δ|(分支开@零 vs 无分支基座)` 必须 **= 0**（精确）；
   - 指标层：每档每 seed 加跑一个 `--mode frozen`（分支在场、冻结在零、开闸）档，
     其 exact 必须 ≈ E-B 冻结基线 **0.6933（s42）/ 0.6500（s43）**，容差 **±0.01**；
     且同 seed 下三档 frozen 必须**逐位相等**（零初值下三档计算逐位同构 ⇒ 各跑一次代表三档的证据）。
3. **训后分支真在干活**：
   - 所有零初始化张量（W2 / attn.proj / mlp.c_proj）训后 `std > 0`，并报逐张量 `max|Δ|`；
   - **关闸后** exact 明显下降：**drop ≥ 10pt** 才算"明显"（E-B 对照：0.7267→0.4300，−29.7pt）。
   - 若关闸不掉：说明分支是空壳或头未与分支共适应 ⇒ 该档结论作废。

## 5. 判定规则（跑前写死）

- **任一档 P2 过线（两 seed）且其 P1 逐位成立、三条空测试全过 ⇒ 架构答案成立**：
  能力 = 冻结核 + 可训并联分支，老卡逐位不动。
- **三档都不过 P2 ⇒ 如实报不成立**，并给机制：曲线是否仍压 ~0.80 平台、是否仍远低于联合档
  （0.9667/0.9233）、与 E-B 线性旁路相比增益是否仍只在 +3~10pt 量级。
- P2 过线但 P1 不成立 ⇒ 架构答案不成立（零损伤被破坏）。
- 两 seed 不同号 ⇒ 只报方向，不下结论。

## 6. 运行清单（跑前计划）

```bash
systemd-run --user --unit=dtseek-core-branch --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  /bin/bash -lc 'exec bash experiments/core_branch/run_all.sh > logs/core_branch_run.log 2>&1'
```

- 空测试②：`mode=frozen` × {mlp,block1,block2} × {42,43} = 6 跑；
- 主档：`mode=bypass` × {mlp,block1,block2} × {42,43} = 6 跑；
- P1：`eval_old_cards_branch.py` × 6 卡 = 6 跑（A/B/C + 逐样本哈希）；
- 日志 `logs/core_branch_*.log`，产物 `experiments/core_branch/cards/*.pt`、
  `old_card_check_{mlp,block1,block2}_s{42,43}.json`，报告 `REPORT.md`。

## 7. 事先声明的解读纪律

- 所有 em 配对只在**同 seed 内**读（评测集划分 seed 不同，A 的绝对值不同是划分造成的）。
- 探索性附注（曲线形状、档间参数量差异）不改判据；判据只有 P1/P2/P3 + 三条空测试。
- 与 `src/`、`training/`、`tests/`、`dev-notes/`、`experiments/additivity/`、
  `experiments/capability_map/` 一律只读；只写 `experiments/core_branch/`、`logs/`、`/tmp`。
