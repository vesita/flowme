# E-D 报告：冻结认知核 + 非线性/更深的并联分支（core_branch）

预注册 `experiments/core_branch/PREREG.md`（03:37 落盘，首个训练 03:44:57 才开始，判据未改）。
主套件 18 跑 + 探索臂 4 跑**全部 rc=0**。

## 1. 三档实现、真实参数量、gate 零算术短路

- **B1 `mlp`**：4 处（blocks[0..2] + norm，与 E-B 同款插入点）`h + W2·SiLU(W1·h)`，hidden=128、无 bias，
  `W1~N(0,1/d)`、`W2` 精确零。**分支 131,072 参数 —— 与 E-B 全秩线性旁路逐一等参**（隔离"非线性"单变量）。
- **B2 `block1`**：1 个完整 `NanoTransformerBlock`（Pre-RMSNorm 残差式）作用于 doc_memory（norm 输出），
  `attn.proj`+`mlp.c_proj` 精确零 ⇒ `block(x)≡x` 初值。**213,252 参数**。
- **B3 `block2`**：2 块堆叠（两块输出投影都零）。**426,504 参数**。
- 可训 = 分支 + 头（629,764）；基座 core 1,688,460 **全部冻结**（SELFTEST_1 实测 `base_core_trainable=0`）。
- **gate 零算术短路**：forward hook 关闸分支 `return out if not self.enabled else ...` —— 关闸直接返回
  **原张量对象**；block 档另有 `forward_pre_hook` 只读记录 RoPE/掩码（不触碰返回值）。
  块直接 import 自 `src/.../nano_doc_encoder.py`，`additivity/` 与 `src/` 未改（`find -newermt` 验证 src 自 E-B 起无改动）。
- 分支在 `fork_rng(devices=[])` + CPU 种子内构造 ⇒ 不扰动全局 RNG 流，头初始化与数据序与 E-B 同 seed 对齐。

## 2. 主表：negation exact_match（1344 步 = 16ep×84，同基座同头同数据）

| 档 | 分支参数 | s42 | s43 | P2 线 0.870/0.831 |
|---|---|---|---|---|
| **B1 mlp** | 131,072 | **0.7967** | **0.8183** | ✗ / ✗ |
| **B2 block1** | 213,252 | **0.7600** | **0.7550** | ✗ / ✗ |
| **B3 block2** | 426,504 | **0.7800** | **0.8017** | ✗ / ✗ |
| frozen（本实验配对，同 eval 节奏） | 0 | 0.6883 | 0.6517 | 下限 |
| E-B r16 线性旁路 | 16,384 | 0.7267 | 0.7433 | 对照 |
| E-B 全秩线性（r128，单 seed） | 131,072 | 0.7383 | — | 等参对照 |
| E-B r32×64ep（4× 预算，单 seed） | — | 0.7967 | — | 平台对照 |
| **joint 联合档（文档值）** | 可训核 | **0.9667** | **0.9233** | 上限 |

- 配对 Δ vs 同 seed frozen：mlp **+10.8/+16.7pt**、block1 +7.2/+10.3、block2 +9.2/+15.0（两 seed 同号）；
- vs E-B r16 同预算：mlp **+7.0/+7.5pt**（两 seed 同号）；等参同预算 vs 全秩线性 s42 **+5.8pt**（单 seed）；
- 曲线（EPOCH_EVAL）：mlp 0.70→0.77→0.79→**0.80-0.82** 仍在缓升；block1 压在 0.76（ep12→16 反而降）；
  block2 0.78-0.80；三档**都远低于联合档**（差 10.5~17pt）。
- **探索臂（不参与判定）**：mlp 32ep = 0.8383/0.8533、64ep = 0.8533（s42）—— 线性旁路的 ~0.80 平台被
  非线性抬到 ~0.85，但 **s42 的 0.870 线在 4× 预算后仍差 1.7pt**（s43 的 32ep 0.8533 过了 0.831，但 P2
  要求两 seed 同时过线）⇒ 放宽到 4× 预算，P2 仍不成立；末段增速仅 +1.5pt/32ep。

## 3. 老卡 A/B/C 三条件（3 档 × 2 seed 共 6 次复核，每次四卡；Δ 全部精确为 0）

| 卡 | A/B（6 卡一致） | Δ(B−A) | 指标逐位同 | 逐样本 predict 哈希 | C 强制开闸 em 范围 | C≠A 样本数 |
|---|---|---|---|---|---|---|
| pronoun | 0.9217(s42) / 0.9600(s43) | **+0.000000** | 是×6 | **是×6** | 0.4967~0.7183 | 410~424 / 600 |
| sentiment | 0.7528 / 0.7716 | **+0.000000** | 是×6 | **是×6** | 0.3259~0.6538 | 2368~2426 / 3200 |
| relation | 0.7306 / 0.7417 | **+0.000000** | 是×6 | **是×6** | **0.0681**~0.7153 | 685~702 / 720 |
| person | 0.3183 / 0.3217 | **+0.000000** | 是×6 | **是×6** | 0.1533~0.3100 | 414~508 / 600 |

- **P1 对全部 3 档 × 2 seed 逐位成立**：Δ exact = 0、指标字典完全相等、逐样本 sha256 一致；
- **敏感性正对照 C 有效**：强制开闸老卡明显掉（最狠 relation s43 mlp 0.7417→0.0681）⇒ 检查非空；
- 交叉核对：引擎路径 negation = 0.7967/0.8183、0.7600/0.7550、0.7800/0.8017，与训练脚本评估**逐位一致**。

## 4. 空测试三条（原始输出）

**① 真实可训参数量（按 requires_grad 实数）**
```
SELFTEST_1 trainable_report(mode=bypass,branch=mlp)={... "base_core_trainable": 0, "head_trainable": 629764,
  "branch_total": 131072, "branch_trainable": 131072, "trainable_total": 760836}
SELFTEST_1 ...(branch=block1)={... "branch_total": 213252, ... "trainable_total": 843016}
SELFTEST_1 ...(branch=block2)={... "branch_total": 426504, ... "trainable_total": 1056268}
```

**② 零初值 = 冻结基线**
```
SELFTEST_2a max|Δ|(分支开@零 vs 关)=0.000e+00 max|Δ|(分支开@零 vs 无分支基座)=0.000e+00 零初值张量全零=True   # 6/6 主档
frozen 指标层：三档 × 同 seed 完全相等 → s42 0.6883 / s43 0.6517（E-B F 档 0.6933/0.6500，Δ=−0.005/+0.002，在 ±0.01 内）
诊断：frozen + --eval-every 0 重跑 = 0.6933（与 E-B F 档 cls/span/bg/em 全字段逐位相同）
      ⇒ 0.005 差 = 训练中途评估扰动轨迹（eval 节奏是协变量），非分支引入
```

**③ 训后分支真在干活**
```
SELFTEST_3a_verdict 零初值张量全部离开零=True (2/2 或 4/4) 张量发生位移=8/8~16/16   # 6/6
SELFTEST_3b 关闸后 exact：mlp 0.7967→0.3400 (−45.7pt) / 0.8183→0.3750 (−44.3pt)
                     block1 0.7600→0.4100 (−35.0) / 0.7550→0.4133 (−34.2)
                     block2 0.7800→0.3867 (−39.3) / 0.8017→0.3550 (−44.7)   # 需 ≥10pt，全过
```
（frozen 档 3a=False 是设计使然：分支冻结在零不移动；关闸 Δ=0 同理。）

## 5. 判定与机制

**判定：三档全部不过 P2（两 seed 同号）⇒ 按预注册判据，架构答案不成立。** P1 对 6 卡逐位成立、
三条空测试全过 —— 即"零损伤"这一半依旧牢固，倒的是"补得回来"这一半。

机制（实测证据）：
1. **非线性方向对但量不够**：等参同预算（131,072）mlp 0.7967 vs 全秩线性 0.7383（+5.8pt，单 seed）；
   同预算 vs r16 +7.0/+7.5pt（两 seed 同号）——"缺非线性变换"假设**部分成立**，但只把冻结-联合差的
   回收率从 12%/34% 抬到 39%/61%（s42/s43），远不到 100%；
2. **深度不是答案**：block2(426k) 仅比 block1(213k) 好 2~4.7pt，两档都**弱于参数最少的 mlp**；
   参数量与深度都不单调解释增益（混杂：B1 插 4 点、B2/B3 单点，见 §7）；
3. **平台被抬高但没被突破**：mlp 曲线 0.80→(32ep)0.84→(64ep)0.85，末段 +1.5pt/32ep，
   s42 距 0.87 仍差 1.7pt 且在减速；三档 1344 步曲线压在 0.76~0.82，与联合档仍差 10~17pt；
4. **分支确在干活且与头共适应**：关闸掉 34~46pt（比 E-B 线性的 −29.7pt 更狠）⇒ 失败不是空壳、
   不是没接到计算图，而是分支+头这套组合在该预算内就是够不到联合档。

## 6. 原始命令 / unit / 日志 / 产物

```bash
# 主套件（12 训练 + 6 老卡复核），03:44:57 → 04:25:44
systemd-run --user --unit=dtseek-core-branch --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  /bin/bash -lc 'exec bash experiments/core_branch/run_all.sh > logs/core_branch_run.log 2>&1'
# 探索臂（4 跑），04:26:10 → 04:34:01
systemd-run --user --unit=dtseek-core-branch-expl ... 'exec bash experiments/core_branch/run_expl.sh > logs/core_branch_expl.log 2>&1'
```
- 代码：`experiments/core_branch/{branch.py, train_branch_card.py, eval_old_cards_branch.py, run_all.sh, run_expl.sh}`；
- 逐跑日志：`logs/core_branch_{f,b}_{mlp,block1,block2}_s{42,43}.log`、`logs/core_branch_oldcard_*.log`、
  `logs/core_branch_x_*.log`；总驱动 `logs/core_branch_run.log` / `_expl.log`；
- 卡产物：`experiments/core_branch/cards/negation_{mlp,block1,block2}_s{42,43}.pt`（+ frozen 6 张 + 探索 4 张）；
- 老卡数据：`experiments/core_branch/old_card_check_{mlp,block1,block2}_s{42,43}.json`。

## 7. 遗留与不确定

**实测**：
- P3 代价（稳态 s/step / 峰值 MB，E-B r16 = 0.0272 / 259.5，frozen = 0.0207 / 172.3）：
  mlp 0.0268 / 275.8（8× 参数、步时持平）、block1 0.0252 / 232.4、block2 0.0303 / 291.2 ——
  block1 便宜是因为它只挂在 norm 输出，反向不必穿过上层编码器（mlp 挂 4 点须保留全图激活）；
- 训练中途评估（--eval-every 4）本身会扰动轨迹 ~0.5pt，主表 6 跑与 frozen 配对档同节奏，配对内部自洽；
- 探索臂单 seed（32ep 的 s43 除外）；全秩线性对照、r32×64ep 对照均为 E-B 单 seed。

**推断（未测）**：
- "分层深度不加分"部分受**插入点不对齐**混杂（B1=4 点，B2/B3=1 点）——多点块插入未试；
- 64ep 曲线仍缓升，0.87 能否靠预算达到**未知**（外推不可靠，E-B 线性 4× 预算只到 0.7967）；
- 零输出投影初始化可能限制块的早期梯度（proj 先动、qkv/c_fc 滞后），是否拖慢块档未单独验证；
- "联合档改写核 17%"是否可由并联分支表达仍未直接检验（本次是端到端能力判据）；
- 未试变体：多点块插入、embedding 后插入、分阶段训练（先头后分支/先分支后头）、蒸馏式关闸训练。
