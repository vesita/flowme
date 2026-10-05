# 预注册：L 臂 = 联合训练但降低基座 lr（lr_base 3e-4 → 1e-4）

## 动机（来自 dev-notes/12 §12.6 的张力）
| 路径 | 老卡 | 新卡 |
|---|---|---|
| N5 联合训练（base lr 3e-4） | relation −12.1pt | negation 0.9667 |
| R 冻结基座 + 新头 | 无可测损伤 | negation 0.6383 |

张力在于：**新卡需要基座一起学，老卡需要基座别乱改**。降低基座 lr 是这条轴上的中间点，
改动最小、最可能兼顾。**若它也不行，就说明必须走旁路适配器或扩基座。**

## 设置
`--tasks pronoun sentiment relation person negation --lr-base 1e-4 --lr-head 1e-3 --seed 42 --epochs 16`
（其余与 A′ / N5 完全一致：同数据、同步数预算、同评测集。）

## 判据（跑前写死）
- **P1 老卡**：四张老卡 `exact_match` 相对 A′ 的下降，**逐卡不超过自己的实测噪声带**
  （pronoun 2.83pt / relation 1.39pt / sentiment 0.41pt / person 0.33pt，来自 4 卡同数据换 seed 42↔43）。
- **P2 新卡**：negation `exact_match` **≥ 0.90**（对齐 N5 的 0.9667）。
- **P3 不乱开火**：negation `bg_fp` **≤ 5%**。
- 判定：
  - 三条全过 ⇒ **增量加卡流程定为「低基座 lr 的联合训练」**；
  - P1 过但 P2 不过 ⇒ 说明"少改基座"和"新卡学好"在该 lr 下不可兼得，需要继续降/升 lr 扫一条曲线，
    或直接上**旁路适配器**（新卡自带低秩通路，基座完全不动）；
  - P1 不过 ⇒ 降 lr 不足以免伤老卡，必须换机制。
- 单 seed 只作方向；定论需第二 seed。
