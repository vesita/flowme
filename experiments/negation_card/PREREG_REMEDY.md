# 预注册：R 臂 = 冻结收敛基座 + 增量加否定卡

## 假设（来自 dev-notes/12 §12）
「直接联合加第 5 张卡」会伤老卡：seed 42 上 relation −12.1pt、sentiment −4.3pt，
而退化位置正是 E3 预测的强共线对（sentiment↔relation）。
§10.9 又实测「从零训练时冻结基座更差」。两者不矛盾，因为场景不同：
- 从零冻结 = 让任务头去读**随机**基座 ⇒ 差；
- **增量冻结 = 让任务头读已收敛基座**，且新卡无法改写老卡依赖的表征 ⇒ 应该能守住。

## 设置
R 臂：`--tasks pronoun sentiment relation person negation --freeze-base --init-from checkpoints/arm_aprime_dtseek.pt
      --seed 42 --epochs 16`
- 基座 = A′（四卡、修复后数据集、seed 42）已收敛的基座，**全程冻结**；
- pronoun/sentiment/relation/person 四头从 A′ **同名热启动**；negation 头为新增（随机初始化）。

## 对照
- A′（四卡，同 seed，联合训练）＝ 老卡的上界参考；
- N5（五卡，同 seed，联合训练）＝ 已实测会伤老卡的那一臂。

## 判据（跑前写死）
- **P1（守住老卡）**：R 的四张老卡 `exact_match` 相对 A′ 下降 **≤ 2.0pt**。
  对照：N5 臂在同一批卡上掉了最多 12.1pt。若 P1 过 ⇒ 冻结确实保护了老卡。
- **P2（新卡仍可用）**：R 的 `negation` `exact_match` **≥ 0.90**。
- **P3（不乱开火）**：R 的 `negation` `bg_fp` **≤ 5%**（该卡欠训时实测会到 20.2%，故必须一并看）。
- 判定规则：
  - P1 且 P2 且 P3 全过 ⇒ **增量加卡流程定为「冻结收敛基座 + 加新头」**；
  - P1 过但 P2/P3 不过 ⇒ 冻结能保护老卡，但新头在冻结表征上学不够，需要换新头的训练配方；
  - P1 不过 ⇒ 冻结也救不了，容量问题只能靠**扩基座**解决。
- 单 seed 只作方向；定论需第二 seed 配对。
