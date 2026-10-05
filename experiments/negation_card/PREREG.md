# 预注册：否定卡加卡的净效果（5 卡 vs 4 卡对照）

## 为什么用这些对照臂
- **A′**（`checkpoints/arm_aprime_metrics.json`，seed 42，四卡：pronoun/sentiment/relation/person）
  是**公平基线**：它与旧 A 臂同配置，只是 relation 用修复后的数据集。
- 旧 A 臂（`multitask_v2_metrics.json`）的 relation 用的是**被英文代码污染的旧数据**，直接比会被混淆，弃用。

## 判据（跑前写死）
- 逐卡比 `exact_match`。**任一原卡相对 A′ 下降 > 2.0pt** ⇒ 判「触到容量/干扰墙」并指出是哪张卡。
- **但单 seed 不可单独下结论**：已实测同数据同配置重跑的轨道噪声就有 ~2pt（dev-notes/12 §11.2）。
  因此本文件先出 seed 42 的**差值**，随后必须补 seed 43 的 4 卡/5 卡配对，只有**两个 seed 同向**才下结论。
- 新卡 `negation` 自己的指标单独报，不参与上面这条判定。
