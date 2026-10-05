# 预注册：联合热身 W=12 → 冻结（C″ 臂）

## 动机
C 臂（W=4）在第 10.5 节被判定失败：min-over-cards exact_match 31.5% < A 臂 31.8%，
sentiment 首切片 47.34% 与「冻结随机基座」负对照 B（47.68%）几乎无差别。
机制归因：第 4 轮结束时基座远未收敛（person loss 仍 44.0），冻结发生在基座没成型的时候。
**本臂检验该归因**：若把热身拉到 12 epoch（A 臂全程 16 epoch 的 3/4），冻结是否能不再等价于负对照。

## 对照
- A 臂（已有）：全参联合 16 epoch，`checkpoints/multitask_v2_metrics.json`。
- B 臂（已有，负对照）：冻结随机基座。
- C 臂（已有，失败）：W=4。
- **C″ 臂（本臂）**：`--freeze-after-warmup 12`，同 seed=42、同 16 epoch、同数据。

## 判据（跑前写死，事后不许改）
- P1：C″ 的四卡 min-over-cards（exact_match） **≥ A 臂**（A = 31.83%）。
- P2：C″ sentiment 首切片类别准确率 **≥ 85%**。
- P3：C″ 所有卡背景误报率 **≤ 2%**。
- P4：C″ relation 对完整命中率 ≥ **90%**。
- P5：墙钟时间与 A 臂（590.8s）的比值。
- 判定规则：P1–P4 只要有一条不过，**如实判 C″ 失败**，并据此把结论收紧为
  「从零联合热身再硬冻结在 16 epoch 预算内不成立」，不得改判据。

## 口径
指标取各任务验证集 split（与 A/B/C 同口径）；`exact_match` 来自 `evaluate_task`，不看首切片。

---

# 预注册：从收敛基座热启动 → 冻结 → 各头独训（C′ 臂）

## 动机
C（W=4）与 C″（W=12）都失败，且 C″ 全面低于 A ⇒ 「从零联合热身到中途再冻结」在 16 epoch 预算内
**从未赢过全程联合**。但 `dev-notes/10` §1 的实证协议不是这个：它是
**基座先被四卡联合训练到收敛 → 再冻结 → 各头独训**（`--init-from` + `--freeze-base`）。
这是该笔记唯一没被本轮测到的档位，本臂关掉它。

## 设置
`--tasks pronoun sentiment relation person --freeze-base --init-from checkpoints/multitask_v2_dtseek.pt
 --epochs 16 --seed 42`（基座从一开始就冻结、且来自 A 臂已收敛的基座；各头从 A 臂同名头热启动后独训 16 epoch）。

## 判据（跑前写死）
- P1：四卡 min-over-cards（exact_match） **> A 臂 0.3183**（严格大于，因为本臂比 A 多训 16 epoch 头）。
- P2：sentiment `exact_match` **> A 的 0.7528**。
- P3：relation `exact_match` **> A 的 0.9500**。
- P4：所有卡背景误报率 ≤ 2%。
- 判定规则：若只持平或更差，则结论是「冻结基座 + 加训头不能超过全程联合」，
  dev-notes/10 的增益应理解为「在**当年那个被四路梯度拉扯过的**基座上解耦各头」，
  而不是「冻结本身更好」。不得改判据。
