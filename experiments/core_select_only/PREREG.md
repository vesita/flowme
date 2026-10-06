# core_select_only —— 预注册：那个增益是「select 梯度进核」，还是「老任务锚定」参与后才可能的？

> 本文件在**任何训练开始之前**写死（mtime 见 REPORT 与 `run_all.sh` 首行打印），跑后不改门槛。
> 目录只写 `experiments/core_select_only/`、`logs/`、`/tmp`；其余目录**只读**。

## 1. 唯一要回答的问题与两臂

`select_semantic_joint` 的主臂（核可训 + 4 张老卡头冻结 + 每步同时跑 4 个老任务批次）把
`heldout_pair` 从 49.52/51.04 拉到 59.36/63.20。两份 probe（`core_grad_probe` / `core_update_probe`）
表明：核上 select 的**原始**梯度被老任务淹没（r = 0.0019/0.0134/0.0035），但 **Adam 更新量级同量级**
（比 0.973/1.085/0.865），**方向仍由老任务主导**（cos(both,old)=0.970/0.996/1.000）。
**两次测量指向同一个缺口**：「只让核学 select、不给老任务」那一档没跑。

**唯一变量 = 每步是否同时跑老任务批次（老卡头是否挂载并参与计算）。**

| 臂 | 每步数据 | 核 | 老卡头 |
|---|---|---|---|
| `joint`（**核对臂，必须零误差复现**） | select batch + **4 个老任务 batch** | 可训 | **挂载并冻结**，参与每步计算 |
| `selonly`（**主臂**） | **只有 select batch** | 可训 | **训练期间不挂载**（不参与任何计算）；训练**结束后**只读加载用于 T5 评测（`torch.no_grad`，不算梯度） |

其余逐项同口径（= `select_semantic_joint` PREREG §2/§3）：
masked mean-pool、**1800 步**、batch **64**、lr_core **3e-4** / lr_head **1e-3**、
AdamW wd **1e-4**、cosine `T_max=1800`、clip **1.0**、seed **42/43**、核全程 `eval()`（dropout=0）、
live 编码（核可训 ⇒ 不读 token 缓存）、同一份数据与评测口径。

**数据只读复用** `experiments/select_rerank/data/clean/`（md5 = `6b1e6fb1…` / `d3d597af…` / `2a5ca5f4…`）
与 `experiments/select_pool/cache/`（token 级缓存，供 T0 对照读），**一个字节都不许改**；
跑前跑后 `md5sum` + `find -newermt <本文件 mtime>` 自证。

## 2. 可预期的代价（来自上游代理的**预告**，不是本档结论）

`select_semantic_joint` PREREG 预先写明：不给老任务的代价**预期落在 R5（老卡退化）**。
⇒ 本档**不是「更好的方案」，而是机制分解**：若 `selonly` 没有增益，
则「让核学 select」必须与「保住老卡」绑在一起 —— 这是**设计结论**，不是失败。

## 3. 判据（跑前写死；SE = sqrt(0.25/1250) = 1.414pt，2×SE = 2.828pt，n=1250）

| # | 判据 | 门槛（写死） |
|---|---|---|
| **T0（门禁）** | `joint` 核对臂**零误差复现** `select_semantic_joint` 主臂 | `heldout_pair` 与 `shifted_pos` 逐 seed **Δ = 0.00**（不是 ≤2×SE）；参照值 59.36/63.20、99.60/99.60 |
| **T1（主）** | `selonly` 的 `heldout_pair` **vs 冻结核基线**（49.52 / 51.04） | 逐 seed **Δ > 2.828pt**，且两 seed **同号（都为正且都过门槛）** |
| **T2** | `selonly` 的 `heldout_pair` **vs `joint`** | 报差值与 SE；「噪声内」= 逐 seed **\|Δ\| ≤ 2.828pt**。**这是「老任务是否必需」的判据** |
| **T3** | `selonly` 的 `shifted_pos` 不塌 | 逐 seed ≥ 基线 − 2.828pt（即 ≥ 89.73 / ≥ 89.65）—— 防「拿表内记忆换」 |
| **T4（门禁）** | 随机标签对照（**`selonly`**，两 seed） | `heldout_pair` 与 `shifted_pos` 均 **≤ 52.83**；且 `randlabel_n_diff > 0`、train 标签计数仍 1:1（真执行） |
| **T5（代价，必报）** | `selonly` 训练后**四张老卡**（共享核 + 各自冻结头）的 Δ | **预期会退化**（本档预测代价）；逐卡报 Δ 与各自噪声带 BAND = pronoun 0.0283 / sentiment 0.0041 / relation 0.0139 / person 0.0033；**不许略过**，不作为门禁 |
| **T6** | 核漂移 `‖core−base‖/‖base‖`、稳态步时、峰值显存 | 数值报告（两臂都报） |

**判定三选一（边界情形也跑前写死）**：
- **① 增益来自 select 梯度进核** = T0 ∧ T4 过，且 **T1 过**，且 **T2 过**（`selonly` 与 `joint` 差距在 2×SE 内）。
- **② 老任务锚定是必需的** = T0 ∧ T4 过，且 **T1 不过**，且 `selonly` 的 `heldout_pair` 逐 seed **≈50%**（\|acc−50\| ≤ 2.828pt）。
- **③ 证据不足** = 门禁（T0/T4）不过；**或 T1 过而 T2 不过**（⇒ 注明「select 梯度有增益，但 `joint` 更高：两机制都有份」，
  仍归 ③，因为既不满足「老任务非必需」也不满足「增益回到 50%」）；**或 T1 不过但 `selonly` 也不 ≈50%**。
- **不许硬选、不许调参刷过**；**≥2 seed**；步数/lr/老任务权重本档**不扫**。

## 4. 空测试（训练前，`selftest.py`，全实测）

1. 两臂**可训参数量**与**冻结实况**（逐参数打印 `requires_grad`）：
   `joint` = 1,688,460（核）+ 164,353（打分头）= 1,852,813，老卡头逐参数 `False`；
   `selonly` = 1,852,813，**老卡头字典为空**。
2. `selonly` **确实没挂载老卡头**：训练期间打印 `named_modules()` 的顶层模块名（只有 `encoder`/`head` 等本模块部件）、
   挂载登记表 `{heads: {}, old_loaders: {}, old_forward_count: 0}`，并 `assert`。
3. **多数类基线**：split 0.5000（train/test/adv）、两 ctype 0.5016（627:623）。
4. **随机标签对照真执行**：randperm 后计数仍 1:1、与真标签不同条数 > 0。
5. **核对臂代码同源实测**：本目录 `train_core.py --arm joint --steps 20` vs 只读 import 的
   `select_semantic_joint/train_sem.py --arm joint --steps 20`（后者内存中把输出路径重指到本目录，**不写他人目录**）
   ⇒ 打分头/核权重 max\|Δ\| 与 `loss_first` 差值（只报，门槛 = 1e-4 量级的 fp32 噪声；**T0 才是硬门禁**）。
6. T4 基线文件就位（读 `select_semantic_joint/results/r4_baseline_s{42,43}.json`，sha256 非空）。

## 5. 运行方式

`systemd-run --user --unit=dtseek-cso-joint` / `dtseek-cso-selonly`（`--collect` + `PYTHONUNBUFFERED=1`），
**绝不用 `setsid nohup &`**；开训前查 `ps` 与 `systemctl --user is-active 'dtseek*'`，忙则排队，**不杀他人进程**。
产物：`experiments/core_select_only/results/*.json`、`weights/*.pt`、`logs/core_select_only_*.log`。
