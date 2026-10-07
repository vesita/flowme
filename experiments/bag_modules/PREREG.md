# PREREG —— 袋项模块分解（类型 / 题元 / 槽型）× 句子特征（位置/ token 级）对**选骨架**的作用

**唯一问题**：把「候选（袋项）」拆成多个**可解释模块**（类型、题元、槽型）之后，"选骨架"能否显著提升，
**尤其在未训槽序 `adv2` 上**？

写死时间：2026-10-07 09:0x（`mtime` 早于首次训练）。**开训前不得改本文件；跑后不改**（改哪条都要在 REPORT 里如实记）。

---

## 1. 被测对象与四臂（唯一变量 = 特征表示）

基线 = `experiments/struct_supervision` 的 **B 臂**口径（同数据、同 loss `L_gen`、同配方、同 seed）：
骨架头 `skel_out = Linear(128→40)`，输入**只有**池化 `h`；袋项向量 `v_items` 只喂指派分支。
池化 `v_sent` = 逐 token 掩码均值 ⇒ **置换不变，顺序信息可证明地不存在**。

| 臂 | 句子特征 | 骨架头额外输入 | 新增可训参数 |
|---|---|---|---|
| **A** | 池化 `v_sent`（现状） | 无 | 0（= 基线，须逐位复现 struct B） |
| **U** | 池化 `v_sent` | **标签序列** `z_lab`（袋项类型/题元/槽型，按句序） | 标签嵌入 + GRU + `skel_aux` |
| **P** | **位置感知注意力池化** `v_sent'`（绕开单点均值，零初始化 ⇒ 第 0 步与 A 逐位相同） | 无 | PosAttPool |
| **U+P** | `v_sent'` | `z_lab` | 两者 |

- 四臂**共用参数初值逐位相同**（构造顺序 trunk → 占位 Linear → GenHead 不变；新模块一律**在 GenHead 之后**构造；
  `--selfcheck` 断言 trunk / `skel_out` / `item_mlp` / `slot_emb` 跨臂 `torch.equal`）。
- 新信息**只进骨架头**：`skel_logits = gen.skel_out(h) + skel_aux(z_lab)`（`skel_aux` **零初始化** ⇒ U 在第 0 步与 A 同 logits）；
  指派分支与 loss 不动，槽位指标作"没被顺手改坏"的旁证。
- 头不变、只换特征 / 只加一条旁路 ⇒ 这就是"同一模型、同数据、同步数、同 seed，只换特征表示"。

## 2. 模块来源与取值域（被测对象之一，必须交代）

取值域**沿用仓库既有规范**（`experiments/card_flow/lexicon.py` 的 `SLOT_TYPES`/`THEMES` = `dev-notes/16 §7.7`）：

| 模块 | 字段 | 取值域 | 来源（三选一） | 标注器 |
|---|---|---|---|---|
| **M1 类型** | `type` | `{名, 动, 形, 其他}` | ② 新构造规则 + ① 现有手写词性闭集做金标 | 形态标记规则（了/着/过/把/被/会/能/要/正在…→动；程度副词+形容词→形；否则名） |
| **M2 题元** | `role` | `{施事, 受事, 时, 其他}` | ② 新构造规则 | 句面邻接标记（被/把/让/给、之前/之后/的时候/在…） |
| **M3 槽型** | `cls` | `{Q,N,M,T,O}`（疑>否>数>时>他） | ① **现有规则**（`struct_supervision/build_data.py::slot_class`，只读 import） | 逐字复用，断言与 adv2 构建口径逐位相同 |
| M4 来源 | — | — | **不做** | train/test 行**没有**语料来源字段（只有 adv 行构造期的 `file`，`to_rows` 已丢弃）⇒ 无来源可报，如实记为"不可构造" |
| 现有卡（①） | — | — | **不作标注器** | `compose_ops/artifacts/cards/*.pt` 是**解码器**（`dtseek.task_card.v1`），不是词典，无法直接给袋项打标；且 `card_flow/report.md:93` 实测其切片与自身词典对齐率 relation **0/152**、sentiment **0/95** ⇒ 引为"不采用①卡作标注器"的实测依据 |

**硬约束（防作弊）**：标注器**只读句面 + span**，**禁止**调用 `match_sentence` / 读 `skel_id` / 读 `skel` 模板 / 读 `assign`。
`skel` 模板**只用于事后算可靠性金标**，不进模型输入。标签是输入的确定函数 ⇒ **不提供输入之外的新信息，只提供可访问性/归纳偏置**（REPORT 必须写这句）。

## 3. B5 模块可靠性口径（跑前写死，逐模块报）

每个模块报四个数：**覆盖率 / 对金标准确率 / 与独立规则一致率 / 类别分布**。

- **M2 题元**：金标 = **结构金标**（由 gold `skel` 模板的槽位邻接字面 + 声明的槽位→题元映射推出，与句面规则是两条独立代码路径）；
  只在**金标确定（非"其他"）**的袋项上算准确率，同时报确定性覆盖率。
- **M1 类型**：金标 = **① 现有手写词性闭集** `src/dtseek/tasks/builtin/cloze_fill/dataset.py::POS_LEXICON`（人工判定，名词/动词/形容词三类）在 gap 内命中的子集；报该子集准确率 + 词典命中覆盖率。**若词典覆盖率 < 5%** ⇒ 按下分支补做：**人工抽样 n=100（seed=20261007 确定性抽样，先标注后看规则输出）**，以人工金标为准。
- **M3 槽型**：与 adv2 构建用的 `slot_class` **同一函数**（只读 import），一致率断言 = 1.0；报非 O 覆盖率。
- **门槛建议（B5）**：某模块**可用** ⟺ 准确率 ≥ **0.70** ∧ 覆盖率 ≥ **0.10**；否则 REPORT 如实写"该模块是噪声 / 低覆盖"，
  并且**该臂若只靠被判噪声的模块**，判定降级为"证据不足"。

## 4. 判据（跑前写死；B1 为主）

| # | 判据 | 门槛 |
|---|---|---|
| **B1（主）** | 某臂 **adv2 骨架** > A | Δ = 该臂 − A，**Δ/SE_pair > 2**，**seed 42/43 两 seed 同号**（SE_pair = 逐行 0/1 差的配对标准误） |
| **B2（副·防换）** | 同臂 **test 骨架不退化** | ≥ A − 2×SE_ind（test n=2500 ⇒ SE=0.0096，2SE=0.0192）。**不许拿表内能力换迁移** |
| **B3（门禁）** | 随机标签对照 | (a) `U-rand`（训练+评测标签跨行 randperm）的 adv2/test **增量对 A 消失**：Δ ≤ 2×SE_pair，两 seed；(b) **只读标签通道**（`skel_aux(z)` 单独 argmax，去掉 `h`）在随机标签下**掉回多数类/盲猜**：test ≤ 多数类 **0.4169** + 2SE，adv2 ≤ max(多数类 **0.288**, max_naive **0.182**) + 2SE。**(b) 是字面门禁**，(a) 防"整臂本来就有 0.64"掩盖 |
| **B4（必报）** | `max_naive` 并列 | 每格报 **卡 − max_naive**：test **0.5329**、adv2 **0.182**、adv1 **0**（来自 `stats.json`/`stats_adv.json`，与 struct 同口径；附注 test 自身拟合值 0.5208）。**低于即如实写"未超过免费规则"** |
| **B5** | 模块可靠性 | §3 逐模块报；不达标即写"模块是噪声" |

**判定（三选一，不许硬选）**
- **可解释分解有效** = B1 ∧ B2 ∧ B3 ∧ **B4 该臂 test 与 adv2 均 > max_naive** ∧ **B5 该臂用到的模块全部可用**。
- **无效** = B1 全臂未过（Δ ≤ 2SE 或两 seed 异号）**且** B2/B3/B5 通过（说明是实验有效、结论为负）。
- **证据不足** = B3 或 B5 未过，或两 seed 异号/ |Δ| ≤ 2SE，或 B2 失败（拿表内换迁移）。
- **U 与 P 效果不同 ⇒ 按两层结论分开写**（"加标签"与"加位置"是两个杠杆），不合并成一句。
- **不许调参刷过**：配方写死（§5）；只许修 bug（修了必须记进 REPORT）。

**机制（跑前定的证据，不看结果挑）**：① `skel_aux(z)` 单独（去 `h`）的骨架准确率 = 标签通道上限；
② **纯规则基线**：train 上按"标签序列"查多数骨架（未见键 → 全局多数），在 test/adv2 上的准确率；
③ **杀模块消融**：U 分别只留 `type` / 只留 `role` / 只留 `cls`（三臂各 2 seed），看 Δ 掉在哪；
④ 对已训 U 做**评测期标签行内打乱**，看 adv2/test 掉多少（模型对标签的依赖度）。

## 5. 配方（写死）

数据 = `two_channel_head/data/{train,test}.jsonl` + `struct_supervision/data/{adv1,adv2}.jsonl` **只读复用，不重建、不扫语料**
（语料口径沿用：文件级汉字占比 ≥0.6 保留 14/18，丢 code_alpaca 0.0004 / gsm8k 0.0585 / coig_math 0.5418 / qwen3 0.5112，
即 `dev-notes/19` 明令禁用的两个代码/英文文件已被排除；**本实验不取文件、不取 fs 顺序**）。
n_train 8000 / test 2500 / adv1 1220 / adv2 1000。

- 1800 步、batch 64、AdamW lr 1e-3 wd 1e-4、cosine、grad clip 1.0、seed **42/43**；
- DataLoader `generator=manual_seed(seed)`（与 struct 逐位同序）；loss = `L_gen = CE(骨架)+CE(指派)`；
- 核 1,688,460 全程 `requires_grad=False` + `eval()`（`freeze_report` 断言）；
- 新模块超参（写死，不调）：标签嵌入 8×3、相对位置桶 4、GRU hidden 32、`skel_aux Linear(32→40, bias=False)`；
  PosAttPool K=4、位置桶 8、查询零初始化门控、混合权重初值 1/4；
- 冒烟：`--steps 60` 四臂全通后才开全量；全量用 `systemd-run --user --unit=dtseek-bagmod --collect … PYTHONUNBUFFERED=1`；
  **绝不 `setsid nohup &`，绝不动他人进程**（开跑前查 `systemctl --user list-units 'dtseek*'`）。
- 允许写：`experiments/bag_modules/`、`logs/`、`/tmp`；其余只读；**禁止 git commit/stash/checkout/restore/clean**；
  `two_channel_head/model.py` 用 importlib **只读**加载，不改其源码。

## 6. 输出

中文 Markdown ≤130 行：四臂特征差异（字段/取值域/来源/逐模块可靠性）、主表（四臂 × 3 split × 3 指标 × 2 seed + SE + 卡−max_naive + 随机标签对照）、
B1–B5 逐条 + 判定三选一 + 机制、命令/unit/日志/产物路径、遗留与不确定（实测/推断分开）。
