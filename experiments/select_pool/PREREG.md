# select_pool 聚合消融（experiments/select_pool）—— 预注册判据

> **本文在首次训练之前写死**（mtime 必须早于任何 train 进程启动时刻），跑完不改；结果填进 `REPORT.md`。
> 口径：「实测」= 本目录跑出来的数字；「推断」= 由数字推出；「口径判断」= 构造时人为规定的标签归属。

## 0. 唯一问题

**B 族（`experiments/select_rerank`）那堵墙是「池化抹掉了信号」，还是「表示里本来就没有语义等价」？**

B 族实测（`select_rerank/REPORT.md` §3.1）：`shifted_pos`（表内词、只换位置）**92.56 / 92.48%**，
而 `heldout_pair`（**留出替换对 = 换说法**）**49.52 / 51.04% ≈ 盲猜**。
⇒ 打分头学到的是「表内替换对的固定偏好 + 位置不变」，不迁移到新说法。
B 族是 **masked mean-pool**（8~24 字片段压成一个 128 维向量），`heldout_pair` 里 **2 字之差**（如 `高兴`：喜悦 vs 痛苦）
在片段里占比极小 ⇒ 疑被稀释（`select_rerank/REPORT.md` §6.4 标为**未做消融的推断**）。

- **墙在池化** = 换一种**让 2 字之差与上下文交互**的聚合方式，`heldout_pair` 能过 Q1；
- **墙在表示** = 四种聚合**全都**过不了 Q1（且口径 Q4 与对照 Q3 都成立）。
- **不许硬选、不许调参刷过**：判据跑前写死，换配方必须另开 PREREG。

## 1. 数据（**只读复用**，不改不重建）

- **直接读** `experiments/select_rerank/data/`（**只读**：本实验不得写该目录任何文件）——
  同一份已验证数据（15 对集合重叠全 0、三条零训练基线均 0.5000、无捷径）才能与 B 族直接对比。
- **只跑 clean（N）臂**：B 族引的 49.52/51.04 与 92.56/92.48 都是 N 臂 adv 的两个 ctype；
  `heldout_pair` 只有 N 臂口径与「同义 vs 反义」一致。S 臂不跑（它的 adv 同构造但结论不引它）。
- 指纹 = train/test/adv 三个 jsonl 的 md5（读文件算，不写）；行序**逐行复刻** `select_rerank/train_rerank.py::load_rows`
  （train → test → adv），保证与 B 侧行序一致。
- **口径同一性硬校验（Q4a，训练前必过）**：
  1) 本实验算出的标签数组与 `select_rerank/cache/*clean*.pt` 的 `labels` **逐元素相等**；
  2) 本实验**由 token 级缓存现算的 masked mean-pool 向量**与该缓存的 `v_ctx / v_cand` **逐元素差 ≤ 1e-6**。
  任一不过 ⇒ 口径不一致，**先查口径、不出结论**。

## 2. 四臂 + 参数量对齐对照臂（其余逐项同 B 族口径）

冻结核 `checkpoints/base_encoder.pt`（`NanoDocEncoder`，hidden 128）**全程冻结、不发射锚点**；
context 编一次、每个候选各编一次（与 B 族同）；差别**只在「token 级隐藏 → 向量」的聚合方式**。
打分头、训练步数、优化器、seed 与 B 族**逐项相同**。

| 臂 | 聚合方式（masked，padding 一律排除） | 聚合层可训参数 |
|---|---|---|
| **A（基线）** | **mean-pool** = `Σ h·m / Σ m`（与 B 族逐位同式） | 0 |
| **B** | **max-pool** = 有效位逐维取最大 | 0 |
| **C** | **可学 attention-pool**：可学 query `q_ctx, q_cand ∈ R^128`，`a = softmax(q·h / √128)`（各侧独立 query），`v = Σ a·h` | **256** |
| **D** | **cross-attention**：候选的每个 token 对**上下文** token 做单头注意力（raw dot × 可学温度 `t=exp(θ)`，`θ` 初值 `−0.5·ln128`；无投影矩阵）再按候选 mask mean-pool；**ctx 向量 = A 的 mean-pool（ctx 侧不变）** ⇒ 唯一改动 = 让 2 字之差与上下文交互 | **1** |
| **A+（Q5 对照）** | mean-pool + 可学逐维缩放 `g_ctx, g_cand ∈ R^128`（初值全 1）——**与臂 C 可训参数量逐个对齐（256）** | **256** |

- **候选等变自检（每臂必过）**：交换两候选 ⇒ 两个 logit 同步交换（allclose ≤1e-6）⇒ 下标不是特征。
- **冻结实况（每跑必打印并写进结果）**：核 `requires_grad` 全 False + `eval()`；可训参数 = 打分头 + 本臂聚合层，逐臂报数。
- **打分头与 B 族完全相同**：`MLP([ctx; cand; ctx⊙cand; |ctx−cand|])`，hidden 256→128→1，softmax over K=2 + CE。
- 参数量跑前算死：核 **1,688,460**（冻结）+ 头 **164,353** + 聚合层（A/B/D=0、C=256、A+=256）
  ⇒ **四臂可训参数差异 ≤ 0.16%**（Q5 要报的数，先摆在这里）。

## 3. 训练与评测口径（与 B 族逐项相同，写死）

- **1800 步**、batch **64**、lr **1e-3**、AdamW(wd 1e-4)、cosine、grad clip 1.0、**seed 42 / 43**
  （= `select_rerank/train_rerank.py::STEPS/BATCH/LR/WD/CLIP`，同配方同 seed 可直接对比）。
- 核冻结 ⇒ token 级隐藏可缓存（键 = 数据 md5 + SPEC），**一次编码多 seed 复用**；缓存只写本目录 `cache/`。
- **随机标签对照（Q3 必做）**：train 标签 `randperm`（保持 1:1，seed 固定 `seed*1000+7`）后同配方重训，
  test/adv 仍用真标签评测；**五臂 seed42 全跑**，过 Q1 的臂**补 seed43**。
- 评测 = 训练结束全量前向（train 8000 / test 2500 / adv 2500），adv **按 `ctype` 拆**（`heldout_pair` / `shifted_pos` 各 1250）。
- 空测试自检（训练脚本内断言）：核冻结实况、可训参数量与聚合层参数对得上、各 eval split 的 n 与
  多数类基线（1:1 ⇒ 0.5000）、logits 非常数（方差 > 0）、randlabel 确实改变了 train 标签且计数仍 1:1。

## 4. 判据（跑前写死；n=1250 ⇒ SE = sqrt(0.25/1250) = **1.414pt**，**2×SE = 2.83pt**）

| # | 判据 | 门槛 |
|---|---|---|
| **Q1（主判据）** | 某臂 `heldout_pair` 准确率 **> 臂 A（mean-pool）**，**逐 seed** 算 Δ = acc_臂 − acc_A | **Δ > 2×SE = 2.83pt**，**两 seed 同号（都过）**；报 SE、Δ、Δ/SE |
| **Q2（副）** | 过 Q1 的臂 `shifted_pos` 不显著低于臂 A | **acc_臂 ≥ acc_A − 2×SE = acc_A − 2.83pt**，两 seed 都满足 |
| **Q3** | 随机标签对照（臂 A 与过 Q1 的臂，seed42；其余臂附报） | `heldout_pair` 与 `shifted_pos` **均 ≤ 50% + 2×SE = 52.83pt**；否则 Q1 作废 ⇒ 判「证据不足」 |
| **Q4** | **臂 A 复现 B 族** | `heldout_pair`：s42 与 **49.52**、s43 与 **51.04** 各差 **≤ 2×SE = 2.83pt**；另报 `shifted_pos` vs 92.56/92.48（只报不判）与 Q4a 口径同一性 |
| **Q5（必报）** | 各臂**参数量**差异 | 逐臂报可训参数量；**A+ = 与 C 参数量逐个对齐的对照臂**（mean-pool + 256 参数）。若某过 Q1 臂可训参数 > 臂 A 的 **1.10 倍**（跑前算死不会发生），再补跑该臂的参数量对齐版本 |

- 附报（**只报不判**）：过 Q1 臂自身 vs 50% 的余量/SE；臂 A 的 test/train；Q3 全臂随机标签数字。
- **判定三选一**：
  - **① 墙在池化** = 存在臂过 **Q1 ∧ Q2 ∧ Q3 ∧ Q4**（两 seed）；
  - **② 墙在表示** = **没有任何臂**过 Q1，且 **Q4、Q3 都成立**（即口径对、判据没失效）；
  - **③ 证据不足** = 其余（Q4 不过 / Q3 不过 / 某臂只在一个 seed 过 / 缺 seed / 自检失败）。
- **≥2 seed 才下结论；不调参刷过**：判 ②/③ 就如实报。所有超参（聚合结构、温度初值、步数、lr）**跑前定死，跑后不改**。

## 5. 写入范围与纪律

只写 `experiments/select_pool/`、`logs/select_pool_*`、`/tmp`；
**不改** `experiments/select_rerank/`（含其 `data/` 与 `PREREG`，**只读**）、`src/`、`training/`、`tests/`、`dev-notes/`、其它 `experiments/`；
不 `git commit/stash/checkout/restore/clean`；
长跑一律 `systemd-run --user --unit=... --setenv=PYTHONUNBUFFERED=1`，**绝不用 `setsid nohup &`**；
开训前查 `ps` 与 `systemctl --user is-active 'dtseek*'`，忙则排队，**绝不杀他人进程**；
单次训练 ≤ 几分钟量级（只训聚合层 + 打分头）。
