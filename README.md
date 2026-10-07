# flowme

**模因空间的卡片框架** —— 三类卡（转译 / 模因 / 转译）+ 黑板调度。

> 名字来历：`flow` + `meme` —— **模因在卡间流动**（`flowme` = 「让模因流向我」）。
> ⚠️ 已知同名/近名项目：FlowMe（中文 · 分布式链路追踪）、`flowmeo`、`flowmepy`（见 `dev-notes/21 §29` 的命名记录）。
> 框架术语仍叫**模因**（meme），项目名是 `flowme`。

---

## 1. 它是什么

| 卡类 | 输入 | 输出 | 职责 |
|---|---|---|---|
| **输入卡**（转译卡）| 外部表示（文本/图/结构）| **模因张量** | 把外部表示变成模因 |
| **思维卡**（模因卡）| **模因张量** | **模因张量** | 模因级推导；**IO 明确的小型网络** |
| **输出卡**（转译卡）| **模因张量** | 特定格式可读信息 | 把模因变回外部表示 |

**硬约束**：**思维区内只流动张量**（不许旁路、不许序列化成文本或标量）。

**模因** = 网络的 **I/O 张量**（不是隐层）；它**不是被设计的，而是由卡片组合涌现**的"神经网络语言"。
**规范与涌现分开**：形状规范（硬，`[n,d]` 变长有序 + mask）/ 语义规范（弱，训练涌现）/ 对齐规范（可判，IO 检查 + 读出探针）。

**设计全文见 [`MEME_FRAMEWORK.md`](MEME_FRAMEWORK.md)** —— 它是本框架的**唯一真相源**（三层规范 / 黑板+手写调度 / 四层度量 L0–L5 / 四阶段 / 经验 E1–E17）。

---

## 2. 当前结论（**全部实测，含限定**）

### 2.1 成立的

| 结论 | 证据 |
|---|---|
| **★ 卡有结构价值，不是"只是多了参数"** | `stages/07_card_vs_params.py`：**参数对齐到 0.010%** 后，带卡 B − 无卡加宽 A0-wide = **+2.97pp acc**（macroF1 −2.74pp±0.06，2 seed 同号 >2SE）；而**纯加等量参数买到 ≈0**（A0-wide − A0 = **−0.42pp**，2 seed 一正一负）|
| **增益是训练量的函数** | `stages/06_steps_attribution.py`：B−A0 = **−0.49 → −0.06 → +0.34 → +1.05pp**（500/1000/2000/4000 步，单调上升；4000 步 2 seed 同号 3.4SE）⇒ **此前"卡没用"的结论部分受训练量限制** |
| **串联优于并行** | `stages/04_true_bypass.py`：B − A1 = +0.70pp（2 seed 同号）⇒ **主变量不是"用不用卡"，而是"串联 vs 并行"** |
| **A0 是"任务是否需要卡"的检验器** | `stages/05_difficulty_sweep.py`：A0（从头不带卡训练）高 ⇒ 任务不需要卡 |

### 2.2 未成立 / 受限（**必须一起读**）

| 项 | 证据 |
|---|---|
| **简单任务上"思维卡是装饰"** | `stages/01_recon_fidelity.py`：P5 恒等消融**逐位相同**（`.0001` = 正常），且 M 臂触到 copy 地板 `0.0` |
| **"删掉卡就崩"是 off-distribution 假象** | `stages/04_true_bypass.py`：**A0（从头不带卡）= 0.82** ≫ 地板 0.53 ⇒ 训练后删卡的暴跌是"head 没见过未过卡的模因"。⇒ **纪律 E17：消融必须 retrain-without，不是 post-hoc removal** |
| **瓶颈（`d_mid<d`）无增益** | `stages/04_true_bypass.py`：C−B 2 seed **反号** |
| **离散中介（T 臂）结论作废** | `stages/01_recon_fidelity.py`：`argmax` 不可导 ⇒ 上游卡 `grad=None`（冻结随机权重）⇒ **测的是"训练了一半 vs 训练完整"**。⇒ **纪律 E16：离散中介必须 STE/Gumbel** |
| **两次实现数值不稳** | B−A0：`stages/06` 报 **+1.05pp**（CPU）vs `stages/07` 报 **+2.55pp**（GPU，自写实现）⇒ **方向一致、幅度未定** |
| **幅度小** | 全部结论在 **+1~3pp** 量级，**非数量级差别** |

### 2.3 度量锚点（怎么看一次实验）

```
1. 先看地板       → 卡超没超「不用理解」的分（majority / 计数规则）
2. 再看恒等消融   → 卡是不是在干活（不看绝对分）
3. 再看 CE / F1   → 干得多好
4. 最后看 Δ ± SE  → 这个改动是真有用还是碰巧（|Δ|>2SE 且 2 seed 同号）
5. 涉及筛选规则   → 补看「覆盖率」与「空集比例」
```

---

## 3. 怎么跑

torch 装在仓内 `.venv`（`uv` 管理）；**根目录没有 torch**，所以用 `--no-sync` 复用：

```bash
cd /home/vesita/coding/my/flowme
CUDA_VISIBLE_DEVICES="" uv run --no-sync python stages/07_card_vs_params.py   # CPU
uv run --no-sync python stages/07_card_vs_params.py                           # 有 GPU 时
```

⚠️ **多进程请用 `spawn`** —— `fork` + torch 会 0% CPU 死锁（已实测，见 `stages/06_steps_attribution.py`）。
⚠️ **长跑用 `systemd-run --user --unit=<名>`**，不要 `setsid nohup &`。
⚠️ **若再次移动仓库根** ⇒ `.venv` 里 23 个脚本的 shebang 与 `pyvenv.cfg` 仍指向旧路径，
  于是 `pytest`/`torchrun` 等 console script 会报 `Failed to spawn`（**代码本身没坏**）。
  修法（`.venv` 未入库，安全）：
  `grep -Il "旧绝对路径" .venv/bin/* .venv/pyvenv.cfg | xargs sed -i 's|旧绝对路径|新绝对路径|g'`

---

## 4. 目录

| 路径 | 是什么 |
|---|---|
| **`MEME_FRAMEWORK.md`** | **本框架的设计（唯一真相源）**：三层规范 / 黑板调度 / L0–L5 度量 / 四阶段 / **E1–E17** |
| **`stages/`** | 主线实验，**按"回答什么问题"命名**（`NN_<问题>.py`）|
| `dev-notes/` | 经验与裁定的家：**`21-口径裁定与新增纪律`**（**D1–D15** + 逐条裁定 + §29 效力表）|
| `methodology/` | 纪律与测量方法的家 |
| **`LEGACY.md`** | **旧框架**（DTSeek 决策模型引擎）的原 README |
| `src/` `tests/` `experiments/` `checkpoints/` `scripts/` | **旧框架**的代码/测试/实验/权重 —— **原地保留，只作「证据与经验来源」**（见 §5）|

---

## 5. ⚠️ 遗留（LEGACY）：`src/` `tests/` `experiments/` `checkpoints/`

**这些属于已被放弃的旧框架**（`LEGACY.md`）。它们**原地保留**，理由有二：

1. **它们是证据** —— `methodology/findings.md` 的 17 条里 **12 条**以 `dev-notes/15:行` 为证据；`experiments/*/REPORT.md` 持有夜间全部实测数字；
2. **物理搬运会打断** `uv run` / `pytest`（基线 **258 passed**）/ 几十处路径引用。

⇒ **效力等级**（`dev-notes/21 §29`）：`MEME_FRAMEWORK.md` / `dev-notes/21` / `methodology/` **> `dev-notes/01–20`**；
**旧档里的数字不再维护**，冲突时以新者为准。

**旧框架仍然成立的四条**（可作为经验复用）：重整+解引用层 / 受约束解释卡 / 指针机械层解释 / **候选集路由收紧**。

---

## 6. 纪律入口

| 在哪 | 是什么 |
|---|---|
| **`MEME_FRAMEWORK.md` §7** | **E1–E17**（新框架经验）：手写元决策 / 先比免费规则(含计数类) / 构念门双类规则 / 读出≠会用 / 事后挑选必须独立出口预注册 / **E16 离散中介必须 STE** / **E17 消融必须 retrain-without** |
| **`dev-notes/21`** | **D1–D15**（口径裁定与测量纪律）+ 逐条裁定 + 我的失误更正记录 |

**日志一律不入库**（`.gitignore`）⇒ **关键数字必须复述进 REPORT / results.json / 提交信息**，不能只活在 stdout 里。
