# E-B 能力可加性实验报告：冻结认知核 + per-module 低秩旁路（bypass 臂）

预注册：`experiments/additivity/PREREG.md`（跑前写死，判据事后未改）。

## 1. 旁路怎么插的（配置与真实参数量）

- **插入点 4 处**：`NanoDocEncoder.blocks[0..2]` 的输出（残差流）+ `norm` 的输出（进解码头前）。
  形式 `h ← h + (alpha/r)·(h @ Aᵀ @ Bᵀ)`；`A ~ N(0, 1/d)` 小初始化，**`B` 精确零初始化**。
- **rank r = 16、alpha = 16（scale = 1.0）**，d = 128。
- **gate 语义**：关闸 = 直接返回原张量（零算术）⇒ 老模块的计算路径与"没装旁路"同一条。
- **可训参数量（SELFTEST_1 实际打印值，不是估算）**：
  - bypass 档 `trainable_total = 646,148` = 头 629,764 + 旁路 **16,384**（基座 core 1,688,460 全部 `requires_grad=False`）
  - frozen 档 `trainable_total = 629,764`（旁路在场 16,384 但全部冻结在零）
- 卡产物：`experiments/additivity/cards/negation_{bypass,frozen}_{base,aprime}_s{42,43}.pt`

## 2. 三档对照（negation `exact_match`）

| 档 | 认知核 | 适配 | seed 42 | seed 43 | 判据 |
|---|---|---|---|---|---|
| joint（文档值 dev-notes/12 §12.5） | 可训 | 无 | **0.9667** | 0.9233 | 上限参考 |
| frozen（文档值，R 臂，A′ 基座） | 冻结 | 无 | **0.6383** | — | 下限参考 |
| frozen F（本次配对，base_encoder） | 冻结 | 无（旁路在场=零） | 0.6933 | 0.6500 | 配对基线 |
| frozen F（本次，A′ 基座=接文档口径） | 冻结 | 无 | 0.6900 | — | 配方复现：+5.2pt vs 文档，轨迹差异 |
| **bypass B（主配置 r16/1344 步）** | 冻结 | 低秩旁路 | **0.7267** | **0.7433** | **P2 ≥0.87：两 seed 都不过** |
| bypass B（A′ 基座） | 冻结 | 低秩旁路 | 0.7117 | 0.7283 | 同向（+2.2pt vs 同基座 F） |

- **P2 结论：不成立。** 两 seed 同号地远低于 0.87（差 13~16pt），不是噪声级差异。
- bypass 相对**同基座同 seed** frozen 的配对 Δ：base +3.34pt(s42) / +9.33pt(s43)，A′ +2.17pt(s42)
  —— 两 seed 同号小正增益，方向是"有帮助但远远不够"。
- `bg_fp`：bypass 13.5%/11.0% vs 文档 joint 1.6%（同向落后）。

## 3. 老模块逐卡 Δ（P1）+ 挂/不挂逐位一致性

固定评测集 = 各卡 val split（`task_samples` 与 A′/N5 同口径，seed 逐跑指定）；三条件：
A 不挂新卡 / B 挂新卡（老卡关闸）/ C 挂新卡且老卡**强制开闸**（敏感性对照）。

| 卡 | 噪声带 | A(s42) | B(s42) | Δ(s42) | A(s43) | B(s43) | Δ(s43) | 指标逐位同 | 逐样本预测逐位同 |
|---|---|---|---|---|---|---|---|---|---|
| pronoun | 2.83pt | 0.9217 | 0.9217 | **+0.000000** | 0.9600 | 0.9600 | **+0.000000** | 是 | 是 |
| sentiment | 0.41pt | 0.7528 | 0.7528 | **+0.000000** | 0.7716 | 0.7716 | **+0.000000** | 是 | 是 |
| relation | 1.39pt | 0.7306 | 0.7306 | **+0.000000** | 0.7417 | 0.7417 | **+0.000000** | 是 | 是 |
| person | 0.33pt | 0.3183 | 0.3183 | **+0.000000** | 0.3217 | 0.3217 | **+0.000000** | 是 | 是 |

- **P1 通过且是逐位的**：Δ = 0.000000（四卡 × 两 seed），`engine.predict` 逐样本 sha256 一致，
  指标字典（含 person 身份指标）完全相等。
- **敏感性对照 C 不是空检查**：强制开闸后与 A 不同的样本数 = pronoun 410/600、sentiment 2369/3200、
  relation 681/720、person 423/600（s42；s43 为 422/2372/696/428）⇒ 这套一致性检查真能测出干扰。
- 交叉核对：引擎路径的 negation 指标（0.7267 / 0.7433）与训练脚本评估**完全一致**。
- 说明：s42 与 s43 的 A 绝对值不同，是因为评测集划分 seed 不同（不是模型不同）；配对只在同 seed 内读。

## 4. 空测试自检三条（原始输出）

**① 生效后的可训参数量（打印实测）**
```
SELFTEST_1 trainable_report(mode=bypass)={"base_core_total": 1688460, "base_core_frozen": 1688460,
  "base_core_trainable": 0, "head_trainable": 629764, "bypass_total": 16384,
  "bypass_trainable": 16384, "trainable_total": 646148}
SELFTEST_1 trainable_report(mode=frozen)={... "base_core_trainable": 0, "head_trainable": 629764,
  "bypass_total": 16384, "bypass_trainable": 0, "trainable_total": 629764}
```

**② 旁路初始为零时 = 冻结基座（每个训练跑都打印）**
```
SELFTEST_2a max|Δ|(bypass开@零 vs 关)=0.000e+00 max|Δ|(bypass开@零 vs 无旁路基座)=0.000e+00 B全零=True
```
指标层面（旁路在场但冻结在零的 F 档 = "零旁路"配置）：F-base 0.6933/0.6500、F-A′ 0.6900
（文档 frozen 0.6383，同基座同预算，差 +5.2pt 归因于轨迹：R 臂的头与 4 张热启动头交替训练、
初始化与数据流的 RNG 不同 —— 见 §7 局限）。

**③ 训练后旁路权重明显离开初值 + 功能性非空壳**
```
SELFTEST_3: B_std_init=0.0 → B_std_final=0.0225~0.0415（4 点），A_max|Δ|=0.089~0.191
SELFTEST_3_verdict B全部离开零初值=True（全部 11 个训练跑）
PROBE（训练好的 B 卡）：per_site 相对扰动 ‖Δh‖/‖h‖ = 0.418/0.359/0.388/0.217（blocks0-2/norm）
  final doc_memory 相对变化 = 1.419；**关闸后 em 0.7267 → 0.4300（-29.7pt）**
```
⇒ 旁路既"动了权重"也"真在计算图里干活"（关闸即崩），不是空壳。

## 5. 判定与机制

**判定：假设不成立（P2 未过，两 seed 同号）。** P1 成立且强于判据（Δ=0 逐位），三自检全过。

机制（实测证据）：
1. **不是"没接到计算图"**：零初始化恒等 max|Δ|=0、训练后关闸掉 29.7pt —— 旁路确实在起作用；
2. **不是秩/容量不够**：全秩旁路（131,072 参数，r=128）同预算只有 0.7383，vs r16 的 0.7267（+1.2pt）、
   r32 0.7300（+0.3pt）—— 加秩几乎不涨；
3. **步数/学习率是主要可回收项但明显饱和**：r16 32ep 0.7633、r32 32ep 0.7917、
   r32 lr 3e-3 16ep 0.7600、**r32 64ep（5376 步 = 4× 预算）0.7967** —— 曲线
   （0.76→0.80→0.8033→0.7967）在 **0.80 附近平台**，离 0.87 仍差 7.3pt；
4. **根因更可能是"认知核要整体跟着改"**：joint(N5) 的基座与两块冻结基座的 Frobenius 相对差 **0.172**
   （A′ vs base_encoder 只有 0.095）—— 联合档把基座整体改写了 ~17%，4 处线性旁路 + 单头梯度在同预算内补不回来；
5. **头与旁路共适应**：B 档的头关掉旁路只剩 0.4300（低于 F 档 0.6933）⇒ 两者是联合训出来的一套，
   旁路不是"给冻结头上加个小补丁"。

探索臂总表（`mode=bypass`，base_encoder，seed 42 单 seed，**不参与判定**）：

| 配置 | em |
|---|---|
| r16 / 16ep / lr1e-3（主配置） | 0.7267 |
| r32 / 16ep | 0.7300 |
| r128（全秩）/ 16ep | 0.7383 |
| r16 / 32ep | 0.7633 |
| r32 / 32ep | 0.7917 |
| r32 / 16ep / lr 3e-3 | 0.7600 |
| r32 / 64ep（5376 步） | **0.7967** |

## 6. P3 代价（与 frozen 档配对：同机、同 seed、同 1344 步）

| 档 | 可训参数 | s/step（稳态，末 200 步中位） | train_sec | peak 显存 |
|---|---|---|---|---|
| frozen F s42 | 629,764 | 0.0207 | 29.59 | 172.3 MB |
| bypass B s42 | 646,148（+16,384，**+2.6%**） | 0.0272（**+31%**） | 37.51（+27%） | 259.5 MB（**+87.2 MB**） |
| frozen F s43 / bypass B s43 | 同上 | 0.0209 / 0.0271 | — | 172.3 / 259.5 MB |
| 旁路 r=128（全秩，探索） | 760,836 | 0.0283 | 37.58 | 267.8 MB |
| 旁路 r32 64ep（探索） | 662,532 | 0.0269 | 150.40（5376 步） | 260.7 MB |

显存增量来自旁路要求梯度穿过冻结基座激活（frozen 档不建图）；参数只加 16k 但激活驻留是主账。

## 7. 遗留与不确定

**实测**：
- 文档 frozen 0.6383 与我的 F-A′ 0.6900 差 5.2pt（同基座、同预算，不同训练轨迹）——
  文档值应读作"frozen 区间 0.64~0.69"，不改判据（P2 阈值 0.87 无论如何都不可能过）；
- 探索臂全部为 seed 42 单 seed（PREREG 登记为"不参与判定"）；主配置两 seed 同号。

**推断（未测）**：
- "17% 基座改写"是相关性证据，不等于因果上不可由旁路表达；
- 探索臂累计最优 0.7967（r32/64ep/单 seed）；未试的旁路变体：接在 embedding 后、
  非线性旁路（MLP）、与头分阶段训练、旁路只在解码时启用的蒸馏式训练。
  lr 提高（3e-3）与步数 4× 都已实测：涨但饱和在 ~0.80。

## 8. 原始命令 / unit / 日志

```bash
# 主套件（7 训练 + 2 老卡复核）
systemd-run --user --unit=dtseek-additivity-byp --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  /bin/bash -lc 'exec bash experiments/additivity/run_all.sh > logs/additivity_run.log 2>&1'
# 收尾（s43 复核重跑 + 探索臂 r32/32ep/全秩 + 机制探针）
systemd-run --user --unit=dtseek-additivity-tail  ... 'exec bash experiments/additivity/run_tail.sh > logs/additivity_tail.log 2>&1'
# lr / 64ep 探针
systemd-run --user --unit=dtseek-additivity-expl2 ... 'exec bash experiments/additivity/run_expl2.sh > logs/additivity_expl2.log 2>&1'
```
- 逐跑日志：`logs/additivity_{b,f}_base_s{42,43}.log`、`{b,f}_aprime_*`、`oldcard_s{42,43}.log`、
  `probe_b_s42.log`、`e_*.log`；总驱动日志 `logs/additivity_run.log` / `_tail.log` / `_expl2.log`。
- 数据：`experiments/additivity/old_card_check_s{42,43}.json`、`probe_bypass_base_s42.json`。
- 已知事故（已修）：`bypass.py` 曾把 `_encoder` 注册成子模块 → `state_dict` 递归
  `RecursionError`（s43 复核首跑失败 rc=1，改 `object.__setattr__` 后重跑 rc=0）；
  s42 是补丁前启动故未受影响。
