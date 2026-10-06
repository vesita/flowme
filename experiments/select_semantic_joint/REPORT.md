# select_semantic_joint 结果报告 —— 条件「乙」：更强表示下 `heldout_pair` 还是 ≈50% 吗？

> 判据与配方见 `PREREG.md`（mtime **2026-10-06 21:22:53**，早于首训 **21:35:59**，跑后未改；跑前 21:26 订正过 §4.3 的一处**事实错误**——两 ctype 子集是 627:623 而非 1:1，门槛本身未动，见 §6.4）。
> 「实测」= 本目录产物直接读出；「推断」= 由实测推出；「口径判断」= 构造时人为规定。
> 数据**只读复用** `experiments/select_rerank/data/` 的 **clean（N）臂** 与 `experiments/select_pool/cache/` 的 token 级缓存。

## 1. 两臂：配置 / 可训参数 / 冻结实况 / 代价（实测）

| | **核对臂 `frozen`** | **主臂 `joint`** |
|---|---|---|
| 聚合 | masked **mean-pool**（= `select_pool` 臂 A，聚合参数 0） | 同左 |
| 打分头 | 164,353，可训 | 164,353，可训 |
| 核 `NanoDocEncoder` | **1,688,460，可训 0**；`requires_grad` 全 False、`training=False`（实测断言） | **1,688,460，全可训**（`eval()` 全程，dropout=0 ⇒ train/eval 逐位同） |
| 四张老卡头 | 不挂载 | 挂载并**冻结**：pronoun 630,278 / sentiment 630,278 / relation 630,021 / person 631,306，**逐参数 `requires_grad=[false]`**（实测打印） |
| **可训合计** | **164,353** | **1,852,813**（= 164,353 + 1,688,460，断言过） |
| 每步数据 | select batch 64（读 `select_pool` 缓存，**只读**） | select batch 64（**live 编码**）+ **4 个老任务 batch**（`core_keep` J3「旧+新同批梯度」口径） |
| 优化器 | AdamW 单组 lr **1e-3**、wd 1e-4 | AdamW 两组：核 **3e-4**（= `core_keep --lr-base`）／头 **1e-3** |
| 共享配方 | 1800 步、batch 64、cosine T=1800、clip 1.0、seed 42/43 | 同左（**唯一变量 = 核是否可训**） |
| **步时（steady 中位）** | **0.0019 / 0.0018 s**（s42/s43） | **0.2543 / 0.2551 s**（rand 0.2527 / 0.2521） |
| **峰值显存** | **81.0 / 81.0 MB** | **1978.8 / 1979.7 MB** |
| 单跑墙钟 | 9.8 / 9.7 s | 476.5 / 478.1 s（rand 475.0 / 490.3 s） |

- **「两臂只差核是否可训」的证据（实测，`selftest.json`）**：把 joint 循环的核冻住跑 20 步 vs 只跑 select 跑 20 步 ⇒
  **打分头权重 `torch.equal = True`，逐张量 max|Δ| = 0.000e+00**（核冻结时老任务批次对可训参数是**严格零梯度**）。
- **live 编码 vs `select_pool` 缓存**：同批文本 max|Δ| = **3.144e-06**、mask 逐位相同（ROCm 1e-6 级不确定性；核对臂始终走缓存，故不受影响）。
- 空测试（`selftest.py`，全实测）：可训参数量、核与老头冻结实况、多数类（split 0.5000 / 两 ctype 0.5016 = 627:623）、
  随机标签真执行（改变 4026/4016 条、1:1 保持）、R4 基线就位 —— 全过。
- **计时/显存探针与端到端对账（同构自检）**：步时**不是另写探针**，而是在**真实训练循环内**逐 `perf_counter()` 取稳态中位；
  逐步 0.2543s × 1800 = **457.7s**，单跑实测 `wall_sec` **476.5s**，差 **18.8s = 4.0%**（= 前后各一次全量 eval + R4/R5 两次老卡全量评测 + IO）。
  unit 级（journald）：`dtseek-ssj-joint` **墙 32min9.5s / CPU 49min59.7s**，我的 4 跑 `wall_sec` 合计 **1919.9s = 32.00min**（差 9.6s = 4 次进程启动）；
  `dtseek-ssj-frozen` 墙 48.5s vs 4 跑合计 39.3s（差 9.2s）。**逐步 / 单跑 / unit 三处咬合（差 <5%）** ⇒ 报出的量的是同一个程序。
- **代价的账（只对上一部分，诚实标注）**：可训参数 + 梯度 + AdamW `exp_avg`/`exp_avg_sq` = 1,688,460 × 4B × 4 ≈ **27 MB**，但实测峰值 **+1897.8 MB**
  ⇒ **其余 ~1.87 GB 不是参数开销**，是 live 编码的反传激活与 4 个老任务 rollout 的计算图（`task_loss` 沿 rollout 逐步累加 CE ⇒ 整条轨迹留图）。
  **这是结构性解释，未逐层字节对账** —— 按「对不上账 = 可能有隐藏开销」的纪律，此条列为遗留第 8 项。

## 2. R0：核对臂是否**零误差**复现 `select_pool` 臂 A（实测）

| 项 | `select_pool` 臂 A | 本实验 `frozen` | **Δ** |
|---|---|---|---|
| `heldout_pair` s42 | 49.52 | **49.52** | **0.00** |
| `heldout_pair` s43 | 51.04 | **51.04** | **0.00** |
| `shifted_pos` s42 | 92.56 | **92.56** | **0.00** |
| `shifted_pos` s43 | 92.48 | **92.48** | **0.00** |
| `test`（两 seed） | 94.48 / 94.48 | **94.48 / 94.48** | 0.00 / 0.00 |
| `adv` 总 | 71.04 / 71.76 | **71.04 / 71.76** | 0.00 / 0.00 |
| `train` | 100.00 | **100.00** | 0.00 |

**R0 过 ✅：逐项 Δ = 0.00（零误差，不只是 ≤2×SE）** ⇒ 同数据、同核、同编码、同配方，往下做有效。

## 3. 主表（准确率 %；n=1250 ⇒ **SE = 1.414pt**，**2×SE = 2.828pt**；多数类 0.5016）

| 臂 | seed | `heldout_pair` | 对 50% 余量/SE | **Δ vs frozen** | Δ/SE | `shifted_pos` | Δ vs frozen | test | train |
|---|---|---|---|---|---|---|---|---|---|
| **frozen** | 42 | **49.52** | −0.34 | — | — | **92.56** | — | 94.48 | 100.00 |
| **frozen** | 43 | **51.04** | +0.73 | — | — | **92.48** | — | 94.48 | 100.00 |
| **joint** | 42 | **59.36** | +6.62 | **+9.84** | **+6.96** | **99.60** | **+7.04** | 99.88 | 100.00 |
| **joint** | 43 | **63.20** | +9.33 | **+12.16** | **+8.60** | **99.60** | **+7.12** | 99.84 | 100.00 |

**R3 随机标签对照（4 跑，门槛 ≤ 52.83；heldout / shifted）**
`frozen` s42 **51.84 / 49.68**｜s43 **51.60 / 49.12**｜`joint` s42 **48.24 / 48.40**｜s43 **49.76 / 49.36**
（4 跑真标签侧 train（对真标签）= 49.55 / 49.60 / 49.65 / 49.80 ⇒ 打乱标签确实被执行）

**2×2 交互表（heldout %，最有信息量的一张）**

| | 真标签 | 随机标签 |
|---|---|---|
| **核冻结** | 49.52 / 51.04 | 51.84 / 51.60 |
| **核可训** | **59.36 / 63.20** | 48.24 / 49.76 |

⇒ **只有「核可训 ∧ 标签为真」这一格离开 50%**；两条件缺一都回 50%。

**Δ 的口径（保守性声明）**：R1/R2 的 Δ 是**同 seed、同一批 1250 条 eval 样本上的配对差**（两臂评测输入完全相同，只是权重不同），
配对差的方差 ≤ 独立差。表里的 SE = 1.414pt 是**独立二项 SE** ⇒ 报出的 **Δ/SE 是下界**，用配对 SE 只会更大，**结论只会更强**（配对 SE 本实验未算，见遗留第 9 项）。

## 4. R1–R6 逐条实测 + 判定 + 机制

| # | 实测 | 判定 |
|---|---|---|
| **R0（门禁）** | 8 项全 Δ=0.00 | **过 ✅** |
| **R1（主）** | Δ = **+9.84 / +12.16pt** = **+6.96 / +8.60 SE**（门槛 2.828pt，余量 +7.01 / +9.33pt），**两 seed 同号**；joint 自身对 50% = +6.62 / +9.33 SE | **过 ✅** |
| **R2（副）** | `shifted` **99.60 / 99.60** vs frozen 92.56 / 92.48 ⇒ **+7.04 / +7.12pt**（门槛 ≥ frozen−2.83 = 89.73 / 89.65） | **过 ✅（不但没塌，还涨了）** |
| **R3（门禁）** | 4 跑随机标签 heldout/shifted = 48.24~51.60 / 48.40~49.68，全 ≤ 52.83 | **过 ✅** |
| **R4（门禁·硬）** | step-0 与「不接本任务」配置的四卡逐样本 pred/exact **sha256 完全相同**（两 seed）；基线 exact 与 `capability_map/summary.json::frozen_exact` Δ=+0.00e+00（8 项），与 `evaluate_task` 互证相等 | **过 ✅** |
| **R5（报告）** | 8/8 格 Δ ≥ −各自带：pronoun **+0.020 / −0.0017**（带 0.0283）、sentiment **+0.1559 / +0.1413**（带 0.0041）、relation **+0.0153 / +0.0250**（带 0.0139）、person **+0.0083 / −0.0017**（带 0.0033） | **过 ✅（无一退化）** |
| **R6（必报）** | 核漂移 **0.1254 / 0.1264**（rand 0.1391 / 0.1385）；步时 **0.2543 / 0.2551 s**（frozen 0.0019 s，**×134**）；峰值显存 **1978.8 / 1979.7 MB**（frozen 81 MB，**+1898 MB**） | 数值如上 |

**判定（PREREG §3 三选一，门禁全过、R1 过）= ② 墙是「核没学过这个任务」。**
⇒ **条件「乙」不成立**：在更强表示下 `heldout_pair` **不是 ≈50%**，而是 **59.36 / 63.20%**（+6.6 / +9.3 SE）。
⇒ 路线从「补表示」改成「**让核参与训练**」。

**机制（实测 → 推断分开）**：
- 【实测】**这是增益，不是交换**：`shifted_pos` 92.56→99.60、`test` 94.48→99.88 **同时上升** ⇒ 与 `select_pool` 臂 D「拿 25pt 表内记忆换 5.6pt」**完全不同**，R2 不是勉强过，是反向过。
- 【实测】核**确实动了**：漂移 0.1254/0.1264，与 `core_keep` J3 实测的 **0.1121** 同量级 ⇒ 「更强表示」不是空标签。
- 【实测】**随机标签把增益清零**（48.24/49.76），且该臂核漂移**更大**（0.139/0.139）⇒ **光有漂移不产生增益**，增益必须由真标签的 select 梯度驱动。
- 【实测】`select loss` 后 50 步均值 3.3e-05 / 1.2e-05、train 100.00 ⇒ 新任务**没被老任务挤掉**。
  ⚠ **口径澄清**：`results/*.json::loss_total_share_old = 0.99999` 是**损失值占比、不是梯度占比** —— 它接近 1 是因为 select CE **已被压到 ~3e-05**，不是因为老任务信号淹没了 select 任务（train 100.00 / test 99.88 就是反证）。
  **梯度量级没有实测拆分**（select CE 与 4 个老任务 loss 的梯度 norm 各自多大，未量，见遗留第 10 项）。
- 【实测】**空测试已排除「自变量没变」**：参数量 164,353 → 1,852,813（×11.3）、步时 0.0019 → 0.2543（×134）、显存 81 → 1979MB（×24）——
  三个可观测量**全都大动**，且「冻核跑 joint 循环」的打分头与「只跑 select」**逐位相同** ⇒ 变的确实是 `requires_grad`，不是别的。
- 【推断·置信度分级】**R0/R1/R2 = 强**（2 seed、同批配对 Δ、4 条门禁全过、且 shifted/test 反向涨）；
  **「增益来自核参与 select 任务」这条机制归因 = 中**（缺「核只学 select、不给老任务」那一档，见遗留第 2 项；随机标签臂已排除「光有漂移就够」，但没排除「老任务锚定 + 真标签 select」与「纯 select 梯度入核」的分解）。
- 【推断】冻结核时，`heldout_pair` 上限被「核从不在本任务上更新」这条卡住：同一个打分头、同一套 mean-pool，放开核就 +10~12pt ⇒ **前四条同类测量测到的「表示里没有语义等价」至少有一部分其实是「表示没被这个任务训过」**。
- 【推断】剩余缺口仍在（59~63% 远未解决），且两 seed 差 3.84pt（≈2.7 SE）⇒ 后续若要追「还能不能更高」需另开 PREREG（步数/lr/老任务权重都未扫）。

## 5. R4 step-0 逐位对账的 sha256（本轮硬门禁，**实测**）

| seed | 配置 B「不接本任务」 | 配置 A「主臂 step-0」 | 相等？ |
|---|---|---|---|
| **42** | `d2ea73d8648deb344f17cb56efcc2c3ffdfab78f6c6ef5e72dc65df42ca7f346` | `d2ea73d8648deb344f17cb56efcc2c3ffdfab78f6c6ef5e72dc65df42ca7f346` | **✅** |
| **43** | `119e825d86f9e1e5bdfdfd359fae2133850b771e7f226a887c582888ba9a3281` | `119e825d86f9e1e5bdfdfd359fae2133850b771e7f226a887c582888ba9a3281` | **✅** |

逐卡（s42 / s43）：pronoun `eb025ba2…3559` / `06b71108…700e`、sentiment `e31b5af1…9233` / `46baaecc…f4a`、
relation `0d5696fc…126` / `29752bf5…5393`、person `540bf9b7…37f1` / `10c787c0…7656` —— 两配置**逐卡逐位相同**。
三处独立计算并互证：`r4_check.py`（配置 B，含 `evaluate_task` 互证）／`train_sem.py` step-0（配置 A，任何 `opt.step()` 之前，且当场与基线断言相等）／`analyze.py` 再对账一次。

## 6. 原始命令 / unit / 日志 / 产物 / 纪律 / 遗留

```bash
uv run python experiments/select_semantic_joint/r4_check.py              # R4 基线（配置 B，两 seed）
uv run python experiments/select_semantic_joint/selftest.py --steps 20   # 空测试 + 等价性 + live/cache
systemd-run --user --unit=dtseek-ssj-frozen --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  --setenv=PYTHONDONTWRITEBYTECODE=1 \
  bash experiments/select_semantic_joint/run_all.sh frozen               # 21:35:59→21:36:47 ALL DONE
systemd-run --user --unit=dtseek-ssj-joint  ... bash .../run_all.sh joint# 21:41:30→22:13:39 ALL DONE
systemd-run --user --unit=dtseek-ssj-probe  ... train_sem.py --arm joint --seed 42 --steps 100 --tag probe
uv run python experiments/select_semantic_joint/analyze.py               # R0–R6 + 判定 → results/summary.json
```

- **unit**（三者现已全部 inactive，`systemctl --user list-units 'dtseek*'` 为空）：
  `dtseek-ssj-frozen.service`（21:35:59→21:36:47，ALL DONE，CPU 2min6.7s / 墙 48.5s / 内存峰值 3.1G；**invocation id 未打印**，以 journal 时间戳为准）、
  `dtseek-ssj-joint.service`（21:41:30→22:13:39，ALL DONE，invocation `2dd9faed4bfb431980f12c93fdefc46f`，CPU 49min59.7s / 墙 32min9.5s / 内存峰值 2.8G）、
  `dtseek-ssj-probe.service`（invocation `98718a17ecba404f9a6a4b2caf00bd87`，CPU 1min5.4s / 墙 44.5s）。
  三次均为 `systemd-run --user --unit=… --collect` + `PYTHONUNBUFFERED=1` + `PYTHONDONTWRITEBYTECODE=1`，**无 `setsid nohup`**。
- **日志**：`logs/select_semantic_joint_{frozen,joint}_s{42,43}[_rand].log`（8 个）+ `select_semantic_joint_probe.log`；单元 stdout 在 `journalctl --user -u dtseek-ssj-joint`。
- **产物**：`experiments/select_semantic_joint/results/*.json`（8 主跑 + `jointprobe_s42.json` + `r4_baseline_s{42,43}.json` + `selftest.json` + **`summary.json`（R0–R6 + 判定）**）；`weights/*.pt`（9）。
- **只读自证**：三个 jsonl 跑前/跑后 md5 = `6b1e6fb1…` / `d3d597af…` / `2a5ca5f4…` **一致**；`find -newermt PREREG` 对 `select_rerank/`、`select_pool/` **为空**；`git status` 对 `experiments/select_rerank`、`select_pool`、`src/`、`training/`、`tests/` **无条目**；脚本全程 `sys.dont_write_bytecode=True` + `PYTHONDONTWRITEBYTECODE=1`（不落他人 `__pycache__`）。
- **无 git 写操作**（未跑 commit/stash/checkout/restore/clean）；开训前查过 `ps` 与 `systemctl --user is-active 'dtseek*'`——发现别的会话的 `dtseek-genphase2.service` 正在跑，**排队等到 21:34:22 它结束**才于 21:35:59 开跑，**未杀任何进程**。
- **非本代理改动（实测，供对账）**：`dev-notes/16-*.md` mtime 21:39:01、`experiments/gen_data_loop/*` 为另一会话产物 —— 本代理只对 `dev-notes/16` 做过 `read`/`grep`，**未写**。

**遗留与不确定（实测 / 推断分开）**：
1. 【实测】只跑了 **mean-pool 臂 A × 两档核状态 × 2 seed**；`select_pool` 的 B/C/D/A+ 四臂**在可训核下未重跑** ⇒ 严格说 `dev-notes/16` 条件「乙」原文的「同一**四臂**仍 ≈50%」被实例化为「同一臂 A」。臂 A 是基线臂、且 D 已被判定不该用（25pt 换 5.6pt），故这一步是**先测清基线的墙**，不是回避。
2. 【实测】主臂每步**另加 4 个老任务批次**（J3 口径）。它的**零梯度等价性已实测证明**（§1，冻核时逐位相同），但它对可训核起**锚定**作用 ⇒ 本次的「更强表示」=「**被老任务锚住的、可训的核**」，不是「只学 select 的核」。R3 的 joint 随机标签臂（老任务照跑、漂移更大、heldout 仍 48~50%）说明老任务数据本身不产生增益，但「**只让核学 select、不给老任务**」这一档**没跑**（预期代价是 R5，PREREG 未预注册该档）。
3. 【实测】100 步探跑 heldout 已 55.12、shifted 96.64、漂移仅 0.0279 ⇒ 增益随步数**未见饱和**；1800 步之外**未扫**（跑前定死，不调参刷过）。
4. 【推断】59~63% 距可用仍远，且 seed 间差 3.84pt ⇒ 「核参与训练」是**必要条件的证据**，不是「问题已解决」。
5. 【实测】R5 的 person 两 seed 为 +0.0083 / −0.0017（带 0.0033，n=600 ⇒ 1σ≈2.04pt）—— s43 的 −0.17pt **低于噪声分辨率**，按字面判过，但该带是「同配置换 seed 的配对差」，比单次二项 SE 紧得多，**读数时别当精确阈值**。
6. 【推断】`eval()` 全程（核 dropout=0）与 `core_keep` J3 训练时 dropout=0.1 **不同口径** —— 这是为了让两臂唯一变量是 `requires_grad`（PREREG §2 写死）；若换成 J3 的 dropout 口径，数字可能变，**未测**。
7. 【实测】PREREG §4.3 在**训练前（21:26）**订正过一处事实错误（两 ctype 子集是 627:623、majority 0.5016，原文误写成 1:1 / 0.5000）；**判据门槛未改**。已知的不一致一律在此列出，不藏。
8. 【实测】**代价的账只对上一部分**：可训参数+梯度+AdamW 二阶态 ≈ 27 MB，实测峰值 +1898 MB ⇒ 余下 ~1.87 GB 归因于 live 反传激活与老任务 rollout 计算图，**未逐层字节对账**；按「对不上账 ⇒ 可能有隐藏开销（重复前向/额外拷贝/未释放 buffer）」的纪律，此项**未排除**。
9. 【实测】R1 的 Δ/SE 用**独立二项 SE（1.414pt）**，而两臂是在**同一批 1250 条**上评测的**配对差** ⇒ 报出的 Δ/SE 是**保守下界**；配对 SE（McNemar 或按样本配对方差）**未算**，算了只会更强。
10. 【实测】**梯度量级未拆分**：select CE 与 4 个老任务 loss 对核的梯度 norm 各自多大、谁在主导，**没有量**（只量了 loss 值，见 §4 口径澄清）。若后续要论证「核是被 select 任务本身驱动的」，这是第一件该补的测量。
11. 【推断】两臂的 `heldout_pair` 都在**同一份 adv 的 `clean(N)` 子集**上评（n=1250），换数据集 / 换阈值 / 换 ctype 的泛化性**未测**。
