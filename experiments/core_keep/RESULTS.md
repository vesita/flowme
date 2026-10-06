# core_keep 结果：联合训练（可改核）+ 保持老卡约束 —— 四档 × 2 seed

预注册：`experiments/core_keep/PREREG.md`（训练开始前落盘）。参照核：`checkpoints/base_encoder.pt`
（与 `multitask_v2_dtseek.pt` 的 `doc_encoder` 26/26 张量逐位相同，二选一无差别，已实测）。

## 1. 实现 / 超参 / 蒸馏形式 / 参照核

| 档 | 起点 | 核 | 老卡头 | 附加项 |
|---|---|---|---|---|
| J0 | 温启动 | 训 | 训 | 无（"纯联合"对照） |
| J1 | 温启动 | 训 | 训 | LwF 蒸馏，λ_kd=1.0，T=2.0 |
| J2 | 温启动 | 训 | 训 | J1 + 老卡 replay，λ_rep=1.0 |
| J3 | 温启动 | 训 | **冻** | 无（分离核/头漂移） |
| F / N5 锚 | 随机 | 训 | 训 | 无（从头联合） |
| J1c | 随机 | 训 | 训 | 同 J1 的蒸馏 |
| J1l4 / J1l16 | 温启动 | 训 | 训 | 同 J1，λ_kd=4 / 16（探索性，仅 s42） |

- 温启动 = 核 ← `base_encoder.pt`、四张老卡头 ← `capability_map/cards/{cap}_frozen_s{S}.pt`、negation 头随机。
- 蒸馏形式：`KL( 老卡头_学生(新核,x) ‖ 老卡头_教师(冻结旧核,x) )`，四头相加、按 step_mask 加权、×T²，与 CE 同为逐步 mean-over-batch 沿步求和。只在**老卡数据**上算。
- 数据/步数：与 `training/train_multitask.py` 逐项同口径，16 epoch × 84 step = 1344 步，AdamW(wd 1e-4)、lr 3e-4/1e-3、cosine、clip 1.0、`torch.manual_seed(seed)`。
- **保真自证**：F 档（我自己的脚本、从头 5 卡）Epoch1 与 `logs/neg5_s42.log` **五项全同**（15.436/24.521/19.409/123.982/13.611），末态 exact 与 N5 **五项全同** ⇒ 联合训练口径复刻无偏差。

## 2. 主表（exact，Δ = 相对冻结基线；带宽 pronoun 2.83 / sentiment 0.41 / relation 1.39 / person 0.33 pt；P2 线 0.870(s42) / 0.831(s43)）

| 档 | seed | pronoun | sentiment | relation | person | negation | P1字面 | P1单侧 | P2 |
|---|---|---|---|---|---|---|---|---|---|
| 冻结基线 | 42/43 | .9533/.9800 | .7937/.8147 | .9639/.9653 | .3100/.3233 | — | — | — | — |
| 从头联合 N5 锚 | 42/43 | .9333/.9300 | .7100/.7034 | .7986/.9194 | .3133/.3217 | .9667/.9233 | — | — | — |
| F（=N5 复刻） | 42 | .9333 (−2.00) | .7100 (−8.38) | .7986 (−16.53) | .3133 (+0.33) | .9667 | ❌ | ❌ | ✅ |
| **J0** | 42/43 | .9667/.9800 | .9166/.9322 | .9931/.9931 | .3200/.3250 | .9800/.9667 | ❌/❌ | ✅/✅ | ✅/✅ |
| **J1** | 42/43 | .9617/.9867 | .8819/.8850 | .9958/.9944 | .3183/.3233 | .9650/.9367 | ❌/❌ | ✅/✅ | ✅/✅ |
| **J2** | 42/43 | .9650/.9867 | .9041/.9159 | .9944/.9986 | .3183/.3233 | .9550/.9367 | ❌/❌ | ✅/✅ | ✅/✅ |
| **J3** | 42/43 | .9800/.9850 | .9147/.9325 | .9931/.9903 | .3167/.3233 | .9817/.9633 | ❌/❌ | ✅/✅ | ✅/✅ |
| J1c（从头+蒸馏） | 42/43 | .9350/.9367 | .7459/.6856 | .9375/.8986 | .3150/.3200 | .9367/.9083 | ❌/❌ | ❌/❌ | ✅/✅ |
| J1l4（λ=4, s42） | 42 | .9700 (+1.67) | .8534 (+5.97) | .9986 (+3.47) | .3150 (+0.50) | .9333 | ❌ | ✅ | ✅ |
| J1l16（λ=16, s42） | 42 | .9700 (+1.67) | .8300 (+3.63) | .9931 (+2.92) | .3167 (+0.67) | .9033 | ❌ | ✅ | ✅ |

括号内为相对基线的 Δ（pt）。P1字面 = 预注册双侧 |Δ|≤带宽；P1单侧 = "不退化"读法 Δ≥−带宽。

## 3. 蒸馏项实际数值 + 三项自检原始输出

```
SELFTEST_1 J0 {"core_total":1688460,...,"trainable_total":4840107}
SELFTEST_1 J3 {...,"pronoun":{"total":630278,"trainable":0},...,"trainable_total":2318224}
SELFTEST_2 J1  kd_first_step=12.234852 kd_mean_all=20.763006 kd_last=6.806323
               kd_share_first=0.542780 kd_share_last=0.284876 kd_share_all=0.437431  verdict 非零=True
SELFTEST_2 J2  kd_first_step=9.989159  kd_mean_all=20.234164 kd_share_all=0.299015 replay_mean=441.770123
SELFTEST_2 J0/J3 本档无蒸馏项 kd_mean_all=0.000000
SELFTEST_3 J3 pronoun/sentiment/relation/person.requires_grad = [False] verdict: 四张老卡头全部 requires_grad=False ✅
SELFTEST_4 s42/s43 基线现算 = capability_map 记录值  一致=True ×8
ALIGN_CHECK {5 卡} eval==val: True ×8 运行
SELFTEST_0（init_check.py，训练前）四卡 exact 与冻结基线 Δ=+0.00e+00 ×8 组合
```

**P3**（1344 步全程）：

| 档 | 可训参数 | s/step | 峰值显存 | kd_share（全程/末epoch） |
|---|---|---|---|---|
| J0 | 4,840,107 | 0.2752–0.2776 | 2365–2373 MB | 0 |
| J1 | 4,840,107 | 0.3585–0.3600 (+30%) | 2396 MB | 43.74% / 45.35%，末 epoch 28.5%/29.7% |
| J2 | 4,840,107 | 0.6020–0.6028 (+119%) | 4335–4358 MB | 29.90% / 30.71%，末 epoch 21.7% |
| J3 | 2,318,224 | 0.2488–0.2498 (−9%) | 2077–2086 MB | 0 |
| J1c | 4,840,107 | 0.3588–0.3596 | 2396 MB | 60.06% / 60.50% |
| J1l4 / J1l16 | 4,840,107 | 0.3604 | 2396 MB | 73.69% / 91.25% |

核漂移量（末态核 vs `base_encoder` 的 Frobenius 相对范数，实测）：J0 0.1090、J3 0.1121、J1 0.1044；从头 F 0.1719、J1c 0.1744。

## 4. 判定 / 机制 / 哪端先崩 / 取舍曲线

**判定（按预注册字面，双侧 P1）**：J1/J2/J3 **都不过 P1**（6/6 组合 P1=❌，全表 13/13 行 P1=❌），P2 **全部 13/13 过** ⇒ 规则落到"如实报"。
**但按字面规则问"哪一端先崩"，答案是：没有任何一端崩。**
- 老卡端：温启动四档 J0~J3 的 32 个 Δ **无一为负**（最小 0.00pt，最大 +12.28pt）；P1 失败**全部因"向上超带"**。
- 新能力端：negation 全部 ≥0.9033，四档主档全部 ≥0.9367，远高于 0.870/0.831。
- 即"崩"的是**双侧判据本身**：它把"老卡被提升了"也判为失败，等于禁止联合训练比基线更好。
- **单侧读法（"不退化" Δ≥−带宽）下 J0~J3 四档 × 2 seed 全部同时过 P1 与 P2，两 seed 同号。**

**机制（逐条给证据）**
1. **伤老卡的不是"改核"，是"从头联合"。** 同配方、同数据、同 1344 步，唯一差别是初始化：随机初始化 F/N5 的 sentiment −8.38/−11.13pt、relation −16.53/−4.59pt；温启动 J0 反而 +12.28/+11.75pt。（实测，2 seed 同号）
2. **核漂移不是真凶，是正贡献。** J3 的四张老卡头**零训练**（requires_grad 全 False，SELFTEST_3 已证），只改核，老卡仍 +12.09/+11.78pt，negation 0.9817/0.9633 —— 老卡提升**完全来自核**，同时新能力到位。（实测，2 seed 同号）
3. **"多训一倍"混淆已排除。** J3 老卡头没训任何一步却涨了 12pt；J0−J3 的 sentiment 差 +0.19/−0.03pt ⇒ 头继续训练的边际贡献 ≈ 0。（实测差值；第四格"冻核+训头"未跑，属推断）
4. **蒸馏在温启动下是负资产。** 方向对（往基线拉），但基线本身低于联合训练能达到的水平 ⇒ 拉低两端：λ=1 时 sentiment 从 +12.3 压到 +8.8pt、negation 0.980→0.965；λ=4 → +6.0pt / 0.933；λ=16 → +3.6pt / 0.903。（实测，仅 s42）
5. **蒸馏在从头分支救不了老卡。** J1c vs N5：sentiment +3.59/−1.78pt、relation +13.89/−2.08pt —— **两 seed 不同号**，无可靠收益；而 negation −3.00/−1.50pt **两 seed 同号下降** ⇒ 蒸馏唯一一致的效果是压新能力。（实测，2 seed）

**取舍曲线（s42，λ_kd ↑）**

| λ_kd | 0 (J0) | 1 (J1) | 4 | 16 |
|---|---|---|---|---|
| kd_share | 0% | 43.7% | 73.7% | 91.3% |
| sentiment Δ | +12.28pt | +8.81pt | +5.97pt | +3.63pt |
| relation Δ | +2.92 | +3.19 | +3.47 | +2.92 |
| negation | 0.9800 | 0.9650 | 0.9333 | 0.9033 |

老卡单调向基线收敛却**始终进不了 0.41pt 的带**；negation 单调下降。**若必须往双侧 P1 挤，先崩的一定是 negation 端**（λ=16 已较 J0 掉 7.7pt，而 sentiment 仍超带 3.6pt）。（实测 4 点；λ→∞ 的极限是推断，未测）

## 5. 命令 / unit / 日志 / 产物

```
systemd-run --user --unit=dtseek-corekeep-run --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  /bin/bash -lc 'exec bash experiments/core_keep/run_all.sh > logs/corekeep_run.log 2>&1'
# 同结构第二单元：dtseek-corekeep-extra → run_extra.sh → logs/corekeep_extra.log
```
- 编排：`experiments/core_keep/run_all.sh`（F → J0~J3×2seed → J1c×2seed → eval）、`run_extra.sh`（λ=4/16 → eval2）。
- 训练脚本 `train_core_keep.py`；评测 `eval_core_keep.py`；P3/自检汇总 `p3_report.py`；起点自检 `init_check.py`。
- 日志 `logs/corekeep_{F,J0,J1,J2,J3,J1c,J1l4,J1l16}_s{42,43}.log`、`logs/corekeep_{run,extra,eval,eval2}.log`。
- 产物 `experiments/core_keep/cards/*_s*.pt` + `*_metrics.json`；`results.json`、`init_check.json`、`summary.md`、`PREREG.md`。
- 耗时：F 459s、J0 374s、J1 484s、J2 813s、J3 337s；全批 04:27–06:19。

## 6. 遗留与不确定（实测 / 推断分开）

- **判据缺陷（实测、需人工裁决）**：预注册 P1 是双侧 |Δ|≤带宽，本实验里它**把"提升"也判失败**；单侧"不退化"读法结论完全相反（全过 vs 全不过）。两个读法都已原样报出，**未回头改判据**。
- **实测**：λ 阶梯只跑了 seed 42（预注册如此）；每个 ckpt 只评一次，无评测侧重复采样。
- **实测**：2×2 缺第四格（温启动 + 冻核 + 训头，原计划 J4 未跑）；靠 J3（老头零训练仍 +12pt）与 J0−J3≈0 已排除两个主要混淆，但正式的头单独效应未直接测。
- **推断**：`train_multitask.py` 口径下"温启动"是否等于部署场景的加卡，未做超预算对照（老卡头从 frozen 卡继续训，本身已训过 1344 步）。
- **实测**：`eval_core_keep.py` 首跑 rc=1，两处笔误（`tok` 未定义、基线行缺 `negation` 键）已修，末次 eval2 rc=0；训练与 `results.json` 不受影响。
- **实测**：全程无人改 `src/`、`training/`、`tests/`、`dev-notes/`；写入仅 `experiments/core_keep/`、`logs/corekeep_*`、`/tmp`。
