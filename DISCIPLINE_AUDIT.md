# 纪律全库审计：R39（eval 泄漏） / R40（batch=1 线程数）

范围：只读静态扫描 `stages/*.py`（48 个活跃 + 50_ P3b 在跑，仅读）+ `cards/*.py`（11 个）。未改任何其它文件。
判定三档：**安全** / **潜在泄漏**（当前读数未被污染，但函数出口留下 `train()` 活陷阱）/ **明确泄漏**（当前即带 dropout 读数）。

## 1. R39 清单（评估路径的 eval 泄漏）

| 文件:行 | 函数 / 上下文 | 判定 | 依据 / 最小修法 |
|---|---|---|---|
| `stages/19_residual_scan.py:612` | `greedy_gen` 出口 `model.train()`（入口 579 `eval()`） | **潜在泄漏（陷阱源）** | R39 由来；改为 `was=model.training; …; if was: model.train()` |
| `stages/19_residual_scan.py:487` | `eval_ce` 出口 `train()`（入口 476 `eval()`） | **潜在泄漏** | 同型；同修法 |
| `stages/19_residual_scan.py:536` | `report_ce` 出口 `train()`（入口 514 `eval()`） | **潜在泄漏** | 同型；同修法 |
| `stages/19_residual_scan.py:431` | `masked_ce` 无 eval | 安全 | 训练前向，故意要 dropout 开 |
| `cards/example_add3d.py:374` | `greedy_gen` 出口 `pipe.train()`（入口 340 `eval()`） | **潜在泄漏** | 被 `example_attribution.py:181/303`、`example_versioning.py:183` 复用 |
| `cards/example_add3d.py:420-436` | `digit_report`（420 eval / 435 `was_training` 恢复） | 安全（正确范式） | 应把此 `was_training` 恢复推广到所有 `greedy_gen` |
| `stages/36_saturation_retest.py:318,337` | `per_sample_ce` 自设 eval、出口 train() | 安全 + 潜在出口 | 主循环 `em_at`→`per_sample_ce` 顺序均自守卫（:430-450） |
| `stages/44_struct_at_12k.py:350,369` | `per_sample_ce` 自设 eval、出口 train()；`eval_state:456` train() | 安全 + 潜在出口 | 调用点 `:454,497,500,543,564,566` 均自守卫 |
| `stages/45_cross_task_transfer.py:430,448` | `per_sample_ce` 自设 eval、出口 train() | 安全 + 潜在出口 | `full_eval:503` 自守卫；`em_at`→`per_sample_ce` 安全 |
| `stages/46_binding_vs_addition.py:314,325,357` | `em_eval`/`swap_sensitivity`/`binding_r29` 各自 eval | 安全 | 三处前向前均有 `model.eval()` |
| `stages/46_binding_vs_addition.py:458-465` | 显式 `train()` 后测 dropout RNG 差 | 安全（有意） | 设计上的 eval/train 对比，非读数路径 |
| `stages/50_binding_at_saturation.py:391,413,424,456` | `digit_ce`/`em_eval`/`swap_sensitivity`/`binding_r29` 各自 eval | 安全 | 四处自守卫（出口 `train()` :404 由下一次入口覆盖） |
| `cards/exp_struct_ablation.py:425,428,458` | 每次前向比较前重设 `oracle.eval()/pipe.eval()` | 安全 | 27c7b39 的修复点；注释已写明陷阱 |
| `cards/example_attribution.py:179,181-182` | `base.eval()`→`em_score`(greedy_gen)→`base.train()` | **潜在泄漏** | 后续读数靠 `digit_report`/`em_score` 自守卫，未被污染；但 `base` 全程处 train 态 |
| `cards/example_versioning.py:181-184` | `train_target`→`em_score`→`digit_report` | 安全 | 两个测量入口均自设 eval |

**明确泄漏：0 处。** 所有 `model.logits(...)` 调用点均已核对，非自守卫者仅训练前向（19:431）与有意的 train-mode 对比（46/50 CHK-EQ）。

## 2. R40 清单（batch=1 逐样本前向的线程数）

| 文件:行 | 是否设线程 / 设几 | batch=1 路径 | 判定 | 建议 |
|---|---|---|---|---|
| `stages/21_info_causal_curve.py:67` | 是 / **12**（S21_THREADS） | 有（EM 主口径 batch=1 全 test，:243-247） | **★未控** | 默认改 4 |
| `stages/16_probe_vs_causal.py:37` | 是 / **8** | 有（`greedy_gen([r],batch=1)` :266） | **★未控** | 默认改 4 |
| `stages/17_probe_causal_n7.py:46` | 是 / **8** | 有（:284） | **★未控** | 默认改 4 |
| `stages/26_meme_anchoring.py:74` | 是 / **12** | 有（batch=1 vs 64 纯前向 :334，与矩阵前向混用） | **★未控·混用** | 前向段临时 4 |
| `stages/30_concentration_vs_pr.py:77` | 是 / **12** | 有（batch=1 EM :281 + 批式 dig_ce :265） | **★未控·典型混用** | 拆成两档线程 |
| `stages/19_residual_scan.py`（:65-66 附近） | **否** | 有（EM batch=1 全量 800 :705） | **★未控**（继承全核） | 加 `set_num_threads(4)` |
| `stages/36_saturation_retest.py` | **否**（exec S19 头） | 有（:311,433） | **★未控** | 同 |
| `stages/44_struct_at_12k.py` / `45` / `46` / `50` | **否** | 有（batch=1 greedy） | **★未控**（均有 cuda assert，GPU 侧风险低；CPU 侧需人工确认） | 加 4 线程兜底 |
| `stages/12/13/14/15/25_*.py` | **否** | 有（EM_BATCH=1） | **★未控**（device=cuda 否则 cpu） | CPU 回退时设 4 |
| `stages/42_data_hygiene.py:42`、`43_clean_em.py:48` | 是 / **4** | 有（CPU batch=1） | ✓ 合规 | — |
| `cards/example_add3d.py:89`、`cards/exp_struct_ablation.py:64` | 是 / **4** | 有 | ✓ 合规 | — |
| `stages/28_replicate_single_card.py:51`、`31_seen_unseen.py:35` | 是 / **4** | 有 | ✓ 合规 | — |
| `stages/33_evidence_dedupe.py:33`、`39_ledger_update.py:31`、`47_ledger_update.py:39` | 是 / 12、12、10 | 无前向（仅读日志） | N/A（需人工确认无前向） | — |
| `stages/01(8)/02(4)/03(6)/06(3)/07(cpu//6)/08(1)` | 是 / 各异 | 无 batch=1 逐样本贪心 | N/A | — |

## 3. 汇总与历史影响面

- **R39：潜在泄漏 6 处**（19:487/536/612、example_add3d:374、example_attribution:179、以及 36/44/45 出口 train()），**明确泄漏 0 处**。所有评估入口自设 eval，故**未发现第二条已入库结论被 dropout 污染**。
- **R39 唯一被污染事件**：`cards/exp_struct_ablation.py` 复用的 `greedy_gen` 出口 `train()` ⇒ 附加 oracle 自检假报 EM=35.25%，已由 27c7b39 加 `eval()` 修正为 0.0。**该修正后的入库数据无需重跑**；但 `stages/19` 与 `cards/example_add3d` 的 `greedy_gen` 出口 `train()` **仍是活陷阱**，任何未来在贪心后新增裸前向都会重演 ⇒ 必须改为 `was_training` 恢复。
- **R40：未控线程 9 个文件**（21/16/17/26/30/19/36/44/45/46/50/12/13/14/15/25）。CPU 重跑这些脚本的 batch=1 贪心会异常慢。
- **R40 直接证据**：`logs/49_gate1_partial_12threads.log`（threads=12）中 case 8/9 耗时 **186.2s / 963.6s**，日志在 case 9 后被截断；最终入库的 `logs/49_gate1.log` 为 **threads=4**、每 case ≈10.5s。⇒ 12 线程那次被弃用，**入库数字无错，只浪费墙钟**，无需复核。
- **结论**：R39 无已入库结论需复核（唯一污染已被 27c7b39 修正并核验：`49_gate1.json` 全部 `d_logits_*=0.000e+00`、EM 与入库 Δ=0.00pp）；R40 无已入库结论需复核，但未控线程脚本在 CPU 重跑时慢 1–2 个数量级。

## 状态更新（2026-10-09）
- ✅ 已修：`cards/example_add3d.py` 的 `greedy_gen` 出口 → `was_training` 恢复范式（本次）
- ⏸ 未改（有意）：`stages/19_residual_scan.py:612` —— 它是历史可比基线，改了会破坏锚点可比性；
  引用它的人必须知道这个陷阱（已写入本文档）
- ⏸ 待办：R40 的 9 个未控线程文件（CPU 重跑会慢 1–2 个数量级）
