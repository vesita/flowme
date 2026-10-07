# P17 淘汰清理记录

**判定：部分完成（U0–U4 全过；98 个权重目标按 U0 前置门跳过，原因见 §5）。**

依据：`experiments/card_catalog/CATALOG.md` §3.2（该删 A/B/C）与 §3.1（有真实空间，绝删）。
执行：删除 5 个 B 档目录；A 档 7 组与 checkpoints 点名 21 个 `.pt`、`idiom_card.pt` 全部跳过。
提交：`c67f302`（删除）、`de79d49`（回滚 additivity）。未使用 `git add -A` / `git stash` / `git clean`。

## 1. 前置门（U0，实测）

- `git status --porcelain` 起始仅 `?? experiments/entity_identity/`、`?? experiments/e2e_d1/`
  （他人会话的未跟踪目录），**不含任何待删路径** ⇒ 无需先提交。
- **关键发现**：`.gitignore:15-16` 为 `checkpoints/` 与 `*.pt`；`git ls-files '*.pt'` = **0**，
  `git log --all --oneline -- '*.pt'` 命中 **0** ⇒ **全仓 `.pt` 从未入库**，
  任何权重文件的 `git log -- <path>` 必然为空。
- 目录级证据（删除前实测，`git log --oneline -1 -- <dir>`）：

| 目标 | git log（删除前） | 目录内入库文件数 |
|---|---|---|
| `experiments/core_branch` | `98233e7` 模块组合架构探索：能力可加性、输入/输出不变量与「改核不伤老卡」 | 13 |
| `experiments/layer_selective` | `eabafce` layer_selective 报告后补 + methodology 升格 | 29 |
| `experiments/core_select_only` | `c96b2bd` 机制链收束：老任务不是增益的必要条件，它是「老卡的保险」 | 17 |
| `experiments/core_ndb` | `2d52d88` 编码缓存修复 + 对话路径落地 + §3.1 必需位语义修正 + 三项实验结论 | 15 |
| `experiments/selection_cards` | `d8c8c18` 阶段二/三：语料入口修复、否定卡、选择族、组合算子、对抗集与容量体检 | 21 |
| `experiments/additivity` | `98233e7` 同 core_branch | 14 |

## 2. 逐目标处置

格式：路径 | 档位 | 处置 | 入库证据 | 释放 .pt 数 / 字节

### 2.1 该删 A（免费规则已解决，7 组）——按 §2 范围只删权重与缓存

| 路径 | 档位 | 处置 | 入库证据 | 释放 .pt / 字节 |
|---|---|---|---|---|
| `experiments/anchored_select/cards/*.pt`（6） | A | **跳过**：git log 空（`*.pt` 未入库） | 空 | 0 / 0 |
| `experiments/value_card/cards/*.pt`（4） | A | **跳过**：同上 | 空 | 0 / 0 |
| `experiments/bag_modules/weights/`（22）+ `cache/`（4） | A | **跳过**：tracked=0、git log 空 | 空 | 0 / 0 |
| `experiments/funcword_minpair/weights/`（15）+ `cache/`（7） | A | **跳过**：同上 | 空 | 0 / 0 |
| `experiments/syllogism_card/weights/`（10） | A | **跳过**：同上 | 空 | 0 / 0 |
| `experiments/gen_data_loop/cards/*.pt`（8） | A | **跳过**：同上 | 空 | 0 / 0 |
| `experiments/sentence_mode` | A | **无可删目标**：`find -name '*.pt'` = 0（CATALOG 已注明「无卡权重」）；REPORT/PREREG/代码按范围保留 | `7f58cd2` | 0 / 0 |

A 档合计释放：**0 个 .pt / 0 字节**（`REPORT.md` / `PREREG.md` / 代码全部保留，负面结果知识未动）。

### 2.2 该删 B（未参与 / 被取代，10 组）

| 路径 | 档位 | 处置 | 入库证据 | 释放 .pt / 字节 |
|---|---|---|---|---|
| `experiments/core_branch/` | B | **删**（整目录，`rm -rf` 前 realpath 确认在 `DTSeek/experiments/` 内） | `98233e7` | 16 / 55,141,768 |
| `experiments/layer_selective/` | B | **删**（整目录） | `eabafce` | 21 / 156,914,571 |
| `experiments/core_select_only/` | B | **删**（整目录） | `c96b2bd` | 9 / 66,799,375 |
| `experiments/core_ndb/` | B | **删**（整目录） | `2d52d88` | 4 / 77,919,028 |
| `experiments/selection_cards/` | B | **删**（整目录） | `d8c8c18` | 11 / 55,060,953 |
| `experiments/additivity/` | B | **删后回滚**：14 个入库文件已恢复（`de79d49`）；14 个 `.pt` 不可恢复 | `98233e7` | 14 / 41,408,906 |
| `checkpoints/e5*`（9 个 `.pt`） | B | **跳过**：git log 空（`checkpoints/` 被 .gitignore，`git ls-files checkpoints/` = 0） | 空 | 0 / 0 |
| `checkpoints/arm_*`（10 个 `.pt`） | B | **跳过**：同上 | 空 | 0 / 0 |
| `checkpoints/reply_pick_joint.pt` | B | **跳过**：同上 | 空 | 0 / 0 |
| `checkpoints/smoke_test.pt` | B | **跳过**：同上 | 空 | 0 / 0 |

### 2.3 该删 C（基座不一致且无说明，2 处）

| 路径 | 档位 | 处置 | 入库证据 | 释放 .pt / 字节 |
|---|---|---|---|---|
| `experiments/gen_dispatch/cache/idiom_card.pt` | C | **跳过**：git log 空。内容侧已核安全——`grep -rn "idiom_card" src/ tests/` 无引用（exit=1），但 U0 门槛不成立 | 空 | 0 / 0 |
| 在用 negation 路径（`compose_ops/artifacts/cards/negation.pt` ← `dialogue.py:105`） | C | **不删**（按任务指示，修复流程处理中） | — | 0 / 0 |

**合计释放：75 个 `.pt` / 453,244,601 字节**（另含 5 个已删目录的非 `.pt` 入库文件 1,092,342 字节）。
**跳过目标：98 个 `.pt`**（A 档 76 + checkpoints 点名 21 + `idiom_card.pt` 1）。

## 3. 删除前后对比（实测）

| 指标 | 删除前 | 删除后 | Δ |
|---|---|---|---|
| `.pt` 总数（全仓，排除 `.venv`） | **337** | **262** | −75 |
| `.pt`（`experiments/` 下） | 304 | 229 | −75 |
| `experiments/` 总大小（`du -sb`） | 3,684,069,650 B（3.5G） | 3,233,028,468 B（3.1G） | −451,041,182 B |
| `checkpoints/` 下 `.pt` | 33 | 33 | 0（全部跳过） |

说明：`du` 差值（451,041,182 B）略小于逐目录释放量（454,336,943 B），因同期
`experiments/e2e_d1/`、`experiments/entity_identity/` 两会在写入（文件数 7→8、8→13）。

## 4. U2 禁删清单确认（删除后逐项 `ls`/`find`，全部在场）

| 目标 | 状态 | 文件数 |
|---|---|---|
| `src/` | OK | 93 |
| `tests/` | OK | 33 |
| `dev-notes/` | OK | 21 |
| `methodology/` | OK | 6 |
| `experiments/card_catalog/` | OK | 2 |
| `experiments/core_probe/` | OK | 22 |
| `checkpoints/base_encoder.pt` | OK | 6,764,277 B |
| 生产四卡 `checkpoints/cards/{person,pronoun,relation,sentiment}.pt` | OK，逐个在场 | 4 |
| `experiments/prod_card_audit/` | OK | 17 |
| `experiments/e2e_d1/` | OK | 8 |
| `experiments/entity_identity/` | OK | 13 |
| 有真实空间 5 组 `struct_supervision` / `two_channel_head` / `mode_conditioned_skel` / `free_rule_floor` / `skeleton_leak` | OK | 71 / 84 / 27 / 50 / 22 |

误删 = **0**。

## 5. 跳过项与原因（逐条）

1. **A 档 76 个 `.pt`**：`git log -- <path>` 为空 ⇒ U0 规定「跳过并记录」。
   根因：`.gitignore:16` `*.pt`，全仓 0 个 `.pt` 被跟踪、全历史 0 命中 ⇒ 权重内容**不在 git 历史里**。
2. **`checkpoints/` 点名 21 个 `.pt`**：`.gitignore:15` `checkpoints/` 整目录忽略，同上为空。
3. **`experiments/gen_dispatch/cache/idiom_card.pt`**：`.gitignore:40` `experiments/**/cache/` +
   `*.pt` 双重忽略，git log 为空。**内容侧本可安全删**（`grep -rn "idiom_card" src/` 无引用），
   但 U0 门槛不成立 ⇒ 跳过。
4. **`experiments/sentence_mode`**：无 `.pt`，A 档范围内无可删对象。
5. **`experiments/additivity/` 的入库文件**：删除后发现它是 4 个存活实验的共享代码依赖
   （见 §6.1）⇒ 回滚；其 14 个 `.pt` 随整目录一并释放且不可恢复。

## 6. 遗留

### 实测

1. **`.pt` 全部不在版本控制内**（`git ls-files '*.pt'` = 0；`git log --all -- '*.pt'` = 0）⇒
   本次 U0 门槛对**所有权重文件**不成立；若后续要真正释放 A 档/checkpoints 的 98 个 `.pt`，
   需先另立「允许无 git 证据删除」的授权或先做异地备份。
2. **`experiments/additivity/` 是存活实验的共享代码依赖**：
   - `experiments/capability_map/eval_cards.py:29` 导入 bypass 模块的 `BypassSet`；
   - `experiments/capability_map/train_cards.py:63` 延迟导入 `train_bypass_card`；
   - `capability_map/{prepare,probe,eval_cards}.py`、`core_keep/train_core_keep.py`、
     `cumulative_add/train_cumulative.py`、`anchored_select/train_joint.py` 均把
     `ROOT/experiments/additivity` 插入 `sys.path`。
3. **`capability_map` 的 negation 档 4 个 ckpt 不可复跑**：
   `eval_cards.py:34,39` 读 `experiments/additivity/cards/negation_{frozen,bypass}_base_s{42,43}.pt`，
   全仓无副本（`find . -name 'negation_*_base_s*.pt'` 空）。结论已留
   `experiments/capability_map/REPORT.md:36`。
4. **其余 5 个已删目录无存活代码依赖**：按字符串形式复查目录名（`"core_branch"` 等），
   存活 `.py`/`.sh` 命中 = 0；仅 `out_invariants/run_reject.py:72`（docstring 口径说明）与
   `cumulative_add/run_all.sh:4`（注释）提及，不构成运行依赖。
5. **U3**：`uv run pytest -q` 删除前 258 passed、删除后 **258 passed**（23.45s），无需回滚。

### 推断（未测，不得当结论）

1. 「A 档 76 个 `.pt` 可由 `weights/`/`cache/` 下脚本重跑再生成」——依据 `.gitignore` 注释
   「体积大、可复现」，**未实际复跑验证**。
2. 「`capability_map` 除 negation 档外的复跑不受影响」——只核到 import 路径已回滚，
   **未实跑该实验**。
3. CATALOG §3.2 B 对 `experiments/additivity` 的「该删」裁定未覆盖其跨实验代码依赖；
   本次以「保留在用资产 / 宁可少删」为准做了部分回滚——**这是本单元的裁量，非 CATALOG 原文要求**。
