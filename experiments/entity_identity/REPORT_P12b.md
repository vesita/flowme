# entity_identity — P12b 补两臂数据 + 训主臂 X + E1–E4 / E6（**不做 E5 解释层**）

> 范围：不做 E5 解释层 / 不做 OOD / 不做免费规则穷举 / 不动核 / 不提交（D8）。口径：**实测**=本目录产物直读，**推断**=由实测推出。
> `PREREG.md` 未改一字（mtime 2026-10-07 13:20:26 +0800，12923 B）；P12a 报告 `REPORT.md`（E0 已过）**未改**，本文件只加不覆盖。

## 1. 两臂数据（n / 丢弃率 / 可构造性）+ 冻结实测

| 臂 | train | test | 丢弃 | 断言（`check_arms.py` 46/46 PASS，exit 0） |
|---|---|---|---|---|
| **X-ir** 零共享字 | 2000 = SAME 1000（B 型）/ DIFF 1000（N1 代称错配 750 + N2 异名对 250） | 1200 = 600/600（600/450/150） | **0/2000、0/1200 = 0.00%** | 提及对零共享字违例 **0**；人/机构、同句/异句两标签各半 |
| **X-cross** 实体级划分（EA 训 / EB 评） | 4000（A/B/C/D 各 1000） | 1200（各 300） | 3/4003 = **0.0749%**、0/1200 = **0.00%**（丢弃原因=文本重复） | 实体 surface 交 **0**、实体 id 交 **0**、名字字面交 **0**；代称 他/她/该校 按设计共享；名池两半不相交 |

**可构造性（实测枚举，不降级）**：人名异名对 **70300/79800** 对零共享字 ⇒ 可构造；机构异名对仅 **55/1128** 对零共享字且**全部**为「学院/中学 ↔ 大學」跨后缀风格 ⇒ 纳入会把传统「大學」字面变成标签线索 ⇒ **不纳入并明说**（数据中 org DIFF-N2 = 0 条）。

**冻结实况（实测打印，`logs/train_full.log`）**：核总参数 **1,688,460**，`requires_grad=True` 的参数 **0 个** —— embedding 1,048,576（F=1）/ blocks 639,756（F=24）/ norm 128（F=1）；样例 `embedding.weight=False`、`blocks.0.ln_1.weight=False`、`blocks.0.attn.qkv.weight=False`…；`enc.training=False`、`blocks[2].training=False`、`norm.training=False`。唯一可训练 = 核外头 **230,146** 参数（`Linear(896,256)→GELU→Linear(256,2)`）。
特征 3 臂 × 2 split = 6 份 × 896 维；截断 0、span 丢弃 0；**hook 的 norm 输出与 enc() 返回值 `torch.equal` 对账 ✓**。

## 2. 主表（臂 × 测试集 × 2 seed；n=1200 ⇒ SE=1.443pt；卡 − `max_naive`）

| 训练臂 | X test | X-ir test | X-cross test | 自臂 `max_naive`（取胜规则） | 卡 − `max_naive` |
|---|---|---|---|---|---|
| **X** | **1.0000 / 1.0000** | 0.9558 / 0.9675 | 1.0000 / 1.0000 | 1.0000（n_cue） | **+0.0000 / +0.0000** |
| **X-ir** | 0.9242 / 0.8867 | **1.0000 / 1.0000** | 0.9225 / 0.9000 | 1.0000（n_cue） | **+0.0000 / +0.0000** |
| **X-cross** | 1.0000 / 0.9992 | 0.9958 / 0.9983 | **1.0000 / 1.0000** | 1.0000（n_cue） | **+0.0000 / +0.0000** |

**地板逐条（fit=train 的 test acc；X / X-ir / X-cross）**：R-same 75.0/50.0/75.0、R-edit 87.5/50.0/87.5、R-ngram 86.7/50.0/84.5、R-lcs 87.5/50.0/87.5、majority 50.0/50.0/50.0、length 52.1/51.8/51.8、structure 50.0/62.5/50.0、**n_cue 100.0/100.0/100.0**、R-pron 75.0/62.5/75.0；披露项 n_cue@字距桶 100/100/100、n_cue@句位 50/62.5/50。
**卡 − 每条（X 臂，两 seed 同值）**：R-same +0.25、R-edit +0.125、R-ngram +0.133、R-lcs +0.125、majority +0.500、length +0.479、structure +0.500、**n_cue +0.000**、R-pron +0.250。

## 3. E1–E4 + E6（逐条实测）与判定

| 判据 | 实测 | 结论 |
|---|---|---|
| **E1（主）** | X test = 1.0000 / 1.0000（seed 42/43），门槛 = 100.000 + 2×1.443pt = **102.886%**；margin = **0.0000 / 0.0000（两 seed 同号，但不 >）**；X-ir、X-cross 同构 100.000 vs 102.886 | **未过**（地板占满，判据无余量；非卡低分） |
| **E2 负例不塌** | 逐型 recall（n=300，SE=2.887pt）：A/B/C/D = **1.0000 / 1.0000 全不塌**；X-ir 逐子型 SAME-B / DIFF-N1 / DIFF-N2 = 1.0000（该型地板 1.0000，门槛 104.1/104.7/108.2%） | recall 意图**过**；字面门 **未过**（单标签子集上恒值规则即 100%） |
| **E3 随机标签** | 0.4725 / 0.4892 ≤ majority 0.5 + 2SE = **0.52886** | **过** |
| **E4 老卡** | Δ = **0.000000**（pronoun/sentiment/relation/person），带 2.83/0.41/1.39/0.33 pt 全在带内、指标 dict 逐位同；`base_encoder.pt`+四卡 **sha256 训前 = 训后** ⇒ core_drift = 0 | **过** |
| **E6 X-cross / X-ir（真靶子）** | 自训：X-cross **1.0000/1.0000**、X-ir **1.0000/1.0000**（盲猜线 0.52886）；X 臂训：cross 1.0000/1.0000、ir 0.9558/0.9675 | 远离盲猜 ⇒ **不判「只记住实体」**；但地板同为 100% ⇒ 亦无增量可言 |

**判定（三选一）：证据不足。** E1 未过（卡=100%，地板=100%，margin 0 ≤ 2SE）；E6 两臂**不是**盲猜 ⇒ 既非「有真能力」也不属「只记住实体」。**不硬选。**
**机制（实测）**：**提及位置 / 字距与标签完全混同** —— X test 字距桶：DIFF 全部落在桶 3（12–15 字），SAME 全部落在桶 6–7（24–31 字），X-ir / X-cross 同构 ⇒ 单条计数规则 `n_cue@字距桶` 三臂均 **100%**；而 PREREG 自带 `structure`（句距）只 50–62.5%、四条字面规则 ≤87.5%（P12a 据此判 E0 通过）⇒ 该混同**须按 D2 加入计数类特征才被地板捕获**。与 P12c `REPORT_disclosed.md` 实测同根因（该单元以 `R-ctxwin` 后窗共享 = 100% 报出「DIFF 的 m2 恒在第二子句、SAME 的 m2 恒在尾子句」），本单元用计数形式**独立复现**（字距 = 子句槽位的单调函数）；两单元结论一致 ⇒ E1 无判别余量。
**边界句**：本单元只回答「能否判定同一性 + 是否泛化到未训实体 / 零共享字」，**不回答因果**（核全程冻结，动核未测）。

## 4. 命令 / unit / 日志 / 遗留

- 命令（`uv run python`，全部 exit 0）：`gen_arms` → `check_arms`(46/46) → `eval_old_cards --tag before` → `train_arm --smoke` → **systemd-run 全量训练** → `battery` → `ecriteria` → `eval_old_cards --tag after` → `cards`。
- 训练 unit：**`dtseek-p12b-train.service`**（invocation `76373bfb5ae141e78881cc3ed00f0095`；`systemd-run --user --unit=… --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1`；8 runs = 3 臂 × seed 42/43 + X × 2 `--randlabel`，各 1000 步；GPU 已被 fix_negation/core_arch 占用，**排队并行、未杀任何他人进程**）。
- 日志：`logs/{gen_arms,check_arms,e4_before,train_smoke,train_full,battery,ecriteria,cards,e4_after}.log`；产物：`data/{xir,xcross}_{train,test}.jsonl`、`data/{arms_meta,arms_assertions}.json`、`features/*.npz`、`results/{battery,train_results,ecriteria,cards,old_cards_*}.json`。
- 契约记录（实测）：N01 门禁 `rule_a_problems=[]`、`slot_schema_problems=[]`，官方表 `validate_skeleton_table=[]`（本单元表规模 30）；三臂 test 共 **3600/3600** 条经 `to_contract()` 后 `contract_problems=[]`（`kind=text`）；拒答演示 2 例 `kind=reject` 且 `to_dict()` **无 `text` 键**、`reason` 非空、`evidence` 空。
- **N01 口径判断**：PREREG §2.1 的 pattern 字面 `[1]不[3][2]。` 与其渲染例 `张伟不是张伟。` + 槽位表（名施事/动/名受事）+ 全局 `assignment=(m1,verb,m2)` 三者不自洽（字面渲出 `张伟不张伟是。`，且全表其余 pattern 槽位引用均单调递增）⇒ 按渲染例实现为 `[1]不[2][3]。`，**判据不受影响**（N01 只承载 DIFF 标签）。
- **遗留-实测**：E5 解释层、OOD、免费规则穷举按范围**未做**；`src/` 未改、既有实验未动、**未 git 提交**（D8）。
- **遗留-推断**：地板被「提及字距」占满 ⇒ 本单元数据**无法**区分卡的判定能力与计数规则；要得到 E1 结论必须换/改数据构造（是否重建 X 臂由主代理决定）。
