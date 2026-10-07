# REPORT —— P4 功能词最小对训练信号（`experiments/funcword_minpair/`）

## 0. 判定（三选一，按 `PREREG.md` §4 写死判据落地）
**无效（仍不读功能词）** —— 条款 2 五条全中：P0 过 ∧ P6 过 ∧ 两 seed P1 主判均未过 ∧
per-side acc 两 seed 均 ≤ 0.5310 ∧ 两 seed 同判。
**但必须同读 §9「不确定（实测）」：同口径伴指标 `pair_2afc` 方向相反，本判定只在
`pair_success`（非受限 40 类 argmax）这一个出口内成立，其机制解释证据不足。**

范围限定（PREREG §4 写死）：核冻结、只训头、1800 步、seed 42/43、λ=1 口径内，不外推。

## 1. 口径与自检（跑前写死 + 实测）
| 项 | 实测 |
|---|---|
| PREREG mtime | 11:37:18 < 首次训练 11:43:40（先落 PREREG 再开跑） |
| 核冻结 / 可训参数 | encoder 1,688,460 trainable=0、training=false；可训只有 trunk+gen = 186,088 |
| 三臂初值 / M loss | 三臂 `torch.equal` 且 == `StructSupModel("B", seed)`；M loss vs `struct_supervision::gen_loss` max\|Δ\| = **0.0** |
| 配方 | 1800 步 / batch 64 / AdamW lr 1e-3 / wd 1e-4 / cosine / clip 1.0 / seed 42,43；冒烟 60 步 ×3 臂全通，step0 parts 三臂同值（assign .9757 / pair .6837） |

## 2. 最小对构造与分离（`data/stats_build.json`）
- 枚举 **R1 = 62、R2 = 21、R3 = 1（#6 `因为[1]，[2]` ↔ #7 `[1]，因为[2]`）**，与 PREREG 写死值一致。
- 覆盖：候选骨架 24/30、候选行 **4274/8000 = 53.43%**；fail-closed 后 **word 伙伴 4205 = 52.56%**、R3 伙伴 **83/91**；
  未覆盖骨架 {#1 2087, #2 1165, #4 224, #3 153, #5 86, #31 11} = 3726 行。
- 失败分类（尝试 4807，**未命中 519 = 10.80%**）：`ok_sentence拒` 402、错骨架 104（#1 56 / #2 16 / #23 7 / #14 6 /
  #31 4 / #15 3 / #38 2 / #12 2 / #11 2 / #20 2 / #6 2 / #13 1 / #25 1）、槽文本不一致 9、无匹配 3；R3 另 8。
- **分离断言 7 条全 0**（train∩heldout、variant∩heldout、heldout∩(train∪test∪adv1∪adv2)、B1∩B2、
  b_pairs∩c_pairs、variant 内部重复、variant∩train）；构造脚本重跑 **sha256 逐字不变**。
- held-out：**B1 520 对**（26 方向 ×20）、**B2 40 对**（2 方向 ×20）；c_pairs 520 组（两侧袋不同 ⇒ 非最小对，
  只报骨架 acc）。语料 14 文件继承 `leak_stats.json::lang.files_kept`，本单元不重扫。

## 3. 主判 P1 + 主表（骨架 acc ± 行级 SE；卡−地板用完备 `max_naive_complete`）
| 臂 seed | B1 pair_success | 门槛 .5439 | B1 pair_2afc | per-side | B2 pair_success | a_bal | 卡−地板 | test | 卡−地板 |
|---|---|---|---|---|---|---|---|---|---|
| M s42 | 0.0000±.0219 | 否 | .1615 | .0288 | 0.0000±.0791 | .0635±.0155 | −.0769 | .6396±.0100 | −.1920 |
| M s43 | 0.0000±.0219 | 否 | .1904 | .0288 | 0.0000±.0791 | .0663±.0155 | −.0741 | .6424±.0100 | −.1892 |
| MP s42 | 0.0000±.0219 | 否 | **.6212** | .0000 | 0.0000±.0791 | .0000±.0155 | −.1404 | .2644±.0100 | −.5672 |
| MP s43 | 0.0000±.0219 | 否 | **.6096** | .0000 | 0.0000±.0791 | .0000±.0155 | −.1404 | .2584±.0100 | −.5732 |
| MP+ s42 | 0.0192±.0219 | 否 | .5404 | .0510 | 0.0250±.0791 | .1183±.0155 | −.0221 | .6092±.0100 | −.2224 |
| MP+ s43 | 0.0173±.0219 | 否 | .5404 | .0548 | 0.0000±.0791 | .1096±.0155 | −.0308 | .6056±.0100 | −.2260 |

- **P0 复现：Δ 全为 0.0000**（test .6396/.6424、a_bal .0635/.0663，对照 `struct_supervision` B /
  `bag_modules` A；2SE = .0200 / .0310）⇒ 过，探针无需修。
- 其余集：a_lit M .1283/.1308、MP .0350/.0408、MP+ .1692/.1650；b_pairs M .0420、MP .0000、MP+ .0946/.0991；
  c_pairs M .0712、MP .0000、MP+ .1240/.1135。
- **配对 SE（逐 0/1 差）**：MP−M test −.3752/−.3840（t −29.0/−30.0）、a_bal −.0635/−.0663（t −8.39/−8.59）、
  B1 ps Δ 0/0；MP+−M test −.0304/−.0368（t −3.87/−4.68）、a_bal +.0548/+.0433（t 7.02/6.06）、
  **B1 ps +.0192/+.0173（t 3.19/3.02，2 seed 同号）**、B2 +.0250/0（**异号**）。

## 4. P3 L 分列（L = {#35,#0,#1,#2} = `n_slots→train多数` 全部输出，已断言；`a_bal` 构造性不含 L）
- **test**（L n=2079 / 非L n=421）：M .7292→.7340 / .1971→.1900；
  MP .3170→.3107 / .0048→.0000；MP+ .6801→.6724 / .2589→.2755。
- **a_lit**（L n=160 / 非L n=1040）：M .5500 / .5500 vs .0635→.0663；
  MP .2625→.3063 / .0000→.0000；MP+ .5000→.5250 / .1183→.1096。
- **b_pairs**（L 对 n=40 / 非L 对 n=520，`ps`=pair_success、`side`=per-side）：
  M ps 全 0（side L .1500→.1250、非L .0173→.0192）；MP ps/side 全 0；
  MP+ ps L .0500→.0250、非L .0173→.0154（side L .0875→.1125、非L .0471→.0500）。
- MP+ 非 L 组高于 M（test +.0618 / +.0855，配对显著），但**两组都远未到主判门槛**。

## 5. P4 机制探针（每臂每 seed；`results/eval_*.json::probe`）
| 臂 seed | test 真→只打乱内容 | a_bal 真→只打乱内容 | b_pairs 真→只打乱内容 | shuffle_all test/a_bal |
|---|---|---|---|---|
| M s42 / s43 | .6405→.6401 / .6433→.6433 | .0637→.0618 / .0666→.0656 | .0420→.0403 / .0420→.0411 | .3180/.0019 · .2852/.0067 |
| MP s42 / s43 | .2645→.2645 / .2589→.2585 | .0000→.0000 / .0000→.0000 | .0000→.0000 / .0000→.0000 | .2588/.0000 · .2380/.0000 |
| MP+ s42 / s43 | .6104→.6096 / .6068→.6064 | .1187→.1168 / .1100→.1071 | .0948→.0921 / .0993→.0975 | .2808/.0058 · .2588/.0106 |

- **`shuffle_content` 不掉**（Δ ≤ .0029，同批幸存行配对；幸存 2495/2500、1036/1040、1118/1120，失败逐原因分类）
  ⇒ 三臂都**读结构字面、不读内容词**。
- **`shuffle_all` 6/6 全部落到门槛以下**（test ≤ .4368、a_bal ≤ .0695）且**低于多数类**而非等于 ⇒ 报「已落回」。
- **`pair_shuffle_content`**（两侧同用另一对槽文本：功能词差异保留、内容换掉；559/560 对幸存）：
  `pair_success` M 0→0、MP 0→0、MP+ .0197→.0179 / .0161→.0161（**塌到地板**）；
  **`pair_2afc` M .1628→.1628、MP .5850→.5868、MP+ .5116→.5027**（机会 .25）。
- **`aux_only`：本模型无标签通道 ⇒ 原口径结构上不适用（如实标、不硬套）**；替代 = 只吃 `n_slots` 的查表读出
  （≡ `n_slots→train多数`，模型无关常数）：test .8316、a_bal .0000、a_lit .1333、b_pairs .0714、c_pairs .0769。

## 6. P5 完备免费地图（fit = train；逐规则 acc 已断言 == `naive_skeleton`，且与 leak_stats 一致）
- **test .8316（`n_slots_rule`）／ a_bal .1404（`tree_depth4`）／ a_lit .1900（`tree_depth4`）／
  b_pairs .1786（`first_char`）／ c_pairs .1788（`tree_depth4`）** —— 5/5 与 `leak_stats` 公布值一致。
- **pair 地板**：b_pairs **.0714（`first_char`）**；`n_slots` 两侧同桶同预测 ⇒ **pair_success ≡ 0**（PREREG 预期）。
- 卡−地板：test M −.1920 / MP −.5672 / MP+ −.2224；a_bal M −.0769 / MP −.1404 / **MP+ −.0221（最好）**；
  三臂在 test / a_bal 上**全部低于**完备地板。

## 7. P6 随机标签负对照（真执行，`--randlabel`，门禁）
类级 randperm 前后计数已打印（s42：`[3335,2087,1165,153,224,86,…] → [167,28,42,68,11,70,…]`，
**分布不变 = True**），每行指派值行内 randperm，评测用真标签。
**6/6 全过**：test .0012–.0088 ≤ .4368；a_bal .0260–.0490 ≤ .0695；B1 `pair_success` 三臂两 seed 均 **0.0000**。

## 8. 判定逐条（PREREG §4）
- **P0 过**（Δ = 0.0000、两 seed、≤ 2SE）；**P2 数字已报**；**P3 已 test/a_lit/b_pairs 三处分列**；
  **P4 四探针 × 臂 × seed 已报**；**P5 完备 + pair 地板已报**；**P6 过**。
- **P1 未过**：M 0.0000/0.0000、MP 0.0000/0.0000、MP+ .0192/.0173，门槛 **.5439**（SE .0219），两 seed 同号。
- **per-side**：M .0288/.0288、MP .0000/.0000、MP+ .0510/.0548，门槛 **.5310** ⇒ 两 seed 均未过。
- ⇒ 落**条款 2「无效」**；条款 3 的触发器「P1 未过但 per-side > 0.5+2SE」**未触发**。

## 9. 遗留（实测）与不确定（实测 / 推断分列）
**遗留（实测）**：① MP 臂骨架标签覆盖仅 52.56%（无伙伴行零梯度）⇒ 绝对校准丢失：test .2644/.2584、
a_bal .0000、`pair_pred_differs` .0154/.0481（对内几乎不换预测）；② b_pairs 仅 17 骨架、**无 n=3/n=4 桶**、
B2 仅 40 对（SE .0791）⇒ **MP-order 无统计力**，MP+ 的 B2 Δ 两 seed **异号** ⇒ 不下结论；
③ 同骨架槽序对未构造、`adv1` 不评测（PREREG §6 跑前声明）。
**不确定（实测）**：④ **伴指标方向相反** —— MP `pair_2afc` = .6212/.6096（机会 .25、n=520 ⇒ 远超），
**换内容后仍 .5868/.5725 保持**（M 仅 .1628/.1825，MP+ .5116/.5063）：限制到 {s,t} 空间里**两侧选对率显著
超机会、且与内容词无关**，与「仍不读功能词」字面相反；PREREG §3.1 写死「伴口径不改判」故不改判，但**必须一并读**；
⑤ 主判要求**非受限 40 类 argmax 同时命中**，MP 臂按构造无 40 类 CE、logits 从未绝对校准 ⇒ 该出口对 MP 臂**结构性不利**。
**不确定（推断）**：⑥ 超机会信号来自结构字面哪一维（功能词身份 / 字面长度 / 标点位置）**未拆开**
（`shuffle_content` 只能证明它不在内容词里）；⑦ 给 MP 臂加最小绝对校准或改配对一致出口是否翻盘 —— **未测，需新 prereg**。

## 10. 命令、unit、日志与产物
- **命令**（全部 `uv run python`，cwd = 仓库根）：`build_data.py` → `model.py --selfcheck` →
  `train.py --arm {M,MP,MP+} --seed {42,43} [--randlabel]` → `eval_arms.py --arm … --seed … [--randlabel]` →
  `analyze.py`；一键 `bash experiments/funcword_minpair/run_all.sh`（内含 60 步三臂冒烟门禁）。
- **unit**（`systemd-run --user --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1`，
  **绝无 `setsid nohup &`**）：`dtseek-fwmp-train`（12 次训练）、`dtseek-fwmp-eval`（12 次评测）；
  脚本 `run_train.sh` / `run_eval.sh` / `run_all.sh`；开跑前查过 `ps` 与 `systemctl --user list-units 'dtseek*'`（无活跃单元）。
- **日志** `experiments/funcword_minpair/logs/`：`build_data.log`、`selfcheck.log`、`smoke_{M,MP,MP+}.log`、
  `train_*.log` ×12、`eval_*.log` ×12、`train_runs.log`、`eval_runs.log`（含 exit 码）、`train.done`、`eval.done`、`analyze.log`。
- **产物**：`PREREG.md`、`REPORT.md`、`data/{train_pairs,train_order_pairs}.jsonl`、`data/stats_build.json`、
  `weights/*_s{42,43}[_rand].pt` ×12 + 冒烟 ×3、`results/eval_*.json` ×12、`results/tables.md`、`results/verdict.json`、
  `results/*.train.json` ×15、`cache/`（只写本目录）。
- **只读复用**：`two_channel_head/data|cache`、`skeleton_leak/data|cache|results/leak_stats.json`、
  `struct_supervision/model.py`；全表见 `results/tables.md`。
