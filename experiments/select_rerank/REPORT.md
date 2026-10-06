# select_rerank 结果报告 —— 方案 B：上下文一次 + 每候选一次 + 共享打分头

> 判据与配方见 `PREREG.md`（mtime **20:20:09**，早于首训 **20:20:27**，跑后未改）。
> 「实测」= 本目录产物直接读出；「推断」= 由实测推出；「口径判断」= 构造时人为规定。

## 1. 重排器（`model.py`）

| 项 | 实测 |
|---|---|
| 结构 | 冻结核 `NanoDocEncoder`（hidden 128 / 3 层 / vocab 8192 / max_len 128）各编码 1 次 context 与 2 次候选（候选**不**拼进上下文、**不**发射锚点）→ masked mean-pool → 共享 MLP 头 `[ctx; cand; ctx⊙cand; \|ctx−cand\|]` → softmax over K=2 + CE |
| 参数量 | 核 **1,688,460**（`encoder_trainable=0`、`encoder_training=false`，每跑打印）+ 头 **164,353** = 1,852,813 |
| 候选等变 | 自检：交换两候选 ⇒ 两个 logit 同步交换（allclose ≤1e-6）⇒ 下标不是特征 |
| 截断口径 | ctx 64 / cand 32，`RerankSpec.from_build_spec()` 直读 `build_data.py::SPEC`，两处不漂移 |

## 2. 数据（`build_data.py` → `data/` + `data/stats.json`）

- 规模：train 8000 / test 2500 / adv 2500 × 2 臂 = **26,000 行**；正负 1:1 与位置 50/50 均为构建期断言。
- 两臂：**S**（有捷径）train/test = 逐字片段 vs 无关片段；**N**（干净）= 同义替换 vs 反义替换（与原词等长、`share(w,syn)==share(w,ant)` 对称）。
- adv **两臂同构造**，各半：`heldout_pair`（留出**替换对**，两个输出串与训练词表零交集）/ `shifted_pos`（表内词、替换位挪到片段尾部）。
- 不相交（实测）：15 对集合的 key/context/candidate 重叠**全 0**，跨字段交叉也 0；7 源句池两两 0；候选串全局唯一 26,000。
- 截断守卫（逐样本实际 encode 比对未截断全长）：ctx/cand 丢弃 **0**（ctx_max 44≤64、cand_max 24≤32）。
- 词表：95 训练对 / 53 留出对；留出输出 106 串 ∩ 训练 270 串 = **0**；共享字不对称对 = **0**。
- 零训练表面基线（`stats.json`）：S-train/test 逐字 **1.0000**；N 三集逐字 **0.5000**、长度 **0.5000**、共享字 **0.5000**；上下文字重叠 0.509~0.517（≤1.7pt，低于 2×SE 门槛）；`clean_train_rule_ge_90 = false`（N 臂数据有效）。

## 3. 主表（准确率 %，chance 50%，SE = sqrt(0.25/n)：n=8000→0.559pt、n=2500→1.00pt）

| 臂 / 集 | seed 42 | seed 43 | 余量/SE（42/43） | 随机标签 42 | 随机标签 43 | 门槛 |
|---|---|---|---|---|---|---|
| **N** train | 100.00 | 100.00 | +89.4 / +89.4 | 49.55 | 49.60 | 参考 |
| **N test** | **94.48** | **94.48** | **+44.5 / +44.5** | 49.20 | 48.32 | P1 >52.00 → **过** |
| **N adv** | **71.04** | **71.76** | **+21.0 / +21.8** | 50.76 | 50.36 | P2 >52.00 → **过** |
| S train | 100.00 | 100.00 | +89.4 / +89.4 | 49.38 | 48.78 | 参考 |
| S test | 98.04 | 98.04 | +48.0 / +48.0 | **52.08** | 50.36 | P4 → 过 |
| S adv | 51.52 | 53.48 | +1.5 / +3.5 | 48.92 | 50.96 | 两 seed **不同号** → 不算过 |

- **P3（随机标签对照，约束臂 = N）**：N 的 test 49.20 / 48.32、adv 50.76 / 50.36，**全部 ≤52.00% → 过**，判据不失效。
- **P4（S 臂）**：test 两 seed 过线；adv 一过（53.48）一不过（51.52）⇒ 按 PREREG「两 seed 同号」口径 **S-adv 不算过线**。
- **P5**：核全程冻结、未走温启动联合 ⇒ **N/A**。
- 附报：4 个随机标签跑里 `shortcut_s42_rand` 的 test = 52.08%（超线 0.08pt，单次 2×SE 检验约 2.3% 假阳率，4 跑内出现 1 次属预期量级）。

### 3.1 adv 按 ctype 拆分（`analyze.py` → `results/breakdown.json`，判据未覆盖、但决定解读）

| N 臂 adv | n | seed 42 | seed 43 | 余量/SE | 过 2×SE |
|---|---|---|---|---|---|
| `shifted_pos`（**表内词** + 换位置） | 1250 | **92.56** | **92.48** | +30.1 / +30.0 | 是 |
| `heldout_pair`（**留出替换对** = 换说法） | 1250 | **49.52** | **51.04** | −0.3 / +0.7 | **否** |

- 逐词：heldout 在 n≥15 的 **22/22 个词上两 seed 一致（\|Δ\|<0.15）**，分布**两极**（min 0.00 / 中位 0.49 / max 1.00，仅 27% 落在 [0.4,0.6]）：出现 1.00/0.98、稳定 0.94/0.94、真实 0.81/0.85 ｜ 开始 **0.14/0.16**、记得 **0.19/0.11**、支持 0.18/0.23、高兴（喜悦 vs 痛苦）**0.25/0.08**。
- `shifted_pos` 逐词：n≥30 的词**无一低于 0.75**（可能 0.99、开始 1.00、帮助 1.00、支持 0.80）。

## 4. 判定与机制

- **按 PREREG 字面：判定 = ①「选择能力存在」**（P1 ∧ P2 ∧ P3 两 seed 全过）。
- **但 P2 是混合指标**（判据没把 adv 按 ctype 拆开 —— 判据缺陷，跑后**未回改**）：71% 全部由 `shifted_pos`（92.5%）撑起，**换说法 `heldout_pair` = 49.5/51.0%（两 seed 合并 n=2500 → 50.3%，+0.6 SE）与盲猜不可区分**。
- **机制（实测）**：头学到的是**表内每个替换对的固定偏好**，且**位置不变**（表内词换到尾部仍 92.5%）；对**从未见过的替换对**则系统性偏到某一边（逐词可复现的错，不是噪声）。**推断**：test 94.48% 与 shifted 92.5% 同源于这张「表」，而非语义等价判断 —— 即本核 + mean-pool + 共享 MLP 头下，「选择」**能学会、但不迁移到新说法**。
- **若把 P2 按其设计意图读作 heldout 子集，则判定落到 ③「证据不足」**；两种读法并列报出，不硬选、不调参补跑。
- **与 A 族（`anchored_select`）的差别**：A = 单次共享编码 + **发射锚点**，锚点双命中 0.75% ⇒ train 只有 68.9/69.2%（发不出去）；B **不发射锚点** ⇒ train 100%、test 94.48%，**A 的失败确在锚点环节**。两族共同点：**「换个说法」都 ≈50%**（A-adv 52.5/53.8% 是「改写 vs 无关」构造，本族 heldout 是「同义 vs 反义」，口径更严）。

## 5. 原始命令 / 单位 / 日志 / 产物

```bash
uv run python experiments/select_rerank/build_data.py --probe          # 量命中率 → 池子建议
uv run python experiments/select_rerank/build_data.py                  # 产出 data/ + stats.json（日志见下）
uv run python experiments/select_rerank/model.py --selfcheck           # 冻结 + 等变自检
systemd-run --user --unit=dtseek-select-rerank --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  bash experiments/select_rerank/run_all.sh                            # 8 跑顺序执行，20:20:27→20:21:56
uv run python experiments/select_rerank/analyze.py                     # 按 ctype/逐词拆分
```

- 单位：`dtseek-select-rerank.service`（invocation `aacc7cd6d89f42a89cce22a75fb12a24`，已 `ALL DONE`，状态 inactive）。
- 日志：`logs/select_rerank_build_data.log`、`logs/select_rerank_{clean,shortcut}_s{42,43}[_rand].log`（8 个，逐跑 `[freeze]/[train]/[eval]` 全在）。
- 产物：`data/{shortcut,clean}/{train,test,adv}.jsonl`、`data/stats.json`、`results/*.json`（8 跑）+ `results/breakdown.json`、`weights/*.pt`（8 个头）、`cache/*.pt`（冻结核向量，按数据 md5+SPEC 键控）。
- 纪律：只写 `experiments/select_rerank/` 与 `logs/select_rerank_*`；无 `git commit/stash/checkout/restore/clean`；开训前查过 `ps` 与 `systemctl --user is-active 'dtseek*'`（均空），未杀任何进程；与另一子代理的目录占用冲突已在 collab board（m_123/m_125/m_130）协商为「它建数据、我建模型与训练」，实际数据仍由我出（见 m_130）。

## 6. 遗留与不确定（实测 / 推断分开）

1. 【实测】PREREG 的 P2 未要求按 ctype 拆 ⇒ 字面判定① 与机制结论不一致，已在 §4 并列报出；**判据跑后未改**。
2. 【实测】heldout 逐词 n 只有 12~211，**单词结论弱于合计结论**；但两 seed 22/22 一致支持「系统性偏差」而非噪声。
3. 【实测】`ctx_char_overlap_rule` 残余 0.509~0.517：是数据里唯一非零的表面偏置，最高只值 1.7pt，低于 2×SE 门槛。
4. 【推断】mean-pool 把 8~24 字片段压成一个向量，**2 字之差被稀释**，可能是 heldout 失败的结构性原因 —— **未做消融**（不许调参补跑）。
5. 【推断】「表内偏好」是行为学解释（由 shifted 高 + heldout 50% + 逐词可复现三条实测推出），没有直接打开头内部的证据。
6. 【实测】只训打分头、2 个 seed、1800 步、未调参；P5 未走（N/A）。**下步若要判语义能力，需在 PREREG 里把 P2 直接写成 `heldout_pair` 子集口径。**
