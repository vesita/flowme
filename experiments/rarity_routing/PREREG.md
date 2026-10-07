# P22 PREREG —— 把「降维路由」从按句式推广为按稀有度（极小单元）

**本文件 mtime 早于首次运行 `run_eval.py`；跑后判据与定义一字未改。**
口径：**实测** = 本目录数字；**推断** = 由数字推出。**只读**复用 P8（`experiments/mode_conditioned_skel/`）、
`skeleton_leak/`、`free_rule_floor/`、`two_channel_head/` 的数据、权重与缓存；**不改 `src/`、不改既有实验、
不 git commit/stash/checkout/restore/clean、不重训核、不训头**（⇒ 「老卡 Δ」不适用，报告中如实写 N/A）。
**边界（写死）**：只回答「路由信号能否替换句式 / 稀有度是否更通用」，**不回答「该不该上线上调度」**（决策留 Lead）。

## 0. 唯一变量 = 路由信号（解码掩码），模型与 logits 完全共享

主模型 **UP**（P8 主口径），seed ∈ {42, 43}；**同一份 logits 只换 keep 集合**；纯 CPU 优先。
出口：**a_bal（无泄露，主判据）** + test、adv2（污染，仅用于 L/非 L 分列与附报）。
复用（只读 import，不复制）：`mode_conditioned_skel/common.py::load_arm/encode_rows/label_tensors`、
`modes.py::skeleton_table/mode_of_sent/subsets_A/MASK_ALIAS`、`skeleton_leak/cache` 编码缓存、
`mode_conditioned_skel/results/rules.json`（免费规则，**只读引用、零新拟规则**）。

## 1. 四臂（唯一变量 = 路由信号）

| 臂 | keep（候选集） |
|---|---|
| **F** | 全 40 类，不掩码（基线，须先复现 P8） |
| **C-mode** | `sA[MASK_ALIAS[mode_of_sent(sent)]]` = P8 的 `C_pred`（句式**规则**标签；不重训分类器） |
| **C-rarity** | 见 §1.1 |
| **C-both** | `C-mode ∩ C-rarity`；空集 ⇒ 回退 F（次数必报） |

### 1.1 C-rarity 路由定义（原文，写死）

> 行级信号 = **`n_slots`**（袋跨度数，**输入侧可观测**；不含句式、不含 gold、不含任何标签）。
> 桶函数 `ns(sid)` = 该骨架在 **train** 中出现行的 `n_slots`（train 覆盖的 30 骨架内**确定性**；
> **train 未见的骨架不入任何桶**，fail-closed）。
> 候选集 **`S(ns) = { sid : ns(sid) = ns }`** —— 即任务书示例 (a)「按 `n_slots` 计数在该桶内的骨架数」。
> **稀有度 / 歧义度 = `|S(ns)|`**（候选集大小）；空桶 ⇒ 回退 F（次数必报）。
> **train 未见骨架被常量排除（与行无关）** ⇒ 该成分在报告中**单列为常量成分**，不冒充行级路由效果。

**选定义的时机（纪律）**：只用 train 与子集结构勘察选定义，**未看 a_bal 任何 acc**；不挑多个定义择优。
**R5 随机对照口径**：行级 `randperm`，`torch.Generator().manual_seed(seed*1000+99)`（P8/skeleton_leak 同口径）。

### 1.2 子集大小预期（结构，非结果）

- C-mode：行数 陈述/疑问/祈使 = 880/80/80；骨架子集 35/3/2（a_bal 内有效 32/2/2）。
- C-rarity：行数 ns=2/ns=1/ns=3 = 920/80/40；骨架子集 ≈24/3/3（**实际分布按 R4 必报**）。
- 二者都是「一个大桶 + 两个小桶」的三分 ⇒ 可比性按 R4 实报，不预设相当。

## 2. 判据 R0–R5（跑前写死）

| # | 判据 | 门槛 |
|---|---|---|
| **R0** | F 与 C-mode 复现 P8（a_bal, UP, 2 seed） | F = **.1567/.1587**、C-mode = **.2606/.2673**，**各在 1×SE(=.0155) 内**；否则先修探针，**不改判据** |
| **R1（主）** | C-rarity > F（a_bal） | 配对 **Δ > 2×配对SE** 且 **2 seed 同号** |
| **R2** | C-rarity vs C-mode | 报 Δ±配对SE、t；**\|Δ\| ≤ 1×配对SE ⇒ 「相当」**；**Δ < −2×配对SE ⇒ 「明显更差」** |
| **R3** | 卡 − 条件化查表（**N2 口径**，fit=train，**只读 P8 `rules.json`**）：R0`n_slots`=.0000、R1`(句式,n_slots)`=.0769、R2`句式`=.0385、max_naive_complete=**.1404**、**最强组合 `n_slots+last_char` = .1615** | 四臂各报；**主判 C-rarity 卡 − .1615 > 0**；**≤0 ⇒ 如实写「收益被免费规则解释」** |
| **R4** | 子集大小可比性 | 必报每桶 \|S\| 与行数、回退次数；与 C-mode 差距明显 ⇒ 说明 |
| **R5** | **L 组 / 非 L 组分列**（a_bal·L **n=0** 如实写；test/adv2 分列）+ **随机标签对照** | 必报；随机对照须 < C-rarity（否则 R1 归因不成立） |

**判定三选一**：① **稀有度路由更通用**（R1 过 ∧ R2 相当或更好）／② **句式是关键**（R1 过但 R2 明显更差）／
③ **证据不足**（R0 或 R1 未过）。**不许硬选。**
**铁律**：并列 `max_naive`；**区分实测/推断**；**不许**把「没测出差异」写成「没有效果」。
随机对照只作方向性对照，**不新增事后门槛**（写死）。
**老卡 Δ**：本单元不训任何头 ⇒ **N/A**（如实写，不补训）。
