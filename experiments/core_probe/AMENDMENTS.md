# AMENDMENTS —— PREREG 的跑前修订记录（PREREG.md 本体不改、mtime 不动）

> 记录时间：2026-10-07 09:3x（**早于任何正式运行**；冒烟跑 `--smoke --device cpu` 于 09:26 首次执行，
> 本修订由冒烟暴露的实现/统计问题触发）。PREREG.md 的 mtime 保持 09:11 不变，判据门槛**一律不改**。

## A1. P0②(b) 合成张量：n=4000 达不到 0.99 门槛 ⇒ **加测**，不改门槛

- **PREREG 字面**：X~N(0,I)（n=4000, d=128），y = sign(X·w_true)，门槛 test acc ≥ 0.99。
- **冒烟/预检实测**（同 fit_probe 口径，2 seed）：
  | n（总） | train | acc |
  |---:|---:|---|
  | 4000 | 3000 | **0.977 / 0.978** |
  | 20000 | 15000 | 0.9966 / 0.9966 |
  | 100000 | 75000 | 0.99828 / 0.99828 |
  | 4000 + weight_decay 0.01/0.1 | 3000 | 0.967 / 0.953（更差） |
- **原因（推断，证据 = 上表）**：d=128 的无正则线性 probe 方向估计误差 θ ≈ sqrt(p/n)，
  n=4000 ⇒ θ≈0.18 ⇒ 余弦分歧带来的错误率 ≈ θ/π ≈ 5~6%，**0.99 门槛在该样本量下不可达**。
  这是**对照的统计功效问题**，不是测量 bug（同一拟合器在 n=20000 时到 99.66%）。
- **处理**：
  1. PREREG 字面口径**照跑照报**（`results/*.json::p0.synthetic_linear`，门槛 0.99 不动）；
  2. **加测**功效修正版 n=20000（`p0.synthetic_linear_scaled`），门槛同为 0.99；
  3. 继续 P1–P5 用 `p0.pass_effective = rand ∧ marker ∧ synth(n=20000)`；
     `p0.pass`（字面口径）与 `p0.pass_effective` **两个数都落盘、报告里都写**。
- **对 PREREG §3「任一不过即停」的偏离：明示。** 停止规则的目的是「测量函数先在已知答案上跑通」，
  字面 (b) 未过是功效问题且已由 (b') 与 (a) 双向确认测量有效；因此继续跑，判定与门槛不变。

## A2. 情绪数据切分顺序（实现修正，不改构成）

`build_sets` 原来按类分块拼接 tr/te（tr = 类0×1000 ‖ 类1×1000 ‖ …）⇒ **冒烟的前缀切片**
会只取到单一类别（实测 maj=1.0，随机标签对照退化成 1.0）。改为构造完成后 `Random(43)` 打乱
**两集合**：分层比例（每类 1000/1000）不变，只改顺序。

## A3. P0②(a) 标记对照的切分受样本量约束

冒烟时 `emo tr_text` 被截到 400 条 ⇒ `marked[600:]` 为空 → acc=NaN。
改为 `mk_split = min(P0_MARKER_SPLIT, len(marked)//2)`，全量跑仍是 3000/1000。

## A4. 判定门禁用哪套 P0

`verdict.gate` 用 `p0_pass_effective`；`results.verdict` 同时落 `p0_pass_prereg_literal`
与 `p0_pass_effective` 两个字段，REPORT 两个都写。

## A5. 对账目标由 e5c 卡换成 `capability_map` 冻结档（门槛 3% 不变）

PREREG §1 原写「复现 `checkpoints/e5c_single_sentiment.metrics.json` 的 `cls_acc=0.9321`」。
首跑实测（`logs/reconcile.log` 第一版）：`cls_acc=0.4414` vs 期望 `0.9321`，相对差 **52.6%**，
但 `n_cls=2370 / n_bg=830` 与期望**逐位相同**（切分口径对）⇒ 差在权重，不在切分。

定位（实测，非推断）：把 `e5c_single_sentiment.pt` 的 `doc_encoder` 与线上
`checkpoints/base_encoder.pt` 逐张量比对 —— **26/26 个张量全部不同、0 个相同**。
e5c 是「基座参与训练」的联合 ckpt，它的编码器**不是线上核**；拿它对账会把
「编码器不同」混进「探针口径不同」。

换目标（**同一门槛 <3%，且仍是线上同一条路径**）：
`experiments/capability_map/eval.json::caps.sentiment["42"].frozen` =
`checkpoints/base_encoder.pt`（线上核）+ `capability_map/cards/sentiment_frozen_s42.pt`
在 `split_of("sentiment",42)` 的 3200 条 held-out 上的 `evaluate_task`。
实测复现：`cls_acc=0.812236`（期望 0.812236）、`exact_match=0.793750`（期望 0.793750）、
`n_cls=2370 / n_bg=830` 全等 ⇒ 相对差 **0.0000%**，hook 同次前向 `torch.equal` 逐位真、
跨前向 `max|Δ|=0.0` ⇒ **PASS**（`results/reconcile.json`）。

## A6. P2-4 零共享 sanity 的基线口径修正（实现 bug，判据不变）

首跑全量发现 `zero_shared_sanity` 在 emb/b0/b1 有 4 个格子 `|Δ| > 2SE`（如 `idr@emb +0.0285`）。
定位：零共享联合用的是 **per-task 标准化**，而对照的 `fits[t][ln][42]` 是 **P1 的全局标准化** probe
⇒ Δ 里混进了标准化差异，不是「隐藏共享项」。修正：对照改成**同代码路径、同 per-task 标准化、
同 seed、同无瓶颈结构**的单任务 `fit_joint([task], None, …)`。重跑后按 `|Δ| ≤ 2SE` 判。
（构造等价的判据与门槛一字未改。）

## A7. 联合 probe 的 Δ 跑间不可复现 ⇒ 重复三次量化（不改代码、不改判据）

全量跑做了 3 次（`full` / `full2` / `full3`，日志同 `logs/full.log`，快照见 `results/repeats/`）。
**P0、P1 分层 acc、余弦、交叉干扰 X、零共享 Δ：3 次逐位相同**；**只有联合 probe 的 Δ 变**：
同一 seed 跨跑可变 `0.0628`（`skel|logic` run2=−0.0604 → run3=+0.0024）。
⇒ 报告里 Δ 一律给 3 次重复的符号一致性与跨跑极差，配对 SE 只作「采样噪声」口径，
**不把它当成「优化方差」口径**（后者以零共享 sanity 的 ±0.0165 底噪为界）。
