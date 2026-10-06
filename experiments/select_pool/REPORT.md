# select_pool 结果报告 —— 四臂聚合消融：那堵墙在「池化」还是「表示」？

> 判据与配方见 `PREREG.md`（mtime **20:37:26**，早于首启 20:45:18 与首训 20:47:13，跑后未改）。
> 「实测」= 本目录产物直接读出；「推断」= 由实测推出；「口径判断」= 构造时人为规定。
> 数据**只读复用** `experiments/select_rerank/data/` 的 **clean（N）臂**（未写该目录一个字节）。

## 1. 四臂 + 对照臂：聚合方式、参数量、冻结实况（实测）

| 臂 | 聚合方式（padding 全 masked） | 聚合层参数 | 打分头 | 可训合计 | 核可训 | 核状态 |
|---|---|---|---|---|---|---|
| **A（基线）** | masked **mean-pool**（与 B 族逐位同式） | 0 | 164,353 | **164,353** | **0** | `eval()`、`requires_grad` 全 False |
| **B** | masked **max-pool** | 0 | 164,353 | **164,353** | 0 | 同上 |
| **C** | 可学 **attention-pool**（`q_ctx/q_cand∈R^128`，`softmax(q·h/√128)`） | 256 | 164,353 | **164,609** | 0 | 同上 |
| **D** | **cross-attention**：候选每个 token 对上下文 token 做单头注意力（raw dot × 可学温度）再按候选 mask mean；**ctx 侧仍 mean** | 1 | 164,353 | **164,354** | 0 | 同上 |
| **A+（Q5 对照）** | mean-pool + 可学逐维缩放 `g_ctx/g_cand`（与 C 参数量对齐） | 256 | 164,353 | **164,609** | 0 | 同上 |

- 核 **1,688,460**、打分头 **164,353** 与 B 族逐位同构（`model_pool.py` 自检断言）；**不发射锚点**。
- 自检 5 臂全过：冻结实况、候选等变（换候选 ⇒ logit 同步换 ≤1e-6）、聚合层有梯度、logits 非常数。
- **Q4a 口径同一性（跑前硬校验）**：标签与 B 族缓存**逐元素相等**；由 token 级缓存现算的 mean-pool 向量与 B 族 `v_ctx/v_cand` **max|Δ| = 0.000e+00（位级相同）** ⇒ 同一份数据、同一个核、同一套编码。

## 2. 主表（准确率 %；n=1250 ⇒ **SE = 1.414pt**，**2×SE = 2.828pt**；chance 50%）

| 臂 | seed | `heldout_pair` | 余量/SE | `shifted_pos` | adv 总 | test | train |
|---|---|---|---|---|---|---|---|
| **A（mean）** | 42 | **49.52** | −0.34 | **92.56** | 71.04 | 94.48 | 100.00 |
| **A（mean）** | 43 | **51.04** | +0.73 | **92.48** | 71.76 | 94.48 | 100.00 |
| B（max） | 42 | 52.32 | +1.64 | 85.36 | 68.84 | 88.28 | 99.45 |
| B（max） | 43 | 50.80 | +0.57 | 87.36 | 69.08 | 89.16 | 99.89 |
| C（att-pool） | 42 | 50.32 | +0.23 | 92.72 | 71.52 | 94.64 | 100.00 |
| C（att-pool） | 43 | 49.52 | −0.34 | 92.72 | 71.12 | 94.64 | 100.00 |
| **D（cross-attn）** | 42 | **55.60** | **+3.96** | 67.76 | 61.68 | 69.84 | 93.33 |
| **D（cross-attn）** | 43 | **55.60** | **+3.96** | 65.92 | 60.76 | 70.12 | 91.90 |
| A+（参数对齐） | 42 | 49.84 | −0.11 | 92.80 | 71.32 | 94.48 | 100.00 |
| A+（参数对齐） | 43 | 51.52 | +1.07 | 92.08 | 71.80 | 94.40 | 100.00 |

**基线（B 族 `select_rerank/REPORT.md` §3.1）**：`heldout_pair` **49.52 / 51.04**、`shifted_pos` **92.56 / 92.48** —— 臂 A **逐位复现（Δ = 0.00 / 0.00）**。

**随机标签对照（Q3，门槛 ≤ 52.83）** — heldout/shifted：
A **51.84 / 49.68**｜B **48.32 / 52.32**｜C **51.92 / 50.56**｜D **49.36 / 50.48**（补跑 s43：**50.96 / 48.48**）｜A+ **50.72 / 50.16** —— **6 跑全部掉回 50% 附近**。

**Δ(臂 − A) 逐 seed**（heldout，单位 pt）：B **+2.80 / −0.24**（+1.98 / −0.17 SE）｜C **+0.80 / −1.52**（+0.57 / −1.07）｜**D +6.08 / +4.56（+4.30 / +3.22 SE，两 seed 同号）**｜A+ **+0.32 / +0.48**（+0.23 / +0.34）。

## 3. Q1–Q5 逐条实测 + 判定 + 机制

| # | 实测 | 结论 |
|---|---|---|
| **Q1（主）** | 只有 **D** 过：Δ = +6.08 / +4.56 > 2.83pt，两 seed 同号（+4.30/+3.22 SE）；B +2.80（**差 0.03pt 未过**）/ −0.24、C 与 A+ 均不过 | **D 过；B/C/A+ 不过** |
| **Q2（副）** | D 的 `shifted_pos` **67.76 / 65.92** vs A **92.56 / 92.48** ⇒ Δ = **−24.80 / −26.56pt**（−17.5 / −18.8 SE），门槛 ≥ A−2.83 | **不过（严重不过）** |
| **Q3** | 随机标签 6 跑 heldout/shifted 全 ≤ 52.83（含 D 两 seed） | **过** |
| **Q4** | 臂 A `heldout` = 49.52 / 51.04、`shifted` = 92.56 / 92.48，**与 B 族 Δ = 0.00**（≤2.83）；Q4a 位级相同 | **过（零误差复现）** |
| **Q5** | 可训参数：A/B **164,353**、D **164,354**（比 A 多 **1** 个，1.000006×）、C/A+ **164,609**；核可训 0；**A+（与 C 参数量对齐）heldout 49.84 / 51.52 ≈ A** | **增益不是参数量**：D 只多 1 参数；对齐臂无增益 |

**判定（PREREG §4 三选一，不硬选）= ③ 证据不足**：Q1 有臂过（D），但 **Q2 未全过** ⇒ 不满足 ①「墙在池化」所需的 Q1∧Q2∧Q3∧Q4；也不满足 ②（② 要求**没有任何臂**过 Q1）。判据跑前写死，未调参、未补跑刷分。

**机制（实测 → 推断分开）**：
- 【实测】**唯一能推动 `heldout_pair` 的聚合是 D（cross-attention）**：+5~6pt、两 seed 完全同值 55.60；B（max）/C（可学 query 池化）/A+（只加参数）**全都没动**。
- 【实测】D 同时把 `shifted_pos` 92.5 → 66~68、`test` 94.48 → ~70 ⇒ D **不是"学会了换说法"，而是换掉了一整套信号**：表内记忆被削弱，交互信号只补回 5.6pt。
- 【实测·跑后描述性拆解】（`results/breakdown.json`，不参与判据）A 的逐词 heldout 是**两极**（出现 1.00 / 方便 0.95 / 稳定 0.94 ｜ 详细 0.00 / 上升 0.00 / 积极 0.06 / 困难 0.11）；D 把这些**系统性偏置拉回 0.5 附近**（详细 0.48/0.52、上升 0.44/0.38、积极 0.41/0.47），同时**丢掉原本高分词**（出现 → 0.59、稳定 → 0.61/0.52、方便 → 0.76/0.71）⇒ 净 +5.6pt。
- 【推断】mean-pool **确实吞掉了一小部分可利用信号**（否则 D 不可能高于 A，且随机标签与参数对照都排除了捷径/容量解释）；但**这部分只值 5~6pt**，且代价是表内信号塌 25pt。
- 【推断】`heldout_pair` 距可用仍有巨大缺口（55.6% 仅 +3.96 SE）⇒ 剩余缺口**更像表示层**，但这是推断，**不是本次判定**。

## 4. 关于"证据是否齐了"

- 本次**不能**写「四次实测 + 本次消融都指向表示层」：**消融里 D 过了 Q1**，即聚合层确实丢过信号 ⇒ 该强结论的证据**不齐**。
- 齐备所需（未做，跑前也未预注册）：一种**既保住 `shifted_pos`（≥ A−2.83）又让 2 字之差与上下文交互**的聚合过 Q1 ⇒ 才算「墙在池化」；或**在更强表示（更大/微调核）下同一四臂仍 ≈50%** ⇒ 才算「墙在表示」。当前停在 ③。
- 【推断】D 的 Q2 失败提示**"交互"与"表内记忆"在冻结核上互相挤占**——这更像表示容量/口径问题，但要判它需另开 PREREG。

## 5. 原始命令 / unit / 日志 / 产物

```bash
uv run python experiments/select_pool/model_pool.py --selfcheck          # 5 臂自检（训练前）
systemd-run --user --unit=dtseek-select-pool --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  bash experiments/select_pool/run_all.sh                                # 15 跑 20:47:13→20:50:43 ALL DONE
systemd-run --user --unit=dtseek-select-pool-q3 --collect ... \
  uv run python experiments/select_pool/train_pool.py --arm D --seed 43 --randlabel   # Q3 补跑
uv run python experiments/select_pool/analyze.py      # → results/summary.json（Q1–Q5 + 判定）
uv run python experiments/select_pool/breakdown.py    # 跑后描述性逐词拆解
```

- unit：`dtseek-select-pool.service`（invocation `5c843fadae37462eb2b8b2d5a8d4d0d8`，**ALL DONE**，inactive）、`dtseek-select-pool-q3.service`（`fbdcd1c6313848e2aa8fb56bdbbfa54f`，inactive）。
- 日志：`logs/select_pool_{A,B,C,D,Aplus}_s{42,43}[_rand].log`（16 个）+ `logs/select_pool_journal.log`（含首启失败记录 `37de7a0fd0924f36b64ababe5e1fca18`：`--arm` 误当数据臂，训练未开始即中止，已修）。
- 产物：`results/{A,B,C,D,Aplus}_s{42,43}[_rand].json`（16 个）+ `results/summary.json` + `results/breakdown.json`；`weights/*.pt`（16）；`cache/clean_28988c64fa91_tok_ctx64_cand32.pt`（0.85 GB，token 级缓存，键与 B 族同 md5）。
- 纪律：只写 `experiments/select_pool/`、`logs/select_pool_*`；**未改** `select_rerank/`（含 data/PREREG，只读）、`src/`、`training/`、`tests/`、`dev-notes/`、其它 `experiments/`；无 git commit/stash/checkout/restore/clean；开训前查过 `ps` 与 `systemctl --user is-active 'dtseek*'`（均空），未杀任何进程；两次启动均为 `systemd-run --user`（含 PYTHONUNBUFFERED=1），无 `setsid nohup`。

## 6. 遗留与不确定（实测 / 推断分开）

1. 【实测】判定 = **③ 证据不足**：Q1 过（D）、Q2 严重不过；不是"全都不过"⇒ 也不能判 ②。
2. 【实测】B 的 s42 Δ = **+2.80pt，离门槛 2.828pt 差 0.03pt**，两 seed 也不同号 ⇒ 按字面不过（不重跑）。
3. 【实测】只测了 **clean（N）臂**、2 seed、1800 步、未调参；`shifted_pos` 退化幅度：**C +0.16/+0.24、A+ +0.24/−0.40（均 ≥ A−2.83）**、**B −7.20/−5.12（退化但本就未过 Q1，Q2 对它无对象）**、**D −24.80/−26.56（Q2 判定依据）**。
4. 【实测】D 的 `heldout` 两 seed **完全同值 55.60**（逐词方向一致支撑：见 breakdown）——不是噪声，但只有 1 个聚合方式过线，**单臂证据**。
5. 【推断】"交互与表内记忆在冻结核上互相挤占"未做直接打开头内部的验证；cross-attn 温度初值 `1/√128` 是跑前定死的（未扫），**若更合适温度能救回 Q2 属未测**。
6. 【推断】要下"墙在表示"需在**更强表示**下重复本消融；要下"墙在池化"需找到过 Q2 的交互式聚合 —— 两者都需**另开 PREREG**。
