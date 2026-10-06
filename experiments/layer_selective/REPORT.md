# layer_selective 结果报告 —— 「共享参数集合」当可设计对象：改核拿新能力 vs 不崩老卡

> 判据与配方见 `PREREG.md`（mtime **2026-10-07 01:08:22**，早于空测试首跑 01:13:48、冒烟首跑 01:14:41 与全量首跑 **01:16:01**，跑后未改）。
> 「实测」= 本目录产物直接读出；「推断」= 由实测推出；「口径判断」= 构造时人为规定的标签归属。
> 数据只读复用 `experiments/select_rerank/data/clean/`：跑前跑后 md5 一致（`6b1e6fb1…` / `d3d597af…` / `2a5ca5f4…`）。

## 1. 五臂可训参数清单（`results/selftest.json` + 各 `results/*.json::inventory` **实测打印**，非声明）

| 臂 | `requires_grad=True` 的参数（实测名） | 可训合计 | 冻结实况（实测） | 编码 |
|---|---|---|---|---|
| **F** | `head.net.{0,2,4}.{weight,bias}`（6 张量） | **164,353** | 核 26 张量全 False（`encoder.embedding.weight`、`blocks.0/1/2.*`、`norm.weight`） | 缓存 |
| **ALL** | 核 26 张量全可训 + `head.*` 6 | **1,852,813** | 冻结 0 张量 | live |
| **EMB** | `encoder.embedding.weight`（1 张量 1,048,576）+ `head.*` 6 | **1,212,929** | 核冻 25 张量（`blocks.0/1/2.*` + `norm.weight`） | live |
| **TOP** | `encoder.blocks.2.{ln_1,attn.qk_scale,attn.qkv,attn.proj,ln_2,mlp.c_fc,mlp.c_fc2,mlp.c_proj}`（8 张量 213,252）+ `head.*` 6 | **377,605** | 核冻 18 张量（`embedding` + `blocks.0/1` + `norm`） | live |
| **ADPT** | `head.*` 6 + `adapter.{fc1,fc2}.{weight,bias}` 4（**65,920 = 核的 3.90%**） | **230,273** | 核 26 张量全 False（`encoder_trainable = 0`） | 缓存 |

**冻结不是声明，是反向后实测**（`selftest.py` + 每跑 `GRADSTAT_STEP0`）：五臂 step-1 反向后
**冻结侧收到梯度的参数 = 0 个**（`frozen_with_grad: []`），可训侧梯度范数 > 0；
F/ADPT 另做一次**把核放进计算图的 live 前向**再反向 ⇒ `encoder_params_with_grad = []`（核在图里也没梯度）。
适配层 step-0 恒等 `max|adapter(x)−x| = 0.0`（零初始化生效）；五臂真标签 `loss_first` 全 = **0.693002 / 0.692196**（构造顺序/RNG 流同源）。
多数类：三个 split 均 0.5000；adv 两个 ctype 各 1250、正例 623/627 ⇒ 0.5016。随机标签真执行：1:1 `[4000,4000]`、不同条数 4026 / 4016。

## 2. 主表（准确率 %；n=1250 ⇒ **SE = 1.414pt**、**2×SE = 2.828pt**；X1 门槛 = ALL − 2SE = **70.77 / 65.57**）

| 臂 | seed | `heldout` | Δ vs F | `shifted` | test | train | adv | rand `heldout`/`shifted` |
|---|---|---|---|---|---|---|---|---|
| F（核冻结基线） | 42 | 49.52 | — | 92.56 | 94.48 | 100.00 | 71.04 | 51.84 / 49.68 |
| F | 43 | 51.04 | — | 92.48 | 94.48 | 100.00 | 71.76 | 51.60 / 49.12 |
| **ALL**（= `selonly`） | 42 | **73.60** | +24.08 | 100.00 | 100.00 | 100.00 | 86.80 | 45.68 / 49.44 |
| **ALL** | 43 | **68.40** | +17.36 | 100.00 | 99.96 | 100.00 | 84.20 | 51.60 / 46.88 |
| **EMB** | 42 | **66.24** | +16.72 | 99.92 | 99.92 | 100.00 | 83.08 | 49.52 / 47.36 |
| **EMB** | 43 | **62.08** | +11.04 | 99.92 | 99.96 | 100.00 | 81.00 | 47.28 / 47.36 |
| **TOP** | 42 | 49.76 | +0.24 | 94.32 | 96.00 | 100.00 | 72.04 | 52.64 / 51.76 |
| **TOP** | 43 | 47.28 | −3.76 | 93.04 | 96.24 | 100.00 | 70.16 | 48.24 / 48.24 |
| **ADPT** | 42 | 50.32 | +0.80 | 94.32 | 95.04 | 100.00 | 72.32 | 50.80 / 51.52 |
| **ADPT** | 43 | 48.08 | −2.96 | 93.84 | 95.00 | 100.00 | 70.96 | 50.56 / 48.64 |

- **X1 逐臂**：F −21.25/−14.53、EMB **−4.53/−3.49**（最接近）、TOP −21.01/−18.29、ADPT −20.45/−17.49 ⇒ **四臂全不过**。
- **X4 随机标签**：10 跑 ×（heldout+shifted）**全部 ≤ 52.83**（最大 52.64）⇒ 门禁过；`train` 对真标签 49.55~49.80 ⇒ 空对照有效。
- 交叉对账（实测）：本档 ALL rand = 45.68/49.44、51.60/46.88 与 `core_select_only::selonly` rand **逐项相同**。

## 3. 老卡表（训练后 = 各臂核 + 冻结老卡头；Δ = post − step0；**带 = PREREG 写死**）

| 臂 | seed | pronoun Δ / 带0.0283 | sentiment Δ / 带0.0041 | relation Δ / 带0.0139 | person Δ / 带0.0033 | X2 |
|---|---|---|---|---|---|---|
| F | 42/43 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **过 ✅**（核漂移 0） |
| ALL | 42 | −0.4617（×16） | −0.4856（×118） | −0.6792（×49） | −0.0050（×1.5） | **8/8 不过 ❌** |
| ALL | 43 | −0.5150（×18） | −0.5022（×122） | −0.6944（×50） | −0.0067（×2.0） | 同上 |
| EMB | 42 | −0.0217 ✅ | −0.0341（×8.3） | −0.0347（×2.5） | 0.0000 ✅ | **4/8 不过 ❌** |
| EMB | 43 | −0.0283 ❌（超 0.000033） | −0.0406（×9.9） | −0.1181（×8.5） | +0.0017 ✅ | 同上 |
| TOP | 42 | −0.0917（×3.2） | −0.1059（×26） | −0.0472（×3.4） | −0.0050（×1.5） | **6/8 不过 ❌** |
| TOP | 43 | −0.0250 ✅ | −0.0706（×17） | −0.0292（×2.1） | −0.0067（×2.0） | 同上 |
| ADPT | 42/43 | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 | **过 ✅（构造保证，见 §4）** |

## 4. X0–X5 逐条实测 + 判定 + 机制

| # | 实测 | 判定 |
|---|---|---|
| **X0（门禁）** | **38 项逐项 Δ = 0.00**（`max_abs_delta = 0.0`）：F 的 heldout 49.52/51.04、shifted 92.56/92.48、test 94.48、train 100.00、adv 71.04/71.76、`loss_first` 0.693002/0.692196、漂移 **0**；ALL 的 heldout 73.60/68.40、shifted 100.00、test 100.00/99.96、adv 86.80/84.20、`loss_select_first50` 0.153912/0.134212、漂移 **0.05003196568723828 / 0.048874110162901115**、老卡 8 项 Δ 全等 | **过 ✅（零误差）** |
| **X1** | 门槛 70.77/65.57；F 49.52/51.04、EMB 66.24/62.08、TOP 49.76/47.28、ADPT 50.32/48.08 | **四臂全不过 ❌** |
| **X2** | F 8/8 过、ADPT 8/8 过；EMB 4/8、TOP 6/8、ALL 8/8 超带 | 只有 F/ADPT 过 |
| **X3** | `shifted` 门槛 89.73/89.65；ALL 100/100、EMB 99.92/99.92、TOP 94.32/93.04、ADPT 94.32/93.84 | **全过 ✅**（未拿表内记忆换） |
| **X4（门禁）** | 5 臂 × 2 seed rand，`heldout`/`shifted` 全 ≤ 52.83（max 52.64），`n_diff` 4026/4016 | **过 ✅** |
| **X5（必报）** | 见下表 | 数值齐全 |

**判定（PREREG §3 跑前写死三选一）= ② 解耦不成立**：X0/X4 门禁过，**没有任何臂同时满足 X1 ∧ X2**。
最接近的是 EMB（X1 差 4.53/3.49pt = 3.2/2.5 SE，且 X2 有 4 格超带）；ADPT/X2 满分但 X1 差 20pt。

**X5（实测）**：可训参数量 / 核漂移 / 稳态步时 / 峰值显存 / 单跑墙钟

| 臂 | 可训 | 核漂移 (s42/s43) | s/step（末200中位） | 峰值 MB | 墙钟 s |
|---|---|---|---|---|---|
| F | 164,353 | 0 / 0 | 0.0015 / 0.0015 | 81.0 | 16.2 / 16.7 |
| ALL | 1,852,813 | 0.050032 / 0.048874 | 0.0372 / 0.0356 | 441.8 | 80.1 / 80.2 |
| EMB | 1,212,929 | 0.029535 / 0.028709 | 0.0265 / 0.0274 | 320.8 | 64.4 / 66.5 |
| TOP | 377,605 | **0.062707 / 0.062632** | 0.0215 / 0.0214 | 203.4 | 58.9 / 56.3 |
| ADPT | 230,273 | 0 / 0 | 0.0027 / 0.0027 | 103.7 | 18.8 / 18.5 |

**机制（实测 → 推断分开）**
- 【实测】**新能力来自浅层（embedding），不来自深层**：只训 `embedding` = 66.24/62.08（比 F 高 +16.72/+11.04pt = +11.8/+7.8 SE）；
  只训 `blocks.2` = 49.76/47.28（**与 F 无差**，s43 还低 3.76pt）。深层是**白付钱**：拿不到新能力却崩老卡（sentiment −10.6pt = ×26 带）。
- 【实测】**漂移量不预测新能力**：TOP 漂移 0.0627（比 ALL 的 0.0500 还大 25%）却零增益；EMB 漂移只有 ALL 的 59% 却拿到 ALL 七成的增益。
  与 `methodology/measurement.md` 一致 —— **杠杆在方向/集合，不在范数大小**。
- 【实测】**私有参数保得住老卡但买不到新能力**：ADPT 老卡 8/8 Δ = 0.000000、核漂移 0；heldout 50.32/48.08 vs F 49.52/51.04 ⇒ Δ = +0.80/−2.96pt，**在 ±2SE 内**。
- 【实测】**动核的两臂都崩老卡**：ALL 8/8 超带、EMB 4/8、TOP 6/8 —— 在本配方（1800 步、lr_core 3e-4、mean-pool、clean 臂）下**没有一个「动核」的配置既过 X1 又过 X2**。
- 【推断·构造保证（非学出来）】ADPT 的老卡零退化是**结构性**的：适配层位于核输出之后、只在选择头路径上，老卡直连 `enc(inp)` 不经过它，且核漂移为 0 ⇒ 它没有被任何选择任务梯度碰过。
- 【推断·两层结论（PREREG 要求分开读）】**层选择**：梯度进浅层（embedding）有真实新能力但必然伤老卡（embedding 是全任务共享输入），进深层（blocks.2）既无能力又伤卡 ⇒ 「选一层」这条路**不成立**；
  **私有参数**：核外私有容量对老卡是**完全免费**的保护，但本设计（token 级点式 MLP、残差零初始化、66K = 核 3.9%）**一分新能力也没买到** ⇒ 「插适配层」这条路在本设计下**不成立**。两者都未过 X1 ⇒ 合并判定 ②。
- 【推断·置信度】X0/X4 门禁 = **强**（零误差 38 项 + 10 个 rand 全清零 + 2 seed）；「TOP 无新能力」= **强**（2 seed 同向、且 train 100% 说明不是没训到）；
  「EMB 比 TOP 强」= **强**（2 seed 同号、差 16.5pt）；「ADPT 无增益」= **中**（2 seed、Δ 在噪声内，但**只测了一种适配层设计**）。

## 5. 原始命令 / unit / 日志 / 产物 / 纪律 / 遗留

```bash
uv run python experiments/layer_selective/selftest.py                       # 01:13:48 首跑，8 项全过（SELFTEST_DONE）
systemd-run --user --unit=dtseek-ls-smoke --collect --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 --setenv=PYTHONDONTWRITEBYTECODE=1 \
  bash experiments/layer_selective/run_all.sh smoke    # ALL 100 步探跑：loss/drift/eval 与 core_select_only 100 步探跑逐位相同
systemd-run --user --unit=dtseek-ls-real ... run_all.sh real   # 五臂 × 2 seed 真标签（01:16:01 → 01:26:xx）
systemd-run --user --unit=dtseek-ls-rand ... run_all.sh rand   # 五臂 × 2 seed 随机标签（→ 01:35:25 ALL DONE）
uv run python experiments/layer_selective/analyze.py                      # X0–X5 + 判定 → results/summary.json
```

- **unit**（现均 inactive，`list-units 'dtseek*'` 为空）：`dtseek-ls-smoke.service`（inv `d5e7d5a8…`，21.1s 墙 / 2.3G）、
  `dtseek-ls-real.service`（inv `3a614547…`，01:16:01→01:24:22 ALL DONE，8min20.9s 墙 / 12min46s CPU / 6.6G）、
  `dtseek-ls-rand.service`（inv `4d364b27…`，→01:35:25 ALL DONE，8min27.2s 墙 / 12min55s CPU / 6.3G / swap 1.4G）；均 `--collect` + `PYTHONUNBUFFERED=1` + `PYTHONDONTWRITEBYTECODE=1`，**无 `setsid nohup`**；开训前查过 `ps` 与 `systemctl --user is-active 'dtseek*'`（空闲），**未杀任何进程**。
- **日志**：`logs/layer_selective_{F,ALL,EMB,TOP,ADPT}_s{42,43}[_rand].log`（20）+ `layer_selective_selftest.log` + `layer_selective_analyze.log`；单元 stdout 在 `journalctl --user -u dtseek-ls-*`。
- **产物**：`experiments/layer_selective/results/*.json`（10 主跑 + 10 rand + 1 探跑 + `selftest.json` + **`summary.json`（X0–X5 + 判定）**）、`weights/*.pt`（21）。
- **只读自证（实测）**：数据 md5 跑前跑后一致；`run_all.sh` 内 `find -newermt PREREG` 对 `select_rerank/select_pool/select_semantic_joint/core_select_only/capability_map/src/training/tests` 的命中**全部来自其它会话**（`src/dtseek/tasks/render.py`、`tests/test_render.py`、`dev-notes/20-*.md`、他人 `__pycache__`），本代理写入仅限 `experiments/layer_selective/`、`logs/layer_selective_*`、`/tmp`；`git status` 只见新增未跟踪目录；**本代理未执行任何 git 写操作**（commit/stash/checkout/restore/clean 均未跑）。

**遗留与不确定（实测 / 推断分开）**：
1. 【实测】**只测了一种适配层**：token 级**点式** MLP（128→256→128、残差零初始化、65,920 参数）位于核输出之后、mean-pool 之前；**学习式 pooling（注意力池化 / 私有 cross-attention）与「适配层插在每层 block 内」未测** ⇒ 「私有参数买不到新能力」只对本设计成立。
2. 【实测】EMB 距 X1 门槛 4.53/3.49pt（3.2/2.5 SE）——**是「差一点」而非「没有」**；但 X2 已明确不过（4 格超带），故不影响 ② 的方向。
3. 【实测】仅 2 seed；步数 / lr / 聚合 / 适配层宽度**均未扫**（跑前定死，不调参刷过）。
4. 【实测】X1/X3 用独立二项 SE（1.414pt），两臂在同一批 1250 条上评测（配对差），配对 SE 未算 ⇒ Δ/SE 是保守下界。
5. 【推断】F/ADPT 走缓存、其余 live（PREREG 写死）；实测 `cache-vs-live max|Δ| = 1.19e-07`，不影响 argmax 口径。
6. 【实测】X4 中 TOP rand s42 的 52.64 距门槛 52.83 仅 0.19pt（未越线，如实报）。
7. 【推断】本结论只对 clean(N) 臂、mean-pool、1800 步、lr_core 3e-4 成立；换数据集 / 更长训练 / 其它共享参数切分**未测**。
