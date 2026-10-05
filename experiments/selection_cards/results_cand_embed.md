# task-11 方案 A（候选槽位嵌入）实测报告 —— reply_pick

> 判据与判定规则**跑前**写死于 `PREREG_CAND_EMBED.md`（早于第一次训练启动）。全部为**实测**；
> 「推断」处单独标注。8 跑 = 2 档（frozen/joint）× 2 臂（cand/nocand）× 2 seed（42/43），每格 1 次。
> **结论：假设不成立（P1–P3 全不过；Δexact 远低于预注册阈值）。**

## 1. 输入构造与候选嵌入具体做法
- `cand_idx ∈ {0,1,2,3,4}`，**逐字符（= 逐 token，本分词器 1 字 1 token，填充位补 0）**：
  - `0` = `问：`+问句正文、以及 tokenizer 填充位；
  - `n` = 第 n 个候选的**标号段 `|n）` + 该候选正文**的全部字符。
- **分隔符归档：`|n）` 归它引入的候选（n 档），不归 0 档。** 理由（PREREG §1）：
  ① 标号段与正文首尾相接 ⇒ 整个候选列表被切成 4 段**连续无缝**分区，不会把数字 n 与正文拆开；
  ② 标号里的数字字面就是 n，`cand_idx=n` 的区间自带一个指向类别 n 的字面锚点 —— 正是类别头
  与指针头要对齐的东西；③ 填充位必须归 0，否则 padding 参与绑定。
- 加法位置：`doc_memory = NanoDocEncoder(ids, mask) + cand_emb(cand_ids)`，
  `cand_emb = nn.Embedding(5, 128)`，`normal_(0, 0.02)`（与基座词嵌入同量级，起点近似基线微扰）。
  **加在 encoder 最后一层 RMSNorm 之后、解码器交叉注意力之前** ⇒ 不会被归一化中和。
- 标签不动：金标仍是 `spans` 的 0-based 半开区间（候选**正文**，不含 `|n）`）。fail-closed：
  渲染不一致 / 锚点字面不符 / 金标没整段落在同编号档位 → 直接抛错（`--check-cand` 实测 300 条通过）。

## 2. 参数量与训练代价（实测）
| 项 | 值 |
|---|---|
| 候选嵌入 | **640**（5×128）= decoder 的 0.10% |
| decoder / encoder | 630,535 / 1,688,460 |
| 可训参数 | frozen 630,535 → **631,175**（+640）；joint 2,318,995 → **2,319,635**（+640） |
| frozen | 12 ep × 150 = **1800 步**，37.4–39.1 s，0.0208–0.0217 s/step，峰值 **151.5 MB** |
| joint | 16 ep × 150 = **2400 步**，108.3–109.0 s，0.0451–0.0454 s/step，峰值 **510.1 MB** |
| 8 跑总墙钟 | 08:19:37 → 08:31:40 = **12 分 03 秒**（含每跑 13 s 数据构造 + 三集合评测） |

加嵌入的增量代价：frozen 步时 +4.3%（0.0208→0.0217）、峰值显存**不变**（151.5 MB）；joint 步时 +0.5%、峰值不变。

## 3. 主表：两档 × 两臂 × 两 seed（评测折 split=eval，1500 条 = 1275 有解 + 225 背景）
**盲猜基线 cls_acc = 1/5 = 20%**（含「无合适候选」）。P1 exact≥0.60 / P2 bg_fp≤0.10 / P3 span_hit≥0.95。

| 档 | 臂 | seed | exact | bg_fp | span_hit | cls | anchor(条件) | P1 | P2 | P3 |
|---|---|---|---|---|---|---|---|---|---|---|
| frozen | **cand** | 42 | 0.055 | 0.693 | 0.024 | 0.209 | 0.023 | × | × | × |
| frozen | **cand** | 43 | 0.055 | 0.676 | 0.022 | 0.195 | 0.021 | × | × | × |
| frozen | nocand | 42 | 0.051 | 0.689 | 0.019 | 0.180 | 0.017 | × | × | × |
| frozen | nocand | 43 | 0.054 | 0.684 | 0.019 | 0.188 | 0.018 | × | × | × |
| joint | **cand** | 42 | 0.167 | 0.840 | 0.172 | 0.233 | 0.171 | × | × | × |
| joint | **cand** | 43 | 0.181 | 0.844 | 0.185 | 0.258 | 0.185 | × | × | × |
| joint | nocand | 42 | 0.184 | 0.689 | 0.164 | 0.229 | 0.163 | × | × | × |
| joint | nocand | 43 | 0.161 | 0.849 | 0.165 | 0.238 | 0.165 | × | × | × |

cand 臂均值：frozen exact 0.055 / bg_fp 0.684 / span 0.023；joint exact 0.174 / bg_fp 0.842 / span 0.178。

## 4. 唯一有效对照：同配置不加嵌入（逐项差值 Δ = cand − nocand）
| 档 | seed | Δexact | Δbg_fp | Δspan_hit | Δcls |
|---|---|---|---|---|---|
| frozen | 42 | **+0.0040** | +0.0044 | +0.0055 | +0.0290 |
| frozen | 43 | **+0.0007** | −0.0089 | +0.0031 | +0.0071 |
| joint | 42 | **−0.0167** | +0.1511 | +0.0078 | +0.0039 |
| joint | 43 | **+0.0193** | −0.0044 | +0.0196 | +0.0196 |
| — | **均值** | **+0.0018** | +0.0356 | **+0.0090** | **+0.0149** |

- 两臂初始化与数据顺序**逐位一致**（nocand 臂照样构造同一嵌入表、只是不接入计算图、不进优化器）。
- 方向一致性：Δspan_hit 与 Δcls **4/4 同号为正**（+0.3~2.9pt）；**Δexact 联合档两 seed 反号**，
  均值 +0.18pt，比预注册要求的 +10pt 小 **55 倍**。
- 背景交叉印证（**非对照**，按预注册不参与判定）：本实验 nocand 基线 frozen exact 0.051/0.054、
  joint 0.184/0.161，与 §9 的 0.052 / 0.172 量级一致 ⇒ 同配方基线复现成功。

## 5. P1–P3 判定与假设是否成立（实测）
- **P1** exact≥0.60：**0/8 过**（最好 0.184）；**P2** bg_fp≤10%：**0/8 过**（68.9%–84.9%）；
  **P3** 锚点完全命中≥95%：**0/8 过**（1.9%–18.5%）。三条全过才算成立 ⇒ **不成立**。
- 预注册第二条（Δexact 两档两 seed 同号且均值 ≥ +0.10）也不满足（+0.0018，联合档反号）。
- ⇒ **按 PREREG_CAND_EMBED §5：假设「加候选嵌入即可建立绑定」不成立**，走 §10 的另一分支：
  问题比「绑定」更深（**推断**：基座本身不编码候选间关系）→ 下一步应上方案 B（候选独立编码 + 重排头）。
- 证据强度：每格 1 次、2 seed；Δcls/Δspan 的 4/4 同号是**小而一致的正信号**，
  只能写「微弱改善、远不够」，**不能**写成「嵌入毫无作用」。

## 6. 机制（实测）
- **frozen：连训练集都没记住** —— train exact 0.107–0.126，loss 12.27→9.87（12 ep 末段仍在降，
  未收敛）；train−eval 差距仅 0.056–0.071（两边都差，不构成「没过拟合」的证据）。
- **joint：完全在记训练集** —— train exact **1.000**，loss 7.7→**0.0001**（第 12 ep 已归零），
  eval exact 0.161–0.184，**差距 0.816–0.839**；加嵌入后差距 0.833/0.819 vs 基线 0.816/0.839
  ⇒ **train/eval 差距没有缩小**（两 seed 方向相反）。
- 训练折 val 与评测折数字贴近（frozen 0.040–0.052 vs 0.051–0.055；joint 0.163–0.175 vs
  0.161–0.184）⇒ 无折间泄漏迹象。
- **不是空测试**：嵌入权重训后 std 0.02→**0.088**（4.4×）、行范数 0.57–1.23、行间最大差 0.52；
  frozen 末损失 9.87 vs 基线 10.24（优化轨迹确实被改变）⇒ 自变量真的变了，是「变了也没用」。
- **测量自检**：nocand 臂用**训好权重**复算，`evaluate_cand` 与 `runtime.evaluate_task` 在 n=200
  上 6 项**全等**（PARITY ok=true，实测 cls 0.189349 / 0.307692 等两处）；绑定标签 300 条通过。

## 7. 命令原文 / unit 名 / 日志
```bash
# 主扫描（8 跑串行，唯一一次训练启动）
systemd-run --user --unit=dtseek-candemb-sweep --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=PYTHONUNBUFFERED=1 \
  /bin/bash -lc 'bash experiments/selection_cards/run_cand_embed_sweep.sh > logs/cand_embed_sweep.log 2>&1; echo $? > logs/cand_embed_sweep.rc'
# 脚本内逐条（8 条，只差 --mode/--cand/--seed）：
uv run python -u experiments/selection_cards/exp_cand_embed.py --mode <frozen|joint> \
  --cand <on|off> --seed <42|43> > logs/cand_embed_<档>_cand<on|off>_s<seed>.log 2>&1
# 跑前自检 / 汇总 / lint
uv run python -u experiments/selection_cards/exp_cand_embed.py --check-cand
uv run python -u experiments/selection_cards/exp_cand_embed.py --parity --mode frozen --cand off --seed 42
uv run python experiments/selection_cards/summarize_cand_embed.py
uv run ruff check experiments/selection_cards/exp_cand_embed.py experiments/selection_cards/summarize_cand_embed.py
```
- unit：**`dtseek-candemb-sweep`**（另有冒烟 `dtseek-candemb-smoke`，已跑 rc=0）。
- 日志：`logs/cand_embed_sweep.log`（主）、`logs/cand_embed_{frozen,joint}_cand{on,off}_s{42,43}.log`（8 份）、`logs/cand_embed_smoke.log`。
- 产物：`experiments/selection_cards/{PREREG_CAND_EMBED.md, exp_cand_embed.py, run_cand_embed_sweep.sh, summarize_cand_embed.py, results_cand_embed_*.json (8), candemb_*.pt (8), results_cand_embed.md}`。
- 纪律：**未改** `src/`、`tests/`、`training/`、`dev-notes/`；只写 `experiments/selection_cards/`、`logs/`、`/tmp`；无 git 操作。

## 8. 遗留与不确定项
1. frozen 臂 1800 步**未收敛**（loss 仍降）⇒ 该档「无增益」有一部分落在共同的未收敛地板上；
   同预算配对下 Δ 仍可比，但**绝对值不是该档的上限**（**推断**：加步数可能整体抬升，但两臂会一起抬）。
2. joint 臂 train exact=1.000 = 纯记忆，本轮**没有**早停/正则对照 ⇒ 无法区分「表征不行」与
   「没有泛化约束」；要拆开需要第三臂（加正则/早停）。
3. 只跑 `reply_pick`；`cloze_fill` 未跑（空位 `__` 的归档规则需先定：归空位所在候选还是归 0 档）。
4. 评测口径 = 与训练同前向（`evaluate_cand`，已与 `runtime.evaluate_task` 对账全等）；
   **engine 部署口径 `MultiTaskEngine.predict` 未对账** —— 本产物含 `cand_emb`，不在现有卡协议里，
   走部署口径需协议级改动（超出本轮范围，属方案 B 的一部分）。
5. 位置偏置未单列判据（本轮 P1–P3 只有三条）：frozen cand s42 按位置 cls
   {1:0.252, 2:0.151, 3:0.177, 4:0.257}，max−min 10.6pt，只报不判。
6. 目录锁：`experiments/selection_cards/` 被另一会话**独占声明**（c_62）；我以独立文件名写入，
   已在留言板 m_69 请求协商/释放 —— 若对方随后写入同名文件需复核。
