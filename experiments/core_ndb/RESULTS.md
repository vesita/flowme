# core_ndb 实验结果（阶段 1 + 阶段 2）

判据：`PREREG.md`（阶段 1，跑前写死）、`PREREG_PHASE2.md`（阶段 2，跑前写死）。
本文只报实测；未完成的格子写"未跑"，不留空。

## 1. 做了什么（文件与管线）

| 文件 | 作用 |
|---|---|
| `PREREG.md` / `PREREG_PHASE2.md` | 两阶段判据，均在对应阶段任何运行之前落盘 |
| `core_memory.py` | `CoreNDB`：核级情节记忆（no_grad 表 + 零初始化读写总闸）+ `_Keyer`（绑定 `mention_ndb._keys/_ensure_flat` 复用同一份键代码） |
| `pipeline.py` | `CoreNDBEngine(MultiTaskEngine)`：把读写挂到**核编码阶段**；`_CoreEncoder` 包装 + 预编码只写 + 卡片阶段表冻结；`card_write` 反面控制钩子 |
| `run_phase1.py` | 阶段 1 判据（R0/G0/G1/G2a/G2a′/G2b + 跨调用形态） |
| `train_core_ndb.py` | 阶段 2 温启动联合训练（臂 C 核级 / 臂 L 卡级，2 seed） |
| `eval_phase2.py` | 阶段 2 评测（P1 强制按句切段 / P2 老卡 Δ / P3 代价） |
| `results_phase1.json` | 阶段 1 全部 sha256、结构差异、诊断、计时 |
| `cards/p2_{C,L}_s{42,43}.pt`、`results/p2_*.json`、`results_phase2.json` | 阶段 2 产物 |

**管线（阶段 1 形式，阶段 2 沿用同一形式）**

1. `predict(text)` 入口：按输入 `reset(1, device)` 清表；
2. **预编码阶段**（任何卡运行之前）：按句切段（输入侧常量 `PRE_CHUNK=64` / `PRE_MAX_LEN=128`，**不取自挂卡 spec**）逐段核编码并**只写**，写完关写；
3. 卡片阶段：表冻结，每段核编码后读 `out = h + mask·g·mass·(v−h)`；
4. 键 = token n-gram 多项式滚动哈希（`mention_ndb` 原码）；值 = 核隐向量 h 的加权和；
5. 聚合用 one-hot × `bmm`（固定归约顺序、无 atomics），为逐位判据的确定性；
6. 门控：`write_gate`（bias +1）/ `read_gate`（bias −2）/ `write_scale` / `read_scale`，零初始化 ⇒ step-0 贡献恰为 0。

**与"挂了哪些卡无关"的关系**：读写全部发生在核编码阶段，卡片阶段只读；预编码的切段用输入侧常量 ⇒ 挂 1 张卡与挂 4 张卡产生完全相同的核级写入序列 ⇒ G1 由构造成立，再由实测把关。

## 2. 阶段 1 实测（`results_phase1.json` / `logs/core_ndb/phase1.log`）

文本集 = `experiments/core_keep/cache/person_6000.pkl` 前 16 条（长度 29–85）；
逐卡汇总 = 16 条 sha256 用 `\n` 拼接再 sha256；payload 剥 `display`/`color`。
条件：`S0` 关记忆 / `S1` 零门控 / `S2r` 只读开写关（空表对照）/ `S2` 门控开 / `S3` 卡写共享表（反面控制）。

| 判据 | 门槛（PREREG §4） | 实测 | 结果 |
|---|---|---|---|
| R0 | 同条件连跑两遍全等 | S0、S2 各 16×4 条 sha256 全等 | PASS |
| G0 | 零门控 ⇒ 与不接记忆逐位相同 | 4/4 卡相等；`w_sum=0.0`、`delta_l1=0.0`（写入总权重与读带来的改动**恰为 0**） | PASS |
| G1 | 4 卡 vs 单卡同一卡逐位不变 | 8/8 相等（S2 与 S1 两种门控态） | PASS |
| G1′（加测） | `tasks=[c]` vs 4 卡同跑（归一哈希） | 4/4 相等 | PASS |
| G2a | 非空测试：门控开 ⇒ 输出变 | 每卡 16/16 条不同；结构（类别+区间）变化 5/10/7/6 条 | PASS |
| G2a′ | 反面：空表（只开读不写）⇒ 必须**不**变 | 4/4 卡仍与 S0 逐位相同 | PASS |
| G2b | 反面控制：卡写共享表 ⇒ 老卡被改 | 见下表 | PASS |

**G0 的逐卡 sha256**（S0 与 S1 完全相同）：
person `916143b64fab060cb2883970e34ecf25381fef2d95df367a836fc258b34246d9`、
pronoun `54753fc28a79a11b6beae03e48587054c8871745ef8182125256b17cc2c2c158`、
relation `3c3d3be813d43fc7508b1a928429eca04871c395d8bde54320e28c728ee4090e`、
sentiment `28743325462603ccc338baad7c85f00b2ee7ea858e40d2e4333d5767b79b5e19`。

**G1 的逐卡 sha256（4 卡引擎 == 单卡引擎，S2 门控开）**：
person `a670398b580781810ade9fccb8009653425dd4f46a0dbad9dba61942e4e684e8`、
pronoun `156bfeb97a9391d2f8f6c4a0afd42d1db2cd89a819a19de3692a2339845c7f4e`、
relation `2787c68eca16d0e78d6b59d9c86498ac0a622aefb0f72d774b5e6fbb416eab58`、
sentiment `8fbcb9ac57dadd7141da29393d30aee51f3eba249e7fb041d542caac28eabf37`。
跨调用形态（归一 payload `{text, tasks:{卡}}`）：single == all，person `b399bb3570036df3…`、
pronoun `b321e50455455eca…`、relation `582791ea9c892a5e…`、sentiment `77c0261dd136df98…`。

**G2a（S2 vs S0）**：person `a670398b…` ≠ `916143b6…`；pronoun `156bfeb9…` ≠ `54753fc2…`；
relation `2787c68e…` ≠ `3c3d3be8…`；sentiment `8fbcb9ac…` ≠ `28743325…`（各 16/16 条不同）。

**G2b 反面控制，老卡被改变了多少**（两种口径都测）

| 口径 | person（写入方） | pronoun | relation | sentiment |
|---|---|---|---|---|
| 单卡形态（S3 vs S2，逐条 predict） | 0/16 | **14/16** 条不同（结构 0 条） | **12/16** 条不同（结构 1 条 / 2 个锚点） | **12/16** 条不同（结构 0 条） |
| 跨调用形态（4 卡同跑，person 写、后跑的老卡读） | 0/16 | **16/16** 条不同（结构 6 条 / 锚点对称差 16） | **16/16** 条不同（结构 7 条 / 锚点对称差 24） | **16/16** 条不同（结构 0 条，仅 confidence 变） |

跨形态的 sha256：person `b399bb35…`（前后相同），pronoun `b321e504…` → `07883a24…`，
relation `582791ea…` → `9b464618…`，sentiment `77c0261d…` → `9c9dd894…`。
⇒ 「共享表 + 由卡写入」确实会改老卡输出 ⇒ G1 的约束不是空话。

**诊断（S1 / S2r / S2 / S3，16 条 × 4 卡累计）**：`S1 w_sum=0.0 delta_l1=0.0`；
`S2r w_sum=0.0 delta_l1=0.0 g_mean=0.119203`（空表对照）；
`S2 w_sum=2503.14 delta_l1=15318.89 mass_mean=0.2616`；`S3 w_sum=3161.10 delta_l1=15435.10`。
`g_mean` / `mass_mean` 是每次 read 调用均值的跨调用平均（`n_read` 见 json）。

**阶段 1 代价**：S0 21.4–31.1 ms/predict，S1 37.8、S2 34.3–36.0、S3 42.3 ms/predict ⇒
预编码 + 读写约 **+10~16 ms**（含一次额外核编码）；表 12288×129×4B ≈ **6.0 MB / 输入**（batch=1）。

## 3. 阶段 2 实测（待填：训练中）

（`results_phase2.json` 生成后填 P1/P2/P3 表）

## 4. 与 §4 / PREREG 规格的出入（如实说，不改规格）

1. **G2b 做了两种口径**：任务书只要求"老卡被改变"，PREREG 未指定调用形态。先按字面做了
   单卡 predict 形态（测到的其实是"自己污染自己"），随后补了更贴近意图的**跨调用形态**
   （4 卡同跑，person 写、老卡读）；两种都报，判据取两者都过。
2. **阶段 1 落盘后补了一处实现**：把 `w_t = σ(write_gate)` 改为 `σ·clamp(write_scale)`
   （喂进读门控的量 = 实际写进去的权重），为阶段 2 给 `write_scale` 一条梯度路径。
   在阶段 1 的尺度取值（0 或 1）下 `clamp(s,0)==s` ⇒ 对 G0/G1/G2 恒等；重跑后所有诊断数
   与 sha256 与改前逐字一致（`w_sum=2503.144646`、`delta_l1=15318.894188` 等全同）。
3. **阶段 2 门控学习率单列 `--lr-gate`（默认 3e-3，模型仍是 lr_base=3e-4 / lr_head=1e-3）**：
   PREREG §1 只写了"lr_base=3e-4、lr_head=1e-3"，没规定记忆模块的 lr；先按 3e-4 跑到
   400 步实测 `read_scale=0.0024`、相对扰动约 4e-7（`logs/core_ndb/train_p2_aborted_lr3e4.log`）
   ⇒ 那是"门没开"而不是"记忆无益"，故单列 10× 门控 lr。**改在任何 P1/P2/P3 评测之前**，
   门槛一个字没动。
4. **门控初始化由「双 0」改为「记满、不读」**：`write_scale=1 / read_scale=0`。
   双 0 初始化实测是梯度死锁（表空 ⇒ mass=0 ⇒ `read_scale` 梯度恒 0，8 个 epoch 尺度仍精确为 0）。
   step-0 输出贡献仍恰为 0（`read_scale=0 ⇒ delta=0`），零初始化门禁不变；只是"表有内容"从
   step-0 就成立。两个尺度另加投影（`clamp_` 回 [0,1]），否则 Adam 会把 `read_scale` 推负、
   而 `clamp` 在负区梯度为 0 ⇒ 卡死。

## 5. 遗留与不确定

- **实测**：阶段 1 六条判据全部 PASS（含跨调用形态），数字见 §2；R0 通过说明 ROCm 上同条件
  可逐位比较。
- **推断（未实测）**：核级记忆能否带来跨段**身份编号**收益存疑 —— 身份 id 由解码器查询态
  （bos 起步按出场顺序数）决定，核没有指针头、也不能存卡相关标签（存了就违反"与挂卡无关"）；
  核级读只改 `doc_memory`。阶段 2 的 P1 将实测这一点。
- **推断（未实测）**：`write_scale` 在训练里梯度约 1e-10（只经 `w_t → 读门控`，早期读门控权重为 0），
  实际长期停在 1 附近（投影值 0.998–1.000），等效于常数。
- 训练/评测只写了 `experiments/core_ndb/`、`logs/core_ndb/`、`/tmp`；`src/`、`training/`、
  `tests/`、`dev-notes/`、其它 `experiments/` 一个字没动。
