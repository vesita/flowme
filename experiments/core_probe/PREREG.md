# PREREG —— P1 核功能可分离性探针（只读前向 + 线性 probe）

> 本文件的 mtime 必须**早于**本目录任何探针运行（对账、P0、P1–P5）的首次执行；
> 跑后不改判据。若实测与判据冲突，如实写，不回头改门槛。
> 「实测」= 本目录产物直接读出；「推断」= 由实测推出；「口径判断」= 本文件人为规定。

## 0. 问题与边界（写死）

问题：**情绪 / 结构-逻辑 / 身份-回忆 三类信息在 `NanoDocEncoder` 的 `doc_memory` 表征里
是「可线性分离」还是「缠绕」？**

**⚠️ 本实验只回答「可线性读出 + 子空间几何」，不回答「因果使用」。**
「核是否**用**了这些信息」需要干预实验（改表征、消融方向再看卡输出），属下一波，本文不做。

硬约束：**只读** `checkpoints/`、`experiments/*/weights|data|cache`、`src/`、`training/`、
`dev-notes/`、`methodology/`；**不训核、不训卡、不改核**；只写 `experiments/core_probe/`。

## 1. 线上口径对账（先做，过门槛才继续）

抽取路径必须与线上一致：同一 `NanoDocEncoder`（`checkpoints/base_encoder.pt`，hidden 128、
3 层）、同一 `NanoCharTokenizer`、同一截断（**按各任务卡 `spec.max_len`**：sentiment/骨架/逻辑
= 64，person = 120 `window`）、同一掩码。

- 对账动作：复现 **`checkpoints/e5c_single_sentiment.metrics.json` 的 `cls_acc = 0.9321`**
  （n_cls=2370 / n_bg=830）。口径 = `training/train_multitask.py` 的切分：
  `card.build_dataset(32000)` → `random.Random(42).shuffle` → `val = data[:3200]` →
  `GenericTaskDataset` → `runtime.evaluate_task`，卡 = `e5c_single_sentiment.pt` 的 decoder。
- 门槛：**相对差 < 3%**（即 |复现 − 0.9321| < 0.0279）。
- 另有构造性对账（零成本、必做）：hook 捕获的 `norm` 输出与直接 `encoder(ids)` 输出
  **逐位相同**（max|Δ| = 0），否则说明抽层方式改了计算图 ⇒ 先修探针。
- **对不上就先修探针，不改判据。**

## 2. 标签与数据（全部现成资产，不新造标注）

| 类 | 任务 id | 数据源（只读） | 标签定义 | n | split |
|---|---|---|---|---|---|
| 情绪 | `emo` | `experiments/core_keep/cache/sentiment_32000.pkl` | 句级 `label` ∈ {0 中性,1 积极,2 愤怒,3 悲伤} | 32000 → 每类 2000 抽 8000 | 分层 4000/4000（seed 42 定 split） |
| 结构-逻辑 | `skel` | `experiments/two_channel_head/data/train.jsonl` | `skel_id`（30 类，官方 train） | 8000 | 官方 train 8000 → 4000/4000（seed 42） |
| 结构-逻辑 | `logic` | 同上句文本 | 「句中含逻辑词」二分类：因为/所以/但是/但/因此/于是/虽然/尽管/而且/并且/如果/要是/只要/除非/总之/可见/由此/结果 | 8000（同 `skel` 句子） | 同 `skel` |
| 身份-回忆 | `idr` | `experiments/core_keep/cache/person_6000.pkl` | 提及级 `is_repeat`：该提及的 id 在**同一条文本更早处**已出现过（= 线上 `repeat_mention_acc` 的分母口径） | 4200 条带提及文本 / 33139 提及 | **按文本分组**切 2100/2100（seed 42），杜绝同文本跨集合 |

- **池化（写死）**：`masked mean`（有效 token 平均）；`idr` 用**提及 span 内** masked mean
  （span 按 char 下标切，tokenizer 为字符级 1:1，截断越界则钳到有效区）。
- **层（写死 5 个）**：`emb`（embedding 输出）、`b0`、`b1`、`b2`（各 block 输出，final-norm 之前）、
  `pool`（`doc_memory = norm(b2)` 池化后）。
- **逐类 `max_naive`（P5 必报，两项都报）**：
  - `max_naive@majority`：**训练集多数类**在测试集上的准确率（免费盲猜基线）；
  - `max_naive@rule`：**免费规则基线**，实测口径见 §6：
    - `emo` = `sentiment.dataset.extract_emotion_spans` 多类投票对 `label` 的一致率；
    - `skel` = `two_channel_head/data/stats.json` 的 `max_naive_train = 0.5329`（tree_depth4 规则，其实测）；
    - `logic` = 词表规则**按定义 100%**（标签本身就是词表匹配 ⇒ 该列必须如实写 100%，probe 不构成增量）；
    - `idr` = 「表面词在前文出现过 ⇒ repeat」字面规则的准确率（**已实测 0.7918**，n=33139）。
  - 铁律：**任何指标并列 `max_naive`**；probe ≤ `max_naive@rule` 时如实写「免费规则已解决，
    核的线性可读性不构成增量」。

## 3. P0 双向对照（门禁，两个都过才继续）

| 对照 | 构造 | 通过门槛（2 seed 都要） |
|---|---|---|
| ① 随机标签 | 每任务把标签按同边缘分布独立随机打乱（与 X 独立），同法拟合评测 | test acc ≤ `max_naive@majority` + 2×SE，SE = sqrt(p(1−p)/n_test) |
| ② 构造可分 | (a) **输入内编码**：4000 条 `emo` 文本随机贴前缀 `甲甲甲甲` / `乙乙乙乙`，标签 = 前缀组；(b) **合成张量**：X~N(0,I)（n=4000,d=128），y = sign(X·w_true) | 两者的 test acc ≥ **0.99** |

①不过 ⇒ 探针在假信号上虚高；②不过 ⇒ 测量函数本身坏。任一不过 ⇒ 停，报失败输出。

## 4. P1 / P3：分层单任务 probe（核冻结，只训 probe）

- probe = 线性（softmax 回归），**特征标准化**（train 的 mean/std），全批 Adam，最多 2000 步，
  早停靠固定步数（不调参挑结果）；核 `eval()` + `no_grad`。
- 每任务 × 每层 × 2 seed（42/43 = probe 初始化/打乱种子；**句子与 split 两 seed 相同** ⇒ 配对）。
- **P1 门槛**：最佳层 `acc(2 seed 均值) > max_naive@majority + 2×SE`，SE = sqrt(p(1−p)/n_test)。
  ≤ ⇒ 如实写「核里读不出这个信息」。
- **P3**：逐层报 acc ± SE（binomial SE 与 seed 间差都报），并与
  `experiments/layer_selective/REPORT.md` 的「新能力在浅层（embedding 66.24/62.08；深层
  blocks.2 零增益且漂移最大）」对照。

## 5. P2 可分离性（主判据）

1. **联合 probe（带瓶颈）**：共享投影 P（128→k，k=8 写死）+ 各任务独立线性头，
   在**四任务各自的句池上联合训练**（每行只对该任务的标签计 loss）；
   对照 = **同结构 k=8 的单任务 probe**。下降 `Δ = acc_joint − acc_single`（2 seed 均值），
   显著 ⟺ `Δ < −2×SE_paired`（SE_paired = 逐样本 correct 差的 SE，配对口径）。
   - 构造性 sanity（必报）：**零共享参数的联合**（去掉共享投影 P，各任务独立线性头、
     参数无交集）与单任务 probe 在构造上等价（loss 可加、梯度互不耦合），实测 |Δ| ≤ 2SE_paired；
     若不等价 ⇒ 实现有隐藏共享项，记为实现问题，P2 不成立。
     ⇒ 因此「联合下降」只在**同瓶颈 k=8** 口径下解释（k=128 的共享 P 仍会耦合，不作该 sanity）。
2. **子空间几何**：每任务在每层用 **8 个初始化 × 2 seed** 拟合方向 w（标准化特征空间，不归一，
   直接算余弦）；报 `within`（同任务不同 init/seed 两两余弦 = 方向稳定性基线）与
   `between`（任务间两两余弦）的 中位数 / p25 / p75。
   - **稳定性门**：若某任务 `within` 中位数 < 0.5 ⇒ 该任务方向不稳定，其 between 余弦**不作判据**。
3. **交叉干扰**：任务 A 的方向 w_A 去解任务 B（在 B 的 train 上只做 **1 维仿射校准**，
   不改方向），报 `acc_B(w_A)`；`X = acc_B(w_B) − acc_B(w_A)`，显著 ⟺ `X > 2×SE_paired`。
4. **共同最佳层 L\***：在 5 层里取 `argmax_L min(acc_A(L)/max_L acc_A, acc_B(L)/max_L acc_B(L))`
   （按各任务自身最佳归一后取 min 最大）。余弦/交叉干扰在**全部 5 层**都算，判据用 L\*。

## 6. 判定（三选一，跑前写死；不许硬选）

先决：P0 两对照都过；P1 通过的任务 ≥ 2 个；`within` 稳定性门通过。任一不满足 ⇒ **证据不足**。

对每个**可读任务对**（两任务都过 P1），在 L\* 处看三条量：
`C` = between 余弦中位数的绝对值；`Δ` = k=8 联合下降；`X` = 交叉干扰。

- **可分离** ⟺ **所有**可读对满足：`C ≤ 0.3` **且** `Δ ≥ −2SE_paired` **且** `X ≤ 2SE_paired`；
- **缠绕** ⟺ **≥ 2 对**同时满足：`C ≥ 0.6` **且** `Δ < −2SE_paired` **且** `X > 2SE_paired`；
- 其余（含判据方向互相矛盾、只有一对满足缠绕、pairwise 结论不一致）⇒ **证据不足**。

报告里必须写清：三类信息各自在哪些层最强、与 `layer_selective`「新能力在浅层」是否一致；
区分实测/推断，标注证据强度（单 seed / 配对 / 2 seed）；**不许把「没测出差异」写成「没有效果」**。

## 7. 运行纪律

- `uv run python`；长跑用 `systemd-run --user --unit=dtseek-core-probe --collect
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0
  --setenv=PYTHONUNBUFFERED=1`；**绝不用 `setsid nohup &`**。
- 开跑前查 `ps` 与 `systemctl --user is-active 'dtseek*'`；GPU 忙则排队，**不杀他人进程**。
- 语料/数据按 `dev-notes/19` 过滤，**不许 `fs[:N]`**，报告贴文件清单。
- 做完 P0–P5 即停：**不训核、不训卡、不改既有实验目录、不加新卡、不写 `dev-notes/`/`methodology/`、
  不派子代理、不做因果干预**。
