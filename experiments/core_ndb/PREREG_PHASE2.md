# PREREG —— core_ndb 阶段 2（温启动联合训 mention 门控）

落盘时间：2026-10-06 18:05 +08:00，**在任何阶段 2 训练/评测之前写死**。
本文件只定义阶段 2 的判据与口径；阶段 1 的判据见同目录 `PREREG.md`（已全部 PASS）。
**跑完不许回头改这里的门槛**；结果如实判，包括"无收益"。

## 1. 臂（两臂除"记忆挂哪一层"外逐项同口径）

| 臂 | 记忆 | 核 | 老头 | 新头 | 记忆门控 |
|---|---|---|---|---|---|
| **C（核级）** | `CoreNDB`：核编码阶段读写，预编码只写、卡片阶段表冻结 | 可训 | 冻结 | negation 可训 | `write_gate/read_gate/write_scale/read_scale` 可训 |
| **L（卡级）** | `MentionNDB` 挂在 person 头上：解码阶段读写 | 可训 | 冻结 | negation 可训 | person 卡的 `MentionNDB` 门控可训 |

- 两臂都走 `experiments/core_keep/train_core_keep.py` 的 J3 口径：温启动
  （核 ← `checkpoints/base_encoder.pt`，四张老卡头 ← `experiments/capability_map/cards/{c}_frozen_s{S}.pt`），
  老头冻结、新头（negation）可训，AdamW + cosine，epochs=16、steps_per_epoch=84、
  batch=64、lr_base=3e-4、lr_head=1e-3、clip=1.0。
- task_order = `pronoun sentiment relation person negation`（与 core_keep 相同，数据只读复用
  `experiments/core_keep/cache/*.pkl`）。
- seed ∈ {42, 43} ⇒ 共 **4 次训练**（C×2 + L×2）。

## 2. 评测口径

- **P1（跨段收益，主判据）**：强制按句切段 —— 对 person 的每条 val 样本用
  `split_with_global_offsets(text, max_chunk_len=40)` 切成若干段（不看 spec 的 window 策略），
  逐段 tokenize（`max_length=spec.max_len=120, truncation`）→ 逐段编码 → 逐段解码，
  锚点回拼全局坐标后与整段真值比。两臂差异：
  - C 臂：核级记忆按**样本**为作用域（写一遍全部段 → 冻结表 → 每段读）；
  - L 臂：person 卡的 `MentionNDB` **每段开始 `reset(1, device)`**（= 现引擎
    `engine.py` 段边界清零的行为，即"逐段 reset 的卡级记忆"）。
- val 集 = `capability_map` 的 `split_of(cap, seed)` 前 `max(200, n//10)` 条（与 core_keep 的
  ALIGN_CHECK 同一批；脚本内自检逐条相等，不等则作废）。
- 指标沿用 `runtime.evaluate_task` 的同口径：`repeat_mention_acc`（重复提及 id 对不对）、
  `first_mention_acc`、`id_acc`、`exact_match`（整句多重集相等）。
- **P2（温启动不伤老卡）**：4 张老卡的 `exact_match`，参照 = 冻结核 + 冻结老头
  （`base_encoder.pt` + `capability_map/cards/{c}_frozen_s{S}.pt`，同一 val 集同代码现算）。
  噪声带按卡、**不许统一阈值**：pronoun 0.0283 / sentiment 0.0041 / relation 0.0139 /
  person 0.0033（抄自 `experiments/core_keep/eval_core_keep.py` 的 `BAND`）。
- **P3（代价）**：训练 `train_sec` / `peak_mem_mb` / `sec_per_step`（两臂各报），
  推理 `ms_per_predict`（同一 16 条文本 × 4 卡，各跑 ≥3 遍取中位）与记忆表显存。

## 3. 判据（跑前写死）

**P1 通过 ⟺** 对两个指标（`repeat_mention_acc`、`exact_match`）分别要求：
1. 每个 seed 上 Δ_s = C臂_s − L臂_s > 0（两 seed **同号**且为正）；
2. Δ_s ≥ `spread`，其中 `spread = max(|C42 − C43|, |L42 − L43|)` —— 即该指标在两臂各自的
   **两 seed 极差**取较大者；
3. 同时另报更严的口径 `spread_strict = max(四次运行) − min(四次运行)`，
   **判据以 `spread` 为准，`spread_strict` 只作参考并如实列出**（不因结果不好而改判）。
   P1 不过 ⇒ 如实判"核级记忆相对逐段 reset 的卡级记忆**无收益**"，不许改判据。

**P2 通过 ⟺** 两臂 × 两 seed 上，4 张老卡全部满足 `Δ = exact(臂) − exact(参照) ≥ −BAND[卡]`，
且每张卡两个 seed 的 Δ **同号**。任一条不满足即 P2 不过，如实写"伤老卡"。

**P3** 不设门槛，只报数值（超时/显存异常须如实说明）。

**阶段 2 总通过 = P1 ∧ P2**（P3 只报告）。P1 不过也必须交 P2/P3 的完整数字。

## 4. 纪律（同任务书）

只许写 `experiments/core_ndb/`、`logs/core_ndb/`、`/tmp`；禁改 `src/`、`training/`、
`tests/`、`dev-notes/`、其它 `experiments/`；禁 `git commit/stash/checkout/restore/clean`。
训练一律 `systemd-run --user --unit=<名> --collect --property=WorkingDirectory=...`
`--setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1`；
开训前 `systemctl --user is-active 'dtseek*'` 忙则排队，绝不杀别人进程；
sentiment 数据缓存只读复用 `experiments/core_keep/cache/sentiment_32000.pkl`（不重建）。
