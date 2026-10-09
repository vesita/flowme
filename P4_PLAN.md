# P4 预研：用 `cards/` 框架重跑 A/B/C/D/E（最小改造 + 判据）

> 只读调研；P3 的 `stages/46_*`、`logs/46_*` 未触碰。证据一律给 `文件:行`。

## 1. 必对齐清单（按风险排序）

| # | 要对齐的 | 依据（现状 / 目标） | 风险 |
|---|---|---|---|
| 1 | 数据 RNG：5 桶顺序 + 两条 python RNG 种子 | `stages/19:313-321`（`Rng(14000+bi*7)`/`Rng(14900+bi*7)`）、`Rng=random.Random` `stages/19:236-253`；`cards/example_add3d.py:82,241-242` 已复刻 | ★★★ 错则 train/test 全变且不报错 |
| 2 | 训练随机流结构（R34）：dropout 独立流 + 重洗吃全局流 | `stages/19:445-446,456`；`cards/example_add3d.py:88,277-283,306-307` | ★★★ F1 实测 −18.5pp，`methodology/RULES.md:45` |
| 3 | 权重初始化顺序 emb→in_enc→card→head | `stages/19:385-388`、`stages/36:228-234`、`stages/44:242-245`；`cards/pipeline.py:37-39` 天然同序 | ★★★ 若构造完 pipeline 再替换 target，head 的初始化抽样错位，XCHECK 抓不到 |
| 4 | 卡内结构逐字（模块顺序/dropout0.1/gelu/norm_first） | `stages/36:146-206`；`cards/target.py:22-30`、`cards/input.py:30-34` | ★★☆ 只覆盖 A，B/C/D/E 要新写 |
| 5 | 残差位置 | `cards/target.py:28-30`「x+think(x)」⇔ `stages/36:151-152`；B/C/D/E 的残差在卡内（D 的 gate、E 的 dropout） | ★★☆ |
| 6 | mask / PE / MAXLEN 截断 | `cards/input.py:20-27,58-64` ⇔ `stages/19:363-369,393-397`；截断 `stages/19:288-307`（MIN_T=48,TAIL_KEEP=24） | ★★☆ 必须显式传 nhead/maxlen/pad_id（pipeline 默认 pad_id=0，实测 PAD=131，`logs/e6_sat.log:8`） |
| 7 | 优化器/超参/全参数训练 | `stages/19:448-463`（AdamW lr=1e-3、batch=32、order 用 Generator(seed)）；`cards/example_add3d.py:570-576` 的 freeze 只是演示 | ★★☆ 入口若沿用 freeze ⇒ 退化成只训 target |
| 8 | 评估口径：贪心 batch=1、parse_ans、gold、MAX_GEN=48、R28 自检 | `stages/19:578-621,637-655`、`stages/36:435-437`；`cards/example_add3d.py:338-385` 已逐字 | ★★☆ |
| 9 | ckpt 键名/格式映射 | 19 是 `{format,contract,state}`（`cards/base.py:168-172`），36/44 是裸 state_dict（`stages/36:423`）；A@12000 来自 S38，`thought.→target.think.`（`stages/44:86,285-288`） | ★★☆ |
| 10 | 设备数值差（CPU vs GPU） | `cards` 默认 CPU（`cards/example_add3d.py:84`）；stages 强 GPU（`stages/19:46-52`、`stages/44:98`） | ★★★ 同设备才谈逐位，跨设备只报 EM ≤0.13pp |
| 11 | 契约：自定义 target 卡须继承 `Card`、三卡契约一致 | `cards/base.py:88-156`、`cards/pipeline.py:52-56` | ★☆☆ |

## 2. 最小改造方案（不改 `cards/` 核心，只加一个实验入口）

新增 `cards/exp_struct_ablation.py`（P4 执行时唯一新文件）：
1. 数据/RNG/评测：`exec` `stages/19` 文本到 `"# ③ 主循环"`（与 36/44 同法，`stages/36:54-57`）⇒ 与实验脚本同源；
2. B/C/D/E：`class CardB/CardD/CardE(Card)`（`io=IO_D_TO_D`），`apply_card` 逐字抄 `stages/36:155-221` 的 forward 体（含模块定义顺序 ⇒ 初始化顺序）；
3. `class StructPipeline(CardPipeline)` 只覆盖 `__init__`：按 input→target(=A|B..E)→output 顺序构造，再补 `blackboard/frozen_cards/route_history`；其余 `route/freeze/snapshot/mv/load` 全部继承 ⇒ 契约/黑板/快照/MV 保持 `cards/` 写法；
4. 参照物（oracle）不重写：把 `stages/36` 的 `CardA..Cards36`、`stages/44` 的 `Cards44/Cards38Clone` 从**原文本 exec** 进本进程 ⇒ 参照物=实验脚本原码，避免"自己和自己比"的空测试（R29 类风险）。
   必须抄 stages 的：数据生成 + 两条 RNG 流 + 超参 + AdamW/重洗 + 贪心评测/parse_ans；
   必须保持 cards 的：形状契约（`cards/base.py:108-156`）、Contract/快照/MV（`cards/base.py:98-105,184-194`、`cards/pipeline.py:140-163`）、黑板 route（`cards/pipeline.py:59-78`）、freeze 的 `max|Δθ|==0` 断言（`cards/pipeline.py:128-134`）。

## 3. XCHECK 判据（写死；零训练，CPU 秒级）

- 门 0（数据）：TEST `sha256[:16]==b34e7ea515203227`、`(a,b)` 泄漏 `==16/800`（`stages/36:79-80,110-133`）。
- 门 1（实现，最硬）：把 36 的 A/B/C/D/E@6000 与 44 的 B/C/D/E@12000 的 ckpt 灌进 cards 路径；**同一设备（都 CPU）**、同一 ids，cards 容器 vs oracle 容器的 logits **`max|Δ|` 必须 `== 0.0`**，且 batch=1 贪心 pred 列表逐样本相同。
- 门 2（EM）：EM(batch=1,n=800) vs 入库值（`logs/e6_sat.log` [MAIN-TABLE]、`logs/44_results.jsonl` 的 `em_full`）⇒ `|ΔEM| ≤ 0.13pp`（=1/800，CPU/GPU 浮点同量级；F1 先例 Δ=+0.00pp，`methodology/RULES.md:45`）。
- 门 3（不通过 ⇒ 按序定位）：① 门 0 挂 ⇒ 数据 RNG/桶序；② `strict` load 的 missing/unexpected ⇒ 键名映射；③ 同设备 `max|Δ|>0` ⇒ 逐卡比 `input.emb/input.enc/target/output.head` 与 mask、`pe[:n]` 的 sha256 ⇒ 定位实现；④ 同设备 `max|Δ|==0` 但 EM 差 >0.13pp ⇒ 评估口径（parse/gold/cap/EOS）；⑤ 别把 CPU/GPU 浮点差当框架差异。
- 通过 ⇒ 可选：1 臂 × 1 seed × 12000 步从零训练，报 EM 差（并报 R28 自检与是否饱和）。

## 4. 预算与已知风险

- 预算：XCHECK 零训练（加载 + 800×贪心，CPU 秒级）；从零训练 12000 步 ≈ 9.1min 训练 / 10.3min 跑完（`logs/p0_struct.log` [META] B/C），6000 步 ≈4.6min（`logs/e6_sat.log` [META]）。R36：结论只认 @12000（`methodology/RULES.md:47`）。
- 风险①：44 不存 A 臂，A@12000 只能用 `logs/38_ckpt_w1_s12000_seed*.pt` 映射 `thought.→target.think.`（`stages/44:86,285-288`）。
- 风险②：MV/接口快照只覆盖 input+output（`cards/version.py:31,139-149`）⇒ B/C/D/E 换 target **MV 相同**；别拿 MV 当结构标签，逐位一致只能靠权重/logits 比。
- 风险③：XCHECK 在 CPU 跑而复现值来自 GPU 训练 ⇒ 只能声明"同设备 `max|Δ|=0` 且 EM ≤0.13pp"，不能声称"与 GPU 训练逐位一致"。
- 风险④：`cards/example_add3d.py:567-579` 的 freeze+5 步是演示，入口沿用会变成只训 target 卡。
