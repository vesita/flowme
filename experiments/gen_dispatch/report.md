# 生成卡接入卡外手写调度层 —— 端到端实验报告（`experiments/gen_dispatch/`）

口径：**实测** = 本目录跑出来的数字（`results_*.json` 为准）；**推断** = 由数字推出；**构造规定** = 我人为规定的规则。
`PREREG.md` mtime **02:23:56**，早于本目录最早日志（`logs/gen_dispatch_sample_023530.log` 02:35:30）与首个真卡批
（`logs/gen_dispatch_h1_024746.log` 02:47:46）⇒ 判据先于任何运行写死；跑完未改一字。

## 1. 接入方式（卡外手写，`src/` 零改动）

- **type → 调用计划**：`dialogue.turn_plan()`（= `dispatch.plan()` 换对话层三张位表）出 `Plan=[Step{...}]`；
  `Terminate(generate)` 那步 `module_id="output.generate"`；`pipeline.annotate_plan()` 只给这一步**追加两个手写参数**
  `bag_cards=person|pronoun|relation|sentiment|negation|idiom`、`skeleton_table=render.SKELETONS(16)`，指针/拒答的 Step 原样不动。
- **执行**：按 `admissible_actions()` 授权集整批调卡（与官方 `dialogue.respond` 同口径；计划里 CallCard 步只点 dispatch 按位序挑的第一张，实际跑过的卡全记在 `cards_run`）→ 算两条可核对谓词 → `rules.switch_channel`（R-CH0..5）→ `rules.seal` **唯一出口**出记录。
- **G**（可合法渲染）＝穷举官方 16 骨架使 `check_structure()==[]` 且 `render()` 出 `kind="text"` 且 `check_deref()==[]`；
  **P**（指针有据）＝计划授权调用过的卡里 ≥1 张发射 ≥1 个合法切片（区间在原文内、逐字、非背景）。
- **生成分支**：`build_bag`（`card_flow` 5 条谓词筛选 + `type_pos` 三段词典定型）→ `search_renderable`
  （穷举 `_SKELETON_LIST` → 官方两道检查）→ `{kind:"text", text, evidence, plan_step_id, instruction, ref_map}`。
- **指针分支**：`dialogue.compose_reply`（模板+逐字切片）；它返回**闭区间**（引擎口径 `s0/e0`），出口统一转**半开区间**再入 evidence。
- **拒答分支**：`seal` 把 render/dialogue 的 `（拒答）…` 占位串并进 `reason`、**丢掉 `text` 键** ⇒ 拒答时 `text` 缺省、`reason` 必填。
- **成语卡（供「动」槽）**：`cumulative_add/ctrl_idiom_s42.pt` 是一体训练快照（`format=None`），`attach()` 直接拒；实测其 `doc_encoder` 与 `checkpoints/base_encoder.pt` **最大绝对差 0.0438**（换基座后 40 句里 18 句锚点变、10 句理想成语句定域更差）⇒ 拆成一对产物落本目录 `cache/`，用**它自带的基座单独起一个引擎**（`idiom_card.py`）；锚点是原文区间、与编码器无关 ⇒ 两引擎锚点合进同一个袋。

## 2. 开关规则原文（跑前写死，`rules.RULE_TEXT`）+ 选中比例

```
R-CH0  T == reject                          ⇒ reject   （text 缺省，reason 必填）
R-CH1  T == generate 且 G                   ⇒ generate （两道检查 0 问题才出 text）
R-CH2  T == generate 且 ¬G 且 P             ⇒ pointer  （降级；reason 记「生成不可行」）
R-CH3  T == generate 且 ¬G 且 ¬P            ⇒ reject
R-CH4  T == pointer  且 P                   ⇒ pointer
R-CH5  T == pointer  且 ¬P                  ⇒ reject   （指针无据不许编，G 也不许越权升级）
行序固定 = 上表顺序；同一输入两次执行命中同一行；未知 T ⇒ fail-closed 拒答。
```

| 批 | n | 终点 generate | generate 通道 | pointer | reject | R-CH1 | R-CH2 | R-CH4 | R-CH5 |
|---|---|---|---|---|---|---|---|---|---|
| 随机（全语料 reservoir seed=42） | 998 | 476 (47.7%) | **2 (0.20%)** | 984 (98.60%) | 12 (1.20%) | 2 | 474 | 510 | 12 |
| 富集（含 `IDIOM_TYPE` 词条） | 398 | 244 (61.3%) | **8 (2.01%)** | 388 (97.49%) | 2 (0.50%) | 8 | 236 | 152 | 2 |

`rule_audit()`（`inspect.getsource` 断言 R-CH 源码不含 `weight|state_dict|model|score|predict(`）：**ok=True, hits=[]**。
规则读的三个量 = dispatch 纯 if-else 的终点 / 穷举手写骨架表+官方谓词 / 卡切片存在性（布尔）；**不含任何学出来的量**。
机制发现（实测）：conf 取**最近一次调卡**的档 ⇒ `plain` 下走到 generate 终点实质要求否定卡发射且 conf≥0.5（`CONF_MID=0.5` 手写常量）。
随机批走到 generate 终点 476 条，其中 G 成立仅 **2** 条（G 本身成立 4 条，另 2 条终点=pointer、按行序仍走指针）⇒ **瓶颈是渲染可行性（缺动槽），不是调度**。

## 3. 真实生成样例（富集批 8 条全给；随机批另有 2 条在 `results_random.json`）

1. 输入「用户关注点不是花哨的图标，炫酷的动画，那些都是锦上添花」→ 计划 `sentiment → output.generate`(step:1, 卡=sentiment,negation)
   → 指令 `R02[0,1]` → 文本「点锦上添花。」→ 引用 `content:点[4,5] + content:锦上添花[23,27] + 表:。` → 通道 **generate**（R-CH1）
2. 输入「- **从微小习惯开始**:自律需要循序渐进」→ 计划 `sentiment → output.generate`(step:1) → 指令 `R02[0,1]` → 文本「小开始。」
   → 引用 `content:小[6,7] + content:开始[9,11] + 表:。` → 通道 **generate**（R-CH1）
3. 输入「如果它本名就叫Bengal，估计中国也不会画蛇添足加上「国」的」→ 计划 `sentiment → output.generate`(step:1)
   → 指令 `R02[0,2]` → 文本「它画蛇添足。」→ 引用 `content:它[2,3] + content:画蛇添足[21,25] + 表:。` → 通道 **generate**（R-CH1）
4. 输入「孩子问「为什么古代诗人那么悠闲,不用养家糊口,还能到处旅游写那么多诗」→ 计划 `sentiment → output.generate`(step:1)
   → 指令 `R02[0,2]` → 文本「代养家糊口。」→ 引用 `content:代[8,9] + content:养家糊口[18,22] + 表:。` → 通道 **generate**（R-CH1）
5. 输入「有你喝的就不错了，还敢挑三拣四，明早不带你的份了」→ 计划 `sentiment → output.generate`(step:1)
   → 指令 `R02[0,2]` → 文本「你挑三拣四。」→ 引用 `content:你[1,2] + content:挑三拣四[11,15] + 表:。` → 通道 **generate**（R-CH1）
6. 输入「如果说，在现实中他真的能这么呼风唤雨，只能为中科大感到遗憾」→ 计划 `sentiment → output.generate`(step:1)
   → 指令 `R01[0,1,2]` → 文本「他呼风唤雨遗憾。」→ 引用 `content:他[8,9] + content:呼风唤雨[14,18] + content:遗憾[27,29] + 表:。` → **generate**（R-CH1）
7. 输入「他们不固执己见,而是保持开放和成长的心态」→ 计划 `sentiment → output.generate`(step:1) → 指令 `R02[0,1]`
   → 文本「他们固执己见。」→ 引用 `content:他们[0,2] + content:固执己见[3,7] + 表:。` → 通道 **generate**（R-CH1）
8. 输入「我姐夫一意孤行，一定要带的孩子去墓地看奶奶」→ 计划 `sentiment → output.generate`(step:1, conf=中) → 指令 `R01[0,3,4]`
   → 文本「我一意孤行孤行。」→ 引用 `content:我[0,1] + content:一意孤行[3,7] + content:孤行[5,7] + 表:。` → 通道 **generate**（R-CH1）

每条的完整 `plan[]` 原文、`plan_steps`（含 `bag_cards`/`skeleton_table` 参数）、`evidence`（含 candidate_id）、`g_stats` 都在
`results_enriched.json` 的 `generate_samples` 里；所有 span 均为半开区间、`text == 输入[span]`（`verify_record` 独立重跑确认）。

## 4. 判据 H1–H6（逐条实测）+ `max_naive` + 判定

| # | 实测数字 | 结论 |
|---|---|---|
| **H1** | (a) `dispatch.plan()` 4860 态 ×2 遍 `as_tuple()` 全等（默认表 **与** 对话三张表各跑一遍，都 True）；(b) 两批各 100 条端到端 ×2 遍：`plan_mismatch=0`、`record_mismatch=0`（整条记录也 `==`） | **过** |
| **H2** | 随机 998 / 富集 398 共 1396 条：`contract_fail=0`；reject 12+2 条里 **`text` 字段出现 0 条、`reason` 空 0 条、正例误拒 0 条**；`kind=text` 里非法输出 **0** 条。C6 两侧一致：`render.render()` 原始 reject **11/11 违例**（带 `（拒答）…` 占位串 + `plan_step_id=''`）、`dialogue._reject()` 原始记录同样违例（且**无 `plan_step_id` 键**）→ 同一谓词归一后 **0 违例** | **过** |
| **H3** | 生成型记录随机 2 + 富集 8 = 10 条，**独立重跑** `check_structure` 问题总数 **0**、`check_deref` 问题总数 **0**（逐条 max=0） | **过** |
| **H4** | 选中比例见 §2 表（两批分别报）；`rule_audit` **ok=True, hits=[]**，规则原文见 §2 | **过** |
| **H5** | 对抗组 **A1–A14 逐条 pass=True**（PREREG §6 表实列 14 条，H5 行写的 A1–A12 是笔误；A1 题元互换/A2 无据逻辑词/A3 正对照/A4 缺块/A5 类型/A6 数字/A7 否定/A8 专名/A9 越界/A10 骨架 id/A11 重指派/A12 换方向/A13 type/A14 空输入） | **过** |
| **H6** | `git status --porcelain` 只有 `?? experiments/gen_dispatch/`；`git diff --stat -- src/` **空**（HEAD `b746fe4`）；`uv run python -m pytest -q tests/` = **246 passed**（与基线逐字同）；全仓 `pytest -q` = **258 passed** = 246 + 本目录 `test_contract.py` 的 12 条 | **过**（不触发 Δ 噪声带分支） |

**`max_naive`（必报，两组口径分开）**

| 基线 | 随机批 n=998 | 富集批 n=398 | 我方同批 |
|---|---|---|---|
| N1 全生成 | 非法 994，合法产出 4，**合法率 0.004** | 非法 385，合法产出 13，**0.0327** | 非法 **0** |
| N2 全指针 | 非法 12，**合法率 0.988**（=本批 `max_naive`） | 非法 2，**0.995**（=本批 `max_naive`） | 合法率 0.988 / 0.995（**与 max_naive 相同**） |
| N3 按 type 静态 | 与 N1/N2 逐数相同（本批全部 `plain`） | 同左 | 同上 |
| N4 词典假卡+同渲染层 | 200 句 **过两道检查 0 条** | 未跑（PREREG §4.5 只在随机批跑 N4） | 真卡 **2 条** |
| N5 echo 复述 | 过 H3 **0**（无 instruction/ref_map、evidence 空） | 0 | 10 条全过 |

`max_naive` = 0.988（随机）/ 0.995（富集）。**必须并列的结论**：我方合法输出率与免费规则 N2 全指针**完全相同**，
差异只在产出构成 —— 随机批多 2 条、富集批多 8 条**生成型** text，而 N2 一条也产不出、N1 会产出 994/385 条非法输出；
N4 上真卡 2 > 词典假卡 0，但 **n 极小、证据强度弱**。生成通道本身也是一条免费规则（穷举手写骨架表），
所以 N1/N2/N3/N5 是「我这条规则 vs 别的免费规则」，N4 才是「真卡 vs 词典卡」，两者不互冒。

**判定：H1–H5 全过（H6 也过）⇒ 接入成立。** 支撑：计划里真出现了 `output.generate` 步并带手写 `bag_cards` 参数、
10 条生成记录过两道检查 0 问题、1396 条记录契约 0 违例且拒答侧 `text` 缺省率 100%、14 条对抗全 pass、规则源码审计干净。
**但**（推断）通道价值被「没有动词卡」卡死：generate 通道只占 0.20%/2.01%，且这数字还依赖富集的选择偏倚。

## 5. 原始命令 / unit / 日志

```
uv run python experiments/gen_dispatch/run_e2e.py --mode sample      # 语料过滤+抽样（69s，16/18 文件）
uv run python experiments/gen_dispatch/run_e2e.py --mode selfcheck   # 假卡自检 + H1a/b + 规则审计
uv run python experiments/gen_dispatch/run_e2e.py --mode adv         # 对抗组 A1–A14
uv run python experiments/gen_dispatch/run_e2e.py --mode random      # 随机批 998（含 N4 假卡 200）
uv run python experiments/gen_dispatch/run_e2e.py --mode enriched    # 富集批 398
uv run python experiments/gen_dispatch/run_e2e.py --mode h1          # H1 专用 100 条
uv run python experiments/gen_dispatch/test_contract.py && uv run python -m pytest -q tests/   # 12 条全过 / 246 passed in 19.74s
systemd-run --user --unit=dtseek-gen-dispatch-all --collect --property=WorkingDirectory=... --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 uv run python .../run_e2e.py --mode all   # 汇总重跑，invocation 407fae10
```

语料过滤（实测，汉字数/非空白字符数，整文件流式）：**保留 16 文件**（与 `dev-notes/19` 名单一致，占比 0.576–0.865）；
**剔除 2 文件**：`code_alpaca_dialogue.txt` 0.0005、`gsm8k_cot_dialogue.txt` 0.0709。句子池 9,274,346 句，含 `IDIOM_TYPE`
（74 词条）3,684 句；随机 998 / 富集 398（去重、两集合互不重叠，seed=42，**未用 `fs[:N]`**）。
日志：`logs/gen_dispatch_{sample,selfcheck,adv,random,enriched,h1,all}.log`；结果 `experiments/gen_dispatch/results_*.json`
（`results_all.json` 与各单跑逐键一致 = 确定性复现）。

## 6. 遗留与不确定

- **实测已证实的瑕疵**：8 条生成样例里 **1 条（第 8 条）出现重叠 span**（`一意孤行[3,7]` 与 `孤行[5,7]`）导致输出重复
  「我一意孤行孤行。」—— 官方两道检查**不覆盖 span 重叠**，因此 H3 仍判 0 问题；我未因此改规则（PREREG 跑后不改）。
- **实测的输入质量**：多个样例的「名」槽取自 1–2 字碎片（点/小/代），R02 拼出的句子语法别扭（如「点锦上添花。」）—— 这是真卡锚点 + 官方骨架约束的结果，不是我加的噪声，但也**不是"好文本"**。
- **实测**：生成样例全部来自富集口径（含成语词条的句子，选择偏倚，PREREG §4.4 已单列）；随机口径 998 句只出 2 条。
- **推断**：`generate` 通道占比低的根因是**没有动词卡**（`card_flow/lexicon.py` 注明 verb 是夹具卡；真卡 120 句 `has动=0`），
  成语卡只能补上 74 词条那一小撮「动/形/名」。要把这条通道做宽，需要一张真动词卡 —— **本实验未实现**。
- **未实现（明说）**：① `experiments/two_channel_head` 的判别头**没有**接进来（PREREG §1 只读引用其数字）；
  ② PREREG §4.5 的 N4 只在随机批跑，富集批没有 N4 对照；③ PREREG §4.4 的扩样阶梯 M=800/1600 **没有触发**（M=400 已出 8 条 ≥5）；
  ④ 样例条目里没带「块 ← 哪张卡」的逐块归属（`evidence.candidate_id` 与 `cards_run` 在，但两者未在样例里连起来）。
- **不确定**：成语卡用自带基座虽比默认基座定域更准（10 句理想句实测），但**没有标注集**可算它的 P/R；两套基座混用的
  长期行为只做了 40 句锚点对比这一处实测。
- **基线数字（引自 `two_channel_head`，非本实验）**：`B_s42_steps30` 骨架 0.422/槽 0.835/joint 0.4148；满长 `B_s42` 骨架 0.6396/槽 0.965833/joint 0.6088（`W4 max_naive_train=0.5329`）。本报告没有与它们比较的实验。
