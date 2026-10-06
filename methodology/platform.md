# platform —— 平台与环境准绳：加速器、长跑、资源、入库

> 四行格式与引用缩写见 [`README.md`](./README.md)。本机 = **AMD RX 6600（Navi23 / RDNA2 / 8G）+ ROCm 6.4**，单卡。

## 1. 加速器

### 每个 GPU 进程必须带 `HSA_OVERRIDE_GFX_VERSION=10.3.0`，否则第一个 kernel 就报 `HIP error: invalid device function`
- **为什么**：本机是 **gfx1032**，ROCm 6.4 轮子只带 gfx1030 ⇒ 不 override 直接失败。
- **怎么用**：脚本首行 `export HSA_OVERRIDE_GFX_VERSION=10.3.0`；`systemd-run` 用 `--setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0`。
  ⚠️ **uv 不会自动加载 `.env`**（实测 `HSA_OVERRIDE` 为 None）—— override 已放在 venv 本地的 `sitecustomize.py`，但**子进程 / 单元启动时仍要显式传**。
- **证据**：实测 ＋ `dev-notes/12:189-192`；实例 `experiments/additivity/run_all.sh:7`、`experiments/capability_map/REPORT.md:127`。
- **边界/反例**：只对本机这套 ROCm 6.4 轮子成立；换机器 / 换轮子要重测 `torch.cuda.is_available()` 与 device 名（`dev-notes/12:57-58`）。

### **永不升级 ROCm 7**：ROCm 7.x 已移除 RDNA2 支持
- **为什么**：本机是 RDNA2；升级到 rocm7 ⇒ 现有可用组合（`torch==2.9.1+rocm6.4` + `pytorch-triton-rocm==3.5.1`，本地 wheel `find-links`）直接失效。
- **怎么用**：依赖升级时**先查 RDNA2 是否还在支持列表**；`uv.toml` 只留本机 wheel `find-links`，别顺手 bump。
- **证据**：文档事实（nanoSeek 已验证的组合 + 明确标注）＋ `dev-notes/12:52-56`、`dev-notes/12:193`。
- **边界/反例**：这是**当前硬件**下的结论；换卡（非 RDNA2）后前提消失，要重新评估，不是"永远不升级"的抽象原则。

## 2. 长跑

### 长跑一律 `systemd-run --user --unit=<名> --collect`，**不用 `setsid nohup`**
- **为什么**：harness 每次工具调用都会建一个 systemd scope，**会话重启会 SIGKILL 整条 cgroup**；`setsid` 只脱离会话/tty，**脱离不了 cgroup** ⇒ 照样被杀（现场特征：进程凭空消失、日志无 traceback、`journalctl` 里有 `Killed unit cgroup … with SIGKILL on client request`）。
  （⚠️ cgroup/SIGKILL 这段的出处在**仓库外**：harness 全局 `~/.dsh/AGENTS.md`；仓库内只固定了"怎么启动"，见下。）
- **怎么用**：`systemd-run --user --unit=<名> --collect --property=WorkingDirectory=<repo> --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 <脚本>`；终态看 `systemctl --user is-active <名>` 由 active → inactive。
- **证据**：规范（多实验统一执行）＋ `experiments/core_generalize/COMMANDS.md:3`、`experiments/capability_map/PREREG.md:136-138`、`experiments/ndb_recheck/REPORT.md:3`、`experiments/additivity/REPORT.md:136-142`。
- **边界/反例**：**有明确终点的任务**（构建 / 测试 / 一次性扫描）正常等到结束即可，不必强行后台化 —— 该不该等看"终点是否可知"，不看耗时长短。

## 3. 单卡资源

### 单卡 GPU：启动前查 `systemctl --user is-active 'dtseek-*'`，**忙则排队，绝不杀别人的进程**
- **为什么**：同一张卡上并跑会互相干扰，既有量化的污染，也有直接 hang。
- **怎么用**：每个训练脚本启动前做占用检查并排队；只 kill **自己**的 unit；报告里如实登记与哪些 unit 并跑过。
- **证据**：规范 ＋ `experiments/cumulative_add/PREREG.md:137-138`。
- **边界/反例**：排队只解决"启动时已知"的占用，**跑中新增**的并跑要靠登记与复跑来判定（下条）。

### 并跑会污染测量：同配置在重载下差 ~13%，且会把进程挂成**空转 2.5h**
- **为什么**：实测 —— `s42` 步 2/3 与 `dtseek-cg-budget4x` 并跑，同配置 `ctrl_ownership` 在重载下 0.339 / 探针负载下 0.299 ⇒ **争用可造成 ~+13%**；另有 stage B 在 15:24:40 后 **CPU 199%、log 不动、空转 2.5h**（疑 ROCm/争用 hang），kill 自己的 unit 重启后重跑。
- **怎么用**：① 关键结论**以未受污染的那次为准**（本项目据此把代价结论**以 s43 为准**、s42 只作上界）；② 报告里写明并跑窗口与"该数字被污染"；③ 出现"CPU 满载但 log 不动"先怀疑争用 hang，kill 自己的 unit 重启，**不杀他人进程**。
- **证据**：单次实测（运维现场）＋ `experiments/cumulative_add/RESULTS.md:93-95`、`:123`、`experiments/anchored_select/results.md:109-111`。
- **边界/反例**：+13% 是**这一处**的读数，不是通用系数；未量化过的负载不要拿它去"校正"数据，只能用来**判定该跑是否可用**。

### ROCm 并发争用会在仓库根掉 `gpucore.*`（单个 ~160MB）—— 已 ignore，但入库前要体检
- **为什么**：崩溃转储体积大且不可复现，直接进 git 会把仓库撑爆。
- **怎么用**：保持 `.gitignore` 的 `gpucore.*`；`git status` 出现大文件先看是不是崩溃转储 / 缓存，**不要 `git add -f`**。
- **证据**：规范（`.gitignore` 注释写明成因与量级）＋ `.gitignore:43-44`。
- **边界/反例**：ignore 不等于不用管 —— 它同时是**"刚才发生过 GPU fault"的信号**，出现时该查并发与显存。

### 权重与训练产物不入库：`*.pt` / `logs/` / `experiments/**/cache/` 已 ignore
- **为什么**：体积大且**可复现**（clone 后跑一次多任务训练即可重建）；历史上 49MB 二进制入库是明确的踩坑。
- **怎么用**：新产物先问"能不能重建"；能重建 → 加 ignore 规则（带一句**为什么**的注释）；不能重建且必须留 → 走大文件方案，不硬塞进 git。
- **证据**：规范 ＋ `.gitignore:15-24,40-42`、`dev-notes/04:152`、`README.md:80`。
- **边界/反例**：**判据文件、`stats.json`、`summary.json`、sha256 记录不是"训练产物"**，它们是结论的证据，必须入库 —— ignore 的是可重建的二进制与缓存。

### 实验派生缓存（`experiments/**/cache/`、`person_val_seed*.json`、`*.pkl`）一律不入库
- **为什么**：数据集转储可由脚本重建，体积大。
- **怎么用**：新建实验目录时确认 `.gitignore` 的 `experiments/**/cache/` 已覆盖；缓存放实验目录**内**而不是仓库根。
- **证据**：规范 ＋ `.gitignore:40-42`。
- **边界/反例**：缓存**参与哈希对账**时（如逐样本 sha256）要保证缓存来源可追溯，不能因为"不入库"就丢掉生成参数。
