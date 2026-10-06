# core_generalize 原始命令 / unit / 日志对照

工作目录 `/home/vesita/coding/my/DTSeek`。所有长跑均用 `systemd-run --user`，
带 `PYTHONUNBUFFERED=1` 与 `HSA_OVERRIDE_GFX_VERSION=10.3.0`；**没有杀过任何别人的进程**。

## 数据准备（前台一次性，约 40 s；sentiment 等 5 份从 capability_map cache 只读复用）

```bash
uv run python -u experiments/core_generalize/prepare.py
```
日志：stdout（本文件上方对话）；产物 `experiments/core_generalize/{data_info.json,cache/}`

## 预注册（**在任何训练之前**）

- `experiments/core_generalize/PREREG.md` mtime `2026-10-06 08:42:31 +0800`
- `experiments/core_generalize/PREREG.ts` = 1791247488 / 2026-10-06T08:44:48+08:00
- 第一次训练（冒烟）启动于 `08:45:04` ⇒ 预注册早于训练 ≥33 s。

## 冒烟（验 step-0 对账，产物写 /tmp，不计入实验）

```bash
systemd-run --user --unit=dtseek-cg-smoke --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  /usr/bin/uv run python -u experiments/core_generalize/train_core_arm.py \
  --arm C1 --seed 42 --epochs 1 --steps-per-epoch 4 \
  --out /tmp/cg_smoke/C1_smoke.pt --metrics /tmp/cg_smoke/C1_smoke_metrics.json
```
结果：`SELFTEST_init ... step-0 exact=0.793750 | frozen基线=0.793750 Δ=0.000e+00 ✅`

## 阶段 A：hold-out 可学性预检（unit `dtseek-cg-pre`）

```bash
systemd-run --user --unit=dtseek-cg-pre --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  /usr/bin/bash experiments/core_generalize/run_precheck.sh
```
日志：`experiments/core_generalize/logs/PRE_{idiom,ownership}_s{42,43}.log`
产物：`experiments/core_generalize/cores/PRE_*_{,.}metrics.json`、`precheck.json`、`holdouts.txt`

## 阶段 B：四个核臂（unit `dtseek-cg-cores`）

```bash
systemd-run --user --unit=dtseek-cg-cores --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  /usr/bin/bash experiments/core_generalize/run_cores.sh
```
日志：`experiments/core_generalize/logs/{C1,C3,C5}_s{42,43}.log`、`C1x5_s42.log`
产物：`experiments/core_generalize/cores/{C1,C3,C5}_s{42,43}_{,.}metrics.json`、`C1x5_s42_*`

## 阶段 C：探针头（unit `dtseek-cg-probes`）

```bash
systemd-run --user --unit=dtseek-cg-probes --collect \
  --property=WorkingDirectory=/home/vesita/coding/my/DTSeek \
  --setenv=HSA_OVERRIDE_GFX_VERSION=10.3.0 --setenv=PYTHONUNBUFFERED=1 \
  /usr/bin/bash experiments/core_generalize/run_probes.sh
```
日志：`experiments/core_generalize/logs/{B0,C1_s42,...}_{idiom,ownership}_s{42,43}.log`
产物：`experiments/core_generalize/probes/*.json`

## 汇总

```bash
uv run python -u experiments/core_generalize/analyze.py   # → summary.json
```
