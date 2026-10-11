#!/usr/bin/env bash
# ★ 本脚本由 systemd-run 拉起，勿直接跑
#
# H2（PREREG_GENOBJ.md）：自由生成必须被当目标训练（生成目标轴）
# 8 个 run = --obj ∈ {tf, roll} × seed ∈ {1234, 5678, 9012, 3456}
#
# 运行方式示例：
#   systemd-run --user --unit=flowme-genobj-run --collect experiments/meme_anchor/runs_genobj.sh
#
set -euo pipefail

cd /home/vesita/coding/my/flowme

export HSA_OVERRIDE_GFX_VERSION=10.3.0

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python"

ck() {
  echo "logs/86g_ckpt_none_AaBb_AbBa_a0.5_l0.0_s$1.pt"
}

# ==========================================
# 跑前预检（fail-closed：缺任一即退出）
# ==========================================
echo "[GENOBJ] 开始跑前预检..."

# 1. 预检驱动
if [ ! -f "stages/86g_zh_en.py" ]; then
  echo "[GENOBJ] 预检失败：缺驱动 stages/86g_zh_en.py" >&2
  exit 2
fi

# 2. 预检 --obj 选项存在
if ! $PY stages/86g_zh_en.py --help 2>&1 | grep -q -- --obj; then
  echo "[GENOBJ] 预检失败：驱动 stages/86g_zh_en.py 缺少 --obj 选项" >&2
  exit 3
fi

# 3. 预检 4 个 seed 的 v0 输入卡 ckpt
SEEDS=(1234 5678 9012 3456)
for S in "${SEEDS[@]}"; do
  CKPT="$(ck "$S")"
  if [ ! -f "$CKPT" ]; then
    echo "[GENOBJ] 预检失败：缺 ckpt $CKPT (seed=$S)" >&2
    exit 4
  fi
done

echo "[GENOBJ] 预检全部通过：驱动、--obj 选项及 4 个 ckpt 均就绪；start $(date -Is)"

# ==========================================
# 顺序跑 8 个 run
# ==========================================
OBJS=(tf roll)

for OBJ in "${OBJS[@]}"; do
  for S in "${SEEDS[@]}"; do
    LOG_FILE="logs/86g_gen_${OBJ}_l0.3_s${S}.log"
    echo "[GENOBJ] ==== 目标 obj=$OBJ seed=$S start $(date -Is) ===="
    
    # 构建命令参数（roll 带 --roll-w 0.5）
    EXTRA_ARGS=()
    if [ "$OBJ" = "roll" ]; then
      EXTRA_ARGS=(--roll-w 0.5)
    fi

    $PY stages/86g_zh_en.py \
      --run 0.3 "$S" \
      --steps 8000 \
      --eval-every 1000 \
      --maps AaBb,AbBa \
      --loss infonce \
      --bridge parser \
      --bridge-mid 64 \
      --align-on bridge \
      --init-inputs "$(ck "$S")" \
      --freeze-inputs \
      --init-outputs \
      --obj "$OBJ" \
      "${EXTRA_ARGS[@]}" \
      > "$LOG_FILE" 2>&1 || true

    echo "[GENOBJ] obj=$OBJ seed=$S finished $(date -Is)"
  done
done

# ==========================================
# 跑后核验：每个 run 的 log 里必须有 [RUN-FINAL]
# ==========================================
echo "[GENOBJ] 开始跑后核验 [RUN-FINAL] 标记..."
ALL_PASSED=true

for OBJ in "${OBJS[@]}"; do
  for S in "${SEEDS[@]}"; do
    LOG_FILE="logs/86g_gen_${OBJ}_l0.3_s${S}.log"
    if [ ! -f "$LOG_FILE" ]; then
      echo "[GENOBJ] 核验失败：日志文件不存在 $LOG_FILE" >&2
      ALL_PASSED=false
    elif ! grep -q "\[RUN-FINAL\]" "$LOG_FILE"; then
      echo "[GENOBJ] 核验失败：日志缺少 [RUN-FINAL] 标记 $LOG_FILE" >&2
      ALL_PASSED=false
    else
      echo "[GENOBJ] 核验通过：$LOG_FILE 包含 [RUN-FINAL]"
    fi
  done
done

if [ "$ALL_PASSED" = false ]; then
  echo "[GENOBJ] 跑后核验失败：部分 run 未正常产出 [RUN-FINAL]，整批非零退出" >&2
  exit 5
fi

echo "[GENOBJ-DONE] 整批 8 个 run 全部完成并通过核验 $(date -Is)"
exit 0
