#!/bin/bash
# 价值观卡 阶段 1 —— 排队等 GPU → 主臂 2 seed + 随机标签对照 2 seed → 评测 → 汇总。
# 只写 experiments/value_card/ 与 logs/value_card_stage1.log；不改 src/、不碰他人进程。
set -euo pipefail
cd /home/vesita/coding/my/DTSeek
EX=experiments/value_card
mkdir -p "$EX/cards" "$EX/results" logs

echo "=== VALUECARD_STAGE1 start $(date -Is) ==="

# --- 1) 排队：等其它实验的训练进程退出（绝不杀他人进程；超时则退出不抢卡） ---
MAX_WAIT=14400
waited=0
while :; do
  busy=0
  for p in $(pgrep -f "\.venv/bin/python" || true); do
    cmd=$(tr '\0' ' ' < "/proc/$p/cmdline" 2>/dev/null || true)
    case "$cmd" in
      *experiments/value_card/*) ;;
      *experiments/*) busy=$((busy+1)) ;;
    esac
  done
  if [ "$busy" -eq 0 ]; then
    break
  fi
  if [ "$waited" -ge "$MAX_WAIT" ]; then
    echo "WAIT_TIMEOUT 其它实验仍占用 GPU（${busy} 个进程），已等 ${waited}s ⇒ 退出，不抢卡"
    exit 3
  fi
  echo "QUEUE 其它实验仍在跑（${busy} 个训练进程），已等 ${waited}s ..."
  sleep 60
  waited=$((waited + 60))
done
echo "=== GPU 空闲（等了 ${waited}s），开训 $(date -Is) ==="

# --- 2) 数据（缓存：stats.json 已在就跳过构建） ---
if [ ! -f "$EX/data/stats.json" ]; then
  uv run python "$EX/build_data.py"
fi

# --- 3) 主臂：只训头，seed 42/43 ---
for S in 42 43; do
  echo "--- [main s$S] train"
  uv run python "$EX/run_card.py" --out "$EX/cards/value_s$S.pt" --seed "$S" \
      --epochs 12 --steps-per-epoch 150 --samples 8100
  echo "--- [main s$S] eval"
  uv run python "$EX/eval_card.py" --ckpt "$EX/cards/value_s$S.pt" \
      --out "$EX/results/main_s$S.json" --tag "main_s$S"
done

# --- 4) 随机标签对照（V-bonus）：test + adversarial ---
for S in 42 43; do
  echo "--- [shuf s$S] train（随机标签对照）"
  uv run python "$EX/run_card.py" --shuf --shuf-seed $((S * 1000 + 7)) \
      --out "$EX/cards/shuf_s$S.pt" --seed "$S" \
      --epochs 12 --steps-per-epoch 150 --samples 8100
  echo "--- [shuf s$S] eval"
  uv run python "$EX/eval_card.py" --ckpt "$EX/cards/shuf_s$S.pt" \
      --out "$EX/results/shuf_s$S.json" --tag "shuf_s$S" --splits test,adversarial
done

# --- 5) 汇总（判据全部来自 PREREG §4） ---
echo "--- summarize"
uv run python "$EX/summarize.py"

echo "=== VALUECARD_STAGE1_DONE $(date -Is) ==="
