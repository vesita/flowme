#!/usr/bin/env bash
# two_channel_head：四臂 × 2 seed + 随机标签对照 + 计时对照 + 老卡 W5（PREREG §3/§4）
# 用 systemd-run --user 启动（绝不 setsid nohup &），日志全部落本目录 logs/
set -uo pipefail
cd /home/vesita/coding/my/DTSeek
HERE=experiments/two_channel_head
LOG=$HERE/logs
mkdir -p "$LOG"

step() { echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "0 空测试自检（冻结实况 / 参数逐项 / 共享证据 / 基线）"
uv run python "$HERE/model.py" --selfcheck > "$LOG/selfcheck.json" 2> "$LOG/selfcheck.err"
grep -E '"encoder_params"|"encoder_trainable"|"head_trainable"|"C.trunk_ptr' "$LOG/selfcheck.json" || true

step "W5 训前老卡基线"
uv run python "$HERE/eval_old_cards.py" --tag before 2>&1 | tee "$LOG/old_cards_before.log"

# ---- 十跑：A/B/C/D × seed 42/43，另加两通道各一的随机标签对照（W3）----
for spec in "A 42" "A 43" "B 42" "B 43" "C 42" "C 43" "D 42" "D 43"; do
  set -- $spec
  arm=$1; seed=$2
  step "run ${arm} seed=${seed}"
  uv run python "$HERE/train.py" --arm "$arm" --seed "$seed" 2>&1 | tee "$LOG/${arm}_s${seed}.log"
done
step "run A randlabel (W3 指针通道)"
uv run python "$HERE/train.py" --arm A --seed 42 --randlabel 2>&1 | tee "$LOG/A_s42_rand.log"
step "run B randlabel (W3 生成通道)"
uv run python "$HERE/train.py" --arm B --seed 42 --randlabel 2>&1 | tee "$LOG/B_s42_rand.log"

step "前向时间对照（轮转交错 + 对照 + 噪声地板）"
uv run python "$HERE/timing_compare.py" 2>&1 | tee "$LOG/timing_compare.log"

step "W5 训后老卡复测"
uv run python "$HERE/eval_old_cards.py" --tag after 2>&1 | tee "$LOG/old_cards_after.log"

step "全部完成"
