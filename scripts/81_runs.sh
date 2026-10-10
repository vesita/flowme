#!/usr/bin/env bash
# L3-r5 / r5′（L3_PLAN §14 / §14.2）驱动
#   formal   : r5 正式 4 run（B0a/B0b × seed{1234,5678}，均值注入，L1）—— 已跑完（15:49）
#   smoke    : r5 两臂 50 步探路
#   r5p      : r5′ 正式 8 run（B0pa/B0pb × seed{1234,5678} × layers{2,1}）—— **判据档 L2 先跑**
#   r5p-smoke: r5′ 两臂 50 步探路（L2）
#   ★日志**按臂分文件**（同名 `>` 会覆盖）。
#   ★所有带变量的展开都在**本脚本**里发生：systemd-run 只调这份脚本，绝不把循环内联进 systemd-run
#     （systemd 会在 bash 之前展开 $VAR，实测把 logs/80_${arm}_*.log 变成 logs/80__*.log）。
#   ★每 run 都带 --cf（shuffle 用 §11.17 动态金标）。
#   ★起卡前先查在跑的 GPU 单元/进程；撞上则**后起者让位**（停自己，绝不 kill 别人的进程）。
set -u
cd /home/vesita/coding/my/flowme
PY=/home/vesita/coding/my/flowme/.venv/bin/python
MODE="${1:-formal}"

run_one () {                       # $1=arm $2=seed $3=logfile $4=extra 参数（可空）
  local arm=$1 seed=$2 log=$3 extra=$4
  echo "=== START arm=${arm} seed=${seed} $(date -Is) ==="
  # shellcheck disable=SC2086
  $PY stages/81_realign_card_ablation.py --arm "$arm" --seed "$seed" --k 8 --data-prefix 76 --cf $extra \
    > "$log" 2>&1
  echo "=== END arm=${arm} seed=${seed} rc=$? $(date -Is) ==="
}

run_one2 () {                      # $1=arm $2=seed $3=realign-layers $4=extra 参数（可空）
  local arm=$1 seed=$2 nl=$3 extra=$4
  local log="logs/81_${arm}_s${seed}_L${nl}.log"
  echo "=== START arm=${arm} seed=${seed} layers=${nl} $(date -Is) ==="
  # shellcheck disable=SC2086
  $PY stages/81_realign_card_ablation.py --arm "$arm" --seed "$seed" --k 8 --data-prefix 76 --cf \
    --realign-layers "$nl" $extra > "$log" 2>&1
  echo "=== END arm=${arm} seed=${seed} layers=${nl} rc=$? $(date -Is) ==="
}

if [ "$MODE" = "smoke" ]; then
  run_one B0a 1234 logs/81_smoke_B0a.log "--smoke"
  run_one B0b 1234 logs/81_smoke_B0b.log "--smoke"
  echo "=== SMOKE-DONE $(date -Is) ==="
  exit 0
fi

if [ "$MODE" = "r5p-smoke" ]; then
  run_one2 B0pa 1234 2 "--smoke"
  run_one2 B0pb 1234 2 "--smoke"
  echo "=== R5P-SMOKE-DONE $(date -Is) ==="
  exit 0
fi

if [ "$MODE" = "r5p" ]; then
  # §14.2：**判据档 = L2（参数对齐档）**，先跑；L1 作旁证并量「少一层」影响
  run_one2 B0pa 1234 2 "--pair-hits logs/79_hits_A_s1234.json"
  run_one2 B0pb 1234 2 "--pair-hits logs/81_hits_B0pa_s1234_L2.json,logs/79_hits_B2_s1234.json"
  run_one2 B0pa 5678 2 "--pair-hits logs/79_hits_A_s5678.json"
  run_one2 B0pb 5678 2 "--pair-hits logs/81_hits_B0pa_s5678_L2.json,logs/79_hits_B2_s5678.json"
  run_one2 B0pa 1234 1 "--pair-hits logs/79_hits_A_s1234.json"
  run_one2 B0pb 1234 1 "--pair-hits logs/81_hits_B0pa_s1234.json,logs/79_hits_B2_s1234.json"
  run_one2 B0pa 5678 1 "--pair-hits logs/79_hits_A_s5678.json"
  run_one2 B0pb 5678 1 "--pair-hits logs/81_hits_B0pa_s5678.json,logs/79_hits_B2_s5678.json"
  $PY stages/81_realign_card_ablation.py --report-r5p --realign-layers 2 > logs/81_report_r5p.log 2>&1
  echo "=== R5P-DONE $(date -Is) ==="
  exit 0
fi

run_one B0a 1234 logs/81_B0a_s1234.log "--pair-hits logs/79_hits_A_s1234.json"
run_one B0b 1234 logs/81_B0b_s1234.log "--pair-hits logs/81_hits_B0a_s1234.json,logs/79_hits_B2_s1234.json"
run_one B0a 5678 logs/81_B0a_s5678.log "--pair-hits logs/79_hits_A_s5678.json"
run_one B0b 5678 logs/81_B0b_s5678.log "--pair-hits logs/81_hits_B0a_s5678.json,logs/79_hits_B2_s5678.json"
$PY stages/81_realign_card_ablation.py --report-r5 > logs/81_report_r5.log 2>&1
echo "=== ALL-DONE $(date -Is) ==="

