#!/usr/bin/env bash
# task-11 方案 A 扫描：2 档（frozen/joint）× 2 臂（cand/nocand）× 2 seed = 8 跑。
# 由 systemd-run --user --unit=dtseek-candemb-sweep 启动（见 PREREG_CAND_EMBED.md）。
# 逐对相邻执行（同 mode 同 seed 的 cand 与 nocand 紧挨着），减少与他人共卡的时间漂移。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1

rc=0
for seed in 42 43; do
  for mode in frozen joint; do
    for arm in on off; do
      name="${mode}_cand${arm}_s${seed}"
      log="logs/cand_embed_${name}.log"
      echo "[sweep] $(date +%T) start ${name} -> ${log}"
      uv run python -u experiments/selection_cards/exp_cand_embed.py \
        --mode "$mode" --cand "$arm" --seed "$seed" > "$log" 2>&1
      r=$?
      echo "[sweep] $(date +%T) done  ${name} rc=${r}"
      if [ "$r" -ne 0 ]; then
        rc=$r
        echo "[sweep] 失败：${name}（见 ${log} 尾部）"
        tail -n 15 "$log"
      fi
    done
  done
done
echo "[sweep] ALL DONE rc=${rc}"
exit "$rc"
