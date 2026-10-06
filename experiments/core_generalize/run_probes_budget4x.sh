#!/usr/bin/env bash
# 【探索性，非判据】4× 预算（64 ep × 84 = 5376 步）的探针头：
#   只为回答两件事 ——
#   ① ownership 在 1344 步时全臂 ≈0.92，是「表示本来就够用」还是「预算天花板」？
#   ② idiom 上 C1 落后，是「学得慢」还是「表示被单任务训练训坏」？
# 产物写 probes_budget4x/（**不进** analyze.py 的主表与 P1/P2/P3）。
set -u
cd /home/vesita/coding/my/DTSeek
export PYTHONUNBUFFERED=1 HSA_OVERRIDE_GFX_VERSION=10.3.0
E=experiments/core_generalize
mkdir -p "$E/logs" "$E/probes_budget4x"
rc_total=0

run() {
  local name="$1"; shift
  local log="$E/logs/${name}.log"
  if [ -f "$E/probes_budget4x/${name}.json" ]; then echo "[cg] skip ${name}"; return 0; fi
  echo "[cg] $(date +%T) start ${name}"
  /usr/bin/uv run python -u "$@" > "$log" 2>&1
  local r=$?
  echo "[cg] $(date +%T) done  ${name} rc=${r}"
  grep -E "CG_PROBE_METRICS|Traceback|SystemExit" "$log" | tail -3
  [ "$r" -ne 0 ] && rc_total=1
  return 0
}

for h in idiom ownership; do
  run "B0_${h}_s42_e64" "$E/train_probe_head.py" --core checkpoints/base_encoder.pt \
      --label B0 --holdout "$h" --seed 42 --epochs 64 \
      --out "$E/probes_budget4x/B0_${h}_s42_e64.json"
  for arm in C1 C5; do
    run "${arm}_s42_${h}_s42_e64" "$E/train_probe_head.py" --core "$E/cores/${arm}_s42.pt" \
        --label "${arm}_s42" --holdout "$h" --seed 42 --epochs 64 \
        --out "$E/probes_budget4x/${arm}_s42_${h}_s42_e64.json"
  done
done
echo "CG_BUDGET4X_ALL_DONE rc=$rc_total $(date +%T)"
exit $rc_total
