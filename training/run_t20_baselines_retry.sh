#!/bin/bash
# T20 基线重跑：修复 use_amp UnboundLocalError 后重跑失败臂（unet 已完成，跳过）
cd /d/AxiomOS_Project
PY="C:/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe"
echo "=== RETRY (use_amp fix) start $(date '+%F %T') ==="
for m in fno_official dpot fno_physreg; do
  if [ -f checkpoints/t20_$m/final.pt ]; then
    echo "=== $m already done, skip $(date '+%F %T') ==="
    continue
  fi
  echo "=== $m retry start $(date '+%F %T') ==="
  "$PY" -u train_baselines_t20.py --model $m > results/logs/t20_$m.log 2>&1
  echo "=== $m exit=$? $(date '+%F %T') ==="
done
if [ -f checkpoints/t20_transolver/final.pt ]; then
  echo "=== transolver already done, skip $(date '+%F %T') ==="
else
  echo "=== transolver retry start $(date '+%F %T') ==="
  "$PY" -u train_baselines_t20.py --model transolver > results/logs/t20_transolver.log 2>&1
  if [ ! -f checkpoints/t20_transolver/final.pt ]; then
    echo "=== transolver AMP failed, retry fp32 $(date '+%F %T') ==="
    "$PY" -u train_baselines_t20.py --model transolver --no_amp > results/logs/t20_transolver_fp32.log 2>&1
  fi
fi
echo "ALL_BASELINES_DONE $(date '+%F %T')"
if [ -f checkpoints/t20_fno_official/final.pt ]; then
  echo "FNO_RERUN_DONE exit=0 $(date '+%F %T')" >> results/logs/t20_fno_rerun.log
fi
