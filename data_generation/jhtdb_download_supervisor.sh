#!/bin/bash
# JHTDB 下载监督循环：硬失败（如 InvalidChunkLength 断流）后自动重启，
# 逐帧 npy 缓存保证续传不重复下载。最多重启 60 次，每次间隔 30s。
cd /d/AxiomOS_Project || exit 1
PY="C:/Users/ASUS/AppData/Local/Programs/Python/Python313/python.exe"
for i in $(seq 1 60); do
    echo "=== supervisor attempt $i $(date '+%F %T') ==="
    "$PY" scripts/data_generation/jhtdb_download_fix2.py && break
    echo "=== attempt $i failed, sleep 30s ==="
    sleep 30
done
echo "=== supervisor exit $(date '+%F %T') ==="
