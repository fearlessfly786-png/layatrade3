#!/bin/bash
# Chained v4 training: 5 chunks from laya-trading-v3, disk-safe.
set -e
cd /home/z/my-project/laya-model
BASE=laya-trading-v3
for n in 1 2 3 4 5; do
  OUT=ckpt_v4_$n
  echo "=== CHUNK $n  base=$BASE  $(date -u +%H:%M:%S) ==="
  python train_chunk_v2.py chunk_v4_$n.jsonl $BASE $OUT 40$n
  if [ -f "$OUT/model.safetensors" ]; then
    if [ "$BASE" != "laya-trading-v3" ]; then rm -rf "$BASE"; fi
    BASE=$OUT
  else
    echo "FATAL: $OUT missing model.safetensors"; exit 1
  fi
done
rm -rf ckpt_v4_4
mv ckpt_v4_5 laya-trading-v4
echo "ALL_DONE $(date -u +%H:%M:%S) -> laya-trading-v4"
