"""Train ONE chunk in a short-lived process, chained onto the previous chunk's
checkpoint. Usage: python3 train_chunk.py <chunk_no 1|2|3> <base_ckpt_or_builtin>

Designed to finish in ~10 min (inside the sandbox kill window):
  - 400 items x 1 epoch, micro_batch 1 (no gradient checkpointing = ~2x faster)
  - max_len 192 (market states are ~160 tokens), head_max_len 96
  - freeze_encoder=True (head-only training), loss=soft-ce
"""
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/home/z/hf-cache")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")

chunk_no = sys.argv[1]
base = sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else f"./ckpt_chunk{chunk_no}"

from laya.train import TrainConfig, finetune

t0 = time.time()
report = finetune(
    f"chunk{chunk_no}.jsonl",
    base,
    out,
    TrainConfig(
        loss="soft-ce",
        freeze_encoder=True,
        epochs=1,
        micro_batch=1,
        grad_accum=16,
        gradient_checkpointing=False,
        amp=False,
        max_len=192,
        head_max_len=96,
        seed=100 + int(chunk_no),
        log_every=100,
    ),
)
print(f"CHUNK {chunk_no} DONE in {(time.time()-t0)/60:.1f} min -> {out}", flush=True)
print("loss:", report.get("epoch_loss"), "temp:", report.get("temperature"))
