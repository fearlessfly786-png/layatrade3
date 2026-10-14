"""Memory-safe Laya fine-tuning on trading decisions (4.1GB RAM CPU sandbox).

Settings chosen for the constraint (OOM-killed twice before these):
  - base=multilingual      -> 322M mmBERT-base (1.29GB fp32) instead of 421M (1.68GB)
  - freeze_encoder=True    -> only the typed decision head trains (CPU-appropriate)
  - micro_batch=1          -> minimal activation footprint
  - grad_accum=16          -> keeps effective batch 16
  - gradient_checkpointing -> trade compute for memory
  - max_len=256            -> market states are ~200 tokens; head_max_len=96
  - loss=soft-ce           -> matches head-only training best (docs #741)
"""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/home/z/hf-cache")
# Memory-creep fixes: glibc arena fragmentation + thread-local buffers are what
# killed the previous runs after ~50 steps. Single-thread everything.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("MALLOC_ARENA_MAX", "1")

import time

from laya.train import TrainConfig, finetune

t0 = time.time()
report = finetune(
    "trade_train.jsonl",
    "multilingual",
    "./laya-trading-checkpoint",
    TrainConfig(
        loss="soft-ce",
        freeze_encoder=True,
        epochs=2,
        micro_batch=1,
        grad_accum=16,
        gradient_checkpointing=True,
        amp=False,
        max_len=256,
        head_max_len=96,
        seed=42,
        log_every=50,
    ),
)

dt = time.time() - t0
print(f"\n=== TRAINING DONE in {dt/60:.1f} min ===")
print("train_items :", report.get("train_items"))
print("calibration :", report.get("calibration_items"))
print("skipped     :", report.get("skipped"))
print("epoch_loss  :", report.get("epoch_loss"))
print("temperature :", report.get("temperature"))
cal = report.get("calibration") or {}
print("calibration report:", cal)
