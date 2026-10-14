import json, os, sys, time
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/home/z/hf-cache")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("MALLOC_ARENA_MAX", "2")
data, base, out = sys.argv[1], sys.argv[2], sys.argv[3]
seed = int(sys.argv[4]) if len(sys.argv) > 4 else 200
from laya.train import TrainConfig, finetune
t0 = time.time()
report = finetune(data, base, out, TrainConfig(
    loss="soft-ce", freeze_encoder=True, epochs=1, micro_batch=1, grad_accum=8,
    gradient_checkpointing=False, amp=False, max_len=192, head_max_len=96,
    seed=seed, log_every=100))
print(f"DONE {(time.time()-t0)/60:.1f} min -> {out}", flush=True)
rep = json.load(open(out + "/train_report.json"))
b, a = rep["before"], rep["after"]
print(f"before loss={b['loss']:.4f} acc={b['accuracy']:.4f} | after loss={a['loss']:.4f} acc={a['accuracy']:.4f}")
