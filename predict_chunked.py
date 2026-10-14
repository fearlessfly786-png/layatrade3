"""Resumable evaluation: appends one prediction per line to preds_partial.jsonl.
Safe against timeouts — re-run and it continues where it left off."""
import json
import os
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/home/z/hf-cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import laya

CKPT = sys.argv[4] if len(sys.argv) > 4 else "./laya-trading-final"
SRC = sys.argv[1] if len(sys.argv) > 1 else "trade_eval.jsonl"
OUT = sys.argv[2] if len(sys.argv) > 2 else "preds_partial.jsonl"
BUDGET_S = float(sys.argv[3]) if len(sys.argv) > 3 else 500.0  # stop before timeout

_qpath = os.path.join(CKPT, "questions.json")
if not os.path.exists(_qpath):
    _qpath = "./laya-trading-final/questions.json"
QUESTIONS = json.load(open(_qpath))

rows = [json.loads(l) for l in open(SRC) if l.strip()]
done = 0
if os.path.exists(OUT):
    with open(OUT) as f:
        done = sum(1 for _ in f if _.strip())
print(f"{SRC}: {len(rows)} rows, {done} already predicted", flush=True)
if done >= len(rows):
    print("ALL DONE")
    sys.exit(0)

print("Loading checkpoint...", flush=True)
t0 = time.time()
agent = laya.load(CKPT)
print(f"loaded in {time.time()-t0:.1f}s", flush=True)

t_start = time.time()
CH = 50
with open(OUT, "a") as fout:
    i = done
    while i < len(rows):
        chunk = rows[i:i + CH]
        results = agent.predict_batch([r["state"] for r in chunk], QUESTIONS, batch_size=8)
        for res in results:
            a = res["answers"]["action"]
            c = res["answers"]["conviction"]
            fout.write(json.dumps({
                "pred_action": a.get("choice"),
                "probs": a.get("probabilities", {}),
                "pred_score": c.get("score"),
                "conv_probs": c.get("probabilities", {}),
            }) + "\n")
        fout.flush()
        i += len(chunk)
        el = time.time() - t_start
        print(f"  {i}/{len(rows)} ({el:.0f}s elapsed)", flush=True)
        if el > BUDGET_S and i < len(rows):
            print(f"STOPPING at {i} — rerun to continue", flush=True)
            sys.exit(0)
print("ALL DONE")
