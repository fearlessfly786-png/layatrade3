"""Split the balanced training data into small chunks for subprocess-chained
fine-tuning (each chunk trains in its own short-lived process, avoiding the
sandbox's ~12 min process kill window), plus a balanced eval sample."""
import json
import random
from collections import defaultdict

rows = [json.loads(l) for l in open("trade_train.jsonl")]
by_label = defaultdict(list)
for r in rows:
    by_label[r["expected"]["action"]].append(r)

rng = random.Random(7)
for v in by_label.values():
    rng.shuffle(v)

# 4 chunks x 400 rows, stratified
CHUNKS = 4
PER = 400
pools = {k: list(v) for k, v in by_label.items()}
chunks = [[] for _ in range(CHUNKS)]
ci = 0
while any(pools.values()):
    for label in ("buy", "sell", "hold"):
        if pools[label]:
            chunks[ci % CHUNKS].append(pools[label].pop())
    ci += 1

out_chunks = []
for i, ch in enumerate(chunks, 1):
    sel = ch[:PER]
    rng.shuffle(sel)
    with open(f"chunk{i}.jsonl", "w") as f:
        for r in sel:
            f.write(json.dumps(r) + "\n")
    dist = defaultdict(int)
    for r in sel:
        dist[r["expected"]["action"]] += 1
    out_chunks.append(f"chunk{i}.jsonl: {len(sel)} rows {dict(dist)}")

# Balanced eval sample: 50 per class from the REAL-distribution eval file
ev = [json.loads(l) for l in open("trade_eval.jsonl")]
ev_by = defaultdict(list)
for r in ev:
    ev_by[r["expected"]["action"]].append(r)
eval_sample = []
for label, n in (("buy", 50), ("sell", 50), ("hold", 50)):
    eval_sample += ev_by[label][:n]
rng.shuffle(eval_sample)
with open("eval_balanced.jsonl", "w") as f:
    for r in eval_sample:
        f.write(json.dumps(r) + "\n")

print("\n".join(out_chunks))
print(f"eval_balanced.jsonl: {len(eval_sample)} rows")
