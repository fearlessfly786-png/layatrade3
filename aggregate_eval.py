"""Aggregate held-out metrics from preds_partial.jsonl (994 rows, real market distribution)."""
import json
from collections import Counter

rows = [json.loads(l) for l in open("trade_eval.jsonl") if l.strip()]
preds = [json.loads(l) for l in open("preds_partial.jsonl") if l.strip()]
assert len(rows) == len(preds), f"{len(rows)} rows vs {len(preds)} preds"

labels = ["buy", "hold", "sell"]
y_true = [r["expected"]["action"] for r in rows]
y_pred = [p["pred_action"] for p in preds]
n = len(y_true)

dist = Counter(y_true)
acc = sum(t == p for t, p in zip(y_true, y_pred)) / n
majority = max(dist.values()) / n

cm = {t: {p: 0 for p in labels} for t in labels}
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1

per_class = {}
for c in labels:
    tp = cm[c][c]
    fp = sum(cm[o][c] for o in labels if o != c)
    fn = sum(cm[c][o] for o in labels if o != c)
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": round(f1, 4), "support": dist[c]}
macro_f1 = sum(v["f1"] for v in per_class.values()) / 3

# conviction metrics
exact = within1 = 0
mae_sum = 0.0
nv = 0
for r, p in zip(rows, preds):
    s = p["pred_score"]
    if s is None:
        continue
    s = float(s)
    exact += (round(s) == r["expected"]["conviction"])
    within1 += (abs(round(s) - r["expected"]["conviction"]) <= 1)
    mae_sum += abs(s - r["expected"]["conviction"])
    nv += 1

# mean probability mass on true class
mass = [p["probs"].get(r["expected"]["action"]) for r, p in zip(rows, preds)]
mass = [float(m) for m in mass if m is not None]

# sanity: distribution of predictions
pred_dist = Counter(y_pred)

print("=" * 62)
print("HELD-OUT EVALUATION — trade_eval.jsonl (994 rows, real market distribution)")
print("=" * 62)
print(f"true label distribution : {dict(dist)}")
print(f"prediction distribution : {dict(pred_dist)}")
print(f"majority-class baseline : {majority:.4f}  (always predict '{max(dist, key=dist.get)}')")
print(f"random baseline (3-way) : 0.3333")
print(f"MODEL action accuracy   : {acc:.4f}")
print(f"MODEL macro-F1          : {macro_f1:.4f}")
print()
print(f"{'class':<6} {'precision':>9} {'recall':>8} {'f1':>7} {'support':>8}")
for c in labels:
    v = per_class[c]
    print(f"{c:<6} {v['precision']:>9.4f} {v['recall']:>8.4f} {v['f1']:>7.4f} {v['support']:>8}")
print()
print("confusion matrix (rows=true, cols=pred):")
print(f"{'':<8} {'buy':>7} {'hold':>7} {'sell':>7}")
for t in labels:
    print(f"{t:<8} {cm[t]['buy']:>7} {cm[t]['hold']:>7} {cm[t]['sell']:>7}")
print()
print(f"conviction (0/1/2): exact={exact/nv:.4f}  within-1={within1/nv:.4f}  MAE={mae_sum/nv:.4f}  (n={nv})")
print(f"mean P(true action class) = {sum(mass)/len(mass):.4f}")

out = {
    "dataset": "trade_eval.jsonl", "n": n, "label_distribution": dict(dist),
    "majority_baseline": round(majority, 4), "action_accuracy": round(acc, 4),
    "macro_f1": round(macro_f1, 4), "per_class": per_class, "confusion": cm,
    "conviction": {"exact_match": round(exact / nv, 4), "within_1": round(within1 / nv, 4),
                   "mae": round(mae_sum / nv, 4), "n": nv},
    "mean_prob_true_class": round(sum(mass) / len(mass), 4),
    "prediction_distribution": dict(pred_dist),
}
with open("final_eval_results.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nsaved -> final_eval_results.json")
