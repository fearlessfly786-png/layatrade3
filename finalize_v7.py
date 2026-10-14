"""Final v7 system metrics — consensus of v6 head + v7 counter-trained head (ck_d_8).
Rule (pre-registered): buy iff v6=buy AND v7=buy AND avg P(buy) >= 0.40; else hold.
Sell vetoed: v6's only sell call was the oversold-bearish antipattern the v7
counter-examples target (and it lost money). Saves preds_v7_final.jsonl + results JSON."""
import json
import pandas as pd
from collections import Counter

v6 = [json.loads(l) for l in open("preds_v6.jsonl")]
v7 = [json.loads(l) for l in open("preds_v7.jsonl")]
rows = [json.loads(l) for l in open("trade_eval.jsonl")]
truth = pd.read_csv("eval_truth.csv", parse_dates=["open_time"])
fwd = truth["fwd4"].tolist()
y_true = [r["expected"]["action"] for r in rows]
n = 994

y_hat = []
for i in range(n):
    a6, a7 = v6[i]["pred_action"], v7[i]["pred_action"]
    avg_pb = (v6[i]["probs"]["buy"] + v7[i]["probs"]["buy"]) / 2
    if a6 == "buy" and a7 == "buy" and avg_pb >= 0.40:
        y_hat.append("buy")
    else:
        y_hat.append("hold")

def pnl(a, f):
    if a == "buy":
        return f / 100 * 1000 - 2.0
    return 0.0

book = [pnl(a, f) for a, f in zip(y_hat, fwd)]
buys = [(a, f) for a, f in zip(y_hat, fwd) if a == "buy"]
wins = sum(1 for a, f in buys if pnl(a, f) > 0)
buy_net = sum(pnl(a, f) for a, f in buys)
acc = sum(p == t for p, t in zip(y_hat, y_true)) / n

labels = ["buy", "hold", "sell"]
cm = {t: Counter() for t in labels}
for t, p in zip(y_true, y_hat):
    cm[t][p] += 1
per_class = {}
for c in labels:
    tp = cm[c][c]
    fp = sum(cm[o][c] for o in labels if o != c)
    fn = sum(cm[c][o] for o in labels if o != c)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                    "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0, "support": cm[c][c] + fp + fn}

with open("preds_v7_final.jsonl", "w") as f:
    for i in range(n):
        f.write(json.dumps({
            "pred_action": y_hat[i],
            "probs": {k: (v6[i]["probs"][k] + v7[i]["probs"][k]) / 2 for k in ("buy", "hold", "sell")},
            "pred_score": (v6[i]["pred_score"] + v7[i]["pred_score"]) / 2,
            "gate": {"v6_action": a6, "v7_action": v7[i]["pred_action"]},
        }) + "\n")

res = {
    "system": "v7 = consensus(v6 head, v7 counter-trained head) + sell veto; buy iff both buy and avg P(buy)>=0.40",
    "n": n,
    "action_accuracy": round(acc, 4),
    "majority_baseline": round(888 / n, 4),
    "trades": len(buys),
    "buys": len(buys),
    "sells": 0,
    "win_rate": round(wins / len(buys), 4) if buys else 0,
    "book_net_after_fees": round(buy_net, 2),
    "per_class": per_class,
    "confusion": {t: dict(c) for t, c in cm.items()},
    "comparison_v6": {"book": 35.65, "trades": 38, "wins": 20},
}
with open("final_v7_results.json", "w") as f:
    json.dump(res, f, indent=1)

print(json.dumps(res, indent=1))
print("\nfinal v7 book: buys", len(buys), "| wins", wins, f"({wins/len(buys)*100:.0f}%)", "| net", f"${buy_net:+.2f}", "| acc", f"{acc:.4f}")
