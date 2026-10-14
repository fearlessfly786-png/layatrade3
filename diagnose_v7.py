"""Deep diagnosis of v6 on the frozen 994-hour held-out set.
Joins: trade_eval.jsonl (states + truth) x preds_v7.jsonl x eval_truth.csv (fwd4 for P&L).
P&L: each call = 4h position, $1,000 notional, 0.2% round-trip fee ($2).
  buy  -> net = fwd4/100*1000 - 2
  sell -> net = -fwd4/100*1000 - 2
"""
import json
from collections import Counter
import pandas as pd

rows = [json.loads(l) for l in open("trade_eval.jsonl") if l.strip()]
preds = [json.loads(l) for l in open("preds_v7.jsonl") if l.strip()]
truth = pd.read_csv("eval_truth.csv", parse_dates=["open_time"])
assert len(rows) == len(preds) == len(truth) == 994

y_true = [r["expected"]["action"] for r in rows]
y_pred = [p["pred_action"] for p in preds]
fwd = truth["fwd4"].tolist()
n = len(y_true)

labels = ["buy", "hold", "sell"]
acc = sum(t == p for t, p in zip(y_true, y_pred)) / n
cm = {t: Counter() for t in labels}
for t, p in zip(y_true, y_pred):
    cm[t][p] += 1
print("=" * 64)
print("v7 DIAGNOSIS on 994 held-out hours")
print("=" * 64)
print(f"accuracy {acc:.4f}   baseline(hold) {888/994:.4f}")
for t in labels:
    print(f"  true {t:<5}: {dict(cm[t])}")

def pnl(action, f):
    if action == "buy":
        return f / 100 * 1000 - 2.0
    if action == "sell":
        return -f / 100 * 1000 - 2.0
    return 0.0

# ---- trade book ----
trades = []
for i, (a, f) in enumerate(zip(y_pred, fwd)):
    if a in ("buy", "sell"):
        trades.append({"i": i, "action": a, "fwd": f, "net": pnl(a, f),
                       "pbuy": preds[i]["probs"].get("buy", 0.0),
                       "psell": preds[i]["probs"].get("sell", 0.0),
                       "phold": preds[i]["probs"].get("hold", 0.0)})
buys = [t for t in trades if t["action"] == "buy"]
sells = [t for t in trades if t["action"] == "sell"]
buy_net = sum(t["net"] for t in buys)
sell_net = sum(t["net"] for t in sells)
print(f"\nBOOK: {len(buys)} buys ${buy_net:+.2f} | {len(sells)} sells ${sell_net:+.2f} | TOTAL ${buy_net+sell_net:+.2f}")
if buys:
    wr = sum(t["net"] > 0 for t in buys) / len(buys)
    print(f"  buys: {wr*100:.0f}% net-positive, mean fwd {sum(t['fwd'] for t in buys)/len(buys):+.2f}%")
if sells:
    wr = sum(t["net"] > 0 for t in sells) / len(sells)
    print(f"  sells: {wr*100:.0f}% net-positive, mean fwd {sum(t['fwd'] for t in sells)/len(sells):+.2f}%")

# ---- buy recall autopsy ----
print("\n--- BUY RECALL AUTOPSY ---")
true_buys = [(i, fwd[i], preds[i]["probs"].get("buy", 0.0), preds[i]["probs"].get("hold", 0.0)) for i in range(n) if y_true[i] == "buy"]
print(f"true buys: {len(true_buys)}")
called = [x for x in true_buys if y_pred[x[0]] == "buy"]
missed = [x for x in true_buys if y_pred[x[0]] != "buy"]
print(f"caught {len(called)}, missed {len(missed)}")
if missed:
    print(f"missed buys: mean fwd {sum(x[1] for x in missed)/len(missed):+.2f}%  "
          f"missed $ = {sum(pnl('buy', x[1]) for x in missed):+.2f}")
    print(f"missed P(buy): mean {sum(x[2] for x in missed)/len(missed):.3f}  "
          f"min {min(x[2] for x in missed):.3f}  max {max(x[2] for x in missed):.3f}")
    print(f"missed P(hold): mean {sum(x[3] for x in missed)/len(missed):.3f}")
# how many missed buys were VERY close to the buy boundary (P(buy) >= P(hold)-0.05)?
close = [x for x in missed if x[2] >= x[3] - 0.05]
print(f"missed buys within 0.05 of hold prob: {len(close)}")
# all true-buy P(buy) stats
print(f"all true buys P(buy): mean {sum(x[2] for x in true_buys)/len(true_buys):.3f}")

# ---- false buys ----
false_buys = [(i, fwd[i], preds[i]["probs"].get("buy", 0.0)) for i in range(n) if y_pred[i] == "buy" and y_true[i] != "buy"]
print(f"\nfalse buys: {len(false_buys)}  cost ${sum(pnl('buy', x[1]) for x in false_buys):+.2f}  "
      f"mean fwd {sum(x[1] for x in false_buys)/max(len(false_buys),1):+.2f}%")

# ---- threshold scan (post-hoc oracle, for v7 design) ----
print("\n--- THRESHOLD SCAN: call buy when P(buy) >= T (else keep model call) ---")
print(f"{'T':<6}{'nBuy':>6}{'buyNet':>10}{'totNet':>10}{'acc':>8}")
for T in [0.30, 0.32, 0.34, 0.36, 0.38, 0.40]:
    tot = 0.0
    nb = 0
    corr = 0
    for i in range(n):
        a = y_pred[i]
        pb = preds[i]["probs"].get("buy", 0.0)
        if a != "buy" and pb >= T:
            a = "buy"
        if a == "buy":
            nb += 1
            tot += pnl("buy", fwd[i])
        elif a == "sell":
            tot += pnl("sell", fwd[i])
        corr += (a == y_true[i])
    print(f"{T:<6.2f}{nb:>6}{tot - (tot - 0):>10.2f}{tot:>10.2f}{corr/n:>8.4f}")

# conviction signal check
big = [p["pred_score"] for p, t in zip(preds, y_true) if t != "hold"]
quiet = [p["pred_score"] for p, t in zip(preds, y_true) if t == "hold"]
print(f"\nconviction: mean {sum(big)/len(big):.3f} on active vs {sum(quiet)/len(quiet):.3f} on quiet")
