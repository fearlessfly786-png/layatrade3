"""v1 vs v2 head-to-head on the SAME 994 held-out rows, scored with ±1.5% truth."""
import json
import numpy as np
import pandas as pd

truth = pd.read_csv("eval_truth.csv")
fwd = truth.fwd4.values
lab = np.where(fwd > 1.5, "buy", np.where(fwd < -1.5, "sell", "hold"))

def load_preds(path):
    preds = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame({
        "pred": [p["pred_action"] for p in preds],
        "p_buy": [p["probs"].get("buy") for p in preds],
        "p_hold": [p["probs"].get("hold") for p in preds],
        "p_sell": [p["probs"].get("sell") for p in preds],
    })

v1 = load_preds("preds_partial.jsonl")
v2 = load_preds("preds_v2.jsonl")
assert len(v1) == len(v2) == 994

labels = ["buy", "hold", "sell"]
def metrics(df, name):
    p = df.pred.values
    acc = (p == lab).mean()
    base = max(pd.Series(lab).value_counts(normalize=True))
    cm = {t: {q: 0 for q in labels} for t in labels}
    for t, q in zip(lab, p):
        cm[t][q] += 1
    f1s = {}
    for c in labels:
        tp = cm[c][c]; fp = sum(cm[o][c] for o in labels if o != c); fn = sum(cm[c][o] for o in labels if o != c)
        pr = tp / (tp + fp) if tp + fp else 0; rc = tp / (tp + fn) if tp + fn else 0
        f1s[c] = 2 * pr * rc / (pr + rc) if pr + rc else 0
    macro = sum(f1s.values()) / 3
    print(f"\n--- {name} ---")
    print(f"  accuracy {acc:.4f} | baseline(hold) {base:.4f} | macro-F1 {macro:.4f}")
    print(f"  F1: buy {f1s['buy']:.3f}  hold {f1s['hold']:.3f}  sell {f1s['sell']:.3f}")
    print(f"  calls: buy {int((p=='buy').sum())}, sell {int((p=='sell').sum())}")
    for act in ("buy", "sell"):
        m = p == act
        n = int(m.sum())
        if n:
            print(f"  after {act.upper()} calls ({n}): mean fwd4 {fwd[m].mean():+.3f}%  "
                  f"direction-right {(fwd[m] > 0 if act == 'buy' else fwd[m] < 0).mean()*100:.0f}%")
    # P(true) separation
    pm = np.where(lab == "buy", df.p_buy, np.where(lab == "hold", df.p_hold, df.p_sell))
    ok = p == lab
    print(f"  P(true): correct {pm[ok].mean():.3f} vs wrong {pm[~ok].mean():.3f}")
    return acc, macro

print("=" * 74)
print("HEAD-TO-HEAD on 994 held-out hours (2026-09-01 -> 2026-10-13), truth ±1.5%")
print("=" * 74)
a1 = metrics(v1, "v1 laya-trading-final (trained ±1.0%, Mar-Aug 2026 data)")
a2 = metrics(v2, "v2 laya-trading-v2 (trained ±1.5%, Oct 2025-Aug 2026 data)")
print(f"\nDELTA: accuracy {a2[0]-a1[0]:+.4f}  macro-F1 {a2[1]-a1[1]:+.4f}")

# recent-regime slice (last 230 rows = Oct 2-13, the live-test window)
print()
print("=" * 74)
print("RECENT REGIME SLICE (last 230 hours, Oct 2-13)")
print("=" * 74)
lr = lab[-230:]; fr = fwd[-230:]
for name, df in (("v1", v1), ("v2", v2)):
    p = df.pred.values[-230:]
    acc = (p == lr).mean()
    base = max(pd.Series(lr).value_counts(normalize=True))
    line = f"  {name}: acc {acc:.4f} (base {base:.4f})"
    for act in ("buy", "sell"):
        m = p == act
        if m.sum():
            line += f" | {act} calls {int(m.sum())}: mean fwd4 {fr[m].mean():+.2f}%"
    print(line)

# conviction agreement with realized magnitude (v2 only)
print()
conv_true = np.where(np.abs(fwd) > 3.0, 2, np.where(np.abs(fwd) > 1.5, 1, 0))
v2c = [json.loads(l)["pred_score"] for l in open("preds_v2.jsonl") if l.strip()]
exact = sum(round(s) == t for s, t in zip(v2c, conv_true)) / len(v2c)
print(f"v2 conviction exact-match vs realized magnitude: {exact:.4f}")
