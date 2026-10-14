"""Compare v3 vs v4 on the identical 994 held-out hours.
Truth: OLD ±1.5% symmetric labels (apples-to-apples accuracy)
Money: buy->long / sell->short 4h, $1,000 notional, 0.2% round-trip fee."""
import json
import numpy as np
import pandas as pd

truth = pd.read_csv("eval_truth.csv")
fwd4 = truth.fwd4.values
lab = np.where(fwd4 > 1.5, "buy", np.where(fwd4 < -1.5, "sell", "hold"))
NOTIONAL = 1000.0
FEE = 0.002  # round trip

def report(name, preds):
    pa = np.array([p["pred_action"] for p in preds])
    pb = np.array([p["probs"]["buy"] for p in preds])
    ps = np.array([p["probs"]["sell"] for p in preds])
    acc = (pa == lab).mean()
    print(f"\n{'='*64}\n{name}: acc {acc:.4f} (hold-baseline {(lab=='hold').mean():.4f})")
    print(pd.crosstab(pd.Series(lab, name="truth"), pd.Series(pa, name="pred")))
    calls = {}
    for side, mask, sgn in [("buy", pa == "buy", +1), ("sell", pa == "sell", -1)]:
        n = int(mask.sum())
        if n == 0:
            print(f"  {side}: 0 calls")
            calls[side] = dict(n=0, wins=0, gross=0.0, net=0.0)
            continue
        r = sgn * fwd4[mask]                       # trade return %
        wins, losses = r > 0, r <= 0
        gross_w = r[wins].sum() / 100 * NOTIONAL
        gross_l = r[losses].sum() / 100 * NOTIONAL
        net = r.sum() / 100 * NOTIONAL - n * FEE * NOTIONAL
        print(f"  {side}: {n:3d} calls | dir-right {(sgn*fwd4[mask]>0).mean()*100:3.0f}% | "
              f"mean fwd {sgn*fwd4[mask].mean():+.3f}% | win rate {wins.mean()*100:3.0f}%")
        print(f"        profit in winners +${gross_w:8.2f} | loss in losers -${abs(gross_l):7.2f} | "
              f"net after fees ${net:+8.2f}")
        calls[side] = dict(n=n, wins=int(wins.sum()), gross=gross_w, net=net)
    # totals
    mask = (pa == "buy") | (pa == "sell")
    sgn = np.where(pa == "buy", 1, -1)
    n = int(mask.sum())
    if n:
        r = sgn[mask] * fwd4[mask]
        gw = r[r > 0].sum() / 100 * NOTIONAL
        gl = r[r <= 0].sum() / 100 * NOTIONAL
        net = r.sum() / 100 * NOTIONAL - n * FEE * NOTIONAL
        print(f"  TOTAL: {n} trades | wins {int((r>0).sum())} losses {int((r<=0).sum())} "
              f"({(r>0).mean()*100:.0f}% win) | profit +${gw:.2f} loss -${abs(gl):.2f} | "
              f"net after fees ${net:+.2f} ({net/NOTIONAL*100:+.2f}%)")
    return dict(acc=round(float(acc), 4), calls=calls)

v3 = [json.loads(l) for l in open("preds_v3.jsonl") if l.strip()]
report("v3", v3)

try:
    v4 = [json.loads(l) for l in open("preds_v4.jsonl") if l.strip()]
    assert len(v4) == 994, f"v4 preds incomplete: {len(v4)}"
    report("v4", v4)
except FileNotFoundError:
    print("\nv4 preds not ready")
