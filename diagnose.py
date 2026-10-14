"""Diagnose v1 model weak points using its 994 eval predictions + ACTUAL fwd returns."""
import json
import re
import numpy as np
import pandas as pd

preds = [json.loads(l) for l in open("preds_partial.jsonl") if l.strip()]
rows = [json.loads(l) for l in open("trade_eval.jsonl") if l.strip()]
truth = pd.read_csv("eval_truth.csv")
assert len(preds) == len(rows) == len(truth) == 994

df = pd.DataFrame({
    "pred": [p["pred_action"] for p in preds],
    "p_buy": [p["probs"].get("buy") for p in preds],
    "p_hold": [p["probs"].get("hold") for p in preds],
    "p_sell": [p["probs"].get("sell") for p in preds],
    "fwd4": truth.fwd4.values,
    "old_index": truth.old_index.values,
    "state": [r["state"] for r in rows],
})

# parse features from state text
def grab(pat, s):
    m = re.search(pat, s)
    return float(m.group(1)) if m else np.nan

df["rsi"] = [grab(r"RSI\(14\) ([\d.]+)", s) for s in df.state]
df["m1h"] = [grab(r"1h ([+-][\d.]+)%", s) for s in df.state]
df["m4h"] = [grab(r"4h ([+-][\d.]+)%", s) for s in df.state]
df["m24h"] = [grab(r"24h ([+-][\d.]+)%", s) for s in df.state]
df["atr"] = [grab(r"ATR\(14\) ([\d.]+)%", s) for s in df.state]
df["bear_ema"] = ["below both EMAs" in s for s in df.state]
df["bull_ema"] = ["above both EMAs" in s for s in df.state]

print("=" * 78)
print("WEAK POINT 1 — accuracy under different label thresholds (same predictions)")
print("=" * 78)
for up, dn in ((1.0, -1.0), (1.5, -1.5), (2.0, -2.0)):
    lab = np.where(df.fwd4 > up, "buy", np.where(df.fwd4 < dn, "sell", "hold"))
    acc = (df.pred.values == lab).mean()
    base = max(pd.Series(lab).value_counts(normalize=True))
    print(f"  ±{up:.1f}%: accuracy {acc:.4f} | always-hold baseline {base:.4f} | "
          f"trades in truth: {(lab!='hold').sum()}/994")

print()
print("=" * 78)
print("WEAK POINT 2 — what ACTUALLY happens after each predicted call")
print("=" * 78)
for act in ("buy", "sell"):
    sub = df[df.pred == act]
    if len(sub) == 0:
        continue
    print(f"  predicted {act.upper()} ({len(sub)} calls):")
    print(f"    mean fwd4 = {sub.fwd4.mean():+.3f}%   median = {sub.fwd4.median():+.3f}%")
    print(f"    fwd4 > 0: {(sub.fwd4>0).mean()*100:.0f}%   fwd4 < 0: {(sub.fwd4<0).mean()*100:.0f}%")
    print(f"    fwd4 > +1.5%: {(sub.fwd4>1.5).mean()*100:.0f}%   fwd4 < -1.5%: {(sub.fwd4<-1.5).mean()*100:.0f}%")

hold = df[df.pred == "hold"]
print(f"  predicted HOLD ({len(hold)}): mean fwd4 = {hold.fwd4.mean():+.3f}%  "
      f"|fwd4|>1.5%: {(hold.fwd4.abs()>1.5).mean()*100:.0f}% (missed moves)")

print()
print("=" * 78)
print("WEAK POINT 3 — market state when model calls SELL (right vs wrong)")
print("=" * 78)
sells = df[df.pred == "sell"]
good = sells[sells.fwd4 < -1.0]   # sell was right
bad = sells[sells.fwd4 >= -1.0]   # sell was wrong
for name, sub in (("sell & correct (fwd<-1%)", good), ("sell & wrong (fwd>=-1%)", bad)):
    print(f"  {name} (n={len(sub)}):")
    print(f"    RSI {sub.rsi.mean():.1f} | 24h mom {sub.m24h.mean():+.2f}% | ATR {sub.atr.mean():.2f}% | "
          f"bear-EMA {sub.bear_ema.mean()*100:.0f}% | P(sell) mean {sub.p_sell.mean():.3f}")

print()
print("=" * 78)
print("WEAK POINT 4 — missed real moves (truth buy/sell, model said hold)")
print("=" * 78)
lab10 = np.where(df.fwd4 > 1.0, "buy", np.where(df.fwd4 < -1.0, "sell", "hold"))
missed = df[(df.pred == "hold") & (lab10 != "hold")]
print(f"  missed moves: {len(missed)}")
print(f"    their RSI {missed.rsi.mean():.1f} | 24h mom {missed.m24h.mean():+.2f}% | fwd4 mean {missed.fwd4.mean():+.2f}%")
miss_buy = missed[missed.fwd4 > 1.0]; miss_sell = missed[missed.fwd4 < -1.0]
print(f"    missed buys: {len(miss_buy)} (RSI {miss_buy.rsi.mean():.1f}, 24h {miss_buy.m24h.mean():+.2f}%)")
print(f"    missed sells: {len(miss_sell)} (RSI {miss_sell.rsi.mean():.1f}, 24h {miss_sell.m24h.mean():+.2f}%)")

print()
print("=" * 78)
print("WEAK POINT 5 — probability separation (does the model discriminate at all?)")
print("=" * 78)
print("  mean P(true class) by whether prediction was correct:")
lab15 = np.where(df.fwd4 > 1.5, "buy", np.where(df.fwd4 < -1.5, "sell", "hold"))
df["lab15"] = lab15
df["ptrue15"] = [p.get(t, 0) for p, t in zip(
    (dict(buy=b, hold=h, sell=s) for b, h, s in zip(df.p_buy, df.p_hold, df.p_sell)), df.lab15)]
for ok in (True, False):
    sub = df[(df.pred == df.lab15) == ok]
    print(f"    {'correct' if ok else 'wrong  '} preds (n={len(sub)}): mean P(true)={sub.ptrue15.mean():.3f}")

# RSI bucket -> actual mean fwd4 (mean reversion structure)
print()
print("=" * 78)
print("MARKET STRUCTURE — mean fwd4 by RSI bucket (is there a learnable signal?)")
print("=" * 78)
df["rsi_bucket"] = pd.cut(df.rsi, [0, 30, 40, 55, 70, 100], labels=["<30", "30-40", "40-55", "55-70", ">70"])
print(df.groupby("rsi_bucket", observed=True).agg(
    n=("fwd4", "size"), mean_fwd4=("fwd4", "mean"), std_fwd4=("fwd4", "std")).round(3))
print()
df["m24_bucket"] = pd.cut(df.m24h, [-100, -3, -1, 1, 3, 100], labels=["<-3%", "-3..-1", "-1..+1", "+1..+3", ">+3%"])
print(df.groupby("m24_bucket", observed=True).agg(
    n=("fwd4", "size"), mean_fwd4=("fwd4", "mean")).round(3))
