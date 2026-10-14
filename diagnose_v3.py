"""Deep weakness diagnosis of v3 on the 994 held-out hours.
Focus: WHY sells lose money, what label/feature change could fix them,
missed buys, and probability-gating options."""
import json
import numpy as np
import pandas as pd

pd.set_option("display.width", 200)

df = pd.read_csv("candles_9000.csv", parse_dates=["open_time"])
c = df["close"]
df["ret_1h"] = c.pct_change() * 100
df["ret_4h"] = c.pct_change(4) * 100
df["ret_24h"] = c.pct_change(24) * 100
df["ema12"] = c.ewm(span=12, adjust=False).mean()
df["ema26"] = c.ewm(span=26, adjust=False).mean()
df["macd"] = df["ema12"] - df["ema26"]
df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
df["macd_hist"] = df["macd"] - df["macd_sig"]
delta = c.diff()
gain = delta.clip(lower=0).rolling(14).mean()
loss = (-delta.clip(upper=0)).rolling(14).mean()
rs = gain / loss.replace(0, np.nan)
df["rsi"] = 100 - 100 / (1 + rs)
df["vol_ma20"] = df["volume"].rolling(20).mean()
df["vol_ratio"] = df["volume"] / df["vol_ma20"]
df["atr14"] = (df["high"] - df["low"]).rolling(14).mean() / c * 100
df["hi24"] = df["high"].rolling(24).max()
df["lo24"] = df["low"].rolling(24).min()
df["dist_hi"] = (c / df["hi24"] - 1) * 100
df["dist_lo"] = (c / df["lo24"] - 1) * 100
fwd = (c.shift(-4) / c - 1) * 100
df["fwd4"] = fwd

truth = pd.read_csv("eval_truth.csv", parse_dates=["open_time"])
preds = [json.loads(l) for l in open("preds_v3.jsonl") if l.strip()]
assert len(truth) == len(preds) == 994

# join truth rows to features by open_time
m = df.set_index("open_time")
feat = m.loc[truth.open_time].reset_index()
fwd4 = truth.fwd4.values
pa = np.array([p["pred_action"] for p in preds])
pb = np.array([p["probs"]["buy"] for p in preds])
ph = np.array([p["probs"]["hold"] for p in preds])
ps = np.array([p["probs"]["sell"] for p in preds])

lab = np.where(fwd4 > 1.5, "buy", np.where(fwd4 < -1.5, "sell", "hold"))
conf = (pa == lab).mean()
print(f"=== v3 on 994 eval rows: acc {conf:.4f} (hold-baseline {(lab=='hold').mean():.4f})")
cm = pd.crosstab(pd.Series(lab, name="truth"), pd.Series(pa, name="pred"))
print(cm)

FEATS = ["rsi", "ret_1h", "ret_4h", "ret_24h", "atr14", "vol_ratio", "dist_hi", "dist_lo"]

def align_flags(idx):
    """bullish: close>ema12>ema26 ; bearish: close<ema12<ema26"""
    r = feat.loc[idx]
    bull = (r.close > r.ema12) & (r.ema12 > r.ema26)
    bear = (r.close < r.ema12) & (r.ema12 < r.ema26)
    return np.where(bull, "bull", np.where(bear, "bear", "mixed"))

print("\n=== SELL CALL AUTOPSY (v3 called sell) ===")
si = np.where(pa == "sell")[0]
al = align_flags(si)
ok = fwd4[si] < 0  # short direction right
d = pd.DataFrame({
    "fwd4": fwd4[si], "dir_right": ok, "align": al,
    "P(sell)": ps[si], "rsi": feat.rsi.values[si], "ret4h": feat.ret_4h.values[si],
    "ret24h": feat.ret_24h.values[si], "atr": feat.atr14.values[si],
    "volr": feat.vol_ratio.values[si],
})
print(d.round(2).to_string())
print(f"\nsells: {len(si)}, dir-right {ok.sum()} ({ok.mean()*100:.0f}%), "
      f"mean fwd4 {fwd4[si].mean():+.3f}%  | by align: "
      + ", ".join(f"{a}: {int((al==a).sum())} calls, {int(((al==a)&ok).sum())} right" for a in pd.unique(al)))

print("\n=== TRUE SELLS IN EVAL (fwd4<=-1.5), what do they look like? ===")
ti = np.where(lab == "sell")[0]
tal = align_flags(ti)
trec = feat.ret_4h.values[ti] < 0
for a in ["bear", "mixed", "bull"]:
    k = tal == a
    if k.sum():
        print(f"  align={a:5s}: {int(k.sum()):3d} true sells, mean fwd4 {fwd4[ti][k].mean():+.2f}%")
for name, mask in [("ret4h<0", trec), ("ret4h>=0", ~trec)]:
    print(f"  {name:9s}: {int(mask.sum()):3d} true sells, mean fwd4 {fwd4[ti][mask].mean():+.2f}%")
print(f"  true sells with P(sell)>=0.40: {int((ps[ti]>=0.40).sum())}/{len(ti)}")

print("\n=== SELL LABEL OPTIONS (train pool = pre-eval hours) ===")
eval_start = truth.open_time.iloc[0]
pool = np.array([i for i in range(60, len(df)) if df.open_time.iloc[i] < eval_start and pd.notna(fwd.iloc[i])])
p_bear = (df.close.values[pool] < df.ema12.values[pool]) & (df.ema12.values[pool] < df.ema26.values[pool])
p_r4 = df.ret_4h.values[pool] < 0
pf = fwd.values[pool]
opts = {
    "A fwd<=-1.5 (v3)": pf <= -1.5,
    "B fwd<=-2.0": pf <= -2.0,
    "C fwd<=-1.5 & bear": (pf <= -1.5) & p_bear,
    "D fwd<=-1.5 & (bear|r4<0)": (pf <= -1.5) & (p_bear | p_r4),
    "E fwd<=-2.0 & bear": (pf <= -2.0) & p_bear,
}
for name, mask in opts.items():
    mm = mask[pool] if len(mask) != len(pool) else mask
    print(f"  {name:28s}: pool {int(mask.sum()):4d}, mean fwd4 {pf[mask].mean():+.2f}%")
# eval coverage of each option (recall ceiling on OLD truth)
for name, mask in opts.items():
    print(f"  eval true-sells covered by {name:24s}: ", end="")
    cov = 0
    for i in ti:
        row = m.loc[truth.open_time.iloc[i]]
        bear = (row.close < row.ema12) and (row.ema12 < row.ema26)
        r4 = row.ret_4h < 0
        f = fwd4[i]
        okk = (f <= -1.5 and (bear if "bear" in name and "r4" not in name else (bear or r4) if "r4" in name else True)) if f <= -1.5 else (f <= -2.0 and (bear if "bear" in name else True))
        if name == "A fwd<=-1.5 (v3)": okk = f <= -1.5
        elif name == "B fwd<=-2.0": okk = f <= -2.0
        elif name == "C fwd<=-1.5 & bear": okk = (f <= -1.5) and bear
        elif name == "D fwd<=-1.5 & (bear|r4<0)": okk = (f <= -1.5) and (bear or r4)
        elif name == "E fwd<=-2.0 & bear": okk = (f <= -2.0) and bear
        cov += bool(okk)
    print(f"{cov}/{len(ti)}")

print("\n=== BUY SIDE ===")
bi = np.where(pa == "buy")[0]
okb = fwd4[bi] > 0
print(f"buys: {len(bi)}, dir-right {okb.sum()} ({okb.mean()*100:.0f}%), mean fwd4 {fwd4[bi].mean():+.3f}%")
mi = np.where((lab == "buy") & (pa == "hold"))[0]
print(f"MISSED true buys (pred hold): {len(mi)}, mean fwd4 {fwd4[mi].mean() if len(mi) else 0:+.2f}%")
mib = pb[mi]
print(f"  their P(buy): mean {mib.mean():.3f}, max {mib.max():.3f}")

print("\n=== GATING (deployment lever, no retrain) ===")
for th in [0.35, 0.40, 0.45, 0.50]:
    bm = pb >= th
    sm = ps >= th
    pnl_b = fwd4[bm].sum()
    pnl_s = -fwd4[sm].sum()
    pnl = (np.where(bm, fwd4, 0) + np.where(sm, -fwd4, 0)).sum()
    fees = (bm.sum() + sm.sum()) * 0.2
    print(f"  th={th:.2f}: buys {int(bm.sum()):3d} (gross {pnl_b:+.2f}%), sells {int(sm.sum()):3d} (gross {pnl_s:+.2f}%), "
          f"net after fee {pnl - fees:+.2f}%")

print("\n=== CONVICTION HEAD ===")
conv01 = np.array([p["conv_probs"]["0"] for p in preds])
conv12 = np.array([p["conv_probs"]["2"] for p in preds])
print(f"P(conv=0): mean {conv01.mean():.3f} std {conv01.std():.3f} | P(conv=2): mean {conv12.mean():.3f}")
big = np.abs(fwd4) > 3
print(f"P(conv=2) when |fwd4|>3: {conv12[big].mean():.3f} vs |fwd4|<1: {conv12[np.abs(fwd4)<1].mean():.3f}")

print("\n=== ERROR CONCENTRATION ===")
wrong = np.where(pa != lab)[0]
w = feat.iloc[wrong]
print(f"wrong rows {len(wrong)}: by ATR tercile:", np.percentile(w.atr14, [25, 50, 75]).round(2))
print("wrong by truth:", pd.Series(lab[wrong]).value_counts().to_dict())
