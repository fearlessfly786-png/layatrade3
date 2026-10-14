"""Attempt D dataset — v6 recipe (simple buys: the class the chain provably fits)
+ counter-example attack on v6's confirmed loss sources:
    - 128 faded-pump counters : fwd4>1.5 but fwd8<=0 (v6's 18 bounce-buy calls, -$108.70)
    - 300 momentum-blip counters: bullish momentum look (ret_4h>+1 & macd rising & not bearish)
      but fwd4<=+0.2 (v6's 29 false buys, -$101.96)
    - 300 oversold-bounce counters (v6's proven sell-leak fix)
Buy = fwd4>1.5 (simple), Sell = fwd4<-1.5 & fwd8<0 (v6), holds = trades. State format unchanged."""
import json
import numpy as np
import pandas as pd
from collections import Counter, defaultdict

UP_TH = 1.5
SYMBOL = "BTCUSDT"

df = pd.read_csv("candles_18000.csv", parse_dates=["open_time"])
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
fwd4 = (c.shift(-4) / c - 1) * 100
fwd8 = (c.shift(-8) / c - 1) * 100

bearish = (df.close < df.ema12) & (df.ema12 < df.ema26)
macd_up = df.macd_hist > df.macd_hist.shift(1)
label = pd.Series("hold", index=df.index)
label[fwd4 > UP_TH] = "buy"                                  # simple (v6) — fits well
label[(fwd4 < -UP_TH) & (fwd8 < 0)] = "sell"                 # v6 continuation dump
conv = pd.Series(0, index=df.index)
mag = fwd4.abs()
conv[(label != "hold") & (mag > UP_TH)] = 1
conv[(label != "hold") & (mag > 2 * UP_TH)] = 2

QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "Given this market state, what trade action should be taken over the next 4 hours?",
        "criteria": {
            "buy": "upward move expected, long position favored",
            "hold": "no clear edge, bounce or fade risk, stay flat",
            "sell": "downward move expected, exit or short favored",
        },
    },
    "conviction": {
        "type": "score",
        "instructions": "How strong is the expected move over the next 4 hours?",
        "criteria": ["weak or no move", "moderate move", "strong move"],
    },
}

def state_text(i: int) -> str:
    r = df.iloc[i]
    trend = ("above both EMAs, bullish alignment" if r.close > r.ema12 > r.ema26
             else "below both EMAs, bearish alignment" if r.close < r.ema12 < r.ema26
             else "mixed EMA alignment")
    macd_dir = "rising" if r.macd_hist > df.macd_hist.iloc[i - 1] else "falling"
    rsi_txt = ("overbought" if r.rsi > 70 else "oversold" if r.rsi < 30 else "neutral")
    return (
        f"{SYMBOL} 1h trading decision. Price {r.close:.0f}. "
        f"RSI(14) {r.rsi:.1f} ({rsi_txt}). "
        f"EMA12 {r.ema12:.0f}, EMA26 {r.ema26:.0f}, price {trend}. "
        f"MACD histogram {r.macd_hist:+.1f} and {macd_dir}. "
        f"Momentum: 1h {r.ret_1h:+.2f}%, 4h {r.ret_4h:+.2f}%, 24h {r.ret_24h:+.2f}%. "
        f"Volume {r.volume:.0f} BTC, {r.vol_ratio:.2f}x the 20h average. "
        f"Volatility ATR(14) {r.atr14:.2f}% per hour. "
        f"Price sits {r.dist_hi:+.2f}% from the 24h high and {r.dist_lo:+.2f}% from the 24h low."
    )

eval_start = pd.Timestamp("2026-09-01 16:00:00+00:00")
WARMUP = 60
pool = [i for i in range(WARMUP, len(df))
        if df.open_time.iloc[i] < eval_start and pd.notna(fwd4.iloc[i]) and pd.notna(fwd8.iloc[i])]

bounce = {i for i in pool if fwd4.iloc[i] > UP_TH and fwd8.iloc[i] <= 0}   # pops that die by 8h
buys = [i for i in pool if label.iloc[i] == "buy" and i not in bounce]     # carve bounces out of buy
sells = [i for i in pool if label.iloc[i] == "sell"]
holds = [i for i in pool if label.iloc[i] == "hold"]
counter_os = [i for i in holds if bearish.iloc[i] and df.rsi.iloc[i] < 38 and fwd4.iloc[i] > -0.5]
counter_fp = sorted(bounce)                                                 # faded pumps -> explicit hold
counter_blip = [i for i in holds if (df.ret_4h.iloc[i] > 1.0) and macd_up.iloc[i]
                and not bearish.iloc[i] and fwd4.iloc[i] <= 0.2]
print(f"pool: buy {len(buys)}, sell {len(sells)}, hold {len(holds)}, bounces {len(bounce)}")
print(f"counters: oversold {len(counter_os)}, faded-pump {len(counter_fp)}, momentum-blip {len(counter_blip)}")

rng = np.random.default_rng(23)
n_buy = min(int(1.5 * len(buys)), 900)
n_sell = min(int(1.5 * len(sells)), 400)
buy_sel = list(rng.choice(buys, size=n_buy, replace=True))
sell_sel = list(rng.choice(sells, size=n_sell, replace=True))
os_sel = list(rng.choice(counter_os, size=min(300, len(counter_os)), replace=False))
fp_sel = list(rng.choice(counter_fp, size=min(4 * len(counter_fp), len(counter_fp) * 4), replace=True))
blip_sel = list(rng.choice(counter_blip, size=min(300, len(counter_blip)), replace=False))
n_hold = max(n_buy - len(os_sel) - len(fp_sel) - len(blip_sel), 0)
rest_hold = [h for h in holds if h not in set(os_sel) | set(fp_sel) | set(blip_sel)]
hold_sel = list(rng.choice(rest_hold, size=min(n_hold, len(rest_hold)), replace=False))
train_idx = buy_sel + sell_sel + os_sel + fp_sel + blip_sel + hold_sel
rng.shuffle(train_idx)
print(f"attempt-D rows: {len(train_idx)}  {Counter(label.iloc[i] for i in train_idx)}")

override = {}
for i in os_sel:
    override[i] = "hold"   # oversold bounce -> hold
for i in fp_sel:
    override[i] = "hold"   # faded pump -> hold (was fwd4>1.5 = buy by rule)
for i in blip_sel:
    override[i] = "hold"   # momentum blip -> hold
rows = [{"state": state_text(i), "questions": QUESTIONS,
         "expected": {"action": override.get(i, label.iloc[i]), "conviction": int(conv.iloc[i])}}
        for i in train_idx]

by_lab = defaultdict(list)
for r in rows:
    by_lab[r["expected"]["action"]].append(r)
for v in by_lab.values():
    rng.shuffle(v)
CHUNKS = 8
chunks = [[] for _ in range(CHUNKS)]
pools = {k: list(v) for k, v in by_lab.items()}
ci = 0
while any(pools.values()):
    for lab in ("buy", "sell", "hold"):
        if pools[lab]:
            chunks[ci % CHUNKS].append(pools[lab].pop())
    ci += 1
for n, ch in enumerate(chunks, 1):
    rng.shuffle(ch)
    with open(f"chunk_d_{n}.jsonl", "w") as f:
        for r in ch:
            f.write(json.dumps(r) + "\n")
    print(f"chunk_d_{n}.jsonl: {len(ch)} rows {dict(Counter(r['expected']['action'] for r in ch))}")
