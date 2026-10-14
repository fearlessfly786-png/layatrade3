"""
Step 1+2 — Download real BTC/USDT hourly candles from Binance and build
a Laya-format trading decision dataset (JSONL), chronologically split.

Teacher labeling (transparent, no lookahead in features):
  forward 4h return > +1.0%  -> BUY
  forward 4h return < -1.0%  -> SELL
  otherwise                  -> HOLD
"""
import json
import time
import urllib.request

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
TOTAL = 5000          # ~7 months of hourly candles
FWD_HOURS = 4         # decision horizon
UP_TH, DOWN_TH = 1.0, -1.0  # % thresholds

# ---------------------------------------------------------------
# 1. Download klines (Binance public REST, no key needed)
# ---------------------------------------------------------------
def fetch_klines(symbol: str, interval: str, total: int) -> pd.DataFrame:
    rows, end_time = [], None
    while len(rows) < total:
        url = (f"https://api.binance.com/api/v3/klines?symbol={symbol}"
               f"&interval={interval}&limit=1000")
        if end_time is not None:
            url += f"&endTime={end_time}"
        with urllib.request.urlopen(url, timeout=30) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        rows = batch + rows
        end_time = batch[0][0] - 1
        time.sleep(0.3)
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy", "taker_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume", "quote_vol"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df

print(f"Downloading {TOTAL} x {INTERVAL} {SYMBOL} candles from Binance ...")
df = fetch_klines(SYMBOL, INTERVAL, TOTAL)
print(f"Got {len(df)} candles: {df.open_time.iloc[0]} -> {df.open_time.iloc[-1]}")

# ---------------------------------------------------------------
# 2. Technical indicators (computed only from data up to time t)
# ---------------------------------------------------------------
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

# ---------------------------------------------------------------
# 3. Teacher labels from FORWARD returns (the decision target)
# ---------------------------------------------------------------
fwd_ret = (c.shift(-FWD_HOURS) / c - 1) * 100
label = pd.Series("HOLD", index=df.index)
label[fwd_ret > UP_TH] = "buy"
label[fwd_ret < DOWN_TH] = "sell"

# conviction score from |forward return| (0..2 over [weak, moderate, strong])
mag = fwd_ret.abs()
conviction = pd.Series(0, index=df.index)
conviction[(mag > UP_TH) | (mag < abs(DOWN_TH))] = 1
conviction[mag > 2.0] = 2
conviction[label == "HOLD"] = 0
label = label.str.lower()  # hold / buy / sell

# ---------------------------------------------------------------
# 4. Render text states (what the model will read)
# ---------------------------------------------------------------
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

QUESTIONS = {
    "action": {
        "type": "choice",
        "instructions": "Given this market state, what trade action should be taken over the next 4 hours?",
        "criteria": {
            "buy": "upward move expected, long position favored",
            "hold": "no clear edge, stay flat",
            "sell": "downward move expected, exit or short favored",
        },
    },
    "conviction": {
        "type": "score",
        "instructions": "How strong is the expected move over the next 4 hours?",
        "criteria": ["weak or no move", "moderate move", "strong move"],
    },
}

# ---------------------------------------------------------------
# 5. Build rows, chronological split 80/20 (never shuffle time!)
# ---------------------------------------------------------------
warmup = 26  # indicator warmup
valid = df.index[(df.index >= warmup) & (fwd_ret.notna())]
rows = []
for i in valid:
    rows.append({
        "state": state_text(i),
        "questions": QUESTIONS,
        "expected": {"action": label[i], "conviction": int(conviction[i])},
    })

split = int(len(rows) * 0.8)
all_train, eval_rows = rows[:split], rows[split:]   # chronological!

# --- Balance the TRAINING set: keep every buy/sell, subsample hold ---
rng = np.random.default_rng(42)
buy = [r for r in all_train if r["expected"]["action"] == "buy"]
sell = [r for r in all_train if r["expected"]["action"] == "sell"]
hold = [r for r in all_train if r["expected"]["action"] == "hold"]
n_hold = min(len(hold), len(buy) + len(sell))  # hold ~50% of train
train_rows = buy + sell + list(rng.choice(hold, size=n_hold, replace=False))
rng.shuffle(train_rows)
# eval stays at the REAL market distribution (no balancing - honest numbers)

with open("trade_train.jsonl", "w") as f:
    for r in train_rows:
        f.write(json.dumps(r) + "\n")
with open("trade_eval.jsonl", "w") as f:
    for r in eval_rows:
        f.write(json.dumps(r) + "\n")

# ---------------------------------------------------------------
# 6. Report
# ---------------------------------------------------------------
print(f"\nDataset: {len(rows)} decisions  (train {len(train_rows)} / eval {len(eval_rows)})")
print("Train action balance:", pd.Series([r['expected']['action'] for r in train_rows]).value_counts().to_dict())
print("Eval  action balance:", pd.Series([r['expected']['action'] for r in eval_rows]).value_counts().to_dict())
print("\nSample state:\n ", train_rows[0]["state"])
print("\nSample expected:", train_rows[0]["expected"])
print("\nSaved: trade_train.jsonl, trade_eval.jsonl")
