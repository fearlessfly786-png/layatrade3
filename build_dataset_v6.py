"""Dataset v6 — fixes v5's confirmed weaknesses:
  W1 buy recall: 17/22 true buys missed  -> buys oversampled 2x
  W2 sells still fire on bear-oversold BOUNCES (16/17 calls!): the v5 relabel
     alone never untaught it. Two-pronged fix:
       a) sell class = dumps that CONTINUE (fwd4<=-1.5 AND fwd8<0), any alignment
       b) explicit counter-examples: bearish+oversold states that BOUNCED
          (fwd4>-0.5) forced into the hold class (~300 rows) -> teaches
          "oversold+bearish = bounce risk = hold"
  W3 regime gap -> pool extended to 2 years (18,000 candles, Sep 2024 -> Sep 2026)
  Hold = 1.0x trades (active balance, v2/v5 style). State format unchanged.
Eval: same frozen 994 states + truth (comparability preserved)."""
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
label = pd.Series("hold", index=df.index)
label[fwd4 > UP_TH] = "buy"
label[(fwd4 < -UP_TH) & (fwd8 < 0)] = "sell"          # continuation-confirmed dumps
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
print(f"train pool hours: {len(pool)}  ({df.open_time.iloc[pool[0]]} -> {df.open_time.iloc[pool[-1]]})")

buys = [i for i in pool if label.iloc[i] == "buy"]
sells = [i for i in pool if label.iloc[i] == "sell"]
holds = [i for i in pool if label.iloc[i] == "hold"]
# counter-examples: bearish + oversold-ish + BOUNCED (this is where v5 wrongly sells)
counter = [i for i in holds if bearish.iloc[i] and df.rsi.iloc[i] < 38 and fwd4.iloc[i] > -0.5]
print(f"pool: buy {len(buys)}, sell(continuation) {len(sells)}, hold {len(holds)}, "
      f"counter-examples {len(counter)} ({len(counter)/max(len(sells),1):.1f} per sell)")

rng = np.random.default_rng(11)
n_buy = min(int(2.0 * len(buys)), 650)
n_sell = min(int(1.5 * len(sells)), 450)
buy_sel = list(rng.choice(buys, size=n_buy, replace=True))
sell_sel = list(rng.choice(sells, size=n_sell, replace=True))
ctr_sel = list(rng.choice(counter, size=min(300, len(counter)), replace=False))
rest_hold = [h for h in holds if h not in set(ctr_sel)]
n_hold = (n_buy + n_sell) - len(ctr_sel)          # holds total ~= trades
hold_sel = list(rng.choice(rest_hold, size=max(n_hold, 0), replace=False))
train_idx = buy_sel + sell_sel + ctr_sel + hold_sel
rng.shuffle(train_idx)
print(f"v6 train rows: {len(train_idx)}  {Counter(label.iloc[i] for i in train_idx)}")

rows = [{"state": state_text(i), "questions": QUESTIONS,
         "expected": {"action": label.iloc[i], "conviction": int(conv.iloc[i])}}
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
    with open(f"chunk_v6_{n}.jsonl", "w") as f:
        for r in ch:
            f.write(json.dumps(r) + "\n")
    print(f"chunk_v6_{n}.jsonl: {len(ch)} rows {dict(Counter(r['expected']['action'] for r in ch))}")
