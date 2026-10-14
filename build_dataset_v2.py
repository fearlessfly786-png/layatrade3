"""Dataset v2 — fixes for diagnosed weak points:
  1. Cleaner labels: ±1.5% (strong moves only) instead of ±1.0%
  2. Bigger + older training pool: Oct 2025 -> Aug 2026 (~8000h, incl. volatile
     regimes the v1 model never saw), all strictly before the eval window.
  3. State text format UNCHANGED (identical template & formulas).
Eval rows reuse trade_eval.jsonl states verbatim, relabeled with ±1.5%.
"""
import json
import numpy as np
import pandas as pd
from collections import Counter

UP_TH, DOWN_TH = 1.5, -1.5
FWD_HOURS = 4
SYMBOL = "BTCUSDT"

df = pd.read_csv("candles_9000.csv", parse_dates=["open_time"])
c = df["close"]

# indicators — identical formulas to build_trading_dataset.py
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

fwd = (c.shift(-FWD_HOURS) / c - 1) * 100
label = pd.Series("hold", index=df.index)
label[fwd > UP_TH] = "buy"
label[fwd < DOWN_TH] = "sell"
mag = fwd.abs()
conv = pd.Series(0, index=df.index)
conv[mag > UP_TH] = 1          # note: |fwd|>1.5 while label!=hold impossible for sell<-1.5 -> mag>1.5; buy same
conv[mag > 2 * UP_TH] = 2
conv[label == "hold"] = 0

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

# ---------------- EVAL: same 994 states as v1, relabeled ----------------
ev_truth = pd.read_csv("eval_truth.csv", parse_dates=["open_time"])
ev_rows_v1 = [json.loads(l) for l in open("trade_eval.jsonl") if l.strip()]
assert len(ev_truth) == len(ev_rows_v1) == 994
eval_v2 = []
for r, (_, t) in zip(ev_rows_v1, ev_truth.iterrows()):
    f = t.fwd4
    lab = "buy" if f > UP_TH else "sell" if f < DOWN_TH else "hold"
    cvv = 0 if lab == "hold" else (2 if abs(f) > 2 * UP_TH else 1)
    eval_v2.append({"state": r["state"], "questions": QUESTIONS,
                    "expected": {"action": lab, "conviction": cvv}})
with open("trade_eval_v2.jsonl", "w") as f:
    for r in eval_v2:
        f.write(json.dumps(r) + "\n")
print(f"eval_v2: {len(eval_v2)} rows  {Counter(r['expected']['action'] for r in eval_v2)}")

# ---------------- TRAIN POOL: everything before the eval window ----------------
eval_start = ev_truth.open_time.iloc[0]
WARMUP = 60  # extra-converged indicators
pool = df.index[(df.open_time < eval_start) & (df.index >= WARMUP) & fwd.notna() & (fwd.index.isin(range(WARMUP, len(df))))]
pool = [i for i in range(WARMUP, len(df))
        if df.open_time.iloc[i] < eval_start and pd.notna(fwd.iloc[i])]
print(f"train pool hours: {len(pool)}  ({df.open_time.iloc[pool[0]]} -> {df.open_time.iloc[pool[-1]]})")

buys = [i for i in pool if label.iloc[i] == "buy"]
sells = [i for i in pool if label.iloc[i] == "sell"]
holds = [i for i in pool if label.iloc[i] == "hold"]
print(f"pool labels: buy {len(buys)}, sell {len(sells)}, hold {len(holds)} "
      f"(trade rate {(len(buys)+len(sells))/len(pool)*100:.1f}%)")

n_hold = min(len(holds), len(buys) + len(sells))  # hold ~50% of train
rng = np.random.default_rng(42)
train_idx = list(buys) + list(sells) + list(rng.choice(holds, size=n_hold, replace=False))
rng.shuffle(train_idx)
print(f"balanced v2 train rows: {len(train_idx)}  "
      f"{Counter(label.iloc[i] for i in train_idx)}")

rows = [{"state": state_text(i), "questions": QUESTIONS,
         "expected": {"action": label.iloc[i], "conviction": int(conv.iloc[i])}}
        for i in train_idx]

# stratified 4 chunks
from collections import defaultdict
by_lab = defaultdict(list)
for r in rows:
    by_lab[r["expected"]["action"]].append(r)
for v in by_lab.values():
    rng.shuffle(v)
CHUNKS = 4
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
    with open(f"chunk_v2_{n}.jsonl", "w") as f:
        for r in ch:
            f.write(json.dumps(r) + "\n")
    print(f"chunk_v2_{n}.jsonl: {len(ch)} rows {dict(Counter(r['expected']['action'] for r in ch))}")
