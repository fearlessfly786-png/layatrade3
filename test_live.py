"""LIVE TEST of the fine-tuned trading checkpoint on fresh Binance data.

Part A — RECENT REAL TEST: last ~56 scoreable hours (open_time <= now-4h).
         These fall in the chronological tail of the market (never used for
         training — training stopped ~Aug 2026; eval region is the Sept->now tail).
         Each hour: model decision vs what the price ACTUALLY did next 4h.
Part B — LIVE CALLS: the 4 freshest candles (after the 05:37 dataset build),
         decisions only — their 4h outcomes are still in the future.
"""
import json
import os
import time
import urllib.request

import numpy as np
import pandas as pd

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HOME", "/home/z/hf-cache")
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

SYMBOL = "BTCUSDT"
FWD_HOURS = 4
UP_TH, DOWN_TH = 1.0, -1.0
DATASET_BUILD_UTC = pd.Timestamp("2026-10-13 05:37", tz="UTC")

# ------------------------------------------------------------------
# 1. Fetch the latest candles
# ------------------------------------------------------------------
def fetch_recent(n: int) -> pd.DataFrame:
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=1h&limit={n}"
    with urllib.request.urlopen(url, timeout=30) as r:
        batch = json.loads(r.read())
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy", "taker_quote", "ignore"]
    df = pd.DataFrame(batch, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.reset_index(drop=True)

print("Fetching latest candles from Binance ...", flush=True)
df = fetch_recent(260)
now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
print(f"got {len(df)} candles: {df.open_time.iloc[0]} -> {df.open_time.iloc[-1]} (now {now_utc:%Y-%m-%d %H:%M} UTC)")

# ------------------------------------------------------------------
# 2. Indicators — identical formulas to build_trading_dataset.py
# ------------------------------------------------------------------
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

fwd_ret = (c.shift(-FWD_HOURS) / c - 1) * 100
label = pd.Series("hold", index=df.index)
label[fwd_ret > UP_TH] = "buy"
label[fwd_ret < DOWN_TH] = "sell"

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

warmup = 26
scoreable = [i for i in df.index
             if i >= warmup and df.open_time.iloc[i] <= DATASET_BUILD_UTC and pd.notna(fwd_ret.iloc[i])]
live = [i for i in df.index
        if i >= warmup and df.open_time.iloc[i] > DATASET_BUILD_UTC]
print(f"Part A (scoreable, unseen region after training split): {len(scoreable)} hours")
print(f"Part B (live, after dataset build, outcome in future):  {len(live)} hours")

# ------------------------------------------------------------------
# 3. Model predictions
# ------------------------------------------------------------------
import laya
print("\nLoading fine-tuned checkpoint ...", flush=True)
t0 = time.time()
agent = laya.load("./laya-trading-v6")
print(f"loaded in {time.time()-t0:.1f}s\n", flush=True)

all_idx = scoreable + live
states = [state_text(i) for i in all_idx]
results = agent.predict_batch(states, QUESTIONS, batch_size=8)

def parse(res):
    a, cv = res["answers"]["action"], res["answers"]["conviction"]
    return a.get("choice"), a.get("probabilities", {}), cv.get("score")

# ------------------------------------------------------------------
# 4. Part A — score against reality
# ------------------------------------------------------------------
print("=" * 96)
print(f"PART A — RECENT REAL TEST: model decisions vs actual next-4h price move")
print("=" * 96)
hits = {"buy": 0, "hold": 0, "sell": 0}
counts = {"buy": 0, "hold": 0, "sell": 0}
pred_dist = {"buy": 0, "hold": 0, "sell": 0}
rows_out = []
for k, i in enumerate(scoreable):
    choice, probs, score = parse(results[k])
    actual = label.iloc[i]
    fr = fwd_ret.iloc[i]
    hit = choice == actual
    counts[actual] += 1
    pred_dist[choice] += 1
    if hit:
        hits[actual] += 1
    rows_out.append({
        "open_time": str(df.open_time.iloc[i]), "close_price": float(c.iloc[i]),
        "pred": choice, "conviction": score, "probs": {k2: round(float(v), 4) for k2, v in probs.items()},
        "actual_4h_return_pct": round(float(fr), 2), "actual_label": actual, "correct": bool(hit),
    })

nA = len(scoreable)
acc = sum(r["correct"] for r in rows_out) / nA
hold_baseline = counts["hold"] / nA
print(f"\n{'time (UTC)':<18}{'price':>9} {'model says':>11} {'conf':>5} {'actual 4h move':>15} {'truth':>6} {'hit':>4}")
for r in rows_out[-24:]:
    t = r["open_time"].replace(" ", " ")[5:16]
    conf = r["probs"].get(r["pred"], 0)
    print(f"{t:<18}{r['close_price']:>9.0f} {r['pred']:>11} {conf:>5.2f} {r['actual_4h_return_pct']:>+14.2f}% {r['actual_label']:>6} {'Y' if r['correct'] else 'n':>4}")
print(f"  ... (showing last 24 of {nA}; full log in live_test_results.json)")

print("\n--- SUMMARY (all scoreable hours) ---")
print(f"model accuracy            : {acc:.4f}  ({sum(r['correct'] for r in rows_out)}/{nA})")
print(f"'always hold' baseline    : {hold_baseline:.4f}")
print(f"true label counts         : {counts}")
print(f"model decision counts     : {pred_dist}")
for act in ("buy", "sell"):
    if pred_dist[act]:
        trade_calls = [r for r in rows_out if r["pred"] == act]
        wins = [r for r in trade_calls if (act == "buy" and r["actual_4h_return_pct"] > 0)
                or (act == "sell" and r["actual_4h_return_pct"] < 0)]
        avg_ret = np.mean([r["actual_4h_return_pct"] for r in trade_calls])
        print(f"model '{act}' calls        : {len(trade_calls)} | directionally right: {len(wins)}/{len(trade_calls)} "
              f"| avg actual 4h return after call: {avg_ret:+.3f}%")
    else:
        print(f"model '{act}' calls        : 0")

# ------------------------------------------------------------------
# 5. Part B — live calls (outcome in the future)
# ------------------------------------------------------------------
print("\n" + "=" * 96)
print("PART B — LIVE CALLS RIGHT NOW (never-seen candles, outcomes still in the future)")
print("=" * 96)
live_out = []
for k, i in enumerate(live):
    choice, probs, score = parse(results[len(scoreable) + k])
    resolves = df.open_time.iloc[i] + pd.Timedelta(hours=FWD_HOURS)
    o = {
        "candle_open_utc": str(df.open_time.iloc[i]),
        "price_at_decision": float(c.iloc[i]),
        "decision": choice, "conviction": score,
        "probabilities": {k2: round(float(v), 4) for k2, v in probs.items()},
        "resolves_utc": str(resolves),
    }
    live_out.append(o)
    print(f"  candle {df.open_time.iloc[i]:%Y-%m-%d %H:%M} UTC | price {c.iloc[i]:,.0f} | "
          f"DECISION: {choice.upper()} (conviction {score:.2f}, "
          f"P(buy)={probs.get('buy',0):.3f} P(hold)={probs.get('hold',0):.3f} P(sell)={probs.get('sell',0):.3f}) "
          f"-> resolves {resolves:%H:%M} UTC")

with open("live_test_results.json", "w") as f:
    json.dump({"part_a": rows_out, "part_a_summary": {
        "n": nA, "accuracy": round(acc, 4), "hold_baseline": round(hold_baseline, 4),
        "true_counts": counts, "pred_counts": pred_dist},
        "part_b_live_calls": live_out}, f, indent=1)
print("\nsaved -> live_test_results.json")
