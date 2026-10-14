"""LIVE TEST of the v7 two-head consensus system on fresh Binance data.
Heads: laya-trading-v6 (quality anchor) + ck_d_8 (v7 counter-trained head).
Gate: buy iff both heads buy AND avg P(buy) >= 0.40; sells vetoed; else hold.
Part A: scoreable recent hours (4h outcome known) vs what actually happened.
Part B: freshest hours — decisions only (outcome in the future). TEST ONLY."""
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
AVG_PB_GATE = 0.40

def fetch_recent(n: int) -> pd.DataFrame:
    url = f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval=1h&limit={n}"
    with urllib.request.urlopen(url, timeout=30) as r:
        batch = json.loads(r.read())
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy", "taker_quote", "ignore"]
    df = pd.DataFrame(batch, columns=cols)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = df[col].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df.reset_index(drop=True)

print("Fetching latest candles from Binance ...", flush=True)
df = fetch_recent(300)
now_utc = pd.Timestamp.utcnow().tz_convert("UTC")
print(f"got {len(df)} candles: {df.open_time.iloc[0]} -> {df.open_time.iloc[-1]} (now {now_utc:%Y-%m-%d %H:%M} UTC)")

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
scoreable = [i for i in df.index if i >= warmup and pd.notna(fwd_ret.iloc[i])][-120:]
live = [i for i in df.index if i >= warmup and pd.isna(fwd_ret.iloc[i])]
print(f"Part A (scoreable): {len(scoreable)} hours | Part B (live, future outcome): {len(live)} hours")

import laya, gc
print("Loading head A (v6) ...", flush=True)
t0 = time.time()
agent_a = laya.load("./laya-trading-v6")
print(f"loaded in {time.time()-t0:.1f}s", flush=True)
all_idx = scoreable + live
states = [state_text(i) for i in all_idx]
ra = agent_a.predict_batch(states, QUESTIONS, batch_size=8)
del agent_a
gc.collect()
print("head A done, freed. Loading head B (v7 counter-trained) ...", flush=True)
t0 = time.time()
agent_b = laya.load("./ck_d_8")
print(f"loaded in {time.time()-t0:.1f}s", flush=True)
rb = agent_b.predict_batch(states, QUESTIONS, batch_size=8)
del agent_b
gc.collect()
print("head B done.\n", flush=True)

def gate(pa, pb):
    a6, p6 = pa["answers"]["action"].get("choice"), pa["answers"]["action"].get("probabilities", {})
    a7, p7 = pb["answers"]["action"].get("choice"), pb["answers"]["action"].get("probabilities", {})
    avg_pb = (p6.get("buy", 0) + p7.get("buy", 0)) / 2
    if a6 == "buy" and a7 == "buy" and avg_pb >= AVG_PB_GATE:
        return "buy", avg_pb
    return "hold", avg_pb

ok = 0
book = 0.0
trades = 0
wins = 0
print("Part A — recent scoreable hours (v7 system vs actual 4h outcome):")
for k, i in enumerate(scoreable):
    act, avg_pb = gate(ra[k], rb[k])
    actual = fwd_ret.iloc[i]
    lab = "buy" if actual > UP_TH else ("sell" if actual < DOWN_TH else "hold")
    correct = act == lab
    ok += correct
    if act == "buy":
        net = actual / 100 * 1000 - 2
        book += net
        trades += 1
        wins += net > 0
    print(f"  {df.open_time.iloc[i]} close {df.close.iloc[i]:.0f} -> {act:<4} (avgPb {avg_pb:.2f}) | next4h {actual:+.2f}% ({lab}) {'OK' if correct else 'X'}")
print(f"Part A accuracy: {ok}/{len(scoreable)} = {ok/len(scoreable)*100:.1f}% | book {trades} trades, {wins} wins, ${book:+.2f}")

print("\nPart B — LIVE calls (outcome in the future):")
for k, i in enumerate(live):
    act, avg_pb = gate(ra[len(scoreable) + k], rb[len(scoreable) + k])
    print(f"  {df.open_time.iloc[i]} close {df.close.iloc[i]:.0f} -> {act} (avgPb {avg_pb:.2f})")

json.dump({
    "tested_at": str(now_utc), "candles": len(df),
    "part_a": {"n": len(scoreable), "acc": round(ok / len(scoreable), 4),
               "trades": trades, "wins": wins, "book": round(book, 2)},
    "part_b": [{"time": str(df.open_time.iloc[i]), "close": float(df.close.iloc[i]),
                "call": gate(ra[len(scoreable) + k], rb[len(scoreable) + k])[0]}
               for k, i in enumerate(live)],
}, open("live_test_v7_results.json", "w"), indent=1)
print("\nsaved -> live_test_v7_results.json")
