"""Fetch expanded BTC/USDT history (9000 candles, ~1 year) and build a truth table
for the eval window (timestamp, close, fwd 4h returns) to diagnose the v1 model.

Mapping: v1 dataset built 2026-10-13 05:37 UTC from 5000 candles ending 05:00 open.
Old candle index 4999 (0-based) == open_time 2026-10-13 05:00.
Eval rows = old indices 4002..4995 -> opens 05:00-(997..4)h.
"""
import json
import time
import urllib.request

import numpy as np
import pandas as pd

TOTAL = 9000

def fetch_klines(total: int) -> pd.DataFrame:
    rows, end_time = [], None
    while len(rows) < total:
        url = (f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT"
               f"&interval=1h&limit=1000")
        if end_time is not None:
            url += f"&endTime={end_time}"
        with urllib.request.urlopen(url, timeout=30) as r:
            batch = json.loads(r.read())
        if not batch:
            break
        rows = batch + rows
        end_time = batch[0][0] - 1
        time.sleep(0.25)
    cols = ["open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_vol", "trades", "taker_buy", "taker_quote", "ignore"]
    df = pd.DataFrame(rows, columns=cols)
    df = df.drop_duplicates("open_time").sort_values("open_time").reset_index(drop=True)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df

print(f"Downloading {TOTAL} x 1h BTCUSDT candles ...", flush=True)
df = fetch_klines(TOTAL)
print(f"got {len(df)} candles: {df.open_time.iloc[0]} -> {df.open_time.iloc[-1]}")

# anchor: old index 4999 == candle opening 2026-10-13 05:00 UTC
anchor = df.index[df.open_time == pd.Timestamp("2026-10-13 05:00", tz="UTC")]
assert len(anchor) == 1, f"anchor not found: {anchor}"
a = anchor[0]
# old index i <-> fresh index a - (4999 - i)
df["old_index"] = 4999 - (a - df.index)

c = df["close"]
df["fwd4"] = (c.shift(-4) / c - 1) * 100

# eval window: old indices 4002..4995
ev = df[(df.old_index >= 4002) & (df.old_index <= 4995)].copy()
print(f"eval window rows: {len(ev)}  ({ev.open_time.iloc[0]} -> {ev.open_time.iloc[-1]})")
print(f"eval fwd4 complete: {ev.fwd4.notna().sum()}/{len(ev)}")

df.to_csv("candles_9000.csv", index=False)
ev[["old_index", "open_time", "close", "fwd4"]].to_csv("eval_truth.csv", index=False)

# quick regime view
print("\nRegime snapshot (monthly mean |4h move|):")
m = df.set_index("open_time").fwd4.abs().resample("ME").mean()
print((m * 1).round(3).tail(14).to_string())
print("\nsaved: candles_9000.parquet, eval_truth.parquet")
