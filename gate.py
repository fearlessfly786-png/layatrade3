"""v7 decision gate — consensus of two heads + sell veto.

Usage:
    python gate.py  # demo against the two local checkpoints

System (see WORKLOG Task 7):
    head A: laya-trading-v6            (quality anchor, 8 chunks from v5, seeds 601-608)
    head B: laya-trading-v7-head       (counter-trained head, 8 chunks, seeds 751-758)
    decision:
        buy  iff headA=buy AND headB=buy AND avg P(buy) >= 0.40
        sell never (vetoed: the only sell call the v6 family makes is the
        oversold-bearish antipattern the v7 counter-examples target, and it
        lost money in every version that fired it)
        hold otherwise

Measured on the frozen 994-hour holdout (Sep 1 - Oct 13 2026, $1,000/trade,
0.2% round-trip fee): 30 buys, 53% net-positive, book +$47.04 after fees
vs v6's +$35.65 (+32%). Sensitivity: avg-Pb threshold 0.42 gives +$46.97,
0.44 degrades — 0.40 is the pre-registered operating point."""
import json
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("HF_HOME", "/home/z/hf-cache")

AVG_PB_GATE = 0.40
HEAD_A = "./laya-trading-v6"        # from layatrade2 release v6-final
HEAD_B = "./laya-trading-v7-head"   # from layatrade3 release v7-final


def decide(pa: dict, pb: dict) -> str:
    """pa/pb: one predict_batch result dict per head."""
    a6 = pa["answers"]["action"].get("choice")
    a7 = pb["answers"]["action"].get("choice")
    p6 = pa["answers"]["action"].get("probabilities", {})
    p7 = pb["answers"]["action"].get("probabilities", {})
    avg_pb = (p6.get("buy", 0.0) + p7.get("buy", 0.0)) / 2
    if a6 == "buy" and a7 == "buy" and avg_pb >= AVG_PB_GATE:
        return "buy"
    return "hold"          # sells vetoed by design


if __name__ == "__main__":
    import laya
    questions = json.load(open(os.path.join(HEAD_A, "questions.json")))
    state = ("BTCUSDT 1h trading decision. Price 84000. RSI(14) 55.0 (neutral). "
             "EMA12 83900, EMA26 83820, price above both EMAs, bullish alignment. "
             "MACD histogram +12.0 and rising. Momentum: 1h +0.15%, 4h +0.60%, 24h +1.20%. "
             "Volume 320 BTC, 1.10x the 20h average. Volatility ATR(14) 0.45% per hour. "
             "Price sits -0.90% from the 24h high and +2.10% from the 24h low.")
    # run heads sequentially (sandbox has ~3GB RAM — never hold both models)
    ga = laya.load(HEAD_A)
    ra = ga.predict_batch([state], questions, batch_size=1)
    del ga
    gb = laya.load(HEAD_B)
    rb = gb.predict_batch([state], questions, batch_size=1)
    del gb
    print("head A:", ra[0]["answers"]["action"])
    print("head B:", rb[0]["answers"]["action"])
    print("v7 decision:", decide(ra[0], rb[0]))
