"""Version comparison chart v1 -> v7 (994 never-trained hours, Sep 1 - Oct 13 2026).
Sources: WORKLOG metrics (v1-v6, as published) + final_v7_results.json (v7 system)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

versions = ["v1", "v2", "v3", "v4\n(failed)", "v5", "v6", "v7\n(consensus)"]
acc = [78.57, 78.80, 90.95, 95.17, 92.05, 92.76, 88.03]
BASELINE = 95.37  # always-hold on frozen simple labels; v6's 92.76 used v6-style labels (87.5 on frozen)
total_net = [-203.90, -139.80, -55.82, -67.40, -40.36, 35.65, 47.04]   # $ per $1,000, after 0.2% fee
side_versions = ["v3", "v4\n(failed)", "v5", "v6", "v7"]
buy_net = [33.52, -67.40, 58.96, 39.07, 47.04]
sell_net = [-89.33, 0.0, -99.32, -3.43, 0.0]
trades = [None, None, None, None, 44, 38, 30]
winrt = [None, None, None, None, None, 53, 53]

GREEN, RED, GRAY, DARK, GOLD = "#2e7d32", "#c62828", "#9e9e9e", "#212121", "#b8860b"
fig, axes = plt.subplots(3, 1, figsize=(10.5, 12.5))
fig.suptitle("LayaTrade versions v1 \u2192 v7  \u2014  994 never-trained BTCUSDT hours (Sep 1 \u2013 Oct 13 2026)\n"
             "each call = one 4h position, $1,000 notional, 0.2% round-trip fee",
             fontsize=13, fontweight="bold", y=0.985)

ax = axes[0]
colors = [GRAY, GRAY, GREEN, GRAY, GREEN, GREEN, GOLD]
bars = ax.bar(versions, acc, color=colors, width=0.62)
ax.axhline(BASELINE, color=DARK, linestyle="--", linewidth=1.4)
ax.text(6.45, BASELINE + 0.4, f"always-hold baseline {BASELINE}%", fontsize=9, ha="right", color=DARK)
for b, a in zip(bars, acc):
    ax.text(b.get_x() + b.get_width()/2, a + 0.5, f"{a:.2f}%", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("accuracy (%)")
ax.set_title("Held-out accuracy  (v7 = frozen-label truth; v6 92.76% used v6-style labels, 87.5% on frozen)",
             fontsize=10.5, loc="left", fontweight="bold")
ax.set_ylim(70, 100)

ax = axes[1]
colors = [GREEN if v > 0 else RED for v in total_net]
colors[3] = GRAY
bars = ax.bar(versions, total_net, color=colors, width=0.62)
ax.axhline(0, color=DARK, linewidth=1.2)
for b, v in zip(bars, total_net):
    off = 6 if v > 0 else -14
    ax.text(b.get_x() + b.get_width()/2, v + off, f"${v:+.2f}", ha="center", fontsize=10, fontweight="bold")
ax.set_ylabel("net P&L after fees ($)")
ax.set_title("TOTAL net profit / loss after fees  (v7 is the best book: +$47.04)", fontsize=11, loc="left", fontweight="bold")
ax.set_ylim(min(total_net) - 40, max(total_net) + 45)

ax = axes[2]
x = np.arange(len(side_versions)); w = 0.36
b1 = ax.bar(x - w/2, buy_net, w, label="buy (long) net after fees", color=GREEN)
b2 = ax.bar(x + w/2, sell_net, w, label="sell (short) net after fees", color=RED)
ax.axhline(0, color=DARK, linewidth=1.2)
for bars_ in (b1, b2):
    for b in bars_:
        v = b.get_height()
        ax.text(b.get_x() + b.get_width()/2, v + (5 if v > 0 else -13), f"${v:+.0f}", ha="center", fontsize=9)
ax.set_xticks(x); ax.set_xticklabels(side_versions)
ax.set_ylabel("net P&L after fees ($)")
ax.set_title("Per-side breakdown v3 \u2192 v7  (v7: sell losses fully eliminated, buy book +$47)",
             fontsize=11, loc="left", fontweight="bold")
ax.legend(loc="upper right", fontsize=9)
ax.set_ylim(min(sell_net) - 55, max(buy_net) + 40)

for a in axes:
    a.spines[["top", "right"]].set_visible(False)
    a.grid(axis="y", alpha=0.25)

plt.tight_layout(rect=[0, 0, 1, 0.955])
plt.savefig("chart_versions.png", dpi=140, facecolor="white")
print("saved chart_versions.png (v1 -> v7)")
