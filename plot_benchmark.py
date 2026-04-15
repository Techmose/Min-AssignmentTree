"""
plot_benchmark.py

Reads benchmark_results.csv and produces line graphs comparing enumerator
performance. One chart is generated per unique N value, with K on the x-axis
and time (seconds) on the y-axis. Each enumerator config is its own line.

Usage:
    python plot_benchmark.py [csv_file]

    csv_file  Path to benchmark_results.csv. Defaults to benchmark_results.csv
              in the current directory.

Output:
    benchmark_n{N}.png  - one plot per N value, saved to the working directory
"""

import sys
import os
import csv
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# =============================================================================
# CONFIGURE
# =============================================================================

CSV_FILE    = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"
OUTPUT_DIR  = "."          # directory to save plots

# Line style per config label — add entries here if you add more configs
STYLES = {
    "OG_FFFF": {"color": "#1f77b4", "marker": "o", "linestyle": "-",  "label": "OG baseline"},
    "OG_TFFF": {"color": "#ff7f0e", "marker": "s", "linestyle": "--", "label": "OG priority queue eviction"},
    "MURTY":   {"color": "#2ca02c", "marker": "^", "linestyle": ":",  "label": "Murty"},
}

# =============================================================================
# Load data
# =============================================================================

# data[k][config] = list of (n, time_s) sorted by n
data = defaultdict(lambda: defaultdict(list))

with open(CSV_FILE, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        n      = int(row["n"])
        k      = int(row["k"])
        config = row["config"].strip()
        time_s = float(row["time_s"])
        data[k][config].append((n, time_s))

# Sort each series by n
for k in data:
    for config in data[k]:
        data[k][config].sort(key=lambda x: x[0])

# =============================================================================
# Plot — one figure per K
# =============================================================================

for k in sorted(data.keys()):
    fig, ax = plt.subplots(figsize=(8, 5))

    for config, points in sorted(data[k].items()):
        ns     = [p[0] for p in points]
        times  = [p[1] for p in points]
        style  = STYLES.get(config, {"color": "gray", "marker": "x",
                                     "linestyle": "-", "label": config})
        ax.plot(ns, times,
                color=style["color"],
                marker=style["marker"],
                linestyle=style["linestyle"],
                label=style["label"],
                linewidth=2,
                markersize=6)

    ax.set_title(f"Enumerator Performance  (k={k:,})", fontsize=14)
    ax.set_xlabel("n  (matrix dimension)", fontsize=12)
    ax.set_ylabel("Time (seconds)", fontsize=12)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.legend(fontsize=10)
    ax.grid(True, which="both", linestyle="--", alpha=0.4)
    fig.tight_layout()

    out_path = os.path.join(OUTPUT_DIR, f"benchmark_k{k}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")

print("Done.")
