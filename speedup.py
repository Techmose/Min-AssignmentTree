"""
speedup.py

Computes the min, max, and average speedup of OG_FTFF over OG_FFFF
using benchmark_results.csv.

Speedup = OG_FFFF time / OG_FTFF time  (> 1 means FTFF is faster)

Usage:
    python speedup.py [csv_file]
"""

import sys
import csv
from collections import defaultdict

CSV_FILE = sys.argv[1] if len(sys.argv) > 1 else "benchmark_results.csv"

# Load: (n, k) -> config -> time_s
times = defaultdict(dict)

with open(CSV_FILE, newline="") as f:
    for row in csv.DictReader(f):
        key = (int(row["n"]), int(row["k"]))
        times[key][row["config"].strip()] = float(row["time_s"])

# Compute speedup for every (n, k) pair that has both configs
speedups = []

for (n, k), configs in sorted(times.items()):
    if "OG_FFFF" in configs and "OG_TFFF" in configs:
        speedup = configs["OG_FFFF"] / configs["OG_TFFF"]
        speedups.append((n, k, configs["OG_FFFF"], configs["OG_TFFF"], speedup))
        print(f"  n={n:<4} k={k:<8} FFFF={configs['OG_FFFF']:.4f}s  TFFF={configs['OG_TFFF']:.4f}s  speedup={speedup:.3f}x")

if not speedups:
    print("No matching rows found. Check that OG_FFFF and OG_TFFF both appear in the CSV.")
    sys.exit(1)

values = [s[4] for s in speedups]
print(f"\nSpeedup of OG_TFFF over OG_FFFF across {len(values)} runs:")
print(f"  min:  {min(values):.3f}x")
print(f"  max:  {max(values):.3f}x")
print(f"  avg:  {sum(values) / len(values):.3f}x")
