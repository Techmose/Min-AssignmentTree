import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import re

# ── Config ────────────────────────────────────────────────────────────────────

STACKS_CSV = "jfr_stacks.csv"
OUTPUT_DIR = "jfr_graphs"
NUM_BUCKETS = 10

# Methods to track in graph 2 — case-insensitive substring match
COMPARISON_PATTERNS = ["hungarian"]
SUM_PATTERNS        = ["bitset", "hashmap", "priorityqueue"]

# How many methods to show individually in graph 1 before collapsing to "other"
TOP_N_METHODS = 8

# ── Helpers ───────────────────────────────────────────────────────────────────

def short_name(cls, method):
    """Strip package prefix, return ClassName.method"""
    simple_cls = cls.split(".")[-1] if "." in cls else cls
    return f"{simple_cls}.{method}"

def matches_any(label, patterns):
    label_lower = label.lower()
    return any(p.lower() in label_lower for p in patterns)

# ── Load data ─────────────────────────────────────────────────────────────────

if not os.path.exists(STACKS_CSV):
    raise FileNotFoundError(f"Could not find {STACKS_CSV} — run JfrExtract first.")

df = pd.read_csv(STACKS_CSV)
df["label"] = df.apply(lambda r: short_name(r["class"], r["method"]), axis=1)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── One pair of graphs per (algorithm, n, k) ──────────────────────────────────

groups = df.groupby(["algorithm", "n", "k"])

for (algorithm, n, k), group in groups:

    safe_name = re.sub(r"[^\w]", "_", f"{algorithm}_n{n}_k{k}")
    print(f"Generating graphs for: {algorithm} n={n} k={k}")

    # Ensure all buckets 0–9 are represented
    buckets = list(range(NUM_BUCKETS))

    # ── GRAPH 1: Stacked bar chart — method breakdown per bucket ──────────────

    # Find top N methods by total sample count across all buckets
    method_totals = group.groupby("label")["sample_count"].sum().sort_values(ascending=False)
    top_methods   = list(method_totals.head(TOP_N_METHODS).index)

    # Build a pivot: bucket_index × label → pct_of_bucket
    # Methods outside top N get collapsed into "other"
    rows = []
    for bucket_idx, bgroup in group.groupby("bucket_index"):
        row = {"bucket": bucket_idx}
        other_pct = 0.0
        for _, r in bgroup.iterrows():
            if r["label"] in top_methods:
                row[r["label"]] = row.get(r["label"], 0.0) + r["pct_of_bucket"]
            else:
                other_pct += r["pct_of_bucket"]
        row["other"] = other_pct
        rows.append(row)

    pivot = pd.DataFrame(rows).set_index("bucket").reindex(buckets).fillna(0)

    # Column order: top methods descending, then other
    col_order = [m for m in top_methods if m in pivot.columns] + \
                (["other"] if "other" in pivot.columns else [])
    pivot = pivot[col_order]

    # Color palette
    cmap   = plt.get_cmap("tab20")
    colors = [cmap(i) for i in range(len(col_order) - 1)] + ["#cccccc"]

    fig1, ax1 = plt.subplots(figsize=(13, 6))

    bottoms = np.zeros(NUM_BUCKETS)
    for col, color in zip(col_order, colors):
        vals = pivot[col].values
        ax1.bar(buckets, vals, bottom=bottoms, color=color, label=col, width=0.75, edgecolor="white", linewidth=0.4)
        bottoms += vals

    ax1.set_title(f"{algorithm}  n={n}  k={k} — Method Breakdown per Bucket", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Bucket (0 = start → 9 = end of run)", fontsize=11)
    ax1.set_ylabel("% of CPU samples in bucket", fontsize=11)
    ax1.set_xticks(buckets)
    ax1.set_xticklabels([f"B{i}" for i in buckets])
    ax1.set_ylim(0, 105)
    ax1.legend(loc="upper right", fontsize=8, framealpha=0.85,
               ncol=2, title="Method", title_fontsize=9)
    ax1.grid(axis="y", linestyle="--", alpha=0.4)

    fig1.tight_layout()
    path1 = os.path.join(OUTPUT_DIR, f"{safe_name}_breakdown.png")
    fig1.savefig(path1, dpi=150)
    plt.close(fig1)
    print(f"  Saved: {path1}")

    # ── GRAPH 2: Hungarian vs sum-of-comparison methods ───────────────────────

    hungarian_pct = []
    sum_pct       = []

    for bucket_idx in buckets:
        bdf = group[group["bucket_index"] == bucket_idx]

        h = bdf[bdf["label"].apply(lambda l: matches_any(l, COMPARISON_PATTERNS))]["pct_of_bucket"].sum()
        s = bdf[bdf["label"].apply(lambda l: matches_any(l, SUM_PATTERNS))]["pct_of_bucket"].sum()

        hungarian_pct.append(h)
        sum_pct.append(s)

    fig2, ax2 = plt.subplots(figsize=(13, 5))

    x = np.arange(NUM_BUCKETS)
    w = 0.35

    bars_h = ax2.bar(x - w/2, hungarian_pct, width=w, color="#4C72B0", label="Hungarian")
    bars_s = ax2.bar(x + w/2, sum_pct,       width=w, color="#DD8452", label="BitSet + HashMap + PriorityQueue")

    # Value labels on bars
    for bar in bars_h:
        h = bar.get_height()
        if h > 1:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=7, color="#4C72B0")

    for bar in bars_s:
        h = bar.get_height()
        if h > 1:
            ax2.text(bar.get_x() + bar.get_width()/2, h + 0.5, f"{h:.1f}%",
                     ha="center", va="bottom", fontsize=7, color="#DD8452")

    ax2.set_title(f"{algorithm}  n={n}  k={k} — Hungarian vs Data Structure Methods Over Time",
                  fontsize=13, fontweight="bold")
    ax2.set_xlabel("Bucket (0 = start → 9 = end of run)", fontsize=11)
    ax2.set_ylabel("% of CPU samples in bucket", fontsize=11)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"B{i}" for i in buckets])
    ax2.legend(fontsize=10)
    ax2.grid(axis="y", linestyle="--", alpha=0.4)

    fig2.tight_layout()
    path2 = os.path.join(OUTPUT_DIR, f"{safe_name}_hungarian_vs_ds.png")
    fig2.savefig(path2, dpi=150)
    plt.close(fig2)
    print(f"  Saved: {path2}")

print(f"\nDone. All graphs written to ./{OUTPUT_DIR}/")
