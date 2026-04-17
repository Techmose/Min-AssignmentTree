"""
graph_jfr.py
Reads CSVs produced by JfrExtract.java and generates comparison plots.

Usage:
    pip install pandas matplotlib
    python graph_jfr.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import os, sys

ALG_CSV   = "jfr_algorithms.csv"
CPU_CSV   = "jfr_cpu.csv"
STACK_CSV = "jfr_stacks.csv"

COLORS = {
    "OG_FFFF": "#e15759",
    "OG_TFFF": "#4e79a7",
    "MURTY":   "#59a14f",
}

# ── Loaders ──────────────────────────────────────────────────────────────────

def load(path, required=True):
    if not os.path.exists(path):
        if required:
            sys.exit(f"Missing {path} — run JfrExtract.java first.")
        print(f"No {path} found — skipping related plot.")
        return None
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    print(f"Loaded {len(df)} rows from {path}")
    return df


# ── Plot 1: Duration by N ─────────────────────────────────────────────────────

def plot_duration_by_n(df):
    ks = sorted(df["k"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(7 * len(ks), 5), sharey=False)
    if len(ks) == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        sub = df[df["k"] == k]
        for algo, grp in sub.groupby("algorithm"):
            g = grp.sort_values("n")
            ax.plot(g["n"], g["duration_ms"], marker="o",
                    label=algo, color=COLORS.get(algo), linewidth=2)
        ax.set_title(f"k = {k:,}", fontsize=13)
        ax.set_xlabel("Matrix size (n)")
        ax.set_ylabel("Duration (ms)")
        ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
        ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    fig.suptitle("Algorithm Duration vs Matrix Size", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot_duration_by_n.png", dpi=150)
    print("Saved: plot_duration_by_n.png")
    plt.show()


# ── Plot 2: Speedup vs Murty ──────────────────────────────────────────────────

def plot_speedup(df):
    ks = sorted(df["k"].unique())
    fig, axes = plt.subplots(1, len(ks), figsize=(7 * len(ks), 5), sharey=True)
    if len(ks) == 1:
        axes = [axes]

    for ax, k in zip(axes, ks):
        sub = df[df["k"] == k]
        murty = (sub[sub["algorithm"] == "MURTY"][["n", "duration_ms"]]
                 .rename(columns={"duration_ms": "murty_ms"}))
        merged = sub[sub["algorithm"] != "MURTY"].merge(murty, on="n")
        merged["speedup"] = merged["murty_ms"] / merged["duration_ms"]

        for algo, grp in merged.groupby("algorithm"):
            g = grp.sort_values("n")
            ax.plot(g["n"], g["speedup"], marker="s",
                    label=algo, color=COLORS.get(algo), linewidth=2)

        ax.axhline(1.0, color="gray", linestyle="--", linewidth=1, label="Murty baseline")
        ax.set_title(f"k = {k:,}", fontsize=13)
        ax.set_xlabel("Matrix size (n)")
        ax.set_ylabel("Speedup over Murty (×)")
        ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)

    fig.suptitle("Speedup vs Murty", fontsize=15, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot_speedup.png", dpi=150)
    print("Saved: plot_speedup.png")
    plt.show()


# ── Plot 3: CPU load ──────────────────────────────────────────────────────────

def plot_cpu(df):
    if df is None:
        return
    df = df.copy()
    df["timestamp_s"] = (df["timestamp_ms"] - df["timestamp_ms"].min()) / 1000.0
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(df["timestamp_s"], df["machineTotal"] * 100,
            label="Machine total %", color="#aaa", linewidth=1)
    ax.plot(df["timestamp_s"], df["jvmUser"] * 100,
            label="JVM user %", color="#4e79a7", linewidth=1.5)
    ax.plot(df["timestamp_s"], df["jvmSystem"] * 100,
            label="JVM system %", color="#e15759", linewidth=1.5)
    ax.set_xlabel("Time (s)"); ax.set_ylabel("CPU %")
    ax.set_title("CPU Load Over Benchmark Run", fontsize=13, fontweight="bold")
    ax.legend(); ax.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig("plot_cpu.png", dpi=150)
    print("Saved: plot_cpu.png")
    plt.show()


# Methods to exclude — top-level entry points that aren't useful to see
EXCLUDE_METHODS = {
    ("BenchmarkJFR",        "main"),
    ("BenchmarkJFR",        "runOrderGraph"),
    ("BenchmarkJFR",        "runMurty"),
    ("OrderGraphEnumerator","enumerate"),
    ("MurtyEnumerator",     "enumerate"),
}

def _filter(df):
    mask = pd.Series([True] * len(df), index=df.index)
    for cls, mth in EXCLUDE_METHODS:
        mask &= ~((df["class"].str.endswith(cls)) & (df["method"] == mth))
    return df[mask]


# ── Plot 4: Top-N hotspot methods per algorithm (horizontal bar) ──────────────

def plot_hotspots(df, top_n=15):
    if df is None:
        return

    algos = df["algorithm"].unique()
    fig, axes = plt.subplots(1, len(algos),
                             figsize=(10 * len(algos), 6), sharey=False)
    if len(algos) == 1:
        axes = [axes]

    for ax, algo in zip(axes, sorted(algos)):
        sub = (_filter(df[df["algorithm"] == algo])
               .groupby(["class", "method"], as_index=False)["sample_count"]
               .sum())

        total = sub["sample_count"].sum()
        sub["pct"] = 100.0 * sub["sample_count"] / total
        sub["label"] = sub["class"].str.split(".").str[-1] + "." + sub["method"]
        top = sub.nlargest(top_n, "pct").sort_values("pct")

        ax.barh(top["label"], top["pct"],
                color=COLORS.get(algo, "#888"), alpha=0.85)
        ax.set_xlabel("% of execution samples")
        ax.set_title(f"{algo} — top {top_n} methods", fontsize=13, fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

        # Annotate bars with percentage
        for _, row in top.iterrows():
            ax.text(row["pct"] + 0.3, row["label"],
                    f"{row['pct']:.1f}%", va="center", fontsize=8)

    fig.suptitle("Method Hotspots by Algorithm (all n & k combined)",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("plot_hotspots.png", dpi=150)
    print("Saved: plot_hotspots.png")
    plt.show()


# ── Plot 5: Hotspots per algorithm AND per n ──────────────────────────────────

def plot_hotspots_by_n(df, algo, top_n=10):
    if df is None:
        return
    sub = df[df["algorithm"] == algo]
    ns  = sorted(sub["n"].unique())

    fig, axes = plt.subplots(1, len(ns),
                             figsize=(8 * len(ns), 5), sharey=False)
    if len(ns) == 1:
        axes = [axes]

    for ax, n in zip(axes, ns):
        s = (_filter(sub[sub["n"] == n])
             .groupby(["class", "method"], as_index=False)["sample_count"].sum())
        total = s["sample_count"].sum()
        s["pct"]   = 100.0 * s["sample_count"] / total
        s["label"] = s["class"].str.split(".").str[-1] + "." + s["method"]
        top = s.nlargest(top_n, "pct").sort_values("pct")

        ax.barh(top["label"], top["pct"],
                color=COLORS.get(algo, "#888"), alpha=0.85)
        ax.set_xlabel("% samples")
        ax.set_title(f"{algo}  n={n}", fontsize=12, fontweight="bold")
        ax.grid(True, axis="x", linestyle="--", alpha=0.5)

    fig.suptitle(f"Method Hotspots — {algo} — by matrix size",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    out = f"plot_hotspots_{algo}_by_n.png"
    plt.savefig(out, dpi=150)
    print(f"Saved: {out}")
    plt.show()


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df_alg   = load(ALG_CSV)
    df_cpu   = load(CPU_CSV,   required=False)
    df_stack = load(STACK_CSV, required=False)

    plot_duration_by_n(df_alg)
    plot_speedup(df_alg)
    plot_cpu(df_cpu)
    plot_hotspots(df_stack)

    # Per-algorithm breakdown by n — edit this list to taste
    if df_stack is not None:
        for algo in df_stack["algorithm"].unique():
            plot_hotspots_by_n(df_stack, algo)
