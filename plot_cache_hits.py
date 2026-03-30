import pandas as pd
import matplotlib.pyplot as plt
import os

def plot_cache_hits(csv_file, output_dir):
    df = pd.read_csv(csv_file, low_memory=False, on_bad_lines="skip")

    # Convert columns to numeric (force)
    cols = ["k", "depth", "cost", "hit_count"]

    for col in cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Group by depth
    grouped = df.groupby("depth")

    for depth, group in grouped:

        # Normalize cost for colormap
        min_cost = group["cost"].min()
        max_cost = group["cost"].max()

        plt.figure()

        # Scatter plot
        scatter = plt.scatter(
            group["k"],
            group["hit_count"],
            c=group["cost"],
            cmap="viridis",  # blue → red
            vmin=min_cost,
            vmax=max_cost,
            s=10,
            alpha=0.7
        )

        # Labels and title
        plt.xlabel("k")
        plt.ylabel("Cache Hit Count")
        plt.title(f"Cache Hits vs k (Depth {depth})")

        # Colorbar
        cbar = plt.colorbar(scatter)
        cbar.set_label(f"Cost (min={min_cost}, max={max_cost})")

        # Save plot
        filename = os.path.join(output_dir, f"depth_{depth}.png")
        plt.savefig(filename, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"Min: {min_cost} Max: {max_cost}")
        print(f"Saved: {filename}")


file = "cache_hits_10X10_3628800.csv"

plot_cache_hits(file, "10x10_graphs")