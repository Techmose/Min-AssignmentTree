#!/usr/bin/env python3
"""
benchmark_murty.py -- times several Java enumerators vs fastmurty and plots time vs n.

  Plot 1: n = 5..10,    k = n!
  Plot 2: n = 100..110, k = 1000

For each n, every Java class in JAVA_CLASSES (each its own JVM process)
and the fastmurty run (in this Python process) execute CONCURRENTLY on
different cores, so each n costs the slowest run's time instead of the sum. Pass
--sequential to get the old one-after-the-other behaviour (cleanest
timings, since the two runs can't compete for cache / memory bandwidth).

Put this next to compare_murty.py and your .java files, then run:
    python benchmark_murty.py --java-src . --fastmurty "fast murty"
"""

import argparse
import math
import os
import subprocess
import tempfile
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_murty import build_java, load_fastmurty, run_fastmurty

# Exact Java class names (must match `public class ...` in each .java file).
# Each one is built into its own folder with its own CompareDriver.
JAVA_CLASSES = [
    "FastCacheEnumerator",
    "JavaFastmurty",
    "MinimalCacheEnumerator",
    "ExplicitFastGraph",
]


def start_java(build_dir, matrix_path, k, repeats):
    """Launch the Java run in the background and return immediately."""
    cmd = ["java", "-cp", build_dir, "CompareDriver", matrix_path, str(k), str(repeats)]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


def finish_java(proc, deadline):
    """Wait for a Java run started by start_java; same results as the old run_java_timed."""
    remaining = max(0.0, deadline - time.monotonic())
    try:
        out, err = proc.communicate(timeout=remaining)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.communicate()
        raise
    if proc.returncode != 0:
        raise RuntimeError(err.strip().splitlines()[-1] if err.strip() else "java failed")
    times, costs = [], []
    for line in out.splitlines():
        if line.startswith("TIME "):
            times.append(int(line.split()[1]) * 1e-9)
        elif line.startswith("SOL "):
            costs.append(int(line.split()[1]))
    return min(times), costs


def collect_java(proc, deadline, timeout):
    """finish_java, but turns failures into (nan, None) plus a message, like before."""
    try:
        jt, jcosts = finish_java(proc, deadline)
        return jt, jcosts, f"{jt:.4f} s"
    except subprocess.TimeoutExpired:
        return float("nan"), None, f"timed out after {timeout} s"
    except Exception as e:
        return float("nan"), None, f"failed ({e})"


def benchmark(ns, k_of_n, build_dirs, matrix_dir, fm, args, title):
    """Returns {name: [time per n]} for every Java class plus "fastmurty"."""
    results = {name: [] for name in list(build_dirs) + ["fastmurty"]}
    for n in ns:
        k = k_of_n(n)
        rng = np.random.default_rng(args.seed + n)
        C = rng.integers(args.min_cost, args.max_cost + 1, size=(n, n), dtype=np.int64)
        matrix_path = os.path.join(matrix_dir, f"matrix_{n}.txt")
        np.savetxt(matrix_path, C, fmt="%d")
        print(f"[{title}] n={n} k={k}", flush=True)
        wall0 = time.monotonic()

        # start every Java class at once (or one at a time with --sequential)
        java_results = {}
        procs = {}
        for cls, bdir in build_dirs.items():
            proc = start_java(bdir, matrix_path, k, args.repeats)
            deadline = time.monotonic() + args.timeout
            if args.sequential:
                java_results[cls] = collect_java(proc, deadline, args.timeout)
            else:
                procs[cls] = (proc, deadline)

        # fastmurty runs here while the JVMs are busy
        ft, fcosts = float("nan"), None
        try:
            sols, times, _ = run_fastmurty(fm, C, k, args.repeats)
            ft = min(times)
            fcosts = [c for c, _ in sols]
            fm_msg = f"{ft:.4f} s"
        except (Exception, SystemExit) as e:
            fm_msg = f"failed ({e})"

        for cls, (proc, deadline) in procs.items():
            java_results[cls] = collect_java(proc, deadline, args.timeout)

        width = max(len(name) for name in results)
        for cls, (jt, jcosts, msg) in java_results.items():
            check = ""
            if jcosts is not None and fcosts is not None:
                check = "  (costs match)" if jcosts == fcosts else "  WARNING: costs differ from fastmurty"
            print(f"    {cls:<{width}} : {msg}{check}")
            results[cls].append(jt)
        print(f"    {'fastmurty':<{width}} : {fm_msg}")
        print(f"    {'wall':<{width}} : {time.monotonic() - wall0:.4f} s")
        results["fastmurty"].append(ft)
    return results


def plot(ns, results, title, path, log_y):
    plt.figure(figsize=(8, 5))
    markers = "os^Dvx*+"
    for i, (name, times) in enumerate(results.items()):
        label = "fastmurty (C)" if name == "fastmurty" else f"{name} (Java)"
        plt.plot(ns, times, markers[i % len(markers)] + "-", label=label)
    plt.xlabel("n")
    plt.ylabel("enumeration time (s)")
    if log_y:
        plt.yscale("log")
    plt.xticks(ns)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")


def save_csv(path, ns, ks, results):
    names = list(results)
    with open(path, "w") as f:
        f.write("n,k," + ",".join(f"{name}_s" for name in names) + "\n")
        for i, (n, k) in enumerate(zip(ns, ks)):
            f.write(f"{n},{k}," + ",".join(str(results[name][i]) for name in names) + "\n")
    print(f"saved {path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--java-src", default=".")
    p.add_argument("--fastmurty", default="fastmurty")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-cost", type=int, default=1)
    p.add_argument("--max-cost", type=int, default=100)
    p.add_argument("--timeout", type=int, default=3600, help="seconds per Java run")
    p.add_argument("--out", default=".", help="folder for plots and CSVs")
    p.add_argument("--sequential", action="store_true",
                   help="run the Java classes one after another (fastmurty after them)")
    args = p.parse_args()

    fm = load_fastmurty(args.fastmurty)
    os.makedirs(args.out, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # one build folder per class, so each gets its own CompareDriver
        build_dirs = {}
        for cls in JAVA_CLASSES:
            bdir = os.path.join(tmp, cls)
            os.makedirs(bdir)
            build_java(args.java_src, bdir, cls)
            build_dirs[cls] = bdir

        # Plot 1: n = 5..11, k = n!
        ns1 = list(range(5, 12))
        r1 = benchmark(ns1, math.factorial, build_dirs, tmp, fm, args, "small")
        save_csv(os.path.join(args.out, "timing_small.csv"), ns1, [math.factorial(n) for n in ns1], r1)
        plot(ns1, r1, "Time vs n  (k = n!)", os.path.join(args.out, "timing_small.png"), log_y=True)

        # Plot 2: n = 100..110, k = 1000
        ns2 = list(range(100, 111))
        r2 = benchmark(ns2, lambda n: 1000, build_dirs, tmp, fm, args, "large")
        save_csv(os.path.join(args.out, "timing_large.csv"), ns2, [1000] * len(ns2), r2)
        plot(ns2, r2, "Time vs n  (k = 1000)", os.path.join(args.out, "timing_large.png"), log_y=False)

if __name__ == "__main__":
    main()
