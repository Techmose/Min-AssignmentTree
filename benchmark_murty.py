#!/usr/bin/env python3
"""
benchmark_murty.py -- times FastCacheEnumerator (Java) vs fastmurty and plots time vs n.

  Plot 1: n = 5..10,    k = n!
  Plot 2: n = 100..110, k = 1000

For each n, the Java run (a separate JVM process) and the fastmurty run
(in this Python process) execute CONCURRENTLY on different cores, so each
n costs max(java, fastmurty) wall time instead of the sum. Pass
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

JAVA_CLASS = "FastCacheEnumerator, JavaFastmurty, MinimalCacheEnumerator, ExplicitFastGraphEnumerator"


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


def benchmark(ns, k_of_n, build_dir, fm, args, title):
    java_t, fm_t = [], []
    for n in ns:
        k = k_of_n(n)
        rng = np.random.default_rng(args.seed + n)
        C = rng.integers(args.min_cost, args.max_cost + 1, size=(n, n), dtype=np.int64)
        matrix_path = os.path.join(build_dir, f"matrix_{n}.txt")
        np.savetxt(matrix_path, C, fmt="%d")
        print(f"[{title}] n={n} k={k}", flush=True)
        wall0 = time.monotonic()

        # Java starts first, in its own process / on its own core(s)
        proc = start_java(build_dir, matrix_path, k, args.repeats)
        deadline = time.monotonic() + args.timeout
        if args.sequential:
            java_result = collect_java(proc, deadline, args.timeout)

        # fastmurty runs here while the JVM is busy
        ft, fcosts = float("nan"), None
        try:
            sols, times, _ = run_fastmurty(fm, C, k, args.repeats)
            ft = min(times)
            fcosts = [c for c, _ in sols]
            fm_msg = f"{ft:.4f} s"
        except (Exception, SystemExit) as e:
            fm_msg = f"failed ({e})"

        if not args.sequential:
            java_result = collect_java(proc, deadline, args.timeout)
        jt, jcosts, java_msg = java_result

        print(f"    java      : {java_msg}")
        print(f"    fastmurty : {fm_msg}")
        print(f"    wall      : {time.monotonic() - wall0:.4f} s")

        if jcosts is not None and fcosts is not None:
            print("    costs match" if jcosts == fcosts else "    WARNING: cost lists differ")

        java_t.append(jt)
        fm_t.append(ft)
    return java_t, fm_t


def plot(ns, java_t, fm_t, title, path, log_y):
    plt.figure(figsize=(8, 5))
    plt.plot(ns, java_t, "o-", label=JAVA_CLASS + " (Java)")
    plt.plot(ns, fm_t, "s-", label="fastmurty (C)")
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


def save_csv(path, ns, ks, java_t, fm_t):
    with open(path, "w") as f:
        f.write("n,k,java_s,fastmurty_s\n")
        for n, k, j, m in zip(ns, ks, java_t, fm_t):
            f.write(f"{n},{k},{j},{m}\n")
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
                   help="run Java and fastmurty one after the other (old behaviour)")
    args = p.parse_args()

    fm = load_fastmurty(args.fastmurty)
    os.makedirs(args.out, exist_ok=True)

    with tempfile.TemporaryDirectory() as build_dir:
        build_java(args.java_src, build_dir, JAVA_CLASS)

        # Plot 1: n = 5..10, k = n!
        ns1 = list(range(5, 12))
        j1, f1 = benchmark(ns1, math.factorial, build_dir, fm, args, "small")
        save_csv(os.path.join(args.out, "timing_small.csv"), ns1, [math.factorial(n) for n in ns1], j1, f1)
        plot(ns1, j1, f1, "Time vs n  (k = n!)", os.path.join(args.out, "timing_small.png"), log_y=True)

        # Plot 2: n = 100..110, k = 1000
        ns2 = list(range(100, 111))
        j2, f2 = benchmark(ns2, lambda n: 1000, build_dir, fm, args, "large")
        save_csv(os.path.join(args.out, "timing_large.csv"), ns2, [1000] * len(ns2), j2, f2)
        plot(ns2, j2, f2, "Time vs n  (k = 1000)", os.path.join(args.out, "timing_large.png"), log_y=False)


if __name__ == "__main__":
    main()
