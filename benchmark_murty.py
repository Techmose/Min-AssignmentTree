#!/usr/bin/env python3
"""
benchmark_murty.py -- times JavaFastmurty and FastCacheEnumerator (Java)
against fastmurty (C) and plots time vs n.

  Plot 1: n = 5..12,                          k = n!
  Plot 2: n = 10, 50, 100, 500, 1000, 3000    k = 1000 (k = 100 for n = 3000)

Put this next to compare_murty.py and your .java files, then run:
    python benchmark_murty.py --java-src . --fastmurty fastmurty

Any run that cannot work for a given n is skipped with the reason
printed, and shows as a gap in the plot:
  - FastCacheEnumerator stores row/column sets in 64-bit masks, so it
    is skipped for n > 64 (larger n would silently give wrong keys).
  - Java is skipped if AssignmentProblem.infinity (sum of all costs + 1)
    would overflow an int.
  - fastmurty is skipped if the shifted costs reach its internal
    infinity (1e9); see compare_murty.py for why the shift is needed.
"""

import argparse
import math
import os
import subprocess
import tempfile

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from compare_murty import build_java, load_fastmurty, run_fastmurty

JAVA_CLASSES = ["JavaFastmurty", "FastCacheEnumerator"]
MAX_N = {"FastCacheEnumerator": 64}   # 64-bit masks
INT_MAX = 2**31 - 1
FASTMURTY_INF = 1e9


def run_java_timed(build_dir, matrix_path, k, repeats, timeout):
    cmd = ["java", "-cp", build_dir, "CompareDriver", matrix_path, str(k), str(repeats)]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(res.stderr.strip().splitlines()[-1] if res.stderr.strip() else "java failed")
    times, costs = [], []
    for line in res.stdout.splitlines():
        if line.startswith("TIME "):
            times.append(int(line.split()[1]) * 1e-9)
        elif line.startswith("SOL "):
            costs.append(int(line.split()[1]))
    return min(times), costs


def java_skip_reason(cls, n, C):
    if n > MAX_N.get(cls, n):
        return f"n > {MAX_N[cls]} (64-bit masks)"
    if int(C.sum()) + 1 > INT_MAX:
        return "sum of costs overflows int (AssignmentProblem.infinity)"
    return None


def fastmurty_skip_reason(n, C):
    S = (n + 1) * int(C.max()) - n * int(C.min()) + 1
    if n * S >= FASTMURTY_INF:
        return f"shifted costs (~{n * S:.1e}) exceed fastmurty's internal inf (1e9)"
    return None


def benchmark(ns, k_of_n, build_dirs, fm, args, title):
    times = {name: [] for name in JAVA_CLASSES + ["fastmurty"]}
    for n in ns:
        k = k_of_n(n)
        rng = np.random.default_rng(args.seed + n)
        C = rng.integers(args.min_cost, args.max_cost + 1, size=(n, n), dtype=np.int64)
        print(f"[{title}] n={n} k={k}", flush=True)

        costs = {}
        matrix_path = None
        for cls in JAVA_CLASSES:
            t = float("nan")
            reason = java_skip_reason(cls, n, C)
            if reason:
                print(f"    {cls:20s}: skipped ({reason})")
            else:
                if matrix_path is None:
                    matrix_path = os.path.join(build_dirs[cls], f"matrix_{n}.txt")
                    np.savetxt(matrix_path, C, fmt="%d")
                try:
                    t, costs[cls] = run_java_timed(build_dirs[cls], matrix_path, k, args.repeats, args.timeout)
                    print(f"    {cls:20s}: {t:.4f} s")
                except subprocess.TimeoutExpired:
                    print(f"    {cls:20s}: timed out after {args.timeout} s")
                except Exception as e:
                    print(f"    {cls:20s}: failed ({e})")
            times[cls].append(t)

        t = float("nan")
        reason = fastmurty_skip_reason(n, C)
        if reason:
            print(f"    {'fastmurty':20s}: skipped ({reason})")
        else:
            try:
                sols, ftimes, _ = run_fastmurty(fm, C, k, args.repeats)
                t = min(ftimes)
                costs["fastmurty"] = [c for c, _ in sols]
                print(f"    {'fastmurty':20s}: {t:.4f} s")
            except (Exception, SystemExit) as e:
                print(f"    {'fastmurty':20s}: failed ({e})")
        times["fastmurty"].append(t)

        if matrix_path and os.path.exists(matrix_path):
            os.remove(matrix_path)

        if len(costs) > 1:
            ref_name, ref = next(iter(costs.items()))
            bad = [name for name, c in costs.items() if c != ref]
            print("    costs match" if not bad else f"    WARNING: cost lists differ from {ref_name}: {bad}")
    return times


def plot(ns, times, title, path, log_x):
    styles = {"JavaFastmurty": "o-", "FastCacheEnumerator": "^-", "fastmurty": "s-"}
    labels = {"JavaFastmurty": "JavaFastmurty (Java)",
              "FastCacheEnumerator": "FastCacheEnumerator (Java)",
              "fastmurty": "fastmurty (C)"}
    plt.figure(figsize=(8, 5))
    for name, ts in times.items():
        if any(not math.isnan(t) for t in ts):
            plt.plot(ns, ts, styles[name], label=labels[name])
    plt.xlabel("n")
    plt.ylabel("enumeration time (s)")
    plt.yscale("log")
    if log_x:
        plt.xscale("log")
    plt.xticks(ns, [str(n) for n in ns])
    plt.minorticks_off()
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"saved {path}")


def save_csv(path, ns, ks, times):
    names = list(times)
    with open(path, "w") as f:
        f.write("n,k," + ",".join(f"{name}_s" for name in names) + "\n")
        for idx, (n, k) in enumerate(zip(ns, ks)):
            f.write(f"{n},{k}," + ",".join(str(times[name][idx]) for name in names) + "\n")
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
    args = p.parse_args()

    fm = load_fastmurty(args.fastmurty)
    os.makedirs(args.out, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        # one build folder per class, since each gets its own CompareDriver
        build_dirs = {}
        for cls in JAVA_CLASSES:
            build_dirs[cls] = os.path.join(tmp, cls)
            os.makedirs(build_dirs[cls])
            build_java(args.java_src, build_dirs[cls], cls)

        # Plot 1: n = 5..12, k = n!
        ns1 = list(range(5, 13))
        ks1 = [math.factorial(n) for n in ns1]
        t1 = benchmark(ns1, math.factorial, build_dirs, fm, args, "small")
        save_csv(os.path.join(args.out, "timing_small.csv"), ns1, ks1, t1)
        plot(ns1, t1, "Time vs n  (k = n!)", os.path.join(args.out, "timing_small.png"), log_x=False)

        # Plot 2: n = 10 .. 3000, k = 1000 (100 for n = 3000)
        ns2 = [10, 50, 100, 500, 1000, 3000]
        k2 = lambda n: 100 if n >= 3000 else 1000
        t2 = benchmark(ns2, k2, build_dirs, fm, args, "large")
        save_csv(os.path.join(args.out, "timing_large.csv"), ns2, [k2(n) for n in ns2], t2)
        plot(ns2, t2, "Time vs n  (k = 1000; k = 100 at n = 3000)",
             os.path.join(args.out, "timing_large.png"), log_x=True)


if __name__ == "__main__":
    main()
