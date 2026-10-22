#!/usr/bin/env python3
"""
compare_murty.py -- run our Java Murty enumerator and fastmurty
(github.com/motrom/fastmurty) on the same cost matrix and compare the
top-k solutions and timings.

Typical use:
    # build fastmurty once (inside the repo):  make
    python3 compare_murty.py --java-src ./java --fastmurty ./fastmurty --n 6
    python3 compare_murty.py --java-src ./java --fastmurty ./fastmurty --n 12 --k 500 --repeats 5
    python3 compare_murty.py --java-src ./java --fastmurty ./fastmurty --matrix my_matrix.txt --k 100

How the problems are made equivalent
------------------------------------
fastmurty solves *data association*: every row may also go unmatched at
cost 0, so it only prefers a match when its cost is negative. Our Java
code solves the classic square assignment problem (every row matched).
To make fastmurty behave like the classic problem we subtract a shift S
from every cell, with S large enough that every full matching is cheaper
than every partial one:

    S = (n+1)*max(C) - n*min(C) + 1

so the k best associations are exactly the k best full matchings, and
    classic_cost = fastmurty_cost + n*S.
"""

import argparse
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

import numpy as np

# ----------------------------------------------------------------------
# Java side
# ----------------------------------------------------------------------

JAVA_DRIVER = r"""
import java.io.*;
import java.util.*;

public class CompareDriver {
    public static void main(String[] args) throws Exception {
        String path = args[0];
        int k = Integer.parseInt(args[1]);
        int repeats = Integer.parseInt(args[2]);

        List<int[]> rows = new ArrayList<>();
        try (BufferedReader br = new BufferedReader(new FileReader(path))) {
            String line;
            while ((line = br.readLine()) != null) {
                line = line.trim();
                if (line.isEmpty()) continue;
                String[] tok = line.split("\\s+");
                int[] r = new int[tok.length];
                for (int j = 0; j < tok.length; j++) r[j] = Integer.parseInt(tok[j]);
                rows.add(r);
            }
        }
        int[][] matrix = rows.toArray(new int[0][]);

        List<AssignmentSolution> sols = null;
        for (int rep = 0; rep < repeats; rep++) {
            AssignmentProblem problem = new AssignmentProblem(matrix);
            __CLASS__ e = new __CLASS__(problem);
            long t0 = System.nanoTime();
            sols = e.enumerate(k);
            long t1 = System.nanoTime();
            System.out.printf("TIME %d%n", t1 - t0);
        }
        for (AssignmentSolution s : sols) {
            StringBuilder sb = new StringBuilder("SOL ").append(s.cost);
            for (int a : s.assignment) sb.append(' ').append(a);
            System.out.println(sb);
        }
    }
}
"""


def build_java(java_src, build_dir, java_class):
    driver_path = os.path.join(build_dir, "CompareDriver.java")
    with open(driver_path, "w") as f:
        f.write(JAVA_DRIVER.replace("__CLASS__", java_class))
    sources = [os.path.join(java_src, f) for f in os.listdir(java_src) if f.endswith(".java")]
    if not sources:
        sys.exit(f"No .java files found in {java_src}")
    cmd = ["javac", "-d", build_dir] + sources + [driver_path]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        sys.exit("javac failed:\n" + res.stderr)


def run_java(build_dir, matrix_path, k, repeats, java_opts):
    cmd = ["java"] + java_opts + ["-cp", build_dir, "CompareDriver", matrix_path, str(k), str(repeats)]
    t0 = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.perf_counter() - t0
    if res.returncode != 0:
        sys.exit("Java run failed:\n" + res.stderr + res.stdout)

    times, sols = [], []
    for line in res.stdout.splitlines():
        if line.startswith("TIME "):
            times.append(int(line.split()[1]) * 1e-9)
        elif line.startswith("SOL "):
            parts = list(map(int, line.split()[1:]))
            sols.append((parts[0], tuple(parts[1:])))
    return sols, times, wall


# ----------------------------------------------------------------------
# fastmurty side
# ----------------------------------------------------------------------

def load_fastmurty(repo_arg):
    """
    Imports mhtdaClink from the fastmurty folder. mhtdaClink does
    CDLL("./mhtda.so"); on Windows (Python 3.8+) ctypes refuses relative
    DLL paths, so we temporarily redirect that call to the absolute path.
    """
    # accept the folder relative to the current directory OR to this script
    candidates = [os.path.abspath(repo_arg),
                  os.path.join(os.path.dirname(os.path.abspath(__file__)), repo_arg)]
    repo = next((c for c in candidates if os.path.isdir(c)), None)
    if repo is None:
        sys.exit("fastmurty folder not found. Looked in:\n  " + "\n  ".join(candidates))

    lib_path = os.path.join(repo, "mhtda.so")
    if not os.path.isfile(lib_path):
        similar = [f for f in os.listdir(repo) if f.lower().startswith("mhtda")]
        sys.exit(f"Could not find {lib_path}\n"
                 f"Files starting with 'mhtda' in that folder: {similar}\n"
                 "The compiled library must be named exactly mhtda.so.")

    import ctypes
    real_cdll = ctypes.CDLL

    def cdll_absolute(name, *a, **kw):
        if isinstance(name, str) and name.replace("\\", "/").startswith("./"):
            name = os.path.join(repo, name[2:])
            if os.name == "nt":
                kw.setdefault("winmode", 0)
        return real_cdll(name, *a, **kw)

    if os.name == "nt":
        for d in (r"C:\msys64\ucrt64\bin", r"C:\msys64\mingw64\bin"):
            if os.path.isdir(d):
                os.add_dll_directory(d)   # runtime DLLs, in case it wasn't built with -static

    old = os.getcwd()
    os.chdir(repo)
    sys.path.insert(0, repo)
    ctypes.CDLL = cdll_absolute
    try:
        import mhtdaClink
    except OSError as e:
        sys.exit(f"Found {lib_path} but could not load it:\n  {e}\n"
                 "Most likely causes: it was built without -static (rebuild with the gcc command "
                 "including -static), or gcc and Python differ in 32/64-bit.")
    finally:
        ctypes.CDLL = real_cdll
        os.chdir(old)
    return mhtdaClink


def run_fastmurty(fm, C, k, repeats):
    n = C.shape[0]
    cmax, cmin = int(C.max()), int(C.min())
    S = (n + 1) * cmax - n * cmin + 1
    if n * S >= 1e8:
        print(f"WARNING: shifted costs (~{n*S:.2e}) approach fastmurty's internal inf (1e9).")
    shifted = C.astype(np.float64) - S

    c_in = fm.sparsifyByRow(shifted, n) if fm.sparse else np.ascontiguousarray(shifted)

    row_priors = np.ones((1, n), dtype=np.bool_)
    col_priors = np.ones((1, n), dtype=np.bool_)
    row_w = np.zeros(1)
    col_w = np.zeros(1)

    times = []
    for _ in range(repeats):
        out_costs = np.zeros(k)
        out_assocs = np.zeros((k, 2 * n, 2), dtype=np.int32)
        workvars = fm.allocateWorkvarsforDA(n, n, k)
        t0 = time.perf_counter()
        fm.mhtda(c_in, row_priors, row_w, col_priors, col_w, out_assocs, out_costs, workvars)
        times.append(time.perf_counter() - t0)
        fm.deallocateWorkvarsforDA(workvars)

    sols = []
    for s in range(k):
        pairs = out_assocs[s]
        matches = pairs[(pairs[:, 0] >= 0) & (pairs[:, 1] >= 0)]
        if len(matches) != n:
            sys.exit(f"fastmurty solution {s} is not a full matching ({len(matches)}/{n} rows) "
                     f"-- shift too small?")
        assign = [0] * n
        for i, j in matches:
            assign[i] = int(j)
        cost = int(round(out_costs[s] + n * S))
        sols.append((cost, tuple(assign)))
    return sols, times, S


# ----------------------------------------------------------------------
# comparison
# ----------------------------------------------------------------------

def true_cost(C, assign):
    return int(sum(C[i, j] for i, j in enumerate(assign)))


def compare(C, java_sols, fm_sols, show):
    ok = True
    n = C.shape[0]

    for name, sols in (("java", java_sols), ("fastmurty", fm_sols)):
        for rank, (cost, a) in enumerate(sols):
            if sorted(a) != list(range(n)):
                print(f"  [{name}] rank {rank}: not a permutation: {a}")
                ok = False
            elif true_cost(C, a) != cost:
                print(f"  [{name}] rank {rank}: reported cost {cost} != recomputed {true_cost(C, a)}")
                ok = False
        if len(set(a for _, a in sols)) != len(sols):
            print(f"  [{name}] contains duplicate assignments")
            ok = False
        if any(sols[i][0] > sols[i + 1][0] for i in range(len(sols) - 1)):
            print(f"  [{name}] costs are not in non-decreasing order")
            ok = False

    if len(java_sols) != len(fm_sols):
        print(f"  different number of solutions: java={len(java_sols)} fastmurty={len(fm_sols)}")
        ok = False

    jc = [c for c, _ in java_sols]
    fc = [c for c, _ in fm_sols]
    if jc != fc:
        ok = False
        first = next((i for i, (a, b) in enumerate(zip(jc, fc)) if a != b), min(len(jc), len(fc)))
        print(f"  cost sequences differ, first at rank {first}: "
              f"java={jc[first] if first < len(jc) else '-'} fastmurty={fc[first] if first < len(fc) else '-'}")

    # Ties: equal-cost solutions may come out in any order, and at the k-th
    # cost the two methods may legitimately keep different members of a tie.
    # So compare assignment sets per cost level, excluding the boundary level.
    if jc and fc:
        boundary = min(jc[-1], fc[-1])
        by_j, by_f = defaultdict(set), defaultdict(set)
        for c, a in java_sols:
            by_j[c].add(a)
        for c, a in fm_sols:
            by_f[c].add(a)
        for c in sorted(set(by_j) | set(by_f)):
            if c >= boundary:
                continue
            if by_j[c] != by_f[c]:
                ok = False
                print(f"  cost {c}: assignment sets differ "
                      f"(java-only={len(by_j[c]-by_f[c])}, fastmurty-only={len(by_f[c]-by_j[c])})")

    ties = sum(1 for i in range(len(jc) - 1) if jc[i] == jc[i + 1])
    if ties:
        print(f"  note: {ties} adjacent tied costs in the java list -- order within ties is not compared")

    if show:
        print("\n  rank |  java cost  assignment            | fastmurty cost  assignment")
        for r in range(min(show, max(len(java_sols), len(fm_sols)))):
            js = java_sols[r] if r < len(java_sols) else ("-", ())
            fs = fm_sols[r] if r < len(fm_sols) else ("-", ())
            mark = "" if js == fs else "   *"
            print(f"  {r:4d} | {js[0]!s:>10} {list(js[1])!s:22} | {fs[0]!s:>14} {list(fs[1])!s}{mark}")
    return ok


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def load_matrix(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.replace(",", " ").strip()
            if line and not line.startswith("#"):
                rows.append([int(float(x)) for x in line.split()])
    C = np.array(rows, dtype=np.int64)
    if C.ndim != 2 or C.shape[0] != C.shape[1]:
        sys.exit(f"Matrix must be square (the Java Hungarian assumes n x n); got {C.shape}")
    return C


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--java-src", required=True, help="directory with our .java files")
    p.add_argument("--fastmurty", required=True, help="path to the fastmurty repo (with mhtda.so built)")
    p.add_argument("--java-class", default="MurtyEnumerator",
                   help="any enumerator class with a (AssignmentProblem) constructor and "
                        "List<AssignmentSolution> enumerate(int k), e.g. FastCacheEnumerator")
    p.add_argument("--matrix", help="text/CSV file with an integer square matrix (overrides --n)")
    p.add_argument("--n", type=int, default=6, help="size of random matrix")
    p.add_argument("--k", type=int, help="number of solutions (default: n!, capped at 5040 unless given)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--min-cost", type=int, default=1)
    p.add_argument("--max-cost", type=int, default=100)
    p.add_argument("--repeats", type=int, default=3,
                   help="runs per implementation; min time is reported (helps with JVM warm-up)")
    p.add_argument("--show", type=int, default=10, help="print the first N solutions side by side (0 = none)")
    p.add_argument("--java-opts", default="", help='extra JVM options, e.g. "-Xmx4g"')
    args = p.parse_args()

    if args.matrix:
        C = load_matrix(args.matrix)
    else:
        rng = np.random.default_rng(args.seed)
        C = rng.integers(args.min_cost, args.max_cost + 1, size=(args.n, args.n), dtype=np.int64)
    n = C.shape[0]

    nfact = math.factorial(n)
    k = args.k if args.k is not None else min(nfact, 5040)
    if k > nfact:
        print(f"k={k} exceeds n!={nfact}; using k={nfact}")
        k = nfact

    fm = load_fastmurty(args.fastmurty)
    print(f"n={n}  k={k}  java class={args.java_class}  fastmurty sparse build={fm.sparse}")
    if n <= 12:
        print("cost matrix:\n" + "\n".join("  " + " ".join(f"{v:4d}" for v in row) for row in C))

    with tempfile.TemporaryDirectory() as tmp:
        matrix_path = os.path.join(tmp, "matrix.txt")
        np.savetxt(matrix_path, C, fmt="%d")
        build_java(args.java_src, tmp, args.java_class)
        java_sols, java_times, java_wall = run_java(tmp, matrix_path, k, args.repeats, args.java_opts.split())

    fm_sols, fm_times, S = run_fastmurty(fm, C, k, args.repeats)

    print(f"\nfastmurty shift S = {S}")
    print("\nTiming (enumeration only, min / mean over repeats):")
    print(f"  java      : {min(java_times)*1e3:10.3f} ms / {np.mean(java_times)*1e3:10.3f} ms"
          f"   (process wall incl. JVM start: {java_wall:.2f} s)")
    print(f"  fastmurty : {min(fm_times)*1e3:10.3f} ms / {np.mean(fm_times)*1e3:10.3f} ms")
    if min(fm_times) > 0:
        print(f"  ratio java/fastmurty (min): {min(java_times)/min(fm_times):.1f}x")

    print("\nCorrectness check:")
    ok = compare(C, java_sols, fm_sols, args.show)
    print("\nRESULT:", "MATCH" if ok else "MISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
