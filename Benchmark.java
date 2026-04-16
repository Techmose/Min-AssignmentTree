import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Benchmark runner for OrderGraphEnumerator vs MurtyEnumerator.
 *
 * Configure runs by editing the three sections marked CONFIGURE below:
 *   1. N_VALUES      - matrix sizes to test
 *   2. K_VALUES      - solution counts to test (run for every N)
 *   3. SEED / COST_RANGE - matrix generation parameters
 *
 * For every (n, k) pair the benchmark runs:
 *   OG_FFFF  - OrderGraph, all optimizations off (baseline)
 *   OG_FTFF  - OrderGraph, cache eviction on only
 *   MURTY    - Murty's algorithm
 *
 * Output: benchmark_results.csv (appended), stdout summary table.
 */
public class Benchmark {

    // =========================================================================
    // CONFIGURE: matrix sizes to benchmark
    // =========================================================================
    static final int MAX_N = 20;

    static final int[] N_VALUES = buildNValues(MAX_N);

    static int[] buildNValues(int maxN) {
        java.util.List<Integer> vals = new java.util.ArrayList<>();
        for (int n = 2; n <= maxN; n += 4)
            vals.add(n);
        return vals.stream().mapToInt(Integer::intValue).toArray();
    }
    // =========================================================================
    // CONFIGURE: solution counts to benchmark (applied to every N above)
    // =========================================================================
    static final int[] K_VALUES = { 100000, 500000 };

    // =========================================================================
    // CONFIGURE: matrix generation
    // =========================================================================
    static final long SEED       = 42L;
    static final int  COST_RANGE = 999999;   // cell values drawn from [0, COST_RANGE)

    // -------------------------------------------------------------------------
    // Fixed — output file
    // -------------------------------------------------------------------------
    static final String OUTPUT_FILE = "benchmark_results.csv";

    // -------------------------------------------------------------------------
    // Entry point
    // -------------------------------------------------------------------------

    public static void main(String[] args) {
        System.out.printf("seed=%d  costRange=%d%n", SEED, COST_RANGE);
        System.out.printf("N values: %s%n", arrayToString(N_VALUES));
        System.out.printf("K values: %s%n", arrayToString(K_VALUES));
        System.out.println();

        // Open CSV writer once for the entire benchmark run
        boolean exists = new java.io.File(OUTPUT_FILE).exists();
        try (PrintWriter csv = new PrintWriter(
                new BufferedWriter(new FileWriter(OUTPUT_FILE, true)))) {

            if (!exists)
                csv.println("n,k,seed,config,time_ns,time_s");

            List<BenchmarkResult> allResults = new ArrayList<>();

            for (int n : N_VALUES) {
                int[][] matrix = randomMatrix(n, COST_RANGE, SEED);
                AssignmentProblem problem = new AssignmentProblem(matrix);
                System.out.printf("--- n=%d ---%n", n);

                for (int k : K_VALUES) {
                    System.out.printf("  k=%-8d%n", k);

                    // OG_FFFF — all optimizations off (baseline)
                    BenchmarkResult r1 = runOrderGraph(problem, n, k,
                        new EnumeratorConfig(false, false, false, LoggingMode.NONE),
                        "OG_FFFF");
                    writeCsvRow(csv, r1);
                    allResults.add(r1);
                    System.out.printf("    %-10s  %.4fs%n", r1.config, r1.timeNs * 1e-9);

                    // OG_TFFF — cache eviction on, everything else off
                    BenchmarkResult r2 = runOrderGraph(problem, n, k,
                        new EnumeratorConfig(true, false, false, LoggingMode.NONE),
                        "OG_TFFF");
                    writeCsvRow(csv, r2);
                    allResults.add(r2);
                    System.out.printf("    %-10s  %.4fs%n", r2.config, r2.timeNs * 1e-9);

                    // Murty
                    BenchmarkResult r3 = runMurty(problem, n, k);
                    writeCsvRow(csv, r3);
                    allResults.add(r3);
                    System.out.printf("    %-10s  %.4fs%n", r3.config, r3.timeNs * 1e-9);

                    System.out.println();
                    sanityCheck(r3, r1);
                    sanityCheck(r3, r2);
                }
                System.out.println();
            }

            System.out.printf("Results written to %s%n%n", OUTPUT_FILE);
            printSummaryTable(allResults);

        } catch (IOException e) {
            System.err.println("Failed to open CSV: " + e.getMessage());
        }
    }

    // -------------------------------------------------------------------------
    // Individual run methods — enumerator is nulled after timing to allow GC
    // -------------------------------------------------------------------------

    static BenchmarkResult runOrderGraph(AssignmentProblem problem, int n, int k,
                                         EnumeratorConfig config, String label) {
        OrderGraphEnumerator og = new OrderGraphEnumerator(problem, config);
        long t0 = System.nanoTime();
        List<AssignmentSolution> solution = og.enumerate(k);
        long elapsed = System.nanoTime() - t0;
        og = null;
        System.gc();
        return new BenchmarkResult(n, k, SEED, label, elapsed, solution);
    }

    static BenchmarkResult runMurty(AssignmentProblem problem, int n, int k) {
        MurtyEnumerator murty = new MurtyEnumerator(problem);
        long t0 = System.nanoTime();
        List<AssignmentSolution> solution = murty.enumerate(k);
        long elapsed = System.nanoTime() - t0;
        murty = null;
        System.gc();
        return new BenchmarkResult(n, k, SEED, "MURTY", elapsed, solution);
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    static int[][] randomMatrix(int n, int costRange, long seed) {
        Random rng = new Random(seed);
        int[][] m = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                m[i][j] = rng.nextInt(costRange);
        return m;
    }

    /** Write a single result row to an already-open CSV writer and flush immediately. */
    static void writeCsvRow(PrintWriter pw, BenchmarkResult r) {
        pw.printf("%d,%d,%d,%s,%d,%.6f%n",
                  r.n, r.k, r.seed, r.config, r.timeNs, r.timeNs * 1e-9);
        pw.flush();
    }

    static void printSummaryTable(List<BenchmarkResult> results) {
        System.out.println("=".repeat(60));
        System.out.printf("%-10s  %-10s  %-10s  %12s  %10s%n",
                          "n", "k", "config", "time_ns", "time_s");
        System.out.println("-".repeat(60));
        for (BenchmarkResult r : results)
            System.out.printf("%-10d  %-10d  %-10s  %12d  %10.4f%n",
                              r.n, r.k, r.config, r.timeNs, r.timeNs * 1e-9);
        System.out.println("=".repeat(60));
    }

    static String arrayToString(int[] arr) {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < arr.length; i++) {
            if (i > 0) sb.append(", ");
            sb.append(arr[i]);
        }
        return sb.append("]").toString();
    }
    static void sanityCheck(BenchmarkResult r1, BenchmarkResult r2){
        boolean match = r2.solution.size() == r2.solution.size();
        if (match) {
            for (int i = 0; i < r2.solution.size(); i++) {
                if (r2.solution.get(i).cost != r2.solution.get(i).cost) {
                    match = false;
                    break;
                }
            }
        }
        if (match) System.out.println("Sanity Check");
    }
}


/** Holds one benchmark run's timing result. */
class BenchmarkResult {
    final int    n;
    final int    k;
    final long   seed;
    final String config;
    final long   timeNs;
    final List<AssignmentSolution> solution;

    BenchmarkResult(int n, int k, long seed, String config, long timeNs, List<AssignmentSolution> solution) {
        this.n      = n;
        this.k      = k;
        this.seed   = seed;
        this.config = config;
        this.timeNs = timeNs;
        this.solution = solution;
    }
}
