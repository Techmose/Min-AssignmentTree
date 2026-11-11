import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;
import java.util.function.Function;


/**
 * Benchmark runner for the top-k assignment enumerators.
 *
 * Configure runs by editing the sections marked CONFIGURE below:
 *   1. N_VALUES          - matrix sizes to test
 *   2. K_VALUES          - solution counts to test (run for every N)
 *   3. SEED / COST_RANGE - matrix generation parameters
 *   4. CONTENDERS        - which enumerators to run
 *
 * For every (n, k) pair, every contender is run in order. The FIRST
 * contender is the reference: every other contender's solution costs
 * are sanity-checked against it.
 *
 * Output: benchmark_results.csv (appended), stdout summary table.
 */
public class Benchmark {

    // =========================================================================
    // CONFIGURE: matrix sizes to benchmark
    // =========================================================================
    static final int MAX_N = 30;

    static final int[] N_VALUES = buildNValues(MAX_N);

    static int[] buildNValues(int maxN) {
        List<Integer> vals = new ArrayList<>();
        for (int n = 10; n <= maxN; n += 10)
            vals.add(n);
        return vals.stream().mapToInt(Integer::intValue).toArray();
    }

    // =========================================================================
    // CONFIGURE: solution counts to benchmark (applied to every N above)
    // =========================================================================
    static final int[] K_VALUES = { 10000, 100000 };

    // =========================================================================
    // CONFIGURE: matrix generation
    // =========================================================================
    static final long SEED       = 42L;
    static final int  COST_RANGE = 999999;   // cell values drawn from [0, COST_RANGE)

    // =========================================================================
    // CONFIGURE: enumerators to run. First entry = reference for sanity checks.
    // To add a new enumerator: make it implement Enumerator, add one line here.
    // =========================================================================
    static final List<Contender> CONTENDERS = List.of(
        //new Contender("MURTY", MurtyEnumerator::new)
        new Contender("FCM",   FastCacheEnumerator::new),
        new Contender("JFM",   JavaFastmurty::new)
        // new Contender("MURTY_OLD", MurtyEnumerator_OLD::new),
        // new Contender("OGE",       OrderGraphEXPEnumerator::new),
        // new Contender("OGE_OLD",   OrderGraphEXPEnumerator_OLD::new),
        // new Contender("OG_TFFF",   p -> new OrderGraphEnumerator(p,
        //         new EnumeratorConfig(true, false, false, LoggingMode.NONE)))
    );

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

                    List<BenchmarkResult> round = new ArrayList<>();
                    for (Contender c : CONTENDERS) {
                        BenchmarkResult r = run(c, problem, n, k);
                        writeCsvRow(csv, r);
                        allResults.add(r);
                        round.add(r);
                        System.out.printf("    %-10s  %.4fs%n", r.config, r.timeNs * 1e-9);
                    }

                    System.out.println();
                    BenchmarkResult reference = round.get(0);
                    for (BenchmarkResult r : round.subList(1, round.size()))
                        sanityCheck(reference, r, r.config + " vs " + reference.config);
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
    // The one run method. Construction happens before the timer starts, same
    // as the old per-enumerator methods; the enumerator goes out of scope
    // before System.gc() so it can be collected.
    // -------------------------------------------------------------------------

    static BenchmarkResult run(Contender c, AssignmentProblem problem, int n, int k) {
        Enumerator e = c.factory.apply(problem);
        long t0 = System.nanoTime();
        List<AssignmentSolution> solution = e.enumerate(k);
        long elapsed = System.nanoTime() - t0;
        e = null;
        System.gc();
        return new BenchmarkResult(n, k, SEED, c.label, elapsed, solution);
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
        return java.util.Arrays.toString(arr);
    }

    static void sanityCheck(BenchmarkResult r1, BenchmarkResult r2, String label) {
        boolean match = r1.solution.size() == r2.solution.size();
        if (match) {
            for (int i = 0; i < r2.solution.size(); i++) {
                if (r1.solution.get(i).cost != r2.solution.get(i).cost) {
                    System.out.println("i=" + i + ": " + r1.solution.get(i) + " != " + r2.solution.get(i));
                    match = false;
                    break;
                }
            }
        } else {
            System.out.println("Size mismatch: " + r1.solution.size() + " != " + r2.solution.size());
        }
        if (match) System.out.println("Sanity Check " + label);
    }
}


/** A labelled way to build an enumerator for a given problem. */
class Contender {
    final String label;
    final Function<AssignmentProblem, Enumerator> factory;

    Contender(String label, Function<AssignmentProblem, Enumerator> factory) {
        this.label   = label;
        this.factory = factory;
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
        this.n        = n;
        this.k        = k;
        this.seed     = seed;
        this.config   = config;
        this.timeNs   = timeNs;
        this.solution = solution;
    }
}
