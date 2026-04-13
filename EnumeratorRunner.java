import java.util.Random;

/**
 * Runs OrderGraphEnumerator across a set of configurations and prints
 * a timing/stats summary for each.
 *
 * Configs (pqReduction | cacheEviction | customHashing | logging):
 *   FFFF, TFFF, FTFF, FFTF, FFFT, TTTF, TTTT
 *
 * The 4th flag maps to LoggingMode: F = NONE, T = ALL.
 */
public class EnumeratorRunner {

    // ----------------------------------------------------------------
    // Problem matrix and k — edit these to change the benchmark target
    // ----------------------------------------------------------------
    static final int    N       = 20;
    static final int    COST_MAX   = 9999;   // costs are in [0, COST_MAX)
    static final long   SEED       = 0;  // set to 0 for a different matrix each run
    static final int    K          = 1000000;
    // ----------------------------------------------------------------
 
    static int[][] randomCostMatrix(int n, int maxCost, long seed) {
        Random rng = new Random(seed);
        int[][] m = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                m[i][j] = rng.nextInt(maxCost);
        return m;
    }

    // ----------------------------------------------------------------

    public static void main(String[] args) {
        int[][] costMatrix = randomCostMatrix(N, COST_MAX, SEED);
        AssignmentProblem problem = new AssignmentProblem(costMatrix);

        // Each row: { pqReduction, cacheEviction, customHashing, logging }
        boolean[][] flags = {
            { false, false, false, false },  // FFFF
            { true,  false, false, false },  // TFFF
            //{ false, false, true,  false },  // FFTF
            //{ true, false, true, false  },  // TFTF
            //{ false, true,  false, false },  // FTFF
            //{ true,  true,  true,  false },  // TTTF
            { false, false, false, true  },  // FFFT
            { true,  false,  false,  true  },  // TFFT
        };

        System.out.printf("%-6s  %10s  %10s  %10s  %10s  %10s  %8s  %8s%n",
            "Config", "total(s)", "hung(s)", "hash(s)", "evict(s)", "pqPrune(s)",
            "cHits", "cEvict");
        System.out.println("-".repeat(88));

        for (boolean[] f : flags) {
            boolean pq      = f[0];
            boolean evict   = f[1];
            boolean hashing = f[2];
            boolean logging = f[3];

            LoggingMode mode = logging ? LoggingMode.ALL : LoggingMode.NONE;
            EnumeratorConfig config = new EnumeratorConfig(pq, evict, hashing, mode);

            OrderGraphEnumerator enumerator = new OrderGraphEnumerator(problem, config);

            long wallStart = System.nanoTime();
            enumerator.enumerate(K);
            long wallEnd = System.nanoTime();

            double wallSec      = (wallEnd - wallStart)            * 1e-9;
            double hungSec      = enumerator.totalTime             * 1e-9;
            double hashSec      = enumerator.cache.hashingTime     * 1e-9;
            double evictSec     = enumerator.cache.cacheEvictionTime * 1e-9;
            double pqPruneSec   = enumerator.pqPruningTime         * 1e-9;
            int    cacheHits    = enumerator.cache.hits;
            int    cacheEvict   = enumerator.cache.evictions;

            String tag = (pq      ? "T" : "F")
                       + (evict   ? "T" : "F")
                       + (hashing ? "T" : "F")
                       + (logging ? "T" : "F");

            System.out.printf("%-6s  %10.4f  %10.4f  %10.4f  %10.4f  %10.4f  %8d  %8d%n",
                tag, wallSec, hungSec, hashSec, evictSec, pqPruneSec,
                cacheHits, cacheEvict);
        }
    }
}
