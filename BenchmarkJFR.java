import java.io.IOException;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.Random;
import jdk.jfr.*;

// Custom marker event — shows up on JMC timeline
@Label("Algorithm Run")
@Category("Benchmark")
class AlgorithmEvent extends Event {
    @Label("Algorithm") String algorithm;
    @Label("N")         int n;
    @Label("K")         int k;
}

public class BenchmarkJFR {

    static final int   MAX_N       = 32;
    static final int[] N_VALUES    = buildNValues(MAX_N);
    static final int[] K_VALUES    = { 100000, 500000 };
    static final long  SEED        = 42L;
    static final int   COST_RANGE  = 999999; // restore realistic range

    static int[] buildNValues(int maxN) {
        java.util.List<Integer> vals = new java.util.ArrayList<>();
        for (int n = 2; n <= maxN; n += 10)
            vals.add(n);
        return vals.stream().mapToInt(Integer::intValue).toArray();
    }

    public static void main(String[] args) throws IOException {

        Recording rec = new Recording();
        rec.enable("jdk.ExecutionSample")   .withPeriod(Duration.ofMillis(10)); // finer sampling
        rec.enable("jdk.NativeMethodSample").withPeriod(Duration.ofMillis(10));
        rec.enable("jdk.CPULoad")           .withPeriod(Duration.ofMillis(100));
        rec.enable("jdk.ThreadCPULoad")     .withPeriod(Duration.ofMillis(100));
        rec.enable("jdk.GarbageCollection");
        rec.enable("jdk.GCPhasePause");     // see GC pauses on timeline

        rec.start();

        for (int n : N_VALUES) {
            int[][] matrix = randomMatrix(n, COST_RANGE, SEED);
            AssignmentProblem problem = new AssignmentProblem(matrix);
            System.out.printf("--- n=%d ---%n", n);

            for (int k : K_VALUES) {
                System.out.printf("  k=%d%n", k);

                runOrderGraph(problem, n, k,
                    new EnumeratorConfig(false, false, false, LoggingMode.NONE), "OG_FFFF");

                runOrderGraph(problem, n, k,
                    new EnumeratorConfig(true, false, false, LoggingMode.NONE), "OG_TFFF");

                runMurty(problem, n, k);
            }
        }

        rec.stop();
        rec.dump(Paths.get("JFR", "benchmark_full_" + System.currentTimeMillis() + ".jfr"));
        rec.close();
    }

    static void runOrderGraph(AssignmentProblem problem, int n, int k,
                              EnumeratorConfig config, String label) {
        AlgorithmEvent e = new AlgorithmEvent();
        e.algorithm = label;
        e.n = n;
        e.k = k;
        e.begin();

        OrderGraphEnumerator og = new OrderGraphEnumerator(problem, config);
        og.enumerate(k);
        og = null;

        e.commit(); // marks end on timeline
    }

    static void runMurty(AssignmentProblem problem, int n, int k) {
        AlgorithmEvent e = new AlgorithmEvent();
        e.algorithm = "MURTY";
        e.n = n;
        e.k = k;
        e.begin();

        MurtyEnumerator murty = new MurtyEnumerator(problem);
        murty.enumerate(k);
        murty = null;

        e.commit();
    }

    static int[][] randomMatrix(int n, int costRange, long seed) {
        Random rng = new Random(seed);
        int[][] m = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                m[i][j] = rng.nextInt(costRange);
        return m;
    }
}