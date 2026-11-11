import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.time.Duration;
import java.util.List;
import jdk.jfr.*;

// Custom marker event — shows up on JMC timeline
@Label("Algorithm Run")
@Category("Benchmark")
class AlgorithmEvent extends Event {
    @Label("Algorithm") String algorithm;
    @Label("N")         int n;
    @Label("K")         int k;
}

/**
 * JFR-profiled benchmark. Every contender run is wrapped in an
 * AlgorithmEvent labelled with the contender's name, n and k, so each
 * run shows up as its own span on the JMC timeline.
 *
 * Uses the same Enumerator interface and Contender class as Benchmark.java
 * (compile the two together).
 */
public class BenchmarkJFR {

    static final int   MAX_N       = 40;
    static final int[] N_VALUES    = buildNValues(MAX_N);
    static final int[] K_VALUES    = { 100000, 200000 };
    static final long  SEED        = 42L;
    static final int   COST_RANGE  = 999999; // restore realistic range

    // =========================================================================
    // CONFIGURE: enumerators to profile. Label = "Algorithm" field in JMC.
    // =========================================================================
    static final List<Contender> CONTENDERS = List.of(
        new Contender("MURTY",      JavaFastmurty::new),
        new Contender("FAST_CACHE", FastCacheEnumerator::new)
        // new Contender("OTHER",   OrderGraphEXPEnumerator::new),
        // new Contender("OG_FFFF", p -> new OrderGraphEnumerator(p,
        //         new EnumeratorConfig(false, false, false, LoggingMode.NONE))),
        // new Contender("OG_TFFF", p -> new OrderGraphEnumerator(p,
        //         new EnumeratorConfig(true, false, false, LoggingMode.NONE)))
    );

    static int[] buildNValues(int maxN) {
        java.util.List<Integer> vals = new java.util.ArrayList<>();
        for (int n = 20; n <= maxN; n += 5)
            vals.add(n);
        return vals.stream().mapToInt(Integer::intValue).toArray();
    }

    public static void main(String[] args) throws IOException {

        Recording rec = new Recording();
        rec.enable("jdk.ExecutionSample")   .withPeriod(Duration.ofMillis(1)); // finer sampling
        rec.enable("jdk.NativeMethodSample").withPeriod(Duration.ofMillis(1));
        rec.enable("jdk.CPULoad")           .withPeriod(Duration.ofMillis(100));
        rec.enable("jdk.ThreadCPULoad")     .withPeriod(Duration.ofMillis(100));
        rec.enable("jdk.MemoryAllocationSample").withPeriod(Duration.ofMillis(10));
        rec.enable("jdk.GarbageCollection");
        rec.enable("jdk.GCPhasePause");     // see GC pauses on timeline

        rec.start();

        for (int n : N_VALUES) {
            int[][] matrix = Benchmark.randomMatrix(n, COST_RANGE, SEED);
            AssignmentProblem problem = new AssignmentProblem(matrix);
            System.out.printf("--- n=%d ---%n", n);

            for (int k : K_VALUES) {
                System.out.printf("  k=%d%n", k);
                for (Contender c : CONTENDERS)
                    run(c, problem, n, k);
            }
        }

        rec.stop();
        Path dir = Paths.get("JFR");
        Files.createDirectories(dir);   // dump() fails if the folder is missing
        rec.dump(dir.resolve("benchmark_full_" + System.currentTimeMillis() + ".jfr"));
        rec.close();
    }

    /**
     * The one run method. Construction is inside the event span, same as
     * the old per-enumerator methods, so JMC shows setup + enumeration.
     */
    static void run(Contender c, AssignmentProblem problem, int n, int k) {
        AlgorithmEvent e = new AlgorithmEvent();
        e.algorithm = c.label;
        e.n = n;
        e.k = k;
        e.begin();

        Enumerator enumerator = c.factory.apply(problem);
        enumerator.enumerate(k);
        enumerator = null;

        e.commit(); // marks end on timeline
    }
}
