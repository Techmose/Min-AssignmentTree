import jdk.jfr.consumer.*;
import java.io.*;
import java.nio.file.*;
import java.util.*;

/**
 * Reads all .jfr files in JFR/ and writes:
 *   jfr_algorithms.csv   - one row per AlgorithmEvent (duration, n, k, algorithm)
 *   jfr_cpu.csv          - one row per CPULoad sample
 *   jfr_stacks.csv       - per-method sample counts, tagged to algorithm/n/k
 *
 * Usage:
 *   javac JfrExtract.java
 *   java  JfrExtract
 */
public class JfrExtract {

    /** Time window for one AlgorithmEvent run. */
    static class AlgoWindow {
        final String algorithm;
        final int    n, k;
        final long   startMs, endMs;

        AlgoWindow(String algorithm, int n, int k, long startMs, long endMs) {
            this.algorithm = algorithm;
            this.n = n; this.k = k;
            this.startMs = startMs;
            this.endMs   = endMs;
        }

        boolean contains(long tsMs) {
            return tsMs >= startMs && tsMs <= endMs;
        }
    }

    public static void main(String[] args) throws Exception {

        Path jfrDir = Paths.get("JFR");
        if (!Files.isDirectory(jfrDir)) {
            System.err.println("No JFR/ directory found. Run from your project root.");
            System.exit(1);
        }

        List<Path> jfrFiles = new ArrayList<>();
        try (var stream = Files.list(jfrDir)) {
            stream.filter(p -> p.toString().endsWith(".jfr"))
                  .sorted()
                  .forEach(jfrFiles::add);
        }

        if (jfrFiles.isEmpty()) {
            System.err.println("No .jfr files found in JFR/");
            System.exit(1);
        }

        System.out.printf("Found %d JFR file(s)%n", jfrFiles.size());

        try (PrintWriter algCsv   = new PrintWriter(new FileWriter("jfr_algorithms.csv"));
             PrintWriter cpuCsv   = new PrintWriter(new FileWriter("jfr_cpu.csv"));
             PrintWriter stackCsv = new PrintWriter(new FileWriter("jfr_stacks.csv"))) {

            algCsv.println("file,algorithm,n,k,duration_ms");
            cpuCsv.println("file,timestamp_ms,machineTotal,jvmUser,jvmSystem");
            stackCsv.println("file,algorithm,n,k,class,method,sample_count,pct_of_algo");

            for (Path jfrFile : jfrFiles) {
                String fname = jfrFile.getFileName().toString();
                System.out.println("  Processing: " + fname);

                // ── Pass 1: collect AlgorithmEvent windows ───────────────────
                List<AlgoWindow> windows = new ArrayList<>();

                try (RecordingFile rf = new RecordingFile(jfrFile)) {
                    while (rf.hasMoreEvents()) {
                        RecordedEvent event = rf.readEvent();
                        String type = event.getEventType().getName();

                        if (type.equals("AlgorithmEvent") || type.endsWith(".AlgorithmEvent")) {
                            long startMs = event.getStartTime().toEpochMilli();
                            long endMs   = event.getEndTime().toEpochMilli();
                            String algo  = safeString(event, "algorithm");
                            int    n     = safeInt(event, "n");
                            int    k     = safeInt(event, "k");

                            windows.add(new AlgoWindow(algo, n, k, startMs, endMs));
                            algCsv.printf("%s,%s,%d,%d,%.3f%n",
                                fname, algo, n, k, (double)(endMs - startMs));
                        }
                    }
                }

                System.out.printf("    Found %d algorithm windows%n", windows.size());

                if (windows.isEmpty()) {
                    System.out.println("    WARNING: No AlgorithmEvents found — " +
                        "make sure you're using the updated BenchmarkJFR.java with custom events.");
                }

                // ── Pass 2: tag ExecutionSamples to windows ──────────────────
                // window -> (class.method -> hit count)
                Map<AlgoWindow, Map<String, Integer>> windowCounts = new LinkedHashMap<>();
                for (AlgoWindow w : windows)
                    windowCounts.put(w, new LinkedHashMap<>());

                int totalSamples = 0, taggedSamples = 0;

                try (RecordingFile rf = new RecordingFile(jfrFile)) {
                    while (rf.hasMoreEvents()) {
                        RecordedEvent event = rf.readEvent();
                        String type = event.getEventType().getName();

                        // CPU Load
                        if (type.equals("jdk.CPULoad")) {
                            cpuCsv.printf("%s,%d,%.4f,%.4f,%.4f%n",
                                fname,
                                event.getStartTime().toEpochMilli(),
                                safeFloat(event, "machineTotal"),
                                safeFloat(event, "jvmUser"),
                                safeFloat(event, "jvmSystem"));
                            continue;
                        }

                        // ExecutionSample
                        if (!type.equals("jdk.ExecutionSample")) continue;
                        totalSamples++;

                        long tsMs = event.getStartTime().toEpochMilli();

                        AlgoWindow matched = null;
                        for (AlgoWindow w : windows) {
                            if (w.contains(tsMs)) { matched = w; break; }
                        }
                        if (matched == null) continue;
                        taggedSamples++;

                        RecordedStackTrace stack = event.getStackTrace();
                        if (stack == null) continue;

                        Map<String, Integer> counts = windowCounts.get(matched);

                        for (RecordedFrame frame : stack.getFrames()) {
                            RecordedMethod m = frame.getMethod();
                            if (m == null) continue;
                            String key = m.getType().getName() + "\t" + m.getName();
                            counts.merge(key, 1, Integer::sum);
                        }
                    }
                }

                System.out.printf("    ExecutionSamples: %d total, %d tagged%n",
                    totalSamples, taggedSamples);

                // ── Write stack rows with percentages ────────────────────────
                for (AlgoWindow w : windows) {
                    Map<String, Integer> counts = windowCounts.get(w);
                    int total = counts.values().stream().mapToInt(Integer::intValue).sum();
                    if (total == 0) {
                        System.out.printf("    WARNING: 0 samples tagged for %s n=%d k=%d%n",
                            w.algorithm, w.n, w.k);
                        continue;
                    }

                    counts.entrySet().stream()
                        .sorted((a, b) -> b.getValue() - a.getValue())
                        .forEach(entry -> {
                            String[] parts = entry.getKey().split("\t", 2);
                            double pct = 100.0 * entry.getValue() / total;
                            stackCsv.printf("%s,%s,%d,%d,%s,%s,%d,%.2f%n",
                                fname, w.algorithm, w.n, w.k,
                                parts[0], parts[1],
                                entry.getValue(), pct);
                        });
                }
            }
        }

        System.out.println("Written: jfr_algorithms.csv");
        System.out.println("Written: jfr_cpu.csv");
        System.out.println("Written: jfr_stacks.csv  ← method hotspot percentages here");
    }

    static String safeString(RecordedEvent e, String f) {
        try { return e.getString(f); } catch (Exception ex) { return ""; }
    }
    static int safeInt(RecordedEvent e, String f) {
        try { return e.getInt(f); } catch (Exception ex) { return -1; }
    }
    static float safeFloat(RecordedEvent e, String f) {
        try { return e.getFloat(f); } catch (Exception ex) { return 0f; }
    }
}
