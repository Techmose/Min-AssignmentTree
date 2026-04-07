import java.io.IOException;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Random;

public class EnumeratorTestRunner {

    static int[][] generateMatrix(int n, int seed) {
        Random rand = new Random(seed);
        int[][] matrix = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                matrix[i][j] = rand.nextInt(9999);
        return matrix;
    }

    static void printMatrix(int[][] matrix) {
        System.out.println("Matrix:");
        for (int[] row : matrix)
            System.out.println(java.util.Arrays.toString(row));
    }

    static List<AssignmentResult> runConfig(int[][] costMatrix, int k, String label, AssignmentTreeEnumerator.EnumerateConfig config) throws IOException {
        System.out.println("\n========================================");
        System.out.println("Config: " + label);
        System.out.println("  logging:       " + config.logging);
        System.out.println("  pqEviction:    " + config.pqEviction);
        System.out.println("  customHash:    " + config.customHash);
        System.out.println("  cacheEviction: " + config.cacheEviction);
        System.out.println("========================================");

        AssignmentTreeEnumerator enumerator = new AssignmentTreeEnumerator(costMatrix);
        
        HashMap<ColSet, int[]> cache = new HashMap<>();

        long start = System.nanoTime();
        List<AssignmentResult> topK = enumerator.enumerate(k, cache, config);
        long end = System.nanoTime();

        System.out.printf("total time:      %.4f s\n", (end - start) * 1e-9);
        System.out.printf("results found:   %d / %d\n", topK.size(), k);
        enumerator.printCacheStats();

        return topK;
    }

    static boolean compareResults(List<AssignmentResult> a, List<AssignmentResult> b, String labelA, String labelB) {
    if (a.size() != b.size()) {
        System.out.printf("MISMATCH: %s has %d results, %s has %d results\n", labelA, a.size(), labelB, b.size());
        return false;
    }
    for (int i = 0; i < a.size(); i++) {
        if (a.get(i).totalCost != b.get(i).totalCost) {
            System.out.printf("MISMATCH at index %d: %s cost=%d, %s cost=%d\n",
                i, labelA, a.get(i).totalCost, labelB, b.get(i).totalCost);
            return false;
        }
    }
    System.out.printf("OK: %s == %s\n", labelA, labelB);
    return true;
    }

    public static void main(String[] args) throws IOException {
        int n = 15;
        int k = 1000000;
        int seed = 42; // fix seed so all configs run on the same matrix

        int[][] costMatrix = generateMatrix(n, seed);
        printMatrix(costMatrix);

        // --- Define configs to compare ---

        List<AssignmentResult> baseline = runConfig(costMatrix, k, "Baseline (no eviction, no logging)",
            new AssignmentTreeEnumerator.EnumerateConfig(
                false,  // logging
                false,  // pqEviction
                false,  // customHash
                false   // cacheEviction
            ));

        List<AssignmentResult> evictOnly = runConfig(costMatrix, k, "PQ Eviction only",
            new AssignmentTreeEnumerator.EnumerateConfig(
                false,  // logging
                true,   // pqEviction
                false,  // customHash
                false   // cacheEviction
            ));

        compareResults(baseline, evictOnly,   "baseline", "evictOnly");

        List<AssignmentResult> loggingOnly = runConfig(costMatrix, k, "Logging only",
            new AssignmentTreeEnumerator.EnumerateConfig(
                true,   // logging
                false,  // pqEviction
                false,  // customHash
                false   // cacheEviction
            ));
            
        List<AssignmentResult> allNoLog = runConfig(costMatrix, k, "All, No logs",
            new AssignmentTreeEnumerator.EnumerateConfig(
                false,   // logging
                true,   // pqEviction
                false,  // customHash
                true   // cacheEviction
            ));
    }
}
