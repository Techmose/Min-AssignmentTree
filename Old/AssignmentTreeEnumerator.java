import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.PriorityQueue;
import java.util.Random;

public class AssignmentTreeEnumerator {
    int[][] costMatrix;
    int numRows;
    int numCols;

    int cacheHits;
    int cacheMisses;
    public int totalCalls;
    long totalTime;
    long cacheHitTime;
    Map<Integer, Integer> callsByDepth;
    Map<Integer, Integer> cachedSolutions;
    int maxCacheSize;

    public AssignmentTreeEnumerator(int[][] costMatrix) {
        this.costMatrix = costMatrix;
        this.numRows = costMatrix.length;
        this.numCols = costMatrix[0].length;

        this.cacheHits = 0;
        this.cacheMisses = 0;
        this.totalCalls = 0;
        this.totalTime = 0;
        this.cacheHitTime = 0;

        this.callsByDepth = new HashMap<>();
        this.cachedSolutions = new HashMap<>();
        this.maxCacheSize = (int) (nCr(this.numCols, this.numCols / 2));
    }

    // -------------------------------------------------------------------------
    // Config
    // -------------------------------------------------------------------------

    public static class EnumerateConfig {
        public boolean logging;       // write pqSize and cacheHit CSV logs
        public boolean pqEviction;    // evict costliest nodes when pq > 2*k
        public boolean customHash;    // placeholder
        public boolean cacheEviction; // placeholder

        public EnumerateConfig(boolean logging, boolean pqEviction, boolean customHash, boolean cacheEviction) {
            this.logging = logging;
            this.pqEviction = pqEviction;
            this.customHash = customHash;
            this.cacheEviction = cacheEviction;
        }
    }

    // -------------------------------------------------------------------------
    // Enumerate
    // -------------------------------------------------------------------------

    public List<AssignmentResult> enumerate(int k, HashMap<ColSet, int[]> cacheOut, EnumerateConfig config) throws IOException {
        List<AssignmentResult> topK = new ArrayList<>();
        PriorityQueue<OrderTreeNode> pq = new PriorityQueue<>();
        // int[] stores {cost, hitCount}
        HashMap<ColSet, int[]> cache = new HashMap<>();
        HashMap<ColSet, Integer> nodeLabels = new HashMap<>();
        int upperBound = Integer.MAX_VALUE;
        int nextLabel = 1;
        ColSet allColsSet = allCols(this.numCols);
        cache.put(allColsSet, new int[]{0, 0});
        nodeLabels.put(allColsSet, nextLabel++);

        long startTime = System.nanoTime();
        int[] assignment = hungarianAlgo.solveHungarian(this.costMatrix);
        long endTime = System.nanoTime();
        this.totalCalls += 1;
        this.totalTime += endTime - startTime;

        // Only open log files if logging is enabled
        BufferedWriter pqLog = null;
        BufferedWriter cacheHitLog = null;
        if (config.logging) {
            pqLog = new BufferedWriter(new FileWriter("pqSize_" + this.numCols + "X" + this.numRows + "_" + k + "_results.csv"));
            pqLog.write("pqSize,k\n");
            cacheHitLog = new BufferedWriter(new FileWriter("cache_hits_" + this.numCols + "X" + this.numRows + "_" + k + ".csv"));
            cacheHitLog.write("k,node_label,depth,cost,hit_count\n");
        }

        int cost = cost(costMatrix, assignment);
        List<Integer> path = new ArrayList<Integer>();
        OrderTreeNode node = new OrderTreeNode(cost, path, new ColSet());
        pq.add(node);

        while (topK.size() < k && !pq.isEmpty()) {
            if (config.logging && pq.size() % 5 == 0) {
                pqLog.write(pq.size() + "," + topK.size() + "\n");
                pqLog.flush();
            }

            if (config.pqEviction && pq.size() > 2 * k) {
                upperBound = Math.min(upperBound, evictCostliest(pq, k));
            }

            node = pq.poll();
            if (node.path.size() == numRows) {
                topK.add(node.result());
                continue;
            }
            ColSet cols = node.colSet;
            for (int col = 0; col < this.numCols; col++) {
                if (cols.contains(col)) continue;
                List<Integer> newPath = new ArrayList<Integer>(node.path);
                newPath.add(col);
                ColSet newCols = new ColSet(cols, col);

                int pathCost = cost(costMatrix, newPath);
                int solCost = 0;
                if (cache.containsKey(newCols)) {
                    this.cacheHits += 1;
                    this.recordHashCall(newCols.size());

                    startTime = System.nanoTime();
                    int[] entry = cache.get(newCols);
                    endTime = System.nanoTime();
                    this.cacheHitTime += endTime - startTime;

                    solCost = entry[0];
                    entry[1] += 1;
                    int label = nodeLabels.get(newCols);
                    if (config.logging) {
                        cacheHitLog.write(topK.size() + "," + label + "," + newCols.size() + "," + solCost + "," + entry[1] + "\n");
                    }
                } else {
                    this.cacheMisses += 1;
                    int[][] newMatrix = subMatrix(newCols);

                    startTime = System.nanoTime();
                    int[] sol = hungarianAlgo.solveHungarian(newMatrix);
                    endTime = System.nanoTime();
                    this.totalCalls += 1;
                    this.totalTime += endTime - startTime;

                    solCost = cost(newMatrix, sol);
                    cache.put(newCols, new int[]{solCost, 0});
                    nodeLabels.put(newCols, nextLabel++);
                    recordMatrixCached(newCols.size());

                    if (config.cacheEviction) {
                        evictCache(cache, nodeLabels);
                    }
                }
                int newCost = pathCost + solCost;

                if (isBelowThreshold(newCost, upperBound)) {
                    pq.add(new OrderTreeNode(newCost, newPath, newCols));
                }
            }
        }

        if (config.logging) {
            pqLog.close();
            cacheHitLog.close();
        }
        cacheOut.putAll(cache);
        return topK;
    }

    /**
     * Sorts the PQ by cost descending, removes the k most costly nodes,
     * and returns the cost of the k-th removed node as the new global upper bound.
     */
    private int evictCostliest(PriorityQueue<OrderTreeNode> pq, int k) {
        List<OrderTreeNode> nodes = new ArrayList<>(pq);
        nodes.sort((a, b) -> b.cost - a.cost);

        // The k-th most costly node (last one removed) becomes the new bound
        int newBound = nodes.get(k - 1).cost;

        pq.clear();
        pq.addAll(nodes.subList(k, nodes.size()));

        return newBound;
    }

    /**
     * Returns true if cost is below the global upper bound.
     */
    private boolean isBelowThreshold(int cost, int upperBound) {
        return cost < upperBound;
    }

    // -------------------------------------------------------------------------
    // Cache Eviction
    // -------------------------------------------------------------------------

    /**
     * Evicts the least-frequently-used cache entries when the cache exceeds
     * maxCacheSize = C(n, n/2) / 2. Keeps the top half by hitCount.
     */
    private void evictCache(HashMap<ColSet, int[]> cache,
                            HashMap<ColSet, Integer> nodeLabels) {
        if (cache.size() <= maxCacheSize) return;

        List<Map.Entry<ColSet, int[]>> entries = new ArrayList<>(cache.entrySet());
        entries.sort((a, b) -> a.getValue()[1] - b.getValue()[1]);

        int toEvict = (int) Math.ceil(cache.size() * 0.10);
        for (int i = 0; i < toEvict; i++) {
            ColSet key = entries.get(i).getKey();
            cache.remove(key);
            nodeLabels.remove(key);
        }
    }

    /**
     * Computes n choose r.
     */
    private static long nCr(int n, int r) {
        if (r > n) return 0;
        if (r == 0 || r == n) return 1;
        r = Math.min(r, n - r);
        long result = 1;
        for (int i = 0; i < r; i++) {
            result = result * (n - i) / (i + 1);
        }
        return result;
    }

    // -------------------------------------------------------------------------
    // Stats
    // -------------------------------------------------------------------------

    public void recordHashCall(int depth) {
        callsByDepth.put(depth, callsByDepth.getOrDefault(depth, 0) + 1);
    }

    public void recordMatrixCached(int depth) {
        cachedSolutions.put(depth, cachedSolutions.getOrDefault(depth, 0) + 1);
    }

    public void printCacheStats() {
        System.out.printf("cache hits:      %d\n", this.cacheHits);
        System.out.printf("cache misses:    %d\n", this.cacheMisses);
        System.out.printf("cache hit time:  %.4f (%d hits)\n", this.cacheHitTime * 1e-9, this.cacheHits);
        System.out.printf("hungarian time:  %.4f (%d calls)\n", this.totalTime * 1e-9, this.totalCalls);
    }

    public void printPerEntryCacheStats(HashMap<ColSet, int[]> cache) {
        System.out.println("\n--- Per-Entry Cache Hit Counts ---");
        Map<Integer, List<int[]>> byDepth = new HashMap<>();
        for (Map.Entry<ColSet, int[]> entry : cache.entrySet()) {
            int depth = entry.getKey().size();
            byDepth.computeIfAbsent(depth, d -> new ArrayList<>()).add(entry.getValue());
        }
        for (int depth : new java.util.TreeSet<>(byDepth.keySet())) {
            List<int[]> entries = byDepth.get(depth);
            int totalHits = entries.stream().mapToInt(e -> e[1]).sum();
            long neverHit = entries.stream().filter(e -> e[1] == 0).count();
            int maxHits = entries.stream().mapToInt(e -> e[1]).max().orElse(0);
            System.out.printf("depth %2d: %3d entries | total hits: %5d | max hits on one entry: %4d | never hit: %3d\n",
                depth, entries.size(), totalHits, maxHits, neverHit);
        }
    }

    // -------------------------------------------------------------------------
    // Helpers
    // -------------------------------------------------------------------------

    static int cost(int[][] matrix, int[] assignment) {
        int cost = 0;
        for (int i = 0; i < assignment.length; i++)
            cost += matrix[i][assignment[i]];
        return cost;
    }

    static int cost(int[][] matrix, List<Integer> assignment) {
        int cost = 0;
        for (int i = 0; i < assignment.size(); i++)
            cost += matrix[i][assignment.get(i)];
        return cost;
    }

    static ColSet allCols(int numCols) {
        HashSet<Integer> all = new HashSet<Integer>(numCols);
        for (int col = 0; col < numCols; col++)
            all.add(col);
        return new ColSet(all);
    }

    int[][] subMatrix(ColSet cols) {
        int colsSize = cols.size();
        int rowsLeft = this.numRows - colsSize;
        int colsLeft = this.numCols - colsSize;
        int[][] matrix = new int[rowsLeft][colsLeft];
        int i = 0;
        for (int row = colsSize; row < this.numRows; row++) {
            int j = 0;
            for (int col = 0; col < this.numCols; col++) {
                if (cols.contains(col)) continue;
                matrix[i][j] = this.costMatrix[row][col];
                j++;
            }
            i++;
        }
        return matrix;
    }

    // -------------------------------------------------------------------------
    // Main
    // -------------------------------------------------------------------------

    public static void main(String[] args) throws IOException {

        /*
        int[][] costMatrix = new int[][] {
            {1, 5, 9},
            {6, 2, 8},
            {7, 4, 3}
        };
        int k = 6;
        */

        /*
        int[][] costMatrix = new int[][] {
            {5,0,3,3,7,9,3,5},
            {2,4,7,6,8,8,1,6},
            {7,7,8,1,5,9,8,9},
            {4,3,0,3,5,0,2,3},
            {8,1,3,3,3,7,0,1},
            {9,9,0,4,7,3,2,7},
            {2,0,0,4,5,5,6,8},
            {4,1,4,9,8,1,1,7}};
        int k = 40320;
        */

        /*
        int[][] costMatrix = new int[][] {
            {5,0,3,3,7,9,3,5,2},
            {4,7,6,8,8,1,6,7,7},
            {8,1,5,9,8,9,4,3,0},
            {3,5,0,2,3,8,1,3,3},
            {3,7,0,1,9,9,0,4,7},
            {3,2,7,2,0,0,4,5,5},
            {6,8,4,1,4,9,8,1,1},
            {7,9,9,3,6,7,2,0,3},
            {5,9,4,4,6,4,4,3,4}};
        int k = 362880;
        */

        /*
        int[][] costMatrix = new int[][] {
        {3,7,2,8,1,5,9,4,6,0,3,7,2,8,5,1,4,9,6,0},
        {6,1,8,3,9,4,0,7,2,5,6,1,8,3,9,4,0,7,2,5},
        {4,9,5,1,7,3,8,2,0,6,4,9,5,1,7,3,8,2,0,6},
        {0,5,3,7,2,8,1,6,9,4,0,5,3,7,2,8,1,6,9,4},
        {8,2,6,4,0,9,5,3,7,1,8,2,6,4,0,9,5,3,7,1},
        {5,4,0,9,6,2,7,1,3,8,5,4,0,9,6,2,7,1,3,8},
        {1,8,7,2,4,6,3,9,5,0,1,8,7,2,4,6,3,9,5,0},
        {9,3,4,6,8,0,2,5,1,7,9,3,4,6,8,0,2,5,1,7},
        {2,6,1,5,3,7,4,0,8,9,2,6,1,5,3,7,4,0,8,9},
        {7,0,9,3,5,1,6,8,4,2,7,0,9,3,5,1,6,8,4,2},
        {3,5,8,0,9,4,1,6,2,7,3,5,8,0,9,4,1,6,2,7},
        {6,9,2,7,1,8,0,4,5,3,6,9,2,7,1,8,0,4,5,3},
        {4,1,6,8,2,5,9,3,7,0,4,1,6,8,2,5,9,3,7,0},
        {0,7,3,4,6,9,2,8,1,5,0,7,3,4,6,9,2,8,1,5},
        {8,4,5,1,7,3,6,2,9,4,8,4,5,1,7,3,6,2,9,4},
        {5,2,9,6,0,7,8,1,3,4,5,2,9,6,0,7,8,1,3,4},
        {1,6,4,9,3,2,5,7,0,8,1,6,4,9,3,2,5,7,0,8},
        {7,8,0,2,5,6,3,4,9,1,7,8,0,2,5,6,3,4,9,1},
        {9,3,7,5,4,1,4,0,6,2,9,3,7,5,4,1,4,0,6,2},
        {2,0,1,8,6,4,7,9,3,5,2,0,1,8,6,4,7,9,3,5}};
        */

        /*
        int[][] costMatrix = new int[][] {
            {5,0,3,3,7,9,3,5,2,4},
            {7,6,8,8,1,6,7,7,8,1},
            {5,9,8,9,4,3,0,3,5,0},
            {2,3,8,1,3,3,3,7,0,1},
            {9,9,0,4,7,3,2,7,2,0},
            {0,4,5,5,6,8,4,1,4,9},
            {8,1,1,7,9,9,3,6,7,2},
            {0,3,5,9,4,4,6,4,4,3},
            {4,4,8,4,3,7,5,5,0,1},
            {5,9,3,0,5,0,1,2,4,2}};
        */

        int n = 20;
        int k = 1000000;

        Random rand = new Random();
        int[][] costMatrix = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                costMatrix[i][j] = rand.nextInt(9999);

        System.out.println("Generated Matrix:");
        for (int i = 0; i < n; i++)
            System.out.println(java.util.Arrays.toString(costMatrix[i]));

        AssignmentTreeEnumerator enumerator = new AssignmentTreeEnumerator(costMatrix);

        EnumerateConfig config = new EnumerateConfig(
            true,   // logging
            true,   // pqEviction
            false,  // customHash (placeholder)
            false   // cacheEviction (placeholder)
        );

        System.out.println("enumerating...");
        long start = System.nanoTime();
        HashMap<ColSet, int[]> cache = new HashMap<>();
        List<AssignmentResult> topK = enumerator.enumerate(k, cache, config);
        long end = System.nanoTime();
        System.out.printf("timer: %.4f\n", ((end - start) * 1e-9));

        enumerator.printCacheStats();
        enumerator.printPerEntryCacheStats(cache);
        System.out.print("Final Count: " + enumerator.callsByDepth);
    }
}


class OrderTreeNode implements Comparable<OrderTreeNode> {
    int cost;
    List<Integer> path;
    ColSet colSet;
    int length;

    public OrderTreeNode(int cost, List<Integer> path, ColSet colSet) {
        this.cost = cost;
        this.path = path;
        this.colSet = colSet;
        this.length = path.size();
    }

    public int compareTo(OrderTreeNode other) {
        if (this.cost < other.cost)
            return -1;
        else if (this.cost == other.cost)
            if (this.length < other.length)
                return -1;
            else if (this.length == other.length) {
                for (int i = 0; i < this.length; i++)
                    if (this.path.get(i) < other.path.get(i))
                        return -1;
                    else if (this.path.get(i) > other.path.get(i))
                        return 1;
                return 0;
            }
        return 1;
    }

    static List<Integer> asList(int[] array) {
        List<Integer> list = new ArrayList<>();
        for (int num : array)
            list.add(num);
        return list;
    }

    public AssignmentResult result() {
        return new AssignmentResult(new ArrayList<>(this.path), this.cost);
    }
}
