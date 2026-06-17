import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.PriorityQueue;
import java.util.Random;



/**
 * Controls which CSV files are written during enumeration.
 *
 * NONE          - no logging.
 * PQ_CACHE_SIZE - writes pqSize.csv and cacheSize.csv only.
 * ALL           - writes pqSize.csv, cacheSize.csv, and cacheHits.csv.
 *                 Note: cacheHits logging is expensive at large k.
 */

/**
 * Handles CSV logging for OrderGraphEnumerator.
 *
 * Files are written to: {rows}x{cols}/cacheHits_k{k}_{configTag}.csv
 *                        {rows}x{cols}/pqSize_k{k}_{configTag}.csv
 *                        {rows}x{cols}/cacheSize_k{k}_{configTag}.csv
 *
 * Each enumerate() call opens fresh writers, overwriting any prior run
 * with the same dimensions, k, and config.
 */
/**
 * Algorithm for enumerating the top-k best solutions to the
 * assignment problem.  This algorithm is based on searching the Order
 * Graph.
 */
public class OrderGraphEnumeratorSplit {
    AssignmentProblem problem;
    OrderGraphCacheSplit cache;
    EnumeratorConfig config;

    // stats for algorithm performance
    int totalCalls;
    long totalTime;
    long pqPruningTime;

    /**
     * Constructor for OrderGraphEnumerator with default config (all optimizations off).
     * @param problem The assignment problem to enumerate.
     */
    public OrderGraphEnumeratorSplit(AssignmentProblem problem) {
        this(problem, EnumeratorConfig.defaults());
    }

    /**
     * Constructor for OrderGraphEnumeratorSplit.
     * @param problem The assignment problem to enumerate.
     * @param config  Feature flags controlling optional optimizations.
     */
    public OrderGraphEnumeratorSplit(AssignmentProblem problem, EnumeratorConfig config) {
        this.problem = problem;
        this.config  = config;

        this.totalCalls    = 0;
        this.totalTime     = 0;
        this.pqPruningTime = 0;
    }

    /**
     * Enumerates the top-k solutions to the assignment problem.
     * @param k The number of solutions to enumerate.
     * @return A list of solutions to the assignment problem.
     */
    public List<AssignmentSolution> enumerate(int k) {
        // initialize data structures
        List<AssignmentSolution> topK = new ArrayList<>();
        this.cache = new OrderGraphCacheSplit(this.problem.numRows,
                                         this.problem.numCols,
                                         this.config);

        // open logger for this enumerate() call
        EnumeratorLogger logger = new EnumeratorLogger(
            this.config.loggingMode,
            this.problem.numRows,
            this.problem.numCols,
            k,
            this.config.configTag()
        );
        logger.open();

        // initial call to Hungarian algorithm
        AssignmentSolution solution = callHungarian(this.problem.costMatrix);

        // initialize priority queue with the root node (empty usedCols, no parent)
        PriorityQueue<OrderGraphNode> pq = new PriorityQueue<>();
        OrderGraphNode node = new OrderGraphNode(solution.cost, this.problem.numCols);
        pq.add(node);

        // persisted lower bound for PQ reduction; Integer.MAX_VALUE means no bound yet
        int pqLowerBound = Integer.MAX_VALUE;

        // iteration counter: increments once per pq.poll()
        int iteration = 0;

        // snapshots of cumulative timers at the start of each iteration,
        // used to compute per-iteration deltas for the timings CSV
        long prevHungarianTime  = this.totalTime;
        long prevCacheTime      = this.cache.hashingTime;
        long prevEvictionTime   = this.cache.cacheEvictionTime;
        long prevPqEvictionTime = this.pqPruningTime;
        long prevGetTime        = this.cache.getTime;
        long prevPutTime        = this.cache.putTime;
        long prevContainsTime   = this.cache.containsTime;

        while (topK.size() < k && !pq.isEmpty()) {
            // pop best solution
            node = pq.poll();
            iteration++;

            if (node.length == this.problem.numRows) {
                // found a leaf node — reconstruct solution by walking parent chain
                topK.add(node.solution());
                continue;
            }

            // usedCols is already stored on the node; no recomputation needed
            BitSet cols = node.usedCols;

            // generate children: try assigning the next row
            for (int col = 0; col < this.problem.numCols; col++) {
                // skip if column is already used
                if (cols.get(col)) continue;

                // Prune on partial path cost before cache lookup or Hungarian.
                // pathCost can only increase as more rows are assigned.
                int pathCost = this.costAt(node, col);
                if (this.config.pqReductionEnabled && pathCost > pqLowerBound)
                    continue;

                // Temporarily mutate cols in-place for the cache lookup, avoiding
                // a clone on every candidate. We clear the bit immediately after,
                // whether the child is pruned or survives to the PQ.
                cols.set(col);
                Integer cached = this.cache.getIfPresent(cols);
                int solCost;
                if (cached != null) {
                    solCost = cached;
                } else {
                    // Need a stable key for storage — clone only on a cache miss.
                    // The cloned BitSet is handed off to the new node below, so
                    // it is not cloned again.
                    BitSet newCols = (BitSet) cols.clone();
                    cols.clear(col); // restore parent's BitSet before any early exit
                    int[][] newMatrix = subMatrix(newCols);
                    solCost = callHungarian(newMatrix).cost;
                    this.cache.put(newCols, solCost);

                    // log cache hit for this child (ALL mode only)
                    logger.logCacheHit(iteration, node.length + 1, col, this.cache.hits);

                    int newCost = pathCost + solCost;
                    if (this.config.pqReductionEnabled && newCost > pqLowerBound)
                        continue;

                    pq.add(new OrderGraphNode(newCost, col, node, newCols));
                    continue;
                }
                cols.clear(col); // restore parent's BitSet after cache hit

                // log cache hit for this child (ALL mode only)
                logger.logCacheHit(iteration, node.length + 1, col, this.cache.hits);

                int newCost = pathCost + solCost;

                // secondary prune on full cost after sub-problem is known
                if (this.config.pqReductionEnabled && newCost > pqLowerBound)
                    continue;

                // Cache hit path: clone only now that we know the child survives.
                BitSet newCols = (BitSet) cols.clone();
                newCols.set(col);
                pq.add(new OrderGraphNode(newCost, col, node, newCols));
            }

            // --- PQ size reduction ---
            // Use quickselect to partition the PQ contents around the k-th cheapest
            // node in O(n) average time, then re-heapify only the k survivors.
            //
            // Complexity: O(n) quickselect + O(k) heapify = O(n) average overall.
            // The previous max-heap approach was O(n log k); full sort was O(n log n).
            if (this.config.pqReductionEnabled && pq.size() > 2 * k) {
                long pqt0 = System.nanoTime();

                // Drain PQ into a flat array for in-place partitioning.
                // toArray avoids an extra copy vs. new ArrayList<>(pq).
                @SuppressWarnings("unchecked")
                OrderGraphNode[] arr = pq.toArray(new OrderGraphNode[0]);

                // Partition so arr[0..k-1] are the k cheapest (unsorted),
                // arr[k] is the k-th cheapest, and arr[k+1..] are more expensive.
                quickselect(arr, 0, arr.length - 1, k - 1);

                // arr[k-1] is now the k-th cheapest — its cost is the new bound.
                int newBound = arr[k - 1].cost;
                if (newBound < pqLowerBound)
                    pqLowerBound = newBound;

                // Re-heapify the k survivors. addAll() on a just-cleared
                // PriorityQueue bulk-loads via sift-down in O(k).
                pq.clear();
                for (int i = 0; i < k; i++)
                    pq.add(arr[i]);

                this.pqPruningTime += System.nanoTime() - pqt0;
            }

            // log PQ and cache sizes once per iteration
            logger.logPQSize(iteration, pq.size());
            logger.logCacheSize(iteration, this.cache.size());

            // log per-iteration timing deltas
            long curHungarian  = this.totalTime;
            long curCache      = this.cache.hashingTime;
            long curEviction   = this.cache.cacheEvictionTime;
            long curPqEviction = this.pqPruningTime;
            long curGetTime    = this.cache.getTime;
            long curPutTime    = this.cache.putTime;
            long curContains   = this.cache.containsTime;
            logger.logTimings(iteration,
                curHungarian  - prevHungarianTime,
                curCache      - prevCacheTime,
                curEviction   - prevEvictionTime,
                curPqEviction - prevPqEvictionTime);
            logger.logCacheOpTimes(iteration,
                curGetTime  - prevGetTime,
                curPutTime  - prevPutTime,
                curContains - prevContainsTime);
            prevHungarianTime  = curHungarian;
            prevCacheTime      = curCache;
            prevEvictionTime   = curEviction;
            prevPqEvictionTime = curPqEviction;
            prevGetTime        = curGetTime;
            prevPutTime        = curPutTime;
            prevContainsTime   = curContains;
        }

        logger.close();
        return topK;
    }

    /**
     * Solve the assignment problem.
     * @param matrix Cost matrix of assignment problem.
     * @return Optimal solution to the assignment problem.
     */
    AssignmentSolution callHungarian(int[][] matrix) {
        long startTime = System.nanoTime();
        AssignmentSolution solution = Hungarian.solve_alone(matrix);
        long endTime = System.nanoTime();
        this.totalCalls += 1;
        this.totalTime  += endTime - startTime;
        return solution;
    }

    /**
     * Computes the partial path cost for the child node that would be created
     * by assigning {@code col} to the next row after {@code parent}.
     *
     * Walks the parent-pointer chain to sum costMatrix[row][assignedCol] for
     * each row already in the path, then adds costMatrix[parent.length][col]
     * for the new assignment. This replaces the old cost(List) call and avoids
     * iterating a copied ArrayList.
     *
     * O(depth) — same asymptotic cost as before, but with no allocation.
     */
    int costAt(OrderGraphNode parent, int col) {
        int total = this.problem.costMatrix[parent.length][col];
        OrderGraphNode cur = parent;
        int row = parent.length - 1;
        while (cur.parent != null) {
            total += this.problem.costMatrix[row][cur.col];
            cur = cur.parent;
            row--;
        }
        return total;
    }

    /**
     * Quickselect: rearranges arr[lo..hi] so that arr[0..k] contains the
     * (k+1) cheapest nodes (by cost ascending) and arr[k] is exactly the
     * (k+1)-th cheapest. Elements within each partition are in no particular order.
     *
     * Average O(n), worst-case O(n²) — median-of-three pivot selection keeps
     * the worst case rare in practice without needing a random number generator.
     */
    private static void quickselect(OrderGraphNode[] arr, int lo, int hi, int k) {
        while (lo < hi) {
            // Median-of-three pivot: compare lo, mid, hi and put the median at hi
            int mid = lo + (hi - lo) / 2;
            if (arr[lo].cost > arr[mid].cost) swap(arr, lo, mid);
            if (arr[lo].cost > arr[hi].cost)  swap(arr, lo, hi);
            if (arr[mid].cost > arr[hi].cost) swap(arr, mid, hi);
            // arr[mid] is now the median; move it to hi-1 as the pivot
            swap(arr, mid, hi);
            int pivot = partition(arr, lo, hi);
            if      (pivot == k) return;
            else if (pivot  < k) lo = pivot + 1;
            else                 hi = pivot - 1;
        }
    }

    /** Lomuto partition around arr[hi]; returns final pivot index. */
    private static int partition(OrderGraphNode[] arr, int lo, int hi) {
        int pivotCost = arr[hi].cost;
        int i = lo;
        for (int j = lo; j < hi; j++) {
            if (arr[j].cost <= pivotCost)
                swap(arr, i++, j);
        }
        swap(arr, i, hi);
        return i;
    }

    private static void swap(OrderGraphNode[] arr, int i, int j) {
        OrderGraphNode tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }

    /** @deprecated No longer called from the hot path; kept for potential external use. */
    static BitSet pathToBitSet(List<Integer> path) {
        BitSet bs = new BitSet();
        for (int i : path)
            bs.set(i);
        return bs;
    }

    /**
     * This functions finds the sub-matrix of the assignment problem's
     * cost matrix found by excluding the given columns, and excluding
     * the same number of the initial rows of the matrix.  That is, it
     * returns the matrix that results from assigning the k given
     * columns to the first k rows.  (The particular assignment does
     * not matter).
     * @param cols Set of columns to _exclude_.
     * @return Corresponding sub-matrix
     */
    int[][] subMatrix(BitSet cols) {
        int colsSize = cols.cardinality();
        int rowsLeft = this.problem.numRows - colsSize;
        int colsLeft = this.problem.numCols - colsSize;
        int[][] matrix = new int[rowsLeft][colsLeft];
        int i = 0;
        for (int row = colsSize; row < this.problem.numRows; row++) {
            int j = 0;
            for (int col = 0; col < this.problem.numCols; col++) {
                if (cols.get(col)) continue;
                matrix[i][j] = this.problem.costMatrix[row][col];
                j++;
            }
            i++;
        }
        return matrix;
    }

    public void printCacheStats() {
        System.out.printf("cache hits:        %d\n", this.cache.hits);
        System.out.printf("cache misses:      %d\n", this.cache.misses);
        System.out.printf("cache size:        %d\n", this.cache.size());
        System.out.printf("cache evictions:   %d\n", this.cache.evictions);
        System.out.printf("time total:        %.4f s\n", this.totalTime            * 1e-9);
        System.out.printf("time hungarian:    %.4f s (%d calls)\n",
                          this.totalTime * 1e-9, this.totalCalls);
        System.out.printf("time hashing:      %.4f s\n", this.cache.hashingTime    * 1e-9);
        System.out.printf("time eviction:     %.4f s\n", this.cache.cacheEvictionTime * 1e-9);
        System.out.printf("time pq pruning:   %.4f s\n", this.pqPruningTime        * 1e-9);
    }

    public static void main(String[] args) {
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
        int k = 3628800;
        */

        int[][] costMatrix = new int[][] {
            {5,0,3,3,7,9,3,5,2,4,7,6,8,8,1,6,7,7,8,1},
            {5,9,8,9,4,3,0,3,5,0,2,3,8,1,3,3,3,7,0,1},
            {9,9,0,4,7,3,2,7,2,0,0,4,5,5,6,8,4,1,4,9},
            {8,1,1,7,9,9,3,6,7,2,0,3,5,9,4,4,6,4,4,3},
            {4,4,8,4,3,7,5,5,0,1,5,9,3,0,5,0,1,2,4,2},
            {0,3,2,0,7,5,9,0,2,7,2,9,2,3,3,2,3,4,1,2},
            {9,1,4,6,8,2,3,0,0,6,0,6,3,3,8,8,8,2,3,2},
            {0,8,8,3,8,2,8,4,3,0,4,3,6,9,8,0,8,5,9,0},
            {9,6,5,3,1,8,0,4,9,6,5,7,8,8,9,2,8,6,6,9},
            {1,6,8,8,3,2,3,6,3,6,5,7,0,8,4,6,5,8,2,3},
            {9,7,5,3,4,5,3,3,7,9,9,9,7,3,2,3,9,7,7,5},
            {1,2,2,8,1,5,8,4,0,2,5,5,0,8,1,1,0,3,8,8},
            {4,4,0,9,3,7,3,2,1,1,2,1,4,2,5,5,5,2,5,7},
            {7,6,1,6,7,2,3,1,9,5,9,9,2,0,9,1,9,0,6,0},
            {4,8,4,3,3,8,8,7,0,3,8,7,7,1,8,4,7,0,4,9},
            {0,6,4,2,4,6,3,3,7,8,5,0,8,5,4,7,4,1,3,3},
            {9,2,5,2,3,5,7,2,7,1,6,5,0,0,3,1,9,9,6,6},
            {7,8,8,7,0,8,6,8,9,8,3,6,1,7,4,9,2,0,8,2},
            {7,8,4,4,1,7,6,9,4,1,5,9,7,1,3,5,7,3,6,6},
            {7,9,1,9,6,0,3,8,4,1,4,5,0,3,1,4,4,4,0,0}};
        int k = 500000;

        AssignmentProblem problem = new AssignmentProblem(costMatrix);

        EnumeratorConfig config = new EnumeratorConfig(
            true,              // pqReductionEnabled
            false,              // cacheEvictionEnabled
            false,              // customHashingEnabled
            LoggingMode.NONE  // loggingMode: NONE, PQ_CACHE_SIZE, or ALL
        );

        OrderGraphEnumeratorSplit ogEnumerator = new OrderGraphEnumeratorSplit(problem, config);
        MurtyEnumerator mEnumerator = new MurtyEnumerator(problem);

        System.out.println("enumerating...");
        long start = System.nanoTime();
        List<AssignmentSolution> topK = ogEnumerator.enumerate(k);
        long end = System.nanoTime();
        System.out.printf("timer: %.4f\n", ((end - start) * 1e-9));

        System.out.println("enumerating (Murty's)...");
        start = System.nanoTime();
        List<AssignmentSolution> topK2 = mEnumerator.enumerate(k);
        end = System.nanoTime();
        System.out.printf("timer: %.4f\n", ((end - start) * 1e-9));
        System.out.printf("count: %d\n", topK.size());

        // sanity check
        boolean ok = true;
        if (topK.size() != topK2.size())
            System.out.println("check: NOT OK");
        else
            for (int i = 0; i < topK.size(); i++) {
                AssignmentSolution r1 = topK.get(i);
                AssignmentSolution r2 = topK2.get(i);
                if (r1.cost != r2.cost) {
                    ok = false;
                    break;
                }
            }

        if (ok)
            System.out.println("check: ok");
        else
            System.out.println("check: NOT OK");

        System.out.println("== Order Graph Stats:");
        ogEnumerator.printCacheStats();
        System.out.println("== Murty Stats:");
        mEnumerator.printCacheStats();
    }
}







