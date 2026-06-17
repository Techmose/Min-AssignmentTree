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
 * Enumerates the top-k solutions to an assignment problem by searching
 * the Order Graph.
 *
 * Two distinct node types are used — keep them clear in your head:
 *
 *   PQNode  (OrderGraphNodeOG)
 *       Lives in the priority queue.  Represents one partial assignment
 *       path from the root to some depth.  Holds a path cost, a parent
 *       pointer, the column assigned at this depth, and a reference to
 *       the OGNode for its usedCols state.
 *
 *   OGNode  (OrderGraphOG.OGNode)
 *       Lives in the OrderGraphOG.  Represents a unique sub-problem
 *       identified by a usedCols BitSet.  Holds the Hungarian sub-cost
 *       and pre-wired edges to every child OGNode.
 *
 * The critical invariant: when a PQNode is popped and processed, we
 * look up its child sub-problems via the OGNode graph first (following
 * ogChildren[] edges, no hashing).  Hashing is only used as a fallback
 * when no direct edge is available yet.
 */
public class OrderGraphEnumeratorOG {
    AssignmentProblem    problem;
    OrderGraphCacheOG    ogCache;   // wraps the OrderGraphOG; named ogCache to be explicit
    EnumeratorConfig     config;

    // stats
    int  totalCalls;
    long totalTime;
    long pqPruningTime;

    /** Default config: all optimizations off. */
    public OrderGraphEnumeratorOG(AssignmentProblem problem) {
        this(problem, EnumeratorConfig.defaults());
    }

    public OrderGraphEnumeratorOG(AssignmentProblem problem, EnumeratorConfig config) {
        this.problem = problem;
        this.config  = config;

        this.totalCalls    = 0;
        this.totalTime     = 0;
        this.pqPruningTime = 0;
    }

    /**
     * Enumerates the top-k solutions to the assignment problem.
     * @param k The number of solutions to enumerate.
     * @return A list of the k cheapest solutions in cost order.
     */
    public List<AssignmentSolution> enumerate(int k) {
        List<AssignmentSolution> topK = new ArrayList<>();

        // ogCache wraps the OrderGraphOG and is the only route to sub-problem costs.
        this.ogCache = new OrderGraphCacheOG(this.problem.numRows,
                                             this.problem.numCols,
                                             this.config);

        EnumeratorLogger logger = new EnumeratorLogger(
            this.config.loggingMode,
            this.problem.numRows,
            this.problem.numCols,
            k,
            this.config.configTag()
        );
        logger.open();

        // Solve the root sub-problem (all rows unassigned) to seed the PQ.
        AssignmentSolution rootSolution = callHungarian(this.problem.costMatrix);

        PriorityQueue<OrderGraphNodeOG> pq = new PriorityQueue<>();

        // rootPQNode carries ogRootNode so the very first child lookups are
        // one array index: rootPQNode.ogNode.ogChildren[col].
        OrderGraphNodeOG rootPQNode = new OrderGraphNodeOG(
                rootSolution.cost,
                this.problem.numCols,
                this.ogCache.orderGraph.ogRootNode);
        pq.add(rootPQNode);

        // persisted lower bound for PQ reduction; MAX_VALUE = no bound yet
        int pqLowerBound = Integer.MAX_VALUE;

        int  iteration         = 0;
        long prevHungarianTime = this.totalTime;
        long prevCacheTime     = this.ogCache.hashingTime;
        long prevEvictionTime  = this.ogCache.cacheEvictionTime;
        long prevPqEvictTime   = this.pqPruningTime;
        long prevGetTime       = this.ogCache.getTime;
        long prevPutTime       = this.ogCache.putTime;
        long prevContainsTime  = this.ogCache.containsTime;

        while (topK.size() < k && !pq.isEmpty()) {
            // pqNode: the PQ node currently being expanded.
            OrderGraphNodeOG pqNode = pq.poll();
            iteration++;

            if (pqNode.length == this.problem.numRows) {
                // Leaf PQ node — full assignment reached.  Reconstruct path.
                topK.add(pqNode.solution());
                continue;
            }

            // pqParentUsedCols: the usedCols BitSet owned by pqNode.
            // Temporarily set/clear bits in-place to avoid cloning every column.
            BitSet pqParentUsedCols = pqNode.usedCols;

            for (int col = 0; col < this.problem.numCols; col++) {
                if (pqParentUsedCols.get(col)) continue;  // already assigned

                // O(1) path cost — stored on the PQNode, no chain walk.
                int pathCost = this.costAt(pqNode, col);
                if (this.config.pqReductionEnabled && pathCost > pqLowerBound)
                    continue;

                // --- Sub-problem lookup: one array index, no hashing ---
                //
                // pqNode.ogNode is the OGNode for this PQNode's sub-problem.
                // ogChildren[col] is the child OGNode reached by assigning col.
                // Backward wiring in insertOGNode() guarantees this slot is
                // filled as soon as the child sub-problem is inserted.
                OrderGraphOG.OGNode childOGNode = pqNode.ogNode.ogChildren[col];

                if (childOGNode != null && childOGNode.isSolved()) {
                    // Cache hit — no hashing, no traversal.
                    pqParentUsedCols.set(col);  // temporarily set for logging
                    logger.logCacheHit(iteration, pqNode.length + 1,
                                       col, this.ogCache.hits);
                    pqParentUsedCols.clear(col);

                    this.ogCache.orderGraph.directEdgeHits++;
                    this.ogCache.orderGraph.hits++;

                    int totalCost = pathCost + childOGNode.subCost;
                    if (this.config.pqReductionEnabled && totalCost > pqLowerBound)
                        continue;

                    // childPQNode carries childOGNode directly — next iteration
                    // will again use ogChildren[col] with no hashing.
                    OrderGraphNodeOG childPQNode = new OrderGraphNodeOG(
                            totalCost, pathCost, col, pqNode,
                            childOGNode.usedCols, childOGNode);
                    pq.add(childPQNode);

                } else {
                    // Cache miss — run Hungarian, insert, get childOGNode back.
                    // Clone usedCols now (only on miss) for the new OGNode.
                    pqParentUsedCols.set(col);
                    BitSet childUsedCols = (BitSet) pqParentUsedCols.clone();
                    pqParentUsedCols.clear(col);

                    int[][] childSubMatrix = subMatrix(childUsedCols);
                    int     solvedSubCost  = callHungarian(childSubMatrix).cost;

                    // insertSolvedOGNode wires BACKWARDS: sets
                    // parentOGNode.ogChildren[col] = childOGNode for every
                    // parent of childUsedCols — including pqNode.ogNode.
                    // So pqNode.ogNode.ogChildren[col] is now set and future
                    // PQNodes at this parent will hit the direct-edge path.
                    childOGNode = this.ogCache.insertSolvedOGNode(
                            childUsedCols, solvedSubCost, pqNode.ogNode, col);

                    logger.logCacheHit(iteration, pqNode.length + 1,
                                       col, this.ogCache.hits);

                    int totalCost = pathCost + solvedSubCost;
                    if (this.config.pqReductionEnabled && totalCost > pqLowerBound)
                        continue;

                    OrderGraphNodeOG childPQNode = new OrderGraphNodeOG(
                            totalCost, pathCost, col, pqNode,
                            childOGNode.usedCols, childOGNode);
                    pq.add(childPQNode);
                }
            }

            // --- PQ size reduction (quickselect) ---
            if (this.config.pqReductionEnabled && pq.size() > 2 * k) {
                long pqt0 = System.nanoTime();

                @SuppressWarnings("unchecked")
                OrderGraphNodeOG[] arr = pq.toArray(new OrderGraphNodeOG[0]);
                quickselect(arr, 0, arr.length - 1, k - 1);

                int newBound = arr[k - 1].cost;
                if (newBound < pqLowerBound)
                    pqLowerBound = newBound;

                pq.clear();
                for (int i = 0; i < k; i++)
                    pq.add(arr[i]);

                this.pqPruningTime += System.nanoTime() - pqt0;
            }

            logger.logPQSize(iteration, pq.size());
            logger.logCacheSize(iteration, this.ogCache.size());

            long curHungarian = this.totalTime;
            long curCache     = this.ogCache.hashingTime;
            long curEviction  = this.ogCache.cacheEvictionTime;
            long curPqEvict   = this.pqPruningTime;
            long curGetTime   = this.ogCache.getTime;
            long curPutTime   = this.ogCache.putTime;
            long curContains  = this.ogCache.containsTime;
            logger.logTimings(iteration,
                curHungarian - prevHungarianTime,
                curCache     - prevCacheTime,
                curEviction  - prevEvictionTime,
                curPqEvict   - prevPqEvictTime);
            logger.logCacheOpTimes(iteration,
                curGetTime  - prevGetTime,
                curPutTime  - prevPutTime,
                curContains - prevContainsTime);
            prevHungarianTime = curHungarian;
            prevCacheTime     = curCache;
            prevEvictionTime  = curEviction;
            prevPqEvictTime   = curPqEvict;
            prevGetTime       = curGetTime;
            prevPutTime       = curPutTime;
            prevContainsTime  = curContains;
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
     * Returns the cumulative path cost for the child node reached by
     * assigning {@code col} to the next row after {@code pqNode}.
     *
     * O(1): reads pqNode.pathCost (accumulated at construction time)
     * and adds the single new cell cost for this row/col assignment.
     */
    int costAt(OrderGraphNodeOG pqNode, int col) {
        return pqNode.pathCost + this.problem.costMatrix[pqNode.length][col];
    }

    /**
     * Quickselect: rearranges arr[lo..hi] so that arr[0..k] contains the
     * (k+1) cheapest nodes (by cost ascending) and arr[k] is exactly the
     * (k+1)-th cheapest. Elements within each partition are in no particular order.
     *
     * Average O(n), worst-case O(n²) — median-of-three pivot selection keeps
     * the worst case rare in practice without needing a random number generator.
     */
    private static void quickselect(OrderGraphNodeOG[] arr, int lo, int hi, int k) {
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
    private static int partition(OrderGraphNodeOG[] arr, int lo, int hi) {
        int pivotCost = arr[hi].cost;
        int i = lo;
        for (int j = lo; j < hi; j++) {
            if (arr[j].cost <= pivotCost)
                swap(arr, i++, j);
        }
        swap(arr, i, hi);
        return i;
    }

    private static void swap(OrderGraphNodeOG[] arr, int i, int j) {
        OrderGraphNodeOG tmp = arr[i];
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
        OrderGraphOG og = this.ogCache.orderGraph;
        System.out.printf("graph traversal hits: %d\n", og.directEdgeHits);
        System.out.printf("hash fallback hits:   %d\n", og.hashFallbackHits);
        System.out.printf("hash fallback misses: %d\n", og.hashFallbackMisses);
        System.out.printf("ogNode count:         %d\n", this.ogCache.size());
        System.out.printf("time total:           %.4f s\n", this.totalTime * 1e-9);
        System.out.printf("time hungarian:       %.4f s (%d calls)\n",
                          this.totalTime * 1e-9, this.totalCalls);
        System.out.printf("time hashing:         %.4f s\n", this.ogCache.hashingTime * 1e-9);
        System.out.printf("time pq pruning:      %.4f s\n", this.pqPruningTime       * 1e-9);
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

        OrderGraphEnumeratorOG ogEnumerator = new OrderGraphEnumeratorOG(problem, config);
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







