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
enum LoggingMode { NONE, PQ_CACHE_SIZE, ALL }


/**
 * Configuration for optional optimizations in OrderGraphEnumerator.
 */
class EnumeratorConfig {
    /** If true, prune the PQ when size exceeds 2k, persisting the cost lower bound. */
    boolean pqReductionEnabled;

    /** If true, evict the bottom 10% of cache entries by LFU when global size exceeds C(n, n/2). */
    boolean cacheEvictionEnabled;

    /** If true, use Zobrist hashing for cache keys instead of Java's default BitSet hash. */
    boolean customHashingEnabled;

    /** Controls which CSV log files are written during enumeration. */
    LoggingMode loggingMode;

    public EnumeratorConfig(boolean pqReductionEnabled,
                            boolean cacheEvictionEnabled,
                            boolean customHashingEnabled,
                            LoggingMode loggingMode) {
        this.pqReductionEnabled   = pqReductionEnabled;
        this.cacheEvictionEnabled = cacheEvictionEnabled;
        this.customHashingEnabled = customHashingEnabled;
        this.loggingMode          = loggingMode;
    }

    /** Default config: all optimizations off, no logging. */
    public static EnumeratorConfig defaults() {
        return new EnumeratorConfig(false, false, false, LoggingMode.NONE);
    }

    /**
     * Returns a three-character config tag for filenames.
     * Each character is T or F for: pqReduction, cacheEviction, customHashing.
     */
    public String configTag() {
        return (pqReductionEnabled   ? "T" : "F")
             + (cacheEvictionEnabled ? "T" : "F")
             + (customHashingEnabled ? "T" : "F");
    }
}


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
class EnumeratorLogger {
    private final LoggingMode mode;
    private final String      dir;
    private final String      fileSuffix; // e.g. "_k100000_TTF.csv"

    private PrintWriter cacheHitsWriter;
    private PrintWriter pqSizeWriter;
    private PrintWriter cacheSizeWriter;
    private PrintWriter timingsWriter;

    /**
     * @param mode       Logging mode from config.
     * @param numRows    Row count of the problem matrix.
     * @param numCols    Column count of the problem matrix.
     * @param k          Target-k passed to enumerate().
     * @param configTag  Three-char tag from EnumeratorConfig.configTag().
     */
    public EnumeratorLogger(LoggingMode mode, int numRows, int numCols,
                            int k, String configTag) {
        this.mode       = mode;
        this.dir        = numRows + "x" + numCols;
        this.fileSuffix = "_k" + k + "_" + configTag + ".csv";
    }

    /**
     * Opens all writers appropriate for the current logging mode.
     * Creates the output directory if it does not exist.
     */
    public void open() {
        if (this.mode == LoggingMode.NONE) return;
        try {
            new File(this.dir).mkdirs();

            this.pqSizeWriter = openWriter("pqSize");
            this.pqSizeWriter.println("k,size");

            this.cacheSizeWriter = openWriter("cacheSize");
            this.cacheSizeWriter.println("k,size");

            this.timingsWriter = openWriter("timings");
            this.timingsWriter.println("k,hungarian_time,cache_time,eviction_time,pq_eviction_time");

            if (this.mode == LoggingMode.ALL) {
                this.cacheHitsWriter = openWriter("cacheHits");
                this.cacheHitsWriter.println("k,depth,nodeLabel,hits");
            }
        } catch (IOException e) {
            System.err.println("Logger failed to open files: " + e.getMessage());
        }
    }

    /** Flushes and closes all open writers. */
    public void close() {
        closeWriter(this.pqSizeWriter);
        closeWriter(this.cacheSizeWriter);
        closeWriter(this.cacheHitsWriter);
        closeWriter(this.timingsWriter);
        this.pqSizeWriter    = null;
        this.cacheSizeWriter = null;
        this.cacheHitsWriter = null;
        this.timingsWriter   = null;
    }

    /**
     * Log one row to timings.csv. Called once per iteration.
     * All time values are per-iteration deltas in nanoseconds.
     *
     * @param iteration       Current outer-loop iteration count.
     * @param hungarianTime   Nanoseconds spent in Hungarian algorithm this iteration.
     * @param cacheTime       Nanoseconds spent on hashing/cache lookups this iteration.
     * @param evictionTime    Nanoseconds spent on cache eviction this iteration.
     * @param pqEvictionTime  Nanoseconds spent on PQ pruning this iteration.
     */
    public void logTimings(int iteration, long hungarianTime, long cacheTime,
                           long evictionTime, long pqEvictionTime) {
        if (this.timingsWriter == null) return;
        this.timingsWriter.printf("%d,%d,%d,%d,%d%n",
            iteration, hungarianTime, cacheTime, evictionTime, pqEvictionTime);
    }

    /** Log one row to pqSize.csv. Called once per iteration after PQ operations. */
    public void logPQSize(int iteration, int pqSize) {
        if (this.pqSizeWriter == null) return;
        this.pqSizeWriter.printf("%d,%d%n", iteration, pqSize);
    }

    /** Log one row to cacheSize.csv. Called once per iteration. */
    public void logCacheSize(int iteration, int cacheSize) {
        if (this.cacheSizeWriter == null) return;
        this.cacheSizeWriter.printf("%d,%d%n", iteration, cacheSize);
    }

    /**
     * Log one row to cacheHits.csv. Called per child node evaluated.
     * Only active under LoggingMode.ALL.
     *
     * @param iteration  Current outer-loop iteration count.
     * @param depth      Depth of the child node (path length after adding col).
     * @param colIndex   Column index representing this child (nodeLabel).
     * @param hits       Total cache hits so far at time of this lookup.
     */
    public void logCacheHit(int iteration, int depth, int colIndex, int hits) {
        if (this.cacheHitsWriter == null) return;
        this.cacheHitsWriter.printf("%d,%d,%d,%d%n", iteration, depth, colIndex, hits);
    }

    private PrintWriter openWriter(String name) throws IOException {
        String path = this.dir + File.separator + name + this.fileSuffix;
        return new PrintWriter(new BufferedWriter(new FileWriter(path, false)));
    }

    private void closeWriter(PrintWriter w) {
        if (w != null) { w.flush(); w.close(); }
    }
}


/**
 * Algorithm for enumerating the top-k best solutions to the
 * assignment problem.  This algorithm is based on searching the Order
 * Graph.
 */
public class OrderGraphEnumerator {
    AssignmentProblem problem;
    OrderGraphCache cache;
    EnumeratorConfig config;

    // stats for algorithm performance
    int totalCalls;
    long totalTime;
    long pqPruningTime;

    /**
     * Constructor for OrderGraphEnumerator with default config (all optimizations off).
     * @param problem The assignment problem to enumerate.
     */
    public OrderGraphEnumerator(AssignmentProblem problem) {
        this(problem, EnumeratorConfig.defaults());
    }

    /**
     * Constructor for OrderGraphEnumerator.
     * @param problem The assignment problem to enumerate.
     * @param config  Feature flags controlling optional optimizations.
     */
    public OrderGraphEnumerator(AssignmentProblem problem, EnumeratorConfig config) {
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
        this.cache = new OrderGraphCache(this.problem.numRows,
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

        // initialize priority queue
        PriorityQueue<OrderGraphNode> pq = new PriorityQueue<>();
        List<Integer> path = new ArrayList<Integer>();
        OrderGraphNode node = new OrderGraphNode(solution.cost, path);
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

        while (topK.size() < k && !pq.isEmpty()) {
            // pop best solution
            node = pq.poll();
            iteration++;

            if (node.path.size() == this.problem.numRows) {
                // found a leaf node
                topK.add(node.solution());
                continue;
            }

            // get set of used columns
            BitSet cols = pathToBitSet(node.path);

            // generate children: try assigning the next row
            for (int col = 0; col < this.problem.numCols; col++) {
                // skip if column is already used
                if (cols.get(col)) continue;

                // update path to node
                List<Integer> newPath = new ArrayList<>(node.path);
                newPath.add(col);

                // Bug fix 3: prune on partial path cost *before* cache lookup or Hungarian.
                // pathCost can only increase as more rows are assigned, so if it already
                // exceeds the bound there is no point evaluating the sub-problem.
                int pathCost = this.problem.cost(newPath);
                if (this.config.pqReductionEnabled && pathCost > pqLowerBound)
                    continue;

                BitSet newCols = (BitSet) cols.clone();
                newCols.set(col);

                // Bug fix 1: single lookup via getIfPresent() instead of contains() + get(),
                // which previously called makeKey() twice on every cache hit.
                Integer cached = this.cache.getIfPresent(newCols);
                int solCost;
                if (cached != null) {
                    solCost = cached;
                } else {
                    int[][] newMatrix = subMatrix(newCols);
                    solCost = callHungarian(newMatrix).cost;
                    this.cache.put(newCols, solCost);
                }

                // log cache hit for this child (ALL mode only)
                logger.logCacheHit(iteration, newPath.size(), col, this.cache.hits);

                int newCost = pathCost + solCost;

                // secondary prune on full cost after sub-problem is known
                if (this.config.pqReductionEnabled && newCost > pqLowerBound)
                    continue;

                // push child onto pq
                OrderGraphNode newNode = new OrderGraphNode(newCost, newPath);
                pq.add(newNode);
            }

            // --- PQ size reduction ---
            // If pq has grown beyond 2k, find the kth-cheapest cost as the new
            // lower bound and discard everything above it.
            if (this.config.pqReductionEnabled && pq.size() > 2 * k) {
                List<OrderGraphNode> nodes = new ArrayList<>(pq);
                long pqt0 = System.nanoTime();
                nodes.sort((a, b) -> b.cost - a.cost);
                int newBound = nodes.get(k - 1).cost;
                if (newBound < pqLowerBound)
                    pqLowerBound = newBound;
                pq.clear();
                pq.addAll(nodes.subList(k, nodes.size()));
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
            logger.logTimings(iteration,
                curHungarian  - prevHungarianTime,
                curCache      - prevCacheTime,
                curEviction   - prevEvictionTime,
                curPqEviction - prevPqEvictionTime);
            prevHungarianTime  = curHungarian;
            prevCacheTime      = curCache;
            prevEvictionTime   = curEviction;
            prevPqEvictionTime = curPqEviction;
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
        int k = 100000;

        AssignmentProblem problem = new AssignmentProblem(costMatrix);

        EnumeratorConfig config = new EnumeratorConfig(
            true,              // pqReductionEnabled
            true,              // cacheEvictionEnabled
            true,              // customHashingEnabled
            LoggingMode.ALL    // loggingMode: NONE, PQ_CACHE_SIZE, or ALL
        );

        OrderGraphEnumerator ogEnumerator = new OrderGraphEnumerator(problem, config);
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


/**
 * Used by Order Graph Enumerator.
 */
class OrderGraphNode implements Comparable<OrderGraphNode> {
    int cost;
    List<Integer> path;
    int length;
    int id;
    static int id_counter = 0;

    public OrderGraphNode(int cost, List<Integer> path) {
        this.cost   = cost;
        this.path   = path;
        this.length = path.size();
        this.id = OrderGraphNode.id_counter++;
    }

    /**
     * Order by cost ascending, then by depth descending (deeper = more progress),
     * then lexicographically for determinism.
     */
    public int compareTo(OrderGraphNode other) {
        if (this.cost < other.cost)       return -1;
        else if (this.cost > other.cost)  return  1;
        else {
            if (this.length > other.length)       return -1;
            else if (this.length < other.length)  return  1;
            else {
                /*
                for (int i = 0; i < this.length; i++)
                    if (this.path.get(i) < other.path.get(i))       return -1;
                    else if (this.path.get(i) > other.path.get(i))  return  1;
                return 0;
                 */
                if (this.id > other.id)         return 1;
                else if (this.id < other.id)    return -1;
                return 0;
            }
        }
    }

    public AssignmentSolution solution() {
        int[] arr = new int[this.path.size()];
        for (int i = 0; i < this.path.size(); i++)
            arr[i] = this.path.get(i);
        return new AssignmentSolution(arr, this.cost);
    }
}


/**
 * Wrapper around BitSet that replaces Java's default hash with a
 * Zobrist hash for better bucket distribution at similar cardinalities.
 *
 * equals() still delegates to BitSet.equals() for correctness.
 */
class ZobristKey {
    final BitSet bits;
    private final int hashCode;

    ZobristKey(BitSet bits, long[] zobristTable) {
        this.bits = bits;
        long h = 0L;
        for (int i = bits.nextSetBit(0); i >= 0; i = bits.nextSetBit(i + 1))
            h ^= zobristTable[i];
        // fold 64-bit Zobrist value into 32-bit int using Murmur3 finalizer
        h ^= (h >>> 33);
        h *= 0xff51afd7ed558ccdL;
        h ^= (h >>> 33);
        h *= 0xc4ceb9fe1a85ec53L;
        h ^= (h >>> 33);
        this.hashCode = (int) h;
    }

    @Override
    public int hashCode() {
        return this.hashCode;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof ZobristKey)) return false;
        return this.bits.equals(((ZobristKey) o).bits);
    }
}


/**
 * Cache for sub-problem costs used by OrderGraphEnumerator.
 *
 * Supports optional Zobrist hashing and LFU-based cache eviction.
 */
class OrderGraphCache {
    // flat cache: ZobristKey -> cost
    HashMap<ZobristKey, Integer> cache;

    // LFU frequency map (only populated when cacheEvictionEnabled)
    HashMap<ZobristKey, Integer> freq;

    // Zobrist random table, one long per column index
    long[] zobristTable;

    // maximum global cache size: C(n, n/2)
    int maxSize;

    // stats
    int hits;
    int misses;
    int evictions;

    // timing (nanoseconds)
    long hashingTime;
    long cacheEvictionTime;

    boolean customHashingEnabled;
    boolean cacheEvictionEnabled;

    public OrderGraphCache(int numRows, int numCols, EnumeratorConfig config) {
        this.customHashingEnabled = config.customHashingEnabled;
        this.cacheEvictionEnabled = config.cacheEvictionEnabled;

        // build Zobrist table regardless; cost is negligible and simplifies the code
        this.zobristTable = buildZobristTable(numCols);

        this.cache = new HashMap<>();
        this.freq  = new HashMap<>();

        // C(n, n/2) as the global cap
        this.maxSize = binomial(numCols, numCols / 2);

        this.hits      = 0;
        this.misses    = 0;
        this.evictions = 0;

        this.hashingTime       = 0;
        this.cacheEvictionTime = 0;

        // prime cache with the fully-assigned (trivial) sub-problem: cost = 0
        BitSet trivialSet = allCols(numCols);
        ZobristKey trivialKey = makeKey(trivialSet);
        this.cache.put(trivialKey, 0);
        if (this.cacheEvictionEnabled)
            this.freq.put(trivialKey, 1);
    }

    /** Build a table of random longs, one per column. */
    static long[] buildZobristTable(int numCols) {
        long[] table = new long[numCols];
        Random rng = new Random(0xdeadbeefL); // fixed seed for reproducibility
        for (int i = 0; i < numCols; i++)
            table[i] = rng.nextLong();
        return table;
    }

    /**
     * Compute C(n, k) as an int. Returns Integer.MAX_VALUE on overflow
     * so the cap is effectively disabled for very large problems.
     */
    static int binomial(int n, int k) {
        if (k < 0 || k > n) return 0;
        if (k == 0 || k == n) return 1;
        k = Math.min(k, n - k);
        long result = 1;
        for (int i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
            if (result > Integer.MAX_VALUE) return Integer.MAX_VALUE;
        }
        return (int) result;
    }

    static BitSet allCols(int numCols) {
        BitSet all = new BitSet(numCols);
        all.set(0, numCols);
        return all;
    }

    /**
     * Wrap a BitSet in a ZobristKey.
     * When customHashingEnabled is false the ZobristKey is still used as the
     * map key, but its hashCode falls back to the BitSet's own hash so the
     * behaviour matches the original code.
     */
    ZobristKey makeKey(BitSet bits) {
        if (this.customHashingEnabled)
            return new ZobristKey(bits, this.zobristTable);
        // fallback: wrap without Zobrist so hashCode delegates to BitSet
        return new ZobristKey(bits, this.zobristTable) {
            @Override public int hashCode() { return bits.hashCode(); }
        };
    }

    /**
     * Returns the cached cost for the given key, or null if not present.
     * Combines the old contains() + get() into a single map lookup,
     * eliminating the double makeKey() call on every cache hit.
     */
    public Integer getIfPresent(BitSet key) {
        long t0 = System.nanoTime();
        ZobristKey zk = makeKey(key);
        this.hashingTime += System.nanoTime() - t0;
        Integer val = this.cache.get(zk);
        if (val != null) {
            this.hits++;
            if (this.cacheEvictionEnabled)
                this.freq.merge(zk, 1, Integer::sum);
        } else {
            this.misses++;
        }
        return val;
    }

    /** @deprecated Use getIfPresent() to avoid double hashing on hits. */
    public boolean contains(BitSet key) {
        long t0 = System.nanoTime();
        ZobristKey zk = makeKey(key);
        this.hashingTime += System.nanoTime() - t0;
        if (this.cache.containsKey(zk)) {
            this.hits++;
            if (this.cacheEvictionEnabled)
                this.freq.merge(zk, 1, Integer::sum);
            return true;
        } else {
            this.misses++;
            return false;
        }
    }

    public int get(BitSet key) {
        long t0 = System.nanoTime();
        ZobristKey zk = makeKey(key);
        this.hashingTime += System.nanoTime() - t0;
        return this.cache.get(zk);
    }

    public void put(BitSet key, int value) {
        long t0 = System.nanoTime();
        ZobristKey zk = makeKey(key);
        this.hashingTime += System.nanoTime() - t0;
        this.cache.put(zk, value);
        if (this.cacheEvictionEnabled) {
            this.freq.put(zk, 1);
            long t1 = System.nanoTime();
            maybeEvict();
            this.cacheEvictionTime += System.nanoTime() - t1;
        }
    }

    public int size() {
        return this.cache.size();
    }

    /**
     * If the cache exceeds maxSize, evict the bottom 10% of entries by
     * frequency (least frequently used).
     */
    void maybeEvict() {
        if (this.cache.size() <= this.maxSize) return;

        // collect and sort all entries by frequency ascending
        List<java.util.Map.Entry<ZobristKey, Integer>> entries =
            new ArrayList<>(this.freq.entrySet());
        entries.sort(java.util.Map.Entry.comparingByValue());

        int toEvict = Math.max(1, this.cache.size() / 10);
        for (int i = 0; i < toEvict && i < entries.size(); i++) {
            ZobristKey victim = entries.get(i).getKey();
            this.cache.remove(victim);
            this.freq.remove(victim);
            this.evictions++;
        }
    }
}
