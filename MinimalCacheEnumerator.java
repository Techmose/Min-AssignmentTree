import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * MinimalCacheEnumerator -- JavaFastmurty plus a deliberately small cache.
 *
 * Question it answers: how much of the cache's benefit comes from a few
 * high-value subproblems? Only subproblems that are likely to be reused
 * AND expensive to solve are ever cached:
 *
 *   1. Early: entries are inserted only while the first k/windowDivisor
 *      solutions are being popped (default k/100). The top solutions tend
 *      to share structure, so subproblems met early are the ones met again.
 *   2. Large: only children with at least minFreeFraction*n free rows
 *      (default n/2) are looked up or inserted. Bigger subproblems cost
 *      more to solve, and small ones are not worth a hash lookup.
 *
 * After the insert window closes the table is frozen: large children still
 * look up the entries already stored, but nothing new is added.
 *
 * Everything else is JavaFastmurty: warm-started FastHungarian steps, the
 * cost bound, and fastmurty's lookahead split order.
 *
 * Cache key
 *   Because the split order fixes rows in a data-dependent order, the
 *   fixed rows are an arbitrary set, not a prefix. A child subproblem is
 *   fully described by
 *     R = its free rows, C = its free columns,
 *     r = the row being re-matched, E = r's forbidden columns (within C).
 *   The free columns are exactly the free rows' columns in the parent's
 *   assignment, so both R and C grow by one element per child p. The key
 *   is a Zobrist hash, XOR-updated in O(1) per child:
 *     XOR zRow[R] ^ XOR zCol[C] ^ zRematch[r] ^ XOR wCol[E]
 *   and an entry stores R, C, r and E exactly so hits can be verified.
 *   E is filtered to free columns: an ancestor's exclusion of a column now
 *   held by a fixed row no longer constrains anything.
 *
 * Cached value
 *   The solved child FastState. Its free rows' matching and duals are
 *   valid for any branch with the same (R, C, r, E); its fixed rows belong
 *   to whichever branch solved it first. A cache-hit child is queued with
 *   that shared state and only rebuilt with its own fixed rows (merge) if
 *   it is actually popped, as in FastCacheEnumerator.
 */
public class MinimalCacheEnumerator implements Enumerator {
    AssignmentProblem problem;
    FastHungarian solver;

    // reusable per-split scratch (same as JavaFastmurty)
    int[] order;
    int[] estimate;
    int[] bestColumn;
    boolean[] columnAvailable;

    // Zobrist keys
    final long[] zRow;      // "this row is free"
    final long[] zCol;      // "this column is free"
    final long[] zRematch;  // "this row is the one being re-matched"
    final long[] wCol;      // "this column is forbidden for the re-matched row"
    final MinimalCacheTable table = new MinimalCacheTable();

    // configuration
    public boolean useCache = true;
    public int windowDivisor = 100;        // insert while the first k/windowDivisor solutions are popped
    public double minFreeFraction = 0.5;   // cache only children with >= minFreeFraction*n free rows
    public boolean verify = true;          // compare the exact key on a hash match

    // stats
    public int cacheHits = 0;
    public int cacheMisses = 0;      // eligible lookups that found nothing
    public int cacheInserts = 0;
    public int skippedSmall = 0;     // children too small to look up
    public int frozenMisses = 0;     // misses after the window closed (could not be inserted)
    public int prunedChildren = 0;
    public int totalCalls = 0;
    public long totalTime = 0;

    public MinimalCacheEnumerator(AssignmentProblem problem) {
        this(problem, 0x9E3779B97F4A7C15L);
    }

    /** @param seed fixed seed so runs are reproducible; any seed gives correct results */
    public MinimalCacheEnumerator(AssignmentProblem problem, long seed) {
        this.problem = problem;
        this.solver = new FastHungarian(problem);
        int n = problem.numRows;
        this.order = new int[n];
        this.estimate = new int[n];
        this.bestColumn = new int[n];
        this.columnAvailable = new boolean[n];

        SplittableRandom rng = new SplittableRandom(seed);
        this.zRow = new long[n];
        this.zCol = new long[n];
        this.zRematch = new long[n];
        this.wCol = new long[n];
        for (int i = 0; i < n; i++) {
            zRow[i] = nonZero(rng);
            zCol[i] = nonZero(rng);
            zRematch[i] = nonZero(rng);
            wCol[i] = nonZero(rng);
        }
    }

    private static long nonZero(SplittableRandom rng) {
        long v;
        do { v = rng.nextLong(); } while (v == 0L);
        return v;
    }

    public List<AssignmentSolution> enumerate(int k) {
        List<AssignmentSolution> topK = new ArrayList<>();
        MurtyQueue pq = new MurtyQueue(k);
        int n = problem.numRows;
        int[][] c = problem.costMatrix;
        int words = (n + 63) >>> 6;
        long[] freeRowBits = new long[words];
        long[] freeColBits = new long[words];

        int windowPops = k / windowDivisor;
        int minFree = (int) Math.ceil(minFreeFraction * n);

        long startTime = System.nanoTime();
        FastState rootState = solver.solveFull();
        totalTime += System.nanoTime() - startTime;
        totalCalls++;
        rootState.fixedRow = new boolean[n + 1];

        AssignmentSolution rootSolution = new AssignmentSolution(rootState.assignmentZeroIndexed(), rootState.cost);
        pq.qInsert(new MinimalCacheNode(rootSolution, rootState, true, rootState.fixedRow, -1, new int[0]));

        while (topK.size() < k && !pq.isEmpty()) {
            MinimalCacheNode node = (MinimalCacheNode) pq.qPopMin();
            topK.add(node.solution);
            if (topK.size() == k) break;

            boolean insertOpen = useCache && topK.size() <= windowPops;

            // a cache-hit node carries another branch's fixed rows; fix them now that it is popped
            if (!node.merged) {
                node.state = mergeFixedRows(node);
                node.merged = true;
            }
            FastState nodeState = node.state;
            int[] assignment = node.solution.assignment;
            int parentCost = node.solution.cost;

            int m = splitOrder(node);
            if (m < 2) continue; // a single free row has no alternative column

            // fixed rows for child p: node's fixed rows plus order[p+1..m-1]
            boolean[] fixedScratch = node.fixed.clone();
            for (int p = 1; p < m; p++) fixedScratch[order[p] + 1] = true;

            // free rows/cols for child p: order[0..p] and their columns.
            // Start with order[0]; each child adds one row and one column.
            Arrays.fill(freeRowBits, 0L);
            Arrays.fill(freeColBits, 0L);
            int r0 = order[0];
            int c0 = assignment[r0];
            freeRowBits[r0 >>> 6] |= 1L << r0;
            freeColBits[c0 >>> 6] |= 1L << c0;
            long freeHash = zRow[r0] ^ zCol[c0];
            int freeParentCost = c[r0][c0];   // parent's cost over the child's free rows

            for (int p = 1; p < m; p++) {
                int row = order[p];
                int col = assignment[row];
                fixedScratch[row + 1] = false;
                freeRowBits[row >>> 6] |= 1L << row;
                freeColBits[col >>> 6] |= 1L << col;
                freeHash ^= zRow[row] ^ zCol[col];
                freeParentCost += c[row][col];

                // the row carrying this node's exclusions keeps them all
                int[] excludedCols;
                if (row == node.splitRow) {
                    excludedCols = new int[node.splitExcluded.length + 1];
                    System.arraycopy(node.splitExcluded, 0, excludedCols, 0, node.splitExcluded.length);
                    excludedCols[node.splitExcluded.length] = col;
                } else {
                    excludedCols = new int[]{col};
                }

                int freeCount = p + 1;
                int fixedCost = parentCost - freeParentCost;  // fixed rows keep the parent's columns
                boolean eligible = useCache && freeCount >= minFree;

                long key = 0L;
                int[] keyExcl = null;
                MinimalCacheTable.Entry hit = null;
                if (eligible) {
                    keyExcl = freeExclusions(excludedCols, freeColBits);
                    key = freeHash ^ zRematch[row];
                    for (int e : keyExcl) key ^= wCol[e];
                    if (table.size() > 0) {
                        hit = table.find(key, freeRowBits, freeColBits, row, keyExcl, verify);
                    }
                    if (hit != null) cacheHits++;
                    else {
                        cacheMisses++;
                        if (!insertOpen) frozenMisses++;
                    }
                } else if (useCache) {
                    skippedSmall++;
                }

                MinimalCacheNode child;
                if (hit != null) {
                    int childCost = fixedCost + hit.state.freeCost;
                    if (pq.isFull() && childCost >= pq.peekMax().solution.cost) continue;
                    int[] childAssignment = buildHitAssignment(assignment, fixedScratch, hit.state);
                    child = new MinimalCacheNode(new AssignmentSolution(childAssignment, childCost),
                                                 hit.state, false, fixedScratch.clone(), row, excludedCols);
                } else {
                    int costBound = pq.isFull() ? pq.peekMax().solution.cost - parentCost : problem.infinity;

                    long stepStart = System.nanoTime();
                    FastState childState = solver.solveStepFixedSet(nodeState, row, excludedCols, fixedScratch, costBound);
                    totalTime += System.nanoTime() - stepStart;
                    totalCalls++;

                    if (childState == null || childState.cost >= problem.infinity) {
                        prunedChildren++;
                        continue; // infeasible, or provably worse than the current worst kept solution
                    }
                    // solveStepFixedSet sets freeCost = cost; the cache needs the free rows' share only
                    childState.freeCost = childState.cost - fixedCost;

                    if (eligible && insertOpen) {
                        table.put(new MinimalCacheTable.Entry(key, freeRowBits.clone(), freeColBits.clone(),
                                                              row, keyExcl, childState));
                        cacheInserts++;
                    }
                    child = new MinimalCacheNode(new AssignmentSolution(childState.assignmentZeroIndexed(), childState.cost),
                                                 childState, true, childState.fixedRow, row, excludedCols);
                }

                if (pq.size() < k) {
                    pq.qInsert(child);
                } else if (child.solution.cost < pq.peekMax().solution.cost) {
                    pq.qReplaceMax(child);
                }
                // else: worse than everything already kept
            }
        }
        printStats();
        return topK;
    }

    /** Sorted copy of the excluded columns that are still free (the only ones that constrain the child). */
    private static int[] freeExclusions(int[] excludedCols, long[] freeColBits) {
        int cnt = 0;
        for (int e : excludedCols) if ((freeColBits[e >>> 6] & (1L << e)) != 0) cnt++;
        int[] out = new int[cnt];
        int i = 0;
        for (int e : excludedCols) if ((freeColBits[e >>> 6] & (1L << e)) != 0) out[i++] = e;
        if (cnt > 1) Arrays.sort(out);
        return out;
    }

    /** Cache-hit child's assignment: fixed rows from this branch, free rows from the cached solve. */
    private static int[] buildHitAssignment(int[] parentAssignment, boolean[] fixed, FastState cached) {
        int n = parentAssignment.length;
        int[] a = new int[n];
        for (int r = 0; r < n; r++) {
            a[r] = fixed[r + 1] ? parentAssignment[r] : cached.rowToColumn[r + 1] - 1;
        }
        return a;
    }

    /** Copy of a cached state with this node's own fixed rows written in. */
    private static FastState mergeFixedRows(MinimalCacheNode node) {
        FastState merged = node.state.copy();
        int[] a = node.solution.assignment;
        boolean[] fixed = node.fixed;
        for (int r = 0; r < a.length; r++) {
            if (!fixed[r + 1]) continue;
            merged.rowToColumn[r + 1] = a[r] + 1;
            merged.columnToRow[a[r] + 1] = r + 1;
        }
        merged.fixedRow = fixed;
        merged.cost = node.solution.cost;
        return merged;
    }

    /**
     * Fills order[0..m-1] with the node's free rows in fastmurty's split
     * order and returns m. Identical to JavaFastmurty.splitOrder.
     */
    private int splitOrder(MinimalCacheNode node) {
        FastState st = node.state;
        int n = problem.numRows;
        int[] a = node.solution.assignment;
        int[][] c = problem.costMatrix;

        int m2 = 0;
        for (int r = 0; r < n; r++) {
            if (!st.fixedRow[r + 1] && r != node.splitRow) order[m2++] = r;
        }
        int m = m2 + (node.splitRow >= 0 ? 1 : 0);

        Arrays.fill(columnAvailable, false);
        for (int idx = 0; idx < m2; idx++) columnAvailable[a[order[idx]]] = true;
        for (int idx = 0; idx < m2; idx++) computeEstimate(idx, a, c, st);

        for (int pos = m2 - 1; pos >= 0; pos--) {
            int maxIdx = 0;
            for (int idx = 1; idx <= pos; idx++) {
                if (estimate[idx] > estimate[maxIdx]) maxIdx = idx;
            }
            swap(maxIdx, pos);
            int deadCol = a[order[pos]];
            columnAvailable[deadCol] = false;
            for (int idx = 0; idx < pos; idx++) {
                if (bestColumn[idx] == deadCol) computeEstimate(idx, a, c, st);
            }
        }

        if (node.splitRow >= 0) order[m2] = node.splitRow;
        return m;
    }

    private void computeEstimate(int idx, int[] a, int[][] c, FastState st) {
        int r = order[idx];
        int own = a[r];
        int best = Integer.MAX_VALUE;
        int bestCol = -1;
        int[] row = c[r];
        int u = st.rowPrice[r + 1];
        for (int j = 0; j < columnAvailable.length; j++) {
            if (!columnAvailable[j] || j == own) continue;
            int reduced = row[j] - u - st.colPrice[j + 1];
            if (reduced < best) {
                best = reduced;
                bestCol = j;
            }
        }
        estimate[idx] = best;
        bestColumn[idx] = bestCol;
    }

    private void swap(int x, int y) {
        int t = order[x]; order[x] = order[y]; order[y] = t;
        t = estimate[x]; estimate[x] = estimate[y]; estimate[y] = t;
        t = bestColumn[x]; bestColumn[x] = bestColumn[y]; bestColumn[y] = t;
    }

    public void printStats() {
        System.out.printf("cache hits: %d%n", cacheHits);
        System.out.printf("cache misses: %d (after window closed: %d)%n", cacheMisses, frozenMisses);
        System.out.printf("cache inserts: %d, skipped (too small): %d%n", cacheInserts, skippedSmall);
        System.out.printf("table: %d entries, %.2f slots probed per lookup, hash collisions caught by verify: %d%n",
            table.size(), table.lookups == 0 ? 0.0 : (double) table.probes / table.lookups, table.hashCollisions);
        System.out.printf("pruned children: %d%n", prunedChildren);
        System.out.printf("incremental solves: %d, total time: %.4fs%n", totalCalls, totalTime * 1e-9);
    }
}

/**
 * Open-addressing table (linear probing, load factor at most 0.5) keyed by
 * a 64-bit Zobrist hash. Each entry also keeps its exact key so a match can
 * be verified.
 */
final class MinimalCacheTable {
    static final class Entry {
        final long hash;
        final long[] freeRows;   // bitmap of free rows; never mutated
        final long[] freeCols;   // bitmap of free columns; never mutated
        final int rematchRow;
        final int[] excl;        // sorted forbidden columns for rematchRow, within freeCols
        final FastState state;

        Entry(long hash, long[] freeRows, long[] freeCols, int rematchRow, int[] excl, FastState state) {
            this.hash = hash;
            this.freeRows = freeRows;
            this.freeCols = freeCols;
            this.rematchRow = rematchRow;
            this.excl = excl;
            this.state = state;
        }
    }

    private Entry[] slots = new Entry[1 << 12];
    private int mask = slots.length - 1;
    private int size;

    long lookups;
    long probes;
    long hashCollisions;

    int size() { return size; }

    private static int index(long h, int m) {
        return (int) (h ^ (h >>> 32)) & m;
    }

    Entry find(long h, long[] freeRows, long[] freeCols, int rematchRow, int[] excl, boolean verify) {
        lookups++;
        int idx = index(h, mask);
        Entry e;
        while ((e = slots[idx]) != null) {
            probes++;
            if (e.hash == h) {
                if (!verify
                    || (e.rematchRow == rematchRow
                        && Arrays.equals(e.freeRows, freeRows)
                        && Arrays.equals(e.freeCols, freeCols)
                        && Arrays.equals(e.excl, excl))) {
                    return e;
                }
                hashCollisions++;
            }
            idx = (idx + 1) & mask;
        }
        probes++;
        return null;
    }

    void put(Entry e) {
        if ((size + 1) * 2 > slots.length) grow();
        insert(slots, mask, e);
        size++;
    }

    private static void insert(Entry[] arr, int m, Entry e) {
        int idx = index(e.hash, m);
        while (arr[idx] != null) idx = (idx + 1) & m;
        arr[idx] = e;
    }

    private void grow() {
        Entry[] bigger = new Entry[slots.length << 1];
        int newMask = bigger.length - 1;
        for (Entry e : slots) if (e != null) insert(bigger, newMask, e);
        slots = bigger;
        mask = newMask;
    }
}

/**
 * Queue node for MinimalCacheEnumerator. Like JavaFastmurtyNode, plus:
 *   merged  false while state is a cache entry from another branch
 *   fixed   this branch's fixed rows (1-based), kept separately because an
 *           unmerged state's fixedRow belongs to the other branch
 */
class MinimalCacheNode extends MurtyNode {
    FastState state;
    boolean merged;
    final boolean[] fixed;
    final int splitRow;
    final int[] splitExcluded;

    MinimalCacheNode(AssignmentSolution solution, FastState state, boolean merged,
                     boolean[] fixed, int splitRow, int[] splitExcluded) {
        super(solution, null, null);
        this.state = state;
        this.merged = merged;
        this.fixed = fixed;
        this.splitRow = splitRow;
        this.splitExcluded = splitExcluded;
    }
}
