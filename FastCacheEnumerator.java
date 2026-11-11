import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.SplittableRandom;

/**
 * Same enumeration as FastCacheEnumerator / FastGraphCacheEnumerator (same
 * partition, same incremental FastHungarian.solveStep, same equivalence:
 * free columns F plus the columns E forbidden for the current row), but the
 * cache key is a 64-bit Zobrist hash maintained incrementally, and the cache
 * is an open-addressing table with no per-lookup allocation.
 *
 * Key identity
 *   A subproblem is (S, E): S = columns fixed to rows [0, i) (the complement
 *   of F, and |S| = i fixes the row), E = columns forbidden for row i.
 *   hash(S, E) = XOR of zCol[c] for c in S  XOR  XOR of wCol[c] for c in E.
 *
 * Why this is cheap
 *   - Child i+1 of a popped node fixes exactly one more column than child i,
 *     so S's hash advances with one XOR and one bit set. The key costs O(1)
 *     per child instead of building sets.
 *   - For i > startPos, E is the single column cur[i]. Only the child at
 *     i == startPos has a larger E, and that is one child per popped node.
 *   - No inclusion/exclusion lists are copied per child; fixed rows are the
 *     prefix of the node's own assignment.

 *
 * Works for any n (the S bitmap is a long[]).
 */
public class FastCacheEnumerator implements Enumerator {
    AssignmentProblem problem;
    FastHungarian solver;
    final long[] zCol;   // random value per column: "this column is fixed"
    final long[] wCol;   // independent random value per column: "forbidden for the current row"
    final ZobristStateTable table = new ZobristStateTable();

    public boolean useCache = true;
    public boolean skipTrivialInfeasible = true;
    public boolean verify = true;

    // stats
    public int cacheHits = 0;
    public int cacheMisses = 0;
    public int prunedChildren = 0;
    public int skippedInfeasible = 0;
    public int totalCalls = 0;
    public long totalTime = 0;

    public FastCacheEnumerator(AssignmentProblem problem) {
        this(problem, 0x9E3779B97F4A7C15L);
    }

    /** @param seed fixed seed so runs are reproducible; any seed gives correct results */
    public FastCacheEnumerator(AssignmentProblem problem, long seed) {
        this.problem = problem;
        this.solver = new FastHungarian(problem);
        SplittableRandom rng = new SplittableRandom(seed);
        int n = problem.numCols;
        this.zCol = new long[n];
        this.wCol = new long[n];
        for (int c = 0; c < n; c++) {
            zCol[c] = nonZero(rng);
            wCol[c] = nonZero(rng);
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
        int[][] costMatrix = problem.costMatrix;
        int n = problem.numRows;
        long[] sBits = new long[(n + 63) >>> 6];   // columns fixed to rows [0, i); reused for every child

        long startTime = System.nanoTime();
        FastState rootState = solver.solveFull();
        totalTime += System.nanoTime() - startTime;
        totalCalls++;

        AssignmentSolution rootSolution = new AssignmentSolution(rootState.assignmentZeroIndexed(), rootState.cost);
        pq.qInsert(new FastZNode(rootSolution, rootState, true, 0, -1, null, 0L));

        while (topK.size() < k && !pq.isEmpty()) {
            FastZNode node = (FastZNode) pq.qPopMin();
            topK.add(node.solution);

            int[] cur = node.solution.assignment;
            int startPos = node.startPos;

            // A cache-hit child carries the state of whichever branch solved that
            // subproblem first; fix its fixed rows only if it is actually popped.
            FastState nodeState = node.state;
            if (!node.merged) {
                nodeState = mergeFixedRows(nodeState, cur, startPos, node.solution.cost);
                node.state = nodeState;
                node.merged = true;
            }

            // S for rows [0, startPos): bitmap, hash and cost, built once per popped node
            Arrays.fill(sBits, 0L);
            long sHash = 0L;
            int prefixCost = 0;
            for (int j = 0; j < startPos; j++) {
                int col = cur[j];
                sBits[col >>> 6] |= 1L << col;
                sHash ^= zCol[col];
                prefixCost += costMatrix[j][col];
            }

            // the last row's child always has E == F (one free column, forbidden)
            int end = n;
            if (skipTrivialInfeasible && startPos <= n - 1) {
                end = n - 1;
                skippedInfeasible++;
            }

            boolean nodeHasExclusions = node.exclCol >= 0 || node.exclMulti != null;

            for (int i = startPos; i < end; i++) {
                if (i > startPos) {
                    // child i fixes one more column than child i-1: one XOR, one bit
                    int pc = cur[i - 1];
                    prefixCost += costMatrix[i - 1][pc];
                    sBits[pc >>> 6] |= 1L << pc;
                    sHash ^= zCol[pc];
                }
                int c = cur[i];

                // forbidden columns for row i in this child
                boolean multi = (i == startPos) && nodeHasExclusions;
                int[] childExcl = null;
                long eHash;
                int childExclCount;
                if (multi) {
                    int[] base = node.exclMulti != null ? node.exclMulti : new int[]{node.exclCol};
                    childExcl = insertSorted(base, c);
                    eHash = node.exclHash ^ wCol[c];
                    childExclCount = childExcl.length;
                } else {
                    eHash = wCol[c];
                    childExclCount = 1;
                }

                if (skipTrivialInfeasible && childExclCount >= n - i) {
                    skippedInfeasible++;
                    continue;
                }

                long h = sHash ^ eHash;
                ZobristStateTable.Entry hit = null;
                if (useCache) hit = table.find(h, sBits, multi ? -1 : c, childExcl, verify);

                FastState childState;
                int childTotalCost;
                boolean childMerged;
                int[] childAssignment;
                if (hit != null) {
                    cacheHits++;
                    childState = hit.state;
                    childTotalCost = prefixCost + childState.freeCost;
                    childAssignment = buildHitAssignment(cur, i, childState);
                    childMerged = false;
                } else {
                    cacheMisses++;
                    int costBound = pq.isFull() ? pq.peekMax().solution.cost - node.solution.cost : problem.infinity;
                    int[] excludedCols = multi ? childExcl : new int[]{c};

                    long stepStart = System.nanoTime();
                    childState = solver.solveStep(nodeState, i, excludedCols, i, prefixCost, costBound);
                    totalTime += System.nanoTime() - stepStart;
                    totalCalls++;

                    if (childState == null) {
                        prunedChildren++;
                        continue;
                    }
                    if (useCache) {
                        table.put(new ZobristStateTable.Entry(h, sBits.clone(), multi ? -1 : c, childExcl, childState));
                    }
                    childTotalCost = childState.cost;
                    childAssignment = childState.assignmentZeroIndexed();
                    childMerged = true;
                }

                if (childTotalCost >= problem.infinity) continue;

                AssignmentSolution childSolution = new AssignmentSolution(childAssignment, childTotalCost);
                FastZNode child = multi
                    ? new FastZNode(childSolution, childState, childMerged, i, -1, childExcl, eHash)
                    : new FastZNode(childSolution, childState, childMerged, i, c, null, eHash);

                if (pq.size() < k) {
                    pq.qInsert(child);
                } else if (childSolution.cost < pq.peekMax().solution.cost) {
                    pq.qReplaceMax(child);
                }
            }
        }
        printStats();
        return topK;
    }

    private static int[] insertSorted(int[] sorted, int v) {
        int[] out = new int[sorted.length + 1];
        int p = 0;
        while (p < sorted.length && sorted[p] < v) {
            out[p] = sorted[p];
            p++;
        }
        out[p] = v;
        System.arraycopy(sorted, p, out, p + 1, sorted.length - p);
        return out;
    }

    /** Reported assignment for a cache-hit child: rows [0, i) from the popped node, the rest from the cached solve. */
    private int[] buildHitAssignment(int[] cur, int i, FastState cached) {
        int n = cached.size;
        int[] assignment = new int[n];
        System.arraycopy(cur, 0, assignment, 0, i);
        for (int row = i + 1; row <= n; row++) {
            assignment[row - 1] = cached.rowToColumn[row] - 1;
        }
        return assignment;
    }

    /** Overwrites a cached state's fixed rows [0, startPos) with this branch's own. */
    private FastState mergeFixedRows(FastState cached, int[] cur, int startPos, int totalCost) {
        FastState merged = cached.copy();
        for (int j = 0; j < startPos; j++) {
            int row = j + 1;
            int col = cur[j] + 1;
            merged.rowToColumn[row] = col;
            merged.columnToRow[col] = row;
        }
        merged.fixedBoundary = startPos;
        merged.cost = totalCost;
        return merged;
    }

    public void printStats() {
        System.out.printf("cache hits: %d%n", cacheHits);
        System.out.printf("cache misses: %d%n", cacheMisses);
        System.out.printf("pruned children: %d%n", prunedChildren);
        System.out.printf("skipped (trivially infeasible): %d%n", skippedInfeasible);
        System.out.printf("table: %d entries, %.2f slots probed per lookup, 64-bit hash collisions caught by verify: %d%n",
            table.size(), table.lookups == 0 ? 0.0 : (double) table.probes / table.lookups, table.hashCollisions);
        System.out.printf("incremental solves: %d, total time: %.4fs%n", totalCalls, totalTime * 1e-9);
    }
}

/**
 * Open-addressing table (linear probing, load factor at most 0.5) keyed by a
 * 64-bit hash. An entry also stores the exact key (S bitmap and E) so a match
 * can be verified without rebuilding anything.
 */
final class ZobristStateTable {
    static final class Entry {
        final long hash;
        final long[] sBits;     // columns fixed to rows [0, i); never mutated
        final int exclCol;      // single forbidden column, or -1 when exclMulti is used
        final int[] exclMulti;  // sorted forbidden columns when there are two or more, else null
        final FastState state;

        Entry(long hash, long[] sBits, int exclCol, int[] exclMulti, FastState state) {
            this.hash = hash;
            this.sBits = sBits;
            this.exclCol = exclCol;
            this.exclMulti = exclMulti;
            this.state = state;
        }
    }

    private Entry[] slots = new Entry[1 << 12];
    private int mask = slots.length - 1;
    private int size;

    long lookups;
    long probes;
    long hashCollisions;   // equal 64-bit hash but different key (only detectable when verifying)

    int size() { return size; }

    private int index(long h) {
        return (int) (h ^ (h >>> 32)) & mask;
    }

    Entry find(long h, long[] sBits, int exclCol, int[] exclMulti, boolean verify) {
        lookups++;
        int idx = index(h);
        Entry e;
        while ((e = slots[idx]) != null) {
            probes++;
            if (e.hash == h) {
                if (!verify
                    || (e.exclCol == exclCol && Arrays.equals(e.sBits, sBits) && Arrays.equals(e.exclMulti, exclMulti))) {
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

    private void insert(Entry[] arr, int m, Entry e) {
        int idx = (int) (e.hash ^ (e.hash >>> 32)) & m;
        while (arr[idx] != null) idx = (idx + 1) & m;
        arr[idx] = e;
    }

    private void grow() {
        Entry[] bigger = new Entry[slots.length << 1];
        int newMask = bigger.length - 1;
        for (Entry e : slots) {
            if (e != null) insert(bigger, newMask, e);
        }
        slots = bigger;
        mask = newMask;
    }
}

/**
 * Queue node carrying the warm-start state and this subproblem's exclusions.
 * MurtyNode's inclusion/exclusion lists are left empty: fixed rows are the
 * prefix of solution.assignment.
 *
 *  startPos   number of fixed rows; also the one row that carries exclusions
 *  exclCol    the single forbidden column for row startPos, or -1 (root, or exclMulti used)
 *  exclMulti  sorted forbidden columns when there are two or more, else null; never mutated
 *  exclHash   XOR of wCol over the forbidden columns
 */
class FastZNode extends MurtyNode {
    FastState state;
    boolean merged;
    final int startPos;
    final int exclCol;
    final int[] exclMulti;
    final long exclHash;

    FastZNode(AssignmentSolution solution, FastState state, boolean merged,
              int startPos, int exclCol, int[] exclMulti, long exclHash) {
        super(solution, Collections.<int[]>emptyList(), Collections.<int[]>emptyList());
        this.state = state;
        this.merged = merged;
        this.startPos = startPos;
        this.exclCol = exclCol;
        this.exclMulti = exclMulti;
        this.exclHash = exclHash;
    }
}
