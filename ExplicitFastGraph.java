import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.SplittableRandom;

/**
 * ExplicitFastGraph -- OrderGraphEXPEnumerator's search with JavaFastmurty's
 * solver speedups.
 *
 * Search (same as OrderGraphEXPEnumerator)
 *   Rows are assigned in order 0, 1, ..., n-1. A queue entry is a path
 *   (columns chosen for rows 0..d-1) with priority
 *       pathCost + value(S)
 *   where S is the set of columns the path uses and value(S) is the
 *   optimal cost of the remaining submatrix (rows d..n-1, columns not in
 *   S). value(S) depends only on the set S, not on the order the path
 *   took, so it is stored once per graph node and shared by every path
 *   that reaches S: through the node's children array first, then through
 *   a table keyed by S. A path of length n is a complete solution.
 *
 * Speedups (from JavaFastmurty / FastHungarian)
 *   1. Warm start. Each graph node keeps the solved FastState of its
 *      subproblem: rows 0..d-1 sit on S (any bijection), rows d..n-1 are
 *      optimally matched with feasible duals. For child S + {c}:
 *        - if the node's own optimal solution already gives row d column
 *          c, the child is solved for free: value = value(S) - C[d][c],
 *          and the child shares the parent's state;
 *        - otherwise row d is pinned to c, the row r that held c is left
 *          unmatched, and ONE augmenting-path search from r (the parent's
 *          duals stay feasible on the smaller problem) gives the child's
 *          optimum. No submatrix is built and no full Hungarian is run.
 *   2. Cost bound. When the queue is full, the augmenting-path search
 *      stops as soon as pathCost + value(S + {c}) provably reaches the
 *      current k-th best. Such a child cannot enter the queue from this
 *      path; it is skipped and NOT cached, since its exact value is
 *      unknown. A later path that reaches the same S re-solves it.
 *   3. Path-cost skip (already in OrderGraphEXPEnumerator): a child whose
 *      path cost alone reaches the k-th best is skipped before any lookup.
 *
 * The split order from JavaFastmurty does not apply here: the order graph
 * needs a fixed row order so that a column set identifies a subproblem.
 *
 * Value identity for the augmenting-path step (u, v = parent's duals,
 * a_d = parent's column for row d, r = row that held c, L = the
 * augmenting path's total reduced length):
 *     value(S + {c}) = value(S) - C[d][a_d] - C[r][c] + u[r] + v[a_d] + L
 * since every Dijkstra iteration raises (sum of free-row u + free-column
 * v) by exactly its step length, and the final matching is tight.
 */
public class ExplicitFastGraph implements Enumerator {
    AssignmentProblem problem;
    FastHungarian rootSolver;
    final long[] zCol;
    final ExplicitGraphTable table = new ExplicitGraphTable();

    // augmenting-path scratch
    final boolean[] visitedColumn;
    final int[] distanceToColumn;
    final int[] columnToLastColumn;
    final int[] freeColumns;

    /** recompute every solved value from scratch and compare (debug only) */
    public boolean checkValues = false;

    // stats
    public long childLinkHits = 0;   // found through the parent's children array
    public long cacheHits = 0;       // found in the table (a different parent solved it)
    public long cacheMisses = 0;
    public long freeChildren = 0;    // solved with no search
    public long boundSkipped = 0;    // path cost alone too high
    public long prunedChildren = 0;  // search stopped by the cost bound (not cached)
    public long totalCalls = 0;      // augmenting-path searches run to completion or pruned
    public long totalTime = 0;

    public ExplicitFastGraph(AssignmentProblem problem) {
        this(problem, 0x9E3779B97F4A7C15L);
    }

    public ExplicitFastGraph(AssignmentProblem problem, long seed) {
        this.problem = problem;
        this.rootSolver = new FastHungarian(problem);
        int n = problem.numRows;
        SplittableRandom rng = new SplittableRandom(seed);
        this.zCol = new long[n];
        for (int c = 0; c < n; c++) {
            long v;
            do { v = rng.nextLong(); } while (v == 0L);
            zCol[c] = v;
        }
        this.visitedColumn = new boolean[n + 1];
        this.distanceToColumn = new int[n + 1];
        this.columnToLastColumn = new int[n + 1];
        this.freeColumns = new int[n + 1];
    }

    public List<AssignmentSolution> enumerate(int k) {
        List<AssignmentSolution> topK = new ArrayList<>();
        int n = problem.numRows;
        int[][] c = problem.costMatrix;
        MurtyQueue pq = new MurtyQueue(k);

        long t0 = System.nanoTime();
        FastState rootState = rootSolver.solveFull();
        totalTime += System.nanoTime() - t0;
        totalCalls++;

        ExplicitGraphNode root = new ExplicitGraphNode(new long[(n + 63) >>> 6], 0L, 0, rootState.cost, rootState);
        table.put(root);
        pq.qInsert(new ExplicitGraphEntry(rootState.cost, OrderGraphPath.emptyPath(), 0, root));

        while (topK.size() < k && !pq.isEmpty()) {
            ExplicitGraphEntry entry = (ExplicitGraphEntry) pq.qPopMin();
            if (entry.length == n) {
                // complete path
                topK.add(entry.toSolution());
                pq.maxSize--;
                continue;
            }

            ExplicitGraphNode og = entry.og;
            int d = og.depth;
            FastState st = og.state;
            int ownCol = st.rowToColumn[d + 1] - 1;   // parent's optimal column for row d
            int[] costRow = c[d];
            long[] cols = og.cols;
            if (og.children == null) og.children = new ExplicitGraphNode[n - d];

            int childIndex = 0;
            for (int col = 0; col < n; col++) {
                long bit = 1L << col;
                int w = col >>> 6;
                if ((cols[w] & bit) != 0) continue;
                int idx = childIndex++;

                int pathCost = entry.pathCost + costRow[col];
                if (pq.isFull() && pathCost >= pq.peekMax().solution.cost) {
                    boundSkipped++;
                    continue;
                }

                ExplicitGraphNode child = og.children[idx];
                if (child != null) {
                    childLinkHits++;
                } else {
                    long childHash = og.hash ^ zCol[col];
                    cols[w] |= bit;   // look up S + {col} without allocating
                    child = table.find(childHash, cols);
                    cols[w] &= ~bit;
                    if (child != null) {
                        cacheHits++;
                    } else {
                        cacheMisses++;
                        int bound = pq.isFull() ? pq.peekMax().solution.cost - pathCost : Integer.MAX_VALUE;
                        child = solveChild(og, col, ownCol, childHash, bound);
                        if (child == null) {
                            prunedChildren++;
                            continue;   // cannot beat the k-th best from here; not cached
                        }
                        table.put(child);
                    }
                    og.children[idx] = child;
                }

                int newCost = pathCost + child.value;
                ExplicitGraphEntry next = new ExplicitGraphEntry(newCost, entry.path.append(col), pathCost, child);
                if (!pq.isFull()) {
                    pq.qInsert(next);
                } else if (pq.maxSize() != 0 && next.compareTo(pq.peekMax()) < 0) {
                    pq.qReplaceMax(next);
                }
            }
        }
        printStats();
        return topK;
    }

    /**
     * Solves subproblem S + {col} from the parent node's state.
     * @param bound stop once value(S + {col}) reaches this (Integer.MAX_VALUE = no bound)
     * @return the new graph node, or null if the bound was reached
     */
    private ExplicitGraphNode solveChild(ExplicitGraphNode og, int col, int ownCol, long childHash, int bound) {
        int d = og.depth;
        int[][] c = problem.costMatrix;
        FastState st = og.state;
        long[] childCols = og.cols.clone();
        childCols[col >>> 6] |= 1L << col;

        if (col == ownCol) {
            // the parent's optimum already uses (d, col): its restriction is optimal
            freeChildren++;
            int value = og.value - c[d][col];
            return new ExplicitGraphNode(childCols, childHash, d + 1, value, st);
        }

        int dRow = d + 1;                       // 1-based row being pinned
        int r = st.columnToRow[col + 1];        // 1-based row that loses col
        int base = og.value - c[d][ownCol] - c[r - 1][col] + st.rowPrice[r] + st.colPrice[ownCol + 1];

        long searchBound = bound == Integer.MAX_VALUE ? Long.MAX_VALUE : (long) bound - base;
        if (searchBound <= 0) return null;   // even a zero-length path would reach the bound

        long t0 = System.nanoTime();
        FastState cs = st.copy();
        cs.fixedRow = null;
        cs.fixedBoundary = dRow;                // rows 1..d+1 fixed
        cs.rowToColumn[dRow] = col + 1;
        cs.columnToRow[col + 1] = dRow;
        cs.columnToRow[ownCol + 1] = 0;
        cs.rowToColumn[r] = 0;
        long length = augment(cs, r, searchBound);
        totalTime += System.nanoTime() - t0;
        totalCalls++;
        if (length < 0) return null;

        int value = base + (int) length;
        if (checkValues) {
            int direct = 0;
            for (int row = dRow + 1; row <= problem.numRows; row++) direct += c[row - 1][cs.rowToColumn[row] - 1];
            int expected = exactValue(subMatrix(childCols, d + 1));
            if (direct != value || expected != value) {
                throw new IllegalStateException("value mismatch: formula=" + value + " direct=" + direct + " exact=" + expected);
            }
        }
        cs.cost = value;
        return new ExplicitGraphNode(childCols, childHash, d + 1, value, cs);
    }

    /**
     * One successive-shortest-path search from 1-based row {@code initialRow}
     * over the columns not held by rows 1..fixedBoundary. Same search as
     * FastHungarian.augmentRow, without exclusions.
     * @return the path's total reduced length, or -1 once it reaches costBound
     */
    private long augment(FastState st, int initialRow, long costBound) {
        int n = problem.numRows;
        int[][] costs = problem.costMatrix;
        int boundary = st.fixedBoundary;
        Arrays.fill(visitedColumn, false);
        Arrays.fill(distanceToColumn, Integer.MAX_VALUE);

        int numFree = 0;
        for (int column = 1; column <= n; column++) {
            int row = st.columnToRow[column];
            if (row == 0 || row > boundary) freeColumns[numFree++] = column;
        }

        st.columnToRow[0] = initialRow;
        int closestColumn = 0;
        long total = 0;
        do {
            visitedColumn[closestColumn] = true;
            int row = st.columnToRow[closestColumn];
            int[] costRow = costs[row - 1];
            int rowPrice = st.rowPrice[row];
            for (int f = 0; f < numFree; f++) {
                int column = freeColumns[f];
                if (visitedColumn[column]) continue;
                int reduced = costRow[column - 1] - rowPrice - st.colPrice[column];
                if (reduced < distanceToColumn[column]) {
                    distanceToColumn[column] = reduced;
                    columnToLastColumn[column] = closestColumn;
                }
            }

            int minDistance = Integer.MAX_VALUE;
            closestColumn = -1;
            for (int f = 0; f < numFree; f++) {
                int column = freeColumns[f];
                if (visitedColumn[column]) continue;
                if (distanceToColumn[column] < minDistance) {
                    minDistance = distanceToColumn[column];
                    closestColumn = column;
                }
            }

            total += minDistance;
            if (total >= costBound) return -1;

            st.rowPrice[st.columnToRow[0]] += minDistance;
            st.colPrice[0] -= minDistance;
            for (int f = 0; f < numFree; f++) {
                int column = freeColumns[f];
                if (visitedColumn[column]) {
                    st.rowPrice[st.columnToRow[column]] += minDistance;
                    st.colPrice[column] -= minDistance;
                } else {
                    distanceToColumn[column] -= minDistance;
                }
            }
        } while (st.columnToRow[closestColumn] != 0);

        int column = closestColumn;
        do {
            int lastColumn = columnToLastColumn[column];
            int row = st.columnToRow[lastColumn];
            st.columnToRow[column] = row;
            st.rowToColumn[row] = column;
            column = lastColumn;
        } while (column != 0);
        return total;
    }

    private static int exactValue(int[][] m) {
        return new Hungarian(new AssignmentProblem(m)).solve(m).cost;
    }

    /** Remaining submatrix for debug checks: rows [depth, n), columns not in cols. */
    private int[][] subMatrix(long[] cols, int depth) {
        int n = problem.numRows;
        int left = n - depth;
        int[][] m = new int[left][left];
        for (int i = 0; i < left; i++) {
            int j = 0;
            for (int col = 0; col < n; col++) {
                if ((cols[col >>> 6] & (1L << col)) != 0) continue;
                m[i][j++] = problem.costMatrix[depth + i][col];
            }
        }
        return m;
    }

    public void printStats() {
        System.out.printf("graph nodes: %d%n", table.size());
        System.out.printf("child-link hits: %d, cache hits: %d, cache misses: %d%n", childLinkHits, cacheHits, cacheMisses);
        System.out.printf("free children: %d, skipped by path cost: %d, pruned by bound (not cached): %d%n",
            freeChildren, boundSkipped, prunedChildren);
        System.out.printf("searches: %d, total time: %.4fs%n", totalCalls, totalTime * 1e-9);
    }
}

/**
 * A node of the order graph: the set S of used columns and the optimal
 * cost of the remaining submatrix. Shared by every path that uses S.
 */
final class ExplicitGraphNode {
    final long[] cols;          // S as a bitmap; never mutated outside a lookup
    final long hash;            // XOR of zCol over S
    final int depth;            // |S| = rows already assigned
    final int value;            // optimal cost of rows depth..n-1 on the columns not in S
    final FastState state;      // solved state for warm-starting children; may be shared with the parent
    ExplicitGraphNode[] children;   // indexed by rank among free columns; allocated on first expansion

    ExplicitGraphNode(long[] cols, long hash, int depth, int value, FastState state) {
        this.cols = cols;
        this.hash = hash;
        this.depth = depth;
        this.value = value;
        this.state = state;
    }
}

/** Open-addressing table of graph nodes keyed by the column set's Zobrist hash. */
final class ExplicitGraphTable {
    private ExplicitGraphNode[] slots = new ExplicitGraphNode[1 << 12];
    private int mask = slots.length - 1;
    private int size;

    int size() { return size; }

    private static int index(long h, int m) {
        return (int) (h ^ (h >>> 32)) & m;
    }

    ExplicitGraphNode find(long h, long[] cols) {
        int idx = index(h, mask);
        ExplicitGraphNode e;
        while ((e = slots[idx]) != null) {
            if (e.hash == h && Arrays.equals(e.cols, cols)) return e;
            idx = (idx + 1) & mask;
        }
        return null;
    }

    void put(ExplicitGraphNode e) {
        if ((size + 1) * 2 > slots.length) grow();
        insert(slots, mask, e);
        size++;
    }

    private static void insert(ExplicitGraphNode[] arr, int m, ExplicitGraphNode e) {
        int idx = index(e.hash, m);
        while (arr[idx] != null) idx = (idx + 1) & m;
        arr[idx] = e;
    }

    private void grow() {
        ExplicitGraphNode[] bigger = new ExplicitGraphNode[slots.length << 1];
        int newMask = bigger.length - 1;
        for (ExplicitGraphNode e : slots) if (e != null) insert(bigger, newMask, e);
        slots = bigger;
        mask = newMask;
    }
}

/**
 * Queue entry: a path through the order graph. Extends MurtyNode so it can
 * use MurtyQueue; solution.cost holds the priority (pathCost + node value)
 * and solution.assignment is only built when the path is complete.
 * Ties go to the longer path (closer to a solution), then to the older entry.
 */
final class ExplicitGraphEntry extends MurtyNode {
    private static long idCounter = 0;
    final ExplicitGraphNode og;
    final OrderGraphPath path;
    final int pathCost;
    final int length;
    final long id;

    ExplicitGraphEntry(int cost, OrderGraphPath path, int pathCost, ExplicitGraphNode og) {
        super(new AssignmentSolution(null, cost), null, null);
        this.og = og;
        this.path = path;
        this.pathCost = pathCost;
        this.length = path.size();
        this.id = idCounter++;
    }

    @Override
    public int compareTo(MurtyNode o) {
        ExplicitGraphEntry other = (ExplicitGraphEntry) o;
        int a = solution.cost, b = other.solution.cost;
        if (a != b) return a < b ? -1 : 1;
        if (length != other.length) return length > other.length ? -1 : 1;
        return Long.compare(id, other.id);
    }

    AssignmentSolution toSolution() {
        int[] sol = new int[length];
        int i = length - 1;
        OrderGraphPath p = path;
        while (!p.isEnd()) {
            sol[i--] = p.value();
            p = p.next();
        }
        solution.assignment = sol;
        return solution;
    }
}
