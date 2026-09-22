import java.util.Arrays;

/**
 * Fast Hungarian / successive-shortest-augmenting-path solver, in the
 * style of fastmurty (https://github.com/motrom/fastmurty), i.e. the
 * Miller/Stone/Cox optimizations to Murty's algorithm.
 *
 * The key idea (mirrors fastmurty's da.c + sspDense.c): a Murty tree
 * node's subproblem differs from its parent's by exactly one new
 * constraint -- one row loses its previously assigned column. Instead
 * of rebuilding the cost matrix and re-running the Hungarian algorithm
 * from scratch (O(n^3) per node), we copy the parent's dual variables
 * (row/column prices) and run a SINGLE incremental shortest-augmenting-
 * path search starting from just that one row (solveStep / spStep).
 * This also supports early cost-bound pruning, abandoning a candidate
 * child the moment its partial cost is already provably too high to
 * enter the top-k queue.
 *
 * This version assumes a SQUARE cost matrix with a guaranteed feasible
 * perfect matching -- AssignmentProblem already pads false-positive /
 * false-negative costs into such a square matrix -- so there is no
 * separate "missing row/column" bookkeeping the way fastmurty's native
 * rectangular solver has; every row is always matched to some column.
 *
 * Internally rows and columns are 1-indexed with 0 reserved as a
 * "dummy"/"unmatched" sentinel, matching the convention already used by
 * Hungarian.java in this project -- this file's augmenting-path search
 * is structurally the same algorithm as Hungarian.solveNext, generalized
 * to (a) start from an already-partially-solved state instead of an
 * empty one, (b) permanently wall off a prefix of "fixed" rows/columns
 * that must never be revisited, and (c) forbid one specific (row,column)
 * edge for the row currently being re-matched, and (d) stop early once
 * a cost bound is provably exceeded.
 */
public class FastHungarian {

    /** Sentinel returned by the internal search when costBound is exceeded. */
    static final int PRUNED = Integer.MIN_VALUE;

    AssignmentProblem problem;
    int[][] costs;
    int size;

    // reusable scratch space (per augmenting-path search call)
    boolean[] visitedColumn;
    int[] distanceToColumn;
    int[] columnToLastColumn;

    public FastHungarian(AssignmentProblem problem) {
        this.problem = problem;
        this.costs = problem.costMatrix;
        this.size = problem.numRows;
        this.visitedColumn = new boolean[size + 1];
        this.distanceToColumn = new int[size + 1];
        this.columnToLastColumn = new int[size + 1];
    }

    /**
     * Solves the root problem (every row and column free) completely
     * from scratch, exactly like Hungarian.solveHungarian, but leaves
     * behind the dual state (row/column prices) needed to warm-start
     * every descendant Murty subproblem.
     */
    private static final int[] NO_EXCLUSIONS = new int[0];

    public FastState solveFull() {
        FastState st = new FastState(size);
        for (int row = 1; row <= size; row++) {
            augmentRow(st, row, NO_EXCLUSIONS, Integer.MAX_VALUE);
        }
        st.cost = AssignmentProblem.cost(costs, st.assignmentZeroIndexed());
        st.freeCost = st.cost; // root: no rows fixed yet, so free cost == total cost
        return st;
    }

    /**
     * Incrementally solves a Murty child subproblem that differs from
     * {@code parent} by one new constraint: row {@code excludedRowZeroIdx}
     * may no longer be matched to column {@code excludedColZeroIdx} (its
     * assignment in the parent). {@code allExcludedColsZeroIdxForRow}
     * must contain every column ever excluded for this row along the
     * whole ancestor chain (including the new one) -- under this
     * partitioning scheme the same row can be excluded again in a
     * deeper descendant (once re-matched to a new column, that new
     * column can itself later be excluded too), and every one of those
     * historical exclusions has to stay permanently blocked, not just
     * the newest -- otherwise a later augmenting path could route the
     * row right back onto a column an ancestor already forbade.
     *
     * @param parent                the parent subproblem's solved state (never mutated)
     * @param excludedRowZeroIdx    0-indexed row that must be re-matched
     * @param allExcludedColsZeroIdxForRow  every 0-indexed column ever
     *                              forbidden for this row on this branch
     * @param newFixedBoundaryZeroIdx  0-indexed count of permanently fixed
     *                                 rows/columns in the child (rows
     *                                 [0, newFixedBoundaryZeroIdx) keep
     *                                 the parent's assignment forever)
     * @param fixedRowsCost         the total cost contributed by those
     *                              same fixed rows (sum of costs[row][its
     *                              assigned column] for row in
     *                              [0, newFixedBoundaryZeroIdx)) -- used
     *                              only to split the recomputed total
     *                              into (this branch's fixed-row cost,
     *                              the free submatrix's own cost), so the
     *                              latter can be cached and reused by a
     *                              future branch with different fixed
     *                              rows but the same free submatrix
     * @param costBound             prune (return null) as soon as the
     *                              incremental cost increase is provably
     *                              &gt;= costBound -- mirrors spStep's
     *                              early "return inf"
     * @return a new, independent FastState for the child, with its own
     *         copied dual state ready to warm-start its own children in
     *         turn -- or null if costBound was exceeded (the caller
     *         should treat the child as infeasible/dominated and skip it)
     */
    public FastState solveStep(FastState parent, int excludedRowZeroIdx,
                                int[] allExcludedColsZeroIdxForRow,
                                int newFixedBoundaryZeroIdx, int fixedRowsCost, int costBound) {
        FastState st = parent.copy();
        int excludedRow = excludedRowZeroIdx + 1;
        st.fixedBoundary = newFixedBoundaryZeroIdx;

        // free the row's old column -- it must not be silently "stuck"
        // pointing back at a row that is about to be re-matched elsewhere
        int oldCol = st.rowToColumn[excludedRow];
        st.columnToRow[oldCol] = 0;
        st.rowToColumn[excludedRow] = 0;

        int[] excludedCols1Idx = new int[allExcludedColsZeroIdxForRow.length];
        for (int k = 0; k < excludedCols1Idx.length; k++) {
            excludedCols1Idx[k] = allExcludedColsZeroIdxForRow[k] + 1;
        }

        // NOTE: we deliberately do not trust the search's own running
        // "minDistance" as the exact cost delta here. That identity
        // (final Dijkstra distance == total cost change) only holds
        // cleanly when the row being augmented starts from a fresh
        // (zero) price, as in a from-scratch build-up. Here the row
        // being re-matched already carries a nonzero warm-started price
        // from the parent, and its augmenting path can cascade through
        // other already-matched rows -- so we run the search to
        // completion (still far cheaper than a full Hungarian rebuild)
        // and get the exact cost by directly re-summing the resulting
        // assignment, which is always correct regardless of any of the
        // above. costBound pruning still happens, just after the search
        // instead of trying to bound it mid-search.
        augmentRow(st, excludedRow, excludedCols1Idx, Integer.MAX_VALUE);

        // Infeasibility check: the search only PENALIZES the excluded
        // edge(s) during the distance computation (a large sentinel), it
        // never actually removes them -- so when every other option is
        // unavailable (e.g. all other columns already permanently
        // claimed by fixed rows), the search can still "succeed" by
        // falling back to a forbidden edge. AssignmentProblem.cost()
        // would then silently read the TRUE (non-infinite) matrix value
        // there and report a bogus low cost that violates this branch's
        // exclusion constraint. Detect that here and report infeasible.
        if (isExcludedForRow(excludedCols1Idx, st.rowToColumn[excludedRow])) {
            return null;
        }

        st.cost = AssignmentProblem.cost(costs, st.assignmentZeroIndexed());
        if (st.cost >= costBound + parent.cost) {
            return null;
        }
        // The free submatrix's own cost -- everything except what this
        // branch's specific fixed rows contribute -- is what's actually
        // safe to cache and hand to a different branch later (see the
        // param doc above).
        st.freeCost = st.cost - fixedRowsCost;
        return st;
    }

    private boolean isFixedColumn(FastState st, int column) {
        int row = st.columnToRow[column];
        return row != 0 && row <= st.fixedBoundary;
    }

    private boolean isExcludedForRow(int[] excludedCols1Idx, int column) {
        for (int c : excludedCols1Idx) {
            if (c == column) return true;
        }
        return false;
    }

    /**
     * Runs one successive-shortest-augmenting-path search starting from
     * {@code initialRow}, updating st.rowPrice/colPrice/columnToRow/
     * rowToColumn in place to match that row (mutating them exactly as
     * Hungarian.solveNext does). Rows/columns already claimed by a
     * permanently fixed row (id &lt;= st.fixedBoundary) are never
     * considered. If {@code excludedCol} &gt;= 1, the direct
     * (initialRow, excludedCol) edge is treated as infinitely costly.
     *
     * @return the total incremental cost of matching initialRow (the
     *         algorithm's final "minDistance"), or PRUNED if the search
     *         proved it would exceed costBound.
     */
    private int augmentRow(FastState st, int initialRow, int[] excludedCols1Idx, int costBound) {
        int n = size;
        Arrays.fill(visitedColumn, 0, n + 1, false);
        Arrays.fill(distanceToColumn, 0, n + 1, Integer.MAX_VALUE);

        st.columnToRow[0] = initialRow;
        int closestColumn = 0;
        int minDistance;

        do {
            visitedColumn[closestColumn] = true;
            int row = st.columnToRow[closestColumn];

            for (int column = 1; column <= n; column++) {
                if (visitedColumn[column] || isFixedColumn(st, column)) continue;
                int rawCost = costs[row - 1][column - 1];
                if (row == initialRow && isExcludedForRow(excludedCols1Idx, column)) rawCost = problem.infinity;
                int reduced = rawCost - st.rowPrice[row] - st.colPrice[column];
                if (reduced < distanceToColumn[column]) {
                    distanceToColumn[column] = reduced;
                    columnToLastColumn[column] = closestColumn;
                }
            }

            minDistance = Integer.MAX_VALUE;
            closestColumn = -1;
            for (int column = 0; column <= n; column++) {
                if (visitedColumn[column] || (column != 0 && isFixedColumn(st, column))) continue;
                if (distanceToColumn[column] < minDistance) {
                    minDistance = distanceToColumn[column];
                    closestColumn = column;
                }
            }

            if (minDistance > costBound) {
                return PRUNED;
            }

            for (int column = 0; column <= n; column++) {
                if (column != 0 && isFixedColumn(st, column)) continue;
                if (visitedColumn[column]) {
                    st.rowPrice[st.columnToRow[column]] += minDistance;
                    st.colPrice[column] -= minDistance;
                } else {
                    distanceToColumn[column] -= minDistance;
                }
            }
        } while (st.columnToRow[closestColumn] != 0);

        updateMatching(closestColumn, st);
        return minDistance;
    }

    private void updateMatching(int column, FastState st) {
        do {
            int lastColumn = columnToLastColumn[column];
            int row = st.columnToRow[lastColumn];
            st.columnToRow[column] = row;
            st.rowToColumn[row] = column;
            column = lastColumn;
        } while (column != 0);
    }
}

/**
 * The persistent, warm-startable state of one Murty-tree subproblem:
 * the current row/column dual prices, the current matching in both
 * directions, how many rows (a prefix, 1..fixedBoundary) are
 * permanently pinned to this state's assignment, and the total cost of
 * the full assignment. Copied (never mutated in place across nodes) so
 * that every Murty node owns an independent state it can safely use to
 * warm-start its own children.
 */
class FastState {
    final int size;
    final int[] rowPrice;    // [0..size], index 0 unused
    final int[] colPrice;    // [0..size], index 0 unused
    final int[] columnToRow; // [0..size], columnToRow[j] = row assigned to column j, or 0 if free
    final int[] rowToColumn; // [0..size], rowToColumn[i] = column assigned to row i, or 0 if free
    int fixedBoundary;       // rows 1..fixedBoundary are permanently fixed
    int cost;                // total cost of the full assignment, for THIS branch's specific fixed rows
    int freeCost;            // cost of just the free (unfixed) rows' assignments -- a pure function
                              // of the equivalence key (free columns + live exclusions), safe to
                              // cache and reuse across any branch that reaches the same key,
                              // regardless of which columns that branch's own fixed rows occupy

    FastState(int size) {
        this.size = size;
        this.rowPrice = new int[size + 1];
        this.colPrice = new int[size + 1];
        this.columnToRow = new int[size + 1];
        this.rowToColumn = new int[size + 1];
        this.fixedBoundary = 0;
        this.cost = 0;
    }

    FastState copy() {
        FastState c = new FastState(size);
        System.arraycopy(rowPrice, 0, c.rowPrice, 0, size + 1);
        System.arraycopy(colPrice, 0, c.colPrice, 0, size + 1);
        System.arraycopy(columnToRow, 0, c.columnToRow, 0, size + 1);
        System.arraycopy(rowToColumn, 0, c.rowToColumn, 0, size + 1);
        c.fixedBoundary = fixedBoundary;
        c.cost = cost;
        c.freeCost = freeCost;
        return c;
    }

    /** 0-indexed row-&gt;column assignment array, as used elsewhere in this project. */
    int[] assignmentZeroIndexed() {
        int[] a = new int[size];
        for (int row = 1; row <= size; row++) {
            a[row - 1] = rowToColumn[row] - 1;
        }
        return a;
    }
}
