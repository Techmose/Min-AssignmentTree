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
 * empty one, (b) permanently wall off a set of "fixed" rows/columns
 * that must never be revisited, and (c) forbid one specific (row,column)
 * edge for the row currently being re-matched, and (d) stop early once
 * a cost bound is provably exceeded.
 *
 * Fixed rows can be described two ways:
 *   - a prefix 1..fixedBoundary (FastState.fixedRow == null), used by
 *     FastCacheEnumerator via solveStep, or
 *   - an arbitrary set (FastState.fixedRow != null), used by
 *     JavaFastmurty via solveStepFixedSet, whose fastmurty-style split
 *     order fixes rows in a data-dependent order rather than a prefix.
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
    // the non-fixed columns for the current search -- built once per
    // search so the inner loops only touch free columns (like
    // fastmurty's cols2use) instead of scanning all n every iteration
    int[] freeColumns;

    public FastHungarian(AssignmentProblem problem) {
        this.problem = problem;
        this.costs = problem.costMatrix;
        this.size = problem.numRows;
        this.visitedColumn = new boolean[size + 1];
        this.distanceToColumn = new int[size + 1];
        this.columnToLastColumn = new int[size + 1];
        this.freeColumns = new int[size + 1];
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
            augmentRow(st, row, NO_EXCLUSIONS, Long.MAX_VALUE);
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
     *                              same fixed rows -- used only to split
     *                              the total into (this branch's
     *                              fixed-row cost, the free submatrix's
     *                              own cost), so the latter can be cached
     * @param costBound             prune (return null) as soon as the
     *                              incremental cost increase is provably
     *                              &gt;= costBound -- mirrors spStep's
     *                              early "return inf"
     * @return a new, independent FastState for the child, or null if
     *         costBound was exceeded or the row has no allowed column
     */
    public FastState solveStep(FastState parent, int excludedRowZeroIdx,
                                int[] allExcludedColsZeroIdxForRow,
                                int newFixedBoundaryZeroIdx, int fixedRowsCost, int costBound) {
        FastState st = parent.copy();
        st.fixedRow = null; // prefix mode
        st.fixedBoundary = newFixedBoundaryZeroIdx;
        if (!stepInPlace(st, parent.cost, excludedRowZeroIdx, allExcludedColsZeroIdxForRow, costBound)) {
            return null;
        }
        // The free submatrix's own cost -- everything except what this
        // branch's specific fixed rows contribute -- is what's actually
        // safe to cache and hand to a different branch later.
        st.freeCost = st.cost - fixedRowsCost;
        return st;
    }

    /**
     * Same as solveStep, but the child's fixed rows are an arbitrary set
     * instead of a prefix. {@code fixedRowScratch} is indexed by 1-based
     * row (index 0 unused) and may be a reusable buffer: it is only read
     * during the search, and cloned into the returned state on success.
     */
    public FastState solveStepFixedSet(FastState parent, int excludedRowZeroIdx,
                                       int[] allExcludedColsZeroIdxForRow,
                                       boolean[] fixedRowScratch, int costBound) {
        FastState st = parent.copy();
        st.fixedRow = fixedRowScratch;
        if (!stepInPlace(st, parent.cost, excludedRowZeroIdx, allExcludedColsZeroIdxForRow, costBound)) {
            return null;
        }
        st.fixedRow = fixedRowScratch.clone();
        st.freeCost = st.cost;
        return st;
    }

    /** Shared body of solveStep / solveStepFixedSet. Returns false if pruned or infeasible. */
    private boolean stepInPlace(FastState st, int parentCost, int excludedRowZeroIdx,
                                int[] allExcludedColsZeroIdxForRow, int costBound) {
        int excludedRow = excludedRowZeroIdx + 1;

        // free the row's old column -- it must not be silently "stuck"
        // pointing back at a row that is about to be re-matched elsewhere
        int oldCol = st.rowToColumn[excludedRow];
        st.columnToRow[oldCol] = 0;
        st.rowToColumn[excludedRow] = 0;

        int[] excludedCols1Idx = new int[allExcludedColsZeroIdxForRow.length];
        for (int k = 0; k < excludedCols1Idx.length; k++) {
            excludedCols1Idx[k] = allExcludedColsZeroIdxForRow[k] + 1;
        }

        // With a full matching minus one row, the only unmatched column
        // is oldCol, so the augmenting path's total reduced length is
        // exactly the child's cost increase over the parent. That lets
        // the search stop the moment it reaches costBound, and gives the
        // child's cost without recomputing it over the whole matrix.
        int increase = augmentRow(st, excludedRow, excludedCols1Idx, costBound);
        if (increase == PRUNED) {
            return false;
        }

        // Infeasibility check: the search only PENALIZES the excluded edge(s)
        if (isExcludedForRow(excludedCols1Idx, st.rowToColumn[excludedRow])) {
            return false;
        }

        st.cost = parentCost + increase;
        return true;
    }

    private boolean isFixedColumn(FastState st, int column) {
        int row = st.columnToRow[column];
        if (row == 0) return false;
        return st.fixedRow != null ? st.fixedRow[row] : row <= st.fixedBoundary;
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
     * Hungarian.solveNext does). Columns claimed by a permanently fixed
     * row are never considered. Columns in {@code excludedCols1Idx} are
     * treated as infinitely costly for initialRow.
     *
     * @return the total reduced length of the augmenting path (the sum
     *         of every iteration's minDistance), or PRUNED as soon as
     *         that running total reaches costBound.
     */
    private int augmentRow(FastState st, int initialRow, int[] excludedCols1Idx, long costBound) {
        int n = size;
        Arrays.fill(visitedColumn, 0, n + 1, false);
        Arrays.fill(distanceToColumn, 0, n + 1, Integer.MAX_VALUE);

        int numFree = 0;
        for (int column = 1; column <= n; column++) {
            if (!isFixedColumn(st, column)) freeColumns[numFree++] = column;
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
                int rawCost = costRow[column - 1];
                if (row == initialRow && isExcludedForRow(excludedCols1Idx, column)) rawCost = problem.infinity;
                int reduced = rawCost - rowPrice - st.colPrice[column];
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
            if (total >= costBound) {
                return PRUNED;
            }

            // column 0 (the dummy) is always visited
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

        updateMatching(closestColumn, st);
        return (int) total;
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
 * directions, which rows are permanently pinned to this state's
 * assignment, and the total cost of the full assignment. Copied (never
 * mutated in place across nodes) so that every Murty node owns an
 * independent state it can safely use to warm-start its own children.
 */
class FastState {
    final int size;
    final int[] rowPrice;    // [0..size], index 0 unused
    final int[] colPrice;    // [0..size], index 0 unused
    final int[] columnToRow; // [0..size], columnToRow[j] = row assigned to column j, or 0 if free
    final int[] rowToColumn; // [0..size], rowToColumn[i] = column assigned to row i, or 0 if free
    int fixedBoundary;       // prefix mode: rows 1..fixedBoundary are permanently fixed
    boolean[] fixedRow;      // set mode (non-null): fixedRow[i] true if 1-based row i is fixed.
                             // Shared by reference on copy(); never mutated after being set.
    int cost;                // total cost of the full assignment, for THIS branch's specific fixed rows
    int freeCost;            // cost of just the free (unfixed) rows' assignments (cache use)

    FastState(int size) {
        this.size = size;
        this.rowPrice = new int[size + 1];
        this.colPrice = new int[size + 1];
        this.columnToRow = new int[size + 1];
        this.rowToColumn = new int[size + 1];
        this.fixedBoundary = 0;
        this.fixedRow = null;
        this.cost = 0;
    }

    FastState copy() {
        FastState c = new FastState(size);
        System.arraycopy(rowPrice, 0, c.rowPrice, 0, size + 1);
        System.arraycopy(colPrice, 0, c.colPrice, 0, size + 1);
        System.arraycopy(columnToRow, 0, c.columnToRow, 0, size + 1);
        System.arraycopy(rowToColumn, 0, c.rowToColumn, 0, size + 1);
        c.fixedBoundary = fixedBoundary;
        c.fixedRow = fixedRow;
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
