import java.util.ArrayList;
import java.util.List;

/**
 * JavaFastmurty -- a Java version of fastmurty's enumerator
 * (github.com/motrom/fastmurty, da.c + murtysplitDense.c), with no
 * subproblem cache.
 *
 * Like FastCacheEnumerator it warm-starts every child from its parent's
 * dual state with one incremental augmenting-path search
 * (FastHungarian), and passes a cost bound so a child that cannot
 * enter the top-k is abandoned mid-search. What it adds is fastmurty's
 * split order:
 *
 *   When a node is popped, its free rows are not split in index order.
 *   For every free row, a lookahead estimate of how much the cost would
 *   rise if that row lost its current column is computed (the smallest
 *   reduced cost among its other available columns). Rows are then
 *   placed from the end backwards, always picking the largest remaining
 *   estimate, so:
 *     - rows that are expensive to change get the children with the
 *       MOST free rows (big subproblems that are unlikely to beat the
 *       k-th best, so the cost bound usually cuts them short), and
 *     - rows that are cheap to change get the children with the FEWEST
 *       free rows (small, cheap subproblems that are likely to be
 *       popped later and split further).
 *   The row that carries this node's own exclusions is always placed
 *   last, so the one child with every row free inherits them and no
 *   other child needs them.
 *
 *   With order o_0..o_{m-1}, child p re-matches row o_p (excluding its
 *   current column), keeps rows o_0..o_{p-1} free, and fixes rows
 *   o_{p+1}..o_{m-1} to their current columns.
 *
 * Because the fixed rows are no longer a prefix, each node's fixed set
 * is stored as a boolean array on its FastState.
 *
 * Purpose: a control for FastCacheEnumerator. Comparing the two isolates
 * the cost/benefit of caching; comparing this against fastmurty (C)
 * checks FastHungarian and the Java overhead.
 */
public class JavaFastmurty {
    AssignmentProblem problem;
    FastHungarian solver;

    // reusable per-split scratch
    int[] order;
    int[] estimate;
    int[] bestColumn;
    boolean[] columnAvailable;

    // stats
    public int prunedChildren = 0;
    public int totalCalls = 0;
    public long totalTime = 0;

    public JavaFastmurty(AssignmentProblem problem) {
        this.problem = problem;
        this.solver = new FastHungarian(problem);
        int n = problem.numRows;
        this.order = new int[n];
        this.estimate = new int[n];
        this.bestColumn = new int[n];
        this.columnAvailable = new boolean[n];
    }

    /**
     * Enumerates the top-k solutions to the assignment problem.
     * @param k The number of solutions to enumerate.
     * @return A list of solutions to the assignment problem, best first.
     */
    public List<AssignmentSolution> enumerate(int k) {
        List<AssignmentSolution> topK = new ArrayList<>();
        MurtyQueue pq = new MurtyQueue(k);
        int n = problem.numRows;

        // root: fully solve from scratch, no fixed rows, no exclusions
        long startTime = System.nanoTime();
        FastState rootState = solver.solveFull();
        totalTime += System.nanoTime() - startTime;
        totalCalls++;
        rootState.fixedRow = new boolean[n + 1];

        AssignmentSolution rootSolution = new AssignmentSolution(rootState.assignmentZeroIndexed(), rootState.cost);
        pq.qInsert(new JavaFastmurtyNode(rootSolution, rootState, -1, new int[0]));

        while (topK.size() < k && !pq.isEmpty()) {
            JavaFastmurtyNode node = (JavaFastmurtyNode) pq.qPopMin();
            topK.add(node.solution);
            if (topK.size() == k) break;

            FastState nodeState = node.state;
            int[] assignment = node.solution.assignment;

            int m = splitOrder(node);
            if (m < 2) continue; // a single free row has no alternative column

            // fixed-row buffer for child p: parent's fixed rows plus
            // order[p+1..m-1]. Children go p = 1, 2, ..., m-1, so start
            // with order[1..m-1] fixed and free one more row each time.
            // (Child p = 0 is skipped: its only free row would be
            // excluded from its only free column, always infeasible.)
            boolean[] fixedScratch = nodeState.fixedRow.clone();
            for (int p = 1; p < m; p++) fixedScratch[order[p] + 1] = true;

            for (int p = 1; p < m; p++) {
                int row = order[p];
                fixedScratch[row + 1] = false;

                // the row carrying this node's exclusions keeps them all
                int[] excludedCols;
                if (row == node.splitRow) {
                    excludedCols = new int[node.splitExcluded.length + 1];
                    System.arraycopy(node.splitExcluded, 0, excludedCols, 0, node.splitExcluded.length);
                    excludedCols[node.splitExcluded.length] = assignment[row];
                } else {
                    excludedCols = new int[]{assignment[row]};
                }

                int costBound = pq.isFull() ? pq.peekMax().solution.cost - node.solution.cost : problem.infinity;

                long stepStart = System.nanoTime();
                FastState childState = solver.solveStepFixedSet(nodeState, row, excludedCols, fixedScratch, costBound);
                totalTime += System.nanoTime() - stepStart;
                totalCalls++;

                if (childState == null || childState.cost >= problem.infinity) {
                    prunedChildren++;
                    continue; // infeasible, or provably worse than the current worst kept solution
                }

                AssignmentSolution childSolution = new AssignmentSolution(childState.assignmentZeroIndexed(), childState.cost);
                JavaFastmurtyNode child = new JavaFastmurtyNode(childSolution, childState, row, excludedCols);

                if (pq.size() < k) {
                    pq.qInsert(child);
                } else if (childSolution.cost < pq.peekMax().solution.cost) {
                    pq.qReplaceMax(child);
                }
                // else: child is discarded -- worse than everything already kept
            }
        }
        return topK;
    }

    /**
     * Fills order[0..m-1] with the node's free rows in fastmurty's split
     * order (see class doc) and returns m.
     */
    private int splitOrder(JavaFastmurtyNode node) {
        FastState st = node.state;
        int n = problem.numRows;
        int[] a = node.solution.assignment;
        int[][] c = problem.costMatrix;

        // lookahead rows: free rows other than the one carrying exclusions
        int m2 = 0;
        for (int r = 0; r < n; r++) {
            if (!st.fixedRow[r + 1] && r != node.splitRow) order[m2++] = r;
        }
        int m = m2 + (node.splitRow >= 0 ? 1 : 0);

        // available lookahead columns: free rows' columns, minus the
        // exclusion-carrying row's column
        java.util.Arrays.fill(columnAvailable, false);
        for (int idx = 0; idx < m2; idx++) columnAvailable[a[order[idx]]] = true;

        for (int idx = 0; idx < m2; idx++) computeEstimate(idx, a, c, st);

        // place rows from position m2-1 down to 0, largest estimate first
        for (int pos = m2 - 1; pos >= 0; pos--) {
            int maxIdx = 0;
            for (int idx = 1; idx <= pos; idx++) {
                if (estimate[idx] > estimate[maxIdx]) maxIdx = idx;
            }
            swap(maxIdx, pos);

            // that row is fixed in every smaller child, so its column is
            // no longer an alternative for the rows still to be placed
            int deadCol = a[order[pos]];
            columnAvailable[deadCol] = false;
            for (int idx = 0; idx < pos; idx++) {
                if (bestColumn[idx] == deadCol) computeEstimate(idx, a, c, st);
            }
        }

        if (node.splitRow >= 0) order[m2] = node.splitRow;
        return m;
    }

    /** estimate[idx] = smallest reduced cost of row order[idx] over its other available columns. */
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
        System.out.printf("pruned children: %d%n", prunedChildren);
        System.out.printf("incremental solves: %d, total time: %.4fs%n", totalCalls, totalTime * 1e-9);
    }
}

/**
 * Queue node for JavaFastmurty. MurtyQueue only compares by
 * solution.cost, so the inclusion/exclusion lists MurtyNode carries are
 * left null; this node instead records which row it was created by
 * re-matching (splitRow, -1 for the root) and every column excluded for
 * that row along its branch (splitExcluded). Those are its only live
 * exclusions -- every earlier split row is now fixed.
 */
class JavaFastmurtyNode extends MurtyNode {
    FastState state;
    int splitRow;
    int[] splitExcluded;

    JavaFastmurtyNode(AssignmentSolution solution, FastState state, int splitRow, int[] splitExcluded) {
        super(solution, null, null);
        this.state = state;
        this.splitRow = splitRow;
        this.splitExcluded = splitExcluded;
    }
}
