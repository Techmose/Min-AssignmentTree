import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * Enumerates the top-k solutions to an assignment problem using the
 * same tree-partitioning scheme as MurtyEnumerator/MurtyCacheEnumerator
 * (fix a prefix of rows, exclude the next row's current column), but:
 *
 *  1. Solves each child incrementally (FastHungarian.solveStep), warm
 *     started from its immediate parent's dual state, instead of
 *     rebuilding the cost matrix and re-running the full Hungarian
 *     algorithm -- this is fastmurty's actual speedup.
 *  2. Layers a subproblem-equivalence cache on top (same equivalence
 *     notion as MurtyCacheEnumerator: free columns + live exclusions).
 *     The cache stores the free submatrix's OWN cost and dual state --
 *     not the branch's total cost -- because the free submatrix's
 *     optimum is a pure function of the key (which free rows/columns,
 *     which live exclusions), whereas the total cost also depends on
 *     which specific columns THIS branch's fixed rows happen to occupy
 *     among the shared non-free set, which can differ between branches
 *     that reach the same key. A hit reconstructs the total as (this
 *     branch's own fixed-row cost) + (the cached free-submatrix cost),
 *     needing no re-solve and no full-matrix recomputation.
 *
 * MurtyQueue is used completely unmodified: it is typed to MurtyNode,
 * so FastMurtyNode (below) simply extends MurtyNode to carry the extra
 * per-node dual state the queue itself has no field for, and every
 * MurtyQueue method accepts it like any other MurtyNode.
 */
public class FastCacheEnumerator {
    AssignmentProblem problem;
    FastHungarian solver;

    // subproblem-equivalence cache: same notion as MurtyCacheEnumerator,
    // but storing only the free submatrix's own cost + dual state (see
    // class doc)
    Map<FastSubproblemKey, FastState> subproblemCache = new HashMap<>();

    // stats
    public int cacheHits = 0;
    public int cacheMisses = 0;
    public int prunedChildren = 0;
    public int totalCalls = 0;
    public long totalTime = 0;

    public FastCacheEnumerator(AssignmentProblem problem) {
        this.problem = problem;
        this.solver = new FastHungarian(problem);
    }

    /**
     * Enumerates the top-k solutions to the assignment problem.
     * @param k The number of solutions to enumerate.
     * @return A list of solutions to the assignment problem, best first.
     */
    public List<AssignmentSolution> enumerate(int k) {
        List<AssignmentSolution> topK = new ArrayList<>();
        MurtyQueue pq = new MurtyQueue(k);
        int[][] costMatrix = problem.costMatrix;

        // root: fully solve from scratch, no fixed rows, no exclusions
        long startTime = System.nanoTime();
        FastState rootState = solver.solveFull();
        totalTime += System.nanoTime() - startTime;
        totalCalls++;

        AssignmentSolution rootSolution = new AssignmentSolution(rootState.assignmentZeroIndexed(), rootState.cost);
        FastMurtyNode rootNode = new FastMurtyNode(rootSolution, new ArrayList<>(), new ArrayList<>(), rootState, true);
        pq.qInsert(rootNode);

        while (topK.size() < k && !pq.isEmpty()) {
            FastMurtyNode node = (FastMurtyNode) pq.qPopMin();
            topK.add(node.solution);

            // A node whose state came from a cache hit was left UNMERGED
            // at insertion time (see below) -- its state's fixed-row
            // portion still belongs to whichever branch discovered that
            // subproblem first. Only now, since this node was actually
            // chosen to be split into children, do we pay for merging in
            // THIS branch's own fixed-row assignment. Most inserted
            // candidates are later discarded or evicted by qReplaceMax
            // without ever being popped, so this defers -- and for the
            // majority of hits, entirely avoids -- that merge cost.
            FastState nodeState = node.state;
            if (!node.merged) {
                nodeState = mergeFixedRows(nodeState, node.inclusions, node.inclusions.size(), node.solution.cost);
                node.state = nodeState;
                node.merged = true;
            }

            int[] currentAssignment = node.solution.assignment;
            int n = currentAssignment.length;

            int startPos = 0;
            for (int[] inc : node.inclusions) {
                startPos = Math.max(startPos, inc[0] + 1);
            }

            // running total of costs[j][currentAssignment[j]] for j in
            // [0, startPos) -- the cost this whole popped node's fixed
            // prefix already contributes, before any of ITS children add
            // one more fixed row. Extended by one row per loop iteration
            // below rather than resummed from scratch each time.
            int prefixCost = 0;
            for (int j = 0; j < startPos; j++) {
                prefixCost += costMatrix[j][currentAssignment[j]];
            }

            for (int i = startPos; i < n; i++) {
                List<int[]> newInclusions = new ArrayList<>(node.inclusions);
                for (int j = startPos; j < i; j++) {
                    newInclusions.add(new int[]{j, currentAssignment[j]});
                }
                List<int[]> newExclusions = new ArrayList<>(node.exclusions);
                newExclusions.add(new int[]{i, currentAssignment[i]});

                // fixedRowsCost = cost of rows [0, i), i.e. prefixCost as
                // it stands at the top of this iteration (rows [startPos, i)
                // have already been folded in by earlier iterations below)
                int fixedRowsCost = prefixCost;

                long freeColumnsMask = computeFreeColumnsMask(newInclusions);
                long freeRowsMask = computeFreeRowsMask(newInclusions);
                Set<Long> liveExclusionKeys = computeLiveExclusionKeys(newExclusions, freeRowsMask);
                FastSubproblemKey key = new FastSubproblemKey(freeColumnsMask, liveExclusionKeys);

                FastState childState = subproblemCache.get(key);
                int childTotalCost;
                boolean childMerged;
                int[] childAssignment;
                if (childState != null) {
                    cacheHits++;
                    // Reuse the cached free-submatrix solve as-is -- its
                    // dual state and freeCost are already correct and
                    // branch-independent. The total cost is a plain O(1)
                    // sum. We deliberately do NOT clone/merge the state's
                    // fixed-row portion here: that's only needed if this
                    // child is later popped and split further, which most
                    // inserted candidates never are. We do still need the
                    // reported ASSIGNMENT ARRAY right now (it's the
                    // solution itself, and becomes `currentAssignment` if
                    // this node is popped later) -- but building just that
                    // one array is far cheaper than cloning all four of
                    // the state's internal dual/matching arrays.
                    childTotalCost = fixedRowsCost + childState.freeCost;
                    childAssignment = buildHitAssignment(newInclusions, childState);
                    childMerged = false;
                } else {
                    cacheMisses++;
                    int costBound = pq.isFull() ? pq.peekMax().solution.cost - node.solution.cost : problem.infinity;

                    // every column ever excluded for row i along this whole
                    // ancestor chain (there can be more than one -- the same
                    // row can be excluded again in a deeper descendant, once
                    // re-matched to a new column) must stay permanently
                    // blocked, not just the newest one
                    List<Integer> excludedForRow = new ArrayList<>();
                    for (int[] exc : newExclusions) {
                        if (exc[0] == i) excludedForRow.add(exc[1]);
                    }
                    int[] excludedCols = new int[excludedForRow.size()];
                    for (int idx = 0; idx < excludedCols.length; idx++) excludedCols[idx] = excludedForRow.get(idx);

                    long stepStart = System.nanoTime();
                    childState = solver.solveStep(nodeState, i, excludedCols, i, fixedRowsCost, costBound);
                    totalTime += System.nanoTime() - stepStart;
                    totalCalls++;

                    if (childState == null) {
                        prunedChildren++;
                        prefixCost += costMatrix[i][currentAssignment[i]];
                        continue; // infeasible, or provably worse than the current worst kept solution
                    }
                    subproblemCache.put(key, childState);
                    childTotalCost = childState.cost;
                    childAssignment = childState.assignmentZeroIndexed();
                    childMerged = true; // solveStep's result is already fully branch-specific
                }

                if (childTotalCost >= problem.infinity) {
                    prefixCost += costMatrix[i][currentAssignment[i]];
                    continue;
                }

                AssignmentSolution childSolution = new AssignmentSolution(childAssignment, childTotalCost);
                FastMurtyNode child = new FastMurtyNode(childSolution, newExclusions, newInclusions, childState, childMerged);

                if (pq.size() < k) {
                    pq.qInsert(child);
                } else if (childSolution.cost < pq.peekMax().solution.cost) {
                    pq.qReplaceMax(child);
                }
                // else: child is discarded -- worse than everything already kept

                prefixCost += costMatrix[i][currentAssignment[i]];
            }
        }
        return topK;
    }

    /**
     * Builds just the reported assignment array for a cache-hit child --
     * fixed rows from this branch's own newInclusions, free rows read
     * straight from the cached (unmerged, still someone-else's-branch)
     * state. Far cheaper than cloning and merging the full FastState,
     * and it's all a cache hit needs unless it's later popped.
     */
    private int[] buildHitAssignment(List<int[]> newInclusions, FastState cachedFreeState) {
        int n = cachedFreeState.size;
        int[] assignment = new int[n];
        for (int row = 1; row <= n; row++) {
            assignment[row - 1] = cachedFreeState.rowToColumn[row] - 1;
        }
        for (int[] inc : newInclusions) {
            assignment[inc[0]] = inc[1];
        }
        return assignment;
    }

    /**
     * @return a bitmask of columns NOT yet pinned down by any inclusion
     *         (bit j set means column j is free). Cheaper than a
     *         HashSet&lt;Integer&gt; -- no boxing, O(1)-ish equals/hashCode
     *         for the resulting cache key -- and fine as long as
     *         numCols &lt;= 64, true for any assignment problem this
     *         project's Hungarian solver can size arrays for anyway.
     */
    private long computeFreeColumnsMask(List<int[]> inclusions) {
        long mask = allOnes(this.problem.numCols);
        for (int[] inc : inclusions) mask &= ~(1L << inc[1]);
        return mask;
    }

    /** @return a bitmask of rows NOT yet pinned down by any inclusion */
    private long computeFreeRowsMask(List<int[]> inclusions) {
        long mask = allOnes(this.problem.numRows);
        for (int[] inc : inclusions) mask &= ~(1L << inc[0]);
        return mask;
    }

    private long allOnes(int bits) {
        return bits >= 64 ? -1L : (1L << bits) - 1;
    }

    /** @return live exclusions -- ones whose row is still free -- as an order-independent key */
    private Set<Long> computeLiveExclusionKeys(List<int[]> exclusions, long freeRowsMask) {
        Set<Long> keys = new HashSet<>();
        for (int[] exc : exclusions) {
            if ((freeRowsMask & (1L << exc[0])) != 0) keys.add(encodePair(exc[0], exc[1]));
        }
        return keys;
    }

    private long encodePair(int row, int col) {
        return (long) row * this.problem.numCols + col;
    }

    /**
     * Overwrites the fixed-row portion of a cached free-submatrix
     * solution with THIS branch's own fixed-row assignment (from
     * inclusions) so the resulting state is ready to warm-start THIS
     * branch's own children via solveStep. Called lazily, only once a
     * cache-hit-sourced node is actually popped for splitting (see
     * enumerate()) -- not at insertion time, since most inserted
     * candidates are discarded or evicted without ever needing this. The
     * free portion's dual state and cost are already correct as cached
     * and untouched; totalCost is supplied by the caller (already known
     * cheaply), so no matrix-wide recomputation happens here.
     */
    private FastState mergeFixedRows(FastState cached, List<int[]> inclusions, int fixedBoundaryZeroIdx, int totalCost) {
        FastState merged = cached.copy();
        for (int[] inc : inclusions) {
            int row = inc[0] + 1;
            int col = inc[1] + 1;
            merged.rowToColumn[row] = col;
            merged.columnToRow[col] = row;
        }
        merged.fixedBoundary = fixedBoundaryZeroIdx;
        merged.cost = totalCost;
        // merged.freeCost is left as cached.freeCost -- correct and
        // unaffected by which branch's fixed rows now sit alongside it
        return merged;
    }

    public void printStats() {
        System.out.printf("cache hits: %d%n", cacheHits);
        System.out.printf("cache misses: %d%n", cacheMisses);
        System.out.printf("pruned children: %d%n", prunedChildren);
        System.out.printf("hungarian/incremental solves: %d, total time: %.4fs%n", totalCalls, totalTime * 1e-9);
    }
}

/**
 * MurtyNode carries no field for this enumerator's extra per-node dual
 * state, and MurtyQueue is hardcoded to the concrete MurtyNode type --
 * but MurtyNode isn't final, so subclassing it lets every MurtyQueue
 * method (qInsert/qPopMin/qReplaceMax/peekMax) accept a FastMurtyNode
 * exactly like any other MurtyNode, with no change to MurtyNode.java or
 * MurtyQueue.java, and no separate identity map to maintain.
 *
 * `merged` is false exactly when `state` is a cache hit's raw,
 * unmerged free-submatrix state (still belongs to whichever branch
 * discovered that subproblem first) -- see enumerate()'s pop-time lazy
 * merge. It's true for the root, for any solveStep-produced (cache
 * miss) state, and for any cache-hit state that has already been
 * merged because this node was popped once already.
 */
class FastMurtyNode extends MurtyNode {
    FastState state;
    boolean merged;

    FastMurtyNode(AssignmentSolution solution, List<int[]> exclusions, List<int[]> inclusions, FastState state, boolean merged) {
        super(solution, exclusions, inclusions);
        this.state = state;
        this.merged = merged;
    }
}

/**
 * Equivalence key for a Murty tree node's induced subproblem: the set
 * of free (unforced) columns, together with the LIVE (row,col) cells
 * excluded on still-free rows. Two nodes with equal keys are guaranteed
 * to induce the identical constrained matrix -- see
 * MurtyCacheEnumerator.SubproblemKey for the original reference version
 * of this idea; this is a standalone copy so this file has no
 * dependency on MurtyCacheEnumerator.java being present.
 */
class FastSubproblemKey {
    final long freeColumnsMask;
    final Set<Long> exclusions;

    FastSubproblemKey(long freeColumnsMask, Set<Long> exclusions) {
        this.freeColumnsMask = freeColumnsMask;
        this.exclusions = exclusions;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof FastSubproblemKey)) return false;
        FastSubproblemKey other = (FastSubproblemKey) o;
        return this.freeColumnsMask == other.freeColumnsMask && this.exclusions.equals(other.exclusions);
    }

    @Override
    public int hashCode() {
        return 31 * Long.hashCode(freeColumnsMask) + exclusions.hashCode();
    }
}
