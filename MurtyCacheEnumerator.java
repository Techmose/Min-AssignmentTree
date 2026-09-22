import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;

/**
 * A variant of MurtyEnumerator that layers a *subproblem-equivalence*
 * cache on top of the ordinary Murty tree expansion. It performs the
 * exact same tree search as MurtyEnumerator -- it still calls Hungarian
 * on every generated node, so the cache is NEVER used to skip work or
 * speed anything up. Its only purpose is instrumentation: it lets us
 * compare the Murty *tree* (every node ever created, including
 * duplicates) against the underlying Murty *graph* (distinct
 * subproblems, with duplicates merged).
 *
 * Node equivalence used here:
 *   Two tree nodes are considered the same subproblem if they agree on
 *     - freeColumns: the set of columns not yet pinned down by any
 *       inclusion (i.e. the columns Hungarian is still free to choose
 *       from), and
 *     - exclusions: only the LIVE exclusions -- (row,col) pairs this
 *       node has forbidden where the row is still free (not yet locked
 *       in by an inclusion). Exclusions on rows that have since been
 *       forced are dead weight (the inclusion already blocks every
 *       other cell in that row) and are deliberately dropped from the
 *       key: keeping them would tie two nodes' identity to their
 *       ancestor history, when what actually determines the matrix
 *       Hungarian solves is only the live restriction.
 *   This does NOT check *how* a node arrived at that state (order of
 *   inclusions/exclusions doesn't matter, and neither does the dead
 *   exclusion history), only the resulting live restriction. Two
 *   different tree branches that happen to produce the same
 *   (freeColumns, live exclusions) pair are treated as one graph node.
 *
 * One deliberate exception: trivial single-row/single-column
 * subproblems (a 1x1 "submatrix") are never added to the cache/graph at
 * all, regardless of how many times they recur -- see recordSubproblem.
 * They are excluded from BOTH the tree and the graph counts entirely; a
 * separate, purely informational tally of how many were skipped is
 * kept alongside (see singleCellNodeCount / SubproblemCacheSummary),
 * but it is not folded into any reported Tree or Graph number.
 */
public class MurtyCacheEnumerator {
    AssignmentProblem problem;
    Hungarian hungarian;
    // reusable space for cost matrix
    int[][] scratchMatrix;

    // stats for algorithm performance (kept for parity with
    // MurtyEnumerator; superseded by the subproblem cache stats below)
    int cacheHits;
    int cacheMisses;
    int totalCalls;
    long totalTime;

    // --- subproblem-equivalence cache ---
    // keyed by (freeColumns, exclusions); values assigned a sequential
    // "discovery index" the first time that subproblem is seen, and a
    // hitCount every time it recurs elsewhere in the tree.
    // This is the big, per-run data structure -- it can grow to
    // millions of entries for large n, so it gets dumped (see
    // summarizeAndDumpSubproblemCache) once its counts have been
    // extracted, rather than held onto indefinitely.
    Map<SubproblemKey, SubproblemNode> subproblemCache = new LinkedHashMap<>();
    int nextSubproblemIndex = 0;

    // --- excluded trivial subproblems ---
    // Single-row/single-column subproblems (a 1x1 "submatrix") are
    // deliberately NOT added to subproblemCache -- they add no useful
    // structure to the graph. They're still tallied here so the tree
    // total remains accurate; see recordSubproblem and
    // summarizeAndDumpSubproblemCache.
    int singleCellNodeCount = 0;
    int singleCellSolvableCount = 0;
    int singleCellUnsolvableCount = 0;

    // --- run history ---
    // one lightweight SubproblemCacheSummary per completed run (i.e.
    // per call to summarizeAndDumpSubproblemCache). Unlike
    // subproblemCache, this is cheap to keep around across many runs --
    // it's just the six counts, not the per-subproblem detail.
    List<SubproblemCacheSummary> history = new ArrayList<>();

    /**
     * Constructor for MurtyCacheEnumerator
     * @param problem the assignment problem to enumerate
     */
    public MurtyCacheEnumerator(AssignmentProblem problem) {
        this.problem = problem;
        this.hungarian = new Hungarian(problem);
        this.scratchMatrix = new int[problem.numRows][problem.numCols];
    }

    /**
     * Enumerates the top-k solutions to the assignment problem, while
     * recording every node generated (including infeasible ones) into
     * the subproblem cache for later reporting.
     * @param k The number of solutions to enumerate.
     * @return A list of solutions to the assignment problem.
     */
    public List<AssignmentSolution> enumerate(int k) {
        // initialize data structures
        List<AssignmentSolution> topK = new ArrayList<>();
        MurtyQueue pq = new MurtyQueue(k);
        int[][] costMatrix = this.problem.costMatrix;

        // initial solution
        List<int[]> rootExclusions = new ArrayList<>();
        List<int[]> rootInclusions = new ArrayList<>();
        AssignmentSolution baseSolution = callHungarian(costMatrix);
        recordSubproblem(rootInclusions, rootExclusions, baseSolution.cost < this.problem.infinity);

        // initial node
        MurtyNode node = new MurtyNode(baseSolution, rootExclusions, rootInclusions);
        pq.qInsert(node);

        while (topK.size() < k && !pq.isEmpty()) {
            // pop best solution
            node = pq.qPopMin();
            topK.add(node.solution);

            int[] currentAssignment = node.solution.assignment;
            int n = currentAssignment.length;

            // Find the first position that's not already forced by inclusions
            int startPos = 0;
            for (int[] inc : node.inclusions) {
                startPos = Math.max(startPos, inc[0] + 1);
            }

            for (int i = startPos; i < n; i++) {
                // Build new inclusions: force all assignments from startPos to i-1
                List<int[]> newInclusions = new ArrayList<>(node.inclusions);
                for (int j = startPos; j < i; j++)
                    newInclusions.add(new int[]{j, currentAssignment[j]});

                // Add exclusion at position i
                List<int[]> newExclusions = new ArrayList<>(node.exclusions);
                newExclusions.add(new int[]{i, currentAssignment[i]});

                // Modify cost matrix
                int[][] modifiedMatrix = enforceConstraints(newExclusions, newInclusions);

                // Solve subproblem
                AssignmentSolution solution = callHungarian(modifiedMatrix);

                // Check if the solution is infeasible based on modified cost
                boolean solvable = solution.cost < this.problem.infinity;

                // record this node in the subproblem cache regardless of
                // feasibility -- this is what lets us count unsolvable
                // nodes in the tree vs. the graph
                recordSubproblem(newInclusions, newExclusions, solvable);

                if (!solvable)
                    continue;

                // Calculate actual cost and add to MurtyQueue
                int actualCost = AssignmentProblem.cost(costMatrix, solution.assignment);
                solution.cost = actualCost;
                if (pq.size() < k){
                    pq.qInsert(new MurtyNode(solution, newExclusions, newInclusions));
                } else if (actualCost < pq.peekMax().solution.cost) {
                    pq.qReplaceMax(new MurtyNode(solution, newExclusions, newInclusions));
                }
            }
        }
        printCacheStats();
        return topK;
    }

    /**
     * Computes the equivalence key for a node (freeColumns, live
     * exclusions), and either registers it as a newly-discovered
     * subproblem (assigning it the next sequential index) or bumps the
     * hitCount of the existing entry if this exact subproblem has
     * already been seen. Trivial single-row/single-column subproblems
     * are skipped entirely -- excluded from BOTH the graph and the tree
     * totals -- see the early-return below. A separate, purely
     * informational tally is still kept (singleCellNodeCount etc.) in
     * case it's useful, but it is NOT folded into any reported count.
     *
     * @param inclusions the node's accumulated inclusions -- used to
     *                    derive both freeColumns and freeRows
     * @param exclusions the node's full accumulated exclusion list;
     *                    only the ones on still-free rows are kept
     * @param solvable whether this node's Hungarian solve was feasible
     */
    private void recordSubproblem(List<int[]> inclusions, List<int[]> exclusions, boolean solvable) {
        Set<Integer> freeColumns = computeFreeColumns(inclusions);
        Set<Integer> freeRows = computeFreeRows(inclusions);

        // Trivial single-row/single-column subproblems (a 1x1
        // submatrix) are deliberately excluded from BOTH the graph and
        // the tree counts -- not cached, and not tallied into any
        // reported total. singleCellNodeCount etc. below are kept only
        // as an informational side-count, separate from Tree/Graph.
        if (freeRows.size() <= 1 || freeColumns.size() <= 1) {
            this.singleCellNodeCount++;
            if (solvable)
                this.singleCellSolvableCount++;
            else
                this.singleCellUnsolvableCount++;
            return;
        }

        Set<Long> liveExclusionKeys = computeLiveExclusionKeys(exclusions, freeRows);
        SubproblemKey key = new SubproblemKey(freeColumns, liveExclusionKeys);

        SubproblemNode existing = this.subproblemCache.get(key);
        if (existing == null) {
            SubproblemNode created = new SubproblemNode(freeColumns, liveExclusionKeys, this.nextSubproblemIndex, solvable);
            this.nextSubproblemIndex++;
            this.subproblemCache.put(key, created);
        } else {
            existing.hitCount++;
            if (existing.solvable != solvable) {
                // Should not happen if the equivalence key is sound --
                // the same (freeColumns, live exclusions) pair must
                // always induce the same matrix and therefore the same
                // feasibility. Flagged loudly rather than silently
                // trusted, in case that assumption is ever violated.
                System.out.printf(
                    "WARNING: solvability mismatch for cached subproblem #%d (recorded=%b, now=%b)%n",
                    existing.index, existing.solvable, solvable);
            }
        }
    }

    /**
     * @param inclusions list of (row,col) pairs forced by ancestors
     * @return the set of columns NOT yet pinned down by any inclusion
     */
    private Set<Integer> computeFreeColumns(List<int[]> inclusions) {
        Set<Integer> free = new HashSet<>();
        for (int c = 0; c < this.problem.numCols; c++)
            free.add(c);
        for (int[] inc : inclusions)
            free.remove(inc[1]);
        return free;
    }

    /**
     * @param inclusions list of (row,col) pairs forced by ancestors
     * @return the set of rows NOT yet pinned down by any inclusion --
     *         i.e. the rows a live exclusion can still meaningfully
     *         apply to
     */
    private Set<Integer> computeFreeRows(List<int[]> inclusions) {
        Set<Integer> free = new HashSet<>();
        for (int r = 0; r < this.problem.numRows; r++)
            free.add(r);
        for (int[] inc : inclusions)
            free.remove(inc[0]);
        return free;
    }

    /**
     * Filters a node's full accumulated exclusion list down to the ones
     * that still matter: exclusions whose row is still free. An
     * exclusion on a row that's since been forced by an inclusion is
     * dead -- the inclusion already blocks every other cell in that
     * row, so the exclusion adds no information and is dropped here.
     *
     * @param exclusions the node's full accumulated exclusion list
     * @param freeRows the rows not yet pinned down by any inclusion
     * @return an order-independent encoding of just the live exclusions
     */
    private Set<Long> computeLiveExclusionKeys(List<int[]> exclusions, Set<Integer> freeRows) {
        Set<Long> keys = new HashSet<>();
        for (int[] exc : exclusions)
            if (freeRows.contains(exc[0]))
                keys.add(encodePair(exc[0], exc[1]));
        return keys;
    }

    private long encodePair(int row, int col) {
        return (long) row * this.problem.numCols + col;
    }

    /**
     * Computes the tree-vs-graph summary from the current subproblem
     * cache, appends it to this enumerator's {@link #getHistory() run
     * history}, and then DUMPS the cache itself -- clears the
     * (potentially huge, for large n) map of per-subproblem detail so
     * its memory can be reclaimed. Only the six summary counts survive
     * that dump, inside the returned/saved SubproblemCacheSummary.
     *
     * Call this once per "run" (i.e. once you're done with a given
     * enumerate() call and its subproblem cache), before moving on to
     * the next run, so memory doesn't accumulate across runs.
     *
     * After this call, getSubproblemsInDiscoveryOrder() and
     * printSubproblemDetails() will report nothing for this run's data,
     * since it has been discarded -- call them (if you want the detail)
     * BEFORE calling this method.
     *
     * Trivial single-row/single-column subproblems were never added to
     * the cache (see recordSubproblem), and their tallies are NOT
     * folded in here either -- they are excluded from BOTH the tree and
     * the graph counts below. The tally is only carried through as a
     * separate, purely informational field on the returned summary.
     *
     * @param label optional tag (e.g. "n=6") identifying this run, shown
     *              in the summary's printout and the combined table.
     *              May be null.
     * @return the saved summary for this run.
     */
    public SubproblemCacheSummary summarizeAndDumpSubproblemCache(String label) {
        int graphTotal = this.subproblemCache.size();
        int graphSolvable = 0;
        int graphUnsolvable = 0;
        int treeSolvable = 0;
        int treeUnsolvable = 0;
        int repeats = 0;

        for (SubproblemNode node : this.subproblemCache.values()) {
            int occurrences = 1 + node.hitCount;
            repeats += node.hitCount;
            if (node.solvable) {
                graphSolvable++;
                treeSolvable += occurrences;
            } else {
                graphUnsolvable++;
                treeUnsolvable += occurrences;
            }
        }

        // Single-cell subproblems are excluded from BOTH totals below --
        // tree total is graph total + repeats only, same as before
        // single-cell tracking existed. this.singleCellNodeCount is
        // carried on the summary purely as an informational side-note.
        int treeTotal = graphTotal + repeats;

        SubproblemCacheSummary summary = new SubproblemCacheSummary(
            label, treeTotal, treeSolvable, treeUnsolvable,
            graphTotal, graphSolvable, graphUnsolvable, repeats,
            this.singleCellNodeCount);
        this.history.add(summary);

        // Dump: release the big per-subproblem cache now that its
        // counts have been extracted into the lightweight summary above.
        this.subproblemCache = new LinkedHashMap<>();
        this.nextSubproblemIndex = 0;
        this.singleCellNodeCount = 0;
        this.singleCellSolvableCount = 0;
        this.singleCellUnsolvableCount = 0;

        return summary;
    }

    /**
     * @return the summaries saved so far, one per completed run, in the
     *         order summarizeAndDumpSubproblemCache(...) was called.
     */
    public List<SubproblemCacheSummary> getHistory() {
        return this.history;
    }

    /**
     * @return the distinct subproblems currently in the cache, ordered
     *         by discovery index (the order in which the tree search
     *         first encountered each one). Empty after the cache has
     *         been dumped by summarizeAndDumpSubproblemCache.
     */
    public List<SubproblemNode> getSubproblemsInDiscoveryOrder() {
        List<SubproblemNode> nodes = new ArrayList<>(this.subproblemCache.values());
        nodes.sort((a, b) -> Integer.compare(a.index, b.index));
        return nodes;
    }

    /**
     * Optional verbose dump: one line per distinct subproblem currently
     * in the cache, in discovery order. Not called automatically (n!
     * trees can have a lot of distinct subproblems) -- call manually,
     * and BEFORE summarizeAndDumpSubproblemCache, if you want the detail
     * behind the summary counts.
     */
    public void printSubproblemDetails() {
        System.out.println();
        System.out.println("=== Distinct subproblems, in discovery order ===");
        for (SubproblemNode node : getSubproblemsInDiscoveryOrder()) {
            System.out.printf("#%d  solvable=%b  hits=%d  freeColumns=%s  exclusions=%s%n",
                node.index, node.solvable, node.hitCount,
                node.freeColumns, decodeExclusions(node.exclusions));
        }
    }

    private List<int[]> decodeExclusions(Set<Long> encoded) {
        List<int[]> pairs = new ArrayList<>();
        for (long e : encoded) {
            int col = (int) (e % this.problem.numCols);
            int row = (int) (e / this.problem.numCols);
            pairs.add(new int[]{row, col});
        }
        return pairs;
    }

    /**
     * Prints every distinct subproblem currently in the cache (in
     * discovery order): its free columns, its exclusions, and the
     * resulting free-row x free-column submatrix of the original cost
     * matrix, with excluded cells marked "X".
     *
     * Must be called BEFORE summarizeAndDumpSubproblemCache, since that
     * empties the cache this reads from.
     *
     * Assumes a square cost matrix (numRows == numCols) -- true for
     * everything this algorithm solves. Free rows aren't stored
     * explicitly on the node, but this implementation always forces a
     * prefix of rows 0..startPos-1 via inclusions, one row per consumed
     * column, so free rows are recoverable purely from the free-column
     * count: they're the LAST freeColumns.size() row indices.
     */
    public void printCacheNodes() {
        printCacheNodes(Integer.MAX_VALUE);
    }

    /**
     * Same as {@link #printCacheNodes()}, but stops after `limit`
     * subproblems (still in discovery order) -- useful since large n
     * can have far more distinct subproblems than is useful to read.
     */
    public void printCacheNodes(int limit) {
        List<SubproblemNode> nodes = getSubproblemsInDiscoveryOrder();
        System.out.println();
        System.out.println("=== Subproblem cache contents, in discovery order ===");
        int shown = 0;
        for (SubproblemNode node : nodes) {
            if (shown >= limit) {
                System.out.printf("... (%d more subproblems not shown)%n", nodes.size() - shown);
                break;
            }
            printSubproblemNode(node);
            shown++;
        }
    }

    private void printSubproblemNode(SubproblemNode node) {
        List<Integer> freeRows = computeFreeRows(node.freeColumns.size());
        List<Integer> freeCols = new ArrayList<>(node.freeColumns);
        Collections.sort(freeCols);

        // node.exclusions is already just the LIVE exclusions -- dead
        // ones on rows since forced by an inclusion are filtered out at
        // recording time (see computeLiveExclusionKeys), so there's no
        // ancestor history left to separate out here.
        List<int[]> liveExclusions = decodeExclusions(node.exclusions);
        Set<Long> blocked = new HashSet<>(node.exclusions);

        System.out.println();
        System.out.printf("--- Subproblem #%d  (solvable=%b, hits=%d) ---%n", node.index, node.solvable, node.hitCount);
        System.out.printf("free columns: %s%n", freeCols);
        System.out.printf("free rows:    %s%n", freeRows);
        System.out.printf("exclusions blocking this submatrix: %s%n", formatPairs(liveExclusions));
        System.out.println("submatrix (original row/col indices, X = excluded):");
        printSubmatrix(freeRows, freeCols, blocked);
    }

    /**
     * @param freeColumnCount number of free columns on a node
     * @return the free row indices -- the last freeColumnCount rows,
     *         since inclusions always force a prefix of rows 0..startPos-1
     *         one-for-one with consumed columns (square matrix assumed)
     */
    private List<Integer> computeFreeRows(int freeColumnCount) {
        List<Integer> freeRows = new ArrayList<>();
        int startPos = this.problem.numRows - freeColumnCount;
        for (int r = startPos; r < this.problem.numRows; r++)
            freeRows.add(r);
        return freeRows;
    }

    private String formatPairs(List<int[]> pairs) {
        if (pairs.isEmpty())
            return "[]";
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < pairs.size(); i++) {
            if (i > 0) sb.append(", ");
            sb.append("(").append(pairs.get(i)[0]).append(",").append(pairs.get(i)[1]).append(")");
        }
        return sb.append("]").toString();
    }

    private void printSubmatrix(List<Integer> freeRows, List<Integer> freeCols, Set<Long> blocked) {
        if (freeRows.isEmpty() || freeCols.isEmpty()) {
            System.out.println("  (empty -- fully assigned)");
            return;
        }

        // column width wide enough for the widest cell value, "X", and
        // the "cN" column headers
        int width = 2;
        for (int r : freeRows)
            for (int c : freeCols)
                width = Math.max(width, Integer.toString(this.problem.costMatrix[r][c]).length());
        for (int c : freeCols)
            width = Math.max(width, ("c" + c).length());
        width += 2;

        StringBuilder header = new StringBuilder(String.format("%" + width + "s", ""));
        for (int c : freeCols)
            header.append(String.format("%" + width + "s", "c" + c));
        System.out.println(header);

        for (int r : freeRows) {
            StringBuilder row = new StringBuilder(String.format("%" + width + "s", "r" + r));
            for (int c : freeCols) {
                String cell = blocked.contains(encodePair(r, c)) ? "X" : Integer.toString(this.problem.costMatrix[r][c]);
                row.append(String.format("%" + width + "s", cell));
            }
            System.out.println(row);
        }
    }

    /**
     *
     * @param exclusions: A list of (row,column) pairs that must be excluded from the solution.
     * @param inclusions: A list of (row,column) pairs that must be included in the solution.
     * @return the modified cost matrix after exclusions and inclusions are enforced (by setting costs to infinity)
     *
     */
    public int[][] enforceConstraints(List<int[]> exclusions, List<int[]> inclusions) {
        int n = this.problem.numRows;
        int m = this.problem.numCols;
        int[][] enforced = copyMatrix(this.problem.costMatrix);

        for (int[] pair : inclusions) {
            int row = pair[0];
            int col = pair[1];
            enforced[row][col] = 0;

            // Block all other cells in this row
            for (int j = 0; j < m; j++)
                if (j != col)
                    enforced[row][j] = this.problem.infinity;

            // Block all other cells in this column
            for (int i = 0; i < n; i++)
                if (i != row)
                    enforced[i][col] = this.problem.infinity;
        }

        for (int[] pair : exclusions) {
            int row = pair[0];
            int col = pair[1];
            enforced[row][col] = this.problem.infinity;
        }

        return enforced;
    }

    /**
     * Solve the assignment problem.
     * @param matrix Cost matrix of assignment problem.
     * @return Optimal solution to the assignment problem.
     */
    AssignmentSolution callHungarian(int[][] matrix) {
        long startTime = System.nanoTime();
        AssignmentSolution solution = this.hungarian.solve(matrix);
        long endTime = System.nanoTime();
        this.totalCalls += 1;
        this.totalTime += endTime-startTime;

        return solution;
    }

    int[][] copyMatrix(int[][] matrix) {
        int n = this.problem.numRows;
        int m = this.problem.numCols;
        for (int i = 0; i < n; i++)
            System.arraycopy(matrix[i],0,this.scratchMatrix[i],0,m);
        return this.scratchMatrix;
    }

    public void printCacheStats() {
        System.out.printf("cache hits: %d\n", this.cacheHits);
        System.out.printf("cache miss: %d\n", this.cacheMisses);
        System.out.printf("hungarian time: %.4f (%d calls)\n", this.totalTime*1e-9, this.totalCalls);
    }

    // -------------------------------------------------------------------
    // Driver: for each n in [3, 10], build a random n x n assignment
    // problem, enumerate ALL n! solutions with Murty's algorithm, and
    // print the tree-vs-graph subproblem counts.
    //
    // WARNING: this is exhaustive enumeration, not top-k. The tree has
    // on the order of n * n! nodes, each requiring a Hungarian solve, so
    // n=9 and especially n=10 can take a long time (Hungarian is O(n^3)
    // per call). Runtime is printed per n so you can see the scaling.
    // -------------------------------------------------------------------

    public static void main(String[] args) {
        int startN = 3;
        int endN = 10;
        if (args.length >= 2) {
            startN = Integer.parseInt(args[0]);
            endN = Integer.parseInt(args[1]);
        }

        // Each run's summary (six small ints) is kept here so we can
        // print a combined table at the end. This list stays cheap even
        // across n=3..10 -- it never holds the big per-subproblem cache,
        // only the counts extracted from it after each run.
        List<SubproblemCacheSummary> allSummaries = new ArrayList<>();

        for (int n = startN; n <= endN; n++) {
            SubproblemCacheSummary summary = runFullEnumeration(n);
            if (summary != null)
                allSummaries.add(summary);
        }

        SubproblemCacheSummary.printTable(allSummaries);
    }

    /**
     * Builds a random n x n cost matrix, enumerates all n! solutions,
     * then summarizes and dumps this run's subproblem cache before
     * returning -- so by the time this method returns, the (potentially
     * huge, for large n) per-subproblem detail has already been
     * released, and only the small summary survives.
     *
     * @return this run's summary, or null if n was skipped.
     */
    private static SubproblemCacheSummary runFullEnumeration(int n) {
        long k = factorial(n);
        if (k > Integer.MAX_VALUE) {
            System.out.printf("Skipping n=%d: %d! = %d exceeds int range for queue capacity.%n", n, n, k);
            return null;
        }

        String label = "n=" + n;
        int[][] costMatrix = randomCostMatrix(n, n, 1, 100, new java.util.Random(1000 + n));
        AssignmentProblem problem = new AssignmentProblem(costMatrix);
        MurtyCacheEnumerator enumerator = new MurtyCacheEnumerator(problem);

        System.out.println();
        System.out.println("=====================================================");
        System.out.printf("  n = %d   (enumerating all %d = %d! solutions)%n", n, k, n);
        System.out.println("=====================================================");

        long startTime = System.nanoTime();
        List<AssignmentSolution> allSolutions = enumerator.enumerate((int) k);
        long elapsedNanos = System.nanoTime() - startTime;

        System.out.printf("  solutions returned: %d (expected %d)%n", allSolutions.size(), k);
        System.out.printf("  wall time: %.3f s%n", elapsedNanos * 1e-9);
        System.out.println("=====================================================");

        // Extract this run's tree-vs-graph counts, print them, and free
        // the big per-subproblem cache before returning -- this is the
        // "count between runs, then dump the cache" step.
        SubproblemCacheSummary summary = enumerator.summarizeAndDumpSubproblemCache(label);
        summary.print();

        return summary;
    }

    private static long factorial(int n) {
        long result = 1;
        for (int i = 2; i <= n; i++)
            result *= i;
        return result;
    }

    private static int[][] randomCostMatrix(int rows, int cols, int minCost, int maxCost, java.util.Random rng) {
        int[][] matrix = new int[rows][cols];
        int range = maxCost - minCost + 1;
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < cols; j++)
                matrix[i][j] = minCost + rng.nextInt(range);
        return matrix;
    }

}

/**
 * Immutable equivalence key for a Murty tree node's induced subproblem:
 * the set of free (unforced) columns, together with the LIVE (row,col)
 * cells this node has excluded -- exclusions on rows already locked in
 * by an inclusion are dropped, since they add no information beyond
 * what the inclusion itself already enforces. Two nodes with equal keys
 * are guaranteed to hand Hungarian the identical constrained matrix.
 */
class SubproblemKey {
    final Set<Integer> freeColumns;
    final Set<Long> exclusions;

    SubproblemKey(Set<Integer> freeColumns, Set<Long> exclusions) {
        this.freeColumns = freeColumns;
        this.exclusions = exclusions;
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof SubproblemKey)) return false;
        SubproblemKey other = (SubproblemKey) o;
        return this.freeColumns.equals(other.freeColumns) && this.exclusions.equals(other.exclusions);
    }

    @Override
    public int hashCode() {
        return 31 * freeColumns.hashCode() + exclusions.hashCode();
    }
}

/**
 * A distinct subproblem discovered during enumeration: which columns
 * are still free, which cells are (live-)excluded, the sequential
 * order in which this subproblem was first discovered, how many
 * additional times (beyond the first) it recurred elsewhere in the
 * tree, and whether it's feasible.
 */
class SubproblemNode {
    final Set<Integer> freeColumns;
    final Set<Long> exclusions;
    final int index;
    int hitCount;
    boolean solvable;

    SubproblemNode(Set<Integer> freeColumns, Set<Long> exclusions, int index, boolean solvable) {
        this.freeColumns = freeColumns;
        this.exclusions = exclusions;
        this.index = index;
        this.hitCount = 0;
        this.solvable = solvable;
    }
}

/**
 * The lightweight, saved result of one enumerate() run: the tree-vs-
 * graph feasible/non-feasible counts plus the repeat count, with the
 * big per-subproblem detail already discarded. Produced by
 * MurtyCacheEnumerator.summarizeAndDumpSubproblemCache(...).
 *
 * excludedSingleCell is a purely informational side-count of trivial
 * single-row/single-column subproblems encountered during the run --
 * it is NOT folded into treeTotal/treeSolvable/treeUnsolvable or any
 * graph count. Those subproblems are excluded from both the tree and
 * the graph entirely (see recordSubproblem).
 */
class SubproblemCacheSummary {
    final String label;
    final int treeTotal;
    final int treeSolvable;
    final int treeUnsolvable;
    final int graphTotal;
    final int graphSolvable;
    final int graphUnsolvable;
    final int repeats;
    final int excludedSingleCell;

    SubproblemCacheSummary(String label,
                            int treeTotal, int treeSolvable, int treeUnsolvable,
                            int graphTotal, int graphSolvable, int graphUnsolvable,
                            int repeats, int excludedSingleCell) {
        this.label = label;
        this.treeTotal = treeTotal;
        this.treeSolvable = treeSolvable;
        this.treeUnsolvable = treeUnsolvable;
        this.graphTotal = graphTotal;
        this.graphSolvable = graphSolvable;
        this.graphUnsolvable = graphUnsolvable;
        this.repeats = repeats;
        this.excludedSingleCell = excludedSingleCell;
    }

    /** Prints this single run's tree-vs-graph table. */
    void print() {
        System.out.println();
        System.out.println("---------------------------------------------------");
        System.out.println("  Subproblem cache report: TREE vs. GRAPH"
            + (label != null ? "  (" + label + ")" : ""));
        System.out.println("---------------------------------------------------");
        System.out.printf("  %-14s%12s%12s%n", "", "Tree", "Graph");
        System.out.printf("  %-14s%12d%12d%n", "Total", treeTotal, graphTotal);
        System.out.printf("  %-14s%12d%12d%n", "Feasible", treeSolvable, graphSolvable);
        System.out.printf("  %-14s%12d%12d%n", "Non-Feasible", treeUnsolvable, graphUnsolvable);
        System.out.println("---------------------------------------------------");
        System.out.printf("  Repeated occurrences (tree - graph): %d%n", repeats);
        System.out.printf("  Single-cell subproblems (excluded from tree AND graph, informational only): %d%n", excludedSingleCell);
        System.out.println("---------------------------------------------------");

        if (treeTotal != treeSolvable + treeUnsolvable) {
            System.out.println("WARNING: tree total does not equal solvable+unsolvable -- bug in bookkeeping.");
        }
    }

    /**
     * Prints the final combined table, one row per run: n (via each
     * summary's label), TreeFeasible, TreeNonFeasible, GraphFeasible,
     * GraphNonFeasible.
     */
    static void printTable(List<SubproblemCacheSummary> summaries) {
        System.out.println();
        System.out.println("=========================================================================");
        System.out.println("  Combined summary across all runs");
        System.out.println("=========================================================================");
        System.out.printf("  %-8s%16s%18s%16s%18s%n",
            "n", "TreeFeasible", "TreeNonFeasible", "GraphFeasible", "GraphNonFeasible");
        for (SubproblemCacheSummary s : summaries) {
            System.out.printf("  %-8s%16d%18d%16d%18d%n",
                s.label != null ? s.label : "",
                s.treeSolvable, s.treeUnsolvable,
                s.graphSolvable, s.graphUnsolvable);
        }
        System.out.println("=========================================================================");
    }
}
