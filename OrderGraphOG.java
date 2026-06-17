/**
 * The Order Graph: a DAG of sub-problem nodes used by OrderGraphEnumeratorOG.
 *
 * Each OGNode represents one unique sub-problem identified by the set of
 * columns already assigned (its usedCols BitSet).
 *
 * LOOKUP — O(1), no traversal, no hashing
 * ----------------------------------------
 * The enumerator holds a direct reference to the parent OGNode on every
 * PQNode (OrderGraphNodeOG.ogNode).  Looking up a child sub-problem is a
 * single array index:
 *
 *   OGNode childOGNode = pqNode.ogNode.ogChildren[col];
 *
 * No traversal from root.  No hashing.  One pointer hop.
 *
 * BACKWARD WIRING
 * ---------------
 * insertOGNode() wires BACKWARDS so that every parent already in the graph
 * gets its ogChildren[col] slot filled immediately.  This guarantees that
 * any PQNode whose ogNode is already set will find its children via the
 * direct array lookup above.
 *
 * HASH INDEX
 * ----------
 * ogNodeIndex is used only during insertOGNode() to avoid creating
 * duplicate OGNodes for the same usedCols, and to look up parent OGNodes
 * during backward wiring.  It is never on the lookup hot path.
 *
 * Naming convention:
 *   OGNode  – a node in this graph (sub-problem node)
 *   PQNode  – a node in the enumerator's priority queue (search-path node)
 */
import java.util.BitSet;
import java.util.HashMap;

public class OrderGraphOG {

    // ------------------------------------------------------------------ //
    //  OGNode                                                              //
    // ------------------------------------------------------------------ //

    public static class OGNode {

        /** Columns already assigned at this sub-problem depth. */
        final BitSet usedCols;

        /**
         * Cost of the optimal solution to the remaining sub-matrix.
         * {@code Integer.MIN_VALUE} = not yet computed (placeholder node).
         */
        int subCost;

        /**
         * Outgoing edges indexed directly by column number.
         * ogChildren[col] is the OGNode reached by also assigning column col.
         * Slots for columns already in usedCols are null (invalid moves).
         * Array length = numCols.
         *
         * Filled by insertOGNode() of the CHILD via backward wiring —
         * when a child is inserted every parent in the graph gets its
         * ogChildren[col] slot set immediately.
         */
        final OGNode[] ogChildren;

        OGNode(BitSet usedCols, int numCols) {
            this.usedCols   = usedCols;
            this.subCost    = Integer.MIN_VALUE;
            this.ogChildren = new OGNode[numCols];
        }

        boolean isSolved() {
            return this.subCost != Integer.MIN_VALUE;
        }
    }

    // ------------------------------------------------------------------ //
    //  Graph state                                                         //
    // ------------------------------------------------------------------ //

    final int numCols;

    /** Root OGNode: depth 0, empty usedCols. */
    final OGNode ogRootNode;

    /** Leaf OGNode: all columns used, subCost = 0. */
    final OGNode ogLeafNode;

    /**
     * Hash-map index of all OGNodes.
     * Used ONLY inside insertOGNode() — never on the lookup hot path.
     */
    private final HashMap<OrderGraphHash, OGNode> ogNodeIndex;

    // stats
    int directEdgeHits;   // lookups satisfied by ogChildren[col] directly
    int hashFallbackHits;  // lookups that needed the hash index
    int hashFallbackMisses;
    int evictions;
    long hashingTime;
    long cacheEvictionTime;
    long getTime;
    long putTime;

    int hits;
    int misses;

    // ------------------------------------------------------------------ //
    //  Construction                                                        //
    // ------------------------------------------------------------------ //

    public OrderGraphOG(int numCols) {
        this.numCols     = numCols;
        this.ogNodeIndex = new HashMap<>();

        BitSet emptyCols = new BitSet(numCols);
        this.ogRootNode = new OGNode(emptyCols, numCols);
        this.ogNodeIndex.put(new OrderGraphHash(emptyCols, 0), this.ogRootNode);

        BitSet allCols = new BitSet(numCols);
        allCols.set(0, numCols);
        this.ogLeafNode         = new OGNode(allCols, numCols);
        this.ogLeafNode.subCost = 0;
        this.ogNodeIndex.put(new OrderGraphHash(allCols, 0), this.ogLeafNode);
    }

    // ------------------------------------------------------------------ //
    //  Lookup — one array index, no hashing                               //
    // ------------------------------------------------------------------ //

    /**
     * Returns the child OGNode reached by assigning {@code col} from
     * {@code parentOGNode}, or null if that edge has not been wired yet.
     *
     * This is the primary lookup path: one array index, no hashing.
     * Called from the enumerator as:
     *
     *   OGNode childOGNode = orderGraph.getChild(pqNode.ogNode, col);
     */
    public OGNode getChild(OGNode parentOGNode, int col) {
        if (parentOGNode == null) return null;
        return parentOGNode.ogChildren[col];
    }

    // ------------------------------------------------------------------ //
    //  Insertion — backward wiring                                        //
    // ------------------------------------------------------------------ //

    /**
     * Inserts a solved OGNode for {@code usedCols} and wires it BACKWARDS.
     *
     * For each set bit col in usedCols:
     *   1. Build parentUsedCols = usedCols & ~{col}
     *   2. Get or create the parent OGNode
     *   3. Set parentOGNode.ogChildren[col] = thisNode
     *
     * After this call, any PQNode whose ogNode corresponds to a parent of
     * this sub-problem can reach this OGNode in O(1) via ogChildren[col].
     *
     * @param usedCols  Owned by the OGNode after this call — do not mutate.
     * @param subCost   Hungarian cost for this sub-problem.
     * @return          The inserted or updated OGNode.
     */
    public OGNode insertOGNode(BitSet usedCols, int subCost) {
        return insertOGNode(usedCols, subCost, null, -1);
    }

    /**
     * Inserts a solved OGNode and wires it BACKWARDS.
     *
     * {@code knownParentOGNode} and {@code knownCol} identify the primary
     * parent — the one the enumerator already has in hand as {@code pqNode.ogNode}.
     * That edge is wired directly without any hash lookup, saving one
     * {@code BitSet.equals()} call on the hot path.
     *
     * All other parents (convergent paths) are still wired via the hash index.
     *
     * @param usedCols          Owned by the OGNode after this call.
     * @param subCost           Hungarian cost for this sub-problem.
     * @param knownParentOGNode The direct parent OGNode already in hand, or null.
     * @param knownCol          The column assigned to reach this node from knownParentOGNode.
     */
    public OGNode insertOGNode(BitSet usedCols, int subCost,
                               OGNode knownParentOGNode, int knownCol) {
        long t0 = System.nanoTime();
        OrderGraphHash key = new OrderGraphHash(usedCols, 0);
        this.hashingTime += System.nanoTime() - t0;

        long t1 = System.nanoTime();

        // Get or create the OGNode for this usedCols.
        OGNode ogNode = this.ogNodeIndex.get(key);
        if (ogNode == null) {
            ogNode = new OGNode(usedCols, this.numCols);
            this.ogNodeIndex.put(key, ogNode);
        }
        ogNode.subCost = subCost;

        // Wire the primary parent directly — no hash needed, we have it in hand.
        if (knownParentOGNode != null)
            knownParentOGNode.ogChildren[knownCol] = ogNode;

        // Backward wiring for all other (convergent) parents via hash index.
        for (int col = usedCols.nextSetBit(0);
             col >= 0;
             col = usedCols.nextSetBit(col + 1)) {

            // Skip the primary parent — already wired above.
            if (col == knownCol && knownParentOGNode != null) continue;

            BitSet parentUsedCols = (BitSet) usedCols.clone();
            parentUsedCols.clear(col);

            OrderGraphHash parentKey = new OrderGraphHash(parentUsedCols, 0);
            OGNode parentOGNode = this.ogNodeIndex.get(parentKey);
            if (parentOGNode == null) {
                parentOGNode = new OGNode(parentUsedCols, this.numCols);
                this.ogNodeIndex.put(parentKey, parentOGNode);
            }
            parentOGNode.ogChildren[col] = ogNode;
        }

        this.putTime += System.nanoTime() - t1;
        return ogNode;
    }

    // ------------------------------------------------------------------ //
    //  Accessors                                                           //
    // ------------------------------------------------------------------ //

    public int size() {
        return this.ogNodeIndex.size();
    }
}
