/**
 * Sub-problem cache for OrderGraphEnumeratorOG, backed by an OrderGraphOG.
 *
 * The hot-path lookup is now a single array index on the PQNode's ogNode:
 *
 *   OGNode childOGNode = pqNode.ogNode.ogChildren[col];
 *
 * This class is responsible only for insertion: insertSolvedOGNode() stores
 * the Hungarian result and wires the OGNode backwards into the graph so all
 * parent ogChildren[col] slots are immediately filled.
 *
 * Naming:
 *   OGNode  – a node in the OrderGraphOG (sub-problem node)
 *   PQNode  – a node in the enumerator's priority queue (search-path node)
 */
import java.util.BitSet;

class OrderGraphCacheOG {

    /** The graph that owns all OGNodes and their sub-problem costs. */
    final OrderGraphOG orderGraph;

    // stats — read directly by OrderGraphEnumeratorOG.printCacheStats()
    int hits;
    int misses;
    int evictions;
    long hashingTime;
    long cacheEvictionTime;
    long getTime;
    long putTime;
    long containsTime;

    boolean customHashingEnabled;
    boolean cacheEvictionEnabled;

    public OrderGraphCacheOG(int numRows, int numCols, EnumeratorConfig config) {
        this.customHashingEnabled = config.customHashingEnabled;
        this.cacheEvictionEnabled = config.cacheEvictionEnabled;
        this.orderGraph = new OrderGraphOG(numCols);
    }

    /**
     * Stores the solved sub-problem cost and wires the OGNode backwards.
     *
     * {@code knownParentOGNode} and {@code knownCol} are the direct parent
     * already in hand from the enumerator — that edge is wired without any
     * hash lookup.  All convergent parents are still wired via the hash index.
     *
     * @param usedCols          Owned by the OGNode after this call.
     * @param knownParentOGNode The pqNode.ogNode that triggered this insert.
     * @param knownCol          The column that was just assigned.
     */
    public OrderGraphOG.OGNode insertSolvedOGNode(BitSet usedCols, int subCost,
                                                OrderGraphOG.OGNode knownParentOGNode,
                                                int knownCol) {
        OrderGraphOG.OGNode ogNode = this.orderGraph.insertOGNode(
                usedCols, subCost, knownParentOGNode, knownCol);
        syncStats();
        return ogNode;
    }

    /** Total number of OGNodes in the graph. */
    public int size() {
        return this.orderGraph.size();
    }

    private void syncStats() {
        this.hits              = this.orderGraph.hits;
        this.misses            = this.orderGraph.misses;
        this.evictions         = this.orderGraph.evictions;
        this.hashingTime       = this.orderGraph.hashingTime;
        this.cacheEvictionTime = this.orderGraph.cacheEvictionTime;
        this.getTime           = this.orderGraph.getTime;
        this.putTime           = this.orderGraph.putTime;
    }
}
