/**
 * Used by Order Graph Enumerator.
 *
 * Uses a parent-pointer structure instead of copying the full path list on
 * every child expansion. Each node stores only the column assigned at this
 * depth and a reference to its parent; the full path is reconstructed by
 * walking the chain only when a leaf solution is needed.
 *
 * Each node also owns the BitSet of used columns for its depth, passed in
 * from the parent — no clone needed in the hot loop.
 */
import java.util.BitSet;

class OrderGraphNode implements Comparable<OrderGraphNode> {
    int cost;
    int length;
    int id;
    static int id_counter = 0;

    /** The column assigned at this node (-1 for the root). */
    final int col;

    /** Parent node; null for the root. */
    final OrderGraphNode parent;

    /**
     * Columns already used at this depth (owned by this node).
     * Built once from the parent's usedCols + this.col; never cloned again.
     */
    final BitSet usedCols;

    /**
     * Root constructor (no parent, no column assigned yet).
     * @param cost     Cost estimate for the full problem.
     * @param numCols  Total number of columns (for initial BitSet sizing).
     */
    public OrderGraphNode(int cost, int numCols) {
        this.cost      = cost;
        this.length    = 0;
        this.col       = -1;
        this.parent    = null;
        this.usedCols  = new BitSet(numCols);
        this.id        = OrderGraphNode.id_counter++;
    }

    /**
     * Child constructor.
     * @param cost       Combined cost (path cost + sub-problem cost).
     * @param col        Column assigned at this depth.
     * @param parent     Parent node.
     * @param childCols  Pre-built BitSet for this child (parent.usedCols | col).
     */
    public OrderGraphNode(int cost, int col, OrderGraphNode parent, BitSet childCols) {
        this.cost     = cost;
        this.col      = col;
        this.parent   = parent;
        this.length   = parent.length + 1;
        this.usedCols = childCols;
        this.id       = OrderGraphNode.id_counter++;
    }

    /**
     * Order by cost ascending, then by depth descending (deeper = more progress),
     * then by insertion order for determinism.
     */
    @Override
    public int compareTo(OrderGraphNode other) {
        if (this.cost < other.cost)       return -1;
        else if (this.cost > other.cost)  return  1;
        else {
            if (this.length > other.length)       return -1;
            else if (this.length < other.length)  return  1;
            else {
                if (this.id < other.id)   return -1;
                if (this.id > other.id)   return  1;
                return 0;
            }
        }
    }

    /**
     * Reconstruct the column-assignment array by walking the parent chain.
     * Called only when a leaf node is dequeued, so this is off the hot path.
     */
    public AssignmentSolution solution() {
        int[] arr = new int[this.length];
        OrderGraphNode cur = this;
        for (int i = this.length - 1; i >= 0; i--) {
            arr[i] = cur.col;
            cur = cur.parent;
        }
        return new AssignmentSolution(arr, this.cost);
    }
}
