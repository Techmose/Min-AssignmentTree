import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;

public class OrderGraphEXP {
    OrderGraphEXPCache cache;
    int size; // number of elements in the order

    public OrderGraphEXP(AssignmentProblem problem) {
        this.size = problem.numRows;
        this.cache = new OrderGraphEXPCache(this);
    }

    public OrderGraphEXPNode makeRoot(int value) {
        BitSet bs = new BitSet(this.size);
        OrderGraphEXPNode root = new OrderGraphEXPNode(this,bs,value);
        this.cache.put(bs,root);
        return root;
    }
}

class OrderGraphEXPNode {
    OrderGraphEXP og;
    BitSet cols;
    int graphSize;
    int nodeSize;
    int value;
    OrderGraphEXPNode[] children;

    public OrderGraphEXPNode(OrderGraphEXP og, BitSet cols, int value) {
        this.og = og;
        this.cols = cols; // key
        this.graphSize = og.size;
        this.nodeSize = cols.cardinality();
        this.value = value;
        this.children = new OrderGraphEXPNode[graphSize-nodeSize];
    }

    public int value() { return this.value; }

    public boolean containsChild(int index, BitSet cols) {
        if (this.children[index] != null)
            return true;
        OrderGraphEXPNode node = this.og.cache.get(cols);
        if (node == null)
            return false;
        else {
            this.children[index] = node;
            return true;
        }
    }

    public OrderGraphEXPNode get(int index) {
        return this.children[index];
    }

    public OrderGraphEXPNode put(int index, BitSet cols, int value) {
        OrderGraphEXPNode child = new OrderGraphEXPNode(this.og,cols,value);
        this.children[index] = child;
        this.og.cache.put(cols,child);
        return child;
    }

}

class OrderGraphEXPCache {
    ArrayList<HashMap<BitSet,OrderGraphEXPNode>> cache;
    int size;
    int hits;
    int misses;

    public OrderGraphEXPCache(OrderGraphEXP graph) {
        // initialize cache
        this.size = graph.size;
        this.cache = new ArrayList<>(this.size+1);
        for (int i = 0; i <= this.size; i++)
            cache.add(new HashMap<BitSet,OrderGraphEXPNode>());
        // prime cache with solved problem, which has 0 remaining cost
        BitSet trivialSet = allCols(size);
        int value = 0;
        OrderGraphEXPNode sink = new OrderGraphEXPNode(graph,trivialSet,value);
        cache.get(size).put(trivialSet,sink);

        this.hits = 0;
        this.misses = 0;
    }

    /**
     * @param numCols number of columns
     * @return A set containing all integers from 0..numCols-1
     */
    static BitSet allCols(int numCols) {
        BitSet all = new BitSet(numCols);
        all.set(0,numCols);
        return all;
    }

    public OrderGraphEXPNode contains(BitSet key) {
        int len = key.cardinality();
        OrderGraphEXPNode node = this.cache.get(len).get(key);
        if (node == null) this.misses++;
        else              this.hits++;
        return node;
    }

    public OrderGraphEXPNode get(BitSet key) {
        return contains(key);
    }

    public void put(BitSet key, OrderGraphEXPNode value) {
        int len = key.cardinality();
        this.cache.get(len).put(key,value);
    }
}


