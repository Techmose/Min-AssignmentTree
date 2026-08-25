import java.util.ArrayList;
import java.util.BitSet;
import java.util.List;
import java.util.Random;

/**
 * Algorithm for enumerating the top-k best solutions to the
 * assignment problem.  This algorithm is based on search the Order
 * Graph.
 *
 */
public class OrderGraphEXPEnumerator {
    AssignmentProblem problem;
    OrderGraphEXP graph;

    // stats for algorithm performance
    int totalCalls;
    long totalTime;

    /**
     * Constructor for OrderGraphEXPEnumerator
     * @param costMatrix of the assignment problem to enumerate
     */
    public OrderGraphEXPEnumerator(AssignmentProblem problem) {
        this.problem = problem;

        // stats for algorithm performance
        this.totalCalls = 0;
        this.totalTime = 0;
    }

    /**
     * Enumerates the top-k solutions to the assignment problem.
     * @param k The number of solutions to enumerate.
     * @return A list of solutions to the assignment problem.
     */
    public List<AssignmentSolution> enumerate(int k) {
        // initialize data structures
        List<AssignmentSolution> topK = new ArrayList<>();

        // initial call to Hungarian algorithm
        AssignmentSolution solution = callHungarian(this.problem.costMatrix);

        // initialize order graph
        this.graph = new OrderGraphEXP(this.problem);
        OrderGraphEXPNode root = this.graph.makeRoot(solution.cost);

        // initialize priority queue
        Queue pq = new Queue(k);
        //List<Integer> path = new ArrayList<Integer>();
        OrderGraphPath path = OrderGraphPath.emptyPath();
        PQNode pqNode = new PQNode(solution.cost,path,0,root);
        pq.qInsert(pqNode);
        

        while (topK.size() < k && !pq.isEmpty()) {
            // pop best solution
            pqNode = pq.qPopMin();
            if (pqNode.path.size() == this.problem.numRows) {
                // found a leaf node
                topK.add(pqNode.solution());
                pq.maxSize--;
                continue;
            }

            // get set of used columns
            int childIndex = 0;
            BitSet cols = pqNode.ogNode.cols;
            // generate children: try assigning the next row
            for (int col = 0; col < this.problem.numCols; col++) {
                // skip if column is already used
                if (cols.get(col)) continue;

                // update path to node
                OrderGraphPath newPath = pqNode.path.append(col);
                BitSet newCols = (BitSet)cols.clone();
                newCols.set(col);
                // combine cost to node and cost of node
                int pathCost = pqNode.pathCost + lastCost(newPath);

                //Check if node cost in within bounds (Cheaper than the current Max) if not, skip it
                if ( pq.isFull() && pathCost > pq.peekMax().cost) {
                // move the child index to the next column, even if we skip it
                    childIndex++;
                    continue;
                }
                
                // check if sub-problem has been solved before
                OrderGraphEXPNode ogNode = pqNode.ogNode;
                OrderGraphEXPNode childOgNode;
                if (ogNode.containsChild(childIndex,newCols))
                    childOgNode = ogNode.get(childIndex);
                else {
                    int[][] newMatrix = subMatrix(newCols);
                    int solCost = callHungarian(newMatrix).cost;
                    childOgNode = ogNode.put(childIndex,newCols,solCost);
                }
                childIndex++;

                
                int newCost = pathCost + childOgNode.value;
                /*
                int pathCost = this.problem.cost(newPath);
                int newCost = pathCost + childOgNode.value;
                */
                // push child onto pq
                PQNode newNode = new PQNode(newCost,newPath,pathCost,childOgNode);

                //If pq isn't full yet, insert the new node.  If it is full, replace the max with the new node, we already checked that it's cheaper
                if (!pq.isFull()) {
                    pq.qInsert(newNode);
                }
                else if ((pq.maxSize() != 0) && newNode.compareTo(pq.peekMax()) < 0){
                    pq.qReplaceMax(newNode);
                }
                
            }
        }
        printCacheStats();
        return topK;
    }

    /**
     * Solve the assignment problem.
     * @param matrix Cost matrix of assignment problem.
     * @return Optimal solution to the assignment problem.
     */
    AssignmentSolution callHungarian(int[][] matrix) {
        long startTime = System.nanoTime();
        AssignmentSolution solution = Hungarian.solve_alone(matrix);
        long endTime = System.nanoTime();
        this.totalCalls += 1;
        this.totalTime += endTime-startTime;

        return solution;
    }

    /**
     * This functions finds the sub-matrix of the assignment problem's
     * cost matrix found by excluding the given columns, and excluding
     * the same number of the initial rows of the matrix.  That is, it
     * returns the matrix that results from assigning the k given
     * columns to the first k rows.  (The particular assignment does
     * not matter).
     * @param cols Set of columns to _exclude_.
     * @return Corresponding sub-matrix
     */
    int[][] subMatrix(BitSet cols) {
        // cols contains set of assigned columns
        int colsSize = cols.cardinality();
        // # of assigned rows == # of assigned columns
        int rowsLeft = this.problem.numRows-colsSize;
        int colsLeft = this.problem.numCols-colsSize;
        int[][] matrix = new int[rowsLeft][colsLeft];
        int i = 0;
        for (int row = colsSize; row < this.problem.numRows; row++) {
            int j = 0;
            for (int col = 0; col < this.problem.numCols; col++) {
                if (cols.get(col)) continue;
                matrix[i][j] = this.problem.costMatrix[row][col];
                j++;
            }
            i++;
        }
        return matrix;
    }

    int lastCost(List<Integer> path) {
        int row = path.size()-1;
        int col = path.get(row);
        return this.problem.costMatrix[row][col];
    }

    int lastCost(OrderGraphPath path) {
        int row = path.size()-1;
        int col = path.value();
        return this.problem.costMatrix[row][col];
    }

    public void printCacheStats() {
        System.out.printf("cache hits: %d\n", this.graph.cache.hits);
        System.out.printf("cache miss: %d\n", this.graph.cache.misses);
        System.out.printf("hungarian time: %.4f (%d calls)\n", this.totalTime*1e-9, this.totalCalls);
    }

    public static void main(String[] args) {
        int n = 10;
        //int k = 100000;
        int k = 3628800/2;
        //int n = 40;
        //int k = 110000;
        int bound = 10;
        int seed = 0;
        long start, end;

        System.out.printf("%d-x-%d matrix, k=%d, bound=%d\n", n, n, k, bound);

        boolean runMurtys = true;

        Random r = new Random(seed);
        int[][] costMatrix = new int[n][n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                costMatrix[i][j] = r.nextInt(bound);

        AssignmentProblem problem = new AssignmentProblem(costMatrix);
        OrderGraphEXPEnumerator ogEnumerator = new OrderGraphEXPEnumerator(problem);
        MurtyEnumerator mEnumerator = new MurtyEnumerator(problem);

        List<AssignmentSolution> topK = null;
        List<AssignmentSolution> topK2 = null;

        //List<AssignmentSolution> topK = new ArrayList<>();
        //long start, end;

        System.out.println("enumerating...");
        start = System.nanoTime();
        topK = ogEnumerator.enumerate(k);
        end = System.nanoTime();
        System.out.printf("timer: %.4f\n", ((end-start)*1e-9));

        if (runMurtys) {
            System.out.println("enumerating (Murty's)...");
            start = System.nanoTime();
            topK2 = mEnumerator.enumerate(k);
            end = System.nanoTime();
            System.out.printf("timer: %.4f\n", ((end-start)*1e-9));
            System.out.printf("count: %d\n", topK.size());
        }

        if (runMurtys) {
            // sanity check
            boolean ok = true;
            if (topK.size() != topK2.size())
                System.out.println("check: NOT OK");
            else
                for (int i = 0; i < topK.size(); i++) {
                    AssignmentSolution r1 = topK.get(i);
                    AssignmentSolution r2 = topK2.get(i);
                    if (r1.cost != r2.cost) {
                        ok = false;
                        System.out.printf("index = %d\n", i);
                        System.out.println(r1);
                        System.out.println(r2);
                        break;
                    }
                }
            if (ok) System.out.println("check: ok");
            else System.out.println("check: NOT OK");
        }

        /*
        // print top-k and bottom-k solutions
        //System.out.println(java.util.Arrays.deepToString(costMatrix));
        System.out.println("==1:");
        for (int i = 0; i < 10; i++) {
            System.out.println(topK.get(i));
        }
        System.out.println("==2:");
        for (int i = 0; i < 10; i++) {
            System.out.println(topK2.get(i));
        }
        System.out.println("==3:");
        for (int i = 0; i < 10; i++) {
            System.out.println(topK.get(topK.size()-i-1));
        }
        */

        System.out.println("== Order Graph Stats:");
        ogEnumerator.printCacheStats();
        if (runMurtys) {
            System.out.println("== Murty Stats:");
            mEnumerator.printCacheStats();
        }
    }

    /**
     * Quickselect: rearranges arr[lo..hi] so that arr[0..k] contains the
     * (k+1) cheapest nodes (by cost ascending) and arr[k] is exactly the
     * (k+1)-th cheapest. Elements within each partition are in no particular order.
     *
     * Average O(n), worst-case O(n²) — median-of-three pivot selection keeps
     * the worst case rare in practice without needing a random number generator.
     */
    private static void quickselect(OrderGraphEXPNode[] arr, int lo, int hi, int k) {
        while (lo < hi) {
            // Median-of-three pivot: compare lo, mid, hi and put the median at hi
            int mid = lo + (hi - lo) / 2;
            if (arr[lo].value > arr[mid].value) swap(arr, lo, mid);
            if (arr[lo].value > arr[hi].value)  swap(arr, lo, hi);
            if (arr[mid].value > arr[hi].value) swap(arr, mid, hi);
            // arr[mid] is now the median; move it to hi-1 as the pivot
            swap(arr, mid, hi);
            int pivot = partition(arr, lo, hi);
            if      (pivot == k) return;
            else if (pivot  < k) lo = pivot + 1;
            else                 hi = pivot - 1;
        }
    }

    /** Lomuto partition around arr[hi]; returns final pivot index. */
    private static int partition(OrderGraphEXPNode[] arr, int lo, int hi) {
        int pivotCost = arr[hi].value;
        int i = lo;
        for (int j = lo; j < hi; j++) {
            if (arr[j].value <= pivotCost)
                swap(arr, i++, j);
        }
        swap(arr, i, hi);
        return i;
    }

    private static void swap(OrderGraphEXPNode[] arr, int i, int j) {
        OrderGraphEXPNode tmp = arr[i];
        arr[i] = arr[j];
        arr[j] = tmp;
    }
}

/**
 * Used by Order Graph Enumerator.
 */
class PQNode implements Comparable<PQNode> {
    OrderGraphEXPNode ogNode;
    int cost;
    OrderGraphPath path;
    int pathCost;
    int length;
    int id;
    static int id_counter = 0;

    public PQNode(int cost, OrderGraphPath path, int pathCost, OrderGraphEXPNode ogNode) {
        this.cost = cost;
        this.ogNode = ogNode;
        this.path = path;
        this.pathCost = pathCost;
        this.length = path.size();
        this.id = id_counter++;
    }

    /**
     * This is for ordering nodes in the priority queue.  Order by
     * cost.
     */
    @Override
    public int compareTo(PQNode other) {
        if (this.cost < other.cost)
            return -1;
        else if (this.cost > other.cost)
            return 1;
        else  {
            if (this.length > other.length)
                return -1;
            else if (this.length < other.length)
                return 1;
            else {
                if (this.id < other.id) return -1;
                else if (this.id > other.id) return 1;
                else return 0;
            }
        }
    }

    public AssignmentSolution solution() {
        int[] sol = new int[this.length];
        int i = this.length-1;
        OrderGraphPath path = this.path;
        while (!path.isEnd()) {
            sol[i--] = path.value();
            path = path.next();
        }
        return new AssignmentSolution(sol,this.cost);
    }
}
