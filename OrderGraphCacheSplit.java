/**
 * Cache for sub-problem costs used by OrderGraphEnumerator.
 *
 * Supports hashing and LFU-based cache eviction.
 *
 * Sub-problems are partitioned by depth: the cardinality of a usedCols
 * BitSet equals the number of rows already assigned, which is exactly the
 * depth of that node in the search tree.  Two nodes at different depths
 * can never share the same BitSet, so each depth bucket is guaranteed
 * overlap-free.  This also eliminates the class of Zobrist hash collisions
 * that could occur between BitSets of different cardinalities, and allows
 * per-depth eviction thresholds sized precisely to C(numCols, depth).
 */
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;


class OrderGraphCacheSplit {
    // per-depth caches: caches[d] holds all sub-problems at depth d
    // (i.e. all usedCols BitSets whose cardinality == d).
    // Length: numCols + 1  (depth 0 .. numCols inclusive).
    HashMap<OrderGraphHash, Integer>[] caches;

    // per-depth LFU frequency maps (only populated when cacheEvictionEnabled)
    HashMap<OrderGraphHash, Integer>[] freqs;

    // per-depth maximum sizes: maxSizes[d] = C(numCols, d)
    int[] maxSizes;

    // stats
    int hits;
    int misses;
    int evictions;

    // timing (nanoseconds)
    long hashingTime;
    long cacheEvictionTime;

    // per-operation timing (nanoseconds, excluding hashing overhead)
    long getTime;
    long putTime;
    long containsTime;

    boolean customHashingEnabled;
    boolean cacheEvictionEnabled;

    @SuppressWarnings("unchecked")
    public OrderGraphCacheSplit(int numRows, int numCols, EnumeratorConfig config) {
        this.customHashingEnabled = config.customHashingEnabled;
        this.cacheEvictionEnabled = config.cacheEvictionEnabled;

        // allocate one bucket per depth level
        this.caches   = new HashMap[numCols + 1];
        this.freqs    = new HashMap[numCols + 1];
        this.maxSizes = new int[numCols + 1];
        for (int d = 0; d <= numCols; d++) {
            this.caches[d]   = new HashMap<>();
            this.freqs[d]    = new HashMap<>();
            this.maxSizes[d] = binomial(numCols, d);
        }

        this.hits      = 0;
        this.misses    = 0;
        this.evictions = 0;

        this.hashingTime       = 0;
        this.cacheEvictionTime = 0;
        this.getTime           = 0;
        this.putTime           = 0;
        this.containsTime      = 0;

        // prime the fully-assigned (trivial) sub-problem at depth numCols: cost = 0
        BitSet trivialSet = allCols(numCols);
        OrderGraphHash trivialKey = new OrderGraphHash(trivialSet, 0);
        this.caches[numCols].put(trivialKey, 0);
        if (this.cacheEvictionEnabled)
            this.freqs[numCols].put(trivialKey, 1);
    }

    static int binomial(int n, int k) {
        if (k < 0 || k > n) return 0;
        if (k == 0 || k == n) return 1;
        k = Math.min(k, n - k);
        long result = 1;
        for (int i = 0; i < k; i++) {
            result = result * (n - i) / (i + 1);
            if (result > Integer.MAX_VALUE) return Integer.MAX_VALUE;
        }
        return (int) result;
    }

    static BitSet allCols(int numCols) {
        BitSet all = new BitSet(numCols);
        all.set(0, numCols);
        return all;
    }

    /**
     * Returns the cached cost for the given key, or null if not present.
     * Selects the bucket by key.cardinality().
     */
    public Integer getIfPresent(BitSet key) {
        int depth = key.cardinality();
        OrderGraphHash hashKey = new OrderGraphHash(key, 0);
        Integer val = this.caches[depth].get(hashKey);
        if (val != null) {
            this.hits++;
            if (this.cacheEvictionEnabled)
                this.freqs[depth].merge(hashKey, 1, Integer::sum);
        } else {
            this.misses++;
        }
        return val;
    }

    public int get(BitSet key) {
        long t0 = System.nanoTime();
        int depth = key.cardinality();
        OrderGraphHash mk = new OrderGraphHash(key, 0);
        this.hashingTime += System.nanoTime() - t0;
        return this.caches[depth].get(mk);
    }

    public void put(BitSet key, int value) {
        long t0 = System.nanoTime();
        int depth = key.cardinality();
        OrderGraphHash mk = new OrderGraphHash(key, 0);
        long t1 = System.nanoTime();
        this.hashingTime += t1 - t0;
        this.caches[depth].put(mk, value);
        this.putTime += System.nanoTime() - t1;
        if (this.cacheEvictionEnabled) {
            this.freqs[depth].put(mk, 1);
            long t2 = System.nanoTime();
            maybeEvict(depth);
            this.cacheEvictionTime += System.nanoTime() - t2;
        }
    }

    /** Total number of entries across all depth buckets. */
    public int size() {
        int total = 0;
        for (HashMap<OrderGraphHash, Integer> bucket : this.caches)
            total += bucket.size();
        return total;
    }

    /**
     * If the bucket for the given depth exceeds its C(numCols, depth) limit,
     * evict the bottom 10% of entries by frequency (least frequently used).
     */
    void maybeEvict(int depth) {
        HashMap<OrderGraphHash, Integer> cache = this.caches[depth];
        HashMap<OrderGraphHash, Integer> freq  = this.freqs[depth];
        if (cache.size() <= this.maxSizes[depth]) return;

        List<java.util.Map.Entry<OrderGraphHash, Integer>> entries =
            new ArrayList<>(freq.entrySet());
        entries.sort(java.util.Map.Entry.comparingByValue());

        int toEvict = Math.max(1, cache.size() / 10);
        for (int i = 0; i < toEvict && i < entries.size(); i++) {
            OrderGraphHash victim = entries.get(i).getKey();
            cache.remove(victim);
            freq.remove(victim);
            this.evictions++;
        }
    }
}
