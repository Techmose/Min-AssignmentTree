/**
 * Cache for sub-problem costs used by OrderGraphEnumerator.
 *
 * Supports hashing and LFU-based cache eviction.
 */
import java.util.ArrayList;
import java.util.BitSet;
import java.util.HashMap;
import java.util.List;

class OrderGraphCache {
    // flat cache: OrderGraph BitSet -> cost
    HashMap<OrderGraphHash, Integer> cache;

    // LFU frequency map (only populated when cacheEvictionEnabled)
    HashMap<OrderGraphHash, Integer> freq;

    // maximum global cache size: C(n, n/2)
    int maxSize;

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

    public OrderGraphCache(int numRows, int numCols, EnumeratorConfig config) {
        this.customHashingEnabled = config.customHashingEnabled;
        this.cacheEvictionEnabled = config.cacheEvictionEnabled;

        this.cache = new HashMap<>();
        this.freq  = new HashMap<>();

        this.maxSize = binomial(numCols, numCols / 2);

        this.hits      = 0;
        this.misses    = 0;
        this.evictions = 0;

        this.hashingTime       = 0;
        this.cacheEvictionTime = 0;
        this.getTime      = 0;
        this.putTime      = 0;
        this.containsTime = 0;

        // prime cache with the fully-assigned (trivial) sub-problem: cost = 0
        BitSet trivialSet = allCols(numCols);
        OrderGraphHash trivialKey = new OrderGraphHash(trivialSet, 0);
        this.cache.put(trivialKey, 0);
        if (this.cacheEvictionEnabled)
            this.freq.put(trivialKey, 1);
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
     */
    public Integer getIfPresent(BitSet key) {
        // Bug 1 fixed: construct a proper OrderGraphHash to look up in the map,
        // and use that same key object (not the int cost) for freq.merge().
        OrderGraphHash hashKey = new OrderGraphHash(key, 0);
        Integer val = this.cache.get(hashKey);
        if (val != null) {
            this.hits++;
            if (this.cacheEvictionEnabled)
                this.freq.merge(hashKey, 1, Integer::sum);
        } else {
            this.misses++;
        }
        return val;
    }

    public int get(BitSet key) {
        long t0 = System.nanoTime();
        // Bug 2 fixed: wrap in OrderGraphHash instead of using raw int hashCode
        OrderGraphHash mk = new OrderGraphHash(key, 0);
        this.hashingTime += System.nanoTime() - t0;
        return this.cache.get(mk);
    }

    public void put(BitSet key, int value) {
        long t0 = System.nanoTime();
        // Bug 2 fixed: wrap in OrderGraphHash instead of using raw int hashCode
        OrderGraphHash mk = new OrderGraphHash(key, 0);
        long t1 = System.nanoTime();
        this.hashingTime += t1 - t0;
        this.cache.put(mk, value);
        this.putTime += System.nanoTime() - t1;
        if (this.cacheEvictionEnabled) {
            // Bug 3 fixed: was `zk` (undefined), should be the constructed key `mk`
            this.freq.put(mk, 1);
            long t2 = System.nanoTime();
            maybeEvict();
            this.cacheEvictionTime += System.nanoTime() - t2;
        }
    }

    public int size() {
        return this.cache.size();
    }

    /**
     * If the cache exceeds maxSize, evict the bottom 10% of entries by
     * frequency (least frequently used).
     */
    void maybeEvict() {
        if (this.cache.size() <= this.maxSize) return;

        // Bug 4 fixed: ArrayList was used without being imported
        List<java.util.Map.Entry<OrderGraphHash, Integer>> entries =
            new ArrayList<>(this.freq.entrySet());
        entries.sort(java.util.Map.Entry.comparingByValue());

        int toEvict = Math.max(1, this.cache.size() / 10);
        for (int i = 0; i < toEvict && i < entries.size(); i++) {
            OrderGraphHash victim = entries.get(i).getKey();
            this.cache.remove(victim);
            this.freq.remove(victim);
            this.evictions++;
        }
    }
}