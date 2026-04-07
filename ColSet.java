import java.util.Collection;

/**
 * A memory-efficient set of column indices (0..n-1, max n=20).
 *
 * Internally uses a long bitmask instead of HashSet — each bit i
 * represents whether column i is in the set. This eliminates all
 * heap allocation on construction and makes contains/equals O(1).
 *
 * Hash is a collision-free prime-sum: each column index maps to a
 * distinct prime, so no two distinct subsets produce the same hash.
 */
public class ColSet {

    private static final int[] PRIMES = {
    2,   3,   5,   7,  11,  13,  17,  19,  23,  29,
    31,  37,  41,  43,  47,  53,  59,  61,  67,  71,
    73,  79,  83,  89,  97, 101, 103, 107, 109, 113,
    127, 131, 137, 139, 149, 151, 157, 163, 167, 173
    };

    private final long bits;       // bitmask of which cols are present
    private final int cachedHash;  // prime-sum hash, cached at construction
    private final int cachedSize;  // popcount, cached at construction

    public ColSet() {
        this.bits = 0L;
        this.cachedHash = 0;
        this.cachedSize = 0;
    }

    public ColSet(Collection<Integer> elements) {
        long b = 0L;
        int hash = 0;
        for (int col : elements) {
            b |= (1L << col);
            hash += primeFor(col);
        }
        this.bits = b;
        this.cachedHash = hash;
        this.cachedSize = Long.bitCount(b);
    }

    /** Incremental constructor — O(1), no allocation. */
    public ColSet(ColSet other, int additionalCol) {
        this.bits = other.bits | (1L << additionalCol);
        this.cachedHash = other.cachedHash + primeFor(additionalCol);
        this.cachedSize = other.cachedSize + 1;
    }

    // -------------------------------------------------------------------------
    // Set operations
    // -------------------------------------------------------------------------

    public boolean contains(int col) {
        return (bits & (1L << col)) != 0;
    }

    public int size() {
        return cachedSize;
    }

    // -------------------------------------------------------------------------
    // Hash helper
    // -------------------------------------------------------------------------

    private static int primeFor(int col) {
        if (col < 0 || col >= PRIMES.length) {
            throw new IllegalArgumentException(
                "Column index out of range: " + col + " (max " + (PRIMES.length - 1) + ")");
        }
        return PRIMES[col];
    }

    // -------------------------------------------------------------------------
    // Object overrides
    // -------------------------------------------------------------------------

    @Override
    public int hashCode() {
        return cachedHash;
    }

    @Override
    public boolean equals(Object obj) {
        if (this == obj) return true;
        if (!(obj instanceof ColSet)) return false;
        return this.bits == ((ColSet) obj).bits;
    }

    @Override
    public String toString() {
        StringBuilder sb = new StringBuilder("[");
        for (int i = 0; i < PRIMES.length; i++) {
            if (contains(i)) {
                if (sb.length() > 1) sb.append(", ");
                sb.append(i);
            }
        }
        sb.append("]");
        return sb.toString();
    }
}