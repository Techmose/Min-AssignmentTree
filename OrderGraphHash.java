

/**
 * Wrapper around BitSet that replaces Java's default hash with a
 * Zobrist hash for better bucket distribution at similar cardinalities.
 *
 * equals() still delegates to BitSet.equals() for correctness.
 */
import java.util.BitSet;
class OrderGraphHash {
    final BitSet bits;
    private final int hashCode;
    private final int cost;

    OrderGraphHash(BitSet bits, int cost) {
        this.bits = bits;
        this.cost = cost;
        this.hashCode = this.hashCode();
    }

    @Override
    public int hashCode() {
        long[] words = bits.toLongArray();
        long hash = 0L;
        for (long w : words)
        hash ^= w;
    return Long.hashCode(hash);
    }

    static public int hashCode(BitSet key) {
        long[] words = key.toLongArray();
        long hash = 0L;
        for (long w : words)
        hash ^= w;
    return Long.hashCode(hash);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (!(o instanceof OrderGraphHash)) return false;
        return this.bits.equals(((OrderGraphHash) o).bits);
    }
}
