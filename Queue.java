import java.util.NoSuchElementException;

/**
 * Queue<T>
 *
 * A double-ended priority queue with a fixed capacity, set once at
 * construction. Backed by a single array using the interval-heap layout:
 * node k lives at array positions (2k, 2k+1) -- 2k holds its "lo" value,
 * 2k+1 its "hi" value. Node k's parent is node (k-1)/2 (integer division);
 * its children are nodes 2k+1 and 2k+2.
 *
 * Invariants, maintained at all times:
 *   1. within a node:        lo <= hi
 *   2. parent -> child:      lo(parent) <= lo(child)  and  hi(child) <= hi(parent)
 *
 * Consequence of invariant 2, applied up to the root: arr[0] is always the
 * global minimum, arr[1] is always the global maximum -- O(1) peek at
 * both ends.
 */
public class Queue<T extends Comparable<T>> {

    private final Object[] arr;
    private int size;

    public Queue(int capacity) {
        if (capacity < 1) throw new IllegalArgumentException("capacity must be >= 1");
        arr = new Object[capacity];
        size = 0;
    }

    public int size() { return size; }

    public int capacity() { return arr.length; }

    public boolean isEmpty() { return size == 0; }

    public boolean isFull() { return size == arr.length; }

    @SuppressWarnings("unchecked")
    private T at(int i) { return (T) arr[i]; }

    private void put(int i, T v) { arr[i] = v; }

    private void swap(int i, int j) {
        T tmp = at(i);
        put(i, at(j));
        put(j, tmp);
    }

    /** O(1) peek at the current minimum. */
    public T peekMin() {
        if (size == 0) throw new NoSuchElementException("heap is empty");
        return at(0);
    }

    /** O(1) peek at the current maximum. */
    public T peekMax() {
        if (size == 0) {
            return null;
        } else {
            return size == 1 ? at(0) : at(1);
        }
    }

    // -------------------------------------------------------------------
    // qInsert -- grows the heap by one. Not part of fastmurty's queue.c
    // (which never needs to grow), but the natural operation to add here.
    // -------------------------------------------------------------------

    public void qInsert(T value) {
        if (size >= arr.length) throw new IllegalStateException("heap is at capacity");
        int pos = size;
        size++;
        put(pos, value);

        if ((pos & 1) == 0) {
            // pos opens a brand-new node with only a lo value -- it has no
            // hi partner yet, so it must be checked against the PARENT's
            // full interval, not just the parent's lo.
            int childNode = pos / 2;
            if (childNode == 0) return; // this IS the root, nothing to check

            int parentNode = (childNode - 1) / 2;
            int parentLoPos = 2 * parentNode;
            int parentHiPos = parentLoPos + 1;

            if (at(pos).compareTo(at(parentLoPos)) < 0) {
                swap(pos, parentLoPos);
                siftUpLo(parentLoPos);
            } else if (at(pos).compareTo(at(parentHiPos)) > 0) {
                swap(pos, parentHiPos);
                siftUpHi(parentHiPos);
            }
            // else: already inside the parent's interval, done

        } else {
            // pos pairs with the lo-only node already sitting at pos-1
            int loPos = pos - 1;
            if (at(pos).compareTo(at(loPos)) < 0) {
                swap(pos, loPos);
                // the value now at loPos is the newly-inserted one -- it's
                // never been checked against any ancestor's lo before,
                // unlike the value that was already there. Give it its
                // own trip up the lo chain.
                siftUpLo(loPos);
            }
            siftUpHi(pos);
        }
    }

    /** Percolates the value at `pos` up the lo chain (ancestors' lo slots only). */
    private void siftUpLo(int pos) {
        int childPos = pos;
        while (true) {
            int childNode = childPos / 2;
            if (childNode == 0) break; // reached the root, no parent
            int parentNode = (childNode - 1) / 2;
            int parentLoPos = 2 * parentNode;
            if (at(childPos).compareTo(at(parentLoPos)) < 0) {
                swap(childPos, parentLoPos);
                childPos = parentLoPos;
            } else {
                break;
            }
        }
    }

    /** Percolates the value at `pos` up the hi chain (ancestors' hi slots only). */
    private void siftUpHi(int pos) {
        int childPos = pos;
        while (true) {
            int childNode = childPos / 2;
            if (childNode == 0) break;
            int parentNode = (childNode - 1) / 2;
            int parentHiPos = 2 * parentNode + 1;
            if (at(childPos).compareTo(at(parentHiPos)) > 0) {
                swap(childPos, parentHiPos);
                childPos = parentHiPos;
            } else {
                break;
            }
        }
    }

    // -------------------------------------------------------------------
    // qPopMin -- removes and returns the minimum, shrinking the heap by one.
    // -------------------------------------------------------------------

    public T qPopMin() {
        if (size == 0) throw new NoSuchElementException("heap is empty");
        int newSize = size - 1;

        T removedMin = at(0);
        if (newSize == 0) {
            put(0, null);
            size = 0;
            return removedMin;
        }

        // the last element is about to fall outside the smaller heap --
        // reuse it to fill the gap left at the root, sifting it DOWN
        T candidate = at(newSize);
        siftDown(candidate, newSize);
        put(newSize, null);
        size = newSize;
        return removedMin;
    }

    /**
     * Sifts `candidate` down from the root's lo slot (position 0) along
     * the lo chain, comparing only against children's lo values, and
     * checking each node's paired hi value to keep lo<=hi intact within
     * that node as candidate passes through. `limit` is the heap size
     * candidate is being placed into (already shrunk by one).
     */
    private void siftDown(T candidate, int limit) {
        int pos = 0;
        int childpos = 2;

        while (childpos < limit) {
            T childVal = at(childpos);

            int rightpos = childpos + 2;
            if (rightpos < limit) {
                T rightVal = at(rightpos);
                if (rightVal.compareTo(childVal) < 0) {
                    childpos = rightpos;
                    childVal = rightVal;
                }
            }

            if (candidate.compareTo(childVal) < 0) {
                break; // candidate is correctly placed here
            }

            put(pos, childVal); // promote the smaller child's lo up to fill the gap

            rightpos = childpos + 1;
            if (rightpos < limit) {
                T hiVal = at(rightpos);
                if (candidate.compareTo(hiVal) > 0) {
                    put(rightpos, candidate);
                    candidate = hiVal;
                }
            }

            pos = childpos;
            childpos = (pos << 1) + 2;
        }

        put(pos, candidate);
    }

    // -------------------------------------------------------------------
    // qReplaceMax -- swaps out the current maximum for a smaller value.
    // Caller must guarantee value < peekMax(); size never changes.
    //
    // This is NOT a simple mirror image of qPopMin's siftDown. The
    // asymmetry: qPopMin always descends through complete (lo,hi) pairs
    // on its way down, since a heap of odd size only ever has an
    // INCOMPLETE node as the very last one -- and popMin's candidate,
    // coming from that exact last slot, never needs to "discover" an
    // incomplete node partway through its own descent from the root.
    // qReplaceMax's descent is different: because it walks the HI chain,
    // it can run straight into that same incomplete last node (which has
    // no hi value at all) partway down, in an odd-sized heap, and has to
    // treat that lone lo-only value specially. That's the extra
    // "corner case" handling below with no equivalent on the min side.
    // -------------------------------------------------------------------

    public T qReplaceMax(T value) {
        if (size == 0) throw new NoSuchElementException("heap is empty");

        if (size == 1) {
            put(0, value);
            return value;
        }

        T candidate = value;
        int pos = 1;
        int childpos = 3;

        T loVal = at(0);
        if (candidate.compareTo(loVal) < 0) {
            put(0, candidate);
            candidate = loVal;
        }

        int sizeLimit = (size - 2) & -4;

        while (childpos < sizeLimit) {
            T childVal = at(childpos);

            int rightpos = childpos + 2;
            T rightVal = at(rightpos);
            if (rightVal.compareTo(childVal) > 0) {
                childpos = rightpos;
                childVal = rightVal;
            }

            if (candidate.compareTo(childVal) > 0) {
                break;
            }

            put(pos, childVal);

            loVal = at(childpos - 1);
            if (candidate.compareTo(loVal) < 0) {
                put(childpos - 1, candidate);
                candidate = loVal;
            }

            pos = childpos;
            childpos = (pos << 1) + 1;
        }

        // corner case: in an odd-sized heap, the very last node has only
        // a lo slot filled (no hi partner) -- handle it as a special
        // comparison target rather than a normal hi-chain sibling
        int cornerstate = size - childpos;
        if (cornerstate >= 0) {
            if (cornerstate == 0) {
                childpos -= 1;
            } else if (cornerstate == 2) {
                int rightpos = childpos + 1;
                if (at(rightpos).compareTo(at(childpos)) > 0) {
                    childpos = rightpos;
                }
            }
            if (candidate.compareTo(at(childpos)) < 0) {
                put(pos, at(childpos));
                pos = childpos;
                if ((pos & 1) == 1) {
                    loVal = at(pos - 1);
                    if (candidate.compareTo(loVal) < 0) {
                        put(pos - 1, candidate);
                        candidate = loVal;
                    }
                }
            }
        }

        put(pos, candidate);
        return at(1);
    }
}