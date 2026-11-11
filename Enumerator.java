import java.util.List;

/**
 * Common contract for every top-k assignment enumerator
 * (MurtyEnumerator, FastCacheEnumerator, JavaFastmurty, OrderGraph*, ...).
 *
 * Each implementing class only needs to add "implements Enumerator" --
 * they already have a matching enumerate(int k) method.
 */
@FunctionalInterface
public interface Enumerator {
    /**
     * @param k number of solutions to enumerate
     * @return the top-k solutions, best first
     */
    List<AssignmentSolution> enumerate(int k);
}
