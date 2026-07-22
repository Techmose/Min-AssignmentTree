public class OrderGraphPath {
    OrderGraphPath next;
    int size;
    int value;

    OrderGraphPath() {
        this.next = null;
        this.size = 0;
        this.value = -1;
    }

    OrderGraphPath(OrderGraphPath next, int value) {
        this.next = next;
        this.size = next.size + 1;
        this.value = value;
    }

    public static OrderGraphPath emptyPath() {
        return new OrderGraphPath();
    }

    public boolean isEnd() { return this.size == 0; }
    public OrderGraphPath next() { return this.next; }
    public int size() { return this.size; }
    public int value() { return this.value; }
    public OrderGraphPath append(int value) { return new OrderGraphPath(this,value); }

}
