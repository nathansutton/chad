public class App {
    static int process(int x) {
        return Util.helper(x);
    }

    static int run() {
        return new Engine().process(process(1));
    }
}
