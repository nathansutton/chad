#include "lib.hpp"

int process(int x) { return helper(x); }

int run() {
    Engine e;
    return e.process(process(1));
}
