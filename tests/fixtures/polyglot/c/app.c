#include "lib.h"

static int process(int x) { return helper(x); }

int run(void) { return engine_process(process(1)); }
