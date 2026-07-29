#include "lib.h"

static int process(int x) { return x - 1; }

int engine_process(int x) { return process(x) * 2; }

int helper(int x) { return x + 1; }
