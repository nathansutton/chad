# `benchmarks/matrix/` — archived

The nine-harness grid behind "Nine coding harnesses vs. your laptop" (same weights, same
MacBook, eight Exercism tasks, three nights in September 2026) is no longer in the tree.
Everything the write-up cites, including the runner, the forcing proxy, the scorecard, the
tasks and every committed row, is at the tag
[`archive/matrix-nine-harnesses`](https://github.com/nathansutton/chad/tree/archive/matrix-nine-harnesses/benchmarks/matrix):

    git checkout archive/matrix-nine-harnesses
    uv run python benchmarks/matrix/run.py  setup|smoke|llama|mlx|table
    uv run python benchmarks/matrix/scorecard.py

chad's one agent eval is now [`benchmarks/polyglot`](../polyglot/README.md).
