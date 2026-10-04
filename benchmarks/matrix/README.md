# `benchmarks/matrix/`: archived

The nine-harness grid (same weights, same MacBook, eight Exercism tasks, three nights in
September 2026) is no longer in the tree. The runner, the forcing proxy, the scorecard,
the tasks and every committed row are at the tag
[`archive/matrix-nine-harnesses`](https://github.com/nathansutton/chad/tree/archive/matrix-nine-harnesses/benchmarks/matrix):

    git checkout archive/matrix-nine-harnesses
    uv run python benchmarks/matrix/run.py  setup|smoke|llama|mlx|table
    uv run python benchmarks/matrix/scorecard.py

chad's one agent eval is now [`benchmarks/polyglot`](../polyglot/README.md).
