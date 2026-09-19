# `benchmarks/stock/` — same model, same Mac, stock engine

The rows behind the comparison table in
[Throughput & performance](../../docs/benchmarks.md#same-model-same-mac-stock-engine):
stock llama.cpp on Unsloth's `Qwen3.8-27B-UD-Q3_K_XL` GGUF, against chad
serial and chad default on its MLX checkpoint of the same recipe — one laptop, one engine
resident at a time, each measured with its own benchmark.

```bash
brew install llama.cpp
uv run python benchmarks/stock/stock.py llama         # llama-bench, pp512 / tg128
uv run python benchmarks/stock/stock.py llama-dflash  # llama-server, serial then DFlash2 drafter
uv run python benchmarks/stock/stock.py chad          # chad-bench, CHAD_NO_DFLASH=1 then default
uv run python benchmarks/stock/stock.py table         # render _runs/*.json as markdown
```

`llama-dflash` needs a llama.cpp with DFlash2 (build 10658 or later). To run one without
replacing a pinned brew install, unpack a release tarball and point `STOCK_LLAMA_BIN` at
it; the drafter (`incoai/Qwen3.8-27B-DFlash2-GGUF`, Q4_K_M) downloads on first use.

Ollama is not a separate arm — it runs llama.cpp's engine underneath, and was measured
without speculative decoding. `_runs/ollama.json` is one hand-run measurement on the same GGUF
(0.32.15, `FROM`-only Modelfile, `num_ctx` 2048, temperature 0, `/api/generate` counters):
96 / 10.9 tok/s, the llama.cpp decode number. No script arm: the import needs ~45 GB of
scratch disk and shows nothing the llama.cpp row does not. Run the arms one at a time — each loads ~13 GB and a 24 GB box cannot hold two. The
measured rows live in `_runs/` as the record; `stock.py`'s docstring has the method.
