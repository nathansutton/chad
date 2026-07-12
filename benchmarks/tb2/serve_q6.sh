#!/bin/bash
# Serve the Q6 quant for the higher-bit TB2 arm (48 GB+ Apple Silicon Mac).
#
# Default: pull the prebuilt quant from Hugging Face (~28.5 GB on first run) —
# nothing to build on the host. Pass a local dir (e.g. one built by
# quantize_q6.py) as $1 to serve that instead.
#
# Sizing for a 48 GB Mac (~36 GB Metal working-set budget): ~28.5 GB weights +
# a 120k-token window's KV/scratch (~5.3 GB at the measured ~41 KB/token stock-mlx
# slope) ≈ 34.5 GB peak, with the engine's memory clamps and pressure-aware
# governor underneath. caffeinate keeps the box awake for overnight passes;
# host/port match run_tb2.sh's host.docker.internal default.
set -u
cd "$(dirname "$0")/../.."   # repo root
MODEL="${1:-nathansutton/Ornith-1.0-35B-Q6-MLX}"
exec caffeinate -is env \
  CHAD_MODEL="$MODEL" \
  CHAD_MAX_CONTEXT="${CHAD_MAX_CONTEXT:-120000}" \
  uv run chad --serve --host 0.0.0.0 --port 8080
