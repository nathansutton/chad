# Published runs

Run output is not committed. A run a write-up cites is bundled by `publish.py` (rows
always, trajectories when the write-up reads them), uploaded to the dataset repository, and
recorded here as one row: what ran, and the sha256 of the `trials.jsonl` it produced.
`fetch.py --label <label>` downloads a row's bundle into `_runs/<label>/` and refuses it if
the hash differs, so `stats.py` and `trace.py` read exactly the rows that were cited.

A run on a task list records the list's path and sha256 in its `meta.json`. The lists that
runs here cite, each committed before any arm ran on it:

- `subsets/harness-36.txt` — sha256 `2d9ac04185b2f699ddea62571b1de1ea1edf10f8a3c809e80819b6e944bc65b1`
  — the harness comparison: 6 tasks per language, `stats.py subset --per-language 6 --seed harness-1`.
- `subsets/harness-3.txt` — sha256 `b848c8f293d8cf400fea62ee069a701e77dd1629d3cce97fee5246975c222011`
  — the first harness night: 3 of those 36, in 3 languages,
  `stats.py subset --spread 3 --seed harness-3 --from subsets/harness-36.txt`.

Append-only. A row whose dataset path no longer resolves is a bug in the row, not history to
tidy; a rerun gets a new label.

| label | date | chad | harness | model | tasks × reps | dataset path | trials.jsonl sha256 |
|---|---|---|---|---|---|---|---|
| design-sample | 2026-09-21 | 2.2.0 (374ec56+dirty) | chad | Qwen3.8-27B-Ternary-Bonsai-2 | 12 × 1 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/design-sample | 285ea7a3cb7dd9d68fa5557497ac1e9f62088ff4f8128902e6c67ebbf5065fa1 |
| q3kxl-chad-h36 | 2026-09-24 | 2.2.0 (e16fd27) | chad | unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf | 36 × 1 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/q3kxl-chad-h36 | a2c89c9efa0169d5fd68e1c915c96b22956591e8cb5dfe299f31120af466cd72 |
| tern-chad-h36 | 2026-09-25 | 2.2.0 (e16fd27) | chad | nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX | 36 × 1 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/tern-chad-h36 | 911b41d9982db504ed3e8d416b0c6026ace32982ae4a3ac93dc7dbd590bcffb9 |
| q3-chad-mlx | 2026-09-23 | 2.2.0 (0f0a161) | chad | nathansutton/Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX | 3 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/q3-chad-mlx | ef153dbeb86de18fc7bebae380c67527d8962464b2dee37d04623c5c85cd75b1 |
| iq3-chad | 2026-09-24 | 2.2.0 (410cad8) | chad | unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf | 3 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/iq3-chad | 6b44c0bc068a27a1d3904e348a9c0ec33786423081253b0e874f6985c24e743f |
| tern-chad | 2026-09-24 | 2.2.0 (410cad8) | chad | nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX | 3 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/tern-chad | 7fbb4f3c6c7577120e73ae3571f473b1ce97e5379513ae580c89b7b78cd0614e |
| q3kxl-chad | 2026-09-24 | 2.2.0 (3d0d0a4) | chad | unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf | 3 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/q3kxl-chad | 0780f999edac0a836f9b57c06e397a8ab6bff923543e79d0ed354c28cf26140c |
