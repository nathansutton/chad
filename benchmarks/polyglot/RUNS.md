# Published runs

Run output is not committed. A run a write-up cites is bundled by `publish.py`, uploaded
to the public dataset
[`nathansutton/chad-polyglot-runs`](https://huggingface.co/datasets/nathansutton/chad-polyglot-runs),
and recorded here as one row: what ran, and the sha256 of its `trials.jsonl`. `fetch.py --label <label>` downloads a row's bundle into `_runs/<label>/`
and refuses it if the hash differs. Append-only: a rerun gets a new label.

The task lists runs here cite, each committed before any arm ran on it:

- `subsets/harness-36.txt`, sha256 `2d9ac04185b2f699ddea62571b1de1ea1edf10f8a3c809e80819b6e944bc65b1`:
  6 tasks per language, `stats.py subset --per-language 6 --seed harness-1`.
- `subsets/harness-3.txt`, sha256 `b848c8f293d8cf400fea62ee069a701e77dd1629d3cce97fee5246975c222011`:
  3 of those 36 in 3 languages, `stats.py subset --spread 3 --seed harness-3 --from subsets/harness-36.txt`.

| label | date | chad | harness | model | tasks × reps | dataset path | trials.jsonl sha256 |
|---|---|---|---|---|---|---|---|
| design-sample | 2026-09-21 | 2.2.0 (374ec56+dirty) | chad | Qwen3.8-27B-Ternary-Bonsai-2 | 12 × 1 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/design-sample | 285ea7a3cb7dd9d68fa5557497ac1e9f62088ff4f8128902e6c67ebbf5065fa1 |
| q3kxl-chad-h36 | 2026-09-24 | 2.2.0 (e16fd27) | chad | unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf | 36 × 1 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/q3kxl-chad-h36 | a2c89c9efa0169d5fd68e1c915c96b22956591e8cb5dfe299f31120af466cd72 |
| tern-chad-h36 | 2026-09-25 | 2.2.0 (e16fd27) | chad | nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX | 36 × 1 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/tern-chad-h36 | 911b41d9982db504ed3e8d416b0c6026ace32982ae4a3ac93dc7dbd590bcffb9 |
| q3-chad-mlx | 2026-09-23 | 2.2.0 (0f0a161) | chad | nathansutton/Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX | 3 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/q3-chad-mlx | ef153dbeb86de18fc7bebae380c67527d8962464b2dee37d04623c5c85cd75b1 |
| iq3-chad | 2026-09-24 | 2.2.0 (410cad8) | chad | unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-IQ3_XXS.gguf | 3 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/iq3-chad | 6b44c0bc068a27a1d3904e348a9c0ec33786423081253b0e874f6985c24e743f |
| tern-chad | 2026-09-24 | 2.2.0 (410cad8) | chad | nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX | 3 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/tern-chad | 7fbb4f3c6c7577120e73ae3571f473b1ce97e5379513ae580c89b7b78cd0614e |
| q3kxl-chad | 2026-09-24 | 2.2.0 (3d0d0a4) | chad | unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf | 3 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/q3kxl-chad | 0780f999edac0a836f9b57c06e397a8ab6bff923543e79d0ed354c28cf26140c |
| h36-chad-llama | 2026-09-29 | 2.3.0 (9315d60) | chad-llama | qwen3.8-27b-local | 36 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/h36-chad-llama | ec825687b9d866ecc202cd80b1c58415ea918959c6a5a7af550242a0c10f05ce |
| h36-pi | 2026-09-29 | 2.3.0 (0db2368) | pi | ternary-bonsai-2-27b | 36 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/h36-pi | 9090c48fdc791cf83f33a683f883044604dba21191a6b512c8625231bb7e528b |
| h36-opencode | 2026-09-30 | 2.3.0 (9315d60) | opencode | qwen3.8-27b-local | 36 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/h36-opencode | 619f3a85142ab929694f663c4d05f754d6f765413c74f74c8abb1c9a0452ff42 |
| h36-dsh | 2026-09-30 | 2.3.0 (9315d60) | dsh | qwen3.8-27b-local | 36 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/h36-dsh | 739e3e566851f322ff34e3c70db7cd13b764c4aa257e5cf1cb0804ee8dfb6d27 |
| h36-goose | 2026-10-01 | 2.3.0 (9315d60) | goose | qwen3.8-27b-local | 36 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/h36-goose | f36fec8ffe3c45f939ebaa862bed9a608900504a797d301216d9096f17e1f379 |
| h36-mini | 2026-10-02 | 2.3.0 (9315d60) | mini | qwen3.8-27b-local | 36 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/h36-mini | ef53a9c4100c8e49c1a0c91c56e41a684e54daf661eb8c590f9289734fd2df26 |
| h36-crush | 2026-10-02 | 2.3.0 (9315d60) | crush | qwen3.8-27b-local | 36 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/h36-crush | e1b91819ce5639909b322986973aa451f1807cae3ee05b62c4d1968a4d95a129 |
| h36-chad-mlx | 2026-10-02 | 2.3.0 (9315d60) | chad | unsloth/Qwen3.8-27B-GGUF/Qwen3.8-27B-UD-Q3_K_XL.gguf | 36 × 3 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/h36-chad-mlx | c2c41deddd65b4fc150385cab87d28b3488e615954e2a210964f3c5a0643beb2 |
