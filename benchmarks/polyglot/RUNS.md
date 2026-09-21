# Published runs

Run output is not committed. A run a write-up cites is bundled by `publish.py` (rows
always, trajectories when the write-up reads them), uploaded to the dataset repository, and
recorded here as one row: what ran, and the sha256 of the `trials.jsonl` it produced.
`fetch.py --label <label>` downloads a row's bundle into `_runs/<label>/` and refuses it if
the hash differs, so `stats.py` and `trace.py` read exactly the rows that were cited.

Append-only. A row whose dataset path no longer resolves is a bug in the row, not history to
tidy; a rerun gets a new label.

| label | date | chad | harness | model | tasks × reps | dataset path | trials.jsonl sha256 |
|---|---|---|---|---|---|---|---|
| design-sample | 2026-09-21 | 2.2.0 (374ec56+dirty) | chad | Qwen3.8-27B-Ternary-Bonsai-2 | 12 × 1 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/design-sample | 285ea7a3cb7dd9d68fa5557497ac1e9f62088ff4f8128902e6c67ebbf5065fa1 |
