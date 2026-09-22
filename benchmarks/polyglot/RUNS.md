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

Append-only. A row whose dataset path no longer resolves is a bug in the row, not history to
tidy; a rerun gets a new label.

| label | date | chad | harness | model | tasks × reps | dataset path | trials.jsonl sha256 |
|---|---|---|---|---|---|---|---|
| design-sample | 2026-09-21 | 2.2.0 (374ec56+dirty) | chad | Qwen3.8-27B-Ternary-Bonsai-2 | 12 × 1 | hf://datasets/nathansutton/chad-polyglot-runs/polyglot/design-sample | 285ea7a3cb7dd9d68fa5557497ac1e9f62088ff4f8128902e6c67ebbf5065fa1 |
