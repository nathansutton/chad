# Troubleshooting

chad runs a small model on your laptop, so a session can ramble, loop, or slow down in
ways a frontier API won't. Every knob that fixes those already exists — this page maps
what you *see* to the one to reach for. The knobs themselves are documented in the
[Configuration reference](configuration.md); this is the symptom index, not a re-explanation.

| You see | What's happening | Reach for |
|---|---|---|
| Long `<think>` before every action | reasoning model, thinking on by default | `--no-think` for well-scoped work, or `--think-budget 1500` to trim only the rambles (see [Turn budgets & think-cap](configuration.md#turn-budgets--think-cap)) |
| A turn burns minutes without landing an edit | runaway turn | the [governor](configuration.md#turn-budgets--think-cap) already ends it for one-shot runs; interactively, ctrl-c and re-scope the ask smaller |
| The same file read over and over | the loop guard will abort it — the ask is likely too vague | a smaller, concrete task; **name the file** you want changed |
| First token slow after `/compact` or a truncation | one-time full re-prefill (Ornith's cache is non-trimmable); the status line shows an advancing % | expected — this is the cache trade chad lives with; see [Design & internals](design.md) |
| Mac swap-storming with other apps open | the 35B (~14 GB peak) plus your apps exceed free RAM | close apps, or force the 9B: `CHAD_MODEL=nathansutton/Ornith-1.0-9B-UD-Q4_K_XL-MLX` (see [Forcing the small model](configuration.md)) |
| chad vanishes mid-turn — no error, **no traceback at all** | an MLX Metal abort under memory pressure; the crash happens below Python, so nothing can print | close memory-heavy apps and re-run; on a borderline box force the 9B: `CHAD_MODEL=nathansutton/Ornith-1.0-9B-UD-Q4_K_XL-MLX`. The crash report (worth attaching to a bug) lands in `~/Library/Logs/DiagnosticReports/` |
| Disk full of old model weights | the Hugging Face cache keeps every revision | `hf cache ls` / `hf cache rm` (older CLIs: `huggingface-cli scan-cache` / `delete-cache`) |
| Wondering what it actually did | the full, redacted trace — **off by default** (privacy-first) | enable with `CHAD_SESSION_LOG=1`, then read `~/.chad/session.log` (rotated) |

The through-line: a small local model rewards a **scoped** ask. "Fix the failing test in
`tests/test_x.py`" lands; "improve my codebase" flails. When in doubt, shrink the task and
name the file.
