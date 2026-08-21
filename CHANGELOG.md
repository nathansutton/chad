# Changelog

Notable, user-visible changes.

## [Unreleased]

### DFlash2 block drafter

Decoding on the shipped model now speculates with a **DFlash2 block drafter** instead of the
checkpoint's MTP head. The drafter (`incoai/Qwen3.8-27B-DFlash2`, 1.9B, fetched once on
first run with the same consent prompt as the model, quantized to a ~1.1 GB sidecar) reads
the main model's residual stream at five tapped layers and proposes a whole block of tokens
in **one** forward; the MTP head had to chain one step per token and its acceptance decayed
past depth 3. Verification, rollback and the exact-acceptance rule are unchanged — both
drafters now run on one shared speculative loop (`Engine._generate_spec`) — so greedy output
is token-identical to plain decoding and sampled output keeps the true distribution.

- Measured on an M4 Pro, shipped 3-bit quant, one load, same prompts, 384-token decodes,
  medians: greedy serial 17.7 → MTP 26.1 → **DFlash2 47.7 tok/s** (2.7×); at the thinking
  sampling preset MTP 22.4 → **41.4**; non-thinking preset 23.7 → 42–43; code 24 → 35–37.
  (Before the matmul kernel below: 28.1 / 27.3 at width 4, the ladder's optimum.)
- The drafter always proposes its full block of 7; a per-round schedule picks how many to
  verify from recent acceptance (`CHAD_DFLASH_ADAPTIVE=0` fixes it, `CHAD_DFLASH_DRAFT=N`
  caps it). At the sampled default a fixed full block swings 12–45 tok/s per prompt; the
  schedule holds a 20 tok/s floor for ~5% on hot greedy prose. `CHAD_NO_DFLASH=1` restores
  the MTP path; `CHAD_DFLASH_PATH` / `CHAD_DFLASH_REPO` point at other drafter weights;
  `python -m chad.mlx_dflash <dir>` builds a sidecar.
- Memory: +1.1 GB resident for the drafter (its context cache is a 2048-row ring, ~40 MB);
  on a 24 GB box that is ~30k tokens of context ceiling.

### Small-M matmul kernel for speculative verify

Verifying a draft of k tokens is a k+1-row forward, and stock `quantized_matmul` priced
every extra row at ~33 ms on the shipped model — the ladder that pinned the block drafter's
width at 4. `mlx_qmm_mma.py` brings in avlp12's `qmm_mma4` MMA kernel (via mlx-dspark, MIT),
which dequantizes each weight group once for all rows, with a width-generic unpack so the
3-bit body and the 5-bit `lm_head` qualify alongside 4/8-bit weights. A load-time probe
races it against the stock kernel per weight shape and per width on the running machine,
keeps only the winners (cached), and everything else stays stock. On the shipped model every
eligible shape wins from six rows up (3-bit MLP 1.31× at six rows, 1.64× at eight; the 5-bit
`lm_head` 1.55× / 1.87×), which flattens the verify ladder from width 5 to 8 — and that is
what moves the block drafter from width 4 (28–29 tok/s) to its full block (42–48).
`CHAD_NO_QMM_MMA=1` is the A/B arm; `CHAD_QMM_MMA_RECAL=1` re-probes after an mlx upgrade.

### A clip is a loan, not a deletion

chad truncates tool output in three places — bash's own head/tail budget, the per-result
backstop cap, and compaction's aging passes. Only the first one used to keep the bytes it
dropped. Measured over the banked session archive: of the results compaction trimmed,
**7.6% were later re-fetched with a byte-identical call** (17 of 225, across 7 sessions) —
a dead turn plus a full re-prefill each, on a box where a turn is 30–60 s. The clips that
already carried a pointer were re-run **0** times.

- **Compaction's trims now carry an executable pointer** (`trim_spill`). Before a tool
  result is head/tail-trimmed, the full original goes to a spill file and the trimmed
  message names it, so the omitted middle costs a `grep` instead of a repeat of the call.
- **The per-result backstop cap joins the same contract** (`result_spill`), and now keeps
  a **tail** as well as a head — a head-only clip of a long error kept the preamble and
  dropped the sentence that said what went wrong.
- **One spill store, one budget.** `spill.py` holds the file/permission/retention
  discipline all three call sites share: per-process session dirs, per-kind file counts, a
  dir-wide byte cap, 0600, and a 7-day sweep of dirs orphaned by dead processes. A body
  that is already spilled re-points at its file rather than being written twice.

### Fixed

- **`/compact` raised `AttributeError`** the moment a session had an untrimmed tool
  result: the manual reclaim path filtered on a module constant that no module defines.
  It had no test, which is how it shipped; it has one now.

### Added

- **`/ctx`** — where the context window actually went, in tokens of the model's own
  tokenizer: system prompt, tool schemas, history, and inside history the think residue,
  the tool results, and the last step's results. `context N` says how full the window is
  and nothing about why; a 40k skill body, an eager MCP server's schemas and a transcript
  of think blocks all read identically on the bare gauge. Read-only and model-free, so it
  works mid-turn — which is when the question gets asked.

## [2.0.0] — 2026-08-19

**The lean release, and a new brain.** chad 2.0.0 is the drastic simplification the 1.x
measurement program earned: every lever pack, alternate tool dialect, and scaffolding layer
was measured against the bare model + tool loop with pre-registered paired contrasts, and
nothing beat it. What survives is the bare loop plus the eight bash-route levers that make
the result channel honest — and those are now the default, all ON. Around that smaller
harness the release changes the model chad runs and lands a substantially faster
decoder.

**Two breaking changes to read before upgrading**: the default model is different (a fresh
~12 GB download; the old weights can be deleted), and chad no longer supports Macs below
24 GB.

### Skills are slash commands now

- **The skill catalog is out of the system prompt.** Agent Skills used to ride in every
  prompt as an `<available_skills>` catalog so the model could select one, and be loaded
  through an `activate_skill` tool. Both are gone. Every installed skill is now a slash
  command: type `/` to see them all in the completion menu, `/ship` to run one,
  `/investigate the flaky test` to run one against a specific ask.
- **The measurement:** against 62 installed skills the catalog cost **4,751 tokens** and
  the tool's enum another 294 — together 65% of the entire system prompt, on every turn of
  every session. The system prompt went **7,975 → 2,824 tokens**. On a 24 GB Mac, where
  the usable window is ~50k, that is 10% of the window handed back, and it no longer grows
  with the number of skills you have installed.
- **A loaded skill is a user turn**, not a tool result and not a prompt block — scoped to
  the task, reclaimable by compaction once it is done (though still protected while the
  task is live: you asked for that guidance by name).
- **The load prints its price.** `loaded skill ship (41,238 tokens)` — big skills are big,
  and on this hardware you should see that when you spend it, not infer it later from a
  surprise compaction.
- This is a deliberate divergence from the Agent Skills spec's tier-1 disclosure. chad
  reads the same `SKILL.md` format from the same directories; it just declines to spend
  60% of a small model's prompt letting it guess at a choice you can make from a menu.

### The purge

- **One tool surface.** The model sees exactly `bash`, `edit`, `write`,
  `write_todos`, and `done` (plus MCP tools where configured).
  Removed wholesale: the dedicated `read`/`grep`/`glob` tools, the line-addressed
  edit family (`replace_lines`/`insert_lines`), the tree-sitter symbolic tools
  (`repo_map`/`overview`/`view_symbol`/`find_symbol`/`definition`/`find_refs`/
  `hover`/`replace_symbol`/`insert_symbol`/`rename_symbol`), the LSP client, and the
  `task` sub-agent system. The former `CHAD_LEAN` arm is simply what chad is now.
- **Eight levers, all ON.** `env_manifest`, `bash_read_skeleton`,
  `bash_empty_diagnose`, `bash_trim_keep_failures`, `verify_baseline`,
  `bash_line_clip`, `edit_miss_diagnose`, `rg_replace_flag_note`. `CHAD_DISABLE=a,b`
  (or `all`) switches them off for leave-one-out ablation; `CHAD_ENABLE` is gone.
  The other ~48 registered levers of 1.x — none of which earned a positive
  pre-registered contrast — were removed along with their machinery.
- **Safety is no longer optional.** The macOS Seatbelt sandbox for yolo-mode bash,
  shadow-git edit checkpoints (`/undo`, `/restore`), the scoped destructive-command
  guard, and credential-shaped env filtering for bash children are all unconditional
  now (escape hatches: `CHAD_NO_SEATBELT`, `CHAD_NO_ENV_GUARD`,
  `CHAD_NO_DESTRUCTIVE_GUARD`; opt-in tier: `CHAD_PROTECT_GIT`).
- **Syntax gate is warn-only.** A mutation that introduces a parse error lands and
  carries the warning in the same tool result; the lever-gated reject/revert tiers
  are gone.
- **Removed A/B knobs**: `CHAD_ENABLE`, `CHAD_LEAN`, `CHAD_NO_SYMBOLS`,
  `CHAD_NO_TASK`, `CHAD_HIDE_TOOLS`, `CHAD_PROFILE` (model profiles are gone; the
  prompt is model-agnostic), `CHAD_OFFLOAD_DIR`, `CHAD_LSP_TIMEOUT`,
  `CHAD_LSP_INIT_TIMEOUT`, `CHAD_LSP_DIAG_TIMEOUT`, `CHAD_LSP_MAX_RSS_MB`.
  `CHAD_THINK_CEILING` now defaults to 0 (off).
- **Dependency drop**: `rustworkx` (the repo-map PageRank is gone; tree-sitter tags
  stay, powering the ambient skeleton/definition-pointer lever).

### Breaking: one model, and it is a different one

- **The default model is now [`Qwen3.8-27B UD-Q3_K_XL-MTP`][model]**, a dense `qwen3_5`
  hybrid (64 layers: 48 GatedDeltaNet + 16 full attention) quantized to 3-bit group-64
  with `lm_head` held at 5-bit. ~12 GB resident, 262k native context. **This is a fresh
  download on first run after upgrading**, and the Ornith weights in your Hugging Face
  cache can be deleted once you are happy (`hf cache` will find them).
- **The Ornith 35B/9B pair is retired**, and with it the RAM-aware pick that chose between
  them and the `--model 35b` / `--model 9b` / `--model 27b` shorthands. `--model` now takes
  `auto` or a repo id / local dir, nothing else — a stale `--model 9b` in a script is now a
  literal (nonexistent) repo id rather than a silent alias, which is deliberate: a
  half-removed alias table would quietly load the default and never tell you.
- **Breaking: 24 GB is the floor.** There is no smaller model to fall back to, so a Mac
  below the target gets a one-line warning at startup and runs anyway — chad advises, it
  does not gate — but ~12 GB of weights plus the ~4.3 GB prefill transient leave too little
  for a usable KV cache. Even at 24 GB the honest window is ~56k of the model's 262k; the
  banner has always stated what you actually got.
- **Why these bits.** On a dense model every parameter is on the critical path for every
  token: no expert redundancy to absorb quantization error, and shrinking the weights is
  the only decode lever there is. The recipe follows what the calibrated GGUF builds of
  this checkpoint agree on — `lm_head` is a second full 1.27B-param tensor (vocab 248,320,
  untied) and is where protection pays, while `embed_tokens` is a lookup table whose
  per-row error never compounds through a matmul and is the cheapest tier to cut. Holding
  both high, as a uniform "sensitive tier" would, spends ~0.78 GB on the tier that needs it
  least — and on a 24 GB box 1 GB of weights is ~16k tokens of context.
- **It ships its trained MTP head** as `mtp.safetensors` alongside the weights, which is
  what the new self-speculative decoder drafts with (below). The conversion is ours rather
  than mlx-community's: identical language weights, minus the 0.86 GB BF16 vision tower
  that mlx-lm's loader discards at load anyway, plus that head.
- **`chad prove` now runs the shipped model** instead of pinning a smaller stand-in. It
  answers "does what I am about to run work on this machine", which a stand-in cannot, and
  it costs no extra download since these are the weights chad was going to fetch anyway.
- **No benchmark claim ships with this release.** The 57% Terminal-Bench 2.1 result quoted
  by 1.x was measured on Ornith 35B and says nothing about the new default, so it has moved
  out of the README into [`docs/benchmarks.md`](docs/benchmarks.md), dated and labelled
  historical. The new default has not been scored yet; per the standing rule, that number
  will flip in its own dedicated commit when the run lands.

[model]: https://huggingface.co/nathansutton/Qwen3.8-27B-UD-Q3_K_XL-MTP-MLX

### Decode speed

- **MTP self-speculative decoding** (`mlx_mtp.py`, default ON where a head exists).
  Drafts with the checkpoint's own trained multi-token-prediction head loaded as a
  sidecar, verifies in one batched forward, and applies exact rejection sampling — so
  greedy output stays token-identical to the non-speculative path and sampled output
  keeps the model's true distribution at any temperature. The shipped model bundles a
  head, so this is on by default; a checkpoint without one is unaffected.
  Measured on an M4 Pro at temp 1.0: 1.38× on quote-heavy spans, 1.11× on novel code,
  1.0× on free prose. `CHAD_NO_MTP=1` disables it.
- **Adaptive draft depth.** Rather than a fixed draft width, a per-round cost model
  picks the depth from recent acceptance, and a full-accept streak qualifies a jump
  onto the measured flat verify-cost plateau (S≥10 verifies cost about the same
  through S=32). Low-acceptance regimes — temp-1 thinking, cold content — collapse to
  depth 1–2 or a free skip, so the schedule degrades to fixed-k behavior rather than
  below it. `CHAD_MTP_ADAPTIVE=0` restores fixed width; `CHAD_MTP_DRAFT` forces one
  (and implies adaptive off); `CHAD_MTP_MAX_DRAFT` lowers the depth cap from its
  default of 31 and `CHAD_MTP_H` seeds the cost model.
- **An S>1 tier for the fused quantized-KV attention kernel** (`mlx_qsdpa.py`).
  Speculative verification dispatches multi-token attention steps, which otherwise
  fall back to dequantizing the whole cache once per attention layer. The fused
  wide path now serves every verify width the draft schedule can pick, up to 24;
  above that the fallback keeps them, because its cost is flat in width while the
  fused kernel's partials slab is not, so the two cross over just under 32.
  Measured on the shipped model as verify-round time (one load, interleaved arms,
  real forwards): **+8.6% at 8k, +19.8% at 20k, +36.4% at 40k** for a width-10
  round — the first rung of the plateau, and the one the schedule reaches most —
  tapering to +2.2/+6.8/+12.3% at width 24. The gain grows with context because
  the fallback re-reads the whole cache and this kernel reads it packed. Widths
  the fused path never serves are unchanged within ±0.5%. Numerics stay at
  output-dtype rounding level against an fp32 reference, the same acceptance class
  as the fallback — though, like any two kernels for the same math, not bit-identical
  to it, so flipping the knob can move a greedy near-tie.
  `CHAD_NO_QSDPA_WIDE` / `CHAD_NO_QSDPA_WIDE_SGM` disable the tiers,
  `CHAD_QSDPA_WIDE_KERNEL` forces the single-kernel variant.
- **Verify-width kernel warming.** The attention kernel is templated on the verify
  width, so a width that has never run is a Metal compile on the critical path of a
  real step. Load now warms exactly the widths *this* configuration can dispatch —
  the ones the draft schedule can pick, intersected with the ones the fused kernel
  serves rather than the dequantize fallback — not the union of everything.
  `CHAD_NO_KERNEL_WARM=1` opts out.
- **Wide prompt-lookup decoding is now opt-in** (`CHAD_USE_PLD=1`), not default.
  PLD drafts from context recurrence, so it can only accelerate text that already
  appeared — and on real agentic traces that is a minority of what this agent
  generates (~62–66% of generated tokens are `<think>`, and reasoning prose replays
  at ~2.3%). Whole-session contribution measured at +2.2% of generated tokens. It
  does not compose with MTP (one generate loop each), so on a checkpoint that
  self-speculates, MTP is the better of the two.
- **CATS sparse decode removed.** It never beat the alternatives on the hybrid and
  could not run alongside speculation.
- **`CHAD_NO_DRAFT_SHORTLIST=1`** disables the 2-bit shortlist readout used by the
  greedy draft chain (draft-side only, never load-bearing — the chained drafts'
  full-vocab `lm_head` read was ~70% of the head-step cost).
- **Dead kernels removed.** Three custom decode paths earned their keep on a model or an
  mlx version chad no longer ships against. Rather than linger as knobs that do nothing,
  they are gone:
  - `mlx_moe_fused.py` (the fused sparse-MoE decode kernels) and the MoE branch of
    `mlx_fastpath.py`. The shipped model is dense; `install()` now declines a MoE
    checkpoint outright instead of half-applying the dense transforms to it. **Removed
    knob: `CHAD_NO_MOE_FUSED`.**
  - The S=1 `simdgroup_matrix` attention tier in `mlx_qsdpa.py`, which needed exactly 8
    query heads per KV head to fill its fixed 8×8 score tile. The shipped model is GQA-6
    and could never dispatch it. **Removed knob: `CHAD_NO_QSDPA_SGM`.** Its *wide* (S>1)
    sibling carries the retile idea forward on virtual rows and has no such constraint —
    that one is live and is what speculative verification runs on.
  - `mlx_qmm_s.py`, the fused small-batch quantized matmul, and the `QMM` indirection
    in `mlx_fastpath.py` that existed only to host it. Re-measured against the stock op
    at bf16 — the dtype the model actually runs, and the one an earlier fp16-only
    measurement got wrong — on real verify shapes with cold weights, it is 1.01× at
    S=3 (the shipped speculation width) and 0.71× at S=6: par at best, slower where it
    matters. A kernel that ships OFF because it loses to the op it replaces is dead
    code, not scaffolding. **Removed knobs: `CHAD_QMMS`, `CHAD_NO_QMMS`.**

### Context, sampling, and other changes

- **Breaking: `CHAD_CTX_RESERVE_GB` is replaced by `CHAD_CTX_SAFETY`.** The headroom
  lever is now a *fraction* of the live Metal budget the auto-sizing may spend
  (default 0.975) rather than an absolute gigabyte reserve, which is what lets the
  same setting mean the same thing on a 16 GB and a 64 GB machine. The compaction
  threshold now also subtracts the **prefill transient** — the attention scratch live
  at the same moment as the cache — which is fixed rather than per-token (measured on
  the 27B: 1.8 GB at 8k, 4.15 GB at 49k, flat thereafter). `CHAD_CTX_LIMIT` still
  forces an exact threshold.
- **New sampler knobs**: `CHAD_TOP_K` (quant-tail trim, off by default) and
  `CHAD_PRESENCE_PENALTY` (0.0 by default and worth leaving there — code is
  inherently repetitive, and measured at 1.5 the model produced 45 steps of pure
  exploration, zero edits, and visibly corrupted tool arguments). All five sampler
  settings now travel as one call, so `chad serve` honors them too — previously a
  server started with `CHAD_MIN_P` ran without it and said nothing.
- **New: `CHAD_REASONING_EFFORT`** (`xhigh` | `medium` | `low`) sets the
  template-level reasoning budget on checkpoints whose chat template accepts one.
  Unset, the argument is not passed at all, so templates without the knob are
  unaffected.
- **Tab-indented files edit correctly.** The `edit` tool's indentation-drift recovery
  mis-indented a majority of edits in tab-indented sources; over the dogfood corpus
  that path went from 33% to 100% correct.
- **The cross-session warm start is now documented, with numbers.** Nothing changed in
  the feature — chad has always checkpointed the stable system+tools KV prefix to disk
  and reloaded it on the next session in the same project — but it was described only in
  passing, as a `chad serve` footnote, and never measured. On the shipped model it is
  worth 75.6 s → 5.5 s to the first tool call and ~103 s → 31.4 s on the whole turn.
  See [Throughput & performance](docs/benchmarks.md).
- **The demo GIF records the warm path.** `docs/demo.tape` built its fixture in a fresh
  `mktemp` directory, which can only ever miss the warm-prefix checkpoint, so the demo
  was recording chad's worst case and then hiding the resulting minute-long prefill
  behind a timed cut. The tape now primes the checkpoint off camera and records the
  session every user gets after their first — short enough to show unbroken, with no
  hidden cut inside the turn.

## [1.13.0] — 2026-08-10

- **Per-step generation cap raised 8192 → 32768** (`CHAD_MAX_GEN_TOKENS`
  overrides it): 8192 fixed the old truncated-write bug but introduced its
  own loss class — on hard problems a long chain of thought can pin the cap
  while still inside `<think>`, so the step ends as discarded reasoning with
  no action and the next step re-derives from scratch. Trace measurement:
  2% of steps pinned the cap, 82% of those mid-think, and a failing trial
  was 2.75× more likely to contain one; an uncapped run of the same model
  almost never exceeds 8192 on its own (p99 ~4.7k), yet the rare long
  thought that does run completes real work when allowed to finish. The cap
  stays as a backstop against non-repetitive runaway garble — literal decode
  loops remain the repeat guard's job.
- **New lever: `verification_matrix` (default OFF).** After ~8 exploratory
  bash steps since the last landed change — *including after an edit has
  landed*, the case `investigation_gate` cannot see, since it freezes the
  moment any edit lands — the turn is pulled into a bounded verification
  phase. It receives the task's own requirement lines (reusing the same
  extractor `done_audit` runs, so the rows are the real predicates, not a
  paraphrase) and must close each by exactly one of: (a) causal evidence
  from a real run through the public path — a snapshot, a self-authored
  "PASS", "no error", or a check built from the same assumption as the code
  do not count — or (b) an explicit "unverified" note. That single rule both
  bounds the loop (a finite checklist has a terminal state) and blocks
  false-done; the honest-unverified escape is load-bearing, letting a turn
  finish a genuinely unverifiable requirement instead of thrashing on it.
  Re-arms up to 6 times per turn and never says a bare "call `done`", so
  `done_audit` still guards the actual completion. Targets the measured #1
  loss class: 69% of failing trials never called `done`, running a long tail
  of exploratory bash (median 39 commands vs 12 on passes) around the same
  edits a passing trial makes, until the step cap killed them.

## [1.12.0] — 2026-08-07

- **Sandbox hardening — the confinement must prove itself before it may be
  claimed:**
  - `yolo_seatbelt`'s startup probe now checks **enforcement in both
    directions** against a throwaway directory pair: the allowed write must
    land AND the denied write must not. The old probe only proved
    `sandbox-exec` could run a permissive profile — a profile failing open
    (OS drift, a malformed rule) would have reported confinement it did not
    have, the worst failure a safety lever can have. A denied write that
    lands now refuses the sandbox with a loud log; either failure runs
    unconfined and says so. (The probe promptly caught a real member of this
    class: an allow rule written against the `$TMPDIR` symlink spelling that
    Seatbelt, matching the kernel's real path, silently denied.)
  - The undo history (`~/.chad/checkpoints`, the `edit_checkpoint` shadow
    repo) is now **write-denied inside the sandbox**, closing the gap where
    it was reachable through the allowed `~/.chad` subpath — a sandboxed
    command could previously delete the very snapshots `/undo` restores
    from. Costs real commands nothing: only chad's own never-sandboxed
    process writes there.
- **Two new "safety" levers (default OFF):**
  - `seatbelt_protect_git` — opt-in tier on top of `yolo_seatbelt`: the
    workspace's git metadata (`.git`, and a worktree's external gitdir +
    common dir) becomes write-denied inside the otherwise-writable
    workspace, so an unreviewed command cannot `rm -rf .git` the project's
    history. The cost is real — every `.git`-writing git command (commit,
    add, checkout) EPERMs, sized statically at ≤2.65% of 91,910 real session
    commands, concentrated in 14% of sessions — which is why it is its own
    lever rather than part of the base profile.
  - `bash_env_guard` — spawned bash children get a filtered copy of the
    environment: variable names shaped like credentials (`…_TOKEN`,
    `…_SECRET`, `…_PASSWORD`, `…_API_KEY`, `…_ACCESS_KEY`, `…_PRIVATE_KEY`)
    are dropped, name-pattern only, values never read. A filesystem sandbox
    that still hands every command the operator's cloud keys is a half-closed
    boundary; commands that legitimately need a credential need the lever
    off (it defaults off, preserving inherit-all).
- **Two new "safety" levers (default OFF, like everything since 1.10.0) —
  blast-radius containment for unattended mutation:**
  - `yolo_seatbelt` — in yolo mode on macOS, each bash command's shell child
    runs under a Seatbelt profile (`sandbox-exec`) that denies file writes
    outside the workspace, temp dirs, tool caches, and `~/.chad`. Reads,
    network, and exec stay open, and the model process itself is never
    sandboxed — only the spawned shell. A detected denial gets a one-line
    explanation appended to the tool result so the model routes around the
    boundary instead of retrying into it. Environments where Seatbelt can't
    apply (CI, nested sandboxes) are probed once and run unconfined.
    Sized against 91,910 real session commands: 94.3% write only inside the
    allowlist; the denials are the system-admin writes the sandbox exists for.
  - `edit_checkpoint` — before a file-mutating tool lands, the workspace is
    committed to a shadow git repo under `~/.chad/checkpoints` (the project's
    own `.git` is never touched, written, or required to exist). New TUI
    commands: `/undo` reverts files to the last checkpoint, `/restore` lists
    checkpoints and reverts to a chosen one. Restores put snapshotted content
    back but never delete files created since — deleting is exactly the blast
    radius this lever contains. Snapshot cost measured at ~56ms per edit on a
    167-file repo (~84ms at 3,000 files).

## [1.11.0] — 2026-08-06

- **New tool: `definition(name)`** — jump from a *use* of a symbol to the one
  place it is really defined, following imports and aliases through the language
  server (pyright/TS/gopls/rust-analyzer/clangd/…), with the honestly-labeled
  tree-sitter name-match as fallback. The capability had existed since 1.0.7;
  no tool schema exposed it — the model could never call it. Asked at a use
  site, not the definition (asking at the definition answers with itself), so
  two files defining `target` resolve to the one your file actually imports.
  Verified across 13 fixture languages.
- **Three new "ambient state" levers (default OFF, like everything since
  1.10.0): the harness states facts in the results the model already reads,
  instead of steering it toward tools it demonstrably won't call.** Trace
  measurement over ~140k tool calls showed the model routes ~83% of its
  searching through `bash` and calls the symbolic tools ~never, under a prompt
  that already says not to — so these meet it on the route it chose:
  - `env_manifest` — a session-start toolchain inventory in the system prompt
    (compilers/interpreters with versions, notable absences, package managers;
    ~75 tokens, built once in <1s, prefix-resident). Answers the
    `which`/`--version`/`pip list` probe chains up front — measured at 3.3
    probes per failing trial, 37% of them failing.
  - `session_ledger` — one cumulative fact line on landed edits and done
    bounces: `[session] edited: x.py·parse() · wrote: answer.txt · last
    verifying run: 2 calls ago (pytest → exit 1)`. Facts with provenance,
    never instructions; revert-aware; elided when unchanged. Aimed at the
    largest measured failure class — confident `done`s built on a stale
    mental model of what changed and what was actually re-run.
  - `bash_read_skeleton` — the first time a source file's content comes back
    through `cat`/`sed`/`read`, the result carries a one-line symbol map
    (`[file] mod.py: alpha() 1-2 · Gamma 9-11`); a zero-hit `grep` for a
    symbol the tags cache knows gets its real definition site. Symbol
    intelligence with no new tool name to learn.

  Enable with `CHAD_ENABLE=env_manifest,session_ledger,bash_read_skeleton` (or
  `all`); each is instrumented (`fired()` telemetry) and A/B-able on its own.

## [1.10.0] — 2026-08-05

- **Every harness lever now defaults OFF: a default run is the bare model + tool
  loop.** Over a year of iteration, 48 named behaviors accumulated around the agent
  loop — steers, gates, reverts, nudges — each justified by a real failure in some
  trace. Clean-slate measurement (bare arms vs the full stack, same tasks, same
  serving) showed the bare loop matches the full stack's benchmark score within
  noise at substantially lower token spend, so the burden of proof has flipped: a
  lever ships ON only behind a measured, positive, pre-registered contrast. Nothing
  is deleted — `CHAD_ENABLE=<a,b>` turns individual levers on, `CHAD_ENABLE=all`
  restores the full pre-1.10 stack, and `CHAD_ENABLE=all CHAD_DISABLE=<x>` is the
  leave-one-out ablation idiom. `chad levers` still prints the registry and what's
  active; unknown names in either variable are still a startup error.
- **The TB2 benchmark agent gained `chad_enable=`** (forwarded as `CHAD_ENABLE`
  in-container) so lever A/B arms stay one flag; its default arm is now the bare
  harness.
- The never-populated `Profile.disables` field is gone; profiles are purely additive
  prompt blocks, themselves behind the (now default-off) `profile_prompt` lever.

## [1.0.9] — 2026-07-30

- **Decode is 5–7% faster on the 35B.** The MoE block — the largest remaining
  inefficiency in the per-token decode step — now runs as two custom Metal kernels
  (JIT-compiled at load, no wheel rebuild): one dispatch computes gate|up for all 8
  routed experts plus the shared expert with the SiLU epilogue in-register, the other
  folds every down-projection, the routing weights, the shared-expert gate, and the
  residual add into a single pass. Routing itself is bit-identical to stock; measured
  on the shipped checkpoint: 70.7 → 74.3 tok/s at 8k context, decode-path perplexity
  a hair *better* (the fused combine accumulates in fp32), prefill unchanged
  (756 tok/s @7k vs 723 baseline), peak memory unchanged. Engages only on the exact
  35B geometry, falls back to the stock graph anywhere else; opt out with
  `CHAD_NO_MOE_FUSED=1`.

## [1.0.8] — 2026-07-29

- **`/speech` now names an install command that can actually work.** The voice-mode
  install hint was hardcoded to `uv sync --extra speech` — true only in a git clone, so
  every published install (`uvx chad-code`, `uv tool install chad-code`) was sent to a
  command with no project to sync. An extra rides on the *install spec*, not on a
  separate step, and `speech.install_hint()` now derives the right form from how chad was
  actually installed: `uv tool install --force 'chad-code[speech]'`, `uvx --from
  'chad-code[speech]' chad`, `uv sync --extra speech` in a clone, or `uv pip install
  'chad-code[speech]'` in a plain venv.
- **Extras documented in one place.** The README install section now states the
  install-spec rule with all three commands side by side, so the PyPI project page
  carries it too. Also drops a stale line claiming LSP-precise find-refs/rename need an
  `lsp` extra — that extra is gone and precision ships in the base install as of 1.0.7.

## [1.0.7] — 2026-07-29

- **Precision code intelligence now ships in the base install.** `find_refs` /
  `rename_symbol` precision used to live behind an optional `lsp` extra
  (`serena-agent`, 33 transitive dependencies) that `uvx chad-code` never installed —
  so every published install ran name-match only. chad now drives language servers
  through its own ~350-line LSP client (`lspclient.py`) plus a declarative registry
  (`lspservers.py`, the nvim-lspconfig shape): pyright via uvx (guaranteed present on
  the uvx install path), the TypeScript 7 native LSP via npx, gopls / rust-analyzer /
  clangd / solargraph / intelephense from PATH. Zero new Python dependencies; the
  `lsp` extra and serena-agent are gone. Servers that would answer confidently-wrong
  empties without project context (clangd without `compile_commands.json`, TS without
  a `tsconfig`) are refused, keeping the honestly-labeled name-match fallback.
- **Qualified symbol paths in every language.** `view_symbol("Engine/process")`,
  `replace_symbol("Engine.process", …)` etc. now resolve the method — not a same-named
  free function — in Python, TS/JS, Go, Rust, Java, Ruby, C#, PHP, Kotlin and C++
  alike, via scope chains derived from tree-sitter spans (plus receiver/impl context
  for Go/Rust/C++, whose methods don't nest lexically). This was Python-only via
  jedi before; jedi is deleted and every language takes one code path (verified
  against jedi on this repo: 230/230 span-identical). A qualified path whose
  container doesn't match is now a miss instead of silently editing the bare name.
- **Broken grammars fixed for 4 of 12 fixture languages — and shell added outright.**
  First polyglot measurement found the language pack ships NO tags query for
  TypeScript/TSX, a PHP query that doesn't compile against the current grammar, C
  with no reference captures, and C++ missing out-of-class method definitions. chad
  now carries its own tags queries for those (`repomap._TAGS_OVERRIDE`), plus a new
  bash query the pack never had: shell functions and top-level variables now appear
  in `repo_map`/`overview`, `find_refs` finds invocations across scripts, and
  `replace_symbol` edits a shell function by name — shell is the one surface nearly
  every real repo (and 89 of 91 terminal-bench-2.1 tasks) contains. All pinned by a
  13-language coverage test.
- **Post-edit typecheck in the edit result.** When a language server is warm, every
  edit/symbol-edit result appends the server's type errors (errors only, capped,
  ~50 tokens) — a type error surfaces immediately instead of one failed test run
  later. Edits never pay a cold server start (lever: `post_edit_diagnostics`).
- **New `hover` tool.** The resolved type signature + docs of one symbol via the
  language server — one line instead of opening the defining file.
- **mcp 2.x, single code path.** The `mcp<2` cap (below) was serena-agent collateral;
  with serena gone chad runs natively on mcp 2.0 (protocol 2026-07-28). The two
  renames that fail silently through getattr defaults (`isError`→`is_error`,
  `readOnlyHint`→`read_only_hint`) are each pinned by a test that fails on the 1.x
  spelling.
- **Fixes a broken install.** `uvx chad-code` (and any other fresh resolve) crashed on
  startup with `ImportError: cannot import name 'streamablehttp_client'`. The `mcp`
  dependency was unbounded, so a clean resolve picked up the SDK's 2.0 major, which
  removed the HTTP transport chad uses, moved to `httpx2`, and split the wire types into
  a separate distribution. (The cap has since become the deliberate 2.x port above.)
- **A broken MCP SDK can no longer take the agent down.** Every other MCP failure already
  degraded — a server that's missing, misconfigured, or crashing costs you its tools and
  nothing else — but the SDK import itself ran unguarded at `chad.mcp` load, which happens
  on every Agent. It's guarded now: an incompatible SDK means no MCP tools plus one
  warning line in `/mcp`, and the rest of chad runs. Relatedly, `/mcp` no longer reports
  "no MCP servers configured" when servers *are* configured but all failed — it shows the
  warnings that explain why.
- **CI now catches this class of break.** The fresh-resolve job and the weekly install
  canary only ran `chad --help`, which exits inside argparse and never imports the modules
  that load on the first turn. Both now import every `chad` submodule against a clean
  resolve and fail if the MCP SDK guard fires.

## [1.0.6] — 2026-07-28

- **Voice mode, all local.** `/speech` in the TUI turns on push-to-talk: ctrl-t opens the
  mic (a red `● rec` pin in the status line — an open mic you can't see is a bug), ctrl-t
  again transcribes into the input box. The transcript is *reviewed* text, not a sent
  message — you read what the model heard and Enter submits, same supervision contract as
  every edit. Replies are read aloud (code blocks dropped, cut at a sentence boundary;
  ctrl-c hushes). STT is **Parakeet TDT v3** on MLX — the engine
  [Hex](https://github.com/kitlangton/Hex) ships as its default, faster than Whisper per
  clip, more accurate on dictation, and without Whisper's repetition-loop hallucination
  mode — via a copy vendored into `chad.parakeet` (Apache-2.0, from
  [parakeet-mlx](https://github.com/senstella/parakeet-mlx)); TTS is macOS `say`. Nothing
  leaves the machine and the audio never touches disk. Vendoring is deliberate: the PyPI
  packages drag torch (mlx-whisper, ~2 GB) or librosa's numba/scipy tree (parakeet-mlx)
  for code paths MLX-format weights never run, so `uv sync --extra speech` adds exactly
  one package — the mic library (sounddevice). The tokenizer vocabulary rides inside the
  model repo, so no data blobs ship in chad. Dictation hygiene is built in:
  trailing/leading silence is trimmed before the model sees it, and a repetition-collapse
  pass keeps any decode loop out of the input box. The weights are **quantized to 8-bit
  at load** (no separate quant repo): A/B'd against bf16 on a 24-clip multi-voice
  battery — transcripts identical up to case/spacing, twice *more* correct — for 786 MB
  resident instead of 1296 and ~40% faster decode. Tune with `CHAD_STT_MODEL` (default
  `parakeet-tdt-0.6b-v3`, multilingual; ~2.5 GB fetched on first use), `CHAD_STT_QUANT`
  (`8` default · `4` halves memory again, clean on the battery but unproven on noisy
  mics · `none` for bf16), `CHAD_VOICE`, `CHAD_SPEECH_RATE`.
- **Dictation length is linear, not quadratic.** The encoder ships as `rel_pos` with
  `att_context_size [-1,-1]` — full global attention, so a take's cost grows with the
  *square* of its own length, on the same GPU the coding model is resident on. chad now
  switches it to the banded local kernel at load: the score matrix goes from `T×T` to
  `(T, 2w+1)`, and w=256 encoder frames is ±20.5s of audio at 12.5 frames/s, so a
  dictation take still sees its whole self. The transcript is **byte-identical** to full
  attention on a real clip (and 8-bit local is byte-identical to bf16 local). A 359s take
  now decodes in 3.9s with peak memory flattening (58s→2.0 GB, 175s→3.1, 359s→3.3) where
  full attention keeps climbing. The switch happens *before* quantization, because
  `set_attention_model` rebuilds the attention modules and `load_weights` rejects a
  quantized layer's scales/biases. A take is also capped at 600s as a backstop: recording
  doesn't stop and the status line says so, the take just stops growing — a mic left open
  is truncated, never lost.
- **`/speech` off gives the memory back, and enable tells you what's broken.** Turning
  voice mode off now unloads the STT weights (747 MB → 0 measured), not just the mic —
  it won't fire mid-transcription, where the worker still holds the model. Enable-time
  validation covers the two failures that were previously invisible until after you'd
  spoken: a misspelled `CHAD_VOICE` (`say` exits nonzero, so nothing ever raised — you
  just got silence forever; now it names the typo and suggests the nearest real voice)
  and a bad `CHAD_STT_QUANT` (which only raised from inside the decode). Both warn rather
  than block — a machine without `say` can still dictate. And a transcript that finishes
  after you've turned `/speech` off is discarded with a note instead of typing itself
  into an input box in a mode you already left.
- **Voice mode capture + correction, borrowed from the best.** Three ideas adapted from
  [Hex](https://github.com/kitlangton/Hex) (MIT), the open-source macOS dictation app:
  while `/speech` is on the mic stays warm feeding a ~1s ring buffer, and each take is
  seeded with 0.45s of *pre-roll* from before the keypress — opening a CoreAudio device
  takes 100–300ms, so a cold-opened mic clips your first syllable ("ix the failing
  test"). The status line says `mic open` the whole time, because an open mic you can't
  see is a bug; `/speech` off releases it. **esc during a take discards it** instead of
  transcribing — a mistaken recording shouldn't cost a decode plus an input-box cleanup.
  And a personal **word table** (`~/.chad/speech_words.json`, or `CHAD_SPEECH_WORDS`) of
  `{"heard": "meant"}` fixes what no ASR model can — your identifiers: `{"pie test":
  "pytest", "you vee": "uv"}`. Case-insensitive, whole-word, longest match first,
  loaded fresh each take; `/speech` reports the count (or the parse error) at enable.
- **numpy 2 allowed.** The `<2` cap dated to 1.0.0 and encoded no known break — it also
  blocked modern ASR dependencies that floor at numpy 2. Now `>=1.26,<3`, verified on
  numpy 2.4.6: full unit gate, live on-device transcription, and `chad prove` 4/4
  (Ornith 9B smoke model) at normal throughput.
- **You can see what you're approving.** The confirm prompt used to render into the
  one-row status line with newlines flattened to `⏎` and a 160-character clip, so a
  multi-line bash command was unreadable — approving it was approving blind. Approvals
  now open a panel above the input that shows the command on its own lines (bounded, so a
  giant heredoc can't swallow the screen), names in plain words what is about to happen
  ("chad wants to run a terminal command"), and puts the destructive-command warning next
  to the command instead of in scrollback.
- **New demo GIF.** The old one predated the banner entirely (no logo, a dead model id,
  and a fixture that no longer matched the checked-in tape). The new one is a 35B session
  end to end in 51 seconds: banner, type-ahead while the weights load, the model finding
  the cent that floor division loses, fixing it, verifying itself, and your own `pytest`
  confirming it stuck.
- **`auto-accept edits` now means what it says.** It auto-approved *everything*, bash
  included. It now auto-approves file edits only — terminal commands and mutating MCP
  tools still ask, which is the mode most people actually wanted. The old
  approve-everything behavior is unchanged but now has its own name, `yolo`, still where
  `--yolo` starts you and now reachable with shift-tab: `normal → auto-accept edits →
  yolo → plan`. A sub-agent's `tools: "all"` request is honored only under `yolo`, since
  `all` includes the shell.

## [1.0.5] — 2026-07-26

The release that makes a 24 GB Mac a first-class target: it runs the 35B now, a Linux
container can drive the real local MLX model through `chad serve` instead of a different
quantization on a remote box, and the CLI surface is smaller and honest — `serve`, `prove`
and `levers` are real subcommands, and five harness-only governor flags became `CHAD_*`
env vars. Plus completion and reasoning-budget accuracy, long-running commands that
survive their timeout, a destructive-command guard that stops crying wolf, and
long-context decode throughput that ships to a `uvx` install — measured end-to-end on
stock PyPI mlx, no custom wheel anywhere. Each behavior change is individually reversible
via `CHAD_DISABLE`.

- **24 GB Macs now get the 35B.** The 35B's floor was 32 GB because its working set
  SIGKILLed a 24 GB machine mid-turn. The fused attention kernel and the
  8-bit-from-the-start KV cache it enables cut the per-token cache cost enough to give the
  headroom back, and the compaction trigger sizes itself from the live Metal budget rather
  than a fixed constant, so a tight box narrows its window instead of dying. 16/18 GB
  still get the 9B, and `--model 9b` puts a 24 GB box back on the small model.
- **`--model` replaces reaching for an env var to change model.** Takes `35b`, `9b`,
  `auto`, or any Hugging Face repo id / local model dir; works on `chad` and `chad serve`.
  `CHAD_MODEL` still works and the flag outranks it, so a globally exported var can't pin
  every run, and `--model auto` forces the RAM-aware pick even when the env var is set.
  Forcing `35b` on a box whose RAM is too small (or unreadable) is honored but warns
  first — chad advises, you decide.
- **`serve`, `prove` and `levers` are real subcommands** with their own `--help`. They
  were dispatched on the literal task string, so `chad --help` never mentioned them while
  still listing `--host`/`--port` as "`chad serve` only" — flags that parsed fine and were
  then silently ignored on every other invocation. `--host`/`--port` now belong to
  `chad serve` and are rejected elsewhere. `chad --levers` still works as an alias.
- **`chad --help` lists 12 flags instead of 19.** The unattended-run governor knobs
  (`--think-ceiling`, `--turn-budget-tokens`, `--turn-budget-s`, `--auto-continue`,
  `--review-pass`) are set by benchmark harnesses, never by hand, and every one already
  had a `CHAD_*` twin, so the flags are **removed** rather than hidden — passing one is now
  an error instead of a silent no-op. `CHAD_AUTO_CONTINUE` fills in the one twin that was
  missing, and the bundled TB2 adapter (`benchmarks/tb2/harbor_chad_tb2.py`) sets all three
  it used through the `CHAD_*` env dict it was already building. `--think-budget` stays on
  the CLI: unlike the rest it is a capability trade a person reaches for interactively.
- **A dead remote backend explains itself.** An unreachable `--base-url` exited through a
  22-line traceback ending in `chad.base_engine.BackendError`, which reads as a chad crash
  rather than "the server you pointed me at isn't up". Now problem/cause/fix, like the
  model-load and download failures.
- **Removed the draft-model code path.** chad has been single-model since before 1.0;
  `Engine(draft_id=...)` was never set by any caller, but its branches still shaped cache
  construction, KV-quant resolution, warm-prefix eligibility and prefill. Prompt-lookup
  decoding is unaffected — it shares the word "draft" and none of the machinery.
- **`chad serve` — drive the local MLX model from a Linux container.** chad's remote arm
  (`--backend llama`) exists because MLX can't run in a benchmark container, which in
  practice meant measuring a *different quantization* of the model on a remote GPU box.
  `chad serve` exposes the in-process MLX engine — real prefix cache and all — over the
  same llama.cpp `/completion` protocol that arm already speaks, so the container drives
  the model people actually run, unchanged and with exact server-side timings.
  `--host`/`--port` (`CHAD_SERVE_HOST`/`CHAD_SERVE_PORT`), loopback and unauthenticated by
  default; `CHAD_SERVE_API_KEY` guards a wider bind. One KV cache means one agent at a
  time; concurrent clients queue.
- **Cache quarantine and warm-prefix come back over the wire.** Against a stock llama.cpp
  server these are unavoidable no-ops (its cache is opaque and its disk isn't ours). A
  `chad serve` server advertises both in `/props` and the client feature-detects, so a
  sub-agent no longer evicts the main transcript's prefix and the stable system+tools
  prefix warm-starts from disk. Nothing changes against a real llama-server, and a failed
  call degrades to the old no-op — these are latency, never correctness.
- **A client that stops reading can no longer take the server down.** Streaming wrote
  each token to the socket from the one thread allowed to touch MLX, with no timeout on
  it — so a client that stopped draining without disconnecting (a suspended container, a
  slept laptop) blocked that thread indefinitely and no other request could be served,
  while `/health` still answered `ok`. A blocked write never raises, so the hung-up-client
  cancel couldn't fire either. Generation now hands SSE lines to the request's own thread
  through a bounded queue: the engine runs ahead of a slow reader instead of being paced
  by it, and a peer that stops reading is written off the same way a disconnected one is.
- **A request no longer silently zeroes the server's sampler config.** `min_p`, `top_p`
  and `temperature` absent from a `/completion` body were defaulted to 0.0 and applied
  anyway, wiping a server started with `CHAD_MIN_P`/`CHAD_TOP_P` on essentially every
  request (the client only sends those knobs when they're armed) and forcing greedy
  decoding. Absent now means "leave it alone"; an explicit `0.0` still disarms a knob for
  that request only.
- **Generated token ids on the wire are the engine's own.** They were recovered by
  slicing the engine's cache past the prompt, which only holds on the main decode path —
  the prompt-lookup path came up one token short every turn, and after a Metal OOM the
  client was told a prefix was resident when the cache had been dropped. The engine now
  reports what it generated, and says so when it reset the cache, so the client's mirror
  and its prefill estimates stay honest on every path.
- **Long turns stopped paying quadratic CPU to watch for stop markers.** The remote
  backend re-scanned the entire generated text after every token; at 16k tokens that was
  846ms of pure scanning on the thread draining the stream. Now incremental: 4.5ms.
- **A long cold prefill no longer kills the turn it is working on.** The stream stayed
  byte-silent from the response headers until the first token, and HTTP clients measure a
  stall in bytes, not in your progress — chad's own passes its timeout to `urlopen`, which
  applies it per read. A big prefill therefore timed out just before the first token, and
  because a socket timeout looks transient the agent re-issued the step and paid for the
  same prefill again. The stream now sends an SSE comment during silence; stock clients
  discard it, so nothing on the wire changes for llama.cpp.
- **A prompt that cannot fit is a 400, not a Metal OOM.** Nothing compared the prompt
  against the context window before prefilling it, and `n_predict` was never clamped to
  the room left beside the prompt — so a client that mis-budgeted (a stale pinned context,
  a plain `curl`) walked the engine into an allocation it might not survive, taking every
  other client with it. `/completion` and `/warm` now refuse an oversized prompt up front
  and clamp the generation budget to what the window can hold. The admission wall is
  **live**, not the number pinned at load: the Metal budget is blind to other processes,
  so a docker stack or a second model started afterwards takes physical pages the KV cache
  needs 1:1, and a prompt that fit an hour ago no longer does. `/props` keeps advertising
  the stable window clients budget against; `/health` reports `safe_ctx` and
  `ctx_pressure` so a tightened wall is visible before a request bounces off it.
- **`chad serve` honors `CHAD_TEMP` / `CHAD_MIN_P` / `CHAD_TOP_P`.** It built its own
  engine and read none of them, so a server started with a sampler config sampled as if
  it had none — the one drift the server exists to eliminate. The three now travel as a
  single shared call the CLI and the server both make, so they cannot drift apart again.
- **Long-context decode is faster on both models.** The fused attention kernel's
  split factor now widens past 16k context instead of staying flat: measured
  end-to-end (one process, interleaved arms, adopted only where the spread cleared
  1.000), the 35B decodes **+1.5% at 32k** and +0.6% at 16k, the 9B **+0.5% at
  32k**, both unchanged at 8k where flat was already optimal. It is the same math
  on a different grid — the split only regroups the partials pass 2 combines — and
  the widened tier gets its own kernel-vs-reference test plus a retile-agreement
  case at that size.
- **`CHAD_FUSED_LAYER` is removed.** The opt-in fused GDN+MoE compiled region was
  measured on real weights against the two-call path it replaced and moved decode
  1.001x-1.002x — indistinguishable from noise — so the code is gone rather than
  shipped as a dead knob. The measurement lives in the source so the idea isn't
  re-derived. Setting the var now does nothing.
- **`CHAD_NO_MLX_TUNING` is removed, along with the tuning it disabled.** The `MLX_*`
  runtime knobs (`MLX_METAL_FAST_SYNCH`, `MLX_MAX_OPS_PER_BUFFER`,
  `MLX_MAX_MB_PER_BUFFER`) were swept on the 35B and every arm lost 1-4% at both 8k
  and 32k, so chad now sets none of them and the module that applied them is gone
  rather than shipped as an empty table with a flag to switch it off. mlx reads
  these vars directly if you want to experiment.
- **A `done` with nothing landed gets the task's own requirements back, once.**
  Previously, ending a turn with no landed-and-verified change stopped the turn
  immediately and banked a progress note — so the done-audit, which quotes the task
  statement's requirement lines and reports what actually exists on disk, never ran
  on exactly the turns that needed it most. That ending now hands off to a single
  audit bounce first, in context with the work the turn already did; a second empty
  ending stops as before. (`audit_churn_handoff`)
- **Reasoning truncated at the token cap now counts against the reasoning budget.**
  A generation that hit the raw output cap while still inside `<think>` credited
  zero reasoning tokens, so the per-turn budget was blind to the largest thinks
  there are — one full cap each — and its throttle never engaged on the turns that
  overspent the most. Those tokens are now counted. No threshold or clamp changed;
  only what the budget can see. (`capped_think_credit`)
- **A `bash` command that outruns its timeout keeps running instead of being killed.**
  It moves to the background rather than having its process group killed: output keeps
  streaming to a file named in the result, ending with an `[exit <code> at HH:MM:SS]`
  footer when it finishes, so a long install, build, or download is not thrown away
  and re-run from zero. Bounded by design — at most two at a time, an absolute
  lifetime each, and all of them terminated when chad exits or is signalled. A user
  interrupt still kills outright. (`bash_auto_background`)
- **The destructive-command guard screens catastrophes, not cleanups.** The
  recursive-delete check matched *any* absolute path, and in a headless container run
  every real path is absolute — so ordinary scoped deletes like
  `rm -rf /tmp/test-deploy` were refused, over and over, while the result claimed
  `[denied by user]` when no user was there. The screen now fires on targets whose
  loss is actually catastrophic — the filesystem root, top-level directories, home
  trees at any depth, the whole cwd or its parent, a glob that empties a top-level
  directory — and passes deeper scoped paths. It also screens every target of every
  `rm` in a compound command, where the old single-target check saw only the first.
  When it does block with nobody to ask, the result now says the guard (not a person)
  refused, and how to narrow the delete. mkfs / dd-to-device / fork-bomb / `curl|sh`
  screens are unchanged. (`scoped_destructive_guard`)
- **Unattended runs no longer strand a third of their time budget.** Extra fresh
  attempts after a stalled turn were only granted while more than half the wall
  budget remained, so a task whose turns kept hitting the step cap ended for good
  with 25-50% of its time unspent. Extras now keep flowing while more than a quarter
  remains; the absolute relaunch ceiling and the minimum-remaining-time floor for
  any relaunch are unchanged. (`late_continue_replenish`)
- **The system prompt steers final verification at the task's own stated check.** A
  model that was bounced for unverified work would usually re-verify — with a weaker
  check than the task named (file-exists for a content requirement, a hand-rolled
  probe instead of the stated command) — then finish wrong with most of its time
  unused. One prompt block now says: run the named check end-to-end, read its output,
  and treat existence as proof of nothing, with a WRONG/CORRECT pair.
  (`steer_verify_specific`)

Also fixed:

- **`chad --version` reported the wrong version.** `__version__` had drifted from the
  packaged version, so 1.0.3 and 1.0.4 both identified themselves as 1.0.2.

## [1.0.4] — 2026-07-22

Tool-result economics: three additions, each individually reversible via `CHAD_DISABLE`.

- **Edits survive typographic-punctuation drift.** A fourth edit-match rung: when
  exact, escape-normalized, and whitespace-flexible matching all miss, the edit
  retries with curly quotes, en/em dashes, ellipsis, and non-breaking spaces folded
  to ASCII on both sides — the drift when the model re-types prose or docstrings it
  saw rendered (in either direction). A unique match is still required, and the
  result discloses the recovery.
- **Duplicate read-only output is elided.** When a `read`/`grep`/`glob`/symbol-tool
  result comes back byte-identical to a result still in the transcript, chad appends
  a short pointer to the earlier copy instead of re-sending the full body — on a
  non-trimmable prefix cache every duplicate body is prefill paid again on every
  later step. Byte-equality against the live transcript is the safety proof: a
  changed file, different arguments, or a compaction rewrite all break equality, so
  content is only elided while a verbatim copy is provably still in context.
- **Sub-agent reports with no evidence are flagged.** A `task` sub-agent that
  returns a confident, non-empty report having dispatched zero tools answered from
  model memory, not this repository; its folded result now carries an explicit
  "verify with grep/read before relying on it" warning. Warn-not-reject on purpose:
  a re-spawn doubles the local GPU cost, and a wrongly-rejected report breaks the
  turn while a wrongly-accepted one merely restores the old behavior.

Plus a privacy default flip:

- **Local diagnostic traces are now opt-in (privacy-first default: off).** The
  readable `~/.chad/session.log` trace — user query, tool-call args (bash commands,
  write/edit content), and result previews — and the persistent input history at
  `~/.chad/history` are no longer written by default. chad is a local, single-user
  agent so nothing ever left the machine, but the log still landed plaintext previews
  under `~/.chad`, so it's now enabled only when you opt in with **`CHAD_SESSION_LOG=1`**.
  `CHAD_NO_SESSION_LOG=1` still works as a hard kill switch and wins over the opt-in.

## [1.0.3] — 2026-07-21

Guardrail-interaction fixes: three cases where the model was doing fine but
overlapping guardrails ended the turn early or accepted a non-answer.

- **A malformed tool call is never accepted as a final answer.** When the model
  slips into a wrong tool-call dialect, garbles now get their own re-nudge budget
  (separate from the token-cap-truncation counter), a canonical call exemplar once
  it's stuck, a scrub of the previous garbled message so the model stops
  few-shotting its own broken dialect, and — if the budget runs out — a clean
  hard stop with a banked progress note instead of shipping the garble as the
  answer. On by default; reversible via `CHAD_DISABLE`.
- **Done-audit re-bounce for still-missing deliverables.** If the audit flags a
  task-named output path as absent and it's *still* absent when the model tries to
  finish (with real time left on the budget), chad bounces once more naming only
  the missing path, then accepts unconditionally. Stops a required file from going
  unwritten while the turn ends "successfully."
- **Investigation gate exempts ops commands.** A bash step that isn't provably
  read-only (`git merge`, `apt-get install`, redirects, `make`, …) now counts as
  action, not investigation, so the "you've only been looking, make an edit" gate
  no longer fires in the middle of a legitimate ops workflow.

## [1.0.2] — 2026-07-20

Reliability tuning for long, budgeted tasks, plus one interactive papercut fix.

- **Syntax gate no longer flags prose & data files.** Plain-text deliverables — a
  `.txt` answer file, `requirements.txt`, markdown, CSV/TSV — were being run through a
  tree-sitter grammar (the language pack maps `.txt` → VIMDOC) and warned on, sometimes
  reverted, at exactly the write that produced the deliverable. The gate now polices
  code languages only; prose/data formats are skipped uniformly across warn, edit-revert,
  and write-reject.
- **`run`-style tasks no longer bail with prose.** System-state asks ("start the
  service", "boot the image", "install X") are completable with zero file edits, and
  used to fall through to the weakest completion path — a give-up in prose could ship
  nothing (qemu-startup did, with most of the wall unspent). They now get their own
  intent class that arms the anti-bail nudges, while still completing cleanly with zero
  edits (the no-empty-diff "done" gate only ever applied to edit tasks).
- **Reasoning budget throttles instead of muting.** Once a turn's cumulative
  reasoning-token budget is spent, chad now forces one no-think action step per ~3k
  further reasoning tokens (a duty cycle) rather than muting thinking for the rest of the
  turn — thinking is restored as soon as the model stops over-spending, avoiding the
  garbled tool-call tails a blanket mute produced.
- **Wall-aware auto-continue.** When most of a task's wall budget is still unspent, an
  exhausted turn is granted a fresh relaunch (bounded), instead of giving up after a
  fixed two attempts with the clock barely touched.
- **Rejected "done" no longer poisons the relaunch.** When a completion claim is
  rejected for landing no verified change, the carried-forward progress note now leads
  with a warning that the claim was rejected and drops any hypothesis that itself asserts
  completion — so a relaunch stops inheriting "already complete and verified" as fact and
  re-confirming it.

## [1.0.1] — 2026-07-19

First complete release: a single-user coding agent that runs **entirely locally on
Apple Silicon** — Claude-Code-style tool use, plan mode, and a full-screen TUI, driven
by an Ornith model picked for your RAM (35B MoE at ≥32 GB, 9B dense below). One
command: `uvx chad-code`.

### The agent

- **Full-screen TUI**: shift-tab permission modes (normal / auto-accept / plan), a
  type-ahead message queue, mid-turn interrupt, mid-run steering (keep typing while it
  works — injected between steps, no re-prefill), and a live status line (↑prefilled /
  ↓generated counts, advancing % on a full re-prefill).
- **Sessions**: resume (`-c`) and list-and-fork (`--resume`) — copy-on-resume, so a
  resume never overwrites the session it branched from.
- **Extensions**: Agent Skills ([agentskills.io](https://agentskills.io)) and MCP
  servers (stdio & Streamable HTTP; OAuth behind `CHAD_MCP_OAUTH=1`), with a
  project-trust gate — a project's `./.mcp.json` servers don't start until `/mcp trust`.
- **Sub-agents** (`task` tool): open-ended exploration runs in a quarantined context
  with a read-only toolset, so the main transcript stays small and the warm cache
  survives the spelunking.

### The engine

- **Persistent prefix KV cache** — the core design: the transcript is kept a strict
  token-prefix of the live cache, so each step prefills only what was appended. The
  stable system+tools prefix checkpoints to disk (a cold start loads it instead of
  re-prefilling); a divergence inside the last turn (a truncated generation
  re-rendered, a retried prompt) takes a **bounded rewind** — snapshot-restore the
  recurrent state, native-trim the attention KV, re-feed one turn — instead of a
  full-transcript rebuild. Bit-exactness pinned by test on the hybrid SSM/attention
  cache.
- **35B-on-24GB stability**: Metal allocator clamps installed at load; a Metal OOM
  inside a prefill chunk rolls the cache back exactly, halves the chunk and retries
  (mlx ≥ 0.32 required for catchable OOMs); adaptive per-chunk prefill sizing
  (MoE 2048 / dense 512, decaying under pressure — worth ~+14% prefill on the 35B);
  the RAM-aware compaction trigger respects host-physical pressure and re-checks per
  turn; compaction that cannot get under the window latches instead of destroying the
  warm cache every step (the old behavior cost a measured 26–28 s per step).
- **Correct sampling off the main thread**: MLX's implicit-key RNG never advances on
  non-main threads, so temp>0 decoding there was quasi-greedy with one frozen noise
  vector (identical prompts → byte-identical responses, markedly loop-prone). An
  explicit split-per-draw key chain (`_KeyedSampler`, fresh entropy per generation)
  fixes it on every thread; greedy decoding is unchanged.
- **Remote arm** (`--backend llama`): the same harness against a llama.cpp server's
  raw `/completion` — token-id prompts (no double chat-template), real cache
  telemetry, `<think>` passed back verbatim. This is the Terminal-Bench arm; see
  `benchmarks/tb2/`.

### Reliability on a small model

The bet: for a C+ student, the harness is worth more than a model upgrade. Every
lever below exists because a logged failure demanded it.

- **Tool calls parsed in four dialects** (JSON, fenced, XML function-call, and the
  hybrid garble a quantized model emits under sampling), then schema-coerced with a
  typed self-repair loop — the model is shown exactly which fields are wrong instead
  of regenerating blindly.
- **Edits go through a forgiveness cascade** (exact → escape-normalized →
  whitespace-flexible with the file's own indentation), and line-addressed
  `replace_lines`/`insert_lines` fit indentation for you. An edit that would newly
  break the file's parse is **reverted, not landed** — with visible-whitespace echoes
  so the model copies real indentation instead of re-guessing.
- **Guard rails with teeth**: loop/repetition/thrash detection with bounded nudges,
  verify-before-done (a display command exiting 0 is not verification), a done-audit
  that quotes the task's own requirements back, a no-empty-diff gate on action tasks,
  and a runaway-turn governor that banks a deterministic progress note and relaunches
  fresh instead of grinding a poisoned context.
- **Progress-aware step cap**: a turn that keeps landing *and verifying* changes earns
  extensions (ceiling 4×); one that stalls banks its progress note so
  `continue`/`--auto-continue` resumes instead of the task dying at the prompt.
- **Bounded prefill everywhere**: every tool result is capped before it hits the
  transcript, and a multi-call step shares one output budget — a whole-file read or
  wide grep can no longer stall the next turn 20–30 s.
- **Degenerate-repetition stop**: decode locking into repeating one short string
  (minutes of dead generation per occurrence) is detected within a few hundred
  tokens; the step is cut off and the model nudged out of the loop.

### The models

- **35B `UD-Q2_K_XL` quant improved at the same speed**: 3-bit expert
  down-projections (3.09 bits/weight, ~13.4 GB resident); held-out code perplexity
  7.25 → 6.86 with prefill/decode unchanged — decode reads only the 8 active experts,
  so the extra bits are effectively free.
- **A 6-bit build for big-memory Macs**:
  [`Ornith-1.0-35B-Q6-MLX`](https://huggingface.co/nathansutton/Ornith-1.0-35B-Q6-MLX)
  (~28.5 GB; needs ≥48 GB) — run it with `CHAD_MODEL=<repo>`.

### Safety & first run

- Catastrophic-bash seatbelt (screened even in `--yolo`; blocked headless),
  deterministic post-edit syntax gate, minimal environment for stdio MCP subprocesses,
  sessions stored `0600`, secret-redacted diagnostic log.
- **`chad prove`**: a two-minute offline smoke test — four tiny fix-it tasks through
  the real agent loop, mechanically verified, network-blocked, timed. A proof of
  life, not a benchmark.
- Disk preflight before the model download (refuses with the shortfall instead of
  dying at 70%); unknown-RAM machines get the safe 9B; guard stops say what to do
  next and point at the troubleshooting symptom map.

### Benchmarks & docs

- **Terminal-Bench reproduction kit** (`benchmarks/tb2/`): the exact Harbor adapter,
  runners (including the leaderboard-compliant `run_tb21_submit.sh`), and the ATIF
  trajectory validator behind the README chart — serve an Ornith GGUF with llama.cpp
  yourself and check the number rather than trust it.
- Design & internals, configuration reference, throughput docs (reproduce with
  `chad-bench`), troubleshooting symptom map, release checklist (`RELEASING.md`),
  and a demo GIF regenerated from a checked-in vhs tape so it can't rot silently.
