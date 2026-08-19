# chad demo tape — regenerates docs/demo.gif with vhs (https://github.com/charmbracelet/vhs)
#
# Run manually on the dev Mac per release (see RELEASING.md). CI cannot run this:
# the demo drives a real model on Apple Silicon. The tape is checked in so the
# demo can't silently rot relative to the actual UI — if the TUI changes, this
# file is the diff that says the GIF is stale.
#
# Two moments are the demo; keep both:
#   * the banner — the session states what it is: version, model, context, cwd. The
#     model name is the whole pitch (a 27B on this laptop, not an API).
#   * the unbroken run — read, reason, edit, verify, with the work visible as it lands.
#
# The session runs `--yolo` deliberately: in the default (normal) mode every edit and
# every shell command opens an approval gate, and a recording can only answer those on a
# timer, which cost ~35s of the runtime sitting on panels waiting to be dismissed. The
# approval UX is real and worth showing — but a GIF is the wrong place to show waiting.
#
# WHY THE PRIMING RUN BELOW EXISTS — READ BEFORE "SIMPLIFYING" IT AWAY:
# chad checkpoints the stable system+tools KV prefix to disk and reloads it on the next
# session in the same project (engine.warm_prefix). The checkpoint key covers the whole
# system prompt, cwd and repo map included, so a FRESH mktemp dir can only ever miss.
# Measured on the shipped 27B, this exact fixture and ask:
#
#              cold (fresh dir, miss)   warm (checkpoint hit)
#   first tool         75.6s                    5.5s
#   whole turn        ~103s                    31.4s
#
# Recording straight into a fresh dir therefore records chad's WORST case — a ~7.4k-token
# prefix prefill no returning user ever pays again — and the old tape hid it behind a
# 17s `Hide` block, which is the misleading half of both options. So: prime the checkpoint
# off camera, then record the warm run every user actually gets after their first session.
# The turn is then short enough to show unbroken, with no hidden cut at all.
#
# The priming run is headless, which is safe because the stable prefix is the same in both
# modes: `build_system_prompt()` takes no headless/TUI/yolo argument, so the checkpoint a
# headless run writes is the one the TUI run reads. It must run AFTER the fixture files
# exist (the repo map is part of the key) and it costs one extra model load per recording.
#
# WHY EVERY TURN BEAT IS A FIXED SLEEP AND NOT `Wait+Screen /…/`:
# vhs's screen matcher polls fine for shell output, but it stops seeing the screen once
# chad's TUI starts a turn (measured: two runs, both frozen at the same line, both burned
# the full WaitTimeout while the thing they waited for sat on screen). So the turn is
# timed. Overshoot: a few dull frames of a finished transcript. Undershoot: `/exit` typed
# into a running turn. When in doubt, lengthen — never shorten.
#
# Offsets below were measured on the shipped Qwen3.8-27B, warm, M4 Pro (24 GB): ready 4.1s,
# first tool +5.5s, edit +23.1s, verify +27.0s, turn done +31.4s. Re-measure if the model
# or the ask changes — run the same ask headless with timestamped output and read them off.
#
# Sampling is not deterministic: a take where the model fumbles (invents a path, re-reads,
# wanders) is a bad demo, not a bad tape. Watch the frames; if it wandered, run it again.

Output docs/demo.gif

Set Shell zsh
Set FontSize 15
Set Width 1200
Set Height 640
Set Padding 12
Set TypingSpeed 40ms
Set WaitTimeout 4m

# --- fixture: a tiny project with one failing test (off camera) -----------------
# `chad` is bound to THIS working tree, not whatever is installed globally — the GIF
# must show the code in the repo it ships with.
Hide
Type `unalias chad 2>/dev/null; chad() { VIRTUAL_ENV= uv run --no-sync --project ~/repo/chad chad "$@" }`
Enter
Type `demo_dir=$(mktemp -d /tmp/chad-demo-XXXX) && cd "$demo_dir"`
Enter
# pytest comes from a venv in the demo dir, so both the on-camera runs and chad's own
# verification resolve it. VIRTUAL_ENV_DISABLE_PROMPT keeps the venv name out of the
# prompt, and clearing VIRTUAL_ENV for `chad` keeps uv from warning about the mismatch.
Type `export VIRTUAL_ENV_DISABLE_PROMPT=1`
Enter
Type `uv venv -q && uv pip install -q pytest && source .venv/bin/activate`
Enter
Sleep 8s
Type `printf 'def split_evenly(total_cents, n):\n    """Split a bill n ways. Every cent has to land somewhere."""\n    return [total_cents // n] * n\n' > budget.py`
Enter
Type `printf 'from budget import split_evenly\n\n\ndef test_no_cent_is_lost():\n    shares = split_evenly(100, 3)\n    assert sum(shares) == 100\n    assert shares == [34, 33, 33]\n\n\ndef test_even_split_is_even():\n    assert split_evenly(90, 3) == [30, 30, 30]\n' > test_budget.py`
Enter

# --- prime the on-disk warm prefix (off camera; see the header) -------------------
# One headless turn writes the checkpoint the recorded session reads back. The ask is
# deliberately trivial and edits nothing; the fixture is rewritten afterwards regardless,
# so a fumbled priming turn cannot leak into the take.
Type `chad --yolo "run the bash command 'true' and then stop" >/dev/null 2>&1`
Enter
Sleep 150s
Type `printf 'def split_evenly(total_cents, n):\n    """Split a bill n ways. Every cent has to land somewhere."""\n    return [total_cents // n] * n\n' > budget.py`
Enter
Type "clear"
Enter
Show

# --- the bug is real (on camera) --------------------------------------------------
# One cent vanishes into floor division. Shell output, so Wait+Screen is reliable here.
Type "pytest -q"
Enter
Wait+Screen /1 failed/
Sleep 2.5s

# --- chad starts: banner up in ~1s, weights still loading -------------------------
Type "chad --yolo"
Enter
Wait+Screen /type ahead/
Sleep 1.5s

# Type the ask WHILE the weights load — the type-ahead queue is a real feature, and it
# means the load costs you nothing.
Type "the first test fails — fix split_evenly in budget.py so no cent is lost, then run pytest -q once to verify"
Sleep 1s
Enter

# --- the run: warm start, then reads, reasons, edits, verifies --------------------
# Nothing is hidden here. The turn opens with chad's own "[warm start: N prefix tokens
# from disk cache]" line, which is the sentence that explains why the rest is quick, then
# runs ~31s to done. The margin below is deliberate: overshooting leaves the finished
# transcript on screen (which the viewer wants to read anyway), undershooting types
# `/exit` into a running turn.
Sleep 42s

# --- green, on a laptop, on local weights ----------------------------------------
Type "/exit"
Enter
Sleep 1.5s

# --- and it stuck: your own shell, your own test run ------------------------------
Type "pytest -q"
Enter
Sleep 4s
