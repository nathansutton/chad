# chad demo tape — regenerates docs/demo.gif with vhs (https://github.com/charmbracelet/vhs)
#
# Run manually on the dev Mac per release (see RELEASING.md). CI cannot run this:
# the demo drives a real model on Apple Silicon. The tape is checked in so the
# demo can't silently rot relative to the actual UI — if the TUI changes, this
# file is the diff that says the GIF is stale.
#
# Two moments are the demo; keep both:
#   * the banner — the session states what it is: version, model, context, cwd. The
#     model name is the whole pitch (a 35B on this laptop, not an API).
#   * the unbroken run — read, reason, edit, verify, with the work visible as it lands.
#
# The session runs `--yolo` deliberately: in the default (normal) mode every edit and
# every shell command opens an approval gate, and a recording can only answer those on a
# timer, which cost ~35s of the runtime sitting on panels waiting to be dismissed. The
# approval UX is real and worth showing — but a GIF is the wrong place to show waiting.
#
# WHY EVERY BEAT BELOW IS A FIXED SLEEP AND NOT `Wait+Screen /…/`:
# vhs's screen matcher polls fine for shell output, but it stops seeing the screen once
# chad's TUI starts a turn (measured: two runs, both frozen at the same line, both burned
# the full WaitTimeout while the thing they waited for sat on screen). So the turn is
# timed. Overshoot: a few dull frames of a finished transcript. Undershoot: `/exit` typed
# into a running turn. When in doubt, lengthen — never shorten.
#
# Sleep values below are measured on the 35B (Q2_K_XL, warm weights): load ~10s, first
# tool ~24s after the ask, edit ~32s, pytest ~33s, a second pytest ~38s (chad re-verifies
# once on its own), done ~40s. Re-measure after a model or prompt change by running the
# same ask headless with timestamped output.
#
# Rehearsal (mechanics only, no 13 GB load):
#   CHAD_MODEL=nathansutton/Ornith-1.0-9B-UD-Q4_K_XL-MLX vhs docs/demo.tape
# The 9B is ~3x slower to the first gate, so it overruns these sleeps — rehearse the
# wiring with it, but never ship a GIF recorded from it.

Output docs/demo.gif

Set Shell zsh
Set FontSize 15
Set Width 1200
Set Height 640
Set Padding 12
Set TypingSpeed 40ms
Set WaitTimeout 2m

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
# means the 10-second load costs you nothing.
Type "the first test fails — fix split_evenly in budget.py so no cent is lost, then run pytest -q once to verify"
Sleep 1s
Enter

# The first ~17s of a turn is a silent prefill: a spinner and ↑0 ↓0, nothing to watch.
# Cut THAT, not the streaming that follows — the reasoning and the Read lines are the
# interesting part, and the first take made exactly this mistake in reverse.
Hide
Sleep 17s
Show

# --- the run: reads, reasons, edits, verifies, re-verifies -------------------------
# ~25s of turn is left after the cut above. The extra margin is deliberate: overshooting
# leaves the finished transcript on screen (which the viewer wants to read anyway),
# undershooting types `/exit` into a running turn.
Sleep 32s

# --- green, on a laptop, on local weights ----------------------------------------
Type "/exit"
Enter
Sleep 1.5s

# --- and it stuck: your own shell, your own test run ------------------------------
Type "pytest -q"
Enter
Sleep 4s
