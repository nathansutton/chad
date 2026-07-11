# chad demo tape — regenerates docs/demo.gif with vhs (https://github.com/charmbracelet/vhs)
#
# Run manually on the dev Mac per release (see RELEASING.md). CI cannot run this:
# the demo drives a real model on Apple Silicon. The tape is checked in so the
# demo can't silently rot relative to the actual UI — if the TUI changes, this
# file is the diff that says the GIF is stale.
#
# The wifi-off moment is the point of the demo: nobody demos the offline claim,
# everybody asserts it. Keep it.
#
# Timings assume the model is already downloaded and was loaded at least once
# (warm weights in the page cache). Expect to tune Sleep/Wait values on first run.

Output docs/demo.gif

Set Shell zsh
Set FontSize 15
Set Width 1200
Set Height 640
Set Padding 12
Set TypingSpeed 40ms
Set WaitTimeout 15m

# --- fixture: a tiny project with one failing test (off camera) -----------------
Hide
Type `demo_dir=$(mktemp -d /tmp/chad-demo-XXXX) && cd "$demo_dir"`
Enter
Type `printf 'def greet(name):\n    return "hello " + name.upper()\n' > greet.py`
Enter
Type `printf 'from greet import greet\n\ndef test_greet():\n    assert greet("nate") == "Hello, Nate"\n' > test_greet.py`
Enter
Type "clear"
Enter
Show

# --- prove the failure, then kill the network (on camera) -----------------------
Type "uv run pytest -q"
Enter
Wait+Screen /1 failed/
Sleep 2s
Type "networksetup -setairportpower en0 off && echo 'wifi: OFF — chad is on his own 🗿'"
Enter
Sleep 2.5s

# --- the task: one scoped ask, headless one-shot ---------------------------------
Type `chad "fix the failing test in test_greet.py — edit greet.py, then run pytest to verify"`
Enter
# The agent reasons, reads, edits, reruns pytest. Wait for the green verification
# in chad's output rather than guessing a duration.
Wait+Screen /1 passed/
Sleep 3s

# --- prove it stuck ---------------------------------------------------------------
Type "uv run pytest -q"
Enter
Wait+Screen /1 passed/
Sleep 4s

# --- restore the network (off camera) ---------------------------------------------
Hide
Type "networksetup -setairportpower en0 on"
Enter
