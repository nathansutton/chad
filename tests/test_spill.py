"""Tests for the shared spill store (spill.py).

Three call sites now depend on this module — bash's head/tail overflow, compaction's
trims, and the per-result backstop cap — so its two properties have to hold for all
of them at once: a pointer the model can act on, and a directory that cannot grow
without bound. chad has already had one unbounded-cache disk incident; the retention
budget is the part of this module most likely to be quietly wrong, so it is what is
asserted hardest below.

`conftest._spill_tmpdir` points CHAD_SPILL_DIR at a per-test tmp dir, so nothing here
touches the developer's real ~/.cache.
"""
import os

from chad import spill

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def test_write_and_point():
    path = spill.write("body\nwith lines\n", "compact")
    check("write returns an absolute path", path and os.path.isabs(path), path)
    check("path is inside the session dir",
          os.path.dirname(path) == os.path.abspath(spill.session_dir()), path)
    check("file is 0600", os.stat(path).st_mode & 0o777 == 0o600,
          oct(os.stat(path).st_mode))
    with open(path) as f:
        check("file holds the body verbatim", f.read() == "body\nwith lines\n")
    check("kind names the file", os.path.basename(path).startswith("compact-"), path)
    notice = f"[… full output saved to {path} — grep it …]"
    check("path_in recovers the pointer", spill.path_in(notice) == path, notice)


def test_path_in_rejects_what_it_cannot_verify():
    """`path_in` reads tool OUTPUT, which is attacker-adjacent text: a result that
    merely quotes the phrase must never be handed back to the model as a live
    pointer. Two independent checks — the file exists, and it lives under the spill
    root — because either alone would pass on a plausible-looking fake."""
    check("no pointer in plain text", spill.path_in("nothing to see here") is None)
    check("a dangling path is not a pointer",
          spill.path_in("saved to /nope/bash-9.log") is None)
    outside = os.path.join(os.path.dirname(spill.base_dir()), "elsewhere-1.log")
    with open(outside, "w") as f:
        f.write("x")
    check("an existing file OUTSIDE the spill root is not a pointer",
          spill.path_in(f"saved to {outside}") is None, outside)
    check("non-string input is not a pointer", spill.path_in(None) is None)


def test_per_kind_file_budget():
    """Each kind keeps its newest KEEP files. The counts differ on purpose: bash
    spills a few times a turn, one compaction pass can trim dozens of results at
    once, so a shared count would let a single compaction evict every bash pointer
    still live in the transcript."""
    for i in range(spill.KEEP["bash"] + 12):
        spill.write(f"bash body {i}", "bash")
    for i in range(5):
        spill.write(f"compact body {i}", "compact")
    names = os.listdir(spill.session_dir())
    check("bash kind capped at its own count",
          len([n for n in names if n.startswith("bash-")]) == spill.KEEP["bash"],
          names)
    check("compaction spills survive a bash flood",
          len([n for n in names if n.startswith("compact-")]) == 5, names)
    check("the newest bash file survived",
          spill.path_in(f"saved to {spill.write('newest', 'bash')}") is not None)


def test_byte_budget(monkeypatch):
    """The dir-wide byte cap is the real backstop — the per-kind counts bound file
    COUNT, and one compaction of a 262k-token window could otherwise write 120 files
    of megabytes each."""
    monkeypatch.setattr(spill, "MAX_DIR_BYTES", 50_000)
    paths = [spill.write("z" * 20_000, "compact") for _ in range(6)]
    d = spill.session_dir()
    total = sum(os.path.getsize(os.path.join(d, n)) for n in os.listdir(d))
    check("dir stays inside the byte budget", total <= spill.MAX_DIR_BYTES, total)
    check("the newest spill survives (its pointer is live)",
          os.path.exists(paths[-1]), paths[-1])
    check("the oldest were evicted first", not os.path.exists(paths[0]), paths[0])


def test_write_failure_is_survivable(monkeypatch):
    """A disk error degrades the notice, never the tool result: every caller treats
    None as 'no pointer available' and falls back to prose guidance."""
    monkeypatch.setenv("CHAD_SPILL_DIR", "/dev/null/nope")
    check("unwritable spill dir returns None", spill.write("x" * 100, "bash") is None)


if __name__ == "__main__":
    test_write_and_point()
    test_path_in_rejects_what_it_cannot_verify()
    test_per_kind_file_budget()
    test_byte_budget()
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
