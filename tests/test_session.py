"""Unit tests for session persistence (session.py) — save/load round-trip + isolation.

Run: `uv run python test_session.py`
"""
import os
import tempfile

from chad import session

PASS = 0
FAIL = 0


def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
    else:
        FAIL += 1
        raise AssertionError(f"{name}  {detail}")


def test_session(tmp_path):
    # tmp_path is pytest's per-test temp dir fixture; the __main__ runner passes its own.
    session.SESS_DIR = os.path.join(tmp_path, "sessions")
    a = tempfile.mkdtemp(prefix="proj_a_")
    b = tempfile.mkdtemp(prefix="proj_b_")

    # nothing saved yet
    check("load empty -> None", session.load_session(a) is None)
    check("summary empty -> ''", session.session_summary(a) == "")

    msgs = [{"role": "system", "content": "sys"},
            {"role": "user", "content": "do X"},
            {"role": "assistant", "content": "did X"}]
    p = session.save_session(a, msgs, {"mode": "auto"})
    check("save returns path", bool(p) and os.path.isfile(p))

    got = session.load_session(a)
    check("round-trips messages", got and got["messages"] == msgs, f"got={got}")
    check("round-trips meta", got and got["meta"] == {"mode": "auto"})
    check("summary counts user turns", "1 prior user turn" in session.session_summary(a),
          session.session_summary(a))

    # per-directory isolation: project b is independent of a
    check("dir b isolated", session.load_session(b) is None)
    session.save_session(b, [{"role": "user", "content": "B1"}], {})
    check("dir b own session", session.load_session(b)["messages"][0]["content"] == "B1")
    check("dir a unchanged", session.load_session(a)["messages"] == msgs)

    # overwrite (latest wins, atomic)
    session.save_session(a, msgs + [{"role": "user", "content": "and Y"}], {})
    check("overwrite updates", len(session.load_session(a)["messages"]) == 4)


def test_session_perms_0600(tmp_path):
    # The conversation store records full tool args/results, so it must never be
    # world-readable: save_session creates it 0600 (os.replace preserves the mode).
    session.SESS_DIR = os.path.join(tmp_path, "sessions")
    a = tempfile.mkdtemp(prefix="proj_perms_")
    msgs = [{"role": "user", "content": "secret bash command"}]
    p = session.save_session(a, msgs, {})
    check("perms save returns path", bool(p) and os.path.isfile(p))
    check("session file is 0600", oct(os.stat(p).st_mode & 0o777) == "0o600",
          oct(os.stat(p).st_mode & 0o777))
    # round-trip still works under the tightened perms
    check("perms round-trips messages", session.load_session(a)["messages"] == msgs)


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as home:
        test_session(home)
        test_session_perms_0600(home)
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
