"""Unit tests for session persistence (session.py) — save/load round-trip + isolation.

Run: `uv run python test_session.py`
"""
import json
import os
import tempfile
import time

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


def test_mint_list_and_fork(tmp_path):
    # Multiple sessions per cwd; resume of an old session forks (new file) and leaves the
    # original byte-for-byte untouched — the entire branching feature (plan 043).
    session.SESS_DIR = os.path.join(tmp_path, "sessions")
    a = tempfile.mkdtemp(prefix="proj_fork_")

    check("minted id shape", session.new_session_id().count("-") == 2)

    old = [{"role": "user", "content": "first task"},
           {"role": "assistant", "content": "ok"}]
    p_old = session.save_session(a, old, {}, session_id="20260101-000000-aaaa")
    session.save_session(a, [{"role": "user", "content": "second"}], {},
                         session_id="20260101-000100-bbbb")

    items = session.list_sessions(a)
    check("lists both sessions", len(items) == 2, items)
    check("newest first", items[0]["session_id"] == "20260101-000100-bbbb")
    check("title from first user msg", items[0]["title"] == "second")

    # load a SPECIFIC (older) session, then fork it
    data = session.load_session(a, "20260101-000000-aaaa")
    check("loads specific session", data["messages"] == old)
    before = open(p_old, "rb").read()

    forked = data["messages"] + [{"role": "user", "content": "branch"}]
    p_fork = session.save_session(a, forked, {}, session_id="20260101-000200-cccc")
    check("fork is a new file", p_fork != p_old and os.path.isfile(p_fork))
    check("original untouched after fork", open(p_old, "rb").read() == before)
    check("now three sessions", len(session.list_sessions(a)) == 3)

    # newest (== load_session with no id, the `-c` path) is the fork
    check("newest == fork", session.load_session(a)["messages"] == forked)


def test_prune_keeps_newest(tmp_path):
    session.SESS_DIR = os.path.join(tmp_path, "sessions")
    a = tempfile.mkdtemp(prefix="proj_prune_")
    for i in range(session.RETAIN + 5):
        session.save_session(a, [{"role": "user", "content": f"t{i}"}], {},
                             session_id=f"20260101-0000{i:02d}-{i:04x}")
    items = session.list_sessions(a)
    check("pruned to RETAIN", len(items) == session.RETAIN, len(items))
    check("oldest removed", all(it["session_id"] != "20260101-000000-0000" for it in items))
    check("oldest file gone",
          not os.path.isfile(session._session_path(a, "20260101-000000-0000")))


def test_adopt_legacy(tmp_path):
    # A pre-043 single-slot <cwdhash>.json is adopted as one session on first listing.
    session.SESS_DIR = os.path.join(tmp_path, "sessions")
    a = tempfile.mkdtemp(prefix="proj_legacy_")
    os.makedirs(session.SESS_DIR, exist_ok=True)
    legacy = session._legacy_path(a)
    with open(legacy, "w") as f:
        json.dump({"cwd": a, "updated": time.time(), "meta": {},
                   "messages": [{"role": "user", "content": "legacy work"}]}, f)

    items = session.list_sessions(a)
    check("legacy adopted as one session", len(items) == 1, items)
    check("legacy title", items[0]["title"] == "legacy work")
    check("legacy file removed", not os.path.isfile(legacy))
    check("legacy messages load", session.load_session(a)["messages"][0]["content"] == "legacy work")
    # idempotent: listing again doesn't re-adopt / duplicate
    check("adopt is once-only", len(session.list_sessions(a)) == 1)


def test_index_0600_and_corrupt_tolerated(tmp_path):
    session.SESS_DIR = os.path.join(tmp_path, "sessions")
    a = tempfile.mkdtemp(prefix="proj_idx_")
    session.save_session(a, [{"role": "user", "content": "hi"}], {},
                         session_id="20260101-000000-abcd")
    ip = session._index_path(a)
    check("index is 0600", oct(os.stat(ip).st_mode & 0o777) == "0o600",
          oct(os.stat(ip).st_mode & 0o777))

    # A corrupt index must not crash listing — it is rebuilt from the session files.
    with open(ip, "w") as f:
        f.write("{ this is not json")
    items = session.list_sessions(a)
    check("survives corrupt index", len(items) == 1, items)
    check("rebuilt from files", items[0]["session_id"] == "20260101-000000-abcd")


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as home:
        test_session(home)
        test_session_perms_0600(home)
        test_mint_list_and_fork(home)
        test_prune_keeps_newest(home)
        test_adopt_legacy(home)
        test_index_0600_and_corrupt_tolerated(home)
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
