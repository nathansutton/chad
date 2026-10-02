"""Unit tests for session persistence (session.py) — save/load round-trip + isolation.

The store itself (CHAD_SESSION_DIR) is pointed at a per-test tmp dir by conftest.

Run: `uv run python tests/test_session.py`
"""
import json
import os
import pathlib
import tempfile
import time

import pytest

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


def _proj(tmp_path, name):
    """A real project dir to key sessions on. A str, because it lands in JSON as `cwd`."""
    d = tmp_path / name
    d.mkdir()
    return str(d)


def test_session(tmp_path):
    a = _proj(tmp_path, "proj_a")
    b = _proj(tmp_path, "proj_b")

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
    a = _proj(tmp_path, "proj_perms")
    msgs = [{"role": "user", "content": "secret bash command"}]
    p = session.save_session(a, msgs, {})
    check("perms save returns path", bool(p) and os.path.isfile(p))
    check("session file is 0600", oct(os.stat(p).st_mode & 0o777) == "0o600",
          oct(os.stat(p).st_mode & 0o777))
    # round-trip still works under the tightened perms
    check("perms round-trips messages", session.load_session(a)["messages"] == msgs)


def test_mint_list_and_fork(tmp_path):
    # Multiple sessions per cwd; resume of an old session forks (new file) and leaves the
    # original byte-for-byte untouched — the entire branching feature.
    a = _proj(tmp_path, "proj_fork")

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
    a = _proj(tmp_path, "proj_prune")
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
    a = _proj(tmp_path, "proj_legacy")
    os.makedirs(session.sessions_root(), exist_ok=True)
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


@pytest.mark.skipif(os.geteuid() == 0, reason="root writes through directory permissions")
def test_adopt_legacy_keeps_the_source_when_the_copy_fails(tmp_path):
    """The legacy file is the only copy of that conversation. If the adopted session
    cannot be written, deleting it destroys the user's transcript; keeping it means the
    next listing simply tries again."""
    a = _proj(tmp_path, "proj_legacy_unwritable")
    os.makedirs(session.sessions_root(), exist_ok=True)
    legacy = session._legacy_path(a)
    with open(legacy, "w") as f:
        json.dump({"cwd": a, "updated": time.time(), "meta": {},
                   "messages": [{"role": "user", "content": "the only copy"}]}, f)
    # The adopted copy is written into the project's session directory; a read-only one
    # makes that write fail for real.
    store = session._dir(a)
    os.mkdir(store)
    os.chmod(store, 0o500)
    try:
        items = session.list_sessions(a)
    finally:
        os.chmod(store, 0o700)
    check("nothing adopted", items == [], items)
    check("legacy file survives a failed write", os.path.isfile(legacy))
    check("and it still parses", json.load(open(legacy))["messages"][0]["content"]
          == "the only copy")


def test_adopt_legacy_sets_an_unparseable_file_aside(tmp_path):
    """A legacy file that will not parse is unreadable to us, not worthless to the user:
    rename it rather than delete it — and renaming also ends the adoption attempt, which
    would otherwise repeat on every listing."""
    a = _proj(tmp_path, "proj_legacy_corrupt")
    os.makedirs(session.sessions_root(), exist_ok=True)
    legacy = session._legacy_path(a)
    with open(legacy, "w") as f:
        f.write("{ this is not json")

    check("corrupt legacy lists as nothing", session.list_sessions(a) == [])
    check("corrupt legacy moved aside", os.path.isfile(legacy + ".corrupt"))
    check("original gone", not os.path.isfile(legacy))
    check("its bytes are intact",
          open(legacy + ".corrupt").read() == "{ this is not json")


def test_index_0600_and_corrupt_tolerated(tmp_path):
    a = _proj(tmp_path, "proj_idx")
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


def test_persisted_copy_masks_known_prefix_secrets(tmp_path):
    # A credential echoed into a tool result or a tool-call argument must not reach disk:
    # the file outlives the session and is replayed on resume. A git sha must survive,
    # prose is never touched, and the in-memory transcript keeps the real values.
    a = _proj(tmp_path, "proj_redact")
    tok = "abcdefghijklmnopqrstuvwxyz0123456789ABCD"  # 40 chars
    sha = "a3f9c1e2b4d6071829abcdef0123456789abcdef"
    call = "<tool_call>\n" + json.dumps({"name": "bash", "arguments": {
        "command": "curl -H 'Authorization: Bearer " + tok + "' https://example.com"}}) \
        + "\n</tool_call>"
    msgs = [{"role": "user", "content": "the token is Bearer " + tok},
            {"role": "assistant", "content": call},
            {"role": "tool", "name": "bash", "content": "Authorization: Bearer " + tok + "\n"},
            {"role": "tool", "name": "bash", "content": "commit " + sha + "\n"},
            {"role": "assistant", "content": "done; it was Bearer " + tok}]
    before = json.loads(json.dumps(msgs))
    session.save_session(a, msgs, {}, session_id="20260101-000000-5ec2")

    masked = [msgs[0],
              {**msgs[1], "content": call.replace(tok, "<redacted:40>")},
              {**msgs[2], "content": "Authorization: Bearer <redacted:40>\n"},
              msgs[3],
              msgs[4]]
    got = session.load_session(a)["messages"]
    check("tool result + tool-call args masked; sha and prose untouched", got == masked, got)
    check("in-memory transcript untouched", msgs == before)

    # resuming forks: the masked transcript saved again is byte-stable (the mask is idempotent)
    session.save_session(a, got, {}, session_id="20260101-000100-5ec2")
    check("re-save of a masked transcript is stable",
          session.load_session(a)["messages"] == masked)


def test_prune_reports_what_it_removed(tmp_path):
    a = _proj(tmp_path, "proj_prune_notice")
    removed = []
    for i in range(session.RETAIN):
        session.save_session(a, [{"role": "user", "content": f"t{i}"}], {},
                             session_id=f"20260101-0000{i:02d}-{i:04x}",
                             on_prune=removed.append)
    check("no notice while under the limit", removed == [], removed)
    session.save_session(a, [{"role": "user", "content": "one more"}], {},
                         session_id="20260101-000100-ffff", on_prune=removed.append)
    check("one notice for the one pruned", removed == [1], removed)


def test_prune_without_a_listener_still_prunes(tmp_path):
    a = _proj(tmp_path, "proj_prune_quiet")
    path = ""
    for i in range(session.RETAIN + 1):
        path = session.save_session(a, [{"role": "user", "content": f"t{i}"}], {},
                                    session_id=f"20260101-0000{i:02d}-{i:04x}")
    check("returns the path", path == session._session_path(a, "20260101-000020-0014"), path)
    check("pruned to RETAIN", len(session.list_sessions(a)) == session.RETAIN)


def test_prune_returns_zero_under_the_limit(tmp_path):
    a = _proj(tmp_path, "proj_prune_zero")
    session.save_session(a, [{"role": "user", "content": "only"}], {})
    check("nothing removed", session._prune(a) == 0)


def test_recap_keeps_the_last_three_user_lines_oldest_first():
    msgs = [{"role": "system", "content": "sys"}]
    for i in range(5):
        msgs += [{"role": "user", "content": f"ask {i}"},
                 {"role": "assistant", "content": f"answer {i}"},
                 {"role": "tool", "content": f"result {i}"}]
    check("last three", session.recap(msgs) == ["ask 2", "ask 3", "ask 4"], session.recap(msgs))


def test_recap_skips_a_harness_prefix():
    msgs = [{"role": "user",
             "content": "[PLAN MODE. Research first …]\n\nadd a retry to fetch()"}]
    check("prefix skipped", session.recap(msgs) == ["add a retry to fetch()"],
          session.recap(msgs))


def test_recap_clips_a_long_line():
    got = session.recap([{"role": "user", "content": "x" * 300}])
    check("clipped to width", len(got[0]) == 100 and got[0].endswith("…"), got)


def test_recap_keeps_only_the_first_line():
    got = session.recap([{"role": "user", "content": "first line\nsecond line"}])
    check("first line only", got == ["first line"], got)


def test_recap_of_nothing_the_user_asked_is_empty():
    check("empty", session.recap([]) == [])
    check("system only", session.recap([{"role": "system", "content": "sys"}]) == [])


def test_recap_tolerates_missing_content():
    check("None content", session.recap([{"role": "user", "content": None}]) == [])


if __name__ == "__main__":
    for test in (test_session, test_session_perms_0600, test_mint_list_and_fork,
                 test_prune_keeps_newest, test_prune_reports_what_it_removed,
                 test_prune_without_a_listener_still_prunes,
                 test_prune_returns_zero_under_the_limit, test_adopt_legacy,
                 test_adopt_legacy_keeps_the_source_when_the_copy_fails,
                 test_adopt_legacy_sets_an_unparseable_file_aside,
                 test_index_0600_and_corrupt_tolerated,
                 test_persisted_copy_masks_known_prefix_secrets):
        # Same isolation conftest gives each test under pytest.
        with tempfile.TemporaryDirectory() as home, pytest.MonkeyPatch.context() as mp:
            mp.setenv("CHAD_SESSION_DIR", os.path.join(home, "sessions"))
            test(pathlib.Path(home))
    print(f"\n{PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
