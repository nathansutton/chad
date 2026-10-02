"""Conversation persistence so a session survives across runs (`cli.py --continue`,
`--resume`, and the TUI `/resume` picker).

Native Claude Code records every conversation, can list them, and can fork one; this is
the local analogue, scoped per project directory (you almost always want to continue the
work for *this* repo). Only the message list is persisted as JSON — NOT the KV cache: the
next turn re-prefills the restored transcript (the system-prefix warm cache still
applies), which is the price of a cold resume. Best-effort throughout: a failure to save
or load never breaks a turn.

Layout:

    ~/.chad/sessions/<cwdhash>/<session_id>.json   one file per session
    ~/.chad/sessions/<cwdhash>/index.json          title/updated/turns per session

`session_id` = `YYYYMMDD-HHMMSS-<4 hex>`, minted at Agent construction. Each save
overwrites only *its own* session file, so resuming a session (which mints a fresh id and
seeds the old messages) never rewrites the original — every resume is implicitly a fork.
A legacy single-slot `~/.chad/sessions/<cwdhash>.json` file is adopted as one session the
first time the directory is listed. Retention keeps the newest N per cwd, pruned on save.

What lands on disk has known-prefix secrets (bearer tokens, `sk-`/`ghp_` keys, …) masked
in tool results and tool-call arguments; the in-memory transcript keeps the real values.
"""
import hashlib
import json
import os
import secrets
import time
from typing import Callable

from . import config
from .diag import redact

SESS_DIR = os.path.expanduser("~/.chad/sessions")
RETAIN = 20            # keep the newest N sessions per cwd; prune older on save
_TITLE_LEN = 60        # first-user-message title truncation for the index / picker


def _key(cwd: str) -> str:
    return hashlib.sha1(os.path.abspath(cwd).encode("utf-8", "ignore")).hexdigest()[:16]


def sessions_root() -> str:
    """The store: `CHAD_SESSION_DIR` when set, else ~/.chad/sessions. Read on every call,
    so a test or eval suite can relocate it without ever writing real home state."""
    return config.env_str("CHAD_SESSION_DIR") or SESS_DIR


def _dir(cwd: str) -> str:
    """Per-cwd session directory."""
    return os.path.join(sessions_root(), _key(cwd))


def _legacy_path(cwd: str) -> str:
    """The pre-043 single-slot file (`<cwdhash>.json`), adopted on first listing."""
    return os.path.join(sessions_root(), _key(cwd) + ".json")


def _session_path(cwd: str, session_id: str) -> str:
    return os.path.join(_dir(cwd), session_id + ".json")


def _index_path(cwd: str) -> str:
    return os.path.join(_dir(cwd), "index.json")


def new_session_id() -> str:
    """A fresh session id: sortable timestamp + 4 random hex (collision-safe within a
    second). Minted at Agent construction; a resume mints a new one → implicit fork."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def _atomic_write_json(path: str, obj) -> bool:
    """Write `obj` as JSON to `path` atomically at 0600. Returns True on success.

    Create 0600 from the start so the conversation store (full tool args and results) is
    never briefly world-readable; os.replace preserves the mode and is atomic on POSIX —
    a reader never sees a half-written file."""
    tmp = path + f".{os.getpid()}.tmp"
    try:
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


def _load_path(path: str):
    """Return a saved {cwd, updated, meta, messages, session_id} dict, or None."""
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("messages"), list):
            return data
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return None


def _first_user_title(messages: list, limit: int = _TITLE_LEN) -> str:
    """The first user message, whitespace-collapsed and truncated — the session title."""
    for m in messages:
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str) and c.strip():
                t = " ".join(c.split())
                return t[:limit] + ("…" if len(t) > limit else "")
    return ""


def _turns(messages: list) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


# -- index (avoids parsing every session file just to list them) -------------

def _load_index(cwd: str) -> dict:
    """The per-cwd index, or an empty one if missing/corrupt (corrupt is tolerated —
    `list_sessions` rebuilds it from the session files on disk)."""
    try:
        with open(_index_path(cwd)) as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("sessions"), dict):
            return data
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return {"sessions": {}}


def _write_index(cwd: str, sessions: dict) -> None:
    _atomic_write_json(_index_path(cwd), {"sessions": sessions})


def _update_index(cwd: str, session_id: str, messages: list, updated: float) -> None:
    idx = _load_index(cwd)
    idx["sessions"][session_id] = {
        "title": _first_user_title(messages),
        "updated": updated,
        "turns": _turns(messages),
    }
    _write_index(cwd, idx["sessions"])


def _adopt_legacy(cwd: str) -> None:
    """Migrate a pre-043 `<cwdhash>.json` file into the sessioned store as one session,
    then remove it so it is adopted exactly once. Best-effort.

    The source file is the ONLY copy of that conversation, so it is removed only once a
    copy is known to exist: the adopted session was written (or was already there from
    an earlier adoption). A file that will not parse is renamed aside rather than
    deleted — it is unreadable to us, not worthless to the user — and renaming it also
    ends the adoption attempt, which would otherwise repeat on every listing."""
    legacy = _legacy_path(cwd)
    if not os.path.isfile(legacy):
        return
    try:
        data = _load_path(legacy)
        os.makedirs(_dir(cwd), exist_ok=True)
        if not data:
            os.replace(legacy, legacy + ".corrupt")
            return
        updated = data.get("updated") or time.time()
        # Derive a stable id from the legacy file's own timestamp so re-listing is
        # idempotent even if the write below races; -0000 marks the adopted slot.
        sid = time.strftime("%Y%m%d-%H%M%S", time.localtime(updated)) + "-0000"
        target = _session_path(cwd, sid)
        adopted = os.path.exists(target)
        if not adopted:
            adopted = _atomic_write_json(target, {
                "cwd": data.get("cwd", os.path.abspath(cwd)),
                "session_id": sid,
                "updated": updated,
                "meta": data.get("meta", {}),
                "messages": data["messages"],
            })
            if adopted:
                _update_index(cwd, sid, data["messages"], updated)
        if adopted:
            os.remove(legacy)
    except OSError:
        pass


def list_sessions(cwd: str, limit: int = None) -> list:
    """Sessions for `cwd`, newest first: `[{session_id, title, updated, turns}, ...]`.

    Adopts a legacy file first, then reconciles the index against the files actually on
    disk — so a corrupt/lost index is rebuilt and orphaned index rows are dropped. Cheap:
    the common path reads only the index."""
    _adopt_legacy(cwd)
    sessions = _load_index(cwd)["sessions"]
    try:
        names = os.listdir(_dir(cwd))
    except OSError:
        names = []
    on_disk = set()
    changed = False
    for name in names:
        if name == "index.json" or not name.endswith(".json") or name.endswith(".tmp"):
            continue
        sid = name[: -len(".json")]
        on_disk.add(sid)
        if sid not in sessions:  # file present but unindexed (corrupt index / crash)
            data = _load_path(_session_path(cwd, sid))
            if data:
                sessions[sid] = {
                    "title": _first_user_title(data["messages"]),
                    "updated": data.get("updated", 0),
                    "turns": _turns(data["messages"]),
                }
                changed = True
    for sid in list(sessions):  # drop rows whose file was pruned/removed
        if sid not in on_disk:
            del sessions[sid]
            changed = True
    if changed:
        _write_index(cwd, sessions)
    items = [{"session_id": sid, **meta} for sid, meta in sessions.items()]
    items.sort(key=lambda it: it.get("updated", 0), reverse=True)
    return items[:limit] if limit else items


def _prune(cwd: str, keep: int = RETAIN) -> int:
    """Retention: keep the newest `keep` sessions, remove older files + index rows.
    Returns how many sessions were removed."""
    sessions = _load_index(cwd)["sessions"]
    if len(sessions) <= keep:
        return 0
    ordered = sorted(sessions.items(), key=lambda kv: kv[1].get("updated", 0), reverse=True)
    for sid, _meta in ordered[keep:]:
        try:
            os.remove(_session_path(cwd, sid))
        except OSError:
            pass
        sessions.pop(sid, None)
    _write_index(cwd, sessions)
    return len(ordered) - keep


def _redacted(messages: list) -> list:
    """A copy with known-prefix secrets masked in tool results and tool-call arguments.
    The in-memory transcript is untouched (the running turn needs the real values); only
    the on-disk copy, which outlives the session and is replayed on resume, is masked.
    Bare high-entropy blobs are left alone so a resumed transcript keeps its git shas.
    Only the messages that change are copied; user and assistant prose is never touched."""
    out = []
    for m in messages:
        c = m.get("content")
        role = m.get("role")
        if isinstance(c, str) and (role == "tool" or (role == "assistant" and "<tool_call>" in c)):
            masked = redact(c, bare_blobs=False)
            if masked != c:
                m = {**m, "content": masked}
        out.append(m)
    return out


def save_session(cwd: str, messages: list, meta: dict = None,
                 session_id: str = None, *,
                 on_prune: Callable[[int], None] | None = None) -> str:
    """Atomically persist the conversation for `cwd` to its own session file. Mints a
    `session_id` if none is given. Updates the index and prunes old sessions (best-effort),
    calling `on_prune` with the count when any were removed.
    Returns the session file path, or '' on failure."""
    try:
        os.makedirs(_dir(cwd), exist_ok=True)
        session_id = session_id or new_session_id()
        updated = time.time()
        path = _session_path(cwd, session_id)
        ok = _atomic_write_json(path, {"cwd": os.path.abspath(cwd), "session_id": session_id,
                                       "updated": updated, "meta": meta or {},
                                       "messages": _redacted(messages)})
        if not ok:
            return ""
        _update_index(cwd, session_id, messages, updated)
        removed = _prune(cwd)
        if removed and on_prune is not None:
            on_prune(removed)
        return path
    except OSError:
        return ""


def load_session(cwd: str, session_id: str = None):
    """Return the saved {cwd, updated, meta, messages, session_id} dict. With `session_id`
    that specific session; without one, the most recent (adopting a legacy file first).
    None if there is nothing to resume."""
    if session_id is None:
        items = list_sessions(cwd, limit=1)
        if not items:
            return None
        session_id = items[0]["session_id"]
    return _load_path(_session_path(cwd, session_id))


def _ago(age_s: int) -> str:
    if age_s >= 3600:
        return f"{age_s // 3600}h ago"
    if age_s >= 60:
        return f"{age_s // 60}m ago"
    return "just now"


def describe(item: dict) -> str:
    """One-line label for a `list_sessions` entry (the picker / resume notice):
    `2h ago · 14 turns · "fix the flaky retry test…"`."""
    when = _ago(max(0, int(time.time() - item.get("updated", 0))))
    turns = item.get("turns", 0)
    title = item.get("title") or "(no title)"
    return f'{when} · {turns} turn{"s" * (turns != 1)} · "{title}"'


def recap(messages: list, last: int = 3, width: int = 100) -> list[str]:
    """The last few things the user asked in a conversation, oldest first, one line
    each — what a resumed session shows so the user is not typing into a conversation
    they cannot see. A harness prefix in square brackets is not something the user
    typed, so it is skipped."""
    asked = []
    for m in messages:
        if m.get("role") != "user":
            continue
        text = str(m.get("content") or "").strip()
        while text.startswith("["):
            end = text.find("]")
            if end < 0:
                break
            text = text[end + 1:].strip()
        line = text.splitlines()[0].strip() if text else ""
        if line:
            asked.append(line if len(line) <= width else line[:width - 1] + "…")
    return asked[-last:]


def session_summary(cwd: str) -> str:
    """One-line description of the most recent session for `cwd` (the `-c` resume notice)."""
    data = load_session(cwd)
    if not data:
        return ""
    users = _turns(data["messages"])
    when = _ago(max(0, int(time.time() - data.get("updated", 0))))
    return f"{users} prior user turn{'s' * (users != 1)}, last active {when}"
