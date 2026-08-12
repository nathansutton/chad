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
"""
import hashlib
import json
import os
import secrets
import time

SESS_DIR = os.path.expanduser("~/.chad/sessions")
RETAIN = 20            # keep the newest N sessions per cwd; prune older on save
_TITLE_LEN = 60        # first-user-message title truncation for the index / picker


def _key(cwd: str) -> str:
    return hashlib.sha1(os.path.abspath(cwd).encode("utf-8", "ignore")).hexdigest()[:16]


def _dir(cwd: str) -> str:
    """Per-cwd session directory."""
    return os.path.join(SESS_DIR, _key(cwd))


def _legacy_path(cwd: str) -> str:
    """The pre-043 single-slot file (`<cwdhash>.json`), adopted on first listing."""
    return os.path.join(SESS_DIR, _key(cwd) + ".json")


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
    # The index carries the real cwd path alongside the rows: the directory NAME is
    # an opaque hash, and the global `sessions_all` view needs to say which project
    # each index belongs to without opening any session file.
    _atomic_write_json(_index_path(cwd),
                       {"cwd": os.path.abspath(cwd), "sessions": sessions})


def _index_row(messages: list, updated: float, meta: dict = None) -> dict:
    """One index row. Thread/status/worktree ride the index so the picker,
    `chad sessions`, and the prune exemption never have to parse session files."""
    meta = meta or {}
    # The explicit goal ("what are you working on?") names the session when given;
    # the first user message stays the fallback so pre-goal sessions look unchanged.
    title = meta.get("goal") or _first_user_title(messages)
    if len(title) > _TITLE_LEN:
        title = title[:_TITLE_LEN] + "…"
    row = {
        "title": title,
        "updated": updated,
        "turns": _turns(messages),
    }
    if meta.get("thread_id"):
        row["thread_id"] = meta["thread_id"]
    if meta.get("status"):
        row["status"] = meta["status"]
    wt = meta.get("worktree")
    if isinstance(wt, dict) and wt.get("path"):
        row["worktree_path"] = wt["path"]
    return row


def _update_index(cwd: str, session_id: str, messages: list, updated: float,
                  meta: dict = None) -> None:
    idx = _load_index(cwd)
    idx["sessions"][session_id] = _index_row(messages, updated, meta)
    _write_index(cwd, idx["sessions"])


def _adopt_legacy(cwd: str) -> None:
    """Migrate a pre-043 `<cwdhash>.json` file into the sessioned store as one session,
    then remove it so it is adopted exactly once. Best-effort."""
    legacy = _legacy_path(cwd)
    if not os.path.isfile(legacy):
        return
    try:
        data = _load_path(legacy)
        os.makedirs(_dir(cwd), exist_ok=True)
        if data:
            updated = data.get("updated") or time.time()
            # Derive a stable id from the legacy file's own timestamp so re-listing is
            # idempotent even if the write below races; -0000 marks the adopted slot.
            sid = time.strftime("%Y%m%d-%H%M%S", time.localtime(updated)) + "-0000"
            target = _session_path(cwd, sid)
            if not os.path.exists(target):
                _atomic_write_json(target, {
                    "cwd": data.get("cwd", os.path.abspath(cwd)),
                    "session_id": sid,
                    "updated": updated,
                    "meta": data.get("meta", {}),
                    "messages": data["messages"],
                })
                _update_index(cwd, sid, data["messages"], updated)
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
                sessions[sid] = _index_row(data["messages"], data.get("updated", 0),
                                           data.get("meta"))
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


def _prune(cwd: str, keep: int = RETAIN) -> None:
    """Retention: keep the newest `keep` sessions, remove older files + index rows.

    Sessions whose worktree checkout still exists are EXEMPT (and don't consume a
    keep slot): the session file is the only thing that knows a kept worktree's
    conversation — pruning it would orphan the worktree, silently breaking the
    `chad -c` re-entry the keep option promised. The exemption ends when the
    worktree is applied or discarded (its path stops existing)."""
    sessions = _load_index(cwd)["sessions"]
    prunable = {sid: row for sid, row in sessions.items()
                if not (row.get("worktree_path")
                        and os.path.isdir(row["worktree_path"]))}
    if len(prunable) <= keep:
        return
    ordered = sorted(prunable.items(), key=lambda kv: kv[1].get("updated", 0), reverse=True)
    for sid, _meta in ordered[keep:]:
        try:
            os.remove(_session_path(cwd, sid))
        except OSError:
            pass
        sessions.pop(sid, None)
    _write_index(cwd, sessions)


def save_session(cwd: str, messages: list, meta: dict = None,
                 session_id: str = None) -> str:
    """Atomically persist the conversation for `cwd` to its own session file. Mints a
    `session_id` if none is given. Updates the index and prunes old sessions (best-effort).
    Returns the session file path, or '' on failure."""
    try:
        os.makedirs(_dir(cwd), exist_ok=True)
        session_id = session_id or new_session_id()
        updated = time.time()
        path = _session_path(cwd, session_id)
        ok = _atomic_write_json(path, {"cwd": os.path.abspath(cwd), "session_id": session_id,
                                       "updated": updated, "meta": meta or {},
                                       "messages": messages})
        if not ok:
            return ""
        _update_index(cwd, session_id, messages, updated, meta)
        _prune(cwd)
        return path
    except OSError:
        return ""


def set_status(cwd: str, session_id: str, status: str) -> bool:
    """Stamp a lifecycle status ('idle', 'needs-attention', 'applied', 'kept',
    'discarded') onto a saved session — file meta + index row. The cli's worktree
    exit gate is the main writer; readers are the picker and `chad sessions`.
    Best-effort like everything here: False on any failure, never raises."""
    data = _load_path(_session_path(cwd, session_id))
    if not data:
        return False
    data.setdefault("meta", {})["status"] = status
    if not _atomic_write_json(_session_path(cwd, session_id), data):
        return False
    _update_index(cwd, session_id, data["messages"], data.get("updated", time.time()),
                  data["meta"])
    return True


def sessions_all(limit: int = 20) -> list:
    """Sessions across EVERY project, newest first — the cross-project "what was I
    doing" view (`chad sessions`) that per-cwd keying otherwise makes impossible.
    Rows are index rows plus the owning cwd; reads only index files, so it stays
    cheap no matter how many projects have history."""
    items = []
    try:
        buckets = os.listdir(SESS_DIR)
    except OSError:
        return []
    for name in buckets:
        idx_path = os.path.join(SESS_DIR, name, "index.json")
        try:
            with open(idx_path) as f:
                idx = json.load(f)
        except (OSError, json.JSONDecodeError, ValueError):
            continue
        if not isinstance(idx, dict) or not isinstance(idx.get("sessions"), dict):
            continue
        cwd = idx.get("cwd") or ""
        if not cwd:  # pre-v2 index: recover the project path from any session file
            for sid in idx["sessions"]:
                data = _load_path(os.path.join(SESS_DIR, name, sid + ".json"))
                if data and data.get("cwd"):
                    cwd = data["cwd"]
                    break
        for sid, row in idx["sessions"].items():
            items.append({"session_id": sid, "cwd": cwd, **row})
    items.sort(key=lambda it: it.get("updated", 0), reverse=True)
    return items[:limit] if limit else items


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
    """One-line label for a `list_sessions`/`sessions_all` entry (the picker /
    `chad sessions`): `2h ago · 14 turns · "fix the flaky retry test…" · kept [wt]`.
    Status and a live-worktree marker only appear when present, so pre-v2 rows
    render exactly as before."""
    when = _ago(max(0, int(time.time() - item.get("updated", 0))))
    turns = item.get("turns", 0)
    title = item.get("title") or "(no title)"
    line = f'{when} · {turns} turn{"s" * (turns != 1)} · "{title}"'
    if item.get("status"):
        line += f' · {item["status"]}'
    if item.get("worktree_path") and os.path.isdir(item["worktree_path"]):
        line += " [wt]"
    return line


def session_summary(cwd: str) -> str:
    """One-line description of the most recent session for `cwd` (the `-c` resume notice)."""
    data = load_session(cwd)
    if not data:
        return ""
    users = _turns(data["messages"])
    when = _ago(max(0, int(time.time() - data.get("updated", 0))))
    return f"{users} prior user turn{'s' * (users != 1)}, last active {when}"
