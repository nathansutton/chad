"""Conversation persistence so a session survives across runs (`cli.py --continue`).

Native Claude Code records every conversation and can resume it; this is the local
analogue, scoped to per-directory resume (you almost always want to continue the work
for *this* project). Only the message list is persisted as JSON — NOT the KV cache: the
next turn re-prefills the restored transcript (the system-prefix warm cache still
applies), which is the price of a cold resume. Best-effort throughout: a failure to
save or load never breaks a turn.
"""
import hashlib
import json
import os
import time

SESS_DIR = os.path.expanduser("~/.chad/sessions")


def _key(cwd: str) -> str:
    return hashlib.sha1(os.path.abspath(cwd).encode("utf-8", "ignore")).hexdigest()[:16]


def _path(cwd: str) -> str:
    return os.path.join(SESS_DIR, _key(cwd) + ".json")


def save_session(cwd: str, messages: list, meta: dict = None) -> str:
    """Atomically persist the conversation for `cwd`. Returns the path, or '' on failure."""
    try:
        os.makedirs(SESS_DIR, exist_ok=True)
        path = _path(cwd)
        tmp = path + f".{os.getpid()}.tmp"
        # Create 0600 from the start so the conversation store (full tool args and
        # results) is never briefly world-readable; os.replace preserves the mode.
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(fd, "w") as f:
            json.dump({"cwd": os.path.abspath(cwd), "updated": time.time(),
                       "meta": meta or {}, "messages": messages}, f)
        os.replace(tmp, path)  # atomic on POSIX — never leaves a half-written session
        return path
    except OSError:
        return ""


def load_session(cwd: str):
    """Return the saved {cwd, updated, meta, messages} dict for `cwd`, or None."""
    path = _path(cwd)
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


def session_summary(cwd: str) -> str:
    """One-line description of the saved session for `cwd` (for the resume notice)."""
    data = load_session(cwd)
    if not data:
        return ""
    users = sum(1 for m in data["messages"] if m.get("role") == "user")
    age = max(0, int(time.time() - data.get("updated", 0)))
    when = f"{age // 3600}h ago" if age >= 3600 else (f"{age // 60}m ago" if age >= 60 else "just now")
    return f"{users} prior user turn{'s' * (users != 1)}, last active {when}"
