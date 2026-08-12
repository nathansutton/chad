"""Git-worktree-per-session isolation: chad edits a disposable checkout, not yours.

A local model is a blunter instrument than a frontier one, so the blast radius of a
wrong edit matters more. Instead of mutating the user's checkout in place, a session
gets its own `git worktree` (own branch, own working tree, shared object store) created
from a clean HEAD. On exit the changes are applied back to the origin checkout as one
reviewed step — or kept for later, or discarded. The user's own branch is never
committed to, merged, or checked out; the only write to the origin checkout is a
`git apply` of the session's diff, done with consent (or automatically in a headless
run that opted in).

Sessions bind to worktrees: the session store (session.py) is keyed by the ORIGIN
project directory — not the worktree path — and each save records which worktree the
conversation lives in, so `chad -c` from the project re-enters the same worktree and
its half-finished state. That is the statefulness this module exists for; a worktree
without its session is just a directory.

Layout (override root with CHAD_WORKTREE_DIR — tests do):

    ~/.chad/worktrees/<repohash>/<wtid>/        the checkout (branch chad/<wtid>)
    ~/.chad/worktrees/<repohash>/<wtid>.json    sidecar: origin, base sha, branch

The sidecar lives BESIDE the checkout, not inside it, so it never shows up in the
session's diff and never needs excluding from `git add -A`.

Failure policy: like checkpoint.py, never raise. Worktree setup is an isolation
upgrade, not a prerequisite — any git hiccup degrades to running in place (the
pre-worktree behavior) with a printed reason, and apply/remove report strings the
caller can show verbatim. The one hard rule: a failed apply KEEPS the worktree.
Work is never lost to a cleanup step.
"""

import hashlib
import json
import os
import secrets
import subprocess
import time

_GIT_TIMEOUT_S = 120  # worktree add on a huge repo is seconds, not minutes; never hang

# The active worktree for THIS process, set by enter()/adopt(). One engine, one agent,
# one session per process — a module global is exact, same reasoning as
# cli._resolved_base_url.
_ACTIVE = None


def root() -> str:
    env = os.environ.get("CHAD_WORKTREE_DIR")
    if env:
        return os.path.realpath(env)
    return os.path.join(os.path.expanduser("~"), ".chad", "worktrees")


def _key(toplevel: str) -> str:
    return hashlib.sha1(os.path.realpath(toplevel).encode("utf-8", "ignore")).hexdigest()[:16]


def _git(cwd: str, *args: str, input_text: str = None):
    """Run git in `cwd`. Identity/signing pinned via -c for the few subcommands that
    might consult them, so the user's global config is never required or mutated."""
    cmd = ["git", "-C", cwd, "-c", "user.email=chad@localhost", "-c", "user.name=chad",
           "-c", "commit.gpgsign=false", *args]
    return subprocess.run(cmd, capture_output=True, text=True, input=input_text,
                          timeout=_GIT_TIMEOUT_S, check=False)


def _new_id() -> str:
    """Sortable timestamp + 4 hex — same shape as session ids, minted separately: a
    worktree outlives any one session id (every resume forks a new session id, all of
    them pointing at the same worktree)."""
    return time.strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def repo_top(cwd: str):
    """The repo toplevel containing `cwd`, or None when not inside a git work tree."""
    try:
        r = _git(cwd, "rev-parse", "--show-toplevel")
        return r.stdout.strip() or None if r.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def _sidecar_path(origin_top: str, wtid: str) -> str:
    return os.path.join(root(), _key(origin_top), wtid + ".json")


def _load_sidecar(path: str):
    try:
        with open(path) as f:
            meta = json.load(f)
        if isinstance(meta, dict) and meta.get("path") and meta.get("origin_top"):
            return meta
    except (OSError, json.JSONDecodeError, ValueError):
        pass
    return None


def _write_sidecar(meta: dict) -> bool:
    path = _sidecar_path(meta["origin_top"], meta["id"])
    tmp = path + f".{os.getpid()}.tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(meta, f)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            os.remove(tmp)
        except OSError:
            pass
        return False


# -- state for this process ---------------------------------------------------

def current():
    """The active worktree meta for this process, or None (running in place)."""
    return _ACTIVE


def current_ref():
    """The compact dict Agent.save records in session meta — enough for a later
    `chad -c` to find and re-enter the worktree, nothing more."""
    if not _ACTIVE:
        return None
    return {"id": _ACTIVE["id"], "branch": _ACTIVE["branch"], "path": _ACTIVE["path"],
            "origin_top": _ACTIVE["origin_top"]}


def session_anchor() -> str:
    """The directory the session store keys by: the ORIGIN project dir when a worktree
    is active, else the cwd. Keying by worktree path would strand every conversation
    under a directory that apply/discard deletes."""
    if _ACTIVE:
        rel = _ACTIVE.get("origin_rel", "")
        return os.path.join(_ACTIVE["origin_top"], rel) if rel else _ACTIVE["origin_top"]
    return os.getcwd()


def enter(meta: dict):
    """chdir into the worktree (at the subdir mirroring where chad was launched from,
    when it exists) and mark it active. Everything downstream — tools, repomap,
    seatbelt, checkpoints — follows the process cwd, so this one chdir is the whole
    redirection. Returns the new cwd, or None on failure (caller stays in place)."""
    global _ACTIVE
    dest = os.path.join(meta["path"], meta.get("origin_rel", ""))
    if not os.path.isdir(dest):
        dest = meta["path"]
    try:
        os.chdir(dest)
    except OSError:
        return None
    _ACTIVE = meta
    return dest


def adopt(cwd: str):
    """If `cwd` is already inside a registered chad worktree (a kept one the user
    cd'ed into, or a resumed session re-launching there), activate it — so sessions
    still key to the origin project and the exit flow still offers apply/keep/discard.
    Returns the meta, or None."""
    global _ACTIVE
    top = repo_top(cwd)
    try:
        inside = top and os.path.realpath(top).startswith(os.path.realpath(root()) + os.sep)
    except OSError:
        inside = False
    if not inside:
        return None
    top = os.path.realpath(top)  # sidecar paths are minted from realpaths — match them
    meta = _load_sidecar(os.path.join(os.path.dirname(top), os.path.basename(top) + ".json"))
    if meta and os.path.isdir(meta["path"]):
        _ACTIVE = meta
        return meta
    return None


# -- lifecycle ----------------------------------------------------------------

def eligibility(cwd: str):
    """(ok, reason) — whether a NEW worktree should be created from `cwd`. Not a git
    repo, an existing chad worktree (no nesting), or a dirty tree all say no. Dirty is
    the important one: a worktree starts from HEAD, so uncommitted work would silently
    NOT be visible to the session — running in place is less surprising than a model
    that can't see the file you just told it about."""
    top = repo_top(cwd)
    if not top:
        return False, "not a git repository"
    try:
        if os.path.realpath(top).startswith(os.path.realpath(root()) + os.sep):
            return False, "already inside a chad worktree"
    except OSError:
        pass
    try:
        r = _git(cwd, "status", "--porcelain")
        if r.returncode != 0:
            return False, "git status failed"
        if r.stdout.strip():
            return False, "uncommitted changes (a worktree starts from HEAD and would " \
                          "not see them)"
    except (OSError, subprocess.SubprocessError) as e:
        return False, f"git unavailable: {e}"
    r = _git(cwd, "rev-parse", "--verify", "--quiet", "HEAD")
    if r.returncode != 0:
        return False, "repository has no commits yet"
    return True, ""


def create(cwd: str):
    """Create a worktree + branch from HEAD for a session launched in `cwd`. Returns
    `(meta, "")` on success or `(None, reason)` — on any failure the caller falls back
    to running in place."""
    top = repo_top(cwd)
    if not top:
        return None, "not a git repository"
    wtid = _new_id()
    path = os.path.join(root(), _key(top), wtid)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        base = _git(cwd, "rev-parse", "HEAD").stdout.strip()
        r = _git(top, "worktree", "add", "-b", f"chad/{wtid}", path, "HEAD")
        if r.returncode != 0:
            return None, r.stderr.strip() or "git worktree add failed"
    except (OSError, subprocess.SubprocessError) as e:
        return None, str(e)
    # Where chad was launched relative to the toplevel, so enter() can land in the
    # matching subdir of the worktree ("" = launched at the toplevel itself).
    rel = os.path.relpath(os.path.realpath(cwd), os.path.realpath(top))
    meta = {
        "id": wtid,
        "branch": f"chad/{wtid}",
        "path": path,
        "base": base,
        "origin_top": os.path.realpath(top),
        "origin_rel": "" if rel == "." else rel,
        "created": time.time(),
    }
    _write_sidecar(meta)  # best-effort: an unwritable sidecar costs re-entry, not work
    return meta, ""


def list_for(cwd: str):
    """Metas of existing worktrees for the repo containing `cwd`, newest first.
    Sidecars whose checkout is gone (user rm -rf'ed it) are pruned as they're seen."""
    top = repo_top(cwd)
    if not top:
        return []
    d = os.path.join(root(), _key(top))
    metas = []
    try:
        names = os.listdir(d)
    except OSError:
        return []
    for name in sorted(names, reverse=True):
        if not name.endswith(".json") or name.endswith(".tmp"):
            continue
        meta = _load_sidecar(os.path.join(d, name))
        if not meta:
            continue
        if os.path.isdir(meta["path"]):
            metas.append(meta)
        else:
            try:
                os.remove(os.path.join(d, name))
                _git(top, "worktree", "prune")
            except (OSError, subprocess.SubprocessError):
                pass
    return metas


def find(cwd: str, wtid: str):
    """The meta for `wtid` (any unambiguous id prefix/suffix works), or None."""
    hits = [m for m in list_for(cwd)
            if m["id"] == wtid or m["id"].startswith(wtid) or m["id"].endswith(wtid)]
    return hits[0] if len(hits) == 1 else None


# -- inspect / apply / remove -------------------------------------------------

def _stage_all(meta: dict) -> bool:
    """`git add -A` in the worktree so untracked files join the diff. Staging chad's
    own scratch branch is safe — nothing else ever commits or diffs its index."""
    try:
        return _git(meta["path"], "add", "-A").returncode == 0
    except (OSError, subprocess.SubprocessError):
        return False


def diff_stat(meta: dict):
    """(n_files, stat_text) for everything the session changed vs its base — working
    tree, staged, and any commits the session made on its branch, all in one diff.
    (0, "") when clean or on any failure (an unreadable worktree has nothing to apply)."""
    if not _stage_all(meta):
        return 0, ""
    try:
        r = _git(meta["path"], "diff", "--stat", meta["base"])
        if r.returncode != 0 or not r.stdout.strip():
            return 0, ""
        names = _git(meta["path"], "diff", "--name-only", meta["base"])
        n = len([ln for ln in names.stdout.splitlines() if ln.strip()])
        return n, r.stdout.rstrip()
    except (OSError, subprocess.SubprocessError):
        return 0, ""


def apply(meta: dict):
    """Apply the session's full diff (vs base) to the origin checkout's working tree.
    Returns (ok, message). No commit is made — the user reviews and commits with their
    own git, which is the whole supervision contract. Plain `git apply` first: it is
    atomic (errors before touching anything) and leaves the changes UNSTAGED, so
    `git diff` shows exactly what came back. Only when the origin drifted since base
    does the `--3way` fallback run — that one stages what it merges (and leaves
    conflict markers it can't), which the message says out loud. A failed apply KEEPS
    the worktree."""
    if not os.path.isdir(meta.get("origin_top", "")):
        return False, "origin checkout no longer exists"
    if not _stage_all(meta):
        return False, "could not stage worktree changes"
    try:
        patch = _git(meta["path"], "diff", "--binary", meta["base"])
        if patch.returncode != 0:
            return False, patch.stderr.strip() or "diff failed"
        if not patch.stdout.strip():
            return True, "no changes to apply"
        n = len([ln for ln in
                 _git(meta["path"], "diff", "--name-only", meta["base"]).stdout.splitlines()
                 if ln.strip()])
        r = _git(meta["origin_top"], "apply", "--binary", "--whitespace=nowarn",
                 input_text=patch.stdout)
        if r.returncode == 0:
            return True, f"applied {n} changed file(s) to {meta['origin_top']}"
        r = _git(meta["origin_top"], "apply", "--binary", "--3way",
                 "--whitespace=nowarn", input_text=patch.stdout)
        if r.returncode != 0:
            return False, (r.stderr.strip().splitlines() or ["git apply failed"])[-1]
        return True, (f"applied {n} changed file(s) to {meta['origin_top']} via 3-way "
                      "merge (the checkout had drifted; merged files are staged)")
    except (OSError, subprocess.SubprocessError) as e:
        return False, str(e)


def remove(meta: dict) -> str:
    """Delete the worktree checkout, its branch, and its sidecar. `--force` because a
    chad worktree is dirty by design (chad never commits); callers gate on diff_stat
    + user consent BEFORE calling this — remove() itself does not second-guess."""
    global _ACTIVE
    msgs = []
    try:
        r = _git(meta["origin_top"], "worktree", "remove", "--force", meta["path"])
        if r.returncode != 0:
            return f"could not remove worktree: {r.stderr.strip()}"
        b = _git(meta["origin_top"], "branch", "-D", meta["branch"])
        if b.returncode != 0:
            msgs.append(f"(branch {meta['branch']} not deleted: {b.stderr.strip()})")
        _git(meta["origin_top"], "worktree", "prune")
    except (OSError, subprocess.SubprocessError) as e:
        return f"could not remove worktree: {e}"
    try:
        os.remove(_sidecar_path(meta["origin_top"], meta["id"]))
    except OSError:
        pass
    if _ACTIVE and _ACTIVE.get("id") == meta.get("id"):
        _ACTIVE = None
    return " ".join(["removed worktree " + meta["id"]] + msgs)


def _ago(age_s: int) -> str:
    if age_s >= 86400:
        return f"{age_s // 86400}d ago"
    if age_s >= 3600:
        return f"{age_s // 3600}h ago"
    if age_s >= 60:
        return f"{age_s // 60}m ago"
    return "just now"


def describe(meta: dict) -> str:
    """One-liner for `chad worktree list` / the TUI: id, age, change count, path."""
    n, _stat = diff_stat(meta)
    when = _ago(max(0, int(time.time() - meta.get("created", 0))))
    change = f"{n} file(s) changed" if n else "no changes"
    return f"{meta['id']} · {when} · {change} · {meta['path']}"
