"""Shadow-git checkpoints for file edits (lever: edit_checkpoint).

Auto-approved edits are justified as "a diff you can read and revert" — this
module supplies the revert. Before each file-mutating tool lands, the workspace
is committed to a *shadow* git repository under ~/.chad/checkpoints/<hash>. The
user's own .git is never opened, never written, never required to exist: the
shadow repo has its own GIT_DIR and treats the workspace purely as a work tree.
/undo and /restore in the TUI check files back out of it.

Failure policy: never raise. An edit must not die because the checkpoint
machinery hiccuped — snapshot() returns None on any failure and the tool call
proceeds unprotected (logged, and visible in `chad levers` firing counts as the
absence of a fire).

Restore policy (deliberately conservative): `git restore --source=<ref>` puts
every snapshotted file back to its snapshotted content. Files *created after*
the snapshot are left in place — deleting files is exactly the blast radius this
lever exists to contain, so the restore path never does it.
"""

import hashlib
import logging
import os
import subprocess
from typing import Optional

log = logging.getLogger("chad")

_GIT_TIMEOUT_S = 120  # first snapshot of a big tree is seconds; never hang a turn

# Junk that snapshots must not swallow when the workspace has no .gitignore of its
# own (a non-git project): written to the shadow repo's info/exclude, which
# composes with any workspace .gitignore rather than replacing it.
_DEFAULT_EXCLUDES = (".venv/\nvenv/\nnode_modules/\n__pycache__/\n*.pyc\n"
                     ".mypy_cache/\n.ruff_cache/\n.pytest_cache/\n.DS_Store\n")


def _history_root() -> str:
    # CHAD_CHECKPOINT_DIR: test/e2e override so suites never write real home state.
    # NOT ~/.chad/history — that name is taken by the TUI's prompt-history FILE.
    env = os.environ.get("CHAD_CHECKPOINT_DIR")
    if env:
        return env
    return os.path.join(os.path.expanduser("~"), ".chad", "checkpoints")


def shadow_dir(workspace: str) -> str:
    ws = os.path.realpath(workspace)
    tag = hashlib.sha1(ws.encode()).hexdigest()[:16]
    return os.path.join(_history_root(), tag, "shadow.git")


def _git(workspace: str, *args: str) -> subprocess.CompletedProcess:
    """Run git against the shadow repo with the workspace as work tree. Identity
    and signing come from -c flags so the user's global config is never consulted
    for authorship and never mutated."""
    cmd = ["git", "--git-dir", shadow_dir(workspace), "--work-tree",
           os.path.realpath(workspace),
           "-c", "user.email=chad@localhost", "-c", "user.name=chad",
           "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null",
           *args]
    return subprocess.run(cmd, capture_output=True, text=True,
                          timeout=_GIT_TIMEOUT_S, check=False)


def _ensure_shadow(workspace: str) -> bool:
    sd = shadow_dir(workspace)
    if os.path.isdir(sd):
        return True
    try:
        os.makedirs(os.path.dirname(sd), exist_ok=True)
        r = subprocess.run(["git", "init", "-q", "--bare", sd],
                           capture_output=True, text=True,
                           timeout=_GIT_TIMEOUT_S, check=False)
        if r.returncode != 0:
            log.warning("CHECKPOINT shadow init failed: %s", r.stderr.strip())
            return False
        info = os.path.join(sd, "info")
        os.makedirs(info, exist_ok=True)
        with open(os.path.join(info, "exclude"), "w", encoding="utf-8") as fh:
            fh.write(_DEFAULT_EXCLUDES)
        return True
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("CHECKPOINT shadow init failed: %s", e)
        return False


def snapshot(workspace: str, label: str) -> Optional[str]:
    """Commit the workspace state to the shadow repo; return the short hash, or
    None if no checkpoint exists after the attempt. An unchanged tree is not a
    failure — the previous snapshot already covers it, so its hash is returned."""
    try:
        if not _ensure_shadow(workspace):
            return None
        add = _git(workspace, "add", "-A")
        if add.returncode != 0:
            log.warning("CHECKPOINT add failed: %s", add.stderr.strip())
            return None
        commit = _git(workspace, "commit", "-q", "-m", label)
        head = _git(workspace, "rev-parse", "--short", "HEAD")
        if head.returncode != 0:
            # Nothing staged AND no prior snapshot — an empty workspace's first
            # checkpoint. Record the empty state anyway so the timeline exists.
            commit = _git(workspace, "commit", "-q", "--allow-empty", "-m", label)
            head = _git(workspace, "rev-parse", "--short", "HEAD")
            if head.returncode != 0:
                log.warning("CHECKPOINT commit failed: %s", commit.stderr.strip())
                return None
        return head.stdout.strip()
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("CHECKPOINT snapshot failed: %s", e)
        return None


def snapshots(workspace: str, limit: int = 10) -> list:
    """Newest-first [(short_hash, iso_time, label)] — empty on any failure."""
    try:
        r = _git(workspace, "log", f"-{limit}", "--format=%h%x09%cI%x09%s")
        if r.returncode != 0:
            return []
        rows = []
        for line in r.stdout.splitlines():
            parts = line.split("\t", 2)
            if len(parts) == 3:
                rows.append((parts[0], parts[1], parts[2]))
        return rows
    except (OSError, subprocess.SubprocessError):
        return []


def restore(workspace: str, ref: str = "HEAD") -> str:
    """Check snapshotted files back out at `ref`. Returns a human-readable
    one-liner (also used verbatim by the TUI)."""
    try:
        ok = _git(workspace, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        if ok.returncode != 0:
            have = snapshots(workspace, limit=1)
            return (f"no checkpoint named {ref!r}" if have else
                    "no checkpoints exist for this workspace yet — snapshots are "
                    "taken before file edits when the edit_checkpoint lever is on")
        changed = _git(workspace, "diff", "--name-only", ref)
        n = len([ln for ln in changed.stdout.splitlines() if ln.strip()])
        r = _git(workspace, "restore", "--worktree", f"--source={ref}", "--", ".")
        if r.returncode != 0:
            return f"restore failed: {r.stderr.strip() or 'unknown git error'}"
        label = _git(workspace, "log", "-1", "--format=%s", ref).stdout.strip()
        return (f"restored {n} file(s) to checkpoint {ref} ({label}). Files created "
                f"since that snapshot were left in place.")
    except (OSError, subprocess.SubprocessError) as e:
        return f"restore failed: {e}"
