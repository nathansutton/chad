"""Shadow-git checkpoints for file edits.

Auto-approved edits are justified as "a diff you can read and revert" — this
module supplies the revert. Before each file-mutating tool lands, the workspace
is committed to a *shadow* git repository under ~/.chad/checkpoints/<hash>. The
user's own .git is never opened, never written, never required to exist: the
shadow repo has its own GIT_DIR and treats the workspace purely as a work tree.
/undo and /restore in the TUI check files back out of it.

Failure policy: never raise. An edit must not die because the checkpoint
machinery hiccuped — snapshot() returns None on any failure and the tool call
proceeds unprotected (logged).

Restore policy (deliberately conservative): `git restore --source=<ref>` puts
every snapshotted file back to its snapshotted content. Files *created after*
the snapshot are left in place — deleting files is exactly the blast radius this
lever exists to contain, so the restore path never does it.
"""

import hashlib
import logging
import os
import shutil
import subprocess
import time
from typing import Optional

from . import config

log = logging.getLogger("chad")

_GIT_TIMEOUT_S = 120  # first snapshot of a big tree is seconds; never hang a turn
_MAX_AGE_S = 30 * 24 * 3600  # a shadow with no snapshot for this long is swept

# Written to the shadow repo's info/exclude, which composes with any workspace
# .gitignore rather than replacing it: junk that snapshots must not swallow when the
# workspace has no .gitignore of its own (a non-git project), then secret-shaped files
# that must never be copied into a history that outlives the workspace.
_DEFAULT_EXCLUDES = (".venv/\nvenv/\nnode_modules/\n__pycache__/\n*.pyc\n"
                     ".mypy_cache/\n.ruff_cache/\n.pytest_cache/\n.DS_Store\n"
                     ".env\n.env.*\n*.pem\n*.key\nid_rsa*\nid_ed25519*\n*.p12\n*.pfx\n")


def _history_root() -> str:
    # CHAD_CHECKPOINT_DIR: test/e2e override so suites never write real home state.
    # NOT ~/.chad/history — that name is taken by the TUI's prompt-history FILE.
    env = config.env_str("CHAD_CHECKPOINT_DIR")
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


def _sync_excludes(workspace: str, sd: str, excludes: str) -> None:
    """Bring the shadow's info/exclude up to `excludes`, so a shadow created before a
    pattern existed gets it on its next snapshot. An exclude only keeps *untracked*
    files out, so anything the shadow already tracks that now matches is dropped from
    its index too; the workspace file itself is never touched."""
    path = os.path.join(sd, "info", "exclude")
    try:
        with open(path, encoding="utf-8") as fh:
            if fh.read() == excludes:
                return
    except OSError:
        pass
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(excludes)
    tracked = _git(workspace, "ls-files", "-z", "--cached", "--ignored", "--exclude-standard")
    paths = [p for p in tracked.stdout.split("\0") if p]
    if tracked.returncode == 0 and paths:
        r = _git(workspace, "--literal-pathspecs", "rm", "-q", "--cached", "--", *paths)
        if r.returncode != 0:
            log.warning("CHECKPOINT untracking excluded files failed: %s", r.stderr.strip())


def _ensure_shadow(workspace: str, excludes: str) -> bool:
    sd = shadow_dir(workspace)
    try:
        if not os.path.isdir(sd):
            # Private like the session store: a snapshot is the whole workspace. The
            # subdirectories git creates keep the umask, but nothing under a 0700 sd
            # is reachable.
            os.makedirs(os.path.dirname(sd), mode=0o700, exist_ok=True)
            r = subprocess.run(["git", "init", "-q", "--bare", sd],
                               capture_output=True, text=True,
                               timeout=_GIT_TIMEOUT_S, check=False)
            if r.returncode != 0:
                log.warning("CHECKPOINT shadow init failed: %s", r.stderr.strip())
                return False
            os.chmod(sd, 0o700)
        _sync_excludes(workspace, sd, excludes)
        return True
    except (OSError, subprocess.SubprocessError) as e:
        log.warning("CHECKPOINT shadow init failed: %s", e)
        return False


def _sweep_stale(root: str, keep: str, max_age_s: float = _MAX_AGE_S) -> None:
    """Delete shadows with no snapshot for `max_age_s`: workspaces long finished or
    deleted, whose history would otherwise be kept forever. Best-effort, never raises,
    never touches `keep` (the shadow being written)."""
    try:
        deadline = time.time() - max_age_s
        for name in os.listdir(root):
            sd = os.path.join(root, name, "shadow.git")
            try:
                if sd != keep and os.path.isdir(sd) and os.path.getmtime(sd) < deadline:
                    shutil.rmtree(sd, ignore_errors=True)
                    os.rmdir(os.path.dirname(sd))  # the per-workspace dir, now empty
            except OSError:
                pass
    except OSError:
        pass


class Snapshotter:
    """Takes checkpoints whose shadows carry `excludes`. The store lock-down and the
    stale sweep run on a snapshotter's first snapshot only; chad keeps one for the
    whole process, so they run once per process."""

    def __init__(self, excludes: str = _DEFAULT_EXCLUDES) -> None:
        self.excludes = excludes
        self._swept = False

    def snapshot(self, workspace: str, label: str) -> Optional[str]:
        """Commit the workspace state to the shadow repo; return the short hash, or
        None if no checkpoint exists after the attempt. An unchanged tree is not a
        failure — the previous snapshot already covers it, so its hash is returned."""
        try:
            if not _ensure_shadow(workspace, self.excludes):
                return None
            sd = shadow_dir(workspace)
            if not self._swept:
                self._swept = True
                root = _history_root()
                try:
                    # makedirs applies its mode to the leaf only, and older stores are 0755
                    os.chmod(root, 0o700)
                except OSError:
                    pass
                _sweep_stale(root, keep=sd)
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
            # The sweep reads sd's mtime, which git does not reliably bump (most of its
            # writes land in subdirectories), so mark the shadow used explicitly.
            try:
                os.utime(sd)
            except OSError:
                pass
            return head.stdout.strip()
        except (OSError, subprocess.SubprocessError) as e:
            log.warning("CHECKPOINT snapshot failed: %s", e)
            return None


_SNAPSHOTTER = Snapshotter()


def snapshot(workspace: str, label: str) -> Optional[str]:
    """Checkpoint the workspace with the process's snapshotter (see
    Snapshotter.snapshot): the short hash, or None when no checkpoint exists."""
    return _SNAPSHOTTER.snapshot(workspace, label)


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
    """Check snapshotted files back out at `ref`. The state about to be overwritten
    is snapshotted first, so a restore can itself be reverted. Returns a
    human-readable one-liner (also used verbatim by the TUI)."""
    try:
        ok = _git(workspace, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        if ok.returncode != 0:
            have = snapshots(workspace, limit=1)
            return (f"no checkpoint named {ref!r}" if have else
                    "no checkpoints exist for this workspace yet — snapshots are "
                    "taken before file edits")
        # Resolve the target BEFORE snapshotting: the snapshot below moves HEAD, and
        # `/undo` means "the checkpoint that was newest when I asked".
        target = ok.stdout.strip()
        short = _git(workspace, "rev-parse", "--short", target).stdout.strip() or ref
        label = _git(workspace, "log", "-1", "--format=%s", target).stdout.strip()
        changed = _git(workspace, "diff", "--name-only", target)
        n = len([ln for ln in changed.stdout.splitlines() if ln.strip()])
        if n == 0:
            return (f"nothing to undo — the files already match checkpoint {short} "
                    f"({label}).")
        # What is about to be overwritten may be the agent's edit or the user's own
        # work since the last snapshot. Either way it is only recoverable if saved now.
        saved = snapshot(workspace, f"before restore to {short}")
        r = _git(workspace, "restore", "--worktree", f"--source={target}", "--", ".")
        if r.returncode != 0:
            return f"restore failed: {r.stderr.strip() or 'unknown git error'}"
        back = (f" What was there is saved as {saved}: /restore {saved} brings it back."
                if saved else " (The previous state could not be saved first.)")
        return (f"restored {n} file(s) to checkpoint {short} ({label}).{back} Files "
                f"created since that snapshot were left in place.")
    except (OSError, subprocess.SubprocessError) as e:
        return f"restore failed: {e}"
