"""macOS Seatbelt confinement for yolo-mode bash commands (lever: yolo_seatbelt).

The destructive-command denylist in guardrails.py is pattern-matching, not a
boundary — it says so itself. This module is the boundary: in yolo mode (the one
mode where a shell command runs with no human in the loop), the spawned shell is
wrapped in `sandbox-exec` with a profile that denies file writes everywhere except
the workspace, temp dirs, caches, and chad's own state dir. Reads and network stay
open; the failure mode this closes is "an unreviewed command wrote outside the
project".

Only the *spawned shell child* is ever sandboxed. The chad process itself must
never enter the sandbox: the model runs in-process on Metal, and GPU access from
inside a Seatbelt profile is not something to gamble a session on. tool_bash asks
`wrap_argv()` for an argv at spawn time; everything else here is profile plumbing.

The active/workspace context is set by the agent immediately before each tool
dispatch and cleared after, so nested agents (a sub-agent's yolo loop inside a
normal-mode parent turn) and the `!cmd` passthrough each get exactly the
confinement of the agent/mode that is actually executing.
"""

import logging
import os
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Optional

from . import levers

log = logging.getLogger("chad")

SANDBOX_EXEC = "/usr/bin/sandbox-exec"

# One-shot enforcement probe. Seatbelt profiles don't nest: when chad itself runs
# inside a sandbox (CI, a harness container, another agent's shell tool),
# sandbox-exec fails to apply and would turn EVERY yolo bash call into an error.
# Probe once per process; on failure run unconfined (logged) rather than broken.
_probe_result: Optional[bool] = None

# The marker a denied write leaves in command output (EPERM strerror). Used by
# tool_bash to detect that the profile bit and to append an explanation the model
# can act on instead of a bare permission error.
DENIAL_MARKER = "Operation not permitted"
DENIAL_NOTE = ("\n[seatbelt: a write outside the workspace was denied — yolo mode "
               "confines file writes to the project directory, temp dirs, and "
               "caches. Work inside the project, or ask the user to run this "
               "command themselves.]")

_ctx: dict = {"active": False, "workspace": None}

# profile path per workspace realpath — the profile embeds absolute paths, so it is
# only reusable for the same workspace.
_profiles: dict = {}


def set_context(active: bool, workspace: Optional[str]) -> None:
    """Called by the agent around each tool dispatch. `active` is 'the executing
    agent is in yolo mode'; whether wrapping actually happens still depends on the
    lever and platform (see wrap_argv)."""
    _ctx["active"] = active
    _ctx["workspace"] = workspace


def available() -> bool:
    return sys.platform == "darwin" and os.path.exists(SANDBOX_EXEC)


def probe() -> bool:
    """Does a sandbox applied from this process actually ENFORCE? Cached per process.

    Running `sandbox-exec` successfully proves nothing about confinement: a profile
    that fails open (OS version drift, a malformed rule) would leave every yolo
    command reported as confined while writing anywhere it likes — a boundary that
    lies is worse than no boundary. So the probe exercises both directions of a
    real deny/allow profile against a throwaway directory pair: the allowed write
    must land AND the denied write must not. A denied write that lands fails the
    probe loudly; either failure means yolo bash runs unconfined (and says so)
    rather than confined-in-name-only."""
    global _probe_result
    if _probe_result is not None:
        return _probe_result
    if not available():
        _probe_result = False
        return False
    _probe_result = False
    leaked = False
    try:
        with tempfile.TemporaryDirectory(prefix="chad-sb-probe-") as tmp:
            # Seatbelt matches the KERNEL's view of a path: $TMPDIR is a symlink
            # (/var/folders -> /private/var/folders), and an allow rule written
            # against the symlinked spelling silently denies everything under it.
            root = os.path.realpath(tmp)
            allowed = os.path.join(root, "allowed")
            denied = os.path.join(root, "denied")
            os.mkdir(allowed)
            os.mkdir(denied)
            prof = ("(version 1)(allow default)(deny file-write*)"
                    f'(allow file-write* (subpath "{_scheme_str(allowed)}"))')
            cmd = (f"printf ok > {shlex.quote(os.path.join(allowed, 'w'))}; "
                   f"printf no > {shlex.quote(os.path.join(denied, 'w'))}")
            subprocess.run([SANDBOX_EXEC, "-p", prof, "/bin/sh", "-c", cmd],
                           capture_output=True, timeout=10, check=False)
            leaked = os.path.isfile(os.path.join(denied, "w"))
            _probe_result = (not leaked
                             and os.path.isfile(os.path.join(allowed, "w")))
    except (OSError, subprocess.SubprocessError):
        _probe_result = False
    if leaked:
        log.error("SEATBELT profile FAILED to enforce (a write that must be denied "
                  "landed) — refusing a confinement that would only be claimed; "
                  "yolo bash runs unconfined")
    elif not _probe_result:
        log.warning("SEATBELT unavailable here (nested sandbox or missing "
                    "support) — yolo bash runs unconfined")
    return _probe_result


def _scheme_str(path: str) -> str:
    """Escape a path for embedding in a Scheme string literal. Paths are embedded
    directly (no -D params): quotes and backslashes in a path must not be able to
    terminate the string and smuggle profile syntax."""
    return path.replace("\\", "\\\\").replace('"', '\\"')


def _worktree_gitdirs(workspace: str) -> list:
    """If the workspace is a `git worktree`, its `.git` is a FILE pointing at an
    external gitdir — and every git command writes there (index, locks) and to the
    shared common dir (objects, refs). Without these two carve-outs the sandbox
    kills all git activity in a worktree checkout."""
    dirs = []
    dotgit = os.path.join(workspace, ".git")
    if not os.path.isfile(dotgit):
        return dirs
    try:
        with open(dotgit, encoding="utf-8", errors="replace") as fh:
            m = re.match(r"gitdir:\s*(.+)", fh.read().strip())
        if not m:
            return dirs
        gitdir = os.path.normpath(os.path.join(workspace, m.group(1).strip()))
        if os.path.isdir(gitdir):
            dirs.append(gitdir)
            common = os.path.join(gitdir, "commondir")
            if os.path.isfile(common):
                with open(common, encoding="utf-8", errors="replace") as fh:
                    rel = fh.read().strip()
                commondir = os.path.normpath(os.path.join(gitdir, rel))
                if os.path.isdir(commondir):
                    dirs.append(commondir)
    except OSError:
        return dirs
    return dirs


def profile_text(workspace: str) -> str:
    """Deny-writes-outside-allowlist profile. `(allow default)` keeps reads,
    network, exec, signals open — v1 confines exactly one thing: where files can
    be written. The allowlist is every place real commands legitimately write:
    the workspace, POSIX and macOS temp roots, tool caches (uv/pip live under
    ~/Library/Caches on macOS, not ~/.cache), and chad's state dir.

    Two carve-backs ride AFTER the allow block (the last matching Seatbelt rule
    wins), because a writable root is not uniformly expendable:

    * `~/.chad/checkpoints` — the shadow-repo undo history. It sits under the
      allowed `~/.chad` subpath, so without its own deny a sandboxed command
      could delete the very snapshots a user would reach for after a bad turn.
      Only chad's own (never-sandboxed) process legitimately writes there, so
      this deny is unconditional and costs real commands nothing.
    * the workspace's git metadata (`.git`, and for a worktree checkout its
      external gitdir + shared common dir) — gated by the `seatbelt_protect_git`
      lever, its own opt-in tier rather than part of the base profile: it makes
      project history un-destroyable from inside the sandbox (`rm -rf .git`
      EPERMs), but any git command that writes — commit, add, checkout — fails
      with it. Sized statically against 91,910 real session commands: at most
      2.65% would EPERM (an upper bound — a clone/init into a subdirectory makes
      a NESTED .git, which stays writable), concentrated in 14% of sessions.
      With the tier off, worktree gitdirs stay on the ALLOW list, since every
      git command writes through them."""
    home = os.path.expanduser("~")
    ws = os.path.realpath(workspace)
    protect_git = levers.enabled("seatbelt_protect_git")
    gitdirs = _worktree_gitdirs(ws)
    write_paths = [ws]
    if workspace != ws:
        write_paths.append(workspace)
    if not protect_git:
        write_paths += gitdirs
    write_paths += [
        "/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp",
        "/private/var/folders",              # $TMPDIR lives here on macOS
        os.path.join(home, ".chad"),
        os.path.join(home, ".cache"),
        os.path.join(home, "Library", "Caches"),
        os.path.join(home, ".local", "share", "uv"),
        os.path.join(home, ".npm"),
    ]
    lines = ["(version 1)", "(allow default)", "(deny file-write*)",
             "(allow file-write*"]
    for p in write_paths:
        lines.append(f'    (subpath "{_scheme_str(p)}")')
    lines.append(f'    (literal "{_scheme_str(os.path.join(home, ".gitconfig"))}")')
    for dev in ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/fd"):
        lines.append(f'    (literal "{dev}")')
    lines.append('    (regex #"^/dev/tty")')
    lines.append(")")
    deny_paths = [os.path.join(home, ".chad", "checkpoints")]
    if protect_git:
        deny_paths.append(os.path.join(ws, ".git"))
        if workspace != ws:
            deny_paths.append(os.path.join(workspace, ".git"))
        deny_paths += gitdirs
    lines.append("(deny file-write*")
    for p in deny_paths:
        lines.append(f'    (subpath "{_scheme_str(p)}")')
    lines.append(")")
    return "\n".join(lines) + "\n"


def _profile_path(workspace: str) -> str:
    # Keyed on the git-protection tier as well as the workspace: the profile body
    # differs, and lever state can change between calls within one process.
    ws = os.path.realpath(workspace)
    key = (ws, levers.enabled("seatbelt_protect_git"))
    path = _profiles.get(key)
    if path and os.path.exists(path):
        return path
    fd, path = tempfile.mkstemp(prefix="chad-seatbelt-", suffix=".sb")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(profile_text(ws))
    _profiles[key] = path
    log.info("SEATBELT profile for %s -> %s", ws, path)
    return path


def wrap_argv(command: str) -> Optional[list]:
    """The argv to spawn `command` sandboxed, or None to run it unconfined.
    None whenever the lever is off, the executing agent is not in yolo mode, or
    the platform can't do Seatbelt — tool_bash falls through to its normal
    shell=True spawn, byte-identical behavior to before this module existed."""
    if not levers.enabled("yolo_seatbelt"):
        return None
    if not _ctx["active"] or not probe():
        return None
    workspace = _ctx["workspace"] or os.getcwd()
    try:
        profile = _profile_path(workspace)
    except OSError as e:
        # No profile means no confinement; in yolo that is the unsafe direction,
        # but killing every bash call is worse — log loudly and run unconfined.
        log.warning("SEATBELT profile write failed (%s) — running unconfined", e)
        return None
    return [SANDBOX_EXEC, "-f", profile, "/bin/sh", "-c", command]
