"""The confinement both arms run under, owned by the runner rather than by either tree.

WHERE THE SANDBOX GOES, AND WHY NOT AROUND THE BLOCK
----------------------------------------------------
The obvious design is to wrap the whole solving block in one `sandbox-exec` and be done.
That does not work here, and it fails in the worst way: the model runs in-process on
Metal, and a process inside a Seatbelt profile does not get GPU access — the load hangs
rather than erroring. chad's own seatbelt module carries the same rule for the same
reason ("the chad process itself must never enter the sandbox"), and it is not a rule
worth re-testing on a machine with two GPU panics on its record.

So confinement is applied one level down, at the only place a trial can reach the
filesystem or the network in the first place: the shell child that the `bash` tool
spawns. Seatbelt confinement is inherited and cannot be dropped, so wrapping that first
shell confines everything under it — a background process, a `make`, a second shell.
The solver's own Python stays outside, where Metal works.

The injection point is `PATH`, not a chad flag. Both trees resolve their shell once, at
import, with `shutil.which("bash")`; a directory placed first on `PATH` containing a
`bash` that re-execs under `sandbox-exec` therefore confines both arms identically,
through code neither tree knows about. That matters more than convenience: a harness
seam added to one tree and ported to the other is exactly the kind of difference this
measurement is supposed to be free of.

The shim fails closed. If the profile variable is unset it refuses to run rather than
falling through to an unconfined shell, because a trial that quietly ran unconfined
would be indistinguishable in the results from one that did not.

chad's inner `yolo_seatbelt` must be OFF (`CHAD_NO_SEATBELT=1`): Seatbelt profiles do
not nest, and the inner one would fail to apply and turn every command into an error.
The runner asserts this rather than assuming it.

WHAT THE PROFILE SAYS
---------------------
Stricter than chad's, which confines writes only:

* **No network.** Loopback stays open so a test that binds a local port still works;
  everything else, DNS included, gets EPERM immediately rather than hanging.
* **Writes** only into the trial's own workspace and the temp roots. The shared
  workbench environments are therefore read-only — one trial cannot leave state that
  another trial, or the other arm, would inherit.
* **Reads of `$HOME` denied**, then re-allowed for the workspace, the environment root
  and uv's managed interpreters. This is the leak boundary that matters on a development
  machine: a newer django or sphinx checkout anywhere under `$HOME` — including this
  repository's own — is unreadable from inside a trial.
"""
from __future__ import annotations

import dataclasses
import os
import stat
import subprocess

SANDBOX_EXEC = "/usr/bin/sandbox-exec"
# Read by the shim at spawn time, so one shim serves a whole block while the profile
# stays per-trial (it names the trial's workspace).
PROFILE_ENV = "SWEBENCH_SB_PROFILE"
SHIM_MISSING_PROFILE_EXIT = 97

_TEMP_ROOTS = ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp",
               "/private/var/folders")


def available() -> bool:
    return os.path.exists(SANDBOX_EXEC)


def _scheme(path: str) -> str:
    """Escape a path for a Scheme string literal, so it cannot terminate the string
    and smuggle profile syntax."""
    return path.replace("\\", "\\\\").replace('"', '\\"')


def _subpaths(paths: tuple[str, ...]) -> list[str]:
    # Seatbelt matches the kernel's view of a path: $TMPDIR is a symlink into
    # /private/var/folders, and a rule written against the symlinked spelling silently
    # matches nothing. Every path is realpath'd, and the unresolved spelling is kept
    # too when they differ.
    out: list[str] = []
    for p in paths:
        for q in {p, os.path.realpath(p)}:
            if q:
                out.append(f'    (subpath "{_scheme(q)}")')
    return sorted(set(out))


def profile_text(workspace: str, read_roots: tuple[str, ...]) -> str:
    """The SBPL profile for one trial. `read_roots` are the paths under `$HOME` a
    trial legitimately reads: the environment root, and uv's interpreter store."""
    home = os.path.expanduser("~")
    lines = ["(version 1)", "(allow default)", "",
             "; network: loopback only",
             "(deny network*)",
             '(allow network-bind (local ip "localhost:*"))',
             '(allow network-outbound (remote ip "localhost:*"))', "",
             "; writes: the trial workspace and temp, nothing else",
             "(deny file-write*)",
             "(allow file-write*"]
    lines += _subpaths((workspace, *_TEMP_ROOTS))
    for dev in ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/fd", "/dev/urandom"):
        lines.append(f'    (literal "{dev}")')
    lines.append('    (regex #"^/dev/tty")')
    lines.append(")")
    lines += ["", "; reads: $HOME is closed, then reopened where the trial needs it",
              "(deny file-read*", f'    (subpath "{_scheme(home)}")', ")",
              "(allow file-read*"]
    lines += _subpaths((workspace, *read_roots, *_TEMP_ROOTS))
    lines.append(")")
    # Metadata, everywhere. `stat` on a path is not a leak — it says a name exists, not
    # what is in it — and denying it breaks tools that have to walk a path to do their
    # job: pytest searches upward from the working directory for its rootdir, and a
    # PermissionError on a parent directory aborts collection before a single test runs.
    # File contents stay denied by the block above; `cat` on the fix next door still
    # fails, and so does listing a directory, which needs its contents and not its
    # metadata. This is the last rule, so it wins for metadata alone.
    lines += ["", "; stat is not a leak; reading is", "(allow file-read-metadata)"]
    return "\n".join(lines) + "\n"


def write_profile(path: str, workspace: str, read_roots: tuple[str, ...]) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(profile_text(workspace, read_roots))
    return path


def write_shim(bin_dir: str) -> str:
    """A `bash` that re-execs itself under the profile named by the environment.

    `exec` rather than a subshell so signals and the process group behave exactly as
    they would without the shim — the bash tool kills the whole group on a timeout, and
    an extra layer would leave the real shell behind.
    """
    os.makedirs(bin_dir, exist_ok=True)
    path = os.path.join(bin_dir, "bash")
    script = (
        "#!/bin/sh\n"
        "# Written by benchmarks/swebench/sandbox.py. Both harness trees resolve their\n"
        "# shell with which(bash), so this is where a trial's commands get confined.\n"
        f'if [ -z "${PROFILE_ENV}" ]; then\n'
        f'  echo "swebench sandbox: {PROFILE_ENV} is unset — refusing to run '
        f'unconfined" >&2\n'
        f"  exit {SHIM_MISSING_PROFILE_EXIT}\n"
        "fi\n"
        f'exec {SANDBOX_EXEC} -f "${PROFILE_ENV}" /bin/bash "$@"\n'
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(script)
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


def run_confined(profile: str, command: str, cwd: str, timeout: int = 60,
                 env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    """Run one command under a profile — what the canary uses, and what the shim does."""
    return subprocess.run([SANDBOX_EXEC, "-f", profile, "/bin/bash", "-c", command],
                          cwd=cwd, capture_output=True, text=True, timeout=timeout,
                          check=False, env=env)


@dataclasses.dataclass(frozen=True)
class Probe:
    name: str
    command: str
    must_succeed: bool
    detail: str


@dataclasses.dataclass(frozen=True)
class ProbeResult:
    probe: Probe
    ok: bool
    output: str


def _assert_outside(path: str, workspace: str, label: str) -> None:
    """The canary's own targets must actually be outside the allowed set.

    A planted secret placed in `/tmp` is readable by design and the probe would report a
    leak that is really a mistake in the test — which is worse than no probe at all,
    because it trains the reader to ignore the canary. Both targets must sit outside the
    workspace and outside every temp root.
    """
    real = os.path.realpath(path)
    if real.startswith(os.path.realpath(workspace) + os.sep):
        raise ValueError(f"canary {label} {path} is inside the workspace")
    for t in _TEMP_ROOTS:
        root = os.path.realpath(t)
        if real == root or real.startswith(root + os.sep):
            raise ValueError(f"canary {label} {path} is under the writable temp root {t}")


def probes(workspace: str, secret_path: str, outside_write: str) -> tuple[Probe, ...]:
    """The leak canary.

    Both directions are checked. A profile that denies everything is not a boundary, it
    is a broken shell, and it would fail every trial in a way that looks like the model
    being bad at its job — so the allowed operations must also be proven to work.
    """
    _assert_outside(secret_path, workspace, "secret")
    _assert_outside(outside_write, workspace, "write target")
    home = os.path.expanduser("~")
    return (
        Probe("network-dns", "curl -sS -m 8 -o /dev/null https://pypi.org/simple/", False,
              "a trial must not be able to reach the network"),
        Probe("network-ip", "curl -sS -m 8 -o /dev/null http://93.184.216.34/", False,
              "denial must not depend on DNS resolution failing"),
        Probe("read-home", f"ls -a {home!r} >/dev/null", False,
              "$HOME must be unreadable"),
        Probe("read-planted-fix", f"cat {secret_path!r}", False,
              "a file outside the workspace must be unreadable even by exact path"),
        Probe("write-outside", f"touch {outside_write!r}", False,
              "writes outside the workspace must be denied"),
        Probe("read-workspace", "cat README.rst >/dev/null 2>&1 || ls >/dev/null", True,
              "the workspace itself must be readable"),
        Probe("write-workspace", "touch .swb-canary && rm -f .swb-canary", True,
              "the workspace itself must be writable"),
        Probe("write-temp", "t=$(mktemp) && printf ok > \"$t\" && rm -f \"$t\"", True,
              "temp must be writable — pytest and git both need it"),
        # The system interpreter by absolute path: `python3` from PATH may resolve to a
        # virtual environment under $HOME, which is unreadable here by design, and the
        # probe would then be reporting on PATH resolution instead of on loopback.
        Probe("loopback",
              "/usr/bin/python3 -c \"import socket;s=socket.socket();"
              "s.bind(('127.0.0.1',0))\"",
              True, "a test that binds a local port must still work"),
    )


def run_probes(profile: str, workspace: str, secret_path: str,
               outside_write: str) -> list[ProbeResult]:
    results: list[ProbeResult] = []
    for p in probes(workspace, secret_path, outside_write):
        try:
            out = run_confined(profile, p.command, cwd=workspace, timeout=30)
            succeeded = out.returncode == 0
            text = (out.stdout + out.stderr).strip()
        except subprocess.TimeoutExpired:
            # A hang is a failure in either direction: a denied operation must EPERM
            # promptly, and an allowed one must complete.
            succeeded, text = False, "timed out"
        results.append(ProbeResult(p, succeeded is p.must_succeed, text[:400]))
    return results


def format_probes(results: list[ProbeResult]) -> str:
    lines = []
    for r in results:
        want = "allowed" if r.probe.must_succeed else "denied"
        lines.append(f"  [{'ok ' if r.ok else 'LEAK' if not r.probe.must_succeed else 'FAIL'}] "
                     f"{r.probe.name:<18} must be {want:<7} — {r.probe.detail}")
        if not r.ok and r.output:
            lines.append(f"         {r.output.splitlines()[0][:160]}")
    return "\n".join(lines)
