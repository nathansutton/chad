"""Any coding agent with a headless command line, driven by one frozen `CliSpec`: an
adapter is data, not code. The entries live in `harnesses.py`.

ISOLATION
---------
The matrix grid this replaces wrote each harness's provider config into the maintainer's
real dotfiles, backing them up first, and let nine auto-approving agents write wherever
the user could. A trial here runs

* in a workspace outside this checkout and outside the home directory. Most harnesses
  read `AGENTS.md` or `CLAUDE.md` from the directories above the one they start in —
  some up to the git root, pi from every ancestor — and from the kit's own `_work/` that
  is this repository's contributor guide, in a foreign agent's prompt. chad reads only
  its working directory's, which is why its in-process arm can stay where it is.
* with `HOME`, the XDG directories and `TMPDIR` in a throwaway home that `prepare()`
  writes the spec's config files into, and an environment rebuilt from a short
  allowlist, so no API key, token or `CHAD_*` variable of the launching shell reaches
  the agent.
* under `sandbox-exec`, with file writes denied everywhere except the workspace, the
  home and the system temp directories. The temp directories are open for the reason
  chad's own bash sandbox opens them: a model that writes a scratch file to /tmp has to
  succeed under every arm, or a paired comparison measures the sandbox.

Reads and the network stay open, as in chad's sandbox: the agent needs the toolchains and
the server. A harness that cannot work this way — it wants the login keychain, a daemon
outside its process group, a global package cache — is marked `unsupported` with the
reason; the profile is never loosened for one arm.

A trial ends when the agent exits or at the wall cap, and either way its whole process
group is killed: a helper that outlives its trial keeps generating against the next
trial's numbers.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from string import Template

import catalog
import workspace
from catalog import JsonValue, is_number, is_object

from chad import seatbelt
from harness import AtifDoc, Solved, Trial

# Outside this checkout and outside the home directory: see ISOLATION.
ROOT = os.path.join(os.path.realpath(tempfile.gettempdir()), "chad-polyglot")
REPO_ROOT = os.path.dirname(os.path.dirname(catalog.ROOT))
API_KEY = "sk-local-dummy"       # local servers ignore it; clients insist on one
VERSION_TIMEOUT_S = 60
SETUP_TIMEOUT_S = 120
OUTPUT_TAIL = 600

# What a trial keeps of the launching shell: locale, identity, and where the toolchains
# are. Everything else stays behind.
_PASSED = ("PATH", "LANG", "LC_ALL", "LC_CTYPE", "USER", "LOGNAME", "SHELL", "TZ",
           "JAVA_HOME", "SDKROOT", "DEVELOPER_DIR", "CARGO_NET_OFFLINE")
_TEMP_ROOTS = ("/tmp", "/private/tmp", "/var/tmp", "/private/var/tmp", "/private/var/folders")
# `0.80.3`, `v0.92.0`, `0.1.1-rc.2`, and the `2.4.6` of "version 2.4.6."
_VERSION = re.compile(r"\d+(?:\.\d+)+(?:-[0-9A-Za-z.]*[0-9A-Za-z])?")


class HarnessError(RuntimeError):
    """An arm that cannot run as specified: unsupported, not installed, not the locked
    version, or unconfinable from this process."""


@dataclass(frozen=True)
class CliSpec:
    """One harness, as data. `argv`, `env`, `files` and `setup` are `string.Template`s
    (`$$` is a literal `$`) over `${prompt}`, `${workspace}`, `${home}`, `${origin}`,
    `${model}`, `${context}`, `${api_key}`, `${tokenizer}` and `${inner_cap_s}` — the wall
    cap less a margin, for a harness whose own timeout should fire before ours. Config
    files are rendered before the workspace exists, so they see only the home and the
    server. `version` is a whole command, over `${binary}` and `${entry_dir}` (where the
    file the binary resolves to lives), for a harness with no `--version` of its own."""

    name: str
    binary: str                   # looked up on the trial PATH
    install: str                  # how to install it; `${version}` comes from the lock
    argv: tuple[str, ...] = ()    # after the binary
    env: Mapping[str, str] = field(default_factory=dict)
    files: Mapping[str, str] = field(default_factory=dict)   # {path under home: text}
    setup: tuple[tuple[str, ...], ...] = ()   # `<binary> <args>`, run by prepare()
    trajectory: str = ""          # a native ATIF document the agent writes, under home
    version: tuple[str, ...] = ("${binary}", "--version")
    locked: bool = True           # False: the binary is this checkout's own
    unsupported: str = ""         # why the arm cannot run isolated; set, it is refused


@dataclass(frozen=True)
class Endpoint:
    """The server every CLI arm of a block talks to."""

    origin: str                   # scheme://host:port, no path
    model: str                    # the id the server answers to
    context: int                  # its window, for configs that ask for one
    tokenizer: str                # the served model's tokenizer, as a local directory


@dataclass(frozen=True)
class Pin:
    """One entry of `harnesses.lock`: the version a result was measured with."""

    version: str
    sha256: str                   # of the resolved entry point, where it was locked
    install: str


def server_context(origin: str, timeout_s: float = 5.0) -> int:
    """The window the server at `origin` enforces, from llama-server's `/props` — the
    same number chad's llama backend sizes itself from. Doubles as the check that a
    server is up before a block spends its first trial finding out."""
    try:
        with urllib.request.urlopen(f"{origin}/props", timeout=timeout_s) as r:
            doc: JsonValue = json.loads(r.read().decode())
    except (OSError, ValueError) as e:
        raise HarnessError(f"no llama-server answers at {origin} ({e})") from e
    settings = doc.get("default_generation_settings") if is_object(doc) else None
    n_ctx = settings.get("n_ctx") if is_object(settings) else None
    if not is_number(n_ctx):
        raise HarnessError(f"{origin}/props reports no n_ctx: is it a llama-server?")
    return int(n_ctx)


def profile(*writable: str) -> str:
    """Deny every file write except under `writable`, the system temp directories and
    the terminal devices. chad's bash sandbox (`chad.seatbelt`) with the real home's
    carve-outs taken away: the same escaping, the same device list. Paths are resolved
    first because Seatbelt matches the kernel's view of a path, and /var is a symlink."""
    lines = ["(version 1)", "(allow default)", "(deny file-write*)", "(allow file-write*"]
    for path in [*(os.path.realpath(p) for p in writable), *_TEMP_ROOTS]:
        lines.append(f'    (subpath "{seatbelt._scheme_str(path)}")')
    for dev in ("/dev/null", "/dev/stdout", "/dev/stderr", "/dev/fd"):
        lines.append(f'    (literal "{dev}")')
    lines += ['    (regex #"^/dev/tty")', ")"]
    return "\n".join(lines) + "\n"


def entry_sha256(exe: str) -> str:
    """The sha256 of what `exe` resolves to: the file that actually runs."""
    with open(os.path.realpath(exe), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _kill_group(proc: subprocess.Popen[bytes]) -> None:
    """The agent and everything it started, then the agent reaped. Watches the group
    rather than the leader, which may already be gone while a helper it spawned is not."""
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(proc.pid, sig)
        except (ProcessLookupError, PermissionError):
            break
        if _emptied(proc, seconds=5):
            break
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        pass                           # unkillable; its exit code stays unknown


def _emptied(proc: subprocess.Popen[bytes], seconds: float) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        proc.poll()                    # reap the leader as soon as it dies
        try:
            os.killpg(proc.pid, 0)
        except (ProcessLookupError, PermissionError):
            return True
        time.sleep(0.1)
    return False


def _tail(path: str) -> str:
    with open(path, "rb") as f:
        f.seek(max(0, os.path.getsize(path) - OUTPUT_TAIL))
        return f.read().decode("utf-8", "replace")


def _fresh(home: str) -> None:
    shutil.rmtree(home, ignore_errors=True)
    os.makedirs(os.path.join(home, "tmp"))


class CliHarness:
    def __init__(self, spec: CliSpec, endpoint: Endpoint, pin: Pin | None,
                 environ: Mapping[str, str] = os.environ,
                 sandbox_ok: Callable[[], bool] = seatbelt.probe, root: str = ROOT):
        self.name, self.root = spec.name, root
        self.spec, self.endpoint, self.pin = spec, endpoint, pin
        self._environ = environ
        self._sandbox_ok = sandbox_ok
        self._binary = ""
        self._version = ""
        self._doc: AtifDoc | None = None

    def resolve(self) -> str:
        """The executable a trial gets, as an absolute path."""
        exe = shutil.which(self.spec.binary, path=workspace.trial_env(self._environ)["PATH"])
        if not exe:
            version = self.pin.version if self.pin else "<version>"
            raise HarnessError(f"{self.name}: `{self.spec.binary}` is not on PATH; install "
                               f"it with `{Template(self.spec.install).safe_substitute(version=version)}`")
        return exe

    def env(self, cwd: str, home: str) -> dict[str, str]:
        base = workspace.trial_env(self._environ)
        env = {k: base[k] for k in _PASSED if k in base}
        env.update({
            "HOME": home, "XDG_CONFIG_HOME": os.path.join(home, ".config"),
            "XDG_DATA_HOME": os.path.join(home, ".local", "share"),
            "XDG_CACHE_HOME": os.path.join(home, ".cache"),
            "XDG_STATE_HOME": os.path.join(home, ".local", "state"),
            "TMPDIR": os.path.join(home, "tmp"), "PWD": cwd,
            # The crates prepare.py fetched are in the real cargo home, and cargo reads
            # them offline without writing there.
            "CARGO_HOME": base.get("CARGO_HOME")
            or os.path.join(self._environ.get("HOME", ""), ".cargo"),
        })
        values = self._values(workspace=cwd, home=home)
        env.update({k: Template(v).substitute(values) for k, v in self.spec.env.items()})
        return env

    def argv(self, trial: Trial) -> list[str]:
        values = self._values(prompt=trial.prompt, workspace=trial.workspace, home=trial.home,
                              inner_cap_s=str(max(trial.wall_cap_s - 30, 60)))
        return [self._binary or self.spec.binary,
                *(Template(a).substitute(values) for a in self.spec.argv)]

    def _values(self, **trial: str) -> dict[str, str]:
        ep = self.endpoint
        return {"origin": ep.origin, "model": ep.model, "context": str(ep.context),
                "tokenizer": ep.tokenizer, "api_key": API_KEY, **trial}

    def _confined(self, argv: list[str], cwd: str, home: str) -> list[str]:
        return [seatbelt.SANDBOX_EXEC, "-p", profile(cwd, home), *argv]

    def _run(self, argv: list[str], home: str, timeout_s: int) -> tuple[int, str]:
        """A short command — `--version`, a setup step — confined like a trial, in `home`."""
        proc = subprocess.Popen(self._confined(argv, home, home), cwd=home,
                                env=self.env(home, home), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            out, _ = proc.communicate(timeout=timeout_s)
        except subprocess.TimeoutExpired:
            _kill_group(proc)
            out, _ = proc.communicate()
        _kill_group(proc)
        return proc.returncode, out.decode("utf-8", "replace")

    def version(self) -> str:
        """What the installed binary says it is, asked the way a trial runs it: some
        harnesses write first-run state on any invocation, --version included."""
        if self._version:
            return self._version
        exe = self._binary or self.resolve()
        home = os.path.join(self.root, "_home", "_version", self.name)
        _fresh(home)
        values = self._values(home=home, binary=exe,
                              entry_dir=os.path.dirname(os.path.realpath(exe)))
        argv = [Template(a).substitute(values) for a in self.spec.version]
        code, out = self._run(argv, home, VERSION_TIMEOUT_S)
        found = _VERSION.search(out)
        if code != 0 or not found:
            raise HarnessError(f"{self.name}: `{' '.join(self.spec.version)}` exited {code}: "
                               f"{out.strip()[-200:]!r}")
        self._version = found.group(0)
        return self._version

    def start(self) -> Mapping[str, JsonValue]:
        spec = self.spec
        if spec.unsupported:
            raise HarnessError(f"{self.name} is marked unsupported: {spec.unsupported}")
        if not self._sandbox_ok():
            raise HarnessError("sandbox-exec does not enforce from this process (not macOS, "
                               "or already inside a sandbox); a CLI arm runs confined or not "
                               "at all")
        templates = [*spec.argv, *spec.env.values(), *spec.files.values()]
        if not self.endpoint.tokenizer and any("${tokenizer}" in t for t in templates):
            raise HarnessError(f"{self.name} renders token ids and needs the served model's "
                               "tokenizer as a local directory; none was found")
        exe = self._binary = self.resolve()
        version = self.version()
        sha256 = entry_sha256(exe)
        if not spec.locked:
            if not os.path.realpath(exe).startswith(REPO_ROOT + os.sep):
                raise HarnessError(f"{self.name}: `{spec.binary}` resolves to {exe}, not to "
                                   f"this checkout — refusing to measure a different tree")
        elif self.pin is None:
            raise HarnessError(f"{self.name} has no entry in harnesses.lock: install the "
                               f"version to measure, then `python harnesses.py lock {self.name}`")
        elif version != self.pin.version:
            raise HarnessError(f"{self.name} {version} is installed, harnesses.lock pins "
                               f"{self.pin.version}: {self.pin.install}")
        elif sha256 != self.pin.sha256:
            print(f"[{self.name}] {version} as locked, but its entry point differs from the "
                  f"locked one ({sha256[:12]} vs {self.pin.sha256[:12]}); meta.json records "
                  "this one", flush=True)
        return {"harness_sha256": sha256,
                "server": self.endpoint.origin, "model": self.endpoint.model,
                "server_ctx": self.endpoint.context,
                "isolation": "throwaway home, allowlisted env, seatbelt"}

    def render(self, home: str) -> dict[str, str]:
        """The config files a trial's home gets, by path under it."""
        values = self._values(home=home)
        return {rel: Template(text).substitute(values) for rel, text in self.spec.files.items()}

    def prepare(self, home: str) -> None:
        _fresh(home)
        for rel, text in self.render(home).items():
            path = os.path.join(home, rel)
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w", encoding="utf-8") as f:
                f.write(text)
        values = self._values(home=home)
        for args in self.spec.setup:
            argv = [self._binary or self.resolve(),
                    *(Template(a).substitute(values) for a in args)]
            code, out = self._run(argv, home, SETUP_TIMEOUT_S)
            if code != 0:
                raise HarnessError(f"{self.name}: setup `{' '.join(args[:1])}` exited "
                                   f"{code}: {out.strip()[-400:]}")

    def solve(self, trial: Trial) -> Solved:
        if not self._binary:
            raise RuntimeError("solve() before start(): the binary is not resolved")
        self._doc = None
        os.makedirs(os.path.dirname(trial.output), exist_ok=True)
        t0 = time.time()
        with open(trial.output, "wb") as out:
            proc = subprocess.Popen(self._confined(self.argv(trial), trial.workspace, trial.home),
                                    cwd=trial.workspace, env=self.env(trial.workspace, trial.home),
                                    stdin=subprocess.DEVNULL, stdout=out,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                proc.wait(timeout=trial.wall_cap_s)
                capped = False
            except subprocess.TimeoutExpired:
                capped = True
            # Read before the kill: tearing down is the runner's cost, not the agent's.
            wall = time.time() - t0
            _kill_group(proc)
        self._doc = self._native(trial)
        fields: dict[str, JsonValue] = {"exit_code": proc.returncode}
        if capped or proc.returncode != 0:
            fields["output_tail"] = _tail(trial.output)
        return Solved(wall, capped, fields)

    def _native(self, trial: Trial) -> AtifDoc | None:
        """The agent's own trajectory, copied out of its home to where the run keeps it."""
        if not self.spec.trajectory:
            return None
        src = os.path.join(trial.home, self.spec.trajectory)
        try:
            with open(src, encoding="utf-8") as f:
                doc: JsonValue = json.load(f)
        except (OSError, ValueError):
            return None
        os.makedirs(os.path.dirname(trial.trajectory), exist_ok=True)
        shutil.copyfile(src, trial.trajectory)
        return doc if is_object(doc) else None

    def trajectory(self) -> AtifDoc | None:
        return self._doc
