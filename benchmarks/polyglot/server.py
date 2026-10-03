"""The llama-server the CLI arms run against — the file chad ships, in llama.cpp — and
the rule that keeps it from ever sharing the machine with the in-process engine.

    python benchmarks/polyglot/server.py          # foreground; Ctrl-C stops it

THE WEIGHTS ARE CHAD'S, BYTE FOR BYTE
-------------------------------------
chad reads a GGUF natively, so the arms serve the file it loads in process: `default_gguf()`
asks chad which file it would pick with no `--model` (`chad.cli._pick_model`, so a
`CHAD_MODEL` override moves every arm together) and hands that path to llama-server. One
file everywhere takes the weights out of the comparison: chad-mlx against chad-llama
isolates the engine, and a foreign arm against chad-llama isolates the harness. Before
chad read GGUF, the llama arms ran a different quant of the same model, and a harness
gap carried a weights gap inside it.

Any stock llama-server reads the file (`binary()`: `POLYGLOT_LLAMA_SERVER`, else PATH);
`version()` records which build in every block's meta.json.

ONE ENGINE
----------
One server serves a whole llama phase: every CLI block of a night talks to it, one block
at a time. It is started once, not per block, because load/teardown cycling of a model
this size has panicked this GPU twice; each block empties the server's slots before its
first trial (`proxy.erase_slots`), so no arm inherits another's prefix cache.

A 24 GB laptop holds one 27B engine. Two resident — llama-server beside an MLX block —
means swap. So an in-process block refuses while a llama-server or another block is
running, a CLI block refuses while another block is running, and this server refuses to
start beside either.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
import urllib.request
from collections.abc import Callable, Mapping

ALIAS = "qwen3.8-27b-local"          # the model id every arm asks the server for
DEFAULT_PORT = 8081
DEFAULT_CTX = 32768
HEALTH_TIMEOUT_S = 600


class EngineBusy(RuntimeError):
    """Another engine is resident; starting this one would put two on the machine."""


def _pgrep(*args: str) -> list[int]:
    out = subprocess.run(["pgrep", *args], capture_output=True, text=True, check=False)
    return [int(p) for p in out.stdout.split()]


def refuse(starting: str, pgrep: Callable[..., list[int]] = _pgrep) -> None:
    """Raise if starting `starting` — "in-process block", "cli block" or "server" —
    would put two engines on the machine."""
    servers = pgrep("-x", "llama-server")
    mine = {os.getpid(), os.getppid()}       # this block and the `uv run` that launched it
    blocks = [p for p in pgrep("-f", "benchmarks/polyglot/run.py") if p not in mine]
    if blocks:
        raise EngineBusy(f"another polyglot block is running (pid {blocks}): one at a time")
    if starting != "cli block" and servers:
        raise EngineBusy(f"llama-server is running (pid {servers}): stop it first — one "
                         "engine at a time")


def binary(environ: Mapping[str, str] = os.environ) -> str:
    """The `llama-server` to run: `POLYGLOT_LLAMA_SERVER` when set, else the one on PATH."""
    override = environ.get("POLYGLOT_LLAMA_SERVER")
    if override:
        return override
    found = shutil.which("llama-server", path=environ.get("PATH"))
    if not found:
        raise EngineBusy("no llama-server on PATH (brew install llama.cpp), and "
                         "POLYGLOT_LLAMA_SERVER names none")
    return found


def shipped_spec() -> str:
    """The model chad loads with no `--model`: its shipped default, or `CHAD_MODEL`."""
    from chad.cli import _pick_model
    return _pick_model()[0]


def default_gguf(spec: Callable[[], str] = shipped_spec,
                 resolve: Callable[[str], str | None] | None = None) -> str:
    """The GGUF file chad in process loads, as a path for llama-server. Never downloaded
    here: a file chad has not fetched is one it has not run."""
    if resolve is None:
        from chad.gguf_pack import resolve_file
        resolve = resolve_file
    name = spec()
    path = resolve(name)
    if path is None:
        raise EngineBusy(f"{name} is not a GGUF file on this disk; the llama arms serve the "
                         "file chad loads in process, so run `chad` once to fetch it")
    return path


def version() -> str:
    out = subprocess.run([binary(), "--version"], capture_output=True, text=True, check=False)
    return next((ln.strip() for ln in (out.stdout + out.stderr).splitlines()
                 if ln.startswith("version")), "?")


class LlamaServer:
    """`with LlamaServer(gguf, log) as origin:` — up and healthy inside, gone after."""

    def __init__(self, gguf: str, log: str, alias: str = ALIAS, port: int = DEFAULT_PORT,
                 ctx: int = DEFAULT_CTX):
        self.gguf, self.log, self.alias, self.port, self.ctx = gguf, log, alias, port, ctx
        self.origin = f"http://127.0.0.1:{port}"
        self.slots = os.path.join(os.path.dirname(os.path.abspath(log)), "llama-slots")
        self._proc: subprocess.Popen[bytes] | None = None

    def argv(self) -> list[str]:
        # --slot-save-path: llama-server refuses every slot action without one, and a
        # block's first act is erasing the slots so it starts on a cold prefix cache.
        return [binary(), "-m", self.gguf, "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(self.ctx), "-ngl", "999", "--jinja", "--metrics", "--alias", self.alias,
                "--slot-save-path", self.slots]

    def __enter__(self) -> str:
        refuse("server")
        os.makedirs(self.slots, exist_ok=True)
        with open(self.log, "ab") as log:
            self._proc = subprocess.Popen(self.argv(), stdout=log, stderr=subprocess.STDOUT)
        deadline = time.time() + HEALTH_TIMEOUT_S
        while time.time() < deadline:
            if self._proc.poll() is not None:
                raise EngineBusy(f"llama-server exited {self._proc.returncode}; see {self.log}")
            try:
                with urllib.request.urlopen(self.origin + "/health", timeout=3) as r:
                    if r.status == 200:
                        return self.origin
            except OSError:
                time.sleep(2)
        self.__exit__()
        raise EngineBusy(f"llama-server never became healthy in {HEALTH_TIMEOUT_S} s; see {self.log}")

    def __exit__(self, *exc: object) -> None:
        if self._proc is None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()
        self._proc = None


def main() -> int:
    log = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_runs", "llama-server.log")
    try:
        gguf = default_gguf()
        with LlamaServer(gguf, log) as origin:
            print(f"llama-server up at {origin} ({version()}) serving {gguf}; log {log}. "
                  "Ctrl-C stops it.", flush=True)
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        return 0
    except EngineBusy as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
