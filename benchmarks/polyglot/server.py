"""The llama-server the CLI arms run against — chad's own weights, in llama.cpp — and the
rule that keeps it from ever sharing the machine with the in-process engine.

    python benchmarks/polyglot/server.py          # foreground; Ctrl-C stops it

THE WEIGHTS ARE CHAD'S
----------------------
The arms run the GGUF build of the same model chad ships: Prism ML's Ternary Bonsai 2 of
Qwen3.8-27B, `PQ2_0` (7.2 GB), the pack its card measures on Apple Silicon, against
chad's MLX 2-bit pack of the same build. Anything else compares harnesses on weights no
chad user runs — and the first harness night, which served `Qwen3.8-27B-UD-Q3_K_XL`,
showed why that matters: chad in process on the ternary pack spiralled into a 20k-token
think on 6 of 9 trials, while the same harness on the conventional quant never did.
With one model everywhere, chad-mlx against chad-llama isolates the engine: if the
ternary pack rambles in llama.cpp too it is the weights, and if it does not it is chad's
own MLX path.

These files need Prism ML's llama.cpp fork — stock llama.cpp rejects `PQ2_0` as an
unknown type. `binary()` looks for it in `_data/llama-prism/` (or `POLYGLOT_LLAMA_SERVER`)
before falling back to whatever `llama-server` is on PATH, which will refuse to load
these weights. Fetch it once, pinned:

    cd benchmarks/polyglot/_data && mkdir -p llama-prism && cd llama-prism
    curl -L -o release.tar.gz https://github.com/PrismML-Eng/llama.cpp/releases/download/\
prism-b10709-9a9394a/llama-prism-b10709-9a9394a-bin-macos-arm64.tar.gz
    tar -xzf release.tar.gz
    hf download prism-ml/Ternary-Bonsai-2-27B-gguf Ternary-Bonsai-2-27B-PQ2_0.gguf

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

GGUF_REPO = "prism-ml/Ternary-Bonsai-2-27B-gguf"
GGUF_FILE = "Ternary-Bonsai-2-27B-PQ2_0.gguf"
ALIAS = "ternary-bonsai-2-27b"
# The fork whose kernels read those weights, pinned to the release these runs used.
FORK_TAG = "prism-b10709-9a9394a"
FORK_URL = (f"https://github.com/PrismML-Eng/llama.cpp/releases/download/{FORK_TAG}/"
            f"llama-{FORK_TAG}-bin-macos-arm64.tar.gz")
FORK_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_data", "llama-prism")
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


def binary(environ: Mapping[str, str] = os.environ, fork_dir: str = FORK_DIR) -> tuple[str, bool]:
    """The `llama-server` to run, and whether it is the fork these weights need: the
    pinned fork under `_data/` (`POLYGLOT_LLAMA_SERVER` overrides), else whatever is on
    PATH — which is stock llama.cpp, and rejects `PQ2_0` as an unknown type."""
    override = environ.get("POLYGLOT_LLAMA_SERVER")
    if override:
        return override, True
    for base, _dirs, files in sorted(os.walk(fork_dir)):
        if "llama-server" in files:
            return os.path.join(base, "llama-server"), True
    found = shutil.which("llama-server", path=environ.get("PATH"))
    if not found:
        raise EngineBusy(f"no llama-server at all: fetch the fork these weights need into "
                         f"{fork_dir} (see this module's docstring, release {FORK_TAG})")
    return found, False


def default_gguf() -> str:
    """chad's own weights in GGUF form, from the Hugging Face cache; never downloaded here."""
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(GGUF_REPO, GGUF_FILE, local_files_only=True)
    except (OSError, ValueError) as e:
        raise EngineBusy(f"{GGUF_REPO}/{GGUF_FILE} is not in the Hugging Face cache; fetch it "
                         f"once with `hf download {GGUF_REPO} {GGUF_FILE}`") from e


def version() -> str:
    out = subprocess.run([binary()[0], "--version"], capture_output=True, text=True,
                         check=False)
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
        return [binary()[0], "-m", self.gguf, "--host", "127.0.0.1", "--port", str(self.port),
                "-c", str(self.ctx), "-ngl", "999", "--jinja", "--metrics", "--alias", self.alias,
                "--slot-save-path", self.slots]

    def __enter__(self) -> str:
        refuse("server")
        if not binary()[1]:
            raise EngineBusy(
                f"{binary()[0]} is stock llama.cpp, which cannot read {os.path.basename(self.gguf)}"
                f" — fetch Prism ML's fork ({FORK_TAG}) into {FORK_DIR}, or point "
                "POLYGLOT_LLAMA_SERVER at it (see this module's docstring)")
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
        with LlamaServer(default_gguf(), log) as origin:
            print(f"llama-server up at {origin} ({version()}); log {log}. Ctrl-C stops it.",
                  flush=True)
            while True:
                time.sleep(3600)
    except KeyboardInterrupt:
        return 0
    except EngineBusy as e:
        print(e, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
