"""Harness x engine: what the agent loop is worth, and what the engine is worth, separately.

`benchmarks/stock/` and `benchmarks/depth/` both answer an ENGINE question — how fast does
this thing read and write tokens. Neither can answer the question chad's README actually
makes, which is about an agent loop: does the harness get the job done faster on the same
laptop? A number like "62 tok/s" cannot be attributed to a harness at all, and comparing
chad against `llama-bench` conflates two variables that move independently.

This runs the 2-D grid instead. Same weights, same laptop, same tasks, same tests:

    pi        + llama.cpp     \\
    opencode  + llama.cpp      >  engine held constant -> what is the HARNESS worth
    chad      + llama.cpp     /
    chad      + MLX              harness held constant -> what is the ENGINE worth

The three llama.cpp arms share ONE `llama-server` process, so they are driven by the same
weights through the same sampler with the same server-side prefix cache. That last point
is the whole reason this benchmark exists: llama.cpp's server caches prefixes for every
client, so pi and opencode get prompt caching for free, and "just use llama.cpp's server"
stops being an argument anyone has to win in prose.

WHAT IS MEASURED, AND WHY NOT tok/s
-----------------------------------
tok/s is meaningless across harnesses — they send different prompts, expose different
tools, and take a different number of turns to reach the same place. What compares is:

  wall_s      wall clock from launch to exit, capped by `--timeout`
  passed      `pytest -q` exit code on the task's OWN test file, run AFTER the harness
              exits. A script, not a judgment; a harness that gives up early scores 0
              rather than scoring fast.
  prefill     prompt tokens the SERVER actually evaluated, read out of llama-server's
              own `/metrics` counters and diffed per task -- so token accounting is
              identical across the three llama arms and no harness is trusted to
              report on itself. (The MLX arm is in-process and reports its own; that
              asymmetry is stated, not hidden.)
  generated   tokens the server actually produced, same source.

Prefill is the headline: it is the dominant cost of a real session (see
`benchmarks/depth/`), it is where a harness's context economy shows up, and it is the
number a harness can inflate without ever looking slower on a microbenchmark.

FAIRNESS CONTROLS
-----------------
- Pristine task stubs are read from git, never from the working tree — a previous run's
  solution left in a stub would silently leak the answer into every later arm.
- One fresh temp dir per (arm, task, rep). Nothing is shared, nothing carries over.
- Every arm gets the SAME `PATH` (this repo's venv first), so `python`/`pytest` resolve
  identically. A bare `python` on PATH has produced phantom failures here before.
- Every arm is told where to work TWICE — as `cwd` and as `$PWD` — and opencode is also
  passed `--dir`. It resolves its project directory from the environment rather than
  from the cwd it is handed, and an arm pointed at the wrong directory spends its whole
  budget searching the filesystem for a task it is standing in.
- Every arm auto-approves its own tools (`-a` / `--auto` / `--yolo`). A harness stopping
  to ask permission is not being measured on the same terms as one that does not.
- Token counts are read only when the server is IDLE, on both sides of the arm. The
  counters advance on slot release, so an in-flight request is invisible to a snapshot
  and a request that outlives its harness is billed to whoever runs next.
- A timeout kills the harness's whole process GROUP, not just the process. opencode runs
  its own local server; a survivor keeps generating against the next arm's numbers.
- Every arm gets the same task prompt, the same timeout, and the same test command.
- The GGUF and the MLX checkpoint are the same dynamic-quant recipe, not the same bytes:
  this is the same "as close as two formats get" caveat `benchmarks/stock/` carries.

WHAT THE PROXY RECORDS PER TURN
-------------------------------
The grid says which harness finished faster. It cannot say why, and "why" is the part a
person running a local model actually feels: does the harness keep its prompt prefix
byte-stable so the server's KV cache hits, how many tokens of fixed tax does it prepend
to every turn, how long is the wait before the first token, how much of the wall clock is
the model even busy. Every request passes through `sampler_proxy.py`, which writes one
line per generation request to `_runs/turns.jsonl` (time-to-first-byte, hashes of the
system prompt and tool list, and llama-server's own `timings`: `prompt_n`, `cache_n`,
`predicted_n`, `prompt_ms`, `predicted_ms`). `scorecard.py` turns those into the
per-harness scorecard. The MLX arms never touch the proxy; chad's `CHAD_PREFILL_TRACE`
records the same fields in-process and the scorecard marks them self-reported.

Run it (ONE engine resident at a time — a 24 GB box holds exactly one):

    uv run python benchmarks/matrix/run.py setup    # write every harness's provider config
    uv run python benchmarks/matrix/run.py smoke    # one short task per llama arm -> smoke_verdict.json
    uv run python benchmarks/matrix/run.py llama --from-smoke   # the arms that passed smoke
    uv run python benchmarks/matrix/run.py mlx      # chad in-process; llama-server MUST be down
    uv run python benchmarks/matrix/run.py table
    uv run python benchmarks/matrix/scorecard.py

Prerequisites: `brew install llama.cpp`, the GGUF (downloaded into the shared HF cache on
first use; `STOCK_GGUF=<path>` points at one you have), and whichever harness binaries
you want as arms — a missing binary marks its arm `skipped` in provenance, never a crash.
The tasks ship in `tasks/` (Exercism, MIT; see `tasks/README.md`). `README.md` beside this
file is the reproduction kit: versions, install lines, what is committed and why.

Nothing written under `_runs/` names the machine: paths are stored relative to this
directory or the repo root, harness output is redacted of the home directory, the temp
directory and the hostname, and provenance records the chip and the engine build but not
the host. `tests/test_matrix_bench.py` scans the committed tree for the same things.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
# Where rows, turn records and traces go. The committed run lives in `_runs/`; point
# MATRIX_RUNS somewhere else to measure without appending to it (every file here is
# an accumulator by design — see overnight.sh).
RUNS = os.path.abspath(os.environ.get("MATRIX_RUNS") or os.path.join(HERE, "_runs"))
VENV_BIN = os.path.join(ROOT, ".venv", "bin")

# The tasks are vendored under `tasks/` and read out of git, never the working tree
# (see `_pristine`). Eight Exercism Python exercises as the aider polyglot benchmark
# packages them; `tasks/README.md` carries the attribution and the license.
TASKS_REL = "benchmarks/matrix/tasks"
GGUF_REPO = "unsloth/Qwen3.8-27B-GGUF"
GGUF_FILE = "Qwen3.8-27B-UD-Q3_K_XL.gguf"
SERVER_MODEL_ID = "qwen3.8-27b-local"
# Harnesses talk to the PROXY; the proxy talks to llama-server. Every arm's sampling
# parameters are overwritten in flight (see `sampler_proxy.py`), because the first run
# of this grid was confounded by two of them and no amount of per-harness config makes
# that verifiable — pi and opencode do not expose min_p at all.
PORT = 8080                       # sampler proxy: what every harness is pointed at
UPSTREAM_PORT = 8081              # llama-server itself
BASE = f"http://127.0.0.1:{PORT}"
UPSTREAM = f"http://127.0.0.1:{UPSTREAM_PORT}"
SERVER_CTX = 32768

# The canonical sampler, mirrored from `sampler_proxy.CANON`. The proxy enforces it for
# the three llama arms; the MLX arms are in-process and get it through CHAD_* env, which
# is why it has to be written twice — and why `_assert_sampler_agreement` checks that
# the two spellings actually agreed once the run is over.
CANON_ENV = {"CHAD_TEMP": "1.0", "CHAD_TOP_K": "20", "CHAD_TOP_P": "0.95",
             "CHAD_MIN_P": "0.05", "CHAD_PRESENCE_PENALTY": "0.0"}

# A difficulty spread that a 27B can plausibly clear. A task set where every arm scores 0
# measures nothing at all, and neither does one where every arm scores 1.
TASKS = ["bowling", "grade-school", "affine-cipher", "transpose",
         "wordy", "book-store", "dominoes", "go-counting"]

PROMPT = (
    "Implement the solution in {stub} so that the test suite passes.\n\n"
    "The task description is in instructions.md and the tests are in {test}.\n"
    "Run `python -m pytest -q {test}` to check your work, and keep going until the "
    "tests pass.\n\n"
    "Do not edit or delete the test file."
)

# -- arms ----------------------------------------------------------------------
#
# (prompt, workdir, args) -> argv. Each harness's own documented headless form, so the
# comparison is between products as their authors ship them — with exactly two things
# equalised. Every arm is handed the workdir explicitly as well as inheriting it as cwd,
# and every arm auto-approves its own tools (`-a` / `--auto` / `--yolo` / …), because a
# harness that stops to ask permission is not being measured on the same terms as one
# that does not. Where the auto-approve lives in a config file rather than argv (dsh,
# crush) the matching `ARM_SETUP` writer / `ARM_ENV` entry carries it.
#
# An arm is one entry here plus, where needed, one in ARM_ENV and one in ARM_SETUP.
# `LLAMA_ARMS` is derived: everything that is not an MLX arm talks to the proxy.
M = SERVER_MODEL_ID
MODEL_NAME = "Qwen3.8-27B UD-Q3_K_XL (llama.cpp)"
KEY = "sk-local-dummy"                    # every local endpoint ignores auth; a value
                                          # is still needed because clients insist
CFG = os.path.join(HERE, "_cfg")          # generated configs that do not live in $HOME


_MINI_YAML_CACHE: list = []


def _mini_yaml() -> str:
    """mini-swe-agent's bundled default config, named explicitly because passing any
    `-c` to mini REPLACES its default (so overrides must layer on top of the real file).

    Resolved through the interpreter that owns the `mini` executable — a uv tool venv, a
    pipx venv, a plain venv all put `python` beside the entry point — instead of a
    hard-coded site-packages path, which is specific to one install and one Python
    minor. `MATRIX_MINI_YAML=<path>` overrides. A missing `mini` returns a placeholder;
    that arm is skipped by name before anything is launched."""
    if _MINI_YAML_CACHE:
        return _MINI_YAML_CACHE[0]
    found = os.environ.get("MATRIX_MINI_YAML")
    exe = shutil.which("mini")
    if not found and exe:
        py = os.path.join(os.path.dirname(os.path.realpath(exe)), "python")
        if os.path.exists(py):
            r = subprocess.run(
                [py, "-c", "import minisweagent, os; print(os.path.join(os.path.dirname("
                           "minisweagent.__file__), 'config', 'mini.yaml'))"],
                capture_output=True, text=True, timeout=60)
            # Last line only: importing minisweagent prints a banner first.
            cand = (r.stdout.strip().splitlines() or [""])[-1].strip()
            if r.returncode == 0 and os.path.exists(cand):
                found = cand
    _MINI_YAML_CACHE.append(found or "mini.yaml")
    return _MINI_YAML_CACHE[0]


def _task_files(w: str) -> tuple:
    """(stub, test) inside a scaffolded workdir — for harnesses that take files as
    arguments (aider) rather than finding them."""
    test = next((f for f in os.listdir(w) if f.endswith("_test.py")), "")
    stub = test.replace("_test.py", ".py")
    return stub, test


ARMS = {
    "pi+llama": lambda p, w, a: ["pi", "-p", p, "--provider", "llama",
                                 "--model", M, "-a"],
    "opencode+llama": lambda p, w, a: ["opencode", "run", p,
                                       "--model", f"llama/{M}",
                                       "--dir", w, "--auto"],
    "chad+llama": lambda p, w, a: ["chad", p, "--yolo", "--backend", "llama",
                                   "--base-url", BASE],
    "chad+mlx": lambda p, w, a: ["chad", p, "--yolo"],
    # Same engine, drafter off. DFlash2 accepts via exact rejection sampling, so it is
    # distribution-preserving by construction: this arm isolates what speculation is
    # worth in wall clock WITHOUT changing what the model would have written. Without
    # it the engine cell measures two things at once and can only report their sum.
    "chad+mlx-nodflash": lambda p, w, a: ["chad", p, "--yolo"],
    # deepseek-harness: `headless` is a profile (one task, print, exit); approval is
    # decided by DSH_PERMISSION_MODE (ARM_ENV), the provider by cordis.patch.yml (setup).
    "dsh+llama": lambda p, w, a: ["dsh", "--profile", "headless", p],
    # deepagents-code: `-S all` is the headless shell allow-list; its own --timeout is
    # kept under ours so it exits cleanly instead of racing the process-group kill.
    "deepagents+llama": lambda p, w, a: ["dcode", "-n", p, "-q", "-S", "all",
                                         "-M", f"openai:{M}", "--max-turns", "200",
                                         "--timeout", str(max(a.timeout - 30, 60))],
    "goose+llama": lambda p, w, a: ["goose", "run", "-t", p, "--no-session", "-q",
                                    "--provider", "openai", "--model", M,
                                    "--with-builtin", "developer",
                                    "--max-turns", "200"],
    # mini-swe-agent: bash-only, no tool schemas at all — the floor of the harness axis.
    # `-c` REPLACES the default config, so the bundled mini.yaml is named first and then
    # overridden; MSWEA_CONFIGURED (ARM_ENV) skips the interactive first-run gate.
    "mini+llama": lambda p, w, a: ["mini", "-t", p, "-y", "--exit-immediately",
                                   "-c", _mini_yaml(), "-c", "agent.cost_limit=0",
                                   "-c", f"model.model_name=openai/{M}",
                                   "-c", f"model.model_kwargs.api_base={BASE}/v1",
                                   "-c", f"model.model_kwargs.api_key={KEY}"],
    # crush yolo lives in crush.json (permissions.skip_requests); `-c` (cwd) is a flag
    # on the `run` subcommand, not a persistent root flag.
    "crush+llama": lambda p, w, a: ["crush", "run", "-c", w, "-q", p],
    "qwen+llama": lambda p, w, a: ["qwen", "-p", p, "--approval-mode", "yolo", "-m", M],
    # aider is an edit-block chat loop, not a tool-calling agent: it is told the files
    # and runs the tests itself after each edit (`--auto-test`). It stays in the grid
    # as the "no tool schemas, no agent loop" reference point and is labelled as such.
    "aider+llama": lambda p, w, a: ["aider", "--message", p, "--yes-always",
                                    "--no-git", "--no-auto-commits",
                                    "--no-show-model-warnings", "--no-check-update",
                                    "--no-analytics", "--no-detect-urls",
                                    "--model", f"openai/{M}",
                                    "--openai-api-base", f"{BASE}/v1",
                                    "--openai-api-key", KEY,
                                    "--auto-test", "--test-cmd",
                                    f"python -m pytest -q {_task_files(w)[1]}",
                                    "--read", "instructions.md", _task_files(w)[0]],
    "cline+llama": lambda p, w, a: ["cline", p, "-P", "openai", "-m", M, "-c", w,
                                    "--auto-approve", "true",
                                    "-t", str(max(a.timeout - 30, 60))],
    # codex speaks only the Responses wire; llama-server serves /v1/responses, so the
    # proxy forces the sampler there too (it is in the proxy's GEN_PATHS).
    "codex+llama": lambda p, w, a: ["codex", "exec",
                                    "--dangerously-bypass-approvals-and-sandbox",
                                    "--skip-git-repo-check", "-C", w, "--json",
                                    "-c", "model_provider=llama", "-c", f'model="{M}"',
                                    p],
}
MLX_ARMS = ["chad+mlx", "chad+mlx-nodflash"]
LLAMA_ARMS = [x for x in ARMS if x not in MLX_ARMS]
# Per-arm environment ON TOP of the canonical sampler. For chad only ever used to turn
# a feature off for a control arm, never to tune one; for the others it is how the
# harness is pointed at the proxy and told not to ask.
ARM_ENV = {
    "chad+mlx-nodflash": {"CHAD_NO_DFLASH": "1"},
    "dsh+llama": {"DSH_PERMISSION_MODE": "danger-full-access", "LLAMA_API_KEY": KEY,
                  "DSH_TELEMETRY_MODE": "DISABLED"},
    "deepagents+llama": {"OPENAI_BASE_URL": f"{BASE}/v1", "OPENAI_API_KEY": KEY},
    "goose+llama": {"GOOSE_PROVIDER": "openai", "GOOSE_MODEL": M,
                    "OPENAI_HOST": BASE, "OPENAI_BASE_PATH": "v1/chat/completions",
                    "OPENAI_API_KEY": KEY, "GOOSE_MODE": "auto",
                    "GOOSE_DISABLE_KEYRING": "1"},
    # mini refuses a $0 turn ("Cost must be > 0.0") even with the zero-cost registry
    # entry; `ignore_errors` is its documented value for local models, "false" is not.
    "mini+llama": {"OPENAI_API_KEY": KEY,
                   "LITELLM_MODEL_REGISTRY_PATH": os.path.join(CFG, "litellm_registry.json"),
                   "MSWEA_CONFIGURED": "true", "MSWEA_MODEL_NAME": f"openai/{M}",
                   "MSWEA_COST_TRACKING": "ignore_errors", "MSWEA_SILENT_STARTUP": "true"},
    "qwen+llama": {"OPENAI_API_KEY": KEY, "OPENAI_BASE_URL": f"{BASE}/v1",
                   "OPENAI_MODEL": M},
    "aider+llama": {"OPENAI_API_KEY": KEY, "OPENAI_API_BASE": f"{BASE}/v1"},
    "codex+llama": {"LLAMA_API_KEY": KEY},
}


class _Args:
    timeout = 1200


def _bin(arm: str) -> str:
    """The executable an arm needs on PATH."""
    return ARMS[arm]("x", tempfile.gettempdir(), _Args)[0]


# Three hand-maintained collections is three ways to forget one. An arm missing from
# LLAMA_ARMS would be routed to the self-reported (MLX) path and print prefill=0.
assert all((x in LLAMA_ARMS) != (x in MLX_ARMS) for x in ARMS), "arm in neither/both lists"
assert set(ARM_ENV) <= set(ARMS), f"ARM_ENV names unknown arms: {set(ARM_ENV) - set(ARMS)}"



def _harness_digest() -> dict:
    """Content hashes of the three files that define the measurement.

    The tree these live in is often untracked while a grid is being iterated on, so
    `commit`/`dirty` above describe chad but say nothing about the harness that drove
    it. Hashing the files makes a row self-describing whether or not they were ever
    committed."""
    import hashlib
    out = {}
    for rel in ("benchmarks/matrix/run.py", "benchmarks/matrix/sampler_proxy.py",
                "benchmarks/matrix/scorecard.py", "benchmarks/matrix/overnight.sh"):
        try:
            with open(os.path.join(ROOT, rel), "rb") as f:
                out[os.path.basename(rel)] = hashlib.sha256(f.read()).hexdigest()[:12]
        except OSError:
            out[os.path.basename(rel)] = "?"
    return out


def _rel(path: str | None) -> str | None:
    """A path as it will be written to disk: relative to the repo root when it is
    inside it, otherwise its basename. Absolute paths name the user and the machine,
    and `_runs/` is committed."""
    if not path:
        return path
    if path.startswith(ROOT + os.sep):
        return os.path.relpath(path, ROOT)
    return os.path.basename(path)


def _redact(text: str) -> str:
    """Harness output with the home directory, the temp directory and the hostname
    replaced. Only ever applied to text that is stored (`tail`); nothing here is
    shown to the harness."""
    if not text:
        return text
    tmp = tempfile.gettempdir().rstrip(os.sep)
    home = os.path.expanduser("~").rstrip(os.sep)
    for raw, sub in ((tmp, "$TMPDIR"), (os.path.realpath(tmp), "$TMPDIR"),
                     (home, "~"), (os.path.realpath(home), "~")):
        if raw and raw != os.sep:
            text = text.replace(raw, sub)
    host = socket.gethostname()
    for h in sorted({host, host.split(".")[0]}, key=len, reverse=True):
        if len(h) >= 4:
            text = text.replace(h, "<host>")
    return text


def _machine() -> dict:
    """The hardware and the engine build — what a reader needs to know whether their
    box is comparable — and nothing that names the box."""
    def _cmd(*a):
        try:
            r = subprocess.run(list(a), capture_output=True, text=True, timeout=20)
            return (r.stdout + r.stderr).strip()
        except Exception:  # noqa: BLE001
            return "?"
    out = {"chip": _cmd("sysctl", "-n", "machdep.cpu.brand_string"),
           "macos": _cmd("sw_vers", "-productVersion"),
           "llama_cpp": _cmd("llama-server", "--version").splitlines()[0]
           if shutil.which("llama-server") else "?"}
    mem = _cmd("sysctl", "-n", "hw.memsize")
    out["memory_gb"] = round(int(mem) / 2 ** 30) if mem.isdigit() else "?"
    return out


def _provenance(extra=None) -> dict:
    """Who exactly produced these numbers. A benchmark whose rows do not name the tree,
    the version and the MLX build that made them is not reproducible, it is an anecdote —
    and an interactive shell's `chad` may well be an alias onto a DIFFERENT checkout, so
    "which chad ran" is a live question every single time."""
    import subprocess as _sp
    def _git(*a):
        try:
            return _sp.run(["git", "-C", ROOT, *a], capture_output=True,
                           text=True).stdout.strip()
        except Exception:  # noqa: BLE001
            return "?"
    out = {"chad_version": "?", "chad_file": "?", "mlx_version": "?",
           "commit": _git("rev-parse", "--short", "HEAD"),
           "branch": _git("rev-parse", "--abbrev-ref", "HEAD"),
           "dirty": bool(_git("status", "--porcelain")),
           "python": _rel(sys.executable),
           "python_version": sys.version.split()[0],
           "machine": _machine(),
           # Two grids that differ only in one sampler value are otherwise
           # indistinguishable on disk, so the level is named here too.
           "canon_sampler": dict(CANON_ENV),
           "harness": _harness_digest()}
    chad_file = "?"
    try:
        import chad
        out["chad_version"], chad_file = chad.__version__, chad.__file__
    except Exception:  # noqa: BLE001
        pass
    try:
        import mlx.core as _mx
        out["mlx_version"] = _mx.__version__
    except Exception:  # noqa: BLE001
        pass
    if extra:
        out.update(extra)
    if not chad_file.startswith(ROOT):
        raise SystemExit(f"chad resolves to {chad_file}, not {ROOT} — refusing to "
                         f"benchmark a tree that is not the one under test")
    out["chad_file"] = _rel(chad_file)
    return out

def _save(name, obj):
    os.makedirs(RUNS, exist_ok=True)
    with open(os.path.join(RUNS, f"{name}.json"), "w") as f:
        json.dump(obj, f, indent=2)


def _load(name):
    p = os.path.join(RUNS, f"{name}.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _gguf():
    local = os.environ.get("STOCK_GGUF")
    if local:
        return local
    from huggingface_hub import hf_hub_download
    return hf_hub_download(GGUF_REPO, GGUF_FILE)


# -- task scaffolding ----------------------------------------------------------

def _pristine(task: str, name: str) -> str:
    """A task file as COMMITTED, not as it sits in the working tree.

    This is not fussiness. Exercise dirs get solved in place by hand and by other runs,
    and a stub containing a previous run's answer hands the solution to every arm that
    follows — a contamination bug that would look like a very good benchmark result."""
    rel = f"{TASKS_REL}/{task}/{name}"
    out = subprocess.run(["git", "-C", ROOT, "show", f"HEAD:{rel}"],
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise FileNotFoundError(f"{rel} is not committed — refusing to use the working "
                                f"tree copy, which may contain a solution")
    return out.stdout


def scaffold(task: str, dest: str) -> tuple:
    """Lay out one task in `dest`: the pristine stub, its tests, the instructions."""
    mod = task.replace("-", "_")
    stub, test = f"{mod}.py", f"{mod}_test.py"
    os.makedirs(dest, exist_ok=True)
    open(os.path.join(dest, stub), "w").write(_pristine(task, stub))
    open(os.path.join(dest, test), "w").write(_pristine(task, test))
    try:
        doc = _pristine(task, "instructions.md")
    except FileNotFoundError:
        doc = f"# {task}\n\nMake the tests in {test} pass.\n"
    open(os.path.join(dest, "instructions.md"), "w").write(doc)
    return stub, test


_TALLY = re.compile(r"(\d+) (passed|failed|errors?|skipped|xfailed|xpassed)\b")


def _tally(out: str) -> dict:
    """pytest's own summary line, as a count. Diagnostic only — `passed` stays the
    metric. An arm that gets 29 of 31 and an arm that never writes a line both score
    zero, and a table that cannot tell them apart is a table nobody can act on."""
    for line in reversed(out.strip().splitlines()):
        found = _TALLY.findall(line)
        if not found:
            continue
        counts: dict[str, int] = {}
        for n, what in found:
            counts[what.rstrip("s")] = counts.get(what.rstrip("s"), 0) + int(n)
        return {"tests_passed": counts.get("passed", 0),
                "tests_total": sum(counts.values())}
    return {"tests_passed": 0, "tests_total": 0}


def verify(dest: str, test: str, timeout: int = 300) -> dict:
    """`pytest -q` on the task's own tests, run after the harness exits. The exit code
    is the only definition of 'done' this benchmark accepts; the tally beside it is
    there so a failed arm is still legible."""
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", test],
                       cwd=dest, capture_output=True, text=True, timeout=timeout)
    return {"passed": r.returncode == 0, **_tally(r.stdout + r.stderr)}


# -- llama-server --------------------------------------------------------------

def _metrics() -> dict:
    """llama-server's own Prometheus counters. Reading token counts from the SERVER
    rather than from each harness is what makes the three llama arms comparable — no
    harness is trusted to report on itself, and they do not agree on what a token is."""
    try:
        with urllib.request.urlopen(f"{UPSTREAM}/metrics", timeout=5) as r:
            body = r.read().decode()
    except Exception:  # noqa: BLE001 — metrics are instrumentation, never the crash
        return {}
    out = {}
    for line in body.splitlines():
        if line.startswith("#") or ":" not in line:
            continue
        key, _, val = line.partition(" ")
        try:
            out[key.split(":")[-1]] = float(val)
        except ValueError:
            continue
    return out


def _drain(timeout: float = 300.0) -> bool:
    """Block until llama-server has nothing in flight.

    The counters only move when a slot RELEASES. A snapshot taken while a request is
    still running credits none of it to anyone, and a request that outlives the harness
    that asked for it lands on the NEXT arm's bill instead. A timed-out arm hits both at
    once, which is how an arm that generated 7k tokens can be recorded as generating 143.
    So: drain before reading `before`, and drain again before reading `after`.

    Returns False if the server never went quiet, which makes the row's counts suspect
    rather than silently wrong.
    """
    t0 = time.time()
    while time.time() - t0 < timeout:
        m = _metrics()
        if not m:
            return False
        if not m.get("requests_processing", 0) and not m.get("requests_deferred", 0):
            return True
        time.sleep(1)
    return False


def _kill_group(proc: subprocess.Popen) -> None:
    """Kill the harness AND everything it spawned.

    Killing only the direct child is not enough here: opencode runs its own local server
    as a separate process, and a survivor keeps talking to llama-server long after its
    arm is over — inflating the next arm's token counts with work it never did.
    """
    try:
        pgid = os.getpgid(proc.pid)
    except ProcessLookupError:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            return
        try:
            proc.wait(timeout=10)
            return
        except subprocess.TimeoutExpired:
            continue


def _self_reported(path: str) -> dict:
    """Token counts for the in-process arm, out of chad's own ATIF trajectory.

    The MLX arm never touches llama-server, so `/metrics` cannot see it and the row
    would otherwise read a flat zero for the benchmark's headline number. chad rewrites
    the trajectory after every step, so a timed-out arm still leaves a usable one.

    The two sources are made to mean the same thing rather than merely look alike:
    llama-server's `prompt_tokens_total` counts tokens it actually evaluated, EXCLUDING
    what the prefix cache served, while ATIF's `total_prompt_tokens` deliberately adds
    cached tokens back in. Subtracting them again is what puts both arms on one axis.
    This is still a self-report, and the row says so in `counts_source`.
    """
    try:
        with open(path) as f:
            m = json.load(f).get("final_metrics") or {}
    except (OSError, ValueError):
        return {}
    if "total_prompt_tokens" not in m:
        return {}
    return {"prompt_tokens_total": m["total_prompt_tokens"]
                                   - m.get("total_cached_tokens", 0),
            "tokens_predicted_total": m.get("total_completion_tokens", 0)}


def _health(timeout: int = 600) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f"{UPSTREAM}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:  # noqa: BLE001 — still booting
            time.sleep(3)
    return False


def start_server(a):
    if not shutil.which("llama-server"):
        sys.exit("llama-server not on PATH — brew install llama.cpp")
    # A second resident engine on a 24 GB box means swap, and swap means the numbers are
    # noise. Refuse rather than produce a plausible-looking bad result.
    busy = subprocess.run(["pgrep", "-f", "benchmarks/depth/run.py"],
                          capture_output=True, text=True).stdout.strip()
    if busy:
        sys.exit("the MLX depth benchmark is running — one engine at a time")
    cmd = ["llama-server", "-m", _gguf(), "--port", str(UPSTREAM_PORT),
           "--host", "127.0.0.1",
           "-c", str(SERVER_CTX), "-ngl", "999", "--metrics",
           "--alias", SERVER_MODEL_ID, "--jinja"]
    print(" ".join(cmd), flush=True)
    log = open(os.path.join(RUNS, "llama-server.log"), "w")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
    if not _health():
        proc.terminate()
        sys.exit("llama-server never became healthy — see _runs/llama-server.log")
    print(f"llama-server up on {UPSTREAM}", flush=True)
    return proc


def start_proxy(a=None):
    """The sampler proxy, in front of llama-server. Fails the run rather than falling
    back to a direct connection: a silent fallback is exactly the failure this whole
    apparatus exists to prevent, and it would look like a clean result."""
    plog = open(os.path.join(RUNS, "sampler-proxy.log"), "w")
    argv = [sys.executable, os.path.join(HERE, "sampler_proxy.py"),
            "--listen", str(PORT), "--upstream", str(UPSTREAM_PORT), "--runs", RUNS]
    if getattr(a, "capture_bodies", False):
        # Whole request bodies, one file per request, for diffing consecutive turns of
        # one harness (`body_diff.py`). Off by default: bodies hold every prompt the
        # harness sent, and `_runs/bodies/` is never committed.
        argv += ["--bodies", os.path.join(RUNS, "bodies")]
    proc = subprocess.Popen(argv, stdout=plog, stderr=subprocess.STDOUT)
    for _ in range(60):
        try:
            with urllib.request.urlopen(f"{BASE}/health", timeout=3) as r:
                if r.status == 200:
                    print(f"sampler proxy up on {BASE} -> {UPSTREAM}", flush=True)
                    return proc
        except Exception:  # noqa: BLE001 — still binding
            time.sleep(1)
    proc.terminate()
    sys.exit("sampler proxy never answered — see _runs/sampler-proxy.log")


def _tell_proxy(arm: str, task: str, rep: int) -> None:
    """Label the audit trail. The proxy cannot tell pi from opencode by itself — both
    are anonymous POSTs to /v1/chat/completions — so the runner names the arm in flight.
    Best-effort: this is bookkeeping, and the MLX arms have no proxy to tell."""
    try:
        req = urllib.request.Request(
            f"{BASE}/__arm", method="POST",
            data=json.dumps({"arm": arm, "task": task, "rep": rep}).encode(),
            headers={"Content-Type": "application/json"})
        urllib.request.urlopen(req, timeout=5).read()
    except Exception:  # noqa: BLE001
        pass


def _assert_sampler_agreement() -> dict:
    """Refuse to report numbers that were not produced under one sampler.

    This is the check the first run did not have. It reads back what the proxy actually
    applied and requires exactly ONE distinct parameter set across every request of every
    llama arm. A second set means an endpoint slipped past the rewrite, and the grid is
    measuring sampling instead of harnesses."""
    try:
        with urllib.request.urlopen(f"{BASE}/__audit", timeout=10) as r:
            audit = json.loads(r.read().decode())
    except Exception as e:  # noqa: BLE001
        sys.exit(f"cannot read the sampler audit ({e}) — refusing to report a grid "
                 f"whose sampling was never verified")
    distinct = audit.get("distinct_forced") or []
    if len(distinct) != 1:
        sys.exit(f"sampler audit shows {len(distinct)} distinct parameter sets across "
                 f"{audit.get('requests')} requests: {json.dumps(distinct, indent=2)}")
    forced = distinct[0]
    # The MLX arms never touch the proxy, so the two spellings of the canonical sampler
    # are checked against each other here rather than assumed to match.
    for env_var, field in (("CHAD_TEMP", "temperature"), ("CHAD_TOP_K", "top_k"),
                           ("CHAD_TOP_P", "top_p"), ("CHAD_MIN_P", "min_p"),
                           ("CHAD_PRESENCE_PENALTY", "presence_penalty")):
        if abs(float(CANON_ENV[env_var]) - float(forced[field])) > 1e-9:
            sys.exit(f"{env_var}={CANON_ENV[env_var]} (MLX arms) but the proxy forced "
                     f"{field}={forced[field]} on the llama arms — the two engines did "
                     f"NOT sample alike")
    print(f"sampler agreement OK: {audit['requests']} requests, one parameter set, "
          f"arms {audit['arms']}", flush=True)
    return forced


# -- provider configs ----------------------------------------------------------

def _home(rel: str) -> str:
    return os.path.expanduser(os.path.join("~", rel))


def _backup(path: str) -> None:
    """Keep the user's own config once, beside the one this benchmark writes."""
    bak = path + ".pre-matrix.bak"
    if os.path.exists(path) and not os.path.exists(bak):
        shutil.copy2(path, bak)


def _write(path: str, text: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _backup(path)
    with open(path, "w") as f:
        f.write(text)
    return path


def _setup_pi() -> str:
    p = _home(".pi/agent/models.json")
    cfg = json.load(open(p)) if os.path.exists(p) else {"providers": {}}
    cfg.setdefault("providers", {})["llama"] = {
        "baseUrl": f"{BASE}/v1", "api": "openai-completions",
        "apiKey": KEY, "models": [{"id": M}]}
    return _write(p, json.dumps(cfg, indent=2))


def _setup_opencode() -> str:
    p = _home(".config/opencode/opencode.jsonc")
    raw = open(p).read() if os.path.exists(p) else '{"provider":{}}'
    # The file is .jsonc but has only ever held plain JSON here; if that changes this
    # raises rather than silently writing a config the tool will ignore.
    cfg = json.loads(raw)
    cfg.setdefault("provider", {})["llama"] = {
        "npm": "@ai-sdk/openai-compatible", "name": "llama-server-local",
        "options": {"baseURL": f"{BASE}/v1", "apiKey": KEY},
        "models": {M: {"name": MODEL_NAME,
                       "limit": {"context": SERVER_CTX, "output": 8192}}}}
    return _write(p, json.dumps(cfg, indent=2))


def _setup_dsh() -> str:
    """deepseek-harness: the profile's own patch layer. `compat` tells pi-ai this is
    not OpenAI itself (no `developer` role, `max_tokens` not `max_completion_tokens`),
    which is what an OpenAI-shaped gateway needs. The headless profile dir is created
    by dsh on first boot; if it is missing, boot it once."""
    p = _home(".dsh/profiles/headless/cordis.patch.yml")
    return _write(p, f"""# Written by benchmarks/matrix/run.py setup — points dsh at the sampler proxy.
- id: llm-pi-ai
  config:
    providers:
      llama:
        displayName: llama-server-local
        apiKeyEnv: LLAMA_API_KEY
        api: openai-completions
        baseURL: {BASE}/v1
        compat:
          supportsDeveloperRole: false
          maxTokensField: max_tokens
        models:
          - id: {M}
            name: {MODEL_NAME}
            contextWindow: {SERVER_CTX}
            maxTokens: 8192
- id: agent-default-model
  config:
    provider: llama
    model: {M}
""")


def _setup_goose() -> str:
    p = _home(".config/goose/config.yaml")
    return _write(p, f"""# Written by benchmarks/matrix/run.py setup.
GOOSE_PROVIDER: openai
GOOSE_MODEL: {M}
GOOSE_MODE: auto
OPENAI_HOST: {BASE}
OPENAI_BASE_PATH: v1/chat/completions
extensions:
  developer:
    enabled: true
    type: builtin
    name: developer
    display_name: Developer
    timeout: 300
    bundled: true
""")


def _setup_mini() -> str:
    """LiteLLM refuses to price a model it has never heard of; a zero-cost registry
    entry is what keeps mini's cost tracking from raising on a local model."""
    p = os.path.join(CFG, "litellm_registry.json")
    reg = {f"openai/{M}": {"max_tokens": 8192, "max_input_tokens": SERVER_CTX,
                           "max_output_tokens": 8192, "input_cost_per_token": 0.0,
                           "output_cost_per_token": 0.0, "litellm_provider": "openai",
                           "mode": "chat"}}
    return _write(p, json.dumps(reg, indent=2))


def _setup_crush() -> str:
    p = _home(".config/crush/crush.json")
    cfg = {"$schema": "https://charm.land/crush.json",
           "providers": {"llama": {"type": "openai-compat", "name": "llama-server-local",
                                   "base_url": f"{BASE}/v1", "api_key": KEY,
                                   "models": [{"id": M, "name": MODEL_NAME,
                                               "context_window": SERVER_CTX,
                                               "default_max_tokens": 8192}]}},
           "models": {"large": {"model": M, "provider": "llama"},
                      "small": {"model": M, "provider": "llama"}},
           "options": {"disable_provider_auto_update": True, "disable_metrics": True,
                       "disable_default_providers": True},
           # yolo, spelled in config as well as `-y`: belt and braces on the one thing
           # that would otherwise stall an arm for its whole budget.
           "permissions": {"skip_requests": True}}
    return _write(p, json.dumps(cfg, indent=2))


def _setup_qwen() -> str:
    p = _home(".qwen/settings.json")
    cfg = json.load(open(p)) if os.path.exists(p) else {}
    cfg.setdefault("security", {}).setdefault("auth", {})["selectedType"] = "openai"
    cfg.setdefault("general", {})["disableAutoUpdate"] = True
    cfg.setdefault("privacy", {})["usageStatisticsEnabled"] = False
    return _write(p, json.dumps(cfg, indent=2))


def _setup_codex() -> str:
    p = _home(".codex/config.toml")
    return _write(p, f"""# Written by benchmarks/matrix/run.py setup — the previous file is beside it as .pre-matrix.bak
model = "{M}"
model_provider = "llama"

[model_providers.llama]
name = "llama-server-local"
base_url = "{BASE}/v1"
wire_api = "responses"
env_key = "LLAMA_API_KEY"
requires_openai_auth = false
""")


def _setup_cline() -> str:
    """cline keeps provider state in its own store; its `auth` subcommand is the
    documented way to write it non-interactively. `openai` is cline's id for the
    OpenAI-*compatible* provider (`openai-native` is OpenAI itself)."""
    r = subprocess.run(["cline", "auth", "-p", "openai", "-k", KEY, "-m", M,
                        "-b", f"{BASE}/v1"], capture_output=True, text=True, timeout=120)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr).strip()[-400:])
    return "cline auth (provider store)"


ARM_SETUP = {
    "pi+llama": _setup_pi, "opencode+llama": _setup_opencode, "dsh+llama": _setup_dsh,
    "goose+llama": _setup_goose, "mini+llama": _setup_mini, "crush+llama": _setup_crush,
    "qwen+llama": _setup_qwen, "codex+llama": _setup_codex, "cline+llama": _setup_cline,
    # deepagents / aider / chad: argv + env only, nothing to write.
}


def setup(a) -> None:
    """Point every requested harness at the proxy. Each writer adds a `llama` provider
    beside whatever the user had (backed up once as *.pre-matrix.bak); a harness that
    is not installed is reported and skipped, never a crash."""
    for arm in a.arms:
        fn = ARM_SETUP.get(arm)
        if fn is None:
            continue
        if not shutil.which(_bin(arm)):
            print(f"{arm:20s} SKIP: `{_bin(arm)}` not on PATH", flush=True)
            continue
        try:
            print(f"{arm:20s} -> {fn()}", flush=True)
        except Exception as e:  # noqa: BLE001 — one tool's config must not block the rest
            print(f"{arm:20s} FAILED: {e}", flush=True)


# -- the grid ------------------------------------------------------------------

def run_one(arm: str, task: str, rep: int, a) -> dict:
    work = tempfile.mkdtemp(prefix=f"mx-{arm.replace('+', '-')}-{task}-")
    stub, test = scaffold(task, work)
    prompt = PROMPT.format(stub=stub, test=test)
    # PWD as well as cwd. opencode resolves its project directory from $PWD, not from
    # the cwd it was handed, and will otherwise run against whatever directory this
    # script was launched from — reading the wrong files and never touching the stub.
    # chad's configuration is DECLARED here, never inherited. Every CHAD_* is stripped
    # from the ambient environment first, because chad's own convention is that a flag is
    # true when merely PRESENT (`config.flag`: "any non-empty value is true, including
    # '0'") — so a stray CHAD_NO_DFLASH in the launching shell would silently turn the
    # drafter off in the arm that is supposed to have it, and the run would look fine.
    # That is the same shape as the bug this whole apparatus exists to kill: a knob
    # nobody set deciding the result.
    env = {k: v for k, v in os.environ.items() if not k.startswith("CHAD_")}
    env.update({"PATH": VENV_BIN + os.pathsep + os.environ["PATH"], "PWD": work})
    # The canonical sampler goes to EVERY arm, not just the ones that read it. pi and
    # opencode ignore CHAD_*; they are equalised by the proxy instead. Setting it
    # unconditionally means the two mechanisms cannot drift apart in this function.
    env.update(CANON_ENV)
    env.update(ARM_ENV.get(arm, {}))
    on_llama = arm in LLAMA_ARMS
    # Kept OUT of the workdir: every arm must see the same three files and nothing else.
    # Deterministic per (arm, task, rep) so the scorecard can find them afterwards; the
    # prefill trace is chad's in-process equivalent of the proxy's turn record.
    trace_dir = os.path.join(RUNS, "traces", f"{arm}-{task}-{rep}")
    os.makedirs(trace_dir, exist_ok=True)
    traj = os.path.join(trace_dir, "trajectory.json")
    prefill_trace = os.path.join(trace_dir, "prefill_trace.jsonl")
    if not on_llama:
        env["CHAD_TRAJECTORY_JSON"] = traj
        env["CHAD_PREFILL_TRACE"] = prefill_trace
    if on_llama:
        _tell_proxy(arm, task, rep)
    quiet_before = _drain() if on_llama else True
    before = _metrics() if on_llama else {}
    argv = ARMS[arm](prompt, work, a)
    stub_before = open(os.path.join(work, stub)).read()
    t0 = time.time()
    timed_out = False
    # Own session: the timeout has to take down the harness's whole process tree.
    proc = subprocess.Popen(argv, cwd=work, env=env, start_new_session=True,
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        out, err = proc.communicate(timeout=a.timeout)
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc, timed_out = -1, True
        _kill_group(proc)
        out, err = proc.communicate()
    # Wall clock is the harness's, capped by --timeout, and is read BEFORE the drain:
    # draining is the benchmark's own accounting, not the arm's cost.
    wall = time.time() - t0
    tail = "TIMEOUT\n" if timed_out else ""
    tail += (out or "")[-2000:] + (err or "")[-2000:]
    quiet_after = _drain() if on_llama else True
    if on_llama:
        after, counts_source = _metrics(), "llama-server /metrics"
    else:
        after, counts_source = _self_reported(traj), "chad trajectory (self-reported)"
        quiet_after = bool(after)
    try:
        v = verify(work, test)
    except Exception as e:  # noqa: BLE001 — a broken verify is a failed task, not a crash
        v = {"passed": False, "tests_passed": 0, "tests_total": 0}
        tail = f"{tail}\nVERIFY ERROR: {e}"
    passed = v["passed"]
    try:
        stub_changed = open(os.path.join(work, stub)).read() != stub_before
    except OSError:
        stub_changed = False
    # What is written is what gets committed: paths relative to the repo, the kept
    # workdir by name only (its full path is printed below), harness output redacted.
    row = dict(arm=arm, task=task, rep=rep, wall_s=round(wall, 1), passed=passed,
               chad_env={k: v for k, v in env.items()
                         if k.startswith("CHAD_")
                         and k not in ("CHAD_TRAJECTORY_JSON", "CHAD_PREFILL_TRACE")},
               timed_out=timed_out, exit_code=rc, workdir=os.path.basename(work),
               stub_changed=stub_changed,
               trajectory_path=_rel(traj) if not on_llama else None,
               prefill_trace=_rel(prefill_trace) if not on_llama else None,
               counts_clean=bool(quiet_before and quiet_after),
               counts_source=counts_source,
               tests_passed=v["tests_passed"], tests_total=v["tests_total"],
               prefill=round(after.get("prompt_tokens_total", 0)
                             - before.get("prompt_tokens_total", 0)),
               generated=round(after.get("tokens_predicted_total", 0)
                               - before.get("tokens_predicted_total", 0)),
               tail=_redact(tail[-1200:]))
    print(f"  {arm:16s} {task:14s} {'PASS' if passed else 'fail':4s} "
          f"{wall:7.1f}s  prefill {row['prefill']:>7}  gen {row['generated']:>6}"
          f"  tests {row['tests_passed']}/{row['tests_total']}"
          f"{'  TIMEOUT' if timed_out else ''}"
          f"{'  COUNTS-SUSPECT' if not row['counts_clean'] else ''}"
          f"{'  kept ' + work if a.keep else ''}", flush=True)
    if not a.keep:
        shutil.rmtree(work, ignore_errors=True)
        row["workdir"] = None
    return row


def _resolved_chad() -> str:
    """The `chad` a subprocess actually gets. NOT necessarily the `chad` in an
    interactive shell: an alias onto another checkout does not survive into
    subprocess.run. This is the assertion that turns a silent wrong-tree measurement
    into a refusal."""
    path = shutil.which("chad", path=VENV_BIN + os.pathsep + os.environ["PATH"])
    if not path or not path.startswith(ROOT):
        sys.exit(f"`chad` resolves to {path!r}, which is not inside {ROOT} — refusing to "
                 f"benchmark a different checkout")
    ver = subprocess.run([path, "--version"], capture_output=True, text=True).stdout.strip()
    print(f"chad under test: {path} ({ver})", flush=True)
    return ver


def _resolved_arms(arms: list) -> tuple:
    """Which of the requested arms can actually run: the binary resolves on the PATH a
    subprocess will get (venv first, then the launching shell's — NOT an interactive
    shell's rc file), and, for arms gated by a smoke run, the smoke said yes. Returns
    (runnable, skipped-with-reason) so provenance names what was not measured."""
    path = VENV_BIN + os.pathsep + os.environ["PATH"]
    ok, skipped = [], {}
    for arm in arms:
        exe = shutil.which(_bin(arm), path=path)
        if not exe:
            skipped[arm] = f"`{_bin(arm)}` not on PATH"
            continue
        ok.append(arm)
    return ok, skipped


def _grid(arms, a) -> dict:
    name = getattr(a, "grid_name", "grid")
    arms, skipped = _resolved_arms(arms)
    for arm, why in skipped.items():
        print(f"  {arm:20s} SKIPPED: {why}", flush=True)
    prov = _load("provenance") or {}
    prov.update(_provenance({"chad_cli": _resolved_chad()}))
    prov.setdefault("skipped", {}).update(skipped)
    prov.setdefault("harness_versions", {}).update(_harness_versions(arms))
    _save("provenance", prov)
    rows = _load(name) or []
    rep0 = getattr(a, "rep_label", 0)
    for rep in range(a.reps):
        for task in a.tasks:
            for arm in arms:
                rows.append(run_one(arm, task, rep0 + rep, a))
                _save(name, rows)
    return skipped


def _harness_versions(arms: list) -> dict:
    """`<bin> --version` per arm, for provenance. A harness that changed under the
    benchmark between two nights is otherwise invisible."""
    out = {}
    path = VENV_BIN + os.pathsep + os.environ["PATH"]
    for arm in arms:
        exe = shutil.which(_bin(arm), path=path)
        try:
            r = subprocess.run([exe, "--version"], capture_output=True, text=True,
                               timeout=60)
            out[arm] = (r.stdout + r.stderr).strip().splitlines()[0][:80]
        except Exception as e:  # noqa: BLE001
            out[arm] = f"? ({type(e).__name__})"
    return out


def _from_smoke(a) -> list:
    """Arms the smoke run cleared. Refuses rather than guessing if there is no verdict."""
    v = _load("smoke_verdict")
    if not v:
        sys.exit("--from-smoke: no _runs/smoke_verdict.json — run `smoke` first")
    keep = [x for x in a.arms if x in MLX_ARMS or v.get(x, {}).get("ok")]
    dropped = [x for x in a.arms if x not in keep]
    if dropped:
        print("dropped by smoke: " + ", ".join(f"{x} ({v.get(x, {}).get('why', '?')})"
                                               for x in dropped), flush=True)
    return keep


def arm_llama(a) -> None:
    if getattr(a, "from_smoke", False):
        a.arms = _from_smoke(a)
    proc = start_server(a)
    prox = start_proxy(a)
    try:
        _grid([x for x in a.arms if x in LLAMA_ARMS], a)
    finally:
        # Banked BEFORE the proxy is torn down — `/__audit` lives in the proxy's memory,
        # and `table` runs long after this process is gone.
        try:
            _save("sampler_audit_summary", _assert_sampler_agreement())
        except SystemExit as e:
            print(f"SAMPLER AUDIT FAILED: {e}", flush=True)
        prox.terminate()
        try:
            prox.wait(timeout=20)
        except subprocess.TimeoutExpired:
            prox.kill()
        proc.terminate()
        try:
            proc.wait(timeout=60)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("llama-server stopped", flush=True)


def arm_mlx(a) -> None:
    if subprocess.run(["pgrep", "-f", "llama-server"],
                      capture_output=True, text=True).stdout.strip():
        sys.exit("llama-server is still running — one engine at a time")
    _grid([x for x in a.arms if x in MLX_ARMS], a)


SMOKE_TASK = "grade-school"     # the quickest task in the set (~2 min for a good arm)


def _turns(rep: int | None = None) -> list:
    p = os.path.join(RUNS, "turns.jsonl")
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if rep is None or r.get("rep") == rep:
                out.append(r)
    return out


def smoke(a) -> None:
    """One short task per llama arm, then a verdict per arm: did it reach the server,
    did the server's `timings` come back through the proxy, did it touch the stub.
    Passing the task is NOT required — a harness can honestly fail grade-school — but
    an arm that never spoke to the model, or whose turns cannot be measured, is dropped
    here with its reason written down, rather than spending an hour of the night and
    then being reported as a loss."""
    a.grid_name, a.rep_label, a.tasks = "smoke", -1, [SMOKE_TASK]
    arm_llama(a)
    rows = _load("smoke") or []
    turns = _turns(rep=-1)
    # Merge into whatever verdict is already banked: a re-smoke of the arms that
    # failed last time must not erase the arms that cleared.
    verdict = _load("smoke_verdict") or {}
    for arm in [x for x in a.arms if x in LLAMA_ARMS]:
        rs = [r for r in rows if r["arm"] == arm]
        ts = [t for t in turns if t.get("arm") == arm]
        if not rs:
            verdict[arm] = {"ok": False, "why": "did not run"}
            continue
        r = rs[-1]
        ok200 = [t for t in ts if t.get("status") == 200]
        timed = [t for t in ok200 if isinstance(t.get("timings"), dict)
                 and t["timings"].get("prompt_n") is not None]
        if not ts:
            why = "no request reached the proxy"
        elif not ok200:
            why = f"every request failed (status {sorted({t.get('status') for t in ts})})"
        elif not timed:
            why = "server timings missing from responses"
        elif not (r["passed"] or r.get("stub_changed")):
            why = "never edited the stub"
        else:
            why = ""
        verdict[arm] = {"ok": not why, "why": why, "passed": r["passed"],
                        "wall_s": r["wall_s"], "turns": len(ts), "timed": len(timed),
                        "tail": r.get("tail", "")[-300:]}
        print(f"  smoke {arm:20s} {'OK  ' if not why else 'DROP'} "
              f"turns {len(ts)} timed {len(timed)} "
              f"{'pass' if r['passed'] else 'fail'} {r['wall_s']:.0f}s {why}", flush=True)
    _save("smoke_verdict", verdict)


def table(a) -> None:
    rows = _load("grid") or []
    if not rows:
        print("no rows yet")
        return
    # The first version of this grid printed a clean-looking table whose arms had been
    # sampling differently the whole night. A table is not a report of what happened
    # unless the sampler was the same for everyone, so the check is a precondition of
    # printing rather than a note underneath it.
    forced = _load("sampler_audit_summary")
    if not forced and not a.unverified:
        sys.exit("no sampler_audit_summary.json — this grid's sampling was never "
                 "verified. Re-run the llama phase, or pass --unverified to print "
                 "anyway (and then do not publish it).")
    arms = [x for x in ARMS if any(r["arm"] == x for r in rows)]
    # One denominator per task, shared by every arm. Some exercises (go-counting) have
    # tests that cannot even be COLLECTED against the pristine stub — they import names
    # the solver is expected to define — so an arm that solves one reports more tests
    # than an arm that does not. Taking the largest count anyone saw puts them back on
    # the same scale instead of flattering whoever got furthest.
    denom = {}
    for r in rows:
        denom[r["task"]] = max(denom.get(r["task"], 0), r.get("tests_total", 0))
    print("\n### Harness x engine — same tasks, same weights, same laptop\n")
    hdr = ["Arm", "Passed", "Tests", "Median wall (passed)", "Total prefill",
           "Total generated", "Timeouts"]
    out = []
    selfrep = False
    for arm in arms:
        rs = [r for r in rows if r["arm"] == arm]
        ok = [r for r in rs if r["passed"]]
        walls = sorted(r["wall_s"] for r in ok)
        med = walls[len(walls) // 2] if walls else 0
        # An arm whose tokens are self-reported is marked in the table itself. The
        # asymmetry is real and belongs next to the number, not in a caveat further down.
        mark = "" if arm in LLAMA_ARMS else " *"
        selfrep = selfrep or bool(mark)
        suspect = [r for r in rs if not r.get("counts_clean", True)]
        tp = sum(r.get("tests_passed", 0) for r in rs)
        tt = sum(denom.get(r["task"], 0) for r in rs)
        out.append([arm + mark, f"{len(ok)}/{len(rs)}",
                    f"{tp}/{tt}" if tt else "-",
                    f"{med:.0f}s" if walls else "-",
                    f"{sum(r['prefill'] for r in rs):,}"
                    + ("?" if suspect else ""),
                    f"{sum(r['generated'] for r in rs):,}"
                    + ("?" if suspect else ""),
                    str(sum(1 for r in rs if r["timed_out"]))])
    w = [max(len(str(r[i])) for r in [hdr] + out) for i in range(len(hdr))]
    print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr)) + " |")
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in out:
        print("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
    if selfrep:
        print("\n`*` token counts are the harness's own, not llama-server's: the MLX arm "
              "is in-process\nand no server sees it. Cached tokens are subtracted so both "
              "columns mean the same thing.")
    if any(not r.get("counts_clean", True) for r in rows):
        print("`?` the server was still busy when this arm's counters were read — "
              "treat the count as a floor.")
    if forced:
        print("\nSampler, forced identically on every arm (proxy for the llama arms, "
              "`CHAD_*` for the\nMLX arms, cross-checked): "
              + ", ".join(f"{k} {v}" for k, v in sorted(forced.items())
                          if k in ("temperature", "top_k", "top_p", "min_p",
                                   "presence_penalty", "repeat_penalty")))
    else:
        print("\n**UNVERIFIED SAMPLER** — arms may not have sampled alike. Do not "
              "publish this table.")
    print("\n### Per task (wall seconds if passed; otherwise tests passed, "
          "`T` = timed out)\n")
    hdr2 = ["Task"] + arms
    out2 = []
    for task in a.tasks:
        row = [task]
        for arm in arms:
            rs = [r for r in rows if r["arm"] == arm and r["task"] == task]
            if not rs:
                row.append("-")
            else:
                r = rs[-1]
                if r["passed"]:
                    row.append(f"{r['wall_s']:.0f}")
                else:
                    # A failed cell carries how far it got. "T 29/31" and "T 0/31" are
                    # the same zero on the metric and nothing like each other.
                    n = denom.get(task, 0)
                    tally = f"{r.get('tests_passed', 0)}/{n}" if n else "0/?"
                    row.append(("T " if r["timed_out"] else "x ") + tally)
        out2.append(row)
    w = [max(len(str(r[i])) for r in [hdr2] + out2) for i in range(len(hdr2))]
    print("| " + " | ".join(h.ljust(w[i]) for i, h in enumerate(hdr2)) + " |")
    print("|" + "|".join("-" * (x + 2) for x in w) + "|")
    for r in out2:
        print("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["setup", "smoke", "llama", "mlx", "table"],
                    nargs="?", default="table")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--from-smoke", action="store_true",
                    help="llama: run only the arms _runs/smoke_verdict.json cleared")
    ap.add_argument("--unverified", action="store_true",
                    help="print a table whose sampler agreement was never verified")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--timeout", type=int, default=1200,
                    help="per-task wall cap; a hung harness must not eat the night")
    ap.add_argument("--keep", action="store_true",
                    help="keep the temp workdirs for inspection")
    ap.add_argument("--capture-bodies", action="store_true",
                    help="llama/smoke: have the proxy write every request body under "
                         "_runs/bodies/ (never committed) for body_diff.py")
    a = ap.parse_args(argv)
    a.tasks = [x for x in a.tasks.split(",") if x.strip()]
    a.arms = [x for x in a.arms.split(",") if x.strip()]
    {"setup": setup, "smoke": smoke, "llama": arm_llama, "mlx": arm_mlx,
     "table": table}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
