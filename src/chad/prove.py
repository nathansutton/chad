"""chad prove — a two-minute smoke test on your own machine.

Not a benchmark. The tasks below come from chad's own development suite — the
agent has been tuned against them for months, which is exactly why they make a
good *smoke test* (they should basically never fail on working hardware) and a
meaningless benchmark. The scorecard says so out loud. The honest numbers here
are the ones your machine produces: time-to-first-token, decode speed, and
wall-clock per task.

Contract (devex design 2026-07-11, decisions D22/D24/D27):
- Pins the validated Ornith-9B on every machine (a 32 GB Mac runs the 35B
  day-to-day; the smoke test still runs the 9B — the copy says so).
- Offline guard engages only AFTER the model cache check/download: HF_HUB_OFFLINE
  plus a socket guard in this process (children only inherit the env var — the
  scorecard claims "library level", nothing stronger).
- Verifier hardening: check scripts are re-written from the read-only sources in
  this module before they run, so an agent edit to a seeded check.py can't spoof
  a pass.
- Exit codes: 0 = all tasks pass, 1 = any task failed, 2 = preflight/environment
  stop. results.json is written to the invoking directory, overwritten per run.
- Share snippet is offered only on a 100% pass.

Gate provenance: task set = survivors of the 2026-07-11 falsification gate
(normal + memlock conditions, 3/3 each, wall <= 60 s; wifi-off waived — all
survivors are filesystem-only by construction). See the gate artifact in the
private workshop (_runs/prove_gate_main/VERDICT.txt).
"""
import contextlib
import io
import json
import os
import socket
import subprocess
import sys
import tempfile
import time

# The proof tasks, ported verbatim from the dev eval suite (disclosed above).
# `check` is the verifier SOURCE — re-written to `check_path` immediately before
# it runs, whether or not the task also seeds it as a visible spec for the agent.
# Ordering matters: the first task carries the two-minute promise, so it's the
# fastest reliable survivor.
TASKS = [
    {
        "name": "casual_typo_fix",
        "files": {"greet.py": "def greet(name):\n    return 'Helo, ' + name\n"},
        "prompt": "greet.py has a typo — it says 'Helo' instead of 'Hello'. "
                  "sort that out for me.",
        "check_path": "_prove_check.py",  # never shown to the agent
        "check": "import greet\n"
                 "assert greet.greet('Sam') == 'Hello, Sam', repr(greet.greet('Sam'))\n"
                 "print('REGRESSION_OK')\n",
        "expect": "REGRESSION_OK",
    },
    {
        "name": "add_function",
        "files": {
            "mathx.py": "def add(a, b):\n    return a + b\n",
            "check.py":
                "from mathx import add, factorial\n"
                "assert add(2, 3) == 5\n"
                "assert factorial(0) == 1 and factorial(5) == 120\n"
                "print('ALL CHECKS PASS')\n",
        },
        "prompt": "Add a factorial(n) function to mathx.py (factorial(0)==1). "
                  "check.py verifies it.",
        "check_path": "check.py",  # visible spec; re-seeded before verify
        "check":
            "from mathx import add, factorial\n"
            "assert add(2, 3) == 5\n"
            "assert factorial(0) == 1 and factorial(5) == 120\n"
            "print('ALL CHECKS PASS')\n",
        "expect": "ALL CHECKS PASS",
    },
    {
        "name": "locate_and_fix",
        "files": {
            "config.py": "DEBUG = False\nVERSION = '1.0'\n",
            "utils.py": "def clamp(x, lo, hi):\n    return max(lo, min(hi, x))\n",
            "discount.py":
                "def apply_discount(price, pct):\n"
                "    # pct is a percentage like 20 for 20% off\n"
                "    return price - pct  # bug: subtracts pct as dollars, not percent\n",
            "check.py":
                "from discount import apply_discount\n"
                "assert apply_discount(100, 20) == 80\n"
                "assert apply_discount(50, 10) == 45\n"
                "print('ALL CHECKS PASS')\n",
        },
        "prompt": "A pricing bug: applying a 20% discount to $100 should give $80, "
                  "but it's wrong. Find the function responsible somewhere in this "
                  "project and fix it. check.py verifies it.",
        "check_path": "check.py",
        "check":
            "from discount import apply_discount\n"
            "assert apply_discount(100, 20) == 80\n"
            "assert apply_discount(50, 10) == 45\n"
            "print('ALL CHECKS PASS')\n",
        "expect": "ALL CHECKS PASS",
    },
    {
        "name": "fix_bug_midtext",
        "files": {
            "construct.py":
                "def construct_addendum(text):\n"
                "    # only handles ADDENDUM: at the very start\n"
                "    if text.startswith(\"ADDENDUM:\"):\n"
                "        return text[len(\"ADDENDUM:\"):].strip()\n"
                "    return text\n",
            "check.py":
                "from construct import construct_addendum\n"
                "assert construct_addendum('ADDENDUM: hi') == 'hi'\n"
                "assert construct_addendum('intro ADDENDUM: consider a follow up') "
                "== 'consider a follow up'\n"
                "print('ALL CHECKS PASS')\n",
        },
        "prompt": "Help me refactor construct_addendum in construct.py so it is "
                  "robust to 'ADDENDUM:' appearing mid-text (e.g. 'intro ADDENDUM: "
                  "consider a follow up' -> 'consider a follow up'). check.py "
                  "verifies it.",
        "check_path": "check.py",
        "check":
            "from construct import construct_addendum\n"
            "assert construct_addendum('ADDENDUM: hi') == 'hi'\n"
            "assert construct_addendum('intro ADDENDUM: consider a follow up') "
            "== 'consider a follow up'\n"
            "print('ALL CHECKS PASS')\n",
        "expect": "ALL CHECKS PASS",
    },
]

TASK_TIMEOUT_S = 120  # 2x the gate's 60 s bar: past that it's a fail, stop burning time

_LOCAL_HOSTS = ("127.0.0.1", "::1", "localhost")


def _install_socket_guard():
    """Best-effort, library-level offline guard for THIS process: refuse outbound
    connections to anything but localhost. Child processes the agent spawns are only
    covered by HF_HUB_OFFLINE in their env — the scorecard's wording matches that.
    Returns an uninstall callable (used by tests)."""
    real_connect = socket.socket.connect

    def guarded_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if isinstance(host, (bytes, str)):
            h = host.decode() if isinstance(host, bytes) else host
            # AF_UNIX paths and localhost stay open; everything else is refused.
            if not (h.startswith("/") or h in _LOCAL_HOSTS):
                raise OSError(f"[chad prove] network access blocked at the library "
                              f"level (offline smoke test): refused connect to {h}")
        return real_connect(self, address)

    socket.socket.connect = guarded_connect

    def uninstall():
        socket.socket.connect = real_connect
    return uninstall


def _verify(task):
    """Re-seed the check script from its read-only source, then run it. The re-write
    is the anti-spoof: an agent edit to a seeded check.py is overwritten here."""
    with open(task["check_path"], "w") as f:
        f.write(task["check"])
    vr = subprocess.run([sys.executable, task["check_path"]],
                        capture_output=True, text=True, timeout=60)
    out = (vr.stdout or "") + (vr.stderr or "")
    return vr.returncode == 0 and task["expect"] in out


def _hardware():
    """'Apple M2, 24 GB' — chip + RAM as the OS reports them (scorecard label)."""
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip()
    except Exception:  # noqa: BLE001 — a label, never a crash
        chip = "unknown chip"
    try:
        gb = int(subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True).strip()) / (1024 ** 3)
        ram = f"{gb:.0f} GB"
    except Exception:  # noqa: BLE001
        ram = "? GB"
    return f"{chip}, {ram}"


def _run_one(engine, task, capture_ttft=False):
    """Run a single proof task in a temp repo. Returns the result row."""
    from .agent import Agent
    start_dir = os.getcwd()
    tmp = tempfile.mkdtemp(prefix=f"prove_{task['name']}_")
    ttft = [None]
    real_generate = engine.generate

    def timed_generate(*a, **k):
        t0 = time.time()
        out = real_generate(*a, **k)
        # engine.generate returns (text, stats); stats.prefill_s = time to first
        # token of the step. TTFT = that, for the very first step only.
        if ttft[0] is None:
            try:
                ttft[0] = out[1].prefill_s
            except Exception:  # noqa: BLE001 — stats shape drift never kills the run
                ttft[0] = time.time() - t0
            engine.generate = real_generate  # unhook after first capture
        return out

    try:
        os.chdir(tmp)
        for fname, content in task["files"].items():
            with open(fname, "w") as f:
                f.write(content)
        subprocess.run(["git", "init", "-q"], capture_output=True)
        subprocess.run(["git", "add", "-A"], capture_output=True)
        deadline = [None]
        agent = Agent(engine, mode="auto", thinking=True,
                      should_stop=lambda: (deadline[0] is not None
                                           and time.time() > deadline[0]))
        if capture_ttft:
            engine.generate = timed_generate
        t0 = time.time()
        deadline[0] = t0 + TASK_TIMEOUT_S
        with contextlib.redirect_stdout(io.StringIO()):
            agent.run_turn(task["prompt"], stream=False)
        wall = time.time() - t0
        passed = _verify(task)
        return {
            "name": task["name"], "passed": passed, "wall": round(wall, 1),
            "timed_out": wall >= TASK_TIMEOUT_S,
            "tok_per_s": round(agent.tok_per_s, 1),
            "gen_tokens": agent.gen_tokens,
            "ttft_s": round(ttft[0], 2) if ttft[0] is not None else None,
        }
    finally:
        engine.generate = real_generate
        os.chdir(start_dir)


def _scorecard(results, meta):
    """Render the scorecard. Always prints, including on failures — a failed proof
    with a plain explanation is recoverable; a stack trace is not."""
    lines = []
    ok = sum(1 for r in results if r["passed"])
    lines.append("")
    lines.append("chad prove — smoke test scorecard 🗿")
    lines.append("=" * 54)
    lines.append("These tasks come from chad's own development suite (the")
    lines.append("agent is tuned on them) — this is a proof of life on YOUR")
    lines.append("hardware, not a benchmark. The honest numbers are yours:")
    lines.append("")
    for r in results:
        mark = "PASS" if r["passed"] else ("TIMEOUT" if r["timed_out"] else "FAIL")
        lines.append(f"  {r['name']:24s} {mark:7s} {r['wall']:6.1f}s"
                     f"  {r['tok_per_s']:5.1f} tok/s decode")
        if not r["passed"]:
            lines.append("    task failed — transcript: ~/.chad/session.log · "
                         "common causes: memory pressure from other apps, thermal "
                         "throttling. Tasks are designed to be repeatable — close "
                         "memory-heavy apps and re-run.")
    lines.append("")
    ttft = next((r["ttft_s"] for r in results if r.get("ttft_s") is not None), None)
    if ttft is not None:
        lines.append(f"  time to first token   {ttft:.2f}s (first task — includes "
                     "prefilling the initial prompt)")
    lines.append(f"  model load            {meta['load_s']:.1f}s")
    lines.append(f"  hardware              {meta['hardware']}")
    lines.append(f"  model                 {meta['model']} ({meta['download_mode']})")
    if meta.get("big_ram_note"):
        lines.append(f"  note                  {meta['big_ram_note']}")
    lines.append("  offline               network access blocked at the library "
                 "level for task execution")
    lines.append(f"  result                {ok}/{len(results)} tasks passed")
    if ok == len(results):
        lines.append("")
        lines.append("share it (copy below the line):")
        lines.append("-" * 54)
        lines.append("```")
        lines.append(f"chad prove 🗿  {ok}/{len(results)} tasks · "
                     f"{meta['hardware']}")
        for r in results:
            lines.append(f"  {r['name']}: {r['wall']:.1f}s @ "
                         f"{r['tok_per_s']:.1f} tok/s")
        lines.append("  local, offline, zero-config — "
                     "https://github.com/nathansutton/chad")
        lines.append("```")
    return "\n".join(lines)


def run(args):
    """Entry point (dispatched from cli.main on the literal task `prove`).
    Returns the process exit code."""
    from . import cli
    if args.backend != "mlx":
        sys.stderr.write(
            f"chad prove verifies the local in-process engine; --backend "
            f"{args.backend} has nothing to prove (it measures a remote server, "
            "not this machine). Run it without --backend.\n")
        return 2
    try:
        cli._preflight("mlx")
    except SystemExit:
        return 2
    if os.environ.get("CHAD_MODEL"):
        sys.stderr.write("[prove pins the validated 9B — CHAD_MODEL is ignored "
                         "for this run]\n")
    model_id = cli._HF_9B
    invoking_dir = os.getcwd()

    from huggingface_hub import try_to_load_from_cache
    cached = isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    try:
        cli._ensure_model(model_id)  # consent + disk preflight + resumable download
    except SystemExit:
        return 2
    download_mode = ("model already cached, offline guard engaged" if cached
                     else "model downloaded this session, offline guard engaged "
                          "for task execution")

    # Offline guard engages only now — after the cache check / download.
    os.environ["HF_HUB_OFFLINE"] = "1"
    uninstall_guard = _install_socket_guard()

    big_ram_note = None
    ram = cli._detect_ram_gb()
    if ram is not None and ram >= cli._BIG_RAM_GB:
        big_ram_note = ("smoke test runs the validated 9B; your machine runs "
                        "the 35B day-to-day")
        sys.stderr.write(f"[{big_ram_note}]\n")

    from .engine import Engine
    # Same engine configuration chad ships: the on-disk KV checkpoint of the stable
    # system+tools prefix (~/.cache/chad/kv) is part of the product's first-token
    # story — prove without it would report a cold prefill no real session pays.
    eng = Engine(model_id=model_id, draft_id=None,
                 cache_dir=os.path.expanduser("~/.cache/chad/kv"))
    sys.stderr.write(f"loading {os.path.basename(model_id.rstrip('/'))} "
                     "[prove: pinned 9B] ...\n")
    try:
        load_s = eng.load()
    except Exception as e:  # noqa: BLE001 — same guidance path as the main CLI
        uninstall_guard()
        try:
            cli._fail_model_load(model_id, e)
        except SystemExit:
            return 2
        return 2

    results = []
    try:
        for i, task in enumerate(TASKS):
            sys.stderr.write(f"task {i + 1}/{len(TASKS)}: {task['name']} ...\n")
            r = _run_one(eng, task, capture_ttft=(i == 0))
            sys.stderr.write(f"  {'PASS' if r['passed'] else 'FAIL'} "
                             f"{r['wall']:.1f}s\n")
            results.append(r)
    finally:
        uninstall_guard()

    meta = {
        "model": os.path.basename(model_id.rstrip("/")),
        "download_mode": download_mode, "hardware": _hardware(),
        "load_s": load_s, "big_ram_note": big_ram_note,
    }
    print(_scorecard(results, meta))
    out_path = os.path.join(invoking_dir, "results.json")
    with open(out_path, "w") as f:  # overwritten per run; scorecard is the artifact
        json.dump({"meta": meta, "results": results}, f, indent=2)
    sys.stderr.write(f"\n[raw rows: {out_path}]\n")
    return 0 if all(r["passed"] for r in results) else 1
