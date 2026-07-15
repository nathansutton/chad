#!/usr/bin/env python3
"""chad — a local, MLX-backed, Claude-Code-style coding agent.

One model (Ornith — 35B on big Macs, 9B on small), one entrypoint, run with uv:

    uv run chad                                # interactive full-screen TUI
    uv run chad "fix the bug in greet.py"      # one-shot, headless
    uv run chad -c                             # resume this directory's conversation

Rare long-session knobs live in env vars — see README "Advanced".
"""
import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time

from . import config, guardrails, levers
from .agent import Agent, repl
from .engine import Engine

# Package dir is src/chad/; the project root (two levels up) is the dev clone. If a
# locally-built weights tree exists at <root>/models/ it's preferred (see _pick_model);
# otherwise — the normal case — the default model ships from Hugging Face and downloads
# into the shared HF cache on first use. Point CHAD_MODEL at any local dir to override.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))

# The shipped models, on Hugging Face. Naming follows Unsloth's dynamic-quant
# convention so the quant scheme is recognizable (UD = Unsloth Dynamic; Q2_K_XL =
# 2-bit experts with a high-bit backbone/router), plus an -MLX suffix for format
# discoverability. The quant itself is MLX group-64 affine, not llama.cpp Q2_K
# k-quants — the model card says so; the tag is for recognition, not bit-for-bit
# equivalence.
_HF_35B = "nathansutton/Ornith-1.0-35B-UD-Q2_K_XL-MLX"   # default: 35B MoE, ~13.4 GB resident
_HF_9B = "nathansutton/Ornith-1.0-9B-UD-Q4_K_XL-MLX"     # low-RAM fallback, ~5 GB resident
# A dev clone that already built the weights locally should use them rather than
# re-download — prefer these dirs when present.
_LOCAL_35B = os.path.join(_PROJECT_ROOT, "models", "Ornith-1.0-35B-dyn2-q2_down3")
_LOCAL_9B = os.path.join(_PROJECT_ROOT, "models", "Ornith-1.0-9B-4bit-awq")
# The 35B (2-bit experts, 3-bit expert down-projections, 6-bit backbone) is ~13.4 GB
# resident + KV + runtime, and the KV grows across a long agentic turn. On a 24 GB Mac
# the Metal wired limit (~2/3 RAM ≈ 16 GB) minus the OS and whatever else is open
# leaves too little headroom — dogfooding SIGKILLed the 35B mid-turn there before the
# plan-075 clamps/governor landed. The floor stays 32 GB until the 24 GB profile is
# proven end-to-end (plan 075 WS1.6); 24 GB and below fall back to the 9B.
_BIG_RAM_GB = 31.5


# These are the STRICT siblings of config.env_int/env_float: a non-numeric value raises
# (int()/float() propagate) rather than warning-and-defaulting. test_cli.py pins that
# contract, and a garbled CHAD_MAX_CONTEXT/CHAD_KV_BITS should fail loud at startup rather
# than silently reverting to the model default. Kept inline here on purpose; the lenient
# config helpers back the mid-run budget knobs in agent.py instead.
def _env_int(name):
    val = os.environ.get(name)
    return int(val) if val else None


def _env_float(name):
    val = os.environ.get(name)
    return float(val) if val else None


def _version_string():
    """chad <version> (<vcs commit>) — commit resolves for git installs via
    dist-info/direct_url.json, or from the dev clone's .git; absent otherwise."""
    from . import __version__
    detail = ""
    try:
        import json
        from importlib.metadata import distribution
        raw = distribution("chad").read_text("direct_url.json") or ""
        commit = json.loads(raw).get("vcs_info", {}).get("commit_id", "") if raw else ""
        if not commit and os.path.isdir(os.path.join(_PROJECT_ROOT, ".git")):
            commit = subprocess.check_output(
                ["git", "-C", _PROJECT_ROOT, "rev-parse", "--short", "HEAD"],
                text=True, stderr=subprocess.DEVNULL).strip()
        if commit:
            detail = f" ({commit[:12]})"
    except Exception:  # noqa: BLE001 — version detail is best-effort, never fatal
        pass
    return f"chad {__version__}{detail}"


def _host_avail_bytes():
    """Host-wide reclaimable memory (free + inactive pages) via vm_stat, or None.
    The Metal `max_recommended_working_set_size` is a static per-GPU number that
    cannot see what Docker/harbor/browsers are holding; on a 24 GB box running the
    benchmark stack beside the model, PHYSICAL pressure is what jetsam kills on, so
    the context budget must respect it too (plan 075 WS1.4)."""
    try:
        out = subprocess.check_output(["vm_stat"], text=True)
        page = 16384
        m = re.search(r"page size of (\d+)", out)
        if m:
            page = int(m.group(1))
        pages = 0
        for key in ("Pages free", "Pages inactive", "Pages speculative"):
            m = re.search(rf"{key}:\s+(\d+)", out)
            if m:
                pages += int(m.group(1))
        return pages * page if pages else None
    except Exception:  # noqa: BLE001 — a pressure probe must never be the crash
        return None


def ram_aware_ctx_limit(eff_ctx, budget_bytes, active_bytes, kv_bytes_per_token,
                        reserve_gb=1.5, safety=0.90, gen_margin=2048, floor=8192,
                        host_avail_bytes=None, slope_factor=1.75):
    """Plan 036: largest prompt-token budget (= the compaction trigger) that keeps the
    growing KV cache inside a safe slice of the Metal *recommended working set*, given the
    model's already-resident footprint and the *measured* per-token KV cost. Pure +
    measured — replaces the magic `CTX_CAP = 120_000`, which was an OOM guard set blind to
    the real per-token cost (it over-compacts: on a 24 GB M4 Pro the 20 KiB/token hybrid
    cache fits ~175 k tokens, not 120 k).

    `budget*safety` leaves a headroom band below Apple's recommendation; subtract the
    resident model+SSM floor (`active_bytes`) and a scratch reserve for prefill/decode
    buffers to get the bytes free for KV; divide by `kv_bytes_per_token` for the token
    ceiling. Capped at `eff_ctx − gen_margin` (the model's real window) and floored so a
    tight box still gets a usable window. Self-calibrates per machine (16 GB → small,
    64 GB → near the window). Returns None if inputs are unusable so the caller can keep
    the old fixed cap."""
    if not (budget_bytes and kv_bytes_per_token and active_bytes):
        return None
    usable = budget_bytes * safety - active_bytes - reserve_gb * 1e9
    # The Metal budget is blind to other processes' physical pressure (Docker VM,
    # harbor, browsers). When the host's reclaimable band is tighter than the Metal
    # band, IT is the binding constraint — the KV cache grows into physical pages
    # 1:1, and jetsam kills on physical pressure, not on Metal accounting.
    if host_avail_bytes:
        usable = min(usable, host_avail_bytes * 0.85 - reserve_gb * 1e9)
    if usable <= 0:
        return floor
    # Peak memory grows FASTER than the KV cache alone: prefill/decode scratch also
    # scales with resident context (2026-07-12 ram_safety_check fit on the 35B, fused
    # wheel: 35.7 KB/token all-in vs 20.5 KB/token KV — the raw KV divisor picked a
    # 175k trigger that extrapolated to 102.9% of budget, the WS1.6 FAIL). The fixed
    # `reserve_gb` cannot cover a term that grows per-token, so fold it into the
    # divisor. 1.75 is the 35B measurement; unmeasured on the 9B, where it errs safe
    # (over-compaction costs a re-prefill, undershoot costs a jetsam kill).
    ram_ctx = int(usable / (kv_bytes_per_token * slope_factor))
    return max(floor, min(eff_ctx - gen_margin, ram_ctx))


def _compute_ctx_limit(eng):
    """The auto-compaction threshold for a loaded engine. On this non-trimmable hybrid
    cache, compaction forces a full body re-prefill (plan 035: ~79 % of all prefill), so
    we compact as rarely as RAM safely allows: size the trigger from the live Metal
    budget + the model's *measured* per-token KV cost (plan 036) instead of a blind 120 k
    cap that over-compacts. CHAD_CTX_LIMIT still wins (evals/tests); CHAD_CTX_RESERVE_GB
    tunes the scratch headroom. Falls back to the old fixed cap if the memory APIs or the
    KV measurement are unavailable. Needs eng.load() to have run (reads effective_ctx +
    kv_bytes_per_token)."""
    ctx_limit = _env_int("CHAD_CTX_LIMIT")
    if not ctx_limit:
        try:
            import mlx.core as mx
            # Subtract the resident KV from active so a LIVE recheck (mid-session,
            # cache already grown) measures the same model floor the startup call
            # does — otherwise the limit would shrink as the cache approaches it.
            active_floor = (mx.get_active_memory()
                            - eng.kv_bytes_per_token * getattr(eng, "resident_tokens", 0))
            ctx_limit = ram_aware_ctx_limit(
                eng.effective_ctx,
                mx.device_info()["max_recommended_working_set_size"],
                active_floor, eng.kv_bytes_per_token,
                reserve_gb=_env_float("CHAD_CTX_RESERVE_GB") or 1.5,
                host_avail_bytes=_host_avail_bytes(),
                slope_factor=_env_float("CHAD_CTX_SLOPE_FACTOR") or 1.75)
        except Exception:  # noqa: BLE001 — never let memory probing break startup
            ctx_limit = None
    if not ctx_limit:
        ctx_limit = min(max(4096, eng.effective_ctx - 2048), 120_000)  # old fixed cap
    return ctx_limit


def _preflight(backend="mlx"):
    """chad's default in-process engine runs only on Apple Silicon — MLX has no CPU/CUDA
    build. Hard-stop with a human message instead of letting `uv sync`/import fail
    cryptically elsewhere. The remote backends (`--backend llama/openai`) load NO MLX —
    only a tokenizer plus HTTP — so they run anywhere (e.g. inside a Linux benchmark
    container reaching a remote server); skip the Apple-Silicon gate for them."""
    if backend in ("openai", "llama"):
        return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        sys.stderr.write(
            "chad: requires an Apple Silicon Mac (arm64 macOS).\n"
            f"  detected: {platform.system()} {platform.machine() or '?'}\n"
            "  MLX ships no CPU/CUDA build — there is no supported non-Apple path.\n"
            "  (For a remote engine on this host, use --backend llama/openai.)\n")
        sys.exit(1)


def _detect_ram_gb():
    """Physical RAM in GiB via sysctl, or None if it can't be read."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1024 ** 3)
    except Exception:  # noqa: BLE001 — any failure → caller picks the safe (smaller) model
        return None


def _pick_model():
    """Resolve the model id and a human label for *why* it was chosen.

    Order: explicit CHAD_MODEL override → RAM-aware default (35B on big boxes, 9B on
    small) → prefer a locally-built models/ dir over the HF repo when it exists.
    """
    env = config.env_str("CHAD_MODEL")
    if env:
        return env, "CHAD_MODEL override"
    ram = _detect_ram_gb()
    if ram is None:
        # RAM unreadable -> the safe (smaller) model: a wrong 9B costs capability, a
        # wrong 35B costs a 12 GB download and possibly an OOM'd first session.
        local, repo = _LOCAL_9B, _HF_9B
        why = "9B (RAM undetectable — choosing the safe smaller model; " \
              "set CHAD_MODEL to override)"
    elif ram < _BIG_RAM_GB:
        local, repo = _LOCAL_9B, _HF_9B
        why = f"9B (default; {ram:.0f} GB RAM < {_BIG_RAM_GB:.0f} GB, 35B would be tight)"
    else:
        local, repo = _LOCAL_35B, _HF_35B
        why = "35B (default)"
    return (local if os.path.isdir(local) else repo), why


def _model_download_gb(model_id):
    """Approximate download size in GiB for the shipped models (for the disk preflight
    and the confirm prompt — display honesty, not accounting)."""
    return 12.0 if "35B" in model_id else 5.0


def _free_disk_gb(path):
    """Free GiB on the filesystem holding `path` (climbing to the nearest existing
    parent), or None if it can't be read — a preflight must never be the crash."""
    import shutil
    p = os.path.expanduser(path)
    while p and not os.path.exists(p):
        parent = os.path.dirname(p)
        if parent == p:
            break
        p = parent
    try:
        return shutil.disk_usage(p).free / (1024 ** 3)
    except OSError:
        return None


def _ensure_model(model_id):
    """If model_id is a HF repo id not yet in the local cache, confirm and download it
    into ~/.cache/huggingface (shared, resumable, paid once per machine). Local dirs
    and already-cached repos return immediately. Headless (no TTY) auto-downloads.
    Preflights free disk BEFORE starting: a 12 GB download that dies at 70% on a full
    disk is the worst first-run outcome (devex review T2)."""
    if os.path.isdir(model_id):
        return  # a local path — nothing to fetch
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    if isinstance(try_to_load_from_cache(model_id, "config.json"), str):
        return  # already in the HF cache
    need_gb = _model_download_gb(model_id)
    hf_home = os.environ.get("HF_HOME", "~/.cache/huggingface")
    free_gb = _free_disk_gb(hf_home)
    # need + 2 GB headroom: the HF cache writes temp blobs beside the final files.
    if free_gb is not None and free_gb < need_gb + 2.0:
        sys.stderr.write(
            f"\nchad: not enough free disk for the model download\n"
            f"  cause: '{model_id}' needs ~{need_gb:.0f} GB (+2 GB headroom); "
            f"{free_gb:.1f} GB free at {hf_home}\n"
            "  fix:   free up space, or clear old model revisions: `hf cache ls` /\n"
            "         `hf cache rm` (older CLIs: `huggingface-cli delete-cache`).\n"
            "         Or point CHAD_MODEL at a local model dir on another volume.\n")
        sys.exit(1)
    size = f"~{need_gb:.0f} GB"
    sys.stderr.write(
        f"\nchad needs the model '{model_id}' "
        f"({size} — minutes on fast fiber, ~20 min on 100 Mbit; resumable).\n"
        "It downloads once into ~/.cache/huggingface and is reused across projects.\n")
    if sys.stdin.isatty():
        ans = input("Download now? [Y/n] ").strip().lower()
        if ans and ans not in ("y", "yes"):
            sys.stderr.write(
                "Aborted. Set CHAD_MODEL to a local model dir to skip the download.\n")
            sys.exit(1)
        sys.stderr.write(
            "While you wait: chad works best run from inside a project, on a scoped\n"
            'ask — "fix the failing test in tests/test_x.py" lands; "improve my\n'
            'codebase" flails. (More: README → Quickstart.)\n')
    else:
        sys.stderr.write("[headless: downloading automatically]\n")
    try:
        snapshot_download(model_id)  # tqdm progress to stderr
    except Exception as e:  # noqa: BLE001 — offline / gated / typo'd repo / full disk → guidance, not a traceback
        no_space = isinstance(e, OSError) and getattr(e, "errno", None) == 28
        extra = ("  note:  the disk filled up mid-download — free space and re-run "
                 "(it resumes).\n" if no_space or "No space left" in str(e) else "")
        sys.stderr.write(
            f"\nchad: could not download '{model_id}'\n"
            f"  cause: {type(e).__name__}: {e}\n" + extra +
            "  fix:   check your connection; if the repo is gated, run `hf auth login`.\n"
            "         Or point CHAD_MODEL at a local model dir you've already built.\n")
        sys.exit(1)


def _fail_model_load(model_id, err):
    """Turn a raw model-load traceback into problem / cause / fix and exit."""
    sys.stderr.write(f"\nchad: could not load model '{model_id}'\n")
    sys.stderr.write(f"  cause: {type(err).__name__}: {err}\n")
    if os.path.isdir(model_id):
        sys.stderr.write(
            "  fix:   the local model dir looks incomplete or corrupt. Re-build it, or\n"
            "         unset CHAD_MODEL to fall back to the Hugging Face download.\n")
    else:
        sys.stderr.write(
            "  fix:   a partial/corrupt download or not enough free RAM. Re-run (the HF\n"
            f"         download resumes), or try the smaller model: CHAD_MODEL={_HF_9B}\n")
    sys.exit(1)


def _maybe_home_dir_note():
    """chad snapshots the working directory into context at startup, so the home dir is
    rarely the intended workspace. Nudge once — no exit, no behavior change. Home-dir
    only: guessing "is this a project" from marker files false-positives on legit
    non-git work dirs."""
    if os.getcwd() == os.path.expanduser("~"):
        sys.stderr.write(
            "note: running in your home directory — chad works best inside a project "
            "(cd into one and rerun).\n")


def _pick_session(items):
    """Prompt the user to pick one of `items` (from session.list_sessions) by number.
    Returns the chosen item, or None to start fresh. Requires a TTY — the caller
    guards that before calling."""
    from . import session
    sys.stderr.write("Resume which session? (this directory's recent sessions)\n")
    for i, it in enumerate(items, 1):
        sys.stderr.write(f"  {i}. {session.describe(it)}\n")
    try:
        raw = input("session number (blank to cancel): ").strip()
    except (EOFError, KeyboardInterrupt):
        return None
    if not raw:
        return None
    try:
        n = int(raw)
    except ValueError:
        sys.stderr.write("not a number; starting fresh\n")
        return None
    if 1 <= n <= len(items):
        return items[n - 1]
    sys.stderr.write("out of range; starting fresh\n")
    return None


def main():
    ap = argparse.ArgumentParser(
        prog="chad",
        description="Local MLX-backed coding agent (Ornith). Run with `uv run chad`.",
    )
    ap.add_argument("--version", action="version", version=_version_string())
    ap.add_argument("task", nargs="?",
                    help="one-shot task to run headless and exit; omit for the interactive TUI")
    ap.add_argument("-c", "--continue", dest="cont", action="store_true",
                    help="resume the most recent saved conversation for this directory")
    ap.add_argument("--resume", action="store_true",
                    help="list this directory's recent sessions and pick one by number "
                         "(resuming forks: the picked session is never overwritten)")
    ap.add_argument("--plan", action="store_true",
                    help="start in read-only plan mode (investigate and propose, no edits)")
    ap.add_argument("--yolo", action="store_true",
                    help="auto-approve bash/write/edit (skip confirmation prompts)")
    ap.add_argument("--no-think", action="store_true",
                    help="skip the model's <think> reasoning blocks (faster)")
    ap.add_argument("--think-budget", type=int, default=None, dest="think_budget",
                    help="soft-cap each step's <think> run at N tokens, then force-close "
                         "it and continue (escalates when stuck); off by default. Also "
                         "settable via CHAD_THINK_BUDGET.")
    ap.add_argument("--think-ceiling", type=int, default=None, dest="think_ceiling",
                    help="close-and-continue (plan 086): force-close a runaway <think> past "
                         "N tokens and keep decoding the action IN THE SAME STEP (vs "
                         "--think-budget, which ends the step). A high pathological cap "
                         "(~6000); off by default. Also settable via CHAD_THINK_CEILING.")
    ap.add_argument("--turn-budget-tokens", type=int, default=None, dest="turn_budget_tokens",
                    help="runaway-turn governor: end a turn once it has burned N cumulative "
                         "prefill tokens WITHOUT landing+verifying a change (nudges at ~50%%, "
                         "banks a progress note and stops at ~80%%). Defaults to 3× the context "
                         "limit; CHAD_NO_GOVERNOR=1 disables. Also CHAD_TURN_BUDGET_TOKENS.")
    ap.add_argument("--turn-budget-s", type=float, default=None, dest="turn_budget_s",
                    help="wall-clock variant of --turn-budget-tokens (seconds); off by "
                         "default (interactive: the human is the wall clock). Also "
                         "CHAD_TURN_BUDGET_S.")
    ap.add_argument("--auto-continue", type=int, default=None, dest="auto_continue",
                    help="on a one-shot run, if the governor or a guardrail hard-stops a "
                         "turn, relaunch a FRESH turn (cleared context) seeded with the "
                         "progress note, up to N times. Default: 2 on an unattended "
                         "(auto-approve) run, 0 otherwise; pass 0 to disable.")
    ap.add_argument("--review-pass", action="store_true", dest="review_pass",
                    help="early-finish self-review (plan 085): on a one-shot run that ends "
                         "CLEANLY with >30%% of --turn-budget-s still unspent, relaunch ONE "
                         "fresh-context turn to independently verify the deliverables and fix "
                         "any mismatch. Off by default (needs --turn-budget-s to arm); also "
                         "CHAD_REVIEW_PASS=1.")
    # Backend selection (plan 046/047 spikes). Default 'mlx' is the in-process engine and
    # the whole point of chad; 'openai' drives the SAME harness against an OpenAI-compatible
    # endpoint (honest degradations apply — see openai_engine.py); 'llama' drives a
    # llama.cpp server's raw /completion with token-id prompts + real cache telemetry
    # (see completion_engine.py). The MLX path below is untouched when --backend is unset.
    ap.add_argument("--backend", choices=("mlx", "openai", "llama"), default="mlx",
                    help="inference backend: 'mlx' (default, in-process KV cache), "
                         "'openai' (spike: OpenAI-compatible /v1/chat/completions), or "
                         "'llama' (spike: llama.cpp raw /completion, token-id prompts). "
                         "The remote backends require --base-url.")
    ap.add_argument("--serve", action="store_true",
                    help="serve the in-process engine as an OpenAI-compatible endpoint "
                         "(/v1/chat/completions) instead of running the agent. This is "
                         "the TB2 Mac arm's server: unlike mlx_lm.server it keeps the "
                         "prefix KV cache warm across requests (plan 075 WS3). One "
                         "request at a time; pair with --n-concurrent-agents 1.")
    ap.add_argument("--host", dest="serve_host", default="127.0.0.1",
                    help="--serve bind address (default 127.0.0.1; use 0.0.0.0 to let "
                         "Docker containers reach it via host.docker.internal)")
    ap.add_argument("--port", dest="serve_port", type=int, default=8090,
                    help="--serve port (default 8090)")
    ap.add_argument("--base-url", dest="base_url", default=None,
                    help="remote-backend base URL: for --backend openai an OpenAI-compatible "
                         "base (e.g. http://localhost:8080/v1, also CHAD_OPENAI_BASE_URL); "
                         "for --backend llama the llama-server origin (e.g. "
                         "http://192.168.87.25:8081, also CHAD_LLAMA_BASE_URL).")
    ap.add_argument("--tokenizer", dest="tokenizer", default=None,
                    help="HF repo/dir whose tokenizer to load for a remote backend, when the "
                         "served model id has no tokenizer files (e.g. a GGUF repo). Must "
                         "share the served model's vocab. Also CHAD_TOKENIZER.")
    ap.add_argument("--api-key-env", dest="api_key_env", default=None,
                    help="name of the env var holding the API key for a remote backend; the "
                         "key is read from that var, never passed on the command line.")
    ap.add_argument("--repl", action="store_true",
                    help="plain line REPL instead of the full-screen TUI")
    # Back-compat: -p/--prompt was the old one-shot spelling, now the positional task.
    ap.add_argument("-p", "--prompt", dest="prompt_flag", help=argparse.SUPPRESS)
    ap.add_argument("--levers", action="store_true",
                    help="print the harness lever registry as JSON and exit (the "
                         "ablation driver enumerates this instead of hardcoding names)")
    args = ap.parse_args()

    # Before _preflight: an ablation driver enumerating levers should not need an
    # Apple-Silicon box or a loadable model just to read the registry.
    if args.levers:
        print(json.dumps({"levers": levers.as_dict(), "groups": levers.groups(),
                          "active": levers.active()}, indent=2))
        return

    # Fail fast on a typo'd CHAD_DISABLE, not mid-run: an unrecognized lever means the
    # harness would run unmodified while an ablation reports the delta as "no effect".
    try:
        levers.validate_env()
    except levers.UnknownLever as e:
        sys.stderr.write(f"chad: {e}\n")
        sys.exit(1)
    _preflight(args.backend)  # Apple Silicon only for MLX; remote backends run anywhere
    # --think-budget (plan 039) reaches the TUI/REPL Agents through the same env knob
    # their __init__ reads, so the flag works on every entrypoint, not just headless.
    if args.think_budget is not None:
        os.environ["CHAD_THINK_BUDGET"] = str(args.think_budget)
    if args.think_ceiling is not None:
        os.environ["CHAD_THINK_CEILING"] = str(args.think_ceiling)
    # --turn-budget-* (plan 040) reach the TUI/REPL Agents through the same env knobs
    # their __init__ reads, so the governor is configurable on every entrypoint.
    if args.turn_budget_tokens is not None:
        os.environ["CHAD_TURN_BUDGET_TOKENS"] = str(args.turn_budget_tokens)
    if args.turn_budget_s is not None:
        os.environ["CHAD_TURN_BUDGET_S"] = str(args.turn_budget_s)
    task = args.task or args.prompt_flag
    # `chad prove` — the bundled smoke test (prove.py). Dispatched on the literal
    # positional so the CLI stays a single entrypoint; a real task named "prove"
    # is vanishingly unlikely and can always be phrased longer.
    if task == "prove":
        from . import prove
        sys.exit(prove.run(args))
    # Ornith; no draft, ever. RAM-aware default, local-dir-preferred, HF fallback.
    model_id, why = _pick_model()

    # Advanced, rarely-touched knobs live in env vars to keep the CLI sane:
    #   CHAD_MAX_CONTEXT       YaRN-extend the window (e.g. 131072 for 128k)
    #   CHAD_CTX_LIMIT         prompt-token budget before old tool outputs compact
    #   CHAD_KV_BITS           KV cache quantization; default AUTO (8-bit where the
    #                          fused kernel covers the model — both shipped Ornith
    #                          models). 0 forces the fp16 cache.
    #   CHAD_KV_CACHE_MAX_GB   cap the on-disk KV cache (LRU-evict above it); 0 = unlimited
    max_context = _env_int("CHAD_MAX_CONTEXT")
    kv_bits = _env_int("CHAD_KV_BITS")

    # ds4-style on-disk KV warm-start of the stable system+tools prefix.
    cache_dir = os.path.expanduser("~/.cache/chad/kv")
    kv_cache_max_gb = _env_int("CHAD_KV_CACHE_MAX_GB")
    kv_cache_max_bytes = (kv_cache_max_gb if kv_cache_max_gb is not None else 8) * 1024**3
    # Clean up push-spills orphaned by a prior killed/crashed session (see engine.py) —
    # runs for every backend: the dir is shared, and a remote-backend run should still
    # reclaim what a dead MLX session leaked.
    from .engine import sweep_orphan_spills
    sweep_orphan_spills(cache_dir, max_age_s=6 * 3600)

    if args.backend in ("openai", "llama"):
        # Plan 046/047 spikes: drive the chad harness against a remote endpoint instead
        # of the in-process MLX engine. Only a tokenizer is loaded locally (to render
        # prompts); generation is proxied over HTTP. Weights are NOT downloaded (we
        # don't run them here). The MLX default path is untouched.
        tokenizer_id = args.tokenizer or config.env_str("CHAD_TOKENIZER")
        api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
        if args.backend == "openai":
            from .openai_engine import OpenAIEngine
            base_url = args.base_url or config.env_str("CHAD_OPENAI_BASE_URL")
            if not base_url:
                sys.stderr.write("chad --backend openai needs --base-url (or "
                                 "CHAD_OPENAI_BASE_URL), e.g. http://localhost:8080/v1\n")
                sys.exit(1)
            eng = OpenAIEngine(model_id=model_id, base_url=base_url, api_key=api_key,
                               tokenizer_id=tokenizer_id,
                               effective_ctx=max_context or 32768)
        else:
            from .completion_engine import CompletionEngine
            base_url = args.base_url or config.env_str("CHAD_LLAMA_BASE_URL")
            if not base_url:
                sys.stderr.write("chad --backend llama needs --base-url (or "
                                 "CHAD_LLAMA_BASE_URL), e.g. http://192.168.87.25:8081\n")
                sys.exit(1)
            # effective_ctx 0 = auto: load() reads the server's /props n_ctx so chad's
            # window matches the wall the server actually enforces.
            eng = CompletionEngine(model_id=model_id, base_url=base_url, api_key=api_key,
                                   tokenizer_id=tokenizer_id,
                                   effective_ctx=max_context or 0)
        sys.stderr.write(f"backend={args.backend} · base_url={base_url} · model={model_id} "
                         f"(tokenizer local, generation proxied) ...\n")
    else:
        _ensure_model(model_id)  # first-run download-on-consent if it's an uncached HF repo
        eng = Engine(
            model_id=model_id,
            draft_id=None,
            kv_bits=kv_bits,
            max_context=max_context,
            cache_dir=cache_dir,
            kv_cache_max_bytes=kv_cache_max_bytes,
        )

    # CHAD_TEMP: sampling temperature override, all backends. The default stays 0.0
    # (greedy — reproducible, and the MLX prompt-lookup fast path requires it), but
    # greedy has a failure mode measured in NIGHT-7: a stall/garbled call replays
    # itself byte-identically on every retry and across "independent" bench reps.
    # Benchmarks and unattended runs should set e.g. CHAD_TEMP=0.7 (what the field
    # harnesses run) so retries can take a different path.
    _temp = config.env_str("CHAD_TEMP")
    if _temp:
        try:
            eng.temp = float(_temp)
        except ValueError:
            sys.stderr.write(f"[ignoring CHAD_TEMP={_temp!r}: not a number]\n")

    # --serve: OpenAI-compatible endpoint on the in-process engine (plan 075 WS3).
    # Loads synchronously (nothing to interact with until it's up), then serves until
    # interrupted. MLX backend only — serving a remote backend would just proxy.
    if args.serve:
        if args.backend != "mlx":
            sys.stderr.write("chad --serve requires the in-process MLX backend\n")
            sys.exit(1)
        from .serve import serve
        sys.stderr.write(f"loading {os.path.basename(model_id.rstrip('/'))} [{why}] ...\n")
        try:
            load_s = eng.load()
        except Exception as e:  # noqa: BLE001 — convert any load failure into guidance
            _fail_model_load(model_id, e)
        sys.stderr.write(f"ready in {load_s:.1f}s | context {eng.effective_ctx} tokens\n")
        serve(eng, host=args.serve_host, port=args.serve_port)
        return

    # The full-screen TUI loads the 11 GB of weights on a BACKGROUND thread so the banner
    # + input come up in ~0.6 s and you can read/queue while it loads (the load itself is
    # disk-bound and can't be made faster). Headless one-shot and the plain --repl still
    # load synchronously — there's nothing to interact with until the model answers, and
    # a background download prompt would be worse than a blocking one. The remote
    # backends only load a tokenizer (cheap), so they stay synchronous too.
    background = args.backend == "mlx" and not task and not args.repl

    ctx_limit = None
    if not background:
        if args.backend == "mlx":
            sys.stderr.write(f"loading {os.path.basename(model_id.rstrip('/'))} [{why}] ...\n")
        try:
            load_s = eng.load()
        except Exception as e:  # noqa: BLE001 — convert any load failure into guidance
            _fail_model_load(model_id, e)
        ctx_limit = _compute_ctx_limit(eng)
        sys.stderr.write(f"ready in {load_s:.1f}s | context {eng.effective_ctx} tokens "
                         f"(compact at {ctx_limit})\n")

    start_mode = "plan" if args.plan else ("auto" if args.yolo else "normal")
    thinking = not args.no_think

    # Live per-turn ctx-limit recheck (plan 075 WS1.4): the startup number was
    # computed on whatever the box looked like at load; Docker/harbor/browsers
    # changing the physical free band mid-session changes what is safe. MLX
    # backend only — the remote backends hold no local KV.
    ctx_limit_fn = (lambda: _compute_ctx_limit(eng)) if args.backend == "mlx" \
        and not _env_int("CHAD_CTX_LIMIT") else None

    # Resume seeds a FRESH Agent's messages; that Agent mints a new session_id, so the
    # picked/newest session is copied, never overwritten (implicit fork — plan 043).
    #   --resume : list recent sessions and pick by number (needs a TTY).
    #   -c       : the most recent session (unchanged simple case).
    resume = None
    if args.resume:
        from . import session
        items = session.list_sessions(os.getcwd(), limit=10)
        if not items:
            sys.stderr.write("no saved sessions for this directory; starting fresh\n")
        elif not sys.stdin.isatty():
            sys.stderr.write("chad --resume needs an interactive terminal to pick a "
                             "session; use -c to resume the most recent one.\n")
            sys.exit(1)
        else:
            pick = _pick_session(items)
            if pick:
                data = session.load_session(os.getcwd(), pick["session_id"])
                if data:
                    resume = data["messages"]
                    sys.stderr.write(f"resuming (forked): {session.describe(pick)}\n")
    elif args.cont:
        from . import session
        data = session.load_session(os.getcwd())
        if data:
            resume = data["messages"]
            sys.stderr.write(f"resuming session ({session.session_summary(os.getcwd())})\n")
        else:
            sys.stderr.write("no saved session for this directory; starting fresh\n")

    if task:
        # A one-shot run is inherently unattended: the interactive confirm prompt
        # reads from stdin, which EOFs with no TTY and would abort every edit. So
        # auto-approve mutating tools unless the user asked for read-only --plan.
        run_mode = start_mode
        if run_mode == "normal" and not sys.stdin.isatty():
            run_mode = "auto"
            sys.stderr.write("[headless: auto-approving tools (use --plan for read-only)]\n")
        agent = Agent(eng, yolo=(run_mode == "auto"), ctx_limit=ctx_limit,
                      mode=run_mode, thinking=thinking, resume=resume, persist=True,
                      think_budget=args.think_budget, think_ceiling=args.think_ceiling,
                      turn_budget_tokens=args.turn_budget_tokens,
                      turn_budget_s=args.turn_budget_s, ctx_limit_fn=ctx_limit_fn)
        # Plan 085: wall time across ALL of this task's turns (initial + any auto-continue
        # relaunches), measured against the wall budget to decide the early-finish review.
        task_start = time.monotonic()
        agent.run_turn(task)
        # Plan 040: if the turn hard-stopped on a budget (governor token/wall budget, the
        # step cap's final window landing nothing, or the iter-2 no-empty-diff gate),
        # optionally relaunch a FRESH turn (new context + reset KV cache) seeded with the
        # deterministic progress note — shedding both the ramble and the huge prefill the
        # stuck model dragged. Unattended runs default to 2 relaunches: headless is
        # exactly where nobody can say 'continue', and a banked half-done task otherwise
        # ships as an empty diff (the NIGHT-7 bail signature).
        continues = args.auto_continue if args.auto_continue is not None \
            else (2 if run_mode == "auto" else 0)
        while agent.budget_note and continues > 0:
            note = agent.budget_note
            continues -= 1
            sys.stderr.write("[governor] previous turn ran out of budget/steps; continuing "
                             "fresh with a progress note\n")
            # A deterministic (temp-0) stall replays itself verbatim on retry — the
            # NIGHT-7 evidence: 3/3 byte-identical failing reps. Give the relaunch a
            # sampling distribution so it can take a different path.
            if getattr(eng, "temp", None) is not None:
                eng.temp = max(eng.temp, 0.6)
            eng.reset()
            agent = Agent(eng, yolo=(run_mode == "auto"), ctx_limit=ctx_limit,
                          mode=run_mode, thinking=thinking, persist=True,
                          think_budget=args.think_budget,
                          turn_budget_tokens=args.turn_budget_tokens,
                          turn_budget_s=args.turn_budget_s, ctx_limit_fn=ctx_limit_fn)
            agent.run_turn(f"{task}\n\n[{note}]")
        # Early-finish self-review (plan 085 scope 3): if the task settled CLEANLY (no
        # banked budget note) with more than 30% of the wall budget still unspent, relaunch
        # ONE fresh-context turn to independently verify the deliverables — the fresh KV
        # cache sheds the poisoned context that convinced the first attempt it was done,
        # catching the confident-wrong `done`. Off unless armed (--review-pass /
        # CHAD_REVIEW_PASS) and a wall budget is set, so interactive/unmetered runs and
        # clean A/B baselines never trigger it.
        review_armed = args.review_pass or config.flag("CHAD_REVIEW_PASS")
        elapsed = time.monotonic() - task_start
        if review_armed and guardrails.review_pass_should_fire(
                not agent.budget_note, args.turn_budget_s, elapsed):
            # The review turn respects the SAME task deadline: give it only the wall time
            # that remains, floored, so its own governor/wrap-up can't blow past the cap.
            review_budget = max(30.0, args.turn_budget_s - elapsed)
            sys.stderr.write(f"[review] task finished with {args.turn_budget_s - elapsed:.0f}s "
                             "of budget left; running one fresh-context verification pass\n")
            eng.reset()
            agent = Agent(eng, yolo=(run_mode == "auto"), ctx_limit=ctx_limit,
                          mode=run_mode, thinking=thinking, persist=True,
                          think_budget=args.think_budget,
                          turn_budget_tokens=args.turn_budget_tokens,
                          turn_budget_s=review_budget, ctx_limit_fn=ctx_limit_fn)
            agent.run_turn(task + guardrails.REVIEW_PASS_PROMPT)
        agent.save()  # persist so a follow-up `chad -c "..."` picks up the thread
    elif args.repl:
        repl(eng, yolo=args.yolo, ctx_limit=ctx_limit, resume=resume, thinking=thinking,
             ctx_limit_fn=ctx_limit_fn)
    else:
        from .engine import peek_context_window
        from .tui import run_tui
        _maybe_home_dir_note()
        # Cheap config-only window for the banner + a provisional compaction limit, both
        # shown instantly; `finalize` runs the real load on the TUI's background thread and
        # returns (load_s, ctx_limit) once weights are in.
        window = peek_context_window(model_id, max_context)
        provisional = ctx_limit or _env_int("CHAD_CTX_LIMIT") \
            or min(max(4096, (window or 32768) - 2048), 120_000)

        def finalize():
            load_s = eng.load()
            return load_s, _compute_ctx_limit(eng)

        run_tui(eng, provisional, mode=start_mode, thinking=thinking, resume=resume,
                ctx_window=window, finalize=finalize, ctx_limit_fn=ctx_limit_fn)


if __name__ == "__main__":
    main()
