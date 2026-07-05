#!/usr/bin/env python3
"""chad — a local, MLX-backed, Claude-Code-style coding agent.

One model (Ornith — 35B on big Macs, 9B on small), one entrypoint, run with uv:

    uv run chad                                # interactive full-screen TUI
    uv run chad "fix the bug in greet.py"      # one-shot, headless
    uv run chad -c                             # resume this directory's conversation

Rare long-session knobs live in env vars — see README "Advanced".
"""
import argparse
import os
import platform
import subprocess
import sys

from . import config
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
_HF_35B = "nathansutton/Ornith-1.0-35B-UD-Q2_K_XL-MLX"   # default: 35B MoE, ~12 GB resident
_HF_9B = "nathansutton/Ornith-1.0-9B-UD-Q4_K_XL-MLX"     # low-RAM fallback, ~5 GB resident
# A dev clone that already built the weights locally should use them rather than
# re-download — prefer these dirs when present.
_LOCAL_35B = os.path.join(_PROJECT_ROOT, "models", "Ornith-1.0-35B-dyn2-q2-awq")
_LOCAL_9B = os.path.join(_PROJECT_ROOT, "models", "Ornith-1.0-9B-4bit-awq")
# The 35B (2-bit experts / 6-bit backbone) is ~12 GB resident + KV + runtime ≈ 14 GB
# peak. Comfortable at ≥24 GB; on 16/18 GB MacBook Pros the default Metal wired limit
# (~2/3 RAM) sits below that, so we fall back to the 9B (fits easily) automatically.
_BIG_RAM_GB = 23.5


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


def ram_aware_ctx_limit(eff_ctx, budget_bytes, active_bytes, kv_bytes_per_token,
                        reserve_gb=1.5, safety=0.90, gen_margin=2048, floor=8192):
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
    if usable <= 0:
        return floor
    ram_ctx = int(usable / kv_bytes_per_token)
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
            ctx_limit = ram_aware_ctx_limit(
                eng.effective_ctx,
                mx.device_info()["max_recommended_working_set_size"],
                mx.get_active_memory(), eng.kv_bytes_per_token,
                reserve_gb=_env_float("CHAD_CTX_RESERVE_GB") or 1.5)
        except Exception:  # noqa: BLE001 — never let memory probing break startup
            ctx_limit = None
    if not ctx_limit:
        ctx_limit = min(max(4096, eng.effective_ctx - 2048), 120_000)  # old fixed cap
    return ctx_limit


def _preflight():
    """chad runs only on Apple Silicon — MLX has no CPU/CUDA build. Hard-stop with a
    human message instead of letting `uv sync`/import fail cryptically elsewhere."""
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        sys.stderr.write(
            "chad: requires an Apple Silicon Mac (arm64 macOS).\n"
            f"  detected: {platform.system()} {platform.machine() or '?'}\n"
            "  MLX ships no CPU/CUDA build — there is no supported non-Apple path.\n")
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
    if ram is not None and ram < _BIG_RAM_GB:
        local, repo = _LOCAL_9B, _HF_9B
        why = f"9B (default; {ram:.0f} GB RAM < {_BIG_RAM_GB:.0f} GB, 35B would be tight)"
    else:
        local, repo = _LOCAL_35B, _HF_35B
        why = "35B (default)"
    return (local if os.path.isdir(local) else repo), why


def _ensure_model(model_id):
    """If model_id is a HF repo id not yet in the local cache, confirm and download it
    into ~/.cache/huggingface (shared, resumable, paid once per machine). Local dirs
    and already-cached repos return immediately. Headless (no TTY) auto-downloads."""
    if os.path.isdir(model_id):
        return  # a local path — nothing to fetch
    from huggingface_hub import snapshot_download, try_to_load_from_cache
    if isinstance(try_to_load_from_cache(model_id, "config.json"), str):
        return  # already in the HF cache
    size = "~12 GB" if "35B" in model_id else "~5 GB"
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
    else:
        sys.stderr.write("[headless: downloading automatically]\n")
    try:
        snapshot_download(model_id)  # tqdm progress to stderr
    except Exception as e:  # noqa: BLE001 — offline / gated / typo'd repo → guidance, not a traceback
        sys.stderr.write(
            f"\nchad: could not download '{model_id}'\n"
            f"  cause: {type(e).__name__}: {e}\n"
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
    ap.add_argument("--turn-budget-tokens", type=int, default=None, dest="turn_budget_tokens",
                    help="runaway-turn governor: end a turn once it has burned N cumulative "
                         "prefill tokens WITHOUT landing+verifying a change (nudges at ~50%%, "
                         "banks a progress note and stops at ~80%%). Defaults to 3× the context "
                         "limit; CHAD_NO_GOVERNOR=1 disables. Also CHAD_TURN_BUDGET_TOKENS.")
    ap.add_argument("--turn-budget-s", type=float, default=None, dest="turn_budget_s",
                    help="wall-clock variant of --turn-budget-tokens (seconds); off by "
                         "default (interactive: the human is the wall clock). Also "
                         "CHAD_TURN_BUDGET_S.")
    ap.add_argument("--auto-continue", type=int, default=0, dest="auto_continue",
                    help="on a one-shot run, if the governor hard-stops a turn, relaunch a "
                         "FRESH turn (cleared context) seeded with the progress note, up to N "
                         "times (default 0 = off). Sheds the ramble and the huge prefill.")
    # Backend selection (plan 046 spike). Default 'mlx' is the in-process engine and the
    # whole point of chad; 'openai' drives the SAME harness against an OpenAI-compatible
    # endpoint to separate harness-value from engine-value (honest degradations apply —
    # see openai_engine.py). The MLX path below is untouched when --backend is unset.
    ap.add_argument("--backend", choices=("mlx", "openai"), default="mlx",
                    help="inference backend: 'mlx' (default, in-process KV cache) or "
                         "'openai' (spike: run the chad harness against an OpenAI-compatible "
                         "/v1/chat/completions endpoint; requires --base-url).")
    ap.add_argument("--base-url", dest="base_url", default=None,
                    help="OpenAI-compatible base URL for --backend openai (e.g. "
                         "http://localhost:8080/v1). Also CHAD_OPENAI_BASE_URL.")
    ap.add_argument("--api-key-env", dest="api_key_env", default=None,
                    help="name of the env var holding the API key for --backend openai; the "
                         "key is read from that var, never passed on the command line.")
    ap.add_argument("--repl", action="store_true",
                    help="plain line REPL instead of the full-screen TUI")
    # Back-compat: -p/--prompt was the old one-shot spelling, now the positional task.
    ap.add_argument("-p", "--prompt", dest="prompt_flag", help=argparse.SUPPRESS)
    args = ap.parse_args()

    _preflight()  # Apple Silicon only — fail clearly before importing/loading MLX
    # --think-budget (plan 039) reaches the TUI/REPL Agents through the same env knob
    # their __init__ reads, so the flag works on every entrypoint, not just headless.
    if args.think_budget is not None:
        os.environ["CHAD_THINK_BUDGET"] = str(args.think_budget)
    # --turn-budget-* (plan 040) reach the TUI/REPL Agents through the same env knobs
    # their __init__ reads, so the governor is configurable on every entrypoint.
    if args.turn_budget_tokens is not None:
        os.environ["CHAD_TURN_BUDGET_TOKENS"] = str(args.turn_budget_tokens)
    if args.turn_budget_s is not None:
        os.environ["CHAD_TURN_BUDGET_S"] = str(args.turn_budget_s)
    task = args.task or args.prompt_flag
    # Ornith; no draft, ever. RAM-aware default, local-dir-preferred, HF fallback.
    model_id, why = _pick_model()

    # Advanced, rarely-touched knobs live in env vars to keep the CLI sane:
    #   CHAD_MAX_CONTEXT  YaRN-extend the window (e.g. 131072 for 128k)
    #   CHAD_CTX_LIMIT    prompt-token budget before old tool outputs compact
    #   CHAD_KV_BITS      quantize the KV cache (e.g. 8) to save RAM on long runs
    max_context = _env_int("CHAD_MAX_CONTEXT")
    kv_bits = _env_int("CHAD_KV_BITS")

    # ds4-style on-disk KV warm-start of the stable system+tools prefix.
    cache_dir = os.path.expanduser("~/.cache/chad/kv")

    if args.backend == "openai":
        # Plan 046 spike: drive the chad harness against an OpenAI-compatible endpoint
        # instead of the in-process MLX engine. Only a tokenizer is loaded locally (to
        # render/decode prompts); generation is proxied over HTTP. Weights are NOT
        # downloaded (we don't run them here). The MLX default path is untouched.
        from .openai_engine import OpenAIEngine
        base_url = args.base_url or config.env_str("CHAD_OPENAI_BASE_URL")
        if not base_url:
            sys.stderr.write("chad --backend openai needs --base-url (or CHAD_OPENAI_BASE_URL), "
                             "e.g. http://localhost:8080/v1\n")
            sys.exit(1)
        api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
        eng = OpenAIEngine(model_id=model_id, base_url=base_url, api_key=api_key,
                           effective_ctx=max_context or 32768)
        sys.stderr.write(f"backend=openai · base_url={base_url} · model={model_id} "
                         f"(tokenizer local, generation proxied) ...\n")
    else:
        _ensure_model(model_id)  # first-run download-on-consent if it's an uncached HF repo
        eng = Engine(
            model_id=model_id,
            draft_id=None,
            kv_bits=kv_bits,
            max_context=max_context,
            cache_dir=cache_dir,
        )

    # The full-screen TUI loads the 11 GB of weights on a BACKGROUND thread so the banner
    # + input come up in ~0.6 s and you can read/queue while it loads (the load itself is
    # disk-bound and can't be made faster). Headless one-shot and the plain --repl still
    # load synchronously — there's nothing to interact with until the model answers, and
    # a background download prompt would be worse than a blocking one. `--backend openai`
    # only loads a tokenizer (cheap), so it stays synchronous too.
    background = args.backend != "openai" and not task and not args.repl

    ctx_limit = None
    if not background:
        if args.backend != "openai":
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
                      think_budget=args.think_budget,
                      turn_budget_tokens=args.turn_budget_tokens,
                      turn_budget_s=args.turn_budget_s)
        agent.run_turn(task)
        # Plan 040: if the governor hard-stopped the turn on budget, optionally relaunch a
        # FRESH turn (new context + reset KV cache) seeded with the deterministic progress
        # note — shedding both the ramble and the huge prefill the stuck model dragged.
        continues = args.auto_continue
        while agent.budget_note and continues > 0:
            note = agent.budget_note
            continues -= 1
            sys.stderr.write("[governor] previous turn ran out of budget; continuing fresh "
                             "with a progress note\n")
            eng.reset()
            agent = Agent(eng, yolo=(run_mode == "auto"), ctx_limit=ctx_limit,
                          mode=run_mode, thinking=thinking, persist=True,
                          think_budget=args.think_budget,
                          turn_budget_tokens=args.turn_budget_tokens,
                          turn_budget_s=args.turn_budget_s)
            agent.run_turn(f"{task}\n\n[{note}]")
        agent.save()  # persist so a follow-up `chad -c "..."` picks up the thread
    elif args.repl:
        repl(eng, yolo=args.yolo, ctx_limit=ctx_limit, resume=resume, thinking=thinking)
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
                ctx_window=window, finalize=finalize)


if __name__ == "__main__":
    main()
