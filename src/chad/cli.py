#!/usr/bin/env python3
"""chad — a local, MLX-backed, Claude-Code-style coding agent.

One model (Ornith — 35B on most Macs, 9B on small), one entrypoint, run with uv:

    uv run chad                                # interactive full-screen TUI
    uv run chad "fix the bug in greet.py"      # one-shot, headless
    uv run chad -c                             # resume this directory's conversation
    uv run chad --model 9b                     # force the small model

Plus three subcommands, each with its own `--help`: `chad serve`, `chad prove`,
`chad levers`.

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
from .base_engine import BackendError
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
# resident + KV + runtime, and the KV grows across a long agentic turn. That used to
# SIGKILL mid-turn on a 24 GB Mac, where the Metal wired limit (~2/3 RAM ≈ 16 GB) minus
# the OS leaves little headroom — so the floor sat at 32 GB. The fused attention kernel
# (mlx_qsdpa) plus the 8-bit-from-the-start KV cache it enables cut the per-token cache
# cost enough to give that headroom back, and `ram_aware_ctx_limit` now sizes the
# compaction trigger from the live Metal budget rather than a blind constant, so a
# tight box self-limits its window instead of dying. 24 GB gets the 35B; 16 GB and
# below still fall back to the 9B.
_BIG_RAM_GB = 23.5
# `--model` shorthands. Anything else is passed through as an HF repo id or local dir.
_MODEL_ALIASES = {"35b": (_LOCAL_35B, _HF_35B), "9b": (_LOCAL_9B, _HF_9B)}


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


# The three sampler knobs travel TOGETHER, as one call, deliberately.
#
# They used to be three sibling blocks inlined in `main()`, which meant every serve path
# that builds its own engine had to remember to copy all three — and `chad serve` didn't
# copy any, so a server started with CHAD_MIN_P ran without it and nothing said so. The
# failure shape is that sibling settings drift ONE AT A TIME: a later fix honors the field
# it touched, looks complete, and leaves its neighbours silently dead. There is one
# function now, so a caller cannot honor `temp` and forget `min_p`.
SAMPLER_ENV = (("temp", "CHAD_TEMP"), ("min_p", "CHAD_MIN_P"), ("top_p", "CHAD_TOP_P"))


def apply_sampler_env(eng):
    """Apply the sampler-knob environment overrides to `eng`, in place.

    CHAD_TEMP: sampling temperature, all backends. The default stays 0.0 (greedy —
    reproducible, and the MLX prompt-lookup fast path requires it), but greedy has a
    failure mode measured in the field: a stall/garbled call replays itself byte-identically
    on every retry and across "independent" bench reps. Benchmarks and unattended runs
    should set e.g. CHAD_TEMP=0.7 (what the field harnesses run) so retries can take a
    different path.

    CHAD_MIN_P / CHAD_TOP_P: quant-tail anti-confabulation knobs, off (0.0) by default —
    trim the sub-noise-floor logit tail without touching temp."""
    for attr, var in SAMPLER_ENV:
        raw = config.env_str(var)
        if not raw:
            continue
        try:
            setattr(eng, attr, float(raw))
        except ValueError:
            sys.stderr.write(f"[ignoring {var}={raw!r}: not a number]\n")


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
    the context budget must respect it too."""
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
    """Largest prompt-token budget (= the compaction trigger) that keeps the
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
    # 175k trigger that extrapolated to 102.9% of budget — over the wall). The fixed
    # `reserve_gb` cannot cover a term that grows per-token, so fold it into the
    # divisor. 1.75 is the 35B measurement; unmeasured on the 9B, where it errs safe
    # (over-compaction costs a re-prefill, undershoot costs a jetsam kill).
    ram_ctx = int(usable / (kv_bytes_per_token * slope_factor))
    return max(floor, min(eff_ctx - gen_margin, ram_ctx))


def _compute_ctx_limit(eng):
    """The auto-compaction threshold for a loaded engine. On this non-trimmable hybrid
    cache, compaction forces a full body re-prefill (~79 % of all prefill), so
    we compact as rarely as RAM safely allows: size the trigger from the live Metal
    budget + the model's *measured* per-token KV cost instead of a blind 120 k
    cap that over-compacts. CHAD_CTX_LIMIT still wins (used by tests); CHAD_CTX_RESERVE_GB
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
    cryptically elsewhere. The remote backend (`--backend llama`) loads NO MLX —
    only a tokenizer plus HTTP — so it runs anywhere (e.g. inside a Linux
    container reaching a remote server); skip the Apple-Silicon gate for it."""
    if backend == "llama":
        return
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        sys.stderr.write(
            "chad: requires an Apple Silicon Mac (arm64 macOS).\n"
            f"  detected: {platform.system()} {platform.machine() or '?'}\n"
            "  MLX ships no CPU/CUDA build — there is no supported non-Apple path.\n"
            "  (For a remote engine on this host, use --backend llama.)\n")
        sys.exit(1)


def _detect_ram_gb():
    """Physical RAM in GiB via sysctl, or None if it can't be read."""
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True)
        return int(out.strip()) / (1024 ** 3)
    except Exception:  # noqa: BLE001 — any failure → caller picks the safe (smaller) model
        return None


def _resolve(local, repo):
    """Prefer a locally-built models/ dir over the HF repo when it exists."""
    return local if os.path.isdir(local) else repo


def _pick_model(spec=None):
    """Resolve the model id and a human label for *why* it was chosen.

    Order: explicit `--model` (`spec`) → CHAD_MODEL → RAM-aware default (35B unless the
    box is small). A resolved shorthand prefers a locally-built models/ dir over the HF
    repo when one exists; an arbitrary spec is passed through untouched.

    Forcing `35b` where the RAM check would not have chosen it (small or undetectable
    RAM) is honored, but warns on stderr first: chad advises, the caller decides.
    """
    # Name the winning source in the reason: it is only ever ambiguous when both the
    # flag and the env var are set, which is exactly when the user needs to be told.
    source = "--model" if spec is not None else "CHAD_MODEL"
    spec = spec or config.env_str("CHAD_MODEL")
    if spec and spec.strip().lower() != "auto":
        key = spec.strip().lower()
        alias = _MODEL_ALIASES.get(key)
        if not alias:
            return spec, f"explicitly requested ({source} override)"
        local, repo = alias
        if key == "35b":
            ram = _detect_ram_gb()
            if ram is None or ram < _BIG_RAM_GB:
                got = "undetectable" if ram is None else f"{ram:.0f} GB"
                sys.stderr.write(
                    f"chad: 35b forced via {source} (RAM {got} < ~{_BIG_RAM_GB:.0f} GB "
                    f"recommended); the 35B needs ~16 GB resident with its KV cache and "
                    f"may OOM or thrash here. Proceeding as asked.\n")
        return _resolve(local, repo), f"{key} (requested via {source})"
    ram = _detect_ram_gb()
    if ram is None:
        # RAM unreadable -> the safe (smaller) model: a wrong 9B costs capability, a
        # wrong 35B costs a 12 GB download and possibly an OOM'd first session.
        local, repo = _LOCAL_9B, _HF_9B
        why = "9B (RAM undetectable — choosing the safe smaller model; " \
              "set CHAD_MODEL or --model to override)"
    elif ram < _BIG_RAM_GB:
        local, repo = _LOCAL_9B, _HF_9B
        why = f"9B (default; {ram:.0f} GB RAM < {_BIG_RAM_GB:.0f} GB, 35B would be tight)"
    else:
        local, repo = _LOCAL_35B, _HF_35B
        why = "35B (default)"
    return _resolve(local, repo), why


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


def _fail_backend(err, base_url):
    """A remote backend that dies mid-run used to exit through a bare traceback ending in
    `chad.base_engine.BackendError`, which reads as a chad crash rather than "the server
    you pointed me at isn't answering". Same problem/cause/fix shape as the model-load
    and download failures."""
    msg = str(err)
    sys.stderr.write("\nchad: the remote backend stopped answering\n")
    sys.stderr.write(f"  cause: {msg}\n")
    if "connection failed" in msg.lower():
        sys.stderr.write(
            f"  fix:   nothing is listening at {base_url or 'the base URL'}. Start the\n"
            "         server, check the host/port, and confirm it is reachable from here\n"
            "         (a container needs the server bound to 0.0.0.0, not 127.0.0.1).\n"
            "         To serve this Mac's own model: `chad serve --host 0.0.0.0`.\n")
    elif "HTTP 401" in msg or "HTTP 403" in msg:
        sys.stderr.write(
            "  fix:   the server rejected the credentials. Pass --api-key-env NAME "
            "naming\n         the env var that holds the key.\n")
    else:
        sys.stderr.write(
            "  fix:   check the server's own logs — it accepted the connection and then\n"
            "         failed the request. A model/tokenizer vocab mismatch is the usual\n"
            "         cause; --tokenizer must name a repo sharing the served vocab.\n")
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


# Real subcommands, dispatched on argv[1] rather than through argparse subparsers.
# The default invocation's positional is a free-form task string, and a subparser layout
# would either shadow it or force `chad -- "some task"`; matching argv[1] exactly keeps
# `chad "serve the API from cache"` a task and `chad serve` a subcommand, which is the
# same rule the old literal-positional dispatch used.
_SUBCOMMANDS = ("serve", "prove", "levers")

# Set by `_main` once the remote backend's URL is resolved, so the top-level BackendError
# handler can name the host that stopped answering. Only the remote backend can raise
# one, and only one engine exists per process, so a module-level value is exact.
_resolved_base_url = None


def _add_model_arg(ap):
    """`--model`, shared by the agent and `chad serve` — both load a local model and both
    need the same escape hatch from the RAM-aware default."""
    ap.add_argument("--model", default=None,
                    help="which model to load: '35b' (big, default on most Macs), '9b' "
                         "(small, low-RAM), 'auto' (choose by RAM), or any Hugging Face "
                         "repo id / local model dir. Also CHAD_MODEL.")


def _agent_parser():
    ap = argparse.ArgumentParser(
        prog="chad",
        description="Local MLX-backed coding agent (Ornith). Run with `uv run chad`.",
        epilog="subcommands (each takes --help): chad serve · chad prove · chad levers. "
               "Long-session and unattended-run knobs live in CHAD_* env vars — "
               "see README \"Advanced\".",
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
    # The unattended-run governor cluster (think ceiling, turn budgets, auto-continue,
    # review pass) is env-only: CHAD_THINK_CEILING / CHAD_TURN_BUDGET_TOKENS /
    # CHAD_TURN_BUDGET_S / CHAD_AUTO_CONTINUE / CHAD_REVIEW_PASS. The only thing that ever
    # sets them is an automated runner, which already builds a CHAD_* env dict, and five
    # knobs no interactive user has wanted were most of what made `--help` unreadable.
    # Backend selection. Default 'mlx' is the in-process engine and the whole point of
    # chad; 'llama' drives a llama.cpp server's raw /completion with token-id prompts +
    # real cache telemetry (see completion_engine.py) — the arm used when chad runs inside
    # a Linux container against a remote server. The MLX path below is untouched
    # when --backend is unset.
    ap.add_argument("--backend", choices=("mlx", "llama"), default="mlx",
                    help="inference backend: 'mlx' (default, in-process KV cache) or "
                         "'llama' (llama.cpp raw /completion, token-id prompts, remote "
                         "server). The remote backend requires --base-url.")
    ap.add_argument("--base-url", dest="base_url", default=None,
                    help="remote-backend base URL: for --backend llama the llama-server "
                         "origin (e.g. http://<host>:8081, also CHAD_LLAMA_BASE_URL).")
    ap.add_argument("--tokenizer", dest="tokenizer", default=None,
                    help="HF repo/dir whose tokenizer to load for a remote backend, when the "
                         "served model id has no tokenizer files (e.g. a GGUF repo). Must "
                         "share the served model's vocab. Also CHAD_TOKENIZER.")
    ap.add_argument("--api-key-env", dest="api_key_env", default=None,
                    help="name of the env var holding the API key for a remote backend; the "
                         "key is read from that var, never passed on the command line.")
    _add_model_arg(ap)
    ap.add_argument("--repl", action="store_true",
                    help="plain line REPL instead of the full-screen TUI")
    # Back-compat: -p/--prompt was the old one-shot spelling, now the positional task;
    # --levers is now `chad levers`.
    ap.add_argument("-p", "--prompt", dest="prompt_flag", help=argparse.SUPPRESS)
    ap.add_argument("--levers", action="store_true", help=argparse.SUPPRESS)
    return ap


def _serve_parser():
    """`chad serve` — expose this machine's MLX engine over the same llama.cpp
    /completion protocol the remote backend speaks, so a chad that can't run MLX
    (a Linux container) drives the local model instead of a remote GGUF."""
    ap = argparse.ArgumentParser(
        prog="chad serve",
        description="Serve this machine's local MLX engine over the llama.cpp "
                    "/completion protocol. Point a client at it with "
                    "`chad \"…\" --backend llama --base-url http://<host>:<port>`.",
    )
    ap.add_argument("--host", default=None,
                    help="bind address (default 127.0.0.1; use 0.0.0.0 to accept clients "
                         "from containers or the LAN — set CHAD_SERVE_API_KEY if you do). "
                         "Also CHAD_SERVE_HOST.")
    ap.add_argument("--port", type=int, default=None,
                    help="TCP port (default 8081). Also CHAD_SERVE_PORT.")
    _add_model_arg(ap)
    # Hidden, and rejected by serve.run: without it `chad serve --backend llama` would
    # die on "unrecognized arguments" instead of explaining why serving a remote client
    # backend is incoherent.
    ap.add_argument("--backend", choices=("mlx", "llama"), default="mlx",
                    help=argparse.SUPPRESS)
    return ap


def _prove_parser():
    ap = argparse.ArgumentParser(
        prog="chad prove",
        description="Run the bundled end-to-end smoke test against the pinned 9B model: "
                    "downloads it if needed, drives a real task, and reports what worked.",
    )
    ap.add_argument("--backend", choices=("mlx", "llama"), default="mlx",
                    help=argparse.SUPPRESS)  # see _serve_parser
    return ap


def _levers_parser():
    return argparse.ArgumentParser(
        prog="chad levers",
        description="Print the harness lever registry as JSON and exit. The ablation "
                    "driver enumerates this instead of hardcoding lever names; "
                    "CHAD_DISABLE=a,b turns individual levers off.",
    )


def _run_levers():
    """No _preflight and no model: an ablation driver enumerating levers should not need
    an Apple-Silicon box or a loadable model just to read the registry."""
    print(json.dumps({"levers": levers.as_dict(), "groups": levers.groups(),
                      "active": levers.active()}, indent=2))
    return 0


def main(argv=None):
    """Console entrypoint. Wraps `_main` only to turn a backend fault that escaped the
    agent's retry loop into guidance — it can surface from the one-shot, --repl, or TUI
    path alike, and all three would otherwise exit through a raw traceback."""
    try:
        return _main(argv)
    except BackendError as e:
        _fail_backend(e, _resolved_base_url)


def _main(argv=None):
    global _resolved_base_url
    argv = list(sys.argv[1:] if argv is None else argv)
    sub = argv[0] if argv and argv[0] in _SUBCOMMANDS else None
    if sub == "levers":
        _levers_parser().parse_args(argv[1:])
        sys.exit(_run_levers())
    if sub == "serve":
        from . import serve
        sys.exit(serve.run(_serve_parser().parse_args(argv[1:])))
    if sub == "prove":
        from . import prove
        sys.exit(prove.run(_prove_parser().parse_args(argv[1:])))

    args = _agent_parser().parse_args(argv)
    if args.levers:  # deprecated spelling of `chad levers`
        sys.exit(_run_levers())

    # Fail fast on a typo'd CHAD_DISABLE, not mid-run: an unrecognized lever means the
    # harness would run unmodified while an ablation reports the delta as "no effect".
    try:
        levers.validate_env()
    except levers.UnknownLever as e:
        sys.stderr.write(f"chad: {e}\n")
        sys.exit(1)
    _preflight(args.backend)  # Apple Silicon only for MLX; the remote backend runs anywhere
    # --think-budget reaches the TUI/REPL Agents through the same env knob
    # their __init__ reads, so the flag works on every entrypoint, not just headless.
    if args.think_budget is not None:
        os.environ["CHAD_THINK_BUDGET"] = str(args.think_budget)
    # The wall budget is read here as well as in Agent.__init__ because the one-shot
    # relaunch/review logic below does its own arithmetic against it (how much of the
    # TASK deadline is left), not just per-turn enforcement.
    turn_budget_s = config.env_float("CHAD_TURN_BUDGET_S")
    task = args.task or args.prompt_flag
    # Ornith. --model / CHAD_MODEL, else the RAM-aware default; local-dir-preferred,
    # HF fallback.
    model_id, why = _pick_model(args.model)

    # Advanced, rarely-touched knobs live in env vars to keep the CLI sane:
    #   CHAD_MAX_CONTEXT       YaRN-extend the window (e.g. 131072 for 128k)
    #   CHAD_CTX_LIMIT         prompt-token budget before old tool outputs compact
    #   CHAD_KV_BITS           KV cache quantization; default AUTO (8-bit where the
    #                          fused kernel covers the model — both shipped Ornith
    #                          models). 0 forces the fp16 cache.
    #   CHAD_KV_CACHE_MAX_GB   cap the on-disk KV cache (LRU-evict above it); 0 = unlimited
    # Unattended-run governor (harness-only; the matching flags are hidden):
    #   CHAD_THINK_CEILING  CHAD_TURN_BUDGET_TOKENS  CHAD_TURN_BUDGET_S
    #   CHAD_AUTO_CONTINUE  CHAD_REVIEW_PASS
    max_context = _env_int("CHAD_MAX_CONTEXT")
    kv_bits = _env_int("CHAD_KV_BITS")

    # On-disk KV warm-start of the stable system+tools prefix.
    cache_dir = os.path.expanduser("~/.cache/chad/kv")
    kv_cache_max_gb = _env_int("CHAD_KV_CACHE_MAX_GB")
    kv_cache_max_bytes = (kv_cache_max_gb if kv_cache_max_gb is not None else 8) * 1024**3
    # Clean up push-spills orphaned by a prior killed/crashed session (see engine.py) —
    # runs for every backend: the dir is shared, and a remote-backend run should still
    # reclaim what a dead MLX session leaked.
    from .engine import sweep_orphan_spills
    sweep_orphan_spills(cache_dir, max_age_s=6 * 3600)

    if args.backend == "llama":
        # Drive the chad harness against a remote llama.cpp server instead
        # of the in-process MLX engine. Only a tokenizer is loaded locally (to render
        # prompts); generation is proxied over HTTP. Weights are NOT downloaded (we
        # don't run them here). The MLX default path is untouched.
        tokenizer_id = args.tokenizer or config.env_str("CHAD_TOKENIZER")
        api_key = os.environ.get(args.api_key_env, "") if args.api_key_env else ""
        from .completion_engine import CompletionEngine
        base_url = args.base_url or config.env_str("CHAD_LLAMA_BASE_URL")
        if not base_url:
            sys.stderr.write("chad --backend llama needs --base-url (or "
                             "CHAD_LLAMA_BASE_URL), e.g. http://<host>:8081\n")
            sys.exit(1)
        _resolved_base_url = base_url
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
            kv_bits=kv_bits,
            max_context=max_context,
            cache_dir=cache_dir,
            kv_cache_max_bytes=kv_cache_max_bytes,
        )

    apply_sampler_env(eng)

    # The full-screen TUI loads the 11 GB of weights on a BACKGROUND thread so the banner
    # + input come up in ~0.6 s and you can read/queue while it loads (the load itself is
    # disk-bound and can't be made faster). Headless one-shot and the plain --repl still
    # load synchronously — there's nothing to interact with until the model answers, and
    # a background download prompt would be worse than a blocking one. The remote
    # backend only loads a tokenizer (cheap), so it stays synchronous too.
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

    start_mode = "plan" if args.plan else ("yolo" if args.yolo else "normal")
    thinking = not args.no_think

    # Live per-turn ctx-limit recheck: the startup number was
    # computed on whatever the box looked like at load; Docker/harbor/browsers
    # changing the physical free band mid-session changes what is safe. MLX
    # backend only — the remote backend holds no local KV.
    ctx_limit_fn = (lambda: _compute_ctx_limit(eng)) if args.backend == "mlx" \
        and not _env_int("CHAD_CTX_LIMIT") else None

    # Resume seeds a FRESH Agent's messages; that Agent mints a new session_id, so the
    # picked/newest session is copied, never overwritten (implicit fork).
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
            run_mode = "yolo"
            sys.stderr.write("[headless: auto-approving tools (use --plan for read-only)]\n")
        agent = Agent(eng, yolo=(run_mode == "yolo"), ctx_limit=ctx_limit,
                      mode=run_mode, thinking=thinking, resume=resume, persist=True,
                      think_budget=args.think_budget,
                      turn_budget_s=turn_budget_s, ctx_limit_fn=ctx_limit_fn)
        # Wall time across ALL of this task's turns (initial + any auto-continue
        # relaunches), measured against the wall budget to decide the early-finish review.
        task_start = time.monotonic()
        agent.run_turn(task)
        # If the turn hard-stopped on a budget (governor token/wall budget, the
        # step cap's final window landing nothing, or the no-empty-diff gate),
        # optionally relaunch a FRESH turn (new context + reset KV cache) seeded with the
        # deterministic progress note — shedding both the ramble and the huge prefill the
        # stuck model dragged. Unattended runs default to 2 relaunches: headless is
        # exactly where nobody can say 'continue', and a banked half-done task otherwise
        # ships as an empty diff — the measured bail signature.
        auto_continue = config.env_int("CHAD_AUTO_CONTINUE")
        continues = auto_continue if auto_continue is not None \
            else (2 if run_mode == "yolo" else 0)
        used_continues = 0
        while agent.budget_note:
            # Base allowance first; past it, keep granting fresh attempts while most
            # of the task wall is still unspent (bounded by AUTO_CONTINUE_TOTAL_CAP) —
            # the fixed base is wall-blind: a long build task gave up after 3
            # step-capped turns with 94.7% of a 12000s budget still unused.
            if continues > 0:
                continues -= 1
            elif (auto_continue is None and turn_budget_s
                    and guardrails.replenish_continue(
                        turn_budget_s, time.monotonic() - task_start,
                        used_continues)):
                sys.stderr.write("[governor] wall budget mostly unspent — granting an "
                                 "extra continue\n")
            else:
                break
            used_continues += 1
            note = agent.budget_note
            # The wall budget is a TASK-level deadline (the harness SIGKILLs the whole chad
            # process), not a per-turn one: a relaunch must inherit only the wall time that
            # REMAINS, or the governor / wrap-up / hard-abort windows (measured from each
            # turn's own start) never open before the process is killed — the relaunched
            # turn rides to a mid-work SIGKILL with nothing landed. With no wall budget set
            # (interactive/unmetered), there is no deadline: keep the old fresh-None budget.
            if turn_budget_s:
                relaunch_s = guardrails.relaunch_budget(
                    turn_budget_s, time.monotonic() - task_start)
                if relaunch_s is None:
                    sys.stderr.write("[governor] previous turn ran out of budget; too "
                                     "little wall left to relaunch — stopping\n")
                    break
            else:
                relaunch_s = None
            sys.stderr.write("[governor] previous turn ran out of budget/steps; continuing "
                             "fresh with a progress note\n")
            # A deterministic (temp-0) stall replays itself verbatim on retry — the
            # measured 3/3 byte-identical failing reps. Give the relaunch a
            # sampling distribution so it can take a different path.
            if getattr(eng, "temp", None) is not None:
                eng.temp = max(eng.temp, 0.6)
            eng.reset()
            agent = Agent(eng, yolo=(run_mode == "yolo"), ctx_limit=ctx_limit,
                          mode=run_mode, thinking=thinking, persist=True,
                          think_budget=args.think_budget,
                          turn_budget_s=relaunch_s, ctx_limit_fn=ctx_limit_fn)
            agent.run_turn(f"{task}\n\n[{note}]")
        # Early-finish self-review: if the task settled CLEANLY (no
        # banked budget note) with more than 30% of the wall budget still unspent, relaunch
        # ONE fresh-context turn to independently verify the deliverables — the fresh KV
        # cache sheds the poisoned context that convinced the first attempt it was done,
        # catching the confident-wrong `done`. Off unless armed (--review-pass /
        # CHAD_REVIEW_PASS) and a wall budget is set, so interactive/unmetered runs and
        # clean A/B baselines never trigger it.
        review_armed = config.flag("CHAD_REVIEW_PASS")
        elapsed = time.monotonic() - task_start
        if review_armed and guardrails.review_pass_should_fire(
                not agent.budget_note, turn_budget_s, elapsed):
            # The review turn respects the SAME task deadline: give it only the wall time
            # that remains, floored, so its own governor/wrap-up can't blow past the cap.
            review_budget = max(30.0, turn_budget_s - elapsed)
            sys.stderr.write(f"[review] task finished with {turn_budget_s - elapsed:.0f}s "
                             "of budget left; running one fresh-context verification pass\n")
            eng.reset()
            agent = Agent(eng, yolo=(run_mode == "yolo"), ctx_limit=ctx_limit,
                          mode=run_mode, thinking=thinking, persist=True,
                          think_budget=args.think_budget,
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
