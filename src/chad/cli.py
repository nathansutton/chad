#!/usr/bin/env python3
"""chad — a local, MLX-backed, Claude-Code-style coding agent.

One model (Qwen3.8-27B, 3-bit, with its DFlash2 drafter), one entrypoint, run with uv:

    uv run chad                                # interactive full-screen TUI
    uv run chad "fix the bug in greet.py"      # one-shot, headless
    uv run chad -c                             # resume this directory's conversation
    uv run chad --model <repo|dir>             # run different weights

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
from .diag import log
from .engine import Engine

# Package dir is src/chad/; the project root (two levels up) is the dev clone. If a
# locally-built weights tree exists at <root>/models/ it's preferred (see _pick_model);
# otherwise — the normal case — the default model ships from Hugging Face and downloads
# into the shared HF cache on first use. Point CHAD_MODEL at any local dir to override.
_HERE = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_HERE))

# The shipped model, on Hugging Face. Naming follows Unsloth's dynamic-quant
# convention so the quant scheme is recognizable (UD = Unsloth Dynamic; Q3_K_XL =
# a 3-bit body with extra bits where they pay), plus an -MLX suffix for format
# discoverability. The quant itself is MLX group-64 affine, not llama.cpp Q3_K
# k-quants — the model card says so; the tag is for recognition, not bit-for-bit
# equivalence. The repo also carries the DFlash2 block drafter, pre-quantized, in
# `dflash/` (~1.1 GB): one download gets the model and its speculative decoder,
# with nothing built on first run (mlx_dflash.py).
#
# Qwen3.8-27B is `qwen3_5` — DENSE (64 layers: 48 GatedDeltaNet + 16 full attention),
# so every parameter is on the critical path for every token and shrinking the model
# is the only decode lever there is. The recipe (`q3_e3h5`) is 3-bit group-64
# throughout, except lm_head at 5-bit: with vocab_size 248,320 and tie_word_embeddings
# false, the head is a second full 1.27B-param tensor, and the calibrated GGUF builds
# of this same checkpoint are unanimous that it is the tier worth protecting while
# embed_tokens — a lookup table, whose per-row error never compounds through a matmul
# — is the cheapest. ~12.1 GB resident.
_HF_MODEL = "nathansutton/Qwen3.8-27B-UD-Q3_K_XL-DFlash2-MLX"
# A dev clone that already built the weights locally should use them rather than
# re-download — prefer this dir when present.
_LOCAL_MODEL = os.path.join(_PROJECT_ROOT, "models", "Qwen3.8-27B-q3_e3h5")
# chad targets 24 GB Apple Silicon and nothing smaller. Below this the model still
# loads, but the context governor has almost nothing left to spend after ~12.1 GB of
# weights and the ~4.3 GB prefill transient, so the window collapses toward its floor.
# We warn and proceed rather than refuse: the harness advises, the caller decides.
_MIN_RAM_GB = 23.5


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
SAMPLER_ENV = (("temp", "CHAD_TEMP"), ("min_p", "CHAD_MIN_P"), ("top_p", "CHAD_TOP_P"),
               ("top_k", "CHAD_TOP_K"), ("presence_penalty", "CHAD_PRESENCE_PENALTY"))

# Qwen3.8's two published sampling recipes. The model card gives DIFFERENT settings
# per mode, and the difference is not cosmetic: non-thinking mode has no reasoning
# block to absorb a loop, so the card calls for a presence penalty ("adjust
# presence_penalty between 0 and 2 to reduce endless repetition") and a tighter
# nucleus. Running --no-think while silently inheriting the thinking recipe is the
# configuration the card warns produces endless repetition — and degenerate
# repetition is already this stack's known failure mode, so the preset is applied
# rather than left to the operator to remember.
#
# Explicit CHAD_* env always wins over the preset (see apply_sampler_env).
THINKING_SAMPLER = {"temp": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
                    "presence_penalty": 0.0}
# presence_penalty stays 0.0 despite the card suggesting up to 1.5 for this mode.
# The card's range is written for chat; CODE is inherently repetitive — identifiers,
# keywords and punctuation must be reused — so a flat penalty on every already-emitted
# token pushes the model off valid syntax. Measured on the ky task at 1.5: 45 steps of
# pure exploration, zero edits, and visibly corrupted tool arguments
# (`Search 'KyOptions|interface KyOptions!!'`). Treat it as a tunable to probe, not a
# default to ship: CHAD_PRESENCE_PENALTY still sets it.
NONTHINKING_SAMPLER = {"temp": 0.7, "top_p": 0.80, "top_k": 20, "min_p": 0.0,
                       "presence_penalty": 0.0}


def apply_sampler_preset(eng, thinking: bool):
    """Apply the model-card sampling recipe for the active reasoning mode.

    Call BEFORE apply_sampler_env so an explicit CHAD_* override still wins."""
    for attr, val in (THINKING_SAMPLER if thinking else NONTHINKING_SAMPLER).items():
        setattr(eng, attr, val)


def apply_sampler_env(eng):
    """Apply the sampler-knob environment overrides to `eng`, in place.

    CHAD_TEMP: sampling temperature, all backends. The default stays 0.0 (greedy —
    reproducible, and the MLX prompt-lookup fast path requires it), but greedy has a
    failure mode measured in the field: a stall/garbled call replays itself byte-identically
    on every retry and across "independent" bench reps. Benchmarks and unattended runs
    should set e.g. CHAD_TEMP=0.7 (what the field harnesses run) so retries can take a
    different path.

    CHAD_MIN_P / CHAD_TOP_P / CHAD_TOP_K: quant-tail anti-confabulation knobs, off
    (0.0 / 0) by default — trim the sub-noise-floor logit tail without touching temp.

    CHAD_PRESENCE_PENALTY: flat score penalty on already-generated tokens; the
    model card's anti-repetition knob for non-thinking mode (useful range 0-2).

    Applied AFTER any mode preset, so an explicit env var always wins."""
    for attr, var in SAMPLER_ENV:
        raw = config.env_str(var)
        if not raw:
            continue
        try:
            # top_k indexes a vocab axis and is passed to mx.topk — it must stay an
            # int, where every other knob here is a float. Coercing the whole family
            # to float would make CHAD_TOP_K=20 a TypeError deep in the sampler.
            setattr(eng, attr, int(float(raw)) if attr == "top_k" else float(raw))
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


# Prefill/decode scratch that is live at the same moment the KV cache is: chunked
# prefill materializes an attention transient sized by chunk x kv_len, and the adaptive
# chunker shrinks the chunk as the free band closes, so it climbs with context and then
# SATURATES rather than growing per-token. Measured on the 27B (q3, 8-bit KV, clamp on,
# one load): 1.82 GB at 8k, 3.19 at 32k, 4.15 at 49k, 4.14 at 65k — flat by 49k, and past
# that point peak grows at 33,936 B/token against a 34,816 B/token KV cache, i.e. the
# marginal cost of a token IS its KV cost. Used as a floor only; `_compute_ctx_limit`
# prefers the live `peak - active` once a real prefill has been through.
PREFILL_TRANSIENT_BYTES = 4.3e9


def ram_aware_ctx_limit(eff_ctx, budget_bytes, active_bytes, kv_bytes_per_token,
                        safety=0.975, gen_margin=2048, floor=8192,
                        host_avail_bytes=None, slope_factor=1.0,
                        transient_bytes=PREFILL_TRANSIENT_BYTES):
    """Largest prompt-token budget (= the compaction trigger) that keeps the
    growing KV cache inside a safe slice of the Metal *recommended working set*, given the
    model's already-resident footprint and the *measured* per-token KV cost. Pure +
    measured — replaces the magic `CTX_CAP = 120_000`, an OOM guard set blind to both the
    real per-token cost and the real fixed cost, and so wrong in both directions depending
    on the model.

    `safety` is the ONLY headroom knob, and it holds back 2.5 %. It replaces a pair —
    `safety=0.90` alongside a flat `reserve_gb=1.5` — that guarded one wall with two
    numbers, neither derived from anything measured on the model being run. The rule now
    is that a knob is the wrong home for anything measurable: what the percentage does not
    cover is the prefill transient, and that gets *measured* and subtracted below rather
    than tuned.

    So: `budget*safety`, minus the resident model+SSM floor (`active_bytes`) and the
    prefill transient live alongside the cache (`transient_bytes`), is the bytes free for
    KV; divide by `kv_bytes_per_token` for the token ceiling. Capped at
    `eff_ctx − gen_margin` (the model's real window) and floored so a tight box still gets
    a usable window. Self-calibrates per machine (16 GB → small, 64 GB → near the window).
    Returns None if inputs are unusable so the caller can keep the old fixed cap.

    `slope_factor` multiplies the per-token cost and now defaults to 1.0, because the
    measurement says a token's marginal cost is its KV cost and nothing more. It was 1.75,
    which was the old way of paying for the transient: smear a fixed multi-gigabyte term
    across the per-token divisor and it comes out roughly right at ONE context length on
    ONE box, and wrong everywhere else — over-charging long windows, under-charging short
    ones. The term is fixed, so it is subtracted like one. Kept as a knob for A/B only."""
    if not (budget_bytes and kv_bytes_per_token and active_bytes):
        return None
    usable = budget_bytes * safety - active_bytes - (transient_bytes or 0)
    # The Metal budget is blind to other processes' physical pressure (Docker VM,
    # harbor, browsers). When the host's reclaimable band is tighter than the Metal
    # band, IT is the binding constraint — the KV cache grows into physical pages
    # 1:1, and jetsam kills on physical pressure, not on Metal accounting.
    if host_avail_bytes:
        # NOT symmetric with the Metal branch, on purpose: the transient is not charged
        # here. Metal is a hard allocator wall — the scratch either fits or the allocation
        # fails — whereas the host band is a soft pressure signal the OS can compress and
        # evict around for a spike that lives for one prefill chunk. What it cannot absorb
        # is the KV cache, which stays resident for the session, so that is what this
        # branch sizes. Charging the transient here instead turns a guard meant to bite
        # only under real pressure into the primary constraint: measured right after a
        # 12.3 GB load the reclaimable band reads 3.9 GB (down from 11.4 — the weights
        # just took it, and nothing has been reclaimed yet), and subtracting a 4.3 GB
        # transient from that lands negative, i.e. the floor, on a box with room to spare.
        usable = min(usable, host_avail_bytes * safety)
    if usable <= 0:
        return floor
    ram_ctx = int(usable / (kv_bytes_per_token * slope_factor))
    return max(floor, min(eff_ctx - gen_margin, ram_ctx))


def _compute_ctx_limit(eng):
    """The auto-compaction threshold for a loaded engine. On this non-trimmable hybrid
    cache, compaction forces a full body re-prefill (~79 % of all prefill), so
    we compact as rarely as RAM safely allows: size the trigger from the live Metal
    budget + the model's *measured* per-token KV cost instead of a blind 120 k
    cap that over-compacts. CHAD_CTX_LIMIT still wins (used by tests); CHAD_CTX_SAFETY is
    the single headroom knob (default 0.975 — hold back 2.5 % of the Metal budget). Falls
    back to the old fixed cap if the memory APIs or the KV measurement are unavailable.
    Needs eng.load() to have run (reads effective_ctx + kv_bytes_per_token)."""
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
                safety=_env_float("CHAD_CTX_SAFETY") or 0.975,
                host_avail_bytes=_host_avail_bytes(),
                slope_factor=_env_float("CHAD_CTX_SLOPE_FACTOR") or 1.0)
            # The governor sets the shape of the whole session and used to explain
            # nothing, which made a wrong pick indistinguishable from a tight box.
            log.info("GOVERNOR ctx_limit=%s | budget=%.2f GB active=%.2f GB "
                     "transient=%.2f GB host_avail=%.2f GB kv=%.0f B/tok",
                     f"{ctx_limit:,}" if ctx_limit else ctx_limit,
                     mx.device_info()["max_recommended_working_set_size"] / 1e9,
                     active_floor / 1e9, PREFILL_TRANSIENT_BYTES / 1e9,
                     (_host_avail_bytes() or 0) / 1e9, eng.kv_bytes_per_token)
        except Exception:  # noqa: BLE001 — never let memory probing break startup
            ctx_limit = None
    if not ctx_limit:
        ctx_limit = min(max(4096, eng.effective_ctx - 2048), 120_000)  # old fixed cap
    return ctx_limit


def peek_ctx_limit(model_id, window):
    """The compaction trigger estimated BEFORE the weights load, so the banner can
    advertise the window this session actually gets instead of a native window the box
    cannot hold. Same governor and the same live memory probes as `_compute_ctx_limit` —
    only the two model-side inputs are config/disk estimates rather than measurements
    (`engine.peek_kv_footprint`), and the measured value replaces this the moment load
    returns. Returns None when the estimate isn't available, leaving the caller on the
    old window-derived provisional."""
    if not window:
        return None
    try:
        import mlx.core as mx

        from .engine import peek_kv_footprint
        kv_bpt, weights = peek_kv_footprint(model_id)
        if not (kv_bpt and weights):
            return None
        # The weights are not resident yet, so the host's reclaimable band still counts
        # the pages they are about to take. Subtract them, or this reads the box as
        # roomier than the session will ever see it.
        avail = _host_avail_bytes()
        return ram_aware_ctx_limit(
            window, mx.device_info()["max_recommended_working_set_size"],
            weights, kv_bpt,
            safety=_env_float("CHAD_CTX_SAFETY") or 0.975,
            host_avail_bytes=max(0, avail - weights) if avail else None,
            slope_factor=_env_float("CHAD_CTX_SLOPE_FACTOR") or 1.0)
    except Exception:  # noqa: BLE001 — never let a banner estimate break startup
        return None


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

    Order: explicit `--model` (`spec`) → CHAD_MODEL → the shipped default. There are no
    size shorthands any more — chad ships exactly one model (2.0.0 retired the Ornith
    35B/9B pair and the RAM-aware pick that chose between them). `auto` still means "the
    default"; anything else is passed through untouched as an HF repo id or local dir.

    A box below the 24 GB target is warned about once, on stderr, and then served
    anyway: chad advises, the caller decides.
    """
    # Name the winning source in the reason: it is only ever ambiguous when both the
    # flag and the env var are set, which is exactly when the user needs to be told.
    source = "--model" if spec is not None else "CHAD_MODEL"
    spec = spec or config.env_str("CHAD_MODEL")
    if spec and spec.strip().lower() != "auto":
        return spec, f"explicitly requested ({source} override)"
    ram = _detect_ram_gb()
    if ram is None or ram < _MIN_RAM_GB:
        got = "undetectable" if ram is None else f"{ram:.0f} GB"
        sys.stderr.write(
            f"chad: RAM {got}, below the ~{_MIN_RAM_GB:.0f} GB chad is built for. The "
            f"model needs ~12 GB resident plus its KV cache, so expect a small context "
            f"window and possible thrashing. Proceeding.\n")
    return _resolve(_LOCAL_MODEL, _HF_MODEL), "default"


def _model_download_gb(model_id):
    """Approximate download size in GiB for the shipped model (for the disk preflight
    and the confirm prompt — display honesty, not accounting): ~12.1 GB of weights
    plus the ~1.1 GB bundled DFlash2 drafter. An arbitrary `--model` is unknowable
    ahead of the resolve, so it gets the same figure."""
    return 13.2


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


def _cached_weights_complete(model_id) -> bool:
    """Whether the HF cache holds this repo's WEIGHTS, not merely its small files.

    The cheap check ("is config.json cached?") is wrong in the one case that matters.
    A download interrupted after the json/tokenizer blobs land — one ctrl-c on a first
    run — leaves a snapshot that answers yes while holding no tensors at all, so the
    guard returns, and the failure surfaces minutes later inside mlx_lm as a
    FileNotFoundError naming an internal blob path. There is no way back from that
    state except hand-deleting the cache. So verify what the loader will actually
    read: every shard the index names, or a single-file/loose layout on disk.
    """
    from huggingface_hub import try_to_load_from_cache
    cached = lambda f: isinstance(try_to_load_from_cache(model_id, f), str)  # noqa: E731
    index = try_to_load_from_cache(model_id, "model.safetensors.index.json")
    if isinstance(index, str):
        try:
            with open(index, encoding="utf-8") as f:
                shards = set(json.load(f).get("weight_map", {}).values())
        except (OSError, ValueError):
            return False  # unreadable index: treat as incomplete, re-fetch resumes
        return bool(shards) and all(cached(s) for s in shards)
    if cached("model.safetensors"):
        return True
    # No index and no conventionally-named file: an unusual layout is not our business
    # to second-guess, so accept any .safetensors already sitting in the snapshot (this
    # is what mlx_lm globs for) rather than forcing a re-download of a working cache.
    config_path = try_to_load_from_cache(model_id, "config.json")
    if isinstance(config_path, str):
        import glob
        return bool(glob.glob(os.path.join(os.path.dirname(config_path), "*.safetensors")))
    return False


def _ensure_model(model_id):
    """If model_id is a HF repo id not yet in the local cache, confirm and download it
    into ~/.cache/huggingface (shared, resumable, paid once per machine). Local dirs
    and already-cached repos return immediately. Headless (no TTY) auto-downloads.
    Preflights free disk BEFORE starting: a 12 GB download that dies at 70% on a full
    disk is the worst first-run outcome (devex review T2)."""
    if os.path.isdir(model_id):
        return  # a local path — nothing to fetch
    from huggingface_hub import snapshot_download
    if _cached_weights_complete(model_id):
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
    # Say WHICH of the two situations this is. "Downloading again" on a machine the
    # user believes already has the model reads as a bug unless the partial cache is
    # named; only the completed blobs are re-used, so the second run is also shorter.
    from huggingface_hub import try_to_load_from_cache
    partial = isinstance(try_to_load_from_cache(model_id, "config.json"), str)
    sys.stderr.write(
        (f"\nchad: the cached copy of '{model_id}' is incomplete (an interrupted "
         f"download left its metadata but not all of its weights).\nResuming — only "
         f"the missing files are fetched, up to {size}.\n" if partial else
         f"\nchad needs the model '{model_id}' "
         f"({size} — minutes on fast fiber, ~20 min on 100 Mbit; resumable).\n"
         "It downloads once into ~/.cache/huggingface and is reused across projects.\n"))
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
            "         download resumes). chad needs ~12 GB resident for weights alone,\n"
            "         so close other memory-hungry apps before retrying.\n")
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
    need the same escape hatch from the shipped default."""
    ap.add_argument("--model", default=None,
                    help="which model to load: 'auto' (the shipped default) or any "
                         "Hugging Face repo id / local model dir. Other weights run "
                         "through the same engine; the tuning is fitted to the shipped "
                         "model, so expect to lose speed, not correctness. "
                         "Also CHAD_MODEL.")


def _agent_parser():
    ap = argparse.ArgumentParser(
        prog="chad",
        description="Local MLX-backed coding agent. Run with `uv run chad`.",
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
        description="Run the bundled end-to-end smoke test against the shipped model: "
                    "downloads it if needed, drives a real task, and reports what worked.",
    )
    ap.add_argument("--backend", choices=("mlx", "llama"), default="mlx",
                    help=argparse.SUPPRESS)  # see _serve_parser
    return ap


def _levers_parser():
    return argparse.ArgumentParser(
        prog="chad levers",
        description="Print the harness lever registry as JSON and exit. The ablation "
                    "driver enumerates this instead of hardcoding lever names. All "
                    "levers default ON; CHAD_DISABLE=a,b (or 'all') switches levers "
                    "off for a leave-one-out arm.",
    )


def _run_levers():
    """No _preflight and no model: an ablation driver enumerating levers should not need
    an Apple-Silicon box or a loadable model just to read the registry."""
    print(json.dumps({"levers": levers.as_dict(),
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

    # Fail fast on a typo'd CHAD_DISABLE, not mid-run: an unrecognized lever
    # means the harness would run unmodified while an ablation reports "no effect".
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
    # --model / CHAD_MODEL, else the shipped default; local-dir-preferred, HF fallback.
    model_id, why = _pick_model(args.model)

    # Advanced, rarely-touched knobs live in env vars to keep the CLI sane:
    #   CHAD_MAX_CONTEXT       YaRN-extend the window (e.g. 131072 for 128k)
    #   CHAD_CTX_LIMIT         prompt-token budget before old tool outputs compact
    #   CHAD_KV_BITS           KV cache quantization; default AUTO (8-bit where the
    #                          fused kernel covers the model, as it does the shipped
    #                          one). 0 forces the fp16 cache.
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

    # Mode preset first, explicit env second — so CHAD_* always wins. `--no-think`
    # switches the whole recipe, not just the reasoning block: see NONTHINKING_SAMPLER
    # for why running it on the thinking recipe is the documented repetition case.
    apply_sampler_preset(eng, thinking=not args.no_think)
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
        # The banner states the EFFECTIVE window — what the RAM governor will hand this
        # session — not the model's native one. On a box where the two differ by 2.5x,
        # advertising the native number is advertising context the run cannot spend.
        provisional = ctx_limit or _env_int("CHAD_CTX_LIMIT") \
            or peek_ctx_limit(model_id, window) \
            or min(max(4096, (window or 32768) - 2048), 120_000)

        def finalize():
            load_s = eng.load()
            return load_s, _compute_ctx_limit(eng)

        run_tui(eng, provisional, mode=start_mode, thinking=thinking, resume=resume,
                ctx_window=provisional, native_ctx=window, finalize=finalize,
                ctx_limit_fn=ctx_limit_fn)


if __name__ == "__main__":
    main()
