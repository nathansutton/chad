"""
MLX inference engine for the chad backend.

The thing that makes this feel like Claude Code's backend is the **persistent prefix
KV cache**: each agentic turn re-renders the whole transcript but only prefills the
*new* tokens appended since last turn, so multi-step tool loops stay snappy instead of
re-reading the whole conversation every step. We keep the KV cache alive across turns
and extend it by diffing token ids (see `_sync_to` / `generate`).

Two decode-speed levers were investigated (see the README's throughput section):

1. **Prompt-lookup decoding (PLD)** — draft-model-free speculative decoding, gated on a
   *trimmable* cache (`engine._trimmable`) and temp==0. It is provably greedy-identical
   and helps trimmable models on quote-heavy work, but the shipped hybrid SSM/attention
   cache is non-trimmable, so PLD falls back cleanly and does not speed up the shipped
   model. (`prompt_lookup_draft` + `_generate_prompt_lookup`.)
2. **Thinking budget** — `--no-think` drops the model's `<think>` overhead; the most
   effective real speedup for time-to-done, since decode is bandwidth-bound (~47 tok/s).

chad is single-model by construction: there is no second "draft" model anywhere in the
engine. PLD gets speculative decoding's accept/rollback benefit from the context itself.
"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

# MLX is Apple-only (no CPU/CUDA build), and the whole `Engine` class below rides on it.
# But two module-level helpers here — `sweep_orphan_spills` and `peek_context_window` —
# are MLX-free and ARE needed on the remote `--backend llama` path, which loads no
# MLX at all. Guard the imports so `import chad.engine` succeeds on a non-Apple host (e.g.
# inside a Linux container that runs chad against a remote server). `Engine`
# itself is only ever CONSTRUCTED on the default MLX path, where these are present; if a
# remote-only host somehow builds one, it fails fast on the first `mx.` use.
try:
    import mlx.core as mx
    from mlx_lm import load, stream_generate
    from mlx_lm.models import cache as cache_utils
    from mlx_lm.sample_utils import apply_min_p, apply_top_p, make_sampler
    _HAS_MLX = True
    _MLX_IMPORT_ERROR: Optional[BaseException] = None
except ImportError as _e:  # non-Apple host: remote backend only
    # `unused-ignore` because the `assignment` ignore is only *needed* where mlx is
    # installed (mac). On the Linux lint runner mlx is absent, `ignore_missing_imports`
    # types these as Any, and the bare ignore would trip `warn_unused_ignores`.
    mx = None  # type: ignore[assignment, unused-ignore]
    load = stream_generate = cache_utils = make_sampler = None  # type: ignore[assignment, unused-ignore]
    apply_min_p = apply_top_p = None  # type: ignore[assignment, unused-ignore]
    _HAS_MLX = False
    # Stash the real cause. A *missing* mlx is the benign Linux case; a mlx that
    # is present but fails to import (e.g. a half-installed mlx-metal wheel whose
    # libmlx.dylib got dropped by a partial `uv sync`) is a broken Apple env, and
    # load() below raises this instead of nulling `load` and dying 300 lines later
    # with a bare `TypeError: 'NoneType' object is not callable`.
    _MLX_IMPORT_ERROR = _e

# GenStats moved to base_engine.py so a non-MLX backend can build one without
# importing mlx.core. Re-exported here so existing `from .engine import GenStats` keeps
# working (bench.py, tests) — the class is unchanged.
from . import config
from .base_engine import THINK_CLOSE, GenStats, think_ceiling_hit
from .diag import log

# checkpoint filename kinds (prefix on the basename) — lets cleanup target the
# ephemeral push-spills without touching durable warm-prefix files.
_CKPT_WARM = "warm"
_CKPT_PUSH = "push"


def _log_mlx_provenance() -> None:
    """Record whether mlx is the PyPI build a `uvx chad-code` user gets, or a
    locally built one.

    Perf work on this repo has twice been measured against a patched local wheel
    whose kernels no PyPI user receives (chad-code depends on `mlx>=0.32,<0.33`
    from PyPI, and `uvx --from git+...` resolves fresh and ignores uv.lock). A
    local build shows up as a PEP 440 local version segment ('+<sha>'). Logging
    it means a throughput number can always be traced to the build that produced
    it, instead of silently describing a configuration we do not ship.
    """
    try:
        import mlx.core as mx
        ver = str(mx.__version__)
    except Exception:  # noqa: BLE001 — diagnostics must never break loading
        return
    if "+" in ver:
        log.warning("mlx %s is a LOCAL build, not the PyPI wheel users get — "
                    "throughput measured here may not be reproducible via "
                    "`uvx chad-code`", ver)
    else:
        log.info("mlx %s (PyPI build — matches what uvx installs)", ver)


def _local_path(model_id: str) -> str:
    """Resolve a cached HF repo id to its on-disk snapshot dir so `mlx_lm.load` (and
    `_read_config`) skip the hub revision check — a ~1s network/stat round-trip on every
    launch, pure overhead once the weights are local. A local dir or an uncached id
    passes through unchanged; the uncached case is downloaded by `cli._ensure_model`
    before `load()` runs, so by then it's a cache hit here too."""
    if os.path.isdir(model_id):
        return model_id
    try:
        from huggingface_hub import try_to_load_from_cache
        hit = try_to_load_from_cache(model_id, "config.json")
        if isinstance(hit, str):
            return os.path.dirname(hit)
    except Exception:  # noqa: BLE001 — never let a resolver hiccup block a real load
        pass
    return model_id


def peek_context_window(model_id: str, max_context: Optional[int] = None) -> Optional[int]:
    """The model's context window read from `config.json` alone — no weights, no
    tokenizer. Lets the startup banner state the window immediately while the weights
    load in the background. Mirrors the default-path result of `_ctx_override`: the
    native window (nested under `text_config` on VL checkpoints), YaRN-extended to
    `max_context` when that's larger. Returns None if the config can't be read."""
    try:
        import json
        path = _local_path(model_id)
        cfg_path = os.path.join(path, "config.json")
        if not os.path.isfile(cfg_path):
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(model_id, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        native = (cfg.get("max_position_embeddings")
                  or cfg.get("text_config", {}).get("max_position_embeddings")
                  or 32768)
        return max(native, max_context) if max_context else native
    except Exception:  # noqa: BLE001 — banner falls back to "context tbd"
        return None


def peek_kv_footprint(model_id: str) -> tuple:
    """`(kv_bytes_per_token, weight_bytes)` from `config.json` plus the on-disk weight
    file sizes alone — no weights loaded, no MLX allocation. These are the two model-side
    inputs the RAM governor needs, so the startup banner can state the window the box
    will ACTUALLY give this session instead of advertising a native window it cannot
    hold. Both are re-derived exactly once the weights are in (the measured
    `_measure_kv_bytes_per_token` and a live `mx.get_active_memory()`); this is a
    deliberate pre-load estimate of the same two quantities.

    Only the attention layers grow with context — on the hybrid checkpoints most layers
    are `linear_attention` and hold a fixed recurrent state — so the per-token cost is
    `n_full_attention * n_kv_heads * head_dim * 2 (K+V)`, at the width `_resolve_kv_bits`
    will pick for this shape. Returns `(None, None)` if the config or the weight files
    can't be read locally, which puts the caller back on the model window."""
    try:
        import glob
        import json

        from . import mlx_qsdpa
        path = _local_path(model_id)
        cfg_path = os.path.join(path, "config.json")
        if not os.path.isfile(cfg_path):
            from huggingface_hub import hf_hub_download
            cfg_path = hf_hub_download(model_id, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        tc = cfg.get("text_config", cfg)
        n_layers = int(tc.get("num_hidden_layers") or 0)
        types = tc.get("layer_types") or []
        if types:
            n_full = sum(1 for t in types if "full" in str(t))
        elif tc.get("full_attention_interval"):
            n_full = n_layers // int(tc["full_attention_interval"])
        else:
            n_full = n_layers          # no hybrid marker: every layer is attention
        n_q = int(tc.get("num_attention_heads") or 0)
        n_kv = int(tc.get("num_key_value_heads") or n_q)
        head_dim = int(tc.get("head_dim") or 0)
        if not head_dim and n_q and tc.get("hidden_size"):
            head_dim = int(tc["hidden_size"]) // n_q
        if not (n_full and n_kv and head_dim):
            return None, None
        # Mirror _resolve_kv_bits: 8-bit group-64 exactly when the fused decode kernel
        # covers the shape, else the fp16 cache. Read from the same `covers` predicate so
        # the estimate cannot drift from the decision — but WITHOUT `mlx_qsdpa.install()`,
        # whose side effects belong to load, not to drawing a banner. If install then
        # fails, load's own measurement corrects this downward.
        gqa = n_q // n_kv if n_kv else 0
        env_bits = config.env_int("CHAD_KV_BITS", -1)
        bits = env_bits if env_bits >= 0 else (8 if mlx_qsdpa.covers(head_dim, gqa) else 0)
        if bits == 8:
            # packed 8-bit weights + one fp16 scale and bias per group of 64
            per_head = head_dim + (head_dim / 64) * 4
        elif bits:
            per_head = head_dim * bits / 8 + (head_dim / 64) * 4
        else:
            per_head = head_dim * 2    # fp16 cache
        kv_bpt = n_full * n_kv * per_head * 2      # K and V
        weights = sum(os.path.getsize(f)
                      for f in glob.glob(os.path.join(path, "*.safetensors")))
        return kv_bpt, (weights or None)
    except Exception:  # noqa: BLE001 — an estimate for a banner is never worth a crash
        return None, None


def prompt_lookup_draft(context, num_draft, ngram_max=3, ngram_min=1):
    """Prompt-lookup (n-gram) drafting: propose the next `num_draft` tokens by
    finding the most recent earlier occurrence of the current suffix in `context`
    and copying what followed it. Returns [] when nothing matches.

    This is "speculative decoding with the context as the draft model": when the
    main model is about to reproduce text already present in the conversation —
    exactly what happens when it quotes file content into an `edit`/`write`, or
    re-states code it just `read` — the n-gram lookup nails the continuation and a
    single verify forward accepts a long run of tokens. Zero extra RAM, no draft
    model, and on bandwidth-bound Apple Silicon a 10-token verify pass costs about
    the same wall time as a 1-token step, so misses are nearly free.

    Longest match wins (try ngram_max down to ngram_min); ties broken by recency
    (rightmost occurrence), which favors the just-read file over older context.
    """
    n = len(context)
    if n < 2 or num_draft <= 0:
        return []
    arr = np.asarray(context)
    hi = min(ngram_max, n - 1)
    for ng in range(hi, ngram_min - 1, -1):
        pat = arr[-ng:]
        # candidate starts: positions (excluding the trailing suffix itself) whose
        # first token matches pat[0]; verify the full n-gram, most-recent first.
        cand = np.nonzero(arr[: n - ng] == pat[0])[0]
        for i in cand[::-1]:
            if np.array_equal(arr[i : i + ng], pat):
                draft = context[i + ng : i + ng + num_draft]
                if draft:
                    return list(draft)
    return []


def prompt_lookup_draft_arr(arr, n, num_draft, ngram_max, ngram_min):
    """Array variant of prompt_lookup_draft for the wide-PLD loop: `arr` is a
    preallocated int buffer with `n` live entries (so the per-step lookup does
    not rebuild a numpy array from a growing python list). Returns
    (draft ids, matched n-gram length); ([], 0) when no suffix of length >=
    ngram_min recurs earlier in the context.

    The full n-gram match is vectorized (rolling AND over shifted compares)
    instead of the scalar candidate walk in prompt_lookup_draft: common first
    tokens (newline, space) can have thousands of candidate positions, and
    this runs every decode step."""
    if n < 2 or num_draft <= 0:
        return [], 0
    hi = min(ngram_max, n - 1)
    for ng in range(hi, ngram_min - 1, -1):
        pat = arr[n - ng : n]
        m = arr[: n - ng] == pat[0]
        for j in range(1, ng):
            m &= arr[j : n - ng + j] == pat[j]
        idx = np.nonzero(m)[0]
        if idx.size:
            i = int(idx[-1])  # rightmost: favor the just-read file
            draft = arr[i + ng : min(i + ng + num_draft, n)]
            if draft.size:
                return [int(t) for t in draft], ng
    return [], 0


def sweep_orphan_spills(cache_dir: str, max_age_s: float) -> int:
    """Delete push-spill checkpoints older than max_age_s. A push-spill lives only
    for the duration of an active sub-agent (seconds–minutes) and is removed by
    pop_cache on the clean path; anything older was orphaned by a killed/crashed
    process and is dead weight. Returns bytes freed. Never raises."""
    freed = 0
    try:
        deadline = time.time() - max_age_s
        for name in os.listdir(cache_dir):
            if not name.startswith(_CKPT_PUSH + "-") or not name.endswith(".safetensors"):
                continue
            path = os.path.join(cache_dir, name)
            try:
                st = os.stat(path)
                if st.st_mtime < deadline:
                    os.remove(path)
                    freed += st.st_size
            except OSError:
                pass
    except OSError:
        pass
    return freed


def enforce_cache_budget(cache_dir: str, max_bytes: int, protect: set) -> int:
    """Keep the checkpoint dir under max_bytes by deleting least-recently-modified
    files first. Files in `protect` (absolute paths currently referenced by a live
    session) are never deleted. max_bytes <= 0 disables the cap. Returns bytes freed.
    Never raises."""
    if max_bytes <= 0:
        return 0
    freed = 0
    try:
        entries = []
        total = 0
        for name in os.listdir(cache_dir):
            if not name.endswith(".safetensors"):
                continue
            path = os.path.join(cache_dir, name)
            try:
                st = os.stat(path)
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, path))
            total += st.st_size
        if total <= max_bytes:
            return 0
        entries.sort()  # oldest st_mtime first
        for _mtime, size, path in entries:
            if total <= max_bytes:
                break
            if path in protect:
                continue
            try:
                os.remove(path)
                total -= size
                freed += size
            except OSError:
                pass
    except OSError:
        pass
    return freed


def apply_top_k(scores, k: int):
    """Keep the k highest-scoring tokens, -inf the rest. Pure; mirrors the shape
    contract of mlx_lm's apply_top_p/apply_min_p (last axis is the vocab, leading
    axes are batch rows) so it composes with them in either order.

    Qwen3.8 asks for top_k=20 in BOTH sampling recipes, and it is the one filter
    mlx_lm does not ship, so it lives here rather than being silently skipped."""
    if not k or k <= 0 or k >= scores.shape[-1]:
        return scores
    # Cut everything below the kth largest. Take the MIN of the top-k block
    # rather than indexing an end of it: mx.topk's ordering within the block is
    # not part of its contract (it returns ascending today, so [..., -1:] would
    # be the maximum and keep exactly one token). Ties at the threshold are kept
    # (>= cutoff), which can leave slightly more than k candidates — the safe
    # direction: never drop the argmax, never empty a row.
    cutoff = mx.min(mx.topk(scores, k, axis=-1), axis=-1, keepdims=True)
    return mx.where(scores >= cutoff, scores, mx.array(-float("inf"),
                                                       dtype=scores.dtype))


def apply_presence_penalty(scores, seen_ids, penalty: float):
    """Subtract `penalty` from the score of each DISTINCT id in `seen_ids`
    (any sequence of ints, or an mx array).

    Presence, not frequency: the penalty applies ONCE per distinct token no
    matter how often it appeared, which is what "presence_penalty" means and what
    the Qwen recipe specifies. Duplicates would otherwise stack into a frequency
    penalty — a different (and much harsher) filter — so they are removed here
    rather than at every call site.

    Cost is O(distinct seen), not O(vocab): a scatter over a few hundred ids
    instead of building a 248k-wide adjustment vector every token."""
    if not penalty or seen_ids is None:
        return scores
    ids = sorted({int(t) for t in seen_ids})
    if not ids:
        return scores
    uniq = mx.array(ids, dtype=mx.uint32)
    if scores.ndim == 1:
        return scores.at[uniq].add(-penalty)
    # batched rows (verify forward): penalize the same ids in every row; callers
    # needing per-row histories call this once per row.
    return scores.at[..., uniq].add(-penalty)


class _KeyedSampler:
    """Categorical sampler on an explicit PRNG key chain.

    MLX's implicit-key RNG never advances on non-main threads: every
    mx.random.categorical() there reuses the same frozen noise, so temp>0
    "sampling" replays one draw per position — byte-identical responses across
    identical requests AND loop-prone quasi-greedy decoding (any worker-thread
    generation hits it; the deterministic replay class). Per-request
    mx.random.seed() cannot fix it because the seeded state is never consumed
    off the main thread. Splitting an explicit key per draw sidesteps global
    RNG state entirely and behaves identically on every thread. Measured on
    mlx 0.32.0: worker-thread draws [7,7,7,...] implicit vs varied explicit.
    """

    def __init__(self, temp: float, seed: Optional[int] = None,
                 min_p: float = 0.0, top_p: float = 0.0, top_k: int = 0,
                 presence_penalty: float = 0.0, seen: Optional[list] = None):
        self._inv_t = 1.0 / temp
        self._min_p = min_p
        self._top_p = top_p
        self._top_k = top_k
        self._presence = presence_penalty
        # Read-only view of the engine's generated-token list, for the presence
        # penalty. The engine owns it and appends already-materialized ints as it
        # commits them; the sampler must never append the token it just drew,
        # which is a lazy array — realizing it here would force a host sync every
        # single token purely for bookkeeping.
        self.seen: list = seen if seen is not None else []
        self._key = mx.random.key(
            int.from_bytes(os.urandom(4), "little") if seed is None else seed)

    def __call__(self, logprobs):
        self._key, sub = mx.random.split(self._key)
        scaled = logprobs * self._inv_t
        # Penalty BEFORE the tail filters: it reshapes the distribution the
        # filters then trim, so a heavily-penalized repeat can fall out of the
        # top-k/top-p set entirely — which is the point. Doing it after would let
        # a loop token survive the cut and only then be nudged.
        if self._presence:
            scaled = apply_presence_penalty(scaled, self.seen, self._presence)
        # (this path receives already-scaled logprobs from mlx_lm, so the penalty
        # lands post-temp here; _spec_scaled owns the pre-temp ordering that the
        # model card's numbers assume. Kept consistent by keeping the DEFAULT 0.)
        if self._top_k:
            scaled = apply_top_k(scaled, self._top_k)
        if 0 < self._top_p < 1.0:
            scaled = apply_top_p(scaled, self._top_p)
        if self._min_p != 0.0:
            scaled = apply_min_p(scaled, self._min_p)
        return mx.random.categorical(scaled, key=sub)


@dataclass
class Engine:
    model_id: str = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
    temp: float = 0.0
    # Sub-noise-floor tail filters: quantized weights have a higher logit
    # noise floor than the bf16 reference this temp=1.0 recipe was validated on, so
    # pure ancestral sampling occasionally draws a garbage token that autoregression
    # then amplifies into a derailed <think>. min_p/top_p trim that tail while leaving
    # on-policy sampling untouched. Both default OFF (0.0 = no-op); off by default
    # pending an eval-gated sign-off before flipping the shipped default.
    min_p: float = 0.0
    top_p: float = 0.0
    top_k: int = 0                  # keep only the k highest-scoring tokens (0 = off)
    # Presence penalty: a flat subtraction from the score of every token ALREADY
    # generated in this turn (OpenAI semantics — once-seen is once-penalized, the
    # count does not matter). Off by default for thinking mode, but it is the
    # mechanism the model card calls for in NON-thinking mode ("adjust
    # presence_penalty between 0 and 2 to reduce endless repetition"), where there
    # is no reasoning block to absorb a loop and degenerate repetition is the
    # documented failure. Applied INSIDE the sampling distribution, so speculative
    # paths stay exact: the penalized distribution simply IS the target p that
    # rejection sampling preserves (see _spec_scaled, which penalizes each verify
    # row against the tokens preceding it).
    presence_penalty: float = 0.0
    # KV cache quantization. None = AUTO: 8-bit when the fused decode kernel
    # (mlx_qsdpa) covers the model's attention shape — the shipped
    # models qualify — else off. 0 forces off; an explicit bit width forces on
    # (an uncovered shape then decodes via mlx_lm's slow unfused path). The
    # cache is quantized FROM THE START (follow-on): prefill runs
    # dequant + fused fp16 attention, decode runs the fused quantized kernel,
    # and the cache's layer types never change mid-session — which is what
    # lets the rewind machinery below stay enabled.
    kv_bits: Optional[int] = None
    max_context: Optional[int] = None  # request a context window; YaRN-extends if > native
    prompt_lookup: bool = True      # n-gram prompt-lookup speculative decoding (no draft model)
    pld_num_draft: int = 10         # tokens to draft per forward via n-gram lookup
    pld_ngram: int = 3              # max suffix length to match for drafting
    # PLD on a hybrid (qwen3_5) cache is correct (bit-exact) but OFF by default:
    # the eval suite measured it ~2x SLOWER on realistic agentic generation. A recurrent
    # model can't rewind to mid-forward, so every partial/total draft rejection costs an
    # extra re-feed forward; on novel-text-heavy work (low n-gram acceptance) that's ~2
    # forwards/token. It only wins on quote-heavy spans (re-emitting a just-read file).
    # Opt in for that workload; the default standard path is faster for general use.
    enable_pld_hybrid: bool = False
    # Wide prompt-lookup on the hybrid (Qwen3.8-class): a separate, on-by-default
    # path that removes both reasons enable_pld_hybrid stayed off. (1) Rejection
    # rollback uses the MTP capture-replay collector (no re-feed forward).
    # (2) Drafting is gated on n-gram MATCH LENGTH instead of periodic blind
    # probes, so cold (novel-text) steps pay only a host-side lookup — the
    # measured S>1 cost cliff makes a missed speculative forward expensive, so
    # the gate must be evidence-first. Drafts go wide (up to pld_wide_draft)
    # because mx.quantized_matmul's batched path is FLAT from S=12 to S=32 on
    # this machine: verifying 31 draft tokens costs the same forward as 11, so
    # once the n-gram evidence clears the bar, maximum width is free. Exact at
    # any temp: point-mass rejection sampling (accept w.p. p(draft)), the same
    # correction _generate_mtp ships. Takes precedence over MTP on the hybrid;
    # MTP is the fallback when this is disabled or ineligible.
    # Gate sizing is measured, not guessed: replaying this exact matcher over real
    # dogfood traces gives accepted-tokens-per-verify, and a wide verify only pays for
    # itself above ~7.5 (the S=32-forward vs plain-step cost ratio on this machine).
    # A 6-token match in tens of thousands of tokens of code context is not evidence —
    # it fired constantly and landed 6.50 accepted/verify, i.e. BELOW break-even. A
    # 16-token match is evidence: 11.22 accepted/verify. Requiring one exact length
    # (min == max) also collapses the descending-ngram search to a single vectorized
    # pass, which is 4x cheaper host-side than the old 6..12 ladder — so this is both a
    # better gate and a cheaper one. Longer maxima (24, 32) were swept and rejected:
    # they add ~1% acceptance and 6x the host-side lookup cost, a net loss per step.
    pld_wide: bool = True
    pld_wide_draft: int = 31        # max draft width (S=32 verify)
    pld_wide_ngram: int = 16        # longest suffix length to try matching
    pld_wide_min_ngram: int = 16    # shortest match that justifies a wide verify
    pld_wide_min_draft: int = 8     # skip verify if the source span is shorter
    # MTP self-speculative decoding: draft with the checkpoint's own trained
    # multi-token-prediction head (loaded as a sidecar — mlx_mtp.py), verify in
    # one batched forward, exact rejection sampling at any temp. Engages only
    # when a head loaded (Qwen3.8-class models); CHAD_NO_MTP disables, and
    # CHAD_MTP_DRAFT overrides the draft width. Default 2: the 27B sweep
    # (M4 Pro, temp 1.0, coding prompt) measured 19.4 tok/s at k=2 vs 17.3 at
    # k=3 and 12.3 at k=5 — mlx 0.32's S>1 forward costs ~1.5x plain at S=3
    # and ~2x at S=4, so deeper drafts lose to verify cost even at good
    # acceptance. Revisit when the small-batch kernel story changes.
    mtp_num_draft: int = 2
    # Adaptive draft schedule (cost-model, mlx_mtp.DepthPolicy): per round,
    # draft the depth 0..mtp_max_draft that maximizes expected committed
    # tokens per unit round time under per-position acceptance EMAs and the
    # measured head-step/verify cost ratio mtp_h. Hot prose runs deep, cold
    # prompts collapse to 1 or a free skip (0); the fixed-k sweep behind
    # mtp_num_draft is the degenerate constant-depth version. Default ON by
    # measurement (M4 Pro, 4-bit 27B, 8-prompt 512-token arena protocol):
    # the verify ladder pays ~25 ms PER ROW through S=8 but is FLAT from
    # S=10 to S=32 (the tiled-GEMM plateau, holds at 12k ctx), and the
    # plateau-aware argmax beat fixed k=2 on 8/8 prompts — median 29.1 vs
    # 26.5 tok/s (1.90x vs 1.73x over serial), +25-40% on hot prose (6-7
    # committed tokens/round), +23% on the hardest prompt (the schedule
    # collapses to depth 1-2 or a free skip where acceptance drops, so
    # low-acceptance regimes like temp-1 thinking degrade to k2-like
    # behavior, not below it). CHAD_MTP_ADAPTIVE=0 restores fixed-width;
    # CHAD_MTP_DRAFT forces a width (and implies adaptive off);
    # CHAD_MTP_MAX_DRAFT / CHAD_MTP_H override the cap and cost seed.
    mtp_adaptive: bool = True
    # 31 = verify width 32, the far edge of the measured FLAT tile plateau
    # (S>=10 verifies cost ~the same through S=32; see DepthPolicy). The
    # policy itself keeps rounds at width <= 6 (the fused-SDPA envelope)
    # until a full-accept streak qualifies a plateau jump, so cold or hard
    # content never sees a wide round. Width >5 verifies can flip near-tie
    # argmaxes off the serial trajectory — the same numerics chad already
    # accepts from wide-PLD's width-32 verifies on this model class.
    mtp_max_draft: int = 31
    # Head-step cost / serial-step cost, MEASURED: 4.3 ms chained head step
    # (1-layer head + lm_head readout) against the 66 ms serial trunk step
    # (M4 Pro, 4-bit 27B). This seeds DepthPolicy's cost ratios; getting it
    # wrong is a depth lock — at the old flat-model 0.35 the seed priced
    # depth 2 below depth 1 and the schedule never explored past 1, so no
    # observation could ever correct the table.
    mtp_h: float = 0.065
    # Measured per-depth round costs T(0)..T(max), any unit; None = the
    # width-flat scalar-h model. See mlx_mtp.DepthPolicy.
    mtp_costs: Optional[list] = None
    # Draft PROPOSAL mode at temp>0. True (default) samples the proposal from
    # the head's distribution (classic Leviathan/Chen): at the recommended
    # temp 1.0 the target is broad and proposal≈target overlap acceptance wins
    # — measured 55% acc / 19.8 tok/s vs 39% / 16.5 for argmax proposals on
    # the 27B. False drafts the head's argmax and accepts with prob p(d)
    # (point-mass proposal) — the better mode only near-greedy. Both preserve
    # the output distribution exactly.
    mtp_sample_drafts: bool = True
    cache_dir: Optional[str] = None # on-disk KV checkpoints; None disables
    kv_cache_max_bytes: int = 8 * 1024**3  # LRU-evict the on-disk KV cache above this; 0 disables

    # mlx model/tokenizer/cache are loaded dynamically (mlx_lm has no stubs); annotate
    # as Any so the gate doesn't chase attributes/calls on objects mypy can't see.
    model: Any = field(init=False, default=None)
    tok: Any = field(init=False, default=None)
    effective_ctx: int = field(init=False, default=32768)
    _cache: Any = field(init=False, default=None)
    _cached_ids: list = field(init=False, default_factory=list)
    # Token ids generated in the CURRENT turn, for the presence penalty. Reset at
    # the top of every generate() so the penalty is per-response (matching the
    # API semantics the Qwen recipe is written against) and cannot leak across
    # turns — a session-lifetime history would penalize the whole vocabulary into
    # noise after a few thousand tokens.
    _seen_ids: list = field(init=False, default_factory=list)
    _trimmable: bool = field(init=False, default=False)
    _pld_hybrid: bool = field(init=False, default=False)
    _model_path: str = field(init=False, default="")  # resolved weights dir
    _warm_prefix_ids: Any = field(init=False, default=None)
    kv_bytes_per_token: float = field(init=False, default=0.0)  # measured at load (036)
    # Model-shape facts read from config at load, for the adaptive prefill chunk.
    # MoE prefill amortizes routing with bigger chunks (+14% measured 512→2048 on
    # a 35B MoE) while a dense model is compute-flat; the unfused head_dim-256
    # attention's transient scales with heads*chunk*kv_len. The shipped model is
    # dense — the MoE arm is here for `--model`.
    _is_moe: bool = field(init=False, default=False)
    _mtp_head: Any = field(init=False, default=None)
    _mtp_draft_readout: Any = field(init=False, default=None)
    _n_attn_heads: int = field(init=False, default=16)
    _n_kv_heads: int = field(init=False, default=0)
    _head_dim: int = field(init=False, default=0)
    # One-deep cache quarantine stack: push_cache stashes the live
    # (cache, cached_ids, flags) here so a subagent can run in a fresh isolated cache;
    # pop_cache restores it bit-identically. Depth 1 only — subagents never nest.
    _cache_stack: list = field(init=False, default_factory=list)
    # Bounded rewind for the non-trimmable hybrid: ONE recurrent-state
    # snapshot per turn, taken at prefill-end (reference copy — the DeltaNet layers
    # reassign their state arrays each step, so old arrays stay immutably valid).
    # {"pos": resident tokens at snapshot, "recurrent": _snap_recurrent()}. Lets
    # _sync_to recover a divergence INSIDE the last turn (truncated generation,
    # a dropped stream, an identical-prompt retry) by restoring the recurrent state,
    # native-trimming the attention KV, and re-feeding the few agreed-on tokens —
    # instead of the full-transcript re-prefill divergence used to cost.
    _rewind_snap: Optional[dict] = field(init=False, default=None)

    def _read_config(self, repo):
        import json
        import os
        local = os.path.join(repo, "config.json")
        if os.path.isfile(local):  # local model directory
            with open(local) as f:
                return json.load(f)
        from huggingface_hub import hf_hub_download
        with open(hf_hub_download(repo, "config.json")) as f:
            return json.load(f)

    def _ctx_override(self, repo):
        """Return (model_config_override, effective_ctx).

        Uses the model's full native window by default. If max_context exceeds the
        native window, enable YaRN rope scaling to extend it (capped at the model's
        documented extended max), so long agentic-coding sessions fit real context.
        """
        cfg = self._read_config(repo)
        # VL checkpoints (e.g. qwen3_5) nest the text model's real window
        # under `text_config`; the top level omits max_position_embeddings, so a
        # naive read silently falls back to 32768 and triggers needless compaction
        # (= a full re-prefill on this non-trimmable cache) at ~8x too small a
        # window. Prefer the nested value when the top level lacks it.
        native = cfg.get("max_position_embeddings") \
            or cfg.get("text_config", {}).get("max_position_embeddings") \
            or 32768
        # documented extended ceiling (Qwen ships this as the tokenizer max)
        ceiling = getattr(self.tok, "model_max_length", native) if self.tok else native
        if not ceiling or ceiling > 10_000_000:
            ceiling = native
        want = self.max_context or native
        want = min(want, max(ceiling, native))
        if want > native:
            factor = round(want / native + 1e-6, 4)
            override = {
                "max_position_embeddings": want,
                "rope_scaling": {
                    "type": "yarn",
                    "rope_type": "yarn",
                    "factor": factor,
                    "original_max_position_embeddings": native,
                },
            }
            return override, want
        return None, want

    def load(self):
        if not _HAS_MLX:
            # We're on the in-process MLX path (Engine was constructed), but the mlx
            # imports failed. Surface the ORIGINAL dlopen/import error — otherwise
            # `load` is None and the next line dies with an opaque NoneType TypeError.
            raise RuntimeError(
                "MLX is unavailable, so the in-process engine cannot load a model. "
                "On Apple Silicon this usually means a broken mlx/mlx-metal install "
                "(e.g. a missing libmlx.dylib after a partial `uv sync`) — try "
                "`uv sync --reinstall-package mlx-metal`. Original import error: "
                f"{_MLX_IMPORT_ERROR!r}"
            ) from _MLX_IMPORT_ERROR
        t0 = time.time()
        # Resolve a cached repo id to its local snapshot dir once, then load from disk —
        # skips the per-launch hub revision check on both the weights and _read_config.
        path = _local_path(self.model_id)
        self._model_path = path
        # Load tokenizer first (cheap) so _ctx_override can read its documented max.
        self.model, self.tok = load(path)
        override, eff = self._ctx_override(path)
        self.effective_ctx = eff
        if override is not None:
            # reload main with YaRN extension applied
            self.model, self.tok = load(path, model_config=override)
        self._read_model_shape(path)
        # Decode fast-path (fused projections + compiled S=1 layer step) for the
        # hybrid MoE checkpoint; silent no-op on any other model or on failure.
        from . import mlx_fastpath
        _log_mlx_provenance()
        mlx_fastpath.install(self.model, model_path=path)
        # Fused quantized-KV decode attention: makes kv_bits=8 a speed win
        # instead of a loss. Patches mlx_lm's quantized SDPA branch
        # only; inert unless a QuantizedKVCache is actually in play.
        from . import mlx_qsdpa
        qsdpa_ok = mlx_qsdpa.install()
        # MTP head sidecar (Qwen3.8-class): pure speed feature, None on any miss.
        if not config.flag("CHAD_NO_MTP"):
            from . import mlx_mtp
            self._mtp_head = mlx_mtp.load_head(self.model, path)
            # 2-bit shortlist readout for the greedy draft chain — pure speed
            # (the chained drafts' full-vocab lm_head read is ~70% of the
            # head-step cost). Draft-side only, never load-bearing.
            self._mtp_draft_readout = None
            if (self._mtp_head is not None
                    and not config.flag("CHAD_NO_DRAFT_SHORTLIST")):
                self._mtp_draft_readout = mlx_mtp.build_draft_readout(
                    self.model, path)
            self._resolve_mtp_knobs()
        self._resolve_kv_bits(qsdpa_ok)
        self._warm_verify_widths()
        self._install_memory_clamp()
        self._reset_cache()
        self.kv_bytes_per_token = self._measure_kv_bytes_per_token()
        return time.time() - t0

    def _read_model_shape(self, path: str) -> None:
        """Capture the config facts the adaptive prefill chunk needs:
        attention head count (sizes the unfused-SDPA transient) and whether the
        model is a MoE (bigger chunks amortize expert routing) — plus the kv-head
        count and head_dim the kv-quantization auto-gate needs. Best-effort —
        a model whose config withholds a field falls back to a conservative
        default rather than failing the load."""
        try:
            cfg = self._read_config(path)
            tc = cfg.get("text_config", cfg)
            self._n_attn_heads = int(tc.get("num_attention_heads") or 16)
            self._n_kv_heads = int(tc.get("num_key_value_heads") or 0)
            self._head_dim = int(tc.get("head_dim") or 0)
            self._is_moe = (tc.get("num_experts") is not None
                            or "moe" in str(cfg.get("model_type", "")).lower()
                            or "moe" in str(tc.get("model_type", "")).lower())
        except Exception:  # noqa: BLE001 — shape probing must never break load
            pass

    def _resolve_kv_bits(self, qsdpa_ok: bool) -> None:
        """Turn the kv_bits request into a concrete mode before the first cache
        is built. None = auto: default ON (8-bit) exactly when the fused decode
        kernel covers this model's attention shape AND installed — measured a
        WIN over the fp16 cache on both time (60.2 vs 55.8 tok/s @32k on the
        35B) and RAM (kv bytes/token halved -> ~2x the governor's ctx_limit ->
        fewer ~100 s compaction re-prefills). 0/falsy forces off; an explicit
        width is honored as-is (uncovered shapes then decode on mlx_lm's slow
        unfused path — the user asked, warn but obey). Draft-model (speculative)
        setups stay fp16: that path was never validated against a quantized
        cache."""
        from . import mlx_qsdpa
        gqa = self._n_attn_heads // self._n_kv_heads if self._n_kv_heads else 0
        covered = qsdpa_ok and mlx_qsdpa.covers(self._head_dim, gqa)
        if self.kv_bits is None:
            self.kv_bits = 8 if covered else None
            if self.kv_bits:
                log.info("KV cache: quantized 8-bit group-64 by default (fused "
                         "decode kernel covers head_dim=%d gqa=%d); "
                         "CHAD_KV_BITS=0 restores fp16", self._head_dim, gqa)
        elif not self.kv_bits:
            self.kv_bits = None            # explicit 0 -> fp16 cache
        elif not covered:
            log.warning("KV cache: kv_bits=%s forced on a shape the fused "
                        "kernel does not cover (head_dim=%s gqa=%s) — decode "
                        "will use the slow unfused path", self.kv_bits,
                        getattr(self, "_head_dim", "?"), gqa or "?")

    def _resolve_mtp_knobs(self) -> None:
        """Fold the CHAD_MTP_* environment overrides into the draft schedule.

        Split out of load() so the clamping contract is testable without
        weights — it has already been wrong once. CHAD_MTP_MAX_DRAFT used to
        clamp to 8, a leftover from before the schedule had plateau candidates:
        the default cap is DepthPolicy.MAX_DEPTH, so any request of 8 or more
        silently delivered LESS than the default and took every deep candidate
        with it. The knob's job is to LOWER the cap: within the policy's own
        ceiling it must deliver exactly what was asked for, and past it, that
        ceiling.
        """
        from .mlx_mtp import DepthPolicy

        av = config.env_str("CHAD_MTP_ADAPTIVE")
        if av is not None and av != "":
            self.mtp_adaptive = av.strip().lower() not in ("0", "false")
        nd = config.env_int("CHAD_MTP_DRAFT", 0)
        if nd:
            self.mtp_num_draft = nd
            self.mtp_adaptive = False       # an explicit width is an order
        md = config.env_int("CHAD_MTP_MAX_DRAFT", 0)
        if md:
            self.mtp_max_draft = min(md, DepthPolicy.MAX_DEPTH)
        hv = config.env_float("CHAD_MTP_H", 0.0)
        if hv:
            self.mtp_h = hv

    def _warm_verify_widths(self) -> None:
        """Build the fused verify kernel's per-width variants at load instead of
        inside the first span that needs one.

        The kernel is templated on the verify width, so a width that has never
        run is a Metal compile on the critical path of a real step. Warming only
        the widths THIS configuration can DISPATCH is the point, and that is two
        conditions, not one: the schedule has to be able to pick the width (a
        fixed draft width produces exactly one; the adaptive one produces its
        candidate set), and the fused kernel has to be the thing that serves it
        (past `mlx_qsdpa._wide_s_max` the step takes the dequantize-and-fuse
        branch, which is one stock kernel at any width). Warming the union of
        everything would pay for variants the run can never reach. Opt out with
        CHAD_NO_KERNEL_WARM."""
        if not self.kv_bits or config.flag("CHAD_NO_KERNEL_WARM"):
            return          # fp16 cache: no fused kernel, so no variants exist
        widths: set[int] = set()
        if self._mtp_head is not None:
            if self.mtp_adaptive:
                # The adaptive schedule only dispatches the policy's candidate
                # widths (bounded on purpose — an unwarmed width is a Metal
                # compile inside a live round, and warming all of 2..32 would
                # pay for variants no round dispatches).
                from .mlx_mtp import DepthPolicy
                widths.update(d + 1 for d in DepthPolicy.CANDIDATES
                              if 0 < d <= self.mtp_max_draft)
            else:
                widths.add(self.mtp_num_draft + 1)
        if (config.flag("CHAD_USE_PLD") and self.prompt_lookup and self.pld_wide
                and self.pld_wide_draft >= self.pld_wide_min_draft):
            # _wide_verify feeds [pending] + draft, and the draft runs from
            # pld_wide_min_draft up to pld_wide_draft tokens.
            widths.update(range(self.pld_wide_min_draft + 1,
                                self.pld_wide_draft + 2))
        if not widths:
            return
        try:
            from mlx.utils import tree_flatten

            from . import mlx_qsdpa
            # Only widths the fused kernel actually dispatches are templated on
            # S; wider steps take the dequantize-and-fuse branch, which is one
            # stock kernel at any width and has nothing to warm. Warming them
            # anyway compiled variants no step could reach.
            if not self._n_kv_heads:
                return          # warm_widths would reject the shape anyway
            cap = mlx_qsdpa._wide_s_max(self._n_attn_heads // self._n_kv_heads)
            widths = {w for w in widths if w <= cap}
            if not widths:
                return
            # Scales/norms carry the model's compute dtype; quantized weights
            # are uint32, so take the first float parameter rather than guess.
            # tree_flatten yields (path, array) pairs; the stub types it loosely
            # enough that mypy reads the element as a str.
            dt = next((p.dtype for _, p in tree_flatten(  # type: ignore[misc]
                self.model.parameters())
                if p.dtype in (mx.float16, mx.bfloat16)), None)
            if dt is None:
                return
            t0 = time.time()
            done = mlx_qsdpa.warm_widths(widths, self._n_attn_heads,
                                         self._n_kv_heads, dt)
            if done:
                log.info("QSDPA warm-up: %d verify width(s) compiled in %.2fs "
                         "(%s)", done, time.time() - t0,
                         ",".join(str(w) for w in sorted(widths)))
        except Exception as e:  # noqa: BLE001 — never let a warm-up break load
            log.warning("QSDPA warm-up skipped (%s)", e)

    def _install_memory_clamp(self) -> None:
        """Give the Metal allocator explicit limits instead of the
        purely predictive ctx-limit formula. `set_wired_limit` keeps the resident
        working set wired up to Apple's recommendation (the mlx-lm `wired_limit()`
        pattern chad never adopted); `set_memory_limit` slightly below it makes the
        allocator back-pressure (block/relieve) on a transient spike instead of
        letting it ride into a jetsam SIGKILL — the documented 35B-on-24GB failure
        mode. Opt out with CHAD_NO_MEMORY_CLAMP."""
        if config.flag("CHAD_NO_MEMORY_CLAMP"):
            return
        try:
            budget = int(mx.device_info()["max_recommended_working_set_size"])
            mx.set_wired_limit(budget)
            mx.set_memory_limit(int(budget * 0.92))
            log.info("MEMORY clamp: wired=%.2f GB, limit=%.2f GB",
                     budget / 1e9, budget * 0.92 / 1e9)
        except Exception as e:  # noqa: BLE001 — clamping is defense-in-depth only
            log.warning("memory clamp unavailable: %s", e)

    def _measure_kv_bytes_per_token(self, probe: int = 8) -> float:
        """One-time: per-token bytes the growing attention KV cache costs, so the
        RAM-aware compaction trigger (cli.py) can size context from real
        per-token cost instead of a magic constant — model/quant-agnostic.

        Feeds a few tokens through a fresh cache and reads each KVCache layer's
        per-token stride straight from its array shape (`nbytes / seq_len`), which is
        exact regardless of MLX's step-padding (both numerator and denominator include
        the padding). The 30 SSM/DeltaNet ArraysCache layers hold a fixed recurrent
        state that does NOT grow with context, so they're excluded — only the 10
        attention layers scale. Cheap (~0.1 s, 8-token forward); leaves a clean reset
        cache behind. Returns 0.0 on any failure (cli then falls back to the old cap)."""
        try:
            self._reset_cache()
            self._prefill(list(range(100, 100 + probe)))
            mx.eval([c.state for c in self._cache])
            bpt = 0.0
            for c in self._cache:
                if isinstance(c, (cache_utils.KVCache, cache_utils.QuantizedKVCache)):
                    for entry in (getattr(c, "keys", None), getattr(c, "values", None)):
                        arrs = entry if isinstance(entry, (tuple, list)) else (entry,)
                        for arr in arrs:
                            if arr is not None and len(arr.shape) >= 3 and arr.shape[2]:
                                bpt += arr.nbytes / arr.shape[2]
            return bpt
        except Exception:  # noqa: BLE001 — instrumentation must never break load
            return 0.0
        finally:
            self._reset_cache()
            mx.clear_cache()

    # -- cache management -------------------------------------------------

    @property
    def resident_tokens(self) -> int:
        """Tokens currently resident in the live KV cache. Lets the governor's live
        ctx-limit recheck (cli._compute_ctx_limit) separate the model's fixed floor
        from the grown cache when reading mx.get_active_memory()."""
        return len(self._cached_ids)

    def reset(self):
        """Public alias for `_reset_cache` (BaseEngine seam). Consumers (TUI /
        REPL / cli governor) call this; the private name stays the engine's own internal
        spelling, used throughout the cache machinery below."""
        self._reset_cache()

    def _reset_cache(self):
        self._cache = cache_utils.make_prompt_cache(self.model)
        if self.kv_bits:
            # Quantized-from-start attention cache (follow-on): the
            # layer types never change mid-session, so trim/rewind stay valid
            # (QuantizedKVCache trims by offset; realloc rebinds but never
            # mutates rows below the write point). Prefill over it runs
            # dequant + fused fp16 sdpa, decode the fused quantized kernel —
            # both installed by mlx_qsdpa at load().
            self._cache = [
                cache_utils.QuantizedKVCache(group_size=64, bits=self.kv_bits)
                if type(c) is cache_utils.KVCache else c
                for c in self._cache
            ]
        self._cached_ids = []
        # Hybrid SSM/attention models (e.g. qwen3_5) keep recurrent state
        # that cannot be rewound, so their cache is not trimmable. KV-trim tricks
        # (prompt-lookup decoding, partial prefix reuse) only work when it is.
        self._set_cache_flags()

    def _set_cache_flags(self):
        """Classify the live cache: trimmable (attention-only) vs a qwen3_5-style
        hybrid we can still roll back specially. Called whenever self._cache is
        replaced (reset, warm-start load, compaction reload)."""
        # A replaced cache invalidates the turn-boundary rewind snapshot — its
        # position is meaningless against new contents. (pop_cache restores the
        # pushed snapshot explicitly, after its direct flag assignments.)
        self._rewind_snap = None
        self._trimmable = cache_utils.can_trim_prompt_cache(self._cache)
        # ...BUT a qwen3_5-style hybrid is recoverable a different way: its
        # Gated-DeltaNet layers (ArraysCache) reassign their state arrays each step
        # rather than mutating them in place, so we can snapshot/restore those layers
        # by *reference* (free) and roll a speculative draft back exactly — restore
        # recurrent + native-trim the attention KV + re-feed the accepted prefix.
        # (Bit-exactness of this rollback is regression-tested in
        # test_engine_pld_hybrid.py against a synthetic unquantized hybrid, with the
        # draft quality injected so accept, partial-accept and total-rejection are all
        # exercised; the trimmable PLD path is covered in test_engine.py tier 2.) That
        # un-gates PLD on a cache `can_trim_prompt_cache` calls non-trimmable.
        self._pld_hybrid = (
            not self._trimmable
            and bool(self._cache)
            and all(isinstance(c, (cache_utils.KVCache, cache_utils.ArraysCache,
                                   cache_utils.QuantizedKVCache))
                    for c in self._cache)
            and any(isinstance(c, (cache_utils.KVCache, cache_utils.QuantizedKVCache))
                    for c in self._cache)
        )

    # -- on-disk KV checkpoints -------------------------------------------
    # The hybrid SSM cache is NOT trimmable, so we can never partially reuse
    # a divergent prefix in RAM. But a STABLE prefix — the system prompt + tool
    # schemas, byte-identical every session (~3.2k tokens) — is pure dead-weight
    # prefill on every cold start and `/reset`. Persisting the cache avoids that by
    # persisting the KV state to disk keyed by the rendered prefix and reloading it
    # instead of re-prefilling. We do the same with mlx_lm's save/load_prompt_cache:
    # the recurrent SSM state serializes fine (a fixed ~51MB floor; cheap for one
    # warm-start file), and on a same-model load the state is bit-for-bit reusable.

    def _ckpt_path(self, ids: list, tag: str = "") -> str:
        h = hashlib.sha1()
        h.update(self.model_id.encode("utf-8", "ignore"))
        h.update(b"\x00")
        # the cache MODE is part of the identity: an fp16 checkpoint loaded into
        # a quantized session (or vice versa) would silently flip the attention
        # path for the whole session
        h.update(f"kv{self.kv_bits or 0}".encode())
        h.update(b"\x00")
        if tag:  # namespace distinct checkpoint kinds (warm-prefix vs push-spill)
            h.update(tag.encode("utf-8", "ignore"))
            h.update(b"\x00")
        h.update(np.asarray(ids, dtype=np.uint32).tobytes())
        # cache_dir is Optional on the dataclass but is always set when checkpointing
        # is enabled, which is the only path that reaches _ckpt_path.
        kind = _CKPT_PUSH if tag == _CKPT_PUSH else _CKPT_WARM
        return os.path.join(self.cache_dir, f"{kind}-{h.hexdigest()}.safetensors")  # type: ignore[arg-type]

    def warm_prefix(self, prefix_ids: list, should_stop=None):
        """Make a cold session start warm. If a disk checkpoint for exactly these
        prefix tokens exists, load it into the live cache (ZERO prefill); otherwise
        prefill the prefix once and persist it for next time. Only valid on a cold
        cache. Returns (status, n_tokens) where status is 'hit' | 'miss' | 'skip'."""
        if not self.cache_dir or not prefix_ids:
            return ("skip", 0)
        if self._cached_ids:               # cache already populated this session
            return ("skip", 0)
        path = self._ckpt_path(prefix_ids)
        if os.path.isfile(path):
            try:
                loaded = cache_utils.load_prompt_cache(path)
                # sanity: a loaded cache must have one entry per model layer
                if len(loaded) == self._n_model_layers():
                    self._cache = loaded
                    self._cached_ids = list(prefix_ids)
                    self._warm_prefix_ids = list(prefix_ids)
                    self._set_cache_flags()
                    return ("hit", len(prefix_ids))
            except Exception:
                pass                       # corrupt/incompatible -> recompute below
        # miss: prefill the prefix into a fresh cache, then persist it.
        self._reset_cache()
        fed = self._prefill(list(prefix_ids), should_stop)
        if fed < len(prefix_ids):          # interrupted -> don't persist a partial
            self._cached_ids = list(prefix_ids[:fed])
            return ("miss", fed)
        self._cached_ids = list(prefix_ids)
        self._warm_prefix_ids = list(prefix_ids)
        try:
            os.makedirs(self.cache_dir, exist_ok=True)
            cache_utils.save_prompt_cache(path, self._cache)
            self._enforce_kv_budget(path)
        except Exception:
            pass                           # disk full / read-only -> just skip persist
        return ("miss", len(prefix_ids))

    def _n_model_layers(self) -> int:
        return len(self.model.layers)

    # -- hybrid (qwen3_5) speculative rollback primitives ------------------
    # The recurrent (Gated-DeltaNet) layers can't be trimmed, but they reassign
    # their state each step, so a snapshot is just a reference copy (no array data
    # moved) and restore is instant. The attention layers trim natively. Together
    # these let PLD roll a rejected draft back exactly on a "non-trimmable" cache.

    @property
    def _seen(self) -> list:
        """Per-turn generated-token history backing the presence penalty.

        Lazily created rather than read straight off the dataclass field: the
        private `_generate_*` methods are independently callable entry points
        (the bit-exactness tests drive them directly, bypassing `generate()`),
        and Engines are routinely built with `object.__new__` in tests, which
        skips dataclass defaults entirely. A per-turn scratch list is exactly the
        kind of state that should not make those call paths crash — and this way
        adding the next piece of scratch state does not mean editing a dozen
        hand-built fixtures. `generate()` still clears it per turn."""
        try:
            return self._seen_ids
        except AttributeError:
            self._seen_ids = []
            return self._seen_ids

    def _snap_recurrent(self):
        return [list(c.cache) for c in self._cache
                if isinstance(c, cache_utils.ArraysCache)]

    def _restore_recurrent(self, snap):
        rec = [c for c in self._cache if isinstance(c, cache_utils.ArraysCache)]
        for c, arrs in zip(rec, snap):
            c.cache = list(arrs)

    def _trim_kv(self, n):
        kv = [c for c in self._cache
              if isinstance(c, (cache_utils.KVCache, cache_utils.QuantizedKVCache))]
        cache_utils.trim_prompt_cache(kv, n)

    def _take_rewind_snapshot(self, pos: int) -> None:
        """Capture the turn-boundary rewind point: the recurrent state as of `pos`
        resident tokens. Called at prefill-end, before decode, on the non-trimmable
        hybrid only (a trimmable cache rewinds natively). `pos` is passed explicitly
        because `_cached_ids` is only reconciled AFTER decode — at prefill-end it
        still holds the pre-turn prefix. The copy is by reference — DeltaNet layers
        REASSIGN their state arrays each step, so the snapshot pins one generation
        of state arrays (~tens of MB) and stays exact."""
        # Gate on the same cache composition the PLD-hybrid rollback validated
        # (_pld_hybrid: non-trimmable, draft-less, KVCache/QuantizedKVCache +
        # ArraysCache layers only): _rewind_to trims via _trim_kv — an exotic
        # cache layer would silently NOT be trimmed and corrupt the rewind.
        # Quantized layers are safe here: the cache is quantized
        # from the start (no mid-decode type conversion) and trims by offset
        # (rows below the trim point are never mutated in place).
        if not self._pld_hybrid or pos <= 0:
            return
        self._rewind_snap = {"pos": pos, "recurrent": self._snap_recurrent()}

    def _rewind_to(self, target_ids: list, upto: int) -> Optional[int]:
        """Bounded rewind on the non-trimmable hybrid: land the cache at exactly
        `target_ids[:upto]` using the turn-boundary snapshot, when the snapshot
        position is at or before `upto` (both sequences agree through `upto` by the
        caller's contract). Restore the recurrent state to the snapshot point,
        native-trim the attention KV back to it, then re-feed the agreed-on tokens
        so both stacks advance together — the same restore/trim/re-feed primitive
        the PLD-hybrid path proved bit-exact (test_pld_hybrid_equals_greedy), over
        a longer range. Returns `upto` on success, None when no usable snapshot
        (caller falls back to the full rebuild)."""
        snap = self._rewind_snap
        if not snap or snap["pos"] > upto or snap["pos"] > len(self._cached_ids):
            return None
        peel = len(self._cached_ids) - snap["pos"]
        self._restore_recurrent(snap["recurrent"])
        if peel:
            self._trim_kv(peel)
        self._cached_ids = self._cached_ids[: snap["pos"]]
        refeed = list(target_ids[snap["pos"] : upto])
        if refeed:
            self._prefill(refeed)
            self._cached_ids = list(target_ids[:upto])
        log.info("REWIND to %d resident tokens (snapshot@%d, re-fed %d) — "
                 "skipped a full re-prefill", upto, snap["pos"], len(refeed))
        return upto

    def _sync_to(self, target_ids: list) -> int:
        """Trim the cache down to the longest common prefix with target_ids.

        Returns the number of leading tokens already resident in the cache
        (i.e. how many tokens we get to skip prefilling).
        """
        common = 0
        for a, b in zip(self._cached_ids, target_ids):
            if a != b:
                break
            common += 1

        extra = len(self._cached_ids) - common
        if extra > 0:
            if cache_utils.can_trim_prompt_cache(self._cache):
                cache_utils.trim_prompt_cache(self._cache, extra)
                self._cached_ids = self._cached_ids[:common]
            elif self._rewind_to(target_ids, common) is not None:
                # Bounded rewind: the divergence sits inside the last
                # turn (truncated generation re-rendered, a dropped serve stream,
                # a retried prompt) — recovered above at the cost of re-feeding at
                # most one turn's tokens instead of the whole transcript.
                pass
            else:
                # Not trimmable (hybrid) -> can't partially rewind in RAM, so rebuild.
                # But the dominant divergence in agentic loops is *compaction*, which
                # only rewrites later turns and leaves the stable system+tools prefix
                # intact. If we persisted that prefix to disk (warm_prefix), reload it
                # (lossless, ~0.3s) instead of re-prefilling its ~3k tokens (~9s).
                self._reset_cache()
                common = self._reload_warm_prefix(target_ids)
        return common

    def _reload_warm_prefix(self, target_ids: list) -> int:
        """If a disk checkpoint of the warm system prefix exists and target_ids still
        begins with it, load it into the freshly-reset cache and return its length
        (tokens we skip re-prefilling). Returns 0 if unavailable/inapplicable."""
        wp = self._warm_prefix_ids
        if not (wp and self.cache_dir and len(target_ids) >= len(wp)
                and target_ids[: len(wp)] == wp):
            return 0
        path = self._ckpt_path(wp)
        if not os.path.isfile(path):
            return 0
        try:
            loaded = cache_utils.load_prompt_cache(path)
            if len(loaded) != self._n_model_layers():
                return 0
            self._cache = loaded
            self._cached_ids = list(wp)
            self._set_cache_flags()
            return len(wp)
        except Exception:
            self._reset_cache()            # corrupt/incompatible -> clean rebuild
            return 0

    # -- one-deep cache quarantine -----------------------------
    # A subagent explores in a SEPARATE small context so the main transcript's warm
    # cache isn't destroyed by the churn (grep/read spelunking). push_cache stashes the
    # live cache aside and hands the subagent a fresh empty one; pop_cache restores the
    # main cache bit-identically. The stash lives in RAM by default (measured cheap: a
    # 30k-token hybrid main cache ≈ 615 MB), but spills to a disk checkpoint when holding
    # it resident alongside the subagent's own growing cache would crowd the Metal budget.

    def _should_spill(self, ids: list) -> bool:
        """Whether to spill the pushed main cache to disk (vs holding it in RAM) while a
        subagent runs. The fast path keeps it in RAM; we only spill when the main cache
        is large enough that keeping it resident would leave too little headroom under
        Apple's recommended working set for the subagent to prefill its own context.
        Needs cache_dir (nowhere to spill), a measured per-token cost, and the live
        Metal memory APIs — returns False (hold in RAM) if any is unavailable."""
        if not self.cache_dir or not self.kv_bytes_per_token or not ids:
            return False
        main_bytes = len(ids) * self.kv_bytes_per_token
        try:
            budget = int(mx.device_info()["max_recommended_working_set_size"])
            active = mx.get_active_memory()
        except Exception:  # noqa: BLE001 — memory probe unavailable -> keep it in RAM
            return False
        # `active` already includes the resident model + the live main cache we're about
        # to push. The free band under the (safety-scaled) working set is what the
        # subagent gets to grow its own cache into. If that band is already tighter than
        # the main cache we'd be holding aside, reclaim the main cache to disk.
        free = budget * 0.90 - active
        return free < main_bytes

    def push_cache(self):
        """Depth-1 cache quarantine: stash the live (cache, cached_ids, flags) and start
        a fresh empty cache so a subagent can run isolated. pop_cache restores it. Raises
        if a cache is already pushed — subagents can't nest, and depth-1 keeps the
        lifecycle trivially auditable. Spills the stashed cache to disk when RAM is tight
        (see _should_spill), reclaiming it on pop."""
        if self._cache_stack:
            raise RuntimeError("push_cache: cache stack is depth-1 only (no nesting)")
        frame = {
            "cached_ids": self._cached_ids,
            "trimmable": self._trimmable,
            "pld_hybrid": self._pld_hybrid,
            "warm_prefix_ids": self._warm_prefix_ids,
            "cache": self._cache,
            "spill_path": None,
            # The rewind snapshot is reference-copied recurrent state belonging to
            # THIS cache; it survives the push in RAM either way (tiny next to the
            # cache itself) and is restored on pop so the parent's rewind window
            # isn't lost to a subagent round-trip.
            "rewind_snap": self._rewind_snap,
        }
        if self._should_spill(self._cached_ids):
            path = self._ckpt_path(self._cached_ids, tag=_CKPT_PUSH)
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_utils.save_prompt_cache(path, self._cache)
                frame["spill_path"] = path
                frame["cache"] = None  # drop the RAM reference; reclaimed on pop
                self._enforce_kv_budget(path)
            except Exception:  # noqa: BLE001 — disk full/read-only -> just hold in RAM
                pass
        self._cache_stack.append(frame)
        self._reset_cache()
        mx.clear_cache()  # release the freed buffers (esp. after a spill drop)

    def pop_cache(self):
        """Restore the cache stashed by push_cache, exactly. After this the main
        session's cache + _cached_ids are bit-identical to before the push, so its
        next turn re-syncs against a fully warm prefix (no re-prefill). Raises if
        nothing was pushed."""
        if not self._cache_stack:
            raise RuntimeError("pop_cache: no pushed cache to restore")
        frame = self._cache_stack.pop()
        spill = frame.get("spill_path")
        if spill:
            try:
                self._cache = cache_utils.load_prompt_cache(spill)
            except Exception:
                # The spilled checkpoint is missing/corrupt (spills only happen under
                # Metal memory pressure, so this is narrow). Degrade to a clean re-prefill
                # rather than propagate: pop_cache must never abort the parent — the
                # invariant is that a stuck sub-agent can't corrupt it. _reset_cache clears
                # _cached_ids, so the next turn re-syncs from empty and warms the prefix.
                log.warning("pop_cache: spilled checkpoint %s unreadable; "
                            "re-prefilling the parent from scratch", spill)
                self._reset_cache()
                try:
                    os.remove(spill)
                except OSError:
                    pass
                mx.clear_cache()
                return
            try:
                os.remove(spill)
            except OSError:
                pass
        else:
            self._cache = frame["cache"]
        self._cached_ids = frame["cached_ids"]
        self._trimmable = frame["trimmable"]
        self._pld_hybrid = frame["pld_hybrid"]
        self._warm_prefix_ids = frame["warm_prefix_ids"]
        self._rewind_snap = frame["rewind_snap"]
        mx.clear_cache()

    def _enforce_kv_budget(self, just_written: str) -> None:
        """LRU-evict the on-disk KV cache dir down to `kv_cache_max_bytes`, protecting
        the file just written and the current live warm-prefix file. Best-effort: never
        raises into the hot path."""
        if not self.cache_dir:
            return
        protect = {just_written}
        if self._warm_prefix_ids:
            protect.add(self._ckpt_path(self._warm_prefix_ids))
        enforce_cache_budget(self.cache_dir, self.kv_cache_max_bytes, protect)

    # -- generation -------------------------------------------------------

    def _adaptive_chunk(self, kv_len: int) -> int:
        """Prefill chunk size for the next chunk, given `kv_len` tokens already
        resident. Two measured facts drive this:

        - A MoE gains +14% prefill throughput from 512→2048 chunks (routing
          amortization, measured on a 35B MoE); a dense model is compute-bound flat
          across chunk sizes, so bigger chunks buy it nothing and only add transient
          memory. The shipped model is dense, so 512 is the shipped path and the
          MoE arm serves `--model`.
        - head_dim-256 attention falls off MLX's fused-SDPA
          path: each chunk materializes an fp32 (heads, chunk, kv_len) score tensor
          per attention layer, so the transient grows with chunk*kv_len — big chunks
          at long context are the prefill-OOM signature.

        So: start from a model-shaped base (MoE 2048, dense 512) and cap the chunk
        so the score-tensor transient stays inside half the free band under the
        Metal budget, floored at 256 so progress never stalls."""
        base = 2048 if getattr(self, "_is_moe", False) else 512
        try:
            budget = int(mx.device_info()["max_recommended_working_set_size"])
            free = budget * 0.90 - mx.get_active_memory()
            allow = max(free * 0.5, 256e6)
            per_tok = 4.0 * getattr(self, "_n_attn_heads", 16) * max(kv_len, 1)
            return max(256, min(base, int(allow / per_tok)))
        except Exception:  # noqa: BLE001 — memory probe unavailable -> static base
            return base

    def _snapshot_cache_refs(self):
        """Reference snapshot of every cache layer's attributes, taken before a
        prefill chunk so a caught Metal OOM (catchable on mlx>=0.32) can roll the
        cache back exactly. MLX arrays are immutable graph nodes — a failed step
        leaves the *python attributes* pointing at poisoned lazy arrays, but the
        pre-step arrays we hold references to are untouched, so restore is exact and
        free. Snapshot the raw `__dict__` (lists copied one level, since ArraysCache
        mutates its list elements in place) rather than the `.state` property — the
        KVCache getter RAISES on an empty cache (keys is None before token one),
        and a snapshot helper must work at every point in the cache lifecycle."""
        snap = []
        for c in self._cache:
            d = {k: (list(v) if isinstance(v, list) else v)
                 for k, v in c.__dict__.items()}
            snap.append((c, d))
        return snap

    def _restore_cache_refs(self, snap) -> None:
        for c, d in snap:
            for k, v in d.items():
                setattr(c, k, list(v) if isinstance(v, list) else v)

    def _prefill(self, ids: list, should_stop=None, chunk: Optional[int] = None,
                 on_progress=None) -> int:
        """Feed token ids through the model into the live cache in chunks, checking
        should_stop between chunks. Returns the count actually fed (< len(ids) when
        interrupted). This is what makes a large re-prefill abortable: MLX runs one
        chunk at a time and we get a chance to bail between them.

        Chunk sizing: an explicit `chunk` argument or CHAD_PREFILL_CHUNK wins;
        otherwise each chunk is sized adaptively from the model shape and the live
        free band (see `_adaptive_chunk`). A Metal OOM inside a chunk (catchable on
        mlx>=0.32) rolls the cache back to the pre-chunk snapshot, drops the scratch
        pool, and retries at half the size — turning the old process-killing spike
        into a slower-but-alive step.

        on_progress(done, total), if given, fires once per chunk with the tokens fed
        so far (monotonic, ending at `total` on a clean pass) so a caller can show a
        live progress %. It is **only** called when not None — the hot loop is
        byte-identical to a no-callback run, which matters because this feeds the
        non-trimmable cache."""
        mc = self._cache
        n = len(ids)
        if chunk is None:
            chunk = config.env_int("CHAD_PREFILL_CHUNK", 0) or None
        kv_base = len(getattr(self, "_cached_ids", None) or [])
        oom_cap: Optional[int] = None  # halved on each caught Metal OOM
        i = 0
        while i < n:
            if should_stop and should_stop():
                break
            step = chunk if chunk else self._adaptive_chunk(kv_base + i)
            if oom_cap:
                step = min(step, oom_cap)
            step = min(step, n - i)
            snap = self._snapshot_cache_refs()
            try:
                self.model(mx.array(ids[i : i + step], dtype=mx.uint32)[None], cache=mc)
                mx.eval([c.state for c in mc])
            except RuntimeError as e:
                if "memory" not in str(e).lower() or step <= 64:
                    raise
                self._restore_cache_refs(snap)
                mx.clear_cache()
                oom_cap = max(64, step // 2)
                log.warning("PREFILL Metal OOM at %d/%d (kv=%d): retrying with "
                            "chunk=%d", i, n, kv_base + i, oom_cap)
                continue
            i += step
            if on_progress:
                on_progress(i, n)
        return i

    def generate(
        self,
        prompt_ids: list,
        max_tokens: int = 2048,
        on_token: Optional[Callable[[str], None]] = None,
        stop_texts: Optional[list] = None,
        should_stop: Optional[Callable[[], bool]] = None,
        on_prefill: Optional[Callable[[int, int], None]] = None,
        on_prefill_progress: Optional[Callable[[int, int], None]] = None,
        stop_condition: Optional[Callable[[str, int], bool]] = None,
        think_ceiling: Optional[int] = None,
    ):
        """Generate a completion for the already-templated prompt_ids.

        Uses the persistent prefix cache: only prompt_ids[common:] is prefilled.
        Streams decoded text to on_token. Returns (text, GenStats). on_prefill(new,
        cached) fires once the prefill size is known, before the (potentially slow)
        prefill runs, so the caller can show honest status. on_prefill_progress(done,
        total) fires once per prefill chunk so the caller can show an advancing % during
        a big re-prefill; both prefill hooks are optional and pure instrumentation.

        stop_condition(text_so_far, n_generated) -> bool is an OPTIONAL caller predicate
        checked after each decoded token (composing with should_stop); when it returns
        True generation halts and stats.stop_condition_fired is set, so the caller can
        distinguish a deliberate early stop from an EOS/max_tokens finish. This is the
        soft think-cap hook: run_turn uses it to stop a ballooning <think>
        run and force-close the block. None => byte-identical to a plain generate call.

        think_ceiling (close-and-continue): a runaway <think> past this many
        tokens is force-closed by injecting THINK_CLOSE ids through the append-only cache,
        and decoding CONTINUES into the action in the SAME step (GenStats.salvaged set) —
        no new step, no re-derivation. None => off. This touches no cache trim/diff logic
        (the high-risk-zone invariant): the injection is a plain prefix-extension.
        """
        # Fairness / measurement knob (CHAD_NO_PREFIX_CACHE): drop the persistent prefix
        # cache before every turn so each step full-prefills from scratch. This forfeits
        # chad's core prefill win on purpose — the apples-to-apples TTFT baseline for
        # head-to-head harness benchmarks (cache OFF on both sides). Off in normal use;
        # matches the CHAD_NO_* opt-out family.
        if config.flag("CHAD_NO_PREFIX_CACHE"):
            self._reset_cache()

        # Presence-penalty history is per-response: clear it here, before any
        # decode path is chosen, so every loop below starts from an empty one.
        self._seen_ids = []

        # Wide prompt-lookup on the hybrid, now OPT-IN behind CHAD_USE_PLD.
        #
        # PLD drafts from CONTEXT RECURRENCE, so it can only accelerate text that
        # already appeared. Measured on real agentic traces, that is a minority of
        # what this agent generates: ~62-66% of generated tokens are <think>, and
        # reasoning prose replays at only ~2.3%. Whole-session contribution came out
        # at +2.2% of generated tokens — and was NEGATIVE before the n-gram gate was
        # raised from 6 to 16. MTP drafts from the model itself, so it is
        # traffic-independent and covers the think mass PLD structurally cannot.
        # They do not compose (one generate loop each), so this is a real choice;
        # MTP is the default and PLD is available for quote-heavy workloads
        # (file rewrites, large verbatim edits) where recurrence is the norm.
        if (config.flag("CHAD_USE_PLD") and self.prompt_lookup and self.pld_wide
                and self._pld_hybrid and not self._trimmable):
            return self._generate_pld_wide(prompt_ids, max_tokens, on_token,
                                           stop_texts, should_stop, on_prefill,
                                           on_prefill_progress, stop_condition,
                                           think_ceiling)

        # MTP self-speculative decoding — the DEFAULT speculative path (see the
        # PLD note above for why). Drafts with the checkpoint's own trained head.
        # Exact at ANY temp (rejection sampling preserves the sampling distribution)
        # and tolerates the quantized KV cache: rollback is offset-trim + rewrite,
        # the same mechanism _take_rewind_snapshot documents as safe on a
        # quantized-from-the-start cache.
        if (self._mtp_head is not None
                and (self._trimmable or self._pld_hybrid)):
            return self._generate_mtp(prompt_ids, max_tokens, on_token,
                                      stop_texts, should_stop, on_prefill,
                                      on_prefill_progress, stop_condition,
                                      think_ceiling)

        # Prompt-lookup decoding path: needs greedy decoding (exact),
        # an unquantized cache, and a trimmable cache (cheap rollback). A qwen3_5-style
        # hybrid CAN also roll back, via recurrent-snapshot + KV-trim + re-feed
        # (self._pld_hybrid), but the re-feed makes it ~2x slower on realistic agentic
        # generation, so it's behind enable_pld_hybrid (opt-in, off by default). PLD
        # shines on trimmable models doing edit-heavy work that re-quotes read files.
        if (self.prompt_lookup and self.temp == 0.0
                and not self.kv_bits
                and (self._trimmable or (self._pld_hybrid and self.enable_pld_hybrid))):
            return self._generate_prompt_lookup(prompt_ids, max_tokens, on_token,
                                                stop_texts, should_stop, on_prefill,
                                                on_prefill_progress, stop_condition,
                                                think_ceiling)

        common = self._sync_to(prompt_ids)
        suffix = prompt_ids[common:]
        stats = GenStats(
            prompt_tokens=len(suffix),
            cached_tokens=common,
        )
        if not suffix:
            # Nothing new to prefill (degenerate: the prompt is fully cached, e.g. an
            # identical prompt regenerated — never in normal append-only turns). We
            # still need one token to condition on, so we re-feed the last token. But
            # the live KV cache already holds it, so we must drop it from the cache in
            # lockstep, or the cache ends up one token LONGER than _cached_ids records
            # — an off-by-one that desyncs the next turn's trim math and silently
            # corrupts generation.
            if self._trimmable:
                # Mirror the PLD path: pop the last token off the live cache so the
                # re-feed below lands the cache back at exactly prompt_ids.
                cache_utils.trim_prompt_cache(self._cache, 1)
                suffix = prompt_ids[-1:]
                self._cached_ids = self._cached_ids[:-1]
                stats.prompt_tokens = 1
                stats.cached_tokens = common - 1
            elif self._rewind_to(prompt_ids, len(prompt_ids) - 1) is not None:
                # Non-trimmable hybrid with a turn-boundary snapshot:
                # land the cache at exactly prompt_ids[:-1] via the bounded rewind,
                # then re-feed the last token as the conditioning input — same shape
                # as the trimmable branch above, no full rebuild.
                common = len(prompt_ids) - 1
                suffix = prompt_ids[-1:]
                stats.prompt_tokens = 1
                stats.cached_tokens = common
            else:
                # Non-trimmable hybrid: we cannot pop a single token off
                # the recurrent state, so trimming is invalid and re-feeding would
                # duplicate the last token in the cache. This degenerate case is rare,
                # so take the safe path: rebuild from scratch and full re-prefill.
                # Slower but correct — a desynced non-trimmable cache silently corrupts
                # every later turn, which is far worse than one extra re-prefill here.
                self._reset_cache()
                common = 0
                suffix = list(prompt_ids)
                stats.prompt_tokens = len(suffix)
                stats.cached_tokens = 0
        if on_prefill:
            on_prefill(stats.prompt_tokens, stats.cached_tokens)

        kwargs = dict(
            max_tokens=max_tokens,
            # `seen` is passed by reference: the loop below appends each decoded
            # id to self._seen_ids as it materializes it, so the sampler reads a
            # live history without ever forcing a sync of its own draw.
            sampler=(_KeyedSampler(self.temp, min_p=self.min_p, top_p=self.top_p,
                                   top_k=self.top_k,
                                   presence_penalty=self.presence_penalty,
                                   seen=self._seen)
                     if self.temp > 0
                     else make_sampler(temp=self.temp, min_p=self.min_p, top_p=self.top_p)),
            prompt_cache=self._cache,
        )
        t0 = time.time()
        # Interruptible prefill: feed everything but the last token ourselves so
        # should_stop is honored between chunks. stream_generate then only has to
        # prefill the final token before decoding.
        gen_prompt = suffix
        resident = common
        if len(suffix) > 1:
            fed = self._prefill(suffix[:-1], should_stop, on_progress=on_prefill_progress)
            if fed < len(suffix) - 1:  # interrupted mid-prefill
                self._cached_ids = prompt_ids[: common + fed]
                stats.prefill_s = time.time() - t0
                return "", stats
            gen_prompt = suffix[-1:]
            resident = common + fed
        # Turn-boundary rewind point: everything resident right now is the
        # agreed-on prompt; whatever this turn appends past here is what a divergence
        # inside it will need to rewind. No-op on trimmable caches.
        self._take_rewind_snapshot(resident)

        text = ""
        gen_ids = []
        first_token_at = None
        oom_degraded = False

        # Decode loop, factored so close-and-continue can RE-ENTER it: after a
        # ceiling force-close, the same persistent cache is fed the </think> ids and
        # decoding resumes into the action. Re-entry is a plain append (stream_generate
        # prefills `seed_ids` onto self._cache, then decodes) — it touches no trim/diff/
        # snapshot logic, so the non-trimmable-cache invariant holds. Returns why it
        # stopped: 'ceiling' | 'stop' | 'condition' | 'eos'.
        def _decode(seed_ids, budget):
            nonlocal first_token_at, text
            kw = dict(kwargs, max_tokens=max(1, budget))
            for resp in stream_generate(self.model, self.tok, mx.array(seed_ids), **kw):
                if first_token_at is None:
                    first_token_at = time.time()
                    stats.prefill_s = first_token_at - t0
                text += resp.text
                gen_ids.append(resp.token)
                self._seen.append(resp.token)
                if on_token:
                    on_token(resp.text)
                if stop_texts and any(s in text for s in stop_texts):
                    return "stop"
                if should_stop and should_stop():
                    return "stop"
                if think_ceiling_hit(text, len(gen_ids), think_ceiling):
                    return "ceiling"
                if stop_condition is not None and stop_condition(text, len(gen_ids)):
                    stats.stop_condition_fired = True
                    return "condition"
            return "eos"

        try:
            if _decode(gen_prompt, max_tokens) == "ceiling":
                # Inject </think> and keep decoding the action IN THE SAME STEP. One salvage
                # per step: THINK_CLOSE is now in `text`, so think_ceiling_hit can't refire.
                close_ids = list(self.tok.encode(THINK_CLOSE, add_special_tokens=False))
                text += THINK_CLOSE
                gen_ids.extend(close_ids)
                stats.salvaged = True
                if len(gen_ids) < max_tokens and not (should_stop and should_stop()):
                    _decode(close_ids, max_tokens - len(gen_ids))
        except RuntimeError as e:
            # Metal OOM mid-decode (catchable on mlx>=0.32). stream_generate manages
            # the cache internally, so its state after a mid-forward failure is
            # unknowable — the only safe recovery is a clean rebuild. Keep the text
            # decoded so far (a partial turn beats a dead process); the next turn
            # warm-starts the system prefix from disk and re-prefills the body.
            if "memory" not in str(e).lower():
                raise
            log.warning("DECODE Metal OOM after %d tokens: dropping cache, keeping "
                        "partial text", len(gen_ids))
            oom_degraded = True
        stats.gen_s = time.time() - (first_token_at or t0)
        stats.generated_tokens = len(gen_ids)
        stats.gen_ids = list(gen_ids)

        if oom_degraded:
            self._reset_cache()
            mx.clear_cache()
            # Nothing is resident now, not even the prompt. Say so: a caller mirroring
            # our cache (chad serve's remote client) would otherwise assume prompt+gen
            # and estimate its next prefill against a cache that no longer exists.
            stats.cache_reset = True
            return text, stats
        # The cache now holds prefix + the tokens we generated.
        self._cached_ids = prompt_ids + gen_ids
        # Return MLX's freed-buffer pool to the OS. The live KV cache is held in
        # self._cache (active memory, untouched); this only releases the transient
        # prefill/decode scratch buffers that otherwise accumulate as cached memory
        # turn-over-turn and show up as a steadily climbing RSS.
        mx.clear_cache()
        return text, stats

    # -- prompt-lookup (n-gram) speculative decoding ----------------------

    def _eos_ids(self) -> set:
        ids = set()
        eid = getattr(self.tok, "eos_token_id", None)
        if eid is not None:
            ids.add(int(eid))
        for extra in getattr(self.tok, "eos_token_ids", None) or []:
            ids.add(int(extra))
        return ids

    def _generate_prompt_lookup(self, prompt_ids, max_tokens, on_token, stop_texts,
                                should_stop=None, on_prefill=None,
                                on_prefill_progress=None, stop_condition=None,
                                think_ceiling=None):
        """Single-model speculative decoding using n-gram prompt lookup as the
        drafter. Mirrors the prefix-cache contract of `generate`: only the new
        suffix is prefilled, drafts are verified in one batched forward, accepted
        runs are committed, and the cache is trimmed back on rejection.

        `think_ceiling` is accepted for signature parity but NOT wired here: this path
        is greedy (temp==0) only, whereas the think-spiral close-and-continue targets a
        temp>0 sampling pathology — that arm runs the llama backend, and the
        interactive MLX default falls to the main `generate` decode loop above."""
        common = self._sync_to(prompt_ids)
        suffix = prompt_ids[common:]
        stats = GenStats(prompt_tokens=len(suffix), cached_tokens=common)
        if on_prefill:
            on_prefill(stats.prompt_tokens, stats.cached_tokens)

        mc = self._cache
        eos = self._eos_ids()
        prefill_step = None  # None -> env override or adaptive sizing (see _prefill)
        # On a hybrid cache we roll drafts back by snapshot/restore instead of trim.
        hybrid = self._pld_hybrid and not self._trimmable

        t0 = time.time()

        def _prefill_head():
            """Prefill suffix[:-1] (all but the conditioning token) through the shared
            chunked `_prefill`, which honors should_stop between chunks. Returns True on
            a clean prefill, False if interrupted (caller returns an empty turn). One
            helper for both branches below — previously this loop was inlined twice and
            the hybrid copy silently dropped the should_stop check (un-abortable)."""
            fed = self._prefill(suffix[:-1], should_stop, chunk=prefill_step,
                                on_progress=on_prefill_progress)
            if fed < len(suffix) - 1:  # interrupted mid-prefill
                self._cached_ids = list(prompt_ids[: common + fed])
                stats.prefill_s = time.time() - t0
                return False
            # Turn-boundary rewind point — see generate(). `common` and
            # `suffix` are rebound by the degenerate branch before its call, so this
            # is correct from both call sites.
            self._take_rewind_snapshot(common + fed)
            return True

        # Prefill everything but the last token (we need a token to condition on).
        if not suffix:
            if hybrid:
                # Can't pop a single token off the recurrent state, so rebuild and
                # re-prefill all-but-last. Rare: only fires when an identical prompt
                # is regenerated (never in normal append-only agentic turns).
                self._reset_cache()
                mc = self._cache
                common, suffix = 0, list(prompt_ids)
                stats.prompt_tokens, stats.cached_tokens = len(suffix) - 1, 0
                if not _prefill_head():
                    return "", stats
                y_val = suffix[-1]
            else:
                # Degenerate: prompt fully cached. Re-feed the last token.
                cache_utils.trim_prompt_cache(mc, 1)
                y_val = prompt_ids[-1]
                stats.prompt_tokens, stats.cached_tokens = 1, common - 1
        else:
            if not _prefill_head():
                return "", stats
            y_val = suffix[-1]
        first_token_at = None

        context = list(prompt_ids)   # n-gram lookup window (prompt + generated)
        out_ids = []                 # tokens generated this turn (for text)
        # fed_ids mirrors exactly what's resident in the KV cache (cache-type
        # agnostic, since ArraysCache exposes no offset). The final pending token
        # (y_val) is intentionally NOT fed, so it isn't included here.
        fed_ids = list(prompt_ids[:-1])
        detok = self.tok.detokenizer
        detok.reset()
        # Adaptive draft width: wide verify forwards are nearly free on a big
        # bandwidth-bound model but cost real compute on a small one. Track a recent
        # acceptance EMA and draft wide only while it's paying off, so PLD never
        # meaningfully slows novel generation (low n-gram hit rate).
        nd_max = self.pld_num_draft
        acc_ema = 0.5
        # Hybrid backoff: on a recurrent cache every rejected draft costs an extra
        # re-feed forward, so drafting only pays inside genuinely high-accept (quote)
        # spans. Stay at nd=0 (zero-tax standard decode) when cold, probe cheaply to
        # detect entering a quote span, and latch wide once a draft is mostly accepted.
        hot, step_i = 0, 0
        PROBE_EVERY, PROBE_ND, HOT_STEPS = 10, 4, 12

        while len(out_ids) < max_tokens:
            if should_stop and should_stop():
                break
            if hybrid:
                if hot > 0:
                    nd = nd_max               # latched in a quote span: draft wide
                elif step_i % PROBE_EVERY == 0:
                    nd = PROBE_ND             # cheap periodic probe for a quote span
                else:
                    nd = 0                    # cold: standard decode, zero re-feed tax
            else:
                nd = nd_max if acc_ema >= 0.25 else 2
            step_i += 1
            draft = prompt_lookup_draft(context, nd, ngram_max=self.pld_ngram)
            k = len(draft)
            # Hybrid: snapshot the recurrent state before the verify forward so a
            # partial accept can roll back exactly (the snapshot is a free reference
            # copy; restore lands us at the pre-forward point to re-feed from).
            rec_snap = self._snap_recurrent() if hybrid else None
            y = mx.array([y_val] + draft, dtype=mx.uint32)
            logits = self.model(y[None], cache=mc)
            toks = mx.argmax(logits[0, -(k + 1):, :], axis=-1)
            mx.eval(toks)
            if first_token_at is None:
                first_token_at = time.time()
                stats.prefill_s = first_token_at - t0
            toks = [int(t) for t in toks.tolist()]
            stats.forwards += 1
            stats.draft_proposed += k

            # Accept the longest prefix of the draft the model agrees with.
            n_acc = 0
            while n_acc < k and toks[n_acc] == draft[n_acc]:
                n_acc += 1
            stats.draft_accepted += n_acc
            if k:
                acc_ema = 0.85 * acc_ema + 0.15 * (n_acc / k)
            if hybrid and k:
                # Latch wide after a mostly-accepted draft; decay back toward probing.
                hot = HOT_STEPS if n_acc >= max(1, int(0.6 * k)) else max(0, hot - 1)

            # Roll the cache back over the rejected drafts. What stays resident:
            # the previously-pending y_val plus the n_acc accepted draft tokens.
            # (Full accept needs no rollback either way.)
            if k - n_acc > 0:
                if hybrid:
                    # Restore the recurrent state to the pre-forward point, trim the
                    # attention KV all the way back too, then re-feed just the kept
                    # prefix so BOTH advance together to exactly y_val+draft[:n_acc].
                    self._restore_recurrent(rec_snap)
                    self._trim_kv(k + 1)
                    refeed = mx.array([y_val] + draft[:n_acc], dtype=mx.uint32)
                    self.model(refeed[None], cache=mc)
                    mx.eval([c.state for c in mc])
                else:
                    cache_utils.trim_prompt_cache(mc, k - n_acc)
            fed_ids.append(y_val)
            fed_ids.extend(draft[:n_acc])

            committed = draft[:n_acc] + [toks[n_acc]]  # accepted run + 1 bonus token
            stop = False
            for tid in committed:
                if tid in eos or len(out_ids) >= max_tokens:
                    stop = True
                    break
                out_ids.append(tid)
                context.append(tid)
                detok.add_token(tid)
                if on_token:
                    seg = detok.last_segment
                    if seg:
                        on_token(seg)
            # Soft think-cap: honor the caller's stop_condition on this path
            # too. Checked on the committed run's text/count; byte-identical when None.
            if not stop and stop_condition is not None \
                    and stop_condition(detok.text, len(out_ids)):
                stats.stop_condition_fired = True
                stop = True
            y_val = toks[n_acc]   # bonus token becomes next pending (unfed) token
            if stop:
                break
            if stop_texts and any(s in detok.text for s in stop_texts):
                break

        detok.finalize()
        if on_token and detok.last_segment:
            on_token(detok.last_segment)
        stats.gen_s = time.time() - (first_token_at or t0)
        stats.generated_tokens = len(out_ids)
        # What we GENERATED, which is not what we FED: `fed_ids` below deliberately
        # excludes the final pending token, so it is the wrong list to report to a
        # caller asking what this turn produced.
        stats.gen_ids = list(out_ids)

        # fed_ids is exactly what's resident in the KV cache, so it's the correct
        # prefix to diff against next turn.
        self._cached_ids = fed_ids
        mx.clear_cache()  # release transient scratch buffers back to the OS
        return detok.text, stats

    # -- wide prompt-lookup speculative decoding (hybrid) ------------------

    def _generate_pld_wide(self, prompt_ids, max_tokens, on_token, stop_texts,
                           should_stop=None, on_prefill=None,
                           on_prefill_progress=None, stop_condition=None,
                           think_ceiling=None):
        """Wide prompt-lookup decoding on the recurrent hybrid, exact at any
        temp.

        Per step: look for a long (>= pld_wide_min_ngram) recurrence of the
        current suffix in the context; if found, copy up to pld_wide_draft
        continuation tokens as the draft and verify them in ONE batched
        forward — the batched matmul path is flat from S=12 to S=32, so a wide
        verify costs the same as a narrow one. If not found, take a plain S=1
        decode step. Acceptance is
        point-mass rejection sampling (accept draft token d w.p. p(d), resample
        the residual on rejection), which preserves the output distribution
        exactly — greedy included.

        Rollback on partial rejection is the MTP capture-replay mechanism:
        the verify forward records each GDN layer's recurrence inputs, and a
        rejection replays the accepted prefix from the captured state — no
        re-feed forward, which is what made enable_pld_hybrid a loss."""
        from . import mlx_fastpath

        common = self._sync_to(prompt_ids)
        suffix = prompt_ids[common:]
        stats = GenStats(prompt_tokens=len(suffix), cached_tokens=common)
        if on_prefill:
            on_prefill(stats.prompt_tokens, stats.cached_tokens)

        mc = self._cache
        eos = self._eos_ids()
        lm = self.model.language_model
        embed = lm.model.embed_tokens

        def _logits(h):
            if lm.args.tie_word_embeddings:
                return embed.as_linear(h)
            return lm.lm_head(h)

        t0 = time.time()

        def _prefill_head():
            fed = self._prefill(suffix[:-1], should_stop,
                                on_progress=on_prefill_progress)
            if fed < len(suffix) - 1:  # interrupted mid-prefill
                self._cached_ids = list(prompt_ids[: common + fed])
                stats.prefill_s = time.time() - t0
                return False
            self._take_rewind_snapshot(common + fed)
            return True

        # Prefill everything but the last token — identical contract to the
        # MTP path, including the degenerate fully-cached-prompt branch.
        if not suffix:
            self._reset_cache()
            mc = self._cache
            common, suffix = 0, list(prompt_ids)
            stats.prompt_tokens, stats.cached_tokens = len(suffix) - 1, 0
            if not _prefill_head():
                return "", stats
            y_val = suffix[-1]
        else:
            if not _prefill_head():
                return "", stats
            y_val = suffix[-1]

        key = mx.random.key(int.from_bytes(os.urandom(4), "little"))
        first_token_at = None
        out_ids = []
        fed_ids = list(prompt_ids[:-1])
        detok = self.tok.detokenizer
        detok.reset()

        # Growing numpy context buffer for the per-step n-gram lookup
        # (always ends at the last MATERIALIZED token).
        ctx = np.empty(len(prompt_ids) + max_tokens + 64, dtype=np.int64)
        ctx[: len(prompt_ids)] = prompt_ids
        ctx_n = len(prompt_ids)

        def _push_ctx(tid):
            nonlocal ctx, ctx_n
            if ctx_n == len(ctx):
                ctx = np.concatenate([ctx, np.empty_like(ctx)])
            ctx[ctx_n] = tid
            ctx_n += 1

        def _commit(tid):
            """Emit one generated token. Returns True when generation must
            stop (eos / budget); the token is then NOT emitted."""
            if tid in eos or len(out_ids) >= max_tokens:
                return True
            out_ids.append(tid)
            self._seen.append(tid)
            _push_ctx(tid)
            detok.add_token(tid)
            if on_token:
                seg = detok.last_segment
                if seg:
                    on_token(seg)
            return False

        def _mark_first():
            nonlocal first_token_at
            if first_token_at is None:
                first_token_at = time.time()
                stats.prefill_s = first_token_at - t0

        def _plain_step(tok_arr):
            """One S=1 decode forward on a (possibly lazy) (1,) token array;
            returns the LAZY sampled next token. The forward takes the compiled
            S=1 layer fast path when installed (S==1, mask None)."""
            nonlocal key
            hid1 = lm.model(tok_arr[None], cache=mc)
            lg = _logits(hid1)[0, -1]
            if self.temp > 0:
                key, sub = mx.random.split(key)
                nt = mx.random.categorical(self._spec_scaled(lg), key=sub)
            else:
                nt = mx.argmax(lg)
            return nt.astype(mx.uint32).reshape(1)

        def _wide_verify(pv, draft):
            """Feed [pv]+draft in one batched forward; exact point-mass
            rejection sampling; capture-replay rollback on partial rejection.
            Returns (n_acc, next_tok)."""
            nonlocal key
            k = len(draft)
            rec_snap = self._snap_recurrent()
            coll = {"conv": [], "args": []}
            y = mx.array([pv] + draft, dtype=mx.uint32)
            try:
                mlx_fastpath.GDN_COLLECTOR = coll
                hid = lm.model(y[None], cache=mc)
                logits = _logits(hid)[0]
            finally:
                mlx_fastpath.GDN_COLLECTOR = None
            if self.temp > 0:
                scaled = self._spec_scaled(logits)
                # Two-phase accept: evaluate the k Bernoulli tests first
                # (tiny), find n_acc on the host, then build ONE residual
                # resample. The MTP path builds all k+1 candidate draws
                # lazily to save a sync, but at k=31 that is 31 wasted
                # 248k-vocab categoricals per step; a second tiny eval is
                # cheaper here.
                p = mx.softmax(scaled, axis=-1)
                dr = mx.array(draft, dtype=mx.uint32)
                p_sel = mx.take_along_axis(p[:k], dr[:, None], axis=-1)[:, 0]
                key, sub = mx.random.split(key)
                u = mx.random.uniform(shape=(k,), key=sub)
                ok = u <= p_sel          # point-mass q: accept w.p. p(d)
                mx.eval(ok)
                okl = ok.tolist()
                n_acc = 0
                while n_acc < k and okl[n_acc]:
                    n_acc += 1
                key, sub = mx.random.split(key)
                if n_acc < k:
                    # residual of a point mass: p with the draft token's
                    # mass removed (renormalized by categorical)
                    resid = mx.where(
                        mx.arange(p.shape[-1]) == dr[n_acc], 0.0, p[n_acc])
                    nt = mx.where(
                        resid.sum() > 0,
                        mx.random.categorical(mx.log(resid + 1e-30), key=sub),
                        mx.random.categorical(scaled[n_acc], key=sub))
                else:
                    nt = mx.random.categorical(scaled[k], key=sub)
                mx.eval(nt)
                next_tok = int(nt)
            else:
                toks = mx.argmax(logits, axis=-1)
                mx.eval(toks)
                toks = [int(t) for t in toks.tolist()]
                n_acc = 0
                while n_acc < k and toks[n_acc] == draft[n_acc]:
                    n_acc += 1
                next_tok = toks[n_acc]
            stats.forwards += 1
            stats.draft_proposed += k
            stats.draft_accepted += n_acc

            # Roll the main cache back over rejected drafts (identical
            # mechanism to _generate_mtp's hybrid branch).
            if k - n_acc > 0:
                arr = [c for c in mc
                       if isinstance(c, cache_utils.ArraysCache)]
                if len(coll["conv"]) == len(arr):
                    from mlx_lm.models.qwen3_5 import gated_delta_update
                    n_keep = n_acc + 1
                    for c, ci, (q_, k_, v_, a_, b_, A_log, dt_bias,
                                st0, use_k) in zip(
                            arr, coll["conv"], coll["args"]):
                        nk = ci.shape[1] - (k + 1)
                        c.cache[0] = mx.contiguous(
                            ci[:, n_keep : n_keep + nk])
                        _, st = gated_delta_update(
                            q_[:, :n_keep], k_[:, :n_keep],
                            v_[:, :n_keep], a_[:, :n_keep],
                            b_[:, :n_keep], A_log, dt_bias, st0, None,
                            use_kernel=use_k)
                        c.cache[1] = st
                    self._trim_kv(k - n_acc)
                else:
                    # Fallback (no fastpath GDN, so no checkpoints): restore
                    # the pre-forward snapshot and re-feed the accepted
                    # prefix — correct but one extra forward.
                    self._restore_recurrent(rec_snap)
                    self._trim_kv(k + 1)
                    refeed = mx.array([pv] + draft[:n_acc], dtype=mx.uint32)
                    self.model(refeed[None], cache=mc)
                    mx.eval([c.state for c in mc])
            fed_ids.append(pv)
            fed_ids.extend(draft[:n_acc])
            return n_acc, next_tok

        # Two-state decode loop.
        #   COMMITTED state (pend_val is not None): the pending token is a
        #     materialized int, already emitted (or a prompt token), NOT yet
        #     fed. This is where wide verifies launch from.
        #   LAZY state (pend is an mx array): the pending token is the lazy
        #     sample of an in-flight forward whose input is already fed —
        #     the same one-behind pipelining as stream_generate, so cold
        #     (no-span) decode pays no per-step sync bubble.
        # The stale-context lookup runs each iteration either way; a wide
        # verify only fires when the lookup's predicted NEXT token (d[0])
        # matches the actual pending token — an evidence gate that both
        # bounds the sync cost and filters false span entries.
        pend_val = y_val         # conditioning token: committed, unfed
        pend = None
        stop = False
        while not stop and len(out_ids) < max_tokens:
            if should_stop and should_stop():
                break
            # The context buffer ends at the last committed token. In the
            # COMMITTED state that token is the (unfed) pending token itself,
            # so d[0] is the first token a verify from it should draft. In the
            # LAZY state the pending token is still in flight, so d[0] is the
            # lookup's PREDICTION of it — the evidence gate below only enters
            # a span when that prediction matches the materialized sample.
            d, ng = prompt_lookup_draft_arr(
                ctx, ctx_n, self.pld_wide_draft + 1,
                self.pld_wide_ngram, self.pld_wide_min_ngram)
            candidate = (ng >= self.pld_wide_min_ngram
                         and len(d) >= self.pld_wide_min_draft + 1)

            draft = []
            pv = None
            if pend_val is None:
                # LAZY state. No span candidate: submit the next forward on
                # the lazy token FIRST, then materialize+commit one behind.
                # (Falls through to the shared stop_condition / think-ceiling
                # checks below — an early `continue` here silently disabled
                # stop_condition on cold decode, which in the agent loop means
                # every step runs to max_tokens.)
                if not candidate:
                    nxt = _plain_step(pend)
                    mx.async_eval(nxt)
                    pvc = int(pend)
                    _mark_first()
                    fed_ids.append(pvc)
                    stop = _commit(pvc)
                    if not stop and stop_texts \
                            and any(s in detok.text for s in stop_texts):
                        stop = True
                    pend = nxt
                else:
                    # Span candidate: materialize the pending token (the only
                    # forward in flight is its own producer). If the lookup
                    # predicted it, verify the continuation; else feed it and
                    # return to the pipeline.
                    pv = int(pend)
                    _mark_first()
                    stop = _commit(pv)
                    if stop:
                        break
                    pend = None
                    if pv == d[0]:
                        draft = d[1:]
            else:
                pv = pend_val
                pend_val = None
                if candidate:
                    draft = d[: self.pld_wide_draft]

            if stop:
                break
            if draft:
                n_acc, next_tok = _wide_verify(pv, draft)
                _mark_first()
                for tid in draft[:n_acc]:
                    stop = _commit(tid)
                    if stop:
                        break
                if not stop:
                    stop = _commit(next_tok)
                if not stop and stop_texts \
                        and any(s in detok.text for s in stop_texts):
                    stop = True
                pend_val = next_tok   # committed, unfed: ready to verify again
            elif pv is not None:
                # Committed pending token, no (confirmed) span: feed it and
                # return to the lazy pipeline.
                fed_ids.append(pv)
                pend = _plain_step(mx.array([pv], dtype=mx.uint32))
                mx.async_eval(pend)

            if not stop and stop_condition is not None \
                    and stop_condition(detok.text, len(out_ids)):
                stats.stop_condition_fired = True
                stop = True

            # -- think-ceiling close-and-continue ---------------------------
            # Same salvage as the MTP path (this loop also runs at temp>0);
            # a plain prefix-extension forward, no trim logic.
            if (not stop and not stats.salvaged
                    and think_ceiling_hit(detok.text, len(out_ids),
                                          think_ceiling)):
                if pend_val is None:
                    pv = int(pend)
                    _mark_first()
                    stop = _commit(pv)
                    if stop:
                        break
                    pend, pend_val = None, pv
                close_ids = list(self.tok.encode(THINK_CLOSE,
                                                 add_special_tokens=False))
                y2 = mx.array([pend_val] + close_ids, dtype=mx.uint32)
                hid2 = lm.model(y2[None], cache=mc)
                logits2 = _logits(hid2)[0, -1]
                if self.temp > 0:
                    key, sub = mx.random.split(key)
                    nt = mx.random.categorical(self._spec_scaled(logits2),
                                               key=sub)
                else:
                    nt = mx.argmax(logits2)
                mx.eval(nt)
                fed_ids.append(pend_val)
                fed_ids.extend(close_ids)
                for tid in close_ids:
                    out_ids.append(tid)
                    _push_ctx(tid)
                    detok.add_token(tid)
                    if on_token:
                        seg = detok.last_segment
                        if seg:
                            on_token(seg)
                stats.salvaged = True
                nt = int(nt)
                if _commit(nt):
                    break
                pend_val = nt

        detok.finalize()
        if on_token and detok.last_segment:
            on_token(detok.last_segment)
        stats.gen_s = time.time() - (first_token_at or t0)
        stats.generated_tokens = len(out_ids)
        stats.gen_ids = list(out_ids)
        self._cached_ids = fed_ids
        mx.clear_cache()
        return detok.text, stats

    # -- MTP self-speculative decoding -------------------------------------

    def _spec_scaled(self, logits, seen_rows=None):
        """Temp-scaled (+ presence/top_k/top_p/min_p filtered) logprobs over the
        last axis — exactly the scores _KeyedSampler hands to
        mx.random.categorical, so softmax of this IS the sampling distribution
        rejection sampling must preserve. Callers guarantee temp > 0.

        This function is the single definition of "the target distribution", which
        is why the presence penalty belongs here and not at the call sites: the
        speculative paths stay exact for free, because a penalized p is still just
        p. Both the draft chain and the verify use it, so drafts are scored under
        the same distribution they are accepted against.

        `seen_rows`, for a batched verify, is one already-generated-id list per
        row; row j must be penalized against the tokens preceding IT (history +
        the drafts before j), not against the pre-forward history alone, or the
        penalty would be a step behind inside every accepted run."""
        lp = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        # Penalty BEFORE the temperature divide, matching the API semantics the
        # model card's numbers are quoted in (logit - penalty, then scale). Applying
        # it after would divide the penalty by temp, making a "1.5" behave like a
        # 1.05 at temp 0.7 — the same knob quietly meaning something else per mode.
        if self.presence_penalty:
            if seen_rows is not None and lp.ndim > 1:
                lp = mx.stack([
                    apply_presence_penalty(lp[j], seen_rows[j],
                                           self.presence_penalty)
                    for j in range(lp.shape[0])])
            else:
                lp = apply_presence_penalty(lp, self._seen, self.presence_penalty)
        scaled = lp * (1.0 / self.temp)
        if self.top_k:
            scaled = apply_top_k(scaled, self.top_k)
        if 0 < self.top_p < 1.0:
            scaled = apply_top_p(scaled, self.top_p)
        if self.min_p != 0.0:
            scaled = apply_min_p(scaled, self.min_p)
        return scaled

    def _generate_mtp(self, prompt_ids, max_tokens, on_token, stop_texts,
                      should_stop=None, on_prefill=None,
                      on_prefill_progress=None, stop_condition=None,
                      think_ceiling=None):
        """Self-speculative decoding with the checkpoint's own MTP head.

        Per step: draft `mtp_num_draft` tokens by chaining the 1-layer head
        (each chained step feeds the head's own output hidden back — the
        serving-engine reference behavior), verify them in ONE batched main
        forward, and accept via exact speculative (rejection) sampling, so the
        output distribution is identical to plain decoding at the same
        temp/top_p/min_p — greedy included (temp 0 reduces to argmax-match).

        Cache contracts:
        - Main cache rollback on rejection is the PLD-hybrid primitive
          (recurrent snapshot/restore + KV offset-trim + re-feed), which also
          holds on the quantized-from-the-start cache: trim moves the offset
          and the re-feed rewrites rows above it before they are ever read.
        - The MTP head is attention-only, so its private KVCache trims
          natively. It is rebuilt fresh each turn and fed only REAL
          (main-model hidden, committed token) pairs after each verify —
          drafting's speculative entries are always trimmed. RoPE is relative
          under attention, so the turn-local (zero-based) drafter context is
          sound; it only costs the head cross-turn context, which affects
          acceptance rate, never correctness.
        """
        from mlx_lm.models.base import create_attention_mask

        from . import mlx_fastpath

        common = self._sync_to(prompt_ids)
        suffix = prompt_ids[common:]
        stats = GenStats(prompt_tokens=len(suffix), cached_tokens=common)
        if on_prefill:
            on_prefill(stats.prompt_tokens, stats.cached_tokens)

        mc = self._cache
        eos = self._eos_ids()
        head = self._mtp_head
        lm = self.model.language_model
        embed = lm.model.embed_tokens

        def _logits(h):
            if lm.args.tie_word_embeddings:
                return embed.as_linear(h)
            return lm.lm_head(h)

        hybrid = self._pld_hybrid and not self._trimmable
        t0 = time.time()

        def _prefill_head():
            fed = self._prefill(suffix[:-1], should_stop,
                                on_progress=on_prefill_progress)
            if fed < len(suffix) - 1:  # interrupted mid-prefill
                self._cached_ids = list(prompt_ids[: common + fed])
                stats.prefill_s = time.time() - t0
                return False
            self._take_rewind_snapshot(common + fed)
            return True

        # Prefill everything but the last token — identical contract to the
        # PLD path, including the degenerate fully-cached-prompt branches.
        if not suffix:
            if hybrid:
                self._reset_cache()
                mc = self._cache
                common, suffix = 0, list(prompt_ids)
                stats.prompt_tokens, stats.cached_tokens = len(suffix) - 1, 0
                if not _prefill_head():
                    return "", stats
                y_val = suffix[-1]
            else:
                cache_utils.trim_prompt_cache(mc, 1)
                y_val = prompt_ids[-1]
                stats.prompt_tokens, stats.cached_tokens = 1, common - 1
        else:
            if not _prefill_head():
                return "", stats
            y_val = suffix[-1]

        mtp_cache = head.make_cache()
        # Fresh per-turn policy state: acceptance statistics are a property of
        # the current prompt/content, not of the session.
        from .mlx_mtp import DepthPolicy
        policy = (DepthPolicy(self.mtp_max_draft, self.mtp_h, self.mtp_costs)
                  if self.mtp_adaptive else None)
        key = mx.random.key(int.from_bytes(os.urandom(4), "little"))
        first_token_at = None
        out_ids = []
        fed_ids = list(prompt_ids[:-1])
        detok = self.tok.detokenizer
        detok.reset()
        # Hidden (post-final-norm, (1,1,H)) of the LAST FED position — the
        # head's `previous hidden` input. None until the first verify forward,
        # so the first step drafts k=0 (a plain decode step that seeds it).
        h_last = None

        def _mtp_feed(tokens, h_prev, hid_seq):
            """Write real (hidden, next-token) pairs into the drafter cache:
            (h_prev, tokens[0]), (hid_seq[:,0], tokens[1]), ... Cheap: one
            1-layer forward of S=len(tokens)."""
            e = embed(mx.array([tokens], dtype=mx.uint32))
            s = len(tokens)
            hs = h_prev if s == 1 else mx.concatenate(
                [h_prev, hid_seq[:, : s - 1]], axis=1)
            m = create_attention_mask(e, mtp_cache) if s > 1 else None
            head(e, hs, cache=mtp_cache, mask=m)

        while len(out_ids) < max_tokens:
            if should_stop and should_stop():
                break
            round_t0 = time.time()
            if h_last is None:
                k = 0
            elif policy is not None:
                k = policy.depth()
            else:
                k = self.mtp_num_draft
            # Never draft past the token budget: the tail of the window is a
            # plain (or shallower) step, not a wasted deep round.
            k = min(k, max(0, max_tokens - len(out_ids) - 1))

            # -- draft: chain the head k times ------------------------------
            # The chain stays LAZY: each drafted token feeds the next step's
            # embedding as an mx array, and one eval at the end fetches all k
            # ids — one host sync instead of k.
            draft, q_rows, d_arrs = [], [], []
            if k:
                # Greedy proposals can take the shortlist readout (2-bit
                # lm_head shadow + exact top-32 rerank — mlx_mtp.DraftReadout)
                # instead of the full-vocab read. Sampled proposals need the
                # whole distribution, so they keep the full readout.
                sampled = self.temp > 0 and self.mtp_sample_drafts
                readout = None if sampled else \
                    getattr(self, "_mtp_draft_readout", None)
                e = embed(mx.array([[y_val]], dtype=mx.uint32))
                hin = h_last
                for _ in range(k):
                    hout = head(e, hin, cache=mtp_cache)
                    if sampled:
                        scaled = self._spec_scaled(_logits(hout)[0, -1])
                        key, sub = mx.random.split(key)
                        d = mx.random.categorical(scaled, key=sub)
                        q_rows.append(mx.softmax(scaled, axis=-1))
                    elif readout is not None:
                        d = readout.argmax_token(hout)
                    else:
                        # argmax proposal — also correct at temp>0 (point-mass
                        # q; see mtp_sample_drafts). argmax of raw logits ==
                        # argmax after temp/top_p/min_p, which never drop the max.
                        d = mx.argmax(_logits(hout)[0, -1])
                    d_arrs.append(d)
                    e = embed(d.reshape(1, 1))
                    hin = hout
                if self.presence_penalty and self.temp > 0:
                    # seen_rows (below) needs the draft ids on the host BEFORE
                    # the verify graph is built. Everyone else reads them after
                    # the accept eval, so the pp-off path (the thinking-mode
                    # recipe) defers this sync into that one — one host
                    # round-trip per step instead of two.
                    mx.eval(d_arrs)
                    draft = [int(x) for x in d_arrs]

            # -- verify: one batched main forward ---------------------------
            # On the hybrid, arm the GDN checkpoint collector: the verify
            # forward then steps the recurrence per position (batched matmuls
            # untouched) and records restorable states, so a partial rejection
            # below rolls back by reference instead of re-feeding — the
            # re-feed tax is what made speculation a wash on recurrent caches.
            rec_snap = self._snap_recurrent() if hybrid else None
            coll = {"conv": [], "args": []} if (hybrid and k) else None
            # Build the verify input from the (possibly still-lazy) draft
            # arrays — no host ids needed to construct the graph.
            y = (mx.concatenate([mx.array([y_val], dtype=mx.uint32)]
                                + [d.reshape(1).astype(mx.uint32)
                                   for d in d_arrs])
                 if k else mx.array([y_val], dtype=mx.uint32))
            try:
                mlx_fastpath.GDN_COLLECTOR = coll
                hid = lm.model(y[None], cache=mc)
                logits = _logits(hid)[0]
            finally:
                mlx_fastpath.GDN_COLLECTOR = None
            if self.temp > 0:
                # Row j of the verify predicts the token AFTER draft[:j], so its
                # presence history is the committed history plus exactly those
                # drafts — which are already materialized ints by here, so this
                # costs no extra sync. Getting this right is what keeps the
                # speculative output distribution identical to plain decoding:
                # rejection sampling preserves whatever p is, so p must be the
                # SAME p a plain step would have built at that position. (The
                # draft chain above is free to use the round-start history — it
                # only defines the proposal q, which the correction cancels.)
                seen_rows = ([self._seen + draft[:j] for j in range(k + 1)]
                             if (self.presence_penalty and k) else None)
                scaled = self._spec_scaled(logits, seen_rows=seen_rows)
                p = mx.softmax(scaled, axis=-1)
                n_acc = 0
                if k:
                    # Acceptance tests AND every possible next-token draw are
                    # built lazily and fetched in ONE eval: candidate j is the
                    # residual resample norm(max(p_j - q_j, 0)) — the
                    # Leviathan/Chen correction that keeps the joint output
                    # distribution exactly p — and candidate k is the bonus
                    # token; the host then just indexes with n_acc.
                    dr = y[1:]                 # the k draft ids, still lazy
                    p_sel = mx.take_along_axis(p[:k], dr[:, None], axis=-1)[:, 0]
                    key, sub = mx.random.split(key)
                    u = mx.random.uniform(shape=(k,), key=sub)
                    if self.mtp_sample_drafts:
                        q = mx.stack(q_rows)
                        q_sel = mx.take_along_axis(q, dr[:, None],
                                                   axis=-1)[:, 0]
                        ok = u * q_sel <= p_sel
                    else:
                        ok = u <= p_sel      # point-mass q: accept w.p. p(d)
                    cand = []
                    for j in range(k):
                        if self.mtp_sample_drafts:
                            resid = mx.maximum(p[j] - q_rows[j], 0.0)
                        else:
                            # residual of a point mass: p with the draft
                            # token's mass removed
                            resid = mx.where(
                                mx.arange(p.shape[-1]) == dr[j], 0.0, p[j])
                        key, sub = mx.random.split(key)
                        cand.append(mx.where(
                            resid.sum() > 0,
                            mx.random.categorical(mx.log(resid + 1e-30),
                                                  key=sub),
                            mx.random.categorical(scaled[j], key=sub)))
                    key, sub = mx.random.split(key)
                    cand.append(mx.random.categorical(scaled[k], key=sub))
                    mx.eval(ok, cand, d_arrs)   # the round's ONE host sync
                    if not draft:
                        draft = [int(x) for x in d_arrs]
                    okl = ok.tolist()
                    while n_acc < k and okl[n_acc]:
                        n_acc += 1
                    next_tok = int(cand[n_acc])
                else:
                    key, sub = mx.random.split(key)
                    nt = mx.random.categorical(scaled[0], key=sub)
                    mx.eval(nt)
                    next_tok = int(nt)
            else:
                toks = mx.argmax(logits, axis=-1)
                # Top-2 values feed the policy's pending-margin gate: a
                # near-tie next token is where the head is about to miss,
                # whatever its streak says. Same eval — no extra host sync.
                t2 = mx.topk(logits, 2, axis=-1) if policy is not None else None
                mx.eval(toks, d_arrs) if t2 is None else mx.eval(toks, d_arrs, t2)
                if k and not draft:
                    draft = [int(x) for x in d_arrs]
                toks = [int(t) for t in toks.tolist()]
                n_acc = 0
                while n_acc < k and toks[n_acc] == draft[n_acc]:
                    n_acc += 1
                next_tok = toks[n_acc]
                if t2 is not None:
                    row = t2[n_acc]
                    policy.margin = abs(float(row[0]) - float(row[1]))
            if first_token_at is None:
                first_token_at = time.time()
                stats.prefill_s = first_token_at - t0
            stats.forwards += 1
            stats.draft_proposed += k
            stats.draft_accepted += n_acc
            if policy is not None and k:
                # An accepted draft that IS a stop token ends the walk without
                # indicting the next position — the round ended because the
                # generation did, not because the head missed.
                policy.record(k, n_acc,
                              stopped_early=(n_acc > 0
                                             and draft[n_acc - 1] in eos))
            if policy is not None:
                # The measured wall of THIS round (draft + verify + the one
                # host sync) re-prices its depth in the policy's cost table —
                # the seed ladder only has to be right about shape, not units.
                policy.observe_cost(k, time.time() - round_t0)

            # -- roll the main cache back over rejected drafts --------------
            if k - n_acc > 0:
                if hybrid:
                    arr = [c for c in mc
                           if isinstance(c, cache_utils.ArraysCache)]
                    if coll is not None and len(coll["conv"]) == len(arr):
                        # Lazy replay: land every GDN layer at exactly
                        # `n_keep` fed tokens by re-running the recurrence
                        # over the accepted prefix from the captured
                        # pre-round state — the recurrence is causal, so the
                        # state after n_keep tokens is independent of the
                        # rejected tail. One tiny lazy kernel per GDN layer,
                        # paid only on rejection. The conv window after n
                        # tokens of a chunk is conv_input[:, n : n+nk];
                        # attention KV trims natively. No re-feed forward.
                        from mlx_lm.models.qwen3_5 import gated_delta_update
                        n_keep = n_acc + 1
                        for c, ci, (q_, k_, v_, a_, b_, A_log, dt_bias,
                                    st0, use_k) in zip(
                                arr, coll["conv"], coll["args"]):
                            nk = ci.shape[1] - (k + 1)
                            c.cache[0] = mx.contiguous(
                                ci[:, n_keep : n_keep + nk])
                            _, st = gated_delta_update(
                                q_[:, :n_keep], k_[:, :n_keep],
                                v_[:, :n_keep], a_[:, :n_keep],
                                b_[:, :n_keep], A_log, dt_bias, st0, None,
                                use_kernel=use_k)
                            c.cache[1] = st
                        self._trim_kv(k - n_acc)
                    else:
                        # Fallback (no fastpath GDN, so no checkpoints):
                        # restore the pre-forward snapshot and re-feed the
                        # accepted prefix — correct but one extra forward.
                        self._restore_recurrent(rec_snap)
                        self._trim_kv(k + 1)
                        refeed = mx.array([y_val] + draft[:n_acc],
                                          dtype=mx.uint32)
                        self.model(refeed[None], cache=mc)
                        mx.eval([c.state for c in mc])
                else:
                    cache_utils.trim_prompt_cache(mc, k - n_acc)

            # -- reconcile the drafter cache --------------------------------
            # Drop ALL k speculative entries, then feed the real pairs for the
            # tokens that actually landed ([y_val] + accepted drafts) using the
            # verify forward's hiddens — bit-identical to what a re-feed would
            # compute, per the PLD-hybrid rollback evidence.
            if k:
                cache_utils.trim_prompt_cache([mtp_cache], k)
            if h_last is not None:
                _mtp_feed([y_val] + draft[:n_acc], h_last, hid)
            h_last = hid[:, n_acc : n_acc + 1]

            fed_ids.append(y_val)
            fed_ids.extend(draft[:n_acc])

            # -- commit -----------------------------------------------------
            committed = draft[:n_acc] + [next_tok]
            stop = False
            for tid in committed:
                if tid in eos or len(out_ids) >= max_tokens:
                    stop = True
                    break
                out_ids.append(tid)
                self._seen.append(tid)
                detok.add_token(tid)
                if on_token:
                    seg = detok.last_segment
                    if seg:
                        on_token(seg)
            if not stop and stop_condition is not None \
                    and stop_condition(detok.text, len(out_ids)):
                stats.stop_condition_fired = True
                stop = True
            y_val = next_tok
            if stop:
                break
            if stop_texts and any(s in detok.text for s in stop_texts):
                break

            # -- think-ceiling close-and-continue ---------------------------
            # This path runs at temp>0 where think spirals live, so unlike PLD
            # it wires the salvage: feed the pending token + THINK_CLOSE ids in
            # one forward (a plain prefix-extension — no trim/diff logic), emit
            # them, catch the drafter up, and keep decoding the action.
            if (not stats.salvaged
                    and think_ceiling_hit(detok.text, len(out_ids), think_ceiling)):
                close_ids = list(self.tok.encode(THINK_CLOSE,
                                                 add_special_tokens=False))
                y2 = mx.array([y_val] + close_ids, dtype=mx.uint32)
                hid2 = lm.model(y2[None], cache=mc)
                logits2 = _logits(hid2)[0, -1]
                if self.temp > 0:
                    key, sub = mx.random.split(key)
                    nt = mx.random.categorical(self._spec_scaled(logits2),
                                               key=sub)
                else:
                    nt = mx.argmax(logits2)
                mx.eval(nt)
                if h_last is not None:
                    _mtp_feed([y_val] + close_ids, h_last, hid2)
                h_last = hid2[:, -1:]
                fed_ids.append(y_val)
                fed_ids.extend(close_ids)
                for tid in close_ids:
                    out_ids.append(tid)
                    detok.add_token(tid)
                    if on_token:
                        seg = detok.last_segment
                        if seg:
                            on_token(seg)
                stats.salvaged = True
                # The freshly sampled token must be EMITTED here as well as
                # made pending: the loop invariant is that the pending token is
                # already in out_ids (it always enters as a committed bonus
                # token), and the next iteration only emits what it commits.
                nt = int(nt)
                if nt in eos or len(out_ids) >= max_tokens:
                    break
                out_ids.append(nt)
                detok.add_token(nt)
                if on_token:
                    seg = detok.last_segment
                    if seg:
                        on_token(seg)
                y_val = nt

        detok.finalize()
        if on_token and detok.last_segment:
            on_token(detok.last_segment)
        stats.gen_s = time.time() - (first_token_at or t0)
        stats.generated_tokens = len(out_ids)
        stats.gen_ids = list(out_ids)
        self._cached_ids = fed_ids
        mx.clear_cache()
        return detok.text, stats

