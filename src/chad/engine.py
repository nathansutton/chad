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
   and helps trimmable models on quote-heavy work, but Ornith's hybrid SSM/attention
   cache is non-trimmable, so PLD falls back cleanly and does not speed up the shipped
   model. (`prompt_lookup_draft` + `_generate_prompt_lookup`.)
2. **Thinking budget** — `--no-think` drops Ornith's `<think>` overhead; the most
   effective real speedup for time-to-done, since decode is bandwidth-bound (~47 tok/s).

Earlier revisions used a main+draft *speculative* layout; chad now ships a single model
with no draft. PLD reuses the same accept/rollback machinery without a second model.
"""

import hashlib
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import mlx.core as mx
import numpy as np
from mlx_lm import load, stream_generate
from mlx_lm.models import cache as cache_utils
from mlx_lm.sample_utils import make_sampler

# GenStats moved to base_engine.py (plan 046) so a non-MLX backend can build one without
# importing mlx.core. Re-exported here so existing `from .engine import GenStats` keeps
# working (bench.py, tests) — the class is unchanged.
from . import config
from .base_engine import GenStats
from .diag import log


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


@dataclass
class Engine:
    model_id: str = "mlx-community/Qwen2.5-Coder-3B-Instruct-4bit"
    draft_id: Optional[str] = "mlx-community/Qwen2.5-Coder-0.5B-Instruct-4bit"
    num_draft_tokens: int = 3
    temp: float = 0.0
    kv_bits: Optional[int] = None   # set to 8 to quantize the KV cache (saves RAM)
    max_context: Optional[int] = None  # request a context window; YaRN-extends if > native
    prompt_lookup: bool = True      # n-gram prompt-lookup speculative decoding (no draft model)
    pld_num_draft: int = 10         # tokens to draft per forward via n-gram lookup
    pld_ngram: int = 3              # max suffix length to match for drafting
    # PLD on a hybrid (qwen3_5/Ornith) cache is correct (bit-exact) but OFF by default:
    # the eval suite measured it ~2x SLOWER on realistic agentic generation. A recurrent
    # model can't rewind to mid-forward, so every partial/total draft rejection costs an
    # extra re-feed forward; on novel-text-heavy work (low n-gram acceptance) that's ~2
    # forwards/token. It only wins on quote-heavy spans (re-emitting a just-read file).
    # Opt in for that workload; the default standard path is faster for general use.
    enable_pld_hybrid: bool = False
    cache_dir: Optional[str] = None # ds4-style on-disk KV checkpoints; None disables

    # mlx model/tokenizer/cache are loaded dynamically (mlx_lm has no stubs); annotate
    # as Any so the gate doesn't chase attributes/calls on objects mypy can't see.
    model: Any = field(init=False, default=None)
    draft: Any = field(init=False, default=None)
    tok: Any = field(init=False, default=None)
    effective_ctx: int = field(init=False, default=32768)
    _cache: Any = field(init=False, default=None)
    _cached_ids: list = field(init=False, default_factory=list)
    _trimmable: bool = field(init=False, default=False)
    _pld_hybrid: bool = field(init=False, default=False)
    _warm_prefix_ids: Any = field(init=False, default=None)
    kv_bytes_per_token: float = field(init=False, default=0.0)  # measured at load (036)
    # One-deep cache quarantine stack (plan 041): push_cache stashes the live
    # (cache, cached_ids, flags) here so a subagent can run in a fresh isolated cache;
    # pop_cache restores it bit-identically. Depth 1 only — subagents never nest.
    _cache_stack: list = field(init=False, default_factory=list)

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
        # VL checkpoints (e.g. Ornith/qwen3_5) nest the text model's real window
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
        t0 = time.time()
        # Load tokenizer first (cheap) so _ctx_override can read its documented max.
        self.model, self.tok = load(self.model_id)
        override, eff = self._ctx_override(self.model_id)
        self.effective_ctx = eff
        if override is not None:
            # reload main with YaRN extension applied
            self.model, self.tok = load(self.model_id, model_config=override)
        if self.draft_id:
            d_override, _ = self._ctx_override(self.draft_id)
            self.draft, _ = load(self.draft_id, model_config=d_override) if d_override \
                else load(self.draft_id)
        self._reset_cache()
        self.kv_bytes_per_token = self._measure_kv_bytes_per_token()
        return time.time() - t0

    def _measure_kv_bytes_per_token(self, probe: int = 8) -> float:
        """One-time: per-token bytes the growing attention KV cache costs, so the
        RAM-aware compaction trigger (plan 036, cli.py) can size context from real
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
                if isinstance(c, cache_utils.KVCache):
                    for arr in (getattr(c, "keys", None), getattr(c, "values", None)):
                        if arr is not None and len(arr.shape) >= 3 and arr.shape[2]:
                            bpt += arr.nbytes / arr.shape[2]
            return bpt
        except Exception:  # noqa: BLE001 — instrumentation must never break load
            return 0.0
        finally:
            self._reset_cache()
            mx.clear_cache()

    # -- cache management -------------------------------------------------

    def reset(self):
        """Public alias for `_reset_cache` (plan 046 BaseEngine seam). Consumers (TUI /
        REPL / cli governor) call this; the private name stays the engine's own internal
        spelling, used throughout the cache machinery below."""
        self._reset_cache()

    def _reset_cache(self):
        self._cache = cache_utils.make_prompt_cache(self.model)
        if self.draft is not None:
            self._cache += cache_utils.make_prompt_cache(self.draft)
        self._cached_ids = []
        # Hybrid SSM/attention models (e.g. Ornith/qwen3_5) keep recurrent state
        # that cannot be rewound, so their cache is not trimmable. KV-trim tricks
        # (prompt-lookup decoding, partial prefix reuse) only work when it is.
        self._set_cache_flags()

    def _set_cache_flags(self):
        """Classify the live cache: trimmable (attention-only) vs a qwen3_5-style
        hybrid we can still roll back specially. Called whenever self._cache is
        replaced (reset, warm-start load, compaction reload)."""
        self._trimmable = cache_utils.can_trim_prompt_cache(self._cache)
        # ...BUT a qwen3_5-style hybrid is recoverable a different way: its
        # Gated-DeltaNet layers (ArraysCache) reassign their state arrays each step
        # rather than mutating them in place, so we can snapshot/restore those layers
        # by *reference* (free) and roll a speculative draft back exactly — restore
        # recurrent + native-trim the attention KV + re-feed the accepted prefix.
        # (The trimmable PLD path's bit-exact equivalence is regression-tested in
        # test_engine.py tier 2; this hybrid recurrent-snapshot rollback has no
        # automated case yet — no hybrid model is loaded in the test env.) That
        # un-gates PLD on a cache `can_trim_prompt_cache` calls non-trimmable.
        self._pld_hybrid = (
            not self._trimmable
            and self.draft is None
            and bool(self._cache)
            and all(isinstance(c, (cache_utils.KVCache, cache_utils.ArraysCache))
                    for c in self._cache)
            and any(isinstance(c, cache_utils.KVCache) for c in self._cache)
        )

    # -- ds4-style on-disk KV checkpoints ---------------------------------
    # Ornith's hybrid SSM cache is NOT trimmable, so we can never partially reuse
    # a divergent prefix in RAM. But a STABLE prefix — the system prompt + tool
    # schemas, byte-identical every session (~3.2k tokens) — is pure dead-weight
    # prefill on every cold start and `/reset`. ds4 (antirez) avoids that by
    # persisting the KV state to disk keyed by the rendered prefix and reloading it
    # instead of re-prefilling. We do the same with mlx_lm's save/load_prompt_cache:
    # the recurrent SSM state serializes fine (a fixed ~51MB floor; cheap for one
    # warm-start file), and on a same-model load the state is bit-for-bit reusable.

    def _ckpt_path(self, ids: list, tag: str = "") -> str:
        h = hashlib.sha1()
        h.update(self.model_id.encode("utf-8", "ignore"))
        h.update(b"\x00")
        if tag:  # namespace distinct checkpoint kinds (warm-prefix vs push-spill)
            h.update(tag.encode("utf-8", "ignore"))
            h.update(b"\x00")
        h.update(np.asarray(ids, dtype=np.uint32).tobytes())
        # cache_dir is Optional on the dataclass but is always set when checkpointing
        # is enabled, which is the only path that reaches _ckpt_path.
        return os.path.join(self.cache_dir, h.hexdigest() + ".safetensors")  # type: ignore[arg-type]

    def warm_prefix(self, prefix_ids: list, should_stop=None):
        """Make a cold session start warm. If a disk checkpoint for exactly these
        prefix tokens exists, load it into the live cache (ZERO prefill); otherwise
        prefill the prefix once and persist it for next time. Only valid on a cold,
        single-model (no-draft) cache — Ornith's normal configuration. Returns
        (status, n_tokens) where status is 'hit' | 'miss' | 'skip'."""
        if not self.cache_dir or not prefix_ids or self.draft is not None:
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
        except Exception:
            pass                           # disk full / read-only -> just skip persist
        return ("miss", len(prefix_ids))

    def _n_model_layers(self) -> int:
        return len(self.model.layers)

    def _sub_caches(self):
        n = self._n_model_layers()
        return self._cache[:n], self._cache[n:]

    # -- hybrid (qwen3_5) speculative rollback primitives ------------------
    # The recurrent (Gated-DeltaNet) layers can't be trimmed, but they reassign
    # their state each step, so a snapshot is just a reference copy (no array data
    # moved) and restore is instant. The attention layers trim natively. Together
    # these let PLD roll a rejected draft back exactly on a "non-trimmable" cache.

    def _snap_recurrent(self):
        return [list(c.cache) for c in self._cache
                if isinstance(c, cache_utils.ArraysCache)]

    def _restore_recurrent(self, snap):
        rec = [c for c in self._cache if isinstance(c, cache_utils.ArraysCache)]
        for c, arrs in zip(rec, snap):
            c.cache = list(arrs)

    def _trim_kv(self, n):
        kv = [c for c in self._cache if isinstance(c, cache_utils.KVCache)]
        cache_utils.trim_prompt_cache(kv, n)

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
            model_cache, draft_cache = self._sub_caches()
            if cache_utils.can_trim_prompt_cache(model_cache):
                cache_utils.trim_prompt_cache(model_cache, extra)
                if draft_cache:
                    cache_utils.trim_prompt_cache(draft_cache, extra)
                self._cached_ids = self._cached_ids[:common]
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

    # -- one-deep cache quarantine (plan 041) -----------------------------
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
        }
        if self._should_spill(self._cached_ids):
            path = self._ckpt_path(self._cached_ids, tag="push")
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                cache_utils.save_prompt_cache(path, self._cache)
                frame["spill_path"] = path
                frame["cache"] = None  # drop the RAM reference; reclaimed on pop
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
        mx.clear_cache()

    # -- generation -------------------------------------------------------

    def _prefill(self, ids: list, should_stop=None, chunk: int = 256,
                 on_progress=None) -> int:
        """Feed token ids through the model into the live cache in chunks, checking
        should_stop between chunks. Returns the count actually fed (< len(ids) when
        interrupted). This is what makes a large re-prefill abortable: MLX runs one
        chunk at a time and we get a chance to bail between them.

        on_progress(done, total), if given, fires once per chunk with the tokens fed
        so far (monotonic, ending at `total` on a clean pass) so a caller can show a
        live progress %. It is **only** called when not None — the hot loop is
        byte-identical to a no-callback run, which matters because this feeds the
        non-trimmable cache."""
        mc = self._cache
        n = len(ids)
        i = 0
        while i < n:
            if should_stop and should_stop():
                break
            step = min(chunk, n - i)
            self.model(mx.array(ids[i : i + step], dtype=mx.uint32)[None], cache=mc)
            mx.eval([c.state for c in mc])
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
        soft think-cap hook (plan 039): run_turn uses it to stop a ballooning <think>
        run and force-close the block. None => byte-identical to a plain generate call.
        """
        # Fairness / measurement knob (CHAD_NO_PREFIX_CACHE): drop the persistent prefix
        # cache before every turn so each step full-prefills from scratch. This forfeits
        # chad's core prefill win on purpose — the apples-to-apples TTFT baseline for
        # head-to-head harness benchmarks (cache OFF on both sides). Off in normal use;
        # matches the CHAD_NO_* opt-out family.
        if config.flag("CHAD_NO_PREFIX_CACHE"):
            self._reset_cache()

        # Prompt-lookup decoding path: needs no draft model, greedy decoding (exact),
        # an unquantized cache, and a trimmable cache (cheap rollback). A qwen3_5-style
        # hybrid (Ornith) CAN also roll back, via recurrent-snapshot + KV-trim + re-feed
        # (self._pld_hybrid), but the re-feed makes it ~2x slower on realistic agentic
        # generation, so it's behind enable_pld_hybrid (opt-in, off by default). PLD
        # shines on trimmable models doing edit-heavy work that re-quotes read files.
        if (self.draft is None and self.prompt_lookup and self.temp == 0.0
                and not self.kv_bits
                and (self._trimmable or (self._pld_hybrid and self.enable_pld_hybrid))):
            return self._generate_prompt_lookup(prompt_ids, max_tokens, on_token,
                                                stop_texts, should_stop, on_prefill,
                                                on_prefill_progress, stop_condition)

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
            else:
                # Non-trimmable hybrid (e.g. Ornith): we cannot pop a single token off
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
            sampler=make_sampler(temp=self.temp),
            prompt_cache=self._cache,
        )
        if self.kv_bits:
            kwargs["kv_bits"] = self.kv_bits
        if self.draft is not None:
            kwargs["draft_model"] = self.draft
            kwargs["num_draft_tokens"] = self.num_draft_tokens

        t0 = time.time()
        # Interruptible prefill (single-model, unquantized cache): feed everything
        # but the last token ourselves so should_stop is honored between chunks.
        # stream_generate then only has to prefill the final token before decoding.
        gen_prompt = suffix
        if self.draft is None and not self.kv_bits and len(suffix) > 1:
            fed = self._prefill(suffix[:-1], should_stop, on_progress=on_prefill_progress)
            if fed < len(suffix) - 1:  # interrupted mid-prefill
                self._cached_ids = prompt_ids[: common + fed]
                stats.prefill_s = time.time() - t0
                return "", stats
            gen_prompt = suffix[-1:]

        text = ""
        gen_ids = []
        first_token_at = None
        for resp in stream_generate(self.model, self.tok, mx.array(gen_prompt), **kwargs):
            if first_token_at is None:
                first_token_at = time.time()
                stats.prefill_s = first_token_at - t0
            text += resp.text
            gen_ids.append(resp.token)
            if on_token:
                on_token(resp.text)
            if stop_texts and any(s in text for s in stop_texts):
                break
            if should_stop and should_stop():
                break
            if stop_condition is not None and stop_condition(text, len(gen_ids)):
                stats.stop_condition_fired = True
                break
        stats.gen_s = time.time() - (first_token_at or t0)
        stats.generated_tokens = len(gen_ids)

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
                                on_prefill_progress=None, stop_condition=None):
        """Single-model speculative decoding using n-gram prompt lookup as the
        drafter. Mirrors the prefix-cache contract of `generate`: only the new
        suffix is prefilled, drafts are verified in one batched forward, accepted
        runs are committed, and the cache is trimmed back on rejection."""
        common = self._sync_to(prompt_ids)
        suffix = prompt_ids[common:]
        stats = GenStats(prompt_tokens=len(suffix), cached_tokens=common)
        if on_prefill:
            on_prefill(stats.prompt_tokens, stats.cached_tokens)

        mc = self._cache  # model-only cache (no draft model on this path)
        eos = self._eos_ids()
        prefill_step = 512
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
            # Soft think-cap (plan 039): honor the caller's stop_condition on this path
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

        # fed_ids is exactly what's resident in the KV cache, so it's the correct
        # prefix to diff against next turn.
        self._cached_ids = fed_ids
        mx.clear_cache()  # release transient scratch buffers back to the OS
        return detok.text, stats
