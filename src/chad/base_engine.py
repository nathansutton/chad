"""The engine seam: `GenStats` + the `BaseEngine` protocol (plan 046).

`BaseEngine` is the *de facto* interface `Agent` already drives on `Engine` — extracted
verbatim into a `typing.Protocol` so a second backend can implement it without touching
the agent loop, and so mypy enforces the boundary. It **documents the seam that exists**;
it does not redesign it. `Engine` (the MLX crown jewel in `engine.py`) satisfies it
unchanged, and `OpenAIEngine` (`openai_engine.py`) is the spike adapter that proves a
different backend can plug into the same slot.

Why a Protocol (not an ABC): zero runtime cost, structural (no base-class edit to
`Engine`), and it keeps `GenStats` — a pure dataclass with no MLX dependency — importable
by a backend that never loads mlx. `GenStats` lives here (not `engine.py`) precisely so
the OpenAI adapter can build one without dragging in `mlx.core`.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional, Protocol, runtime_checkable

# Close-and-continue salvage (plan 086): the marker injected to force-close a <think> run
# that has blown past the ceiling, so decoding continues into the ACTION *in the same step*
# — vs plan 039's think-cap, which ends the step and lets the next one re-derive the whole
# reasoning (the 084 anti-fix: force-close-then-new-step measured 3.8x total think). Both
# engines inject this same string: the MLX path tokenizes it and feeds the ids through the
# append-only cache; the openai path hands it back as an assistant-prefix continuation.
THINK_CLOSE = "\n</think>\n\n"


def think_ceiling_hit(text: str, n_tokens: int, ceiling: Optional[int]) -> bool:
    """True when generation is still inside the auto-opened <think> block (no </think>
    emitted yet) and has run past `ceiling` tokens — the close-and-continue trigger
    (plan 086). `ceiling` None/0 disables it (byte-identical to no ceiling). Pure so both
    engines share one definition and it is unit-testable without a model."""
    if not ceiling:
        return False
    return n_tokens >= ceiling and "</think>" not in text


class BackendError(RuntimeError):
    """A generation backend refused or failed a request.

    `transient` marks the failures worth re-issuing: 5xx from a remote server, or a
    mid-stream error chunk. llama.cpp returns 500 "The model produced output that does
    not match the expected Content-only format" when its chat parser can't reconcile a
    completion — a *sampling*-dependent fault, so at temp>0 a re-roll usually clears it.
    Retrying is only safe because chad rebuilds the prompt from `messages` each step and
    never appends a failed generation.
    """

    def __init__(self, message: str, transient: bool = False):
        super().__init__(message)
        self.transient = transient


@dataclass
class GenStats:
    prompt_tokens: int = 0          # tokens actually prefilled this turn
    cached_tokens: int = 0          # tokens served from the prefix cache
    generated_tokens: int = 0
    prefill_s: float = 0.0
    gen_s: float = 0.0
    forwards: int = 0               # model forward passes (PLD: < generated_tokens)
    draft_proposed: int = 0         # PLD: n-gram tokens proposed
    draft_accepted: int = 0         # PLD: n-gram tokens accepted
    stop_condition_fired: bool = False  # generation halted by the caller's stop_condition
                                    # (plan 039 soft think-cap), not by EOS / max_tokens
    salvaged: bool = False          # close-and-continue fired (plan 086): a runaway <think>
                                    # was force-closed mid-step and decoding CONTINUED into
                                    # the action in the same step. Distinct from
                                    # stop_condition_fired (which ENDS the step) — a salvaged
                                    # step returns a complete turn the caller parses normally.
    approximate: bool = False       # stats are best-effort, not exact (plan 046): an
                                    # OpenAI-style backend can't report cached_tokens /
                                    # per-forward accounting, so it sets this and callers
                                    # know the throughput/prefill numbers are estimates.

    @property
    def tok_per_s(self) -> float:
        return self.generated_tokens / self.gen_s if self.gen_s > 0 else 0.0

    @property
    def accept_rate(self) -> float:
        return self.draft_accepted / self.draft_proposed if self.draft_proposed else 0.0

    @property
    def tokens_per_forward(self) -> float:
        return self.generated_tokens / self.forwards if self.forwards else 0.0


@runtime_checkable
class BaseEngine(Protocol):
    """The interface `Agent` drives on an engine (plan 046). Every member here is one
    `Agent` (or the TUI/REPL) already touches on `Engine` today — nothing aspirational.

    The two backends that satisfy it:
      - `engine.Engine`       — MLX, in-process, persistent prefix KV cache (the default).
      - `openai_engine.OpenAIEngine` — an OpenAI-compatible `/v1/chat/completions` client.

    Data members (`tok`, `model_id`, `effective_ctx`, `cache_dir`, `_cached_ids`) are read
    by the agent loop; the methods are the calls it makes each turn. The warm-prefix /
    cache-quarantine members are honored by the MLX engine and no-op'd by a stateless
    backend (see `OpenAIEngine`), which is exactly why they're part of the documented seam
    rather than hidden inside `Engine`.
    """

    # --- data members the agent loop reads -------------------------------
    tok: Any                     # tokenizer: apply_chat_template / decode / detokenizer
    model_id: str
    effective_ctx: int           # usable context window (tokens)
    cache_dir: Optional[str]     # on-disk KV checkpoint dir; None disables warm-start
    _cached_ids: list            # tokens currently resident in the prefix cache

    # --- generation ------------------------------------------------------
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
    ) -> tuple[str, GenStats]:
        """Generate a completion for the already-templated `prompt_ids`, streaming decoded
        text to `on_token`. Returns `(text, GenStats)`. Signature mirrors `Engine.generate`
        exactly so `Engine` satisfies this protocol unchanged.

        `think_ceiling` (plan 086): when set, a <think> run that exceeds this many tokens is
        force-closed (THINK_CLOSE injected) and decoding CONTINUES into the action in the
        same step, setting `GenStats.salvaged`. None => off (byte-identical to before)."""
        ...

    # --- cache lifecycle -------------------------------------------------
    def reset(self) -> None:
        """Drop any conversational state / prefix cache (public alias of the MLX engine's
        `_reset_cache`). Called on `/reset` and by the governor's fresh-turn relaunch."""
        ...

    def warm_prefix(self, prefix_ids: list, should_stop: Optional[Callable[[], bool]] = None
                    ) -> tuple[str, int]:
        """Warm-start the stable system+tools prefix from disk (MLX) or no-op ('skip', 0)
        on a stateless backend. Returns (status, n_tokens)."""
        ...

    def push_cache(self) -> None:
        """Quarantine the live cache so a sub-agent can run isolated (MLX); no-op on a
        stateless backend, which has no cache to protect."""
        ...

    def pop_cache(self) -> None:
        """Restore the cache stashed by `push_cache` (MLX); no-op on a stateless backend."""
        ...
