"""The engine seam: `GenStats` + the `BaseEngine` protocol.

`BaseEngine` is the *de facto* interface `Agent` already drives on `Engine` — extracted
verbatim into a `typing.Protocol` so a second backend can implement it without touching
the agent loop, and so mypy enforces the boundary. It **documents the seam that exists**;
it does not redesign it. `Engine` (the MLX crown jewel in `engine.py`) satisfies it
unchanged, and `CompletionEngine` (`completion_engine.py`) is the remote adapter that
proves a different backend can plug into the same slot.

Why a Protocol (not an ABC): zero runtime cost, structural (no base-class edit to
`Engine`), and it keeps `GenStats` — a pure dataclass with no MLX dependency — importable
by a backend that never loads mlx. `GenStats` lives here (not `engine.py`) precisely so
the remote adapter can build one without dragging in `mlx.core`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Protocol, runtime_checkable

# Close-and-continue salvage: the marker injected to force-close a <think> run
# that has blown past the ceiling, so decoding continues into the ACTION *in the same step*
# — vs the soft think-cap, which ends the step and lets the next one re-derive the whole
# reasoning (the measured anti-fix: force-close-then-new-step cost 3.8x total think). Both
# engines inject this same string: the MLX path tokenizes it and feeds the ids through the
# append-only cache; the llama path hands it back as an assistant-prefix continuation.
THINK_CLOSE = "\n</think>\n\n"


def think_ceiling_hit(text: str, n_tokens: int, ceiling: Optional[int],
                      think_closed: Optional[bool] = None) -> bool:
    """True when generation is still inside the auto-opened <think> block (no </think>
    emitted yet) and has run past `ceiling` tokens — the close-and-continue trigger
`ceiling` None/0 disables it (byte-identical to no ceiling). Pure so both
    engines share one definition and it is unit-testable without a model.

    Called once per decoded token by both engines, so the cheap token-count test runs
    FIRST: below the ceiling — which is almost always — this never touches `text` at all.
    `think_closed` lets a caller that already tracks the marker incrementally (see
    `TailWatch`) skip the scan entirely; None means "work it out from `text`", which is
    the original behavior and what the unit tests exercise."""
    if not ceiling or n_tokens < ceiling:
        return False
    if think_closed is None:
        think_closed = "</think>" in text
    return not think_closed


class TailWatch:
    """Has any of these needles appeared yet in an append-only text? Answered per
    segment, in constant time, without re-reading the text.

    The naive spelling — `any(n in text for n in needles)` after every decoded token —
    is quadratic in the generated length, because every token re-scans everything
    generated so far. Measured on the streaming client: 500 tokens costs 6ms, 2k costs
    41ms, 8k costs 242ms, 16k costs 777ms of pure CPU spent re-reading. That CPU is on
    the thread draining the token stream, so it is also what decides how fast the
    stream gets drained.

    A needle can only become newly complete if it straddles the segment just appended,
    so only `len(needle) - 1` characters of carry-over are ever relevant:

        …already scanned…│ tail │ new segment │
                          └──────┬──────────┘
                          the only window a new match can live in

    Latching: once found, always found (the text only grows), so `hit` never goes back
    to False and later feeds short-circuit."""

    def __init__(self, needles: Iterable[str]):
        self._needles = [n for n in (needles or []) if n]
        self._carry = max((len(n) for n in self._needles), default=1) - 1
        self._tail = ""
        self.hit = False

    def feed(self, seg: str) -> bool:
        """Absorb the next segment; return whether any needle has been seen yet."""
        if self.hit or not self._needles or not seg:
            return self.hit
        window = self._tail + seg
        if any(n in window for n in self._needles):
            self.hit = True
        self._tail = window[-self._carry:] if self._carry else ""
        return self.hit


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
    prompt_tokens: int = 0          # tokens actually PREFILLED this turn — NEW tokens only,
                                    # never including the cached prefix. This is the contract
                                    # on EVERY backend: the MLX engine counts the
                                    # suffix it forwards, and completion_engine adopts the
                                    # server's prompt_n (same meaning). The raw full-prompt
                                    # figure, when a server reports one, lands in
                                    # prompt_total_tokens.
    cached_tokens: int = 0          # tokens served from the prefix cache
    prompt_total_tokens: int = 0    # FULL prompt size (cached + prefilled) when a server
                                    # reports one, kept raw for forensics. 0 = not reported
                                    # (MLX/llama paths derive totals from prompt_tokens +
                                    # cached_tokens).
    generated_tokens: int = 0
    prefill_s: float = 0.0
    gen_s: float = 0.0
    forwards: int = 0               # model forward passes (PLD: < generated_tokens)
    draft_proposed: int = 0         # PLD: n-gram tokens proposed
    draft_accepted: int = 0         # PLD: n-gram tokens accepted
    stop_condition_fired: bool = False  # generation halted by the caller's stop_condition
                                    # (soft think-cap), not by EOS / max_tokens
    salvaged: bool = False          # close-and-continue fired: a runaway <think>
                                    # was force-closed mid-step and decoding CONTINUED into
                                    # the action in the same step. Distinct from
                                    # stop_condition_fired (which ENDS the step) — a salvaged
                                    # step returns a complete turn the caller parses normally.
    approximate: bool = False       # stats are best-effort, not exact: a remote backend
                                    # that can't report cached_tokens / per-forward
                                    # accounting sets this so callers know the
                                    # throughput/prefill numbers are estimates.
    gen_ids: list = field(default_factory=list)
                                    # the token ids generated this turn. The engine knows
                                    # them exactly; a caller must never re-derive them by
                                    # slicing `_cached_ids`, whose shape differs per decode
                                    # path (the prompt-lookup path stores what it FED the
                                    # cache, which omits the final pending token, and an
                                    # OOM empties it). `chad serve` puts these on the wire
                                    # so a remote client can mirror the server's cache.
    cache_reset: bool = False       # the prefix cache was DROPPED during this turn (Metal
                                    # OOM recovery), so nothing is resident afterwards —
                                    # not even the prompt. A caller mirroring cache state
                                    # must clear its mirror rather than assume prompt+gen,
                                    # which is the one thing `gen_ids` alone cannot say.

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
    """The interface `Agent` drives on an engine. Every member here is one
    `Agent` (or the TUI/REPL) already touches on `Engine` today — nothing aspirational.

    The two backends that satisfy it:
      - `engine.Engine`       — MLX, in-process, persistent prefix KV cache (the default).
      - `completion_engine.CompletionEngine` — a llama.cpp raw `/completion` client.

    Data members (`tok`, `model_id`, `effective_ctx`, `cache_dir`, `_cached_ids`) are read
    by the agent loop; the methods are the calls it makes each turn. The warm-prefix /
    cache-quarantine members are honored by the MLX engine and no-op'd by a remote
    backend (see `CompletionEngine`), which is exactly why they're part of the documented
    seam rather than hidden inside `Engine`.
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

        `think_ceiling`: when set, a <think> run that exceeds this many tokens is
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
