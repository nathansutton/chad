"""OpenAI-compatible backend adapter (plan 046 SPIKE — behind `--backend openai`).

WHAT THIS IS, AND ISN'T
-----------------------
This is an **architecture spike**, not a production backend. It exists to answer one
research question — *how much of chad's result is the harness vs the in-process engine?* —
by letting the exact same agent loop (guardrails, validate/repair, symbolic tools, edit
cascade) drive **any** OpenAI-compatible `/v1/chat/completions` endpoint (mlx_lm.server,
llama.cpp, LM Studio, a cloud model). It is a `BaseEngine` (see base_engine.py), so it
drops into the `Agent` slot where `engine.Engine` normally sits. The default MLX path is
UNTOUCHED; you only get here with `chad --backend openai …`.

The recorded verdict (README, plan 046) is that the in-process engine is the moat: the
stateless HTTP boundary throws away everything that makes chad fast. This adapter makes
that cost *measurable* rather than merely asserted — and it necessarily reproduces the
loss. The degradations below are not bugs to fix; they are the finding.

HONEST DEGRADATIONS (loud, on purpose)
--------------------------------------
1. DETOKENIZE — the agent hands the engine rendered *token ids* (`prompt_ids`); a chat
   endpoint wants *messages*. We pick **decode-the-ids**: `tok.decode(prompt_ids)` and
   ship the result verbatim as a single user message. WHY this over re-rendering a
   message list: the ids ARE the seam the agent provides, and decoding reproduces chad's
   exact rendered prompt (system prompt, tool schemas, `<think>` scaffolding, every prior
   turn) byte-for-byte — so the ablation measures *the real harness context over a
   server*, not a lossily re-derived one. The re-render alternative isn't even available
   at this seam without changing `generate`'s contract (the adapter never receives the
   structured `messages`), and chasing it would grow model-specific templating past the
   plan's 50-line stop line. COST: the server re-applies ITS OWN chat template around our
   already-templated text (a double chat-template), and multi-role structure flattens to
   one user turn. Acceptable for a TTFT/pass-rate spike; not for production chat fidelity.
2. `cached_tokens` is UNKNOWABLE across the boundary — the server owns its cache and
   doesn't report reuse. We report `cached_tokens=0` and set `GenStats.approximate=True`
   so callers know the prefill/throughput numbers are estimates, not the exact accounting
   the MLX engine gives.
3. NO PREFILL PROGRESS — the server prefills opaquely; `on_prefill` fires once (best-effort
   size, 0 cached) and `on_prefill_progress` is never called. The status line can't show
   an advancing %; there is nothing to advance.
4. INTERRUPTS = DROPPING THE STREAM — there is no server-side cancel; honoring
   `should_stop` means we stop reading and close the HTTP response mid-generation. Tokens
   already produced server-side are simply discarded.

There is also no persistent prefix KV cache, no warm-prefix disk checkpoint, and no
cache quarantine — those require owning the cache object, which a stateless endpoint
forbids. `warm_prefix`/`push_cache`/`pop_cache` are therefore no-ops here (satisfying the
protocol without pretending to do work).
"""

import json
import time
import urllib.request
from typing import Any, Callable, Iterator, Optional

from .base_engine import GenStats


def build_chat_body(model_id: str, prompt_text: str, max_tokens: int, temp: float,
                    stream: bool = True) -> dict:
    """Build the `/v1/chat/completions` request body (pure — no network, unit-tested).

    The decoded chad prompt goes in as a SINGLE user message (see the module docstring's
    DETOKENIZE note). `stream_options.include_usage` asks the server to append a final
    usage chunk so we can report real token counts when it obliges."""
    return {
        "model": model_id,
        "messages": [{"role": "user", "content": prompt_text}],
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": stream,
        "stream_options": {"include_usage": True},
    }


def parse_sse_chunk(line: str) -> Optional[dict]:
    """Parse one Server-Sent-Events line from a streamed chat completion (pure).

    Returns the decoded JSON payload for a `data: {…}` line, or None for blanks,
    comments, and the terminal `data: [DONE]` sentinel. Raises nothing on a normal
    stream; malformed JSON propagates so the caller can decide (we treat it as end)."""
    line = line.strip()
    if not line or line.startswith(":"):
        return None
    if not line.startswith("data:"):
        return None
    payload = line[len("data:"):].strip()
    if payload == "[DONE]":
        return None
    return json.loads(payload)


def delta_text(chunk: dict) -> str:
    """Extract the incremental assistant text from a streamed chat-completion chunk
    (pure). Tolerates the usage-only final chunk (empty choices) by returning ''."""
    choices = chunk.get("choices") or []
    if not choices:
        return ""
    return choices[0].get("delta", {}).get("content") or ""


class OpenAIEngine:
    """A `BaseEngine` that talks to an OpenAI-compatible `/v1/chat/completions` endpoint.

    Constructed by cli.py under `--backend openai`. Loads only a *tokenizer* (no weights)
    so the agent can render/decode prompts; all generation happens over HTTP. See the
    module docstring for the degradations this backend deliberately embodies."""

    def __init__(self, model_id: str, base_url: str, api_key: str = "",
                 tokenizer_id: Optional[str] = None, effective_ctx: int = 32768,
                 temp: float = 0.0, timeout: float = 600.0):
        self.model_id = model_id
        # Normalize to the completions URL. Accept a base like "http://host:8080/v1" or
        # the full ".../v1/chat/completions"; be forgiving about the trailing slash.
        base = base_url.rstrip("/")
        self.url = base if base.endswith("/chat/completions") else base + "/chat/completions"
        self.api_key = api_key
        self.temp = temp
        self.timeout = timeout
        self.effective_ctx = effective_ctx
        # tokenizer_id lets you point at HF tokenizer files when the served model_id isn't
        # a resolvable repo (e.g. an mlx_lm.server alias); defaults to model_id.
        self._tokenizer_id = tokenizer_id or model_id
        # --- BaseEngine / drop-in data members ---
        self.tok: Any = None            # loaded in load()
        self.cache_dir: Optional[str] = None   # no on-disk KV across a stateless boundary
        self._cached_ids: list = []     # never populated; kept for seam compatibility
        self.draft = None               # no draft model (repl status line reads this)
        self.kv_bytes_per_token: float = 0.0   # cli's RAM-aware ctx sizing reads this

    # -- lifecycle --------------------------------------------------------

    def load(self) -> float:
        """Load ONLY the tokenizer (no weights) so `Agent._render` can template prompts and
        we can decode ids back to text. Returns elapsed seconds (mirrors Engine.load)."""
        t0 = time.time()
        from transformers import AutoTokenizer  # heavy import; defer to actual use
        self.tok = AutoTokenizer.from_pretrained(self._tokenizer_id)
        # Prefer the tokenizer's documented window if it's sane and we weren't told one.
        mml = getattr(self.tok, "model_max_length", None)
        if isinstance(mml, int) and 0 < mml < 10_000_000:
            self.effective_ctx = min(self.effective_ctx, mml) if self.effective_ctx else mml
        return time.time() - t0

    def reset(self) -> None:
        """No conversational state to drop (the server is stateless); clear the vestigial
        id list so the seam behaves identically to a fresh engine."""
        self._cached_ids = []

    _reset_cache = reset   # cli/bench call the private spelling; keep it working

    def warm_prefix(self, prefix_ids: list,
                    should_stop: Optional[Callable[[], bool]] = None) -> tuple[str, int]:
        """No disk KV warm-start across a stateless boundary — the server owns any cache
        and won't hand us its state. Always 'skip'."""
        return ("skip", 0)

    def push_cache(self) -> None:
        """No cache to quarantine (stateless); a sub-agent just shares the same endpoint."""
        return None

    def pop_cache(self) -> None:
        return None

    # -- HTTP (isolated so tests can stub it; no network in tests) ---------

    def _stream_completion(self, body: dict) -> Iterator[str]:
        """POST the chat-completion request and yield raw SSE lines. The ONLY network code
        in this file; unit tests monkeypatch it with a canned line generator so the pure
        request-build / SSE-parse / generate-accounting logic is exercised offline."""
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            for raw in resp:
                yield raw.decode("utf-8", "replace")

    # -- generation -------------------------------------------------------

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
    ) -> tuple[str, GenStats]:
        """Stream a completion from the endpoint. See the module docstring for the honest
        degradations baked into this path (detokenize, cached_tokens=0/approximate, no
        prefill progress, interrupt = drop the stream)."""
        # DETOKENIZE (degradation #1): decode chad's rendered ids back to text, keeping the
        # special role/think markers so the server receives chad's exact prompt verbatim.
        prompt_text = self.tok.decode(prompt_ids, skip_special_tokens=False)
        stats = GenStats(prompt_tokens=len(prompt_ids), cached_tokens=0, approximate=True)
        # on_prefill fires ONCE with a best-effort size; there is no cached prefix (0) and
        # no progress stream to advance (on_prefill_progress is intentionally never called).
        if on_prefill:
            on_prefill(stats.prompt_tokens, 0)

        body = build_chat_body(self.model_id, prompt_text, max_tokens, self.temp)
        text = ""
        n_out = 0
        t0 = time.time()
        first_at: Optional[float] = None
        usage: Optional[dict] = None
        stream = self._stream_completion(body)
        try:
            for raw in stream:
                # INTERRUPT (degradation #4): no server-side cancel — stop reading and let
                # the `finally` close the response, discarding whatever it kept generating.
                if should_stop and should_stop():
                    break
                for sub in raw.splitlines():   # a read may carry >1 SSE line
                    chunk = parse_sse_chunk(sub)
                    if chunk is None:
                        continue
                    if chunk.get("usage"):     # final include_usage chunk
                        usage = chunk["usage"]
                    seg = delta_text(chunk)
                    if not seg:
                        continue
                    if first_at is None:
                        first_at = time.time()
                        stats.prefill_s = first_at - t0
                    text += seg
                    n_out += 1
                    if on_token:
                        on_token(seg)
                    if stop_texts and any(s in text for s in stop_texts):
                        should_stop = lambda: True  # noqa: E731 — bail on next outer read
                        break
                    if stop_condition is not None and stop_condition(text, n_out):
                        stats.stop_condition_fired = True
                        should_stop = lambda: True  # noqa: E731
                        break
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()

        stats.gen_s = time.time() - (first_at or t0)
        # Prefer the server's real token counts when it sent a usage chunk; otherwise the
        # streamed-chunk count is our (approximate) generated-token estimate.
        if usage:
            stats.generated_tokens = int(usage.get("completion_tokens", n_out))
            stats.prompt_tokens = int(usage.get("prompt_tokens", stats.prompt_tokens))
        else:
            stats.generated_tokens = n_out
        return text, stats
