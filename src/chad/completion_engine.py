"""llama.cpp raw `/completion` backend adapter (plan 047 — behind `--backend llama`).

WHAT THIS IS
------------
The second remote backend, one honesty rung above `openai_engine.py`. The OpenAI
adapter (plan 046) pays two deliberate costs at the `/v1/chat/completions` boundary:
it must DETOKENIZE chad's rendered ids into text that the server then re-templates
(a double chat-template), and the server's prefix cache is invisible, so every
telemetry number is an estimate (`GenStats.approximate=True`). llama.cpp's native
`/completion` endpoint removes both:

1. TOKEN-ID PROMPTS — `/completion` accepts `prompt` as an array of token ids, so
   the server evaluates chad's exact rendered prompt, byte-for-byte, with no second
   chat template and no re-tokenization drift. The one prerequisite: the server's
   GGUF must share the tokenizer vocab with the HF tokenizer loaded here (true when
   both were converted from the same base model — verified for Ornith).
2. REAL CACHE TELEMETRY — with `cache_prompt: true` the server reuses the longest
   common prefix of the previous evaluation, exactly like chad's own MLX prefix
   cache, and its final stream chunk reports real `timings` (`prompt_n` = tokens
   actually prefilled, `prompt_ms`, `predicted_n`, `predicted_ms`). When those
   arrive, `GenStats` carries exact server-side numbers and `approximate` stays
   False. `return_tokens: true` additionally gives us the generated ids, so
   `_cached_ids` mirrors the server's cache state (`prompt + generation`) the same
   way `engine.Engine` tracks its own — which makes the *pre*-generation
   `on_prefill(new, cached)` estimate accurate in chad's append-only turns.
3. REAL CANCEL — llama.cpp aborts the slot when the HTTP connection closes, so
   honoring `should_stop` by dropping the stream actually stops generation
   server-side (the OpenAI adapter can only stop *reading*).

WHAT IS STILL DEGRADED (inherent to any remote boundary)
--------------------------------------------------------
- No `on_prefill_progress` stream: the server prefills opaquely and reports timings
  only in the FINAL chunk, so the status line can't show an advancing %.
- No warm-prefix disk checkpoint (`warm_prefix` → ('skip', 0)): the cache lives in
  the server process, not on our disk.
- No cache quarantine (`push_cache`/`pop_cache` are no-ops): a single-slot server
  has ONE prefix cache, so a sub-agent's prompt evicts the main transcript's prefix
  and the return trip re-prefills it. Correctness is unaffected; latency pays.
"""

import json
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator, Optional

from .base_engine import GenStats
from .openai_engine import parse_sse_chunk  # same SSE framing; reuse the parser


def build_completion_body(prompt_ids: list, max_tokens: int, temp: float,
                          stream: bool = True) -> dict:
    """Build the raw `/completion` request body (pure — no network, unit-tested).

    `prompt` is the token-id array VERBATIM — the whole point of this backend (no
    detokenize, no double chat-template). `cache_prompt` asks the server to reuse
    the common prefix with the previous evaluation; `return_tokens` asks for the
    generated ids in each chunk so we can mirror the server's cache state."""
    return {
        "prompt": list(prompt_ids),
        "n_predict": max_tokens,
        "temperature": temp,
        "stream": stream,
        "cache_prompt": True,
        "return_tokens": True,
    }


def chunk_text(chunk: dict) -> str:
    """Extract the incremental text from a streamed `/completion` chunk (pure).
    Tolerates the final stop-chunk (empty content) by returning ''."""
    return chunk.get("content") or ""


def common_prefix_len(a: list, b: list) -> int:
    """Length of the shared prefix of two token-id lists (pure) — our client-side
    mirror of the reuse `cache_prompt` will get, used for the pre-generation
    `on_prefill` estimate and as the fallback when the server omits timings."""
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class CompletionEngine:
    """A `BaseEngine` that talks to a llama.cpp server's raw `/completion` endpoint.

    Constructed by cli.py under `--backend llama`. Loads only a *tokenizer* (no
    weights) so the agent can render prompts; all generation happens over HTTP.
    See the module docstring for what this recovers vs the OpenAI adapter and what
    stays degraded."""

    def __init__(self, model_id: str, base_url: str, api_key: str = "",
                 tokenizer_id: Optional[str] = None, effective_ctx: int = 0,
                 temp: float = 0.0, timeout: float = 600.0):
        self.model_id = model_id
        # Normalize to the /completion URL. Accept an origin like "http://host:8081"
        # (the documented spelling) or the full ".../completion"; forgive the slash.
        base = base_url.rstrip("/")
        self.url = base if base.endswith("/completion") else base + "/completion"
        self._origin = self.url[: -len("/completion")]
        self.api_key = api_key
        self.temp = temp
        self.timeout = timeout
        # 0 = auto: load() asks the server (/props n_ctx) so chad's window matches
        # the wall the server actually enforces; falls back to 32768 offline.
        self.effective_ctx = effective_ctx
        # tokenizer_id points at the HF repo whose tokenizer matches the served GGUF's
        # vocab (the served model_id is usually a GGUF repo with no tokenizer files).
        self._tokenizer_id = tokenizer_id or model_id
        # --- BaseEngine / drop-in data members ---
        self.tok: Any = None            # loaded in load()
        self.cache_dir: Optional[str] = None   # KV lives in the server, not on our disk
        self._cached_ids: list = []     # mirror of the server's cached prompt+generation
        self.draft = None               # no draft model (repl status line reads this)
        self.kv_bytes_per_token: float = 0.0   # cli's RAM-aware ctx sizing reads this

    # -- lifecycle --------------------------------------------------------

    def load(self) -> float:
        """Load ONLY the tokenizer (no weights) so `Agent._render` can template
        prompts, then size the context window from the server's own /props (the wall
        it will actually enforce). Returns elapsed seconds (mirrors Engine.load)."""
        t0 = time.time()
        from transformers import AutoTokenizer  # heavy import; defer to actual use
        self.tok = AutoTokenizer.from_pretrained(self._tokenizer_id)
        if not self.effective_ctx:
            props = self._fetch_props() or {}
            n_ctx = (props.get("default_generation_settings") or {}).get("n_ctx")
            self.effective_ctx = int(n_ctx) if n_ctx else 32768
        return time.time() - t0

    def reset(self) -> None:
        """Forget the mirrored cache state. The server keeps its slot cache, but the
        next prompt will simply prefix-match whatever is there (or not) — dropping
        our mirror just makes the next `on_prefill` estimate conservative."""
        self._cached_ids = []

    _reset_cache = reset   # cli/bench call the private spelling; keep it working

    def warm_prefix(self, prefix_ids: list,
                    should_stop: Optional[Callable[[], bool]] = None) -> tuple[str, int]:
        """No disk KV warm-start — the cache lives in the server process. 'skip'."""
        return ("skip", 0)

    def push_cache(self) -> None:
        """No quarantine on a single-slot server: a sub-agent's prompt evicts the main
        prefix and the return trip re-prefills it (latency, not correctness)."""
        return None

    def pop_cache(self) -> None:
        return None

    # -- HTTP (isolated so tests can stub it; no network in tests) ---------

    def _fetch_props(self) -> Optional[dict]:
        """GET the server's /props (read-only) for n_ctx discovery. Best-effort:
        any failure returns None and load() falls back to a safe default."""
        try:
            with urllib.request.urlopen(self._origin + "/props", timeout=10) as resp:
                return json.loads(resp.read().decode("utf-8", "replace"))
        except Exception:  # noqa: BLE001 — offline/older server → fallback, not a crash
            return None

    def _stream_completion(self, body: dict) -> Iterator[str]:
        """POST the /completion request and yield raw SSE lines. The ONLY generation
        network code in this file; unit tests monkeypatch it with a canned line
        generator. An HTTP error surfaces the server's message (e.g. llama.cpp's
        'exceeds the available context size') instead of a bare status code."""
        data = json.dumps(body).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                for raw in resp:
                    yield raw.decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:2000]
            raise RuntimeError(f"llama-server HTTP {e.code}: {detail}") from e

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
        """Stream a completion for the exact `prompt_ids` from the server. Stats come
        from the server's final-chunk `timings` when present (exact — `approximate`
        stays False); otherwise from client-side estimates (`approximate=True`)."""
        prompt_ids = list(prompt_ids)
        # Pre-generation estimate: `cache_prompt` reuses the longest common prefix of
        # the previous evaluation, which our `_cached_ids` mirrors — same numbers the
        # MLX engine reports at this point in the turn.
        est_cached = common_prefix_len(self._cached_ids, prompt_ids)
        stats = GenStats(prompt_tokens=len(prompt_ids) - est_cached,
                         cached_tokens=est_cached)
        if on_prefill:
            on_prefill(stats.prompt_tokens, stats.cached_tokens)
        # No progress stream across the boundary: on_prefill_progress never fires.

        body = build_completion_body(prompt_ids, max_tokens, self.temp)
        text = ""
        gen_ids: list = []
        n_chunks = 0
        t0 = time.time()
        first_at: Optional[float] = None
        timings: Optional[dict] = None
        stream = self._stream_completion(body)
        try:
            for raw in stream:
                # Dropping the stream closes the connection, which llama.cpp treats
                # as a cancel — generation genuinely stops server-side.
                if should_stop and should_stop():
                    break
                for sub in raw.splitlines():   # a read may carry >1 SSE line
                    chunk = parse_sse_chunk(sub)
                    if chunk is None:
                        continue
                    if chunk.get("error"):
                        raise RuntimeError(f"llama-server error: {chunk['error']}")
                    if chunk.get("timings"):   # final stop-chunk carries real telemetry
                        timings = chunk["timings"]
                    gen_ids.extend(chunk.get("tokens") or [])
                    seg = chunk_text(chunk)
                    if not seg:
                        continue
                    if first_at is None:
                        first_at = time.time()
                        stats.prefill_s = first_at - t0
                    text += seg
                    n_chunks += 1
                    if on_token:
                        on_token(seg)
                    if stop_texts and any(s in text for s in stop_texts):
                        should_stop = lambda: True  # noqa: E731 — bail on next outer read
                        break
                    if stop_condition is not None and stop_condition(text, n_chunks):
                        stats.stop_condition_fired = True
                        should_stop = lambda: True  # noqa: E731
                        break
        finally:
            close = getattr(stream, "close", None)
            if close:
                close()

        stats.gen_s = time.time() - (first_at or t0)
        stats.generated_tokens = len(gen_ids) or n_chunks
        if timings:
            # Exact server-side accounting: prompt_n = tokens actually prefilled this
            # call (the server subtracted its cache hit), so cached = the remainder.
            n_eval = int(timings.get("prompt_n", stats.prompt_tokens))
            stats.prompt_tokens = n_eval
            stats.cached_tokens = max(0, len(prompt_ids) - n_eval)
            if timings.get("prompt_ms") is not None:
                stats.prefill_s = float(timings["prompt_ms"]) / 1000.0
            if timings.get("predicted_n"):
                stats.generated_tokens = int(timings["predicted_n"])
            if timings.get("predicted_ms") is not None:
                stats.gen_s = float(timings["predicted_ms"]) / 1000.0
        else:
            # Older server / dropped stream: keep the client-side estimates and say so.
            stats.approximate = True
        # Mirror the server's slot cache (prompt + generation), exactly like the MLX
        # engine's `_cached_ids = prompt_ids + gen_ids` — this is what makes the next
        # turn's pre-generation estimate accurate in append-only transcripts.
        self._cached_ids = prompt_ids + gen_ids
        return text, stats
