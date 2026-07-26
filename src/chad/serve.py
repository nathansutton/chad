"""chad serve — the in-process MLX engine behind llama.cpp's `/completion` wire API.

WHAT THIS IS
------------
The mirror image of `completion_engine.py`. That module is the CLIENT: chad's agent
loop driving a llama.cpp server over raw `/completion` when chad runs somewhere MLX
cannot (a Linux container). This module is the SERVER for that same protocol, backed
by the local `engine.Engine` — so a chad running in a container on this machine (or
anywhere on the LAN) drives THIS Mac's MLX model with the exact bytes it would
otherwise have sent to llama.cpp. Nothing on either side has to learn a new dialect.

Why it exists: a container measuring chad against a remote GGUF measures a different
quantization, a different kernel and a different sampler than the model people
actually run locally. Pointed at `chad serve`, that same container measures *the*
model, through the real persistent prefix cache, with only the harness boundary left
as a difference.

THE PROTOCOL (exactly what `completion_engine.CompletionEngine` sends)
----------------------------------------------------------------------
- ``GET  /props``       → ``{"default_generation_settings": {"n_ctx": N}, ...}``
  Also carries a ``chad`` block advertising the extensions below. A stock llama.cpp
  server has no such block, which is precisely how the client feature-detects.
- ``POST /completion``  → SSE. Body is ``{"prompt": [token ids], "n_predict": …,
  "temperature": …, "stream": true, "cache_prompt": true, "return_tokens": true}``.
  Each generated token is one ``data: {"content": "…"}`` line; the final line carries
  the generated ids and real ``timings`` so the client's stats stay exact rather than
  estimated. Dropping the connection cancels generation, as llama.cpp does.
- ``GET  /health``      → liveness, plus whether a generation is in flight.

THE EXTENSIONS (only chad speaks these; a stock server simply won't advertise them)
-----------------------------------------------------------------------------------
Because both ends are ours, the two capabilities the remote boundary otherwise
forfeits come back:

- ``POST /cache/push`` + ``POST /cache/pop`` — cache quarantine. On a single-slot
  llama.cpp server a sub-agent's prompt evicts the main transcript's prefix and the
  return trip re-prefills it. Here the sub-agent's excursion is bracketed by a real
  ``Engine.push_cache``/``pop_cache``, so the main transcript's KV survives it.
- ``POST /warm`` — the on-disk KV warm-start of the stable system+tools prefix, which
  a remote client cannot do because the checkpoint lives on the *server's* disk.

Both are latency, never correctness: a client that doesn't speak them, or a call that
fails, degrades to the plain remote behavior.

ONE ENGINE THREAD (NOT JUST A LOCK)
-----------------------------------
Every call that touches the engine — load, generate, push/pop, warm — is submitted to a
single-worker executor and awaited, so all MLX work happens on ONE thread for the life of
the process. Two independent reasons, and the second is not optional:

1. `Engine` holds a single persistent prefix cache. A second concurrent generation would
   not merely queue, it would thrash the prefix the first one is built on.
2. **MLX streams are thread-local.** An HTTP server hands each connection to a fresh
   thread, and the engine's own GPU stream does not exist there — evaluating a cache on
   the wrong thread dies with `RuntimeError: There is no Stream(gpu, N) in current
   thread` partway through a prefill. A lock alone does not fix this: it serializes
   access but still runs the work on whichever thread the request arrived on.

Read-only endpoints (`/props`, `/health`) touch no MLX and never enter the executor, so a
monitor can poll while a long turn decodes.

…AND THE ENGINE THREAD NEVER TOUCHES A SOCKET
----------------------------------------------
The one thread that decodes must not also be the thread that writes the wire, or TCP
backpressure becomes MLX backpressure: a socket send blocks when the peer stops draining,
and a blocked send on the engine thread stalls decoding for every client, not just the
slow one. Worse, a *blocked* write never raises, so the hung-up-client cancel below can
never fire — a client that stops reading without closing (a suspended container, a slept
laptop) wedges the server until TCP keepalive notices, hours later, while `/health`
happily reports `ok`.

So generation pushes SSE lines onto a bounded queue and the request's own HTTP thread
drains it and does the blocking write:

    engine thread                    queue                    HTTP thread
    ─────────────                    ─────                    ───────────
    decode token ──► enqueue(line) ──► [ ][ ][ ] ──► get() ──► wfile.write()
         ▲                │                                        │
         │                │ Full after WRITE_STALL_S               │ raises on a
         │                ▼                                        │ dead/stalled peer
         └──── cancel ◄── gone.is_set() ◄──────────────────────────┘

Both failure signals converge on the SAME path the dead-socket cancel already used: a
`write` that raises. The drain thread turns a socket error into one; a peer that stops
reading long enough to fill the queue turns a `put` timeout into one. `Engine.generate`
stops on the next token either way, and the socket timeout on the handler bounds the
drain itself so no thread blocks forever.

WHAT IS STILL NOT THE LOCAL PRODUCT
-----------------------------------
Generated token ids ride in the final SSE line rather than per-chunk (the engine's
token callback yields decoded text, not ids). A *cancelled* request therefore lands no
ids in the client's cache mirror, which makes its next pre-generation prefill estimate
conservative — the server's own diff, and the stats it reports, stay exact.
"""

import json
import os
import queue
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Optional

# Advertised in /props under "chad". The client turns each name into a real call only
# when it appears here, so an older/newer server on either side degrades instead of
# erroring — and a stock llama.cpp server, which sends no "chad" block at all, is
# indistinguishable from a chad server with every extension switched off.
CAP_CACHE_QUARANTINE = "cache_quarantine"
CAP_WARM_PREFIX = "warm_prefix"
CAPABILITIES = [CAP_CACHE_QUARANTINE, CAP_WARM_PREFIX]

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8081
MAX_BODY_BYTES = 64 * 1024 * 1024   # a 128k-token id array is ~1 MB; this is slack

# Streaming backpressure budget (see "…AND THE ENGINE THREAD NEVER TOUCHES A SOCKET").
# The queue absorbs a client that reads in bursts; the timeout is how long a client may
# stop draining before we call it dead and stop burning GPU on it. Generous on both
# axes: a legitimate client that pauses to run a tool must not be cancelled, and 512
# lines is well under a megabyte.
STREAM_QUEUE_MAX = 512
WRITE_STALL_S = 60.0
# How long the stream may stay byte-silent before we send an SSE comment to hold the
# client's idle-read timeout off. A long cold prefill emits no tokens for minutes, and
# clients time out on bytes, not on our progress. Comfortably under the tightest default
# a real client ships with (chad's own is 600s; Node's undici is 300s).
KEEPALIVE_S = 10.0

# Admission wall (see ServerState.safe_ctx). The TTL keeps a burst of requests from each
# paying for a ~3 ms memory probe; pressure does not move meaningfully faster than this.
# The floor is the point past which tightening stops being a safety measure and becomes
# an outage — under real pressure a short turn should still be servable.
SAFE_CTX_TTL_S = 1.0
SAFE_CTX_FLOOR = 4096
# Bounds every socket operation on a connection, so the drain thread cannot block
# forever on a peer that has stopped reading. Long generations are unaffected: while
# there is nothing to send the drain thread waits on the QUEUE, not the socket.
SOCKET_TIMEOUT_S = 300.0

# A failed hand-off IS the cancel signal, whichever half failed: the socket rejected the
# bytes (peer gone) or the queue stayed full past WRITE_STALL_S (peer not reading).
# `queue.Full` is not an OSError, so it has to be named explicitly.
WRITE_FAILED = (BrokenPipeError, ConnectionResetError, OSError, queue.Full)


# --- pure helpers (no engine, no socket — unit-tested offline) --------------

def props_payload(n_ctx: int, model_id: str = "",
                  capabilities: Optional[list] = None) -> dict:
    """The `/props` body. `default_generation_settings.n_ctx` is the field the client
    reads to size its window to the wall this server actually enforces; the `chad`
    block is the extension handshake (absent on stock llama.cpp)."""
    return {
        "default_generation_settings": {"n_ctx": int(n_ctx)},
        "model_path": model_id,
        "chad": {
            "server": "chad-serve",
            "backend": "mlx",
            "capabilities": list(CAPABILITIES if capabilities is None else capabilities),
        },
    }


def sse(payload: dict) -> str:
    """Encode one payload as a Server-Sent-Events line, in the shape
    `completion_engine.parse_sse_chunk` consumes (`data: {…}` + a blank line)."""
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


def parse_completion_request(body: dict, tok: Any = None) -> dict:
    """Validate + normalize a `/completion` body into the args `Engine.generate` wants.

    `prompt` is normally the token-id array chad sends (the point of this protocol:
    no detokenize, no second chat template). A plain string is accepted too — purely
    so the endpoint is reachable from `curl` for a smoke test — and is encoded with
    the server's own tokenizer. Raises ValueError on anything malformed so the handler
    can answer 400 with a message instead of failing mid-stream."""
    if not isinstance(body, dict):
        raise ValueError("request body must be a JSON object")
    prompt = body.get("prompt")
    if isinstance(prompt, str):
        if tok is None:
            raise ValueError("string prompts need a tokenizer; send token ids")
        prompt_ids = list(tok.encode(prompt, add_special_tokens=False))
    elif isinstance(prompt, list):
        if not all(isinstance(t, int) and not isinstance(t, bool) for t in prompt):
            raise ValueError("prompt array must contain only integer token ids")
        prompt_ids = list(prompt)
    else:
        raise ValueError("prompt is required (array of token ids, or a string)")
    if not prompt_ids:
        raise ValueError("prompt is empty")

    def _num(name: str, default: Optional[float]) -> Optional[float]:
        v = body.get(name, default)
        if v is None:
            return default
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"{name} must be a number")
        return float(v)

    n_predict = int(_num("n_predict", 512) or 512)
    if n_predict <= 0:                      # llama.cpp's -1 means "until context end"
        n_predict = 512
    return {
        "prompt_ids": prompt_ids,
        "n_predict": n_predict,
        # Sampler knobs are None when ABSENT, which is not the same as 0.0. chad's
        # convention is 0 = OFF, so a client that means "disable min_p" says 0.0 and a
        # client that means "whatever the server is configured with" says nothing at all
        # — and `build_completion_body` only sends min_p/top_p when they are armed, so
        # "nothing at all" is the common case. Defaulting these to 0.0 here would ZERO a
        # server started with CHAD_MIN_P on every single request, which is the opposite
        # of leaving it alone (see stream_completion, which applies only what arrived).
        "temperature": _num("temperature", None),
        "min_p": _num("min_p", None),
        "top_p": _num("top_p", None),
        "cache_prompt": bool(body.get("cache_prompt", True)),
    }


def admit(prompt_len: int, n_predict: int, n_ctx: int,
          safe_ctx: Optional[int] = None) -> tuple[int, Optional[str]]:
    """Fit a request inside the context wall BEFORE anything is prefilled.

    Returns `(n_predict, error)`. A non-None error means refuse the request outright.

    TWO WALLS, and they are deliberately different numbers. `n_ctx` is what `/props`
    advertises: pinned at load, never moving, because clients read it ONCE and budget
    against it forever — a window that drifts with system load silently corrupts every
    budget derived from it. `safe_ctx` is what actually fits in GPU memory RIGHT NOW,
    and it must be live, because the Metal budget is blind to other processes: a docker
    stack or a second model started after us takes physical pages, and the KV cache grows
    into physical pages 1:1. Advertise the stable number, admit against the live one.

    This is admission control, not politeness. The KV cache has to hold prompt +
    generation, and a prompt that overruns it does not fail cleanly: it walks into a Metal
    allocation the engine may not be able to recover from, and an OOM raised on a Metal
    completion handler can take the process with it. A 400 costs the caller one request;
    an OOM costs every client the server has, plus whatever the running turn was worth.

    The client normally makes this unreachable — it reads `n_ctx` from /props and budgets
    against it — which is exactly why the check belongs here too: the guard exists for the
    caller that DIDN'T (a stale pin, a plain `curl`, a non-chad client, an off-by-a-turn
    budget), and those are the ones that would otherwise take the box down.

        0                        prompt_len              n_ctx
        ├───────── prompt ──────────┤─── room ─────────────┤
                                    └─ n_predict clamped to fit

    `n_ctx <= 0` means the engine hasn't reported a window yet; invent no wall rather than
    reject everything against a made-up number."""
    if n_ctx <= 0:
        return n_predict, None
    wall = n_ctx if not safe_ctx else min(n_ctx, safe_ctx)
    if prompt_len >= wall:
        if prompt_len < n_ctx:
            # Fits the advertised window but not today's memory. Say which, because the
            # two have different remedies: this one clears up on its own.
            return 0, (f"prompt is {prompt_len} tokens and this server advertises a "
                       f"{n_ctx}-token window, but only {wall} tokens fit in GPU memory "
                       f"right now — another process is holding it. Retry when memory "
                       f"frees, or send a shorter prompt.")
        return 0, (f"prompt is {prompt_len} tokens but this server's context window is "
                   f"{n_ctx}; send a shorter prompt (GET /props reports the window "
                   f"clients should budget against)")
    # llama.cpp semantics: generation simply stops when the context ends, so silently
    # fitting the budget is the expected behavior, not a refusal.
    return min(n_predict, wall - prompt_len), None


def timings_payload(stats: Any) -> dict:
    """Map a `GenStats` onto llama.cpp's final-chunk `timings`. These are what let the
    client report EXACT numbers (`approximate` stays False): `prompt_n` is the count
    actually prefilled after the prefix diff, not the whole prompt."""
    return {
        "prompt_n": int(getattr(stats, "prompt_tokens", 0)),
        "prompt_ms": float(getattr(stats, "prefill_s", 0.0)) * 1000.0,
        "predicted_n": int(getattr(stats, "generated_tokens", 0)),
        "predicted_ms": float(getattr(stats, "gen_s", 0.0)) * 1000.0,
        # Not a llama.cpp field; harmless to a client that ignores unknown keys and
        # useful in a raw curl session to see the prefix cache doing its job.
        "cache_n": int(getattr(stats, "cached_tokens", 0)),
    }


# --- generation (engine-driven, socket-agnostic: `write` is injected) -------

def stream_completion(eng: Any, params: dict, write: Callable[[str], None]) -> dict:
    """Run ONE `/completion` against `eng`, pushing SSE lines through `write`.

    `write` is the only I/O: tests pass a list-appender, the handler passes a socket
    writer. A write that raises (the client hung up) is the cancel signal — the flag it
    sets is what `should_stop` returns, so `Engine.generate` stops decoding on the next
    token instead of finishing a turn nobody is listening to.

    Returns a small dict summarizing the request (for the server's own log line)."""
    prompt_ids = params["prompt_ids"]
    gone = threading.Event()

    def emit(payload: dict) -> None:
        if gone.is_set():
            return
        try:
            write(sse(payload))
        except WRITE_FAILED:
            gone.set()

    def on_token(seg: str) -> None:
        # One SSE line per decoded token: the client counts emitted tokens by
        # non-empty content chunks, so batching text here would undercount its
        # think-ceiling and stop-condition bookkeeping.
        emit({"content": seg, "stop": False})

    if not params["cache_prompt"]:
        # Explicit opt-out: forfeit the prefix cache for this request. The measurement
        # arm that wants every prompt to prefill from scratch.
        eng.reset()

    # Per-request sampler knobs: apply ONLY what the request actually carried, and
    # restore afterwards so one request can't silently re-tune the next. A knob the
    # request omitted is left exactly as the operator configured it — writing a neutral
    # 0.0 over it would disarm CHAD_MIN_P/CHAD_TOP_P on every request, and forcing
    # temperature to 0.0 would additionally decide the engine's decode strategy for it
    # (`Engine.generate` gates the prompt-lookup path on temp == 0.0). Runs on the engine
    # thread, one request at a time, so there is no interleave.
    knobs = (("temp", "temperature"), ("min_p", "min_p"), ("top_p", "top_p"))
    saved = {attr: getattr(eng, attr, 0.0) for attr, _ in knobs}
    for attr, key in knobs:
        v = params.get(key)
        if v is not None:
            setattr(eng, attr, v)
    try:
        text, stats = eng.generate(
            prompt_ids,
            max_tokens=params["n_predict"],
            on_token=on_token,
            should_stop=gone.is_set,
        )
    finally:
        for attr, _ in knobs:
            setattr(eng, attr, saved[attr])

    if gone.is_set():
        # Client hung up: no final chunk to deliver it to. The engine's own cache is
        # intact and correct; only the client's mirror of it goes stale.
        return {"cancelled": True, "generated": 0}

    # The generated ids, reported by the engine itself. This used to be recovered by
    # slicing `eng._cached_ids` past the prompt, which is only `prompt + generation` on
    # the main decode path: the prompt-lookup path stores what it FED the cache (short by
    # the final pending token) and a Metal OOM empties it outright, so the slice was
    # quietly wrong on two of three paths. `GenStats.gen_ids` is exact on all of them.
    #
    # `cache_reset` is the case ids alone cannot express: after an OOM the server's cache
    # is EMPTY, so a client that mirrored `prompt + generation` would believe a prefix is
    # resident that isn't. It tells the client to drop its mirror instead of guessing.
    final = {
        "content": "",
        "stop": True,
        "tokens": list(getattr(stats, "gen_ids", None) or []),
        "timings": timings_payload(stats),
    }
    if getattr(stats, "cache_reset", False):
        final["cache_reset"] = True
    emit(final)
    emit_done(write, gone)
    return {"cancelled": False, "generated": int(getattr(stats, "generated_tokens", 0)),
            "prefilled": int(getattr(stats, "prompt_tokens", 0)),
            "cached": int(getattr(stats, "cached_tokens", 0)), "text_len": len(text)}


def emit_done(write: Callable[[str], None], gone: threading.Event) -> None:
    """Terminal `data: [DONE]` sentinel. The client treats it as end-of-stream (its
    parser returns None for it), so this is politeness for other SSE consumers."""
    if gone.is_set():
        return
    try:
        write("data: [DONE]\n\n")
    except (BrokenPipeError, ConnectionResetError, OSError):
        gone.set()


# --- server state -----------------------------------------------------------

class ServerState:
    """Everything the handlers touch: the engine, the single thread all engine work is
    funnelled onto, and the optional bearer token."""

    def __init__(self, eng: Any, model_id: str = "", api_key: str = "",
                 quiet: bool = False):
        self.eng = eng
        self.model_id = model_id
        self.api_key = api_key
        self.quiet = quiet
        self.busy = False
        self._safe_ctx_cache: tuple = (0.0, 0)   # (measured_at, tokens); see safe_ctx()
        # One worker: serializes access to the single KV cache AND keeps every MLX call
        # on one thread, which the thread-local GPU streams require (see module docstring).
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chad-engine")

    def submit(self, fn: Callable, *args: Any, **kw: Any) -> Any:
        """Queue `fn` for the engine thread and return its Future WITHOUT waiting. The
        streaming path needs this: its caller has to stay free to drain the SSE queue
        while the engine fills it (see `_Handler._completion`)."""
        return self._pool.submit(fn, *args, **kw)

    def call(self, fn: Callable, *args: Any, **kw: Any) -> Any:
        """Run `fn` on the engine thread and wait for it. Exceptions propagate to the
        caller unchanged, so a handler still answers with a real error."""
        return self.submit(fn, *args, **kw).result()

    def close(self) -> None:
        self._pool.shutdown(wait=False)

    def n_ctx(self) -> int:
        # The ADVERTISED window: pinned at load, never moving. Clients read it once and
        # budget against it for the life of the process, so it must not drift with system
        # load. Reads a plain int off the engine object — no MLX, so no executor hop, so
        # /props and /health answer during a long generation.
        return int(getattr(self.eng, "effective_ctx", 0) or 0)

    def safe_ctx(self) -> int:
        """How many tokens actually fit in GPU memory right now — the admission wall.

        The pinned window was computed against the memory this machine had at load. That
        is the wrong number to admit against later: the Metal budget is blind to other
        processes, so a docker stack or a browser started afterwards takes physical pages
        the KV cache needs 1:1, and the prompt that fit an hour ago now walks into an
        allocation failure the engine may not survive. Recomputed from LIVE memory, which
        is the whole point — a pinned safety check is not a safety check.

        Reuses `cli.ram_aware_ctx_limit`, the same measured math the local CLI sizes its
        compaction trigger with (measured per-token KV cost, Metal recommendation, and the
        host's own reclaimable band, since jetsam kills on physical pressure rather than
        Metal accounting). Its `floor`/`gen_margin` defaults are compaction policy, not
        admission policy, so they are passed explicitly here: never admit more than the
        pinned window, and never tighten below `SAFE_CTX_FLOOR` — a server that refuses
        everything under pressure is just a different outage.

        Cached briefly: ~3 ms per probe is fine per request but wasteful in a burst, and
        memory pressure does not move meaningfully inside a second. Never raises — if the
        probe is unavailable (no MLX, engine not loaded) the pinned window stands, which
        is exactly the behavior before this existed."""
        pinned = self.n_ctx()
        now = time.monotonic()
        cached_at, cached = self._safe_ctx_cache
        if now - cached_at < SAFE_CTX_TTL_S and cached:
            return min(pinned, cached)
        try:
            import mlx.core as mx

            from . import cli
            kv_per_token = float(getattr(self.eng, "kv_bytes_per_token", 0.0) or 0.0)
            # Subtract the resident KV so the floor measured mid-session is the same
            # model floor measured at startup — otherwise the wall would shrink as the
            # cache fills, which is the cache doing its job, not memory pressure.
            active_floor = (mx.get_active_memory()
                            - kv_per_token * len(getattr(self.eng, "_cached_ids", ()) or ()))
            live = cli.ram_aware_ctx_limit(
                pinned, mx.device_info()["max_recommended_working_set_size"],
                active_floor, kv_per_token,
                reserve_gb=cli._env_float("CHAD_CTX_RESERVE_GB") or 1.5,
                host_avail_bytes=cli._host_avail_bytes(),
                slope_factor=cli._env_float("CHAD_CTX_SLOPE_FACTOR") or 1.75,
                floor=SAFE_CTX_FLOOR, gen_margin=0)
        except Exception:  # noqa: BLE001 — a guard that can crash is worse than no guard
            live = None
        if not live:
            return pinned
        self._safe_ctx_cache = (now, int(live))
        return min(pinned, int(live))

    def spill_status(self) -> dict:
        """Whether the engine can reclaim a quarantined cache to disk under memory
        pressure. Observability, because this is invisible from the client side and
        silently disarms if either half is missing: the engine needs somewhere to write
        (`cache_dir`) AND a measured per-token cost to decide with. Both are plain
        attribute reads — no MLX, so no executor hop."""
        cache_dir = getattr(self.eng, "cache_dir", None)
        per_token = float(getattr(self.eng, "kv_bytes_per_token", 0.0) or 0.0)
        return {"cache_dir": cache_dir, "kv_bytes_per_token": per_token,
                "budget_bytes": int(getattr(self.eng, "kv_cache_max_bytes", 0) or 0),
                "armed": bool(cache_dir and per_token)}


def _make_handler(state: ServerState) -> type:
    """Build a handler class bound to `state` (avoids module-level mutable globals,
    which would make two servers in one process — as the tests run — share an engine)."""

    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "chad-serve"
        # Bounds every socket operation, so a peer that stops reading can never park a
        # thread forever. Without it `timeout` is None and a blocked send simply waits —
        # which is how a single suspended client used to take the whole server down.
        timeout = SOCKET_TIMEOUT_S
        # One SSE line per token is ~30 bytes: the exact small-write workload Nagle was
        # built to coalesce, and exactly where that coalescing shows up as token latency
        # against a client on the far side of a real link. Nothing to batch here, so
        # don't. (Measured no difference on loopback, where ACKs are instant and Nagle
        # never engages; this is for the container/LAN case the module docstring is about.)
        disable_nagle_algorithm = True

        # -- plumbing --------------------------------------------------------

        def log_message(self, fmt: str, *a: Any) -> None:
            if not state.quiet:
                sys.stderr.write("[serve] %s\n" % (fmt % a))

        def _authorized(self) -> bool:
            if not state.api_key:
                return True
            hdr = self.headers.get("Authorization", "")
            return hdr.strip() == f"Bearer {state.api_key}"

        def _read_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            if n > MAX_BODY_BYTES:
                raise ValueError("request body too large")
            if not n:
                return {}
            return json.loads(self.rfile.read(n).decode("utf-8", "replace"))

        def _send_json(self, code: int, obj: dict) -> None:
            data = json.dumps(obj).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, msg: str) -> None:
            self._send_json(code, {"error": {"code": code, "message": msg}})

        def _emit_raw(self, text: str, dead: threading.Event) -> None:
            """Write one already-encoded SSE payload to the socket, marking the peer dead
            if it refuses. The single place this thread touches the wire — a token line
            and a keepalive fail the same way and must be handled the same way."""
            if dead.is_set():
                return
            try:
                self.wfile.write(text.encode("utf-8"))
                self.wfile.flush()
            except (OSError, ValueError):
                # Includes socket.timeout via SOCKET_TIMEOUT_S: a peer that stopped
                # reading is indistinguishable from one that hung up, and both deserve
                # the same answer — stop decoding for it.
                dead.set()

        # -- routes ----------------------------------------------------------

        def do_GET(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's spelling
            if not self._authorized():
                return self._error(401, "missing or bad Authorization bearer token")
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            if path == "/props":
                # Deliberately lock-free: a monitor polling /props must not queue
                # behind a multi-minute generation.
                return self._send_json(200, props_payload(state.n_ctx(), state.model_id))
            if path in ("/health", "/"):
                # `n_ctx` is what clients budget against; `safe_ctx` is what memory
                # allows right now. They are equal on an idle machine and diverge under
                # pressure — which is invisible from the client side, so report it here
                # rather than leaving a 400 to be the first anyone hears of it.
                safe = state.safe_ctx()
                return self._send_json(200, {"status": "ok", "busy": state.busy,
                                             "n_ctx": state.n_ctx(),
                                             "safe_ctx": safe,
                                             "ctx_pressure": safe < state.n_ctx(),
                                             "model": state.model_id,
                                             "spill": state.spill_status()})
            return self._error(404, f"no such endpoint: {path}")

        def do_POST(self) -> None:  # noqa: N802
            if not self._authorized():
                return self._error(401, "missing or bad Authorization bearer token")
            path = self.path.split("?", 1)[0].rstrip("/") or "/"
            try:
                body = self._read_json()
            except (ValueError, json.JSONDecodeError) as e:
                return self._error(400, f"bad request body: {e}")

            if path == "/completion":
                return self._completion(body)
            if path == "/cache/push":
                return self._cache_op("push")
            if path == "/cache/pop":
                return self._cache_op("pop")
            if path == "/warm":
                return self._warm(body)
            return self._error(404, f"no such endpoint: {path}")

        # -- handlers --------------------------------------------------------

        def _completion(self, body: dict) -> None:
            try:
                params = parse_completion_request(body, tok=getattr(state.eng, "tok", None))
            except ValueError as e:
                return self._error(400, str(e))
            # Refuse what cannot fit BEFORE the response starts: once the stream is open
            # the status code is spent, and the only way left to report a failure is to
            # hang up mid-generation.
            params["n_predict"], too_big = admit(len(params["prompt_ids"]),
                                                 params["n_predict"], state.n_ctx(),
                                                 state.safe_ctx())
            if too_big:
                return self._error(400, too_big)

            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            # No Content-Length is knowable up front, so the stream IS the message and
            # the connection ends it. HTTP/1.1 keep-alive is off for this response only.
            self.send_header("Connection", "close")
            self.end_headers()
            self.close_connection = True

            # Generation runs on the engine thread (one KV cache → one generation, and
            # MLX's thread-local streams only exist there); THIS thread owns the socket.
            # The queue is the seam. See the module docstring for why the two must not be
            # the same thread.
            lines: queue.Queue = queue.Queue(maxsize=STREAM_QUEUE_MAX)
            done = object()             # sentinel: generation finished, drain what's left
            dead = threading.Event()    # set by this thread when the socket refuses bytes

            def enqueue(chunk: str) -> None:
                """`stream_completion`'s only I/O. Raising is the documented cancel
                signal, and both ways this can fail mean the same thing: nobody is
                reading. Never blocks longer than WRITE_STALL_S."""
                if dead.is_set():
                    raise BrokenPipeError("client hung up")
                lines.put(chunk, timeout=WRITE_STALL_S)      # queue.Full → cancel

            def generate() -> dict:
                try:
                    return stream_completion(state.eng, params, enqueue)
                finally:
                    try:
                        lines.put_nowait(done)
                    except queue.Full:
                        # Never wait here. A full queue at this point means the drain is
                        # still stuck on a client that stopped reading — the exact case
                        # that just cancelled us — and blocking would make the engine
                        # thread serve out a SECOND stall timeout before coming free.
                        # The drain's future-done check retires it instead.
                        pass

            state.busy = True
            fut = state.submit(generate)
            last_sent = time.monotonic()
            try:
                while True:
                    try:
                        chunk = lines.get(timeout=0.5)
                    except queue.Empty:
                        if fut.done():
                            break
                        # Silence. A cold prefill of a long prompt produces no tokens for
                        # minutes, and a client measures the wait on BYTES, not on our
                        # progress: every HTTP client has an idle-read timeout, chad's own
                        # included (CompletionEngine passes `timeout` to urlopen, which
                        # applies it per read). Staying quiet through a long prefill gets
                        # the turn killed just before the first token, and because a socket
                        # timeout looks transient the agent re-issues the step — paying for
                        # the same prefill again, and again. So keep the socket warm.
                        if (not dead.is_set()
                                and time.monotonic() - last_sent >= KEEPALIVE_S):
                            last_sent = time.monotonic()
                            self._emit_raw(": keepalive\n\n", dead)
                        continue
                    if chunk is done:
                        break
                    if dead.is_set():
                        # Socket is gone; keep draining anyway so the engine never blocks
                        # on a full queue while it winds down.
                        continue
                    last_sent = time.monotonic()
                    self._emit_raw(chunk, dead)
                info = fut.result()
            finally:
                state.busy = False
            if not state.quiet:
                sys.stderr.write(f"[serve] completion {info}\n")

        def _cache_op(self, which: str) -> None:
            """Cache quarantine: bracket a sub-agent's excursion so its prompt doesn't
            evict the main transcript's prefix. A failure here is reported but is never
            fatal — the caller degrades to the plain remote behavior (re-prefill)."""
            try:
                state.call(state.eng.push_cache if which == "push"
                           else state.eng.pop_cache)
            except Exception as e:  # noqa: BLE001 — latency feature, not correctness
                return self._send_json(200, {"ok": False, "error": str(e)})
            return self._send_json(200, {"ok": True, "op": which})

        def _warm(self, body: dict) -> None:
            """On-disk KV warm-start of a stable prefix — impossible for a remote client
            to do itself, since the checkpoint lives on this machine's disk."""
            prefix = body.get("prefix")
            if not isinstance(prefix, list) or not all(
                    isinstance(t, int) and not isinstance(t, bool) for t in prefix):
                return self._error(400, "prefix must be an array of token ids")
            # Same wall as /completion: warming a prefix that cannot fit is a prefill
            # that cannot fit, and it reaches the allocator the same way.
            _, too_big = admit(len(prefix), 0, state.n_ctx(), state.safe_ctx())
            if too_big:
                return self._error(400, too_big)
            state.busy = True
            try:
                status, fed = state.call(state.eng.warm_prefix, list(prefix))
            except Exception as e:  # noqa: BLE001 — same: degrade, don't fail a run
                return self._send_json(200, {"status": "error", "fed": 0,
                                             "error": str(e)})
            finally:
                state.busy = False
            return self._send_json(200, {"status": status, "fed": int(fed)})

    return _Handler


def build_server(state: ServerState, host: str = DEFAULT_HOST,
                 port: int = DEFAULT_PORT) -> ThreadingHTTPServer:
    """Bind the HTTP server. Threaded so `/props` and `/health` answer while a
    generation holds the engine lock; the lock, not the thread count, is what keeps
    the single KV cache safe."""
    srv = ThreadingHTTPServer((host, port), _make_handler(state))
    srv.daemon_threads = True
    return srv


# --- entry point (dispatched from cli.main on the literal task `serve`) -----

def run(args: Any) -> int:
    """Load the local model and serve it. Returns the process exit code."""
    from . import cli, config

    if getattr(args, "backend", "mlx") != "mlx":
        sys.stderr.write(
            f"chad serve exposes the LOCAL MLX engine; --backend {args.backend} has "
            "nothing to serve (it is itself a client of a remote server).\n")
        return 2
    try:
        cli._preflight("mlx")
    except SystemExit:
        return 2

    host = args.host or config.env_str("CHAD_SERVE_HOST", DEFAULT_HOST)
    port = int(args.port or config.env_int("CHAD_SERVE_PORT", DEFAULT_PORT))
    # Unauthenticated by default, which is why the default bind is loopback. A LAN
    # bind (what a container on another host needs) should set a token.
    api_key = config.env_str("CHAD_SERVE_API_KEY", "") or ""
    if host not in ("127.0.0.1", "localhost", "::1") and not api_key:
        sys.stderr.write(
            f"[serve] warning: binding {host} with no CHAD_SERVE_API_KEY — anyone who "
            "can reach this port can spend your GPU.\n")

    from .engine import Engine, sweep_orphan_spills
    model_id, why = cli._pick_model(getattr(args, "model", None))
    cli._ensure_model(model_id)
    cache_dir = os.path.expanduser("~/.cache/chad/kv")
    sweep_orphan_spills(cache_dir, max_age_s=6 * 3600)
    kv_cache_max_gb = cli._env_int("CHAD_KV_CACHE_MAX_GB")
    eng = Engine(
        model_id=model_id,
        kv_bits=cli._env_int("CHAD_KV_BITS"),
        max_context=cli._env_int("CHAD_MAX_CONTEXT"),
        cache_dir=cache_dir,
        kv_cache_max_bytes=(kv_cache_max_gb if kv_cache_max_gb is not None else 8) * 1024**3,
    )
    # The same sampler-env application the local CLI does, through the same call — a
    # server started with CHAD_MIN_P must sample like a local chad started with it, or the
    # whole point of serving THIS model (measuring what people actually run) is lost.
    # Shared rather than re-listed here on purpose: see cli.apply_sampler_env.
    cli.apply_sampler_env(eng)
    sys.stderr.write(f"loading {os.path.basename(model_id.rstrip('/'))} [{why}] ...\n")
    # The weights load ON the engine thread, not this one: MLX's streams are
    # thread-local, so whatever thread loads the model must be the thread that later
    # prefills against its cache.
    state = ServerState(eng, model_id=model_id, api_key=api_key)
    try:
        load_s = state.call(eng.load)
    except Exception as e:  # noqa: BLE001 — same guidance the CLI gives on a bad load
        state.close()
        cli._fail_model_load(model_id, e)
        return 2
    # Warm the memory probe before anything can call it. The first `safe_ctx()` pays a
    # module import and a cold Metal device query (~55 ms measured); that belongs to
    # startup, not to whichever request happens to arrive first.
    state.safe_ctx()
    try:
        srv = build_server(state, host, port)
    except OSError as e:
        state.close()
        sys.stderr.write(f"chad serve: cannot bind {host}:{port} — {e}\n")
        return 2

    sys.stderr.write(
        f"ready in {load_s:.1f}s | context {state.n_ctx()} tokens\n"
        f"serving {model_id} on http://{host}:{port} "
        f"(llama.cpp /completion protocol{', auth on' if api_key else ''})\n"
        f"point a client at it with:  chad \"…\" --backend llama --base-url "
        f"http://{host}:{port}\n")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        sys.stderr.write("\n[serve] shutting down\n")
    finally:
        srv.server_close()
        state.close()
    return 0
