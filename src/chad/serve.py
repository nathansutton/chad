"""`chad --serve` — an OpenAI-compatible endpoint on chad's OWN engine (plan 075 WS3).

WHY THIS EXISTS
---------------
Terminal-Bench 2.0 runs chad *inside* each task container with a remote backend
pointed at the host Mac. The stock server for that arm (`mlx_lm.server`) cannot
reuse a hybrid-model prompt cache across requests, so every agent step re-prefills
the ENTIRE growing transcript — ~45 s/step at 30k context, the real reason the
Mac arm timed out. chad's in-process engine already solves exactly this with the
persistent prefix KV cache; this module puts a thin OpenAI-shaped HTTP surface on
that engine so the warm-cache machinery serves remote chad agents too.

THE TRICK: TEXT-DIFF, NOT RE-TEMPLATE
-------------------------------------
chad's `--backend openai` client ships the *fully rendered* prompt (decoded token
ids, special tokens intact) as a single user message. Re-encoding that whole text
each request would risk BPE merges across the previous generation boundary and
desync the cache. Instead, per session we remember the last request's text and the
exact token ids the cache holds: when the new text merely EXTENDS the old (the
append-only agentic common case), we encode only the appended suffix and splice it
onto the known ids — exact id continuity, zero re-encode, warm prefix every step.
A non-extending request (client compacted, new task, different session) falls back
to a full encode, and the engine's own `_sync_to` prefix diffing takes it from there.

Generic OpenAI clients (multi-message bodies without chad's rendered markers) get
the model's chat template applied server-side, like any other OpenAI-compatible
server. One request at a time (a lock serializes the engine); TB2 runs with
`--n-concurrent-agents 1`, which is exactly this shape.
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from .diag import log

# Marker that identifies a chad-client "pre-rendered" prompt (the Qwen-family chat
# template chad ships). Presence in a single-user-message body means the text IS the
# final prompt — encode it verbatim, do not re-apply a chat template around it.
_RENDERED_MARK = "<|im_start|>"


def is_prerendered(messages: list) -> bool:
    """A chad `--backend openai` client sends exactly one user message whose content
    is the already-templated prompt. Detect it by the template's own markers so the
    client needs no protocol extension (and other servers keep working unchanged)."""
    return (len(messages) == 1
            and messages[0].get("role") == "user"
            and _RENDERED_MARK in (messages[0].get("content") or ""))


def extend_ids(tok, prev_text: str, new_text: str, prev_ids: list) -> Optional[list]:
    """Exact-id splice for an append-only session: if `new_text` extends `prev_text`
    and we know the ids `prev_text` corresponds to, encode ONLY the appended suffix
    and splice. Returns None when the request does not extend the session (caller
    falls back to a full encode). Avoids re-encoding the transcript AND the BPE
    boundary-merge hazard at the previous generation's tail."""
    if not (prev_text and prev_ids and new_text.startswith(prev_text)):
        return None
    suffix = new_text[len(prev_text):]
    if not suffix:
        return list(prev_ids)
    return list(prev_ids) + tok.encode(suffix, add_special_tokens=False)


def parse_request(body: dict) -> dict:
    """Pull the fields we honor out of a /v1/chat/completions body (pure)."""
    stop = body.get("stop")
    if isinstance(stop, str):
        stop = [stop]
    return {
        "messages": body.get("messages") or [],
        "stream": bool(body.get("stream")),
        "max_tokens": int(body.get("max_tokens")
                          or body.get("max_completion_tokens") or 2048),
        "temperature": body.get("temperature"),
        "stop": stop or None,
        "model": body.get("model") or "",
    }


def chunk_payload(model: str, created: int, rid: str, *, delta: str = None,
                  finish: str = None, usage: dict = None) -> dict:
    """One streamed chat-completion chunk (pure). Exactly one of delta/finish/usage
    modes; the usage chunk has empty choices per `stream_options.include_usage`."""
    base = {"id": rid, "object": "chat.completion.chunk", "created": created,
            "model": model}
    if usage is not None:
        return {**base, "choices": [], "usage": usage}
    delta_obj = {"content": delta} if delta is not None else {}
    return {**base, "choices": [{"index": 0, "delta": delta_obj,
                                 "finish_reason": finish}]}


def usage_dict(prompt_tokens: int, cached: int, completion: int) -> dict:
    """OpenAI-shaped usage block; cache reuse rides in prompt_tokens_details the way
    OpenAI reports its own prompt caching, so standard clients can read it."""
    return {"prompt_tokens": prompt_tokens + cached,
            "completion_tokens": completion,
            "total_tokens": prompt_tokens + cached + completion,
            "prompt_tokens_details": {"cached_tokens": cached}}


class _Session:
    """Text/ids bookkeeping for the (single) live conversation. `text` is the last
    request's prompt plus what we generated for it; `ids` are the exact token ids
    that text corresponds to (and that the engine's cache holds)."""

    def __init__(self):
        self.text: str = ""
        self.ids: list = []

    def update(self, prompt_text: str, prompt_ids: list, gen_text: str,
               engine_ids: list) -> None:
        """Record the post-generation state. `engine_ids` (the engine's resident ids)
        is authoritative when it extends the prompt — it includes the REAL generated
        ids, no re-encode. If the engine degraded (e.g. OOM reset), drop the session:
        the next request full-encodes and re-prefills, which is always correct."""
        if len(engine_ids) >= len(prompt_ids) and engine_ids[:len(prompt_ids)] == prompt_ids:
            self.text = prompt_text + gen_text
            self.ids = list(engine_ids)
        else:
            self.text = ""
            self.ids = []


def make_server(engine: Any, host: str = "127.0.0.1", port: int = 8090,
                served_model: str = "") -> ThreadingHTTPServer:
    """Build (not start) the HTTP server bound to `engine`. Split from `serve` so
    tests can drive a real HTTP round-trip against a fake engine on an OS-picked
    port."""
    # ALL engine work runs on this single worker thread, permanently. MLX streams
    # and the KV-cache arrays are thread-affine: ThreadingHTTPServer hands every
    # request a fresh thread, and a cache built on request 1's thread blows up with
    # "There is no Stream(gpu, N) in current thread" when request 2 touches it.
    # One worker = one stream = one live cache, and it doubles as the serialization
    # point (one request at a time; queued requests wait on their future).
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="chad-serve-engine")
    session = _Session()
    model_name = served_model or getattr(engine, "model_id", "chad")

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, fmt, *args):  # route http.server noise to our log
            log.info("serve: " + fmt, *args)

        def _json(self, code: int, payload: dict) -> None:
            data = json.dumps(payload).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            if self.path == "/health":
                self._json(200, {"status": "ok"})
            elif self.path == "/v1/models":
                self._json(200, {"object": "list", "data": [
                    {"id": model_name, "object": "model", "created": 0,
                     "owned_by": "chad"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self._json(404, {"error": "not found"})
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n) or b"{}")
                req = parse_request(body)
                if not req["messages"]:
                    self._json(400, {"error": "messages required"})
                    return
            except (ValueError, json.JSONDecodeError) as e:
                self._json(400, {"error": f"bad request: {e}"})
                return
            # Hand the request to the single engine worker thread and wait. The
            # worker writes the response (including SSE streaming) directly to this
            # handler's wfile; the handler thread just blocks until it's done.
            try:
                worker.submit(self._complete, req).result()
            except BrokenPipeError:
                log.info("serve: client dropped mid-stream")
            except Exception as e:  # noqa: BLE001 — a request must not kill the server
                log.warning("serve: request failed: %s", e)
                try:
                    self._json(500, {"error": str(e)})
                except Exception:  # noqa: BLE001 — headers may already be out
                    pass

        # -- request handling (inside the lock) ---------------------------

        def _prompt_ids(self, req: dict) -> tuple[list, str, bool]:
            """(ids, raw_text, prerendered). Chad clients: exact-id splice against
            the session, else full encode of the rendered text. Generic clients:
            the model's chat template."""
            msgs = req["messages"]
            if is_prerendered(msgs):
                text = msgs[0]["content"]
                ids = extend_ids(engine.tok, session.text, text, session.ids)
                if ids is None:
                    ids = engine.tok.encode(text, add_special_tokens=False)
                return ids, text, True
            ids = engine.tok.apply_chat_template(msgs, add_generation_prompt=True)
            return list(ids), "", False

        def _complete(self, req: dict) -> None:
            # Sampling entropy: engine._KeyedSampler draws a fresh explicit key per
            # generate call. Do NOT rely on mx.random.seed() here — this method runs
            # on the engine worker thread, where MLX's implicit-key RNG state is
            # never consumed (seeding it is a no-op and unseeded draws replay one
            # frozen noise vector — byte-identical responses, the night-7 class).
            ids, raw_text, prerendered = self._prompt_ids(req)
            if req["temperature"] is not None:
                engine.temp = float(req["temperature"])
            rid = f"chatcmpl-{int(time.time() * 1000)}"
            created = int(time.time())
            t0 = time.time()
            dropped = {"pipe": False}

            def send_sse(payload: dict) -> None:
                if dropped["pipe"]:
                    return
                try:
                    self.wfile.write(b"data: " + json.dumps(payload).encode("utf-8")
                                     + b"\n\n")
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    dropped["pipe"] = True

            on_token = None
            if req["stream"]:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Connection", "close")
                self.end_headers()
                on_token = lambda seg: send_sse(  # noqa: E731
                    chunk_payload(model_name, created, rid, delta=seg))

            text, stats = engine.generate(
                list(ids),
                max_tokens=req["max_tokens"],
                on_token=on_token,
                stop_texts=req["stop"],
                should_stop=(lambda: dropped["pipe"]) if req["stream"] else None,
            )
            if prerendered:
                session.update(raw_text, list(ids), text,
                               getattr(engine, "_cached_ids", []) or [])
            usage = usage_dict(stats.prompt_tokens, stats.cached_tokens,
                               stats.generated_tokens)
            log.info("serve: %d new + %d cached -> %d tok in %.1fs%s",
                     stats.prompt_tokens, stats.cached_tokens,
                     stats.generated_tokens, time.time() - t0,
                     " [client dropped]" if dropped["pipe"] else "")
            if req["stream"]:
                send_sse(chunk_payload(model_name, created, rid, finish="stop"))
                send_sse(chunk_payload(model_name, created, rid, usage=usage))
                if not dropped["pipe"]:
                    try:
                        self.wfile.write(b"data: [DONE]\n\n")
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                self.close_connection = True
            else:
                self._json(200, {
                    "id": rid, "object": "chat.completion", "created": created,
                    "model": model_name,
                    "choices": [{"index": 0, "finish_reason": "stop",
                                 "message": {"role": "assistant", "content": text}}],
                    "usage": usage,
                })

    return ThreadingHTTPServer((host, port), Handler)


def serve(engine: Any, host: str = "127.0.0.1", port: int = 8090) -> None:
    """Run the server until interrupted. The engine must already be loaded."""
    # Mirror the "chad" logger to stdout for the serve lifetime: a server run's
    # forensics (serve: request lines, engine REWIND/prefill lines) must land in the
    # captured stdout of whatever launched us (TB2 jobs tee it), not only in the
    # rotating ~/.chad/session.log on the serving machine.
    import logging
    import sys
    mirror = logging.StreamHandler(sys.stdout)
    mirror.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger("chad").addHandler(mirror)
    httpd = make_server(engine, host, port)
    log.info("serve: listening on %s:%d (model %s)", host, port,
             getattr(engine, "model_id", "?"))
    print(f"chad serving {getattr(engine, 'model_id', '?')} "
          f"on http://{host}:{port}/v1 (ctrl-c to stop)", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nchad serve: stopped")
    finally:
        httpd.server_close()
