"""`chad-serve`: the MLX engine behind an OpenAI-compatible /v1/chat/completions.

Other coding agents (Pi, anything that speaks OpenAI chat completions) get the same
drafted decode and persistent prefix cache `chad` itself runs on, without chad's agent
loop. One model, one request at a time, loopback by default: this is a single-user
server for the machine it runs on, not a multi-tenant endpoint.

The default model is the ternary Bonsai 2 27B pack with its DFlash2 drafter, which is
the build that fits a 16 GB Mac; `--model` / CHAD_MODEL select another.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import re
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING

from . import config

if TYPE_CHECKING:
    from .engine import Engine

DEFAULT_MODEL = "nathansutton/Qwen3.8-27B-Ternary-Bonsai-2-DFlash2-MLX"
MODEL_ALIAS = "bonsai-2-27b-mlx"
DEFAULT_MAX_TOKENS = 8192
THINK_END = "</think>"
TOOL_MARKS = ("<tool_call>", "<function=")
# Bonsai 2 does not support low effort; anything below medium runs as medium.
EFFORT = {"minimal": "medium", "low": "medium", "medium": "medium",
          "high": "xhigh", "xhigh": "xhigh"}

_FUNC_RE = re.compile(r"<function=([^>\s]+)\s*>(.*?)(?:</function>|\Z)", re.DOTALL)
_PARAM_RE = re.compile(r"<parameter=([^>\s]+)\s*>(.*?)</parameter>", re.DOTALL)
_JSON_CALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)


def log(msg: str) -> None:
    sys.stderr.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n")
    sys.stderr.flush()


# --- request -> template messages ------------------------------------------------------

def text_of(content) -> str:
    """Flatten OpenAI content (string or parts list) to text; the model is text-only."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for item in content:
        if isinstance(item, str):
            parts.append(item)
        elif isinstance(item, dict) and isinstance(item.get("text"), str):
            parts.append(item["text"])
        elif isinstance(item, dict) and item.get("type") in ("image_url", "image"):
            parts.append("[image omitted: this model is text-only]")
    return "\n".join(parts)


def memo_key(content: str, calls: list) -> str:
    sig = json.dumps([content.strip(),
                      [[c["function"]["name"], c["function"]["arguments"]] for c in calls]],
                     sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(sig.encode()).hexdigest()


def normalize_messages(messages: list, memo: dict) -> list:
    """Map OpenAI messages onto what the Qwen3.8 template renders.

    The template re-renders each assistant turn from reasoning + content + tool calls.
    Clients often drop the reasoning on the way back, which would make the re-render
    diverge from the tokens already in the KV cache and force a re-prefill; `memo`
    restores the reasoning this server generated for that turn."""
    out, system = [], []
    for m in messages:
        role = m.get("role")
        content = text_of(m.get("content"))
        if role in ("system", "developer"):
            system.append(content)
        elif role == "user":
            out.append({"role": "user", "content": content})
        elif role == "assistant":
            calls = []
            for tc in m.get("tool_calls") or []:
                fn = tc.get("function") or {}
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args.strip() else {}
                    except json.JSONDecodeError:
                        args = {"arguments": args}
                if not isinstance(args, dict):
                    args = {}
                calls.append({"type": "function",
                              "function": {"name": fn.get("name") or "", "arguments": args}})
            reasoning = (m.get("reasoning_content") or m.get("reasoning")
                         or memo.get(memo_key(content, calls)) or "")
            msg: dict = {"role": "assistant", "content": content,
                   "reasoning_content": reasoning if isinstance(reasoning, str) else ""}
            if calls:
                msg["tool_calls"] = calls
            out.append(msg)
        elif role == "tool":
            out.append({"role": "tool", "content": content})
    if system:
        out.insert(0, {"role": "system", "content": "\n\n".join(s for s in system if s)})
    return out


# --- model text -> OpenAI message ------------------------------------------------------

def _partial_suffix(s: str, marks) -> int:
    """Length of the longest tail of `s` that could still grow into one of `marks`."""
    best = 0
    for mark in marks:
        for k in range(min(len(mark) - 1, len(s)), 0, -1):
            if s.endswith(mark[:k]):
                best = max(best, k)
                break
    return best


class Splitter:
    """Split the streamed completion into reasoning, content and the tool-call tail.

    Generation starts inside the template-opened <think> block, so everything up to
    </think> is reasoning. Text that might be the start of a marker is held back until
    it is known not to be one."""

    def __init__(self):
        self.phase = "think"
        self.buf = ""
        self.reasoning, self.content = [], []
        self.tool = ""

    def feed(self, seg: str) -> list:
        out = []
        self.buf += seg
        while True:
            if self.phase == "think":
                i = self.buf.find(THINK_END)
                if i >= 0:
                    out.append(("reasoning", self.buf[:i]))
                    self.buf = self.buf[i + len(THINK_END):]
                    self.phase = "content"
                    continue
                keep = _partial_suffix(self.buf, (THINK_END,))
                emit, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                out.append(("reasoning", emit))
            elif self.phase == "content":
                if not self.content:
                    self.buf = self.buf.lstrip()
                hits = [i for i in (self.buf.find(m) for m in TOOL_MARKS) if i >= 0]
                if hits:
                    i = min(hits)
                    out.append(("content", self.buf[:i].rstrip()))
                    self.tool, self.buf, self.phase = self.buf[i:], "", "tool"
                else:
                    # Whitespace before a possible marker is held too: it is dropped if a
                    # tool call follows.
                    keep = _partial_suffix(self.buf, TOOL_MARKS)
                    body = self.buf[:len(self.buf) - keep]
                    keep += len(body) - len(body.rstrip())
                    emit, self.buf = self.buf[:len(self.buf) - keep], self.buf[len(self.buf) - keep:]
                    out.append(("content", emit))
            else:
                self.tool += self.buf
                self.buf = ""
            break
        return self._record(out)

    def finish(self) -> list:
        out = []
        if self.phase == "think":
            out.append(("reasoning", self.buf))
        elif self.phase == "content":
            out.append(("content", self.buf.rstrip()))
        self.buf = ""
        return self._record(out)

    def _record(self, out: list) -> list:
        out = [(kind, text) for kind, text in out if text]
        for kind, text in out:
            (self.reasoning if kind == "reasoning" else self.content).append(text)
        return out


def _coerce(value: str, schema) -> object:
    """XML parameters arrive as text; turn non-string ones back into JSON values."""
    types = schema.get("type") if isinstance(schema, dict) else None
    types = types if isinstance(types, list) else [types]
    if "string" in types or types == [None]:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def parse_tool_calls(text: str, tools: list) -> list:
    """Parse the tool-call tail into [(name, args)]. Handles the template's XML dialect
    and the JSON-in-<tool_call> form quantized models sometimes fall back to."""
    schemas = {}
    for t in tools or []:
        fn = t.get("function") or t
        schemas[fn.get("name")] = ((fn.get("parameters") or {}).get("properties") or {})
    calls = []
    for fm in _FUNC_RE.finditer(text):
        name = fm.group(1).strip()
        props = schemas.get(name, {})
        args = {}
        for pm in _PARAM_RE.finditer(fm.group(2)):
            key, val = pm.group(1).strip(), pm.group(2)
            val = val[1:] if val.startswith("\n") else val
            val = val[:-1] if val.endswith("\n") else val
            args[key] = _coerce(val, props.get(key))
        calls.append((name, args))
    if calls:
        return calls
    for jm in _JSON_CALL_RE.finditer(text):
        try:
            obj = json.loads(jm.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", obj.get("parameters", {}))
        if isinstance(obj.get("name"), str):
            calls.append((obj["name"], args if isinstance(args, dict) else {}))
    return calls


# --- engine worker ---------------------------------------------------------------------

def _number(value, default: float) -> float:
    """A request's sampler knob, or the engine's own setting when it sent none."""
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) \
        else default


class Job:
    def __init__(self, body: dict):
        self.body = body
        self.events: queue.Queue = queue.Queue()
        self.cancel = threading.Event()


class Worker(threading.Thread):
    """Owns the engine. MLX state is thread-bound, so load and every generate run here."""

    def __init__(self, model_spec: str | None = None):
        super().__init__(daemon=True)
        self.model_spec = model_spec
        self.jobs: queue.Queue = queue.Queue()
        self.ready = threading.Event()
        self.failed: str | None = None
        self.memo: dict = {}

    def run(self):
        try:
            from .cli import (
                _compute_ctx_limit,
                _ensure_model,
                _env_int,
                _pick_model,
                apply_sampler_env,
                apply_sampler_preset,
            )
            from .engine import Engine
            model_id, why = _pick_model(self.model_spec)
            _ensure_model(model_id)
            eng = Engine(model_id=model_id, cache_dir=None,
                         kv_bits=_env_int("CHAD_KV_BITS"),
                         max_context=_env_int("CHAD_MAX_CONTEXT"))
            apply_sampler_preset(eng, thinking=True)
            apply_sampler_env(eng)
            self._sampler = (eng.temp, eng.top_p, eng.top_k, eng.min_p, eng.presence_penalty)
            log(f"loading {model_id} [{why}] ...")
            load_s = eng.load()
            log(f"ready in {load_s:.1f}s | context {eng.effective_ctx} tokens "
                f"(safe limit {_compute_ctx_limit(eng)})")
        except Exception as e:  # noqa: BLE001 — surface any load failure on /health
            self.failed = f"{type(e).__name__}: {e}"
            log(f"load failed: {self.failed}")
            self.ready.set()
            return
        self.ready.set()
        while True:
            job = self.jobs.get()
            if job.cancel.is_set():
                continue
            try:
                self._run(job, eng, _compute_ctx_limit(eng))
            except Exception as e:  # noqa: BLE001 — one bad request must not kill the engine
                log(f"generation failed: {type(e).__name__}: {e}")
                try:
                    eng.reset()
                except Exception:  # noqa: BLE001
                    pass
                job.events.put(("error", 500, f"{type(e).__name__}: {e}"))

    def _run(self, job: Job, eng: "Engine", ctx_limit: int | None):
        body = job.body
        tools = body.get("tools") or None
        messages = normalize_messages(body.get("messages") or [], self.memo)
        effort = (EFFORT.get(str(body.get("reasoning_effort") or "").lower())
                  or eng.reasoning_effort_default or "medium")
        try:
            ids = eng.tok.apply_chat_template(messages, tools=tools, add_generation_prompt=True,
                                              enable_thinking=True, reasoning_effort=effort)
        except Exception as e:  # noqa: BLE001 — template errors are the client's request
            job.events.put(("error", 400, f"could not render messages: {e}"))
            return
        ids = list(ids)

        limit = min(ctx_limit or eng.effective_ctx, eng.effective_ctx)
        if len(ids) >= limit:
            job.events.put(("error", 400,
                            f"context length exceeded: prompt is {len(ids)} tokens, "
                            f"maximum context length is {limit} tokens"))
            return
        want = body.get("max_completion_tokens") or body.get("max_tokens") or DEFAULT_MAX_TOKENS
        max_new = max(1, min(int(want), eng.effective_ctx - len(ids) - 8))

        # The model card's thinking recipe is the default; a request may override it.
        temp, top_p, top_k, min_p, presence = self._sampler
        eng.temp = _number(body.get("temperature"), temp)
        eng.top_p = _number(body.get("top_p"), top_p)
        eng.top_k = int(_number(body.get("top_k"), top_k))
        eng.min_p = _number(body.get("min_p"), min_p)
        eng.presence_penalty = _number(body.get("presence_penalty"), presence)

        text, stats = eng.generate(
            ids, max_tokens=max_new,
            on_token=lambda seg: job.events.put(("token", seg)),
            should_stop=job.cancel.is_set,
            on_prefill=lambda new, cached: job.events.put(("prefill", new, cached)))
        log(f"prompt {stats.prompt_tokens} new + {stats.cached_tokens} cached "
            f"({stats.prefill_s:.1f}s) | generated {stats.generated_tokens} at "
            f"{stats.tok_per_s:.1f} tok/s | draft accept {stats.accept_rate:.0%}")
        job.events.put(("done", stats, stats.generated_tokens >= max_new))


WORKER = Worker()   # replaced by main() once the model choice is known


# --- HTTP ------------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def _json(self, code: int, obj: dict):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _error(self, code: int, message: str):
        kind = "invalid_request_error" if code < 500 else "server_error"
        self._json(code, {"error": {"message": message, "type": kind, "code": code}})

    def do_GET(self):
        path = self.path.split("?")[0].rstrip("/")
        if path in ("/health", "/v1/health"):
            if WORKER.failed:
                self._json(500, {"status": "error", "error": WORKER.failed})
            elif WORKER.ready.is_set():
                self._json(200, {"status": "ok"})
            else:
                self._json(503, {"status": "loading"})
        elif path == "/v1/models":
            self._json(200, {"object": "list", "data": [
                {"id": MODEL_ALIAS, "object": "model", "owned_by": "local"}]})
        else:
            self._error(404, "not found")

    def do_POST(self):
        if self.path.split("?")[0].rstrip("/") != "/v1/chat/completions":
            return self._error(404, "not found")
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        except (ValueError, json.JSONDecodeError):
            return self._error(400, "request body is not valid JSON")
        WORKER.ready.wait()
        if WORKER.failed:
            return self._error(500, f"model failed to load: {WORKER.failed}")
        job = Job(body)
        WORKER.jobs.put(job)
        try:
            if body.get("stream"):
                self._stream(job)
            else:
                self._blocking(job)
        except (BrokenPipeError, ConnectionResetError):
            job.cancel.set()

    # Both paths drain the same event queue; `emit` is None for the blocking one.
    def _consume(self, job: Job, emit, idle):
        split = Splitter()
        while True:
            try:
                ev = job.events.get(timeout=5)
            except queue.Empty:
                idle()
                continue
            if ev[0] == "token":
                for kind, text in split.feed(ev[1]):
                    if emit:
                        emit({"reasoning_content" if kind == "reasoning" else "content": text})
            elif ev[0] == "error":
                return ev, split
            elif ev[0] == "done":
                for kind, text in split.finish():
                    if emit:
                        emit({"reasoning_content" if kind == "reasoning" else "content": text})
                return ev, split

    def _result(self, job: Job, split: Splitter, stats, hit_cap: bool):
        tools = job.body.get("tools") or []
        reasoning = "".join(split.reasoning).strip()
        content = "".join(split.content).strip()
        calls = parse_tool_calls(split.tool, tools) if split.tool else []
        leftover = "" if calls else split.tool   # an unparseable tail is shown, not dropped
        tool_calls = [{"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                       "function": {"name": name,
                                    "arguments": json.dumps(args, ensure_ascii=False)}}
                      for name, args in calls]
        if reasoning:
            key_calls = [{"function": {"name": n, "arguments": a}} for n, a in calls]
            WORKER.memo[memo_key(content + leftover, key_calls)] = reasoning
        finish = "tool_calls" if tool_calls else ("length" if hit_cap else "stop")
        total_prompt = stats.prompt_tokens + stats.cached_tokens
        usage = {"prompt_tokens": total_prompt, "completion_tokens": stats.generated_tokens,
                 "total_tokens": total_prompt + stats.generated_tokens,
                 "prompt_tokens_details": {"cached_tokens": stats.cached_tokens}}
        timings = {"prompt_n": stats.prompt_tokens, "cache_n": stats.cached_tokens,
                   "prompt_s": round(stats.prefill_s, 3),
                   "predicted_n": stats.generated_tokens,
                   "predicted_per_second": round(stats.tok_per_s, 2),
                   "draft_accept_rate": round(stats.accept_rate, 3)}
        return reasoning, content, leftover, tool_calls, finish, usage, timings

    def _blocking(self, job: Job):
        ev, split = self._consume(job, None, lambda: None)
        if ev[0] == "error":
            return self._error(ev[1], ev[2])
        reasoning, content, leftover, tool_calls, finish, usage, timings = \
            self._result(job, split, ev[1], ev[2])
        message = {"role": "assistant", "content": content + leftover}
        if reasoning:
            message["reasoning_content"] = reasoning
        if tool_calls:
            message["tool_calls"] = tool_calls
        self._json(200, {"id": f"chatcmpl-{uuid.uuid4().hex}", "object": "chat.completion",
                         "created": int(time.time()), "model": MODEL_ALIAS,
                         "choices": [{"index": 0, "message": message,
                                      "finish_reason": finish}],
                         "usage": usage, "timings": timings})

    def _stream(self, job: Job):
        cid, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True

        def send(obj: dict):
            self.wfile.write(b"data: " + json.dumps(obj).encode() + b"\n\n")
            self.wfile.flush()

        def chunk(delta: dict, finish=None, **extra):
            send({"id": cid, "object": "chat.completion.chunk", "created": created,
                  "model": MODEL_ALIAS,
                  "choices": [{"index": 0, "delta": delta, "finish_reason": finish}], **extra})

        def idle():   # keeps the connection alive through a long prefill, and notices a hangup
            self.wfile.write(b": waiting\n\n")
            self.wfile.flush()

        chunk({"role": "assistant", "content": ""})
        ev, split = self._consume(job, chunk, idle)
        if ev[0] == "error":
            send({"error": {"message": ev[2], "type": "server_error", "code": ev[1]}})
        else:
            _, _, leftover, tool_calls, finish, usage, timings = \
                self._result(job, split, ev[1], ev[2])
            if leftover:
                chunk({"content": leftover})
            if tool_calls:
                chunk({"tool_calls": [{"index": i, **tc} for i, tc in enumerate(tool_calls)]})
            chunk({}, finish, usage=usage, timings=timings)
            if (job.body.get("stream_options") or {}).get("include_usage"):
                send({"id": cid, "object": "chat.completion.chunk", "created": created,
                      "model": MODEL_ALIAS, "choices": [], "usage": usage})
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()


def main(argv=None) -> int:
    global WORKER
    ap = argparse.ArgumentParser(
        prog="chad-serve",
        description="Serve chad's MLX engine as an OpenAI-compatible chat completions API.")
    ap.add_argument("--host", default="127.0.0.1",
                    help="address to bind (default: 127.0.0.1; there is no authentication)")
    ap.add_argument("--port", type=int, default=8081, help="port to listen on (default: 8081)")
    ap.add_argument("--model", default=None,
                    help="HF repo id, local model dir or GGUF, as `chad --model` takes it "
                         f"(default: CHAD_MODEL, else {DEFAULT_MODEL})")
    args = ap.parse_args(argv)
    WORKER = Worker(args.model or config.env_str("CHAD_MODEL") or DEFAULT_MODEL)
    WORKER.start()
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    server.daemon_threads = True
    log(f"listening on http://{args.host}:{args.port}/v1 (model id: {MODEL_ALIAS})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
