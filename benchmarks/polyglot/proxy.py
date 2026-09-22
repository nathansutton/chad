"""A recording proxy in front of llama-server: the one place a CLI arm's sampler is
forced, and the one place its cost is read without asking the harness.

    harness -> 127.0.0.1:<port> (this, a thread in run.py) -> llama-server

FORCED
------
Every generation request gets the shipped sampler — chad's thinking preset — written
into its body explicitly, zeros included, and every other knob llama.cpp's sampler chain
reads held at identity. Explicit matters: an omitted knob is not "off" but the server's
default, and llama-server's `min_p` default is 0.05 against chad's 0.0 — which is how the
matrix grid's first night compared two samplers instead of two harnesses. What the
harness asked for is recorded beside what it got. `/completion` echoes the sampler it
applied, so those requests are checked one by one; the chat and Responses endpoints echo
nothing, so `audit()` checks them once per block, before the first trial, through
`/slots`.

RECORDED
--------
One line per generation request into the log of the trial in flight (`route()`): when it
arrived and when its first and last bytes went back, the request as the harness sent it,
what came back — text, reasoning, tool calls, why it stopped — and llama-server's own
`timings`: tokens prefilled (`prompt_n`), served from the prefix cache (`cache_n`),
generated (`predicted_n`), and the milliseconds each took. Every endpoint ends its reply
with that object, and no harness reports the same numbers the same way, if at all.
`proxy_atif.py` turns a trial's log into an ATIF trajectory.

Bytes pass through as they arrive and are parsed only after the reply ends, so the proxy
adds no latency a harness can see. A client that hangs up mid-stream has its upstream
connection closed too, which is what makes llama-server abort the slot.
"""
from __future__ import annotations

import http.client
import json
import os
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from catalog import JsonValue, is_array, is_number, is_object, is_text

GEN_PATHS = frozenset({"/completion", "/completions", "/v1/completions", "/chat/completions",
                       "/v1/chat/completions", "/responses", "/v1/responses", "/infill"})
# Request headers that describe one hop, not the request; Content-Length is recomputed
# because a forced body is a different length.
_HOP = frozenset({"connection", "keep-alive", "proxy-authenticate", "proxy-authorization",
                  "te", "trailer", "trailers", "transfer-encoding", "upgrade", "host",
                  "content-length"})
# llama.cpp's sampler chain beyond the preset, at the values that make each a no-op.
_IDENTITY: dict[str, JsonValue] = {
    "frequency_penalty": 0.0, "repeat_penalty": 1.0, "dry_multiplier": 0.0,
    "xtc_probability": 0.0, "typical_p": 1.0, "top_n_sigma": -1.0, "mirostat": 0}
_TOLERANCE = 1e-6                # the server echoes float32: 0.95 comes back 0.9499999


def shipped_sampler() -> dict[str, JsonValue]:
    """chad's thinking preset in llama.cpp's field names, then the identity knobs.
    Read from chad itself, so the forced sampler cannot drift from the shipped one."""
    from chad.cli import THINKING_SAMPLER
    preset = THINKING_SAMPLER
    return {"temperature": preset["temp"], "top_k": int(preset["top_k"]),
            "top_p": preset["top_p"], "min_p": preset["min_p"],
            "presence_penalty": preset["presence_penalty"], **_IDENTITY}


@dataclass(frozen=True)
class ToolCall:
    id: str
    name: str
    arguments: str               # as the model wrote it: JSON text, possibly malformed


@dataclass
class Reply:
    """A generation's outcome, whichever endpoint and whether or not it streamed."""

    content: str = ""
    reasoning: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    finish: str = ""
    timings: dict[str, JsonValue] = field(default_factory=dict)
    settings: dict[str, JsonValue] = field(default_factory=dict)   # applied, if echoed


def _text(value: JsonValue) -> str:
    return value if is_text(value) else ""


def _docs(raw: str) -> list[Mapping[str, JsonValue]]:
    """The JSON objects of a reply: each `data:` line of a stream, or the one body."""
    stripped = raw.lstrip()
    lines = ([ln[5:].strip() for ln in raw.splitlines() if ln.startswith("data:")]
             if stripped.startswith(("data:", "event:")) else [stripped])
    docs = []
    for line in lines:
        if not line or line == "[DONE]":
            continue
        try:
            doc: JsonValue = json.loads(line)
        except ValueError:
            continue
        if is_object(doc):
            docs.append(doc)
    return docs


class _Calls:
    """Tool calls assembled from fragments: chat streams them by index, argument text
    a few characters at a time."""

    def __init__(self) -> None:
        self.parts: dict[str, list[str]] = {}

    def add(self, key: str, call_id: str, name: str, arguments: str) -> None:
        part = self.parts.setdefault(key, ["", "", ""])
        part[0] = part[0] or call_id
        part[1] = part[1] or name
        part[2] += arguments

    def done(self) -> list[ToolCall]:
        return [ToolCall(*p) for p in self.parts.values()]


def _fold_chat(choice: Mapping[str, JsonValue], reply: Reply, calls: _Calls) -> None:
    body = choice.get("delta") if is_object(choice.get("delta")) else choice.get("message")
    if is_object(body):
        reply.content += _text(body.get("content"))
        reply.reasoning += _text(body.get("reasoning_content"))
        fragments = body.get("tool_calls")
        for n, fragment in enumerate(fragments if is_array(fragments) else ()):
            if not is_object(fragment):
                continue
            index = fragment.get("index")
            function = fragment.get("function")
            fn = function if is_object(function) else {}
            calls.add(str(index if is_number(index) else n), _text(fragment.get("id")),
                      _text(fn.get("name")), _text(fn.get("arguments")))
    reply.finish = _text(choice.get("finish_reason")) or reply.finish


def _fold_responses_item(item: Mapping[str, JsonValue], reply: Reply, calls: _Calls) -> None:
    kind = item.get("type")
    if kind == "function_call":
        calls.add(_text(item.get("call_id")) or _text(item.get("id")), _text(item.get("call_id")),
                  _text(item.get("name")), _text(item.get("arguments")))
    parts = item.get("content") if kind in ("message", "reasoning") else None
    for part in parts if is_array(parts) else ():
        if is_object(part):
            if kind == "message":
                reply.content += _text(part.get("text"))
            else:
                reply.reasoning += _text(part.get("text"))


def parse_reply(raw: bytes) -> Reply:
    """Fold a reply's objects into one `Reply`. Chat streams deltas, `/completion` streams
    text chunks and ends on `stop`, Responses streams typed events and repeats the whole
    output in `response.completed` (read only when nothing streamed, or it counts twice)."""
    reply, calls = Reply(), _Calls()
    streamed_items = False
    for doc in _docs(raw.decode("utf-8", "replace")):
        timings, settings = doc.get("timings"), doc.get("generation_settings")
        if is_object(timings):
            reply.timings = dict(timings)
        if is_object(settings):
            reply.settings = dict(settings)
        choices = doc.get("choices")
        kind = _text(doc.get("type"))
        if is_array(choices):
            for choice in choices:
                if is_object(choice):
                    _fold_chat(choice, reply, calls)
        elif kind == "response.output_text.delta":
            reply.content += _text(doc.get("delta"))
        elif kind in ("response.reasoning_text.delta", "response.reasoning_summary_text.delta"):
            reply.reasoning += _text(doc.get("delta"))
        elif kind == "response.output_item.done":
            item = doc.get("item")
            if is_object(item) and item.get("type") == "function_call":
                streamed_items = True
                _fold_responses_item(item, reply, calls)
        elif kind == "response.completed" or doc.get("object") == "response":
            response = doc.get("response") if kind else doc
            if is_object(response):
                reply.finish = _text(response.get("status")) or reply.finish
                output = response.get("output")
                if not (streamed_items or reply.content or reply.reasoning):
                    for item in output if is_array(output) else ():
                        if is_object(item):
                            _fold_responses_item(item, reply, calls)
        elif "content" in doc and not kind:
            reply.content += _text(doc.get("content"))
            if doc.get("stop") is True:
                reply.finish = _text(doc.get("stop_type")) or "stop"
    reply.tool_calls = calls.done()
    return reply


def sampler_check(forced: Mapping[str, JsonValue], applied: Mapping[str, JsonValue]) -> str:
    """`ok` when the server says it applied every forced value, `unechoed` when it says
    nothing (chat, Responses), else the first field it did not apply."""
    if not applied:
        return "unechoed"
    for key, want in forced.items():
        got = applied.get(key)
        if key not in applied:
            continue                   # an older build does not echo every knob
        if not (is_number(got) and is_number(want) and abs(float(got) - float(want)) <= _TOLERANCE):
            return f"mismatch: {key} {got!r} != {want!r}"
    return "ok"


def reply_record(reply: Reply) -> dict[str, JsonValue]:
    return {"content": reply.content, "reasoning": reply.reasoning,
            "tool_calls": [{"id": c.id, "name": c.name, "arguments": c.arguments}
                           for c in reply.tool_calls],
            "finish": reply.finish, "timings": reply.timings}


class Proxy:
    """One block's proxy: a thread serving 127.0.0.1:`port` (0 picks one), forwarding
    to the llama-server at `upstream`."""

    def __init__(self, upstream: str, forced: Mapping[str, JsonValue], stray: str,
                 port: int = 0):
        split = urllib.parse.urlsplit(upstream)
        self.upstream = upstream.rstrip("/")
        self.host, self.port = split.hostname or "127.0.0.1", split.port or 80
        self.forced = dict(forced)
        self.stray = stray             # where a request outside any trial is logged
        self._log = ""
        self._lock = threading.Lock()
        self._server = ThreadingHTTPServer(("127.0.0.1", port), _handler(self))
        self._server.daemon_threads = True
        self.origin = f"http://127.0.0.1:{self._server.server_address[1]}"

    def start(self) -> None:
        threading.Thread(target=self._server.serve_forever, daemon=True).start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    def route(self, log: str) -> None:
        """Send later records to `log` (a trial's), or with "" to the stray log."""
        with self._lock:
            self._log = log

    def record(self, entry: Mapping[str, JsonValue]) -> None:
        with self._lock:
            path = self._log or self.stray
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry) + "\n")

    def force(self, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
        return {**body, **self.forced}


def _handler(proxy: Proxy) -> type[BaseHTTPRequestHandler]:
    class Relay(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: object) -> None:
            pass                       # every generation request is logged as a record

        def _body(self) -> bytes:
            """The request body, whether sized by Content-Length or sent chunked."""
            if "chunked" not in (self.headers.get("Transfer-Encoding") or "").lower():
                return self.rfile.read(int(self.headers.get("Content-Length") or 0))
            body = bytearray()
            while True:
                size = int(self.rfile.readline().split(b";")[0].strip() or b"0", 16)
                if not size:
                    self.rfile.readline()
                    return bytes(body)
                body += self.rfile.read(size)
                self.rfile.readline()

        def do_GET(self) -> None:
            self._relay()

        def do_POST(self) -> None:
            self._relay()

        def _relay(self) -> None:
            t0 = time.time()
            body = self._body()
            path = urllib.parse.urlsplit(self.path).path
            sent: JsonValue = None
            if self.command == "POST" and path in GEN_PATHS:
                try:
                    sent = json.loads(body or b"null")
                except ValueError:
                    sent = None
            generating = is_object(sent)
            if is_object(sent):
                body = json.dumps(proxy.force(sent)).encode()
            conn = http.client.HTTPConnection(proxy.host, proxy.port, timeout=None)
            headers = {k: v for k, v in self.headers.items() if k.lower() not in _HOP}
            headers["Content-Length"] = str(len(body))
            try:
                conn.request(self.command, self.path, body=body, headers=headers)
                upstream = conn.getresponse()
            except OSError as e:
                self.send_error(502, f"llama-server unreachable: {e}")
                return
            self.send_response_only(upstream.status, upstream.reason)
            chunked = "chunked" in (upstream.getheader("Transfer-Encoding") or "").lower()
            for key, value in upstream.getheaders():
                if key.lower() not in _HOP or (key.lower() == "content-length" and not chunked):
                    self.send_header(key, value)
            self.send_header("Connection", "close")
            self.end_headers()
            kept, first, gone = bytearray(), 0.0, False
            while True:
                chunk = upstream.read1(65536)
                if not chunk:
                    break
                if generating:
                    kept += chunk
                try:
                    self.wfile.write(chunk)
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    gone = True
                    break
                first = first or time.time()
            conn.close()
            self.close_connection = True
            if is_object(sent):
                reply = parse_reply(bytes(kept))
                proxy.record({
                    "t": round(t0, 3), "path": path, "status": upstream.status,
                    "ttfb_s": round(first - t0, 3) if first else None,
                    "total_s": round(time.time() - t0, 3), "client_gone": gone,
                    "asked": {k: sent[k] for k in proxy.forced if k in sent},
                    "sampler": sampler_check(proxy.forced, reply.settings),
                    "request": sent, "reply": reply_record(reply)})

    return Relay


def audit(proxy: Proxy, model: str) -> dict[str, str]:
    """Before a block's first trial: one one-token request per endpoint family through
    the proxy, then the parameters of the slot that served it, from `/slots`. `ok` per
    family when every forced value is what the server applied; anything else is a
    reason the block's rows cannot claim a forced sampler for that family."""
    # Each probe's length limit is distinct, and `/slots` reports it back as `n_predict`:
    # that, not the order of task ids, is what names the slot that served it.
    probes: dict[str, tuple[str, int, dict[str, JsonValue]]] = {
        "chat": ("/v1/chat/completions", 2, {"model": model, "max_tokens": 2,
                                             "messages": [{"role": "user", "content": "hi"}]}),
        "responses": ("/v1/responses", 3, {"model": model, "max_output_tokens": 3,
                                           "input": [{"role": "user", "content": "hi"}]}),
        "completion": ("/completion", 4, {"prompt": "hi", "n_predict": 4}),
    }
    verdicts = {}
    for family, (path, marker, body) in probes.items():
        req = urllib.request.Request(proxy.origin + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                r.read()
            with urllib.request.urlopen(proxy.upstream + "/slots", timeout=30) as r:
                slots: JsonValue = json.loads(r.read())
        except (OSError, ValueError) as e:
            verdicts[family] = f"unverified: {e}"
            continue
        verdicts[family] = _slot_verdict(proxy.forced, slots, marker)
    return verdicts


def erase_slots(upstream: str) -> int:
    """Empty every slot's prompt cache, so the block that follows starts cold whatever the
    server served before it. The number erased, or -1 if the server would not say."""
    try:
        with urllib.request.urlopen(upstream + "/slots", timeout=30) as r:
            slots: JsonValue = json.loads(r.read())
        erased = 0
        for slot in slots if is_array(slots) else ():
            slot_id = slot.get("id") if is_object(slot) else None
            if is_number(slot_id):
                req = urllib.request.Request(f"{upstream}/slots/{int(slot_id)}?action=erase",
                                             data=b"{}", method="POST",
                                             headers={"Content-Type": "application/json"})
                with urllib.request.urlopen(req, timeout=30) as r:
                    r.read()
                erased += 1
        return erased
    except (OSError, ValueError):
        return -1


def _slot_verdict(forced: Mapping[str, JsonValue], slots: JsonValue, marker: int = 0) -> str:
    """The sampler of the slot that served a probe: the one reporting the probe's
    `n_predict` marker, else the one with the newest task (nothing else is in flight
    before a block starts)."""
    served: Mapping[str, JsonValue] = {}
    best = -2.0
    for slot in slots if is_array(slots) else ():
        if not is_object(slot):
            continue
        params = slot.get("params")
        task = slot.get("id_task")
        rank = float(task) if is_number(task) else -1.0
        if marker and is_object(params) and params.get("n_predict") == marker:
            rank += 1e12               # the marker outranks any task id
        if rank > best:
            best, served = rank, slot
    params = served.get("params")
    if not is_object(params):
        return "unverified: /slots names no parameters"
    return sampler_check(forced, params)
