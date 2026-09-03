"""A forcing proxy: every arm samples identically, by construction rather than by hope.

WHY THIS EXISTS
---------------
The first run of this grid was confounded, and not in a way any amount of care in
`run.py` would have caught. Polling llama-server's `/slots` mid-run showed the four arms
were NOT sampling alike:

    pi+llama        temp 1.0  top_k 20  top_p 0.95  min_p 0.05
    opencode+llama  temp 1.0  top_k 20  top_p 1.00  min_p 0.05     <- top_p
    chad+llama      temp 1.0  top_k 20  top_p 0.95  min_p 0.05     <- min_p
    chad+mlx        temp 1.0  top_k 20  top_p 0.95  min_p 0.00     <- min_p

Two independent leaks. opencode asks for a different nucleus than the other two. And
`min_p` split the *engine* comparison down the middle: chad's `build_completion_body`
sends a knob only when it is armed (0 = OFF is chad's convention), so `min_p=0.0` is
omitted from the request — and an omitted knob is not "off", it is "whatever the server
defaults to", which on llama.cpp is 0.05. The in-process MLX engine applies the 0.0
directly. Same harness, same config, two different tails, and the arm with the fatter
tail generated 2.16x the tokens.

Qwen ships `generation_config.json` with temperature 1.0 / top_k 20 / top_p 0.95 and says
NOTHING about min_p. So there is no authoritative value to pick: 0.00 is what
transformers and MLX do by default, 0.05 is what llama.cpp does. The disagreement is
between the ecosystems, not between a right and a wrong answer.

WHY A PROXY AND NOT CONFIG
--------------------------
Because "configure all three harnesses to match" is not verifiable. pi and opencode
expose whatever knobs their authors chose to expose; neither documents min_p, and an
unset knob silently becomes a server default again. Rewriting the request body in flight
is the only mechanism that does not depend on a harness cooperating:

    harness -> :8080 (this proxy, overwrites sampling fields) -> :8081 (llama-server)

Nothing in `src/chad/` is touched, so the tree under test is the shipping tree and no one
can argue the benchmark tuned chad for the benchmark. Whatever a harness asks for is
recorded verbatim in `sampler_audit.jsonl` before it is overwritten -- so "what each
agent asks for" becomes data instead of a thing we had to trust.

WHAT IS FORCED
--------------
`CANON` below, onto every generation request, on both endpoint families:
`/completion` (chad's raw token-id path) and `/v1/chat/completions` (pi, opencode).
Values are written EXPLICITLY, including zeros -- writing `min_p: 0.0` is the whole
point, since omitting it is what caused the divergence.

Sampler ORDER is left alone deliberately. llama.cpp filters then scales by temp; chad
scales then filters. At temp 1.0 the scale is the identity and the two orders agree, so
at the model card's temperature this is a non-issue. At any other temperature it is NOT,
and `--samplers` would have to be pinned here as well; `main()` refuses to start at
temp != 1.0 without an explicit acknowledgement, rather than letting that silently ride.

WHAT IS RECORDED PER TURN
-------------------------
The proxy is also the one place every harness's traffic passes through, so it is where
"how does this harness treat a local engine" can be measured without trusting any of
them. For every generation request one line goes to `_runs/turns.jsonl`:

  t_arrive / t_first_byte / t_last_byte   wall clock at the proxy: time-to-first-token as
                                          the harness itself experiences it
  n_messages, sys_sha, tools_sha          shape of the request and content hashes of the
                                          system message and the tool list -- a hash that
                                          changes between turns is a prefix the server
                                          cannot reuse, and it names WHY the cache missed
  prompt_n, cache_n, predicted_n,         llama-server's own `timings` object from the
  prompt_ms, predicted_ms                 response: tokens evaluated, tokens served from
                                          the prefix cache, tokens produced, and the time
                                          the model was actually busy on each

`timings_per_token` is forced on so the streamed form carries `timings` too; the tail of
the stream is parsed AFTER it ends, never per chunk, so nothing here adds latency to the
bytes a harness is waiting on.

`--bodies DIR` additionally writes every generation request body, as the harness sent it
and before the sampler rewrite, to `DIR/<arm>/<task>-<rep>/<n>.json`. That is how a
cache miss that the hashes cannot explain gets diagnosed: `body_diff.py` on two
consecutive files shows exactly where the byte-stable prefix ends. Off by default — a
body is the harness's whole prompt — and `_runs/bodies/` is never committed.

Run:
    python benchmarks/matrix/sampler_proxy.py --listen 8080 --upstream 8081
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "_runs")

# The one sampler every arm gets. temperature/top_k/top_p are Qwen's own shipped
# generation_config.json; min_p is the parameter Qwen does not specify, pinned here
# explicitly so neither engine gets to fall back to its ecosystem's default.
CANON = {
    "temperature": 1.0,
    "top_k": 20,
    "top_p": 0.95,
    "min_p": 0.05,
    "presence_penalty": 0.0,
    "frequency_penalty": 0.0,
    "repeat_penalty": 1.0,
}

# Fields we overwrite on the OpenAI-shaped endpoint. llama-server accepts top_k/min_p
# there as extensions, so the same canonical set reaches the sampler either way.
OAI_FIELDS = ("temperature", "top_k", "top_p", "min_p",
              "presence_penalty", "frequency_penalty", "repeat_penalty")
# Everything llama.cpp's sampler chain reads that is NOT in CANON. Left in place they
# would be a second uncontrolled axis, so they are neutralized rather than ignored.
NEUTRALIZE = {"typical_p": 1.0, "tfs_z": 1.0, "top_n_sigma": -1.0,
              "xtc_probability": 0.0, "dry_multiplier": 0.0, "mirostat": 0}

GEN_PATHS = ("/completion", "/completions", "/v1/completions",
             "/chat/completions", "/v1/chat/completions", "/infill",
             "/responses", "/v1/responses")

# How much of a streamed response to keep for the after-the-fact `timings` parse. The
# final chunk is a few hundred bytes; 64 KB is generous without ever being a buffer.
TAIL_CAP = 64 * 1024

_lock = threading.Lock()
_state = {"arm": "?", "task": "?", "rep": 0}
_audit: list = []
_bodies: dict = {"dir": None, "n": 0}


def _audit_path() -> str:
    return os.path.join(RUNS, "sampler_audit.jsonl")


def _turns_path() -> str:
    return os.path.join(RUNS, "turns.jsonl")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()[:16]


def _request_stats(doc: dict, raw: bytes, path: str) -> dict:
    """The shape of one request, with the system prompt and tool list reduced to hashes.

    Three wire shapes come through here: OpenAI chat (`messages` + `tools`), OpenAI
    Responses (`input` + `instructions` + `tools`), and llama.cpp's raw `/completion`
    (`prompt` as a token-id array -- chad's path). The raw form has no messages to hash,
    so its prefix stability shows up only in `cache_n`; the row says so via `shape`."""
    msgs = doc.get("messages")
    if msgs is None and isinstance(doc.get("input"), list):
        msgs = doc["input"]
    sys_txt = doc.get("instructions") if isinstance(doc.get("instructions"), str) else None
    if sys_txt is None and isinstance(msgs, list):
        for m in msgs:
            if isinstance(m, dict) and m.get("role") in ("system", "developer"):
                c = m.get("content")
                if isinstance(c, list):   # content-part form
                    c = "".join(p.get("text", "") for p in c if isinstance(p, dict))
                sys_txt = c if isinstance(c, str) else json.dumps(c, sort_keys=True)
                break
    tools = doc.get("tools")
    prompt = doc.get("prompt")
    shape = "chat" if isinstance(msgs, list) else "completion"
    return {
        "shape": shape,
        "n_messages": len(msgs) if isinstance(msgs, list) else None,
        "sys_sha": _sha(sys_txt) if isinstance(sys_txt, str) else None,
        "sys_chars": len(sys_txt) if isinstance(sys_txt, str) else None,
        "n_tools": len(tools) if isinstance(tools, list) else None,
        "tools_sha": _sha(json.dumps(tools, sort_keys=True)) if tools is not None else None,
        "prompt_len": len(prompt) if isinstance(prompt, list) else None,
        "body_chars": len(raw),
        "stream": bool(doc.get("stream")),
    }


def _parse_sse_tail(tail: bytes) -> tuple:
    """`timings` and `usage` out of the last few `data:` lines of a finished stream.

    Walks backwards so the FINAL chunk's numbers win (with `timings_per_token` every
    chunk carries a timings object; only the last one is the whole turn). A partial line
    at the front of the tail is skipped by construction."""
    timings = usage = None
    for line in reversed(tail.decode("utf-8", "ignore").splitlines()):
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            obj = json.loads(payload)
        except ValueError:
            continue
        if not isinstance(obj, dict):
            continue
        # Responses API wraps the terminal object: {"type": "response.completed",
        # "response": {..., "usage": {...}}}
        inner = obj.get("response") if isinstance(obj.get("response"), dict) else obj
        if timings is None and isinstance(inner.get("timings"), dict):
            timings = inner["timings"]
        if usage is None and isinstance(inner.get("usage"), dict):
            usage = inner["usage"]
        if timings is not None and usage is not None:
            break
    return timings, usage


def _record_turn(rec: dict) -> None:
    """One line per generation request. Telemetry: a failure here is logged and
    swallowed, never allowed to take a harness's request down with it."""
    try:
        with _lock:
            os.makedirs(RUNS, exist_ok=True)
            with open(_turns_path(), "a") as f:
                f.write(json.dumps(rec) + "\n")
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[proxy] turn record failed: {e}\n")


def _force(body: dict, path: str) -> tuple:
    """Overwrite the sampling fields in `body`. Returns (asked, forced) for the audit.

    `asked` records only the sampling keys the client actually sent -- a key that is
    ABSENT is recorded as absent, not as a default, because absence is precisely the bug
    this proxy exists to neutralize."""
    keys = set(OAI_FIELDS) | set(NEUTRALIZE)
    asked = {k: body[k] for k in keys if k in body}
    for k, v in CANON.items():
        body[k] = v
    for k, v in NEUTRALIZE.items():
        body[k] = v
    # llama.cpp's `samplers` array would let a client reorder the chain underneath us.
    body.pop("samplers", None)
    # Not a sampling knob: this makes llama-server put its `timings` object on the
    # streamed chunks as well as the final JSON, which is where the per-turn record
    # reads `cache_n` / `prompt_n` / `predicted_n` from. Harmless to a client that
    # ignores unknown keys, and every client here does.
    body["timings_per_token"] = True
    if body.get("stream") and path.endswith("chat/completions"):
        opts = body.get("stream_options")
        body["stream_options"] = {**(opts if isinstance(opts, dict) else {}),
                                  "include_usage": True}
    forced = {k: body[k] for k in sorted(keys)}
    return asked, forced


def _save_body(label: dict, doc: dict) -> str | None:
    """One request body to disk, numbered in arrival order across the whole run so two
    consecutive files are two consecutive requests whatever arm or task they belong to.
    Returns the path relative to the bodies dir, or None when capture is off."""
    if not _bodies["dir"]:
        return None
    with _lock:
        _bodies["n"] += 1
        n = _bodies["n"]
    sub = os.path.join(str(label.get("arm")), f"{label.get('task')}-{label.get('rep')}")
    rel = os.path.join(sub, f"{n:05d}.json")
    try:
        os.makedirs(os.path.join(_bodies["dir"], sub), exist_ok=True)
        with open(os.path.join(_bodies["dir"], rel), "w") as f:
            json.dump(doc, f)
    except Exception as e:  # noqa: BLE001 — capture is diagnostics, never the request
        sys.stderr.write(f"[proxy] body capture failed: {e}\n")
        return None
    return rel


def _record(path: str, asked: dict, forced: dict) -> None:
    with _lock:
        rec = {"arm": _state["arm"], "task": _state["task"], "rep": _state["rep"],
               "path": path, "asked": asked, "forced": forced}
        _audit.append(rec)
        os.makedirs(RUNS, exist_ok=True)
        with open(_audit_path(), "a") as f:
            f.write(json.dumps(rec) + "\n")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    upstream_port = 8081

    def log_message(self, *a):  # noqa: A003 - silence per-request stderr spam
        pass

    # -- control plane (never forwarded) ---------------------------------------
    def _control(self) -> bool:
        if self.path == "/__arm" and self.command == "POST":
            n = int(self.headers.get("Content-Length") or 0)
            with _lock:
                _state.update(json.loads(self.rfile.read(n) or b"{}"))
            return self._json({"ok": True, **_state})
        if self.path == "/__audit":
            with _lock:
                # The distinct canonical sets actually applied. The runner asserts
                # this has exactly one element before it will print a table.
                seen = sorted({json.dumps(r["forced"], sort_keys=True) for r in _audit})
                return self._json({"requests": len(_audit),
                                   "distinct_forced": [json.loads(x) for x in seen],
                                   "arms": sorted({r["arm"] for r in _audit})})
        return False

    def _json(self, obj) -> bool:
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)
        return True

    # -- proxy -----------------------------------------------------------------
    def _pass(self, method: str) -> None:
        if self._control():
            return
        t_arrive = time.time()
        n = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(n) if n else b""
        base = self.path.split("?")[0]
        # The arm label is snapshotted NOW, not when the row is written: a slow stream
        # from arm N can still be draining when the runner relabels the proxy for arm
        # N+1, and a row written then would be billed to the wrong harness.
        with _lock:
            label = dict(_state)
        turn = None
        if body and any(base.endswith(p) for p in GEN_PATHS):
            try:
                doc = json.loads(body)
            except ValueError:
                doc = None
            if isinstance(doc, dict):
                body_rel = _save_body(label, doc)
                asked, forced = _force(doc, base)
                _record(base, asked, forced)
                turn = {**label, "path": base, "t_arrive": t_arrive,
                        **_request_stats(doc, body, base)}
                if body_rel:
                    turn["body"] = body_rel
                body = json.dumps(doc).encode()

        conn = http.client.HTTPConnection("127.0.0.1", self.upstream_port, timeout=3600)
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in ("content-length", "host", "accept-encoding")}
        headers["Content-Length"] = str(len(body))
        headers["Host"] = f"127.0.0.1:{self.upstream_port}"
        try:
            conn.request(method, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except Exception as e:  # noqa: BLE001
            self.send_response(502)
            self.send_header("Content-Length", "0")
            self.end_headers()
            sys.stderr.write(f"[proxy] upstream error: {e}\n")
            if turn is not None:
                _record_turn({**turn, "status": 502, "error": str(e)[:200]})
            return

        self.send_response(resp.status)
        hop = ("connection", "transfer-encoding", "content-length")
        for k, v in resp.getheaders():
            if k.lower() not in hop:
                self.send_header(k, v)
        t_first = t_last = None
        timings = usage = None
        disconnected = False
        # Streamed SSE has no Content-Length; chunk it back so tokens arrive as they
        # are produced. A harness that buffers a whole turn would still work, but its
        # wall clock would no longer be the model's.
        streaming = "text/event-stream" in (resp.getheader("Content-Type") or "")
        try:
            if streaming:
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                tail = bytearray()
                while True:
                    # read1, NOT read: `read(n)` blocks until n bytes are available or
                    # the stream ends, which buffers a whole turn and hands every arm
                    # the proxy's wall clock instead of the model's. Measured: 5 chunks
                    # 300 ms apart all arrived together at 1.53 s.
                    buf = resp.read1(4096)
                    if not buf:
                        break
                    if t_first is None:
                        t_first = time.time()
                    self.wfile.write(b"%x\r\n%s\r\n" % (len(buf), buf))
                    self.wfile.flush()
                    tail += buf
                    if len(tail) > TAIL_CAP:
                        del tail[:len(tail) - TAIL_CAP]
                t_last = time.time()
                self.wfile.write(b"0\r\n\r\n")
                timings, usage = _parse_sse_tail(bytes(tail))
            else:
                payload = resp.read()
                t_first = t_last = time.time()
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                try:
                    obj = json.loads(payload)
                    if isinstance(obj, dict):
                        timings = obj.get("timings") if isinstance(obj.get("timings"), dict) else None
                        usage = obj.get("usage") if isinstance(obj.get("usage"), dict) else None
                except ValueError:
                    pass
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError) as e:
            # The harness went away mid-turn -- typically the runner's timeout killing
            # its process group. That is exactly the turn worth recording, so the row is
            # still written with whatever was seen, and says so.
            disconnected = True
            if t_last is None:
                t_last = time.time()
            sys.stderr.write(f"[proxy] client disconnected: {e}\n")
        finally:
            conn.close()
        if turn is not None:
            _record_turn({**turn, "status": resp.status,
                          "t_first_byte": t_first, "t_last_byte": t_last,
                          "ttft_s": round(t_first - t_arrive, 3) if t_first else None,
                          "timings": timings, "usage": usage,
                          "client_disconnected": disconnected})

    def do_GET(self):     # noqa: N802
        self._pass("GET")

    def do_POST(self):    # noqa: N802
        self._pass("POST")

    def do_DELETE(self):  # noqa: N802
        self._pass("DELETE")


def main() -> None:
    global RUNS
    ap = argparse.ArgumentParser()
    ap.add_argument("--listen", type=int, default=8080)
    ap.add_argument("--upstream", type=int, default=8081)
    ap.add_argument("--runs", default=RUNS, metavar="DIR",
                    help="where turns.jsonl and sampler_audit.jsonl are appended")
    ap.add_argument("--bodies", default=None, metavar="DIR",
                    help="write every generation request body under DIR (diagnostic; "
                         "see the module docstring)")
    ap.add_argument("--allow-temp-drift", action="store_true",
                    help="permit a canonical temperature != 1.0 (requires pinning "
                         "llama.cpp's sampler ORDER too -- see the module docstring)")
    a = ap.parse_args()
    if CANON["temperature"] != 1.0 and not a.allow_temp_drift:
        sys.exit("CANON temperature != 1.0: llama.cpp filters before scaling and chad "
                 "scales before filtering, so top_p/min_p would select different token "
                 "sets per engine. Pin `samplers` here first, then pass "
                 "--allow-temp-drift.")
    Handler.upstream_port = a.upstream
    RUNS = os.path.abspath(a.runs)
    if a.bodies:
        _bodies["dir"] = a.bodies
        os.makedirs(a.bodies, exist_ok=True)
    os.makedirs(RUNS, exist_ok=True)
    srv = ThreadingHTTPServer(("127.0.0.1", a.listen), Handler)
    srv.daemon_threads = True
    print(f"sampler proxy :{a.listen} -> :{a.upstream}  forcing {CANON}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
