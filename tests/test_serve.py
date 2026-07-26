"""Tests for `chad serve` — the MLX engine behind llama.cpp's /completion protocol.

No model, no weights, no MLX: the engine is a fake that records what it was asked and
replays a canned generation, and `stream_completion` writes through an injected sink
rather than a socket. The end-to-end tests bind a real loopback HTTP server (fast — no
model to load) and drive it with the REAL client, `CompletionEngine`, which is the only
way to prove the two halves actually agree on the wire.
"""

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from chad import serve
from chad.base_engine import GenStats
from chad.completion_engine import CompletionEngine


class FakeEngine:
    """Stands in for `engine.Engine`: same surface the server touches, plus a log of
    what happened so tests can assert on cache bookkeeping."""

    def __init__(self, out="hello world", gen_ids=(101, 102), effective_ctx=8192):
        self.out = out
        self.cache_dir = "/tmp/kv"          # engine.Engine has these; spill_status reads them
        self.kv_bytes_per_token = 1234.0
        self.kv_cache_max_bytes = 8 * 1024**3
        self.gen_ids = list(gen_ids)
        self.effective_ctx = effective_ctx
        self.temp = 0.0
        self.min_p = 0.0
        self.top_p = 0.0
        self.tok = None
        self._cached_ids = []
        self.calls = []
        self.threads = set()
        self.sampler_seen = None
        self.stop_after = None      # emit N tokens then pretend the client is gone
        self.cache_reset = False    # report an OOM cache drop in the turn's stats
        self.hold = None            # threading.Event: block mid-generation on demand
        self.produced = 0           # tokens emitted so far (perf tests poll this)
        self.delay = 0.0            # per-token cost, so a turn can still be IN FLIGHT
                                    # when a test does something to the client

    def _note_thread(self):
        # MLX streams are thread-local, so EVERY engine call must land on the same
        # thread or a prefill dies with "no Stream(gpu, N) in current thread".
        self.threads.add(threading.get_ident())

    def generate(self, prompt_ids, max_tokens=2048, on_token=None, should_stop=None,
                 **kw):
        self._note_thread()
        self.calls.append(("generate", list(prompt_ids), max_tokens))
        self.sampler_seen = (self.temp, self.min_p, self.top_p)
        pieces = self.out.split(" ")
        emitted = 0
        for i, p in enumerate(pieces):
            if should_stop and should_stop():
                break
            if self.hold is not None:
                self.hold.wait(10)
            if self.delay:
                time.sleep(self.delay)
            on_token and on_token(p if i == 0 else " " + p)
            emitted += 1
            self.produced = emitted
        self._cached_ids = list(prompt_ids) + self.gen_ids
        return self.out, GenStats(prompt_tokens=7, cached_tokens=3, prefill_s=0.5,
                                  generated_tokens=emitted, gen_s=0.25,
                                  gen_ids=list(self.gen_ids),
                                  cache_reset=self.cache_reset)

    def reset(self):
        self._note_thread()
        self.calls.append(("reset",))
        self._cached_ids = []

    def push_cache(self):
        self._note_thread()
        self.calls.append(("push",))

    def pop_cache(self):
        self._note_thread()
        self.calls.append(("pop",))

    def warm_prefix(self, prefix_ids, should_stop=None):
        self._note_thread()
        self.calls.append(("warm", list(prefix_ids)))
        self._cached_ids = list(prefix_ids)
        return ("hit", len(prefix_ids))


def sink():
    """A `write` that collects lines, and a decoder for the payloads it captured."""
    lines = []

    def write(chunk):
        lines.append(chunk)

    def payloads():
        out = []
        for raw in lines:
            body = raw.strip()
            if not body.startswith("data:"):
                continue
            body = body[len("data:"):].strip()
            if body == "[DONE]":
                continue
            out.append(json.loads(body))
        return out

    return write, payloads


# --- pure helpers ---------------------------------------------------------

def test_props_advertises_ctx_and_capabilities():
    p = serve.props_payload(32768, "ornith-35b")
    # the field the client actually reads to size its window
    assert p["default_generation_settings"]["n_ctx"] == 32768
    caps = p["chad"]["capabilities"]
    assert serve.CAP_CACHE_QUARANTINE in caps and serve.CAP_WARM_PREFIX in caps


def test_sse_line_shape_matches_the_client_parser():
    from chad.completion_engine import parse_sse_chunk
    line = serve.sse({"content": "hi", "stop": False})
    assert line.endswith("\n\n")
    assert parse_sse_chunk(line.strip()) == {"content": "hi", "stop": False}


def test_parse_completion_request_takes_token_ids_verbatim():
    p = serve.parse_completion_request({"prompt": [1, 2, 3], "n_predict": 64,
                                        "temperature": 0.7, "min_p": 0.05})
    assert p["prompt_ids"] == [1, 2, 3]
    assert p["n_predict"] == 64
    assert p["temperature"] == 0.7
    assert p["min_p"] == 0.05
    assert p["top_p"] is None       # ABSENT — distinct from an explicit 0.0 (= OFF)
    assert p["cache_prompt"] is True
    # …and a knob the client explicitly disables is not the same as one it never sent
    assert serve.parse_completion_request({"prompt": [1], "top_p": 0.0})["top_p"] == 0.0


def test_parse_completion_request_rejects_junk():
    for bad in ({}, {"prompt": []}, {"prompt": [1, "x"]}, {"prompt": 5},
                {"prompt": [1], "temperature": "hot"}):
        with pytest.raises(ValueError):
            serve.parse_completion_request(bad)


def test_parse_completion_request_encodes_a_string_prompt_for_curl():
    class Tok:
        def encode(self, s, add_special_tokens=False):
            return [ord(c) for c in s]
    p = serve.parse_completion_request({"prompt": "hi"}, tok=Tok())
    assert p["prompt_ids"] == [104, 105]
    # …but without a tokenizer it is an error, not a silent empty prompt
    with pytest.raises(ValueError):
        serve.parse_completion_request({"prompt": "hi"})


def test_nonpositive_n_predict_falls_back_rather_than_generating_nothing():
    # llama.cpp spells "until the context ends" as -1; we substitute a sane budget
    assert serve.parse_completion_request({"prompt": [1], "n_predict": -1})["n_predict"] > 0


def test_timings_map_genstats_onto_the_llama_cpp_fields():
    t = serve.timings_payload(GenStats(prompt_tokens=12, cached_tokens=900,
                                       prefill_s=1.5, generated_tokens=30, gen_s=3.0))
    assert t["prompt_n"] == 12          # tokens actually prefilled, not prompt length
    assert t["prompt_ms"] == 1500.0
    assert t["predicted_n"] == 30
    assert t["predicted_ms"] == 3000.0
    assert t["cache_n"] == 900


# --- streaming ------------------------------------------------------------

def test_stream_emits_one_chunk_per_token_then_ids_and_timings():
    eng = FakeEngine(out="a b c", gen_ids=[7, 8, 9])
    write, payloads = sink()
    serve.stream_completion(eng, serve.parse_completion_request({"prompt": [1, 2]}),
                            write)
    got = payloads()
    assert [p["content"] for p in got[:-1]] == ["a", " b", " c"]
    final = got[-1]
    assert final["stop"] is True
    assert final["tokens"] == [7, 8, 9]          # what the client mirrors its cache on
    assert final["timings"]["predicted_n"] == 3


def test_stream_applies_then_restores_per_request_sampler_knobs():
    eng = FakeEngine()
    eng.temp, eng.min_p, eng.top_p = 0.0, 0.0, 0.0
    write, _ = sink()
    serve.stream_completion(eng, serve.parse_completion_request(
        {"prompt": [1], "temperature": 0.7, "min_p": 0.05, "top_p": 0.9}), write)
    assert eng.sampler_seen == (0.7, 0.05, 0.9)     # the request's knobs were live…
    assert (eng.temp, eng.min_p, eng.top_p) == (0.0, 0.0, 0.0)   # …and didn't leak


def test_cache_prompt_false_drops_the_prefix_cache_first():
    eng = FakeEngine()
    write, _ = sink()
    serve.stream_completion(eng, serve.parse_completion_request(
        {"prompt": [1, 2], "cache_prompt": False}), write)
    assert ("reset",) in eng.calls
    eng2 = FakeEngine()
    serve.stream_completion(eng2, serve.parse_completion_request({"prompt": [1, 2]}),
                            sink()[0])
    assert ("reset",) not in eng2.calls     # the default keeps the cache, obviously


def test_a_dead_client_stops_generation_and_sends_no_final_chunk():
    """The cancel path: a write that raises IS the signal, and it has to reach
    `should_stop` — otherwise the box keeps decoding a turn nobody is reading."""
    eng = FakeEngine(out="a b c d e")
    n = {"w": 0}

    def write(chunk):
        n["w"] += 1
        if n["w"] >= 2:
            raise BrokenPipeError("client hung up")

    info = serve.stream_completion(eng, serve.parse_completion_request({"prompt": [1]}),
                                   write)
    assert info["cancelled"] is True
    # generation stopped early rather than running out the full 5 tokens
    assert n["w"] == 2


# --- end to end over a real socket, driven by the real client -------------

@pytest.fixture
def live_server():
    """Bind on an ephemeral loopback port and serve until the test is done."""
    eng = FakeEngine()
    state = serve.ServerState(eng, model_id="ornith-test", quiet=True)
    srv = serve.build_server(state, "127.0.0.1", 0)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", eng, state
    finally:
        srv.shutdown()
        srv.server_close()
        state.close()


def _client(url, **kw):
    """A CompletionEngine wired to the live server, with the tokenizer load skipped
    (nothing here needs to render a prompt — only the wire behavior is under test)."""
    c = CompletionEngine(model_id="ornith", base_url=url, **kw)
    c._absorb_props(c._fetch_props() or {})   # exactly what load() does, minus the tokenizer
    return c


def test_client_reads_context_window_and_capabilities_from_props(live_server):
    url, _, _ = live_server
    c = _client(url)
    assert c.effective_ctx == 8192      # the server's wall, not the 32768 fallback
    assert c._caps == set(serve.CAPABILITIES)


def test_round_trip_gives_the_client_exact_stats_not_estimates(live_server):
    url, eng, _ = live_server
    c = _client(url)
    text, stats = c.generate([1, 2, 3, 4, 5], max_tokens=32)
    assert text == "hello world"
    # the whole reason for this protocol: server-side timings, so nothing is guessed
    assert stats.approximate is False
    assert stats.prompt_tokens == 7 and stats.generated_tokens == 2
    # and the prompt crossed the wire as ids, never as text
    assert eng.calls[0][:1] == ("generate",) and eng.calls[0][1] == [1, 2, 3, 4, 5]


def test_round_trip_mirrors_the_server_cache_state(live_server):
    url, eng, _ = live_server
    c = _client(url)
    c.generate([1, 2, 3], max_tokens=8)
    # client mirror == prompt + generated ids == what the server's own cache holds
    assert c._cached_ids == [1, 2, 3] + eng.gen_ids == eng._cached_ids


def test_quarantine_round_trip_pushes_pops_and_restores_the_mirror(live_server):
    url, eng, _ = live_server
    c = _client(url)
    c.generate([1, 2, 3], max_tokens=8)
    main_prefix = list(c._cached_ids)
    c.push_cache()
    assert ("push",) in eng.calls
    assert c._cached_ids == []                  # sub-agent starts cold, as on the server
    c.generate([9, 9], max_tokens=8)            # the sub-agent's excursion
    c.pop_cache()
    assert ("pop",) in eng.calls
    assert c._cached_ids == main_prefix         # main transcript's prefix is back


def test_warm_prefix_round_trip(live_server):
    url, eng, _ = live_server
    c = _client(url)
    assert c.warm_prefix([1, 2, 3, 4]) == ("hit", 4)
    assert ("warm", [1, 2, 3, 4]) in eng.calls
    assert c._cached_ids == [1, 2, 3, 4]


def test_cache_calls_stay_no_ops_against_a_server_without_the_capabilities(live_server):
    """The stock-llama.cpp path: no `chad` block in /props → the client must not call
    the extension endpoints at all, and must not fail when it doesn't."""
    url, eng, _ = live_server
    c = _client(url)
    c._caps = set()                             # pretend a plain llama-server
    c.push_cache()
    c.pop_cache()
    assert c.warm_prefix([1, 2, 3]) == ("skip", 0)
    assert not [x for x in eng.calls if x[0] in ("push", "pop", "warm")]


def test_capabilities_are_handshaken_lazily_when_the_window_is_pinned(live_server):
    """A pinned window skips the /props probe at load (an operator who pinned it all
    shouldn't wait on a socket), so the extensions have to discover it themselves —
    once — the first time one is used."""
    url, eng, _ = live_server
    c = CompletionEngine(model_id="ornith", base_url=url, effective_ctx=4096)
    assert c._caps == set()                 # nothing probed yet
    c.push_cache()
    assert c._caps == set(serve.CAPABILITIES) and ("push",) in eng.calls
    assert c.effective_ctx == 4096          # the pinned window still wins
    probes = {"n": 0}
    real = c._fetch_props
    c._fetch_props = lambda: (probes.update(n=probes["n"] + 1), real())[1]
    c.pop_cache()
    assert probes["n"] == 0                 # handshake is once per process, not per call


def test_every_engine_call_lands_on_one_dedicated_thread(live_server):
    """MLX streams are thread-local. An HTTP server hands each connection to a fresh
    thread, so running the engine on the handler's thread dies mid-prefill with
    `RuntimeError: There is no Stream(gpu, N) in current thread` — which is exactly what
    happened on a real 9B eval run. A lock would NOT catch this: it serializes access
    but still runs the work wherever the request arrived. Pin the thread, not just the
    order."""
    url, eng, _ = live_server
    c = _client(url)
    for _ in range(3):                      # each is a separate connection → own handler thread
        c.generate([1, 2, 3], max_tokens=4)
    c.push_cache()
    c.pop_cache()
    c.warm_prefix([1, 2, 3])
    assert len(eng.threads) == 1, f"engine touched from {len(eng.threads)} threads"
    # …and it is emphatically not the main thread doing the HTTP work either
    assert eng.threads != {threading.get_ident()}


def test_warm_start_gate_opens_only_when_the_server_can_warm(live_server):
    """agent.py gates its disk warm-start on `engine.cache_dir` being truthy — the one
    and only consumer of that attribute. The client used to hardcode it to None, so the
    warm-start block was skipped and `warm_prefix` was never called at all: the endpoint
    existed and nothing ever reached it. This pins the gate to what the server can
    actually do."""
    url, _, _ = live_server
    c = _client(url)
    assert c.cache_dir                       # server advertises warm_prefix -> gate open
    c._caps = set()                          # a stock llama-server cannot warm for us
    assert c.cache_dir is None               # …so the gate closes again, as it must


def test_bad_request_is_a_400_with_a_message_not_a_broken_stream(live_server):
    url, _, _ = live_server
    req = urllib.request.Request(url + "/completion", data=b'{"prompt": []}',
                                 headers={"Content-Type": "application/json"},
                                 method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400
    assert "prompt" in e.value.read().decode()


def test_unknown_endpoint_is_404(live_server):
    url, _, _ = live_server
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url + "/v1/chat/completions", timeout=5)
    assert e.value.code == 404


# --- auth -----------------------------------------------------------------

@pytest.fixture
def guarded_server():
    eng = FakeEngine()
    state = serve.ServerState(eng, model_id="ornith-test", api_key="s3cret", quiet=True)
    srv = serve.build_server(state, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", eng
    finally:
        srv.shutdown()
        srv.server_close()
        state.close()


def test_api_key_gates_a_lan_bound_server(guarded_server):
    url, _ = guarded_server
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(url + "/props", timeout=5)
    assert e.value.code == 401
    # the client passes it the same way it does for a remote llama.cpp server
    c = CompletionEngine(model_id="m", base_url=url, api_key="s3cret")
    text, _ = c.generate([1, 2], max_tokens=8)
    assert text == "hello world"


def test_health_answers_while_the_engine_thread_is_busy(live_server):
    """`/health` and `/props` must answer during a generation — a poll that queued
    behind the engine would make a long turn look like a hung server."""
    url, _, state = live_server
    started, release = threading.Event(), threading.Event()

    def hog():
        started.set()
        release.wait(10)

    blocked = threading.Thread(target=lambda: state.call(hog), daemon=True)
    blocked.start()
    started.wait(5)
    try:
        with urllib.request.urlopen(url + "/health", timeout=5) as r:
            body = json.loads(r.read())
        assert body["status"] == "ok" and body["n_ctx"] == 8192
    finally:
        release.set()
        blocked.join(5)


# --- backpressure, cancellation, and the cache-state contract -------------
#
# These are behavioral-PERFORMANCE tests: not "how fast", but "what happens when the
# other end misbehaves". Every one of them corresponds to a way the server used to lose,
# and none needs a model — the fake engine streams as fast as the socket will take it.
#
#   client behavior          before                     asserted here
#   ───────────────          ──────                     ─────────────
#   hangs up (FIN/RST)       write raises → cancel ✓    still cancels
#   stops reading, stays up  ENGINE BLOCKS FOREVER ✗    engine finishes, client killed
#   reads slowly             engine ran at reader speed  engine runs ahead of reader
#   second request arrives   queued ✓                   still served, not dropped

def _post(url, body, read=True):
    """Raw socket POST so a test can choose NOT to read the response."""
    import socket
    host, port = url[len("http://"):].split(":")
    s = socket.create_connection((host, int(port)), timeout=10)
    payload = json.dumps(body).encode()
    s.sendall(b"POST /completion HTTP/1.1\r\nHost: x\r\n"
              b"Content-Type: application/json\r\nContent-Length: "
              + str(len(payload)).encode() + b"\r\n\r\n" + payload)
    if read:
        s.recv(65536)
    return s


def _wide_engine(ntok=4000, seglen=400):
    """A fake whose tokens are big enough to fill socket buffers quickly, so a
    non-reading client reaches backpressure in a test-sized amount of time."""
    return FakeEngine(out=" ".join("x" * seglen for _ in range(ntok)))


def test_a_client_that_stops_reading_cannot_wedge_the_engine_thread(monkeypatch):
    """THE regression test. A client that connects, asks for a completion and then
    stops reading — a suspended container, a slept laptop, a debugger breakpoint in the
    client's token loop — used to block the engine thread inside a socket write, with no
    timeout on it. That thread is the ONLY one allowed to touch MLX, so one such client
    took the whole server down for everyone, indefinitely, while /health still answered
    `ok`. `should_stop` could not save it either: it fires when a write RAISES, and a
    blocked write does not raise, it waits.

    The engine must finish (or abandon) this turn on its own and be free afterwards."""
    # Production waits a full minute before writing a client off; the mechanism is
    # identical at two seconds and the test doesn't have to sit through the wait.
    monkeypatch.setattr(serve, "WRITE_STALL_S", 2.0)
    eng = _wide_engine()
    state = serve.ServerState(eng, model_id="t", quiet=True)
    srv = serve.build_server(state, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        dead_client = _post(url, {"prompt": [1, 2, 3], "n_predict": 4000}, read=False)
        # Let it stream until the socket buffers are full and it has to block.
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            n = eng.produced
            time.sleep(0.5)
            if eng.produced == n and n > 0:
                break               # production has stalled: backpressure reached
        assert eng.produced > 0, "never streamed anything, test proves nothing"
        # The engine must NOT still be pinned there. Whether it cancelled or drained,
        # the one unacceptable outcome is "stuck forever holding the only engine
        # thread". One stall budget plus slack is the whole allowance.
        released = state.submit(lambda: "free").result(timeout=30)
        assert released == "free", "engine thread never came back"
        dead_client.close()
    finally:
        srv.shutdown()
        srv.server_close()
        state.close()


def test_a_slow_reader_does_not_gate_the_engine():
    """Decoupling, stated positively: the engine writes into a queue, so it runs ahead
    of a client that drains slowly instead of being paced by it. Before, the engine
    thread did the socket write itself and a 50ms-per-read consumer dropped its measured
    rate from 82k to 5.8k tok/s — TCP backpressure applied directly to MLX."""
    eng = FakeEngine(out=" ".join("y" * 20 for _ in range(300)))
    state = serve.ServerState(eng, model_id="t", quiet=True)
    srv = serve.build_server(state, "127.0.0.1", 0)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    url = f"http://127.0.0.1:{srv.server_address[1]}"
    try:
        s = _post(url, {"prompt": [1], "n_predict": 300})
        time.sleep(1.0)             # read nothing at all for a beat…
        # …by which time the engine, unblocked, has produced far more than the handful
        # of tokens a lockstep writer could have pushed into an undrained socket.
        assert eng.produced >= 100, f"engine only produced {eng.produced} while idle"
        s.close()
    finally:
        srv.shutdown()
        srv.server_close()
        state.close()


def test_a_second_completion_is_queued_not_dropped(live_server):
    """One KV cache means one generation at a time, so a concurrent request must WAIT
    and then run — not fail, and not race the prefix the running turn is built on."""
    url, eng, state = live_server
    eng.hold = threading.Event()
    out = {}

    def first():
        c = _client(url)
        out["first"] = c.generate([1, 2, 3], max_tokens=8)[0]

    t = threading.Thread(target=first, daemon=True)
    t.start()
    time.sleep(0.3)                         # first request is inside the engine, blocked
    second = threading.Thread(
        target=lambda: out.update(second=_client(url).generate([4], max_tokens=4)[0]),
        daemon=True)
    second.start()
    time.sleep(0.3)
    assert "second" not in out, "second generation ran while the first held the engine"
    eng.hold.set()                          # release the first
    t.join(10)
    second.join(10)
    assert out.get("first") and out.get("second"), "a queued request was dropped"


def test_hangup_mid_stream_still_cancels_through_the_queue(live_server, monkeypatch):
    """The queue sits between the engine and the socket now, so prove the cancel signal
    still crosses it: a client that hangs up must stop generation, not merely stop the
    writes."""
    monkeypatch.setattr(serve, "WRITE_STALL_S", 2.0)
    url, eng, _ = live_server
    eng.out = " ".join("z" * 200 for _ in range(2000))
    eng.delay = 0.002           # ~4s of generation, so the turn is genuinely in flight
    s = _post(url, {"prompt": [1], "n_predict": 2000})
    time.sleep(0.3)
    s.close()
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        n = eng.produced
        time.sleep(0.5)
        if eng.produced == n:
            break
    assert eng.produced < 2000, "generation ran to completion for a client that left"


# --- the sampler-config contract ------------------------------------------

def test_an_unset_sampler_knob_leaves_the_server_config_alone():
    """A server started with CHAD_MIN_P / CHAD_TOP_P must keep them. The knobs used to
    default to 0.0 and get written onto the engine unconditionally, so every request
    that didn't mention them ZEROED the operator's config — and chad's own client only
    sends min_p/top_p when they are armed, so that was nearly every request. Forcing
    temperature to 0.0 also picks the engine's decode strategy for it (the prompt-lookup
    path is gated on temp == 0.0)."""
    eng = FakeEngine()
    eng.temp, eng.min_p, eng.top_p = 0.7, 0.05, 0.95        # the operator's config
    serve.stream_completion(eng, serve.parse_completion_request({"prompt": [1]}),
                            sink()[0])
    assert eng.sampler_seen == (0.7, 0.05, 0.95), "an absent knob overwrote the config"
    assert (eng.temp, eng.min_p, eng.top_p) == (0.7, 0.05, 0.95)


def test_an_explicit_zero_still_disarms_a_knob():
    """0 = OFF is chad's convention, so a client that deliberately sends 0.0 must be
    able to turn a server-side knob off. That is the whole reason absent is None."""
    eng = FakeEngine()
    eng.min_p = 0.05
    serve.stream_completion(eng, serve.parse_completion_request(
        {"prompt": [1], "min_p": 0.0}), sink()[0])
    assert eng.sampler_seen[1] == 0.0       # explicitly disarmed for this request…
    assert eng.min_p == 0.05                # …and restored for the next one


# --- the cache-state contract on the wire ---------------------------------

def test_generated_ids_come_from_the_engine_not_a_cache_slice():
    """The ids on the wire must be what the engine says it generated. They used to be
    recovered by slicing `_cached_ids` past the prompt, which only holds on the main
    decode path: the prompt-lookup path stores what it FED the cache (the final pending
    token is deliberately absent) so the slice came up short by one every turn."""
    eng = FakeEngine(out="a b c", gen_ids=[7, 8, 9])
    write, payloads = sink()
    # a PLD-shaped cache: short by the last generated token, exactly as fed_ids is
    real_generate = eng.generate

    def pld_generate(prompt_ids, **kw):
        text, stats = real_generate(prompt_ids, **kw)
        eng._cached_ids = list(prompt_ids) + [7, 8]     # what was FED, not what was made
        return text, stats

    eng.generate = pld_generate
    serve.stream_completion(eng, serve.parse_completion_request({"prompt": [1, 2]}),
                            write)
    assert payloads()[-1]["tokens"] == [7, 8, 9], "fell back to the cache slice"


def test_a_cache_reset_tells_the_client_to_drop_its_mirror(live_server):
    """After a Metal OOM the engine rebuilds from empty, so NOTHING is resident — not
    even the prompt. Generated ids alone cannot say that, and a client that mirrored
    prompt+generation would size its next prefill against a cache that doesn't exist.
    The flag is the only honest answer."""
    url, eng, _ = live_server
    eng.cache_reset = True
    c = _client(url)
    text, stats = c.generate([1, 2, 3], max_tokens=8)
    assert text == "hello world"
    assert stats.cache_reset is True
    assert c._cached_ids == [], "client mirrored a cache the server had already dropped"


# --- admission control: never let a prompt reach the Metal allocator ------

def test_admit_clamps_the_budget_and_refuses_what_cannot_fit():
    """Pure, so the wall is testable without a model. Generation has to fit BESIDE the
    prompt in one KV cache — the budget is what's left, not what was asked for."""
    assert serve.admit(1000, 512, 8192) == (512, None)      # fits, untouched
    assert serve.admit(8000, 512, 8192) == (192, None)      # clamped to the room left
    n, err = serve.admit(8192, 512, 8192)                   # no room for even one token
    assert n == 0 and "context window is 8192" in err
    n, err = serve.admit(99999, 512, 8192)
    assert n == 0 and "99999 tokens" in err
    # an engine that hasn't reported a window yet must not invent one and reject
    # everything against a made-up number
    assert serve.admit(99999, 512, 0) == (512, None)


def test_admit_tightens_to_the_live_memory_wall_but_never_widens_past_it():
    """Two walls, on purpose. The advertised window is pinned so clients can budget
    against a number that doesn't move; the admission wall is live, because the Metal
    budget is blind to other processes and the KV cache grows into physical pages 1:1.
    Live pressure may only ever TIGHTEN — a memory probe that returned more than the
    model's own window must never be allowed to admit past it."""
    assert serve.admit(5000, 512, 32768, 8192) == (512, None)   # inside both walls
    n, err = serve.admit(5000, 512, 32768, 4096)     # fits the window, not today's memory
    assert n == 0 and "only 4096 tokens fit in GPU memory right now" in err
    assert "advertises a 32768-token window" in err  # so the remedy is obvious
    # under pressure the budget is clamped to the LIVE room, not the advertised room
    assert serve.admit(3000, 4096, 32768, 4096) == (1096, None)
    # a probe that reads HIGH cannot widen the wall past the real window
    assert serve.admit(3000, 99999, 8192, 999999) == (5192, None)
    # no probe available (no MLX, engine not loaded) → pinned window stands, as before
    assert serve.admit(3000, 512, 8192, 0) == (512, None)


def test_health_surfaces_memory_pressure_before_a_400_does(live_server):
    """A tightened wall is invisible from the client side — it budgeted against /props
    and everything looks fine until a request bounces. Report it where a monitor can see
    it coming."""
    url, _, state = live_server
    with urllib.request.urlopen(url + "/health", timeout=5) as r:
        body = json.loads(r.read())
    assert body["n_ctx"] == 8192
    assert body["safe_ctx"] <= body["n_ctx"]        # tightening only, never widening
    assert body["ctx_pressure"] is (body["safe_ctx"] < body["n_ctx"])
    # simulate another process taking the GPU: the wall tightens, the ADVERTISED
    # window does not move (clients have already baked it in)
    state.safe_ctx = lambda: 4096
    with urllib.request.urlopen(url + "/health", timeout=5) as r:
        body = json.loads(r.read())
    assert body["n_ctx"] == 8192 and body["safe_ctx"] == 4096
    assert body["ctx_pressure"] is True
    with urllib.request.urlopen(url + "/props", timeout=5) as r:
        assert json.loads(r.read())["default_generation_settings"]["n_ctx"] == 8192


def test_a_prompt_that_no_longer_fits_in_memory_is_a_400_not_an_oom(live_server):
    """The field failure this guard exists for: the prompt fit at load, then a docker
    stack took tens of GB. The pinned number still says yes; physical memory says no.
    A Metal OOM raised on a completion handler is not catchable — prevention is the only
    lever, so the request has to bounce before it reaches the allocator."""
    url, eng, state = live_server
    state.safe_ctx = lambda: 4096                   # pressure arrived after load
    req = urllib.request.Request(
        url + "/completion",
        data=json.dumps({"prompt": list(range(6000))}).encode(),   # < 8192, > 4096
        headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400
    assert "GPU memory right now" in e.value.read().decode()
    assert not eng.calls, "a prompt that could not fit still reached the engine"


def test_the_live_wall_survives_a_probe_that_cannot_run(live_server):
    """The guard must never be the thing that breaks the server. No MLX, an unloaded
    engine, a device_info that raises — all of it degrades to the pinned window."""
    url, _, state = live_server
    state.eng.kv_bytes_per_token = 0.0              # nothing measured to divide by
    state._safe_ctx_cache = (0.0, 0)
    assert state.safe_ctx() == 8192                 # pinned window, no exception
    c = _client(url)
    assert c.generate([1, 2, 3], max_tokens=4)[0] == "hello world"


def test_an_oversized_prompt_is_a_400_before_anything_is_prefilled(live_server):
    """The guard exists for the client that DIDN'T budget: a stale pinned context, a
    plain curl, a non-chad client. Overrunning the window is not a clean failure — it is
    a Metal allocation the engine may not survive, and an OOM raised on a completion
    handler can take the process with it. One caller gets a 400; nobody gets an OOM."""
    url, eng, _ = live_server                       # effective_ctx == 8192
    req = urllib.request.Request(
        url + "/completion",
        data=json.dumps({"prompt": list(range(9000))}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400
    assert "context window is 8192" in e.value.read().decode()
    assert not eng.calls, "the engine was touched by a request that could never fit"


def test_the_generation_budget_is_clamped_to_the_room_left(live_server):
    url, eng, _ = live_server
    c = _client(url)
    c.generate(list(range(8000)), max_tokens=4096)
    # prompt 8000 + budget 4096 would need 12096 of an 8192 window
    assert eng.calls[0][2] == 192, f"max_tokens reached the engine as {eng.calls[0][2]}"


def test_warm_refuses_a_prefix_that_cannot_fit(live_server):
    """A warm-start is a prefill, so it hits the same wall — and this one is easy to
    overshoot, since the prefix is assembled from a system prompt plus every tool schema."""
    url, eng, _ = live_server
    req = urllib.request.Request(
        url + "/warm", data=json.dumps({"prefix": list(range(9000))}).encode(),
        headers={"Content-Type": "application/json"}, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req, timeout=5)
    assert e.value.code == 400
    assert not [x for x in eng.calls if x[0] == "warm"]


# --- keepalive: a silent stream is a dead stream to any HTTP client -------

def test_a_long_silent_prefill_keeps_the_socket_warm(live_server, monkeypatch):
    """Clients time out on BYTES, not on our progress. A cold prefill of a long prompt
    emits no tokens for minutes, and chad's own client passes `timeout` to urlopen, which
    applies it PER READ — so silence gets the turn killed just before the first token.
    Worse, a socket timeout reads as transient, so the agent re-issues the step and pays
    for the same prefill again. SSE comment lines exist for exactly this; the client's
    parser already discards them (`parse_sse_chunk` returns None for a line starting ':')."""
    monkeypatch.setattr(serve, "KEEPALIVE_S", 0.2)
    url, eng, _ = live_server
    eng.hold = threading.Event()                # stall "prefill" before the first token
    s = _post(url, {"prompt": [1, 2, 3]}, read=False)
    s.settimeout(5)
    got = b""
    deadline = time.monotonic() + 4
    while time.monotonic() < deadline and b"keepalive" not in got:
        try:
            got += s.recv(4096)
        except OSError:
            break
    eng.hold.set()
    s.close()
    assert b": keepalive" in got, "stream went byte-silent through a long prefill"


def test_a_keepalive_is_invisible_to_the_client_parser():
    """The wire is shared with stock llama.cpp clients, so the keepalive must be a no-op
    to any SSE reader — not a chunk they have to know about."""
    from chad.completion_engine import parse_sse_chunk
    assert parse_sse_chunk(": keepalive") is None


def test_health_reports_whether_the_disk_spill_path_is_armed(live_server):
    """A quarantined cache is reclaimed to disk under memory pressure — but only if the
    engine has BOTH somewhere to write and a measured per-token cost. Either missing and
    the spill silently never happens, which is invisible from the client, so /health
    reports it."""
    url, eng, _ = live_server
    with urllib.request.urlopen(url + "/health", timeout=5) as r:
        spill = json.loads(r.read())["spill"]
    assert spill["armed"] is True
    assert spill["cache_dir"] == "/tmp/kv" and spill["kv_bytes_per_token"] == 1234.0
    # lose either half and it must report disarmed rather than quietly doing nothing
    eng.kv_bytes_per_token = 0.0
    with urllib.request.urlopen(url + "/health", timeout=5) as r:
        assert json.loads(r.read())["spill"]["armed"] is False
