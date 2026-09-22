"""The polyglot kit's proxy, the log → ATIF conversion, the cross-check, and the rule that
keeps two engines off the machine. The upstream is a stub serving replies captured from
llama-server build 10470 (`tests/fixtures/llama_server/`); no model, no real server."""
import http.server
import json
import os
import sys
import threading
import time
import urllib.request

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "benchmarks", "polyglot"))

import proxy  # noqa: E402
import proxy_atif  # noqa: E402
import run  # noqa: E402
import server  # noqa: E402
from harness import cli  # noqa: E402
from harness.proxied import Proxied  # noqa: E402

from chad import seatbelt  # noqa: E402
from test_polyglot_harness import _fake  # noqa: E402
from test_polyglot_kit import _exercise  # noqa: E402

FIXTURES = os.path.join(REPO, "tests", "fixtures", "llama_server")
FORCED = proxy.shipped_sampler()


def _fixture(name):
    with open(os.path.join(FIXTURES, name), "rb") as f:
        return f.read()


@pytest.mark.parametrize("name", ["chat_tool.json", "chat_tool_stream.sse"])
def test_a_chat_tool_call_reads_the_same_streamed_or_not(name):
    reply = proxy.parse_reply(_fixture(name))
    assert [(c.name, json.loads(c.arguments)) for c in reply.tool_calls] == \
        [("bash", {"command": "ls -la"})]
    assert reply.finish == "tool_calls" and reply.timings["predicted_n"] == 28
    assert reply.tool_calls[0].id


def test_a_streamed_reply_keeps_the_final_timings():
    timings = proxy.parse_reply(_fixture("chat_tool_stream.sse")).timings
    assert (timings["prompt_n"], timings["cache_n"]) == (4, 287)


def test_reasoning_streams_apart_from_the_answer():
    reply = proxy.parse_reply(_fixture("chat_think_stream.sse"))
    assert reply.reasoning.startswith("The") and reply.timings["prompt_n"] == 327


def test_a_responses_stream_is_read_once():
    reply = proxy.parse_reply(_fixture("responses_stream.sse"))
    assert [c.name for c in reply.tool_calls] == ["bash"]
    assert reply.finish == "completed" and reply.timings["prompt_n"] == 279


def test_a_stream_that_opens_with_a_comment_is_still_a_stream():
    reply = proxy.parse_reply(b": keep-alive\n\n" + _fixture("completion_stream.sse"))
    assert reply.content.startswith("<think>") and reply.timings["predicted_n"] == 24


def test_completion_echoes_the_sampler_it_applied():
    reply = proxy.parse_reply(_fixture("completion_stream.sse"))
    assert reply.content.startswith("<think>") and reply.timings["predicted_n"] == 24
    assert proxy.sampler_check(FORCED, reply.settings) == "ok"
    assert proxy.sampler_check(FORCED, {**reply.settings, "min_p": 0.05}).startswith("mismatch: min_p")
    assert proxy.sampler_check(FORCED, {}) == "unechoed"


def test_the_forced_sampler_is_the_shipped_one_with_every_zero_explicit():
    from chad.cli import THINKING_SAMPLER
    assert FORCED["temperature"] == THINKING_SAMPLER["temp"]
    assert FORCED["min_p"] == 0.0 and FORCED["top_k"] == 20 and FORCED["dry_multiplier"] == 0.0


class _Upstream:
    """A stand-in llama-server: every generation request gets `reply`, in two writes."""

    def __init__(self, reply, applied=None):
        self.bodies = []
        upstream = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def _json(self, doc):
                body = json.dumps(doc).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.startswith("/slots"):
                    return self._json([{"id": 0, "id_task": 3, "params": applied or FORCED},
                                       {"id": 1, "id_task": 1, "params": {"min_p": 0.05}}])
                return self._json({"default_generation_settings": {"n_ctx": 32768}})

            def do_POST(self):
                body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                if "action=erase" in self.path:
                    return self._json({})
                upstream.bodies.append(json.loads(body))
                if json.loads(body).get("hold"):
                    time.sleep(5)        # a non-streamed generation that outlives its trial
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                half = len(reply) // 2
                self.wfile.write(reply[:half])
                self.wfile.flush()
                self.wfile.write(reply[half:])
                self.close_connection = True

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        self.origin = f"http://127.0.0.1:{self.server.server_port}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


@pytest.fixture
def relay(tmp_path):
    upstream = _Upstream(_fixture("chat_tool_stream.sse"))
    p = proxy.Proxy(upstream.origin, FORCED, str(tmp_path / "stray.jsonl"))
    p.start()
    yield p, upstream
    p.stop()
    upstream.close()


def _post(origin, path, body):
    req = urllib.request.Request(origin + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=10) as r:
        return r.read()


def test_the_proxy_forces_the_sampler_and_passes_the_reply_through(relay, tmp_path):
    p, upstream = relay
    p.route(str(tmp_path / "trial.jsonl"))
    body = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": True,
            "temperature": 0.3, "top_p": 1.0}
    assert _post(p.origin, "/v1/chat/completions", body) == _fixture("chat_tool_stream.sse")
    sent = upstream.bodies[-1]
    assert {k: sent[k] for k in FORCED} == FORCED and sent["messages"] == body["messages"]
    record = proxy_atif.load(str(tmp_path / "trial.jsonl"))[0]
    assert record["asked"] == {"temperature": 0.3, "top_p": 1.0}
    assert record["request"]["temperature"] == 0.3            # as the harness sent it
    assert record["sampler"] == "unechoed" and record["reply"]["timings"]["cache_n"] == 287
    with urllib.request.urlopen(p.origin + "/props", timeout=10) as r:
        assert json.loads(r.read())["default_generation_settings"]["n_ctx"] == 32768
    p.route("")
    _post(p.origin, "/v1/chat/completions", body)
    assert len(proxy_atif.load(str(tmp_path / "stray.jsonl"))) == 1


def test_a_request_belongs_to_the_trial_it_arrived_in_and_ends_with_it(relay, tmp_path):
    p, _ = relay
    p.route(str(tmp_path / "trial-a.jsonl"))
    body = {"model": "m", "messages": [SYSTEM, USER], "hold": True}   # the agent's own thread
    caller = threading.Thread(target=lambda: _post_quietly(p.origin, body))
    caller.start()
    time.sleep(0.5)
    t0 = time.time()
    p.route(str(tmp_path / "trial-b.jsonl"))              # trial A is over
    caller.join(timeout=10)
    deadline = time.time() + 5
    while not (tmp_path / "trial-a.jsonl").exists() and time.time() < deadline:
        time.sleep(0.05)
    record = proxy_atif.load(str(tmp_path / "trial-a.jsonl"))[0]
    assert record["aborted"] is True and time.time() - t0 < 4    # hung up, not waited out
    assert not (tmp_path / "trial-b.jsonl").exists()
    doc = proxy_atif.convert([*_chat_log()[:1], record], "x", "0", "m")
    assert doc["steps"][-1]["extra"] == {"in_flight": True, "aborted": True}
    assert doc["final_metrics"]["extra"]["aborted"] == 1


def test_a_client_that_hangs_up_takes_its_generation_with_it(relay, tmp_path):
    """A non-streaming client that times out and leaves: llama-server would cancel the
    generation for a client it saw leave, so the proxy must not keep it alive."""
    import socket as sock
    p, _ = relay
    p.route(str(tmp_path / "trial.jsonl"))
    body = json.dumps({"model": "m", "messages": [SYSTEM, USER], "hold": True}).encode()
    host, port = p.origin.removeprefix("http://").split(":")
    client = sock.create_connection((host, int(port)))
    client.sendall(b"POST /v1/chat/completions HTTP/1.1\r\nHost: x\r\nContent-Type: "
                   b"application/json\r\nContent-Length: " + str(len(body)).encode() +
                   b"\r\n\r\n" + body)
    time.sleep(0.5)
    t0 = time.time()
    client.close()                                     # the client's own timeout
    deadline = time.time() + 5
    while not (tmp_path / "trial.jsonl").exists() and time.time() < deadline:
        time.sleep(0.05)
    record = proxy_atif.load(str(tmp_path / "trial.jsonl"))[0]
    assert record["client_gone"] is True and record["aborted"] is False
    assert time.time() - t0 < 3                        # hung up, not left to finish
    doc = proxy_atif.convert([*_chat_log()[:1], record], "x", "0", "m")
    assert doc["steps"][-1]["extra"] == {"in_flight": True, "aborted": False}
    assert doc["final_metrics"]["extra"]["abandoned"] == 1


def _post_quietly(origin, body):
    try:
        _post(origin, "/v1/chat/completions", body)
    except OSError:
        pass                           # the proxy hung up, as it should


def test_the_audit_reads_the_slot_that_served_each_probe(relay):
    p, _ = relay
    assert proxy.audit(p, "m") == {"chat": "ok", "responses": "ok", "completion": "ok"}
    assert proxy.erase_slots(p.upstream) == 2
    assert proxy._slot_verdict(FORCED, [{"id_task": 1, "params": {**FORCED, "top_k": 40}}]) \
        .startswith("mismatch: top_k")
    assert proxy._slot_verdict(FORCED, [{"id_task": 1}]).startswith("unverified")
    # An idle slot may keep no task id: the probe's own n_predict still names it.
    slots = [{"id_task": -1, "params": {**FORCED, "n_predict": 3}},
             {"id_task": -1, "params": {**FORCED, "min_p": 0.05, "n_predict": 2}}]
    assert proxy._slot_verdict(FORCED, slots, marker=3) == "ok"
    assert proxy._slot_verdict(FORCED, slots, marker=2).startswith("mismatch: min_p")


def _record(t, messages, reply, timings, tools=True):
    request = {"messages": messages, "stream": True}
    if tools:
        request["tools"] = [{"type": "function", "function": {"name": "bash"}}]
    return {"t": t, "ttfb_s": 1.0, "sampler": "unechoed", "request": request,
            "reply": {"content": reply.get("content", ""), "reasoning": reply.get("reasoning", ""),
                      "tool_calls": reply.get("tool_calls", []), "finish": reply.get("finish", "stop"),
                      "timings": timings}}


SYSTEM = {"role": "system", "content": "you are an agent"}
USER = {"role": "user", "content": "fix it"}
CALL = {"id": "c1", "name": "bash", "arguments": '{"command": "sh run-tests.sh"}'}


def _chat_log():
    first = _record(1.0, [SYSTEM, USER],
                    {"reasoning": "let me run the tests", "tool_calls": [CALL], "finish": "tool_calls"},
                    {"prompt_n": 300, "cache_n": 0, "predicted_n": 40, "prompt_ms": 3000,
                     "predicted_ms": 4000})
    side = _record(1.5, [{"role": "system", "content": "write a title"}, USER],
                   {"content": "Fixing it"}, {"prompt_n": 50, "cache_n": 0, "predicted_n": 3,
                                              "prompt_ms": 500, "predicted_ms": 300}, tools=False)
    second = _record(2.0, [SYSTEM, USER, {"role": "assistant", "tool_calls": []},
                           {"role": "tool", "tool_call_id": "c1", "content": "1 passed"}],
                     {"content": "done"}, {"prompt_n": 20, "cache_n": 300, "predicted_n": 5,
                                           "prompt_ms": 200, "predicted_ms": 500})
    return [first, side, second]


def test_a_chat_log_becomes_a_trajectory_with_side_requests_apart():
    doc = proxy_atif.convert(_chat_log(), "pi", "0.80.3", "m",
                             count_tokens=lambda text: len(text.split()))
    steps = doc["steps"]
    assert [s["source"] for s in steps] == ["system", "user", "agent", "agent"]
    act = steps[2]
    assert act["tool_calls"] == [{"tool_call_id": "c1", "function_name": "bash",
                                  "arguments": {"command": "sh run-tests.sh"}}]
    assert act["observation"] == {"results": [{"source_call_id": "c1", "content": "1 passed"}]}
    assert act["metrics"]["extra"]["think_tokens"] == 5
    assert steps[3]["metrics"]["prompt_tokens"] == 320 and steps[3]["metrics"]["cached_tokens"] == 300
    final = doc["final_metrics"]
    assert (final["total_prompt_tokens"], final["total_completion_tokens"]) == (620, 45)
    assert final["extra"]["side"]["requests"] == 1 and final["extra"]["requests"] == 3
    assert final["extra"]["total_think_tokens"] == 5


def test_a_completion_log_is_parsed_the_way_chad_parses_its_own_text():
    call = '<tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>'
    record = {"t": 1.0, "request": {"prompt": [1, 2, 3], "stream": True}, "sampler": "ok",
              "reply": {"content": f"<think>look first</think>{call}", "finish": "eos",
                        "timings": {"prompt_n": 3, "cache_n": 0, "predicted_n": 9}}}
    steps = proxy_atif.convert([record], "chad-llama", "2.2.0", "m")["steps"]
    assert [c["function_name"] for c in steps[-1]["tool_calls"]] == ["bash"]
    assert steps[-1]["reasoning_content"] == "look first"
    assert "think_tokens" not in steps[-1]["metrics"]["extra"]


def test_the_crosscheck_names_every_disagreement():
    doc = proxy_atif.convert(_chat_log(), "pi", "0", "m")
    assert proxy_atif.crosscheck(doc, doc) == []
    fewer = {**doc, "steps": doc["steps"][:-1]}
    problems = proxy_atif.crosscheck(doc, fewer)
    assert any(p.startswith("agent steps") for p in problems)
    assert any(p.startswith("prompt_tokens") for p in problems)
    shifted = json.loads(json.dumps(doc))
    shifted["steps"][2]["metrics"]["prompt_tokens"] += 3          # 3 of 620: inside 1%
    assert proxy_atif.crosscheck(doc, shifted) == []


def test_two_engines_are_never_resident():
    def pgrep(blocks=(), servers=()):
        return lambda *args: list(servers if "-x" in args else blocks)

    server.refuse("cli block", pgrep(servers=[11]))                  # its own server: fine
    with pytest.raises(server.EngineBusy, match="llama-server is running"):
        server.refuse("in-process block", pgrep(servers=[11]))
    with pytest.raises(server.EngineBusy, match="another polyglot block"):
        server.refuse("cli block", pgrep(blocks=[99999]))
    with pytest.raises(server.EngineBusy, match="llama-server is running"):
        server.refuse("server", pgrep(servers=[11]))
    server.refuse("in-process block", pgrep(blocks=[os.getpid()]))  # itself does not count


def test_the_server_is_started_so_its_slots_can_be_erased(tmp_path):
    argv = server.LlamaServer("m.gguf", str(tmp_path / "server.log")).argv()
    assert argv[argv.index("--slot-save-path") + 1] == str(tmp_path / "llama-slots")
    assert argv[argv.index("-c") + 1] == "32768" and "--jinja" in argv


needs_sandbox = pytest.mark.skipif(not seatbelt.Seatbelt().probe(),
                                   reason="sandbox-exec cannot enforce from here")

# The fake agent's one model call: through the proxy, then the reference solution.
_CALLS_MODEL = """
req = urllib.request.Request(os.environ["FAKE_ORIGIN"] + "/v1/chat/completions",
    data=json.dumps({"model": "m", "stream": True, "temperature": 0.3,
                     "messages": [{"role": "system", "content": "sys"},
                                  {"role": "user", "content": "go"}]}).encode(),
    headers={"Content-Type": "application/json"})
urllib.request.urlopen(req, timeout=30).read()
"""


@needs_sandbox
def test_a_proxied_cli_trial_gets_the_servers_counts_and_a_trajectory(tmp_path, relay):
    p, _ = relay
    task = _exercise(tmp_path / "up")
    arm = _fake(tmp_path, {"gold": os.path.join(task.source, ".meta", "example.py"),
                           "outside": str(tmp_path / "outside.txt")})
    exe = tmp_path / "bin" / "fake-agent"
    exe.write_text(exe.read_text().replace(
        "shutil.copyfile(config", "import urllib.request\n" + _CALLS_MODEL + "shutil.copyfile(config", 1))
    arm.endpoint = cli.Endpoint(p.origin, "m", 32768, "")
    wrapped = Proxied(arm, p, count_tokens=lambda text: len(text.split()))
    block = run.Block("arm", wrapped, wall_cap=120, runs=str(tmp_path / "_runs"))
    meta = wrapped.start()
    assert meta["sampler_audit"] == {"chat": "ok", "responses": "ok", "completion": "ok"}
    record = block.solve(task, 1)
    assert record["passed"] is True and record["exit_code"] == 0
    assert (record["gen_tokens"], record["requests"], record["side_requests"]) == (28, 1, 0)
    assert record["sampler"] == {"unechoed": 1}
    assert record["trajectory"] == "trajectories/python/two-fer.rep1.json"
    with open(tmp_path / "_runs" / "arm" / record["trajectory"], encoding="utf-8") as f:
        doc = json.load(f)
    assert [s["source"] for s in doc["steps"]] == ["system", "user", "agent"]
    assert doc["steps"][2]["tool_calls"][0]["function_name"] == "bash"
    assert record["metrics"]["total_completion_tokens"] == 28


def test_a_runs_output_is_not_read_as_more_trials(tmp_path):
    import trace as kit_trace
    for rel in ("trajectories/go/x.rep1.json", "output/go/x.rep1.proxy.json"):
        path = tmp_path / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"steps": [{"source": "user"}]}))
    assert [p for p, _ in kit_trace.load_run(str(tmp_path))] == ["trajectories/go/x.rep1.json"]
