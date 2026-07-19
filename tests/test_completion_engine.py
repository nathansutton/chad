"""Tests for the llama.cpp backend (`completion_engine.CompletionEngine`).

All tests run WITHOUT a model and WITHOUT network — the adapter's generation network
code lives in `_stream_completion` (stubbed with a canned SSE line generator) and its
props probe in `_fetch_props` (stubbed with a dict), so the pure request-build /
chunk-parse / cache-mirror / stats-accounting logic is exercised offline. Mirrors
tests/test_openai_engine.py; the deltas ARE the point: token-id prompts go over the
wire verbatim (no decode), and server timings produce exact (non-approximate) stats.
"""

from chad.base_engine import BaseEngine
from chad.completion_engine import (
    CompletionEngine,
    build_completion_body,
    chunk_text,
    common_prefix_len,
)

# --- protocol conformance -------------------------------------------------

def test_completion_adapter_satisfies_base_engine_protocol():
    ad = CompletionEngine(model_id="ornith", base_url="http://x:8081")
    assert isinstance(ad, BaseEngine)


def test_base_url_normalization():
    # documented spelling is the server origin; forgive slashes and the full path
    assert CompletionEngine("m", "http://h:8081").url == "http://h:8081/completion"
    assert CompletionEngine("m", "http://h:8081/").url == "http://h:8081/completion"
    assert CompletionEngine("m", "http://h:8081/completion").url == "http://h:8081/completion"


# --- pure helpers ---------------------------------------------------------

def test_build_completion_body_ships_token_ids_verbatim():
    body = build_completion_body([5, 6, 7], max_tokens=128, temp=0.0)
    assert body["prompt"] == [5, 6, 7]        # ids, NOT decoded text — the whole point
    assert body["n_predict"] == 128
    assert body["temperature"] == 0.0
    assert body["stream"] is True
    assert body["cache_prompt"] is True       # server-side prefix reuse
    assert body["return_tokens"] is True      # generated ids → cache mirror


def test_chunk_text():
    assert chunk_text({"content": "hi", "stop": False}) == "hi"
    assert chunk_text({"content": "", "stop": True, "timings": {}}) == ""  # final chunk
    assert chunk_text({}) == ""


def test_common_prefix_len():
    assert common_prefix_len([], [1, 2]) == 0
    assert common_prefix_len([1, 2, 3], [1, 2, 9]) == 2
    assert common_prefix_len([1, 2], [1, 2, 3, 4]) == 2


# --- generate() over a stubbed stream (no network) ------------------------

def _sse(*chunks):
    """Turn python dicts into a canned SSE line generator (llama.cpp framing)."""
    import json
    for c in chunks:
        yield f"data: {json.dumps(c)}\n"


def _adapter_with_stream(lines):
    ad = CompletionEngine(model_id="ornith", base_url="http://x:8081")
    ad._stream_completion = lambda body: iter(lines)   # stub: no network
    return ad


def test_generate_streams_tokens_with_exact_server_stats():
    lines = list(_sse(
        {"content": "Hello", "tokens": [11], "stop": False},
        {"content": " world", "tokens": [12], "stop": False},
        {"content": "", "tokens": [], "stop": True,
         "timings": {"prompt_n": 3, "prompt_ms": 250.0,
                     "predicted_n": 2, "predicted_ms": 100.0}},
    ))
    ad = _adapter_with_stream(lines)
    seen = []
    text, stats = ad.generate([1, 2, 3], max_tokens=64, on_token=seen.append)

    assert text == "Hello world"
    assert seen == ["Hello", " world"]
    # exact server accounting: nothing cached on a cold slot, real timings, NOT approximate
    assert stats.approximate is False
    assert stats.prompt_tokens == 3
    assert stats.cached_tokens == 0
    assert stats.generated_tokens == 2
    assert stats.prefill_s == 0.25
    assert stats.gen_s == 0.1
    # the cache mirror now holds prompt + generation, like the MLX engine's
    assert ad._cached_ids == [1, 2, 3, 11, 12]


def test_generate_reports_server_cache_hit():
    # server prefilled only 2 of 5 prompt tokens → 3 came from its prefix cache
    lines = list(_sse(
        {"content": "x", "tokens": [9], "stop": False},
        {"content": "", "tokens": [], "stop": True,
         "timings": {"prompt_n": 2, "prompt_ms": 10.0,
                     "predicted_n": 1, "predicted_ms": 5.0}},
    ))
    ad = _adapter_with_stream(lines)
    _, stats = ad.generate([1, 2, 3, 4, 5])
    assert stats.prompt_tokens == 2
    assert stats.cached_tokens == 3
    assert stats.approximate is False


def test_generate_falls_back_to_estimates_without_timings():
    lines = list(_sse({"content": "a", "stop": False}, {"content": "b", "stop": False}))
    ad = _adapter_with_stream(lines)
    text, stats = ad.generate([9], max_tokens=8)
    assert text == "ab"
    assert stats.approximate is True     # no server telemetry → honest estimate
    assert stats.generated_tokens == 2   # counted from streamed chunks
    assert stats.prompt_tokens == 1


def test_on_prefill_estimate_uses_the_cache_mirror():
    # first turn: cold — on_prefill(new=3, cached=0)
    lines1 = list(_sse(
        {"content": "x", "tokens": [4], "stop": False},
        {"content": "", "tokens": [], "stop": True,
         "timings": {"prompt_n": 3, "prompt_ms": 1.0, "predicted_n": 1, "predicted_ms": 1.0}},
    ))
    ad = _adapter_with_stream(lines1)
    calls = []
    ad.generate([1, 2, 3], on_prefill=lambda n, c: calls.append((n, c)))
    assert calls == [(3, 0)]
    # second turn extends prompt + generation append-only, as chad's transcript does:
    # the mirror [1,2,3,4] is a full prefix → on_prefill(new=1, cached=4)
    ad._stream_completion = lambda body: iter(_sse(
        {"content": "y", "tokens": [6], "stop": False},
        {"content": "", "tokens": [], "stop": True,
         "timings": {"prompt_n": 1, "prompt_ms": 1.0, "predicted_n": 1, "predicted_ms": 1.0}},
    ))
    calls.clear()
    progress = []
    ad.generate([1, 2, 3, 4, 5], on_prefill=lambda n, c: calls.append((n, c)),
                on_prefill_progress=lambda d, t: progress.append((d, t)))
    assert calls == [(1, 4)]
    assert progress == []                # no prefill progress across the boundary


def test_generate_interrupt_drops_the_stream():
    ad = _adapter_with_stream(list(_sse({"content": "nope", "stop": False})))
    text, stats = ad.generate([1], should_stop=lambda: True)
    assert text == ""
    assert stats.generated_tokens == 0
    assert stats.approximate is True     # dropped before the timings chunk


def test_generate_honors_stop_texts_and_keeps_the_marker():
    lines = list(_sse(
        {"content": "abc</tool_call>", "tokens": [1], "stop": False},
        {"content": "TRAILING", "tokens": [2], "stop": False},
    ))
    ad = _adapter_with_stream(lines)
    text, _ = ad.generate([1], stop_texts=["</tool_call>"])
    assert text == "abc</tool_call>"     # marker kept, later chunks dropped


def test_generate_honors_stop_condition():
    lines = list(_sse(
        {"content": "aa", "tokens": [1], "stop": False},
        {"content": "bb", "tokens": [2], "stop": False},
    ))
    ad = _adapter_with_stream(lines)
    text, stats = ad.generate([1], stop_condition=lambda t, n: n >= 1)
    assert text == "aa"
    assert stats.stop_condition_fired is True


def test_generate_raises_on_server_error_chunk():
    import pytest
    lines = list(_sse({"error": {"code": 400, "message": "exceeds the available context size"}}))
    ad = _adapter_with_stream(lines)
    with pytest.raises(RuntimeError, match="exceeds the available context size"):
        ad.generate([1])


def test_client_error_chunk_is_not_transient():
    """A 4xx is the prompt's fault — re-rolling it would just fail again."""
    import pytest

    from chad.base_engine import BackendError
    lines = list(_sse({"error": {"code": 400, "message": "exceeds the available context size"}}))
    with pytest.raises(BackendError) as ei:
        _adapter_with_stream(lines).generate([1])
    assert ei.value.transient is False


def test_server_error_chunk_is_transient():
    """llama.cpp answers 500 'does not match the expected Content-only format' when its
    chat parser can't reconcile a completion. That is sampling-dependent, so the agent is
    allowed to re-roll it (TB2 make-mips-interpreter died on exactly this)."""
    import pytest

    from chad.base_engine import BackendError
    lines = list(_sse({"error": {"code": 500, "message": "The model produced output that "
                                                        "does not match the expected "
                                                        "Content-only format"}}))
    with pytest.raises(BackendError) as ei:
        _adapter_with_stream(lines).generate([1])
    assert ei.value.transient is True


def test_http_5xx_is_transient_and_4xx_is_not(monkeypatch):
    """Exercises the real `_stream_completion` HTTPError->BackendError conversion by
    stubbing urlopen (stubbing _stream_completion itself would skip the code under test)."""
    import io
    import urllib.error
    import urllib.request

    import pytest

    from chad.base_engine import BackendError

    for code, transient in ((503, True), (422, False)):
        def _urlopen(req, timeout=None, _code=code):
            raise urllib.error.HTTPError("http://x", _code, "boom", {}, io.BytesIO(b"detail"))
        monkeypatch.setattr(urllib.request, "urlopen", _urlopen)
        ad = CompletionEngine(model_id="ornith", base_url="http://x:8081")
        with pytest.raises(BackendError) as ei:
            ad.generate([1])
        assert ei.value.transient is transient, code


def test_reset_clears_the_cache_mirror():
    ad = CompletionEngine(model_id="ornith", base_url="http://x:8081")
    ad._cached_ids = [1, 2]
    ad.reset()
    assert ad._cached_ids == []


# --- load() context sizing (props stubbed; tokenizer module stubbed so the fast
# gate never pays the real `transformers` import) ----------------------------

def _stub_transformers(monkeypatch):
    import sys
    import types
    fake = types.ModuleType("transformers")
    fake.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda _id: object())
    monkeypatch.setitem(sys.modules, "transformers", fake)


def test_load_sizes_context_from_server_props(monkeypatch):
    _stub_transformers(monkeypatch)
    ad = CompletionEngine(model_id="ornith", base_url="http://x:8081")  # effective_ctx=0 → auto
    ad._fetch_props = lambda: {"default_generation_settings": {"n_ctx": 131072}}
    ad.load()
    assert ad.effective_ctx == 131072


# --- Close-and-continue (think-spiral salvage) ------------------

class _FakeTok:
    """Minimal stand-in for the HF tokenizer: encode() only, since this backend never
    decodes (token-id prompts go over the wire verbatim)."""

    def encode(self, text, add_special_tokens=False):
        # deterministic, distinguishable ids so the continuation prompt is inspectable
        return [1000 + i for i in range(len(text))]


def _salvage_adapter(first_lines, cont_lines):
    """A CompletionEngine whose stubbed stream returns `first_lines` for the initial
    request and `cont_lines` for any continuation (identified by a longer `prompt`
    than the original). Records every request body so the test can assert the
    continuation shape."""
    ad = CompletionEngine(model_id="ornith", base_url="http://x:8081")
    ad.tok = _FakeTok()
    bodies = []

    def stream(body):
        bodies.append(body)
        is_continuation = len(bodies) > 1
        return iter(cont_lines if is_continuation else first_lines)

    ad._stream_completion = stream
    return ad, bodies


def test_generate_close_and_continue_salvages_a_runaway_think():
    # First request: 3 think chunks, no </think> -> crosses ceiling=3 and the salvage
    # fires (the 4th chunk is never consumed). Continuation: the action, in ONE turn.
    first = list(_sse(
        {"content": "reasoning ", "tokens": [11], "stop": False},
        {"content": "and more ", "tokens": [12], "stop": False},
        {"content": "and more ", "tokens": [13], "stop": False},
        {"content": "NEVER-REACHED", "tokens": [14], "stop": False},
    ))
    cont = list(_sse(
        {"content": '<tool_call>{"name":"read"}</tool_call>', "tokens": [99], "stop": False},
        {"content": "", "tokens": [], "stop": True,
         "timings": {"prompt_n": 2, "prompt_ms": 5.0,
                     "predicted_n": 1, "predicted_ms": 10.0}},
    ))
    ad, bodies = _salvage_adapter(first, cont)
    text, stats = ad.generate([1, 2, 3], max_tokens=200, think_ceiling=3)

    # One turn: the runaway think, a single injected close, then the action.
    assert stats.salvaged is True
    assert text.count("</think>") == 1
    assert text == 'reasoning and more and more \n</think>\n\n<tool_call>{"name":"read"}</tool_call>'
    assert "NEVER-REACHED" not in text          # the 4th think chunk was cut off
    # Exactly two requests: the initial and one continuation.
    assert len(bodies) == 2
    # The continuation's prompt is the original ids + generation-so-far + the tokenized
    # close marker — a strict PREFIX-extension (the KV/prefix-cache invariant): no
    # detokenize round-trip on this backend, unlike the openai adapter.
    cont_prompt = bodies[1]["prompt"]
    assert cont_prompt[:3] == [1, 2, 3]
    assert cont_prompt[3:6] == [11, 12, 13]     # the runaway think's own generated ids
    # a strict prefix-extension of the first request's prompt (the KV-prefix invariant)
    assert cont_prompt[:len(bodies[0]["prompt"])] == bodies[0]["prompt"]
    # The runaway request's 3 tokens must NOT be dropped just because its stream was
    # closed before the server could send a final timings chunk for it: 3 (no-timings
    # fallback, request 1) + 1 (predicted_n, the continuation) = 4. A prior bug silently
    # discarded request 1's count whenever ANY request in the step reported timings.
    assert stats.generated_tokens == 4
    # cache mirror ends up exactly what the server's slot cache now holds
    assert ad._cached_ids[:6] == [1, 2, 3, 11, 12, 13]
    assert ad._cached_ids[-1] == 99


def test_generate_no_salvage_when_think_closes_before_the_ceiling():
    # The model closes </think> and acts on its own, under the ceiling -> no salvage,
    # a single request, byte-identical to a plain generate.
    lines = list(_sse(
        {"content": "brief think</think>\nact", "tokens": [5], "stop": False},
        {"content": "", "tokens": [], "stop": True,
         "timings": {"prompt_n": 1, "prompt_ms": 1.0, "predicted_n": 1, "predicted_ms": 1.0}},
    ))
    ad, bodies = _salvage_adapter(lines, lines)
    text, stats = ad.generate([1], max_tokens=200, think_ceiling=6000)
    assert stats.salvaged is False
    assert len(bodies) == 1
    assert text == "brief think</think>\nact"


def test_generate_no_salvage_when_ceiling_unset():
    # think_ceiling=None (the default): a long open think is NOT force-closed.
    lines = list(_sse(
        {"content": "a ", "tokens": [1], "stop": False},
        {"content": "b ", "tokens": [2], "stop": False},
        {"content": "c ", "tokens": [3], "stop": False},
        {"content": "", "tokens": [], "stop": True,
         "timings": {"prompt_n": 1, "prompt_ms": 1.0, "predicted_n": 3, "predicted_ms": 1.0}},
    ))
    ad, bodies = _salvage_adapter(lines, lines)
    text, stats = ad.generate([1], max_tokens=200, think_ceiling=None)
    assert stats.salvaged is False
    assert len(bodies) == 1


def test_load_keeps_explicit_ctx_and_survives_offline_props(monkeypatch):
    _stub_transformers(monkeypatch)
    ad = CompletionEngine(model_id="ornith", base_url="http://x:8081", effective_ctx=4096)
    ad._fetch_props = lambda: (_ for _ in ()).throw(AssertionError("must not probe"))
    ad.load()
    assert ad.effective_ctx == 4096      # explicit setting wins; no probe
    ad2 = CompletionEngine(model_id="ornith", base_url="http://x:8081")
    ad2._fetch_props = lambda: None      # offline / older server
    ad2.load()
    assert ad2.effective_ctx == 32768    # safe fallback
