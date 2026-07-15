"""Tests for the plan 046 engine seam: the `BaseEngine` protocol and the OpenAI-backend
adapter (`openai_engine.OpenAIEngine`).

All tests run WITHOUT a model and WITHOUT network — the adapter's only network code lives
in `_stream_completion`, which we stub with a canned SSE line generator so the pure
request-build / SSE-parse / generate-accounting logic is exercised offline. The protocol
tests assert both backends structurally satisfy `BaseEngine` (runtime_checkable isinstance);
the MLX `Engine` is checked without loading weights.
"""

from chad.base_engine import BaseEngine, GenStats
from chad.engine import Engine
from chad.openai_engine import (
    OpenAIEngine,
    build_chat_body,
    delta_text,
    parse_sse_chunk,
)

# --- protocol conformance -------------------------------------------------

def test_mlx_engine_satisfies_base_engine_protocol():
    # Build a bare Engine without loading weights (mirrors test_engine.py). The only
    # instance-set field a runtime_checkable data-member check needs is `_cached_ids`
    # (a default_factory field, so not a class attribute); the rest are dataclass
    # defaults present on the class.
    eng = object.__new__(Engine)
    eng._cached_ids = []
    assert isinstance(eng, BaseEngine)
    # the plan's public reset() alias must exist and be callable-compatible
    assert hasattr(Engine, "reset")


def test_openai_adapter_satisfies_base_engine_protocol():
    ad = OpenAIEngine(model_id="ornith", base_url="http://localhost:8080/v1")
    assert isinstance(ad, BaseEngine)


def test_openai_adapter_base_url_normalization():
    # forgiving about trailing slash and about being handed the full completions path
    assert OpenAIEngine("m", "http://h:8080/v1").url == "http://h:8080/v1/chat/completions"
    assert OpenAIEngine("m", "http://h:8080/v1/").url == "http://h:8080/v1/chat/completions"
    assert (OpenAIEngine("m", "http://h:8080/v1/chat/completions").url
            == "http://h:8080/v1/chat/completions")


# --- pure helpers ---------------------------------------------------------

def test_build_chat_body_ships_decoded_prompt_as_single_user_message():
    body = build_chat_body("ornith", "SYS+TOOLS+turns", max_tokens=128, temp=0.0)
    assert body["model"] == "ornith"
    assert body["messages"] == [{"role": "user", "content": "SYS+TOOLS+turns"}]
    assert body["max_tokens"] == 128
    assert body["temperature"] == 0.0
    assert body["stream"] is True
    assert body["stream_options"] == {"include_usage": True}
    assert "stop" not in body  # no local stop markers -> field omitted


def test_build_chat_body_forwards_stop_markers_capped_at_four():
    # Stop markers must reach the SERVER: a client that stops reading locally and
    # drops the connection desyncs a warm-prefix server's cache tail from the
    # transcript it keeps (plan 075 canary finding). OpenAI-compatible servers cap
    # `stop` at 4 entries.
    body = build_chat_body("m", "p", max_tokens=8, temp=0.0,
                           stop=["</tool_call>", "a", "b", "c", "d"])
    assert body["stop"] == ["</tool_call>", "a", "b", "c"]
    assert build_chat_body("m", "p", 8, 0.0, stop=[]).get("stop") is None


def test_parse_sse_chunk():
    assert parse_sse_chunk("") is None
    assert parse_sse_chunk(": keep-alive comment") is None
    assert parse_sse_chunk("data: [DONE]") is None
    assert parse_sse_chunk('data: {"choices": []}') == {"choices": []}
    # tolerates the "data:" spelling without a trailing space
    assert parse_sse_chunk('data:{"x": 1}') == {"x": 1}


def test_delta_text():
    assert delta_text({"choices": [{"delta": {"content": "hi"}}]}) == "hi"
    assert delta_text({"choices": []}) == ""            # usage-only final chunk
    assert delta_text({"choices": [{"delta": {}}]}) == ""  # role-only opening chunk


# --- generate() over a stubbed stream (no network) ------------------------

class _FakeTok:
    """Minimal stand-in for the HF tokenizer: records the decode call so we can assert
    the adapter decodes the ids (degradation #1) and returns a deterministic string."""

    def __init__(self):
        self.decoded_with = None

    def decode(self, ids, skip_special_tokens=False):
        self.decoded_with = (list(ids), skip_special_tokens)
        return "DECODED_PROMPT"


def _sse(*lines):
    """Turn python dicts / literals into a canned SSE line generator."""
    import json
    for x in lines:
        yield f"data: {x if isinstance(x, str) else json.dumps(x)}\n"


def _adapter_with_stream(lines):
    ad = OpenAIEngine(model_id="ornith", base_url="http://x/v1")
    ad.tok = _FakeTok()
    ad._stream_completion = lambda body: iter(lines)   # stub: no network
    return ad


def test_generate_streams_tokens_and_marks_stats_approximate():
    lines = list(_sse(
        {"choices": [{"delta": {"role": "assistant"}}]},
        {"choices": [{"delta": {"content": "Hello"}}]},
        {"choices": [{"delta": {"content": " world"}}]},
        {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 2}},
        "[DONE]",
    ))
    ad = _adapter_with_stream(lines)
    seen = []
    text, stats = ad.generate([1, 2, 3], max_tokens=64, on_token=seen.append)

    assert text == "Hello world"
    assert seen == ["Hello", " world"]
    # decode-the-ids degradation: the exact prompt ids were decoded, special tokens kept
    assert ad.tok.decoded_with == ([1, 2, 3], False)
    # honest degradations recorded in the stats
    assert stats.approximate is True
    assert stats.cached_tokens == 0
    # server usage is preferred when present
    assert stats.generated_tokens == 2
    assert stats.prompt_tokens == 42


def test_generate_adopts_server_cached_tokens():
    """chad --serve reports prefix-cache reuse in prompt_tokens_details; the client
    must adopt it (TB2 trajectories logged cached_tokens=0 against a warm server)."""
    lines = list(_sse(
        {"choices": [{"delta": {"content": "hi"}}]},
        {"choices": [], "usage": {"prompt_tokens": 42, "completion_tokens": 1,
                                  "prompt_tokens_details": {"cached_tokens": 40}}},
        "[DONE]",
    ))
    ad = _adapter_with_stream(lines)
    _, stats = ad.generate([1, 2, 3], max_tokens=8)
    assert stats.prompt_tokens == 42
    assert stats.cached_tokens == 40


def test_generate_falls_back_to_chunk_count_without_usage():
    lines = list(_sse(
        {"choices": [{"delta": {"content": "a"}}]},
        {"choices": [{"delta": {"content": "b"}}]},
        "[DONE]",
    ))
    ad = _adapter_with_stream(lines)
    text, stats = ad.generate([9], max_tokens=8)
    assert text == "ab"
    assert stats.generated_tokens == 2   # counted from streamed chunks
    assert stats.prompt_tokens == 1      # len(prompt_ids), no usage reported


def test_generate_fires_on_prefill_once_and_never_progress():
    lines = list(_sse({"choices": [{"delta": {"content": "x"}}]}, "[DONE]"))
    ad = _adapter_with_stream(lines)
    prefill_calls = []
    progress_calls = []
    ad.generate([1, 2], on_prefill=lambda n, c: prefill_calls.append((n, c)),
                on_prefill_progress=lambda d, t: progress_calls.append((d, t)))
    assert prefill_calls == [(2, 0)]     # once, with 0 cached (nothing is cacheable)
    assert progress_calls == []          # no prefill progress across the boundary


def test_generate_interrupt_drops_the_stream():
    # should_stop() true from the start: we bail before consuming any content.
    consumed = []

    def gen():
        for x in _sse({"choices": [{"delta": {"content": "nope"}}]}, "[DONE]"):
            consumed.append(x)
            yield x

    ad = OpenAIEngine(model_id="ornith", base_url="http://x/v1")
    ad.tok = _FakeTok()
    ad._stream_completion = lambda body: gen()
    text, stats = ad.generate([1], should_stop=lambda: True)
    assert text == ""
    assert stats.generated_tokens == 0


def test_generate_honors_stop_condition():
    lines = list(_sse(
        {"choices": [{"delta": {"content": "aa"}}]},
        {"choices": [{"delta": {"content": "bb"}}]},
        {"choices": [{"delta": {"content": "cc"}}]},
        "[DONE]",
    ))
    ad = _adapter_with_stream(lines)
    # stop after the first decoded chunk
    text, stats = ad.generate([1], stop_condition=lambda t, n: n >= 1)
    assert text == "aa"
    assert stats.stop_condition_fired is True


def test_gen_stats_approximate_default_false():
    # the new field must default off so the MLX engine's stats are unaffected
    assert GenStats().approximate is False


# --- plan 086: close-and-continue (think-spiral salvage) ------------------

def test_think_ceiling_hit_only_inside_an_open_think_over_budget():
    from chad.base_engine import think_ceiling_hit
    # inside <think> (no close emitted) and over budget -> fire
    assert think_ceiling_hit("still reasoning", 6000, 6000)
    assert think_ceiling_hit("reasoning", 7000, 6000)
    # under budget -> no
    assert not think_ceiling_hit("reasoning", 5999, 6000)
    # </think> already emitted (thinking done) -> no, whatever the count
    assert not think_ceiling_hit("done</think>\nact", 9000, 6000)
    # ceiling off (None / 0) -> never fires
    assert not think_ceiling_hit("reasoning", 10**9, None)
    assert not think_ceiling_hit("reasoning", 10**9, 0)


def test_build_chat_body_appends_assistant_prefix_for_continuation():
    # A continuation request (plan 086) carries the force-closed <think> as a trailing
    # assistant message the server CONTINUES rather than answers.
    body = build_chat_body("m", "PROMPT", max_tokens=64, temp=1.0,
                           assistant_prefix="<think>reasoned\n</think>\n\n")
    assert body["messages"] == [
        {"role": "user", "content": "PROMPT"},
        {"role": "assistant", "content": "<think>reasoned\n</think>\n\n"},
    ]
    # absent prefix -> the plain single-user-message shape is unchanged
    assert build_chat_body("m", "P", 8, 1.0)["messages"] == [{"role": "user", "content": "P"}]


def _salvage_adapter(first_lines, cont_lines):
    """An OpenAIEngine whose stubbed stream returns `first_lines` for the initial request
    and `cont_lines` for the continuation (identified by the assistant-prefix message).
    Records every request body so the test can assert the continuation shape."""
    ad = OpenAIEngine(model_id="ornith", base_url="http://x/v1")
    ad.tok = _FakeTok()
    bodies = []

    def stream(body):
        bodies.append(body)
        is_continuation = any(m["role"] == "assistant" for m in body["messages"])
        return iter(cont_lines if is_continuation else first_lines)

    ad._stream_completion = stream
    return ad, bodies


def test_generate_close_and_continue_salvages_a_runaway_think():
    # First request: 3 think chunks, no </think> -> crosses ceiling=3 and the salvage
    # fires (the 4th chunk is never consumed). Continuation: the action, in ONE turn.
    first = list(_sse(
        {"choices": [{"delta": {"content": "reasoning "}}]},
        {"choices": [{"delta": {"content": "and more "}}]},
        {"choices": [{"delta": {"content": "and more "}}]},
        {"choices": [{"delta": {"content": "NEVER-REACHED"}}]},
        "[DONE]",
    ))
    cont = list(_sse(
        {"choices": [{"delta": {"content": '<tool_call>{"name":"read"}</tool_call>'}}]},
        {"choices": [], "usage": {"prompt_tokens": 120, "completion_tokens": 5}},
        "[DONE]",
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
    # The continuation resumes the assistant turn from the force-closed text, and that
    # text is a strict PREFIX of the final turn (the KV-prefix / prefix-cache invariant).
    prefix = bodies[1]["messages"][-1]
    assert prefix["role"] == "assistant"
    assert prefix["content"] == "reasoning and more and more \n</think>\n\n"
    assert text.startswith(prefix["content"])
    # The runaway request's 3 tokens must NOT be dropped just because its stream was
    # closed before the server could send usage for it: 3 (no-usage fallback, request 1)
    # + 5 (usage.completion_tokens, the continuation) = 8. A prior bug silently discarded
    # request 1's count whenever ANY request in the step reported usage.
    assert stats.generated_tokens == 8


def test_generate_no_salvage_when_think_closes_before_the_ceiling():
    # The model closes </think> and acts on its own, under the ceiling -> no salvage,
    # a single request, byte-identical to a plain generate.
    lines = list(_sse(
        {"choices": [{"delta": {"content": "brief think</think>\nact"}}]},
        {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        "[DONE]",
    ))
    ad, bodies = _salvage_adapter(lines, lines)
    text, stats = ad.generate([1], max_tokens=200, think_ceiling=6000)
    assert stats.salvaged is False
    assert len(bodies) == 1
    assert text == "brief think</think>\nact"


def test_generate_no_salvage_when_ceiling_unset():
    # think_ceiling=None (the default): a long open think is NOT force-closed.
    lines = list(_sse(
        {"choices": [{"delta": {"content": "a "}}]},
        {"choices": [{"delta": {"content": "b "}}]},
        {"choices": [{"delta": {"content": "c "}}]},
        {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        "[DONE]",
    ))
    ad, bodies = _salvage_adapter(lines, lines)
    text, stats = ad.generate([1], max_tokens=200, think_ceiling=None)
    assert stats.salvaged is False
    assert len(bodies) == 1
    assert text == "a b c "
