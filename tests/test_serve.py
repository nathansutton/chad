"""Tier-1 tests for chad-serve's pure halves (serve.py): the request normalizer, the
stream splitter and the tool-call parser. No model load, no socket.

These are the format-fragile surfaces of the server. The splitter has to give the same
answer however the completion is chunked, because the engine streams drafted blocks of
uneven width; the normalizer has to hand the template back the reasoning a client
dropped, or every follow-up turn re-prefills the whole transcript.
"""
import pytest

from chad import serve

TOOLS = [{"type": "function", "function": {"name": "read", "parameters": {"properties": {
    "path": {"type": "string"}, "limit": {"type": "integer"}, "deep": {"type": "boolean"}}}}}]

CALL_TURN = ("Need to read the file.\n</think>\n\nLet me look.\n\n<tool_call>\n<function=read>\n"
             "<parameter=path>\n  a < b.py\n</parameter>\n<parameter=limit>\n20\n</parameter>\n"
             "<parameter=deep>\ntrue\n</parameter>\n</function>\n</tool_call>")


def _split(text: str, step: int) -> serve.Splitter:
    split = serve.Splitter()
    for i in range(0, len(text), step):
        split.feed(text[i:i + step])
    split.finish()
    return split


@pytest.mark.parametrize("step", [1, 2, 3, 7, 500])
def test_split_is_chunking_independent(step):
    split = _split(CALL_TURN, step)
    assert "".join(split.reasoning).strip() == "Need to read the file."
    assert "".join(split.content) == "Let me look."
    assert serve.parse_tool_calls(split.tool, TOOLS) == [
        ("read", {"path": "  a < b.py", "limit": 20, "deep": True})]


@pytest.mark.parametrize("step", [1, 2, 5])
def test_angle_brackets_in_prose_are_not_swallowed(step):
    """`<` may open a tool-call marker, so it is held back; prose that merely contains
    one must still come out whole, inner blank lines included."""
    split = _split("thinking\n</think>\n\nif a < b and x <t then\n\n  fine.\n", step)
    assert "".join(split.content) == "if a < b and x <t then\n\n  fine."
    assert not split.tool


def test_unclosed_think_is_all_reasoning():
    split = _split("never closes", 4)
    assert "".join(split.reasoning) == "never closes"
    assert not split.content and not split.tool


def test_json_tool_call_dialect():
    split = _split('r</think>\n\n<tool_call>\n{"name":"read","arguments":{"path":"x"}}\n'
                   "</tool_call>", 5)
    assert not split.content
    assert serve.parse_tool_calls(split.tool, TOOLS) == [("read", {"path": "x"})]


def test_string_parameter_keeps_json_looking_text():
    """A string parameter whose value happens to parse as JSON stays a string."""
    tail = "<function=read>\n<parameter=path>\n123\n</parameter>\n</function>"
    assert serve.parse_tool_calls(tail, TOOLS) == [("read", {"path": "123"})]


def test_normalize_maps_openai_messages_onto_the_template():
    messages = serve.normalize_messages([
        {"role": "developer", "content": "sys"},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "1", "type": "function",
             "function": {"name": "read", "arguments": '{"path":"x"}'}}]},
        {"role": "tool", "tool_call_id": "1", "content": "data"},
    ], {})
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "tool"]
    assert messages[1]["content"] == "hi"
    assert messages[2]["tool_calls"][0]["function"]["arguments"] == {"path": "x"}


def test_normalize_restores_dropped_reasoning():
    """The reasoning the server generated for a turn comes back from the memo when the
    client's copy of that turn has none, so the re-render matches the cached tokens."""
    calls = [{"function": {"name": "read", "arguments": {"path": "x"}}}]
    memo = {serve.memo_key("Looking.", calls): "why I am reading it"}
    turn = {"role": "assistant", "content": "Looking.\n", "tool_calls": [
        {"id": "1", "type": "function",
         "function": {"name": "read", "arguments": '{"path": "x"}'}}]}
    [message] = serve.normalize_messages([turn], memo)
    assert message["reasoning_content"] == "why I am reading it"
    kept = dict(turn, reasoning_content="the client's own")
    [message] = serve.normalize_messages([kept], memo)
    assert message["reasoning_content"] == "the client's own"
