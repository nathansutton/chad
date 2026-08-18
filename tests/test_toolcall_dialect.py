"""One tool-call dialect across every surface the model reads.

The chat template is the authority: it renders the tool list, states the call format,
and closes with an <IMPORTANT> block saying a call MUST be a `<function=…>` nested in
`<tool_call>` — and it renders all of that IMMEDIATELY ABOVE chad's system prompt.
chad used to answer that, in the very next line and under a "CRITICAL — read first"
heading, with "the ONLY way to execute anything is to emit a tool call as a JSON object
inside <tool_call></tool_call> tags", plus a worked JSON example. Two mutually exclusive
"ONLY" contracts, back to back, on the one thing the model cannot afford to get wrong.

The cost was not hypothetical. `toolcall_parse._HYBRID_NAME_RE` exists to salvage
`<tool_call>{"name": "bash" <parameter=command>…</parameter></function>` — which is not
a random garble but a splice of the two formats, one clause from each. Worse, every
RECOVERY message doubled down: the exemplar shown on the second consecutive garble
spelled out the JSON form and explicitly forbade `<function=…>`/`<parameter=…>`, so the
message whose only job is to break a wrong-dialect streak taught the wrong dialect to a
model that was already unsure.

So: no chad-authored text that the model reads may teach the JSON call form. The parser
stays permissive and still accepts it — accepting more than we teach is the right
asymmetry — but nothing chad writes may contradict the template.
"""

import re

import pytest

from chad import guardrails, prompt, validate

# A JSON-dialect tool call: a `{"name": …}` object as the body of a call. Matching the
# SHAPE rather than the words keeps this from fighting prose that merely mentions JSON
# (tool ARGUMENTS legitimately contain JSON, and `write_todos` takes a JSON array).
_JSON_CALL = re.compile(r'\{\s*["\']name["\']\s*:\s*["\'][a-z_]+["\']')


def _model_facing_prompts():
    """Every system prompt chad renders, by name."""
    return {
        "main": prompt.build_system_prompt(),
        "subagent": prompt.build_subagent_prompt(),
    }


@pytest.mark.parametrize("lean", [False, True])
def test_no_prompt_teaches_the_json_call_form(lean, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    if lean:
        monkeypatch.setenv("CHAD_LEAN", "1")
    else:
        monkeypatch.delenv("CHAD_LEAN", raising=False)
    for name, text in _model_facing_prompts().items():
        assert not _JSON_CALL.search(text), (
            f"the {name} prompt (lean={lean}) shows a JSON-dialect tool call, which the "
            f"chat template rendered directly above it forbids")


@pytest.mark.parametrize("lean", [False, True])
def test_prompts_point_at_the_template_dialect(lean, monkeypatch, tmp_path):
    """Not naming the wrong dialect is half of it; the model must still be told which
    format is real, because chad's own contribution here — that a fenced code block
    executes nothing — only makes sense next to what a real call looks like."""
    monkeypatch.chdir(tmp_path)
    if lean:
        monkeypatch.setenv("CHAD_LEAN", "1")
    else:
        monkeypatch.delenv("CHAD_LEAN", raising=False)
    text = prompt.build_system_prompt()
    assert "<function=" in text and "<parameter=" in text


def test_the_garble_exemplar_is_in_the_template_dialect():
    """The exemplar fires on the second consecutive garble — the moment the model is
    demonstrably confused about the format. It must show the format the template
    mandates, and must not forbid it."""
    ex = guardrails.TOOLCALL_EXEMPLAR
    assert "<function=bash>" in ex and "<parameter=command>" in ex
    assert not _JSON_CALL.search(ex)
    assert "No <function=" not in ex


def test_repair_messages_never_teach_the_json_form():
    """The repair message sits directly above the model's retry, so whatever call shape
    it contains is a few-shot example whether or not it was meant as one."""
    garbled = "grep</argstr"
    msg = validate.render_repair(garbled, {}, validate.coerce_and_validate(garbled, {})[1])
    assert "<function=grep>" in msg and not _JSON_CALL.search(msg)

    # The invalid-arguments path echoes the model's OWN call back to it.
    args = {"path": "a.py", "old": "x"}          # `new` missing -> repair message
    coerced, errs = validate.coerce_and_validate("edit", args)
    echoed = validate.render_repair("edit", coerced, errs)
    assert errs and "<function=edit>" in echoed and "<parameter=path>" in echoed
    assert not _JSON_CALL.search(echoed)


def test_the_echo_round_trips_through_the_parser():
    """The echoed call is what we are asking the model to copy, so it has to be a call
    chad can actually parse — otherwise the repair path hands back a shape that fails
    the same way the original did."""
    from chad import toolcall_parse
    echo = validate._echo_call("bash", {"command": "ls /app\nwc -l x.py"})
    calls = toolcall_parse.parse_tool_calls(echo)
    assert calls == [("bash", {"command": "ls /app\nwc -l x.py"})]


def test_nudges_do_not_demand_json_arguments():
    """The malformed/garbled nudges are the other recovery surface."""
    base = dict(made_edit=False, unverified_edit=False, read_only_intent=False,
                action_task=True, truncation_nudges=0, answer_nudges=0, verify_nudges=0)
    for kwargs in ({"open_tool_call": True}, {"open_tool_call": False, "garbled_call": True}):
        _, nudge = guardrails.nudge_for_no_calls(
            "<tool_call>{...", hit_cap=False, **{**base, **kwargs,
                                                 "open_tool_call": kwargs.get("open_tool_call", False)})
        assert nudge and "JSON object" not in nudge, nudge
